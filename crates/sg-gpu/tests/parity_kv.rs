//! Parity: KV plumbing kernels (plan 02 step 7) — append into ring/linear
//! stores (incl. ring wraparound) and the f16↔Q8_0 codecs in their
//! structure-of-arrays page format, whose scales must match the CPU
//! reference BIT-EXACTLY (cache2 pages round-trip through them; plan 04
//! resume bit-exactness). Skips without a GPU.

mod reference;

use reference::{Rng, dequant_q8_0, from_f16_bits, quant_q8_0, through_f16, to_f16_bits};
use sg_gpu::{GpuContext, StepState};
use sg_gpu::{Buffer, BufferBinding, BufferUsage};

fn ctx() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("skipping: no usable GPU ({e})");
            None
        }
    }
}

/// Step buffer carrying the append position.
fn step_buf(ctx: &GpuContext, pos: u32) -> Buffer<u32> {
    let buf = ctx.new_step_buffer().unwrap();
    StepState {
        pos,
        ..Default::default()
    }
    .write_to(&buf)
    .unwrap();
    buf
}

#[test]
fn kv_append_sliding_wraps_the_ring() {
    let Some(ctx) = ctx() else { return };
    let kernel = ctx.load_kernel("kv_append_sliding").unwrap();
    const ROW: usize = 4096; // 16 heads × 256
    const RING: usize = 1024;
    let mut rng = Rng::new(0xB0);

    // Appending 5 tokens at pos 1022 must land in slots 1022, 1023, 0, 1, 2.
    let n_tokens = 5usize;
    let pos = 1022u32;
    let src = through_f16(&rng.f32_vec(n_tokens * ROW));
    let sentinel = half::f16::from_f32(-99.0).to_bits();

    let src_buf = ctx
        .buffer_from_iter(to_f16_bits(&src), BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let ring_buf = ctx
        .buffer_from_iter(
            (0..RING * ROW).map(|_| sentinel),
            BufferUsage::STORAGE_BUFFER,
        )
        .unwrap();

    ctx.dispatch_blocking(
        &kernel,
        vec![
            BufferBinding::buffer(0, src_buf),
            BufferBinding::buffer(1, ring_buf.clone()),
            BufferBinding::buffer(2, step_buf(&ctx, pos)),
        ],
        None::<u32>,
        kernel.groups_for((n_tokens * ROW) as u64),
    )
    .unwrap();

    let ring = from_f16_bits(&ring_buf.read().unwrap());
    for (t, &slot) in [1022usize, 1023, 0, 1, 2].iter().enumerate() {
        assert_eq!(
            ring[slot * ROW..][..ROW],
            src[t * ROW..][..ROW],
            "token {t} → slot {slot}"
        );
    }
    // Everything else untouched.
    let untouched = (3..1022).all(|s| ring[s * ROW..][..ROW].iter().all(|&v| v == -99.0));
    assert!(untouched, "sentinel slots were overwritten");
}

#[test]
fn kv_append_global_is_linear() {
    let Some(ctx) = ctx() else { return };
    let kernel = ctx.load_kernel("kv_append_global").unwrap();
    const ROW: usize = 2048; // 4 heads × 512
    let mut rng = Rng::new(0xB1);

    let n_tokens = 3usize;
    let pos = 7u32;
    let slots = 16usize;
    let src = through_f16(&rng.f32_vec(n_tokens * ROW));
    let src_buf = ctx
        .buffer_from_iter(to_f16_bits(&src), BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let dst_buf = ctx
        .buffer_from_iter((0..slots * ROW).map(|_| 0u16), BufferUsage::STORAGE_BUFFER)
        .unwrap();

    ctx.dispatch_blocking(
        &kernel,
        vec![
            BufferBinding::buffer(0, src_buf),
            BufferBinding::buffer(1, dst_buf.clone()),
            BufferBinding::buffer(2, step_buf(&ctx, pos)),
        ],
        None::<u32>,
        kernel.groups_for((n_tokens * ROW) as u64),
    )
    .unwrap();

    let dst = from_f16_bits(&dst_buf.read().unwrap());
    for t in 0..n_tokens {
        assert_eq!(
            dst[(pos as usize + t) * ROW..][..ROW],
            src[t * ROW..][..ROW],
            "token {t}"
        );
    }
    assert!(dst[..pos as usize * ROW].iter().all(|&v| v == 0.0));
    assert!(
        dst[(pos as usize + n_tokens) * ROW..]
            .iter()
            .all(|&v| v == 0.0)
    );
}

#[test]
fn kv_quant_q8_matches_reference_bit_exactly() {
    let Some(ctx) = ctx() else { return };
    let quant_k = ctx.load_kernel("kv_quant_q8").unwrap();
    let dequant_k = ctx.load_kernel("kv_dequant_q8").unwrap();
    let mut rng = Rng::new(0xB2);

    // A sliding row's worth plus change; includes an all-zero block
    // (d = 0 → id = 0 path).
    let n = 4096 + 256;
    let mut x = through_f16(&rng.f32_vec(n));
    for v in x[64..96].iter_mut() {
        *v = 0.0;
    }

    let (want_scales, want_quants) = quant_q8_0(&x);

    let src_buf = ctx
        .buffer_from_iter(to_f16_bits(&x), BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let scale_buf = ctx
        .new_buffer::<u16>((n / 32) as u64, BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let quant_buf = ctx
        .new_buffer::<u32>((n / 4) as u64, BufferUsage::STORAGE_BUFFER)
        .unwrap();
    ctx.dispatch_blocking(
        &quant_k,
        vec![
            BufferBinding::buffer(0, src_buf),
            BufferBinding::buffer(1, scale_buf.clone()),
            BufferBinding::buffer(2, quant_buf.clone()),
        ],
        None::<u32>,
        quant_k.groups_for((n / 32) as u64),
    )
    .unwrap();

    let got_scales: Vec<u16> = scale_buf.read().unwrap().to_vec();
    let got_quants: Vec<u8> = quant_buf
        .read()
        .unwrap()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    // The f16 scales must match bit-exactly; the quants may differ by ±1
    // where the GPU's 2.5-ULP reciprocal lands an x·(1/d) product on the
    // other side of a rounding boundary (see the kernel header).
    assert_eq!(got_scales, want_scales, "f16 scales differ");
    assert_eq!(got_quants.len(), want_quants.len());
    let mut flipped = 0usize;
    for (j, (&g, &w)) in got_quants.iter().zip(&want_quants).enumerate() {
        let delta = (g as i8 as i32 - w as i8 as i32).abs();
        assert!(delta <= 1, "quant {j}: got {g} want {w}");
        flipped += (delta == 1) as usize;
    }
    assert!(
        flipped <= n / 32, // sanity: boundary flips are rare
        "{flipped} quants flipped — more than rounding-boundary noise"
    );

    // The GPU dequantizer must match the CPU dequantizer EXACTLY on the
    // same (GPU-produced) data: both sides are exact f32 multiplies.
    let want_deq = dequant_q8_0(&got_scales, &got_quants);
    let deq_buf = ctx
        .new_buffer::<u16>(n as u64, BufferUsage::STORAGE_BUFFER)
        .unwrap();
    ctx.dispatch_blocking(
        &dequant_k,
        vec![
            BufferBinding::buffer(0, scale_buf),
            BufferBinding::buffer(1, quant_buf),
            BufferBinding::buffer(2, deq_buf.clone()),
        ],
        None::<u32>,
        dequant_k.groups_for((n / 32) as u64),
    )
    .unwrap();
    let got_deq = from_f16_bits(&deq_buf.read().unwrap());
    assert_eq!(got_deq, want_deq, "dequantized values differ");
}

/// Plan 02/06: the quantizer (incl. its 2.5-ULP reciprocal) is
/// bit-deterministic across runs — what cache2 page identity rests on.
#[test]
fn kv_quant_q8_is_bit_deterministic() {
    let Some(ctx) = ctx() else { return };
    let kernel = ctx.load_kernel("kv_quant_q8").unwrap();
    let mut rng = Rng::new(0xB3);
    let n = 8192usize;
    let x = to_f16_bits(&rng.f32_vec(n));

    let mut outs = Vec::new();
    for _ in 0..2 {
        let src_buf = ctx
            .buffer_from_iter(x.iter().copied(), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let scale_buf = ctx
            .new_buffer::<u16>((n / 32) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let quant_buf = ctx
            .new_buffer::<u32>((n / 4) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        ctx.dispatch_blocking(
            &kernel,
            vec![
                BufferBinding::buffer(0, src_buf),
                BufferBinding::buffer(1, scale_buf.clone()),
                BufferBinding::buffer(2, quant_buf.clone()),
            ],
            None::<u32>,
            kernel.groups_for((n / 32) as u64),
        )
        .unwrap();
        let mut combined: Vec<u32> = scale_buf
            .read()
            .unwrap()
            .iter()
            .map(|&s| s as u32)
            .collect();
        combined.extend(quant_buf.read().unwrap().iter());
        outs.push(combined);
    }
    assert_eq!(outs[0], outs[1]);
}
