//! Parity: rmsnorm / rope / geglu kernels vs the f64 CPU references, on
//! randomized inputs over the real shapes. Skips without a GPU (Tier 1).

mod reference;

use reference::{Rng, assert_close, from_f16_bits, through_f16, to_f16_bits};
use sg_gpu::GpuContext;
use sg_gpu::{BufferBinding, BufferUsage};

fn ctx() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("skipping: no usable GPU ({e})");
            None
        }
    }
}

/// Plan 02 tolerance for norms/rope/activations: ≤ 1e-3 relative; the atol
/// floor covers f16 output quantization around zero.
const RTOL: f32 = 1e-3;
const ATOL: f32 = 2e-3;

#[test]
fn rmsnorm_matches_reference_for_all_row_lengths() {
    let Some(ctx) = ctx() else { return };
    let mut rng = Rng::new(0xA11CE);

    for (variant, row_len, plus_one) in [
        ("rmsnorm_5376", 5376usize, false),
        ("rmsnorm_5376_plus1", 5376, true),
        ("rmsnorm_512", 512, false),
        ("rmsnorm_512_plus1", 512, true),
        ("rmsnorm_256", 256, false),
        ("rmsnorm_256_plus1", 256, true),
    ] {
        let kernel = ctx.load_kernel(variant).expect(variant);
        let rows = 7usize;
        let x = through_f16(&rng.f32_vec(rows * row_len));
        let w = rng.f32_vec(row_len);

        let x_buf = ctx
            .buffer_from_iter(to_f16_bits(&x), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let w_buf = ctx
            .buffer_from_iter(w.iter().copied(), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let y_buf = ctx
            .new_buffer::<u16>((rows * row_len) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();

        ctx.dispatch_blocking(
            &kernel,
            vec![
                BufferBinding::buffer(0, x_buf),
                BufferBinding::buffer(1, w_buf.clone()),
                BufferBinding::buffer(2, y_buf.clone()),
            ],
            None::<u32>,
            [rows as u32, 1, 1], // one workgroup per row
        )
        .expect(variant);

        let got = from_f16_bits(&y_buf.read().unwrap());
        let want = reference::rmsnorm(&x, &w, row_len, 1e-6, plus_one);
        assert_close(&got, &want, ATOL, RTOL, variant);
    }
}

#[test]
fn rope_matches_reference_for_all_sites() {
    let Some(ctx) = ctx() else { return };
    let mut rng = Rng::new(0x40FE);

    for (variant, head_dim, rot_dims, n_heads, theta) in [
        ("rope_sliding_q", 256usize, 256usize, 32usize, 10_000.0f64),
        ("rope_sliding_k", 256, 256, 16, 10_000.0),
        ("rope_global_q", 512, 128, 32, 1_000_000.0),
        ("rope_global_k", 512, 128, 4, 1_000_000.0),
    ] {
        let kernel = ctx.load_kernel(variant).expect(variant);
        let inv_freq = reference::inv_freqs(rot_dims, theta);
        // Positions deep into the context exercise large-angle precision
        // (handled by the f64 CPU table, not GPU trig).
        for start_pos in [0u32, 1, 1023, 100_000] {
            let tokens = 3usize;
            let n = tokens * n_heads * head_dim;
            let x = through_f16(&rng.f32_vec(n));
            let cos_sin = reference::cos_sin_table(&inv_freq, start_pos, tokens);

            let buf = ctx
                .buffer_from_iter(to_f16_bits(&x), BufferUsage::STORAGE_BUFFER)
                .unwrap();
            let cs_buf = ctx
                .buffer_from_iter(cos_sin.iter().copied(), BufferUsage::STORAGE_BUFFER)
                .unwrap();
            let pairs = (n / head_dim) * (rot_dims / 2);
            ctx.dispatch_blocking(
                &kernel,
                vec![
                    BufferBinding::buffer(0, buf.clone()),
                    BufferBinding::buffer(1, cs_buf),
                ],
                None::<u32>,
                kernel.groups_for(pairs as u64),
            )
            .expect(variant);

            let got = from_f16_bits(&buf.read().unwrap());
            let mut want = x.clone();
            reference::rope(&mut want, head_dim, rot_dims, n_heads, &cos_sin);
            assert_close(&got, &want, ATOL, RTOL, &format!("{variant}@{start_pos}"));

            // The frozen pairs must pass through untouched: with NEOX
            // pairing (i, i+head_dim/2), that's dims [rot/2, head_dim/2)
            // and [head_dim/2 + rot/2, head_dim).
            if rot_dims < head_dim {
                let (live, half) = (rot_dims / 2, head_dim / 2);
                for row in 0..tokens * n_heads {
                    let base = row * head_dim;
                    assert_eq!(
                        got[base + live..base + half],
                        x[base + live..base + half],
                        "{variant}: frozen low-half dims modified in row {row}"
                    );
                    assert_eq!(
                        got[base + half + live..base + head_dim],
                        x[base + half + live..base + head_dim],
                        "{variant}: frozen high-half dims modified in row {row}"
                    );
                }
            }
        }
    }
}

#[test]
fn geglu_matches_reference() {
    let Some(ctx) = ctx() else { return };
    let mut rng = Rng::new(0x6E61);
    let kernel = ctx.load_kernel("geglu").unwrap();

    let n = 21504 * 3 + 17; // real width × a few tokens, plus a ragged tail
    let gate = through_f16(&rng.f32_vec(n));
    let up = through_f16(&rng.f32_vec(n));

    let g_buf = ctx
        .buffer_from_iter(to_f16_bits(&gate), BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let u_buf = ctx
        .buffer_from_iter(to_f16_bits(&up), BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let y_buf = ctx
        .new_buffer::<u16>(n as u64, BufferUsage::STORAGE_BUFFER)
        .unwrap();

    ctx.dispatch_blocking(
        &kernel,
        vec![
            BufferBinding::buffer(0, g_buf),
            BufferBinding::buffer(1, u_buf),
            BufferBinding::buffer(2, y_buf.clone()),
        ],
        None::<u32>,
        kernel.groups_for(n as u64),
    )
    .unwrap();

    let got = from_f16_bits(&y_buf.read().unwrap());
    let want = reference::geglu(&gate, &up);
    assert_close(&got, &want, ATOL, RTOL, "geglu");
}

#[test]
fn add_scaled_matches_reference() {
    let Some(ctx) = ctx() else { return };
    let mut rng = Rng::new(0xADD5);
    let kernel = ctx.load_kernel("add_scaled").unwrap();

    let n = 5376 * 3 + 5; // hidden rows × a few tokens, plus a ragged tail
    let a = through_f16(&rng.f32_vec(n));
    let b = through_f16(&rng.f32_vec(n));

    for scale in [1.0f32, 0.83] {
        let a_buf = ctx
            .buffer_from_iter(to_f16_bits(&a), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let b_buf = ctx
            .buffer_from_iter(to_f16_bits(&b), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let y_buf = ctx
            .new_buffer::<u16>(n as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();

        ctx.dispatch_blocking(
            &kernel,
            vec![
                BufferBinding::buffer(0, a_buf),
                BufferBinding::buffer(1, b_buf),
                BufferBinding::buffer(2, y_buf.clone()),
            ],
            Some(scale),
            kernel.groups_for(n as u64),
        )
        .unwrap();

        let got = from_f16_bits(&y_buf.read().unwrap());
        let want: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| (x + y) * scale).collect();
        assert_close(&got, &want, ATOL, RTOL, &format!("add_scaled s={scale}"));
    }
}

/// Plan 02 determinism requirement: same inputs → bit-identical outputs.
#[test]
fn rmsnorm_is_bit_deterministic() {
    let Some(ctx) = ctx() else { return };
    let mut rng = Rng::new(0xDE7E);
    let kernel = ctx.load_kernel("rmsnorm_5376").unwrap();
    let rows = 4usize;
    let x = to_f16_bits(&rng.f32_vec(rows * 5376));
    let w = rng.f32_vec(5376);

    let mut outputs = Vec::new();
    for _ in 0..2 {
        let x_buf = ctx
            .buffer_from_iter(x.iter().copied(), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let w_buf = ctx
            .buffer_from_iter(w.iter().copied(), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let y_buf = ctx
            .new_buffer::<u16>((rows * 5376) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        ctx.dispatch_blocking(
            &kernel,
            vec![
                BufferBinding::buffer(0, x_buf),
                BufferBinding::buffer(1, w_buf),
                BufferBinding::buffer(2, y_buf.clone()),
            ],
            None::<u32>,
            [rows as u32, 1, 1],
        )
        .unwrap();
        outputs.push(y_buf.read().unwrap().to_vec());
    }
    assert_eq!(outputs[0], outputs[1], "outputs differ between runs");
}
