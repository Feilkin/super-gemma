//! Parity: gemv_q6_k_logits (the tied Q6_K LM head with fused tanh-30
//! softcap, plan 02 step 7) vs the f64 CPU reference built on
//! `sg_gguf::q6_k::BlockQ6K::dequantize`. Skips without a GPU.

mod reference;

use reference::{Rng, assert_close, through_f16, to_f16_bits};
use sg_gguf::q6_k::{BLOCK_Q6_K_SIZE, QK6_K, blocks_from_bytes};
use sg_gpu::GpuContext;
use vulkano::buffer::BufferUsage;
use vulkano::descriptor_set::WriteDescriptorSet;

const K: usize = 5376;
const BLOCKS_PER_ROW: usize = K / QK6_K; // 21
const ROW_BYTES: usize = BLOCKS_PER_ROW * BLOCK_Q6_K_SIZE; // 4410
/// Rows are padded to a word-aligned stride (matches the kernel define).
const ROW_WORDS: usize = 1104; // 4416 bytes
const SOFTCAP: f64 = 30.0;

fn ctx() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("skipping: no usable GPU ({e})");
            None
        }
    }
}

/// Random Q6_K rows in the padded GPU layout. d and the sub-scales are kept
/// small so logits land in the softcap's responsive range rather than
/// saturating tanh.
fn random_q6k_rows(rng: &mut Rng, n: usize) -> Vec<u32> {
    let mut words = vec![0u32; n * ROW_WORDS];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut words);
    for row in 0..n {
        let row_bytes = &mut bytes[row * ROW_WORDS * 4..][..ROW_BYTES];
        for b in row_bytes.iter_mut() {
            *b = (rng.next_u64() & 0xFF) as u8;
        }
        for blk in 0..BLOCKS_PER_ROW {
            let block = &mut row_bytes[blk * BLOCK_Q6_K_SIZE..][..BLOCK_Q6_K_SIZE];
            // Sub-scales in [-8, 7] instead of full i8 range.
            for s in block[192..208].iter_mut() {
                *s = ((*s & 0x0F) as i8 - 8) as u8;
            }
            let d = half::f16::from_f32(rng.f32() * 1e-3);
            block[208..210].copy_from_slice(&d.to_bits().to_le_bytes());
        }
    }
    words
}

#[test]
fn gemv_q6_k_logits_matches_reference() {
    let Some(ctx) = ctx() else { return };
    let kernel = ctx.load_kernel("gemv_q6_k_logits").unwrap();
    let mut rng = Rng::new(0xC6);
    let n = 96usize; // output rows (vocab slice)

    let weights = random_q6k_rows(&mut rng, n);
    let x = through_f16(&rng.f32_vec(K));

    let w_buf = ctx
        .buffer_from_iter(weights.iter().copied(), BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let x_buf = ctx
        .buffer_from_iter(to_f16_bits(&x), BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let y_buf = ctx
        .new_buffer::<f32>(n as u64, BufferUsage::STORAGE_BUFFER)
        .unwrap();

    ctx.dispatch_blocking(
        &kernel,
        vec![
            WriteDescriptorSet::buffer(0, w_buf),
            WriteDescriptorSet::buffer(1, x_buf),
            WriteDescriptorSet::buffer(2, y_buf.clone()),
        ],
        None::<u32>,
        [n as u32, 1, 1],
    )
    .unwrap();
    let got: Vec<f32> = y_buf.read().unwrap().to_vec();

    let bytes: &[u8] = bytemuck::cast_slice(&weights);
    let mut want = Vec::with_capacity(n);
    for row in 0..n {
        let row_bytes = &bytes[row * ROW_WORDS * 4..][..ROW_BYTES];
        let blocks = blocks_from_bytes(row_bytes).unwrap();
        let mut dot = 0.0f64;
        let mut w_row = [0.0f32; QK6_K];
        for (blk_idx, blk) in blocks.iter().enumerate() {
            blk.dequantize(&mut w_row);
            for (j, &w) in w_row.iter().enumerate() {
                dot += w as f64 * x[blk_idx * QK6_K + j] as f64;
            }
        }
        want.push((SOFTCAP * (dot / SOFTCAP).tanh()) as f32);
    }

    assert_close(&got, &want, 2e-2, 2e-2, "gemv_q6_k_logits");
}

/// Plan 02/06: bit-identical across runs.
#[test]
fn logits_are_bit_deterministic() {
    let Some(ctx) = ctx() else { return };
    let kernel = ctx.load_kernel("gemv_q6_k_logits").unwrap();
    let mut rng = Rng::new(0xC7);
    let n = 64usize;
    let weights = random_q6k_rows(&mut rng, n);
    let x = to_f16_bits(&rng.f32_vec(K));

    let mut outs = Vec::new();
    for _ in 0..2 {
        let w_buf = ctx
            .buffer_from_iter(weights.iter().copied(), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let x_buf = ctx
            .buffer_from_iter(x.iter().copied(), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let y_buf = ctx
            .new_buffer::<f32>(n as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        ctx.dispatch_blocking(
            &kernel,
            vec![
                WriteDescriptorSet::buffer(0, w_buf),
                WriteDescriptorSet::buffer(1, x_buf),
                WriteDescriptorSet::buffer(2, y_buf.clone()),
            ],
            None::<u32>,
            [n as u32, 1, 1],
        )
        .unwrap();
        outs.push(
            y_buf
                .read()
                .unwrap()
                .iter()
                .map(|f| f.to_bits())
                .collect::<Vec<u32>>(),
        );
    }
    assert_eq!(outs[0], outs[1]);
}
