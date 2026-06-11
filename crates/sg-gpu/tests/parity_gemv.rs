//! Parity: gemv_q4_0 vs the f64 CPU reference built on sg-gguf's scalar
//! dequant (the kernel ground truth, plan 01/02). All eight real matmul
//! shapes plus the generic-K baseline. Skips without a GPU.

mod reference;

use reference::{Rng, assert_close, from_f16_bits, through_f16, to_f16_bits};
use sg_gguf::q4_0::{BLOCK_Q4_0_SIZE, QK4_0, blocks_from_bytes};
use sg_gpu::GpuContext;
use vulkano::buffer::BufferUsage;
use vulkano::descriptor_set::WriteDescriptorSet;

fn ctx() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("skipping: no usable GPU ({e})");
            None
        }
    }
}

/// Random Q4_0 weight bytes: nibbles uniform, scales f16 in ~[-0.1, 0.1)
/// (realistic Q4_0 scale magnitudes).
fn random_q4_0(rng: &mut Rng, n_blocks: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_blocks * BLOCK_Q4_0_SIZE);
    for _ in 0..n_blocks {
        let d = half::f16::from_f32(rng.f32() * 0.05);
        out.extend_from_slice(&d.to_bits().to_le_bytes());
        for _ in 0..16 {
            out.push((rng.next_u64() & 0xFF) as u8);
        }
    }
    out
}

/// f64 GEMV over Q4_0 rows for a subset of output rows.
fn reference_rows(weights: &[u8], x: &[f32], k: usize, rows: &[usize]) -> Vec<f32> {
    let blocks_per_row = k / QK4_0;
    let row_bytes = blocks_per_row * BLOCK_Q4_0_SIZE;
    rows.iter()
        .map(|&row| {
            let row_data = &weights[row * row_bytes..][..row_bytes];
            let blocks = blocks_from_bytes(row_data).expect("whole blocks");
            let mut acc = 0.0f64;
            let mut dequant = [0.0f32; QK4_0];
            for (b, block) in blocks.iter().enumerate() {
                block.dequantize(&mut dequant);
                for (j, &w) in dequant.iter().enumerate() {
                    acc += w as f64 * x[b * QK4_0 + j] as f64;
                }
            }
            acc as f32
        })
        .collect()
}

/// Sample of output rows to reference (full N×K f64 GEMV on the big shapes
/// would dominate test time without adding signal).
fn sample_rows(n: usize) -> Vec<usize> {
    let mut rows: Vec<usize> = (0..n.min(64)).collect();
    rows.extend((64..n).step_by((n / 97).max(1)));
    rows.push(n - 1);
    rows.sort_unstable();
    rows.dedup();
    rows
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct GenericPush {
    k: u32,
}

/// All eight (K, N) matmul sites (plan 02: the complete shape set).
const SHAPES: &[(usize, usize)] = &[
    (5376, 8192),  // sliding attn_q
    (5376, 4096),  // sliding attn_k / attn_v
    (8192, 5376),  // sliding attn_output
    (5376, 16384), // global attn_q
    (5376, 2048),  // global attn_k (K=V)
    (16384, 5376), // global attn_output
    (5376, 21504), // ffn_gate / ffn_up
    (21504, 5376), // ffn_down
];

#[test]
fn gemv_q4_0_matches_reference_on_all_shapes() {
    let Some(ctx) = ctx() else { return };
    let mut rng = Rng::new(0x6E44);

    for &(k, n) in SHAPES {
        let variant = format!("gemv_q4_0_k{k}");
        let kernel = ctx.load_kernel(&variant).expect(&variant);

        let weights = random_q4_0(&mut rng, n * k / QK4_0);
        let x = through_f16(&rng.f32_vec(k));

        let w_buf = ctx
            .buffer_from_iter(
                weights
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap())),
                BufferUsage::STORAGE_BUFFER,
            )
            .unwrap();
        let x_buf = ctx
            .buffer_from_iter(to_f16_bits(&x), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let y_buf = ctx
            .new_buffer::<u16>(n as u64, BufferUsage::STORAGE_BUFFER)
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
        .expect(&variant);

        let got_all = from_f16_bits(&y_buf.read().unwrap());
        let rows = sample_rows(n);
        let got: Vec<f32> = rows.iter().map(|&r| got_all[r]).collect();
        let want = reference_rows(&weights, &x, k, &rows);
        // Plan 02 matmul tolerance: ≤ 2e-2 relative vs f64 at f16 math; the
        // atol floor covers f16 output quantization of near-zero sums.
        assert_close(&got, &want, 2e-2, 2e-2, &format!("{variant} ({k}x{n})"));
    }
}

#[test]
fn gemv_generic_matches_specialized() {
    let Some(ctx) = ctx() else { return };
    let mut rng = Rng::new(0xAB);
    let (k, n) = (5376usize, 4096usize);

    let weights = random_q4_0(&mut rng, n * k / QK4_0);
    let x = to_f16_bits(&through_f16(&rng.f32_vec(k)));
    let mut outs = Vec::new();
    for (variant, push) in [
        ("gemv_q4_0_k5376", None),
        ("gemv_q4_0_generic", Some(GenericPush { k: k as u32 })),
    ] {
        let kernel = ctx.load_kernel(variant).expect(variant);
        let w_buf = ctx
            .buffer_from_iter(
                weights
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap())),
                BufferUsage::STORAGE_BUFFER,
            )
            .unwrap();
        let x_buf = ctx
            .buffer_from_iter(x.iter().copied(), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let y_buf = ctx
            .new_buffer::<u16>(n as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        ctx.dispatch_blocking(
            &kernel,
            vec![
                WriteDescriptorSet::buffer(0, w_buf),
                WriteDescriptorSet::buffer(1, x_buf),
                WriteDescriptorSet::buffer(2, y_buf.clone()),
            ],
            push,
            [n as u32, 1, 1],
        )
        .expect(variant);
        outs.push(y_buf.read().unwrap().to_vec());
    }
    // Identical math order → bit-identical results.
    assert_eq!(outs[0], outs[1], "generic vs specialized");
}

/// Plan 02/06: same inputs → bit-identical outputs across runs.
#[test]
fn gemv_is_bit_deterministic() {
    let Some(ctx) = ctx() else { return };
    let mut rng = Rng::new(0xD0);
    let (k, n) = (5376usize, 21504usize);
    let kernel = ctx.load_kernel("gemv_q4_0_k5376").unwrap();
    let weights = random_q4_0(&mut rng, n * k / QK4_0);
    let x = to_f16_bits(&rng.f32_vec(k));

    let mut outs = Vec::new();
    for _ in 0..2 {
        let w_buf = ctx
            .buffer_from_iter(
                weights
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap())),
                BufferUsage::STORAGE_BUFFER,
            )
            .unwrap();
        let x_buf = ctx
            .buffer_from_iter(x.iter().copied(), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let y_buf = ctx
            .new_buffer::<u16>(n as u64, BufferUsage::STORAGE_BUFFER)
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
        outs.push(y_buf.read().unwrap().to_vec());
    }
    assert_eq!(outs[0], outs[1]);
}
