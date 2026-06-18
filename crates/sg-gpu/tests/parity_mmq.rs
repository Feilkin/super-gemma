//! int8-MMQ GEMM parity (profile rank #2; docs/naga-int8-coopmat-patch.md).
//!
//! Two checks:
//!  - `mmq_reference_tracks_full_precision`: the `mmq_q4_0_q8` oracle vs the
//!    dequantized matmul. Q4_0 weights are exactly `d·(q−8)`, so the ONLY
//!    error is the Q8_0 activation quant → MMQ must track truth within the
//!    int8 envelope (≈ 1 %). Pins the oracle. No GPU.
//!  - `gemm_q4_0_i8_matches_mmq_reference`: the `gemm_q4_0_i8` coopmat kernel
//!    vs that oracle (same Q8_0 quant of the SAME activations → the only
//!    divergence is f32 accumulation order + the f16 output). Needs the GPU
//!    and the int8-coopmat naga fork.

mod reference;

use reference::{
    Rng, assert_close, from_f16_bits, mmq_q4_0_q8, quant_q8_0, through_f16, to_f16_bits,
};
use sg_gguf::q4_0::{BLOCK_Q4_0_SIZE, QK4_0, blocks_from_bytes};
use sg_gpu::GpuContext;
use vulkano::buffer::BufferUsage;
use vulkano::descriptor_set::WriteDescriptorSet;

/// Q4_0 weight bytes, `n_blocks` blocks (matches parity_gemm's generator:
/// small f16 scale, random nibbles).
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

/// Full-precision truth: W dequantized to f32 (`d·(q−8)`, exact), f32
/// activations, f64 accumulation. `Y = X · Wᵀ`.
fn true_matmul(weights: &[u8], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let row_bytes = k / QK4_0 * BLOCK_Q4_0_SIZE;
    let mut out = vec![0f32; m * n];
    let mut tmp = [0.0f32; QK4_0];
    for ni in 0..n {
        let blocks = blocks_from_bytes(&weights[ni * row_bytes..][..row_bytes]).unwrap();
        let mut wrow = vec![0.0f32; k];
        for (b, blk) in blocks.iter().enumerate() {
            blk.dequantize(&mut tmp);
            wrow[b * QK4_0..(b + 1) * QK4_0].copy_from_slice(&tmp);
        }
        for mi in 0..m {
            let acc: f64 = (0..k).map(|j| wrow[j] as f64 * x[mi * k + j] as f64).sum();
            out[mi * n + ni] = acc as f32;
        }
    }
    out
}

fn nrmse(got: &[f32], want: &[f32]) -> f32 {
    let mse: f64 = got
        .iter()
        .zip(want)
        .map(|(&g, &w)| ((g - w) as f64).powi(2))
        .sum::<f64>()
        / got.len() as f64;
    let denom: f64 = want.iter().map(|&w| (w as f64).powi(2)).sum::<f64>() / want.len() as f64;
    (mse.sqrt() / denom.sqrt().max(1e-12)) as f32
}

#[test]
fn mmq_reference_tracks_full_precision() {
    let mut rng = Rng::new(0x319);
    // Small shape — pure CPU, exercises multiple 32-blocks per row.
    let (m, k, n) = (16usize, 512usize, 64usize);

    let weights = random_q4_0(&mut rng, n * k / QK4_0);
    // f16-representable activations (what the kernel sees from the prior layer).
    let x = through_f16(&rng.f32_vec(m * k));

    let mmq = mmq_q4_0_q8(&weights, &x, m, k, n);
    let truth = true_matmul(&weights, &x, m, k, n);

    let err = nrmse(&mmq, &truth);
    let max_abs = mmq
        .iter()
        .zip(&truth)
        .map(|(&g, &w)| (g - w).abs())
        .fold(0.0f32, f32::max);
    eprintln!("MMQ vs full-precision: nrmse {err:.5}, max abs {max_abs:.5}");

    // int8 activation quant only — should track truth within ~1 %.
    assert!(err < 0.02, "MMQ nrmse {err} exceeds the int8 envelope");
}

fn ctx() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping: no usable GPU ({e})");
            None
        }
    }
}

/// The FFN PRODUCTION shapes (k5376_n21504 up, k21504_n5376 down) at 2×2, fed
/// activations quantized by the GPU `kv_quant_q8` kernel (the prefill path),
/// not the CPU `quant_q8_0`. Localizes the int8-FFN graph regression: the other
/// parity case only exercises k512 logic with CPU-quantized activations, so a
/// production-K or GPU-quant-chain bug hides there. Tolerance is looser — GPU
/// quant may differ ±1 from the CPU oracle's quant at rounding boundaries.
#[test]
fn gemm_q4_0_i8_ffn_shapes_with_gpu_quant() {
    let Some(ctx) = ctx() else { return };
    if !ctx.cooperative_matrix {
        eprintln!("skipping: no VK_KHR_cooperative_matrix");
        return;
    }
    let quant = ctx.load_kernel("kv_quant_q8").expect("kv_quant_q8");
    let mut rng = Rng::new(0x5E);
    // (variant, m, k, n) at 2×2 (m_rows = n_cols = 32).
    let cases = [
        (
            "gemm_q4_0_i8_t22_k5376_n21504",
            64usize,
            5376usize,
            21504usize,
        ),
        ("gemm_q4_0_i8_t22_k21504_n5376", 64, 21504, 5376),
    ];
    for (variant, m, k, n) in cases {
        let kernel = ctx.load_kernel(variant).expect(variant);
        let nb = k / QK4_0;

        let weights = random_q4_0(&mut rng, n * k / QK4_0);
        let x = through_f16(&rng.f32_vec(m * k));
        let want = mmq_q4_0_q8(&weights, &x, m, k, n);

        // GPU-quantize the activations with kv_quant_q8 → x_i8 (u32-packed) +
        // x_scales (f16), exactly the prefill path.
        let x_f16 = ctx
            .buffer_from_iter(to_f16_bits(&x), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let x_scales = ctx
            .new_buffer::<u16>((m * nb) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let x_i8 = ctx
            .new_buffer::<u32>((m * k / 4) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        ctx.dispatch_blocking(
            &quant,
            vec![
                WriteDescriptorSet::buffer(0, x_f16),
                WriteDescriptorSet::buffer(1, x_scales.clone()),
                WriteDescriptorSet::buffer(2, x_i8.clone()),
            ],
            None::<u32>,
            quant.groups_for((m * nb) as u64),
        )
        .unwrap();

        let w_buf = ctx
            .buffer_from_iter(
                weights
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap())),
                BufferUsage::STORAGE_BUFFER,
            )
            .unwrap();
        let y_buf = ctx
            .new_buffer::<u16>((m * n) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        ctx.dispatch_blocking(
            &kernel,
            vec![
                WriteDescriptorSet::buffer(0, w_buf),
                WriteDescriptorSet::buffer(1, x_i8.clone()),
                WriteDescriptorSet::buffer(2, x_scales.clone()),
                WriteDescriptorSet::buffer(3, y_buf.clone()),
            ],
            None::<u32>,
            [(n / 32) as u32, (m / 32) as u32, 1],
        )
        .unwrap();

        let got = from_f16_bits(&y_buf.read().unwrap());
        let err = nrmse(&got, &want);
        eprintln!("{variant}: GPU-quant int8 gemm vs oracle nrmse {err:.6}");
        assert_close(&got, &want, 3e-2, 3e-2, variant);
    }
}

#[test]
fn gemm_q4_0_i8_matches_mmq_reference() {
    let Some(ctx) = ctx() else { return };
    if !ctx.cooperative_matrix {
        eprintln!("skipping: no VK_KHR_cooperative_matrix");
        return;
    }
    let mut rng = Rng::new(0x4D);
    // (variant, m, k, n, M_TILES·16, N_TILES·16) — 1×1 and 2×4 tilings, so
    // both the single-tile path and the tiled rescale/index logic are covered.
    let cases = [
        (
            "gemm_q4_0_i8_k512_n64",
            16usize,
            512usize,
            64usize,
            16usize,
            16usize,
        ),
        ("gemm_q4_0_i8_k512_n128", 32, 512, 128, 32, 64),
        // Occupancy-sweep tilings (validate the 2×2 / 1×2 index paths).
        ("gemm_q4_0_i8_t22_k512_n128", 32, 512, 128, 32, 32),
        ("gemm_q4_0_i8_t12_k512_n128", 16, 512, 128, 16, 32),
        // 4×4 tiling (f16-equivalent), validates the 64×64 tile index path.
        ("gemm_q4_0_i8_t44_k512_n256", 64, 512, 256, 64, 64),
    ];

    for (variant, m, k, n, m_rows, n_cols) in cases {
        let kernel = ctx.load_kernel(variant).expect(variant);
        let nb = k / QK4_0;

        let weights = random_q4_0(&mut rng, n * k / QK4_0);
        let x = through_f16(&rng.f32_vec(m * k));

        // Oracle: same Q8_0 quant of the same activations.
        let want = mmq_q4_0_q8(&weights, &x, m, k, n);

        // GPU operands: W stays Q4_0 packed (unpacked to i8 in-kernel, uploaded
        // as u32 words); X → Q8_0 i8 + scales (per row).
        let mut x_i8 = Vec::with_capacity(m * k);
        let mut x_scales = Vec::with_capacity(m * nb);
        for mi in 0..m {
            let (sc, q) = quant_q8_0(&x[mi * k..][..k]);
            x_scales.extend_from_slice(&sc);
            x_i8.extend(q.iter().map(|&b| b as i8));
        }

        let w_buf = ctx
            .buffer_from_iter(
                weights
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap())),
                BufferUsage::STORAGE_BUFFER,
            )
            .unwrap();
        let x_buf = ctx
            .buffer_from_iter(x_i8, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let xs_buf = ctx
            .buffer_from_iter(x_scales, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let y_buf = ctx
            .new_buffer::<u16>((m * n) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();

        ctx.dispatch_blocking(
            &kernel,
            vec![
                WriteDescriptorSet::buffer(0, w_buf),
                WriteDescriptorSet::buffer(1, x_buf),
                WriteDescriptorSet::buffer(2, xs_buf),
                WriteDescriptorSet::buffer(3, y_buf.clone()),
            ],
            None::<u32>,
            [(n / n_cols) as u32, (m / m_rows) as u32, 1],
        )
        .unwrap();

        let got = from_f16_bits(&y_buf.read().unwrap());
        let err = nrmse(&got, &want);
        eprintln!("{variant}: kernel vs MMQ oracle nrmse {err:.6}");
        assert_close(&got, &want, 2e-2, 2e-2, variant);
    }
}
