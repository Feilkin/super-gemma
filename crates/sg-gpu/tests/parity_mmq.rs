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
        // Depth-2 weight prefetch (w_next2 shift/prologue); same 2×2 as t22.
        ("gemm_q4_0_i8_pd2_k512_n128", 32, 512, 128, 32, 32),
        // Activation prefetch (hoisted A-loads / reordered MMA); same 2×2.
        ("gemm_q4_0_i8_axpf_k512_n128", 32, 512, 128, 32, 32),
        // Cross-barrier activation prefetch (ap0/ap1 issued pre-barrier); same 2×2.
        ("gemm_q4_0_i8_axpf2_k512_n128", 32, 512, 128, 32, 32),
        ("gemm_q4_0_i8_t12_k512_n128", 16, 512, 128, 16, 32),
        // 4×4 tiling (f16-equivalent), validates the 64×64 tile index path.
        ("gemm_q4_0_i8_t44_k512_n256", 64, 512, 256, 64, 64),
        // Multi-wave kernel (one tile/wave): b22 exercises the 2D wave grid
        // (wm/wn both vary), b41 the 1D-M headline config (wn≡0). m_rows=BM,
        // n_cols=BN; dispatch [n/BN, m/BM] (these variants are SWIZZLE=0).
        ("gemm_q4_0_i8_mw_b22_k512_n128", 64, 512, 128, 32, 32),
        ("gemm_q4_0_i8_mw_b41_k512_n128", 64, 512, 128, 64, 16),
        // RM=2 register-tiled (half-occupancy): 2 waves, each 2 M-tiles. BM=64,
        // BN=16; exercises the RM>1 MMA + epilogue path.
        ("gemm_q4_0_i8_mw_r2_k512_n128", 64, 512, 128, 64, 16),
        // Clean no-frills 4×1 baseline (all barriers on): de-interleaved blocks,
        // 0-stride rescale, LDS-scratch epilogue. m_rows=64, n_cols=16.
        ("gemm_q4_0_i8_basic_k512_n128", 64, 512, 128, 64, 16),
        // Direct f16-coopmat store epilogue (no LDS scratch); same 4×1.
        ("gemm_q4_0_i8_basic_dir_k512_n128", 64, 512, 128, 64, 16),
        // Per-tile epilogue passes (EPI_TILES=1, smaller scratch); same 4×1.
        ("gemm_q4_0_i8_basic_e1_k512_n128", 64, 512, 128, 64, 16),
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

/// The L2-blocking kernel (`gemm_q4_0_i8_l2`, hardcoded down shape, 1D dispatch
/// with the `tile_index` decode) must compute the SAME output as the
/// oracle-verified `basic_dir` for the same shape — the only difference is which
/// workgroup computes which tile. Cross-checks against `basic_dir` (fast, on-GPU)
/// rather than the slow CPU oracle. M=128 → 2 M-blocks × 336 N-blocks = 672 tiles.
#[test]
fn gemm_q4_0_i8_l2_matches_basic_dir() {
    let Some(ctx) = ctx() else { return };
    if !ctx.cooperative_matrix {
        eprintln!("skipping: no VK_KHR_cooperative_matrix");
        return;
    }
    let (m, k, n) = (256usize, 21504usize, 5376usize); // M=256: kernel hardcodes NB_M=4
    let nb = k / QK4_0;
    let mut rng = Rng::new(0x5C);
    let weights = random_q4_0(&mut rng, n * k / QK4_0);
    let x = through_f16(&rng.f32_vec(m * k));
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
    let x_buf = ctx.buffer_from_iter(x_i8, BufferUsage::STORAGE_BUFFER).unwrap();
    let xs_buf = ctx
        .buffer_from_iter(x_scales, BufferUsage::STORAGE_BUFFER)
        .unwrap();

    // (variant, grid) — basic_dir is 2D [N-blocks, M-blocks]; l2 is 1D [tiles].
    let run = |variant: &str, grid: [u32; 3]| -> Vec<f32> {
        let kernel = ctx.load_kernel(variant).expect(variant);
        let y_buf = ctx
            .new_buffer::<u16>((m * n) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        ctx.dispatch_blocking(
            &kernel,
            vec![
                WriteDescriptorSet::buffer(0, w_buf.clone()),
                WriteDescriptorSet::buffer(1, x_buf.clone()),
                WriteDescriptorSet::buffer(2, xs_buf.clone()),
                WriteDescriptorSet::buffer(3, y_buf.clone()),
            ],
            None::<u32>,
            grid,
        )
        .unwrap();
        from_f16_bits(&y_buf.read().unwrap())
    };
    let nb_n = (n / 16) as u32;
    let want = run("gemm_q4_0_i8_basic_dir_k21504_n5376", [nb_n, (m / 64) as u32, 1]);
    // l2 (4×1, M_ROWS=64) and l2_m8 (8×1, M_ROWS=128) — both decode tiles internally
    // and must reproduce basic_dir. Grid = [(N/16)·(M/M_ROWS), 1, 1].
    for (variant, m_rows) in [
        ("gemm_q4_0_i8_l2", 64u32),
        ("gemm_q4_0_i8_l2_m8", 128u32),
        ("gemm_q4_0_i8_l2_b4_pf", 64u32), // weight prefetch
        ("gemm_q4_0_i8_l2_b4_b2", 64u32), // β×2 interleave
        ("gemm_q4_0_i8_l2_b4_pf_b2", 64u32), // prefetch + β×2 (full combo)
        ("gemm_q4_0_i8_l2_b4_b2_sf16", 64u32), // f16 scale staging
        ("gemm_q4_0_i8_l2_b4_b2_pf5", 64u32), // minimal-fetch prefetch (PFW=5)
        ("gemm_q4_0_i8_l2_b4_b2_axp", 64u32), // activation prefetch (hoist)
        ("gemm_q4_0_i8_l2_b4_b2_axp2", 64u32), // activation prefetch (cross-iter)
        ("gemm_q4_0_i8_l2_b4_b2_axp3", 64u32), // activation prefetch (cross-iter, double-buffered)
        ("gemm_q4_0_i8_l2_axp4", 64u32), // static-unroll ping-pong (dedicated file)
        ("gemm_q4_0_i8_l2_axp4_pf", 64u32), // ping-pong + weight prefetch
        ("gemm_q4_0_i8_bb_m4_swz_sb1", 64u32), // bb_m4 + super-block swizzle (1D)
        ("gemm_q4_0_i8_bb_m4_swz_sb2", 64u32),
        ("gemm_q4_0_i8_bb_m4_swz_sb4", 64u32),
        ("gemm_q4_0_i8_bb_m4_swz_sb8", 64u32),
    ] {
        let got = run(variant, [nb_n * (m as u32 / m_rows), 1, 1]);
        let err = nrmse(&got, &want);
        eprintln!("{variant} vs basic_dir nrmse {err:.8}");
        assert_close(&got, &want, 1e-4, 1e-4, variant);
    }

    // Split-M kernel: M-block from a push constant, grid = [N-strips, 1, 1]. The
    // host issues NB_M=4 dispatches (push 0..3), each writing a disjoint M-block of
    // Y, which together must reproduce basic_dir.
    {
        let kernel = ctx
            .load_kernel("gemm_q4_0_i8_bb_m4_split")
            .expect("split");
        let y_buf = ctx
            .new_buffer::<u16>((m * n) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let nb_m = (m / 64) as u32;
        for mb in 0..nb_m {
            ctx.dispatch_blocking(
                &kernel,
                vec![
                    WriteDescriptorSet::buffer(0, w_buf.clone()),
                    WriteDescriptorSet::buffer(1, x_buf.clone()),
                    WriteDescriptorSet::buffer(2, xs_buf.clone()),
                    WriteDescriptorSet::buffer(3, y_buf.clone()),
                ],
                Some(mb),
                [nb_n, 1, 1],
            )
            .unwrap();
        }
        let got = from_f16_bits(&y_buf.read().unwrap());
        let err = nrmse(&got, &want);
        eprintln!("gemm_q4_0_i8_bb_m4_split vs basic_dir nrmse {err:.8}");
        assert_close(&got, &want, 1e-4, 1e-4, "gemm_q4_0_i8_bb_m4_split");
    }

    // Full-occupancy kernel (gemm_q4_0_i8_fo) — plain 2D [M-blocks, N-blocks] grid
    // (wg.x = m_block, wg.y = n_block), tile height M_ROWS = M_TILES·16. Must
    // reproduce basic_dir for every tiling / ILP / prefetch / barrier toggle.
    for (variant, m_rows) in [
        ("gemm_q4_0_i8_fo", 32u32),          // M_TILES=2 (headline)
        ("gemm_q4_0_i8_fo_m1", 16u32),       // 1×1
        ("gemm_q4_0_i8_fo_m4", 64u32),       // 4×1
        ("gemm_q4_0_i8_fo_m1_b2", 16u32),    // β×2 ILP
        ("gemm_q4_0_i8_fo_m2_b2", 32u32),    // β×2 ILP
        ("gemm_q4_0_i8_fo_m2_pd1", 32u32),   // weight prefetch
        ("gemm_q4_0_i8_fo_m2_sxp", 32u32),   // activation-scale hoist
        ("gemm_q4_0_i8_fo_m2_b2_sxp", 32u32), // β×2 + scale hoist
        ("gemm_q4_0_i8_fo_m1_b2_sxp", 16u32),
        ("gemm_q4_0_i8_bb", 32u32),          // fully-unrolled bigboy
        ("gemm_q4_0_i8_bb_pf", 32u32),       // bb + weight prefetch
        ("gemm_q4_0_i8_bb_m4", 64u32),       // bb, M_TILES=4 (deployed tile height)
        ("gemm_q4_0_i8_bb_m4_pf", 64u32),    // bb_m4 + weight prefetch
    ] {
        let got = run(variant, [m as u32 / m_rows, nb_n, 1]);
        let err = nrmse(&got, &want);
        eprintln!("{variant} vs basic_dir nrmse {err:.8}");
        assert_close(&got, &want, 1e-4, 1e-4, variant);
    }

    // Deployed swz down (4×1, SWIZZLE=1 → 2D [M-blocks, N-blocks]) and its EPI=1
    // direct-coopStore epilogue: EPI only changes the y store path, so the epi
    // variant must be BIT-IDENTICAL to the deployed kernel.
    let dep_grid = [(m / 64) as u32, nb_n, 1];
    let dep = run("gemm_q4_0_i8_swz_m4n1_k21504_n5376", dep_grid);
    let epi = run("gemm_q4_0_i8_swz_m4n1_epi_k21504_n5376", dep_grid);
    let err = nrmse(&epi, &dep);
    eprintln!("swz_m4n1_epi vs deployed nrmse {err:.8}");
    assert_close(&epi, &dep, 1e-6, 1e-6, "swz_m4n1_epi");
}

/// Barrier probe (STATUS 2026-06-21): which of the basic 4×1 kernel's three
/// workgroupBarriers are actually required? Runs the all-on baseline and each
/// single-barrier-dropped variant and PRINTS nrmse — non-asserting (a dropped
/// barrier that races shows a large/garbage nrmse; one that's redundant stays
/// ~2e-4). Run with `--nocapture`. Only the baseline is assert-checked so CI
/// stays green regardless of what the probe reveals.
#[test]
fn gemm_q4_0_i8_basic_barrier_probe() {
    let Some(ctx) = ctx() else { return };
    if !ctx.cooperative_matrix {
        eprintln!("skipping: no VK_KHR_cooperative_matrix");
        return;
    }
    let (m, k, n, m_rows, n_cols) = (64usize, 512usize, 128usize, 64usize, 16usize);
    let nb = k / QK4_0;
    let mut rng = Rng::new(0x4D);
    let weights = random_q4_0(&mut rng, n * k / QK4_0);
    let x = through_f16(&rng.f32_vec(m * k));
    let want = mmq_q4_0_q8(&weights, &x, m, k, n);

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
    let x_buf = ctx.buffer_from_iter(x_i8, BufferUsage::STORAGE_BUFFER).unwrap();
    let xs_buf = ctx
        .buffer_from_iter(x_scales, BufferUsage::STORAGE_BUFFER)
        .unwrap();

    for variant in [
        "gemm_q4_0_i8_basic_k512_n128",      // all barriers on
        "gemm_q4_0_i8_basic_nowb_k512_n128", // drop RAW wb
        "gemm_q4_0_i8_basic_noda_k512_n128", // drop RAW da_l
        "gemm_q4_0_i8_basic_nowar_k512_n128", // drop WAR
    ] {
        let kernel = ctx.load_kernel(variant).expect(variant);
        let y_buf = ctx
            .new_buffer::<u16>((m * n) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        ctx.dispatch_blocking(
            &kernel,
            vec![
                WriteDescriptorSet::buffer(0, w_buf.clone()),
                WriteDescriptorSet::buffer(1, x_buf.clone()),
                WriteDescriptorSet::buffer(2, xs_buf.clone()),
                WriteDescriptorSet::buffer(3, y_buf.clone()),
            ],
            None::<u32>,
            [(n / n_cols) as u32, (m / m_rows) as u32, 1],
        )
        .unwrap();
        let got = from_f16_bits(&y_buf.read().unwrap());
        eprintln!("BARRIER PROBE {variant}: nrmse {:.6}", nrmse(&got, &want));
    }
    eprintln!("(baseline must be ~2e-4; a needed barrier shows large nrmse when dropped)");
}
