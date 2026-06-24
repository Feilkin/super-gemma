//! int8-MMQ down-gemm steady-state A/B — the TRUSTWORTHY harness for the small
//! prefetch deltas. `mmq_tflops` has no clock warm-up, and its prefetch-variant
//! numbers swung 1.5–5.4% run-to-run (2026-06-18) — unusable for a few-percent
//! comparison. This (a) warms to the boost clock first, then (b) times the
//! variants ROUND-ROBIN within each batch, so any residual drift hits every
//! variant equally and the pairwise Δ survives it. Reports per-variant median
//! TFLOPS + CV; if the CV bands overlap the Δ is a wash.
//! Run: `cargo bench -p sg-gpu --bench mmq_variance`. Pin perf=high first.

use std::time::{Duration, Instant};

use sg_gpu::{Buffer, BufferBinding, BufferUsage, CommandGraph, GpuContext, Kernel};

const N_BLOCK: u32 = 16;
const BATCHES: usize = 50;
const WARMUP: Duration = Duration::from_secs(12);

/// Dispatches recorded per timed submit. **Default 1** — with >1, back-to-back
/// dispatches in one submit can OVERLAP (no barrier between them: a coopStore to
/// `y` is invisible to vulkano auto-sync, so it inserts none). Whether they overlap
/// differs per kernel, and an overlapping kernel thrashes its own L2 across the
/// concurrent dispatches → ~2× slower, confounding cross-kernel A/Bs (the l2 vs
/// basic_dir red herring, 2026-06-21). One dispatch per submit serializes via the
/// fence — the production-representative per-dispatch rate (the real prefill graph
/// barriers GEMMs on their data deps). `SG_BENCH_DISPATCHES>1` only to study overlap.
fn dispatches() -> usize {
    std::env::var("SG_BENCH_DISPATCHES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
}

/// Shape + M from env (default down/256 = the original config). `SG_BENCH_SHAPE`
/// = down|up, `SG_BENCH_M` = prefill chunk rows. The occupancy/bytes confirmation
/// (STATUS 2026-06-21) sweeps the up shape and a second M for the basic vs
/// basic_dir (LDS-scratch vs direct-store) A/B.
fn config() -> (usize, usize, usize, &'static [(&'static str, &'static str, u32)]) {
    let m: usize = std::env::var("SG_BENCH_M")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    match std::env::var("SG_BENCH_SHAPE").as_deref() {
        Ok("up") => (m, 5376, 21504, VARIANTS_UP),
        _ => (m, 21504, 5376, VARIANTS_DOWN),
    }
}

/// Pick which variants actually run. The full table is the historical A/B archive
/// (mostly documented negatives); running all ~40 each pass sustains a lot of heat.
/// Row 0 (the deployed reference) is ALWAYS kept — it's the Δ baseline.
///   unset             → active set: deployed reference + the current line of work
///                       (the `bb` family); falls back to the full table if that
///                       selects nothing (e.g. the tiny `up` set).
///   SG_BENCH_ONLY=all → the full table.
///   SG_BENCH_ONLY=a,b → row 0 plus every other row whose label or kernel name
///                       contains one of the comma-separated needles.
fn select(
    all: &'static [(&'static str, &'static str, u32)],
) -> Vec<(&'static str, &'static str, u32)> {
    match std::env::var("SG_BENCH_ONLY").ok().as_deref() {
        Some("all") => all.to_vec(),
        Some(list) => {
            let needles: Vec<&str> = list
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .collect();
            let mut out = vec![all[0]];
            out.extend(all.iter().skip(1).copied().filter(|(label, kern, _)| {
                needles.iter().any(|n| label.contains(n) || kern.contains(n))
            }));
            out
        }
        None => {
            let active: Vec<_> = all
                .iter()
                .copied()
                .filter(|(_, kern, _)| *kern == all[0].1 || kern.contains("bb"))
                .collect();
            if active.len() <= 1 {
                all.to_vec()
            } else {
                active
            }
        }
    }
}

/// Down-gemm variants under test — same shape/bindings, so the only difference
/// is the kernel body + its tile decomposition (`m_block` = M-rows per
/// workgroup). `deployed` carries the banked deep-prefetch (was +6.1% over the
/// pre-prefetch kernel here, 2026-06-18); `s1` (single-buffered scale) is the
/// standing occupancy A/B baseline (−2.1%); `occ` is the max-occupancy 1×1
/// rewrite (16-row blocks → many waves/SIMD; the occupancy-vs-reuse test).
const VARIANTS_DOWN: &[(&str, &str, u32)] = &[
    ("deployed (s2+pf)", "gemm_q4_0_i8_swz_m4n1_k21504_n5376", 64),
    ("deployed epi (coopStore)", "gemm_q4_0_i8_swz_m4n1_epi_k21504_n5376", 64),
    // basic: clean no-frills 4×1 (no β×2/prefetch/swizzle) — the readable baseline
    // (STATUS 2026-06-21). How much do all the deployed optimizations actually buy?
    ("basic", "gemm_q4_0_i8_basic_k21504_n5376", 64),
    // basic with the direct f16-coopmat store epilogue (no LDS scratch) — epilogue
    // A/B vs the LDS round-trip.
    ("basic dir", "gemm_q4_0_i8_basic_dir_k21504_n5376", 64),
    // basic with per-tile epilogue (EPI_TILES=1): higher occupancy, SAME coalesced
    // store — isolates occupancy-thrash from the store pattern.
    ("basic e1", "gemm_q4_0_i8_basic_e1_k21504_n5376", 64),
    // L2-blocking experiment kernel (1D dispatch). l2 = transpose (BN_SB=1);
    // _b2/_b4/_b8 = 2D super-block sweep (BN_SB n-blocks, m-outer).
    ("l2 (b1)", "gemm_q4_0_i8_l2", 64),
    ("l2 b2", "gemm_q4_0_i8_l2_b2", 64),
    ("l2 b4", "gemm_q4_0_i8_l2_b4", 64),
    ("l2 b8", "gemm_q4_0_i8_l2_b8", 64),
    // weight prefetch (PF=1) — the MLP lever, on b4 and the plain transpose.
    ("l2 b4 pf", "gemm_q4_0_i8_l2_b4_pf", 64),
    // β×2 WMMA-ILP: alone vs + prefetch (the full combo vs deployed's low-occ ILP).
    // BN_SB sweep WITH β×2.
    ("l2 sb1 b2", "gemm_q4_0_i8_l2_b1_b2", 64),
    ("l2 sb2 b2", "gemm_q4_0_i8_l2_b2_b2", 64),
    ("l2 sb4 b2", "gemm_q4_0_i8_l2_b4_b2", 64),
    ("l2 sb8 b2", "gemm_q4_0_i8_l2_b8_b2", 64),
    ("l2 b4 pf b2", "gemm_q4_0_i8_l2_b4_pf_b2", 64),
    ("l2 b4 b2 sf16", "gemm_q4_0_i8_l2_b4_b2_sf16", 64),
    ("l2 b4 b2 pf5", "gemm_q4_0_i8_l2_b4_b2_pf5", 64),
    ("l2 b4 b2 axp", "gemm_q4_0_i8_l2_b4_b2_axp", 64),
    ("l2 b4 b2 axp2", "gemm_q4_0_i8_l2_b4_b2_axp2", 64),
    ("l2 b4 b2 axp3", "gemm_q4_0_i8_l2_b4_b2_axp3", 64),
    ("l2 axp4 (ping-pong)", "gemm_q4_0_i8_l2_axp4", 64),
    ("l2 axp4+pf", "gemm_q4_0_i8_l2_axp4_pf", 64),
    // 8×1 tile (M_ROWS=128 → mb=128 for the grid math): register-level weight reuse.
    ("l2 m8", "gemm_q4_0_i8_l2_m8", 128),
    ("l2 m8 b4", "gemm_q4_0_i8_l2_m8_b4", 128),
    // PD=2: deeper weight prefetch (2 loads outstanding/wave) — the MLP A/B for
    // the memory-latency-bound GEMM (+9 VGPR; STATUS 2026-06-21).
    ("pd2", "gemm_q4_0_i8_swz_m4n1_pd2_k21504_n5376", 64),
    // axpf: post-barrier hoist of X loads (proven moot — ACO already schedules it).
    ("axpf", "gemm_q4_0_i8_swz_m4n1_axpf_k21504_n5376", 64),
    // axpf2: CROSS-BARRIER X prefetch (issued before unpack+barrier) — the real
    // attack on the fenced vmcnt stall (RGP 2026-06-21).
    ("axpf2", "gemm_q4_0_i8_swz_m4n1_axpf2_k21504_n5376", 64),
    ("s1", "gemm_q4_0_i8_swz_m4n1_s1_k21504_n5376", 64),
    ("occ 1×1", "gemm_q4_0_i8_occ_k21504_n5376", 16),
    // Full-occupancy small-tile family (gemm_q4_0_i8_fo): 0-stride scales + direct
    // coopStore + minimal LDS → ~11/16 waves at 60 VGPR (the occ footprint) but
    // with 2× weight reuse (m2) and no barrier-serializing LDS stage/epilogue. The
    // occupancy-vs-deployed-ILP A/B; sweep tile height, ILP, prefetch, barriers.
    ("fo m2", "gemm_q4_0_i8_fo", 32),
    ("fo m1", "gemm_q4_0_i8_fo_m1", 16),
    ("fo m4", "gemm_q4_0_i8_fo_m4", 64),
    ("fo m2 b2", "gemm_q4_0_i8_fo_m2_b2", 32),
    ("fo m1 b2", "gemm_q4_0_i8_fo_m1_b2", 16),
    ("fo m2 pd1", "gemm_q4_0_i8_fo_m2_pd1", 32),
    // d_a activation-scale hoist (SXP): the post-WMMA 16-bit x_scales load stalls
    // ~2K clk (RGP); hoist it to the top of the β-iter to overlap with the WMMAs.
    ("fo m2 sxp", "gemm_q4_0_i8_fo_m2_sxp", 32),
    ("fo m2 b2 sxp", "gemm_q4_0_i8_fo_m2_b2_sxp", 32),
    ("fo m1 b2 sxp", "gemm_q4_0_i8_fo_m1_b2_sxp", 16),
    // "Bigboy": fully hand-unrolled, all loads batched → one stall/iter (low occ).
    ("bb", "gemm_q4_0_i8_bb", 32),
    ("bb pf", "gemm_q4_0_i8_bb_pf", 32),
    // bb at the deployed tile height (M_TILES=4): half the waves, 2× weight reuse.
    ("bb m4", "gemm_q4_0_i8_bb_m4", 64),
    ("bb m4 pf", "gemm_q4_0_i8_bb_m4_pf", 64),
    // bb_m4 + depth-D cooperative-LDS weight prefetch — the weight-stall sweep.
    ("bb pfd1", "gemm_q4_0_i8_bb_pfd1", 64),
    ("bb pfd2", "gemm_q4_0_i8_bb_pfd2", 64),
    ("bb pfd4", "gemm_q4_0_i8_bb_pfd4", 64),
    // bb_m4 + super-block (BN_SB) swizzle — the L2-schedule sweep.
    ("bb m4 swz sb1", "gemm_q4_0_i8_bb_m4_swz_sb1", 64),
    ("bb m4 swz sb2", "gemm_q4_0_i8_bb_m4_swz_sb2", 64),
    ("bb m4 swz sb4", "gemm_q4_0_i8_bb_m4_swz_sb4", 64),
    ("bb m4 swz sb8", "gemm_q4_0_i8_bb_m4_swz_sb8", 64),
];

/// Up shape (FFN gate/up, K=5376 N=21504) — the occupancy/bytes confirmation set:
/// deployed reference + the basic (LDS scratch, 5/16 occ) vs basic_dir (direct
/// store, 9/16 occ) A/B.
const VARIANTS_UP: &[(&str, &str, u32)] = &[
    ("deployed (s2+pf)", "gemm_q4_0_i8_swz_m4n1_k5376_n21504", 64),
    ("basic", "gemm_q4_0_i8_basic_k5376_n21504", 64),
    ("basic dir", "gemm_q4_0_i8_basic_dir_k5376_n21504", 64),
];

fn sclk_mhz() -> String {
    std::fs::read_to_string("/sys/class/drm/card1/device/pp_dpm_sclk")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.contains('*'))
                .map(|l| l.split_whitespace().nth(1).unwrap_or("?").to_string())
        })
        .unwrap_or_else(|| "?".into())
}

fn main() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping: no usable GPU ({e})");
            return;
        }
    };
    if !ctx.cooperative_matrix {
        eprintln!("skipping: no VK_KHR_cooperative_matrix");
        return;
    }
    let (mdim, kdim, ndim, variants_all) = config();
    let variants = select(variants_all);
    let n_disp = dispatches();
    eprintln!(
        "running {} / {} variants (SG_BENCH_ONLY to widen/narrow)",
        variants.len(),
        variants_all.len()
    );

    // int8 operands: Q4_0 weights (u32 words), Q8 activations (i8 packed u32),
    // f16 scales, f16 out. Dummy-filled — steady-state timing is data-independent.
    let w: Buffer<u32> = ctx
        .buffer_from_iter(
            (0..(ndim * kdim / 32 * 18 / 4) as u32).map(|i| i.wrapping_mul(0x9E37_79B9)),
            BufferUsage::STORAGE_BUFFER,
        )
        .unwrap();
    let x = ctx
        .new_buffer::<u32>((mdim * kdim / 4) as u64, BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let xs = ctx
        .new_buffer::<u16>((mdim * kdim / 32) as u64, BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let y = ctx
        .new_buffer::<u16>((mdim * ndim) as u64, BufferUsage::STORAGE_BUFFER)
        .unwrap();

    let kernels: Vec<Kernel> = variants
        .iter()
        .map(|(_, kn, _)| ctx.load_kernel(kn).expect(kn))
        .collect();
    // Per-variant grid (m_block differs: 64 for the 4×1 tile, 16 for the 1×1 occ
    // kernel — both cover the full M×N). Swizzled kernels take [M-blocks,
    // N-blocks]; the swizzle-less `basic` baseline takes the transposed
    // [N-blocks, M-blocks] (wg.x=N, wg.y=M).
    let grids: Vec<[u32; 3]> = variants
        .iter()
        .map(|(_, kern, mb)| {
            if kern.contains("_l2") || kern.contains("bb_m4_swz") {
                // 1D: one workgroup per output tile, decoded in-kernel (tile_index).
                [(ndim as u32 / N_BLOCK) * (mdim as u32 / mb), 1, 1]
            } else if kern.contains("basic") {
                [ndim as u32 / N_BLOCK, mdim as u32 / mb, 1]
            } else {
                [mdim as u32 / mb, ndim as u32 / N_BLOCK, 1]
            }
        })
        .collect();
    // GPU timestamps bracket the dispatches (BottomOfPipe→BottomOfPipe), so we
    // can compare PURE on-GPU execution against the CPU wall clock and confirm the
    // CB-build/submit overhead isn't in the number. One pre-recorded graph per
    // variant; `run` re-submits it and returns the GPU ns. `dispatch_overlapping`
    // (no inter-dispatch barrier) preserves the historical n_disp>1 overlap
    // semantics this harness is built around (default n_disp=1 → no overlap).
    let timer = ctx.new_timer(2).expect("timer");
    let graphs: Vec<CommandGraph> = kernels
        .iter()
        .zip(&grids)
        .map(|(k, &grid)| {
            ctx.record_graph(|rec| {
                rec.reset_timer(&timer)?;
                rec.timestamp(&timer, 0)?;
                for _ in 0..n_disp {
                    rec.dispatch_overlapping(
                        k,
                        vec![
                            BufferBinding::buffer(0, w.clone()),
                            BufferBinding::buffer(1, x.clone()),
                            BufferBinding::buffer(2, xs.clone()),
                            BufferBinding::buffer(3, y.clone()),
                        ],
                        None::<u32>,
                        grid,
                    )?;
                }
                rec.timestamp(&timer, 1)?;
                Ok(())
            })
            .unwrap()
        })
        .collect();
    let run = |g: &CommandGraph| -> f64 {
        ctx.submit_blocking(g).unwrap();
        let ts = timer.read_ns().unwrap();
        ts[1] - ts[0]
    };

    // Warm up to the boost clock (round-robin so no variant is favoured).
    let t0 = Instant::now();
    while t0.elapsed() < WARMUP {
        for g in &graphs {
            run(g);
        }
    }
    eprintln!("warmed up, sclk={}", sclk_mhz());

    let flops = 2.0 * mdim as f64 * ndim as f64 * kdim as f64 * n_disp as f64;
    // Per variant: wall-clock TFLOPS (CPU round-trip) and GPU-timestamp TFLOPS
    // (BottomOfPipe interval). The two columns quantify the CB/submit overhead.
    let mut wall: Vec<Vec<f64>> = vec![Vec::with_capacity(BATCHES); variants.len()];
    let mut gpu: Vec<Vec<f64>> = vec![Vec::with_capacity(BATCHES); variants.len()];
    for _ in 0..BATCHES {
        for (i, g) in graphs.iter().enumerate() {
            let start = Instant::now();
            let gpu_ns = run(g);
            wall[i].push(flops / start.elapsed().as_secs_f64() / 1e12);
            gpu[i].push(flops / (gpu_ns / 1e9) / 1e12);
        }
    }

    eprintln!(
        "int8 gemm steady-state K={kdim} N={ndim} M={mdim} ({BATCHES} batches × {n_disp} \
         dispatches), sclk={}:",
        sclk_mhz()
    );
    let median = |v: &Vec<f64>| {
        let mut v = v.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let cv = |v: &Vec<f64>| {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        100.0 * sd / mean
    };
    let wall_base = median(&wall[0]);
    let gpu_base = median(&gpu[0]);
    eprintln!("  {:18} {:>22}   {:>22}", "", "wall (CPU round-trip)", "gpu (timestamps)");
    for (i, (label, _, _)) in variants.iter().enumerate() {
        let (wm, gm) = (median(&wall[i]), median(&gpu[i]));
        eprintln!(
            "  {label:18} {wm:6.2} TFLOPS cv {:.2}% Δ{:+5.1}%   {gm:6.2} TFLOPS cv {:.2}% Δ{:+5.1}%",
            cv(&wall[i]),
            100.0 * (wm - wall_base) / wall_base,
            cv(&gpu[i]),
            100.0 * (gm - gpu_base) / gpu_base,
        );
    }
}
