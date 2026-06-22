//! Single-submit dispatch of one prefill GEMM, for RGP/SQTT capture
//! (docs/rgp-capture.md). One `dispatch_blocking` == one queue submit, so under
//! `MESA_VK_TRACE=rgp MESA_VK_TRACE_PER_SUBMIT=true` each loop iteration emits
//! one `.rgp` — take the LAST (warmest). Buffers are dummy: SQTT captures the
//! dispatch's wave/stall/occupancy behavior, not the result (a GEMM has no
//! data-dependent control flow, so the trace is representative).

use sg_gpu::{BufferUsage, GpuContext, WriteDescriptorSet};

const SUBMITS: usize = 8;

/// Prefill chunk rows (the M dimension). Default 256 (production chunk); override
/// with `SG_RGP_M` for the two-M GEMM capture (weight re-stream scales M/64,
/// activation traffic scales M·K — the slope splits the 800M local-video).
fn rgp_m() -> usize {
    std::env::var("SG_RGP_M")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256)
}

/// (K, N, N-block, M-block, int8, swizzled) for the supported capture targets.
/// `swizzled` kernels take a transposed `[M-blocks, N-blocks]` grid (the L2
/// fast-varying-M dispatch); the rest take `[N-blocks, M-blocks]`.
fn shape(kernel: &str) -> Option<(usize, usize, u32, u32, bool, bool)> {
    Some(match kernel {
        // int8 4×1 swizzled (N-block 16, M-block 64) — the DEPLOYED prefill
        // tiling (graph.rs); the occupancy/L2 recapture target (STATUS).
        "gemm_q4_0_i8_swz_m4n1_k21504_n5376" => (21504, 5376, 16, 64, true, true), // FFN down
        "gemm_q4_0_i8_swz_m4n1_epi_k21504_n5376" => (21504, 5376, 16, 64, true, true), // FFN down, direct-coopStore epilogue
        "gemm_q4_0_i8_swz_m4n1_k5376_n21504" => (5376, 21504, 16, 64, true, true), // FFN gate/up
        // Clean no-frills 4×1 baseline (swizzle-less → [N-blocks, M-blocks] grid).
        "gemm_q4_0_i8_basic_k21504_n5376" => (21504, 5376, 16, 64, true, false), // FFN down
        "gemm_q4_0_i8_basic_k5376_n21504" => (5376, 21504, 16, 64, true, false), // FFN gate/up
        // Direct f16-coopmat store epilogue (no LDS) — the −25% A/B vs basic.
        "gemm_q4_0_i8_basic_dir_k21504_n5376" => (21504, 5376, 16, 64, true, false), // FFN down
        // Per-tile epilogue (higher occupancy, same coalesced store) — occupancy isolation.
        "gemm_q4_0_i8_basic_e1_k21504_n5376" => (21504, 5376, 16, 64, true, false), // FFN down
        // L2-blocking experiment kernel (1D dispatch, hardcoded down shape).
        "gemm_q4_0_i8_l2" => (21504, 5376, 16, 64, true, false), // FFN down
        "gemm_q4_0_i8_l2_k21504_n5376" => (21504, 5376, 16, 64, true, false), // alias
        "gemm_q4_0_i8_l2_b4" => (21504, 5376, 16, 64, true, false),      // 4×1, BN_SB=4 (best)
        "gemm_q4_0_i8_l2_b4_pf" => (21504, 5376, 16, 64, true, false),   // 4×1, BN_SB=4 + prefetch
        "gemm_q4_0_i8_l2_b4_b2" => (21504, 5376, 16, 64, true, false),   // 4×1, BN_SB=4 + β×2 (best l2)
        "gemm_q4_0_i8_l2_b4_pf_b2" => (21504, 5376, 16, 64, true, false), // + prefetch (worse)
        "gemm_q4_0_i8_l2_b4_b2_sf16" => (21504, 5376, 16, 64, true, false), // + f16 scale staging
        "gemm_q4_0_i8_l2_b4_b2_pf5" => (21504, 5376, 16, 64, true, false), // + minimal-fetch prefetch (PFW=5)
        "gemm_q4_0_i8_l2_b4_b2_axp" => (21504, 5376, 16, 64, true, false), // + activation prefetch (hoist)
        "gemm_q4_0_i8_l2_b4_b2_axp2" => (21504, 5376, 16, 64, true, false), // + activation prefetch (cross-iter)
        "gemm_q4_0_i8_l2_b4_b2_axp3" => (21504, 5376, 16, 64, true, false), // + activation prefetch (cross-iter, double-buffered)
        "gemm_q4_0_i8_l2_axp4" => (21504, 5376, 16, 64, true, false), // static-unroll ping-pong
        "gemm_q4_0_i8_l2_axp4_pf" => (21504, 5376, 16, 64, true, false), // ping-pong + weight prefetch
        "gemm_q4_0_i8_l2_m8" => (21504, 5376, 16, 128, true, false),     // 8×1 (M_ROWS=128)
        "gemm_q4_0_i8_l2_m8_b4" => (21504, 5376, 16, 128, true, false),  // 8×1, BN_SB=4
        // PD=2 (deeper weight prefetch) A/B vs the deployed 4×1 — the MLP lever for
        // the memory-latency-bound GEMM (STATUS 2026-06-21). Read occupancy +
        // whether the first-WMMA vmcnt stall shrinks vs the PD=1 capture.
        "gemm_q4_0_i8_swz_m4n1_pd2_k21504_n5376" => (21504, 5376, 16, 64, true, true), // FFN down
        "gemm_q4_0_i8_swz_m4n1_pd2_k5376_n21504" => (5376, 21504, 16, 64, true, true), // FFN gate/up
        // Activation prefetch (hoisted X loads) — attacks the vmcnt stall directly.
        "gemm_q4_0_i8_swz_m4n1_axpf_k21504_n5376" => (21504, 5376, 16, 64, true, true), // FFN down
        "gemm_q4_0_i8_swz_m4n1_axpf_k5376_n21504" => (5376, 21504, 16, 64, true, true), // FFN gate/up
        // Cross-barrier activation prefetch (issued before unpack+barrier).
        "gemm_q4_0_i8_swz_m4n1_axpf2_k21504_n5376" => (21504, 5376, 16, 64, true, true), // FFN down
        "gemm_q4_0_i8_swz_m4n1_axpf2_k5376_n21504" => (5376, 21504, 16, 64, true, true), // FFN gate/up
        // int8 max-occupancy 1×1 (N-block = M-block = 16) — the occupancy-vs-reuse
        // A/B against the deployed 4×1 down-gemm (mmq_variance −47.8%; this trace
        // confirms it reached high occupancy yet lost — STATUS rank #2).
        "gemm_q4_0_i8_occ_k21504_n5376" => (21504, 5376, 16, 16, true, true), // FFN down, 1×1
        // Full-occupancy GEMM (gemm_q4_0_i8_fo) — small-tile, plain [M-blocks,
        // N-blocks] dispatch (swz convention). mb = M_TILES·16; the occupancy A/B
        // against the deployed 4×1 (read VGPR/waves here, time on mmq_variance).
        "gemm_q4_0_i8_fo" => (21504, 5376, 16, 32, true, true),          // M_TILES=2
        "gemm_q4_0_i8_fo_m1" => (21504, 5376, 16, 16, true, true),       // M_TILES=1
        "gemm_q4_0_i8_fo_m4" => (21504, 5376, 16, 64, true, true),       // M_TILES=4
        "gemm_q4_0_i8_fo_m1_b2" => (21504, 5376, 16, 16, true, true),
        "gemm_q4_0_i8_fo_m2_b2" => (21504, 5376, 16, 32, true, true),
        "gemm_q4_0_i8_fo_m2_pd1" => (21504, 5376, 16, 32, true, true),
        "gemm_q4_0_i8_fo_m2_sxp" => (21504, 5376, 16, 32, true, true),    // d_a hoist
        "gemm_q4_0_i8_fo_m2_b2_sxp" => (21504, 5376, 16, 32, true, true), // β×2 + d_a hoist
        "gemm_q4_0_i8_fo_m1_b2_sxp" => (21504, 5376, 16, 16, true, true),
        // "Bigboy" fully-unrolled FO (M_TILES=2): all loads in one batch, one stall/iter.
        "gemm_q4_0_i8_bb" => (21504, 5376, 16, 32, true, true),
        "gemm_q4_0_i8_bb_pf" => (21504, 5376, 16, 32, true, true), // bb + weight prefetch
        // Multi-wave occupancy GEMM (one 16×16 tile/wave, shared LDS weight strip).
        // n-block = BN, m-block = BM; swizzled like the deployed 4×1. The down-shape
        // family A/Bs the deployed m4n1 down-gemm; b41 up matches the gate/up site.
        "gemm_q4_0_i8_mw_b41_k21504_n5376" => (21504, 5376, 16, 64, true, true), // 4 waves, reuse 64
        "gemm_q4_0_i8_mw_b22_k21504_n5376" => (21504, 5376, 32, 32, true, true), // 4 waves, reuse 32
        "gemm_q4_0_i8_mw_b42_k21504_n5376" => (21504, 5376, 32, 64, true, true), // 8 waves, reuse 64
        "gemm_q4_0_i8_mw_b81_k21504_n5376" => (21504, 5376, 16, 128, true, true), // 8 waves, reuse 128
        "gemm_q4_0_i8_mw_b41_k5376_n21504" => (5376, 21504, 16, 64, true, true), // gate/up, reuse 64
        // Half-occupancy A/B vs b41 (same 64×16 block + reuse 64): 2 waves × RM=2.
        "gemm_q4_0_i8_mw_r2_k21504_n5376" => (21504, 5376, 16, 64, true, true), // 2 waves, RM=2
        "gemm_q4_0_i8_mw_r2_k5376_n21504" => (5376, 21504, 16, 64, true, true), // gate/up, RM=2
        // int8 2×2 (N-block = M-block = 32) — the pre-cache-block baseline.
        "gemm_q4_0_i8_t22_k21504_n5376" => (21504, 5376, 32, 32, true, false), // FFN down
        "gemm_q4_0_i8_t22_k5376_n21504" => (5376, 21504, 32, 32, true, false), // FFN gate/up
        // f16 4×4 (N-block = M-block = 64) — every prefill GEMM site, for the
        // f16 A/B and the un-RGP'd-f16 investigation (STATUS).
        "gemm_q4_0_k5376_n21504" => (5376, 21504, 64, 64, false, false), // FFN gate/up
        "gemm_q4_0_k21504_n5376" => (21504, 5376, 64, 64, false, false), // FFN down
        "gemm_q4_0_k5376_n8192" => (5376, 8192, 64, 64, false, false),   // Q sliding
        "gemm_q4_0_k5376_n4096" => (5376, 4096, 64, 64, false, false),   // KV sliding
        "gemm_q4_0_k8192_n5376" => (8192, 5376, 64, 64, false, false),   // O sliding
        "gemm_q4_0_k5376_n16384" => (5376, 16384, 64, 64, false, false), // Q global
        "gemm_q4_0_k5376_n2048" => (5376, 2048, 64, 64, false, false),   // KV global
        "gemm_q4_0_k16384_n5376" => (16384, 5376, 64, 64, false, false), // O global
        _ => return None,
    })
}

/// Capture the real recorded prefill graph (not an isolated GEMM): one PER-LAYER
/// `.rgp` showing the full cross-kernel mix of a prefill layer — QKV gemms, flash
/// attention, the O gemm, FFN gate/up/down gemms, rmsnorm, rope, residual joins —
/// with their real overlap, at the production chunk size M=256.
///
/// Per-layer, not per-chunk, because SQTT records a continuous token stream whose
/// buffer fills on GPU *duration*: a single gemm fits (the `rgp <kernel>` harness),
/// but a 60-layer chunk — even one 6-layer watchdog segment — runs far too long to
/// hold (tested: overflows 1 GB). A layer is the repeating unit; layer 4 samples a
/// sliding layer and layer 5 the global one (`layer % 6 == 5`), so the two traces
/// together cover every prefill kernel. Drive `q0` for the operating point: 0 is
/// FFN-gemm-bound short context, 32512 is global-attention-bound long context.
///
/// Each layer is submitted on its own (`record_prefill_layers(i..i+1)`), so under
/// `MESA_VK_TRACE_PER_SUBMIT=true` each is one `.rgp`. We submit `PASSES` warm-up
/// rounds first (env is process-wide, so uploads + warm-ups also emit captures) —
/// **take the LAST two** `.rgp`: sliding layer 4 then global layer 5 of the warm
/// pass. Bump `RADV_THREAD_TRACE_BUFFER_SIZE` if a layer still overflows; drop
/// `RADV_THREAD_TRACE_INSTRUCTION_TIMING` to shrink it further (loses per-op timing
/// but keeps the occupancy/event/cache timeline).
pub fn run_prefill(model_path: &std::path::Path, q0: u32) -> anyhow::Result<()> {
    const CHUNK: usize = 256; // production prefill chunk (profile.rs sweet spot)
    const GLOBAL_CAP: usize = 32 * 1024;
    const PASSES: usize = 3; // warm-ups + final; take the LAST 2 .rgp (layers 4,5)
    const REP_LAYERS: [usize; 2] = [4, 5]; // a sliding then a global layer

    let file = sg_gguf::GgufFile::open(model_path)?;
    let gguf = file.parse().map_err(|e| anyhow::anyhow!("parse: {e}"))?;
    let ctx = sg_gpu::GpuContext::new().map_err(|e| anyhow::anyhow!("gpu: {e}"))?;
    eprintln!("uploading weights …");
    let mut model = sg_model::GpuModel::new(&ctx, &gguf, GLOBAL_CAP, CHUNK)
        .map_err(|e| anyhow::anyhow!("upload: {e}"))?;
    // One single-layer graph per representative layer (sliding 4, global 5).
    let layer_graphs: Vec<(usize, sg_gpu::CommandGraph)> = REP_LAYERS
        .iter()
        .map(|&i| {
            model
                .record_prefill_layers(i..i + 1, CHUNK, CHUNK)
                .map(|g| (i, g))
        })
        .collect::<Result<_, _>>()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let chunk_tokens: Vec<u32> = (0..CHUNK as u32).map(|i| 1000 + i * 7).collect();
    model.reset();

    eprintln!(
        "RGP prefill capture: per-layer (sliding L4, global L5), M={CHUNK}, q0={q0}.\n\
         {PASSES} passes → {} layer .rgp; TAKE THE LAST 2 (warm L4 then L5).",
        PASSES * REP_LAYERS.len()
    );
    for pass in 0..PASSES {
        model.pos = q0;
        model
            .stage_prefill_chunk(&chunk_tokens)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        for (i, g) in &layer_graphs {
            model.submit(g).map_err(|e| anyhow::anyhow!("{e}"))?;
            let kind = if i % 6 == 5 { "global" } else { "sliding" };
            eprintln!("  pass {pass} layer {i} ({kind})");
        }
    }
    Ok(())
}

/// Decode GEMV targets (`y[N] = W[N×K]·x[K]`, M=1) — the BANDWIDTH-bound
/// reference: gemv hits ~91% of the membw ceiling (bench: gemv_bw), so its trace
/// is the "what does a memory-system that actually streams look like" calibration
/// for the latency-stalled prefill GEMMs. K is baked in the variant; N is the
/// dispatch row count. k21504/n5376 mirrors the FFN-down gemm shape.
fn gemv_shape(kernel: &str) -> Option<(usize, usize)> {
    Some(match kernel {
        "gemv_q4_0_k21504" => (21504, 5376), // FFN down
        "gemv_q4_0_k5376" => (5376, 21504),  // FFN gate/up
        "gemv_q4_0_k8192" => (8192, 5376),   // O sliding
        "gemv_q4_0_k16384" => (16384, 5376), // O global
        _ => return None,
    })
}

/// One decode GEMV per submit (3 bindings: Q4_0 weights, f16 x[K], f16 y[N];
/// grid = one wave-sized workgroup per output row). Dummy buffers — SQTT records
/// the wave/stall/occupancy behavior, not the result.
fn run_gemv(kernel: &str, k: usize, n: usize) -> anyhow::Result<()> {
    let ctx = GpuContext::new().map_err(|e| anyhow::anyhow!("gpu: {e}"))?;
    let kern = ctx.load_kernel(kernel).map_err(|e| anyhow::anyhow!("{e}"))?;
    let u = BufferUsage::STORAGE_BUFFER;
    let nb_err = |e: sg_gpu::GpuError| anyhow::anyhow!("{e}");

    let w = ctx
        .new_buffer::<u32>((n * k / 32 * 18 / 4) as u64, u)
        .map_err(nb_err)?;
    let x = ctx.new_buffer::<u16>(k as u64, u).map_err(nb_err)?;
    let y = ctx.new_buffer::<u16>(n as u64, u).map_err(nb_err)?;

    let groups = [n as u32, 1, 1]; // one workgroup per output row
    eprintln!(
        "RGP capture target: {kernel}  (GEMV M=1 K={k} N={n})  grid {groups:?}\n\
         {SUBMITS} submits — under MESA_VK_TRACE_PER_SUBMIT take the LAST .rgp."
    );
    for i in 0..SUBMITS {
        ctx.dispatch_blocking(
            &kern,
            vec![
                WriteDescriptorSet::buffer(0, w.clone()),
                WriteDescriptorSet::buffer(1, x.clone()),
                WriteDescriptorSet::buffer(2, y.clone()),
            ],
            None::<u32>,
            groups,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        eprintln!("  submit {i}/{SUBMITS} done");
    }
    Ok(())
}

/// Phase-0 MALL probe: stream-read a `SG_PROBE_MB`-sized buffer `SG_PROBE_REPS`
/// times (push = {elems, reps}). Run sub- (e.g. 8) vs super-MALL (e.g. 64) and
/// compare RGP duration + "local video memory bytes": equal logical reads, so a
/// duration gap = MALL bandwidth vs DRAM, and whether local-video tracks the
/// MALL-resident case tells us if it counts Infinity Cache hits (STATUS).
fn run_probe() -> anyhow::Result<()> {
    let mb: usize = std::env::var("SG_PROBE_MB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let reps: u32 = std::env::var("SG_PROBE_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let elems = mb * 1024 * 1024 / 4; // u32 words
    let total_threads = 1024u32 * 256; // 1024 groups × WG 256; stride = full sweep

    let ctx = GpuContext::new().map_err(|e| anyhow::anyhow!("gpu: {e}"))?;
    let kern = ctx.load_kernel("mall_probe").map_err(|e| anyhow::anyhow!("{e}"))?;
    let u = BufferUsage::STORAGE_BUFFER;
    let nb_err = |e: sg_gpu::GpuError| anyhow::anyhow!("{e}");
    let src = ctx.new_buffer::<u32>(elems as u64, u).map_err(nb_err)?;
    let dst = ctx
        .new_buffer::<u32>(total_threads as u64, u)
        .map_err(nb_err)?;

    let logical_gib = (elems as f64 * 4.0 * reps as f64) / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "RGP MALL probe: {mb} MiB buffer ({} {}MALL) × {reps} reps = {logical_gib:.2} GiB \
         logical reads\n  grid [1024,1,1] WG 256; {SUBMITS} submits — take the LAST .rgp; \
         read duration + local-video bytes.",
        if mb <= 32 { "≤32, fits" } else { ">32, exceeds" },
        if mb <= 32 { "" } else { "super-" },
    );
    let push = [elems as u32, reps];
    for i in 0..SUBMITS {
        ctx.dispatch_blocking(
            &kern,
            vec![
                WriteDescriptorSet::buffer(0, src.clone()),
                WriteDescriptorSet::buffer(1, dst.clone()),
            ],
            Some(push),
            [1024, 1, 1],
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        eprintln!("  submit {i}/{SUBMITS} done");
    }
    Ok(())
}

pub fn run(kernel: &str) -> anyhow::Result<()> {
    if kernel == "mall_probe" {
        return run_probe();
    }
    if let Some((k, n)) = gemv_shape(kernel) {
        return run_gemv(kernel, k, n);
    }
    let m = rgp_m();
    let (k, n, nb, mb, int8, swz) = shape(kernel).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown rgp kernel `{kernel}`; the deployed int8 FFN gemms \
             (gemm_q4_0_i8_swz_m4n1_k{{21504_n5376,5376_n21504}}), the int8 2×2 \
             baseline (…_i8_t22_…), or any f16 prefill site (gemm_q4_0_k<K>_n<N>) \
             — see `shape()`"
        )
    })?;

    let ctx = GpuContext::new().map_err(|e| anyhow::anyhow!("gpu: {e}"))?;
    let kern = ctx
        .load_kernel(kernel)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let u = BufferUsage::STORAGE_BUFFER;
    let nb_err = |e: sg_gpu::GpuError| anyhow::anyhow!("{e}");

    // Q4_0 weights (u32 words): N·K/32 blocks × 18 bytes. Dummy-filled.
    let w = ctx
        .new_buffer::<u32>((n * k / 32 * 18 / 4) as u64, u)
        .map_err(nb_err)?;
    let y = ctx.new_buffer::<u16>((m * n) as u64, u).map_err(nb_err)?;
    // int8 path: x = Q8 quants (u32-packed = i8 bytes), x_scales = f16. f16 path:
    // x = f16 activations.
    let x_i8 = ctx
        .new_buffer::<u32>((m * k / 4) as u64, u)
        .map_err(nb_err)?;
    let xs = ctx
        .new_buffer::<u16>((m * k / 32) as u64, u)
        .map_err(nb_err)?;
    let x_f16 = ctx.new_buffer::<u16>((m * k) as u64, u).map_err(nb_err)?;

    // Swizzled kernels expect [M-blocks, N-blocks]; the rest [N-blocks, M-blocks].
    // The l2 kernel is 1D (one workgroup per output tile, tile_index decode) — its
    // bench/winning config, so capture it that way.
    let groups = if kernel.contains("_l2") {
        [(n as u32 / nb) * (m as u32 / mb), 1, 1]
    } else if swz {
        [m as u32 / mb, n as u32 / nb, 1]
    } else {
        [n as u32 / nb, m as u32 / mb, 1]
    };
    eprintln!(
        "RGP capture target: {kernel}  (M={m} K={k} N={n})  grid {groups:?}\n\
         {SUBMITS} submits — under MESA_VK_TRACE_PER_SUBMIT take the LAST .rgp."
    );
    for i in 0..SUBMITS {
        let writes = if int8 {
            vec![
                WriteDescriptorSet::buffer(0, w.clone()),
                WriteDescriptorSet::buffer(1, x_i8.clone()),
                WriteDescriptorSet::buffer(2, xs.clone()),
                WriteDescriptorSet::buffer(3, y.clone()),
            ]
        } else {
            vec![
                WriteDescriptorSet::buffer(0, w.clone()),
                WriteDescriptorSet::buffer(1, x_f16.clone()),
                WriteDescriptorSet::buffer(2, y.clone()),
            ]
        };
        ctx.dispatch_blocking(&kern, writes, None::<u32>, groups)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        eprintln!("  submit {i}/{SUBMITS} done");
    }
    Ok(())
}
