//! Split-M dispatch A/B: bb_m4 (one dispatch, [NB_M, NB_N] grid, ~400 waves) vs
//! bb_m4_split (NB_M=4 dispatches of [NB_N,1,1], ≤336 waves each so one M-block's
//! 1.31 MiB X-slab stays L2-resident). Tests whether capping concurrency at the
//! L2-friendly window makes the GPU work faster.
//!
//! Measurement is GPU-timestamp per dispatch. The split kernel's coopStore/coopLoad
//! accesses are invisible to vulkano's auto-sync, so the 4 dispatches can't be
//! serialized cheaply in one command buffer (no anchor for a barrier) — they run as
//! separate fenced submits. We therefore SUM the four dispatches' on-GPU durations,
//! which excludes the CPU fence gaps between submits and isolates the L2 effect from
//! the (unavoidable, for now) serialization overhead. The wall-clock total — gaps
//! included — is reported separately as the production-reality figure.

use sg_gpu::{BufferUsage, GpuContext, BufferBinding};

const K: usize = 21504;
const N: usize = 5376;
const M: usize = 256;
const WARMUP: usize = 30;
const BATCHES: usize = 50;

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

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn cv(v: &[f64]) -> f64 {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    100.0 * sd / mean
}

pub fn run() -> anyhow::Result<()> {
    let ctx = GpuContext::new().map_err(|e| anyhow::anyhow!("{e}"))?;
    if !ctx.cooperative_matrix {
        anyhow::bail!("no VK_KHR_cooperative_matrix");
    }
    let u = BufferUsage::STORAGE_BUFFER;
    let w = ctx
        .buffer_from_iter(
            (0..(N * K / 32 * 18 / 4) as u32).map(|i| i.wrapping_mul(0x9E37_79B9)),
            u,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let x = ctx
        .new_buffer::<u32>((M * K / 4) as u64, u)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let xs = ctx
        .new_buffer::<u16>((M * K / 32) as u64, u)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let y = ctx
        .new_buffer::<u16>((M * N) as u64, u)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let writes = || {
        vec![
            BufferBinding::buffer(0, w.clone()),
            BufferBinding::buffer(1, x.clone()),
            BufferBinding::buffer(2, xs.clone()),
            BufferBinding::buffer(3, y.clone()),
        ]
    };

    let bb = ctx
        .load_kernel("gemm_q4_0_i8_bb_m4")
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let split = ctx
        .load_kernel("gemm_q4_0_i8_bb_m4_split")
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let nb_n = (N / 16) as u32; // 336 N-strips
    let nb_m = (M / 64) as u32; // 4 M-blocks
    let flops = 2.0 * M as f64 * N as f64 * K as f64;
    let timer = ctx.new_timer(2).map_err(|e| anyhow::anyhow!("{e}"))?;

    // bb_m4: one dispatch. split: NB_M dispatches in ONE graph — the visibility
    // shim makes auto-sync insert a real compute→compute barrier between them, so
    // they serialize GPU-side in a single submit (no CPU fence gaps).
    let g_bb = ctx
        .record_graph_with_marks(&timer, |r| {
            r.dispatch(&bb, writes(), None::<u32>, [nb_m, nb_n, 1])?;
            r.mark("end")?;
            Ok(())
        })
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .0;
    let g_split = ctx
        .record_graph_with_marks(&timer, |r| {
            for mb in 0..nb_m {
                r.dispatch(&split, writes(), Some(mb), [nb_n, 1, 1])?;
            }
            r.mark("end")?;
            Ok(())
        })
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .0;

    // Capture mode: submit the split graph a few times (for MESA_VK_TRACE=rgp
    // PER_SUBMIT — take the last .rgp) so the 4-dispatch barriered graph can be
    // inspected: are the M-block dispatches serialized (barrier present), and what
    // is their wave/L2 behaviour? `SG_SPLIT_CAPTURE=bb` captures bb_m4 instead.
    if let Ok(which) = std::env::var("SG_SPLIT_CAPTURE") {
        let g = if which == "bb" { &g_bb } else { &g_split };
        for _ in 0..8 {
            ctx.submit_blocking(g).map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        eprintln!("captured {} (8 submits)", if which == "bb" { "bb_m4" } else { "split" });
        return Ok(());
    }

    // GPU ns of a single recorded graph (BottomOfPipe ts[1] − ts[0]).
    let gpu_ns = |g: &_| -> anyhow::Result<f64> {
        ctx.submit_blocking(g).map_err(|e| anyhow::anyhow!("{e}"))?;
        let ts = timer.read_ns().map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(ts[1] - ts[0])
    };

    for _ in 0..WARMUP {
        gpu_ns(&g_bb)?;
        gpu_ns(&g_split)?;
    }
    eprintln!("warmed up, sclk={}", sclk_mhz());

    let mut bb_t: Vec<f64> = Vec::with_capacity(BATCHES);
    let mut sp_t: Vec<f64> = Vec::with_capacity(BATCHES);
    for _ in 0..BATCHES {
        bb_t.push(flops / (gpu_ns(&g_bb)? / 1e9) / 1e12);
        sp_t.push(flops / (gpu_ns(&g_split)? / 1e9) / 1e12);
    }

    let bb_cv = cv(&bb_t);
    let sp_cv = cv(&sp_t);
    let bb_med = median(&mut bb_t);
    let sp_med = median(&mut sp_t);
    eprintln!(
        "split-M A/B (GPU-timestamp, K={K} N={N} M={M}, {BATCHES} batches), sclk={}:",
        sclk_mhz()
    );
    eprintln!("  bb_m4       (1 dispatch, ~400 waves)   {bb_med:6.2} TFLOPS cv {bb_cv:.2}%");
    eprintln!(
        "  bb_m4_split (4 barriered, ≤336 waves)  {sp_med:6.2} TFLOPS cv {sp_cv:.2}%  Δ{:+5.1}%",
        100.0 * (sp_med - bb_med) / bb_med
    );
    Ok(())
}
