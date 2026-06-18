//! int8-MMQ down-gemm steady-state A/B — the TRUSTWORTHY harness for the small
//! prefetch deltas. `mmq_tflops` has no clock warm-up, and its prefetch-variant
//! numbers swung 1.5–5.4% run-to-run (2026-06-18) — unusable for a few-percent
//! comparison. This (a) warms to the boost clock first, then (b) times the
//! variants ROUND-ROBIN within each batch, so any residual drift hits every
//! variant equally and the pairwise Δ survives it. Reports per-variant median
//! TFLOPS + CV; if the CV bands overlap the Δ is a wash.
//! Run: `cargo bench -p sg-gpu --bench mmq_variance`. Pin perf=high first.

use std::time::{Duration, Instant};

use sg_gpu::{GpuContext, Kernel};
use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::sync::GpuFuture;

const M: usize = 256; // prefill chunk
const K: usize = 21504; // FFN-down K
const N: usize = 5376;
const N_BLOCK: u32 = 16;
const DISPATCHES: usize = 8; // per timed batch
const BATCHES: usize = 50;
const WARMUP: Duration = Duration::from_secs(12);

/// Down-gemm variants under test — same shape/bindings, so the only difference
/// is the kernel body + its tile decomposition (`m_block` = M-rows per
/// workgroup). `deployed` carries the banked deep-prefetch (was +6.1% over the
/// pre-prefetch kernel here, 2026-06-18); `s1` (single-buffered scale) is the
/// standing occupancy A/B baseline (−2.1%); `occ` is the max-occupancy 1×1
/// rewrite (16-row blocks → many waves/SIMD; the occupancy-vs-reuse test).
const VARIANTS: &[(&str, &str, u32)] = &[
    ("deployed (s2+pf)", "gemm_q4_0_i8_swz_m4n1_k21504_n5376", 64),
    ("s1", "gemm_q4_0_i8_swz_m4n1_s1_k21504_n5376", 64),
    ("occ 1×1", "gemm_q4_0_i8_occ_k21504_n5376", 16),
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

    // int8 operands: Q4_0 weights (u32 words), Q8 activations (i8 packed u32),
    // f16 scales, f16 out. Dummy-filled — steady-state timing is data-independent.
    let w: Subbuffer<[u32]> = ctx
        .buffer_from_iter(
            (0..(N * K / 32 * 18 / 4) as u32).map(|i| i.wrapping_mul(0x9E37_79B9)),
            BufferUsage::STORAGE_BUFFER,
        )
        .unwrap();
    let x = ctx
        .new_buffer::<u32>((M * K / 4) as u64, BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let xs = ctx
        .new_buffer::<u16>((M * K / 32) as u64, BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let y = ctx
        .new_buffer::<u16>((M * N) as u64, BufferUsage::STORAGE_BUFFER)
        .unwrap();

    let kernels: Vec<Kernel> = VARIANTS
        .iter()
        .map(|(_, n, _)| ctx.load_kernel(n).expect(n))
        .collect();
    // Per-variant swizzled [M-blocks, N-blocks] grid (m_block differs: 64 for
    // the 4×1 tile, 16 for the 1×1 occ kernel — both cover the full M×N).
    let grids: Vec<[u32; 3]> = VARIANTS
        .iter()
        .map(|(_, _, mb)| [M as u32 / mb, N as u32 / N_BLOCK, 1])
        .collect();
    let sets: Vec<_> = kernels
        .iter()
        .map(|k| {
            DescriptorSet::new(
                ctx.descriptor_set_allocator().clone(),
                k.layout().set_layouts()[0].clone(),
                vec![
                    WriteDescriptorSet::buffer(0, w.clone()),
                    WriteDescriptorSet::buffer(1, x.clone()),
                    WriteDescriptorSet::buffer(2, xs.clone()),
                    WriteDescriptorSet::buffer(3, y.clone()),
                ],
                [],
            )
            .unwrap()
        })
        .collect();

    let run = |k: &Kernel, set: &std::sync::Arc<DescriptorSet>, grid: [u32; 3]| {
        let mut b = AutoCommandBufferBuilder::primary(
            ctx.command_buffer_allocator().clone(),
            ctx.queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        b.bind_pipeline_compute(k.pipeline().clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                k.layout().clone(),
                0,
                set.clone(),
            )
            .unwrap();
        for _ in 0..DISPATCHES {
            // SAFETY: swizzled [M-blocks, N-blocks] grid, the kernel's contract.
            unsafe { b.dispatch(grid) }.unwrap();
        }
        b.build()
            .unwrap()
            .execute(ctx.queue().clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    };

    // Warm up to the boost clock (round-robin so no variant is favoured).
    let t0 = Instant::now();
    while t0.elapsed() < WARMUP {
        for ((k, s), &g) in kernels.iter().zip(&sets).zip(&grids) {
            run(k, s, g);
        }
    }
    eprintln!("warmed up, sclk={}", sclk_mhz());

    let flops = 2.0 * M as f64 * N as f64 * K as f64 * DISPATCHES as f64;
    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(BATCHES); VARIANTS.len()];
    for _ in 0..BATCHES {
        for (i, ((k, s), &g)) in kernels.iter().zip(&sets).zip(&grids).enumerate() {
            let start = Instant::now();
            run(k, s, g);
            samples[i].push(flops / start.elapsed().as_secs_f64() / 1e12);
        }
    }

    eprintln!(
        "int8 down-gemm steady-state ({BATCHES} interleaved batches × {DISPATCHES} dispatches), \
         sclk={}:",
        sclk_mhz()
    );
    let median = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let base = median(&mut samples[0].clone());
    for (i, (label, _, _)) in VARIANTS.iter().enumerate() {
        let mut s = samples[i].clone();
        let med = median(&mut s);
        let mean = s.iter().sum::<f64>() / s.len() as f64;
        let sd = (s.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / s.len() as f64).sqrt();
        eprintln!(
            "  {label:18} median {med:6.2} TFLOPS  cv {:.2}%  Δ vs deployed {:+.1}%",
            100.0 * sd / mean,
            100.0 * (med - base) / base,
        );
    }
}
