//! f16 coopmat GEMM steady-state variance probe (NOT criterion). Warms the
//! GPU to its boost clock FIRST (the ~8 s sclk ramp otherwise contaminates
//! short measurements), then times many identical fixed-size batches and
//! reports the distribution. If steady-state spread is more than a few %, the
//! measurement methodology is unsound and no TFLOPS comparison can be trusted.
//! Run: `cargo bench -p sg-gpu --bench gemm_variance`.

use std::time::{Duration, Instant};

use sg_gpu::GpuContext;
use vulkano::buffer::BufferUsage;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::sync::GpuFuture;

const M: usize = 512;
const K: usize = 5376;
const N: usize = 21504;
const DISPATCHES: usize = 8; // per timed batch
const BATCHES: usize = 40;
const WARMUP: Duration = Duration::from_secs(12);

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
    let kernel = ctx.load_kernel("gemm_q4_0_k5376_n21504").expect("kernel");
    let layout = kernel.layout().clone();

    let weight_words = N * K / 32 * 18 / 4;
    let w = ctx
        .buffer_from_iter(
            (0..weight_words as u32).map(|i| i.wrapping_mul(0x9E37_79B9)),
            BufferUsage::STORAGE_BUFFER,
        )
        .unwrap();
    let x = ctx
        .buffer_from_iter(
            (0..M * K).map(|i| half::f16::from_f32((i % 11) as f32 * 0.05).to_bits()),
            BufferUsage::STORAGE_BUFFER,
        )
        .unwrap();
    let y = ctx
        .new_buffer::<u16>((M * N) as u64, BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let set = DescriptorSet::new(
        ctx.descriptor_set_allocator().clone(),
        layout.set_layouts()[0].clone(),
        vec![
            WriteDescriptorSet::buffer(0, w.clone()),
            WriteDescriptorSet::buffer(1, x.clone()),
            WriteDescriptorSet::buffer(2, y.clone()),
        ],
        [],
    )
    .unwrap();

    let run_batch = || {
        let mut builder = AutoCommandBufferBuilder::primary(
            ctx.command_buffer_allocator().clone(),
            ctx.queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder
            .bind_pipeline_compute(kernel.pipeline().clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, layout.clone(), 0, set.clone())
            .unwrap();
        for _ in 0..DISPATCHES {
            // SAFETY: tile grid covering M×N, the kernel's contract.
            unsafe { builder.dispatch([(N / 64) as u32, (M / 64) as u32, 1]) }.unwrap();
        }
        builder
            .build()
            .unwrap()
            .execute(ctx.queue().clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    };

    // Warm up to the boost clock.
    let t0 = Instant::now();
    while t0.elapsed() < WARMUP {
        run_batch();
    }
    eprintln!("warmed up, sclk={}", sclk_mhz());

    let flops = 2.0 * M as f64 * N as f64 * K as f64 * DISPATCHES as f64;
    let mut samples = Vec::with_capacity(BATCHES);
    for _ in 0..BATCHES {
        let start = Instant::now();
        run_batch();
        let tflops = flops / start.elapsed().as_secs_f64() / 1e12;
        samples.push(tflops);
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    let min = samples[0];
    let max = samples[n - 1];
    let median = samples[n / 2];
    let mean = samples.iter().sum::<f64>() / n as f64;
    let sd = (samples.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    eprintln!(
        "f16 gemm steady-state ({BATCHES} batches × {DISPATCHES} dispatches), sclk={}:",
        sclk_mhz()
    );
    eprintln!("  min {min:.2}  median {median:.2}  mean {mean:.2}  max {max:.2}  TFLOPS",);
    eprintln!(
        "  sd {sd:.3}  spread {:.1}%  (cv {:.1}%)",
        100.0 * (max - min) / median,
        100.0 * sd / mean,
    );
}
