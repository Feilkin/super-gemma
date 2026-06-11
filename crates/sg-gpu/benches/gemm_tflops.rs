//! gemm_q4_0 (cooperative matrix) compute-rate microbench (plan 02: ≥ 30 %
//! of the ~59 TFLOPS f16 peak initially, stretch ≥ 50 %). Skips without a
//! coopmat GPU.

use criterion::{Criterion, criterion_group, criterion_main};
use sg_gpu::GpuContext;
use vulkano::buffer::BufferUsage;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::sync::GpuFuture;

const M: usize = 512; // prefill chunk
const K: usize = 5376;
const N: usize = 21504; // ffn gate/up — the prefill flop bucket
const DISPATCHES: usize = 4;
// Must match the gemm variant defines (M_TILES=2, N_TILES=4).
const M_BLOCK: usize = 32;
const N_BLOCK: usize = 64;

fn bench(c: &mut Criterion) {
    let ctx = match GpuContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("skipping gemm bench: no usable GPU ({e})");
            return;
        }
    };
    let kernels = [
        ("gemm_st_q4_0_k5376_n21504", 64usize, 64usize),
        ("gemm_q4_0_k5376_n21504", M_BLOCK, N_BLOCK),
    ];

    let weight_words = N * K / 32 * 18 / 4;
    let w_buf = ctx
        .buffer_from_iter(
            (0..weight_words as u32).map(|i| i.wrapping_mul(0x9E37_79B9)),
            BufferUsage::STORAGE_BUFFER,
        )
        .expect("weights");
    let x_buf = ctx
        .buffer_from_iter(
            (0..M * K).map(|i| half::f16::from_f32((i % 11) as f32 * 0.05).to_bits()),
            BufferUsage::STORAGE_BUFFER,
        )
        .expect("x");
    let y_buf = ctx
        .new_buffer::<u16>((M * N) as u64, BufferUsage::STORAGE_BUFFER)
        .expect("y");

    let flops = 2.0 * M as f64 * N as f64 * K as f64 * DISPATCHES as f64;
    let mut group = c.benchmark_group("gemm_q4_0");
    for (name, m_block, n_block) in kernels {
        if name.starts_with("gemm_q4_0") && !ctx.cooperative_matrix {
            continue;
        }
        let kernel = ctx.load_kernel(name).expect(name);
        let layout = kernel.layout().clone();
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator().clone(),
            layout.set_layouts()[0].clone(),
            vec![
                WriteDescriptorSet::buffer(0, w_buf.clone()),
                WriteDescriptorSet::buffer(1, x_buf.clone()),
                WriteDescriptorSet::buffer(2, y_buf.clone()),
            ],
            [],
        )
        .expect("set");

        group.bench_function(format!("{name}_m{M}"), |b| {
            b.iter_custom(|iters| {
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    let mut builder = AutoCommandBufferBuilder::primary(
                        ctx.command_buffer_allocator().clone(),
                        ctx.queue().queue_family_index(),
                        CommandBufferUsage::OneTimeSubmit,
                    )
                    .unwrap();
                    builder
                        .bind_pipeline_compute(kernel.pipeline().clone())
                        .unwrap()
                        .bind_descriptor_sets(
                            PipelineBindPoint::Compute,
                            layout.clone(),
                            0,
                            set.clone(),
                        )
                        .unwrap();
                    for _ in 0..DISPATCHES {
                        // SAFETY: tile grid covering M×N, the kernel's contract.
                        unsafe {
                            builder.dispatch([(N / n_block) as u32, (M / m_block) as u32, 1])
                        }
                        .unwrap();
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
                }
                let elapsed = start.elapsed();
                let tflops = flops * iters as f64 / elapsed.as_secs_f64() / 1e12;
                eprintln!("  -> {tflops:.2} TFLOPS (f16 peak ≈ 59)");
                elapsed
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
