//! gemv_q4_0 streaming-bandwidth microbench (plan 02: ≥ 85 % of the
//! bandwidth ceiling; decode speed is this number). Skips without a GPU.
//!
//! The weight matrix is sized at ~260 MB so Strix Halo's 32 MB MALL can't
//! serve repeat iterations from cache; each iteration submits one command
//! buffer with several dispatches to amortize submit/fence overhead.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use sg_gpu::GpuContext;
use vulkano::buffer::BufferUsage;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::sync::GpuFuture;

const K: usize = 5376;
const N: usize = 4 * 21504; // ~260 MB of Q4_0 — larger than the MALL
const DISPATCHES: usize = 4;

fn bench(c: &mut Criterion) {
    let ctx = match GpuContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("skipping gemv bench: no usable GPU ({e})");
            return;
        }
    };
    let kernel = ctx.load_kernel("gemv_q4_0_k5376").expect("kernel");

    let row_words = K / 64 * 9;
    let weight_words = N * row_words;
    // Pseudo-random-ish payload; values don't affect timing.
    let w_buf = ctx
        .buffer_from_iter(
            (0..weight_words as u32).map(|i| i.wrapping_mul(0x9E37_79B9)),
            BufferUsage::STORAGE_BUFFER,
        )
        .expect("weights");
    let x_buf = ctx
        .buffer_from_iter(
            (0..K).map(|i| half::f16::from_f32((i % 7) as f32 * 0.1).to_bits()),
            BufferUsage::STORAGE_BUFFER,
        )
        .expect("x");
    let y_buf = ctx
        .new_buffer::<u16>(N as u64, BufferUsage::STORAGE_BUFFER)
        .expect("y");

    let layout = kernel.layout().clone();
    let set = DescriptorSet::new(
        ctx.descriptor_set_allocator().clone(),
        layout.set_layouts()[0].clone(),
        vec![
            WriteDescriptorSet::buffer(0, w_buf),
            WriteDescriptorSet::buffer(1, x_buf),
            WriteDescriptorSet::buffer(2, y_buf),
        ],
        [],
    )
    .expect("set");

    let weight_bytes = (weight_words * 4) as u64;
    let mut group = c.benchmark_group("gemv_q4_0");
    group.throughput(Throughput::Bytes(weight_bytes * DISPATCHES as u64));
    group.bench_function(format!("k{K}_n{N}"), |b| {
        b.iter(|| {
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
                // SAFETY: N workgroups over N output rows, the kernel's
                // contract; vulkano inserts the write-write barriers.
                unsafe { builder.dispatch([N as u32, 1, 1]) }.unwrap();
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
        })
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
