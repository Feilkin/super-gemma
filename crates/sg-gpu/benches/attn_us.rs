//! Attention kernel timings in µs per dispatch (plan 06: attention benched
//! at ctx ∈ {1K, 8K, 32K}). Skips without a GPU.
//!
//! decode_global runs its full split-K pipeline (partial + reduce) per
//! dispatch, swept over split counts to inform the engine's n_splits
//! policy.

use criterion::{Criterion, criterion_group, criterion_main};
use sg_gpu::{GpuContext, StepState};
use vulkano::buffer::{BufferContents, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::sync::GpuFuture;

const N_Q_HEADS: usize = 32;
const SL_KV_HEADS: usize = 16;
const SL_DIM: usize = 256;
const SL_SCALE: f32 = 0.0625;
const GL_KV_HEADS: usize = 4;
const GL_DIM: usize = 512;
const GL_SCALE: f32 = 0.044194174;
const SL_PART_STRIDE: usize = SL_DIM + 2;
const GL_PART_STRIDE: usize = GL_DIM + 2;
const DISPATCHES: usize = 8;

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct PushSplitScale {
    n_splits: u32,
    scale: f32,
}

fn f16_fill(n: usize) -> impl ExactSizeIterator<Item = u16> {
    (0..n).map(|i| half::f16::from_f32((i % 23) as f32 * 0.1 - 1.1).to_bits())
}

/// Step buffer holding the per-step dynamic state.
fn step_buf(ctx: &GpuContext, state: StepState) -> Subbuffer<[u32]> {
    let buf = ctx.new_step_buffer().unwrap();
    state.write_to(&buf).unwrap();
    buf
}

fn bench(c: &mut Criterion) {
    let ctx = match GpuContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("skipping attn bench: no usable GPU ({e})");
            return;
        }
    };
    let mut group = c.benchmark_group("attn");

    // --- decode_sliding: split-K pipeline, full ring (1024 = the window). ---
    {
        let part_k = ctx.load_kernel("attn_decode_sliding").expect("kernel");
        let red_k = ctx.load_kernel("attn_reduce_d256").expect("kernel");
        let kv_len = 1024usize;
        let q = ctx
            .buffer_from_iter(f16_fill(N_Q_HEADS * SL_DIM), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let k = ctx
            .buffer_from_iter(
                f16_fill(kv_len * SL_KV_HEADS * SL_DIM),
                BufferUsage::STORAGE_BUFFER,
            )
            .unwrap();
        let v = ctx
            .buffer_from_iter(
                f16_fill(kv_len * SL_KV_HEADS * SL_DIM),
                BufferUsage::STORAGE_BUFFER,
            )
            .unwrap();
        let out = ctx
            .new_buffer::<u16>((N_Q_HEADS * SL_DIM) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        for n_splits in [1u32, 8, 16] {
            let part = ctx
                .new_buffer::<f32>(
                    (N_Q_HEADS * n_splits as usize * SL_PART_STRIDE) as u64,
                    BufferUsage::STORAGE_BUFFER,
                )
                .unwrap();
            let p_layout = part_k.layout().clone();
            let p_set = DescriptorSet::new(
                ctx.descriptor_set_allocator().clone(),
                p_layout.set_layouts()[0].clone(),
                vec![
                    WriteDescriptorSet::buffer(0, q.clone()),
                    WriteDescriptorSet::buffer(1, k.clone()),
                    WriteDescriptorSet::buffer(2, v.clone()),
                    WriteDescriptorSet::buffer(3, part.clone()),
                    WriteDescriptorSet::buffer(
                        4,
                        step_buf(
                            &ctx,
                            StepState {
                                kv_len_sliding: kv_len as u32,
                                ..Default::default()
                            },
                        ),
                    ),
                ],
                [],
            )
            .unwrap();
            let r_layout = red_k.layout().clone();
            let r_set = DescriptorSet::new(
                ctx.descriptor_set_allocator().clone(),
                r_layout.set_layouts()[0].clone(),
                vec![
                    WriteDescriptorSet::buffer(0, part.clone()),
                    WriteDescriptorSet::buffer(1, out.clone()),
                ],
                [],
            )
            .unwrap();

            group.bench_function(format!("decode_sliding_ring1024_splits{n_splits}"), |b| {
                b.iter_custom(|iters| {
                    let start = std::time::Instant::now();
                    for _ in 0..iters {
                        let mut builder = AutoCommandBufferBuilder::primary(
                            ctx.command_buffer_allocator().clone(),
                            ctx.queue().queue_family_index(),
                            CommandBufferUsage::OneTimeSubmit,
                        )
                        .unwrap();
                        for _ in 0..DISPATCHES {
                            builder
                                .bind_pipeline_compute(part_k.pipeline().clone())
                                .unwrap()
                                .bind_descriptor_sets(
                                    PipelineBindPoint::Compute,
                                    p_layout.clone(),
                                    0,
                                    p_set.clone(),
                                )
                                .unwrap()
                                .push_constants(
                                    p_layout.clone(),
                                    0,
                                    PushSplitScale {
                                        n_splits,
                                        scale: SL_SCALE,
                                    },
                                )
                                .unwrap();
                            // SAFETY: [kv_heads, splits] grid, the kernel's contract.
                            unsafe { builder.dispatch([SL_KV_HEADS as u32, n_splits, 1]) }.unwrap();
                            builder
                                .bind_pipeline_compute(red_k.pipeline().clone())
                                .unwrap()
                                .bind_descriptor_sets(
                                    PipelineBindPoint::Compute,
                                    r_layout.clone(),
                                    0,
                                    r_set.clone(),
                                )
                                .unwrap()
                                .push_constants(r_layout.clone(), 0, n_splits)
                                .unwrap();
                            // SAFETY: one workgroup per query head.
                            unsafe { builder.dispatch([N_Q_HEADS as u32, 1, 1]) }.unwrap();
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
                    let us = elapsed.as_secs_f64() * 1e6 / (iters as usize * DISPATCHES) as f64;
                    eprintln!("  -> {us:.1} µs/step (partial+reduce)");
                    elapsed
                })
            });
        }
    }

    // --- decode_global: split-K pipeline at ctx × n_splits. ---
    {
        let part_k = ctx.load_kernel("attn_decode_global").expect("kernel");
        let red_k = ctx.load_kernel("attn_reduce_d512").expect("kernel");
        for (kv_len, splits) in [
            (1024usize, vec![1u32, 8]),
            (8192, vec![1, 8]),
            (32768, vec![1, 8, 32]),
        ] {
            let q = ctx
                .buffer_from_iter(f16_fill(N_Q_HEADS * GL_DIM), BufferUsage::STORAGE_BUFFER)
                .unwrap();
            let kv = ctx
                .buffer_from_iter(
                    f16_fill(kv_len * GL_KV_HEADS * GL_DIM),
                    BufferUsage::STORAGE_BUFFER,
                )
                .unwrap();
            let out = ctx
                .new_buffer::<u16>((N_Q_HEADS * GL_DIM) as u64, BufferUsage::STORAGE_BUFFER)
                .unwrap();
            for &n_splits in &splits {
                let part = ctx
                    .new_buffer::<f32>(
                        (N_Q_HEADS * n_splits as usize * GL_PART_STRIDE) as u64,
                        BufferUsage::STORAGE_BUFFER,
                    )
                    .unwrap();
                let p_layout = part_k.layout().clone();
                let p_set = DescriptorSet::new(
                    ctx.descriptor_set_allocator().clone(),
                    p_layout.set_layouts()[0].clone(),
                    vec![
                        WriteDescriptorSet::buffer(0, q.clone()),
                        WriteDescriptorSet::buffer(1, kv.clone()),
                        WriteDescriptorSet::buffer(2, part.clone()),
                        WriteDescriptorSet::buffer(
                            3,
                            step_buf(
                                &ctx,
                                StepState {
                                    kv_len_global: kv_len as u32,
                                    ..Default::default()
                                },
                            ),
                        ),
                    ],
                    [],
                )
                .unwrap();
                let r_layout = red_k.layout().clone();
                let r_set = DescriptorSet::new(
                    ctx.descriptor_set_allocator().clone(),
                    r_layout.set_layouts()[0].clone(),
                    vec![
                        WriteDescriptorSet::buffer(0, part.clone()),
                        WriteDescriptorSet::buffer(1, out.clone()),
                    ],
                    [],
                )
                .unwrap();

                group.bench_function(format!("decode_global_ctx{kv_len}_splits{n_splits}"), |b| {
                    b.iter_custom(|iters| {
                        let start = std::time::Instant::now();
                        for _ in 0..iters {
                            let mut builder = AutoCommandBufferBuilder::primary(
                                ctx.command_buffer_allocator().clone(),
                                ctx.queue().queue_family_index(),
                                CommandBufferUsage::OneTimeSubmit,
                            )
                            .unwrap();
                            for _ in 0..DISPATCHES {
                                builder
                                    .bind_pipeline_compute(part_k.pipeline().clone())
                                    .unwrap()
                                    .bind_descriptor_sets(
                                        PipelineBindPoint::Compute,
                                        p_layout.clone(),
                                        0,
                                        p_set.clone(),
                                    )
                                    .unwrap()
                                    .push_constants(
                                        p_layout.clone(),
                                        0,
                                        PushSplitScale {
                                            n_splits,
                                            scale: GL_SCALE,
                                        },
                                    )
                                    .unwrap();
                                // SAFETY: [kv_heads, splits] grid, the kernel's contract.
                                unsafe { builder.dispatch([GL_KV_HEADS as u32, n_splits, 1]) }
                                    .unwrap();
                                builder
                                    .bind_pipeline_compute(red_k.pipeline().clone())
                                    .unwrap()
                                    .bind_descriptor_sets(
                                        PipelineBindPoint::Compute,
                                        r_layout.clone(),
                                        0,
                                        r_set.clone(),
                                    )
                                    .unwrap()
                                    .push_constants(r_layout.clone(), 0, n_splits)
                                    .unwrap();
                                // SAFETY: one workgroup per query head.
                                unsafe { builder.dispatch([N_Q_HEADS as u32, 1, 1]) }.unwrap();
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
                        let us = elapsed.as_secs_f64() * 1e6 / (iters as usize * DISPATCHES) as f64;
                        eprintln!("  -> {us:.1} µs/step (partial+reduce)");
                        elapsed
                    })
                });
            }
        }
    }

    // --- prefill pair: one chunk of M tokens. ---
    {
        let m = 256usize;
        // Sliding: history deep enough that every query has a full window.
        let kernel = ctx.load_kernel("attn_prefill_sliding").expect("kernel");
        let q0 = 4096usize;
        let l = q0 + m;
        let q = ctx
            .buffer_from_iter(
                f16_fill(m * N_Q_HEADS * SL_DIM),
                BufferUsage::STORAGE_BUFFER,
            )
            .unwrap();
        let k = ctx
            .buffer_from_iter(
                f16_fill(l * SL_KV_HEADS * SL_DIM),
                BufferUsage::STORAGE_BUFFER,
            )
            .unwrap();
        let v = ctx
            .buffer_from_iter(
                f16_fill(l * SL_KV_HEADS * SL_DIM),
                BufferUsage::STORAGE_BUFFER,
            )
            .unwrap();
        let out = ctx
            .new_buffer::<u16>((m * N_Q_HEADS * SL_DIM) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let layout = kernel.layout().clone();
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator().clone(),
            layout.set_layouts()[0].clone(),
            vec![
                WriteDescriptorSet::buffer(0, q),
                WriteDescriptorSet::buffer(1, k),
                WriteDescriptorSet::buffer(2, v),
                WriteDescriptorSet::buffer(3, out),
                WriteDescriptorSet::buffer(
                    4,
                    step_buf(
                        &ctx,
                        StepState {
                            q0: q0 as u32,
                            ..Default::default()
                        },
                    ),
                ),
            ],
            [],
        )
        .unwrap();
        group.bench_function(format!("prefill_sliding_m{m}"), |b| {
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
                        .unwrap()
                        .push_constants(layout.clone(), 0, SL_SCALE)
                        .unwrap();
                    for _ in 0..DISPATCHES {
                        // SAFETY: [kv_heads, M] grid, the kernel's contract.
                        unsafe { builder.dispatch([SL_KV_HEADS as u32, m as u32, 1]) }.unwrap();
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
                let us = elapsed.as_secs_f64() * 1e6 / (iters as usize * DISPATCHES) as f64;
                eprintln!("  -> {us:.1} µs/chunk");
                elapsed
            })
        });

        // Global: chunk at the end of an 8K context.
        let kernel = ctx.load_kernel("attn_prefill_global").expect("kernel");
        let l = 8192usize;
        let q0 = l - m;
        let q = ctx
            .buffer_from_iter(
                f16_fill(m * N_Q_HEADS * GL_DIM),
                BufferUsage::STORAGE_BUFFER,
            )
            .unwrap();
        let kv = ctx
            .buffer_from_iter(
                f16_fill(l * GL_KV_HEADS * GL_DIM),
                BufferUsage::STORAGE_BUFFER,
            )
            .unwrap();
        let out = ctx
            .new_buffer::<u16>((m * N_Q_HEADS * GL_DIM) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let layout = kernel.layout().clone();
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator().clone(),
            layout.set_layouts()[0].clone(),
            vec![
                WriteDescriptorSet::buffer(0, q),
                WriteDescriptorSet::buffer(1, kv),
                WriteDescriptorSet::buffer(2, out),
                WriteDescriptorSet::buffer(
                    3,
                    step_buf(
                        &ctx,
                        StepState {
                            q0: q0 as u32,
                            ..Default::default()
                        },
                    ),
                ),
            ],
            [],
        )
        .unwrap();
        group.bench_function(format!("prefill_global_m{m}_ctx{l}"), |b| {
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
                        .unwrap()
                        .push_constants(layout.clone(), 0, GL_SCALE)
                        .unwrap();
                    for _ in 0..DISPATCHES {
                        // SAFETY: [kv_heads, M] grid, the kernel's contract.
                        unsafe { builder.dispatch([GL_KV_HEADS as u32, m as u32, 1]) }.unwrap();
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
                let us = elapsed.as_secs_f64() * 1e6 / (iters as usize * DISPATCHES) as f64;
                eprintln!("  -> {us:.1} µs/chunk");
                elapsed
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
