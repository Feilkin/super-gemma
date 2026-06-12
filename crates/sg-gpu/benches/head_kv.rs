//! gemv_q6_k_logits at the real LM-head size (262144 × 5376, ~1.16 GB of
//! Q6_K weights streamed per token — the decode-step budget item) plus
//! kv_quant/kv_dequant throughput. Skips without a GPU.

use criterion::{Criterion, criterion_group, criterion_main};
use sg_gpu::GpuContext;
use vulkano::buffer::BufferUsage;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::sync::GpuFuture;

const VOCAB: usize = 262_144;
const K: usize = 5376;
const ROW_WORDS: usize = 1104; // padded Q6_K row stride (4416 bytes)

fn bench(c: &mut Criterion) {
    let ctx = match GpuContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("skipping head/kv bench: no usable GPU ({e})");
            return;
        }
    };
    let mut group = c.benchmark_group("head_kv");

    // --- LM head: logits for one token. ---
    {
        let kernel = ctx.load_kernel("gemv_q6_k_logits").expect("kernel");
        let weight_words = VOCAB * ROW_WORDS;
        let w_buf = ctx
            .buffer_from_iter(
                (0..weight_words as u32).map(|i| i.wrapping_mul(0x9E37_79B9) | 0x0001_0001),
                BufferUsage::STORAGE_BUFFER,
            )
            .expect("weights");
        let x_buf = ctx
            .buffer_from_iter(
                (0..K).map(|i| half::f16::from_f32((i % 13) as f32 * 0.03).to_bits()),
                BufferUsage::STORAGE_BUFFER,
            )
            .expect("x");
        let y_buf = ctx
            .new_buffer::<f32>(VOCAB as u64, BufferUsage::STORAGE_BUFFER)
            .expect("logits");
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
        let bytes = (weight_words * 4) as f64;

        group.bench_function("lm_head_logits", |b| {
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
                    // SAFETY: one workgroup per vocab row, the kernel's contract.
                    unsafe { builder.dispatch([VOCAB as u32, 1, 1]) }.unwrap();
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
                let secs = elapsed.as_secs_f64() / iters as f64;
                eprintln!(
                    "  -> {:.2} ms/token, {:.0} GiB/s",
                    secs * 1e3,
                    bytes / secs / (1u64 << 30) as f64
                );
                elapsed
            })
        });
    }

    // --- KV Q8_0 codec: one sliding layer's full ring (8 MiB f16). ---
    {
        let quant_k = ctx.load_kernel("kv_quant_q8").expect("kernel");
        let dequant_k = ctx.load_kernel("kv_dequant_q8").expect("kernel");
        let n = 1024 * 16 * 256; // ring slots × heads × head_dim
        let src_buf = ctx
            .buffer_from_iter(
                (0..n).map(|i| half::f16::from_f32((i % 19) as f32 * 0.1 - 0.9).to_bits()),
                BufferUsage::STORAGE_BUFFER,
            )
            .unwrap();
        let scale_buf = ctx
            .new_buffer::<u16>((n / 32) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let quant_buf = ctx
            .new_buffer::<u32>((n / 4) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let dst_buf = ctx
            .new_buffer::<u16>(n as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();

        for (name, kernel, writes) in [
            (
                "kv_quant_q8_ring",
                &quant_k,
                vec![
                    WriteDescriptorSet::buffer(0, src_buf.clone()),
                    WriteDescriptorSet::buffer(1, scale_buf.clone()),
                    WriteDescriptorSet::buffer(2, quant_buf.clone()),
                ],
            ),
            (
                "kv_dequant_q8_ring",
                &dequant_k,
                vec![
                    WriteDescriptorSet::buffer(0, scale_buf.clone()),
                    WriteDescriptorSet::buffer(1, quant_buf.clone()),
                    WriteDescriptorSet::buffer(2, dst_buf.clone()),
                ],
            ),
        ] {
            let layout = kernel.layout().clone();
            let set = DescriptorSet::new(
                ctx.descriptor_set_allocator().clone(),
                layout.set_layouts()[0].clone(),
                writes,
                [],
            )
            .unwrap();
            let groups = kernel.groups_for((n / 32) as u64);
            group.bench_function(name, |b| {
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
                        // SAFETY: one thread per block, the kernel's contract.
                        unsafe { builder.dispatch(groups) }.unwrap();
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
                    let secs = elapsed.as_secs_f64() / iters as f64;
                    eprintln!(
                        "  -> {:.0} µs/ring, {:.0} GiB/s (f16 side)",
                        secs * 1e6,
                        (n * 2) as f64 / secs / (1u64 << 30) as f64
                    );
                    elapsed
                })
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
