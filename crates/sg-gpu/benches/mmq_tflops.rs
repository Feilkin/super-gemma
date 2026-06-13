//! int8-MMQ vs f16 coopmat GEMM compute-rate microbench (profile rank #2).
//! Same shape (M×K×N = 512×5376×21504, the FFN gate/up bucket), same effective
//! flop count (2·M·N·K) — the decisive question is whether the int8 MMA
//! throughput beats f16 *after* paying the per-block i32→f32 rescale through
//! LDS (docs/naga-int8-coopmat-patch.md). Skips without a coopmat GPU.

use criterion::{Criterion, criterion_group, criterion_main};
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
const NB: usize = K / 32;
const DISPATCHES: usize = 4;

fn bench(c: &mut Criterion) {
    let ctx = match GpuContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("skipping mmq bench: no usable GPU ({e})");
            return;
        }
    };
    if !ctx.cooperative_matrix {
        eprintln!("skipping mmq bench: no VK_KHR_cooperative_matrix");
        return;
    }

    // f16 gemm operands.
    let weight_words = N * K / 32 * 18 / 4;
    let w_f16 = ctx
        .buffer_from_iter(
            (0..weight_words as u32).map(|i| i.wrapping_mul(0x9E37_79B9)),
            BufferUsage::STORAGE_BUFFER,
        )
        .expect("w_f16");
    let x_f16 = ctx
        .buffer_from_iter(
            (0..M * K).map(|i| half::f16::from_f32((i % 11) as f32 * 0.05).to_bits()),
            BufferUsage::STORAGE_BUFFER,
        )
        .expect("x_f16");
    // int8 MMQ operands (values irrelevant to timing).
    let w_i8 = ctx
        .buffer_from_iter(
            (0..N * K).map(|i| (i % 15) as i8 - 7),
            BufferUsage::STORAGE_BUFFER,
        )
        .expect("w_i8");
    let w_sc = ctx
        .buffer_from_iter(
            (0..N * NB).map(|i| half::f16::from_f32((i % 7) as f32 * 0.01).to_bits()),
            BufferUsage::STORAGE_BUFFER,
        )
        .expect("w_sc");
    let x_i8 = ctx
        .buffer_from_iter(
            (0..M * K).map(|i| (i % 31) as i8 - 15),
            BufferUsage::STORAGE_BUFFER,
        )
        .expect("x_i8");
    let x_sc = ctx
        .buffer_from_iter(
            (0..M * NB).map(|i| half::f16::from_f32((i % 5) as f32 * 0.02).to_bits()),
            BufferUsage::STORAGE_BUFFER,
        )
        .expect("x_sc");
    let y = ctx
        .new_buffer::<u16>((M * N) as u64, BufferUsage::STORAGE_BUFFER)
        .expect("y");

    // (name, descriptor writes, n_block, m_block).
    let f16_writes = vec![
        WriteDescriptorSet::buffer(0, w_f16.clone()),
        WriteDescriptorSet::buffer(1, x_f16.clone()),
        WriteDescriptorSet::buffer(2, y.clone()),
    ];
    let i8_writes = vec![
        WriteDescriptorSet::buffer(0, w_i8.clone()),
        WriteDescriptorSet::buffer(1, w_sc.clone()),
        WriteDescriptorSet::buffer(2, x_i8.clone()),
        WriteDescriptorSet::buffer(3, x_sc.clone()),
        WriteDescriptorSet::buffer(4, y.clone()),
    ];
    let raw_writes = vec![
        WriteDescriptorSet::buffer(0, w_i8.clone()),
        WriteDescriptorSet::buffer(1, x_i8.clone()),
        WriteDescriptorSet::buffer(2, y.clone()),
    ];
    let cases: [(&str, Vec<WriteDescriptorSet>, u32, u32); 5] = [
        ("gemm_q4_0_k5376_n21504", f16_writes, 64, 64), // f16, 4×4 tiles
        ("gemm_q4_0_i8_k5376_n21504", i8_writes.clone(), 64, 32), // int8 MMQ, 2×4 tiles
        ("gemm_q4_0_i8_t22_k5376_n21504", i8_writes.clone(), 32, 32), // 2×2 tiles
        ("gemm_q4_0_i8_t12_k5376_n21504", i8_writes, 32, 16), // 1×2 tiles
        ("gemm_q4_0_i8_raw_k5376_n21504", raw_writes, 64, 32), // int8 MMA ceiling, no rescale
    ];

    let flops = 2.0 * M as f64 * N as f64 * K as f64 * DISPATCHES as f64;
    let mut group = c.benchmark_group("mmq_vs_f16");
    for (name, writes, n_block, m_block) in cases {
        let kernel = ctx.load_kernel(name).expect(name);
        let layout = kernel.layout().clone();
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator().clone(),
            layout.set_layouts()[0].clone(),
            writes,
            [],
        )
        .expect("set");

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
                    for _ in 0..DISPATCHES {
                        // SAFETY: tile grid covering M×N, the kernel's contract.
                        unsafe {
                            builder.dispatch([(N as u32 / n_block), (M as u32 / m_block), 1])
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
                eprintln!("  -> {name}: {tflops:.2} TFLOPS");
                elapsed
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
