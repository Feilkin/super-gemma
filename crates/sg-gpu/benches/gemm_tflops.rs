//! gemm_q4_0 (cooperative matrix) compute-rate microbench (plan 02: ≥ 30 %
//! of the ~59 TFLOPS f16 peak initially, stretch ≥ 50 %). Skips without a
//! coopmat GPU.

use criterion::{Criterion, criterion_group, criterion_main};
use sg_gpu::{BufferBinding, BufferUsage, GpuContext};

const M: usize = 512; // prefill chunk
const K: usize = 5376;
const N: usize = 21504; // ffn gate/up — the prefill flop bucket
const DISPATCHES: usize = 4;
// Must match the gemm variant defines (M_TILES=4, N_TILES=4).
const M_BLOCK: usize = 64;
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
        let grid = [(N / n_block) as u32, (M / m_block) as u32, 1];
        // `dispatch_overlapping` (no inter-dispatch barrier) reproduces the
        // historical vulkano behavior: a coopStore to `y` was invisible to its
        // auto-sync, so the DISPATCHES dispatches overlapped. Use `dispatch`
        // for a serialized per-dispatch rate.
        let graph = ctx
            .record_graph(|rec| {
                for _ in 0..DISPATCHES {
                    rec.dispatch_overlapping(
                        &kernel,
                        vec![
                            BufferBinding::buffer(0, w_buf.clone()),
                            BufferBinding::buffer(1, x_buf.clone()),
                            BufferBinding::buffer(2, y_buf.clone()),
                        ],
                        None::<u32>,
                        grid,
                    )?;
                }
                Ok(())
            })
            .expect("record");

        group.bench_function(format!("{name}_m{M}"), |b| {
            b.iter_custom(|iters| {
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    ctx.submit_blocking(&graph).unwrap();
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
