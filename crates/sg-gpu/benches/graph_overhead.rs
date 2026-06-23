//! Per-submit overhead of a pre-recorded decode-shaped graph (plan 02
//! §submission model: < 300 µs CPU-side per decode step). The graph is 60
//! "layers" × 4 small GEMVs ping-ponging two activation buffers — the real
//! decode graph's dispatch count and dependency structure without the 18 GB
//! of weights. Reports wall, GPU (timestamps), and CPU overhead
//! (wall − GPU) per submit. Skips without a GPU.

use criterion::{Criterion, criterion_group, criterion_main};
use sg_gpu::{BufferBinding, BufferUsage, GpuContext};

const LAYERS: usize = 60;
const MATMULS_PER_LAYER: usize = 4;
const K: usize = 512; // small synthetic shape; structure is what matters
const N: usize = 512;

fn bench(c: &mut Criterion) {
    let ctx = match GpuContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("skipping graph bench: no usable GPU ({e})");
            return;
        }
    };
    let kernel = ctx.load_kernel("gemv_q4_0_generic").expect("kernel");

    let weight_words = N * K / 32 * 18 / 4;
    let w_buf = ctx
        .buffer_from_iter(
            (0..weight_words as u32).map(|i| i.wrapping_mul(0x9E37_79B9)),
            BufferUsage::STORAGE_BUFFER,
        )
        .unwrap();
    let a_buf = ctx
        .buffer_from_iter(
            (0..K).map(|i| half::f16::from_f32((i % 7) as f32 * 0.1).to_bits()),
            BufferUsage::STORAGE_BUFFER,
        )
        .unwrap();
    let b_buf = ctx
        .new_buffer::<u16>(N as u64, BufferUsage::STORAGE_BUFFER)
        .unwrap();

    let timer = ctx.new_timer(2).unwrap();
    let graph = ctx
        .record_graph(|rec| {
            rec.reset_timer(&timer)?.timestamp(&timer, 0)?;
            for _ in 0..LAYERS {
                for m in 0..MATMULS_PER_LAYER {
                    // Ping-pong so every dispatch depends on the previous one,
                    // forcing the same barrier pattern as a real layer stack.
                    let (src, dst) = if m % 2 == 0 {
                        (a_buf.clone(), b_buf.clone())
                    } else {
                        (b_buf.clone(), a_buf.clone())
                    };
                    rec.dispatch(
                        &kernel,
                        vec![
                            BufferBinding::buffer(0, w_buf.clone()),
                            BufferBinding::buffer(1, src),
                            BufferBinding::buffer(2, dst),
                        ],
                        Some(K as u32),
                        [N as u32, 1, 1],
                    )?;
                }
            }
            rec.timestamp(&timer, 1)?;
            Ok(())
        })
        .unwrap();

    let dispatches = LAYERS * MATMULS_PER_LAYER;
    c.benchmark_group("graph").bench_function(
        format!("decode_shaped_{dispatches}_dispatches"),
        |b| {
            b.iter_custom(|iters| {
                let start = std::time::Instant::now();
                let mut gpu_ns = 0.0f64;
                for _ in 0..iters {
                    ctx.submit_blocking(&graph).unwrap();
                    let ts = timer.read_ns().unwrap();
                    gpu_ns += ts[1] - ts[0];
                }
                let elapsed = start.elapsed();
                let wall_us = elapsed.as_secs_f64() * 1e6 / iters as f64;
                let gpu_us = gpu_ns / 1e3 / iters as f64;
                eprintln!(
                    "  -> wall {wall_us:.0} µs/submit, GPU {gpu_us:.0} µs, CPU overhead \
                     {:.0} µs (target < 300)",
                    wall_us - gpu_us
                );
                elapsed
            })
        },
    );
}

criterion_group!(benches, bench);
criterion_main!(benches);
