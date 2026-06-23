//! Attention kernel timings in µs per dispatch (plan 06: attention benched
//! at ctx ∈ {1K, 8K, 32K}). Skips without a GPU.
//!
//! decode_global runs its full split-K pipeline (partial + reduce) per
//! dispatch, swept over split counts to inform the engine's n_splits
//! policy.

use criterion::{Criterion, criterion_group, criterion_main};
use sg_gpu::{Buffer, BufferBinding, BufferUsage, GpuContext, StepState};

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

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct PushSplitScale {
    n_splits: u32,
    scale: f32,
}

fn f16_fill(n: usize) -> impl ExactSizeIterator<Item = u16> {
    (0..n).map(|i| half::f16::from_f32((i % 23) as f32 * 0.1 - 1.1).to_bits())
}

/// Step buffer holding the per-step dynamic state.
fn step_buf(ctx: &GpuContext, state: StepState) -> Buffer<u32> {
    let buf = ctx.new_step_buffer().unwrap();
    state.write_to(&buf).unwrap();
    buf
}

/// Time `kernel` over `CMP_DISPATCHES` back-to-back dispatches of `grid`,
/// reporting µs/chunk. Shared by the naive-vs-flash global comparison; the
/// global-prefill push constant is always `GL_SCALE`. The dispatch count is
/// kept low so a single command buffer (naive global is ~146 ms/dispatch at
/// 32K) stays well under the 2 s amdgpu watchdog.
fn time_prefill(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    ctx: &GpuContext,
    kernel: &sg_gpu::Kernel,
    writes: Vec<BufferBinding>,
    grid: [u32; 3],
    name: String,
) {
    const CMP_DISPATCHES: usize = 4;
    // The CMP_DISPATCHES dispatches re-run the same kernel into the same `out`,
    // so the recorder's pre-dispatch barrier serializes them (a clean per-chunk
    // time, no overlap artifact). The global-prefill push is always GL_SCALE.
    let graph = ctx
        .record_graph(|rec| {
            for _ in 0..CMP_DISPATCHES {
                rec.dispatch(kernel, writes.clone(), Some(GL_SCALE), grid)?;
            }
            Ok(())
        })
        .expect("record");
    group.bench_function(name, |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                ctx.submit_blocking(&graph).unwrap();
            }
            let elapsed = start.elapsed();
            let us = elapsed.as_secs_f64() * 1e6 / (iters as usize * CMP_DISPATCHES) as f64;
            eprintln!("  -> {us:.1} µs/chunk");
            elapsed
        })
    });
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
            let p_writes = vec![
                BufferBinding::buffer(0, q.clone()),
                BufferBinding::buffer(1, k.clone()),
                BufferBinding::buffer(2, v.clone()),
                BufferBinding::buffer(3, part.clone()),
                BufferBinding::buffer(
                    4,
                    step_buf(
                        &ctx,
                        StepState {
                            kv_len_sliding: kv_len as u32,
                            ..Default::default()
                        },
                    ),
                ),
            ];
            let r_writes = vec![
                BufferBinding::buffer(0, part.clone()),
                BufferBinding::buffer(1, out.clone()),
            ];
            // Each step is partial→reduce (a real data dependency); the
            // recorder barriers between every dispatch.
            let graph = ctx
                .record_graph(|rec| {
                    for _ in 0..DISPATCHES {
                        rec.dispatch(
                            &part_k,
                            p_writes.clone(),
                            Some(PushSplitScale {
                                n_splits,
                                scale: SL_SCALE,
                            }),
                            [SL_KV_HEADS as u32, n_splits, 1],
                        )?;
                        rec.dispatch(&red_k, r_writes.clone(), Some(n_splits), [N_Q_HEADS as u32, 1, 1])?;
                    }
                    Ok(())
                })
                .expect("record");

            group.bench_function(format!("decode_sliding_ring1024_splits{n_splits}"), |b| {
                b.iter_custom(|iters| {
                    let start = std::time::Instant::now();
                    for _ in 0..iters {
                        ctx.submit_blocking(&graph).unwrap();
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
            let kv_k = ctx
                .buffer_from_iter(
                    f16_fill(kv_len * GL_KV_HEADS * GL_DIM),
                    BufferUsage::STORAGE_BUFFER,
                )
                .unwrap();
            let kv_v = ctx
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
                let p_writes = vec![
                    BufferBinding::buffer(0, q.clone()),
                    BufferBinding::buffer(1, kv_k.clone()),
                    BufferBinding::buffer(2, kv_v.clone()),
                    BufferBinding::buffer(3, part.clone()),
                    BufferBinding::buffer(
                        4,
                        step_buf(
                            &ctx,
                            StepState {
                                kv_len_global: kv_len as u32,
                                ..Default::default()
                            },
                        ),
                    ),
                ];
                let r_writes = vec![
                    BufferBinding::buffer(0, part.clone()),
                    BufferBinding::buffer(1, out.clone()),
                ];
                let graph = ctx
                    .record_graph(|rec| {
                        for _ in 0..DISPATCHES {
                            rec.dispatch(
                                &part_k,
                                p_writes.clone(),
                                Some(PushSplitScale {
                                    n_splits,
                                    scale: GL_SCALE,
                                }),
                                [GL_KV_HEADS as u32, n_splits, 1],
                            )?;
                            rec.dispatch(
                                &red_k,
                                r_writes.clone(),
                                Some(n_splits),
                                [N_Q_HEADS as u32, 1, 1],
                            )?;
                        }
                        Ok(())
                    })
                    .expect("record");

                group.bench_function(format!("decode_global_ctx{kv_len}_splits{n_splits}"), |b| {
                    b.iter_custom(|iters| {
                        let start = std::time::Instant::now();
                        for _ in 0..iters {
                            ctx.submit_blocking(&graph).unwrap();
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
        let writes = vec![
            BufferBinding::buffer(0, q),
            BufferBinding::buffer(1, k),
            BufferBinding::buffer(2, v),
            BufferBinding::buffer(3, out),
            BufferBinding::buffer(
                4,
                step_buf(
                    &ctx,
                    StepState {
                        q0: q0 as u32,
                        ..Default::default()
                    },
                ),
            ),
        ];
        let graph = ctx
            .record_graph(|rec| {
                for _ in 0..DISPATCHES {
                    rec.dispatch(&kernel, writes.clone(), Some(SL_SCALE), [SL_KV_HEADS as u32, m as u32, 1])?;
                }
                Ok(())
            })
            .expect("record");
        group.bench_function(format!("prefill_sliding_m{m}"), |b| {
            b.iter_custom(|iters| {
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    ctx.submit_blocking(&graph).unwrap();
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
        let kv_k = ctx
            .buffer_from_iter(
                f16_fill(l * GL_KV_HEADS * GL_DIM),
                BufferUsage::STORAGE_BUFFER,
            )
            .unwrap();
        let kv_v = ctx
            .buffer_from_iter(
                f16_fill(l * GL_KV_HEADS * GL_DIM),
                BufferUsage::STORAGE_BUFFER,
            )
            .unwrap();
        let out = ctx
            .new_buffer::<u16>((m * N_Q_HEADS * GL_DIM) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let writes = vec![
            BufferBinding::buffer(0, q),
            BufferBinding::buffer(1, kv_k),
            BufferBinding::buffer(2, kv_v),
            BufferBinding::buffer(3, out),
            BufferBinding::buffer(
                4,
                step_buf(
                    &ctx,
                    StepState {
                        q0: q0 as u32,
                        ..Default::default()
                    },
                ),
            ),
        ];
        let graph = ctx
            .record_graph(|rec| {
                for _ in 0..DISPATCHES {
                    rec.dispatch(&kernel, writes.clone(), Some(GL_SCALE), [GL_KV_HEADS as u32, m as u32, 1])?;
                }
                Ok(())
            })
            .expect("record");
        group.bench_function(format!("prefill_global_m{m}_ctx{l}"), |b| {
            b.iter_custom(|iters| {
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    ctx.submit_blocking(&graph).unwrap();
                }
                let elapsed = start.elapsed();
                let us = elapsed.as_secs_f64() * 1e6 / (iters as usize * DISPATCHES) as f64;
                eprintln!("  -> {us:.1} µs/chunk");
                elapsed
            })
        });
    }

    group.finish();

    // --- global prefill: naive scalar vs coopmat flash (v2 register-O two-
    //     pass), at the three profile operating points (chunk at the end of a
    //     256/8K/32K context). Re-measured at perf=high — the parked flash
    //     numbers in STATUS (~1.2× slower) were taken at perf=auto, before the
    //     fabric clock was pinned. m=256 matches the production chunk. ---
    {
        let m = 256usize;
        let naive = ctx.load_kernel("attn_prefill_global").expect("kernel");
        let flash = ctx
            .load_kernel("attn_prefill_global_flash")
            .expect("kernel");
        let flash_sp = ctx
            .load_kernel("attn_prefill_global_flash_sp")
            .expect("kernel");
        // Q8-KV flash (Piece B): K/V from the Q8 cache, dequant-on-load (the
        // f16-convert dead end) vs int8 QKᵀ (the right approach, _iq).
        let flash_sp_q8 = ctx
            .load_kernel("attn_prefill_global_flash_sp_q8")
            .expect("kernel");
        let flash_sp_iq = ctx
            .load_kernel("attn_prefill_global_flash_sp_iq")
            .expect("kernel");
        // int8 QKᵀ AND int8 PV (Piece B): V also streams i8 from the cache.
        let flash_sp_ipv = ctx
            .load_kernel("attn_prefill_global_flash_sp_ipv")
            .expect("kernel");
        // _ipv with both rescales built by 0-stride coopLoad broadcasts.
        let flash_sp_ipv_bcast = ctx
            .load_kernel("attn_prefill_global_flash_sp_ipv_bcast")
            .expect("kernel");
        let mut cmp = c.benchmark_group("attn_flash_cmp");
        cmp.sample_size(10)
            .warm_up_time(std::time::Duration::from_secs(1))
            .measurement_time(std::time::Duration::from_secs(3));
        for l in [256usize, 8192, 32768] {
            let q0 = (l - m) as u32; // chunk sits at the end of an l-token context
            let q = ctx
                .buffer_from_iter(
                    f16_fill(m * N_Q_HEADS * GL_DIM),
                    BufferUsage::STORAGE_BUFFER,
                )
                .unwrap();
            // l is a multiple of 64, so the flash N_K=64 tiling needs no extra
            // KV padding (the last query's key tile ends exactly at l).
            let kv_k = ctx
                .buffer_from_iter(
                    f16_fill(l * GL_KV_HEADS * GL_DIM),
                    BufferUsage::STORAGE_BUFFER,
                )
                .unwrap();
            let kv_v = ctx
                .buffer_from_iter(
                    f16_fill(l * GL_KV_HEADS * GL_DIM),
                    BufferUsage::STORAGE_BUFFER,
                )
                .unwrap();
            let out = ctx
                .new_buffer::<u16>((m * N_Q_HEADS * GL_DIM) as u64, BufferUsage::STORAGE_BUFFER)
                .unwrap();
            let step = step_buf(
                &ctx,
                StepState {
                    q0,
                    ..Default::default()
                },
            );
            let writes = || {
                vec![
                    BufferBinding::buffer(0, q.clone()),
                    BufferBinding::buffer(1, kv_k.clone()),
                    BufferBinding::buffer(2, kv_v.clone()),
                    BufferBinding::buffer(3, out.clone()),
                    BufferBinding::buffer(4, step.clone()),
                ]
            };
            let set_naive = writes();
            let set_flash = writes();
            let set_flash_sp = writes();
            // Q8 KV buffers (dummy: timing is data-independent). Blocks of 32
            // over the [l × GL_KV_HEADS × GL_DIM] cache → l·64 blocks each.
            let blocks = l * GL_KV_HEADS * GL_DIM / 32;
            let q8buf = || {
                let qn = ctx
                    .buffer_from_iter((0..blocks * 8).map(|i| i as u32), BufferUsage::STORAGE_BUFFER)
                    .unwrap();
                let sc = ctx
                    .buffer_from_iter(f16_fill(blocks), BufferUsage::STORAGE_BUFFER)
                    .unwrap();
                (qn, sc)
            };
            let (kq, ks) = q8buf();
            let (vq, vs) = q8buf();
            // Pre-quantized Q (i8 + f16 scales) for the int8-QKᵀ variant.
            let q_i8 = ctx
                .buffer_from_iter(
                    (0..m * N_Q_HEADS * GL_DIM / 4).map(|i| i as u32),
                    BufferUsage::STORAGE_BUFFER,
                )
                .unwrap();
            let q_sc = ctx
                .buffer_from_iter(
                    f16_fill(m * N_Q_HEADS * GL_DIM / 32),
                    BufferUsage::STORAGE_BUFFER,
                )
                .unwrap();
            let set_flash_sp_iq = vec![
                BufferBinding::buffer(0, q_i8.clone()),
                BufferBinding::buffer(1, q_sc.clone()),
                BufferBinding::buffer(2, kq.clone()),
                BufferBinding::buffer(3, ks.clone()),
                BufferBinding::buffer(4, kv_v.clone()),
                BufferBinding::buffer(5, out.clone()),
                BufferBinding::buffer(6, step.clone()),
            ];
            let set_flash_sp_ipv = vec![
                BufferBinding::buffer(0, q_i8.clone()),
                BufferBinding::buffer(1, q_sc.clone()),
                BufferBinding::buffer(2, kq.clone()),
                BufferBinding::buffer(3, ks.clone()),
                BufferBinding::buffer(4, vq.clone()),
                BufferBinding::buffer(5, vs.clone()),
                BufferBinding::buffer(6, out.clone()),
                BufferBinding::buffer(7, step.clone()),
            ];
            let set_flash_sp_ipv_bcast = vec![
                BufferBinding::buffer(0, q_i8.clone()),
                BufferBinding::buffer(1, q_sc.clone()),
                BufferBinding::buffer(2, kq.clone()),
                BufferBinding::buffer(3, ks.clone()),
                BufferBinding::buffer(4, vq.clone()),
                BufferBinding::buffer(5, vs.clone()),
                BufferBinding::buffer(6, out.clone()),
                BufferBinding::buffer(7, step.clone()),
            ];
            let set_flash_sp_q8 = vec![
                BufferBinding::buffer(0, q.clone()),
                BufferBinding::buffer(1, kq.clone()),
                BufferBinding::buffer(2, ks.clone()),
                BufferBinding::buffer(3, vq.clone()),
                BufferBinding::buffer(4, vs.clone()),
                BufferBinding::buffer(5, out.clone()),
                BufferBinding::buffer(6, step.clone()),
            ];
            // naive grid [kv_heads, M]; flash grids [q_heads, M/M_Q] (M_Q=16).
            time_prefill(
                &mut cmp,
                &ctx,
                &naive,
                set_naive,
                [GL_KV_HEADS as u32, m as u32, 1],
                format!("naive_ctx{l}"),
            );
            time_prefill(
                &mut cmp,
                &ctx,
                &flash,
                set_flash,
                [N_Q_HEADS as u32, (m / 16) as u32, 1],
                format!("flash_ctx{l}"),
            );
            time_prefill(
                &mut cmp,
                &ctx,
                &flash_sp,
                set_flash_sp,
                [N_Q_HEADS as u32, (m / 16) as u32, 1],
                format!("flash_sp_ctx{l}"),
            );
            time_prefill(
                &mut cmp,
                &ctx,
                &flash_sp_q8,
                set_flash_sp_q8,
                [N_Q_HEADS as u32, (m / 16) as u32, 1],
                format!("flash_sp_q8_ctx{l}"),
            );
            time_prefill(
                &mut cmp,
                &ctx,
                &flash_sp_iq,
                set_flash_sp_iq,
                [N_Q_HEADS as u32, (m / 16) as u32, 1],
                format!("flash_sp_iq_ctx{l}"),
            );
            time_prefill(
                &mut cmp,
                &ctx,
                &flash_sp_ipv,
                set_flash_sp_ipv,
                [N_Q_HEADS as u32, (m / 16) as u32, 1],
                format!("flash_sp_ipv_ctx{l}"),
            );
            time_prefill(
                &mut cmp,
                &ctx,
                &flash_sp_ipv_bcast,
                set_flash_sp_ipv_bcast,
                [N_Q_HEADS as u32, (m / 16) as u32, 1],
                format!("flash_sp_ipv_bcast_ctx{l}"),
            );
        }
        cmp.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
