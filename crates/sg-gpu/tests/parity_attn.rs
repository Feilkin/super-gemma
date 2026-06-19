//! Parity: the four attention kernels (plan 02 step 6) vs the f64 CPU
//! reference, covering window-boundary positions, partial/full rings, and
//! split-K (incl. empty splits). Skips without a GPU.
//!
//! Ring wraparound note: `attn_decode_sliding` iterates ring slots in
//! physical order and softmax is order-invariant, so a wrapped ring is
//! indistinguishable from an unwrapped one to the kernel — the "full ring"
//! case here covers it. Bit-exactness across reruns of the same physical
//! state is what plan 06 requires, tested below.

mod reference;

use reference::{
    Rng, assert_close, attention_head, from_f16_bits, q8_quant, q8_quant_v, through_f16,
    to_f16_bits,
};
use sg_gpu::{GpuContext, StepState};
use vulkano::buffer::{BufferContents, BufferUsage, Subbuffer};
use vulkano::descriptor_set::WriteDescriptorSet;

const N_Q_HEADS: usize = 32;
/// Sliding layers: GQA 32:16, head_dim 256, 1/√256.
const SL_KV_HEADS: usize = 16;
const SL_DIM: usize = 256;
const SL_SCALE: f32 = 0.0625;
/// Global layers: GQA 32:4, head_dim 512, K=V, 1/√512.
const GL_KV_HEADS: usize = 4;
const GL_DIM: usize = 512;
const GL_SCALE: f32 = 0.044194174;
/// Partials stride of the split-K pipelines (acc + m + l).
const SL_PART_STRIDE: usize = SL_DIM + 2;
const GL_PART_STRIDE: usize = GL_DIM + 2;

const ATOL: f32 = 1e-2;
const RTOL: f32 = 1e-2;

fn ctx() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("skipping: no usable GPU ({e})");
            None
        }
    }
}

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct PushSplitScale {
    n_splits: u32,
    scale: f32,
}

/// Step buffer holding the per-step dynamic state (kv lengths / q0).
fn step_buf(ctx: &GpuContext, state: StepState) -> Subbuffer<[u32]> {
    let buf = ctx.new_step_buffer().unwrap();
    state.write_to(&buf).unwrap();
    buf
}

#[test]
fn attn_decode_sliding_matches_reference() {
    let Some(ctx) = ctx() else { return };
    let part_k = ctx.load_kernel("attn_decode_sliding").unwrap();
    let red_k = ctx.load_kernel("attn_reduce_d256").unwrap();
    let mut rng = Rng::new(0xA77);

    // (kv_len, n_splits): partially filled ring single-split, partial ring
    // with uneven splits, and a full ring (== the window) split as in
    // production.
    for (kv_len, n_splits) in [(17usize, 1u32), (17, 4), (1024, 8)] {
        let q = through_f16(&rng.f32_vec(N_Q_HEADS * SL_DIM));
        let k = through_f16(&rng.f32_vec(kv_len * SL_KV_HEADS * SL_DIM));
        let v = through_f16(&rng.f32_vec(kv_len * SL_KV_HEADS * SL_DIM));

        let q_buf = ctx
            .buffer_from_iter(to_f16_bits(&q), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let k_buf = ctx
            .buffer_from_iter(to_f16_bits(&k), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let v_buf = ctx
            .buffer_from_iter(to_f16_bits(&v), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let part_buf = ctx
            .new_buffer::<f32>(
                (N_Q_HEADS * n_splits as usize * SL_PART_STRIDE) as u64,
                BufferUsage::STORAGE_BUFFER,
            )
            .unwrap();
        let out_buf = ctx
            .new_buffer::<u16>((N_Q_HEADS * SL_DIM) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();

        ctx.dispatch_blocking(
            &part_k,
            vec![
                WriteDescriptorSet::buffer(0, q_buf),
                WriteDescriptorSet::buffer(1, k_buf),
                WriteDescriptorSet::buffer(2, v_buf),
                WriteDescriptorSet::buffer(3, part_buf.clone()),
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
            Some(PushSplitScale {
                n_splits,
                scale: SL_SCALE,
            }),
            [SL_KV_HEADS as u32, n_splits, 1],
        )
        .unwrap();
        ctx.dispatch_blocking(
            &red_k,
            vec![
                WriteDescriptorSet::buffer(0, part_buf),
                WriteDescriptorSet::buffer(1, out_buf.clone()),
            ],
            Some(n_splits),
            [N_Q_HEADS as u32, 1, 1],
        )
        .unwrap();
        let got = from_f16_bits(&out_buf.read().unwrap());

        for qh in 0..N_Q_HEADS {
            let want = attention_head(
                &q,
                &k,
                &v,
                0,
                qh,
                N_Q_HEADS,
                SL_KV_HEADS,
                SL_DIM,
                SL_SCALE as f64,
                0,
                kv_len - 1,
            );
            assert_close(
                &got[qh * SL_DIM..][..SL_DIM],
                &want,
                ATOL,
                RTOL,
                &format!("decode_sliding kv_len={kv_len} splits={n_splits} qh={qh}"),
            );
        }
    }
}

#[test]
fn attn_decode_global_matches_reference() {
    let Some(ctx) = ctx() else { return };
    let part_k = ctx.load_kernel("attn_decode_global").unwrap();
    let red_k = ctx.load_kernel("attn_reduce_d512").unwrap();
    let mut rng = Rng::new(0xA78);

    // (kv_len, n_splits): single split, uneven split, splits beyond kv_len
    // (empty partials), and a longer context.
    for (kv_len, n_splits) in [(333usize, 1u32), (333, 5), (3, 8), (4096, 4)] {
        let q = through_f16(&rng.f32_vec(N_Q_HEADS * GL_DIM));
        let k = through_f16(&rng.f32_vec(kv_len * GL_KV_HEADS * GL_DIM));
        let v = through_f16(&rng.f32_vec(kv_len * GL_KV_HEADS * GL_DIM));

        let q_buf = ctx
            .buffer_from_iter(to_f16_bits(&q), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let k_buf = ctx
            .buffer_from_iter(to_f16_bits(&k), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let v_buf = ctx
            .buffer_from_iter(to_f16_bits(&v), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let part_buf = ctx
            .new_buffer::<f32>(
                (N_Q_HEADS * n_splits as usize * GL_PART_STRIDE) as u64,
                BufferUsage::STORAGE_BUFFER,
            )
            .unwrap();
        let out_buf = ctx
            .new_buffer::<u16>((N_Q_HEADS * GL_DIM) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();

        ctx.dispatch_blocking(
            &part_k,
            vec![
                WriteDescriptorSet::buffer(0, q_buf),
                WriteDescriptorSet::buffer(1, k_buf),
                WriteDescriptorSet::buffer(2, v_buf),
                WriteDescriptorSet::buffer(3, part_buf.clone()),
                WriteDescriptorSet::buffer(
                    4,
                    step_buf(
                        &ctx,
                        StepState {
                            kv_len_global: kv_len as u32,
                            ..Default::default()
                        },
                    ),
                ),
            ],
            Some(PushSplitScale {
                n_splits,
                scale: GL_SCALE,
            }),
            [GL_KV_HEADS as u32, n_splits, 1],
        )
        .unwrap();
        ctx.dispatch_blocking(
            &red_k,
            vec![
                WriteDescriptorSet::buffer(0, part_buf),
                WriteDescriptorSet::buffer(1, out_buf.clone()),
            ],
            Some(n_splits),
            [N_Q_HEADS as u32, 1, 1],
        )
        .unwrap();
        let got = from_f16_bits(&out_buf.read().unwrap());

        for qh in 0..N_Q_HEADS {
            let want = attention_head(
                &q,
                &k,
                &v,
                0,
                qh,
                N_Q_HEADS,
                GL_KV_HEADS,
                GL_DIM,
                GL_SCALE as f64,
                0,
                kv_len - 1,
            );
            assert_close(
                &got[qh * GL_DIM..][..GL_DIM],
                &want,
                ATOL,
                RTOL,
                &format!("decode_global kv_len={kv_len} splits={n_splits} qh={qh}"),
            );
        }
    }
}

#[test]
fn attn_prefill_sliding_matches_reference() {
    let Some(ctx) = ctx() else { return };
    let kernel = ctx.load_kernel("attn_prefill_sliding").unwrap();
    let mut rng = Rng::new(0xA79);
    let m = 64usize;
    const WINDOW: usize = 1024;

    // q0 = history length. 0: short pure-causal chunk. 1000: queries span
    // positions 1000..1063, crossing the 1023→1024 window-saturation
    // boundary mid-chunk.
    for q0 in [0usize, 1000] {
        let l = q0 + m;
        let q = through_f16(&rng.f32_vec(m * N_Q_HEADS * SL_DIM));
        let k = through_f16(&rng.f32_vec(l * SL_KV_HEADS * SL_DIM));
        let v = through_f16(&rng.f32_vec(l * SL_KV_HEADS * SL_DIM));

        let q_buf = ctx
            .buffer_from_iter(to_f16_bits(&q), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let k_buf = ctx
            .buffer_from_iter(to_f16_bits(&k), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let v_buf = ctx
            .buffer_from_iter(to_f16_bits(&v), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let out_buf = ctx
            .new_buffer::<u16>((m * N_Q_HEADS * SL_DIM) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();

        ctx.dispatch_blocking(
            &kernel,
            vec![
                WriteDescriptorSet::buffer(0, q_buf),
                WriteDescriptorSet::buffer(1, k_buf),
                WriteDescriptorSet::buffer(2, v_buf),
                WriteDescriptorSet::buffer(3, out_buf.clone()),
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
            Some(SL_SCALE),
            [SL_KV_HEADS as u32, m as u32, 1],
        )
        .unwrap();
        let got = from_f16_bits(&out_buf.read().unwrap());

        // Sampled query heads across the GQA groups (full f64 reference on
        // all 32 heads × 64 queries adds minutes, not signal); every token.
        for qh in [0usize, 1, 9, 16, 31] {
            for i in 0..m {
                let qpos = q0 + i;
                let t0 = (qpos + 1).saturating_sub(WINDOW);
                let want = attention_head(
                    &q,
                    &k,
                    &v,
                    i,
                    qh,
                    N_Q_HEADS,
                    SL_KV_HEADS,
                    SL_DIM,
                    SL_SCALE as f64,
                    t0,
                    qpos,
                );
                assert_close(
                    &got[(i * N_Q_HEADS + qh) * SL_DIM..][..SL_DIM],
                    &want,
                    ATOL,
                    RTOL,
                    &format!("prefill_sliding q0={q0} i={i} qh={qh}"),
                );
            }
        }
    }
}

/// Two-range sliding prefill (plan 03 §prefill): history keys from the
/// pre-append ring (`pos % 1024` slots), the chunk's own keys from the
/// chunk buffers. Reference: the same f64 attention over an equivalent
/// LINEAR key layout. Cases: cold start, partial ring, window saturating
/// mid-chunk, and a wrapped ring.
#[test]
fn attn_prefill_sliding_ring_matches_reference() {
    let Some(ctx) = ctx() else { return };
    let kernel = ctx.load_kernel("attn_prefill_sliding_ring").unwrap();
    let mut rng = Rng::new(0xA7C);
    let m = 64usize;
    const WINDOW: usize = 1024;
    const RING: usize = 1024;
    let row = SL_KV_HEADS * SL_DIM;

    // q0 = history length: 0 cold; 100 partial ring; 1000 window saturates
    // mid-chunk (positions 1000..1063 cross 1023); 1531 wrapped ring.
    for q0 in [0usize, 100, 1000, 1531] {
        let l = q0 + m;
        let q = through_f16(&rng.f32_vec(m * N_Q_HEADS * SL_DIM));
        // Linear ground truth over all positions...
        let k_all = through_f16(&rng.f32_vec(l * row));
        let v_all = through_f16(&rng.f32_vec(l * row));
        // ...scattered into the two ranges the kernel reads: history rows
        // at ring slot pos % RING (pre-append state), chunk rows linear.
        let mut k_ring = vec![0.0f32; RING * row];
        let mut v_ring = vec![0.0f32; RING * row];
        for t in q0.saturating_sub(RING)..q0 {
            let slot = t % RING;
            k_ring[slot * row..][..row].copy_from_slice(&k_all[t * row..][..row]);
            v_ring[slot * row..][..row].copy_from_slice(&v_all[t * row..][..row]);
        }
        let k_chunk = &k_all[q0 * row..][..m * row];
        let v_chunk = &v_all[q0 * row..][..m * row];

        let buf16 = |data: &[f32]| {
            ctx.buffer_from_iter(to_f16_bits(data), BufferUsage::STORAGE_BUFFER)
                .unwrap()
        };
        let out_buf = ctx
            .new_buffer::<u16>((m * N_Q_HEADS * SL_DIM) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();

        ctx.dispatch_blocking(
            &kernel,
            vec![
                WriteDescriptorSet::buffer(0, buf16(&q)),
                WriteDescriptorSet::buffer(1, buf16(&k_ring)),
                WriteDescriptorSet::buffer(2, buf16(&v_ring)),
                WriteDescriptorSet::buffer(3, buf16(k_chunk)),
                WriteDescriptorSet::buffer(4, buf16(v_chunk)),
                WriteDescriptorSet::buffer(5, out_buf.clone()),
                WriteDescriptorSet::buffer(
                    6,
                    step_buf(
                        &ctx,
                        StepState {
                            q0: q0 as u32,
                            ..Default::default()
                        },
                    ),
                ),
            ],
            Some(SL_SCALE),
            [SL_KV_HEADS as u32, m as u32, 1],
        )
        .unwrap();
        let got = from_f16_bits(&out_buf.read().unwrap());

        for qh in [0usize, 1, 9, 16, 31] {
            for i in 0..m {
                let qpos = q0 + i;
                let t0 = (qpos + 1).saturating_sub(WINDOW);
                let want = attention_head(
                    &q,
                    &k_all,
                    &v_all,
                    i,
                    qh,
                    N_Q_HEADS,
                    SL_KV_HEADS,
                    SL_DIM,
                    SL_SCALE as f64,
                    t0,
                    qpos,
                );
                assert_close(
                    &got[(i * N_Q_HEADS + qh) * SL_DIM..][..SL_DIM],
                    &want,
                    ATOL,
                    RTOL,
                    &format!("prefill_sliding_ring q0={q0} i={i} qh={qh}"),
                );
            }
        }
    }
}

#[test]
fn attn_prefill_global_matches_reference() {
    let Some(ctx) = ctx() else { return };
    let kernel = ctx.load_kernel("attn_prefill_global").unwrap();
    let mut rng = Rng::new(0xA7A);
    let m = 64usize;

    for q0 in [0usize, 200] {
        let l = q0 + m;
        let q = through_f16(&rng.f32_vec(m * N_Q_HEADS * GL_DIM));
        let k = through_f16(&rng.f32_vec(l * GL_KV_HEADS * GL_DIM));
        let v = through_f16(&rng.f32_vec(l * GL_KV_HEADS * GL_DIM));

        let q_buf = ctx
            .buffer_from_iter(to_f16_bits(&q), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let k_buf = ctx
            .buffer_from_iter(to_f16_bits(&k), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let v_buf = ctx
            .buffer_from_iter(to_f16_bits(&v), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let out_buf = ctx
            .new_buffer::<u16>((m * N_Q_HEADS * GL_DIM) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();

        ctx.dispatch_blocking(
            &kernel,
            vec![
                WriteDescriptorSet::buffer(0, q_buf),
                WriteDescriptorSet::buffer(1, k_buf),
                WriteDescriptorSet::buffer(2, v_buf),
                WriteDescriptorSet::buffer(3, out_buf.clone()),
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
            Some(GL_SCALE),
            [GL_KV_HEADS as u32, m as u32, 1],
        )
        .unwrap();
        let got = from_f16_bits(&out_buf.read().unwrap());

        for qh in [0usize, 7, 15, 31] {
            for i in 0..m {
                let want = attention_head(
                    &q,
                    &k,
                    &v,
                    i,
                    qh,
                    N_Q_HEADS,
                    GL_KV_HEADS,
                    GL_DIM,
                    GL_SCALE as f64,
                    0,
                    q0 + i,
                );
                assert_close(
                    &got[(i * N_Q_HEADS + qh) * GL_DIM..][..GL_DIM],
                    &want,
                    ATOL,
                    RTOL,
                    &format!("prefill_global q0={q0} i={i} qh={qh}"),
                );
            }
        }
    }
}

/// Coopmat flash rewrite of the global prefill kernel (profile rank #1).
/// Same contract as `attn_prefill_global_matches_reference`, new dispatch:
/// one wave-sized workgroup per (query head, M_Q-row tile), grid
/// [N_Q_HEADS, m/M_Q]. The kernel coop-loads whole N_K-key tiles up to each
/// tile's causal bound, so the K/V buffers are padded to a tile boundary
/// (in the engine the resident KV buffer is over-allocated, so reads past
/// the logical end stay in-bounds; the masked positions never contribute).
#[test]
fn attn_prefill_global_flash_matches_reference() {
    const M_Q: usize = 16;
    const N_K: usize = 64;
    let Some(ctx) = ctx() else { return };
    let kernel = ctx.load_kernel("attn_prefill_global_flash").unwrap();
    let mut rng = Rng::new(0xA7C);
    let m = 64usize; // multiple of M_Q

    for q0 in [0usize, 200] {
        // Padded key count: the last query (row m−1) sees key q0+m−1, whose
        // N_K-tile may extend past it.
        let l_pad = ((q0 + m - 1) / N_K + 1) * N_K;
        let q = through_f16(&rng.f32_vec(m * N_Q_HEADS * GL_DIM));
        let k = through_f16(&rng.f32_vec(l_pad * GL_KV_HEADS * GL_DIM));
        let v = through_f16(&rng.f32_vec(l_pad * GL_KV_HEADS * GL_DIM));

        let q_buf = ctx
            .buffer_from_iter(to_f16_bits(&q), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let k_buf = ctx
            .buffer_from_iter(to_f16_bits(&k), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let v_buf = ctx
            .buffer_from_iter(to_f16_bits(&v), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let out_buf = ctx
            .new_buffer::<u16>((m * N_Q_HEADS * GL_DIM) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();

        ctx.dispatch_blocking(
            &kernel,
            vec![
                WriteDescriptorSet::buffer(0, q_buf),
                WriteDescriptorSet::buffer(1, k_buf),
                WriteDescriptorSet::buffer(2, v_buf),
                WriteDescriptorSet::buffer(3, out_buf.clone()),
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
            Some(GL_SCALE),
            [N_Q_HEADS as u32, (m / M_Q) as u32, 1],
        )
        .unwrap();
        let got = from_f16_bits(&out_buf.read().unwrap());

        for qh in [0usize, 7, 15, 31] {
            for i in 0..m {
                let want = attention_head(
                    &q,
                    &k,
                    &v,
                    i,
                    qh,
                    N_Q_HEADS,
                    GL_KV_HEADS,
                    GL_DIM,
                    GL_SCALE as f64,
                    0,
                    q0 + i,
                );
                assert_close(
                    &got[(i * N_Q_HEADS + qh) * GL_DIM..][..GL_DIM],
                    &want,
                    ATOL,
                    RTOL,
                    &format!("prefill_global_flash q0={q0} i={i} qh={qh}"),
                );
            }
        }
    }
}

/// Single-pass flash (design c): online softmax + in-register `O *= corr`
/// rescale via the coopmat-arith fork ops. Same contract/dispatch as the
/// two-pass `attn_prefill_global_flash_matches_reference`; the rescale path is
/// what's under test here.
#[test]
fn attn_prefill_global_flash_sp_matches_reference() {
    const M_Q: usize = 16;
    const N_K: usize = 64;
    let Some(ctx) = ctx() else { return };
    let kernel = ctx.load_kernel("attn_prefill_global_flash_sp").unwrap();
    let mut rng = Rng::new(0xA7C);
    let m = 64usize; // multiple of M_Q

    for q0 in [0usize, 200] {
        // Padded key count: the last query (row m−1) sees key q0+m−1, whose
        // N_K-tile may extend past it.
        let l_pad = ((q0 + m - 1) / N_K + 1) * N_K;
        let q = through_f16(&rng.f32_vec(m * N_Q_HEADS * GL_DIM));
        let k = through_f16(&rng.f32_vec(l_pad * GL_KV_HEADS * GL_DIM));
        let v = through_f16(&rng.f32_vec(l_pad * GL_KV_HEADS * GL_DIM));

        let q_buf = ctx
            .buffer_from_iter(to_f16_bits(&q), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let k_buf = ctx
            .buffer_from_iter(to_f16_bits(&k), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let v_buf = ctx
            .buffer_from_iter(to_f16_bits(&v), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let out_buf = ctx
            .new_buffer::<u16>((m * N_Q_HEADS * GL_DIM) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();

        ctx.dispatch_blocking(
            &kernel,
            vec![
                WriteDescriptorSet::buffer(0, q_buf),
                WriteDescriptorSet::buffer(1, k_buf),
                WriteDescriptorSet::buffer(2, v_buf),
                WriteDescriptorSet::buffer(3, out_buf.clone()),
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
            Some(GL_SCALE),
            [N_Q_HEADS as u32, (m / M_Q) as u32, 1],
        )
        .unwrap();
        let got = from_f16_bits(&out_buf.read().unwrap());

        for qh in [0usize, 7, 15, 31] {
            for i in 0..m {
                let want = attention_head(
                    &q,
                    &k,
                    &v,
                    i,
                    qh,
                    N_Q_HEADS,
                    GL_KV_HEADS,
                    GL_DIM,
                    GL_SCALE as f64,
                    0,
                    q0 + i,
                );
                assert_close(
                    &got[(i * N_Q_HEADS + qh) * GL_DIM..][..GL_DIM],
                    &want,
                    ATOL,
                    RTOL,
                    &format!("prefill_global_flash_sp q0={q0} i={i} qh={qh}"),
                );
            }
        }
    }
}

/// int8-matmul QKᵀ flash (Piece B): Q and K stream as Q8 (i8 + f16 scale), QKᵀ
/// is an int8 coopmat product with per-block rescale; V/PV stay f16. Checked
/// against the dequanted-value reference (the int8 dot is exact, so it matches
/// f16 attention on the Q8-roundtripped Q/K).
#[test]
fn attn_prefill_global_flash_sp_iq_matches_reference() {
    const M_Q: usize = 16;
    const N_K: usize = 64;
    let Some(ctx) = ctx() else { return };
    let kernel = ctx
        .load_kernel("attn_prefill_global_flash_sp_iq")
        .unwrap();
    let mut rng = Rng::new(0xA7C);
    let m = 64usize;

    for q0 in [0usize, 200] {
        let l_pad = ((q0 + m - 1) / N_K + 1) * N_K;
        let q = through_f16(&rng.f32_vec(m * N_Q_HEADS * GL_DIM));
        let k = through_f16(&rng.f32_vec(l_pad * GL_KV_HEADS * GL_DIM));
        let v = through_f16(&rng.f32_vec(l_pad * GL_KV_HEADS * GL_DIM));
        // Q8-quantize Q and K per 32-block along head-dim (the QKᵀ contraction);
        // the reference uses the dequanted values the kernel multiplies.
        let (q_quants, q_scales, q_deq) = q8_quant(&q);
        let (k_quants, k_scales, k_deq) = q8_quant(&k);

        let q_buf = ctx
            .buffer_from_iter(q_quants, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let qs_buf = ctx
            .buffer_from_iter(q_scales, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let k_buf = ctx
            .buffer_from_iter(k_quants, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let ks_buf = ctx
            .buffer_from_iter(k_scales, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let v_buf = ctx
            .buffer_from_iter(to_f16_bits(&v), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let out_buf = ctx
            .new_buffer::<u16>((m * N_Q_HEADS * GL_DIM) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();

        ctx.dispatch_blocking(
            &kernel,
            vec![
                WriteDescriptorSet::buffer(0, q_buf),
                WriteDescriptorSet::buffer(1, qs_buf),
                WriteDescriptorSet::buffer(2, k_buf),
                WriteDescriptorSet::buffer(3, ks_buf),
                WriteDescriptorSet::buffer(4, v_buf),
                WriteDescriptorSet::buffer(5, out_buf.clone()),
                WriteDescriptorSet::buffer(
                    6,
                    step_buf(
                        &ctx,
                        StepState {
                            q0: q0 as u32,
                            ..Default::default()
                        },
                    ),
                ),
            ],
            Some(GL_SCALE),
            [N_Q_HEADS as u32, (m / M_Q) as u32, 1],
        )
        .unwrap();
        let got = from_f16_bits(&out_buf.read().unwrap());

        for qh in [0usize, 7, 15, 31] {
            for i in 0..m {
                let want = attention_head(
                    &q_deq,
                    &k_deq,
                    &v,
                    i,
                    qh,
                    N_Q_HEADS,
                    GL_KV_HEADS,
                    GL_DIM,
                    GL_SCALE as f64,
                    0,
                    q0 + i,
                );
                assert_close(
                    &got[(i * N_Q_HEADS + qh) * GL_DIM..][..GL_DIM],
                    &want,
                    ATOL,
                    RTOL,
                    &format!("flash_sp_iq q0={q0} i={i} qh={qh}"),
                );
            }
        }
    }
}

/// Global prefill flash with int8 QKᵀ AND int8 PV (Piece B). V streams i8 from
/// the cache (per-key q8_quant_v scales) and P is quantized i8 in-kernel per
/// 32-key block. The reference uses the dequanted Q/K/V; P-quant error is
/// absorbed by the int8 tolerance (P∈[0,1], per-block ~0.4% relative).
#[test]
fn attn_prefill_global_flash_sp_ipv_matches_reference() {
    const M_Q: usize = 16;
    const N_K: usize = 64;
    let Some(ctx) = ctx() else { return };
    let kernel = ctx
        .load_kernel("attn_prefill_global_flash_sp_ipv")
        .unwrap();
    let mut rng = Rng::new(0xB7C);
    let m = 64usize;

    for q0 in [0usize, 200] {
        let l_pad = ((q0 + m - 1) / N_K + 1) * N_K;
        let q = through_f16(&rng.f32_vec(m * N_Q_HEADS * GL_DIM));
        let k = through_f16(&rng.f32_vec(l_pad * GL_KV_HEADS * GL_DIM));
        let v = through_f16(&rng.f32_vec(l_pad * GL_KV_HEADS * GL_DIM));
        let (q_quants, q_scales, q_deq) = q8_quant(&q);
        let (k_quants, k_scales, k_deq) = q8_quant(&k);
        // V quantized per 32-KEY block (the PV contraction axis); reference
        // multiplies the dequanted V the kernel actually reads.
        let (v_quants, v_scales, v_deq) = q8_quant_v(&v, l_pad, GL_KV_HEADS, GL_DIM);

        let q_buf = ctx
            .buffer_from_iter(q_quants, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let qs_buf = ctx
            .buffer_from_iter(q_scales, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let k_buf = ctx
            .buffer_from_iter(k_quants, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let ks_buf = ctx
            .buffer_from_iter(k_scales, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let vq_buf = ctx
            .buffer_from_iter(v_quants, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let vs_buf = ctx
            .buffer_from_iter(v_scales, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let out_buf = ctx
            .new_buffer::<u16>((m * N_Q_HEADS * GL_DIM) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();

        ctx.dispatch_blocking(
            &kernel,
            vec![
                WriteDescriptorSet::buffer(0, q_buf),
                WriteDescriptorSet::buffer(1, qs_buf),
                WriteDescriptorSet::buffer(2, k_buf),
                WriteDescriptorSet::buffer(3, ks_buf),
                WriteDescriptorSet::buffer(4, vq_buf),
                WriteDescriptorSet::buffer(5, vs_buf),
                WriteDescriptorSet::buffer(6, out_buf.clone()),
                WriteDescriptorSet::buffer(
                    7,
                    step_buf(
                        &ctx,
                        StepState {
                            q0: q0 as u32,
                            ..Default::default()
                        },
                    ),
                ),
            ],
            Some(GL_SCALE),
            [N_Q_HEADS as u32, (m / M_Q) as u32, 1],
        )
        .unwrap();
        let got = from_f16_bits(&out_buf.read().unwrap());

        for qh in [0usize, 7, 15, 31] {
            for i in 0..m {
                let want = attention_head(
                    &q_deq,
                    &k_deq,
                    &v_deq,
                    i,
                    qh,
                    N_Q_HEADS,
                    GL_KV_HEADS,
                    GL_DIM,
                    GL_SCALE as f64,
                    0,
                    q0 + i,
                );
                // The reference uses full-precision P; the kernel quantizes P
                // i8 per 32-key block, an error the reference doesn't model.
                // On adversarial random-sign V this reaches ~3e-2 abs on
                // near-cancellation output elements (worse than structured real
                // activations). Bound generously for kernel parity; real
                // quality is gated by prefill_parity + perplexity once wired.
                assert_close(
                    &got[(i * N_Q_HEADS + qh) * GL_DIM..][..GL_DIM],
                    &want,
                    4e-2,
                    RTOL,
                    &format!("flash_sp_ipv q0={q0} i={i} qh={qh}"),
                );
            }
        }
    }
}

/// CPU-only (no GPU): the per-key V Q8 reference for Piece B (int8 PV). Pins
/// the key-axis blocking, key-blocked scale layout, 4/u32 packing, and dequant
/// consistency — `keys` deliberately not a multiple of 32 to hit the short
/// final block.
#[test]
fn q8_quant_v_reference_is_consistent() {
    let mut rng = Rng::new(0x5EED);
    let (keys, nkv, hd) = (70usize, GL_KV_HEADS, GL_DIM);
    let v = through_f16(&rng.f32_vec(keys * nkv * hd));
    let (quants, scales, deq) = q8_quant_v(&v, keys, nkv, hd);

    let kblk = keys.div_ceil(32);
    assert_eq!(quants.len(), v.len() / 4);
    assert_eq!(scales.len(), kblk * nkv * hd);
    assert_eq!(deq.len(), v.len());

    for h in 0..nkv {
        for c in 0..hd {
            for kb in 0..kblk {
                let (k0, k1) = (kb * 32, (kb * 32 + 32).min(keys));
                let dd = half::f16::from_bits(scales[(kb * nkv + h) * hd + c]).to_f32();
                // The scale is the amax over this 32-KEY block (not head-dim):
                // this is what makes the per-block scale factor out of the PV
                // key-contraction.
                let amax = (k0..k1).fold(0f32, |a, key| a.max(v[(key * nkv + h) * hd + c].abs()));
                assert_eq!(dd, half::f16::from_f32(amax / 127.0).to_f32(), "scale {kb},{h},{c}");
                for key in k0..k1 {
                    let idx = (key * nkv + h) * hd + c;
                    let qi = (quants[idx / 4] >> ((idx % 4) * 8)) as u8 as i8;
                    // dequant == unpacked i8 × the block scale, exactly.
                    assert_eq!(deq[idx], dd * qi as f32, "deq {idx}");
                    // reconstruction within one quant step of the f16'd input.
                    assert!((v[idx] - deq[idx]).abs() <= dd + 1e-4, "recon {idx}");
                }
            }
        }
    }
}

/// Global decode against the Q8 K cache (Piece A): K is dequanted from i8 +
/// f16 scales in-kernel; the reference uses the same dequanted K. V/Q f16.
#[test]
fn attn_decode_global_q8k_matches_reference() {
    let Some(ctx) = ctx() else { return };
    let part_k = ctx.load_kernel("attn_decode_global_q8k").unwrap();
    let red_k = ctx.load_kernel("attn_reduce_d512").unwrap();
    let mut rng = Rng::new(0xA79);

    for (kv_len, n_splits) in [(333usize, 1u32), (333, 5), (3, 8), (4096, 4)] {
        let q = through_f16(&rng.f32_vec(N_Q_HEADS * GL_DIM));
        let k = through_f16(&rng.f32_vec(kv_len * GL_KV_HEADS * GL_DIM));
        let v = through_f16(&rng.f32_vec(kv_len * GL_KV_HEADS * GL_DIM));
        // Q8-quantize K per 32-block along head-dim (one (token,head) row =
        // HD_BLOCKS=16 contiguous blocks); the reference uses k_deq.
        let (k_quants, k_scales, k_deq) = q8_quant(&k);

        let q_buf = ctx
            .buffer_from_iter(to_f16_bits(&q), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let kq_buf = ctx
            .buffer_from_iter(k_quants, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let ks_buf = ctx
            .buffer_from_iter(k_scales, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let v_buf = ctx
            .buffer_from_iter(to_f16_bits(&v), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let part_buf = ctx
            .new_buffer::<f32>(
                (N_Q_HEADS * n_splits as usize * GL_PART_STRIDE) as u64,
                BufferUsage::STORAGE_BUFFER,
            )
            .unwrap();
        let out_buf = ctx
            .new_buffer::<u16>((N_Q_HEADS * GL_DIM) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();

        ctx.dispatch_blocking(
            &part_k,
            vec![
                WriteDescriptorSet::buffer(0, q_buf),
                WriteDescriptorSet::buffer(1, kq_buf),
                WriteDescriptorSet::buffer(2, ks_buf),
                WriteDescriptorSet::buffer(3, v_buf),
                WriteDescriptorSet::buffer(4, part_buf.clone()),
                WriteDescriptorSet::buffer(
                    5,
                    step_buf(
                        &ctx,
                        StepState {
                            kv_len_global: kv_len as u32,
                            ..Default::default()
                        },
                    ),
                ),
            ],
            Some(PushSplitScale {
                n_splits,
                scale: GL_SCALE,
            }),
            [GL_KV_HEADS as u32, n_splits, 1],
        )
        .unwrap();
        ctx.dispatch_blocking(
            &red_k,
            vec![
                WriteDescriptorSet::buffer(0, part_buf),
                WriteDescriptorSet::buffer(1, out_buf.clone()),
            ],
            Some(n_splits),
            [N_Q_HEADS as u32, 1, 1],
        )
        .unwrap();
        let got = from_f16_bits(&out_buf.read().unwrap());

        for qh in 0..N_Q_HEADS {
            let want = attention_head(
                &q,
                &k_deq,
                &v,
                0,
                qh,
                N_Q_HEADS,
                GL_KV_HEADS,
                GL_DIM,
                GL_SCALE as f64,
                0,
                kv_len - 1,
            );
            assert_close(
                &got[qh * GL_DIM..][..GL_DIM],
                &want,
                ATOL,
                RTOL,
                &format!("decode_global_q8k kv_len={kv_len} splits={n_splits} qh={qh}"),
            );
        }
    }
}

/// Plan 02/06: bit-identical across runs (fixed split count and order).
#[test]
fn attn_is_bit_deterministic() {
    let Some(ctx) = ctx() else { return };
    let part_k = ctx.load_kernel("attn_decode_global").unwrap();
    let red_k = ctx.load_kernel("attn_reduce_d512").unwrap();
    let pre_k = ctx.load_kernel("attn_prefill_sliding").unwrap();
    let mut rng = Rng::new(0xA7B);

    let (kv_len, n_splits) = (1000usize, 3u32);
    let q = to_f16_bits(&rng.f32_vec(N_Q_HEADS * GL_DIM));
    let kv_k = to_f16_bits(&rng.f32_vec(kv_len * GL_KV_HEADS * GL_DIM));
    let kv_v = to_f16_bits(&rng.f32_vec(kv_len * GL_KV_HEADS * GL_DIM));
    let m = 16usize;
    let pq = to_f16_bits(&rng.f32_vec(m * N_Q_HEADS * SL_DIM));
    let pk = to_f16_bits(&rng.f32_vec(m * SL_KV_HEADS * SL_DIM));
    let pv = to_f16_bits(&rng.f32_vec(m * SL_KV_HEADS * SL_DIM));

    let mut outs = Vec::new();
    for _ in 0..2 {
        let q_buf = ctx
            .buffer_from_iter(q.iter().copied(), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let k_buf = ctx
            .buffer_from_iter(kv_k.iter().copied(), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let v_buf = ctx
            .buffer_from_iter(kv_v.iter().copied(), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let part_buf = ctx
            .new_buffer::<f32>(
                (N_Q_HEADS * n_splits as usize * GL_PART_STRIDE) as u64,
                BufferUsage::STORAGE_BUFFER,
            )
            .unwrap();
        let out_buf = ctx
            .new_buffer::<u16>((N_Q_HEADS * GL_DIM) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        ctx.dispatch_blocking(
            &part_k,
            vec![
                WriteDescriptorSet::buffer(0, q_buf),
                WriteDescriptorSet::buffer(1, k_buf),
                WriteDescriptorSet::buffer(2, v_buf),
                WriteDescriptorSet::buffer(3, part_buf.clone()),
                WriteDescriptorSet::buffer(
                    4,
                    step_buf(
                        &ctx,
                        StepState {
                            kv_len_global: kv_len as u32,
                            ..Default::default()
                        },
                    ),
                ),
            ],
            Some(PushSplitScale {
                n_splits,
                scale: GL_SCALE,
            }),
            [GL_KV_HEADS as u32, n_splits, 1],
        )
        .unwrap();
        ctx.dispatch_blocking(
            &red_k,
            vec![
                WriteDescriptorSet::buffer(0, part_buf),
                WriteDescriptorSet::buffer(1, out_buf.clone()),
            ],
            Some(n_splits),
            [N_Q_HEADS as u32, 1, 1],
        )
        .unwrap();

        let pq_buf = ctx
            .buffer_from_iter(pq.iter().copied(), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let pk_buf = ctx
            .buffer_from_iter(pk.iter().copied(), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let pv_buf = ctx
            .buffer_from_iter(pv.iter().copied(), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let pout_buf = ctx
            .new_buffer::<u16>((m * N_Q_HEADS * SL_DIM) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        ctx.dispatch_blocking(
            &pre_k,
            vec![
                WriteDescriptorSet::buffer(0, pq_buf),
                WriteDescriptorSet::buffer(1, pk_buf),
                WriteDescriptorSet::buffer(2, pv_buf),
                WriteDescriptorSet::buffer(3, pout_buf.clone()),
                WriteDescriptorSet::buffer(4, step_buf(&ctx, StepState::default())),
            ],
            Some(SL_SCALE),
            [SL_KV_HEADS as u32, m as u32, 1],
        )
        .unwrap();

        let mut combined = out_buf.read().unwrap().to_vec();
        combined.extend_from_slice(&pout_buf.read().unwrap());
        outs.push(combined);
    }
    assert_eq!(outs[0], outs[1]);
}
