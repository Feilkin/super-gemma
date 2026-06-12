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

use reference::{Rng, assert_close, attention_head, from_f16_bits, through_f16, to_f16_bits};
use sg_gpu::GpuContext;
use vulkano::buffer::{BufferContents, BufferUsage};
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
struct PushLenScale {
    len: u32,
    scale: f32,
}

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct PushSplit {
    kv_len: u32,
    n_splits: u32,
    scale: f32,
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
            ],
            Some(PushSplit {
                kv_len: kv_len as u32,
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
        let kv = through_f16(&rng.f32_vec(kv_len * GL_KV_HEADS * GL_DIM));

        let q_buf = ctx
            .buffer_from_iter(to_f16_bits(&q), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let kv_buf = ctx
            .buffer_from_iter(to_f16_bits(&kv), BufferUsage::STORAGE_BUFFER)
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
                WriteDescriptorSet::buffer(1, kv_buf),
                WriteDescriptorSet::buffer(2, part_buf.clone()),
            ],
            Some(PushSplit {
                kv_len: kv_len as u32,
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
                &kv,
                &kv, // K = V
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
            ],
            Some(PushLenScale {
                len: q0 as u32,
                scale: SL_SCALE,
            }),
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

#[test]
fn attn_prefill_global_matches_reference() {
    let Some(ctx) = ctx() else { return };
    let kernel = ctx.load_kernel("attn_prefill_global").unwrap();
    let mut rng = Rng::new(0xA7A);
    let m = 64usize;

    for q0 in [0usize, 200] {
        let l = q0 + m;
        let q = through_f16(&rng.f32_vec(m * N_Q_HEADS * GL_DIM));
        let kv = through_f16(&rng.f32_vec(l * GL_KV_HEADS * GL_DIM));

        let q_buf = ctx
            .buffer_from_iter(to_f16_bits(&q), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let kv_buf = ctx
            .buffer_from_iter(to_f16_bits(&kv), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let out_buf = ctx
            .new_buffer::<u16>((m * N_Q_HEADS * GL_DIM) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();

        ctx.dispatch_blocking(
            &kernel,
            vec![
                WriteDescriptorSet::buffer(0, q_buf),
                WriteDescriptorSet::buffer(1, kv_buf),
                WriteDescriptorSet::buffer(2, out_buf.clone()),
            ],
            Some(PushLenScale {
                len: q0 as u32,
                scale: GL_SCALE,
            }),
            [GL_KV_HEADS as u32, m as u32, 1],
        )
        .unwrap();
        let got = from_f16_bits(&out_buf.read().unwrap());

        for qh in [0usize, 7, 15, 31] {
            for i in 0..m {
                let want = attention_head(
                    &q,
                    &kv,
                    &kv,
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
    let kv = to_f16_bits(&rng.f32_vec(kv_len * GL_KV_HEADS * GL_DIM));
    let m = 16usize;
    let pq = to_f16_bits(&rng.f32_vec(m * N_Q_HEADS * SL_DIM));
    let pk = to_f16_bits(&rng.f32_vec(m * SL_KV_HEADS * SL_DIM));
    let pv = to_f16_bits(&rng.f32_vec(m * SL_KV_HEADS * SL_DIM));

    let mut outs = Vec::new();
    for _ in 0..2 {
        let q_buf = ctx
            .buffer_from_iter(q.iter().copied(), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let kv_buf = ctx
            .buffer_from_iter(kv.iter().copied(), BufferUsage::STORAGE_BUFFER)
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
                WriteDescriptorSet::buffer(1, kv_buf),
                WriteDescriptorSet::buffer(2, part_buf.clone()),
            ],
            Some(PushSplit {
                kv_len: kv_len as u32,
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
            ],
            Some(PushLenScale {
                len: 0,
                scale: SL_SCALE,
            }),
            [SL_KV_HEADS as u32, m as u32, 1],
        )
        .unwrap();

        let mut combined = out_buf.read().unwrap().to_vec();
        combined.extend_from_slice(&pout_buf.read().unwrap());
        outs.push(combined);
    }
    assert_eq!(outs[0], outs[1]);
}
