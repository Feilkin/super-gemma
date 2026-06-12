//! The GPU forward graph, decode-shaped (plan 03): the per-layer dispatch
//! sequence over `sg-gpu` kernels, recordable once and driven per token via
//! the step buffer + CPU-rewritten embedding/rope-table buffers (the M2.7
//! contract). Semantics follow `docs/reference/gemma4-forward-graph.md`;
//! parity vs [`crate::CpuModel`] is the M3 gate.
//!
//! Per layer:
//! ```text
//! rmsnorm(attn_norm) → gemv q/k[/v] → qk-norm + V-norm(ones) → rope q,k
//!   → kv_append k,v → split-K attention + reduce → gemv o
//!   → rmsnorm(post_attention) → add_scaled(×1.0)
//!   → rmsnorm(ffn) → gemv gate,up → geglu → gemv down
//!   → rmsnorm(post_ffw) → add_scaled(×layer_output_scale)
//! ```
//! Then `rmsnorm(output_norm)` → Q6_K LM head (softcap fused).
//!
//! Split-K counts are baked at record time (plan 03: part of the
//! determinism contract); kv_len growth only shrinks the recorded splits'
//! chunks via the step buffer.

use std::ops::Range;

use half::f16;
use sg_gguf::{GgmlType, Gguf, LayerKind, ModelDesc, q6_k};
use sg_gpu::{
    BufferUsage, CommandGraph, GpuContext, GpuError, GraphRecorder, Kernel, StepState, Subbuffer,
    WriteDescriptorSet,
};

use crate::reference::RefError;
use crate::rope::cos_sin_table;
use crate::weights::{GpuWeights, UploadError};

/// Decode split counts, baked at record time (plan 03 defaults; revisit
/// with the e2e profile).
const SPLITS_SLIDING: u32 = 16;
const SPLITS_GLOBAL: u32 = 32;

const HIDDEN: usize = 5376;
const FFN: usize = 21504;

/// A single-token (decode-shaped) GPU forward pass over the whole stack.
///
/// Holds every buffer the recorded graph binds; the CPU rewrites the small
/// dynamic ones (embedding row, step state, rope tables) between submits.
pub struct GpuModel<'a> {
    ctx: &'a GpuContext,
    gguf: &'a Gguf<'a>,
    pub desc: ModelDesc,
    pub weights: GpuWeights,
    k: Kernels,
    b: Bufs,
    /// Per-layer KV stores: sliding = 1024-slot ring, global = linear.
    kv: Vec<KvStore>,
    step: Subbuffer<[u32]>,
    /// (cos, sin) for the current position: 128 live pairs sliding.
    cs_sliding: Subbuffer<[f32]>,
    /// 64 live pairs global (the frozen 192 are identities, never stored).
    cs_global: Subbuffer<[f32]>,
    /// Absolute position of the NEXT token to decode.
    pub pos: u32,
}

struct KvStore {
    k: Subbuffer<[u16]>,
    v: Subbuffer<[u16]>,
}

struct Kernels {
    rms5376: Kernel,
    rms512: Kernel,
    rms256: Kernel,
    rope_sl_q: Kernel,
    rope_sl_k: Kernel,
    rope_gl_q: Kernel,
    rope_gl_k: Kernel,
    gemv5376: Kernel,
    gemv8192: Kernel,
    gemv16384: Kernel,
    gemv21504: Kernel,
    geglu: Kernel,
    add: Kernel,
    append_sl: Kernel,
    append_gl: Kernel,
    attn_sl: Kernel,
    attn_gl: Kernel,
    reduce256: Kernel,
    reduce512: Kernel,
    logits: Kernel,
}

/// Activation buffers (single token). Sized per layer kind where the
/// kind's geometry differs — kernels derive bounds from `arrayLength`.
struct Bufs {
    x: Subbuffer<[u16]>,
    xn: Subbuffer<[u16]>,
    q_raw_sl: Subbuffer<[u16]>,
    q_sl: Subbuffer<[u16]>,
    q_raw_gl: Subbuffer<[u16]>,
    q_gl: Subbuffer<[u16]>,
    kp_sl: Subbuffer<[u16]>,
    k_sl: Subbuffer<[u16]>,
    vp_sl: Subbuffer<[u16]>,
    v_sl: Subbuffer<[u16]>,
    kp_gl: Subbuffer<[u16]>,
    k_gl: Subbuffer<[u16]>,
    v_gl: Subbuffer<[u16]>,
    attn_sl: Subbuffer<[u16]>,
    attn_gl: Subbuffer<[u16]>,
    part_sl: Subbuffer<[f32]>,
    part_gl: Subbuffer<[f32]>,
    o: Subbuffer<[u16]>,
    on: Subbuffer<[u16]>,
    x2: Subbuffer<[u16]>,
    fin: Subbuffer<[u16]>,
    g: Subbuffer<[u16]>,
    u: Subbuffer<[u16]>,
    gu: Subbuffer<[u16]>,
    f: Subbuffer<[u16]>,
    fn2: Subbuffer<[u16]>,
    logits: Subbuffer<[f32]>,
}

impl<'a> GpuModel<'a> {
    /// Upload weights and allocate all graph state. `global_cap` bounds the
    /// linear global-KV stores (tokens); tests keep it small, the engine
    /// will size it from the configured context limit.
    pub fn new(
        ctx: &'a GpuContext,
        gguf: &'a Gguf<'a>,
        global_cap: usize,
    ) -> Result<Self, UploadError> {
        let desc = ModelDesc::from_gguf(gguf).map_err(|e| RefError::BadTensor {
            name: "<model>".into(),
            what: e.to_string(),
        })?;
        let weights = GpuWeights::upload(ctx, gguf, &desc)?;

        let usage = BufferUsage::STORAGE_BUFFER;
        let f16buf = |len: usize| ctx.new_buffer::<u16>(len as u64, usage);
        let q_dim_sl = 32 * desc.sliding.head_dim; // 8192
        let q_dim_gl = 32 * desc.global.head_dim; // 16384
        let kv_dim_sl = desc.sliding.n_kv_heads * desc.sliding.head_dim; // 4096
        let kv_dim_gl = desc.global.n_kv_heads * desc.global.head_dim; // 2048

        let b = Bufs {
            x: f16buf(HIDDEN)?,
            xn: f16buf(HIDDEN)?,
            q_raw_sl: f16buf(q_dim_sl)?,
            q_sl: f16buf(q_dim_sl)?,
            q_raw_gl: f16buf(q_dim_gl)?,
            q_gl: f16buf(q_dim_gl)?,
            kp_sl: f16buf(kv_dim_sl)?,
            k_sl: f16buf(kv_dim_sl)?,
            vp_sl: f16buf(kv_dim_sl)?,
            v_sl: f16buf(kv_dim_sl)?,
            kp_gl: f16buf(kv_dim_gl)?,
            k_gl: f16buf(kv_dim_gl)?,
            v_gl: f16buf(kv_dim_gl)?,
            attn_sl: f16buf(q_dim_sl)?,
            attn_gl: f16buf(q_dim_gl)?,
            part_sl: ctx.new_buffer::<f32>(
                (32 * SPLITS_SLIDING as usize * (desc.sliding.head_dim + 2)) as u64,
                usage,
            )?,
            part_gl: ctx.new_buffer::<f32>(
                (32 * SPLITS_GLOBAL as usize * (desc.global.head_dim + 2)) as u64,
                usage,
            )?,
            o: f16buf(HIDDEN)?,
            on: f16buf(HIDDEN)?,
            x2: f16buf(HIDDEN)?,
            fin: f16buf(HIDDEN)?,
            g: f16buf(FFN)?,
            u: f16buf(FFN)?,
            gu: f16buf(FFN)?,
            f: f16buf(HIDDEN)?,
            fn2: f16buf(HIDDEN)?,
            logits: ctx.new_buffer::<f32>(desc.vocab_size as u64, usage)?,
        };

        let kv = desc
            .layer_kinds
            .iter()
            .map(|kind| {
                let slots = match kind {
                    LayerKind::Sliding => desc.sliding_window * kv_dim_sl,
                    LayerKind::Global => global_cap * kv_dim_gl,
                };
                Ok(KvStore {
                    k: f16buf(slots)?,
                    v: f16buf(slots)?,
                })
            })
            .collect::<Result<Vec<_>, GpuError>>()?;

        let load = |n: &str| ctx.load_kernel(n);
        let k = Kernels {
            rms5376: load("rmsnorm_5376")?,
            rms512: load("rmsnorm_512")?,
            rms256: load("rmsnorm_256")?,
            rope_sl_q: load("rope_sliding_q")?,
            rope_sl_k: load("rope_sliding_k")?,
            rope_gl_q: load("rope_global_q")?,
            rope_gl_k: load("rope_global_k")?,
            gemv5376: load("gemv_q4_0_k5376")?,
            gemv8192: load("gemv_q4_0_k8192")?,
            gemv16384: load("gemv_q4_0_k16384")?,
            gemv21504: load("gemv_q4_0_k21504")?,
            geglu: load("geglu")?,
            add: load("add_scaled")?,
            append_sl: load("kv_append_sliding")?,
            append_gl: load("kv_append_global")?,
            attn_sl: load("attn_decode_sliding")?,
            attn_gl: load("attn_decode_global")?,
            reduce256: load("attn_reduce_d256")?,
            reduce512: load("attn_reduce_d512")?,
            logits: load("gemv_q6_k_logits")?,
        };

        let cs_sliding = ctx.new_buffer::<f32>((desc.sliding.head_dim / 2 * 2) as u64, usage)?;
        let cs_global = ctx.new_buffer::<f32>((desc.global.head_dim / 8 * 2) as u64, usage)?;
        Ok(Self {
            ctx,
            gguf,
            desc,
            weights,
            k,
            b,
            kv,
            step: ctx.new_step_buffer()?,
            cs_sliding,
            cs_global,
            pos: 0,
        })
    }

    /// Record a graph covering `layers` (and optionally the final norm +
    /// LM head). The full decode graph is `record(0..60, true)`; the
    /// per-layer parity harness records narrower ranges and inspects
    /// buffers between submits.
    pub fn record(
        &self,
        layers: Range<usize>,
        with_logits: bool,
    ) -> Result<CommandGraph, GpuError> {
        self.ctx.record_graph(|rec| {
            for i in layers.clone() {
                self.record_layer(rec, i)?;
            }
            if with_logits {
                self.record_logits(rec)?;
            }
            Ok(())
        })
    }

    fn record_layer(&self, rec: &mut GraphRecorder<'_>, i: usize) -> Result<(), GpuError> {
        let lw = &self.weights.layers[i];
        let b = &self.b;
        let kv = &self.kv[i];
        let sliding = lw.kind == LayerKind::Sliding;
        let (hd, n_kv) = match lw.kind {
            LayerKind::Sliding => (self.desc.sliding.head_dim, self.desc.sliding.n_kv_heads),
            LayerKind::Global => (self.desc.global.head_dim, self.desc.global.n_kv_heads),
        };
        let q_dim = 32 * hd;
        let kv_dim = n_kv * hd;
        let (q_raw, q, kp, k, v) = if sliding {
            (&b.q_raw_sl, &b.q_sl, &b.kp_sl, &b.k_sl, &b.v_sl)
        } else {
            (&b.q_raw_gl, &b.q_gl, &b.kp_gl, &b.k_gl, &b.v_gl)
        };
        let (rms_hd, rope_q, rope_k) = if sliding {
            (&self.k.rms256, &self.k.rope_sl_q, &self.k.rope_sl_k)
        } else {
            (&self.k.rms512, &self.k.rope_gl_q, &self.k.rope_gl_k)
        };
        let cs = if sliding {
            &self.cs_sliding
        } else {
            &self.cs_global
        };
        let wo_gemv = if sliding {
            &self.k.gemv8192
        } else {
            &self.k.gemv16384
        };

        let no_push = None::<u32>;

        // Attention block.
        rms(rec, &self.k.rms5376, &b.x, &lw.attn_norm, &b.xn, 1)?;
        rec.dispatch(
            &self.k.gemv5376,
            vec![
                buf(0, lw.attn_q.clone()),
                buf(1, b.xn.clone()),
                buf(2, q_raw.clone()),
            ],
            no_push,
            [q_dim as u32, 1, 1],
        )?;
        rec.dispatch(
            &self.k.gemv5376,
            vec![
                buf(0, lw.attn_k.clone()),
                buf(1, b.xn.clone()),
                buf(2, kp.clone()),
            ],
            no_push,
            [kv_dim as u32, 1, 1],
        )?;
        // V projection: own tensor on sliding layers, the K projection
        // output on global ones (they diverge through the norms below).
        let vp = match &lw.attn_v {
            Some(wv) => {
                rec.dispatch(
                    &self.k.gemv5376,
                    vec![
                        buf(0, wv.clone()),
                        buf(1, b.xn.clone()),
                        buf(2, b.vp_sl.clone()),
                    ],
                    no_push,
                    [kv_dim as u32, 1, 1],
                )?;
                &b.vp_sl
            }
            None => kp,
        };
        rms(rec, rms_hd, q_raw, &lw.attn_q_norm, q, 32)?;
        rms(rec, rms_hd, kp, &lw.attn_k_norm, k, n_kv)?;
        rms(rec, rms_hd, vp, &self.weights.norm_ones, v, n_kv)?; // V-norm: weightless
        // Live rotation pairs per head: full head for sliding (ROT_DIMS =
        // 256), the unfrozen quarter for global (ROT_DIMS = 128) — must
        // match the rope variant defines in sg-gpu's build.rs.
        let live_pairs = if sliding { hd / 2 } else { hd / 8 };
        let rope_pairs = |heads: usize| (heads * live_pairs) as u64;
        rec.dispatch(
            rope_q,
            vec![buf(0, q.clone()), buf(1, cs.clone())],
            no_push,
            rope_q.groups_for(rope_pairs(32)),
        )?;
        rec.dispatch(
            rope_k,
            vec![buf(0, k.clone()), buf(1, cs.clone())],
            no_push,
            rope_k.groups_for(rope_pairs(n_kv)),
        )?;
        let append = if sliding {
            &self.k.append_sl
        } else {
            &self.k.append_gl
        };
        for (src, dst) in [(k, &kv.k), (v, &kv.v)] {
            rec.dispatch(
                append,
                vec![
                    buf(0, src.clone()),
                    buf(1, dst.clone()),
                    buf(2, self.step.clone()),
                ],
                no_push,
                append.groups_for(kv_dim as u64),
            )?;
        }
        let (attn_k, red_k, part, attn_out, n_splits) = if sliding {
            (
                &self.k.attn_sl,
                &self.k.reduce256,
                &b.part_sl,
                &b.attn_sl,
                SPLITS_SLIDING,
            )
        } else {
            (
                &self.k.attn_gl,
                &self.k.reduce512,
                &b.part_gl,
                &b.attn_gl,
                SPLITS_GLOBAL,
            )
        };
        // Push = { n_splits: u32, scale: f32 }, as two words (scale 1.0 —
        // QK-norm replaces 1/√d, pinned).
        rec.dispatch(
            attn_k,
            vec![
                buf(0, q.clone()),
                buf(1, kv.k.clone()),
                buf(2, kv.v.clone()),
                buf(3, part.clone()),
                buf(4, self.step.clone()),
            ],
            Some([n_splits, 1.0f32.to_bits()]),
            [n_kv as u32, n_splits, 1],
        )?;
        rec.dispatch(
            red_k,
            vec![buf(0, part.clone()), buf(1, attn_out.clone())],
            Some(n_splits),
            [32, 1, 1],
        )?;
        rec.dispatch(
            wo_gemv,
            vec![
                buf(0, lw.attn_output.clone()),
                buf(1, attn_out.clone()),
                buf(2, b.o.clone()),
            ],
            no_push,
            [HIDDEN as u32, 1, 1],
        )?;
        rms(
            rec,
            &self.k.rms5376,
            &b.o,
            &lw.post_attention_norm,
            &b.on,
            1,
        )?;
        rec.dispatch(
            &self.k.add,
            vec![
                buf(0, b.x.clone()),
                buf(1, b.on.clone()),
                buf(2, b.x2.clone()),
            ],
            Some(1.0f32),
            self.k.add.groups_for(HIDDEN as u64),
        )?;

        // FFN block.
        rms(rec, &self.k.rms5376, &b.x2, &lw.ffn_norm, &b.fin, 1)?;
        for (w, dst) in [(&lw.ffn_gate, &b.g), (&lw.ffn_up, &b.u)] {
            rec.dispatch(
                &self.k.gemv5376,
                vec![
                    buf(0, w.clone()),
                    buf(1, b.fin.clone()),
                    buf(2, (*dst).clone()),
                ],
                no_push,
                [FFN as u32, 1, 1],
            )?;
        }
        rec.dispatch(
            &self.k.geglu,
            vec![
                buf(0, b.g.clone()),
                buf(1, b.u.clone()),
                buf(2, b.gu.clone()),
            ],
            no_push,
            self.k.geglu.groups_for(FFN as u64),
        )?;
        rec.dispatch(
            &self.k.gemv21504,
            vec![
                buf(0, lw.ffn_down.clone()),
                buf(1, b.gu.clone()),
                buf(2, b.f.clone()),
            ],
            no_push,
            [HIDDEN as u32, 1, 1],
        )?;
        rms(rec, &self.k.rms5376, &b.f, &lw.post_ffw_norm, &b.fn2, 1)?;
        // Residual + layer_output_scale, back into x for the next layer.
        rec.dispatch(
            &self.k.add,
            vec![
                buf(0, b.x2.clone()),
                buf(1, b.fn2.clone()),
                buf(2, b.x.clone()),
            ],
            Some(lw.layer_output_scale),
            self.k.add.groups_for(HIDDEN as u64),
        )?;
        Ok(())
    }

    fn record_logits(&self, rec: &mut GraphRecorder<'_>) -> Result<(), GpuError> {
        rms(
            rec,
            &self.k.rms5376,
            &self.b.x,
            &self.weights.output_norm,
            &self.b.xn,
            1,
        )?;
        rec.dispatch(
            &self.k.logits,
            vec![
                buf(0, self.weights.token_embd.clone()),
                buf(1, self.b.xn.clone()),
                buf(2, self.b.logits.clone()),
            ],
            None::<u32>,
            [self.desc.vocab_size as u32, 1, 1],
        )
        .map(|_| ())
    }

    /// CPU-side per-token staging (plan 03 decode loop step 1): embedding
    /// row into `x`, step state, rope tables for the current `pos`.
    pub fn stage_token(&self, token: u32) -> Result<(), GpuError> {
        let row = self.embed_row(token);
        {
            let mut w = self
                .b
                .x
                .write()
                .map_err(|e| GpuError::Validation(e.to_string()))?;
            for (dst, &val) in w.iter_mut().zip(&row) {
                *dst = f16::from_f32(val).to_bits();
            }
        }
        StepState {
            pos: self.pos,
            kv_len_sliding: (self.pos + 1).min(self.desc.sliding_window as u32),
            kv_len_global: self.pos + 1,
            q0: 0,
        }
        .write_to(&self.step)?;

        let write_cs = |buf: &Subbuffer<[f32]>, table: &[f32]| -> Result<(), GpuError> {
            let mut w = buf
                .write()
                .map_err(|e| GpuError::Validation(e.to_string()))?;
            w.copy_from_slice(table);
            Ok(())
        };
        let sl = cos_sin_table(
            &[self.pos],
            self.desc.sliding.head_dim,
            self.desc.sliding.rope_theta,
            None,
        );
        write_cs(&self.cs_sliding, &sl)?;
        let gl = cos_sin_table(
            &[self.pos],
            self.desc.global.head_dim,
            self.desc.global.rope_theta,
            Some(&self.weights.rope_factors),
        );
        // Only the live pairs (first quarter) are stored; the frozen tail
        // pairs are identities the kernel never touches.
        write_cs(&self.cs_global, &gl[..self.desc.global.head_dim / 8 * 2])?;
        Ok(())
    }

    /// Dequantized, scaled embedding row (CPU-side lookup, plan 03).
    fn embed_row(&self, token: u32) -> Vec<f32> {
        let info = self.gguf.tensor("token_embd.weight").expect("validated");
        debug_assert_eq!(info.dtype, GgmlType::Q6_K);
        let row_blocks = HIDDEN / q6_k::QK6_K;
        let row_bytes = row_blocks * q6_k::BLOCK_Q6_K_SIZE;
        let data = &self.gguf.data_of(info)[token as usize * row_bytes..][..row_bytes];
        let blocks = q6_k::blocks_from_bytes(data).expect("validated size");
        let scale = (HIDDEN as f32).sqrt();
        let mut row = vec![0.0f32; HIDDEN];
        let mut tmp = [0.0f32; q6_k::QK6_K];
        for (bi, block) in blocks.iter().enumerate() {
            block.dequantize(&mut tmp);
            for (dst, &val) in row[bi * q6_k::QK6_K..][..q6_k::QK6_K].iter_mut().zip(&tmp) {
                *dst = val * scale;
            }
        }
        row
    }

    pub fn submit(&self, graph: &CommandGraph) -> Result<(), GpuError> {
        self.ctx.submit_blocking(graph)
    }

    /// Advance to the next position (call once per fully-processed token).
    pub fn advance(&mut self) {
        self.pos += 1;
    }

    /// Restart from position 0. KV store contents become stale, which is
    /// safe: kv_len masks them and re-decoding overwrites the same slots.
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Hidden state `x` as f32 (the residual stream between layers).
    pub fn read_hidden(&self) -> Result<Vec<f32>, GpuError> {
        let r = self
            .b
            .x
            .read()
            .map_err(|e| GpuError::Validation(e.to_string()))?;
        Ok(r.iter().map(|&b| f16::from_bits(b).to_f32()).collect())
    }

    pub fn read_logits(&self) -> Result<Vec<f32>, GpuError> {
        let r = self
            .b
            .logits
            .read()
            .map_err(|e| GpuError::Validation(e.to_string()))?;
        Ok(r.to_vec())
    }

    /// One full decode step: stage → submit `graph` (expected to be
    /// `record(0..n_layers, true)`) → advance → logits.
    pub fn decode_step(&mut self, graph: &CommandGraph, token: u32) -> Result<Vec<f32>, GpuError> {
        self.stage_token(token)?;
        self.submit(graph)?;
        self.advance();
        self.read_logits()
    }
}

/// `WriteDescriptorSet::buffer`, kept generic (a `let` alias would
/// monomorphize to the first element type used).
fn buf(binding: u32, buffer: Subbuffer<impl ?Sized>) -> WriteDescriptorSet {
    WriteDescriptorSet::buffer(binding, buffer)
}

/// Record one rmsnorm dispatch: `rows` rows of `w.len()` elements.
fn rms(
    rec: &mut GraphRecorder<'_>,
    kernel: &Kernel,
    x: &Subbuffer<[u16]>,
    w: &Subbuffer<[f32]>,
    y: &Subbuffer<[u16]>,
    rows: usize,
) -> Result<(), GpuError> {
    rec.dispatch(
        kernel,
        vec![
            WriteDescriptorSet::buffer(0, x.clone()),
            WriteDescriptorSet::buffer(1, w.clone()),
            WriteDescriptorSet::buffer(2, y.clone()),
        ],
        None::<u32>,
        [rows as u32, 1, 1],
    )
    .map(|_| ())
}
