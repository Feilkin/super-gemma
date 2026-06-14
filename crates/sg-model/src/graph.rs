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

/// Layers per prefill SUBMISSION (one global layer each). A prefill chunk
/// is split into segment command buffers because amdgpu's job watchdog
/// (`lockup_timeout`, default 2000 ms on this kernel) kills any single
/// submission running longer — a whole 60-layer chunk crosses 2 s near
/// q0 ≈ 10K and was being ring-reset (diagnosed 2026-06-12; every "GPU
/// hang" that day was this cliff). Segments also keep the desktop
/// responsive. At 128K context the global-attention cost (~1 s/layer)
/// approaches the limit again — the flash-attention rewrite is the
/// durable fix; raising `amdgpu.lockup_timeout` on the box is the
/// operational backstop.
const PREFILL_SEGMENT_LAYERS: usize = 6;

const HIDDEN: usize = 5376;
const FFN: usize = 21504;

/// Which logits a prefill graph computes. (The all-positions perplexity
/// path is NOT a mode here: its LM-head work runs as separate batched
/// submissions — see `prefill_all_logits`.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum LogitsMode {
    /// None — interior chunks of a prompt.
    None,
    /// Final norm + LM head over the chunk's last real row (generation).
    Last,
}

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
    p: PrefillBufs,
    /// Prefill chunk capacity (multiple of the gemm M_BLOCK 64).
    max_chunk: usize,
    /// Capacity of the linear global-KV stores, in tokens.
    global_cap: usize,
    /// Recorded prefill SEGMENT graphs keyed by (padded_len, real_len,
    /// logits mode); segments are submitted in order (see
    /// [`PREFILL_SEGMENT_LAYERS`]).
    prefill_graphs: std::collections::HashMap<(usize, usize, LogitsMode), Vec<CommandGraph>>,
    /// `[max_chunk × vocab]` f32, allocated on first all-logits prefill
    /// (the perplexity path); ~270 MB at the default chunk size.
    logits_all: Option<Subbuffer<[f32]>>,
    /// Final-norm-over-all-rows graphs (keyed by m_pad) and LM-head row
    /// batches (keyed by row range), for the all-logits path. Batched into
    /// SEPARATE submissions: a single graph carrying the whole 60-layer
    /// chunk plus 256 LM-head gemvs (~1.4 s of saturated-bandwidth work)
    /// tripped the amdgpu watchdog's soft recovery — observed as a "context
    /// lost / guilty of hard recovery" device loss on the SECOND chunk and
    /// silently-cancelled waves on the first.
    norm_all_graphs: std::collections::HashMap<usize, CommandGraph>,
    logits_batch_graphs: std::collections::HashMap<(usize, usize), CommandGraph>,
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
    // Prefill (chunked, coopmat gemm path).
    gemm_q_sl: Kernel,
    gemm_q_gl: Kernel,
    gemm_kv_sl: Kernel,
    gemm_kv_gl: Kernel,
    gemm_o_sl: Kernel,
    gemm_o_gl: Kernel,
    gemm_up: Kernel,
    gemm_down: Kernel,
    // int8-MMQ FFN (feature int8-ffn): Q8 activation quant + Q4_0-reading int8
    // gemm at 2×2. Loaded unconditionally; the dispatch path is cfg-gated.
    quant_q8: Kernel,
    gemm_up_i8: Kernel,
    gemm_down_i8: Kernel,
    prefill_sl: Kernel,
    prefill_gl: Kernel,
    /// Sync shim: the gemm kernels read `x` only via coopmat loads, which
    /// vulkano's auto-sync cannot see — touch the buffer first so the
    /// producer's write→read barrier is emitted (touch.wgsl).
    touch: Kernel,
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

/// Prefill activation buffers, sized for `max_chunk` tokens. Separate from
/// the decode set: several kernels derive their bounds from `arrayLength`,
/// so decode on chunk-sized buffers would over-dispatch ~chunk×.
///
/// Smaller (padded-tail) chunks reuse these buffers' prefixes; dispatch
/// grids are recorded per chunk shape, and the `arrayLength`-bounded
/// elementwise kernels merely process the stale tail rows (garbage in,
/// garbage out — nothing reads them, and KV appends bind sliced sources).
struct PrefillBufs {
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
    o: Subbuffer<[u16]>,
    on: Subbuffer<[u16]>,
    x2: Subbuffer<[u16]>,
    fin: Subbuffer<[u16]>,
    g: Subbuffer<[u16]>,
    u: Subbuffer<[u16]>,
    gu: Subbuffer<[u16]>,
    f: Subbuffer<[u16]>,
    fn2: Subbuffer<[u16]>,
    /// int8-ffn activation-quant scratch: Q8_0 quants (u32-packed, the format
    /// `kv_quant_q8` writes and the int8 gemm reads as `array<i8>`) + f16 scales,
    /// for the FFN input (`fin`, HIDDEN) and the gate⊙up product (`gu`, FFN).
    fin_i8: Subbuffer<[u32]>,
    fin_scales: Subbuffer<[u16]>,
    gu_i8: Subbuffer<[u32]>,
    gu_scales: Subbuffer<[u16]>,
    /// Per-chunk rope tables: `[max_chunk × live_pairs × 2]` f32.
    cs_sl: Subbuffer<[f32]>,
    cs_gl: Subbuffer<[f32]>,
}

impl<'a> GpuModel<'a> {
    /// Upload weights and allocate all graph state. `global_cap` bounds the
    /// linear global-KV stores (tokens); tests keep it small, the engine
    /// will size it from the configured context limit. `max_chunk` is the
    /// prefill chunk capacity (rounded up to the gemm M_BLOCK 64; plan 03
    /// default 256).
    pub fn new(
        ctx: &'a GpuContext,
        gguf: &'a Gguf<'a>,
        global_cap: usize,
        max_chunk: usize,
    ) -> Result<Self, UploadError> {
        let max_chunk = max_chunk.next_multiple_of(64);
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

        let m = max_chunk;
        let p = PrefillBufs {
            x: f16buf(m * HIDDEN)?,
            xn: f16buf(m * HIDDEN)?,
            q_raw_sl: f16buf(m * q_dim_sl)?,
            q_sl: f16buf(m * q_dim_sl)?,
            q_raw_gl: f16buf(m * q_dim_gl)?,
            q_gl: f16buf(m * q_dim_gl)?,
            kp_sl: f16buf(m * kv_dim_sl)?,
            k_sl: f16buf(m * kv_dim_sl)?,
            vp_sl: f16buf(m * kv_dim_sl)?,
            v_sl: f16buf(m * kv_dim_sl)?,
            kp_gl: f16buf(m * kv_dim_gl)?,
            k_gl: f16buf(m * kv_dim_gl)?,
            v_gl: f16buf(m * kv_dim_gl)?,
            attn_sl: f16buf(m * q_dim_sl)?,
            attn_gl: f16buf(m * q_dim_gl)?,
            o: f16buf(m * HIDDEN)?,
            on: f16buf(m * HIDDEN)?,
            x2: f16buf(m * HIDDEN)?,
            fin: f16buf(m * HIDDEN)?,
            g: f16buf(m * FFN)?,
            u: f16buf(m * FFN)?,
            gu: f16buf(m * FFN)?,
            f: f16buf(m * HIDDEN)?,
            fn2: f16buf(m * HIDDEN)?,
            fin_i8: ctx.new_buffer::<u32>((m * HIDDEN / 4) as u64, usage)?,
            fin_scales: f16buf(m * HIDDEN / 32)?,
            gu_i8: ctx.new_buffer::<u32>((m * FFN / 4) as u64, usage)?,
            gu_scales: f16buf(m * FFN / 32)?,
            cs_sl: ctx.new_buffer::<f32>((m * desc.sliding.head_dim / 2 * 2) as u64, usage)?,
            cs_gl: ctx.new_buffer::<f32>((m * desc.global.head_dim / 8 * 2) as u64, usage)?,
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
            gemm_q_sl: load("gemm_q4_0_k5376_n8192")?,
            gemm_q_gl: load("gemm_q4_0_k5376_n16384")?,
            gemm_kv_sl: load("gemm_q4_0_k5376_n4096")?,
            gemm_kv_gl: load("gemm_q4_0_k5376_n2048")?,
            gemm_o_sl: load("gemm_q4_0_k8192_n5376")?,
            gemm_o_gl: load("gemm_q4_0_k16384_n5376")?,
            gemm_up: load("gemm_q4_0_k5376_n21504")?,
            gemm_down: load("gemm_q4_0_k21504_n5376")?,
            quant_q8: load("kv_quant_q8")?,
            gemm_up_i8: load("gemm_q4_0_i8_t22_k5376_n21504")?,
            gemm_down_i8: load("gemm_q4_0_i8_t22_k21504_n5376")?,
            prefill_sl: load("attn_prefill_sliding_ring")?,
            prefill_gl: load("attn_prefill_global")?,
            touch: load("touch")?,
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
            p,
            max_chunk,
            global_cap,
            prefill_graphs: std::collections::HashMap::new(),
            logits_all: None,
            norm_all_graphs: std::collections::HashMap::new(),
            logits_batch_graphs: std::collections::HashMap::new(),
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

    /// Record one prefill chunk as SEGMENT graphs of
    /// [`PREFILL_SEGMENT_LAYERS`] layers, submitted in order. `m_pad`
    /// padded rows (multiple of 64), `n_real` real tokens. Padding rows
    /// compute garbage nothing reads: the causal mask hides their keys
    /// from real queries, KV appends bind `n_real`-sliced sources, and
    /// logits read the last REAL row.
    fn record_prefill(
        &self,
        m_pad: usize,
        n_real: usize,
        logits: LogitsMode,
    ) -> Result<Vec<CommandGraph>, GpuError> {
        let n_layers = self.desc.n_layers;
        let mut segments = Vec::new();
        let mut start = 0;
        while start < n_layers {
            let end = (start + PREFILL_SEGMENT_LAYERS).min(n_layers);
            let last = end == n_layers;
            segments.push(self.ctx.record_graph(|rec| {
                for i in start..end {
                    self.record_prefill_layer(rec, i, m_pad, n_real)?;
                }
                match logits {
                    LogitsMode::Last if last => self.record_prefill_logits(rec, n_real),
                    _ => Ok(()),
                }
            })?);
            start = end;
        }
        Ok(segments)
    }

    /// Chunked prefill returning per-position logits, `[n × vocab]` f32 —
    /// the perplexity evaluation path (plan 06 rung 4 / the M4 gate).
    /// Slower than [`Self::prefill`] by ~5.5 ms LM-head cost per position;
    /// the head runs in batched submissions of [`LOGITS_BATCH`] rows to
    /// stay far from the GPU watchdog (see `norm_all_graphs`).
    pub fn prefill_all_logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>, GpuError> {
        /// ~175 ms of LM-head work per submission.
        const LOGITS_BATCH: usize = 32;
        if self.logits_all.is_none() {
            self.logits_all = Some(self.ctx.new_buffer::<f32>(
                (self.max_chunk * self.desc.vocab_size) as u64,
                BufferUsage::STORAGE_BUFFER,
            )?);
        }
        let vocab = self.desc.vocab_size;
        let mut out = Vec::with_capacity(tokens.len() * vocab);
        for chunk in tokens.chunks(self.max_chunk).collect::<Vec<_>>() {
            let n_real = chunk.len();
            let m_pad = n_real.next_multiple_of(64);

            // The 60 layers (same graph as an interior prompt chunk).
            let key = (m_pad, n_real, LogitsMode::None);
            if !self.prefill_graphs.contains_key(&key) {
                let g = self.record_prefill(m_pad, n_real, LogitsMode::None)?;
                self.prefill_graphs.insert(key, g);
            }
            self.stage_prefill_chunk(chunk)?;
            for segment in &self.prefill_graphs[&key] {
                self.ctx.submit_blocking(segment)?;
            }
            self.pos += n_real as u32;

            // Final norm over all rows, then the LM head in row batches.
            if !self.norm_all_graphs.contains_key(&m_pad) {
                let g = self.ctx.record_graph(|rec| {
                    rms(
                        rec,
                        &self.k.rms5376,
                        &self.p.x,
                        &self.weights.output_norm,
                        &self.p.xn,
                        m_pad,
                    )
                })?;
                self.norm_all_graphs.insert(m_pad, g);
            }
            self.ctx.submit_blocking(&self.norm_all_graphs[&m_pad])?;
            let mut r0 = 0usize;
            while r0 < n_real {
                let len = LOGITS_BATCH.min(n_real - r0);
                if !self.logits_batch_graphs.contains_key(&(r0, len)) {
                    let all = self.logits_all.as_ref().unwrap();
                    let g = self.ctx.record_graph(|rec| {
                        for r in r0 as u64..(r0 + len) as u64 {
                            rec.dispatch(
                                &self.k.logits,
                                vec![
                                    buf(0, self.weights.token_embd.clone()),
                                    buf(
                                        1,
                                        self.p
                                            .xn
                                            .clone()
                                            .slice(r * HIDDEN as u64..(r + 1) * HIDDEN as u64),
                                    ),
                                    buf(
                                        2,
                                        all.clone().slice(r * vocab as u64..(r + 1) * vocab as u64),
                                    ),
                                ],
                                None::<u32>,
                                [vocab as u32, 1, 1],
                            )?;
                        }
                        Ok(())
                    })?;
                    self.logits_batch_graphs.insert((r0, len), g);
                }
                self.ctx
                    .submit_blocking(&self.logits_batch_graphs[&(r0, len)])?;
                r0 += len;
            }

            let r = self
                .logits_all
                .as_ref()
                .unwrap()
                .read()
                .map_err(|e| GpuError::Validation(e.to_string()))?;
            out.extend_from_slice(&r[..n_real * vocab]);
        }
        Ok(out)
    }

    /// One full-size interior prefill chunk (no logits) as segment graphs,
    /// uninstrumented — the profiler wall-times the segment sequence.
    pub fn record_prefill_chunk_plain(&self) -> Result<Vec<CommandGraph>, GpuError> {
        self.record_prefill(self.max_chunk, self.max_chunk, LogitsMode::None)
    }

    /// Decode graph over `layers` only, with per-dispatch timestamps —
    /// kernel-level detail for a small representative range (e.g. one
    /// sliding + one global layer, extrapolated by layer counts). Keep the
    /// range SMALL: each timestamp drains the pipeline, which is harmless
    /// on the serial decode path but destroys prefill's cross-layer
    /// overlap — a fully-marked prefill chunk inflated past the 10 s
    /// amdgpu watchdog (measured 2026-06-12). Totals come from wall-timing
    /// the UNinstrumented graphs.
    pub fn record_layers_kernel_profiled(
        &self,
        layers: Range<usize>,
        timer: &sg_gpu::GpuTimer,
    ) -> Result<(CommandGraph, Vec<&'static str>), GpuError> {
        self.ctx.record_graph_profiled(timer, |rec| {
            for i in layers.clone() {
                self.record_layer(rec, i)?;
            }
            Ok(())
        })
    }

    /// Prefill analog of [`Self::record_layers_kernel_profiled`]; the
    /// same small-range caveat applies with more force (see there).
    pub fn record_prefill_kernel_profiled(
        &self,
        layers: Range<usize>,
        timer: &sg_gpu::GpuTimer,
    ) -> Result<(CommandGraph, Vec<&'static str>), GpuError> {
        let m = self.max_chunk;
        self.ctx.record_graph_profiled(timer, |rec| {
            for i in layers.clone() {
                self.record_prefill_layer(rec, i, m, m)?;
            }
            Ok(())
        })
    }

    /// Per-layer-range prefill graph for the parity harness (mirrors
    /// [`Self::record`] on the decode side): submit one layer at a time
    /// after [`Self::stage_prefill_chunk`] and inspect
    /// [`Self::read_prefill_hidden`] between submits.
    pub fn record_prefill_layers(
        &self,
        layers: Range<usize>,
        m_pad: usize,
        n_real: usize,
    ) -> Result<CommandGraph, GpuError> {
        self.ctx.record_graph(|rec| {
            for i in layers.clone() {
                self.record_prefill_layer(rec, i, m_pad, n_real)?;
            }
            Ok(())
        })
    }

    /// Read a prefill activation buffer by oracle tap name, as f32 — the
    /// parity harness's window into the per-stage intermediates. Names
    /// match [`crate::CpuModel::forward`]'s tap points.
    pub fn read_prefill_tap(&self, name: &str, sliding: bool) -> Result<Vec<f32>, GpuError> {
        let p = &self.p;
        let b: &Subbuffer<[u16]> = match (name, sliding) {
            ("attn_norm", _) => &p.xn,
            ("q_rope", true) => &p.q_sl,
            ("q_rope", false) => &p.q_gl,
            ("k_rope", true) => &p.k_sl,
            ("k_rope", false) => &p.k_gl,
            ("v_norm", true) => &p.v_sl,
            ("v_norm", false) => &p.v_gl,
            ("attn", true) => &p.attn_sl,
            ("attn", false) => &p.attn_gl,
            ("attn_out", _) => &p.o,
            ("post_attn_norm", _) => &p.on,
            ("h_attn", _) => &p.x2,
            ("ffn_norm", _) => &p.fin,
            ("ffn_out", _) => &p.f,
            ("post_ffn_norm", _) => &p.fn2,
            ("layer_out", _) => &p.x,
            _ => return Err(GpuError::Validation(format!("unknown tap `{name}`"))),
        };
        let r = b.read().map_err(|e| GpuError::Validation(e.to_string()))?;
        Ok(r.iter().map(|&v| f16::from_bits(v).to_f32()).collect())
    }

    /// The prefill residual stream, first `n` tokens, as f32.
    pub fn read_prefill_hidden(&self, n: usize) -> Result<Vec<f32>, GpuError> {
        let r = self
            .p
            .x
            .read()
            .map_err(|e| GpuError::Validation(e.to_string()))?;
        Ok(r[..n * HIDDEN]
            .iter()
            .map(|&b| f16::from_bits(b).to_f32())
            .collect())
    }

    fn record_prefill_layer(
        &self,
        rec: &mut GraphRecorder<'_>,
        i: usize,
        m_pad: usize,
        n_real: usize,
    ) -> Result<(), GpuError> {
        let lw = &self.weights.layers[i];
        let p = &self.p;
        let kv = &self.kv[i];
        let sliding = lw.kind == LayerKind::Sliding;
        let (hd, n_kv) = match lw.kind {
            LayerKind::Sliding => (self.desc.sliding.head_dim, self.desc.sliding.n_kv_heads),
            LayerKind::Global => (self.desc.global.head_dim, self.desc.global.n_kv_heads),
        };
        let q_dim = 32 * hd;
        let kv_dim = n_kv * hd;
        let mg = (m_pad / 64) as u32; // gemm M-block count
        let (q_raw, q, kp, k, v) = if sliding {
            (&p.q_raw_sl, &p.q_sl, &p.kp_sl, &p.k_sl, &p.v_sl)
        } else {
            (&p.q_raw_gl, &p.q_gl, &p.kp_gl, &p.k_gl, &p.v_gl)
        };
        let (rms_hd, rope_q, rope_k, cs) = if sliding {
            (
                &self.k.rms256,
                &self.k.rope_sl_q,
                &self.k.rope_sl_k,
                &p.cs_sl,
            )
        } else {
            (
                &self.k.rms512,
                &self.k.rope_gl_q,
                &self.k.rope_gl_k,
                &p.cs_gl,
            )
        };
        let (gemm_q, gemm_kv, gemm_o) = if sliding {
            (&self.k.gemm_q_sl, &self.k.gemm_kv_sl, &self.k.gemm_o_sl)
        } else {
            (&self.k.gemm_q_gl, &self.k.gemm_kv_gl, &self.k.gemm_o_gl)
        };
        let no_push = None::<u32>;

        // The gemms read their activation input only via coopmat loads,
        // invisible to vulkano's auto-sync: a `touch` of the buffer makes
        // the producer's write→read barrier materialize (touch.wgsl).
        let touch = |rec: &mut GraphRecorder<'_>, b: &Subbuffer<[u16]>| -> Result<(), GpuError> {
            rec.dispatch(
                &self.k.touch,
                vec![buf(0, b.clone())],
                None::<u32>,
                [1, 1, 1],
            )
            .map(|_| ())
        };

        // ── Attention block ──────────────────────────────────────────────
        rms(rec, &self.k.rms5376, &p.x, &lw.attn_norm, &p.xn, m_pad)?;
        touch(rec, &p.xn)?;
        rec.dispatch(
            gemm_q,
            vec![
                buf(0, lw.attn_q.clone()),
                buf(1, p.xn.clone()),
                buf(2, q_raw.clone()),
            ],
            no_push,
            [(q_dim / 64) as u32, mg, 1],
        )?;
        rec.dispatch(
            gemm_kv,
            vec![
                buf(0, lw.attn_k.clone()),
                buf(1, p.xn.clone()),
                buf(2, kp.clone()),
            ],
            no_push,
            [(kv_dim / 64) as u32, mg, 1],
        )?;
        let vp = match &lw.attn_v {
            Some(wv) => {
                rec.dispatch(
                    gemm_kv,
                    vec![
                        buf(0, wv.clone()),
                        buf(1, p.xn.clone()),
                        buf(2, p.vp_sl.clone()),
                    ],
                    no_push,
                    [(kv_dim / 64) as u32, mg, 1],
                )?;
                &p.vp_sl
            }
            None => kp,
        };
        rms(rec, rms_hd, q_raw, &lw.attn_q_norm, q, m_pad * 32)?;
        rms(rec, rms_hd, kp, &lw.attn_k_norm, k, m_pad * n_kv)?;
        rms(rec, rms_hd, vp, &self.weights.norm_ones, v, m_pad * n_kv)?;
        let live_pairs = if sliding { hd / 2 } else { hd / 8 };
        rec.dispatch(
            rope_q,
            vec![buf(0, q.clone()), buf(1, cs.clone())],
            no_push,
            rope_q.groups_for((m_pad * 32 * live_pairs) as u64),
        )?;
        rec.dispatch(
            rope_k,
            vec![buf(0, k.clone()), buf(1, cs.clone())],
            no_push,
            rope_k.groups_for((m_pad * n_kv * live_pairs) as u64),
        )?;

        // KV-append sources sliced to the REAL rows (padding must not land
        // in the stores). Global appends BEFORE attending (linear store,
        // causal mask makes the chunk's own keys visible at the right
        // queries); sliding attends from ring + chunk FIRST, then appends
        // (plan 03 §prefill — appending first would overwrite early
        // queries' windows).
        let append = if sliding {
            &self.k.append_sl
        } else {
            &self.k.append_gl
        };
        let real = (n_real * kv_dim) as u64;
        let do_appends = |rec: &mut GraphRecorder<'_>| -> Result<(), GpuError> {
            for (src, dst) in [(k, &kv.k), (v, &kv.v)] {
                rec.dispatch(
                    append,
                    vec![
                        buf(0, src.clone().slice(0..real)),
                        buf(1, dst.clone()),
                        buf(2, self.step.clone()),
                    ],
                    None::<u32>,
                    append.groups_for(real),
                )?;
            }
            Ok(())
        };

        let attn_out = if sliding { &p.attn_sl } else { &p.attn_gl };
        if sliding {
            rec.dispatch(
                &self.k.prefill_sl,
                vec![
                    buf(0, q.clone()),
                    buf(1, kv.k.clone()),
                    buf(2, kv.v.clone()),
                    buf(3, k.clone()),
                    buf(4, v.clone()),
                    buf(5, attn_out.clone()),
                    buf(6, self.step.clone()),
                ],
                Some(1.0f32),
                [n_kv as u32, m_pad as u32, 1],
            )?;
            do_appends(rec)?;
        } else {
            do_appends(rec)?;
            rec.dispatch(
                &self.k.prefill_gl,
                vec![
                    buf(0, q.clone()),
                    buf(1, kv.k.clone()),
                    buf(2, kv.v.clone()),
                    buf(3, attn_out.clone()),
                    buf(4, self.step.clone()),
                ],
                Some(1.0f32),
                [n_kv as u32, m_pad as u32, 1],
            )?;
        }

        touch(rec, attn_out)?;
        rec.dispatch(
            gemm_o,
            vec![
                buf(0, lw.attn_output.clone()),
                buf(1, attn_out.clone()),
                buf(2, p.o.clone()),
            ],
            no_push,
            [(HIDDEN / 64) as u32, mg, 1],
        )?;
        rms(
            rec,
            &self.k.rms5376,
            &p.o,
            &lw.post_attention_norm,
            &p.on,
            m_pad,
        )?;
        rec.dispatch(
            &self.k.add,
            vec![
                buf(0, p.x.clone()),
                buf(1, p.on.clone()),
                buf(2, p.x2.clone()),
            ],
            Some(1.0f32),
            self.k.add.groups_for(p.x2.len()),
        )?;

        // ── FFN block ────────────────────────────────────────────────────
        rms(rec, &self.k.rms5376, &p.x2, &lw.ffn_norm, &p.fin, m_pad)?;
        touch(rec, &p.fin)?;
        // FFN gate+up: f16 gemm, or (feature int8-ffn) Q8-quantize `fin` once
        // (shared by gate and up) and run the Q4_0-reading int8 gemm at 2×2.
        if cfg!(feature = "int8-ffn") {
            rec.dispatch(
                &self.k.quant_q8,
                vec![
                    buf(0, p.fin.clone()),
                    buf(1, p.fin_scales.clone()),
                    buf(2, p.fin_i8.clone()),
                ],
                no_push,
                self.k.quant_q8.groups_for((m_pad * HIDDEN / 32) as u64),
            )?;
            // The int8 gemm coopLoads its quants → invisible to auto-sync.
            rec.dispatch(&self.k.touch, vec![buf(0, p.fin_i8.clone())], no_push, [1, 1, 1])?;
            let mg2 = (m_pad / 32) as u32; // int8 2×2 M-block count
            for (w, dst) in [(&lw.ffn_gate, &p.g), (&lw.ffn_up, &p.u)] {
                rec.dispatch(
                    &self.k.gemm_up_i8,
                    vec![
                        buf(0, w.clone()),
                        buf(1, p.fin_i8.clone()),
                        buf(2, p.fin_scales.clone()),
                        buf(3, (*dst).clone()),
                    ],
                    no_push,
                    [(FFN / 32) as u32, mg2, 1],
                )?;
            }
        } else {
            for (w, dst) in [(&lw.ffn_gate, &p.g), (&lw.ffn_up, &p.u)] {
                rec.dispatch(
                    &self.k.gemm_up,
                    vec![
                        buf(0, w.clone()),
                        buf(1, p.fin.clone()),
                        buf(2, (*dst).clone()),
                    ],
                    no_push,
                    [(FFN / 64) as u32, mg, 1],
                )?;
            }
        }
        rec.dispatch(
            &self.k.geglu,
            vec![
                buf(0, p.g.clone()),
                buf(1, p.u.clone()),
                buf(2, p.gu.clone()),
            ],
            no_push,
            self.k.geglu.groups_for(p.gu.len()),
        )?;
        // FFN down: f16 gemm, or (feature int8-ffn) Q8-quantize `gu` and int8 gemm.
        if cfg!(feature = "int8-ffn") {
            rec.dispatch(
                &self.k.quant_q8,
                vec![
                    buf(0, p.gu.clone()),
                    buf(1, p.gu_scales.clone()),
                    buf(2, p.gu_i8.clone()),
                ],
                no_push,
                self.k.quant_q8.groups_for((m_pad * FFN / 32) as u64),
            )?;
            rec.dispatch(&self.k.touch, vec![buf(0, p.gu_i8.clone())], no_push, [1, 1, 1])?;
            let mg2 = (m_pad / 32) as u32;
            rec.dispatch(
                &self.k.gemm_down_i8,
                vec![
                    buf(0, lw.ffn_down.clone()),
                    buf(1, p.gu_i8.clone()),
                    buf(2, p.gu_scales.clone()),
                    buf(3, p.f.clone()),
                ],
                no_push,
                [(HIDDEN / 32) as u32, mg2, 1],
            )?;
        } else {
            touch(rec, &p.gu)?;
            rec.dispatch(
                &self.k.gemm_down,
                vec![
                    buf(0, lw.ffn_down.clone()),
                    buf(1, p.gu.clone()),
                    buf(2, p.f.clone()),
                ],
                no_push,
                [(HIDDEN / 64) as u32, mg, 1],
            )?;
        }
        rms(rec, &self.k.rms5376, &p.f, &lw.post_ffw_norm, &p.fn2, m_pad)?;
        rec.dispatch(
            &self.k.add,
            vec![
                buf(0, p.x2.clone()),
                buf(1, p.fn2.clone()),
                buf(2, p.x.clone()),
            ],
            Some(lw.layer_output_scale),
            self.k.add.groups_for(p.x.len()),
        )?;
        Ok(())
    }

    /// Final norm + LM head over the chunk's LAST REAL row only (plan 03:
    /// no logits GEMM over prefill positions). Reuses the decode-side
    /// `xn`/`logits` buffers.
    fn record_prefill_logits(
        &self,
        rec: &mut GraphRecorder<'_>,
        n_real: usize,
    ) -> Result<(), GpuError> {
        let last = self
            .p
            .x
            .clone()
            .slice(((n_real - 1) * HIDDEN) as u64..(n_real * HIDDEN) as u64);
        rec.dispatch(
            &self.k.rms5376,
            vec![
                buf(0, last),
                buf(1, self.weights.output_norm.clone()),
                buf(2, self.b.xn.clone()),
            ],
            None::<u32>,
            [1, 1, 1],
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

    /// CPU-side per-chunk staging: embedding rows, step state, rope tables
    /// for positions `pos .. pos + tokens.len()`.
    pub fn stage_prefill_chunk(&self, tokens: &[u32]) -> Result<(), GpuError> {
        {
            let mut w = self
                .p
                .x
                .write()
                .map_err(|e| GpuError::Validation(e.to_string()))?;
            for (t, &tok) in tokens.iter().enumerate() {
                let row = self.embed_row(tok);
                for (dst, &val) in w[t * HIDDEN..][..HIDDEN].iter_mut().zip(&row) {
                    *dst = f16::from_f32(val).to_bits();
                }
            }
        }
        StepState {
            pos: self.pos,
            kv_len_sliding: 0, // unused by the prefill kernels
            kv_len_global: 0,  // unused by the prefill kernels
            q0: self.pos,
        }
        .write_to(&self.step)?;

        let positions: Vec<u32> = (0..tokens.len() as u32).map(|i| self.pos + i).collect();
        let write_cs = |buf: &Subbuffer<[f32]>, table: &[f32]| -> Result<(), GpuError> {
            let mut w = buf
                .write()
                .map_err(|e| GpuError::Validation(e.to_string()))?;
            w[..table.len()].copy_from_slice(table);
            Ok(())
        };
        let sl = cos_sin_table(
            &positions,
            self.desc.sliding.head_dim,
            self.desc.sliding.rope_theta,
            None,
        );
        write_cs(&self.p.cs_sl, &sl)?;
        // Global: keep only the live pairs of each position's table entry.
        let gl_full = cos_sin_table(
            &positions,
            self.desc.global.head_dim,
            self.desc.global.rope_theta,
            Some(&self.weights.rope_factors),
        );
        let (full, live) = (
            self.desc.global.head_dim / 2 * 2,
            self.desc.global.head_dim / 8 * 2,
        );
        let gl: Vec<f32> = gl_full
            .chunks_exact(full)
            .flat_map(|t| t[..live].iter().copied())
            .collect();
        write_cs(&self.p.cs_gl, &gl)?;
        Ok(())
    }

    /// Chunked prefill of `tokens` starting at the current `pos` (plan 03
    /// §prefill). Returns the last token's logits. Graphs are recorded per
    /// chunk shape and cached.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, GpuError> {
        assert!(!tokens.is_empty());
        assert!(
            self.pos as usize + tokens.len() <= self.global_cap,
            "prefill past the global-KV capacity ({} + {} > {})",
            self.pos,
            tokens.len(),
            self.global_cap
        );
        let n_chunks = tokens.len().div_ceil(self.max_chunk);
        for (ci, chunk) in tokens.chunks(self.max_chunk).enumerate() {
            let n_real = chunk.len();
            let m_pad = n_real.next_multiple_of(64);
            let mode = if ci == n_chunks - 1 {
                LogitsMode::Last
            } else {
                LogitsMode::None
            };
            let key = (m_pad, n_real, mode);
            if !self.prefill_graphs.contains_key(&key) {
                let g = self.record_prefill(m_pad, n_real, mode)?;
                self.prefill_graphs.insert(key, g);
            }
            self.stage_prefill_chunk(chunk)?;
            for segment in &self.prefill_graphs[&key] {
                self.ctx.submit_blocking(segment)?;
            }
            self.pos += n_real as u32;
        }
        self.read_logits()
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
