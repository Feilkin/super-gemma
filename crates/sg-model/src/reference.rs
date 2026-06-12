//! CPU reference model: the M3 parity oracle (plan 03 step 1).
//!
//! A slow, obvious implementation of the pinned Gemma 4 forward graph
//! (`docs/reference/gemma4-forward-graph.md`) straight off the mmap'd GGUF:
//! weights stay quantized and are dequantized on the fly inside the matmuls
//! (an f32 copy of 31B weights would not fit in RAM), values are f32 with
//! f64 accumulation in every reduction (dots, norms, softmax). Bit-exact
//! reruns; no GPU, no vulkano.
//!
//! Verify-item semantics sit behind [`Conventions`] so the parity harness
//! can flip a single knob if the pinned reading of the reference ever
//! disagrees with llama.cpp logits (plan 03 "swappable functions").

use rayon::prelude::*;
use sg_gguf::{GgmlType, Gguf, LayerKind, ModelDesc, q4_0, q6_k};

use crate::rope::{apply_rope, cos_sin_table};

/// Verify-item knobs (plan 00 §verify-against-reference), defaults = the
/// pinned reading. Flipping any of these is a parity-harness experiment,
/// not a configuration surface.
#[derive(Debug, Clone)]
pub struct Conventions {
    /// RMSNorm `x̂·(1+w)` instead of the pinned `x̂·w`.
    pub norm_one_plus_w: bool,
    /// Embedding rows are scaled by this; pinned: `√hidden_size` in f32
    /// (llama.cpp). HF-bf16 would round it to 73.5.
    pub embed_scale: f32,
    /// Softmax logit scale; pinned: 1.0 (QK-norm replaces `1/√d`).
    pub attn_scale: f32,
    /// Weightless RMS-norm on V after projection; pinned: on.
    pub v_norm: bool,
    /// Multiply the hidden state by `layer_output_scale` at layer end;
    /// pinned: on.
    pub layer_output_scale: bool,
}

impl Conventions {
    fn pinned(hidden_size: usize) -> Self {
        Self {
            norm_one_plus_w: false,
            embed_scale: (hidden_size as f32).sqrt(),
            attn_scale: 1.0,
            v_norm: true,
            layer_output_scale: true,
        }
    }
}

/// Reference-side KV cache: full linear history per layer, even for sliding
/// layers (the window is applied as a mask — clarity over memory; tests run
/// short contexts).
pub struct CpuKvCache {
    /// Per layer: `[pos × n_kv_heads × head_dim]` f32, K and V.
    layers: Vec<(Vec<f32>, Vec<f32>)>,
    /// Tokens already cached (= absolute position of the next token).
    len: usize,
}

impl CpuKvCache {
    pub fn new(desc: &ModelDesc) -> Self {
        Self {
            layers: vec![(Vec::new(), Vec::new()); desc.n_layers],
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Errors constructing the reference model. Forward passes don't error:
/// every shape is validated here or by `ModelDesc`.
#[derive(Debug, thiserror::Error)]
pub enum RefError {
    #[error("tensor `{0}` missing from GGUF")]
    MissingTensor(String),
    #[error("tensor `{name}`: {what}")]
    BadTensor { name: String, what: String },
}

/// Activation tap: `(name, layer, data)` called at every named point of the
/// forward pass (layer = `usize::MAX` for pre/post-stack points). Names are
/// the parity-harness contract; see `forward` for the call sites.
pub type Tap<'t> = &'t mut dyn FnMut(&str, usize, &[f32]);

/// The CPU reference model. Borrows the parsed GGUF (and through it the
/// mmap) — keep the `GgufFile` alive.
pub struct CpuModel<'a> {
    gguf: &'a Gguf<'a>,
    pub desc: ModelDesc,
    pub conv: Conventions,
    /// `rope_freqs.weight`, validated at load: `[1.0 ×64, ≥1e20 ×192]`.
    rope_factors: Vec<f32>,
}

/// A named quantized matrix: rows of `n_in` elements, `n_out` rows.
struct MatWeight<'a> {
    bytes: &'a [u8],
    dtype: GgmlType,
    n_in: usize,
    n_out: usize,
}

impl<'a> CpuModel<'a> {
    pub fn new(gguf: &'a Gguf<'a>) -> Result<Self, RefError> {
        let desc = ModelDesc::from_gguf(gguf).map_err(|e| RefError::BadTensor {
            name: "<model>".into(),
            what: e.to_string(),
        })?;
        let conv = Conventions::pinned(desc.hidden_size);

        let rope_factors = f32_tensor(gguf, "rope_freqs.weight")?;
        let half = desc.global.head_dim / 2;
        if rope_factors.len() != half {
            return Err(RefError::BadTensor {
                name: "rope_freqs.weight".into(),
                what: format!("{} entries, expected {half}", rope_factors.len()),
            });
        }
        // Pinned table: partial_rotary_factor 0.25 → first quarter of the
        // pairs live (factor 1.0), the rest frozen by a huge divisor.
        let live = half / 4;
        for (i, &f) in rope_factors.iter().enumerate() {
            let ok = if i < live { f == 1.0 } else { f >= 1e20 };
            if !ok {
                return Err(RefError::BadTensor {
                    name: "rope_freqs.weight".into(),
                    what: format!("factor[{i}] = {f}, outside the pinned shape"),
                });
            }
        }

        Ok(Self {
            gguf,
            desc,
            conv,
            rope_factors,
        })
    }

    /// Run `tokens` (absolute positions `cache.len()..`) through the full
    /// stack, appending to `cache`. Returns logits for every input token,
    /// `[n_tokens × vocab]`, softcap applied.
    ///
    /// Tap points per layer: `attn_norm`, `q_rope`, `k_rope`, `v_norm`,
    /// `attn_out`, `post_attn_norm`, `h_attn` (post-residual), `ffn_norm`,
    /// `ffn_out`, `post_ffn_norm`, `layer_out`; plus `embed`, `final_norm`,
    /// `logits` with layer = `usize::MAX`.
    pub fn forward(&self, tokens: &[u32], cache: &mut CpuKvCache, tap: Tap<'_>) -> Vec<f32> {
        let d = &self.desc;
        let n = tokens.len();
        let hidden = d.hidden_size;
        const NO_LAYER: usize = usize::MAX;

        // Embedding lookup × embed_scale.
        let embd = self.mat("token_embd.weight", hidden, d.vocab_size);
        let mut x = vec![0.0f32; n * hidden];
        for (t, &tok) in tokens.iter().enumerate() {
            let row = &mut x[t * hidden..][..hidden];
            dequant_row(&embd, tok as usize, row);
            for v in row.iter_mut() {
                *v *= self.conv.embed_scale;
            }
        }
        tap("embed", NO_LAYER, &x);

        let positions: Vec<u32> = (0..n).map(|i| (cache.len + i) as u32).collect();
        // Per-kind rope tables for this chunk's positions.
        let cs_sliding = cos_sin_table(&positions, d.sliding.head_dim, d.sliding.rope_theta, None);
        let cs_global = cos_sin_table(
            &positions,
            d.global.head_dim,
            d.global.rope_theta,
            Some(&self.rope_factors),
        );

        for layer in 0..d.n_layers {
            let kind = d.layer_kinds[layer];
            let geo = match kind {
                LayerKind::Sliding => &d.sliding,
                LayerKind::Global => &d.global,
            };
            let hd = geo.head_dim;
            let n_kv = geo.n_kv_heads;
            let nq = d.n_q_heads;
            let q_dim = nq * hd;
            let kv_dim = n_kv * hd;
            let (cs, cs_half) = match kind {
                LayerKind::Sliding => (&cs_sliding, d.sliding.head_dim / 2),
                LayerKind::Global => (&cs_global, d.global.head_dim / 2),
            };
            let t_name = |suffix: &str| format!("blk.{layer}.{suffix}.weight");

            // ── Attention ────────────────────────────────────────────────
            let mut xn = x.clone();
            self.rmsnorm_rows(&mut xn, hidden, &t_name("attn_norm"));
            tap("attn_norm", layer, &xn);

            let wq = self.mat(&t_name("attn_q"), hidden, q_dim);
            let mut q = matmul(&wq, &xn, n);
            let q_norm_w = self.norm_weight(&t_name("attn_q_norm"));
            for (t, head) in head_rows(&mut q, nq, hd) {
                self.rmsnorm_one(head, Some(&q_norm_w));
                apply_rope(head, &cs[t * cs_half * 2..][..cs_half * 2]);
            }
            tap("q_rope", layer, &q);

            let wk = self.mat(&t_name("attn_k"), hidden, kv_dim);
            let k_proj = matmul(&wk, &xn, n);
            // Global layers share the K projection for V (attention_k_eq_v);
            // K and V still diverge through their norms + rope.
            let mut v = match kind {
                LayerKind::Sliding => {
                    let wv = self.mat(&t_name("attn_v"), hidden, kv_dim);
                    matmul(&wv, &xn, n)
                }
                LayerKind::Global => k_proj.clone(),
            };
            let mut k = k_proj;
            let k_norm_w = self.norm_weight(&t_name("attn_k_norm"));
            for (t, head) in head_rows(&mut k, n_kv, hd) {
                self.rmsnorm_one(head, Some(&k_norm_w));
                apply_rope(head, &cs[t * cs_half * 2..][..cs_half * 2]);
            }
            tap("k_rope", layer, &k);
            if self.conv.v_norm {
                for (_, head) in head_rows(&mut v, n_kv, hd) {
                    self.rmsnorm_one(head, None);
                }
            }
            tap("v_norm", layer, &v);

            let (ck, cv) = &mut cache.layers[layer];
            ck.extend_from_slice(&k);
            cv.extend_from_slice(&v);

            let window = match kind {
                LayerKind::Sliding => Some(d.sliding_window),
                LayerKind::Global => None,
            };
            let attn = attention(
                &q,
                ck,
                cv,
                n,
                cache.len,
                nq,
                n_kv,
                hd,
                window,
                self.conv.attn_scale,
            );

            let wo = self.mat(&t_name("attn_output"), q_dim, hidden);
            let mut o = matmul(&wo, &attn, n);
            tap("attn_out", layer, &o);
            self.rmsnorm_rows(&mut o, hidden, &t_name("post_attention_norm"));
            tap("post_attn_norm", layer, &o);
            for (h, r) in x.iter_mut().zip(&o) {
                *h += r;
            }
            tap("h_attn", layer, &x);

            // ── FFN ──────────────────────────────────────────────────────
            let mut fn_in = x.clone();
            self.rmsnorm_rows(&mut fn_in, hidden, &t_name("ffn_norm"));
            tap("ffn_norm", layer, &fn_in);
            let wg = self.mat(&t_name("ffn_gate"), hidden, d.ffn_size);
            let wu = self.mat(&t_name("ffn_up"), hidden, d.ffn_size);
            let mut gate = matmul(&wg, &fn_in, n);
            let up = matmul(&wu, &fn_in, n);
            for (g, u) in gate.iter_mut().zip(&up) {
                *g = gelu_tanh(*g) * u;
            }
            let wd = self.mat(&t_name("ffn_down"), d.ffn_size, hidden);
            let mut f = matmul(&wd, &gate, n);
            tap("ffn_out", layer, &f);
            self.rmsnorm_rows(&mut f, hidden, &t_name("post_ffw_norm"));
            tap("post_ffn_norm", layer, &f);
            for (h, r) in x.iter_mut().zip(&f) {
                *h += r;
            }

            if self.conv.layer_output_scale {
                let s = f32_tensor(self.gguf, &t_name("layer_output_scale"))
                    .expect("validated by ModelDesc")[0];
                for v in x.iter_mut() {
                    *v *= s;
                }
            }
            tap("layer_out", layer, &x);
        }

        cache.len += n;

        self.rmsnorm_rows(&mut x, hidden, "output_norm.weight");
        tap("final_norm", NO_LAYER, &x);

        // Tied Q6_K LM head + tanh-30 softcap.
        let mut logits = matmul(&embd, &x, n);
        let cap = d.final_logit_softcap;
        for v in logits.iter_mut() {
            *v = cap * (*v / cap).tanh();
        }
        tap("logits", NO_LAYER, &logits);
        logits
    }

    /// Look up a matrix tensor; panics on shape mismatch (everything here
    /// was validated by `ModelDesc` at construction).
    fn mat(&self, name: &str, n_in: usize, n_out: usize) -> MatWeight<'a> {
        let info = self.gguf.tensor(name).expect("validated by ModelDesc");
        assert_eq!(
            info.dims,
            vec![n_in as u64, n_out as u64],
            "{name}: unexpected shape"
        );
        MatWeight {
            bytes: self.gguf.data_of(info),
            dtype: info.dtype,
            n_in,
            n_out,
        }
    }

    fn norm_weight(&self, name: &str) -> Vec<f32> {
        f32_tensor(self.gguf, name).expect("validated by ModelDesc")
    }

    /// RMS-normalize each `row_len` row of `x` with the named F32 weight.
    fn rmsnorm_rows(&self, x: &mut [f32], row_len: usize, weight: &str) {
        let w = self.norm_weight(weight);
        assert_eq!(w.len(), row_len, "{weight}: length");
        for row in x.chunks_mut(row_len) {
            self.rmsnorm_one(row, Some(&w));
        }
    }

    /// RMS-normalize one row: `x̂ = x / sqrt(mean(x²) + eps)`, then `x̂·w`
    /// (or `x̂·(1+w)` under the flipped convention); `None` = weightless
    /// (the V-norm). f64 accumulation, f32 application.
    fn rmsnorm_one(&self, row: &mut [f32], w: Option<&[f32]>) {
        let ms: f64 = row.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / row.len() as f64;
        let inv = 1.0 / (ms + self.desc.rms_norm_eps as f64).sqrt();
        match w {
            Some(w) => {
                let add = if self.conv.norm_one_plus_w { 1.0 } else { 0.0 };
                for (v, &wi) in row.iter_mut().zip(w) {
                    *v = (*v as f64 * inv) as f32 * (wi + add);
                }
            }
            None => {
                for v in row.iter_mut() {
                    *v = (*v as f64 * inv) as f32;
                }
            }
        }
    }
}

/// Iterate `(token_index, head_row)` over `[token × n_heads × head_dim]`.
fn head_rows(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
) -> impl Iterator<Item = (usize, &mut [f32])> {
    x.chunks_mut(head_dim)
        .enumerate()
        .map(move |(i, c)| (i / n_heads, c))
}

/// `y = W·x` for all tokens: `x` is `[n_tokens × n_in]` token-major,
/// result `[n_tokens × n_out]` token-major. Rows are dequantized on the
/// fly; dots accumulate in f64. Deterministic (no reduction reordering).
fn matmul(w: &MatWeight<'_>, x: &[f32], n_tokens: usize) -> Vec<f32> {
    assert_eq!(x.len(), n_tokens * w.n_in);
    // Row-major intermediate so rayon can hand each weight row a disjoint
    // output slice; transposed to token-major at the end.
    let mut rt = vec![0.0f32; w.n_out * n_tokens];
    rt.par_chunks_mut(n_tokens).enumerate().for_each_init(
        Vec::new,
        |row_buf: &mut Vec<f32>, (r, dst)| {
            row_buf.resize(w.n_in, 0.0);
            dequant_row(w, r, row_buf);
            for (t, d) in dst.iter_mut().enumerate() {
                let xs = &x[t * w.n_in..][..w.n_in];
                let acc: f64 = row_buf
                    .iter()
                    .zip(xs)
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum();
                *d = acc as f32;
            }
        },
    );
    let mut y = vec![0.0f32; n_tokens * w.n_out];
    for r in 0..w.n_out {
        for t in 0..n_tokens {
            y[t * w.n_out + r] = rt[r * n_tokens + t];
        }
    }
    y
}

/// Dequantize row `r` of a quantized matrix into `out` (`n_in` long).
fn dequant_row(w: &MatWeight<'_>, r: usize, out: &mut [f32]) {
    assert!(r < w.n_out, "row {r} out of range ({} rows)", w.n_out);
    assert_eq!(out.len(), w.n_in);
    match w.dtype {
        GgmlType::Q4_0 => {
            let bpr = w.n_in / q4_0::QK4_0;
            let row_bytes =
                &w.bytes[r * bpr * q4_0::BLOCK_Q4_0_SIZE..][..bpr * q4_0::BLOCK_Q4_0_SIZE];
            let blocks = q4_0::blocks_from_bytes(row_bytes).expect("validated tensor size");
            let mut buf = [0.0f32; q4_0::QK4_0];
            for (b, block) in blocks.iter().enumerate() {
                block.dequantize(&mut buf);
                out[b * q4_0::QK4_0..][..q4_0::QK4_0].copy_from_slice(&buf);
            }
        }
        GgmlType::Q6_K => {
            let bpr = w.n_in / q6_k::QK6_K;
            let row_bytes =
                &w.bytes[r * bpr * q6_k::BLOCK_Q6_K_SIZE..][..bpr * q6_k::BLOCK_Q6_K_SIZE];
            let blocks = q6_k::blocks_from_bytes(row_bytes).expect("validated tensor size");
            let mut buf = [0.0f32; q6_k::QK6_K];
            for (b, block) in blocks.iter().enumerate() {
                block.dequantize(&mut buf);
                out[b * q6_k::QK6_K..][..q6_k::QK6_K].copy_from_slice(&buf);
            }
        }
        other => panic!("matmul on unsupported dtype {other}"),
    }
}

/// Causal (optionally windowed) GQA attention over the linear cache.
/// `q`: `[n × n_q_heads × hd]`, cache `[len+n × n_kv × hd]`; `q0` is the
/// absolute position of query token 0. Softmax in f64, scale applied to
/// the logits. Returns `[n × n_q_heads × hd]`.
#[allow(clippy::too_many_arguments)]
fn attention(
    q: &[f32],
    ck: &[f32],
    cv: &[f32],
    n: usize,
    q0: usize,
    n_q_heads: usize,
    n_kv: usize,
    hd: usize,
    window: Option<usize>,
    scale: f32,
) -> Vec<f32> {
    let kv_len = ck.len() / (n_kv * hd);
    let q_per_kv = n_q_heads / n_kv;
    let mut out = vec![0.0f32; n * n_q_heads * hd];
    out.par_chunks_mut(hd).enumerate().for_each(|(i, dst)| {
        let (t, h) = (i / n_q_heads, i % n_q_heads);
        let kvh = h / q_per_kv;
        let q_pos = q0 + t;
        let qr = &q[(t * n_q_heads + h) * hd..][..hd];

        let lo = window.map_or(0, |w| (q_pos + 1).saturating_sub(w));
        // keys [lo, q_pos] are visible (mask: q_pos - k_pos >= window).
        let mut scores = Vec::with_capacity(q_pos + 1 - lo);
        for kp in lo..=q_pos {
            debug_assert!(kp < kv_len);
            let kr = &ck[(kp * n_kv + kvh) * hd..][..hd];
            let dot: f64 = qr.iter().zip(kr).map(|(&a, &b)| a as f64 * b as f64).sum();
            scores.push(dot * scale as f64);
        }
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0f64;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        let mut acc = vec![0.0f64; hd];
        for (j, kp) in (lo..=q_pos).enumerate() {
            let wgt = scores[j] / sum;
            let vr = &cv[(kp * n_kv + kvh) * hd..][..hd];
            for (a, &v) in acc.iter_mut().zip(vr) {
                *a += wgt * v as f64;
            }
        }
        for (d, &a) in dst.iter_mut().zip(&acc) {
            *d = a as f32;
        }
    });
    out
}

/// `gelu_pytorch_tanh`: `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`.
fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
    let x = x as f64;
    (0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())) as f32
}

/// Read an F32 tensor fully into a Vec.
fn f32_tensor(gguf: &Gguf<'_>, name: &str) -> Result<Vec<f32>, RefError> {
    let info = gguf
        .tensor(name)
        .ok_or_else(|| RefError::MissingTensor(name.into()))?;
    if info.dtype != GgmlType::F32 {
        return Err(RefError::BadTensor {
            name: name.into(),
            what: format!("dtype {} != F32", info.dtype),
        });
    }
    let bytes = gguf.data_of(info);
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_tanh_reference_values() {
        // torch.nn.functional.gelu(x, approximate="tanh") at f64.
        assert_eq!(gelu_tanh(0.0), 0.0);
        assert!((gelu_tanh(1.0) - 0.841_192).abs() < 1e-6);
        assert!((gelu_tanh(-2.0) - (-0.045_402_3)).abs() < 1e-6);
        // Large |x|: saturates to x or 0.
        assert_eq!(gelu_tanh(20.0), 20.0);
        assert_eq!(gelu_tanh(-20.0), -0.0);
    }

    #[test]
    fn softmax_attention_uniform_when_keys_equal() {
        // 2 kv heads, 4 q heads, head_dim 2, 3 cached tokens, all keys
        // identical → uniform weights → output = mean of values.
        let hd = 2;
        let ck = vec![1.0f32; 3 * 2 * hd];
        let cv: Vec<f32> = (0..3 * 2 * hd).map(|i| i as f32).collect();
        let q = vec![1.0f32; 4 * hd];
        let out = attention(&q, &ck, &cv, 1, 2, 4, 2, hd, None, 1.0);
        // kv head 0 values at dim 0: tokens 0,1,2 → 0, 4, 8 → mean 4.
        assert!((out[0] - 4.0).abs() < 1e-6);
        // q heads 2,3 read kv head 1: dim 0 values 2, 6, 10 → mean 6.
        assert!((out[2 * hd] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn sliding_window_masks_old_keys() {
        // window 2: query at pos 2 sees keys {1, 2} only.
        let hd = 1;
        let ck = vec![1.0f32, 1.0, 1.0]; // 1 kv head, 3 tokens
        let cv = vec![10.0f32, 20.0, 30.0];
        let q = vec![1.0f32];
        let out = attention(&q, &ck, &cv, 1, 2, 1, 1, hd, Some(2), 1.0);
        assert!((out[0] - 25.0).abs() < 1e-6); // mean of 20, 30
    }
}
