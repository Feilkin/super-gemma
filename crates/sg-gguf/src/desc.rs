//! `ModelDesc`: the validated description of the one model this server runs.
//!
//! Built by cross-checking three sources (plan 01): the `gemma4.*` GGUF
//! metadata, the actual tensor table (names, shapes, dtypes), and hard-coded
//! expectations from the bf16 `config.json` (checked in under
//! `docs/reference/`, summarized in plan 00). *Any* disagreement fails with a
//! report listing every discrepancy — the primary defense against "QAT GGUF
//! doesn't match the config" surprises.
//!
//! Facts the real file resolved (2026-06-11):
//! - Token embeddings (tied LM head) are **Q6_K**, not f16/Q8_0.
//! - Global layers ship **no `attn_v` tensor at all**: K=V is materialized as
//!   the single `attn_k` projection (5376 → 2048 = 4 heads × 512).
//! - Every layer has a scalar `layer_output_scale.weight` (F32 `[1]`) that
//!   appears in neither config.json nor plan 00 — graph semantics are an M3
//!   verify-item.
//! - A top-level `rope_freqs.weight` (F32 `[256]`) ships precomputed
//!   frequencies; usage is an M3 verify-item alongside proportional RoPE.

use crate::meta::Metadata;
use crate::parse::Gguf;
use crate::tensor::{GgmlType, TensorInfo};

/// Which attention scheme a layer uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Sliding,
    Global,
}

/// Geometry of one attention flavor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttnGeometry {
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
}

/// Validated model description. All numbers agree across metadata, tensor
/// shapes, and config.json by construction.
#[derive(Debug, Clone)]
pub struct ModelDesc {
    pub n_layers: usize,
    pub hidden_size: usize,
    pub ffn_size: usize,
    pub vocab_size: usize,
    pub max_context: usize,
    pub n_q_heads: usize,
    pub sliding: AttnGeometry,
    pub global: AttnGeometry,
    pub sliding_window: usize,
    /// Per-layer attention kind (50 sliding, 10 global; every 6th is global).
    pub layer_kinds: Vec<LayerKind>,
    pub rms_norm_eps: f32,
    pub final_logit_softcap: f32,
    /// dtype of `token_embd.weight`, which is also the tied LM head: Q6_K.
    pub embed_dtype: GgmlType,
}

/// Hard-coded expectations from `docs/reference/gemma-4-31b-it.config.json`
/// (plan 00 model-facts table) plus facts pinned from the first real-file
/// parse. Source of truth for the third leg of the validation.
mod expect {
    use crate::tensor::GgmlType;

    pub const ARCHITECTURE: &str = "gemma4";
    pub const N_LAYERS: usize = 60;
    pub const HIDDEN: u64 = 5376;
    pub const FFN: u64 = 21504;
    pub const VOCAB: u64 = 262_144;
    pub const CONTEXT: u64 = 262_144;
    pub const N_Q_HEADS: u64 = 32;
    pub const SLIDING_KV_HEADS: u64 = 16;
    pub const SLIDING_HEAD_DIM: u64 = 256;
    pub const SLIDING_THETA: f32 = 10_000.0;
    pub const SLIDING_WINDOW: u64 = 1024;
    pub const GLOBAL_KV_HEADS: u64 = 4;
    pub const GLOBAL_HEAD_DIM: u64 = 512;
    pub const GLOBAL_THETA: f32 = 1_000_000.0;
    pub const RMS_EPS: f32 = 1e-6;
    pub const SOFTCAP: f32 = 30.0;
    pub const EMBED_DTYPE: GgmlType = GgmlType::Q6_K;
    pub const WEIGHT_DTYPE: GgmlType = GgmlType::Q4_0;

    /// Layer pattern: 5× sliding then 1× global, repeated.
    pub fn layer_kind(layer: usize) -> super::LayerKind {
        if layer % 6 == 5 {
            super::LayerKind::Global
        } else {
            super::LayerKind::Sliding
        }
    }
}

/// The validation report: every discrepancy found, one line each.
#[derive(Debug, thiserror::Error)]
#[error(
    "model description validation failed with {n} issue(s):\n{report}",
    n = issues.len(),
    report = issues.join("\n")
)]
pub struct DescError {
    pub issues: Vec<String>,
}

impl ModelDesc {
    pub fn from_gguf(gguf: &Gguf<'_>) -> Result<Self, DescError> {
        Self::validate(&gguf.metadata, gguf.tensors())
    }

    /// Cross-check metadata, tensor table, and config expectations; build the
    /// description only if every check passes.
    pub fn validate(meta: &Metadata, tensors: &[TensorInfo]) -> Result<Self, DescError> {
        let mut issues = Vec::new();
        check_metadata(meta, &mut issues);
        check_per_layer_metadata(meta, &mut issues);
        check_tensors(tensors, &mut issues);

        if !issues.is_empty() {
            return Err(DescError { issues });
        }

        Ok(Self {
            n_layers: expect::N_LAYERS,
            hidden_size: expect::HIDDEN as usize,
            ffn_size: expect::FFN as usize,
            vocab_size: expect::VOCAB as usize,
            max_context: expect::CONTEXT as usize,
            n_q_heads: expect::N_Q_HEADS as usize,
            sliding: AttnGeometry {
                n_kv_heads: expect::SLIDING_KV_HEADS as usize,
                head_dim: expect::SLIDING_HEAD_DIM as usize,
                rope_theta: expect::SLIDING_THETA,
            },
            global: AttnGeometry {
                n_kv_heads: expect::GLOBAL_KV_HEADS as usize,
                head_dim: expect::GLOBAL_HEAD_DIM as usize,
                rope_theta: expect::GLOBAL_THETA,
            },
            sliding_window: expect::SLIDING_WINDOW as usize,
            layer_kinds: (0..expect::N_LAYERS).map(expect::layer_kind).collect(),
            rms_norm_eps: expect::RMS_EPS,
            final_logit_softcap: expect::SOFTCAP,
            embed_dtype: expect::EMBED_DTYPE,
        })
    }

    /// Every tensor the file must contain: `(name, dims in ggml order,
    /// dtype)`. Also drives the weight uploader's layout pass.
    pub fn expected_tensors() -> Vec<(String, Vec<u64>, GgmlType)> {
        use expect::*;
        let mut v: Vec<(String, Vec<u64>, GgmlType)> = vec![
            ("token_embd.weight".into(), vec![HIDDEN, VOCAB], EMBED_DTYPE),
            ("output_norm.weight".into(), vec![HIDDEN], GgmlType::F32),
            // Precomputed RoPE frequencies (global layers' proportional rope;
            // 256 = head_dim 512 / 2). Usage verified in M3.
            (
                "rope_freqs.weight".into(),
                vec![GLOBAL_HEAD_DIM / 2],
                GgmlType::F32,
            ),
        ];
        for layer in 0..N_LAYERS {
            let kind = layer_kind(layer);
            let (kv_heads, head_dim) = match kind {
                LayerKind::Sliding => (SLIDING_KV_HEADS, SLIDING_HEAD_DIM),
                LayerKind::Global => (GLOBAL_KV_HEADS, GLOBAL_HEAD_DIM),
            };
            let q_dim = N_Q_HEADS * head_dim;
            let kv_dim = kv_heads * head_dim;
            let mut t = |suffix: &str, dims: Vec<u64>, dtype: GgmlType| {
                v.push((format!("blk.{layer}.{suffix}.weight"), dims, dtype));
            };
            t("attn_q", vec![HIDDEN, q_dim], WEIGHT_DTYPE);
            t("attn_k", vec![HIDDEN, kv_dim], WEIGHT_DTYPE);
            if kind == LayerKind::Sliding {
                // Global layers share K=V: a single attn_k tensor, no attn_v.
                t("attn_v", vec![HIDDEN, kv_dim], WEIGHT_DTYPE);
            }
            t("attn_output", vec![q_dim, HIDDEN], WEIGHT_DTYPE);
            t("attn_q_norm", vec![head_dim], GgmlType::F32);
            t("attn_k_norm", vec![head_dim], GgmlType::F32);
            t("attn_norm", vec![HIDDEN], GgmlType::F32);
            t("post_attention_norm", vec![HIDDEN], GgmlType::F32);
            t("ffn_norm", vec![HIDDEN], GgmlType::F32);
            t("post_ffw_norm", vec![HIDDEN], GgmlType::F32);
            t("ffn_gate", vec![HIDDEN, FFN], WEIGHT_DTYPE);
            t("ffn_up", vec![HIDDEN, FFN], WEIGHT_DTYPE);
            t("ffn_down", vec![FFN, HIDDEN], WEIGHT_DTYPE);
            // Per-layer output scalar: in the file, not in config.json.
            t("layer_output_scale", vec![1], GgmlType::F32);
        }
        v
    }
}

/// Compare scalar metadata against config expectations (two of the three
/// sources; tensor shapes weigh in via `check_tensors`).
fn check_metadata(meta: &Metadata, issues: &mut Vec<String>) {
    let mut str_key = |key: &str, config: &str| match meta.get_str(key) {
        Err(e) => issues.push(e.to_string()),
        Ok(None) => issues.push(format!("missing metadata key `{key}`")),
        Ok(Some(v)) if v != config => {
            issues.push(format!("`{key}`: metadata {v:?} != config {config:?}"));
        }
        Ok(Some(_)) => {}
    };
    str_key("general.architecture", expect::ARCHITECTURE);

    let mut uint = |key: &str, config: u64| match meta.get_uint(key) {
        Err(e) => issues.push(e.to_string()),
        Ok(None) => issues.push(format!("missing metadata key `{key}`")),
        Ok(Some(v)) if v != config => {
            issues.push(format!("`{key}`: metadata {v} != config {config}"));
        }
        Ok(Some(_)) => {}
    };
    uint("gemma4.block_count", expect::N_LAYERS as u64);
    uint("gemma4.context_length", expect::CONTEXT);
    uint("gemma4.embedding_length", expect::HIDDEN);
    uint("gemma4.feed_forward_length", expect::FFN);
    uint("gemma4.attention.head_count", expect::N_Q_HEADS);
    uint("gemma4.attention.sliding_window", expect::SLIDING_WINDOW);
    uint("gemma4.attention.key_length", expect::GLOBAL_HEAD_DIM);
    uint("gemma4.attention.value_length", expect::GLOBAL_HEAD_DIM);
    uint("gemma4.attention.key_length_swa", expect::SLIDING_HEAD_DIM);
    uint(
        "gemma4.attention.value_length_swa",
        expect::SLIDING_HEAD_DIM,
    );
    uint("gemma4.attention.shared_kv_layers", 0);
    uint("gemma4.rope.dimension_count", expect::GLOBAL_HEAD_DIM);
    uint("gemma4.rope.dimension_count_swa", expect::SLIDING_HEAD_DIM);

    let mut f32_key = |key: &str, config: f32| match meta.get_f32(key) {
        Err(e) => issues.push(e.to_string()),
        Ok(None) => issues.push(format!("missing metadata key `{key}`")),
        Ok(Some(v)) if v != config => {
            issues.push(format!("`{key}`: metadata {v} != config {config}"));
        }
        Ok(Some(_)) => {}
    };
    f32_key("gemma4.rope.freq_base", expect::GLOBAL_THETA);
    f32_key("gemma4.rope.freq_base_swa", expect::SLIDING_THETA);
    f32_key("gemma4.attention.layer_norm_rms_epsilon", expect::RMS_EPS);
    f32_key("gemma4.final_logit_softcapping", expect::SOFTCAP);
}

/// Per-layer arrays: KV head counts and the sliding/global pattern.
fn check_per_layer_metadata(meta: &Metadata, issues: &mut Vec<String>) {
    match meta.require_i32_array("gemma4.attention.head_count_kv") {
        Err(e) => issues.push(e.to_string()),
        Ok(counts) => {
            if counts.len() != expect::N_LAYERS {
                issues.push(format!(
                    "`gemma4.attention.head_count_kv`: {} entries, config has {} layers",
                    counts.len(),
                    expect::N_LAYERS
                ));
            }
            for (layer, &n) in counts.iter().enumerate() {
                let config = match expect::layer_kind(layer) {
                    LayerKind::Sliding => expect::SLIDING_KV_HEADS,
                    LayerKind::Global => expect::GLOBAL_KV_HEADS,
                };
                if n as i64 != config as i64 {
                    issues.push(format!(
                        "`gemma4.attention.head_count_kv`[{layer}]: metadata {n} != config \
                         {config} ({:?} layer)",
                        expect::layer_kind(layer)
                    ));
                }
            }
        }
    }

    match meta.require_bool_array("gemma4.attention.sliding_window_pattern") {
        Err(e) => issues.push(e.to_string()),
        Ok(pattern) => {
            if pattern.len() != expect::N_LAYERS {
                issues.push(format!(
                    "`gemma4.attention.sliding_window_pattern`: {} entries, config has {} layers",
                    pattern.len(),
                    expect::N_LAYERS
                ));
            }
            for (layer, &sliding) in pattern.iter().enumerate() {
                let config = expect::layer_kind(layer) == LayerKind::Sliding;
                if sliding != config {
                    issues.push(format!(
                        "`gemma4.attention.sliding_window_pattern`[{layer}]: metadata {sliding} \
                         != config {config}"
                    ));
                }
            }
        }
    }
}

/// Diff the actual tensor table against the config-derived expectation, both
/// directions.
fn check_tensors(tensors: &[TensorInfo], issues: &mut Vec<String>) {
    let actual: std::collections::HashMap<&str, &TensorInfo> =
        tensors.iter().map(|t| (t.name.as_str(), t)).collect();

    let expected = ModelDesc::expected_tensors();
    for (name, dims, dtype) in &expected {
        let Some(t) = actual.get(name.as_str()) else {
            issues.push(format!(
                "missing tensor `{name}` (expected {dtype} {dims:?})"
            ));
            continue;
        };
        if &t.dims != dims {
            issues.push(format!(
                "tensor `{name}`: file shape {:?} != config-derived {dims:?}",
                t.dims
            ));
        }
        if t.dtype != *dtype {
            issues.push(format!(
                "tensor `{name}`: file dtype {} != expected {dtype}",
                t.dtype
            ));
        }
    }

    let expected_names: std::collections::HashSet<&str> =
        expected.iter().map(|(n, _, _)| n.as_str()).collect();
    for t in tensors {
        if !expected_names.contains(t.name.as_str()) {
            issues.push(format!(
                "unexpected tensor `{}` ({} {:?}) — not in the config-derived set",
                t.name, t.dtype, t.dims
            ));
        }
    }
}
