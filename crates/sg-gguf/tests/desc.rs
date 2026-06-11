//! `ModelDesc` validation tests on fabricated metadata + tensor tables.
//!
//! These verify the *discrepancy reporting*; the ground-truth anchor is the
//! real-file test in `real_file.rs` (target box only).

use sg_gguf::meta::{MetaArray, MetaValue, Metadata};
use sg_gguf::{GgmlType, LayerKind, ModelDesc, TensorInfo};

/// Metadata matching the real QAT GGUF (values transcribed from the dump).
fn good_metadata() -> Metadata {
    let mut m = Metadata::default();
    let mut kv = |k: &str, v: MetaValue| assert!(m.insert(k.into(), v));
    kv("general.architecture", MetaValue::String("gemma4".into()));
    kv("gemma4.block_count", MetaValue::U32(60));
    kv("gemma4.context_length", MetaValue::U32(262_144));
    kv("gemma4.embedding_length", MetaValue::U32(5376));
    kv("gemma4.feed_forward_length", MetaValue::U32(21504));
    kv("gemma4.attention.head_count", MetaValue::U32(32));
    kv("gemma4.attention.sliding_window", MetaValue::U32(1024));
    kv("gemma4.attention.key_length", MetaValue::U32(512));
    kv("gemma4.attention.value_length", MetaValue::U32(512));
    kv("gemma4.attention.key_length_swa", MetaValue::U32(256));
    kv("gemma4.attention.value_length_swa", MetaValue::U32(256));
    kv("gemma4.attention.shared_kv_layers", MetaValue::U32(0));
    kv("gemma4.rope.dimension_count", MetaValue::U32(512));
    kv("gemma4.rope.dimension_count_swa", MetaValue::U32(256));
    kv("gemma4.rope.freq_base", MetaValue::F32(1_000_000.0));
    kv("gemma4.rope.freq_base_swa", MetaValue::F32(10_000.0));
    kv(
        "gemma4.attention.layer_norm_rms_epsilon",
        MetaValue::F32(1e-6),
    );
    kv("gemma4.final_logit_softcapping", MetaValue::F32(30.0));
    let kv_heads: Vec<i32> = (0..60).map(|i| if i % 6 == 5 { 4 } else { 16 }).collect();
    kv(
        "gemma4.attention.head_count_kv",
        MetaValue::Array(MetaArray::I32(kv_heads)),
    );
    let pattern: Vec<bool> = (0..60).map(|i| i % 6 != 5).collect();
    kv(
        "gemma4.attention.sliding_window_pattern",
        MetaValue::Array(MetaArray::Bool(pattern)),
    );
    m
}

/// Tensor table exactly matching the config-derived expectation. Offsets and
/// byte lengths are dummies — `ModelDesc` only reads names, dims, dtypes.
fn good_tensors() -> Vec<TensorInfo> {
    ModelDesc::expected_tensors()
        .into_iter()
        .map(|(name, dims, dtype)| TensorInfo {
            name,
            dims,
            dtype,
            offset: 0,
            byte_len: 0,
        })
        .collect()
}

#[test]
fn validates_a_conforming_file() {
    let desc =
        ModelDesc::validate(&good_metadata(), &good_tensors()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(desc.n_layers, 60);
    assert_eq!(desc.layer_kinds.len(), 60);
    assert_eq!(desc.layer_kinds[5], LayerKind::Global);
    assert_eq!(desc.layer_kinds[4], LayerKind::Sliding);
    assert_eq!(
        desc.layer_kinds
            .iter()
            .filter(|k| **k == LayerKind::Global)
            .count(),
        10
    );
    assert_eq!(desc.embed_dtype, GgmlType::Q6_K);
    assert_eq!(desc.global.head_dim, 512);
    assert_eq!(desc.sliding.n_kv_heads, 16);
}

#[test]
fn expected_tensor_set_shape() {
    let expected = ModelDesc::expected_tensors();
    // 3 top-level + 50 sliding × 14 + 10 global × 13 (no attn_v) = 833,
    // matching the real file's tensor count.
    assert_eq!(expected.len(), 3 + 50 * 14 + 10 * 13);
    assert!(!expected.iter().any(|(n, ..)| n == "blk.5.attn_v.weight"));
    assert!(expected.iter().any(|(n, ..)| n == "blk.4.attn_v.weight"));
}

#[test]
fn reports_metadata_mismatch_with_both_values() {
    let mut meta = good_metadata();
    // insert() rejects dups, so rebuild the key with a wrong value.
    let mut wrong = Metadata::default();
    for (k, v) in meta.iter() {
        let v = if k == "gemma4.block_count" {
            MetaValue::U32(48)
        } else {
            v.clone()
        };
        wrong.insert(k.to_owned(), v);
    }
    meta = wrong;

    let err = ModelDesc::validate(&meta, &good_tensors()).unwrap_err();
    assert_eq!(err.issues.len(), 1, "{err}");
    assert!(err.issues[0].contains("gemma4.block_count"), "{err}");
    assert!(
        err.issues[0].contains("48") && err.issues[0].contains("60"),
        "{err}"
    );
}

#[test]
fn reports_missing_wrong_shape_wrong_dtype_and_unexpected_tensors() {
    let mut tensors = good_tensors();
    // Missing: drop blk.0.attn_q.
    tensors.retain(|t| t.name != "blk.0.attn_q.weight");
    // Wrong shape: give blk.5 (global) a sliding-sized attn_k.
    tensors
        .iter_mut()
        .find(|t| t.name == "blk.5.attn_k.weight")
        .unwrap()
        .dims = vec![5376, 4096];
    // Wrong dtype: f16 embeddings.
    tensors
        .iter_mut()
        .find(|t| t.name == "token_embd.weight")
        .unwrap()
        .dtype = GgmlType::F16;
    // Unexpected: a v_proj on a global layer.
    tensors.push(TensorInfo {
        name: "blk.5.attn_v.weight".into(),
        dims: vec![5376, 2048],
        dtype: GgmlType::Q4_0,
        offset: 0,
        byte_len: 0,
    });

    let err = ModelDesc::validate(&good_metadata(), &tensors).unwrap_err();
    let all = err.issues.join("\n");
    assert_eq!(err.issues.len(), 4, "{err}");
    assert!(
        all.contains("missing tensor `blk.0.attn_q.weight`"),
        "{err}"
    );
    assert!(
        all.contains("blk.5.attn_k.weight") && all.contains("[5376, 4096]"),
        "{err}"
    );
    assert!(
        all.contains("token_embd.weight") && all.contains("F16"),
        "{err}"
    );
    assert!(
        all.contains("unexpected tensor `blk.5.attn_v.weight`"),
        "{err}"
    );
}

#[test]
fn reports_per_layer_array_disagreements() {
    let mut wrong = Metadata::default();
    for (k, v) in good_metadata().iter() {
        let v = match k {
            // Swap one layer's KV head count: global where sliding expected.
            "gemma4.attention.head_count_kv" => {
                let mut counts: Vec<i32> =
                    (0..60).map(|i| if i % 6 == 5 { 4 } else { 16 }).collect();
                counts[0] = 4;
                MetaValue::Array(MetaArray::I32(counts))
            }
            // Truncate the pattern.
            "gemma4.attention.sliding_window_pattern" => {
                MetaValue::Array(MetaArray::Bool(vec![true; 59]))
            }
            _ => v.clone(),
        };
        wrong.insert(k.to_owned(), v);
    }

    let err = ModelDesc::validate(&wrong, &good_tensors()).unwrap_err();
    let all = err.issues.join("\n");
    assert!(all.contains("head_count_kv`[0]"), "{err}");
    assert!(all.contains("sliding_window_pattern`: 59 entries"), "{err}");
}

#[test]
fn missing_metadata_keys_are_each_reported() {
    let err = ModelDesc::validate(&Metadata::default(), &good_tensors()).unwrap_err();
    let all = err.issues.join("\n");
    assert!(
        all.contains("missing metadata key `general.architecture`"),
        "{err}"
    );
    assert!(
        all.contains("missing metadata key `gemma4.block_count`"),
        "{err}"
    );
    assert!(all.contains("gemma4.attention.head_count_kv"), "{err}");
}
