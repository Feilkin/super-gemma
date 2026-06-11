//! Smoke tests against the real model GGUF (plan 01: parse + validate).
//!
//! Self-skipping when the model is absent, so Tier 1 CI (no model file)
//! stays green and Tier 2 on the target box exercises them for real.
//! Override the path with `SG_MODEL_GGUF`.

use std::path::PathBuf;

use sg_gguf::{GgmlType, GgufFile, LayerKind, ModelDesc};

fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SG_MODEL_GGUF") {
        return Some(p.into());
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/gemma-4-31B_q4_0-it.gguf");
    p.exists().then_some(p)
}

#[test]
fn real_gguf_parses_and_validates() {
    let Some(path) = model_path() else {
        eprintln!("skipping: model GGUF not present (set SG_MODEL_GGUF)");
        return;
    };
    let file = GgufFile::open(&path).expect("open/mmap model file");
    let gguf = file.parse().unwrap_or_else(|e| panic!("parse: {e}"));

    assert_eq!(gguf.version, 3);
    assert_eq!(gguf.tensors().len(), 833);

    let desc = ModelDesc::from_gguf(&gguf).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(desc.n_layers, 60);
    assert_eq!(desc.embed_dtype, GgmlType::Q6_K);
    assert_eq!(desc.layer_kinds[5], LayerKind::Global);

    // Spot-check data plumbing: a known F32 norm tensor must decode to sane
    // values (RMSNorm weights are O(1), never NaN/Inf/zero-everywhere).
    let norm = gguf.tensor_data("output_norm.weight").unwrap();
    assert_eq!(norm.len(), 5376 * 4);
    let vals: Vec<f32> = norm
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert!(vals.iter().all(|v| v.is_finite()), "non-finite norm weight");
    let mean_abs = vals.iter().map(|v| v.abs()).sum::<f32>() / vals.len() as f32;
    assert!(
        (0.01..100.0).contains(&mean_abs),
        "implausible output_norm magnitude {mean_abs}"
    );

    // And a Q4_0 tensor round-trips through the block reference type.
    let q = gguf.tensor_data("blk.0.attn_q.weight").unwrap();
    let blocks = sg_gguf::q4_0::blocks_from_bytes(q).unwrap();
    assert_eq!(blocks.len() as u64, 5376 * 8192 / 32);
    let mut out = [0.0f32; 32];
    blocks[0].dequantize(&mut out);
    assert!(out.iter().all(|v| v.is_finite()));
}
