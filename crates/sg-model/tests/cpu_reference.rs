//! CPU reference model invariants against the real GGUF (plan 03 step 1).
//!
//! Self-skipping when the model is absent (Tier 1 CI); the logit-level
//! parity against llama.cpp fixtures lives in `llamacpp_parity.rs`.
//! These tests pin the *internal* contracts: decode == prefill bit-exactly
//! on the CPU oracle, K ≠ V divergence on global layers, softcap bounds.

use std::path::PathBuf;

use sg_model::{CpuKvCache, CpuModel};

fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SG_MODEL_GGUF") {
        return Some(p.into());
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/gemma-4-31B_q4_0-it.gguf");
    p.exists().then_some(p)
}

/// `<bos>` then a few common tokens — raw ids on purpose: this file tests
/// the model graph, not the tokenizer.
const PROMPT: &[u32] = &[2, 9259, 1003, 476, 2121];

#[test]
fn forward_invariants_and_decode_prefill_bitexact() {
    let Some(path) = model_path() else {
        eprintln!("skipping: model GGUF not present (set SG_MODEL_GGUF)");
        return;
    };
    let file = sg_gguf::GgufFile::open(&path).expect("open model");
    let gguf = file.parse().expect("parse model");
    let model = CpuModel::new(&gguf).expect("construct reference model");
    let vocab = model.desc.vocab_size;
    let cap = model.desc.final_logit_softcap;

    // K ≠ V on global layers, K == V-input on the shared projection:
    // capture the layer-5 (first global) taps.
    let mut k5 = Vec::new();
    let mut v5 = Vec::new();

    // Path A: prefill the whole prompt at once.
    let mut cache_a = CpuKvCache::new(&model.desc);
    let logits_a = model.forward(PROMPT, &mut cache_a, &mut |name, layer, data| {
        if layer == 5 && name == "k_rope" {
            k5 = data.to_vec();
        }
        if layer == 5 && name == "v_norm" {
            v5 = data.to_vec();
        }
    });
    assert_eq!(logits_a.len(), PROMPT.len() * vocab);
    assert_eq!(cache_a.len(), PROMPT.len());

    // Sanity: finite, softcap-bounded, non-degenerate logits.
    let last = &logits_a[(PROMPT.len() - 1) * vocab..];
    assert!(last.iter().all(|v| v.is_finite()));
    assert!(last.iter().all(|v| v.abs() <= cap));
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in last {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    assert!(hi - lo > 1.0, "logits suspiciously flat: [{lo}, {hi}]");

    // The pinned K≠V finding (docs/reference/gemma4-forward-graph.md):
    // same projection, different norms + rope ⇒ cached K must differ from
    // cached V on a global layer.
    assert_eq!(k5.len(), v5.len());
    assert!(!k5.is_empty(), "taps did not fire for layer 5");
    let diff = k5
        .iter()
        .zip(&v5)
        .filter(|(a, b)| (**a - **b).abs() > 1e-3)
        .count();
    assert!(
        diff > k5.len() / 2,
        "global-layer K and V are near-identical ({diff}/{} differ) — \
         the K≠V finding would be wrong",
        k5.len()
    );

    // Path B: prefill all but the last token, then decode it. The CPU
    // oracle performs identical f64 ops per token either way ⇒ bit-exact.
    let mut cache_b = CpuKvCache::new(&model.desc);
    let (head, tail) = PROMPT.split_at(PROMPT.len() - 1);
    model.forward(head, &mut cache_b, &mut |_, _, _| {});
    let logits_b = model.forward(tail, &mut cache_b, &mut |_, _, _| {});
    assert_eq!(logits_b.len(), vocab);
    assert_eq!(
        last,
        &logits_b[..],
        "decode logits differ bitwise from prefill logits"
    );
}
