//! M5 gate: in-memory incremental sessions (plan 00 milestone table —
//! "cache-on vs cache-off logits bit-identical", in the form that is
//! actually achievable, plan 03 §testing):
//!
//! 1. REPLAY determinism, bit-exact: resetting and re-running the same
//!    request sequence reproduces the final logits bitwise (same path
//!    shapes ⇒ same float ops).
//! 2. Resume vs cold, tolerance: a resumed conversation (suffix-only
//!    prefill over resident KV) generates the same greedy tokens as a
//!    cold session given the full prompt — the resident KV was written by
//!    decode (gemv) where the cold run uses prefill (gemm), so logits are
//!    only tolerance-equal, never bitwise (plan 03).
//! 3. Divergence resets cleanly and matches a cold run bitwise (both
//!    prefill from position 0 with identical chunk shapes).

use std::path::PathBuf;

use sg_model::{GenerateParams, GpuModel, SamplerParams, Session, StopReason};

fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SG_MODEL_GGUF") {
        return Some(p.into());
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/gemma-4-31B_q4_0-it.gguf");
    p.exists().then_some(p)
}

/// "The capital of Finland is" (ids pinned by the llama.cpp fixtures).
const P1: &[u32] = &[2, 818, 5279, 529, 37038, 563];

fn greedy(max_tokens: usize) -> GenerateParams {
    GenerateParams {
        sampling: SamplerParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
        },
        seed: 7,
        max_tokens,
    }
}

#[test]
fn session_resume_replay_and_divergence() {
    let Some(path) = model_path() else {
        eprintln!("skipping: model GGUF not present (set SG_MODEL_GGUF)");
        return;
    };
    let Ok(ctx) = sg_gpu::GpuContext::new() else {
        eprintln!("skipping: no usable GPU");
        return;
    };
    let file = sg_gguf::GgufFile::open(&path).expect("open model");
    let gguf = file.parse().expect("parse model");
    let model = GpuModel::new(&ctx, &gguf, 512, 64).expect("upload");
    let mut sess = Session::new(model).expect("session");

    // Turn 1.
    let (g1, r1) = sess.request(P1, &greedy(6), |_| true).expect("turn 1");
    assert!(matches!(r1, StopReason::MaxTokens | StopReason::Eos(_)));
    let hist_after_1 = sess.history().to_vec();

    // Turn 2: the API-shaped continuation — full conversation + new user
    // tokens. The resident history must be a prefix of it.
    let mut p2 = P1.to_vec();
    p2.extend_from_slice(&g1);
    p2.extend_from_slice(&[1003, 476, 2121]); // arbitrary new-turn tokens
    assert!(
        p2.starts_with(&hist_after_1),
        "history not a prefix of turn 2"
    );
    let resumed_from = sess.history().len();
    let (g2, _) = sess.request(&p2, &greedy(6), |_| true).expect("turn 2");
    let resumed_logits = sess.read_logits().expect("logits");
    assert!(
        resumed_from > 0 && resumed_from < p2.len(),
        "turn 2 did not resume incrementally (resumed_from {resumed_from})"
    );

    // (2) Resume vs cold: same greedy tokens from a cold session fed the
    // same full prompt (tolerance form — see header).
    sess.reset();
    let (g2_cold, _) = sess
        .request(&p2, &greedy(6), |_| true)
        .expect("cold turn 2");
    assert_eq!(
        g2, g2_cold,
        "resumed and cold greedy generations diverged (gemv/gemm low bits \
         flipping a knife-edge argmax would show here — investigate, don't relax)"
    );

    // (1) Replay determinism, bit-exact: rerun the exact request sequence.
    sess.reset();
    sess.request(P1, &greedy(6), |_| true)
        .expect("replay turn 1");
    let (g2_replay, _) = sess
        .request(&p2, &greedy(6), |_| true)
        .expect("replay turn 2");
    let replay_logits = sess.read_logits().expect("logits");
    assert_eq!(g2, g2_replay, "replay generated different tokens");
    assert_eq!(
        resumed_logits, replay_logits,
        "replayed session logits differ bitwise"
    );

    // (3) Divergence: a prompt sharing only part of P1 resets and matches
    // a fresh cold run bitwise (identical chunk shapes from position 0).
    let p3: Vec<u32> = [&P1[..3], &[9259, 1003, 476]].concat();
    let (g3, _) = sess.request(&p3, &greedy(4), |_| true).expect("diverged");
    let g3_logits = sess.read_logits().expect("logits");
    sess.reset();
    let (g3_cold, _) = sess.request(&p3, &greedy(4), |_| true).expect("cold p3");
    assert_eq!(g3, g3_cold);
    assert_eq!(
        g3_logits,
        sess.read_logits().expect("logits"),
        "divergence path logits differ bitwise from cold"
    );
}
