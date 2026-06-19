//! M3 parity gate (plan 03 / plan 06 rung 3): the GPU forward graph vs the
//! CPU reference model on the real GGUF, layer by layer and end to end.
//!
//! Skips without the model file or a usable GPU (Tier 1 CI). The prompt is
//! fed token-by-token through the decode-shaped graph; after every layer
//! the GPU residual stream is compared against the CPU oracle's `layer_out`
//! tap, so a layer-type-specific bug (rope variant, K≠V, window mask)
//! localizes to its first layer immediately.
//!
//! Thresholds (initial calibration 2026-06-12; Q8-K update 2026-06-19): the
//! GPU runs f16 activations against the oracle's f32/f64. Decode's global
//! attention now reads the Q8 K cache (Piece A), so the residual stream
//! carries the int8/Q8 divergence-toward-llama.cpp envelope — the same one
//! `prefill_parity` bounds at 0.045 on the GLOBAL worst (not a per-layer
//! early-exit, since the error accumulates down the stack). Final-logit
//! agreement stays rank-based (argmax equal, top-20 overlap ≥ 18,
//! |Δ| ≤ 0.25 on logits with |cpu| > 1) — the real correctness gate, with
//! the f64 oracle, alongside perplexity and the kernel parity tests.

use std::path::PathBuf;

use sg_model::{CpuKvCache, CpuModel, GpuModel};

fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SG_MODEL_GGUF") {
        return Some(p.into());
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/gemma-4-31B_q4_0-it.gguf");
    p.exists().then_some(p)
}

fn gpu() -> Option<sg_gpu::GpuContext> {
    match sg_gpu::GpuContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("skipping: no usable GPU ({e})");
            None
        }
    }
}

const PROMPT: &[u32] = &[2, 9259, 1003, 476, 2121];

/// rms(got - want) / rms(want).
fn nrmse(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (&g, &w) in got.iter().zip(want) {
        num += (g as f64 - w as f64).powi(2);
        den += (w as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

#[test]
fn gpu_graph_matches_cpu_reference_per_layer_and_logits() {
    let Some(path) = model_path() else {
        eprintln!("skipping: model GGUF not present (set SG_MODEL_GGUF)");
        return;
    };
    let Some(ctx) = gpu() else { return };
    let file = sg_gguf::GgufFile::open(&path).expect("open model");
    let gguf = file.parse().expect("parse model");

    let cpu = CpuModel::new(&gguf).expect("CPU reference");
    let mut gpu = GpuModel::new(&ctx, &gguf, 64, 64).expect("GPU model upload");
    let n_layers = cpu.desc.n_layers;
    let vocab = cpu.desc.vocab_size;

    // One graph per layer (for localization) + the LM head graph. The full
    // single-graph path is exercised separately below.
    let layer_graphs: Vec<_> = (0..n_layers)
        .map(|i| gpu.record(i..i + 1, false).expect("record layer"))
        .collect();
    let logits_graph = gpu.record(0..0, true).expect("record logits");

    let mut cache = CpuKvCache::new(&cpu.desc);
    let mut worst: (f64, usize, usize) = (0.0, 0, 0); // (nrmse, token, layer)

    for (t, &tok) in PROMPT.iter().enumerate() {
        // CPU oracle, capturing every layer's residual-stream output.
        let mut layer_out: Vec<Vec<f32>> = vec![Vec::new(); n_layers];
        let cpu_logits = cpu.forward(&[tok], &mut cache, &mut |name, layer, data| {
            if name == "layer_out" {
                layer_out[layer] = data.to_vec();
            }
        });

        // GPU, layer by layer over the same token.
        gpu.stage_token(tok).expect("stage");
        for (i, g) in layer_graphs.iter().enumerate() {
            gpu.submit(g).expect("submit layer");
            let got = gpu.read_hidden().expect("read hidden");
            let e = nrmse(&got, &layer_out[i]);
            if e > worst.0 {
                worst = (e, t, i);
            }
        }
        gpu.submit(&logits_graph).expect("submit logits");
        let gpu_logits = gpu.read_logits().expect("read logits");
        gpu.advance();

        // End-to-end logit agreement for this token.
        let top = |v: &[f32]| {
            let mut idx: Vec<u32> = (0..vocab as u32).collect();
            idx.sort_unstable_by(|&a, &b| v[b as usize].total_cmp(&v[a as usize]));
            idx.truncate(20);
            idx
        };
        let (ct, gt) = (top(&cpu_logits), top(&gpu_logits));
        assert_eq!(ct[0], gt[0], "token {t}: argmax mismatch");
        let overlap = ct.iter().filter(|id| gt.contains(id)).count();
        assert!(overlap >= 18, "token {t}: top-20 overlap {overlap}/20");
        for &id in &ct {
            let (c, g) = (cpu_logits[id as usize], gpu_logits[id as usize]);
            if c.abs() > 1.0 {
                assert!(
                    (c - g).abs() <= 0.25,
                    "token {t} id {id}: cpu {c:.4} vs gpu {g:.4}"
                );
            }
        }
        eprintln!(
            "token {t}: layers ok (worst so far nrmse {:.5} @ token {} layer {}), \
             logits top-20 overlap {overlap}/20",
            worst.0, worst.1, worst.2
        );
    }

    // Per-layer envelope, asserted on the GLOBAL worst (the Q8-K decode
    // divergence accumulates down the stack — see the header). The f64-oracle
    // logit checks above are the real correctness gate.
    let tol = 0.045;
    assert!(
        worst.0 <= tol,
        "worst layer {} token {} ({:?}): nrmse {:.5} > {tol}",
        worst.2,
        worst.1,
        cpu.desc.layer_kinds[worst.2],
        worst.0,
    );
    eprintln!(
        "per-layer decode ok (worst nrmse {:.5} @ token {} layer {})",
        worst.0, worst.1, worst.2
    );

    // The same tokens through ONE pre-recorded full graph (the production
    // decode graph, driven only by step-buffer/embedding rewrites) must
    // reproduce the per-layer path bit-exactly: same kernels, same order.
    let gpu_logits_layered = gpu.read_logits().expect("read logits");
    let full = gpu.record(0..n_layers, true).expect("record full");
    gpu.reset();
    let mut gpu_logits_full = Vec::new();
    for &tok in PROMPT {
        gpu_logits_full = gpu.decode_step(&full, tok).expect("decode step");
    }
    assert_eq!(
        gpu_logits_full, gpu_logits_layered,
        "full pre-recorded graph diverges bitwise from the per-layer path"
    );
}
