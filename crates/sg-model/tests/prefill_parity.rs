//! Chunked-prefill parity (plan 03 §prefill, M4): the gemm-based prefill
//! graph vs the CPU oracle, and decode==prefill consistency. Skips without
//! the model file or a GPU.
//!
//! The prompt spans multiple chunks (max_chunk 64 → 64 + 64 + padded tail)
//! so chunk boundaries, the two-range sliding attention with non-zero q0,
//! and tail padding are all exercised. Ring wraparound at q0 > 1024 is
//! covered at kernel level (`sg-gpu` parity, q0 = 1531 case) — a >1K-token
//! oracle run here would cost minutes for no extra graph coverage.
//!
//! Thresholds mirror `gpu_parity.rs` (f16 graph vs f32/f64 oracle): argmax
//! equal, top-20 overlap ≥ 18, |Δ| ≤ 0.25 on logits with |cpu| > 1. The
//! decode-vs-prefill check is GPU-vs-GPU and tighter (overlap ≥ 19,
//! |Δ| ≤ 0.15): same kernels, different attention path + matmul kernels
//! (gemv vs gemm differ in low bits by construction — never bitwise,
//! plan 03 §testing).

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

/// Deterministic 149-token prompt: BOS then LCG-spread ids. Parity is a
/// numerics property — the ids only need to be valid and varied.
fn prompt(vocab: usize) -> Vec<u32> {
    let mut ids = vec![2u32];
    let mut s = 0x5EED_u64;
    for _ in 0..148 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Skip the special/control range at the very start of the vocab.
        ids.push(1000 + ((s >> 33) as u32) % (vocab as u32 - 2000));
    }
    ids
}

fn top20(v: &[f32]) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..v.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| v[b as usize].total_cmp(&v[a as usize]));
    idx.truncate(20);
    idx
}

fn assert_logit_agreement(a: &[f32], b: &[f32], min_overlap: usize, dtol: f32, what: &str) {
    let (ta, tb) = (top20(a), top20(b));
    assert_eq!(ta[0], tb[0], "{what}: argmax mismatch");
    let overlap = ta.iter().filter(|id| tb.contains(id)).count();
    assert!(
        overlap >= min_overlap,
        "{what}: top-20 overlap {overlap}/20"
    );
    let mut max_d = 0.0f32;
    for &id in &ta {
        let (x, y) = (a[id as usize], b[id as usize]);
        if x.abs() > 1.0 {
            max_d = max_d.max((x - y).abs());
            assert!((x - y).abs() <= dtol, "{what}: id {id}: {x:.4} vs {y:.4}");
        }
    }
    eprintln!("{what}: overlap {overlap}/20, max Δ {max_d:.4} — ok");
}

/// Per-layer localization (the prefill analog of `gpu_parity`'s decode
/// check): one 64-token chunk, every layer's residual stream vs the
/// oracle. A layer-kind-specific prefill bug (two-range window, gemm
/// path, append ordering) pins to its first layer here.
#[test]
fn prefill_single_chunk_per_layer_parity() {
    let Some(path) = model_path() else {
        eprintln!("skipping: model GGUF not present (set SG_MODEL_GGUF)");
        return;
    };
    let Some(ctx) = gpu() else { return };
    let file = sg_gguf::GgufFile::open(&path).expect("open model");
    let gguf = file.parse().expect("parse model");
    let cpu = CpuModel::new(&gguf).expect("CPU reference");
    let gpu = GpuModel::new(&ctx, &gguf, 256, 64).expect("GPU model upload");

    let ids: Vec<u32> = prompt(cpu.desc.vocab_size)[..64].to_vec();
    let n = ids.len();

    let mut layer_out: Vec<Vec<f32>> = vec![Vec::new(); cpu.desc.n_layers];
    let mut cache = CpuKvCache::new(&cpu.desc);
    cpu.forward(&ids, &mut cache, &mut |name, layer, data| {
        if name == "layer_out" {
            layer_out[layer] = data.to_vec();
        }
    });

    gpu.stage_prefill_chunk(&ids).expect("stage");
    let mut worst = (0.0f64, 0usize);
    for (i, want) in layer_out.iter().enumerate() {
        let g = gpu.record_prefill_layers(i..i + 1, 64, n).expect("record");
        gpu.submit(&g).expect("submit");
        let got = gpu.read_prefill_hidden(n).expect("read");
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&g, &w) in got.iter().zip(want) {
            num += (g as f64 - w as f64).powi(2);
            den += (w as f64).powi(2);
        }
        let e = (num / den.max(1e-30)).sqrt();
        if e > worst.0 {
            worst = (e, i);
        }
    }
    // int8-ffn carries a wider per-layer envelope than the f16 path (the whole
    // attention block + FFN on int8 → worst ~0.04) — this is divergence *toward*
    // llama.cpp (which also int8-quantizes activations), not quality loss: the
    // perplexity gate passes (code 21.95 vs 22.33, wikitext <0.5%). Asserted on
    // the global worst (not per-layer early-exit) so the bound is the real max.
    let tol = if cfg!(feature = "int8-ffn") { 0.045 } else { 0.02 };
    assert!(
        worst.0 <= tol,
        "worst layer {} ({:?}): nrmse {:.5} > {tol}",
        worst.1,
        cpu.desc.layer_kinds[worst.1],
        worst.0,
    );
    eprintln!(
        "per-layer prefill ok (worst nrmse {:.5} @ layer {})",
        worst.0, worst.1
    );
}

#[test]
fn chunked_prefill_matches_oracle_and_decode() {
    let Some(path) = model_path() else {
        eprintln!("skipping: model GGUF not present (set SG_MODEL_GGUF)");
        return;
    };
    let Some(ctx) = gpu() else { return };
    let file = sg_gguf::GgufFile::open(&path).expect("open model");
    let gguf = file.parse().expect("parse model");

    let cpu = CpuModel::new(&gguf).expect("CPU reference");
    let ids = prompt(cpu.desc.vocab_size);

    // Oracle logits for the full prompt.
    let mut cache = CpuKvCache::new(&cpu.desc);
    let cpu_logits = cpu.forward(&ids, &mut cache, &mut |_, _, _| {});

    // GPU chunked prefill (3 chunks: 64 + 64 + 21-token padded tail).
    let mut gpu = GpuModel::new(&ctx, &gguf, 256, 64).expect("GPU model upload");
    let gpu_prefill = gpu.prefill(&ids).expect("prefill");
    assert_logit_agreement(&cpu_logits, &gpu_prefill, 18, 0.25, "prefill vs oracle");

    // decode == prefill: (prefill N−1, decode 1) must agree with
    // (prefill N) — catches ring/rope/position bugs across the two paths.
    gpu.reset();
    let decode_graph = gpu
        .record(0..cpu.desc.n_layers, true)
        .expect("record decode");
    let (head, tail) = ids.split_at(ids.len() - 1);
    gpu.prefill(head).expect("prefill head");
    let gpu_decode = gpu
        .decode_step(&decode_graph, tail[0])
        .expect("decode step");
    assert_logit_agreement(&gpu_prefill, &gpu_decode, 19, 0.15, "prefill vs decode");
}
