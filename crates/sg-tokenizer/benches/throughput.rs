//! Tokenizer throughput (plan 01 target: > 1 M tok/s encode — it sits on the
//! count_tokens endpoint and the prefill path).
//!
//! Needs the real model GGUF for the vocab; prints a skip notice and
//! registers nothing without it. Run: `cargo bench -p sg-tokenizer`.

use std::path::PathBuf;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use sg_tokenizer::{SpecialTokens, Tokenizer};

fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SG_MODEL_GGUF") {
        return Some(p.into());
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/gemma-4-31B_q4_0-it.gguf");
    p.exists().then_some(p)
}

/// The parity corpus doubles as a realistic benchmark mix (code, prose,
/// multilingual, whitespace pathologies).
fn corpus() -> Vec<String> {
    let fixture = include_str!("../tests/fixtures/encode_parity.jsonl");
    fixture
        .lines()
        .skip(1) // header
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).expect("fixture line");
            v["text"].as_str().expect("text").to_owned()
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let Some(path) = model_path() else {
        eprintln!("skipping tokenizer benches: model GGUF not present (set SG_MODEL_GGUF)");
        return;
    };
    let file = sg_gguf::GgufFile::open(&path).expect("open model");
    let gguf = file.parse().expect("parse model");
    let tokenizer = Tokenizer::from_metadata(&gguf.metadata).expect("vocab");

    let lines = corpus();
    let line_tokens: u64 = lines
        .iter()
        .map(|t| tokenizer.encode(t, SpecialTokens::Match).len() as u64)
        .sum();

    // One ~300 KB contiguous document: the worst case for this tokenizer's
    // no-pre-tokenization BPE (one merge heap spanning the whole text).
    let doc: String = lines.join("\n");
    let doc_ids = tokenizer.encode(&doc, SpecialTokens::Plain);
    let doc_tokens = doc_ids.len() as u64;

    let mut group = c.benchmark_group("encode");
    group.throughput(Throughput::Elements(line_tokens));
    group.bench_function("parity_corpus_lines", |b| {
        b.iter(|| {
            let mut n = 0usize;
            for text in &lines {
                n += tokenizer.encode(text, SpecialTokens::Match).len();
            }
            n
        })
    });
    group.throughput(Throughput::Elements(doc_tokens));
    group.bench_function("single_long_doc", |b| {
        b.iter(|| tokenizer.encode(&doc, SpecialTokens::Plain).len())
    });
    group.finish();

    let mut group = c.benchmark_group("decode");
    group.throughput(Throughput::Elements(doc_tokens));
    group.bench_function("single_long_doc", |b| {
        b.iter(|| tokenizer.decode(&doc_ids).expect("decode").len())
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
