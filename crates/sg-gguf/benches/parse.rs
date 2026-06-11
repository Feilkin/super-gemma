//! GGUF parse time on the real model file (plan 01). The parse cost is
//! dominated by materializing the tokenizer payload (262k vocab strings,
//! 514k merges) into owned metadata.
//!
//! Needs the model GGUF; prints a skip notice and registers nothing without
//! it. Run: `cargo bench -p sg-gguf --bench parse`.

use std::path::PathBuf;

use criterion::{Criterion, criterion_group, criterion_main};
use sg_gguf::{Gguf, GgufFile, ModelDesc};

fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SG_MODEL_GGUF") {
        return Some(p.into());
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/gemma-4-31B_q4_0-it.gguf");
    p.exists().then_some(p)
}

fn bench(c: &mut Criterion) {
    let Some(path) = model_path() else {
        eprintln!("skipping parse benches: model GGUF not present (set SG_MODEL_GGUF)");
        return;
    };
    let file = GgufFile::open(&path).expect("open model");

    let mut group = c.benchmark_group("gguf");
    group.sample_size(20);
    group.bench_function("parse_real_file", |b| {
        b.iter(|| Gguf::parse(file.bytes()).expect("parse"))
    });

    let gguf = file.parse().expect("parse");
    group.bench_function("model_desc_validate", |b| {
        b.iter(|| ModelDesc::from_gguf(&gguf).expect("validate"))
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
