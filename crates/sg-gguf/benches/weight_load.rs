//! Weight-load wall time on the target (plan 01): the full 16.4 GiB data
//! section through `DirectSource` (O_DIRECT) and `MmapCopySource`.
//!
//! Each iteration moves ~16 GiB, so this bench takes a few minutes; run it
//! deliberately: `cargo bench -p sg-gguf --bench weight_load`.
//! Needs the model GGUF; prints a skip notice and registers nothing without
//! it.

use std::path::PathBuf;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use sg_gguf::{MmapCopySource, WeightSource};

fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SG_MODEL_GGUF") {
        return Some(p.into());
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/gemma-4-31B_q4_0-it.gguf");
    p.exists().then_some(p)
}

fn bench(c: &mut Criterion) {
    let Some(path) = model_path() else {
        eprintln!("skipping weight-load benches: model GGUF not present (set SG_MODEL_GGUF)");
        return;
    };
    let file = sg_gguf::GgufFile::open(&path).expect("open model");
    let gguf = file.parse().expect("parse");
    let len = gguf.data_section().len();
    let mut dst = vec![0u8; len];

    let mut group = c.benchmark_group("weight_load");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(6))
        .measurement_time(Duration::from_secs(60));
    group.throughput(Throughput::Bytes(len as u64));

    #[cfg(target_os = "linux")]
    {
        let direct = sg_gguf::DirectSource::new(&path, &gguf);
        group.bench_function("o_direct_pread", |b| {
            b.iter(|| direct.load(&mut dst).expect("load"))
        });
    }

    // Page-cache path; the first iteration faults the mmap in, later ones
    // measure cache-warm memcpy — representative of a restart soon after a
    // previous load.
    let mmap = MmapCopySource::new(&gguf);
    group.bench_function("mmap_copy", |b| {
        b.iter(|| mmap.load(&mut dst).expect("load"))
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
