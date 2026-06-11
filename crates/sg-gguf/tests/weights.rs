//! WeightSource tests: both load paths must reproduce the data section
//! byte-for-byte, regardless of destination alignment phase.

mod common;

use common::Builder;
use sg_gguf::{GgufFile, MmapCopySource, WeightSource};

/// Synthetic file with enough data to span several 4 KiB blocks and an
/// odd-sized tail. `name` keeps parallel tests from racing on one path.
fn write_fixture(name: &str) -> std::path::PathBuf {
    let q4_blocks = 1500; // 27000 bytes of Q4_0
    let q4_data: Vec<u8> = (0..q4_blocks * 18).map(|i| (i % 251) as u8).collect();
    let f32_data: Vec<u8> = (0..5000u32)
        .flat_map(|i| (i as f32).to_le_bytes())
        .collect();
    let bytes = Builder::new()
        .kv_str("general.architecture", "test")
        .tensor("a", &[32 * q4_blocks as u64], 2, &q4_data)
        .tensor("b", &[5000], 0, &f32_data)
        .tensor("c", &[3], 0, &[7u8; 12]) // 12-byte tail
        .build();
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let path = dir.join(format!("weights_fixture_{name}.gguf"));
    std::fs::write(&path, bytes).expect("write fixture");
    path
}

/// Destination buffer sliced from a larger allocation so its 4 KiB phase
/// relative to the file's data offset is controlled.
fn load_with_phase(source: &dyn WeightSource, data_offset: usize, matched: bool) -> Vec<u8> {
    let len = source.data_len();
    let mut backing = vec![0u8; len + 8192];
    let base = backing.as_ptr() as usize;
    let want_rem = if matched {
        data_offset % 4096
    } else {
        (data_offset + 1234) % 4096
    };
    let start = (want_rem + 4096 - base % 4096) % 4096;
    source.load(&mut backing[start..start + len]).expect("load");
    backing[start..start + len].to_vec()
}

#[test]
fn mmap_copy_reproduces_the_data_section() {
    let path = write_fixture("mmap");
    let file = GgufFile::open(&path).unwrap();
    let gguf = file.parse().unwrap();
    let source = MmapCopySource::new(&gguf);

    assert_eq!(source.data_len(), gguf.data_section().len());
    let mut dst = vec![0u8; source.data_len()];
    source.load(&mut dst).unwrap();
    assert_eq!(dst, gguf.data_section());

    // Wrong-size destination is rejected.
    let mut short = vec![0u8; source.data_len() - 1];
    assert!(source.load(&mut short).is_err());
}

/// Full-size load of the real model's data section (target box only;
/// self-skips like the other real-file tests).
#[cfg(target_os = "linux")]
#[test]
fn o_direct_loads_the_real_model() {
    let path = match std::env::var("SG_MODEL_GGUF") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => {
            let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../models/gemma-4-31B_q4_0-it.gguf");
            if !p.exists() {
                eprintln!("skipping: model GGUF not present (set SG_MODEL_GGUF)");
                return;
            }
            p
        }
    };
    let file = GgufFile::open(&path).unwrap();
    let gguf = file.parse().unwrap();
    let source = sg_gguf::DirectSource::new(&path, &gguf);

    let start = std::time::Instant::now();
    let mut dst = vec![0u8; source.data_len()];
    source
        .load(&mut dst)
        .unwrap_or_else(|e| panic!("load: {e}"));
    let secs = start.elapsed().as_secs_f64();
    let gib = source.data_len() as f64 / (1u64 << 30) as f64;
    eprintln!(
        "loaded {gib:.1} GiB in {secs:.2}s ({:.2} GiB/s)",
        gib / secs
    );

    // Sampled comparison against the mmap (full memcmp would double the
    // test's IO time for no extra signal).
    let data = gguf.data_section();
    assert_eq!(dst.len(), data.len());
    let step = data.len() / 64;
    for i in 0..64 {
        let at = i * step;
        let len = (1 << 20).min(data.len() - at);
        assert_eq!(dst[at..at + len], data[at..at + len], "region at {at}");
    }
    assert_eq!(dst[data.len() - 4096..], data[data.len() - 4096..], "tail");
}

#[cfg(target_os = "linux")]
#[test]
fn o_direct_reproduces_the_data_section_in_both_phases() {
    let path = write_fixture("direct");
    let file = GgufFile::open(&path).unwrap();
    let gguf = file.parse().unwrap();
    let source = sg_gguf::DirectSource::new(&path, &gguf);

    // tmpfs and friends don't support O_DIRECT; the fixture lives under
    // target/ on a real filesystem, but skip gracefully if not.
    let mut probe = vec![0u8; source.data_len()];
    if let Err(e) = source.load(&mut probe) {
        assert_eq!(
            e.raw_os_error(),
            Some(libc::EINVAL),
            "unexpected O_DIRECT failure: {e}"
        );
        eprintln!("skipping: filesystem does not support O_DIRECT ({e})");
        return;
    }
    assert_eq!(probe, gguf.data_section());

    for matched in [true, false] {
        let got = load_with_phase(&source, gguf.data_offset(), matched);
        assert_eq!(got, gguf.data_section(), "phase matched={matched}");
    }
}
