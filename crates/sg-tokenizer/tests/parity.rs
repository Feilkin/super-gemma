//! Golden-corpus parity vs HF `tokenizers` (plan 01 / M1 exit criterion:
//! 100 % id-sequence match). Vocab comes from the real model GGUF, which also
//! cross-validates that the GGUF tokenizer payload equals the HF
//! `tokenizer.json` it was exported from. Self-skips when the model file is
//! absent (Tier 1 CI); override the path with `SG_MODEL_GGUF`.

use std::path::PathBuf;

use sg_tokenizer::{SpecialTokens, Tokenizer};

fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SG_MODEL_GGUF") {
        return Some(p.into());
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/gemma-4-31B_q4_0-it.gguf");
    p.exists().then_some(p)
}

#[test]
fn encode_parity_and_round_trip() {
    let Some(path) = model_path() else {
        eprintln!("skipping: model GGUF not present (set SG_MODEL_GGUF)");
        return;
    };
    let file = sg_gguf::GgufFile::open(&path).expect("open model file");
    let gguf = file.parse().unwrap_or_else(|e| panic!("parse: {e}"));
    let tok = Tokenizer::from_metadata(&gguf.metadata).unwrap_or_else(|e| panic!("vocab: {e}"));

    let fixture = include_str!("fixtures/encode_parity.jsonl");
    let mut lines = fixture.lines();
    let header = lines.next().expect("fixture header");
    eprintln!("fixture: {header}");

    let mut total = 0u32;
    let mut mismatches = 0u32;
    for line in lines {
        let case: serde_json::Value = serde_json::from_str(line).expect("fixture line");
        let text = case["text"].as_str().expect("text");
        let want: Vec<u32> = case["ids"]
            .as_array()
            .expect("ids")
            .iter()
            .map(|v| u32::try_from(v.as_u64().expect("id")).expect("id range"))
            .collect();

        total += 1;
        let got = tok.encode(text, SpecialTokens::Match);
        if got != want {
            mismatches += 1;
            if mismatches <= 5 {
                eprintln!("MISMATCH {text:?}\n  want {want:?}\n  got  {got:?}");
            }
        }

        // Round-trip: decode(encode(s)) == s, except texts containing a
        // literal ▁ (which legitimately decodes to a space).
        if !text.contains('\u{2581}') {
            let rt = tok.decode(&got).unwrap_or_else(|e| panic!("decode: {e}"));
            assert_eq!(rt, text, "round-trip mismatch");
        }
    }
    assert_eq!(mismatches, 0, "{mismatches}/{total} parity mismatches");
    assert!(total > 12_000, "suspiciously small corpus: {total}");
}
