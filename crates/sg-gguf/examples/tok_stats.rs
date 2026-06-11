//! Scratch inspector for the GGUF tokenizer payload (algorithm forensics).

use std::collections::HashMap;

use sg_gguf::GgufFile;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: tok_stats <file.gguf>");
    let file = GgufFile::open(&path).expect("open");
    let g = file.parse().expect("parse");
    let m = &g.metadata;

    let tokens = m.require_str_array("tokenizer.ggml.tokens").unwrap();
    let scores = m.require_f32_array("tokenizer.ggml.scores").unwrap();
    let types = m.require_i32_array("tokenizer.ggml.token_type").unwrap();
    let merges = m.require_str_array("tokenizer.ggml.merges").unwrap();

    let mut type_hist: HashMap<i32, usize> = HashMap::new();
    for &t in types {
        *type_hist.entry(t).or_default() += 1;
    }
    println!("token_type histogram: {type_hist:?}");

    let mut score_hist: HashMap<String, usize> = HashMap::new();
    for &s in scores {
        *score_hist.entry(format!("{s}")).or_default() += 1;
    }
    let mut top: Vec<_> = score_hist.iter().collect();
    top.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
    println!(
        "distinct scores: {}; top: {:?}",
        score_hist.len(),
        &top[..top.len().min(8)]
    );

    // Byte tokens?
    let byte_tokens: Vec<usize> = tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| t.starts_with("<0x") && t.ends_with('>'))
        .map(|(i, _)| i)
        .take(5)
        .collect();
    println!("byte-token sample ids: {byte_tokens:?}");

    // Tokens of interest.
    for needle in [
        "<start_of_turn>",
        "<end_of_turn>",
        "<bos>",
        "<eos>",
        "\n",
        "▁hello",
        "hello",
        " hello",
        "<tool_call>",
        "<unused99>",
    ] {
        let id = tokens.iter().position(|t| t == needle);
        println!("token {needle:?} -> {id:?}");
    }

    // Sample tokens around interesting ranges.
    for range in [0..16usize, 100..120, 255990..256010] {
        for i in range {
            if i < tokens.len() {
                println!(
                    "id {i}: {:?} score {} type {}",
                    tokens[i], scores[i], types[i]
                );
            }
        }
    }

    println!("merges: {} total; first 10:", merges.len());
    for mge in merges.iter().take(10) {
        println!("  {mge:?}");
    }
    println!("last 3:");
    for mge in merges.iter().rev().take(3) {
        println!("  {mge:?}");
    }
}
