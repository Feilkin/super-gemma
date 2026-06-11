//! `gguf-dump`-style report: metadata (large arrays summarized) and the full
//! tensor table. The checked-in report for the real model lives in
//! `docs/reference/` (plan 01).
//!
//! Usage: `cargo run -p sg-gguf --example dump -- <file.gguf>`

use sg_gguf::{GgufFile, MetaArray, MetaValue};

fn main() {
    let path = std::env::args().nth(1).expect("usage: dump <file.gguf>");
    let file = GgufFile::open(&path).expect("open/mmap");
    let g = file.parse().expect("parse");

    println!("# GGUF dump: {path}");
    println!(
        "version {} | alignment {} | {} metadata keys | {} tensors",
        g.version,
        g.alignment,
        g.metadata.len(),
        g.tensors().len()
    );

    println!("\n## Metadata");
    for (key, value) in g.metadata.iter() {
        println!("{key} = {}", fmt_value(value));
    }

    println!("\n## Tensor table");
    println!(
        "{:<40} {:>6} {:>24} {:>16} {:>14}",
        "name", "dtype", "dims (ggml order)", "offset", "bytes"
    );
    for t in g.tensors() {
        println!(
            "{:<40} {:>6} {:>24} {:>16} {:>14}",
            t.name,
            t.dtype.to_string(),
            format!("{:?}", t.dims),
            t.offset,
            t.byte_len
        );
    }
}

fn fmt_value(v: &MetaValue) -> String {
    match v {
        MetaValue::String(s) if s.len() > 80 => {
            let mut end = 77;
            while !s.is_char_boundary(end) {
                end -= 1;
            }
            format!("{:?}… ({} bytes)", &s[..end], s.len())
        }
        MetaValue::Array(a) => fmt_array(a),
        other => format!("{other:?}"),
    }
}

fn fmt_array(a: &MetaArray) -> String {
    let n = a.len();
    if n <= 8 {
        return format!("{a:?}");
    }
    let head: Vec<String> = match a {
        MetaArray::String(v) => v.iter().take(8).map(|s| format!("{s:?}")).collect(),
        MetaArray::F32(v) => v.iter().take(8).map(|x| x.to_string()).collect(),
        MetaArray::I32(v) => v.iter().take(8).map(|x| x.to_string()).collect(),
        MetaArray::U32(v) => v.iter().take(8).map(|x| x.to_string()).collect(),
        _ => vec![format!("{}", a.type_name())],
    };
    format!("[{}, …] ({n} × {})", head.join(", "), a.type_name())
}
