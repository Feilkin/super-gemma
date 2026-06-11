//! Parser tests against synthetic mini-GGUF fixtures (plan 01: every KV type,
//! alignment edge cases, malformed/truncated input must error, never panic).

mod common;

use common::{Builder, w_u32, w_u64};
use sg_gguf::{GgmlType, Gguf, GgufError, MetaArray, MetaValue};

/// A reasonably rich, valid file exercising most of the format.
fn rich() -> Vec<u8> {
    Builder::new()
        .kv_str("general.architecture", "gemma4")
        .kv_u8("t.u8", 7)
        .kv_i8("t.i8", -7)
        .kv_u16("t.u16", 1000)
        .kv_i16("t.i16", -1000)
        .kv_u32("t.u32", 70_000)
        .kv_i32("t.i32", -70_000)
        .kv_f32("t.f32", 1.5)
        .kv_bool("t.bool", true)
        .kv_u64("t.u64", 1 << 40)
        .kv_i64("t.i64", -(1 << 40))
        .kv_f64("t.f64", 2.25)
        .kv_u8_array("a.u8", &[1, 2, 3])
        .kv_u16_array("a.u16", &[1, 2])
        .kv_i32_array("a.i32", &[-1, 0, 1])
        .kv_f32_array("a.f32", &[0.5, -0.5])
        .kv_bool_array("a.bool", &[true, false])
        .kv_str_array("a.str", &["<pad>", "<eos>", "▁hello"])
        .kv_i64_array("a.i64", &[i64::MIN, i64::MAX])
        .kv_f64_array("a.f64", &[1.0])
        .kv_nested_u32_array("a.nested", &[&[1, 2], &[], &[3]])
        .tensor("norm.weight", &[4], 0, &[0u8; 16]) // F32, 4 elems
        .tensor("blk.0.attn_q.weight", &[64, 2], 2, &[0u8; 72]) // Q4_0, 128 elems = 4 blocks
        .build()
}

#[test]
fn parses_rich_file() {
    let bytes = rich();
    let g = Gguf::parse(&bytes).unwrap();
    assert_eq!(g.version, 3);
    assert_eq!(g.alignment, 32);
    assert_eq!(
        g.metadata.require_str("general.architecture").unwrap(),
        "gemma4"
    );
    assert_eq!(g.metadata.require_uint("t.u8").unwrap(), 7);
    assert_eq!(g.metadata.require_uint("t.u64").unwrap(), 1 << 40);
    assert_eq!(g.metadata.get_int("t.i32").unwrap(), Some(-70_000));
    assert_eq!(g.metadata.require_f32("t.f32").unwrap(), 1.5);
    assert_eq!(g.metadata.get_bool("t.bool").unwrap(), Some(true));
    assert_eq!(
        g.metadata.get("t.f64"),
        Some(&MetaValue::F64(2.25)),
        "f64 round-trip"
    );
    assert_eq!(
        g.metadata.require_str_array("a.str").unwrap(),
        ["<pad>", "<eos>", "▁hello"]
    );
    assert_eq!(g.metadata.require_f32_array("a.f32").unwrap(), [0.5, -0.5]);
    assert_eq!(g.metadata.require_i32_array("a.i32").unwrap(), [-1, 0, 1]);
    assert_eq!(
        g.metadata.get("a.nested").unwrap().as_array().unwrap(),
        &MetaArray::Nested(vec![
            MetaArray::U32(vec![1, 2]),
            MetaArray::U32(vec![]),
            MetaArray::U32(vec![3]),
        ])
    );
    assert_eq!(g.metadata.len(), 21);
}

#[test]
fn tensor_table_and_data_views() {
    let q4_data: Vec<u8> = (0..72u32).map(|i| i as u8).collect();
    let f32_data: Vec<u8> = 1.0f32
        .to_le_bytes()
        .iter()
        .chain(2.0f32.to_le_bytes().iter())
        .copied()
        .collect();
    let bytes = Builder::new()
        .tensor("a", &[2], 0, &f32_data)
        .tensor("b", &[32, 4], 2, &q4_data)
        .build();
    let g = Gguf::parse(&bytes).unwrap();

    assert_eq!(g.tensors().len(), 2);
    let a = g.tensor("a").unwrap();
    assert_eq!((a.dtype, a.offset, a.byte_len), (GgmlType::F32, 0, 8));
    assert_eq!(g.data_of(a), &f32_data[..]);

    let b = g.tensor("b").unwrap();
    assert_eq!(b.dtype, GgmlType::Q4_0);
    assert_eq!(b.dims, [32, 4]);
    assert_eq!(b.elem_count(), 128);
    assert_eq!(b.offset % g.alignment as u64, 0);
    assert_eq!(g.tensor_data("b").unwrap(), &q4_data[..]);
    // Q4_0 tensor data feeds the block reference type.
    let blocks = sg_gguf::q4_0::blocks_from_bytes(g.tensor_data("b").unwrap()).unwrap();
    assert_eq!(blocks.len(), 4);

    assert!(g.tensor("missing").is_none());
    assert!(g.tensor_data("missing").is_none());
}

#[test]
fn respects_custom_alignment() {
    let bytes = Builder::new()
        .alignment(64)
        .tensor("a", &[1], 0, &[1, 2, 3, 4])
        .tensor("b", &[1], 0, &[5, 6, 7, 8])
        .build();
    let g = Gguf::parse(&bytes).unwrap();
    assert_eq!(g.alignment, 64);
    assert_eq!(g.tensor("b").unwrap().offset, 64);
    assert_eq!(g.tensor_data("a").unwrap(), [1, 2, 3, 4]);
    assert_eq!(g.tensor_data("b").unwrap(), [5, 6, 7, 8]);
}

#[test]
fn parses_empty_file() {
    let bytes = Builder::new().build();
    let g = Gguf::parse(&bytes).unwrap();
    assert!(g.metadata.is_empty());
    assert!(g.tensors().is_empty());
}

#[test]
fn rejects_bad_magic_and_version() {
    let mut bytes = rich();
    bytes[0] = b'X';
    assert!(matches!(Gguf::parse(&bytes), Err(GgufError::BadMagic(_))));

    let bytes = Builder::new().version(2).build();
    assert!(matches!(
        Gguf::parse(&bytes),
        Err(GgufError::UnsupportedVersion(2))
    ));
}

#[test]
fn rejects_unknown_kv_type() {
    let bytes = Builder::new().kv_raw("weird", 13, &[0; 8]).build();
    let err = Gguf::parse(&bytes).unwrap_err();
    assert!(
        matches!(&err, GgufError::UnknownValueType { key, raw: 13 } if key == "weird"),
        "{err}"
    );
}

#[test]
fn rejects_unknown_tensor_type() {
    // Q4_1 (id 3) exists in ggml but not in this model.
    let bytes = Builder::new().tensor("w", &[32], 3, &[0; 20]).build();
    let err = Gguf::parse(&bytes).unwrap_err();
    assert!(
        matches!(&err, GgufError::UnsupportedTensorType { name, raw: 3 } if name == "w"),
        "{err}"
    );
}

#[test]
fn rejects_invalid_bool() {
    let bytes = Builder::new().kv_raw("b", 7, &[2]).build();
    assert!(matches!(
        Gguf::parse(&bytes),
        Err(GgufError::InvalidBool { value: 2, .. })
    ));
}

#[test]
fn rejects_duplicates() {
    let bytes = Builder::new().kv_u32("k", 1).kv_u32("k", 2).build();
    assert!(matches!(Gguf::parse(&bytes), Err(GgufError::DuplicateKey(k)) if k == "k"));

    let bytes = Builder::new()
        .tensor("t", &[1], 0, &[0; 4])
        .tensor("t", &[1], 0, &[0; 4])
        .build();
    assert!(matches!(Gguf::parse(&bytes), Err(GgufError::DuplicateTensor(t)) if t == "t"));
}

#[test]
fn rejects_bad_alignment() {
    for a in [0u32, 3] {
        let bytes = Builder::new().kv_u32("general.alignment", a).build();
        assert!(
            matches!(Gguf::parse(&bytes), Err(GgufError::BadAlignment(got)) if got == a as u64),
            "alignment {a}"
        );
    }
    // Wrong type for general.alignment is a MetaError, not a silent default.
    let bytes = Builder::new().kv_str("general.alignment", "32").build();
    assert!(matches!(Gguf::parse(&bytes), Err(GgufError::Meta(_))));
}

#[test]
fn rejects_partial_q4_0_block() {
    // 16 elements is half a Q4_0 block.
    let bytes = Builder::new().tensor("w", &[16], 2, &[0; 9]).build();
    assert!(matches!(
        Gguf::parse(&bytes),
        Err(GgufError::BadBlockCount { elems: 16, .. })
    ));
}

#[test]
fn rejects_too_many_dims_and_dim_overflow() {
    let bytes = Builder::new()
        .tensor("w", &[1, 1, 1, 1, 1], 0, &[0; 4])
        .build();
    assert!(matches!(
        Gguf::parse(&bytes),
        Err(GgufError::TooManyDims { n_dims: 5, .. })
    ));

    let bytes = Builder::new()
        .tensor("w", &[u64::MAX, u64::MAX], 0, &[])
        .build();
    assert!(matches!(
        Gguf::parse(&bytes),
        Err(GgufError::DimOverflow { .. })
    ));
}

#[test]
fn rejects_misaligned_and_out_of_bounds_tensors() {
    let bytes = Builder::new()
        .tensor_raw("w", &[4], 0, 8, &[0; 16]) // offset 8 not a multiple of 32
        .build();
    assert!(matches!(
        Gguf::parse(&bytes),
        Err(GgufError::MisalignedTensor { offset: 8, .. })
    ));

    // Declared F32 [256] = 1 KiB, but only 16 bytes of data present.
    let bytes = Builder::new().tensor("w", &[256], 0, &[0; 16]).build();
    assert!(matches!(
        Gguf::parse(&bytes),
        Err(GgufError::TensorOutOfBounds { .. })
    ));

    // Offset itself beyond the data section.
    let bytes = Builder::new()
        .tensor_raw("w", &[1], 0, 1 << 16, &[])
        .build();
    let truncated = &bytes[..bytes.len() - (1 << 16)];
    assert!(matches!(
        Gguf::parse(truncated),
        Err(GgufError::TensorOutOfBounds { .. })
    ));
}

#[test]
fn huge_declared_lengths_error_without_allocating() {
    // String length u64::MAX.
    let mut bytes = Vec::new();
    w_u32(&mut bytes, common::MAGIC);
    w_u32(&mut bytes, 3);
    w_u64(&mut bytes, 0); // tensors
    w_u64(&mut bytes, 1); // kvs
    w_u64(&mut bytes, u64::MAX); // key length
    assert!(matches!(
        Gguf::parse(&bytes),
        Err(GgufError::Truncated { .. })
    ));

    // f64 array with count that overflows count*8.
    let mut payload = Vec::new();
    w_u32(&mut payload, 12);
    w_u64(&mut payload, u64::MAX / 2);
    let bytes = Builder::new().kv_raw("a", 9, &payload).build();
    assert!(matches!(
        Gguf::parse(&bytes),
        Err(GgufError::Truncated { .. })
    ));

    // Tensor count claims more entries than the file holds.
    let mut bytes = Vec::new();
    w_u32(&mut bytes, common::MAGIC);
    w_u32(&mut bytes, 3);
    w_u64(&mut bytes, u64::MAX); // tensors
    w_u64(&mut bytes, 0); // kvs
    assert!(matches!(
        Gguf::parse(&bytes),
        Err(GgufError::Truncated { .. })
    ));
}

#[test]
fn rejects_deep_array_nesting() {
    // 32 levels of array-of-array headers, deeper than MAX_ARRAY_DEPTH.
    let mut payload = Vec::new();
    for _ in 0..32 {
        w_u32(&mut payload, 9); // elem type: array
        w_u64(&mut payload, 1); // one element
    }
    w_u32(&mut payload, 4); // innermost: u32
    w_u64(&mut payload, 0);
    let bytes = Builder::new().kv_raw("deep", 9, &payload).build();
    assert!(matches!(
        Gguf::parse(&bytes),
        Err(GgufError::ArrayTooDeep(_))
    ));
}

#[test]
fn rejects_invalid_utf8_strings() {
    let mut payload = Vec::new();
    w_u64(&mut payload, 2);
    payload.extend_from_slice(&[0xFF, 0xFE]);
    let bytes = Builder::new().kv_raw("s", 8, &payload).build();
    assert!(matches!(
        Gguf::parse(&bytes),
        Err(GgufError::InvalidUtf8 { .. })
    ));
}

/// Plan 01: the parser must never panic, only error. Every truncation point
/// and every single-byte corruption of a rich valid file must yield
/// `Ok`/`Err`, not a panic. (cargo-fuzz does this open-endedly; this is the
/// deterministic in-tree version.)
#[test]
fn truncation_and_corruption_never_panic() {
    let bytes = rich();
    for len in 0..bytes.len() {
        let _ = Gguf::parse(&bytes[..len]);
    }
    assert!(Gguf::parse(&bytes).is_ok());

    let mut mutated = bytes.clone();
    for (i, &orig) in bytes.iter().enumerate() {
        mutated[i] = orig ^ 0xA5;
        let _ = Gguf::parse(&mutated);
        mutated[i] = orig;
    }
}

#[test]
fn metadata_preserves_file_order() {
    let bytes = Builder::new()
        .kv_u32("z.last", 1)
        .kv_u32("a.first", 2)
        .build();
    let g = Gguf::parse(&bytes).unwrap();
    let keys: Vec<&str> = g.metadata.iter().map(|(k, _)| k).collect();
    assert_eq!(keys, ["z.last", "a.first"]);
}
