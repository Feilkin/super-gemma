//! GGUF v3 binary parser.
//!
//! A bounds-checked cursor walks the mmapped file. Parsing never panics on
//! malformed input — every failure path is a typed [`GgufError`] — and never
//! allocates based on untrusted lengths before checking them against the
//! bytes actually present. Tensor data is not read here; only its offsets are
//! validated against the data section, and [`Gguf::tensor_data`] hands out
//! zero-copy slices afterwards.

use std::collections::HashMap;

use crate::meta::{MetaArray, MetaError, MetaValue, Metadata};
use crate::tensor::{GgmlType, TensorInfo};

/// `GGUF` as a little-endian u32.
pub const GGUF_MAGIC: u32 = 0x4655_4747;
/// The only supported version (what current llama.cpp exports write).
pub const GGUF_VERSION: u32 = 3;
/// Data-section alignment when `general.alignment` is absent.
pub const DEFAULT_ALIGNMENT: usize = 32;

/// ggml caps tensors at 4 dimensions.
const MAX_DIMS: u32 = 4;
/// Real files nest arrays at most ~2 deep; the cap only guards the recursive
/// parser's stack against crafted input.
const MAX_ARRAY_DEPTH: u32 = 8;

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("not a GGUF file (magic {0:#010x})")]
    BadMagic(u32),
    #[error("unsupported GGUF version {0} (expected {GGUF_VERSION})")]
    UnsupportedVersion(u32),
    #[error(
        "file truncated reading {what} at offset {offset}: need {needed} bytes, {available} available"
    )]
    Truncated {
        what: &'static str,
        offset: usize,
        needed: u64,
        available: usize,
    },
    #[error("invalid UTF-8 in {what} at offset {offset}")]
    InvalidUtf8 { what: &'static str, offset: usize },
    #[error("unknown metadata value type {raw} for key `{key}`")]
    UnknownValueType { key: String, raw: u32 },
    #[error("invalid bool byte {value:#04x} in metadata key `{key}`")]
    InvalidBool { key: String, value: u8 },
    #[error("metadata array under key `{0}` nests deeper than {MAX_ARRAY_DEPTH}")]
    ArrayTooDeep(String),
    #[error("duplicate metadata key `{0}`")]
    DuplicateKey(String),
    #[error("invalid general.alignment {0}: must be a nonzero power of two")]
    BadAlignment(u64),
    #[error(
        "tensor `{name}`: unsupported ggml type id {raw} \
         (this model ships only F32, F16, Q4_0, Q8_0, Q6_K)"
    )]
    UnsupportedTensorType { name: String, raw: u32 },
    #[error("tensor `{name}`: {n_dims} dimensions, ggml supports at most {MAX_DIMS}")]
    TooManyDims { name: String, n_dims: u32 },
    #[error("tensor `{name}`: dimension product overflows u64")]
    DimOverflow { name: String },
    #[error(
        "tensor `{name}`: element count {elems} is not a valid {dtype} tensor size \
         (block of {block} elements)"
    )]
    BadBlockCount {
        name: String,
        elems: u64,
        dtype: GgmlType,
        block: u64,
    },
    #[error("duplicate tensor name `{0}`")]
    DuplicateTensor(String),
    #[error("tensor `{name}`: offset {offset} is not a multiple of the alignment {alignment}")]
    MisalignedTensor {
        name: String,
        offset: u64,
        alignment: usize,
    },
    #[error(
        "tensor `{name}`: data range {offset}..{offset}+{byte_len} exceeds the \
         {data_len}-byte data section"
    )]
    TensorOutOfBounds {
        name: String,
        offset: u64,
        byte_len: u64,
        data_len: usize,
    },
    #[error(transparent)]
    Meta(#[from] MetaError),
}

/// A parsed GGUF file: owned metadata and tensor table, zero-copy tensor data
/// borrowed from the underlying bytes (usually an mmap via
/// [`GgufFile`](crate::GgufFile)).
#[derive(Debug)]
pub struct Gguf<'a> {
    pub version: u32,
    /// Data-section alignment (`general.alignment`, default 32).
    pub alignment: usize,
    pub metadata: Metadata,
    tensors: Vec<TensorInfo>,
    index: HashMap<String, usize>,
    /// Byte offset of the data section in the file.
    data_start: usize,
    data: &'a [u8],
}

impl<'a> Gguf<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, GgufError> {
        let mut cur = Cur { buf: bytes, pos: 0 };

        let magic = cur.read_u32("magic")?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic(magic));
        }
        let version = cur.read_u32("version")?;
        if version != GGUF_VERSION {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let tensor_count = cur.read_u64("tensor count")?;
        let kv_count = cur.read_u64("metadata KV count")?;

        let mut metadata = Metadata::default();
        for _ in 0..kv_count {
            let key = cur.read_string("metadata key")?.to_owned();
            let raw_ty = cur.read_u32("metadata value type")?;
            let value = read_value(&mut cur, raw_ty, &key)?;
            if !metadata.insert(key.clone(), value) {
                return Err(GgufError::DuplicateKey(key));
            }
        }

        let alignment = match metadata.get("general.alignment") {
            None => DEFAULT_ALIGNMENT,
            Some(v) => {
                let a = v.as_uint().ok_or_else(|| MetaError::WrongType {
                    key: "general.alignment".into(),
                    expected: "unsigned integer",
                    found: v.type_name(),
                })?;
                if a == 0 || !a.is_power_of_two() {
                    return Err(GgufError::BadAlignment(a));
                }
                usize::try_from(a).map_err(|_| GgufError::BadAlignment(a))?
            }
        };

        let mut tensors = Vec::new();
        let mut index = HashMap::new();
        for _ in 0..tensor_count {
            let info = read_tensor_info(&mut cur)?;
            if index.contains_key(&info.name) {
                return Err(GgufError::DuplicateTensor(info.name));
            }
            index.insert(info.name.clone(), tensors.len());
            tensors.push(info);
        }

        // The data section starts at the next alignment boundary after the
        // tensor table and runs to EOF.
        let data_start = cur
            .pos
            .checked_add(alignment - 1)
            .map(|p| p / alignment * alignment)
            .ok_or(GgufError::BadAlignment(alignment as u64))?;
        if data_start > bytes.len() && !tensors.is_empty() {
            return Err(GgufError::Truncated {
                what: "tensor data section",
                offset: cur.pos,
                needed: (data_start - bytes.len()) as u64,
                available: 0,
            });
        }
        let data = &bytes[data_start.min(bytes.len())..];

        for t in &tensors {
            if !t.offset.is_multiple_of(alignment as u64) {
                return Err(GgufError::MisalignedTensor {
                    name: t.name.clone(),
                    offset: t.offset,
                    alignment,
                });
            }
            let oob = GgufError::TensorOutOfBounds {
                name: t.name.clone(),
                offset: t.offset,
                byte_len: t.byte_len,
                data_len: data.len(),
            };
            match t.offset.checked_add(t.byte_len) {
                Some(end) if end <= data.len() as u64 => {}
                _ => return Err(oob),
            }
        }

        Ok(Self {
            version,
            alignment,
            metadata,
            tensors,
            index,
            data_start: data_start.min(bytes.len()),
            data,
        })
    }

    /// Tensor table in file order.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// The whole tensor-data section (everything tensor offsets are relative
    /// to).
    pub fn data_section(&self) -> &'a [u8] {
        self.data
    }

    /// Byte offset of the data section within the file.
    pub fn data_offset(&self) -> usize {
        self.data_start
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    /// Zero-copy view of a tensor's serialized data, by name.
    pub fn tensor_data(&self, name: &str) -> Option<&'a [u8]> {
        self.tensor(name).map(|t| self.data_of(t))
    }

    /// Zero-copy view of a tensor's serialized data.
    ///
    /// `info` must come from this file's [`tensors`](Self::tensors) table
    /// (whose ranges were validated at parse time); a foreign `TensorInfo`
    /// may panic.
    pub fn data_of(&self, info: &TensorInfo) -> &'a [u8] {
        &self.data[info.offset as usize..][..info.byte_len as usize]
    }
}

fn read_tensor_info(cur: &mut Cur<'_>) -> Result<TensorInfo, GgufError> {
    let name = cur.read_string("tensor name")?.to_owned();
    let n_dims = cur.read_u32("tensor dimension count")?;
    if n_dims > MAX_DIMS {
        return Err(GgufError::TooManyDims { name, n_dims });
    }
    let mut dims = Vec::with_capacity(n_dims as usize);
    for _ in 0..n_dims {
        dims.push(cur.read_u64("tensor dimension")?);
    }
    let raw_ty = cur.read_u32("tensor type")?;
    let offset = cur.read_u64("tensor offset")?;

    let dtype = GgmlType::from_raw(raw_ty).ok_or_else(|| GgufError::UnsupportedTensorType {
        name: name.clone(),
        raw: raw_ty,
    })?;
    let elems = dims
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| GgufError::DimOverflow { name: name.clone() })?;
    let byte_len = dtype
        .byte_len(elems)
        .ok_or_else(|| GgufError::BadBlockCount {
            name: name.clone(),
            elems,
            dtype,
            block: dtype.block_elems(),
        })?;

    Ok(TensorInfo {
        name,
        dims,
        dtype,
        offset,
        byte_len,
    })
}

fn read_value(cur: &mut Cur<'_>, raw_ty: u32, key: &str) -> Result<MetaValue, GgufError> {
    let what = "metadata value";
    Ok(match raw_ty {
        0 => MetaValue::U8(cur.read_u8(what)?),
        1 => MetaValue::I8(cur.read_i8(what)?),
        2 => MetaValue::U16(cur.read_u16(what)?),
        3 => MetaValue::I16(cur.read_i16(what)?),
        4 => MetaValue::U32(cur.read_u32(what)?),
        5 => MetaValue::I32(cur.read_i32(what)?),
        6 => MetaValue::F32(cur.read_f32(what)?),
        7 => MetaValue::Bool(read_bool(cur, key)?),
        8 => MetaValue::String(cur.read_string(what)?.to_owned()),
        9 => MetaValue::Array(read_array(cur, key, 0)?),
        10 => MetaValue::U64(cur.read_u64(what)?),
        11 => MetaValue::I64(cur.read_i64(what)?),
        12 => MetaValue::F64(cur.read_f64(what)?),
        raw => {
            return Err(GgufError::UnknownValueType {
                key: key.into(),
                raw,
            });
        }
    })
}

fn read_bool(cur: &mut Cur<'_>, key: &str) -> Result<bool, GgufError> {
    match cur.read_u8("bool value")? {
        0 => Ok(false),
        1 => Ok(true),
        value => Err(GgufError::InvalidBool {
            key: key.into(),
            value,
        }),
    }
}

fn read_array(cur: &mut Cur<'_>, key: &str, depth: u32) -> Result<MetaArray, GgufError> {
    if depth >= MAX_ARRAY_DEPTH {
        return Err(GgufError::ArrayTooDeep(key.into()));
    }
    let elem_ty = cur.read_u32("array element type")?;
    let count = cur.read_u64("array length")?;
    let what = "array elements";
    Ok(match elem_ty {
        0 => MetaArray::U8(cur.take_n(count, 1, what)?.to_vec()),
        1 => MetaArray::I8(vec_from_le(cur.take_n(count, 1, what)?, i8::from_le_bytes)),
        2 => MetaArray::U16(vec_from_le(cur.take_n(count, 2, what)?, u16::from_le_bytes)),
        3 => MetaArray::I16(vec_from_le(cur.take_n(count, 2, what)?, i16::from_le_bytes)),
        4 => MetaArray::U32(vec_from_le(cur.take_n(count, 4, what)?, u32::from_le_bytes)),
        5 => MetaArray::I32(vec_from_le(cur.take_n(count, 4, what)?, i32::from_le_bytes)),
        6 => MetaArray::F32(vec_from_le(cur.take_n(count, 4, what)?, f32::from_le_bytes)),
        7 => MetaArray::Bool(
            cur.take_n(count, 1, what)?
                .iter()
                .map(|&b| match b {
                    0 => Ok(false),
                    1 => Ok(true),
                    value => Err(GgufError::InvalidBool {
                        key: key.into(),
                        value,
                    }),
                })
                .collect::<Result<_, _>>()?,
        ),
        8 => {
            // No pre-reserve: `count` is untrusted, but each element consumes
            // at least 8 bytes of input, so growth is bounded by the file.
            let mut v = Vec::new();
            for _ in 0..count {
                v.push(cur.read_string("array string")?.to_owned());
            }
            MetaArray::String(v)
        }
        9 => {
            let mut v = Vec::new();
            for _ in 0..count {
                v.push(read_array(cur, key, depth + 1)?);
            }
            MetaArray::Nested(v)
        }
        10 => MetaArray::U64(vec_from_le(cur.take_n(count, 8, what)?, u64::from_le_bytes)),
        11 => MetaArray::I64(vec_from_le(cur.take_n(count, 8, what)?, i64::from_le_bytes)),
        12 => MetaArray::F64(vec_from_le(cur.take_n(count, 8, what)?, f64::from_le_bytes)),
        raw => {
            return Err(GgufError::UnknownValueType {
                key: key.into(),
                raw,
            });
        }
    })
}

fn vec_from_le<const N: usize, T>(bytes: &[u8], from_le: fn([u8; N]) -> T) -> Vec<T> {
    bytes
        .chunks_exact(N)
        .map(|c| from_le(c.try_into().expect("chunks_exact yields N-byte chunks")))
        .collect()
}

/// Bounds-checked little-endian cursor.
struct Cur<'a> {
    buf: &'a [u8],
    pos: usize,
}

macro_rules! read_prim {
    ($name:ident, $ty:ty) => {
        fn $name(&mut self, what: &'static str) -> Result<$ty, GgufError> {
            let bytes = self.take(size_of::<$ty>() as u64, what)?;
            Ok(<$ty>::from_le_bytes(
                bytes
                    .try_into()
                    .expect("take returned the requested length"),
            ))
        }
    };
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: u64, what: &'static str) -> Result<&'a [u8], GgufError> {
        let available = self.buf.len() - self.pos;
        if n > available as u64 {
            return Err(GgufError::Truncated {
                what,
                offset: self.pos,
                needed: n,
                available,
            });
        }
        let n = n as usize;
        let slice = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    /// `take(count * elem_size)` with overflow folded into the bounds check.
    fn take_n(
        &mut self,
        count: u64,
        elem_size: u64,
        what: &'static str,
    ) -> Result<&'a [u8], GgufError> {
        let n = count.checked_mul(elem_size).ok_or(GgufError::Truncated {
            what,
            offset: self.pos,
            needed: u64::MAX,
            available: self.buf.len() - self.pos,
        })?;
        self.take(n, what)
    }

    read_prim!(read_u8, u8);
    read_prim!(read_i8, i8);
    read_prim!(read_u16, u16);
    read_prim!(read_i16, i16);
    read_prim!(read_u32, u32);
    read_prim!(read_i32, i32);
    read_prim!(read_u64, u64);
    read_prim!(read_i64, i64);
    read_prim!(read_f32, f32);
    read_prim!(read_f64, f64);

    fn read_string(&mut self, what: &'static str) -> Result<&'a str, GgufError> {
        let offset = self.pos;
        let len = self.read_u64(what)?;
        let bytes = self.take(len, what)?;
        str::from_utf8(bytes).map_err(|_| GgufError::InvalidUtf8 { what, offset })
    }
}
