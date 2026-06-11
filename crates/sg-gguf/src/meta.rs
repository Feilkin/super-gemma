//! Typed storage and accessors for GGUF metadata key/value pairs.
//!
//! Metadata is parsed eagerly into owned values: unlike tensor data (which
//! stays zero-copy in the mmap), the metadata section is small — a few MB of
//! tokenizer vocab at worst — and owning it keeps lifetimes simple for the
//! consumers (`ModelDesc`, the tokenizer).

use std::collections::HashMap;

/// One GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(MetaArray),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetaValue {
    /// Human-readable type name for error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::U8(_) => "u8",
            Self::I8(_) => "i8",
            Self::U16(_) => "u16",
            Self::I16(_) => "i16",
            Self::U32(_) => "u32",
            Self::I32(_) => "i32",
            Self::F32(_) => "f32",
            Self::Bool(_) => "bool",
            Self::String(_) => "string",
            Self::Array(a) => a.type_name(),
            Self::U64(_) => "u64",
            Self::I64(_) => "i64",
            Self::F64(_) => "f64",
        }
    }

    /// Lenient unsigned-integer view: any unsigned integer type widens to
    /// u64. GGUF writers are inconsistent about integer widths.
    pub fn as_uint(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(v.into()),
            Self::U16(v) => Some(v.into()),
            Self::U32(v) => Some(v.into()),
            Self::U64(v) => Some(v),
            _ => None,
        }
    }

    /// Lenient signed-integer view: any integer type that fits in i64.
    pub fn as_int(&self) -> Option<i64> {
        match *self {
            Self::I8(v) => Some(v.into()),
            Self::I16(v) => Some(v.into()),
            Self::I32(v) => Some(v.into()),
            Self::I64(v) => Some(v),
            _ => self.as_uint().and_then(|v| i64::try_from(v).ok()),
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Self::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&MetaArray> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }
}

/// A homogeneous GGUF metadata array, stored as a typed vector so the large
/// arrays (262k vocab strings, scores) don't pay a per-element enum tag.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaArray {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
    /// Array of arrays (legal per spec, rare in practice).
    Nested(Vec<MetaArray>),
}

impl MetaArray {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::U8(_) => "array of u8",
            Self::I8(_) => "array of i8",
            Self::U16(_) => "array of u16",
            Self::I16(_) => "array of i16",
            Self::U32(_) => "array of u32",
            Self::I32(_) => "array of i32",
            Self::F32(_) => "array of f32",
            Self::Bool(_) => "array of bool",
            Self::String(_) => "array of string",
            Self::U64(_) => "array of u64",
            Self::I64(_) => "array of i64",
            Self::F64(_) => "array of f64",
            Self::Nested(_) => "array of array",
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::String(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::Nested(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Error from a typed metadata accessor.
#[derive(Debug, thiserror::Error)]
pub enum MetaError {
    #[error("missing metadata key `{0}`")]
    Missing(String),
    #[error("metadata key `{key}`: expected {expected}, found {found}")]
    WrongType {
        key: String,
        expected: &'static str,
        found: &'static str,
    },
}

/// All metadata of a GGUF file, preserving file order for reporting.
#[derive(Debug, Default)]
pub struct Metadata {
    entries: Vec<(String, MetaValue)>,
    index: HashMap<String, usize>,
}

impl Metadata {
    /// Insert a key; returns `false` (without inserting) on a duplicate.
    ///
    /// Public so tests (and fixture generators) can fabricate metadata
    /// without serializing a file; production metadata comes from
    /// [`Gguf::parse`](crate::Gguf::parse).
    pub fn insert(&mut self, key: String, value: MetaValue) -> bool {
        if self.index.contains_key(&key) {
            return false;
        }
        self.index.insert(key.clone(), self.entries.len());
        self.entries.push((key, value));
        true
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate in file order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &MetaValue)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.index.get(key).map(|&i| &self.entries[i].1)
    }

    pub fn require(&self, key: &str) -> Result<&MetaValue, MetaError> {
        self.get(key).ok_or_else(|| MetaError::Missing(key.into()))
    }

    fn typed<'a, T>(
        &'a self,
        key: &str,
        expected: &'static str,
        view: impl FnOnce(&'a MetaValue) -> Option<T>,
    ) -> Result<Option<T>, MetaError> {
        let Some(value) = self.get(key) else {
            return Ok(None);
        };
        match view(value) {
            Some(v) => Ok(Some(v)),
            None => Err(MetaError::WrongType {
                key: key.into(),
                expected,
                found: value.type_name(),
            }),
        }
    }

    fn required<T>(key: &str, got: Result<Option<T>, MetaError>) -> Result<T, MetaError> {
        got?.ok_or_else(|| MetaError::Missing(key.into()))
    }

    /// Optional unsigned integer (any width). Present-but-wrong-type is an
    /// error, not `None`.
    pub fn get_uint(&self, key: &str) -> Result<Option<u64>, MetaError> {
        self.typed(key, "unsigned integer", MetaValue::as_uint)
    }

    pub fn require_uint(&self, key: &str) -> Result<u64, MetaError> {
        Self::required(key, self.get_uint(key))
    }

    pub fn get_int(&self, key: &str) -> Result<Option<i64>, MetaError> {
        self.typed(key, "integer", MetaValue::as_int)
    }

    pub fn get_f32(&self, key: &str) -> Result<Option<f32>, MetaError> {
        self.typed(key, "f32", MetaValue::as_f32)
    }

    pub fn require_f32(&self, key: &str) -> Result<f32, MetaError> {
        Self::required(key, self.get_f32(key))
    }

    pub fn get_bool(&self, key: &str) -> Result<Option<bool>, MetaError> {
        self.typed(key, "bool", MetaValue::as_bool)
    }

    pub fn get_str(&self, key: &str) -> Result<Option<&str>, MetaError> {
        self.typed(key, "string", MetaValue::as_str)
    }

    pub fn require_str(&self, key: &str) -> Result<&str, MetaError> {
        Self::required(key, self.get_str(key))
    }

    pub fn require_str_array(&self, key: &str) -> Result<&[String], MetaError> {
        Self::required(
            key,
            self.typed(key, "array of string", |v| match v {
                MetaValue::Array(MetaArray::String(s)) => Some(s.as_slice()),
                _ => None,
            }),
        )
    }

    pub fn require_f32_array(&self, key: &str) -> Result<&[f32], MetaError> {
        Self::required(
            key,
            self.typed(key, "array of f32", |v| match v {
                MetaValue::Array(MetaArray::F32(s)) => Some(s.as_slice()),
                _ => None,
            }),
        )
    }

    pub fn require_i32_array(&self, key: &str) -> Result<&[i32], MetaError> {
        Self::required(
            key,
            self.typed(key, "array of i32", |v| match v {
                MetaValue::Array(MetaArray::I32(s)) => Some(s.as_slice()),
                _ => None,
            }),
        )
    }

    pub fn require_bool_array(&self, key: &str) -> Result<&[bool], MetaError> {
        Self::required(
            key,
            self.typed(key, "array of bool", |v| match v {
                MetaValue::Array(MetaArray::Bool(s)) => Some(s.as_slice()),
                _ => None,
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Metadata {
        let mut m = Metadata::default();
        assert!(m.insert("a.u32".into(), MetaValue::U32(7)));
        assert!(m.insert("a.u8".into(), MetaValue::U8(3)));
        assert!(m.insert("a.str".into(), MetaValue::String("hi".into())));
        assert!(m.insert("a.f32".into(), MetaValue::F32(1.5)));
        assert!(m.insert(
            "a.scores".into(),
            MetaValue::Array(MetaArray::F32(vec![0.5, -0.5])),
        ));
        m
    }

    #[test]
    fn uint_coerces_widths() {
        let m = sample();
        assert_eq!(m.require_uint("a.u32").unwrap(), 7);
        assert_eq!(m.require_uint("a.u8").unwrap(), 3);
    }

    #[test]
    fn wrong_type_is_an_error_even_for_optional_get() {
        let m = sample();
        let err = m.get_uint("a.str").unwrap_err();
        assert!(matches!(err, MetaError::WrongType { .. }), "{err}");
        assert!(m.get_uint("a.absent").unwrap().is_none());
    }

    #[test]
    fn missing_key_names_the_key() {
        let m = sample();
        let err = m.require_str("nope").unwrap_err();
        assert_eq!(err.to_string(), "missing metadata key `nope`");
    }

    #[test]
    fn typed_arrays() {
        let m = sample();
        assert_eq!(m.require_f32_array("a.scores").unwrap(), &[0.5, -0.5]);
        assert!(m.require_str_array("a.scores").is_err());
    }

    #[test]
    fn duplicate_insert_is_rejected() {
        let mut m = sample();
        assert!(!m.insert("a.u32".into(), MetaValue::U32(9)));
        assert_eq!(m.require_uint("a.u32").unwrap(), 7);
    }
}
