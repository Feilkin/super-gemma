//! Tensor-table types: the ggml dtypes this model can ship and per-tensor
//! metadata.

use std::fmt;

/// The ggml tensor dtypes `sg-gguf` supports — exactly the set the Gemma 4 QAT
/// Q4_0 export contains (plan 01): `Q4_0` for matmul weights, `F32` for norms,
/// `Q6_K` for the token embeddings (found in the real file, resolving plan
/// 01's open question), plus `F16`/`Q8_0` which KV snapshots use. Any other
/// type id in the tensor table is a parse error naming the tensor; we support
/// what the file contains, not the whole ggml zoo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)] // ggml's canonical type names
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q8_0,
    Q6_K,
}

impl GgmlType {
    /// Map a raw ggml type id from the tensor table to a supported type.
    pub fn from_raw(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            8 => Some(Self::Q8_0),
            14 => Some(Self::Q6_K),
            _ => None,
        }
    }

    /// Elements per quantization block (1 for unquantized types).
    pub const fn block_elems(self) -> u64 {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q4_0 | Self::Q8_0 => 32,
            Self::Q6_K => 256,
        }
    }

    /// Serialized bytes per block.
    pub const fn block_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 18,
            Self::Q8_0 => 34,
            // ql[128] + qh[64] + scales[16] + d(f16)
            Self::Q6_K => 210,
        }
    }

    /// Serialized byte length of a tensor with `elems` elements.
    ///
    /// `None` if `elems` is not a whole number of blocks (or the size
    /// overflows u64).
    pub fn byte_len(self, elems: u64) -> Option<u64> {
        if !elems.is_multiple_of(self.block_elems()) {
            return None;
        }
        (elems / self.block_elems()).checked_mul(self.block_bytes())
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q8_0 => "Q8_0",
            Self::Q6_K => "Q6_K",
        })
    }
}

/// One entry of the GGUF tensor table.
///
/// All offsets and sizes were bounds-checked against the data section during
/// parsing.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions in ggml order: `dims[0]` is the fastest-varying axis (row
    /// length), i.e. the *reverse* of the usual row-major shape notation.
    pub dims: Vec<u64>,
    pub dtype: GgmlType,
    /// Byte offset of the tensor data relative to the start of the data
    /// section (a multiple of the file's alignment).
    pub offset: u64,
    /// Total serialized byte length.
    pub byte_len: u64,
}

impl TensorInfo {
    /// Total number of elements (product of `dims`; overflow was rejected at
    /// parse time).
    pub fn elem_count(&self) -> u64 {
        self.dims.iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_len_rejects_partial_blocks() {
        assert_eq!(GgmlType::Q4_0.byte_len(64), Some(36));
        assert_eq!(GgmlType::Q4_0.byte_len(48), None);
        assert_eq!(GgmlType::Q8_0.byte_len(32), Some(34));
        assert_eq!(GgmlType::F32.byte_len(3), Some(12));
        assert_eq!(GgmlType::F16.byte_len(u64::MAX), None); // overflow
    }

    #[test]
    fn raw_ids_match_ggml() {
        assert_eq!(GgmlType::from_raw(0), Some(GgmlType::F32));
        assert_eq!(GgmlType::from_raw(1), Some(GgmlType::F16));
        assert_eq!(GgmlType::from_raw(2), Some(GgmlType::Q4_0));
        assert_eq!(GgmlType::from_raw(8), Some(GgmlType::Q8_0));
        assert_eq!(GgmlType::from_raw(14), Some(GgmlType::Q6_K));
        assert_eq!(GgmlType::from_raw(3), None); // Q4_1: not in this model
    }

    #[test]
    fn q6_k_block_geometry() {
        assert_eq!(GgmlType::Q6_K.byte_len(256), Some(210));
        assert_eq!(GgmlType::Q6_K.byte_len(512), Some(420));
        assert_eq!(GgmlType::Q6_K.byte_len(128), None);
    }
}
