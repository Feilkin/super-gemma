//! Q4_0 quantization block type and scalar reference dequantization.
//!
//! Layout matches ggml: blocks of 32 weights, 18 bytes each — an f16 scale `d`
//! followed by 16 bytes of 4-bit quants. Weight `j` (0..16) is the low nibble
//! of `qs[j]`; weight `j + 16` is the high nibble. Dequant: `w = d * (q - 8)`.
//!
//! The scalar path here is the ground truth the WGSL kernels are validated
//! against; it is not a performance path.

use bytemuck::{Pod, Zeroable};
use half::f16;

/// Number of weights per Q4_0 block.
pub const QK4_0: usize = 32;

/// Serialized size of one block in bytes.
pub const BLOCK_Q4_0_SIZE: usize = 18;

/// One Q4_0 block as stored in a GGUF tensor.
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ4_0 {
    /// f16 scale factor.
    pub d: f16,
    /// 4-bit quants, two per byte: low nibbles are weights 0..16, high nibbles 16..32.
    pub qs: [u8; 16],
}

const _: () = assert!(size_of::<BlockQ4_0>() == BLOCK_Q4_0_SIZE);

impl BlockQ4_0 {
    /// Dequantize all 32 weights into `out`.
    pub fn dequantize(&self, out: &mut [f32; QK4_0]) {
        let d = self.d.to_f32();
        for j in 0..16 {
            let q_lo = (self.qs[j] & 0x0F) as i32 - 8;
            let q_hi = (self.qs[j] >> 4) as i32 - 8;
            out[j] = d * q_lo as f32;
            out[j + 16] = d * q_hi as f32;
        }
    }
}

/// Reinterpret a raw Q4_0 tensor byte slice as blocks.
///
/// Errors if the length is not a whole number of blocks.
pub fn blocks_from_bytes(data: &[u8]) -> Result<&[BlockQ4_0], InvalidLength> {
    if !data.len().is_multiple_of(BLOCK_Q4_0_SIZE) {
        return Err(InvalidLength { len: data.len() });
    }
    // bytemuck checks alignment; BlockQ4_0 has align 2 so callers must hand us
    // at least 2-byte-aligned data (GGUF tensor data is 32-byte aligned).
    Ok(bytemuck::cast_slice(data))
}

/// Byte slice length is not a multiple of the Q4_0 block size.
#[derive(Debug, thiserror::Error)]
#[error("byte length {len} is not a multiple of the Q4_0 block size ({BLOCK_Q4_0_SIZE})")]
pub struct InvalidLength {
    pub len: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_size_is_18_bytes() {
        assert_eq!(size_of::<BlockQ4_0>(), 18);
        assert_eq!(align_of::<BlockQ4_0>(), 2);
    }

    #[test]
    fn dequantize_known_block() {
        // d = 2.0; qs[0] = 0x39 -> low nibble 9 (w0 = 2*(9-8) = 2), high nibble 3 (w16 = 2*(3-8) = -10)
        let mut qs = [0x88u8; 16]; // all nibbles 8 -> weight 0.0
        qs[0] = 0x39;
        let block = BlockQ4_0 {
            d: f16::from_f32(2.0),
            qs,
        };
        let mut out = [f32::NAN; QK4_0];
        block.dequantize(&mut out);
        assert_eq!(out[0], 2.0);
        assert_eq!(out[16], -10.0);
        for (i, w) in out.iter().enumerate() {
            if i != 0 && i != 16 {
                assert_eq!(*w, 0.0, "weight {i}");
            }
        }
    }

    #[test]
    fn dequantize_extremes() {
        // Nibble 0 -> -8d (most negative), nibble 0xF -> +7d (most positive).
        let block = BlockQ4_0 {
            d: f16::from_f32(1.5),
            qs: [0xF0; 16],
        };
        let mut out = [0.0; QK4_0];
        block.dequantize(&mut out);
        assert_eq!(out[0], -12.0); // 1.5 * (0 - 8)
        assert_eq!(out[16], 10.5); // 1.5 * (15 - 8)
    }

    #[test]
    fn blocks_from_bytes_validates_length() {
        let ok = vec![0u8; BLOCK_Q4_0_SIZE * 4];
        assert_eq!(blocks_from_bytes(&ok).unwrap().len(), 4);
        let bad = vec![0u8; BLOCK_Q4_0_SIZE * 4 + 1];
        assert!(blocks_from_bytes(&bad).is_err());
    }
}
