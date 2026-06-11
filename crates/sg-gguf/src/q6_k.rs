//! Q6_K quantization block type and scalar reference dequantization.
//!
//! The token embeddings (and, via the tied LM head, the final matmul input)
//! ship as Q6_K in the QAT GGUF. Layout matches ggml (`ggml-common.h`,
//! verified against upstream 2026-06-11): super-blocks of 256 weights,
//! 210 bytes — 128 bytes of low 4-bit quants `ql`, 64 bytes of high 2-bit
//! quants `qh`, 16 signed 8-bit sub-block scales, and an f16 super-block
//! scale `d`. Weight = `d * scale[sub_block] * (q - 32)` with `q` the
//! 6-bit quant.
//!
//! The scalar path here is the ground truth the WGSL kernels are validated
//! against (mirrors `dequantize_row_q6_K`); it is not a performance path.

use bytemuck::{Pod, Zeroable};
use half::f16;

/// Number of weights per Q6_K super-block.
pub const QK6_K: usize = 256;

/// Serialized size of one block in bytes.
pub const BLOCK_Q6_K_SIZE: usize = 210;

/// One Q6_K super-block as stored in a GGUF tensor.
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ6K {
    /// Low 4 bits of the quants.
    pub ql: [u8; QK6_K / 2],
    /// High 2 bits of the quants, four per byte.
    pub qh: [u8; QK6_K / 4],
    /// Signed sub-block scales (16 sub-blocks of 16 weights).
    pub scales: [i8; QK6_K / 16],
    /// f16 super-block scale.
    pub d: f16,
}

const _: () = assert!(size_of::<BlockQ6K>() == BLOCK_Q6_K_SIZE);

impl BlockQ6K {
    /// Dequantize all 256 weights into `out` (ggml `dequantize_row_q6_K`).
    pub fn dequantize(&self, out: &mut [f32; QK6_K]) {
        let d = self.d.to_f32();
        // Two halves of 128 weights; within each half, 32 lanes spread
        // across four 32-weight groups sharing the same qh byte.
        for half in 0..2 {
            let ql = &self.ql[half * 64..];
            let qh = &self.qh[half * 32..];
            let sc = &self.scales[half * 8..];
            let y = &mut out[half * 128..][..128];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0F) | ((qh[l] & 0x03) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i32 - 32;
                y[l] = d * sc[is] as f32 * q1 as f32;
                y[l + 32] = d * sc[is + 2] as f32 * q2 as f32;
                y[l + 64] = d * sc[is + 4] as f32 * q3 as f32;
                y[l + 96] = d * sc[is + 6] as f32 * q4 as f32;
            }
        }
    }
}

/// Reinterpret a raw Q6_K tensor byte slice as blocks.
///
/// Errors if the length is not a whole number of blocks.
pub fn blocks_from_bytes(data: &[u8]) -> Result<&[BlockQ6K], InvalidLength> {
    if !data.len().is_multiple_of(BLOCK_Q6_K_SIZE) {
        return Err(InvalidLength { len: data.len() });
    }
    // BlockQ6K has align 2 (f16); GGUF tensor data is 32-byte aligned.
    Ok(bytemuck::cast_slice(data))
}

/// Byte slice length is not a multiple of the Q6_K block size.
#[derive(Debug, thiserror::Error)]
#[error("byte length {len} is not a multiple of the Q6_K block size ({BLOCK_Q6_K_SIZE})")]
pub struct InvalidLength {
    pub len: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_size_is_210_bytes() {
        assert_eq!(size_of::<BlockQ6K>(), 210);
        assert_eq!(align_of::<BlockQ6K>(), 2);
    }

    #[test]
    fn dequantize_known_block() {
        let mut block = BlockQ6K {
            ql: [0; 128],
            qh: [0; 64],
            scales: [1; 16],
            d: f16::from_f32(0.5),
        };
        // All-zero quants: q = 0 + 0 - 32 = -32 → w = 0.5 * 1 * -32 = -16.
        let mut out = [f32::NAN; QK6_K];
        block.dequantize(&mut out);
        assert!(out.iter().all(|&w| w == -16.0));

        // Weight 0 lives in ql[0] low nibble + qh[0] bits 0..2.
        // q = 0xF | (0b11 << 4) = 63 → w = 0.5 * 2 * (63 - 32) = 31.
        block.ql[0] = 0x0F;
        block.qh[0] = 0x03;
        block.scales[0] = 2;
        block.dequantize(&mut out);
        assert_eq!(out[0], 31.0);
        // Weight 64 (third group, l=0): ql[0] high nibble | qh[0] bits 4..6,
        // scale index 4. q = 0 | 0 - 32; w = 0.5 * 1 * -32 (unchanged scale).
        assert_eq!(out[64], -16.0);

        // Weight 96 (fourth group, l=0): ql[32] high nibble | qh[0] bits 6..8.
        block.ql[32] = 0xA0; // high nibble 0xA
        block.qh[0] |= 0b1100_0000; // top bits 0b11
        block.scales[6] = -3;
        block.dequantize(&mut out);
        // q = 0xA | (0b11 << 4) = 58 → w = 0.5 * -3 * (58 - 32) = -39.
        assert_eq!(out[96], -39.0);
    }

    #[test]
    fn negative_scales_and_signed_range() {
        let block = BlockQ6K {
            ql: [0xFF; 128],
            qh: [0xFF; 64],
            scales: [-128; 16],
            d: f16::from_f32(1.0),
        };
        let mut out = [0.0; QK6_K];
        block.dequantize(&mut out);
        // q = 0xF | (3 << 4) = 63 → w = 1.0 * -128 * 31 = -3968 everywhere.
        assert!(out.iter().all(|&w| w == -3968.0));
    }

    #[test]
    fn blocks_from_bytes_validates_length() {
        let ok = vec![0u8; BLOCK_Q6_K_SIZE * 2];
        assert_eq!(blocks_from_bytes(&ok).unwrap().len(), 2);
        assert!(blocks_from_bytes(&ok[..BLOCK_Q6_K_SIZE + 1]).is_err());
    }
}
