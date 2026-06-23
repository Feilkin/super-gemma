//! Weight upload: GGUF → per-tensor GPU buffers (plan 03 §weight upload).
//!
//! One buffer per tensor (`maxStorageBufferRange` is 4 GB; the whole-section
//! buffer is impossible). Q4_0 matmul weights upload verbatim — the
//! gemv/gemm kernels consume the GGUF block layout directly. The Q6_K
//! embedding tensor is repacked to a 4416-byte row stride (21 × 210-byte
//! blocks padded to the next word boundary) as `gemv_q6_k_logits` requires.
//! F32 norm weights upload as-is; `layer_output_scale` stays CPU-side (it is
//! baked into the `add_scaled` push constant at graph-record time).
//!
//! Load path: memcpy out of the GGUF mmap, per tensor. Cold cost is one
//! page-cache streaming of the file; the phase-matched O_DIRECT scatter
//! load (plan 01's `WeightSource`, extended per-tensor) is an engine-startup
//! optimization for M4+ — correctness and layout are what M3 needs.

use sg_gguf::{GgmlType, Gguf, LayerKind, ModelDesc, q6_k};
use sg_gpu::{BufferUsage, GpuContext, GpuError, Buffer};

use crate::reference::RefError;

/// Q6_K rows are padded from 4410 to 4416 bytes = 1104 words, so every row
/// starts word-aligned (must match the `gemv_q6_k_logits` build variant).
pub const Q6K_ROW_WORDS: usize = 1104;

#[derive(Debug, thiserror::Error)]
pub enum UploadError {
    #[error(transparent)]
    Gpu(#[from] GpuError),
    #[error(transparent)]
    Model(#[from] RefError),
    #[error("tensor `{name}`: {what}")]
    BadTensor { name: String, what: String },
}

/// Per-layer GPU-resident weights. Field names mirror the GGUF tensor names.
pub struct LayerWeights {
    pub kind: LayerKind,
    pub attn_norm: Buffer<f32>,
    pub attn_q_norm: Buffer<f32>,
    pub attn_k_norm: Buffer<f32>,
    pub post_attention_norm: Buffer<f32>,
    pub ffn_norm: Buffer<f32>,
    pub post_ffw_norm: Buffer<f32>,
    pub attn_q: Buffer<u32>,
    pub attn_k: Buffer<u32>,
    /// `None` on global layers: K and V share the `attn_k` projection
    /// (they still diverge through their norms + rope downstream).
    pub attn_v: Option<Buffer<u32>>,
    pub attn_output: Buffer<u32>,
    pub ffn_gate: Buffer<u32>,
    pub ffn_up: Buffer<u32>,
    pub ffn_down: Buffer<u32>,
    /// Scalar applied by the FFN-join `add_scaled` at record time.
    pub layer_output_scale: f32,
}

/// All model weights on the GPU, plus the small CPU-side leftovers the
/// graph builder needs.
pub struct GpuWeights {
    pub layers: Vec<LayerWeights>,
    /// Q6_K embeddings / tied LM head, repacked to [`Q6K_ROW_WORDS`] stride.
    pub token_embd: Buffer<u32>,
    pub output_norm: Buffer<f32>,
    /// `[512]` of 1.0: the weight of the weightless V-norm (`x̂·1.0` is
    /// exact). Serves both head_dims — the 256 variant reads a prefix.
    pub norm_ones: Buffer<f32>,
    /// `rope_freqs.weight`, validated; input to the cos/sin table builder.
    pub rope_factors: Vec<f32>,
}

impl GpuWeights {
    pub fn upload(
        ctx: &GpuContext,
        gguf: &Gguf<'_>,
        desc: &ModelDesc,
    ) -> Result<Self, UploadError> {
        let usage = BufferUsage::STORAGE_BUFFER;

        let words = |name: &str| -> Result<Buffer<u32>, UploadError> {
            let bytes = tensor_bytes(gguf, name, GgmlType::Q4_0)?;
            Ok(ctx.buffer_from_iter(
                bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap())),
                usage,
            )?)
        };
        let f32s = |name: &str| -> Result<Buffer<f32>, UploadError> {
            let bytes = tensor_bytes(gguf, name, GgmlType::F32)?;
            Ok(ctx.buffer_from_iter(
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap())),
                usage,
            )?)
        };

        let mut layers = Vec::with_capacity(desc.n_layers);
        for (i, &kind) in desc.layer_kinds.iter().enumerate() {
            let t = |suffix: &str| format!("blk.{i}.{suffix}.weight");
            let scale_bytes = tensor_bytes(gguf, &t("layer_output_scale"), GgmlType::F32)?;
            layers.push(LayerWeights {
                kind,
                attn_norm: f32s(&t("attn_norm"))?,
                attn_q_norm: f32s(&t("attn_q_norm"))?,
                attn_k_norm: f32s(&t("attn_k_norm"))?,
                post_attention_norm: f32s(&t("post_attention_norm"))?,
                ffn_norm: f32s(&t("ffn_norm"))?,
                post_ffw_norm: f32s(&t("post_ffw_norm"))?,
                attn_q: words(&t("attn_q"))?,
                attn_k: words(&t("attn_k"))?,
                attn_v: match kind {
                    LayerKind::Sliding => Some(words(&t("attn_v"))?),
                    LayerKind::Global => None,
                },
                attn_output: words(&t("attn_output"))?,
                ffn_gate: words(&t("ffn_gate"))?,
                ffn_up: words(&t("ffn_up"))?,
                ffn_down: words(&t("ffn_down"))?,
                layer_output_scale: f32::from_le_bytes(scale_bytes[..4].try_into().unwrap()),
            });
        }

        let token_embd = upload_q6k_repacked(ctx, gguf, desc)?;
        let output_norm = f32s("output_norm.weight")?;
        let norm_ones = ctx.buffer_from_iter(std::iter::repeat_n(1.0f32, 512), usage)?;

        let rope_factors = {
            let bytes = tensor_bytes(gguf, "rope_freqs.weight", GgmlType::F32)?;
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };

        Ok(Self {
            layers,
            token_embd,
            output_norm,
            norm_ones,
            rope_factors,
        })
    }
}

/// Repack the Q6_K embedding tensor from its packed 4410-byte rows to the
/// word-aligned [`Q6K_ROW_WORDS`] stride and upload.
fn upload_q6k_repacked(
    ctx: &GpuContext,
    gguf: &Gguf<'_>,
    desc: &ModelDesc,
) -> Result<Buffer<u32>, UploadError> {
    let src = tensor_bytes(gguf, "token_embd.weight", GgmlType::Q6_K)?;
    let row_blocks = desc.hidden_size / q6_k::QK6_K; // 21
    let row_bytes = row_blocks * q6_k::BLOCK_Q6_K_SIZE; // 4410
    let stride = Q6K_ROW_WORDS * 4; // 4416
    debug_assert!(stride >= row_bytes && stride.is_multiple_of(4));

    let mut packed = vec![0u8; desc.vocab_size * stride];
    for r in 0..desc.vocab_size {
        packed[r * stride..][..row_bytes].copy_from_slice(&src[r * row_bytes..][..row_bytes]);
    }
    Ok(ctx.buffer_from_iter(
        packed
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap())),
        BufferUsage::STORAGE_BUFFER,
    )?)
}

fn tensor_bytes<'a>(gguf: &Gguf<'a>, name: &str, dtype: GgmlType) -> Result<&'a [u8], UploadError> {
    let info = gguf
        .tensor(name)
        .ok_or_else(|| RefError::MissingTensor(name.into()))?;
    if info.dtype != dtype {
        return Err(UploadError::BadTensor {
            name: name.into(),
            what: format!("dtype {} != expected {dtype}", info.dtype),
        });
    }
    Ok(gguf.data_of(info))
}
