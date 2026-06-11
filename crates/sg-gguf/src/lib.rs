//! GGUF file parsing for the Gemma 4 31B QAT Q4_0 model.
//!
//! Scope (see `docs/plans/01-gguf-and-tokenizer.md`): mmap-based zero-copy GGUF
//! parser, three-way validated `ModelDesc`, Q4_0 block types with a scalar
//! dequantization reference, and the `WeightSource` upload path.
//!
//! Entry point: [`GgufFile::open`] then [`GgufFile::parse`] for a [`Gguf`]
//! view — owned [`Metadata`] and tensor table, zero-copy tensor data slices
//! into the mmap.

pub mod desc;
mod file;
pub mod meta;
mod parse;
pub mod q4_0;
pub mod tensor;

pub use desc::{AttnGeometry, DescError, LayerKind, ModelDesc};
pub use file::GgufFile;
pub use meta::{MetaArray, MetaError, MetaValue, Metadata};
pub use parse::{DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION, Gguf, GgufError};
pub use tensor::{GgmlType, TensorInfo};
