//! GGUF file parsing for the Gemma 4 31B QAT Q4_0 model.
//!
//! Scope (see `docs/plans/01-gguf-and-tokenizer.md`): mmap-based zero-copy GGUF
//! parser, three-way validated `ModelDesc`, Q4_0 block types with a scalar
//! dequantization reference, and the `WeightSource` upload path.

pub mod q4_0;
