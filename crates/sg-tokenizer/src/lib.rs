//! SentencePiece-compatible tokenizer built from the GGUF-embedded vocab,
//! plus the Gemma 4 chat/tool-call template and a streaming UTF-8-safe
//! detokenizer.
//!
//! Scope and design: `docs/plans/01-gguf-and-tokenizer.md`. Lands in M1.
