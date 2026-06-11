//! Request orchestration: prompt build → tokenize → cache2 lookup →
//! resume-or-prefill → decode loop → streamed output. Owns the single
//! in-flight conversation and glues `sg-model`, `sg-cache`, and `sg-gpu`
//! together across their dedicated threads.
//!
//! Scope and design: `docs/plans/03-inference-pipeline.md`. Lands in M4–M6.
