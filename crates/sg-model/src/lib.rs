//! Gemma 4 forward graph: prefill/decode dispatch sequences over `sg-gpu`
//! kernels, sampling, and the CPU reference model used as the M3 parity
//! oracle.
//!
//! Scope and design: `docs/plans/03-inference-pipeline.md`. Lands in M3–M4.
