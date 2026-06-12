//! Gemma 4 forward graph: prefill/decode dispatch sequences over `sg-gpu`
//! kernels, sampling, and the CPU reference model used as the M3 parity
//! oracle.
//!
//! Scope and design: `docs/plans/03-inference-pipeline.md`. The pinned
//! reference semantics (RoPE, norms, scales, K=V) live in
//! `docs/reference/gemma4-forward-graph.md`. Lands in M3–M4.

pub mod generate;
pub mod graph;
pub mod reference;
pub mod rope;
pub mod sampler;
pub mod session;
pub mod weights;

pub use generate::{EOS_TOKENS, GenerateParams, StopReason};
pub use graph::GpuModel;
pub use reference::{Conventions, CpuKvCache, CpuModel, RefError};
pub use sampler::{Sampler, SamplerParams};
pub use session::Session;
pub use weights::{GpuWeights, UploadError};
