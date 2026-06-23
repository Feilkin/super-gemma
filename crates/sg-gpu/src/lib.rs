//! Vulkan compute runtime on raw `ash`: device/queue management, build-time
//! WGSL→SPIR-V kernel library (naga-oil + naga, see `build.rs`), buffer and
//! descriptor plumbing on unified memory, and command-graph recording.
//!
//! Scope and design: `docs/plans/02-gpu-runtime-and-kernels.md`. Lands in M2.
//! The GPU layer is `ash` (not vulkano) so we own the pipeline barriers between
//! dispatches — vulkano's auto-sync could not see cooperative-matrix accesses
//! and so never barriered consecutive coopmat GEMMs (`docs/ash-migration-rationale.md`).
//!
//! GPU-dependent tests construct a [`GpuContext`] and skip when it errors
//! (Tier 1 CI has no GPU); kernel SPIR-V is still compiled and validated by
//! `build.rs` everywhere.

mod buffer;
mod context;
mod exec;
mod graph;
mod kernel;

pub use buffer::{Buffer, BufferBinding, BufferUsage};
pub use context::GpuContext;
// The element bound for [`Buffer`]/push constants, re-exported so consumers can
// name it without taking a direct bytemuck dependency.
pub use bytemuck::Pod;
pub use graph::{CommandGraph, GpuTimer, GraphRecorder, STEP_WORDS, StepState};
pub use kernel::{KERNELS, Kernel, KernelBlob, kernel_blob};

/// Runtime GPU errors. Off-target conditions (no Vulkan, no capable device)
/// are ordinary variants so tests can skip rather than fail.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("Vulkan library unavailable: {0}")]
    Library(String),
    #[error("Vulkan: {0}")]
    Vk(String),
    #[error("no suitable GPU (need {0})")]
    NoDevice(&'static str),
    #[error("unknown kernel `{0}`")]
    UnknownKernel(String),
    #[error("pipeline: {0}")]
    Pipeline(String),
    #[error("{0}")]
    Validation(String),
}

#[cfg(test)]
mod tests {
    #[test]
    fn registry_contains_valid_spirv() {
        assert!(!super::KERNELS.is_empty());
        for k in super::KERNELS {
            assert!(
                k.spv.len() > 20,
                "{}: SPIR-V blob suspiciously small",
                k.name
            );
            assert_eq!(k.spv.len() % 4, 0, "{}: SPIR-V is 32-bit words", k.name);
            let magic = u32::from_le_bytes(k.spv[0..4].try_into().unwrap());
            assert_eq!(magic, 0x0723_0203, "{}: SPIR-V magic", k.name);
            assert!(k.workgroup.iter().all(|&d| d > 0), "{}: workgroup", k.name);
        }
        assert!(super::kernel_blob("stub").is_some());
    }
}
