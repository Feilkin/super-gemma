//! Vulkan compute runtime: device/queue management (vulkano), build-time
//! WGSL→SPIR-V kernel library (naga-oil + naga, see `build.rs`), buffer and
//! descriptor plumbing on unified memory, and command-graph recording.
//!
//! Scope and design: `docs/plans/02-gpu-runtime-and-kernels.md`. Lands in M2.

/// SPIR-V for the build-pipeline smoke-test kernel, compiled at build time
/// from `shaders/stub.wgsl`.
pub const STUB_KERNEL_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/stub.spv"));

#[cfg(test)]
mod tests {
    #[test]
    fn stub_kernel_is_valid_spirv() {
        let spv = super::STUB_KERNEL_SPV;
        assert!(spv.len() > 20, "SPIR-V blob suspiciously small");
        assert_eq!(spv.len() % 4, 0, "SPIR-V is a stream of 32-bit words");
        let magic = u32::from_le_bytes(spv[0..4].try_into().unwrap());
        assert_eq!(magic, 0x0723_0203, "SPIR-V magic number");
    }
}
