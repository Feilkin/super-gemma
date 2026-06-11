// Build-pipeline smoke-test kernel: proves WGSL -> naga-oil -> SPIR-V works.
// Replaced by the real kernel library in M2 (docs/plans/02-gpu-runtime-and-kernels.md).

@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x < arrayLength(&data) {
        data[gid.x] = data[gid.x] + 1u;
    }
}
