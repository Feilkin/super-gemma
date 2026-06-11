// Build-pipeline smoke-test kernel: proves WGSL -> naga-oil -> SPIR-V works,
// and exercises the runtime's dispatch path in tests (buffer + push constant).

struct Push {
    add: u32,
}

// naga 29 spells the push-constant address space `immediate`.
var<immediate> push: Push;

@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(#{WG_X})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x < arrayLength(&data) {
        data[gid.x] = data[gid.x] + push.add;
    }
}
