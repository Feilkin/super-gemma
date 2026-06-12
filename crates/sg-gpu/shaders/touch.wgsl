// Synchronization shim, not a compute kernel. vulkano's auto-sync derives
// inter-dispatch barriers from SPIR-V reflection of each pipeline's buffer
// usage — and reflection does NOT see cooperative-matrix accesses (the same
// blindness that forces kernel.rs to build descriptor layouts manually).
// A buffer consumed only via coopLoad (gemm_q4_0's `x`) therefore gets no
// write→read barrier after its producer, racing inside recorded graphs.
//
// Dispatching this kernel on that buffer between producer and gemm fixes
// it: the DECLARED read_write usage below is what auto-sync tracks, so it
// emits a barrier whose execution scope orders all subsequent compute
// (including the gemm) after the producer, and whose memory scope makes
// the buffer's writes visible. The guard is never true (gid.x < 2^32−1 for
// any real dispatch), so nothing is actually written; naga and ACO keep
// the access because the condition is runtime-dependent.
//
// Dispatch: [1, 1, 1] — the workgroup itself does nothing.

@group(0) @binding(0) var<storage, read_write> b: array<u32>;

@compute @workgroup_size(#{WG_X})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x == 0xFFFFFFFFu {
        b[0] = 0u;
    }
}
