// KV append (plan 02): copy freshly projected K or V rows for `n` tokens
// into the per-layer KV store. The destination slot comes from a push
// constant so the pre-recorded decode graph never needs re-recording.
//
// Variants: kv_append_sliding writes into the 1024-slot ring (slot =
// (pos + token) mod RING); kv_append_global appends linearly (slot =
// pos + token). Rows are [n_kv_heads × head_dim] f16, contiguous in both
// source and destination — the kernel is a strided copy, one thread per
// element.
//
// Dispatch: x covers n_tokens × ROW_LEN elements (arrayLength of src).

enable f16;

@group(0) @binding(0) var<storage, read> src: array<f16>; // [n_tokens × ROW_LEN]
@group(0) @binding(1) var<storage, read_write> dst: array<f16>; // [slots × ROW_LEN]

struct Push {
    /// Absolute position of the first appended token.
    pos: u32,
}
var<immediate> push: Push;

const ROW_LEN: u32 = #{ROW_LEN}u;

@compute @workgroup_size(#{WG_X})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= arrayLength(&src) {
        return;
    }
    let token = gid.x / ROW_LEN;
    let j = gid.x % ROW_LEN;
#ifdef RING
    let slot = (push.pos + token) % #{RING}u;
#else
    let slot = push.pos + token;
#endif
    dst[slot * ROW_LEN + j] = src[gid.x];
}
