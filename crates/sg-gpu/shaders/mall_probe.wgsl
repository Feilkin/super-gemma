// Phase-0 MALL probe (STATUS 2026-06-21): does RGP "local video memory bytes"
// count Infinity Cache (MALL) hits, or only DRAM past it? And how fast is MALL?
//
// One streaming read: REPS passes, each sweeping the whole `src` buffer
// (strided by the total thread count, fully coalesced). Between passes the
// buffer's first words are long-evicted from the 2 MB L2, so every pass is an
// L2 miss — but a buffer that fits the 32 MB MALL stays MALL-resident, while a
// buffer larger than MALL misses to DRAM. Run it both ways (SG_PROBE_MB above
// and below 32) and compare, per pass: total logical reads is identical, so
//   - duration  → MALL bandwidth (sub-MALL) vs DRAM bandwidth (super-MALL);
//   - RGP "local video memory bytes" → counts MALL hits if it tracks the
//     logical reads in BOTH cases, excludes them if it only tracks the
//     super-MALL (DRAM) case. That settles whether the GEMM's 800M is real
//     DRAM traffic or partly free MALL hits inflating the "86% bandwidth".
//
// The per-pass index shift (idx = i + r, wrapped) makes each pass read a
// distinct address stream so ACO can't prove the loads loop-invariant and
// compute one pass × REPS. dst[tid] = acc defeats dead-code elimination.

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

struct Push {
    elems: u32, // u32 words in src (= SG_PROBE_MB * 1024 * 256)
    reps: u32,  // sweeps of the whole buffer
}
var<immediate> push: Push;

const WG: u32 = #{WG_X}u;

@compute @workgroup_size(#{WG_X})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngroups: vec3<u32>,
) {
    let stride = ngroups.x * WG; // total threads = full-buffer sweep step
    let elems = push.elems;
    var acc = 0u;
    for (var r = 0u; r < push.reps; r += 1u) {
        var i = gid.x;
        while (i < elems) {
            var idx = i + r;
            if (idx >= elems) {
                idx -= elems;
            }
            acc += src[idx];
            i += stride;
        }
    }
    dst[gid.x] = acc;
}
