// Decode attention split-K reduction (plan 02): merges the per-split
// streaming-softmax partials written by `attn_decode_global` /
// `attn_decode_sliding` into the final normalized f16 output. Splits merge
// in fixed sequential order — bit-deterministic. Variants per head_dim:
// attn_reduce_d512 (global) and attn_reduce_d256 (sliding).
//
// Dispatch: x = 32 query heads, one wave-sized workgroup each.

enable f16;

@group(0) @binding(0) var<storage, read> part: array<f32>; // [q_head × split × (HEAD_DIM + 2)]
@group(0) @binding(1) var<storage, read_write> out: array<f16>; // [32 × HEAD_DIM]

struct Push {
    n_splits: u32,
}
var<immediate> push: Push;

const HEAD_DIM: u32 = #{HEAD_DIM}u;
const WG: u32 = #{WG_X}u;
const D: u32 = HEAD_DIM / WG;
const PART_STRIDE: u32 = HEAD_DIM + 2u;

@compute @workgroup_size(#{WG_X})
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let lid = lid_v.x;
    let h = wg_id.x;
    let d0 = lid * D;

    var m = -3.0e38;
    var l = 0.0;
    var acc: array<f32, D>;
    for (var d = 0u; d < D; d += 1u) {
        acc[d] = 0.0;
    }

    for (var s = 0u; s < push.n_splits; s += 1u) {
        let row = (h * push.n_splits + s) * PART_STRIDE;
        let ms = part[row + HEAD_DIM];
        let ls = part[row + HEAD_DIM + 1u];
        let m_new = max(m, ms);
        let ca = exp(m - m_new);
        let cb = exp(ms - m_new); // empty split: exp(-3e38 - m) = 0
        for (var d = 0u; d < D; d += 1u) {
            acc[d] = acc[d] * ca + part[row + d0 + d] * cb;
        }
        l = l * ca + ls * cb;
        m = m_new;
    }

    for (var d = 0u; d < D; d += 1u) {
        out[h * HEAD_DIM + d0 + d] = f16(acc[d] / l);
    }
}
