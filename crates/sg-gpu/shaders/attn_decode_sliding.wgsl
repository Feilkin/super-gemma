// Sliding-window decode attention, split-K pass (plan 02): one query token
// against the per-layer KV ring (≤ 1024 slots, which is exactly the
// window). GQA 32:16, head_dim 256.
//
// Dispatch [N_KV_HEADS, n_splits]: each wave-sized workgroup covers one KV
// head and one contiguous slot chunk, accumulating streaming-softmax
// partials (m, l, unnormalized acc) for the Q_PER_KV query heads sharing
// the KV head; each ring element is read once per workgroup. The per-slot
// score is a subgroup reduction (hardware tree — deterministic).
// `attn_reduce_d256` merges the splits. Split-K exists for parallelism: 16
// workgroups leave a 40-CU GPU latency-bound (measured 315 µs → ~50 µs
// with 8 splits); the engine derives n_splits deterministically from
// kv_len, so the pipeline stays bit-deterministic.
//
// The kernel iterates ring slots in PHYSICAL order: softmax is
// order-invariant mathematically, and bit-exactness (plan 06) only needs a
// reproducible order, which physical order is — cache resume restores the
// exact ring layout (plan 04). No ring-head arithmetic in the kernel; the
// first `kv_len` slots are valid and all in-window by construction.
//
// naga 29 implements subgroupAdd but not the `enable subgroups` directive,
// so the directive is omitted and the shader compiles via plain naga
// (`raw: true`).
//
// Partials layout: [q_head × split × (HEAD_DIM + 2)] f32; the +2 are m and l.

enable f16;

@group(0) @binding(0) var<storage, read> q: array<f16>; // [32 × HEAD_DIM]
@group(0) @binding(1) var<storage, read> k_ring: array<f16>; // [slot × N_KV_HEADS × HEAD_DIM]
@group(0) @binding(2) var<storage, read> v_ring: array<f16>; // same layout
@group(0) @binding(3) var<storage, read_write> part: array<f32>;

struct Push {
    kv_len: u32,
    n_splits: u32,
    scale: f32,
}
var<immediate> push: Push;

const HEAD_DIM: u32 = #{HEAD_DIM}u;
const N_KV_HEADS: u32 = #{N_KV_HEADS}u;
const Q_PER_KV: u32 = #{Q_PER_KV}u;
const WG: u32 = #{WG_X}u;
const D: u32 = HEAD_DIM / WG; // dims per lane
const PART_STRIDE: u32 = HEAD_DIM + 2u;

@compute @workgroup_size(#{WG_X})
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let lid = lid_v.x;
    let kvh = wg_id.x;
    let split = wg_id.y;
    let d0 = lid * D;

    let chunk = (push.kv_len + push.n_splits - 1u) / push.n_splits;
    let t_begin = min(split * chunk, push.kv_len);
    let t_end = min(t_begin + chunk, push.kv_len);

    // This lane's slice of the Q_PER_KV query heads.
    var qr: array<array<f32, D>, Q_PER_KV>;
    for (var qi = 0u; qi < Q_PER_KV; qi += 1u) {
        let qbase = (kvh * Q_PER_KV + qi) * HEAD_DIM + d0;
        for (var d = 0u; d < D; d += 1u) {
            qr[qi][d] = f32(q[qbase + d]);
        }
    }

    var m: array<f32, Q_PER_KV>;
    var l: array<f32, Q_PER_KV>;
    var acc: array<array<f32, D>, Q_PER_KV>;
    for (var qi = 0u; qi < Q_PER_KV; qi += 1u) {
        m[qi] = -3.0e38; // empty split stays -3e38/0; the reducer handles it
        l[qi] = 0.0;
        for (var d = 0u; d < D; d += 1u) {
            acc[qi][d] = 0.0;
        }
    }

    for (var t = t_begin; t < t_end; t += 1u) {
        let kv_base = (t * N_KV_HEADS + kvh) * HEAD_DIM + d0;
        var kk: array<f32, D>;
        var vv: array<f32, D>;
        for (var d = 0u; d < D; d += 1u) {
            kk[d] = f32(k_ring[kv_base + d]);
            vv[d] = f32(v_ring[kv_base + d]);
        }
        for (var qi = 0u; qi < Q_PER_KV; qi += 1u) {
            var dot = 0.0;
            for (var d = 0u; d < D; d += 1u) {
                dot += qr[qi][d] * kk[d];
            }
            let s = subgroupAdd(dot) * push.scale;
            let m_new = max(m[qi], s);
            let corr = exp(m[qi] - m_new); // first slot: exp(-inf) = 0
            let w = exp(s - m_new);
            l[qi] = l[qi] * corr + w;
            for (var d = 0u; d < D; d += 1u) {
                acc[qi][d] = acc[qi][d] * corr + w * vv[d];
            }
            m[qi] = m_new;
        }
    }

    for (var qi = 0u; qi < Q_PER_KV; qi += 1u) {
        let row = ((kvh * Q_PER_KV + qi) * push.n_splits + split) * PART_STRIDE;
        for (var d = 0u; d < D; d += 1u) {
            part[row + d0 + d] = acc[qi][d];
        }
        if lid == 0u {
            part[row + HEAD_DIM] = m[qi];
            part[row + HEAD_DIM + 1u] = l[qi];
        }
    }
}
