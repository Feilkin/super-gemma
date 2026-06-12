// Global decode attention, split-K pass (plan 02): one query token against
// the full resident context. GQA 32:4, head_dim 512. K and V are SEPARATE
// stores even though the model shares their projection: cached K is
// k_norm-weighted + roped, cached V is weightless-normed and unroped
// (docs/reference/gemma4-forward-graph.md, amended 2026-06-12 — the M2
// "K = V native" reading was wrong).
//
// Dispatch [N_KV_HEADS, n_splits]: each wave-sized workgroup covers one KV
// head and one contiguous context chunk, accumulating streaming-softmax
// partials (m, l, unnormalized acc) for the Q_PER_KV query heads sharing
// the KV head. `attn_reduce_d512` merges the splits — splits in fixed
// order, so the pipeline is bit-deterministic for a fixed split count (the
// engine derives n_splits deterministically from kv_len).
//
// Partials layout: [q_head × split × (HEAD_DIM + 2)] f32; the +2 are m and l.

enable f16;

@group(0) @binding(0) var<storage, read> q: array<f16>; // [32 × HEAD_DIM]
@group(0) @binding(1) var<storage, read> k: array<f16>; // [token × N_KV_HEADS × HEAD_DIM]
@group(0) @binding(2) var<storage, read> v: array<f16>; // [token × N_KV_HEADS × HEAD_DIM]
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
// Per-step dynamic state, rewritten by the CPU between submits of the
// pre-recorded graph (sg_gpu::StepState): [pos, kv_len_sliding,
// kv_len_global, q0].
@group(0) @binding(4) var<storage, read> step: array<u32>;

struct Push {
    n_splits: u32,
    scale: f32,
}
var<immediate> push: Push;

const HEAD_DIM: u32 = #{HEAD_DIM}u;
const N_KV_HEADS: u32 = #{N_KV_HEADS}u;
const Q_PER_KV: u32 = #{Q_PER_KV}u;
const WG: u32 = #{WG_X}u;
const D: u32 = HEAD_DIM / WG;
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

    let kv_len = step[2];
    let chunk = (kv_len + push.n_splits - 1u) / push.n_splits;
    let t_begin = min(split * chunk, kv_len);
    let t_end = min(t_begin + chunk, kv_len);

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
            kk[d] = f32(k[kv_base + d]);
            vv[d] = f32(v[kv_base + d]);
        }
        for (var qi = 0u; qi < Q_PER_KV; qi += 1u) {
            var dot = 0.0;
            for (var d = 0u; d < D; d += 1u) {
                dot += qr[qi][d] * kk[d];
            }
            let s = subgroupAdd(dot) * push.scale;
            let m_new = max(m[qi], s);
            let corr = exp(m[qi] - m_new);
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
