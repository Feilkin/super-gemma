// Two-range sliding-window prefill attention (plan 03 §prefill): a chunk
// of M query tokens whose visible keys live in TWO places — history
// (absolute positions < q0) in the per-layer RING as it stands BEFORE this
// chunk's append, and the chunk's own keys (positions q0 ..= q0+M−1) in
// the chunk K/V activation buffers. The chunk cannot be appended first:
// early queries' windows would already be overwritten (M2 finding). Order
// per layer is attend, then append.
//
// GQA 32:16, head_dim 256, banded causal mask: query token i (absolute
// position q0 + i) attends positions (q0+i+1−WINDOW) ..= q0+i. History
// positions resolve to ring slots via pos % RING; chunk positions to index
// pos − q0. One streaming softmax walks both ranges in ascending position
// order (ring range first — all history positions precede q0), so the
// reduction order is deterministic and identical to a linear walk.
//
// One wave-sized workgroup per (KV head, query token) computes the
// Q_PER_KV query heads sharing the KV head. The per-key score is a
// subgroup reduction (hardware tree — deterministic).
//
// naga 29 implements subgroupAdd but not the `enable subgroups` directive,
// so the directive is omitted and the shader compiles via plain naga
// (`raw: true`).
//
// Dispatch: [N_KV_HEADS, M].

enable f16;

@group(0) @binding(0) var<storage, read> q: array<f16>; // [M × 32 × HEAD_DIM]
@group(0) @binding(1) var<storage, read> k_ring: array<f16>; // [RING × N_KV_HEADS × HEAD_DIM]
@group(0) @binding(2) var<storage, read> v_ring: array<f16>; // same layout
@group(0) @binding(3) var<storage, read> k_chunk: array<f16>; // [M × N_KV_HEADS × HEAD_DIM]
@group(0) @binding(4) var<storage, read> v_chunk: array<f16>; // same layout
@group(0) @binding(5) var<storage, read_write> out: array<f16>; // [M × 32 × HEAD_DIM]
// Per-step dynamic state, rewritten by the CPU between submits of the
// pre-recorded graph (sg_gpu::StepState): [pos, kv_len_sliding,
// kv_len_global, q0]. q0 = absolute position of the chunk's first token.
@group(0) @binding(6) var<storage, read> step: array<u32>;

struct Push {
    scale: f32,
}
var<immediate> push: Push;

const HEAD_DIM: u32 = #{HEAD_DIM}u;
const N_KV_HEADS: u32 = #{N_KV_HEADS}u;
const Q_PER_KV: u32 = #{Q_PER_KV}u;
const N_Q_HEADS: u32 = N_KV_HEADS * Q_PER_KV;
const WINDOW: u32 = #{WINDOW}u;
const RING: u32 = #{RING}u;
const WG: u32 = #{WG_X}u;
const D: u32 = HEAD_DIM / WG;

@compute @workgroup_size(#{WG_X})
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let lid = lid_v.x;
    let kvh = wg_id.x;
    let i = wg_id.y; // query token within the chunk
    let d0 = lid * D;

    let q0 = step[3];
    let qpos = q0 + i; // inclusive last visible position
    var t_begin = 0u;
    if qpos + 1u > WINDOW {
        t_begin = qpos + 1u - WINDOW;
    }

    var qr: array<array<f32, D>, Q_PER_KV>;
    for (var qi = 0u; qi < Q_PER_KV; qi += 1u) {
        let qbase = (i * N_Q_HEADS + kvh * Q_PER_KV + qi) * HEAD_DIM + d0;
        for (var d = 0u; d < D; d += 1u) {
            qr[qi][d] = f32(q[qbase + d]);
        }
    }

    var m: array<f32, Q_PER_KV>;
    var l: array<f32, Q_PER_KV>;
    var acc: array<array<f32, D>, Q_PER_KV>;
    for (var qi = 0u; qi < Q_PER_KV; qi += 1u) {
        m[qi] = -3.0e38;
        l[qi] = 0.0;
        for (var d = 0u; d < D; d += 1u) {
            acc[qi][d] = 0.0;
        }
    }

    // Range 1: history positions [t_begin, q0) from the pre-append ring.
    let hist_end = min(q0, qpos + 1u);
    for (var t = t_begin; t < hist_end; t += 1u) {
        let kv_base = ((t % RING) * N_KV_HEADS + kvh) * HEAD_DIM + d0;
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
            let corr = exp(m[qi] - m_new); // first key: exp(-inf) = 0
            let w = exp(s - m_new);
            l[qi] = l[qi] * corr + w;
            for (var d = 0u; d < D; d += 1u) {
                acc[qi][d] = acc[qi][d] * corr + w * vv[d];
            }
            m[qi] = m_new;
        }
    }

    // Range 2: the chunk's own positions [max(q0, t_begin), qpos].
    var c_begin = 0u;
    if t_begin > q0 {
        c_begin = t_begin - q0;
    }
    for (var c = c_begin; c <= i; c += 1u) {
        let kv_base = (c * N_KV_HEADS + kvh) * HEAD_DIM + d0;
        var kk: array<f32, D>;
        var vv: array<f32, D>;
        for (var d = 0u; d < D; d += 1u) {
            kk[d] = f32(k_chunk[kv_base + d]);
            vv[d] = f32(v_chunk[kv_base + d]);
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
        let obase = (i * N_Q_HEADS + kvh * Q_PER_KV + qi) * HEAD_DIM + d0;
        for (var d = 0u; d < D; d += 1u) {
            out[obase + d] = f16(acc[qi][d] / l[qi]);
        }
    }
}
