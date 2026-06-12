// Sliding-window prefill attention (plan 02): a chunk of M query tokens
// against a LINEAR key buffer covering the chunk plus its visible history
// (the engine keeps prefill KV linear; the ring is a decode-time
// structure). GQA 32:16, head_dim 256, banded causal mask: query token i
// (at key index q0 + i, q0 = number of history keys) attends keys
// (q0+i+1−WINDOW) ..= q0+i.
//
// One wave-sized workgroup per (KV head, query token) computes the
// Q_PER_KV query heads sharing the KV head, so each key element is read
// once per workgroup. Streaming softmax; the per-key score is a subgroup
// reduction (hardware tree — deterministic).
//
// naga 29 implements subgroupAdd but not the `enable subgroups` directive,
// so the directive is omitted and the shader compiles via plain naga
// (`raw: true`).
//
// Dispatch: [N_KV_HEADS, M].

enable f16;

@group(0) @binding(0) var<storage, read> q: array<f16>; // [M × 32 × HEAD_DIM]
@group(0) @binding(1) var<storage, read> k: array<f16>; // [L × N_KV_HEADS × HEAD_DIM]
@group(0) @binding(2) var<storage, read> v: array<f16>; // [L × N_KV_HEADS × HEAD_DIM]
@group(0) @binding(3) var<storage, read_write> out: array<f16>; // [M × 32 × HEAD_DIM]

struct Push {
    q0: u32,
    scale: f32,
}
var<immediate> push: Push;

const HEAD_DIM: u32 = #{HEAD_DIM}u;
const N_KV_HEADS: u32 = #{N_KV_HEADS}u;
const Q_PER_KV: u32 = #{Q_PER_KV}u;
const N_Q_HEADS: u32 = N_KV_HEADS * Q_PER_KV;
const WINDOW: u32 = #{WINDOW}u;
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

    let qpos = push.q0 + i; // inclusive last visible key
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

    for (var t = t_begin; t <= qpos; t += 1u) {
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
            let corr = exp(m[qi] - m_new); // first key: exp(-inf) = 0
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
