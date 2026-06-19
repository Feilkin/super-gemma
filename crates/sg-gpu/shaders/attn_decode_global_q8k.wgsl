// Global decode attention against a Q8 K cache (Piece A — bank the
// int8-QKᵀ flash win end-to-end by halving the dominant global K traffic).
// Byte-identical to attn_decode_global EXCEPT K streams from the Q8_0 cache
// (i8 quants + per-32-block f16 scales) and is dequanted per element:
// kk[d] = scale · f32(i8). V stays f16; decode Q stays f16 (no Q-quant on
// the decode side — the GEMV-like path is bandwidth-cheap and the i8 K is
// the lever). The dequant math mirrors kv_dequant_q8.
//
// Q8 K layout (kv_quant_q8 / kv_append_global_q8 SoA): quants i8
// [token × N_KV_HEADS × HEAD_DIM]; scales f16 [token × N_KV_HEADS × HD_BLOCKS],
// block b of (token,head) covers head-dim b·32 ..+31.
//
// (The subgroupAdd kernel compiles via plain naga — raw:true — like the f16
// variant; naga_oil rejects `enable subgroups`.)

enable f16;

@group(0) @binding(0) var<storage, read> q: array<f16>;        // [32 × HEAD_DIM]
@group(0) @binding(1) var<storage, read> k_quants: array<i8>;  // [token × N_KV_HEADS × HEAD_DIM]
@group(0) @binding(2) var<storage, read> k_scales: array<f16>; // [token × N_KV_HEADS × HD_BLOCKS]
@group(0) @binding(3) var<storage, read> v: array<f16>;        // [token × N_KV_HEADS × HEAD_DIM]
@group(0) @binding(4) var<storage, read_write> part: array<f32>;
// Per-step dynamic state (sg_gpu::StepState): [pos, kv_len_sliding,
// kv_len_global, q0].
@group(0) @binding(5) var<storage, read> step: array<u32>;

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
const HD_BLOCKS: u32 = HEAD_DIM / 32u;
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
        let sc_base = (t * N_KV_HEADS + kvh) * HD_BLOCKS;
        var kk: array<f32, D>;
        var vv: array<f32, D>;
        for (var d = 0u; d < D; d += 1u) {
            let hd_idx = d0 + d;
            kk[d] = f32(k_scales[sc_base + hd_idx / 32u]) * f32(k_quants[kv_base + d]);
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
