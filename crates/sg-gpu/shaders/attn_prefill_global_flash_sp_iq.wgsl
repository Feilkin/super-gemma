// int8-matmul QKᵀ flash (Piece B, the RIGHT approach — cf. the f16-convert
// `_q8` dead end). QKᵀ runs as a SIGNED int8 cooperative-matrix product with a
// per-32-block rescale, mirroring gemm_q4_0_i8: K streams as i8 STRAIGHT from
// the Q8 cache into the coopmat (no LDS staging, no f16 convert — Q8 quants are
// already i8, unlike Q4_0 nibbles), Q is pre-quantized to i8 (a kv_quant_q8-
// style pass after rope), each block's i8 dot accumulates to i32 and is scaled
// to f32 by the block's q_scale ⊗ k_scale outer product (the coopmat-arith fork:
// f32(coop<i32>) · scale). Halves the K DRAM traffic AND runs QKᵀ in int8.
//
// PV STAYS f16: the PV matmul contracts over KEYS, but V is quantized per-32-
// block along HEAD-DIM, so the V scale sits inside the key-sum and cannot factor
// out of an int8 dot (unlike K, whose head-dim quant aligns with the QKᵀ
// contraction). So V is read f16 here; halving V traffic needs a different V
// layout (open follow-up). Everything but QKᵀ — online softmax, PV, epilogue —
// is byte-identical to attn_prefill_global_flash_sp.
//
// Q8 layout (kv_quant_q8 SoA): quants i8 [L × N_KV_HEADS × HEAD_DIM]; scales f16
// [L × N_KV_HEADS × HD_BLOCKS], block b of (key,head) = b·32 ..+31 of head-dim.
// Element layout / determinism / raw:true as the f16 kernel + gemm_q4_0_i8.

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> q_i8: array<i8>;        // [M × N_Q_HEADS × HEAD_DIM] Q8 quants
@group(0) @binding(1) var<storage, read> q_scales: array<f16>;   // [M × N_Q_HEADS × HD_BLOCKS]
@group(0) @binding(2) var<storage, read> k_quants: array<i8>;    // [L × N_KV_HEADS × HEAD_DIM] Q8 quants
@group(0) @binding(3) var<storage, read> k_scales: array<f16>;   // [L × N_KV_HEADS × HD_BLOCKS]
@group(0) @binding(4) var<storage, read> v: array<f16>;          // [L × N_KV_HEADS × HEAD_DIM] (f16)
@group(0) @binding(5) var<storage, read_write> out: array<f16>;  // [M × N_Q_HEADS × HEAD_DIM]
@group(0) @binding(6) var<storage, read> step: array<u32>;

struct Push {
    scale: f32,
}
var<immediate> push: Push;

const HEAD_DIM: u32 = #{HEAD_DIM}u;
const N_KV_HEADS: u32 = #{N_KV_HEADS}u;
const Q_PER_KV: u32 = #{Q_PER_KV}u;
const M_Q: u32 = #{M_Q}u;
const N_K: u32 = #{N_K}u;
const WG: u32 = #{WG_X}u;

const N_Q_HEADS: u32 = N_KV_HEADS * Q_PER_KV;
const Q_ROW_STRIDE: u32 = N_Q_HEADS * HEAD_DIM;     // i8 stride between query tokens
const KV_ROW_STRIDE: u32 = N_KV_HEADS * HEAD_DIM;   // i8 stride between keys
const HD_BLOCKS: u32 = HEAD_DIM / 32u;              // Q8 blocks per head (contraction blocks)
const QS_STRIDE: u32 = N_Q_HEADS * HD_BLOCKS;       // q_scales stride between query tokens
const KS_STRIDE: u32 = N_KV_HEADS * HD_BLOCKS;      // k_scales stride between keys
const N_K_TILES: u32 = N_K / 16u;
const O_TILES: u32 = HEAD_DIM / 16u;
const NEG_INF: f32 = -3.0e38;

var<workgroup> s_stage: array<f32, #{S_STAGE_LEN}>;
var<workgroup> p_stage: array<f16, #{S_STAGE_LEN}>;
var<workgroup> corr_stage: array<f32, 256>;
var<workgroup> o_stage: array<f32, 256>;
var<workgroup> m_row: array<f32, #{M_Q}>;
var<workgroup> l_row: array<f32, #{M_Q}>;
// Per-block QKᵀ rescale: the 16 query / 16 key scales and their outer product.
var<workgroup> qs_l: array<f32, 16>;
var<workgroup> ks_l: array<f32, 16>;
var<workgroup> sc_stage: array<f32, 256>; // [16 q × 16 key] = qs_l ⊗ ks_l

@compute @workgroup_size(#{WG_X})
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let lid = lid_v.x;
    let qh = wg_id.x;
    let kvh = qh / Q_PER_KV;
    let m0 = wg_id.y * M_Q;
    let q0 = step[3];
    let q_base = (m0 * N_Q_HEADS + qh) * HEAD_DIM; // i8 base of this query tile
    let last_key = q0 + m0 + (M_Q - 1u);
    let n_key_tiles = last_key / N_K + 1u;

    if (lid < M_Q) {
        m_row[lid] = NEG_INF;
        l_row[lid] = 0.0;
    }

    var zero_i: coop_mat16x16<i32, C>;
    var zero_c: coop_mat16x16<f32, C>;
    var acc: coop_mat16x16<i32, C>;
    var cs: coop_mat16x16<f32, C>;
    var o: array<coop_mat16x16<f32, C>, O_TILES>;
    for (var ot = 0u; ot < O_TILES; ot += 1u) {
        o[ot] = zero_c;
    }
    workgroupBarrier();

    for (var kt = 0u; kt < n_key_tiles; kt += 1u) {
        // S = Q·Kᵀ in int8: per 16-key sub-tile, sum each head-dim block's
        // i8 dot (i32) rescaled by q_scale ⊗ k_scale into the f32 score tile.
        for (var nt = 0u; nt < N_K_TILES; nt += 1u) {
            let key0 = kt * N_K + nt * 16u;
            cs = zero_c;
            for (var blk = 0u; blk < HD_BLOCKS; blk += 1u) {
                acc = zero_i;
                for (var s = 0u; s < 2u; s += 1u) {
                    let d = blk * 32u + s * 16u;
                    let a = coopLoadT<coop_mat16x16<i8, A>>(&q_i8[q_base + d], Q_ROW_STRIDE);
                    let b = coopLoad<coop_mat16x16<i8, B>>(
                        &k_quants[(key0 * N_KV_HEADS + kvh) * HEAD_DIM + d], KV_ROW_STRIDE);
                    acc = coopMultiplyAdd(a, b, acc);
                }
                // Build this block's [16 q × 16 key] scale outer product in LDS.
                if (lid < 16u) {
                    qs_l[lid] = f32(q_scales[(m0 + lid) * QS_STRIDE + qh * HD_BLOCKS + blk]);
                    ks_l[lid] = f32(k_scales[(key0 + lid) * KS_STRIDE + kvh * HD_BLOCKS + blk]);
                }
                workgroupBarrier();
                for (var i = lid; i < 256u; i += WG) {
                    sc_stage[i] = qs_l[i / 16u] * ks_l[i % 16u];
                }
                workgroupBarrier();
                let scf = coopLoadT<coop_mat16x16<f32, C>>(&sc_stage[0], 16u);
                cs = cs + scf * f32(acc); // arith fork: cs += scale ⊙ f32(i8 dot)
                workgroupBarrier(); // before next blk overwrites qs_l/ks_l/sc_stage
            }
            coopStoreT(cs, &s_stage[nt * 16u], N_K);
        }
        workgroupBarrier();

        // Online softmax (identical to the f16 kernel).
        if (lid < M_Q) {
            let qpos = q0 + m0 + lid;
            let row = lid * N_K;
            let m_old = m_row[lid];
            var m_new = m_old;
            for (var j = 0u; j < N_K; j += 1u) {
                if (kt * N_K + j <= qpos) {
                    m_new = max(m_new, s_stage[row + j] * push.scale);
                }
            }
            let corr = exp(m_old - m_new);
            var lsum = l_row[lid] * corr;
            for (var j = 0u; j < N_K; j += 1u) {
                var p = 0.0;
                if (kt * N_K + j <= qpos) {
                    p = exp(s_stage[row + j] * push.scale - m_new);
                }
                p_stage[row + j] = f16(p);
                lsum += p;
            }
            l_row[lid] = lsum;
            m_row[lid] = m_new;
            for (var c = 0u; c < 16u; c += 1u) {
                corr_stage[lid * 16u + c] = corr;
            }
        }
        workgroupBarrier();

        // O *= corr, then O += P·V in f16 (V stays f16 — see header).
        let cf = coopLoadT<coop_mat16x16<f32, C>>(&corr_stage[0], 16u);
        for (var ot = 0u; ot < O_TILES; ot += 1u) {
            o[ot] = o[ot] * cf;
            for (var s2 = 0u; s2 < N_K_TILES; s2 += 1u) {
                let ap = coopLoadT<coop_mat16x16<f16, A>>(&p_stage[s2 * 16u], N_K);
                let v_key0 = ((kt * N_K + s2 * 16u) * N_KV_HEADS + kvh) * HEAD_DIM;
                let bv = coopLoadT<coop_mat16x16<f16, B>>(&v[v_key0 + ot * 16u], KV_ROW_STRIDE);
                o[ot] = coopMultiplyAdd(ap, bv, o[ot]);
            }
        }
        workgroupBarrier();
    }

    // Epilogue (identical to the f16 kernel).
    for (var ot = 0u; ot < O_TILES; ot += 1u) {
        coopStoreT(o[ot], &o_stage[0], 16u);
        workgroupBarrier();
        for (var i = lid; i < M_Q * 16u; i += WG) {
            let r = i / 16u;
            let c = i % 16u;
            out[(m0 + r) * Q_ROW_STRIDE + qh * HEAD_DIM + ot * 16u + c] = f16(o_stage[i] / l_row[r]);
        }
        workgroupBarrier();
    }
}
