// Q8-KV variant of attn_prefill_global_flash_sp (Piece B): identical online-
// softmax single-pass flash, but K and V stream from the Q8 global KV cache
// (i8 quants + f16 scales, kv_quant_q8 SoA format) instead of f16. Halves the
// K/V DRAM traffic the flash A/B identified as the long-context bottleneck.
//
// Dequant-on-load via an LDS slab, NOT per-tile registers: the f16 kernel
// already pins 256 VGPR (157 spilled), so a per-tile register dequant would
// spill harder. Instead each 16-key × HEAD_DIM K (or V) slab is bulk-dequanted
// to `kv_slab` by all 64 lanes (coalesced global reads, latency hidden across
// the wave) and the matmul coopLoads from LDS — this also lifts the global
// K/V reads OUT of the latency-exposed coopLoad→s_waitcnt matmul loop (the RGP
// stall) into the bulk phase. The PV loop is restructured s2-outer so each V
// slab is dequanted once; the O *= corr rescale is hoisted to a single pass
// before it (it must happen once per key tile, not per s2).
//
// Q8 block layout (matches kv_quant_q8 / kv_append_global_q8): the KV row is
// [N_KV_HEADS × HEAD_DIM]; block b of key `key_g` is scales[key_g·BPR + b] and
// quants[(key_g·BPR + b)·8 + w], BPR = N_KV_HEADS·HEAD_DIM/32 blocks per row,
// block b covers row positions b·32 ..+31. Element (kvh, hd) sits in block
// kvh·(HEAD_DIM/32) + hd/32 at byte hd%32. Parity/determinism as the f16 kernel
// (the dequant is the bit-exact kv_dequant_q8 math).

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> q: array<f16>;          // [M × N_Q_HEADS × HEAD_DIM]
@group(0) @binding(1) var<storage, read> k_quants: array<u32>;   // Q8 i8 quants (8 u32/block)
@group(0) @binding(2) var<storage, read> k_scales: array<f16>;   // Q8 f16 scales (1/block)
@group(0) @binding(3) var<storage, read> v_quants: array<u32>;
@group(0) @binding(4) var<storage, read> v_scales: array<f16>;
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
const Q_ROW_STRIDE: u32 = N_Q_HEADS * HEAD_DIM;
const BPR: u32 = N_KV_HEADS * HEAD_DIM / 32u;  // Q8 blocks per KV row
const HD_BLOCKS: u32 = HEAD_DIM / 32u;         // Q8 blocks per head
const S_TILES: u32 = HEAD_DIM / 16u;           // contraction substeps for QKᵀ
const N_K_TILES: u32 = N_K / 16u;              // key substeps within a tile
const O_TILES: u32 = HEAD_DIM / 16u;           // output head-dim tiles
const SLAB: u32 = 16u * HEAD_DIM;              // one 16-key slab, f16
const NEG_INF: f32 = -3.0e38;

var<workgroup> s_stage: array<f32, #{S_STAGE_LEN}>;
var<workgroup> p_stage: array<f16, #{S_STAGE_LEN}>;
var<workgroup> corr_stage: array<f32, 256>;
var<workgroup> o_stage: array<f32, 256>;
var<workgroup> m_row: array<f32, #{M_Q}>;
var<workgroup> l_row: array<f32, #{M_Q}>;
// Dequant scratch: one 16-key × HEAD_DIM f16 slab, reused for K then V.
var<workgroup> kv_slab: array<f16, SLAB>;

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
    let q_base = (m0 * N_Q_HEADS + qh) * HEAD_DIM;
    let last_key = q0 + m0 + (M_Q - 1u);
    let n_key_tiles = last_key / N_K + 1u;

    if (lid < M_Q) {
        m_row[lid] = NEG_INF;
        l_row[lid] = 0.0;
    }

    var zero_c: coop_mat16x16<f32, C>;
    var cs: coop_mat16x16<f32, C>;
    var o: array<coop_mat16x16<f32, C>, O_TILES>;
    for (var ot = 0u; ot < O_TILES; ot += 1u) {
        o[ot] = zero_c;
    }
    workgroupBarrier();

    for (var kt = 0u; kt < n_key_tiles; kt += 1u) {
        // S = Q·Kᵀ: per 16-key sub-tile, bulk-dequant the K slab → kv_slab,
        // then coopLoad K from LDS.
        for (var nt = 0u; nt < N_K_TILES; nt += 1u) {
            let key_base = kt * N_K + nt * 16u;
            for (var i = lid; i < SLAB; i += WG) {
                let key_g = key_base + i / HEAD_DIM;
                let hd = i % HEAD_DIM;
                let blk = key_g * BPR + kvh * HD_BLOCKS + hd / 32u;
                let qw = k_quants[blk * 8u + (hd % 32u) / 4u];
                let qb = (hd % 32u) % 4u;
                let qv = (i32((qw >> (8u * qb)) & 0xFFu) << 24u) >> 24u; // sign-extend i8
                kv_slab[i] = f16(f32(k_scales[blk]) * f32(qv));
            }
            workgroupBarrier();
            cs = zero_c;
            for (var s = 0u; s < S_TILES; s += 1u) {
                let a = coopLoadT<coop_mat16x16<f16, A>>(&q[q_base + s * 16u], Q_ROW_STRIDE);
                let b = coopLoad<coop_mat16x16<f16, B>>(&kv_slab[s * 16u], HEAD_DIM);
                cs = coopMultiplyAdd(a, b, cs);
            }
            coopStoreT(cs, &s_stage[nt * 16u], N_K);
            workgroupBarrier(); // before next nt overwrites kv_slab
        }

        // Online softmax (unchanged from the f16 kernel).
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

        // Rescale all O once, then O += P·V with V dequanted per s2 slab.
        let cf = coopLoadT<coop_mat16x16<f32, C>>(&corr_stage[0], 16u);
        for (var ot = 0u; ot < O_TILES; ot += 1u) {
            o[ot] = o[ot] * cf;
        }
        for (var s2 = 0u; s2 < N_K_TILES; s2 += 1u) {
            let key_base = kt * N_K + s2 * 16u;
            for (var i = lid; i < SLAB; i += WG) {
                let key_g = key_base + i / HEAD_DIM;
                let hd = i % HEAD_DIM;
                let blk = key_g * BPR + kvh * HD_BLOCKS + hd / 32u;
                let qw = v_quants[blk * 8u + (hd % 32u) / 4u];
                let qb = (hd % 32u) % 4u;
                let qv = (i32((qw >> (8u * qb)) & 0xFFu) << 24u) >> 24u;
                kv_slab[i] = f16(f32(v_scales[blk]) * f32(qv));
            }
            workgroupBarrier();
            let ap = coopLoadT<coop_mat16x16<f16, A>>(&p_stage[s2 * 16u], N_K);
            for (var ot = 0u; ot < O_TILES; ot += 1u) {
                let bv = coopLoadT<coop_mat16x16<f16, B>>(&kv_slab[ot * 16u], HEAD_DIM);
                o[ot] = coopMultiplyAdd(ap, bv, o[ot]);
            }
            workgroupBarrier(); // before next s2 overwrites kv_slab
        }
    }

    // Epilogue: out = O / l, unchanged.
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
