// PROTOTYPE: _ipv with the per-block rescale's [16×16] scale built by 0-stride
// coopLoad broadcasts instead of a 256-element scalar LDS build (both the QKᵀ
// and PV rescales). Removes the sc_stage LDS array, the scalar outer-product
// loop, and one barrier per rescale. See the rescale doc in the QKᵀ loop.
//
// int8-matmul QKᵀ AND PV flash (Piece B). Extends attn_prefill_global_flash_sp_iq
// (int8 QKᵀ, f16 PV) by also running PV in signed int8 — halving V DRAM traffic
// on top of K. Both matmuls are i8×i8→i32 coopmat products with a per-32-block
// rescale (the gemm_q4_0_i8 pattern); the only new wrinkle is that V's quant
// blocks run along the KEY axis (the PV contraction) so the per-block V scale
// factors out of the i8 key-dot — see docs/q8-kv-flash-impl.md §2 and the
// q8_quant_v reference.
//
// QKᵀ: contraction = head-dim, K i8 straight from the Q8 cache, Q pre-quantized
// i8, rescale q_scale ⊗ k_scale per 32-head-dim block (identical to _iq).
//
// PV: contraction = KEYS. V streams i8 from the cache (q8_quant_v: quants keep
// the [L × N_KV_HEADS × HEAD_DIM] layout; scales are key-blocked,
// [ceil(L/32) × N_KV_HEADS × HEAD_DIM]). P is quantized to i8 IN-KERNEL per
// 32-KEY block (amax → f16 scale → i8, the PV analog of Q-quant) into a
// workgroup array<i8> and coopLoad'd — in-kernel i8 operand production is the
// proven gemm_q4_0_i8 / coop_i8_lds_smoke idiom. Each 32-key block's i8 dot
// accumulates to i32 (within the MMA — never a coopmat `+`, cf. the arith fork's
// silent FAdd) and is rescaled by p_scale[q,kb] ⊗ v_scale[kb,c] into the f32 O.
//
// Element layout / determinism / raw:true as _iq + gemm_q4_0_i8.

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> q_i8: array<i8>;        // [M × N_Q_HEADS × HEAD_DIM] Q8 quants
@group(0) @binding(1) var<storage, read> q_scales: array<f16>;   // [M × N_Q_HEADS × HD_BLOCKS]
@group(0) @binding(2) var<storage, read> k_quants: array<i8>;    // [L × N_KV_HEADS × HEAD_DIM] Q8 quants
@group(0) @binding(3) var<storage, read> k_scales: array<f16>;   // [L × N_KV_HEADS × HD_BLOCKS]
@group(0) @binding(4) var<storage, read> v_quants: array<i8>;    // [L × N_KV_HEADS × HEAD_DIM] Q8 quants
@group(0) @binding(5) var<storage, read> v_scales: array<f16>;   // [ceil(L/32) × N_KV_HEADS × HEAD_DIM]
@group(0) @binding(6) var<storage, read_write> out: array<f16>;  // [M × N_Q_HEADS × HEAD_DIM]
@group(0) @binding(7) var<storage, read> step: array<u32>;

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
const HD_BLOCKS: u32 = HEAD_DIM / 32u;              // Q8 blocks per head (QKᵀ contraction blocks)
const QS_STRIDE: u32 = N_Q_HEADS * HD_BLOCKS;       // q_scales stride between query tokens
const KS_STRIDE: u32 = N_KV_HEADS * HD_BLOCKS;      // k_scales stride between keys
const VS_STRIDE: u32 = N_KV_HEADS * HEAD_DIM;       // v_scales stride between key-blocks
const N_K_TILES: u32 = N_K / 16u;
const KB: u32 = N_K / 32u;                          // 32-key (PV contraction) blocks per N_K tile
const O_TILES: u32 = HEAD_DIM / 16u;
const NEG_INF: f32 = -3.0e38;

var<workgroup> s_stage: array<f32, #{S_STAGE_LEN}>;       // scores, reused to hold P (f32) for quant
var<workgroup> p_i8_stage: array<i8, #{S_STAGE_LEN}>;     // P quantized i8 [16 q × N_K key]
var<workgroup> p_scales: array<f32, #{P_SCALES_LEN}>;     // P scale per (q row, 32-key block)
var<workgroup> corr_stage: array<f32, 256>;
var<workgroup> o_stage: array<f32, 256>;
var<workgroup> m_row: array<f32, #{M_Q}>;
var<workgroup> l_row: array<f32, #{M_Q}>;
// Per-block rescale row/col scale vectors (16 each). The [16×16] outer product
// is built from these by 0-stride coopLoads (see the rescale doc in the QKᵀ
// loop) — no LDS matrix, unlike the f16 PV baseline / the scalar build.
var<workgroup> qs_l: array<f32, 16>;
var<workgroup> ks_l: array<f32, 16>;

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
                // --- Per-block rescale -------------------------------------
                // `acc` holds the i32 dot Σ Q_i8·K_i8 for this 32-elem block.
                // The true score is acc[q][key] · q_scale[q] · k_scale[key]
                // (each operand was quantized with its own per-32-block scale),
                // so we must multiply `acc` by the rank-1 outer product
                // qs[q] ⊗ ks[key]. A coopmat has no per-element register access,
                // so the scale has to enter as its OWN [16×16] fragment `scf`
                // applied with the whole-fragment op `scf * f32(acc)`.
                //
                // We build `scf` from the two 16-element scale vectors with
                // 0-STRIDE coopLoads (broadcast a vector across the fragment —
                // verified by coop_bcast_lds_smoke), so the element-wise product
                // is the outer product, with NO 16×16 LDS materialization. One
                // load is transposed (coopLoadT) so the two broadcasts run on
                // opposite axes; which vector gets the T is pinned by parity
                // (it must land in the same fragment layout as `acc`, which the
                // scalar-`sc_stage` baseline reached via coopLoadT(.,16)).
                if (lid < 16u) {
                    qs_l[lid] = f32(q_scales[(m0 + lid) * QS_STRIDE + qh * HD_BLOCKS + blk]);
                    ks_l[lid] = f32(k_scales[(key0 + lid) * KS_STRIDE + kvh * HD_BLOCKS + blk]);
                }
                workgroupBarrier(); // qs_l/ks_l written before the 0-stride loads
                let scf = coopLoad<coop_mat16x16<f32, C>>(&qs_l[0], 0u)
                    * coopLoadT<coop_mat16x16<f32, C>>(&ks_l[0], 0u);
                cs = cs + scf * f32(acc); // arith fork: cs += scale ⊙ f32(i8 dot)
                workgroupBarrier(); // before next blk overwrites qs_l/ks_l
            }
            coopStoreT(cs, &s_stage[nt * 16u], N_K);
        }
        workgroupBarrier();

        // Online softmax + in-kernel P-quant. P is written f32 into s_stage
        // (scores no longer needed), then quantized per 32-KEY block into
        // p_i8_stage with one f16-rounded scale per (row, block).
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
                s_stage[row + j] = p;
            }
            m_row[lid] = m_new;
            // Quantize P per 32-key block AND accumulate the softmax denominator
            // from the DEQUANTIZED P, so l matches the i8 P the PV numerator uses
            // — otherwise small P's that round to 0 drop from the numerator but
            // stay in l, biasing O low (a systematic ppl shift, not just noise).
            for (var kb = 0u; kb < KB; kb += 1u) {
                var amax = 0.0;
                for (var t = 0u; t < 32u; t += 1u) {
                    amax = max(amax, abs(s_stage[row + kb * 32u + t]));
                }
                let d = amax * (1.0 / 127.0);
                var id = 0.0;
                if (d > 0.0) {
                    id = 1.0 / d;
                }
                let ds = f32(f16(d)); // f16-rounded scale, matches q8 dequant
                p_scales[lid * KB + kb] = ds;
                for (var t = 0u; t < 32u; t += 1u) {
                    let q = clamp(i32(round(s_stage[row + kb * 32u + t] * id)), -128, 127);
                    p_i8_stage[row + kb * 32u + t] = i8(q);
                    lsum += ds * f32(q);
                }
            }
            l_row[lid] = lsum;
            for (var c = 0u; c < 16u; c += 1u) {
                corr_stage[lid * 16u + c] = corr;
            }
        }
        workgroupBarrier();

        // O *= corr, then O += P·V in int8: per 32-KEY block, i8 dot (i32)
        // rescaled by p_scale[q,kb] ⊗ v_scale[kb,c] into the f32 O tile.
        let cf = coopLoadT<coop_mat16x16<f32, C>>(&corr_stage[0], 16u);
        let vkb0 = (kt * N_K) / 32u; // global key-block index of this tile's first block
        for (var ot = 0u; ot < O_TILES; ot += 1u) {
            o[ot] = o[ot] * cf;
            for (var kb = 0u; kb < KB; kb += 1u) {
                acc = zero_i;
                for (var s2 = 0u; s2 < 2u; s2 += 1u) {
                    let ks = kb * 2u + s2; // 16-key sub-tile within this N_K tile
                    let ap = coopLoadT<coop_mat16x16<i8, A>>(&p_i8_stage[ks * 16u], N_K);
                    let v_key0 = ((kt * N_K + ks * 16u) * N_KV_HEADS + kvh) * HEAD_DIM;
                    let bv = coopLoadT<coop_mat16x16<i8, B>>(&v_quants[v_key0 + ot * 16u], KV_ROW_STRIDE);
                    acc = coopMultiplyAdd(ap, bv, acc);
                }
                // Per-block PV rescale (same construction as the QKᵀ rescale
                // above): scale the i32 PV dot by p_scale[q,kb] ⊗ v_scale[kb,c],
                // the row/col scales built into a fragment by 0-stride coopLoads.
                if (lid < 16u) {
                    qs_l[lid] = p_scales[lid * KB + kb];
                    ks_l[lid] = f32(v_scales[((vkb0 + kb) * N_KV_HEADS + kvh) * HEAD_DIM + ot * 16u + lid]);
                }
                workgroupBarrier(); // qs_l/ks_l written before the 0-stride loads
                let scf = coopLoad<coop_mat16x16<f32, C>>(&qs_l[0], 0u)
                    * coopLoadT<coop_mat16x16<f32, C>>(&ks_l[0], 0u);
                o[ot] = o[ot] + scf * f32(acc);
                workgroupBarrier(); // before next (ot,kb) overwrites qs_l/ks_l
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
