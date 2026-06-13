// Coopmat global prefill, register-resident-O two-pass (plan 02, profile
// rank #1 rewrite — v2). Replaces attn_prefill_global's per-(kv_head, token)
// serial key walk (O(ctx²)/prompt, 141 ms/layer @32K, watchdog-cliff driver).
//
// Y[i] = softmax(Qᵢ·Kᵀ · scale + causal_mask) · V, GQA 32:4, head_dim 512.
// K and V are SEPARATE stores (M3 amendment: cached K is k_norm-weighted +
// roped, cached V is weightless-normed and unroped —
// docs/reference/gemma4-forward-graph.md). Query token at row m0+r sits at
// key index q0 + m0 + r and attends keys 0 ..= that.
//
// WHY TWO-PASS (and not classic flash with a running rescale): the target
// has VK_KHR_cooperative_matrix (coopmat1), NOT NV_coopmat2 — there is no
// per-element fragment access, only matmul + coopLoad/coopStore with f16
// operands. Flash's per-tile rescale O *= exp(m_old−m_new) would therefore
// need a diag(corr)·O matmul, which forces a C→f16→B roundtrip through LDS
// (barriers every key tile + progressive f16 loss in the accumulator). The
// LDS-resident-O v1 that paid that cost in scalar LDS was 2–3× SLOWER than
// the naive kernel (occupancy 1 + barrier-bound rescale). Instead:
//   Pass 1: row max m[r] over ALL visible keys (coopmat QKᵀ → LDS → reduce).
//   Pass 2: m fixed, so P = exp(S·scale − m) needs NO correction; O and l
//           accumulate straight across key tiles. O lives in f32 C fragments
//           (register-resident, exact) and accumulates exactly like gemm's
//           acc — no roundtrip, no rescale, no drift.
// Cost: QKᵀ computed twice (recompute is cheaper than storing M×L scores or
// the rescale machinery). PV once.
//
// One wave-sized workgroup per (query head, M_Q-row query tile); grid
// [N_Q_HEADS, ceil(m_pad / M_Q)]. Matmuls map onto the gemm_q4_0 coopmat
// conventions (coopLoadT = load row-major as-is, coopLoad = load transpose):
//   S = Q·Kᵀ : A = Q (coopLoadT), B = Kᵀ (coopLoad of row-major K).
//   O = P·V  : A = P (coopLoadT from LDS), B = V (coopLoadT — V is already
//              [key × head_dim] = B layout).
// Q/K/V are f16 in global memory, so operands load straight from the storage
// buffers with the KV/Q head stride as the coop stride — no dequant staging.
//
// Determinism: keys merge in fixed ascending order in both passes; coopmat
// reductions are hardware-tree deterministic → bit-exact rerun (cache-resume
// invariant). naga_oil corrupts coopmat IR, so this compiles via plain naga
// (`raw: true`); no `enable subgroups` (row reductions are single-thread-
// per-row loops). NOTE: a `var` coop-mat re-declared inside a loop is NOT
// re-zeroed per iteration (naga/ACO keeps the registers live) — `cs` is
// re-zeroed from an untouched `zero_c`; `o[]` is declared once and genuinely
// accumulates.

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> q: array<f16>;        // [M × N_Q_HEADS × HEAD_DIM]
@group(0) @binding(1) var<storage, read> k: array<f16>;        // [L × N_KV_HEADS × HEAD_DIM]
@group(0) @binding(2) var<storage, read> v: array<f16>;        // [L × N_KV_HEADS × HEAD_DIM]
@group(0) @binding(3) var<storage, read_write> out: array<f16>; // [M × N_Q_HEADS × HEAD_DIM]
// Per-step dynamic state (sg_gpu::StepState): [pos, kv_len_sliding,
// kv_len_global, q0]. q0 = number of history keys before this chunk.
@group(0) @binding(4) var<storage, read> step: array<u32>;

struct Push {
    scale: f32,
}
var<immediate> push: Push;

const HEAD_DIM: u32 = #{HEAD_DIM}u;
const N_KV_HEADS: u32 = #{N_KV_HEADS}u;
const Q_PER_KV: u32 = #{Q_PER_KV}u;
const M_Q: u32 = #{M_Q}u;            // query rows per tile (one 16-row C tile)
const N_K: u32 = #{N_K}u;            // key tile width (multiple of 16)
const WG: u32 = #{WG_X}u;

const N_Q_HEADS: u32 = N_KV_HEADS * Q_PER_KV;
const Q_ROW_STRIDE: u32 = N_Q_HEADS * HEAD_DIM;   // between query tokens
const KV_ROW_STRIDE: u32 = N_KV_HEADS * HEAD_DIM; // between keys/values
const S_TILES: u32 = HEAD_DIM / 16u;              // contraction substeps for QKᵀ
const N_K_TILES: u32 = N_K / 16u;                 // key substeps within a tile
const O_TILES: u32 = HEAD_DIM / 16u;              // output head-dim tiles
const NEG_INF: f32 = -3.0e38;

// Q·Kᵀ scores for the current key tile (coopStoreT target, then softmax).
var<workgroup> s_stage: array<f32, #{S_STAGE_LEN}>;   // M_Q × N_K
// Softmax probabilities (f16) feeding the PV matmul.
var<workgroup> p_stage: array<f16, #{S_STAGE_LEN}>;   // M_Q × N_K
// One output tile staged for the f32→f16 epilogue write (reused per tile).
var<workgroup> o_stage: array<f32, 256>;              // 16 × 16
// Per-row softmax state.
var<workgroup> m_row: array<f32, #{M_Q}>;
var<workgroup> l_row: array<f32, #{M_Q}>;

@compute @workgroup_size(#{WG_X})
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let lid = lid_v.x;
    let qh = wg_id.x;            // query head 0..N_Q_HEADS
    let kvh = qh / Q_PER_KV;     // shared KV head
    let m0 = wg_id.y * M_Q;      // first query row (token) of this tile
    let q0 = step[3];
    let q_base = (m0 * N_Q_HEADS + qh) * HEAD_DIM;
    let last_key = q0 + m0 + (M_Q - 1u); // causal tile bound
    let n_key_tiles = last_key / N_K + 1u;

    if (lid < M_Q) {
        m_row[lid] = NEG_INF;
        l_row[lid] = 0.0;
    }
    workgroupBarrier();

    // Re-zeroed each QKᵀ tile from this untouched accumulator (see header).
    var zero_c: coop_mat16x16<f32, C>;
    var cs: coop_mat16x16<f32, C>;

    // ── PASS 1: per-row max over all visible keys ────────────────────────
    for (var kt = 0u; kt < n_key_tiles; kt += 1u) {
        for (var nt = 0u; nt < N_K_TILES; nt += 1u) {
            cs = zero_c;
            let key_col0 = ((kt * N_K + nt * 16u) * N_KV_HEADS + kvh) * HEAD_DIM;
            for (var s = 0u; s < S_TILES; s += 1u) {
                let a = coopLoadT<coop_mat16x16<f16, A>>(&q[q_base + s * 16u], Q_ROW_STRIDE);
                let b = coopLoad<coop_mat16x16<f16, B>>(&k[key_col0 + s * 16u], KV_ROW_STRIDE);
                cs = coopMultiplyAdd(a, b, cs);
            }
            coopStoreT(cs, &s_stage[nt * 16u], N_K);
        }
        workgroupBarrier();
        if (lid < M_Q) {
            let qpos = q0 + m0 + lid;
            let row = lid * N_K;
            var rmax = m_row[lid];
            for (var j = 0u; j < N_K; j += 1u) {
                if (kt * N_K + j <= qpos) {
                    rmax = max(rmax, s_stage[row + j] * push.scale);
                }
            }
            m_row[lid] = rmax;
        }
        workgroupBarrier(); // before reusing s_stage next tile
    }

    // ── PASS 2: O = Σ P·V, l = Σ P with the fixed row max ────────────────
    var o: array<coop_mat16x16<f32, C>, O_TILES>;
    for (var ot = 0u; ot < O_TILES; ot += 1u) {
        o[ot] = zero_c;
    }

    for (var kt = 0u; kt < n_key_tiles; kt += 1u) {
        for (var nt = 0u; nt < N_K_TILES; nt += 1u) {
            cs = zero_c;
            let key_col0 = ((kt * N_K + nt * 16u) * N_KV_HEADS + kvh) * HEAD_DIM;
            for (var s = 0u; s < S_TILES; s += 1u) {
                let a = coopLoadT<coop_mat16x16<f16, A>>(&q[q_base + s * 16u], Q_ROW_STRIDE);
                let b = coopLoad<coop_mat16x16<f16, B>>(&k[key_col0 + s * 16u], KV_ROW_STRIDE);
                cs = coopMultiplyAdd(a, b, cs);
            }
            coopStoreT(cs, &s_stage[nt * 16u], N_K);
        }
        workgroupBarrier();
        if (lid < M_Q) {
            let qpos = q0 + m0 + lid;
            let row = lid * N_K;
            let m = m_row[lid];
            var lsum = l_row[lid];
            for (var j = 0u; j < N_K; j += 1u) {
                var p = 0.0;
                if (kt * N_K + j <= qpos) {
                    p = exp(s_stage[row + j] * push.scale - m);
                }
                p_stage[row + j] = f16(p);
                lsum += p;
            }
            l_row[lid] = lsum;
        }
        workgroupBarrier();
        // O += P·V (straight accumulation — m is fixed, no rescale).
        for (var ot = 0u; ot < O_TILES; ot += 1u) {
            for (var s2 = 0u; s2 < N_K_TILES; s2 += 1u) {
                let ap = coopLoadT<coop_mat16x16<f16, A>>(&p_stage[s2 * 16u], N_K);
                let v_key0 = ((kt * N_K + s2 * 16u) * N_KV_HEADS + kvh) * HEAD_DIM;
                let bv = coopLoadT<coop_mat16x16<f16, B>>(&v[v_key0 + ot * 16u], KV_ROW_STRIDE);
                o[ot] = coopMultiplyAdd(ap, bv, o[ot]);
            }
        }
        workgroupBarrier(); // before reusing s_stage/p_stage next tile
    }

    // ── Epilogue: out = O / l, one tile at a time through o_stage ─────────
    for (var ot = 0u; ot < O_TILES; ot += 1u) {
        coopStoreT(o[ot], &o_stage[0], 16u);
        workgroupBarrier();
        for (var i = lid; i < M_Q * 16u; i += WG) {
            let r = i / 16u;
            let c = i % 16u;
            out[(m0 + r) * Q_ROW_STRIDE + qh * HEAD_DIM + ot * 16u + c] = f16(o_stage[i] / l_row[r]);
        }
        workgroupBarrier(); // before next tile overwrites o_stage
    }
}
