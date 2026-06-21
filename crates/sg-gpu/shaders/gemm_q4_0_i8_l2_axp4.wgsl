// Static-unroll ping-pong activation prefetch — dedicated variant (FFN-down shape
// K=21504 N=5376, 1D dispatch, BN_SB=4, β×2). Derived from gemm_q4_0_i8_l2.
//
// Goal: hide the X-before-WMMA `vmcnt` stall by prefetching each K-iter's
// activations a FULL iteration ahead, double-buffered — WITHOUT the two failure
// modes the l2 AXP=2/3 probes hit:
//   - AXP=2 (single buffer, consume-then-reload): ACO rotates the reloaded frags
//     back into the array's home registers via `v_swap_b32`, dragging the reload
//     `vmcnt` onto the critical path.
//   - AXP=3 (double buffer, β-parity via UNIFORM DYNAMIC INDEX): registers aren't
//     addressable, so ACO materialized select-chains for every indexed frag
//     read/write → 8× the instruction count (4807 vs ~600), −43%.
// Fix: STATIC unroll — process TWO β-pairs per loop iteration with two EXPLICITLY
// NAMED buffers A and B that swap current/next roles between the two sub-iters.
// Every buffer index is then a compile-time constant: no dynamic index, no select
// chains, no swaps. The cost is the inherent 2× activation-fragment VGPR (both
// buffers live), at deployed-equal occupancy (~3/16) — the open question is whether
// fully hiding X at that occupancy beats the cheap within-iter hoist (axp, +3.7%).

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> weights: array<u32>;   // [N×K] Q4_0 packed
@group(0) @binding(1) var<storage, read> x: array<i8>;          // [M×K] Q8 quants
@group(0) @binding(2) var<storage, read> x_scales: array<f16>;  // [M×(K/32)] d_a
@group(0) @binding(3) var<storage, read_write> y: array<f16>;   // [M×N]

const K: u32 = 21504u;
const N: u32 = 5376u;
const M_TILES: u32 = 4u;
const N_COLS: u32 = 16u;
const M_ROWS: u32 = M_TILES * 16u;    // 64
const M_TOTAL: u32 = 256u;
const NB: u32 = K / 32u;              // 672 32-blocks per row (divisible by 4 → step-4 clean)
const ROW_WORDS: u32 = NB * 18u / 4u; // 3024 Q4_0 words per W row
const NB_N: u32 = N / N_COLS;         // 336
const NB_M: u32 = M_TOTAL / M_ROWS;   // 4
const BN_SB: u32 = 4u;                // L2 super-block (the best schedule)

// 2D super-block walk, m-OUTER (see gemm_q4_0_i8_l2 for the rationale).
fn tile_index(t: u32) -> vec2<u32> {
    let per_sb = BN_SB * NB_M;
    let sb = t / per_sb;
    let in_sb = t % per_sb;
    let m_block = in_sb / BN_SB;
    let n_block = sb * BN_SB + in_sb % BN_SB;
    return vec2(n_block, m_block);
}

var<workgroup> wb: array<i8, N_COLS * 64u>;
var<workgroup> dw2: array<f32, 2u * N_COLS>;
var<workgroup> da_l: array<f32, M_ROWS>;

fn unpack_block(q0: u32, q1: u32, q2: u32, q3: u32, base: u32) {
    var qs = array<u32, 4>(q0, q1, q2, q3);
    for (var w = 0u; w < 4u; w += 1u) {
        let word = qs[w];
        for (var b = 0u; b < 4u; b += 1u) {
            let byte = (word >> (8u * b)) & 0xFFu;
            let j = 4u * w + b;
            wb[base + j] = i8(i32(byte & 0xFu) - 8);
            wb[base + 16u + j] = i8(i32(byte >> 4u) - 8);
        }
    }
}

// Inline weight load + dequant of one block-pair (β) into wb/dw2 (16 lanes).
fn load_dequant(beta: u32, n0: u32, lid: u32) {
    if (lid < N_COLS) {
        let wp = (n0 + lid) * ROW_WORDS + (beta / 2u) * 9u;
        var w: array<u32, 9>;
        for (var i = 0u; i < 9u; i += 1u) {
            w[i] = weights[wp + i];
        }
        dw2[lid] = unpack2x16float(w[0]).x;
        unpack_block(
            (w[0] >> 16u) | (w[1] << 16u),
            (w[1] >> 16u) | (w[2] << 16u),
            (w[2] >> 16u) | (w[3] << 16u),
            (w[3] >> 16u) | (w[4] << 16u),
            lid * 64u,
        );
        dw2[N_COLS + lid] = unpack2x16float(w[4]).y;
        unpack_block(w[5], w[6], w[7], w[8], lid * 64u + 32u);
    }
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let lid = lid_v.x;
    let t = wg.y * NB_N + wg.x;
    let tile = tile_index(t);
    let m0 = tile.y * M_ROWS;
    let n0 = tile.x * N_COLS;

    var zero_f: coop_mat16x16<f32, C>;
    var zero_i: coop_mat16x16<i32, C>;
    var yacc: array<coop_mat16x16<f32, C>, M_TILES>;
    for (var i = 0u; i < M_TILES; i += 1u) {
        yacc[i] = zero_f;
    }

    // Two explicitly-named activation buffers; A is current for the first sub-iter,
    // B for the second. Prologue loads A with β=0's X.
    var xa0: array<coop_mat16x16<i8, A>, 2u * M_TILES>;
    var xa1: array<coop_mat16x16<i8, A>, 2u * M_TILES>;
    var xb0: array<coop_mat16x16<i8, A>, 2u * M_TILES>;
    var xb1: array<coop_mat16x16<i8, A>, 2u * M_TILES>;
    for (var s = 0u; s < 2u; s += 1u) {
        for (var mt = 0u; mt < M_TILES; mt += 1u) {
            xa0[s * M_TILES + mt] = coopLoadT<coop_mat16x16<i8, A>>(&x[(m0 + mt * 16u) * K + s * 16u], K);
            xa1[s * M_TILES + mt] = coopLoadT<coop_mat16x16<i8, A>>(&x[(m0 + mt * 16u) * K + 32u + s * 16u], K);
        }
    }

    var acc0: array<coop_mat16x16<i32, C>, M_TILES>;
    var acc1: array<coop_mat16x16<i32, C>, M_TILES>;

    for (var beta = 0u; beta < NB; beta += 4u) {
        // ===================== sub-iter 1: current = A, prefetch β+2 → B =========
        load_dequant(beta, n0, lid);
        workgroupBarrier();
        // Prefetch β+2's X into B (in flight through the MMA + rescale below).
        let k1 = min((beta + 2u) * 32u, (NB - 2u) * 32u);
        for (var s = 0u; s < 2u; s += 1u) {
            for (var mt = 0u; mt < M_TILES; mt += 1u) {
                xb0[s * M_TILES + mt] = coopLoadT<coop_mat16x16<i8, A>>(&x[(m0 + mt * 16u) * K + k1 + s * 16u], K);
                xb1[s * M_TILES + mt] = coopLoadT<coop_mat16x16<i8, A>>(&x[(m0 + mt * 16u) * K + k1 + 32u + s * 16u], K);
            }
        }
        for (var mt = 0u; mt < M_TILES; mt += 1u) {
            acc0[mt] = zero_i;
            acc1[mt] = zero_i;
        }
        for (var s = 0u; s < 2u; s += 1u) {
            let bt0 = coopLoad<coop_mat16x16<i8, B>>(&wb[s * 16u], 64u);
            let bt1 = coopLoad<coop_mat16x16<i8, B>>(&wb[32u + s * 16u], 64u);
            for (var mt = 0u; mt < M_TILES; mt += 1u) {
                acc0[mt] = coopMultiplyAdd(xa0[s * M_TILES + mt], bt0, acc0[mt]);
                acc1[mt] = coopMultiplyAdd(xa1[s * M_TILES + mt], bt1, acc1[mt]);
            }
        }
        for (var u = 0u; u < 2u; u += 1u) {
            for (var i = lid; i < M_ROWS; i += 64u) {
                da_l[i] = f32(x_scales[(m0 + i) * NB + (beta + u)]);
            }
            workgroupBarrier();
            for (var mt = 0u; mt < M_TILES; mt += 1u) {
                let scale = coopLoad<coop_mat16x16<f32, C>>(&da_l[mt * 16u], 0u)
                    * coopLoadT<coop_mat16x16<f32, C>>(&dw2[u * N_COLS], 0u);
                if (u == 0u) {
                    yacc[mt] = yacc[mt] + scale * f32(acc0[mt]);
                } else {
                    yacc[mt] = yacc[mt] + scale * f32(acc1[mt]);
                }
            }
            workgroupBarrier();
        }

        // ===================== sub-iter 2: current = B (β+2), prefetch β+4 → A ====
        load_dequant(beta + 2u, n0, lid);
        workgroupBarrier();
        let k2 = min((beta + 4u) * 32u, (NB - 2u) * 32u);
        for (var s = 0u; s < 2u; s += 1u) {
            for (var mt = 0u; mt < M_TILES; mt += 1u) {
                xa0[s * M_TILES + mt] = coopLoadT<coop_mat16x16<i8, A>>(&x[(m0 + mt * 16u) * K + k2 + s * 16u], K);
                xa1[s * M_TILES + mt] = coopLoadT<coop_mat16x16<i8, A>>(&x[(m0 + mt * 16u) * K + k2 + 32u + s * 16u], K);
            }
        }
        for (var mt = 0u; mt < M_TILES; mt += 1u) {
            acc0[mt] = zero_i;
            acc1[mt] = zero_i;
        }
        for (var s = 0u; s < 2u; s += 1u) {
            let bt0 = coopLoad<coop_mat16x16<i8, B>>(&wb[s * 16u], 64u);
            let bt1 = coopLoad<coop_mat16x16<i8, B>>(&wb[32u + s * 16u], 64u);
            for (var mt = 0u; mt < M_TILES; mt += 1u) {
                acc0[mt] = coopMultiplyAdd(xb0[s * M_TILES + mt], bt0, acc0[mt]);
                acc1[mt] = coopMultiplyAdd(xb1[s * M_TILES + mt], bt1, acc1[mt]);
            }
        }
        for (var u = 0u; u < 2u; u += 1u) {
            for (var i = lid; i < M_ROWS; i += 64u) {
                da_l[i] = f32(x_scales[(m0 + i) * NB + (beta + 2u + u)]);
            }
            workgroupBarrier();
            for (var mt = 0u; mt < M_TILES; mt += 1u) {
                let scale = coopLoad<coop_mat16x16<f32, C>>(&da_l[mt * 16u], 0u)
                    * coopLoadT<coop_mat16x16<f32, C>>(&dw2[u * N_COLS], 0u);
                if (u == 0u) {
                    yacc[mt] = yacc[mt] + scale * f32(acc0[mt]);
                } else {
                    yacc[mt] = yacc[mt] + scale * f32(acc1[mt]);
                }
            }
            workgroupBarrier();
        }
    }

    for (var mt = 0u; mt < M_TILES; mt += 1u) {
        coopStoreT(f16(yacc[mt]), &y[(m0 + mt * 16u) * N + n0], N);
    }
}
