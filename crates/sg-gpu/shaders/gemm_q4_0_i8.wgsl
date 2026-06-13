// int8-MMQ GEMM, Q4_0 weights × Q8_0 activations (plan 02, profile rank #2).
// Y[M×N] = X[M×K] · W[N×K]ᵀ via SIGNED int8 cooperative matrices (enabled by
// the naga fork — docs/naga-int8-coopmat-patch.md). The MMQ alternative to
// the f16 `gemm_q4_0`: weights stay int8 (no dequant), activations are
// quantized to Q8_0, and each 32-element block's dot is an EXACT i8×i8→i32
// cooperative-matrix product, rescaled to f32 by the block's `d_a·d_w`.
//
// One wave-sized workgroup computes an (M_TILES·16)×(N_TILES·16) block of Y.
// PER-BLOCK RESCALE (the crux): the (d_a[m]·d_w[n]) scale is per-32-block AND
// an outer product over the tile, so a single i32 accumulation across all K is
// impossible — every block's i32 dot must be pulled out, converted to f32 and
// scaled before it can be summed into Y.
//
// IN-REGISTER RESCALE (the coopmat-arith fork — docs/naga-coopmat-arith-patch.md
// — unblocks this): the f32 output `yacc` is register-resident across all NB
// blocks (a `coop_mat<f32,C>` array, zeroed once). Per 32-block β:
//   - the ACC = M_TILES·N_TILES i32 accumulators get the block's dot via 2
//     substeps of 16 (a Q4_0 block is 32 = low-nibble half then high-nibble
//     half, in the logical order the reference pins);
//   - the rank-1 scale `d_a[m]·d_w[n]` is materialized into a small LDS buffer
//     (`stage`), then loaded per tile as a `coop_mat<f32,C>` and applied with
//     component-wise ops: `yacc += scale * f32(acc)` (OpConvertSToF on the i32
//     dot, OpFMul for the scale, OpFAdd to accumulate — all per-lane, no LDS
//     round-trip of the dot, no per-element scalar loop).
// vs the prior version this removes the per-block i32 `coopStore` (8 KB at 2×4)
// and the per-element f32 rescale-into-LDS loop.
//
// Three ISA-guided tunings got 2×4 from 2.52 → 7.85 TFLOPS / 0.83× f16 (RADV
// asm + bench: mmq_tflops, perf=high; STATUS 2026-06-14):
//   1. SCALE BUILD: build the row/col scale VECTORS in LDS (`da_l`, `dw_l`) and
//      form `stage` = their outer product from LDS — vs reading the full product
//      from global, ~64 redundant f16 loads + ~340 address ops/thread/block
//      (tiling-invariant → naive sat ~2.5 at every tile size). [2.52→4.80]
//   2. β-LOOP UNROLLED ×2: two independent blocks' MMAs interleaved so WMMA ILP
//      covers the per-block substep0→substep1 stall — RDNA3 hides WMMA latency
//      via within-wave ILP, not occupancy (more waves did NOT help). [4.80→7.22]
//   3. The outer-product fill is ×4-unrolled so independent `da_l` loads pipeline
//      instead of one load→lgkmcnt(0)→mul→store per element. [7.22→7.85]
// The rescale (ACC `coopLoad`s + per-lane convert/FMul/FAdd) was never the cost;
// feeding it the scale was. `stage` is DOUBLE-BUFFERED (halves 0/1 = the two
// unrolled blocks); two barriers/block frame da_l/dw_l→stage→coopLoad, and the
// single da_l/dw_l buffer is safe to reuse (the stage→coopLoad barrier orders the
// next block's overwrite). Occupancy is not the binder (no spill through 4×4);
// the residual binder is the scale's LDS round-trip, forced by coopmat1 having no
// fragment-element access (can't scale the accumulator in registers).
//
// Element layout: `coopMultiplyAdd(A=Xᵀ-loaded, B=Wᵀ)` gives `acc[i][j]` = the
// dot for output row (tile_m+j), col (tile_n+i) — the transpose of the output
// tile (derived from the prior kernel's parity-green coopStoreT path). So the
// scale, laid out [M_ROWS×N_COLS] row-major as `d_a[m]·d_w[n]` and loaded with
// `coopLoadT`, lines up element-for-element with `acc`; the epilogue
// `coopStoreT`s `yacc` back to that same layout to write y row-major.
//
// W is i8 in global → coopLoad reads B=Wᵀ directly (no dequant staging, unlike
// f16 gemm). A=X (coopLoadT), B=Wᵀ (coopLoad of row-major W). Operand i8 data
// must live in `array<i8>` buffers (VK_KHR_8bit_storage; the <i8> generic on
// coopLoad is ignored — the scalar comes from the pointer base type).
//
// naga_oil corrupts coopmat IR → raw: true. A `var` coop-mat re-declared in a
// loop is NOT re-zeroed (naga/ACO keeps the registers live) → `acc[]`/`yacc[]`
// are declared once and explicitly seeded from `zero_i`/`zero_f`.

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> w: array<i8>;          // [N×K] Q4_0 quants (q−8)
@group(0) @binding(1) var<storage, read> w_scales: array<f16>;  // [N×(K/32)] d_w
@group(0) @binding(2) var<storage, read> x: array<i8>;          // [M×K] Q8_0 quants
@group(0) @binding(3) var<storage, read> x_scales: array<f16>;  // [M×(K/32)] d_a
@group(0) @binding(4) var<storage, read_write> y: array<f16>;   // [M×N]

const K: u32 = #{K_DIM}u;
const N: u32 = #{N_DIM}u;
const WG: u32 = #{WG_X}u;
const M_TILES: u32 = #{M_TILES}u;
const N_TILES: u32 = #{N_TILES}u;

const NB: u32 = K / 32u;            // 32-blocks per row
const ACC: u32 = M_TILES * N_TILES; // 16×16 output tiles per workgroup
const M_ROWS: u32 = M_TILES * 16u;
const N_COLS: u32 = N_TILES * 16u;
const TILE_ELEMS: u32 = ACC * 256u;

// The per-block outer-product scale d_a[m]·d_w[n], [M_ROWS×N_COLS] row-major,
// DOUBLE-BUFFERED (two halves of TILE_ELEMS, selected by β&1). Buffer 0 is also
// the f32→f16 epilogue staging scratch.
var<workgroup> stage: array<f32, 2u * TILE_ELEMS>;
// The block's row/col scale VECTORS (M_ROWS + N_COLS values), loaded from global
// once per block; `stage` is then the LDS-only outer product of these. Loading
// the full outer product straight from global instead cost ~64 redundant f16
// global loads + ~340 address-arith ops per thread per block (RADV asm) — the
// dominant term, swamping the 16 MMAs (ISA-evidenced; STATUS 2026-06-14).
var<workgroup> da_l: array<f32, M_ROWS>;
var<workgroup> dw_l: array<f32, N_COLS>;

@compute @workgroup_size(#{WG_X})
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let lid = lid_v.x;
    let m0 = wg.y * M_ROWS; // output row block base
    let n0 = wg.x * N_COLS; // output col block base

    var zero_i: coop_mat16x16<i32, C>;
    var zero_f: coop_mat16x16<f32, C>;
    var acc: array<coop_mat16x16<i32, C>, ACC>;
    var acc2: array<coop_mat16x16<i32, C>, ACC>; // 2nd block (β-loop unrolled ×2)
    var yacc: array<coop_mat16x16<f32, C>, ACC>; // register-resident output
    for (var i = 0u; i < ACC; i += 1u) {
        yacc[i] = zero_f;
    }

    // β-loop UNROLLED ×2 (NB even on every shape). Two independent blocks' MMAs
    // are interleaved so the WMMA ILP covers the per-block substep0→substep1
    // dependency stall — RDNA3 hides WMMA latency via within-wave ILP, not
    // wave-switching (the bare-MMA kernel jumped 1.1→3.3 TFLOPS under the same
    // unroll; RADV 2026-06-14). The two blocks rescale into the two `stage`
    // halves (0 / 1), so the double buffer is now consumed within one iteration.
    for (var beta = 0u; beta < NB; beta += 2u) {
        for (var i = 0u; i < ACC; i += 1u) {
            acc[i] = zero_i;
            acc2[i] = zero_i;
        }
        let kb0 = beta * 32u;
        let kb1 = kb0 + 32u;
        for (var s = 0u; s < 2u; s += 1u) {
            var b0: array<coop_mat16x16<i8, B>, N_TILES>;
            var b1: array<coop_mat16x16<i8, B>, N_TILES>;
            for (var nt = 0u; nt < N_TILES; nt += 1u) {
                b0[nt] = coopLoad<coop_mat16x16<i8, B>>(&w[(n0 + nt * 16u) * K + kb0 + s * 16u], K);
                b1[nt] = coopLoad<coop_mat16x16<i8, B>>(&w[(n0 + nt * 16u) * K + kb1 + s * 16u], K);
            }
            for (var mt = 0u; mt < M_TILES; mt += 1u) {
                let a0 = coopLoadT<coop_mat16x16<i8, A>>(&x[(m0 + mt * 16u) * K + kb0 + s * 16u], K);
                let a1 = coopLoadT<coop_mat16x16<i8, A>>(&x[(m0 + mt * 16u) * K + kb1 + s * 16u], K);
                for (var nt = 0u; nt < N_TILES; nt += 1u) {
                    acc[mt * N_TILES + nt] = coopMultiplyAdd(a0, b0[nt], acc[mt * N_TILES + nt]);
                    acc2[mt * N_TILES + nt] = coopMultiplyAdd(a1, b1[nt], acc2[mt * N_TILES + nt]);
                }
            }
        }
        // Rescale both blocks into yacc. Each `u` is one block: load its row/col
        // scale vectors, form the outer product in its `stage` half (u = the
        // half index since β is even), then yacc += scale · f32(dot).
        for (var u = 0u; u < 2u; u += 1u) {
            let bb = beta + u;
            for (var i = lid; i < M_ROWS; i += WG) {
                da_l[i] = f32(x_scales[(m0 + i) * NB + bb]);
            }
            for (var i = lid; i < N_COLS; i += WG) {
                dw_l[i] = f32(w_scales[(n0 + i) * NB + bb]);
            }
            workgroupBarrier(); // da_l/dw_l written before the outer product reads them
            let base = u * TILE_ELEMS;
            let n = lid % N_COLS;
            let dw = dw_l[n];
            // Walk the column ×4-unrolled: issue four independent `da_l` loads
            // before the multiplies so the LDS latency pipelines, instead of one
            // load→lgkmcnt(0)→mul→store per element (ACO won't batch it itself;
            // RADV asm 2026-06-14). M_ROWS/step is a multiple of 4 every variant.
            let step = WG / N_COLS;
            for (var m = lid / N_COLS; m < M_ROWS; m += 4u * step) {
                let d0 = da_l[m];
                let d1 = da_l[m + step];
                let d2 = da_l[m + 2u * step];
                let d3 = da_l[m + 3u * step];
                stage[base + m * N_COLS + n] = d0 * dw;
                stage[base + (m + step) * N_COLS + n] = d1 * dw;
                stage[base + (m + 2u * step) * N_COLS + n] = d2 * dw;
                stage[base + (m + 3u * step) * N_COLS + n] = d3 * dw;
            }
            workgroupBarrier(); // stage written before coopLoad
            for (var mt = 0u; mt < M_TILES; mt += 1u) {
                for (var nt = 0u; nt < N_TILES; nt += 1u) {
                    let t = mt * N_TILES + nt;
                    let scale = coopLoadT<coop_mat16x16<f32, C>>(
                        &stage[base + (mt * 16u) * N_COLS + nt * 16u], N_COLS);
                    if (u == 0u) {
                        yacc[t] = yacc[t] + scale * f32(acc[t]);
                    } else {
                        yacc[t] = yacc[t] + scale * f32(acc2[t]);
                    }
                }
            }
        }
    }

    // Epilogue: convert yacc (f32) to f16 IN-REGISTER and coopStore straight to
    // y — no LDS staging, no scalar convert loop. Tests the fork's OpFConvert on
    // a coopmat (f16(coop<f32>)) + coopStore of a C-use f16 matrix to array<f16>.
    for (var mt = 0u; mt < M_TILES; mt += 1u) {
        for (var nt = 0u; nt < N_TILES; nt += 1u) {
            coopStoreT(
                f16(yacc[mt * N_TILES + nt]),
                &y[(m0 + mt * 16u) * N + n0 + nt * 16u], N);
        }
    }
}
