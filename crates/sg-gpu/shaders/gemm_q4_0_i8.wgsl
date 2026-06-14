// int8-MMQ GEMM, Q4_0 weights × Q8_0 activations (plan 02, profile rank #2).
// Y[M×N] = X[M×K] · W[N×K]ᵀ via SIGNED int8 cooperative matrices (enabled by
// the naga fork — docs/naga-int8-coopmat-patch.md). The MMQ alternative to
// the f16 `gemm_q4_0`: each 32-element block's dot is an EXACT i8×i8→i32
// cooperative-matrix product, rescaled to f32 by the block's `d_a·d_w`.
//
// WEIGHTS read Q4_0 PACKED, like the f16 gemm (apples-to-apples — and the
// deployable form: no weight repack, no 2× memory). Each thread bulk-loads its
// W row's aligned 9-word block PAIR (= 64 weights = the β-unroll's two blocks),
// unpacks the nibbles to i8 (q−8) in an LDS tile (`wb`) and writes the two
// per-block `d_w` scales — folding away the separate w_scales buffer. The MMA
// then `coopLoad`s the int8 B tiles straight from `wb` (LDS i8 coopmat verified
// by coop_i8_lds_smoke). Activations stay pre-quantized Q8_0 (a separate
// `kv_quant_q8` pass over X — small and amortized across a layer's gemms).
//
// One wave-sized workgroup computes an (M_TILES·16)×(N_TILES·16) block of Y.
// PER-BLOCK RESCALE (the crux): the (d_a[m]·d_w[n]) scale is per-32-block AND
// an outer product over the tile, so a single i32 accumulation across all K is
// impossible — every block's i32 dot must be pulled out, converted to f32 and
// scaled before it can be summed into Y. The f32 output `yacc` is
// register-resident across all blocks (a `coop_mat<f32,C>` array, zeroed once);
// the scale is built in LDS and applied with component-wise coopmat ops
// (`yacc += scale * f32(acc)`: OpConvertSToF + OpFMul + OpFAdd, per-lane).
//
// ISA-guided tuning + reading Q4_0 directly: ~11 TFLOPS at 2×2 ≈ 1.03× f16
// (bench: mmq_tflops, perf=high; STATUS 2026-06-14). 2×2 is the production
// tiling (the i8 weight-staging shifted the sweet spot off 2×4). The steps,
// each from RADV asm/shaderstats:
//   1. SCALE BUILD: build the row/col scale VECTORS in LDS (`da_l`, `dw{a,b}`)
//      and form `stage` = their outer product from LDS — not the full product
//      from global (which was ~64 redundant loads + ~340 address ops/thread).
//   2. β-LOOP UNROLLED ×2: two independent blocks' MMAs interleaved so WMMA ILP
//      covers the per-block substep0→substep1 stall — RDNA3 hides WMMA latency
//      via within-wave ILP, not occupancy.
//   3. The outer-product fill is ×4-unrolled so independent `da_l` loads
//      pipeline instead of one load→lgkmcnt(0)→mul→store per element.
// `stage` is double- or single-buffered per `STAGE_BUFS` (double overlaps the
// rescale on small-K shapes; single frees LDS for occupancy on large-K). The
// residual binder is the scale's LDS round-trip, forced by coopmat1 having no
// fragment-element access (can't scale the accumulator in registers).
//
// Element layout: `coopMultiplyAdd(A=Xᵀ-loaded, B=Wᵀ)` gives `acc[i][j]` = the
// dot for output row (tile_m+j), col (tile_n+i) — the transpose of the output
// tile (derived from the prior kernel's parity-green coopStoreT path). So the
// scale, laid out [M_ROWS×N_COLS] row-major as `d_a[m]·d_w[n]` and loaded with
// `coopLoadT`, lines up element-for-element with `acc`; the epilogue stages
// `yacc` to LDS and writes y via normal stores (coopStore to y is invisible to
// auto-sync — see the epilogue).
//
// naga_oil corrupts coopmat IR → raw: true. A `var` coop-mat re-declared in a
// loop is NOT re-zeroed (naga/ACO keeps the registers live) → `acc[]`/`yacc[]`
// are declared once and explicitly seeded from `zero_i`/`zero_f`.

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> weights: array<u32>;   // [N×K] Q4_0 packed (18 B/block)
@group(0) @binding(1) var<storage, read> x: array<i8>;          // [M×K] Q8_0 quants
@group(0) @binding(2) var<storage, read> x_scales: array<f16>;  // [M×(K/32)] d_a
@group(0) @binding(3) var<storage, read_write> y: array<f16>;   // [M×N]

const K: u32 = #{K_DIM}u;
const N: u32 = #{N_DIM}u;
const WG: u32 = #{WG_X}u;
const M_TILES: u32 = #{M_TILES}u;
const N_TILES: u32 = #{N_TILES}u;
// `stage` buffering: 2 = double (one pass's coopLoad overlaps the next pass's
// fill — wins on small-K rescale-bound shapes); 1 = single (halves the LDS →
// more waves, wins on large-K shapes that are vmcnt-stalled with LDS-capped
// occupancy — RGP 2026-06-14). Identical barriers either way.
const STAGE_BUFS: u32 = #{STAGE_BUFS}u;

const NB: u32 = K / 32u;            // 32-blocks per row
const ACC: u32 = M_TILES * N_TILES; // 16×16 output tiles per workgroup
const M_ROWS: u32 = M_TILES * 16u;
const N_COLS: u32 = N_TILES * 16u;
const TILE_ELEMS: u32 = ACC * 256u;
const ROW_WORDS: u32 = NB * 18u / 4u; // Q4_0 words per W row (NB even → exact)

// Unpacked int8 weights for the current 64-K block pair: [N_COLS rows × 64 K]
// row-major (cols 0..32 = block β, 32..64 = block β+1).
var<workgroup> wb: array<i8, N_COLS * 64u>;
// The two blocks' per-row d_w scales, extracted from the Q4_0 blocks.
var<workgroup> dwa: array<f32, N_COLS>;
var<workgroup> dwb: array<f32, N_COLS>;
// The block's row scales d_a, and the outer-product scale d_a[m]·d_w[n]
// (STAGE_BUFS halves; buffer 0 also the f32→f16 epilogue scratch).
var<workgroup> da_l: array<f32, M_ROWS>;
var<workgroup> stage: array<f32, STAGE_BUFS * TILE_ELEMS>;

// Unpack one Q4_0 block (4 qs words in registers) to i8 (q−8) at wb[base..+32].
// Low nibble of byte j → position j, high nibble → position 16+j (the
// BlockQ4_0 logical order the reference pins).
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

    // β-loop unrolled ×2 = one Q4_0 block PAIR (64 K) per iteration.
    for (var beta = 0u; beta < NB; beta += 2u) {
        for (var i = 0u; i < ACC; i += 1u) {
            acc[i] = zero_i;
            acc2[i] = zero_i;
        }
        // Each thread (one W row) bulk-loads + unpacks its aligned 9-word block
        // pair into `wb`, and writes the two block scales.
        if (lid < N_COLS) {
            let wbase = (n0 + lid) * ROW_WORDS + (beta / 2u) * 9u;
            let w0 = weights[wbase];
            let w1 = weights[wbase + 1u];
            let w2 = weights[wbase + 2u];
            let w3 = weights[wbase + 3u];
            let w4 = weights[wbase + 4u];
            let w5 = weights[wbase + 5u];
            let w6 = weights[wbase + 6u];
            let w7 = weights[wbase + 7u];
            let w8 = weights[wbase + 8u];
            // Block β: d in w0.lo, 16 qs bytes spanning w0.hi..w4.lo.
            dwa[lid] = unpack2x16float(w0).x;
            unpack_block(
                (w0 >> 16u) | (w1 << 16u),
                (w1 >> 16u) | (w2 << 16u),
                (w2 >> 16u) | (w3 << 16u),
                (w3 >> 16u) | (w4 << 16u),
                lid * 64u,
            );
            // Block β+1: d in w4.hi, qs in w5..w8.
            dwb[lid] = unpack2x16float(w4).y;
            unpack_block(w5, w6, w7, w8, lid * 64u + 32u);
        }
        workgroupBarrier(); // wb + dwa/dwb written before MMA / rescale read them

        // Interleaved MMA for the two blocks (independent → WMMA ILP). B tiles
        // come from `wb` (LDS i8); A tiles from X (global i8). Block β occupies
        // wb cols 0..32, block β+1 cols 32..64; row stride is 64.
        let k0 = beta * 32u;
        for (var s = 0u; s < 2u; s += 1u) {
            var b0: array<coop_mat16x16<i8, B>, N_TILES>;
            var b1: array<coop_mat16x16<i8, B>, N_TILES>;
            for (var nt = 0u; nt < N_TILES; nt += 1u) {
                b0[nt] = coopLoad<coop_mat16x16<i8, B>>(&wb[(nt * 16u) * 64u + s * 16u], 64u);
                b1[nt] = coopLoad<coop_mat16x16<i8, B>>(&wb[(nt * 16u) * 64u + 32u + s * 16u], 64u);
            }
            for (var mt = 0u; mt < M_TILES; mt += 1u) {
                let a0 = coopLoadT<coop_mat16x16<i8, A>>(&x[(m0 + mt * 16u) * K + k0 + s * 16u], K);
                let a1 = coopLoadT<coop_mat16x16<i8, A>>(&x[(m0 + mt * 16u) * K + k0 + 32u + s * 16u], K);
                for (var nt = 0u; nt < N_TILES; nt += 1u) {
                    acc[mt * N_TILES + nt] = coopMultiplyAdd(a0, b0[nt], acc[mt * N_TILES + nt]);
                    acc2[mt * N_TILES + nt] = coopMultiplyAdd(a1, b1[nt], acc2[mt * N_TILES + nt]);
                }
            }
        }

        // Rescale both blocks into yacc. Each `u` is one block: load its row
        // scales `d_a`, form the outer product with the block's `d_w` (dwa/dwb)
        // in its `stage` buffer (`base`), then yacc += scale · f32(dot). At
        // STAGE_BUFS=2 the two passes use different halves (overlap); at 1 they
        // share one (the next pass's `da_l` barrier already orders this pass's
        // coopLoad before the overwrite, so no extra barrier either way).
        for (var u = 0u; u < 2u; u += 1u) {
            let bb = beta + u;
            for (var i = lid; i < M_ROWS; i += WG) {
                da_l[i] = f32(x_scales[(m0 + i) * NB + bb]);
            }
            // da_l written; at STAGE_BUFS=1 also orders the previous pass's stage
            // coopLoad before this pass overwrites `stage`.
            workgroupBarrier();
            let base = (u % STAGE_BUFS) * TILE_ELEMS;
            let n = lid % N_COLS;
            let dw = select(dwb[n], dwa[n], u == 0u);
            // ×4-unrolled column walk so independent da_l loads pipeline.
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

    // Epilogue: coopStore yacc (f32) to LDS, then write y via NORMAL stores.
    // coopStore straight to y is INVISIBLE to vulkano's reflection-based
    // auto-sync (like coopLoad — touch.wgsl), so the recorded prefill graph
    // would race the consumer (geglu/rms). The LDS round-trip keeps the y write
    // reflection-visible, exactly as the f16 gemm does. `stage` (buffer 0) is
    // free here — its last loop use was the rescale's coopLoad.
    workgroupBarrier(); // last rescale's stage coopLoad done before reuse
    for (var t = 0u; t < ACC; t += 1u) {
        coopStoreT(yacc[t], &stage[t * 256u], 16u);
    }
    workgroupBarrier();
    for (var i = lid; i < TILE_ELEMS; i += WG) {
        let t = i / 256u;
        let e = i % 256u;
        let row = m0 + (t / N_TILES) * 16u + e / 16u;
        let col = n0 + (t % N_TILES) * 16u + e % 16u;
        y[row * N + col] = f16(stage[i]);
    }
}
