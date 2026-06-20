// ⚠ TESTED DEAD END (2026-06-21) — kept as a documented A/B baseline, NOT wired
// into any graph (like gemm_q4_0_i8_occ). The multi-wave occupancy idea is
// strictly dominated by the deployed single-wave `gemm_q4_0_i8`. Max-occupancy
// (RM=1, the `_b41` variant): ~9.2 TFLOPS vs deployed ~14.4 (down) / ~17.4 (up)
// (bench: mmq_tflops, perf=high). RGP: the high occupancy (12/16 vs the
// deployed's 3/16 waves/SIMD) THRASHES the cache — 4× resident wavefronts spread
// the working set, L2 hit% DECAYS over the run, and a real multi-wave `s_barrier`
// (free in the single-wave kernels) becomes the top stall. Lowering occupancy via
// register M-tiles (RM=2, the `_r2` variant) re-warms L2 and lifts to ~10.5/11.0,
// but is still well short and just converges toward the deployed design.
// THE finding that retired this whole line (STATUS 2026-06-21): the deployed GEMM
// is ALREADY ~86% bandwidth-bound (~206 GB/s, measured by RGP local-video-bytes ÷
// duration, gemv-calibrated). So occupancy and tile shape cannot help — the only
// lever is moving FEWER bytes (the activation re-reads across N-strips). See the
// [[memory-stall-pct-is-not-bandwidth-util]] memory.
//
// int8-MMQ GEMM, MULTI-WAVE occupancy variant. Y[M×N] = X[M×K]·W[N×K]ᵀ with
// SIGNED int8 cooperative matrices (naga fork — docs/naga-int8-coopmat-patch.md),
// Q4_0 packed weights × Q8_0 activations, per-32-block d_a·d_w rescale. Same
// math and operand formats as `gemm_q4_0_i8`, a DELIBERATELY DIFFERENT design.
//
// THE IDEA (a fresh take on "occupancy-maximized" after the 1×1 `_occ` dead end).
// Both existing kernels are SINGLE-WAVE (WG_X=64); this one tiles the output
// block ACROSS WAVES of one workgroup, sharing the unpacked weight strip in LDS.
// A workgroup of BM_TILES·BN_TILES waves computes a BM×BN output block; each wave
// owns an RM×1 column of 16×16 tiles (so BM = BM_TILES·RM·16, BN = BN_TILES·16).
// The BN weight rows are unpacked to i8 in LDS once per K-step and shared by all
// BM_TILES waves down the column → weight-reuse = BM, like the deployed 4×1, but
// spread over many small-footprint waves instead of one fat register-tiled wave.
//
// RM = register M-tiles per wave (the occupancy knob):
//   RM=1  → one 16×16 tile per wave: minimum footprint, MAXIMUM occupancy (the
//           `_b41` draft — RGP: 60 VGPR, 12 waves/SIMD, but L2-thrashed and
//           barrier-bound; high occupancy desyncs the waves and spreads the 2 MB
//           L2 working set, raising memory-latency stalls — STATUS rank #2).
//   RM≥2  → each wave does RM M-tiles, loading its B fragment ONCE and feeding it
//           to RM MMAs (in-wave ILP + register weight reuse) → ~RM× the VGPR →
//           ~half the occupancy at RM=2: the half-occupancy A/B against `_b41`
//           (same BM×BN block + reuse, fewer/fatter waves — does warmer L2 win?).
//
// SIMPLE BY CONSTRUCTION: NO software weight prefetch, NO β×2 MMA interleave, and
// the per-block scale is ALWAYS the 0-stride coopLoad broadcast
// (coopLoad(da,0)·coopLoadT(dw,0) = the d_a[m]·d_w[n] outer product —
// coop_bcast_lds_smoke) so there is no LDS `stage` outer-product fill. wb/dw/da
// hold BOTH blocks of the pair, so one barrier guards the whole pair's
// MMA+rescale and one closes the K-step.
//
// Element layout matches `gemm_q4_0_i8` exactly (parity-green there):
// coopMultiplyAdd(A=coopLoadT(x), B=coopLoad(wb)) gives acc[i][j] = the dot for
// output row (tile_m+j), col (tile_n+i); the scale loaded as
// coopLoad(da,0)[i][j]=da[j], coopLoadT(dw,0)[i][j]=dw[i] lines up element-for-
// element; the epilogue coopStoreT's each yacc tile to a per-wave LDS slice and
// writes y via normal stores (coopStore straight to y is invisible to auto-sync).
//
// naga_oil corrupts coopmat IR → raw: true. A `var` coop-mat re-declared in a
// loop is NOT re-zeroed → zero_i/zero_f declared once and assigned.
//
// Constraints: WG_X == BM_TILES·BN_TILES·64; M % BM == 0 (engine pads chunks to
// 64), N % BN == 0, K % 64 == 0. Control flow is wave-uniform (coopmat needs
// subgroup-, not workgroup-, uniformity — each wave has its own wm/wn).

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> weights: array<u32>;   // [N×K] Q4_0 packed (18 B/block)
@group(0) @binding(1) var<storage, read> x: array<i8>;          // [M×K] Q8_0 quants
@group(0) @binding(2) var<storage, read> x_scales: array<f16>;  // [M×(K/32)] d_a
@group(0) @binding(3) var<storage, read_write> y: array<f16>;   // [M×N]

const K: u32 = #{K_DIM}u;
const N: u32 = #{N_DIM}u;
const WG: u32 = #{WG_X}u;
const BM_TILES: u32 = #{BM_TILES}u;  // waves down M
const BN_TILES: u32 = #{BN_TILES}u;  // waves across N (= 16×16 tiles across N)
const RM: u32 = #{RM}u;              // register M-tiles per wave (occupancy knob)
// Workgroup→block mapping (see gemm_q4_0_i8): 0 = x:N-block, y:M-block;
// 1 = x:M-block, y:N-block so one N-strip's M-blocks launch consecutively and
// its weight strip stays hot in L2. Dispatch transposed at 1.
const SWIZZLE: u32 = #{SWIZZLE}u;

const WAVES: u32 = BM_TILES * BN_TILES;   // = WG / 64
const BM: u32 = BM_TILES * RM * 16u;      // output rows per workgroup
const BN: u32 = BN_TILES * 16u;           // output cols per workgroup
const NB: u32 = K / 32u;                  // 32-blocks per row
const ROW_WORDS: u32 = NB * 18u / 4u;     // Q4_0 words per W row (NB even → exact)

// One block PAIR's unpacked i8 weights for the BN-row strip: [BN rows × 64 K]
// row-major (cols 0..32 = block β, 32..64 = block β+1). The two blocks' per-col
// d_w (block β in [0,BN), β+1 in [BN,2·BN)) and per-row d_a (same split over BM).
var<workgroup> wb: array<i8, BN * 64u>;
var<workgroup> dw: array<f32, 2u * BN>;
var<workgroup> da: array<f32, 2u * BM>;
// Per-wave f32→f16 epilogue scratch: RM tiles × 256 per wave.
var<workgroup> stg: array<f32, WAVES * RM * 256u>;

// Unpack one Q4_0 block (4 qs words) to i8 (q−8) at wb[base..+32]. Low nibble of
// byte j → position j, high nibble → 16+j (the BlockQ4_0 logical order).
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
    let wave = lid / 64u;            // which wave
    let wm = wave / BN_TILES;        // this wave's M-position (owns RM tiles)
    let wn = wave % BN_TILES;        // this wave's N-tile
    let m_blk = select(wg.y, wg.x, SWIZZLE == 1u);
    let n_blk = select(wg.x, wg.y, SWIZZLE == 1u);
    let m0 = m_blk * BM;             // output row block base
    let n0 = n_blk * BN;             // output col block base
    let row0 = wm * RM * 16u;        // this wave's first M-row within the block

    var zero_i: coop_mat16x16<i32, C>;
    var zero_f: coop_mat16x16<f32, C>;
    var yacc: array<coop_mat16x16<f32, C>, RM>; // register-resident output column
    for (var r = 0u; r < RM; r += 1u) {
        yacc[r] = zero_f;
    }

    for (var beta = 0u; beta < NB; beta += 2u) {
        // Cooperatively unpack the BN-row weight strip (one block pair = 64 K)
        // into wb + the two d_w scale rows. The first BN threads each own one W
        // row's word-aligned 9-word pair (a single 18 B block isn't word-aligned).
        if (lid < BN) {
            let wb0 = (n0 + lid) * ROW_WORDS + (beta / 2u) * 9u;
            var w: array<u32, 9>;
            for (var i = 0u; i < 9u; i += 1u) {
                w[i] = weights[wb0 + i];
            }
            dw[lid] = unpack2x16float(w[0]).x;
            unpack_block(
                (w[0] >> 16u) | (w[1] << 16u),
                (w[1] >> 16u) | (w[2] << 16u),
                (w[2] >> 16u) | (w[3] << 16u),
                (w[3] >> 16u) | (w[4] << 16u),
                lid * 64u,
            );
            dw[BN + lid] = unpack2x16float(w[4]).y;
            unpack_block(w[5], w[6], w[7], w[8], lid * 64u + 32u);
        }
        // The BM row scales for both blocks (f16→f32), strided across all threads.
        for (var r = lid; r < BM; r += WG) {
            da[r] = f32(x_scales[(m0 + r) * NB + beta]);
            da[BM + r] = f32(x_scales[(m0 + r) * NB + beta + 1u]);
        }
        workgroupBarrier(); // wb, dw, da written before MMA / rescale read them

        // Each wave processes the pair's two blocks into its RM yacc tiles. The B
        // fragment is coopLoad'd ONCE per substep and fed to all RM MMAs (in-wave
        // ILP + register weight reuse); A tiles load per (r, substep) from global.
        for (var u = 0u; u < 2u; u += 1u) {
            var acc: array<coop_mat16x16<i32, C>, RM>;
            for (var r = 0u; r < RM; r += 1u) {
                acc[r] = zero_i;
            }
            let coff = u * 32u;          // wb column base for this block
            let kb = beta * 32u + coff;  // global K base for this block
            for (var s = 0u; s < 2u; s += 1u) {
                let b = coopLoad<coop_mat16x16<i8, B>>(&wb[(wn * 16u) * 64u + coff + s * 16u], 64u);
                for (var r = 0u; r < RM; r += 1u) {
                    let a = coopLoadT<coop_mat16x16<i8, A>>(&x[(m0 + row0 + r * 16u) * K + kb + s * 16u], K);
                    acc[r] = coopMultiplyAdd(a, b, acc[r]);
                }
            }
            // 0-stride broadcast scale. The col half (dw) is shared across the RM
            // M-tiles → load it once; the row half (da) differs per tile.
            let dwm = coopLoadT<coop_mat16x16<f32, C>>(&dw[u * BN + wn * 16u], 0u);
            for (var r = 0u; r < RM; r += 1u) {
                let scale = coopLoad<coop_mat16x16<f32, C>>(&da[u * BM + row0 + r * 16u], 0u) * dwm;
                yacc[r] = yacc[r] + scale * f32(acc[r]);
            }
        }
        workgroupBarrier(); // MMA/rescale reads done before next pair overwrites LDS
    }

    // Epilogue: coopStoreT each yacc tile (f32) to this wave's LDS slice, then
    // write y via NORMAL stores (coopStore straight to y is invisible to vulkano
    // auto-sync — touch.wgsl). Per-wave slices don't collide; the barrier orders
    // the stores before the lane reads.
    let so = wave * RM * 256u;
    for (var r = 0u; r < RM; r += 1u) {
        coopStoreT(yacc[r], &stg[so + r * 256u], 16u);
    }
    workgroupBarrier();
    let lane = lid - wave * 64u;
    for (var r = 0u; r < RM; r += 1u) {
        for (var e = lane; e < 256u; e += 64u) {
            let row = m0 + row0 + r * 16u + e / 16u;
            let col = n0 + wn * 16u + e % 16u;
            y[row * N + col] = f16(stg[so + r * 256u + e]);
        }
    }
}
