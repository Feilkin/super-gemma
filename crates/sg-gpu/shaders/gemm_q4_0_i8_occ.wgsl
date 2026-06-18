// int8-MMQ GEMM, MAX-OCCUPANCY variant (1×1 tile) — the occupancy-side A/B
// against the deployed tall-thin 4×1 `gemm_q4_0_i8.wgsl`. Same math, same Q4_0
// packed weights × Q8 activations, same per-block i8×i8→i32 rescale; the ONLY
// difference is the design philosophy.
//
// The deployed kernel is a coherent LOW-occupancy design: a 4×1 tile, a β×2 MMA
// interleave (`acc2`), and a software weight prefetch (`w_cur/w_next`) all spend
// VGPRs to hide DRAM weight-read latency via *in-wave* ILP — RGP showed it sits
// at 3 waves/SIMD, latency-bound (~46 of 256 GB/s achieved). This kernel is its
// mirror: a 1×1 tile (one 16×16 output block per workgroup) with NO register-
// resident M-reuse, NO MMA interleave (`acc` is reused sequentially across the
// two blocks of a pair), NO prefetch, single-buffer scale `stage`. Footprint is
// minimal → many waves/SIMD; the bet is that more outstanding loads pulls more
// of the 256 GB/s and hides the latency by wave-switching instead — at the cost
// of in-register weight reuse (now leaned entirely on the L2 swizzle). The
// RDNA3-WMMA rule says unroll/ILP helps when occupancy is LOW and regresses when
// it's already high, so at max occupancy the de-interleaved loop is the right
// form. KILL CRITERION (STATUS): must clear ≥5 waves/SIMD on RGP AND beat
// deployed on `mmq_variance`, else it dies.
//
// Weights are still loaded as word-aligned 9-word PAIRS (= 2 Q4_0 blocks = 36 B;
// a single 18 B block is not word-aligned), unpacked to i8 in `wb`; but the two
// blocks are then MMA'd + rescaled SEQUENTIALLY into one `acc`/`yacc` (each
// 32-block has its own d_a·d_w scale, so each block's i32 dot must be pulled out
// and scaled before summing into the f32 output — same constraint as deployed).
// Element layout, the coopLoadT/coopStoreT transpose, and the epilogue LDS
// round-trip (coopStore to `y` is invisible to vulkano auto-sync) all match the
// deployed kernel — see gemm_q4_0_i8.wgsl for the full rationale.

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> weights: array<u32>;   // [N×K] Q4_0 packed (18 B/block)
@group(0) @binding(1) var<storage, read> x: array<i8>;          // [M×K] Q8_0 quants
@group(0) @binding(2) var<storage, read> x_scales: array<f16>;  // [M×(K/32)] d_a
@group(0) @binding(3) var<storage, read_write> y: array<f16>;   // [M×N]

const K: u32 = #{K_DIM}u;
const N: u32 = #{N_DIM}u;
const WG: u32 = #{WG_X}u;
const SWIZZLE: u32 = #{SWIZZLE}u;

const M_ROWS: u32 = 16u;            // 1×1 tile: one 16×16 output block/workgroup
const N_COLS: u32 = 16u;
const NB: u32 = K / 32u;            // 32-blocks per row
const ROW_WORDS: u32 = NB * 18u / 4u;

// One block pair's unpacked i8 weights: [16 rows × 64 K] (cols 0..32 = block β,
// 32..64 = block β+1). The two blocks' per-row d_w scales, and the row scales /
// outer-product scale tile (single-buffered — 16×16).
var<workgroup> wb: array<i8, N_COLS * 64u>;
var<workgroup> dwa: array<f32, N_COLS>;
var<workgroup> dwb: array<f32, N_COLS>;
var<workgroup> da_l: array<f32, M_ROWS>;
var<workgroup> stage: array<f32, 256u>;

// Unpack one Q4_0 block (4 qs words) to i8 (q−8) at wb[base..+32]; low nibble of
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
    let m_blk = select(wg.y, wg.x, SWIZZLE == 1u);
    let n_blk = select(wg.x, wg.y, SWIZZLE == 1u);
    let m0 = m_blk * M_ROWS;
    let n0 = n_blk * N_COLS;

    var zero_i: coop_mat16x16<i32, C>;
    var zero_f: coop_mat16x16<f32, C>;
    var yacc = zero_f; // register-resident output (single 16×16 tile)

    // β-loop walks word-aligned block PAIRS; the two blocks are processed
    // sequentially (de-interleaved) into one `acc`.
    for (var beta = 0u; beta < NB; beta += 2u) {
        // Each of the 16 cols' threads loads + unpacks its 9-word pair.
        if (lid < N_COLS) {
            let wb0 = (n0 + lid) * ROW_WORDS + (beta / 2u) * 9u;
            var w: array<u32, 9>;
            for (var i = 0u; i < 9u; i += 1u) {
                w[i] = weights[wb0 + i];
            }
            // Block β: d in w[0].lo, qs span w[0].hi..w[4].lo.
            dwa[lid] = unpack2x16float(w[0]).x;
            unpack_block(
                (w[0] >> 16u) | (w[1] << 16u),
                (w[1] >> 16u) | (w[2] << 16u),
                (w[2] >> 16u) | (w[3] << 16u),
                (w[3] >> 16u) | (w[4] << 16u),
                lid * 64u,
            );
            // Block β+1: d in w[4].hi, qs in w[5..8].
            dwb[lid] = unpack2x16float(w[4]).y;
            unpack_block(w[5], w[6], w[7], w[8], lid * 64u + 32u);
        }
        workgroupBarrier(); // wb + dwa/dwb written before MMA / rescale read them

        let k0 = beta * 32u;
        for (var u = 0u; u < 2u; u += 1u) {
            // One 32-block = two 16-K substeps into a fresh `acc`.
            var acc = zero_i;
            let coff = u * 32u;       // wb column base for this block
            let kb = k0 + u * 32u;    // global K base for this block
            for (var s = 0u; s < 2u; s += 1u) {
                let b = coopLoad<coop_mat16x16<i8, B>>(&wb[coff + s * 16u], 64u);
                let a = coopLoadT<coop_mat16x16<i8, A>>(&x[m0 * K + kb + s * 16u], K);
                acc = coopMultiplyAdd(a, b, acc);
            }
            // Rescale: build the 16×16 scale = d_a[m]·d_w[n] in `stage`, coopLoadT
            // it, yacc += scale · f32(acc).
            for (var i = lid; i < M_ROWS; i += WG) {
                da_l[i] = f32(x_scales[(m0 + i) * NB + beta + u]);
            }
            workgroupBarrier(); // da_l written; orders prev block's stage coopLoad too
            let n = lid % N_COLS;
            let dw = select(dwb[n], dwa[n], u == 0u);
            for (var m = lid / N_COLS; m < M_ROWS; m += WG / N_COLS) {
                stage[m * N_COLS + n] = da_l[m] * dw;
            }
            workgroupBarrier(); // stage written before coopLoad
            let scale = coopLoadT<coop_mat16x16<f32, C>>(&stage[0], N_COLS);
            yacc = yacc + scale * f32(acc);
            workgroupBarrier(); // scale coopLoad done before next block overwrites stage
        }
    }

    // Epilogue: coopStore yacc (f32) to LDS, then write y via NORMAL stores
    // (coopStore straight to y is invisible to auto-sync — see gemm_q4_0_i8).
    coopStoreT(yacc, &stage[0], 16u);
    workgroupBarrier();
    for (var i = lid; i < 256u; i += WG) {
        let row = m0 + i / 16u;
        let col = n0 + i % 16u;
        y[row * N + col] = f16(stage[i]);
    }
}
