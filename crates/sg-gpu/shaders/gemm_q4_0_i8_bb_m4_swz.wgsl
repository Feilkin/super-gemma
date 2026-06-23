// "BIGBOY M4 + SUPER-BLOCK SWIZZLE" int8-MMQ GEMM — FFN-down (K=21504, N=5376),
// M_TILES=4, 1D dispatch. bb_m4's one-batch body, but the linear workgroup id is
// decoded by `tile_index` (mirrored from gemm_q4_0_i8_l2) into a super-block order
// instead of the plain [M-blocks, N-blocks] launch.
//
// bb_m4 launches M-fast: consecutive tiles are the 4 M-blocks of one n-strip, so
// the whole X (all 4 M-blocks, ~5.5 MB) stays co-resident and overflows L2 — the
// thrash that grew with concurrency in the bb traces. The super-block fix: each
// super-block is BN_SB adjacent n-strips × all NB_M m-blocks, ordered m-OUTER (for
// each m, sweep the BN_SB n-strips). So the BN_SB weight strips are reused down the
// M dimension while only ONE 64×K activation block is hot → footprint ≈
// BN_SB·(K·16 weights) + one 64×K X, tuned to the 2 MB L2. BN_SB=1 is the plain
// transpose; BN_SB=4 was the proven l2 sweet spot.
//
// Keeps bb_m4's levers: one-batch loads, 0-stride scale rescale, direct f16
// coopStore, no barriers (single wave). Dispatch 1D [(M/64)·(N/16), 1, 1].

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> weights: array<u32>;   // [N×K] Q4_0 packed
@group(0) @binding(1) var<storage, read> x: array<i8>;          // [M×K] Q8 quants
@group(0) @binding(2) var<storage, read> x_scales: array<u32>;  // [M×(K/32)] d_a, two f16 packed/word
@group(0) @binding(3) var<storage, read_write> y: array<f16>;   // [M×N]

const K: u32 = 21504u;
const N: u32 = 5376u;
const M_TOTAL: u32 = 256u;             // prefill chunk (experiment kernel, M=256 only)
const N_COLS: u32 = 16u;               // N_TILES = 1
const M_ROWS: u32 = 64u;               // M_TILES = 4 (hand-unrolled)
const NB: u32 = K / 32u;               // 672 32-blocks per row
const ROW_WORDS: u32 = NB * 18u / 4u;  // Q4_0 words per W row
const NB_N: u32 = N / N_COLS;          // 336 n-blocks
const NB_M: u32 = M_TOTAL / M_ROWS;    // 4 m-blocks
const PF: u32 = #{PF}u;                // weight prefetch (0 = inline, best for bb_m4)

// Super-block width: BN_SB adjacent n-strips share the co-resident window, swept
// m-outer. NB_N must be divisible by BN_SB (336 = 2·2·2·3·7, so 1/2/4/8 all clean).
const BN_SB: u32 = #{BN_SB}u;

// Each super-block holds BN_SB·NB_M tiles, m OUTER: for each m, sweep BN_SB n.
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

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let lid = lid_v.x;
    // Linear tile id (1D dispatch [(M/64)·(N/16), 1, 1] → t = wg.x). Super-block decode.
    let t = wg.x;
    let tile = tile_index(t);
    let m0 = tile.y * M_ROWS;
    let n0 = tile.x * N_COLS;
    let r0 = m0 * K;
    let r1 = (m0 + 16u) * K;
    let r2 = (m0 + 32u) * K;
    let r3 = (m0 + 48u) * K;

    var zero_f: coop_mat16x16<f32, C>;
    var yacc0 = zero_f;
    var yacc1 = zero_f;
    var yacc2 = zero_f;
    var yacc3 = zero_f;
    var zero_i: coop_mat16x16<i32, C>;

    var wn0 = 0u; var wn1 = 0u; var wn2 = 0u; var wn3 = 0u; var wn4 = 0u;
    var wn5 = 0u; var wn6 = 0u; var wn7 = 0u; var wn8 = 0u;
    if (PF == 1u && lid < N_COLS) {
        let wp0 = (n0 + lid) * ROW_WORDS;
        wn0 = weights[wp0]; wn1 = weights[wp0 + 1u]; wn2 = weights[wp0 + 2u];
        wn3 = weights[wp0 + 3u]; wn4 = weights[wp0 + 4u]; wn5 = weights[wp0 + 5u];
        wn6 = weights[wp0 + 6u]; wn7 = weights[wp0 + 7u]; wn8 = weights[wp0 + 8u];
    }

    for (var beta = 0u; beta < NB; beta += 2u) {
        let k0 = beta * 32u;
        let wp = (n0 + lid) * ROW_WORDS + (beta / 2u) * 9u;

        var w0 = 0u; var w1 = 0u; var w2 = 0u; var w3 = 0u; var w4 = 0u;
        var w5 = 0u; var w6 = 0u; var w7 = 0u; var w8 = 0u;
        if (PF == 1u) {
            w0 = wn0; w1 = wn1; w2 = wn2; w3 = wn3; w4 = wn4;
            w5 = wn5; w6 = wn6; w7 = wn7; w8 = wn8;
            if (lid < N_COLS && beta + 2u < NB) {
                let wpn = (n0 + lid) * ROW_WORDS + ((beta + 2u) / 2u) * 9u;
                wn0 = weights[wpn]; wn1 = weights[wpn + 1u]; wn2 = weights[wpn + 2u];
                wn3 = weights[wpn + 3u]; wn4 = weights[wpn + 4u]; wn5 = weights[wpn + 5u];
                wn6 = weights[wpn + 6u]; wn7 = weights[wpn + 7u]; wn8 = weights[wpn + 8u];
            }
        } else if (lid < N_COLS) {
            w0 = weights[wp]; w1 = weights[wp + 1u]; w2 = weights[wp + 2u];
            w3 = weights[wp + 3u]; w4 = weights[wp + 4u]; w5 = weights[wp + 5u];
            w6 = weights[wp + 6u]; w7 = weights[wp + 7u]; w8 = weights[wp + 8u];
        }
        let a0_00 = coopLoadT<coop_mat16x16<i8, A>>(&x[r0 + k0], K);
        let a0_01 = coopLoadT<coop_mat16x16<i8, A>>(&x[r0 + k0 + 16u], K);
        let a0_10 = coopLoadT<coop_mat16x16<i8, A>>(&x[r1 + k0], K);
        let a0_11 = coopLoadT<coop_mat16x16<i8, A>>(&x[r1 + k0 + 16u], K);
        let a0_20 = coopLoadT<coop_mat16x16<i8, A>>(&x[r2 + k0], K);
        let a0_21 = coopLoadT<coop_mat16x16<i8, A>>(&x[r2 + k0 + 16u], K);
        let a0_30 = coopLoadT<coop_mat16x16<i8, A>>(&x[r3 + k0], K);
        let a0_31 = coopLoadT<coop_mat16x16<i8, A>>(&x[r3 + k0 + 16u], K);
        let a1_00 = coopLoadT<coop_mat16x16<i8, A>>(&x[r0 + k0 + 32u], K);
        let a1_01 = coopLoadT<coop_mat16x16<i8, A>>(&x[r0 + k0 + 48u], K);
        let a1_10 = coopLoadT<coop_mat16x16<i8, A>>(&x[r1 + k0 + 32u], K);
        let a1_11 = coopLoadT<coop_mat16x16<i8, A>>(&x[r1 + k0 + 48u], K);
        let a1_20 = coopLoadT<coop_mat16x16<i8, A>>(&x[r2 + k0 + 32u], K);
        let a1_21 = coopLoadT<coop_mat16x16<i8, A>>(&x[r2 + k0 + 48u], K);
        let a1_30 = coopLoadT<coop_mat16x16<i8, A>>(&x[r3 + k0 + 32u], K);
        let a1_31 = coopLoadT<coop_mat16x16<i8, A>>(&x[r3 + k0 + 48u], K);
        let dpair = unpack2x16float(x_scales[(m0 + lid) * (NB / 2u) + (beta / 2u)]);
        let da0 = dpair.x;
        let da1 = dpair.y;

        if (lid < N_COLS) {
            dw2[lid] = unpack2x16float(w0).x;
            unpack_block(
                (w0 >> 16u) | (w1 << 16u),
                (w1 >> 16u) | (w2 << 16u),
                (w2 >> 16u) | (w3 << 16u),
                (w3 >> 16u) | (w4 << 16u),
                lid * 64u,
            );
            dw2[N_COLS + lid] = unpack2x16float(w4).y;
            unpack_block(w5, w6, w7, w8, lid * 64u + 32u);
        }

        let b0_0 = coopLoad<coop_mat16x16<i8, B>>(&wb[0u], 64u);
        let b0_1 = coopLoad<coop_mat16x16<i8, B>>(&wb[16u], 64u);
        let b1_0 = coopLoad<coop_mat16x16<i8, B>>(&wb[32u], 64u);
        let b1_1 = coopLoad<coop_mat16x16<i8, B>>(&wb[48u], 64u);
        var acc0_0 = coopMultiplyAdd(a0_00, b0_0, zero_i);
        acc0_0 = coopMultiplyAdd(a0_01, b0_1, acc0_0);
        var acc0_1 = coopMultiplyAdd(a0_10, b0_0, zero_i);
        acc0_1 = coopMultiplyAdd(a0_11, b0_1, acc0_1);
        var acc0_2 = coopMultiplyAdd(a0_20, b0_0, zero_i);
        acc0_2 = coopMultiplyAdd(a0_21, b0_1, acc0_2);
        var acc0_3 = coopMultiplyAdd(a0_30, b0_0, zero_i);
        acc0_3 = coopMultiplyAdd(a0_31, b0_1, acc0_3);
        var acc1_0 = coopMultiplyAdd(a1_00, b1_0, zero_i);
        acc1_0 = coopMultiplyAdd(a1_01, b1_1, acc1_0);
        var acc1_1 = coopMultiplyAdd(a1_10, b1_0, zero_i);
        acc1_1 = coopMultiplyAdd(a1_11, b1_1, acc1_1);
        var acc1_2 = coopMultiplyAdd(a1_20, b1_0, zero_i);
        acc1_2 = coopMultiplyAdd(a1_21, b1_1, acc1_2);
        var acc1_3 = coopMultiplyAdd(a1_30, b1_0, zero_i);
        acc1_3 = coopMultiplyAdd(a1_31, b1_1, acc1_3);

        da_l[lid] = da0;
        let dw_0 = coopLoadT<coop_mat16x16<f32, C>>(&dw2[0u], 0u);
        yacc0 = yacc0 + (coopLoad<coop_mat16x16<f32, C>>(&da_l[0u], 0u) * dw_0) * f32(acc0_0);
        yacc1 = yacc1 + (coopLoad<coop_mat16x16<f32, C>>(&da_l[16u], 0u) * dw_0) * f32(acc0_1);
        yacc2 = yacc2 + (coopLoad<coop_mat16x16<f32, C>>(&da_l[32u], 0u) * dw_0) * f32(acc0_2);
        yacc3 = yacc3 + (coopLoad<coop_mat16x16<f32, C>>(&da_l[48u], 0u) * dw_0) * f32(acc0_3);
        da_l[lid] = da1;
        let dw_1 = coopLoadT<coop_mat16x16<f32, C>>(&dw2[N_COLS], 0u);
        yacc0 = yacc0 + (coopLoad<coop_mat16x16<f32, C>>(&da_l[0u], 0u) * dw_1) * f32(acc1_0);
        yacc1 = yacc1 + (coopLoad<coop_mat16x16<f32, C>>(&da_l[16u], 0u) * dw_1) * f32(acc1_1);
        yacc2 = yacc2 + (coopLoad<coop_mat16x16<f32, C>>(&da_l[32u], 0u) * dw_1) * f32(acc1_2);
        yacc3 = yacc3 + (coopLoad<coop_mat16x16<f32, C>>(&da_l[48u], 0u) * dw_1) * f32(acc1_3);
    }

    coopStoreT(f16(yacc0), &y[m0 * N + n0], N);
    coopStoreT(f16(yacc1), &y[(m0 + 16u) * N + n0], N);
    coopStoreT(f16(yacc2), &y[(m0 + 32u) * N + n0], N);
    coopStoreT(f16(yacc3), &y[(m0 + 48u) * N + n0], N);
}
