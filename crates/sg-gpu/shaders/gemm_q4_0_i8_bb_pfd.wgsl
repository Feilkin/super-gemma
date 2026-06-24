// "BIGBOY PFD" — bb_m4 with DEPTH-D COOPERATIVE LDS WEIGHT PREFETCH.
// Both FFN shapes via #{K_DIM}/#{N_DIM} (down K=21504 N=5376, up K=5376 N=21504),
// M_TILES=4, single wave (WG=64).
//
// Premise (bb_m4 RGP, 2026-06-24, [[bb-m4-binding-stall-is-weight-load]]): bb_m4's
// first & biggest K-loop stall is the WEIGHT-load vmcnt before the WMMAs — exposed
// because the 4×row footprint runs at low occupancy, so too few waves to hide it
// cross-wave. bb_m4 only uses lanes 0-15 (N_COLS=16) to load weights; the other 48
// are idle.
//
// Lever: use ALL 64 lanes to cooperatively prefetch PFD pairs of weights at once,
// staged PACKED in double-buffered LDS, then consume one pair/iter from LDS — so the
// weight-load vmcnt is paid once per PFD pairs and hidden behind PFD pairs of MMA
// instead of once per pair. Per-lane register carry is 9 words REGARDLESS of PFD
// (the prefetched group lives in regs t0..t8 across one group's inner loop, then is
// stored to LDS), same as bb_m4_pf's 1-deep register prefetch. PFD buys a deeper
// hide-window + more lanes issuing in parallel, at the cost of 2*PFD*N_COLS*9 u32 of
// LDS and PFD*N_COLS*9 simultaneous loads.
// RISK: at PFD=4 that's 576 in-flight loads — may exceed per-SIMD VMEM and serialize;
// and the PFD-unrolled consume body may pile up a-fragments → spill. Read shaderstats.
//
// Static-unrolled group loop (NOT a dynamic slot index — that was axp3's 8×-instr
// trap, [[activation-prefetch-static-ping-pong-best-l2]]). No workgroupBarriers: WG=64
// single wave, in-wave LDS RAW covered by ACO waitcnt
// ([[single-wave-workgroupbarrier-redundant]]). Keeps the fo levers: 0-stride rescale
// + direct f16 coopStore epilogue.

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> weights: array<u32>;   // [N×K] Q4_0 packed
@group(0) @binding(1) var<storage, read> x: array<i8>;          // [M×K] Q8 quants
@group(0) @binding(2) var<storage, read> x_scales: array<u32>;  // [M×(K/32)] d_a, two f16 packed/word
@group(0) @binding(3) var<storage, read_write> y: array<f16>;   // [M×N]

const K: u32 = #{K_DIM}u;
const N: u32 = #{N_DIM}u;
const N_COLS: u32 = 16u;               // N_TILES = 1
const M_ROWS: u32 = 64u;               // M_TILES = 4 (hand-unrolled)
const NB: u32 = K / 32u;               // 672 32-blocks per row
const ROW_WORDS: u32 = NB * 18u / 4u;  // Q4_0 words per W row
// Prefetch depth: pairs of 32-blocks fetched per cooperative LDS fill (1, 2, 4).
const PFD: u32 = #{PFD}u;
const PAIRS: u32 = NB / 2u;            // 336 pairs per row
const NUM_GROUPS: u32 = PAIRS / PFD;   // PFD evenly divides 336 (336/1/2/4 all exact)
const WBUF: u32 = PFD * N_COLS * 9u;   // packed u32 per LDS buffer

// LDS: double-buffered PACKED weights (group g in buf g&1, group g+1 prefetched into
// buf (g+1)&1), the unpacked weight block-pair (16 N-rows × 64 K), the pair's two d_w
// per col, and the current block's d_a per row (for the 0-stride broadcast).
var<workgroup> wpack: array<u32, 2u * PFD * N_COLS * 9u>;
var<workgroup> wb: array<i8, N_COLS * 64u>;
var<workgroup> dw2: array<f32, 2u * N_COLS>;
var<workgroup> da_l: array<f32, M_ROWS>;

// Unpack one Q4_0 block (4 qs words) to i8 (q−8) at wb[base..+32].
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
    let m0 = wg.x * M_ROWS;  // [M-blocks, N-blocks] dispatch
    let n0 = wg.y * N_COLS;
    let r0 = m0 * K;          // x base for tile rows  0..15
    let r1 = (m0 + 16u) * K;  // x base for tile rows 16..31
    let r2 = (m0 + 32u) * K;  // x base for tile rows 32..47
    let r3 = (m0 + 48u) * K;  // x base for tile rows 48..63

    var zero_f: coop_mat16x16<f32, C>;
    var yacc0 = zero_f;
    var yacc1 = zero_f;
    var yacc2 = zero_f;
    var yacc3 = zero_f;
    var zero_i: coop_mat16x16<i32, C>;

    // ── Prologue: cooperatively fill buf 0 with group 0 (pairs 0..PFD-1). Lane
    // lid<PFD·N_COLS handles (slot = lid/N_COLS, col = lid%N_COLS); dest = 9·lid since
    // 9·(slot·N_COLS + col) = 9·lid. This fill's vmcnt is the unavoidable pipeline
    // prime, amortized over the K-sweep.
    if (lid < PFD * N_COLS) {
        let col = lid % N_COLS;
        let pair = lid / N_COLS;
        let wp = (n0 + col) * ROW_WORDS + pair * 9u;
        let d = 9u * lid;
        wpack[d] = weights[wp]; wpack[d + 1u] = weights[wp + 1u]; wpack[d + 2u] = weights[wp + 2u];
        wpack[d + 3u] = weights[wp + 3u]; wpack[d + 4u] = weights[wp + 4u]; wpack[d + 5u] = weights[wp + 5u];
        wpack[d + 6u] = weights[wp + 6u]; wpack[d + 7u] = weights[wp + 7u]; wpack[d + 8u] = weights[wp + 8u];
    }

    for (var g = 0u; g < NUM_GROUPS; g = g + 1u) {
        let cur = (g & 1u) * WBUF;          // current group's buffer offset
        let nxt = ((g + 1u) & 1u) * WBUF;   // next group's buffer offset
        let do_pf = (g + 1u) < NUM_GROUPS;

        // ── Issue the cooperative prefetch of group g+1 into registers (NOT yet
        // stored → its vmcnt hides behind this group's MMAs). Same 9-reg/lane carry as
        // bb_m4_pf, independent of PFD.
        var t0 = 0u; var t1 = 0u; var t2 = 0u; var t3 = 0u; var t4 = 0u;
        var t5 = 0u; var t6 = 0u; var t7 = 0u; var t8 = 0u;
        if (do_pf && lid < PFD * N_COLS) {
            let col = lid % N_COLS;
            let pair = (g + 1u) * PFD + lid / N_COLS;
            let wp = (n0 + col) * ROW_WORDS + pair * 9u;
            t0 = weights[wp]; t1 = weights[wp + 1u]; t2 = weights[wp + 2u];
            t3 = weights[wp + 3u]; t4 = weights[wp + 4u]; t5 = weights[wp + 5u];
            t6 = weights[wp + 6u]; t7 = weights[wp + 7u]; t8 = weights[wp + 8u];
        }

        // ── Consume this group's PFD pairs from cur buffer (PFD const → ACO unrolls;
        // static slot index, no dynamic addressing).
        for (var s = 0u; s < PFD; s = s + 1u) {
            let p = g * PFD + s;     // pair index
            let k0 = p * 64u;        // beta*32, beta = p*2

            // This pair's 9 packed words from LDS (col = lid, lanes 0-15).
            let wbase = cur + 9u * (s * N_COLS + lid);
            var w0 = 0u; var w1 = 0u; var w2 = 0u; var w3 = 0u; var w4 = 0u;
            var w5 = 0u; var w6 = 0u; var w7 = 0u; var w8 = 0u;
            if (lid < N_COLS) {
                w0 = wpack[wbase]; w1 = wpack[wbase + 1u]; w2 = wpack[wbase + 2u];
                w3 = wpack[wbase + 3u]; w4 = wpack[wbase + 4u]; w5 = wpack[wbase + 5u];
                w6 = wpack[wbase + 6u]; w7 = wpack[wbase + 7u]; w8 = wpack[wbase + 8u];
            }

            // a{block}_{rowtile}{khalf}: block β = b0, block β+1 = b1 (+32 K).
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
            // d_a pair: two consecutive f16 (beta, beta+1) in ONE u32. All 64 lanes
            // load their own row's scale (M_ROWS == WG).
            let dpair = unpack2x16float(x_scales[(m0 + lid) * PAIRS + p]);
            let da0 = dpair.x;
            let da1 = dpair.y;

            // ── Unpack the weights (VALU) from LDS-packed into LDS-unpacked.
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

            // ── B-fragments from LDS (block × K-half), then the 16 WMMAs.
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

            // ── Rescale block β (d_a·d_w 0-stride broadcast) into yacc, then β+1.
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

        // ── Store the prefetched group g+1 into nxt buffer (its vmcnt is now resolved
        // behind the MMAs above). Next outer iter reads it from cur=(g+1)&1.
        if (do_pf && lid < PFD * N_COLS) {
            let d = nxt + 9u * lid;
            wpack[d] = t0; wpack[d + 1u] = t1; wpack[d + 2u] = t2;
            wpack[d + 3u] = t3; wpack[d + 4u] = t4; wpack[d + 5u] = t5;
            wpack[d + 6u] = t6; wpack[d + 7u] = t7; wpack[d + 8u] = t8;
        }
    }

    coopStoreT(f16(yacc0), &y[m0 * N + n0], N);
    coopStoreT(f16(yacc1), &y[(m0 + 16u) * N + n0], N);
    coopStoreT(f16(yacc2), &y[(m0 + 32u) * N + n0], N);
    coopStoreT(f16(yacc3), &y[(m0 + 48u) * N + n0], N);
}
