// "BIGBOY" int8-MMQ GEMM — FFN-down shape ONLY (K=21504, N=5376), M_TILES=2.
// One wave (WG=64) per 32×16 output tile; plain [M-blocks, N-blocks] dispatch.
//
// The thesis (from the fo β×2 traces): a stall site costs ~one memory latency
// REGARDLESS of how many loads are queued behind it — what matters is how many
// independent loads are in flight when you hit the first `s_waitcnt vmcnt`. The
// β×2 fo kernel already groups its 8 activation loads → 8 WMMAs (one stall for
// the whole batch), but it STILL has separate stall regions for (a) the weight
// word load — whose `array<u32,9>` reuses one register window so the loads
// serialize behind their own waitcnts — and (b) the activation batch, plus the
// hoisted d_a scales.
//
// BIGBOY removes EVERY inner loop. No `for s`, no `for mt`, no `for u`, no β×2
// branch — the whole block-PAIR is straight-line code, the only loop left is the
// K-sweep over pairs. Every global load for the iteration (the 9 weight words in
// 9 DISTINCT registers, all 8 activation fragments, both d_a scales) is issued up
// front as one independent batch → ACO packs them back-to-back and a SINGLE vmcnt
// stall covers the lot; the weight unpack (VALU) + the 8 WMMAs then run while the
// rest of the batch lands. The cost is register pressure (8 live a-fragments + 4
// i32 accumulators + 2 f32 yacc + 9 weight words) → lower occupancy; the bet is
// that one consolidated stall/iter beats many waves each paying several.
//
// Keeps the two fo levers: 0-STRIDE scale rescale (no LDS stage) and the DIRECT
// f16-coopStore epilogue (no LDS scratch). No workgroupBarriers — WG=64 is a
// single wave, the in-wave LDS RAW is covered by ACO's waitcnt
// ([[single-wave-workgroupbarrier-redundant]]).

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> weights: array<u32>;   // [N×K] Q4_0 packed
@group(0) @binding(1) var<storage, read> x: array<i8>;          // [M×K] Q8 quants
@group(0) @binding(2) var<storage, read> x_scales: array<u32>;  // [M×(K/32)] d_a, two f16 packed/word
@group(0) @binding(3) var<storage, read_write> y: array<f16>;   // [M×N]

const K: u32 = 21504u;
const N: u32 = 5376u;
const N_COLS: u32 = 16u;               // N_TILES = 1
const M_ROWS: u32 = 32u;               // M_TILES = 2 (hand-unrolled)
const NB: u32 = K / 32u;               // 672 32-blocks per row
const ROW_WORDS: u32 = NB * 18u / 4u;  // Q4_0 words per W row
// Weight prefetch. The bb trace collapsed the iteration to ONE stall site — the
// 16-lane weight load (`buffer_load_b64`, 2729 clk). That reverses the old "weights
// aren't the binding stall" finding (which held while ACTIVATIONS were the stall;
// bb hid those). PF=1 software-pipelines the weight pair: hold the NEXT pair's 9
// words in registers, issue their load inside THIS iteration's batch so the latency
// lands behind the 8 WMMAs + rescale, and this iter's words are already resident →
// no top-of-iter weight stall. PF=0 = inline load (the baseline bb).
const PF: u32 = #{PF}u;

// The only LDS: the unpacked weight block-pair (16 N-rows × 64 K) + the pair's two
// d_w per col + the current block's d_a per row (for the 0-stride broadcast).
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
    let r0 = m0 * K;         // x base for tile rows 0..15
    let r1 = (m0 + 16u) * K; // x base for tile rows 16..31

    var zero_f: coop_mat16x16<f32, C>;
    var yacc0 = zero_f;
    var yacc1 = zero_f;
    var zero_i: coop_mat16x16<i32, C>;

    // PF=1 prologue: prefetch pair 0's 9 words into 9 distinct registers.
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

        // ── Batch 1: issue EVERY global load for this pair, back-to-back. The 9
        // weight words go to 9 DISTINCT registers (no array reuse → no serializing
        // waitcnt). All 8 activation fragments + both d_a scales are independent of
        // the weights and of each other → one vmcnt stall covers the whole batch.
        var w0 = 0u; var w1 = 0u; var w2 = 0u; var w3 = 0u; var w4 = 0u;
        var w5 = 0u; var w6 = 0u; var w7 = 0u; var w8 = 0u;
        if (PF == 1u) {
            // This pair's words are already resident (prefetched last iter); consume
            // them, then issue the NEXT pair's load into wn* as part of this batch.
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
        // block β (b0): rows {0,16} × K-halves {0,16}; block β+1 (b1): + 32 K.
        let a0_00 = coopLoadT<coop_mat16x16<i8, A>>(&x[r0 + k0], K);
        let a0_01 = coopLoadT<coop_mat16x16<i8, A>>(&x[r0 + k0 + 16u], K);
        let a0_10 = coopLoadT<coop_mat16x16<i8, A>>(&x[r1 + k0], K);
        let a0_11 = coopLoadT<coop_mat16x16<i8, A>>(&x[r1 + k0 + 16u], K);
        let a1_00 = coopLoadT<coop_mat16x16<i8, A>>(&x[r0 + k0 + 32u], K);
        let a1_01 = coopLoadT<coop_mat16x16<i8, A>>(&x[r0 + k0 + 48u], K);
        let a1_10 = coopLoadT<coop_mat16x16<i8, A>>(&x[r1 + k0 + 32u], K);
        let a1_11 = coopLoadT<coop_mat16x16<i8, A>>(&x[r1 + k0 + 48u], K);
        // d_a pair: the two consecutive f16 (beta, beta+1) sit in ONE u32 (beta even,
        // row stride NB even → u32-aligned). Issue the RAW u32 load NOW, as part of this
        // batch — but do NOT unpack yet. unpack2x16float is the consumer that forces the
        // vmcnt wait; deferring it to the rescale (past the 8 WMMAs) lets the load's
        // latency overlap the WMMAs, so the word has landed by first-use → no dedicated
        // stall in the lid<M_ROWS branch.
        var da_raw = 0u;
        if (lid < M_ROWS) {
            da_raw = x_scales[(m0 + lid) * (NB / 2u) + (beta / 2u)];
        }

        // ── Unpack the weights (VALU — hides the batch's vmcnt) into LDS.
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

        // ── B-fragments from LDS (block × K-half), then the 8 WMMAs.
        let b0_0 = coopLoad<coop_mat16x16<i8, B>>(&wb[0u], 64u);
        let b0_1 = coopLoad<coop_mat16x16<i8, B>>(&wb[16u], 64u);
        let b1_0 = coopLoad<coop_mat16x16<i8, B>>(&wb[32u], 64u);
        let b1_1 = coopLoad<coop_mat16x16<i8, B>>(&wb[48u], 64u);
        var acc0_0 = coopMultiplyAdd(a0_00, b0_0, zero_i); // block β, row-tile 0
        acc0_0 = coopMultiplyAdd(a0_01, b0_1, acc0_0);
        var acc0_1 = coopMultiplyAdd(a0_10, b0_0, zero_i); // block β, row-tile 1
        acc0_1 = coopMultiplyAdd(a0_11, b0_1, acc0_1);
        var acc1_0 = coopMultiplyAdd(a1_00, b1_0, zero_i); // block β+1, row-tile 0
        acc1_0 = coopMultiplyAdd(a1_01, b1_1, acc1_0);
        var acc1_1 = coopMultiplyAdd(a1_10, b1_0, zero_i); // block β+1, row-tile 1
        acc1_1 = coopMultiplyAdd(a1_11, b1_1, acc1_1);

        // ── Rescale block β (d_a·d_w 0-stride broadcast) into yacc, then β+1.
        // Unpack the scale word HERE (first use) — the load issued back in the batch has
        // long since returned, so this is the convert only, no memory wait.
        let dpair = unpack2x16float(da_raw);
        if (lid < M_ROWS) { da_l[lid] = dpair.x; }
        let s0_0 = coopLoad<coop_mat16x16<f32, C>>(&da_l[0u], 0u)
            * coopLoadT<coop_mat16x16<f32, C>>(&dw2[0u], 0u);
        let s0_1 = coopLoad<coop_mat16x16<f32, C>>(&da_l[16u], 0u)
            * coopLoadT<coop_mat16x16<f32, C>>(&dw2[0u], 0u);
        yacc0 = yacc0 + s0_0 * f32(acc0_0);
        yacc1 = yacc1 + s0_1 * f32(acc0_1);
        if (lid < M_ROWS) { da_l[lid] = dpair.y; }
        let s1_0 = coopLoad<coop_mat16x16<f32, C>>(&da_l[0u], 0u)
            * coopLoadT<coop_mat16x16<f32, C>>(&dw2[N_COLS], 0u);
        let s1_1 = coopLoad<coop_mat16x16<f32, C>>(&da_l[16u], 0u)
            * coopLoadT<coop_mat16x16<f32, C>>(&dw2[N_COLS], 0u);
        yacc0 = yacc0 + s1_0 * f32(acc1_0);
        yacc1 = yacc1 + s1_1 * f32(acc1_1);
    }

    coopStoreT(f16(yacc0), &y[m0 * N + n0], N);
    coopStoreT(f16(yacc1), &y[(m0 + 16u) * N + n0], N);
}
