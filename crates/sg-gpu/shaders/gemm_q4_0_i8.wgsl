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
// impossible. Per 32-block β:
//   - the ACC = M_TILES·N_TILES i32 accumulators get the block's dot via 2
//     substeps of 16 (a Q4_0 block is 32 = low-nibble half then high-nibble
//     half, in the logical order the reference pins);
//   - all i32 tiles are stored to LDS, converted to f32, scaled per-element by
//     d_a[m]·d_w[n], and added into the persistent f32 output (`facc`).
// KHR coopmat1 has no fragment element access, so the i32→f32 rescale goes
// through LDS each block — the analog of the f16 path's dequant staging, and
// the overhead the microbench MEASURES against f16 (it decides whether int8
// wins: int8 MMA throughput vs this per-block roundtrip).
//
// W is i8 in global → coopLoad reads B=Wᵀ directly (no dequant staging, unlike
// f16 gemm). A=X (coopLoadT), B=Wᵀ (coopLoad of row-major W). Operand i8 data
// must live in `array<i8>` buffers (VK_KHR_8bit_storage; the <i8> generic on
// coopLoad is ignored — the scalar comes from the pointer base type).
//
// naga_oil corrupts coopmat IR → raw: true. A `var` coop-mat re-declared in a
// loop is NOT re-zeroed (naga/ACO keeps the registers live) → `acc[]` is
// declared once and explicitly re-zeroed from `zero_i` each block.

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

// Persistent f32 output (ACC × 16×16), accumulated across blocks.
var<workgroup> facc: array<f32, TILE_ELEMS>;
// Per-block i32 dots (coopStore target), same tile-major layout as facc.
var<workgroup> s32: array<i32, TILE_ELEMS>;
// Per-block scales for this block's rows / cols.
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

    for (var i = lid; i < TILE_ELEMS; i += WG) {
        facc[i] = 0.0;
    }
    workgroupBarrier();

    var zero_i: coop_mat16x16<i32, C>;
    var acc: array<coop_mat16x16<i32, C>, ACC>;

    for (var beta = 0u; beta < NB; beta += 1u) {
        for (var i = 0u; i < ACC; i += 1u) {
            acc[i] = zero_i;
        }
        let kbase = beta * 32u;
        // Two 16-wide substeps cover the 32-block (low then high nibbles).
        for (var s = 0u; s < 2u; s += 1u) {
            var b: array<coop_mat16x16<i8, B>, N_TILES>;
            for (var nt = 0u; nt < N_TILES; nt += 1u) {
                b[nt] = coopLoad<coop_mat16x16<i8, B>>(&w[(n0 + nt * 16u) * K + kbase + s * 16u], K);
            }
            for (var mt = 0u; mt < M_TILES; mt += 1u) {
                let a = coopLoadT<coop_mat16x16<i8, A>>(&x[(m0 + mt * 16u) * K + kbase + s * 16u], K);
                for (var nt = 0u; nt < N_TILES; nt += 1u) {
                    acc[mt * N_TILES + nt] = coopMultiplyAdd(a, b[nt], acc[mt * N_TILES + nt]);
                }
            }
        }
        for (var t = 0u; t < ACC; t += 1u) {
            coopStoreT(acc[t], &s32[t * 256u], 16u);
        }
        for (var i = lid; i < M_ROWS; i += WG) {
            da_l[i] = f32(x_scales[(m0 + i) * NB + beta]);
        }
        for (var i = lid; i < N_COLS; i += WG) {
            dw_l[i] = f32(w_scales[(n0 + i) * NB + beta]);
        }
        workgroupBarrier();
        for (var i = lid; i < TILE_ELEMS; i += WG) {
            let t = i / 256u;
            let e = i % 256u;
            let m = (t / N_TILES) * 16u + e / 16u;
            let n = (t % N_TILES) * 16u + e % 16u;
            facc[i] += f32(s32[i]) * da_l[m] * dw_l[n];
        }
        workgroupBarrier(); // before next block overwrites s32
    }

    for (var i = lid; i < TILE_ELEMS; i += WG) {
        let t = i / 256u;
        let e = i % 256u;
        let row = m0 + (t / N_TILES) * 16u + e / 16u;
        let col = n0 + (t % N_TILES) * 16u + e % 16u;
        y[row * N + col] = f16(facc[i]);
    }
}
