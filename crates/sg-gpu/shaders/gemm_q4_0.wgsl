// Cooperative-matrix Q4_0 GEMM (plan 02): Y[M×N] = X[M×K] · W[N×K]ᵀ for
// prefill. THE compute-critical kernel.
//
// One wave-sized workgroup computes an (M_TILES·16) × (N_TILES·16) block of
// Y as a grid of 16×16 coopmat accumulators. Each K-step dequantizes an
// (N_TILES·16)(rows) × 64(K) strip of W — one aligned Q4_0 block pair per
// thread, bulk-loaded as 9 words like the gemv kernel — into workgroup
// memory, then runs M_TILES × N_TILES × 4 coopMultiplyAdds (KHR cooperative
// matrix, probed f16×f16→f32 16×16×16 config). A tiles are loaded once per
// K-substep and reused across the N strip; the column-major B load from the
// staging buffer transposes Wᵀ for free.
//
// Constraints: M must be a multiple of M_TILES·16 (the engine pads prefill
// chunks); N_TILES·16 == WG_X so the dequant assigns one W row per thread;
// K is a multiple of 64 in this model. Control flow is workgroup-uniform,
// as the cooperative ops require.

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f16>;
@group(0) @binding(2) var<storage, read_write> y: array<f16>;

const K: u32 = #{K_DIM}u;
const N: u32 = #{N_DIM}u;
const WG: u32 = #{WG_X}u;
const M_TILES: u32 = #{M_TILES}u;
const N_TILES: u32 = #{N_TILES}u;
const BLOCKS_PER_ROW: u32 = K / 32u;
const STRIP_ROWS: u32 = N_TILES * 16u;

// STRIP_ROWS W rows × 64 dequantized K columns, row-major.
// (A tiles deliberately load straight from global memory: an LDS-staged A
// was measured 30 % slower — the hardware coopLoadT hits L2 just fine.)
var<workgroup> b_tile: array<f16, #{B_TILE_LEN}>;
// f32 C-tile staging for the f16 output conversion.
var<workgroup> c_stage: array<f32, 256>;

@compute @workgroup_size(#{WG_X})
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let lid = lid_v.x;
    let n0 = wg_id.x * STRIP_ROWS; // output-column block
    let m0 = wg_id.y * M_TILES * 16u; // output-row block

    var acc: array<coop_mat16x16<f32, C>, #{ACC_LEN}>; // zero-initialized

    // This thread dequantizes one aligned Q4_0 block PAIR (64 weights, 36
    // bytes = 9 contiguous words, the gemv access pattern) of W row n0+lid
    // each K-step. One bulk load instead of per-byte global reads.
    let row_n = n0 + lid;
    let row_word_base = row_n * (BLOCKS_PER_ROW * 18u / 4u);

    for (var k0 = 0u; k0 < K; k0 += 64u) {
        let wb = row_word_base + (k0 / 64u) * 9u;
        let w0 = weights[wb];
        let w1 = weights[wb + 1u];
        let w2 = weights[wb + 2u];
        let w3 = weights[wb + 3u];
        let w4 = weights[wb + 4u];
        let w5 = weights[wb + 5u];
        let w6 = weights[wb + 6u];
        let w7 = weights[wb + 7u];
        let w8 = weights[wb + 8u];
        // Block A: d in w0.lo, 16 qs bytes spanning w0.hi..w4.lo.
        dequant_block(
            unpack2x16float(w0).x,
            (w0 >> 16u) | (w1 << 16u),
            (w1 >> 16u) | (w2 << 16u),
            (w2 >> 16u) | (w3 << 16u),
            (w3 >> 16u) | (w4 << 16u),
            lid * 64u,
        );
        // Block B: d in w4.hi, qs in w5..w8.
        dequant_block(unpack2x16float(w4).y, w5, w6, w7, w8, lid * 64u + 32u);
        workgroupBarrier();

        // Four 16-wide K substeps over the staged strip; A reused across
        // the whole N strip.
        for (var s = 0u; s < 4u; s += 1u) {
            for (var mt = 0u; mt < M_TILES; mt += 1u) {
                let a = coopLoadT<coop_mat16x16<f16, A>>(
                    &x[(m0 + mt * 16u) * K + k0 + s * 16u],
                    K,
                );
                for (var nt = 0u; nt < N_TILES; nt += 1u) {
                    let b = coopLoad<coop_mat16x16<f16, B>>(
                        &b_tile[nt * 16u * 64u + s * 16u],
                        64u,
                    );
                    acc[mt * N_TILES + nt] = coopMultiplyAdd(a, b, acc[mt * N_TILES + nt]);
                }
            }
        }
        workgroupBarrier();
    }

    // f32 accumulators → workgroup staging → f16 output tiles (the f32→f16
    // matrix conversion has no WGSL spelling; the round trip is cheap).
    for (var mt = 0u; mt < M_TILES; mt += 1u) {
        for (var nt = 0u; nt < N_TILES; nt += 1u) {
            coopStoreT(acc[mt * N_TILES + nt], &c_stage[0u], 16u);
            workgroupBarrier();
            for (var i = lid; i < 256u; i += WG) {
                let rr = i / 16u;
                let cc = i % 16u;
                y[(m0 + mt * 16u + rr) * N + n0 + nt * 16u + cc] = f16(c_stage[i]);
            }
            workgroupBarrier();
        }
    }
}

// Dequantize one Q4_0 block (d + 4 qs words already in registers) into
// b_tile[base..base+32].
fn dequant_block(d: f32, q0: u32, q1: u32, q2: u32, q3: u32, base: u32) {
    var qs = array<u32, 4>(q0, q1, q2, q3);
    for (var w = 0u; w < 4u; w += 1u) {
        let word = qs[w];
        for (var b = 0u; b < 4u; b += 1u) {
            let byte = (word >> (8u * b)) & 0xFFu;
            let j = 4u * w + b;
            b_tile[base + j] = f16((f32(byte & 0xFu) - 8.0) * d);
            b_tile[base + 16u + j] = f16((f32(byte >> 4u) - 8.0) * d);
        }
    }
}
