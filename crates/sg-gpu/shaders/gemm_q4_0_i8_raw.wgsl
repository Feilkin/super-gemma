// DIAGNOSTIC, not production: the int8 coopmat MMA throughput CEILING, measured
// with the SAME dependency structure as the real kernel.
//
// `acc` is RE-ZEROED every 32-block (chain depth 2 — two 16-wide MMA substeps),
// exactly like `gemm_q4_0_i8`, so the ACC·NB block-dots are independent across
// blocks: maximal MMA parallelism, the headroom the per-block rescale competes
// against. (The prior version of this kernel accumulated ONE `acc` chain across
// ALL K — a K/16-deep dependent MMA chain — which is latency-bound, NOT a
// ceiling; it floored at ~1 TFLOPS and was misleading. MMQ can't use that free
// in-MMA accumulation anyway: the per-block scale forces the dot out every
// block, which is the whole cost being measured.)
//
// NO rescale. Each block's i32 dot is summed UNSCALED into a register `sink`
// (one bare i32 coopmat add per tile — no LDS, no barrier, no convert, no
// scale), the minimal consume that keeps every MMA observable (defeats
// dead-code elimination). So `sink += acc` isolates MMA + operand-load
// throughput plus the cost of a single coopmat add outside the MMA. The output
// is wrong (scales ignored) — timing only.

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> w: array<i8>; // [N×K]
@group(0) @binding(1) var<storage, read> x: array<i8>; // [M×K]
@group(0) @binding(2) var<storage, read_write> y: array<f16>; // [M×N]

const K: u32 = #{K_DIM}u;
const N: u32 = #{N_DIM}u;
const WG: u32 = #{WG_X}u;
const M_TILES: u32 = #{M_TILES}u;
const N_TILES: u32 = #{N_TILES}u;
const NB: u32 = K / 32u;
const ACC: u32 = M_TILES * N_TILES;
const M_ROWS: u32 = M_TILES * 16u;
const N_COLS: u32 = N_TILES * 16u;
const TILE_ELEMS: u32 = ACC * 256u;

@compute @workgroup_size(#{WG_X})
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let lid = lid_v.x;
    let m0 = wg.y * M_ROWS;
    let n0 = wg.x * N_COLS;

    var zero_i: coop_mat16x16<i32, C>;
    var acc: array<coop_mat16x16<i32, C>, ACC>;
    var acc2: array<coop_mat16x16<i32, C>, ACC>; // 2nd block, unrolled for ILP
    var sink: array<coop_mat16x16<i32, C>, ACC>; // minimal cross-block consume
    for (var i = 0u; i < ACC; i += 1u) {
        sink[i] = zero_i;
    }

    // β loop UNROLLED ×2 (NB even on every shape): two independent blocks'
    // depth-2 MMA chains interleaved in one body, so block β+1's MMAs fill the
    // bubble left by β's substep0→substep1 dependency — testing whether ACO not
    // overlapping consecutive (rolled) iterations is what floors this kernel.
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
        // Minimal consume: unscaled i32 register add (no LDS/barrier/convert).
        for (var t = 0u; t < ACC; t += 1u) {
            sink[t] = sink[t] + acc2[t];
        }
        for (var t = 0u; t < ACC; t += 1u) {
            sink[t] = sink[t] + acc[t];
        }
    }

    // Coopmat f16 epilogue: convert in-register and store straight to y (no LDS,
    // no barrier, no scalar loop) — same path as gemm_q4_0_i8's epilogue.
    for (var mt = 0u; mt < M_TILES; mt += 1u) {
        for (var nt = 0u; nt < N_TILES; nt += 1u) {
            coopStoreT(
                f16(f32(sink[mt * N_TILES + nt])),
                &y[(m0 + mt * 16u) * N + n0 + nt * 16u], N);
        }
    }
}
