// DIAGNOSTIC, not production: the int8 coopmat MMA throughput CEILING. Same
// MMA count and tiling as gemm_q4_0_i8, but accumulates i32 across ALL K (like
// the f16 gemm accumulates f32) — NO per-block rescale, so the output is wrong
// (scales ignored). Measures whether int8 WMMA beats f16 WMMA on this box, to
// decide if cracking the MMQ per-block rescale is worth it (rank #2).

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
const ACC: u32 = M_TILES * N_TILES;
const M_ROWS: u32 = M_TILES * 16u;
const N_COLS: u32 = N_TILES * 16u;
const TILE_ELEMS: u32 = ACC * 256u;

var<workgroup> s32: array<i32, TILE_ELEMS>;

@compute @workgroup_size(#{WG_X})
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let lid = lid_v.x;
    let m0 = wg.y * M_ROWS;
    let n0 = wg.x * N_COLS;

    // Accumulate i32 across the whole K (declared once → genuine accumulation).
    var acc: array<coop_mat16x16<i32, C>, ACC>;

    for (var k0 = 0u; k0 < K; k0 += 16u) {
        var b: array<coop_mat16x16<i8, B>, N_TILES>;
        for (var nt = 0u; nt < N_TILES; nt += 1u) {
            b[nt] = coopLoad<coop_mat16x16<i8, B>>(&w[(n0 + nt * 16u) * K + k0], K);
        }
        for (var mt = 0u; mt < M_TILES; mt += 1u) {
            let a = coopLoadT<coop_mat16x16<i8, A>>(&x[(m0 + mt * 16u) * K + k0], K);
            for (var nt = 0u; nt < N_TILES; nt += 1u) {
                acc[mt * N_TILES + nt] = coopMultiplyAdd(a, b[nt], acc[mt * N_TILES + nt]);
            }
        }
    }

    for (var t = 0u; t < ACC; t += 1u) {
        coopStoreT(acc[t], &s32[t * 256u], 16u);
    }
    workgroupBarrier();
    for (var i = lid; i < TILE_ELEMS; i += WG) {
        let t = i / 256u;
        let e = i % 256u;
        let row = m0 + (t / N_TILES) * 16u + e / 16u;
        let col = n0 + (t % N_TILES) * 16u + e % 16u;
        y[row * N + col] = f16(f32(s32[i]));
    }
}
