// L2-blocking experiment kernel — hardcoded to the FFN-down shape (K=21504,
// N=5376), 1D dispatch. ONE linear workgroup id per output tile; `tile_index`
// decodes it to (m_block, n_block) and is THE swappable L2-scheduling knob:
// the identity decode reproduces today's launch order; a blocked walk makes the
// co-resident workgroup window a compact tile-rectangle that shares an L2-sized
// weight+activation footprint (the fix for occupancy→L2-thrash, STATUS
// 2026-06-21 [[occupancy-l2-thrash-not-the-store]]).
//
// Derived from gemm_q4_0_i8_basic: 4×1 tile, de-interleaved blocks, 0-stride
// rescale broadcast, DIRECT f16-coopmat store (no LDS scratch → 9/16 occupancy,
// the kernel we want to make L2-friendly). See basic for the per-block-rescale
// and coopStore-scalar-match rationale. M is NOT baked — it flows through the
// dispatch size: dispatch [(M/64)·(N/16), 1, 1]; m_block = wg.x / NB_N.

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> weights: array<u32>;   // [N×K] Q4_0 packed
@group(0) @binding(1) var<storage, read> x: array<i8>;          // [M×K] Q8 quants
@group(0) @binding(2) var<storage, read> x_scales: array<f16>;  // [M×(K/32)] d_a
@group(0) @binding(3) var<storage, read_write> y: array<f16>;   // [M×N]

const K: u32 = 21504u;
const N: u32 = 5376u;
const M_TILES: u32 = #{M_TILES}u; // tall-thin M_TILES×1 (register-level weight reuse)
const N_COLS: u32 = 16u;          // N_TILES = 1
const M_ROWS: u32 = M_TILES * 16u;
const M_TOTAL: u32 = 256u;            // prefill chunk under test (hardcoded for the decode)
const NB: u32 = K / 32u;              // 672 32-blocks per row
const ROW_WORDS: u32 = NB * 18u / 4u; // Q4_0 words per W row
const NB_N: u32 = N / N_COLS;         // 336 N-blocks
const NB_M: u32 = M_TOTAL / M_ROWS;   // 4 M-blocks (tile grid is NB_N × NB_M)

// THE L2-SCHEDULING KNOB. Map a linear workgroup id → (n_block, m_block); the
// launch-order-consecutive ids are the co-resident window. 2D super-block walk:
// each super-block is BN_SB adjacent n-blocks × all NB_M m-blocks, ordered
// m-OUTER (for each m, sweep the BN_SB n-blocks). So the BN_SB weight strips are
// reused across all NB_M m-blocks (read once vs NB_M× from DRAM), while only ONE
// activation block X[m] is hot at a time — footprint ≈ BN_SB·(K·16 weights) + one
// 64×K activation, tuned to the 2 MB L2. BN_SB=1 is the plain transpose.
const BN_SB: u32 = #{BN_SB}u;

fn tile_index(t: u32) -> vec2<u32> {
    let per_sb = BN_SB * NB_M;        // tiles per super-block
    let sb = t / per_sb;             // super-block index along N
    let in_sb = t % per_sb;
    let m_block = in_sb / BN_SB;      // m OUTER: 0..NB_M
    let n_block = sb * BN_SB + in_sb % BN_SB;
    return vec2(n_block, m_block);    // .x = n_block, .y = m_block
}

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
    // Linear tile id — 1D dispatch [tiles,1,1] (wg.y=0 → t=wg.x) OR 2D [NB_N,NB_M,1]
    // (t = wg.y·NB_N + wg.x). tile_index decodes it to (n_block, m_block).
    let t = wg.y * NB_N + wg.x;
    let tile = tile_index(t);
    let m0 = tile.y * M_ROWS;
    let n0 = tile.x * N_COLS;

    var zero_f: coop_mat16x16<f32, C>;
    var yacc: array<coop_mat16x16<f32, C>, M_TILES>;
    for (var i = 0u; i < M_TILES; i += 1u) {
        yacc[i] = zero_f;
    }

    for (var beta = 0u; beta < NB; beta += 2u) {
        if (lid < N_COLS) {
            let wp = (n0 + lid) * ROW_WORDS + (beta / 2u) * 9u;
            var w: array<u32, 9>;
            for (var i = 0u; i < 9u; i += 1u) {
                w[i] = weights[wp + i];
            }
            dw2[lid] = unpack2x16float(w[0]).x;
            unpack_block(
                (w[0] >> 16u) | (w[1] << 16u),
                (w[1] >> 16u) | (w[2] << 16u),
                (w[2] >> 16u) | (w[3] << 16u),
                (w[3] >> 16u) | (w[4] << 16u),
                lid * 64u,
            );
            dw2[N_COLS + lid] = unpack2x16float(w[4]).y;
            unpack_block(w[5], w[6], w[7], w[8], lid * 64u + 32u);
        }
        workgroupBarrier(); // wb + dw2 written before coopLoad/rescale read them

        for (var u = 0u; u < 2u; u += 1u) {
            var zero_i: coop_mat16x16<i32, C>;
            var acc: array<coop_mat16x16<i32, C>, M_TILES>;
            for (var mt = 0u; mt < M_TILES; mt += 1u) {
                acc[mt] = zero_i;
            }
            let k0 = (beta + u) * 32u;
            for (var s = 0u; s < 2u; s += 1u) {
                let bt = coopLoad<coop_mat16x16<i8, B>>(&wb[u * 32u + s * 16u], 64u);
                for (var mt = 0u; mt < M_TILES; mt += 1u) {
                    let at = coopLoadT<coop_mat16x16<i8, A>>(&x[(m0 + mt * 16u) * K + k0 + s * 16u], K);
                    acc[mt] = coopMultiplyAdd(at, bt, acc[mt]);
                }
            }
            for (var i = lid; i < M_ROWS; i += 64u) {
                da_l[i] = f32(x_scales[(m0 + i) * NB + (beta + u)]);
            }
            workgroupBarrier(); // da_l written before the 0-stride broadcast reads it
            for (var mt = 0u; mt < M_TILES; mt += 1u) {
                let scale = coopLoad<coop_mat16x16<f32, C>>(&da_l[mt * 16u], 0u)
                    * coopLoadT<coop_mat16x16<f32, C>>(&dw2[u * N_COLS], 0u);
                yacc[mt] = yacc[mt] + scale * f32(acc[mt]);
            }
            workgroupBarrier(); // broadcast read done before next block overwrites da_l
        }
    }

    // Direct f16-coopmat store (no LDS scratch → high occupancy).
    for (var mt = 0u; mt < M_TILES; mt += 1u) {
        coopStoreT(f16(yacc[mt]), &y[(m0 + mt * 16u) * N + n0], N);
    }
}
