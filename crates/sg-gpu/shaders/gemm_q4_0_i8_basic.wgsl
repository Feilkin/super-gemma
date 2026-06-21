// Clean, no-frills 4×1 int8 MMQ GEMM — a READABLE BASELINE + barrier probe.
// Y[M×N] = X[M×K]·W[N×K]ᵀ, one wave (64 lanes) computing a 64×16 output block.
//
// Deliberately strips EVERY optimization from the deployed `gemm_q4_0_i8`:
//   - no β×2 interleave (one accumulator, two blocks processed in sequence);
//   - no weight prefetch (load the block-pair inline);
//   - no swizzle (plain [N-blocks, M-blocks] dispatch);
//   - no stage double-buffer / no LDS epilogue (coopStore straight to y — this
//     is a standalone probe, not graph-wired, so the auto-sync LDS round-trip
//     the deployed kernel needs for reflection visibility is irrelevant here).
// Weights are still read in aligned 9-word PAIRS: an 18-byte Q4_0 block is not
// word-aligned, so the pair (36 B = 9 words = 2 blocks) is the natural read unit
// — alignment, not an optimization. Per-block rescale uses the 0-stride coopLoad
// broadcast (no LDS scale materialization).
//
// BARRIER PROBE: the three workgroupBarriers are independent toggles. They were
// inherited from the first GEMM kernel and never tested. Drop one (set its const
// to 0) and re-run parity to learn whether it's actually required:
//   BAR_WB  — RAW: `wb`/`dw2` written by the unpack → read by coopLoad/rescale.
//   BAR_DA  — RAW: `da_l` written → read by the 0-stride broadcast.
//   BAR_WAR — WAR: those LDS reads done → next block/pair overwrites da_l/wb.

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> weights: array<u32>;   // [N×K] Q4_0 packed
@group(0) @binding(1) var<storage, read> x: array<i8>;          // [M×K] Q8 quants
@group(0) @binding(2) var<storage, read> x_scales: array<f16>;  // [M×(K/32)] d_a
@group(0) @binding(3) var<storage, read_write> y: array<f16>;   // [M×N]

const K: u32 = #{K_DIM}u;
const N: u32 = #{N_DIM}u;
const WG: u32 = #{WG_X}u;
const BAR_WB: u32 = #{BAR_WB}u;
const BAR_DA: u32 = #{BAR_DA}u;
const BAR_WAR: u32 = #{BAR_WAR}u;
// Epilogue. 0 = coopStore f32 tiles to an LDS scratch, convert f32→f16 on the
// normal-store copy. 1 = convert each tile to an f16 coop-matrix in registers
// (`f16(yacc)`) and coopStore straight to y — scalar now matches f16 y, so no LDS
// scratch and no epilogue barrier.
const EPI: u32 = #{EPI}u;
// Tiles staged in `ystg` at once (EPI==0 path). M_TILES (default) = one shot, the
// 4096 B scratch that caps occupancy at 5/16. 1 = stage one tile, coalesced-write
// it, reuse the 1024 B scratch over M_TILES passes → smaller LDS → HIGHER occupancy
// with the SAME coalesced store pattern. The isolation probe: does raising
// occupancy (less LDS) alone reproduce basic_dir's +21% DRAM traffic, or was it the
// direct store? (STATUS 2026-06-21.)
const EPI_TILES: u32 = #{EPI_TILES}u;

const M_TILES: u32 = 4u;          // tall-thin 4×1
const N_COLS: u32 = 16u;          // N_TILES = 1
const M_ROWS: u32 = 64u;          // M_TILES * 16
const NB: u32 = K / 32u;              // 32-blocks per row
const ROW_WORDS: u32 = NB * 18u / 4u; // Q4_0 words per W row (NB even → exact)

// One block PAIR's unpacked i8 weights: [N_COLS rows × 64 K] row-major
// (cols 0..32 = block β, 32..64 = block β+1). Plus the pair's two d_w per col
// and the current block's d_a per row.
var<workgroup> wb: array<i8, N_COLS * 64u>;
var<workgroup> dw2: array<f32, 2u * N_COLS>;
var<workgroup> da_l: array<f32, M_ROWS>;
// Epilogue scratch (f32): coopStore requires the destination scalar to EQUAL the
// accumulator scalar (naga fork valid/function.rs CooperativeStore: ptr_scalar ==
// target_scalar), so an f32 accumulator can't coopStore into the f16 `y` directly.
// Stage f32 here, then convert f32→f16 on the normal-store copy. (coopLoad has no
// such check — valid/expression.rs only requires the pointer base to be a scalar.)
var<workgroup> ystg: array<f32, EPI_TILES * 256u>;

// Unpack one Q4_0 block (4 qs words) to i8 (q−8) at wb[base..+32]: low nibble of
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
    let m0 = wg.y * M_ROWS;
    let n0 = wg.x * N_COLS;

    var zero_f: coop_mat16x16<f32, C>;
    var yacc: array<coop_mat16x16<f32, C>, M_TILES>;
    for (var i = 0u; i < M_TILES; i += 1u) {
        yacc[i] = zero_f;
    }

    for (var beta = 0u; beta < NB; beta += 2u) {
        // Load + unpack this aligned block-pair into wb; write the two d_w scales.
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
        if (BAR_WB == 1u) { workgroupBarrier(); }

        // Process the pair's two blocks in sequence (no interleave).
        for (var u = 0u; u < 2u; u += 1u) {
            var zero_i: coop_mat16x16<i32, C>;
            var acc: array<coop_mat16x16<i32, C>, M_TILES>;
            for (var mt = 0u; mt < M_TILES; mt += 1u) {
                acc[mt] = zero_i;
            }
            let k0 = (beta + u) * 32u;
            // The 32-block is two 16-wide K-halves (the WMMA K-dim is 16).
            for (var s = 0u; s < 2u; s += 1u) {
                let bt = coopLoad<coop_mat16x16<i8, B>>(&wb[u * 32u + s * 16u], 64u);
                for (var mt = 0u; mt < M_TILES; mt += 1u) {
                    let at = coopLoadT<coop_mat16x16<i8, A>>(&x[(m0 + mt * 16u) * K + k0 + s * 16u], K);
                    acc[mt] = coopMultiplyAdd(at, bt, acc[mt]);
                }
            }

            // Rescale block (β+u) into yacc: scale[i][j] = d_a[row m0+mt·16+j] ·
            // d_w[col n0+i], built on the fly with 0-stride coopLoad broadcasts.
            for (var i = lid; i < M_ROWS; i += WG) {
                da_l[i] = f32(x_scales[(m0 + i) * NB + (beta + u)]);
            }
            if (BAR_DA == 1u) { workgroupBarrier(); }
            for (var mt = 0u; mt < M_TILES; mt += 1u) {
                let scale = coopLoad<coop_mat16x16<f32, C>>(&da_l[mt * 16u], 0u)
                    * coopLoadT<coop_mat16x16<f32, C>>(&dw2[u * N_COLS], 0u);
                yacc[mt] = yacc[mt] + scale * f32(acc[mt]);
            }
            if (BAR_WAR == 1u) { workgroupBarrier(); }
        }
    }

    // Epilogue (transpose-store matches the coopMultiplyAdd element layout — out
    // row mt·16+e/16, col e%16).
    if (EPI == 1u) {
        // Convert f32 tile → f16 coop-matrix in registers, coopStore straight to
        // the f16 y (scalars match → valid; no LDS scratch / barrier).
        for (var mt = 0u; mt < M_TILES; mt += 1u) {
            coopStoreT(f16(yacc[mt]), &y[(m0 + mt * 16u) * N + n0], N);
        }
    } else {
        // Stage EPI_TILES tiles in `ystg`, coalesced-write them to y, reuse the
        // scratch over M_TILES/EPI_TILES passes. EPI_TILES==M_TILES → one pass (the
        // 4096 B scratch); EPI_TILES==1 → M_TILES passes of a 1024 B scratch (same
        // coalesced store, less LDS → higher occupancy — the isolation probe).
        for (var t0 = 0u; t0 < M_TILES; t0 += EPI_TILES) {
            for (var j = 0u; j < EPI_TILES; j += 1u) {
                coopStoreT(yacc[t0 + j], &ystg[j * 256u], 16u);
            }
            workgroupBarrier(); // ystg written by coopStore before the normal-store reads
            for (var i = lid; i < EPI_TILES * 256u; i += WG) {
                let j = i / 256u;
                let e = i % 256u;
                let row = m0 + (t0 + j) * 16u + e / 16u;
                let col = n0 + e % 16u;
                y[row * N + col] = f16(ystg[i]);
            }
            workgroupBarrier(); // coalesced reads done before next pass overwrites ystg
        }
    }
}
