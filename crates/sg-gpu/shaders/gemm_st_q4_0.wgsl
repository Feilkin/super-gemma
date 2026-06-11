// Subgroup-tiled Q4_0 GEMM (plan 02): the non-coopmat prefill matmul,
// kept as the A/B baseline and fallback while the cooperative-matrix
// variant's RADV lowering underperforms.
//
// Classic register-blocked GEMM: a 256-thread workgroup computes a 64×64
// block of Y = X[M×K]·W[N×K]ᵀ; each thread owns a 4×4 f32 accumulator
// patch. Each K-step stages a 64(M)×32(K) tile of X and a dequantized
// 64(N)×32(K) tile of W in workgroup memory (8 KB), then runs 32 FMA
// sweeps. f32 accumulation, deterministic (fixed K order per thread).
//
// Constraints: M multiple of 64 (engine pads prefill chunks); every site's
// N is a multiple of 64; K a multiple of 32.

enable f16;

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f16>;
@group(0) @binding(2) var<storage, read_write> y: array<f16>;

const K: u32 = #{K_DIM}u;
const N: u32 = #{N_DIM}u;
const BLOCKS_PER_ROW: u32 = K / 32u;

var<workgroup> a_tile: array<f16, 2048>; // 64 M-rows × 32 K
var<workgroup> b_tile: array<f16, 2048>; // 64 N-rows × 32 K

@compute @workgroup_size(#{WG_X})
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let lid = lid_v.x;
    let n0 = wg_id.x * 64u;
    let m0 = wg_id.y * 64u;
    // Thread (tx, ty) computes C rows m0+4ty.. and cols n0+4tx.. .
    let tx = lid % 16u;
    let ty = lid / 16u;

    var acc: array<f32, 16>;

    // Per-step staging assignments.
    let dq_row = lid / 4u; // W row 0..64, four threads per row
    let dq_q = lid % 4u; // quarter-block: bytes 4q..4q+4

    for (var k0 = 0u; k0 < K; k0 += 32u) {
        // Stage A: 2048 f16, 8 per thread (coalesced rows).
        for (var i = 0u; i < 8u; i += 1u) {
            let e = lid * 8u + i;
            a_tile[e] = x[(m0 + e / 32u) * K + k0 + (e % 32u)];
        }
        // Stage dequantized W: one quarter Q4_0 block per thread.
        let byte_base = (n0 + dq_row) * BLOCKS_PER_ROW * 18u + (k0 / 32u) * 18u;
        let d = unpack_f16_at(byte_base);
        for (var b = 0u; b < 4u; b += 1u) {
            let byte = load_byte(byte_base + 2u + 4u * dq_q + b);
            b_tile[dq_row * 32u + 4u * dq_q + b] = f16((f32(byte & 0xFu) - 8.0) * d);
            b_tile[dq_row * 32u + 16u + 4u * dq_q + b] = f16((f32(byte >> 4u) - 8.0) * d);
        }
        workgroupBarrier();

        for (var kk = 0u; kk < 32u; kk += 1u) {
            var a_reg: array<f32, 4>;
            var b_reg: array<f32, 4>;
            for (var i = 0u; i < 4u; i += 1u) {
                a_reg[i] = f32(a_tile[(ty * 4u + i) * 32u + kk]);
                b_reg[i] = f32(b_tile[(tx * 4u + i) * 32u + kk]);
            }
            for (var mi = 0u; mi < 4u; mi += 1u) {
                for (var ni = 0u; ni < 4u; ni += 1u) {
                    acc[mi * 4u + ni] += a_reg[mi] * b_reg[ni];
                }
            }
        }
        workgroupBarrier();
    }

    for (var mi = 0u; mi < 4u; mi += 1u) {
        for (var ni = 0u; ni < 4u; ni += 1u) {
            y[(m0 + ty * 4u + mi) * N + n0 + tx * 4u + ni] = f16(acc[mi * 4u + ni]);
        }
    }
}

fn unpack_f16_at(byte_offset: u32) -> f32 {
    let word = weights[byte_offset / 4u];
    let half_bits = select(word & 0xFFFFu, word >> 16u, (byte_offset & 2u) != 0u);
    return unpack2x16float(half_bits).x;
}

fn load_byte(byte_offset: u32) -> u32 {
    return (weights[byte_offset / 4u] >> (8u * (byte_offset % 4u))) & 0xFFu;
}
