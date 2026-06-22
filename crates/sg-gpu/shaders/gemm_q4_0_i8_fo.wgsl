// FULL-OCCUPANCY int8-MMQ GEMM — FFN-down shape ONLY (K=21504, N=5376).
// Y[M×N] = X[M×K]·W[N×K]ᵀ. One wave (WG=64) per output tile; plain 2D dispatch
// [M-blocks, N-blocks] (no swizzle — this is the large-K case, weight reuse is
// the question we're probing, not L2 scheduling).
//
// THE BET. The deployed 4×1 kernel is a coherent LOW-occupancy design: it spends
// VGPRs (168) on a β×2 MMA interleave + software weight prefetch to hide DRAM
// latency via *in-wave* ILP, sitting at ~3 waves/SIMD (RGP: latency-bound,
// ~46/256 GB/s). This kernel is the opposite philosophy: shrink the per-wave
// footprint as far as it goes so MANY waves/SIMD co-reside, and let the hardware
// hide the same latency by WAVE-SWITCHING instead of in-wave ILP.
//
// Why this isn't the dead `gemm_q4_0_i8_occ` (1×1, −47.8%): that kernel had the
// footprint idea right but kept the machinery that DEFEATS it — an LDS scale
// `stage` + an LDS epilogue round-trip + their barriers, which serialize the
// waves at the barrier (every wave stalls together → no overlap → VMEM-in-flight
// actually DROPPED to 4.6% vs deployed's 8.5%). Here the two levers Ada called
// out remove exactly that machinery:
//   - 0-STRIDE SCALE rescale: the per-block d_a·d_w outer product is built on the
//     fly from two 0-stride coopLoad broadcasts (no LDS `stage`, no barriers).
//   - DIRECT coopStore EPILOGUE: f16(yacc) coop-matrix stored straight to `y`
//     (scalars match → valid), no LDS scratch, no epilogue barrier.
// What's left in LDS is just the unpacked weight tile `wb` (1 KB) + the two tiny
// scale-source rows. There are NO workgroupBarriers: WG=64 is a single wave per
// workgroup, so the in-wave LDS RAW (wb/dw2/da_l write → coopLoad read) is covered
// by the s_waitcnt ACO inserts — the barriers were verified bit-exact-redundant
// here ([[single-wave-workgroupbarrier-redundant]]).
//
// THE EXPERIMENT. Default config is PURE occupancy: small tile (M_TILES=2),
// NO β×2 ILP, NO prefetch — the leanest VGPR footprint. The open question
// ([[prefill-gemm-is-mlp-bound-not-byte-bound]]: the binding stall is the
// X-before-WMMA vmcnt) is whether enough co-resident waves hide that X stall
// *without* the explicit activation prefetch the low-occupancy l2 kernel needed
// (axp4 ping-pong, +5%). Knobs let the sweep walk the occupancy↔ILP frontier:
//   M_TILES — tile height in 16-row units (1/2/4): SMALLER = fewer accumulator
//             VGPRs = more waves, but less register weight reuse (each n-strip is
//             re-read by more m-blocks — the DRAM cost occupancy must out-hide).
//   B2      — β×2 WMMA interleave (in-wave ILP). Expected to REGRESS once
//             occupancy is high ([[rdna3-wmma-latency-hidden-by-ilp-not-occupancy]]);
//             the A/B confirms it.
//   PD      — single-ahead weight prefetch (1 = hold next pair's words, issue its
//             load a block ahead). Cheap MLP; weights weren't the binding stall in
//             l2, so expected ~flat, but cheap to test at this footprint.

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> weights: array<u32>;   // [N×K] Q4_0 packed
@group(0) @binding(1) var<storage, read> x: array<i8>;          // [M×K] Q8 quants
@group(0) @binding(2) var<storage, read> x_scales: array<f16>;  // [M×(K/32)] d_a
@group(0) @binding(3) var<storage, read_write> y: array<f16>;   // [M×N]

const K: u32 = 21504u;
const N: u32 = 5376u;
const M_TILES: u32 = #{M_TILES}u; // tile height in 16-row units (the occupancy knob)
const N_COLS: u32 = 16u;          // N_TILES = 1 (tall-thin)
const M_ROWS: u32 = M_TILES * 16u;
const NB: u32 = K / 32u;              // 672 32-blocks per row
const ROW_WORDS: u32 = NB * 18u / 4u; // Q4_0 words per W row (NB even → exact)

// β×2 WMMA-ILP: 0 = process the pair's two blocks sequentially (one accumulator
// set); 1 = interleave both blocks' MMAs (independent → hide each other's
// latency) at the cost of a 2nd accumulator set (VGPR → fewer waves).
const B2: u32 = #{B2}u;
// Weight prefetch depth: 0 = inline load the pair's 9 words then unpack; 1 =
// software pipeline — hold the NEXT pair's 9 words in registers, issue their load
// a block-pair ahead so the DRAM latency hides behind this pair's unpack+MMA.
const PD: u32 = #{PD}u;
// Activation-SCALE hoist. The per-block d_a (an f16 in x_scales → a 16-bit
// buffer_load_d16_b16) is, per RGP, issued right before the rescale with nothing
// to cover it → a full memory-latency stall (~2K clk, same as the 128-bit X
// loads — latency is size-independent). 1 = hoist BOTH blocks' d_a global loads
// to the top of the β-iteration (into registers, before the weight load) so their
// latency overlaps the dequant + WMMAs; the rescale then only does the cheap LDS
// store + 0-stride broadcast. 0 = load inline in the rescale.
const SXP: u32 = #{SXP}u;

// Unpacked weight tile (one block-pair: 16 N-rows × 64 K, cols 0..32 = block β,
// 32..64 = block β+1) + the pair's two d_w per col + the current block's d_a per
// row. With 0-stride scales + direct coopStore, this is the ONLY LDS the kernel
// uses — keeping it small is what lets the waves stack.
var<workgroup> wb: array<i8, N_COLS * 64u>;
var<workgroup> dw2: array<f32, 2u * N_COLS>;
var<workgroup> da_l: array<f32, M_ROWS>;

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

// Unpack a block-pair's 9 words (held in registers) into wb + dw2 for this lane's
// N-row. Shared by the inline and prefetch paths.
fn unpack_pair(w: array<u32, 9>, lid: u32) {
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

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let lid = lid_v.x;
    // Plain [M-blocks, N-blocks] dispatch — wg.x = m_block, wg.y = n_block.
    let m0 = wg.x * M_ROWS;
    let n0 = wg.y * N_COLS;

    var zero_f: coop_mat16x16<f32, C>;
    var yacc: array<coop_mat16x16<f32, C>, M_TILES>;
    for (var i = 0u; i < M_TILES; i += 1u) {
        yacc[i] = zero_f;
    }

    // PD=1 prologue: prefetch block-pair 0's 9 words into registers.
    var w_next: array<u32, 9>;
    if (PD == 1u && lid < N_COLS) {
        let wp = (n0 + lid) * ROW_WORDS;
        for (var i = 0u; i < 9u; i += 1u) {
            w_next[i] = weights[wp + i];
        }
    }

    for (var beta = 0u; beta < NB; beta += 2u) {
        // SXP=1: hoist both blocks' d_a (activation scale) global loads to the top
        // of the iteration — issued here (batched with the weight loads), consumed
        // only in the rescale after the WMMAs, so the ~2K-clk memory latency hides
        // behind the dequant + MMA instead of stalling the rescale. M_ROWS ≤ 64 =
        // WG, so each lane owns ≤ 1 row → two scalar registers.
        var da_r0: f32;
        var da_r1: f32;
        if (SXP == 1u && lid < M_ROWS) {
            da_r0 = f32(x_scales[(m0 + lid) * NB + beta]);
            da_r1 = f32(x_scales[(m0 + lid) * NB + beta + 1u]);
        }

        // Get this pair's 9 words — from the prefetch ring (PD=1) or inline (PD=0)
        // — then issue the NEXT pair's prefetch before consuming (so its DRAM
        // round-trip overlaps this pair's unpack + MMA).
        var w: array<u32, 9>;
        if (PD == 1u) {
            // Consume the prefetched pair (element-wise copy — a whole-array `w =
            // w_next` miscompiles here), then issue the NEXT pair's load so its
            // round-trip hides behind this pair's unpack + MMA + rescale.
            for (var i = 0u; i < 9u; i += 1u) {
                w[i] = w_next[i];
            }
            if (lid < N_COLS && beta + 2u < NB) {
                let wp = (n0 + lid) * ROW_WORDS + ((beta + 2u) / 2u) * 9u;
                for (var i = 0u; i < 9u; i += 1u) {
                    w_next[i] = weights[wp + i];
                }
            }
        } else if (lid < N_COLS) {
            let wp = (n0 + lid) * ROW_WORDS + (beta / 2u) * 9u;
            for (var i = 0u; i < 9u; i += 1u) {
                w[i] = weights[wp + i];
            }
        }
        if (lid < N_COLS) {
            unpack_pair(w, lid);
        }

        var zero_i: coop_mat16x16<i32, C>;
        if (B2 == 1u) {
            // Interleave the pair's two blocks' MMAs (independent → WMMA ILP).
            var acc0: array<coop_mat16x16<i32, C>, M_TILES>;
            var acc1: array<coop_mat16x16<i32, C>, M_TILES>;
            for (var mt = 0u; mt < M_TILES; mt += 1u) {
                acc0[mt] = zero_i;
                acc1[mt] = zero_i;
            }
            let k0 = beta * 32u;
            for (var s = 0u; s < 2u; s += 1u) {
                let bt0 = coopLoad<coop_mat16x16<i8, B>>(&wb[s * 16u], 64u);
                let bt1 = coopLoad<coop_mat16x16<i8, B>>(&wb[32u + s * 16u], 64u);
                for (var mt = 0u; mt < M_TILES; mt += 1u) {
                    let a0 = coopLoadT<coop_mat16x16<i8, A>>(&x[(m0 + mt * 16u) * K + k0 + s * 16u], K);
                    let a1 = coopLoadT<coop_mat16x16<i8, A>>(&x[(m0 + mt * 16u) * K + k0 + 32u + s * 16u], K);
                    acc0[mt] = coopMultiplyAdd(a0, bt0, acc0[mt]);
                    acc1[mt] = coopMultiplyAdd(a1, bt1, acc1[mt]);
                }
            }
            // Rescale both blocks into yacc (0-stride d_a·d_w broadcast, no LDS stage).
            for (var u = 0u; u < 2u; u += 1u) {
                if (SXP == 1u) {
                    if (lid < M_ROWS) { da_l[lid] = select(da_r1, da_r0, u == 0u); }
                } else {
                    for (var i = lid; i < M_ROWS; i += 64u) {
                        da_l[i] = f32(x_scales[(m0 + i) * NB + (beta + u)]);
                    }
                }
                for (var mt = 0u; mt < M_TILES; mt += 1u) {
                    let scale = coopLoad<coop_mat16x16<f32, C>>(&da_l[mt * 16u], 0u)
                        * coopLoadT<coop_mat16x16<f32, C>>(&dw2[u * N_COLS], 0u);
                    if (u == 0u) {
                        yacc[mt] = yacc[mt] + scale * f32(acc0[mt]);
                    } else {
                        yacc[mt] = yacc[mt] + scale * f32(acc1[mt]);
                    }
                }
            }
        } else {
            // Sequential: one accumulator set, the pair's two blocks in turn.
            for (var u = 0u; u < 2u; u += 1u) {
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
                if (SXP == 1u) {
                    if (lid < M_ROWS) { da_l[lid] = select(da_r1, da_r0, u == 0u); }
                } else {
                    for (var i = lid; i < M_ROWS; i += 64u) {
                        da_l[i] = f32(x_scales[(m0 + i) * NB + (beta + u)]);
                    }
                }
                for (var mt = 0u; mt < M_TILES; mt += 1u) {
                    let scale = coopLoad<coop_mat16x16<f32, C>>(&da_l[mt * 16u], 0u)
                        * coopLoadT<coop_mat16x16<f32, C>>(&dw2[u * N_COLS], 0u);
                    yacc[mt] = yacc[mt] + scale * f32(acc[mt]);
                }
            }
        }
    }

    // Direct f16-coopmat store (no LDS scratch → high occupancy).
    for (var mt = 0u; mt < M_TILES; mt += 1u) {
        coopStoreT(f16(yacc[mt]), &y[(m0 + mt * 16u) * N + n0], N);
    }
}
