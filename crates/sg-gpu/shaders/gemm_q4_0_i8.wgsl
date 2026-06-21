// int8-MMQ GEMM, Q4_0 weights × Q8_0 activations (plan 02, profile rank #2).
// Y[M×N] = X[M×K] · W[N×K]ᵀ via SIGNED int8 cooperative matrices (enabled by
// the naga fork — docs/naga-int8-coopmat-patch.md). The MMQ alternative to
// the f16 `gemm_q4_0`: each 32-element block's dot is an EXACT i8×i8→i32
// cooperative-matrix product, rescaled to f32 by the block's `d_a·d_w`.
//
// WEIGHTS read Q4_0 PACKED, like the f16 gemm (apples-to-apples — and the
// deployable form: no weight repack, no 2× memory). Each thread bulk-loads its
// W row's aligned 9-word block PAIR (= 64 weights = the β-unroll's two blocks),
// unpacks the nibbles to i8 (q−8) in an LDS tile (`wb`) and writes the two
// per-block `d_w` scales — folding away the separate w_scales buffer. The MMA
// then `coopLoad`s the int8 B tiles straight from `wb` (LDS i8 coopmat verified
// by coop_i8_lds_smoke). Activations stay pre-quantized Q8_0 (a separate
// `kv_quant_q8` pass over X — small and amortized across a layer's gemms).
//
// One wave-sized workgroup computes an (M_TILES·16)×(N_TILES·16) block of Y.
// PER-BLOCK RESCALE (the crux): the (d_a[m]·d_w[n]) scale is per-32-block AND
// an outer product over the tile, so a single i32 accumulation across all K is
// impossible — every block's i32 dot must be pulled out, converted to f32 and
// scaled before it can be summed into Y. The f32 output `yacc` is
// register-resident across all blocks (a `coop_mat<f32,C>` array, zeroed once);
// the scale fragment is applied with component-wise coopmat ops (`yacc += scale *
// f32(acc)`: OpConvertSToF + OpFMul + OpFAdd, per-lane). It is built either in LDS
// (BCAST=0, `stage` outer product) or on the fly via 0-stride coopLoad broadcasts
// (BCAST=1) — see the `const BCAST` doc; deployed BCAST=1 on the K=5376 shapes
// (+4..10% e2e prefill, bench: sg-bench profile, perf=high; STATUS 2026-06-20).
//
// ISA-guided tuning + reading Q4_0 directly: ~11 TFLOPS at 2×2 ≈ 1.03× f16
// (bench: mmq_tflops, perf=high; STATUS 2026-06-14). 2×2 is the production
// tiling (the i8 weight-staging shifted the sweet spot off 2×4). The steps,
// each from RADV asm/shaderstats:
//   1. SCALE BUILD: build the row/col scale VECTORS in LDS (`da_l`, `dw{a,b}`)
//      and form `stage` = their outer product from LDS — not the full product
//      from global (which was ~64 redundant loads + ~340 address ops/thread).
//   2. β-LOOP UNROLLED ×2: two independent blocks' MMAs interleaved so WMMA ILP
//      covers the per-block substep0→substep1 stall — RDNA3 hides WMMA latency
//      via within-wave ILP, not occupancy.
//   3. The outer-product fill is ×4-unrolled so independent `da_l` loads
//      pipeline instead of one load→lgkmcnt(0)→mul→store per element.
// `stage` is double- or single-buffered per `STAGE_BUFS` (double overlaps the
// rescale on small-K shapes; single frees LDS for occupancy on large-K). The
// residual binder is the scale's LDS round-trip, forced by coopmat1 having no
// fragment-element access (can't scale the accumulator in registers).
//
// Element layout: `coopMultiplyAdd(A=Xᵀ-loaded, B=Wᵀ)` gives `acc[i][j]` = the
// dot for output row (tile_m+j), col (tile_n+i) — the transpose of the output
// tile (derived from the prior kernel's parity-green coopStoreT path). So the
// scale, laid out [M_ROWS×N_COLS] row-major as `d_a[m]·d_w[n]` and loaded with
// `coopLoadT`, lines up element-for-element with `acc`; the epilogue stages
// `yacc` to LDS and writes y via normal stores (coopStore to y is invisible to
// auto-sync — see the epilogue).
//
// naga_oil corrupts coopmat IR → raw: true. A `var` coop-mat re-declared in a
// loop is NOT re-zeroed (naga/ACO keeps the registers live) → `acc[]`/`yacc[]`
// are declared once and explicitly seeded from `zero_i`/`zero_f`.

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> weights: array<u32>;   // [N×K] Q4_0 packed (18 B/block)
@group(0) @binding(1) var<storage, read> x: array<i8>;          // [M×K] Q8_0 quants
@group(0) @binding(2) var<storage, read> x_scales: array<f16>;  // [M×(K/32)] d_a
@group(0) @binding(3) var<storage, read_write> y: array<f16>;   // [M×N]

const K: u32 = #{K_DIM}u;
const N: u32 = #{N_DIM}u;
const WG: u32 = #{WG_X}u;
const M_TILES: u32 = #{M_TILES}u;
const N_TILES: u32 = #{N_TILES}u;
// `stage` buffering: 2 = double (one pass's coopLoad overlaps the next pass's
// fill — wins on small-K rescale-bound shapes); 1 = single (halves the LDS →
// more waves, wins on large-K shapes that are vmcnt-stalled with LDS-capped
// occupancy — RGP 2026-06-14). Identical barriers either way.
const STAGE_BUFS: u32 = #{STAGE_BUFS}u;
// Workgroup→tile mapping (see gemm_q4_0.wgsl): 0 = x:N, y:M; 1 = x:M, y:N so one
// N-strip's M-blocks launch consecutively and its weight strip stays hot in L2
// (prefill is weight-traffic-bound — RGP 2026-06-14). Dispatch transposed at 1.
const SWIZZLE: u32 = #{SWIZZLE}u;
// Rescale path. 1 = build the per-block scale fragment on the fly from the row/col
// scale VECTORS via 0-stride coopLoad broadcasts (no LDS `stage` fill); 0 = form
// the [M_ROWS×N_COLS] outer product in `stage` and coopLoad it. Per-shape A/B
// (mmq_tflops, perf=high): 0-stride wins K=5376 (−4..−16%, rescale-bound) but
// loses large-K (+2..+12%, weight-bound — the freed LDS/occupancy oversubscribes
// the weight stream). build.rs sets BCAST=1 on the K=5376 variants, 0 elsewhere.
const BCAST: u32 = #{BCAST}u;
// Epilogue. 0 = stage yacc (f32) to LDS then write y via reflection-visible normal
// stores (the deployed path — a direct coopStore to y is invisible to vulkano's
// auto-sync and would race the graph consumer). 1 = convert yacc→f16 coopmat in
// registers and coopStoreT straight to y (no LDS round-trip, like the l2/basic_dir
// kernels). EPI=1 is an A/B of the epilogue mechanism ONLY — NOT graph-safe without a
// manual barrier; and at BCAST=0 it doesn't free LDS (the rescale still needs `stage`).
const EPI: u32 = #{EPI}u;
// Software-prefetch depth on the WEIGHT stream (the cold DRAM read; X is the
// cache-hot one). 1 = the deployed pipeline (load pair β+2 while computing β);
// 2 keeps two weight loads outstanding per wave, adding memory-level parallelism
// to hide DRAM latency WITHOUT raising occupancy — the lever for a kernel that's
// memory-LATENCY-bound (runs +32% over its all-DRAM floor at 3/16 occupancy),
// not bandwidth-bound (MALL probe, STATUS 2026-06-21). Costs +9 VGPR; whether
// that drops a wave is the A/B. Default 1 (build.rs) → the PD>=2 paths const-fold
// away, leaving the deployed variants byte-identical. ONLY 1 or 2 are valid: this
// kernel carries one extra buffer (`w_next2`); depth 3 needs a `w_next3` + a
// 3-pair prologue, else pairs get skipped.
const PD: u32 = #{PREFETCH_DEPTH}u;
// Activation prefetch. 0 = the deployed inline load (one `coopLoadT` of X per
// consuming WMMA — zero issue-distance, so the X-load latency lands on the WMMA;
// this is the `vmcnt` half of the pre-first-WMMA stall, RGP 2026-06-21). 1 =
// hoist all M_TILES A-tile loads for the s-step ahead of the MMA loop (proven
// moot — ACO already schedules them there, and the workgroupBarrier fences them).
// 2 = issue EVERY A-load for the β before the unpack + barrier (the A tiles
// depend only on k0/m0, not `wb`), so the X-latency overlaps the unpack, barrier
// and B-load instead of landing on the first WMMA. Costs 4·M_TILES tiny i8 A
// fragments held across the barrier. Default 0 (build.rs) → deployed byte-identical.
const AXPF: u32 = #{AXPF}u;

const NB: u32 = K / 32u;            // 32-blocks per row
const ACC: u32 = M_TILES * N_TILES; // 16×16 output tiles per workgroup
const M_ROWS: u32 = M_TILES * 16u;
const N_COLS: u32 = N_TILES * 16u;
const TILE_ELEMS: u32 = ACC * 256u;
const ROW_WORDS: u32 = NB * 18u / 4u; // Q4_0 words per W row (NB even → exact)

// Unpacked int8 weights for the current 64-K block pair: [N_COLS rows × 64 K]
// row-major (cols 0..32 = block β, 32..64 = block β+1).
var<workgroup> wb: array<i8, N_COLS * 64u>;
// The two blocks' per-col d_w scales (block β in [0,N_COLS), β+1 in
// [N_COLS,2·N_COLS)), extracted from the Q4_0 blocks — combined into one array so
// the BCAST=1 rescale can take a pointer per block (`select` can't pick between
// two array vars; the BCAST=0 path indexes it as the old dwa/dwb).
var<workgroup> dw2: array<f32, 2u * N_COLS>;
// The block's per-row d_a scales, plus (BCAST=0) the [M_ROWS×N_COLS] outer-product
// scale `stage` (STAGE_BUFS-buffered; buffer 0 is also the f32→f16 epilogue
// scratch). BCAST=1 builds the scale fragment on the fly and uses `stage` ONLY for
// the epilogue → set STAGE_BUFS=1 on those variants so it isn't double-sized.
var<workgroup> da_l: array<f32, M_ROWS>;
var<workgroup> stage: array<f32, STAGE_BUFS * TILE_ELEMS>;

// Unpack one Q4_0 block (4 qs words in registers) to i8 (q−8) at wb[base..+32].
// Low nibble of byte j → position j, high nibble → position 16+j (the
// BlockQ4_0 logical order the reference pins).
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
    let m_blk = select(wg.y, wg.x, SWIZZLE == 1u);
    let n_blk = select(wg.x, wg.y, SWIZZLE == 1u);
    let m0 = m_blk * M_ROWS; // output row block base
    let n0 = n_blk * N_COLS; // output col block base

    var zero_i: coop_mat16x16<i32, C>;
    var zero_f: coop_mat16x16<f32, C>;
    var acc: array<coop_mat16x16<i32, C>, ACC>;
    var acc2: array<coop_mat16x16<i32, C>, ACC>; // 2nd block (β-loop unrolled ×2)
    var yacc: array<coop_mat16x16<f32, C>, ACC>; // register-resident output
    for (var i = 0u; i < ACC; i += 1u) {
        yacc[i] = zero_f;
    }

    // Prefetch the first PD block-pairs' 9 weight words into registers (the
    // prologue of the software pipeline; each thread owns one W row's words).
    // Loading PD pairs ahead keeps PD weight loads outstanding so DRAM latency
    // hides behind the MMA + rescale instead of stalling the WMMAs on vmcnt —
    // the lever at this kernel's low (3-wave) occupancy. `w_next` is the pair
    // consumed next; `w_next2` the one after (dead-stripped when PD==1).
    var w_next: array<u32, 9>;
    var w_next2: array<u32, 9>;
    if (lid < N_COLS) {
        let wb0 = (n0 + lid) * ROW_WORDS;
        for (var i = 0u; i < 9u; i += 1u) {
            w_next[i] = weights[wb0 + i];
        }
        if (PD >= 2u) {
            for (var i = 0u; i < 9u; i += 1u) {
                w_next2[i] = weights[wb0 + 9u + i]; // pair 1
            }
        }
    }

    // β-loop unrolled ×2 = one Q4_0 block PAIR (64 K) per iteration.
    for (var beta = 0u; beta < NB; beta += 2u) {
        for (var i = 0u; i < ACC; i += 1u) {
            acc[i] = zero_i;
            acc2[i] = zero_i;
        }
        // Consume this pair's prefetched words, then ISSUE a load PD pairs ahead
        // (its DRAM latency hides behind this + the next PD-1 iterations' MMA +
        // rescale; the vmcnt wait lands PD iterations later). Each thread unpacks
        // its 9-word block pair into `wb`. PD==1: reload `w_next` with pair β+2.
        // PD>=2: shift `w_next2`→`w_next` and reload `w_next2` with pair β+2·PD.
        let w_cur = w_next;
        if (lid < N_COLS) {
            if (PD >= 2u) {
                w_next = w_next2;
                if (beta + 2u * PD < NB) {
                    let wbn = (n0 + lid) * ROW_WORDS + ((beta + 2u * PD) / 2u) * 9u;
                    for (var i = 0u; i < 9u; i += 1u) {
                        w_next2[i] = weights[wbn + i];
                    }
                }
            } else if (beta + 2u < NB) {
                let wbn = (n0 + lid) * ROW_WORDS + ((beta + 2u) / 2u) * 9u;
                for (var i = 0u; i < 9u; i += 1u) {
                    w_next[i] = weights[wbn + i];
                }
            }
        }
        // [AXPF==2] CROSS-BARRIER activation prefetch: A tiles depend only on
        // k0/m0 (not on `wb`), so issue every A `coopLoadT` for this β BEFORE the
        // unpack + barrier — the VMEM latency then overlaps the unpack, the
        // workgroupBarrier and the B coopLoad, instead of landing on the first
        // WMMA (the fenced `vmcnt` stall). Held across the barrier in [s·M_TILES+mt].
        // coopLoadT is uniform (all lanes) → must sit outside the unpack's
        // `lid < N_COLS` divergence. Unused/stripped at AXPF 0/1.
        let k0 = beta * 32u;
        var ap0: array<coop_mat16x16<i8, A>, 2u * M_TILES>;
        var ap1: array<coop_mat16x16<i8, A>, 2u * M_TILES>;
        if (AXPF == 2u) {
            for (var s = 0u; s < 2u; s += 1u) {
                for (var mt = 0u; mt < M_TILES; mt += 1u) {
                    ap0[s * M_TILES + mt] = coopLoadT<coop_mat16x16<i8, A>>(&x[(m0 + mt * 16u) * K + k0 + s * 16u], K);
                    ap1[s * M_TILES + mt] = coopLoadT<coop_mat16x16<i8, A>>(&x[(m0 + mt * 16u) * K + k0 + 32u + s * 16u], K);
                }
            }
        }
        if (lid < N_COLS) {
            // Block β: d in w_cur[0].lo, 16 qs bytes spanning w_cur[0].hi..[4].lo.
            dw2[lid] = unpack2x16float(w_cur[0]).x;
            unpack_block(
                (w_cur[0] >> 16u) | (w_cur[1] << 16u),
                (w_cur[1] >> 16u) | (w_cur[2] << 16u),
                (w_cur[2] >> 16u) | (w_cur[3] << 16u),
                (w_cur[3] >> 16u) | (w_cur[4] << 16u),
                lid * 64u,
            );
            // Block β+1: d in w_cur[4].hi, qs in w_cur[5..8].
            dw2[N_COLS + lid] = unpack2x16float(w_cur[4]).y;
            unpack_block(w_cur[5], w_cur[6], w_cur[7], w_cur[8], lid * 64u + 32u);
        }
        workgroupBarrier(); // wb + dwa/dwb written before MMA / rescale read them

        // Interleaved MMA for the two blocks (independent → WMMA ILP). B tiles
        // come from `wb` (LDS i8); A tiles from X (global i8). Block β occupies
        // wb cols 0..32, block β+1 cols 32..64; row stride is 64.
        for (var s = 0u; s < 2u; s += 1u) {
            var b0: array<coop_mat16x16<i8, B>, N_TILES>;
            var b1: array<coop_mat16x16<i8, B>, N_TILES>;
            for (var nt = 0u; nt < N_TILES; nt += 1u) {
                b0[nt] = coopLoad<coop_mat16x16<i8, B>>(&wb[(nt * 16u) * 64u + s * 16u], 64u);
                b1[nt] = coopLoad<coop_mat16x16<i8, B>>(&wb[(nt * 16u) * 64u + 32u + s * 16u], 64u);
            }
            if (AXPF == 2u) {
                // Consume the cross-barrier prefetched A tiles (already in flight
                // since before the barrier).
                for (var mt = 0u; mt < M_TILES; mt += 1u) {
                    for (var nt = 0u; nt < N_TILES; nt += 1u) {
                        acc[mt * N_TILES + nt] = coopMultiplyAdd(ap0[s * M_TILES + mt], b0[nt], acc[mt * N_TILES + nt]);
                        acc2[mt * N_TILES + nt] = coopMultiplyAdd(ap1[s * M_TILES + mt], b1[nt], acc2[mt * N_TILES + nt]);
                    }
                }
            } else if (AXPF == 1u) {
                // Issue ALL M_TILES A-loads for this s-step up front (vmcnt
                // accumulates), THEN the WMMAs drain behind one wait instead of
                // stalling 1:1 on each inline load. A i8 fragments are ~1 VGPR.
                var a0: array<coop_mat16x16<i8, A>, M_TILES>;
                var a1: array<coop_mat16x16<i8, A>, M_TILES>;
                for (var mt = 0u; mt < M_TILES; mt += 1u) {
                    a0[mt] = coopLoadT<coop_mat16x16<i8, A>>(&x[(m0 + mt * 16u) * K + k0 + s * 16u], K);
                    a1[mt] = coopLoadT<coop_mat16x16<i8, A>>(&x[(m0 + mt * 16u) * K + k0 + 32u + s * 16u], K);
                }
                for (var mt = 0u; mt < M_TILES; mt += 1u) {
                    for (var nt = 0u; nt < N_TILES; nt += 1u) {
                        acc[mt * N_TILES + nt] = coopMultiplyAdd(a0[mt], b0[nt], acc[mt * N_TILES + nt]);
                        acc2[mt * N_TILES + nt] = coopMultiplyAdd(a1[mt], b1[nt], acc2[mt * N_TILES + nt]);
                    }
                }
            } else {
                for (var mt = 0u; mt < M_TILES; mt += 1u) {
                    let a0 = coopLoadT<coop_mat16x16<i8, A>>(&x[(m0 + mt * 16u) * K + k0 + s * 16u], K);
                    let a1 = coopLoadT<coop_mat16x16<i8, A>>(&x[(m0 + mt * 16u) * K + k0 + 32u + s * 16u], K);
                    for (var nt = 0u; nt < N_TILES; nt += 1u) {
                        acc[mt * N_TILES + nt] = coopMultiplyAdd(a0, b0[nt], acc[mt * N_TILES + nt]);
                        acc2[mt * N_TILES + nt] = coopMultiplyAdd(a1, b1[nt], acc2[mt * N_TILES + nt]);
                    }
                }
            }
        }

        // Rescale both blocks into yacc. Each `u` is one block: load its row scales
        // `d_a`, then apply the per-block scale d_a[m]·d_w[n] (an outer product over
        // the tile) — which must enter as its own [16×16] fragment (coopmat has no
        // per-element access). Two paths (BCAST), identical barrier count:
        //   BCAST=1: build the fragment on the fly with 0-stride coopLoads —
        //     coopLoad(da_l,0)[i][j]=da_l[j], coopLoadT(dw2,0)[i][j]=dw2[i], product
        //     = da_l[j]·dw2[i], lining up with acc[i][j] (out row mt·16+j, col
        //     nt·16+i). No `stage` fill, no LDS materialization.
        //   BCAST=0: form the outer product in `stage` (×4-unrolled) and coopLoad it.
        for (var u = 0u; u < 2u; u += 1u) {
            let bb = beta + u;
            for (var i = lid; i < M_ROWS; i += WG) {
                da_l[i] = f32(x_scales[(m0 + i) * NB + bb]);
            }
            // da_l written before it is read below; at BCAST=0/STAGE_BUFS=1 this
            // also orders the previous pass's `stage` coopLoad before the overwrite.
            workgroupBarrier();
            if (BCAST == 1u) {
                let dwo = u * N_COLS; // block u's d_w base in dw2
                for (var mt = 0u; mt < M_TILES; mt += 1u) {
                    for (var nt = 0u; nt < N_TILES; nt += 1u) {
                        let t = mt * N_TILES + nt;
                        let scale = coopLoad<coop_mat16x16<f32, C>>(&da_l[mt * 16u], 0u)
                            * coopLoadT<coop_mat16x16<f32, C>>(&dw2[dwo + nt * 16u], 0u);
                        if (u == 0u) {
                            yacc[t] = yacc[t] + scale * f32(acc[t]);
                        } else {
                            yacc[t] = yacc[t] + scale * f32(acc2[t]);
                        }
                    }
                }
                workgroupBarrier(); // coopLoad(da_l) done before next pass overwrites da_l
            } else {
                let base = (u % STAGE_BUFS) * TILE_ELEMS;
                let n = lid % N_COLS;
                let dw = select(dw2[N_COLS + n], dw2[n], u == 0u);
                // ×4-unrolled column walk so independent da_l loads pipeline.
                let step = WG / N_COLS;
                for (var m = lid / N_COLS; m < M_ROWS; m += 4u * step) {
                    let d0 = da_l[m];
                    let d1 = da_l[m + step];
                    let d2 = da_l[m + 2u * step];
                    let d3 = da_l[m + 3u * step];
                    stage[base + m * N_COLS + n] = d0 * dw;
                    stage[base + (m + step) * N_COLS + n] = d1 * dw;
                    stage[base + (m + 2u * step) * N_COLS + n] = d2 * dw;
                    stage[base + (m + 3u * step) * N_COLS + n] = d3 * dw;
                }
                workgroupBarrier(); // stage written before coopLoad
                for (var mt = 0u; mt < M_TILES; mt += 1u) {
                    for (var nt = 0u; nt < N_TILES; nt += 1u) {
                        let t = mt * N_TILES + nt;
                        let scale = coopLoadT<coop_mat16x16<f32, C>>(
                            &stage[base + (mt * 16u) * N_COLS + nt * 16u], N_COLS);
                        if (u == 0u) {
                            yacc[t] = yacc[t] + scale * f32(acc[t]);
                        } else {
                            yacc[t] = yacc[t] + scale * f32(acc2[t]);
                        }
                    }
                }
            }
        }
    }

    // Epilogue: coopStore yacc (f32) to LDS, then write y via NORMAL stores.
    // coopStore straight to y is INVISIBLE to vulkano's reflection-based
    // auto-sync (like coopLoad — touch.wgsl), so the recorded prefill graph
    // would race the consumer (geglu/rms). The LDS round-trip keeps the y write
    // reflection-visible, exactly as the f16 gemm does. `stage` (buffer 0) is
    // free here — its last loop use was the rescale's coopLoad.
    if (EPI == 1u) {
        // Direct: f16(yacc) coopmat in registers → coopStoreT straight to y, no LDS
        // round-trip (NOT graph-safe — invisible to auto-sync; A/B only).
        for (var t = 0u; t < ACC; t += 1u) {
            coopStoreT(f16(yacc[t]), &y[(m0 + (t / N_TILES) * 16u) * N + n0 + (t % N_TILES) * 16u], N);
        }
    } else {
        workgroupBarrier(); // last rescale's stage coopLoad done before reuse
        for (var t = 0u; t < ACC; t += 1u) {
            coopStoreT(yacc[t], &stage[t * 256u], 16u);
        }
        workgroupBarrier();
        for (var i = lid; i < TILE_ELEMS; i += WG) {
            let t = i / 256u;
            let e = i % 256u;
            let row = m0 + (t / N_TILES) * 16u + e / 16u;
            let col = n0 + (t % N_TILES) * 16u + e % 16u;
            y[row * N + col] = f16(stage[i]);
        }
    }
}
