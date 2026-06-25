# Down-GEMM optimization — consolidated findings

The int8-MMQ **FFN-down** GEMM (Q4_0 weights × Q8 activations, **K=21504 N=5376**,
M=256 prefill chunk) is the rank-#2 prefill lever ([[goal-fastest-gemma4-server-optimization-is-core]]).
This file collects ~a week of exploration into one place — the mechanism we landed on,
the levers that won and died (with the *measured* reason), and what's down-specific.
It is the synthesis; `STATUS.md` is the dated log, and the linked `[[memory]]` slugs
hold the atomic detail.

**Headline:** deployed `swz_m4n1` → **`bb_pfd4` = +21.5%** (the current best down-gemm,
GPU timestamps, sclk 2900). The up shape (K=5376) is a *different problem* — the whole
weight-prefetch arc is flat there.

> **Number-comparability warning.** The ash migration (vulkano→ash) moved the baseline
> mid-arc: it removed a ~23 µs periodic memory spike, so **post-migration numbers are the
> only trustworthy absolute baseline** ([[vulkano-cant-barrier-coopmat-ash-migration]]).
> bb_m4-vs-deployed even *flipped sign* across it (−4.6% pre → +11.9% post). Trust
> *within-era, within-run Δ*; don't compare absolute TFLOPS across the migration or across
> clocks. cv is the real trust proxy ([[pp-dpm-sclk-is-ceiling-not-actual-clock]]).

---

## The mechanism (what the down-GEMM actually is)

1. **It is memory-MLP / latency-bound, NOT byte/bandwidth-bound.**
   The decisive test: an 8×1 register-reuse tile read **2.5× fewer bytes yet ran slower**
   — it under-drove the memory system (too few waves → too few outstanding requests).
   The lever is *keeping memory busy* (outstanding requests), measured by **effective BW
   (local-video ÷ duration), never byte count.** ([[prefill-gemm-is-mlp-bound-not-byte-bound]],
   [[memory-stall-pct-is-not-bandwidth-util]])

2. **Occupancy is double-edged and must be *scheduled*, not just maximized.**
   Raw max-occupancy (1×1, 11/16 waves) lost −47.8% — it cut per-wave VMEM-in-flight and
   4×'d weight re-reads. Multi-wave + shared-LDS lost (L2 thrash + barrier). But
   *under*-occupancy under-drives memory. The win is **high occupancy + an L2-friendly
   schedule** (super-block tile_index) that keeps the working set warm.
   ([[next-int8-gemm-lever-max-occupancy]], [[multi-wave-occupancy-gemm-dead-end]],
   [[occupancy-l2-thrash-not-the-store]])

3. **The binding stall MOVES with the occupancy regime — scope every conclusion per-kernel.**
   This is the single most-repeated lesson:
   - **Higher-occupancy kernels** (deployed `swz_m4n1`, the `l2` family): the before-WMMA
     stall is the **activation (X) load**; weight latency is cross-wave-hidden. Every
     weight-side lever failed there. ([[activation-prefetch-static-ping-pong-best-l2]])
   - **Low-occupancy `bb_m4`** (4×row → ~half the waves): the weight latency is **exposed**
     and becomes the #1 stall. The *same* weight prefetch that was FLAT on the 2×row `bb`/`fo`
     is the big lever here. ([[bb-m4-binding-stall-is-weight-load]], [[fo-bb-occupancy-negative-sxp-hoist-pending]])
   - After batching activations, the *next* stall is the **d_a scale load**, not weights.

4. **The bb_pfd win = keep memory 100% busy + L2 warm.**
   Depth-D cooperative-LDS weight prefetch (all 64 lanes fetch PFD pairs at once into
   double-buffered packed LDS) made `bb_pfd4` the **first down-gemm to keep memory 100%
   busy the entire run** (bb_m4 decays to 100%/95% as later wave-batches launch) and hold
   **L2 ~88%** (bb_m4 decays to ~75%). Fewer VRAM bytes (1793 vs 2920 MB) is a *consequence*
   of the warm L2, not the cause — consistent with (1). Mechanism *hypothesis* (Ada's,
   unproven): deeper prefetch lets later waves' loads merge/catch up → synchronizes loop
   starts → aligns same-address activation loads → L2 stays warm.
   ([[bb-pfd-depth4-weight-prefetch-win]])

---

## Levers that WON

| lever | effect | where it lives |
|---|---|---|
| **Tall-thin tile** (M_TILES=4, N_TILES=1) | +22% e2e prefill, won 7/8 shapes; weight LDS scales with N only → max M, min N | deployed, bb_m4, bb_pfd ([[tall-thin-tile-weight-reuse]]) |
| **Workgroup super-block swizzle** | +16% kernel / +5.6% e2e prefill, free; keeps one X-block L2-hot | deployed ([[occupancy-l2-thrash-not-the-store]]) |
| **0-stride scale rescale (K-split, `BCAST`)** | −4..−16% on K=5376 gemms; loses large-K (+2..12%) → deployed *selectively* per shape | deployed `const BCAST` ([[gemm-0stride-rescale-k-split]]) |
| **β×2 WMMA-ILP** | the dominant `l2` lever, **+34%**; batches WMMAs so loads hide under them | l2 family ([[prefill-gemm-is-mlp-bound-not-byte-bound]], [[rdna3-wmma-latency-hidden-by-ilp-not-occupancy]]) |
| **Deep weight prefetch (software-pipelined)** | the deployed "s2+pf"; substitutes for β×2 (they hide the same latency — don't stack) | deployed |
| **bb "one-batch" body** | every global load of an iteration issued up front → one vmcnt stall/iter | bb, bb_m4 |
| **bb_m4 4×row tall tile** | +11.9% over deployed (post-migration); ½ the waves, 2× weight reuse | bb_m4 |
| **bb_pfd depth-4 cooperative LDS weight prefetch** | **+21.5% over deployed** — the current best; uses the idle 48 lanes | bb_pfd ([[bb-pfd-depth4-weight-prefetch-win]]) |

Note the deployed kernel is itself the product of the early arc (tall-thin + swizzle +
BCAST + deep prefetch). `bb_pfd4` is the challenger that beats it.

## Levers that DIED (don't re-tread)

| dead lever | measured outcome | why ([[memory]]) |
|---|---|---|
| Byte reduction (8×1 register reuse) | −33% despite 2.5× fewer bytes | under-drives memory; it's MLP-bound ([[prefill-gemm-is-mlp-bound-not-byte-bound]]) |
| Max-occupancy 1×1 | −47.8% | cut VMEM-in-flight + 4× weight re-reads ([[next-int8-gemm-lever-max-occupancy]]) |
| Multi-wave + shared-LDS weight strip | slower | L2 thrash + barrier ([[multi-wave-occupancy-gemm-dead-end]]) |
| Cooperative *dequant* (full-wave unpack) | −8% / −15% | the weight load was already hidden behind the prior MMA; only added traffic ([[prefill-gemm-is-mlp-bound-not-byte-bound]]) |
| 1-deep weight prefetch (pfd1 / bb_m4_pf / bb_pf) | regresses vs no-prefetch bb_m4 (+8% vs +11.9%) | store→next-iter-load on critical path; needs depth ≥2 ([[bb-pfd-depth4-weight-prefetch-win]]) |
| N-tile super-block swizzle on bb_m4 (sb2/4/8) | worse than bb_m4 | the 4×1 tile already captures the reuse the swizzle targets (hypothesis) |
| Full-occupancy `fo`/`bb` occupancy family | ceiling −15.5% | occupancy isn't the lever ([[fo-bb-occupancy-negative-sxp-hoist-pending]]) |
| int8 V/PV flash (Piece B) | slower e2e, gap grows w/ ctx | shelved; QKᵀ int8 (Piece A) kept ([[int8-pv-needs-inkernel-i8-operand]]) |
| PD / AXPF micro-variants | flat at pinned clock (−0..−1.2%) | the apparent "+4%" was an auto-clock DVFS artifact ([[pd-axpf-flat-at-pinned-clock]]) |
| **d_a-scale prefetch (`bb_pfd` DAP)** | pfd4_dap +12.5% vs pfd4 +21.2%, **at byte-identical footprint** (144 VGPR / 6144 LDS) | the 57K-clk d_a stall ISN'T real headroom — the kernel is memory-pipe-bound (100% busy), so d_a is just part of the saturated load stream; prefetching it a slot early only adds another in-flight load. Tried LDS-staged (dead: dropped occupancy, see [[aco-vgpr-budget-tracks-binding-occupancy]]) AND register cross-iter carry (dead even at identical footprint). The SXP hoist gave identical ISA (ACO already schedules it as early as possible). |
| **Transposed [N,M] dispatch (`bb_pfd` TPOSE, wg.x=N)** | pfd4_tp +10.1% vs pfd4 +21.2% | loses the weight reuse: weights (62 MiB) > MALL so reuse must come from co-residency; wg.x=M co-locates the 4 M-blocks sharing a weight column, wg.x=N re-fetches weights per M-block. Measured: VRAM/MALL **430 vs 341 MiB**, lower L2 hit rate. (Activations are 5.25 MiB → cached either way, so reusing *them* buys nothing.) See the dispatch-order analysis below. |

**Still pending / not-yet-ported (live):** the `d_a`-scale hoist (SXP) was +4–9% VGPR-free
on `fo` but never ported to the shipping kernels ([[fo-bb-occupancy-negative-sxp-hoist-pending]]).

### Dispatch order: wg.x=M ([M-blocks, N-blocks]) is correct, quantified

The dispatch must put **M in the fast-varying (x) dimension** so the 4 M-blocks that
share each N-column's weights launch co-resident. Why, by the numbers (down M=256):

| operand | per-tile | total unique | fits MALL (~32 MB)? |
|---|---|---|---|
| weights (16 cols × K, Q4_0) | 189 KiB | **62 MiB** | **no** |
| activations (64 rows × K, int8) | 1.31 MiB | 5.25 MiB | yes |

Per-tile the activation load is ~7× the weights, but that's the wrong level: activations
(5.25 MiB) stay cached regardless of order, so the only reuse that matters is **weights**,
and weights are too big to cache → reuse must come from co-residency. DRAM consequence:
**wg.x=M ≈ 67 MiB DRAM** (weights fetched ~once, reused ×4) vs **wg.x=N ≈ 253 MiB**
(weights re-fetched ×4). Measured VRAM/MALL 341 vs 430 MiB confirms the direction.
Corollary: this is also why `bb_m4_swz` (reuse *both* operands) didn't beat plain — the
activation half was already free. **RGP note:** the "write size" memory counter is a
VRAM-level eviction count (96 B–3 KB), NOT the logical 2.6 MB output write — Y is
cache-resident in all variants, so don't read the write column for signal.

## Methodology lessons (what made the measurements trustworthy)

- **Measure before declaring dead — repeatedly.** Causal stories were wrong ~4× per session
  (cooperative-dequant "execz bottleneck", PF "+12 VGPR → occupancy", axp3 "occupancy
  collapsed"); the *outcomes* held, the *mechanisms* didn't. Separate measured-outcome from
  asserted-cause; hand Ada the trace. ([[measure-before-declaring-dead]],
  [[dont-declare-kernel-perf-conclusions-without-profiling]])
- **Pin the clock (`high`/2900) before any A/B.** Auto idles fabric/socclk ≈ −30% and DVFS
  artifacts masquerade as wins. ([[pin-perf-level-high-fabric-clock-idles-under-auto]],
  [[pd-axpf-flat-at-pinned-clock]])
- **cv is the trust proxy, not pp_dpm_sclk** (which lies under bursty load); read rocm-smi.
  ([[pp-dpm-sclk-is-ceiling-not-actual-clock]])
- **RGP for structure only; warm round-robin GPU-timestamp bench (`mmq_variance`) for timing**
  — durations aren't cross-kernel comparable. **n_disp=1** (dispatch overlap confounds).
  ([[rgp-durations-not-cross-kernel-comparable]], [[dispatch-overlap-confounds-cross-kernel-bench]])
- **`RADV_DEBUG=shaderstats` is the cheap VGPR/occupancy fact-check** that killed several
  false occupancy stories. ([[radv-shaderstats-vgpr-occupancy]])
- **Sustained load can thermal/power-cut the box** — run correctness gates at auto, benches
  individually with cooldown, watch `sensors` junction not rocm-smi edge.
  ([[sustained-load-thermal-power-cut]])

## Down-specific vs general (for the next-gen UP kernel)

- **Down-specific (do NOT assume on up):** the entire weight-prefetch arc. bb_pfd is flat on
  up (pfd4 +0.8% noise, pfd1 −5.4%). Up's short K (168 blocks vs 672) doesn't expose the
  weight stall the way down's long K does, and the prologue fill amortizes over far fewer
  iters. **The up shape needs its own analysis** — start from *where its* binding stall is
  (RGP the up `bb_m4`/deployed), don't port down's lever.
- **Probably general:** tall-thin tile, the MLP-bound framing, the measurement discipline,
  single-wave barrier elision, "scope the stall per occupancy regime."

## Open threads

1. **The 57K clk/wave stall before the last 4 WMMAs** — bb_pfd4's new #1 stall, the next down lever.
2. **Deploy swap** deferred (deliberate): `graph.rs` still loads `swz_m4n1`. Move both FFN
   gemms together once the up kernel exists.
3. **Next-gen up kernel** — separate session; down's findings give the method, not the lever.
4. **SXP d_a-scale hoist** — pending port to a shipping kernel (+4–9% on fo).
