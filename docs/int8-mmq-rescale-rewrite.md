# Task guide — int8-MMQ in-register rescale rewrite (+ re-bench)

**Task for this session:** rewrite `crates/sg-gpu/shaders/gemm_q4_0_i8.wgsl` to do the per-block
rescale **in registers** (using coopmat ops now available in the naga fork), eliminating the
per-block i32 `coopStore` + barriers, keep parity, and re-benchmark against f16. This is the payoff
step for the int8 prefill optimization (profile rank #2). Self-contained — read this + the files it
points at.

## Why (context)

Goal: **the fastest Gemma 4 31B Q4_0 server on this Framework Desktop** (`AGENTS.md`) — kernel
optimization is core work. The int8-MMQ GEMM is the rank-#2 prefill lever.

Current state: `gemm_q4_0_i8.wgsl` is **parity-green** (nrmse ~2e-4 vs the `mmq_q4_0_q8` oracle
across tilings; bench: parity_mmq) but only **4.4 TFLOPS (bench: mmq_tflops, perf=high) ≈ 0.46× f16**
(~9.5). The bottleneck (measured — an occupancy sweep ruled occupancy out, STATUS 2026-06-13) is the
**per-32-block rescale**: `coopStore` the i32 dot tile to LDS + 2 barriers + a scalar rescale into
an LDS `facc`, ×168 blocks. The spill + barriers dominate.

The MMQ math: `Y[m][n] = Σ_β (d_a[m][β]·d_w[n][β]) · DOTβ[m][n]`, where `DOTβ` is the int8×int8→i32
block dot. The scale is per-block and a rank-1 outer product over the tile — so the i32 dot must be
pulled out and scaled every block. On KHR coopmat1 we had no fragment element access → forced spill.

**Now unblocked** (fork rev `8366d92e`, wired in `Cargo.toml`; verified on the box by
`coop_arith_smoke` + its test):
- `f32(coop_mat<i32,C>)` → `OpConvertSToF` (convert the i32 dot to f32 in-register),
- `coop * coop` → component-wise `OpFMul` (apply the per-element scale),
- `coop + coop` → `OpFAdd` (already worked; accumulate Y in registers),
- all on **C-use** matrices, and `coopLoad` into a **C-use** matrix works.

## The rewrite — in-register rescale

```
Y[ACC] : coop_mat<f32,C>          // persistent, register-resident; zero once at start
per 32-block β:
    # 1. MMA → i32 dot per tile (unchanged)
    for t in ACC: acc[t] = 0          // i32
    for s in 0..2: load aq/wq i8 tiles; acc[..] = wmma(a, b[..], acc[..])
    # 2. build the rank-1 scale matrix d_a[m]·d_w[n] in a small LDS buffer
    for i in lid..(M_ROWS*N_COLS):
        scale_buf[i] = f32(x_scales[(m0 + i/N_COLS)*NB + β]) * f32(w_scales[(n0 + i%N_COLS)*NB + β])
    barrier
    # 3. in-register rescale + accumulate (per-lane coopmat ops — no barrier)
    for mt,nt:
        let scale = coopLoad<coop_mat16x16<f32,C>>(&scale_buf[(mt*16)*N_COLS + nt*16], N_COLS)
        Y[mt*N_TILES+nt] = Y[mt*N_TILES+nt] + scale * f32(acc[mt*N_TILES+nt])
    barrier   // before next block overwrites scale_buf (drop via double-buffering — measure)
epilogue: write Y (f32) → y (f16)
```

vs the current kernel this removes: the per-block i32 `coopStore` (8 KB at 2×4), the `facc` LDS, and
the scalar rescale loop. `Y` lives in registers; only a small `scale_buf` (~`M_ROWS*N_COLS*4` B; 8 KB
at 2×4) touches LDS, written once/block.

### Design choices / things to decide empirically
- **Barriers:** the `scale_buf` write→`coopLoad` needs one barrier; a single buffer needs a second
  before the next block reuses it. **Double-buffer `scale_buf`** to drop the second (costs 2× the
  scale LDS) — measure whether the barrier or the LDS matters more (occupancy isn't the binder here,
  per the earlier sweep, but re-check).
- **Scale construction alternative:** instead of materializing the full outer product, build a
  `d_a`-broadcast and a `d_w`-broadcast coop matrix and `OpFMul` them — fewer LDS writes. Try both.
- **Epilogue `f32 Y → f16 y`:** `coopStore` writes the matrix's component type (f32). **Verify
  whether fork rev `8366d92e` supports `f16(coop<f32>)` (`OpFConvert`)** — `coop_arith_smoke` only
  proved i32→f32. If not, `coopStore` `Y` to a 16×16 f32 LDS scratch per tile (one-time, epilogue
  only) and have threads write `f16(scratch)` to `y`. Cheap (once per tile, not per block).
- **Re-sweep tiling.** The bottleneck changed; VGPR now also holds `Y[ACC]` (f32) + `acc[ACC]` (i32)
  live. Check VGPR / waves-per-SIMD via `MESA_SHADER_CACHE_DISABLE=1 RADV_DEBUG=shaderstats` (the
  cache suppresses re-emission). Bigger tiles amortize the per-block scale-build but cost VGPR
  (256 ceiling). Keep the existing parity variants and add bench variants per tiling as needed.

## Verification (in order)

1. **Parity FIRST** — `cargo nextest run -p sg-gpu gemm_q4_0_i8_matches_mmq_reference` must stay green
   (nrmse ~2e-4 vs `mmq_q4_0_q8`) for every tiling you bench. A kernel that skips work can look fast;
   don't trust any TFLOPS until parity passes. Reuse/extend `tests/parity_mmq.rs`.
2. **Bench at `perf=high`** — `mmq_tflops` vs the **4.4 (int8) / ~9.5 (f16) TFLOPS** baseline.
   **Pin the clock first** (auto reads ~30 % low — STATUS Known gotchas):
   `echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level` (needs root;
   ask Ada). Use `gemm_variance`-style steady-state methodology if you add a probe.
3. **Cite every number** `(bench: mmq_tflops, perf=high)` and update STATUS's int8 numbers +
   citations in the same commit (the AGENTS perf-number rule). Update `crates/sg-gpu/shaders/
   gemm_q4_0_i8.wgsl`'s header comment too.
4. **Regression** — the full coopmat suite stays green: `cargo nextest run -p sg-gpu -E
   'binary(parity_attn) or binary(parity_gemm) or binary(parity_mmq) or binary(coop_i8_smoke) or
   binary(coop_arith_smoke)'`.

## If int8 beats f16
Follow-on (separate task): wire `gemm_q4_0_i8` into the prefill graph (`sg-model/src/graph.rs`,
per-shape variants like the f16 path), and gate activation-Q8 accuracy on the **perplexity harness**
behind a build flag (activation quant is the accuracy risk — the parity oracle only checks the MMQ
arithmetic, not the quality vs full precision). If it doesn't beat f16, bank the finding with the
measured numbers and keep the kernel in-tree.

## References
- `crates/sg-gpu/shaders/gemm_q4_0_i8.wgsl` (current kernel) + `gemm_q4_0_i8_raw.wgsl` (MMA-ceiling
  diagnostic), `tests/parity_mmq.rs`, `tests/reference/mod.rs` (`mmq_q4_0_q8` oracle + `quant_q8_0`).
- `crates/sg-gpu/benches/mmq_tflops.rs` (f16 vs int8 tilings) + `gemm_variance.rs` (pinned steady
  state).
- `crates/sg-gpu/shaders/coop_arith_smoke.wgsl` + test — the new ops in action.
- `docs/naga-coopmat-arith-patch.md` (what the fork added), `docs/STATUS.md`
  "Prefill-optimization investigation (2026-06-13)" + Known gotchas (perf-level), `AGENTS.md`
  (goal, bespoke, perf-number rule).
