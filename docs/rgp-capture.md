# RGP / SQTT capture on the target box (RADV)

Real GPU profiling — wavefront occupancy over time, where waves stall (VMEM /
LDS / ALU / barrier), cache hit rates, per-instruction latency — beyond what
`RADV_DEBUG=shaderstats` (static VGPR/occupancy) and `RADV_DEBUG=asm` (ISA) give.

We're on **Mesa/RADV** (26.1.1), so the capture is the driver's built-in **SQTT**
export — NOT the AMD DevDriver / GPUOpen server path (that's AMDVLK/Windows).
Capture locally → produces a `.rgp` → open in the **Radeon GPU Profiler** GUI on
your machine. (RGP GUI: free from GPUOpen; the only external piece — the capture
itself needs nothing extra installed.)

## Capture one GEMM (the focused harness)

`sg-bench rgp <kernel>` dispatches one GEMM in isolation — one
`dispatch_blocking` == one queue submit, so with per-submit capture each submit
emits one clean `.rgp` (no swapchain/present needed; a GEMM has no
data-dependent control flow, so dummy buffers give a representative trace).

```sh
cargo build --release -p sg-bench   # once

MESA_VK_TRACE=rgp \
MESA_VK_TRACE_PER_SUBMIT=true \
RADV_THREAD_TRACE_BUFFER_SIZE=268435456 \
RADV_THREAD_TRACE_INSTRUCTION_TIMING=true \
  ./target/release/sg-bench rgp gemm_q4_0_i8_swz_m4n1_k21504_n5376
```

- Output: `/tmp/sg-bench_<timestamp>_submitN.rgp`, one per submit (0..7). **Take
  the last** (`submit7`) — warmest; they're structurally identical at pinned
  `perf=high` so any is fine for occupancy/stall analysis.
- `RADV_THREAD_TRACE_BUFFER_SIZE` (bytes) — bump if you see
  "Failed to capture RGP … buffer too small". 256 MB is plenty here.
- `RADV_THREAD_TRACE_INSTRUCTION_TIMING=true` — adds per-instruction latency in
  RGP's Instruction Timing view (the killer feature for "which op stalls").
- Pin clocks first (else the trace reflects idled fabric clock):
  `echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level`.

### Supported capture targets (the FFN A/B)

| kernel | what | tiling |
|---|---|---|
| `gemm_q4_0_i8_swz_m4n1_k21504_n5376` | int8 FFN **down**, **DEPLOYED** | 4×1 swz |
| `gemm_q4_0_i8_swz_m4n1_k5376_n21504` | int8 FFN **gate/up**, **DEPLOYED** | 4×1 swz |
| `gemm_q4_0_i8_t22_k21504_n5376` | int8 FFN down (pre-cache-block baseline) | 2×2 |
| `gemm_q4_0_i8_t22_k5376_n21504` | int8 FFN gate/up (baseline) | 2×2 |
| `gemm_q4_0_k21504_n5376` | f16 FFN down (A/B) | 4×4 |
| `gemm_q4_0_k5376_n21504` | f16 FFN gate/up (A/B) | 4×4 |

The `swz_m4n1` variants are the production prefill tiling (graph.rs) and take a
transposed `[M-blocks, N-blocks]` grid; the harness handles that. All at M=256
(the prefill chunk). Add more in `crates/sg-bench/src/rgp.rs::shape`.

## Capture a full prefill layer (the real graph) — `rgp-prefill`

`sg-bench rgp-prefill` traces the *actual* recorded prefill graph (every kernel
of a layer in flight, real overlap) instead of one isolated GEMM: the QKV gemms,
flash attention, the O gemm, FFN gate/up/down gemms, rmsnorm, rope, residual
joins. It submits **one layer at a time** (`record_prefill_layers(i..i+1)`), so
under per-submit capture each layer is its own `.rgp`.

**Per-layer, not per-chunk, by necessity.** SQTT records a continuous token
stream whose buffer fills on GPU *duration*, not dispatch count. A single gemm
fits; a 60-layer chunk — even one 6-layer watchdog segment — runs far too long to
hold (tested: overflows 1 GB). A layer is the repeating unit; layer 4 samples a
sliding layer and layer 5 a global one (`layer % 6 == 5`), so the two `.rgp`
together cover every prefill kernel.

```sh
echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
MESA_VK_TRACE=rgp \
MESA_VK_TRACE_PER_SUBMIT=true \
RADV_THREAD_TRACE_BUFFER_SIZE=2147483648 \
  ./target/release/sg-bench rgp-prefill --q0 0
```

- `--q0 0` is the FFN-gemm-bound short-context regime (cheap attention);
  `--q0 32512` would be the global-attention-bound long-context regime.
- 3 passes × 2 layers → 6 `.rgp` (`submit0..5`). **Take the LAST two**: `submit4`
  = warm sliding L4, `submit5` = warm global L5. (Weight uploads and warm-up
  passes don't pollute the file list — only the per-layer submits emit captures.)
- Build with `--features sg-model/int8-ffn` to trace the deployed int8 path.

**RADV gotchas this surfaced (corrected 2026-06-20 — an earlier version of this
section blamed the watchdog and trace size; both were wrong):**

- `RADV_THREAD_TRACE_INSTRUCTION_TIMING` **defaults to `true`** — omitting it does
  NOT disable it. Set `=false` explicitly to shrink the trace (drops per-op
  timing, keeps the occupancy/event/cache timeline).
- **The buffer-size env is a 32-bit byte field — it WRAPS mod 2³² (4 GiB).**
  MEASURED via the `initial buffer size: N MiB` line RADV prints: 2 GB→2048,
  3 GB→3072, **4 GB→0, 6 GB→2048, 8 GB→0**. So any value ≥ 4 GiB silently becomes
  a ~0-byte buffer; the SQTT writer (`SQG`) then page-faults writing the trace →
  `gfxhub page fault … Faulty UTCL2 client ID: SQG`, gfx ring timeout, "device
  wedged / hard recovery". Looks like a hang, but the dmesg `SQG` client ID is the
  tell. **Use a value just under 4 GiB** (`0xFC000000` = `4227858432`, 4032 MiB),
  and **set BOTH** `RADV_THREAD_TRACE_BUFFER_SIZE` and
  `RADV_CACHE_COUNTERS_BUFFER_SIZE` (RADV's "too small" error names both; cache
  counters are on by default and need room too). Auto-resize is off by default.
- **Long-context capture is NOT watchdog- or size-limited** (the old claim).
  `lockup_timeout` is 30 s on this box now (not 2 s — read
  `/sys/module/amdgpu/parameters/lockup_timeout`), and the global-flash layer
  trace is small: instruction-timing-ON `rgp-prefill` is 198 / 268 / 399 / 618 MB
  at q0 = 4096 / 8192 / 16384 / 32512 — nowhere near 4 GiB. With a valid <4 GiB
  buffer + both env vars, q0=32512 timing-on captures cleanly. In the RGP GUI,
  select the flash dispatch inside a full-layer trace to read its per-op timing —
  no need to isolate the kernel into its own submit. Working recipe:

  ```sh
  MESA_VK_TRACE=rgp MESA_VK_TRACE_PER_SUBMIT=true \
  RADV_THREAD_TRACE_BUFFER_SIZE=4227858432 \
  RADV_CACHE_COUNTERS_BUFFER_SIZE=4227858432 \
  RADV_THREAD_TRACE_INSTRUCTION_TIMING=true \
    ./target/release/sg-bench rgp-prefill --q0 32512
  ```

## Move it to your machine + open

```sh
scp <box>:/tmp/sg-bench_*_submit7.rgp .   # from your machine
```

Open in Radeon GPU Profiler. For our open questions:

- **Down-gemm regression** (`k21504_n5376` int8 vs f16): Wavefront Occupancy +
  the Events timeline — is it ALU-bound on the unpack/rescale (the
  rescale-cost-scales-with-K = 672 blocks hypothesis), LDS-bound, or
  occupancy-limited? Compare against the int8 gate/up (`k5376_n21504`, 168
  blocks) — if the per-block rescale dominates, the down trace shows ~4× the
  VALU/LDS time per output relative to the WMMAs.
- **2×2 `vmcnt` stalls**: the Instruction Timing view shows whether the WMMAs
  actually stall on the activation loads or are wave-hidden (the ×4-unroll
  experiment said occupancy hides them — confirm).
- **General**: cache hit rates (VMEM) on the Q4_0 weight reads; LDS bank
  conflicts on the `stage`/`wb` buffers.
