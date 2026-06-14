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
  ./target/release/sg-bench rgp gemm_q4_0_i8_t22_k21504_n5376
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
| `gemm_q4_0_i8_t22_k21504_n5376` | int8 FFN **down** (the regressor) | 2×2 |
| `gemm_q4_0_i8_t22_k5376_n21504` | int8 FFN **gate/up** | 2×2 |
| `gemm_q4_0_k21504_n5376` | f16 FFN down (A/B) | 4×4 |
| `gemm_q4_0_k5376_n21504` | f16 FFN gate/up (A/B) | 4×4 |

All at M=256 (the prefill chunk). Add more in `crates/sg-bench/src/rgp.rs::shape`.

## Capture the real prefill graph (optional, advanced)

To trace the *actual* recorded prefill graph (all kernels in flight, real
overlap) instead of one isolated GEMM, use the **trigger file** with the e2e
profile and capture a single in-app window:

```sh
MESA_VK_TRACE=rgp \
MESA_VK_TRACE_TRIGGER=/tmp/rgp_trigger \
RADV_THREAD_TRACE_BUFFER_SIZE=536870912 \
  ./target/release/sg-bench profile &
# when it prints "prefill 256-chunk @ q0 0 …", trigger one capture:
touch /tmp/rgp_trigger
```

(Or `--features sg-model/int8-ffn` on the build to trace the int8 FFN path.) The
SQTT buffer caps how much fits — a full 60-layer prefill won't; the trigger grabs
the work around the trigger event. Prefer the focused per-kernel capture for
clean single-kernel analysis.

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
