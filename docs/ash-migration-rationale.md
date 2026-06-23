# Why move the GPU layer from vulkano to ash

**Status:** planned task (not started). This documents *why*, and the workarounds
already tried, so the next agent doesn't re-walk the dead ends. Written 2026-06-23.

## The core limitation

vulkano's `AutoCommandBufferBuilder` performs **automatic synchronization**: it
derives the pipeline barriers between dispatches from **SPIR-V reflection** of each
pipeline's buffer usage. That reflection **does not see cooperative-matrix
accesses** (`coopLoad`/`coopStore`) — they're emitted by our naga fork as ops the
reflector doesn't recognise as buffer reads/writes.

Consequences, both already in-tree before this session:
- A GEMM's `y` (written via `coopStore`) and `x` (read via `coopLoad`) are invisible
  to auto-sync, so **no barrier is inserted between consecutive coopmat dispatches**.
  `shaders/touch.wgsl` and `kernel.rs`'s manual descriptor-layout construction exist
  *only* to work around this blindness (see their comments).
- Measured cost of the resulting overlap: with `SG_BENCH_DISPATCHES=40` (40 bb_m4
  dispatches in one submit, no barrier between → they overlap and L2-thrash),
  bb_m4 = **13.17 TFLOPS**; at `n_disp=1` (serial), **16.32 TFLOPS** — same kernel,
  same verified 2900 MHz clock. ~24% apart purely from uncontrolled overlap.

We have **no way to insert a barrier we control** (execution vs memory; precise
stage/access scopes) between coopmat dispatches through vulkano's safe API.

## Workarounds tried this session (all inadequate)

1. **`touch.wgsl` shim dispatch between producer and consumer.** Works only when the
   *producer's* write is **visible** (a normal `OpStore`) and the consumer's read is
   the invisible coopLoad. It **cannot anchor to an invisible coopStore write** — so
   it does not serialize two coopmat dispatches that both write `y`. Dead for our
   case (split-M, where every dispatch is a coopStore producer).

2. **In-kernel visibility shim** — a never-true guarded *normal* store to `y`
   (`if (lid == 0xFFFFFFFFu) { y[0] = 0.0h; }`, the `touch.wgsl` trick inlined). This
   makes reflection see `y` as written, so auto-sync inserts a real compute→compute
   WAW barrier between dispatches. **It works** (used in `gemm_q4_0_i8_bb_m4_split.wgsl`
   to serialize the M-block dispatches in one submit). But it is a hack: a fake dead
   store, only a **conservative buffer-granularity** barrier, **no control** over the
   barrier's stage/access scope or execution-vs-memory choice, and it relies on
   naga/ACO not eliding the dead store.

3. **Four separate fence-serialized submits** (`dispatch_blocking` ×4). Guarantees
   serialization but pays a full CPU fence wait + command-buffer rebuild per boundary
   — measured as **~250 µs of GPU-idle gap per boundary** (this is the "~290 µs
   barrier" seen in RGP), and it makes the GPU bursty, which destabilises the clock
   and the timing (high cv). A production non-starter and a measurement non-starter.

4. **Raw `vulkano::command_buffer::sys::RecordingCommandBuffer` + `pipeline_barrier`.**
   This is vulkano's unsafe manual-recording API and *does* expose
   `pipeline_barrier(&DependencyInfo)`. **But it cannot be submitted**: `end()`
   returns a `sys::CommandBuffer`, which does **not** implement
   `PrimaryCommandBufferAbstract`; `CommandBufferSubmitInfo::new` requires exactly
   `Arc<dyn PrimaryCommandBufferAbstract>`, and `QueueGuard::submit` only takes
   `SubmitInfo` built from `CommandBufferSubmitInfo`. So a manually-recorded command
   buffer cannot reach the queue through vulkano's API — you'd drop to raw `ash`
   `vkQueueSubmit2` with the raw `vk::CommandBuffer` handle **anyway**.

## Conclusion

The "manual" vulkano path (#4) already forces raw `ash` for the *submit*, and the
only working barrier (#2) is a hack with no scope control. To get **real,
controllable barriers** (and to stop fighting auto-sync's coopmat blindness, which
also forces `touch.wgsl` and manual descriptor layouts), move the GPU layer to
**ash** (direct Vulkan).

## Scope notes for the migration

vulkano conveniences currently relied on, each needing an ash equivalent:
- Pipeline / descriptor-set-layout creation (already partly manual in `kernel.rs`
  because reflection is unreliable — see its comment).
- `GraphRecorder` / `CommandGraph` (record-once / submit-many), `GpuTimer`
  (timestamp queries), `dispatch_blocking` (one-shot submit) in `exec.rs`/`graph.rs`.
- Descriptor set allocation + binding, push constants (`var<immediate>` in naga 29).

Once real barriers exist, **remove the hacks**: `shaders/touch.wgsl`, and the
visibility shim in `gemm_q4_0_i8_bb_m4_split.wgsl`.

## Tangential measurement finding (carry this over)

`pp_dpm_sclk` (what the benches print as `sclk=…`) is the DPM **ceiling**, not the
achieved clock. Read the **actual** clock via `rocm-smi`/`gpu_metrics`. Evidence:
under **pegged** 100%-busy load the actual clock held a solid **2900 MHz** (rocm-smi,
cv 0.13%); under **bursty** load (`split-bench`, 15–45% busy) the DVFS governor ramps
the clock per-dispatch, `pp_dpm_sclk` sampled idle states (e.g. 1738 MHz) while the
dispatches still ran near 2900, and the per-dispatch timing variance blew cv to
2–6.5%. The bench's `sclk=` print is therefore misleading; gate trusted runs on a
stable rocm-smi clock, not `pp_dpm_sclk`.
