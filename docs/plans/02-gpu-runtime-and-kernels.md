# 02 — GPU Runtime & Kernels (`sg-gpu`)

## Scope

Vulkan compute foundation: device/queue management via vulkano, WGSL kernel library preprocessed
with naga-oil and compiled to SPIR-V, buffer/descriptor management on unified memory, command-graph
recording for prefill and decode, and per-kernel validation against CPU references.

## Runtime design

### Device & memory

- One `Instance`/`Device`, one compute queue. Required features asserted at startup (all confirmed
  present on the target): `shaderFloat16`, `storageBuffer16BitAccess`, `VK_KHR_cooperative_matrix`
  (usable from WGSL via naga — verified on the target machine), subgroup ops (basic/arithmetic/
  shuffle, expect wave32 on RDNA 3.5), `timelineSemaphore`. Not available on target: BF16,
  NV_cooperative_vector — all math f16 with f32 accumulation. M0 probe still enumerates the
  supported coopmat M/N/K/type configurations (drives which GEMM variants get compiled).
- All buffers HOST_VISIBLE|DEVICE_LOCAL (unified memory). Buffer classes:
  - `weights` (18 GB, immutable, written once at load)
  - `kv_global` (resident global KV, grows with context, written by kernels and by cache2 loads)
  - `kv_sliding` (ring, fixed ~820 MB)
  - `activations` (ping-pong scratch, sized for max prefill chunk)
  - `logits` (1 MB f32, read by CPU sampler each token)
  - `io_staging` (registered with uring for cache2 page traffic, if direct-into-mapped fails)
- One persistent descriptor set per (kernel, buffer-class) combination; per-dispatch variation goes
  through push constants (positions, lengths, ring offsets, layer index) — **no descriptor churn in
  the decode loop**.

### Shader build pipeline

- WGSL sources in `sg-gpu/shaders/`, composed with naga-oil (`#import` for shared blocks: Q4_0
  decode, RMSNorm body, RoPE math, subgroup reduction helpers; `#ifdef` specialization for layer
  type, head dims, tile sizes).
- Build script compiles WGSL → SPIR-V at **build time** (naga), embeds blobs; runtime does no shader
  compilation. Specialization that depends on probed hardware (subgroup size, coopmat tile shape) is
  done by compiling the small finite set of variants up front.
- **Per-shape kernel specialization:** one model means the complete set of matmul shapes is small
  and known at build time (~10 distinct M×N×K: sliding q/kv/o, global q/kv/o, MLP gate/up/down,
  LM head — × {GEMV M=1, GEMM M=chunk_size variants}). Bake dimensions, strides, head counts, and
  layer-type constants in via naga-oil defines and compile **one pipeline per shape**, leaving push
  constants/uniforms only for genuinely dynamic state (position, ring head, kv_len). Lets the
  compiler fully unroll inner loops and fold addressing. Binding a different pipeline per dispatch
  inside one pre-recorded command buffer is near-free, but verify: an A/B microbench
  (shape-specialized vs uniform-dimension generic kernel) is an explicit M2 work item; keep the
  generic variant as the comparison baseline and fallback.
- Each kernel has a stable Rust-side descriptor (`KernelDesc { entry, workgroup, push_layout }`)
  so callers never touch WGSL details.

### Submission model

- Dedicated GPU thread (plan 00). Two pre-recorded command-buffer families:
  - **decode step**: the full 60-layer single-token graph + logits, recorded once; per-token
    mutable state (position, ring head, KV length) lives in a small uniform buffer updated before
    each submit (one 64-byte write — cheaper than re-recording).
  - **prefill chunk**: same graph at chunk granularity (chunk = N tokens, N tunable 64–512),
    recorded per chunk-size variant.
- Timeline semaphore per conversation; CPU sampler waits on the value, writes the sampled token id
  into the next step's uniform region. Target: < 300 µs CPU-side overhead per decode step.

## Kernel inventory

| Kernel | Used in | Notes |
|---|---|---|
| `gemv_q4_0` | decode: all matmuls | fused dequant; one workgroup per output-row tile; subgroup reduction; f16 math/f32 acc; shape-specialized per matmul site. THE bandwidth-critical kernel: must stream weights at near-peak bw |
| `gemm_q4_0` | prefill matmuls | **cooperative-matrix** tiles (KHR coopmat via naga), shared-memory staging of dequantized tiles, f16 math/f32 acc, shape-specialized; subgroup-tiled fallback variant kept for A/B. THE compute-critical kernel |
| `rmsnorm` | 4×/layer + final | fused optional residual-add; weight convention (`w` vs `1+w`) is a specialization constant decided by parity tests |
| `rope_sliding` | sliding layers | θ=10k, full rotary over head_dim 256; fused QK-norm option |
| `rope_global` | global layers | θ=1M, `proportional` type, partial rotary (rotate first ¼ of 512 dims — exact formula pinned by parity tests); fused QK-norm option |
| `attn_prefill` | prefill | flash-attention-style streaming softmax over KV tiles; two variants: sliding (banded mask, window 1024) and global (causal); separate K/V buffers on both (M3 amendment: cached K ≠ cached V even on the shared-projection global layers) |
| `attn_decode_sliding` | decode | 1 query token vs ring buffer (≤1024 KV); GQA 32:16, head_dim 256 |
| `attn_decode_global` | decode | 1 query vs full context; GQA 32:4, head_dim 512, separate K/V buffers (M3 amendment — the K=V single-read optimization was based on a misreading); split-K across workgroups + reduction pass for long contexts |
| `mlp_geglu` | both | gate·GELU(tanh)·up fused where profitable; down-proj via gemv/gemm |
| `kv_append` / `kv_quant` | both | write new K/V into ring & resident global KV; optional f16→Q8_0 for cache2 page flush; Q8_0→f16 on page load |
| `logits_softcap` | last step | tied-embedding matmul (reuses gemv/gemm with embedding tensor) + tanh cap 30; emits f32 logits |
| `attn_verify_*` | MTP verify (plan 07, M7.5) | K+1 queries vs ring / resident global KV — between decode (1 query) and prefill; plus small-M (2..16) GEMM shape variants of every target matmul |
| drafter kernels | MTP draft (plan 07, M7.5) | tiny GEMMs (hidden 1024), shared-KV attention reading target KV buffers, centroid head (centroid GEMV → top-32 select → gather → scored GEMV over ~4k candidates) |
| `argmax_partial` (optional) | decode | GPU top-k prereduction if CPU sampling over 262k ever shows up in profile (don't build until measured) |

~~Attention K=V on global layers means `attn_decode_global` reads each KV element once and uses it
as both key and value.~~ **Amended in M3 (2026-06-12): wrong.** The model ties only the K/V
*projection*; cached K (weighted k_norm + rope) ≠ cached V (weightless norm, no rope), so the
global kernels bind separate K and V buffers and global attention reads 2× the KV bytes
(`docs/reference/gemma4-forward-graph.md`).

## Performance plan

- Decode targets: `gemv_q4_0` ≥ 85 % of streaming bandwidth ceiling (measure with standalone
  bandwidth probe from M0); full decode step ≤ 75 ms at 8K ctx (≈13 tok/s), ≤ 95 ms at 100K.
- Prefill: `gemm_q4_0` ≥ 30 % of f16 peak initially with coopmat, stretch ≥ 50 % (tile-shape sweep
  over the probed coopmat configs, dequant-staging layout, dual-issue-friendly epilogue). The
  subgroup-tiled non-coopmat variant is the floor/fallback; both are benched in M2.
- Profile with RGP/radv `RADV_DEBUG=...`, `VK_KHR_performance_query` if exposed; wall-clock
  per-kernel timing via timestamp queries built into the bench harness.

## Implementation steps

1. Device init + capability probe (shared with M0 probe bin); allocator; descriptor plumbing.
2. naga-oil build pipeline + first trivial kernel (vector add) end-to-end with a test.
3. `rmsnorm`, `rope_*`, `mlp` activation kernels + parity tests.
4. `gemv_q4_0` + parity + bandwidth microbench (this gates everything; do it early).
5. `gemm_q4_0` (coopmat + subgroup-tiled variants) + parity + TFLOPS microbench; shape-specialized
   vs generic-dimensions A/B.
6. Attention kernels (prefill pair, decode pair) + parity vs CPU reference with randomized
   shapes/positions, including window-boundary and ring-wraparound cases.
7. `kv_append`/quant kernels; logits + softcap.
8. Pre-recorded decode/prefill command graphs + uniform-update mechanism + timing harness.

## Testing & validation

- **Per-kernel parity:** CPU reference implementations (f64 accumulation) in `sg-gpu/tests/ref/`;
  randomized inputs over real shapes; tolerances: matmul rel-err ≤ 2e-2 elementwise vs f64 ref at
  f16 math (calibrate against llama.cpp's accepted Q4_0 error), norms/rope ≤ 1e-3. Every kernel
  variant (both layer types, both ring states) covered.
- **Determinism test:** same inputs twice → bit-identical outputs (fixed reduction order in
  kernels; no atomics-ordering dependence). Required for cache-resume bit-exactness (plan 04).
- **Stress:** 24 h loop of randomized dispatches watching for device-lost/VRAM leaks
  (vulkano leak check + `amdgpu` dmesg scrape in the harness).
- **Benchmarks:** criterion-driven microbench per kernel (GB/s for gemv, TFLOPS for gemm, µs for
  attention at ctx ∈ {1K, 8K, 32K, 128K}); results emitted as JSON for tracking (plan 06).

## Risks / open questions

- Coopmat M/N/K/type configurations exposed by RADV on gfx1151 unknown until probed; tile-shape
  choice and Q4_0 dequant-staging layout for coopmat operands need empirical tuning (M2).
- Per-shape pipeline count (~10 shapes × variants) is trivially small, but keep an eye on pipeline
  switch cost in the decode command buffer — covered by the A/B microbench.
- Subgroup size: RDNA 3.5 wave32 vs wave64 modes; kernels parameterized, probe decides.
- Max storage buffer binding size / GTT limits for the 18 GB weight buffer: may need to split
  weights into ≤4 GB bindings depending on `maxStorageBufferRange` — design the allocator for
  multi-binding from day one.
