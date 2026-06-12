# 00 — High-Level Architecture

## Mission

A bespoke, single-purpose inference server for **Gemma 4 31B QAT Q4_0 (GGUF)** on a
**Framework Desktop (AMD Ryzen AI Max+ 395 "Strix Halo", 128 GB unified LPDDR5X, Radeon 8060S iGPU, NVMe)**
running **Linux**. One conversation at a time, text generation for AI coding agents, nothing else.

**Non-goals:** training, batching, multi-model support, multi-node, vision/audio input,
portability beyond this hardware/model pair, recreating llama.cpp/transformers.

## Model facts (from `google/gemma-4-31B-it` config.json, verified 2026-06-09)

| Parameter | Value |
|---|---|
| Layers | 60, repeating pattern: 5× sliding_attention + 1× full_attention (50 sliding, 10 global) |
| hidden_size | 5376 |
| MLP | gated GeGLU (gelu_pytorch_tanh), intermediate 21504 |
| Sliding layers | GQA 32 Q heads : 16 KV heads, head_dim 256, **separate K and V**, RoPE θ=10 000 (full rotary), window 1024 |
| Global layers | 32 Q heads × head_dim **512**, **4 KV heads** (`num_global_key_value_heads`), **K=V single shared projection** (`attention_k_eq_v: true`), RoPE θ=1 000 000, type `proportional`, partial_rotary_factor 0.25 |
| QK-norm | RMSNorm on Q and K per head (per reference impl) |
| Norms per layer | 4 RMSNorms: input, post-attention, pre-FFN, post-FFN |
| Embeddings | tied with LM head, scaled by √hidden_size, vocab 262 144 |
| Final logits | tanh softcapping at 30.0 |
| Context | max_position_embeddings 262 144 |
| EOS | token ids 1 (`<eos>`) and 106 (`<end_of_turn>`); BOS 2; PAD 0 |
| rms_norm_eps | 1e-6 |

**Verify-against-reference items** (transformers `models/gemma4/modeling_gemma4.py`, transformers ≥5.5;
each gets a parity test in M3 before being trusted):
- Exact `proportional` RoPE formula and how partial_rotary_factor 0.25 is applied (likely: only the
  first 128 of 512 dims rotated on global layers).
- Whether attention-logit softcapping applies to text layers (reference suggests it exists in
  `eager_attention_forward`; Gemma 3 dropped it in favor of QK-norm — confirm from code, then from GGUF metadata).
- RMSNorm weight convention: `x̂·w` vs Gemma-classic `x̂·(1+w)`. GGUF exporters sometimes pre-add the 1.
- Attention scaling: `1/√head_dim` vs `query_pre_attn_scalar` vs `scaling=1.0` after QK-norm.
- Global-layer Q layout: 32×512 (q_proj 5376→16384) per reference read — confirm via GGUF tensor shapes.

## Hardware facts & first-order budgets

- **Memory:** 128 GB unified LPDDR5X-8000, 256-bit bus → **~256 GB/s** theoretical bandwidth, shared CPU/GPU.
  iGPU accesses it via GTT; all buffers are HOST_VISIBLE|DEVICE_LOCAL → zero-copy CPU↔GPU.
- **GPU:** Radeon 8060S, RDNA 3.5, 40 CUs, ~59 TFLOPS peak f16 (dual-issue). RADV (Mesa) Vulkan driver.
  Confirmed available: `shaderFloat16`, `storageBuffer16BitAccess`, `VK_KHR_cooperative_matrix`
  (usable from WGSL via naga — verified on the target machine). **Not** available: BF16, NV_cooperative_vector.
  All math is f16 with f32 accumulation.
- **NVMe:** assume ≥5 GB/s sequential read (measure in M0; the cache2 cost model takes the measured number).

| Budget item | Size |
|---|---|
| Weights Q4_0 (~4.5 bit/weight, 31B) | ~18 GB |
| Sliding KV ring (50 layers × 1024 tok × (K+V) 16 KB, f16) | ~820 MB |
| Global KV per token (10 layers × 4 heads × 512 × f16, **K and V cached separately** — the shared projection diverges through the norms/rope, see `docs/reference/gemma4-forward-graph.md`) | **80 KB/token** |
| Global KV resident @128K ctx / @256K ctx | 10.5 GB / 21 GB |
| Activations + scratch | < 1 GB |
| **Total GPU-visible @256K ctx** | **< 32 GB** (huge headroom in 128 GB) |

Decode is bandwidth-bound: ~17.3 GB weights + ~0.9 GB sliding-KV + (82 KB × ctx) global-KV per token
→ ceiling ≈ **13–14 tok/s** at short context, ≈ **11 tok/s** at 100K; expect 80–90 % of that from
real streaming kernels. **MTP speculative decoding (plan 07) multiplies this by the expected
accepted-tokens-per-verify (~2–3.5× on agent workloads → ~25–35 tok/s effective)** by amortizing
one weight read over K+1 verified positions. Prefill is compute-bound: 62 GFLOP/token →
**~300–600 tok/s** at 30–60 % matmul efficiency using cooperative-matrix GEMM (KHR coopmat is
available through naga on the target — verified). These ceilings calibrate all benchmark targets.

**Consequence that shapes everything:** prefill at ~300 tok/s vs NVMe at ~5 GB/s means loading cached
global KV (80 KB/token ≈ 62K tok/s) is **~2 orders of magnitude cheaper than recomputing**. cache2
exists to convert NVMe bytes into skipped prefill.

## System overview

```
                    ┌──────────────────────────────────────────────┐
 HTTP (axum) ──────►│ sg-server: /v1/messages, /v1/messages/        │
  x-api-key auth    │ count_tokens; SSE streaming; FIFO queue       │
                    └───────────────┬──────────────────────────────┘
                                    │ InferenceRequest (tokens, sampling, abort)
                    ┌───────────────▼──────────────────────────────┐
                    │ sg-engine: scheduler / conversation pipeline  │
                    │  prompt build → tokenize → cache2 lookup →    │
                    │  resume-or-prefill → decode loop → detokenize │
                    └───┬───────────────────────┬──────────────────┘
            commands    │                       │ page load/store, snapshot
                    ┌───▼────────────┐   ┌──────▼─────────────────┐
                    │ sg-gpu          │   │ sg-cache (cache2)      │
                    │ vulkano ctx,    │   │ RAM: radix-trie index, │
                    │ WGSL→SPIR-V     │   │  resident global KV,   │
                    │ kernels,        │   │  sliding ring buffer   │
                    │ dedicated submit│   │ NVMe: paged KV extents,│
                    │ thread          │   │  tail snapshots,       │
                    └───┬────────────┘   │  tokio-uring O_DIRECT  │
                        │                └──────┬─────────────────┘
                  unified memory (zero-copy)    │ dedicated uring thread
                        └───────────┬───────────┘
                            128 GB LPDDR5X / NVMe
```

## Threading model

- **axum / tokio multi-thread runtime**: HTTP, auth, SSE fan-out, request queue.
- **GPU executor thread** (1 dedicated OS thread): owns the vulkano queue, records/submits command
  buffers, waits on fences. Receives work via bounded channel; sends back sampled tokens. No Vulkan
  calls from any other thread.
- **cache2 IO thread** (1 dedicated OS thread): runs a `tokio-uring` current-thread runtime; owns all
  NVMe reads/writes (O_DIRECT, registered buffers targeting GPU-mapped memory). Channel API.
- **Engine task**: async orchestration on the main runtime; talks to both threads via channels.
  Single in-flight conversation by construction (FIFO).

This isolates the two non-Send subsystems (vulkano queue discipline, tokio-uring) and keeps the decode
hot loop free of runtime scheduling jitter.

## Crate layout (Cargo workspace)

| Crate | Contents | Plan |
|---|---|---|
| `sg-gguf` | mmap GGUF parser, metadata, tensor table, Q4_0 types | 01 |
| `sg-tokenizer` | SentencePiece encode/decode from GGUF vocab, chat template, tool-call format | 01 |
| `sg-gpu` | vulkano device/queue, naga-oil WGSL build, kernel library, command-graph recording | 02 |
| `sg-model` | Gemma 4 graph: layer definitions, prefill/decode passes, sampling | 03 |
| `sg-cache` | cache2 radix trie + NVMe pager + eviction; sliding ring; tail snapshots | 04 |
| `sg-engine` | scheduler, conversation pipeline, cache↔gpu glue | 03 |
| `sg-server` | axum app, Anthropic-style API, auth, SSE, queue | 05 |
| `sg-validate` (bin) | parity harness vs reference (activation dumps, logit KL) | 06 |
| `sg-bench` (bin) | end-to-end + micro benchmarks, JSON output | 06 |

Dev happens on this Windows machine; **build/test/run targets Linux only** (tokio-uring). CI and all
execution via WSL2 or the target box. No Windows fallback paths in production code; unit tests that
don't touch uring/Vulkan remain platform-neutral so most of the suite runs anywhere.

## Key decisions (and why)

1. **Q4_0 weights stay quantized in memory; dequant fused into matmul kernels.** Bandwidth is the
   decode bottleneck; reading 4.5 bits/weight instead of 16 is the whole game.
2. **f16 KV everywhere by default; Q8_0 option for NVMe pages and snapshots** (config flag).
   Verify quality impact in M6 with perplexity tests before defaulting to Q8_0 on disk.
3. **Global-layer KV is the only per-token state that grows with context** (sliding layers cap at
   1024 tokens). cache2 therefore stores *only* global KV pages — exactly as specified — plus
   **tail snapshots** of the sliding ring, because resuming a conversation exactly requires the
   sliding state too (see plan 04 for why this is unavoidable).
4. **Decode command buffers are pre-recorded once and resubmitted per token** (static graph;
   position/length via push constants or a small uniform update). CPU overhead per token must be ≪ 1 ms.
5. **Sampling on CPU** from GPU-written logits in unified memory (262 144 × f32 = 1 MB, zero-copy).
   Simple, debuggable, and free on this hardware.
6. **No Jinja at runtime**: the HF chat template (incl. tool-call format) is hand-ported to Rust and
   golden-tested against `transformers.apply_chat_template` outputs.

## Milestones

| # | Deliverable | Exit criterion |
|---|---|---|
| M0 | Workspace, CI (WSL2 + target box runner), hardware probe bin (Vulkan caps, NVMe speed, mem bw) | probe report checked in |
| M1 | `sg-gguf` + `sg-tokenizer` | parses real GGUF; tokenizer parity 100 % on golden corpus |
| M2 | `sg-gpu` kernel library | every kernel passes f32-reference parity within tolerance |
| M3 | Single-token forward pass | per-layer activation parity vs reference impl on same GGUF (see 06) |
| M4 | Full prefill + decode, CLI text generation | coherent output; perplexity within noise of llama.cpp Q4_0 on same GGUF |
| M5 | In-memory caching (sliding ring + resident global KV), incremental decode | cache-on vs cache-off logits bit-identical |
| M6 | cache2 NVMe trie + snapshots + eviction | warm-resume TTFT meets target; crash-consistency tests pass |
| M7 | `sg-server` API complete | Anthropic-SDK-driven integration suite green; agent (e.g. coding agent) runs against it |
| M7.5 | MTP speculative decoding (plan 07) | greedy MTP ≡ greedy non-MTP bit-exact; ≥2× effective tok/s on agent-loop bench |
| M8 | Performance tuning | hits calibrated targets (see 06) or documented why not |
| M9 | Hardening | fuzzing, fault injection, soak test (24 h agent loop) clean |

M1, M2 are parallelizable. M5 before M6 so cache2 lands on a proven in-memory substrate. M7.5 only
needs M4+M5 and can be pulled earlier in parallel with M6/M7 if scheduling allows.

## Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Coopmat tile shapes / scheduling on RADV gfx1151 may underperform | prefill below the 30–60 % efficiency band | bench coopmat GEMM early (M2); fall back to subgroup-tiled GEMM per shape if a coopmat variant loses; kernel interface opaque to callers either way |
| Gemma 4 fine details misread (proportional RoPE, softcap, norm convention) | garbage output, subtle quality loss | M3 activation-parity harness catches all of these layer-by-layer before anything is built on top |
| RADV quirks on gfx1151 (subgroup ops, f16 storage, GTT size limits) | kernel rewrites | M0 probe bin enumerates caps; kernels gated on probed features; raise `amdgpu.gttsize` if needed |
| tokio-uring + O_DIRECT into GPU-mapped memory not accepted by kernel/driver | extra memcpy on cache load | fallback: uring read into registered bounce buffers + memcpy (still unified memory, ~10 GB/s); measure both in M0 |
| QAT Q4_0 GGUF tensor layout deviates from config (e.g. fused QKV, q8 embeddings) | loader rework | loader derives shapes from tensor table, validates against config-derived expectations, fails loudly with diff |
| Sliding-state snapshots are large (~820 MB f16) | NVMe space pressure, slow resume | Q8_0 snapshots (~410 MB), snapshot only at message boundaries, eviction cost model weighs them honestly (plan 04) |
| Single dev hardware target unavailable for CI | regressions land unnoticed | self-hosted runner on the Framework box for the GPU/uring suites; everything else runs in plain Linux CI |
| MTP drafter semantics (conditioning, drafting mechanism, acceptance rule) unconfirmed until reference code is read; bf16 release vs no-bf16 target | M7.5 slip or quality bug | verify-items gate the milestone (plan 07); feature-flagged off until invariant 5 green; bf16→f16 load-time scan with f32 fallback |
