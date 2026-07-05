# 03 — Inference Pipeline (`sg-model`, `sg-engine`)

## Scope

The Gemma 4 forward graph (prefill and decode), sampling, stop handling, and the engine that
orchestrates a request end-to-end: prompt build → tokenize → cache2 lookup → resume-or-prefill →
decode loop → streamed output. Single conversation in flight, always.

## Forward graph (`sg-model`)

Per layer (60×, type from `layer_types` pattern — 5 sliding then 1 global, repeating):

```
x ── rmsnorm(input) ── q_proj/k_proj[/v_proj] ── qk-norm (+ weightless V-norm) ── rope(layer type, q/k only)
   ── attention(layer type) ── o_proj ── rmsnorm(post_attn) ── (+residual)
   ── rmsnorm(pre_ffn) ── geglu mlp ── rmsnorm(post_ffn) ── (+residual)
```

- Embedding lookup scaled by √5376; final rmsnorm; tied-embedding LM head; tanh softcap 30 → logits.
- **Sliding layers:** Q/K/V 32:16:16 heads × 256; K,V separate; RoPE θ=10k; attend to ring buffer
  (window 1024).
- **Global layers:** Q 32 × 512; K/V share one 4 × 512 projection but are CACHED separately
  (K: k_norm + rope; V: weightless norm, no rope — M3 finding); RoPE θ=1M proportional, partial
  rotary 0.25; attend to full context (resident global KV).
- All "verify-against-reference" details (plan 00) are isolated behind small, swappable functions
  so the M3 parity harness can flip them and identify the correct combination empirically if the
  reference reading is ambiguous: rope formula (`proportional` + `rope_freqs.weight`
  consumption), norm convention (`w` vs `1+w`), attn softcap on/off, attention scaling,
  **`layer_output_scale` semantics** (per-layer scalar found in the real GGUF, M1), and the
  **GQA head-mapping convention** — the M2 kernels assume q-head h reads kv-head
  h / (n_q/n_kv); if the reference interleaves differently, it is fixed by reordering head
  blocks at weight upload, so it must be pinned before upload code is written.
- Embedding lookup is **CPU-side** (no kernel exists, deliberately): dequantize the token's Q6_K
  row via `sg-gguf`, scale by √5376, write ~10.5 KB into the activations buffer — microseconds
  per decode token, ~3 MB per prefill chunk. Revisit only if the e2e profile disagrees.

`sg-model` is *declarative*: it describes the per-layer dispatch sequence against `sg-gpu` kernel
descriptors; `sg-gpu` records it. No Vulkan types leak above `sg-gpu`.

## Weight upload

- **One buffer per tensor.** The M0 probe pins `maxStorageBufferRange` at 4 GB, so a single
  18 GB weights buffer is impossible (the plan 02 open risk, now resolved); the largest single
  tensor (Q6_K embeddings, 1.16 GB) fits comfortably.
- Q4_0 tensors upload verbatim (the gemv/gemm kernels consume the GGUF block-pair layout
  directly). The Q6_K embedding tensor is **repacked to a 4416-byte row stride** (21 × 210-byte
  blocks padded so every row starts word-aligned) — the layout `gemv_q6_k_logits` requires; the
  CPU-side embedding lookup reads the same repacked buffer.

## Prefill

- Chunked (default 256 tokens/chunk, tunable; must stay a multiple of the gemm M_BLOCK = 64):
  bounds activation memory, lets cache2 stream page writes behind compute, and gives natural
  points for sliding-ring **tail snapshot capture** at message boundaries (plan 04).
- Per chunk: embed → 60 layers → KV appended to ring + resident global KV; logits computed only for
  the final token of the last chunk (gemv on that token's hidden state — no logits GEMM).
- **Sliding prefill attention sources keys from TWO places** (M2 finding: the chunk cannot be
  appended to the ring before attending — early queries' windows would already be overwritten).
  Order per layer: attend, then append. The kernel becomes a two-range variant of
  `attn_prefill_sliding`: history keys (positions [q0−1023, q0)) read from the ring pre-append
  via `pos % 1024` indexing, the chunk's own keys read from the chunk K/V activations; one
  streaming softmax across both ranges in position order (deterministic). No extra memory.
  Fallback if the kernel proves fiddly: a rolling linear KV scratch (last 1023 + chunk per
  layer, ~1 GB total + per-chunk copies) feeding the existing linear-view kernel unchanged.
  Global prefill needs nothing: the resident global KV is already linear.
- Prefill begins at `resume_pos` (0 if cold): cache2 hands the engine `(resume_pos, loaded global KV
  [0..resume_pos], sliding ring snapshot at resume_pos)`; the engine only computes the suffix.

## Decode loop

Steady-state per token:

1. CPU writes the sampled token's embedding row into the activations buffer (CPU-side lookup)
   and rewrites the 16-byte step buffer (`sg_gpu::StepState`: pos, kv_len_sliding,
   kv_len_global, q0 — the M2.7 contract; ring-head arithmetic is CPU bookkeeping only, the
   kernels iterate ring slots in physical order). GPU thread submits the pre-recorded decode
   graph.
2. CPU waits for completion, reads logits from unified memory. (M2.7 measured 39 µs CPU
   overhead per decode-shaped submit with the plain blocking-fence path — the < 300 µs target
   is already met; the timeline-semaphore submission is an optimization to adopt only if the
   engine's pipelining wants it.)
3. Sampler (CPU): temperature → top-k → top-p → categorical draw with per-request RNG seed
   (LLM-API API params; greedy if temperature 0).
4. Stop check: EOS ids {1, 106}, `stop_sequences` (decoded-text matcher with overlap buffer),
   `max_tokens`.
5. Streaming detok (UTF-8-safe) → SSE delta out; incremental tool-call parser fed in parallel.
6. Ring/KV bookkeeping advances; goto 1.

**Split-K policy:** the decode graph's attention split counts are push constants, baked at
record time (one graph, fixed splits — e.g. 16 sliding / 32 global; revisit with the e2e
profile). kv_len growth shrinks the per-split chunks via the step buffer; empty splits are
handled by the reducers. Split count changes the float reduction order, so it is part of the
determinism contract: same recorded graph ⇒ bit-identical reruns.

Pipelining: while GPU runs step N, CPU does sampling/detok/SSE for N−1 — hides essentially all
CPU work. Abort (client disconnect, timeout) checked each iteration; aborts must still flush
cache2 writes for the tokens already computed (a cancelled agent request often retries with the
same prefix — that's a cache *opportunity*).

**Speculative mode (M7.5, plan 07):** the loop body becomes draft → verify(K+1) → accept/rollback
→ emit up to K+1 tokens. Detok/stop/SSE handling is written batch-oriented from the start so the
non-speculative path is just the batch-size-1 case. Rollback touches only the ring head pointer
and global-KV length counter.

## Engine flow per request (`sg-engine`)

```
validate → PromptBuilder (messages+tools → token ids) 
        → cache2.lookup(ids)            # longest resumable prefix (plan 04)
        → cache2.load(prefix)           # NVMe → GPU-visible memory, async
        → prefill(suffix)               # overlapped with remaining loads
        → decode loop                   # stream out
        → cache2.commit(new pages, tail snapshot)   # async, after stream ends
```

- The FIFO queue (plan 05) delivers one request at a time; the engine owns model state exclusively.
- `count_tokens` requests bypass the queue (tokenizer-only, no model state).
- Context overflow: if prompt tokens + max_tokens > configured context limit (default 128K,
  hardware allows 256K), reject with the API's standard validation error — agents manage their own
  context; we don't truncate silently.

## Sampling details

- Logit pipeline: f32 logits → repetition handling **none** (not in LLM-API API; keep out),
  temperature scale → top_k partial-select (k ≤ 1024 via quickselect) → top_p prefix → sample.
  < 2 ms budget on 262k vocab; if profiling disagrees, enable the GPU prereduction kernel (plan 02).
- **CPU vs GPU sampling:** start CPU. On unified memory the logits copy is free, and at ~75 ms/token
  a 1–2 ms sampler plus fence-wake latency is < 3 % — pipelined behind the next GPU step anyway.
  Escalation ladder, driven by M4 profiling: (1) GPU top-k prereduction (262k → 1k candidates,
  CPU finishes — captures most of the win, keeps RNG/top-p logic testable on CPU); (2) full GPU
  sampler (temperature/top-k/top-p/categorical + counter-based RNG, e.g. philox, for seed
  determinism). Step (2) is only genuinely *enabling* for one thing: chaining K decode steps in a
  single submission with the sampled token fed GPU-side into the next embedding lookup, removing
  the CPU from the per-token critical path entirely (stop/detok then run async, possibly
  overshooting a stop by a few tokens and truncating). That's an M8 experiment, justified only if
  measured CPU-side gap per step is material; it complicates determinism tests and stop semantics,
  so it must pay for itself.
- Deterministic given (seed, prompt, cache state); bit-exact across cache-hit vs cold paths
  (depends on kernel determinism, plan 02, and exact-resume semantics, plan 04). This is a tested
  invariant, not an aspiration.

## Implementation steps

1. CPU reference model (f32, ndarray-style, slow) for a **single layer** then full graph — this is
   the parity oracle for M3 and the kernel tests' ground truth. Must load the real GGUF (scalar
   dequant from `sg-gguf`).
2. Weight upload: per-tensor buffers, Q4_0 verbatim, Q6_K repacked to the padded row stride
   (see §Weight upload); GQA head order pinned before this lands.
3. GPU graph assembly for one layer → parity vs CPU ref → all 60 layers → end-to-end single-token
   parity (M3 gate). Includes the two-range sliding-prefill kernel variant (§Prefill).
4. Chunked prefill + KV append; greedy decode loop; CLI bin (`sg run --prompt`) for eyeballing (M4).
5. Sampler + stop handling + streaming detok integration.
6. Engine orchestration with cache2 stubbed (in-memory only, M5), then real cache2 (M6).
7. Abort/cancellation paths; overlap tuning (prefill vs page-load, decode vs SSE).

## Testing & validation

- **M3 parity gate:** activation dump comparison per layer (CPU ref vs GPU) on 3 fixed prompts;
  max rel-err thresholds per tensor; any layer-type-specific bug (rope variant, K=V, window mask)
  localizes to a layer immediately. Cross-check CPU ref itself against transformers (bf16) on a
  short prompt: top-20 logit ranks must broadly agree (Q4_0 vs bf16 differences expected; compare
  Q4_0-dequantized HF run if feasible, else rank-overlap + KL thresholds vs llama.cpp on the same
  GGUF — see plan 06).
- **Perplexity:** wikitext-2 + a code corpus slice through prefill path; must match llama.cpp same-
  GGUF perplexity within 0.5 % (M4 gate).
- **Decode==prefill consistency:** logits for token N via (prefill N) vs (prefill N−1 + decode 1).
  Bit-identical is only achievable where the two paths perform the same float ops (M2 finding):
  decode attention is split-K, so the bit-exact form of this test runs a **single-split** decode
  graph at ctx ≤ 1024 (unwrapped ring; verified op-for-op equal to the prefill kernel's
  streaming softmax there). The general case (production split counts, wrapped ring) asserts
  tight tolerance + top-k rank stability instead — still catches ring/RoPE/position bugs.
  Related non-goal: K/V rows for the same token computed by decode (gemv) vs prefill (coopmat
  gemm) differ in low bits by construction — never assert bitwise equality between them; the
  determinism invariant is "given (seed, prompt, cache state)", and cached pages are
  internally consistent whichever path produced them.
- **Sampler unit tests:** distribution tests (chi-squared vs expected for known logits), top-k/p
  edge cases, determinism per seed, stop-sequence matcher property tests (random overlap splits
  across token boundaries).
- **Long-context tests:** needle-retrieval smoke at 32K/100K (correctness of global attention +
  ring wraparound at scale).
- **Abort tests:** cancel at every pipeline stage (tokenize/load/prefill/decode); no leaks, no
  poisoned state, next request clean (soak in M9).
- **Benchmarks:** TTFT vs prompt length (cold), decode tok/s vs context (1K→128K), prefill tok/s,
  CPU overhead per decode step, sampler latency. Targets calibrated from plan 00 budgets.
