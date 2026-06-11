# 03 — Inference Pipeline (`sg-model`, `sg-engine`)

## Scope

The Gemma 4 forward graph (prefill and decode), sampling, stop handling, and the engine that
orchestrates a request end-to-end: prompt build → tokenize → cache2 lookup → resume-or-prefill →
decode loop → streamed output. Single conversation in flight, always.

## Forward graph (`sg-model`)

Per layer (60×, type from `layer_types` pattern — 5 sliding then 1 global, repeating):

```
x ── rmsnorm(input) ── q_proj/k_proj[/v_proj] ── qk-norm ── rope(layer type)
   ── attention(layer type) ── o_proj ── rmsnorm(post_attn) ── (+residual)
   ── rmsnorm(pre_ffn) ── geglu mlp ── rmsnorm(post_ffn) ── (+residual)
```

- Embedding lookup scaled by √5376; final rmsnorm; tied-embedding LM head; tanh softcap 30 → logits.
- **Sliding layers:** Q/K/V 32:16:16 heads × 256; K,V separate; RoPE θ=10k; attend to ring buffer
  (window 1024).
- **Global layers:** Q 32 × 512; K=V 4 × 512 single projection; RoPE θ=1M proportional, partial
  rotary 0.25; attend to full context (resident global KV).
- All "verify-against-reference" details (plan 00) are isolated behind small, swappable functions
  (rope formula, norm convention, attn softcap on/off, scaling) so the M3 parity harness can flip
  them and identify the correct combination empirically if the reference reading is ambiguous.

`sg-model` is *declarative*: it describes the per-layer dispatch sequence against `sg-gpu` kernel
descriptors; `sg-gpu` records it. No Vulkan types leak above `sg-gpu`.

## Prefill

- Chunked (default 256 tokens/chunk, tunable): bounds activation memory, lets cache2 stream page
  writes behind compute, and gives natural points for sliding-ring **tail snapshot capture** at
  message boundaries (plan 04).
- Per chunk: embed → 60 layers → KV appended to ring + resident global KV; logits computed only for
  the final token of the last chunk.
- Prefill begins at `resume_pos` (0 if cold): cache2 hands the engine `(resume_pos, loaded global KV
  [0..resume_pos], sliding ring snapshot at resume_pos)`; the engine only computes the suffix.

## Decode loop

Steady-state per token:

1. GPU thread submits pre-recorded decode graph (uniform: position, ring head, kv_len, prev token).
2. CPU waits on timeline value, reads logits from unified memory.
3. Sampler (CPU): temperature → top-k → top-p → categorical draw with per-request RNG seed
   (Anthropic API params; greedy if temperature 0).
4. Stop check: EOS ids {1, 106}, `stop_sequences` (decoded-text matcher with overlap buffer),
   `max_tokens`.
5. Streaming detok (UTF-8-safe) → SSE delta out; incremental tool-call parser fed in parallel.
6. Token id written into next step's uniform; ring/KV bookkeeping advances; goto 1.

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

- Logit pipeline: f32 logits → repetition handling **none** (not in Anthropic API; keep out),
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
2. GPU graph assembly for one layer → parity vs CPU ref → all 60 layers → end-to-end single-token
   parity (M3 gate).
3. Chunked prefill + KV append; greedy decode loop; CLI bin (`sg run --prompt`) for eyeballing (M4).
4. Sampler + stop handling + streaming detok integration.
5. Engine orchestration with cache2 stubbed (in-memory only, M5), then real cache2 (M6).
6. Abort/cancellation paths; overlap tuning (prefill vs page-load, decode vs SSE).

## Testing & validation

- **M3 parity gate:** activation dump comparison per layer (CPU ref vs GPU) on 3 fixed prompts;
  max rel-err thresholds per tensor; any layer-type-specific bug (rope variant, K=V, window mask)
  localizes to a layer immediately. Cross-check CPU ref itself against transformers (bf16) on a
  short prompt: top-20 logit ranks must broadly agree (Q4_0 vs bf16 differences expected; compare
  Q4_0-dequantized HF run if feasible, else rank-overlap + KL thresholds vs llama.cpp on the same
  GGUF — see plan 06).
- **Perplexity:** wikitext-2 + a code corpus slice through prefill path; must match llama.cpp same-
  GGUF perplexity within 0.5 % (M4 gate).
- **Decode==prefill consistency:** logits for token N via (prefill N) vs (prefill N−1 + decode 1)
  bit-identical — catches ring/RoPE/position bugs.
- **Sampler unit tests:** distribution tests (chi-squared vs expected for known logits), top-k/p
  edge cases, determinism per seed, stop-sequence matcher property tests (random overlap splits
  across token boundaries).
- **Long-context tests:** needle-retrieval smoke at 32K/100K (correctness of global attention +
  ring wraparound at scale).
- **Abort tests:** cancel at every pipeline stage (tokenize/load/prefill/decode); no leaks, no
  poisoned state, next request clean (soak in M9).
- **Benchmarks:** TTFT vs prompt length (cold), decode tok/s vs context (1K→128K), prefill tok/s,
  CPU overhead per decode step, sampler latency. Targets calibrated from plan 00 budgets.
