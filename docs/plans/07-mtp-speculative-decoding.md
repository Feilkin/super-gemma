# 07 — MTP Speculative Decoding (drafter: `gemma-4-31B-it-assistant`)

## Why this is in scope

Decode is memory-bandwidth-bound (~17.3 GB weight reads per token, plan 00). Speculative decoding
amortizes one weight read over K+1 verified positions, which is the single biggest lever available
on this hardware. Google claims up to 2.5–3× for the Gemma 4 family with the official MTP drafter;
the claim is consistent with our own bandwidth math (below). Coding-agent output (code, diffs,
tool-call JSON) is high-predictability text, i.e. the favorable end of acceptance-rate
distributions.

## Drafter facts (from `google/gemma-4-31B-it-assistant` config.json, fetched 2026-06-10)

| Parameter | Value |
|---|---|
| Architecture | `Gemma4AssistantForCausalLM`, ~0.5B params, bf16 release, Apache 2.0 |
| Layers | 4: 3× sliding_attention + 1× full_attention, hidden 1024, MLP 8192 (GeGLU) |
| Backbone conditioning | `backbone_hidden_size: 5376` — consumes the 31B's activations |
| KV | `num_kv_shared_layers: 4` — **shares the target's KV cache**; KV geometry identical to the 31B (sliding 16 heads × 256; global 4 × 512; `attention_k_eq_v: true`; same rope params, window 1024) |
| Output head | centroid-clustered: `num_centroids: 2048`, `centroid_intermediate_top_k: 32`; tied embeddings, vocab 262 144; no final softcapping |
| Drafting | per the launch blog: multiple tokens proposed in a **single forward pass**, verified in parallel by the target, which contributes one bonus token |

Two structural consequences worth stating loudly:

1. **cache2 and the sliding ring are untouched.** The drafter brings no KV cache of its own.
   Rejection rollback = truncate the ring head pointer and the global-KV length counter. Tail
   snapshots are only ever taken at accepted-token boundaries, so plan 04 is unchanged.
2. The drafter is tiny and stays fully resident (~1 GB f16 incl. embeddings); its per-step cost is
   one small forward pass (~0.5 GB of reads), not per-drafted-token.

## Verify-against-reference items (transformers ≥5.7 `gemma4_assistant`; gate M7.5 start)

- How backbone activations condition the drafter (projection 5376→1024? concat with token
  embeddings? which backbone layer's hidden state?).
- The single-pass multi-token mechanism: how many tokens K, chain vs tree, position handling,
  whether K is a runtime knob.
- Exact KV-sharing map (which target layers the 4 drafter layers read; whether drafter has its own
  K/V projections at all — param-count arithmetic works either way).
- Centroid head algorithm (centroid scoring → top-32 clusters → token scoring within clusters?)
  and whether `use_ordered_embeddings: false` changes the layout.
- Acceptance rule: greedy token match vs full speculative rejection sampling, and its interaction
  with temperature/top-p/top-k.
- bf16 → f16 weight conversion safety (target has no bf16): scan for |x| > 65504 and for values
  that lose meaningful precision; fall back to f32 for offending tensors (norm scales are the
  usual suspects).

## Integration design

- **Weights**: load from safetensors directly (no GGUF needed for a 0.5B model); convert bf16→f16
  at load with the overflow check above. Lives in `sg-gguf` (it's the model-files crate) as a
  second `WeightSource`.
- **`sg-model`**: drafter graph module + `speculative_step()` =
  draft (1 drafter pass) → verify (target pass at M = K+1, logits at *all* positions) →
  accept/rollback → bonus token. The verify pass is the existing prefill-chunk graph with a
  small-M variant and all-position logits.
- **`sg-gpu` additions** (all reuse the per-shape specialization machinery):
  - small-M GEMM variants of every target matmul (M ∈ {2..16} as compiled shape variants);
  - multi-query decode attention (K+1 queries vs ring / vs resident global KV) — sits between the
    existing decode (1 query) and prefill kernels;
  - drafter kernels: tiny GEMMs (hidden 1024) + shared-KV attention reusing target KV buffers;
  - centroid head: centroid GEMV (1024×2048), top-32 select, gather cluster token embeddings,
    scored GEMV over ~4k candidates.
- **`sg-engine`**: decode loop gains a speculative mode (config flag `mtp.enabled`, default off
  until parity is green; `mtp.draft_tokens` if K is a runtime knob). Detok/SSE/stop-sequence
  handling already consumes token *batches* per iteration — emit up to K+1 per step. Stop matching
  runs on the accepted prefix only.
- **Sampling**: greedy (temperature 0) → accepted tokens are exact-match, output is **bit-identical
  to non-speculative greedy** (tested invariant). Stochastic → implement whatever the reference
  acceptance rule is; if it's standard speculative rejection sampling, the output distribution is
  the target's by construction.

## Performance model (against plan 00 budgets)

Per speculative step: target verify ≈ 17.3 GB + (K+1)·KV reads + drafter ≈ 0.5 GB.
With expected accepted+bonus = A, effective bytes/token ≈ (17.8 + KV)/A:

| A (accepted + bonus) | effective tok/s ceiling @8K ctx (256 GB/s) |
|---|---|
| 2.0 | ~26 |
| 3.0 | ~38 |
| 3.5 (Google's ~2.5× claim) | ~33–35 vs our 13.8 baseline ✓ |

Realistic target: **≥ 2× measured end-to-end on the agent-loop benchmark**, acceptance-rate
dependent. Compute is not at risk: verify at M≤16 is still bandwidth-dominated; the drafter is
~1 GFLOP/step.

## Milestone: M7.5

- **Prereqs**: M4 (working prefill+decode) and M5 (in-memory caches). Independent of cache2 (M6)
  and the server (M7) — can be developed in parallel with either once M5 lands.
- Scheduled after M7 so the serving path stabilizes first; pulled earlier if M4/M5 finish ahead of
  the GGUF/target-box availability for cache2 work.
- Feature-flagged off until the full test column below is green.

## Testing & validation

- **Drafter parity** (extends the rung-3 harness): activation + logit comparison vs transformers
  `gemma4_assistant` on fixed prompts, incl. the centroid head (top-32 cluster sets must match,
  then final token logits within tolerance).
- **Invariant 5** (added to plan 06): greedy MTP output ≡ greedy non-MTP output, bit-exact, across
  the same prompt set used for cache invariants — catches rollback/position bugs precisely.
- **Acceptance accounting tests**: synthetic drafter stub with scripted proposals to unit-test
  accept/rollback/bonus logic (ring pointer, KV length, snapshot boundaries) without real models.
- **Stochastic correctness**: if rejection sampling — statistical test on a tiny-vocab toy model
  (drafted vs direct sampling distributions, chi-squared).
- **Benchmarks** (plan 06 additions): effective tok/s vs K, acceptance rate per workload class
  (code / prose / tool-JSON), MTP on/off on every headline row, drafter step latency, verify-pass
  latency vs K.
- **Quality regression**: perplexity is unchanged by construction (outputs come from the target);
  assert exactly that via invariant 5 rather than re-measuring.

## Risks

| Risk | Mitigation |
|---|---|
| Reference semantics unknown until `gemma4_assistant` code is read | verify-items gate the milestone; acceptance logic unit-tested against stubs first |
| bf16→f16 conversion artifacts | load-time scan + per-tensor f32 fallback; drafter parity harness catches the rest |
| Acceptance rate on our real agent traffic lower than claimed | bench measures it directly; K tunable; feature flag means worst case is "off" |
| Small-M GEMM variants underperform (verify pass slower than K× decode) | covered by the M2 per-shape A/B machinery; verify-pass latency is a tracked bench metric |
