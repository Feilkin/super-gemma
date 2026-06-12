# Gemma 4 forward graph — pinned reference semantics

Resolves every plan 00 §verify-against-reference item. Pinned 2026-06-12 from two
independent sources that agree on all points:

- **transformers `models/gemma4/modeling_gemma4.py`** (main branch, fetched 2026-06-12;
  the model's own `config.json` says `transformers_version 5.5.0.dev0`).
- **llama.cpp `src/models/gemma4.cpp` at tag b9254** — the exact build installed on the
  target box, which runs our GGUF. Authoritative for GGUF-side conventions (what the
  stored tensors mean at runtime).

Final confirmation is empirical: the M3 parity harness vs llama.cpp logits on the same
GGUF (plan 06). Until that gate is green, treat this file as "pinned reading", not proof.

## Resolved verify-items

| Item | Resolution |
|---|---|
| RMSNorm convention | **plain `x̂·w`**, NOT `(1+w)`. `x̂ = x / sqrt(mean(x²) + eps)`, eps **inside** the sqrt, eps = 1e-6, computed in f32. (HF `Gemma4RMSNorm`: weight init to ones, multiplied directly; llama.cpp `build_norm` RMS = `rms_norm · w`.) |
| Attention scale | **1.0** — no `1/√d`. (HF `self.scaling = 1.0`; llama.cpp `f_attention_scale = 1.0f`.) QK-norm replaces scaling. |
| Attention-logit softcap | **None on text layers.** Softcapping exists only in the audio encoder. Final-logit softcap tanh·30 only. |
| `layer_output_scale` semantics | Scalar multiply of the **entire hidden state at the very end of the layer** (after the FFN residual add): `x = x * s`. (HF `hidden_states *= self.layer_scalar`; llama.cpp `ggml_mul(cur, out_scale)` as the last op before `l_out`.) |
| GQA head mapping | **Contiguous blocks**: q-head `h` reads kv-head `h / (n_q/n_kv)` (HF `repeat_kv` = repeat_interleave). Matches the M2 kernel assumption — **no weight reordering at upload**. |
| Embedding scale | lookup × `√hidden_size` = √5376 in f32 (llama.cpp `ggml_scale(inpL, sqrtf(n_embd))`). NB: HF-bf16 rounds the scale to bf16 (73.3212 → 73.5, see HF PR 29402 comment); we follow llama.cpp's f32 since that GGUF run is our parity target. |
| Sliding-window boundary | key `p0` masked for query `p1` iff `p1 − p0 ≥ 1024`: attend to the **last 1024 keys including self** (llama.cpp `LLAMA_SWA_TYPE_STANDARD`). Matches the M2 1024-slot ring. |
| RoPE convention | **NEOX / rotate-half** on both layer types: pair `(i, i + head_dim/2)`, `x'ᵢ = xᵢ·cos − x_{i+d/2}·sin`, `x'_{i+d/2} = x_{i+d/2}·cos + xᵢ·sin`. Angle in f32 (HF forces f32 for cos/sin). |
| Sliding RoPE | full rotation over head_dim 256: `θᵢ = pos · 10000^(−2i/256)`, i ∈ [0, 128). |
| Proportional RoPE (global) | `θᵢ = pos · 1000000^(−2i/512) / rope_freqs[i]`, i ∈ [0, 256), pairs at stride 256. The GGUF `rope_freqs.weight` (F32 [256]) is ggml `freq_factors` (a **divisor**): values are `[1.0 ×64, 1e30 ×192]` (dumped from the file) — only the first 64 pairs rotate, the rest are identity (θ → ~0). This *is* `partial_rotary_factor 0.25` (`int(0.25·512//2) = 64` live pairs); HF builds the same table as inv_freq with a zero tail (`_compute_proportional_rope_parameters`). "Proportional" = exponent uses /512 (full head_dim), not /128. `gemma4.rope.dimension_count = 512` (= ggml `n_rot`, the full rotation span), not 128. |

## The per-layer graph (60×, no MoE / per-layer-embed for 31B: those code paths are dead)

```
x_in ── attn_norm(RMS,w) ── xn
  q = Wq·xn          → per head: q_norm(RMS,w[head_dim]) → rope
  k = Wk·xn          → per head: k_norm(RMS,w[head_dim]) → rope     → cache K
  v = Wv·xn (sliding) or the SAME k-projection output (global)
                     → per head: v_norm(RMS, NO weight)  → NO rope  → cache V
  attn = softmax(q·Kᵀ · 1.0  + mask) · V        (softmax in f32)
  o = Wo·attn ── post_attention_norm(RMS,w) ── (+ x_in) ── h
  ffn_norm(RMS,w) ── gelu_tanh(Wgate·…) ⊙ (Wup·…) ── Wdown ── post_ffw_norm(RMS,w)
  ── (+ h) ── × layer_output_scale ── x_out
```

Head order in the projection output dim is head-major (`reshape(head_dim, n_heads, …)`
in both references); within a head, NEOX pairing as above.

Final: `output_norm(RMS,w)` → tied Q6_K head → `logits = 30·tanh(logits/30)`.

## Findings that AMEND M2 contracts (discovered 2026-06-12)

1. **Cached K ≠ cached V on global layers.** `attention_k_eq_v` ties only the
   *projection*. K = weighted `k_norm` + RoPE; V = the same projection output but
   weightless RMS-norm and **no RoPE**. Consequences:
   - `attn_decode_global` / `attn_prefill_global` must bind **separate K and V buffers**
     (M2 built them with one `kv` buffer read once for score *and* weighted sum).
   - `kv_append_global` runs **twice** per layer (K rows, V rows) — the kernel itself is
     a generic copy, no shader change.
   - Global KV is **80 KB/token, not 40** (plan 00 §budgets, plan 04 page format/cost
     model): 10.5 GB @128K, 21 GB @256K. Still fits.
2. **Global rope kernel pairing is wrong in M2.** The `rope` variant with
   `ROT_DIMS=128` rotates pairs `(i, i+64)`; the reference pairs `(i, i+256)` with only
   the first 64 pairs live. Fix: pair stride is `HEAD_DIM/2`, pair count
   `ROT_PAIRS = 64`. The sliding variant (full rotation) is unaffected — the two
   conventions coincide when ROT_DIMS = HEAD_DIM.
3. **V-norm is a new dispatch** in the per-layer sequence (all 60 layers): weightless
   RMS-norm over head_dim rows. No new kernel — reuse the `w` rmsnorm variant with a
   ones weight buffer (`x̂·1.0` is exact).
4. The `1+w` rmsnorm variants are dead (convention pinned to `w`); keep or cull at M3
   cleanup.

## Notes

- **BOS convention (found via the M4 ppl gate):** llama.cpp *overrides* the GGUF's
  `tokenizer.ggml.add_bos_token = false` to **true** for the Gemma4 arch (`load: override
  'tokenizer.ggml.add_bos_token' to 'true' for Gemma4`), so plain completions and its
  perplexity tool always start the stream with `<bos>` (ppl additionally replaces each
  chunk's first fed token with it). Our chat path is unaffected (the template emits
  `<bos>` itself), but anything comparing against llama.cpp on raw text must mirror the
  override — the effect is huge on this IT model: BOS-anchored raw text scores ~8× worse
  ppl than un-anchored mid-text continuation (measured 2026-06-12).

- The cos/sin tables the M2 rope kernels consume are CPU-built; the proportional
  formula above lives in ONE place (`sg-model`'s table builder) for both the CPU
  reference and the GPU graph. Build angles in f64, store (cos, sin) f32; consume
  `rope_freqs.weight` from the GGUF rather than hard-coding the factor table, and
  validate it equals `[1.0 ×64, 1e30 ×192]` at load.
- `num_kv_shared_layers = 0`, `enable_moe_block = false`,
  `hidden_size_per_layer_input = 0` for 31B: the KV-sharing / MoE / per-layer-embed
  branches in both references are inert for this model.
- llama.cpp computes V-norm with `ggml_rms_norm` (weightless) and HF with
  `with_scale=False` — identical semantics.
