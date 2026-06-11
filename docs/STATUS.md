# STATUS — read this first

Last updated: **2026-06-11**, working on the Framework Desktop target box. The conversation
history that produced this repo is gone; everything needed to continue is in this file,
`AGENTS.md`, and `docs/plans/`.

## Where the project stands

**M0 is complete (probe reports checked in from the target). M1 is nearly done: GGUF parser,
ModelDesc, tokenizer (100 % HF parity), chat template, and tool-call parser are all green;
the weight-upload path remains.**

Done on the target box (2026-06-11):

- M0 closed: `docs/probe/{vulkan,membw,nvme}.json` generated on this box and committed.
  Coopmat f16×f16→f32 16×16×16 available, subgroup 64, 80 GB DEVICE_LOCAL heap, NVMe
  5.8 GiB/s O_DIRECT, CPU memcpy 62 GiB/s multithread.
- `sg-gguf`: GGUF v3 parser (typed metadata, tensor table, zero-copy tensor views; malformed
  input errors, never panics — truncation/corruption sweep tests). `ModelDesc` three-way
  validation (metadata vs tensor shapes vs config expectations) passes against the real QAT
  GGUF. Dump report checked in at `docs/reference/gemma-4-31b-q4_0.gguf-dump.txt`; model
  sha256s in `docs/reference/model-checksums.txt`.

**Real-file findings (resolve several plan 00/01 verify-items, feed others):**

- Embeddings (`token_embd.weight`, tied LM head) are **Q6_K** — plan 02's tied-head matmul
  kernel must read Q6_K, and a Q6_K scalar dequant reference is still needed in `sg-gguf`.
- Global layers ship **no `attn_v` tensor**: K=V materialized as the single `attn_k`
  (5376→2048). Global q is 5376→16384 (32 heads × 512), as plan 00 predicted.
- **New, in no plan:** every layer has `blk.N.layer_output_scale.weight` (F32 scalar), and the
  file ships a top-level `rope_freqs.weight` (F32 [256], likely the proportional-RoPE
  frequency table). Both are M3 verify-items; the graph must consume them.
- `gemma4.rope.dimension_count` = 512 (global) / 256 (swa) — how partial_rotary_factor 0.25
  interacts with that and `rope_freqs` is still an M3 verify-item.
- **Tokenizer is NOT plain SPM-unigram as plan 01 assumed**: `tokenizer.ggml.model = "gemma4"`
  (not `"llama"`), with a 514 906-entry `tokenizer.ggml.merges` array, scores present but
  -1000 for early tokens, `add_bos_token = false`, `add_space_prefix = false`. Looks like
  SentencePiece-style **BPE (merge-driven)**. Pin the algorithm from llama.cpp's `gemma4`
  tokenizer handling + HF tokenizer.json before implementing; parity corpus stays the gate.
  `add_bos_token=false` likely means the chat template inserts `<bos>` itself — verify.

Done (verified on the dev machine, 2026-06-10):

- Plans 00–07 written; plan 00 is the entry point. Model facts in them come from the real configs
  (checked in under `docs/reference/`), not from memory.
- Cargo workspace, 10 crates (see `AGENTS.md` crate map). `cargo fmt` / `clippy -D warnings` /
  `cargo nextest run` (8/8) / doctests all pass.
- `sg-gguf`: Q4_0 block type + scalar dequant reference, unit-tested. This is the ground truth
  for kernel validation.
- `sg-gpu`: build-time WGSL→SPIR-V pipeline works end-to-end (`build.rs`: naga-oil 0.22
  composition → naga 29 validation → spv-out; versions verified compatible). Stub kernel compiles
  and its SPIR-V is checked by a test.
- `sg-probe`: fully implemented (vulkan caps incl. raw `vkGetPhysicalDeviceCooperativeMatrix-
  PropertiesKHR` enumeration via ash 0.38; membw; nvme with O_DIRECT). Smoke-tested against an
  Intel iGPU on the dev machine — those numbers mean nothing for the target.
- CI: `.github/workflows/ci.yml` (Tier 1, hosted) and `target-box.yml` (Tier 2, manual until the
  self-hosted runner exists).

Done since (all on the target box, 2026-06-11):

- `sg-tokenizer`: Gemma 4 BPE (byte fallback, `" "→"▁"`, 24 leftmost-longest specials), built
  from GGUF metadata with full validation. **100 % parity with HF tokenizers on a 12 104-case
  golden corpus** (`tools/gen_tokenizer_fixtures.py`) plus round-trip; streaming `DetokBuffer`;
  `SpecialTokens::Plain` mode so user text can't inject control tokens.
- Chat template hand-ported (`template::render_prompt`), byte-identical to jinja2 on the golden
  corpus (`tools/gen_template_fixtures.py`; template checked in at
  `docs/reference/gemma-4-chat-template.jinja`). Note: the upstream template *crashes* on a
  tool message whose function name is unresolvable (no `name`, no matching `tool_call_id`) —
  the server must always resolve names (plan 05).
- Streaming `TurnParser`: token-id-driven split of model output into content / thought-channel
  / tool-call events, plus the Gemma argument-syntax parser (`<|"|>`-quoted strings →
  `serde_json::Value`). Tool-calling convention (plan 01 open question) is resolved: native
  special tokens `<|turn>`/`<turn|>`, `<|channel>thought…<channel|>`,
  `<|tool_call>call:name{args}<tool_call|>`, documented in `response_schema` of
  `tokenizer_config.json`.

## Immediate next steps (in order)

1. **M1 weight upload**: `WeightSource` trait (uring O_DIRECT impl + mmap fallback) into a
   vulkano HOST_VISIBLE|DEVICE_LOCAL buffer. Also add the Q6_K scalar dequant reference
   (embeddings/tied head are Q6_K).
2. **M1 benchmarks** (plan 01): criterion tokenizer throughput (>1 M tok/s target), GGUF parse
   time, weight-load wall time.
3. Optionally set up the self-hosted runner (labels: `self-hosted, linux, framework`) and
   enable Tier 2 triggers in `target-box.yml`.

## Open questions / verify-items (do not guess these)

- Plan 00 §verify-against-reference: proportional-RoPE formula (now incl. `rope_freqs.weight`
  and `rope.dimension_count` 512/256), attn softcap on text layers, RMSNorm `w` vs `1+w`,
  attention scaling, **`layer_output_scale` semantics** → resolved by the M3 parity harness
  against transformers `gemma4` code; isolated behind swappable functions in `sg-model`.
  (Global Q-head layout and missing global v_proj: resolved by the real tensor table, see
  findings above.)
- Plan 01 §open questions: ~~embedding dtype~~ (Q6_K, resolved); Gemma 4 tool-call convention
  (from the chat template — it's in the GGUF); exact `gemma4` tokenizer algorithm.
- Plan 07 §verify-items: all MTP drafter semantics (conditioning, K, KV-sharing map, centroid
  head, acceptance rule) → from transformers ≥5.7 `gemma4_assistant` before M7.5 starts.

## Decisions already made (don't relitigate without Ada)

- Linux-only production code; no Windows fallbacks. All IO via tokio-uring on a dedicated thread.
- API scope: SSE streaming + tool use + count_tokens; **no images** (400). FIFO queue
  (depth/timeout config) for concurrent requests; single conversation in flight.
- Per-shape kernel specialization via naga-oil defines (Ada's call; A/B vs generic kernels in M2).
- Coopmat GEMM for prefill — **naga supports coopmat in WGSL; Ada has shipped it on this box.**
- f16 everywhere with f32 accumulation (target has no BF16, no NV_coop_vector).
- CPU sampling first; GPU sampler only via the escalation ladder in plan 03.
- cache2 stores global-KV pages **plus sliding-ring tail snapshots** (resume is impossible without
  them — rationale in plan 04); Q8_0-on-disk default pending the M6 quality gate.
- MTP (plan 07) in scope as M7.5, feature-flagged off until invariant 5 is green.
- Tests via cargo-nextest; deps added with `cargo add` (latest versions); commit style in git log.

## Known gotchas

- The crates.io sparse index occasionally served stale entries during `cargo add`
  (spurious "failed to select a version"); a retry / `cargo update` fixes it.
- vulkano 0.35 does not wrap the coopmat properties query — `sg-probe/src/vulkan.rs` calls it raw
  via ash; ash version must stay in lock-step with vulkano's (0.38).
- `git add` warns about LF→CRLF from the Windows-side history; harmless, goes away once the repo
  lives on Linux.
