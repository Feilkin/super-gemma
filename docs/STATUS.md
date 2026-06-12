# STATUS — read this first

Last updated: **2026-06-11**, working on the Framework Desktop target box. The conversation
history that produced this repo is gone; everything needed to continue is in this file,
`AGENTS.md`, and `docs/plans/`.

## Where the project stands

**M0 and M1 are complete.** GGUF parser, ModelDesc, tokenizer (100 % HF parity), chat
template, tool-call parser, Q6_K reference, WeightSource load path, and criterion benchmarks
are all green on the target box.

M1 benchmark numbers (this box, 2026-06-11, `cargo bench -p sg-tokenizer / -p sg-gguf`):

| Metric | Result | Target |
|---|---|---|
| encode, mixed parity corpus | **8.96 M tok/s** | > 1 M tok/s (plan 01) |
| encode, single ~300 KB doc | 5.96 M tok/s | — |
| decode | 67.6 M tok/s | — |
| GGUF parse (real file) | 22.9 ms | — |
| ModelDesc validation | 115 µs | — |
| weight load, O_DIRECT (bounce path) | 3.50 s = 4.69 GiB/s | ~2 s ballpark (plan 00); probe ceiling 5.8 GiB/s |
| weight load, mmap+memcpy (cache-warm) | 0.62 s = 26.5 GiB/s | — |

The O_DIRECT number is the worst case (Vec destination → every chunk bounces); the
phase-matched M2 layout removes the memcpy. Cache-warm mmap is fastest but only after a prior
read has paid the cold cost and polluted 17 GB of page cache.

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

- `sg-gguf::weights`: `WeightSource` trait filling a caller-provided `&mut [u8]` (so the crate
  stays vulkano-free; M2's GPU allocator passes the mapped buffer). `MmapCopySource` fallback +
  Linux `DirectSource` (O_DIRECT chunked pread; zero-copy into destinations whose 4 KiB phase
  matches `data_offset`, bounce-buffer otherwise — M2 should allocate with matching phase).
  Real-model load verified: 16.4 GiB in ~5 s via the bounce path; phase-matched should approach
  the probe's 5.8 GiB/s. **Deviation from plan 01:** load path A uses plain pread O_DIRECT, not
  tokio-uring — sequential QD1 already saturates the drive and the dedicated uring thread only
  arrives with cache2 (M6); the trait is the swap-in seam if that changes.
- `sg-gguf::q6_k`: Q6_K block type + scalar dequant reference (layout verified against upstream
  ggml `dequantize_row_q6_K`), needed because embeddings/tied head are Q6_K.

## M2 progress (2026-06-11, in flight)

Done, all parity-tested against f64 CPU references and bit-deterministic:

- GPU runtime: `GpuContext` (8060S, all required features + coopmat), unified-memory buffers,
  build-time kernel variant registry (naga-oil defines; explicit binding layouts because
  vulkano's reflection misses coopmat-only buffers), one-shot dispatch for tests.
- Kernels green: rmsnorm ×6 (5376/512/256 × w/1+w), rope ×4 (CPU-filled f64 cos/sin table —
  GPU trig loses ~1e-2 by pos 100K), geglu, gemv_q4_0 ×4 K-shapes + generic baseline,
  gemm_q4_0 (coopmat) ×8 shapes + gemm_st_q4_0 (subgroup-tiled fallback) ×8.
- **gemv_q4_0: 218 GiB/s (91 % of bandwidth ceiling — beats the ≥85 % target).** Decode is set.
- **Attention (M2.5) green: all four kernels + shared split-K reducer**, parity vs f64 reference
  (window-boundary mid-chunk, partial/full ring, uneven and empty splits) and bit-deterministic.
  Both decode kernels are split-K + reduce (`attn_reduce_d256`/`_d512`): their natural workgroup
  counts (16/4 KV heads) leave the 40-CU GPU latency-bound — sliding decode measured 322 µs →
  37 µs with 16 splits. Global decode with splits ∝ ctx holds ~1.2 ms/layer at any context
  (32 splits at 32K; 36.7 ms unsplit). K=V on global layers is native: one read serves score and
  weighted sum. GQA: workgroup per KV head computing its Q_PER_KV query heads. Softmax scale is
  a push constant (M3 verify-item pins the value). Prefill: sliding 3.6 ms/chunk (M=256, full
  windows); **global 34 ms/chunk at 8K ctx and O(ctx²)** — fine to start, the known optimization
  target is a coopmat flash-attention rewrite, deferred until e2e profiling (decision with Ada
  2026-06-12: no more kernel micro-tuning before the full pipeline runs).
- **KV plumbing + LM head (M2.6) green:** `kv_append_sliding` (ring, wraps via push-constant
  pos) / `kv_append_global` (linear); `kv_quant_q8`/`kv_dequant_q8` — Q8_0 semantics in a
  structure-of-arrays page format WE define (f16 scales array + word-aligned i8 quants array;
  plan 04 pages use this), ~155 GiB/s each way, scales bit-exact vs CPU, quants ±1 at FDiv
  rounding boundaries (GPU FDiv is 2.5 ULP; the GPU is the authoritative page producer and is
  bit-deterministic — tested). `gemv_q6_k_logits`: the tied Q6_K LM head fused with the
  tanh-30 softcap, f32 logits out — **5.45 ms/token at 198 GiB/s** (83 % of ceiling; decode
  budget item). Q6_K rows are uploaded padded 4410 → 4416 bytes (`ROW_WORDS` 1104) so every
  row starts word-aligned; the engine must repack on upload. Embedding LOOKUP (input side,
  gather + dequant + sqrt(hidden) scale verify-item) is NOT an M2 kernel — M3 decides
  CPU-vs-kernel.
- Two more f32 gotchas pinned (kernel comments + here): **GPU FDiv is 2.5 ULP** (rcp-based) —
  anything that must match the CPU bit-exactly must avoid runtime division (the Q8 scale is
  amax·(1/127), a multiply); **pack2x16float truncates toward zero on RADV** (unspecified
  rounding in SPIR-V) — use an `f16()` value conversion when RTNE matters. Also naga 29 and
  naga_oil 0.22 both miscompile `bitcast<u32>(vec2<f16>)` (per-component lowering) — avoid.
- Attention layouts (kernels and engine must agree): activations `[token × head × head_dim]`;
  ring/linear KV `[slot|token × n_kv_heads × head_dim]`; decode iterates ring slots in PHYSICAL
  order (order-invariant softmax; no ring-head arithmetic in-kernel); prefill takes a linear KV
  view with `q0` history keys, query i at key index q0+i. Split-K partials
  `[q_head × split × (head_dim + 2)]` f32 (acc, m, l); n_splits must be a deterministic
  function of kv_len for bit-exact reruns.
- **gemm_q4_0 (coopmat): 11.5–12.5 TFLOPS (≈20 % of peak; run-to-run clock drift); gemm_st
  fallback: 3.3 TFLOPS.** Target ≥30 % of peak (17.7) not yet met → see below. Was 0.4 before
  fixing three poisons: per-byte serialized global loads in dequant (now 9-word block-pair
  loads like gemv), naga's injected per-iteration loop bounding (`force_loop_bounding: false`),
  and single-lane LDS zero-init (`zero_initialize_workgroup_memory: None` — kernels never read
  unwritten LDS). Then ~1.9× from M_TILES 2→4 (64×64 C block per workgroup: halves W traffic,
  doubles FLOPs per dequant) + B tiles hoisted into registers across the M loop.
- **Tuning dead-ends, all measured (do not retry without RGP evidence; list also in the shader
  header):** LDS-staged A (−30 %), M_TILES=8 (−10 %, VGPR pressure), K-step 128 (−3 %),
  register prefetch of next weight words (−20 %), double-buffered b_tile (−40 %),
  tile-contiguous B LDS layout (−3 %), wave32 via `required_subgroup_size` (−25 %: per-lane
  acc VGPRs double → 256-VGPR ceiling; RDNA3.5 wmma shows no wave64 penalty). The plumbing for
  pinning a subgroup size (KernelBlob/Variant `subgroup_size`) is in place but unused; using
  it again requires enabling `subgroup_size_control` in `GpuContext`.
- Toolchain gotchas pinned in code comments: naga 29 spells push constants `var<immediate>`;
  naga_oil 0.22 corrupts coopmat IR (those shaders compile via plain naga, `raw: true`);
  WGSL coopmat = `enable wgpu_cooperative_matrix`, `coop_mat16x16<f16, A/B/C>`,
  `coopLoad/coopLoadT/coopStore/coopMultiplyAdd`; **naga emits user functions as real SPIR-V
  calls — a helper wrapping the hot 9-word load+dequant cost 1.6×; keep hot loops inline**;
  naga 29 implements `subgroupAdd` but not the `enable subgroups` directive — omit the
  directive and compile those shaders via plain naga (`raw: true`, naga_oil rejects them too).

## Immediate next steps (in order)

1. **gemm_q4_0 tuning to ≥30 % of peak (17.7 TFLOPS; at ~12).** The cheap structural levers
   are exhausted (see dead-ends above — every variant of the naive
   load→dequant→barrier→MMA→barrier loop measured slower). Next step is evidence-first:
   profile with RGP/`RADV_DEBUG` wave occupancy counters to find the actual stall reason
   before touching the kernel again. At ~12 TFLOPS prefill is ~190 tok/s vs the ≥300 target;
   may also revisit after M2.7 command graphs (dispatch overhead currently in every number).
2. **M2.7 pre-recorded command graphs** + uniform update + timestamp timing (plan 02 step 8) —
   the last M2 step; after it, the full-pipeline e2e profile (agreed with Ada 2026-06-12)
   decides all further kernel optimization priorities.
3. Optionally set up the self-hosted runner and enable Tier 2 triggers in `target-box.yml`.

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
