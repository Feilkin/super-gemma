# STATUS — read this first

Last updated: **2026-06-13** (kernel-optimization + benchmark-baselining session), working on the
Framework Desktop target box. The conversation history that produced this repo is gone; everything
needed to continue is in this file, `AGENTS.md`, and `docs/plans/`.

**Perf-number rule (AGENTS.md):** every performance number here cites its benchmark + operating
point, e.g. `(bench: gemm_variance, perf=high)`. Numbers are at `perf=high` unless noted; `auto`
reads ~30 % low. When a benchmark changes, update the numbers (grep for the old value).

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
  (32 splits at 32K; 36.7 ms unsplit). ~~K=V on global layers is native: one read serves score
  and weighted sum~~ (**WRONG — amended in M3**: cached K ≠ cached V; the global kernels now
  bind separate K/V buffers, see the M3 section). GQA: workgroup per KV head computing its
  Q_PER_KV query heads. Softmax scale is a push constant (pinned 1.0 in M3). Prefill: sliding 3.6 ms/chunk (M=256, full
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
- **Command graphs (M2.7) green — M2 is complete.** `record_graph`/`submit_blocking` +
  `GraphRecorder::dispatch` (vulkano auto-sync inserts the inter-dispatch barriers), recorded
  once and re-submitted. Per-step dynamic state moved OUT of push constants (those are baked at
  record time) into a 16-byte step buffer (`StepState`: pos, kv_len_sliding, kv_len_global, q0
  — layout pinned in graph.rs and each shader header) that the CPU rewrites between submits;
  push constants now carry only record-time statics (scale, n_splits). Split-K grids are
  recorded at a fixed split count; kv_len shrinks the chunks (empty splits already handled).
  `GpuTimer` wraps timestamp queries for per-kernel GPU timing inside a graph.
  **Decode-shaped overhead: 39 µs CPU per 240-dispatch submit (target < 300 µs)** with the
  blocking fence path — the timeline-semaphore submission belongs to the engine loop (M3),
  the criterion is already met without it. Graph test: mini decode step (append K/V → split-K
  attention → reduce) recorded once, driven 3 steps by step-buffer rewrites, matches the f64
  reference each step and is bit-identical when re-driven.
- Attention layouts (kernels and engine must agree): activations `[token × head × head_dim]`;
  ring/linear KV `[slot|token × n_kv_heads × head_dim]`; decode iterates ring slots in PHYSICAL
  order (order-invariant softmax; no ring-head arithmetic in-kernel); prefill takes a linear KV
  view with `q0` history keys, query i at key index q0+i. Split-K partials
  `[q_head × split × (head_dim + 2)]` f32 (acc, m, l); n_splits must be a deterministic
  function of kv_len for bit-exact reruns.
- **gemm_q4_0 (coopmat): ~9.5 TFLOPS (bench: gemm_variance / gemm_tflops, perf=high; ~16 % of the
  ~59 TFLOPS peak); gemm_st fallback: ~3.3 TFLOPS (bench: gemm_tflops, perf=high).** Target ≥30 %
  of peak (17.7 TFLOPS) (target) not yet met → see below. (Re-baselined 2026-06-13: the original
  "11.5–12.5 TFLOPS" here was **not reproducible** even at pinned-high clocks on the same
  byte-identical bench/kernel — likely a drifty/optimistic reading; see the operating-point finding
  in the 2026-06-13 section. `auto` perf level reads ~7.2 — pin `high`.) Was 0.4 before
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

## M3 progress (2026-06-12)

**The M3 gate is green**: per-layer activation parity (CPU oracle vs GPU graph) and
end-to-end single-token logit parity on the real GGUF, plus a llama.cpp cross-check on the
same file. All verify-items are pinned — **see `docs/reference/gemma4-forward-graph.md`**,
resolved by reading transformers `gemma4` (main) + llama.cpp b9254 (the installed build) and
confirmed empirically. Highlights: RMSNorm is plain `x̂·w`; attention scale is **1.0**; no
attention softcap; `layer_output_scale` multiplies the whole hidden state at layer end; GQA
mapping is contiguous blocks (no upload reordering); embedding scale √5376 in f32;
`rope_freqs.weight` = ggml freq divisors `[1.0 ×64, 1e30 ×192]` (= partial rotary 0.25).

**Two M2 contracts were wrong and have been amended** (details in the findings doc):

- **Cached K ≠ cached V on global layers.** `attention_k_eq_v` ties only the projection; V
  gets a *weightless* RMS-norm and **no rope** (all layers have this V-norm — new dispatch).
  The two global attention shaders now bind separate K and V; `kv_append_global` runs twice;
  **global KV is 80 KB/token, not 40** (21 GB @256K — still fits; plans 00/04 numbers stale).
- **Global rope pairing was wrong** (paired `(i, i+64)`): NEOX pairs `(i, i+head_dim/2)` with
  only the first 64 pairs live. Fixed in `rope.wgsl`; sliding variants unaffected.

Landed in `sg-model` (+ `sg-gpu` amendments), all green:

- **CPU reference model** (`reference.rs`): full 60-layer f32 oracle with f64 accumulation,
  dequant-on-the-fly off the mmap (an f32 copy wouldn't fit in RAM), verify-items behind
  `Conventions` knobs, activation taps, linear KV with window-as-mask. Decode == prefill
  **bit-exact** on the oracle. Speed (Ada asked for fast oracle tests): 8-lane f64 dot
  accumulation (fixed merge order — still deterministic), `target-cpu=native` via
  `.cargo/config.toml` (AVX-512; one box, see the file's comment), logits for the last token
  only (matches plan 03 production semantics). Net: short Tier-2 suite 205 s → 25 s; the
  2054-token oracle forward 31 min → 10.4 min (now bounded by re-streaming the activation
  matrix per weight row — a blocked GEMM would fix it if nightly time ever matters).
- **rope.rs**: the ONE home of the pinned rope-table math (CPU ref and GPU graph share it).
- **llama.cpp parity** (`tests/llamacpp_parity.rs` + `tools/gen_logits_fixtures.py`):
  **all three fixtures pass, including the 2054-token window-crossing one** (argmax ok,
  overlap 18–19/20, KL 0.00001–0.016). Fixtures are generated from llama.cpp's **CPU
  backend** (`-ngl 0`, f32 accumulation) — measured 2026-06-12: at 2054 tokens llama.cpp's
  own HIP fa-on/fa-off/CPU backends disagree by up to Δ 0.41 logprob (top-20 KL 0.030
  between its two GPU modes), so the original HIP-flash fixture + tight per-token Δ
  threshold produced a spurious long-context failure. Thresholds are context-calibrated
  (short: KL ≤ 0.005 + Δ ≤ 0.15; long: KL ≤ 0.05, no per-token Δ — it cannot be tighter
  than the reference's own backend spread); the test evaluates and prints ALL metrics
  before asserting, so one expensive run yields complete data. `long_window` is `#[ignore]`
  (nightly / explicit runs).
- **Weight upload** (`weights.rs`): one buffer per tensor, Q4_0 verbatim, Q6_K embeddings
  repacked to the 4416-byte stride, norm weights f32, ones buffer for the V-norm,
  `layer_output_scale` CPU-side (baked into `add_scaled` push at record time). Simple
  mmap-memcpy path; the per-tensor O_DIRECT scatter load is an M4 startup optimization.
- **GPU graph** (`graph.rs`): decode-shaped 60-layer dispatch sequence (~23 dispatches/layer
  incl. the new `add_scaled` residual-join kernel), recordable per-range (parity drilling) or
  whole (production); driven per token by step buffer + CPU-rewritten embedding/rope-table
  buffers. Split-K baked at record time (16 sliding / 32 global).
- **M3 parity test** (`tests/gpu_parity.rs`): 5-token prompt token-by-token; per-layer
  nrmse ≤ 0.02 (observed worst **0.012**, layer 57), logits top-20 overlap **20/20** every
  token, |Δ| ≤ 0.25 on top logits; the full pre-recorded graph is **bit-identical** to the
  per-layer submission path.

## M4 progress (2026-06-12) — COMPLETE

Working end to end: **the model generates coherent text on this box** —
`cargo run --release -p sg-model --example run -- --prompt "…"` streams answers at
**12.2 tok/s decode** (plan 00 ceiling 13–14), TTFT 0.5 s on short prompts, model load
3.8 s. Landed, all parity-tested:

- **Two-range sliding prefill kernel** (`attn_prefill_sliding_ring`): history from the
  pre-append ring (`pos % 1024`) + the chunk's own K/V, one position-ordered streaming
  softmax. Kernel parity incl. wrapped-ring and window-saturation cases.
- **Chunked prefill graph** (coopmat gemm path): chunks padded to M_BLOCK 64 (default 256),
  global layers append-then-attend, sliding attend-then-append, appends bind n_real-sliced
  sources, logits for the last real row. Graphs cached per chunk shape. Per-layer parity
  ≤ 0.0025 nrmse; prefill-vs-oracle and prefill-vs-decode logits 20/20 top-20 overlap.
- **Found + fixed a recorded-graph race**: vulkano auto-sync derives barriers from SPIR-V
  reflection, which does NOT see cooperative-matrix accesses — gemm's coopLoad-only `x`
  binding got no write→read barrier (scattered ~9 % corruption on real data; direct
  dispatches were clean because fences sync everything). Fix: `touch` no-op kernel with a
  reflection-visible read_write, dispatched on the producer buffer before each gemm
  (touch.wgsl documents the mechanism). **Any future coopmat kernel in a recorded graph
  needs the same treatment.**
- **Sampler** (`sampler.rs`): temperature → top-k (quickselect ≤ 1024) → top-p →
  categorical; self-contained xoshiro256++ (seed determinism never depends on a crate
  version); with top-k off, top-p measures against the FULL distribution's mass (working
  set expands as needed). Chi-squared + edge-case tests.
- **Generation loop** (`generate.rs`): prefill → sample → decode with EOS {1, 106} /
  max_tokens / abort-callback; tokenizer-free by design. The CLI example (`examples/run.rs`)
  layers chat template / raw mode, streaming detok, and stop-sequence matching with
  longest-prefix holdback.
- **Perplexity gate** (plan 06 rung 4, the M4 exit): methodology byte-matched to
  llama-perplexity at b9254 (n_ctx 512, fresh context, NLL over the window's second half,
  **chunk-level BOS anchoring — llama.cpp overrides `add_bos_token` to true for Gemma4**,
  see the findings doc; the convention is worth ~8× in ppl on this IT model). Corpus
  fixtures committed (wikitext-2 test slice + a code snapshot);
  `tools/gen_ppl_baseline.py` pins the baselines. **wikitext: ours 1115.66 vs llama.cpp
  1119.35 — 0.33 %, PASSES the 0.5 % gate.** Code slice: 2.11 % BELOW llama.cpp — token
  streams verified identical; located by a three-way measurement (3 chunks cumulative):
  our f64-accumulation oracle 79.26, our GPU 78.99 (−0.3 %), llama.cpp GPU 82.78 (+4.4 %)
  — **our pipeline tracks the high-precision oracle; llama.cpp is the outlier** (its CPU
  and GPU backends quantize ACTIVATIONS to int8/Q8 for quantized-weight matmuls, a bias
  that code's peaked distributions amplify; its own backend spread there is ±0.6 %).
  Code-corpus tolerance calibrated to 3 % with this evidence (perplexity.rs documents the
  numbers); tightening below the reference's own bias envelope is not meaningful.
- Two GPU-watchdog lessons pinned: a single command buffer with ~1.4 s of saturated
  LM-head work tripped amdgpu soft recovery (context lost on the NEXT submit, silently
  cancelled waves on the current one) — the all-logits ppl path now submits the LM head
  in 32-row batches.

Suite hygiene: the model-heavy GPU tests each upload 17.5 GB and overflow the 80 GB heap
when nextest runs them concurrently — they're serialized via a `gpu-model` test group in
`.config/nextest.toml`. Full workspace suite: 109 tests, ~5 min.

## E2e profile (2026-06-12) — DONE; and THE WATCHDOG FINDING

`cargo run --release -p sg-bench -- profile` → `bench/results/<sha>-e2e-profile.json`.
Wall-timed uninstrumented graphs + per-dispatch timestamps over one representative layer
of each kind (extrapolate ×50/×10).

**Critical operational finding: this kernel's `amdgpu.lockup_timeout` default is 2000 ms**
— any single GPU submission over ~2 s is killed (gfx ring reset, context lost). Every "GPU
hang" chased on 2026-06-12 was this one cliff: the whole-chunk prefill graph crosses 2 s
near q0 ≈ 10K; timestamp-drained graphs and the unbatched ppl LM head crossed it earlier.
Diagnosed via `RADV_DEBUG=hang` dumps (hung pipeline with zero active waves = killed, not
looping) + the rep0-pass/rep1-fail pattern sitting exactly on a 2.0 s boundary.
Consequences, both landed:
- **Prefill submits in 6-layer segments** (`PREFILL_SEGMENT_LAYERS`, one global layer
  each): worst segment ≈ 0.3 s at 32K, ~1 s headroom at 128K. Parity unaffected (same
  ops, fences between segments).
- The perplexity LM head was already batched 32 rows/submission (same root cause,
  misattributed to a 10 s watchdog at first).
- **Recommendation for the box (Ada):** set `amdgpu.lockup_timeout=10000` (kernel
  cmdline) as a backstop — a 2 s budget is tight for an inference workstation, and the
  flash-attention rewrite only lowers, never removes, long-context submission times.

Measured (median of 5, this box; bench: `sg-bench profile`). **NOTE: taken at `perf=auto`, which
idles the fabric clock and reads ~30 % low (2026-06-13 finding) — re-profile at `perf=high` for
true numbers.** Targets are from plan 06.

| Phase | Result (perf=auto) | Target (plan 06) |
|---|---|---|
| decode @ 1K / 8K / 32K | **11.7 / 11.4 / 10.3 tok/s** | ≥ 10 / 10 / 9.5 ✓ |
| prefill 256-chunk @ q0 0 / 8K / 32K | **179 / 128 / 83 tok/s** | ≥ 300 ✗ |
| CPU per decode step | stage 23 µs + sampler ≤ 423 µs + overhead ~310 µs | ≪ 75 ms budget ✓ |

Optimization ranking (per-layer medians from the rep-layer breakdown):

1. **`attn_prefill_global`: 0.42 → 35.6 → 141 ms/layer at q0 0 / 8K / 32K** — 46 % of the
   chunk at 32K and growing linearly per chunk (quadratic per prompt). The coopmat
   flash-attention rewrite is both the prefill-throughput fix at context and the
   watchdog-pressure fix. Clear #1.
2. **Coopmat GEMM** (~9.5 TFLOPS, bench: gemm_variance/gemm_tflops, perf=high — this profile's
   per-layer ms were taken at perf=auto, so they read low; re-profile at high): the FFN pair
   (`n21504` + `n5376`) is ~70 % of short-context prefill; 179 vs ≥300 tok/s (target) is mostly
   this.
   Structural levers exhausted (M2 dead-ends); the **int8 coopmat path
   (SINT8×SINT8→SINT32, probed available) is the MMQ-style candidate** — likely faster
   AND more accurate than f16×f16 (llama.cpp's quantization bias measured in the ppl
   work was on its activation side; ours would quantize activations Q8 too — needs a
   quality gate).
3. Decode is healthy: gemv-dominated (~1.16 ms/layer of weight streaming = the bandwidth
   floor), `attn_decode_global` 1.23 ms/layer at 32K (the K≠V ×2 traffic is visible but
   only ~12 % of a decode step). LM head 5.45 ms ≈ 6 %. Tuning here buys little until
   MTP (M7.5) multiplies decode value.
4. CPU side is a non-issue (stage_chunk 3.6 ms per 256 tokens, single-threaded dequant —
   rayon it if it ever shows).

## Prefill-optimization investigation (2026-06-13) — operating point fixed, int8 path identified

**Operating point (the foundational finding).** Under `power_dpm_force_performance_level=auto` the
GPU boosts sclk (2900 MHz) and mclk (1000 MHz) but **idles the fabric/SoC clocks (fclk/socclk)** —
costing ~30 % on compute kernels. Pinning `=high` pins every clock domain and took f16 gemm from
**7.2 → ~9.5 TFLOPS, +32 % (bench: gemm_variance / gemm_tflops, perf=high vs auto).** This also
killed the phantom "12 TFLOPS" baseline: not reproducible even pinned on the byte-identical
bench/kernel (commit e68e502) — a drifty/optimistic original reading, re-baselined to ~9.5.
**Pin `high` for all benchmarking AND production** (perf-level gotcha, Known gotchas below).

**Reproducible benchmarks built (the reason to optimize now, before cache2/server overhead):**
`gemm_variance` (steady-state f16 gemm after a clock warm-up; 0.2 % CV at perf=high — trustworthy),
`mmq_tflops` (f16 vs int8 tilings, same shape). Pinned-clock methodology is the prerequisite for
every comparison — every perf number below is at perf=high unless noted.

**Rank #2 — int8-MMQ GEMM: working, understood, clear path to beating f16 (NOT a dead end).**
The naga i8 fork landed (`coop_i8_smoke` proves signed `i8×i8→i32` on the box, bit-exact;
`docs/naga-int8-coopmat-patch.md`). The kernel `gemm_q4_0_i8.wgsl` is parity-green (nrmse ~2e-4 vs
the `mmq_q4_0_q8` oracle across 1×1/2×4/2×2/1×2 tilings; bench: parity_mmq) and runs **4.4 TFLOPS
(bench: mmq_tflops, perf=high) ≈ 0.46× f16.** The bottleneck is the **per-32-block rescale**
(coopStore the i32 dot + 2 barriers/block), NOT occupancy — a tiling sweep proved it: raising
occupancy 4→8→12 waves/SIMD (2×4→2×2→1×2 tiles) made it *slower*, 4.4→3.8→1.0 TFLOPS
(bench: mmq_tflops + RADV shaderstats), because smaller tiles amortize the fixed per-block barrier
cost worse. The fix is an **in-register rescale** (convert the i32 dot to f32, apply the scale with
component-wise coopmat ops, keep Y in registers → no per-block coopStore, no per-block barriers).
SPV_KHR_cooperative_matrix + the hardware support the needed ops (`OpConvertSToF`, component-wise
`OpFMul`); naga doesn't expose them yet → **`docs/naga-coopmat-arith-patch.md`** (handed to a
separate session). int8 WMMA peak is ~2× f16 on gfx11, so the upside is real once the barriers go.

**Rank #1 — coopmat flash rewrite of `attn_prefill_global`: parked (regressed twice).** Two designs
both lost to the naive scalar kernel — (a) LDS-resident O: 2–3× slower (32 KB o_lds → occupancy 1 +
barrier-bound rescale); (b) register-O two-pass: ~1.2× slower (0.66 / 42.6 / 168 ms-per-layer vs
naive 0.42 / 35.6 / 141 @ q0 0/8K/32K; bench: `sg-bench profile` — measured at perf=auto, re-measure
at high). Same root cause as int8's barrier problem: KHR coopmat1 (the box has this, not
NV_coopmat2) has no per-element fragment access. `attn_prefill_global_flash.wgsl` + its parity case
stay in-tree (unwired). **Revisit once the coopmat-arith ops land** — the same convert /
component-wise ops may enable a better in-register flash rescale; the naive kernel stays in use
meanwhile.

**Decode** is near the bandwidth ceiling (gemv ~91 %, bench: gemv_bw); its lever is MTP (M7.5), not
these kernels. `cache2` (M6) is the orthogonal win for the append-only workload (prefix reuse
avoids cold long-context prefill).

## M5 (2026-06-12) — in-memory incremental sessions, gate green

`sg_model::Session` (plan 03 engine flow with cache2 stubbed): one resident conversation;
a request's prompt reuses the longest common prefix of the resident history and prefills
only the suffix. Reuse is **append-only**: rolling back to a mid-history position is
impossible in-memory — every position `p` past the rollback point overwrote ring slot
`p % 1024`, which belonged to position `p − 1024`, inside the rollback point's window
(this is exactly why plan 04's cache2 keeps tail snapshots; until M6, divergence and
full-prompt retries reset and recompute). Gate (`tests/session.rs`): replay of the same
request sequence is **bit-exact**; resumed-vs-cold greedy generations agree exactly;
the divergence path matches a cold run bitwise.

## Immediate next steps (in order)

1. ~~**naga coopmat-arith patch**~~ **DONE (2026-06-13):** the `Feilkin/wgpu` fork rev
   `8366d92e` adds component-wise `OpFMul` + `f32(coopmat<i32>)` conversion (`OpConvertSToF`);
   `Cargo.toml` `[patch]` bumped. Verified on the box: `coop_arith_smoke` (kernel + test) does
   `out = sc * f32(di)` bit-exact, SPIR-V shows `OpConvertSToF`/`OpFMul` on C-use coopmats
   (confirms `coopLoad` into C-use works); all coopmat parity green on the new rev.
   `docs/naga-coopmat-arith-patch.md`.
2. **int8-MMQ in-register rescale + re-bench** (now unblocked): rewrite `gemm_q4_0_i8.wgsl` to
   convert the i32 dot in-register, apply the scale with component-wise mul, accumulate Y in
   registers (scale via a 1 KB LDS buffer) — removing the per-block coopStore + barriers. Re-run
   `parity_mmq` and `mmq_tflops` (perf=high) vs the 4.4 (int8) / ~9.5 (f16) TFLOPS baseline; gate
   activation-Q8 accuracy on the perplexity harness behind a build flag.
3. **Operational:** pin `power_dpm_force_performance_level=high` on the box (the ~30 % fabric-clock
   finding) and set `amdgpu.lockup_timeout=10000` (the watchdog finding).
4. **Revisit rank #1 flash attention** with the new coopmat-arith ops (in-register rescale may now
   beat the naive kernel) and/or re-measure the parked variants at perf=high.
5. **M6: cache2** (NVMe radix trie, paging, tail snapshots, eviction) — the append-only-workload
   win; M5 substrate proven; plan 04. (Optimizing the inference kernels first is deliberate — see
   the top of this file and AGENTS.md.)
6. Optionally set up the self-hosted runner and enable Tier 2 triggers in `target-box.yml`.

## Open questions / verify-items (do not guess these)

- ~~Plan 00 §verify-against-reference~~ — **ALL RESOLVED 2026-06-12**, pinned in
  `docs/reference/gemma4-forward-graph.md` and confirmed by llama.cpp logit parity at both
  short and window-crossing (2054-token) contexts.
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

- **GPU perf level: pin `high` before measuring anything (and in production).** Under
  `power_dpm_force_performance_level=auto` the box boosts sclk/mclk but idles the fabric/SoC
  clocks (fclk/socclk) — ~30 % low on compute (f16 gemm 7.2 vs ~9.5 TFLOPS; bench: gemm_variance).
  `rocm-smi`/`pp_dpm_*` only report fclk/socclk once `high` is forced. Needs root:
  `echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level`. This
  invalidated the old "12 TFLOPS" gemm number — re-baseline at `high`.
- The crates.io sparse index occasionally served stale entries during `cargo add`
  (spurious "failed to select a version"); a retry / `cargo update` fixes it.
- vulkano 0.35 does not wrap the coopmat properties query — `sg-probe/src/vulkan.rs` calls it raw
  via ash; ash version must stay in lock-step with vulkano's (0.38).
- `git add` warns about LF→CRLF from the Windows-side history; harmless, goes away once the repo
  lives on Linux.
