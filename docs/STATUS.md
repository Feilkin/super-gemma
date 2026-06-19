# STATUS — read this first

Last updated: **2026-06-19** (int8 made the default prefill path; f16 GEMMs removed), working on the
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

Measured (median of 5, this box; bench: `sg-bench profile`). The original table below was taken at
`perf=auto`, which idles the fabric clock and reads ~30 % low (2026-06-13). **int8-MMQ is now the
DEFAULT and ONLY prefill path (2026-06-19) — the f16 prefill GEMMs were removed from the graph (kept
only as `mmq_tflops`/`gemm_variance` bench baselines), the `int8-ffn` feature flag is gone.** Current
prefill at `perf=high` (whole block int8 + L2 swizzle + 4×1 cache-blocking + weight prefetch + the
single-pass flash attention): **287 / 191 / 115 tok/s @ q0 0 / 8K / 32K** (bench: `sg-bench profile`).
Decode is unaffected (GEMV path). Targets are from plan 06.

| Phase | Result (perf=auto) | Target (plan 06) |
|---|---|---|
| decode @ 1K / 8K / 32K | **11.7 / 11.4 / 10.3 tok/s** | ≥ 10 / 10 / 9.5 ✓ |
| prefill 256-chunk @ q0 0 / 8K / 32K | **287 / 191 / 115** (high, int8 default) | ≥ 300 ✗ |
| CPU per decode step | stage 23 µs + sampler ≤ 423 µs + overhead ~310 µs | ≪ 75 ms budget ✓ |

Optimization ranking (per-layer medians from the rep-layer breakdown):

1. **`attn_prefill_global`: was 0.42 → 35.6 → 141 ms/layer at q0 0 / 8K / 32K** — 46 % of the
   chunk at 32K and growing linearly per chunk (quadratic per prompt). **ADDRESSED (2026-06-18):**
   the single-pass coopmat flash rewrite (`attn_prefill_global_flash_sp`) is wired and drops the
   dominant 32K layer to ~118 ms (−19 % kernel; +12–16 % e2e prefill at 32K, the watchdog-pressure
   fix too) — see the rank-#1 flash section. Was the clear #1; remaining prefill gap to the ≥300
   target is L2 weight traffic + attention still being O(ctx²).
2. **Coopmat GEMM** — the FFN pair (`n21504` + `n5376`) is ~70 % of short-context prefill. **LARGELY
   ADDRESSED (2026-06-14, behind `--features int8-ffn`):** the whole transformer block runs int8-MMQ
   (Q4_0 × Q8) with the L2 swizzle and the 4×1 cache-blocked tile — see the dedicated sections below.
   Prefill @ q0 0 went f16 ~182 → int8+swizzle+cache-block **270 tok/s (perf=high)**, a ~48 % arc;
   quality stays within the llama.cpp perplexity gate. The int8 path turned out faster than f16 mainly
   via *weight-traffic* wins (swizzle + tall-thin tile), not the dtype itself — the gemms are
   memory-bound, not compute-bound (the big finding below). Still short of the ≥300 stretch target;
   the residual gate is L2 size and attention at long context (#1).
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

**Rank #2 — int8-MMQ GEMM: BEATS f16, ~1.03× (2026-06-14).** `gemm_q4_0_i8.wgsl` reads the SAME Q4_0
packed weights as the f16 gemm (unpacks nibbles → i8 in LDS in-kernel; `coop_i8_lds_smoke` proves the
LDS-i8 coopLoad) and does the per-block rescale **in registers** (the coopmat-arith fork —
`f32(coop<i32>)` `OpConvertSToF` + component-wise `OpFMul` + `OpFAdd`; `docs/naga-coopmat-arith-patch.md`):
the f32 output `yacc` is register-resident, the scale built in LDS and applied with coopmat ops.
Parity-green (nrmse ~2e-4 vs the `mmq_q4_0_q8` oracle across 1×1/2×4/2×2/1×2/4×4 tilings; bench:
parity_mmq). **Best: ~10.8–11.1 TFLOPS at 2×2 = ~1.03–1.04× f16, same bench/session (bench: mmq_tflops,
perf=high)** — int8 measured *warmer* than f16 in both runs (f16 ran first/cool), so the win is
conservative. (Absolute mmq_tflops numbers run hot — no clock warmup unlike the trusted `gemm_variance`
f16 9.5; the same-run *ratio* is the valid comparison.) **This is the deployable form: no weight repack,
no 2× weight memory — it reuses the uploaded Q4_0 buffers directly.** The path, each step from RADV
`asm`/`shaderstats`, not guesses:

- **2.52 → 4.80**: the naive scale build re-read `d_a`/`d_w` from global per output element — ~64
  redundant f16 loads + ~340 address-arith ops per thread per block (tiling-invariant, which is why
  it sat ~2.5 at every tile size). Fix: load the `M_ROWS` + `N_COLS` scale *vectors* once, form the
  outer product from LDS.
- **4.80 → 7.22**: the β-loop was rolled; ACO did not overlap consecutive (independent) blocks' WMMAs,
  so each block's substep0→substep1 dependency stalled the matrix unit. Fix: **unroll the β-loop ×2**,
  interleaving two blocks' MMAs. (RDNA3 hides WMMA latency via *within-wave* ILP, not wave-switching —
  more occupancy did NOT help, unrolling did.)
- **7.22 → 7.85**: the LDS outer-product fill was serialized (`load da_l[m]` → `lgkmcnt(0)` → mul →
  store, per element). Fix: ×4-unroll the fill so independent `da_l` loads pipeline. (These three were
  measured on the *prepacked-i8* weight path — an unfair bench: int8 was spoon-fed unpacked i8 + a
  scale buffer while f16 read packed Q4_0 and dequanted in-kernel.)
- **7.85 → ~11 (2×2)**: read Q4_0 PACKED weights like f16, unpacking nibbles → i8 in LDS in-kernel.
  Half the weight bandwidth (4.5 vs 8 bits/weight) more than paid for the unpack: 2×4 went 7.85 → 9.67,
  and the sweet spot shifted to 2×2 (the i8 weight-staging + scale LDS changed the occupancy balance).
  This is also the deployable form — no repack. **The graph's int8 path should use the 2×2 variant.**

**The `_raw` "MMA ceiling" was bogus** — it's WMMA-latency-exposed (rolled loop, no concurrent work
to fill the depth-2 bubble), so it read ~1.1 TFLOPS; unrolling *it* ×2 → 3.3. The lesson: a kernel
doing *more* work is faster when that work (scale build on VALU/LDS, + unrolled WMMAs) keeps the matrix
unit fed; bare back-to-back WMMAs can't feed it alone.

4×4 is LDS-occupancy-bound by its 32 KB double-buffered `stage` (~5.8, kept in the bench as the
documented slower point). The residual binder is the scale's LDS round-trip (`da_l/dw` → `stage` →
`coopLoad`), forced because KHR coopmat1 has **no fragment-element access** (can't scale the
accumulator in registers). Fork quirk found: coopmat `+` emits `OpFAdd` regardless of scalar kind, so
*integer* coopmat accumulation is silently a float add (harmless here — `_raw` discards output, the
kernel only adds f32 `yacc` — but a latent fork bug to guard).

**FFN int8 prefill slice — WIRED + quality-validated (2026-06-14), behind `--features int8-ffn`.**
First e2e vertical slice: the FFN gate/up/down projections run the int8 2×2 gemm (`kv_quant_q8`
quantizes `fin`/`gu` to Q8 once per site → int8 gemm reads Q4_0 + the Q8 activations). **Perplexity
matches llama.cpp within tolerance** (wikitext 1120.7 vs 1119.3, rel 0.0012; code 21.92 vs 22.33, rel
0.019) — the activation-Q8 cost is **negligible** because llama.cpp itself Q8-quantizes activations
for Q4_0 matmuls, so int8 FFN tracks the reference. (Per-layer nrmse vs the *f16* GPU path is ~0.022,
the int8 envelope — divergence toward llama.cpp, not quality loss.) **Bug fixed in the same change:**
the int8 gemm `coopStore`d its output straight to `y`, which is INVISIBLE to vulkano's reflection
auto-sync (like coopLoad/`touch.wgsl`) → the recorded graph raced the consumer (0.24 nrmse). The gemm
itself was proven correct in isolation (production shapes + GPU-quant, host-synced: nrmse 2e-4); fix
is the f16 pattern — stage `yacc`→LDS→normal-store `y` (perf-neutral, 2×2 still 10.99 vs f16 10.84).
A latent correctness bug in the committed kernel, exposed only once it was used in a graph.

**RGP-validated kernel fix + e2e timing (2026-06-14).** First RADV SQTT/RGP capture (`sg-bench rgp`,
`docs/rgp-capture.md`) on the FFN down-gemm: it was **vmcnt-stalled (global weight loads) with
occupancy capped by LDS** — NOT the per-block rescale ALU I'd inferred. The 2×2's stage round-trip
wasn't the issue; the double-buffered `stage` was eating LDS and throttling waves, so the weight-load
latency couldn't hide. Fix: **`STAGE_BUFS` define — single-buffer `stage` on large-K shapes** (frees
LDS → more waves), double-buffer on small-K (the two passes' coopLoad/fill overlap, which the up shape
needs). Down-gemm **8.0 → 5.4 ms/layer** (≈ f16's 5.6), int8-ffn prefill **162 → 176 tok/s** @ q0 0,
now **~parity with f16** (182, within the ±5 % run-noise). The double buffer was *not* a barrier
saving (the per-pass `da_l` barrier already orders the single-buffer reuse) — it's an overlap win,
real only on the rescale-bound small-K shapes. **Net:** even optimized, int8-ffn is ~parity, not a
win — the ~3 % gemm edge is offset by the activation-quant passes. int8 pays off where the quant
amortizes over many small-K gemms: **attention Q/K/V (1 quant → 3 gemms, all K=5376) is the sweet
spot**; the FFN (esp. down: 1 quant → 1 large-K gemm) is the worst case.

**Sweet spot CONFIRMED — whole attention block on int8 (2026-06-14), behind `--features int8-ffn`.**
Extended the int8 path from FFN-only to attention: Q8-quantize `xn` ONCE (shared by Q/K/V — `[mg2,
N/32]` swizzled 2×2 gemms) and the attention output once (→ O gemm). 6 new swizzled int8 variants
(Q/KV/O × sliding/global; O carries `STAGE_BUFS=1` like FFN down — same large-K→n5376 shape). **e2e
prefill A/B (perf=high): 206/148/91 → 220/156/93 tok/s @ q0 0/8K/32K, +7.0 / +5.2 / +3.2 %** — a real
win, unlike FFN's parity, exactly because the single `xn` quant amortizes over 3 gemms. **Quality:
perplexity still within tolerance** (code 21.95 vs 22.33; wikitext <0.5 %) — int8 attention tracks
llama.cpp (which also Q8-quantizes attention activations). Per-layer-vs-f16 nrmse rises to **0.040**
(worst @ layer 57) — divergence *toward* llama.cpp, not quality loss; `prefill_parity --features
int8-ffn` asserts on the **global worst** (not per-layer early-exit) with a 0.045 int8 bound. So the
whole transformer block (attention QKV+O, FFN gate/up/down) now runs int8 under the feature.
**O-gemm `STAGE_BUFS` tuned (measured, not assumed):** single-buffer beats double at perf=high —
sliding k8192 **14.50 vs 12.69** (+14 %), global k16384 **14.02 vs 12.70** (+10 %); the FFN-down
analogy held, the deployed `STAGE_BUFS=1` is optimal. `mmq_tflops` is now per-case `(k, n)` so it
benches any shape; the `swz_s2_*` double-buffered variants stay as the slower-by-proof baseline.

**Cache-blocking — tall-thin 4×1 int8 tile (2026-06-14), the structural weight-reuse win.** With the
gemms memory-bound on weight reads, the lever beyond the swizzle is **more M-rows per weight load**:
`M_TILES` sets how many activation tiles share each weight tile, and the weight LDS (`wb`) scales with
`N_COLS` ONLY — so a **4×1 tile (M_TILES=4, N_TILES=1)** quadruples M-reuse while *shrinking* `wb` to
1 KB, dropping VGPR 192→108 and lifting occupancy. Swept on `mmq_tflops` (perf=high): 4×1 beats the
deployed 2×2 on **7 of 8 int8 shapes** — the K=5376 family big (**FFN up +29 %, Q +22/27 %, KV
+32/33 %**), the n5376 family modest (down +8 %, O-sliding +4 %); **O-global (largest K) regressed −6 %
on double-buffer but recovers to +3 % single-buffered** (s1 — its long-K stage caps occupancy),
mirroring the down/O `STAGE_BUFS` story. 8×1 over-spends VGPR (204) and 4×2 keeps `N_COLS=32`, both
lose to 4×1. **Deployed all 8 int8 gemm sites to `m4n1` (O-global s1); dispatch is 64-row × 16-col
blocks (reuses the f16 `mg = m_pad/64`).** Bit-identical (per-layer nrmse 0.04016 unchanged).
**e2e prefill A/B (perf=high): 220/156/93 → 270/179/101 tok/s @ q0 0/8K/32K, +22.6 / +14.9 / +8.3 %.**
Cumulative prefill arc (f16 → int8-FFN-swizzle → int8-attention → cache-blocking): **~182 → 270 tok/s
@ q0 0**. The `m8n1`/`m4n2`/`*_s1` sweep variants stay in `mmq_tflops` as proof. Next: the same tile
sweep may lift the **f16** gemms (decode path / non-int8 build), and a higher M_TILES could help once
`m_pad` exceeds 64 rows reliably.

**Deep weight prefetch — software-pipelined the Q4_0 weight load (2026-06-18, banked).** At the 4×1
tile's low (3-wave) occupancy the int8 down-gemm could not hide the DRAM weight-read latency by
switching waves, so RGP showed *every* WMMA preceded by an exposed `s_waitcnt vmcnt` (the up-front
stall was 2412 clk). Fix is in-wave ILP, not occupancy: load each block-pair's 9 weight words into a
register double-buffer (`w_cur`/`w_next`) **one β-iteration ahead**, so the load latency hides behind
the current iteration's MMA + rescale and the vmcnt wait lands at the next `w_cur = w_next` (by which
time it's done). Folded into the shared `gemm_q4_0_i8.wgsl`, so **all 8 int8 gemm sites** inherit it.
**+6.1 % kernel (15.28 → 16.20 TFLOPS, down shape, CV 0.77 %, bench: `mmq_variance` — the warm-up +
round-robin harness, since the delta is below `mmq_tflops`'s run-to-run swing); e2e prefill @ q0 0
271 → 287 tok/s (+5.9 %, the clean FFN-dominated point)**, 8K/32K move within the attention-noise
band. `parity_mmq` + `prefill_parity --features int8-ffn` green. RGP after: the up-front vmcnt stall
is gone (2412 → 898), and the residual 898-clk first-WMMA wait is raw memory latency with no more
independent work to hide it behind at 3 waves — i.e. this kernel structure is at its practical floor.
**Max-occupancy rewrite TESTED AND KILLED (2026-06-18).** `gemm_q4_0_i8_occ.wgsl` is the clean
min-footprint mirror of the deployed kernel (1×1 tile, de-interleaved β-loop, no prefetch,
single-buffer scale): it compiles to **60 VGPR vs 144** and RGP confirms it reached **11/16 waves vs
the deployed 3/16** — yet it lost **8.32 vs 15.95 TFLOPS = −47.8 %** (bench: `mmq_variance`, perf=high,
CV 0.41 %). The mechanism (RGP): despite 11 waves it pulls **LESS** memory bandwidth — **VMEM util
4.6 % (occ) vs 8.5 % (deployed)** — because 1×1 produces only 16 M-rows per weight strip vs 64, paying
~4× the weight unpack + L2→L1 traffic + 4× the workgroup launches; the extra waves just multiply stall
sites (occ **5-6 sites of 600-1600 clk** vs the deployed **single 956-clk**) instead of hiding latency.
**Occupancy ≠ memory throughput here; the lever is weight REUSE, not occupancy** — the deployed 4×1 +
prefetch kernel extracts ~2× the VMEM with ¼ the waves and is at its real floor. The "skip-LDS B"
idea is separately blocked (Q4_0 is 4-bit packed → must unpack to i8 before any `coopLoad`, and you
can't coopLoad from registers). occ kernel kept as the documented dead-end baseline (`mmq_variance`
`occ 1×1` row + `rgp` target `gemm_q4_0_i8_occ_k21504_n5376`), like the `*_s1` variant.
Audited alongside: single-buffered scale (`STAGE_BUFS=1`) is **−2.1 %** (frees LDS but never crosses
a wave threshold, so it just loses the buffering) and the β×2 MMA interleaving is neutral (+0.0 % vs a
de-interleaved variant) — both confirmed on the same stable harness; the `*_s1` variant stays as the
standing occupancy A/B baseline, the bc/pfseq prototypes were culled.

**THE big finding — prefill gemms are MEMORY-bound, not compute-bound (RGP, 2026-06-14).** RGP'd the
f16 gemm (the compute-critical kernel): **memory unit 100 % busy / 99 % STALLED, VALU 4.6 %, WMMA
idle, 25 % occupancy (4/16 waves)** — it's memory-LATENCY-bound on the Q4_0 weight reads, achieving
**~46 GB/s of the 256 GB/s** unified LPDDR5x (the "0.5 GB VRAM" is a framebuffer carve-out of the same
memory, not faster). GPU **L2 is only 2 MB** « the 65 MB weight matrix, so weights stream from DRAM
and get re-read once per M-block. Occupancy is structurally capped (LDS `b_tile` pinned by N_TILES=4);
lowering M_TILES did NOT raise it (192 VGPR / 9216 LDS unchanged) but throughput scaled **linearly
with weight reuse** — 4×4/2×4/1×4 = **11.7 / 6.4 / 3.2 TFLOPS** (`gemm_q4_0_{,m2_,m1_}k5376_n21504`,
bench: mmq_tflops) — textbook memory-bound. **So int8-vs-f16 was a sideshow: both leave the matrix
unit ~6× idle, gated by weight-read traffic.** The lever is **weight reuse, not occupancy or the data
type**: an N-strip is ~774 KB and fits in the 2 MB L2, so a **workgroup-order swizzle** (M-blocks
fast-varying → one N-strip's M-blocks run consecutively, reused from L2 instead of re-streamed) cuts
traffic — helps f16 AND int8. **DONE & it works: the `SWIZZLE` define (transposed dispatch) takes f16
4×4 from 10.6 → 12.35 TFLOPS, +16 %** (bench: mmq_tflops, `gemm_q4_0_swz_k5376_n21504`),
bit-identical to plain (parity_gemm `gemm_q4_0_swizzle_matches_plain`) — a *free* win, no tile/resource
change. **DEPLOYED to the prefill graph (2026-06-14):** the 8 f16 gemm sites load `gemm_q4_0_swz_*`
(SWIZZLE=1) with transposed `[M-blocks, N-blocks]` dispatch; `prefill_parity` green (bit-identical,
worst nrmse 0.00244 unchanged). **e2e: gate/up gemm 11.3 → 9.4 ms/layer (−17 %); prefill @ q0 0
182 → 192 tok/s (+5.6 %, the clean FFN-dominated point** — higher-q0 deltas include ±4 % cross-run
attention noise; bench: sg-bench profile). A real prefill win, diluted e2e by the attention/rms work
the swizzle doesn't touch. **Same swizzle DEPLOYED to the int8 FFN gemm (2026-06-14):** the int8 2×2
kernel is the same memory-bound shape; `SWIZZLE` takes it **11.07 → 13.29 TFLOPS, +20 %** at perf=high
(bench: mmq_tflops `gemm_q4_0_i8_swz_t22_k5376_n21504`; the earlier +12 % was a perf=auto reading).
`gemm_up_i8`/`gemm_down_i8` load the `swz` variants with transposed dispatch; `prefill_parity
--features int8-ffn` green and **bit-identical** (pre/post both 0.02255 per-layer nrmse — launch order
only; threshold made dtype-aware, 0.025 under int8-ffn vs 0.02 f16). **e2e A/B (sg-bench profile, both
perf=high, int8-ffn): prefill @ q0 0/8K/32K = 166/127/82 → 206/148/91 tok/s, +24 % / +17 % / +11 %.**
The int8 FFN up/down gemms are ~75 % of int8-ffn prefill (≈10.8 + 8.0 ms/layer), so the kernel win
carries the total; the win shrinks with q0 as un-swizzled attention takes a larger share. (Cache-
blocking for *more* reuse than the 2 MB L2 incidentally gives — DONE, the 4×1 tile section below.)
The plain `gemm_q4_0_k*` / `gemm_q4_0_i8_t22_*` stay for the bench/parity baseline; decode is
unaffected (GEMV).

**Rank #1 — coopmat flash rewrite of `attn_prefill_global`: two parked designs both lost; building a
third.** (a) LDS-resident O: 2–3× slower (32 KB o_lds → occupancy 1 + barrier-bound rescale),
discarded. (b) register-O two-pass (`attn_prefill_global_flash.wgsl`, in-tree + parity case): still
loses. **Re-measured at perf=high (2026-06-18, bench: `attn_us` group `attn_flash_cmp`, naive vs
flash µs/chunk):**

| ctx | naive (deployed) | flash v2 two-pass | flash/naive |
|---|---|---|---|
| 256 (q0≈0) | 0.48 ms | 0.63 ms | 1.31× |
| 8K | 33.8 ms | 40.9 ms | 1.21× |
| 32K | 147 ms | 164 ms | 1.11× |

Pinning `high` did **not** flip it — both kernels are memory/latency-bound, so the fabric clock buys
them ~nothing (naive unchanged vs the perf=auto 0.42/35.6/141; flash 0.66/42.6/168→0.63/40.9/164).
The gap narrows with context (1.31→1.21→1.11×) but never crosses: v2 pays for QKᵀ **twice** (two-pass,
to avoid the per-tile rescale coopmat1 can't express in registers), and that recompute cancels the
matrix-unit win. Root cause is the int8 barrier problem: KHR coopmat1 (this box, not NV_coopmat2) has
no per-element fragment access.

**Design (c) — single-pass flash with in-register rescale: BUILT & WINS (2026-06-18).**
`attn_prefill_global_flash_sp.wgsl` computes QKᵀ **once**; the online-softmax rescale
`O *= exp(m_old − m_new)` is a single component-wise coopmat multiply — build a row-broadcast `corr`
fragment in LDS, `coopLoad` it as `coop_mat16x16<f32, C>`, `o[ot] = o[ot] * cf` (the arith-fork
`OpFMul`; one small load per key tile, NOT v1's full C→f16→LDS roundtrip). Parity green
(`attn_prefill_global_flash_sp_matches_reference`). **Beats the naive kernel at every context, win
grows with ctx (bench: attn_flash_cmp, perf=high, ms/chunk = one global layer):**

| ctx | naive | flash v2 | **flash_sp** | sp vs naive |
|---|---|---|---|---|
| 256 (q0≈0) | 0.485 | 0.617 | **0.450** | **−7 %** |
| 8K | 34.5 | 40.6 | **29.3** | **−15 %** |
| 32K | 146 | 164 | **118** | **−19 %** |

**WIRED into the prefill graph (2026-06-18).** `graph.rs` global-prefill layers now dispatch
`attn_prefill_global_flash_sp` (grid `[32, m_pad/16]`) with three `touch` barriers on q/kv.k/kv.v
first — flash reads them via `coopLoad`, invisible to vulkano reflection auto-sync, so the producer
write→read barrier needs forcing (same trap as the int8 gemm). `prefill_parity` green both builds
(f16 per-layer nrmse 0.00254 ≈ the naive 0.00244; int8-ffn 0.03707 < 0.045; logits 20/20 vs oracle).
The prefill-vs-decode dtol is now dtype-aware (0.25 int8 / 0.15 f16): prefill (flash) and decode
(split-K) are different kernels — f16 agrees to Δ0.006, int8 amplifies to ~0.2 — and the f64-oracle
check is the real gate. **e2e prefill A/B (bench: sg-bench profile, perf=high) @ q0 0/8K/32K:**

| build | naive (before) | flash_sp (after) | Δ |
|---|---|---|---|
| f16 default | 193 / 134 / 85 | 187 / 143 / 99 | −3 % / +6 % / **+16 %** |
| int8-ffn | 271 / 180 / 101 | 271 / 186 / 114 | −0 % / +3 % / **+12 %** |

The win grows with context (the rank-#1 goal: long-context global attention is the dominant cost and
watchdog driver). q0 0 is ~neutral — attention is ~3 % of that FFN-bound point, and the 3 touch
barriers/global-layer cost a hair. Decode is untouched (split-K path). The two-pass v2 + naive kernels
stay registered as `attn_flash_cmp` baselines. **Watchdog headroom also improved** (the dominant 32K
kernel dropped 146→118 ms/layer). Follow-up: the naive kernel could be retired once a perplexity-gate
run confirms quality (oracle parity already green); MTP/cache2 are the next milestones.

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
2. **int8-MMQ GEMM — DONE & DEPLOYED (2026-06-14)** behind `--features int8-ffn`: the whole
   transformer block (attention QKV+O, FFN gate/up/down) runs int8 with the L2 swizzle and the 4×1
   cache-blocked tile (per-shape variants + `kv_quant_q8` activation passes, 4 bindings/site in
   `sg-model/src/graph.rs`); perplexity-gated, +48 % prefill arc. See the dedicated sections above.
   **int8 is now the DEFAULT (2026-06-19):** the `int8-ffn` feature flag and the f16 prefill GEMM
   path were removed from the graph (un-gated the 4 cfg sites + deleted the f16 `else` branches and
   the `gemm_q/kv/o/up/down` kernel fields); the f16 `gemm_q4_0*` variants stay in `build.rs` only as
   `mmq_tflops`/`gemm_variance` bench baselines. Default prefill 287 / 191 / 115 @ q0 0/8K/32K;
   prefill_parity green (worst nrmse 0.03707). **Open follow-up:** int8 (Q8) KV cache — the flash A/B
   (2026-06-19) proved long-context is K/V-streaming-bound, so a Q8 KV cache halves that traffic (the
   next long-context lever; aligns with cache2's Q8 pages). The head-dim-split flash experiment was a
   measured dead end (hs2 +25 % / hs4 +113 %: recompute + 2–4× K traffic swamps the spill it saves —
   the 157-VGPR register spill is NOT the flash bottleneck).
3. **Operational:** pin `power_dpm_force_performance_level=high` on the box (the ~30 % fabric-clock
   finding) and set `amdgpu.lockup_timeout=10000` (the watchdog finding).
4. **Rank #1 flash attention — IN PROGRESS (2026-06-18).** Parked variants re-measured at perf=high
   (still 1.1–1.3× slower; pinning didn't flip — see the rank-#1 section). Now building design (c):
   single-pass flash with an in-register rescale using the coopmat-arith ops. A/B via the `attn_us`
   `attn_flash_cmp` group; naive kernel stays deployed until (c) beats it under parity.
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
