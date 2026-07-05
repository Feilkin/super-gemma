# 06 — Cross-Cutting Testing, Validation & Benchmarking (`sg-validate`, `sg-bench`)

Per-component test plans live in their scoped docs (01–05). This doc covers the cross-cutting
strategy: the correctness ladder, reference baselines, CI topology, and the benchmark program.

## Correctness ladder (each rung gates the next)

| Rung | What | Oracle | Gate |
|---|---|---|---|
| 1 | GGUF parse, tokenizer, template | HF tooling fixtures | 100 % match (M1) |
| 2 | Each GPU kernel | CPU f64 reference | per-kernel tolerance (M2) |
| 3 | Per-layer activations, full graph, 1 token | CPU reference model (f32, real GGUF weights) | rel-err thresholds per layer (M3) |
| 4 | End-to-end logits & perplexity | llama.cpp on the **same GGUF file** | top-20 rank overlap ≥ threshold, KL ≤ threshold, ppl Δ ≤ 0.5 % (M4) |
| 5 | Cache-path equivalence | ourselves (cold path) | bit-identical logits f16 / bounded KL Q8_0 (M5/M6) |
| 6 | API conformance | LLM-API SDKs + captured fixtures | suites green (M7) |
| 7 | Behavioral quality | needle tests, small eval set (HumanEval-subset via the API), real agent session | no regression vs llama.cpp baseline (M8) |

Why llama.cpp as the rung-4 oracle even though we're not recreating it: it's the only independent
implementation that consumes the identical Q4_0 file, which isolates *our* numerics from
quantization effects. Differences vs bf16 transformers are expected; differences vs llama.cpp on
the same file are bugs (modulo its own kernel tolerances — hence rank/KL thresholds, not equality).
Validation scripts under `tools/` pin llama.cpp to a specific commit.

The CPU reference model (rung 3) doubles as the place where ambiguous architecture details
(proportional RoPE, norm convention, attn softcap, scaling — plan 00) are resolved: implement the
variants behind flags, find the combination that matches transformers' activations on a short
prompt, freeze it, and encode it as the spec for the GPU path.

## Determinism & cache-exactness invariants (tested, not assumed)

1. Same (prompt, seed, config, cache state) → identical output bytes.
2. (prefill N) ≡ (prefill K + resume + prefill N−K) for all K at page/snapshot boundaries —
   bit-identical logits in f16 mode.
3. Cache disabled ≡ cache enabled (f16 mode).
4. Any crash at any point → next start serves correct results (possibly cold).
5. Greedy MTP ≡ greedy non-MTP, bit-exact (from M7.5, plan 07) — over the same prompt set as 2/3.

These invariants are the soul of the test suite; they run nightly with randomized K/crash
points in addition to fixed PR-time cases.

## CI topology

- **Tier 1 — any Linux (GitHub-hosted / WSL2):** fmt, clippy (deny warnings), unit + property
  tests for gguf/tokenizer/trie/API-with-MockEngine, WGSL→SPIR-V compile of all kernel variants
  (naga validation catches shader breakage without a GPU), fuzz smoke (short budgets). Every PR.
- **Tier 2 — self-hosted runner on the Framework box:** kernel parity, M3/M4 parity suites,
  cache2 IO + crash tests against real NVMe, full integration, benchmark run. Every PR touching
  hot paths (path-filtered), plus nightly full run.
- **Nightly:** long fuzzing, randomized invariant runs, soak hours, benchmark trend update.
- Model file + golden fixtures cached on the runner; fixture-regeneration scripts (HF, llama.cpp)
  in `tools/` with pinned versions so oracles are reproducible.

## Benchmark program (`sg-bench`)

All benchmarks emit JSON (`bench/results/<git-sha>.json`); a small script renders trend reports and
fails CI on >5 % regression of headline metrics. Each metric is reported with the
hardware-derived ceiling next to it (plan 00 budgets) so "good" is defined, not vibes.

### Headline metrics

| Metric | Conditions | Initial target (calibrate in M8) |
|---|---|---|
| Decode throughput | ctx 1K / 8K / 32K / 100K | ≥ 10 / 10 / 9.5 / 8.5 tok/s (≥ 75 % of bw ceiling) |
| TTFT cold | prompt 1K / 8K / 32K / 100K | prefill-bound: prompt/prefill_tok_s + ≤ 150 ms overhead |
| TTFT warm (cache2 full-prefix hit) | same prompts | ≤ 0.5 s / ≤ 1 s / ≤ 1.5 s / ≤ 3 s (snapshot+pages load + first token) |
| TTFT partial hit | 75 % prefix cached | interpolates accordingly |
| Prefill throughput | 8K prompt | ≥ 300 tok/s initially (coopmat GEMM); stretch 500+ |
| Decode jitter | p99/p50 step time, with background cache flush | p99 ≤ 1.3 × p50 |
| Queue overhead | request → engine start | ≤ 2 ms |
| Agent-loop composite | scripted 50-turn coding-agent session (tools, growing context) | wall time + cache hit-rate tracked as the single most representative number |
| MTP effectiveness (M7.5+) | every headline row MTP on/off; acceptance rate per workload class (code/prose/tool-JSON); effective tok/s vs K | ≥ 2× effective decode on agent-loop |

### Micro/diagnostic

Kernel GB/s & TFLOPS (plan 02), tokenizer & sampler throughput, trie lookup, NVMe load curves vs
queue depth and page size (16→512 sweep to pick the default empirically), snapshot save/restore,
weight-load time, memory high-water marks (RSS, GTT), perplexity tracking (quality regression
catch — quality is a benchmark too).

### Methodology

Fixed CPU governor/GPU clocks where the platform allows, warmup iterations, ≥ 5 repetitions with
median+MAD reported, thermals logged (Strix Halo throttling would silently poison trends), dmesg
scraped for amdgpu resets. A `bench/PROTOCOL.md` documents the exact procedure so numbers are
comparable across months.

## Tooling deliverables

- `sg-validate`: subcommands `kernels`, `layers` (activation dump+diff), `logits` (vs llama.cpp),
  `ppl`, `invariants` (the four above), `tokenizer`. Used by CI and by hand during bring-up.
- `sg-bench`: subcommands matching the tables above; `--json`; workload definitions in TOML
  (including the agent-loop script).
- `tools/`: fixture generators (HF tokenizer/template dumps, llama.cpp logit dumps), llama.cpp
  pinning, model download/verify (sha256) script, M0 hardware probe.
