# AGENTS.md — working on super-gemma

**The goal: the FASTEST Gemma 4 31B QAT Q4_0 (GGUF) inference server for the Framework Desktop
mainboard.** Not just *a* server — the fastest one. The strict one-model/one-machine focus
(Framework Desktop, AMD Ryzen AI Max+ 395, 128 GB unified RAM, Radeon 8060S iGPU, NVMe, **Linux**)
is the whole point: it *licenses and demands* hardware-specific optimization. Speed is a primary
goal, not a finishing polish — kernel optimization, cache2, and tokio_uring all carry equal weight.
Exposes an Anthropic-style `/v1/messages` API for AI coding agents. Not a general framework: no
training, no batching, no portability work. The only second model in scope is the official 0.5B MTP
drafter (`gemma-4-31B-it-assistant`) for speculative decoding — see
`docs/plans/07-mtp-speculative-decoding.md`.

**What "bespoke" means (operating principle):** hand-tailored for *this* model on *this* hardware.
We do **not** follow llama.cpp/transformers patterns — no hardware compatibility, no wide model
support to preserve. If a standard technique (e.g. flash attention) is slow on this hardware, we're
free to drop it or, better, redesign it *for* this hardware. Innovating and thinking outside the box
is expected — the quality/parity tests (plan 06) exist precisely so we can. The toolchain is not a
hard limit either: prefer naga (and upstream our changes), but extending the naga fork — or even
hand-writing SPIR-V — is on the table when it yields the best performance.

**When to optimize (why now is correct):** correctness + quality gates come first (M0–M5, done),
but with the full e2e pipeline proven, optimizing the inference kernels *before* layering on
cache2/server is deliberate — it's the cleanest point to measure (no extra overhead), and the
reproducible benchmarks built here make optimizing the later layers tractable. Building those
benchmarks and pinning the operating point (see the perf-level gotcha) is part of the work.

**Read `docs/STATUS.md` first** — current milestone state, immediate next steps, decisions already
made, and known gotchas. **The plans under `docs/plans/` are the source of truth** for architecture
and scope (`00-architecture.md` is the entry point; each crate's plan doc is referenced from its
`lib.rs`). Checked-in copies of the model configs and source URLs live in `docs/reference/`.

## Hard facts (do not "fix" these)

- **The target machine is not this machine.** Development may happen on Windows or any Linux; the
  server only runs on the Linux target. `tokio-uring`/`libc` deps are target-gated; keep production
  code Linux-only (no Windows fallbacks), keep platform-neutral code testable anywhere.
- **naga DOES support cooperative matrix (KHR) in WGSL** — verified on the target hardware. Do not
  claim otherwise or remove coopmat paths.
- Target GPU: `shaderFloat16`, `storageBuffer16BitAccess`, `VK_KHR_cooperative_matrix` — yes.
  BF16, `NV_cooperative_vector` — no. All GPU math is f16 with f32 accumulation.
- Gemma 4 model facts (layer pattern, K=V global attention, head dims) are pinned in
  `docs/plans/00-architecture.md` from the real config.json. Items marked **verify-against-reference**
  there are resolved by parity tests (M3), never by guessing.
- Local GPU/NVMe/bandwidth numbers measured on a dev machine are *not* evidence about the target.

## Commands

```sh
cargo check --workspace                                  # fast sanity
cargo nextest run --workspace                            # tests (ALWAYS nextest, not `cargo test`)
cargo test --workspace --doc                             # doctests (nextest doesn't run them)
cargo fmt --all                                          # format
cargo clippy --workspace --all-targets -- -D warnings    # lint (CI enforces)
```

- Tests run via **cargo-nextest** (config in `.config/nextest.toml`; CI uses `--profile ci`).
- Building `sg-gpu` compiles all WGSL shaders to SPIR-V via its `build.rs` (naga-oil → naga).
  A shader error is a build error — no GPU needed to catch it.

## Dependencies

- Add with `cargo add -p <crate> <dep>` (gets latest versions); per-crate manifests, no
  `workspace.dependencies` table.
- Linux-only deps must be target-gated: `cargo add -p <crate> --target 'cfg(target_os = "linux")' <dep>`.
- Member manifests inherit `version/edition/license/publish` and `[lints]` from the workspace root.

## Crate map

| Crate | Role | Plan |
|---|---|---|
| `sg-gguf` | GGUF parser, ModelDesc validation, Q4_0 types (scalar dequant = kernel ground truth) | 01 |
| `sg-tokenizer` | SentencePiece from GGUF vocab, chat/tool template, streaming detok | 01 |
| `sg-gpu` | vulkano runtime, WGSL→SPIR-V kernel library (`shaders/`, `build.rs`), command graphs | 02 |
| `sg-model` | Gemma 4 graph, sampling, CPU reference model (M3 parity oracle) | 03 |
| `sg-cache` | cache2 radix trie + NVMe pager + eviction; tail snapshots; sliding-ring bookkeeping | 04 |
| `sg-engine` | request orchestration across the GPU / uring / HTTP threads | 03 |
| `sg-server` | axum app: /v1/messages, SSE, auth, FIFO queue | 05 |
| `sg-probe` | M0 hardware probe (Vulkan caps, membw, NVMe) — report from the target goes in `docs/probe/` | 00 |
| `sg-validate` | parity/invariant harness | 06 |
| `sg-bench` | benchmark harness, JSON to `bench/results/` | 06 |

## Conventions

- **Every performance number must cite its source, inline, wherever it appears** (docs, code
  comments, commit messages, STATUS). Measured results name the benchmark and the operating point,
  e.g. `9.1 TFLOPS (bench: gemm_variance, perf=high)` or `4.4 TFLOPS (bench: mmq_tflops, perf=high)`;
  non-measured numbers are labelled `(target)`, `(probe)` (measured HW ceiling), or `(peak)`
  (theoretical). A bare number with no source is a documentation bug — an uncited "12 TFLOPS" once
  cost a full session of chasing a number that wasn't reproducible.
- **Keep numbers and benchmarks in sync.** When a benchmark's result changes, update every number
  derived from it (grep the repo for the old value). Before committing a change that affects
  performance or benchmark results, **re-run the relevant benchmark(s) and update the numbers (and
  citations) in the same commit** — at the pinned operating point (`perf=high`; see the perf-level
  gotcha, `auto` idles the fabric clock and reads ~30% low).
- Correctness ladder (plan 06) gates milestones: don't build on a rung whose tests aren't green.
- The four cache/determinism invariants in plan 06 are tested guarantees — code that would break
  bit-exact cache resume needs a plan change, not a tolerance bump.
- Golden fixtures are generated by pinned scripts in `tools/` (HF tokenizer, llama.cpp logits);
  never hand-edit fixtures.
- The model GGUF is never committed; tests that need it are gated and run on the target box
  (Tier 2 CI, `.github/workflows/target-box.yml`).
- CI Tier 1 (`ci.yml`) must stay green on hosted Linux with no GPU and no model file.
