# super-gemma

Bespoke inference server for **Gemma 4 31B QAT Q4_0** on a Framework Desktop
(AMD Ryzen AI Max+ 395, 128 GB unified RAM, Linux). Rust: axum + tokio-uring + naga-oil + vulkano.
Single-conversation text generation for AI coding agents behind an LLM-API-style `/v1/messages`
API. NVMe-backed paged radix-trie KV cache (**cache2**) for global-attention layers; in-memory
ring buffers for sliding-attention layers.

Not a general inference framework. One model, one box, one job.

**Current state and next steps: [docs/STATUS.md](docs/STATUS.md).** Working on this repo as an
agent: [AGENTS.md](AGENTS.md). Target-box bring-up: [docs/target-setup.md](docs/target-setup.md).

## Plans

| Doc | Scope |
|---|---|
| [00 — Architecture](docs/plans/00-architecture.md) | goals, model/hardware facts, budgets, system overview, milestones, risks |
| [01 — GGUF & Tokenizer](docs/plans/01-gguf-and-tokenizer.md) | loader, weights upload, SentencePiece, chat/tool template |
| [02 — GPU Runtime & Kernels](docs/plans/02-gpu-runtime-and-kernels.md) | vulkano, WGSL→SPIR-V, kernel inventory, submission model |
| [03 — Inference Pipeline](docs/plans/03-inference-pipeline.md) | forward graph, prefill/decode, sampling, engine orchestration |
| [04 — KV Caching](docs/plans/04-kv-cache.md) | sliding ring, resident global KV, cache2 trie/NVMe/eviction/snapshots |
| [05 — Server & API](docs/plans/05-server-and-api.md) | /v1/messages, SSE, tool use, auth, queue, observability |
| [06 — Testing, Validation & Benchmarking](docs/plans/06-testing-validation-benchmarking.md) | correctness ladder, invariants, CI, benchmark program |
| [07 — MTP Speculative Decoding](docs/plans/07-mtp-speculative-decoding.md) | 0.5B shared-KV drafter, draft/verify loop, acceptance, ~2–3× effective decode |

Development on any platform; **build/run targets Linux only** (io_uring). GPU- and NVMe-dependent
test tiers run on the target box.
