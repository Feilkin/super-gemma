# STATUS — read this first

Last updated: **2026-06-11**, on handoff from the Windows dev machine to the Framework Desktop
target. The conversation history that produced this repo is gone; everything needed to continue
is in this file, `AGENTS.md`, and `docs/plans/`.

## Where the project stands

**M0 (scaffolding) is code-complete and green; its target-box items are still open.**

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

## Immediate next steps (on the Framework box, in order)

1. **Box bring-up**: `docs/target-setup.md` (toolchain, Vulkan/Mesa, model downloads).
2. **Finish M0**: run `sg-probe` (all three subcommands), check reports into `docs/probe/`
   (instructions in `docs/probe/README.md`). The `cooperative_matrix_configs` list and NVMe
   numbers feed plan 02 (GEMM variants) and plan 04 (eviction cost model). Optionally set up the
   self-hosted runner (labels: `self-hosted, linux, framework`) and enable Tier 2 triggers.
3. **Start M1** (plan 01): GGUF parser against synthetic fixtures, then parse the real GGUF —
   which resolves plan 01's open questions (embedding tensor dtype; whether global layers ship a
   fused/missing v_proj). Then the SPM tokenizer + chat template port.

## Open questions / verify-items (do not guess these)

- Plan 00 §verify-against-reference: proportional-RoPE formula, attn softcap on text layers,
  RMSNorm `w` vs `1+w`, attention scaling, global Q-head layout → resolved by the M3 parity
  harness against transformers `gemma4` code; isolated behind swappable functions in `sg-model`.
- Plan 01 §open questions: embedding dtype in the QAT GGUF; Gemma 4 tool-call convention (from
  the chat template).
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
