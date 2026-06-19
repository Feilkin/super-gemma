# Q8 KV cache — Piece A SHIPPED (reference) + Piece B plan (int8 V/PV)

**Piece A (Q8 global K cache) is done and merged to `main` (2026-06-19).** This
doc is now (a) the as-landed reference for what exists, and (b) the starting
point for **Piece B — int8 V/PV**, which is intended for a clean context. Read
§1–§2 (the Q8_0 format + layout reasoning, still load-bearing) and the "Piece A
as-landed" map before starting B.

## Status

- **Piece A — DONE, quality-green, e2e win banked.** Commits `d9a7d30` (code),
  `c57d06e` / `47e2bca` / `46104c5` (docs + tests) on `main`. Global K is Q8
  (i8 quants + f16 per-32-block scales); **V stays f16**, sliding stays f16.
  - Gates: kernel parity (`attn_prefill_global_flash_sp_iq_matches_reference`,
    `attn_decode_global_q8k_matches_reference`); `prefill_parity` per-layer
    nrmse 0.039; `gpu_parity` decode worst 0.025 (logits 20/20); **perplexity
    wikitext 0.10 % / code 1.11 %** within tolerance.
  - e2e (sg-bench profile, perf=high): prefill **287/191/115 → 286/209/141
    tok/s @ q0 0/8K/32K** (+0 % / +9.4 % / +22.6 %); global attention layer
    118 → 79 ms at 32K. Decode unchanged. ~67 MB/global-layer saved.
- **Piece B — NOT started.** int8 V/PV, to halve V traffic too (likely another
  similar long-context increment). The harder kernel problem (§B). V's quant
  axis ≠ the PV contraction — that's the whole difficulty.

## Piece A as-landed — the map B extends

Kernels (sg-gpu `shaders/` + `build.rs`):
- `attn_prefill_global_flash_sp_iq.wgsl` — global prefill: int8 QKᵀ (Q8 K +
  pre-quant i8 Q, per-block rescale), **f16 PV** (bindings: `q_i8, q_scales,
  k_quants, k_scales, v, out, step`). PV is the f16 block at lines ~156–165.
- `attn_decode_global_q8k.wgsl` — global decode: scalar-dequant K (i8+f16) +
  subgroupAdd, **f16 V** (bindings: `q, k_quants, k_scales, v, part, step`).
- `kv_append_global_q8.wgsl` — f16→Q8 quantize-and-append for global K.
- Retired from the graph (kept in build.rs as bench/parity baselines, NOT in
  `Kernels`): `attn_decode_global`, `attn_prefill_global_flash_sp`,
  `attn_prefill_global_flash_sp_q8` (the 7×-slow f16-convert dead end).

Graph (`crates/sg-model/src/graph.rs`, anchors as of `46104c5`):
- `KvStore` (struct ~113): sliding `k: Some([u16])`; global `k_quants:
  Some([u32])` + `k_scales: Some([u16])`; **`v: [u16]` f16 on both kinds**.
  Alloc ~365–386 (sliding vs global branch).
- Prefill global (`record_prefill_layer`): append `do_appends` ~1126 (global K
  → `kv_append_global_q8` ~1145, **V → `kv_append_global` ~1154 f16**);
  Q-quant ~1196 (`kv_quant_q8` on `q` → `p.q_i8_gl`/`p.q_scales_gl`, fields
  ~253, alloc ~356); touches ~1209–1212; `prefill_gl_iq` dispatch ~1214
  (**V bound at buf(4) ~1220**).
- Decode global (`record_layer`): append ~590 (global K → q8 ~591, **V →
  `kv_append_global` ~601 f16**); `attn_gl_q8` dispatch ~634 (**V bound at
  buf(3) ~639**).
- `p.q_i8_gl`/`p.q_scales_gl` are the per-chunk Q-quant scratch (mirror
  `xn_i8`/`xn_scales`); reuse the same pattern if B needs V-side scratch.

**So every place V is touched today is plain f16** — that's exactly the set
Piece B must change (KvStore.v, both global appends, the prefill PV read, the
decode V read), plus a new reference + parity.

## 1. The Q8_0 SoA format (one definition, used by K and Q; template for V)

Blocks of **32** elements along the contiguous array. Per block: one f16 scale
`d = amax·(1/127)` and 32 i8 quants `round_ties_even(x/d)` packed 4-per-u32
(8 u32/block). For a `[rows × ROW_LEN]` array (ROW_LEN a multiple of 32):
- scale of (row r, block b) → `scales[r·(ROW_LEN/32) + b]` (f16)
- i8 quant of (row r, element e) → byte `e%32` of `quants[(r·(ROW_LEN/32) + e/32)·8 + (e%32)/4]`
  — i.e. as a flat `array<i8>`, element index is just `r·ROW_LEN + e`.

`d` uses a reciprocal multiply (GPU FDiv is 2.5 ULP), the scale stores through
`f16()` (RTNE; `pack2x16float` truncates on RADV), `round` is round-half-even.
The GPU is the authoritative producer; ±1 quant at exact boundaries is fine.
CPU mirror: `q8_quant` in `crates/sg-gpu/tests/reference/mod.rs`.

## 2. Layout consistency (why K/Q agree for free — and why V is HARD)

K and Q share the §1 format over their shapes, so `kv_quant_q8` (Q),
`kv_append_global_q8` (K) and `_iq` (reader) agree by index algebra (verified;
HD_BLOCKS = HEAD_DIM/32 = 16, N_KV_HEADS = 4, N_Q_HEADS = 32). The crux: **K is
quantized along head-dim, which IS the QKᵀ contraction axis**, so each 32-block's
scale factors out of the i8 dot. That is the only reason int8 QKᵀ works.

**V does not have this property** (the entire Piece B problem). PV computes
`O[q][d] = Σ_key P[q][key]·V[key][d]`, contracting over **keys**. V is currently
stored quantized-able along **head-dim** (per-vector, compact) — which does NOT
align with the key contraction, so a per-block V scale sits *inside* the key-sum
and cannot be pulled out of an int8 dot. Hence f16 PV in Piece A.

## 3. Validation gates (same as Piece A used)

Run in cheap→expensive order:
- `cargo build --release --workspace --tests --benches` + `clippy -D warnings`.
- **Kernel parity** (fast, no model): `parity_attn` — add a V/PV parity case
  mirroring the `_iq` test, quantizing V in whatever layout B chooses; reference
  uses the dequanted V (`q8_quant`-style).
- **Prefill quality proxy** (~70 s): `prefill_parity` — per-layer nrmse (Q8-K
  alone landed 0.039; adding Q8-V should stay ≤ ~0.045, re-check the bound) +
  chunked-vs-oracle/decode.
- **Decode correctness**: `gpu_parity` (global-worst nrmse ≤ 0.045; logits the
  real gate).
- **Perplexity** (minutes): wikitext < 0.5 %, code within the calibrated band.
- **e2e win**: `sg-bench profile`, compare prefill @ q0 8K/32K.

**Test-running + operational gotchas (recorded the hard way):**
- `SG_MODEL_GGUF` must be an **absolute** path (nextest runs from the crate dir;
  the model is `models/gemma-4-31B_q4_0-it.gguf`).
- Model-heavy GPU tests serialize via the `gpu-model` nextest group
  (`.config/nextest.toml`); ~17.5 GB each. Filter by `binary(...)`.
- **Pin `perf=high` for benches; run sustained correctness gates (perplexity) at
  `perf=auto`.** The first Piece-A perplexity run hard-power-cut the box — a
  thermal trip under sustained perf=high load with a warm room/intake (traceless
  in the journal; NOT a code/GPU fault — the GPU watchdog recovers gracefully).
  Re-ran green at auto. Check ambient/fans before any long sustained run.
- coopLoad-only buffers are invisible to vulkano auto-sync → `touch` every such
  producer before the consuming dispatch (touch.wgsl). Any new int8-V kernel
  reading V via coopLoad needs the same treatment.

---

## Piece B — int8 V/PV (the clean-context task)

**Goal:** halve V DRAM traffic too (V is currently the other half of the
long-context global KV stream). Expected ~another similar long-context
increment on top of Piece A's +9 %/+23 %.

**The constraint (restate §2):** for an int8 PV the quantization blocks must
align with the **key** contraction so the per-block scale factors out of the i32
dot. V's natural (compact, per-vector) quant is along head-dim — wrong axis.

**Design directions (open — this is the research part):**
1. **Quantize V along keys** (per 32-*key* block, per head-dim) — aligns with
   the PV contraction. Awkward storage (scale per (key-block, head-dim)) and the
   append quantizes *across* keys, not within a vector, so `kv_append_global_q8`
   does NOT transfer — V needs its own append/quant kernel and its own §2-style
   layout-consistency proof against the PV reader. Probably the cleanest int8 PV.
2. **Fold a per-key V scale into P** (`P' = P·v_scale[key]`, then `P'(f16) ×
   V_q(i8)`) — a mixed-type matmul, not pure int8; check whether the fork's
   coopmat supports f16×i8, else convert.
3. **Transpose the PV problem.** Any of these needs a fresh design + parity +
   an `attn_flash_cmp` A/B.

**Concrete hook points (all currently f16, from the as-landed map above):**
- `KvStore.v` (graph.rs ~117) + its alloc (~378/385) — add `v_quants`/`v_scales`
  (mirror the K fields) or a new layout; keep sliding V f16.
- Global V append: graph.rs ~601 (decode) and ~1154 (prefill) — currently
  `kv_append_global`. Replace with a V-specific quantize-and-append in B's
  chosen layout.
- Prefill PV: `attn_prefill_global_flash_sp_iq.wgsl` ~156–165 (the `coopLoadT<…
  f16, B>(&v…)` + `coopMultiplyAdd(ap, bv, …)`). This is where int8 PV lands;
  `ap` (the P fragment) is f16 today.
- Decode V: `attn_decode_global_q8k.wgsl` (binding 3 `v`, the `vv[d] =
  f32(v[…])` read) + dispatch graph.rs ~639. Decode PV is scalar (GEMV-like) so
  it's a straightforward dequant-on-read, like the K side already is.
- Reference + parity: `q8_quant` (head-dim) is the template, but B's V layout
  (likely per-key) needs its own quantizer in `tests/reference/mod.rs` and a new
  parity case.

**Sequencing note:** Piece A deliberately wired the graph for Q8 K only and left
V f16 (banked the validated win without blocking on the hard kernel). B touches
the global KV path a second time for V — that's expected and additive; the K
plumbing (KvStore Option split, touch barriers, append/quant pattern) is the
reusable template.
