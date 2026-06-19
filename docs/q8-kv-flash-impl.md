# Q8 global KV cache — implementation plan (Piece A) + B starting point

Goal: bank the validated int8-QKᵀ Q8-K flash win (−25–30% long-context prefill,
commit `e99307a`) end-to-end. The kernels are committed and parity-green; this is
the graph plumbing + quality gate. **Read this whole doc before editing — the
layout-consistency in §2 is the one non-obvious trap.**

Status of the pieces (all committed, bench/parity-green, NOT wired into the graph):
- `attn_prefill_global_flash_sp_iq.wgsl` — int8 QKᵀ (Q8 K + pre-quantized i8 Q,
  per-block rescale), f16 PV. 7 bindings: `q_i8, q_scales, k_quants, k_scales, v,
  out, step`.
- `kv_append_global_q8.wgsl` — f16→Q8 quantize-and-append for the global K cache
  (linear slot = pos+token). 4 bindings: `src(f16), scales(f16,rw), quants(u32,rw),
  step`.
- `kv_quant_q8` / `kv_dequant_q8` — generic Q8_0 codecs (reuse `kv_quant_q8` for the
  Q-quant; `kv_dequant_q8`'s dequant math is the template for the decode kernel).
- `attn_prefill_global_flash_sp_q8.wgsl` — the f16-convert DEAD END (7× slower),
  kept only as the labeled `attn_flash_cmp` baseline. Do not wire it.
- Usage references: the `attn_flash_cmp` bench (`benches/attn_us.rs`) and the
  parity test `attn_prefill_global_flash_sp_iq_matches_reference`
  (`tests/parity_attn.rs`) show buffer setup + the dequanted-value reference; the
  CPU `q8_quant` helper is in `tests/reference/mod.rs`.

Scope: **global layers only** (global KV grows with context = the bottleneck;
sliding is window-capped at 1024 → stays f16). **K → Q8, V → f16** (V's int8 is
blocked — see §6/B).

---

## 1. The Q8_0 SoA format (one definition, used everywhere)

Blocks of **32** elements along the contiguous array. Per block: one f16 scale
`d = amax·(1/127)` and 32 i8 quants `round_ties_even(x/d)` packed 4-per-u32
(8 u32/block). For a `[rows × ROW_LEN]` array (ROW_LEN a multiple of 32):
- scale of (row r, block b) → `scales[r·(ROW_LEN/32) + b]` (f16)
- i8 quant of (row r, element e) → byte `e%32` of `quants[(r·(ROW_LEN/32) + e/32)·8 + (e%32)/4]`
  — i.e. as a flat `array<i8>`, element index is just `r·ROW_LEN + e`.

`d` uses a reciprocal multiply (GPU FDiv is 2.5 ULP), the scale stores through
`f16()` (RTNE; `pack2x16float` truncates on RADV), `round` is round-half-even.
The GPU is the authoritative producer; ±1 quant at exact boundaries is fine.

## 2. Layout consistency — VERIFIED, do not deviate

The whole point: `kv_quant_q8` (Q-quant), `kv_append_global_q8` (K append) and
`_iq` (reader) all agree **for free** because they share the §1 format over these
shapes. Proven by index algebra (HD_BLOCKS = HEAD_DIM/32 = 16, N_KV_HEADS = 4,
N_Q_HEADS = 32):

- **Global K cache:** quants `[L × N_KV_HEADS × HEAD_DIM]` i8, scales
  `[L × N_KV_HEADS × HD_BLOCKS]` f16. `kv_append_global_q8` (ROW_LEN=2048) writes
  key `slot`'s block `(kvh, blk)` to `scales[slot·64 + kvh·16 + blk]` and i8 element
  `(kvh,hd)` to flat index `slot·2048 + kvh·512 + hd`. `_iq` reads exactly those
  (`k_scales[(key·N_KV_HEADS + kvh)·HD_BLOCKS + blk]`, `k_quants[(key·N_KV_HEADS +
  kvh)·HEAD_DIM + hd]`). ✓
- **Q:** run `kv_quant_q8` on the roped global `q_gl` `[m × N_Q_HEADS × HEAD_DIM]`
  f16 → `q_i8` `[m × N_Q_HEADS × HEAD_DIM]` i8 + `q_scales`
  `[m × N_Q_HEADS × HD_BLOCKS]` f16. `_iq` reads
  `q_scales[(query·N_Q_HEADS + qh)·HD_BLOCKS + blk]` and `q_i8[(query·N_Q_HEADS +
  qh)·HEAD_DIM + hd]` — exactly the flat 32-block layout `kv_quant_q8` produces. ✓

So **do not invent a new layout** — quantize Q with the existing `kv_quant_q8`,
append K with `kv_append_global_q8`, and the reader just works. The `_iq` parity
test already exercises this format end-to-end on the CPU side (`q8_quant`).

## 3. `KvStore` → Q8 K (graph.rs:108 + alloc ~340)

Make the global K buffers Q8; keep V f16 and the sliding path f16 (don't allocate
a full f16 global K — that's the memory the Q8 saves). Option fields keep the
`sliding` branch unchanged:

```rust
struct KvStore {
    k: Option<Subbuffer<[u16]>>,        // sliding: f16 K
    k_quants: Option<Subbuffer<[u32]>>, // global: Q8 K quants  (kv_dim_gl·slots/4 u32)
    k_scales: Option<Subbuffer<[u16]>>, // global: Q8 K scales  (kv_dim_gl·slots/32 f16)
    v: Subbuffer<[u16]>,                // f16 V (both)
}
```
Alloc per kind: Sliding → `k: Some(f16buf(sliding_window·kv_dim_sl))`, quants/scales
None. Global → `k: None`, `k_quants: u32buf(global_cap·kv_dim_gl/4)`,
`k_scales: f16buf(global_cap·kv_dim_gl/32)`, `v: f16buf(global_cap·kv_dim_gl)`
(kv_dim_gl = 2048). Saves ~67 MB/global-layer.

## 4. Prefill global path (record_prefill_layer, the `prefill_gl` dispatch ~1107)

Currently: `touch(q); touch(kv.k); touch(kv.v); dispatch(prefill_gl = flash_sp, {q,
kv.k, kv.v, attn_gl, step})`. Change the **global** branch (sliding untouched) to:

1. **Append**: the `do_appends` loop is symmetric f16 today. Split it for global:
   K via `kv_append_global_q8` (src = roped `k`, → `kv.k_quants`/`kv.k_scales`,
   grid = `(n_real·kv_dim_gl/32)` blocks); V via the existing `kv_append_global`
   (→ `kv.v`). Sliding stays the symmetric f16 loop.
2. **Q-quant**: after rope, dispatch `kv_quant_q8` on `q_gl` (n_real·q_dim_gl
   elements; q_dim_gl = 16384) → new scratch `p.q_i8_gl` (u32, m·q_dim_gl/4) +
   `p.q_scales_gl` (f16, m·q_dim_gl/32). Add those two fields to the prefill `Bufs`
   (mirror `xn_i8`/`xn_scales`, graph.rs:330).
3. **Dispatch `_iq`** instead of flash_sp: load `prefill_gl_iq =
   "attn_prefill_global_flash_sp_iq"`. Bindings `{q_i8_gl, q_scales_gl, kv.k_quants,
   kv.k_scales, kv.v, attn_gl, step}`, grid `[N_Q_HEADS, m_pad/16, 1]`, push = scale.
4. **Touch barriers** (coopmat reads are invisible to auto-sync): touch
   `q_i8_gl`, `kv.k_quants`, `kv.k_scales`, `kv.v` before the dispatch (replaces the
   current 3 touches). `k_scales` is read by normal indexing in-kernel, not coopLoad
   — but touch it too unless you confirm reflection sees it.

## 5. Decode global path (record_layer, the `attn_gl` dispatch ~560) + new kernel

Decode reads the SAME Q8 K cache, so it needs Q8 K too. `attn_decode_global` is
GEMV-like (scalar `kk[d] = f32(k[...])` + `subgroupAdd`), so this is cheap — no
coopmat. Write `attn_decode_global_q8k.wgsl`: copy `attn_decode_global.wgsl`,
replace the K binding (f16) with `k_quants: array<i8>` + `k_scales: array<f16>`,
and change `kk[d] = f32(k[kv_base + d])` to dequant per element using
`kv_dequant_q8`'s math: block = `(t·N_KV_HEADS + kvh)·HD_BLOCKS + (d_within_head)/32`,
`kk[d] = f32(k_scales[block]) · f32(sign_extend_i8(k_quants[...]))`. Decode Q stays
f16 (no Q-quant on the decode side). The decode global K append also switches to
`kv_append_global_q8` (one token). Register the variant (5→6 bindings) and load it
as a new `attn_gl_q8` kernel; dispatch it for global decode layers.

## 6. Validation (in order; the expensive runs are last)

- `cargo build --release --workspace --tests --benches` + `cargo clippy ... -D warnings`.
- **Kernel parity** (fast, no model): `parity_attn` — the `_iq` test is already
  green; add a `attn_decode_global_q8k` parity test (mirror `attn_decode_global`,
  Q8-quantize K).
- **Prefill quality proxy** (≈70 s, model upload): `prefill_parity` — per-layer
  nrmse (expect ≤ ~0.05, Q8-K should be mild) and chunked-vs-oracle/decode. This is
  the fast read on Q8-K quality before the full perplexity run.
- **Decode correctness**: `gpu_parity` (decode path).
- **Perplexity gate** (minutes): the M4 methodology (STATUS) — wikitext < 0.5 %,
  code within the calibrated band.
- **e2e win**: `sg-bench profile` (perf=high), compare prefill @ q0 8K/32K.

**Test-running gotchas (recorded the hard way):**
- `SG_MODEL_GGUF` must be an **absolute** path — nextest runs tests from the crate
  dir, so a relative `models/...` resolves wrong and the test "skips/NotFound".
- Model-heavy GPU tests are serialized via the `gpu-model` nextest group
  (`.config/nextest.toml`); each uploads ~17.5 GB. Filter by test name, e.g.
  `-E 'test(/prefill_single_chunk|chunked_prefill/)'`.
- Pin `perf=high` before any bench (`power_dpm_force_performance_level`).

---

## B starting point — int8 V/PV (the bigger follow-up)

**The constraint (well-defined):** PV computes `O[q][d] = Σ_key P[q][key]·V[key][d]`,
contracting over **keys**. For an int8 PV the quantization blocks must align with
the contraction (keys) so the per-block scale factors out of the i32 dot — exactly
why int8 QKᵀ works (K is quantized along head-dim = the QKᵀ contraction). But V is
stored quantized along **head-dim** (per-vector, compact), which does NOT align with
the keys contraction, so the V scale sits inside the key-sum and cannot be pulled
out. That's why this kernel leaves V/PV in f16.

**Design directions (open — this is the research part):**
- Quantize V **along keys** (per 32-key block, per head-dim) — aligns with the PV
  contraction, but it's an awkward storage layout (scale per (key-block, head-dim))
  and the append would quantize across keys, not within a vector.
- Or fold a single per-key V scale into P (`P' = P·v_scale[key]`) and do `P'(f16) ×
  V_q(i8)` — but that's a mixed-type matmul, not int8.
- Or transpose the PV problem. All need a fresh design + parity + an `attn_flash_cmp`
  A/B. Halving V traffic too should be worth ~another similar increment.

Likely worth doing **before** wiring, so the graph is wired once for full Q8 KV —
but it's the harder kernel problem, hence a fresh-context task.
