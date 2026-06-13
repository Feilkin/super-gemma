# Patch guide — component-wise multiply + scalar conversion on cooperative matrices (naga fork)

**Task for this session:** extend the existing `Feilkin/wgpu` naga fork with two
more cooperative-matrix operations, verify them on the box, and hand back
verified diffs for the fork to be updated. **Scope is the naga patch + a smoke
test only** — the kernel that uses these ops (the int8-MMQ in-register rescale)
is deliberately a separate, later task.

## Why (context — you won't have it otherwise)

super-gemma's goal is **the fastest Gemma 4 31B Q4_0 inference server for this
one Framework Desktop** (see `AGENTS.md`). Hardware-specific kernel optimization
is core work, and we're free to extend the toolchain (the parity/quality tests
are the safety net).

The int8-MMQ GEMM (`gemm_q4_0_i8.wgsl`, profile rank #2) is bottlenecked by its
**per-32-block rescale**: each block's i32 dot must be pulled out of the coopmat
accumulator, converted to f32, scaled by `d_a[m]·d_w[n]`, and summed into Y.
Because KHR coopmat1 has no per-element fragment access, today that's done by
`coopStore`-ing the i32 tile to LDS + a scalar loop + **two workgroup barriers
per block** (measured as the bottleneck — an occupancy sweep proved it's the
barriers, not occupancy). The fix is to do the rescale **in registers**:

```
DOT_i = wmma(aq, wq)          # coop_mat<i32,C>
DOT_f = f32(DOT_i)           # OpConvertSToF  ← NEEDS PATCH (conversion on coopmat)
Y_f   = Y_f + scale * DOT_f   # OpFMul (component-wise) ← NEEDS PATCH ; OpFAdd already works
```

`OpFAdd`/`OpFSub`/`OpMatrixTimesScalar` on coopmats are **already wired** in
naga; we only need **component-wise `OpFMul`** and **`f32(coopmat<i32>)`
conversion**. Both are supported by SPV_KHR_cooperative_matrix and the RDNA3.5
hardware — they're just not exposed by naga yet. (This is the same kind of work
as the i8 scalar addition — see the sibling guide `docs/naga-int8-coopmat-patch.md`
for fork mechanics, `[patch.crates-io]` setup, the `build.rs` SPIR-V capability
whitelist, and `MESA_SHADER_CACHE_DISABLE=1 RADV_DEBUG=asm/shaderstats` for
inspecting output.)

## Fork state

- Fork: `github.com/Feilkin/wgpu`, current rev **`a10bf033`** — already contains
  the i8/u8 scalar + signed-int8 coopmat MulAdd changes. **Branch from there**
  (it's naga `29.0.3`; keep the version `29.0.3` so the `[patch.crates-io]` in
  super-gemma's workspace `Cargo.toml` stays SemVer-compatible).
- super-gemma `Cargo.toml` already has:
  `[patch.crates-io] naga = { git = "https://github.com/Feilkin/wgpu", rev = "a10bf033..." }`.
- **Hosting/ownership:** Ada owns the fork push. Develop against a *local
  writable* clone and `[patch.crates-io] naga = { path = "…" }` for iteration;
  when verified, hand Ada the diffs to commit + push, then bump the rev.

## The patch — two capabilities, ~6 edits (paths are naga 29.0.3; grep to confirm lines)

### A. Component-wise `OpFMul` (coop_mat × coop_mat) — small, mirrors the existing Add

The frontend already produces a `Binary{Multiply}` for `a * b`; only the
validator and backend reject coopmat×coopmat.

**A1. `src/valid/expression.rs`** — the `Bo::Multiply` arm (~970, right after the
`// Scalar * coop matrix.` case). Add a coopmat×coopmat case mirroring the Add
arm (which already does `Ti::CooperativeMatrix { .. } => left_inner == right_inner`):
```rust
// component-wise coop matrix * coop matrix
(&Ti::CooperativeMatrix { scalar: s1, .. }, &Ti::CooperativeMatrix { scalar: s2, .. })
    => s1 == s2 && left_inner == right_inner,
```

**A2. `src/back/spv/block.rs`** — the Multiply dimension match (~1166). Today:
```rust
(Dimension::CooperativeMatrix, Dimension::CooperativeMatrix)
//Note: technically can do `FMul` but IR doesn't have matrix per-component multiplication
| (Dimension::CooperativeMatrix, _)
| (_, Dimension::CooperativeMatrix) => {
    unimplemented!()
}
```
Replace the `(CooperativeMatrix, CooperativeMatrix)` case with the component-wise
op chosen by scalar kind (like the Vector/Scalar cases just above):
```rust
(Dimension::CooperativeMatrix, Dimension::CooperativeMatrix) => {
    match left_ty_inner.scalar_kind() {
        Some(crate::ScalarKind::Float) => spirv::Op::FMul,
        Some(crate::ScalarKind::Sint | crate::ScalarKind::Uint) => spirv::Op::IMul,
        _ => return Err(Error::Validation("unsupported coop-matrix component multiply")),
    }
}
```
Keep the remaining `(CooperativeMatrix, _) | (_, CooperativeMatrix) => unimplemented!()`
(those are matrix×scalar mixes already handled by the MatrixTimesScalar arm above;
verify the ordering so the scalar cases win first).

### B. `f32(coop_mat<i32>)` conversion → `OpConvertSToF` — touches frontend, validator, typifier, backend

This is the standard WGSL conversion-constructor path (`f32(x)`, like
`f32(vec3<i32>)`), extended to accept a cooperative matrix and convert it
component-wise.

**B1. `src/valid/expression.rs`** — the `E::As { expr, kind, convert }` arm
(~1241). It computes `base_scalar` from the operand, currently matching
`Scalar | Vector => scalar`, `Matrix => scalar`, else `InvalidCastArgument`.
Add:
```rust
crate::TypeInner::CooperativeMatrix { scalar, .. } => scalar,
```

**B2. `src/proc/typifier.rs`** — the `Expression::As` arm (~732). It rebuilds the
result type from the operand (Scalar→Scalar, Vector→Vector, Matrix→Matrix) with
the new `kind`/`width`. Add a `CooperativeMatrix` arm preserving `columns/rows/
role` and replacing the scalar:
```rust
Ti::CooperativeMatrix { columns, rows, role, scalar: crate::Scalar { width, .. } } =>
    TypeResolution::Value(Ti::CooperativeMatrix {
        columns, rows, role,
        scalar: crate::Scalar { kind, width: convert.unwrap_or(width) },
    }),
```

**B3. `src/back/spv/block.rs`** — `write_as_expression` (~2198). It special-cases
`TypeInner::Matrix` by extracting/converting each column. A **cooperative matrix
converts component-wise with a single instruction** (no column extraction), so
add a `CooperativeMatrix` branch *before* the generic scalar/vector path that
emits the right conversion op on the whole matrix value:
- `Sint → Float`: `OpConvertSToF`
- `Uint → Float`: `OpConvertUToF`
- `Float → Sint`: `OpConvertFToS`; `Float → Float` (width change): `OpFConvert`;
  `Float → Uint`: `OpConvertFToU`; int width changes: `OpSConvert`/`OpUConvert`.
  (For the rescale we only strictly need **Sint(i32) → Float(f32) = `OpConvertSToF`**;
  implement that path first, others as time allows.)
Emit a single `Instruction::unary(op, result_type_id, id, expr_id)` with the
coopmat result type.

**B4. `src/front/wgsl/lower/construction.rs`** — `construct` (~107) is where
`f32(x)` becomes `Expression::As { … }` (see the emissions at ~194/216/259).
**Verify** that a single coop-matrix argument reaches one of those `As` paths;
if the constructor dispatch rejects coopmat operands earlier, add a coop-matrix
case that emits `Expression::As { expr, kind, convert: Some(width) }`. This is
the most likely spot to need exploratory work — trace `f32(some_coopmat)` and
make it lower to `As`.

### Capabilities / build side
- super-gemma's `crates/sg-gpu/build.rs` already whitelists `Capability::Int8`,
  `CooperativeMatrixKHR`, etc. `OpConvertSToF`/`OpFMul` on coopmats need no new
  SPIR-V capability beyond what's already enabled. `GpuContext` already enables
  `cooperative_matrix` + `vulkan_memory_model` + `storage_buffer8_bit_access`.

## Verification (do this before handing back)

1. **Compile check** — a smoke shader (`raw: true`, like `coop_i8_smoke.wgsl`)
   exercising both new ops:
   ```wgsl
   enable f16;
   enable wgpu_cooperative_matrix;
   @group(0) @binding(0) var<storage, read> di: array<i32>;   // 16×16 i32
   @group(0) @binding(1) var<storage, read> sc: array<f32>;   // 16×16 f32 scale
   @group(0) @binding(2) var<storage, read_write> out: array<f32>;
   @compute @workgroup_size(64)
   fn main() {
       let d_i = coopLoadT<coop_mat16x16<i32, C>>(&di[0], 16u);
       let s   = coopLoadT<coop_mat16x16<f32, C>>(&sc[0], 16u);
       let d_f = f32(d_i);          // OpConvertSToF on a coopmat   [B]
       coopStoreT(s * d_f, &out[0], 16u);   // component-wise OpFMul [A]
   }
   ```
   (Verify `coopLoad` into a **C-use** matrix is accepted — the real kernel needs
   the scale loaded as C-use to multiply the C-use converted dot. If coopLoad
   rejects role=C, note it; it may need a small allowance too.)
2. **SPIR-V check** — build sg-gpu (the shader compiles at build time;
   `MESA_SHADER_CACHE_DISABLE=1 RADV_DEBUG=asm` to dump). Confirm the disassembly
   contains `OpConvertSToF` and `OpFMul` operating on `OpTypeCooperativeMatrixKHR`
   values, and that it validates (no naga panic).
3. **GPU parity** — dispatch on known inputs, check `out[m][n] == sc[m][n] *
   f32(di[m][n])` exactly (integers → f32 is exact here; this catches layout/role
   bugs). Mirror the structure of `crates/sg-gpu/tests/coop_i8_smoke.rs`.
4. **Regression** — `cargo nextest run -p sg-gpu -E 'binary(parity_attn) or
   binary(parity_gemm) or binary(parity_mmq)'` must stay green (the patch is
   additive; existing coopmat kernels must be unaffected).

## Hand-back
- Provide the diffs (per-file) for Ada to apply to `Feilkin/wgpu` and push; note
  the new rev so super-gemma's `[patch.crates-io]` can be bumped.
- Leave the smoke shader + test in super-gemma (`coop_arith_smoke.wgsl` +
  test) as the regression anchor, like `coop_i8_smoke`.
- **Out of scope (next session):** rewriting `gemm_q4_0_i8.wgsl` to use the
  in-register rescale, and re-benchmarking against the 4.4 (int8) / 9.5 (f16)
  TFLOPS baseline at **pinned-high clocks** (`power_dpm_force_performance_level
  = high` — see the perf-level finding; `auto` idles the fabric clock and costs
  ~30%).

## References
- `docs/naga-int8-coopmat-patch.md` — sibling guide; fork mechanics, branch
  point (wgpu commit `4cbe623`, `path_in_vcs: naga`), `[patch]` setup, build.rs
  capability whitelist, cache-busting for shaderstats/asm dumps.
- naga 29.0.3 source (for line references):
  `~/.cargo/registry/src/index.crates.io-*/naga-29.0.3/`.
- SPV_KHR_cooperative_matrix: cooperative matrices may be operands to
  component-wise arithmetic (`OpFMul`, `OpFAdd`, …), `OpMatrixTimesScalar`, and
  conversions (`OpConvertSToF`, …) — all already validated by the hardware
  (`docs/probe/vulkan.json`).
- super-gemma STATUS / the int8-MMQ work: `gemm_q4_0_i8.wgsl`, `parity_mmq.rs`,
  `benches/mmq_tflops.rs`.
