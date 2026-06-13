# Patch guide — int8 cooperative matrices in a naga 29.0.3 fork

> **STATUS: APPLIED & VERIFIED (2026-06-13).** Fork
> `github.com/Feilkin/wgpu` rev `a10bf033`, wired via `[patch.crates-io]` in
> the workspace `Cargo.toml`. The `coop_i8_smoke` kernel/test computes a signed
> `i8×i8→i32` 16×16×16 MMA bit-exact vs CPU on the box (RADV accepts the SPIR-V:
> `OpTypeInt 8 1` operands, `OpCooperativeMatrixMulAddKHR` with all four
> `*SignedComponentsKHR` flags). f16 coopmat gemm/attn parity unaffected.
> **Correction to the original "open item": 8-bit storage is REQUIRED, not
> optional** — `coopLoad` takes its component scalar from the pointer's base
> type, so int8 operands must live in `array<i8>` buffers (`GpuContext` now
> enables `storage_buffer8_bit_access`). Remaining work: the MMQ GEMM kernel
> (last section).

**Goal.** Unblock the int8-MMQ GEMM (profile rank #2): the box exposes
`M16xN16xK16 A=SINT8 B=SINT8 C=SINT32` coopmat (and UINT8 variants —
`docs/probe/vulkan.json`), but **naga 29's WGSL front-end has no `i8`/`u8`
scalar**, so an 8-bit cooperative matrix is unspellable. This patch adds the
minimum needed to write `coop_mat16x16<i8, A>` / `<i8, B>` with an
`<i32, C>` accumulator and have it lower to correct signed-int SPIR-V.

Scope is deliberately tiny: int8 is needed **only as a coopmat component
type**, not for general WGSL scalar arithmetic — so no vectors, literals,
math overloads, or non-SPIR-V backends are touched (cf. wgpu PR #9412, which
did the *full* i16 treatment; we need far less). The SPIR-V backend already
emits 8-bit ints and the `Int8` capability; the typifier already resolves
`coopMultiplyAdd` to the C operand's type; the coop-op validator already
checks operand *roles* only (not that A/B/C scalars match). So mixed-precision
int8→int32 MMA is already IR-legal — we only open the front-end + two
validator gates + one backend signedness detail.

## Fork branch point

naga 29.0.3 was published from **gfx-rs/wgpu commit
`4cbe6232b2d7c289b6e1a38416a6ae1461a22e81`**, subdir `naga/`
(`.cargo_vcs_info.json` of the crates.io source). Branch the wgpu fork from
that commit and edit `naga/` so the crate version stays `29.0.3` — this keeps
the `[patch.crates-io]` SemVer-compatible with `naga_oil 0.22` (`^29`) and our
`build.rs` (`29.0.3`). Do **not** patch from wgpu `main` (PR #9412 landed
there → newer naga → version mismatch breaks the patch).

`spirv` crate available to naga is `0.4.0+sdk-1.4.341.0`; it has
`spirv::CooperativeMatrixOperands` (used in change 4).

---

## The patch — 4 changes, all in `naga/src/`

### 1. `proc/type_methods.rs` — add `Scalar::I8` / `U8` consts

In `impl crate::Scalar` (next to the existing `I32`/`U32`/`F16`… consts,
~line 26):

```rust
    pub const I8: Self = Self {
        kind: crate::ScalarKind::Sint,
        width: 1,
    };
    pub const U8: Self = Self {
        kind: crate::ScalarKind::Uint,
        width: 1,
    };
```

### 2. `front/wgsl/parse/conv.rs` — parse the `i8` / `u8` keywords

In `map_predeclared` (the `match word` block, ~line 451; aliases `Sc = Scalar`,
`Ti = TypeInner` are already `use`d at ~445). After the `"f16"` line add:

```rust
        "i8" => Ti::Scalar(Sc::I8).into(),
        "u8" => Ti::Scalar(Sc::U8).into(),
```

> Minimal/ungated: makes `i8`/`u8` always-available type keywords in this
> fork. Cleaner (optional, mirrors PR #9412's `enable wgpu_int16;`): gate them
> behind a new `enable wgpu_int8;` extension. Not required for us — the int8
> kernel compiles via the plain-naga path (`raw: true`), and we control the
> shader source.

### 3. `valid/type.rs` — allow integer coopmats + width-1 scalars

**3a. Cooperative-matrix element gate** (~line 437). Today it allows only
`Float` width 2/4. Replace:

```rust
                // Allow f16 (width 2) and f32 (width 4) for cooperative matrices
                if scalar.kind != crate::ScalarKind::Float
                    || (scalar.width != 2 && scalar.width != 4)
                {
                    return Err(TypeError::MatrixElementNotFloat);
                }
```

with:

```rust
                // f16/f32 floats; i8/u8 operands and i32/u32 accumulators (the
                // box's coopmat configs: SINT8/UINT8 A/B → SINT32/UINT32 C).
                let ok = match scalar.kind {
                    crate::ScalarKind::Float => scalar.width == 2 || scalar.width == 4,
                    crate::ScalarKind::Sint | crate::ScalarKind::Uint => {
                        scalar.width == 1 || scalar.width == 4
                    }
                    _ => false,
                };
                if !ok {
                    return Err(TypeError::MatrixElementNotFloat); // (name now stale; fine)
                }
```

**3b. `check_width`** (~line 318/331) — the intermediate `Scalar(I8/U8)` type
that the coopmat generic creates is validated standalone, so width 1 must be
accepted. In **both** the `Sint` and `Uint` arms change the fallback
`scalar.width == 4` to:

```rust
                    scalar.width == 4 || scalar.width == 1
```

> Minimal: ungated. Cleaner: add a `Capabilities::SHADER_INT8` flag and gate
> width-1 behind it (naga 29 has no such flag yet; PR #9412 added
> `SHADER_INT16` — follow that shape if you want parity). Our `build.rs`
> validates with `Capabilities::all()`, so ungated is safe for us.

### 4. `back/spv` — emit signedness operands for integer MulAdd

KHR `OpCooperativeMatrixMulAddKHR` defaults to *unsigned* components; signed
int8 needs the **Cooperative Matrix Operands** word. f16 must omit it.

**4a. `back/spv/instructions.rs`** — `coop_mul_add` (~line 1296): add an
optional operands word.

```rust
    pub(super) fn coop_mul_add(
        result_type_id: Word,
        id: Word,
        a: Word,
        b: Word,
        c: Word,
        operands: Option<spirv::CooperativeMatrixOperands>,
    ) -> Self {
        let mut instruction = Self::new(Op::CooperativeMatrixMulAddKHR);
        instruction.set_type(result_type_id);
        instruction.set_result(id);
        instruction.add_operand(a);
        instruction.add_operand(b);
        instruction.add_operand(c);
        if let Some(ops) = operands {
            instruction.add_operand(ops.bits());
        }
        instruction
    }
```

**4b. `back/spv/block.rs`** — the `CooperativeMultiplyAdd` arm (~line 2172).
Resolve C's scalar kind and pass signed flags when it's a signed int:

```rust
            crate::Expression::CooperativeMultiplyAdd { a, b, c } => {
                self.writer.require_any(
                    "CooperativeMatrix",
                    &[spirv::Capability::CooperativeMatrixKHR],
                )?;
                // Integer MulAdd must declare component signedness; float omits it.
                let operands = match *self.fun_info[c].ty.inner_with(&self.ir_module.types) {
                    crate::TypeInner::CooperativeMatrix { scalar, .. }
                        if scalar.kind == crate::ScalarKind::Sint =>
                    {
                        Some(
                            spirv::CooperativeMatrixOperands::MATRIX_A_SIGNED_COMPONENTS_KHR
                                | spirv::CooperativeMatrixOperands::MATRIX_B_SIGNED_COMPONENTS_KHR
                                | spirv::CooperativeMatrixOperands::MATRIX_C_SIGNED_COMPONENTS_KHR
                                | spirv::CooperativeMatrixOperands::MATRIX_RESULT_SIGNED_COMPONENTS_KHR,
                        )
                    }
                    _ => None,
                };
                let a_id = self.cached[a];
                let b_id = self.cached[b];
                let c_id = self.cached[c];
                let id = self.gen_id();
                block.body.push(Instruction::coop_mul_add(
                    result_type_id, id, a_id, b_id, c_id, operands,
                ));
                id
            }
```

> This sets ALL-signed when C is `Sint` (our case: A=SINT8, B=SINT8,
> C=SINT32 — matches the probed config). It does not handle mixed A/B
> signedness (e.g. SINT8×UINT8); we don't need it (Q4_0 quants and Q8
> activations are both signed). If a UINT8 path is wanted later, derive each
> flag from the respective operand's scalar kind.

---

## Wire into super-gemma

Workspace root `Cargo.toml`, add:

```toml
[patch.crates-io]
# int8 coopmat support (docs/naga-int8-coopmat-patch.md). naga_oil 0.22
# inherits this transitively; the int8 kernel itself uses the raw-naga path.
naga = { git = "https://github.com/<ada>/wgpu", rev = "<fork-commit>" }
```

For local iteration before the fork is pushed, point at a clone instead:
`naga = { path = "/home/ada/forks/wgpu/naga" }` (or a copy of the crates.io
29.0.3 source). `build.rs` already lists `Capability::Int8` (line ~724) and
validates with `Capabilities::all()`, so nothing else changes there.

`cargo build -p sg-gpu` must still succeed (the patch is additive; the f16
coopmat kernels are unaffected — verify the existing `parity_attn` /
`parity_matmul` tests stay green).

---

## Verification milestone (do this BEFORE the MMQ kernel)

Prove the toolchain end-to-end with a throwaway kernel, no GPU needed for the
compile step (a shader/validation error is a build error):

1. Add `shaders/coop_i8_smoke.wgsl`:

```wgsl
enable f16;
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<i32>;
@compute @workgroup_size(64)
fn main() {
    let ma = coopLoadT<coop_mat16x16<i8, A>>(&a[0], 16u);
    let mb = coopLoad<coop_mat16x16<i8, B>>(&b[0], 16u);
    var acc: coop_mat16x16<i32, C>;
    acc = coopMultiplyAdd(ma, mb, acc);
    coopStoreT(acc, &out[0], 16u);
}
```

2. Register it in `build.rs` (`raw: true`, bindings 3). `cargo build -p sg-gpu`
   should emit `coop_i8_smoke.spv`. Confirm with `spirv-dis` that it contains
   `OpTypeInt 8 1`, `OpCooperativeMatrixMulAddKHR` with the trailing operands
   word (= `0xF`), and `OpCapability Int8` + `OpCapability CooperativeMatrixKHR`.
3. GPU smoke: dispatch it on small known int8 inputs and check the int32 dot
   against a CPU reference (one 16×16×16 tile). This confirms the driver
   accepts the config and signedness.

**Open items to confirm at this step:**
- **8-bit storage access.** The probe didn't query `storageBuffer8BitAccess`
  (only what f16 needed). The smoke kernel loads int8 from `array<u32>` via
  `coopLoad` (naga's coopLoad validator requires only *a* scalar on the pointer
  base, not a matching one), which should avoid needing 8-bit storage. If the
  driver rejects it, either add the 8-bit-storage capability/feature in
  `GpuContext` + `build.rs`, or keep packing 4×i8 per u32.
- **Saturating vs non-saturating.** Both are exposed. Use non-saturating
  (`C=SINT32` accumulator won't overflow for our K) unless a parity issue
  shows; saturating = add `SATURATING_ACCUMULATION_KHR` to the operands word.

---

## Then: the int8-MMQ GEMM kernel (separate work)

Once the toolchain is proven, write `gemm_q4_0_i8.wgsl` (one variant per
prefill matmul shape, like the f16 `gemm_q4_0`). MMQ structure:

- **Activations → Q8** per 32-block: scale `d_a = amax/127`, int8 quants
  (reuse the `kv_quant_q8` semantics / CPU ref; this is the new accuracy risk).
- **Q4_0 weights:** quants are `q − 8 ∈ [-8,7]` (signed int8); block scale `d_w`.
- **Per-block int8 MMA:** coopmat contracts 16 at a time, but the
  `(d_w·d_a)` scale is per-32-block, so accumulate `i32` over each block, then
  convert to f32 with `d_w·d_a` and add into an f32 tile — you cannot run a
  single i32 accumulation across all K (scales differ per block). This is the
  core MMQ trick and the main kernel-design question (block-granular i32→f32).
- **Determinism + quality gate:** fixed accumulation order; gate on the
  perplexity harness (activation-Q8 introduces bias — STATUS records the f16
  path already tracks the oracle within 0.33%/3%; int8 needs its own check
  before becoming the default, behind a build flag).

References: profile rank #2 and the 2026-06-13 prefill investigation in
`docs/STATUS.md`; `docs/plans/02-gpu-runtime-and-kernels.md` (gemm contract);
memory `int8-coopmat-not-expressible-in-wgsl` (now resolved by this patch).
