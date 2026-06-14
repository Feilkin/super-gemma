//! Compiles every kernel variant from WGSL to SPIR-V at build time:
//! naga-oil composition (defines for per-shape specialization) -> naga
//! validation -> spv-out. The runtime never compiles shaders; a shader error
//! is a build error, catchable without a GPU.
//!
//! Output: one `<variant>.spv` per entry in `VARIANTS`, plus a generated
//! `kernels.rs` registry included by `src/kernel.rs`.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::{env, fs, path::PathBuf};

use naga_oil::compose::ShaderDefValue;

/// One compiled pipeline: a WGSL source plus its specialization defines.
struct Variant {
    /// Registry/pipeline name; also the `.spv` file stem.
    name: &'static str,
    /// Source file under `shaders/`, without extension.
    src: &'static str,
    /// naga-oil shader defs (`#ifdef` / `#{NAME}` substitution).
    defs: &'static [(&'static str, u32)],
    /// Workgroup size, exported to the runtime for dispatch math. Sources
    /// reference it as `#{WG_X}` etc. so this table is the single source of
    /// truth.
    workgroup: [u32; 3],
    /// Number of storage-buffer bindings (set 0, bindings 0..n); vulkano's
    /// reflection misses buffers consumed only by cooperative-matrix ops, so
    /// layouts are built from this instead.
    bindings: u32,
    /// Push-constant byte size (0 = none).
    push_bytes: u32,
    /// Required subgroup size, 0 = driver default (wave64 on this box). All
    /// kernels currently use 0: the coopmat GEMM measured NO wave64 penalty and
    /// forcing wave32 was −25% (STATUS "Tuning dead-ends"). The plumbing exists
    /// but is unused; pinning a size also needs `subgroup_size_control` enabled
    /// in `GpuContext`.
    subgroup_size: u32,
    /// Skip naga-oil and compile with plain naga (textual `#{NAME}`
    /// substitution only, no `#ifdef`/`#import`). Required for cooperative-
    /// matrix shaders: naga_oil 0.22's IR cloner copies
    /// `Expression::CooperativeLoad`'s inner pointer/stride handles without
    /// remapping them, corrupting the module.
    raw: bool,
}

/// Gemma 4 geometry (validated by `sg_gguf::ModelDesc` at load time): the
/// full set of shapes is known here, so every kernel is shape-specialized
/// (plan 02).
const VARIANTS: &[Variant] = &[
    Variant {
        name: "stub",
        src: "stub",
        defs: &[],
        workgroup: [64, 1, 1],
        bindings: 1,
        push_bytes: 4,
        subgroup_size: 0,
        raw: false,
    },
    // RMSNorm: hidden rows + the two QK-norm head_dims, each in both weight
    // conventions (W_PLUS_ONE picked by M3 parity).
    Variant {
        name: "rmsnorm_5376",
        src: "rmsnorm",
        defs: &[("ROW_LEN", 5376)],
        workgroup: [256, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "rmsnorm_5376_plus1",
        src: "rmsnorm",
        defs: &[("ROW_LEN", 5376), ("W_PLUS_ONE", 1)],
        workgroup: [256, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "rmsnorm_512",
        src: "rmsnorm",
        defs: &[("ROW_LEN", 512)],
        workgroup: [256, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "rmsnorm_512_plus1",
        src: "rmsnorm",
        defs: &[("ROW_LEN", 512), ("W_PLUS_ONE", 1)],
        workgroup: [256, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "rmsnorm_256",
        src: "rmsnorm",
        defs: &[("ROW_LEN", 256)],
        workgroup: [256, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "rmsnorm_256_plus1",
        src: "rmsnorm",
        defs: &[("ROW_LEN", 256), ("W_PLUS_ONE", 1)],
        workgroup: [256, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    // RoPE per attention site. Sliding: full rotation, θ=10k. Global:
    // partial rotation (0.25 × 512 = 128 dims), θ=1M.
    Variant {
        name: "rope_sliding_q",
        src: "rope",
        defs: &[("HEAD_DIM", 256), ("ROT_DIMS", 256), ("N_HEADS", 32)],
        workgroup: [256, 1, 1],
        bindings: 2,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "rope_sliding_k",
        src: "rope",
        defs: &[("HEAD_DIM", 256), ("ROT_DIMS", 256), ("N_HEADS", 16)],
        workgroup: [256, 1, 1],
        bindings: 2,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "rope_global_q",
        src: "rope",
        defs: &[("HEAD_DIM", 512), ("ROT_DIMS", 128), ("N_HEADS", 32)],
        workgroup: [256, 1, 1],
        bindings: 2,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "rope_global_k",
        src: "rope",
        defs: &[("HEAD_DIM", 512), ("ROT_DIMS", 128), ("N_HEADS", 4)],
        workgroup: [256, 1, 1],
        bindings: 2,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "geglu",
        src: "geglu",
        defs: &[],
        workgroup: [256, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    // Synchronization shim for recorded graphs (see touch.wgsl): a no-op
    // with a reflection-VISIBLE read_write on its binding, dispatched on a
    // buffer that a following coopmat kernel reads invisibly.
    Variant {
        name: "touch",
        src: "touch",
        defs: &[],
        workgroup: [1, 1, 1],
        bindings: 1,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    // Residual join: y = (a + b) * s; s carries layer_output_scale on the
    // FFN join, 1.0 on the attention join (M3 graph).
    Variant {
        name: "add_scaled",
        src: "add_scaled",
        defs: &[],
        workgroup: [256, 1, 1],
        bindings: 3,
        push_bytes: 4,
        subgroup_size: 0,
        raw: false,
    },
    // Q4_0 GEMV, one variant per matmul-site K (the N dimension is the
    // dispatch size). Workgroup = one wave (probe: subgroup 64).
    Variant {
        name: "gemv_q4_0_k5376",
        src: "gemv_q4_0",
        defs: &[("K_DIM", 5376)],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "gemv_q4_0_k8192",
        src: "gemv_q4_0",
        defs: &[("K_DIM", 8192)],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "gemv_q4_0_k16384",
        src: "gemv_q4_0",
        defs: &[("K_DIM", 16384)],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "gemv_q4_0_k21504",
        src: "gemv_q4_0",
        defs: &[("K_DIM", 21504)],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "gemv_q4_0_generic",
        src: "gemv_q4_0",
        defs: &[("GENERIC_K", 1)],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 4,
        subgroup_size: 0,
        raw: false,
    },
    // KV plumbing (plan 02 step 7): append into ring/linear stores, and
    // f16↔Q8_0 block-pair codecs for cache2 page traffic.
    Variant {
        name: "kv_append_sliding",
        src: "kv_append",
        defs: &[("ROW_LEN", 4096), ("RING", 1024)],
        workgroup: [256, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "kv_append_global",
        src: "kv_append",
        defs: &[("ROW_LEN", 2048)],
        workgroup: [256, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "kv_quant_q8",
        src: "kv_quant_q8",
        defs: &[],
        workgroup: [256, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "kv_dequant_q8",
        src: "kv_dequant_q8",
        defs: &[],
        workgroup: [256, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    // Q6_K LM head with fused softcap (plan 02 step 7): rows padded from
    // 4410 to 4416 bytes so each starts word-aligned.
    Variant {
        name: "gemv_q6_k_logits",
        src: "gemv_q6_k_logits",
        defs: &[("BLOCKS_PER_ROW", 21), ("ROW_WORDS", 1104)],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    // Attention (plan 02 step 6). Sliding: GQA 32:16, head_dim 256, window
    // 1024. Global: GQA 32:4, head_dim 512. K and V are separate stores on
    // BOTH layer kinds (the global "K = V aliased" M2 reading was wrong —
    // docs/reference/gemma4-forward-graph.md).
    // Workgroups cover one KV head (× query token / split), computing the
    // Q_PER_KV query heads that share it.
    // (The subgroupAdd kernels compile via plain naga, like coopmat:
    // naga_oil rejects `enable subgroups`.)
    // Both decode kernels are split-K + reduce: their natural workgroup
    // counts (16/4 KV heads) leave a 40-CU GPU latency-bound.
    Variant {
        name: "attn_decode_sliding",
        src: "attn_decode_sliding",
        defs: &[("HEAD_DIM", 256), ("N_KV_HEADS", 16), ("Q_PER_KV", 2)],
        workgroup: [64, 1, 1],
        bindings: 5,
        push_bytes: 8,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "attn_decode_global",
        src: "attn_decode_global",
        defs: &[("HEAD_DIM", 512), ("N_KV_HEADS", 4), ("Q_PER_KV", 8)],
        workgroup: [64, 1, 1],
        bindings: 5,
        push_bytes: 8,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "attn_reduce_d256",
        src: "attn_reduce",
        defs: &[("HEAD_DIM", 256)],
        workgroup: [64, 1, 1],
        bindings: 2,
        push_bytes: 4,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "attn_reduce_d512",
        src: "attn_reduce",
        defs: &[("HEAD_DIM", 512)],
        workgroup: [64, 1, 1],
        bindings: 2,
        push_bytes: 4,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "attn_prefill_sliding",
        src: "attn_prefill_sliding",
        defs: &[
            ("HEAD_DIM", 256),
            ("N_KV_HEADS", 16),
            ("Q_PER_KV", 2),
            ("WINDOW", 1024),
        ],
        workgroup: [64, 1, 1],
        bindings: 5,
        push_bytes: 4,
        subgroup_size: 0,
        raw: true,
    },
    // Two-range sliding prefill (plan 03 §prefill): history from the
    // pre-append ring + the chunk's own K/V, one position-ordered
    // streaming softmax. The production prefill path; the linear-view
    // `attn_prefill_sliding` above remains the A/B reference.
    Variant {
        name: "attn_prefill_sliding_ring",
        src: "attn_prefill_sliding_ring",
        defs: &[
            ("HEAD_DIM", 256),
            ("N_KV_HEADS", 16),
            ("Q_PER_KV", 2),
            ("WINDOW", 1024),
            ("RING", 1024),
        ],
        workgroup: [64, 1, 1],
        bindings: 7,
        push_bytes: 4,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "attn_prefill_global",
        src: "attn_prefill_global",
        defs: &[("HEAD_DIM", 512), ("N_KV_HEADS", 4), ("Q_PER_KV", 8)],
        workgroup: [64, 1, 1],
        bindings: 5,
        push_bytes: 4,
        subgroup_size: 0,
        raw: true,
    },
    // int8-MMQ GEMM (profile rank #2), one variant per (K, N) prefill site.
    // k512_n64 (1×1 tiles) pins parity; k5376_n21504 (2×4) is the bench shape.
    Variant {
        name: "gemm_q4_0_i8_k512_n64",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 512),
            ("N_DIM", 64),
            ("WG_X", 64),
            ("M_TILES", 1),
            ("N_TILES", 1),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Small 2×4-tile shape to parity-check the tiled path before benching.
    Variant {
        name: "gemm_q4_0_i8_k512_n128",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 512),
            ("N_DIM", 128),
            ("WG_X", 64),
            ("M_TILES", 2),
            ("N_TILES", 4),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_k5376_n21504",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 21504),
            ("WG_X", 64),
            ("M_TILES", 2),
            ("N_TILES", 4),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Occupancy sweep (rank #2): smaller tilings cut VGPR (~24/tile) and LDS
    // (2 KB/tile). int8 is occupancy-bound, not bandwidth-bound, so smaller
    // tiles should raise waves/SIMD with no bandwidth penalty. Parity variant
    // + bench variant per tiling.
    Variant {
        name: "gemm_q4_0_i8_t22_k512_n128",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 512),
            ("N_DIM", 128),
            ("WG_X", 64),
            ("M_TILES", 2),
            ("N_TILES", 2),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_t12_k512_n128",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 512),
            ("N_DIM", 128),
            ("WG_X", 64),
            ("M_TILES", 1),
            ("N_TILES", 2),
            ("STAGE_BUFS", 1), // single-buffer path parity coverage
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_t22_k5376_n21504",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 21504),
            ("WG_X", 64),
            ("M_TILES", 2),
            ("N_TILES", 2),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // FFN down-projection (K=21504→N=5376), 2×2 — the int8 prefill FFN slice.
    Variant {
        name: "gemm_q4_0_i8_t22_k21504_n5376",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 21504),
            ("N_DIM", 5376),
            ("WG_X", 64),
            ("M_TILES", 2),
            ("N_TILES", 2),
            ("STAGE_BUFS", 1), // large-K: single-buffer stage (occupancy; RGP)
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_t12_k5376_n21504",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 21504),
            ("WG_X", 64),
            ("M_TILES", 1),
            ("N_TILES", 2),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // 4×4 tiling — the f16 gemm's tiling, for an apples-to-apples compare. 16
    // accumulators is the most ILP to hide the coopmat-rescale chain, but yacc
    // (f32) + acc (i32) both live = high VGPR; watch for spill (shaderstats).
    Variant {
        name: "gemm_q4_0_i8_t44_k512_n256",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 512),
            ("N_DIM", 256),
            ("WG_X", 64),
            ("M_TILES", 4),
            ("N_TILES", 4),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_t44_k5376_n21504",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 21504),
            ("WG_X", 64),
            ("M_TILES", 4),
            ("N_TILES", 4),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Diagnostic: int8 MMA throughput ceiling (no rescale). 3 bindings.
    Variant {
        name: "gemm_q4_0_i8_raw_k5376_n21504",
        src: "gemm_q4_0_i8_raw",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 21504),
            ("WG_X", 64),
            ("M_TILES", 2),
            ("N_TILES", 4),
        ],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // coopmat-arith smoke test (docs/naga-coopmat-arith-patch.md): proves the
    // fork's component-wise FMul + f32(coop<i32>) convert. 3 bindings.
    Variant {
        name: "coop_arith_smoke",
        src: "coop_arith_smoke",
        defs: &[],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // int8-coopmat toolchain smoke test (docs/naga-int8-coopmat-patch.md):
    // proves the naga fork emits signed-int8 coopmat SPIR-V. Not production.
    Variant {
        name: "coop_i8_smoke",
        src: "coop_i8_smoke",
        defs: &[],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Probe: coopLoad an int8 coopmat from WORKGROUP (LDS) memory (for the
    // Q4_0-reading int8 GEMM's unpack-to-LDS path). Not production.
    Variant {
        name: "coop_i8_lds_smoke",
        src: "coop_i8_lds_smoke",
        defs: &[],
        workgroup: [64, 1, 1],
        bindings: 2,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Coopmat global-prefill rewrite (profile rank #1), register-resident-O
    // two-pass. S_STAGE_LEN = M_Q·N_K.
    Variant {
        name: "attn_prefill_global_flash",
        src: "attn_prefill_global_flash",
        defs: &[
            ("HEAD_DIM", 512),
            ("N_KV_HEADS", 4),
            ("Q_PER_KV", 8),
            ("M_Q", 16),
            ("N_K", 64),
            ("WG_X", 64),
            ("S_STAGE_LEN", 1024),
        ],
        workgroup: [64, 1, 1],
        bindings: 5,
        push_bytes: 4,
        subgroup_size: 0,
        raw: true,
    },
    // Coopmat Q4_0 GEMM (prefill), one variant per (K, N) site.
    Variant {
        name: "gemm_q4_0_k5376_n8192",
        src: "gemm_q4_0",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 8192),
            ("M_TILES", 4),
            ("N_TILES", 4),
            ("B_TILE_LEN", 4096),
            ("ACC_LEN", 16),
        ],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_k5376_n4096",
        src: "gemm_q4_0",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 4096),
            ("M_TILES", 4),
            ("N_TILES", 4),
            ("B_TILE_LEN", 4096),
            ("ACC_LEN", 16),
        ],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_k8192_n5376",
        src: "gemm_q4_0",
        defs: &[
            ("K_DIM", 8192),
            ("N_DIM", 5376),
            ("M_TILES", 4),
            ("N_TILES", 4),
            ("B_TILE_LEN", 4096),
            ("ACC_LEN", 16),
        ],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_k5376_n16384",
        src: "gemm_q4_0",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 16384),
            ("M_TILES", 4),
            ("N_TILES", 4),
            ("B_TILE_LEN", 4096),
            ("ACC_LEN", 16),
        ],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_k5376_n2048",
        src: "gemm_q4_0",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 2048),
            ("M_TILES", 4),
            ("N_TILES", 4),
            ("B_TILE_LEN", 4096),
            ("ACC_LEN", 16),
        ],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_k16384_n5376",
        src: "gemm_q4_0",
        defs: &[
            ("K_DIM", 16384),
            ("N_DIM", 5376),
            ("M_TILES", 4),
            ("N_TILES", 4),
            ("B_TILE_LEN", 4096),
            ("ACC_LEN", 16),
        ],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_k5376_n21504",
        src: "gemm_q4_0",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 21504),
            ("M_TILES", 4),
            ("N_TILES", 4),
            ("B_TILE_LEN", 4096),
            ("ACC_LEN", 16),
        ],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Occupancy-lever experiments: lower M_TILES → fewer acc VGPRs → more waves
    // (f16 prefill gemm is memory-latency-bound at 25 % occupancy; RGP 2026-06-14).
    // N_TILES stays 4 (one W-row per thread = WG_X), so b_tile LDS is unchanged.
    // Workgroup swizzle (M-blocks fast-varying) for L2 weight-strip reuse — the
    // memory-bound lever (RGP 2026-06-14). Dispatch is transposed in the bench.
    Variant {
        name: "gemm_q4_0_swz_k5376_n21504",
        src: "gemm_q4_0",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 21504),
            ("M_TILES", 4),
            ("N_TILES", 4),
            ("B_TILE_LEN", 4096),
            ("ACC_LEN", 16),
            ("SWIZZLE", 1),
        ],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Swizzled f16 gemms for every prefill site (SWIZZLE=1 → graph dispatches
    // transposed); the plain `gemm_q4_0_k*` stay for the bench/parity baseline.
    Variant {
        name: "gemm_q4_0_swz_k5376_n8192",
        src: "gemm_q4_0",
        defs: &[("K_DIM", 5376), ("N_DIM", 8192), ("M_TILES", 4), ("N_TILES", 4),
                ("B_TILE_LEN", 4096), ("ACC_LEN", 16), ("SWIZZLE", 1)],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_swz_k5376_n4096",
        src: "gemm_q4_0",
        defs: &[("K_DIM", 5376), ("N_DIM", 4096), ("M_TILES", 4), ("N_TILES", 4),
                ("B_TILE_LEN", 4096), ("ACC_LEN", 16), ("SWIZZLE", 1)],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_swz_k8192_n5376",
        src: "gemm_q4_0",
        defs: &[("K_DIM", 8192), ("N_DIM", 5376), ("M_TILES", 4), ("N_TILES", 4),
                ("B_TILE_LEN", 4096), ("ACC_LEN", 16), ("SWIZZLE", 1)],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_swz_k5376_n16384",
        src: "gemm_q4_0",
        defs: &[("K_DIM", 5376), ("N_DIM", 16384), ("M_TILES", 4), ("N_TILES", 4),
                ("B_TILE_LEN", 4096), ("ACC_LEN", 16), ("SWIZZLE", 1)],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_swz_k5376_n2048",
        src: "gemm_q4_0",
        defs: &[("K_DIM", 5376), ("N_DIM", 2048), ("M_TILES", 4), ("N_TILES", 4),
                ("B_TILE_LEN", 4096), ("ACC_LEN", 16), ("SWIZZLE", 1)],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_swz_k16384_n5376",
        src: "gemm_q4_0",
        defs: &[("K_DIM", 16384), ("N_DIM", 5376), ("M_TILES", 4), ("N_TILES", 4),
                ("B_TILE_LEN", 4096), ("ACC_LEN", 16), ("SWIZZLE", 1)],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_swz_k21504_n5376",
        src: "gemm_q4_0",
        defs: &[("K_DIM", 21504), ("N_DIM", 5376), ("M_TILES", 4), ("N_TILES", 4),
                ("B_TILE_LEN", 4096), ("ACC_LEN", 16), ("SWIZZLE", 1)],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_m2_k5376_n21504",
        src: "gemm_q4_0",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 21504),
            ("M_TILES", 2),
            ("N_TILES", 4),
            ("B_TILE_LEN", 4096),
            ("ACC_LEN", 8),
        ],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_m1_k5376_n21504",
        src: "gemm_q4_0",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 21504),
            ("M_TILES", 1),
            ("N_TILES", 4),
            ("B_TILE_LEN", 4096),
            ("ACC_LEN", 4),
        ],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_k21504_n5376",
        src: "gemm_q4_0",
        defs: &[
            ("K_DIM", 21504),
            ("N_DIM", 5376),
            ("M_TILES", 4),
            ("N_TILES", 4),
            ("B_TILE_LEN", 4096),
            ("ACC_LEN", 16),
        ],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Subgroup-tiled Q4_0 GEMM: non-coopmat baseline/fallback.
    Variant {
        name: "gemm_st_q4_0_k5376_n8192",
        src: "gemm_st_q4_0",
        defs: &[("K_DIM", 5376), ("N_DIM", 8192)],
        workgroup: [256, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "gemm_st_q4_0_k5376_n4096",
        src: "gemm_st_q4_0",
        defs: &[("K_DIM", 5376), ("N_DIM", 4096)],
        workgroup: [256, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "gemm_st_q4_0_k8192_n5376",
        src: "gemm_st_q4_0",
        defs: &[("K_DIM", 8192), ("N_DIM", 5376)],
        workgroup: [256, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "gemm_st_q4_0_k5376_n16384",
        src: "gemm_st_q4_0",
        defs: &[("K_DIM", 5376), ("N_DIM", 16384)],
        workgroup: [256, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "gemm_st_q4_0_k5376_n2048",
        src: "gemm_st_q4_0",
        defs: &[("K_DIM", 5376), ("N_DIM", 2048)],
        workgroup: [256, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "gemm_st_q4_0_k16384_n5376",
        src: "gemm_st_q4_0",
        defs: &[("K_DIM", 16384), ("N_DIM", 5376)],
        workgroup: [256, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "gemm_st_q4_0_k5376_n21504",
        src: "gemm_st_q4_0",
        defs: &[("K_DIM", 5376), ("N_DIM", 21504)],
        workgroup: [256, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    Variant {
        name: "gemm_st_q4_0_k21504_n5376",
        src: "gemm_st_q4_0",
        defs: &[("K_DIM", 21504), ("N_DIM", 5376)],
        workgroup: [256, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
];

fn main() {
    println!("cargo::rerun-if-changed=shaders");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let mut registry = String::from(
        "/// Generated by build.rs — one entry per compiled kernel variant.\n\
         pub static KERNELS: &[KernelBlob] = &[\n",
    );
    for v in VARIANTS {
        let path = PathBuf::from(format!("shaders/{}.wgsl", v.src));
        let spv = compile(&path, v);
        let bytes: Vec<u8> = spv.iter().flat_map(|w| w.to_le_bytes()).collect();
        fs::write(out_dir.join(format!("{}.spv", v.name)), bytes).expect("write .spv");
        writeln!(
            registry,
            "    KernelBlob {{ name: {:?}, spv: include_bytes!(concat!(env!(\"OUT_DIR\"), \
             \"/{}.spv\")), workgroup: {:?}, bindings: {}, push_bytes: {}, subgroup_size: {} }},",
            v.name, v.name, v.workgroup, v.bindings, v.push_bytes, v.subgroup_size
        )
        .unwrap();
    }
    registry.push_str("];\n");
    fs::write(out_dir.join("kernels.rs"), registry).expect("write kernels.rs");
}

fn compile(path: &std::path::Path, variant: &Variant) -> Vec<u32> {
    let source =
        fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let display = format!("{} [{}]", path.display(), variant.name);

    let module = if variant.raw {
        // Plain naga path (see `Variant::raw`): textual `#{NAME}` substitution
        // only.
        let mut substituted = source;
        for (k, v) in variant.defs {
            substituted = substituted.replace(&format!("#{{{k}}}"), &v.to_string());
        }
        for (k, v) in [
            ("WG_X", variant.workgroup[0]),
            ("WG_Y", variant.workgroup[1]),
            ("WG_Z", variant.workgroup[2]),
        ] {
            substituted = substituted.replace(&format!("#{{{k}}}"), &v.to_string());
        }
        // gemm_q4_0_i8 reads `#{STAGE_BUFS}`; default to 2 (double-buffer) unless
        // the variant set it in `defs` above (large-K shapes use 1 — occupancy).
        substituted = substituted.replace("#{STAGE_BUFS}", "2");
        // gemm_q4_0 reads `#{SWIZZLE}`; default 0 (x=N, y=M) unless overridden.
        substituted = substituted.replace("#{SWIZZLE}", "0");
        naga::front::wgsl::parse_str(&substituted)
            .unwrap_or_else(|e| panic!("parse {display}: {}", e.emit_to_string(&substituted)))
    } else {
        let mut shader_defs: HashMap<String, ShaderDefValue> = HashMap::new();
        shader_defs.insert("WG_X".into(), ShaderDefValue::UInt(variant.workgroup[0]));
        shader_defs.insert("WG_Y".into(), ShaderDefValue::UInt(variant.workgroup[1]));
        shader_defs.insert("WG_Z".into(), ShaderDefValue::UInt(variant.workgroup[2]));
        for (k, v) in variant.defs {
            shader_defs.insert((*k).to_owned(), ShaderDefValue::UInt(*v));
        }

        // The composer validates internally with its own capability set;
        // default capabilities reject immediates (push constants), f16,
        // subgroups, …
        let mut composer = naga_oil::compose::Composer::default()
            .with_capabilities(naga::valid::Capabilities::all());
        composer
            .make_naga_module(naga_oil::compose::NagaModuleDescriptor {
                source: &source,
                file_path: &display,
                shader_defs,
                ..Default::default()
            })
            .unwrap_or_else(|e| panic!("compose {display}: {e}"))
    };

    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("validate {display}: {e:?}"));

    // Whitelist the SPIR-V capabilities matching the device features
    // GpuContext enables; without this naga reaches for capabilities like
    // StorageInputOutput16 (16-bit stage IO) that compute storage access
    // doesn't need and the device doesn't enable. Needing a capability
    // outside this set is a build error — extend both this list and
    // `context::required_features` together.
    let capabilities = [
        naga::back::spv::Capability::Shader,
        naga::back::spv::Capability::Float16,
        naga::back::spv::Capability::Int8,
        naga::back::spv::Capability::Int16,
        naga::back::spv::Capability::StorageBuffer8BitAccess,
        naga::back::spv::Capability::StorageBuffer16BitAccess,
        naga::back::spv::Capability::UniformAndStorageBuffer16BitAccess,
        naga::back::spv::Capability::GroupNonUniform,
        naga::back::spv::Capability::GroupNonUniformArithmetic,
        naga::back::spv::Capability::GroupNonUniformBallot,
        naga::back::spv::Capability::GroupNonUniformShuffle,
        naga::back::spv::Capability::GroupNonUniformShuffleRelative,
        naga::back::spv::Capability::CooperativeMatrixKHR,
        naga::back::spv::Capability::VulkanMemoryModel,
    ];
    let options = naga::back::spv::Options {
        capabilities: Some(capabilities.into_iter().collect()),
        // No f16 stage IO in compute; without this naga declares
        // StorageInputOutput16 whenever f16 is enabled.
        use_storage_input_output_16: false,
        // Our kernels never read workgroup memory they haven't written; the
        // polyfill zeroing is a serialized single-lane LDS sweep at every
        // workgroup launch (clearly visible in the gemm ISA dump).
        zero_initialize_workgroup_memory: naga::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
        // The injected loop-bound guards add a scalar-compare + branch chain
        // per loop iteration in the hot kernels; all our loops have baked
        // compile-time bounds.
        force_loop_bounding: false,
        ..Default::default()
    };
    naga::back::spv::write_vec(&module, &info, &options, None)
        .unwrap_or_else(|e| panic!("spv-out {display}: {e}"))
}
