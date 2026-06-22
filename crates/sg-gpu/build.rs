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
    // Phase-0 MALL probe (STATUS 2026-06-21): strided streaming re-read, push =
    // {elems, reps}. Run sub- vs super-MALL (SG_PROBE_MB) to disambiguate whether
    // RGP "local video memory bytes" counts Infinity Cache hits or only DRAM.
    Variant {
        name: "mall_probe",
        src: "mall_probe",
        defs: &[],
        workgroup: [256, 1, 1],
        bindings: 2,
        push_bytes: 8,
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
    // Quantize-and-append for the Q8 global KV cache (Piece B): f16→Q8_0 with
    // linear slot placement (ROW_LEN = kv_dim_gl = 2048). One thread per block.
    Variant {
        name: "kv_append_global_q8",
        src: "kv_append_global_q8",
        defs: &[("ROW_LEN", 2048)],
        workgroup: [256, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: false,
    },
    // Quantize-and-append for the int8 V of the global cache (Piece B): V is
    // blocked along the KEY axis (32-key blocks) so the PV per-block scale
    // factors out of the i8 dot. One thread per (32-key block, head, 4-col).
    Variant {
        name: "kv_append_global_v_q8",
        src: "kv_append_global_v_q8",
        defs: &[("N_KV_HEADS", 4), ("HEAD_DIM", 512), ("WG_X", 64)],
        workgroup: [64, 1, 1],
        bindings: 4,
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
    // Global decode against the Q8 K cache (Piece A): K binding split into
    // i8 quants + f16 scales (6 bindings); V/Q stay f16. Same shape/defs as
    // attn_decode_global.
    Variant {
        name: "attn_decode_global_q8k",
        src: "attn_decode_global_q8k",
        defs: &[("HEAD_DIM", 512), ("N_KV_HEADS", 4), ("Q_PER_KV", 8)],
        workgroup: [64, 1, 1],
        bindings: 6,
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
    // Depth-2 weight prefetch at the parity shape (correctness of the w_next2
    // shift/prologue; the PD path is tile-independent so 2×2 covers it).
    Variant {
        name: "gemm_q4_0_i8_pd2_k512_n128",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 512),
            ("N_DIM", 128),
            ("WG_X", 64),
            ("M_TILES", 2),
            ("N_TILES", 2),
            ("PREFETCH_DEPTH", 2),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Activation prefetch at the parity shape (hoisted A-tile loads; the AXPF
    // path is tile-independent so 2×2 covers the load/MMA reorder correctness).
    Variant {
        name: "gemm_q4_0_i8_axpf_k512_n128",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 512),
            ("N_DIM", 128),
            ("WG_X", 64),
            ("M_TILES", 2),
            ("N_TILES", 2),
            ("AXPF", 1),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Cross-barrier activation prefetch at the parity shape (ap0/ap1 issued before
    // the barrier, consumed after — load-before-fence + held-across reorder).
    Variant {
        name: "gemm_q4_0_i8_axpf2_k512_n128",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 512),
            ("N_DIM", 128),
            ("WG_X", 64),
            ("M_TILES", 2),
            ("N_TILES", 2),
            ("AXPF", 2),
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
    // Swizzled int8 FFN gemms (SWIZZLE=1 → graph dispatches transposed) for the
    // L2 weight-reuse lever; plain `t22_*` stay for bench/parity baseline.
    Variant {
        name: "gemm_q4_0_i8_swz_t22_k5376_n21504",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 21504),
            ("WG_X", 64),
            ("M_TILES", 2),
            ("N_TILES", 2),
            ("SWIZZLE", 1),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Tall-thin tile sweep (cache-blocking: M_TILES sets how many M-rows share
    // each weight DRAM/LDS load; small N_TILES keeps `wb` LDS low for occupancy).
    // Benched against the 2×2 above on the FFN-up shape — mmq_tflops.
    Variant {
        name: "gemm_q4_0_i8_swz_m4n1_k5376_n21504",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 21504),
            ("WG_X", 64),
            ("M_TILES", 4),
            ("N_TILES", 1),
            ("SWIZZLE", 1),
            ("BCAST", 1),      // 0-stride rescale (A/B win on K=5376)
            ("STAGE_BUFS", 1), // `stage` is epilogue-only under BCAST → don't double-size
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // 4×1 on the down shape (large-K → n5376), both stage-buffer settings.
    Variant {
        name: "gemm_q4_0_i8_swz_m4n1_k21504_n5376",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 21504),
            ("N_DIM", 5376),
            ("WG_X", 64),
            ("M_TILES", 4),
            ("N_TILES", 1),
            ("SWIZZLE", 1),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Direct-coopStore epilogue A/B (EPI=1) on the deployed down kernel — drops the
    // LDS round-trip in the epilogue (mechanism A/B; not graph-safe, see EPI comment).
    Variant {
        name: "gemm_q4_0_i8_swz_m4n1_epi_k21504_n5376",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 21504),
            ("N_DIM", 5376),
            ("WG_X", 64),
            ("M_TILES", 4),
            ("N_TILES", 1),
            ("SWIZZLE", 1),
            ("EPI", 1),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_swz_m4n1_s1_k21504_n5376",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 21504),
            ("N_DIM", 5376),
            ("WG_X", 64),
            ("M_TILES", 4),
            ("N_TILES", 1),
            ("STAGE_BUFS", 1),
            ("SWIZZLE", 1),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Depth-2 weight-prefetch A/Bs of the two deployed 4×1 FFN shapes (PD=2 vs the
    // default PD=1): the MLP lever for the memory-LATENCY-bound GEMM (STATUS
    // 2026-06-21). Identical to the deployed down/up variants but PREFETCH_DEPTH=2;
    // +9 VGPR/wave — bench (mmq_variance) + RGP (occupancy + the vmcnt/first-WMMA
    // stall) decide whether the extra outstanding load beats the lost wave.
    Variant {
        name: "gemm_q4_0_i8_swz_m4n1_pd2_k21504_n5376",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 21504),
            ("N_DIM", 5376),
            ("WG_X", 64),
            ("M_TILES", 4),
            ("N_TILES", 1),
            ("SWIZZLE", 1),
            ("PREFETCH_DEPTH", 2),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_swz_m4n1_pd2_k5376_n21504",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 21504),
            ("WG_X", 64),
            ("M_TILES", 4),
            ("N_TILES", 1),
            ("SWIZZLE", 1),
            ("BCAST", 1),
            ("STAGE_BUFS", 1),
            ("PREFETCH_DEPTH", 2),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Activation-prefetch A/Bs of the two deployed 4×1 FFN shapes (AXPF=1 vs the
    // default inline X load): the indicated lever — the pre-first-WMMA stall's
    // vmcnt half is the unprefetched `coopLoadT` of X (RGP 2026-06-21). Hoists
    // M_TILES A-loads ahead of the MMA loop; ~8 tiny i8 A-fragments of VGPR.
    Variant {
        name: "gemm_q4_0_i8_swz_m4n1_axpf_k21504_n5376",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 21504),
            ("N_DIM", 5376),
            ("WG_X", 64),
            ("M_TILES", 4),
            ("N_TILES", 1),
            ("SWIZZLE", 1),
            ("AXPF", 1),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_swz_m4n1_axpf_k5376_n21504",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 21504),
            ("WG_X", 64),
            ("M_TILES", 4),
            ("N_TILES", 1),
            ("SWIZZLE", 1),
            ("BCAST", 1),
            ("STAGE_BUFS", 1),
            ("AXPF", 1),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // AXPF=2: CROSS-BARRIER activation prefetch (issue every A-load before the
    // unpack+barrier so X-latency overlaps them) — the indicated fix for the
    // fenced vmcnt stall the AXPF=1 hoist couldn't reach (RGP 2026-06-21).
    Variant {
        name: "gemm_q4_0_i8_swz_m4n1_axpf2_k21504_n5376",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 21504),
            ("N_DIM", 5376),
            ("WG_X", 64),
            ("M_TILES", 4),
            ("N_TILES", 1),
            ("SWIZZLE", 1),
            ("AXPF", 2),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_swz_m4n1_axpf2_k5376_n21504",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 21504),
            ("WG_X", 64),
            ("M_TILES", 4),
            ("N_TILES", 1),
            ("SWIZZLE", 1),
            ("BCAST", 1),
            ("STAGE_BUFS", 1),
            ("AXPF", 2),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // ── Clean no-frills 4×1 baseline (gemm_q4_0_i8_basic) — readable reference +
    // barrier probe (STATUS 2026-06-21). No β×2/prefetch/swizzle/stage. Default
    // all barriers on; the _nowb/_noda/_nowar parity variants drop one each to
    // learn which workgroupBarrier is actually required.
    Variant {
        name: "gemm_q4_0_i8_basic_k512_n128",
        src: "gemm_q4_0_i8_basic",
        defs: &[("K_DIM", 512), ("N_DIM", 128), ("WG_X", 64)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_basic_nowb_k512_n128",
        src: "gemm_q4_0_i8_basic",
        defs: &[("K_DIM", 512), ("N_DIM", 128), ("WG_X", 64), ("BAR_WB", 0)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_basic_noda_k512_n128",
        src: "gemm_q4_0_i8_basic",
        defs: &[("K_DIM", 512), ("N_DIM", 128), ("WG_X", 64), ("BAR_DA", 0)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_basic_nowar_k512_n128",
        src: "gemm_q4_0_i8_basic",
        defs: &[("K_DIM", 512), ("N_DIM", 128), ("WG_X", 64), ("BAR_WAR", 0)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // L2-blocking experiment kernel: hardcoded down shape, 1D dispatch, swappable
    // tile_index decode (STATUS 2026-06-21). M flows through the dispatch size:
    // dispatch [(M/64)·(N/16), 1, 1].
    Variant {
        name: "gemm_q4_0_i8_l2",
        src: "gemm_q4_0_i8_l2",
        defs: &[],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // 2D super-block sweep (BN_SB n-blocks per super-block, m-outer).
    Variant {
        name: "gemm_q4_0_i8_l2_b2",
        src: "gemm_q4_0_i8_l2",
        defs: &[("BN_SB", 2)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_l2_b4",
        src: "gemm_q4_0_i8_l2",
        defs: &[("BN_SB", 4)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_l2_b8",
        src: "gemm_q4_0_i8_l2",
        defs: &[("BN_SB", 8)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Weight prefetch (PF=1, software pipeline) on the best L2-schedule (BN_SB=4)
    // and the plain transpose — the MLP lever (STATUS 2026-06-21).
    Variant {
        name: "gemm_q4_0_i8_l2_b4_pf",
        src: "gemm_q4_0_i8_l2",
        defs: &[("BN_SB", 4), ("PF", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_l2_pf",
        src: "gemm_q4_0_i8_l2",
        defs: &[("PF", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // β×2 WMMA-ILP: alone, + prefetch, + prefetch on the BN_SB=4 schedule (full combo).
    Variant {
        name: "gemm_q4_0_i8_l2_b4_b2",
        src: "gemm_q4_0_i8_l2",
        defs: &[("BN_SB", 4), ("B2", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_l2_b4_pf_b2",
        src: "gemm_q4_0_i8_l2",
        defs: &[("BN_SB", 4), ("PF", 1), ("B2", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Activation prefetch (AXP=1: hoist all this-iter X loads before the MMAs) on the
    // best combo — attacks the binding before-WMMA vmcnt stall on X. A/B vs b4_b2.
    Variant {
        name: "gemm_q4_0_i8_l2_b4_b2_axp",
        src: "gemm_q4_0_i8_l2",
        defs: &[("BN_SB", 4), ("B2", 1), ("AXP_L2", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // AXP=2: cross-iter activation pipeline (prefetch next β's X a whole iter ahead).
    Variant {
        name: "gemm_q4_0_i8_l2_b4_b2_axp2",
        src: "gemm_q4_0_i8_l2",
        defs: &[("BN_SB", 4), ("B2", 1), ("AXP_L2", 2)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // AXP=3: cross-iter, DOUBLE-BUFFERED (β-parity buffer select) — kills axp2's swaps.
    Variant {
        name: "gemm_q4_0_i8_l2_b4_b2_axp3",
        src: "gemm_q4_0_i8_l2",
        defs: &[("BN_SB", 4), ("B2", 1), ("AXP_L2", 3)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // axp4: static-unroll ping-pong (dedicated file) — double-buffered X prefetch
    // with two named buffers, step-by-4, no dynamic index / no swaps.
    Variant {
        name: "gemm_q4_0_i8_l2_axp4",
        src: "gemm_q4_0_i8_l2_axp4",
        defs: &[],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // axp4 + weight prefetch (WPF=1): cross-iter X ping-pong AND software-pipelined
    // weight load — the full deployed-style combo on the l2 design.
    Variant {
        name: "gemm_q4_0_i8_l2_axp4_pf",
        src: "gemm_q4_0_i8_l2_axp4",
        defs: &[("WPF", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Minimal-fetch prefetch (PFW=5: prefetch block β only, inline-load β+1) on the
    // best combo — hides the 0x124 weight-load stall at ~half the prefetch VGPR of
    // full PF. A/B vs b4_b2 (no PF) and b4_pf_b2 (full PF).
    Variant {
        name: "gemm_q4_0_i8_l2_b4_b2_pf5",
        src: "gemm_q4_0_i8_l2",
        defs: &[("BN_SB", 4), ("PF", 1), ("PFW", 5), ("B2", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // f16 scale staging on the best combo (BN_SB=4, β×2): stage dw2/da_l at native
    // f16 width instead of f32, convert per-fragment to f32 before the multiply. A/B
    // vs b4_b2 — LDS isn't the occupancy limiter here, so this probes whether it matters.
    Variant {
        name: "gemm_q4_0_i8_l2_b4_b2_sf16",
        src: "gemm_q4_0_i8_l2",
        defs: &[("BN_SB", 4), ("B2", 1), ("SF16", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // BN_SB sweep WITH β×2 (1st b = BN_SB, 2nd b2 = β×2) — does the L2-schedule optimum
    // shift once ILP changes the access timing? (b4_b2 above is BN_SB=4.)
    Variant {
        name: "gemm_q4_0_i8_l2_b1_b2",
        src: "gemm_q4_0_i8_l2",
        defs: &[("BN_SB", 1), ("B2", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_l2_b2_b2",
        src: "gemm_q4_0_i8_l2",
        defs: &[("BN_SB", 2), ("B2", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_l2_b8_b2",
        src: "gemm_q4_0_i8_l2",
        defs: &[("BN_SB", 8), ("B2", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // 8×1 tile (M_TILES=8): register-level weight reuse, NB_M=2. Plain + BN_SB=4.
    Variant {
        name: "gemm_q4_0_i8_l2_m8",
        src: "gemm_q4_0_i8_l2",
        defs: &[("M_TILES", 8)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_l2_m8_b4",
        src: "gemm_q4_0_i8_l2",
        defs: &[("M_TILES", 8), ("BN_SB", 4)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Occupancy isolation: basic with the per-tile epilogue (EPI_TILES=1, 1024 B
    // scratch) — higher occupancy, SAME coalesced store. Does it reproduce
    // basic_dir's +21% DRAM traffic (→ occupancy) or stay fast (→ the store)?
    Variant {
        name: "gemm_q4_0_i8_basic_e1_k512_n128",
        src: "gemm_q4_0_i8_basic",
        defs: &[("K_DIM", 512), ("N_DIM", 128), ("WG_X", 64), ("EPI_TILES", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_basic_e1_k21504_n5376",
        src: "gemm_q4_0_i8_basic",
        defs: &[("K_DIM", 21504), ("N_DIM", 5376), ("WG_X", 64), ("EPI_TILES", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Direct-store epilogue (f16 coopmat → coopStore to y, no LDS scratch): parity
    // of the f16-coopmat conversion + scalar-matched store, and the perf A/B.
    Variant {
        name: "gemm_q4_0_i8_basic_dir_k512_n128",
        src: "gemm_q4_0_i8_basic",
        defs: &[("K_DIM", 512), ("N_DIM", 128), ("WG_X", 64), ("EPI", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_basic_dir_k21504_n5376",
        src: "gemm_q4_0_i8_basic",
        defs: &[("K_DIM", 21504), ("N_DIM", 5376), ("WG_X", 64), ("EPI", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_basic_dir_k5376_n21504",
        src: "gemm_q4_0_i8_basic",
        defs: &[("K_DIM", 5376), ("N_DIM", 21504), ("WG_X", 64), ("EPI", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // basic baseline at the two deployed FFN shapes (bench + RGP reference).
    Variant {
        name: "gemm_q4_0_i8_basic_k21504_n5376",
        src: "gemm_q4_0_i8_basic",
        defs: &[("K_DIM", 21504), ("N_DIM", 5376), ("WG_X", 64)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // All three probed barriers OFF (parity-redundant on the single wave) — the
    // perf A/B vs basic: do the barriers cost cycles, or did ACO already elide them?
    Variant {
        name: "gemm_q4_0_i8_basic_nobar_k21504_n5376",
        src: "gemm_q4_0_i8_basic",
        defs: &[
            ("K_DIM", 21504),
            ("N_DIM", 5376),
            ("WG_X", 64),
            ("BAR_WB", 0),
            ("BAR_DA", 0),
            ("BAR_WAR", 0),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_basic_k5376_n21504",
        src: "gemm_q4_0_i8_basic",
        defs: &[("K_DIM", 5376), ("N_DIM", 21504), ("WG_X", 64)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Max-occupancy 1×1 variant (gemm_q4_0_i8_occ): one 16×16 output block per
    // workgroup, minimal VGPR → many waves/SIMD. The occupancy-side A/B against
    // the deployed 4×1 on the down shape — does latency-hiding-by-occupancy beat
    // in-register weight reuse? (mmq_variance + RGP wave count; STATUS rank #2.)
    Variant {
        name: "gemm_q4_0_i8_occ_k21504_n5376",
        src: "gemm_q4_0_i8_occ",
        defs: &[
            ("K_DIM", 21504),
            ("N_DIM", 5376),
            ("WG_X", 64),
            ("SWIZZLE", 1),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // FULL-OCCUPANCY GEMM (gemm_q4_0_i8_fo) — FFN-down shape only (K=21504,
    // N=5376), plain [M-blocks, N-blocks] dispatch (no swizzle). Lean small-tile
    // design: 0-stride scale rescale + direct coopStore epilogue + minimal LDS →
    // many waves/SIMD, the latency hidden by wave-switching instead of the
    // deployed kernel's in-wave ILP. Sweep the occupancy↔ILP frontier:
    // M_TILES (tile height) × B2 (β×2 ILP) × PD (weight prefetch). The headline
    // `fo` is M_TILES=2, no ILP, no prefetch (pure occupancy). mb (M_ROWS) for the
    // dispatch grid = M_TILES·16.
    Variant {
        name: "gemm_q4_0_i8_fo",
        src: "gemm_q4_0_i8_fo",
        defs: &[("M_TILES", 2)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_fo_m1",
        src: "gemm_q4_0_i8_fo",
        defs: &[("M_TILES", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_fo_m4",
        src: "gemm_q4_0_i8_fo",
        defs: &[("M_TILES", 4)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // β×2 ILP on each tile height (expected to regress once occupancy is high).
    Variant {
        name: "gemm_q4_0_i8_fo_m1_b2",
        src: "gemm_q4_0_i8_fo",
        defs: &[("M_TILES", 1), ("B2", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_fo_m2_b2",
        src: "gemm_q4_0_i8_fo",
        defs: &[("M_TILES", 2), ("B2", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Single-ahead weight prefetch on the headline tile (cheap MLP A/B).
    Variant {
        name: "gemm_q4_0_i8_fo_m2_pd1",
        src: "gemm_q4_0_i8_fo",
        defs: &[("M_TILES", 2), ("PD", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Activation-scale hoist (SXP=1): issue the per-block d_a load at the top of the
    // β-iter so its ~2K-clk latency overlaps the WMMAs instead of stalling the
    // rescale (RGP: the lone 16-bit x_scales load stalls as hard as the X loads).
    Variant {
        name: "gemm_q4_0_i8_fo_m2_sxp",
        src: "gemm_q4_0_i8_fo",
        defs: &[("M_TILES", 2), ("SXP", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_fo_m2_b2_sxp",
        src: "gemm_q4_0_i8_fo",
        defs: &[("M_TILES", 2), ("B2", 1), ("SXP", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_fo_m1_b2_sxp",
        src: "gemm_q4_0_i8_fo",
        defs: &[("M_TILES", 1), ("B2", 1), ("SXP", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // "BIGBOY" fully-unrolled FO (gemm_q4_0_i8_bb) — FFN-down shape, M_TILES=2, no
    // knobs. Every inner loop hand-unrolled so all of an iteration's global loads
    // (9 distinct-register weight words + 8 activation fragments + 2 d_a scales)
    // issue as one batch → a single vmcnt stall/iter, paid for in occupancy.
    Variant {
        name: "gemm_q4_0_i8_bb",
        src: "gemm_q4_0_i8_bb",
        defs: &[],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // bb + weight prefetch (PF=1): the bb trace showed the lone remaining stall is
    // the weight load — software-pipeline it (next pair's words issued in this
    // iter's batch) so it lands behind the WMMAs. Tests "weights are now the stall".
    Variant {
        name: "gemm_q4_0_i8_bb_pf",
        src: "gemm_q4_0_i8_bb",
        defs: &[("PF", 1)],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // MULTI-WAVE occupancy GEMM (gemm_q4_0_i8_mw) — TESTED DEAD END (2026-06-21),
    // kept as documented A/B baselines like gemm_q4_0_i8_occ; NOT in any graph. A
    // workgroup of BM_TILES·BN_TILES waves, ONE 16×16 tile per wave, sharing the
    // LDS weight strip (high occupancy without the _occ kernel's reuse loss) — but
    // RGP showed the occupancy thrashes L2 + a multi-wave s_barrier tax, and the
    // deployed GEMM is already ~86% bandwidth-bound, so occupancy can't help (see
    // the kernel header + STATUS 2026-06-21). WG_X = BM_TILES·BN_TILES·64. Parity
    // variants (k512_n128) cover the 2D wave grid (b22: wm/wn both vary), the 1D-M
    // max-occupancy config (b41: wn≡0), and the register-tiled half-occupancy path
    // (r2: RM=2). The k21504_n5376 b41/r2 pair is the down-shape bench/RGP A/B.
    Variant {
        name: "gemm_q4_0_i8_mw_b22_k512_n128",
        src: "gemm_q4_0_i8_mw",
        defs: &[
            ("K_DIM", 512),
            ("N_DIM", 128),
            ("WG_X", 256),
            ("BM_TILES", 2),
            ("BN_TILES", 2),
        ],
        workgroup: [256, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_mw_b41_k512_n128",
        src: "gemm_q4_0_i8_mw",
        defs: &[
            ("K_DIM", 512),
            ("N_DIM", 128),
            ("WG_X", 256),
            ("BM_TILES", 4),
            ("BN_TILES", 1),
        ],
        workgroup: [256, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Bench shapes (FFN up k5376_n21504, down k21504_n5376), SWIZZLE=1 → graph/
    // bench dispatch transposed. The occupancy×reuse sweep against the deployed
    // 4×1: b41 (reuse 64, 4 waves) is the headline; b22 (reuse 32, 4 waves, wider
    // N), b42 (reuse 64, 8 waves), b81 (reuse 128, 8 waves).
    Variant {
        name: "gemm_q4_0_i8_mw_b41_k5376_n21504",
        src: "gemm_q4_0_i8_mw",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 21504),
            ("WG_X", 256),
            ("BM_TILES", 4),
            ("BN_TILES", 1),
            ("SWIZZLE", 1),
        ],
        workgroup: [256, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_mw_b22_k5376_n21504",
        src: "gemm_q4_0_i8_mw",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 21504),
            ("WG_X", 256),
            ("BM_TILES", 2),
            ("BN_TILES", 2),
            ("SWIZZLE", 1),
        ],
        workgroup: [256, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_mw_b42_k5376_n21504",
        src: "gemm_q4_0_i8_mw",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 21504),
            ("WG_X", 512),
            ("BM_TILES", 4),
            ("BN_TILES", 2),
            ("SWIZZLE", 1),
        ],
        workgroup: [512, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_mw_b81_k5376_n21504",
        src: "gemm_q4_0_i8_mw",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 21504),
            ("WG_X", 512),
            ("BM_TILES", 8),
            ("BN_TILES", 1),
            ("SWIZZLE", 1),
        ],
        workgroup: [512, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_mw_b41_k21504_n5376",
        src: "gemm_q4_0_i8_mw",
        defs: &[
            ("K_DIM", 21504),
            ("N_DIM", 5376),
            ("WG_X", 256),
            ("BM_TILES", 4),
            ("BN_TILES", 1),
            ("SWIZZLE", 1),
        ],
        workgroup: [256, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_mw_b22_k21504_n5376",
        src: "gemm_q4_0_i8_mw",
        defs: &[
            ("K_DIM", 21504),
            ("N_DIM", 5376),
            ("WG_X", 256),
            ("BM_TILES", 2),
            ("BN_TILES", 2),
            ("SWIZZLE", 1),
        ],
        workgroup: [256, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_mw_b42_k21504_n5376",
        src: "gemm_q4_0_i8_mw",
        defs: &[
            ("K_DIM", 21504),
            ("N_DIM", 5376),
            ("WG_X", 512),
            ("BM_TILES", 4),
            ("BN_TILES", 2),
            ("SWIZZLE", 1),
        ],
        workgroup: [512, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_mw_b81_k21504_n5376",
        src: "gemm_q4_0_i8_mw",
        defs: &[
            ("K_DIM", 21504),
            ("N_DIM", 5376),
            ("WG_X", 512),
            ("BM_TILES", 8),
            ("BN_TILES", 1),
            ("SWIZZLE", 1),
        ],
        workgroup: [512, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Half-occupancy A/B vs b41: SAME 64×16 block + reuse 64, but 2 waves each
    // doing RM=2 register M-tiles (B coopLoad'd once, fed to both → in-wave ILP +
    // register reuse) → ~2× VGPR → ~half the waves/SIMD. Tests whether warmer L2
    // (fewer resident wavefronts) beats max occupancy. r2 parity covers the RM>1
    // register-tile + epilogue path.
    Variant {
        name: "gemm_q4_0_i8_mw_r2_k512_n128",
        src: "gemm_q4_0_i8_mw",
        defs: &[
            ("K_DIM", 512),
            ("N_DIM", 128),
            ("WG_X", 128),
            ("BM_TILES", 2),
            ("BN_TILES", 1),
            ("RM", 2),
        ],
        workgroup: [128, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_mw_r2_k21504_n5376",
        src: "gemm_q4_0_i8_mw",
        defs: &[
            ("K_DIM", 21504),
            ("N_DIM", 5376),
            ("WG_X", 128),
            ("BM_TILES", 2),
            ("BN_TILES", 1),
            ("RM", 2),
            ("SWIZZLE", 1),
        ],
        workgroup: [128, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_mw_r2_k5376_n21504",
        src: "gemm_q4_0_i8_mw",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 21504),
            ("WG_X", 128),
            ("BM_TILES", 2),
            ("BN_TILES", 1),
            ("RM", 2),
            ("SWIZZLE", 1),
        ],
        workgroup: [128, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // 4×1 attention shapes (Q/KV/O × sliding/global) for the cache-blocking
    // deployment — default STAGE_BUFS=2 (the 4×1 tile's low LDS makes the
    // double buffer free even on the large-K O shapes).
    Variant {
        name: "gemm_q4_0_i8_swz_m4n1_k5376_n8192", // Q sliding
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 8192),
            ("WG_X", 64),
            ("M_TILES", 4),
            ("N_TILES", 1),
            ("SWIZZLE", 1),
            ("BCAST", 1),      // 0-stride rescale (A/B win on K=5376)
            ("STAGE_BUFS", 1), // `stage` is epilogue-only under BCAST → don't double-size
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_swz_m4n1_k5376_n16384", // Q global
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 16384),
            ("WG_X", 64),
            ("M_TILES", 4),
            ("N_TILES", 1),
            ("SWIZZLE", 1),
            ("BCAST", 1),      // 0-stride rescale (A/B win on K=5376)
            ("STAGE_BUFS", 1), // `stage` is epilogue-only under BCAST → don't double-size
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_swz_m4n1_k5376_n4096", // KV sliding
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 4096),
            ("WG_X", 64),
            ("M_TILES", 4),
            ("N_TILES", 1),
            ("SWIZZLE", 1),
            ("BCAST", 1),      // 0-stride rescale (A/B win on K=5376)
            ("STAGE_BUFS", 1), // `stage` is epilogue-only under BCAST → don't double-size
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_swz_m4n1_k5376_n2048", // KV global
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 2048),
            ("WG_X", 64),
            ("M_TILES", 4),
            ("N_TILES", 1),
            ("SWIZZLE", 1),
            ("BCAST", 1),      // 0-stride rescale (A/B win on K=5376)
            ("STAGE_BUFS", 1), // `stage` is epilogue-only under BCAST → don't double-size
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_swz_m4n1_k8192_n5376", // O sliding
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 8192),
            ("N_DIM", 5376),
            ("WG_X", 64),
            ("M_TILES", 4),
            ("N_TILES", 1),
            ("SWIZZLE", 1),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_swz_m4n1_k16384_n5376", // O global
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 16384),
            ("N_DIM", 5376),
            ("WG_X", 64),
            ("M_TILES", 4),
            ("N_TILES", 1),
            ("SWIZZLE", 1),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // O global single-buffered: 4×1-s2 regressed on this largest-K shape; does
    // halving the stage LDS recover occupancy?
    Variant {
        name: "gemm_q4_0_i8_swz_m4n1_s1_k16384_n5376",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 16384),
            ("N_DIM", 5376),
            ("WG_X", 64),
            ("M_TILES", 4),
            ("N_TILES", 1),
            ("STAGE_BUFS", 1),
            ("SWIZZLE", 1),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_swz_m8n1_k5376_n21504",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 21504),
            ("WG_X", 64),
            ("M_TILES", 8),
            ("N_TILES", 1),
            ("SWIZZLE", 1),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_swz_m4n2_k5376_n21504",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 21504),
            ("WG_X", 64),
            ("M_TILES", 4),
            ("N_TILES", 2),
            ("SWIZZLE", 1),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_swz_t22_k21504_n5376",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 21504),
            ("N_DIM", 5376),
            ("WG_X", 64),
            ("M_TILES", 2),
            ("N_TILES", 2),
            ("STAGE_BUFS", 1),
            ("SWIZZLE", 1),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Swizzled int8 ATTENTION gemms (feature int8-ffn): Q/K/V read the shared
    // Q8-quantized `xn` (K=5376), O reads the Q8-quantized attention output.
    // Sliding/global differ in head count → distinct N (Q/KV) or K (O).
    Variant {
        name: "gemm_q4_0_i8_swz_t22_k5376_n8192", // Q sliding
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 8192),
            ("WG_X", 64),
            ("M_TILES", 2),
            ("N_TILES", 2),
            ("SWIZZLE", 1),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_swz_t22_k5376_n16384", // Q global
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 16384),
            ("WG_X", 64),
            ("M_TILES", 2),
            ("N_TILES", 2),
            ("SWIZZLE", 1),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_swz_t22_k5376_n4096", // KV sliding
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 4096),
            ("WG_X", 64),
            ("M_TILES", 2),
            ("N_TILES", 2),
            ("SWIZZLE", 1),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_swz_t22_k5376_n2048", // KV global
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 2048),
            ("WG_X", 64),
            ("M_TILES", 2),
            ("N_TILES", 2),
            ("SWIZZLE", 1),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_swz_t22_k8192_n5376", // O sliding (large-K → n5376)
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 8192),
            ("N_DIM", 5376),
            ("WG_X", 64),
            ("M_TILES", 2),
            ("N_TILES", 2),
            ("STAGE_BUFS", 1),
            ("SWIZZLE", 1),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_swz_t22_k16384_n5376", // O global
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 16384),
            ("N_DIM", 5376),
            ("WG_X", 64),
            ("M_TILES", 2),
            ("N_TILES", 2),
            ("STAGE_BUFS", 1),
            ("SWIZZLE", 1),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Double-buffered (STAGE_BUFS=2) O gemms — bench-only A/B against the
    // single-buffered deployed variants above (mmq_tflops; the O shape's K sits
    // between FFN up's 5376 and down's 21504, so it's not obvious which wins).
    Variant {
        name: "gemm_q4_0_i8_swz_s2_t22_k8192_n5376",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 8192),
            ("N_DIM", 5376),
            ("WG_X", 64),
            ("M_TILES", 2),
            ("N_TILES", 2),
            ("STAGE_BUFS", 2),
            ("SWIZZLE", 1),
        ],
        workgroup: [64, 1, 1],
        bindings: 4,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    Variant {
        name: "gemm_q4_0_i8_swz_s2_t22_k16384_n5376",
        src: "gemm_q4_0_i8",
        defs: &[
            ("K_DIM", 16384),
            ("N_DIM", 5376),
            ("WG_X", 64),
            ("M_TILES", 2),
            ("N_TILES", 2),
            ("STAGE_BUFS", 2),
            ("SWIZZLE", 1),
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
    // 0-stride coopLoad broadcast smoke test: build a 16×16 outer product from
    // 16+16 elements without an LDS matrix. Not production.
    Variant {
        name: "coop_bcast_smoke",
        src: "coop_bcast_smoke",
        defs: &[],
        workgroup: [64, 1, 1],
        bindings: 3,
        push_bytes: 0,
        subgroup_size: 0,
        raw: true,
    },
    // Same, but the 0-stride loads read from WORKGROUP (LDS) memory.
    Variant {
        name: "coop_bcast_lds_smoke",
        src: "coop_bcast_lds_smoke",
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
    // Coopmat global-prefill rewrite (profile rank #1), single-pass flash with
    // an in-register rescale (supersedes the two-pass above). S_STAGE_LEN =
    // M_Q·N_K; corr_stage/o_stage are M_Q·16 = 256.
    Variant {
        name: "attn_prefill_global_flash_sp",
        src: "attn_prefill_global_flash_sp",
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
    // Q8-KV flash (Piece B), f16-convert dead end (7× slow — kept as the labeled
    // "wrong approach" A/B baseline): dequant K/V → f16 in LDS, then f16 matmul.
    Variant {
        name: "attn_prefill_global_flash_sp_q8",
        src: "attn_prefill_global_flash_sp_q8",
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
        bindings: 7,
        push_bytes: 4,
        subgroup_size: 0,
        raw: true,
    },
    // Q8-KV flash (Piece B), the RIGHT approach: int8 QKᵀ (K i8 straight from the
    // Q8 cache + pre-quantized Q i8, per-block rescale like gemm_q4_0_i8), f16 PV
    // (V's quant axis ≠ the PV contraction). 7 bindings (q_i8, q_s, k_q, k_s, v,
    // out, step).
    Variant {
        name: "attn_prefill_global_flash_sp_iq",
        src: "attn_prefill_global_flash_sp_iq",
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
        bindings: 7,
        push_bytes: 4,
        subgroup_size: 0,
        raw: true,
    },
    // Q8-KV flash (Piece B) extended: int8 QKᵀ AND int8 PV. V streams i8 from the
    // cache (key-blocked scales, q8_quant_v) and P is quantized i8 in-kernel per
    // 32-key block; both matmuls i8×i8→i32 with per-block rescale. 8 bindings
    // (q_i8, q_s, k_q, k_s, v_q, v_s, out, step). P_SCALES_LEN = M_Q·(N_K/32).
    Variant {
        name: "attn_prefill_global_flash_sp_ipv",
        src: "attn_prefill_global_flash_sp_ipv",
        defs: &[
            ("HEAD_DIM", 512),
            ("N_KV_HEADS", 4),
            ("Q_PER_KV", 8),
            ("M_Q", 16),
            ("N_K", 64),
            ("WG_X", 64),
            ("S_STAGE_LEN", 1024),
            ("P_SCALES_LEN", 32),
        ],
        workgroup: [64, 1, 1],
        bindings: 8,
        push_bytes: 4,
        subgroup_size: 0,
        raw: true,
    },
    // PROTOTYPE: _ipv with both rescales' [16×16] scale built by 0-stride coopLoad
    // broadcasts (no LDS sc_stage, one fewer barrier per rescale). Same 8 bindings.
    Variant {
        name: "attn_prefill_global_flash_sp_ipv_bcast",
        src: "attn_prefill_global_flash_sp_ipv_bcast",
        defs: &[
            ("HEAD_DIM", 512),
            ("N_KV_HEADS", 4),
            ("Q_PER_KV", 8),
            ("M_Q", 16),
            ("N_K", 64),
            ("WG_X", 64),
            ("S_STAGE_LEN", 1024),
            ("P_SCALES_LEN", 32),
        ],
        workgroup: [64, 1, 1],
        bindings: 8,
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
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 8192),
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
    Variant {
        name: "gemm_q4_0_swz_k5376_n4096",
        src: "gemm_q4_0",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 4096),
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
    Variant {
        name: "gemm_q4_0_swz_k8192_n5376",
        src: "gemm_q4_0",
        defs: &[
            ("K_DIM", 8192),
            ("N_DIM", 5376),
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
    Variant {
        name: "gemm_q4_0_swz_k5376_n16384",
        src: "gemm_q4_0",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 16384),
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
    Variant {
        name: "gemm_q4_0_swz_k5376_n2048",
        src: "gemm_q4_0",
        defs: &[
            ("K_DIM", 5376),
            ("N_DIM", 2048),
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
    Variant {
        name: "gemm_q4_0_swz_k16384_n5376",
        src: "gemm_q4_0",
        defs: &[
            ("K_DIM", 16384),
            ("N_DIM", 5376),
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
    Variant {
        name: "gemm_q4_0_swz_k21504_n5376",
        src: "gemm_q4_0",
        defs: &[
            ("K_DIM", 21504),
            ("N_DIM", 5376),
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
        // gemm_q4_0_i8 reads `#{BCAST}`; default 0 (LDS `stage` rescale) unless the
        // variant opted into the 0-stride rescale (the K=5376 shapes — A/B win).
        substituted = substituted.replace("#{BCAST}", "0");
        // gemm_q4_0_i8_mw reads `#{RM}` (register M-tiles per wave); default 1
        // (one tile/wave, max occupancy) unless the variant lowers occupancy by
        // tiling RM tiles per wave (the half-occupancy A/B).
        substituted = substituted.replace("#{RM}", "1");
        // gemm_q4_0_i8 reads `#{PREFETCH_DEPTH}`; default 1 (deployed 1-deep weight
        // prefetch) unless a variant deepens it (the MLP A/B — latency-bound, not
        // bandwidth-bound; STATUS 2026-06-21). Only 1 or 2 are valid.
        substituted = substituted.replace("#{PREFETCH_DEPTH}", "1");
        // gemm_q4_0_i8 reads `#{AXPF}`; default 0 (deployed inline X load) unless a
        // variant hoists the A-tile loads ahead of the MMA loop (the activation-
        // prefetch A/B — the vmcnt stall is on X, not weights; STATUS 2026-06-21).
        substituted = substituted.replace("#{AXPF}", "0");
        // gemm_q4_0_i8_basic reads `#{BAR_WB}`/`#{BAR_DA}`/`#{BAR_WAR}`; default 1
        // (all barriers on = correct). A variant drops one (→ 0) to probe whether
        // that workgroupBarrier is actually required (STATUS 2026-06-21).
        substituted = substituted.replace("#{BAR_WB}", "1");
        substituted = substituted.replace("#{BAR_DA}", "1");
        substituted = substituted.replace("#{BAR_WAR}", "1");
        // gemm_q4_0_i8_basic reads `#{EPI}`; default 0 (LDS-scratch epilogue). 1 =
        // convert to f16 coopmat in registers + direct coopStore to y (no LDS).
        substituted = substituted.replace("#{EPI}", "0");
        // gemm_q4_0_i8_basic reads `#{EPI_TILES}`; default 4 (= M_TILES, one-shot
        // 4096 B scratch). 1 = per-tile passes, 1024 B scratch → higher occupancy,
        // same coalesced store (the occupancy-vs-store isolation probe).
        substituted = substituted.replace("#{EPI_TILES}", "4");
        // gemm_q4_0_i8_l2 reads `#{BN_SB}` (n-blocks per L2 super-block); default 1
        // (plain transpose). Variants sweep it (STATUS 2026-06-21).
        substituted = substituted.replace("#{BN_SB}", "1");
        // gemm_q4_0_i8_l2 reads `#{M_TILES}` (tall-thin M_TILES×1); default 4.
        substituted = substituted.replace("#{M_TILES}", "4");
        // gemm_q4_0_i8_l2 reads `#{PF}` (weight prefetch); default 0 (inline load).
        substituted = substituted.replace("#{PF}", "0");
        // gemm_q4_0_i8_l2 reads `#{PFW}` (words prefetched/pair when PF=1); default 9
        // (full pair). 5 = minimal-fetch (block β only, β+1 inline). Only 5..9 valid.
        substituted = substituted.replace("#{PFW}", "9");
        // gemm_q4_0_i8_l2 reads `#{B2}` (β×2 WMMA-ILP interleave); default 0.
        substituted = substituted.replace("#{B2}", "0");
        // gemm_q4_0_i8_l2 reads `#{AXP_L2}` (activation prefetch in the β×2 path);
        // default 0 (inline a-loads). 1 = hoist all this-iter X loads before the MMAs.
        substituted = substituted.replace("#{AXP_L2}", "0");
        // gemm_q4_0_i8_l2 scale-staging type: SF16=1 → stage the f16-sourced scales
        // as f16 in LDS and f32()-convert each fragment before the f32 multiply;
        // default (SF16 absent/0) keeps them f32 (SCALE_CVT empty — read is already
        // f32). String tokens, not the integer `defs` path, because they pick a type.
        let sf16 = variant.defs.iter().any(|(k, v)| *k == "SF16" && *v == 1);
        let (scale_ty, scale_cvt) = if sf16 { ("f16", "f32") } else { ("f32", "") };
        substituted = substituted.replace("#{SCALE_TY}", scale_ty);
        substituted = substituted.replace("#{SCALE_CVT}", scale_cvt);
        // gemm_q4_0_i8_l2_axp4 reads `#{WPF}` (weight prefetch on the ping-pong); default 0.
        substituted = substituted.replace("#{WPF}", "0");
        // gemm_q4_0_i8_fo reads `#{PD}` (weight prefetch depth, 0/1); default 0
        // (inline load). It also reuses `#{M_TILES}` (default 4, the occupancy knob)
        // and `#{B2}` (default 0). It has no barriers (WG=64 single-wave).
        substituted = substituted.replace("#{PD}", "0");
        // gemm_q4_0_i8_fo reads `#{SXP}` (hoist the d_a activation-scale load to the
        // top of the β-iter so its memory latency hides behind the WMMAs); default 0.
        substituted = substituted.replace("#{SXP}", "0");
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
