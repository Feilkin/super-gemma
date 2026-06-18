//! Single-submit dispatch of one prefill GEMM, for RGP/SQTT capture
//! (docs/rgp-capture.md). One `dispatch_blocking` == one queue submit, so under
//! `MESA_VK_TRACE=rgp MESA_VK_TRACE_PER_SUBMIT=true` each loop iteration emits
//! one `.rgp` — take the LAST (warmest). Buffers are dummy: SQTT captures the
//! dispatch's wave/stall/occupancy behavior, not the result (a GEMM has no
//! data-dependent control flow, so the trace is representative).

use sg_gpu::{BufferUsage, GpuContext, WriteDescriptorSet};

const M: usize = 256; // prefill chunk rows (the M dimension)
const SUBMITS: usize = 8;

/// (K, N, N-block, M-block, int8, swizzled) for the supported capture targets.
/// `swizzled` kernels take a transposed `[M-blocks, N-blocks]` grid (the L2
/// fast-varying-M dispatch); the rest take `[N-blocks, M-blocks]`.
fn shape(kernel: &str) -> Option<(usize, usize, u32, u32, bool, bool)> {
    Some(match kernel {
        // int8 4×1 swizzled (N-block 16, M-block 64) — the DEPLOYED prefill
        // tiling (graph.rs); the occupancy/L2 recapture target (STATUS).
        "gemm_q4_0_i8_swz_m4n1_k21504_n5376" => (21504, 5376, 16, 64, true, true), // FFN down
        "gemm_q4_0_i8_swz_m4n1_k5376_n21504" => (5376, 21504, 16, 64, true, true), // FFN gate/up
        // int8 2×2 (N-block = M-block = 32) — the pre-cache-block baseline.
        "gemm_q4_0_i8_t22_k21504_n5376" => (21504, 5376, 32, 32, true, false), // FFN down
        "gemm_q4_0_i8_t22_k5376_n21504" => (5376, 21504, 32, 32, true, false), // FFN gate/up
        // f16 4×4 (N-block = M-block = 64) — every prefill GEMM site, for the
        // f16 A/B and the un-RGP'd-f16 investigation (STATUS).
        "gemm_q4_0_k5376_n21504" => (5376, 21504, 64, 64, false, false), // FFN gate/up
        "gemm_q4_0_k21504_n5376" => (21504, 5376, 64, 64, false, false), // FFN down
        "gemm_q4_0_k5376_n8192" => (5376, 8192, 64, 64, false, false),   // Q sliding
        "gemm_q4_0_k5376_n4096" => (5376, 4096, 64, 64, false, false),   // KV sliding
        "gemm_q4_0_k8192_n5376" => (8192, 5376, 64, 64, false, false),   // O sliding
        "gemm_q4_0_k5376_n16384" => (5376, 16384, 64, 64, false, false), // Q global
        "gemm_q4_0_k5376_n2048" => (5376, 2048, 64, 64, false, false),   // KV global
        "gemm_q4_0_k16384_n5376" => (16384, 5376, 64, 64, false, false), // O global
        _ => return None,
    })
}

pub fn run(kernel: &str) -> anyhow::Result<()> {
    let (k, n, nb, mb, int8, swz) = shape(kernel).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown rgp kernel `{kernel}`; the deployed int8 FFN gemms \
             (gemm_q4_0_i8_swz_m4n1_k{{21504_n5376,5376_n21504}}), the int8 2×2 \
             baseline (…_i8_t22_…), or any f16 prefill site (gemm_q4_0_k<K>_n<N>) \
             — see `shape()`"
        )
    })?;

    let ctx = GpuContext::new().map_err(|e| anyhow::anyhow!("gpu: {e}"))?;
    let kern = ctx.load_kernel(kernel).map_err(|e| anyhow::anyhow!("{e}"))?;
    let u = BufferUsage::STORAGE_BUFFER;
    let nb_err = |e: sg_gpu::GpuError| anyhow::anyhow!("{e}");

    // Q4_0 weights (u32 words): N·K/32 blocks × 18 bytes. Dummy-filled.
    let w = ctx
        .new_buffer::<u32>((n * k / 32 * 18 / 4) as u64, u)
        .map_err(nb_err)?;
    let y = ctx.new_buffer::<u16>((M * n) as u64, u).map_err(nb_err)?;
    // int8 path: x = Q8 quants (u32-packed = i8 bytes), x_scales = f16. f16 path:
    // x = f16 activations.
    let x_i8 = ctx.new_buffer::<u32>((M * k / 4) as u64, u).map_err(nb_err)?;
    let xs = ctx.new_buffer::<u16>((M * k / 32) as u64, u).map_err(nb_err)?;
    let x_f16 = ctx.new_buffer::<u16>((M * k) as u64, u).map_err(nb_err)?;

    // Swizzled kernels expect [M-blocks, N-blocks]; the rest [N-blocks, M-blocks].
    let groups = if swz {
        [M as u32 / mb, n as u32 / nb, 1]
    } else {
        [n as u32 / nb, M as u32 / mb, 1]
    };
    eprintln!(
        "RGP capture target: {kernel}  (M={M} K={k} N={n})  grid {groups:?}\n\
         {SUBMITS} submits — under MESA_VK_TRACE_PER_SUBMIT take the LAST .rgp."
    );
    for i in 0..SUBMITS {
        let writes = if int8 {
            vec![
                WriteDescriptorSet::buffer(0, w.clone()),
                WriteDescriptorSet::buffer(1, x_i8.clone()),
                WriteDescriptorSet::buffer(2, xs.clone()),
                WriteDescriptorSet::buffer(3, y.clone()),
            ]
        } else {
            vec![
                WriteDescriptorSet::buffer(0, w.clone()),
                WriteDescriptorSet::buffer(1, x_f16.clone()),
                WriteDescriptorSet::buffer(2, y.clone()),
            ]
        };
        ctx.dispatch_blocking(&kern, writes, None::<u32>, groups)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        eprintln!("  submit {i}/{SUBMITS} done");
    }
    Ok(())
}
