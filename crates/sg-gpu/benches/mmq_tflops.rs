//! int8-MMQ vs f16 coopmat GEMM compute-rate microbench (profile rank #2).
//! Same shape (M×K×N = 512×5376×21504, the FFN gate/up bucket), same effective
//! flop count (2·M·N·K) — the decisive question is whether the int8 MMA
//! throughput beats f16 *after* paying the per-block i32→f32 rescale through
//! LDS (docs/naga-int8-coopmat-patch.md). Skips without a coopmat GPU.

use criterion::{Criterion, criterion_group, criterion_main};
use sg_gpu::{BufferBinding, BufferUsage, GpuContext};

const M: usize = 512;
const DISPATCHES: usize = 4;
// Buffers are sized to the largest benched shape; each case binds the prefix it
// needs (the kernel's K_DIM/N_DIM defines + dispatch grid bound the reads).
const MAX_K: usize = 16384; // O global
const MAX_N: usize = 21504; // FFN up
const WEIGHT_NK: usize = 21504 * 5376; // largest N·K (FFN up) — covers every shape

fn bench(c: &mut Criterion) {
    let ctx = match GpuContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("skipping mmq bench: no usable GPU ({e})");
            return;
        }
    };
    if !ctx.cooperative_matrix {
        eprintln!("skipping mmq bench: no VK_KHR_cooperative_matrix");
        return;
    }

    // f16 gemm operands (sized to the largest shape; values irrelevant to timing).
    let weight_words = WEIGHT_NK / 32 * 18 / 4;
    let w_f16 = ctx
        .buffer_from_iter(
            (0..weight_words as u32).map(|i| i.wrapping_mul(0x9E37_79B9)),
            BufferUsage::STORAGE_BUFFER,
        )
        .expect("w_f16");
    let x_f16 = ctx
        .buffer_from_iter(
            (0..M * MAX_K).map(|i| half::f16::from_f32((i % 11) as f32 * 0.05).to_bits()),
            BufferUsage::STORAGE_BUFFER,
        )
        .expect("x_f16");
    // int8 MMQ operands.
    let w_i8 = ctx
        .buffer_from_iter(
            (0..WEIGHT_NK).map(|i| (i % 15) as i8 - 7),
            BufferUsage::STORAGE_BUFFER,
        )
        .expect("w_i8");
    let x_i8 = ctx
        .buffer_from_iter(
            (0..M * MAX_K).map(|i| (i % 31) as i8 - 15),
            BufferUsage::STORAGE_BUFFER,
        )
        .expect("x_i8");
    let x_sc = ctx
        .buffer_from_iter(
            (0..M * MAX_K / 32).map(|i| half::f16::from_f32((i % 5) as f32 * 0.02).to_bits()),
            BufferUsage::STORAGE_BUFFER,
        )
        .expect("x_sc");
    let y = ctx
        .new_buffer::<u16>((M * MAX_N) as u64, BufferUsage::STORAGE_BUFFER)
        .expect("y");

    // (name, descriptor writes, n_block, m_block).
    let f16_writes = vec![
        BufferBinding::buffer(0, w_f16.clone()),
        BufferBinding::buffer(1, x_f16.clone()),
        BufferBinding::buffer(2, y.clone()),
    ];
    // int8 MMQ now reads the SAME Q4_0 packed weights the f16 gemm does (it
    // unpacks nibbles → i8 in LDS in-kernel) — apples-to-apples, no repack.
    let i8_writes = vec![
        BufferBinding::buffer(0, w_f16.clone()),
        BufferBinding::buffer(1, x_i8.clone()),
        BufferBinding::buffer(2, x_sc.clone()),
        BufferBinding::buffer(3, y.clone()),
    ];
    let raw_writes = vec![
        BufferBinding::buffer(0, w_i8.clone()),
        BufferBinding::buffer(1, x_i8.clone()),
        BufferBinding::buffer(2, y.clone()),
    ];
    // (name, writes, n_block, m_block, swizzle, k, n). `swizzle` transposes the
    // dispatch to [M/m_block, N/n_block] for the M-fast-varying L2 lever; (k, n)
    // give the shape so flops + grid are computed per case.
    type Case = (
        &'static str,
        Vec<BufferBinding>,
        u32,
        u32,
        bool,
        usize,
        usize,
    );
    let cases: Vec<Case> = vec![
        (
            "gemm_q4_0_k5376_n21504",
            f16_writes.clone(),
            64,
            64,
            false,
            5376,
            21504,
        ), // f16, 4×4 tiles
        (
            "gemm_q4_0_swz_k5376_n21504",
            f16_writes.clone(),
            64,
            64,
            true,
            5376,
            21504,
        ), // f16 4×4 + L2 swizzle
        (
            "gemm_q4_0_m2_k5376_n21504",
            f16_writes.clone(),
            64,
            32,
            false,
            5376,
            21504,
        ), // f16 2×4 (occupancy lever)
        (
            "gemm_q4_0_m1_k5376_n21504",
            f16_writes,
            64,
            16,
            false,
            5376,
            21504,
        ), // f16 1×4 (occupancy lever)
        (
            "gemm_q4_0_i8_k5376_n21504",
            i8_writes.clone(),
            64,
            32,
            false,
            5376,
            21504,
        ), // int8 MMQ, 2×4 tiles
        (
            "gemm_q4_0_i8_t22_k5376_n21504",
            i8_writes.clone(),
            32,
            32,
            false,
            5376,
            21504,
        ), // 2×2 tiles
        (
            "gemm_q4_0_i8_swz_t22_k5376_n21504",
            i8_writes.clone(),
            32,
            32,
            true,
            5376,
            21504,
        ), // 2×2 + L2 swizzle
        // tall-thin tile sweep (cache-blocking: more M-rows per weight load)
        (
            "gemm_q4_0_i8_swz_m4n1_k5376_n21504",
            i8_writes.clone(),
            16,
            64,
            true,
            5376,
            21504,
        ), // 4×1
        (
            "gemm_q4_0_i8_swz_m8n1_k5376_n21504",
            i8_writes.clone(),
            16,
            128,
            true,
            5376,
            21504,
        ), // 8×1
        (
            "gemm_q4_0_i8_swz_m4n2_k5376_n21504",
            i8_writes.clone(),
            32,
            64,
            true,
            5376,
            21504,
        ), // 4×2
        // down shape (large-K): current 2×2-s1 baseline vs 4×1 (both stage settings)
        (
            "gemm_q4_0_i8_swz_t22_k21504_n5376",
            i8_writes.clone(),
            32,
            32,
            true,
            21504,
            5376,
        ), // down 2×2 s1 (deployed)
        (
            "gemm_q4_0_i8_swz_m4n1_k21504_n5376",
            i8_writes.clone(),
            16,
            64,
            true,
            21504,
            5376,
        ), // down 4×1 s2
        (
            "gemm_q4_0_i8_swz_m4n1_s1_k21504_n5376",
            i8_writes.clone(),
            16,
            64,
            true,
            21504,
            5376,
        ), // down 4×1 s1
        (
            "gemm_q4_0_i8_t12_k5376_n21504",
            i8_writes.clone(),
            32,
            16,
            false,
            5376,
            21504,
        ), // 1×2 tiles
        (
            "gemm_q4_0_i8_t44_k5376_n21504",
            i8_writes.clone(),
            64,
            64,
            false,
            5376,
            21504,
        ), // 4×4 tiles (f16-equivalent)
        (
            "gemm_q4_0_i8_raw_k5376_n21504",
            raw_writes,
            64,
            32,
            false,
            5376,
            21504,
        ), // int8 MMA ceiling, no rescale
        // O-gemm STAGE_BUFS A/B (both swizzled 2×2): single- vs double-buffer `stage`.
        (
            "gemm_q4_0_i8_swz_t22_k8192_n5376",
            i8_writes.clone(),
            32,
            32,
            true,
            8192,
            5376,
        ), // O sliding, STAGE_BUFS=1
        (
            "gemm_q4_0_i8_swz_s2_t22_k8192_n5376",
            i8_writes.clone(),
            32,
            32,
            true,
            8192,
            5376,
        ), // O sliding, STAGE_BUFS=2
        (
            "gemm_q4_0_i8_swz_t22_k16384_n5376",
            i8_writes.clone(),
            32,
            32,
            true,
            16384,
            5376,
        ), // O global, STAGE_BUFS=1
        (
            "gemm_q4_0_i8_swz_s2_t22_k16384_n5376",
            i8_writes.clone(),
            32,
            32,
            true,
            16384,
            5376,
        ), // O global, STAGE_BUFS=2
        // attention 4×1 cache-blocking A/B vs deployed 2×2 (Q/KV/O × sliding/global).
        (
            "gemm_q4_0_i8_swz_t22_k5376_n8192",
            i8_writes.clone(),
            32,
            32,
            true,
            5376,
            8192,
        ), // Q sl 2×2
        (
            "gemm_q4_0_i8_swz_m4n1_k5376_n8192",
            i8_writes.clone(),
            16,
            64,
            true,
            5376,
            8192,
        ), // Q sl 4×1
        (
            "gemm_q4_0_i8_swz_t22_k5376_n16384",
            i8_writes.clone(),
            32,
            32,
            true,
            5376,
            16384,
        ), // Q gl 2×2
        (
            "gemm_q4_0_i8_swz_m4n1_k5376_n16384",
            i8_writes.clone(),
            16,
            64,
            true,
            5376,
            16384,
        ), // Q gl 4×1
        (
            "gemm_q4_0_i8_swz_t22_k5376_n4096",
            i8_writes.clone(),
            32,
            32,
            true,
            5376,
            4096,
        ), // KV sl 2×2
        (
            "gemm_q4_0_i8_swz_m4n1_k5376_n4096",
            i8_writes.clone(),
            16,
            64,
            true,
            5376,
            4096,
        ), // KV sl 4×1
        (
            "gemm_q4_0_i8_swz_t22_k5376_n2048",
            i8_writes.clone(),
            32,
            32,
            true,
            5376,
            2048,
        ), // KV gl 2×2
        (
            "gemm_q4_0_i8_swz_m4n1_k5376_n2048",
            i8_writes.clone(),
            16,
            64,
            true,
            5376,
            2048,
        ), // KV gl 4×1
        (
            "gemm_q4_0_i8_swz_m4n1_k8192_n5376",
            i8_writes.clone(),
            16,
            64,
            true,
            8192,
            5376,
        ), // O sl 4×1
        (
            "gemm_q4_0_i8_swz_m4n1_k16384_n5376",
            i8_writes.clone(),
            16,
            64,
            true,
            16384,
            5376,
        ), // O gl 4×1 s2
        (
            "gemm_q4_0_i8_swz_m4n1_s1_k16384_n5376",
            i8_writes.clone(),
            16,
            64,
            true,
            16384,
            5376,
        ), // O gl 4×1 s1
        // Multi-wave occupancy GEMM: one 16×16 tile/wave, BM_TILES·BN_TILES waves
        // sharing the LDS weight strip — high occupancy at reuse=BM. A/B vs the
        // deployed 4×1 on the FFN up (k5376) + down (k21504) shapes. m_block=BM,
        // n_block=BN; these are SWIZZLE=1 → dispatch transposed [m/BM, n/BN].
        (
            "gemm_q4_0_i8_mw_b41_k5376_n21504",
            i8_writes.clone(),
            16,
            64,
            true,
            5376,
            21504,
        ), // reuse 64, 4 waves (headline)
        (
            "gemm_q4_0_i8_mw_b22_k5376_n21504",
            i8_writes.clone(),
            32,
            32,
            true,
            5376,
            21504,
        ), // reuse 32, 4 waves, wider N
        (
            "gemm_q4_0_i8_mw_b42_k5376_n21504",
            i8_writes.clone(),
            32,
            64,
            true,
            5376,
            21504,
        ), // reuse 64, 8 waves
        (
            "gemm_q4_0_i8_mw_b81_k5376_n21504",
            i8_writes.clone(),
            16,
            128,
            true,
            5376,
            21504,
        ), // reuse 128, 8 waves
        (
            "gemm_q4_0_i8_mw_b41_k21504_n5376",
            i8_writes.clone(),
            16,
            64,
            true,
            21504,
            5376,
        ), // down: reuse 64, 4 waves
        (
            "gemm_q4_0_i8_mw_b22_k21504_n5376",
            i8_writes.clone(),
            32,
            32,
            true,
            21504,
            5376,
        ), // down: reuse 32, 4 waves
        (
            "gemm_q4_0_i8_mw_b42_k21504_n5376",
            i8_writes.clone(),
            32,
            64,
            true,
            21504,
            5376,
        ), // down: reuse 64, 8 waves
        (
            "gemm_q4_0_i8_mw_b81_k21504_n5376",
            i8_writes.clone(),
            16,
            128,
            true,
            21504,
            5376,
        ), // down: reuse 128, 8 waves
        // Half-occupancy A/B vs b41 (same 64×16 block + reuse 64): 2 waves × RM=2.
        (
            "gemm_q4_0_i8_mw_r2_k5376_n21504",
            i8_writes.clone(),
            16,
            64,
            true,
            5376,
            21504,
        ), // up: 2 waves, RM=2 (half occ)
        (
            "gemm_q4_0_i8_mw_r2_k21504_n5376",
            i8_writes,
            16,
            64,
            true,
            21504,
            5376,
        ), // down: 2 waves, RM=2 (half occ)
    ];

    let mut group = c.benchmark_group("mmq_vs_f16");
    for (name, writes, n_block, m_block, swizzle, k, n) in cases {
        let kernel = ctx.load_kernel(name).expect(name);
        let flops = 2.0 * M as f64 * n as f64 * k as f64 * DISPATCHES as f64;
        let (n, m) = (n as u32, M as u32);
        // SWIZZLE=1 kernels read m-block from wg.x, n-block from wg.y → transpose.
        let grid = if swizzle {
            [m / m_block, n / n_block, 1]
        } else {
            [n / n_block, m / m_block, 1]
        };
        // `dispatch_overlapping` (no inter-dispatch barrier) reproduces the
        // historical vulkano behavior for these coopStore kernels.
        let graph = ctx
            .record_graph(|rec| {
                for _ in 0..DISPATCHES {
                    rec.dispatch_overlapping(&kernel, writes.clone(), None::<u32>, grid)?;
                }
                Ok(())
            })
            .expect("record");

        group.bench_function(name, |b| {
            b.iter_custom(|iters| {
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    ctx.submit_blocking(&graph).unwrap();
                }
                let elapsed = start.elapsed();
                let tflops = flops * iters as f64 / elapsed.as_secs_f64() / 1e12;
                eprintln!("  -> {name}: {tflops:.2} TFLOPS");
                elapsed
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
