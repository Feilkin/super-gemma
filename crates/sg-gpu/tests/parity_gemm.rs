//! Parity: both prefill GEMM flavors (cooperative-matrix `gemm_q4_0` and
//! subgroup-tiled `gemm_st_q4_0`) vs the f64 CPU reference on all eight
//! matmul shapes. Skips without a GPU; the coopmat flavor additionally
//! skips without VK_KHR_cooperative_matrix.

mod reference;

use reference::{Rng, assert_close, from_f16_bits, through_f16, to_f16_bits};
use sg_gguf::q4_0::{BLOCK_Q4_0_SIZE, QK4_0, blocks_from_bytes};
use sg_gpu::GpuContext;
use sg_gpu::{BufferBinding, BufferUsage};

fn ctx() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("skipping: no usable GPU ({e})");
            None
        }
    }
}

/// (kernel prefix, M block, N block per workgroup). Must match build.rs.
fn flavors(ctx: &GpuContext) -> Vec<(&'static str, usize, usize)> {
    let mut v = vec![("gemm_st_q4_0", 64, 64)];
    if ctx.cooperative_matrix {
        v.push(("gemm_q4_0", M_BLOCK, N_BLOCK));
    } else {
        eprintln!("note: skipping coopmat flavor (extension absent)");
    }
    v
}

fn random_q4_0(rng: &mut Rng, n_blocks: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_blocks * BLOCK_Q4_0_SIZE);
    for _ in 0..n_blocks {
        let d = half::f16::from_f32(rng.f32() * 0.05);
        out.extend_from_slice(&d.to_bits().to_le_bytes());
        for _ in 0..16 {
            out.push((rng.next_u64() & 0xFF) as u8);
        }
    }
    out
}

/// Dequantize one W row, rounding through f16 exactly like the kernel's
/// workgroup staging (the coopmat B operand is f16).
fn dequant_row(weights: &[u8], k: usize, row: usize) -> Vec<f32> {
    let row_bytes = k / QK4_0 * BLOCK_Q4_0_SIZE;
    let blocks = blocks_from_bytes(&weights[row * row_bytes..][..row_bytes]).unwrap();
    let mut out = vec![0.0f32; k];
    let mut tmp = [0.0f32; QK4_0];
    for (blk, block) in blocks.iter().enumerate() {
        block.dequantize(&mut tmp);
        for (o, &v) in out[blk * QK4_0..(blk + 1) * QK4_0].iter_mut().zip(&tmp) {
            *o = half::f16::from_f32(v).to_f32();
        }
    }
    out
}

/// Output block per workgroup (M_TILES=2 × N_TILES=4 of 16×16 tiles; must
/// match the gemm variant defines in build.rs).
const M_BLOCK: usize = 64;
const N_BLOCK: usize = 64;

const SHAPES: &[(usize, usize)] = &[
    (5376, 8192),
    (5376, 4096),
    (8192, 5376),
    (5376, 16384),
    (5376, 2048),
    (16384, 5376),
    (5376, 21504),
    (21504, 5376),
];

#[test]
fn gemm_q4_0_matches_reference_on_all_shapes() {
    let Some(ctx) = ctx() else { return };
    let mut rng = Rng::new(0x6E33);
    let m = 64usize; // prefill rows; multiple of M_BLOCK per the kernel contract

    for (prefix, m_block, n_block) in flavors(&ctx) {
        for &(k, n) in SHAPES {
            let variant = format!("{prefix}_k{k}_n{n}");
            let kernel = ctx.load_kernel(&variant).expect(&variant);

            let weights = random_q4_0(&mut rng, n * k / QK4_0);
            let x = through_f16(&rng.f32_vec(m * k));

            let w_buf = ctx
                .buffer_from_iter(
                    weights
                        .chunks_exact(4)
                        .map(|c| u32::from_le_bytes(c.try_into().unwrap())),
                    BufferUsage::STORAGE_BUFFER,
                )
                .unwrap();
            let x_buf = ctx
                .buffer_from_iter(to_f16_bits(&x), BufferUsage::STORAGE_BUFFER)
                .unwrap();
            let y_buf = ctx
                .new_buffer::<u16>((m * n) as u64, BufferUsage::STORAGE_BUFFER)
                .unwrap();

            ctx.dispatch_blocking(
                &kernel,
                vec![
                    BufferBinding::buffer(0, w_buf),
                    BufferBinding::buffer(1, x_buf),
                    BufferBinding::buffer(2, y_buf.clone()),
                ],
                None::<u32>,
                [(n / n_block) as u32, (m / m_block) as u32, 1],
            )
            .expect(&variant);
            let got_all = from_f16_bits(&y_buf.read().unwrap());

            // Reference on a sample of output columns (full f64 GEMM on the big
            // shapes adds minutes, not signal), every row.
            let cols: Vec<usize> = (0..n).step_by((n / 61).max(1)).chain([n - 1]).collect();
            let mut got = Vec::new();
            let mut want = Vec::new();
            for &col in &cols {
                let w_row = dequant_row(&weights, k, col);
                for row in 0..m {
                    let acc: f64 = (0..k)
                        .map(|j| w_row[j] as f64 * x[row * k + j] as f64)
                        .sum();
                    want.push(acc as f32);
                    got.push(got_all[row * n + col]);
                }
            }
            assert_close(&got, &want, 2e-2, 2e-2, &format!("{variant} (M={m})"));
        }
    }
}

/// The L2-reuse workgroup swizzle (SWIZZLE=1: M-blocks fast-varying, dispatch
/// transposed) must produce BIT-IDENTICAL output to the plain kernel — it only
/// reorders which workgroup computes which tile. m=128 (2 M-blocks) exercises
/// the M-block ordering.
#[test]
fn gemm_q4_0_swizzle_matches_plain() {
    let Some(ctx) = ctx() else { return };
    if !ctx.cooperative_matrix {
        return;
    }
    let mut rng = Rng::new(0x5712);
    let (m, k, n) = (128usize, 5376usize, 21504usize);

    let weights = random_q4_0(&mut rng, n * k / QK4_0);
    let x = through_f16(&rng.f32_vec(m * k));
    let w_buf = ctx
        .buffer_from_iter(
            weights
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap())),
            BufferUsage::STORAGE_BUFFER,
        )
        .unwrap();
    let x_buf = ctx
        .buffer_from_iter(to_f16_bits(&x), BufferUsage::STORAGE_BUFFER)
        .unwrap();

    let run = |variant: &str, grid: [u32; 3]| -> Vec<u16> {
        let kernel = ctx.load_kernel(variant).expect(variant);
        let y = ctx
            .new_buffer::<u16>((m * n) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        ctx.dispatch_blocking(
            &kernel,
            vec![
                BufferBinding::buffer(0, w_buf.clone()),
                BufferBinding::buffer(1, x_buf.clone()),
                BufferBinding::buffer(2, y.clone()),
            ],
            None::<u32>,
            grid,
        )
        .expect(variant);
        y.read().unwrap().to_vec()
    };

    // Plain: dispatch [N/strip, M/tile]. Swizzled: transposed [M/tile, N/strip].
    let plain = run(
        "gemm_q4_0_k5376_n21504",
        [(n / 64) as u32, (m / 64) as u32, 1],
    );
    let swz = run(
        "gemm_q4_0_swz_k5376_n21504",
        [(m / 64) as u32, (n / 64) as u32, 1],
    );
    assert_eq!(plain, swz, "swizzled gemm output differs from plain");
}

/// Plan 02/06: bit-identical across runs.
#[test]
fn gemm_is_bit_deterministic() {
    let Some(ctx) = ctx() else { return };
    let mut rng = Rng::new(0xD1);
    let (k, n, m) = (5376usize, 4096usize, 64usize);
    let kernel = ctx.load_kernel("gemm_q4_0_k5376_n4096").unwrap();
    let weights = random_q4_0(&mut rng, n * k / QK4_0);
    let x = to_f16_bits(&rng.f32_vec(m * k));

    let mut outs = Vec::new();
    for _ in 0..2 {
        let w_buf = ctx
            .buffer_from_iter(
                weights
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap())),
                BufferUsage::STORAGE_BUFFER,
            )
            .unwrap();
        let x_buf = ctx
            .buffer_from_iter(x.iter().copied(), BufferUsage::STORAGE_BUFFER)
            .unwrap();
        let y_buf = ctx
            .new_buffer::<u16>((m * n) as u64, BufferUsage::STORAGE_BUFFER)
            .unwrap();
        ctx.dispatch_blocking(
            &kernel,
            vec![
                BufferBinding::buffer(0, w_buf),
                BufferBinding::buffer(1, x_buf),
                BufferBinding::buffer(2, y_buf.clone()),
            ],
            None::<u32>,
            [(n / N_BLOCK) as u32, (m / M_BLOCK) as u32, 1],
        )
        .unwrap();
        outs.push(y_buf.read().unwrap().to_vec());
    }
    assert_eq!(outs[0], outs[1]);
}
