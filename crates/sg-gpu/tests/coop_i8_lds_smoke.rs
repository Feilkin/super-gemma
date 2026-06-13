//! Probe test: coopLoad an int8 cooperative matrix from WORKGROUP (LDS) memory.
//! The Q4_0-reading int8 GEMM wants to unpack nibbles → i8 in LDS → coopLoad;
//! 8-bit *shared* memory is a different path from the proven 8-bit storage-buffer
//! loads, so confirm RADV both ACCEPTS the pipeline and computes it correctly.
//! The kernel loads src (16×16 i8) into LDS, then A = coopLoadT(lds) (transpose)
//! and B = coopLoad(lds) (row-major), C = A·B (i32). Skips without a GPU.

use sg_gpu::GpuContext;
use vulkano::buffer::BufferUsage;
use vulkano::descriptor_set::WriteDescriptorSet;

const N: usize = 16;

#[test]
fn coop_i8_lds_smoke_matches_reference() {
    let Some(ctx) = (match GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping: no usable GPU ({e})");
            None
        }
    }) else {
        return;
    };
    let kernel = ctx.load_kernel("coop_i8_lds_smoke").unwrap();

    let src: Vec<i8> = (0..N * N)
        .map(|n| (((n / N) + (n % N)) as i32 % 7 - 3) as i8)
        .collect();

    // The kernel stages `lds = i8(src - 1)`, then A·B with A = ldsᵀ (coopLoadT),
    // B = lds (coopLoad), so C[i][j] = Σ_k lds[k][i]·lds[k][j].
    let lds: Vec<i8> = src.iter().map(|&v| (v as i32 - 1) as i8).collect();
    let mut want = vec![0i32; N * N];
    for i in 0..N {
        for j in 0..N {
            let mut acc = 0i32;
            for k in 0..N {
                acc += lds[k * N + i] as i32 * lds[k * N + j] as i32;
            }
            want[i * N + j] = acc;
        }
    }

    let src_buf = ctx
        .buffer_from_iter(src.iter().copied(), BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let out_buf = ctx
        .new_buffer::<i32>((N * N) as u64, BufferUsage::STORAGE_BUFFER)
        .unwrap();

    ctx.dispatch_blocking(
        &kernel,
        vec![
            WriteDescriptorSet::buffer(0, src_buf),
            WriteDescriptorSet::buffer(1, out_buf.clone()),
        ],
        None::<u32>,
        [1, 1, 1],
    )
    .unwrap();

    let got = out_buf.read().unwrap();
    assert_eq!(&got[..], &want[..], "int8 coopmat MMA from LDS mismatch");
}
