//! Toolchain smoke test for the int8-coopmat naga fork: dispatch the
//! `coop_i8_smoke` kernel (one 16×16×16 signed-int8 MMA, C in i32) and check
//! the result against a CPU reference. Proves the patched naga emits SPIR-V
//! the RADV driver accepts AND computes correctly (signedness, packing).
//! Skips without a GPU. See docs/naga-int8-coopmat-patch.md.

use sg_gpu::GpuContext;
use vulkano::buffer::BufferUsage;
use vulkano::descriptor_set::WriteDescriptorSet;

const N: usize = 16;

#[test]
fn coop_i8_smoke_matches_reference() {
    let Some(ctx) = (match GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping: no usable GPU ({e})");
            None
        }
    }) else {
        return;
    };
    let kernel = ctx.load_kernel("coop_i8_smoke").unwrap();

    // Deterministic small signed int8 inputs, row-major [16×16]. Values span
    // negatives so the signed-components flags actually matter.
    let a: Vec<i8> = (0..N * N)
        .map(|n| (((n / N) + (n % N)) as i32 % 7 - 3) as i8)
        .collect();
    let b: Vec<i8> = (0..N * N)
        .map(|n| (((n / N) * (n % N)) as i32 % 11 - 5) as i8)
        .collect();

    // Reference: C[i][j] = Σ_k A[i][k]·B[k][j], i32.
    let mut want = vec![0i32; N * N];
    for i in 0..N {
        for j in 0..N {
            let mut acc = 0i32;
            for k in 0..N {
                acc += a[i * N + k] as i32 * b[k * N + j] as i32;
            }
            want[i * N + j] = acc;
        }
    }

    let a_buf = ctx
        .buffer_from_iter(a.iter().copied(), BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let b_buf = ctx
        .buffer_from_iter(b.iter().copied(), BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let out_buf = ctx
        .new_buffer::<i32>((N * N) as u64, BufferUsage::STORAGE_BUFFER)
        .unwrap();

    ctx.dispatch_blocking(
        &kernel,
        vec![
            WriteDescriptorSet::buffer(0, a_buf),
            WriteDescriptorSet::buffer(1, b_buf),
            WriteDescriptorSet::buffer(2, out_buf.clone()),
        ],
        None::<u32>,
        [1, 1, 1],
    )
    .unwrap();

    let got = out_buf.read().unwrap();
    assert_eq!(&got[..], &want[..], "int8 coopmat MMA mismatch");
}
