//! Smoke test for the coopmat-arith naga ops (fork rev 8366d92e): dispatch the
//! `coop_arith_smoke` kernel (`out = sc * f32(di)`, using component-wise
//! `OpFMul` and `OpConvertSToF` on cooperative matrices) and check vs a CPU
//! reference.
//! Proves the patched naga emits SPIR-V the RADV driver accepts AND computes
//! correctly — the toolchain prerequisite for the int8-MMQ in-register rescale
//! (docs/naga-coopmat-arith-patch.md). Skips without a coopmat GPU.

use sg_gpu::GpuContext;
use sg_gpu::{BufferBinding, BufferUsage};

const N: usize = 16 * 16;

#[test]
fn coop_arith_smoke_matches_reference() {
    let Some(ctx) = (match GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping: no usable GPU ({e})");
            None
        }
    }) else {
        return;
    };
    if !ctx.cooperative_matrix {
        eprintln!("skipping: no VK_KHR_cooperative_matrix");
        return;
    }
    let kernel = ctx.load_kernel("coop_arith_smoke").unwrap();

    // Small signed ints (exact in f32) × simple f32 scales (products exact).
    let di: Vec<i32> = (0..N).map(|i| (i as i32 % 7) - 3).collect();
    let sc: Vec<f32> = (0..N).map(|i| (i % 5) as f32 * 0.5).collect();
    let want: Vec<f32> = (0..N).map(|i| sc[i] * di[i] as f32).collect();

    let di_buf = ctx
        .buffer_from_iter(di.iter().copied(), BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let sc_buf = ctx
        .buffer_from_iter(sc.iter().copied(), BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let out_buf = ctx
        .new_buffer::<f32>(N as u64, BufferUsage::STORAGE_BUFFER)
        .unwrap();

    ctx.dispatch_blocking(
        &kernel,
        vec![
            BufferBinding::buffer(0, di_buf),
            BufferBinding::buffer(1, sc_buf),
            BufferBinding::buffer(2, out_buf.clone()),
        ],
        None::<u32>,
        [1, 1, 1],
    )
    .unwrap();

    let got = out_buf.read().unwrap();
    assert_eq!(
        &got[..],
        &want[..],
        "coopmat convert+component-mul mismatch"
    );
}
