//! 0-stride coopLoad broadcast from WORKGROUP (LDS) memory — the attention
//! rescale needs it (the scale vectors are workgroup-local), and storage-buffer
//! broadcast (coop_bcast_smoke) does not prove LDS behaves the same.

use sg_gpu::GpuContext;
use sg_gpu::{BufferBinding, BufferUsage};

#[test]
fn coop_bcast_lds_smoke_matches_reference() {
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
    let kernel = ctx.load_kernel("coop_bcast_lds_smoke").unwrap();

    let pvec: Vec<f32> = (0..16).map(|q| (q + 1) as f32).collect();
    let vvec: Vec<f32> = (0..16).map(|c| 10.0 * (c + 1) as f32).collect();
    let want: Vec<f32> = (0..16 * 16).map(|i| pvec[i / 16] * vvec[i % 16]).collect();

    let p_buf = ctx
        .buffer_from_iter(pvec.iter().copied(), BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let v_buf = ctx
        .buffer_from_iter(vvec.iter().copied(), BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let out_buf = ctx
        .new_buffer::<f32>(256, BufferUsage::STORAGE_BUFFER)
        .unwrap();

    ctx.dispatch_blocking(
        &kernel,
        vec![
            BufferBinding::buffer(0, p_buf),
            BufferBinding::buffer(1, v_buf),
            BufferBinding::buffer(2, out_buf.clone()),
        ],
        None::<u32>,
        [1, 1, 1],
    )
    .unwrap();

    let got = out_buf.read().unwrap();
    eprintln!(
        "out[0][0..2]={:?} out[1][0..2]={:?}  want[0][0..2]={:?} want[1][0..2]={:?}",
        &got[0..2],
        &got[16..18],
        &want[0..2],
        &want[16..18],
    );
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-3,
            "element {i} (q={}, c={}): got {g} want {w}",
            i / 16,
            i % 16
        );
    }
}
