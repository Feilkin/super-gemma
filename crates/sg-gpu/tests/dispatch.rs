//! End-to-end dispatch on a real GPU: context creation, buffer round-trip,
//! push constants. Skips when no capable Vulkan device is present (Tier 1).

use sg_gpu::GpuContext;
use vulkano::buffer::BufferUsage;
use vulkano::descriptor_set::WriteDescriptorSet;

fn ctx() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(ctx) => {
            eprintln!("device: {}", ctx.device_name());
            Some(ctx)
        }
        Err(e) => {
            eprintln!("skipping: no usable GPU ({e})");
            None
        }
    }
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct StubPush {
    add: u32,
}

#[test]
fn stub_kernel_increments_a_buffer() {
    let Some(ctx) = ctx() else { return };
    let kernel = ctx.load_kernel("stub").expect("load stub");

    let n = 1000u64; // not a workgroup multiple: exercises the bounds check
    let buf = ctx
        .buffer_from_iter(0..n as u32, BufferUsage::STORAGE_BUFFER)
        .expect("buffer");

    ctx.dispatch_blocking(
        &kernel,
        vec![WriteDescriptorSet::buffer(0, buf.clone())],
        Some(StubPush { add: 7 }),
        kernel.groups_for(n),
    )
    .expect("dispatch");

    let read = buf.read().expect("map");
    for (i, &v) in read.iter().enumerate() {
        assert_eq!(v, i as u32 + 7, "element {i}");
    }
}

#[test]
fn context_reports_target_capabilities() {
    let Some(ctx) = ctx() else { return };
    // Informational on dev machines; the target box must have coopmat.
    eprintln!(
        "subgroup_size={} cooperative_matrix={}",
        ctx.subgroup_size, ctx.cooperative_matrix
    );
    assert!(ctx.subgroup_size.is_power_of_two());
}
