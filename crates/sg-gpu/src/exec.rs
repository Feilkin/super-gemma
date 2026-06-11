//! One-shot blocking dispatch, used by kernel parity tests and microbenches.
//! The production decode/prefill path uses pre-recorded command graphs
//! instead (plan 02 §submission model; M2 step 8).

use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::sync::GpuFuture;

use crate::{GpuContext, GpuError, Kernel};

impl GpuContext {
    /// Bind `writes` as descriptor set 0, push `push` (if any), dispatch
    /// `groups`, submit, and wait for completion.
    pub fn dispatch_blocking<P: vulkano::buffer::BufferContents>(
        &self,
        kernel: &Kernel,
        writes: Vec<WriteDescriptorSet>,
        push: Option<P>,
        groups: [u32; 3],
    ) -> Result<(), GpuError> {
        let layout = kernel.layout().clone();
        let set = DescriptorSet::new(
            self.descriptor_set_allocator().clone(),
            layout.set_layouts()[0].clone(),
            writes,
            [],
        )
        .map_err(GpuError::validated)?;

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator().clone(),
            self.queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(GpuError::validated)?;
        builder
            .bind_pipeline_compute(kernel.pipeline().clone())
            .map_err(|e| GpuError::Pipeline(e.to_string()))?
            .bind_descriptor_sets(PipelineBindPoint::Compute, layout.clone(), 0, set)
            .map_err(|e| GpuError::Pipeline(e.to_string()))?;
        if let Some(push) = push {
            builder
                .push_constants(layout, 0, push)
                .map_err(|e| GpuError::Pipeline(e.to_string()))?;
        }
        // SAFETY: dispatch bounds are the caller's contract with the kernel;
        // all our kernels bounds-check against arrayLength or baked sizes.
        unsafe { builder.dispatch(groups) }.map_err(|e| GpuError::Pipeline(e.to_string()))?;

        let cb = builder.build().map_err(GpuError::validated)?;
        cb.execute(self.queue().clone())
            .map_err(|e| GpuError::Pipeline(e.to_string()))?
            .then_signal_fence_and_flush()
            .map_err(GpuError::validated)?
            .wait(None)
            .map_err(GpuError::validated)?;
        Ok(())
    }
}
