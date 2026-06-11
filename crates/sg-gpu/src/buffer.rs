//! Buffer creation on unified memory: everything is HOST_VISIBLE |
//! DEVICE_LOCAL on the target (plan 02), so the CPU writes weights/uniforms
//! and reads logits with no staging copies.

use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};

use crate::{GpuContext, GpuError};

impl GpuContext {
    /// Zero-initialized device-local, host-mappable slice buffer.
    pub fn new_buffer<T: BufferContents>(
        &self,
        len: u64,
        usage: BufferUsage,
    ) -> Result<Subbuffer<[T]>, GpuError> {
        Buffer::new_slice(
            self.memory_allocator().clone(),
            BufferCreateInfo {
                usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            len,
        )
        .map_err(GpuError::validated)
    }

    /// Device-local, host-mappable buffer initialized from an iterator.
    pub fn buffer_from_iter<T, I>(
        &self,
        iter: I,
        usage: BufferUsage,
    ) -> Result<Subbuffer<[T]>, GpuError>
    where
        T: BufferContents,
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        Buffer::from_iter(
            self.memory_allocator().clone(),
            BufferCreateInfo {
                usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            iter,
        )
        .map_err(GpuError::validated)
    }
}
