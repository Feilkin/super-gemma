//! Buffer creation on unified memory: everything is HOST_VISIBLE |
//! DEVICE_LOCAL on the target (plan 02), so the CPU writes weights/uniforms
//! and reads logits with no staging copies. Each buffer gets a dedicated
//! `vkAllocateMemory` that is mapped once at creation and stays mapped for the
//! buffer's life; `read`/`write` hand out RAII guards over that mapping (and
//! flush/invalidate when the memory type is not HOST_COHERENT).

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use ash::vk;
use bytemuck::Pod;

use crate::context::DeviceCtx;
use crate::{GpuContext, GpuError};

/// Buffer usage flags. A thin wrapper over `vk::BufferUsageFlags` exposing the
/// constants this crate uses; the name is kept (vs the old vulkano re-export)
/// so consumer `use sg_gpu::BufferUsage` imports are unchanged.
#[derive(Clone, Copy, Debug)]
pub struct BufferUsage(pub(crate) vk::BufferUsageFlags);

impl BufferUsage {
    pub const STORAGE_BUFFER: Self = Self(vk::BufferUsageFlags::STORAGE_BUFFER);
    pub const TRANSFER_SRC: Self = Self(vk::BufferUsageFlags::TRANSFER_SRC);
    pub const TRANSFER_DST: Self = Self(vk::BufferUsageFlags::TRANSFER_DST);

    pub const fn empty() -> Self {
        Self(vk::BufferUsageFlags::empty())
    }
}

impl std::ops::BitOr for BufferUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// The buffer's Vulkan objects + its persistent mapping. Shared (via `Arc`) by
/// every clone of a [`Buffer`] and by any [`BufferBinding`] recorded against
/// it, so the allocation outlives the command graphs that reference it.
pub(crate) struct BufferInner {
    ctx: Arc<DeviceCtx>,
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    /// Persistent host mapping of `memory` (valid for the buffer's whole life).
    mapped: *mut u8,
    /// Byte length of the allocation.
    size: vk::DeviceSize,
    coherent: bool,
}

// SAFETY: `mapped` is a raw pointer into host-visible device memory. Access is
// externally synchronized — the dedicated GPU thread (plan 00) owns these, and
// `read`/`write` hand out borrow-checked guards. The handles are plain device
// objects, valid to move between threads.
unsafe impl Send for BufferInner {}
// SAFETY: see the `Send` impl above — the same external synchronization makes
// shared `&BufferInner` access across threads sound.
unsafe impl Sync for BufferInner {}

impl Drop for BufferInner {
    fn drop(&mut self) {
        // SAFETY: this runs when the last Arc is gone, so no guard borrows the
        // mapping and no in-flight submission references the buffer (graphs that
        // bound it hold an Arc, keeping this alive until they too are dropped).
        unsafe {
            self.ctx.device.unmap_memory(self.memory);
            self.ctx.device.destroy_buffer(self.buffer, None);
            self.ctx.device.free_memory(self.memory, None);
        }
    }
}

/// A device-local, host-mappable slice of `T` on unified memory. Clones share
/// the underlying allocation (cheap `Arc` bump), matching the old
/// `Subbuffer<[T]>` currency this crate exposed.
pub struct Buffer<T: Pod> {
    pub(crate) inner: Arc<BufferInner>,
    /// Element offset of this (possibly sub-sliced) view into the allocation.
    offset: u64,
    /// Element count of this view.
    len: u64,
    _marker: PhantomData<T>,
}

impl<T: Pod> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            offset: self.offset,
            len: self.len,
            _marker: PhantomData,
        }
    }
}

impl<T: Pod> Buffer<T> {
    /// Number of `T` elements in this view.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// A sub-range view `[range.start, range.end)` (in elements), sharing the
    /// underlying allocation — the `Subbuffer::slice` this crate relied on.
    pub fn slice(self, range: std::ops::Range<u64>) -> Self {
        assert!(
            range.start <= range.end && range.end <= self.len,
            "slice {range:?} out of bounds (len {})",
            self.len
        );
        Self {
            inner: self.inner,
            offset: self.offset + range.start,
            len: range.end - range.start,
            _marker: PhantomData,
        }
    }

    fn byte_offset(&self) -> u64 {
        self.offset * std::mem::size_of::<T>() as u64
    }

    /// Borrow the contents for reading. On non-coherent memory this invalidates
    /// the mapped range first so the host sees the device's latest writes.
    pub fn read(&self) -> Result<ReadGuard<'_, T>, GpuError> {
        if !self.inner.coherent {
            self.inner.invalidate()?;
        }
        // SAFETY: the mapping is valid for the buffer's life and `len` elements
        // of `T` at `byte_offset` fit in `size`; the guard's lifetime ties the
        // slice to `self`.
        let base = unsafe { self.inner.mapped.add(self.byte_offset() as usize) };
        // SAFETY: `[base, base + len*size_of::<T>())` lies within the mapping.
        let slice = unsafe { std::slice::from_raw_parts(base as *const T, self.len as usize) };
        Ok(ReadGuard {
            slice,
            _buf: &self.inner,
        })
    }

    /// Borrow the contents for writing. On non-coherent memory the range is
    /// flushed when the guard drops so the device sees the host's writes.
    pub fn write(&self) -> Result<WriteGuard<'_, T>, GpuError> {
        // SAFETY: as `read`, but a unique mapped slice — the borrow checker
        // enforces no overlapping guard exists (callers hold `&self`).
        let base = unsafe { self.inner.mapped.add(self.byte_offset() as usize) };
        // SAFETY: `[base, base + len*size_of::<T>())` lies within the mapping;
        // `&self` plus the borrow checker preclude an overlapping live guard.
        let slice =
            unsafe { std::slice::from_raw_parts_mut(base as *mut T, self.len as usize) };
        Ok(WriteGuard {
            slice,
            buf: &self.inner,
        })
    }
}

impl BufferInner {
    fn flush(&self) -> Result<(), GpuError> {
        let range = vk::MappedMemoryRange::default()
            .memory(self.memory)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        // SAFETY: valid memory handle; whole-range flush.
        unsafe { self.ctx.device.flush_mapped_memory_ranges(&[range]) }
            .map_err(|e| GpuError::Vk(format!("flush_mapped_memory_ranges: {e}")))
    }

    fn invalidate(&self) -> Result<(), GpuError> {
        let range = vk::MappedMemoryRange::default()
            .memory(self.memory)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        // SAFETY: valid memory handle; whole-range invalidate.
        unsafe { self.ctx.device.invalidate_mapped_memory_ranges(&[range]) }
            .map_err(|e| GpuError::Vk(format!("invalidate_mapped_memory_ranges: {e}")))
    }
}

/// Read borrow of a [`Buffer`]'s contents.
pub struct ReadGuard<'a, T> {
    slice: &'a [T],
    _buf: &'a BufferInner,
}

impl<T> Deref for ReadGuard<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.slice
    }
}

/// Write borrow of a [`Buffer`]'s contents; flushes on drop when non-coherent.
pub struct WriteGuard<'a, T> {
    slice: &'a mut [T],
    buf: &'a BufferInner,
}

impl<T> Deref for WriteGuard<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.slice
    }
}

impl<T> DerefMut for WriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.slice
    }
}

impl<T> Drop for WriteGuard<'_, T> {
    fn drop(&mut self) {
        if !self.buf.coherent {
            // Best-effort: a failed flush would surface as wrong GPU results,
            // not a recoverable error at drop time.
            let _ = self.buf.flush();
        }
    }
}

/// A buffer bound at a descriptor-set binding for one dispatch (was
/// `WriteDescriptorSet`). Type-erased over `T`; carries an `Arc` to the
/// allocation so a recorded graph keeps it alive.
#[derive(Clone)]
pub struct BufferBinding {
    pub(crate) binding: u32,
    pub(crate) keep: Arc<BufferInner>,
    /// Byte offset + range of the (possibly sliced) view within the allocation.
    pub(crate) offset: vk::DeviceSize,
    pub(crate) range: vk::DeviceSize,
}

impl BufferBinding {
    /// Bind `buf` at descriptor `binding` (set 0). Mirrors the old
    /// `WriteDescriptorSet::buffer(binding, subbuffer)`.
    pub fn buffer<T: Pod>(binding: u32, buf: Buffer<T>) -> Self {
        let elem = std::mem::size_of::<T>() as u64;
        Self {
            binding,
            offset: buf.offset * elem,
            range: (buf.len * elem).max(elem),
            keep: buf.inner,
        }
    }
}

impl GpuContext {
    /// Zero-initialized device-local, host-mappable slice buffer.
    pub fn new_buffer<T: Pod>(
        &self,
        len: u64,
        usage: BufferUsage,
    ) -> Result<Buffer<T>, GpuError> {
        let inner = self.alloc_buffer(len, std::mem::size_of::<T>() as u64, usage)?;
        // Zero the freshly mapped allocation.
        // SAFETY: `mapped` covers `size` bytes (just allocated + mapped).
        unsafe { std::ptr::write_bytes(inner.mapped, 0, inner.size as usize) };
        if !inner.coherent {
            inner.flush()?;
        }
        Ok(Buffer {
            inner: Arc::new(inner),
            offset: 0,
            len,
            _marker: PhantomData,
        })
    }

    /// Device-local, host-mappable buffer initialized from an iterator.
    pub fn buffer_from_iter<T, I>(
        &self,
        iter: I,
        usage: BufferUsage,
    ) -> Result<Buffer<T>, GpuError>
    where
        T: Pod,
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let iter = iter.into_iter();
        let len = iter.len() as u64;
        let inner = self.alloc_buffer(len, std::mem::size_of::<T>() as u64, usage)?;
        // SAFETY: `mapped` holds `len` elements of `T` (size = len * size_of).
        let dst = unsafe {
            std::slice::from_raw_parts_mut(inner.mapped as *mut T, len as usize)
        };
        for (d, s) in dst.iter_mut().zip(iter) {
            *d = s;
        }
        if !inner.coherent {
            inner.flush()?;
        }
        Ok(Buffer {
            inner: Arc::new(inner),
            offset: 0,
            len,
            _marker: PhantomData,
        })
    }

    /// Create + bind + map a dedicated allocation for `len * elem_size` bytes.
    fn alloc_buffer(
        &self,
        len: u64,
        elem_size: u64,
        usage: BufferUsage,
    ) -> Result<BufferInner, GpuError> {
        let ctx = &self.ctx;
        // Vulkan forbids zero-size buffers; round empty buffers up to one
        // element so `len()` still reports 0 while the handle is valid.
        let size = (len * elem_size).max(elem_size.max(1));
        let create = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage.0)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: valid device; create info borrows nothing past the call.
        let buffer = unsafe { ctx.device.create_buffer(&create, None) }
            .map_err(|e| GpuError::Vk(format!("create_buffer: {e}")))?;

        // SAFETY: `buffer` was just created on this device.
        let reqs = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
        let required =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let (type_index, coherent) = ctx
            .find_mem_type(reqs.memory_type_bits, required)
            .or_else(|| ctx.find_mem_type(reqs.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE))
            .ok_or(GpuError::NoDevice("host-visible memory type"))?;

        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(type_index);
        // SAFETY: valid device; size/type satisfy the buffer's requirements.
        let memory = unsafe { ctx.device.allocate_memory(&alloc, None) }.map_err(|e| {
            // SAFETY: destroy the orphan buffer on the alloc failure path.
            unsafe { ctx.device.destroy_buffer(buffer, None) };
            GpuError::Vk(format!("allocate_memory ({} bytes): {e}", reqs.size))
        })?;

        // SAFETY: fresh buffer + memory, neither bound yet.
        if let Err(e) = unsafe { ctx.device.bind_buffer_memory(buffer, memory, 0) } {
            // SAFETY: tear down both objects on the bind failure path.
            unsafe {
                ctx.device.free_memory(memory, None);
                ctx.device.destroy_buffer(buffer, None);
            }
            return Err(GpuError::Vk(format!("bind_buffer_memory: {e}")));
        }

        // SAFETY: host-visible memory, whole range, mapped once for its life.
        let mapped = unsafe {
            ctx.device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }
        .map_err(|e| {
            // SAFETY: tear down on the map failure path.
            unsafe {
                ctx.device.free_memory(memory, None);
                ctx.device.destroy_buffer(buffer, None);
            }
            GpuError::Vk(format!("map_memory: {e}"))
        })? as *mut u8;

        Ok(BufferInner {
            ctx: ctx.clone(),
            buffer,
            memory,
            mapped,
            size,
            coherent,
        })
    }
}
