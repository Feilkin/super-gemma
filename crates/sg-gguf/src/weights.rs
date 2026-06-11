//! `WeightSource`: load the tensor-data section into GPU-visible memory
//! (plan 01 step 6).
//!
//! The destination is a plain `&mut [u8]` so this crate stays vulkano-free:
//! in M2, `sg-gpu` allocates one big HOST_VISIBLE|DEVICE_LOCAL buffer laid
//! out exactly like the file's data section (tensor offsets preserved) and
//! passes its mapped slice here; one sequential load covers every tensor.
//!
//! Two implementations:
//! - [`MmapCopySource`]: memcpy out of the existing mmap. Portable fallback,
//!   page-cache-warmed cost on repeat loads.
//! - [`DirectSource`] (Linux): `O_DIRECT` chunked sequential `pread` at NVMe
//!   speed (M0 probe: 5.8 GiB/s) without polluting 17 GB of page cache.
//!   Chunks whose 4 KiB alignment phase matches the destination are read
//!   zero-copy into `dst`; head/tail fragments and phase-mismatched layouts
//!   go through one aligned bounce buffer. For full zero-copy, the M2
//!   allocator should place the data section at an address with
//!   `addr % 4096 == data_offset % 4096`.
//!
//! A tokio-uring implementation can replace `DirectSource`'s internals
//! behind the same trait if the one-time startup load ever needs queue
//! depth; sequential QD1 already saturates this drive.

use std::io;

use crate::parse::Gguf;

/// Fills a caller-provided buffer with the GGUF tensor-data section.
pub trait WeightSource {
    /// Byte length of the data section (`dst` must match exactly).
    fn data_len(&self) -> usize;

    /// Fill `dst` with the data section. `dst.len()` must equal
    /// [`data_len`](Self::data_len).
    fn load(&self, dst: &mut [u8]) -> io::Result<()>;
}

fn check_dst(expected: usize, dst: &[u8]) -> io::Result<()> {
    if dst.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "destination is {} bytes, data section is {expected}",
                dst.len()
            ),
        ));
    }
    Ok(())
}

/// Fallback path: copy from the file mmap.
#[derive(Debug)]
pub struct MmapCopySource<'a> {
    data: &'a [u8],
}

impl<'a> MmapCopySource<'a> {
    pub fn new(gguf: &Gguf<'a>) -> Self {
        Self {
            data: gguf.data_section(),
        }
    }
}

impl WeightSource for MmapCopySource<'_> {
    fn data_len(&self) -> usize {
        self.data.len()
    }

    fn load(&self, dst: &mut [u8]) -> io::Result<()> {
        check_dst(self.data.len(), dst)?;
        dst.copy_from_slice(self.data);
        Ok(())
    }
}

#[cfg(target_os = "linux")]
pub use direct::DirectSource;

#[cfg(target_os = "linux")]
mod direct {
    use std::fs::File;
    use std::io;
    use std::os::unix::fs::{FileExt, OpenOptionsExt};
    use std::path::{Path, PathBuf};

    use super::{WeightSource, check_dst};
    use crate::parse::Gguf;

    /// O_DIRECT requires file offset, read length, and destination address
    /// all aligned to the logical block size; 4096 covers both 512e and 4Kn
    /// drives.
    const BLOCK: usize = 4096;
    /// Read granularity. Large enough to saturate sequential NVMe, small
    /// enough that the bounce buffer is irrelevant next to the 17.5 GB
    /// payload.
    const CHUNK: usize = 16 << 20;

    /// O_DIRECT sequential loader for the data section of one GGUF file.
    #[derive(Debug)]
    pub struct DirectSource {
        path: PathBuf,
        /// Absolute file offset of the data section.
        data_offset: u64,
        data_len: usize,
    }

    impl DirectSource {
        pub fn new(path: impl AsRef<Path>, gguf: &Gguf<'_>) -> Self {
            Self {
                path: path.as_ref().to_owned(),
                data_offset: gguf.data_offset() as u64,
                data_len: gguf.data_section().len(),
            }
        }
    }

    impl WeightSource for DirectSource {
        fn data_len(&self) -> usize {
            self.data_len
        }

        fn load(&self, dst: &mut [u8]) -> io::Result<()> {
            check_dst(self.data_len, dst)?;
            if dst.is_empty() {
                return Ok(());
            }
            // Fails with EINVAL on filesystems without O_DIRECT (e.g. tmpfs);
            // the caller falls back to MmapCopySource.
            let file = File::options()
                .read(true)
                .custom_flags(libc::O_DIRECT)
                .open(&self.path)?;

            let mut bounce = AlignedBuf::new(CHUNK);
            // Aligned file window covering [data_offset, data_offset + len).
            let aligned_start = self.data_offset / BLOCK as u64 * BLOCK as u64;
            let end = self.data_offset + self.data_len as u64;
            // `dst[i]` holds file byte `data_offset + i`; a chunk at aligned
            // file offset `o` may be read straight into dst iff its target
            // address keeps the same 4 KiB phase as `o` (which is 0).
            let direct_ok = (dst.as_ptr() as u64 + aligned_start - self.data_offset)
                .is_multiple_of(BLOCK as u64);

            let mut offset = aligned_start;
            while offset < end {
                // O_DIRECT lets the read extend past EOF padding only up to
                // the file size; clamp the request to an aligned length and
                // accept the short read at the tail.
                let want = CHUNK.min((end - offset).next_multiple_of(BLOCK as u64) as usize);
                let dst_pos = offset as i64 - self.data_offset as i64;
                let whole_chunk_in_dst = dst_pos >= 0 && dst_pos as usize + want <= dst.len();

                if direct_ok && whole_chunk_in_dst {
                    let n = read_full(&file, &mut dst[dst_pos as usize..][..want], offset)?;
                    offset += n as u64;
                } else {
                    let n = read_full(&file, &mut bounce.as_mut_slice()[..want], offset)?;
                    // Intersect [offset, offset+n) with the data section.
                    let copy_start = offset.max(self.data_offset);
                    let copy_end = (offset + n as u64).min(end);
                    if copy_start < copy_end {
                        let len = (copy_end - copy_start) as usize;
                        let src = (copy_start - offset) as usize;
                        let at = (copy_start - self.data_offset) as usize;
                        dst[at..at + len].copy_from_slice(&bounce.as_mut_slice()[src..src + len]);
                    }
                    offset += n as u64;
                }
            }
            Ok(())
        }
    }

    /// Read until `buf` is full or EOF; error on zero progress before EOF.
    fn read_full(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        let mut done = 0;
        while done < buf.len() {
            match file.read_at(&mut buf[done..], offset + done as u64) {
                Ok(0) => break, // EOF (tail chunk)
                Ok(n) => done += n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        if done == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("EOF at file offset {offset} with bytes still expected"),
            ));
        }
        Ok(done)
    }

    /// 4 KiB-aligned heap buffer for O_DIRECT bounce reads.
    struct AlignedBuf {
        ptr: *mut u8,
        layout: std::alloc::Layout,
    }

    impl std::fmt::Debug for AlignedBuf {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("AlignedBuf")
                .field("len", &self.layout.size())
                .finish()
        }
    }

    impl AlignedBuf {
        fn new(len: usize) -> Self {
            let layout = std::alloc::Layout::from_size_align(len, BLOCK).expect("valid layout");
            // SAFETY: layout has non-zero size; allocation checked below.
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Self { ptr, layout }
        }

        fn as_mut_slice(&mut self) -> &mut [u8] {
            // SAFETY: ptr is a live allocation of layout.size() initialized
            // bytes, exclusively borrowed through &mut self.
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.layout.size()) }
        }
    }

    impl Drop for AlignedBuf {
        fn drop(&mut self) {
            // SAFETY: ptr was allocated with exactly this layout.
            unsafe { std::alloc::dealloc(self.ptr, self.layout) };
        }
    }

    // SAFETY: AlignedBuf owns its allocation outright; no shared state.
    unsafe impl Send for AlignedBuf {}
}
