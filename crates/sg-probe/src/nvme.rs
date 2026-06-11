//! NVMe sequential-read probe. Uses O_DIRECT on Linux (the cache2 IO path) so
//! the page cache doesn't inflate numbers; falls back to buffered reads where
//! O_DIRECT is unavailable (and says so in the report).

use serde::Serialize;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

const READ_CHUNK: usize = 8 * 1024 * 1024;
const DIRECT_ALIGN: usize = 4096;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

#[derive(Serialize)]
pub struct NvmeReport {
    pub path: String,
    pub bytes_read: u64,
    pub seconds: f64,
    pub read_gib_s: f64,
    /// False means buffered IO was used; treat the number as an upper bound.
    pub o_direct: bool,
}

pub fn probe(path: &Path, max_bytes: u64) -> anyhow::Result<NvmeReport> {
    let (mut file, o_direct) = open(path)?;
    let mut buf = AlignedBuf::new(READ_CHUNK, DIRECT_ALIGN);

    let mut total: u64 = 0;
    let start = Instant::now();
    while total < max_bytes {
        let n = file.read(buf.as_mut_slice())?;
        if n == 0 {
            break;
        }
        std::hint::black_box(&buf.as_mut_slice()[0]);
        total += n as u64;
    }
    let seconds = start.elapsed().as_secs_f64();

    Ok(NvmeReport {
        path: path.display().to_string(),
        bytes_read: total,
        seconds,
        read_gib_s: total as f64 / seconds / GIB,
        o_direct,
    })
}

#[cfg(target_os = "linux")]
fn open(path: &Path) -> anyhow::Result<(File, bool)> {
    use std::os::unix::fs::OpenOptionsExt;
    // Some filesystems (tmpfs, some overlayfs) reject O_DIRECT; fall back.
    match std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
    {
        Ok(file) => Ok((file, true)),
        Err(_) => Ok((File::open(path)?, false)),
    }
}

#[cfg(not(target_os = "linux"))]
fn open(path: &Path) -> anyhow::Result<(File, bool)> {
    Ok((File::open(path)?, false))
}

/// Heap buffer with explicit alignment, as required by O_DIRECT.
struct AlignedBuf {
    ptr: std::ptr::NonNull<u8>,
    layout: std::alloc::Layout,
}

impl AlignedBuf {
    fn new(len: usize, align: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(len, align).expect("valid layout");
        // SAFETY: layout has non-zero size.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = std::ptr::NonNull::new(ptr).expect("allocation failed");
        Self { ptr, layout }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr is valid for layout.size() bytes and exclusively
        // borrowed through &mut self.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.layout.size()) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: ptr was allocated by alloc_zeroed with this exact layout.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_buf_is_aligned() {
        let mut buf = AlignedBuf::new(READ_CHUNK, DIRECT_ALIGN);
        assert_eq!(buf.as_mut_slice().as_ptr() as usize % DIRECT_ALIGN, 0);
        assert_eq!(buf.as_mut_slice().len(), READ_CHUNK);
    }

    #[test]
    fn probe_reads_a_file() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("sg-probe-test-{}", std::process::id()));
        std::fs::write(&path, vec![7u8; 1024 * 1024]).unwrap();
        let report = probe(&path, u64::MAX).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(report.bytes_read, 1024 * 1024);
        assert!(report.read_gib_s > 0.0);
    }
}
