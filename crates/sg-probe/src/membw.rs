//! CPU memory bandwidth probe (memcpy). On the target's unified memory this
//! approximates the DRAM streaming rate available to any single client and
//! calibrates the "GPU-visible copy" fallback paths.

use serde::Serialize;
use std::time::Instant;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

#[derive(Serialize)]
pub struct MembwReport {
    pub buffer_mib: usize,
    pub iters: usize,
    pub threads: usize,
    /// Bytes copied per second, single thread. Actual DRAM traffic is ~2-3x
    /// this (read + write, plus write-allocate depending on the copy path).
    pub single_thread_copy_gib_s: f64,
    pub multi_thread_copy_gib_s: f64,
}

pub fn probe(size_mib: usize, iters: usize) -> MembwReport {
    let len = size_mib * 1024 * 1024;
    let src = vec![1u8; len];
    let mut dst = vec![0u8; len];

    dst.copy_from_slice(&src); // warmup: faults pages in

    let start = Instant::now();
    for _ in 0..iters {
        dst.copy_from_slice(&src);
        std::hint::black_box(&mut dst);
    }
    let single = (len as f64 * iters as f64) / start.elapsed().as_secs_f64() / GIB;

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let chunk = len.div_ceil(threads);
    let start = Instant::now();
    std::thread::scope(|scope| {
        for (src_chunk, dst_chunk) in src.chunks(chunk).zip(dst.chunks_mut(chunk)) {
            scope.spawn(move || {
                for _ in 0..iters {
                    dst_chunk.copy_from_slice(src_chunk);
                    std::hint::black_box(&dst_chunk[0]);
                }
            });
        }
    });
    let multi = (len as f64 * iters as f64) / start.elapsed().as_secs_f64() / GIB;

    MembwReport {
        buffer_mib: size_mib,
        iters,
        threads,
        single_thread_copy_gib_s: single,
        multi_thread_copy_gib_s: multi,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn probe_smoke() {
        let report = super::probe(4, 1);
        assert!(report.single_thread_copy_gib_s > 0.0);
        assert!(report.multi_thread_copy_gib_s > 0.0);
    }
}
