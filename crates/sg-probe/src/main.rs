//! M0 hardware probe: enumerates Vulkan capabilities and measures CPU memory
//! bandwidth and NVMe sequential-read throughput. The JSON report from the
//! target Framework Desktop gets checked in under `docs/probe/` and feeds the
//! cache2 cost model and kernel specialization choices.

mod membw;
mod nvme;
mod vulkan;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    version,
    about = "super-gemma M0 hardware probe (JSON report on stdout)"
)]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Enumerate Vulkan devices: features, limits, cooperative-matrix configs, memory heaps.
    Vulkan,
    /// Measure CPU memory bandwidth (memcpy, single- and multi-threaded).
    Membw {
        /// Size of each copy buffer in MiB.
        #[arg(long, default_value_t = 1024)]
        size_mib: usize,
        /// Number of timed copy iterations.
        #[arg(long, default_value_t = 8)]
        iters: usize,
    },
    /// Measure sequential read throughput of a file (O_DIRECT on Linux).
    Nvme {
        /// File to read; use a large file on the target NVMe (e.g. the GGUF).
        path: std::path::PathBuf,
        /// Stop after reading this many bytes.
        #[arg(long, default_value_t = 8 * 1024 * 1024 * 1024)]
        max_bytes: u64,
    },
}

fn main() -> anyhow::Result<()> {
    match Args::parse().cmd {
        Cmd::Vulkan => emit(&vulkan::probe()?),
        Cmd::Membw { size_mib, iters } => emit(&membw::probe(size_mib, iters)),
        Cmd::Nvme { path, max_bytes } => emit(&nvme::probe(&path, max_bytes)?),
    }
}

fn emit<T: serde::Serialize>(report: &T) -> anyhow::Result<()> {
    println!("{}", serde_json::to_string_pretty(report)?);
    Ok(())
}
