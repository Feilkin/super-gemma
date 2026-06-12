//! Benchmark harness: headline metrics (decode tok/s, TTFT cold/warm,
//! prefill, jitter, agent-loop composite) and micro/diagnostic benchmarks,
//! emitting JSON under `bench/results/`.
//!
//! Scope: `docs/plans/06-testing-validation-benchmarking.md`. Lands from M2
//! (kernel micro) onward.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

mod profile;

#[derive(Parser)]
#[command(version, about = "super-gemma benchmark harness (JSON results)")]
struct Args {
    /// Path to the model GGUF.
    #[arg(long, default_value = "models/gemma-4-31B_q4_0-it.gguf")]
    model: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Kernel microbenchmarks: GB/s, TFLOPS, attention latencies (M2).
    Kernels,
    /// Decode throughput vs context length (M4+).
    Decode,
    /// TTFT cold / warm / partial-hit (M6+).
    Ttft,
    /// Prefill throughput (M4+).
    Prefill,
    /// Per-kernel e2e profile of decode steps and prefill chunks.
    Profile,
    /// Scripted coding-agent session composite (M7+).
    AgentLoop,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match args.cmd {
        Cmd::Profile => profile::run(&args.model),
        cmd => anyhow::bail!(
            "`{cmd:?}` is not implemented yet; see the milestone map in this binary's docs"
        ),
    }
}
