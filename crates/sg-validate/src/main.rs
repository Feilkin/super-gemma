//! Correctness harness: kernel/layer/logit parity against references,
//! perplexity, and the cache/determinism invariants.
//!
//! Scope: `docs/plans/06-testing-validation-benchmarking.md`. Subcommands
//! land with their milestones (kernels: M2, layers: M3, logits/ppl: M4,
//! invariants: M5/M6, tokenizer: M1).

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(version, about = "super-gemma correctness harness")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Per-kernel parity vs CPU references (M2).
    Kernels,
    /// Per-layer activation dump + diff vs the CPU reference model (M3).
    Layers,
    /// End-to-end logit comparison vs llama.cpp on the same GGUF (M4).
    Logits,
    /// Perplexity on fixed corpora (M4).
    Ppl,
    /// Determinism and cache-exactness invariants (M5/M6).
    Invariants,
    /// Tokenizer + chat-template parity vs HF fixtures (M1).
    Tokenizer,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::bail!(
        "`{:?}` is not implemented yet; see the milestone map in this binary's docs",
        args.cmd
    )
}
