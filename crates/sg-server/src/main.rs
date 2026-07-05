//! super-gemma inference server: LLM-API-style `/v1/messages` API over the
//! single-conversation Gemma 4 engine.
//!
//! Scope and design: `docs/plans/05-server-and-api.md`. Lands in M7.

use clap::Parser;

#[derive(Parser)]
#[command(version, about = "super-gemma inference server")]
struct Args {
    /// Path to the server TOML config.
    #[arg(long, default_value = "/etc/super-gemma.toml")]
    config: std::path::PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::bail!(
        "sg-server is not implemented yet (lands in M7); would load config from {}",
        args.config.display()
    )
}
