//! Minimal end-to-end runner: prompt → prefill → sampled decode → text
//! streamed to stdout, so you can WATCH the model generate. This is a dev
//! driver, not the production engine (sg-engine, M5+) — it wires the existing
//! pieces (`template::render_prompt`, `Tokenizer`, `GpuModel::generate`)
//! together with a tokenizer and stdout.
//!
//! Run (decode is memory-bound, not max-power like the gemm benches, but it is
//! sustained — `perf=auto` is the safe choice here, not pinned `high`):
//!
//! ```sh
//! SG_MODEL_GGUF=/abs/path/model.gguf \
//!   cargo run --release -p sg-model --example generate -- "Your prompt here"
//! ```
//!
//! Env knobs: `SG_MAX_TOKENS` (default 256), `SG_TEMP` (default 0 = greedy),
//! `SG_SEED` (default 0), `SG_CTX` (global-KV token cap, default 4096),
//! `SG_THINK` (1 = enable the thinking block, default 0).

use std::io::Write;
use std::time::Instant;

use sg_model::{EOS_TOKENS, GenerateParams, GpuModel, SamplerParams};
use sg_tokenizer::template::{ChatMessage, ChatRole, PromptOptions, render_prompt};
use sg_tokenizer::{DetokBuffer, SpecialTokens, Tokenizer};

type BoxErr = Box<dyn std::error::Error + Send + Sync>;

/// Box any Display error (the crate errors aren't all guaranteed `Send+Sync`).
fn boxed(e: impl std::fmt::Display) -> BoxErr {
    e.to_string().into()
}

fn env<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<(), BoxErr> {
    let path = std::env::var("SG_MODEL_GGUF")
        .map_err(|_| boxed("set SG_MODEL_GGUF to an absolute .gguf path"))?;
    let prompt: String = {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.is_empty() {
            "Write a haiku about cooperative matrices.".to_string()
        } else {
            args.join(" ")
        }
    };
    let max_tokens = env("SG_MAX_TOKENS", 256usize);
    let temperature = env("SG_TEMP", 0.0f32);
    let seed = env("SG_SEED", 0u64);
    let ctx_cap = env("SG_CTX", 4096usize);
    let think = env("SG_THINK", 0u32) != 0;

    eprintln!("loading {path} …");
    let file = sg_gguf::GgufFile::open(&path).map_err(boxed)?;
    let gguf = file.parse().map_err(boxed)?;
    let tokenizer = Tokenizer::from_metadata(&gguf.metadata).map_err(boxed)?;
    let ctx = sg_gpu::GpuContext::new().map_err(boxed)?;
    eprintln!("uploading weights to {} …", ctx.device_name());
    let mut model = GpuModel::new(&ctx, &gguf, ctx_cap, 256).map_err(boxed)?;

    // Render the Gemma chat prompt (byte-identical to the GGUF's chat template),
    // then tokenize with `Match` so `<bos>` / `<start_of_turn>` etc. are taken
    // as their special tokens rather than BPE-split.
    let messages = [ChatMessage::text(ChatRole::User, prompt)];
    let rendered = render_prompt(
        &messages,
        &[],
        PromptOptions {
            add_generation_prompt: true,
            enable_thinking: think,
        },
    );
    let tokens = tokenizer.encode(&rendered, SpecialTokens::Match);
    eprintln!(
        "prompt: {} tokens; generating (temp {temperature}, max {max_tokens}) …\n",
        tokens.len()
    );

    let params = GenerateParams {
        sampling: SamplerParams {
            temperature,
            ..SamplerParams::default()
        },
        seed,
        max_tokens,
    };

    // Stream each sampled token's text as it is produced. The detok buffer holds
    // partial UTF-8 across byte-tokens so a code point is never split.
    let mut buf = DetokBuffer::new();
    let mut piece = String::new();
    let start = Instant::now();
    let (out, reason) = model
        .generate(&tokens, &params, |tok| {
            if !EOS_TOKENS.contains(&tok) {
                piece.clear();
                let _ = tokenizer.decode_streaming(tok, &mut buf, &mut piece);
                print!("{piece}");
                let _ = std::io::stdout().flush();
            }
            true
        })
        .map_err(boxed)?;

    let dt = start.elapsed().as_secs_f64();
    let n = out.len();
    eprintln!(
        "\n\n[{n} tokens in {dt:.1}s = {:.1} tok/s; stop = {reason:?}]",
        n as f64 / dt.max(1e-9)
    );
    Ok(())
}
