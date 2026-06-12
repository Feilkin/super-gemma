//! Eyeballing CLI (plan 03 step 4, M4): generate text from a prompt with
//! streaming output.
//!
//! ```sh
//! cargo run --release -p sg-model --example run -- --prompt "Why is the sky blue?"
//! ```
//!
//! Flags: `--raw` (plain continuation, no chat template), `--temperature`,
//! `--top-k`, `--top-p`, `--seed`, `--max-tokens`, `--model <path>`,
//! `--stop <text>` (repeatable). This is the M4 eyeballing tool; the
//! production binary is `sg-server` (M7).

use std::io::Write as _;
use std::path::PathBuf;

use sg_model::{EOS_TOKENS, GenerateParams, GpuModel, SamplerParams, StopReason};
use sg_tokenizer::template::{ChatMessage, ChatRole, PromptOptions, render_prompt};
use sg_tokenizer::{DetokBuffer, SpecialTokens, Tokenizer};

struct Args {
    model: PathBuf,
    prompt: String,
    raw: bool,
    params: GenerateParams,
    stops: Vec<String>,
}

fn parse_args() -> Args {
    let mut args = Args {
        model: PathBuf::from("models/gemma-4-31B_q4_0-it.gguf"),
        prompt: String::new(),
        raw: false,
        params: GenerateParams {
            sampling: SamplerParams::default(),
            seed: 0,
            max_tokens: 512,
        },
        stops: Vec::new(),
    };
    let mut it = std::env::args().skip(1);
    let value = |it: &mut dyn Iterator<Item = String>, flag: &str| {
        it.next().unwrap_or_else(|| panic!("{flag} needs a value"))
    };
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => args.model = value(&mut it, "--model").into(),
            "--prompt" => args.prompt = value(&mut it, "--prompt"),
            "--raw" => args.raw = true,
            "--temperature" => {
                args.params.sampling.temperature = value(&mut it, "--temperature").parse().unwrap()
            }
            "--top-k" => args.params.sampling.top_k = value(&mut it, "--top-k").parse().unwrap(),
            "--top-p" => args.params.sampling.top_p = value(&mut it, "--top-p").parse().unwrap(),
            "--seed" => args.params.seed = value(&mut it, "--seed").parse().unwrap(),
            "--max-tokens" => {
                args.params.max_tokens = value(&mut it, "--max-tokens").parse().unwrap()
            }
            "--stop" => args.stops.push(value(&mut it, "--stop")),
            other => panic!("unknown flag `{other}` (see the example header for usage)"),
        }
    }
    assert!(!args.prompt.is_empty(), "--prompt is required");
    args
}

fn main() {
    let args = parse_args();

    eprintln!("loading {} …", args.model.display());
    let t0 = std::time::Instant::now();
    let file = sg_gguf::GgufFile::open(&args.model).expect("open model");
    let gguf = file.parse().expect("parse model");
    let tokenizer = Tokenizer::from_metadata(&gguf.metadata).expect("tokenizer");
    let ctx = sg_gpu::GpuContext::new().expect("GPU");
    // 32K context, 256-token prefill chunks (plan 03 defaults for the CLI).
    let mut model = GpuModel::new(&ctx, &gguf, 32 * 1024, 256).expect("upload weights");
    eprintln!("ready in {:.1}s", t0.elapsed().as_secs_f32());

    // The chat template emits <bos> itself; raw mode prepends it manually
    // (tokenizer.ggml.add_bos_token = false).
    let text = if args.raw {
        args.prompt.clone()
    } else {
        render_prompt(
            &[ChatMessage::text(ChatRole::User, args.prompt.clone())],
            &[],
            PromptOptions {
                add_generation_prompt: true,
                enable_thinking: false,
            },
        )
    };
    let mut ids = if args.raw { vec![2] } else { Vec::new() };
    ids.extend(tokenizer.encode(&text, SpecialTokens::Match));
    eprintln!("prompt: {} tokens", ids.len());

    let t1 = std::time::Instant::now();
    let mut detok = DetokBuffer::new();
    let mut text_out = String::new();
    let mut printed = 0usize;
    let mut n_gen = 0usize;
    let mut ttft = None;
    let stops = args.stops.clone();
    let (_, reason) = model
        .generate(&ids, &args.params, |tok| {
            n_gen += 1;
            ttft.get_or_insert_with(|| t1.elapsed());
            if EOS_TOKENS.contains(&tok) {
                return true;
            }
            tokenizer
                .decode_streaming(tok, &mut detok, &mut text_out)
                .expect("decode");
            // Stop-sequence matching over the decoded text. Withhold the
            // longest stop-prefix overlap at the tail so a stop split
            // across tokens never leaks (plan 03 decode loop step 4).
            if let Some(at) = stops.iter().filter_map(|s| text_out.find(s)).min() {
                print!("{}", &text_out[printed..at]);
                let _ = std::io::stdout().flush();
                printed = text_out.len();
                return false;
            }
            let holdback = stops
                .iter()
                .map(|s| tail_overlap(&text_out, s))
                .max()
                .unwrap_or(0);
            let safe = text_out.len() - holdback;
            if safe > printed {
                // Respect char boundaries (holdback counts bytes).
                let mut end = safe;
                while !text_out.is_char_boundary(end) {
                    end -= 1;
                }
                print!("{}", &text_out[printed..end]);
                let _ = std::io::stdout().flush();
                printed = end;
            }
            true
        })
        .expect("generate");
    // Flush whatever detok still buffers (unless a stop sequence cut off).
    if reason != StopReason::Aborted {
        tokenizer.flush_streaming(&mut detok, &mut text_out);
        print!("{}", &text_out[printed..]);
    }
    println!();

    let dt = t1.elapsed().as_secs_f32();
    let ttft = ttft.unwrap_or_default().as_secs_f32();
    eprintln!(
        "\n[{n_gen} tokens, stop: {reason:?}; ttft {ttft:.2}s, decode {:.2} tok/s]",
        (n_gen.max(2) - 1) as f32 / (dt - ttft).max(1e-6)
    );
}

/// Length of the longest suffix of `text` that is a proper prefix of
/// `stop` — the bytes that might become a stop match with more tokens.
fn tail_overlap(text: &str, stop: &str) -> usize {
    let max = stop.len().saturating_sub(1).min(text.len());
    for l in (1..=max).rev() {
        if text.is_char_boundary(text.len() - l) && stop.starts_with(&text[text.len() - l..]) {
            return l;
        }
    }
    0
}
