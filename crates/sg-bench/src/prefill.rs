//! Full-sequence prefill throughput — wall-time prefilling L tokens from an
//! empty KV cache, `tok/s = L / time`. Built to be apples-to-apples with
//! llama.cpp's `llama-bench -p L` (`tools/llama_bench_sweep.sh`): same model /
//! quant / flash-attn, both at perf=auto. Emits JSON under `bench/results/`.
//!
//! This differs from `profile`'s prefill metric, which times a single 256-token
//! chunk at a fixed history `q0`. Here we prefill the whole sequence from
//! `pos=0`, growing the KV chunk-by-chunk — the quantity llama-bench reports.
//!
//! Synthetic tokens: kernel cost is data-independent (see `profile`), so we
//! feed `(1000 + i*7) % 100000`, always a valid embedding row for the 256K
//! vocab. Run at perf=auto and Tctl-guarded — a long prefill sustained-pegs the
//! GPU; pinned-high overheats >=8K on this box (see tools/llama_bench_sweep.sh).

use std::collections::BTreeMap;
use std::io::Write as _;
use std::time::Instant;

use sg_model::GpuModel;

// Reps run back-to-back with no cooldown, so a long-ctx run is a sustained peg
// (a 32K prefill is ~minutes each). Prefill timing is very stable (cv ~0%), so
// keep reps low to bound continuous load; run at perf=auto so the clock
// throttles to a plateau, and watch Tctl (see tools/llama_bench_sweep.sh).
const REPS: usize = 2;
const WARMUP: usize = 1;

fn env_u32s(var: &str, default: &[u32]) -> Vec<u32> {
    std::env::var(var)
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .filter(|v: &Vec<u32>| !v.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

/// Full-sequence prefill context lengths. Override `SG_PREFILL_CTXS=...`.
fn prefill_ctxs() -> Vec<u32> {
    env_u32s("SG_PREFILL_CTXS", &[256, 8192, 32 * 1024])
}

/// Prefill chunk (gemm M). Production value 256; rounded to the M_BLOCK (64).
fn prefill_chunk() -> usize {
    std::env::var("SG_PREFILL_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_owned())
        .unwrap_or_else(|| "unknown".into())
}

fn perf_level() -> String {
    std::fs::read_to_string("/sys/class/drm/card1/device/power_dpm_force_performance_level")
        .map(|s| s.trim().to_owned())
        .unwrap_or_else(|_| "unknown".into())
}

#[derive(serde::Serialize)]
struct Point {
    tok_s: f64,
    total_ms: f64,
    /// Coefficient of variation across timed reps (the trust proxy).
    cv: f64,
    reps: usize,
}

#[derive(serde::Serialize)]
struct Report {
    git_sha: String,
    when: String,
    perf_level: String,
    chunk: usize,
    global_cap: usize,
    /// ctx (full-sequence length) -> measured point.
    prefill: BTreeMap<u32, Point>,
}

pub fn run(model_path: &std::path::Path) -> anyhow::Result<()> {
    let file = sg_gguf::GgufFile::open(model_path)?;
    let gguf = file.parse().map_err(|e| anyhow::anyhow!("parse: {e}"))?;
    let ctx = sg_gpu::GpuContext::new().map_err(|e| anyhow::anyhow!("gpu: {e}"))?;
    let chunk = prefill_chunk().next_multiple_of(64);
    let ctxs = prefill_ctxs();
    let max_ctx = *ctxs.iter().max().expect("non-empty ctxs") as usize;
    // KV buffers size from the cap; it must cover the longest prefill. Default
    // to max_ctx; raise SG_GLOBAL_CAP for headroom or to match a longer run.
    let cap = std::env::var("SG_GLOBAL_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(max_ctx);
    if cap < max_ctx {
        anyhow::bail!("global_cap {cap} < max ctx {max_ctx}; raise SG_GLOBAL_CAP");
    }

    let perf = perf_level();
    if perf == "high" {
        eprintln!(
            "WARNING: perf=high — a long prefill sustained-pegs the GPU and overheats \
             >=8K on this box. Run at perf=auto (see tools/llama_bench_sweep.sh)."
        );
    }
    eprintln!("uploading weights … (chunk = {chunk}, global_cap = {cap}, perf = {perf})");
    let mut model =
        GpuModel::new(&ctx, &gguf, cap, chunk).map_err(|e| anyhow::anyhow!("upload: {e}"))?;

    let mut prefill = BTreeMap::new();
    for &l in &ctxs {
        eprintln!("prefill {l} tokens (full sequence) …");
        let tokens: Vec<u32> = (0..l).map(|i| (1000 + i * 7) % 100_000).collect();
        let mut totals = Vec::new();
        for rep in 0..REPS + WARMUP {
            model.reset(); // pos = 0: prefill the whole sequence from empty
            let t0 = Instant::now();
            model
                .prefill_plain(&tokens)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let warm = rep < WARMUP;
            eprintln!("  rep {rep}: {ms:.1} ms{}", if warm { " (warmup)" } else { "" });
            if !warm {
                totals.push(ms);
            }
        }
        let total_ms = median(totals.clone());
        let mean = totals.iter().sum::<f64>() / totals.len() as f64;
        let var = totals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / totals.len() as f64;
        prefill.insert(
            l,
            Point {
                tok_s: l as f64 / (total_ms / 1e3),
                total_ms,
                cv: if mean > 0.0 { var.sqrt() / mean } else { 0.0 },
                reps: totals.len(),
            },
        );
    }

    let report = Report {
        git_sha: git_sha(),
        when: format!(
            "unix:{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        ),
        perf_level: perf,
        chunk,
        global_cap: cap,
        prefill,
    };

    println!(
        "\nfull-sequence prefill (perf={}, chunk={}, cap={})",
        report.perf_level, report.chunk, report.global_cap
    );
    println!("| ctx | tok/s | total ms | cv |");
    println!("|----:|------:|---------:|---:|");
    for (l, p) in &report.prefill {
        println!(
            "| {l} | {:.1} | {:.1} | {:.1}% |",
            p.tok_s,
            p.total_ms,
            p.cv * 100.0
        );
    }

    let dir = std::path::Path::new("bench/results");
    std::fs::create_dir_all(dir)?;
    let out = dir.join(format!(
        "{}-prefill.json",
        &report.git_sha[..12.min(report.git_sha.len())]
    ));
    let mut f = std::fs::File::create(&out)?;
    f.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    eprintln!("wrote {}", out.display());
    Ok(())
}
