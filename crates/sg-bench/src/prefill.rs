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
use std::path::PathBuf;
use std::time::{Duration, Instant};

use sg_model::GpuModel;

/// Timed reps (`SG_PREFILL_REPS`) and warmups (`SG_PREFILL_WARMUP`). Each rep is
/// a fresh full-sequence prefill — a sustained peg of minutes at long ctx — so
/// we cool to a floor (`SG_PREFILL_COOL`, °C; 0 disables) BEFORE each one. That
/// caps the soak at a single rep's safe plateau instead of letting back-to-back
/// reps accumulate heat (32K hit 88 °C un-cooled vs llama's 84). Timing is very
/// stable (cv ~0.3%), so 2 reps is plenty; drop to 1 for 128K. Run at perf=auto.
fn reps() -> usize {
    env_usize("SG_PREFILL_REPS", 2)
}
fn warmup() -> usize {
    env_usize("SG_PREFILL_WARMUP", 1)
}
/// Cool to this Tctl (°C) before each rep. 0 disables cooling. Default 50 —
/// near idle on this box, a cool start that matches llama's gated runs.
fn cool_floor() -> i64 {
    std::env::var("SG_PREFILL_COOL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50)
}

fn env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

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

/// Tctl (°C) from the k10temp hwmon — the die sensor that tracks the thermal
/// limit on this APU (the amdgpu `edge` sensor under-reports). `None` if no
/// k10temp node is found. Reads the `_input` directly; `sensors -u` ordering
/// is what silently broke an earlier scraping guard (see thermal memory).
fn read_tctl() -> Option<i64> {
    for e in std::fs::read_dir("/sys/class/hwmon").ok()?.flatten() {
        let p = e.path();
        if std::fs::read_to_string(p.join("name"))
            .unwrap_or_default()
            .trim()
            != "k10temp"
        {
            continue;
        }
        for n in 1..=8 {
            if std::fs::read_to_string(p.join(format!("temp{n}_label")))
                .unwrap_or_default()
                .trim()
                == "Tctl"
            {
                if let Ok(milli) = std::fs::read_to_string(p.join(format!("temp{n}_input")))
                    .unwrap_or_default()
                    .trim()
                    .parse::<i64>()
                {
                    return Some(milli / 1000);
                }
            }
        }
        // k10temp without labels: temp1 is Tctl.
        if let Ok(milli) = std::fs::read_to_string(p.join("temp1_input"))
            .unwrap_or_default()
            .trim()
            .parse::<i64>()
        {
            return Some(milli / 1000);
        }
    }
    None
}

/// Block until Tctl <= `floor` °C (poll every 5 s). `floor <= 0` disables
/// cooling. With cooling on, ABORT if Tctl can't be read — never peg the GPU
/// blind, which is what burned us at 113 °C.
fn cool_to(floor: i64) -> anyhow::Result<()> {
    if floor <= 0 {
        return Ok(());
    }
    loop {
        let t = read_tctl().ok_or_else(|| {
            anyhow::anyhow!(
                "cooling on (SG_PREFILL_COOL={floor}) but Tctl unreadable — \
                 set SG_PREFILL_COOL=0 to disable or fix the sensor"
            )
        })?;
        if t <= floor {
            eprintln!("  Tctl {t}°C (<= {floor}°C)");
            return Ok(());
        }
        eprintln!("  Tctl {t}°C > {floor}°C — cooling …");
        std::thread::sleep(Duration::from_secs(5));
    }
}

#[derive(Clone, serde::Serialize)]
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

    let (n_reps, n_warm, floor) = (reps(), warmup(), cool_floor());
    eprintln!(
        "reps = {n_reps} (+{n_warm} warmup), cool = {}",
        if floor > 0 {
            format!("to {floor}°C before each rep")
        } else {
            "disabled".into()
        }
    );

    // Write the JSON after every ctx (incremental): a long-ctx run can take many
    // minutes per point and a thermal abort must not lose completed points.
    let sha = git_sha();
    let when = format!(
        "unix:{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    );
    let dir = std::path::Path::new("bench/results");
    std::fs::create_dir_all(dir)?;
    let out: PathBuf = dir.join(format!("{}-prefill.json", &sha[..12.min(sha.len())]));
    let write_report = |prefill: &BTreeMap<u32, Point>| -> anyhow::Result<()> {
        let report = Report {
            git_sha: sha.clone(),
            when: when.clone(),
            perf_level: perf.clone(),
            chunk,
            global_cap: cap,
            prefill: prefill.clone(),
        };
        let mut f = std::fs::File::create(&out)?;
        f.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
        Ok(())
    };

    let mut prefill = BTreeMap::new();
    for &l in &ctxs {
        eprintln!("prefill {l} tokens (full sequence) …");
        let tokens: Vec<u32> = (0..l).map(|i| (1000 + i * 7) % 100_000).collect();
        let mut totals = Vec::new();
        for rep in 0..n_reps + n_warm {
            cool_to(floor)?; // start each rep cool — caps the soak at one rep's plateau
            model.reset(); // pos = 0: prefill the whole sequence from empty
            let t0 = Instant::now();
            model
                .prefill_plain(&tokens)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let warm = rep < n_warm;
            eprintln!("  rep {rep}: {ms:.1} ms{}", if warm { " (warmup)" } else { "" });
            if !warm {
                totals.push(ms);
            }
        }
        let total_ms = median(totals.clone());
        let mean = totals.iter().sum::<f64>() / totals.len() as f64;
        let var = totals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / totals.len() as f64;
        let point = Point {
            tok_s: l as f64 / (total_ms / 1e3),
            total_ms,
            cv: if mean > 0.0 { var.sqrt() / mean } else { 0.0 },
            reps: totals.len(),
        };
        eprintln!("  => {:.1} tok/s (cv {:.1}%)", point.tok_s, point.cv * 100.0);
        prefill.insert(l, point);
        write_report(&prefill)?;
    }

    println!("\nfull-sequence prefill (perf={perf}, chunk={chunk}, cap={cap})");
    println!("| ctx | tok/s | total ms | cv |");
    println!("|----:|------:|---------:|---:|");
    for (l, p) in &prefill {
        println!(
            "| {l} | {:.1} | {:.1} | {:.1}% |",
            p.tok_s,
            p.total_ms,
            p.cv * 100.0
        );
    }
    eprintln!("wrote {}", out.display());
    Ok(())
}
