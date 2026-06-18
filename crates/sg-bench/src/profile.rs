//! Full-pipeline e2e profile (plan 06; the post-M4 "what to optimize next"
//! ranking agreed 2026-06-12).
//!
//! Two-tier timing: WALL time around the uninstrumented graphs for the
//! totals, plus per-DISPATCH timestamps over one representative sliding +
//! global layer, extrapolated by layer counts (50/10). Two constraints
//! shaped this (both diagnosed 2026-06-12, several gfx ring resets):
//! amdgpu's job watchdog kills any submission over `lockup_timeout`
//! (default 2000 ms on this kernel) — prefill is segmented for exactly
//! that reason (sg-model graph.rs) — and per-dispatch timestamps drain the
//! pipeline, which would push the overlapped prefill far over that limit.
//!
//! KV contents are zero-filled at synthetic contexts (`pos` is just set);
//! kernel cost is data-independent. Attribution at overlap boundaries is
//! fuzzy (sg-gpu docs) — fine for ranking.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::time::Instant;

use sg_model::{GpuModel, Sampler, SamplerParams};

const DECODE_CTXS: &[u32] = &[1024, 8192, 32 * 1024];
const REPS: usize = 5;
const WARMUP: usize = 2;
const GLOBAL_CAP: usize = 32 * 1024;
/// Prefill chunk (token rows / gemm M). Default 256 — the production value
/// (`examples/run.rs`, `Session`) and the measured sweet spot. Override with
/// `SG_PREFILL_CHUNK=<n>` to re-run the sweep; rounded up to the gemm M_BLOCK
/// (64) here to match `GpuModel::new`.
///
/// Sweep finding (int8-ffn, perf=high, 2026-06-18) — chunk size is a ±7 %
/// lever with NO single winner, so we keep 256:
///   chunk | q0 0  | q0 8K | q0 32K   (e2e prefill tok/s)
///     128 | 258.0 | 159.4 |  85.2    strictly worse — half the weight reuse
///     256 | 271.0 | 180.1 | 101.2    best at short ctx (gemm-bound)
///     512 | 255.2 | 182.5 | 108.6    best at long ctx (KV-history reuse)
/// At q0 0 the FFN gemms dominate and 256 beats 512: the L2 swizzle already
/// reuses each weight N-strip across a 256-chunk's 4 M-blocks, and the 774 KB
/// strip can't survive 8 M-blocks' activation streaming through the 2 MB L2
/// (weight reuse is L2-bounded in M, not unbounded). At long ctx the global
/// attention amortizes its KV-history read over more query rows, so 512 pulls
/// ahead — a context-adaptive chunk (256→512 past ~8K) is the open follow-up.
fn prefill_chunk() -> usize {
    std::env::var("SG_PREFILL_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256)
}
/// Representative layers: 4 = sliding, 5 = global (the pattern repeats).
const REP_SLIDING: usize = 4;
const REP_GLOBAL: usize = 5;

#[derive(serde::Serialize)]
struct Report {
    git_sha: String,
    when: String,
    decode: BTreeMap<u32, Phase>,
    prefill_chunk256: BTreeMap<u32, Phase>,
    cpu: CpuSide,
}

#[derive(serde::Serialize)]
struct Phase {
    /// Wall time around the uninstrumented graph's blocking submit.
    total_ms: f64,
    tok_s: f64,
    /// Kernel-level ms for ONE representative layer of each kind
    /// (drain-serialized — exact per kernel, sums slightly above the
    /// overlapped per-layer reality).
    sliding_kernels: Vec<(String, f64)>,
    global_kernels: Vec<(String, f64)>,
}

#[derive(serde::Serialize)]
struct CpuSide {
    stage_token_us: f64,
    stage_chunk256_us: f64,
    sampler_us: f64,
    submit_overhead_us: f64,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Sum each label's intervals for one submission, in ms.
fn by_label(labels: &[&'static str], ts: &[f64]) -> BTreeMap<String, f64> {
    let mut out: BTreeMap<String, f64> = BTreeMap::new();
    for (i, l) in labels.iter().enumerate() {
        *out.entry((*l).to_owned()).or_default() += (ts[i + 1] - ts[i]) / 1e6;
    }
    out
}

/// Median of each label across reps, descending by time.
fn med_by_label(reps: &[BTreeMap<String, f64>]) -> Vec<(String, f64)> {
    let mut out: Vec<(String, f64)> = reps[0]
        .keys()
        .map(|k| (k.clone(), median(reps.iter().map(|r| r[k]).collect())))
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

struct Profiled {
    graph: sg_gpu::CommandGraph,
    labels: Vec<&'static str>,
    n_ts: u32,
}

impl Profiled {
    /// Submit + read one rep; returns (per-label ms, total GPU ms, wall ms).
    fn run(
        &self,
        model: &GpuModel<'_>,
        timer: &sg_gpu::GpuTimer,
    ) -> anyhow::Result<(BTreeMap<String, f64>, f64, f64)> {
        let t0 = Instant::now();
        model
            .submit(&self.graph)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let wall = t0.elapsed().as_secs_f64() * 1e3;
        let ts = timer
            .read_ns_prefix(self.n_ts)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let total = (ts[ts.len() - 1] - ts[0]) / 1e6;
        Ok((by_label(&self.labels, &ts), total, wall))
    }
}

pub fn run(model_path: &std::path::Path) -> anyhow::Result<()> {
    let file = sg_gguf::GgufFile::open(model_path)?;
    let gguf = file.parse().map_err(|e| anyhow::anyhow!("parse: {e}"))?;
    let ctx = sg_gpu::GpuContext::new().map_err(|e| anyhow::anyhow!("gpu: {e}"))?;
    // GpuModel rounds the chunk up to the gemm M_BLOCK (64); match that here.
    let chunk = prefill_chunk().next_multiple_of(64);
    eprintln!("uploading weights … (prefill chunk = {chunk})");
    let mut model = GpuModel::new(&ctx, &gguf, GLOBAL_CAP, chunk)
        .map_err(|e| anyhow::anyhow!("upload: {e}"))?;
    let timer = ctx
        .new_timer(256)
        .map_err(|e| anyhow::anyhow!("timer: {e}"))?;
    let err = |e: sg_gpu::GpuError| anyhow::anyhow!("{e}");

    eprintln!("recording graphs …");
    let mk = |g: Result<(sg_gpu::CommandGraph, Vec<&'static str>), sg_gpu::GpuError>| {
        g.map(|(graph, labels)| {
            let n_ts = labels.len() as u32 + 1;
            Profiled {
                graph,
                labels,
                n_ts,
            }
        })
        .map_err(err)
    };
    let d_plain = model.record(0..model.desc.n_layers, true).map_err(err)?;
    let d_sl = mk(model.record_layers_kernel_profiled(REP_SLIDING..REP_SLIDING + 1, &timer))?;
    let d_gl = mk(model.record_layers_kernel_profiled(REP_GLOBAL..REP_GLOBAL + 1, &timer))?;
    let p_plain = model.record_prefill_chunk_plain().map_err(err)?;
    let p_sl = mk(model.record_prefill_kernel_profiled(REP_SLIDING..REP_SLIDING + 1, &timer))?;
    let p_gl = mk(model.record_prefill_kernel_profiled(REP_GLOBAL..REP_GLOBAL + 1, &timer))?;

    // ── Decode at several contexts ───────────────────────────────────────
    let mut decode = BTreeMap::new();
    let mut submit_overheads = Vec::new();
    for &ctx_len in DECODE_CTXS {
        eprintln!("decode @ ctx {ctx_len} …");
        model.reset();
        model.pos = ctx_len - 1; // decode the token at position ctx_len−1
        let mut totals = Vec::new();
        let (mut sl_reps, mut gl_reps) = (Vec::new(), Vec::new());
        for rep in 0..REPS + WARMUP {
            model.stage_token(9259).map_err(err)?;
            let t0 = Instant::now();
            model.submit(&d_plain).map_err(err)?;
            let wall = t0.elapsed().as_secs_f64() * 1e3;
            let (sl, sl_total, sl_wall) = d_sl.run(&model, &timer)?;
            let (gl, _, _) = d_gl.run(&model, &timer)?;
            if rep >= WARMUP {
                totals.push(wall);
                submit_overheads.push((sl_wall - sl_total) * 1e3);
                sl_reps.push(sl);
                gl_reps.push(gl);
            }
        }
        let total_ms = median(totals);
        decode.insert(
            ctx_len,
            Phase {
                total_ms,
                tok_s: 1e3 / total_ms,
                sliding_kernels: med_by_label(&sl_reps),
                global_kernels: med_by_label(&gl_reps),
            },
        );
    }

    // ── Prefill chunk at several histories ───────────────────────────────
    let chunk_tokens: Vec<u32> = (0..chunk as u32).map(|i| 1000 + i * 7).collect();
    // Last history keeps q0 + chunk within the global KV cap.
    let prefill_q0s: [u32; 3] = [0, 8192, GLOBAL_CAP as u32 - chunk as u32];
    let mut prefill = BTreeMap::new();
    for &q0 in &prefill_q0s {
        eprintln!("prefill {chunk}-chunk @ q0 {q0} …");
        model.reset();
        let mut totals = Vec::new();
        let (mut sl_reps, mut gl_reps) = (Vec::new(), Vec::new());
        for rep in 0..REPS + WARMUP {
            model.pos = q0;
            model.stage_prefill_chunk(&chunk_tokens).map_err(err)?;
            let t0 = Instant::now();
            for segment in &p_plain {
                model.submit(segment).map_err(err)?;
            }
            let wall = t0.elapsed().as_secs_f64() * 1e3;
            let (sl, _, _) = p_sl.run(&model, &timer)?;
            let (gl, _, _) = p_gl.run(&model, &timer)?;
            if rep >= WARMUP {
                totals.push(wall);
                sl_reps.push(sl);
                gl_reps.push(gl);
            }
        }
        let total_ms = median(totals);
        prefill.insert(
            q0,
            Phase {
                total_ms,
                tok_s: chunk as f64 / (total_ms / 1e3),
                sliding_kernels: med_by_label(&sl_reps),
                global_kernels: med_by_label(&gl_reps),
            },
        );
    }

    // ── CPU-side costs ────────────────────────────────────────────────────
    eprintln!("cpu-side …");
    model.reset();
    model.pos = 1024;
    let time_n = |n: usize, mut f: Box<dyn FnMut() -> anyhow::Result<()> + '_>| {
        let mut times = Vec::new();
        for _ in 0..n {
            let t0 = Instant::now();
            f()?;
            times.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        Ok::<f64, anyhow::Error>(median(times))
    };
    let stage_token_us = time_n(
        20,
        Box::new(|| model.stage_token(9259).map_err(|e| anyhow::anyhow!("{e}"))),
    )?;
    let stage_chunk256_us = time_n(
        10,
        Box::new(|| {
            model
                .stage_prefill_chunk(&chunk_tokens)
                .map_err(|e| anyhow::anyhow!("{e}"))
        }),
    )?;
    let logits = vec![0.013f32; model.desc.vocab_size]; // flat worst case
    let mut sampler = Sampler::new(7);
    let params = SamplerParams::default();
    let sampler_us = time_n(
        50,
        Box::new(|| {
            std::hint::black_box(sampler.sample(&logits, &params));
            Ok(())
        }),
    )?;

    let report = Report {
        git_sha: git_sha(),
        when: format!(
            "unix:{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        ),
        decode,
        prefill_chunk256: prefill,
        cpu: CpuSide {
            stage_token_us,
            stage_chunk256_us,
            sampler_us,
            submit_overhead_us: median(submit_overheads),
        },
    };

    print_table(&report);
    let dir = std::path::Path::new("bench/results");
    std::fs::create_dir_all(dir)?;
    let out = dir.join(format!(
        "{}-e2e-profile.json",
        &report.git_sha[..12.min(report.git_sha.len())]
    ));
    std::fs::write(&out, serde_json::to_string_pretty(&report)?)?;
    eprintln!("\nwrote {}", out.display());
    Ok(())
}

fn print_table(r: &Report) {
    let mut o = std::io::stderr().lock();
    let phase = |o: &mut dyn std::io::Write, name: String, p: &Phase, scale_note: &str| {
        let _ = writeln!(o, "\n{name}: {:.2} ms = {:.2} tok/s", p.total_ms, p.tok_s);
        let _ = writeln!(o, "  one sliding layer{scale_note}:");
        for (k, ms) in p.sliding_kernels.iter().take(6) {
            let _ = writeln!(o, "    {k:26} {:9.1} µs", ms * 1e3);
        }
        let _ = writeln!(o, "  one global layer{scale_note}:");
        for (k, ms) in p.global_kernels.iter().take(6) {
            let _ = writeln!(o, "    {k:26} {:9.1} µs", ms * 1e3);
        }
    };
    for (ctx, p) in &r.decode {
        phase(
            &mut o,
            format!("decode @ ctx {ctx}"),
            p,
            " (×50 / ×10 for totals)",
        );
    }
    for (q0, p) in &r.prefill_chunk256 {
        phase(
            &mut o,
            format!("prefill chunk @ q0 {q0}"),
            p,
            " (×50 / ×10 for totals)",
        );
    }
    let _ = writeln!(
        o,
        "\nCPU: stage_token {:.0} µs, stage_chunk256 {:.0} µs, sampler {:.0} µs (flat-logits \
         worst case), submit overhead {:.0} µs",
        r.cpu.stage_token_us, r.cpu.stage_chunk256_us, r.cpu.sampler_us, r.cpu.submit_overhead_us
    );
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
