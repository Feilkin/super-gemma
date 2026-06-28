#!/usr/bin/env bash
#
# llama_bench_sweep.sh — prefill + decode throughput of llama.cpp vs context
# length, shaped to be comparable to our synthetic `sg-bench profile`.
#
# For each context length L it runs two llama-bench points on the GPU (ROCm):
#   prefill  : -p L  -n 0  -d 0      → pp throughput (tok/s) processing L tokens
#   decode   : -p 0  -n N  -d L      → tg throughput (tok/s) decoding N tokens
#                                       after an L-token KV history (depth)
# This mirrors our harness, which measures decode at a set `model.pos` and
# prefill as a token chunk at several histories q0. llama-bench generates the
# token sequences internally, so no prompt dataset is required.
#
# Results: one combined JSON array under bench/results/, plus a markdown table
# on stdout. Each point's full llama-bench JSON object is preserved (it records
# avg_ts, stddev_ts, samples, KV types, n_ubatch, flash_attn, etc.).
#
# SAFETY: a true 128K prefill processes 131072 tokens once — tens of minutes of
# sustained 100% GPU. This box has hard-power-cut at ~112 °C junction under
# sustained load, so we pin perf=high, guard on junction temp, and cool down
# between points. Reps auto-drop to 1 at L >= 65536. Run the big points attended.
#
# Knobs (env):
#   SG_LLAMA_MODEL   model gguf           (default models/gemma-4-31B_q4_0-it.gguf)
#   SG_BENCH_CTXS    context lengths      (default "256 8192 32768 65536 131072")
#   SG_BENCH_NGEN    decode output tokens (default 256)
#   SG_BENCH_REPS    reps at L < 65536    (default 3; L >= 65536 is forced to 1)
#   SG_BENCH_FA      flash attention 0|1  (default 1, matches our flash kernels)
#   SG_BENCH_CTK     primary K cache type (default f16 — see KV note below)
#   SG_BENCH_CTV     V cache type         (default f16)
#   SG_BENCH_BRACKET_MIN  ctx at/above which we ALSO run a q8_0-K bracket pass
#                    (default 32768; set huge to disable). Skipped if CTK=q8_0.
#   SG_BENCH_NGL     gpu layers           (default 99)
#   SG_BENCH_COOLDOWN  base cooldown secs (default 30; scaled up after hot points)
#   SG_BENCH_TEMPMAX   junction °C guard  (default 100; pause until it drops below)
#
# Examples:
#   tools/llama_bench_sweep.sh                              # full sweep
#   SG_BENCH_CTXS="256 8192 32768" tools/llama_bench_sweep.sh   # cheap points only
#   SG_BENCH_CTK=q8_0 SG_BENCH_CTXS=32768 tools/llama_bench_sweep.sh  # Q8-K pass only
#
# KV note: our engine's cache is MIXED — 50 sliding layers keep f16 K, the 10
# global layers store K as Q8_0 (i8 quants + f16 scales); V is f16 everywhere.
# llama-bench's -ctk is a single global type, so neither value matches us
# exactly. f16/f16 is the closest single setting (exact for 50/60 layers + all
# V). The Q8_0 difference lives entirely in the global-K, which dominates at
# long ctx — and our own Q8-K change measured +9%/+23% prefill at 8K/32K, so it
# is NOT obviously negligible. Hence the bracket: at ctx >= BRACKET_MIN we run
# both -ctk f16 and -ctk q8_0 so the real effect is measured, not assumed.
# (q8_0 over-quantizes the 50 sliding layers, but those are windowed at 1024
# tokens so the size/perf error there is tiny.)

set -euo pipefail

MODEL="${SG_LLAMA_MODEL:-models/gemma-4-31B_q4_0-it.gguf}"
CTXS="${SG_BENCH_CTXS:-256 8192 32768 65536 131072}"
NGEN="${SG_BENCH_NGEN:-256}"
REPS="${SG_BENCH_REPS:-1}"          # 1 rep: minimize the uninterruptible peg per point
FA="${SG_BENCH_FA:-1}"
CTK="${SG_BENCH_CTK:-f16}"
CTV="${SG_BENCH_CTV:-f16}"
NGL="${SG_BENCH_NGL:-99}"
# Thermal gating (Tctl / k10temp — the meaningful die sensor on this APU; the
# amdgpu `edge` hwmon under-reports badly). We can only check temp BETWEEN
# llama-bench invocations — a single run pegs the GPU uninterrupted — so the
# real protection is: start cool, keep each run short (reps=1), cool to a floor
# after. This box went 40->113C in ~2-3 min of load; keep these conservative.
TCTL_START="${SG_BENCH_TCTL_START:-50}"   # don't START a run unless Tctl <= this
TCTL_FLOOR="${SG_BENCH_TCTL_FLOOR:-40}"   # after each run, cool until Tctl <= this
COOLDOWN="${SG_BENCH_COOLDOWN:-30}"       # base sleep before the cool-to-floor wait
BRACKET_MIN="${SG_BENCH_BRACKET_MIN:-32768}"

PERF_NODE="/sys/class/drm/card1/device/power_dpm_force_performance_level"
OUTDIR="bench/results"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUTDIR}/llama-bench-sweep-${STAMP}.json"

command -v llama-bench >/dev/null || { echo "llama-bench not found in PATH" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }
mkdir -p "$OUTDIR"

# --- clock / perf level ------------------------------------------------------
if [ -r "$PERF_NODE" ]; then
  lvl="$(cat "$PERF_NODE")"
  if [ "$lvl" != "high" ]; then
    echo "WARNING: perf level is '$lvl', not 'high'. Pin it for stable clocks:" >&2
    echo "  echo high | sudo tee $PERF_NODE" >&2
  else
    echo "perf level: high (pinned)"
  fi
fi

# --- Tctl temperature (k10temp direct hwmon, whole °C) -----------------------
# The amdgpu `edge` hwmon and rocm-smi under-report on this APU; Tctl is the
# die sensor that actually tracks the thermal limit. Read the hwmon input
# directly — `sensors -u` prints the label BEFORE its value, which is what
# silently broke the previous awk-based guard.
read_tctl() {
  local h l
  for h in /sys/class/hwmon/hwmon*; do
    [ "$(cat "$h/name" 2>/dev/null)" = "k10temp" ] || continue
    for l in "$h"/temp*_label; do
      [ -f "$l" ] || continue
      if [ "$(cat "$l")" = "Tctl" ]; then
        echo $(( $(cat "${l%_label}_input") / 1000 )); return
      fi
    done
    [ -f "$h/temp1_input" ] && { echo $(( $(cat "$h/temp1_input") / 1000 )); return; }
  done
}

# Block until Tctl <= $1, polling every 10s. Aborts the whole sweep if it can't
# read Tctl (no silent "assume it's fine" — that's what burned us).
cool_to() {
  local target="$1" t
  t="$(read_tctl)"
  if [ -z "$t" ]; then echo "FATAL: cannot read Tctl — refusing to run blind" >&2; exit 1; fi
  while [ "$t" -gt "$target" ]; do
    echo "  Tctl ${t}°C > ${target}°C — cooling 10s …"
    sleep 10
    t="$(read_tctl)"
  done
  echo "  Tctl ${t}°C (<= ${target}°C)"
}

# After a run: base sleep, then wait down to the floor before the next peg.
cooldown() {
  echo "  cooldown ${COOLDOWN}s, then cool to ${TCTL_FLOOR}°C …"
  sleep "$COOLDOWN"
  cool_to "$TCTL_FLOOR"
}

# --- run one llama-bench point, append its JSON object to a temp file --------
ACC="$(mktemp)"
trap 'rm -f "$ACC"' EXIT
first=1

run_point() {
  local label="$1" p="$2" n="$3" d="$4" reps="$5" ctk="$6" ctv="$7"
  # Start-gate: never begin a run hot — we can't check temp once it's pegged.
  cool_to "$TCTL_START"
  echo ">>> ${label}: -p ${p} -n ${n} -d ${d}  (reps=${reps}, fa=${FA}, ctk=${ctk}, ctv=${ctv})"
  local start elapsed json
  start=$(date +%s)
  # -o json prints a JSON array (one object per point) on stdout; stderr = logs.
  json="$(llama-bench -m "$MODEL" -p "$p" -n "$n" -d "$d" \
            -fa "$FA" -ctk "$ctk" -ctv "$ctv" -ngl "$NGL" -r "$reps" -o json 2>/dev/null)"
  elapsed=$(( $(date +%s) - start ))
  echo "    Tctl after run: $(read_tctl)°C"
  # Strip the array brackets, tag each object with our label, accumulate.
  python3 - "$ACC" "$label" "$elapsed" "$first" <<'PY' "$json"
import json, sys
acc, label, elapsed, first = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
objs = json.loads(sys.argv[5])
with open(acc, "a") as f:
    for o in objs:
        o["sweep_label"] = label
        o["wall_secs"] = elapsed
        if first != "1":
            f.write(",\n")
        f.write(json.dumps(o, indent=2))
        first = "0"
PY
  first=0
  echo "    done in ${elapsed}s"
}

echo "model: $MODEL"
echo "writing: $OUT"
echo

for L in $CTXS; do
  reps="$REPS"
  [ "$L" -ge 65536 ] && reps=1   # the big points are minutes-long and hot

  # KV passes for this ctx: always the primary CTK/CTV; add a q8_0-K bracket at
  # long ctx (where global-K dominates) unless the primary already is q8_0.
  ctks="$CTK"
  if [ "$L" -ge "$BRACKET_MIN" ] && [ "$CTK" != "q8_0" ]; then
    ctks="$CTK q8_0"
  fi

  for ctk in $ctks; do
    run_point "prefill_ctx${L}_k${ctk}" "$L" 0 0 "$reps" "$ctk" "$CTV"
    cooldown "$COOLDOWN"
    run_point "decode_ctx${L}_n${NGEN}_k${ctk}" 0 "$NGEN" "$L" "$reps" "$ctk" "$CTV"
    cooldown "$COOLDOWN"
  done
done

# --- finalize JSON -----------------------------------------------------------
{ echo "["; cat "$ACC"; echo; echo "]"; } > "$OUT"
echo
echo "wrote $OUT"

# --- markdown summary --------------------------------------------------------
echo
python3 - "$OUT" "$NGEN" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1])); ngen = sys.argv[2]
# Key by (ctx, K-type); a "row" is the (prefill, decode) pair for that combo.
pre, dec = {}, {}
for o in rows:
    L = o["n_depth"] if o["n_gen"] else o["n_prompt"]
    key = (L, o.get("type_k"))
    (dec if o["n_gen"] else pre)[key] = o
keys = sorted(set(pre) | set(dec))
print(f"| ctx | K/V | prefill tok/s | ± | decode tok/s (n={ngen}) | ± |")
print("|----:|:----|--------------:|--:|------------------------:|--:|")
for key in keys:
    L, tk = key
    p, d = pre.get(key), dec.get(key)
    tv = (p or d).get("type_v")
    pv = f'{p["avg_ts"]:.1f}' if p else "—"
    ps = f'{p["stddev_ts"]:.1f}' if p else ""
    dv = f'{d["avg_ts"]:.1f}' if d else "—"
    ds = f'{d["stddev_ts"]:.1f}' if d else ""
    print(f"| {L} | {tk}/{tv} | {pv} | {ps} | {dv} | {ds} |")
print()
print("flash_attn:", rows[0].get("flash_attn"), "| n_ubatch:", rows[0].get("n_ubatch"),
      "| q8_0 rows = bracket (matches our global-K; f16 = our sliding-K + all V)")
PY
