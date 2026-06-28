#!/usr/bin/env python3
"""Join super-gemma vs llama.cpp throughput into one comparison table.

Prefill is the headline: our `sg-bench prefill` JSON (full-sequence tok/s) vs a
`llama_bench_sweep.sh` JSON, matched on context length. With --decode, also
joins decode tok/s (our `sg-bench profile` synthetic decode vs llama's `-d L`).

We compare against llama's f16/f16 KV — established as both its faster config
and the closest match to our mixed cache (sliding f16-K, global Q8_0-K, V f16).
For a fair read, both sides should have been measured at perf=auto.

Usage:
  tools/compare_prefill.py <sg-prefill.json> <llama-bench-sweep.json>
                           [--decode <sg-e2e-profile.json>]
"""
import argparse
import json


def load(p):
    with open(p) as f:
        return json.load(f)


def llama_by_ctx(rows, *, decode):
    """f16-KV llama-bench points -> {ctx: tok/s}. Prefill points have n_gen==0
    (ctx = n_prompt); decode points have n_gen>0 (ctx = n_depth)."""
    out = {}
    for o in rows:
        if o.get("type_k") != "f16":
            continue
        is_decode = o["n_gen"] != 0
        if is_decode != decode:
            continue
        out[o["n_depth"] if is_decode else o["n_prompt"]] = o["avg_ts"]
    return out


def table(title, sg, la, unit):
    print(f"\n## {title}")
    print(f"| ctx | super-gemma {unit} | llama.cpp {unit} | sg / llama |")
    print("|----:|------------------:|---------------:|-----------:|")
    for ctx in sorted(set(sg) | set(la)):
        s, l = sg.get(ctx), la.get(ctx)
        sv = f"{s:.1f}" if s is not None else "—"
        lv = f"{l:.1f}" if l is not None else "—"
        ratio = f"{s / l:.2f}×" if s and l else "—"
        print(f"| {ctx} | {sv} | {lv} | {ratio} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sg_prefill", help="sg-bench prefill JSON (<sha>-prefill.json)")
    ap.add_argument("llama_sweep", help="llama_bench_sweep.sh JSON")
    ap.add_argument("--decode", help="sg-bench profile JSON (<sha>-e2e-profile.json)")
    args = ap.parse_args()

    sg = load(args.sg_prefill)
    llama = load(args.llama_sweep)

    sg_perf = sg.get("perf_level", "?")
    print(f"super-gemma perf={sg_perf}, chunk={sg.get('chunk')} · llama.cpp f16/f16, fa=on")

    sg_pre = {int(k): v["tok_s"] for k, v in sg["prefill"].items()}
    table("prefill tok/s (full sequence)", sg_pre, llama_by_ctx(llama, decode=False), "tok/s")

    if args.decode:
        prof = load(args.decode)
        sg_dec = {int(k): v["tok_s"] for k, v in prof.get("decode", {}).items()}
        table("decode tok/s", sg_dec, llama_by_ctx(llama, decode=True), "tok/s")
        print("\nnote: our decode is synthetic-position; llama's builds a real KV at depth.")


if __name__ == "__main__":
    main()
