#!/usr/bin/env python3
"""Generate the llama.cpp perplexity baselines (plan 06 rung 4, M4 gate).

Runs the installed `llama-perplexity` on the committed corpus fixtures with
the same methodology `sg-model`'s perplexity test replicates (n_ctx 512,
fresh context per chunk, NLL over the second half of each window — see
tools/perplexity in the llama.cpp source). The GPU backend is fine here:
perplexity aggregates thousands of NLL terms, so backend noise (~1e-3 nll)
is far inside the 0.5 % gate.

Corpora (committed, regenerate deliberately):
  - fixtures/ppl_wikitext.txt: wikitext-2-raw-v1 test split, first ~59 KB
    (paragraph-aligned slice; see git history for the fetch snippet).
  - fixtures/ppl_code.txt: snapshot of this repo's own source (30 KB).

Run from the repo root:
    python3 tools/gen_ppl_baseline.py

Outputs: crates/sg-model/tests/fixtures/ppl_baseline.json
"""

import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models/gemma-4-31B_q4_0-it.gguf"
FIXTURES = REPO / "crates/sg-model/tests/fixtures"
N_CTX = 512


def baseline(corpus: Path) -> dict:
    out = subprocess.run(
        [
            "llama-perplexity",
            "-m", str(MODEL),
            "-f", str(corpus),
            "-c", str(N_CTX),
            "-ngl", "99",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    text = out.stdout + out.stderr
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", text)
    if not m:
        raise SystemExit(f"no PPL in llama-perplexity output:\n{text[-2000:]}")
    chunks = re.search(r"calculating perplexity over (\d+) chunks", text)
    return {
        "file": corpus.name,
        "ppl": float(m.group(1)),
        "n_chunk": int(chunks.group(1)) if chunks else None,
        "n_ctx": N_CTX,
    }


def main():
    version = subprocess.run(
        ["llama-perplexity", "--version"], capture_output=True, text=True, check=False
    )
    results = {
        "generator": (version.stderr + version.stdout).strip().splitlines()[0],
        "corpora": [
            baseline(FIXTURES / "ppl_wikitext.txt"),
            baseline(FIXTURES / "ppl_code.txt"),
        ],
    }
    out = FIXTURES / "ppl_baseline.json"
    out.write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
