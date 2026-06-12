#!/usr/bin/env python3
"""Generate llama.cpp logit parity fixtures (plan 06 rung 4, used from M3 on).

Runs the *installed* llama.cpp server (pinned by the distro package, see the
header line of the output) against the same GGUF this repo serves, and
records the top-N pre-sampling logprobs of the next token after each fixed
prompt. `sg-model`'s `llamacpp_parity` test must match these within
rank-overlap / KL thresholds — that's the gate that empirically confirms the
pinned forward-graph reading (docs/reference/gemma4-forward-graph.md).

Run from the repo root (no Python deps beyond stdlib):
    python3 tools/gen_logits_fixtures.py

Inputs:  models/gemma-4-31B_q4_0-it.gguf
Outputs: crates/sg-model/tests/fixtures/llamacpp_logits.jsonl

Never hand-edit the output; regenerate. Prompts are fixed below — the third
deliberately exceeds the 1024-token sliding window.
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models/gemma-4-31B_q4_0-it.gguf"
OUT = REPO / "crates/sg-model/tests/fixtures/llamacpp_logits.jsonl"
PORT = 18283
BOS = 2
TOP_N = 20

# (name, text, min_expected_tokens) — text is tokenized by the server
# (add_special=false), then BOS is prepended explicitly so the ids in the
# fixture are exactly what the reference model consumes.
PROMPTS = [
    ("short", "The capital of Finland is", 0),
    (
        "code",
        "// Rust: iterative fibonacci\nfn fib(n: u64) -> u64 {\n"
        "    let (mut a, mut b) = (0u64, 1);\n    for _ in 0..n {",
        0,
    ),
    # Long enough that early tokens fall out of the 1024-token sliding
    # window while global layers still see them: exercises the window mask
    # and >1K rope positions. Varied sentences (repetition would make the
    # tail distribution degenerate).
    (
        "long_window",
        " ".join(
            f"Paragraph {i}: the {w} {a} ran toward the {b}."
            for i, (w, a, b) in enumerate(
                (w, a, b)
                for w in ("quick", "lazy", "clever", "sturdy", "quiet")
                for a in ("fox", "badger", "otter", "lynx", "stoat", "hare")
                for b in ("river", "forest", "meadow", "burrow", "ridge")
            )
        )
        + " In summary, the animal that ran toward the river first was the",
        1100,
    ),
]


def req(path: str, payload: dict | None = None):
    url = f"http://127.0.0.1:{PORT}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=600)
    return json.loads(r.read())


def wait_healthy(proc, deadline_s: float = 300.0):
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        if proc.poll() is not None:
            sys.exit(f"llama-server exited early with {proc.returncode}")
        try:
            if req("/health").get("status") == "ok":
                return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(1.0)
    sys.exit("llama-server did not become healthy in time")


def main():
    version = subprocess.run(
        ["llama-server", "--version"], capture_output=True, text=True, check=False
    )
    version_str = (version.stderr + version.stdout).strip().splitlines()[0]

    proc = subprocess.Popen(
        [
            "llama-server",
            "--model", str(MODEL),
            "--port", str(PORT),
            "--ctx-size", "4096",
            "-ngl", "99",
            "--no-warmup",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_healthy(proc)
        records = []
        for name, text, min_tokens in PROMPTS:
            ids = req("/tokenize", {"content": text, "add_special": False})["tokens"]
            ids = [BOS] + ids
            if len(ids) < min_tokens:
                sys.exit(f"prompt `{name}`: only {len(ids)} tokens, need {min_tokens}")
            res = req(
                "/completion",
                {
                    "prompt": ids,
                    "n_predict": 1,
                    "n_probs": TOP_N,
                    # Neutral sampler chain + pre-sampling probs so the
                    # reported distribution is the raw softmax.
                    "temperature": 1.0,
                    "top_k": 0,
                    "top_p": 1.0,
                    "min_p": 0.0,
                    "post_sampling_probs": False,
                },
            )
            cp = res["completion_probabilities"][0]
            top = [
                {"id": t["id"], "logprob": t["logprob"]}
                for t in cp["top_logprobs"]
            ]
            records.append(
                {"name": name, "ids": ids, "top_logprobs": top}
            )
            print(f"{name}: {len(ids)} tokens, top id {top[0]['id']} "
                  f"(logprob {top[0]['logprob']:.4f})")

        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("w") as f:
            f.write(json.dumps({"generator": version_str, "top_n": TOP_N}) + "\n")
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {OUT}")
    finally:
        proc.terminate()
        proc.wait(timeout=30)


if __name__ == "__main__":
    main()
