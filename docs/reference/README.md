# Reference material

Checked-in copies of external sources the plans were derived from, so nothing has to be
re-fetched or re-derived. If an upstream file changes, re-fetch, diff, and update the plans —
don't edit these by hand.

| File | Source | Fetched |
|---|---|---|
| `gemma-4-31b-it.config.json` | https://huggingface.co/google/gemma-4-31B-it/raw/main/config.json | 2026-06-09 |
| `gemma-4-31b-it-assistant.config.json` | https://huggingface.co/google/gemma-4-31B-it-assistant/raw/main/config.json (MTP drafter) | 2026-06-10 |
| `gemma-4-31b-q4_0.gguf-dump.txt` | generated on the target box by `cargo run -p sg-gguf --example dump` from the QAT GGUF below | 2026-06-11 |
| `model-checksums.txt` | sha256 of the downloaded model files on the target box | 2026-06-11 |
| `gemma4-forward-graph.md` | pinned forward-graph semantics, derived from transformers `gemma4` (main) + llama.cpp b9254 sources (resolves all plan 00 verify-items) | 2026-06-12 |

## Other sources referenced by the plans (not checked in)

- **Quantized model (the one we run):** https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf
  — same hyperparameters as the bf16 config above (confirmed by Ada).
- **Reference implementation** (resolves the "verify-against-reference" items in plan 00):
  transformers `src/transformers/models/gemma4/modeling_gemma4.py` (≥5.5 for the main model,
  ≥5.7 for `gemma4_assistant` / MTP, plan 07).
- **Chat/tool template** (plan 01): `tokenizer_config.json` of `google/gemma-4-31B-it`.
- **MTP announcement** (speedup claims, drafting overview):
  https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/

## Provenance notes

- The 31B config was fetched via a summarizing proxy; the JSON was requested verbatim, but if a
  value ever looks suspicious, re-fetch from the URL above before acting on it.
- Knowledge-cutoff warning for agents: Gemma 4, transformers 5.x, and the MTP drafter postdate
  early-2026 training data. Trust these files and the plans over model memory.
