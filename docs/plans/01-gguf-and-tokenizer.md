# 01 — GGUF Loading & Tokenizer (`sg-gguf`, `sg-tokenizer`)

## Scope

Parse the Gemma 4 31B QAT Q4_0 GGUF file, expose typed metadata and zero-copy tensor views, get
weights into GPU-visible memory with no redundant copies, and implement a SentencePiece-compatible
tokenizer plus the Gemma 4 chat/tool-call template, all from data embedded in the GGUF.

## `sg-gguf` design

### Parser

- mmap the file read-only (`memmap2`). GGUF v3: magic `GGUF`, version, tensor count, metadata KV
  count; typed KV section; tensor table (name, dims, ggml type, offset); data section aligned to
  `general.alignment` (default 32).
- Zero-copy: tensor views are `&[u8]` slices into the mmap with typed accessors. No deserialization
  of tensor data at parse time.
- Support exactly the ggml types this model ships: expect `Q4_0` for matmul weights, `F32`/`F16` for
  norms, possibly `Q8_0`/`F16` for token embeddings. The loader **enumerates the actual tensor table
  first** and fails with a clear diff if it sees anything else — we support what the file contains,
  not the whole ggml zoo.
- `Q4_0` block: 32 weights, 18 bytes = f16 scale `d` + 16 nibble bytes; dequant `w = d·(q−8)`.
  Define `#[repr(C)] BlockQ4_0` and the same struct mirrored in WGSL (plan 02).

### Model description

`ModelDesc` is built by cross-checking three sources, in priority order:

1. GGUF metadata keys (`gemma4.*`: layer count, head counts, dims, window, rope params, layer types).
2. Actual tensor shapes from the tensor table (e.g. confirms global q_proj is 5376→16384, confirms
   missing `v_proj` on global layers if K=V is materialized as a single tensor — **the export might
   instead ship duplicate K/V tensors; handle both**).
3. Hard-coded expectations from config.json (see plan 00 table).

Any disagreement is a startup error with a printed three-way diff. This is the primary defense
against "QAT GGUF doesn't match the bf16 config" surprises.

### Weight upload (Linux target)

- Allocate one big vulkano buffer (HOST_VISIBLE|DEVICE_LOCAL, sub-allocated per tensor with
  alignment for the kernels' access patterns).
- Load path A (preferred): tokio-uring `read_at` with O_DIRECT directly into the mapped GPU pointer
  (registered buffers). Load path B (fallback, behind the same trait): mmap + `memcpy` into mapped
  buffer. M0 probe decides; on unified memory even path B is a one-time ~2 s cost for 18 GB.
- Weights are immutable after load; no further IO involvement.

## `sg-tokenizer` design

- Vocab 262 144, SentencePiece. Source of truth: GGUF keys `tokenizer.ggml.model` (expect
  `llama`-style SPM), `tokens`, `scores`, `token_type`, plus BOS=2, EOS=1, `<end_of_turn>`=106,
  PAD=0 (validate against metadata, don't hardcode silently).
- Implement SPM **unigram** encode: trie over vocab + Viterbi best-path segmentation with byte
  fallback, standard SPM normalization (whitespace → `▁`). Decode: concatenate pieces, reverse
  normalization, byte-token handling.
- Special tokens (`<start_of_turn>`, `<end_of_turn>`, control tokens) are matched before SPM
  segmentation and never produced by it.
- Streaming-safe decode: a `DetokBuffer` that withholds bytes until UTF-8 complete (multi-token
  codepoints) — needed for SSE deltas.

### Chat template & tool-call format

- Fetch `tokenizer_config.json` chat template for `gemma-4-31B-it` once, during implementation;
  hand-port to Rust (`PromptBuilder`). Covers: system prompt handling, `<start_of_turn>user/model`
  framing, **tool declaration block, tool-call output format, tool-result injection format** —
  whatever Gemma 4's convention is (likely JSON-in-template like Gemma 3; pin down from the actual
  template, do not assume).
- Golden tests: a corpus of (messages, tools) inputs run through HF `apply_chat_template` offline
  (script checked into `tools/`), outputs stored as fixtures; Rust builder must match byte-for-byte.
- The tool-call **parser** (model output → structured tool_use) lives here too, as a streaming
  incremental parser (plan 05 consumes it).

## Implementation steps

1. GGUF parser + metadata typing + tensor table (fixture-driven, synthetic mini-GGUFs).
2. `ModelDesc` three-way validation against the real file (needs the downloaded GGUF; also dump
   `gguf-dump`-style report for the repo docs).
3. Q4_0 block types + scalar dequant reference (used by kernel tests in plan 02).
4. SPM unigram tokenizer + special tokens + streaming detok.
5. Chat template port + tool-call parser.
6. Weight upload path behind `WeightSource` trait (uring impl on Linux; mmap+copy fallback).

## Testing & validation

- **Parser:** synthetic GGUF fixtures (hand-built minimal files: every KV type, alignment edge
  cases, truncated-file fuzzing via `cargo-fuzz` — parser must never panic, only error).
- **Real-file smoke test** (target box / WSL with the model downloaded): parse, validate, checksum
  a few tensors against `gguf` Python tooling output.
- **Tokenizer parity:** golden corpus (≥10k lines: code in several languages, multilingual text,
  emoji, whitespace pathologies, long tokens) encoded offline with HF tokenizer → fixtures; require
  100 % id-sequence match and round-trip decode match. Property test: decode(encode(s)) == s for
  arbitrary UTF-8.
- **Template parity:** byte-exact vs HF for the golden (messages, tools) corpus, including
  multi-turn with tool results.
- **Benchmarks (criterion):** tokenizer throughput (target: >1 M tok/s encode — it sits on the
  count_tokens endpoint and prefill path), GGUF parse time, 18 GB weight-load wall time on target.

## Open questions

- Does the QAT GGUF ship embeddings quantized (Q4_0/Q8_0) or f16? (Tied LM head means the same
  tensor feeds the final matmul — kernel choice in plan 02 depends on this.) → resolved by tensor
  table at first real-file parse.
- Gemma 4 tool-calling convention: native special tokens vs JSON-in-text → resolved when porting
  the chat template; affects parser in this crate and API mapping in plan 05.
