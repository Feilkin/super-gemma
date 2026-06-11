# 05 — Server & Anthropic-Style API (`sg-server`)

## Scope

axum HTTP server exposing an Anthropic-compatible surface: `POST /v1/messages` (blocking + SSE
streaming), `POST /v1/messages/count_tokens`, API-key auth, FIFO request queue with timeout,
config, and observability. Compatible enough that standard Anthropic SDKs and coding agents work
by switching base URL + key.

## Endpoints

### `POST /v1/messages`

- Request: `model` (accepted, validated against configured alias list, otherwise ignored),
  `system`, `messages` (text content blocks; `tool_use`/`tool_result` blocks for agent loops),
  `tools`, `tool_choice` (`auto`/`any`/`tool`/`none`), `max_tokens` (required), `stop_sequences`,
  `temperature`, `top_p`, `top_k`, `stream`, `metadata`. **Image blocks → 400** with a clear error
  (text-only server).
- Response (non-stream): Anthropic message shape — `id`, `type:"message"`, `role:"assistant"`,
  `content` (text and/or tool_use blocks), `stop_reason` (`end_turn`|`max_tokens`|`stop_sequence`|
  `tool_use`), `stop_sequence`, `usage` {input_tokens, output_tokens, plus extension fields:
  `cache_read_input_tokens` style reporting for cache2 hits — agents and we both want to see this}.
- Streaming: exact Anthropic SSE event sequence — `message_start`, `content_block_start`,
  `content_block_delta` (`text_delta` / `input_json_delta` for tool args), `content_block_stop`,
  `message_delta` (stop_reason + usage), `message_stop`; `ping` keepalives; `error` events.
  Client disconnect → abort signal into the engine (plan 03), cache writes still flushed.

### `POST /v1/messages/count_tokens`

Tokenizer-only (PromptBuilder → count). Bypasses the FIFO queue entirely; sub-millisecond,
unlimited concurrency.

### Tool use mapping

- `tools` (name, description, input_schema) rendered into Gemma 4's tool-declaration prompt format;
  `tool_choice` mapped as the template allows (if Gemma 4 has no forced-call mechanism, emulate
  `any`/`tool` via constrained retry-or-error and document the gap).
- Model output runs through the incremental tool-call parser (plan 01): tool-call text →
  `tool_use` blocks with streamed `input_json_delta` events; `stop_reason:"tool_use"`.
  Malformed JSON from the model → configurable: pass through as text or fail the block (default:
  pass through as text, log warning).
- `tool_result` blocks in subsequent requests render back into the template's expected format.

## Queueing & lifecycle

- Global FIFO: `tokio::sync::mpsc` of accepted requests; one engine consumer. Config:
  `max_queue_depth` (default 8), `queue_timeout_ms` (default 30 000). Overflow / timeout → 429
  with `retry-after` and Anthropic-style error body (`overloaded_error`).
- Per-request timeout (`max_request_ms`, generous default) → abort + `error` SSE event.
- Graceful shutdown: stop accepting, drain or abort in-flight with cache flush, persist cache2
  index, exit.

## Auth & hardening

- `x-api-key` header (also accept `Authorization: Bearer` for SDK friendliness). Keys in config as
  argon2 hashes; constant-time verify; per-key name for logs. 401/403 Anthropic-style error bodies.
- `anthropic-version` header accepted and echoed, not enforced (log unknown values).
- Request body limit (configurable, default 32 MB — coding-agent prompts are big), JSON depth
  limits, strict-but-tolerant deserialization (unknown fields ignored + logged once).
- Bind address config; TLS terminated by a reverse proxy if needed (out of scope — LAN service).
  No CORS by default.

## Config (TOML + env overrides)

`[server]` bind, body limits, timeouts, queue; `[auth]` keys; `[model]` gguf path, context limit,
model alias names; `[cache2]` (plan 04 surface); `[gpu]` probed-feature overrides, prefill chunk;
`[log]` level, format. Single binary `sg-server --config /etc/super-gemma.toml`; systemd unit
example in `deploy/`.

## Observability

- `tracing` + JSON logs; per-request span: key name, tokens in/out, cache2 (matched/resumed/saved
  tokens), TTFT, decode tok/s, queue wait.
- `GET /metrics` (Prometheus, no auth on loopback / key-gated otherwise): request counts/latencies,
  queue depth, TTFT/tok-s histograms, cache2 hit ratio, bytes loaded/stored, evictions, NVMe IO,
  GPU step time, resident KV bytes.
- `GET /healthz` (no model touch), `GET /readyz` (model loaded, GPU responsive — submits a trivial
  kernel with timeout).

## Implementation steps

1. axum skeleton + config + auth + error-body conventions + healthz; integration-test harness with
   a `MockEngine` trait implementation (scripted token streams) — the entire API layer is testable
   without a GPU.
2. /v1/messages non-streaming against MockEngine; serde types golden-tested against captured
   real Anthropic API fixtures (request and response shapes).
3. SSE streaming: event encoder unit-tested against Anthropic SDK *as client* (the official Rust /
   TS SDK pointed at the test server must parse every stream we emit).
4. Queue + timeouts + abort propagation; count_tokens.
5. Tool-use round-trip against MockEngine (declaration render is plan 01; here: block assembly,
   delta framing, stop_reason).
6. Metrics/tracing; graceful shutdown; systemd packaging.
7. Swap MockEngine → real engine (M7); end-to-end agent smoke test (run a real coding agent
   against it with base-URL override).

## Testing & validation

- **API conformance:** fixture suite of real Anthropic request/response/SSE captures; byte-level
  SSE framing tests (event order, data: JSON shapes, ping cadence); official SDKs as test clients
  in CI (TS + Python: send/stream/tool-loop/count_tokens against MockEngine server).
- **Queue:** concurrency tests (N parallel clients → strict FIFO order observed via MockEngine
  timestamps; overflow → 429; timeout → 429; disconnect mid-queue → dropped cleanly).
- **Auth:** wrong/missing/malformed keys; timing-safety smoke (statistical, coarse).
- **Robustness:** fuzz JSON bodies (arbitrary + structure-aware) — never panic, always valid error
  body; slowloris/partial-body timeouts; huge prompts at the body limit.
- **Soak (M9):** 24 h loop of a scripted agent (tool calls, parallel subagent bursts, aborts)
  against the real engine; zero leaks (RSS/VRAM flat), zero 5xx, cache hit-rate sane.
- **Benchmarks:** queue/SSE overhead vs raw engine (target: < 2 ms added TTFT, negligible per-token
  cost), count_tokens throughput.
