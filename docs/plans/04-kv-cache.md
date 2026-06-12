# 04 — KV Caching: sliding ring, resident global KV, and cache2 (`sg-cache`)

## Scope

Three cooperating state stores:

1. **Sliding ring** (RAM/GPU): per sliding layer, a 1024-token ring of K and V. Fixed ~820 MB f16.
2. **Resident global KV** (RAM/GPU): the *active conversation's* global-layer KV for its full
   context — what `attn_decode_global` reads. 80 KB/token (separate K and V — M3 amendment, see `docs/reference/gemma4-forward-graph.md`), up to ~21 GB at 256K.
3. **cache2** (NVMe): persistent paged radix trie over token-id sequences storing global-layer KV
   pages + sliding-ring **tail snapshots**, enabling prefix resume across requests and restarts.

## Why tail snapshots are required (design-critical)

Global KV pages alone cannot shortcut prefill. To resume generation at position N you need the
sliding-layer state (last 1024 tokens × 50 layers) *at* N — and reconstructing it exactly requires
re-running the full prefill, because each layer's sliding KV depends on previous hidden states
recursively. Cached global KV doesn't reduce that recomputation at all (the matmuls still have to
run to produce hidden states).

Therefore cache2 stores two artifact kinds:

- **Global KV pages** (radix-trie nodes): deduplicated storage of per-token global KV across
  conversations sharing prefixes. Needed *in full* for decode once resumed.
- **Tail snapshots**: the complete sliding-ring state captured at position N (plus RNG-irrelevant
  metadata: N, ring head, trie node ref). A snapshot at N + global pages covering [0, N) = exact
  resume at N. f16 ~820 MB, Q8_0 ~410 MB each (configurable; default Q8_0 pending M6 quality check).

**Capture policy:** snapshots are cheap to take during prefill/decode (device-local copy of the
ring, then async flush). Capture at: (a) end of every completed generation (the append-only
coding-agent case: next request extends this exact point); (b) message boundaries during prefill
when the position corresponds to an existing trie branch point (a second conversation diverging
there — e.g. shared system prompt across subagents — makes that boundary snapshot-worthy);
(c) optional periodic checkpoint every `snapshot_interval` tokens (default off).

Resume rule: longest prefix match in the trie, then walk back to the deepest **snapshot ≤ match
length**; resume there, prefill the rest. A branch with pages but no usable snapshot only saves
storage (dedup), not compute — the eviction cost model must value it accordingly.

## cache2 design

### Radix trie (in RAM)

- Keys: token-id sequences. Edges hold token spans; nodes own **pages** of global KV.
- Page = `page_size` tokens (config 16–512, default 256), all 10 global layers bundled:
  256 tok × 80 KB = 20 MB/page (K and V), one contiguous NVMe extent → single large sequential read.
- Node = { edge tokens, page refs, child map, snapshot refs, stats (last_use, hit_count,
  created_at, bytes, tokens) }.
- Insert on commit: split nodes at divergence points (page-aligned splits preferred; a split
  mid-page rewrites that one page's tail — pages are immutable otherwise).
- Lookup is pure CPU (token compare), O(prompt length); budget < 1 ms for 100K tokens.

### NVMe layout

- One preallocated extent file (`cache2.dat`, size = configured max, e.g. 200–500 GB) +
  extent allocator (page-sized slabs per artifact class; snapshots use large extents).
  O_DIRECT, 4 KiB-aligned everything; all IO via the dedicated tokio-uring thread.
- **Index file** (`cache2.idx`): periodic snapshot of the trie + extent map (serialized, checksummed,
  double-buffered A/B writes) + small append-only journal between snapshots. On startup: load last
  good index, replay journal; any inconsistency → **drop the cache and start cold** (it's a cache;
  correctness over salvage). Page/snapshot payloads carry checksums (xxh3) verified on load.
- Writes happen during/after generation (page flush per completed page, snapshot at end) on the
  uring thread, never blocking decode. fsync policy: journal fsync'd at commit; payload fdatasync
  before its journal entry (write-ahead: payload → journal → visible).

### Eviction (cost-aware LRU)

Triggered when allocator pressure > high-watermark; evicts until low-watermark. Score per evictable
unit (leaf-ward subtree or individual snapshot) — evict lowest:

```
value = (recompute_cost − load_cost) × P(reuse)
recompute_cost ≈ tokens_saved / prefill_tok_s          (measured, plan 06)
load_cost      ≈ bytes / nvme_read_bw + fixed_latency  (measured at startup)
P(reuse)       ≈ recency decay (exponential over last_use) × hit_count factor
```

- `tokens_saved` for a subtree counts only tokens actually resumable through a surviving snapshot —
  pages above the deepest snapshot of any live descendant contribute storage-dedup value only
  (weighted low).
- Invariants: never evict pages in [0, N) while keeping a snapshot at N that depends on them
  (snapshot without its pages is useless — evict snapshot first); never evict artifacts of the
  in-flight conversation.
- Constants (decay half-life, dedup weight) are config with sane defaults; the simulator (below)
  tunes them.

### Concurrency & lifecycle

- Single-writer (one conversation in flight) simplifies everything: trie mutations only from the
  engine task; uring thread does pure IO against immutable extents; eviction runs between requests
  (or async for snapshot-only eviction). An `Arc<CacheHandle>` channel API; no locks on hot paths.
- Resident global KV ↔ cache2: on resume, pages are read NVMe → GPU-visible resident buffer
  (Q8_0 pages pass through the dequant kernel, plan 02). On commit, resident KV is sliced into
  pages → (optional Q8_0 quant kernel) → NVMe.

## Sliding ring details

- Per sliding layer: K-ring and V-ring, 1024 × 16 heads × 256 × f16; head pointer per conversation
  (uniform-driven, plan 02). Wraparound handled in kernels via modular indexing.
- Snapshot capture = GPU copy of all 50 layers' rings + heads into a staging region → async NVMe
  flush. Restore = reverse. ~820 MB f16 ≈ 160 ms NVMe at 5 GB/s (Q8_0 ≈ 80 ms) — dwarfed by the
  prefill it replaces (minutes at 100K tokens).

## Config surface

`page_size_tokens` (16–512), `max_cache_bytes`, `kv_disk_format` (f16|q8_0), `snapshot_format`,
`snapshot_interval` (0=off), watermarks, `cache_dir`, `eviction_half_life`, `nvme_*` overrides
(else measured at startup). All in the server TOML (plan 05).

## Implementation steps

1. Pure in-memory trie + lookup/insert/split + property tests (no IO, no GPU — fast unit cycle).
2. Extent allocator + uring IO layer (read/write/checksum/journal) against a temp file; fault-
   injection harness (short writes, torn pages via kill points, bitflips → checksum catch).
3. Index snapshot/journal/recovery cycle; crash-consistency test rig (kill -9 matrix).
4. Eviction scorer + **discrete-event simulator**: replay synthetic agent workloads (append-only
   chains, subagent fan-out from shared prefixes, abandoned branches) over a small fake cache;
   assert hit-rate beats plain-LRU baseline; tune constants.
5. GPU integration: resident-KV slicing, Q8_0 round-trip, snapshot capture/restore (needs M5).
6. End-to-end: engine resume path, commit path, abort-flush path (M6 gate).

## Testing & validation

- **Trie:** property tests (random insert/lookup/split/evict sequences vs a naive reference map;
  invariants: prefix correctness, page coverage exactness, no orphan extents), fuzzing.
- **Exactness:** resume-vs-cold **bit-identical logits** (f16 path; Q8_0 path: KL(logits) below
  threshold and perplexity delta < 0.1 % — measured in M6, gates the Q8_0 default).
- **Crash consistency:** scripted kill -9 at every journal/payload write stage × 100 randomized
  runs; recovery must yield either a consistent cache or a clean cold start, never wrong KV.
  (Wrong-KV = silent garbage generation — the worst failure mode in this system; checksums +
  write-ahead ordering are both load-bearing.)
- **Eviction:** simulator regression suite with fixed workload seeds; invariant checks
  (snapshot/page dependency rule) as runtime debug_asserts + tests.
- **Benchmarks:** lookup latency vs trie size; NVMe page-load throughput (queue depth sweep);
  snapshot save/restore wall time; warm TTFT vs cold TTFT at {1K, 8K, 32K, 100K} prefix
  (headline numbers, plan 06); sustained write bandwidth during decode (must not perturb decode
  latency — measure jitter with and without background flush).
