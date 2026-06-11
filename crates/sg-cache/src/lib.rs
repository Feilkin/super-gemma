//! cache2: NVMe-backed paged radix-trie KV cache for global-attention layers,
//! sliding-ring tail snapshots, cost-aware eviction; plus sliding ring
//! bookkeeping shared with the GPU runtime.
//!
//! Scope and design: `docs/plans/04-kv-cache.md`. Trie lands in M5/M6.
//! All NVMe IO runs on a dedicated `tokio-uring` thread (Linux-only,
//! target-gated in Cargo.toml).

/// Inclusive bounds for the configurable cache2 page size, in tokens.
pub const PAGE_SIZE_TOKENS: std::ops::RangeInclusive<u32> = 16..=512;

/// Default cache2 page size, in tokens (256 tok x 40 KB/tok = 10 MB pages).
pub const DEFAULT_PAGE_SIZE_TOKENS: u32 = 256;
