//! In-memory incremental session (M5, plan 03 §engine flow with cache2
//! stubbed): one conversation's KV state stays resident across requests;
//! a new request reuses the longest common prefix of its prompt with the
//! resident history and prefills only the suffix.
//!
//! Scope: prefix reuse is all-or-nothing per request (full reset on
//! divergence) — the radix-trie partial reuse arrives with cache2 (M6).
//! Determinism contract (plan 06): replaying the same request sequence on
//! a reset session reproduces logits BIT-exactly (same path shapes ⇒ same
//! float ops). Resume vs cold recompute of the same conversation is only
//! tolerance-equal: prefill (gemm) and decode (gemv) produce KV rows that
//! differ in low bits by construction (plan 03 §testing).

use sg_gpu::{CommandGraph, GpuError};

use crate::generate::{EOS_TOKENS, GenerateParams, StopReason};
use crate::graph::GpuModel;
use crate::sampler::Sampler;

/// A single resident conversation over a [`GpuModel`].
pub struct Session<'a> {
    model: GpuModel<'a>,
    /// Tokens whose KV is resident; always `model.pos` long.
    history: Vec<u32>,
    decode_graph: CommandGraph,
}

impl<'a> Session<'a> {
    pub fn new(model: GpuModel<'a>) -> Result<Self, GpuError> {
        let decode_graph = model.record(0..model.desc.n_layers, true)?;
        Ok(Self {
            model,
            history: Vec::new(),
            decode_graph,
        })
    }

    /// Tokens currently resident (prompt + generated, all requests).
    pub fn history(&self) -> &[u32] {
        &self.history
    }

    /// Drop all resident state (the divergence path; also the test seam).
    pub fn reset(&mut self) {
        self.model.reset();
        self.history.clear();
    }

    /// Serve one request: `prompt` is the FULL conversation so far (the
    /// stateless-API shape). The resident prefix is reused; only the
    /// suffix is prefilled. Returns the sampled tokens and stop reason;
    /// `on_token` as in [`GpuModel::generate`].
    pub fn request(
        &mut self,
        prompt: &[u32],
        params: &GenerateParams,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<(Vec<u32>, StopReason), GpuError> {
        assert!(!prompt.is_empty());
        let common = self
            .history
            .iter()
            .zip(prompt)
            .take_while(|(a, b)| a == b)
            .count();
        // Reuse is append-only: rolling back to a mid-history position is
        // NOT possible in-memory — every position `p` past the rollback
        // point overwrote ring slot `p % 1024`, which belonged to position
        // `p − 1024`, inside the rollback point's attention window. (This
        // is exactly why cache2 keeps sliding-ring tail snapshots, plan
        // 04.) So: reset on divergence, and also when the prompt is fully
        // resident (a retry — its last token must re-run for logits, which
        // would be a 1-token rollback).
        if common < self.history.len() || self.history.len() >= prompt.len() {
            self.reset();
        }
        let resume = self.history.len();
        debug_assert!(resume < prompt.len());
        debug_assert_eq!(resume, self.model.pos as usize);

        let mut logits = self.model.prefill(&prompt[resume..])?;
        self.history.extend_from_slice(&prompt[resume..]);

        let mut sampler = Sampler::new(params.seed);
        let mut out = Vec::new();
        loop {
            let tok = sampler.sample(&logits, &params.sampling);
            out.push(tok);
            if !on_token(tok) {
                return Ok((out, StopReason::Aborted));
            }
            if EOS_TOKENS.contains(&tok) {
                return Ok((out, StopReason::Eos(tok)));
            }
            if out.len() >= params.max_tokens {
                return Ok((out, StopReason::MaxTokens));
            }
            logits = self.model.decode_step(&self.decode_graph, tok)?;
            self.history.push(tok);
        }
    }

    /// Last-token logits of the most recent forward pass (test seam).
    pub fn read_logits(&self) -> Result<Vec<f32>, GpuError> {
        self.model.read_logits()
    }
}
