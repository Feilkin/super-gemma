//! Token-level generation loop (plan 03 §decode loop, M4 shape): prefill
//! the prompt in chunks, then sample-decode until EOS / `max_tokens` /
//! caller abort. Text-level concerns (streaming detokenization,
//! stop-sequence matching) deliberately live with the tokenizer's
//! consumers — this crate stays tokenizer-free; the engine (M5+) layers
//! the abort/cache plumbing on top of the same loop shape.

use sg_gpu::GpuError;

use crate::graph::GpuModel;
use crate::sampler::{Sampler, SamplerParams};

/// `<eos>` and `<end_of_turn>` (config.json `eos_token_id`, plan 03).
pub const EOS_TOKENS: [u32; 2] = [1, 106];

#[derive(Debug, Clone, Copy)]
pub struct GenerateParams {
    pub sampling: SamplerParams,
    pub seed: u64,
    pub max_tokens: usize,
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Eos(u32),
    MaxTokens,
    Aborted,
}

impl GpuModel<'_> {
    /// Generate up to `max_tokens` tokens after `prompt`. `on_token` is
    /// called for every sampled token (including a terminating EOS);
    /// returning `false` aborts — the client-disconnect seam (plan 03).
    /// Returns the sampled tokens and the stop reason.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        params: &GenerateParams,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<(Vec<u32>, StopReason), GpuError> {
        assert!(!prompt.is_empty());
        let decode_graph = self.record(0..self.desc.n_layers, true)?;
        let mut sampler = Sampler::new(params.seed);
        let mut out = Vec::new();

        let mut logits = self.prefill(prompt)?;
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
            logits = self.decode_step(&decode_graph, tok)?;
        }
    }
}
