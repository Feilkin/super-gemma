//! CPU sampling (plan 03 §sampling): f32 logits → temperature → top-k
//! (quickselect) → top-p prefix → categorical draw. Greedy when
//! temperature is 0. Deterministic given (params, seed, logits): the RNG
//! is a self-contained xoshiro256++ (no external crate whose algorithm
//! could drift across versions — same-seed-same-output is a tested
//! invariant, plan 06), and every float reduction runs in index order.
//!
//! Budget < 2 ms on the 262k vocab: the quickselect is O(n); only the
//! ≤ 1024 surviving candidates are sorted.

/// Sampling parameters (the Anthropic API surface; repetition penalties
/// are deliberately absent — not in the API, plan 03).
#[derive(Debug, Clone, Copy)]
pub struct SamplerParams {
    /// 0.0 = greedy (argmax; ties break to the lowest id).
    pub temperature: f32,
    /// 0 = disabled. Values are clamped to the plan's 1024 working-set cap
    /// only for the initial selection; top-p can expand past it.
    pub top_k: u32,
    /// 1.0 = disabled. Smallest prefix of descending-probability candidates
    /// whose cumulative probability reaches `top_p`.
    pub top_p: f32,
}

impl Default for SamplerParams {
    /// The model's shipped defaults (`general.sampling.*` GGUF metadata).
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 64,
            top_p: 0.95,
        }
    }
}

/// Seeded sampler; one per request (plan 03: per-request RNG seed).
pub struct Sampler {
    s: [u64; 4],
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        // splitmix64 expansion of the seed into xoshiro state (the
        // reference seeding procedure; never all-zero).
        let mut x = seed;
        let mut next = || {
            x = x.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        Self {
            s: [next(), next(), next(), next()],
        }
    }

    /// xoshiro256++.
    fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Draw one token id.
    pub fn sample(&mut self, logits: &[f32], p: &SamplerParams) -> u32 {
        assert!(!logits.is_empty());
        if p.temperature <= 0.0 {
            return argmax(logits);
        }

        // Candidate set: top-k by logit (temperature scaling is monotone,
        // so selection happens on raw logits). Working set is capped at
        // 1024 and expanded only if top-p needs more mass.
        let mut k = match p.top_k {
            0 => 1024,
            k => (k as usize).min(1024),
        }
        .min(logits.len());
        let top_k_active = p.top_k != 0;

        // With top-k active, top-p renormalizes within the top-k set (the
        // composed filter semantics, matching the HF warper order). With
        // top-k disabled, top-p is against the FULL distribution — its
        // mass must be computed over all logits, the 1024 working set is
        // only a candidate-selection optimization.
        let full_total: Option<f64> = (!top_k_active && p.top_p < 1.0).then(|| {
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let t = p.temperature as f64;
            logits.iter().map(|&v| ((v as f64 - max) / t).exp()).sum()
        });

        loop {
            let mut cand = select_top(logits, k);
            // Descending sort of the (small) candidate set; ties by id for
            // determinism.
            cand.sort_unstable_by(|&a, &b| {
                logits[b as usize]
                    .total_cmp(&logits[a as usize])
                    .then(a.cmp(&b))
            });

            // Softmax over the candidates at the given temperature, f64,
            // index order.
            let max = logits[cand[0] as usize] as f64;
            let t = p.temperature as f64;
            let probs: Vec<f64> = cand
                .iter()
                .map(|&id| ((logits[id as usize] as f64 - max) / t).exp())
                .collect();
            let total: f64 = probs.iter().sum();

            // Top-p prefix. The reference mass is the full distribution
            // when top-k is off, the top-k set's own mass when it's on.
            let mut keep = cand.len();
            if p.top_p < 1.0 {
                let target = p.top_p as f64 * full_total.unwrap_or(total);
                let mut acc = 0.0;
                let mut reached = false;
                for (i, &q) in probs.iter().enumerate() {
                    acc += q;
                    if acc >= target {
                        keep = i + 1;
                        reached = true;
                        break;
                    }
                }
                // The working set didn't hold the whole top-p prefix —
                // widen and retry (only reachable with top-k off).
                if !reached && !top_k_active && cand.len() < logits.len() {
                    k = (k * 4).min(logits.len());
                    continue;
                }
            }

            let total_kept: f64 = probs[..keep].iter().sum();
            let mut r = self.next_f64() * total_kept;
            for (i, &q) in probs[..keep].iter().enumerate() {
                r -= q;
                if r <= 0.0 {
                    return cand[i];
                }
            }
            return cand[keep - 1]; // float underflow tail
        }
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// Indices of the `k` largest logits (unordered), via quickselect.
fn select_top(logits: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    if k < idx.len() {
        idx.select_nth_unstable_by(k, |&a, &b| {
            logits[b as usize].total_cmp(&logits[a as usize])
        });
        idx.truncate(k);
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_is_argmax_with_low_id_ties() {
        let mut s = Sampler::new(7);
        let p = SamplerParams {
            temperature: 0.0,
            ..Default::default()
        };
        assert_eq!(s.sample(&[0.1, 3.0, 3.0, -1.0], &p), 1);
    }

    #[test]
    fn top_k_one_is_always_argmax() {
        let mut s = Sampler::new(7);
        let p = SamplerParams {
            temperature: 1.5,
            top_k: 1,
            top_p: 1.0,
        };
        for _ in 0..100 {
            assert_eq!(s.sample(&[0.0, 1.0, 5.0, 2.0], &p), 2);
        }
    }

    #[test]
    fn tiny_top_p_keeps_only_the_top_token() {
        let mut s = Sampler::new(7);
        let p = SamplerParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.01,
        };
        for _ in 0..100 {
            assert_eq!(s.sample(&[2.0, 1.0, 0.0, -1.0], &p), 0);
        }
    }

    #[test]
    fn same_seed_same_sequence() {
        let logits: Vec<f32> = (0..1000).map(|i| ((i * 37) % 17) as f32 * 0.3).collect();
        let p = SamplerParams::default();
        let draw = |seed: u64| -> Vec<u32> {
            let mut s = Sampler::new(seed);
            (0..50).map(|_| s.sample(&logits, &p)).collect()
        };
        assert_eq!(draw(42), draw(42));
        assert_ne!(draw(42), draw(43));
    }

    #[test]
    fn distribution_matches_expected_probabilities() {
        // 4 tokens with logits ln(1), ln(2), ln(3), ln(4) → probs 0.1,
        // 0.2, 0.3, 0.4. Chi-squared over 100k draws, 3 dof: critical
        // value at p=0.001 is 16.27.
        let logits = [0.0f32, 2.0f32.ln(), 3.0f32.ln(), 4.0f32.ln()];
        let expected = [0.1, 0.2, 0.3, 0.4];
        let n = 100_000usize;
        let mut counts = [0usize; 4];
        let mut s = Sampler::new(0xDEC0DE);
        let p = SamplerParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
        };
        for _ in 0..n {
            counts[s.sample(&logits, &p) as usize] += 1;
        }
        let chi2: f64 = counts
            .iter()
            .zip(&expected)
            .map(|(&c, &e)| {
                let exp = e * n as f64;
                (c as f64 - exp).powi(2) / exp
            })
            .sum();
        assert!(chi2 < 16.27, "chi² {chi2:.2} (counts {counts:?})");
    }

    #[test]
    fn temperature_sharpens_and_flattens() {
        let logits = [0.0f32, 1.0];
        let count_top = |temp: f32, seed: u64| {
            let mut s = Sampler::new(seed);
            let p = SamplerParams {
                temperature: temp,
                top_k: 0,
                top_p: 1.0,
            };
            (0..20_000).filter(|_| s.sample(&logits, &p) == 1).count()
        };
        let cold = count_top(0.25, 1); // p(top) = 1/(1+e^-4) ≈ 0.982
        let hot = count_top(4.0, 2); // p(top) = 1/(1+e^-0.25) ≈ 0.562
        assert!(cold > 19_300, "cold draws {cold}");
        assert!((10_600..11_900).contains(&hot), "hot draws {hot}");
    }

    #[test]
    fn top_p_expands_past_the_working_set_when_needed() {
        // 4096 equal logits, top_p 0.9: the prefix needs ~3687 tokens —
        // more than the 1024 working set, forcing expansion.
        let logits = vec![1.0f32; 4096];
        let mut s = Sampler::new(99);
        let p = SamplerParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.9,
        };
        let mut seen_high = false;
        for _ in 0..200 {
            if s.sample(&logits, &p) >= 1024 {
                seen_high = true;
                break;
            }
        }
        assert!(seen_high, "samples never escaped the initial working set");
    }

    #[test]
    fn explicit_top_k_caps_the_candidate_set() {
        // With top_k = 2, ids 2 and 3 (the two largest) are the only
        // possible outcomes even at high temperature.
        let logits = [1.0f32, 1.1, 5.0, 4.9];
        let mut s = Sampler::new(5);
        let p = SamplerParams {
            temperature: 10.0,
            top_k: 2,
            top_p: 1.0,
        };
        for _ in 0..200 {
            assert!(s.sample(&logits, &p) >= 2);
        }
    }
}
