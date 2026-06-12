//! CPU reference implementations with f64 accumulation (plan 02 §testing:
//! the oracle every kernel variant is validated against) plus shared test
//! plumbing (deterministic RNG, f16 round-trips, tolerance checks).

#![allow(dead_code)] // each test binary uses a subset

/// xorshift64* — deterministic across platforms, no dependency.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [-2, 2): activation-ish magnitudes.
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 4.0 - 2.0
    }

    pub fn f32_vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f32()).collect()
    }
}

/// Round through f16: GPU inputs are f16, so the reference must consume
/// exactly the values the kernel sees.
pub fn through_f16(xs: &[f32]) -> Vec<f32> {
    xs.iter()
        .map(|&x| half::f16::from_f32(x).to_f32())
        .collect()
}

pub fn to_f16_bits(xs: &[f32]) -> Vec<u16> {
    xs.iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect()
}

pub fn from_f16_bits(xs: &[u16]) -> Vec<f32> {
    xs.iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect()
}

/// Max combined error: |got−want| ≤ atol + rtol·|want|, reported with index.
pub fn assert_close(got: &[f32], want: &[f32], atol: f32, rtol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = (0usize, 0.0f32);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs() - rtol * w.abs();
        if err > worst.1 {
            worst = (i, err);
        }
    }
    let (i, err) = worst;
    assert!(
        err <= atol,
        "{what}: element {i}: got {} want {} (excess error {err:e}, atol {atol:e})",
        got[i],
        want[i]
    );
}

/// RMSNorm rows of `row_len`, f64 accumulation.
pub fn rmsnorm(x: &[f32], w: &[f32], row_len: usize, eps: f64, plus_one: bool) -> Vec<f32> {
    assert_eq!(w.len(), row_len);
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(row_len) {
        let ss: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let inv = 1.0 / (ss / row_len as f64 + eps).sqrt();
        for (i, &v) in row.iter().enumerate() {
            let weight = if plus_one {
                1.0 + w[i] as f64
            } else {
                w[i] as f64
            };
            out.push((v as f64 * inv * weight) as f32);
        }
    }
    out
}

/// The standard RoPE frequency table: `theta^(-2i/rot_dims)` (M3 swaps in
/// the GGUF `rope_freqs.weight` values / pinned `proportional` formula).
pub fn inv_freqs(rot_dims: usize, theta: f64) -> Vec<f64> {
    (0..rot_dims / 2)
        .map(|i| theta.powf(-2.0 * i as f64 / rot_dims as f64))
        .collect()
}

/// The CPU-filled cos/sin table the rope kernel consumes:
/// `[token × half_rot]` of interleaved (cos, sin), f64 math, f32 storage.
/// This function *is* the production table filler's reference semantics.
pub fn cos_sin_table(inv_freq: &[f64], start_pos: u32, tokens: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(tokens * inv_freq.len() * 2);
    for token in 0..tokens {
        let pos = (start_pos as usize + token) as f64;
        for &f in inv_freq {
            let (s, c) = (pos * f).sin_cos();
            out.push(c as f32);
            out.push(s as f32);
        }
    }
    out
}

/// Rotate-half RoPE over rows of [token × head × head_dim]: NEOX pairing
/// over the FULL head — pair `i` couples dims `(i, i + head_dim/2)` — with
/// only the first `rot_dims/2` pairs live (tabulated); the frozen tail
/// pairs are identities (docs/reference/gemma4-forward-graph.md).
pub fn rope(x: &mut [f32], head_dim: usize, rot_dims: usize, n_heads: usize, cos_sin: &[f32]) {
    let live = rot_dims / 2;
    let partner = head_dim / 2;
    for (row_idx, row) in x.chunks_exact_mut(head_dim).enumerate() {
        let token = row_idx / n_heads;
        for pair in 0..live {
            let c = cos_sin[(token * live + pair) * 2] as f64;
            let s = cos_sin[(token * live + pair) * 2 + 1] as f64;
            let a = row[pair] as f64;
            let b = row[pair + partner] as f64;
            row[pair] = (a * c - b * s) as f32;
            row[pair + partner] = (b * c + a * s) as f32;
        }
    }
}

/// Full-softmax attention for one (query token, query head) over the
/// inclusive key range [t0, t1], f64 throughout. GQA: query head `qh` reads
/// KV head `qh / (n_q_heads / n_kv_heads)`. Pass `v = k` for the K=V global
/// layers. Layouts match the kernels: q `[m × n_q_heads × head_dim]`,
/// k/v `[l × n_kv_heads × head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn attention_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    i: usize,
    qh: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    t0: usize,
    t1: usize,
) -> Vec<f32> {
    let kvh = qh / (n_q_heads / n_kv_heads);
    let qrow = &q[(i * n_q_heads + qh) * head_dim..][..head_dim];
    let scores: Vec<f64> = (t0..=t1)
        .map(|t| {
            let krow = &k[(t * n_kv_heads + kvh) * head_dim..][..head_dim];
            let dot: f64 = qrow
                .iter()
                .zip(krow)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum();
            dot * scale
        })
        .collect();
    let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
    let denom: f64 = exps.iter().sum();
    let mut out = vec![0.0f32; head_dim];
    for (d, o) in out.iter_mut().enumerate() {
        let acc: f64 = exps
            .iter()
            .zip(t0..=t1)
            .map(|(&e, t)| e * v[(t * n_kv_heads + kvh) * head_dim + d] as f64)
            .sum();
        *o = (acc / denom) as f32;
    }
    out
}

/// Q8_0 quantization in the kernels' structure-of-arrays page format,
/// mirroring the GPU op-for-op in f32: per 32 weights, d = amax·(1/127)
/// (f32, stored f16; a multiply so it is bit-identical to the GPU — FDiv
/// is not), q = round_ties_even(x · 1/d). The 1/d reciprocal IS a
/// division: the GPU may differ by an ulp, flipping a quant by ±1 at exact
/// rounding boundaries (the parity test allows it). Returns (f16 scale
/// bits, i8 quants as bytes).
pub fn quant_q8_0(x: &[f32]) -> (Vec<u16>, Vec<u8>) {
    assert!(x.len().is_multiple_of(32));
    let mut scales = Vec::with_capacity(x.len() / 32);
    let mut quants = Vec::with_capacity(x.len());
    for block in x.chunks_exact(32) {
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax * (1.0 / 127.0);
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        scales.push(half::f16::from_f32(d).to_bits());
        for &v in block {
            quants.push((v * id).round_ties_even() as i8 as u8);
        }
    }
    (scales, quants)
}

/// Q8_0 dequantization mirroring the GPU kernel: x = f16(f16_d · q).
pub fn dequant_q8_0(scales: &[u16], quants: &[u8]) -> Vec<f32> {
    assert_eq!(quants.len(), scales.len() * 32);
    let mut out = Vec::with_capacity(quants.len());
    for (blk, &s) in scales.iter().enumerate() {
        let d = half::f16::from_bits(s).to_f32();
        for &q in &quants[blk * 32..][..32] {
            out.push(half::f16::from_f32(d * (q as i8) as f32).to_f32());
        }
    }
    out
}

/// gelu_pytorch_tanh(gate) * up.
pub fn geglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(&g, &u)| {
            let g = g as f64;
            let inner = (2.0 / std::f64::consts::PI).sqrt() * (g + 0.044715 * g * g * g);
            (0.5 * g * (1.0 + inner.tanh()) * u as f64) as f32
        })
        .collect()
}
