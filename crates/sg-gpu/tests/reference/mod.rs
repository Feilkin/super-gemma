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

/// Q8_0 quantize f32 in 32-blocks (matches `kv_quant_q8`): returns the i8 quants
/// packed 4 per u32, the f16 scale bits (one per 32-block), and the dequant f32
/// using the f16-rounded scale (the values the int8 kernel actually multiplies).
/// `vals.len()` must be a multiple of 32.
pub fn q8_quant(vals: &[f32]) -> (Vec<u32>, Vec<u16>, Vec<f32>) {
    assert!(vals.len().is_multiple_of(32));
    let nblk = vals.len() / 32;
    let mut quants = vec![0u32; vals.len() / 4];
    let mut scales = vec![0u16; nblk];
    let mut deq = vec![0f32; vals.len()];
    for b in 0..nblk {
        let s = b * 32;
        let amax = vals[s..s + 32].iter().fold(0f32, |a, &x| a.max(x.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let d16 = half::f16::from_f32(d);
        scales[b] = d16.to_bits();
        let dd = d16.to_f32();
        for i in 0..32 {
            let qi = ((vals[s + i] * id).round_ties_even() as i32).clamp(-128, 127) as i8;
            quants[(s + i) / 4] |= ((qi as u8 as u32) & 0xFF) << (((s + i) % 4) * 8);
            deq[s + i] = dd * qi as f32;
        }
    }
    (quants, scales, deq)
}

/// Per-KEY Q8 quantize of V for int8 PV (Piece B). Same Q8_0 math as
/// `q8_quant`, but the 32-blocks run along the KEY axis (the PV contraction)
/// instead of head-dim, so each block's scale factors out of the i8 key-dot
/// — the reason int8 PV is expressible at all (cf. docs/q8-kv-flash-impl.md §2:
/// V's natural head-dim quant does NOT align with the key contraction).
///
/// `v` is the natural `[keys × n_kv_heads × head_dim]` V (row-major, the order
/// `attention_head` reads). For each (head, head-dim column c) the `keys` are
/// grouped into `ceil(keys/32)` blocks of ≤32; one f16 amax scale per
/// (key-block, head, c). The i8 quants KEEP the `[keys × n_kv_heads × head_dim]`
/// layout of the f16 V they replace (and of the K quants), packed 4/u32 — only
/// the scales array is key-blocked. A short final block (keys not a multiple of
/// 32) is quantized over its live keys, as the kernel will with zero-padded P.
///
/// Returns `(quants u32-packed [keys·n_kv_heads·head_dim / 4], scales f16
/// [ceil(keys/32)·n_kv_heads·head_dim], dequant f32 in the input `[keys × …]`
/// order — the values the int8 PV kernel actually multiplies)`.
pub fn q8_quant_v(
    v: &[f32],
    keys: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> (Vec<u32>, Vec<u16>, Vec<f32>) {
    assert_eq!(v.len(), keys * n_kv_heads * head_dim);
    assert!(v.len().is_multiple_of(4));
    let row = n_kv_heads * head_dim;
    let kblk = keys.div_ceil(32);
    let mut quants = vec![0u32; v.len() / 4];
    let mut scales = vec![0u16; kblk * row];
    let mut deq = vec![0f32; v.len()];
    for h in 0..n_kv_heads {
        for c in 0..head_dim {
            for kb in 0..kblk {
                let k0 = kb * 32;
                let k1 = (k0 + 32).min(keys);
                let amax = (k0..k1).fold(0f32, |a, key| {
                    a.max(v[(key * n_kv_heads + h) * head_dim + c].abs())
                });
                let d = amax / 127.0;
                let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                let d16 = half::f16::from_f32(d);
                scales[(kb * n_kv_heads + h) * head_dim + c] = d16.to_bits();
                let dd = d16.to_f32();
                for key in k0..k1 {
                    let idx = (key * n_kv_heads + h) * head_dim + c;
                    let qi = ((v[idx] * id).round_ties_even() as i32).clamp(-128, 127) as i8;
                    quants[idx / 4] |= ((qi as u8 as u32) & 0xFF) << ((idx % 4) * 8);
                    deq[idx] = dd * qi as f32;
                }
            }
        }
    }
    (quants, scales, deq)
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

/// int8-MMQ reference — the EXACT arithmetic the int8 coopmat GEMM performs,
/// f64 accumulation. `Y[M×N] = X[M×K] · W[N×K]ᵀ` where X is quantized per
/// 32-element block to Q8_0 (`quant_q8_0`) and W is Q4_0 (row-major `[N×K]`
/// bytes). Per block β the dot is `d_a·d_w·Σ(aq·wq)` with `wq = nibble − 8`
/// and `aq` the Q8_0 i8 quant — an EXACT i32 inner product, scaled to f32.
///
/// Weight logical position `k = 32β + j` pairs with activation `X[m][k]`:
/// within a Q4_0 block, `j < 16` is the low nibble of `qs[j]`, `j ≥ 16` the
/// high nibble of `qs[j−16]` (the `BlockQ4_0::dequantize` layout). The kernel
/// must extract its i8 operands in this same logical order.
pub fn mmq_q4_0_q8(weights: &[u8], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    use sg_gguf::q4_0::{BLOCK_Q4_0_SIZE, blocks_from_bytes};
    assert!(k.is_multiple_of(32), "K must be a multiple of 32");
    let nb = k / 32;
    let row_bytes = nb * BLOCK_Q4_0_SIZE;
    let mut out = vec![0f32; m * n];
    for mi in 0..m {
        let (a_scales, a_quants) = quant_q8_0(&x[mi * k..][..k]);
        for ni in 0..n {
            let wblocks = blocks_from_bytes(&weights[ni * row_bytes..][..row_bytes]).unwrap();
            let mut acc = 0f64;
            for (b, wblk) in wblocks.iter().enumerate() {
                let da = half::f16::from_bits(a_scales[b]).to_f32() as f64;
                let dw = wblk.d.to_f32() as f64;
                let mut dot = 0i32;
                for j in 0..32usize {
                    let wq = if j < 16 {
                        (wblk.qs[j] & 0x0F) as i32 - 8
                    } else {
                        (wblk.qs[j - 16] >> 4) as i32 - 8
                    };
                    dot += (a_quants[b * 32 + j] as i8 as i32) * wq;
                }
                acc += da * dw * dot as f64;
            }
            out[mi * n + ni] = acc as f32;
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
