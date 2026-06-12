//! RoPE cos/sin table construction — the ONE place the pinned rotary
//! semantics live (`docs/reference/gemma4-forward-graph.md`). Both the CPU
//! reference model and the GPU graph consume tables built here; the M2 rope
//! kernels deliberately contain no angle math (GPU trig loses ~1e-2 by
//! position 100K).
//!
//! Pinned semantics (NEOX / rotate-half, both layer types):
//! pair `i` couples dims `(i, i + head_dim/2)` for `i < head_dim/2`, rotated
//! by `θᵢ = pos · theta^(−2i/head_dim) / factor[i]`.
//!
//! - Sliding layers: head_dim 256, θ base 10 000, no factors (all 1.0) —
//!   full rotation.
//! - Global layers: head_dim 512, θ base 1 000 000, factors =
//!   `rope_freqs.weight` from the GGUF (`[1.0 ×64, 1e30 ×192]`): ggml-style
//!   *divisors*, so pairs 64.. get θ ≈ 0 — identity, which is how
//!   `partial_rotary_factor 0.25` materializes ("proportional" RoPE).
//!
//! Angles are computed in f64 and stored as (cos, sin) f32 — the layout the
//! M2 `rope` kernels bind (`cos_sin[token * half + pair]`, `vec2<f32>`).

/// Build the `(cos, sin)` table for `positions`, flattened as
/// `[position × (head_dim/2) × 2]` f32.
///
/// `factors` are per-pair frequency *divisors* (ggml `freq_factors`);
/// `None` means all 1.0. Panics if `factors` length ≠ `head_dim / 2` —
/// caller bugs, not data errors (the GGUF table is validated at load).
pub fn cos_sin_table(
    positions: &[u32],
    head_dim: usize,
    theta: f32,
    factors: Option<&[f32]>,
) -> Vec<f32> {
    let half = head_dim / 2;
    if let Some(f) = factors {
        assert_eq!(f.len(), half, "freq factors length != head_dim/2");
    }
    let mut out = Vec::with_capacity(positions.len() * half * 2);
    for &pos in positions {
        for i in 0..half {
            let freq = (theta as f64).powf(-2.0 * i as f64 / head_dim as f64);
            let factor = factors.map_or(1.0, |f| f[i] as f64);
            let angle = pos as f64 * freq / factor;
            out.push(angle.cos() as f32);
            out.push(angle.sin() as f32);
        }
    }
    out
}

/// Rotate one head row in place using a table built by [`cos_sin_table`]
/// for that row's position: `cs` is the `[head_dim/2 × 2]` slice for the
/// position. f32 math, NEOX pairing — mirrors the M2 `rope` kernel
/// element-for-element.
pub fn apply_rope(row: &mut [f32], cs: &[f32]) {
    let half = row.len() / 2;
    debug_assert_eq!(cs.len(), half * 2);
    for i in 0..half {
        let (cos, sin) = (cs[2 * i], cs[2 * i + 1]);
        let a = row[i];
        let b = row[i + half];
        row[i] = a * cos - b * sin;
        row[i + half] = b * cos + a * sin;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pos_zero_is_identity() {
        let t = cos_sin_table(&[0], 256, 10_000.0, None);
        assert_eq!(t.len(), 256);
        for i in 0..128 {
            assert_eq!(t[2 * i], 1.0, "cos pair {i}");
            assert_eq!(t[2 * i + 1], 0.0, "sin pair {i}");
        }
    }

    #[test]
    fn first_pair_rotates_by_pos_radians() {
        // Pair 0 has frequency theta^0 = 1: angle = pos exactly.
        let t = cos_sin_table(&[3], 256, 10_000.0, None);
        assert_eq!(t[0], (3.0f64).cos() as f32);
        assert_eq!(t[1], (3.0f64).sin() as f32);
    }

    #[test]
    fn proportional_factors_freeze_tail_pairs() {
        // Global-layer shape: 256 pairs, first 64 live, rest divided by 1e30.
        let mut factors = vec![1.0f32; 64];
        factors.extend(std::iter::repeat_n(1e30f32, 192));
        let t = cos_sin_table(&[100_000], 512, 1_000_000.0, Some(&factors));
        // Pair 64 would rotate fast without the factor; with it, identity.
        for i in 64..256 {
            assert_eq!(t[2 * i], 1.0, "cos pair {i}");
            assert!(t[2 * i + 1].abs() < 1e-15, "sin pair {i}");
        }
        // Pair 0: angle = pos / 1 = 100000 rad.
        assert_eq!(t[0], (100_000.0f64).cos() as f32);
        // Pair 63 uses exponent -2*63/512 of 1e6 — matches the formula.
        let angle = 100_000.0f64 * (1e6f64).powf(-2.0 * 63.0 / 512.0);
        assert_eq!(t[2 * 63], angle.cos() as f32);
    }

    #[test]
    fn apply_rope_neox_pairing() {
        // head_dim 4: pairs (0,2) and (1,3). Rotate pair 0 by 90°.
        let cs = [0.0, 1.0, 1.0, 0.0]; // pair0: cos 0 sin 1; pair1: identity
        let mut row = [1.0, 5.0, 2.0, 7.0];
        apply_rope(&mut row, &cs);
        assert_eq!(row, [-2.0, 5.0, 1.0, 7.0]);
    }
}
