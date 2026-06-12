// Rotary position embedding, rotate-half (NeoX) convention, in place over
// rows of [token × head × HEAD_DIM] f16. One thread per (row, rotation pair).
//
// cos/sin come from a per-chunk table `cos_sin[token * HALF_ROT + pair]`
// filled by the CPU in f64 (≤ 512 KB per prefill chunk, 1 KB per decode
// step). GPU-side angle math is deliberately absent: hardware sin/cos loses
// ~1e-2 absolute accuracy by position 100K (f32 range reduction), while a
// CPU-filled table is exact, bit-deterministic, and the single place where
// M3 pins the `proportional` formula / `rope_freqs.weight` semantics
// (plan 00 verify-items).
//
// Variants per matmul site: sliding q/k (HEAD_DIM 256, full rotation) and
// global q/k (HEAD_DIM 512, ROT_DIMS 128 = partial_rotary 0.25).
//
// Pairing is NEOX over the FULL head: pair i couples dims (i, i+HEAD_DIM/2),
// and only the first ROT_DIMS/2 pairs rotate — the reference's frozen tail
// pairs (rope_freqs divisor 1e30 → θ≈0) are skipped as exact identities
// (docs/reference/gemma4-forward-graph.md, amended 2026-06-12: the original
// M2 variant paired (i, i+ROT_DIMS/2), which is wrong for partial rotation).
// The cos_sin table carries only the live pairs.
//
// QK-norm fusion (plan 02 option) is deferred with rmsnorm's residual fusion.

enable f16;

@group(0) @binding(0) var<storage, read_write> xs: array<f16>;
/// `[token × HALF_ROT]` of (cos, sin).
@group(0) @binding(1) var<storage, read> cos_sin: array<vec2<f32>>;

const HEAD_DIM: u32 = #{HEAD_DIM}u;
const ROT_DIMS: u32 = #{ROT_DIMS}u;
const N_HEADS: u32 = #{N_HEADS}u;
const HALF_ROT: u32 = ROT_DIMS / 2u;

@compute @workgroup_size(#{WG_X})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pairs_total = arrayLength(&xs) / HEAD_DIM * HALF_ROT;
    if gid.x >= pairs_total {
        return;
    }
    let row = gid.x / HALF_ROT;
    let pair = gid.x % HALF_ROT;
    let token = row / N_HEADS;

    let cs = cos_sin[token * HALF_ROT + pair];
    let base = row * HEAD_DIM;
    let partner = HEAD_DIM / 2u;
    let a = f32(xs[base + pair]);
    let b = f32(xs[base + pair + partner]);
    xs[base + pair] = f16(a * cs.x - b * cs.y);
    xs[base + pair + partner] = f16(b * cs.x + a * cs.y);
}
