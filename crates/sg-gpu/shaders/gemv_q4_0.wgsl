// Fused-dequant Q4_0 GEMV (plan 02): y[N] = W[N×K]·x[K] for decode (M=1).
// THE bandwidth-critical kernel — weights stream once per token.
//
// Weight layout is the GGUF tensor verbatim: N rows, each K/32 Q4_0 blocks
// of 18 bytes (f16 scale + 16 nibble bytes). Every shape in this model has
// an even block count per row, so threads walk aligned 36-byte block PAIRS
// (9 u32 words) — no unaligned access, fully coalesced 2304-byte wavefront
// reads. One wave-sized workgroup per output row; deterministic tree
// reduction (plan 06 bit-exactness).
//
// Variants: one per matmul site with K baked (full unroll of the inner
// nibble loops), plus a GENERIC_K push-constant variant kept as the A/B
// baseline (plan 02).

enable f16;

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f16>;
@group(0) @binding(2) var<storage, read_write> y: array<f16>;

#ifdef GENERIC_K
struct Push {
    k: u32,
}
var<immediate> push: Push;
#else
const K: u32 = #{K_DIM}u;
#endif

const WG: u32 = #{WG_X}u;

var<workgroup> sums: array<f32, #{WG_X}>;

// One Q4_0 block's dot product with x[x_base..x_base+32]: 16 bytes of
// nibbles; low nibbles are weights 0..16, high nibbles 16..32, w = d·(q−8).
fn block_dot(d: f32, q0: u32, q1: u32, q2: u32, q3: u32, x_base: u32) -> f32 {
    var sum = 0.0;
    var qs = array<u32, 4>(q0, q1, q2, q3);
    for (var w = 0u; w < 4u; w += 1u) {
        let word = qs[w];
        for (var b = 0u; b < 4u; b += 1u) {
            let byte = (word >> (8u * b)) & 0xFFu;
            let lo = f32(byte & 0xFu) - 8.0;
            let hi = f32(byte >> 4u) - 8.0;
            let j = x_base + 4u * w + b;
            sum += lo * f32(x[j]) + hi * f32(x[j + 16u]);
        }
    }
    return d * sum;
}

@compute @workgroup_size(#{WG_X})
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
#ifdef GENERIC_K
    let k_dim = push.k;
#else
    let k_dim = K;
#endif
    let lid = lid_v.x;
    let row = wg_id.x;
    let pairs = k_dim / 64u; // block pairs per row
    let row_words = pairs * 9u; // 36 bytes per pair
    let base = row * row_words;

    var acc = 0.0;
    var p = lid;
    while p < pairs {
        let w0 = weights[base + p * 9u];
        let w1 = weights[base + p * 9u + 1u];
        let w2 = weights[base + p * 9u + 2u];
        let w3 = weights[base + p * 9u + 3u];
        let w4 = weights[base + p * 9u + 4u];
        let w5 = weights[base + p * 9u + 5u];
        let w6 = weights[base + p * 9u + 6u];
        let w7 = weights[base + p * 9u + 7u];
        let w8 = weights[base + p * 9u + 8u];

        // Block A: d in w0 low half, 16 qs bytes in w0.hi .. w4.lo.
        let d_a = unpack2x16float(w0).x;
        let xa = p * 64u;
        acc += block_dot(
            d_a,
            (w0 >> 16u) | (w1 << 16u),
            (w1 >> 16u) | (w2 << 16u),
            (w2 >> 16u) | (w3 << 16u),
            (w3 >> 16u) | (w4 << 16u),
            xa,
        );
        // Block B: d in w4 high half, qs bytes in w5..w8.
        let d_b = unpack2x16float(w4).y;
        acc += block_dot(d_b, w5, w6, w7, w8, xa + 32u);

        p += WG;
    }

    sums[lid] = acc;
    workgroupBarrier();
    var stride = WG / 2u;
    while stride > 0u {
        if lid < stride {
            sums[lid] += sums[lid + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }
    if lid == 0u {
        y[row] = f16(sums[0]);
    }
}
