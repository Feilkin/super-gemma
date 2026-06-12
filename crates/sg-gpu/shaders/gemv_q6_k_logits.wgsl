// Q6_K LM-head GEMV with fused logit softcap (plan 02):
// logits[N] = 30·tanh((E[N×K]·x[K]) / 30) for decode. E is the tied Q6_K
// token-embedding matrix (262144 × 5376) — ~1.16 GB streamed per token;
// bandwidth-critical like gemv_q4_0.
//
// Weight layout: GGUF Q6_K rows (21 × 210-byte super-blocks) padded to a
// ROW_WORDS·4-byte stride so every row starts word-aligned (the engine
// uploads the embedding tensor with this stride; 210·21 = 4410 → 4416).
// Per block, lane = half·32 + l computes the reference's four weights
// {half·128 + l, +32, +64, +96} — a 64-lane wave covers the 256-weight
// super-block exactly (see sg_gguf::q6_k::BlockQ6K::dequantize).
//
// One wave-sized workgroup per output row; deterministic LDS tree
// reduction; f32 logits out (the CPU sampler consumes them directly).

enable f16;

@group(0) @binding(0) var<storage, read> weights: array<u32>; // padded Q6_K rows
@group(0) @binding(1) var<storage, read> x: array<f16>; // [K]
@group(0) @binding(2) var<storage, read_write> logits: array<f32>; // [N]

const BLOCKS_PER_ROW: u32 = #{BLOCKS_PER_ROW}u;
const ROW_WORDS: u32 = #{ROW_WORDS}u;
const SOFTCAP: f32 = 30.0;
const WG: u32 = #{WG_X}u;

var<workgroup> sums: array<f32, #{WG_X}>;

// One byte of the current row (byte offset within the row).
fn row_byte(row_word_base: u32, off: u32) -> f32 {
    return f32((weights[row_word_base + off / 4u] >> (8u * (off % 4u))) & 0xFFu);
}

// One SIGNED byte of the current row.
fn row_i8(row_word_base: u32, off: u32) -> f32 {
    let b = (weights[row_word_base + off / 4u] >> (8u * (off % 4u))) & 0xFFu;
    return f32((i32(b) << 24u) >> 24u);
}

@compute @workgroup_size(#{WG_X})
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let lid = lid_v.x;
    let row = wg_id.x;
    let base = row * ROW_WORDS;
    let half = lid / 32u;
    let l = lid % 32u;
    let is = l / 16u;

    var acc = 0.0;
    for (var b = 0u; b < BLOCKS_PER_ROW; b += 1u) {
        let bb = b * 210u; // block byte offset within the row
        // d at byte 208 (2-byte aligned in every block: 210·b + 208 is even).
        let d_off = bb + 208u;
        let d = unpack2x16float((weights[base + d_off / 4u] >> (8u * (d_off % 4u))) & 0xFFFFu).x;

        let ql_a = u32(row_byte(base, bb + half * 64u + l));
        let ql_b = u32(row_byte(base, bb + half * 64u + l + 32u));
        let qh = u32(row_byte(base, bb + 128u + half * 32u + l));
        let sc_base = bb + 192u + half * 8u + is;
        let sc1 = row_i8(base, sc_base);
        let sc2 = row_i8(base, sc_base + 2u);
        let sc3 = row_i8(base, sc_base + 4u);
        let sc4 = row_i8(base, sc_base + 6u);

        let q1 = f32(i32((ql_a & 0xFu) | ((qh & 3u) << 4u)) - 32);
        let q2 = f32(i32((ql_b & 0xFu) | (((qh >> 2u) & 3u) << 4u)) - 32);
        let q3 = f32(i32((ql_a >> 4u) | (((qh >> 4u) & 3u) << 4u)) - 32);
        let q4 = f32(i32((ql_b >> 4u) | (((qh >> 6u) & 3u) << 4u)) - 32);

        let xb = b * 256u + half * 128u + l;
        acc += d
            * (sc1 * q1 * f32(x[xb]) + sc2 * q2 * f32(x[xb + 32u])
                + sc3 * q3 * f32(x[xb + 64u])
                + sc4 * q4 * f32(x[xb + 96u]));
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
        logits[row] = SOFTCAP * tanh(sums[0] / SOFTCAP);
    }
}
