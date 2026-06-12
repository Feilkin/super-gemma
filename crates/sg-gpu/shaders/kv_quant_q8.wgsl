// f16 → Q8_0 quantization for cache2 page flushes (plan 02/04). Q8_0
// semantics per ggml — blocks of 32 weights, f16 scale d = amax·(1/127),
// quants q = round(x/d) in i8 — but laid out structure-of-arrays (OUR page
// format, plan 04): a scales array (f16, one per block) and a quants array
// (i8, word-aligned 8 u32 per block). SoA keeps every store aligned and
// lets the f32→f16 scale conversion go through a plain f16() store —
// round-to-nearest-even, matching the CPU reference bit-for-bit
// (pack2x16float does NOT: RADV truncates toward zero).
//
// d uses a reciprocal multiply, not /127: GPU FDiv is only 2.5-ULP
// accurate and the scale must be bit-exact vs the CPU. The remaining 1/d
// reciprocal CAN differ from the CPU by an ulp, flipping a quant by ±1 at
// an exact rounding boundary — harmless (½-step quantization error) and
// bit-deterministic on the GPU, which is the authoritative producer of
// cache2 pages.
//
// round() is round-half-even — bit-reproducible and mirrored by the CPU
// reference with `round_ties_even`. One thread per 32-weight block.
//
// Dispatch: x covers arrayLength(src)/32 blocks.

enable f16;

@group(0) @binding(0) var<storage, read> src: array<f16>;
@group(0) @binding(1) var<storage, read_write> scales: array<f16>; // one per block
@group(0) @binding(2) var<storage, read_write> quants: array<u32>; // 8 words per block

const WG: u32 = #{WG_X}u;

@compute @workgroup_size(#{WG_X})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let blk = gid.x;
    if blk >= arrayLength(&src) / 32u {
        return;
    }
    let s = blk * 32u;

    var amax = 0.0;
    for (var i = 0u; i < 32u; i += 1u) {
        amax = max(amax, abs(f32(src[s + i])));
    }
    let d = amax * (1.0 / 127.0);
    var id = 0.0;
    if d > 0.0 {
        id = 1.0 / d;
    }

    scales[blk] = f16(d);
    for (var w = 0u; w < 8u; w += 1u) {
        var packed = 0u;
        for (var b = 0u; b < 4u; b += 1u) {
            let q = i32(round(f32(src[s + w * 4u + b]) * id));
            packed |= (u32(q) & 0xFFu) << (8u * b);
        }
        quants[blk * 8u + w] = packed;
    }
}
