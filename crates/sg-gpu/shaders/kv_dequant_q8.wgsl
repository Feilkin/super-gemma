// Q8_0 → f16 dequantization for cache2 page loads (plan 02/04), the
// inverse of kv_quant_q8's structure-of-arrays format: x = f16(d · q).
// One thread per 32-weight block.
//
// Dispatch: x covers arrayLength(dst)/32 blocks.

enable f16;

@group(0) @binding(0) var<storage, read> scales: array<f16>; // one per block
@group(0) @binding(1) var<storage, read> quants: array<u32>; // 8 words per block
@group(0) @binding(2) var<storage, read_write> dst: array<f16>;

const WG: u32 = #{WG_X}u;

@compute @workgroup_size(#{WG_X})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let blk = gid.x;
    if blk >= arrayLength(&dst) / 32u {
        return;
    }
    let d = f32(scales[blk]);
    for (var w = 0u; w < 8u; w += 1u) {
        let word = quants[blk * 8u + w];
        for (var b = 0u; b < 4u; b += 1u) {
            let q = (i32((word >> (8u * b)) & 0xFFu) << 24u) >> 24u; // sign-extend i8
            dst[blk * 32u + w * 4u + b] = f16(d * f32(q));
        }
    }
}
