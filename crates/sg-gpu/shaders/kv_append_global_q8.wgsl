// Quantize-and-append for the Q8 GLOBAL KV cache (plan 02; Piece B — halve the
// long-context K/V streaming traffic the flash A/B identified as the global
// prefill bottleneck). Fuses kv_quant_q8's f16→Q8_0 with kv_append_global's
// linear slot placement: each freshly projected K (or V) row is quantized in
// 32-element blocks and stored at slot = pos + token.
//
// Q8_0 SoA format (matches kv_quant_q8 / kv_dequant_q8, plan 04 pages): a f16
// scales array (one per 32-block) and an i8 quants array (word-aligned, 8 u32
// per block). d = amax·(1/127) via a reciprocal multiply (GPU FDiv is 2.5 ULP;
// the scale goes through f16() for RTNE — pack2x16float truncates on RADV),
// round-half-even quants — bit-reproducible, the GPU is the authoritative
// producer. Global only: K=V≠ here, separate K/V stores (M3 amendment), linear
// append (no ring).
//
// One thread per 32-block. Dispatch: x covers n_tokens × (ROW_LEN/32) blocks.

enable f16;

@group(0) @binding(0) var<storage, read> src: array<f16>;          // [n_tokens × ROW_LEN]
@group(0) @binding(1) var<storage, read_write> scales: array<f16>; // [slots × ROW_LEN/32]
@group(0) @binding(2) var<storage, read_write> quants: array<u32>; // [slots × ROW_LEN/32 × 8]
// Per-step dynamic state (sg_gpu::StepState): [pos, …]. pos = absolute
// position of the first appended token (linear global slot base).
@group(0) @binding(3) var<storage, read> step: array<u32>;

const ROW_LEN: u32 = #{ROW_LEN}u;
const BLOCKS_PER_ROW: u32 = ROW_LEN / 32u;

@compute @workgroup_size(#{WG_X})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let blk = gid.x;
    if blk >= arrayLength(&src) / 32u {
        return;
    }
    let token = blk / BLOCKS_PER_ROW;
    let local = blk % BLOCKS_PER_ROW; // block within the row
    let slot = step[0] + token;       // linear global placement
    let s = blk * 32u;                // src element base (contiguous rows)
    let dblk = slot * BLOCKS_PER_ROW + local; // destination block index

    var amax = 0.0;
    for (var i = 0u; i < 32u; i += 1u) {
        amax = max(amax, abs(f32(src[s + i])));
    }
    let d = amax * (1.0 / 127.0);
    var id = 0.0;
    if d > 0.0 {
        id = 1.0 / d;
    }

    scales[dblk] = f16(d);
    for (var w = 0u; w < 8u; w += 1u) {
        var packed = 0u;
        for (var b = 0u; b < 4u; b += 1u) {
            let q = i32(round(f32(src[s + w * 4u + b]) * id));
            packed |= (u32(q) & 0xFFu) << (8u * b);
        }
        quants[dblk * 8u + w] = packed;
    }
}
