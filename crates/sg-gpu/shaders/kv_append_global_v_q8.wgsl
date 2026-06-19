// Quantize-and-append for the int8 V of the GLOBAL KV cache (Piece B). Unlike
// the K append (kv_append_global_q8), which quantizes each row in 32-element
// HEAD-DIM blocks, V is quantized in 32-KEY blocks along the key axis — the PV
// contraction — so the per-block scale factors out of the int8 PV dot (see the
// q8_quant_v reference / docs/q8-kv-flash-impl.md §2). Prefill chunks start at a
// 32-aligned position and each 32-key block lies wholly inside one chunk, so no
// cross-chunk requant is needed here; the final partial block (last chunk) is
// quantized over its live keys.
//
// v_quants i8 [L × N_KV_HEADS × HEAD_DIM] (cols packed 4/u32, same layout as the
// f16 V it parallels); v_scales f16 [ceil(L/32) × N_KV_HEADS × HEAD_DIM], one
// per (32-key block, head, head-dim col). Q8_0 math identical to the K append:
// d = amax·(1/127), 1/d reciprocal multiply, f16() RTNE scale, round-half-even.
//
// One thread per (32-key block, head, 4-col group). Dispatch x covers
// ceil(n_real/32) × N_KV_HEADS × (HEAD_DIM/4) threads.

enable f16;

@group(0) @binding(0) var<storage, read> src: array<f16>;          // [n_real × N_KV_HEADS × HEAD_DIM]
@group(0) @binding(1) var<storage, read_write> scales: array<f16>; // v_scales
@group(0) @binding(2) var<storage, read_write> quants: array<u32>; // v_quants (i8 packed 4/u32)
// Per-step dynamic state (sg_gpu::StepState): step[0] = pos, the 32-aligned
// absolute position of the chunk's first key.
@group(0) @binding(3) var<storage, read> step: array<u32>;

const N_KV_HEADS: u32 = #{N_KV_HEADS}u;
const HEAD_DIM: u32 = #{HEAD_DIM}u;
const KV_DIM: u32 = N_KV_HEADS * HEAD_DIM; // row stride between keys
const CG: u32 = HEAD_DIM / 4u;             // 4-col groups per head

@compute @workgroup_size(#{WG_X})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_real = arrayLength(&src) / KV_DIM;
    let n_blocks = (n_real + 31u) / 32u;
    if gid.x >= n_blocks * N_KV_HEADS * CG {
        return;
    }
    let cg = gid.x % CG;
    let h = (gid.x / CG) % N_KV_HEADS;
    let tb = gid.x / (CG * N_KV_HEADS); // 32-key block within the chunk

    let c0 = cg * 4u;
    let k0 = tb * 32u;               // first chunk-local key of the block
    let k1 = min(k0 + 32u, n_real);  // one past the last live key
    let pos = step[0];
    let kb = pos / 32u + tb;          // global key-block index

    // One scale per col: amax over the block's live keys at that col.
    var id: array<f32, 4>;
    for (var c = 0u; c < 4u; c += 1u) {
        var amax = 0.0;
        for (var k = k0; k < k1; k += 1u) {
            amax = max(amax, abs(f32(src[k * KV_DIM + h * HEAD_DIM + c0 + c])));
        }
        let d = amax * (1.0 / 127.0);
        var idc = 0.0;
        if d > 0.0 {
            idc = 1.0 / d;
        }
        id[c] = idc;
        scales[(kb * N_KV_HEADS + h) * HEAD_DIM + c0 + c] = f16(d);
    }

    // Pack the 4 cols' i8 quants per live key into one u32 (matches q8_quant_v).
    for (var k = k0; k < k1; k += 1u) {
        let gkey = pos + k;
        var packed = 0u;
        for (var c = 0u; c < 4u; c += 1u) {
            let q = i32(round(f32(src[k * KV_DIM + h * HEAD_DIM + c0 + c]) * id[c]));
            packed |= (u32(q) & 0xFFu) << (8u * c);
        }
        quants[((gkey * N_KV_HEADS + h) * HEAD_DIM + c0) / 4u] = packed;
    }
}
