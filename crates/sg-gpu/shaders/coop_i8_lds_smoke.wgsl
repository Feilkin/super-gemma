// Probe: can we coopLoad an int8 cooperative matrix from WORKGROUP (LDS) memory?
// The f16 GEMM stages dequantized weights in LDS and coopLoads f16 from there;
// the Q4_0-reading int8 GEMM wants the same but with i8 (unpack nibbles → i8 LDS
// → coopLoad). 8-bit *shared* memory is a different path from the proven 8-bit
// *storage-buffer* loads, so verify the toolchain accepts it. NOT production.

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> src: array<i8>;        // 16×16 i8
@group(0) @binding(1) var<storage, read_write> out: array<i32>; // 16×16 i32

var<workgroup> lds: array<i8, 256>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid_v: vec3<u32>) {
    let lid = lid_v.x;
    // Construct the i8 LDS value ARITHMETICALLY (i32 expr → i8) — the unpack
    // path needs `i8(nibble - 8)`, which the fork's i8 support hasn't exercised
    // (only reads). Here: lds = i8(src - 1).
    for (var i = lid; i < 256u; i += 64u) {
        lds[i] = i8(i32(src[i]) - 1);
    }
    workgroupBarrier();
    let a = coopLoadT<coop_mat16x16<i8, A>>(&lds[0], 16u); // coopLoad i8 from LDS
    let b = coopLoad<coop_mat16x16<i8, B>>(&lds[0], 16u);
    var acc: coop_mat16x16<i32, C>;
    acc = coopMultiplyAdd(a, b, acc);
    coopStoreT(acc, &out[0], 16u);
}
