// Toolchain smoke test for the int8-coopmat naga fork (i8/u8 scalars +
// signed integer MulAdd). NOT a production kernel — proves that
// `coop_mat16x16<i8, …>` parses, validates, and lowers to signed-int8 SPIR-V
// (OpTypeInt 8 1, OpCooperativeMatrixMulAddKHR with the signedness operands
// word, OpCapability Int8). See docs/naga-int8-coopmat-patch.md.
//
// One 16×16×16 tile: C[i32] = A[i8] · B[i8]. coopLoad takes its component
// scalar from the POINTER's base type (the <i8> generic only carries
// rows/cols/role), so A and B must be `array<i8>` — which requires 8-bit
// storage access (VK_KHR_8bit_storage; enabled in GpuContext). Dispatch [1,1,1].

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> a: array<i8>;   // 16×16 i8, row-major
@group(0) @binding(1) var<storage, read> b: array<i8>;   // 16×16 i8, row-major
@group(0) @binding(2) var<storage, read_write> out: array<i32>; // 16×16 i32

@compute @workgroup_size(64)
fn main() {
    let ma = coopLoadT<coop_mat16x16<i8, A>>(&a[0], 16u);
    let mb = coopLoadT<coop_mat16x16<i8, B>>(&b[0], 16u);
    var acc: coop_mat16x16<i32, C>;
    acc = coopMultiplyAdd(ma, mb, acc);
    coopStoreT(acc, &out[0], 16u);
}
