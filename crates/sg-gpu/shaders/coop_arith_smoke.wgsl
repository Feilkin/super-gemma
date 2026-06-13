// Toolchain smoke test for the coopmat-arith naga ops (fork rev 8366d92e):
// component-wise `OpFMul` (coop × coop) and `f32(coop<i32>)` conversion
// (`OpConvertSToF`). These unblock the int8-MMQ in-register rescale
// (docs/naga-coopmat-arith-patch.md). NOT a production kernel.
//
// out[m][n] = sc[m][n] * f32(di[m][n]). Exercises: coopLoad into a C-use
// matrix, i32→f32 conversion on a coopmat, and component-wise multiply.

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> di: array<i32>;        // 16×16 i32
@group(0) @binding(1) var<storage, read> sc: array<f32>;        // 16×16 f32
@group(0) @binding(2) var<storage, read_write> out: array<f32>; // 16×16 f32

@compute @workgroup_size(64)
fn main() {
    let d_i = coopLoadT<coop_mat16x16<i32, C>>(&di[0], 16u);
    let s = coopLoadT<coop_mat16x16<f32, C>>(&sc[0], 16u);
    let d_f = f32(d_i);               // OpConvertSToF on a coopmat   [NEW]
    coopStoreT(s * d_f, &out[0], 16u); // component-wise OpFMul        [NEW]
}
