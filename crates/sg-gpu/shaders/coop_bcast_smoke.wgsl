// Smoke test: build a 16×16 outer product p[q]·v[c] from two 16-element vectors
// using 0-STRIDE coopLoads — no LDS materialization of the matrix. Hypothesis:
// a coopLoad with row-stride 0 broadcasts the 16-element vector across the
// fragment; coopLoad vs coopLoadT broadcast on the two different axes, so the
// element-wise product of the two is the outer product. If this works, the
// attention/GEMM per-block rescale can drop its 256-element LDS scale build.
// NOT production — proves the toolchain accepts stride-0 loads and what layout
// they yield.

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> pvec: array<f32>;      // 16 (row scale)
@group(0) @binding(1) var<storage, read> vvec: array<f32>;      // 16 (col scale)
@group(0) @binding(2) var<storage, read_write> out: array<f32>; // 16×16

@compute @workgroup_size(64)
fn main() {
    // coopLoadT(.,0): hypothesis [q][c] = pvec[q] (broadcast across columns).
    let pmat = coopLoadT<coop_mat16x16<f32, C>>(&pvec[0], 0u);
    // coopLoad(.,0):  hypothesis [q][c] = vvec[c] (broadcast down rows).
    let vmat = coopLoad<coop_mat16x16<f32, C>>(&vvec[0], 0u);
    coopStoreT(pmat * vmat, &out[0], 16u);
}
