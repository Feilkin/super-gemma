// Smoke test: does 0-stride coopLoad broadcast the same way from WORKGROUP (LDS)
// memory as it does from storage? (coop_bcast_smoke proved storage; the
// attention rescale needs LDS — the scale vectors are workgroup-local.) Stage
// the two vectors into LDS, then build the outer product with 0-stride loads.

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> pvec: array<f32>;      // 16
@group(0) @binding(1) var<storage, read> vvec: array<f32>;      // 16
@group(0) @binding(2) var<storage, read_write> out: array<f32>; // 16×16

var<workgroup> p_lds: array<f32, 16>;
var<workgroup> v_lds: array<f32, 16>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid_v: vec3<u32>) {
    let lid = lid_v.x;
    if (lid < 16u) {
        p_lds[lid] = pvec[lid];
        v_lds[lid] = vvec[lid];
    }
    workgroupBarrier();
    let pmat = coopLoadT<coop_mat16x16<f32, C>>(&p_lds[0], 0u);
    let vmat = coopLoad<coop_mat16x16<f32, C>>(&v_lds[0], 0u);
    coopStoreT(pmat * vmat, &out[0], 16u);
}
