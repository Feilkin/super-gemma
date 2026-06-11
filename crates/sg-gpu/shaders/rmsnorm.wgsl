// RMSNorm over rows of length ROW_LEN (plan 02): hidden-size rows (5376) for
// the four per-layer norms + final norm, head_dim rows (256/512) for QK-norm.
// f16 storage, f32 math, deterministic tree reduction (no subgroup ops — the
// reduction order must be fixed for cache-resume bit-exactness, plan 06).
//
// Variants: ROW_LEN ∈ {5376, 512, 256} × {w, 1+w} (W_PLUS_ONE — the GGUF
// export convention is pinned by M3 parity tests).
//
// Residual-add fusion (plan 02 "fused optional residual-add") is deferred to
// the M3 graph work, when the topology fixes which sums must be materialized.

enable f16;

@group(0) @binding(0) var<storage, read> x: array<f16>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f16>;

const ROW_LEN: u32 = #{ROW_LEN}u;
const EPS: f32 = 1e-6;
const WG: u32 = #{WG_X}u;

var<workgroup> sums: array<f32, #{WG_X}>;

@compute @workgroup_size(#{WG_X})
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let lid = lid_v.x;
    let base = wg_id.x * ROW_LEN;

    var acc = 0.0;
    var i = lid;
    while i < ROW_LEN {
        let v = f32(x[base + i]);
        acc += v * v;
        i += WG;
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
    let inv = inverseSqrt(sums[0] / f32(ROW_LEN) + EPS);

    i = lid;
    while i < ROW_LEN {
#ifdef W_PLUS_ONE
        let weight = 1.0 + w[i];
#else
        let weight = w[i];
#endif
        y[base + i] = f16(f32(x[base + i]) * inv * weight);
        i += WG;
    }
}
