// Elementwise residual add with scalar scale (plan 02's deferred "fused
// residual-add", landed for the M3 graph): y = (a + b) * s.
//
// Covers both per-layer residual joins of the Gemma 4 graph; the second one
// fuses the layer_output_scale scalar (s = 1.0 for the attention join).
// f16 storage, f32 math; one thread per element.
//
// Dispatch: x covers arrayLength(&y) elements.

enable f16;

@group(0) @binding(0) var<storage, read> a: array<f16>;
@group(0) @binding(1) var<storage, read> b: array<f16>;
@group(0) @binding(2) var<storage, read_write> y: array<f16>;

struct Push {
    scale: f32,
}
var<immediate> push: Push;

@compute @workgroup_size(#{WG_X})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= arrayLength(&y) {
        return;
    }
    y[gid.x] = f16((f32(a[gid.x]) + f32(b[gid.x])) * push.scale);
}
