// Fused GeGLU activation (plan 02 `mlp_geglu`): y = gelu_tanh(gate) * up,
// elementwise over the 21504-wide MLP intermediate. f16 storage, f32 math.
// The surrounding projections run as gemv/gemm.

enable f16;

@group(0) @binding(0) var<storage, read> gate: array<f16>;
@group(0) @binding(1) var<storage, read> up: array<f16>;
@group(0) @binding(2) var<storage, read_write> y: array<f16>;

// gelu_pytorch_tanh: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
const SQRT_2_OVER_PI: f32 = 0.7978845608028654;

@compute @workgroup_size(#{WG_X})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= arrayLength(&y) {
        return;
    }
    let g = f32(gate[gid.x]);
    let u = f32(up[gid.x]);
    let inner = SQRT_2_OVER_PI * (g + 0.044715 * g * g * g);
    let gelu = 0.5 * g * (1.0 + tanh(inner));
    y[gid.x] = f16(gelu * u);
}
