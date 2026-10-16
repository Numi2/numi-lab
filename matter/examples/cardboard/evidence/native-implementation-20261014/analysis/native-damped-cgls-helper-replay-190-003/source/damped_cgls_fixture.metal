#include "metal/common.metalinc"
#include "metal/fem.metalinc"
#include "base_cgls.metalinc"
using namespace metal;

// Tiny nonsymmetric operator fixture. The production CGLS recurrence kernels
// above are used unchanged; this kernel supplies only test A and A^T actions.
kernel void damped_fixture_apply(
    device const float* matrix [[buffer(0)]],
    device const float4* input [[buffer(1)]],
    device float4* output [[buffer(2)]],
    constant uint& unknownCount [[buffer(3)]],
    constant uint& transpose [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= unknownCount) return;
    if (index != 0u) {
        output[index] = float4(0.0f);
        return;
    }
    const float3 x = input[0].xyz;
    float3 y = float3(0.0f);
    for (uint row = 0u; row < 3u; ++row) {
        float value = 0.0f;
        for (uint col = 0u; col < 3u; ++col) {
            const uint matrixIndex = transpose != 0u
                ? col * 3u + row : row * 3u + col;
            value = fma(matrix[matrixIndex], x[col], value);
        }
        y[row] = value;
    }
    output[0] = float4(y, 0.0f);
}
