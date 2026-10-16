#include "metal/common.metalinc"
#include "metal/fem.metalinc"
#include "base_cgls.metalinc"
#include "production_layout_extract.metalinc"

using namespace metal;

// Test-only block diagonal A/A^T action. The production layout/state and CGLS
// kernels are included unchanged from the source-bound production snapshot.
kernel void reinit_fixture_apply(
    device const float* matrices [[buffer(0)]],
    device const float4* input [[buffer(1)]],
    device float4* output [[buffer(2)]],
    constant uint& unknownCount [[buffer(3)]],
    constant uint& nodeCount [[buffer(4)]],
    constant uint& transpose [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= unknownCount) return;
    const uint mechanicalRows = unknownCount / 2u;
    if (index >= mechanicalRows) {
        output[index] = float4(0.0f);
        return;
    }
    if (nodeCount == 0u) {
        output[index] = float4(0.0f);
        return;
    }
    const uint environment = index / nodeCount;
    const uint matrixBase = (environment * nodeCount + (index % nodeCount)) * 9u;
    const float3 x = input[index].xyz;
    float3 y = float3(0.0f);
    for (uint row = 0u; row < 3u; ++row) {
        float value = 0.0f;
        for (uint col = 0u; col < 3u; ++col) {
            const uint matrixIndex = transpose != 0u
                ? matrixBase + col * 3u + row
                : matrixBase + row * 3u + col;
            value = fma(matrices[matrixIndex], x[col], value);
        }
        y[row] = value;
    }
    output[index] = float4(y, 0.0f);
}
