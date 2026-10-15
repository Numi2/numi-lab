#include <metal_stdlib>
#include "base_cgls.metalinc"

using namespace metal;

namespace numi_matter_metal {

kernel void nm_base_cgls_check_clear(
    constant uint& unknownCount [[buffer(0)]],
    device float4* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index < unknownCount) output[index] = float4(0.0f);
}

kernel void nm_base_cgls_check_apply_A(
    constant uint& nodeCount [[buffer(0)]],
    constant uint& freeNodeCount [[buffer(1)]],
    constant uint& dofCount [[buffer(2)]],
    device const uint* rowOffsets [[buffer(3)]],
    device const uint* columns [[buffer(4)]],
    device const float* values [[buffer(5)]],
    device const uint* freeNodes [[buffer(6)]],
    device const float4* input [[buffer(7)]],
    device float4* output [[buffer(8)]],
    device const uint4* control [[buffer(9)]],
    uint freeNodeIndex [[thread_position_in_grid]]) {
    if (freeNodeIndex >= freeNodeCount || nodeCount == 0u || dofCount == 0u ||
        control[0].x != kBaseCGLSRunning) return;
    const uint node = freeNodes[freeNodeIndex];
    if (node >= nodeCount) return;
    float3 result = float3(0.0f);
    for (uint axis = 0u; axis < 3u; ++axis) {
        const uint row = 3u * freeNodeIndex + axis;
        if (row >= dofCount) return;
        float sum = 0.0f;
        for (uint entry = rowOffsets[row]; entry < rowOffsets[row + 1u]; ++entry) {
            const uint column = columns[entry];
            if (column >= dofCount) return;
            const uint columnNode = freeNodes[column / 3u];
            if (columnNode >= nodeCount) return;
            const float component = input[columnNode][column % 3u];
            sum = fma(values[entry], component, sum);
        }
        result[axis] = sum;
    }
    output[node] = float4(result, 0.0f);
}

kernel void nm_base_cgls_check_apply_AT(
    constant uint& nodeCount [[buffer(0)]],
    constant uint& freeNodeCount [[buffer(1)]],
    constant uint& dofCount [[buffer(2)]],
    device const uint* rowOffsets [[buffer(3)]],
    device const uint* columns [[buffer(4)]],
    device const float* values [[buffer(5)]],
    device const uint* freeNodes [[buffer(6)]],
    device const float4* input [[buffer(7)]],
    device float4* output [[buffer(8)]],
    device const uint4* control [[buffer(9)]],
    uint freeNodeIndex [[thread_position_in_grid]]) {
    if (freeNodeIndex >= freeNodeCount || nodeCount == 0u || dofCount == 0u ||
        control[0].x != kBaseCGLSRunning) return;
    const uint node = freeNodes[freeNodeIndex];
    if (node >= nodeCount) return;
    float3 result = float3(0.0f);
    for (uint axis = 0u; axis < 3u; ++axis) {
        const uint row = 3u * freeNodeIndex + axis;
        if (row >= dofCount) return;
        float sum = 0.0f;
        for (uint entry = rowOffsets[row]; entry < rowOffsets[row + 1u]; ++entry) {
            const uint column = columns[entry];
            if (column >= dofCount) return;
            const uint columnNode = freeNodes[column / 3u];
            if (columnNode >= nodeCount) return;
            const float component = input[columnNode][column % 3u];
            sum = fma(values[entry], component, sum);
        }
        result[axis] = sum;
    }
    output[node] = float4(result, 0.0f);
}

} // namespace numi_matter_metal
