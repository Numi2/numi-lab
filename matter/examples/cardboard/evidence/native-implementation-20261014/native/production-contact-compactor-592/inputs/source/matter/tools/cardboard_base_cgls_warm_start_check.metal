#include "metal/common.metalinc"
#include "metal/fem.metalinc"
#include "base_cgls.metalinc"

// Keep this checker translation unit's helper byte-for-byte aligned with the
// production helper in fgmres.metalinc. The production installer continues
// to call that original helper in both default and warm modes.
inline void fgmresCompensatedAdd(
    thread float& sum,
    thread float& correction,
    const float value
) {
    const float adjusted = value - correction;
    const float next = sum + adjusted;
    correction = (next - sum) - adjusted;
    sum = next;
}

#include "base_cgls_runtime.metalinc"

using namespace metal;
namespace numi_matter_metal {

kernel void nm_warm_check_clear(
    constant uint& count [[buffer(0)]],
    device float4* values [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i < count) values[i] = float4(0.0f);
}

kernel void nm_warm_check_apply_A(
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
    if (freeNodeIndex >= freeNodeCount || nodeCount == 0u ||
        dofCount == 0u || control[0].x != kBaseCGLSRunning) return;
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
            sum = fma(values[entry], input[columnNode][column % 3u], sum);
        }
        result[axis] = sum;
    }
    output[node] = float4(result, 0.0f);
}

} // namespace numi_matter_metal
