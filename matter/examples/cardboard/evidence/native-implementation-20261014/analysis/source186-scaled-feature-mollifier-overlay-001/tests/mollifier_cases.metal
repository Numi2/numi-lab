#include "matter_kernels.metal"

namespace numi_matter_metal {
kernel void nm_ipc_mollifier_scale_cases(
    device const float4* vectors [[buffer(0)]],
    device float4* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    const float3 first = vectors[2u * tid].xyz;
    const float3 second = vectors[2u * tid + 1u].xyz;
    float3 firstGradient, secondGradient;
    const float value = ipcFeatureMollifierDifferential(
        first, second, firstGradient, secondGradient);
    output[3u * tid] = float4(value, 0.0f, 0.0f, 0.0f);
    output[3u * tid + 1u] = float4(firstGradient, 0.0f);
    output[3u * tid + 2u] = float4(secondGradient, 0.0f);
}
} // namespace numi_matter_metal
