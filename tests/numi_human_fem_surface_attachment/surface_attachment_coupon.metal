#include "matter_kernels.metal"

kernel void nm_numi_human_surface_attachment_state_coupon(
    constant NMNumiHumanTendonFEMLoadDispatchGPU& dispatch [[buffer(0)]],
    device const NMNumiHumanFEMSurfaceAttachmentGPU* samples [[buffer(1)]],
    device const NMFEMNodeStateGPU* nodes [[buffer(2)]],
    device const MRArticulatedBodyPoseGPU* poses [[buffer(3)]],
    device float4* output [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    const NMNumiHumanFEMSurfaceAttachmentGPU sample = samples[index];
    float3 slavePoint, targetPoint, gap, force;
    float pressure, strain, energy;
    const bool valid =
        sample.bodyIndex >= dispatch.articulationFirstBody &&
        sample.bodyIndex - dispatch.articulationFirstBody <
            dispatch.bodyPoseStride &&
        sample.node[0] < dispatch.femNodeCount &&
        sample.node[1] < dispatch.femNodeCount &&
        sample.node[2] < dispatch.femNodeCount &&
        numi_matter_metal::humanFEMSurfaceAttachmentState(
            dispatch, sample,
            nodes[sample.node[0]].positionAndMass.xyz,
            nodes[sample.node[1]].positionAndMass.xyz,
            nodes[sample.node[2]].positionAndMass.xyz,
            poses[sample.bodyIndex],
            slavePoint, targetPoint, gap, force,
            pressure, strain, energy);
    output[index] = valid
        ? float4(1.0f, energy, length(gap), pressure)
        : float4(0.0f, NAN, NAN, NAN);
}
