#include "matter_kernels.metal"

using namespace numi_matter_metal;

kernel void nm_numi_human_fem_all_body_moving_capacity_coupon(
    constant NMNumiHumanTendonFEMLoadDispatchGPU& dispatch [[buffer(0)]],
    device const NMNumiHumanFEMBodyContactSampleGPU* samples [[buffer(1)]],
    device const NMFEMNodeStateGPU* nodes [[buffer(2)]],
    device const MRArticulatedBodyPoseGPU* poses [[buffer(3)]],
    device const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU* ranges [[buffer(4)]],
    device const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU* triangles [[buffer(5)]],
    device const NMNumiHumanFEMBodyContactBVHNodeGPU* bvhNodes [[buffer(6)]],
    device float4* output [[buffer(7)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= dispatch.femBodyContactSampleCount) return;
    const NMNumiHumanFEMBodyContactSampleGPU sample=samples[index];
    float3 slavePoint,bodyPoint,normal;
    float closure,area,tangentialSlip;
    float3x3 tangent;
    const bool valid=humanFEMBodyContactStateForSample(
        index,dispatch,sample,ranges,triangles,bvhNodes,
        nodes[sample.slaveNode],poses[sample.bodyIndex],
        slavePoint,bodyPoint,normal,closure,area,tangent,tangentialSlip);
    const uint base=index*8u;
    if (!valid) {
        for (uint i=0u;i<8u;++i) output[base+i]=float4(NAN);
        return;
    }
    const float stiffness=sample.bodyLocalNormalAndReferenceSeparation.w;
    const float3 force=normal*(stiffness*closure*area);
    const float3 reaction=-force;
    output[base+0u]=float4(force,closure);
    output[base+1u]=float4(reaction,area);
    output[base+2u]=float4(bodyPoint,tangentialSlip);
    output[base+3u]=float4(tangent[0],0.0f);
    output[base+4u]=float4(tangent[1],0.0f);
    output[base+5u]=float4(tangent[2],0.0f);
    output[base+6u]=float4(cross(bodyPoint-poses[sample.bodyIndex].position.xyz,
                                  reaction),0.0f);
    output[base+7u]=float4(normal,1.0f);
}
