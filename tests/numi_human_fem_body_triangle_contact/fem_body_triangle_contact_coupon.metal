#include "matter_kernels.metal"
// Appended only to the isolated coupon source, never to the production kernel file.
kernel void nm_numi_human_fem_body_triangle_contact_coupon(
    device const NMNumiHumanFEMBodyContactSampleGPU* samples [[buffer(0)]],
    device const NMFEMNodeStateGPU* nodes [[buffer(1)]],
    device const MRArticulatedBodyPoseGPU* poses [[buffer(2)]],
    device float4* output [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    float3 slavePoint, bodyPoint, normal;
    float closure, area, tangentialSlip;
    float3x3 tangent;
    const bool valid = numi_matter_metal::humanFEMBodyContactState(
        samples[index], nodes[index], poses[index], slavePoint, bodyPoint,
        normal, closure, area, tangent, tangentialSlip);
    if (!valid) {
        output[index*8u+0u]=float4(NAN);
        return;
    }
    const bool triangle =
        samples[index].flags ==
            (NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
             NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT);
    const float stiffness = triangle
        ? samples[index].bodyLocalNormalAndReferenceSeparation.w
        : samples[index].stiffnessAndNormalStrainPerPressure.x;
    const float3 force = normal*(stiffness*closure*area);
    const float3 reaction = -force;
    output[index*8u+0u]=float4(force,closure);
    output[index*8u+1u]=float4(reaction,area);
    output[index*8u+2u]=float4(bodyPoint,tangentialSlip);
    output[index*8u+3u]=float4(tangent[0],0.0f);
    output[index*8u+4u]=float4(tangent[1],0.0f);
    output[index*8u+5u]=float4(tangent[2],0.0f);
    output[index*8u+6u]=float4(cross(bodyPoint-poses[index].position.xyz,
                                    reaction),0.0f);
    output[index*8u+7u]=float4(normal,1.0f);
}

kernel void nm_numi_human_fem_body_contact_neighborhood_coupon(
    constant NMNumiHumanTendonFEMLoadDispatchGPU& dispatch [[buffer(0)]],
    device const NMNumiHumanFEMBodyContactSampleGPU* samples [[buffer(1)]],
    device const NMFEMNodeStateGPU* nodes [[buffer(2)]],
    device const MRArticulatedBodyPoseGPU* poses [[buffer(3)]],
    device float4* output [[buffer(4)]],
    device const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU*
        ranges [[buffer(5)]],
    device const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU*
        triangles [[buffer(6)]],
    uint index [[thread_position_in_grid]]
) {
    const auto sample=samples[index];
    float3 slavePoint, bodyPoint, normal;
    float closure, area, tangentialSlip;
    float3x3 tangent;
    const bool valid =
        numi_matter_metal::humanFEMBodyContactStateForSample(
            index,dispatch,sample,ranges,triangles,nodes[sample.slaveNode],
            poses[sample.bodyIndex],slavePoint,bodyPoint,normal,closure,area,
            tangent,tangentialSlip);
    if (!valid) {
        output[index*2u]=float4(NAN);
        output[index*2u+1u]=float4(NAN);
        return;
    }
    const bool triangle = sample.flags ==
        (NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
         NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT);
    const float stiffness = triangle
        ? sample.bodyLocalNormalAndReferenceSeparation.w
        : sample.stiffnessAndNormalStrainPerPressure.x;
    output[index*2u]=float4(normal*(stiffness*closure*area),closure);
    output[index*2u+1u]=float4(bodyPoint,normal.z);
}
