#include "matter_kernels.metal"

kernel void nm_numi_human_fem_body_triangle_small_face_coupon(
    device const NMNumiHumanFEMBodyContactSampleGPU* samples [[buffer(0)]],
    device const NMFEMNodeStateGPU* nodes [[buffer(1)]],
    device const MRArticulatedBodyPoseGPU* poses [[buffer(2)]],
    device float4* output [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    float3 slavePoint, bodyPoint, normal;
    float closure, area, tangentialSlip;
    float3x3 tangent;
    const bool valid = numi_matter_metal::humanFEMBodyTriangleContactState(
        samples[index],nodes[index],poses[index],slavePoint,bodyPoint,normal,
        closure,area,tangent,tangentialSlip);
    if (!valid) {
        output[index*8u]=float4(NAN);
        return;
    }
    const float stiffness =
        samples[index].bodyLocalNormalAndReferenceSeparation.w;
    const float3 force=normal*(stiffness*closure*area);
    const float3 reaction=-force;
    const numi_matter_metal::HumanFEMBodyTriangleClosest debugClosest =
        numi_matter_metal::humanClosestFeatureOnTriangle(
            nodes[index].positionAndMass.xyz,
            samples[index].bodyLocalPointAndArea.xyz,
            samples[index].bodyLocalNormalAndReferenceSeparation.xyz,
            samples[index].stiffnessAndNormalStrainPerPressure.xyz);
    output[index*8u+0u]=float4(force,closure);
    output[index*8u+1u]=float4(reaction,
        debugClosest.barycentric.x);
    output[index*8u+2u]=float4(bodyPoint,float(debugClosest.feature));
    output[index*8u+3u]=float4(tangent[0],debugClosest.barycentric.y);
    output[index*8u+4u]=float4(tangent[1],debugClosest.barycentric.z);
    output[index*8u+5u]=float4(tangent[2],0.0f);
    output[index*8u+6u]=float4(cross(bodyPoint-poses[index].position.xyz,
                                    reaction),0.0f);
    output[index*8u+7u]=float4(normal,1.0f);
}
