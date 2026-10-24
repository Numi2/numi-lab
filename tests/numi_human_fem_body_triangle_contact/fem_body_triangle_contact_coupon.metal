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
    device const NMNumiHumanFEMBodyContactBVHNodeGPU*
        bvhNodes [[buffer(7)]],
    uint index [[thread_position_in_grid]]
) {
    const auto sample=samples[index];
    float3 slavePoint, bodyPoint, normal;
    float closure, area, tangentialSlip;
    float3x3 tangent;
    const bool valid =
        numi_matter_metal::humanFEMBodyContactStateForSample(
            index,dispatch,sample,ranges,triangles,bvhNodes,
            nodes[sample.slaveNode],
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

namespace numi_matter_metal {
kernel void nm_numi_human_assemble_fem_body_contact_loads_dense_oracle(
    constant NMNumiHumanTendonFEMLoadDispatchGPU& dispatch [[buffer(0)]],
    device const NMNumiHumanFEMBodyContactSampleGPU* samples [[buffer(1)]],
    device const NMFEMNodeStateGPU* acceptedNodes [[buffer(2)]],
    device const MRArticulatedBodyPoseGPU* bodyPoses [[buffer(3)]],
    device float4* externalForces [[buffer(4)]],
    device const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU*
        femBodyContactNeighborhoodRanges [[buffer(5)]],
    device const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU*
        femBodyContactNeighborhoodTriangles [[buffer(6)]],
    device const NMNumiHumanFEMBodyContactBVHNodeGPU*
        femBodyContactBVHNodes [[buffer(7)]],
    uint global [[thread_position_in_grid]]
) {
    const uint total = dispatch.environmentCount * dispatch.femNodeCount;
    if (global >= total || dispatch.femBodyContactSampleCount == 0u) return;
    const uint environment = global / dispatch.femNodeCount;
    const uint node = global - environment * dispatch.femNodeCount;
    const uint nodeBase = environment * dispatch.femNodeCount;
    const uint poseBase = environment * dispatch.bodyPoseStride;
    float3 nodeForce = float3(0.0f);
    for (uint index = 0u; index < dispatch.femBodyContactSampleCount; ++index) {
        const NMNumiHumanFEMBodyContactSampleGPU sample = samples[index];
        if (sample.slaveNode != node || !validBody(sample.bodyIndex, dispatch))
            continue;
        float3 slavePoint;
        float3 bodyPoint;
        float3 normal;
        float closure;
        float contactArea;
        float3x3 contactTangent;
        float tangentialSlip;
        if (!humanFEMBodyContactStateForSample(
                index, dispatch, sample, femBodyContactNeighborhoodRanges,
                femBodyContactNeighborhoodTriangles, femBodyContactBVHNodes,
                acceptedNodes[nodeBase + node],
                bodyPoses[poseBase + sample.bodyIndex -
                    dispatch.articulationFirstBody],
                slavePoint, bodyPoint, normal, closure, contactArea,
                contactTangent, tangentialSlip)) return;
        const bool triangle =
            sample.flags == (NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
                             NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT);
        const float pressureOverclosure = triangle
            ? sample.bodyLocalNormalAndReferenceSeparation.w
            : sample.stiffnessAndNormalStrainPerPressure.x;
        const float pressure = pressureOverclosure * closure;
        nodeForce += normal * (pressure * contactArea);
    }
    externalForces[global].xyz += nodeForce;
}

}


kernel void nm_numi_human_static_complete_bvh_coupon(
    constant NMNumiHumanTendonFEMLoadDispatchGPU& dispatch [[buffer(0)]],
    device const NMNumiHumanFEMBodyContactSampleGPU* samples [[buffer(1)]],
    device const NMFEMNodeStateGPU* nodes [[buffer(2)]],
    device const MRArticulatedBodyPoseGPU* poses [[buffer(3)]],
    device float4* output [[buffer(4)]],
    device const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU*
        ranges [[buffer(5)]],
    device const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU*
        triangles [[buffer(6)]],
    device const NMNumiHumanFEMBodyContactBVHNodeGPU*
        bvhNodes [[buffer(7)]],
    uint index [[thread_position_in_grid]]
) {
    const auto sample=samples[index];
    float3 slavePoint,bodyPoint,normal;
    float closure,area,slip;
    float3x3 tangent;
    const bool valid=numi_matter_metal::humanFEMBodyContactStateForSample(
        index,dispatch,sample,ranges,triangles,bvhNodes,
        nodes[sample.slaveNode],poses[sample.bodyIndex],slavePoint,bodyPoint,
        normal,closure,area,tangent,slip);
    if (!valid) {
        output[index*2u]=float4(NAN);
        output[index*2u+1u]=float4(NAN);
        return;
    }
    output[index*2u]=float4(bodyPoint,closure);
    output[index*2u+1u]=float4(normal,area);
}


kernel void nm_numi_human_contact_abi16_mismatch_coupon(
    constant NMNumiHumanTendonFEMLoadDispatchGPU& dispatch [[buffer(0)]],
    device float4* output [[buffer(1)]],
    uint global [[thread_position_in_grid]]
) {
    if (global != 0u) return;
    NMNumiHumanFEMBodyContactSampleGPU sample{};
    sample.slaveNode = 0u;
    sample.bodyIndex = 0u;
    sample.flags = NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
        NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT;
    NMFEMNodeStateGPU node{};
    node.positionAndMass = float4(0.25f, 0.25f, 0.01f, 1.0f);
    MRArticulatedBodyPoseGPU pose{};
    pose.position = float4(0.0f);
    pose.orientation = float4(0.0f, 0.0f, 0.0f, 1.0f);
    float3 slavePoint, bodyPoint, normal;
    float closure, area, slip;
    float3x3 tangent;
    const bool valid = numi_matter_metal::humanFEMBodyContactStateForSample(
        0u, dispatch, sample, nullptr, nullptr, nullptr, node, pose,
        slavePoint, bodyPoint, normal, closure, area, tangent, slip);
    output[0] = float4(valid ? 1.0f : 0.0f,
        float(dispatch.abiVersion),
        float(dispatch.femBodyContactCandidateTriangleCount), 0.0f);
}

kernel void nm_numi_human_legacy_precision_coupon(
    constant NMNumiHumanTendonFEMLoadDispatchGPU& dispatch [[buffer(0)]],
    device const NMNumiHumanFEMBodyContactSampleGPU* samples [[buffer(1)]],
    device const NMFEMNodeStateGPU* nodes [[buffer(2)]],
    device const MRArticulatedBodyPoseGPU* poses [[buffer(3)]],
    device float4* output [[buffer(4)]],
    device const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU*
        ranges [[buffer(5)]],
    device const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU*
        triangles [[buffer(6)]],
    device const NMNumiHumanFEMBodyContactBVHNodeGPU*
        bvhNodes [[buffer(7)]],
    uint index [[thread_position_in_grid]]
) {
    const auto sample=samples[index];
    float3 slavePoint,bodyPoint,normal;
    float closure,area,slip;
    float3x3 tangent;
    const bool valid=numi_matter_metal::humanFEMBodyContactStateForSample(
        index,dispatch,sample,ranges,triangles,bvhNodes,
        nodes[sample.slaveNode],poses[sample.bodyIndex],slavePoint,bodyPoint,
        normal,closure,area,tangent,slip);
    if (!valid) {
        output[index*2u]=float4(NAN);
        output[index*2u+1u]=float4(NAN);
        return;
    }
    output[index*2u]=float4(bodyPoint,closure);
    output[index*2u+1u]=float4(normal,area);
}

kernel void nm_numi_human_closed_pseudonormal_state_coupon(
    constant NMNumiHumanTendonFEMLoadDispatchGPU& dispatch [[buffer(0)]],
    device const NMNumiHumanFEMBodyContactSampleGPU* samples [[buffer(1)]],
    device const NMFEMNodeStateGPU* nodes [[buffer(2)]],
    device const MRArticulatedBodyPoseGPU* poses [[buffer(3)]],
    device float4* output [[buffer(4)]],
    device const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU*
        ranges [[buffer(5)]],
    device const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU*
        triangles [[buffer(6)]],
    device const NMNumiHumanFEMBodyContactBVHNodeGPU* bvhNodes [[buffer(7)]],
    uint index [[thread_position_in_grid]]
) {
    const auto sample=samples[index];
    float3 slavePoint,bodyPoint,normal;
    float closure,area,slip;
    float3x3 tangent;
    const bool valid=numi_matter_metal::humanFEMBodyContactStateForSample(
        index,dispatch,sample,ranges,triangles,bvhNodes,nodes[sample.slaveNode],
        poses[index],slavePoint,bodyPoint,normal,closure,area,tangent,slip);
    if(!valid) {
        output[index*8u]=float4(NAN);
        return;
    }
    const float stiffness=sample.bodyLocalNormalAndReferenceSeparation.w;
    const float3 force=normal*(stiffness*closure*area);
    const float3 reaction=-force;
    output[index*8u+0u]=float4(force,closure);
    output[index*8u+1u]=float4(reaction,area);
    output[index*8u+2u]=float4(bodyPoint,slip);
    output[index*8u+3u]=float4(tangent[0],0.0f);
    output[index*8u+4u]=float4(tangent[1],0.0f);
    output[index*8u+5u]=float4(tangent[2],0.0f);
    output[index*8u+6u]=float4(cross(bodyPoint-poses[index].position.xyz,reaction),0.0f);
    output[index*8u+7u]=float4(normal,1.0f);
}
