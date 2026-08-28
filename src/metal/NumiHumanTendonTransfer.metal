#include <metal_stdlib>

#include "metalrobo/mujoco_muscle_gpu.h"
#include "metalrobo/numi_human_tendon_gpu.h"

using namespace metal;

namespace {

inline bool finite4(const float4 value) { return all(isfinite(value)); }

inline float3 quaternionRotate(const float4 quaternion, const float3 value) {
    const float3 doubledCross = 2.0f * cross(quaternion.xyz, value);
    return value + quaternion.w * doubledCross + cross(quaternion.xyz, doubledCross);
}

inline float3 quaternionConjugateRotate(const float4 quaternion, const float3 value) {
    const float4 conjugate = float4(-quaternion.xyz, quaternion.w);
    return quaternionRotate(conjugate, value);
}

inline float3 mappedForce(
    const MRNumiHumanTendonEnvelopeGPU envelope,
    const uint node,
    const float3 force
) {
    const uint base = 3u * node;
    return float3(
        dot(envelope.forceMapRows[base].xyz, force),
        dot(envelope.forceMapRows[base + 1u].xyz, force),
        dot(envelope.forceMapRows[base + 2u].xyz, force)
    );
}

inline float3 pointJacobian(
    const uint environment,
    const uint dof,
    const MRNumiHumanTendonTransferDispatchGPU dispatch,
    device const MRArticulatedBodyPoseGPU* bodyPoses,
    device const float* pointJacobians,
    const uint bodyIndex,
    const float3 worldPoint
) {
    const uint localBody = bodyIndex - dispatch.articulationFirstBody;
    const MRArticulatedBodyPoseGPU pose = bodyPoses[
        environment * dispatch.bodyPoseStride + localBody
    ];
    const uint bodyPoint = dispatch.bodyJacobianPointOffset +
        localBody * dispatch.bodyJacobianPointStride;
    const uint centerBase = environment * dispatch.pointJacobianStride +
        bodyPoint * 3u * dispatch.dofCount;
    const float3 center = float3(
        pointJacobians[centerBase + dof],
        pointJacobians[centerBase + dispatch.dofCount + dof],
        pointJacobians[centerBase + 2u * dispatch.dofCount + dof]
    );
    float3 angular = float3(0.0f);
    for (uint axis = 0u; axis < 3u; ++axis) {
        const float3 localAxis = axis == 0u
            ? float3(1.0f, 0.0f, 0.0f)
            : (axis == 1u
                ? float3(0.0f, 1.0f, 0.0f)
                : float3(0.0f, 0.0f, 1.0f));
        const float3 worldAxis = quaternionRotate(pose.orientation, localAxis);
        const uint axisBase = centerBase +
            (axis + 1u) * 3u * dispatch.dofCount;
        const float3 probe = float3(
            pointJacobians[axisBase + dof],
            pointJacobians[axisBase + dispatch.dofCount + dof],
            pointJacobians[axisBase + 2u * dispatch.dofCount + dof]
        );
        angular += 0.5f * cross(worldAxis, probe - center);
    }
    return center + cross(angular, worldPoint - pose.position.xyz);
}

} // namespace

kernel void mr_numi_human_tendon_transfer(
    constant MRNumiHumanTendonTransferDispatchGPU& dispatch [[buffer(0)]],
    device const MRNumiHumanTendonBindingGPU* bindings [[buffer(1)]],
    device const MRNumiHumanTendonEnvelopeGPU* envelopes [[buffer(2)]],
    device const MRMujocoMuscleResultGPU* muscleResults [[buffer(3)]],
    device const MRArticulatedBodyPoseGPU* bodyPoses [[buffer(4)]],
    device const float* pointJacobians [[buffer(5)]],
    device MRNumiHumanTendonTransferResultGPU* results [[buffer(6)]],
    device float* generalizedCorrections [[buffer(7)]],
    uint globalIndex [[thread_position_in_grid]]
) {
    if (dispatch.endpointCount == 0u ||
        globalIndex >= dispatch.environmentCount * dispatch.endpointCount) {
        return;
    }
    const uint environment = globalIndex / dispatch.endpointCount;
    const uint bindingIndex = globalIndex - environment * dispatch.endpointCount;
    const uint correctionBase = globalIndex * dispatch.dofCount;
    for (uint dof = 0u; dof < dispatch.dofCount; ++dof) {
        generalizedCorrections[correctionBase + dof] = 0.0f;
    }
    MRNumiHumanTendonTransferResultGPU result{};
    result.status = MR_NUMI_HUMAN_TENDON_TRANSFER_INVALID_DISPATCH;
    result.environment = environment;
    result.bindingIndex = bindingIndex;
    result.envelopeIndex = MR_INVALID_INDEX;
    if (dispatch.abiVersion != MR_NUMI_HUMAN_TENDON_TRANSFER_GPU_ABI_VERSION ||
        dispatch.reserved0 != 0u || dispatch.endpointCount != 2u * dispatch.muscleCount ||
        dispatch.environmentCount == 0u || dispatch.dofCount == 0u ||
        dispatch.bodyPoseStride == 0u || dispatch.pointJacobianStride == 0u ||
        dispatch.bodyJacobianPointStride != 4u) {
        results[globalIndex] = result;
        return;
    }
    const uint pointCount = dispatch.pointJacobianStride /
        (3u * dispatch.dofCount);
    const MRNumiHumanTendonBindingGPU binding = bindings[bindingIndex];
    if (binding.muscleIndex >= dispatch.muscleCount || binding.endpointOrdinal > 1u ||
        binding.bodyIndex < dispatch.articulationFirstBody ||
        binding.bodyIndex - dispatch.articulationFirstBody >= dispatch.bodyPoseStride ||
        dispatch.bodyJacobianPointOffset > pointCount ||
        binding.bodyIndex - dispatch.articulationFirstBody >
            (pointCount - dispatch.bodyJacobianPointOffset) /
                dispatch.bodyJacobianPointStride ||
        binding.reserved0 != 0u || binding.reserved1 != 0u ||
        !finite4(binding.sourceLocalPoint) || binding.sourceLocalPoint.w != 0.0f) {
        result.status = MR_NUMI_HUMAN_TENDON_TRANSFER_INVALID_BINDING;
        results[globalIndex] = result;
        return;
    }
    const MRMujocoMuscleResultGPU muscle = muscleResults[
        environment * dispatch.muscleCount + binding.muscleIndex
    ];
    const float4 gradient = muscle.endpointLengthGradients[binding.endpointOrdinal];
    if (muscle.status != MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS ||
        muscle.environment != environment || muscle.muscleIndex != binding.muscleIndex ||
        !finite4(gradient) || gradient.w != 0.0f ||
        !finite4(muscle.activeForceAndReserved) ||
        any(muscle.activeForceAndReserved.yzw != float3(0.0f))) {
        result.status = MR_NUMI_HUMAN_TENDON_TRANSFER_INVALID_MUSCLE_RESULT;
        results[globalIndex] = result;
        return;
    }
    const float representedForce = muscle.activeForceAndReserved.x;
    const float3 terminalWorldForce = representedForce * gradient.xyz;
    if (!all(isfinite(terminalWorldForce))) {
        result.status = MR_NUMI_HUMAN_TENDON_TRANSFER_NONFINITE_RESULT;
        results[globalIndex] = result;
        return;
    }
    result.terminalWorldForce = float4(terminalWorldForce, 0.0f);
    result.residualsAndForce.w = representedForce;
    if (binding.mode == MR_NUMI_HUMAN_TENDON_TRANSFER_SOURCE_POINT) {
        if (binding.envelopeIndex != MR_INVALID_INDEX || binding.boneStableId != 0u) {
            result.status = MR_NUMI_HUMAN_TENDON_TRANSFER_INVALID_BINDING;
            results[globalIndex] = result;
            return;
        }
        result.nodalWorldForces[0] = float4(terminalWorldForce, 0.0f);
        result.status = MR_NUMI_HUMAN_TENDON_TRANSFER_SUCCESS;
        results[globalIndex] = result;
        return;
    }
    if (binding.mode != MR_NUMI_HUMAN_TENDON_TRANSFER_DISTRIBUTED_ENVELOPE ||
        binding.envelopeIndex >= dispatch.envelopeCount || binding.boneStableId == 0u) {
        result.status = MR_NUMI_HUMAN_TENDON_TRANSFER_INVALID_BINDING;
        results[globalIndex] = result;
        return;
    }
    const MRNumiHumanTendonEnvelopeGPU envelope = envelopes[binding.envelopeIndex];
    if (envelope.bodyIndex != binding.bodyIndex ||
        envelope.boneStableId != binding.boneStableId || envelope.nodeCount != 4u ||
        !finite4(envelope.metrics) || !(envelope.metrics.y > 0.0f) ||
        !(envelope.metrics.z > 0.0f) || !(envelope.metrics.w > 0.0f)) {
        result.status = MR_NUMI_HUMAN_TENDON_TRANSFER_INVALID_BINDING;
        results[globalIndex] = result;
        return;
    }
    const uint localBody = binding.bodyIndex - dispatch.articulationFirstBody;
    const MRArticulatedBodyPoseGPU pose = bodyPoses[
        environment * dispatch.bodyPoseStride + localBody
    ];
    if (!finite4(pose.position) || !finite4(pose.orientation)) {
        result.status = MR_NUMI_HUMAN_TENDON_TRANSFER_INVALID_BINDING;
        results[globalIndex] = result;
        return;
    }
    const float3 terminalLocalForce = quaternionConjugateRotate(
        pose.orientation, terminalWorldForce
    );
    float3 localResultant = float3(0.0f);
    float3 localMoment = float3(0.0f);
    float3 localNodalForces[4];
    float3 worldNodes[4];
    const float3 sourceWorld = pose.position.xyz + quaternionRotate(
        pose.orientation, binding.sourceLocalPoint.xyz
    );
    for (uint node = 0u; node < 4u; ++node) {
        if (!finite4(envelope.localNodes[node])) {
            result.status = MR_NUMI_HUMAN_TENDON_TRANSFER_INVALID_BINDING;
            results[globalIndex] = result;
            return;
        }
        for (uint row = 0u; row < 3u; ++row) {
            if (!finite4(envelope.forceMapRows[3u * node + row]) ||
                envelope.forceMapRows[3u * node + row].w != 0.0f) {
                result.status = MR_NUMI_HUMAN_TENDON_TRANSFER_INVALID_BINDING;
                results[globalIndex] = result;
                return;
            }
        }
        localNodalForces[node] = mappedForce(envelope, node, terminalLocalForce);
        worldNodes[node] = pose.position.xyz + quaternionRotate(
            pose.orientation, envelope.localNodes[node].xyz
        );
        const float3 worldForce = quaternionRotate(
            pose.orientation, localNodalForces[node]
        );
        result.nodalWorldForces[node] = float4(worldForce, 0.0f);
        localResultant += localNodalForces[node];
        localMoment += cross(
            envelope.localNodes[node].xyz - binding.sourceLocalPoint.xyz,
            localNodalForces[node]
        );
    }
    const float forceResidual = length(localResultant - terminalLocalForce);
    const float momentResidual = length(localMoment);
    float maximumCorrection = 0.0f;
    for (uint dof = 0u; dof < dispatch.dofCount; ++dof) {
        const float sourceGeneralized = dot(
            terminalWorldForce,
            pointJacobian(environment, dof, dispatch, bodyPoses, pointJacobians,
                          binding.bodyIndex, sourceWorld)
        );
        float distributedGeneralized = 0.0f;
        for (uint node = 0u; node < 4u; ++node) {
            distributedGeneralized += dot(
                result.nodalWorldForces[node].xyz,
                pointJacobian(environment, dof, dispatch, bodyPoses, pointJacobians,
                              binding.bodyIndex, worldNodes[node])
            );
        }
        const float correction = distributedGeneralized - sourceGeneralized;
        generalizedCorrections[correctionBase + dof] = correction;
        maximumCorrection = max(maximumCorrection, abs(correction));
    }
    if (!isfinite(forceResidual) || !isfinite(momentResidual) ||
        !isfinite(maximumCorrection)) {
        result.status = MR_NUMI_HUMAN_TENDON_TRANSFER_NONFINITE_RESULT;
        results[globalIndex] = result;
        return;
    }
    result.envelopeIndex = binding.envelopeIndex;
    result.residualsAndForce.xyz = float3(
        forceResidual, momentResidual, maximumCorrection
    );
    result.status = MR_NUMI_HUMAN_TENDON_TRANSFER_SUCCESS;
    results[globalIndex] = result;
}
