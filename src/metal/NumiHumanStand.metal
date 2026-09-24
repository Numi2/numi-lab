#include <metal_stdlib>

#include "metalrobo/numi_human_joint_equality_gpu.h"
#include "metalrobo/numi_human_stand_gpu.h"
#include "metalrobo/numi_human_constraint_projection.h"
#include "metalrobo/numi_human_friction.h"
#include "metalrobo/numi_human_bilateral.h"
#include "metalrobo/numi_human_passive_joint.h"
#include "metalrobo/mujoco_muscle_gpu.h"
#include "metalrobo/numi_human_tendon_gpu.h"

using namespace metal;

namespace {

constant float kPivotFloor = 1.0e-10f;
constant float kResponseRegularization = 1.0e-7f;
// 16 KiB factor cache leaves room for the three 160-entry work vectors on
// devices with 32 KiB threadgroup memory. Larger authored blocks retain the
// same device factor path; the supported equality count is unchanged.
constant uint kCachedEqualityCapacity = 64u;

inline bool finite4(const float4 value) { return all(isfinite(value)); }

inline float4 quaternionConjugate(const float4 value) {
    return float4(-value.xyz, value.w);
}

inline float4 quaternionMultiply(const float4 left, const float4 right) {
    return float4(
        left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
        left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
        left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
        left.w * right.w - dot(left.xyz, right.xyz)
    );
}

inline float3 quaternionRotate(const float4 quaternion, const float3 value) {
    const float3 doubledCross = 2.0f * cross(quaternion.xyz, value);
    return value + quaternion.w * doubledCross + cross(quaternion.xyz, doubledCross);
}

inline bool normalizedQuaternion(const float4 input, thread float4& output) {
    const float normSquared = dot(input, input);
    if (!finite4(input) || !(normSquared > 1.0e-12f) || !isfinite(normSquared)) {
        return false;
    }
    output = input * rsqrt(normSquared);
    return finite4(output);
}

inline float4 quaternionFromRotationVector(const float3 rotationVector) {
    const float angleSquared = dot(rotationVector, rotationVector);
    if (angleSquared < 1.0e-12f) {
        return normalize(float4(0.5f * rotationVector, 1.0f));
    }
    const float angle = sqrt(angleSquared);
    return normalize(float4(
        rotationVector * (sin(0.5f * angle) / angle), cos(0.5f * angle)
    ));
}

inline float3 worldInertiaMultiply(
    device const MRBodyPropertiesGPU& body,
    const float4 orientation,
    const float3 worldVector
) {
    const float3 local = quaternionRotate(
        quaternionConjugate(orientation), worldVector
    );
    const float3 localResult{
        dot(body.inertiaRow0.xyz, local),
        dot(body.inertiaRow1.xyz, local),
        dot(body.inertiaRow2.xyz, local),
    };
    return quaternionRotate(orientation, localResult);
}

template <typename WorkspacePointer, typename OutputPointer>
inline bool solveFactor(
    device const float* factor,
    WorkspacePointer workspace,
    OutputPointer output,
    const uint nv
) {
    for (uint row = 0u; row < nv; ++row) {
        float value = output[row];
        for (uint column = 0u; column < row; ++column) {
            value -= factor[row * nv + column] * workspace[column];
        }
        const float diagonal = factor[row * nv + row];
        if (!(diagonal > 0.0f) || !isfinite(diagonal)) return false;
        workspace[row] = value / diagonal;
    }
    for (uint reverse = 0u; reverse < nv; ++reverse) {
        const uint row = nv - 1u - reverse;
        float value = workspace[row];
        for (uint column = row + 1u; column < nv; ++column) {
            value -= factor[column * nv + row] * output[column];
        }
        output[row] = value / factor[row * nv + row];
        if (!isfinite(output[row])) return false;
    }
    return true;
}

inline float pointJacobianAxis(
    device const float* pointJacobians,
    const uint base,
    const uint point,
    const uint nv,
    const uint dof,
    const float3 direction
) {
    const uint pointBase = base + point * 3u * nv;
    return direction.x * pointJacobians[pointBase + 0u * nv + dof] +
        direction.y * pointJacobians[pointBase + 1u * nv + dof] +
        direction.z * pointJacobians[pointBase + 2u * nv + dof];
}

inline bool evaluateJointEquality(
    device const MRNumiHumanJointEqualityGPU& equality,
    device const float* q,
    const uint qBase,
    const uint nq,
    const uint nv,
    thread float& target,
    thread float& derivative,
    thread float& error
) {
    const bool fixed = equality.indices.z == MR_INVALID_INDEX &&
        equality.indices.w == MR_INVALID_INDEX;
    const bool coupled = equality.indices.z < nq && equality.indices.w < nv;
    if (equality.indices.x >= nq || equality.indices.y >= nv ||
        (!fixed && !coupled) ||
        (coupled && (equality.indices.x == equality.indices.z ||
                     equality.indices.y == equality.indices.w)) ||
        !finite4(equality.referencesAndCoefficients0) ||
        !finite4(equality.coefficients1) || !finite4(equality.solref) ||
        !finite4(equality.solimp0) || !finite4(equality.solimp1) ||
        equality.coefficients1.w != 0.0f || equality.solref.z != 0.0f ||
        equality.solref.w != 0.0f || equality.solimp1.y != 0.0f ||
        equality.solimp1.z != 0.0f || equality.solimp1.w != 0.0f) {
        return false;
    }
    const float delta = fixed
        ? 0.0f
        : q[qBase + equality.indices.z] -
            equality.referencesAndCoefficients0.y;
    const float a0 = equality.referencesAndCoefficients0.z;
    const float a1 = equality.referencesAndCoefficients0.w;
    const float a2 = equality.coefficients1.x;
    const float a3 = equality.coefficients1.y;
    const float a4 = equality.coefficients1.z;
    const float polynomial = a0 + delta * (
        a1 + delta * (a2 + delta * (a3 + delta * a4))
    );
    derivative = fixed
        ? 0.0f
        : a1 + delta * (
            2.0f * a2 + delta * (3.0f * a3 + 4.0f * delta * a4)
        );
    target = equality.referencesAndCoefficients0.x + polynomial;
    error = q[qBase + equality.indices.x] - target;
    return isfinite(target) && isfinite(derivative) && isfinite(error);
}

inline void fail(
    device MRNumiHumanStandStatusGPU& status,
    const uint code,
    const uint index
) {
    if (status.code == MR_NUMI_HUMAN_STAND_SUCCESS) {
        status.code = code;
        status.failingIndex = index;
    }
}

} // namespace

// Large-state Human dynamics deliberately consumes the already-authoritative
// Metal kinematics/Jacobian and MyoSim J^T streams. Matrix assembly is spread
// across the threadgroup; lane zero performs deterministic Cholesky, source
// support projection, and state publication. The first release retains a
// low-velocity bias model (gravity, gyroscopic and authored body damping) and
// exposes that evidence boundary to the host rather than pretending to be an
// exact high-speed RNEA replacement.
kernel void mr_numi_human_stand_step(
    device const MRWorldGPU* worlds [[buffer(0)]],
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const MRDofPropertiesGPU* dofs [[buffer(2)]],
    device const MRBodyPropertiesGPU* bodies [[buffer(3)]],
    constant const MRNumiHumanStandDispatchGPU& dispatch [[buffer(4)]],
    device float* qState [[buffer(5)]],
    device float* vState [[buffer(6)]],
    device const MRArticulatedBodyPoseGPU* bodyPoses [[buffer(7)]],
    device const MRArticulatedPointWorldGPU* pointWorld [[buffer(8)]],
    device const float* pointJacobians [[buffer(9)]],
    device const float* generalizedForceWorkspace [[buffer(10)]],
    device const MRNumiHumanStandContactGPU* contacts [[buffer(11)]],
    device float* spatialJacobianScratch [[buffer(12)]],
    device float4* bodyMotionScratch [[buffer(13)]],
    device float* factorScratch [[buffer(14)]],
    device float* vectorScratch [[buffer(15)]],
    device float* responseScratch [[buffer(16)]],
    device MRNumiHumanStandStatusGPU* statuses [[buffer(17)]],
    device const MRNumiHumanTendonBindingGPU* tendonBindings [[buffer(18)]],
    device const MRNumiHumanTendonTransferResultGPU* tendonTransfers [[buffer(19)]],
    device const MRNumiHumanJointEqualityGPU* jointEqualities [[buffer(20)]],
    device MRCompensatedRootTranslationGPU* rootTranslations [[buffer(21)]],
    device const float4* bodyPositionLow [[buffer(22)]],
    device const float4* pointPositionLow [[buffer(23)]],
    device const float* passiveJointProgram [[buffer(24)]],
    device float* sourceDynamicsWitness [[buffer(25)]],
    uint environment [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint threadCount [[threads_per_threadgroup]]
) {
    if (environment >= dispatch.environmentCount) return;
    device MRNumiHumanStandStatusGPU& status = statuses[environment];
    device const MRWorldGPU& world = worlds[0];
    device const MRArticulationGPU& articulation =
        articulations[dispatch.articulationIndex];
    const uint bodyCount = articulation.bodyCount;
    const uint nv = articulation.nv;
    const uint nq = articulation.nq;
    const uint qBase = environment * dispatch.qStride;
    const uint vBase = environment * dispatch.vStride;
    const uint bodyPoseBase = environment * dispatch.bodyPoseStride;
    const uint pointBase = environment * dispatch.pointWorldStride;
    const uint pointJacobianBase = environment * dispatch.pointJacobianStride;
    const uint forceBase = environment * dispatch.generalizedForceStride +
        dispatch.generalizedForceOffset;
    const uint spatialBase = environment * bodyCount *
        MR_NUMI_HUMAN_STAND_SPATIAL_SCRATCH_ROWS * nv;
    const uint inertiaWeightedBase = spatialBase + bodyCount * 6u * nv;
    const uint bodyMotionBase = environment * bodyCount * 2u;
    const uint factorBase = environment * nv * nv;
    const uint vectorStride = nv + 3u * nv +
        12u * dispatch.supportContactCount + dispatch.jointEqualityCount +
        (((dispatch.flags & MR_NUMI_HUMAN_STAND_EXPORT_SOURCE_LIMIT_IMPULSES) != 0u)
            ? nv + nq + nv : 0u);
    const uint preloadBase = environment * vectorStride;
    const uint vectorBase = preloadBase + nv;
    const uint equalityCount = dispatch.jointEqualityCount;
    const uint responseColumns = (dispatch.supportContactCount * 3u + equalityCount + nv) * nv;
    const uint responseStride = responseColumns + equalityCount * (equalityCount + 3u) +
        nv * equalityCount;
    const uint responseBase = environment * responseStride;
    device float* equalityFactor = responseScratch + responseBase + responseColumns;
    device float* equalityScale = equalityFactor + equalityCount * equalityCount;
    device float* equalityPivots = equalityScale + equalityCount;
    // These vectors are repeatedly read and updated by the scalar constraint
    // solve. Keep them in the threadgroup instead of round-tripping every
    // scalar dependency through device memory. Retain the exported scratch
    // layout and publish its final values below.
    threadgroup float equalityRhsStorage[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup float equalityFactorStorage[
        kCachedEqualityCapacity * kCachedEqualityCapacity];
    threadgroup float candidateVStorage[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup float workspaceStorage[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup float* equalityRhs = equalityRhsStorage;
    // Equality multiplier corrections for each conditioned limit response.
    device float* limitEqualityCorrections = equalityPivots + 2u * equalityCount;
    device float* bias = vectorScratch + vectorBase;
    threadgroup float* candidateV = candidateVStorage;
    threadgroup float* workspace = workspaceStorage;
    device float* lambdas = bias + 3u * nv;
    device float* equalityLambdas =
        lambdas + 3u * dispatch.supportContactCount;
    device float* contactMatrices =
        equalityLambdas + dispatch.jointEqualityCount;
    device float* sourceLimitImpulseEvidence =
        contactMatrices + 9u * dispatch.supportContactCount;
    device float* preProjectionQEvidence = sourceLimitImpulseEvidence + nv;
    device float* preProjectionVEvidence = preProjectionQEvidence + nq;
    device float* factor = factorScratch + factorBase;
    const bool captureSourceDynamics =
        (dispatch.flags & MR_NUMI_HUMAN_STAND_PREDICT_VELOCITY_ONLY) != 0u;
    const uint sourceDynamicsBase = environment * 3u * nv;

    if (lane == 0u) {
        if (dispatch.stepIndex == 0u) {
            status = {};
            status.code = MR_NUMI_HUMAN_STAND_SUCCESS;
            status.environment = environment;
            status.failingIndex = MR_INVALID_INDEX;
            status.jointEqualityCounts.w = MR_INVALID_INDEX;
            status.constraintImpulseOwners = uint4(MR_INVALID_INDEX);
            status.velocityDiagnosticOwners = uint4(MR_INVALID_INDEX);
            status.contactAndAcceleration.x =
                (dispatch.flags & MR_NUMI_HUMAN_STAND_ENABLE_CONTACT) != 0u &&
                    dispatch.supportContactCount != 0u
                    ? INFINITY
                    : 0.0f;
            status.factorAndAssistance.x = INFINITY;
            for (uint index = 0u;
                 index < 3u * dispatch.supportContactCount;
                 ++index) {
                lambdas[index] = 0.0f;
            }
        }
        if (dispatch.abiVersion != MR_NUMI_HUMAN_STAND_ABI_VERSION ||
            dispatch.stepCount == 0u ||
            dispatch.stepCount > MR_NUMI_HUMAN_STAND_MAX_HORIZON_STEPS ||
            dispatch.stepIndex >= dispatch.stepCount ||
            dispatch.articulationIndex >= world.articulationCount ||
            dispatch.qStride < nq || dispatch.vStride < nv ||
            dispatch.bodyPoseStride < bodyCount ||
            dispatch.generalizedForceStride < nv ||
            dispatch.bodyJacobianPointOffset > dispatch.pointWorldStride ||
            bodyCount >
                (dispatch.pointWorldStride -
                 dispatch.bodyJacobianPointOffset) / 4u ||
            dispatch.pointJacobianStride /
                max(3u * nv, 1u) < dispatch.pointWorldStride ||
            dispatch.supportContactCount > MR_NUMI_HUMAN_STAND_MAX_CONTACTS ||
            dispatch.jointEqualityCount > nv ||
            dispatch.contactIterationCount == 0u ||
            dispatch.contactIterationCount > 64u ||
            !(dispatch.groundPointAndTimestep.w > 0.0f) ||
            !finite4(dispatch.groundPointAndTimestep) ||
            !finite4(dispatch.groundNormal) ||
            !finite4(dispatch.targetRootPosition) ||
            !finite4(dispatch.targetRootOrientation) ||
            !finite4(dispatch.assistanceGains) ||
            !mrNumiHumanTimedRootForceValid(
                dispatch.timedRootForce, dispatch.stepCount) ||
            dispatch.groundNormal.w != 0.0f ||
            dispatch.targetRootPosition.w != 0.0f ||
            dispatch.tendonTransferStride < dispatch.tendonEndpointCount ||
            ((dispatch.tendonEndpointCount == 0u) !=
             ((dispatch.flags & MR_NUMI_HUMAN_STAND_HAS_TENDON_LOADS) == 0u)) ||
            (dispatch.tendonEndpointCount != 0u &&
             (dispatch.tendonEndpointCount % 2u) != 0u) ||
            ((dispatch.jointEqualityCount == 0u) !=
             ((dispatch.flags &
               MR_NUMI_HUMAN_STAND_HAS_JOINT_EQUALITIES) == 0u)) ||
            (dispatch.flags & ~(
                MR_NUMI_HUMAN_STAND_ENABLE_CONTACT |
                MR_NUMI_HUMAN_STAND_ENABLE_ROOT_ASSISTANCE |
                MR_NUMI_HUMAN_STAND_HAS_TENDON_LOADS |
                MR_NUMI_HUMAN_STAND_HAS_JOINT_EQUALITIES |
                MR_NUMI_HUMAN_STAND_PREDICT_VELOCITY_ONLY |
                MR_NUMI_HUMAN_STAND_HAS_PASSIVE_JOINT_PROGRAM |
                MR_NUMI_HUMAN_STAND_EXPORT_SOURCE_LIMIT_IMPULSES |
                MR_NUMI_HUMAN_STAND_PREPARE_ONLY
            )) != 0u ||
            ((dispatch.flags & MR_NUMI_HUMAN_STAND_PREPARE_ONLY) != 0u &&
             (dispatch.flags & MR_NUMI_HUMAN_STAND_PREDICT_VELOCITY_ONLY) != 0u) ||
            ((dispatch.flags & MR_NUMI_HUMAN_STAND_PREDICT_VELOCITY_ONLY) != 0u &&
             ((dispatch.flags & (MR_NUMI_HUMAN_STAND_ENABLE_CONTACT |
                                 MR_NUMI_HUMAN_STAND_HAS_JOINT_EQUALITIES)) != 0u ||
              dispatch.supportContactCount != 0u ||
              dispatch.jointEqualityCount != 0u))) {
            fail(status, MR_NUMI_HUMAN_STAND_INVALID_DISPATCH, MR_INVALID_INDEX);
        } else if (articulation.rootType != MR_ROOT_FLOATING ||
                   bodyCount == 0u || bodyCount > MR_NUMI_HUMAN_STAND_MAX_BODIES ||
                   nv < 6u || nv > MR_NUMI_HUMAN_STAND_MAX_DOFS ||
                   nq < 7u || nq > MR_NUMI_HUMAN_STAND_MAX_Q ||
                   articulation.firstBody + bodyCount > world.bodyCount ||
                   articulation.vOffset + nv > world.nv ||
                   articulation.qOffset + nq > world.nq) {
            fail(status, MR_NUMI_HUMAN_STAND_INVALID_MODEL, MR_INVALID_INDEX);
        } else {
            const float normalLengthSquared = dot(
                dispatch.groundNormal.xyz, dispatch.groundNormal.xyz
            );
            if (!isfinite(normalLengthSquared) ||
                abs(normalLengthSquared - 1.0f) > 2.0e-4f) {
                fail(status, MR_NUMI_HUMAN_STAND_INVALID_DISPATCH, MR_INVALID_INDEX);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;

    // Validate the exact per-step terminal-load transaction before any Human
    // state is advanced. These loads are wrench-equivalent to MyoSim's
    // existing source-route J^T force; they are exposed to bone/deformable
    // consumers and deliberately are not added here as a second joint torque.
    // A registered pre-dynamics consumer may already have replaced a declared
    // J^T share in generalizedForceWorkspace with a solved anchor reaction.
    if (lane == 0u && dispatch.tendonEndpointCount != 0u) {
        const uint transferBase = environment * dispatch.tendonTransferStride;
        for (uint endpoint = 0u; endpoint < dispatch.tendonEndpointCount;
             ++endpoint) {
            device const MRNumiHumanTendonBindingGPU& binding =
                tendonBindings[endpoint];
            device const MRNumiHumanTendonTransferResultGPU& transfer =
                tendonTransfers[transferBase + endpoint];
            bool validTransfer =
                transfer.status == MR_NUMI_HUMAN_TENDON_TRANSFER_SUCCESS &&
                transfer.environment == environment &&
                transfer.bindingIndex == endpoint &&
                finite4(transfer.terminalWorldForce) &&
                finite4(transfer.residualsAndForce) &&
                transfer.residualsAndForce.x >= 0.0f &&
                transfer.residualsAndForce.y >= 0.0f &&
                transfer.residualsAndForce.z >= 0.0f;
            for (uint node = 0u; node < 4u && validTransfer; ++node) {
                validTransfer = finite4(transfer.nodalWorldForces[node]) &&
                    transfer.nodalWorldForces[node].w == 0.0f;
            }
            if (binding.mode == MR_NUMI_HUMAN_TENDON_TRANSFER_SOURCE_POINT) {
                validTransfer = validTransfer &&
                    transfer.envelopeIndex == MR_INVALID_INDEX;
            } else if (binding.mode ==
                       MR_NUMI_HUMAN_TENDON_TRANSFER_DISTRIBUTED_ENVELOPE) {
                validTransfer = validTransfer &&
                    binding.envelopeIndex < dispatch.tendonEnvelopeCount &&
                    transfer.envelopeIndex == binding.envelopeIndex;
            } else {
                validTransfer = false;
            }
            if (!validTransfer) {
                ++status.tendonFailureCount;
                fail(status, MR_NUMI_HUMAN_STAND_TENDON_TRANSFER_FAILED,
                     endpoint);
                break;
            }
            ++status.tendonTransferCount;
            if (binding.mode ==
                MR_NUMI_HUMAN_TENDON_TRANSFER_DISTRIBUTED_ENVELOPE) {
                ++status.tendonEnvelopeTransferCount;
            } else {
                ++status.tendonPointTransferCount;
            }
            status.tendonDiagnostics = max(
                status.tendonDiagnostics,
                abs(transfer.residualsAndForce)
            );
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;

    // Lane-zero validation avoids racing writes to the diagnostic status.
    if (lane == 0u) {
        for (uint index = 0u; index < nq; ++index) {
            if (!isfinite(qState[qBase + index])) {
                fail(status, MR_NUMI_HUMAN_STAND_NONFINITE_INPUT, index);
                break;
            }
        }
        if (status.code == MR_NUMI_HUMAN_STAND_SUCCESS) {
            for (uint index = 0u; index < nv; ++index) {
                if (!isfinite(vState[vBase + index]) ||
                    !isfinite(generalizedForceWorkspace[forceBase + index])) {
                    fail(status, MR_NUMI_HUMAN_STAND_NONFINITE_INPUT, index);
                    break;
                }
            }
        }
        // Contact impulses start cold. Reusing the old values without first
        // applying them to candidateV would make the projected deltas wrong.
        for (uint index = 0u;
             index < 3u * dispatch.supportContactCount;
             ++index) {
            lambdas[index] = 0.0f;
        }
        for (uint index = 0u;
             index < dispatch.jointEqualityCount;
             ++index) {
            equalityLambdas[index] = 0.0f;
            device const MRNumiHumanJointEqualityGPU& equality =
                jointEqualities[index];
            float target = 0.0f;
            float derivative = 0.0f;
            float error = 0.0f;
            bool valid = evaluateJointEquality(
                equality, qState, qBase, nq, nv,
                target, derivative, error
            );
            valid = valid &&
                dofs[articulation.vOffset + equality.indices.y].qIndex ==
                    articulation.qOffset + equality.indices.x;
            if (valid && equality.indices.w != MR_INVALID_INDEX) {
                valid = dofs[
                    articulation.vOffset + equality.indices.w
                ].qIndex == articulation.qOffset + equality.indices.z;
            }
            for (uint prior = 0u; prior < index && valid; ++prior) {
                const uint priorDependent = jointEqualities[prior].indices.y;
                valid = priorDependent != equality.indices.y &&
                    priorDependent != equality.indices.w;
            }
            for (uint later = index + 1u;
                 later < dispatch.jointEqualityCount && valid; ++later) {
                valid = jointEqualities[later].indices.y != equality.indices.w;
            }
            if (!valid) {
                ++status.jointEqualityCounts.z;
                fail(status, MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED, index);
                break;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;

    // Reconstruct a world spatial Jacobian for every body from its COM and
    // three unit body-axis point probes.
    const uint spatialElements = bodyCount * nv;
    for (uint index = lane; index < spatialElements; index += threadCount) {
        const uint localBody = index / nv;
        const uint dof = index - localBody * nv;
        const uint probe = dispatch.bodyJacobianPointOffset + 4u * localBody;
        const uint probeBase = pointJacobianBase + probe * 3u * nv;
        const float3 linear{
            pointJacobians[probeBase + 0u * nv + dof],
            pointJacobians[probeBase + 1u * nv + dof],
            pointJacobians[probeBase + 2u * nv + dof],
        };
        const float3 dx{
            pointJacobians[probeBase + 3u * nv + 0u * nv + dof] - linear.x,
            pointJacobians[probeBase + 3u * nv + 1u * nv + dof] - linear.y,
            pointJacobians[probeBase + 3u * nv + 2u * nv + dof] - linear.z,
        };
        const float3 dy{
            pointJacobians[probeBase + 6u * nv + 0u * nv + dof] - linear.x,
            pointJacobians[probeBase + 6u * nv + 1u * nv + dof] - linear.y,
            pointJacobians[probeBase + 6u * nv + 2u * nv + dof] - linear.z,
        };
        const float3 dz{
            pointJacobians[probeBase + 9u * nv + 0u * nv + dof] - linear.x,
            pointJacobians[probeBase + 9u * nv + 1u * nv + dof] - linear.y,
            pointJacobians[probeBase + 9u * nv + 2u * nv + dof] - linear.z,
        };
        const float4 orientation = bodyPoses[bodyPoseBase + localBody].orientation;
        const float3 axisX = quaternionRotate(orientation, float3(1.0f, 0.0f, 0.0f));
        const float3 axisY = quaternionRotate(orientation, float3(0.0f, 1.0f, 0.0f));
        const float3 axisZ = quaternionRotate(orientation, float3(0.0f, 0.0f, 1.0f));
        const float3 angular = 0.5f * (
            cross(axisX, dx) + cross(axisY, dy) + cross(axisZ, dz)
        );
        const uint base = spatialBase + localBody * 6u * nv + dof;
        spatialJacobianScratch[base + 0u * nv] = angular.x;
        spatialJacobianScratch[base + 1u * nv] = angular.y;
        spatialJacobianScratch[base + 2u * nv] = angular.z;
        spatialJacobianScratch[base + 3u * nv] = linear.x;
        spatialJacobianScratch[base + 4u * nv] = linear.y;
        spatialJacobianScratch[base + 5u * nv] = linear.z;
        // I_world J_angular is shared by every mass-matrix row. Compute it
        // once per body/column, retaining the same world-inertia operation
        // and body-ordered dot-product reduction used by the original path.
        const float3 inertiaWeighted = worldInertiaMultiply(
            bodies[articulation.firstBody + localBody], orientation, angular);
        const uint weighted = inertiaWeightedBase + localBody * 3u * nv + dof;
        spatialJacobianScratch[weighted + 0u * nv] = inertiaWeighted.x;
        spatialJacobianScratch[weighted + 1u * nv] = inertiaWeighted.y;
        spatialJacobianScratch[weighted + 2u * nv] = inertiaWeighted.z;
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (uint localBody = lane; localBody < bodyCount; localBody += threadCount) {
        float3 angular{0.0f};
        float3 linear{0.0f};
        const uint base = spatialBase + localBody * 6u * nv;
        for (uint dof = 0u; dof < nv; ++dof) {
            const float velocity = vState[vBase + dof];
            angular += velocity * float3(
                spatialJacobianScratch[base + 0u * nv + dof],
                spatialJacobianScratch[base + 1u * nv + dof],
                spatialJacobianScratch[base + 2u * nv + dof]
            );
            linear += velocity * float3(
                spatialJacobianScratch[base + 3u * nv + dof],
                spatialJacobianScratch[base + 4u * nv + dof],
                spatialJacobianScratch[base + 5u * nv + dof]
            );
        }
        bodyMotionScratch[bodyMotionBase + 2u * localBody + 0u] = float4(angular, 0.0f);
        bodyMotionScratch[bodyMotionBase + 2u * localBody + 1u] = float4(linear, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (uint row = lane; row < nv; row += threadCount) {
        float value = 0.0f;
        for (uint localBody = 0u; localBody < bodyCount; ++localBody) {
            const uint globalBody = articulation.firstBody + localBody;
            device const MRBodyPropertiesGPU& body = bodies[globalBody];
            const uint base = spatialBase + localBody * 6u * nv;
            const float3 jw{
                spatialJacobianScratch[base + 0u * nv + row],
                spatialJacobianScratch[base + 1u * nv + row],
                spatialJacobianScratch[base + 2u * nv + row],
            };
            const float3 jv{
                spatialJacobianScratch[base + 3u * nv + row],
                spatialJacobianScratch[base + 4u * nv + row],
                spatialJacobianScratch[base + 5u * nv + row],
            };
            const float3 angular = bodyMotionScratch[
                bodyMotionBase + 2u * localBody + 0u
            ].xyz;
            const float3 linear = bodyMotionScratch[
                bodyMotionBase + 2u * localBody + 1u
            ].xyz;
            const float4 orientation = bodyPoses[bodyPoseBase + localBody].orientation;
            const float3 angularMomentum = worldInertiaMultiply(body, orientation, angular);
            const float3 requiredTorque = cross(angular, angularMomentum) +
                body.dampingAndSpeedLimits.y * angular;
            const float3 requiredForce =
                -body.massAndInverseMass.x * world.gravityAndTimestep.xyz +
                body.dampingAndSpeedLimits.x * linear;
            value += dot(jw, requiredTorque) + dot(jv, requiredForce);
        }
        device const MRDofPropertiesGPU& dof =
            dofs[articulation.vOffset + row];
        // MyoSim joint damping is passive generalized resistance. The Human
        // payload deliberately carries it without MR_DOF_FLAG_DRIVE so these
        // coordinates remain muscle-driven rather than becoming hidden PD
        // motors.
        if ((dof.flags & MR_DOF_FLAG_DRIVE) == 0u) {
            value += dof.drive.y * vState[vBase + row];
        }
        if ((dispatch.flags & MR_NUMI_HUMAN_STAND_HAS_PASSIVE_JOINT_PROGRAM) != 0u) {
            for (uint column = 6u; column < nv; ++column) {
                const float stiffness = passiveJointProgram[row * nv + column];
                if (stiffness == 0.0f) continue;
                const uint sourceQ = dofs[articulation.vOffset + column].qIndex;
                const float displacement = qState[qBase + sourceQ - articulation.qOffset] -
                    passiveJointProgram[nv * nv + column];
                value += mrNumiHumanPassiveImplicitBias(stiffness, displacement,
                    vState[vBase + column], dispatch.groundPointAndTimestep.w);
            }
        }
        bias[row] = value;
        if (captureSourceDynamics) {
            // Preserve the raw device bias before the vector row becomes the
            // free-velocity scratch later in this same kernel.
            sourceDynamicsWitness[sourceDynamicsBase + nv + row] = value;
        }
    }

    const uint matrixElements = nv * nv;
    for (uint index = lane; index < matrixElements; index += threadCount) {
        const uint row = index / nv;
        const uint column = index - row * nv;
        float value = 0.0f;
        for (uint localBody = 0u; localBody < bodyCount; ++localBody) {
            const uint globalBody = articulation.firstBody + localBody;
            device const MRBodyPropertiesGPU& body = bodies[globalBody];
            const uint base = spatialBase + localBody * 6u * nv;
            const float3 leftAngular{
                spatialJacobianScratch[base + 0u * nv + row],
                spatialJacobianScratch[base + 1u * nv + row],
                spatialJacobianScratch[base + 2u * nv + row],
            };
            const uint weighted = inertiaWeightedBase + localBody * 3u * nv + column;
            const float3 rightInertiaWeighted{
                spatialJacobianScratch[weighted + 0u * nv],
                spatialJacobianScratch[weighted + 1u * nv],
                spatialJacobianScratch[weighted + 2u * nv],
            };
            const float3 leftLinear{
                spatialJacobianScratch[base + 3u * nv + row],
                spatialJacobianScratch[base + 4u * nv + row],
                spatialJacobianScratch[base + 5u * nv + row],
            };
            const float3 rightLinear{
                spatialJacobianScratch[base + 3u * nv + column],
                spatialJacobianScratch[base + 4u * nv + column],
                spatialJacobianScratch[base + 5u * nv + column],
            };
            value += dot(leftAngular, rightInertiaWeighted) +
                body.massAndInverseMass.x * dot(leftLinear, rightLinear);
        }
        if (row == column) {
            device const MRDofPropertiesGPU& dof =
                dofs[articulation.vOffset + row];
            value += dof.drive.z;
            if ((dof.flags & MR_DOF_FLAG_DRIVE) == 0u) {
                // Backward-Euler passive damping: (M + hD)a = tau-b-Dv.
                value += dispatch.groundPointAndTimestep.w * dof.drive.y;
            }
        }
        if ((dispatch.flags & MR_NUMI_HUMAN_STAND_HAS_PASSIVE_JOINT_PROGRAM) != 0u) {
            value += mrNumiHumanPassiveEffectiveInertia(
                passiveJointProgram[index], dispatch.groundPointAndTimestep.w);
        }
        factor[index] = value;
        if (captureSourceDynamics && row == column) {
            // The upper triangle remains the exact source A0, but Cholesky
            // overwrites its diagonal. Preserve that missing diagonal here,
            // before the factorization barrier.
            sourceDynamicsWitness[sourceDynamicsBase + row] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;
    // The spatial Jacobian/inertia columns are dead after mass assembly and
    // the barrier above. Reuse two nv rows as per-step equality linearization
    // caches; unlike the persistent preload prefix, this scratch is rebuilt
    // before every stand step. q is unchanged throughout the coupled sweeps.
    device float* equalityDerivativeCache =
        spatialJacobianScratch + spatialBase;
    device float* equalityTargetVelocityCache =
        equalityDerivativeCache + nv;
    // The full-body spatial Jacobian is dead after response construction.
    // Reuse its arena for DOF-major equality responses consumed on every
    // coupled sweep. The derivative and target prefixes remain intact.
    device float* equalityResponseByDof =
        equalityTargetVelocityCache + nv;
    const bool cacheEqualityResponseByDof =
        bodyCount * MR_NUMI_HUMAN_STAND_SPATIAL_SCRATCH_ROWS >=
        2u + equalityCount;

    float minimumPivot = INFINITY;
    float maximumPivot = 0.0f;
    if (lane == 0u) {
        for (uint row = 0u; row < nv && status.code == MR_NUMI_HUMAN_STAND_SUCCESS; ++row) {
            float scale = 0.0f;
            for (uint column = 0u; column < nv; ++column) {
                scale = max(scale, abs(factor[row * nv + column]));
            }
            for (uint column = 0u; column <= row; ++column) {
                float value = factor[row * nv + column];
                for (uint inner = 0u; inner < column; ++inner) {
                    value -= factor[row * nv + inner] * factor[column * nv + inner];
                }
                if (row == column) {
                    if (!(value > max(
                            kPivotFloor,
                            scale * 8.0f * 1.1920928955078125e-7f
                        )) ||
                        !isfinite(value)) {
                        fail(status, MR_NUMI_HUMAN_STAND_FACTORIZATION_FAILED, row);
                        break;
                    }
                    factor[row * nv + row] = sqrt(value);
                    minimumPivot = min(minimumPivot, factor[row * nv + row]);
                    maximumPivot = max(maximumPivot, factor[row * nv + row]);
                } else {
                    factor[row * nv + column] =
                        value / factor[column * nv + column];
                }
            }
        }
    }
    // Every inverse-mass response has an independent RHS. Preserve each
    // scalar substitution's operation order while assigning columns to GPU
    // lanes, rather than running all contact/equality/limit solves on lane 0.
    threadgroup atomic_uint responseFailure;
    if (lane == 0u) {
        atomic_store_explicit(&responseFailure, MR_INVALID_INDEX, memory_order_relaxed);
        if ((dispatch.flags & MR_NUMI_HUMAN_STAND_ENABLE_CONTACT) != 0u) {
            for (uint contact = 0u; contact < dispatch.supportContactCount; ++contact) {
                device const auto& support = contacts[contact];
                if (support.bodyIndex < articulation.firstBody ||
                    support.bodyIndex >= articulation.firstBody + bodyCount ||
                    support.pointQueryIndex >= dispatch.pointWorldStride ||
                    support.reserved0 != 0u ||
                    !finite4(support.frictionSlopAndStabilization) ||
                    support.frictionSlopAndStabilization.x < 0.0f ||
                    support.frictionSlopAndStabilization.y < 0.0f ||
                    support.frictionSlopAndStabilization.z < 0.0f ||
                    support.frictionSlopAndStabilization.z > 1.0f ||
                    support.frictionSlopAndStabilization.w < 0.0f) {
                    fail(status, MR_NUMI_HUMAN_STAND_INVALID_DISPATCH, contact);
                    break;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;
    const uint contactColumns = 3u * dispatch.supportContactCount;
    const uint equalityColumnsEnd = contactColumns + equalityCount;
    const bool contactEnabled = (dispatch.flags & MR_NUMI_HUMAN_STAND_ENABLE_CONTACT) != 0u;
    if (contactEnabled || equalityCount != 0u) {
        const float3 responseNormal = dispatch.groundNormal.xyz;
        const float3 responseReference = abs(responseNormal.x) < 0.8f
            ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
        const float3 responseTangent0 = normalize(
            responseReference - dot(responseReference, responseNormal) * responseNormal);
        const float3 responseDirections[3] = {
            responseNormal, responseTangent0, cross(responseNormal, responseTangent0)};
        float independentWorkspace[MR_NUMI_HUMAN_STAND_MAX_DOFS];
        for (uint column = lane; column < equalityColumnsEnd + nv; column += threadCount) {
            device float* response = responseScratch + responseBase + column * nv;
            if (column < contactColumns) {
                if (!contactEnabled) continue;
                device const auto& support = contacts[column / 3u];
                const uint pointIndex = pointBase + support.pointQueryIndex;
                const float gap = dot(mrCompensatedPositionDifference(
                    pointWorld[pointIndex].position, pointPositionLow[pointIndex],
                    dispatch.groundPointAndTimestep, float4(0.0f)).xyz, responseNormal);
                if (gap > support.frictionSlopAndStabilization.y) continue;
                for (uint dof = 0u; dof < nv; ++dof)
                    response[dof] = pointJacobianAxis(pointJacobians, pointJacobianBase,
                        support.pointQueryIndex, nv, dof, responseDirections[column % 3u]);
            } else if (column < equalityColumnsEnd) {
                device const auto& equality = jointEqualities[column - contactColumns];
                float target = 0.0f, derivative = 0.0f, error = 0.0f;
                if (!evaluateJointEquality(equality, qState, qBase, nq, nv,
                        target, derivative, error)) {
                    atomic_fetch_min_explicit(&responseFailure, column, memory_order_relaxed);
                    continue;
                }
                for (uint dof = 0u; dof < nv; ++dof) response[dof] = 0.0f;
                response[equality.indices.y] = 1.0f;
                if (equality.indices.w != MR_INVALID_INDEX) response[equality.indices.w] = -derivative;
            } else {
                const uint dof = column - equalityColumnsEnd;
                if (!contactEnabled ||
                    (dofs[articulation.vOffset + dof].flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u)
                    continue;
                for (uint index = 0u; index < nv; ++index) response[index] = 0.0f;
                response[dof] = 1.0f;
            }
            if (!solveFactor(factor, independentWorkspace, response, nv))
                atomic_fetch_min_explicit(&responseFailure, column, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (lane == 0u) {
        const uint failedResponse = atomic_load_explicit(&responseFailure, memory_order_relaxed);
        if (failedResponse != MR_INVALID_INDEX) {
            if (failedResponse < contactColumns)
                fail(status, MR_NUMI_HUMAN_STAND_CONTACT_FAILED, failedResponse / 3u);
            else if (failedResponse < equalityColumnsEnd) {
                ++status.jointEqualityCounts.z;
                fail(status, MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED, failedResponse - contactColumns);
            } else
                fail(status, MR_NUMI_HUMAN_STAND_FACTORIZATION_FAILED, failedResponse - equalityColumnsEnd);

        }
    }

    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;
    const float timestep = dispatch.groundPointAndTimestep.w;
    float maximumEqualityPositionError = 0.0f;
    if (lane == 0u && equalityCount != 0u) {
        for (uint equalityIndex = 0u;
             equalityIndex < dispatch.jointEqualityCount;
             ++equalityIndex) {
            device const MRNumiHumanJointEqualityGPU& equality =
                jointEqualities[equalityIndex];
            float target = 0.0f;
            float derivative = 0.0f;
            float error = 0.0f;
            if (!evaluateJointEquality(
                    equality, qState, qBase, nq, nv,
                    target, derivative, error
                )) {
                ++status.jointEqualityCounts.z;
                fail(status, MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED,
                     equalityIndex);
                break;
            }
            equalityDerivativeCache[equalityIndex] = derivative;
            equalityTargetVelocityCache[equalityIndex] = clamp(
                -0.2f * error / timestep, -4.0f, 4.0f
            );
            maximumEqualityPositionError = max(
                maximumEqualityPositionError, abs(error)
            );

        }
        // Solve all bilateral rows together, not as scalar Gauss-Seidel
        // updates which can undo each other. Keep the actual nonsymmetric
        // FP32 contractions instead of silently adding diagonal compliance.
        for (uint row=0u; row<equalityCount &&
                status.code == MR_NUMI_HUMAN_STAND_SUCCESS; ++row) {
            device const MRNumiHumanJointEqualityGPU& equality=jointEqualities[row];
            const float derivative = equalityDerivativeCache[row];
            for (uint column=0u; column<equalityCount; ++column) {
                device const float* response=responseScratch+responseBase+
                    (3u*dispatch.supportContactCount+column)*nv;
                float value=response[equality.indices.y];
                if (equality.indices.w!=MR_INVALID_INDEX)
                    value=fma(-derivative,response[equality.indices.w],value);
                equalityFactor[row*equalityCount+column]=value;
            }
        }
        if (status.code == MR_NUMI_HUMAN_STAND_SUCCESS &&
            !mrNumiHumanBilateralFactor(equalityFactor,equalityScale,equalityPivots,equalityCount)) {
            fail(status,MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED,MR_INVALID_INDEX);

        }
        if (status.code == MR_NUMI_HUMAN_STAND_SUCCESS &&
            equalityCount <= kCachedEqualityCapacity) {
            for (uint index = 0u; index < equalityCount * equalityCount; ++index)
                equalityFactorStorage[index] = equalityFactor[index];
        }
    }

    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;
    // Condition source-limit responses independently, preserving the original
    // two refinement passes and each column's exact reduction order. The
    // subsequent coupled impulse sweeps retain their sequential ordering.
    if (contactEnabled) {
        float independentRaw[MR_NUMI_HUMAN_STAND_MAX_DOFS];
        float independentRhs[MR_NUMI_HUMAN_STAND_MAX_DOFS];
        for (uint dof = lane; dof < nv; dof += threadCount) {
            if ((dofs[articulation.vOffset + dof].flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u)
                continue;
            device float* response = responseScratch + responseBase +
                (equalityColumnsEnd + dof) * nv;
            bool validConditioning = true;
            // Eliminate ALL bilateral rows from this limit's response. A pair
            // correction can satisfy one equality while violating another. These
            // columns use the already factored E M_eff^-1 E^T; no new global
            // solve or artificial compliance is introduced.
            device float* equalityCorrection = limitEqualityCorrections +
                dof * equalityCount;
            for (uint ei = 0u; ei < equalityCount; ++ei)
                equalityCorrection[ei] = 0.0f;
            if (equalityCount != 0u) {
                const float rawDiagonal = response[dof];
                // Preserve the raw column in private per-lane scratch
                // for unresolved/rank-dependent directions without dropping rows.
                for (uint index = 0u; index < nv; ++index)
                    independentRaw[index] = response[index];
                for (uint refinement = 0u; refinement < 2u && validConditioning; ++refinement) {
                    for (uint ei = 0u; ei < equalityCount; ++ei) {
                        device const MRNumiHumanJointEqualityGPU& eq = jointEqualities[ei];
                        const float derivative = equalityDerivativeCache[ei];
                        float residual = response[eq.indices.y];
                        if (eq.indices.w != MR_INVALID_INDEX)
                            residual = fma(-derivative, response[eq.indices.w], residual);
                        independentRhs[ei] = residual;
                    }
                    const bool equalitySolved = equalityCount <= kCachedEqualityCapacity
                        ? mrNumiHumanBilateralSolve(equalityFactorStorage, equalityScale,
                            equalityPivots, independentRhs, equalityCount)
                        : mrNumiHumanBilateralSolve(equalityFactor, equalityScale,
                            equalityPivots, independentRhs, equalityCount);
                    if (!equalitySolved) {
                        atomic_fetch_min_explicit(&responseFailure, 2u * dof, memory_order_relaxed);
                        validConditioning = false;
                        break;
                    }
                    for (uint ei = 0u; ei < equalityCount; ++ei)
                        equalityCorrection[ei] -= independentRhs[ei];
                    for (uint index = 0u; index < nv; ++index) {
                        float correction = 0.0f;
                        for (uint ei = 0u; ei < equalityCount; ++ei) {
                            device const float* er = responseScratch + responseBase +
                                (3u * dispatch.supportContactCount + ei) * nv;
                            correction = fma(independentRhs[ei], er[index], correction);
                        }
                        response[index] -= correction;
                        if (!isfinite(response[index])) {
                            atomic_fetch_min_explicit(&responseFailure, 2u * dof + 1u, memory_order_relaxed);
                            validConditioning = false;
                            break;
                        }
                    }
                }
                // Cancellation in a direction already fixed by E must not be
                // inverted as a new independent limit. Retain the original scalar
                // coupled update for unresolved directions; do not manufacture
                // response with a diagonal floor, omit the row, or loosen gates.
                if (validConditioning && !(response[dof] > 1.0e-6f * rawDiagonal)) {
                    for (uint index = 0u; index < nv; ++index)
                        response[index] = independentRaw[index];
                    for (uint ei = 0u; ei < equalityCount; ++ei)
                        equalityCorrection[ei] = 0.0f;
                }
            }

        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    // Prepare once while all lanes are available. The scalar coupled solve
    // otherwise fetches one value from each nv-strided response column for
    // every DOF on every sweep.
    if (status.code == MR_NUMI_HUMAN_STAND_SUCCESS &&
        cacheEqualityResponseByDof) {
        for (uint dof = lane; dof < nv; dof += threadCount) {
            for (uint row = 0u; row < equalityCount; ++row) {
                equalityResponseByDof[dof * equalityCount + row] =
                    responseScratch[responseBase +
                        (3u * dispatch.supportContactCount + row) * nv + dof];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    // An optional split owner keeps prepared matrices and response columns on
    // the device for a following completion kernel in this command buffer.
    // Report response failures before ending this phase; no state is advanced.
    if ((dispatch.flags & MR_NUMI_HUMAN_STAND_PREPARE_ONLY) != 0u) {
        if (lane == 0u) {
            const uint failedCondition = atomic_load_explicit(
                &responseFailure, memory_order_relaxed);
            if (failedCondition != MR_INVALID_INDEX) {
                fail(status, (failedCondition & 1u) == 0u
                    ? MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED
                    : MR_NUMI_HUMAN_STAND_NONFINITE_RESULT,
                    failedCondition / 2u);
            }
        }
        return;
    }
#include "NumiHumanStandSolve.metalinc"

}

// The split completion entry consumes the prepare phase in the same
// authoritative command buffer. Contact and limit decisions retain their
// ordered owner; contact, equality, and limit response updates distribute
// disjoint DOFs. Contact velocity axes and limit reaction rows also use
// independent lanes while scalar impulse work keeps its FP32 order.
kernel void mr_numi_human_stand_finish(
    device const MRWorldGPU* worlds [[buffer(0)]],
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const MRDofPropertiesGPU* dofs [[buffer(2)]],
    device const MRBodyPropertiesGPU* bodies [[buffer(3)]],
    constant const MRNumiHumanStandDispatchGPU& dispatch [[buffer(4)]],
    device float* qState [[buffer(5)]],
    device float* vState [[buffer(6)]],
    device const MRArticulatedBodyPoseGPU* bodyPoses [[buffer(7)]],
    device const MRArticulatedPointWorldGPU* pointWorld [[buffer(8)]],
    device const float* pointJacobians [[buffer(9)]],
    device const float* generalizedForceWorkspace [[buffer(10)]],
    device const MRNumiHumanStandContactGPU* contacts [[buffer(11)]],
    device float* spatialJacobianScratch [[buffer(12)]],
    device float4* bodyMotionScratch [[buffer(13)]],
    device float* factorScratch [[buffer(14)]],
    device float* vectorScratch [[buffer(15)]],
    device float* responseScratch [[buffer(16)]],
    device MRNumiHumanStandStatusGPU* statuses [[buffer(17)]],
    device const MRNumiHumanTendonBindingGPU* tendonBindings [[buffer(18)]],
    device const MRNumiHumanTendonTransferResultGPU* tendonTransfers [[buffer(19)]],
    device const MRNumiHumanJointEqualityGPU* jointEqualities [[buffer(20)]],
    device MRCompensatedRootTranslationGPU* rootTranslations [[buffer(21)]],
    device const float4* bodyPositionLow [[buffer(22)]],
    device const float4* pointPositionLow [[buffer(23)]],
    device const float* passiveJointProgram [[buffer(24)]],
    device float* sourceDynamicsWitness [[buffer(25)]],
    uint environment [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint threadCount [[threads_per_threadgroup]]
) {
    if (environment >= dispatch.environmentCount) return;
    device MRNumiHumanStandStatusGPU& status = statuses[environment];
    device const MRArticulationGPU& articulation =
        articulations[dispatch.articulationIndex];
    const uint bodyCount = articulation.bodyCount;
    const uint nv = articulation.nv;
    const uint nq = articulation.nq;
    const uint qBase = environment * dispatch.qStride;
    const uint vBase = environment * dispatch.vStride;
    const uint pointBase = environment * dispatch.pointWorldStride;
    const uint pointJacobianBase = environment * dispatch.pointJacobianStride;
    const uint forceBase = environment * dispatch.generalizedForceStride +
        dispatch.generalizedForceOffset;
    const uint spatialBase = environment * bodyCount *
        MR_NUMI_HUMAN_STAND_SPATIAL_SCRATCH_ROWS * nv;
    const uint factorBase = environment * nv * nv;
    const uint vectorStride = nv + 3u * nv +
        12u * dispatch.supportContactCount + dispatch.jointEqualityCount +
        (((dispatch.flags & MR_NUMI_HUMAN_STAND_EXPORT_SOURCE_LIMIT_IMPULSES) != 0u)
            ? nv + nq + nv : 0u);
    const uint preloadBase = environment * vectorStride;
    const uint vectorBase = preloadBase + nv;
    const uint equalityCount = dispatch.jointEqualityCount;
    const uint responseColumns = (dispatch.supportContactCount * 3u + equalityCount + nv) * nv;
    const uint responseStride = responseColumns + equalityCount * (equalityCount + 3u) +
        nv * equalityCount;
    const uint responseBase = environment * responseStride;
    device float* equalityFactor = responseScratch + responseBase + responseColumns;
    device float* equalityScale = equalityFactor + equalityCount * equalityCount;
    device float* equalityPivots = equalityScale + equalityCount;
    // These vectors are repeatedly read and updated by the scalar constraint
    // solve. Keep them in the threadgroup instead of round-tripping every
    // scalar dependency through device memory. Retain the exported scratch
    // layout and publish its final values below.
    threadgroup float equalityRhsStorage[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup float equalityFactorStorage[
        kCachedEqualityCapacity * kCachedEqualityCapacity];
    threadgroup float candidateVStorage[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup float workspaceStorage[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    // Lane zero owns the ordered unilateral decisions. The other lanes apply
    // each accepted limit response to disjoint velocity DOFs.
    threadgroup uint cooperativeLimitCount;
    threadgroup float cooperativeLimitImpulse;
    threadgroup float cooperativeLimitEqualityImpulses[
        MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup float cooperativeLimitEqualityVelocities[
        MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup uint cooperativeContactActive[
        MR_NUMI_HUMAN_STAND_MAX_CONTACTS];
    threadgroup float cooperativeContactGap[
        MR_NUMI_HUMAN_STAND_MAX_CONTACTS];
    threadgroup float cooperativeContactVelocities[3];
    threadgroup float cooperativeContactApplied[3];
    threadgroup float* equalityRhs = equalityRhsStorage;
    // Equality multiplier corrections for each conditioned limit response.
    device float* limitEqualityCorrections = equalityPivots + 2u * equalityCount;
    device float* bias = vectorScratch + vectorBase;
    threadgroup float* candidateV = candidateVStorage;
    threadgroup float* workspace = workspaceStorage;
    device float* lambdas = bias + 3u * nv;
    device float* equalityLambdas =
        lambdas + 3u * dispatch.supportContactCount;
    device float* contactMatrices =
        equalityLambdas + dispatch.jointEqualityCount;
    device float* sourceLimitImpulseEvidence =
        contactMatrices + 9u * dispatch.supportContactCount;
    device float* preProjectionQEvidence = sourceLimitImpulseEvidence + nv;
    device float* preProjectionVEvidence = preProjectionQEvidence + nq;
    device float* factor = factorScratch + factorBase;
    const bool captureSourceDynamics =
        (dispatch.flags & MR_NUMI_HUMAN_STAND_PREDICT_VELOCITY_ONLY) != 0u;
    const uint sourceDynamicsBase = environment * 3u * nv;

    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;
    device float* equalityDerivativeCache = spatialJacobianScratch + spatialBase;
    device float* equalityTargetVelocityCache = equalityDerivativeCache + nv;
    device const float* equalityResponseByDof =
        equalityTargetVelocityCache + nv;
    const bool cacheEqualityResponseByDof =
        bodyCount * MR_NUMI_HUMAN_STAND_SPATIAL_SCRATCH_ROWS >=
        2u + equalityCount;
    const float timestep = dispatch.groundPointAndTimestep.w;
    float minimumPivot = INFINITY;
    float maximumPivot = 0.0f;
    float maximumEqualityPositionError = 0.0f;
    threadgroup atomic_uint responseFailure;
    if (lane == 0u) {
        atomic_store_explicit(&responseFailure, MR_INVALID_INDEX,
                              memory_order_relaxed);
        for (uint row = 0u; row < nv; ++row) {
            const float pivot = factor[row * nv + row];
            minimumPivot = min(minimumPivot, pivot);
            maximumPivot = max(maximumPivot, pivot);
        }
        for (uint row = 0u; row < equalityCount; ++row) {
            float target = 0.0f, derivative = 0.0f, error = 0.0f;
            if (!evaluateJointEquality(jointEqualities[row], qState, qBase,
                                       nq, nv, target, derivative, error)) {
                fail(status, MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED, row);
                break;
            }
            maximumEqualityPositionError = max(
                maximumEqualityPositionError, abs(error));
        }
        if (equalityCount <= kCachedEqualityCapacity) {
            for (uint index = 0u; index < equalityCount * equalityCount;
                 ++index) {
                equalityFactorStorage[index] = equalityFactor[index];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;
#define MR_NH_COOPERATIVE_FINISH 1
#include "NumiHumanStandSolve.metalinc"
#undef MR_NH_COOPERATIVE_FINISH
}

// Ordinary stand/tendon accepted-step owner. Derived poses/routes/factors are
// recomputed on the next step; the contact vector arena includes persistent
// warm starts and therefore belongs to the restored state.
kernel void mr_numi_human_stand_reconcile(
    constant uint4& shape [[buffer(0)]], // environments, attempted step, muscles, vectors
    constant uint4& strides [[buffer(1)]], // q, v
    device float* q [[buffer(2)]],
    device float* v [[buffer(3)]],
    device MRMujocoMuscleStateGPU* muscles [[buffer(4)]],
    device MRNumiHumanStandStatusGPU* statuses [[buffer(5)]],
    device float* vectors [[buffer(6)]],
    device const float* acceptedQ [[buffer(7)]],
    device const float* acceptedV [[buffer(8)]],
    device const MRMujocoMuscleStateGPU* acceptedMuscles [[buffer(9)]],
    device const MRNumiHumanStandStatusGPU* acceptedStatuses [[buffer(10)]],
    device const float* acceptedVectors [[buffer(11)]],
    device MRCompensatedRootTranslationGPU* rootTranslations [[buffer(12)]],
    device const MRCompensatedRootTranslationGPU* acceptedRootTranslations [[buffer(13)]],
    uint environment [[thread_position_in_grid]]
) {
    if (environment >= shape.x) return;
    const MRNumiHumanStandStatusGPU attempt = statuses[environment];
    if (attempt.code == MR_NUMI_HUMAN_STAND_SUCCESS &&
        attempt.environment == environment && attempt.completedSteps == shape.y + 1u) return;
    rootTranslations[environment] = acceptedRootTranslations[environment];
    for (uint i=0u; i<strides.x; ++i) q[environment*strides.x+i] = acceptedQ[environment*strides.x+i];
    for (uint i=0u; i<strides.y; ++i) v[environment*strides.y+i] = acceptedV[environment*strides.y+i];
    for (uint i=0u; i<shape.z; ++i) muscles[environment*shape.z+i] = acceptedMuscles[environment*shape.z+i];
    for (uint i=0u; i<shape.w; ++i) vectors[environment*shape.w+i] = acceptedVectors[environment*shape.w+i];
    MRNumiHumanStandStatusGPU restored = shape.y == 0u
        ? MRNumiHumanStandStatusGPU{} : acceptedStatuses[environment];
    restored.environment = environment;
    restored.code = attempt.code == MR_NUMI_HUMAN_STAND_SUCCESS
        ? MR_NUMI_HUMAN_STAND_INVALID_DISPATCH : attempt.code;
    restored.failingIndex = attempt.failingIndex;
    statuses[environment] = restored;
}
