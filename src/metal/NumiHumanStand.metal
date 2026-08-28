#include <metal_stdlib>

#include "metalrobo/numi_human_joint_equality_gpu.h"
#include "metalrobo/numi_human_stand_gpu.h"
#include "metalrobo/numi_human_tendon_gpu.h"

using namespace metal;

namespace {

constant float kPivotFloor = 1.0e-10f;
constant float kResponseRegularization = 1.0e-7f;

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

inline bool solveFactor(
    device const float* factor,
    device float* workspace,
    device float* output,
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

inline bool inverseSymmetric3x3(
    const float3 row0,
    const float3 row1,
    const float3 row2,
    thread float3& inverse0,
    thread float3& inverse1,
    thread float3& inverse2
) {
    const float determinant =
        row0.x * (row1.y * row2.z - row1.z * row2.y) -
        row0.y * (row1.x * row2.z - row1.z * row2.x) +
        row0.z * (row1.x * row2.y - row1.y * row2.x);
    if (!(determinant > 1.0e-18f) || !isfinite(determinant)) return false;
    const float inverseDeterminant = 1.0f / determinant;
    inverse0 = inverseDeterminant * float3(
        row1.y * row2.z - row1.z * row2.y,
        row0.z * row2.y - row0.y * row2.z,
        row0.y * row1.z - row0.z * row1.y
    );
    inverse1 = inverseDeterminant * float3(
        row1.z * row2.x - row1.x * row2.z,
        row0.x * row2.z - row0.z * row2.x,
        row0.z * row1.x - row0.x * row1.z
    );
    inverse2 = inverseDeterminant * float3(
        row1.x * row2.y - row1.y * row2.x,
        row0.y * row2.x - row0.x * row2.y,
        row0.x * row1.y - row0.y * row1.x
    );
    return all(isfinite(inverse0)) && all(isfinite(inverse1)) &&
        all(isfinite(inverse2));
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
    const uint spatialBase = environment * bodyCount * 6u * nv;
    const uint bodyMotionBase = environment * bodyCount * 2u;
    const uint factorBase = environment * nv * nv;
    const uint vectorStride = 3u * nv +
        12u * dispatch.supportContactCount + dispatch.jointEqualityCount;
    const uint vectorBase = environment * vectorStride;
    const uint responseBase = environment *
        (dispatch.supportContactCount * 3u + dispatch.jointEqualityCount) * nv;
    device float* bias = vectorScratch + vectorBase;
    device float* candidateV = bias + nv;
    device float* workspace = candidateV + nv;
    device float* lambdas = workspace + nv;
    device float* equalityLambdas =
        lambdas + 3u * dispatch.supportContactCount;
    device float* contactMatrices =
        equalityLambdas + dispatch.jointEqualityCount;
    device float* factor = factorScratch + factorBase;

    if (lane == 0u) {
        if (dispatch.stepIndex == 0u) {
            status = {};
            status.code = MR_NUMI_HUMAN_STAND_SUCCESS;
            status.environment = environment;
            status.failingIndex = MR_INVALID_INDEX;
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
            dispatch.stepCount > MR_NUMI_HUMAN_STAND_MAX_STEPS ||
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
                MR_NUMI_HUMAN_STAND_HAS_JOINT_EQUALITIES
            )) != 0u) {
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
    // consumers and deliberately are not added as a second joint torque.
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
        bias[row] = value;
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
            const float3 rightAngular{
                spatialJacobianScratch[base + 0u * nv + column],
                spatialJacobianScratch[base + 1u * nv + column],
                spatialJacobianScratch[base + 2u * nv + column],
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
            value += dot(
                leftAngular,
                worldInertiaMultiply(
                    body, bodyPoses[bodyPoseBase + localBody].orientation,
                    rightAngular
                )
            ) + body.massAndInverseMass.x * dot(leftLinear, rightLinear);
        }
        if (row == column) {
            value += dofs[articulation.vOffset + row].drive.z;
        }
        factor[index] = value;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;
    if (lane != 0u) return;

    float minimumPivot = INFINITY;
    float maximumPivot = 0.0f;
    for (uint row = 0u; row < nv; ++row) {
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
                    return;
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

    float3 assistanceForce{0.0f};
    float3 assistanceTorque{0.0f};
    if ((dispatch.flags & MR_NUMI_HUMAN_STAND_ENABLE_ROOT_ASSISTANCE) != 0u) {
        assistanceForce = dispatch.assistanceGains.x *
            (dispatch.targetRootPosition.xyz - float3(
                qState[qBase + 0u], qState[qBase + 1u], qState[qBase + 2u]
            )) - dispatch.assistanceGains.y * float3(
                vState[vBase + 0u], vState[vBase + 1u], vState[vBase + 2u]
            );
        float4 currentOrientation;
        float4 targetOrientation;
        if (!normalizedQuaternion(float4(
                qState[qBase + 3u], qState[qBase + 4u],
                qState[qBase + 5u], qState[qBase + 6u]
            ), currentOrientation) ||
            !normalizedQuaternion(dispatch.targetRootOrientation, targetOrientation)) {
            fail(status, MR_NUMI_HUMAN_STAND_NONFINITE_INPUT, 3u);
            return;
        }
        float4 error = quaternionMultiply(
            targetOrientation, quaternionConjugate(currentOrientation)
        );
        if (error.w < 0.0f) error = -error;
        assistanceTorque = dispatch.assistanceGains.z * 2.0f * error.xyz -
            dispatch.assistanceGains.w * float3(
                vState[vBase + 3u], vState[vBase + 4u], vState[vBase + 5u]
            );
    }
    for (uint dof = 0u; dof < nv; ++dof) {
        float effort = generalizedForceWorkspace[forceBase + dof];
        if (dof < 3u) effort += assistanceForce[dof];
        else if (dof < 6u) effort += assistanceTorque[dof - 3u];
        candidateV[dof] = effort - bias[dof];
    }
    if (!solveFactor(factor, workspace, candidateV, nv)) {
        fail(status, MR_NUMI_HUMAN_STAND_FACTORIZATION_FAILED, MR_INVALID_INDEX);
        return;
    }
    const float timestep = dispatch.groundPointAndTimestep.w;
    float maximumAcceleration = 0.0f;
    for (uint dof = 0u; dof < nv; ++dof) {
        maximumAcceleration = max(maximumAcceleration, abs(candidateV[dof]));
        candidateV[dof] = vState[vBase + dof] + timestep * candidateV[dof];
    }

    uint activeContacts = 0u;
    float minimumGap = INFINITY;
    float maximumPenetration = 0.0f;
    const float3 normal = dispatch.groundNormal.xyz;
    const float3 reference = abs(normal.x) < 0.8f
        ? float3(1.0f, 0.0f, 0.0f)
        : float3(0.0f, 1.0f, 0.0f);
    const float3 tangent0 = normalize(reference - dot(reference, normal) * normal);
    const float3 tangent1 = cross(normal, tangent0);
    const float3 directions[3] = {normal, tangent0, tangent1};
    float maximumEqualityPositionError = 0.0f;
    float maximumEqualityVelocityError = 0.0f;
    float maximumEqualityImpulse = 0.0f;
    float totalEqualityImpulse = 0.0f;

    if ((dispatch.flags & MR_NUMI_HUMAN_STAND_ENABLE_CONTACT) != 0u) {
        for (uint contact = 0u; contact < dispatch.supportContactCount; ++contact) {
            device const MRNumiHumanStandContactGPU& support = contacts[contact];
            if (support.bodyIndex < articulation.firstBody ||
                support.bodyIndex >= articulation.firstBody + bodyCount ||
                support.pointQueryIndex >= dispatch.pointWorldStride ||
                support.reserved0 != 0u ||
                !finite4(support.frictionSlopAndStabilization) ||
                support.frictionSlopAndStabilization.x < 0.0f ||
                support.frictionSlopAndStabilization.y < 0.0f ||
                support.frictionSlopAndStabilization.z < 0.0f ||
                support.frictionSlopAndStabilization.z > 1.0f ||
                support.frictionSlopAndStabilization.w != 0.0f) {
                fail(status, MR_NUMI_HUMAN_STAND_INVALID_DISPATCH, contact);
                return;
            }
            const float3 point = pointWorld[
                pointBase + support.pointQueryIndex
            ].position.xyz;
            const float gap = dot(
                point - dispatch.groundPointAndTimestep.xyz, normal
            );
            minimumGap = min(minimumGap, gap);
            maximumPenetration = max(maximumPenetration, max(-gap, 0.0f));
            if (gap > support.frictionSlopAndStabilization.y) {
                lambdas[3u * contact + 0u] = 0.0f;
                lambdas[3u * contact + 1u] = 0.0f;
                lambdas[3u * contact + 2u] = 0.0f;
                continue;
            }
            ++activeContacts;
            for (uint axis = 0u; axis < 3u; ++axis) {
                device float* response = responseScratch + responseBase +
                    (3u * contact + axis) * nv;
                for (uint dof = 0u; dof < nv; ++dof) {
                    response[dof] = pointJacobianAxis(
                        pointJacobians, pointJacobianBase,
                        support.pointQueryIndex, nv, dof, directions[axis]
                    );
                }
                if (!solveFactor(factor, workspace, response, nv)) {
                    fail(status, MR_NUMI_HUMAN_STAND_CONTACT_FAILED, contact);
                    return;
                }
            }
            device float* matrix = contactMatrices + 9u * contact;
            for (uint row = 0u; row < 3u; ++row) {
                for (uint column = 0u; column < 3u; ++column) {
                    float value = row == column ? kResponseRegularization : 0.0f;
                    device const float* response = responseScratch + responseBase +
                        (3u * contact + column) * nv;
                    for (uint dof = 0u; dof < nv; ++dof) {
                        value += pointJacobianAxis(
                            pointJacobians, pointJacobianBase,
                            support.pointQueryIndex, nv, dof, directions[row]
                        ) * response[dof];
                    }
                    matrix[3u * row + column] = value;
                }
            }
        }

        for (uint iteration = 0u;
             iteration < dispatch.contactIterationCount;
             ++iteration) {
            for (uint contact = 0u; contact < dispatch.supportContactCount; ++contact) {
                device const MRNumiHumanStandContactGPU& support = contacts[contact];
                const float3 point = pointWorld[
                    pointBase + support.pointQueryIndex
                ].position.xyz;
                const float gap = dot(
                    point - dispatch.groundPointAndTimestep.xyz, normal
                );
                if (gap > support.frictionSlopAndStabilization.y) continue;
                float3 velocity{0.0f};
                for (uint axis = 0u; axis < 3u; ++axis) {
                    for (uint dof = 0u; dof < nv; ++dof) {
                        velocity[axis] += pointJacobianAxis(
                            pointJacobians, pointJacobianBase,
                            support.pointQueryIndex, nv, dof, directions[axis]
                        ) * candidateV[dof];
                    }
                }
                const float targetNormalVelocity = max(
                    0.0f,
                    -support.frictionSlopAndStabilization.z * min(gap, 0.0f) /
                        timestep
                );
                device float* matrix = contactMatrices + 9u * contact;
                float3 inverse0, inverse1, inverse2;
                if (!inverseSymmetric3x3(
                        float3(matrix[0], matrix[1], matrix[2]),
                        float3(matrix[3], matrix[4], matrix[5]),
                        float3(matrix[6], matrix[7], matrix[8]),
                        inverse0, inverse1, inverse2
                    )) {
                    fail(status, MR_NUMI_HUMAN_STAND_CONTACT_FAILED, contact);
                    return;
                }
                const float3 residual{
                    targetNormalVelocity - velocity.x,
                    -velocity.y,
                    -velocity.z,
                };
                const float3 delta{
                    dot(inverse0, residual),
                    dot(inverse1, residual),
                    dot(inverse2, residual),
                };
                const float3 oldLambda{
                    lambdas[3u * contact + 0u],
                    lambdas[3u * contact + 1u],
                    lambdas[3u * contact + 2u],
                };
                float3 newLambda = oldLambda + delta;
                newLambda.x = max(newLambda.x, 0.0f);
                const float tangentLimit =
                    support.frictionSlopAndStabilization.x * newLambda.x;
                const float tangentNorm = length(newLambda.yz);
                if (tangentNorm > tangentLimit && tangentNorm > 0.0f) {
                    newLambda.yz *= tangentLimit / tangentNorm;
                }
                const float3 applied = newLambda - oldLambda;
                lambdas[3u * contact + 0u] = newLambda.x;
                lambdas[3u * contact + 1u] = newLambda.y;
                lambdas[3u * contact + 2u] = newLambda.z;
                for (uint axis = 0u; axis < 3u; ++axis) {
                    device const float* response = responseScratch + responseBase +
                        (3u * contact + axis) * nv;
                    for (uint dof = 0u; dof < nv; ++dof) {
                        candidateV[dof] += applied[axis] * response[dof];
                    }
                }
            }
        }
    }

    // Equality rows use the same factored mass matrix as contact. Solving
    // them after the unilateral support pass makes the authored anatomical
    // manifold exact without injecting a hidden joint motor.
    if ((dispatch.flags & MR_NUMI_HUMAN_STAND_HAS_JOINT_EQUALITIES) != 0u) {
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
                return;
            }
            maximumEqualityPositionError = max(
                maximumEqualityPositionError, abs(error)
            );
            device float* response = responseScratch + responseBase +
                (3u * dispatch.supportContactCount + equalityIndex) * nv;
            for (uint dof = 0u; dof < nv; ++dof) response[dof] = 0.0f;
            response[equality.indices.y] = 1.0f;
            if (equality.indices.w != MR_INVALID_INDEX) {
                response[equality.indices.w] = -derivative;
            }
            if (!solveFactor(factor, workspace, response, nv)) {
                ++status.jointEqualityCounts.z;
                fail(status, MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED,
                     equalityIndex);
                return;
            }
        }

        for (uint iteration = 0u;
             iteration < dispatch.contactIterationCount;
             ++iteration) {
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
                    return;
                }
                device const float* response = responseScratch + responseBase +
                    (3u * dispatch.supportContactCount + equalityIndex) * nv;
                float effectiveMass = response[equality.indices.y];
                float constraintVelocity = candidateV[equality.indices.y];
                if (equality.indices.w != MR_INVALID_INDEX) {
                    effectiveMass -= derivative * response[equality.indices.w];
                    constraintVelocity -=
                        derivative * candidateV[equality.indices.w];
                }
                effectiveMass += kResponseRegularization;
                if (!(effectiveMass > kResponseRegularization) ||
                    !isfinite(effectiveMass)) {
                    ++status.jointEqualityCounts.z;
                    fail(status, MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED,
                         equalityIndex);
                    return;
                }
                const float targetVelocity = clamp(
                    -0.2f * error / timestep, -4.0f, 4.0f
                );
                const float delta =
                    (targetVelocity - constraintVelocity) / effectiveMass;
                equalityLambdas[equalityIndex] += delta;
                for (uint dof = 0u; dof < nv; ++dof) {
                    candidateV[dof] += delta * response[dof];
                }
            }
        }
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
                return;
            }
            float velocityError = candidateV[equality.indices.y];
            if (equality.indices.w != MR_INVALID_INDEX) {
                velocityError -= derivative * candidateV[equality.indices.w];
            }
            const float targetVelocity = clamp(
                -0.2f * error / timestep, -4.0f, 4.0f
            );
            maximumEqualityVelocityError = max(
                maximumEqualityVelocityError,
                abs(velocityError - targetVelocity)
            );
            const float absoluteImpulse = abs(equalityLambdas[equalityIndex]);
            maximumEqualityImpulse = max(
                maximumEqualityImpulse, absoluteImpulse
            );
            totalEqualityImpulse += absoluteImpulse;
        }
    }

    for (uint dof = 0u; dof < nv; ++dof) {
        if (!isfinite(candidateV[dof])) {
            fail(status, MR_NUMI_HUMAN_STAND_NONFINITE_RESULT, dof);
            return;
        }
        vState[vBase + dof] = candidateV[dof];
    }
    qState[qBase + 0u] += timestep * candidateV[0u];
    qState[qBase + 1u] += timestep * candidateV[1u];
    qState[qBase + 2u] += timestep * candidateV[2u];
    float4 orientation;
    if (!normalizedQuaternion(float4(
            qState[qBase + 3u], qState[qBase + 4u],
            qState[qBase + 5u], qState[qBase + 6u]
        ), orientation)) {
        fail(status, MR_NUMI_HUMAN_STAND_NONFINITE_RESULT, 3u);
        return;
    }
    const float4 increment = quaternionFromRotationVector(
        timestep * float3(candidateV[3u], candidateV[4u], candidateV[5u])
    );
    const float4 nextOrientation = normalize(quaternionMultiply(increment, orientation));
    qState[qBase + 3u] = nextOrientation.x;
    qState[qBase + 4u] = nextOrientation.y;
    qState[qBase + 5u] = nextOrientation.z;
    qState[qBase + 6u] = nextOrientation.w;
    for (uint dof = 6u; dof < nv; ++dof) {
        device const MRDofPropertiesGPU& properties =
            dofs[articulation.vOffset + dof];
        if (properties.qIndex == MR_INVALID_INDEX ||
            properties.qIndex < articulation.qOffset ||
            properties.qIndex >= articulation.qOffset + nq) {
            fail(status, MR_NUMI_HUMAN_STAND_INVALID_MODEL, dof);
            return;
        }
        qState[qBase + properties.qIndex - articulation.qOffset] +=
            timestep * candidateV[dof];
    }
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
            return;
        }
        maximumEqualityPositionError = max(
            maximumEqualityPositionError, abs(error)
        );
        qState[qBase + equality.indices.x] = target;
        const float dependentVelocity =
            equality.indices.w == MR_INVALID_INDEX
                ? 0.0f
                : derivative * vState[vBase + equality.indices.w];
        if (!isfinite(target) || !isfinite(dependentVelocity)) {
            ++status.jointEqualityCounts.z;
            fail(status, MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED,
                 equalityIndex);
            return;
        }
        vState[vBase + equality.indices.y] = dependentVelocity;
        candidateV[equality.indices.y] = dependentVelocity;
    }

    float totalNormalImpulse = 0.0f;
    for (uint contact = 0u; contact < dispatch.supportContactCount; ++contact) {
        totalNormalImpulse += lambdas[3u * contact + 0u];
    }
    status.completedSteps = dispatch.stepIndex + 1u;
    status.activeContactCount = activeContacts;
    status.maximumActiveContactCount = max(
        status.maximumActiveContactCount, activeContacts
    );
    status.contactIterations = dispatch.contactIterationCount;
    status.flags = dispatch.flags;
    status.contactAndAcceleration.x = min(
        status.contactAndAcceleration.x, minimumGap
    );
    status.contactAndAcceleration.y = max(
        status.contactAndAcceleration.y, maximumPenetration
    );
    status.contactAndAcceleration.z = totalNormalImpulse;
    status.contactAndAcceleration.w = max(
        status.contactAndAcceleration.w, maximumAcceleration
    );
    status.factorAndAssistance.x = min(
        status.factorAndAssistance.x, minimumPivot
    );
    status.factorAndAssistance.y = max(
        status.factorAndAssistance.y, maximumPivot
    );
    status.factorAndAssistance.z = max(
        status.factorAndAssistance.z, length(assistanceForce)
    );
    status.factorAndAssistance.w = max(
        status.factorAndAssistance.w, length(assistanceTorque)
    );
    status.jointEqualityCounts.x = dispatch.jointEqualityCount;
    status.jointEqualityCounts.y = max(
        status.jointEqualityCounts.y, dispatch.jointEqualityCount
    );
    status.jointEqualityDiagnostics.x = max(
        status.jointEqualityDiagnostics.x, maximumEqualityPositionError
    );
    status.jointEqualityDiagnostics.y = max(
        status.jointEqualityDiagnostics.y, maximumEqualityVelocityError
    );
    status.jointEqualityDiagnostics.z = max(
        status.jointEqualityDiagnostics.z, maximumEqualityImpulse
    );
    status.jointEqualityDiagnostics.w += totalEqualityImpulse;
}
