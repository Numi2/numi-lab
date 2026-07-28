#include <metal_stdlib>

#include "metalrobo/engine_types.h"

using namespace metal;

namespace {

constant float kQuaternionTolerance = 2.0e-5f;
constant float kQuaternionMinimum = 1.0e-12f;
constant float kFloatEpsilon = 1.1920928955078125e-7f;
constant float kAbsolutePivotFloor = 1.0e-12f;
constant uint kMaxBodies = MR_ARTICULATED_ABA_MAX_BODIES;
constant uint kMaxDofs = MR_ARTICULATED_ABA_MAX_DOFS;
constant uint kMaxQ = MR_ARTICULATED_ABA_MAX_Q;
constant uint kMaxRhs = MR_ARTICULATED_INVERSE_MASS_MAX_RHS;

inline bool finite3(const float3 value) {
    return all(isfinite(value));
}

inline bool finite4(const float4 value) {
    return all(isfinite(value));
}

inline float4 quaternionConjugate(const float4 value) {
    return float4(-value.xyz, value.w);
}

inline float4 quaternionMultiply(
    const float4 left,
    const float4 right
) {
    return float4(
        left.w * right.x + left.x * right.w +
            left.y * right.z - left.z * right.y,
        left.w * right.y - left.x * right.z +
            left.y * right.w + left.z * right.x,
        left.w * right.z + left.x * right.y -
            left.y * right.x + left.z * right.w,
        left.w * right.w - dot(left.xyz, right.xyz)
    );
}

inline float3 quaternionRotate(
    const float4 quaternion,
    const float3 value
) {
    const float3 doubledCross =
        2.0f * cross(quaternion.xyz, value);
    return value +
        quaternion.w * doubledCross +
        cross(quaternion.xyz, doubledCross);
}

inline bool normalizedQuaternion(
    const float4 input,
    thread float4& output,
    const bool requireUnit
) {
    if (!finite4(input)) {
        return false;
    }
    const float normSquared = dot(input, input);
    if (!(normSquared > kQuaternionMinimum) ||
        !isfinite(normSquared)) {
        return false;
    }
    const float norm = sqrt(normSquared);
    if (requireUnit &&
        abs(norm - 1.0f) > kQuaternionTolerance) {
        return false;
    }
    output = input / norm;
    return finite4(output);
}

inline float4 axisAngleQuaternion(
    const float3 normalizedAxis,
    const float angle
) {
    const float halfAngle = 0.5f * angle;
    return float4(
        normalizedAxis * sin(halfAngle),
        cos(halfAngle)
    );
}

inline float spatialComponent(
    const float3 angular,
    const float3 linear,
    const uint component
) {
    return component < 3u
        ? angular[component]
        : linear[component - 3u];
}

inline void setSpatialComponent(
    thread float3& angular,
    thread float3& linear,
    const uint component,
    const float value
) {
    if (component < 3u) {
        angular[component] = value;
    } else {
        linear[component - 3u] = value;
    }
}

inline float motionTransformElement(
    const float3 parentToChild,
    const uint row,
    const uint column
) {
    if (row == column) {
        return 1.0f;
    }
    if (row < 3u || column >= 3u) {
        return 0.0f;
    }
    const uint linearRow = row - 3u;
    if (linearRow == 0u && column == 1u) {
        return parentToChild.z;
    }
    if (linearRow == 0u && column == 2u) {
        return -parentToChild.y;
    }
    if (linearRow == 1u && column == 0u) {
        return -parentToChild.z;
    }
    if (linearRow == 1u && column == 2u) {
        return parentToChild.x;
    }
    if (linearRow == 2u && column == 0u) {
        return parentToChild.y;
    }
    if (linearRow == 2u && column == 1u) {
        return -parentToChild.x;
    }
    return 0.0f;
}

inline float inertiaBodyElement(
    device const MRBodyPropertiesGPU& body,
    const uint row,
    const uint column
) {
    if (row == 0u) {
        return body.inertiaRow0[column];
    }
    if (row == 1u) {
        return body.inertiaRow1[column];
    }
    return body.inertiaRow2[column];
}

inline float rotationElement(
    const float3 column0,
    const float3 column1,
    const float3 column2,
    const uint row,
    const uint column
) {
    return column == 0u
        ? column0[row]
        : (column == 1u ? column1[row] : column2[row]);
}

inline float worldInertiaElement(
    device const MRBodyPropertiesGPU& body,
    const float3 rotationColumn0,
    const float3 rotationColumn1,
    const float3 rotationColumn2,
    const uint row,
    const uint column
) {
    float result = 0.0f;
    for (uint bodyRow = 0u; bodyRow < 3u; ++bodyRow) {
        const float left = rotationElement(
            rotationColumn0,
            rotationColumn1,
            rotationColumn2,
            row,
            bodyRow
        );
        for (uint bodyColumn = 0u;
             bodyColumn < 3u;
             ++bodyColumn) {
            result +=
                left *
                inertiaBodyElement(body, bodyRow, bodyColumn) *
                rotationElement(
                    rotationColumn0,
                    rotationColumn1,
                    rotationColumn2,
                    column,
                    bodyColumn
                );
        }
    }
    return result;
}

inline void setFailure(
    thread MRInverseMassStatusGPU& status,
    const uint code,
    const uint failingIndex
) {
    status.code = code;
    status.failingIndex = failingIndex;
}

inline bool validVectorStrides(
    const uint vectorStride,
    const uint environmentStride,
    const uint vectorCount,
    const uint vectorWidth
) {
    if (vectorStride < vectorWidth) {
        return false;
    }
    const ulong required =
        ulong(vectorCount - 1u) * ulong(vectorStride) +
        ulong(vectorWidth);
    return required <= ulong(environmentStride);
}

} // namespace

// Applies M(q)^-1 to one to three generalized vectors. One 32-lane
// threadgroup owns one environment; lane zero performs deterministic tree
// sweeps while environments execute in parallel. The articulated-inertia
// factorization is shared by every RHS and output is published only after all
// RHS vectors pass finite checks.
kernel void mr_articulated_inverse_mass(
    device const MRWorldGPU* worlds [[buffer(0)]],
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const MRJointDescriptorGPU* joints [[buffer(2)]],
    device const MRDofPropertiesGPU* dofs [[buffer(3)]],
    device const MRBodyPropertiesGPU* bodies [[buffer(4)]],
    device const MRInverseMassDispatchGPU& dispatch [[buffer(5)]],
    device const float* q [[buffer(6)]],
    device const float* rightHandSides [[buffer(7)]],
    device float* output [[buffer(8)]],
    device MRInverseMassStatusGPU* statuses [[buffer(9)]],
    uint environment [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (lane != 0u || environment >= dispatch.environmentCount) {
        return;
    }

    threadgroup float3 bodyPosition[kMaxBodies];
    threadgroup float4 bodyRotation[kMaxBodies];
    threadgroup float3 motionAngular[kMaxBodies];
    threadgroup float3 motionLinear[kMaxBodies];
    threadgroup float3 parentToBody[kMaxBodies];
    threadgroup float articulatedInertia[kMaxBodies * 36u];
    threadgroup float projectedInertia[kMaxBodies * 6u];
    threadgroup float jointDenominator[kMaxBodies];
    threadgroup float articulatedBias[kMaxBodies * 6u];
    threadgroup float jointResidual[kMaxBodies];
    threadgroup float3 accelerationAngular[kMaxBodies];
    threadgroup float3 accelerationLinear[kMaxBodies];
    threadgroup uint inboundJoint[kMaxBodies];
    threadgroup uint parentLocal[kMaxBodies];
    threadgroup uint traversal[kMaxBodies];
    threadgroup uchar known[kMaxBodies];
    threadgroup float candidateOutput[kMaxRhs * kMaxDofs];
    threadgroup float rootFactor[36u];
    threadgroup float rootIntermediate[6u];

    MRInverseMassStatusGPU status = {};
    status.code = MR_INVERSE_MASS_SUCCESS;
    status.environment = environment;
    status.articulationIndex = dispatch.articulationIndex;
    status.failingIndex = MR_INVALID_INDEX;
    status.rhsCount = dispatch.rhsCount;

    device const MRWorldGPU& world = worlds[0];
    if (world.abiVersion != MR_ENGINE_ABI_VERSION ||
        dispatch.articulationIndex >= world.articulationCount ||
        dispatch.environmentCount == 0u ||
        dispatch.rhsCount == 0u ||
        dispatch.rhsCount > kMaxRhs ||
        dispatch.reserved0 != 0u ||
        dispatch.reserved1 != 0u ||
        dispatch.reserved2 != 0u ||
        dispatch.reserved3 != 0u) {
        setFailure(
            status,
            MR_INVERSE_MASS_INVALID_DISPATCH,
            MR_INVALID_INDEX
        );
        statuses[environment] = status;
        return;
    }

    device const MRArticulationGPU& articulation =
        articulations[dispatch.articulationIndex];
    status.bodyCount = articulation.bodyCount;
    status.nq = articulation.nq;
    status.nv = articulation.nv;
    if ((articulation.rootType != MR_ROOT_FIXED &&
         articulation.rootType != MR_ROOT_FLOATING) ||
        articulation.bodyCount == 0u ||
        articulation.bodyCount > kMaxBodies ||
        articulation.nv == 0u ||
        articulation.nv > kMaxDofs ||
        articulation.nq > kMaxQ ||
        articulation.firstBody > world.bodyCount ||
        articulation.bodyCount >
            world.bodyCount - articulation.firstBody ||
        articulation.firstJoint > world.jointCount ||
        articulation.jointCount >
            world.jointCount - articulation.firstJoint ||
        articulation.qOffset > world.nq ||
        articulation.nq > world.nq - articulation.qOffset ||
        articulation.vOffset > world.nv ||
        articulation.nv > world.nv - articulation.vOffset ||
        articulation.rootBody < articulation.firstBody ||
        articulation.rootBody >=
            articulation.firstBody + articulation.bodyCount ||
        articulation.jointCount + 1u != articulation.bodyCount ||
        dispatch.qStride < articulation.nq ||
        !validVectorStrides(
            dispatch.rhsVectorStride,
            dispatch.rhsEnvironmentStride,
            dispatch.rhsCount,
            articulation.nv
        ) ||
        !validVectorStrides(
            dispatch.outputVectorStride,
            dispatch.outputEnvironmentStride,
            dispatch.rhsCount,
            articulation.nv
        )) {
        setFailure(
            status,
            articulation.bodyCount > kMaxBodies ||
                articulation.nv > kMaxDofs ||
                articulation.nq > kMaxQ
                ? MR_INVERSE_MASS_UNSUPPORTED_TOPOLOGY
                : MR_INVERSE_MASS_INVALID_MODEL,
            MR_INVALID_INDEX
        );
        statuses[environment] = status;
        return;
    }

    const uint rootLocal =
        articulation.rootBody - articulation.firstBody;
    for (uint localBody = 0u;
         localBody < articulation.bodyCount;
         ++localBody) {
        inboundJoint[localBody] = MR_INVALID_INDEX;
        parentLocal[localBody] = MR_INVALID_INDEX;
        known[localBody] = 0u;
        jointDenominator[localBody] = 0.0f;
    }

    uint expectedNq =
        articulation.rootType == MR_ROOT_FLOATING ? 7u : 0u;
    uint expectedNv =
        articulation.rootType == MR_ROOT_FLOATING ? 6u : 0u;
    for (uint localJoint = 0u;
         localJoint < articulation.jointCount;
         ++localJoint) {
        const uint globalJoint =
            articulation.firstJoint + localJoint;
        device const MRJointDescriptorGPU& joint =
            joints[globalJoint];
        if (joint.parentBody < articulation.firstBody ||
            joint.parentBody >=
                articulation.firstBody + articulation.bodyCount ||
            joint.childBody < articulation.firstBody ||
            joint.childBody >=
                articulation.firstBody + articulation.bodyCount ||
            joint.childBody == articulation.rootBody ||
            joint.parentBody == joint.childBody ||
            joint.flags != 0u ||
            !finite4(joint.parentAnchor) ||
            !finite4(joint.childAnchor) ||
            !finite4(joint.parentRotation) ||
            !finite4(joint.childRotation)) {
            setFailure(
                status,
                MR_INVERSE_MASS_INVALID_MODEL,
                globalJoint
            );
            statuses[environment] = status;
            return;
        }
        const bool scalarJoint =
            joint.jointType == MR_JOINT_REVOLUTE ||
            joint.jointType == MR_JOINT_CONTINUOUS;
        const bool fixedJoint =
            joint.jointType == MR_JOINT_FIXED;
        if ((!scalarJoint && !fixedJoint) ||
            joint.nq != (scalarJoint ? 1u : 0u) ||
            joint.nv != (scalarJoint ? 1u : 0u) ||
            joint.qOffset != articulation.qOffset + expectedNq ||
            joint.vOffset != articulation.vOffset + expectedNv) {
            setFailure(
                status,
                !scalarJoint && !fixedJoint
                    ? MR_INVERSE_MASS_UNSUPPORTED_TOPOLOGY
                    : MR_INVERSE_MASS_INVALID_MODEL,
                globalJoint
            );
            statuses[environment] = status;
            return;
        }
        if (scalarJoint) {
            const float axisNormSquared =
                dot(joint.axis0.xyz, joint.axis0.xyz);
            if (!finite4(joint.axis0) ||
                !(axisNormSquared > kQuaternionMinimum)) {
                setFailure(
                    status,
                    MR_INVERSE_MASS_INVALID_MODEL,
                    globalJoint
                );
                statuses[environment] = status;
                return;
            }
            device const MRDofPropertiesGPU& dof =
                dofs[joint.vOffset];
            if (dof.articulationIndex !=
                    dispatch.articulationIndex ||
                dof.jointIndex != globalJoint ||
                dof.qIndex != joint.qOffset ||
                dof.vIndex != joint.vOffset ||
                dof.localDof != 0u) {
                setFailure(
                    status,
                    MR_INVERSE_MASS_INVALID_MODEL,
                    joint.vOffset
                );
                statuses[environment] = status;
                return;
            }
        }
        const uint localChild =
            joint.childBody - articulation.firstBody;
        if (inboundJoint[localChild] != MR_INVALID_INDEX) {
            setFailure(
                status,
                MR_INVERSE_MASS_UNSUPPORTED_TOPOLOGY,
                joint.childBody
            );
            statuses[environment] = status;
            return;
        }
        inboundJoint[localChild] = globalJoint;
        parentLocal[localChild] =
            joint.parentBody - articulation.firstBody;
        expectedNq += scalarJoint ? 1u : 0u;
        expectedNv += scalarJoint ? 1u : 0u;
    }
    if (expectedNq != articulation.nq ||
        expectedNv != articulation.nv ||
        inboundJoint[rootLocal] != MR_INVALID_INDEX) {
        setFailure(
            status,
            MR_INVERSE_MASS_INVALID_MODEL,
            MR_INVALID_INDEX
        );
        statuses[environment] = status;
        return;
    }

    for (uint localV = 0u;
         localV < articulation.nv;
         ++localV) {
        const uint globalV = articulation.vOffset + localV;
        device const MRDofPropertiesGPU& dof = dofs[globalV];
        if (dof.articulationIndex != dispatch.articulationIndex ||
            dof.vIndex != globalV ||
            dof.reserved0 != 0u ||
            dof.reserved1 != 0u ||
            !finite4(dof.drive) ||
            dof.drive.z < 0.0f) {
            setFailure(
                status,
                MR_INVERSE_MASS_INVALID_MODEL,
                globalV
            );
            statuses[environment] = status;
            return;
        }
        if (articulation.rootType == MR_ROOT_FLOATING &&
            localV < 6u) {
            const uint expectedQ =
                localV < 3u
                    ? articulation.qOffset + localV
                    : MR_INVALID_INDEX;
            if (dof.jointIndex != MR_INVALID_INDEX ||
                dof.qIndex != expectedQ ||
                dof.localDof != localV) {
                setFailure(
                    status,
                    MR_INVERSE_MASS_INVALID_MODEL,
                    globalV
                );
                statuses[environment] = status;
                return;
            }
        }
    }

    known[rootLocal] = 1u;
    traversal[0] = rootLocal;
    uint discovered = 1u;
    for (uint pass = 0u;
         pass < articulation.bodyCount &&
             discovered < articulation.bodyCount;
         ++pass) {
        bool progressed = false;
        for (uint localJoint = 0u;
             localJoint < articulation.jointCount;
             ++localJoint) {
            const uint globalJoint =
                articulation.firstJoint + localJoint;
            device const MRJointDescriptorGPU& joint =
                joints[globalJoint];
            const uint localParent =
                joint.parentBody - articulation.firstBody;
            const uint localChild =
                joint.childBody - articulation.firstBody;
            if (known[localParent] != 0u &&
                known[localChild] == 0u) {
                known[localChild] = 1u;
                traversal[discovered++] = localChild;
                progressed = true;
            }
        }
        if (!progressed) {
            break;
        }
    }
    if (discovered != articulation.bodyCount) {
        setFailure(
            status,
            MR_INVERSE_MASS_UNSUPPORTED_TOPOLOGY,
            MR_INVALID_INDEX
        );
        statuses[environment] = status;
        return;
    }

    const uint qBase = environment * dispatch.qStride;
    device const float* environmentQ = q + qBase;
    for (uint localQ = 0u;
         localQ < articulation.nq;
         ++localQ) {
        if (!isfinite(environmentQ[localQ])) {
            setFailure(
                status,
                MR_INVERSE_MASS_NONFINITE_INPUT,
                localQ
            );
            statuses[environment] = status;
            return;
        }
    }
    float maximumInput = 0.0f;
    const uint rhsEnvironmentBase =
        environment * dispatch.rhsEnvironmentStride;
    for (uint rhsIndex = 0u;
         rhsIndex < dispatch.rhsCount;
         ++rhsIndex) {
        const uint rhsBase =
            rhsEnvironmentBase +
            rhsIndex * dispatch.rhsVectorStride;
        for (uint localV = 0u;
             localV < articulation.nv;
             ++localV) {
            const float value =
                rightHandSides[rhsBase + localV];
            if (!isfinite(value)) {
                setFailure(
                    status,
                    MR_INVERSE_MASS_NONFINITE_INPUT,
                    rhsIndex * articulation.nv + localV
                );
                statuses[environment] = status;
                return;
            }
            maximumInput = max(maximumInput, abs(value));
        }
    }

    if (articulation.rootType == MR_ROOT_FLOATING) {
        float4 checkedRootRotation;
        if (!finite3(float3(
                environmentQ[0],
                environmentQ[1],
                environmentQ[2]
            )) ||
            !normalizedQuaternion(
                float4(
                    environmentQ[3],
                    environmentQ[4],
                    environmentQ[5],
                    environmentQ[6]
                ),
                checkedRootRotation,
                true
            )) {
            setFailure(
                status,
                MR_INVERSE_MASS_INVALID_QUATERNION,
                0u
            );
            statuses[environment] = status;
            return;
        }
        bodyPosition[rootLocal] = float3(
            environmentQ[0],
            environmentQ[1],
            environmentQ[2]
        );
        bodyRotation[rootLocal] = checkedRootRotation;
    } else {
        bodyPosition[rootLocal] = float3(0.0f);
        bodyRotation[rootLocal] =
            float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    parentToBody[rootLocal] = float3(0.0f);
    motionAngular[rootLocal] = float3(0.0f);
    motionLinear[rootLocal] = float3(0.0f);

    for (uint traversalIndex = 1u;
         traversalIndex < articulation.bodyCount;
         ++traversalIndex) {
        const uint localChild = traversal[traversalIndex];
        const uint globalJoint = inboundJoint[localChild];
        device const MRJointDescriptorGPU& joint =
            joints[globalJoint];
        const uint localParent = parentLocal[localChild];
        float4 parentJointRotation;
        float4 childJointRotation;
        if (!normalizedQuaternion(
                joint.parentRotation,
                parentJointRotation,
                true
            ) ||
            !normalizedQuaternion(
                joint.childRotation,
                childJointRotation,
                true
            )) {
            setFailure(
                status,
                MR_INVERSE_MASS_INVALID_MODEL,
                globalJoint
            );
            statuses[environment] = status;
            return;
        }
        const float4 parentToJointRotation =
            quaternionMultiply(
                bodyRotation[localParent],
                parentJointRotation
            );
        float3 axisInJoint = float3(1.0f, 0.0f, 0.0f);
        float4 motionRotation =
            float4(0.0f, 0.0f, 0.0f, 1.0f);
        if (joint.nv == 1u) {
            axisInJoint = normalize(joint.axis0.xyz);
            const uint localQ =
                joint.qOffset - articulation.qOffset;
            motionRotation = axisAngleQuaternion(
                axisInJoint,
                environmentQ[localQ]
            );
        }
        float4 checkedChildRotation;
        if (!normalizedQuaternion(
                quaternionMultiply(
                    quaternionMultiply(
                        parentToJointRotation,
                        motionRotation
                    ),
                    quaternionConjugate(childJointRotation)
                ),
                checkedChildRotation,
                false
            )) {
            setFailure(
                status,
                MR_INVERSE_MASS_NONFINITE_RESULT,
                articulation.firstBody + localChild
            );
            statuses[environment] = status;
            return;
        }
        bodyRotation[localChild] = checkedChildRotation;
        const float3 jointPosition =
            bodyPosition[localParent] +
            quaternionRotate(
                bodyRotation[localParent],
                joint.parentAnchor.xyz
            );
        const float3 jointAxis = quaternionRotate(
            parentToJointRotation,
            axisInJoint
        );
        const float3 childAnchor = quaternionRotate(
            bodyRotation[localChild],
            joint.childAnchor.xyz
        );
        bodyPosition[localChild] =
            jointPosition - childAnchor;
        parentToBody[localChild] =
            bodyPosition[localChild] -
            bodyPosition[localParent];
        motionAngular[localChild] =
            joint.nv == 1u ? jointAxis : float3(0.0f);
        motionLinear[localChild] =
            joint.nv == 1u
                ? -cross(jointAxis, childAnchor)
                : float3(0.0f);
        if (!finite3(bodyPosition[localChild]) ||
            !finite4(bodyRotation[localChild]) ||
            !finite3(motionAngular[localChild]) ||
            !finite3(motionLinear[localChild])) {
            setFailure(
                status,
                MR_INVERSE_MASS_NONFINITE_RESULT,
                articulation.firstBody + localChild
            );
            statuses[environment] = status;
            return;
        }
    }

    for (uint localBody = 0u;
         localBody < articulation.bodyCount;
         ++localBody) {
        const uint globalBody =
            articulation.firstBody + localBody;
        device const MRBodyPropertiesGPU& body =
            bodies[globalBody];
        const uint expectedParent =
            localBody == rootLocal
                ? MR_INVALID_INDEX
                : joints[inboundJoint[localBody]].parentBody;
        const uint expectedInbound =
            localBody == rootLocal
                ? MR_INVALID_INDEX
                : inboundJoint[localBody];
        if (body.articulationIndex !=
                dispatch.articulationIndex ||
            body.parentBody != expectedParent ||
            body.inboundJoint != expectedInbound ||
            body.motionType != MR_MOTION_DYNAMIC ||
            !(body.massAndInverseMass.x > 0.0f) ||
            !finite4(body.massAndInverseMass) ||
            !finite4(body.inertiaRow0) ||
            !finite4(body.inertiaRow1) ||
            !finite4(body.inertiaRow2)) {
            setFailure(
                status,
                MR_INVERSE_MASS_INVALID_MODEL,
                globalBody
            );
            statuses[environment] = status;
            return;
        }
        const uint matrixBase = localBody * 36u;
        for (uint entry = 0u; entry < 36u; ++entry) {
            articulatedInertia[matrixBase + entry] = 0.0f;
        }
        const float3 rotationColumn0 = quaternionRotate(
            bodyRotation[localBody],
            float3(1.0f, 0.0f, 0.0f)
        );
        const float3 rotationColumn1 = quaternionRotate(
            bodyRotation[localBody],
            float3(0.0f, 1.0f, 0.0f)
        );
        const float3 rotationColumn2 = quaternionRotate(
            bodyRotation[localBody],
            float3(0.0f, 0.0f, 1.0f)
        );
        for (uint row = 0u; row < 3u; ++row) {
            for (uint column = 0u;
                 column < 3u;
                 ++column) {
                articulatedInertia[
                    matrixBase + row * 6u + column
                ] = worldInertiaElement(
                    body,
                    rotationColumn0,
                    rotationColumn1,
                    rotationColumn2,
                    row,
                    column
                );
            }
            articulatedInertia[
                matrixBase +
                (3u + row) * 6u +
                (3u + row)
            ] = body.massAndInverseMass.x;
        }
    }

    if (articulation.rootType == MR_ROOT_FLOATING) {
        const uint rootMatrixBase = rootLocal * 36u;
        for (uint axis = 0u; axis < 3u; ++axis) {
            articulatedInertia[
                rootMatrixBase +
                (3u + axis) * 6u +
                (3u + axis)
            ] += dofs[articulation.vOffset + axis].drive.z;
            articulatedInertia[
                rootMatrixBase + axis * 6u + axis
            ] += dofs[
                articulation.vOffset + 3u + axis
            ].drive.z;
        }
    }

    float minimumPivot = INFINITY;
    float maximumPivot = 0.0f;
    for (uint reverse = 0u;
         reverse + 1u < articulation.bodyCount;
         ++reverse) {
        const uint traversalIndex =
            articulation.bodyCount - 1u - reverse;
        const uint localBody = traversal[traversalIndex];
        const uint localParent = parentLocal[localBody];
        const uint globalJoint = inboundJoint[localBody];
        device const MRJointDescriptorGPU& joint =
            joints[globalJoint];
        const uint matrixBase = localBody * 36u;

        if (joint.nv == 1u) {
            float3 projectedAngular = float3(0.0f);
            float3 projectedLinear = float3(0.0f);
            for (uint row = 0u; row < 6u; ++row) {
                float value = 0.0f;
                for (uint column = 0u;
                     column < 6u;
                     ++column) {
                    value +=
                        articulatedInertia[
                            matrixBase + row * 6u + column
                        ] *
                        spatialComponent(
                            motionAngular[localBody],
                            motionLinear[localBody],
                            column
                        );
                }
                projectedInertia[localBody * 6u + row] = value;
                setSpatialComponent(
                    projectedAngular,
                    projectedLinear,
                    row,
                    value
                );
            }
            const uint localV =
                joint.vOffset - articulation.vOffset;
            const float denominator =
                dot(
                    motionAngular[localBody],
                    projectedAngular
                ) +
                dot(
                    motionLinear[localBody],
                    projectedLinear
                ) +
                dofs[
                    articulation.vOffset + localV
                ].drive.z;
            float maximumInertia = 0.0f;
            for (uint entry = 0u; entry < 36u; ++entry) {
                maximumInertia = max(
                    maximumInertia,
                    abs(articulatedInertia[matrixBase + entry])
                );
            }
            const float pivotFloor = max(
                kAbsolutePivotFloor,
                maximumInertia * 6.0f * kFloatEpsilon
            );
            if (!(denominator > pivotFloor) ||
                !isfinite(denominator)) {
                setFailure(
                    status,
                    MR_INVERSE_MASS_FACTORIZATION_FAILED,
                    localV
                );
                status.diagnostics = float4(
                    minimumPivot,
                    maximumPivot,
                    denominator,
                    maximumInertia
                );
                statuses[environment] = status;
                return;
            }
            jointDenominator[localBody] = denominator;
            const float pivot = sqrt(denominator);
            minimumPivot = min(minimumPivot, pivot);
            maximumPivot = max(maximumPivot, pivot);
            for (uint row = 0u; row < 6u; ++row) {
                for (uint column = 0u;
                     column < 6u;
                     ++column) {
                    articulatedInertia[
                        matrixBase + row * 6u + column
                    ] -=
                        projectedInertia[
                            localBody * 6u + row
                        ] *
                        projectedInertia[
                            localBody * 6u + column
                        ] /
                        denominator;
                }
            }
        }

        const float3 offset = parentToBody[localBody];
        const uint parentMatrixBase = localParent * 36u;
        for (uint row = 0u; row < 6u; ++row) {
            for (uint column = 0u; column < 6u; ++column) {
                float transformed = 0.0f;
                for (uint left = 0u; left < 6u; ++left) {
                    const float leftTransform =
                        motionTransformElement(offset, left, row);
                    if (leftTransform == 0.0f) {
                        continue;
                    }
                    for (uint right = 0u;
                         right < 6u;
                         ++right) {
                        transformed +=
                            leftTransform *
                            articulatedInertia[
                                matrixBase +
                                left * 6u +
                                right
                            ] *
                            motionTransformElement(
                                offset,
                                right,
                                column
                            );
                    }
                }
                articulatedInertia[
                    parentMatrixBase + row * 6u + column
                ] += transformed;
            }
        }
    }

    if (articulation.rootType == MR_ROOT_FLOATING) {
        const uint rootMatrixBase = rootLocal * 36u;
        float maximumRootInertia = 0.0f;
        for (uint entry = 0u; entry < 36u; ++entry) {
            maximumRootInertia = max(
                maximumRootInertia,
                abs(articulatedInertia[rootMatrixBase + entry])
            );
            rootFactor[entry] = 0.0f;
        }
        const float pivotFloor = max(
            kAbsolutePivotFloor,
            maximumRootInertia * 6.0f * kFloatEpsilon
        );
        for (uint row = 0u; row < 6u; ++row) {
            for (uint column = 0u;
                 column <= row;
                 ++column) {
                float value = articulatedInertia[
                    rootMatrixBase + row * 6u + column
                ];
                for (uint inner = 0u;
                     inner < column;
                     ++inner) {
                    value -=
                        rootFactor[row * 6u + inner] *
                        rootFactor[column * 6u + inner];
                }
                if (row == column) {
                    if (!(value > pivotFloor) ||
                        !isfinite(value)) {
                        setFailure(
                            status,
                            MR_INVERSE_MASS_FACTORIZATION_FAILED,
                            row
                        );
                        status.diagnostics = float4(
                            minimumPivot,
                            maximumPivot,
                            value,
                            maximumRootInertia
                        );
                        statuses[environment] = status;
                        return;
                    }
                    const float pivot = sqrt(value);
                    rootFactor[row * 6u + row] = pivot;
                    minimumPivot = min(minimumPivot, pivot);
                    maximumPivot = max(maximumPivot, pivot);
                } else {
                    rootFactor[row * 6u + column] =
                        value /
                        rootFactor[
                            column * 6u + column
                        ];
                }
            }
        }
    }

    float maximumOutput = 0.0f;
    for (uint rhsIndex = 0u;
         rhsIndex < dispatch.rhsCount;
         ++rhsIndex) {
        const uint rhsBase =
            rhsEnvironmentBase +
            rhsIndex * dispatch.rhsVectorStride;
        for (uint localBody = 0u;
             localBody < articulation.bodyCount;
             ++localBody) {
            for (uint component = 0u;
                 component < 6u;
                 ++component) {
                articulatedBias[
                    localBody * 6u + component
                ] = 0.0f;
            }
            jointResidual[localBody] = 0.0f;
        }
        if (articulation.rootType == MR_ROOT_FLOATING) {
            for (uint axis = 0u; axis < 3u; ++axis) {
                articulatedBias[
                    rootLocal * 6u + axis
                ] = -rightHandSides[rhsBase + 3u + axis];
                articulatedBias[
                    rootLocal * 6u + 3u + axis
                ] = -rightHandSides[rhsBase + axis];
            }
        }

        for (uint reverse = 0u;
             reverse + 1u < articulation.bodyCount;
             ++reverse) {
            const uint traversalIndex =
                articulation.bodyCount - 1u - reverse;
            const uint localBody = traversal[traversalIndex];
            const uint localParent = parentLocal[localBody];
            const uint globalJoint = inboundJoint[localBody];
            device const MRJointDescriptorGPU& joint =
                joints[globalJoint];
            float3 propagatedTorque = float3(
                articulatedBias[localBody * 6u + 0u],
                articulatedBias[localBody * 6u + 1u],
                articulatedBias[localBody * 6u + 2u]
            );
            float3 propagatedForce = float3(
                articulatedBias[localBody * 6u + 3u],
                articulatedBias[localBody * 6u + 4u],
                articulatedBias[localBody * 6u + 5u]
            );
            if (joint.nv == 1u) {
                const uint localV =
                    joint.vOffset - articulation.vOffset;
                const float residual =
                    rightHandSides[rhsBase + localV] -
                    dot(
                        motionAngular[localBody],
                        propagatedTorque
                    ) -
                    dot(
                        motionLinear[localBody],
                        propagatedForce
                    );
                jointResidual[localBody] = residual;
                for (uint component = 0u;
                     component < 6u;
                     ++component) {
                    const float contribution =
                        projectedInertia[
                            localBody * 6u + component
                        ] *
                        residual /
                        jointDenominator[localBody];
                    if (component < 3u) {
                        propagatedTorque[component] +=
                            contribution;
                    } else {
                        propagatedForce[component - 3u] +=
                            contribution;
                    }
                }
            }
            propagatedTorque += cross(
                parentToBody[localBody],
                propagatedForce
            );
            for (uint component = 0u;
                 component < 6u;
                 ++component) {
                articulatedBias[
                    localParent * 6u + component
                ] += spatialComponent(
                    propagatedTorque,
                    propagatedForce,
                    component
                );
            }
        }

        if (articulation.rootType == MR_ROOT_FLOATING) {
            for (uint row = 0u; row < 6u; ++row) {
                float value =
                    -articulatedBias[
                        rootLocal * 6u + row
                    ];
                for (uint column = 0u;
                     column < row;
                     ++column) {
                    value -=
                        rootFactor[row * 6u + column] *
                        rootIntermediate[column];
                }
                rootIntermediate[row] =
                    value / rootFactor[row * 6u + row];
            }
            float3 rootAngular = float3(0.0f);
            float3 rootLinear = float3(0.0f);
            for (uint reverse = 0u;
                 reverse < 6u;
                 ++reverse) {
                const uint row = 5u - reverse;
                float value = rootIntermediate[row];
                for (uint column = row + 1u;
                     column < 6u;
                     ++column) {
                    value -=
                        rootFactor[column * 6u + row] *
                        spatialComponent(
                            rootAngular,
                            rootLinear,
                            column
                        );
                }
                setSpatialComponent(
                    rootAngular,
                    rootLinear,
                    row,
                    value / rootFactor[row * 6u + row]
                );
            }
            accelerationAngular[rootLocal] = rootAngular;
            accelerationLinear[rootLocal] = rootLinear;
            candidateOutput[
                rhsIndex * kMaxDofs + 0u
            ] = rootLinear.x;
            candidateOutput[
                rhsIndex * kMaxDofs + 1u
            ] = rootLinear.y;
            candidateOutput[
                rhsIndex * kMaxDofs + 2u
            ] = rootLinear.z;
            candidateOutput[
                rhsIndex * kMaxDofs + 3u
            ] = rootAngular.x;
            candidateOutput[
                rhsIndex * kMaxDofs + 4u
            ] = rootAngular.y;
            candidateOutput[
                rhsIndex * kMaxDofs + 5u
            ] = rootAngular.z;
        } else {
            accelerationAngular[rootLocal] = float3(0.0f);
            accelerationLinear[rootLocal] = float3(0.0f);
        }

        for (uint traversalIndex = 1u;
             traversalIndex < articulation.bodyCount;
             ++traversalIndex) {
            const uint localBody = traversal[traversalIndex];
            const uint localParent = parentLocal[localBody];
            const uint globalJoint = inboundJoint[localBody];
            device const MRJointDescriptorGPU& joint =
                joints[globalJoint];
            const float3 parentAngular =
                accelerationAngular[localParent];
            const float3 parentLinear =
                accelerationLinear[localParent] +
                cross(
                    accelerationAngular[localParent],
                    parentToBody[localBody]
                );
            float3 angular = parentAngular;
            float3 linear = parentLinear;
            if (joint.nv == 1u) {
                float projectedParent = 0.0f;
                for (uint component = 0u;
                     component < 6u;
                     ++component) {
                    projectedParent +=
                        projectedInertia[
                            localBody * 6u + component
                        ] *
                        spatialComponent(
                            parentAngular,
                            parentLinear,
                            component
                        );
                }
                const float jointAcceleration =
                    (
                        jointResidual[localBody] -
                        projectedParent
                    ) /
                    jointDenominator[localBody];
                const uint localV =
                    joint.vOffset - articulation.vOffset;
                candidateOutput[
                    rhsIndex * kMaxDofs + localV
                ] = jointAcceleration;
                angular +=
                    motionAngular[localBody] *
                    jointAcceleration;
                linear +=
                    motionLinear[localBody] *
                    jointAcceleration;
            }
            accelerationAngular[localBody] = angular;
            accelerationLinear[localBody] = linear;
        }

        for (uint localV = 0u;
             localV < articulation.nv;
             ++localV) {
            const float value =
                candidateOutput[
                    rhsIndex * kMaxDofs + localV
                ];
            if (!isfinite(value)) {
                setFailure(
                    status,
                    MR_INVERSE_MASS_NONFINITE_RESULT,
                    rhsIndex * articulation.nv + localV
                );
                statuses[environment] = status;
                return;
            }
            maximumOutput = max(maximumOutput, abs(value));
        }
    }

    const uint outputEnvironmentBase =
        environment * dispatch.outputEnvironmentStride;
    for (uint rhsIndex = 0u;
         rhsIndex < dispatch.rhsCount;
         ++rhsIndex) {
        const uint outputBase =
            outputEnvironmentBase +
            rhsIndex * dispatch.outputVectorStride;
        for (uint localV = 0u;
             localV < articulation.nv;
             ++localV) {
            output[outputBase + localV] =
                candidateOutput[
                    rhsIndex * kMaxDofs + localV
                ];
        }
    }
    status.diagnostics = float4(
        minimumPivot,
        maximumPivot,
        maximumOutput,
        maximumInput
    );
    statuses[environment] = status;
}
