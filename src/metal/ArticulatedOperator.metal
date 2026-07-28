#include <metal_stdlib>

#include "metalrobo/engine_types.h"

using namespace metal;

namespace {

constant float kQuaternionTolerance = 2.0e-5f;
constant float kQuaternionMinimum = 1.0e-12f;
constant float kFloatEpsilon = 1.1920928955078125e-7f;
constant float kFactorAbsolutePivotFloor = 1.0e-12f;
constant float kInertiaSymmetryTolerance = 1.0e-6f;
constant float kInverseConsistencyTolerance = 2.0e-4f;

struct MotionColumn {
    float3 linear;
    float3 angular;
};

inline bool finite3(const float3 value) {
    return all(isfinite(value));
}

inline bool finite4(const float4 value) {
    return all(isfinite(value));
}

inline float maximumAbsolute3(const float3 value) {
    return max(max(abs(value.x), abs(value.y)), abs(value.z));
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
    const bool requireUnitInput
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
    if (requireUnitInput &&
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

inline float3 inertiaMultiply(
    device const MRBodyPropertiesGPU& body,
    const float3 bodyAngular
) {
    return float3(
        dot(body.inertiaRow0.xyz, bodyAngular),
        dot(body.inertiaRow1.xyz, bodyAngular),
        dot(body.inertiaRow2.xyz, bodyAngular)
    );
}

inline bool symmetricInertia(
    const float3 row0,
    const float3 row1,
    const float3 row2
) {
    return
        abs(row0.y - row1.x) <=
            kInertiaSymmetryTolerance &&
        abs(row0.z - row2.x) <=
            kInertiaSymmetryTolerance &&
        abs(row1.z - row2.y) <=
            kInertiaSymmetryTolerance;
}

inline bool positiveDefiniteInertia(
    const float3 input0,
    const float3 input1,
    const float3 input2
) {
    // Scale before evaluating Sylvester's criterion. Canonical CPU records
    // are FP32, but their products need not fit FP32 without scaling.
    const float scale = max(
        maximumAbsolute3(input0),
        max(
            maximumAbsolute3(input1),
            maximumAbsolute3(input2)
        )
    );
    if (!(scale > 0.0f) || !isfinite(scale)) {
        return false;
    }
    const float3 row0 = input0 / scale;
    const float3 row1 = input1 / scale;
    const float3 row2 = input2 / scale;
    const float firstMinor = row0.x;
    const float secondMinor =
        row0.x * row1.y - row0.y * row1.x;
    const float determinant = dot(row0, cross(row1, row2));
    return
        firstMinor > 0.0f &&
        secondMinor > 0.0f &&
        determinant > 0.0f &&
        isfinite(secondMinor) &&
        isfinite(determinant);
}

inline bool inverseConsistentInertia(
    const float3 inertia0,
    const float3 inertia1,
    const float3 inertia2,
    const float3 inverse0,
    const float3 inverse1,
    const float3 inverse2
) {
    const float3 column0 =
        float3(inverse0.x, inverse1.x, inverse2.x);
    const float3 column1 =
        float3(inverse0.y, inverse1.y, inverse2.y);
    const float3 column2 =
        float3(inverse0.z, inverse1.z, inverse2.z);
    const float3 product0 = float3(
        dot(inertia0, column0),
        dot(inertia0, column1),
        dot(inertia0, column2)
    );
    const float3 product1 = float3(
        dot(inertia1, column0),
        dot(inertia1, column1),
        dot(inertia1, column2)
    );
    const float3 product2 = float3(
        dot(inertia2, column0),
        dot(inertia2, column1),
        dot(inertia2, column2)
    );
    return
        finite3(product0) &&
        finite3(product1) &&
        finite3(product2) &&
        all(abs(product0 - float3(1.0f, 0.0f, 0.0f)) <=
            float3(kInverseConsistencyTolerance)) &&
        all(abs(product1 - float3(0.0f, 1.0f, 0.0f)) <=
            float3(kInverseConsistencyTolerance)) &&
        all(abs(product2 - float3(0.0f, 0.0f, 1.0f)) <=
            float3(kInverseConsistencyTolerance));
}

inline bool validBodyInertia(
    device const MRBodyPropertiesGPU& body
) {
    if (!finite4(body.inertiaRow0) ||
        !finite4(body.inertiaRow1) ||
        !finite4(body.inertiaRow2) ||
        !finite4(body.inverseInertiaRow0) ||
        !finite4(body.inverseInertiaRow1) ||
        !finite4(body.inverseInertiaRow2)) {
        return false;
    }
    const float3 inertia0 = body.inertiaRow0.xyz;
    const float3 inertia1 = body.inertiaRow1.xyz;
    const float3 inertia2 = body.inertiaRow2.xyz;
    const float3 inverse0 = body.inverseInertiaRow0.xyz;
    const float3 inverse1 = body.inverseInertiaRow1.xyz;
    const float3 inverse2 = body.inverseInertiaRow2.xyz;
    return
        symmetricInertia(inertia0, inertia1, inertia2) &&
        symmetricInertia(inverse0, inverse1, inverse2) &&
        positiveDefiniteInertia(inertia0, inertia1, inertia2) &&
        positiveDefiniteInertia(inverse0, inverse1, inverse2) &&
        inverseConsistentInertia(
            inertia0,
            inertia1,
            inertia2,
            inverse0,
            inverse1,
            inverse2
        );
}

inline bool zero4(const float4 value) {
    return all(value == float4(0.0f));
}

inline bool validDofParameters(
    device const MRDofPropertiesGPU& dof,
    const bool root,
    const uint jointType
) {
    constexpr uint knownFlags =
        MR_DOF_FLAG_ROOT |
        MR_DOF_FLAG_ACTUATED |
        MR_DOF_FLAG_POSITION_LIMIT |
        MR_DOF_FLAG_VELOCITY_LIMIT |
        MR_DOF_FLAG_EFFORT_LIMIT |
        MR_DOF_FLAG_DRIVE;
    if (!finite4(dof.limits) ||
        !finite4(dof.drive) ||
        dof.reserved0 != 0u ||
        dof.reserved1 != 0u ||
        (dof.flags & ~knownFlags) != 0u ||
        any(dof.drive < float4(0.0f))) {
        return false;
    }
    if (root) {
        return dof.flags == MR_DOF_FLAG_ROOT &&
            zero4(dof.limits) &&
            zero4(dof.drive);
    }
    if ((dof.flags & MR_DOF_FLAG_ROOT) != 0u) {
        return false;
    }
    const bool actuated =
        (dof.flags & MR_DOF_FLAG_ACTUATED) != 0u;
    const bool positionLimited =
        (dof.flags & MR_DOF_FLAG_POSITION_LIMIT) != 0u;
    const bool velocityLimited =
        (dof.flags & MR_DOF_FLAG_VELOCITY_LIMIT) != 0u;
    const bool effortLimited =
        (dof.flags & MR_DOF_FLAG_EFFORT_LIMIT) != 0u;
    const bool driven =
        (dof.flags & MR_DOF_FLAG_DRIVE) != 0u;
    return
        (actuated || (!effortLimited && !driven)) &&
        (!driven || actuated) &&
        (driven ||
         (dof.drive.x == 0.0f && dof.drive.y == 0.0f)) &&
        (!positionLimited ||
         (dof.qIndex != MR_INVALID_INDEX &&
          jointType != MR_JOINT_CONTINUOUS &&
          dof.limits.x <= dof.limits.y)) &&
        (positionLimited ||
         (dof.limits.x == 0.0f && dof.limits.y == 0.0f)) &&
        (velocityLimited
             ? dof.limits.z > 0.0f
             : dof.limits.z == 0.0f) &&
        (effortLimited
             ? dof.limits.w > 0.0f
             : dof.limits.w == 0.0f);
}

inline void setFailure(
    thread MRArticulatedOperatorStatusGPU& status,
    const uint code,
    const uint failingIndex
) {
    status.code = code;
    status.failingIndex = failingIndex;
}

inline MotionColumn bodyMotionForDof(
    const uint localBody,
    const uint dof,
    device const MRArticulationGPU& articulation,
    device const MRJointDescriptorGPU* joints,
    threadgroup const float3* bodyPosition,
    threadgroup const float3* jointPosition,
    threadgroup const float3* jointAxis,
    threadgroup const uint* inboundJoint,
    threadgroup const uint* parentLocal
) {
    MotionColumn result;
    result.linear = float3(0.0f);
    result.angular = float3(0.0f);

    const uint rootLocal =
        articulation.rootBody - articulation.firstBody;
    if (articulation.rootType == MR_ROOT_FLOATING) {
        if (dof < 3u) {
            result.linear[dof] = 1.0f;
            return result;
        }
        if (dof < 6u) {
            result.angular[dof - 3u] = 1.0f;
            result.linear = cross(
                result.angular,
                bodyPosition[localBody] - bodyPosition[rootLocal]
            );
            return result;
        }
    }

    uint cursor = localBody;
    for (uint depth = 0u;
         depth < articulation.bodyCount && cursor != rootLocal;
         ++depth) {
        const uint globalJoint = inboundJoint[cursor];
        if (globalJoint == MR_INVALID_INDEX) {
            return result;
        }
        device const MRJointDescriptorGPU& joint =
            joints[globalJoint];
        if (joint.nv == 1u &&
            joint.vOffset - articulation.vOffset == dof) {
            if (joint.jointType == MR_JOINT_PRISMATIC) {
                result.linear = jointAxis[cursor];
            } else {
                result.angular = jointAxis[cursor];
                result.linear = cross(
                    result.angular,
                    bodyPosition[localBody] -
                        jointPosition[cursor]
                );
            }
            return result;
        }
        cursor = parentLocal[cursor];
    }
    return result;
}

inline float massElement(
    const uint row,
    const uint column,
    device const MRArticulationGPU& articulation,
    device const MRJointDescriptorGPU* joints,
    device const MRBodyPropertiesGPU* bodies,
    threadgroup const float3* bodyPosition,
    threadgroup const float4* bodyRotation,
    threadgroup const float3* jointPosition,
    threadgroup const float3* jointAxis,
    threadgroup const uint* inboundJoint,
    threadgroup const uint* parentLocal
) {
    float value = 0.0f;
    for (uint localBody = 0u;
         localBody < articulation.bodyCount;
         ++localBody) {
        const MotionColumn left = bodyMotionForDof(
            localBody,
            row,
            articulation,
            joints,
            bodyPosition,
            jointPosition,
            jointAxis,
            inboundJoint,
            parentLocal
        );
        const MotionColumn right = bodyMotionForDof(
            localBody,
            column,
            articulation,
            joints,
            bodyPosition,
            jointPosition,
            jointAxis,
            inboundJoint,
            parentLocal
        );
        device const MRBodyPropertiesGPU& body =
            bodies[articulation.firstBody + localBody];
        const float3 leftBodyAngular = quaternionRotate(
            quaternionConjugate(bodyRotation[localBody]),
            left.angular
        );
        const float3 rightBodyAngular = quaternionRotate(
            quaternionConjugate(bodyRotation[localBody]),
            right.angular
        );
        value +=
            body.massAndInverseMass.x *
                dot(left.linear, right.linear) +
            dot(
                leftBodyAngular,
                inertiaMultiply(body, rightBodyAngular)
            );
    }
    return value;
}

inline bool validDispatch(
    device const MRWorldGPU& world,
    device const MRArticulatedOperatorDispatchGPU& dispatch,
    thread MRArticulatedOperatorStatusGPU& status
) {
    if (world.abiVersion != MR_ENGINE_ABI_VERSION ||
        dispatch.articulationIndex >= world.articulationCount ||
        dispatch.environmentCount == 0u ||
        dispatch.pointCount > MR_ARTICULATED_OPERATOR_MAX_POINTS ||
        dispatch.reserved0 != 0u ||
        (dispatch.flags &
         ~MR_ARTICULATED_OPERATOR_WRITE_DIAGNOSTIC_MASS) != 0u) {
        setFailure(
            status,
            MR_ARTICULATED_OPERATOR_INVALID_DISPATCH,
            MR_INVALID_INDEX
        );
        return false;
    }
    return true;
}

inline bool validModelAndLayout(
    device const MRWorldGPU& world,
    device const MRArticulationGPU& articulation,
    device const MRJointDescriptorGPU* joints,
    device const MRDofPropertiesGPU* dofs,
    device const MRBodyPropertiesGPU* bodies,
    device const MRArticulatedOperatorDispatchGPU& dispatch,
    threadgroup uint* inboundJoint,
    threadgroup uint* parentLocal,
    threadgroup uchar* known,
    thread MRArticulatedOperatorStatusGPU& status
) {
    if ((articulation.rootType != MR_ROOT_FIXED &&
         articulation.rootType != MR_ROOT_FLOATING) ||
        articulation.bodyCount == 0u ||
        articulation.bodyCount >
            MR_ARTICULATED_OPERATOR_MAX_BODIES ||
        articulation.nv == 0u ||
        articulation.nv > MR_ARTICULATED_OPERATOR_MAX_DOFS ||
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
        articulation.jointCount + 1u != articulation.bodyCount) {
        setFailure(
            status,
            articulation.bodyCount >
                    MR_ARTICULATED_OPERATOR_MAX_BODIES ||
                articulation.nv >
                    MR_ARTICULATED_OPERATOR_MAX_DOFS
                ? MR_ARTICULATED_OPERATOR_CAPACITY_OVERFLOW
                : MR_ARTICULATED_OPERATOR_INVALID_MODEL,
            MR_INVALID_INDEX
        );
        return false;
    }

    const uint rootQ =
        articulation.rootType == MR_ROOT_FLOATING ? 7u : 0u;
    const uint rootV =
        articulation.rootType == MR_ROOT_FLOATING ? 6u : 0u;
    uint expectedNq = rootQ;
    uint expectedNv = rootV;
    const uint rootLocal =
        articulation.rootBody - articulation.firstBody;
    for (uint localDof = 0u;
         localDof < rootV;
         ++localDof) {
        const uint globalV =
            articulation.vOffset + localDof;
        const uint expectedQ =
            localDof < 3u
                ? articulation.qOffset + localDof
                : MR_INVALID_INDEX;
        device const MRDofPropertiesGPU& dof = dofs[globalV];
        if (dof.articulationIndex !=
                dispatch.articulationIndex ||
            dof.jointIndex != MR_INVALID_INDEX ||
            dof.qIndex != expectedQ ||
            dof.vIndex != globalV ||
            dof.localDof != localDof ||
            !validDofParameters(dof, true, MR_JOINT_FREE)) {
            setFailure(
                status,
                MR_ARTICULATED_OPERATOR_INVALID_MODEL,
                globalV
            );
            return false;
        }
    }
    for (uint localBody = 0u;
         localBody < articulation.bodyCount;
         ++localBody) {
        inboundJoint[localBody] = MR_INVALID_INDEX;
        parentLocal[localBody] = MR_INVALID_INDEX;
        known[localBody] = 0u;
    }

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
                MR_ARTICULATED_OPERATOR_INVALID_MODEL,
                globalJoint
            );
            return false;
        }

        uint jointNq = 0u;
        uint jointNv = 0u;
        if (joint.jointType == MR_JOINT_REVOLUTE ||
            joint.jointType == MR_JOINT_CONTINUOUS ||
            joint.jointType == MR_JOINT_PRISMATIC) {
            jointNq = 1u;
            jointNv = 1u;
        } else if (joint.jointType != MR_JOINT_FIXED) {
            setFailure(
                status,
                MR_ARTICULATED_OPERATOR_UNSUPPORTED_TOPOLOGY,
                globalJoint
            );
            return false;
        }
        if (joint.nq != jointNq ||
            joint.nv != jointNv ||
            expectedNq > articulation.nq ||
            jointNq > articulation.nq - expectedNq ||
            expectedNv > articulation.nv ||
            jointNv > articulation.nv - expectedNv ||
            joint.qOffset != articulation.qOffset + expectedNq ||
            joint.vOffset != articulation.vOffset + expectedNv) {
            setFailure(
                status,
                MR_ARTICULATED_OPERATOR_INVALID_MODEL,
                globalJoint
            );
            return false;
        }
        if (jointNv == 1u) {
            const float axisNormSquared =
                dot(joint.axis0.xyz, joint.axis0.xyz);
            if (!finite4(joint.axis0) ||
                !(axisNormSquared > kQuaternionMinimum)) {
                setFailure(
                    status,
                    MR_ARTICULATED_OPERATOR_INVALID_MODEL,
                    globalJoint
                );
                return false;
            }
            device const MRDofPropertiesGPU& dof =
                dofs[joint.vOffset];
            if (dof.articulationIndex !=
                    dispatch.articulationIndex ||
                dof.jointIndex != globalJoint ||
                dof.qIndex != joint.qOffset ||
                dof.vIndex != joint.vOffset ||
                dof.localDof != 0u ||
                !validDofParameters(
                    dof,
                    false,
                    joint.jointType
                )) {
                setFailure(
                    status,
                    MR_ARTICULATED_OPERATOR_INVALID_MODEL,
                    joint.vOffset
                );
                return false;
            }
        }
        float4 checkedRotation;
        if (!normalizedQuaternion(
                joint.parentRotation,
                checkedRotation,
                true
            ) ||
            !normalizedQuaternion(
                joint.childRotation,
                checkedRotation,
                true
            )) {
            setFailure(
                status,
                MR_ARTICULATED_OPERATOR_INVALID_MODEL,
                globalJoint
            );
            return false;
        }

        const uint localChild =
            joint.childBody - articulation.firstBody;
        if (inboundJoint[localChild] != MR_INVALID_INDEX) {
            setFailure(
                status,
                MR_ARTICULATED_OPERATOR_UNSUPPORTED_TOPOLOGY,
                joint.childBody
            );
            return false;
        }
        inboundJoint[localChild] = globalJoint;
        parentLocal[localChild] =
            joint.parentBody - articulation.firstBody;
        expectedNq += jointNq;
        expectedNv += jointNv;
    }

    if (expectedNq != articulation.nq ||
        expectedNv != articulation.nv ||
        inboundJoint[rootLocal] != MR_INVALID_INDEX ||
        dispatch.qStride < articulation.nq ||
        dispatch.pointStride < dispatch.pointCount ||
        dispatch.bodyPoseStride < articulation.bodyCount ||
        dispatch.pointWorldStride < dispatch.pointCount ||
        dispatch.pointJacobianStride <
            dispatch.pointCount * 3u * articulation.nv ||
        dispatch.generalizedStride < articulation.nv ||
        ((dispatch.flags &
          MR_ARTICULATED_OPERATOR_WRITE_DIAGNOSTIC_MASS) != 0u &&
         dispatch.massMatrixStride <
            articulation.nv * articulation.nv)) {
        setFailure(
            status,
            MR_ARTICULATED_OPERATOR_INVALID_DISPATCH,
            MR_INVALID_INDEX
        );
        return false;
    }

    for (uint localBody = 0u;
         localBody < articulation.bodyCount;
         ++localBody) {
        const uint globalBody =
            articulation.firstBody + localBody;
        device const MRBodyPropertiesGPU& body = bodies[globalBody];
        if (body.articulationIndex !=
                dispatch.articulationIndex ||
            body.motionType != MR_MOTION_DYNAMIC ||
            !finite4(body.massAndInverseMass) ||
            !(body.massAndInverseMass.x > 0.0f) ||
            !(body.massAndInverseMass.y > 0.0f) ||
            abs(
                body.massAndInverseMass.x *
                    body.massAndInverseMass.y -
                1.0f
            ) > 3.0e-5f ||
            !finite4(body.centerOfMass) ||
            !validBodyInertia(body) ||
            !finite4(body.dampingAndSpeedLimits) ||
            any(body.dampingAndSpeedLimits < float4(0.0f))) {
            setFailure(
                status,
                MR_ARTICULATED_OPERATOR_INVALID_MODEL,
                globalBody
            );
            return false;
        }
        if (localBody == rootLocal) {
            if (body.parentBody != MR_INVALID_INDEX ||
                body.inboundJoint != MR_INVALID_INDEX) {
                setFailure(
                    status,
                    MR_ARTICULATED_OPERATOR_INVALID_MODEL,
                    globalBody
                );
                return false;
            }
        } else if (inboundJoint[localBody] == MR_INVALID_INDEX ||
                   body.parentBody !=
                       articulation.firstBody +
                           parentLocal[localBody] ||
                   body.inboundJoint != inboundJoint[localBody]) {
            setFailure(
                status,
                MR_ARTICULATED_OPERATOR_INVALID_MODEL,
                globalBody
            );
            return false;
        }
    }
    return true;
}

inline bool buildKinematics(
    device const MRArticulationGPU& articulation,
    device const MRJointDescriptorGPU* joints,
    device const float* q,
    threadgroup float3* bodyPosition,
    threadgroup float4* bodyRotation,
    threadgroup float3* jointPosition,
    threadgroup float3* jointAxis,
    threadgroup uchar* known,
    thread MRArticulatedOperatorStatusGPU& status
) {
    const uint rootLocal =
        articulation.rootBody - articulation.firstBody;
    if (articulation.rootType == MR_ROOT_FLOATING) {
        float4 checkedRootRotation;
        if (!finite3(float3(q[0], q[1], q[2])) ||
            !normalizedQuaternion(
                float4(q[3], q[4], q[5], q[6]),
                checkedRootRotation,
                true
            )) {
            setFailure(
                status,
                MR_ARTICULATED_OPERATOR_NONFINITE_INPUT,
                0u
            );
            return false;
        }
        bodyPosition[rootLocal] =
            float3(q[0], q[1], q[2]);
        bodyRotation[rootLocal] = checkedRootRotation;
    } else {
        bodyPosition[rootLocal] = float3(0.0f);
        bodyRotation[rootLocal] =
            float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    known[rootLocal] = 1u;

    uint discovered = 1u;
    for (uint pass = 0u;
         pass < articulation.bodyCount && discovered <
             articulation.bodyCount;
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
            if (known[localParent] == 0u ||
                known[localChild] != 0u) {
                continue;
            }

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
                    MR_ARTICULATED_OPERATOR_INVALID_MODEL,
                    globalJoint
                );
                return false;
            }
            const float4 parentToJointRotation =
                quaternionMultiply(
                    bodyRotation[localParent],
                    parentJointRotation
                );
            float4 motionRotation =
                float4(0.0f, 0.0f, 0.0f, 1.0f);
            float3 axisInJoint = float3(1.0f, 0.0f, 0.0f);
            float jointCoordinate = 0.0f;
            if (joint.nv == 1u) {
                const float axisMagnitude =
                    length(joint.axis0.xyz);
                axisInJoint = joint.axis0.xyz / axisMagnitude;
                const uint localQ =
                    joint.qOffset - articulation.qOffset;
                if (!isfinite(q[localQ])) {
                    setFailure(
                        status,
                        MR_ARTICULATED_OPERATOR_NONFINITE_INPUT,
                        localQ
                    );
                    return false;
                }
                jointCoordinate = q[localQ];
                if (joint.jointType == MR_JOINT_REVOLUTE ||
                    joint.jointType == MR_JOINT_CONTINUOUS) {
                    motionRotation = axisAngleQuaternion(
                        axisInJoint,
                        jointCoordinate
                    );
                }
            }

            const float4 candidateRotation = quaternionMultiply(
                quaternionMultiply(
                    parentToJointRotation,
                    motionRotation
                ),
                quaternionConjugate(childJointRotation)
            );
            float4 checkedChildRotation;
            if (!normalizedQuaternion(
                    candidateRotation,
                    checkedChildRotation,
                    false
                )) {
                setFailure(
                    status,
                    MR_ARTICULATED_OPERATOR_NONFINITE_RESULT,
                    joint.childBody
                );
                return false;
            }
            bodyRotation[localChild] = checkedChildRotation;
            jointAxis[localChild] = quaternionRotate(
                parentToJointRotation,
                axisInJoint
            );
            jointPosition[localChild] =
                bodyPosition[localParent] +
                quaternionRotate(
                    bodyRotation[localParent],
                    joint.parentAnchor.xyz
                ) +
                (joint.jointType == MR_JOINT_PRISMATIC
                    ? jointAxis[localChild] * jointCoordinate
                    : float3(0.0f));
            bodyPosition[localChild] =
                jointPosition[localChild] -
                quaternionRotate(
                    bodyRotation[localChild],
                    joint.childAnchor.xyz
                );
            if (!finite3(jointPosition[localChild]) ||
                !finite3(jointAxis[localChild]) ||
                !finite3(bodyPosition[localChild])) {
                setFailure(
                    status,
                    MR_ARTICULATED_OPERATOR_NONFINITE_RESULT,
                    joint.childBody
                );
                return false;
            }
            known[localChild] = 1u;
            ++discovered;
            progressed = true;
        }
        if (!progressed) {
            break;
        }
    }

    if (discovered != articulation.bodyCount) {
        setFailure(
            status,
            MR_ARTICULATED_OPERATOR_UNSUPPORTED_TOPOLOGY,
            MR_INVALID_INDEX
        );
        return false;
    }
    return true;
}

inline bool validatePoints(
    const uint environment,
    device const MRArticulationGPU& articulation,
    device const MRArticulatedOperatorDispatchGPU& dispatch,
    device const MRArticulatedPointImpulseGPU* points,
    thread MRArticulatedOperatorStatusGPU& status
) {
    const uint base = environment * dispatch.pointStride;
    for (uint point = 0u; point < dispatch.pointCount; ++point) {
        device const MRArticulatedPointImpulseGPU& query =
            points[base + point];
        if (query.bodyIndex < articulation.firstBody ||
            query.bodyIndex >=
                articulation.firstBody + articulation.bodyCount ||
            query.flags != 0u ||
            query.reserved0 != 0u ||
            query.reserved1 != 0u ||
            !finite4(query.localPoint) ||
            !finite4(query.worldImpulse) ||
            query.localPoint.w != 0.0f ||
            query.worldImpulse.w != 0.0f) {
            setFailure(
                status,
                MR_ARTICULATED_OPERATOR_NONFINITE_INPUT,
                point
            );
            return false;
        }
    }
    return true;
}

} // namespace

// Generic batched articulated operator. One threadgroup owns one environment.
// Model/kinematic validation, Cholesky, and publication remain lane-zero
// ordered so their status and transactional semantics stay exact. Dense mass
// assembly is independent per symmetric matrix entry and is distributed over
// the group; every entry retains the same deterministic body-order reduction.
kernel void mr_articulated_operator(
    device const MRWorldGPU* worlds [[buffer(0)]],
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const MRJointDescriptorGPU* joints [[buffer(2)]],
    device const MRDofPropertiesGPU* dofs [[buffer(3)]],
    device const MRBodyPropertiesGPU* bodies [[buffer(4)]],
    device const MRArticulatedOperatorDispatchGPU& dispatch [[buffer(5)]],
    device const float* q [[buffer(6)]],
    device const MRArticulatedPointImpulseGPU* points [[buffer(7)]],
    device MRArticulatedBodyPoseGPU* bodyPoses [[buffer(8)]],
    device MRArticulatedPointWorldGPU* pointWorld [[buffer(9)]],
    device float* diagnosticMassMatrix [[buffer(10)]],
    device float* pointJacobians [[buffer(11)]],
    device float* generalizedImpulse [[buffer(12)]],
    device float* deltaVelocity [[buffer(13)]],
    device MRArticulatedOperatorStatusGPU* statuses [[buffer(14)]],
    uint environment [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint threadsPerThreadgroup [[threads_per_threadgroup]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }

    threadgroup uint initializationSucceeded;
    threadgroup float3 bodyPosition[
        MR_ARTICULATED_OPERATOR_MAX_BODIES
    ];
    threadgroup float4 bodyRotation[
        MR_ARTICULATED_OPERATOR_MAX_BODIES
    ];
    threadgroup float3 jointPosition[
        MR_ARTICULATED_OPERATOR_MAX_BODIES
    ];
    threadgroup float3 jointAxis[
        MR_ARTICULATED_OPERATOR_MAX_BODIES
    ];
    threadgroup uint inboundJoint[
        MR_ARTICULATED_OPERATOR_MAX_BODIES
    ];
    threadgroup uint parentLocal[
        MR_ARTICULATED_OPERATOR_MAX_BODIES
    ];
    threadgroup uchar known[
        MR_ARTICULATED_OPERATOR_MAX_BODIES
    ];
    threadgroup float factor[
        MR_ARTICULATED_OPERATOR_MAX_DOFS *
        MR_ARTICULATED_OPERATOR_MAX_DOFS
    ];
    threadgroup float rightHandSide[
        MR_ARTICULATED_OPERATOR_MAX_DOFS
    ];
    threadgroup float intermediate[
        MR_ARTICULATED_OPERATOR_MAX_DOFS
    ];
    threadgroup float solution[
        MR_ARTICULATED_OPERATOR_MAX_DOFS
    ];

    MRArticulatedOperatorStatusGPU status = {};
    status.code = MR_ARTICULATED_OPERATOR_SUCCESS;
    status.environment = environment;
    status.articulationIndex = dispatch.articulationIndex;
    status.failingIndex = MR_INVALID_INDEX;

    if (lane == 0u) {
        initializationSucceeded = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const MRWorldGPU& world = worlds[0];
    if (lane == 0u) {
        initializationSucceeded =
            validDispatch(world, dispatch, status) ? 1u : 0u;
        if (initializationSucceeded == 0u) {
            statuses[environment] = status;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (initializationSucceeded == 0u) {
        return;
    }

    device const MRArticulationGPU& articulation =
        articulations[dispatch.articulationIndex];
    device const float* environmentQ =
        q + environment * dispatch.qStride;
    if (lane == 0u) {
        status.bodyCount = articulation.bodyCount;
        status.nq = articulation.nq;
        status.nv = articulation.nv;
        status.pointCount = dispatch.pointCount;
        initializationSucceeded =
            validModelAndLayout(
                world,
                articulation,
                joints,
                dofs,
                bodies,
                dispatch,
                inboundJoint,
                parentLocal,
                known,
                status
            ) &&
            buildKinematics(
                articulation,
                joints,
                environmentQ,
                bodyPosition,
                bodyRotation,
                jointPosition,
                jointAxis,
                known,
                status
            ) &&
            validatePoints(
                environment,
                articulation,
                dispatch,
                points,
                status
            )
            ? 1u
            : 0u;
        if (initializationSucceeded == 0u) {
            statuses[environment] = status;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (initializationSucceeded == 0u) {
        return;
    }

    // A lower-triangular entry owns both symmetric destinations, so there are
    // no write races. The reduction inside massElement intentionally remains
    // serial and body ordered, preserving the previous bit pattern.
    const uint packedEntryCount =
        articulation.nv * (articulation.nv + 1u) / 2u;
    for (uint packedEntry = lane;
         packedEntry < packedEntryCount;
         packedEntry += threadsPerThreadgroup) {
        uint row = 0u;
        uint rowStart = 0u;
        while (packedEntry >= rowStart + row + 1u) {
            rowStart += row + 1u;
            ++row;
        }
        const uint column = packedEntry - rowStart;
        const float value = massElement(
            row,
            column,
            articulation,
            joints,
            bodies,
            bodyPosition,
            bodyRotation,
            jointPosition,
            jointAxis,
            inboundJoint,
            parentLocal
        );
        factor[
            row * MR_ARTICULATED_OPERATOR_MAX_DOFS + column
        ] = value;
        factor[
            column * MR_ARTICULATED_OPERATOR_MAX_DOFS + row
        ] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane != 0u) {
        return;
    }

    // Lane zero performs the deterministic validation/max reduction before
    // mutating the assembled matrix into its Cholesky factor.
    float maximumMass = 0.0f;
    for (uint row = 0u; row < articulation.nv; ++row) {
        for (uint column = 0u; column <= row; ++column) {
            const float value = factor[
                row * MR_ARTICULATED_OPERATOR_MAX_DOFS + column
            ];
            if (!isfinite(value)) {
                setFailure(
                    status,
                    MR_ARTICULATED_OPERATOR_NONFINITE_RESULT,
                    row
                );
                statuses[environment] = status;
                return;
            }
            maximumMass = max(maximumMass, abs(value));
        }
    }
    // Armature is generalized-coordinate inertia, independent of whether a
    // drive is enabled. Add it before factorization so diagnostic M and the
    // impulse response use exactly the same physical operator.
    for (uint dof = 0u; dof < articulation.nv; ++dof) {
        const float armature =
            dofs[articulation.vOffset + dof].drive.z;
        const uint diagonal =
            dof * MR_ARTICULATED_OPERATOR_MAX_DOFS + dof;
        const float value = factor[diagonal] + armature;
        if (!isfinite(value)) {
            setFailure(
                status,
                MR_ARTICULATED_OPERATOR_NONFINITE_RESULT,
                articulation.vOffset + dof
            );
            statuses[environment] = status;
            return;
        }
        factor[diagonal] = value;
        maximumMass = max(maximumMass, abs(value));
    }

    float minimumPivot = INFINITY;
    float maximumPivot = 0.0f;
    // A pivot is rejected relative to the matrix magnitude, dimension, and
    // FP32 resolution, with an absolute floor only for the all-small regime.
    // This reports ill-conditioned input instead of silently regularizing M.
    const float pivotFloor = max(
        kFactorAbsolutePivotFloor,
        maximumMass * float(articulation.nv) * kFloatEpsilon
    );
    for (uint row = 0u; row < articulation.nv; ++row) {
        for (uint column = 0u;
             column <= row;
             ++column) {
            float value = factor[
                row * MR_ARTICULATED_OPERATOR_MAX_DOFS + column
            ];
            for (uint inner = 0u; inner < column; ++inner) {
                value -=
                    factor[
                        row * MR_ARTICULATED_OPERATOR_MAX_DOFS +
                        inner
                    ] *
                    factor[
                        column * MR_ARTICULATED_OPERATOR_MAX_DOFS +
                        inner
                    ];
            }
            if (row == column) {
                if (!(value > pivotFloor) ||
                    !isfinite(value)) {
                    setFailure(
                        status,
                        MR_ARTICULATED_OPERATOR_FACTORIZATION_FAILED,
                        row
                    );
                    status.diagnostics = float4(
                        minimumPivot,
                        maximumPivot,
                        0.0f,
                        maximumMass
                    );
                    statuses[environment] = status;
                    return;
                }
                const float pivot = sqrt(value);
                factor[
                    row * MR_ARTICULATED_OPERATOR_MAX_DOFS + row
                ] = pivot;
                minimumPivot = min(minimumPivot, pivot);
                maximumPivot = max(maximumPivot, pivot);
            } else {
                value /= factor[
                    column * MR_ARTICULATED_OPERATOR_MAX_DOFS +
                    column
                ];
                if (!isfinite(value)) {
                    setFailure(
                        status,
                        MR_ARTICULATED_OPERATOR_FACTORIZATION_FAILED,
                        row
                    );
                    statuses[environment] = status;
                    return;
                }
                factor[
                    row * MR_ARTICULATED_OPERATOR_MAX_DOFS +
                    column
                ] = value;
            }
        }
    }

    for (uint dof = 0u; dof < articulation.nv; ++dof) {
        rightHandSide[dof] = 0.0f;
    }
    const uint pointBase = environment * dispatch.pointStride;
    for (uint point = 0u;
         point < dispatch.pointCount;
         ++point) {
        device const MRArticulatedPointImpulseGPU& query =
            points[pointBase + point];
        const uint localBody =
            query.bodyIndex - articulation.firstBody;
        const float3 pointOffset = quaternionRotate(
            bodyRotation[localBody],
            query.localPoint.xyz
        );
        for (uint dof = 0u; dof < articulation.nv; ++dof) {
            const MotionColumn bodyMotion = bodyMotionForDof(
                localBody,
                dof,
                articulation,
                joints,
                bodyPosition,
                jointPosition,
                jointAxis,
                inboundJoint,
                parentLocal
            );
            const float3 pointLinear =
                bodyMotion.linear +
                cross(bodyMotion.angular, pointOffset);
            rightHandSide[dof] +=
                dot(pointLinear, query.worldImpulse.xyz);
        }
    }

    for (uint row = 0u; row < articulation.nv; ++row) {
        float value = rightHandSide[row];
        for (uint column = 0u; column < row; ++column) {
            value -=
                factor[
                    row * MR_ARTICULATED_OPERATOR_MAX_DOFS +
                    column
                ] *
                intermediate[column];
        }
        intermediate[row] =
            value /
            factor[
                row * MR_ARTICULATED_OPERATOR_MAX_DOFS + row
            ];
    }
    for (uint reverse = 0u;
         reverse < articulation.nv;
         ++reverse) {
        const uint row = articulation.nv - 1u - reverse;
        float value = intermediate[row];
        for (uint column = row + 1u;
             column < articulation.nv;
             ++column) {
            value -=
                factor[
                    column * MR_ARTICULATED_OPERATOR_MAX_DOFS +
                    row
                ] *
                solution[column];
        }
        solution[row] =
            value /
            factor[
                row * MR_ARTICULATED_OPERATOR_MAX_DOFS + row
            ];
        if (!isfinite(solution[row])) {
            setFailure(
                status,
                MR_ARTICULATED_OPERATOR_NONFINITE_RESULT,
                row
            );
            statuses[environment] = status;
            return;
        }
    }

    // Reuse the checked Cholesky factor for the residual. This is both the
    // production operation being audited and substantially cheaper than
    // reassembling dense body Jacobians after the solve.
    for (uint row = 0u; row < articulation.nv; ++row) {
        float value = 0.0f;
        for (uint column = row;
             column < articulation.nv;
             ++column) {
            value +=
                factor[
                    column * MR_ARTICULATED_OPERATOR_MAX_DOFS + row
                ] *
                solution[column];
        }
        intermediate[row] = value;
    }
    float maximumResidual = 0.0f;
    float maximumRightHandSide = 0.0f;
    float maximumSolution = 0.0f;
    for (uint row = 0u; row < articulation.nv; ++row) {
        float action = 0.0f;
        for (uint column = 0u;
             column <= row;
             ++column) {
            action +=
                factor[
                    row * MR_ARTICULATED_OPERATOR_MAX_DOFS + column
                ] *
                intermediate[column];
        }
        maximumResidual = max(
            maximumResidual,
            abs(action - rightHandSide[row])
        );
        maximumRightHandSide = max(
            maximumRightHandSide,
            abs(rightHandSide[row])
        );
        maximumSolution = max(
            maximumSolution,
            abs(solution[row])
        );
    }
    const float residualDenominator =
        maximumRightHandSide +
        maximumMass * float(articulation.nv) * maximumSolution +
        1.0e-30f;
    const float relativeResidual =
        maximumResidual / residualDenominator;
    if (!isfinite(relativeResidual) ||
        relativeResidual >
            MR_ARTICULATED_OPERATOR_MAX_RELATIVE_RESIDUAL) {
        setFailure(
            status,
            isfinite(relativeResidual)
                ? MR_ARTICULATED_OPERATOR_ACCURACY_FAILED
                : MR_ARTICULATED_OPERATOR_NONFINITE_RESULT,
            MR_INVALID_INDEX
        );
        status.diagnostics = float4(
            minimumPivot,
            maximumPivot,
            relativeResidual,
            maximumMass
        );
        statuses[environment] = status;
        return;
    }

    // Audit every value that publication will materialize. These expressions
    // are intentionally evaluated again during commit: lane-zero validation
    // keeps the payload transactional without a second full output scratch
    // allocation.
    for (uint localBody = 0u;
         localBody < articulation.bodyCount;
         ++localBody) {
        if (!finite3(bodyPosition[localBody]) ||
            !finite4(bodyRotation[localBody])) {
            setFailure(
                status,
                MR_ARTICULATED_OPERATOR_NONFINITE_RESULT,
                articulation.firstBody + localBody
            );
            statuses[environment] = status;
            return;
        }
    }
    for (uint point = 0u;
         point < dispatch.pointCount;
         ++point) {
        device const MRArticulatedPointImpulseGPU& query =
            points[pointBase + point];
        const uint localBody =
            query.bodyIndex - articulation.firstBody;
        const float3 pointOffset = quaternionRotate(
            bodyRotation[localBody],
            query.localPoint.xyz
        );
        const float3 candidateWorld =
            bodyPosition[localBody] + pointOffset;
        if (!finite3(pointOffset) || !finite3(candidateWorld)) {
            setFailure(
                status,
                MR_ARTICULATED_OPERATOR_NONFINITE_RESULT,
                point
            );
            statuses[environment] = status;
            return;
        }
        for (uint dof = 0u; dof < articulation.nv; ++dof) {
            const MotionColumn bodyMotion = bodyMotionForDof(
                localBody,
                dof,
                articulation,
                joints,
                bodyPosition,
                jointPosition,
                jointAxis,
                inboundJoint,
                parentLocal
            );
            const float3 pointLinear =
                bodyMotion.linear +
                cross(bodyMotion.angular, pointOffset);
            if (!finite3(pointLinear)) {
                setFailure(
                    status,
                    MR_ARTICULATED_OPERATOR_NONFINITE_RESULT,
                    point
                );
                statuses[environment] = status;
                return;
            }
        }
    }
    for (uint dof = 0u; dof < articulation.nv; ++dof) {
        if (!isfinite(rightHandSide[dof]) ||
            !isfinite(solution[dof])) {
            setFailure(
                status,
                MR_ARTICULATED_OPERATOR_NONFINITE_RESULT,
                dof
            );
            statuses[environment] = status;
            return;
        }
    }
    if ((dispatch.flags &
         MR_ARTICULATED_OPERATOR_WRITE_DIAGNOSTIC_MASS) != 0u) {
        for (uint row = 0u; row < articulation.nv; ++row) {
            for (uint column = 0u;
                 column < articulation.nv;
                 ++column) {
                float value = 0.0f;
                const uint innerCount = min(row, column) + 1u;
                for (uint inner = 0u;
                     inner < innerCount;
                     ++inner) {
                    value +=
                        factor[
                            row *
                                MR_ARTICULATED_OPERATOR_MAX_DOFS +
                            inner
                        ] *
                        factor[
                            column *
                                MR_ARTICULATED_OPERATOR_MAX_DOFS +
                            inner
                        ];
                }
                if (!isfinite(value)) {
                    setFailure(
                        status,
                        MR_ARTICULATED_OPERATOR_NONFINITE_RESULT,
                        row
                    );
                    statuses[environment] = status;
                    return;
                }
            }
        }
    }

    // Publish only after validation, kinematics, factorization, solve, and
    // residual evaluation all succeed. Error paths above leave payloads
    // untouched and write only the per-environment status.
    const uint poseBase =
        environment * dispatch.bodyPoseStride;
    for (uint localBody = 0u;
         localBody < articulation.bodyCount;
         ++localBody) {
        MRArticulatedBodyPoseGPU pose;
        pose.position =
            float4(bodyPosition[localBody], 1.0f);
        pose.orientation = bodyRotation[localBody];
        bodyPoses[poseBase + localBody] = pose;
    }

    const uint pointWorldBase =
        environment * dispatch.pointWorldStride;
    const uint jacobianBase =
        environment * dispatch.pointJacobianStride;
    for (uint point = 0u;
         point < dispatch.pointCount;
         ++point) {
        device const MRArticulatedPointImpulseGPU& query =
            points[pointBase + point];
        const uint localBody =
            query.bodyIndex - articulation.firstBody;
        const float3 pointOffset = quaternionRotate(
            bodyRotation[localBody],
            query.localPoint.xyz
        );
        MRArticulatedPointWorldGPU worldPoint;
        worldPoint.position = float4(
            bodyPosition[localBody] + pointOffset,
            1.0f
        );
        pointWorld[pointWorldBase + point] = worldPoint;
        for (uint dof = 0u; dof < articulation.nv; ++dof) {
            const MotionColumn bodyMotion = bodyMotionForDof(
                localBody,
                dof,
                articulation,
                joints,
                bodyPosition,
                jointPosition,
                jointAxis,
                inboundJoint,
                parentLocal
            );
            const float3 pointLinear =
                bodyMotion.linear +
                cross(bodyMotion.angular, pointOffset);
            pointJacobians[
                jacobianBase +
                (point * 3u + 0u) * articulation.nv + dof
            ] = pointLinear.x;
            pointJacobians[
                jacobianBase +
                (point * 3u + 1u) * articulation.nv + dof
            ] = pointLinear.y;
            pointJacobians[
                jacobianBase +
                (point * 3u + 2u) * articulation.nv + dof
            ] = pointLinear.z;
        }
    }

    const uint generalizedBase =
        environment * dispatch.generalizedStride;
    for (uint dof = 0u; dof < articulation.nv; ++dof) {
        generalizedImpulse[generalizedBase + dof] =
            rightHandSide[dof];
        deltaVelocity[generalizedBase + dof] =
            solution[dof];
    }

    if ((dispatch.flags &
         MR_ARTICULATED_OPERATOR_WRITE_DIAGNOSTIC_MASS) != 0u) {
        const uint massBase =
            environment * dispatch.massMatrixStride;
        for (uint row = 0u; row < articulation.nv; ++row) {
            for (uint column = 0u;
                 column < articulation.nv;
                 ++column) {
                float value = 0.0f;
                const uint innerCount = min(row, column) + 1u;
                for (uint inner = 0u;
                     inner < innerCount;
                     ++inner) {
                    value +=
                        factor[
                            row *
                                MR_ARTICULATED_OPERATOR_MAX_DOFS +
                            inner
                        ] *
                        factor[
                            column *
                                MR_ARTICULATED_OPERATOR_MAX_DOFS +
                            inner
                        ];
                }
                diagnosticMassMatrix[
                    massBase + row * articulation.nv + column
                ] = value;
            }
        }
    }

    status.diagnostics = float4(
        minimumPivot,
        maximumPivot,
        relativeResidual,
        maximumMass
    );
    statuses[environment] = status;
}
