#include <metal_stdlib>

#include "metalrobo/engine_types.h"
#include "metalrobo/parallel_aba_shared.h"

using namespace metal;

namespace {

#ifndef MR_ABA_COMPILED_MAX_BODIES
#define MR_ABA_COMPILED_MAX_BODIES \
    MR_ARTICULATED_ABA_MAX_BODIES
#endif
#ifndef MR_ABA_COMPILED_MAX_DOFS
#define MR_ABA_COMPILED_MAX_DOFS MR_ARTICULATED_ABA_MAX_DOFS
#endif
#ifndef MR_ABA_COMPILED_MAX_Q
#define MR_ABA_COMPILED_MAX_Q MR_ARTICULATED_ABA_MAX_Q
#endif
#ifndef MR_ABA_KERNEL_NAME
#define MR_ABA_KERNEL_NAME mr_articulated_aba_step
#endif
#ifndef MR_ABA_BODY_PARAMETERS
#define MR_ABA_BODY_PARAMETERS 0
#endif

constant float kQuaternionTolerance = 2.0e-5f;
constant float kQuaternionMinimum = 1.0e-12f;
constant float kFloatEpsilon = 1.1920928955078125e-7f;
constant float kFloatMaximum = 3.402823466e38f;
constant float kTwoPi = 6.28318530717958647692f;
constant float kAbsolutePivotFloor = 1.0e-12f;
constant uint kABAMaxBodies = MR_ABA_COMPILED_MAX_BODIES;
constant uint kABAMaxDofs = MR_ABA_COMPILED_MAX_DOFS;
constant uint kABAMaxQ = MR_ABA_COMPILED_MAX_Q;
constant uint kABASymmetricInertiaEntries = 21u;

struct MRABAScratch {
    float3 bodyPosition[MR_ABA_COMPILED_MAX_BODIES];
    float4 bodyRotation[MR_ABA_COMPILED_MAX_BODIES];
    float3 angularVelocity[MR_ABA_COMPILED_MAX_BODIES];
    float3 linearVelocity[MR_ABA_COMPILED_MAX_BODIES];
    float3 motionAngular[MR_ABA_COMPILED_MAX_BODIES];
    float3 motionLinear[MR_ABA_COMPILED_MAX_BODIES];
    float3 biasAngular[MR_ABA_COMPILED_MAX_BODIES];
    float3 biasLinear[MR_ABA_COMPILED_MAX_BODIES];
    float3 parentToBody[MR_ABA_COMPILED_MAX_BODIES];
    float articulatedInertia[
        MR_ABA_COMPILED_MAX_BODIES * kABASymmetricInertiaEntries
    ];
    float articulatedBias[MR_ABA_COMPILED_MAX_BODIES * 6u];
    float projectedInertia[MR_ABA_COMPILED_MAX_BODIES * 6u];
    float jointDenominator[MR_ABA_COMPILED_MAX_BODIES];
    float jointResidual[MR_ABA_COMPILED_MAX_BODIES];
    uint inboundJoint[MR_ABA_COMPILED_MAX_BODIES];
    uint parentLocal[MR_ABA_COMPILED_MAX_BODIES];
    uint traversal[MR_ABA_COMPILED_MAX_BODIES];
    uchar known[MR_ABA_COMPILED_MAX_BODIES];
    float candidateAcceleration[MR_ABA_COMPILED_MAX_DOFS];
    float candidateNextV[MR_ABA_COMPILED_MAX_DOFS];
    float candidateNextQ[MR_ABA_COMPILED_MAX_Q];
    float rootFactor[36u];
    float rootIntermediate[6u];
};

} // namespace

namespace {

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

inline uint symmetric6Index(const uint row, const uint column) {
    const uint major = max(row, column);
    const uint minor = min(row, column);
    return major * (major + 1u) / 2u + minor;
}

inline float symmetric6Value(
    threadgroup const float* matrix,
    const uint matrixBase,
    const uint row,
    const uint column
) {
    return matrix[
        matrixBase + symmetric6Index(row, column)
    ];
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

inline float3 multiplyWorldInertia(
    device const MRBodyPropertiesGPU& body,
    const float4 orientation,
    const float3 value
) {
    const float3 bodyValue = quaternionRotate(
        quaternionConjugate(orientation),
        value
    );
    const float3 bodyResult = float3(
        dot(body.inertiaRow0.xyz, bodyValue),
        dot(body.inertiaRow1.xyz, bodyValue),
        dot(body.inertiaRow2.xyz, bodyValue)
    );
    return quaternionRotate(orientation, bodyResult);
}

inline void setFailure(
    thread MRABAStatusGPU& status,
    const uint code,
    const uint failingIndex
) {
    status.code = code;
    status.failingIndex = failingIndex;
}

inline float effectiveArmature(
    device const MRDofPropertiesGPU& dof,
    const uint dispatchFlags,
    const float timestep,
    const float gainScale,
    const float dampingScale
) {
    float value = dof.drive.z;
    if ((dispatchFlags & MR_ABA_IMPLICIT_DRIVES) != 0u &&
        (dof.flags & MR_DOF_FLAG_DRIVE) != 0u) {
        value +=
            timestep * dampingScale * dof.drive.y +
            timestep * timestep * gainScale * dof.drive.x;
    }
    return value;
}

inline bool validDispatch(
    device const MRWorldGPU& world,
    device const MRABADispatchGPU& dispatch,
    thread MRABAStatusGPU& status
) {
    constexpr uint knownFlags =
        MR_ABA_HAS_BODY_WRENCHES |
        MR_ABA_APPLY_BODY_DAMPING |
        MR_ABA_IMPLICIT_DRIVES;
    if (world.abiVersion != MR_ENGINE_ABI_VERSION ||
        dispatch.articulationIndex >= world.articulationCount ||
        dispatch.environmentCount == 0u ||
        (dispatch.flags & ~knownFlags) != 0u ||
        dispatch.reserved0 != 0u ||
        dispatch.reserved1 != 0u ||
        !(world.gravityAndTimestep.w > 0.0f) ||
        !finite4(world.gravityAndTimestep)) {
        setFailure(
            status,
            MR_ABA_INVALID_DISPATCH,
            MR_INVALID_INDEX
        );
        return false;
    }
    return true;
}

} // namespace

// O(n) articulated-body forward dynamics and symplectic state update.
// One 32-lane threadgroup owns one environment. The initial executable spine
// is lane-zero ordered: tree traversal and sibling accumulation therefore
// have a single deterministic owner and require no atomics. The ABI and
// scratch layout permit level-parallel traversal without changing results.
kernel void MR_ABA_KERNEL_NAME(
    device const MRWorldGPU* worlds [[buffer(0)]],
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const MRJointDescriptorGPU* joints [[buffer(2)]],
    device const MRDofPropertiesGPU* dofs [[buffer(3)]],
    device const MRBodyPropertiesGPU* bodies [[buffer(4)]],
#ifdef MR_ABA_MULTI_ARTICULATION
    device const MRMultiABADispatchGPU* dispatchInputs [[buffer(5)]],
#else
    device const MRABADispatchGPU& dispatchInput [[buffer(5)]],
#endif
    device const float* qInput [[buffer(6)]],
    device const float* vInput [[buffer(7)]],
    device const float* effortInput [[buffer(8)]],
    device const MRABABodyWrenchGPU* bodyWrenchInput [[buffer(9)]],
    device float* accelerationOutputBase [[buffer(10)]],
    device float* nextVOutputBase [[buffer(11)]],
    device float* nextQOutputBase [[buffer(12)]],
    device MRABAStatusGPU* statusOutput [[buffer(13)]],
#if MR_ABA_BODY_PARAMETERS
    device const float4* bodyParameters [[buffer(14)]],
    device const float4* controllerParameters [[buffer(15)]],
#endif
#ifdef MR_ABA_MULTI_ARTICULATION
    uint2 workPosition [[threadgroup_position_in_grid]],
#else
    uint environment [[threadgroup_position_in_grid]],
#endif
    uint lane [[thread_index_in_threadgroup]]
) {
#ifdef MR_ABA_MULTI_ARTICULATION
    const uint environment = workPosition.x;
    const uint articulationWork = workPosition.y;
    device const MRMultiABADispatchGPU& multiDispatch =
        dispatchInputs[articulationWork];
    device const MRABADispatchGPU& dispatch =
        multiDispatch.dispatch;
    device const float* q = qInput + multiDispatch.qBase;
    device const float* v = vInput + multiDispatch.vBase;
    device const float* effort =
        effortInput + multiDispatch.effortBase;
    device const MRABABodyWrenchGPU* bodyWrenches =
        bodyWrenchInput + multiDispatch.wrenchBase;
    device float* accelerationOutput =
        accelerationOutputBase + multiDispatch.accelerationBase;
    device float* nextVOutput =
        nextVOutputBase + multiDispatch.nextVBase;
    device float* nextQOutput =
        nextQOutputBase + multiDispatch.nextQBase;
    device MRABAStatusGPU* statuses =
        statusOutput + multiDispatch.statusBase;
#else
    device const MRABADispatchGPU& dispatch = dispatchInput;
    device const float* q = qInput;
    device const float* v = vInput;
    device const float* effort = effortInput;
    device const MRABABodyWrenchGPU* bodyWrenches =
        bodyWrenchInput;
    device float* accelerationOutput = accelerationOutputBase;
    device float* nextVOutput = nextVOutputBase;
    device float* nextQOutput = nextQOutputBase;
    device MRABAStatusGPU* statuses = statusOutput;
#endif
    if (lane != 0u || environment >= dispatch.environmentCount) {
        return;
    }

    threadgroup MRABAScratch scratch;
    threadgroup MRABAScratch& localScratch = scratch;
    threadgroup float3* bodyPosition = localScratch.bodyPosition;
    threadgroup float4* bodyRotation = localScratch.bodyRotation;
    threadgroup float3* angularVelocity =
        localScratch.angularVelocity;
    threadgroup float3* linearVelocity = localScratch.linearVelocity;
    threadgroup float3* motionAngular = localScratch.motionAngular;
    threadgroup float3* motionLinear = localScratch.motionLinear;
    threadgroup float3* biasAngular = localScratch.biasAngular;
    threadgroup float3* biasLinear = localScratch.biasLinear;
    threadgroup float3* parentToBody = localScratch.parentToBody;
    threadgroup float* articulatedInertia =
        localScratch.articulatedInertia;
    threadgroup float* articulatedBias = localScratch.articulatedBias;
    threadgroup float* projectedInertia =
        localScratch.projectedInertia;
    threadgroup float* jointDenominator =
        localScratch.jointDenominator;
    threadgroup float* jointResidual = localScratch.jointResidual;
    // Position and angular-velocity storage is dead after body inertia and
    // bias initialization. Reuse it for the later acceleration sweep so the
    // full-capacity kernel remains below the next Apple GPU residency tier.
    threadgroup float3* bodyAccelerationAngular = bodyPosition;
    threadgroup float3* bodyAccelerationLinear = angularVelocity;
    threadgroup uint* inboundJoint = localScratch.inboundJoint;
    threadgroup uint* parentLocal = localScratch.parentLocal;
    threadgroup uint* traversal = localScratch.traversal;
    threadgroup uchar* known = localScratch.known;
    threadgroup float* candidateAcceleration =
        localScratch.candidateAcceleration;
    threadgroup float* candidateNextV = localScratch.candidateNextV;
    threadgroup float* candidateNextQ = localScratch.candidateNextQ;
    threadgroup float* rootFactor = localScratch.rootFactor;
    threadgroup float* rootIntermediate =
        localScratch.rootIntermediate;

    MRABAStatusGPU status = {};
    status.code = MR_ABA_SUCCESS;
    status.environment = environment;
    status.articulationIndex = dispatch.articulationIndex;
    status.failingIndex = MR_INVALID_INDEX;
    status.flags = dispatch.flags;

    device const MRWorldGPU& world = worlds[0];
    if (!validDispatch(world, dispatch, status)) {
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
        articulation.bodyCount > kABAMaxBodies ||
        articulation.nv == 0u ||
        articulation.nv > kABAMaxDofs ||
        articulation.nq > kABAMaxQ ||
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
        dispatch.vStride < articulation.nv ||
        dispatch.effortStride < articulation.nv ||
        dispatch.accelerationStride < articulation.nv ||
        dispatch.nextVStride < articulation.nv ||
        dispatch.nextQStride < articulation.nq ||
        ((dispatch.flags & MR_ABA_HAS_BODY_WRENCHES) != 0u &&
         dispatch.wrenchStride < articulation.bodyCount)) {
        setFailure(
            status,
            articulation.bodyCount >
                    kABAMaxBodies ||
                articulation.nv > kABAMaxDofs ||
                articulation.nq > kABAMaxQ
                ? MR_ABA_UNSUPPORTED_TOPOLOGY
                : MR_ABA_INVALID_MODEL,
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
        jointResidual[localBody] = 0.0f;
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
            setFailure(status, MR_ABA_INVALID_MODEL, globalJoint);
            statuses[environment] = status;
            return;
        }
        const bool scalarJoint =
            joint.jointType == MR_JOINT_REVOLUTE ||
            joint.jointType == MR_JOINT_CONTINUOUS ||
            joint.jointType == MR_JOINT_PRISMATIC;
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
                    ? MR_ABA_UNSUPPORTED_TOPOLOGY
                    : MR_ABA_INVALID_MODEL,
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
                setFailure(status, MR_ABA_INVALID_MODEL, globalJoint);
                statuses[environment] = status;
                return;
            }
        }
        const uint localChild =
            joint.childBody - articulation.firstBody;
        if (inboundJoint[localChild] != MR_INVALID_INDEX) {
            setFailure(
                status,
                MR_ABA_UNSUPPORTED_TOPOLOGY,
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
            MR_ABA_INVALID_MODEL,
            MR_INVALID_INDEX
        );
        statuses[environment] = status;
        return;
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
            MR_ABA_UNSUPPORTED_TOPOLOGY,
            MR_INVALID_INDEX
        );
        statuses[environment] = status;
        return;
    }

    const uint qBase = environment * dispatch.qStride;
    const uint vBase = environment * dispatch.vStride;
    const uint effortBase = environment * dispatch.effortStride;
#if MR_ABA_BODY_PARAMETERS
    const float4 controller = controllerParameters[environment];
    const float gainScale = max(controller.x, 0.0f);
    const float driveDampingScale = max(controller.y, 0.0f);
#else
    constexpr float gainScale = 1.0f;
    constexpr float driveDampingScale = 1.0f;
#endif
    device const float* environmentQ = q + qBase;
    device const float* environmentV = v + vBase;
    device const float* environmentEffort = effort + effortBase;
    for (uint localQ = 0u; localQ < articulation.nq; ++localQ) {
        if (!isfinite(environmentQ[localQ])) {
            setFailure(status, MR_ABA_NONFINITE_INPUT, localQ);
            statuses[environment] = status;
            return;
        }
        candidateNextQ[localQ] = environmentQ[localQ];
    }
    for (uint localV = 0u; localV < articulation.nv; ++localV) {
        device const MRDofPropertiesGPU& dof =
            dofs[articulation.vOffset + localV];
        if (!isfinite(environmentV[localV]) ||
            !isfinite(environmentEffort[localV]) ||
            !finite4(dof.drive) ||
            dof.drive.z < 0.0f) {
            setFailure(status, MR_ABA_NONFINITE_INPUT, localV);
            statuses[environment] = status;
            return;
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
            setFailure(status, MR_ABA_INVALID_QUATERNION, 0u);
            statuses[environment] = status;
            return;
        }
        bodyPosition[rootLocal] = float3(
            environmentQ[0],
            environmentQ[1],
            environmentQ[2]
        );
        bodyRotation[rootLocal] = checkedRootRotation;
        linearVelocity[rootLocal] = float3(
            environmentV[0],
            environmentV[1],
            environmentV[2]
        );
        angularVelocity[rootLocal] = float3(
            environmentV[3],
            environmentV[4],
            environmentV[5]
        );
    } else {
        bodyPosition[rootLocal] = float3(0.0f);
        bodyRotation[rootLocal] =
            float4(0.0f, 0.0f, 0.0f, 1.0f);
        linearVelocity[rootLocal] = float3(0.0f);
        angularVelocity[rootLocal] = float3(0.0f);
    }
    parentToBody[rootLocal] = float3(0.0f);
    motionAngular[rootLocal] = float3(0.0f);
    motionLinear[rootLocal] = float3(0.0f);
    biasAngular[rootLocal] = float3(0.0f);
    biasLinear[rootLocal] = float3(0.0f);

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
            setFailure(status, MR_ABA_INVALID_MODEL, globalJoint);
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
        float jointCoordinate = 0.0f;
        float jointRate = 0.0f;
        if (joint.nv == 1u) {
            axisInJoint = normalize(joint.axis0.xyz);
            const uint localQ =
                joint.qOffset - articulation.qOffset;
            const uint localV =
                joint.vOffset - articulation.vOffset;
            jointCoordinate = environmentQ[localQ];
            if (joint.jointType == MR_JOINT_REVOLUTE ||
                joint.jointType == MR_JOINT_CONTINUOUS) {
                motionRotation = axisAngleQuaternion(
                    axisInJoint,
                    jointCoordinate
                );
            }
            jointRate = environmentV[localV];
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
                MR_ABA_NONFINITE_RESULT,
                articulation.firstBody + localChild
            );
            statuses[environment] = status;
            return;
        }
        bodyRotation[localChild] = checkedChildRotation;
        const float3 jointAxis = quaternionRotate(
            parentToJointRotation,
            axisInJoint
        );
        const bool prismatic =
            joint.jointType == MR_JOINT_PRISMATIC;
        const float3 jointPosition =
            bodyPosition[localParent] +
            quaternionRotate(
                bodyRotation[localParent],
                joint.parentAnchor.xyz
            ) +
            (prismatic
                ? jointAxis * jointCoordinate
                : float3(0.0f));
        const float3 childAnchor = quaternionRotate(
            bodyRotation[localChild],
            joint.childAnchor.xyz
        );
        bodyPosition[localChild] =
            jointPosition - childAnchor;
        parentToBody[localChild] =
            bodyPosition[localChild] -
            bodyPosition[localParent];
        const float3 parentToJoint =
            jointPosition - bodyPosition[localParent];
        angularVelocity[localChild] =
            angularVelocity[localParent] +
            (!prismatic && joint.nv == 1u
                ? jointAxis * jointRate
                : float3(0.0f));
        linearVelocity[localChild] =
            linearVelocity[localParent] +
            cross(
                angularVelocity[localParent],
                parentToJoint
            ) +
            (prismatic
                ? jointAxis * jointRate
                : float3(0.0f)) -
            cross(
                angularVelocity[localChild],
                childAnchor
            );
        motionAngular[localChild] =
            joint.nv == 1u && !prismatic
                ? jointAxis
                : float3(0.0f);
        motionLinear[localChild] =
            prismatic
                ? jointAxis
                : (joint.nv == 1u
                    ? -cross(jointAxis, childAnchor)
                    : float3(0.0f));
        biasAngular[localChild] =
            joint.nv == 1u && !prismatic
                ? cross(
                    angularVelocity[localParent],
                    jointAxis
                ) * jointRate
                : float3(0.0f);
        const float3 prismaticCoriolis =
            prismatic
                ? 2.0f * cross(
                    angularVelocity[localParent],
                    jointAxis
                ) * jointRate
                : float3(0.0f);
        biasLinear[localChild] =
            cross(
                angularVelocity[localParent],
                cross(
                    angularVelocity[localParent],
                    parentToJoint
                )
            ) +
            prismaticCoriolis -
            cross(biasAngular[localChild], childAnchor) -
            cross(
                angularVelocity[localChild],
                cross(
                    angularVelocity[localChild],
                    childAnchor
                )
            );
        if (!finite3(bodyPosition[localChild]) ||
            !finite4(bodyRotation[localChild]) ||
            !finite3(angularVelocity[localChild]) ||
            !finite3(linearVelocity[localChild]) ||
            !finite3(motionAngular[localChild]) ||
            !finite3(motionLinear[localChild]) ||
            !finite3(biasAngular[localChild]) ||
            !finite3(biasLinear[localChild])) {
            setFailure(
                status,
                MR_ABA_NONFINITE_RESULT,
                articulation.firstBody + localChild
            );
            statuses[environment] = status;
            return;
        }
    }

    const uint wrenchBase = environment * dispatch.wrenchStride;
    for (uint localBody = 0u;
         localBody < articulation.bodyCount;
         ++localBody) {
        const uint globalBody =
            articulation.firstBody + localBody;
        device const MRBodyPropertiesGPU& body =
            bodies[globalBody];
#if MR_ABA_BODY_PARAMETERS
        const float4 physical = bodyParameters[
            environment * world.bodyCount + globalBody
        ];
        const float massScale = max(physical.x, 1.0e-4f);
        // The body-parameter ABI is mass, friction, restitution, damping.
        // Inertia scales with mass, matching free-body materialization;
        // physical.z remains contact-only.
        const float inertiaScale = massScale;
        const float dampingScale = max(physical.w, 0.0f);
#else
        constexpr float massScale = 1.0f;
        constexpr float inertiaScale = 1.0f;
        constexpr float dampingScale = 1.0f;
#endif
        if (body.articulationIndex !=
                dispatch.articulationIndex ||
            body.motionType != MR_MOTION_DYNAMIC ||
            !(body.massAndInverseMass.x > 0.0f) ||
            !finite4(body.massAndInverseMass) ||
            !finite4(body.inertiaRow0) ||
            !finite4(body.inertiaRow1) ||
            !finite4(body.inertiaRow2) ||
            !finite4(body.dampingAndSpeedLimits)) {
            setFailure(status, MR_ABA_INVALID_MODEL, globalBody);
            statuses[environment] = status;
            return;
        }
        const uint matrixBase =
            localBody * kABASymmetricInertiaEntries;
        for (uint entry = 0u;
             entry < kABASymmetricInertiaEntries;
             ++entry) {
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
            for (uint column = 0u; column <= row; ++column) {
                articulatedInertia[
                    matrixBase + symmetric6Index(row, column)
                ] = inertiaScale * worldInertiaElement(
                    body,
                    rotationColumn0,
                    rotationColumn1,
                    rotationColumn2,
                    row,
                    column
                );
            }
            articulatedInertia[
                matrixBase + symmetric6Index(
                    3u + row,
                    3u + row
                )
            ] = massScale * body.massAndInverseMass.x;
        }
        float3 externalForce =
            massScale * body.massAndInverseMass.x *
            world.gravityAndTimestep.xyz;
        float3 externalTorque = float3(0.0f);
        if ((dispatch.flags & MR_ABA_HAS_BODY_WRENCHES) != 0u) {
            device const MRABABodyWrenchGPU& wrench =
                bodyWrenches[wrenchBase + localBody];
            if (!finite4(wrench.force) ||
                !finite4(wrench.torque) ||
                wrench.force.w != 0.0f ||
                wrench.torque.w != 0.0f) {
                setFailure(status, MR_ABA_NONFINITE_INPUT, globalBody);
                statuses[environment] = status;
                return;
            }
            externalForce += wrench.force.xyz;
            externalTorque += wrench.torque.xyz;
        }
        float3 biasTorque = cross(
            angularVelocity[localBody],
            inertiaScale * multiplyWorldInertia(
                body,
                bodyRotation[localBody],
                angularVelocity[localBody]
            )
        ) - externalTorque;
        float3 biasForce = -externalForce;
        if ((dispatch.flags & MR_ABA_APPLY_BODY_DAMPING) != 0u) {
            biasTorque +=
                dampingScale * body.dampingAndSpeedLimits.y *
                angularVelocity[localBody];
            biasForce +=
                dampingScale * body.dampingAndSpeedLimits.x *
                linearVelocity[localBody];
        }
        for (uint component = 0u; component < 6u; ++component) {
            articulatedBias[localBody * 6u + component] =
                spatialComponent(
                    biasTorque,
                    biasForce,
                    component
                );
        }
    }

    if (articulation.rootType == MR_ROOT_FLOATING) {
        const uint rootMatrixBase =
            rootLocal * kABASymmetricInertiaEntries;
        for (uint axis = 0u; axis < 3u; ++axis) {
            articulatedInertia[
                rootMatrixBase + symmetric6Index(
                    3u + axis,
                    3u + axis
                )
            ] += effectiveArmature(
                dofs[articulation.vOffset + axis],
                dispatch.flags,
                world.gravityAndTimestep.w,
                gainScale,
                driveDampingScale
            );
            articulatedInertia[
                rootMatrixBase + symmetric6Index(axis, axis)
            ] += effectiveArmature(
                dofs[articulation.vOffset + 3u + axis],
                dispatch.flags,
                world.gravityAndTimestep.w,
                gainScale,
                driveDampingScale
            );
            articulatedBias[rootLocal * 6u + axis] -=
                environmentEffort[3u + axis];
            articulatedBias[
                rootLocal * 6u + 3u + axis
            ] -= environmentEffort[axis];
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
        const uint matrixBase =
            localBody * kABASymmetricInertiaEntries;

        float3 inertiaTimesBiasAngular = float3(0.0f);
        float3 inertiaTimesBiasLinear = float3(0.0f);
        for (uint row = 0u; row < 6u; ++row) {
            float value = 0.0f;
            for (uint column = 0u; column < 6u; ++column) {
                value +=
                    symmetric6Value(
                        articulatedInertia,
                        matrixBase,
                        row,
                        column
                    ) *
                    spatialComponent(
                        biasAngular[localBody],
                        biasLinear[localBody],
                        column
                    );
            }
            setSpatialComponent(
                inertiaTimesBiasAngular,
                inertiaTimesBiasLinear,
                row,
                value
            );
        }
        float3 propagatedTorque =
            float3(
                articulatedBias[localBody * 6u + 0u],
                articulatedBias[localBody * 6u + 1u],
                articulatedBias[localBody * 6u + 2u]
            ) + inertiaTimesBiasAngular;
        float3 propagatedForce =
            float3(
                articulatedBias[localBody * 6u + 3u],
                articulatedBias[localBody * 6u + 4u],
                articulatedBias[localBody * 6u + 5u]
            ) + inertiaTimesBiasLinear;

        if (joint.nv == 1u) {
            float3 projectedAngular = float3(0.0f);
            float3 projectedLinear = float3(0.0f);
            for (uint row = 0u; row < 6u; ++row) {
                float value = 0.0f;
                for (uint column = 0u;
                     column < 6u;
                     ++column) {
                    value +=
                        symmetric6Value(
                            articulatedInertia,
                            matrixBase,
                            row,
                            column
                        ) *
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
                effectiveArmature(
                    dofs[articulation.vOffset + localV],
                    dispatch.flags,
                    world.gravityAndTimestep.w,
                    gainScale,
                    driveDampingScale
                );
            float maximumInertia = 0.0f;
            for (uint entry = 0u;
                 entry < kABASymmetricInertiaEntries;
                 ++entry) {
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
                    MR_ABA_FACTORIZATION_FAILED,
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
            const float residual =
                environmentEffort[localV] -
                dot(motionAngular[localBody], propagatedTorque) -
                dot(motionLinear[localBody], propagatedForce);
            jointResidual[localBody] = residual;
            const float pivot = sqrt(denominator);
            minimumPivot = min(minimumPivot, pivot);
            maximumPivot = max(maximumPivot, pivot);
            propagatedTorque +=
                projectedAngular * (residual / denominator);
            propagatedForce +=
                projectedLinear * (residual / denominator);
            for (uint row = 0u; row < 6u; ++row) {
                for (uint column = 0u;
                     column <= row;
                     ++column) {
                    articulatedInertia[
                        matrixBase + symmetric6Index(row, column)
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
        const uint parentMatrixBase =
            localParent * kABASymmetricInertiaEntries;
        for (uint row = 0u; row < 6u; ++row) {
            for (uint column = 0u; column <= row; ++column) {
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
                            symmetric6Value(
                                articulatedInertia,
                                matrixBase,
                                left,
                                right
                            ) *
                            motionTransformElement(
                                offset,
                                right,
                                column
                            );
                    }
                }
                articulatedInertia[
                    parentMatrixBase +
                    symmetric6Index(row, column)
                ] += transformed;
            }
        }
        propagatedTorque += cross(offset, propagatedForce);
        for (uint component = 0u; component < 6u; ++component) {
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
        const uint rootMatrixBase =
            rootLocal * kABASymmetricInertiaEntries;
        float maximumRootInertia = 0.0f;
        for (uint entry = 0u;
             entry < kABASymmetricInertiaEntries;
             ++entry) {
            maximumRootInertia = max(
                maximumRootInertia,
                abs(articulatedInertia[rootMatrixBase + entry])
            );
        }
        for (uint entry = 0u; entry < 36u; ++entry) {
            rootFactor[entry] = 0.0f;
        }
        const float pivotFloor = max(
            kAbsolutePivotFloor,
            maximumRootInertia * 6.0f * kFloatEpsilon
        );
        for (uint row = 0u; row < 6u; ++row) {
            for (uint column = 0u; column <= row; ++column) {
                float value = symmetric6Value(
                    articulatedInertia,
                    rootMatrixBase,
                    row,
                    column
                );
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
                            MR_ABA_FACTORIZATION_FAILED,
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
        for (uint row = 0u; row < 6u; ++row) {
            float value =
                -articulatedBias[rootLocal * 6u + row];
            for (uint column = 0u; column < row; ++column) {
                value -=
                    rootFactor[row * 6u + column] *
                    rootIntermediate[column];
            }
            rootIntermediate[row] =
                value / rootFactor[row * 6u + row];
        }
        float3 rootAngularAcceleration = float3(0.0f);
        float3 rootLinearAcceleration = float3(0.0f);
        for (uint reverse = 0u; reverse < 6u; ++reverse) {
            const uint row = 5u - reverse;
            float value = rootIntermediate[row];
            for (uint column = row + 1u;
                 column < 6u;
                 ++column) {
                value -=
                    rootFactor[column * 6u + row] *
                    spatialComponent(
                        rootAngularAcceleration,
                        rootLinearAcceleration,
                        column
                    );
            }
            setSpatialComponent(
                rootAngularAcceleration,
                rootLinearAcceleration,
                row,
                value / rootFactor[row * 6u + row]
            );
        }
        bodyAccelerationAngular[rootLocal] =
            rootAngularAcceleration;
        bodyAccelerationLinear[rootLocal] =
            rootLinearAcceleration;
        candidateAcceleration[0] = rootLinearAcceleration.x;
        candidateAcceleration[1] = rootLinearAcceleration.y;
        candidateAcceleration[2] = rootLinearAcceleration.z;
        candidateAcceleration[3] = rootAngularAcceleration.x;
        candidateAcceleration[4] = rootAngularAcceleration.y;
        candidateAcceleration[5] = rootAngularAcceleration.z;
    } else {
        bodyAccelerationAngular[rootLocal] = float3(0.0f);
        bodyAccelerationLinear[rootLocal] = float3(0.0f);
    }

    for (uint traversalIndex = 1u;
         traversalIndex < articulation.bodyCount;
         ++traversalIndex) {
        const uint localBody = traversal[traversalIndex];
        const uint localParent = parentLocal[localBody];
        const uint globalJoint = inboundJoint[localBody];
        device const MRJointDescriptorGPU& joint =
            joints[globalJoint];
        const float3 transformedParentAngular =
            bodyAccelerationAngular[localParent];
        const float3 transformedParentLinear =
            bodyAccelerationLinear[localParent] +
            cross(
                bodyAccelerationAngular[localParent],
                parentToBody[localBody]
            );
        float3 angular =
            transformedParentAngular + biasAngular[localBody];
        float3 linear =
            transformedParentLinear + biasLinear[localBody];
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
                        transformedParentAngular,
                        transformedParentLinear,
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
            candidateAcceleration[localV] = jointAcceleration;
            angular +=
                motionAngular[localBody] * jointAcceleration;
            linear +=
                motionLinear[localBody] * jointAcceleration;
        }
        bodyAccelerationAngular[localBody] = angular;
        bodyAccelerationLinear[localBody] = linear;
    }

    float maximumAcceleration = 0.0f;
    const float timestep = world.gravityAndTimestep.w;
    for (uint localV = 0u; localV < articulation.nv; ++localV) {
        const float acceleration = candidateAcceleration[localV];
        const float nextVelocity =
            environmentV[localV] + timestep * acceleration;
        if (!isfinite(acceleration) ||
            !isfinite(nextVelocity)) {
            setFailure(status, MR_ABA_NONFINITE_RESULT, localV);
            statuses[environment] = status;
            return;
        }
        candidateNextV[localV] = nextVelocity;
        maximumAcceleration = max(
            maximumAcceleration,
            abs(acceleration)
        );
    }

    float quaternionNormError = 0.0f;
    if (articulation.rootType == MR_ROOT_FLOATING) {
        candidateNextQ[0] =
            environmentQ[0] + timestep * candidateNextV[0];
        candidateNextQ[1] =
            environmentQ[1] + timestep * candidateNextV[1];
        candidateNextQ[2] =
            environmentQ[2] + timestep * candidateNextV[2];
        const float3 rotationVector =
            timestep * float3(
                candidateNextV[3],
                candidateNextV[4],
                candidateNextV[5]
            );
        const float angle = length(rotationVector);
        const float reducedAngle = angle > kTwoPi
            ? fmod(angle, kTwoPi)
            : angle;
        const float halfAngle = 0.5f * reducedAngle;
        const float scale = angle > 1.0e-6f
            ? sin(halfAngle) / angle
            : 0.5f - angle * angle / 48.0f;
        const float4 increment = float4(
            rotationVector * scale,
            cos(halfAngle)
        );
        float4 updated;
        if (!normalizedQuaternion(
                quaternionMultiply(
                    increment,
                    float4(
                        environmentQ[3],
                        environmentQ[4],
                        environmentQ[5],
                        environmentQ[6]
                    )
                ),
                updated,
                false
            )) {
            status.diagnostics = float4(
                maximumAcceleration,
                isfinite(angle) ? angle : kFloatMaximum,
                max(
                    abs(candidateNextV[3]),
                    max(
                        abs(candidateNextV[4]),
                        abs(candidateNextV[5])
                    )
                ),
                length(
                    float4(
                        environmentQ[3],
                        environmentQ[4],
                        environmentQ[5],
                        environmentQ[6]
                    )
                )
            );
            setFailure(status, MR_ABA_NONFINITE_RESULT, 3u);
            statuses[environment] = status;
            return;
        }
        candidateNextQ[3] = updated.x;
        candidateNextQ[4] = updated.y;
        candidateNextQ[5] = updated.z;
        candidateNextQ[6] = updated.w;
        quaternionNormError =
            abs(length(updated) - 1.0f);
    }
    for (uint localJoint = 0u;
         localJoint < articulation.jointCount;
         ++localJoint) {
        device const MRJointDescriptorGPU& joint =
            joints[articulation.firstJoint + localJoint];
        if (joint.nv == 1u) {
            const uint localQ =
                joint.qOffset - articulation.qOffset;
            const uint localV =
                joint.vOffset - articulation.vOffset;
            candidateNextQ[localQ] =
                environmentQ[localQ] +
                timestep * candidateNextV[localV];
        }
    }
    for (uint localQ = 0u; localQ < articulation.nq; ++localQ) {
        if (!isfinite(candidateNextQ[localQ])) {
            setFailure(status, MR_ABA_NONFINITE_RESULT, localQ);
            statuses[environment] = status;
            return;
        }
    }

    const uint accelerationBase =
        environment * dispatch.accelerationStride;
    const uint nextVBase =
        environment * dispatch.nextVStride;
    const uint nextQBase =
        environment * dispatch.nextQStride;
    for (uint localV = 0u; localV < articulation.nv; ++localV) {
        accelerationOutput[accelerationBase + localV] =
            candidateAcceleration[localV];
        nextVOutput[nextVBase + localV] =
            candidateNextV[localV];
    }
    for (uint localQ = 0u; localQ < articulation.nq; ++localQ) {
        nextQOutput[nextQBase + localQ] =
            candidateNextQ[localQ];
    }
    status.diagnostics = float4(
        isfinite(minimumPivot) ? minimumPivot : 0.0f,
        maximumPivot,
        maximumAcceleration,
        quaternionNormError
    );
    statuses[environment] = status;
}
