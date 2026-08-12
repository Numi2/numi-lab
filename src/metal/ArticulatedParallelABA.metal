#define MR_ABA_EMIT_SERIAL_KERNEL 0
#include "ArticulatedABA.metal"

#ifndef MR_PARALLEL_ABA_KERNEL_NAME
#define MR_PARALLEL_ABA_KERNEL_NAME mr_parallel_articulated_aba_step
#endif

namespace {

struct MRParallelABAAuxScratch {
    float propagatedInertia[
        MR_ABA_COMPILED_MAX_BODIES * kABASymmetricInertiaEntries
    ];
    uint laneFailureCodes[32u];
    uint laneFailureIndices[32u];
    float4 laneFailureDiagnostics[32u];
    uint selectedFailureCode;
    uint selectedFailureIndex;
    float4 selectedFailureDiagnostics;
    float minimumPivot;
    float maximumPivot;
    float maximumAcceleration;
    float quaternionNormError;
};

inline void clearParallelABAFailure(
    threadgroup uint* codes,
    threadgroup uint* indices,
    threadgroup float4* diagnostics,
    const uint lane
) {
    codes[lane] = MR_ABA_SUCCESS;
    indices[lane] = MR_INVALID_INDEX;
    diagnostics[lane] = float4(0.0f);
}

inline bool collectParallelABAFailure(
    threadgroup uint* codes,
    threadgroup uint* indices,
    threadgroup float4* diagnostics,
    threadgroup uint* selectedCode,
    threadgroup uint* selectedIndex,
    threadgroup float4* selectedDiagnostics,
    const uint lane
) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        *selectedCode = MR_ABA_SUCCESS;
        *selectedIndex = MR_INVALID_INDEX;
        *selectedDiagnostics = float4(0.0f);
        for (uint candidate = 0u; candidate < 32u; ++candidate) {
            if (codes[candidate] != MR_ABA_SUCCESS) {
                *selectedCode = codes[candidate];
                *selectedIndex = indices[candidate];
                *selectedDiagnostics = diagnostics[candidate];
                break;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *selectedCode != MR_ABA_SUCCESS;
}

inline void publishParallelABAFailure(
    thread MRABAStatusGPU& status,
    device MRABAStatusGPU* statuses,
    const uint statusIndex,
    const uint code,
    const uint index,
    const float4 diagnostics
) {
    status.code = code;
    status.failingIndex = index;
    status.diagnostics = diagnostics;
    statuses[statusIndex] = status;
}

} // namespace

// One SIMD32 threadgroup owns one articulation/environment packet. Body lanes
// evaluate independent frontiers; parent lanes reduce immutable child
// contributions in cooked order, avoiding floating-point atomics.
kernel void MR_PARALLEL_ABA_KERNEL_NAME(
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
    device const MRParallelABAArticulationGPU*
        scheduleArticulations [[buffer(16)]],
    device const MRParallelABALevelGPU* scheduleLevels [[buffer(17)]],
    device const MRParallelABAParentReductionGPU*
        parentReductions [[buffer(18)]],
    device const uint* levelBodies [[buffer(19)]],
    device const uint* scheduleParentLocal [[buffer(20)]],
    device const uint* scheduleInboundJoint [[buffer(21)]],
    device const uint* childOffsets [[buffer(22)]],
    device const uint* childIndices [[buffer(23)]],
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
    device const MRABADispatchGPU& dispatch = multiDispatch.dispatch;
    device const float* q = qInput + multiDispatch.qBase;
    device const float* v = vInput + multiDispatch.vBase;
    device const float* effort = effortInput + multiDispatch.effortBase;
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
    device const MRABABodyWrenchGPU* bodyWrenches = bodyWrenchInput;
    device float* accelerationOutput = accelerationOutputBase;
    device float* nextVOutput = nextVOutputBase;
    device float* nextQOutput = nextQOutputBase;
    device MRABAStatusGPU* statuses = statusOutput;
#endif

    threadgroup MRABAScratch scratch;
    threadgroup MRParallelABAAuxScratch auxiliary;
    threadgroup float3* bodyPosition = scratch.bodyPosition;
    threadgroup float4* bodyRotation = scratch.bodyRotation;
    threadgroup float3* angularVelocity = scratch.angularVelocity;
    threadgroup float3* linearVelocity = scratch.linearVelocity;
    threadgroup float3* motionAngular = scratch.motionAngular;
    threadgroup float3* motionLinear = scratch.motionLinear;
    threadgroup float3* biasAngular = scratch.biasAngular;
    threadgroup float3* biasLinear = scratch.biasLinear;
    threadgroup float3* parentToBody = scratch.parentToBody;
    threadgroup float* articulatedInertia = scratch.articulatedInertia;
    threadgroup float* propagatedInertia = auxiliary.propagatedInertia;
    threadgroup float* articulatedBias = scratch.articulatedBias;
    threadgroup float* projectedInertia = scratch.projectedInertia;
    threadgroup float* jointDenominator = scratch.jointDenominator;
    threadgroup float* jointResidual = scratch.jointResidual;
    threadgroup float* candidateAcceleration =
        scratch.candidateAcceleration;
    threadgroup float* candidateNextV = scratch.candidateNextV;
    threadgroup float* candidateNextQ = scratch.candidateNextQ;
    threadgroup float* rootFactor = scratch.rootFactor;
    threadgroup float* rootIntermediate = scratch.rootIntermediate;
    threadgroup uint* laneFailureCodes = auxiliary.laneFailureCodes;
    threadgroup uint* laneFailureIndices = auxiliary.laneFailureIndices;
    threadgroup float4* laneFailureDiagnostics =
        auxiliary.laneFailureDiagnostics;

    MRABAStatusGPU status = {};
    status.code = MR_ABA_SUCCESS;
    status.environment = environment;
    status.articulationIndex = dispatch.articulationIndex;
    status.failingIndex = MR_INVALID_INDEX;
    status.flags = dispatch.flags;

    clearParallelABAFailure(
        laneFailureCodes,
        laneFailureIndices,
        laneFailureDiagnostics,
        lane
    );
    if (lane == 0u) {
        MRABAStatusGPU validationStatus = status;
        if (environment >= dispatch.environmentCount ||
            !validDispatch(worlds[0], dispatch, validationStatus)) {
            laneFailureCodes[0] = validationStatus.code == MR_ABA_SUCCESS
                ? MR_ABA_INVALID_DISPATCH
                : validationStatus.code;
            laneFailureIndices[0] = validationStatus.failingIndex;
        }
    }
    if (collectParallelABAFailure(
            laneFailureCodes,
            laneFailureIndices,
            laneFailureDiagnostics,
            &auxiliary.selectedFailureCode,
            &auxiliary.selectedFailureIndex,
            &auxiliary.selectedFailureDiagnostics,
            lane
        )) {
        if (lane == 0u && environment < dispatch.environmentCount) {
            publishParallelABAFailure(
                status,
                statuses,
                environment,
                auxiliary.selectedFailureCode,
                auxiliary.selectedFailureIndex,
                auxiliary.selectedFailureDiagnostics
            );
        }
        return;
    }

    const MRWorldGPU world = worlds[0];
    const MRArticulationGPU articulation =
        articulations[dispatch.articulationIndex];
    const MRParallelABAArticulationGPU schedule =
        scheduleArticulations[dispatch.articulationIndex];
    status.bodyCount = articulation.bodyCount;
    status.nq = articulation.nq;
    status.nv = articulation.nv;
    clearParallelABAFailure(
        laneFailureCodes,
        laneFailureIndices,
        laneFailureDiagnostics,
        lane
    );
    if (lane == 0u &&
        ((articulation.rootType != MR_ROOT_FIXED &&
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
          dispatch.wrenchStride < articulation.bodyCount) ||
         schedule.abiVersion !=
             MR_PARALLEL_ABA_SCHEDULE_ABI_VERSION ||
         schedule.articulationIndex != dispatch.articulationIndex ||
         schedule.bodyCount != articulation.bodyCount ||
         schedule.jointCount != articulation.jointCount ||
         schedule.rootLocalBody !=
             articulation.rootBody - articulation.firstBody ||
         schedule.forwardLevelCount == 0u ||
         schedule.forwardLevelCount != schedule.maximumDepth + 1u ||
         schedule.reverseLevelCount != schedule.maximumDepth ||
         schedule.maximumLevelWidth == 0u ||
         schedule.maximumLevelWidth > 32u)) {
        laneFailureCodes[0] =
            articulation.bodyCount > kABAMaxBodies ||
                articulation.nv > kABAMaxDofs ||
                articulation.nq > kABAMaxQ ||
                schedule.maximumLevelWidth > 32u
            ? MR_ABA_UNSUPPORTED_TOPOLOGY
            : MR_ABA_INVALID_MODEL;
    }
    if (collectParallelABAFailure(
            laneFailureCodes,
            laneFailureIndices,
            laneFailureDiagnostics,
            &auxiliary.selectedFailureCode,
            &auxiliary.selectedFailureIndex,
            &auxiliary.selectedFailureDiagnostics,
            lane
        )) {
        if (lane == 0u) {
            publishParallelABAFailure(
                status,
                statuses,
                environment,
                auxiliary.selectedFailureCode,
                auxiliary.selectedFailureIndex,
                auxiliary.selectedFailureDiagnostics
            );
        }
        return;
    }

    const uint rootLocal = schedule.rootLocalBody;
    const uint qBase = environment * dispatch.qStride;
    const uint vBase = environment * dispatch.vStride;
    const uint effortBase = environment * dispatch.effortStride;
    const uint wrenchBase = environment * dispatch.wrenchStride;
    device const float* environmentQ = q + qBase;
    device const float* environmentV = v + vBase;
    device const float* environmentEffort = effort + effortBase;
#if MR_ABA_BODY_PARAMETERS
    const float4 controller = controllerParameters[environment];
    const float gainScale = max(controller.x, 0.0f);
    const float driveDampingScale = max(controller.y, 0.0f);
#else
    constexpr float gainScale = 1.0f;
    constexpr float driveDampingScale = 1.0f;
#endif

    clearParallelABAFailure(
        laneFailureCodes,
        laneFailureIndices,
        laneFailureDiagnostics,
        lane
    );
#if MR_ABA_BODY_PARAMETERS
    if (lane == 0u && !finite4(controller)) {
        laneFailureCodes[0] = MR_ABA_NONFINITE_INPUT;
    }
#endif
    for (uint localQ = lane;
         localQ < articulation.nq;
         localQ += 32u) {
        const float value = environmentQ[localQ];
        if (!isfinite(value) &&
            laneFailureCodes[lane] == MR_ABA_SUCCESS) {
            laneFailureCodes[lane] = MR_ABA_NONFINITE_INPUT;
            laneFailureIndices[lane] = localQ;
        }
        candidateNextQ[localQ] = value;
    }
    for (uint localV = lane;
         localV < articulation.nv;
         localV += 32u) {
        device const MRDofPropertiesGPU& dof =
            dofs[articulation.vOffset + localV];
        if ((!isfinite(environmentV[localV]) ||
             !isfinite(environmentEffort[localV]) ||
             !finite4(dof.drive) ||
             dof.drive.z < 0.0f) &&
            laneFailureCodes[lane] == MR_ABA_SUCCESS) {
            laneFailureCodes[lane] = MR_ABA_NONFINITE_INPUT;
            laneFailureIndices[lane] = localV;
        }
    }
    if (collectParallelABAFailure(
            laneFailureCodes,
            laneFailureIndices,
            laneFailureDiagnostics,
            &auxiliary.selectedFailureCode,
            &auxiliary.selectedFailureIndex,
            &auxiliary.selectedFailureDiagnostics,
            lane
        )) {
        if (lane == 0u) {
            publishParallelABAFailure(
                status,
                statuses,
                environment,
                auxiliary.selectedFailureCode,
                auxiliary.selectedFailureIndex,
                auxiliary.selectedFailureDiagnostics
            );
        }
        return;
    }

    for (uint forwardLevel = 0u;
         forwardLevel < schedule.forwardLevelCount;
         ++forwardLevel) {
        const MRParallelABALevelGPU level = scheduleLevels[
            schedule.forwardLevelOffset + forwardLevel
        ];
        clearParallelABAFailure(
            laneFailureCodes,
            laneFailureIndices,
            laneFailureDiagnostics,
            lane
        );
        if (lane < level.bodyCount) {
            const uint localBody =
                levelBodies[level.bodyOffset + lane];
            if (localBody >= articulation.bodyCount) {
                laneFailureCodes[lane] = MR_ABA_INVALID_MODEL;
                laneFailureIndices[lane] = localBody;
            } else if (forwardLevel == 0u) {
                if (level.bodyCount != 1u || localBody != rootLocal) {
                    laneFailureCodes[lane] = MR_ABA_INVALID_MODEL;
                    laneFailureIndices[lane] = localBody;
                } else if (articulation.rootType == MR_ROOT_FLOATING) {
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
                        laneFailureCodes[lane] =
                            MR_ABA_INVALID_QUATERNION;
                        laneFailureIndices[lane] = 0u;
                    } else {
                        bodyPosition[localBody] = float3(
                            environmentQ[0],
                            environmentQ[1],
                            environmentQ[2]
                        );
                        bodyRotation[localBody] = checkedRootRotation;
                        linearVelocity[localBody] = float3(
                            environmentV[0],
                            environmentV[1],
                            environmentV[2]
                        );
                        angularVelocity[localBody] = float3(
                            environmentV[3],
                            environmentV[4],
                            environmentV[5]
                        );
                    }
                } else {
                    bodyPosition[localBody] = float3(0.0f);
                    bodyRotation[localBody] =
                        float4(0.0f, 0.0f, 0.0f, 1.0f);
                    linearVelocity[localBody] = float3(0.0f);
                    angularVelocity[localBody] = float3(0.0f);
                }
                if (laneFailureCodes[lane] == MR_ABA_SUCCESS) {
                    parentToBody[localBody] = float3(0.0f);
                    motionAngular[localBody] = float3(0.0f);
                    motionLinear[localBody] = float3(0.0f);
                    biasAngular[localBody] = float3(0.0f);
                    biasLinear[localBody] = float3(0.0f);
                }
            } else {
                const uint localParent = scheduleParentLocal[
                    schedule.parentLocalOffset + localBody
                ];
                const uint globalJoint = scheduleInboundJoint[
                    schedule.inboundJointOffset + localBody
                ];
                if (localParent >= articulation.bodyCount ||
                    globalJoint < articulation.firstJoint ||
                    globalJoint >= articulation.firstJoint +
                        articulation.jointCount) {
                    laneFailureCodes[lane] = MR_ABA_INVALID_MODEL;
                    laneFailureIndices[lane] = globalJoint;
                } else {
                    const MRJointDescriptorGPU joint = joints[globalJoint];
                    const bool scalarJoint =
                        joint.jointType == MR_JOINT_REVOLUTE ||
                        joint.jointType == MR_JOINT_CONTINUOUS ||
                        joint.jointType == MR_JOINT_PRISMATIC;
                    const bool fixedJoint =
                        joint.jointType == MR_JOINT_FIXED;
                    float4 parentJointRotation;
                    float4 childJointRotation;
                    if (joint.parentBody !=
                            articulation.firstBody + localParent ||
                        joint.childBody !=
                            articulation.firstBody + localBody ||
                        joint.flags != 0u ||
                        (!scalarJoint && !fixedJoint) ||
                        joint.nq != (scalarJoint ? 1u : 0u) ||
                        joint.nv != (scalarJoint ? 1u : 0u) ||
                        !finite4(joint.parentAnchor) ||
                        !finite4(joint.childAnchor) ||
                        !normalizedQuaternion(
                            joint.parentRotation,
                            parentJointRotation,
                            true
                        ) ||
                        !normalizedQuaternion(
                            joint.childRotation,
                            childJointRotation,
                            true
                        ) ||
                        (scalarJoint &&
                         (joint.qOffset < articulation.qOffset ||
                          joint.qOffset >=
                              articulation.qOffset + articulation.nq ||
                          joint.vOffset < articulation.vOffset ||
                          joint.vOffset >=
                              articulation.vOffset + articulation.nv))) {
                        laneFailureCodes[lane] =
                            !scalarJoint && !fixedJoint
                            ? MR_ABA_UNSUPPORTED_TOPOLOGY
                            : MR_ABA_INVALID_MODEL;
                        laneFailureIndices[lane] = globalJoint;
                    } else {
                        float3 axisInJoint =
                            float3(1.0f, 0.0f, 0.0f);
                        float4 motionRotation =
                            float4(0.0f, 0.0f, 0.0f, 1.0f);
                        float jointCoordinate = 0.0f;
                        float jointRate = 0.0f;
                        if (scalarJoint) {
                            const float axisNormSquared =
                                dot(joint.axis0.xyz, joint.axis0.xyz);
                            if (!finite4(joint.axis0) ||
                                !(axisNormSquared >
                                    kQuaternionMinimum)) {
                                laneFailureCodes[lane] =
                                    MR_ABA_INVALID_MODEL;
                                laneFailureIndices[lane] = globalJoint;
                            } else {
                                axisInJoint = joint.axis0.xyz /
                                    sqrt(axisNormSquared);
                                const uint localQ = joint.qOffset -
                                    articulation.qOffset;
                                const uint localV = joint.vOffset -
                                    articulation.vOffset;
                                jointCoordinate = environmentQ[localQ];
                                jointRate = environmentV[localV];
                                if (joint.jointType ==
                                        MR_JOINT_REVOLUTE ||
                                    joint.jointType ==
                                        MR_JOINT_CONTINUOUS) {
                                    motionRotation = axisAngleQuaternion(
                                        axisInJoint,
                                        jointCoordinate
                                    );
                                }
                            }
                        }
                        if (laneFailureCodes[lane] == MR_ABA_SUCCESS) {
                            const float4 parentToJointRotation =
                                quaternionMultiply(
                                    bodyRotation[localParent],
                                    parentJointRotation
                                );
                            float4 checkedChildRotation;
                            if (!normalizedQuaternion(
                                    quaternionMultiply(
                                        quaternionMultiply(
                                            parentToJointRotation,
                                            motionRotation
                                        ),
                                        quaternionConjugate(
                                            childJointRotation
                                        )
                                    ),
                                    checkedChildRotation,
                                    false
                                )) {
                                laneFailureCodes[lane] =
                                    MR_ABA_NONFINITE_RESULT;
                                laneFailureIndices[lane] =
                                    articulation.firstBody + localBody;
                            } else {
                                bodyRotation[localBody] =
                                    checkedChildRotation;
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
                                    bodyRotation[localBody],
                                    joint.childAnchor.xyz
                                );
                                bodyPosition[localBody] =
                                    jointPosition - childAnchor;
                                parentToBody[localBody] =
                                    bodyPosition[localBody] -
                                    bodyPosition[localParent];
                                const float3 parentToJoint =
                                    jointPosition -
                                    bodyPosition[localParent];
                                angularVelocity[localBody] =
                                    angularVelocity[localParent] +
                                    (!prismatic && scalarJoint
                                         ? jointAxis * jointRate
                                         : float3(0.0f));
                                linearVelocity[localBody] =
                                    linearVelocity[localParent] +
                                    cross(
                                        angularVelocity[localParent],
                                        parentToJoint
                                    ) +
                                    (prismatic
                                         ? jointAxis * jointRate
                                         : float3(0.0f)) -
                                    cross(
                                        angularVelocity[localBody],
                                        childAnchor
                                    );
                                motionAngular[localBody] =
                                    scalarJoint && !prismatic
                                    ? jointAxis
                                    : float3(0.0f);
                                motionLinear[localBody] =
                                    prismatic
                                    ? jointAxis
                                    : (scalarJoint
                                           ? -cross(jointAxis, childAnchor)
                                           : float3(0.0f));
                                biasAngular[localBody] =
                                    scalarJoint && !prismatic
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
                                biasLinear[localBody] =
                                    cross(
                                        angularVelocity[localParent],
                                        cross(
                                            angularVelocity[localParent],
                                            parentToJoint
                                        )
                                    ) +
                                    prismaticCoriolis -
                                    cross(
                                        biasAngular[localBody],
                                        childAnchor
                                    ) -
                                    cross(
                                        angularVelocity[localBody],
                                        cross(
                                            angularVelocity[localBody],
                                            childAnchor
                                        )
                                    );
                                if (!finite3(bodyPosition[localBody]) ||
                                    !finite4(bodyRotation[localBody]) ||
                                    !finite3(
                                        angularVelocity[localBody]
                                    ) ||
                                    !finite3(linearVelocity[localBody]) ||
                                    !finite3(motionAngular[localBody]) ||
                                    !finite3(motionLinear[localBody]) ||
                                    !finite3(biasAngular[localBody]) ||
                                    !finite3(biasLinear[localBody])) {
                                    laneFailureCodes[lane] =
                                        MR_ABA_NONFINITE_RESULT;
                                    laneFailureIndices[lane] =
                                        articulation.firstBody +
                                        localBody;
                                }
                            }
                        }
                    }
                }
            }
        }
        if (collectParallelABAFailure(
                laneFailureCodes,
                laneFailureIndices,
                laneFailureDiagnostics,
                &auxiliary.selectedFailureCode,
                &auxiliary.selectedFailureIndex,
                &auxiliary.selectedFailureDiagnostics,
                lane
            )) {
            if (lane == 0u) {
                publishParallelABAFailure(
                    status,
                    statuses,
                    environment,
                    auxiliary.selectedFailureCode,
                    auxiliary.selectedFailureIndex,
                    auxiliary.selectedFailureDiagnostics
                );
            }
            return;
        }
    }

    clearParallelABAFailure(
        laneFailureCodes,
        laneFailureIndices,
        laneFailureDiagnostics,
        lane
    );
    if (lane < articulation.bodyCount) {
        const uint localBody = lane;
        const uint globalBody = articulation.firstBody + localBody;
        device const MRBodyPropertiesGPU& body = bodies[globalBody];
#if MR_ABA_BODY_PARAMETERS
        const float4 physical = bodyParameters[
            environment * world.bodyCount + globalBody
        ];
        const float massScale = max(physical.x, 1.0e-4f);
        const float inertiaScale = massScale;
        const float dampingScale = max(physical.w, 0.0f);
#else
        constexpr float massScale = 1.0f;
        constexpr float inertiaScale = 1.0f;
        constexpr float dampingScale = 1.0f;
#endif
        if (body.articulationIndex != dispatch.articulationIndex ||
            body.motionType != MR_MOTION_DYNAMIC ||
            !(body.massAndInverseMass.x > 0.0f) ||
            !finite4(body.massAndInverseMass) ||
            !finite4(body.inertiaRow0) ||
            !finite4(body.inertiaRow1) ||
            !finite4(body.inertiaRow2) ||
            !finite4(body.dampingAndSpeedLimits)
#if MR_ABA_BODY_PARAMETERS
            || !finite4(physical)
#endif
        ) {
            laneFailureCodes[lane] = MR_ABA_INVALID_MODEL;
            laneFailureIndices[lane] = globalBody;
        } else {
            const uint matrixBase =
                localBody * kABASymmetricInertiaEntries;
            for (uint entry = 0u;
                 entry < kABASymmetricInertiaEntries;
                 ++entry) {
                articulatedInertia[matrixBase + entry] = 0.0f;
                propagatedInertia[matrixBase + entry] = 0.0f;
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
                     column <= row;
                     ++column) {
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
                    laneFailureCodes[lane] = MR_ABA_NONFINITE_INPUT;
                    laneFailureIndices[lane] = globalBody;
                } else {
                    externalForce += wrench.force.xyz;
                    externalTorque += wrench.torque.xyz;
                }
            }
            if (laneFailureCodes[lane] == MR_ABA_SUCCESS) {
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
                jointDenominator[localBody] = 0.0f;
                jointResidual[localBody] = 0.0f;
            }
        }
    }
    if (collectParallelABAFailure(
            laneFailureCodes,
            laneFailureIndices,
            laneFailureDiagnostics,
            &auxiliary.selectedFailureCode,
            &auxiliary.selectedFailureIndex,
            &auxiliary.selectedFailureDiagnostics,
            lane
        )) {
        if (lane == 0u) {
            publishParallelABAFailure(
                status,
                statuses,
                environment,
                auxiliary.selectedFailureCode,
                auxiliary.selectedFailureIndex,
                auxiliary.selectedFailureDiagnostics
            );
        }
        return;
    }

    if (lane == 0u && articulation.rootType == MR_ROOT_FLOATING) {
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
            articulatedBias[rootLocal * 6u + 3u + axis] -=
                environmentEffort[axis];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float laneMinimumPivot = INFINITY;
    float laneMaximumPivot = 0.0f;
    for (uint reverseLevel = 0u;
         reverseLevel < schedule.reverseLevelCount;
         ++reverseLevel) {
        const MRParallelABALevelGPU level = scheduleLevels[
            schedule.reverseLevelOffset + reverseLevel
        ];
        clearParallelABAFailure(
            laneFailureCodes,
            laneFailureIndices,
            laneFailureDiagnostics,
            lane
        );
        if (lane < level.bodyCount) {
            const uint localBody =
                levelBodies[level.bodyOffset + lane];
            const uint globalJoint = scheduleInboundJoint[
                schedule.inboundJointOffset + localBody
            ];
            const MRJointDescriptorGPU joint = joints[globalJoint];
            const uint matrixBase =
                localBody * kABASymmetricInertiaEntries;
            float3 inertiaTimesBiasAngular = float3(0.0f);
            float3 inertiaTimesBiasLinear = float3(0.0f);
            for (uint row = 0u; row < 6u; ++row) {
                float value = 0.0f;
                for (uint column = 0u; column < 6u; ++column) {
                    value += symmetric6Value(
                        articulatedInertia,
                        matrixBase,
                        row,
                        column
                    ) * spatialComponent(
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
            float3 propagatedTorque = float3(
                articulatedBias[localBody * 6u + 0u],
                articulatedBias[localBody * 6u + 1u],
                articulatedBias[localBody * 6u + 2u]
            ) + inertiaTimesBiasAngular;
            float3 propagatedForce = float3(
                articulatedBias[localBody * 6u + 3u],
                articulatedBias[localBody * 6u + 4u],
                articulatedBias[localBody * 6u + 5u]
            ) + inertiaTimesBiasLinear;

            if (joint.nv == 1u) {
                float3 projectedAngular = float3(0.0f);
                float3 projectedLinear = float3(0.0f);
                for (uint row = 0u; row < 6u; ++row) {
                    float value = 0.0f;
                    for (uint column = 0u; column < 6u; ++column) {
                        value += symmetric6Value(
                            articulatedInertia,
                            matrixBase,
                            row,
                            column
                        ) * spatialComponent(
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
                    laneFailureCodes[lane] =
                        MR_ABA_FACTORIZATION_FAILED;
                    laneFailureIndices[lane] = localV;
                    laneFailureDiagnostics[lane] = float4(
                        0.0f,
                        0.0f,
                        denominator,
                        maximumInertia
                    );
                } else {
                    jointDenominator[localBody] = denominator;
                    const float residual =
                        environmentEffort[localV] -
                        dot(
                            motionAngular[localBody],
                            propagatedTorque
                        ) -
                        dot(
                            motionLinear[localBody],
                            propagatedForce
                        );
                    jointResidual[localBody] = residual;
                    const float pivot = sqrt(denominator);
                    laneMinimumPivot = min(laneMinimumPivot, pivot);
                    laneMaximumPivot = max(laneMaximumPivot, pivot);
                    propagatedTorque +=
                        projectedAngular * (residual / denominator);
                    propagatedForce +=
                        projectedLinear * (residual / denominator);
                    for (uint row = 0u; row < 6u; ++row) {
                        for (uint column = 0u;
                             column <= row;
                             ++column) {
                            articulatedInertia[
                                matrixBase +
                                symmetric6Index(row, column)
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
            } else if (joint.nv != 0u) {
                laneFailureCodes[lane] =
                    MR_ABA_UNSUPPORTED_TOPOLOGY;
                laneFailureIndices[lane] = globalJoint;
            }

            if (laneFailureCodes[lane] == MR_ABA_SUCCESS) {
                const float3 offset = parentToBody[localBody];
                for (uint row = 0u; row < 6u; ++row) {
                    for (uint column = 0u;
                         column <= row;
                         ++column) {
                        float transformed = 0.0f;
                        for (uint left = 0u; left < 6u; ++left) {
                            const float leftTransform =
                                motionTransformElement(
                                    offset,
                                    left,
                                    row
                                );
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
                        propagatedInertia[
                            matrixBase +
                            symmetric6Index(row, column)
                        ] = transformed;
                    }
                }
                propagatedTorque += cross(offset, propagatedForce);
                for (uint component = 0u;
                     component < 6u;
                     ++component) {
                    articulatedBias[localBody * 6u + component] =
                        spatialComponent(
                            propagatedTorque,
                            propagatedForce,
                            component
                        );
                }
            }
        }
        if (collectParallelABAFailure(
                laneFailureCodes,
                laneFailureIndices,
                laneFailureDiagnostics,
                &auxiliary.selectedFailureCode,
                &auxiliary.selectedFailureIndex,
                &auxiliary.selectedFailureDiagnostics,
                lane
            )) {
            if (lane == 0u) {
                publishParallelABAFailure(
                    status,
                    statuses,
                    environment,
                    auxiliary.selectedFailureCode,
                    auxiliary.selectedFailureIndex,
                    auxiliary.selectedFailureDiagnostics
                );
            }
            return;
        }

        clearParallelABAFailure(
            laneFailureCodes,
            laneFailureIndices,
            laneFailureDiagnostics,
            lane
        );
        if (lane < level.parentReductionCount) {
            const MRParallelABAParentReductionGPU reduction =
                parentReductions[
                    level.parentReductionOffset + lane
                ];
            if (reduction.parentLocalBody >= articulation.bodyCount) {
                laneFailureCodes[lane] = MR_ABA_INVALID_MODEL;
                laneFailureIndices[lane] =
                    reduction.parentLocalBody;
            } else {
                const uint expectedFirst = childOffsets[
                    schedule.childOffsetOffset +
                    reduction.parentLocalBody
                ];
                const uint expectedEnd = childOffsets[
                    schedule.childOffsetOffset +
                    reduction.parentLocalBody + 1u
                ];
                if (reduction.firstChildIndex != expectedFirst ||
                    expectedEnd < expectedFirst ||
                    reduction.childCount !=
                        expectedEnd - expectedFirst ||
                    reduction.firstChildIndex <
                        schedule.childIndexOffset ||
                    reduction.firstChildIndex + reduction.childCount >
                        schedule.childIndexOffset +
                            schedule.childIndexCount) {
                    laneFailureCodes[lane] = MR_ABA_INVALID_MODEL;
                    laneFailureIndices[lane] =
                        reduction.parentLocalBody;
                }
            }
        }
        if (collectParallelABAFailure(
                laneFailureCodes,
                laneFailureIndices,
                laneFailureDiagnostics,
                &auxiliary.selectedFailureCode,
                &auxiliary.selectedFailureIndex,
                &auxiliary.selectedFailureDiagnostics,
                lane
            )) {
            if (lane == 0u) {
                publishParallelABAFailure(
                    status,
                    statuses,
                    environment,
                    auxiliary.selectedFailureCode,
                    auxiliary.selectedFailureIndex,
                    auxiliary.selectedFailureDiagnostics
                );
            }
            return;
        }

        if (lane < level.parentReductionCount) {
            const MRParallelABAParentReductionGPU reduction =
                parentReductions[
                    level.parentReductionOffset + lane
                ];
            const uint parentMatrixBase =
                reduction.parentLocalBody *
                kABASymmetricInertiaEntries;
            for (uint childOrdinal = 0u;
                 childOrdinal < reduction.childCount;
                 ++childOrdinal) {
                const uint localChild = childIndices[
                    reduction.firstChildIndex + childOrdinal
                ];
                const uint childMatrixBase =
                    localChild * kABASymmetricInertiaEntries;
                for (uint entry = 0u;
                     entry < kABASymmetricInertiaEntries;
                     ++entry) {
                    articulatedInertia[parentMatrixBase + entry] +=
                        propagatedInertia[childMatrixBase + entry];
                }
                for (uint component = 0u;
                     component < 6u;
                     ++component) {
                    articulatedBias[
                        reduction.parentLocalBody * 6u + component
                    ] += articulatedBias[
                        localChild * 6u + component
                    ];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float reducedMinimumPivot = simd_min(laneMinimumPivot);
    const float reducedMaximumPivot = simd_max(laneMaximumPivot);
    if (lane == 0u) {
        auxiliary.minimumPivot = reducedMinimumPivot;
        auxiliary.maximumPivot = reducedMaximumPivot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float3* bodyAccelerationAngular = bodyPosition;
    threadgroup float3* bodyAccelerationLinear = angularVelocity;
    clearParallelABAFailure(
        laneFailureCodes,
        laneFailureIndices,
        laneFailureDiagnostics,
        lane
    );
    if (lane == 0u) {
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
                for (uint column = 0u;
                     column <= row;
                     ++column) {
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
                            laneFailureCodes[0] =
                                MR_ABA_FACTORIZATION_FAILED;
                            laneFailureIndices[0] = row;
                            laneFailureDiagnostics[0] = float4(
                                auxiliary.minimumPivot,
                                auxiliary.maximumPivot,
                                value,
                                maximumRootInertia
                            );
                            break;
                        }
                        const float pivot = sqrt(value);
                        rootFactor[row * 6u + row] = pivot;
                        auxiliary.minimumPivot = min(
                            auxiliary.minimumPivot,
                            pivot
                        );
                        auxiliary.maximumPivot = max(
                            auxiliary.maximumPivot,
                            pivot
                        );
                    } else {
                        rootFactor[row * 6u + column] =
                            value / rootFactor[column * 6u + column];
                    }
                }
                if (laneFailureCodes[0] != MR_ABA_SUCCESS) {
                    break;
                }
            }
            if (laneFailureCodes[0] == MR_ABA_SUCCESS) {
                for (uint row = 0u; row < 6u; ++row) {
                    float value =
                        -articulatedBias[rootLocal * 6u + row];
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
            }
        } else {
            bodyAccelerationAngular[rootLocal] = float3(0.0f);
            bodyAccelerationLinear[rootLocal] = float3(0.0f);
        }
    }
    if (collectParallelABAFailure(
            laneFailureCodes,
            laneFailureIndices,
            laneFailureDiagnostics,
            &auxiliary.selectedFailureCode,
            &auxiliary.selectedFailureIndex,
            &auxiliary.selectedFailureDiagnostics,
            lane
        )) {
        if (lane == 0u) {
            publishParallelABAFailure(
                status,
                statuses,
                environment,
                auxiliary.selectedFailureCode,
                auxiliary.selectedFailureIndex,
                auxiliary.selectedFailureDiagnostics
            );
        }
        return;
    }

    for (uint forwardLevel = 1u;
         forwardLevel < schedule.forwardLevelCount;
         ++forwardLevel) {
        const MRParallelABALevelGPU level = scheduleLevels[
            schedule.forwardLevelOffset + forwardLevel
        ];
        if (lane < level.bodyCount) {
            const uint localBody =
                levelBodies[level.bodyOffset + lane];
            const uint localParent = scheduleParentLocal[
                schedule.parentLocalOffset + localBody
            ];
            const uint globalJoint = scheduleInboundJoint[
                schedule.inboundJointOffset + localBody
            ];
            const MRJointDescriptorGPU joint = joints[globalJoint];
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
                    projectedParent += projectedInertia[
                        localBody * 6u + component
                    ] * spatialComponent(
                        transformedParentAngular,
                        transformedParentLinear,
                        component
                    );
                }
                const float jointAcceleration =
                    (jointResidual[localBody] - projectedParent) /
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
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    clearParallelABAFailure(
        laneFailureCodes,
        laneFailureIndices,
        laneFailureDiagnostics,
        lane
    );
    float laneMaximumAcceleration = 0.0f;
    const float timestep = world.gravityAndTimestep.w;
    for (uint localV = lane;
         localV < articulation.nv;
         localV += 32u) {
        const float acceleration = candidateAcceleration[localV];
        const float nextVelocity =
            environmentV[localV] + timestep * acceleration;
        if (!isfinite(acceleration) || !isfinite(nextVelocity)) {
            laneFailureCodes[lane] = MR_ABA_NONFINITE_RESULT;
            laneFailureIndices[lane] = localV;
        } else {
            candidateNextV[localV] = nextVelocity;
            laneMaximumAcceleration = max(
                laneMaximumAcceleration,
                abs(acceleration)
            );
        }
    }
    const float maximumAcceleration = simd_max(
        laneMaximumAcceleration
    );
    if (lane == 0u) {
        auxiliary.maximumAcceleration = maximumAcceleration;
        auxiliary.quaternionNormError = 0.0f;
    }
    if (collectParallelABAFailure(
            laneFailureCodes,
            laneFailureIndices,
            laneFailureDiagnostics,
            &auxiliary.selectedFailureCode,
            &auxiliary.selectedFailureIndex,
            &auxiliary.selectedFailureDiagnostics,
            lane
        )) {
        if (lane == 0u) {
            publishParallelABAFailure(
                status,
                statuses,
                environment,
                auxiliary.selectedFailureCode,
                auxiliary.selectedFailureIndex,
                auxiliary.selectedFailureDiagnostics
            );
        }
        return;
    }

    clearParallelABAFailure(
        laneFailureCodes,
        laneFailureIndices,
        laneFailureDiagnostics,
        lane
    );
    if (lane == 0u && articulation.rootType == MR_ROOT_FLOATING) {
        candidateNextQ[0] =
            environmentQ[0] + timestep * candidateNextV[0];
        candidateNextQ[1] =
            environmentQ[1] + timestep * candidateNextV[1];
        candidateNextQ[2] =
            environmentQ[2] + timestep * candidateNextV[2];
        const float3 rotationVector = timestep * float3(
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
            laneFailureCodes[0] = MR_ABA_NONFINITE_RESULT;
            laneFailureIndices[0] = 3u;
            laneFailureDiagnostics[0] = float4(
                auxiliary.maximumAcceleration,
                isfinite(angle) ? angle : kFloatMaximum,
                max(
                    abs(candidateNextV[3]),
                    max(
                        abs(candidateNextV[4]),
                        abs(candidateNextV[5])
                    )
                ),
                length(float4(
                    environmentQ[3],
                    environmentQ[4],
                    environmentQ[5],
                    environmentQ[6]
                ))
            );
        } else {
            candidateNextQ[3] = updated.x;
            candidateNextQ[4] = updated.y;
            candidateNextQ[5] = updated.z;
            candidateNextQ[6] = updated.w;
            auxiliary.quaternionNormError =
                abs(length(updated) - 1.0f);
        }
    }
    if (lane < articulation.bodyCount && lane != rootLocal) {
        const uint globalJoint = scheduleInboundJoint[
            schedule.inboundJointOffset + lane
        ];
        const MRJointDescriptorGPU joint = joints[globalJoint];
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
    if (collectParallelABAFailure(
            laneFailureCodes,
            laneFailureIndices,
            laneFailureDiagnostics,
            &auxiliary.selectedFailureCode,
            &auxiliary.selectedFailureIndex,
            &auxiliary.selectedFailureDiagnostics,
            lane
        )) {
        if (lane == 0u) {
            publishParallelABAFailure(
                status,
                statuses,
                environment,
                auxiliary.selectedFailureCode,
                auxiliary.selectedFailureIndex,
                auxiliary.selectedFailureDiagnostics
            );
        }
        return;
    }

    clearParallelABAFailure(
        laneFailureCodes,
        laneFailureIndices,
        laneFailureDiagnostics,
        lane
    );
    for (uint localQ = lane;
         localQ < articulation.nq;
         localQ += 32u) {
        if (!isfinite(candidateNextQ[localQ])) {
            laneFailureCodes[lane] = MR_ABA_NONFINITE_RESULT;
            laneFailureIndices[lane] = localQ;
        }
    }
    if (collectParallelABAFailure(
            laneFailureCodes,
            laneFailureIndices,
            laneFailureDiagnostics,
            &auxiliary.selectedFailureCode,
            &auxiliary.selectedFailureIndex,
            &auxiliary.selectedFailureDiagnostics,
            lane
        )) {
        if (lane == 0u) {
            publishParallelABAFailure(
                status,
                statuses,
                environment,
                auxiliary.selectedFailureCode,
                auxiliary.selectedFailureIndex,
                auxiliary.selectedFailureDiagnostics
            );
        }
        return;
    }

    const uint accelerationBase =
        environment * dispatch.accelerationStride;
    const uint nextVBase = environment * dispatch.nextVStride;
    const uint nextQBase = environment * dispatch.nextQStride;
    for (uint localV = lane;
         localV < articulation.nv;
         localV += 32u) {
        accelerationOutput[accelerationBase + localV] =
            candidateAcceleration[localV];
        nextVOutput[nextVBase + localV] = candidateNextV[localV];
    }
    for (uint localQ = lane;
         localQ < articulation.nq;
         localQ += 32u) {
        nextQOutput[nextQBase + localQ] = candidateNextQ[localQ];
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (lane == 0u) {
        status.diagnostics = float4(
            isfinite(auxiliary.minimumPivot)
                ? auxiliary.minimumPivot
                : 0.0f,
            auxiliary.maximumPivot,
            auxiliary.maximumAcceleration,
            auxiliary.quaternionNormError
        );
        statuses[environment] = status;
    }
}
