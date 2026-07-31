#include <metal_stdlib>

#include "metalrobo/engine_types.h"
#include "metalrobo/rod_gpu_shared.h"

using namespace metal;

namespace {

constant float kFloatMaximum = 3.402823466e38f;

inline bool validWorldDispatch(
    device const MRMetalWorldDispatchGPU& dispatch
) {
    constexpr uint knownFlags =
        MR_METAL_WORLD_APPLY_BODY_DAMPING |
        MR_METAL_WORLD_DETERMINISTIC |
        MR_METAL_WORLD_HAS_RESETS |
        MR_METAL_WORLD_FREE_MOTION_ONLY |
        MR_METAL_WORLD_CONTACTS |
        MR_METAL_WORLD_IMPLICIT_POSITION_DRIVES |
        MR_METAL_WORLD_NATIVE_TASK;
    const uint modeFlags =
        dispatch.flags &
        (MR_METAL_WORLD_FREE_MOTION_ONLY |
         MR_METAL_WORLD_CONTACTS);
    return
        dispatch.abiVersion == MR_METAL_WORLD_ABI_VERSION &&
        dispatch.environmentCount > 0u &&
        dispatch.controlStepCount > 0u &&
        dispatch.physicsSubsteps > 0u &&
        dispatch.physicsSubsteps <=
            MR_METAL_WORLD_MAX_PHYSICS_SUBSTEPS &&
        dispatch.nq > 0u &&
        dispatch.nv > 0u &&
        dispatch.qStride >= dispatch.nq &&
        dispatch.vStride >= dispatch.nv &&
        dispatch.effortEnvironmentStride >= dispatch.nv &&
        dispatch.observationEnvironmentStride >=
            dispatch.nq + dispatch.nv &&
        dispatch.effortStepStride >=
            dispatch.environmentCount *
                dispatch.effortEnvironmentStride &&
        dispatch.observationStepStride >=
            dispatch.environmentCount *
                dispatch.observationEnvironmentStride &&
        dispatch.accelerationStepStride >=
            dispatch.environmentCount * dispatch.nv &&
        (dispatch.flags & ~knownFlags) == 0u &&
        (modeFlags == MR_METAL_WORLD_FREE_MOTION_ONLY ||
         modeFlags == MR_METAL_WORLD_CONTACTS);
}

inline uint mapABAStatus(const uint code) {
    switch (code) {
    case MR_ABA_SUCCESS:
        return MR_STEP_SUCCESS;
    case MR_ABA_NONFINITE_INPUT:
    case MR_ABA_INVALID_QUATERNION:
        return MR_STEP_NONFINITE_INPUT;
    case MR_ABA_FACTORIZATION_FAILED:
        return MR_STEP_FACTORIZATION_FAILED;
    case MR_ABA_NONFINITE_RESULT:
        return MR_STEP_NONFINITE_RESULT;
    case MR_ABA_INVALID_DISPATCH:
    case MR_ABA_INVALID_MODEL:
    case MR_ABA_UNSUPPORTED_TOPOLOGY:
    default:
        return MR_STEP_UNSUPPORTED;
    }
}

inline uint mapRodStatus(const uint code) {
    switch (code) {
    case MR_ROD_GPU_SUCCESS:
        return MR_STEP_SUCCESS;
    case MR_ROD_GPU_INVALID_DISPATCH:
        return MR_STEP_UNSUPPORTED;
    case MR_ROD_GPU_DEGENERATE_GEOMETRY:
        return MR_STEP_DID_NOT_CONVERGE;
    case MR_ROD_GPU_NONFINITE_RESULT:
        return MR_STEP_NONFINITE_RESULT;
    case MR_ROD_GPU_DID_NOT_CONVERGE:
        return MR_STEP_DID_NOT_CONVERGE;
    default:
        return MR_STEP_UNSUPPORTED;
    }
}

inline void initializeStatus(
    thread MRMetalWorldStatusGPU& status,
    const uint environment,
    const uint controlStep,
    const uint flags
) {
    status.code = MR_STEP_SUCCESS;
    status.environment = environment;
    status.controlStep = controlStep;
    status.successfulSubsteps = 0u;
    status.abaCode = MR_ABA_SUCCESS;
    status.failingSubstep = MR_INVALID_INDEX;
    status.failingIndex = MR_INVALID_INDEX;
    status.flags = flags;
    status.diagnostics = float4(kFloatMaximum, 0.0f, 0.0f, 0.0f);
}

} // namespace

// Starts one transactional control step. Reset, when requested, is part of
// the step-start state and therefore survives a later substep rollback.
// The checkpoint remains immutable until capture completes.
kernel void mr_metal_world_prepare(
    device const MRMetalWorldDispatchGPU& dispatch [[buffer(0)]],
    constant MRMetalWorldPassGPU& pass [[buffer(1)]],
    device const float* effortTrajectory [[buffer(2)]],
    device const uint* resetMasks [[buffer(3)]],
    device const float* resetQ [[buffer(4)]],
    device const float* resetV [[buffer(5)]],
    device float* stateQ [[buffer(6)]],
    device float* stateV [[buffer(7)]],
    device float* checkpointQ [[buffer(8)]],
    device float* checkpointV [[buffer(9)]],
    device float* workingEffort [[buffer(10)]],
    device MRMetalWorldStatusGPU* statuses [[buffer(11)]],
    device const MRWorldGPU& world [[buffer(12)]],
    device const MRArticulationGPU* articulations [[buffer(13)]],
    device const MRDofPropertiesGPU* dofs [[buffer(14)]],
    device const MRActuatorProfileGPU* actuatorProfiles
        [[buffer(15)]],
    device const float4* taskControllerParameters
        [[buffer(16)]],
    uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }

    MRMetalWorldStatusGPU status{};
    initializeStatus(
        status,
        environment,
        pass.controlStep,
        dispatch.flags
    );
    if (!validWorldDispatch(dispatch) ||
        pass.controlStep >= dispatch.controlStepCount ||
        pass.physicsSubstep != MR_INVALID_INDEX ||
        pass.reserved0 != 0u ||
        pass.reserved1 != 0u) {
        status.code = MR_STEP_UNSUPPORTED;
        status.abaCode = MR_ABA_INVALID_DISPATCH;
        statuses[environment] = status;
        return;
    }

    const bool hasResets =
        (dispatch.flags & MR_METAL_WORLD_HAS_RESETS) != 0u;
    const bool applyReset =
        hasResets &&
        resetMasks[
            pass.controlStep * dispatch.resetMaskStepStride +
            environment
        ] != 0u;
    const uint qBase = environment * dispatch.qStride;
    const uint vBase = environment * dispatch.vStride;
    for (uint coordinate = 0u;
         coordinate < dispatch.nq;
         ++coordinate) {
        const float value = applyReset
            ? resetQ[qBase + coordinate]
            : stateQ[qBase + coordinate];
        stateQ[qBase + coordinate] = value;
        checkpointQ[qBase + coordinate] = value;
    }
    for (uint coordinate = 0u;
         coordinate < dispatch.nv;
         ++coordinate) {
        const float value = applyReset
            ? resetV[vBase + coordinate]
            : stateV[vBase + coordinate];
        stateV[vBase + coordinate] = value;
        checkpointV[vBase + coordinate] = value;
        float command = effortTrajectory[
            pass.controlStep * dispatch.effortStepStride +
            environment * dispatch.effortEnvironmentStride +
            coordinate
        ];
        if ((dispatch.flags &
             MR_METAL_WORLD_IMPLICIT_POSITION_DRIVES) != 0u) {
            const bool nativeTask =
                (dispatch.flags &
                 MR_METAL_WORLD_NATIVE_TASK) != 0u;
            const float4 controller = nativeTask
                ? taskControllerParameters[environment]
                : float4(1.0f);
            device const MRDofPropertiesGPU& dof =
                dofs[coordinate];
            command = 0.0f;
            if ((dof.flags & MR_DOF_FLAG_DRIVE) != 0u &&
                dof.qIndex != MR_INVALID_INDEX &&
                dof.qIndex < dispatch.nq) {
                float target = effortTrajectory[
                    pass.controlStep * dispatch.effortStepStride +
                    environment *
                        dispatch.effortEnvironmentStride +
                    coordinate
                ];
                if ((dof.flags &
                     MR_DOF_FLAG_POSITION_LIMIT) != 0u) {
                    target = clamp(
                        target,
                        dof.limits.x,
                        dof.limits.y
                    );
                }
                const float timestep =
                    world.gravityAndTimestep.w;
                command =
                    controller.x * dof.drive.x *
                        (
                            target -
                            stateQ[qBase + dof.qIndex] -
                            timestep * value
                        ) -
                    controller.y * dof.drive.y * value;
                const float dryFriction = dof.drive.w;
                if (dryFriction > 0.0f) {
                    if (abs(value) > 1.0e-4f) {
                        command -=
                            copysign(dryFriction, value);
                    } else {
                        command -= clamp(
                            command,
                            -dryFriction,
                            dryFriction
                        );
                    }
                }
                if ((dof.flags &
                     MR_DOF_FLAG_EFFORT_LIMIT) != 0u &&
                    dof.limits.w > 0.0f) {
                    command = clamp(
                        command,
                        -dof.limits.w,
                        dof.limits.w
                    );
                }
                command *= controller.w;
            }
        }
        device const MRActuatorProfileGPU& actuator =
            actuatorProfiles[coordinate];
        const float speedFraction = clamp(
            abs(value) /
                max(
                    actuator.motorAndSpeed.z,
                    1.175494351e-38f
                ),
            0.0f,
            1.0f
        );
        const float envelope = min(
            dofs[coordinate].limits.w,
            actuator.transmissionAndEnvelope.z *
                actuator.motorAndSpeed.w *
                (1.0f - speedFraction)
        );
        command = clamp(command, -envelope, envelope);
        workingEffort[
            environment * dispatch.effortEnvironmentStride +
            coordinate
        ] = command;
    }
    statuses[environment] = status;
}

// Commits one ABA candidate into the ping-pong state. The first failed
// substep latches a typed status and every later pass restores the immutable
// checkpoint, making the entire control step transactional per environment.
kernel void mr_metal_world_commit(
    device const MRMetalWorldDispatchGPU& dispatch [[buffer(0)]],
    constant MRMetalWorldPassGPU& pass [[buffer(1)]],
    device const MRABAStatusGPU* abaStatuses [[buffer(2)]],
    device const float* candidateQ [[buffer(3)]],
    device const float* candidateV [[buffer(4)]],
    device float* destinationQ [[buffer(5)]],
    device float* destinationV [[buffer(6)]],
    device MRMetalWorldStatusGPU* statuses [[buffer(7)]],
    device const float* checkpointQ [[buffer(8)]],
    device const float* checkpointV [[buffer(9)]],
    device const MRWorldGPU& world [[buffer(10)]],
    uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }

    MRMetalWorldStatusGPU status = statuses[environment];
    const bool validPass =
        validWorldDispatch(dispatch) &&
        world.abiVersion == MR_ENGINE_ABI_VERSION &&
        world.articulationCount > 0u &&
        pass.controlStep < dispatch.controlStepCount &&
        pass.physicsSubstep < dispatch.physicsSubsteps &&
        pass.reserved0 == 0u &&
        pass.reserved1 == 0u;
    bool validABARecord = validPass;
    bool allABASucceeded = validPass;
    uint failureCode = MR_ABA_SUCCESS;
    uint failureIndex = MR_INVALID_INDEX;
    float4 abaDiagnostics =
        float4(kFloatMaximum, 0.0f, 0.0f, 0.0f);
    for (uint articulation = 0u;
         articulation < world.articulationCount;
         ++articulation) {
        const MRABAStatusGPU aba =
            abaStatuses[
                articulation * dispatch.environmentCount +
                environment
            ];
        const bool validRecord =
            aba.environment == environment &&
            aba.articulationIndex == articulation &&
            aba.code <= MR_ABA_UNSUPPORTED_TOPOLOGY;
        validABARecord = validABARecord && validRecord;
        allABASucceeded =
            allABASucceeded &&
            validRecord &&
            aba.code == MR_ABA_SUCCESS;
        if (failureCode == MR_ABA_SUCCESS &&
            (!validRecord ||
             aba.code != MR_ABA_SUCCESS)) {
            failureCode = validRecord
                ? aba.code
                : uint(MR_ABA_INVALID_DISPATCH);
            failureIndex = validRecord
                ? aba.failingIndex
                : MR_INVALID_INDEX;
        }
        if (validRecord && aba.code == MR_ABA_SUCCESS) {
            abaDiagnostics.x = min(
                abaDiagnostics.x,
                aba.diagnostics.x
            );
            abaDiagnostics.y = max(
                abaDiagnostics.y,
                aba.diagnostics.y
            );
            abaDiagnostics.z = max(
                abaDiagnostics.z,
                aba.diagnostics.z
            );
            abaDiagnostics.w = max(
                abaDiagnostics.w,
                aba.diagnostics.w
            );
        }
    }
    const bool commitCandidate =
        status.code == MR_STEP_SUCCESS &&
        allABASucceeded;

    if (!commitCandidate && status.code == MR_STEP_SUCCESS) {
        status.code = validPass && validABARecord
            ? mapABAStatus(failureCode)
            : MR_STEP_UNSUPPORTED;
        status.abaCode = validABARecord
            ? failureCode
            : static_cast<uint>(MR_ABA_INVALID_DISPATCH);
        status.failingSubstep = pass.physicsSubstep;
        status.failingIndex = validABARecord
            ? failureIndex
            : MR_INVALID_INDEX;
    }

    const uint qBase = environment * dispatch.qStride;
    const uint vBase = environment * dispatch.vStride;
    for (uint coordinate = 0u;
         coordinate < dispatch.nq;
         ++coordinate) {
        destinationQ[qBase + coordinate] = commitCandidate
            ? candidateQ[qBase + coordinate]
            : checkpointQ[qBase + coordinate];
    }
    for (uint coordinate = 0u;
         coordinate < dispatch.nv;
         ++coordinate) {
        destinationV[vBase + coordinate] = commitCandidate
            ? candidateV[vBase + coordinate]
            : checkpointV[vBase + coordinate];
    }

    if (commitCandidate) {
        if (status.successfulSubsteps == 0u) {
            status.diagnostics = abaDiagnostics;
        } else {
            status.diagnostics.x = min(
                status.diagnostics.x,
                abaDiagnostics.x
            );
            status.diagnostics.y = max(
                status.diagnostics.y,
                abaDiagnostics.y
            );
            status.diagnostics.z = max(
                status.diagnostics.z,
                abaDiagnostics.z
            );
            status.diagnostics.w = max(
                status.diagnostics.w,
                abaDiagnostics.w
            );
        }
        ++status.successfulSubsteps;
    }
    statuses[environment] = status;
}

// Captures a stable q+v observation and the final successful substep
// acceleration. Failed control steps expose zero acceleration and their
// checkpoint-restored state, plus the exact latched failure record.
kernel void mr_metal_world_capture(
    device const MRMetalWorldDispatchGPU& dispatch [[buffer(0)]],
    constant MRMetalWorldPassGPU& pass [[buffer(1)]],
    device const float* stateQ [[buffer(2)]],
    device const float* stateV [[buffer(3)]],
    device const float* candidateAcceleration [[buffer(4)]],
    device MRMetalWorldStatusGPU* statuses [[buffer(5)]],
    device float* observations [[buffer(6)]],
    device float* accelerationTrajectory [[buffer(7)]],
    device MRMetalWorldStatusGPU* publicStatuses [[buffer(8)]],
    uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }

    MRMetalWorldStatusGPU status = statuses[environment];
    if (!validWorldDispatch(dispatch) ||
        pass.controlStep >= dispatch.controlStepCount ||
        pass.physicsSubstep != MR_INVALID_INDEX ||
        pass.reserved0 != 0u ||
        pass.reserved1 != 0u) {
        status.code = MR_STEP_UNSUPPORTED;
        status.abaCode = MR_ABA_INVALID_DISPATCH;
        status.failingSubstep = MR_INVALID_INDEX;
        status.failingIndex = MR_INVALID_INDEX;
    }
    if (status.successfulSubsteps == 0u) {
        status.diagnostics = float4(0.0f);
    }

    const uint qBase = environment * dispatch.qStride;
    const uint vBase = environment * dispatch.vStride;
    const uint observationBase =
        pass.controlStep * dispatch.observationStepStride +
        environment * dispatch.observationEnvironmentStride;
    const uint accelerationBase =
        pass.controlStep * dispatch.accelerationStepStride +
        environment * dispatch.nv;
    for (uint coordinate = 0u;
         coordinate < dispatch.nq;
         ++coordinate) {
        observations[observationBase + coordinate] =
            stateQ[qBase + coordinate];
    }
    const bool succeeded = status.code == MR_STEP_SUCCESS;
    for (uint coordinate = 0u;
         coordinate < dispatch.nv;
         ++coordinate) {
        observations[
            observationBase + dispatch.nq + coordinate
        ] = stateV[vBase + coordinate];
        accelerationTrajectory[
            accelerationBase + coordinate
        ] = succeeded
            ? candidateAcceleration[vBase + coordinate]
            : 0.0f;
    }
    publicStatuses[
        pass.controlStep * dispatch.environmentCount +
        environment
    ] = status;
    statuses[environment] = status;
}

// Rod mechanics are semantic world state, not native-object state. These
// kernels bridge the explicit packed node/edge PyTree representation to the
// existing SIMD32 DER cohort without any host staging or hidden singleton.
kernel void mr_world_prepare_rod_state(
    device const MRMetalWorldDispatchGPU& worldDispatch [[buffer(0)]],
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(1)]],
    constant MRMetalWorldPassGPU& pass [[buffer(2)]],
    device const uint* resetMasks [[buffer(3)]],
    device const MRRodNodeStateGPU* resetNodes [[buffer(4)]],
    device const MRRodEdgeStateGPU* resetEdges [[buffer(5)]],
    device MRRodNodeStateGPU* stateNodes [[buffer(6)]],
    device MRRodEdgeStateGPU* stateEdges [[buffer(7)]],
    device MRRodNodeStateGPU* checkpointNodes [[buffer(8)]],
    device MRRodEdgeStateGPU* checkpointEdges [[buffer(9)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= worldDispatch.environmentCount ||
        pass.controlStep >= worldDispatch.controlStepCount ||
        pass.physicsSubstep != MR_INVALID_INDEX) {
        return;
    }
    const bool hasResets =
        (worldDispatch.flags & MR_METAL_WORLD_HAS_RESETS) != 0u;
    const bool applyReset =
        hasResets &&
        resetMasks[
            pass.controlStep *
                worldDispatch.resetMaskStepStride +
            environment
        ] != 0u;
    const uint nodeBase =
        environment * contactDispatch.rodNodeCount;
    for (uint node = 0u;
         node < contactDispatch.rodNodeCount;
         ++node) {
        const MRRodNodeStateGPU value = applyReset
            ? resetNodes[nodeBase + node]
            : stateNodes[nodeBase + node];
        stateNodes[nodeBase + node] = value;
        checkpointNodes[nodeBase + node] = value;
    }
    const uint edgeBase =
        environment * contactDispatch.rodEdgeCount;
    for (uint edge = 0u;
         edge < contactDispatch.rodEdgeCount;
         ++edge) {
        const MRRodEdgeStateGPU value = applyReset
            ? resetEdges[edgeBase + edge]
            : stateEdges[edgeBase + edge];
        stateEdges[edgeBase + edge] = value;
        checkpointEdges[edgeBase + edge] = value;
    }
}

kernel void mr_world_prepare_rod_contact_cache(
    device const MRMetalWorldDispatchGPU& worldDispatch [[buffer(0)]],
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(1)]],
    constant MRMetalWorldPassGPU& pass [[buffer(2)]],
    device const uint* resetMasks [[buffer(3)]],
    device const MRRodToolWitnessGPU* published [[buffer(4)]],
    device MRRodToolWitnessGPU* checkpoint [[buffer(5)]],
    device MRRodToolWitnessGPU* candidate [[buffer(6)]],
    const uint flatWitness [[thread_position_in_grid]]
) {
    const uint witnessStride =
        contactDispatch.rodToolPairCount *
        MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
    const uint witnessCount =
        worldDispatch.environmentCount * witnessStride;
    if (flatWitness >= witnessCount ||
        pass.controlStep >= worldDispatch.controlStepCount ||
        pass.physicsSubstep != MR_INVALID_INDEX) {
        return;
    }
    const uint environment =
        witnessStride == 0u
        ? 0u
        : flatWitness / witnessStride;
    const bool applyReset =
        (worldDispatch.flags & MR_METAL_WORLD_HAS_RESETS) != 0u &&
        resetMasks[
            pass.controlStep *
                worldDispatch.resetMaskStepStride +
            environment
        ] != 0u;
    MRRodToolWitnessGPU value = {};
    if (!applyReset) {
        value = published[flatWitness];
    }
    checkpoint[flatWitness] = value;
    candidate[flatWitness] = value;
}

// Rod nodes, twist state, and persistent tool witnesses participate in the
// same literal-event transaction as articulations, free bodies, and rigid
// manifolds. Two private slots are initialized once per microstep and then
// ping-ponged by the statically encoded event graph.
kernel void mr_world_initialize_rod_event_state(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRRodNodeStateGPU* sourceNodes [[buffer(1)]],
    device const MRRodEdgeStateGPU* sourceEdges [[buffer(2)]],
    device const MRRodToolWitnessGPU* sourceWitnesses [[buffer(3)]],
    device MRRodNodeStateGPU* nodesA [[buffer(4)]],
    device MRRodEdgeStateGPU* edgesA [[buffer(5)]],
    device MRRodToolWitnessGPU* witnessesA [[buffer(6)]],
    device MRRodNodeStateGPU* nodesB [[buffer(7)]],
    device MRRodEdgeStateGPU* edgesB [[buffer(8)]],
    device MRRodToolWitnessGPU* witnessesB [[buffer(9)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    const uint nodeBase = environment * dispatch.rodNodeCount;
    for (uint node = 0u; node < dispatch.rodNodeCount; ++node) {
        const MRRodNodeStateGPU value =
            sourceNodes[nodeBase + node];
        nodesA[nodeBase + node] = value;
        nodesB[nodeBase + node] = value;
    }
    const uint edgeBase = environment * dispatch.rodEdgeCount;
    for (uint edge = 0u; edge < dispatch.rodEdgeCount; ++edge) {
        const MRRodEdgeStateGPU value =
            sourceEdges[edgeBase + edge];
        edgesA[edgeBase + edge] = value;
        edgesB[edgeBase + edge] = value;
    }
    const uint witnessStride =
        dispatch.rodToolPairCount *
        MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
    const uint witnessBase = environment * witnessStride;
    for (uint witness = 0u; witness < witnessStride; ++witness) {
        const MRRodToolWitnessGPU value =
            sourceWitnesses[witnessBase + witness];
        witnessesA[witnessBase + witness] = value;
        witnessesB[witnessBase + witness] = value;
    }
}

// Fixed MLX event graphs continue dispatching after an environment has
// consumed its full microstep. Restore that environment's accepted rod state
// and cache into the ordinary candidate buffers so later commit is branch-free.
kernel void mr_world_restore_inactive_rod_event_candidate(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRRodNodeStateGPU* sourceNodes [[buffer(1)]],
    device const MRRodEdgeStateGPU* sourceEdges [[buffer(2)]],
    device const MRRodToolWitnessGPU* sourceWitnesses [[buffer(3)]],
    device MRRodNodeStateGPU* candidateNodes [[buffer(4)]],
    device MRRodEdgeStateGPU* candidateEdges [[buffer(5)]],
    device MRRodToolWitnessGPU* candidateWitnesses [[buffer(6)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(7)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount ||
        statuses[environment].code != MR_STEP_FIXED_BUDGET_COMPLETE) {
        return;
    }
    const uint nodeBase = environment * dispatch.rodNodeCount;
    for (uint node = 0u; node < dispatch.rodNodeCount; ++node) {
        candidateNodes[nodeBase + node] =
            sourceNodes[nodeBase + node];
    }
    const uint edgeBase = environment * dispatch.rodEdgeCount;
    for (uint edge = 0u; edge < dispatch.rodEdgeCount; ++edge) {
        candidateEdges[edgeBase + edge] =
            sourceEdges[edgeBase + edge];
    }
    const uint witnessStride =
        dispatch.rodToolPairCount *
        MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
    const uint witnessBase = environment * witnessStride;
    for (uint witness = 0u; witness < witnessStride; ++witness) {
        candidateWitnesses[witnessBase + witness] =
            sourceWitnesses[witnessBase + witness];
    }
}

// Advances the private rod event-state ping-pong. A failed event candidate
// copies the last accepted source, matching the rigid event transaction.
kernel void mr_world_publish_rod_event_segment(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRRodNodeStateGPU* sourceNodes [[buffer(1)]],
    device const MRRodEdgeStateGPU* sourceEdges [[buffer(2)]],
    device const MRRodToolWitnessGPU* sourceWitnesses [[buffer(3)]],
    device const MRRodNodeStateGPU* candidateNodes [[buffer(4)]],
    device const MRRodEdgeStateGPU* candidateEdges [[buffer(5)]],
    device const MRRodToolWitnessGPU* candidateWitnesses [[buffer(6)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(7)]],
    device MRRodNodeStateGPU* destinationNodes [[buffer(8)]],
    device MRRodEdgeStateGPU* destinationEdges [[buffer(9)]],
    device MRRodToolWitnessGPU* destinationWitnesses [[buffer(10)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    const bool publish =
        statuses[environment].code == MR_STEP_SUCCESS;
    const uint nodeBase = environment * dispatch.rodNodeCount;
    for (uint node = 0u; node < dispatch.rodNodeCount; ++node) {
        destinationNodes[nodeBase + node] = publish
            ? candidateNodes[nodeBase + node]
            : sourceNodes[nodeBase + node];
    }
    const uint edgeBase = environment * dispatch.rodEdgeCount;
    for (uint edge = 0u; edge < dispatch.rodEdgeCount; ++edge) {
        destinationEdges[edgeBase + edge] = publish
            ? candidateEdges[edgeBase + edge]
            : sourceEdges[edgeBase + edge];
    }
    const uint witnessStride =
        dispatch.rodToolPairCount *
        MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
    const uint witnessBase = environment * witnessStride;
    for (uint witness = 0u; witness < witnessStride; ++witness) {
        destinationWitnesses[witnessBase + witness] = publish
            ? candidateWitnesses[witnessBase + witness]
            : sourceWitnesses[witnessBase + witness];
    }
}

kernel void mr_world_pack_rod_state(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRRodNodeStateGPU* nodes [[buffer(1)]],
    device const MRRodEdgeStateGPU* edges [[buffer(2)]],
    device float4* positions [[buffer(3)]],
    device float4* velocities [[buffer(4)]],
    device float* twists [[buffer(5)]],
    device float* twistRates [[buffer(6)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    const uint nodeBase =
        environment * dispatch.rodNodeCount;
    for (uint node = 0u;
         node < dispatch.rodNodeCount;
         ++node) {
        const MRRodNodeStateGPU state =
            nodes[nodeBase + node];
        positions[nodeBase + node] = state.position;
        velocities[nodeBase + node] = state.velocity;
    }
    const uint edgeBase =
        environment * dispatch.rodEdgeCount;
    for (uint edge = 0u;
         edge < dispatch.rodEdgeCount;
         ++edge) {
        const float4 state =
            edges[edgeBase + edge].twistAndRate;
        twists[edgeBase + edge] = state.x;
        twistRates[edgeBase + edge] = state.y;
    }
}

kernel void mr_world_unpack_rod_state(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const float4* positions [[buffer(1)]],
    device const float4* velocities [[buffer(2)]],
    device const float* twists [[buffer(3)]],
    device const float* twistRates [[buffer(4)]],
    device MRRodNodeStateGPU* nodes [[buffer(5)]],
    device MRRodEdgeStateGPU* edges [[buffer(6)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    const uint nodeBase =
        environment * dispatch.rodNodeCount;
    for (uint node = 0u;
         node < dispatch.rodNodeCount;
         ++node) {
        MRRodNodeStateGPU state;
        state.position = positions[nodeBase + node];
        state.velocity = velocities[nodeBase + node];
        nodes[nodeBase + node] = state;
    }
    const uint edgeBase =
        environment * dispatch.rodEdgeCount;
    for (uint edge = 0u;
         edge < dispatch.rodEdgeCount;
         ++edge) {
        MRRodEdgeStateGPU state;
        state.twistAndRate = float4(
            twists[edgeBase + edge],
            twistRates[edgeBase + edge],
            0.0f,
            0.0f
        );
        edges[edgeBase + edge] = state;
    }
}

kernel void mr_world_latch_rod_status(
    device const MRMetalWorldDispatchGPU& worldDispatch [[buffer(0)]],
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(1)]],
    constant MRMetalWorldPassGPU& pass [[buffer(2)]],
    device const MRRodGPUStatus* rodStatuses [[buffer(3)]],
    device MRMetalWorldStatusGPU* statuses [[buffer(4)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= worldDispatch.environmentCount) {
        return;
    }
    MRMetalWorldStatusGPU status = statuses[environment];
    if (status.code != MR_STEP_SUCCESS) {
        return;
    }
    for (uint rod = 0u;
         rod < contactDispatch.rodCount;
         ++rod) {
        const MRRodGPUStatus rodStatus =
            rodStatuses[
                rod * worldDispatch.environmentCount +
                environment
            ];
        if (rodStatus.environment != environment ||
            rodStatus.code != MR_ROD_GPU_SUCCESS) {
            status.code =
                rodStatus.environment == environment
                ? mapRodStatus(rodStatus.code)
                : uint(MR_STEP_UNSUPPORTED);
            status.failingSubstep = pass.physicsSubstep;
            status.failingIndex =
                rodStatus.environment == environment
                ? rodStatus.failingIndex
                : MR_INVALID_INDEX;
            statuses[environment] = status;
            return;
        }
    }
}

// Rod mechanics participates in the same per-environment transaction as
// collision and constraints. Mirror a DER failure into the contact status so
// an event segment cannot become the accepted CCD source merely because the
// rigid side remained healthy.
kernel void mr_world_latch_rod_contact_status(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    constant MRMetalWorldPassGPU& pass [[buffer(1)]],
    device const MRRodGPUStatus* rodStatuses [[buffer(2)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(3)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    if (status.code != MR_STEP_SUCCESS) {
        return;
    }
    for (uint rod = 0u; rod < dispatch.rodCount; ++rod) {
        const MRRodGPUStatus rodStatus =
            rodStatuses[
                rod * dispatch.environmentCount + environment
            ];
        if (rodStatus.environment == environment &&
            rodStatus.code == MR_ROD_GPU_SUCCESS) {
            continue;
        }
        status.code =
            rodStatus.environment == environment
            ? mapRodStatus(rodStatus.code)
            : uint(MR_STEP_UNSUPPORTED);
        status.controlStep = pass.controlStep;
        status.physicsSubstep = pass.physicsSubstep;
        status.firstFailingConstraint =
            rodStatus.environment == environment
            ? rodStatus.failingIndex
            : MR_INVALID_INDEX;
        status.firstFailingStableKeyLow =
            status.firstFailingConstraint;
        status.firstFailingStableKeyHigh =
            0x524f4400u | (rod & 0xffu);
        status.firstFailingEventKeyLow =
            status.firstFailingStableKeyLow;
        status.firstFailingEventKeyHigh =
            status.firstFailingStableKeyHigh;
        status.solverIterations = rodStatus.iterations;
        status.residuals = rodStatus.diagnostics;
        statuses[environment] = status;
        return;
    }
}

kernel void mr_world_commit_rod_state(
    device const MRMetalWorldDispatchGPU& worldDispatch [[buffer(0)]],
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(1)]],
    constant MRMetalWorldPassGPU& pass [[buffer(2)]],
    device const MRMetalWorldStatusGPU* statuses [[buffer(3)]],
    device const MRRodNodeStateGPU* candidateNodes [[buffer(4)]],
    device const MRRodEdgeStateGPU* candidateEdges [[buffer(5)]],
    device const MRRodNodeStateGPU* checkpointNodes [[buffer(6)]],
    device const MRRodEdgeStateGPU* checkpointEdges [[buffer(7)]],
    device MRRodNodeStateGPU* destinationNodes [[buffer(8)]],
    device MRRodEdgeStateGPU* destinationEdges [[buffer(9)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= worldDispatch.environmentCount) {
        return;
    }
    const bool publish =
        pass.physicsSubstep < worldDispatch.physicsSubsteps &&
        statuses[environment].code == MR_STEP_SUCCESS;
    const uint nodeBase =
        environment * contactDispatch.rodNodeCount;
    for (uint node = 0u;
         node < contactDispatch.rodNodeCount;
         ++node) {
        destinationNodes[nodeBase + node] = publish
            ? candidateNodes[nodeBase + node]
            : checkpointNodes[nodeBase + node];
    }
    const uint edgeBase =
        environment * contactDispatch.rodEdgeCount;
    for (uint edge = 0u;
         edge < contactDispatch.rodEdgeCount;
         ++edge) {
        destinationEdges[edgeBase + edge] = publish
            ? candidateEdges[edgeBase + edge]
            : checkpointEdges[edgeBase + edge];
    }
}

kernel void mr_world_commit_rod_contact_cache(
    device const MRMetalWorldDispatchGPU& worldDispatch [[buffer(0)]],
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(1)]],
    constant MRMetalWorldPassGPU& pass [[buffer(2)]],
    device const MRMetalWorldStatusGPU* statuses [[buffer(3)]],
    device const MRRodToolWitnessGPU* candidate [[buffer(4)]],
    device const MRRodToolWitnessGPU* checkpoint [[buffer(5)]],
    device MRRodToolWitnessGPU* destination [[buffer(6)]],
    const uint flatWitness [[thread_position_in_grid]]
) {
    const uint witnessStride =
        contactDispatch.rodToolPairCount *
        MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
    const uint witnessCount =
        worldDispatch.environmentCount * witnessStride;
    if (flatWitness >= witnessCount || witnessStride == 0u) {
        return;
    }
    const uint environment = flatWitness / witnessStride;
    const bool publish =
        pass.physicsSubstep < worldDispatch.physicsSubsteps &&
        statuses[environment].code == MR_STEP_SUCCESS;
    destination[flatWitness] = publish
        ? candidate[flatWitness]
        : checkpoint[flatWitness];
}
