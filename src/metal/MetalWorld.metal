#include <metal_stdlib>

#include "metalrobo/engine_types.h"
#include "metalrobo/rod_gpu_shared.h"
#include "metalrobo/runtime_abi_generated.h"

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
        MR_METAL_WORLD_NATIVE_TASK |
        MR_METAL_WORLD_NATIVE_SENSORS |
        MR_METAL_WORLD_INITIALIZE_ACTUATORS;
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

// TaskIR must expose reset state to pre-policy kinematics, but that state is
// not committed until physics succeeds. Snapshot the resident physical state
// before any task reset, push, or scene randomization mutates its working slot.
kernel void mr_world_checkpoint_task_state(
    device const MRMetalWorldDispatchGPU& dispatch
        [[buffer(MR_TASK_PHYSICAL_CHECKPOINT_WORLD_DISPATCH)]],
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(MR_TASK_PHYSICAL_CHECKPOINT_CONTACT_DISPATCH)]],
    constant MRMetalWorldPassGPU& pass
        [[buffer(MR_TASK_PHYSICAL_CHECKPOINT_PASS)]],
    device const float* stateQ
        [[buffer(MR_TASK_PHYSICAL_CHECKPOINT_STATE_Q)]],
    device const float* stateV
        [[buffer(MR_TASK_PHYSICAL_CHECKPOINT_STATE_V)]],
    device const MRBodyStateGPU* sceneState
        [[buffer(MR_TASK_PHYSICAL_CHECKPOINT_SCENE_STATE)]],
    device float* checkpointQ
        [[buffer(MR_TASK_PHYSICAL_CHECKPOINT_CHECKPOINT_Q)]],
    device float* checkpointV
        [[buffer(MR_TASK_PHYSICAL_CHECKPOINT_CHECKPOINT_V)]],
    device MRBodyStateGPU* checkpointSceneState
        [[buffer(MR_TASK_PHYSICAL_CHECKPOINT_CHECKPOINT_SCENE_STATE)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount ||
        pass.controlStep >= dispatch.controlStepCount ||
        pass.physicsSubstep != MR_INVALID_INDEX) {
        return;
    }
    const uint qBase = environment * dispatch.qStride;
    for (uint coordinate = 0u;
         coordinate < dispatch.nq;
         ++coordinate) {
        checkpointQ[qBase + coordinate] =
            stateQ[qBase + coordinate];
    }
    const uint vBase = environment * dispatch.vStride;
    for (uint coordinate = 0u;
         coordinate < dispatch.nv;
         ++coordinate) {
        checkpointV[vBase + coordinate] =
            stateV[vBase + coordinate];
    }
    const uint sceneBase =
        environment * contactDispatch.sceneBodyStride;
    for (uint body = 0u;
         body < contactDispatch.sceneBodyCount;
         ++body) {
        checkpointSceneState[sceneBase + body] =
            sceneState[sceneBase + body];
    }
}

// Starts one transactional control step. The checkpoint is always the last
// committed resident state. Reset is applied only to the working source, so a
// rejected transition restores the pre-reset state byte-for-byte.
kernel void mr_metal_world_prepare(
    device const MRMetalWorldDispatchGPU& dispatch
        [[buffer(MR_WORLD_PREPARE_DISPATCH)]],
    constant MRMetalWorldPassGPU& pass
        [[buffer(MR_WORLD_PREPARE_PASS)]],
    device const float* effortTrajectory
        [[buffer(MR_WORLD_PREPARE_EFFORT_TRAJECTORY)]],
    device const uint* resetMasks
        [[buffer(MR_WORLD_PREPARE_RESET_MASKS)]],
    device const float* resetQ
        [[buffer(MR_WORLD_PREPARE_RESET_Q)]],
    device const float* resetV
        [[buffer(MR_WORLD_PREPARE_RESET_V)]],
    device float* stateQ
        [[buffer(MR_WORLD_PREPARE_STATE_Q)]],
    device float* stateV
        [[buffer(MR_WORLD_PREPARE_STATE_V)]],
    device float* checkpointQ
        [[buffer(MR_WORLD_PREPARE_CHECKPOINT_Q)]],
    device float* checkpointV
        [[buffer(MR_WORLD_PREPARE_CHECKPOINT_V)]],
    device float* workingEffort
        [[buffer(MR_WORLD_PREPARE_WORKING_EFFORT)]],
    device MRMetalWorldStatusGPU* statuses
        [[buffer(MR_WORLD_PREPARE_STATUSES)]],
    device const MRWorldGPU& world
        [[buffer(MR_WORLD_PREPARE_WORLD)]],
    device const MRArticulationGPU* articulations
        [[buffer(MR_WORLD_PREPARE_ARTICULATIONS)]],
    device const MRDofPropertiesGPU* dofs
        [[buffer(MR_WORLD_PREPARE_DOFS)]],
    device const MRActuatorProfileGPU* actuatorProfiles
        [[buffer(MR_WORLD_PREPARE_ACTUATOR_PROFILES)]],
    device const float4* taskControllerParameters
        [[buffer(MR_WORLD_PREPARE_TASK_CONTROLLER_PARAMETERS)]],
    device MRActuatorRuntimeStateGPU* actuatorStates
        [[buffer(MR_WORLD_PREPARE_ACTUATOR_STATES)]],
    device float* actuatorCommandHistory
        [[buffer(MR_WORLD_PREPARE_ACTUATOR_COMMAND_HISTORY)]],
    device MRActuatorRuntimeStateGPU* checkpointActuatorStates
        [[buffer(MR_WORLD_PREPARE_CHECKPOINT_ACTUATOR_STATES)]],
    device float* checkpointActuatorCommandHistory
        [[buffer(MR_WORLD_PREPARE_CHECKPOINT_ACTUATOR_COMMAND_HISTORY)]],
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
    const bool nativeTask =
        (dispatch.flags & MR_METAL_WORLD_NATIVE_TASK) != 0u;
    const bool implicitPositionDrive =
        (dispatch.flags &
         MR_METAL_WORLD_IMPLICIT_POSITION_DRIVES) != 0u;
    const bool initializeActuators =
        pass.controlStep == 0u &&
        (dispatch.flags &
         MR_METAL_WORLD_INITIALIZE_ACTUATORS) != 0u;
    for (uint coordinate = 0u;
         coordinate < dispatch.nq;
         ++coordinate) {
        const float committed = stateQ[qBase + coordinate];
        if (!nativeTask) {
            checkpointQ[qBase + coordinate] = committed;
        }
        const float value = applyReset
            ? resetQ[qBase + coordinate]
            : committed;
        stateQ[qBase + coordinate] = value;
    }
    for (uint coordinate = 0u;
         coordinate < dispatch.nv;
         ++coordinate) {
        const float committed = stateV[vBase + coordinate];
        if (!nativeTask) {
            checkpointV[vBase + coordinate] = committed;
        }
        const float value = applyReset
            ? resetV[vBase + coordinate]
            : committed;
        stateV[vBase + coordinate] = value;
        float rawCommand = effortTrajectory[
            pass.controlStep * dispatch.effortStepStride +
            environment * dispatch.effortEnvironmentStride +
            coordinate
        ];
        device const MRActuatorProfileGPU& actuator =
            actuatorProfiles[coordinate];
        device const MRDofPropertiesGPU& dof = dofs[coordinate];
        const uint historySlots = actuator.identity.w;
        const uint delaySteps = actuator.identity.z;
        if (actuator.identity.x != coordinate ||
            historySlots == 0u || delaySteps >= historySlots) {
            status.code = MR_STEP_UNSUPPORTED;
            status.abaCode = MR_ABA_INVALID_MODEL;
            status.failingIndex = coordinate;
            workingEffort[
                environment * dispatch.effortEnvironmentStride +
                coordinate
            ] = 0.0f;
            continue;
        }
        const bool driven =
            (dof.flags & MR_DOF_FLAG_DRIVE) != 0u &&
            dof.qIndex != MR_INVALID_INDEX &&
            dof.qIndex < dispatch.nq;
        const bool actuated =
            (dof.flags & MR_DOF_FLAG_ACTUATED) != 0u;
        float neutralCommand = 0.0f;
        if (implicitPositionDrive && driven) {
            neutralCommand = stateQ[qBase + dof.qIndex];
            if ((dof.flags & MR_DOF_FLAG_POSITION_LIMIT) != 0u) {
                rawCommand = clamp(
                    rawCommand,
                    dof.limits.x,
                    dof.limits.y
                );
            }
        } else if (!actuated) {
            rawCommand = 0.0f;
        }

        const uint actuatorStateIndex =
            environment * dispatch.nv + coordinate;
        MRActuatorRuntimeStateGPU actuatorState =
            actuatorStates[actuatorStateIndex];
        MRActuatorRuntimeStateGPU neutralState{};
        neutralState.command = float4(
            neutralCommand,
            neutralCommand,
            neutralCommand,
            neutralCommand
        );
        neutralState.status = uint4(
            1u,
            0u,
            0u,
            implicitPositionDrive ? 1u : 0u
        );
        checkpointActuatorStates[actuatorStateIndex] =
            initializeActuators ? neutralState : actuatorState;
        const bool resetActuator = initializeActuators || applyReset;
        if (resetActuator || actuatorState.status.x != 1u) {
            actuatorState = neutralState;
        }
        for (uint slot = 0u; slot < historySlots; ++slot) {
            const ulong historyIndex =
                (static_cast<ulong>(environment) * historySlots +
                 slot) * dispatch.nv + coordinate;
            const float previous = initializeActuators
                ? neutralCommand
                : actuatorCommandHistory[historyIndex];
            checkpointActuatorCommandHistory[historyIndex] =
                previous;
            if (resetActuator) {
                actuatorCommandHistory[historyIndex] =
                    neutralCommand;
            }
        }
        const uint writeSlot = actuatorState.status.y < historySlots
            ? actuatorState.status.y
            : 0u;
        const ulong writeIndex =
            (static_cast<ulong>(environment) * historySlots +
             writeSlot) * dispatch.nv + coordinate;
        actuatorCommandHistory[writeIndex] = rawCommand;
        const uint delayedSlot =
            (writeSlot + historySlots - delaySteps) % historySlots;
        const ulong delayedIndex =
            (static_cast<ulong>(environment) * historySlots +
             delayedSlot) * dispatch.nv + coordinate;
        const float delayedCommand =
            actuatorCommandHistory[delayedIndex];
        float effectiveCommand = delayedCommand;
        if (implicitPositionDrive && driven) {
            const float halfPlay =
                0.5f * max(
                    actuator.transmissionAndEnvelope.x,
                    0.0f
                );
            effectiveCommand = clamp(
                actuatorState.command.z,
                delayedCommand - halfPlay,
                delayedCommand + halfPlay
            );
        }

        const float4 controller = nativeTask
            ? taskControllerParameters[environment]
            : float4(1.0f);
        float unclampedMotorEffort = 0.0f;
        if (implicitPositionDrive && driven && actuated) {
            const float timestep = world.gravityAndTimestep.w;
            unclampedMotorEffort = controller.w * (
                controller.x * dof.drive.x *
                    (
                        effectiveCommand -
                        stateQ[qBase + dof.qIndex] -
                        timestep * value
                    ) -
                controller.y * dof.drive.y * value
            );
        } else if (!implicitPositionDrive && actuated) {
            unclampedMotorEffort = effectiveCommand;
        }
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
            (dof.flags & MR_DOF_FLAG_EFFORT_LIMIT) != 0u &&
                    dof.limits.w > 0.0f
                ? dof.limits.w
                : 0.0f,
            actuator.transmissionAndEnvelope.z *
                actuator.motorAndSpeed.w *
                (1.0f - speedFraction)
        );
        const float motorEffort = clamp(
            unclampedMotorEffort,
            -envelope,
            envelope
        );
        float passiveFriction = 0.0f;
        if (actuated && dof.drive.w > 0.0f) {
            passiveFriction = abs(value) > 1.0e-4f
                ? -copysign(dof.drive.w, value)
                : -clamp(
                      motorEffort,
                      -dof.drive.w,
                      dof.drive.w
                  );
        }
        const float generalizedEffort =
            motorEffort + passiveFriction;
        workingEffort[
            environment * dispatch.effortEnvironmentStride +
            coordinate
        ] = generalizedEffort;
        actuatorState.command = float4(
            rawCommand,
            delayedCommand,
            effectiveCommand,
            neutralCommand
        );
        actuatorState.effort = float4(
            unclampedMotorEffort,
            motorEffort,
            passiveFriction,
            generalizedEffort
        );
        actuatorState.envelope = float4(
            envelope,
            max(abs(unclampedMotorEffort) - envelope, 0.0f),
            speedFraction,
            controller.w
        );
        actuatorState.status = uint4(
            1u,
            (writeSlot + 1u) % historySlots,
            abs(unclampedMotorEffort) > envelope ? 1u : 0u,
            implicitPositionDrive ? 1u : 0u
        );
        actuatorStates[actuatorStateIndex] = actuatorState;
    }
    statuses[environment] = status;
}

// Commits one ABA candidate into the ping-pong state. The first failed
// substep latches a typed status and every later pass restores the immutable
// checkpoint, making the entire control step transactional per environment.
kernel void mr_metal_world_commit(
    device const MRMetalWorldDispatchGPU& dispatch
        [[buffer(MR_WORLD_COMMIT_DISPATCH)]],
    constant MRMetalWorldPassGPU& pass
        [[buffer(MR_WORLD_COMMIT_PASS)]],
    device const MRABAStatusGPU* abaStatuses
        [[buffer(MR_WORLD_COMMIT_ABA_STATUSES)]],
    device const float* candidateQ
        [[buffer(MR_WORLD_COMMIT_CANDIDATE_Q)]],
    device const float* candidateV
        [[buffer(MR_WORLD_COMMIT_CANDIDATE_V)]],
    device float* destinationQ
        [[buffer(MR_WORLD_COMMIT_DESTINATION_Q)]],
    device float* destinationV
        [[buffer(MR_WORLD_COMMIT_DESTINATION_V)]],
    device MRMetalWorldStatusGPU* statuses
        [[buffer(MR_WORLD_COMMIT_STATUSES)]],
    device const float* checkpointQ
        [[buffer(MR_WORLD_COMMIT_CHECKPOINT_Q)]],
    device const float* checkpointV
        [[buffer(MR_WORLD_COMMIT_CHECKPOINT_V)]],
    device const MRWorldGPU& world
        [[buffer(MR_WORLD_COMMIT_WORLD)]],
    device MRActuatorRuntimeStateGPU* actuatorStates
        [[buffer(MR_WORLD_COMMIT_ACTUATOR_STATES)]],
    device float* actuatorCommandHistory
        [[buffer(MR_WORLD_COMMIT_ACTUATOR_COMMAND_HISTORY)]],
    device const MRActuatorRuntimeStateGPU* checkpointActuatorStates
        [[buffer(MR_WORLD_COMMIT_CHECKPOINT_ACTUATOR_STATES)]],
    device const float* checkpointActuatorCommandHistory
        [[buffer(MR_WORLD_COMMIT_CHECKPOINT_ACTUATOR_COMMAND_HISTORY)]],
    device const MRActuatorProfileGPU* actuatorProfiles
        [[buffer(MR_WORLD_COMMIT_ACTUATOR_PROFILES)]],
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
    if (!commitCandidate) {
        const uint historySlots = actuatorProfiles[0].identity.w;
        for (uint coordinate = 0u;
             coordinate < dispatch.nv;
             ++coordinate) {
            const uint actuatorStateIndex =
                environment * dispatch.nv + coordinate;
            actuatorStates[actuatorStateIndex] =
                checkpointActuatorStates[actuatorStateIndex];
            for (uint slot = 0u; slot < historySlots; ++slot) {
                const ulong historyIndex =
                    (static_cast<ulong>(environment) * historySlots +
                     slot) * dispatch.nv + coordinate;
                actuatorCommandHistory[historyIndex] =
                    checkpointActuatorCommandHistory[historyIndex];
            }
        }
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
        const MRRodNodeStateGPU committed =
            stateNodes[nodeBase + node];
        checkpointNodes[nodeBase + node] = committed;
        const MRRodNodeStateGPU value = applyReset
            ? resetNodes[nodeBase + node]
            : committed;
        stateNodes[nodeBase + node] = value;
    }
    const uint edgeBase =
        environment * contactDispatch.rodEdgeCount;
    for (uint edge = 0u;
         edge < contactDispatch.rodEdgeCount;
         ++edge) {
        const MRRodEdgeStateGPU committed =
            stateEdges[edgeBase + edge];
        checkpointEdges[edgeBase + edge] = committed;
        const MRRodEdgeStateGPU value = applyReset
            ? resetEdges[edgeBase + edge]
            : committed;
        stateEdges[edgeBase + edge] = value;
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
    const MRRodToolWitnessGPU committed =
        published[flatWitness];
    checkpoint[flatWitness] = committed;
    MRRodToolWitnessGPU working = {};
    if (!applyReset) {
        working = committed;
    }
    candidate[flatWitness] = working;
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
