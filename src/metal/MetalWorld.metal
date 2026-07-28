#include <metal_stdlib>

#include "metalrobo/engine_types.h"

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
        MR_METAL_WORLD_IMPLICIT_POSITION_DRIVES;
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
    const MRArticulationGPU articulation =
        articulations[dispatch.articulationIndex];
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
            device const MRDofPropertiesGPU& dof =
                dofs[articulation.vOffset + coordinate];
            command = 0.0f;
            if ((dof.flags & MR_DOF_FLAG_DRIVE) != 0u &&
                dof.qIndex != MR_INVALID_INDEX &&
                dof.qIndex >= articulation.qOffset &&
                dof.qIndex <
                    articulation.qOffset + articulation.nq) {
                const uint localQ =
                    dof.qIndex - articulation.qOffset;
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
                    dof.drive.x *
                        (
                            target -
                            stateQ[qBase + localQ] -
                            timestep * value
                        ) -
                    dof.drive.y * value;
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
            }
        }
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
    uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }

    MRMetalWorldStatusGPU status = statuses[environment];
    const MRABAStatusGPU aba = abaStatuses[environment];
    const bool validPass =
        validWorldDispatch(dispatch) &&
        pass.controlStep < dispatch.controlStepCount &&
        pass.physicsSubstep < dispatch.physicsSubsteps &&
        pass.reserved0 == 0u &&
        pass.reserved1 == 0u;
    const bool validABARecord =
        aba.environment == environment &&
        aba.articulationIndex == dispatch.articulationIndex &&
        aba.code <= MR_ABA_UNSUPPORTED_TOPOLOGY;
    const bool commitCandidate =
        validPass &&
        status.code == MR_STEP_SUCCESS &&
        validABARecord &&
        aba.code == MR_ABA_SUCCESS;

    if (!commitCandidate && status.code == MR_STEP_SUCCESS) {
        status.code = validPass && validABARecord
            ? mapABAStatus(aba.code)
            : MR_STEP_UNSUPPORTED;
        status.abaCode = validABARecord
            ? aba.code
            : static_cast<uint>(MR_ABA_INVALID_DISPATCH);
        status.failingSubstep = pass.physicsSubstep;
        status.failingIndex = validABARecord
            ? aba.failingIndex
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
            status.diagnostics = aba.diagnostics;
        } else {
            status.diagnostics.x = min(
                status.diagnostics.x,
                aba.diagnostics.x
            );
            status.diagnostics.y = max(
                status.diagnostics.y,
                aba.diagnostics.y
            );
            status.diagnostics.z = max(
                status.diagnostics.z,
                aba.diagnostics.z
            );
            status.diagnostics.w = max(
                status.diagnostics.w,
                aba.diagnostics.w
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
