#include <metal_stdlib>
using namespace metal;

#include "metalrobo/numanx_human_io_gpu.h"

namespace {

inline bool finite4(const float4 value) {
    return all(isfinite(value));
}

constant constexpr ulong kFnvOffset = 14695981039346656037ul;
constant constexpr ulong kFnvPrime = 1099511628211ul;

inline void mixU32(thread ulong& hash, const uint value) {
    for (uint byte = 0u; byte < 4u; ++byte) {
        hash ^= static_cast<ulong>((value >> (byte * 8u)) & 0xffu);
        hash *= kFnvPrime;
    }
}

inline void mixU64(thread ulong& hash, const ulong value) {
    for (uint byte = 0u; byte < 8u; ++byte) {
        hash ^= (value >> (byte * 8u)) & 0xfful;
        hash *= kFnvPrime;
    }
}

inline void mixFloat(thread ulong& hash, const float value) {
    mixU32(hash, as_type<uint>(value));
}

inline ulong readyGateFingerprint(
    const device MRNumanXBrainMotorReadyGateGPU& gate
) {
    const device uchar* bytes =
        reinterpret_cast<const device uchar*>(&gate);
    ulong hash = kFnvOffset;
    for (uint index = 0u; index < 152u; ++index) {
        hash ^= static_cast<ulong>(bytes[index]);
        hash *= kFnvPrime;
    }
    return hash == 0ul ? kFnvOffset : hash;
}

inline bool validReceptorSource(
    const device MRMujocoMuscleStateGPU& state,
    const device MRMujocoMuscleResultGPU& result,
    const uint environment,
    const uint muscle
) {
    return result.status == MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS &&
        result.environment == environment &&
        result.muscleIndex == muscle &&
        finite4(state.excitationAndActivation) &&
        finite4(result.pathForceAndActivationDerivative) &&
        finite4(result.activeForceAndReserved) &&
        finite4(result.fiberStateTendonForceResidual);
}

} // namespace

// One thread authenticates one complete environment motor output. This is a
// deliberate serial fold: it is byte-for-byte equivalent to NumiBrain's
// NBMotorOutputHeader + FP32 payload FNV-1a contract and therefore avoids an
// order-dependent parallel hash reduction. The following admission encoder is
// command-ordered after this kernel.
kernel void numanx_human_validate_motor_output(
    const device MRNumanXBrainMotorOutputHeaderGPU* headers [[buffer(0)]],
    const device float* excitations [[buffer(1)]],
    device uint* headerValidation [[buffer(2)]],
    constant MRNumanXHumanMotorDispatchGPU& dispatch [[buffer(3)]],
    const device MRNumanXBrainMotorReadyGateGPU* motorReadyGate [[buffer(4)]],
    uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }

    const bool decisionShadow =
        (dispatch.flags &
         MR_NUMANX_BRAIN_MOTOR_CANDIDATE_DECISION_SHADOW) != 0u;
    const device MRNumanXBrainMotorReadyGateGPU& ready = motorReadyGate[0];
    if (decisionShadow &&
        (ready.abiVersion != MR_NUMANX_BRAIN_MOTOR_READY_ABI_VERSION ||
         ready.structBytes != MR_NUMANX_BRAIN_MOTOR_READY_GATE_BYTE_COUNT ||
         ready.status != MR_NUMANX_BRAIN_READY_GATE_SUCCESS ||
         ready.environment != dispatch.environmentIdentifierBase ||
         ready.substepIndex != 0u ||
         ready.muscleCount != dispatch.muscleCount ||
         ready.actuatorCommandKind != dispatch.actuatorCommandKind ||
         ready.transactionFingerprint != dispatch.transactionFingerprint ||
         ready.substepFingerprint == 0ul ||
         ready.candidateFingerprint != dispatch.motorCandidateFingerprint ||
         ready.motorOutputFingerprint == 0ul ||
         ready.motorProfileFingerprint != dispatch.motorProfileFingerprint ||
         ready.brainGeneration != dispatch.acceptedBrainGeneration ||
         ready.acceptedBrainTimestampMicroseconds !=
             dispatch.acceptedBrainTimestampMicroseconds ||
         ready.randomCounterGeneration == 0ul ||
         ready.speciesTemplateFingerprint == 0ul ||
         ready.compiledSpeciesTemplateFingerprint == 0ul ||
         ready.brainProgramFingerprint == 0ul ||
         ready.fastProgramFingerprint == 0ul ||
         ready.decisionGateFingerprint == 0ul ||
         ready.reserved64_0 != 0ul ||
         ready.gateFingerprint == 0ul ||
         ready.gateFingerprint != readyGateFingerprint(ready))) {
        headerValidation[environment] =
            MR_NUMANX_HUMAN_MOTOR_HEADER_READY_GATE;
        return;
    }
    headerValidation[environment] = MR_NUMANX_HUMAN_MOTOR_HEADER_PENDING;
    if (dispatch.abiVersion != MR_NUMANX_HUMAN_IO_ABI_VERSION ||
        dispatch.motorOutputFormatVersion !=
            MR_NUMANX_BRAIN_MOTOR_OUTPUT_VERSION ||
        dispatch.actuatorCommandKind !=
            MR_NUMANX_BRAIN_ACTUATOR_MUSCLE_EXCITATION ||
        dispatch.headerEnvironmentStride != 1u) {
        headerValidation[environment] =
            MR_NUMANX_HUMAN_MOTOR_HEADER_FORMAT;
        return;
    }

    const device MRNumanXBrainMotorOutputHeaderGPU& header =
        headers[environment * dispatch.headerEnvironmentStride];
    if (header.formatVersion != dispatch.motorOutputFormatVersion) {
        headerValidation[environment] =
            MR_NUMANX_HUMAN_MOTOR_HEADER_FORMAT;
        return;
    }
    if ((header.flags & MR_NUMANX_BRAIN_MOTOR_OUTPUT_VALID) == 0u ||
        (header.flags & ~MR_NUMANX_BRAIN_MOTOR_OUTPUT_KNOWN_FLAGS) != 0u) {
        headerValidation[environment] =
            MR_NUMANX_HUMAN_MOTOR_HEADER_FLAGS;
        return;
    }
    if (header.muscleCount != dispatch.muscleCount ||
        header.environmentIdentifier !=
            dispatch.environmentIdentifierBase + environment ||
        header.profileFingerprint != dispatch.motorProfileFingerprint ||
        header.profileFingerprint == 0ul ||
        header.protectiveCommandFingerprint == 0ul) {
        headerValidation[environment] =
            MR_NUMANX_HUMAN_MOTOR_HEADER_IDENTITY;
        return;
    }
    if (header.timestampMicroseconds !=
            dispatch.acceptedBrainTimestampMicroseconds ||
        header.brainGeneration != dispatch.acceptedBrainGeneration ||
        (header.brainGeneration == 0ul &&
            header.timestampMicroseconds != 0ul)) {
        headerValidation[environment] =
            MR_NUMANX_HUMAN_MOTOR_HEADER_GENERATION;
        return;
    }
    if (!isfinite(header.motorInhibition) ||
        !isfinite(header.autonomicArousal) ||
        !isfinite(header.outputMinimum) ||
        !isfinite(header.outputMaximum)) {
        headerValidation[environment] =
            MR_NUMANX_HUMAN_MOTOR_HEADER_NONFINITE;
        return;
    }
    if (header.motorInhibition < 0.0f ||
        header.motorInhibition > 1.0f ||
        header.autonomicArousal < 0.0f ||
        header.autonomicArousal > 1.0f ||
        !(header.outputMinimum < header.outputMaximum)) {
        headerValidation[environment] =
            MR_NUMANX_HUMAN_MOTOR_HEADER_RANGE;
        return;
    }
    if (header.actuatorCommandKind != dispatch.actuatorCommandKind ||
        header.actuatorCommandKind !=
            MR_NUMANX_BRAIN_ACTUATOR_MUSCLE_EXCITATION ||
        header.reserved != 0u) {
        headerValidation[environment] =
            MR_NUMANX_HUMAN_MOTOR_HEADER_COMMAND_KIND;
        return;
    }
    const bool emergency =
        (header.flags & MR_NUMANX_BRAIN_MOTOR_OUTPUT_EMERGENCY_STOP) != 0u;
    if (emergency != (header.motorInhibition == 1.0f)) {
        headerValidation[environment] =
            MR_NUMANX_HUMAN_MOTOR_HEADER_RELATION;
        return;
    }

    ulong hash = kFnvOffset;
    mixU32(hash, MR_NUMANX_BRAIN_MOTOR_OUTPUT_VERSION);
    mixU32(hash, header.formatVersion);
    mixU32(hash, header.flags);
    mixU64(hash, header.timestampMicroseconds);
    mixU64(hash, header.brainGeneration);
    mixU64(hash, header.profileFingerprint);
    mixU64(hash, header.protectiveCommandFingerprint);
    mixU32(hash, header.muscleCount);
    mixU32(hash, header.environmentIdentifier);
    mixFloat(hash, header.motorInhibition);
    mixFloat(hash, header.autonomicArousal);
    mixU32(hash, header.actuatorCommandKind);
    mixU32(hash, header.reserved);
    mixFloat(hash, header.outputMinimum);
    mixFloat(hash, header.outputMaximum);
    const ulong excitationBase =
        static_cast<ulong>(environment) *
            dispatch.excitationEnvironmentStride;
    for (uint muscle = 0u; muscle < dispatch.muscleCount; ++muscle) {
        const float command = excitations[excitationBase + muscle];
        if (!isfinite(command) || command < header.outputMinimum ||
            command > header.outputMaximum) {
            headerValidation[environment] =
                MR_NUMANX_HUMAN_MOTOR_HEADER_PAYLOAD;
            return;
        }
        mixFloat(hash, command);
    }
    if (header.outputFingerprint == 0ul ||
        header.outputFingerprint != hash ||
        (decisionShadow && ready.motorOutputFingerprint != hash)) {
        headerValidation[environment] =
            MR_NUMANX_HUMAN_MOTOR_HEADER_FINGERPRINT;
        return;
    }
    headerValidation[environment] = MR_NUMANX_HUMAN_MOTOR_HEADER_VALID;
}

// beginStep admission. Every source slot is written exactly once. Invalid
// motor values are replaced by zero so they cannot poison kinematics or
// MyoSim, but their validation word remains failed and the post-stand gate
// converts that failure into owning Human transaction failure.
kernel void numanx_human_admit_excitations(
    const device float* excitations [[buffer(0)]],
    device MRMujocoMuscleStateGPU* states [[buffer(1)]],
    device uint* motorValidation [[buffer(2)]],
    const device uint* headerValidation [[buffer(3)]],
    constant MRNumanXHumanMotorDispatchGPU& dispatch [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    const uint count = dispatch.environmentCount * dispatch.muscleCount;
    if (index >= count ||
        dispatch.abiVersion != MR_NUMANX_HUMAN_IO_ABI_VERSION) {
        return;
    }

    const uint environment = index / dispatch.muscleCount;
    const uint muscle = index - environment * dispatch.muscleCount;
    const ulong excitationIndex =
        static_cast<ulong>(environment) *
            dispatch.excitationEnvironmentStride +
        muscle;
    const ulong stateIndex =
        static_cast<ulong>(environment) * dispatch.stateStride + muscle;
    const float value = excitations[excitationIndex];
    const bool finite = isfinite(value);
    const bool inUnitInterval = finite && value >= 0.0f && value <= 1.0f;
    const bool authenticated = headerValidation[environment] ==
        MR_NUMANX_HUMAN_MOTOR_HEADER_VALID;

    uint validation = MR_NUMANX_HUMAN_MOTOR_COPIED;
    if (finite) {
        validation |= MR_NUMANX_HUMAN_MOTOR_FINITE;
    }
    if (inUnitInterval) {
        validation |= MR_NUMANX_HUMAN_MOTOR_IN_UNIT_INTERVAL;
    }
    if (authenticated) {
        validation |= MR_NUMANX_HUMAN_MOTOR_HEADER_AUTHENTICATED;
    }

    // clamp() makes the accepted interval explicit; rejected finite and
    // non-finite values are deliberately substituted by zero.
    states[stateIndex].excitationAndActivation.x =
        authenticated && inUnitInterval
        ? clamp(value, 0.0f, 1.0f)
        : 0.0f;
    motorValidation[index] = validation;
}

// One thread owns each environment status update, avoiding atomic races. The
// kernel executes after stand dynamics. It first verifies the owning stand
// status and completion count, then validates every causal MyoSim receptor
// source. Any motor or receptor error is propagated into the Human status so
// MetalArticulatedOperatorSubmission::wait() rejects the whole publication.
kernel void numanx_human_gate_proprioception(
    const device MRMujocoMuscleStateGPU* states [[buffer(0)]],
    const device MRMujocoMuscleResultGPU* results [[buffer(1)]],
    const device uint* motorValidation [[buffer(2)]],
    device MRNumiHumanStandStatusGPU* standStatuses [[buffer(3)]],
    device uint* environmentGate [[buffer(4)]],
    constant MRNumanXHumanProprioceptionDispatchGPU& dispatch [[buffer(5)]],
    uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    environmentGate[environment] = 0u;
    if (dispatch.abiVersion != MR_NUMANX_HUMAN_IO_ABI_VERSION ||
        dispatch.featureCount !=
            MR_NUMANX_HUMAN_PROPRIOCEPTION_FEATURE_COUNT) {
        return;
    }

    device MRNumiHumanStandStatusGPU& status =
        standStatuses[environment];
    if (status.environment != environment ||
        status.code != MR_NUMI_HUMAN_STAND_SUCCESS) {
        return;
    }
    if (status.completedSteps != dispatch.stepIndex + 1u) {
        status.code = MR_NUMI_HUMAN_STAND_INVALID_DISPATCH;
        status.failingIndex = dispatch.stepIndex;
        return;
    }

    constexpr uint requiredMotorValidation =
        MR_NUMANX_HUMAN_MOTOR_FINITE |
        MR_NUMANX_HUMAN_MOTOR_IN_UNIT_INTERVAL |
        MR_NUMANX_HUMAN_MOTOR_COPIED |
        MR_NUMANX_HUMAN_MOTOR_HEADER_AUTHENTICATED;
    for (uint muscle = 0u; muscle < dispatch.muscleCount; ++muscle) {
        const ulong compactIndex =
            static_cast<ulong>(environment) * dispatch.muscleCount +
            muscle;
        if ((motorValidation[compactIndex] & requiredMotorValidation) !=
            requiredMotorValidation) {
            status.code =
                (motorValidation[compactIndex] &
                    MR_NUMANX_HUMAN_MOTOR_FINITE) == 0u
                ? MR_NUMI_HUMAN_STAND_NONFINITE_INPUT
                : MR_NUMI_HUMAN_STAND_INVALID_DISPATCH;
            status.failingIndex = muscle;
            return;
        }

        const ulong stateIndex =
            static_cast<ulong>(environment) * dispatch.stateStride + muscle;
        const ulong resultIndex =
            static_cast<ulong>(environment) * dispatch.resultStride + muscle;
        if (!validReceptorSource(
                states[stateIndex],
                results[resultIndex],
                environment,
                muscle
            )) {
            status.code = MR_NUMI_HUMAN_STAND_NONFINITE_RESULT;
            status.failingIndex = muscle;
            return;
        }
    }

    environmentGate[environment] = 1u;
}

// Environment-major, step-major receptor publication. A failed environment
// is deterministically zero-filled for this step with a zero UInt32 validity
// mask. A successful row publishes all ten features and sets the corresponding
// lower ten bits.
kernel void numanx_human_write_proprioception(
    const device MRMujocoMuscleStateGPU* states [[buffer(0)]],
    const device MRMujocoMuscleResultGPU* results [[buffer(1)]],
    const device uint* environmentGate [[buffer(2)]],
    device float* proprioception [[buffer(3)]],
    device uint* validity [[buffer(4)]],
    device float* interoception [[buffer(5)]],
    device uint* interoceptionValidity [[buffer(6)]],
    constant MRNumanXHumanProprioceptionDispatchGPU& dispatch [[buffer(7)]],
    uint index [[thread_position_in_grid]]
) {
    const uint count = dispatch.environmentCount * dispatch.muscleCount;
    if (index >= count ||
        dispatch.abiVersion != MR_NUMANX_HUMAN_IO_ABI_VERSION ||
        dispatch.featureCount !=
            MR_NUMANX_HUMAN_PROPRIOCEPTION_FEATURE_COUNT) {
        return;
    }

    const uint environment = index / dispatch.muscleCount;
    const uint muscle = index - environment * dispatch.muscleCount;
    const ulong outputBase =
        static_cast<ulong>(environment) *
            dispatch.proprioceptionEnvironmentStride +
        static_cast<ulong>(dispatch.stepIndex) *
            dispatch.proprioceptionStepStride +
        static_cast<ulong>(muscle) * dispatch.featureCount;
    const ulong validityIndex =
        static_cast<ulong>(environment) * dispatch.validityEnvironmentStride +
        static_cast<ulong>(dispatch.stepIndex) * dispatch.validityStepStride +
        muscle;

    if (environmentGate[environment] == 0u) {
        for (uint feature = 0u; feature < dispatch.featureCount; ++feature) {
            proprioception[outputBase + feature] = 0.0f;
        }
        validity[validityIndex] = 0u;
        interoception[validityIndex] = 0.0f;
        interoceptionValidity[validityIndex] = 0u;
        return;
    }

    const ulong stateIndex =
        static_cast<ulong>(environment) * dispatch.stateStride + muscle;
    const ulong resultIndex =
        static_cast<ulong>(environment) * dispatch.resultStride + muscle;
    const float4 state = states[stateIndex].excitationAndActivation;
    const device MRMujocoMuscleResultGPU& result = results[resultIndex];

    proprioception[outputBase + MR_NUMANX_HUMAN_FEATURE_EXCITATION] = state.x;
    proprioception[outputBase + MR_NUMANX_HUMAN_FEATURE_ACTIVATION] = state.y;
    proprioception[
        outputBase + MR_NUMANX_HUMAN_FEATURE_FIBRE_LENGTH_METRES
    ] = state.z;
    proprioception[
        outputBase +
        MR_NUMANX_HUMAN_FEATURE_FIBRE_VELOCITY_METRES_PER_SECOND
    ] = state.w;
    proprioception[
        outputBase + MR_NUMANX_HUMAN_FEATURE_PATH_LENGTH_METRES
    ] = result.pathForceAndActivationDerivative.x;
    proprioception[
        outputBase +
        MR_NUMANX_HUMAN_FEATURE_PATH_VELOCITY_METRES_PER_SECOND
    ] = result.pathForceAndActivationDerivative.y;
    proprioception[
        outputBase + MR_NUMANX_HUMAN_FEATURE_APPLIED_ACTIVE_FORCE_NEWTONS
    ] = result.activeForceAndReserved.x;
    proprioception[
        outputBase + MR_NUMANX_HUMAN_FEATURE_TENDON_TENSION_NEWTONS
    ] = result.fiberStateTendonForceResidual.z;
    proprioception[
        outputBase +
        MR_NUMANX_HUMAN_FEATURE_ACTIVATION_DERIVATIVE_PER_SECOND
    ] = result.pathForceAndActivationDerivative.w;
    proprioception[
        outputBase +
        MR_NUMANX_HUMAN_FEATURE_NORMALIZED_EQUILIBRIUM_RESIDUAL
    ] = result.fiberStateTendonForceResidual.w;
    validity[validityIndex] =
        MR_NUMANX_HUMAN_PROPRIOCEPTION_VALIDITY_ALL;
    interoception[validityIndex] = clamp(
        0.5f * (state.x + state.y), 0.0f, 1.0f);
    interoceptionValidity[validityIndex] =
        MR_NUMANX_HUMAN_INTEROCEPTION_VALIDITY_ALL;
}
