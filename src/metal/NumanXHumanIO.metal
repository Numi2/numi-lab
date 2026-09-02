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

inline float4 quaternionMultiply(const float4 first, const float4 second) {
    return float4(
        first.w * second.xyz + second.w * first.xyz +
            cross(first.xyz, second.xyz),
        first.w * second.w - dot(first.xyz, second.xyz)
    );
}

inline float3 quaternionRotate(const float4 quaternion, const float3 value) {
    return value + 2.0f * cross(
        quaternion.xyz,
        cross(quaternion.xyz, value) + quaternion.w * value
    );
}

inline float3 quaternionInverseRotate(
    const float4 quaternion,
    const float3 value
) {
    return quaternionRotate(
        float4(-quaternion.xyz, quaternion.w), value);
}

inline float rayBoxDistance(
    const float3 origin,
    const float3 direction,
    const float3 minimum,
    const float3 maximum
) {
    float nearDistance = -INFINITY;
    float farDistance = INFINITY;
    for (uint axis = 0u; axis < 3u; ++axis) {
        const float component = direction[axis];
        if (abs(component) < 1.0e-7f) {
            if (origin[axis] < minimum[axis] ||
                origin[axis] > maximum[axis]) {
                return INFINITY;
            }
            continue;
        }
        const float inverse = 1.0f / component;
        const float first = (minimum[axis] - origin[axis]) * inverse;
        const float second = (maximum[axis] - origin[axis]) * inverse;
        nearDistance = max(nearDistance, min(first, second));
        farDistance = min(farDistance, max(first, second));
    }
    if (farDistance < max(nearDistance, 0.0f)) return INFINITY;
    return nearDistance > 0.0f ? nearDistance : farDistance;
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
        header.outputFingerprint != hash) {
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
        const ulong interoceptionBase =
            validityIndex * MR_NUMANX_HUMAN_INTEROCEPTION_FEATURE_COUNT;
        for (uint feature = 0u;
             feature < MR_NUMANX_HUMAN_INTEROCEPTION_FEATURE_COUNT;
             ++feature) {
            interoception[interoceptionBase + feature] = 0.0f;
        }
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
    const ulong interoceptionBase =
        validityIndex * MR_NUMANX_HUMAN_INTEROCEPTION_FEATURE_COUNT;
    const float activationLoad = clamp(
        0.5f * (state.x + state.y), 0.0f, 1.0f);
    const float normalizedVelocity = clamp(
        abs(state.w) * max(dispatch.timestepSecondsAndReserved.x, 0.0f),
        0.0f,
        1.0f);
    const float normalizedTension = clamp(
        abs(result.fiberStateTendonForceResidual.z) /
            (1.0f + abs(result.fiberStateTendonForceResidual.z)),
        0.0f,
        1.0f);
    const float normalizedResidual = clamp(
        abs(result.fiberStateTendonForceResidual.w), 0.0f, 1.0f);
    interoception[
        interoceptionBase +
        MR_NUMANX_HUMAN_INTEROCEPTION_ENERGY_AVAILABILITY
    ] = 1.0f - activationLoad;
    interoception[
        interoceptionBase +
        MR_NUMANX_HUMAN_INTEROCEPTION_OXYGEN_AVAILABILITY
    ] = clamp(1.0f - state.y, 0.0f, 1.0f);
    interoception[
        interoceptionBase +
        MR_NUMANX_HUMAN_INTEROCEPTION_CARBON_DIOXIDE_LOAD
    ] = clamp(state.y, 0.0f, 1.0f);
    interoception[
        interoceptionBase +
        MR_NUMANX_HUMAN_INTEROCEPTION_THERMAL_LOAD
    ] = clamp(0.5f * normalizedVelocity + 0.5f * activationLoad, 0.0f, 1.0f);
    interoception[
        interoceptionBase +
        MR_NUMANX_HUMAN_INTEROCEPTION_FATIGUE_LOAD
    ] = clamp(0.5f * activationLoad + 0.5f * normalizedTension, 0.0f, 1.0f);
    interoception[
        interoceptionBase +
        MR_NUMANX_HUMAN_INTEROCEPTION_TISSUE_STRESS
    ] = clamp(0.5f * normalizedTension + 0.5f * normalizedResidual, 0.0f, 1.0f);
    interoceptionValidity[validityIndex] =
        MR_NUMANX_HUMAN_INTEROCEPTION_VALIDITY_ALL;
}

// Five zero-readback sensor channels derived from the exact articulated
// transaction. The command buffer already orders this kernel after stand and
// HumanIO's owning environment gate. Body/point geometry is the transaction's
// current device state; visual bounds are compact source-pack evidence loaded
// once by the native runtime.
kernel void numanx_human_write_supplemental_sensors(
    const device float* q [[buffer(0)]],
    const device float* v [[buffer(1)]],
    const device MRArticulatedBodyPoseGPU* bodyPoses [[buffer(2)]],
    const device MRArticulatedPointWorldGPU* pointWorld [[buffer(3)]],
    const device MRNumiHumanStandStatusGPU* standStatuses [[buffer(4)]],
    const device MRNumanXActiveSensingCommandGPU* activeSensing [[buffer(5)]],
    const device MRNumanXVisualBodyBoundsGPU* bodyBounds [[buffer(6)]],
    device float* kinesthesia [[buffer(7)]],
    device uint* kinesthesiaValidity [[buffer(8)]],
    device float* vestibular [[buffer(9)]],
    device uint* vestibularValidity [[buffer(10)]],
    device float* audition [[buffer(11)]],
    device uint* auditionValidity [[buffer(12)]],
    device float* vision [[buffer(13)]],
    device uint* visionValidity [[buffer(14)]],
    device float* touch [[buffer(15)]],
    device uint* touchValidity [[buffer(16)]],
    constant MRNumanXHumanSupplementalDispatchGPU& dispatch [[buffer(17)]],
    const device MRNumanXHumanSupportConsequenceGPU* supportConsequences
        [[buffer(18)]],
    uint index [[thread_position_in_grid]]
) {
    (void)pointWorld;
    const bool validDispatch =
        dispatch.abiVersion == MR_NUMANX_HUMAN_IO_ABI_VERSION &&
        dispatch.qCoordinateCount == 129u &&
        dispatch.dofCount == MR_NUMANX_HUMAN_KINESTHESIA_RECEPTOR_COUNT &&
        dispatch.bodyCount > dispatch.headBodyIndex &&
        dispatch.pointCount >=
            dispatch.supportPointOffset + dispatch.supportPointCount &&
        dispatch.supportPointCount == MR_NUMANX_HUMAN_TOUCH_RECEPTOR_COUNT &&
        dispatch.visionWidth == MR_NUMANX_HUMAN_VISION_WIDTH &&
        dispatch.visionHeight == MR_NUMANX_HUMAN_VISION_HEIGHT &&
        dispatch.bodyBoundsCount == dispatch.bodyCount &&
        dispatch.sensorGeneration != 0ul &&
        dispatch.transactionFingerprint != 0ul &&
        dispatch.substepFingerprint != 0ul &&
        dispatch.expectedActiveSensingGPUAddress != 0ul &&
        dispatch.visualSourceFingerprint != 0ul &&
        dispatch.programFingerprint != 0ul &&
        dispatch.expectedSupportConsequencesGPUAddress != 0ul &&
        dispatch.matterProgramFingerprint != 0ul &&
        dispatch.reserved0 == 0u &&
        all(isfinite(dispatch.groundPoint)) &&
        all(isfinite(dispatch.groundNormal)) &&
        all(isfinite(dispatch.cameraLocalPosition)) &&
        all(isfinite(dispatch.cameraLocalOrientation)) &&
        all(isfinite(dispatch.visionIntrinsics)) &&
        all(isfinite(dispatch.visionDepthAndTimestep));
    const bool environmentValid = validDispatch &&
        standStatuses[0].environment == 0u &&
        standStatuses[0].code == MR_NUMI_HUMAN_STAND_SUCCESS &&
        standStatuses[0].completedSteps == 1u;

    if (index < MR_NUMANX_HUMAN_KINESTHESIA_RECEPTOR_COUNT) {
        const ulong base =
            static_cast<ulong>(index) *
            MR_NUMANX_HUMAN_KINESTHESIA_FEATURE_COUNT;
        for (uint feature = 0u;
             feature < MR_NUMANX_HUMAN_KINESTHESIA_FEATURE_COUNT;
             ++feature) {
            kinesthesia[base + feature] = 0.0f;
        }
        kinesthesiaValidity[index] = 0u;
        if (environmentValid) {
            const uint qIndex = index < 6u ? index : index + 1u;
            const float position = q[qIndex];
            const float velocity = v[index];
            const float activity = clamp(abs(velocity) *
                dispatch.visionDepthAndTimestep.w, 0.0f, 1.0f);
            kinesthesia[base + 0u] = position;
            kinesthesia[base + 1u] = velocity;
            kinesthesia[base + 2u] = abs(position);
            kinesthesia[base + 3u] = abs(velocity);
            kinesthesia[base + 4u] = position * velocity;
            kinesthesia[base + 5u] = activity;
            kinesthesia[base + 6u] = clamp(abs(position), 0.0f, 1.0f);
            kinesthesiaValidity[index] =
                MR_NUMANX_HUMAN_KINESTHESIA_VALIDITY_ALL;
        }
    }

    if (index == 0u) {
        for (uint feature = 0u;
             feature < MR_NUMANX_HUMAN_VESTIBULAR_FEATURE_COUNT;
             ++feature) {
            vestibular[feature] = 0.0f;
        }
        vestibularValidity[0] = 0u;
        if (environmentValid) {
            const device MRArticulatedBodyPoseGPU& head =
                bodyPoses[dispatch.headBodyIndex];
            for (uint feature = 0u; feature < 7u; ++feature) {
                vestibular[feature] = q[feature];
            }
            for (uint feature = 0u; feature < 6u; ++feature) {
                vestibular[7u + feature] = v[feature];
            }
            vestibular[13u] = head.position.x;
            vestibular[14u] = head.position.y;
            vestibular[15u] = head.position.z;
            vestibular[16u] = head.orientation.x;
            vestibular[17u] = head.orientation.y;
            vestibular[18u] = head.orientation.z;
            vestibular[19u] = head.orientation.w;
            const float3 groundNormal = normalize(dispatch.groundNormal.xyz);
            vestibular[20u] = dot(
                head.position.xyz - dispatch.groundPoint.xyz,
                groundNormal);
            vestibular[21u] = dot(float3(v[0], v[1], v[2]), groundNormal);
            vestibularValidity[0] =
                MR_NUMANX_HUMAN_VESTIBULAR_VALIDITY_ALL;
        }
    }

    if (index < MR_NUMANX_HUMAN_AUDITION_RECEPTOR_COUNT) {
        const ulong base =
            static_cast<ulong>(index) * MR_NUMANX_HUMAN_AUDITION_FEATURE_COUNT;
        for (uint feature = 0u;
             feature < MR_NUMANX_HUMAN_AUDITION_FEATURE_COUNT;
             ++feature) {
            audition[base + feature] = 0.0f;
        }
        auditionValidity[index] = 0u;
        if (environmentValid) {
            const uint body = min(index, dispatch.bodyCount - 1u);
            const float3 relative =
                bodyPoses[body].position.xyz - bodyPoses[0].position.xyz;
            const float band =
                (static_cast<float>(index) + 0.5f) /
                MR_NUMANX_HUMAN_AUDITION_RECEPTOR_COUNT;
            audition[base + 0u] = relative.x;
            audition[base + 1u] = relative.y;
            audition[base + 2u] = dispatch.sensorGeneration > 1ul
                ? dot(float3(v[0], v[1], v[2]), normalize(relative + 1.0e-6f))
                : 0.0f;
            audition[base + 3u] = relative.z;
            audition[base + 4u] = length(relative);
            audition[base + 5u] = band;
            audition[base + 6u] = sinpi(band) * length(float3(v[0], v[1], v[2]));
            audition[base + 7u] = cospi(band) * length(float3(v[3], v[4], v[5]));
            auditionValidity[index] = dispatch.sensorGeneration > 1ul
                ? MR_NUMANX_HUMAN_AUDITION_VALIDITY_ALL
                : MR_NUMANX_HUMAN_AUDITION_FIRST_VALIDITY;
        }
    }

    if (index < MR_NUMANX_HUMAN_TOUCH_RECEPTOR_COUNT) {
        const ulong base =
            static_cast<ulong>(index) * MR_NUMANX_HUMAN_TOUCH_FEATURE_COUNT;
        for (uint feature = 0u;
             feature < MR_NUMANX_HUMAN_TOUCH_FEATURE_COUNT;
             ++feature) {
            touch[base + feature] = 0.0f;
        }
        touchValidity[index] = 0u;
        if (environmentValid) {
            const MRNumanXHumanSupportConsequenceGPU consequence =
                supportConsequences[index];
            const bool consequenceValid =
                consequence.identity.x == index &&
                consequence.identity.w ==
                    MR_NUMANX_HUMAN_SUPPORT_CONSEQUENCE_VERSION &&
                all(isfinite(consequence.pointAndSeparation)) &&
                all(isfinite(consequence.impulseAndNormal)) &&
                all(isfinite(consequence.tangentVelocityAndImpulse)) &&
                consequence.impulseAndNormal.w >= 0.0f &&
                consequence.tangentVelocityAndImpulse.w >= 0.0f;
            const float3 point = consequence.pointAndSeparation.xyz;
            const float inverseTimestep = 1.0f /
                max(dispatch.visionDepthAndTimestep.w, 1.0e-12f);
            touch[base + 0u] = point.x;
            touch[base + 1u] = point.y;
            touch[base + 2u] = point.z;
            touch[base + 3u] = consequence.pointAndSeparation.w;
            touch[base + 4u] =
                consequence.impulseAndNormal.w * inverseTimestep;
            touch[base + 5u] =
                consequence.tangentVelocityAndImpulse.w * inverseTimestep;
            touch[base + 6u] = length(
                consequence.tangentVelocityAndImpulse.xyz);
            touchValidity[index] = consequenceValid
                ? MR_NUMANX_HUMAN_TOUCH_VALIDITY_ALL : 0u;
        }
    }

    if (index < MR_NUMANX_HUMAN_VISION_RECEPTOR_COUNT) {
        const ulong base =
            static_cast<ulong>(index) * MR_NUMANX_HUMAN_VISION_FEATURE_COUNT;
        for (uint feature = 0u;
             feature < MR_NUMANX_HUMAN_VISION_FEATURE_COUNT;
             ++feature) {
            vision[base + feature] = 0.0f;
        }
        visionValidity[index] = 0u;
        if (environmentValid) {
            const uint x = index % dispatch.visionWidth;
            const uint y = index / dispatch.visionWidth;
            const device MRArticulatedBodyPoseGPU& head =
                bodyPoses[dispatch.headBodyIndex];
            const device MRNumanXActiveSensingCommandGPU& sensing =
                activeSensing[0];
            const float command = isfinite(sensing.command)
                ? clamp(sensing.command, -1.0f, 1.0f) : 0.0f;
            const float confidence = isfinite(sensing.confidence)
                ? clamp(sensing.confidence, 0.0f, 1.0f) : 0.0f;
            const float yaw = command * confidence * 0.65f;
            const float downwardSearch = abs(command) * confidence * 0.35f;
            const float halfYaw = 0.5f * yaw;
            const float halfPitch = 0.5f * downwardSearch;
            const float4 activeRotation = quaternionMultiply(
                float4(0.0f, 0.0f, sin(halfYaw), cos(halfYaw)),
                float4(0.0f, sin(halfPitch), 0.0f, cos(halfPitch)));
            const float4 cameraOrientation = normalize(quaternionMultiply(
                quaternionMultiply(
                    head.orientation,
                    dispatch.cameraLocalOrientation),
                activeRotation));
            const float3 cameraOrigin = head.position.xyz +
                quaternionRotate(
                    head.orientation,
                    dispatch.cameraLocalPosition.xyz);
            const float3 sensorRay = normalize(float3(
                1.0f,
                -(static_cast<float>(x) - dispatch.visionIntrinsics.z) /
                    dispatch.visionIntrinsics.x,
                -(static_cast<float>(y) - dispatch.visionIntrinsics.w) /
                    dispatch.visionIntrinsics.y));
            const float3 worldRay = quaternionRotate(
                cameraOrientation, sensorRay);
            const float minimumDepth = dispatch.visionDepthAndTimestep.x;
            const float maximumDepth = dispatch.visionDepthAndTimestep.y;
            float depth = INFINITY;
            uint semantic = 0u;
            const float groundDenominator = dot(
                worldRay, dispatch.groundNormal.xyz);
            if (abs(groundDenominator) > 1.0e-7f) {
                const float groundDepth = dot(
                    dispatch.groundPoint.xyz - cameraOrigin,
                    dispatch.groundNormal.xyz) / groundDenominator;
                if (groundDepth >= minimumDepth &&
                    groundDepth <= maximumDepth) {
                    depth = groundDepth;
                    semantic = 1u;
                }
            }
            for (uint body = 0u; body < dispatch.bodyBoundsCount; ++body) {
                const float3 minimum = bodyBounds[body].minimum.xyz;
                const float3 maximum = bodyBounds[body].maximum.xyz;
                if (any(minimum > maximum)) continue;
                const float3 localOrigin = quaternionInverseRotate(
                    bodyPoses[body].orientation,
                    cameraOrigin - bodyPoses[body].position.xyz);
                const float3 localDirection = quaternionInverseRotate(
                    bodyPoses[body].orientation, worldRay);
                const float candidateDepth = rayBoxDistance(
                    localOrigin, localDirection, minimum, maximum);
                if (candidateDepth >= minimumDepth && candidateDepth < depth &&
                    candidateDepth <= maximumDepth) {
                    depth = candidateDepth;
                    semantic = body + 2u;
                }
            }
            vision[base + 0u] = worldRay.x;
            vision[base + 1u] = worldRay.y;
            vision[base + 2u] = worldRay.z;
            visionValidity[index] = MR_NUMANX_HUMAN_VISION_VALIDITY_RAY;
            if (isfinite(depth)) {
                const float quantum = max(
                    dispatch.visionDepthAndTimestep.z, 1.0e-7f);
                const float quantizedDepth = rint(depth / quantum) * quantum;
                const float3 hit = cameraOrigin + worldRay * quantizedDepth;
                vision[base + 3u] = quantizedDepth;
                vision[base + 4u] = hit.x;
                vision[base + 5u] = hit.y;
                vision[base + 6u] = hit.z;
                vision[base + 7u] = static_cast<float>(semantic);
                visionValidity[index] |=
                    MR_NUMANX_HUMAN_VISION_VALIDITY_DEPTH |
                    MR_NUMANX_HUMAN_VISION_VALIDITY_GEOMETRY;
            }
        }
    }
}
