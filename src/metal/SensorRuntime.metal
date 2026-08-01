#include <metal_stdlib>

#include "metalrobo/counter_rng.h"
#include "metalrobo/engine_types.h"
#include "metalrobo/runtime_abi_generated.h"
#include "metalrobo/world_compiler_types.h"

using namespace metal;

namespace {

inline ulong unpack64(const uint low, const uint high) {
    return static_cast<ulong>(low) |
        (static_cast<ulong>(high) << 32u);
}

inline uint low32(const ulong value) {
    return static_cast<uint>(value);
}

inline uint high32(const ulong value) {
    return static_cast<uint>(value >> 32u);
}

inline bool finite4(const float4 value) {
    return all(isfinite(value));
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
    return value + quaternion.w * doubledCross +
        cross(quaternion.xyz, doubledCross);
}

inline float3 quaternionRotateInverse(
    const float4 quaternion,
    const float3 value
) {
    return quaternionRotate(
        float4(-quaternion.xyz, quaternion.w),
        value
    );
}

inline float3 normalizedOr(
    const float3 value,
    const float3 fallback
) {
    const float lengthSquared = dot(value, value);
    return lengthSquared > 1.0e-12f &&
            isfinite(lengthSquared)
        ? value * rsqrt(lengthSquared)
        : fallback;
}

inline float3 stableContactTangent(const float3 normal) {
    const float3 absoluteNormal = abs(normal);
    const float3 reference =
        absoluteNormal.x <= absoluteNormal.y &&
        absoluteNormal.x <= absoluteNormal.z
        ? float3(1.0f, 0.0f, 0.0f)
        : absoluteNormal.y <= absoluteNormal.z
        ? float3(0.0f, 1.0f, 0.0f)
        : float3(0.0f, 0.0f, 1.0f);
    return normalizedOr(
        cross(reference, normal),
        float3(1.0f, 0.0f, 0.0f)
    );
}

inline bool contactFilterAllows(
    device const uint* filterBodies,
    const uint offset,
    const uint count,
    const uint body
) {
    if (count == 0u) {
        return true;
    }
    for (uint index = 0u; index < count; ++index) {
        if (filterBodies[offset + index] == body) {
            return true;
        }
    }
    return false;
}

inline bool normalizedQuaternion(
    const float4 input,
    thread float4& result
) {
    if (!finite4(input)) {
        return false;
    }
    const float squared = dot(input, input);
    if (!(squared > 1.0e-12f) || !isfinite(squared)) {
        return false;
    }
    result = input * rsqrt(squared);
    return finite4(result);
}

inline ulong sensorRandomIdentity(const uint4 identity) {
    return static_cast<ulong>(identity.x) |
        (static_cast<ulong>(identity.y) << 32u);
}

inline float sensorGaussian(
    const ulong seed,
    const uint environment,
    const uint episode,
    const ulong sensorIdentity,
    const ulong sample,
    const uint channel,
    const uint purpose
) {
    const float first = max(
        mr_sensor_counter_uniform(
            seed,
            environment,
            episode,
            sensorIdentity,
            sample,
            channel,
            purpose
        ),
        1.0f / 16777216.0f
    );
    const float second = mr_sensor_counter_uniform(
        seed,
        environment,
        episode,
        sensorIdentity,
        sample,
        channel,
        purpose + 1u
    );
    return sqrt(-2.0f * log(first)) *
        cos(6.28318530717958647692f * second);
}

inline float sensorContinuousDelta(
    const ulong seed,
    const uint environment,
    const uint episode,
    const ulong sensorIdentity,
    const ulong sample,
    const uint channel,
    const float valueSigma,
    const float biasSigma
) {
    float result = 0.0f;
    if (biasSigma > 0.0f) {
        result += biasSigma * sensorGaussian(
            seed,
            environment,
            episode,
            sensorIdentity,
            0u,
            channel,
            0u
        );
    }
    if (valueSigma > 0.0f) {
        result += valueSigma * sensorGaussian(
            seed,
            environment,
            episode,
            sensorIdentity,
            sample,
            channel,
            2u
        );
    }
    return result;
}

inline bool corruptSensorSample(
    device float* outputs,
    const uint outputBase,
    const uint outputCount,
    const uint sensorKind,
    const float valueSigma,
    const float biasSigma,
    const ulong seed,
    const uint environment,
    const uint episode,
    const ulong sensorIdentity,
    const ulong sample
) {
    if (valueSigma == 0.0f && biasSigma == 0.0f) {
        return true;
    }
    if (sensorKind == MR_WORLD_SENSOR_STATE) {
        for (uint channel = 0u; channel < 3u; ++channel) {
            outputs[outputBase + channel] +=
                sensorContinuousDelta(
                    seed,
                    environment,
                    episode,
                    sensorIdentity,
                    sample,
                    channel,
                    valueSigma,
                    biasSigma
                );
        }
        const float3 tangent(
            sensorContinuousDelta(
                seed,
                environment,
                episode,
                sensorIdentity,
                sample,
                3u,
                valueSigma,
                biasSigma
            ),
            sensorContinuousDelta(
                seed,
                environment,
                episode,
                sensorIdentity,
                sample,
                4u,
                valueSigma,
                biasSigma
            ),
            sensorContinuousDelta(
                seed,
                environment,
                episode,
                sensorIdentity,
                sample,
                5u,
                valueSigma,
                biasSigma
            )
        );
        const float angle = length(tangent);
        if (!isfinite(angle)) {
            return false;
        }
        const float halfAngle = 0.5f * angle;
        const float scale = angle > 1.0e-7f
            ? sin(halfAngle) / angle
            : 0.5f;
        const float4 delta(
            tangent * scale,
            cos(halfAngle)
        );
        float4 orientation;
        if (!normalizedQuaternion(
                quaternionMultiply(
                    float4(
                        outputs[outputBase + 3u],
                        outputs[outputBase + 4u],
                        outputs[outputBase + 5u],
                        outputs[outputBase + 6u]
                    ),
                    delta
                ),
                orientation
            )) {
            return false;
        }
        outputs[outputBase + 3u] = orientation.x;
        outputs[outputBase + 4u] = orientation.y;
        outputs[outputBase + 5u] = orientation.z;
        outputs[outputBase + 6u] = orientation.w;
    } else {
        const uint firstContinuous =
            sensorKind == MR_WORLD_SENSOR_CONTACT_STATE
            ? 2u
            : 0u;
        for (uint channel = firstContinuous;
             channel < outputCount;
             ++channel) {
            outputs[outputBase + channel] +=
                sensorContinuousDelta(
                    seed,
                    environment,
                    episode,
                    sensorIdentity,
                    sample,
                    channel,
                    valueSigma,
                    biasSigma
                );
            if (sensorKind == MR_WORLD_SENSOR_CONTACT_STATE) {
                outputs[outputBase + channel] = max(
                    outputs[outputBase + channel],
                    0.0f
                );
            }
        }
    }
    for (uint channel = 0u; channel < outputCount; ++channel) {
        if (!isfinite(outputs[outputBase + channel])) {
            return false;
        }
    }
    return true;
}

} // namespace

// Canonical control-boundary SensorIR executor for parent-frame pose,
// world-twist, IMU, and solver-authoritative contact modalities. Reset-only
// execution seeds a new episode before its first action; advance execution
// samples the accepted post-physics state for the next action. One thread owns
// one environment/sensor history ring, so publication is deterministic and
// requires no atomics. A reset-only pass journals only reset environments;
// the accepted-state pass restores that journal if physics rejects the step.
// Successful steps and ordinary failed steps therefore perform no history
// copy. Presentation and tactile domains remain separate until their existing
// native kernels are folded into this schedule; descriptors for those domains
// are rejected by the host execution gate rather than silently skipped in
// production.
kernel void mr_sensor_sample_control_boundary(
    constant MRSensorDispatchGPU& dispatch
        [[buffer(MR_SENSOR_SAMPLE_DISPATCH)]],
    device const MRSensorProgramHeaderGPU& program
        [[buffer(MR_SENSOR_SAMPLE_PROGRAM)]],
    device const MRSensorDescriptorGPU* descriptors
        [[buffer(MR_SENSOR_SAMPLE_DESCRIPTORS)]],
    device const uint* filterBodies
        [[buffer(MR_SENSOR_SAMPLE_FILTER_BODIES)]],
    constant MRMetalWorldPassGPU& pass
        [[buffer(MR_SENSOR_SAMPLE_PASS)]],
    device const uint* resetMasks
        [[buffer(MR_SENSOR_SAMPLE_RESET_MASKS)]],
    device const MRArticulatedBodyPoseGPU* bodyPoses
        [[buffer(MR_SENSOR_SAMPLE_BODY_POSES)]],
    device const MRBodyStateGPU* bodyStates
        [[buffer(MR_SENSOR_SAMPLE_BODY_STATES)]],
    device const MRBodyStateGPU* sceneBodies
        [[buffer(MR_SENSOR_SAMPLE_SCENE_BODIES)]],
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(MR_SENSOR_SAMPLE_CONTACT_DISPATCH)]],
    device const MRContactConstraintGPU* contacts
        [[buffer(MR_SENSOR_SAMPLE_CONTACTS)]],
    device const MRMetalWorldContactStatusGPU* contactStatuses
        [[buffer(MR_SENSOR_SAMPLE_CONTACT_STATUSES)]],
    device const MRMetalWorldStatusGPU* worldStatuses
        [[buffer(MR_SENSOR_SAMPLE_WORLD_STATUSES)]],
    device const MRWorldGPU& world
        [[buffer(MR_SENSOR_SAMPLE_WORLD)]],
    device MRSensorRuntimeStateGPU* states
        [[buffer(MR_SENSOR_SAMPLE_STATES)]],
    device float* history
        [[buffer(MR_SENSOR_SAMPLE_HISTORY)]],
    device float* outputs
        [[buffer(MR_SENSOR_SAMPLE_OUTPUTS)]],
    device MRSensorSampleMetadataGPU* metadata
        [[buffer(MR_SENSOR_SAMPLE_METADATA)]],
    device MRSensorRuntimeStateGPU* checkpointStates
        [[buffer(MR_SENSOR_SAMPLE_CHECKPOINT_STATES)]],
    device float* checkpointHistory
        [[buffer(MR_SENSOR_SAMPLE_CHECKPOINT_HISTORY)]],
    device float* checkpointOutputs
        [[buffer(MR_SENSOR_SAMPLE_CHECKPOINT_OUTPUTS)]],
    device MRSensorSampleMetadataGPU* checkpointMetadata
        [[buffer(MR_SENSOR_SAMPLE_CHECKPOINT_METADATA)]],
    const uint threadIndex [[thread_position_in_grid]]
) {
    const uint environmentCount = dispatch.counts.x;
    const uint sensorCount = dispatch.counts.y;
    const uint total = environmentCount * sensorCount;
    if (threadIndex >= total || sensorCount == 0u ||
        program.sensorFingerprint != dispatch.sensorFingerprint ||
        program.reserved.x != MR_SENSOR_PROGRAM_ABI_VERSION ||
        program.counts.x != sensorCount ||
        (dispatch.counts.z != MR_SENSOR_EXECUTION_RESET_ONLY &&
         dispatch.counts.z != MR_SENSOR_EXECUTION_ADVANCE)) {
        return;
    }
    const uint environment = threadIndex / sensorCount;
    const uint sensorIndex = threadIndex - environment * sensorCount;
    const MRSensorDescriptorGPU descriptor =
        descriptors[sensorIndex];
    const bool poseSensor =
        descriptor.identity.x == MR_WORLD_SENSOR_STATE;
    const bool twistSensor =
        descriptor.identity.x ==
            MR_WORLD_SENSOR_FRAME_TWIST_WORLD;
    const bool forceTorqueSensor =
        descriptor.identity.x == MR_WORLD_SENSOR_FORCE_TORQUE;
    const bool imuSensor =
        descriptor.identity.x == MR_WORLD_SENSOR_IMU;
    const bool contactStateSensor =
        descriptor.identity.x == MR_WORLD_SENSOR_CONTACT_STATE;
    const uint expectedOutputCount = poseSensor
        ? 7u
        : contactStateSensor ? 5u : 6u;
    if ((!poseSensor && !twistSensor && !forceTorqueSensor &&
         !imuSensor && !contactStateSensor) ||
        descriptor.schedule.z !=
            MR_WORLD_SENSOR_PHASE_PRE_CONTROL ||
        descriptor.source.w != 0u ||
        descriptor.output.y != expectedOutputCount ||
        descriptor.filter.x > program.reserved.y ||
        descriptor.filter.y >
            program.reserved.y - descriptor.filter.x ||
        (!contactStateSensor && descriptor.filter.y != 0u) ||
        descriptor.output.w == 0u) {
        return;
    }

    const uint stateIndex = environment * sensorCount + sensorIndex;
    MRSensorRuntimeStateGPU state = states[stateIndex];
    const bool resetRequested =
        (dispatch.counts.w & MR_SENSOR_DISPATCH_HAS_RESETS) != 0u &&
        resetMasks[
            pass.controlStep * environmentCount + environment
        ] != 0u;
    const bool reset =
        dispatch.counts.z == MR_SENSOR_EXECUTION_RESET_ONLY &&
        resetRequested;
    if (dispatch.counts.z == MR_SENSOR_EXECUTION_RESET_ONLY &&
        !reset) {
        return;
    }
    const uint outputEnvironmentBase =
        environment * program.counts.y;
    const uint historyEnvironmentBase =
        environment * program.counts.z;
    const uint outputBase =
        outputEnvironmentBase + descriptor.output.x;
    const uint historyBase =
        historyEnvironmentBase + descriptor.output.z;
    const uint outputCount = descriptor.output.y;
    const uint historyLength = descriptor.output.w;

    if (reset) {
        const uint nextEpisode = state.randomIdentity.x + 1u;
        checkpointStates[stateIndex] = state;
        checkpointMetadata[stateIndex] = metadata[stateIndex];
        for (uint value = 0u;
             value < outputCount;
             ++value) {
            checkpointOutputs[outputBase + value] =
                outputs[outputBase + value];
        }
        for (uint value = 0u;
             value < outputCount * historyLength;
             ++value) {
            checkpointHistory[historyBase + value] =
                history[historyBase + value];
        }
        state = {};
        state.randomIdentity.x = nextEpisode;
        for (uint value = 0u;
             value < outputCount;
             ++value) {
            outputs[outputBase + value] = 0.0f;
        }
        for (uint value = 0u;
             value < outputCount * historyLength;
             ++value) {
            history[historyBase + value] = 0.0f;
        }
        metadata[stateIndex] = {};
    }

    if (dispatch.counts.z == MR_SENSOR_EXECUTION_ADVANCE) {
        const bool contactSucceeded =
            (dispatch.counts.w &
             MR_SENSOR_DISPATCH_HAS_CONTACTS) == 0u ||
            contactStatuses[environment].code == MR_STEP_SUCCESS;
        if (worldStatuses[environment].code != MR_STEP_SUCCESS ||
            !contactSucceeded) {
            if (resetRequested) {
                states[stateIndex] = checkpointStates[stateIndex];
                metadata[stateIndex] =
                    checkpointMetadata[stateIndex];
                for (uint value = 0u;
                     value < outputCount;
                     ++value) {
                    outputs[outputBase + value] =
                        checkpointOutputs[outputBase + value];
                }
                for (uint value = 0u;
                     value < outputCount * historyLength;
                     ++value) {
                    history[historyBase + value] =
                        checkpointHistory[historyBase + value];
                }
            }
            return;
        }
    }

    ulong phase = unpack64(
        state.phaseAndSequence.x,
        state.phaseAndSequence.y
    );
    ulong sequence = unpack64(
        state.phaseAndSequence.z,
        state.phaseAndSequence.w
    );
    ulong timestamp = unpack64(
        state.timestampAgeValidity.x,
        state.timestampAgeValidity.y
    );
    const ulong controlPeriod = unpack64(
        dispatch.controlPeriodAndStrides.x,
        dispatch.controlPeriodAndStrides.y
    );
    const ulong samplePeriod = unpack64(
        descriptor.schedule.x,
        descriptor.schedule.y
    );
    if (controlPeriod == 0u || samplePeriod == 0u) {
        state.timestampAgeValidity.w =
            MR_SENSOR_SAMPLE_NONFINITE;
        states[stateIndex] = state;
        return;
    }

    bool sample = reset;
    if (dispatch.counts.z == MR_SENSOR_EXECUTION_ADVANCE) {
        timestamp += controlPeriod;
        phase += controlPeriod;
        if (phase >= samplePeriod) {
            phase %= samplePeriod;
            sample = true;
        }
        // A zero sequence means no reset boundary was available (for
        // example, a recovered offline state). Publish a coherent first
        // sample rather than leaving an uninitialized output indefinitely.
        sample = sample || sequence == 0u;
    }
    if (!sample) {
        state.phaseAndSequence.x = low32(phase);
        state.phaseAndSequence.y = high32(phase);
        state.timestampAgeValidity.x = low32(timestamp);
        state.timestampAgeValidity.y = high32(timestamp);
        state.timestampAgeValidity.z += 1u;
        const uint previousValidity =
            state.timestampAgeValidity.w;
        state.timestampAgeValidity.w =
            (previousValidity & MR_SENSOR_SAMPLE_VALID) != 0u
            ? MR_SENSOR_SAMPLE_VALID |
                MR_SENSOR_SAMPLE_STALE
            : (previousValidity &
               MR_SENSOR_SAMPLE_DROPPED) != 0u
            ? MR_SENSOR_SAMPLE_DROPPED |
                MR_SENSOR_SAMPLE_STALE
            : 0u;
        states[stateIndex] = state;
        MRSensorSampleMetadataGPU sampleMetadata =
            metadata[stateIndex];
        sampleMetadata.ageValidityAndLayout.x =
            state.timestampAgeValidity.z;
        sampleMetadata.ageValidityAndLayout.y =
            state.timestampAgeValidity.w;
        metadata[stateIndex] = sampleMetadata;
        return;
    }

    float3 parentPosition;
    float4 parentOrientation;
    float3 parentLinearVelocity = float3(0.0f);
    float3 parentAngularVelocity = float3(0.0f);
    if (descriptor.identity.y ==
        MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK) {
        const uint bodyStride =
            dispatch.controlPeriodAndStrides.w;
        const MRArticulatedBodyPoseGPU pose =
            bodyPoses[
                environment * bodyStride + descriptor.identity.z
            ];
        parentPosition = pose.position.xyz;
        parentOrientation = pose.orientation;
        if (twistSensor || imuSensor) {
            const MRBodyStateGPU body = bodyStates[
                environment * bodyStride + descriptor.identity.z
            ];
            parentLinearVelocity =
                body.linearVelocityAndInverseMass.xyz;
            parentAngularVelocity = body.angularVelocity.xyz;
        }
    } else if (descriptor.identity.y ==
               MR_WORLD_SENSOR_PARENT_RIGID_BODY) {
        const uint sceneStride =
            dispatch.controlPeriodAndStrides.z;
        const MRBodyStateGPU body = sceneBodies[
            environment * sceneStride + descriptor.source.x
        ];
        parentPosition = body.position.xyz;
        parentOrientation = body.orientation;
        parentLinearVelocity =
            body.linearVelocityAndInverseMass.xyz;
        parentAngularVelocity = body.angularVelocity.xyz;
    } else {
        state.timestampAgeValidity.w =
            MR_SENSOR_SAMPLE_NONFINITE;
        states[stateIndex] = state;
        return;
    }

    float4 parentUnit = float4(0.0f, 0.0f, 0.0f, 1.0f);
    float4 sensorOrientation = parentUnit;
    const bool poseValid =
        all(isfinite(parentPosition)) &&
        normalizedQuaternion(parentOrientation, parentUnit) &&
        normalizedQuaternion(
            quaternionMultiply(
                parentUnit,
                descriptor.localOrientation
            ),
            sensorOrientation
        );
    const float3 sensorPosition = parentPosition +
        quaternionRotate(parentUnit, descriptor.localPosition.xyz);
    if (!poseValid || !all(isfinite(sensorPosition)) ||
        ((twistSensor || imuSensor) &&
         (!all(isfinite(parentLinearVelocity)) ||
          !all(isfinite(parentAngularVelocity))))) {
        state.timestampAgeValidity.w =
            MR_SENSOR_SAMPLE_NONFINITE;
        states[stateIndex] = state;
        MRSensorSampleMetadataGPU failed = metadata[stateIndex];
        failed.ageValidityAndLayout.y =
            MR_SENSOR_SAMPLE_NONFINITE;
        metadata[stateIndex] = failed;
        return;
    }

    const uint writeSlot =
        static_cast<uint>(sequence % historyLength);
    const uint writeBase =
        historyBase + writeSlot * outputCount;
    if (poseSensor) {
        history[writeBase + 0u] = sensorPosition.x;
        history[writeBase + 1u] = sensorPosition.y;
        history[writeBase + 2u] = sensorPosition.z;
        history[writeBase + 3u] = sensorOrientation.x;
        history[writeBase + 4u] = sensorOrientation.y;
        history[writeBase + 5u] = sensorOrientation.z;
        history[writeBase + 6u] = sensorOrientation.w;
    } else if (twistSensor) {
        const float3 sensorOffset = quaternionRotate(
            parentUnit,
            descriptor.localPosition.xyz
        );
        const float3 pointVelocity =
            parentLinearVelocity +
            cross(parentAngularVelocity, sensorOffset);
        history[writeBase + 0u] = pointVelocity.x;
        history[writeBase + 1u] = pointVelocity.y;
        history[writeBase + 2u] = pointVelocity.z;
        history[writeBase + 3u] = parentAngularVelocity.x;
        history[writeBase + 4u] = parentAngularVelocity.y;
        history[writeBase + 5u] = parentAngularVelocity.z;
    } else if (imuSensor) {
        const float3 sensorOffset = quaternionRotate(
            parentUnit,
            descriptor.localPosition.xyz
        );
        const float3 pointVelocity =
            parentLinearVelocity +
            cross(parentAngularVelocity, sensorOffset);
        const ulong previousTimestamp = unpack64(
            state.previousSampleTimestamp.x,
            state.previousSampleTimestamp.y
        );
        const bool previousValid =
            state.previousSampleTimestamp.z != 0u;
        float3 pointAcceleration = float3(0.0f);
        if (previousValid && timestamp > previousTimestamp) {
            const float elapsedSeconds =
                static_cast<float>(timestamp - previousTimestamp) *
                1.0e-9f;
            if (!(elapsedSeconds > 0.0f) ||
                !isfinite(elapsedSeconds)) {
                state.timestampAgeValidity.w =
                    MR_SENSOR_SAMPLE_NONFINITE;
                states[stateIndex] = state;
                MRSensorSampleMetadataGPU failed =
                    metadata[stateIndex];
                failed.ageValidityAndLayout.y =
                    MR_SENSOR_SAMPLE_NONFINITE;
                metadata[stateIndex] = failed;
                return;
            }
            pointAcceleration =
                (pointVelocity -
                 state.previousPointVelocity.xyz) /
                elapsedSeconds;
        }
        const float3 specificForceLocal =
            quaternionRotateInverse(
                sensorOrientation,
                pointAcceleration - world.gravityAndTimestep.xyz
            );
        const float3 angularVelocityLocal =
            quaternionRotateInverse(
                sensorOrientation,
                parentAngularVelocity
            );
        if (!all(isfinite(pointVelocity)) ||
            !all(isfinite(specificForceLocal)) ||
            !all(isfinite(angularVelocityLocal)) ||
            !all(isfinite(world.gravityAndTimestep.xyz))) {
            state.timestampAgeValidity.w =
                MR_SENSOR_SAMPLE_NONFINITE;
            states[stateIndex] = state;
            MRSensorSampleMetadataGPU failed = metadata[stateIndex];
            failed.ageValidityAndLayout.y =
                MR_SENSOR_SAMPLE_NONFINITE;
            metadata[stateIndex] = failed;
            return;
        }
        history[writeBase + 0u] = specificForceLocal.x;
        history[writeBase + 1u] = specificForceLocal.y;
        history[writeBase + 2u] = specificForceLocal.z;
        history[writeBase + 3u] = angularVelocityLocal.x;
        history[writeBase + 4u] = angularVelocityLocal.y;
        history[writeBase + 5u] = angularVelocityLocal.z;
        state.previousSampleTimestamp = {
            low32(timestamp),
            high32(timestamp),
            1u,
            0u,
        };
        state.previousPointVelocity = float4(
            pointVelocity,
            0.0f
        );
    } else {
        float3 forceWorld = float3(0.0f);
        float3 torqueWorld = float3(0.0f);
        float3 tangentialImpulseWorld = float3(0.0f);
        float normalImpulse = 0.0f;
        float maximumPenetration = 0.0f;
        uint contactCount = 0u;
        float inverseTimestep = 0.0f;
        bool contactsValid = reset;
        if (!reset &&
            contactDispatch.abiVersion ==
                MR_METAL_WORLD_CONTACT_ABI_VERSION &&
            contactDispatch.environmentCount == environmentCount &&
            contactDispatch.timestepAndBias.x > 0.0f &&
            isfinite(contactDispatch.timestepAndBias.x)) {
            const MRMetalWorldContactStatusGPU contactStatus =
                contactStatuses[environment];
            contactsValid = contactStatus.code == MR_STEP_SUCCESS;
            const uint publishedConstraints = contactsValid
                ? min(
                      contactStatus.requiredConstraints,
                      contactDispatch.constraintCapacity
                  )
                : 0u;
            const uint contactBase =
                environment * contactDispatch.constraintStride;
            const uint parentBody = descriptor.identity.z;
            inverseTimestep =
                1.0f / contactDispatch.timestepAndBias.x;
            for (uint constraintIndex = min(
                     contactDispatch.authoredConstraintCount,
                     publishedConstraints
                 );
                 constraintIndex < publishedConstraints;
                 ++constraintIndex) {
                const MRContactConstraintGPU constraint =
                    contacts[contactBase + constraintIndex];
                if ((constraint.flags &
                     (MR_CONSTRAINT_FLAG_DISABLED |
                      MR_CONSTRAINT_FLAG_GENERALIZED)) != 0u) {
                    continue;
                }
                const bool parentA =
                    constraint.bodyA == parentBody;
                const bool parentB =
                    constraint.bodyB == parentBody;
                if (parentA == parentB) {
                    continue;
                }
                const uint counterpart = parentA
                    ? constraint.bodyB
                    : constraint.bodyA;
                if (contactStateSensor &&
                    !contactFilterAllows(
                        filterBodies,
                        descriptor.filter.x,
                        descriptor.filter.y,
                        counterpart
                    )) {
                    continue;
                }
                const float3 normal = normalizedOr(
                    constraint.normal.xyz,
                    float3(0.0f, 0.0f, 1.0f)
                );
                const float3 authoredTangent =
                    constraint.tangent.xyz -
                    normal * dot(normal, constraint.tangent.xyz);
                const float3 tangent = normalizedOr(
                    authoredTangent,
                    stableContactTangent(normal)
                );
                const float3 bitangent = cross(normal, tangent);
                const float3 normalImpulseOnA =
                    -normal * constraint.impulses.x;
                const float3 tangentImpulseOnA = -(
                    tangent * constraint.impulses.y +
                    bitangent * constraint.impulses.z
                );
                const float3 impulseOnA =
                    normalImpulseOnA + tangentImpulseOnA;
                const float3 impulseOnParent =
                    parentA ? impulseOnA : -impulseOnA;
                const float3 tangentImpulseOnParent =
                    parentA
                    ? tangentImpulseOnA
                    : -tangentImpulseOnA;
                const float3 contactForce =
                    impulseOnParent * inverseTimestep;
                forceWorld += contactForce;
                torqueWorld += cross(
                    constraint.pointAndSeparation.xyz -
                        sensorPosition,
                    contactForce
                );
                torqueWorld +=
                    (parentA ? -1.0f : 1.0f) *
                    normal * constraint.impulses.w *
                    inverseTimestep;
                normalImpulse += max(
                    constraint.impulses.x,
                    0.0f
                );
                tangentialImpulseWorld +=
                    tangentImpulseOnParent;
                maximumPenetration = max(
                    maximumPenetration,
                    max(
                        -constraint.pointAndSeparation.w,
                        0.0f
                    )
                );
                ++contactCount;
            }
        }
        if (contactStateSensor) {
            const float normalForce =
                normalImpulse * inverseTimestep;
            const float tangentialForce =
                length(tangentialImpulseWorld) * inverseTimestep;
            if (!contactsValid ||
                !isfinite(normalForce) ||
                !isfinite(tangentialForce) ||
                !isfinite(maximumPenetration)) {
                state.timestampAgeValidity.w =
                    MR_SENSOR_SAMPLE_NONFINITE;
                states[stateIndex] = state;
                MRSensorSampleMetadataGPU failed =
                    metadata[stateIndex];
                failed.ageValidityAndLayout.y =
                    MR_SENSOR_SAMPLE_NONFINITE;
                metadata[stateIndex] = failed;
                return;
            }
            history[writeBase + 0u] =
                contactCount != 0u ? 1.0f : 0.0f;
            history[writeBase + 1u] =
                static_cast<float>(contactCount);
            history[writeBase + 2u] = normalForce;
            history[writeBase + 3u] = tangentialForce;
            history[writeBase + 4u] = maximumPenetration;
        } else {
            const float3 forceLocal = quaternionRotateInverse(
                sensorOrientation,
                forceWorld
            );
            const float3 torqueLocal = quaternionRotateInverse(
                sensorOrientation,
                torqueWorld
            );
            if (!contactsValid ||
                !all(isfinite(forceLocal)) ||
                !all(isfinite(torqueLocal))) {
                state.timestampAgeValidity.w =
                    MR_SENSOR_SAMPLE_NONFINITE;
                states[stateIndex] = state;
                MRSensorSampleMetadataGPU failed =
                    metadata[stateIndex];
                failed.ageValidityAndLayout.y =
                    MR_SENSOR_SAMPLE_NONFINITE;
                metadata[stateIndex] = failed;
                return;
            }
            history[writeBase + 0u] = forceLocal.x;
            history[writeBase + 1u] = forceLocal.y;
            history[writeBase + 2u] = forceLocal.z;
            history[writeBase + 3u] = torqueLocal.x;
            history[writeBase + 4u] = torqueLocal.y;
            history[writeBase + 5u] = torqueLocal.z;
        }
    }
    ++sequence;

    const uint latencySamples = descriptor.schedule.w;
    uint validity = reset ? MR_SENSOR_SAMPLE_RESET : 0u;
    ulong publishedTimestamp = 0u;
    if (sequence > latencySamples) {
        const ulong publishedSequence =
            sequence - 1u - latencySamples;
        const uint readSlot = static_cast<uint>(
            publishedSequence % historyLength
        );
        const uint readBase =
            historyBase + readSlot * outputCount;
        for (uint value = 0u;
             value < outputCount;
             ++value) {
            outputs[outputBase + value] =
                history[readBase + value];
        }
        const ulong latencyTicks =
            static_cast<ulong>(latencySamples) * samplePeriod;
        publishedTimestamp = timestamp >= latencyTicks
            ? timestamp - latencyTicks
            : 0u;
        const ulong randomIdentity =
            sensorRandomIdentity(descriptor.randomIdentity);
        const bool dropped = descriptor.noise.z > 0.0f &&
            mr_sensor_counter_uniform(
                dispatch.seed,
                environment,
                state.randomIdentity.x,
                randomIdentity,
                publishedSequence,
                0x7fffffffu,
                4u
            ) < descriptor.noise.z;
        if (dropped) {
            for (uint value = 0u;
                 value < outputCount;
                 ++value) {
                outputs[outputBase + value] = 0.0f;
            }
            validity |=
                MR_SENSOR_SAMPLE_FRESH |
                MR_SENSOR_SAMPLE_DROPPED;
        } else if (!corruptSensorSample(
                       outputs,
                       outputBase,
                       outputCount,
                       descriptor.identity.x,
                       descriptor.noise.x,
                       descriptor.noise.y,
                       dispatch.seed,
                       environment,
                       state.randomIdentity.x,
                       randomIdentity,
                       publishedSequence
                   )) {
            for (uint value = 0u;
                 value < outputCount;
                 ++value) {
                outputs[outputBase + value] = 0.0f;
            }
            validity |=
                MR_SENSOR_SAMPLE_FRESH |
                MR_SENSOR_SAMPLE_NONFINITE;
        } else {
            validity |=
                MR_SENSOR_SAMPLE_VALID |
                MR_SENSOR_SAMPLE_FRESH;
        }
    } else {
        for (uint value = 0u;
             value < outputCount;
             ++value) {
            outputs[outputBase + value] = 0.0f;
        }
    }

    state.phaseAndSequence = {
        low32(phase),
        high32(phase),
        low32(sequence),
        high32(sequence),
    };
    state.timestampAgeValidity = {
        low32(timestamp),
        high32(timestamp),
        0u,
        validity,
    };
    states[stateIndex] = state;
    metadata[stateIndex] = {
        {
            low32(sequence),
            high32(sequence),
            low32(publishedTimestamp),
            high32(publishedTimestamp),
        },
        {
            0u,
            validity,
            descriptor.output.x,
            descriptor.output.y,
        },
    };
}
