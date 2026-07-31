#include <metal_stdlib>

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

} // namespace

// First canonical SensorIR executor: parent-frame pose sampling across the
// two control boundaries. Reset-only execution seeds a new episode before its
// first action; advance execution samples the accepted post-physics state for
// the next action. One thread owns one environment/sensor history ring, so
// reset and publication are transactional without atomics. Presentation and
// tactile domains remain separate until their existing native kernels are
// folded into this schedule; descriptors for those domains are rejected by
// the host execution gate rather than silently skipped in production.
kernel void mr_sensor_sample_control_boundary(
    constant MRSensorDispatchGPU& dispatch
        [[buffer(MR_SENSOR_SAMPLE_DISPATCH)]],
    device const MRSensorProgramHeaderGPU& program
        [[buffer(MR_SENSOR_SAMPLE_PROGRAM)]],
    device const MRSensorDescriptorGPU* descriptors
        [[buffer(MR_SENSOR_SAMPLE_DESCRIPTORS)]],
    constant MRMetalWorldPassGPU& pass
        [[buffer(MR_SENSOR_SAMPLE_PASS)]],
    device const uint* resetMasks
        [[buffer(MR_SENSOR_SAMPLE_RESET_MASKS)]],
    device const MRArticulatedBodyPoseGPU* bodyPoses
        [[buffer(MR_SENSOR_SAMPLE_BODY_POSES)]],
    device const MRBodyStateGPU* sceneBodies
        [[buffer(MR_SENSOR_SAMPLE_SCENE_BODIES)]],
    device MRSensorRuntimeStateGPU* states
        [[buffer(MR_SENSOR_SAMPLE_STATES)]],
    device float* history
        [[buffer(MR_SENSOR_SAMPLE_HISTORY)]],
    device float* outputs
        [[buffer(MR_SENSOR_SAMPLE_OUTPUTS)]],
    device MRSensorSampleMetadataGPU* metadata
        [[buffer(MR_SENSOR_SAMPLE_METADATA)]],
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
    if (descriptor.identity.x != MR_WORLD_SENSOR_STATE ||
        descriptor.schedule.z !=
            MR_WORLD_SENSOR_PHASE_PRE_CONTROL ||
        descriptor.source.w != 0u ||
        descriptor.output.y != 7u ||
        descriptor.output.w == 0u) {
        return;
    }

    const uint stateIndex = environment * sensorCount + sensorIndex;
    MRSensorRuntimeStateGPU state = states[stateIndex];
    const bool reset =
        dispatch.counts.z == MR_SENSOR_EXECUTION_RESET_ONLY &&
        (dispatch.counts.w & 1u) != 0u &&
        resetMasks[
            pass.controlStep * environmentCount + environment
        ] != 0u;
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
        state = {};
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
    } else if (descriptor.identity.y ==
               MR_WORLD_SENSOR_PARENT_RIGID_BODY) {
        const uint sceneStride =
            dispatch.controlPeriodAndStrides.z;
        const MRBodyStateGPU body = sceneBodies[
            environment * sceneStride + descriptor.source.x
        ];
        parentPosition = body.position.xyz;
        parentOrientation = body.orientation;
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
    if (!poseValid || !all(isfinite(sensorPosition))) {
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
    history[writeBase + 0u] = sensorPosition.x;
    history[writeBase + 1u] = sensorPosition.y;
    history[writeBase + 2u] = sensorPosition.z;
    history[writeBase + 3u] = sensorOrientation.x;
    history[writeBase + 4u] = sensorOrientation.y;
    history[writeBase + 5u] = sensorOrientation.z;
    history[writeBase + 6u] = sensorOrientation.w;
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
        validity |=
            MR_SENSOR_SAMPLE_VALID | MR_SENSOR_SAMPLE_FRESH;
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
