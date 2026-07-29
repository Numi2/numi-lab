#include <metal_stdlib>

#include "metalrobo/engine_types.h"
#include "metalrobo/r2s2r_types.h"
#include "metalrobo/world_compiler_types.h"

using namespace metal;

namespace {

constant uint kPhiloxM0 = 0xd2511f53u;
constant uint kPhiloxM1 = 0xcd9e8d57u;
constant uint kPhiloxW0 = 0x9e3779b9u;
constant uint kPhiloxW1 = 0xbb67ae85u;

uint4 philoxRound(const uint4 counter, const uint2 key) {
    const ulong product0 = ulong(kPhiloxM0) * ulong(counter.x);
    const ulong product1 = ulong(kPhiloxM1) * ulong(counter.z);
    return uint4(
        uint(product1 >> 32u) ^ counter.y ^ key.x,
        uint(product1),
        uint(product0 >> 32u) ^ counter.w ^ key.y,
        uint(product0)
    );
}

uint4 philox(
    const uint environment,
    const uint variation,
    const MRWorldVariationGPU descriptor,
    const MRWorldFamilySampleUniformsGPU uniforms
) {
    uint4 counter(
        environment,
        variation,
        descriptor.random.x,
        descriptor.random.y ^ descriptor.random.z
    );
    uint2 key(uniforms.identity.x, uniforms.identity.y);
    for (uint round = 0u; round < 10u; ++round) {
        counter = philoxRound(counter, key);
        key.x += kPhiloxW0;
        key.y += kPhiloxW1;
    }
    return counter;
}

float uniform01(const uint bits) {
    return (float(bits) + 0.5f) * (1.0f / 4294967296.0f);
}

// Acklam's inverse-normal approximation. Sampling every continuous
// distribution from an explicit quantile makes aligned particles and
// feedback regions representation-independent and replayable.
float inverseNormalCDF(const float probability) {
    const float p = clamp(probability, 1.0e-7f, 1.0f - 1.0e-7f);
    constexpr float lowerTail = 0.02425f;
    constexpr float upperTail = 1.0f - lowerTail;
    if (p < lowerTail) {
        const float q = sqrt(-2.0f * log(p));
        const float numerator =
            (((((-0.007784894002430293f * q -
                  0.3223964580411365f) * q -
                 2.400758277161838f) * q -
                2.549732539343734f) * q +
               4.374664141464968f) * q +
              2.938163982698783f);
        const float denominator =
            ((((0.007784695709041462f * q +
                0.3224671290700398f) * q +
               2.445134137142996f) * q +
              3.754408661907416f) * q +
             1.0f);
        return numerator / denominator;
    }
    if (p > upperTail) {
        const float q = sqrt(-2.0f * log(1.0f - p));
        const float numerator =
            (((((-0.007784894002430293f * q -
                  0.3223964580411365f) * q -
                 2.400758277161838f) * q -
                2.549732539343734f) * q +
               4.374664141464968f) * q +
              2.938163982698783f);
        const float denominator =
            ((((0.007784695709041462f * q +
                0.3224671290700398f) * q +
               2.445134137142996f) * q +
              3.754408661907416f) * q +
             1.0f);
        return -numerator / denominator;
    }
    const float q = p - 0.5f;
    const float r = q * q;
    const float numerator =
        (((((-39.69683028665376f * r +
              220.9460984245205f) * r -
             275.9285104469687f) * r +
            138.3577518672690f) * r -
           30.66479806614716f) * r +
          2.506628277459239f) * q;
    const float denominator =
        (((((-54.47609879822406f * r +
              161.5858368580409f) * r -
             155.6989798598866f) * r +
            66.80131188771972f) * r -
           13.28068155288572f) * r +
          1.0f);
    return numerator / denominator;
}

struct SampledValue {
    float scalar;
    uint categorical;
    uint categoricalOrdinal;
    float quantile;
};

SampledValue sampleValueAtQuantile(
    const MRWorldVariationGPU descriptor,
    const device uint* categoricalValues,
    const uint categoricalCount,
    const float requestedQuantile
) {
    const float quantile = clamp(
        requestedQuantile,
        1.0e-7f,
        1.0f - 1.0e-7f
    );
    SampledValue result{0.0f, 0u, 0u, 0.5f};
    switch (descriptor.binding.y) {
    case MR_WORLD_DISTRIBUTION_CONSTANT:
        result.scalar = descriptor.parameters.x;
        return result;
    case MR_WORLD_DISTRIBUTION_UNIFORM:
        result.quantile = quantile;
        result.scalar = mix(
            descriptor.parameters.x,
            descriptor.parameters.y,
            quantile
        );
        return result;
    case MR_WORLD_DISTRIBUTION_LOG_UNIFORM:
        result.quantile = quantile;
        result.scalar = exp(mix(
            log(descriptor.parameters.x),
            log(descriptor.parameters.y),
            quantile
        ));
        return result;
    case MR_WORLD_DISTRIBUTION_NORMAL_CLAMPED: {
        result.quantile = quantile;
        const float normal = inverseNormalCDF(quantile);
        result.scalar = clamp(
            descriptor.parameters.x + descriptor.parameters.y * normal,
            descriptor.parameters.z,
            descriptor.parameters.w
        );
        return result;
    }
    case MR_WORLD_DISTRIBUTION_CATEGORICAL: {
        const uint first = descriptor.categorical.x;
        const uint count = descriptor.categorical.y;
        if (count == 0u || first > categoricalCount ||
            count > categoricalCount - first) {
            return result;
        }
        const uint selected = min(
            uint(quantile * float(count)),
            count - 1u
        );
        result.quantile = quantile;
        result.categorical = categoricalValues[first + selected];
        result.categoricalOrdinal = selected;
        result.scalar = float(result.categorical);
        return result;
    }
    default:
        return result;
    }
}

SampledValue sampleValue(
    const MRWorldVariationGPU descriptor,
    const device uint* categoricalValues,
    const uint categoricalCount,
    const uint4 random
) {
    return sampleValueAtQuantile(
        descriptor,
        categoricalValues,
        categoricalCount,
        uniform01(random.x)
    );
}

ulong join64(const uint2 words) {
    return (ulong(words.y) << 32u) | ulong(words.x);
}

ulong mix64(ulong value) {
    value ^= value >> 30u;
    value *= 0xbf58476d1ce4e5b9ul;
    value ^= value >> 27u;
    value *= 0x94d049bb133111ebul;
    value ^= value >> 31u;
    return value;
}

uint4 adaptiveRandom(
    const uint environment,
    const uint stream,
    const MRWorldVariationGPU descriptor,
    const MRWorldFamilySampleUniformsGPU base,
    const MRWorldAdaptiveSampleUniformsGPU adaptive
) {
    MRWorldVariationGPU salted = descriptor;
    const ulong episode = join64(adaptive.identity.xy);
    salted.random.x ^= uint(episode);
    salted.random.y ^= uint(episode >> 32u);
    salted.random.z ^= stream * 0x9e3779b9u;
    return philox(environment, stream, salted, base);
}

uint selectAlignmentParticle(
    const device MRWorldAlignmentParticleGPU* particles,
    const uint particleCount,
    const float selector
) {
    if (particleCount == 0u) {
        return MR_INVALID_INDEX;
    }
    uint lower = 0u;
    uint upper = particleCount;
    while (lower < upper) {
        const uint middle = lower + ((upper - lower) >> 1u);
        if (selector <= particles[middle].statistics.y) {
            upper = middle;
        } else {
            lower = middle + 1u;
        }
    }
    return min(lower, particleCount - 1u);
}

uint selectFeedbackRegion(
    const device MRWorldFeedbackRegionGPU* regions,
    const uint regionCount,
    const uint kind,
    const float selector
) {
    uint fallback = MR_INVALID_INDEX;
    for (uint region = 0u; region < regionCount; ++region) {
        if (regions[region].identity.x != kind) {
            continue;
        }
        fallback = region;
        if (selector <= regions[region].statistics.y) {
            return region;
        }
    }
    return fallback;
}

float4 quaternionProduct(const float4 a, const float4 b) {
    return float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

float4 axisAngle(const float3 axis, const float angle) {
    const float halfAngle = 0.5f * angle;
    return float4(axis * sin(halfAngle), cos(halfAngle));
}

void applyOrientationDelta(
    thread float4& orientation,
    const uint target,
    const float value
) {
    float3 axis(0.0f);
    switch (target) {
    case MR_WORLD_TARGET_ASSET_ORIENTATION_ROLL:
    case MR_WORLD_TARGET_SENSOR_ORIENTATION_ROLL:
        axis = float3(1.0f, 0.0f, 0.0f);
        break;
    case MR_WORLD_TARGET_ASSET_ORIENTATION_PITCH:
    case MR_WORLD_TARGET_SENSOR_ORIENTATION_PITCH:
        axis = float3(0.0f, 1.0f, 0.0f);
        break;
    case MR_WORLD_TARGET_ASSET_ORIENTATION_YAW:
    case MR_WORLD_TARGET_SENSOR_ORIENTATION_YAW:
        axis = float3(0.0f, 0.0f, 1.0f);
        break;
    default:
        return;
    }
    orientation = normalize(
        quaternionProduct(orientation, axisAngle(axis, value))
    );
}

void applyAssetVariation(
    thread MRWorldAssetInstanceGPU& asset,
    const uint target,
    const SampledValue value
) {
    switch (target) {
    case MR_WORLD_TARGET_ASSET_POSITION_X:
        asset.positionAndScale.x += value.scalar;
        break;
    case MR_WORLD_TARGET_ASSET_POSITION_Y:
        asset.positionAndScale.y += value.scalar;
        break;
    case MR_WORLD_TARGET_ASSET_POSITION_Z:
        asset.positionAndScale.z += value.scalar;
        break;
    case MR_WORLD_TARGET_ASSET_SCALE:
        asset.positionAndScale.w *= value.scalar;
        break;
    case MR_WORLD_TARGET_ASSET_ORIENTATION_ROLL:
    case MR_WORLD_TARGET_ASSET_ORIENTATION_PITCH:
    case MR_WORLD_TARGET_ASSET_ORIENTATION_YAW:
        applyOrientationDelta(asset.orientation, target, value.scalar);
        break;
    case MR_WORLD_TARGET_ASSET_MASS_SCALE:
        asset.physical.x *= value.scalar;
        break;
    case MR_WORLD_TARGET_ASSET_FRICTION_SCALE:
        asset.physical.y *= value.scalar;
        break;
    case MR_WORLD_TARGET_ASSET_RESTITUTION_SCALE:
        asset.physical.z *= value.scalar;
        break;
    case MR_WORLD_TARGET_ASSET_DAMPING_SCALE:
        asset.physical.w *= value.scalar;
        break;
    case MR_WORLD_TARGET_ROBOT_GAIN_SCALE:
        asset.controller.x *= value.scalar;
        break;
    case MR_WORLD_TARGET_ROBOT_DAMPING_SCALE:
        asset.controller.y *= value.scalar;
        break;
    case MR_WORLD_TARGET_ROBOT_LATENCY_SECONDS:
        asset.controller.z += value.scalar;
        break;
    case MR_WORLD_TARGET_ROBOT_PAYLOAD_SCALE:
        asset.controller.w *= value.scalar;
        break;
    case MR_WORLD_TARGET_ASSET_RENDER_ALTERNATIVE:
    case MR_WORLD_TARGET_CLUTTER_SET:
        asset.identity.y = value.categorical;
        break;
    case MR_WORLD_TARGET_ASSET_COLLISION_ALTERNATIVE:
        asset.identity.z = value.categorical;
        break;
    default:
        break;
    }
}

void applySensorVariation(
    thread MRWorldSensorInstanceGPU& sensor,
    const uint target,
    const float value
) {
    switch (target) {
    case MR_WORLD_TARGET_SENSOR_POSITION_X:
        sensor.positionAndFocalScale.x += value;
        break;
    case MR_WORLD_TARGET_SENSOR_POSITION_Y:
        sensor.positionAndFocalScale.y += value;
        break;
    case MR_WORLD_TARGET_SENSOR_POSITION_Z:
        sensor.positionAndFocalScale.z += value;
        break;
    case MR_WORLD_TARGET_SENSOR_ORIENTATION_ROLL:
    case MR_WORLD_TARGET_SENSOR_ORIENTATION_PITCH:
    case MR_WORLD_TARGET_SENSOR_ORIENTATION_YAW:
        applyOrientationDelta(sensor.orientation, target, value);
        break;
    case MR_WORLD_TARGET_SENSOR_FOCAL_SCALE:
        sensor.positionAndFocalScale.w *= value;
        break;
    case MR_WORLD_TARGET_SENSOR_LATENCY_SECONDS:
        sensor.noiseAndLatency.w += value;
        break;
    case MR_WORLD_TARGET_SENSOR_COLOR_NOISE:
        sensor.noiseAndLatency.x += value;
        break;
    case MR_WORLD_TARGET_SENSOR_DEPTH_NOISE:
        sensor.noiseAndLatency.y += value;
        break;
    case MR_WORLD_TARGET_SENSOR_DEPTH_DROPOUT:
        sensor.noiseAndLatency.z =
            clamp(sensor.noiseAndLatency.z + value, 0.0f, 1.0f);
        break;
    default:
        break;
    }
}

void applyAppearanceVariation(
    thread MRWorldAppearanceInstanceGPU& appearance,
    const uint target,
    const float value
) {
    switch (target) {
    case MR_WORLD_TARGET_APPEARANCE_EXPOSURE:
        appearance.colorAndLight.x += value;
        break;
    case MR_WORLD_TARGET_APPEARANCE_WHITE_BALANCE:
        appearance.colorAndLight.y *= value;
        break;
    case MR_WORLD_TARGET_APPEARANCE_SATURATION:
        appearance.colorAndLight.z *= value;
        break;
    case MR_WORLD_TARGET_APPEARANCE_LIGHT_INTENSITY:
        appearance.colorAndLight.w *= value;
        break;
    default:
        break;
    }
}

struct RowMatrix3 {
    float3 row0;
    float3 row1;
    float3 row2;
};

RowMatrix3 rotationMatrix(const float4 quaternion) {
    const float x = quaternion.x;
    const float y = quaternion.y;
    const float z = quaternion.z;
    const float w = quaternion.w;
    RowMatrix3 result;
    result.row0 = float3(
        1.0f - 2.0f * (y * y + z * z),
        2.0f * (x * y - z * w),
        2.0f * (x * z + y * w)
    );
    result.row1 = float3(
        2.0f * (x * y + z * w),
        1.0f - 2.0f * (x * x + z * z),
        2.0f * (y * z - x * w)
    );
    result.row2 = float3(
        2.0f * (x * z - y * w),
        2.0f * (y * z + x * w),
        1.0f - 2.0f * (x * x + y * y)
    );
    return result;
}

float3 multiply(
    const RowMatrix3 matrix,
    const float3 value
) {
    return float3(
        dot(matrix.row0, value),
        dot(matrix.row1, value),
        dot(matrix.row2, value)
    );
}

void writeScaledWorldInverseInertia(
    thread MRBodyStateGPU& state,
    const device MRBodyPropertiesGPU& properties,
    const float massScale
) {
    if (properties.motionType != MR_MOTION_DYNAMIC ||
        !(massScale > 0.0f)) {
        state.inverseInertiaWorldRow0 = float4(0.0f);
        state.inverseInertiaWorldRow1 = float4(0.0f);
        state.inverseInertiaWorldRow2 = float4(0.0f);
        return;
    }
    RowMatrix3 inverseBody;
    inverseBody.row0 = properties.inverseInertiaRow0.xyz;
    inverseBody.row1 = properties.inverseInertiaRow1.xyz;
    inverseBody.row2 = properties.inverseInertiaRow2.xyz;
    const RowMatrix3 rotation = rotationMatrix(state.orientation);
    const float3 rows[3] = {
        rotation.row0,
        rotation.row1,
        rotation.row2,
    };
    float3 worldRows[3];
    for (uint row = 0u; row < 3u; ++row) {
        worldRows[row] = float3(
            dot(rows[row], multiply(inverseBody, rows[0])),
            dot(rows[row], multiply(inverseBody, rows[1])),
            dot(rows[row], multiply(inverseBody, rows[2]))
        ) / massScale;
    }
    state.inverseInertiaWorldRow0 = float4(worldRows[0], 0.0f);
    state.inverseInertiaWorldRow1 = float4(worldRows[1], 0.0f);
    state.inverseInertiaWorldRow2 = float4(worldRows[2], 0.0f);
}

} // namespace

kernel void mr_world_family_sample(
    const device MRWorldAssetInstanceGPU* baseAssets [[buffer(0)]],
    const device MRWorldSensorInstanceGPU* baseSensors [[buffer(1)]],
    const device MRWorldAppearanceInstanceGPU* baseAppearances [[buffer(2)]],
    const device MRWorldVariationGPU* variations [[buffer(3)]],
    const device uint* categoricalValues [[buffer(4)]],
    constant MRWorldFamilySampleUniformsGPU& uniforms [[buffer(5)]],
    device MRWorldInstanceHeaderGPU* instances [[buffer(6)]],
    device MRWorldAssetInstanceGPU* assets [[buffer(7)]],
    device MRWorldSensorInstanceGPU* sensors [[buffer(8)]],
    device MRWorldAppearanceInstanceGPU* appearances [[buffer(9)]],
    device MRWorldScenarioHeaderGPU* scenarioHeaders [[buffer(10)]],
    device MRWorldScenarioValueGPU* scenarioValues [[buffer(11)]],
    constant MRWorldAdaptiveSampleUniformsGPU&
        adaptive [[buffer(12)]],
    const device MRWorldAlignmentParticleGPU*
        alignmentParticles [[buffer(13)]],
    const device float* alignmentQuantiles [[buffer(14)]],
    const device MRWorldFeedbackRegionGPU*
        feedbackRegions [[buffer(15)]],
    const device float4* feedbackBounds [[buffer(16)]],
    uint environment [[thread_position_in_grid]]
) {
    if (environment >= uniforms.counts.x ||
        uniforms.program.w != MR_WORLD_COMPILER_ABI_VERSION ||
        adaptive.abi.x != MR_R2S2R_ABI_VERSION ||
        adaptive.counts.z != uniforms.program.x) {
        return;
    }

    const uint firstAsset = environment * uniforms.counts.y;
    const uint firstSensor = environment * uniforms.counts.z;
    const uint firstAppearance = environment * uniforms.counts.w;
    const ulong familyFingerprint =
        (ulong(uniforms.identity.w) << 32u) |
        ulong(uniforms.identity.z);
    const ulong seed =
        (ulong(uniforms.identity.y) << 32u) |
        ulong(uniforms.identity.x);
    const ulong episodeCounter =
        join64(adaptive.identity.xy) + ulong(environment);
    const ulong alignmentFingerprint = join64(adaptive.identity.zw);
    const ulong feedbackFingerprint = join64(adaptive.provenance.xy);

    MRWorldVariationGPU selectionDescriptor = {};
    selectionDescriptor.random = uint4(
        uint(alignmentFingerprint),
        uint(alignmentFingerprint >> 32u),
        uint(feedbackFingerprint),
        uint(feedbackFingerprint >> 32u)
    );
    const uint4 selectionRandom = adaptiveRandom(
        environment,
        uniforms.program.x + 0x10001u,
        selectionDescriptor,
        uniforms,
        adaptive
    );
    const float sourceSelector = uniform01(selectionRandom.x);
    uint source = MR_WORLD_SAMPLE_BROAD;
    if (adaptive.counts.w == MR_WORLD_SAMPLING_CURRICULUM) {
        const float broadEnd = adaptive.mixture.x;
        const float failureEnd = broadEnd + adaptive.mixture.y;
        if (sourceSelector >= broadEnd &&
            sourceSelector < failureEnd) {
            source = MR_WORLD_SAMPLE_FAILURE;
        } else if (sourceSelector >= failureEnd) {
            source = MR_WORLD_SAMPLE_UNCERTAINTY;
        }
    } else if (adaptive.counts.x > 0u) {
        source = MR_WORLD_SAMPLE_ALIGNMENT;
    }

    uint selectedIndex = MR_INVALID_INDEX;
    uint selectedParticle = MR_INVALID_INDEX;
    uint selectedRegion = MR_INVALID_INDEX;
    float componentWeight = 1.0f;
    if (source == MR_WORLD_SAMPLE_FAILURE ||
        source == MR_WORLD_SAMPLE_UNCERTAINTY) {
        const uint kind = source == MR_WORLD_SAMPLE_FAILURE
            ? MR_FEEDBACK_REGION_FAILURE
            : MR_FEEDBACK_REGION_UNCERTAINTY;
        selectedRegion = selectFeedbackRegion(
            feedbackRegions,
            adaptive.counts.y,
            kind,
            uniform01(selectionRandom.y)
        );
        if (selectedRegion == MR_INVALID_INDEX) {
            source = adaptive.counts.x > 0u
                ? (adaptive.counts.w == MR_WORLD_SAMPLING_COVERAGE
                    ? MR_WORLD_SAMPLE_ALIGNMENT
                    : MR_WORLD_SAMPLE_BROAD)
                : MR_WORLD_SAMPLE_BROAD;
        } else {
            selectedIndex = selectedRegion;
            componentWeight =
                feedbackRegions[selectedRegion].statistics.x;
        }
    }
    if (source == MR_WORLD_SAMPLE_BROAD ||
        source == MR_WORLD_SAMPLE_ALIGNMENT) {
        selectedParticle =
            adaptive.counts.w == MR_WORLD_SAMPLING_REPLAY
            ? environment
            : selectAlignmentParticle(
                alignmentParticles,
                adaptive.counts.x,
                uniform01(selectionRandom.z)
            );
        if (selectedParticle != MR_INVALID_INDEX) {
            selectedIndex = selectedParticle;
            componentWeight =
                alignmentParticles[selectedParticle].statistics.x;
        }
    }

    const ulong scenarioKey = mix64(
        familyFingerprint ^ mix64(seed) ^
        mix64(ulong(environment) + 1ul) ^
        mix64(episodeCounter) ^
        mix64(alignmentFingerprint) ^
        mix64(feedbackFingerprint) ^
        ulong(adaptive.counts.w) ^
        (ulong(source) << 32u)
    );

    instances[environment].ranges = uint4(
        firstAsset,
        uniforms.counts.y,
        firstSensor,
        uniforms.counts.z
    );
    instances[environment].program = uint4(
        firstAppearance,
        uniforms.counts.w,
        0u,
        0u
    );
    instances[environment].identity = uint4(
        uint(scenarioKey),
        uint(scenarioKey >> 32u),
        uniforms.program.z,
        MR_WORLD_COMPILER_ABI_VERSION
    );
    scenarioHeaders[environment].identity = uint4(
        uint(scenarioKey),
        uint(scenarioKey >> 32u),
        uint(episodeCounter),
        uint(episodeCounter >> 32u)
    );
    scenarioHeaders[environment].provenance = uint4(
        uint(alignmentFingerprint),
        uint(alignmentFingerprint >> 32u),
        uint(feedbackFingerprint),
        uint(feedbackFingerprint >> 32u)
    );
    scenarioHeaders[environment].sampling = uint4(
        adaptive.counts.w,
        source,
        selectedIndex,
        MR_R2S2R_ABI_VERSION
    );

    for (uint index = 0u; index < uniforms.counts.y; ++index) {
        assets[firstAsset + index] = baseAssets[index];
    }
    for (uint index = 0u; index < uniforms.counts.z; ++index) {
        sensors[firstSensor + index] = baseSensors[index];
    }
    for (uint index = 0u; index < uniforms.counts.w; ++index) {
        appearances[firstAppearance + index] = baseAppearances[index];
    }

    for (uint variationIndex = 0u;
         variationIndex < uniforms.program.x;
         ++variationIndex) {
        const MRWorldVariationGPU descriptor = variations[variationIndex];
        const uint4 random = adaptiveRandom(
            environment,
            variationIndex,
            descriptor,
            uniforms,
            adaptive
        );
        float quantile = uniform01(random.x);
        if (selectedRegion != MR_INVALID_INDEX) {
            const float4 bounds = feedbackBounds[
                selectedRegion * uniforms.program.x + variationIndex
            ];
            quantile = mix(bounds.x, bounds.y, quantile);
        } else if (selectedParticle != MR_INVALID_INDEX) {
            const float aligned = alignmentQuantiles[
                selectedParticle * uniforms.program.x + variationIndex
            ];
            const float jitter =
                adaptive.counts.w == MR_WORLD_SAMPLING_REPLAY
                ? 0.0f
                : (uniform01(random.y) * 2.0f - 1.0f) *
                    adaptive.mixture.w;
            quantile = clamp(aligned + jitter, 0.0f, 1.0f);
        }
        const SampledValue value = sampleValueAtQuantile(
            descriptor,
            categoricalValues,
            uniforms.program.y,
            quantile
        );
        MRWorldScenarioValueGPU scenarioValue = {};
        scenarioValue.value = float4(
            value.scalar,
            value.quantile,
            componentWeight,
            0.0f
        );
        scenarioValue.identity = uint4(
            variationIndex,
            value.categorical,
            value.categoricalOrdinal,
            descriptor.binding.y ==
                MR_WORLD_DISTRIBUTION_CATEGORICAL
                ? MR_SCENARIO_VALUE_CATEGORICAL
                : 0u
        );
        scenarioValues[
            environment * uniforms.program.x + variationIndex
        ] = scenarioValue;
        const uint target = descriptor.binding.z;
        const uint targetIndex = descriptor.binding.w;
        if ((target <= MR_WORLD_TARGET_ASSET_RENDER_ALTERNATIVE ||
             target == MR_WORLD_TARGET_CLUTTER_SET ||
             target ==
                 MR_WORLD_TARGET_ASSET_COLLISION_ALTERNATIVE) &&
            targetIndex < uniforms.counts.y) {
            MRWorldAssetInstanceGPU asset =
                assets[firstAsset + targetIndex];
            applyAssetVariation(asset, target, value);
            assets[firstAsset + targetIndex] = asset;
        } else if (
            target >= MR_WORLD_TARGET_SENSOR_POSITION_X &&
            target <= MR_WORLD_TARGET_SENSOR_DEPTH_DROPOUT &&
            targetIndex < uniforms.counts.z
        ) {
            MRWorldSensorInstanceGPU sensor =
                sensors[firstSensor + targetIndex];
            applySensorVariation(sensor, target, value.scalar);
            sensors[firstSensor + targetIndex] = sensor;
        } else if (
            target >= MR_WORLD_TARGET_APPEARANCE_EXPOSURE &&
            target <= MR_WORLD_TARGET_APPEARANCE_LIGHT_INTENSITY &&
            targetIndex < uniforms.counts.w
        ) {
            MRWorldAppearanceInstanceGPU appearance =
                appearances[firstAppearance + targetIndex];
            applyAppearanceVariation(appearance, target, value.scalar);
            appearances[firstAppearance + targetIndex] = appearance;
        }
    }
}

// Resolves semantic WorldAsset bindings into the exact packed state consumed
// by MetalWorld. One thread owns one environment, so resets can be regenerated
// independently without atomics or host-side per-environment work.
kernel void mr_world_family_materialize_physics(
    const device float* baseQ [[buffer(0)]],
    const device float* baseV [[buffer(1)]],
    const device MRBodyStateGPU* baseSceneBodies [[buffer(2)]],
    const device uint* bodyToScene [[buffer(3)]],
    const device MRWorldAssetBindingGPU* bindings [[buffer(4)]],
    const device uint* bindingIndices [[buffer(5)]],
    const device MRWorldAssetInstanceGPU* assets [[buffer(6)]],
    constant MRWorldFamilyMaterializeUniformsGPU& uniforms [[buffer(7)]],
    device float* resetQ [[buffer(8)]],
    device float* resetV [[buffer(9)]],
    device MRBodyStateGPU* resetSceneBodies [[buffer(10)]],
    device MRWorldBodyParametersGPU* bodyParameters [[buffer(11)]],
    device MRWorldControllerParametersGPU*
        controllerParameters [[buffer(12)]],
    const device MRBodyPropertiesGPU*
        bodyProperties [[buffer(13)]],
    uint environment [[thread_position_in_grid]]
) {
    if (environment >= uniforms.stateCounts.x ||
        uniforms.identity.x != MR_WORLD_COMPILER_ABI_VERSION) {
        return;
    }

    const uint nq = uniforms.stateCounts.y;
    const uint nv = uniforms.stateCounts.z;
    const uint sceneBodyCount = uniforms.stateCounts.w;
    const uint bodyCount = uniforms.topology.x;
    const uint articulationCount = uniforms.topology.y;
    const uint assetCount = uniforms.topology.z;
    const uint assetBase = environment * assetCount;

    for (uint coordinate = 0u; coordinate < nq; ++coordinate) {
        resetQ[environment * nq + coordinate] = baseQ[coordinate];
    }
    for (uint coordinate = 0u; coordinate < nv; ++coordinate) {
        resetV[environment * nv + coordinate] = baseV[coordinate];
    }
    for (uint localScene = 0u;
         localScene < sceneBodyCount;
         ++localScene) {
        resetSceneBodies[
            environment * sceneBodyCount + localScene
        ] = baseSceneBodies[localScene];
    }
    for (uint body = 0u; body < bodyCount; ++body) {
        MRWorldBodyParametersGPU parameters = {};
        parameters.physical = float4(1.0f);
        parameters.identity = uint4(
            MR_INVALID_INDEX,
            MR_WORLD_DYNAMICS_STATIC,
            0u,
            0u
        );
        bodyParameters[environment * bodyCount + body] = parameters;
    }
    for (uint articulation = 0u;
         articulation < articulationCount;
         ++articulation) {
        MRWorldControllerParametersGPU parameters = {};
        parameters.controller = float4(1.0f, 1.0f, 0.0f, 1.0f);
        parameters.identity = uint4(
            MR_INVALID_INDEX,
            articulation,
            0u,
            0u
        );
        controllerParameters[
            environment * articulationCount + articulation
        ] = parameters;
    }

    for (uint assetIndex = 0u;
         assetIndex < assetCount;
         ++assetIndex) {
        const MRWorldAssetBindingGPU binding = bindings[assetIndex];
        const MRWorldAssetInstanceGPU asset =
            assets[assetBase + assetIndex];
        const uint firstBody = binding.geometryRanges.x;
        const uint boundBodyCount = binding.geometryRanges.y;
        for (uint localBody = 0u;
             localBody < boundBodyCount;
             ++localBody) {
            const uint body = bindingIndices[firstBody + localBody];
            if (body >= bodyCount) {
                continue;
            }
            MRWorldBodyParametersGPU parameters = {};
            parameters.physical = asset.physical;
            parameters.identity = uint4(
                assetIndex,
                binding.dynamics.x,
                0u,
                0u
            );
            bodyParameters[
                environment * bodyCount + body
            ] = parameters;

            const uint localScene = bodyToScene[body];
            if (localScene >= sceneBodyCount) {
                continue;
            }
            MRBodyStateGPU state =
                baseSceneBodies[localScene];
            state.position = float4(
                asset.positionAndScale.xyz,
                1.0f
            );
            state.orientation = normalize(asset.orientation);
            state.linearVelocityAndInverseMass =
                float4(
                    asset.linearVelocity.xyz,
                    bodyProperties[body].motionType ==
                            MR_MOTION_DYNAMIC &&
                        asset.physical.x > 0.0f
                    ? bodyProperties[body].massAndInverseMass.y /
                        asset.physical.x
                    : 0.0f
                );
            state.angularVelocity =
                float4(asset.angularVelocity.xyz, 0.0f);
            writeScaledWorldInverseInertia(
                state,
                bodyProperties[body],
                asset.physical.x
            );
            resetSceneBodies[
                environment * sceneBodyCount + localScene
            ] = state;
        }

        const uint articulation = binding.dynamics.y;
        if (articulation < articulationCount) {
            MRWorldControllerParametersGPU parameters = {};
            parameters.controller = asset.controller;
            parameters.identity = uint4(
                assetIndex,
                articulation,
                0u,
                0u
            );
            controllerParameters[
                environment * articulationCount + articulation
            ] = parameters;
        }
    }
}
