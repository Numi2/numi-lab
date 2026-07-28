#include <metal_stdlib>

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

struct SampledValue {
    float scalar;
    uint categorical;
};

SampledValue sampleValue(
    const MRWorldVariationGPU descriptor,
    const device uint* categoricalValues,
    const uint categoricalCount,
    const uint4 random
) {
    const float u0 = uniform01(random.x);
    const float u1 = uniform01(random.y);
    SampledValue result{0.0f, 0u};
    switch (descriptor.binding.y) {
    case MR_WORLD_DISTRIBUTION_CONSTANT:
        result.scalar = descriptor.parameters.x;
        return result;
    case MR_WORLD_DISTRIBUTION_UNIFORM:
        result.scalar = mix(
            descriptor.parameters.x,
            descriptor.parameters.y,
            u0
        );
        return result;
    case MR_WORLD_DISTRIBUTION_LOG_UNIFORM:
        result.scalar = exp(mix(
            log(descriptor.parameters.x),
            log(descriptor.parameters.y),
            u0
        ));
        return result;
    case MR_WORLD_DISTRIBUTION_NORMAL_CLAMPED: {
        const float radius = sqrt(-2.0f * log(max(u0, 1.0e-12f)));
        const float normal = radius * cos(2.0f * M_PI_F * u1);
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
        const uint selected = min(uint(u0 * float(count)), count - 1u);
        result.categorical = categoricalValues[first + selected];
        result.scalar = float(result.categorical);
        return result;
    }
    default:
        return result;
    }
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
    uint environment [[thread_position_in_grid]]
) {
    if (environment >= uniforms.counts.x ||
        uniforms.program.w != MR_WORLD_COMPILER_ABI_VERSION) {
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
    const ulong scenarioKey =
        familyFingerprint ^ seed ^
        (0x9e3779b97f4a7c15ul * (ulong(environment) + 1ul));

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
        const SampledValue value = sampleValue(
            descriptor,
            categoricalValues,
            uniforms.program.y,
            philox(environment, variationIndex, descriptor, uniforms)
        );
        const uint target = descriptor.binding.z;
        const uint targetIndex = descriptor.binding.w;
        if ((target <= MR_WORLD_TARGET_ASSET_RENDER_ALTERNATIVE ||
             target == MR_WORLD_TARGET_CLUTTER_SET) &&
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
