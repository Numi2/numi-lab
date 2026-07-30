#include <metal_stdlib>
#include <Metal/MTLAccelerationStructureTypes.h>

#include "metalrobo/engine_types.h"
#include "metalrobo/hybrid_renderer_types.h"
#include "metalrobo/world_compiler_types.h"
#include "VisualPBR.metal"

using namespace metal;
using namespace raytracing;

constant uint kLiveCurrent = 1u << 0u;
constant uint kLivePrevious = 1u << 1u;
constant uint kMaximumSceneTexturesV3 = 2048u;
constant uint kSceneSamplerCountV3 = 108u;

struct VisualResourceTableV3 {
    array<texture2d<float>, kMaximumSceneTexturesV3>
        textures [[id(0)]];
    array<sampler, kSceneSamplerCountV3>
        samplers [[id(kMaximumSceneTexturesV3)]];
    texturecube<float> diffuseIrradiance
        [[id(kMaximumSceneTexturesV3 + kSceneSamplerCountV3)]];
    texturecube<float> prefilteredSpecular
        [[id(kMaximumSceneTexturesV3 + kSceneSamplerCountV3 + 1u)]];
    texture2d<float> brdfLut
        [[id(kMaximumSceneTexturesV3 + kSceneSamplerCountV3 + 2u)]];
};

namespace {

struct BoundPose {
    float3 position;
    float4 orientation;
    float scale;
    bool valid;
};

struct CameraProjection {
    float2 pixel;
    float depth;
    bool valid;
};

float4 quaternionProduct(const float4 a, const float4 b) {
    return float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

float3 rotateVector(const float4 quaternion, const float3 vector) {
    const float4 vectorQuaternion(vector, 0.0f);
    const float4 conjugate(
        -quaternion.x,
        -quaternion.y,
        -quaternion.z,
        quaternion.w
    );
    return quaternionProduct(
        quaternionProduct(quaternion, vectorQuaternion),
        conjugate
    ).xyz;
}

float3 inverseRotateVector(
    const float4 quaternion,
    const float3 vector
) {
    return rotateVector(
        float4(
            -quaternion.x,
            -quaternion.y,
            -quaternion.z,
            quaternion.w
        ),
        vector
    );
}

float4 interpolateQuaternionFast(
    float4 first,
    float4 second,
    const float fraction
) {
    if (dot(first, second) < 0.0f) {
        second = -second;
    }
    return normalize(mix(first, second, fraction));
}

float2 projectAxis(
    const float3 mean,
    const float3 axis,
    const float fx,
    const float fy
) {
    const float inverseDepthSquared =
        1.0f / max(mean.z * mean.z, 1.0e-12f);
    return float2(
        fx * (axis.x * mean.z - mean.x * axis.z) *
            inverseDepthSquared,
        fy * (axis.y * mean.z - mean.y * axis.z) *
            inverseDepthSquared
    );
}

uint randomHash(uint value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

float uniformSigned(const uint value) {
    return (float(randomHash(value)) + 0.5f) *
               (2.0f / 4294967296.0f) -
        1.0f;
}

float unitVarianceNoise(const uint value) {
    // A bounded uniform sample scaled to unit variance, so configured
    // sensor sigma values retain their statistical meaning.
    return 1.7320508075688772f * uniformSigned(value);
}

float3 applyAppearance(
    float3 color,
    const MRWorldAppearanceInstanceGPU appearance
) {
    const float exposure = exp2(appearance.colorAndLight.x);
    const float whiteBalance =
        max(appearance.colorAndLight.y, 1.0e-3f);
    color *= float3(
        whiteBalance,
        1.0f,
        1.0f / whiteBalance
    );
    const float luminance =
        dot(color, float3(0.2126f, 0.7152f, 0.0722f));
    color = mix(
        float3(luminance),
        color,
        appearance.colorAndLight.z
    );
    color =
        (color - 0.5f) * appearance.material.y + 0.5f;
    return max(color * exposure, 0.0f);
}

BoundPose worldPose() {
    return {
        float3(0.0f),
        float4(0.0f, 0.0f, 0.0f, 1.0f),
        1.0f,
        true,
    };
}

BoundPose assetPose(
    const device MRWorldAssetInstanceGPU* assets,
    const MRWorldInstanceHeaderGPU instance,
    const uint asset
) {
    const MRWorldAssetInstanceGPU state =
        assets[instance.ranges.x + asset];
    return {
        state.positionAndScale.xyz,
        normalize(state.orientation),
        state.positionAndScale.w,
        true,
    };
}

BoundPose bodyPose(
    const device MRBodyStateGPU* bodies,
    const uint environment,
    const uint body,
    constant MRHybridRenderUniformsGPU& uniforms,
    const bool previous
) {
    const uint required =
        previous ? kLivePrevious : kLiveCurrent;
    if ((uniforms.live.y & required) == 0u ||
        body >= uniforms.live.x) {
        BoundPose result = worldPose();
        result.valid = false;
        return result;
    }
    float3 position;
    float4 orientation;
    if (uniforms.rayTiming.w > 0.5f &&
        uniforms.ray.y >= 2u) {
        const float time = clamp(
            previous
                ? uniforms.rayTiming.z
                : uniforms.exposure.x,
            0.0f,
            1.0f
        );
        const float scaled =
            time * float(uniforms.ray.y - 1u);
        const uint firstKeyframe =
            min(uint(floor(scaled)), uniforms.ray.y - 1u);
        const uint secondKeyframe =
            min(firstKeyframe + 1u, uniforms.ray.y - 1u);
        const float fraction =
            scaled - float(firstKeyframe);
        const uint statesPerKeyframe =
            uniforms.counts.x * uniforms.live.x;
        const uint bodyOffset =
            environment * uniforms.live.x + body;
        const MRBodyStateGPU first =
            bodies[
                firstKeyframe * statesPerKeyframe + bodyOffset
            ];
        const MRBodyStateGPU second =
            bodies[
                secondKeyframe * statesPerKeyframe + bodyOffset
            ];
        position = mix(
            first.position.xyz,
            second.position.xyz,
            fraction
        );
        orientation = interpolateQuaternionFast(
            first.orientation,
            second.orientation,
            fraction
        );
    } else {
        const MRBodyStateGPU state =
            bodies[environment * uniforms.live.x + body];
        position = state.position.xyz;
        orientation = normalize(state.orientation);
    }
    return {
        position,
        orientation,
        1.0f,
        all(isfinite(position)) &&
            all(isfinite(orientation)),
    };
}

BoundPose resolvePose(
    const uint bindingKind,
    const uint bindingIndex,
    const uint asset,
    const uint environment,
    const MRWorldInstanceHeaderGPU instance,
    const device MRWorldAssetInstanceGPU* assets,
    const device MRBodyStateGPU* bodies,
    constant MRHybridRenderUniformsGPU& uniforms,
    const bool previous
) {
    switch (bindingKind) {
    case MR_VISUAL_BINDING_WORLD:
        return worldPose();
    case MR_VISUAL_BINDING_ASSET:
        return assetPose(assets, instance, asset);
    case MR_VISUAL_BINDING_RIGID_BODY:
    case MR_VISUAL_BINDING_ARTICULATED_LINK:
        return bodyPose(
            bodies,
            environment,
            bindingIndex,
            uniforms,
            previous
        );
    default: {
        BoundPose result = worldPose();
        result.valid = false;
        return result;
    }
    }
}

BoundPose composeInstancePose(
    const BoundPose parent,
    const MRVisualInstanceGPUV2 instance
) {
    if (!parent.valid) {
        return parent;
    }
    BoundPose result;
    result.position = parent.position + rotateVector(
        parent.orientation,
        instance.translationAndScale.xyz * parent.scale
    );
    result.orientation = normalize(
        quaternionProduct(
            parent.orientation,
            instance.orientation
        )
    );
    result.scale =
        parent.scale * instance.translationAndScale.w;
    result.valid = all(isfinite(result.position)) &&
        all(isfinite(result.orientation)) &&
        isfinite(result.scale) && result.scale > 0.0f;
    return result;
}

BoundPose cameraPose(
    const uint environment,
    const MRWorldInstanceHeaderGPU instance,
    const MRWorldSensorInstanceGPU sensor,
    const uint cameraIndex,
    const device MRWorldAssetInstanceGPU* assets,
    const device MRVisualSensorBindingGPU* sensorBindings,
    const device MRBodyStateGPU* bodies,
    constant MRHybridRenderUniformsGPU& uniforms,
    const bool previous
) {
    BoundPose parent;
    if (cameraIndex < uniforms.live.z) {
        const MRVisualSensorBindingGPU binding =
            sensorBindings[cameraIndex];
        parent = resolvePose(
            binding.identity.x,
            binding.identity.y,
            binding.identity.z,
            environment,
            instance,
            assets,
            bodies,
            uniforms,
            previous
        );
    } else {
        parent = assetPose(
            assets,
            instance,
            sensor.identity.x
        );
    }
    if (!parent.valid) {
        return parent;
    }
    parent.position += rotateVector(
        parent.orientation,
        sensor.positionAndFocalScale.xyz * parent.scale
    );
    parent.orientation = normalize(
        quaternionProduct(
            parent.orientation,
            sensor.orientation
        )
    );
    parent.scale = 1.0f;
    return parent;
}

CameraProjection projectPoint(
    const float3 world,
    const BoundPose camera,
    const MRWorldSensorInstanceGPU sensor
) {
    const float3 cameraPoint = inverseRotateVector(
        camera.orientation,
        world - camera.position
    );
    CameraProjection result;
    result.depth = cameraPoint.z;
    result.valid =
        camera.valid && cameraPoint.z > 1.0e-4f &&
        all(isfinite(cameraPoint));
    if (!result.valid) {
        result.pixel = 0.0f;
        return result;
    }
    const float focalScale = sensor.positionAndFocalScale.w;
    const float2 focal =
        sensor.intrinsics.xy * focalScale;
    float2 normalized = cameraPoint.xy / cameraPoint.z;
    const float radiusSquared = dot(normalized, normalized);
    const float radial =
        1.0f + sensor.distortion.x * radiusSquared +
        sensor.distortion.y * radiusSquared * radiusSquared;
    normalized =
        normalized * radial +
        float2(
            2.0f * sensor.distortion.z *
                    normalized.x * normalized.y +
                sensor.distortion.w *
                    (radiusSquared +
                     2.0f * normalized.x * normalized.x),
            sensor.distortion.z *
                    (radiusSquared +
                     2.0f * normalized.y * normalized.y) +
                2.0f * sensor.distortion.w *
                    normalized.x * normalized.y
        );
    result.pixel =
        normalized * focal + sensor.intrinsics.zw;
    return result;
}

float edge(
    const float2 a,
    const float2 b,
    const float2 point
) {
    return (point.x - a.x) * (b.y - a.y) -
        (point.y - a.y) * (b.x - a.x);
}

uint frameSeed(
    const MRWorldInstanceHeaderGPU instance,
    constant MRHybridRenderUniformsGPU& uniforms,
    const uint pixel
) {
    return instance.identity.x ^ instance.identity.y ^
        uniforms.timing.x ^ uniforms.timing.y ^
        uniforms.timing.z * 0x85ebca6bu ^
        uniforms.render.x * 0xc2b2ae35u ^
        pixel * 0x9e3779b9u;
}

bool measureDepth(
    const float geometricDepth,
    const MRWorldSensorInstanceGPU sensor,
    const uint seed,
    constant MRHybridRenderUniformsGPU& uniforms,
    thread float& measuredDepth
) {
    const float minimumDepth =
        uniforms.sensorRangeAndResponse.x;
    const float maximumDepth =
        uniforms.sensorRangeAndResponse.y;
    if (!isfinite(geometricDepth) ||
        geometricDepth < minimumDepth ||
        geometricDepth > maximumDepth) {
        measuredDepth = uniforms.clearColorAndDepth.w;
        return false;
    }
    const float dropout =
        (float(randomHash(seed ^ 0x63d83595u)) + 0.5f) /
        4294967296.0f;
    if (dropout < sensor.noiseAndLatency.z) {
        measuredDepth = uniforms.clearColorAndDepth.w;
        return false;
    }
    measuredDepth = max(
        0.0f,
        geometricDepth +
            sensor.noiseAndLatency.y *
                unitVarianceNoise(seed ^ 0xa511e9b3u)
    );
    const float quantum =
        uniforms.sensorRangeAndResponse.z;
    if (quantum > 0.0f) {
        measuredDepth =
            round(measuredDepth / quantum) * quantum;
    }
    if (measuredDepth < minimumDepth ||
        measuredDepth > maximumDepth) {
        measuredDepth = uniforms.clearColorAndDepth.w;
        return false;
    }
    return true;
}

float3 transformPoint(
    const BoundPose pose,
    const float3 local
) {
    return pose.position +
        rotateVector(pose.orientation, local * pose.scale);
}

float3 transformNormal(
    const BoundPose pose,
    const float3 local
) {
    return normalize(rotateVector(pose.orientation, local));
}

float4 sampleVisualTexture(
    const device MRVisualTextureBindingGPUV2* bindings,
    constant VisualResourceTableV3& resources,
    const uint bindingIndex,
    const float4 texcoord01,
    const float lod,
    constant MRHybridRenderUniformsGPU& uniforms
) {
    if (bindingIndex == MR_INVALID_INDEX ||
        bindingIndex >= uniforms.presentation.x) {
        return 1.0f;
    }
    const MRVisualTextureBindingGPUV2 binding =
        bindings[bindingIndex];
    if (binding.resource.x >= kMaximumSceneTexturesV3 ||
        binding.resource.y >= kSceneSamplerCountV3) {
        return 1.0f;
    }
    const float2 authoredUv =
        binding.resource.z == 0u
        ? texcoord01.xy
        : texcoord01.zw;
    const float3 homogeneousUv = float3(authoredUv, 1.0f);
    const float2 transformedUv = float2(
        dot(binding.uvTransform0.xyz, homogeneousUv),
        dot(binding.uvTransform1.xyz, homogeneousUv)
    );
    return resources.textures[binding.resource.x].sample(
        resources.samplers[binding.resource.y],
        transformedUv,
        level(max(lod, 0.0f))
    );
}

float materialRoughness(
    const MRVisualMaterialGPUV2 material,
    const device MRVisualTextureBindingGPUV2* bindings,
    constant VisualResourceTableV3& resources,
    const float4 texcoord01,
    constant MRHybridRenderUniformsGPU& uniforms
) {
    const float textureValue =
        material.reserved.z != MR_INVALID_INDEX
        ? sampleVisualTexture(
              bindings,
              resources,
              material.reserved.z,
              texcoord01,
              0.0f,
              uniforms
          ).x
        : sampleVisualTexture(
              bindings,
              resources,
              material.textureIndices0.y,
              texcoord01,
              0.0f,
              uniforms
          ).y;
    return material.surface.x * textureValue;
}

float materialClearcoatRoughness(
    const MRVisualMaterialGPUV2 material,
    const device MRVisualTextureBindingGPUV2* bindings,
    constant VisualResourceTableV3& resources,
    const float4 texcoord01,
    constant MRHybridRenderUniformsGPU& uniforms
) {
    if (material.reserved.w != MR_INVALID_INDEX) {
        const float gloss = sampleVisualTexture(
            bindings,
            resources,
            material.reserved.w,
            texcoord01,
            0.0f,
            uniforms
        ).x;
        const float authoredGloss =
            1.0f - material.coatingAndAlphaCutoff.y;
        return 1.0f - authoredGloss * gloss;
    }
    return material.coatingAndAlphaCutoff.y *
        sampleVisualTexture(
            bindings,
            resources,
            material.textureIndices1.z,
            texcoord01,
            0.0f,
            uniforms
        ).y;
}

float3 rotateEnvironmentDirection(
    float3 direction,
    constant MRVisualEnvironmentGPUV2& environment
) {
    const float rotation = environment.parameters.y;
    const float sine = sin(rotation);
    const float cosine = cos(rotation);
    direction.xy = float2(
        cosine * direction.x - sine * direction.y,
        sine * direction.x + cosine * direction.y
    );
    return direction;
}

float3 evaluateEnvironmentIBL(
    constant VisualResourceTableV3& resources,
    constant MRVisualEnvironmentGPUV2& environment,
    const float3 normal,
    const float3 viewDirection,
    const float3 baseColor,
    const float metallic,
    const float perceptualRoughness,
    const float3 f0,
    const float ao
) {
    constexpr sampler environmentSampler(
        filter::linear,
        mip_filter::linear,
        address::clamp_to_edge
    );
    const float noV =
        max(dot(normal, viewDirection), 1.0e-4f);
    const float3 irradiance =
        resources.diffuseIrradiance.sample(
            environmentSampler,
            rotateEnvironmentDirection(normal, environment)
        ).xyz;
    const float3 reflection = reflect(-viewDirection, normal);
    const float mip = perceptualRoughness *
        float(max(environment.dimensions.x, 1u) - 1u);
    const float3 radiance =
        resources.prefilteredSpecular.sample(
            environmentSampler,
            rotateEnvironmentDirection(
                reflection,
                environment
            ),
            level(mip)
        ).xyz;
    constexpr sampler lutSampler(
        filter::linear,
        address::clamp_to_edge
    );
    const float2 dfg = resources.brdfLut.sample(
        lutSampler,
        float2(noV, perceptualRoughness)
    ).xy;
    const float3 specular =
        metalrobo_pbr::multiscatterSpecular(
            radiance,
            irradiance,
            f0,
            dfg
        );
    const float3 diffuseEnergy =
        metalrobo_pbr::multiscatterDiffuseEnergy(f0, dfg);
    const float3 diffuse =
        irradiance * baseColor * (1.0f - metallic) *
        diffuseEnergy * M_1_PI_F;
    return max(
        (diffuse + specular) *
            environment.parameters.x * ao,
        0.0f
    );
}

struct ShadowProjection {
    float2 pixel;
    float depth;
    bool valid;
};

void lightBasis(
    const float3 direction,
    thread float3& right,
    thread float3& up
) {
    const float3 reference =
        abs(direction.z) < 0.99f
        ? float3(0.0f, 0.0f, 1.0f)
        : float3(0.0f, 1.0f, 0.0f);
    right = normalize(cross(reference, direction));
    up = normalize(cross(direction, right));
}

ShadowProjection projectShadow(
    const float3 world,
    const MRVisualLightGPUV1 light,
    constant MRHybridRenderUniformsGPU& uniforms
) {
    ShadowProjection result{};
    const float3 direction = normalize(light.directionAndSpot.xyz);
    float3 right;
    float3 up;
    lightBasis(direction, right, up);
    const float3 relative = world - light.positionAndRange.xyz;
    const float range = max(light.positionAndRange.w, 1.0e-3f);
    if (light.identity.x == MR_VISUAL_LIGHT_DIRECTIONAL) {
        result.pixel = (
            float2(dot(relative, right), dot(relative, up)) /
                range +
            0.5f
        ) * float2(uniforms.shadow.xy);
        result.depth = dot(relative, direction) + 0.5f * range;
        result.valid = result.depth >= 0.0f &&
            result.depth <= range;
    } else {
        const float z = dot(relative, direction);
        const float outer = clamp(
            light.directionAndSpot.w,
            -0.999f,
            0.999f
        );
        const float tangent =
            max(sqrt(max(1.0f - outer * outer, 0.0f)) /
                    max(outer, 0.05f),
                0.05f);
        result.pixel = (
            float2(dot(relative, right), dot(relative, up)) /
                max(z * 2.0f * tangent, 1.0e-5f) +
            0.5f
        ) * float2(uniforms.shadow.xy);
        result.depth = z;
        result.valid = z > 1.0e-4f && z <= range;
    }
    result.valid = result.valid &&
        all(isfinite(result.pixel)) &&
        result.pixel.x >= 0.0f &&
        result.pixel.y >= 0.0f &&
        result.pixel.x < float(uniforms.shadow.x) &&
        result.pixel.y < float(uniforms.shadow.y);
    return result;
}

float shadowVisibility(
    const device uint* shadowAtlas,
    const float3 world,
    const float3 normal,
    const float3 lightDirection,
    const MRVisualLightGPUV1 light,
    const uint lightIndex,
    const uint environment,
    constant MRHybridRenderUniformsGPU& uniforms
) {
    if (light.shadow.x == 0u || uniforms.shadow.x == 0u ||
        lightIndex != uniforms.shadow.w ||
        environment < uniforms.shadowBatch.x ||
        environment >=
            uniforms.shadowBatch.x + uniforms.shadowBatch.y) {
        return 1.0f;
    }
    const ShadowProjection projection =
        projectShadow(world, light, uniforms);
    if (!projection.valid) {
        return 1.0f;
    }
    const int radius = int(min(uniforms.shadow.z, 3u));
    const uint localEnvironment =
        environment - uniforms.shadowBatch.x;
    const uint layerOffset =
        localEnvironment * uniforms.shadow.x * uniforms.shadow.y;
    const float bias =
        0.0008f + 0.003f *
            (1.0f - max(dot(normal, lightDirection), 0.0f));
    float visible = 0.0f;
    float samples = 0.0f;
    const int2 center = int2(floor(projection.pixel));
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            const int2 pixel = center + int2(x, y);
            if (pixel.x < 0 || pixel.y < 0 ||
                pixel.x >= int(uniforms.shadow.x) ||
                pixel.y >= int(uniforms.shadow.y)) {
                continue;
            }
            const float stored = as_type<float>(
                shadowAtlas[
                    layerOffset + uint(pixel.y) * uniforms.shadow.x +
                    uint(pixel.x)
                ]
            );
            visible += projection.depth - bias <= stored ? 1.0f : 0.0f;
            samples += 1.0f;
        }
    }
    return samples > 0.0f ? visible / samples : 1.0f;
}

bool pixelInBand(
    const uint x,
    const uint y,
    constant MRHybridRenderUniformsGPU& uniforms
) {
    const uint coordinate =
        uniforms.band.z == 0u ? y : x;
    return coordinate >= uniforms.band.x &&
        coordinate < uniforms.band.x + uniforms.band.y;
}

bool tileIntersectsBand(
    const uint tileX,
    const uint tileY,
    constant MRHybridRenderUniformsGPU& uniforms
) {
    const uint tileCoordinate =
        uniforms.band.z == 0u ? tileY : tileX;
    const uint firstPixel =
        tileCoordinate * MR_HYBRID_TILE_SIZE;
    const uint lastPixel =
        firstPixel + MR_HYBRID_TILE_SIZE;
    return lastPixel > uniforms.band.x &&
        firstPixel < uniforms.band.x + uniforms.band.y;
}

uint bandPixelCountPerEnvironment(
    constant MRHybridRenderUniformsGPU& uniforms
) {
    return uniforms.band.z == 0u
        ? uniforms.image.x * uniforms.band.y
        : uniforms.image.y * uniforms.band.y;
}

uint globalPixelFromBandIndex(
    const uint compactPixel,
    const uint environmentOffset,
    constant MRHybridRenderUniformsGPU& uniforms
) {
    const uint compactPerEnvironment =
        bandPixelCountPerEnvironment(uniforms);
    const uint environment =
        compactPixel / compactPerEnvironment +
        environmentOffset;
    const uint local =
        compactPixel % compactPerEnvironment;
    uint x = 0u;
    uint y = 0u;
    if (uniforms.band.z == 0u) {
        x = local % uniforms.image.x;
        y = uniforms.band.x + local / uniforms.image.x;
    } else {
        x = uniforms.band.x + local % uniforms.band.y;
        y = local / uniforms.band.y;
    }
    return environment * uniforms.image.x * uniforms.image.y +
        y * uniforms.image.x + x;
}

uint bandTileCountPerEnvironment(
    constant MRHybridRenderUniformsGPU& uniforms
) {
    const uint firstTile =
        uniforms.band.x / MR_HYBRID_TILE_SIZE;
    const uint lastTile =
        (
            uniforms.band.x + uniforms.band.y - 1u
        ) / MR_HYBRID_TILE_SIZE;
    const uint bandTileCount = lastTile - firstTile + 1u;
    return uniforms.band.z == 0u
        ? uniforms.image.z * bandTileCount
        : uniforms.image.w * bandTileCount;
}

uint globalTileFromBandIndex(
    const uint compactTile,
    constant MRHybridRenderUniformsGPU& uniforms
) {
    const uint compactPerEnvironment =
        bandTileCountPerEnvironment(uniforms);
    const uint environment =
        compactTile / compactPerEnvironment;
    const uint local = compactTile % compactPerEnvironment;
    const uint firstTile =
        uniforms.band.x / MR_HYBRID_TILE_SIZE;
    const uint bandTileCount =
        uniforms.band.z == 0u
        ? compactPerEnvironment / uniforms.image.z
        : compactPerEnvironment / uniforms.image.w;
    uint tileX = 0u;
    uint tileY = 0u;
    if (uniforms.band.z == 0u) {
        tileX = local % uniforms.image.z;
        tileY = firstTile + local / uniforms.image.z;
    } else {
        tileX = firstTile + local % bandTileCount;
        tileY = local / bandTileCount;
    }
    return environment * uniforms.image.z * uniforms.image.w +
        tileY * uniforms.image.z + tileX;
}

} // namespace

kernel void mr_hybrid_clear_tiles(
    device atomic_uint* tileCounts [[buffer(0)]],
    device atomic_uint* overflowCounts [[buffer(1)]],
    constant MRHybridRenderUniformsGPU& uniforms [[buffer(2)]],
    const uint index [[thread_position_in_grid]]
) {
    const uint compactTileCount =
        bandTileCountPerEnvironment(uniforms) *
        uniforms.counts.x;
    if (index < compactTileCount) {
        atomic_store_explicit(
            tileCounts + globalTileFromBandIndex(index, uniforms),
            0u,
            memory_order_relaxed
        );
    }
    if (index < uniforms.counts.x) {
        atomic_store_explicit(
            overflowCounts + index,
            0u,
            memory_order_relaxed
        );
    }
}

kernel void mr_visual_rebase_indices_v3(
    device uint* indices [[buffer(0)]],
    constant uint4& range [[buffer(1)]],
    const uint localIndex [[thread_position_in_grid]]
) {
    if (localIndex < range.y) {
        indices[range.x + localIndex] += range.z;
    }
}

kernel void mr_visual_rebase_material_bindings_v3(
    device MRVisualMaterialGPUV2* materials [[buffer(0)]],
    constant uint4& range [[buffer(1)]],
    const uint localIndex [[thread_position_in_grid]]
) {
    if (localIndex >= range.y) {
        return;
    }
    MRVisualMaterialGPUV2 material =
        materials[range.x + localIndex];
    if (material.textureIndices0.x != MR_INVALID_INDEX) {
        material.textureIndices0.x += range.z;
    }
    if (material.textureIndices0.y != MR_INVALID_INDEX) {
        material.textureIndices0.y += range.z;
    }
    if (material.textureIndices0.z != MR_INVALID_INDEX) {
        material.textureIndices0.z += range.z;
    }
    if (material.textureIndices0.w != MR_INVALID_INDEX) {
        material.textureIndices0.w += range.z;
    }
    if (material.textureIndices1.x != MR_INVALID_INDEX) {
        material.textureIndices1.x += range.z;
    }
    if (material.textureIndices1.y != MR_INVALID_INDEX) {
        material.textureIndices1.y += range.z;
    }
    if (material.textureIndices1.z != MR_INVALID_INDEX) {
        material.textureIndices1.z += range.z;
    }
    if (material.reserved.x != MR_INVALID_INDEX) {
        material.reserved.x += range.z;
    }
    if (material.reserved.y != MR_INVALID_INDEX) {
        material.reserved.y += range.z;
    }
    if (material.reserved.z != MR_INVALID_INDEX) {
        material.reserved.z += range.z;
    }
    if (material.reserved.w != MR_INVALID_INDEX) {
        material.reserved.w += range.z;
    }
    materials[range.x + localIndex] = material;
}

kernel void mr_visual_expand_triangles_v3(
    const device uint* indices [[buffer(0)]],
    const device MRVisualPrimitiveGPUV2* primitives [[buffer(1)]],
    device MRVisualTriangleGPUV2* triangles [[buffer(2)]],
    constant uint& primitiveCount [[buffer(3)]],
    const uint primitiveIndex [[thread_position_in_grid]]
) {
    if (primitiveIndex >= primitiveCount) {
        return;
    }
    const MRVisualPrimitiveGPUV2 primitive =
        primitives[primitiveIndex];
    const uint triangleBase = primitive.geometry.x / 3u;
    const uint triangleCount = primitive.geometry.y / 3u;
    for (uint triangle = 0u;
         triangle < triangleCount;
         ++triangle) {
        const uint first = primitive.geometry.x + triangle * 3u;
        MRVisualTriangleGPUV2 result;
        result.verticesAndPrimitive = uint4(
            indices[first],
            indices[first + 1u],
            indices[first + 2u],
            primitiveIndex
        );
        triangles[triangleBase + triangle] = result;
    }
}

kernel void mr_hybrid_clear_observations(
    const device MRWorldInstanceHeaderGPU* instances [[buffer(0)]],
    const device MRWorldAppearanceInstanceGPU* appearances [[buffer(1)]],
    device float4* rgb [[buffer(2)]],
    device float* depth [[buffer(3)]],
    device uint* segmentation [[buffer(4)]],
    device uint4* identities [[buffer(5)]],
    device float4* normals [[buffer(6)]],
    device float4* motion [[buffer(7)]],
    device uint* validity [[buffer(8)]],
    constant MRHybridRenderUniformsGPU& uniforms [[buffer(9)]],
    const uint compactPixel [[thread_position_in_grid]]
) {
    const uint compactCount =
        uniforms.counts.x *
        bandPixelCountPerEnvironment(uniforms);
    if (compactPixel >= compactCount) {
        return;
    }
    const uint pixel =
        globalPixelFromBandIndex(compactPixel, 0u, uniforms);
    const uint pixelsPerEnvironment =
        uniforms.image.x * uniforms.image.y;
    const uint environment = pixel / pixelsPerEnvironment;
    const MRWorldInstanceHeaderGPU instance =
        instances[environment];
    const MRWorldAppearanceInstanceGPU appearance =
        appearances[instance.program.x];
    rgb[pixel] = float4(
        applyAppearance(
            uniforms.clearColorAndDepth.xyz,
            appearance
        ),
        0.0f
    );
    depth[pixel] = uniforms.clearColorAndDepth.w;
    segmentation[pixel] = MR_INVALID_INDEX;
    identities[pixel] = uint4(MR_INVALID_INDEX);
    normals[pixel] = 0.0f;
    motion[pixel] = 0.0f;
    validity[pixel] = MR_VISUAL_VALIDITY_FRAME;
}

kernel void mr_hybrid_clear_mesh_winners(
    device atomic_uint* winners [[buffer(0)]],
    constant MRHybridRenderUniformsGPU& uniforms [[buffer(1)]],
    const uint pixel [[thread_position_in_grid]]
) {
    const uint compactCount =
        uniforms.counts.x * uniforms.image.x * uniforms.image.y;
    const uint bandCount =
        uniforms.counts.x *
        bandPixelCountPerEnvironment(uniforms);
    if (pixel >= bandCount) {
        return;
    }
    const uint globalPixel =
        globalPixelFromBandIndex(pixel, 0u, uniforms);
    atomic_store_explicit(
        winners + globalPixel,
        0x7f800000u,
        memory_order_relaxed
    );
    atomic_store_explicit(
        winners + compactCount + globalPixel,
        0xffffffffu,
        memory_order_relaxed
    );
}

kernel void mr_hybrid_clear_shadow_atlas(
    device atomic_uint* shadowAtlas [[buffer(0)]],
    constant MRHybridRenderUniformsGPU& uniforms [[buffer(1)]],
    const uint index [[thread_position_in_grid]]
) {
    const uint count =
        uniforms.shadowBatch.y *
        uniforms.shadow.x * uniforms.shadow.y;
    if (index < count) {
        atomic_store_explicit(
            shadowAtlas + index,
            0x7f800000u,
            memory_order_relaxed
        );
    }
}

kernel void mr_hybrid_rasterize_shadow_atlas(
    const device MRVisualVertexGPUV2* vertices [[buffer(0)]],
    const device MRVisualTriangleGPUV2* triangles [[buffer(1)]],
    const device MRVisualPrimitiveGPUV2* primitives [[buffer(2)]],
    const device MRVisualInstanceGPUV2* visualInstances [[buffer(3)]],
    const device MRWorldInstanceHeaderGPU* instances [[buffer(4)]],
    const device MRWorldAssetInstanceGPU* assets [[buffer(5)]],
    const device MRBodyStateGPU* currentBodies [[buffer(6)]],
    const device MRVisualLightGPUV1* lights [[buffer(7)]],
    device atomic_uint* shadowAtlas [[buffer(8)]],
    constant MRHybridRenderUniformsGPU& uniforms [[buffer(9)]],
    const uint index [[thread_position_in_grid]]
) {
    const uint triangleCount = uniforms.live.w;
    const uint total =
        uniforms.shadowBatch.y * triangleCount;
    if (index >= total || triangleCount == 0u ||
        uniforms.presentation.y == 0u ||
        uniforms.shadow.w >= uniforms.presentation.y) {
        return;
    }
    const uint localEnvironment = index / triangleCount;
    const uint environment =
        uniforms.shadowBatch.x + localEnvironment;
    if (environment >= uniforms.counts.x) {
        return;
    }
    const uint triangleIndex =
        index - environment * triangleCount;
    const MRVisualTriangleGPUV2 triangle =
        triangles[triangleIndex];
    const MRVisualPrimitiveGPUV2 primitive =
        primitives[triangle.verticesAndPrimitive.w];
    const MRVisualInstanceGPUV2 visualInstance =
        visualInstances[primitive.geometry.w];
    if ((visualInstance.binding.w &
         MR_VISUAL_INSTANCE_CASTS_SHADOW) == 0u) {
        return;
    }
    const MRWorldInstanceHeaderGPU instance =
        instances[environment];
    const BoundPose binding = composeInstancePose(resolvePose(
        visualInstance.binding.z,
        visualInstance.binding.y,
        visualInstance.binding.x,
        environment,
        instance,
        assets,
        currentBodies,
        uniforms,
        false
    ), visualInstance);
    if (!binding.valid) {
        return;
    }
    const uint3 indices = triangle.verticesAndPrimitive.xyz;
    const MRVisualLightGPUV1 light = lights[uniforms.shadow.w];
    const ShadowProjection p0 = projectShadow(
        transformPoint(binding, vertices[indices.x].position.xyz),
        light,
        uniforms
    );
    const ShadowProjection p1 = projectShadow(
        transformPoint(binding, vertices[indices.y].position.xyz),
        light,
        uniforms
    );
    const ShadowProjection p2 = projectShadow(
        transformPoint(binding, vertices[indices.z].position.xyz),
        light,
        uniforms
    );
    if (!p0.valid || !p1.valid || !p2.valid) {
        return;
    }
    const float area = edge(p0.pixel, p1.pixel, p2.pixel);
    if (abs(area) <= 1.0e-8f || !isfinite(area)) {
        return;
    }
    int minimumX = max(
        0,
        int(floor(min(p0.pixel.x, min(p1.pixel.x, p2.pixel.x))))
    );
    int maximumX = min(
        int(uniforms.shadow.x) - 1,
        int(ceil(max(p0.pixel.x, max(p1.pixel.x, p2.pixel.x))))
    );
    int minimumY = max(
        0,
        int(floor(min(p0.pixel.y, min(p1.pixel.y, p2.pixel.y))))
    );
    int maximumY = min(
        int(uniforms.shadow.y) - 1,
        int(ceil(max(p0.pixel.y, max(p1.pixel.y, p2.pixel.y))))
    );
    const uint layerOffset =
        localEnvironment * uniforms.shadow.x * uniforms.shadow.y;
    for (int y = minimumY; y <= maximumY; ++y) {
        for (int x = minimumX; x <= maximumX; ++x) {
            const float2 pixel = float2(x, y) + 0.5f;
            const float w0 =
                edge(p1.pixel, p2.pixel, pixel) / area;
            const float w1 =
                edge(p2.pixel, p0.pixel, pixel) / area;
            const float w2 = 1.0f - w0 - w1;
            if (min(w0, min(w1, w2)) < -1.0e-5f) {
                continue;
            }
            const float depth =
                w0 * p0.depth + w1 * p1.depth + w2 * p2.depth;
            if (depth >= 0.0f && isfinite(depth)) {
                atomic_fetch_min_explicit(
                    shadowAtlas +
                        layerOffset + uint(y) * uniforms.shadow.x +
                        uint(x),
                    as_type<uint>(depth),
                    memory_order_relaxed
                );
            }
        }
    }
}

kernel void mr_hybrid_clear_temporal_accumulation(
    device half4* accumulation [[buffer(0)]],
    constant MRHybridRenderUniformsGPU& uniforms [[buffer(1)]],
    const uint pixel [[thread_position_in_grid]]
) {
    const uint count =
        uniforms.counts.x *
        bandPixelCountPerEnvironment(uniforms);
    if (pixel < count) {
        accumulation[
            globalPixelFromBandIndex(pixel, 0u, uniforms)
        ] = 0.0f;
    }
}

kernel void mr_hybrid_accumulate_temporal_sample(
    const device float4* rgb [[buffer(0)]],
    device half4* accumulation [[buffer(1)]],
    constant MRHybridRenderUniformsGPU& uniforms [[buffer(2)]],
    const uint pixel [[thread_position_in_grid]]
) {
    const uint count =
        uniforms.counts.x *
        bandPixelCountPerEnvironment(uniforms);
    if (pixel < count) {
        const uint globalPixel =
            globalPixelFromBandIndex(pixel, 0u, uniforms);
        accumulation[globalPixel] = half4(
            float4(accumulation[globalPixel]) +
            rgb[globalPixel]
        );
    }
}

kernel void mr_hybrid_resolve_temporal_accumulation(
    const device half4* accumulation [[buffer(0)]],
    device float4* rgb [[buffer(1)]],
    constant MRHybridRenderUniformsGPU& uniforms [[buffer(2)]],
    const uint pixel [[thread_position_in_grid]]
) {
    const uint count =
        uniforms.counts.x *
        bandPixelCountPerEnvironment(uniforms);
    if (pixel < count) {
        const uint globalPixel =
            globalPixelFromBandIndex(pixel, 0u, uniforms);
        rgb[globalPixel] =
            float4(accumulation[globalPixel]) /
            float(max(uniforms.shutter.w, 1u));
    }
}

kernel void mr_hybrid_bin_gaussians(
    const device MRHybridGaussianGPU* gaussians [[buffer(0)]],
    const device MRWorldInstanceHeaderGPU* instances [[buffer(1)]],
    const device MRWorldAssetInstanceGPU* assets [[buffer(2)]],
    const device MRWorldSensorInstanceGPU* sensors [[buffer(3)]],
    const device MRVisualSensorBindingGPU* sensorBindings [[buffer(4)]],
    const device MRBodyStateGPU* currentBodies [[buffer(5)]],
    const device MRBodyStateGPU* previousBodies [[buffer(6)]],
    device MRHybridProjectedGaussianGPU* projected [[buffer(7)]],
    device atomic_uint* tileCounts [[buffer(8)]],
    device uint* tileIndices [[buffer(9)]],
    device atomic_uint* overflowCounts [[buffer(10)]],
    constant MRHybridRenderUniformsGPU& uniforms [[buffer(11)]],
    const uint index [[thread_position_in_grid]]
) {
    const uint environmentCount = uniforms.counts.x;
    const uint gaussianCount = uniforms.counts.y;
    if (index >= environmentCount * gaussianCount) {
        return;
    }
    const uint environment = index / gaussianCount;
    const uint gaussianIndex = index - environment * gaussianCount;
    const MRHybridGaussianGPU gaussian = gaussians[gaussianIndex];
    const MRWorldInstanceHeaderGPU instance = instances[environment];
    const uint cameraIndex = uniforms.render.x;
    const MRWorldSensorInstanceGPU sensor =
        sensors[instance.ranges.z + cameraIndex];
    const BoundPose camera = cameraPose(
        environment,
        instance,
        sensor,
        cameraIndex,
        assets,
        sensorBindings,
        currentBodies,
        uniforms,
        false
    );
    const BoundPose previousCamera = cameraPose(
        environment,
        instance,
        sensor,
        cameraIndex,
        assets,
        sensorBindings,
        previousBodies,
        uniforms,
        true
    );

    BoundPose binding = worldPose();
    BoundPose previousBinding = binding;
    if (gaussian.binding.w == MR_HYBRID_GAUSSIAN_ASSET_LOCAL) {
        binding = assetPose(assets, instance, gaussian.binding.x);
        previousBinding = binding;
    } else if (
        gaussian.binding.w == MR_HYBRID_GAUSSIAN_BODY_LOCAL
    ) {
        const uint kind = MR_VISUAL_BINDING_RIGID_BODY;
        binding = resolvePose(
            kind,
            gaussian.binding.y,
            gaussian.binding.x,
            environment,
            instance,
            assets,
            currentBodies,
            uniforms,
            false
        );
        previousBinding = resolvePose(
            kind,
            gaussian.binding.y,
            gaussian.binding.x,
            environment,
            instance,
            assets,
            previousBodies,
            uniforms,
            true
        );
    }

    const float3 worldMean = transformPoint(
        binding,
        gaussian.meanAndOpacity.xyz
    );
    const float4 worldOrientation = normalize(
        quaternionProduct(
            binding.orientation,
            gaussian.orientation
        )
    );
    const CameraProjection projection =
        projectPoint(worldMean, camera, sensor);

    MRHybridProjectedGaussianGPU result{};
    result.centerDepthRadius =
        float4(0.0f, 0.0f, projection.depth, -1.0f);
    result.colorAndOpacity = float4(
        gaussian.colorAndEmission.xyz,
        clamp(gaussian.meanAndOpacity.w, 0.0f, 0.999f)
    );
    result.identity = uint4(
        gaussian.binding.z,
        gaussian.binding.x + 1u,
        gaussian.binding.w == MR_HYBRID_GAUSSIAN_BODY_LOCAL
            ? gaussian.binding.y
            : MR_INVALID_INDEX,
        gaussianIndex
    );
    if (!projection.valid || !binding.valid) {
        projected[index] = result;
        return;
    }

    const float3 cameraMean = inverseRotateVector(
        camera.orientation,
        worldMean - camera.position
    );
    const float focalScale = sensor.positionAndFocalScale.w;
    const float fx = sensor.intrinsics.x * focalScale;
    const float fy = sensor.intrinsics.y * focalScale;
    const float3 worldAxisX = rotateVector(
        worldOrientation,
        float3(
            gaussian.scaleAndImportance.x * binding.scale,
            0.0f,
            0.0f
        )
    );
    const float3 worldAxisY = rotateVector(
        worldOrientation,
        float3(
            0.0f,
            gaussian.scaleAndImportance.y * binding.scale,
            0.0f
        )
    );
    const float3 worldAxisZ = rotateVector(
        worldOrientation,
        float3(
            0.0f,
            0.0f,
            gaussian.scaleAndImportance.z * binding.scale
        )
    );
    const float2 derivativeX = projectAxis(
        cameraMean,
        inverseRotateVector(camera.orientation, worldAxisX),
        fx,
        fy
    );
    const float2 derivativeY = projectAxis(
        cameraMean,
        inverseRotateVector(camera.orientation, worldAxisY),
        fx,
        fy
    );
    const float2 derivativeZ = projectAxis(
        cameraMean,
        inverseRotateVector(camera.orientation, worldAxisZ),
        fx,
        fy
    );
    const float covarianceXX =
        derivativeX.x * derivativeX.x +
        derivativeY.x * derivativeY.x +
        derivativeZ.x * derivativeZ.x + 0.25f;
    const float covarianceXY =
        derivativeX.x * derivativeX.y +
        derivativeY.x * derivativeY.y +
        derivativeZ.x * derivativeZ.y;
    const float covarianceYY =
        derivativeX.y * derivativeX.y +
        derivativeY.y * derivativeY.y +
        derivativeZ.y * derivativeZ.y + 0.25f;
    const float determinant = max(
        covarianceXX * covarianceYY -
            covarianceXY * covarianceXY,
        1.0e-12f
    );
    const float inverseXX = covarianceYY / determinant;
    const float inverseXY = -covarianceXY / determinant;
    const float inverseYY = covarianceXX / determinant;
    const float trace = covarianceXX + covarianceYY;
    const float eigenDiscriminant = sqrt(
        max(0.0f, trace * trace - 4.0f * determinant)
    );
    const float largestEigenvalue =
        0.5f * (trace + eigenDiscriminant);
    float pixelRadius = max(
        0.5f,
        3.0f * sqrt(max(largestEigenvalue, 0.0f))
    );
    result.centerDepthRadius = float4(
        projection.pixel,
        cameraMean.z,
        pixelRadius
    );
    result.conicAndBounds = float4(
        inverseXX,
        inverseXY,
        inverseYY,
        pixelRadius
    );
    result.normalAndValidity = float4(
        normalize(inverseRotateVector(
            camera.orientation,
            worldAxisZ
        )),
        1.0f
    );

    float2 motion = 0.0f;
    if ((uniforms.live.y & kLivePrevious) != 0u &&
        previousBinding.valid && previousCamera.valid) {
        const float3 previousWorld = transformPoint(
            previousBinding,
            gaussian.meanAndOpacity.xyz
        );
        const CameraProjection previousProjection =
            projectPoint(previousWorld, previousCamera, sensor);
        if (previousProjection.valid) {
            motion =
                projection.pixel - previousProjection.pixel;
        }
    }
    const float blurScale =
        uniforms.sensorRangeAndResponse.w;
    if (blurScale > 0.0f) {
        const float2 blurMotion = motion * blurScale;
        const float blurredRadius =
            pixelRadius + 0.5f * length(blurMotion);
        if (blurredRadius > pixelRadius) {
            const float conicScale =
                (pixelRadius * pixelRadius) /
                (blurredRadius * blurredRadius);
            result.centerDepthRadius.xy -=
                0.5f * blurMotion;
            result.centerDepthRadius.w = blurredRadius;
            result.conicAndBounds.xyz *= conicScale;
            result.conicAndBounds.w = blurredRadius;
            pixelRadius = blurredRadius;
        }
    }
    result.motionAndVisibility = float4(motion, 1.0f, 0.0f);
    projected[index] = result;

    int minimumX = max(
        0,
        int(floor(
            (result.centerDepthRadius.x - pixelRadius) /
            float(MR_HYBRID_TILE_SIZE)
        ))
    );
    int maximumX = min(
        int(uniforms.image.z) - 1,
        int(floor(
            (result.centerDepthRadius.x + pixelRadius) /
            float(MR_HYBRID_TILE_SIZE)
        ))
    );
    int minimumY = max(
        0,
        int(floor(
            (result.centerDepthRadius.y - pixelRadius) /
            float(MR_HYBRID_TILE_SIZE)
        ))
    );
    int maximumY = min(
        int(uniforms.image.w) - 1,
        int(floor(
            (result.centerDepthRadius.y + pixelRadius) /
            float(MR_HYBRID_TILE_SIZE)
        ))
    );
    const int firstBandTile =
        int(uniforms.band.x / MR_HYBRID_TILE_SIZE);
    const int lastBandTile = int(
        (
            uniforms.band.x + uniforms.band.y - 1u
        ) / MR_HYBRID_TILE_SIZE
    );
    if (uniforms.band.z == 0u) {
        minimumY = max(minimumY, firstBandTile);
        maximumY = min(maximumY, lastBandTile);
    } else {
        minimumX = max(minimumX, firstBandTile);
        maximumX = min(maximumX, lastBandTile);
    }
    if (minimumX > maximumX || minimumY > maximumY) {
        return;
    }

    const uint tilesPerEnvironment =
        uniforms.image.z * uniforms.image.w;
    const uint maximumPerTile = uniforms.render.y;
    for (int tileY = minimumY;
         tileY <= maximumY;
         ++tileY) {
        for (int tileX = minimumX;
             tileX <= maximumX;
             ++tileX) {
            const uint tile =
                environment * tilesPerEnvironment +
                uint(tileY) * uniforms.image.z + uint(tileX);
            const uint slot = atomic_fetch_add_explicit(
                tileCounts + tile,
                1u,
                memory_order_relaxed
            );
            if (slot < maximumPerTile) {
                tileIndices[
                    tile * maximumPerTile + slot
                ] = index;
            } else {
                atomic_fetch_add_explicit(
                    overflowCounts + environment,
                    1u,
                    memory_order_relaxed
                );
            }
        }
    }
}

kernel void mr_hybrid_render_tiles(
    const device MRHybridProjectedGaussianGPU* projected [[buffer(0)]],
    const device atomic_uint* tileCounts [[buffer(1)]],
    const device uint* tileIndices [[buffer(2)]],
    const device MRWorldInstanceHeaderGPU* instances [[buffer(3)]],
    const device MRWorldSensorInstanceGPU* sensors [[buffer(4)]],
    const device MRWorldAppearanceInstanceGPU* appearances [[buffer(5)]],
    device float4* rgb [[buffer(6)]],
    device float* depth [[buffer(7)]],
    device uint* segmentation [[buffer(8)]],
    device uint4* identities [[buffer(9)]],
    device float4* normals [[buffer(10)]],
    device float4* motion [[buffer(11)]],
    device uint* validity [[buffer(12)]],
    constant MRHybridRenderUniformsGPU& uniforms [[buffer(13)]],
    const uint compactTile [[threadgroup_position_in_grid]],
    const uint localIndex [[thread_position_in_threadgroup]]
) {
    threadgroup uint
        sortedIndices[MR_HYBRID_MAX_GAUSSIANS_PER_TILE];
    threadgroup float
        sortedDepth[MR_HYBRID_MAX_GAUSSIANS_PER_TILE];

    const uint tile =
        globalTileFromBandIndex(compactTile, uniforms);
    const uint tilesPerEnvironment =
        uniforms.image.z * uniforms.image.w;
    const uint environment = tile / tilesPerEnvironment;
    if (environment >= uniforms.counts.x) {
        return;
    }
    const uint tileInEnvironment =
        tile - environment * tilesPerEnvironment;
    const uint tileX = tileInEnvironment % uniforms.image.z;
    const uint tileY = tileInEnvironment / uniforms.image.z;
    if (!tileIntersectsBand(tileX, tileY, uniforms)) {
        return;
    }
    const uint count = min(
        atomic_load_explicit(
            tileCounts + tile,
            memory_order_relaxed
        ),
        uniforms.render.y
    );
    const uint pixelX =
        tileX * MR_HYBRID_TILE_SIZE +
        localIndex % MR_HYBRID_TILE_SIZE;
    const uint pixelY =
        tileY * MR_HYBRID_TILE_SIZE +
        localIndex / MR_HYBRID_TILE_SIZE;
    if (count == 0u) {
        if (pixelX < uniforms.image.x &&
            pixelY < uniforms.image.y &&
            pixelInBand(pixelX, pixelY, uniforms)) {
            const MRWorldInstanceHeaderGPU instance =
                instances[environment];
            const MRWorldAppearanceInstanceGPU appearance =
                appearances[instance.program.x];
            const uint pixel =
                environment * uniforms.image.x * uniforms.image.y +
                pixelY * uniforms.image.x + pixelX;
            rgb[pixel] = float4(
                applyAppearance(
                    uniforms.clearColorAndDepth.xyz,
                    appearance
                ),
                0.0f
            );
            depth[pixel] = uniforms.clearColorAndDepth.w;
            segmentation[pixel] = MR_INVALID_INDEX;
            identities[pixel] = uint4(MR_INVALID_INDEX);
            normals[pixel] = 0.0f;
            motion[pixel] = 0.0f;
            validity[pixel] = MR_VISUAL_VALIDITY_FRAME;
        }
        return;
    }
    const uint sourceIndex =
        tile * uniforms.render.y + localIndex;
    if (localIndex < count) {
        sortedIndices[localIndex] = tileIndices[sourceIndex];
        sortedDepth[localIndex] =
            projected[
                sortedIndices[localIndex]
            ].centerDepthRadius.z;
    } else {
        sortedIndices[localIndex] = 0xffffffffu;
        sortedDepth[localIndex] = INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sortCapacity = 1u;
    while (sortCapacity < count) {
        sortCapacity <<= 1u;
    }
    for (uint width = 2u;
         width <= sortCapacity;
         width <<= 1u) {
        for (uint stride = width >> 1u;
             stride > 0u;
             stride >>= 1u) {
            const uint partner = localIndex ^ stride;
            if (localIndex < sortCapacity &&
                partner > localIndex) {
                const bool ascending =
                    (localIndex & width) == 0u;
                const bool swapValues =
                    ascending
                    ? sortedDepth[localIndex] >
                          sortedDepth[partner]
                    : sortedDepth[localIndex] <
                          sortedDepth[partner];
                if (swapValues) {
                    const float savedDepth =
                        sortedDepth[localIndex];
                    sortedDepth[localIndex] =
                        sortedDepth[partner];
                    sortedDepth[partner] = savedDepth;
                    const uint savedIndex =
                        sortedIndices[localIndex];
                    sortedIndices[localIndex] =
                        sortedIndices[partner];
                    sortedIndices[partner] = savedIndex;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (pixelX >= uniforms.image.x ||
        pixelY >= uniforms.image.y ||
        !pixelInBand(pixelX, pixelY, uniforms)) {
        return;
    }

    float3 accumulatedColor = 0.0f;
    float transmittance = 1.0f;
    float nearestDepth = uniforms.clearColorAndDepth.w;
    uint4 identity =
        uint4(MR_INVALID_INDEX);
    float4 normal = 0.0f;
    float4 pixelMotion = 0.0f;
    for (uint slot = 0u; slot < count; ++slot) {
        const uint projectedIndex = sortedIndices[slot];
        if (projectedIndex == 0xffffffffu) {
            continue;
        }
        const MRHybridProjectedGaussianGPU gaussian =
            projected[projectedIndex];
        const float2 delta =
            (float2(float(pixelX), float(pixelY)) + 0.5f) -
            gaussian.centerDepthRadius.xy;
        const float mahalanobis =
            gaussian.conicAndBounds.x * delta.x * delta.x +
            2.0f * gaussian.conicAndBounds.y *
                delta.x * delta.y +
            gaussian.conicAndBounds.z * delta.y * delta.y;
        if (mahalanobis > 9.0f) {
            continue;
        }
        const float alpha = min(
            0.999f,
            gaussian.colorAndOpacity.w *
                exp(-0.5f * mahalanobis)
        );
        if (alpha <= 1.0e-4f) {
            continue;
        }
        const float contribution = transmittance * alpha;
        accumulatedColor +=
            contribution * gaussian.colorAndOpacity.xyz;
        if (identity.x == MR_INVALID_INDEX &&
            contribution > 1.0e-3f) {
            identity = gaussian.identity;
            nearestDepth = gaussian.centerDepthRadius.z;
            normal = gaussian.normalAndValidity;
            pixelMotion = gaussian.motionAndVisibility;
        }
        transmittance *= 1.0f - alpha;
        if (transmittance <= 1.0e-3f) {
            break;
        }
    }
    accumulatedColor +=
        transmittance * uniforms.clearColorAndDepth.xyz;

    const MRWorldInstanceHeaderGPU instance =
        instances[environment];
    const MRWorldAppearanceInstanceGPU appearance =
        appearances[instance.program.x];
    accumulatedColor =
        applyAppearance(accumulatedColor, appearance);
    const uint pixel =
        environment * uniforms.image.x * uniforms.image.y +
        pixelY * uniforms.image.x + pixelX;
    uint valid = MR_VISUAL_VALIDITY_FRAME;
    if (nearestDepth < uniforms.clearColorAndDepth.w) {
        valid |= MR_VISUAL_VALIDITY_GEOMETRY;
    }
    rgb[pixel] = float4(
        accumulatedColor,
        1.0f - transmittance
    );
    depth[pixel] = nearestDepth;
    segmentation[pixel] = identity.x;
    identities[pixel] = identity;
    normals[pixel] = normal;
    motion[pixel] = pixelMotion;
    validity[pixel] = valid;
}

void rasterizeMesh(
    const device MRVisualVertexGPUV2* vertices,
    const device MRVisualTriangleGPUV2* triangles,
    const device MRVisualPrimitiveGPUV2* primitives,
    const device MRVisualInstanceGPUV2* visualInstances,
    const device MRWorldInstanceHeaderGPU* instances,
    const device MRWorldAssetInstanceGPU* assets,
    const device MRWorldSensorInstanceGPU* sensors,
    const device MRVisualSensorBindingGPU* sensorBindings,
    const device MRBodyStateGPU* currentBodies,
    device atomic_uint* winners,
    constant MRHybridRenderUniformsGPU& uniforms,
    const uint index,
    const bool selectTriangle
) {
    const uint triangleCount = uniforms.live.w;
    const uint total = uniforms.counts.x * triangleCount;
    if (index >= total || triangleCount == 0u) {
        return;
    }
    const uint environment = index / triangleCount;
    const uint triangleIndex =
        index - environment * triangleCount;
    const MRVisualTriangleGPUV2 triangle =
        triangles[triangleIndex];
    const MRVisualPrimitiveGPUV2 primitive =
        primitives[triangle.verticesAndPrimitive.w];
    const MRVisualInstanceGPUV2 visualInstance =
        visualInstances[primitive.geometry.w];
    if ((visualInstance.binding.w &
         MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR) == 0u) {
        return;
    }
    const MRWorldInstanceHeaderGPU instance =
        instances[environment];
    const MRWorldSensorInstanceGPU sensor =
        sensors[instance.ranges.z + uniforms.render.x];
    const BoundPose camera = cameraPose(
        environment,
        instance,
        sensor,
        uniforms.render.x,
        assets,
        sensorBindings,
        currentBodies,
        uniforms,
        false
    );
    const BoundPose binding = composeInstancePose(resolvePose(
        visualInstance.binding.z,
        visualInstance.binding.y,
        visualInstance.binding.x,
        environment,
        instance,
        assets,
        currentBodies,
        uniforms,
        false
    ), visualInstance);
    if (!camera.valid || !binding.valid) {
        return;
    }

    const uint3 indices = triangle.verticesAndPrimitive.xyz;
    const float3 world0 = transformPoint(
        binding,
        vertices[indices.x].position.xyz
    );
    const float3 world1 = transformPoint(
        binding,
        vertices[indices.y].position.xyz
    );
    const float3 world2 = transformPoint(
        binding,
        vertices[indices.z].position.xyz
    );
    const CameraProjection p0 =
        projectPoint(world0, camera, sensor);
    const CameraProjection p1 =
        projectPoint(world1, camera, sensor);
    const CameraProjection p2 =
        projectPoint(world2, camera, sensor);
    if (!p0.valid || !p1.valid || !p2.valid) {
        return;
    }
    const float area = edge(p0.pixel, p1.pixel, p2.pixel);
    if (abs(area) <= 1.0e-8f || !isfinite(area)) {
        return;
    }
    int minimumX = max(
        0,
        int(floor(min(
            p0.pixel.x,
            min(p1.pixel.x, p2.pixel.x)
        )))
    );
    int maximumX = min(
        int(uniforms.image.x) - 1,
        int(ceil(max(
            p0.pixel.x,
            max(p1.pixel.x, p2.pixel.x)
        )))
    );
    int minimumY = max(
        0,
        int(floor(min(
            p0.pixel.y,
            min(p1.pixel.y, p2.pixel.y)
        )))
    );
    int maximumY = min(
        int(uniforms.image.y) - 1,
        int(ceil(max(
            p0.pixel.y,
            max(p1.pixel.y, p2.pixel.y)
        )))
    );
    const int firstBandPixel = int(uniforms.band.x);
    const int lastBandPixel = int(
        uniforms.band.x + uniforms.band.y - 1u
    );
    if (uniforms.band.z == 0u) {
        minimumY = max(minimumY, firstBandPixel);
        maximumY = min(maximumY, lastBandPixel);
    } else {
        minimumX = max(minimumX, firstBandPixel);
        maximumX = min(maximumX, lastBandPixel);
    }
    if (minimumX > maximumX || minimumY > maximumY) {
        return;
    }
    for (int y = minimumY; y <= maximumY; ++y) {
        for (int x = minimumX; x <= maximumX; ++x) {
            const float2 pixel =
                float2(float(x), float(y)) + 0.5f;
            const float w0 =
                edge(p1.pixel, p2.pixel, pixel) / area;
            const float w1 =
                edge(p2.pixel, p0.pixel, pixel) / area;
            const float w2 = 1.0f - w0 - w1;
            if (min(w0, min(w1, w2)) < -1.0e-5f) {
                continue;
            }
            const float inverseDepth =
                w0 / p0.depth + w1 / p1.depth +
                w2 / p2.depth;
            if (!(inverseDepth > 0.0f) ||
                !isfinite(inverseDepth)) {
                continue;
            }
            const float depth = 1.0f / inverseDepth;
            const uint depthBits = as_type<uint>(depth);
            const uint flat =
                environment *
                    uniforms.image.x * uniforms.image.y +
                uint(y) * uniforms.image.x + uint(x);
            if (selectTriangle) {
                const uint pixelCount =
                    uniforms.counts.x *
                    uniforms.image.x *
                    uniforms.image.y;
                if (atomic_load_explicit(
                        winners + flat,
                        memory_order_relaxed
                    ) == depthBits) {
                    atomic_fetch_min_explicit(
                        winners + pixelCount + flat,
                        triangleIndex,
                        memory_order_relaxed
                    );
                }
            } else {
                atomic_fetch_min_explicit(
                    winners + flat,
                    depthBits,
                    memory_order_relaxed
                );
            }
        }
    }
}

kernel void mr_hybrid_rasterize_mesh(
    const device MRVisualVertexGPUV2* vertices [[buffer(0)]],
    const device MRVisualTriangleGPUV2* triangles [[buffer(1)]],
    const device MRVisualPrimitiveGPUV2* primitives [[buffer(2)]],
    const device MRVisualInstanceGPUV2* visualInstances [[buffer(3)]],
    const device MRWorldInstanceHeaderGPU* instances [[buffer(4)]],
    const device MRWorldAssetInstanceGPU* assets [[buffer(5)]],
    const device MRWorldSensorInstanceGPU* sensors [[buffer(6)]],
    const device MRVisualSensorBindingGPU* sensorBindings [[buffer(7)]],
    const device MRBodyStateGPU* currentBodies [[buffer(8)]],
    device atomic_uint* winners [[buffer(9)]],
    constant MRHybridRenderUniformsGPU& uniforms [[buffer(10)]],
    const uint index [[thread_position_in_grid]]
) {
    rasterizeMesh(
        vertices,
        triangles,
        primitives,
        visualInstances,
        instances,
        assets,
        sensors,
        sensorBindings,
        currentBodies,
        winners,
        uniforms,
        index,
        false
    );
}

kernel void mr_hybrid_select_mesh(
    const device MRVisualVertexGPUV2* vertices [[buffer(0)]],
    const device MRVisualTriangleGPUV2* triangles [[buffer(1)]],
    const device MRVisualPrimitiveGPUV2* primitives [[buffer(2)]],
    const device MRVisualInstanceGPUV2* visualInstances [[buffer(3)]],
    const device MRWorldInstanceHeaderGPU* instances [[buffer(4)]],
    const device MRWorldAssetInstanceGPU* assets [[buffer(5)]],
    const device MRWorldSensorInstanceGPU* sensors [[buffer(6)]],
    const device MRVisualSensorBindingGPU* sensorBindings [[buffer(7)]],
    const device MRBodyStateGPU* currentBodies [[buffer(8)]],
    device atomic_uint* winners [[buffer(9)]],
    constant MRHybridRenderUniformsGPU& uniforms [[buffer(10)]],
    const uint index [[thread_position_in_grid]]
) {
    rasterizeMesh(
        vertices,
        triangles,
        primitives,
        visualInstances,
        instances,
        assets,
        sensors,
        sensorBindings,
        currentBodies,
        winners,
        uniforms,
        index,
        true
    );
}

kernel void mr_hybrid_composite_mesh(
    const device MRVisualVertexGPUV2* vertices [[buffer(0)]],
    const device MRVisualTriangleGPUV2* triangles [[buffer(1)]],
    const device MRVisualPrimitiveGPUV2* primitives [[buffer(2)]],
    const device MRVisualInstanceGPUV2* visualInstances [[buffer(3)]],
    const device MRVisualMaterialGPUV2* materials [[buffer(4)]],
    const device MRWorldInstanceHeaderGPU* instances [[buffer(5)]],
    const device MRWorldAssetInstanceGPU* assets [[buffer(6)]],
    const device MRWorldSensorInstanceGPU* sensors [[buffer(7)]],
    const device MRWorldAppearanceInstanceGPU* appearances [[buffer(8)]],
    const device MRVisualSensorBindingGPU* sensorBindings [[buffer(9)]],
    const device MRBodyStateGPU* currentBodies [[buffer(10)]],
    const device MRBodyStateGPU* previousBodies [[buffer(11)]],
    const device atomic_uint* winners [[buffer(12)]],
    device float4* rgb [[buffer(13)]],
    device float* depth [[buffer(14)]],
    device uint* segmentation [[buffer(15)]],
    device uint4* identities [[buffer(16)]],
    device float4* normals [[buffer(17)]],
    device float4* motion [[buffer(18)]],
    device uint* validity [[buffer(19)]],
    const device MRVisualTextureBindingGPUV2* textureBindings [[buffer(20)]],
    constant VisualResourceTableV3& resources [[buffer(21)]],
    const device MRVisualLightGPUV1* lights [[buffer(22)]],
    constant MRVisualEnvironmentGPUV2& environmentLighting
        [[buffer(23)]],
    const device uint* shadowAtlas [[buffer(24)]],
    constant MRHybridRenderUniformsGPU& uniforms [[buffer(25)]],
    const uint batchPixel [[thread_position_in_grid]]
) {
    const uint pixelsPerEnvironment =
        uniforms.image.x * uniforms.image.y;
    const uint batchPixelCount =
        uniforms.shadowBatch.y *
        bandPixelCountPerEnvironment(uniforms);
    const uint pixelCount =
        uniforms.counts.x * pixelsPerEnvironment;
    if (batchPixel >= batchPixelCount) {
        return;
    }
    const uint pixel = globalPixelFromBandIndex(
        batchPixel,
        uniforms.shadowBatch.x,
        uniforms
    );
    if (pixel >= pixelCount) {
        return;
    }
    const uint meshDepthBits = atomic_load_explicit(
        winners + pixel,
        memory_order_relaxed
    );
    const uint triangleIndex = atomic_load_explicit(
        winners + pixelCount + pixel,
        memory_order_relaxed
    );
    if (triangleIndex == 0xffffffffu ||
        triangleIndex >= uniforms.live.w) {
        return;
    }
    const float meshDepth =
        as_type<float>(meshDepthBits);
    if (!(meshDepth < depth[pixel])) {
        return;
    }
    const uint environment = pixel / pixelsPerEnvironment;
    const uint localPixel =
        pixel - environment * pixelsPerEnvironment;
    const uint pixelX = localPixel % uniforms.image.x;
    const uint pixelY = localPixel / uniforms.image.x;
    if (!pixelInBand(pixelX, pixelY, uniforms)) {
        return;
    }
    const MRVisualTriangleGPUV2 triangle =
        triangles[triangleIndex];
    const MRVisualPrimitiveGPUV2 primitive =
        primitives[triangle.verticesAndPrimitive.w];
    const MRVisualInstanceGPUV2 visualInstance =
        visualInstances[primitive.geometry.w];
    const MRWorldInstanceHeaderGPU instance =
        instances[environment];
    const MRWorldSensorInstanceGPU sensor =
        sensors[instance.ranges.z + uniforms.render.x];
    const BoundPose camera = cameraPose(
        environment,
        instance,
        sensor,
        uniforms.render.x,
        assets,
        sensorBindings,
        currentBodies,
        uniforms,
        false
    );
    const BoundPose previousCamera = cameraPose(
        environment,
        instance,
        sensor,
        uniforms.render.x,
        assets,
        sensorBindings,
        previousBodies,
        uniforms,
        true
    );
    const BoundPose binding = composeInstancePose(resolvePose(
        visualInstance.binding.z,
        visualInstance.binding.y,
        visualInstance.binding.x,
        environment,
        instance,
        assets,
        currentBodies,
        uniforms,
        false
    ), visualInstance);
    const BoundPose previousBinding = composeInstancePose(resolvePose(
        visualInstance.binding.z,
        visualInstance.binding.y,
        visualInstance.binding.x,
        environment,
        instance,
        assets,
        previousBodies,
        uniforms,
        true
    ), visualInstance);
    if (!camera.valid || !binding.valid) {
        return;
    }

    const uint3 vertexIndices =
        triangle.verticesAndPrimitive.xyz;
    const MRVisualVertexGPUV2 vertex0 =
        vertices[vertexIndices.x];
    const MRVisualVertexGPUV2 vertex1 =
        vertices[vertexIndices.y];
    const MRVisualVertexGPUV2 vertex2 =
        vertices[vertexIndices.z];
    const float3 world0 =
        transformPoint(binding, vertex0.position.xyz);
    const float3 world1 =
        transformPoint(binding, vertex1.position.xyz);
    const float3 world2 =
        transformPoint(binding, vertex2.position.xyz);
    const CameraProjection p0 =
        projectPoint(world0, camera, sensor);
    const CameraProjection p1 =
        projectPoint(world1, camera, sensor);
    const CameraProjection p2 =
        projectPoint(world2, camera, sensor);
    const float area = edge(p0.pixel, p1.pixel, p2.pixel);
    if (!p0.valid || !p1.valid || !p2.valid ||
        abs(area) <= 1.0e-8f) {
        return;
    }
    const float2 pixelCenter =
        float2(float(pixelX), float(pixelY)) + 0.5f;
    float3 weights = float3(
        edge(p1.pixel, p2.pixel, pixelCenter) / area,
        edge(p2.pixel, p0.pixel, pixelCenter) / area,
        0.0f
    );
    weights.z = 1.0f - weights.x - weights.y;
    const float3 perspective = weights / float3(
        p0.depth,
        p1.depth,
        p2.depth
    );
    const float perspectiveSum =
        perspective.x + perspective.y + perspective.z;
    if (!(perspectiveSum > 0.0f)) {
        return;
    }
    const float3 corrected = perspective / perspectiveSum;
    const float3 localPosition =
        corrected.x * vertex0.position.xyz +
        corrected.y * vertex1.position.xyz +
        corrected.z * vertex2.position.xyz;
    float3 localNormal = normalize(
        corrected.x * vertex0.normalAndTangentSign.xyz +
        corrected.y * vertex1.normalAndTangentSign.xyz +
        corrected.z * vertex2.normalAndTangentSign.xyz
    );
    float3 localTangent = normalize(
        corrected.x * vertex0.tangent.xyz +
        corrected.y * vertex1.tangent.xyz +
        corrected.z * vertex2.tangent.xyz
    );
    const float tangentSign =
        corrected.x * vertex0.normalAndTangentSign.w +
        corrected.y * vertex1.normalAndTangentSign.w +
        corrected.z * vertex2.normalAndTangentSign.w < 0.0f
        ? -1.0f
        : 1.0f;
    const float4 texcoord01 =
        corrected.x * vertex0.texcoord01 +
        corrected.y * vertex1.texcoord01 +
        corrected.z * vertex2.texcoord01;
    const float4 vertexColor =
        corrected.x * vertex0.color +
        corrected.y * vertex1.color +
        corrected.z * vertex2.color;

    const MRVisualMaterialGPUV2 material =
        materials[primitive.geometry.z];
    float4 base =
        material.baseColorAndOpacity * vertexColor;
    base *= sampleVisualTexture(
        textureBindings,
        resources,
        material.textureIndices0.x,
        texcoord01,
        0.0f,
        uniforms
    );
    if (material.reserved.y != MR_INVALID_INDEX) {
        base.w *= sampleVisualTexture(
            textureBindings,
            resources,
            material.reserved.y,
            texcoord01,
            0.0f,
            uniforms
        ).x;
    }
    if (material.flags.x == MR_VISUAL_ALPHA_MASK &&
        base.w < material.coatingAndAlphaCutoff.w) {
        return;
    }
    const float3 sampledNormal = sampleVisualTexture(
        textureBindings,
        resources,
        material.textureIndices0.z,
        texcoord01,
        0.0f,
        uniforms
    ).xyz * 2.0f - 1.0f;
    if (material.textureIndices0.z != MR_INVALID_INDEX) {
        const float3 localBitangent =
            normalize(cross(localNormal, localTangent)) *
            tangentSign;
        localNormal = normalize(
            localTangent *
                (sampledNormal.x * material.surface.z) +
            localBitangent *
                (sampledNormal.y * material.surface.z) +
            localNormal * max(sampledNormal.z, 1.0e-4f)
        );
    }
    const float3 worldNormal =
        transformNormal(binding, localNormal);
    const float3 cameraNormal = normalize(
        inverseRotateVector(camera.orientation, worldNormal)
    );
    const float3 worldPosition =
        transformPoint(binding, localPosition);
    float2 pixelMotion = 0.0f;
    if ((uniforms.live.y & kLivePrevious) != 0u &&
        previousBinding.valid && previousCamera.valid) {
        const float3 previousWorld =
            transformPoint(previousBinding, localPosition);
        const CameraProjection previousProjection =
            projectPoint(previousWorld, previousCamera, sensor);
        if (previousProjection.valid) {
            pixelMotion =
                pixelCenter - previousProjection.pixel;
        }
    }
    uint4 outputIdentity = primitive.identity;
    if (visualInstance.identity.x != 0u) {
        outputIdentity.x = visualInstance.identity.x;
    }
    if (visualInstance.identity.y != 0u) {
        outputIdentity.y = visualInstance.identity.y;
    }
    if (visualInstance.identity.z != MR_INVALID_INDEX) {
        outputIdentity.z = visualInstance.identity.z;
    }
    if (uniforms.band.w != 0u) {
        depth[pixel] = meshDepth;
        segmentation[pixel] = outputIdentity.x;
        identities[pixel] = outputIdentity;
        normals[pixel] = float4(cameraNormal, 1.0f);
        motion[pixel] = float4(pixelMotion, 1.0f, 0.0f);
        validity[pixel] =
            MR_VISUAL_VALIDITY_FRAME |
            MR_VISUAL_VALIDITY_GEOMETRY;
        return;
    }

    const float4 metallicRoughness = sampleVisualTexture(
        textureBindings,
        resources,
        material.textureIndices0.y,
        texcoord01,
        0.0f,
        uniforms
    );
    const MRWorldAppearanceInstanceGPU appearance =
        appearances[instance.program.x];
    const float roughness = clamp(
        materialRoughness(
            material,
            textureBindings,
            resources,
            texcoord01,
            uniforms
        ) *
            appearance.material.z,
        0.045f,
        1.0f
    );
    const float metallicTexture =
        material.reserved.x != MR_INVALID_INDEX
        ? sampleVisualTexture(
              textureBindings,
              resources,
              material.reserved.x,
              texcoord01,
              0.0f,
              uniforms
          ).x
        : metallicRoughness.z;
    const float metallic = clamp(
        material.surface.y * metallicTexture *
            appearance.material.w,
        0.0f,
        1.0f
    );
    const float3 viewDirection =
        normalize(camera.position - worldPosition);
    const float noV =
        max(dot(worldNormal, viewDirection), 1.0e-4f);
    const float3 f0 = mix(
        float3(0.04f * material.coatingAndAlphaCutoff.z),
        base.xyz,
        metallic
    );
    const float aoSample = sampleVisualTexture(
        textureBindings,
        resources,
        material.textureIndices0.w,
        texcoord01,
        0.0f,
        uniforms
    ).x;
    const float ao = mix(
        1.0f,
        aoSample,
        clamp(material.surface.w, 0.0f, 1.0f)
    );
    float3 color = 0.0f;
    if ((material.flags.y & MR_VISUAL_MATERIAL_UNLIT) == 0u) {
        for (uint lightIndex = 0u;
             lightIndex < uniforms.presentation.y;
             ++lightIndex) {
            const MRVisualLightGPUV1 light = lights[lightIndex];
            float3 lightDirection = 0.0f;
            float attenuation = 1.0f;
            if (light.identity.x == MR_VISUAL_LIGHT_DIRECTIONAL) {
                lightDirection =
                    normalize(-light.directionAndSpot.xyz);
                attenuation =
                    light.colorAndIntensity.w * 0.001f;
            } else {
                const float3 toLight =
                    light.positionAndRange.xyz - worldPosition;
                const float distanceSquared =
                    max(dot(toLight, toLight), 1.0e-4f);
                const float distance = sqrt(distanceSquared);
                lightDirection = toLight / distance;
                const float rangeFade = saturate(
                    1.0f -
                    pow(
                        distance /
                            max(light.positionAndRange.w, 1.0e-3f),
                        4.0f
                    )
                );
                attenuation =
                    light.colorAndIntensity.w * 0.01f *
                    rangeFade * rangeFade / distanceSquared;
                if (light.identity.x == MR_VISUAL_LIGHT_SPOT) {
                    const float cosine = dot(
                        normalize(light.directionAndSpot.xyz),
                        -lightDirection
                    );
                    attenuation *= smoothstep(
                        light.directionAndSpot.w,
                        light.shape.z,
                        cosine
                    );
                }
            }
            const float noL =
                max(dot(worldNormal, lightDirection), 0.0f);
            if (noL <= 0.0f || attenuation <= 0.0f) {
                continue;
            }
            const float3 halfVector =
                normalize(viewDirection + lightDirection);
            const float noH =
                max(dot(worldNormal, halfVector), 0.0f);
            const float voH =
                max(dot(viewDirection, halfVector), 0.0f);
            const float3 fresnel = metalrobo_pbr::fresnelSchlick(voH, f0);
            const float distribution =
                metalrobo_pbr::distributionGGX(noH, roughness);
            const float visibility = metalrobo_pbr::visibilitySmithGGX(
                noV,
                noL,
                roughness
            );
            const float3 specular =
                distribution * visibility * fresnel;
            const float3 diffuse =
                (1.0f - fresnel) *
                (1.0f - metallic) * base.xyz / M_PI_F;
            const float shadow = shadowVisibility(
                shadowAtlas,
                worldPosition,
                worldNormal,
                lightDirection,
                light,
                lightIndex,
                environment,
                uniforms
            );
            color +=
                (diffuse + specular) * noL * shadow *
                light.colorAndIntensity.xyz * attenuation;
        }
        color += evaluateEnvironmentIBL(
            resources,
            environmentLighting,
            worldNormal,
            viewDirection,
            base.xyz,
            metallic,
            roughness,
            f0,
            ao
        );
        const float clearcoat =
            material.coatingAndAlphaCutoff.x *
            sampleVisualTexture(
                textureBindings,
                resources,
                material.textureIndices1.y,
                texcoord01,
                0.0f,
                uniforms
            ).x;
        if (clearcoat > 0.0f) {
            const float clearRoughness = clamp(
                materialClearcoatRoughness(
                    material,
                    textureBindings,
                    resources,
                    texcoord01,
                    uniforms
                ),
                0.045f,
                1.0f
            );
            const float coatFresnel =
                metalrobo_pbr::fresnelSchlick(
                    noV,
                    float3(0.04f)
                ).x;
            color *= 1.0f - clearcoat * coatFresnel;
            color += clearcoat * evaluateEnvironmentIBL(
                resources,
                environmentLighting,
                worldNormal,
                viewDirection,
                float3(0.0f),
                1.0f,
                clearRoughness,
                float3(0.04f),
                ao
            );
        }
    } else {
        color = base.xyz;
    }
    const float3 emissiveTexture = sampleVisualTexture(
        textureBindings,
        resources,
        material.textureIndices1.x,
        texcoord01,
        0.0f,
        uniforms
    ).xyz;
    color =
        color * appearance.colorAndLight.w +
        material.emissionAndStrength.xyz *
            emissiveTexture *
            material.emissionAndStrength.w;
    color = applyAppearance(color, appearance);

    rgb[pixel] = float4(color, base.w);
    depth[pixel] = meshDepth;
    segmentation[pixel] = outputIdentity.x;
    identities[pixel] = outputIdentity;
    normals[pixel] = float4(cameraNormal, 1.0f);
    motion[pixel] = float4(pixelMotion, 1.0f, 0.0f);
    validity[pixel] =
        MR_VISUAL_VALIDITY_FRAME |
        MR_VISUAL_VALIDITY_GEOMETRY;
}

kernel void mr_hybrid_prepare_ray_instances(
    const device MRVisualInstanceGPUV2* visualInstances [[buffer(0)]],
    const device uint* visibleInstances [[buffer(1)]],
    const device uint* blasIndices [[buffer(2)]],
    const device MRWorldInstanceHeaderGPU* instances [[buffer(3)]],
    const device MRWorldAssetInstanceGPU* assets [[buffer(4)]],
    const device MRBodyStateGPU* motionBodies [[buffer(5)]],
    device MTLAccelerationStructureMotionInstanceDescriptor*
        rayInstances [[buffer(6)]],
    device MTLComponentTransform* motionTransforms [[buffer(7)]],
    constant MRHybridRenderUniformsGPU& uniforms [[buffer(8)]],
    const uint descriptorIndex [[thread_position_in_grid]]
) {
    const uint visibleCount = uniforms.ray.x;
    const uint descriptorCount =
        uniforms.counts.x * visibleCount;
    if (descriptorIndex >= descriptorCount ||
        visibleCount == 0u ||
        uniforms.ray.y < 2u) {
        return;
    }
    const uint environment = descriptorIndex / visibleCount;
    const uint slot =
        descriptorIndex - environment * visibleCount;
    const uint visualInstanceIndex = visibleInstances[slot];
    const MRVisualInstanceGPUV2 visualInstance =
        visualInstances[visualInstanceIndex];
    const MRWorldInstanceHeaderGPU instance =
        instances[environment];

    bool valid = true;
    const uint firstTransform =
        descriptorIndex * uniforms.ray.y;
    const uint statesPerKeyframe =
        uniforms.counts.x * uniforms.live.x;
    for (uint keyframe = 0u;
         keyframe < uniforms.ray.y;
         ++keyframe) {
        const device MRBodyStateGPU* bodies =
            motionBodies + keyframe * statesPerKeyframe;
        const BoundPose pose = composeInstancePose(resolvePose(
            visualInstance.binding.z,
            visualInstance.binding.y,
            visualInstance.binding.x,
            environment,
            instance,
            assets,
            bodies,
            uniforms,
            false
        ), visualInstance);
        valid = valid && pose.valid;
        MTLComponentTransform transform;
        const float scale =
            pose.valid ? pose.scale : 1.0f;
        transform.scale = packed_float3(scale);
        transform.shear = packed_float3(0.0f);
        transform.pivot = packed_float3(0.0f);
        transform.rotation = packed_float4(
            pose.valid
                ? pose.orientation
                : float4(0.0f, 0.0f, 0.0f, 1.0f)
        );
        transform.translation = packed_float3(
            pose.valid ? pose.position : float3(0.0f)
        );
        motionTransforms[firstTransform + keyframe] =
            transform;
    }

    MTLAccelerationStructureMotionInstanceDescriptor descriptor{};
    descriptor.options =
        MTLAccelerationStructureInstanceOptionNone;
    uint mask = 0u;
    if ((visualInstance.binding.w &
         MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR) != 0u) {
        mask |= 1u;
    }
    if ((visualInstance.binding.w &
         MR_VISUAL_INSTANCE_CASTS_SHADOW) != 0u &&
        (visualInstance.binding.w &
         MR_VISUAL_INSTANCE_GAUSSIAN_RECEIVER_PROXY) == 0u) {
        mask |= 2u;
        if (visualInstance.binding.z ==
                MR_VISUAL_BINDING_RIGID_BODY ||
            visualInstance.binding.z ==
                MR_VISUAL_BINDING_ARTICULATED_LINK) {
            mask |= 4u;
        }
    }
    descriptor.mask = valid ? mask : 0u;
    descriptor.intersectionFunctionTableOffset = 0u;
    descriptor.accelerationStructureIndex = blasIndices[slot];
    descriptor.userID = visualInstanceIndex;
    descriptor.motionTransformsStartIndex = firstTransform;
    descriptor.motionTransformsCount = uniforms.ray.y;
    descriptor.motionStartBorderMode = MTLMotionBorderModeClamp;
    descriptor.motionEndBorderMode = MTLMotionBorderModeClamp;
    descriptor.motionStartTime = 0.0f;
    descriptor.motionEndTime = 1.0f;
    rayInstances[descriptorIndex] = descriptor;
}

using MRReferenceAccelerationStructure =
    acceleration_structure<instancing, instance_motion>;
using MRReferenceIntersector =
    intersector<
        triangle_data,
        instancing,
        world_space_data,
        instance_motion
    >;
using MRReferenceIntersection =
    MRReferenceIntersector::result_type;

struct MRReferenceSurface {
    uint environment;
    uint visualInstanceIndex;
    uint primitiveIndex;
    float3 worldPosition;
    float3 worldNormal;
    float3 worldTangent;
    float tangentSign;
    float4 texcoord01;
    float4 vertexColor;
    float4 base;
    MRVisualMaterialGPUV2 material;
    uint4 identity;
};

float4 referenceQuaternionSlerp(
    float4 first,
    float4 second,
    const float fraction
) {
    first = normalize(first);
    second = normalize(second);
    float cosine = dot(first, second);
    if (cosine < 0.0f) {
        second = -second;
        cosine = -cosine;
    }
    if (cosine > 0.9995f) {
        return normalize(mix(first, second, fraction));
    }
    const float angle = acos(clamp(cosine, -1.0f, 1.0f));
    const float inverseSine =
        1.0f / max(sin(angle), 1.0e-6f);
    return normalize(
        sin((1.0f - fraction) * angle) * inverseSine * first +
        sin(fraction * angle) * inverseSine * second
    );
}

BoundPose referenceBodyPose(
    const device MRBodyStateGPU* motionBodies,
    const uint environment,
    const uint body,
    const float time,
    constant MRHybridRenderUniformsGPU& uniforms
) {
    if (body >= uniforms.live.x || uniforms.ray.y < 2u) {
        BoundPose invalid = worldPose();
        invalid.valid = false;
        return invalid;
    }
    const float scaled =
        clamp(time, 0.0f, 1.0f) *
        float(uniforms.ray.y - 1u);
    const uint firstKeyframe =
        min(uint(floor(scaled)), uniforms.ray.y - 1u);
    const uint secondKeyframe =
        min(firstKeyframe + 1u, uniforms.ray.y - 1u);
    const float fraction = scaled - float(firstKeyframe);
    const uint statesPerKeyframe =
        uniforms.counts.x * uniforms.live.x;
    const uint bodyOffset =
        environment * uniforms.live.x + body;
    const MRBodyStateGPU first =
        motionBodies[
            firstKeyframe * statesPerKeyframe + bodyOffset
        ];
    const MRBodyStateGPU second =
        motionBodies[
            secondKeyframe * statesPerKeyframe + bodyOffset
        ];
    BoundPose result;
    result.position = mix(
        first.position.xyz,
        second.position.xyz,
        fraction
    );
    result.orientation = referenceQuaternionSlerp(
        first.orientation,
        second.orientation,
        fraction
    );
    result.scale = 1.0f;
    result.valid =
        all(isfinite(result.position)) &&
        all(isfinite(result.orientation));
    return result;
}

BoundPose referenceResolvePose(
    const uint bindingKind,
    const uint bindingIndex,
    const uint asset,
    const uint environment,
    const MRWorldInstanceHeaderGPU instance,
    const device MRWorldAssetInstanceGPU* assets,
    const device MRBodyStateGPU* motionBodies,
    const float time,
    constant MRHybridRenderUniformsGPU& uniforms
) {
    if (bindingKind == MR_VISUAL_BINDING_WORLD) {
        return worldPose();
    }
    if (bindingKind == MR_VISUAL_BINDING_ASSET) {
        return assetPose(assets, instance, asset);
    }
    if (bindingKind == MR_VISUAL_BINDING_RIGID_BODY ||
        bindingKind == MR_VISUAL_BINDING_ARTICULATED_LINK) {
        return referenceBodyPose(
            motionBodies,
            environment,
            bindingIndex,
            time,
            uniforms
        );
    }
    BoundPose invalid = worldPose();
    invalid.valid = false;
    return invalid;
}

BoundPose referenceCameraPose(
    const uint environment,
    const MRWorldInstanceHeaderGPU instance,
    const MRWorldSensorInstanceGPU sensor,
    const device MRWorldAssetInstanceGPU* assets,
    const device MRVisualSensorBindingGPU* sensorBindings,
    const device MRBodyStateGPU* motionBodies,
    const float time,
    constant MRHybridRenderUniformsGPU& uniforms
) {
    BoundPose parent;
    if (uniforms.render.x < uniforms.live.z) {
        const MRVisualSensorBindingGPU binding =
            sensorBindings[uniforms.render.x];
        parent = referenceResolvePose(
            binding.identity.x,
            binding.identity.y,
            binding.identity.z,
            environment,
            instance,
            assets,
            motionBodies,
            time,
            uniforms
        );
    } else {
        parent = assetPose(
            assets,
            instance,
            sensor.identity.x
        );
    }
    if (!parent.valid) {
        return parent;
    }
    parent.position += rotateVector(
        parent.orientation,
        sensor.positionAndFocalScale.xyz * parent.scale
    );
    parent.orientation = normalize(quaternionProduct(
        parent.orientation,
        sensor.orientation
    ));
    parent.scale = 1.0f;
    return parent;
}

float referenceScanFraction(
    const uint2 pixel,
    constant MRHybridRenderUniformsGPU& uniforms
) {
    if (uniforms.shutter.x != MR_VISUAL_SHUTTER_ROLLING) {
        return 0.0f;
    }
    switch (uniforms.shutter.y) {
    case MR_VISUAL_SHUTTER_BOTTOM_TO_TOP:
        return (
            float(uniforms.image.y) -
            (float(pixel.y) + 0.5f)
        ) / max(float(uniforms.image.y), 1.0f);
    case MR_VISUAL_SHUTTER_LEFT_TO_RIGHT:
        return (float(pixel.x) + 0.5f) /
            max(float(uniforms.image.x), 1.0f);
    case MR_VISUAL_SHUTTER_RIGHT_TO_LEFT:
        return (
            float(uniforms.image.x) -
            (float(pixel.x) + 0.5f)
        ) / max(float(uniforms.image.x), 1.0f);
    default:
        return (float(pixel.y) + 0.5f) /
            max(float(uniforms.image.y), 1.0f);
    }
}

ray referenceCameraRay(
    const uint2 pixel,
    const MRWorldSensorInstanceGPU sensor,
    const BoundPose camera,
    constant MRHybridRenderUniformsGPU& uniforms
) {
    const float focalScale =
        sensor.positionAndFocalScale.w;
    const float2 focal =
        max(sensor.intrinsics.xy * focalScale, 1.0e-6f);
    const float2 distorted =
        (
            float2(pixel) + 0.5f -
            sensor.intrinsics.zw
        ) / focal;
    float2 normalized = distorted;
    // Invert the Brown-Conrady lens model used by projectPoint.
    for (uint iteration = 0u; iteration < 5u; ++iteration) {
        const float radiusSquared = dot(normalized, normalized);
        const float radial =
            1.0f +
            sensor.distortion.x * radiusSquared +
            sensor.distortion.y *
                radiusSquared * radiusSquared;
        const float2 tangential = float2(
            2.0f * sensor.distortion.z *
                    normalized.x * normalized.y +
                sensor.distortion.w *
                    (radiusSquared +
                     2.0f * normalized.x * normalized.x),
            sensor.distortion.z *
                    (radiusSquared +
                     2.0f * normalized.y * normalized.y) +
                2.0f * sensor.distortion.w *
                    normalized.x * normalized.y
        );
        normalized +=
            distorted - (normalized * radial + tangential);
    }
    ray result;
    result.origin = camera.position;
    result.direction = normalize(rotateVector(
        camera.orientation,
        normalize(float3(normalized, 1.0f))
    ));
    result.max_distance =
        uniforms.sensorRangeAndResponse.y * 2.0f;
    return result;
}

bool referenceSurface(
    const MRReferenceIntersection intersection,
    const float3 worldPosition,
    const uint visibleCount,
    const device uint* visibleInstances,
    const device MRVisualVertexGPUV2* vertices,
    const device uint* indices,
    const device MRVisualPrimitiveGPUV2* primitives,
    const device MRVisualInstanceGPUV2* visualInstances,
    const device MRVisualMaterialGPUV2* materials,
    const device MRVisualTextureBindingGPUV2* textureBindings,
    constant VisualResourceTableV3& resources,
    constant MRHybridRenderUniformsGPU& uniforms,
    thread MRReferenceSurface& surface
) {
    if (visibleCount == 0u ||
        intersection.instance_id >=
            uniforms.counts.x * visibleCount) {
        return false;
    }
    const uint environment =
        intersection.instance_id / visibleCount;
    const uint slot =
        intersection.instance_id -
        environment * visibleCount;
    const uint visualInstanceIndex =
        visibleInstances[slot];
    const MRVisualInstanceGPUV2 visualInstance =
        visualInstances[visualInstanceIndex];
    if (intersection.geometry_id >=
        visualInstance.geometry.y) {
        return false;
    }
    const uint primitiveIndex =
        visualInstance.geometry.x +
        intersection.geometry_id;
    const MRVisualPrimitiveGPUV2 primitive =
        primitives[primitiveIndex];
    const uint firstIndex =
        primitive.geometry.x +
        intersection.primitive_id * 3u;
    if (firstIndex + 2u >=
        primitive.geometry.x + primitive.geometry.y) {
        return false;
    }
    const uint3 vertexIndices = uint3(
        indices[firstIndex],
        indices[firstIndex + 1u],
        indices[firstIndex + 2u]
    );
    const MRVisualVertexGPUV2 vertex0 =
        vertices[vertexIndices.x];
    const MRVisualVertexGPUV2 vertex1 =
        vertices[vertexIndices.y];
    const MRVisualVertexGPUV2 vertex2 =
        vertices[vertexIndices.z];
    const float2 barycentric =
        intersection.triangle_barycentric_coord;
    const float3 weights = float3(
        1.0f - barycentric.x - barycentric.y,
        barycentric.x,
        barycentric.y
    );
    float3 localNormal = normalize(
        weights.x * vertex0.normalAndTangentSign.xyz +
        weights.y * vertex1.normalAndTangentSign.xyz +
        weights.z * vertex2.normalAndTangentSign.xyz
    );
    float3 localTangent = normalize(
        weights.x * vertex0.tangent.xyz +
        weights.y * vertex1.tangent.xyz +
        weights.z * vertex2.tangent.xyz
    );
    const float tangentSign =
        dot(
            weights,
            float3(
                vertex0.normalAndTangentSign.w,
                vertex1.normalAndTangentSign.w,
                vertex2.normalAndTangentSign.w
            )
        ) < 0.0f
        ? -1.0f
        : 1.0f;
    const float4 texcoord01 =
        weights.x * vertex0.texcoord01 +
        weights.y * vertex1.texcoord01 +
        weights.z * vertex2.texcoord01;
    const float4 vertexColor =
        weights.x * vertex0.color +
        weights.y * vertex1.color +
        weights.z * vertex2.color;
    const MRVisualMaterialGPUV2 material =
        materials[primitive.geometry.z];
    float4 base =
        material.baseColorAndOpacity * vertexColor;
    base *= sampleVisualTexture(
        textureBindings,
        resources,
        material.textureIndices0.x,
        texcoord01,
        0.0f,
        uniforms
    );
    if (material.reserved.y != MR_INVALID_INDEX) {
        base.w *= sampleVisualTexture(
            textureBindings,
            resources,
            material.reserved.y,
            texcoord01,
            0.0f,
            uniforms
        ).x;
    }

    if (material.textureIndices0.z != MR_INVALID_INDEX) {
        const float3 sampledNormal = sampleVisualTexture(
            textureBindings,
            resources,
            material.textureIndices0.z,
            texcoord01,
            0.0f,
            uniforms
        ).xyz * 2.0f - 1.0f;
        const float3 localBitangent =
            normalize(cross(localNormal, localTangent)) *
            tangentSign;
        localNormal = normalize(
            localTangent *
                (sampledNormal.x * material.surface.z) +
            localBitangent *
                (sampledNormal.y * material.surface.z) +
            localNormal * max(sampledNormal.z, 1.0e-4f)
        );
    }
    const float4x3 transform =
        intersection.object_to_world_transform;
    surface.environment = environment;
    surface.visualInstanceIndex = visualInstanceIndex;
    surface.primitiveIndex = primitiveIndex;
    surface.worldPosition = worldPosition;
    surface.worldNormal = normalize(
        transform * float4(localNormal, 0.0f)
    );
    surface.worldTangent = normalize(
        transform * float4(localTangent, 0.0f)
    );
    surface.tangentSign = tangentSign;
    surface.texcoord01 = texcoord01;
    surface.vertexColor = vertexColor;
    surface.base = base;
    surface.material = material;
    surface.identity = primitive.identity;
    if (visualInstance.identity.x != 0u) {
        surface.identity.x = visualInstance.identity.x;
    }
    if (visualInstance.identity.y != 0u) {
        surface.identity.y = visualInstance.identity.y;
    }
    if (visualInstance.identity.z != MR_INVALID_INDEX) {
        surface.identity.z = visualInstance.identity.z;
    }
    return all(isfinite(surface.worldPosition)) &&
        all(isfinite(surface.worldNormal));
}

bool referenceTrace(
    ray queryRay,
    const uint mask,
    const float time,
    const uint seed,
    MRReferenceAccelerationStructure accelerationStructure,
    const device uint* visibleInstances,
    const device MRVisualVertexGPUV2* vertices,
    const device uint* indices,
    const device MRVisualPrimitiveGPUV2* primitives,
    const device MRVisualInstanceGPUV2* visualInstances,
    const device MRVisualMaterialGPUV2* materials,
    const device MRVisualTextureBindingGPUV2* textureBindings,
    constant VisualResourceTableV3& resources,
    constant MRHybridRenderUniformsGPU& uniforms,
    thread MRReferenceSurface& surface,
    thread float& totalDistance
) {
    MRReferenceIntersector intersector;
    intersector.assume_geometry_type(geometry_type::triangle);
    intersector.force_opacity(forced_opacity::opaque);
    intersector.accept_any_intersection(false);
    totalDistance = 0.0f;
    for (uint layer = 0u; layer < 8u; ++layer) {
        const MRReferenceIntersection intersection =
            intersector.intersect(
                queryRay,
                accelerationStructure,
                mask,
                time
            );
        if (intersection.type == intersection_type::none) {
            return false;
        }
        const float distance =
            totalDistance + intersection.distance;
        MRReferenceSurface candidate;
        if (!referenceSurface(
                intersection,
                queryRay.origin +
                    queryRay.direction * intersection.distance,
                uniforms.ray.x,
                visibleInstances,
                vertices,
                indices,
                primitives,
                visualInstances,
                materials,
                textureBindings,
                resources,
                uniforms,
                candidate
            )) {
            return false;
        }
        bool accepted = true;
        if (candidate.material.flags.x ==
                MR_VISUAL_ALPHA_MASK) {
            accepted =
                candidate.base.w >=
                candidate.material.coatingAndAlphaCutoff.w;
        } else if (candidate.material.flags.x ==
                   MR_VISUAL_ALPHA_BLEND) {
            const float sample =
                (float(randomHash(
                    seed ^ layer * 0x9e3779b9u
                )) + 0.5f) /
                4294967296.0f;
            accepted = sample < candidate.base.w;
        }
        if (accepted) {
            surface = candidate;
            totalDistance = distance;
            return true;
        }
        const float advance =
            intersection.distance + 1.0e-4f;
        if (!(advance < queryRay.max_distance)) {
            return false;
        }
        queryRay.origin += queryRay.direction * advance;
        queryRay.max_distance -= advance;
        totalDistance += advance;
    }
    return false;
}

float referenceShadow(
    const MRReferenceSurface surface,
    const float3 lightDirection,
    const float lightDistance,
    const uint mask,
    const float time,
    const uint seed,
    MRReferenceAccelerationStructure accelerationStructure,
    const device uint* visibleInstances,
    const device MRVisualVertexGPUV2* vertices,
    const device uint* indices,
    const device MRVisualPrimitiveGPUV2* primitives,
    const device MRVisualInstanceGPUV2* visualInstances,
    const device MRVisualMaterialGPUV2* materials,
    const device MRVisualTextureBindingGPUV2* textureBindings,
    constant VisualResourceTableV3& resources,
    constant MRHybridRenderUniformsGPU& uniforms
) {
    ray shadowRay;
    shadowRay.origin =
        surface.worldPosition +
        surface.worldNormal * 1.0e-4f;
    shadowRay.direction = lightDirection;
    shadowRay.max_distance =
        max(lightDistance - 2.0e-4f, 1.0e-4f);
    MRReferenceSurface blocker;
    float blockerDistance = 0.0f;
    return referenceTrace(
        shadowRay,
        mask,
        time,
        seed,
        accelerationStructure,
        visibleInstances,
        vertices,
        indices,
        primitives,
        visualInstances,
        materials,
        textureBindings,
        resources,
        uniforms,
        blocker,
        blockerDistance
    ) ? 0.0f : 1.0f;
}

float3 referenceShade(
    const MRReferenceSurface surface,
    const float3 viewDirection,
    const float time,
    const uint seed,
    MRReferenceAccelerationStructure accelerationStructure,
    const device uint* visibleInstances,
    const device MRVisualVertexGPUV2* vertices,
    const device uint* indices,
    const device MRVisualPrimitiveGPUV2* primitives,
    const device MRVisualInstanceGPUV2* visualInstances,
    const device MRVisualMaterialGPUV2* materials,
    const device MRVisualTextureBindingGPUV2* textureBindings,
    constant VisualResourceTableV3& resources,
    const device MRVisualLightGPUV1* lights,
    constant MRVisualEnvironmentGPUV2& environmentLighting,
    const device MRWorldInstanceHeaderGPU* instances,
    const device MRWorldAppearanceInstanceGPU* appearances,
    constant MRHybridRenderUniformsGPU& uniforms
) {
    const MRWorldInstanceHeaderGPU instance =
        instances[surface.environment];
    const MRWorldAppearanceInstanceGPU appearance =
        appearances[instance.program.x];
    const float4 metallicRoughness = sampleVisualTexture(
        textureBindings,
        resources,
        surface.material.textureIndices0.y,
        surface.texcoord01,
        0.0f,
        uniforms
    );
    const float roughness = clamp(
        materialRoughness(
            surface.material,
            textureBindings,
            resources,
            surface.texcoord01,
            uniforms
        ) *
            appearance.material.z,
        0.045f,
        1.0f
    );
    const float metallicTexture =
        surface.material.reserved.x != MR_INVALID_INDEX
        ? sampleVisualTexture(
              textureBindings,
              resources,
              surface.material.reserved.x,
              surface.texcoord01,
              0.0f,
              uniforms
          ).x
        : metallicRoughness.z;
    const float metallic = clamp(
        surface.material.surface.y *
            metallicTexture *
            appearance.material.w,
        0.0f,
        1.0f
    );
    const float noV = max(
        dot(surface.worldNormal, viewDirection),
        1.0e-4f
    );
    const float3 f0 = mix(
        float3(
            0.04f *
            surface.material.coatingAndAlphaCutoff.z
        ),
        surface.base.xyz,
        metallic
    );
    const float aoSample = sampleVisualTexture(
        textureBindings,
        resources,
        surface.material.textureIndices0.w,
        surface.texcoord01,
        0.0f,
        uniforms
    ).x;
    const float ao = mix(
        1.0f,
        aoSample,
        clamp(surface.material.surface.w, 0.0f, 1.0f)
    );
    float3 color = 0.0f;
    if ((surface.material.flags.y &
         MR_VISUAL_MATERIAL_UNLIT) == 0u) {
        for (uint lightIndex = 0u;
             lightIndex < uniforms.presentation.y;
             ++lightIndex) {
            const MRVisualLightGPUV1 light =
                lights[lightIndex];
            const uint sampleCount =
                light.identity.x == MR_VISUAL_LIGHT_RECTANGLE
                ? max(uniforms.ray.z, 1u)
                : 1u;
            float3 sampledContribution = 0.0f;
            for (uint lightSample = 0u;
                 lightSample < sampleCount;
                 ++lightSample) {
                float3 lightDirection = 0.0f;
                float attenuation = 1.0f;
                float lightDistance =
                    uniforms.sensorRangeAndResponse.y * 4.0f;
                if (light.identity.x ==
                    MR_VISUAL_LIGHT_DIRECTIONAL) {
                    lightDirection =
                        normalize(-light.directionAndSpot.xyz);
                    attenuation =
                        light.colorAndIntensity.w * 0.001f;
                } else {
                    float3 lightPosition =
                        light.positionAndRange.xyz;
                    if (light.identity.x ==
                        MR_VISUAL_LIGHT_RECTANGLE) {
                        const float3 forward = normalize(
                            light.directionAndSpot.xyz
                        );
                        const float3 helper =
                            abs(forward.z) < 0.99f
                            ? float3(0.0f, 0.0f, 1.0f)
                            : float3(0.0f, 1.0f, 0.0f);
                        const float3 right =
                            normalize(cross(helper, forward));
                        const float3 up =
                            normalize(cross(forward, right));
                        const uint grid = uint(ceil(sqrt(
                            float(sampleCount)
                        )));
                        const uint sampleSeed = randomHash(
                            seed ^
                            lightIndex * 0x85ebca6bu ^
                            lightSample * 0xc2b2ae35u
                        );
                        const float2 jitter = float2(
                            uniformSigned(sampleSeed),
                            uniformSigned(
                                sampleSeed ^ 0x27d4eb2fu
                            )
                        ) * 0.25f;
                        const float2 cell = (
                            float2(
                                lightSample % grid,
                                lightSample / grid
                            ) +
                            0.5f + jitter
                        ) / float(grid) - 0.5f;
                        lightPosition +=
                            right * cell.x * light.shape.x +
                            up * cell.y * light.shape.y;
                    }
                    const float3 toLight =
                        lightPosition - surface.worldPosition;
                    const float distanceSquared =
                        max(dot(toLight, toLight), 1.0e-4f);
                    lightDistance = sqrt(distanceSquared);
                    lightDirection =
                        toLight / lightDistance;
                    const float rangeFade = saturate(
                        1.0f -
                        pow(
                            lightDistance /
                                max(
                                    light.positionAndRange.w,
                                    1.0e-3f
                                ),
                            4.0f
                        )
                    );
                    attenuation =
                        light.colorAndIntensity.w * 0.01f *
                        rangeFade * rangeFade /
                        distanceSquared;
                    if (light.identity.x ==
                        MR_VISUAL_LIGHT_SPOT) {
                        const float cosine = dot(
                            normalize(
                                light.directionAndSpot.xyz
                            ),
                            -lightDirection
                        );
                        attenuation *= smoothstep(
                            light.directionAndSpot.w,
                            light.shape.z,
                            cosine
                        );
                    }
                    if (light.identity.x ==
                        MR_VISUAL_LIGHT_RECTANGLE) {
                        attenuation *= saturate(dot(
                            normalize(
                                light.directionAndSpot.xyz
                            ),
                            -lightDirection
                        ));
                    }
                }
                const float noL = max(
                    dot(surface.worldNormal, lightDirection),
                    0.0f
                );
                if (noL <= 0.0f || attenuation <= 0.0f) {
                    continue;
                }
                const float3 halfVector =
                    normalize(viewDirection + lightDirection);
                const float noH = max(
                    dot(surface.worldNormal, halfVector),
                    0.0f
                );
                const float voH = max(
                    dot(viewDirection, halfVector),
                    0.0f
                );
                const float3 fresnel =
                    metalrobo_pbr::fresnelSchlick(voH, f0);
                const float3 specular =
                    metalrobo_pbr::distributionGGX(
                        noH,
                        roughness
                    ) *
                    metalrobo_pbr::visibilitySmithGGX(
                        noV,
                        noL,
                        roughness
                    ) *
                    fresnel;
                const float3 diffuse =
                    (1.0f - fresnel) *
                    (1.0f - metallic) *
                    surface.base.xyz / M_PI_F;
                const float shadow = referenceShadow(
                    surface,
                    lightDirection,
                    lightDistance,
                    2u,
                    time,
                    seed ^
                        lightIndex * 0x165667b1u ^
                        lightSample,
                    accelerationStructure,
                    visibleInstances,
                    vertices,
                    indices,
                    primitives,
                    visualInstances,
                    materials,
                    textureBindings,
                    resources,
                    uniforms
                );
                sampledContribution +=
                    (diffuse + specular) * noL * shadow *
                    light.colorAndIntensity.xyz *
                    attenuation;
            }
            color += sampledContribution / float(sampleCount);
        }
        color += evaluateEnvironmentIBL(
            resources,
            environmentLighting,
            surface.worldNormal,
            viewDirection,
            surface.base.xyz,
            metallic,
            roughness,
            f0,
            ao
        );
        const float clearcoat =
            surface.material.coatingAndAlphaCutoff.x *
            sampleVisualTexture(
                textureBindings,
                resources,
                surface.material.textureIndices1.y,
                surface.texcoord01,
                0.0f,
                uniforms
            ).x;
        if (clearcoat > 0.0f) {
            const float clearRoughness = clamp(
                materialClearcoatRoughness(
                    surface.material,
                    textureBindings,
                    resources,
                    surface.texcoord01,
                    uniforms
                ),
                0.045f,
                1.0f
            );
            const float coatFresnel =
                metalrobo_pbr::fresnelSchlick(
                    noV,
                    float3(0.04f)
                ).x;
            color *= 1.0f - clearcoat * coatFresnel;
            color += clearcoat * evaluateEnvironmentIBL(
                resources,
                environmentLighting,
                surface.worldNormal,
                viewDirection,
                float3(0.0f),
                1.0f,
                clearRoughness,
                float3(0.04f),
                ao
            );
        }
    } else {
        color = surface.base.xyz;
    }
    const float3 emissiveTexture = sampleVisualTexture(
        textureBindings,
        resources,
        surface.material.textureIndices1.x,
        surface.texcoord01,
        0.0f,
        uniforms
    ).xyz;
    color =
        color * appearance.colorAndLight.w +
        surface.material.emissionAndStrength.xyz *
            emissiveTexture *
            surface.material.emissionAndStrength.w;
    return applyAppearance(color, appearance);
}

float referenceReceiverShadow(
    const MRReferenceSurface surface,
    const float time,
    const uint seed,
    MRReferenceAccelerationStructure accelerationStructure,
    const device uint* visibleInstances,
    const device MRVisualVertexGPUV2* vertices,
    const device uint* indices,
    const device MRVisualPrimitiveGPUV2* primitives,
    const device MRVisualInstanceGPUV2* visualInstances,
    const device MRVisualMaterialGPUV2* materials,
    const device MRVisualTextureBindingGPUV2* textureBindings,
    constant VisualResourceTableV3& resources,
    const device MRVisualLightGPUV1* lights,
    constant MRHybridRenderUniformsGPU& uniforms
) {
    for (uint lightIndex = 0u;
         lightIndex < uniforms.presentation.y;
         ++lightIndex) {
        const MRVisualLightGPUV1 light = lights[lightIndex];
        if (light.shadow.x == 0u) {
            continue;
        }
        float3 direction;
        float distance;
        if (light.identity.x == MR_VISUAL_LIGHT_DIRECTIONAL) {
            direction =
                normalize(-light.directionAndSpot.xyz);
            distance =
                uniforms.sensorRangeAndResponse.y * 4.0f;
        } else {
            const float3 toLight =
                light.positionAndRange.xyz -
                surface.worldPosition;
            distance = length(toLight);
            direction = toLight / max(distance, 1.0e-6f);
        }
        return referenceShadow(
            surface,
            direction,
            distance,
            4u,
            time,
            seed,
            accelerationStructure,
            visibleInstances,
            vertices,
            indices,
            primitives,
            visualInstances,
            materials,
            textureBindings,
            resources,
            uniforms
        );
    }
    return 1.0f;
}

kernel void mr_hybrid_render_reference(
    const device MRVisualVertexGPUV2* vertices [[buffer(0)]],
    const device uint* indices [[buffer(1)]],
    const device MRVisualPrimitiveGPUV2* primitives [[buffer(2)]],
    const device MRVisualInstanceGPUV2* visualInstances [[buffer(3)]],
    const device uint* visibleInstances [[buffer(4)]],
    const device MRVisualMaterialGPUV2* materials [[buffer(5)]],
    const device MRVisualTextureBindingGPUV2* textureBindings [[buffer(6)]],
    constant VisualResourceTableV3& resources [[buffer(7)]],
    const device MRVisualLightGPUV1* lights [[buffer(8)]],
    constant MRVisualEnvironmentGPUV2& environmentLighting
        [[buffer(9)]],
    const device MRWorldInstanceHeaderGPU* instances [[buffer(10)]],
    const device MRWorldAssetInstanceGPU* assets [[buffer(11)]],
    const device MRWorldSensorInstanceGPU* sensors [[buffer(12)]],
    const device MRWorldAppearanceInstanceGPU* appearances [[buffer(13)]],
    const device MRVisualSensorBindingGPU* sensorBindings [[buffer(14)]],
    const device MRBodyStateGPU* motionBodies [[buffer(15)]],
    device float4* rgb [[buffer(16)]],
    device float* depth [[buffer(17)]],
    device uint* segmentation [[buffer(18)]],
    device uint4* identities [[buffer(19)]],
    device float4* normals [[buffer(20)]],
    device float4* motion [[buffer(21)]],
    device uint* validity [[buffer(22)]],
    constant MRHybridRenderUniformsGPU& uniforms [[buffer(23)]],
    MRReferenceAccelerationStructure accelerationStructure
        [[buffer(24)]],
    const uint pixel [[thread_position_in_grid]]
) {
    const uint pixelsPerEnvironment =
        uniforms.image.x * uniforms.image.y;
    const uint pixelCount =
        uniforms.counts.x * pixelsPerEnvironment;
    if (pixel >= pixelCount ||
        uniforms.ray.x == 0u ||
        uniforms.ray.y < 2u ||
        uniforms.shutter.w == 0u) {
        return;
    }
    const uint environment =
        pixel / pixelsPerEnvironment;
    const uint localPixel =
        pixel - environment * pixelsPerEnvironment;
    const uint2 coordinate = uint2(
        localPixel % uniforms.image.x,
        localPixel / uniforms.image.x
    );
    const MRWorldInstanceHeaderGPU instance =
        instances[environment];
    const MRWorldSensorInstanceGPU sensor =
        sensors[instance.ranges.z + uniforms.render.x];
    const float3 background = rgb[pixel].xyz;
    const float backgroundDepth = depth[pixel];
    const float scanFraction =
        referenceScanFraction(coordinate, uniforms);
    const float exposure =
        max(uniforms.sensorTiming.y, 0.0f);
    const float readout =
        uniforms.shutter.x == MR_VISUAL_SHUTTER_ROLLING
        ? max(uniforms.sensorTiming.z, 0.0f)
        : 0.0f;
    const float shutterWindow =
        max(uniforms.rayTiming.x, 1.0e-8f);
    const uint seed =
        frameSeed(instance, uniforms, pixel);

    float3 integrated = 0.0f;
    for (uint sample = 0u;
         sample < uniforms.shutter.w;
         ++sample) {
        const float sampleFraction =
            (float(sample) + 0.5f) /
            float(uniforms.shutter.w);
        const float time = clamp(
            (
                scanFraction * readout +
                sampleFraction * exposure
            ) / shutterWindow,
            0.0f,
            1.0f
        );
        const BoundPose camera = referenceCameraPose(
            environment,
            instance,
            sensor,
            assets,
            sensorBindings,
            motionBodies,
            time,
            uniforms
        );
        if (!camera.valid) {
            integrated += background;
            continue;
        }
        const ray primaryRay = referenceCameraRay(
            coordinate,
            sensor,
            camera,
            uniforms
        );
        MRReferenceSurface surface;
        float hitDistance = 0.0f;
        if (!referenceTrace(
                primaryRay,
                1u,
                time,
                seed ^ sample * 0x9e3779b9u,
                accelerationStructure,
                visibleInstances,
                vertices,
                indices,
                primitives,
                visualInstances,
                materials,
                textureBindings,
                resources,
                uniforms,
                surface,
                hitDistance
            )) {
            integrated += background;
            continue;
        }
        const float hitDepth = inverseRotateVector(
            camera.orientation,
            surface.worldPosition - camera.position
        ).z;
        if (!(hitDepth > 0.0f) ||
            !(hitDepth < backgroundDepth)) {
            integrated += background;
            continue;
        }
        const MRVisualInstanceGPUV2 visualInstance =
            visualInstances[surface.visualInstanceIndex];
        if ((visualInstance.binding.w &
             MR_VISUAL_INSTANCE_GAUSSIAN_RECEIVER_PROXY) != 0u) {
            integrated += background *
                mix(
                    0.55f,
                    1.0f,
                    referenceReceiverShadow(
                        surface,
                        time,
                        seed ^ sample,
                        accelerationStructure,
                        visibleInstances,
                        vertices,
                        indices,
                        primitives,
                        visualInstances,
                        materials,
                        textureBindings,
                        resources,
                        lights,
                        uniforms
                    )
                );
        } else {
            integrated += referenceShade(
                surface,
                normalize(camera.position - surface.worldPosition),
                time,
                seed ^ sample * 0x85ebca6bu,
                accelerationStructure,
                visibleInstances,
                vertices,
                indices,
                primitives,
                visualInstances,
                materials,
                textureBindings,
                resources,
                lights,
                environmentLighting,
                instances,
                appearances,
                uniforms
            );
        }
    }
    rgb[pixel] = float4(
        integrated / float(uniforms.shutter.w),
        1.0f
    );

    // Deployable geometry channels use the exact row-exposure midpoint.
    const float truthTime = clamp(
        (
            scanFraction * readout +
            0.5f * exposure
        ) / shutterWindow,
        0.0f,
        1.0f
    );
    const BoundPose truthCamera = referenceCameraPose(
        environment,
        instance,
        sensor,
        assets,
        sensorBindings,
        motionBodies,
        truthTime,
        uniforms
    );
    if (!truthCamera.valid) {
        return;
    }
    const ray truthRay = referenceCameraRay(
        coordinate,
        sensor,
        truthCamera,
        uniforms
    );
    MRReferenceSurface truthSurface;
    float truthDistance = 0.0f;
    if (!referenceTrace(
            truthRay,
            1u,
            truthTime,
            seed ^ 0x27d4eb2fu,
            accelerationStructure,
            visibleInstances,
            vertices,
            indices,
            primitives,
            visualInstances,
            materials,
            textureBindings,
            resources,
            uniforms,
            truthSurface,
            truthDistance
        )) {
        return;
    }
    const float truthDepth = inverseRotateVector(
        truthCamera.orientation,
        truthSurface.worldPosition - truthCamera.position
    ).z;
    const MRVisualInstanceGPUV2 truthInstance =
        visualInstances[truthSurface.visualInstanceIndex];
    if (!(truthDepth > 0.0f) ||
        !(truthDepth < depth[pixel]) ||
        (truthInstance.binding.w &
         MR_VISUAL_INSTANCE_GAUSSIAN_RECEIVER_PROXY) != 0u) {
        return;
    }
    depth[pixel] = truthDepth;
    segmentation[pixel] = truthSurface.identity.x;
    identities[pixel] = truthSurface.identity;
    normals[pixel] = float4(
        normalize(inverseRotateVector(
            truthCamera.orientation,
            truthSurface.worldNormal
        )),
        1.0f
    );
    motion[pixel] = 0.0f;
    validity[pixel] =
        MR_VISUAL_VALIDITY_FRAME |
        MR_VISUAL_VALIDITY_GEOMETRY;
}

kernel void mr_hybrid_apply_sensor(
    const device MRWorldInstanceHeaderGPU* instances [[buffer(0)]],
    const device MRWorldSensorInstanceGPU* sensors [[buffer(1)]],
    device float4* rgb [[buffer(2)]],
    device float* depth [[buffer(3)]],
    device uint* validity [[buffer(4)]],
    constant MRHybridRenderUniformsGPU& uniforms [[buffer(5)]],
    const uint pixel [[thread_position_in_grid]]
) {
    const uint compactPixelCount =
        uniforms.counts.x *
        bandPixelCountPerEnvironment(uniforms);
    if (pixel >= compactPixelCount) {
        return;
    }
    const uint globalPixel =
        globalPixelFromBandIndex(pixel, 0u, uniforms);
    const uint pixelsPerEnvironment =
        uniforms.image.x * uniforms.image.y;
    const uint environment =
        globalPixel / pixelsPerEnvironment;
    const MRWorldInstanceHeaderGPU instance =
        instances[environment];
    const MRWorldSensorInstanceGPU sensor =
        sensors[instance.ranges.z + uniforms.render.x];
    const uint seed = frameSeed(instance, uniforms, globalPixel);

    rgb[globalPixel].xyz = max(
        rgb[globalPixel].xyz +
            sensor.noiseAndLatency.x *
                float3(
                    unitVarianceNoise(seed ^ 0x243f6a88u),
                    unitVarianceNoise(seed ^ 0x85a308d3u),
                    unitVarianceNoise(seed ^ 0x13198a2eu)
                ),
        0.0f
    );

    uint valid =
        validity[globalPixel] & ~MR_VISUAL_VALIDITY_DEPTH;
    if ((valid & MR_VISUAL_VALIDITY_GEOMETRY) != 0u) {
        float measuredDepth = depth[globalPixel];
        if (measureDepth(
                depth[globalPixel],
                sensor,
                seed,
                uniforms,
                measuredDepth
            )) {
            depth[globalPixel] = measuredDepth;
            valid |= MR_VISUAL_VALIDITY_DEPTH;
        } else {
            depth[globalPixel] = uniforms.clearColorAndDepth.w;
        }
    } else {
        depth[globalPixel] = uniforms.clearColorAndDepth.w;
    }
    validity[globalPixel] = valid;
}
