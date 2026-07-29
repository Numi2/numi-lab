#include <metal_stdlib>

#include "metalrobo/engine_types.h"
#include "metalrobo/hybrid_renderer_types.h"
#include "metalrobo/world_compiler_types.h"

using namespace metal;

namespace {

constant uint kLiveCurrent = 1u << 0u;
constant uint kLivePrevious = 1u << 1u;

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
    const MRBodyStateGPU state =
        bodies[environment * uniforms.live.x + body];
    return {
        state.position.xyz,
        normalize(state.orientation),
        1.0f,
        all(isfinite(state.position.xyz)) &&
            all(isfinite(state.orientation)),
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
                uniformSigned(seed ^ 0xa511e9b3u)
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

} // namespace

kernel void mr_hybrid_clear_tiles(
    device atomic_uint* tileCounts [[buffer(0)]],
    device atomic_uint* overflowCounts [[buffer(1)]],
    constant MRHybridRenderUniformsGPU& uniforms [[buffer(2)]],
    const uint index [[thread_position_in_grid]]
) {
    const uint tileCount =
        uniforms.image.z * uniforms.image.w * uniforms.counts.x;
    if (index < tileCount) {
        atomic_store_explicit(
            tileCounts + index,
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

kernel void mr_hybrid_clear_mesh_winners(
    device atomic_uint* winners [[buffer(0)]],
    constant MRHybridRenderUniformsGPU& uniforms [[buffer(1)]],
    const uint pixel [[thread_position_in_grid]]
) {
    const uint count =
        uniforms.counts.x * uniforms.image.x * uniforms.image.y;
    if (pixel >= count) {
        return;
    }
    atomic_store_explicit(
        winners + pixel,
        0x7f800000u,
        memory_order_relaxed
    );
    atomic_store_explicit(
        winners + count + pixel,
        0xffffffffu,
        memory_order_relaxed
    );
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

    const int minimumX = max(
        0,
        int(floor(
            (result.centerDepthRadius.x - pixelRadius) /
            float(MR_HYBRID_TILE_SIZE)
        ))
    );
    const int maximumX = min(
        int(uniforms.image.z) - 1,
        int(floor(
            (result.centerDepthRadius.x + pixelRadius) /
            float(MR_HYBRID_TILE_SIZE)
        ))
    );
    const int minimumY = max(
        0,
        int(floor(
            (result.centerDepthRadius.y - pixelRadius) /
            float(MR_HYBRID_TILE_SIZE)
        ))
    );
    const int maximumY = min(
        int(uniforms.image.w) - 1,
        int(floor(
            (result.centerDepthRadius.y + pixelRadius) /
            float(MR_HYBRID_TILE_SIZE)
        ))
    );
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
    const uint tile [[threadgroup_position_in_grid]],
    const uint localIndex [[thread_position_in_threadgroup]]
) {
    threadgroup uint
        sortedIndices[MR_HYBRID_MAX_GAUSSIANS_PER_TILE];
    threadgroup float
        sortedDepth[MR_HYBRID_MAX_GAUSSIANS_PER_TILE];

    const uint tilesPerEnvironment =
        uniforms.image.z * uniforms.image.w;
    const uint environment = tile / tilesPerEnvironment;
    if (environment >= uniforms.counts.x) {
        return;
    }
    const uint tileInEnvironment =
        tile - environment * tilesPerEnvironment;
    const uint count = min(
        atomic_load_explicit(
            tileCounts + tile,
            memory_order_relaxed
        ),
        uniforms.render.y
    );
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

    for (uint width = 2u;
         width <= MR_HYBRID_MAX_GAUSSIANS_PER_TILE;
         width <<= 1u) {
        for (uint stride = width >> 1u;
             stride > 0u;
             stride >>= 1u) {
            const uint partner = localIndex ^ stride;
            if (partner > localIndex) {
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

    const uint tileX = tileInEnvironment % uniforms.image.z;
    const uint tileY = tileInEnvironment / uniforms.image.z;
    const uint pixelX =
        tileX * MR_HYBRID_TILE_SIZE +
        localIndex % MR_HYBRID_TILE_SIZE;
    const uint pixelY =
        tileY * MR_HYBRID_TILE_SIZE +
        localIndex / MR_HYBRID_TILE_SIZE;
    if (pixelX >= uniforms.image.x ||
        pixelY >= uniforms.image.y) {
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
    const MRWorldSensorInstanceGPU sensor =
        sensors[instance.ranges.z + uniforms.render.x];
    const MRWorldAppearanceInstanceGPU appearance =
        appearances[instance.program.x];
    accumulatedColor =
        applyAppearance(accumulatedColor, appearance);
    const uint pixel =
        environment * uniforms.image.x * uniforms.image.y +
        pixelY * uniforms.image.x + pixelX;
    const uint seed = frameSeed(instance, uniforms, pixel);
    accumulatedColor = max(
        accumulatedColor +
            sensor.noiseAndLatency.x * uniformSigned(seed),
        0.0f
    );
    uint valid = 1u;
    if (nearestDepth < uniforms.clearColorAndDepth.w) {
        float measuredDepth = nearestDepth;
        if (!measureDepth(
                nearestDepth,
                sensor,
                seed,
                uniforms,
                measuredDepth
            )) {
            nearestDepth = measuredDepth;
            identity = uint4(MR_INVALID_INDEX);
            normal = 0.0f;
            pixelMotion = 0.0f;
        } else {
            nearestDepth = measuredDepth;
            valid |= 2u | 4u;
        }
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
    const device MRVisualMeshVertexGPU* vertices,
    const device MRVisualMeshTriangleGPU* triangles,
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
    const MRVisualMeshTriangleGPU triangle =
        triangles[triangleIndex];
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
    const BoundPose binding = resolvePose(
        triangle.binding.z,
        triangle.binding.y,
        triangle.binding.x,
        environment,
        instance,
        assets,
        currentBodies,
        uniforms,
        false
    );
    if (!camera.valid || !binding.valid) {
        return;
    }

    const uint3 indices = triangle.verticesAndMaterial.xyz;
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
    const int minimumX = max(
        0,
        int(floor(min(
            p0.pixel.x,
            min(p1.pixel.x, p2.pixel.x)
        )))
    );
    const int maximumX = min(
        int(uniforms.image.x) - 1,
        int(ceil(max(
            p0.pixel.x,
            max(p1.pixel.x, p2.pixel.x)
        )))
    );
    const int minimumY = max(
        0,
        int(floor(min(
            p0.pixel.y,
            min(p1.pixel.y, p2.pixel.y)
        )))
    );
    const int maximumY = min(
        int(uniforms.image.y) - 1,
        int(ceil(max(
            p0.pixel.y,
            max(p1.pixel.y, p2.pixel.y)
        )))
    );
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
    const device MRVisualMeshVertexGPU* vertices [[buffer(0)]],
    const device MRVisualMeshTriangleGPU* triangles [[buffer(1)]],
    const device MRWorldInstanceHeaderGPU* instances [[buffer(2)]],
    const device MRWorldAssetInstanceGPU* assets [[buffer(3)]],
    const device MRWorldSensorInstanceGPU* sensors [[buffer(4)]],
    const device MRVisualSensorBindingGPU* sensorBindings [[buffer(5)]],
    const device MRBodyStateGPU* currentBodies [[buffer(6)]],
    device atomic_uint* winners [[buffer(7)]],
    constant MRHybridRenderUniformsGPU& uniforms [[buffer(8)]],
    const uint index [[thread_position_in_grid]]
) {
    rasterizeMesh(
        vertices,
        triangles,
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
    const device MRVisualMeshVertexGPU* vertices [[buffer(0)]],
    const device MRVisualMeshTriangleGPU* triangles [[buffer(1)]],
    const device MRWorldInstanceHeaderGPU* instances [[buffer(2)]],
    const device MRWorldAssetInstanceGPU* assets [[buffer(3)]],
    const device MRWorldSensorInstanceGPU* sensors [[buffer(4)]],
    const device MRVisualSensorBindingGPU* sensorBindings [[buffer(5)]],
    const device MRBodyStateGPU* currentBodies [[buffer(6)]],
    device atomic_uint* winners [[buffer(7)]],
    constant MRHybridRenderUniformsGPU& uniforms [[buffer(8)]],
    const uint index [[thread_position_in_grid]]
) {
    rasterizeMesh(
        vertices,
        triangles,
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
    const device MRVisualMeshVertexGPU* vertices [[buffer(0)]],
    const device MRVisualMeshTriangleGPU* triangles [[buffer(1)]],
    const device MRVisualMaterialGPU* materials [[buffer(2)]],
    const device MRWorldInstanceHeaderGPU* instances [[buffer(3)]],
    const device MRWorldAssetInstanceGPU* assets [[buffer(4)]],
    const device MRWorldSensorInstanceGPU* sensors [[buffer(5)]],
    const device MRWorldAppearanceInstanceGPU* appearances [[buffer(6)]],
    const device MRVisualSensorBindingGPU* sensorBindings [[buffer(7)]],
    const device MRBodyStateGPU* currentBodies [[buffer(8)]],
    const device MRBodyStateGPU* previousBodies [[buffer(9)]],
    const device atomic_uint* winners [[buffer(10)]],
    device float4* rgb [[buffer(11)]],
    device float* depth [[buffer(12)]],
    device uint* segmentation [[buffer(13)]],
    device uint4* identities [[buffer(14)]],
    device float4* normals [[buffer(15)]],
    device float4* motion [[buffer(16)]],
    device uint* validity [[buffer(17)]],
    constant MRHybridRenderUniformsGPU& uniforms [[buffer(18)]],
    const uint pixel [[thread_position_in_grid]]
) {
    const uint pixelCount =
        uniforms.counts.x * uniforms.image.x * uniforms.image.y;
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
    const uint pixelsPerEnvironment =
        uniforms.image.x * uniforms.image.y;
    const uint environment = pixel / pixelsPerEnvironment;
    const uint localPixel =
        pixel - environment * pixelsPerEnvironment;
    const uint pixelX = localPixel % uniforms.image.x;
    const uint pixelY = localPixel / uniforms.image.x;
    const MRVisualMeshTriangleGPU triangle =
        triangles[triangleIndex];
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
    const BoundPose binding = resolvePose(
        triangle.binding.z,
        triangle.binding.y,
        triangle.binding.x,
        environment,
        instance,
        assets,
        currentBodies,
        uniforms,
        false
    );
    const BoundPose previousBinding = resolvePose(
        triangle.binding.z,
        triangle.binding.y,
        triangle.binding.x,
        environment,
        instance,
        assets,
        previousBodies,
        uniforms,
        true
    );
    if (!camera.valid || !binding.valid) {
        return;
    }

    const uint3 vertexIndices =
        triangle.verticesAndMaterial.xyz;
    const MRVisualMeshVertexGPU vertex0 =
        vertices[vertexIndices.x];
    const MRVisualMeshVertexGPU vertex1 =
        vertices[vertexIndices.y];
    const MRVisualMeshVertexGPU vertex2 =
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
    const float3 localNormal = normalize(
        corrected.x * vertex0.normalAndU.xyz +
        corrected.y * vertex1.normalAndU.xyz +
        corrected.z * vertex2.normalAndU.xyz
    );
    const float3 worldNormal =
        transformNormal(binding, localNormal);
    const float3 cameraNormal = normalize(
        inverseRotateVector(camera.orientation, worldNormal)
    );

    const MRVisualMaterialGPU material =
        materials[triangle.verticesAndMaterial.w];
    float4 base = material.baseColorAndOpacity;
    if (triangle.colorAndOpacity.w > 0.0f) {
        base = triangle.colorAndOpacity;
    }
    const float3 lightDirection =
        normalize(float3(-0.35f, -0.55f, 1.0f));
    const float diffuse =
        0.24f + 0.76f * max(dot(worldNormal, lightDirection), 0.0f);
    const MRWorldAppearanceInstanceGPU appearance =
        appearances[instance.program.x];
    float3 color =
        base.xyz * diffuse * appearance.colorAndLight.w +
        material.emissionAndStrength.xyz *
            material.emissionAndStrength.w;
    color = applyAppearance(color, appearance);
    const uint seed = frameSeed(instance, uniforms, pixel);
    color = max(
        color +
            sensor.noiseAndLatency.x * uniformSigned(seed),
        0.0f
    );
    float publishedDepth = meshDepth;
    if (!measureDepth(
            meshDepth,
            sensor,
            seed,
            uniforms,
            publishedDepth
        )) {
        validity[pixel] = 1u;
        identities[pixel] = uint4(MR_INVALID_INDEX);
        segmentation[pixel] = MR_INVALID_INDEX;
        normals[pixel] = 0.0f;
        motion[pixel] = 0.0f;
        rgb[pixel] = float4(color, base.w);
        depth[pixel] = publishedDepth;
        return;
    }

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
    rgb[pixel] = float4(color, base.w);
    depth[pixel] = publishedDepth;
    segmentation[pixel] = triangle.identity.x;
    identities[pixel] = triangle.identity;
    normals[pixel] = float4(cameraNormal, 1.0f);
    motion[pixel] = float4(pixelMotion, 1.0f, 0.0f);
    validity[pixel] = 1u | 2u | 4u;
}
