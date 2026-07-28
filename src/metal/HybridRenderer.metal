#include <metal_stdlib>

#include "metalrobo/hybrid_renderer_types.h"
#include "metalrobo/world_compiler_types.h"

using namespace metal;

namespace {

float4 quaternionProduct(const float4 a, const float4 b) {
    return float4(a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                  a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                  a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
                  a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

float3 rotateVector(const float4 quaternion, const float3 vector) {
    const float4 vectorQuaternion(vector, 0.0f);
    const float4 conjugate(-quaternion.x, -quaternion.y, -quaternion.z,
                           quaternion.w);
    return quaternionProduct(quaternionProduct(quaternion, vectorQuaternion),
                             conjugate)
        .xyz;
}

float3 inverseRotateVector(const float4 quaternion, const float3 vector) {
    return rotateVector(
        float4(-quaternion.x, -quaternion.y, -quaternion.z, quaternion.w),
        vector);
}

float2 projectAxis(const float3 mean, const float3 axis, const float fx,
                   const float fy) {
    const float inverseDepthSquared = 1.0f / max(mean.z * mean.z, 1.0e-12f);
    return float2(
        fx * (axis.x * mean.z - mean.x * axis.z) * inverseDepthSquared,
        fy * (axis.y * mean.z - mean.y * axis.z) * inverseDepthSquared);
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
    return (float(randomHash(value)) + 0.5f) * (2.0f / 4294967296.0f) - 1.0f;
}

float3 applyAppearance(float3 color,
                       const MRWorldAppearanceInstanceGPU appearance) {
    const float exposure = exp2(appearance.colorAndLight.x);
    const float whiteBalance = max(appearance.colorAndLight.y, 1.0e-3f);
    color *= float3(whiteBalance, 1.0f, 1.0f / whiteBalance);
    const float luminance = dot(color, float3(0.2126f, 0.7152f, 0.0722f));
    color = mix(float3(luminance), color, appearance.colorAndLight.z);
    color = (color - 0.5f) * appearance.material.y + 0.5f;
    return max(color * exposure, 0.0f);
}

} // namespace

kernel void mr_hybrid_clear_tiles(device atomic_uint* tileCounts [[buffer(0)]],
                                  device atomic_uint* overflowCounts
                                  [[buffer(1)]],
                                  constant MRHybridRenderUniformsGPU& uniforms
                                  [[buffer(2)]],
                                  const uint index
                                  [[thread_position_in_grid]]) {
    const uint tileCount =
        uniforms.image.z * uniforms.image.w * uniforms.counts.x;
    if (index < tileCount) {
        atomic_store_explicit(tileCounts + index, 0u, memory_order_relaxed);
    }
    if (index < uniforms.counts.x) {
        atomic_store_explicit(overflowCounts + index, 0u, memory_order_relaxed);
    }
}

kernel void mr_hybrid_bin_gaussians(
    const device MRHybridGaussianGPU* gaussians [[buffer(0)]],
    const device MRWorldInstanceHeaderGPU* instances [[buffer(1)]],
    const device MRWorldAssetInstanceGPU* assets [[buffer(2)]],
    const device MRWorldSensorInstanceGPU* sensors [[buffer(3)]],
    device MRHybridProjectedGaussianGPU* projected [[buffer(4)]],
    device atomic_uint* tileCounts [[buffer(5)]],
    device uint* tileIndices [[buffer(6)]],
    device atomic_uint* overflowCounts [[buffer(7)]],
    constant MRHybridRenderUniformsGPU& uniforms [[buffer(8)]],
    const uint index [[thread_position_in_grid]]) {
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

    const uint parentAsset = instance.ranges.x + sensor.identity.x;
    const MRWorldAssetInstanceGPU cameraParent = assets[parentAsset];
    const float4 cameraOrientation = normalize(
        quaternionProduct(cameraParent.orientation, sensor.orientation));
    const float3 cameraPosition =
        cameraParent.positionAndScale.xyz +
        rotateVector(cameraParent.orientation,
                     sensor.positionAndFocalScale.xyz *
                         cameraParent.positionAndScale.w);

    float3 worldMean = gaussian.meanAndOpacity.xyz;
    float4 worldOrientation = normalize(gaussian.orientation);
    float worldScale = 1.0f;
    if (gaussian.binding.w == MR_HYBRID_GAUSSIAN_ASSET_LOCAL) {
        const uint assetIndex = instance.ranges.x + gaussian.binding.x;
        const MRWorldAssetInstanceGPU asset = assets[assetIndex];
        worldMean =
            asset.positionAndScale.xyz +
            rotateVector(asset.orientation, gaussian.meanAndOpacity.xyz *
                                                asset.positionAndScale.w);
        worldOrientation = normalize(
            quaternionProduct(asset.orientation, gaussian.orientation));
        worldScale = asset.positionAndScale.w;
    }

    const float3 cameraMean =
        inverseRotateVector(cameraOrientation, worldMean - cameraPosition);
    MRHybridProjectedGaussianGPU result{};
    result.centerDepthRadius = float4(0.0f, 0.0f, cameraMean.z, -1.0f);
    result.conicAndBounds = 0.0f;
    result.colorAndOpacity =
        float4(gaussian.colorAndEmission.xyz,
               clamp(gaussian.meanAndOpacity.w, 0.0f, 0.999f));
    result.identity = uint4(gaussian.binding.z, gaussianIndex, 0u, 0u);
    if (cameraMean.z <= 1.0e-4f) {
        projected[index] = result;
        return;
    }

    const float focalScale = sensor.positionAndFocalScale.w;
    const float fx = sensor.intrinsics.x * focalScale;
    const float fy = sensor.intrinsics.y * focalScale;
    float2 normalized = cameraMean.xy / cameraMean.z;
    const float radiusSquared = dot(normalized, normalized);
    const float radial = 1.0f + sensor.distortion.x * radiusSquared +
                         sensor.distortion.y * radiusSquared * radiusSquared;
    normalized =
        normalized * radial +
        float2(2.0f * sensor.distortion.z * normalized.x * normalized.y +
                   sensor.distortion.w *
                       (radiusSquared + 2.0f * normalized.x * normalized.x),
               sensor.distortion.z *
                       (radiusSquared + 2.0f * normalized.y * normalized.y) +
                   2.0f * sensor.distortion.w * normalized.x * normalized.y);
    const float2 center = normalized * float2(fx, fy) + sensor.intrinsics.zw;
    const float3 worldAxisX = rotateVector(
        worldOrientation,
        float3(gaussian.scaleAndImportance.x * worldScale, 0.0f, 0.0f));
    const float3 worldAxisY = rotateVector(
        worldOrientation,
        float3(0.0f, gaussian.scaleAndImportance.y * worldScale, 0.0f));
    const float3 worldAxisZ = rotateVector(
        worldOrientation,
        float3(0.0f, 0.0f, gaussian.scaleAndImportance.z * worldScale));
    const float2 derivativeX = projectAxis(
        cameraMean, inverseRotateVector(cameraOrientation, worldAxisX), fx, fy);
    const float2 derivativeY = projectAxis(
        cameraMean, inverseRotateVector(cameraOrientation, worldAxisY), fx, fy);
    const float2 derivativeZ = projectAxis(
        cameraMean, inverseRotateVector(cameraOrientation, worldAxisZ), fx, fy);
    const float covarianceXX = derivativeX.x * derivativeX.x +
                               derivativeY.x * derivativeY.x +
                               derivativeZ.x * derivativeZ.x + 0.25f;
    const float covarianceXY = derivativeX.x * derivativeX.y +
                               derivativeY.x * derivativeY.y +
                               derivativeZ.x * derivativeZ.y;
    const float covarianceYY = derivativeX.y * derivativeX.y +
                               derivativeY.y * derivativeY.y +
                               derivativeZ.y * derivativeZ.y + 0.25f;
    const float determinant = max(
        covarianceXX * covarianceYY - covarianceXY * covarianceXY, 1.0e-12f);
    const float inverseXX = covarianceYY / determinant;
    const float inverseXY = -covarianceXY / determinant;
    const float inverseYY = covarianceXX / determinant;
    const float trace = covarianceXX + covarianceYY;
    const float eigenDiscriminant =
        sqrt(max(0.0f, trace * trace - 4.0f * determinant));
    const float largestEigenvalue = 0.5f * (trace + eigenDiscriminant);
    const float pixelRadius =
        max(0.5f, 3.0f * sqrt(max(largestEigenvalue, 0.0f)));
    result.centerDepthRadius = float4(center, cameraMean.z, pixelRadius);
    result.conicAndBounds =
        float4(inverseXX, inverseXY, inverseYY, pixelRadius);
    projected[index] = result;

    const int minimumX = max(
        0, int(floor((center.x - pixelRadius) / float(MR_HYBRID_TILE_SIZE))));
    const int maximumX =
        min(int(uniforms.image.z) - 1,
            int(floor((center.x + pixelRadius) / float(MR_HYBRID_TILE_SIZE))));
    const int minimumY = max(
        0, int(floor((center.y - pixelRadius) / float(MR_HYBRID_TILE_SIZE))));
    const int maximumY =
        min(int(uniforms.image.w) - 1,
            int(floor((center.y + pixelRadius) / float(MR_HYBRID_TILE_SIZE))));
    if (minimumX > maximumX || minimumY > maximumY) {
        return;
    }

    const uint tilesPerEnvironment = uniforms.image.z * uniforms.image.w;
    const uint maximumPerTile = uniforms.render.y;
    for (int tileY = minimumY; tileY <= maximumY; ++tileY) {
        for (int tileX = minimumX; tileX <= maximumX; ++tileX) {
            const uint tile = environment * tilesPerEnvironment +
                              uint(tileY) * uniforms.image.z + uint(tileX);
            const uint slot = atomic_fetch_add_explicit(tileCounts + tile, 1u,
                                                        memory_order_relaxed);
            if (slot < maximumPerTile) {
                tileIndices[tile * maximumPerTile + slot] = index;
            } else {
                atomic_fetch_add_explicit(overflowCounts + environment, 1u,
                                          memory_order_relaxed);
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
    device float4* rgb [[buffer(6)]], device float* depth [[buffer(7)]],
    device uint* segmentation [[buffer(8)]],
    constant MRHybridRenderUniformsGPU& uniforms [[buffer(9)]],
    const uint tile [[threadgroup_position_in_grid]],
    const uint localIndex [[thread_position_in_threadgroup]]) {
    threadgroup uint sortedIndices[MR_HYBRID_MAX_GAUSSIANS_PER_TILE];
    threadgroup float sortedDepth[MR_HYBRID_MAX_GAUSSIANS_PER_TILE];

    const uint tilesPerEnvironment = uniforms.image.z * uniforms.image.w;
    const uint environment = tile / tilesPerEnvironment;
    if (environment >= uniforms.counts.x) {
        return;
    }
    const uint tileInEnvironment = tile - environment * tilesPerEnvironment;
    const uint count =
        min(atomic_load_explicit(tileCounts + tile, memory_order_relaxed),
            uniforms.render.y);
    const uint sourceIndex = tile * uniforms.render.y + localIndex;
    if (localIndex < count) {
        sortedIndices[localIndex] = tileIndices[sourceIndex];
        sortedDepth[localIndex] =
            projected[sortedIndices[localIndex]].centerDepthRadius.z;
    } else {
        sortedIndices[localIndex] = 0xffffffffu;
        sortedDepth[localIndex] = INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint width = 2u; width <= MR_HYBRID_MAX_GAUSSIANS_PER_TILE;
         width <<= 1u) {
        for (uint stride = width >> 1u; stride > 0u; stride >>= 1u) {
            const uint partner = localIndex ^ stride;
            if (partner > localIndex) {
                const bool ascending = (localIndex & width) == 0u;
                const bool swapValues =
                    ascending ? sortedDepth[localIndex] > sortedDepth[partner]
                              : sortedDepth[localIndex] < sortedDepth[partner];
                if (swapValues) {
                    const float savedDepth = sortedDepth[localIndex];
                    sortedDepth[localIndex] = sortedDepth[partner];
                    sortedDepth[partner] = savedDepth;
                    const uint savedIndex = sortedIndices[localIndex];
                    sortedIndices[localIndex] = sortedIndices[partner];
                    sortedIndices[partner] = savedIndex;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const uint tileX = tileInEnvironment % uniforms.image.z;
    const uint tileY = tileInEnvironment / uniforms.image.z;
    const uint pixelX =
        tileX * MR_HYBRID_TILE_SIZE + localIndex % MR_HYBRID_TILE_SIZE;
    const uint pixelY =
        tileY * MR_HYBRID_TILE_SIZE + localIndex / MR_HYBRID_TILE_SIZE;
    if (pixelX >= uniforms.image.x || pixelY >= uniforms.image.y) {
        return;
    }

    float3 accumulatedColor = 0.0f;
    float transmittance = 1.0f;
    float nearestDepth = uniforms.clearColorAndDepth.w;
    uint semantic = 0xffffffffu;
    for (uint slot = 0u; slot < count; ++slot) {
        const uint projectedIndex = sortedIndices[slot];
        if (projectedIndex == 0xffffffffu) {
            continue;
        }
        const MRHybridProjectedGaussianGPU gaussian = projected[projectedIndex];
        const float2 delta = (float2(float(pixelX), float(pixelY)) + 0.5f) -
                             gaussian.centerDepthRadius.xy;
        const float mahalanobis =
            gaussian.conicAndBounds.x * delta.x * delta.x +
            2.0f * gaussian.conicAndBounds.y * delta.x * delta.y +
            gaussian.conicAndBounds.z * delta.y * delta.y;
        if (mahalanobis > 9.0f) {
            continue;
        }
        const float alpha =
            min(0.999f, gaussian.colorAndOpacity.w * exp(-0.5f * mahalanobis));
        if (alpha <= 1.0e-4f) {
            continue;
        }
        const float contribution = transmittance * alpha;
        accumulatedColor += contribution * gaussian.colorAndOpacity.xyz;
        if (semantic == 0xffffffffu && contribution > 1.0e-3f) {
            semantic = gaussian.identity.x;
            nearestDepth = gaussian.centerDepthRadius.z;
        }
        transmittance *= 1.0f - alpha;
        if (transmittance <= 1.0e-3f) {
            break;
        }
    }
    accumulatedColor += transmittance * uniforms.clearColorAndDepth.xyz;

    const MRWorldInstanceHeaderGPU instance = instances[environment];
    const MRWorldSensorInstanceGPU sensor =
        sensors[instance.ranges.z + uniforms.render.x];
    const MRWorldAppearanceInstanceGPU appearance =
        appearances[instance.program.x];
    accumulatedColor = applyAppearance(accumulatedColor, appearance);
    const uint pixel = environment * uniforms.image.x * uniforms.image.y +
                       pixelY * uniforms.image.x + pixelX;
    const uint scenarioSeed =
        instance.identity.x ^ instance.identity.y ^ pixel * 0x9e3779b9u;
    const float colorNoise =
        sensor.noiseAndLatency.x * uniformSigned(scenarioSeed);
    accumulatedColor = max(accumulatedColor + colorNoise, 0.0f);
    if (nearestDepth < uniforms.clearColorAndDepth.w) {
        nearestDepth = max(
            0.0f, nearestDepth + sensor.noiseAndLatency.y *
                                     uniformSigned(scenarioSeed ^ 0xa511e9b3u));
        const float dropout =
            (float(randomHash(scenarioSeed ^ 0x63d83595u)) + 0.5f) /
            4294967296.0f;
        if (dropout < sensor.noiseAndLatency.z) {
            nearestDepth = uniforms.clearColorAndDepth.w;
            semantic = 0xffffffffu;
        }
    }
    rgb[pixel] = float4(accumulatedColor, 1.0f - transmittance);
    depth[pixel] = nearestDepth;
    segmentation[pixel] = semantic;
}
