#pragma once

#include "metalrobo/gpu_types.h"
#include "metalrobo/visual_platform_types.h"

#define MR_HYBRID_RENDERER_ABI_VERSION 8u
#define MR_HYBRID_TILE_SIZE 16u
#define MR_HYBRID_MAX_GAUSSIANS_PER_TILE 256u
#define MR_HYBRID_MAX_MESH_TRIANGLES_PER_TILE 512u
#define MR_HYBRID_MESH_TILE_BATCH 128u
#define MR_HYBRID_MESH_MICRO_TRIANGLE_PIXELS 1024u
#define MR_HYBRID_MAX_NEAR_CLIPPED_TRIANGLES 4096u
#define MR_HYBRID_NEAR_CLIPPED_RESOLVE_THREADS 256u

enum {
    MR_HYBRID_OUTPUT_SEGMENTATION = 1u << 0u,
    MR_HYBRID_OUTPUT_IDENTITIES = 1u << 1u,
    MR_HYBRID_OUTPUT_NORMALS = 1u << 2u,
    MR_HYBRID_OUTPUT_MOTION = 1u << 3u,
    MR_HYBRID_OUTPUT_ALL_TRUTH =
        MR_HYBRID_OUTPUT_SEGMENTATION |
        MR_HYBRID_OUTPUT_IDENTITIES |
        MR_HYBRID_OUTPUT_NORMALS |
        MR_HYBRID_OUTPUT_MOTION,
};

enum {
    MR_HYBRID_GAUSSIAN_ASSET_LOCAL = 0u,
    MR_HYBRID_GAUSSIAN_BODY_LOCAL = 1u,
    MR_HYBRID_GAUSSIAN_WORLD = 2u,
};

typedef struct MR_ALIGN16 MRHybridGaussianGPU {
    // xyz mean in the binding frame, w base opacity.
    mr_float4 meanAndOpacity;
    // xyz standard deviation in metres, w LOD importance.
    mr_float4 scaleAndImportance;
    // xyzw Gaussian orientation in the binding frame.
    mr_float4 orientation;
    // linear RGB and optional emissive scale.
    mr_float4 colorAndEmission;
    // asset index, body index, semantic label, binding mode.
    mr_uint4 binding;
} MRHybridGaussianGPU;

typedef struct MR_ALIGN16 MRHybridProjectedGaussianGPU {
    // pixel center x/y, camera-space depth, three-sigma pixel radius.
    mr_float4 centerDepthRadius;
    // inverse screen covariance xx, xy, yy, and three-sigma bound.
    mr_float4 conicAndBounds;
    // linear RGB and opacity.
    mr_float4 colorAndOpacity;
    // semantic label, instance id, link id or invalid, source Gaussian.
    mr_uint4 identity;
    // xyz camera-space normal, w geometric validity.
    mr_float4 normalAndValidity;
    // xy previous-to-current pixel motion, z visibility, w reserved.
    mr_float4 motionAndVisibility;
} MRHybridProjectedGaussianGPU;

typedef struct MR_ALIGN16 MRHybridCameraStateGPU {
    // xyz current world position, w validity.
    mr_float4 currentPositionAndValidity;
    // xyzw current world orientation.
    mr_float4 currentOrientation;
    // xyz previous world position, w validity.
    mr_float4 previousPositionAndValidity;
    // xyzw previous world orientation.
    mr_float4 previousOrientation;
} MRHybridCameraStateGPU;

typedef struct MR_ALIGN16 MRHybridVisualInstanceStateGPU {
    // xyz current world position, w uniform scale or a negative invalid mark.
    mr_float4 currentPositionAndScale;
    mr_float4 currentOrientation;
    // xyz previous world position, w uniform scale or a negative invalid mark.
    mr_float4 previousPositionAndScale;
    mr_float4 previousOrientation;
} MRHybridVisualInstanceStateGPU;

typedef struct MR_ALIGN16 MRHybridNearClippedTriangleGPU {
    mr_float4 worldVertex0;
    mr_float4 worldVertex1;
    mr_float4 worldVertex2;
    // Source triangle index and reserved.
    mr_uint4 identity;
} MRHybridNearClippedTriangleGPU;

typedef struct MR_ALIGN16 MRHybridMeshTileRecordGPU {
    // p0.xy, p1.xy.
    mr_float4 projected01;
    // p2.xy, inverse depth 0, inverse depth 1.
    mr_float4 projected2AndInverse01;
    // inverse depth 2, signed area, triangle bits, reserved.
    mr_float4 inverse2AreaAndTriangle;
} MRHybridMeshTileRecordGPU;

typedef struct MR_ALIGN16 MRHybridRenderUniformsGPU {
    // active environments, Gaussian count, assets/world, sensors/world.
    mr_uint4 counts;
    // image width/height and tile grid width/height.
    mr_uint4 image;
    // selected camera, max Gaussians/tile, tile size, ABI version.
    mr_uint4 render;
    // body count, live-state flags, sensor-binding count, triangle count.
    mr_uint4 live;
    // frame low/high, sensor sequence, source.
    mr_uint4 timing;
    // clear RGB and clear depth.
    mr_float4 clearColorAndDepth;
    // nominal rate, exposure, shutter readout, deterministic jitter.
    mr_float4 sensorTiming;
    // minimum/maximum depth, depth quantum, motion-blur scale.
    mr_float4 sensorRangeAndResponse;
    // texture count, light count, renderer profile, profile flags.
    mr_uint4 presentation;
    // Mesh triangles/tile, cooperative batch, microtriangle pixels, and the
    // statically selected optional truth-output mask.
    mr_uint4 meshTiling;
    // shutter model, scan direction, temporal sample, sample count.
    mr_uint4 shutter;
    // temporal sample fraction, truth fraction, environment intensity/rotation.
    mr_float4 exposure;
    // shadow atlas width/height, PCF radius, active shadow light.
    mr_uint4 shadow;
    // first environment, environments in batch, atlas layer capacity, reserved.
    mr_uint4 shadowBatch;
    // first row/column, band size, band axis, truth-only pass.
    mr_uint4 band;
    // visible ray instances, motion keyframes, area samples, visual instances.
    mr_uint4 ray;
    // full shutter window, truth time fraction, sample weight, reserved.
    mr_float4 rayTiming;
} MRHybridRenderUniformsGPU;

#ifndef __METAL_VERSION__
#include <cstddef>
#include <type_traits>

static_assert(std::is_trivially_copyable_v<MRHybridGaussianGPU>);
static_assert(std::is_trivially_copyable_v<MRHybridProjectedGaussianGPU>);
static_assert(std::is_trivially_copyable_v<MRHybridCameraStateGPU>);
static_assert(
    std::is_trivially_copyable_v<MRHybridVisualInstanceStateGPU>
);
static_assert(
    std::is_trivially_copyable_v<MRHybridNearClippedTriangleGPU>
);
static_assert(
    std::is_trivially_copyable_v<MRHybridMeshTileRecordGPU>
);
static_assert(std::is_trivially_copyable_v<MRHybridRenderUniformsGPU>);
static_assert(sizeof(MRHybridGaussianGPU) == 80u);
static_assert(sizeof(MRHybridProjectedGaussianGPU) == 96u);
static_assert(sizeof(MRHybridCameraStateGPU) == 64u);
static_assert(sizeof(MRHybridVisualInstanceStateGPU) == 64u);
static_assert(sizeof(MRHybridNearClippedTriangleGPU) == 64u);
static_assert(sizeof(MRHybridMeshTileRecordGPU) == 48u);
static_assert(sizeof(MRHybridRenderUniformsGPU) == 272u);
#endif
