#pragma once

// Pointer-free visual-platform ABI shared by C++, Objective-C++, and Metal.
// Runtime object ownership, strings, and provider implementations remain in
// the host-side VisualPlatform IR.

#include "metalrobo/gpu_types.h"

#define MR_VISUAL_PLATFORM_ABI_VERSION 1u

enum MRVisualRepresentation : mr_u32 {
    MR_VISUAL_REPRESENTATION_NONE = 0u,
    MR_VISUAL_REPRESENTATION_TRIANGLE_MESH = 1u,
    MR_VISUAL_REPRESENTATION_GAUSSIAN_FIELD = 2u,
    MR_VISUAL_REPRESENTATION_PROCEDURAL = 3u,
};

enum MRVisualBindingKind : mr_u32 {
    MR_VISUAL_BINDING_WORLD = 0u,
    MR_VISUAL_BINDING_ASSET = 1u,
    MR_VISUAL_BINDING_RIGID_BODY = 2u,
    MR_VISUAL_BINDING_ARTICULATED_LINK = 3u,
};

enum MRVisualFrameSource : mr_u32 {
    MR_VISUAL_SOURCE_SIMULATION = 0u,
    MR_VISUAL_SOURCE_CAPTURE = 1u,
    MR_VISUAL_SOURCE_REPLAY = 2u,
};

enum MRVisualCoordinateFrame : mr_u32 {
    MR_VISUAL_FRAME_PIXEL = 0u,
    MR_VISUAL_FRAME_CAMERA = 1u,
    MR_VISUAL_FRAME_ROBOT_BASE = 2u,
    MR_VISUAL_FRAME_WORLD = 3u,
    MR_VISUAL_FRAME_OBJECT = 4u,
};

enum MRVisualStorageKind : mr_u32 {
    MR_VISUAL_STORAGE_HOST = 0u,
    MR_VISUAL_STORAGE_METAL_BUFFER = 1u,
    MR_VISUAL_STORAGE_MLX_ARRAY = 2u,
    MR_VISUAL_STORAGE_COREML_TENSOR = 3u,
};

enum MRVisualPixelFormat : mr_u32 {
    MR_VISUAL_FORMAT_UNKNOWN = 0u,
    MR_VISUAL_FORMAT_RGBA32_FLOAT = 1u,
    MR_VISUAL_FORMAT_R32_FLOAT = 2u,
    MR_VISUAL_FORMAT_RGBA32_UINT = 3u,
    MR_VISUAL_FORMAT_R32_UINT = 4u,
    MR_VISUAL_FORMAT_R8_UINT = 5u,
    MR_VISUAL_FORMAT_RG32_FLOAT = 6u,
};

enum MRVisualModality : mr_u32 {
    MR_VISUAL_MODALITY_RGB = 1u << 0u,
    MR_VISUAL_MODALITY_DEPTH = 1u << 1u,
    MR_VISUAL_MODALITY_DEPTH_VALIDITY = 1u << 2u,
    MR_VISUAL_MODALITY_NORMAL = 1u << 3u,
    MR_VISUAL_MODALITY_MOTION = 1u << 4u,
    MR_VISUAL_MODALITY_SEMANTIC = 1u << 5u,
    MR_VISUAL_MODALITY_INSTANCE = 1u << 6u,
    MR_VISUAL_MODALITY_LINK = 1u << 7u,
    MR_VISUAL_MODALITY_KEYPOINT = 1u << 8u,
    MR_VISUAL_MODALITY_FEATURE = 1u << 9u,
    MR_VISUAL_MODALITY_OBJECT_POSE = 1u << 10u,
};

enum MRPerceptionCapability : mr_u32 {
    MR_PERCEPTION_CAP_DENSE_DEPTH = 1u << 0u,
    MR_PERCEPTION_CAP_SEMANTIC = 1u << 1u,
    MR_PERCEPTION_CAP_INSTANCE = 1u << 2u,
    MR_PERCEPTION_CAP_TRACKING = 1u << 3u,
    MR_PERCEPTION_CAP_OBJECT_POSE = 1u << 4u,
    MR_PERCEPTION_CAP_KEYPOINT = 1u << 5u,
    MR_PERCEPTION_CAP_DENSE_FEATURE = 1u << 6u,
    MR_PERCEPTION_CAP_EMBEDDING = 1u << 7u,
};

typedef struct MR_ALIGN16 MRVisualSensorBindingGPU {
    // parent kind, parent body/asset, owning asset, flags.
    mr_uint4 identity;
    // nominal rate Hz, exposure seconds, shutter readout seconds, jitter sec.
    mr_float4 timing;
    // minimum range, maximum range, depth quantum, motion-blur scale.
    mr_float4 rangeAndResponse;
} MRVisualSensorBindingGPU;

typedef struct MR_ALIGN16 MRVisualMeshVertexGPU {
    // xyz position in the triangle binding frame, w homogeneous one.
    mr_float4 position;
    // xyz normal in the triangle binding frame, w texture u.
    mr_float4 normalAndU;
    // xyz tangent in the triangle binding frame, w texture v.
    mr_float4 tangentAndV;
} MRVisualMeshVertexGPU;

typedef struct MR_ALIGN16 MRVisualMeshTriangleGPU {
    // three vertex indices and material index.
    mr_uint4 verticesAndMaterial;
    // owning asset, body index, binding kind, flags.
    mr_uint4 binding;
    // semantic class, instance, link, stable primitive id.
    mr_uint4 identity;
    // Optional per-triangle overrides. Zero uses the material.
    mr_float4 colorAndOpacity;
} MRVisualMeshTriangleGPU;

typedef struct MR_ALIGN16 MRVisualMaterialGPU {
    // Linear RGB base color and opacity.
    mr_float4 baseColorAndOpacity;
    // Linear RGB emissive color and emissive strength.
    mr_float4 emissionAndStrength;
    // Roughness, metallic, normal scale, occlusion scale.
    mr_float4 surface;
    // Clearcoat, clearcoat roughness, specular scale, flags-as-float.
    mr_float4 coating;
} MRVisualMaterialGPU;

typedef struct MR_ALIGN16 MRVisualFrameMetadataGPU {
    // environments, views, width, height.
    mr_uint4 dimensions;
    // frame low/high, sensor sequence, source.
    mr_uint4 identity;
    // capture seconds, frame age seconds, exposure seconds, readout seconds.
    mr_float4 timing;
    // active modalities, coordinate convention, ABI version, flags.
    mr_uint4 contract;
} MRVisualFrameMetadataGPU;

typedef struct MR_ALIGN16 MRVisualKeypointGPU {
    // xyz position, w visibility in [0, 1].
    mr_float4 positionAndVisibility;
    // semantic, instance, link, keypoint id.
    mr_uint4 identity;
} MRVisualKeypointGPU;

typedef struct MR_ALIGN16 MRVisualPoseGPU {
    // xyz position in the declared frame, w homogeneous one.
    mr_float4 position;
    // Normalized xyzw orientation in the declared frame.
    mr_float4 orientation;
    // semantic, instance, link, flags.
    mr_uint4 identity;
} MRVisualPoseGPU;

typedef struct MR_ALIGN16 MRVisualContactAnnotationGPU {
    // xyz contact position in the declared frame, w normal impulse.
    mr_float4 positionAndImpulse;
    // xyz outward normal, w separation.
    mr_float4 normalAndSeparation;
    // first instance, second instance, contact id, flags.
    mr_uint4 identity;
} MRVisualContactAnnotationGPU;

#ifndef __METAL_VERSION__
#include <type_traits>

static_assert(std::is_trivially_copyable_v<MRVisualSensorBindingGPU>);
static_assert(std::is_trivially_copyable_v<MRVisualMeshVertexGPU>);
static_assert(std::is_trivially_copyable_v<MRVisualMeshTriangleGPU>);
static_assert(std::is_trivially_copyable_v<MRVisualMaterialGPU>);
static_assert(std::is_trivially_copyable_v<MRVisualFrameMetadataGPU>);
static_assert(std::is_trivially_copyable_v<MRVisualKeypointGPU>);
static_assert(std::is_trivially_copyable_v<MRVisualPoseGPU>);
static_assert(std::is_trivially_copyable_v<MRVisualContactAnnotationGPU>);
static_assert(sizeof(MRVisualSensorBindingGPU) == 48u);
static_assert(sizeof(MRVisualMeshVertexGPU) == 48u);
static_assert(sizeof(MRVisualMeshTriangleGPU) == 64u);
static_assert(sizeof(MRVisualMaterialGPU) == 64u);
static_assert(sizeof(MRVisualFrameMetadataGPU) == 64u);
static_assert(sizeof(MRVisualKeypointGPU) == 32u);
static_assert(sizeof(MRVisualPoseGPU) == 48u);
static_assert(sizeof(MRVisualContactAnnotationGPU) == 48u);
#endif
