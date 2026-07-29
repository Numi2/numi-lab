#pragma once

// Pointer-free visual-platform ABI shared by C++, Objective-C++, and Metal.
// Runtime object ownership, strings, and provider implementations remain in
// the host-side VisualPlatform IR.

#include "metalrobo/gpu_types.h"

#define MR_VISUAL_PLATFORM_ABI_VERSION 1u
#define MR_VISUAL_PRESENTATION_ABI_VERSION 2u

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

enum MRVisualRendererProfileKind : mr_u32 {
    MR_VISUAL_RENDERER_SENSOR_FAST = 0u,
    MR_VISUAL_RENDERER_SENSOR_REFERENCE = 1u,
};

enum MRVisualShutterModel : mr_u32 {
    MR_VISUAL_SHUTTER_GLOBAL = 0u,
    MR_VISUAL_SHUTTER_ROLLING = 1u,
};

enum MRVisualShutterDirection : mr_u32 {
    MR_VISUAL_SHUTTER_TOP_TO_BOTTOM = 0u,
    MR_VISUAL_SHUTTER_BOTTOM_TO_TOP = 1u,
    MR_VISUAL_SHUTTER_LEFT_TO_RIGHT = 2u,
    MR_VISUAL_SHUTTER_RIGHT_TO_LEFT = 3u,
};

enum MRVisualAlphaMode : mr_u32 {
    MR_VISUAL_ALPHA_OPAQUE = 0u,
    MR_VISUAL_ALPHA_MASK = 1u,
    MR_VISUAL_ALPHA_BLEND = 2u,
};

enum MRVisualLightKind : mr_u32 {
    MR_VISUAL_LIGHT_DIRECTIONAL = 0u,
    MR_VISUAL_LIGHT_POINT = 1u,
    MR_VISUAL_LIGHT_SPOT = 2u,
    MR_VISUAL_LIGHT_RECTANGLE = 3u,
};

enum MRVisualLightUnit : mr_u32 {
    MR_VISUAL_LIGHT_UNIT_LUX = 0u,
    MR_VISUAL_LIGHT_UNIT_CANDELA = 1u,
    MR_VISUAL_LIGHT_UNIT_LUMEN = 2u,
    MR_VISUAL_LIGHT_UNIT_NIT = 3u,
};

enum MRVisualTextureFlag : mr_u32 {
    MR_VISUAL_TEXTURE_SRGB = 1u << 0u,
    MR_VISUAL_TEXTURE_CLAMP_U = 1u << 1u,
    MR_VISUAL_TEXTURE_CLAMP_V = 1u << 2u,
};

enum MRVisualInstanceFlag : mr_u32 {
    MR_VISUAL_INSTANCE_CASTS_SHADOW = 1u << 0u,
    MR_VISUAL_INSTANCE_RECEIVES_SHADOW = 1u << 1u,
    MR_VISUAL_INSTANCE_GAUSSIAN_RECEIVER_PROXY = 1u << 2u,
    MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR = 1u << 3u,
};

enum MRVisualMaterialFlag : mr_u32 {
    MR_VISUAL_MATERIAL_DOUBLE_SIDED = 1u << 0u,
    MR_VISUAL_MATERIAL_UNLIT = 1u << 1u,
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

enum MRVisualValidityFlag : mr_u32 {
    // A camera sample was produced for this pixel.
    MR_VISUAL_VALIDITY_FRAME = 1u << 0u,
    // The deployable depth sensor returned a usable measurement.
    MR_VISUAL_VALIDITY_DEPTH = 1u << 1u,
    // Rendered geometry exists, independently of sensor dropout/noise.
    MR_VISUAL_VALIDITY_GEOMETRY = 1u << 2u,
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
    // shutter model, scan direction, reserved, reserved.
    mr_uint4 shutter;
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

// Indexed authored-asset ABI. Asset packs retain these records verbatim;
// the renderer may build profile-specific visibility records at compile time.
typedef struct MR_ALIGN16 MRVisualVertexGPUV2 {
    // xyz position in the primitive's local frame, w homogeneous one.
    mr_float4 position;
    // xyz unit normal, w tangent handedness (+1 or -1).
    mr_float4 normalAndTangentSign;
    // xyz unit tangent, w reserved.
    mr_float4 tangent;
    // UV0.xy and UV1.xy.
    mr_float4 texcoord01;
    // Linear RGBA vertex color.
    mr_float4 color;
} MRVisualVertexGPUV2;

typedef struct MR_ALIGN16 MRVisualPrimitiveGPUV2 {
    // first index, index count, material slot, instance index.
    mr_uint4 geometry;
    // semantic class, instance, link, stable primitive id.
    mr_uint4 identity;
    // Local axis-aligned bounds.
    mr_float4 boundsMinimum;
    mr_float4 boundsMaximum;
} MRVisualPrimitiveGPUV2;

typedef struct MR_ALIGN16 MRVisualTriangleGPUV2 {
    // Three shared vertex indices and the owning primitive index.
    mr_uint4 verticesAndPrimitive;
} MRVisualTriangleGPUV2;

typedef struct MR_ALIGN16 MRVisualInstanceGPUV2 {
    // xyz local translation, w uniform scale.
    mr_float4 translationAndScale;
    // Normalized xyzw local orientation.
    mr_float4 orientation;
    // owning asset, body index, binding kind, instance flags.
    mr_uint4 binding;
    // semantic class, instance, link, stable instance id.
    mr_uint4 identity;
    // first primitive, primitive count, material variant, reserved.
    mr_uint4 geometry;
} MRVisualInstanceGPUV2;

typedef struct MR_ALIGN16 MRVisualMaterialGPUV2 {
    // glTF base-color factor in linear RGB and opacity.
    mr_float4 baseColorAndOpacity;
    // Linear emissive factor and emissive strength.
    mr_float4 emissionAndStrength;
    // Perceptual roughness, metallic, normal scale, occlusion strength.
    mr_float4 surface;
    // Clearcoat, clearcoat roughness, specular scale, alpha cutoff.
    mr_float4 coatingAndAlphaCutoff;
    // Base color, metallic-roughness, normal, occlusion texture indices.
    mr_uint4 textureIndices0;
    // Emissive, clearcoat, clearcoat-roughness, reserved texture indices.
    mr_uint4 textureIndices1;
    // alpha mode, material flags, UV set mask, stable material id.
    mr_uint4 flags;
    mr_uint4 reserved;
} MRVisualMaterialGPUV2;

typedef struct MR_ALIGN16 MRVisualTextureGPUV1 {
    // Width, height, mip count, texture flags.
    mr_uint4 dimensions;
    // First texel for mip 0, total texels, stable texture id, reserved.
    mr_uint4 storage;
    // First texel offsets for mips 1 through 4, or MR_INVALID_INDEX.
    mr_uint4 mipOffsets0;
    // First texel offsets for mips 5 through 8, or MR_INVALID_INDEX.
    mr_uint4 mipOffsets1;
} MRVisualTextureGPUV1;

typedef struct MR_ALIGN16 MRVisualLightGPUV1 {
    // xyz position in metres, w effective range.
    mr_float4 positionAndRange;
    // xyz direction from the light, w outer spot cosine.
    mr_float4 directionAndSpot;
    // Linear RGB radiometric color, w authored physical intensity.
    mr_float4 colorAndIntensity;
    // Rectangle width/height, inner spot cosine, source radius.
    mr_float4 shape;
    // kind, physical unit, authored priority, stable light id.
    mr_uint4 identity;
    // casts shadow, shadow map layer, sample count, flags.
    mr_uint4 shadow;
} MRVisualLightGPUV1;

typedef struct MR_ALIGN16 MRVisualShutterProfileGPUV1 {
    // model, scan direction, temporal samples, rolling bands.
    mr_uint4 mode;
    // exposure seconds, readout seconds, exposure center, reserved.
    mr_float4 timing;
} MRVisualShutterProfileGPUV1;

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
static_assert(std::is_trivially_copyable_v<MRVisualVertexGPUV2>);
static_assert(std::is_trivially_copyable_v<MRVisualPrimitiveGPUV2>);
static_assert(std::is_trivially_copyable_v<MRVisualTriangleGPUV2>);
static_assert(std::is_trivially_copyable_v<MRVisualInstanceGPUV2>);
static_assert(std::is_trivially_copyable_v<MRVisualMaterialGPUV2>);
static_assert(std::is_trivially_copyable_v<MRVisualTextureGPUV1>);
static_assert(std::is_trivially_copyable_v<MRVisualLightGPUV1>);
static_assert(std::is_trivially_copyable_v<MRVisualShutterProfileGPUV1>);
static_assert(std::is_trivially_copyable_v<MRVisualFrameMetadataGPU>);
static_assert(std::is_trivially_copyable_v<MRVisualKeypointGPU>);
static_assert(std::is_trivially_copyable_v<MRVisualPoseGPU>);
static_assert(std::is_trivially_copyable_v<MRVisualContactAnnotationGPU>);
static_assert(sizeof(MRVisualSensorBindingGPU) == 64u);
static_assert(sizeof(MRVisualMeshVertexGPU) == 48u);
static_assert(sizeof(MRVisualMeshTriangleGPU) == 64u);
static_assert(sizeof(MRVisualMaterialGPU) == 64u);
static_assert(sizeof(MRVisualVertexGPUV2) == 80u);
static_assert(sizeof(MRVisualPrimitiveGPUV2) == 64u);
static_assert(sizeof(MRVisualTriangleGPUV2) == 16u);
static_assert(sizeof(MRVisualInstanceGPUV2) == 80u);
static_assert(sizeof(MRVisualMaterialGPUV2) == 128u);
static_assert(sizeof(MRVisualTextureGPUV1) == 64u);
static_assert(sizeof(MRVisualLightGPUV1) == 96u);
static_assert(sizeof(MRVisualShutterProfileGPUV1) == 32u);
static_assert(sizeof(MRVisualFrameMetadataGPU) == 64u);
static_assert(sizeof(MRVisualKeypointGPU) == 32u);
static_assert(sizeof(MRVisualPoseGPU) == 48u);
static_assert(sizeof(MRVisualContactAnnotationGPU) == 48u);
#endif
