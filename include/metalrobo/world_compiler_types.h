#pragma once

// Pointer-free world-compiler ABI shared by C++, Objective-C++, and Metal.
// These records describe already-compiled environment instances. Strings,
// provenance, and authored distributions remain in the host-side IR.

#include "metalrobo/gpu_types.h"

#define MR_WORLD_COMPILER_ABI_VERSION 2u

enum MRWorldAssetRole : mr_u32 {
    MR_WORLD_ASSET_BACKGROUND = 0u,
    MR_WORLD_ASSET_ROBOT = 1u,
    MR_WORLD_ASSET_MANIPULATED = 2u,
    MR_WORLD_ASSET_FIXTURE = 3u,
    MR_WORLD_ASSET_CLUTTER = 4u,
    MR_WORLD_ASSET_SENSOR_RIG = 5u,
};

enum MRWorldRenderRepresentation : mr_u32 {
    MR_WORLD_RENDER_NONE = 0u,
    MR_WORLD_RENDER_GAUSSIAN_FIELD = 1u,
    MR_WORLD_RENDER_MESH_PBR = 2u,
    MR_WORLD_RENDER_NEURAL_RESIDUAL = 3u,
    MR_WORLD_RENDER_PROCEDURAL = 4u,
};

enum MRWorldCollisionRepresentation : mr_u32 {
    MR_WORLD_COLLISION_NONE = 0u,
    MR_WORLD_COLLISION_PRIMITIVES = 1u,
    MR_WORLD_COLLISION_CONVEX = 2u,
    MR_WORLD_COLLISION_TRIANGLE_MESH = 3u,
    MR_WORLD_COLLISION_SDF = 4u,
    MR_WORLD_COLLISION_DEFORMABLE_SURFACE = 5u,
};

enum MRWorldDynamicsRepresentation : mr_u32 {
    MR_WORLD_DYNAMICS_STATIC = 0u,
    MR_WORLD_DYNAMICS_KINEMATIC = 1u,
    MR_WORLD_DYNAMICS_RIGID = 2u,
    MR_WORLD_DYNAMICS_ARTICULATED = 3u,
    MR_WORLD_DYNAMICS_ROD = 4u,
    MR_WORLD_DYNAMICS_SHELL = 5u,
    MR_WORLD_DYNAMICS_SOFT_VOLUME = 6u,
};

enum MRWorldSensorKind : mr_u32 {
    MR_WORLD_SENSOR_RGB = 0u,
    MR_WORLD_SENSOR_DEPTH = 1u,
    MR_WORLD_SENSOR_RGBD = 2u,
    MR_WORLD_SENSOR_SEGMENTATION = 3u,
    MR_WORLD_SENSOR_STATE = 4u,
    MR_WORLD_SENSOR_FORCE_TORQUE = 5u,
    // Dense metric normal-penetration atlas. This is not camera depth and
    // does not use camera intrinsics.
    MR_WORLD_SENSOR_TACTILE_DEPTH = 6u,
};

enum MRWorldSensorParentKind : mr_u32 {
    MR_WORLD_SENSOR_PARENT_ASSET = 0u,
    MR_WORLD_SENSOR_PARENT_RIGID_BODY = 1u,
    MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK = 2u,
    MR_WORLD_SENSOR_PARENT_WORLD = 3u,
};

enum MRWorldSensorSchedulePhase : mr_u32 {
    // Sample the accepted state before action application. This is the
    // ordinary actor-observation boundary.
    MR_WORLD_SENSOR_PHASE_PRE_CONTROL = 0u,
    // Sample the newly accepted physical/contact state after integration.
    MR_WORLD_SENSOR_PHASE_POST_PHYSICS = 1u,
    // Sample on the native visual presentation timeline.
    MR_WORLD_SENSOR_PHASE_PRESENTATION = 2u,
};

enum MRWorldSensorConsumerFlags : mr_u32 {
    MR_WORLD_SENSOR_CONSUMER_ACTOR = 1u << 0u,
    MR_WORLD_SENSOR_CONSUMER_CRITIC = 1u << 1u,
    MR_WORLD_SENSOR_CONSUMER_TRUTH = 1u << 2u,
    MR_WORLD_SENSOR_CONSUMER_RECORDER = 1u << 3u,
};

enum MRWorldVariationAxis : mr_u32 {
    MR_WORLD_VARIATION_APPEARANCE = 0u,
    MR_WORLD_VARIATION_OBJECT_CONFIGURATION = 1u,
    MR_WORLD_VARIATION_CLUTTER = 2u,
    MR_WORLD_VARIATION_PHYSICS = 3u,
    MR_WORLD_VARIATION_ROBOT_STATE = 4u,
    MR_WORLD_VARIATION_CAMERA = 5u,
};

enum MRWorldDistributionKind : mr_u32 {
    MR_WORLD_DISTRIBUTION_CONSTANT = 0u,
    MR_WORLD_DISTRIBUTION_UNIFORM = 1u,
    MR_WORLD_DISTRIBUTION_LOG_UNIFORM = 2u,
    MR_WORLD_DISTRIBUTION_NORMAL_CLAMPED = 3u,
    MR_WORLD_DISTRIBUTION_CATEGORICAL = 4u,
};

enum MRWorldVariationTarget : mr_u32 {
    MR_WORLD_TARGET_ASSET_POSITION_X = 0u,
    MR_WORLD_TARGET_ASSET_POSITION_Y = 1u,
    MR_WORLD_TARGET_ASSET_POSITION_Z = 2u,
    MR_WORLD_TARGET_ASSET_SCALE = 3u,
    MR_WORLD_TARGET_ASSET_ORIENTATION_ROLL = 4u,
    MR_WORLD_TARGET_ASSET_ORIENTATION_PITCH = 5u,
    MR_WORLD_TARGET_ASSET_ORIENTATION_YAW = 6u,
    MR_WORLD_TARGET_ASSET_MASS_SCALE = 7u,
    MR_WORLD_TARGET_ASSET_FRICTION_SCALE = 8u,
    MR_WORLD_TARGET_ASSET_RESTITUTION_SCALE = 9u,
    MR_WORLD_TARGET_ASSET_DAMPING_SCALE = 10u,
    MR_WORLD_TARGET_ROBOT_GAIN_SCALE = 11u,
    MR_WORLD_TARGET_ROBOT_DAMPING_SCALE = 12u,
    MR_WORLD_TARGET_ROBOT_LATENCY_SECONDS = 13u,
    MR_WORLD_TARGET_ROBOT_PAYLOAD_SCALE = 14u,
    MR_WORLD_TARGET_ASSET_RENDER_ALTERNATIVE = 15u,
    MR_WORLD_TARGET_SENSOR_POSITION_X = 16u,
    MR_WORLD_TARGET_SENSOR_POSITION_Y = 17u,
    MR_WORLD_TARGET_SENSOR_POSITION_Z = 18u,
    MR_WORLD_TARGET_SENSOR_ORIENTATION_ROLL = 19u,
    MR_WORLD_TARGET_SENSOR_ORIENTATION_PITCH = 20u,
    MR_WORLD_TARGET_SENSOR_ORIENTATION_YAW = 21u,
    MR_WORLD_TARGET_SENSOR_FOCAL_SCALE = 22u,
    MR_WORLD_TARGET_SENSOR_LATENCY_SECONDS = 23u,
    MR_WORLD_TARGET_SENSOR_COLOR_NOISE = 24u,
    MR_WORLD_TARGET_SENSOR_DEPTH_NOISE = 25u,
    MR_WORLD_TARGET_SENSOR_DEPTH_DROPOUT = 26u,
    MR_WORLD_TARGET_APPEARANCE_EXPOSURE = 27u,
    MR_WORLD_TARGET_APPEARANCE_WHITE_BALANCE = 28u,
    MR_WORLD_TARGET_APPEARANCE_SATURATION = 29u,
    MR_WORLD_TARGET_APPEARANCE_LIGHT_INTENSITY = 30u,
    MR_WORLD_TARGET_CLUTTER_SET = 31u,
    MR_WORLD_TARGET_ASSET_COLLISION_ALTERNATIVE = 32u,
};

enum MRWorldInstanceFlags : mr_u32 {
    MR_WORLD_INSTANCE_TWIN_RENDER = 1u << 0u,
    MR_WORLD_INSTANCE_FAMILY_RENDER = 1u << 1u,
    MR_WORLD_INSTANCE_PRIVILEGED_STATE = 1u << 2u,
};

enum MRWorldTemplateCapabilities : mr_u32 {
    MR_WORLD_CAP_STATE = 1u << 0u,
    MR_WORLD_CAP_RGB = 1u << 1u,
    MR_WORLD_CAP_DEPTH = 1u << 2u,
    MR_WORLD_CAP_SEGMENTATION = 1u << 3u,
    MR_WORLD_CAP_GAUSSIAN_RENDER = 1u << 4u,
    MR_WORLD_CAP_MESH_RENDER = 1u << 5u,
    MR_WORLD_CAP_RIGID_DYNAMICS = 1u << 6u,
    MR_WORLD_CAP_ARTICULATED_DYNAMICS = 1u << 7u,
    MR_WORLD_CAP_ROD_DYNAMICS = 1u << 8u,
    MR_WORLD_CAP_SHELL_DYNAMICS = 1u << 9u,
    MR_WORLD_CAP_SOFT_VOLUME_DYNAMICS = 1u << 10u,
    MR_WORLD_CAP_TACTILE_DEPTH = 1u << 11u,
};

typedef struct MR_ALIGN16 MRWorldInstanceHeaderGPU {
    // first asset, asset count, first sensor, sensor count.
    mr_uint4 ranges;
    // first appearance, appearance count, program index, topology cohort.
    mr_uint4 program;
    // scenario key low/high, flags, ABI version.
    mr_uint4 identity;
} MRWorldInstanceHeaderGPU;

typedef struct MR_ALIGN16 MRWorldAssetInstanceGPU {
    // xyz position in the world frame, w uniform render/collision scale.
    mr_float4 positionAndScale;
    // xyzw world orientation.
    mr_float4 orientation;
    // xyz linear velocity, w reserved.
    mr_float4 linearVelocity;
    // xyz angular velocity, w reserved.
    mr_float4 angularVelocity;
    // mass scale, friction scale, restitution scale, damping scale.
    mr_float4 physical;
    // gain scale, damping scale, latency seconds, payload scale.
    mr_float4 controller;
    // template asset, render alternative, collision alternative, dynamics.
    mr_uint4 identity;
} MRWorldAssetInstanceGPU;

typedef struct MR_ALIGN16 MRWorldSensorInstanceGPU {
    // xyz position relative to the parent frame, w focal-length scale.
    mr_float4 positionAndFocalScale;
    // xyzw orientation relative to the parent frame.
    mr_float4 orientation;
    // fx, fy, cx, cy in pixels.
    mr_float4 intrinsics;
    // radial k1, k2, tangential p1, p2.
    mr_float4 distortion;
    // color noise sigma, depth noise sigma, depth dropout, latency seconds.
    mr_float4 noiseAndLatency;
    // parent asset, sensor kind, width, height.
    mr_uint4 identity;
} MRWorldSensorInstanceGPU;

typedef struct MR_ALIGN16 MRWorldAppearanceInstanceGPU {
    // exposure stops, white-balance kelvin scale, saturation, light intensity.
    mr_float4 colorAndLight;
    // hue radians, contrast, roughness scale, metallic scale.
    mr_float4 material;
    // appearance alternative, environment map, flags, reserved.
    mr_uint4 identity;
} MRWorldAppearanceInstanceGPU;

typedef struct MR_ALIGN16 MRWorldVariationGPU {
    // variation axis, distribution kind, target kind, target index.
    mr_uint4 binding;
    // Distribution parameters:
    // constant: x=value
    // uniform/log-uniform: x=lower, y=upper
    // normal-clamped: x=mean, y=sigma, z=lower, w=upper.
    mr_float4 parameters;
    // first categorical value, value count, authored order, reserved.
    mr_uint4 categorical;
    // stream selector, salt low/high, flags.
    mr_uint4 random;
} MRWorldVariationGPU;

typedef struct MR_ALIGN16 MRWorldAssetBindingGPU {
    // template asset, semantic role, render representation, collision rep.
    mr_uint4 identity;
    // dynamics representation, articulation, topology cohort, flags.
    mr_uint4 dynamics;
    // body-index arena offset/count, shape-index arena offset/count.
    mr_uint4 geometryRanges;
    // material-index arena offset/count, render alternative, collision alt.
    mr_uint4 materialRangeAndAlternatives;
} MRWorldAssetBindingGPU;

typedef struct MR_ALIGN16 MRWorldBodyParametersGPU {
    // mass, friction, restitution, and damping scales for this body.
    mr_float4 physical;
    // owning asset, dynamics representation, flags, reserved.
    mr_uint4 identity;
} MRWorldBodyParametersGPU;

typedef struct MR_ALIGN16 MRWorldControllerParametersGPU {
    // gain scale, damping scale, latency seconds, payload scale.
    mr_float4 controller;
    // owning asset, articulation, flags, reserved.
    mr_uint4 identity;
} MRWorldControllerParametersGPU;

typedef struct MR_ALIGN16 MRWorldFamilySampleUniformsGPU {
    // environment count, asset count, sensor count, appearance count.
    mr_uint4 counts;
    // variation count, categorical value count, instance flags, ABI version.
    mr_uint4 program;
    // seed low/high, family fingerprint low/high.
    mr_uint4 identity;
} MRWorldFamilySampleUniformsGPU;

typedef struct MR_ALIGN16 MRWorldFamilyMaterializeUniformsGPU {
    // environment count, primary articulation nq/nv, scene-body count.
    mr_uint4 stateCounts;
    // body count, articulation count, asset count, primary articulation.
    mr_uint4 topology;
    // compiler ABI version and reserved fields.
    mr_uint4 identity;
} MRWorldFamilyMaterializeUniformsGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(MRWorldInstanceHeaderGPU) == 48u);
static_assert(sizeof(MRWorldAssetInstanceGPU) == 112u);
static_assert(sizeof(MRWorldSensorInstanceGPU) == 96u);
static_assert(sizeof(MRWorldAppearanceInstanceGPU) == 48u);
static_assert(sizeof(MRWorldVariationGPU) == 64u);
static_assert(sizeof(MRWorldAssetBindingGPU) == 64u);
static_assert(sizeof(MRWorldBodyParametersGPU) == 32u);
static_assert(sizeof(MRWorldControllerParametersGPU) == 32u);
static_assert(sizeof(MRWorldFamilySampleUniformsGPU) == 48u);
static_assert(sizeof(MRWorldFamilyMaterializeUniformsGPU) == 48u);
#endif
