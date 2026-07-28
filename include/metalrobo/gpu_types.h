#pragma once

// This header is shared by C++ and Metal. Keep it free of STL and Objective-C.

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#define MR_ALIGN16 alignas(16)
typedef uint mr_u32;
typedef int mr_i32;
typedef ulong mr_u64;
typedef float4 mr_float4;
#else
#include <stdint.h>
#define MR_ALIGN16 alignas(16)
typedef uint32_t mr_u32;
typedef int32_t mr_i32;
typedef uint64_t mr_u64;
typedef struct MR_ALIGN16 mr_float4 {
    float x;
    float y;
    float z;
    float w;
} mr_float4;
#endif

#define MR_MAX_DOF 32u
#define MR_MAX_LINKS 33u
#define MR_MAX_COLLIDERS 64u
#define MR_MAX_OBSERVATIONS 128u
#define MR_SIMD_WIDTH 32u

enum MRShapeType : mr_u32 {
    MR_SHAPE_SPHERE = 0u,
    MR_SHAPE_CAPSULE = 1u,
    MR_SHAPE_BOX = 2u,
    MR_SHAPE_PLANE = 3u,
    MR_SHAPE_CYLINDER = 4u,
    MR_SHAPE_CONVEX = 5u,
    MR_SHAPE_TRIANGLE_MESH = 6u,
    MR_SHAPE_HEIGHTFIELD = 7u,
    MR_SHAPE_SDF = 8u,
};

typedef struct MR_ALIGN16 MRModelGPU {
    mr_u32 dofCount;
    mr_u32 linkCount;
    mr_u32 colliderCount;
    mr_u32 observationCount;

    mr_u32 actionCount;
    mr_u32 episodeHorizon;
    mr_u32 substeps;
    mr_u32 flags;

    // xyz = gravitational acceleration in world coordinates, w = control dt.
    mr_float4 gravityAndTimestep;
    // xyz = plane normal, w = signed plane offset: dot(n, x) - offset = 0.
    mr_float4 groundPlane;
    // xyz = lower target sampling extent, w = success radius.
    mr_float4 targetLowerAndRadius;
    // xyz = upper target sampling extent, w = terminal success bonus.
    mr_float4 targetUpperAndBonus;
    // distance scale, action penalty, velocity penalty, contact penalty.
    mr_float4 rewardScales;
} MRModelGPU;

typedef struct MR_ALIGN16 MRJointGPU {
    mr_i32 parentLink;
    mr_u32 childLink;
    mr_u32 jointType;
    mr_u32 reserved;

    // Joint axis in the child body frame at q=0.
    mr_float4 axis;
    // Child origin in the parent body frame at q=0.
    mr_float4 parentOffset;
    // Quaternion (x, y, z, w), child-to-parent orientation at q=0.
    mr_float4 parentRotation;
    // lower, upper, maximum velocity, maximum effort.
    mr_float4 limits;
    // kp, kd, normalized-action position scale, armature.
    mr_float4 drive;
} MRJointGPU;

typedef struct MR_ALIGN16 MRLinkGPU {
    // mass, center of mass xyz in the body frame.
    mr_float4 massAndCOMX;
    // Symmetric inertia about the COM:
    // row0 = (ixx, ixy, ixz, 0)
    // row1 = (ixy, iyy, iyz, 0)
    // row2 = (ixz, iyz, izz, 0)
    mr_float4 inertiaRow0;
    mr_float4 inertiaRow1;
    mr_float4 inertiaRow2;
} MRLinkGPU;

typedef struct MR_ALIGN16 MRColliderGPU {
    mr_i32 linkIndex;
    mr_u32 shapeType;
    mr_u32 collisionGroup;
    mr_u32 collisionMask;

    // Sphere: xyz center, w radius.
    // Capsule: xyz first endpoint, w radius; second endpoint is in extent.xyz.
    // Box: xyz center; half extent is in extent.xyz.
    mr_float4 centerAndRadius;
    mr_float4 extent;
    // friction, restitution, normal stiffness, normal damping.
    mr_float4 material;
} MRColliderGPU;

typedef struct MR_ALIGN16 MRStepUniformsGPU {
    mr_u32 environmentCount;
    mr_u32 seedLo;
    mr_u32 seedHi;
    mr_u32 resetAll;
    mr_u32 autoReset;
    mr_u32 captureBodyPoses;
    mr_u32 reserved0;
    mr_u32 reserved1;
} MRStepUniformsGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(mr_float4) == 16);
static_assert(sizeof(MRModelGPU) % 16 == 0);
static_assert(sizeof(MRJointGPU) % 16 == 0);
static_assert(sizeof(MRLinkGPU) % 16 == 0);
static_assert(sizeof(MRColliderGPU) % 16 == 0);
static_assert(sizeof(MRStepUniformsGPU) % 16 == 0);
#endif
