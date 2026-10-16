#pragma once

// Pointer-free tactile ABI shared by C++, Objective-C++, Metal, and Swift.
// The primary signal is always metric normal penetration in metres. Optional
// channels and reductions must not change that definition.

#include "metalrobo/engine_types.h"

#define MR_TACTILE_ABI_VERSION 1u
#define MR_TACTILE_MAX_TARGETS_PER_SENSOR 4096u

enum MRTactileSurfaceKind : mr_u32 {
    MR_TACTILE_SURFACE_FLAT = 0u,
    MR_TACTILE_SURFACE_CURVED = 1u,
    MR_TACTILE_SURFACE_CUSTOM_ATLAS = 2u,
};

enum MRTactileSensorFlags : mr_u32 {
    // The backing shape's positive rest offset is the virtual elastomer
    // thickness. Its authored surface remains the rigid backing.
    MR_TACTILE_SENSOR_COMPLIANT_SHELL = 1u << 0u,
};

enum MRTactileSampleFlags : mr_u32 {
    MR_TACTILE_SAMPLE_VALID = 1u << 0u,
};

enum MRTactileValidityFlags : mr_u32 {
    // The atlas location maps to a physical surface sample.
    MR_TACTILE_VALIDITY_SAMPLE = 1u << 0u,
    // A configured target occupies the undeformed sensing surface and exits
    // within the local inward-normal sensing segment.
    MR_TACTILE_VALIDITY_CONTACT = 1u << 1u,
    // The metric depth reached the configured maximum measurable depth.
    MR_TACTILE_VALIDITY_SATURATED = 1u << 2u,
    // The selected target is also admitted by the backing collider's static
    // collision filters.
    MR_TACTILE_VALIDITY_FILTERED_TARGET = 1u << 3u,
};

enum MRTactileContactFlags : mr_u32 {
    // worldImpulseOnA contains the complete solver impulse applied to shape A
    // over the observation interval.
    MR_TACTILE_CONTACT_SOLVER_IMPULSE = 1u << 0u,
};

enum MRTactileSummaryFlags : mr_u32 {
    MR_TACTILE_SUMMARY_DEPTH_VALID = 1u << 0u,
    MR_TACTILE_SUMMARY_WRENCH_VALID = 1u << 1u,
    MR_TACTILE_SUMMARY_UPDATED = 1u << 2u,
    MR_TACTILE_SUMMARY_RESET = 1u << 3u,
};

enum MRTactileQueryBackend : mr_u32 {
    MR_TACTILE_QUERY_CPU_REFERENCE = 0u,
    // Analytic primitives, convex half spaces, and the engine's shared
    // stackless BVH4 for closed triangle meshes.
    MR_TACTILE_QUERY_METAL_ANALYTIC_BVH4 = 1u,
    MR_TACTILE_QUERY_METAL_INTERSECTION_FUNCTION = 2u,
};

enum MRTactileStatusCode : mr_u32 {
    MR_TACTILE_SUCCESS = 0u,
    MR_TACTILE_INVALID_CONFIGURATION = 1u,
    MR_TACTILE_CAPACITY_OVERFLOW = 2u,
    MR_TACTILE_NONFINITE_INPUT = 3u,
    MR_TACTILE_UNSUPPORTED_GEOMETRY = 4u,
    MR_TACTILE_METAL_UNAVAILABLE = 5u,
    MR_TACTILE_PIPELINE_FAILURE = 6u,
    MR_TACTILE_COMMAND_FAILURE = 7u,
};

typedef struct MR_ALIGN16 MRTactileSensorGPU {
    // Parent body, reserved, first sample, sample count.
    mr_uint4 topology;
    // First backing-shape arena record, backing count, reserved, reserved.
    mr_uint4 backingRange;
    // Atlas width/height, first target-shape index, target count.
    mr_uint4 atlasAndTargets;
    // Update period in physics steps, surface kind, flags, stable ordinal.
    mr_uint4 scheduleAndIdentity;

    // Sensor-frame origin relative to the parent body; w = outward query
    // epsilon. Samples are stored in this sensor frame.
    mr_float4 localPositionAndQueryEpsilon;
    // Sensor-to-parent normalized quaternion (xyzw).
    mr_float4 localOrientation;
    // Maximum depth, active-depth threshold, shell thickness, and maximum
    // bounded contact-relative tangential displacement (all metres).
    mr_float4 depth;
} MRTactileSensorGPU;

typedef struct MR_ALIGN16 MRTactileSampleGPU {
    // Rest position in the sensor frame; w = represented physical area (m2).
    mr_float4 localPositionAndArea;
    // Outward unit normal; w = per-sample maximum depth (m).
    mr_float4 localNormalAndMaximumDepth;
    // Unit tangent frame. w is reserved.
    mr_float4 localTangentU;
    mr_float4 localTangentV;
    // Atlas u/v, owning sensor ordinal, MRTactileSampleFlags.
    mr_uint4 atlasAndIdentity;
} MRTactileSampleGPU;

typedef struct MR_ALIGN16 MRTactileDispatchGPU {
    // Environments, bodies/environment, sensors, total samples.
    mr_uint4 counts;
    // Shapes, geometry headers, geometry vertices, convex faces.
    mr_uint4 geometryCounts;
    // BVH nodes, mesh triangles, target indices, contacts/environment.
    mr_uint4 queryCounts;
    // Frame index low/high, ABI version, reserved.
    mr_uint4 frameAndAbi;
    // Base simulation/control-step dt, its inverse, timestamp seconds, and
    // inverse interval represented by the solver impulses.
    mr_float4 timing;
} MRTactileDispatchGPU;

typedef struct MR_ALIGN16 MRTactileContactGPU {
    // Shape A, shape B, MRTactileContactFlags, reserved.
    mr_uint4 shapesAndFlags;
    // World contact point.
    mr_float4 worldPoint;
    // Complete world impulse applied to shape A during the explicitly
    // supplied solver-impulse interval. Force is impulse / that interval.
    mr_float4 worldImpulseOnA;
    // Normal impulse magnitude, tangential impulse magnitude, static
    // friction, and dynamic friction. These are solver evidence, not values
    // inferred from the geometric depth map.
    mr_float4 solverImpulseAndFriction;
} MRTactileContactGPU;

typedef struct MR_ALIGN16 MRTactileTangentialMotionGPU {
    // Bounded contact-relative displacement in the cooked sample tangent
    // frame (metres), followed by instantaneous target-minus-sensor surface
    // velocity in the same frame (metres/second). The displacement is bounded
    // rigid-body contact-history kinematics, not membrane deformation.
    mr_float4 displacementAndVelocity;
} MRTactileTangentialMotionGPU;

typedef struct MR_ALIGN16 MRTactileHitGPU {
    // World exit point from the occupying target; w = metric depth.
    mr_float4 worldPointAndDepth;
    // Outward sensor normal in world coordinates; w = ray-exit parameter.
    mr_float4 worldNormalAndRayParameter;
    // Target shape, atlas u/v packed ordinal, validity flags, reserved.
    mr_uint4 identityAndFlags;
} MRTactileHitGPU;

typedef struct MR_ALIGN16 MRTactileSummaryGPU {
    // Sensor origin in world coordinates; w = timestamp seconds.
    mr_float4 posePositionAndTimestamp;
    // Sensor-to-world normalized quaternion.
    mr_float4 poseOrientation;
    // Net solver force on the sensor in world coordinates; w = active area.
    mr_float4 netForceAndContactArea;
    // Net solver torque about the sensor origin; w = maximum depth.
    mr_float4 netTorqueAndMaximumDepth;
    // Area-weighted contact centroid in sensor coordinates; w = mean depth.
    mr_float4 centroidLocalAndMeanDepth;
    // Area-weighted contact centroid in world coordinates; w = active count.
    mr_float4 centroidWorldAndActiveCount;
    // Solver-force-magnitude-weighted center of pressure in the sensor frame;
    // w = sum of contributing force magnitudes in newtons.
    mr_float4 centerOfPressureLocalAndForceWeight;
    // The same center of pressure in world coordinates; w = contributing
    // solver-contact count. This is a compact resultant-contact descriptor,
    // not a complete pressure distribution.
    mr_float4 centerOfPressureWorldAndContactCount;
    // Area-weighted RMS tangential speed (m/s), maximum tangential
    // displacement (m), force-weighted mean friction utilization, and
    // maximum friction utilization. Utilization is derived from solver
    // impulses and authored friction, never from penetration depth.
    mr_float4 tangentialMotionAndFriction;
    // Saturated count, contributing solver contacts, summary flags, object ID
    // when all active samples agree (otherwise MR_INVALID_INDEX).
    mr_uint4 statisticsAndIdentity;
} MRTactileSummaryGPU;

typedef struct MR_ALIGN16 MRTactileStatusGPU {
    mr_u32 code;
    mr_u32 environment;
    mr_u32 sensor;
    // Failing sample or solver-contact ordinal, otherwise MR_INVALID_INDEX.
    mr_u32 sample;
    // Query backend, visited targets, visited BVH children, triangle tests.
    mr_uint4 diagnostics;
} MRTactileStatusGPU;

#ifndef __METAL_VERSION__
#include <type_traits>

static_assert(std::is_trivially_copyable_v<MRTactileSensorGPU>);
static_assert(std::is_trivially_copyable_v<MRTactileSampleGPU>);
static_assert(std::is_trivially_copyable_v<MRTactileDispatchGPU>);
static_assert(std::is_trivially_copyable_v<MRTactileContactGPU>);
static_assert(
    std::is_trivially_copyable_v<MRTactileTangentialMotionGPU>
);
static_assert(std::is_trivially_copyable_v<MRTactileHitGPU>);
static_assert(std::is_trivially_copyable_v<MRTactileSummaryGPU>);
static_assert(std::is_trivially_copyable_v<MRTactileStatusGPU>);
static_assert(sizeof(MRTactileSensorGPU) == 112u);
static_assert(sizeof(MRTactileSampleGPU) == 80u);
static_assert(sizeof(MRTactileDispatchGPU) == 80u);
static_assert(sizeof(MRTactileContactGPU) == 64u);
static_assert(sizeof(MRTactileTangentialMotionGPU) == 16u);
static_assert(sizeof(MRTactileHitGPU) == 48u);
static_assert(sizeof(MRTactileSummaryGPU) == 160u);
static_assert(sizeof(MRTactileStatusGPU) == 32u);
#endif
