#pragma once

#include "metalrobo/EngineModel.hpp"
#include "metalrobo/tactile_types.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

// Tactile poses use the same xyzw/body-to-parent convention as EngineModel,
// but remain independent of the visual camera authoring structures.
struct TactilePose {
    mr_float4 position{0.0f, 0.0f, 0.0f, 0.0f};
    mr_float4 orientation{0.0f, 0.0f, 0.0f, 1.0f};
};

struct TactileSampleSpec {
    mr_float4 localPosition{0.0f, 0.0f, 0.0f, 0.0f};
    mr_float4 localNormal{0.0f, 0.0f, 1.0f, 0.0f};
    mr_float4 localTangentU{1.0f, 0.0f, 0.0f, 0.0f};
    mr_float4 localTangentV{0.0f, 1.0f, 0.0f, 0.0f};
    float areaSquareMeters = 0.0f;
    float maximumDepthMeters = 0.0f;
    std::uint32_t atlasU = 0u;
    std::uint32_t atlasV = 0u;
    bool valid = true;
};

// Static authoring record. Flat and spherical helpers populate `samples`;
// arbitrary curved sensors may provide a UV atlas directly. Every position,
// normal, tangent, and area is cooked once and shared by all environments.
struct TactileSensorSpec {
    std::string id;
    std::uint32_t parentBodyIndex = MR_INVALID_INDEX;
    std::uint32_t backingShapeIndex = MR_INVALID_INDEX;
    TactilePose localPose;
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    MRTactileSurfaceKind surfaceKind = MR_TACTILE_SURFACE_FLAT;
    float maximumDepthMeters = 0.0f;
    float activeDepthThresholdMeters = 1.0e-6f;
    float queryEpsilonMeters = 1.0e-6f;
    std::uint32_t updatePeriodSteps = 1u;
    std::uint32_t flags = MR_TACTILE_SENSOR_COMPLIANT_SHELL;
    // Empty means "cook all collision-filter-compatible shapes other than the
    // sensor's own body." Explicit lists remain ordered and deterministic.
    std::vector<std::uint32_t> targetShapeIndices;
    std::vector<TactileSampleSpec> samples;
};

// Pixel centres are sampled over physical width/height. The undeformed plane
// is sensor-local z=0 and the outward normal is +z.
[[nodiscard]] TactileSensorSpec makeFlatTactileSensor(
    std::string id,
    std::uint32_t parentBodyIndex,
    std::uint32_t backingShapeIndex,
    TactilePose localPose,
    std::uint32_t width,
    std::uint32_t height,
    float physicalWidthMeters,
    float physicalHeightMeters,
    float maximumDepthMeters
);

// Samples a spherical patch whose centre is `sphereCenterLocal`; +z is the
// centre atlas normal. Horizontal and vertical angular spans are in radians.
[[nodiscard]] TactileSensorSpec makeSphericalTactileSensor(
    std::string id,
    std::uint32_t parentBodyIndex,
    std::uint32_t backingShapeIndex,
    TactilePose localPose,
    std::uint32_t width,
    std::uint32_t height,
    mr_float4 sphereCenterLocal,
    float undeformedRadiusMeters,
    float horizontalSpanRadians,
    float verticalSpanRadians,
    float maximumDepthMeters
);

enum class TactileCookStatus : std::uint32_t {
    success = 0u,
    invalidSpecification,
    invalidBackingShape,
    invalidSampleAtlas,
    unsupportedGeometry,
    capacityOverflow,
};

struct TactileCookResult {
    TactileCookStatus status = TactileCookStatus::success;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == TactileCookStatus::success;
    }
};

struct TactileObservationSchema {
    std::uint32_t abiVersion = MR_TACTILE_ABI_VERSION;
    std::uint64_t fingerprint = 0u;
    std::uint32_t sensorCount = 0u;
    std::uint32_t totalSampleCount = 0u;
    // Canonical dense channel order. Depth and depth velocity are float32;
    // validity and object identity are uint32. All dense arrays are flattened
    // environment-major, then sensor/sample-major.
    std::vector<std::string> denseChannels{
        "penetration_depth_m",
        "validity_bits",
        "object_shape_id",
        "depth_velocity_m_per_s",
    };
    std::vector<std::string> summaryChannels{
        "sensor_pose",
        "timestamp_s",
        "net_force_n",
        "net_torque_nm",
        "center_of_pressure_sensor_m",
        "center_of_pressure_force_weight_n",
        "geometric_contact_centroid_m",
        "contact_area_m2",
        "maximum_depth_m",
        "mean_depth_m",
    };
};

struct CookedTactileSystem {
    std::uint32_t abiVersion = MR_TACTILE_ABI_VERSION;
    std::uint64_t fingerprint = 0u;
    std::vector<std::string> sensorIds;
    std::vector<MRTactileSensorGPU> sensors;
    std::vector<MRTactileSampleGPU> samples;
    std::vector<std::uint32_t> targetShapeIndices;

    [[nodiscard]] bool valid(
        const EngineModel& model,
        std::string* reason = nullptr
    ) const;

    [[nodiscard]] TactileObservationSchema observationSchema() const;
};

[[nodiscard]] TactileCookResult cookTactileSystem(
    std::span<const TactileSensorSpec> sensors,
    const EngineModel& model,
    CookedTactileSystem& output
);

struct TactileObservationBatch {
    std::uint32_t environmentCount = 0u;
    std::uint32_t sensorCount = 0u;
    std::uint32_t sampleCount = 0u;
    std::uint64_t frameIndex = 0u;
    double timestampSeconds = 0.0;
    std::vector<float> penetrationDepthMeters;
    std::vector<float> depthVelocityMetersPerSecond;
    std::vector<std::uint32_t> validity;
    std::vector<std::uint32_t> objectShapeIds;
    std::vector<MRTactileHitGPU> debugHits;
    std::vector<MRTactileSummaryGPU> summaries;
    std::vector<MRTactileStatusGPU> statuses;
};

struct TactileCpuFrame {
    std::uint32_t environmentCount = 0u;
    std::span<const MRBodyStateGPU> bodies;
    // Environment-major fixed-capacity solver-contact arena.
    std::span<const MRTactileContactGPU> contacts;
    std::span<const std::uint32_t> contactCounts;
    std::uint32_t contactCapacityPerEnvironment = 0u;
    // Optional previous dense map and reset mask. A reset produces zero depth
    // velocity while the current geometric map remains valid.
    std::span<const float> previousDepthMeters;
    std::span<const std::uint32_t> previousValidity;
    std::span<const std::uint32_t> previousObjectShapeIds;
    std::span<const MRTactileHitGPU> previousDebugHits;
    std::span<const std::uint32_t> resetMask;
    // Base simulation/control-step interval. A sensor updated every N frames
    // reports (currentDepth - previousDepth) / (N * this interval).
    float observationTimestepSeconds = 0.0f;
    // Interval over which each packed solver impulse was accumulated. This
    // commonly equals one physics substep and is deliberately independent of
    // the tactile observation period used for depth velocity.
    float contactImpulseTimestepSeconds = 0.0f;
    std::uint64_t frameIndex = 0u;
    double timestampSeconds = 0.0;
};

struct TactileObserveResult {
    MRTactileStatusCode status = MR_TACTILE_SUCCESS;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MR_TACTILE_SUCCESS;
    }
};

struct TactileSolverContactFrame {
    std::uint32_t environmentCount = 0u;
    std::uint32_t bodyCount = 0u;
    std::uint32_t contactCapacityPerEnvironment = 0u;
    std::uint32_t manifoldCapacityPerEnvironment = 0u;
    std::span<const MRShapeGPU> shapes;
    std::span<const MRBodyStateGPU> bodies;
    std::span<const MRContactConstraintGPU> constraints;
    std::span<const MRContactPointMetaGPU> metadata;
    std::span<const MRManifoldHeaderGPU> manifoldHeaders;
    std::span<const std::uint32_t> activeContactCounts;
};

struct TactileSolverContactBatch {
    std::uint32_t environmentCount = 0u;
    std::uint32_t capacityPerEnvironment = 0u;
    std::vector<MRTactileContactGPU> contacts;
    std::vector<std::uint32_t> counts;
};

// Converts the physics engine's final solved contact cache into the explicit
// world-impulse records consumed by tactile reductions. Tangential impulses
// use the same persisted manifold frame as the solver; no force is inferred
// from penetration depth.
[[nodiscard]] TactileObserveResult packTactileSolverContacts(
    const TactileSolverContactFrame& frame,
    TactileSolverContactBatch& output
);

// Slow FP64 geometry oracle. It deliberately performs bounded scalar work in
// deterministic order and is the executable definition against which Metal
// is tested. Output is transactional.
[[nodiscard]] TactileObserveResult observeTactileCpuReference(
    const CookedTactileSystem& tactile,
    const EngineModel& model,
    const TactileCpuFrame& frame,
    TactileObservationBatch& output
);

// Stable JSON metadata saved beside policy checkpoints and deterministic
// replays. It records the metric representation, atlas shapes, update rates,
// and exact cooked fingerprint; it contains no model weights.
[[nodiscard]] std::string tactileObservationMetadataJSON(
    const CookedTactileSystem& tactile
);

[[nodiscard]] const char*
tactileCookStatusName(TactileCookStatus status) noexcept;

} // namespace metalrobo
