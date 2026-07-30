#include "metalrobo/MetalTactile.hpp"
#include "metalrobo/GeometryCooker.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numbers>
#include <ranges>
#include <stdexcept>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr float kShell = 0.004f;
constexpr float kRadius = 0.010f;
constexpr std::uint32_t kWidth = 41u;
constexpr std::uint32_t kHeight = 41u;

template <typename Result>
requires requires(const Result& value) {
    value.succeeded();
    value.message;
}
void require(const Result& result, const char* operation) {
    if (!result.succeeded()) {
        throw std::runtime_error(
            std::string{operation} + ": " + result.message
        );
    }
}

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

mr_float4 quaternionAxisAngle(
    const float x,
    const float y,
    const float z,
    const float radians
) {
    const float half = 0.5f * radians;
    const float sine = std::sin(half);
    return {
        x * sine,
        y * sine,
        z * sine,
        std::cos(half),
    };
}

mr_float4 rotate(
    const mr_float4 quaternion,
    const mr_float4 value
) {
    const mr_float4 twiceCross{
        2.0f * (
            quaternion.y * value.z -
            quaternion.z * value.y
        ),
        2.0f * (
            quaternion.z * value.x -
            quaternion.x * value.z
        ),
        2.0f * (
            quaternion.x * value.y -
            quaternion.y * value.x
        ),
        0.0f,
    };
    return {
        value.x + quaternion.w * twiceCross.x +
            quaternion.y * twiceCross.z -
            quaternion.z * twiceCross.y,
        value.y + quaternion.w * twiceCross.y +
            quaternion.z * twiceCross.x -
            quaternion.x * twiceCross.z,
        value.z + quaternion.w * twiceCross.z +
            quaternion.x * twiceCross.y -
            quaternion.y * twiceCross.x,
        value.w,
    };
}

metalrobo::EngineModel makeModel() {
    metalrobo::EngineModel model;
    model.name = "tactile_analytic_probe";
    model.bodies.resize(5u);
    for (std::uint32_t index = 0u;
         index < model.bodies.size();
         ++index) {
        model.bodies[index].articulationIndex = MR_INVALID_INDEX;
        model.bodies[index].parentBody = MR_INVALID_INDEX;
        model.bodies[index].inboundJoint = MR_INVALID_INDEX;
        model.bodies[index].motionType =
            index == 0u ? MR_MOTION_STATIC : MR_MOTION_KINEMATIC;
    }
    const auto shape = [](
        const std::uint32_t body,
        const std::uint32_t type,
        const mr_float4 dimensions,
        const mr_float4 localPosition,
        const mr_float4 localRotation
    ) {
        MRShapeGPU result{};
        result.bodyIndex = body;
        result.shapeType = type;
        result.materialIndex = 0u;
        result.collisionGroup = 1u;
        result.collisionMask = 1u;
        result.slotGeneration = 1u;
        result.geometryOffset = 0u;
        result.localPosition = localPosition;
        result.localRotation = localRotation;
        result.dimensions = dimensions;
        result.contactRestAndBoundingRadius = {
            0.001f,
            0.0f,
            0.1f,
            0.0f,
        };
        return result;
    };
    model.shapes.push_back(shape(
        0u,
        MR_SHAPE_BOX,
        {0.020f, 0.020f, 0.005f, 0.0f},
        {0.0f, 0.0f, -0.005f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f}
    ));
    model.shapes[0].contactRestAndBoundingRadius = {
        kShell,
        kShell,
        0.05f,
        0.0f,
    };
    model.shapes.push_back(shape(
        1u,
        MR_SHAPE_SPHERE,
        {kRadius, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f}
    ));
    model.shapes.push_back(shape(
        2u,
        MR_SHAPE_CYLINDER,
        {0.008f, 0.012f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
        quaternionAxisAngle(
            1.0f,
            0.0f,
            0.0f,
            0.5f * std::numbers::pi_v<float>
        )
    ));
    model.shapes.push_back(shape(
        3u,
        MR_SHAPE_BOX,
        {0.007f, 0.009f, 0.006f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
        quaternionAxisAngle(
            0.0f,
            0.0f,
            1.0f,
            0.25f * std::numbers::pi_v<float>
        )
    ));
    model.shapes.push_back(shape(
        4u,
        MR_SHAPE_SPHERE,
        {0.007f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f}
    ));
    return model;
}

MRBodyStateGPU makeBody(
    const mr_float4 position,
    const mr_float4 orientation = {0.0f, 0.0f, 0.0f, 1.0f}
) {
    MRBodyStateGPU result{};
    result.position = position;
    result.orientation = orientation;
    result.flagsAndIndices[0] = MR_MOTION_KINEMATIC;
    result.flagsAndIndices[1] = MR_INVALID_INDEX;
    result.flagsAndIndices[2] = MR_INVALID_INDEX;
    return result;
}

std::vector<MRBodyStateGPU> sphereContactBodies(
    const float penetration
) {
    std::vector<MRBodyStateGPU> bodies(5u);
    bodies[0] = makeBody({0.0f, 0.0f, 0.0f, 1.0f});
    bodies[0].flagsAndIndices[0] = MR_MOTION_STATIC;
    bodies[1] = makeBody({
        0.0f,
        0.0f,
        kShell + kRadius - penetration,
        1.0f,
    });
    bodies[2] = makeBody({1.0f, 0.0f, 1.0f, 1.0f});
    bodies[3] = makeBody({-1.0f, 0.0f, 1.0f, 1.0f});
    bodies[4] = makeBody({0.0f, 1.0f, 1.0f, 1.0f});
    return bodies;
}

metalrobo::CookedTactileSystem makeFlatSystem(
    const metalrobo::EngineModel& model,
    const std::uint32_t updatePeriod = 1u,
    const float maximumTangentialDisplacement = kShell
) {
    metalrobo::TactilePose pose;
    pose.position = {0.0f, 0.0f, kShell, 0.0f};
    auto sensor = metalrobo::makeFlatTactileSensor(
        "probe_flat",
        0u,
        0u,
        pose,
        kWidth,
        kHeight,
        0.024f,
        0.024f,
        kShell
    );
    sensor.targetShapeIndices = {1u, 2u, 3u, 4u};
    sensor.updatePeriodSteps = updatePeriod;
    sensor.maximumTangentialDisplacementMeters =
        maximumTangentialDisplacement;
    metalrobo::CookedTactileSystem tactile;
    require(
        metalrobo::cookTactileSystem(
            std::span<const metalrobo::TactileSensorSpec>{
                &sensor,
                1u,
            },
            model,
            tactile
        ),
        "cook flat tactile"
    );
    return tactile;
}

metalrobo::TactileObservationBatch observeCpu(
    const metalrobo::CookedTactileSystem& tactile,
    const metalrobo::EngineModel& model,
    const std::span<const MRBodyStateGPU> bodies,
    const std::span<const float> previous = {},
    const std::span<const std::uint32_t> previousValidity = {},
    const std::span<const std::uint32_t> previousObjects = {},
    const std::uint64_t frameIndex = 0u,
    const std::span<const std::uint32_t> reset = {},
    const std::span<const MRTactileTangentialMotionGPU>
        previousTangentialMotion = {},
    const std::span<const mr_float4> previousAnchors = {},
    const std::span<const MRTactileContactGPU> contacts = {},
    const std::span<const std::uint32_t> contactCounts = {},
    const std::uint32_t contactCapacity = 0u,
    const float contactImpulseTimestepSeconds = 0.0f
) {
    metalrobo::TactileObservationBatch output;
    metalrobo::TactileCpuFrame frame;
    frame.environmentCount = 1u;
    frame.bodies = bodies;
    frame.previousDepthMeters = previous;
    frame.previousValidity = previousValidity;
    frame.previousObjectShapeIds = previousObjects;
    frame.previousTangentialMotion =
        previousTangentialMotion;
    frame.previousTargetLocalContactAnchors =
        previousAnchors;
    frame.contacts = contacts;
    frame.contactCounts = contactCounts;
    frame.contactCapacityPerEnvironment = contactCapacity;
    frame.resetMask = reset;
    frame.observationTimestepSeconds = 0.01f;
    frame.contactImpulseTimestepSeconds =
        contactImpulseTimestepSeconds;
    frame.frameIndex = frameIndex;
    frame.timestampSeconds = 1.0 + frameIndex * 0.01;
    require(
        metalrobo::observeTactileCpuReference(
            tactile,
            model,
            frame,
            output
        ),
        "CPU tactile reference"
    );
    return output;
}

void requireMetalMatchesCpu(
    const metalrobo::CookedTactileSystem& tactile,
    const metalrobo::EngineModel& model,
    const std::span<const MRBodyStateGPU> bodies,
    const metalrobo::TactileObservationBatch& cpu,
    const std::string_view label
) {
    metalrobo::MetalTactileConfig config;
    config.contactCapacityPerEnvironment = 1u;
    metalrobo::MetalTactileContext context(config);
    const auto compiled = context.compile(tactile, model, 1u);
    require(
        compiled.succeeded(),
        std::string{label} + " Metal compile: " +
            compiled.message
    );
    metalrobo::MetalTactileHostFrame frame;
    frame.environmentCount = 1u;
    frame.bodies = bodies;
    frame.observationTimestepSeconds = 0.01f;
    frame.frameIndex = 0u;
    frame.timestampSeconds = 1.0;
    const auto observed = context.observe(frame);
    require(
        observed.succeeded(),
        std::string{label} + " Metal observe: " +
            observed.message
    );
    metalrobo::TactileObservationBatch gpu;
    const auto readback = context.readback(1u, gpu);
    require(
        readback.succeeded(),
        std::string{label} + " Metal readback: " +
            readback.message
    );
    require(
        gpu.penetrationDepthMeters.size() ==
            cpu.penetrationDepthMeters.size() &&
        gpu.tangentialMotion.size() ==
            cpu.tangentialMotion.size() &&
        gpu.validity.size() == cpu.validity.size() &&
        gpu.objectShapeIds.size() == cpu.objectShapeIds.size() &&
        gpu.summaries.size() == cpu.summaries.size(),
        std::string{label} + " Metal output extent mismatch"
    );
    double maximumDifference = 0.0;
    for (std::size_t index = 0u;
         index < cpu.penetrationDepthMeters.size();
         ++index) {
        maximumDifference = std::max(
            maximumDifference,
            static_cast<double>(std::abs(
                gpu.penetrationDepthMeters[index] -
                cpu.penetrationDepthMeters[index]
            ))
        );
        require(
            gpu.validity[index] == cpu.validity[index] &&
            gpu.objectShapeIds[index] == cpu.objectShapeIds[index],
            std::string{label} +
                " Metal validity or identity differs from CPU"
        );
        const mr_float4 cpuMotion =
            cpu.tangentialMotion[index].
                displacementAndVelocity;
        const mr_float4 gpuMotion =
            gpu.tangentialMotion[index].
                displacementAndVelocity;
        require(
            std::abs(cpuMotion.x - gpuMotion.x) < 2.0e-6f &&
            std::abs(cpuMotion.y - gpuMotion.y) < 2.0e-6f &&
            std::abs(cpuMotion.z - gpuMotion.z) < 2.0e-6f &&
            std::abs(cpuMotion.w - gpuMotion.w) < 2.0e-6f,
            std::string{label} +
                " Metal tangential motion differs from CPU"
        );
    }
    require(
        maximumDifference < 2.0e-6,
        std::string{label} +
            " Metal depth differs from CPU by more than 2 micrometres"
    );
    require(
        gpu.summaries[0u].statisticsAndIdentity.z ==
            cpu.summaries[0u].statisticsAndIdentity.z,
        std::string{label} +
            " Metal summary validity differs from CPU"
    );
    const mr_float4 cpuMotionSummary =
        cpu.summaries[0u].tangentialMotionAndFriction;
    const mr_float4 gpuMotionSummary =
        gpu.summaries[0u].tangentialMotionAndFriction;
    require(
        std::abs(cpuMotionSummary.x - gpuMotionSummary.x) <
                2.0e-6f &&
            std::abs(cpuMotionSummary.y - gpuMotionSummary.y) <
                2.0e-6f &&
            std::abs(cpuMotionSummary.z - gpuMotionSummary.z) <
                2.0e-6f &&
            std::abs(cpuMotionSummary.w - gpuMotionSummary.w) <
                2.0e-6f,
        std::string{label} +
            " Metal tangential reduction differs from CPU"
    );
}

void validateSphereField(
    const metalrobo::CookedTactileSystem& tactile,
    const metalrobo::TactileObservationBatch& observation,
    const float penetration
) {
    double maximumError = 0.0;
    for (std::uint32_t index = 0u;
         index < tactile.samples.size();
         ++index) {
        const auto& sample = tactile.samples[index];
        const double x = sample.localPositionAndArea.x;
        const double y = sample.localPositionAndArea.y;
        const double radialSquared = x * x + y * y;
        double expected = 0.0;
        if (radialSquared < kRadius * kRadius) {
            expected = std::max(
                0.0,
                std::sqrt(kRadius * kRadius - radialSquared) -
                    (kRadius - penetration)
            );
        }
        expected = std::min<double>(expected, kShell);
        maximumError = std::max(
            maximumError,
            std::abs(
                observation.penetrationDepthMeters[index] -
                expected
            )
        );
    }
    require(
        maximumError < 2.0e-7,
        "sphere-flat analytic depth error exceeded 0.2 micrometres: " +
            std::to_string(maximumError)
    );
}

void validateRigidTransformInvariance(
    const metalrobo::CookedTactileSystem& tactile,
    const metalrobo::EngineModel& model,
    const metalrobo::TactileObservationBatch& baseline,
    std::vector<MRBodyStateGPU> bodies
) {
    const mr_float4 orientation =
        quaternionAxisAngle(0.0f, 1.0f, 0.0f, 0.71f);
    const mr_float4 translation{0.31f, -0.22f, 0.17f, 0.0f};
    for (MRBodyStateGPU& body : bodies) {
        const mr_float4 rotated = rotate(
            orientation,
            body.position
        );
        body.position = {
            rotated.x + translation.x,
            rotated.y + translation.y,
            rotated.z + translation.z,
            1.0f,
        };
        body.orientation = orientation;
    }
    const auto transformed =
        observeCpu(tactile, model, bodies);
    requireMetalMatchesCpu(
        tactile,
        model,
        bodies,
        transformed,
        "rigid-transform"
    );
    double maximumDifference = 0.0;
    for (std::uint32_t index = 0u;
         index < tactile.samples.size();
         ++index) {
        maximumDifference = std::max(
            maximumDifference,
            static_cast<double>(std::abs(
                transformed.penetrationDepthMeters[index] -
                baseline.penetrationDepthMeters[index]
            ))
        );
    }
    require(
        maximumDifference < 2.0e-7,
        "rigid-transform invariance failed"
    );
}

void validateCylinderAndBox(
    const metalrobo::CookedTactileSystem& tactile,
    const metalrobo::EngineModel& model
) {
    auto bodies = sphereContactBodies(0.0f);
    constexpr float cylinderDepth = 0.0025f;
    bodies[1].position = {1.0f, 0.0f, 1.0f, 1.0f};
    bodies[2] = makeBody({
        0.0f,
        0.0f,
        kShell + 0.012f - cylinderDepth,
        1.0f,
    });
    const auto cylinder = observeCpu(tactile, model, bodies);
    requireMetalMatchesCpu(
        tactile,
        model,
        bodies,
        cylinder,
        "cylinder-flat"
    );
    const std::size_t center =
        (kHeight / 2u) * kWidth + kWidth / 2u;
    require(
        std::abs(
            cylinder.penetrationDepthMeters[center] -
            cylinderDepth
        ) < 2.0e-7f,
        "cylinder cap did not produce constant metric penetration"
    );

    constexpr float boxDepth = 0.003f;
    bodies[2].position = {1.0f, 0.0f, 1.0f, 1.0f};
    bodies[3] = makeBody({
        0.0f,
        0.0f,
        kShell + 0.006f - boxDepth,
        1.0f,
    });
    const auto box = observeCpu(tactile, model, bodies);
    requireMetalMatchesCpu(
        tactile,
        model,
        bodies,
        box,
        "rotated-box-flat"
    );
    require(
        std::abs(
            box.penetrationDepthMeters[center] - boxDepth
        ) < 2.0e-7f,
        "rotated box face depth is incorrect"
    );
    const auto active = std::count_if(
        box.penetrationDepthMeters.begin(),
        box.penetrationDepthMeters.end(),
        [](const float depth) { return depth > 0.0f; }
    );
    require(active > 200, "box contact patch is unexpectedly empty");
}

void validateMultipleContactsAndAtlasBoundary(
    const metalrobo::CookedTactileSystem& tactile,
    const metalrobo::EngineModel& model
) {
    auto bodies = sphereContactBodies(0.002f);
    bodies[1].position.x = -0.007f;
    bodies[4] = makeBody({
        0.008f,
        0.0f,
        kShell + 0.007f - 0.0015f,
        1.0f,
    });
    const auto simultaneous =
        observeCpu(tactile, model, bodies);
    requireMetalMatchesCpu(
        tactile,
        model,
        bodies,
        simultaneous,
        "simultaneous-contacts"
    );
    require(
        std::ranges::find(
            simultaneous.objectShapeIds,
            1u
        ) != simultaneous.objectShapeIds.end() &&
        std::ranges::find(
            simultaneous.objectShapeIds,
            4u
        ) != simultaneous.objectShapeIds.end(),
        "simultaneous target contacts did not preserve object identity"
    );

    bodies = sphereContactBodies(0.004f);
    bodies[1].position.x = 0.015f;
    const auto boundary = observeCpu(tactile, model, bodies);
    requireMetalMatchesCpu(
        tactile,
        model,
        bodies,
        boundary,
        "atlas-boundary"
    );
    bool rightBoundaryActive = false;
    for (std::uint32_t row = 0u; row < kHeight; ++row) {
        const std::size_t index =
            static_cast<std::size_t>(row) * kWidth +
            (kWidth - 1u);
        rightBoundaryActive =
            rightBoundaryActive ||
            boundary.penetrationDepthMeters[index] > 0.0f;
    }
    require(
        rightBoundaryActive,
        "contact crossing the tactile atlas boundary was lost"
    );

    metalrobo::TactilePose pose;
    pose.position = {0.0f, 0.0f, kShell, 0.0f};
    auto invalidSelf = metalrobo::makeFlatTactileSensor(
        "invalid_self_target",
        0u,
        0u,
        pose,
        5u,
        5u,
        0.01f,
        0.01f,
        kShell
    );
    invalidSelf.targetShapeIndices = {0u};
    metalrobo::CookedTactileSystem rejected;
    require(
        !metalrobo::cookTactileSystem(
            std::span<const metalrobo::TactileSensorSpec>{
                &invalidSelf,
                1u,
            },
            model,
            rejected
        ).succeeded(),
        "tactile cooking accepted a sensor self-hit target"
    );
}

void validateCurvedSensor() {
    metalrobo::EngineModel model;
    model.name = "curved_tactile_check";
    model.bodies.resize(2u);
    model.shapes.resize(2u);
    model.shapes[0].bodyIndex = 0u;
    model.shapes[0].shapeType = MR_SHAPE_SPHERE;
    model.shapes[0].collisionGroup = 1u;
    model.shapes[0].collisionMask = 1u;
    model.shapes[0].slotGeneration = 1u;
    model.shapes[0].localRotation = {0.0f, 0.0f, 0.0f, 1.0f};
    model.shapes[0].dimensions = {0.020f, 0.0f, 0.0f, 0.0f};
    model.shapes[0].contactRestAndBoundingRadius = {
        0.003f,
        0.003f,
        0.023f,
        0.0f,
    };
    model.shapes[1].bodyIndex = 1u;
    model.shapes[1].shapeType = MR_SHAPE_SPHERE;
    model.shapes[1].collisionGroup = 1u;
    model.shapes[1].collisionMask = 1u;
    model.shapes[1].slotGeneration = 1u;
    model.shapes[1].localRotation = {0.0f, 0.0f, 0.0f, 1.0f};
    model.shapes[1].dimensions = {0.009f, 0.0f, 0.0f, 0.0f};
    model.shapes[1].contactRestAndBoundingRadius = {
        0.001f,
        0.0f,
        0.01f,
        0.0f,
    };
    auto sensor = metalrobo::makeSphericalTactileSensor(
        "probe_curved",
        0u,
        0u,
        {},
        33u,
        25u,
        {0.0f, 0.0f, 0.0f, 0.0f},
        0.023f,
        1.2f,
        0.9f,
        0.003f
    );
    sensor.targetShapeIndices = {1u};
    metalrobo::CookedTactileSystem tactile;
    require(
        metalrobo::cookTactileSystem(
            std::span<const metalrobo::TactileSensorSpec>{
                &sensor,
                1u,
            },
            model,
            tactile
        ),
        "cook curved tactile"
    );
    std::array bodies{
        makeBody({0.0f, 0.0f, 0.0f, 1.0f}),
        makeBody({
            0.0f,
            0.0f,
            0.023f + 0.009f - 0.002f,
            1.0f,
        }),
    };
    bodies[1u].linearVelocityAndInverseMass = {
        0.04f,
        -0.03f,
        0.07f,
        0.0f,
    };
    const auto observation = observeCpu(tactile, model, bodies);
    requireMetalMatchesCpu(
        tactile,
        model,
        bodies,
        observation,
        "sphere-curved"
    );
    const std::size_t center = 12u * 33u + 16u;
    require(
        std::abs(
            observation.penetrationDepthMeters[center] -
            0.002f
        ) < 3.0e-5f,
        "curved fingertip centre did not use its local surface normal"
    );
    require(
        observation.summaries[0].
            netForceAndContactArea.w > 0.0f,
        "curved tactile contact area is empty"
    );
    require(
        std::abs(
            observation.tangentialMotion[center].
                displacementAndVelocity.z -
            0.04f
        ) < 2.0e-6f &&
        std::abs(
            observation.tangentialMotion[center].
                displacementAndVelocity.w +
            0.03f
        ) < 2.0e-6f,
        "curved fingertip velocity was not projected into its "
        "sample-local tangent frame"
    );

    model.shapes[1].shapeType = MR_SHAPE_CYLINDER;
    model.shapes[1].dimensions =
        {0.009f, 0.012f, 0.0f, 0.0f};
    model.shapes[1].localRotation = quaternionAxisAngle(
        1.0f,
        0.0f,
        0.0f,
        0.5f * std::numbers::pi_v<float>
    );
    bodies[1].position = {
        0.0f,
        0.0f,
        0.023f + 0.012f - 0.0015f,
        1.0f,
    };
    const auto cylinder = observeCpu(tactile, model, bodies);
    require(
        std::abs(
            cylinder.penetrationDepthMeters[center] -
            0.0015f
        ) < 3.0e-5f,
        "curved fingertip cylinder did not use the local normal"
    );
    requireMetalMatchesCpu(
        tactile,
        model,
        bodies,
        cylinder,
        "cylinder-curved"
    );
}

void validateCookedGeometryBackends() {
    const std::array<mr_float4, 8u> vertices{{
        {-0.006f, -0.006f, -0.006f, 1.0f},
        {0.006f, -0.006f, -0.006f, 1.0f},
        {0.006f, 0.006f, -0.006f, 1.0f},
        {-0.006f, 0.006f, -0.006f, 1.0f},
        {-0.006f, -0.006f, 0.006f, 1.0f},
        {0.006f, -0.006f, 0.006f, 1.0f},
        {0.006f, 0.006f, 0.006f, 1.0f},
        {-0.006f, 0.006f, 0.006f, 1.0f},
    }};
    const std::array<std::uint32_t, 36u> indices{{
        0u, 2u, 1u, 0u, 3u, 2u,
        4u, 5u, 6u, 4u, 6u, 7u,
        0u, 1u, 5u, 0u, 5u, 4u,
        1u, 2u, 6u, 1u, 6u, 5u,
        2u, 3u, 7u, 2u, 7u, 6u,
        3u, 0u, 4u, 3u, 4u, 7u,
    }};
    const auto baseModel = [] {
        metalrobo::EngineModel model;
        model.name = "tactile_cooked_geometry_probe";
        model.bodies.resize(2u);
        for (std::uint32_t body = 0u; body < 2u; ++body) {
            model.bodies[body].articulationIndex =
                MR_INVALID_INDEX;
            model.bodies[body].parentBody = MR_INVALID_INDEX;
            model.bodies[body].inboundJoint = MR_INVALID_INDEX;
            model.bodies[body].motionType =
                body == 0u
                ? MR_MOTION_STATIC
                : MR_MOTION_KINEMATIC;
        }
        model.shapes.resize(2u);
        model.shapes[0u].bodyIndex = 0u;
        model.shapes[0u].shapeType = MR_SHAPE_BOX;
        model.shapes[0u].collisionGroup = 1u;
        model.shapes[0u].collisionMask = 1u;
        model.shapes[0u].slotGeneration = 1u;
        model.shapes[0u].localPosition =
            {0.0f, 0.0f, -0.005f, 0.0f};
        model.shapes[0u].localRotation =
            {0.0f, 0.0f, 0.0f, 1.0f};
        model.shapes[0u].dimensions =
            {0.02f, 0.02f, 0.005f, 0.0f};
        model.shapes[0u].contactRestAndBoundingRadius =
            {kShell, kShell, 0.05f, 0.0f};
        model.shapes[1u].bodyIndex = 1u;
        model.shapes[1u].collisionGroup = 1u;
        model.shapes[1u].collisionMask = 1u;
        model.shapes[1u].slotGeneration = 1u;
        model.shapes[1u].localRotation =
            {0.0f, 0.0f, 0.0f, 1.0f};
        model.shapes[1u].contactRestAndBoundingRadius =
            {0.001f, 0.0f, 0.02f, 0.0f};
        return model;
    };
    const auto tactileFor = [](const metalrobo::EngineModel& model) {
        metalrobo::TactilePose pose;
        pose.position = {0.0f, 0.0f, kShell, 0.0f};
        auto sensor = metalrobo::makeFlatTactileSensor(
            "probe_cooked_geometry",
            0u,
            0u,
            pose,
            17u,
            17u,
            0.020f,
            0.020f,
            kShell
        );
        sensor.targetShapeIndices = {1u};
        metalrobo::CookedTactileSystem tactile;
        require(
            metalrobo::cookTactileSystem(
                std::span<const metalrobo::TactileSensorSpec>{
                    &sensor,
                    1u,
                },
                model,
                tactile
            ),
            "cook general-geometry tactile"
        );
        return tactile;
    };
    const auto validate = [&](
        const metalrobo::EngineModel& model,
        const float centerHeight,
        const std::string_view label
    ) {
        const auto tactile = tactileFor(model);
        std::array bodies{
            makeBody({0.0f, 0.0f, 0.0f, 1.0f}),
            makeBody({
                0.0f,
                0.0f,
                centerHeight,
                1.0f,
            }),
        };
        const auto cpu = observeCpu(tactile, model, bodies);
        const std::size_t center = 8u * 17u + 8u;
        require(
            std::abs(
                cpu.penetrationDepthMeters[center] -
                0.002f
            ) < 3.0e-6f,
            std::string{label} +
                " centre penetration is incorrect"
        );
        requireMetalMatchesCpu(
            tactile,
            model,
            bodies,
            cpu,
            label
        );
    };

    metalrobo::EngineModel capsule = baseModel();
    capsule.shapes[1u].shapeType = MR_SHAPE_CAPSULE;
    capsule.shapes[1u].dimensions =
        {0.006f, 0.008f, 0.0f, 0.0f};
    validate(
        capsule,
        kShell + 0.006f - 0.002f,
        "capsule-flat"
    );

    metalrobo::EngineModel convex = baseModel();
    const auto cookedConvex = metalrobo::cookConvexGeometry(
        convex,
        vertices,
        indices
    );
    require(cookedConvex, "cook tactile convex");
    convex.shapes[1u].shapeType = MR_SHAPE_CONVEX;
    convex.shapes[1u].geometryOffset =
        cookedConvex.geometryIndex;
    convex.shapes[1u].geometryCount = 1u;
    convex.shapes[1u].dimensions =
        {1.0f, 1.0f, 1.0f, 0.0f};
    validate(
        convex,
        kShell + 0.006f - 0.002f,
        "convex-flat"
    );

    metalrobo::EngineModel mesh = baseModel();
    const auto cookedMesh = metalrobo::cookTriangleMeshGeometry(
        mesh,
        vertices,
        indices
    );
    require(cookedMesh, "cook tactile closed mesh");
    mesh.shapes[1u].shapeType = MR_SHAPE_TRIANGLE_MESH;
    mesh.shapes[1u].geometryOffset = cookedMesh.geometryIndex;
    mesh.shapes[1u].geometryCount = 1u;
    mesh.shapes[1u].dimensions =
        {1.0f, 1.0f, 1.0f, 0.0f};
    validate(
        mesh,
        kShell + 0.006f - 0.002f,
        "closed-mesh-flat"
    );
}

void validateDecimationAndReset(
    const metalrobo::EngineModel& model
) {
    const auto tactile = makeFlatSystem(model, 2u);
    auto firstBodies = sphereContactBodies(0.001f);
    const auto first =
        observeCpu(tactile, model, firstBodies, {}, {}, {}, 0u);
    auto movedBodies = sphereContactBodies(0.003f);
    const auto held = observeCpu(
        tactile,
        model,
        movedBodies,
        first.penetrationDepthMeters,
        first.validity,
        first.objectShapeIds,
        1u
    );
    require(
        held.penetrationDepthMeters ==
            first.penetrationDepthMeters,
        "decimated tactile frame did not hold its previous map"
    );
    const auto updated = observeCpu(
        tactile,
        model,
        movedBodies,
        held.penetrationDepthMeters,
        held.validity,
        held.objectShapeIds,
        2u
    );
    require(
        *std::max_element(
            updated.penetrationDepthMeters.begin(),
            updated.penetrationDepthMeters.end()
        ) > 0.0029f,
        "updated tactile frame did not observe the moved object"
    );
    require(
        std::abs(
            *std::max_element(
                updated.depthVelocityMetersPerSecond.begin(),
                updated.depthVelocityMetersPerSecond.end()
            ) -
            0.1f
        ) < 2.0e-4f,
        "decimated depth velocity did not use the base-step interval"
    );
    const std::array<std::uint32_t, 1u> reset{1u};
    const auto resetUpdated = observeCpu(
        tactile,
        model,
        movedBodies,
        held.penetrationDepthMeters,
        held.validity,
        held.objectShapeIds,
        2u,
        reset
    );
    require(
        std::ranges::all_of(
            resetUpdated.depthVelocityMetersPerSecond,
            [](const float velocity) {
                return velocity == 0.0f;
            }
        ),
        "environment reset did not clear tactile depth velocity"
    );
}

void validateTangentialMotion(
    const metalrobo::EngineModel& model
) {
    constexpr float maximumMotion = 0.0002f;
    const auto tactile = makeFlatSystem(
        model,
        1u,
        maximumMotion
    );
    auto initialBodies = sphereContactBodies(0.003f);
    const auto initial =
        observeCpu(tactile, model, initialBodies);
    const std::size_t center =
        (kHeight / 2u) * kWidth + kWidth / 2u;
    require(
        initial.objectShapeIds[center] == 1u &&
        initial.tangentialMotion[center].
                displacementAndVelocity.x == 0.0f &&
        initial.tangentialMotion[center].
                displacementAndVelocity.y == 0.0f,
        "contact onset did not establish a zero-motion anchor"
    );

    auto movedBodies = initialBodies;
    movedBodies[1u].position.x += 0.0005f;
    movedBodies[1u].linearVelocityAndInverseMass.x = 0.12f;
    movedBodies[1u].linearVelocityAndInverseMass.y = -0.04f;
    const auto moved = observeCpu(
        tactile,
        model,
        movedBodies,
        initial.penetrationDepthMeters,
        initial.validity,
        initial.objectShapeIds,
        1u,
        {},
        initial.tangentialMotion,
        initial.targetLocalContactAnchors
    );
    const mr_float4 movedMotion =
        moved.tangentialMotion[center].
            displacementAndVelocity;
    require(
        std::abs(
            std::hypot(movedMotion.x, movedMotion.y) -
            maximumMotion
        ) < 2.0e-7f,
        "tangential displacement did not saturate at the authored "
        "kinematic-proxy range"
    );
    require(
        std::abs(movedMotion.z - 0.12f) < 2.0e-6f &&
        std::abs(movedMotion.w + 0.04f) < 2.0e-6f,
        "flat-surface relative velocity was not projected into "
        "the cooked tangent frame"
    );
    require(
        std::abs(
            moved.summaries[0u].
                tangentialMotionAndFriction.y -
            maximumMotion
        ) < 2.0e-7f &&
        moved.summaries[0u].
                tangentialMotionAndFriction.x > 0.12f,
        "tangential motion reductions are incorrect"
    );

    auto coMovingBodies = initialBodies;
    coMovingBodies[0u].linearVelocityAndInverseMass.x = 0.08f;
    coMovingBodies[0u].linearVelocityAndInverseMass.y = -0.03f;
    coMovingBodies[1u].linearVelocityAndInverseMass.x = 0.08f;
    coMovingBodies[1u].linearVelocityAndInverseMass.y = -0.03f;
    const auto coMoving =
        observeCpu(tactile, model, coMovingBodies);
    require(
        std::abs(
            coMoving.tangentialMotion[center].
                displacementAndVelocity.z
        ) < 1.0e-7f &&
        std::abs(
            coMoving.tangentialMotion[center].
                displacementAndVelocity.w
        ) < 1.0e-7f,
        "co-moving sensor and target produced relative motion"
    );

    auto changedIdentityBodies = movedBodies;
    changedIdentityBodies[1u].position =
        {1.0f, 0.0f, 1.0f, 1.0f};
    changedIdentityBodies[4u] = makeBody({
        0.0005f,
        0.0f,
        kShell + 0.007f - 0.003f,
        1.0f,
    });
    const auto changedIdentity = observeCpu(
        tactile,
        model,
        changedIdentityBodies,
        moved.penetrationDepthMeters,
        moved.validity,
        moved.objectShapeIds,
        2u,
        {},
        moved.tangentialMotion,
        moved.targetLocalContactAnchors
    );
    require(
        changedIdentity.objectShapeIds[center] == 4u &&
        std::hypot(
            changedIdentity.tangentialMotion[center].
                displacementAndVelocity.x,
            changedIdentity.tangentialMotion[center].
                displacementAndVelocity.y
        ) < 1.0e-7f,
        "target identity change did not reset the contact anchor"
    );

    const std::array<std::uint32_t, 1u> reset{1u};
    const auto resetMotion = observeCpu(
        tactile,
        model,
        movedBodies,
        moved.penetrationDepthMeters,
        moved.validity,
        moved.objectShapeIds,
        2u,
        reset,
        moved.tangentialMotion,
        moved.targetLocalContactAnchors
    );
    require(
        std::hypot(
            resetMotion.tangentialMotion[center].
                displacementAndVelocity.x,
            resetMotion.tangentialMotion[center].
                displacementAndVelocity.y
        ) < 1.0e-7f,
        "environment reset did not reset the contact anchor"
    );

    metalrobo::MetalTactileConfig config;
    config.contactCapacityPerEnvironment = 1u;
    metalrobo::MetalTactileContext context(config);
    require(
        context.compile(tactile, model, 1u),
        "Metal tangential-motion compile"
    );
    metalrobo::MetalTactileHostFrame frame;
    frame.environmentCount = 1u;
    frame.bodies = initialBodies;
    frame.observationTimestepSeconds = 0.01f;
    frame.frameIndex = 0u;
    frame.timestampSeconds = 0.0;
    require(context.observe(frame), "Metal tangential-motion onset");
    frame.bodies = movedBodies;
    frame.frameIndex = 1u;
    frame.timestampSeconds = 0.01;
    require(context.observe(frame), "Metal tangential-motion update");
    metalrobo::TactileObservationBatch gpu;
    require(
        context.readback(1u, gpu),
        "Metal tangential-motion readback"
    );
    const mr_float4 gpuMotion =
        gpu.tangentialMotion[center].
            displacementAndVelocity;
    require(
        std::abs(gpuMotion.x - movedMotion.x) < 2.0e-6f &&
        std::abs(gpuMotion.y - movedMotion.y) < 2.0e-6f &&
        std::abs(gpuMotion.z - movedMotion.z) < 2.0e-6f &&
        std::abs(gpuMotion.w - movedMotion.w) < 2.0e-6f,
        "Metal tangential history differs from the CPU oracle"
    );
}

void validateWrenchAndCenterOfPressure(
    const metalrobo::EngineModel& model,
    const metalrobo::CookedTactileSystem& tactile,
    const std::vector<MRBodyStateGPU>& bodies
) {
    std::array<MRTactileContactGPU, 2u> contacts{};
    contacts[0u].shapesAndFlags = {
        0u,
        1u,
        MR_TACTILE_CONTACT_SOLVER_IMPULSE,
        0u,
    };
    contacts[0u].worldPoint = {
        0.004f,
        0.0f,
        kShell,
        0.0f,
    };
    contacts[0u].worldImpulseOnA = {
        0.010f,
        0.020f,
        0.030f,
        0.0f,
    };
    contacts[0u].solverImpulseAndFriction = {
        0.040f,
        0.010f,
        0.50f,
        0.40f,
    };
    contacts[1u].shapesAndFlags = {
        2u,
        0u,
        MR_TACTILE_CONTACT_SOLVER_IMPULSE,
        0u,
    };
    contacts[1u].worldPoint = {
        -0.002f,
        0.003f,
        kShell,
        0.0f,
    };
    contacts[1u].worldImpulseOnA = {
        0.005f,
        -0.010f,
        -0.015f,
        0.0f,
    };
    contacts[1u].solverImpulseAndFriction = {
        0.020f,
        0.020f,
        0.50f,
        0.40f,
    };
    const std::array<std::uint32_t, 1u> contactCounts{2u};
    metalrobo::TactileCpuFrame cpuFrame;
    cpuFrame.environmentCount = 1u;
    cpuFrame.bodies = bodies;
    cpuFrame.contacts = contacts;
    cpuFrame.contactCounts = contactCounts;
    cpuFrame.contactCapacityPerEnvironment = 2u;
    cpuFrame.observationTimestepSeconds = 0.020f;
    cpuFrame.contactImpulseTimestepSeconds = 0.005f;
    cpuFrame.frameIndex = 3u;
    cpuFrame.timestampSeconds = 1.5;
    metalrobo::TactileObservationBatch cpu;
    require(
        metalrobo::observeTactileCpuReference(
            tactile,
            model,
            cpuFrame,
            cpu
        ),
        "CPU tactile wrench and CoP"
    );
    const auto& summary = cpu.summaries[0u];
    const auto close = [](const float left, const float right) {
        return std::abs(left - right) < 2.0e-5f;
    };
    require(
        close(summary.netForceAndContactArea.x, 1.0f) &&
        close(summary.netForceAndContactArea.y, 6.0f) &&
        close(summary.netForceAndContactArea.z, 9.0f) &&
        close(summary.netTorqueAndMaximumDepth.x, 0.009f) &&
        close(summary.netTorqueAndMaximumDepth.y, -0.018f) &&
        close(summary.netTorqueAndMaximumDepth.z, 0.015f),
        "solver impulse interval did not produce the expected wrench"
    );
    require(
        (summary.statisticsAndIdentity.z &
         MR_TACTILE_SUMMARY_WRENCH_VALID) != 0u,
        "CPU summary did not mark supplied solver evidence valid"
    );
    require(
        close(
            summary.centerOfPressureLocalAndForceWeight.x,
            0.002f
        ) &&
        close(
            summary.centerOfPressureLocalAndForceWeight.y,
            0.001f
        ) &&
        close(
            summary.centerOfPressureLocalAndForceWeight.z,
            0.0f
        ) &&
        close(
            summary.centerOfPressureLocalAndForceWeight.w,
            3.0f * std::sqrt(14.0f)
        ),
        "force-weighted center of pressure is incorrect"
    );
    require(
        close(
            summary.tangentialMotionAndFriction.z,
            2.0f / 3.0f
        ) &&
        close(
            summary.tangentialMotionAndFriction.w,
            1.0f
        ),
        "solver friction utilization is incorrect"
    );

    metalrobo::MetalTactileConfig config;
    config.contactCapacityPerEnvironment = 2u;
    config.enableDebugHits = true;
    metalrobo::MetalTactileContext context(config);
    require(
        context.compile(tactile, model, 1u),
        "Metal tactile CoP compile"
    );
    metalrobo::MetalTactileHostFrame metalFrame;
    metalFrame.environmentCount = 1u;
    metalFrame.bodies = bodies;
    metalFrame.contacts = contacts;
    metalFrame.contactCounts = contactCounts;
    metalFrame.observationTimestepSeconds = 0.020f;
    metalFrame.contactImpulseTimestepSeconds = 0.005f;
    metalFrame.frameIndex = 3u;
    metalFrame.timestampSeconds = 1.5;
    require(
        context.observe(metalFrame),
        "Metal tactile wrench and CoP"
    );
    metalrobo::TactileObservationBatch gpu;
    require(
        context.readback(1u, gpu),
        "Metal tactile CoP readback"
    );
    const auto& gpuSummary = gpu.summaries[0u];
    require(
        gpu.debugHits.size() == tactile.samples.size() &&
        context.nativeBuffer(
            metalrobo::MetalTactileBuffer::debugHits
        ) != nullptr,
        "opt-in Metal tactile debug-hit stream is unavailable"
    );
    require(
        close(
            gpuSummary.netForceAndContactArea.x,
            summary.netForceAndContactArea.x
        ) &&
        close(
            gpuSummary.netForceAndContactArea.y,
            summary.netForceAndContactArea.y
        ) &&
        close(
            gpuSummary.netForceAndContactArea.z,
            summary.netForceAndContactArea.z
        ) &&
        close(
            gpuSummary.centerOfPressureLocalAndForceWeight.x,
            summary.centerOfPressureLocalAndForceWeight.x
        ) &&
        close(
            gpuSummary.centerOfPressureLocalAndForceWeight.y,
            summary.centerOfPressureLocalAndForceWeight.y
        ) &&
        close(
            gpuSummary.centerOfPressureLocalAndForceWeight.z,
            summary.centerOfPressureLocalAndForceWeight.z
        ) &&
        close(
            gpuSummary.tangentialMotionAndFriction.z,
            summary.tangentialMotionAndFriction.z
        ) &&
        close(
            gpuSummary.tangentialMotionAndFriction.w,
            summary.tangentialMotionAndFriction.w
        ),
        "Metal wrench or CoP differs from the CPU oracle"
    );
    require(
        (gpuSummary.statisticsAndIdentity.z &
         MR_TACTILE_SUMMARY_WRENCH_VALID) != 0u &&
        gpuSummary.statisticsAndIdentity.z ==
            summary.statisticsAndIdentity.z,
        "Metal solver-wrench validity differs from the CPU oracle"
    );

}

double validateMetal(
    const metalrobo::EngineModel& model,
    const metalrobo::CookedTactileSystem& tactile,
    const metalrobo::TactileObservationBatch& cpu,
    const std::vector<MRBodyStateGPU>& oneEnvironment,
    const std::uint32_t environmentCount,
    const std::uint32_t iterationCount
) {
    std::vector<MRBodyStateGPU> bodies;
    bodies.reserve(environmentCount * oneEnvironment.size());
    for (std::uint32_t environment = 0u;
         environment < environmentCount;
         ++environment) {
        bodies.insert(
            bodies.end(),
            oneEnvironment.begin(),
            oneEnvironment.end()
        );
    }
    metalrobo::MetalTactileConfig config;
    config.contactCapacityPerEnvironment = 4u;
    metalrobo::MetalTactileContext context(config);
    require(
        context.compile(tactile, model, environmentCount),
        "Metal tactile compile"
    );
    metalrobo::MetalTactileHostFrame frame;
    frame.environmentCount = environmentCount;
    frame.bodies = bodies;
    frame.observationTimestepSeconds = 0.01f;
    frame.frameIndex = 0u;
    frame.timestampSeconds = 2.0;
    metalrobo::MetalTactileDiagnostics measured;
    for (std::uint32_t iteration = 0u;
         iteration < iterationCount;
         ++iteration) {
        frame.frameIndex = iteration;
        frame.timestampSeconds =
            2.0 + 0.01 * static_cast<double>(iteration);
        measured = context.observe(frame);
        require(measured, "Metal tactile observe");
    }
    metalrobo::TactileObservationBatch gpu;
    require(
        context.readback(environmentCount, gpu),
        "Metal tactile readback"
    );
    double maximumDifference = 0.0;
    for (std::uint32_t index = 0u;
         index < tactile.samples.size();
         ++index) {
        maximumDifference = std::max(
            maximumDifference,
            static_cast<double>(std::abs(
                gpu.penetrationDepthMeters[index] -
                cpu.penetrationDepthMeters[index]
            ))
        );
        require(
            gpu.objectShapeIds[index] ==
                cpu.objectShapeIds[index],
            "Metal tactile object identity differs from CPU oracle"
        );
    }
    require(
        maximumDifference < 2.0e-6,
        "Metal tactile depth differs from CPU oracle by more than "
        "2 micrometres"
    );
    return maximumDifference;
}

} // namespace

int main(const int argc, const char* const* argv) {
    try {
        std::uint32_t environmentCount = 256u;
        std::uint32_t iterationCount = 5u;
        require(
            (argc - 1) % 2 == 0,
            "usage: metalrobo_tactile_check "
            "[--environments COUNT] [--iterations COUNT]"
        );
        for (int argument = 1; argument < argc; argument += 2) {
            const std::string_view option{argv[argument]};
            const std::string_view value{argv[argument + 1]};
            std::uint32_t* destination = nullptr;
            if (option == "--environments") {
                destination = &environmentCount;
            } else if (option == "--iterations") {
                destination = &iterationCount;
            } else {
                require(false, "unknown tactile-check option");
            }
            const auto parsed = std::from_chars(
                value.data(),
                value.data() + value.size(),
                *destination
            );
            require(
                parsed.ec == std::errc{} &&
                parsed.ptr == value.data() + value.size() &&
                *destination > 0u,
                "invalid tactile-check option value"
            );
        }
        const metalrobo::EngineModel model = makeModel();
        const auto tactile = makeFlatSystem(model);
        constexpr float penetration = 0.002f;
        const auto bodies = sphereContactBodies(penetration);
        const auto cpu = observeCpu(tactile, model, bodies);
        validateSphereField(tactile, cpu, penetration);
        validateRigidTransformInvariance(
            tactile,
            model,
            cpu,
            bodies
        );
        validateCylinderAndBox(tactile, model);
        validateMultipleContactsAndAtlasBoundary(
            tactile,
            model
        );
        validateCurvedSensor();
        validateCookedGeometryBackends();
        validateDecimationAndReset(model);
        validateTangentialMotion(model);
        validateWrenchAndCenterOfPressure(
            model,
            tactile,
            bodies
        );

        auto noContactBodies = sphereContactBodies(0.0f);
        noContactBodies[1].position =
            {0.0f, 0.0f, 0.2f, 1.0f};
        const auto noContact =
            observeCpu(tactile, model, noContactBodies);
        requireMetalMatchesCpu(
            tactile,
            model,
            noContactBodies,
            noContact,
            "no-contact"
        );
        require(
            std::ranges::all_of(
                noContact.penetrationDepthMeters,
                [](const float depth) { return depth == 0.0f; }
            ),
            "no-contact map contains non-zero depth"
        );

        const auto saturated = observeCpu(
            tactile,
            model,
            sphereContactBodies(0.008f)
        );
        const auto saturatedBodies =
            sphereContactBodies(0.008f);
        requireMetalMatchesCpu(
            tactile,
            model,
            saturatedBodies,
            saturated,
            "maximum-depth-saturation"
        );
        require(
            *std::max_element(
                saturated.penetrationDepthMeters.begin(),
                saturated.penetrationDepthMeters.end()
            ) == kShell,
            "maximum-depth saturation is incorrect"
        );
        require(
            std::ranges::any_of(
                saturated.validity,
                [](const std::uint32_t flags) {
                    return
                        (flags &
                         MR_TACTILE_VALIDITY_SATURATED) != 0u;
                }
            ),
            "saturated depth was not marked"
        );

        const double maximumMetalError = validateMetal(
            model,
            tactile,
            cpu,
            bodies,
            environmentCount,
            iterationCount
        );
        std::cout
            << "tactile_check: ok"
            << " max_cpu_gpu_error_m=" << maximumMetalError
            << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_tactile_check: "
                  << error.what() << '\n';
        return 1;
    }
}
