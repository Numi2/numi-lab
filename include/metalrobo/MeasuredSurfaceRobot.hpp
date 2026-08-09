#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

constexpr std::uint32_t kMeasuredSurfaceActionCount = 24u;

enum class MeasuredSurfaceComponent : std::uint32_t {
    body = 1u,
    leftWing = 2u,
    rightWing = 3u,
    tail = 4u,
};

struct MeasuredSurfaceComponentRange {
    MeasuredSurfaceComponent component = MeasuredSurfaceComponent::body;
    std::uint32_t vertexOffset = 0u;
    std::uint32_t vertexCount = 0u;
    std::uint32_t triangleOffset = 0u;
    std::uint32_t triangleCount = 0u;
};

struct MeasuredSurfaceAction {
    std::string name;
    float lowerBound = -1.0f;
    float upperBound = 1.0f;
    float naturalFrequencyHertz = 12.0f;
    float dampingRatio = 0.85f;
};

// Robot semantics compiled over an immutable measured surface. The source
// positions and topology remain provenance-locked inputs; this pack only adds
// bounded actuation and physical root properties.
struct MeasuredSurfaceRobotPack {
    std::string id;
    std::string datasetIdentifier;
    std::string manifestSHA256;
    std::string positionsSHA256;
    std::string trianglesSHA256;
    std::uint32_t frameCount = 0u;
    std::uint32_t vertexCount = 0u;
    std::uint32_t triangleCount = 0u;
    float sampleRateHertz = 0.0f;
    bool sourcePeriodic = false;
    std::vector<MeasuredSurfaceComponentRange> components;
    std::array<MeasuredSurfaceAction, kMeasuredSurfaceActionCount> actions;
    float bodyMassKilograms = 0.35f;
    std::array<float, 3> principalInertiaKilogramMetersSquared {
        0.0018f, 0.0024f, 0.0031f
    };
    float airDensityKilogramsPerCubicMeter = 1.225f;
};

struct CompiledMeasuredSurfaceRobot {
    MeasuredSurfaceRobotPack pack;
    std::uint64_t fingerprint = 0u;
    std::vector<std::uint8_t> vertexComponents;
    std::vector<std::uint8_t> triangleComponents;
};

struct MeasuredSurfaceActuatorState {
    std::array<float, kMeasuredSurfaceActionCount> position {};
    std::array<float, kMeasuredSurfaceActionCount> velocity {};
};

[[nodiscard]] std::array<MeasuredSurfaceAction, kMeasuredSurfaceActionCount>
makeMeasuredSurfaceFlightActions();

[[nodiscard]] CompiledMeasuredSurfaceRobot compileMeasuredSurfaceRobot(
    const MeasuredSurfaceRobotPack& pack);

// One transactional actuator step. Throws before publishing if inputs or the
// candidate state are non-finite. Targets are clamped to the compiled bounds.
void stepMeasuredSurfaceActuators(
    const CompiledMeasuredSurfaceRobot& robot,
    std::span<const float, kMeasuredSurfaceActionCount> targets,
    float timeStepSeconds,
    MeasuredSurfaceActuatorState& state);

} // namespace metalrobo
