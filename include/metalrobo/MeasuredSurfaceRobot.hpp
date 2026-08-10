#pragma once

#include "metalrobo/measured_surface_types.h"

#include <array>
#include <cstdint>
#include <filesystem>
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

enum class MeasuredSurfacePhaseBoundary : std::uint32_t {
    clamp = MR_MEASURED_SURFACE_PHASE_CLAMP,
    reflect = MR_MEASURED_SURFACE_PHASE_REFLECT,
    wrap = MR_MEASURED_SURFACE_PHASE_WRAP,
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
    // Live prefix of the fixed-capacity action ABI. Complete measured birds
    // use all 24 lanes; bilateral wing-only morphologies omit the four tail
    // lanes without inventing an unmeasured tail surface.
    std::uint32_t actionCount = kMeasuredSurfaceActionCount;
    float sampleRateHertz = 0.0f;
    bool sourcePeriodic = false;
    MeasuredSurfacePhaseBoundary phaseBoundary =
        MeasuredSurfacePhaseBoundary::clamp;
    std::vector<MeasuredSurfaceComponentRange> components;
    std::array<MeasuredSurfaceAction, kMeasuredSurfaceActionCount> actions;
    // Normalized residual center. All zeros preserve the exact measured
    // replay; recovery variants may author a qualified aerodynamic trim.
    std::array<float, kMeasuredSurfaceActionCount> normalizedActionBias {};
    float bodyMassKilograms = 0.35f;
    std::array<float, 3> principalInertiaKilogramMetersSquared {
        0.0018f, 0.0024f, 0.0031f
    };
    float airDensityKilogramsPerCubicMeter = 1.225f;
    float normalDragCoefficient = 1.15f;
    float tangentialDragCoefficient = 0.08f;
    // Local incidence ratio where separated-flow attenuation begins, retained
    // normal loading at full incidence, maximum near-ground lift increment,
    // and ground-effect height scale in spans.
    std::array<float, 4> aerodynamicCorrections {
        0.65f, 0.40f, 0.16f, 1.0f
    };
    // Immutable frame-major xyz payload and fixed triangle topology. These
    // are the robot morphology; compilation never remeshes or substitutes it.
    std::vector<float> frameMajorPositions;
    std::vector<std::uint16_t> triangleIndices;
    std::vector<float> frameTimesSeconds;
};

struct CompiledMeasuredSurfaceRobot {
    MeasuredSurfaceRobotPack pack;
    std::uint64_t fingerprint = 0u;
    std::vector<std::uint8_t> vertexComponents;
    std::vector<std::uint8_t> triangleComponents;
    MRMeasuredSurfaceModelGPU gpuModel{};
    std::array<MRMeasuredSurfaceActionGPU,
               kMeasuredSurfaceActionCount> gpuActions{};
    std::vector<MRMeasuredSurfaceComponentGPU> gpuComponents;
};

struct CompiledMeasuredSurfaceBinding {
    CompiledMeasuredSurfaceRobot robot;
    std::uint32_t articulationIndex = MR_INVALID_INDEX;
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    std::uint32_t qOffset = MR_INVALID_INDEX;
    std::uint32_t vOffset = MR_INVALID_INDEX;
    std::uint32_t firstAction = MR_INVALID_INDEX;
    std::uint64_t fingerprint = 0u;

    [[nodiscard]] bool valid() const noexcept {
        return robot.fingerprint != 0u &&
            articulationIndex != MR_INVALID_INDEX &&
            bodyIndex != MR_INVALID_INDEX &&
            qOffset != MR_INVALID_INDEX &&
            vOffset != MR_INVALID_INDEX &&
            firstAction != MR_INVALID_INDEX && fingerprint != 0u;
    }
};

struct MeasuredSurfaceActuatorState {
    std::array<float, kMeasuredSurfaceActionCount> position {};
    std::array<float, kMeasuredSurfaceActionCount> velocity {};
};

[[nodiscard]] std::array<MeasuredSurfaceAction, kMeasuredSurfaceActionCount>
makeMeasuredSurfaceFlightActions();

[[nodiscard]] std::array<float, kMeasuredSurfaceActionCount>
measuredSurfaceRecoveryTrimActions();

[[nodiscard]] MeasuredSurfaceRobotPack loadDeetjenMeasuredDoveRobotPack(
    const std::filesystem::path& manifestPath);

// Loads Numifly's immutable bilateral artifact: Maeda's measured right wing,
// a fingerprinted sagittal reflection for the left wing, and explicit robot
// scale/mount adaptations. The manifest remains measured-wing-only evidence;
// it is not a complete biological bird reconstruction.
[[nodiscard]] MeasuredSurfaceRobotPack loadNumiflyMaedaWingPack(
    const std::filesystem::path& manifestPath);

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
