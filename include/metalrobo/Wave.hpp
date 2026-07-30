#pragma once

#include "metalrobo/RobotDescriptionCooker.hpp"
#include "metalrobo/Tactile.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

inline constexpr char kSharpaWaveUrdfRevision[] =
    "6eea427eb24189519f32b9f21674cd534d3f973c";
inline constexpr char kSharpaWaveTactileRevision[] =
    "865530a98a0ca0e69d177f2121833f8bb3ed94de";

enum class WaveHandSide : std::uint32_t {
    left = 0u,
    right = 1u,
};

enum class WaveAssetStatus : std::uint32_t {
    success = 0u,
    invalidConfiguration,
    sourceRevisionMismatch,
    ioFailure,
    invalidTactileArray,
    robotCookFailure,
    semanticMismatch,
    tactileCookFailure,
    internalFailure,
};

struct WaveAssetConfig {
    std::filesystem::path urdfRepositoryRoot;
    std::filesystem::path tactileRepositoryRoot;
    WaveHandSide side = WaveHandSide::right;
    std::uint32_t atlasWidth = 32u;
    std::uint32_t atlasHeight = 32u;
    // The published tactile map supplies geometry, not a compliance
    // calibration. Rigid backing is therefore the default. Enabling a shell
    // explicitly authors this value as both rest offset and depth bound.
    float maximumDepthMeters = 0.00515f;
    bool compliantShell = false;
    bool requirePinnedGitRevisions = true;
};

struct WaveHandAssets {
    EngineModel model;
    RobotDescriptionDiagnostics robotDiagnostics;
    std::vector<TactileSensorSpec> tactileSensors;
    // Published controller order, suitable for mapping one 22D hand slice
    // into the cooker's generalized-velocity order.
    std::array<std::uint32_t, 22u> controllerDofIndices{};
    // Exact column locations in the published 65D Origami state/action
    // vectors. These describe layout only; they do not promote the dataset
    // card's unverified action-command semantics.
    std::array<std::uint32_t, 22u> origamiStateIndices{};
    std::array<std::uint32_t, 22u> origamiActionIndices{};
    // Exact column locations in the published 60D fingertip wrench vector,
    // ordered thumb/index/middle/ring/little then fx/fy/fz/tx/ty/tz.
    // Units and sensor frames remain governed by the physical stream
    // contract and are not inferred from this map.
    std::array<std::uint32_t, 30u> origamiWrenchIndices{};
    std::uint64_t sourceFingerprint = 0u;
    std::string urdfRevision;
    std::string tactileRevision;
};

struct WavePairAssets {
    WaveHandAssets left;
    WaveHandAssets right;
    // Left then right hand, each in published 22-joint controller order.
    std::array<std::uint32_t, 44u> origamiStateIndices{};
    std::array<std::uint32_t, 44u> origamiActionIndices{};
    // Left then right fingertip wrench columns.
    std::array<std::uint32_t, 60u> origamiWrenchIndices{};
    std::uint64_t sourceFingerprint = 0u;
};

struct WaveAssetDiagnostics {
    WaveAssetStatus status = WaveAssetStatus::success;
    std::uint32_t validSampleCount = 0u;
    std::uint32_t sensorCount = 0u;
    std::uint64_t sourceFingerprint = 0u;
    std::string element;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == WaveAssetStatus::success;
    }
};

// Cooks one pinned, fixed-root 22-DoF hand and five 32x32 geometry atlases.
// Point maps are converted from authored millimetres to metres and attached
// to each elastomer body's COM-centred runtime frame.
[[nodiscard]] WaveAssetDiagnostics cookSharpaWaveHand(
    const WaveAssetConfig& config,
    WaveHandAssets& output
);

// Cooks both pinned hand assets and publishes the exact bimanual Origami
// column layout without inventing the unpublished arm/torso robot model.
[[nodiscard]] WaveAssetDiagnostics cookSharpaWavePair(
    const WaveAssetConfig& config,
    WavePairAssets& output
);

// Rebases sensor topology after the caller composes the hand with its own
// authoritative arm/world model. Empty targets retain automatic compatible
// target selection in the final model.
[[nodiscard]] std::vector<TactileSensorSpec>
rebaseSharpaWaveTactileSensors(
    std::span<const TactileSensorSpec> sensors,
    std::uint32_t bodyOffset,
    std::uint32_t shapeOffset,
    std::span<const std::uint32_t> targetShapeIndices = {}
);

[[nodiscard]] const char* waveAssetStatusName(
    WaveAssetStatus status
) noexcept;

} // namespace metalrobo
