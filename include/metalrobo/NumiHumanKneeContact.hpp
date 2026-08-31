#pragma once

#include "metalrobo/NumiHumanKnee.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

struct NumiHumanKneeContactMaterial {
    double elasticModulusPascals = 0.0;
    double poissonRatio = 0.0;
    double thicknessMeters = 0.0;
};

struct NumiHumanKneeContactRegionMaterial {
    std::uint32_t regionIndex = NUMI_HUMAN_KNEE_INVALID_INDEX;
    NumiHumanKneeContactMaterial material;
};

struct NumiHumanKneeContactSample {
    std::uint32_t slaveNode = NUMI_HUMAN_KNEE_INVALID_INDEX;
    std::array<std::uint32_t, 3u> masterNodes{};
    std::array<double, 3u> masterBarycentric{};
    std::array<double, 3u> referenceNormal{};
    double tributaryAreaSquareMeters = 0.0;
    double referenceSeparationMeters = 0.0;
};

struct NumiHumanKneeContactPairModel {
    std::string name;
    std::uint32_t sourcePairIndex = NUMI_HUMAN_KNEE_INVALID_INDEX;
    std::uint32_t masterRegionIndex = NUMI_HUMAN_KNEE_INVALID_INDEX;
    std::uint32_t slaveRegionIndex = NUMI_HUMAN_KNEE_INVALID_INDEX;
    std::uint32_t firstSample = 0u;
    std::uint32_t sampleCount = 0u;
    double effectiveFoundationStiffnessPascalsPerMeter = 0.0;
    double tributaryAreaSquareMeters = 0.0;
};

struct NumiHumanKneeContactModel {
    std::uint32_t nodeCount = 0u;
    std::vector<NumiHumanKneeContactPairModel> pairs;
    std::vector<NumiHumanKneeContactSample> samples;
};

struct NumiHumanKneeContactPairResult {
    std::string name;
    std::uint32_t activeSampleCount = 0u;
    double minimumGapChangeMeters = 0.0;
    double maximumPressurePascals = 0.0;
    double contactAreaSquareMeters = 0.0;
    double normalForceNewtons = 0.0;
    double storedEnergyJoules = 0.0;
};

struct NumiHumanKneeContactResult {
    std::vector<std::array<double, 3u>> nodalForcesNewtons;
    std::vector<NumiHumanKneeContactPairResult> pairs;
    std::array<double, 3u> forceResidualNewtons{};
    std::array<double, 3u> momentResidualNewtonMeters{};
    double forceL1Newtons = 0.0;
    double storedEnergyJoules = 0.0;
};

enum class NumiHumanKneeContactStatus : std::uint32_t {
    success = 0u,
    invalidInput,
    incompleteAnatomy,
    invalidMaterial,
    invalidTopology,
    numericalFailure,
};

struct NumiHumanKneeContactDiagnostics {
    NumiHumanKneeContactStatus status =
        NumiHumanKneeContactStatus::success;
    std::uint32_t failingIndex = NUMI_HUMAN_KNEE_INVALID_INDEX;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == NumiHumanKneeContactStatus::success;
    }
};

// Builds fixed closest-point correspondences for the seven authored Open
// Knee(s) cartilage/meniscus contact pairs. This is a small-deformation,
// frictionless elastic-foundation operator; it does not silently admit the
// payload's ligament collision pairs.
[[nodiscard]] NumiHumanKneeContactDiagnostics
buildNumiHumanKneeArticularContactModel(
    const NumiHumanKneePayload& payload,
    std::span<const std::array<double, 3u>> referenceWorldNodes,
    std::span<const NumiHumanKneeContactRegionMaterial> materials,
    NumiHumanKneeContactModel& model
);

// prescribedClosureMeters is an optional compressive test displacement. In a
// live solve it is zero and closure comes from currentWorldNodes relative to
// the reference correspondence. Positive pressure is compressive only.
[[nodiscard]] NumiHumanKneeContactDiagnostics evaluateNumiHumanKneeContact(
    const NumiHumanKneeContactModel& model,
    std::span<const std::array<double, 3u>> currentWorldNodes,
    double prescribedClosureMeters,
    NumiHumanKneeContactResult& result
);

[[nodiscard]] const char* numiHumanKneeContactStatusName(
    NumiHumanKneeContactStatus status
) noexcept;

} // namespace metalrobo
