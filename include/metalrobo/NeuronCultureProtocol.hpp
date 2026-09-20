#pragma once

#include "metalrobo/NeuronCulture.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace metalrobo {

inline constexpr float kPotterReferenceStimulationCurrent = 20000.0f;
inline constexpr std::uint32_t kPotterProtocolAlgorithmVersion = 6u;

enum class PotterProtocolAblation : std::uint32_t {
    none = 0u,
    adaptiveSelectionOff = 1u,
    patternedTrainingOff = 2u,
    stdpOff = 3u,
};

enum class PotterProtocolPhase : std::uint32_t {
    calibration = 0u,
    baseline = 1u,
    postSwitch = 2u,
    complete = 3u,
};

struct PotterProtocolConfig {
    std::uint64_t seed = 2056u;
    // Selects one of five independently authored CPS sensory sets. The
    // experimental perturbation remains the fixed Q1/Q3 exchange for all five.
    std::uint32_t sensoryMapping = 0u;
    std::uint32_t windowTicks = 5000u;
    std::uint32_t probeResponseTicks = 100u;
    std::uint32_t calibrationWindowsPerContext = 5u;
    std::uint32_t baselineWindows = 120u;
    // The reference experiment allowed four hours after the mapping switch.
    // One probe window is five seconds, so 2,880 windows is the canonical
    // ceiling. Sessions may complete earlier only after a full trailing
    // ten-minute interval reaches the fixed 90 percent goal criterion.
    std::uint32_t postSwitchWindows = 2880u;
    // A one-tick pulse must reliably evoke the local electrode population in
    // the deterministic LIF model. At 1 ms, 20,000 contributes 20 mV, enough
    // to cross the -50 mV threshold from the -68 mV reset potential without
    // inventing a host-side spike override.
    float stimulationCurrent = kPotterReferenceStimulationCurrent;
    float initialRadius = 24.0f;
    float goalRadius = 5.0f;
    float outerRadius = 50.0f;
};

struct PotterProtocolWindow {
    PotterProtocolPhase phase = PotterProtocolPhase::calibration;
    std::uint32_t index = 0u;
    std::uint32_t context = 0u;
    std::uint32_t encodedContext = 0u;
    // PTS pools are keyed by the exact preceding CPS identity, not by the
    // animat quadrant after sensory remapping.
    std::uint32_t trainingContext = 0xffffffffu;
    std::uint32_t trainingPattern = 0xffffffffu;
    // Exact movement that caused this PTS to be selected.  The next movement
    // is compared with this value; an older movement from the same quadrant
    // is not valid reinforcement credit.
    float trainingBaselineRadialDelta = 0.0f;
    NeuronCultureWindowRequest request;
};

struct PotterProtocolObservation {
    PotterProtocolPhase phase = PotterProtocolPhase::calibration;
    std::uint32_t window = 0u;
    std::uint32_t context = 0u;
    std::uint64_t spikes = 0u;
    float centerX = 0.0f;
    float centerY = 0.0f;
    float actionX = 0.0f;
    float actionY = 0.0f;
    float distanceBefore = 0.0f;
    float distanceAfter = 0.0f;
    bool inward = false;
};

struct PotterProtocolResult {
    std::uint64_t protocolFingerprint = 0u;
    std::uint32_t completedWindows = 0u;
    std::uint32_t baselineInward = 0u;
    std::uint32_t baselineMeasured = 0u;
    std::uint32_t postSwitchInward = 0u;
    std::uint32_t postSwitchMeasured = 0u;
    std::array<std::uint32_t, 4u> postSwitchContextInward{};
    std::array<std::uint32_t, 4u> postSwitchContextMeasured{};
    // The first ten minutes after Q1/Q3 exchange are kept separately from the
    // trailing promotion interval. This is a causal perturbation diagnostic:
    // a setup that never loses the old behavior has not demonstrated the
    // switched task, regardless of its eventual score.
    std::uint32_t switchIntervalInward = 0u;
    std::uint32_t switchIntervalMeasured = 0u;
    std::uint32_t finalIntervalInward = 0u;
    std::uint32_t finalIntervalMeasured = 0u;
    std::uint32_t patternedTrainingWindows = 0u;
    std::uint32_t randomBackgroundWindows = 0u;
    std::uint32_t patternReinforcements = 0u;
    std::uint32_t patternCopyRemovals = 0u;
    std::uint32_t trainingPoolCopies = 0u;
    std::uint32_t distinctWeightedPatterns = 0u;
    std::uint32_t maximumPatternMultiplicity = 0u;
    float baselineSuccess = 0.0f;
    float switchIntervalSuccess = 0.0f;
    // Success over the trailing ten-minute interval (or every available
    // post-switch window for bounded smoke configurations).
    float postSwitchSuccess = 0.0f;
    bool reachedGoal = false;
    bool switchGeometryValid = false;
    std::array<std::array<std::uint32_t, 3u>, 4u> contextElectrodes{};
    std::array<std::array<float, 2u>, 4u> calibrationMeanCenter{};
    std::array<float, 2u> calibrationCenterOffset{};
    std::array<std::array<float, 2u>, 4u> switchedMeanAction{};
    float finalX = 0.0f;
    float finalY = 0.0f;
};

struct PotterProtocolPairedTrial {
    std::uint64_t networkSeed = 0u;
    std::uint32_t sensoryMapping = 0u;
    float adaptiveSuccess = 0.0f;
    float patternedTrainingOffSuccess = 0.0f;
};

struct PotterProtocolQualification {
    std::uint64_t fingerprint = 0u;
    std::uint32_t trialCount = 0u;
    std::uint32_t bootstrapSamples = 0u;
    float meanImprovement = 0.0f;
    float lower95 = 0.0f;
    float requiredImprovement = 0.10f;
    bool promoted = false;
};

class PotterProtocolSession {
public:
    PotterProtocolSession(
        const CompiledNeuronCulture& culture,
        PotterProtocolConfig config = {},
        PotterProtocolAblation ablation = PotterProtocolAblation::none
    );

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] bool complete() const noexcept;
    [[nodiscard]] PotterProtocolPhase phase() const noexcept;
    [[nodiscard]] PotterProtocolWindow nextWindow();
    [[nodiscard]] bool observe(std::span<const std::uint32_t> cumulativeCounts);
    [[nodiscard]] const PotterProtocolObservation& lastObservation() const noexcept;
    [[nodiscard]] PotterProtocolResult result() const noexcept;

private:
    const CompiledNeuronCulture* culture_ = nullptr;
    PotterProtocolConfig config_{};
    PotterProtocolAblation ablation_ = PotterProtocolAblation::none;
    std::uint64_t rng_ = 0u;
    std::uint64_t fingerprint_ = 0u;
    std::uint32_t window_ = 0u;
    float x_ = 16.97056275f;
    float y_ = 16.97056275f;
    std::array<std::array<double, 2u>, 4u> calibrationSum_{};
    std::array<std::uint32_t, 4u> calibrationCount_{};
    std::array<float, 2u> centerOffset_{};
    std::array<std::array<float, 4u>, 4u> transforms_{};
    std::array<std::vector<std::uint16_t>, 4u> patternWeights_{};
    std::array<std::array<std::uint32_t, 3u>, 4u> contextElectrodes_{};
    std::array<std::array<std::uint32_t, 2u>, 4u> contextIntervals_{};
    std::array<std::uint32_t, 4u> repeatPattern_{
        0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
    std::vector<std::uint32_t> previousCounts_;
    std::vector<std::uint8_t> postSwitchHistory_;
    std::optional<PotterProtocolWindow> pendingWindow_;
    PotterProtocolObservation last_{};
    PotterProtocolResult result_{};
    bool earlyComplete_ = false;
};

[[nodiscard]] bool validPotterProtocolConfig(
    const PotterProtocolConfig& config
) noexcept;

// The promotion threshold is intentionally part of the implementation, not a
// caller-controlled parameter. Exactly three network seeds crossed with five
// sensory mappings are required, and promotion needs both a >=10 percentage
// point paired mean and a strictly positive deterministic bootstrap lower
// bound.
[[nodiscard]] PotterProtocolQualification qualifyPotterProtocol(
    std::span<const PotterProtocolPairedTrial> trials,
    std::uint32_t bootstrapSamples = 10000u,
    std::uint64_t bootstrapSeed = 0x504f545445525631ull
) noexcept;

} // namespace metalrobo
