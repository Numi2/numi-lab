#include "metalrobo/NeuronCultureProtocol.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>

namespace metalrobo {
namespace {

constexpr std::uint64_t kOffset = 14695981039346656037ull;
constexpr std::uint64_t kPrime = 1099511628211ull;

template <typename T> void mix(std::uint64_t& hash, const T& value) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
    for (std::size_t i = 0u; i < sizeof(T); ++i) { hash ^= bytes[i]; hash *= kPrime; }
}

std::uint64_t random64(std::uint64_t& state) {
    state += 0x9e3779b97f4a7c15ull;
    std::uint64_t value = state;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

std::uint32_t contextFor(float x, float y) {
    if (x >= 0.0f && y >= 0.0f) return 0u;
    if (x < 0.0f && y >= 0.0f) return 1u;
    if (x < 0.0f && y < 0.0f) return 2u;
    return 3u;
}

std::array<float, 2u> desired(std::uint32_t context) {
    constexpr float s = 0.70710678118f;
    switch (context) {
        case 0u: return {-s, -s};
        case 1u: return {s, -s};
        case 2u: return {s, s};
        default: return {-s, s};
    }
}

// Chao, Bakkum, and Potter used five independently selected CPS sets per
// network, but the experimental perturbation itself was always the same:
// exchange Q1 and Q3 while leaving Q2 and Q4 unchanged. `sensoryMapping`
// therefore selects a deterministic CPS set; it is not a second permutation
// of the four embodied contexts.
std::uint32_t switchedContext(std::uint32_t context) {
    if (context == 0u) return 2u;
    if (context == 2u) return 0u;
    return context;
}

std::uint64_t pulseFingerprint(
    std::uint64_t protocol, std::uint32_t window, std::uint32_t ordinal
) {
    std::uint64_t hash = kOffset;
    mix(hash, protocol); mix(hash, window); mix(hash, ordinal);
    return hash == 0u ? kOffset : hash;
}

} // namespace

bool validPotterProtocolConfig(const PotterProtocolConfig& c) noexcept {
    const std::uint64_t calibration = static_cast<std::uint64_t>(c.calibrationWindowsPerContext) * 4u;
    const std::uint64_t total = calibration + c.baselineWindows + c.postSwitchWindows;
    return c.seed != 0u && c.sensoryMapping < 5u && c.windowTicks >= 1000u &&
        c.probeResponseTicks == 100u &&
        c.calibrationWindowsPerContext > 0u && c.baselineWindows > 0u &&
        c.postSwitchWindows > 0u && total <= 100000u && std::isfinite(c.stimulationCurrent) &&
        c.stimulationCurrent > 0.0f && c.stimulationCurrent <= 100000.0f &&
        std::isfinite(c.initialRadius) && std::isfinite(c.goalRadius) &&
        std::isfinite(c.outerRadius) && c.goalRadius > 0.0f &&
        c.initialRadius > c.goalRadius && c.outerRadius > c.initialRadius;
}

PotterProtocolSession::PotterProtocolSession(
    const CompiledNeuronCulture& culture,
    PotterProtocolConfig config,
    PotterProtocolAblation ablation
) : culture_(culture.valid() && culture.electrodes().size() == 60u ? &culture : nullptr),
    config_(config), ablation_(ablation), rng_(config.seed) {
    // Potter timing is specified in milliseconds. Keep that claim exact:
    // this v1 schedule intentionally admits only the reference 1 ms neural
    // tick rather than silently reinterpreting its 100/200...400
    // tick constants on a differently authored culture.
    if (!culture_ ||
        std::bit_cast<std::uint32_t>(culture_->header().neuralTimestepSeconds) !=
            std::bit_cast<std::uint32_t>(0.001f) ||
        !validPotterProtocolConfig(config_) ||
        static_cast<std::uint32_t>(ablation_) >
            static_cast<std::uint32_t>(PotterProtocolAblation::stdpOff)) {
        culture_ = nullptr;
        return;
    }
    x_ = config_.initialRadius * 0.70710678118f;
    y_ = x_;
    previousCounts_.assign(culture_->electrodes().size(), 0u);
    for (auto& weights : patternWeights_) weights.assign(60u * 11u, 1u);
    // Experimental mappings are authored from the trial/network seed. They
    // must not change when a dynamics ablation changes the culture runtime
    // fingerprint; otherwise an STDP-off pair receives different CPS input.
    std::uint64_t contextRng = config_.seed ^ culture_->header().seed ^
        (0x9e3779b97f4a7c15ull *
            (static_cast<std::uint64_t>(config_.sensoryMapping) + 1u));
    std::array<bool, 60u> probeUsed{};
    for (std::uint32_t context = 0u; context < 4u; ++context) {
        contextElectrodes_[context][0] =
            static_cast<std::uint32_t>(random64(contextRng) % 60u);
        do {
            contextElectrodes_[context][1] =
                static_cast<std::uint32_t>(random64(contextRng) % 60u);
        } while (contextElectrodes_[context][1] ==
                contextElectrodes_[context][0]);
        do {
            contextElectrodes_[context][2] =
                static_cast<std::uint32_t>(random64(contextRng) % 60u);
        } while (contextElectrodes_[context][2] ==
                contextElectrodes_[context][0] ||
            contextElectrodes_[context][2] ==
                contextElectrodes_[context][1] ||
            probeUsed[contextElectrodes_[context][2]]);
        probeUsed[contextElectrodes_[context][2]] = true;
        contextIntervals_[context][0] = 200u +
            static_cast<std::uint32_t>(random64(contextRng) % 201u);
        contextIntervals_[context][1] = 200u +
            static_cast<std::uint32_t>(random64(contextRng) % 201u);
        result_.contextElectrodes[context] = contextElectrodes_[context];
    }
    fingerprint_ = kOffset;
    mix(fingerprint_, culture_->fingerprint()); mix(fingerprint_, config_.seed);
    mix(fingerprint_, kPotterProtocolAlgorithmVersion);
    mix(fingerprint_, config_.sensoryMapping);
    mix(fingerprint_, config_.windowTicks); mix(fingerprint_, config_.probeResponseTicks);
    mix(fingerprint_, config_.calibrationWindowsPerContext);
    mix(fingerprint_, config_.baselineWindows); mix(fingerprint_, config_.postSwitchWindows);
    mix(fingerprint_, config_.stimulationCurrent); mix(fingerprint_, ablation_);
    if (fingerprint_ == 0u) fingerprint_ = kOffset;
    result_.protocolFingerprint = fingerprint_;
}

bool PotterProtocolSession::valid() const noexcept { return culture_ != nullptr; }

PotterProtocolPhase PotterProtocolSession::phase() const noexcept {
    if (!valid()) return PotterProtocolPhase::complete;
    const std::uint32_t calibration = 4u * config_.calibrationWindowsPerContext;
    if (window_ < calibration) return PotterProtocolPhase::calibration;
    if (window_ < calibration + config_.baselineWindows) return PotterProtocolPhase::baseline;
    if (window_ < calibration + config_.baselineWindows + config_.postSwitchWindows)
        return PotterProtocolPhase::postSwitch;
    return PotterProtocolPhase::complete;
}

bool PotterProtocolSession::complete() const noexcept {
    return earlyComplete_ || phase() == PotterProtocolPhase::complete;
}

PotterProtocolWindow PotterProtocolSession::nextWindow() {
    if (pendingWindow_) return *pendingWindow_;
    PotterProtocolWindow output;
    if (!valid() || complete()) return output;
    output.phase = phase();
    output.index = window_;
    const std::uint32_t calibration = 4u * config_.calibrationWindowsPerContext;
    output.context = output.phase == PotterProtocolPhase::calibration ?
        window_ % 4u : contextFor(x_, y_);
    output.encodedContext = output.context;
    if (output.phase == PotterProtocolPhase::postSwitch) {
        output.encodedContext = switchedContext(output.context);
    }
    output.request = {
        .cultureFingerprint = culture_->fingerprint(),
        .rootFingerprint = fingerprint_ ^ (static_cast<std::uint64_t>(window_) + 1u),
        .tickCount = config_.windowTicks,
        .recordingStartTick = 0u,
        .recordingDurationTicks = config_.probeResponseTicks,
        .plasticityEnabled = ablation_ != PotterProtocolAblation::stdpOff,
    };
    const auto& cps = contextElectrodes_[output.encodedContext];
    const auto& intervals = contextIntervals_[output.encodedContext];
    const std::uint32_t cpsDuration = intervals[0] + intervals[1];
    const std::uint32_t cpsStart = config_.windowTicks -
        config_.probeResponseTicks - cpsDuration - 1u;
    const std::array<std::uint32_t, 3u> cpsTicks{
        cpsStart, cpsStart + intervals[0], cpsStart + cpsDuration,
    };
    // The discrete LIF update consumes the probe current and emits its evoked
    // spikes within this tick. Starting at +1 discarded that local response
    // and decoded only the later network-wide tail. Treat the completed probe
    // tick as the first sample of the 100 ms post-probe response window.
    output.request.recordingStartTick = cpsTicks[2];
    std::uint32_t ordinal = 0u;
    const bool train = output.phase == PotterProtocolPhase::postSwitch &&
        last_.phase == PotterProtocolPhase::postSwitch && !last_.inward &&
        ablation_ != PotterProtocolAblation::patternedTrainingOff;
    if (train) {
        // The paper selects PTSQn from the pool associated with the preceding
        // CPSQn.  After the fixed Q1/Q3 sensory exchange, CPS identity and
        // physical quadrant identity differ.
        output.trainingContext = switchedContext(last_.context);
        output.trainingBaselineRadialDelta =
            last_.distanceAfter - last_.distanceBefore;
        auto& weights = patternWeights_[output.trainingContext];
        std::uint32_t pattern = repeatPattern_[output.trainingContext];
        if (ablation_ == PotterProtocolAblation::adaptiveSelectionOff) {
            // Preserve the full 660-pattern intervention and behavioral
            // contingency while removing only adaptive weighting/repetition.
            pattern = static_cast<std::uint32_t>(
                random64(rng_) % weights.size());
        } else if (pattern == 0xffffffffu) {
            std::uint64_t total = std::accumulate(
                weights.begin(), weights.end(), std::uint64_t{0});
            if (total == 0u) {
                patternWeights_[output.trainingContext].assign(660u, 1u);
                total = 660u;
            }
            std::uint64_t choice = random64(rng_) % total;
            pattern = 0u;
            while (pattern + 1u < weights.size() && choice >= weights[pattern]) {
                choice -= weights[pattern++];
            }
        }
        output.trainingPattern = pattern;
        const std::uint32_t second = pattern / 11u;
        const int offset = (static_cast<int>(pattern % 11u) - 5) * 20;
        const std::uint32_t trainingProbe =
            contextElectrodes_[output.trainingContext].back();
        std::uint64_t local = rng_ ^ (static_cast<std::uint64_t>(window_) << 32u);
        for (std::uint32_t start = 0u; start + 100u < cpsStart;) {
            const int a = static_cast<int>(start);
            const int b = a + offset;
            const std::uint32_t interval = 400u +
                static_cast<std::uint32_t>(random64(local) % 401u);
            if (b < 0) {
                start += interval;
                continue;
            }
            const std::uint32_t firstTick = static_cast<std::uint32_t>(std::min(a, b));
            const std::uint32_t secondTick = static_cast<std::uint32_t>(std::max(a, b));
            if (secondTick >= cpsStart) break;
            const std::uint32_t firstElectrode = offset < 0 ? second : trainingProbe;
            const std::uint32_t secondElectrode = offset < 0 ? trainingProbe : second;
            output.request.pulses.push_back({
                .electrode = firstElectrode, .startTick = firstTick, .durationTicks = 1u,
                .source = NeuronCultureStimulusSource::patternedTraining,
                .current = config_.stimulationCurrent,
                .sourceFingerprint = pulseFingerprint(fingerprint_, window_, ordinal++),
            });
            start += interval;
            output.request.pulses.push_back({
                .electrode = secondElectrode, .startTick = secondTick, .durationTicks = 1u,
                .source = NeuronCultureStimulusSource::patternedTraining,
                .current = config_.stimulationCurrent,
                .sourceFingerprint = pulseFingerprint(fingerprint_, window_, ordinal++),
            });
        }
    } else {
        // Potter-style random/shuffled background occupies the inter-probe
        // interval only when no PTS is requested. max(U[0,200], U[0,200])
        // yields a triangular 200...400 ms interval with a mean near 333 ms:
        // both the paper's stated range and its aggregate 3 Hz rate.
        std::uint64_t background = rng_ ^
            (static_cast<std::uint64_t>(window_) << 32u) ^ 0x524253u;
        for (std::uint32_t start = 0u; start < cpsStart;) {
            output.request.pulses.push_back({
                .electrode = static_cast<std::uint32_t>(random64(background) % 60u),
                .startTick = start, .durationTicks = 1u,
                .source = NeuronCultureStimulusSource::randomBackground,
                .current = config_.stimulationCurrent,
                .sourceFingerprint = pulseFingerprint(fingerprint_, window_, ordinal++),
            });
            const std::uint32_t a = static_cast<std::uint32_t>(
                random64(background) % 201u);
            const std::uint32_t b = static_cast<std::uint32_t>(
                random64(background) % 201u);
            start += 200u + std::max(a, b);
        }
    }
    // The context-control sequence follows PTS/SBS and ends with the probe;
    // the immediately following 100 ms is the sole motor-response window.
    for (std::uint32_t index = 0u; index < cps.size(); ++index) {
        output.request.pulses.push_back({
            .electrode = cps[index], .startTick = cpsTicks[index], .durationTicks = 1u,
            .source = NeuronCultureStimulusSource::contextProbe,
            .current = config_.stimulationCurrent,
            .sourceFingerprint = pulseFingerprint(fingerprint_, window_, ordinal++),
        });
    }
    std::stable_sort(output.request.pulses.begin(), output.request.pulses.end(),
        [](const auto& a, const auto& b) {
            if (a.startTick != b.startTick) return a.startTick < b.startTick;
            return a.electrode < b.electrode;
        });
    (void)calibration;
    pendingWindow_ = output;
    return *pendingWindow_;
}

bool PotterProtocolSession::observe(std::span<const std::uint32_t> cumulativeCounts) {
    if (!valid() || complete() || cumulativeCounts.size() != previousCounts_.size()) return false;
    const auto window = nextWindow();
    std::vector<std::uint32_t> delta(cumulativeCounts.size(), 0u);
    std::uint64_t total = 0u;
    double weightedX = 0.0;
    double weightedY = 0.0;
    for (std::size_t index = 0u; index < cumulativeCounts.size(); ++index) {
        if (cumulativeCounts[index] < previousCounts_[index]) return false;
        delta[index] = cumulativeCounts[index] - previousCounts_[index];
        total += delta[index];
        weightedX += static_cast<double>(delta[index]) * culture_->electrodes()[index].x;
        weightedY += static_cast<double>(delta[index]) * culture_->electrodes()[index].y;
    }
    previousCounts_.assign(cumulativeCounts.begin(), cumulativeCounts.end());
    const float centerX = total == 0u ? 0.0f : static_cast<float>(weightedX / total - 1.5);
    const float centerY = total == 0u ? 0.0f : static_cast<float>(weightedY / total - 1.5);
    last_ = {.phase = window.phase, .window = window.index, .context = window.context,
             .spikes = total, .centerX = centerX, .centerY = centerY};
    if (window.phase == PotterProtocolPhase::calibration) {
        calibrationSum_[window.context][0] += centerX;
        calibrationSum_[window.context][1] += centerY;
        ++calibrationCount_[window.context];
        if (window.index + 1u == 4u * config_.calibrationWindowsPerContext) {
            for (std::uint32_t context = 0u; context < 4u; ++context) {
                const double cx = calibrationSum_[context][0] / calibrationCount_[context];
                const double cy = calibrationSum_[context][1] / calibrationCount_[context];
                result_.calibrationMeanCenter[context] = {
                    static_cast<float>(cx), static_cast<float>(cy)};
                centerOffset_[0] += static_cast<float>(0.25 * cx);
                centerOffset_[1] += static_cast<float>(0.25 * cy);
            }
            // The deterministic model has a fixed spatial firing bias where
            // the source model used zero-mean Gaussian noise. Remove the
            // common four-CPS centroid before applying equation (3). This is
            // the CA-offset construction explicitly suggested by the paper
            // for making movement directions more uniformly decodable; it
            // does not use post-switch outcomes or alter neural state.
            result_.calibrationCenterOffset = centerOffset_;
            for (std::uint32_t context = 0u; context < 4u; ++context) {
                const double cx = result_.calibrationMeanCenter[context][0] -
                    centerOffset_[0];
                const double cy = result_.calibrationMeanCenter[context][1] -
                    centerOffset_[1];
                const auto target = desired(context);
                // Chao et al. equation (3) is a diagonal transformation with
                // two independent scale factors: [alpha*CAx, beta*CAy].  A
                // similarity rotation erases the intended geometry of the
                // Q1/Q3 sensory exchange and turns the perturbation into an
                // arbitrary direction. Fail closed when a CPS cannot define
                // both scale factors; such a response is not an admissible
                // motor mapping for this protocol.
                if (!std::isfinite(cx) || !std::isfinite(cy) ||
                    std::abs(cx) < 1.0e-9 || std::abs(cy) < 1.0e-9) {
                    earlyComplete_ = true;
                    return false;
                }
                transforms_[context] = {
                    static_cast<float>(target[0] / cx), 0.0f,
                    0.0f, static_cast<float>(target[1] / cy),
                };
            }
            for (std::uint32_t context = 0u; context < 4u; ++context) {
                const auto source = switchedContext(context);
                const auto& center = result_.calibrationMeanCenter[source];
                const auto& transform = transforms_[context];
                result_.switchedMeanAction[context] = {
                    transform[0] * (center[0] - centerOffset_[0]),
                    transform[3] * (center[1] - centerOffset_[1])};
            }
            const auto inwardProjection = [&](const std::uint32_t context) {
                const auto target = desired(context);
                const auto& action = result_.switchedMeanAction[context];
                return target[0] * action[0] + target[1] * action[1];
            };
            result_.switchGeometryValid = inwardProjection(0u) < 0.0f &&
                inwardProjection(2u) < 0.0f && inwardProjection(1u) > 0.0f &&
                inwardProjection(3u) > 0.0f;
        }
    } else {
        const auto& matrix = transforms_[window.context];
        last_.actionX = matrix[0] * (centerX - centerOffset_[0]);
        last_.actionY = matrix[3] * (centerY - centerOffset_[1]);
        if (!std::isfinite(last_.actionX) || !std::isfinite(last_.actionY)) {
            earlyComplete_ = true;
            return false;
        }
        last_.distanceBefore = std::hypot(x_, y_);
        x_ += last_.actionX;
        y_ += last_.actionY;
        last_.distanceAfter = std::hypot(x_, y_);
        last_.inward = last_.distanceAfter < last_.distanceBefore;
        if (last_.distanceAfter > config_.outerRadius) {
            constexpr double tau = 6.28318530717958647692;
            const float radius = config_.goalRadius * std::sqrt(
                static_cast<float>(static_cast<double>(random64(rng_) >> 11u) *
                    (1.0 / 9007199254740992.0)));
            const float angle = static_cast<float>(tau) *
                static_cast<float>(static_cast<double>(random64(rng_) >> 11u) *
                    (1.0 / 9007199254740992.0));
            x_ = radius * std::cos(angle);
            y_ = radius * std::sin(angle);
        }
        if (window.trainingContext < 4u && window.trainingPattern != 0xffffffffu &&
            ablation_ == PotterProtocolAblation::none) {
            auto& weight = patternWeights_[window.trainingContext][window.trainingPattern];
            const float radialDelta = last_.distanceAfter - last_.distanceBefore;
            if (radialDelta < window.trainingBaselineRadialDelta) {
                weight = static_cast<std::uint16_t>(std::min<unsigned>(65535u, weight + 1u));
                ++result_.patternReinforcements;
                // Repeat an improving PTS only while movement is still
                // outward. Once behavior is desirable, retain the added pool
                // copy but let a future failure draw from the weighted pool.
                repeatPattern_[window.trainingContext] = last_.inward ?
                    0xffffffffu : window.trainingPattern;
            } else {
                // Remove one surplus copy of a worsening PTS, but preserve its
                // original copy. This is the paper's bounded probability
                // update: every one of the 660 types remains discoverable.
                if (weight > 1u) {
                    --weight;
                    ++result_.patternCopyRemovals;
                }
                repeatPattern_[window.trainingContext] = 0xffffffffu;
            }
        }
        if (window.phase == PotterProtocolPhase::baseline) {
            ++result_.baselineMeasured;
            result_.baselineInward += last_.inward;
        } else {
            ++result_.postSwitchMeasured;
            result_.postSwitchInward += last_.inward;
            ++result_.postSwitchContextMeasured[window.context];
            result_.postSwitchContextInward[window.context] += last_.inward;
            constexpr std::uint32_t switchIntervalWindows = 120u;
            if (result_.switchIntervalMeasured < switchIntervalWindows) {
                ++result_.switchIntervalMeasured;
                result_.switchIntervalInward += last_.inward;
                result_.switchIntervalSuccess =
                    static_cast<float>(result_.switchIntervalInward) /
                    result_.switchIntervalMeasured;
            }
            if (window.trainingPattern == 0xffffffffu) {
                ++result_.randomBackgroundWindows;
            } else {
                ++result_.patternedTrainingWindows;
            }
            postSwitchHistory_.push_back(last_.inward ? 1u : 0u);
            constexpr std::size_t finalIntervalWindows = 120u;
            const std::size_t intervalBegin = postSwitchHistory_.size() >
                    finalIntervalWindows ?
                postSwitchHistory_.size() - finalIntervalWindows : 0u;
            result_.finalIntervalMeasured = static_cast<std::uint32_t>(
                postSwitchHistory_.size() - intervalBegin);
            result_.finalIntervalInward = std::accumulate(
                postSwitchHistory_.begin() + intervalBegin,
                postSwitchHistory_.end(), std::uint32_t{0});
            result_.postSwitchSuccess = result_.finalIntervalMeasured == 0u ? 0.0f :
                static_cast<float>(result_.finalIntervalInward) /
                    result_.finalIntervalMeasured;
            result_.reachedGoal = result_.finalIntervalMeasured ==
                    finalIntervalWindows &&
                result_.finalIntervalInward >= 108u;
            earlyComplete_ = result_.reachedGoal;
        }
    }
    pendingWindow_.reset();
    ++window_;
    result_.completedWindows = window_;
    result_.baselineSuccess = result_.baselineMeasured == 0u ? 0.0f :
        static_cast<float>(result_.baselineInward) / result_.baselineMeasured;
    if (result_.postSwitchMeasured == 0u) result_.postSwitchSuccess = 0.0f;
    result_.finalX = x_; result_.finalY = y_;
    return true;
}

const PotterProtocolObservation& PotterProtocolSession::lastObservation() const noexcept {
    return last_;
}

PotterProtocolResult PotterProtocolSession::result() const noexcept {
    auto result = result_;
    for (const auto& pool : patternWeights_) {
        for (const std::uint16_t copies : pool) {
            result.trainingPoolCopies += copies;
            result.distinctWeightedPatterns += copies > 1u;
            result.maximumPatternMultiplicity = std::max<std::uint32_t>(
                result.maximumPatternMultiplicity, copies);
        }
    }
    return result;
}

PotterProtocolQualification qualifyPotterProtocol(
    std::span<const PotterProtocolPairedTrial> trials,
    const std::uint32_t bootstrapSamples,
    const std::uint64_t bootstrapSeed
) noexcept {
    PotterProtocolQualification result;
    result.trialCount = static_cast<std::uint32_t>(trials.size());
    result.bootstrapSamples = bootstrapSamples;
    if (trials.size() != 15u || bootstrapSamples < 1000u ||
        bootstrapSamples > 1'000'000u || bootstrapSeed == 0u) {
        return result;
    }
    std::array<bool, 15u> seen{};
    std::array<std::uint64_t, 3u> seeds{};
    std::uint32_t seedCount = 0u;
    std::array<float, 15u> differences{};
    std::uint64_t fingerprint = kOffset;
    double sum = 0.0;
    for (std::size_t index = 0u; index < trials.size(); ++index) {
        const auto& trial = trials[index];
        if (trial.networkSeed == 0u || trial.sensoryMapping >= 5u ||
            !std::isfinite(trial.adaptiveSuccess) ||
            !std::isfinite(trial.patternedTrainingOffSuccess) ||
            trial.adaptiveSuccess < 0.0f || trial.adaptiveSuccess > 1.0f ||
            trial.patternedTrainingOffSuccess < 0.0f ||
            trial.patternedTrainingOffSuccess > 1.0f) {
            return PotterProtocolQualification{};
        }
        std::uint32_t seedOrdinal = seedCount;
        for (std::uint32_t item = 0u; item < seedCount; ++item) {
            if (seeds[item] == trial.networkSeed) {
                seedOrdinal = item;
                break;
            }
        }
        if (seedOrdinal == seedCount) {
            if (seedCount == seeds.size()) return PotterProtocolQualification{};
            seeds[seedCount++] = trial.networkSeed;
        }
        const std::uint32_t cell = seedOrdinal * 5u + trial.sensoryMapping;
        if (seen[cell]) return PotterProtocolQualification{};
        seen[cell] = true;
        differences[index] = trial.adaptiveSuccess -
            trial.patternedTrainingOffSuccess;
        sum += differences[index];
        mix(fingerprint, trial.networkSeed);
        mix(fingerprint, trial.sensoryMapping);
        mix(fingerprint, trial.adaptiveSuccess);
        mix(fingerprint, trial.patternedTrainingOffSuccess);
    }
    if (!std::all_of(seen.begin(), seen.end(), [](bool value) { return value; })) {
        return PotterProtocolQualification{};
    }
    result.meanImprovement = static_cast<float>(sum / differences.size());
    std::vector<float> means(bootstrapSamples);
    std::uint64_t rng = bootstrapSeed;
    for (float& mean : means) {
        double sample = 0.0;
        for (std::size_t draw = 0u; draw < differences.size(); ++draw) {
            sample += differences[random64(rng) % differences.size()];
        }
        mean = static_cast<float>(sample / differences.size());
    }
    std::sort(means.begin(), means.end());
    const std::size_t lowerIndex = static_cast<std::size_t>(
        std::floor(0.025 * static_cast<double>(means.size() - 1u)));
    result.lower95 = means[lowerIndex];
    mix(fingerprint, bootstrapSamples);
    mix(fingerprint, bootstrapSeed);
    result.fingerprint = fingerprint == 0u ? kOffset : fingerprint;
    result.promoted = result.meanImprovement >= result.requiredImprovement &&
        result.lower95 > 0.0f;
    return result;
}

} // namespace metalrobo
