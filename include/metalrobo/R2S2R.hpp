#pragma once

#include "metalrobo/WorldCompiler.hpp"
#include "metalrobo/r2s2r_types.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

inline constexpr std::uint32_t kHardwareOutcomeSchemaVersion = 1u;
inline constexpr std::size_t kMaximumAlignmentParticles = 4096u;
inline constexpr std::size_t kMaximumFeedbackRegions = 64u;

struct ScenarioFeatureSpec {
    std::string id;
    MRWorldVariationAxis axis = MR_WORLD_VARIATION_APPEARANCE;
    MRWorldDistributionKind distribution =
        MR_WORLD_DISTRIBUTION_CONSTANT;
    MRWorldVariationTarget target = MR_WORLD_TARGET_APPEARANCE_EXPOSURE;
    std::string targetId;
    mr_float4 parameters{};
    std::vector<std::uint32_t> categoricalValues;
    std::uint32_t ordinal = 0u;
};

struct ScenarioSchema {
    std::string id;
    std::uint64_t familyFingerprint = 0u;
    std::uint64_t programFingerprint = 0u;
    std::uint64_t fingerprint = 0u;
    std::vector<ScenarioFeatureSpec> features;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
    [[nodiscard]] std::uint32_t
    featureIndex(const std::string& id) const noexcept;
};

[[nodiscard]] ScenarioSchema compileScenarioSchema(
    const WorldFamily& family
);

struct WorldAlignmentParticle {
    // One normalized base-distribution quantile per ScenarioSchema feature.
    std::vector<float> quantiles;
    double weight = 0.0;
    double replayResidual = 0.0;
};

struct WorldAlignmentPopulation {
    std::string id;
    std::uint64_t schemaFingerprint = 0u;
    std::uint64_t fingerprint = 0u;
    std::vector<WorldAlignmentParticle> particles;

    [[nodiscard]] bool valid(
        const ScenarioSchema& schema,
        std::string* reason = nullptr
    ) const;
};

struct ReplayAlignmentConfig {
    std::uint32_t rounds = 4u;
    std::uint32_t maximumParticles = 4096u;
    double huberDelta = 1.0;
    double minimumEffectiveSampleFraction = 0.5;
    double jitterScale = 0.05;
    std::uint64_t seed = 1u;
};

// The evaluator may encode a complete replay batch on Metal/MLX. It receives
// an environment-major quantile tensor and must return one residual row per
// environment without changing the immutable anchor world.
using ReplayResidualEvaluator = std::function<bool(
    std::span<const float> candidateQuantiles,
    std::uint32_t candidateCount,
    std::uint32_t residualCount,
    std::vector<float>& replayResiduals,
    std::string* reason
)>;

// Four-round sequential Monte Carlo with robust replay likelihoods,
// effective-sample resampling, and bounded local jitter. Discontinuous contact
// modes remain separate weighted particles in the published artifact.
[[nodiscard]] bool fitAlignmentPopulationSMC(
    const ScenarioSchema& schema,
    std::span<const float> initialQuantiles,
    std::uint32_t candidateCount,
    std::uint32_t residualCount,
    const ReplayAlignmentConfig& config,
    const ReplayResidualEvaluator& evaluator,
    WorldAlignmentPopulation& output,
    std::string* reason = nullptr
);

// Builds a normalized, multimodal posterior from candidate quantile vectors
// and replay residual components. The candidate/residual rows are
// environment-major and must match. GPU/MLX callers use the same weighting
// semantics and publish through this artifact type.
[[nodiscard]] bool fitAlignmentPopulation(
    const ScenarioSchema& schema,
    std::span<const float> candidateQuantiles,
    std::span<const float> replayResiduals,
    std::uint32_t candidateCount,
    std::uint32_t residualCount,
    const ReplayAlignmentConfig& config,
    WorldAlignmentPopulation& output,
    std::string* reason = nullptr
);

struct FeedbackRegion {
    MRFeedbackRegionKind kind = MR_FEEDBACK_REGION_FAILURE;
    double weight = 0.0;
    std::vector<float> lowerQuantiles;
    std::vector<float> upperQuantiles;
};

struct WorldFeedbackProgram {
    std::string id;
    std::uint64_t schemaFingerprint = 0u;
    std::uint64_t taskFingerprint = 0u;
    std::uint64_t policyFingerprint = 0u;
    std::uint64_t sourceModelFingerprint = 0u;
    std::uint64_t fingerprint = 0u;
    double broadWeight = 0.5;
    double failureWeight = 0.3;
    double uncertaintyWeight = 0.2;
    std::vector<FeedbackRegion> regions;

    [[nodiscard]] bool valid(
        const ScenarioSchema& schema,
        std::string* reason = nullptr
    ) const;
};

// Seals a declarative feedback program after its model-produced regions have
// been populated. The fingerprint covers policy/task/model identity, mixture,
// and every quantile-space bound.
[[nodiscard]] bool finalizeWorldFeedbackProgram(
    const ScenarioSchema& schema,
    WorldFeedbackProgram& program,
    std::string* reason = nullptr
);

struct CompiledWorldSamplingProgram {
    std::uint64_t schemaFingerprint = 0u;
    std::uint64_t alignmentFingerprint = 0u;
    std::uint64_t feedbackFingerprint = 0u;
    double broadWeight = 1.0;
    double failureWeight = 0.0;
    double uncertaintyWeight = 0.0;
    float alignmentJitter = 0.05f;
    std::vector<MRWorldAlignmentParticleGPU> alignmentParticles;
    // Particle-major normalized quantiles.
    std::vector<float> alignmentQuantiles;
    std::vector<MRWorldFeedbackRegionGPU> feedbackRegions;
    // Region-major; x/y are lower/upper normalized quantiles.
    std::vector<mr_float4> feedbackBounds;

    [[nodiscard]] bool valid(
        const ScenarioSchema& schema,
        std::string* reason = nullptr
    ) const;
};

[[nodiscard]] bool compileWorldSamplingProgram(
    const ScenarioSchema& schema,
    const WorldAlignmentPopulation* alignment,
    const WorldFeedbackProgram* feedback,
    CompiledWorldSamplingProgram& output,
    std::string* reason = nullptr
);

struct PolicyDescriptor {
    std::string id;
    std::string contentHash;
    std::uint64_t fingerprint = 0u;
    std::uint64_t observationSchemaFingerprint = 0u;
    std::uint64_t actionSchemaFingerprint = 0u;
    std::uint64_t embodimentFingerprint = 0u;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

struct TaskOutcomeSchema {
    std::string id;
    std::uint64_t fingerprint = 0u;
    std::vector<std::string> failureTags;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

struct OutcomeArtifact {
    std::string kind;
    std::string uri;
    std::string contentHash;
};

struct EpisodeOutcome {
    std::string id;
    std::string runId;
    std::uint64_t scenarioKey = 0u;
    std::uint64_t episodeCounter = 0u;
    std::uint64_t familyFingerprint = 0u;
    std::uint64_t alignmentFingerprint = 0u;
    std::uint64_t feedbackFingerprint = 0u;
    std::uint64_t policyFingerprint = 0u;
    std::uint64_t taskFingerprint = 0u;
    std::uint64_t embodimentFingerprint = 0u;
    MREpisodeSource source = MR_EPISODE_SOURCE_SIMULATION;
    MREpisodeTermination termination = MR_EPISODE_TERMINATION_HORIZON;
    bool success = false;
    std::uint64_t failureMask = 0u;
    std::uint32_t physicsStatus = 0u;
    std::uint32_t stepCount = 0u;
    double episodeReturn = 0.0;
    double taskMargin = 0.0;
    double safetyMargin = 0.0;
    double durationSeconds = 0.0;
    double minimumVisibility = 0.0;
    double integratedContactLoad = 0.0;
    double peakContactLoad = 0.0;
    std::vector<float> scenarioValues;
    std::vector<std::uint8_t> scenarioValueMask;
    std::vector<OutcomeArtifact> artifacts;

    [[nodiscard]] bool valid(
        const ScenarioSchema* schema = nullptr,
        std::string* reason = nullptr
    ) const;
};

struct HardwareOutcomeManifest {
    std::uint32_t schemaVersion = kHardwareOutcomeSchemaVersion;
    std::string scenarioSchemaId;
    std::string policyId;
    std::string robotId;
    std::string taskId;
    std::vector<std::string> scenarioFeatureIds;
    std::vector<std::uint8_t> scenarioMissingMask;
    EpisodeOutcome outcome;
};

enum class R2S2RStatus : std::uint32_t {
    success = 0u,
    invalidArgument,
    invalidManifest,
    schemaMismatch,
    ioFailure,
    databaseFailure,
    unsupportedVersion,
    internalFailure,
};

struct R2S2RResult {
    R2S2RStatus status = R2S2RStatus::success;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == R2S2RStatus::success;
    }
};

[[nodiscard]] R2S2RResult loadHardwareOutcomeManifestJSON(
    const std::filesystem::path& path,
    const ScenarioSchema& schema,
    const TaskOutcomeSchema& task,
    HardwareOutcomeManifest& output
);

struct PolicyEvaluationSummary {
    std::uint64_t policyFingerprint = 0u;
    std::uint64_t simulationEpisodes = 0u;
    std::uint64_t hardwareEpisodes = 0u;
    double simulationSuccessRate = 0.0;
    std::optional<double> hardwareSuccessRate;
    std::optional<double> calibratedHardwareSuccessRate;
    std::vector<double> failureRates;
};

struct PolicyEvaluationReport {
    std::uint64_t taskFingerprint = 0u;
    std::uint64_t scenarioSchemaFingerprint = 0u;
    std::uint64_t pairedScenarioCount = 0u;
    std::vector<PolicyEvaluationSummary> policies;
    // Policy fingerprints ordered best-to-worst by calibrated hardware
    // success when available, otherwise by paired simulation success.
    std::vector<std::uint64_t> relativeOrdering;
};

[[nodiscard]] PolicyEvaluationReport evaluatePolicies(
    const TaskOutcomeSchema& task,
    const ScenarioSchema& scenarios,
    std::span<const EpisodeOutcome> outcomes
);

class R2S2RStore {
public:
    explicit R2S2RStore(std::filesystem::path path);
    ~R2S2RStore();

    R2S2RStore(R2S2RStore&&) noexcept;
    R2S2RStore& operator=(R2S2RStore&&) noexcept;
    R2S2RStore(const R2S2RStore&) = delete;
    R2S2RStore& operator=(const R2S2RStore&) = delete;

    [[nodiscard]] R2S2RResult open();
    [[nodiscard]] R2S2RResult registerScenarioSchema(
        const ScenarioSchema& schema
    );
    [[nodiscard]] R2S2RResult registerPolicy(
        const PolicyDescriptor& policy
    );
    [[nodiscard]] R2S2RResult appendOutcome(
        const EpisodeOutcome& outcome
    );
    [[nodiscard]] R2S2RResult appendOutcomes(
        std::span<const EpisodeOutcome> outcomes
    );
    [[nodiscard]] R2S2RResult loadOutcomes(
        std::uint64_t taskFingerprint,
        std::vector<EpisodeOutcome>& output
    ) const;
    [[nodiscard]] const std::filesystem::path& path() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] const char* r2s2rStatusName(R2S2RStatus status) noexcept;

} // namespace metalrobo
