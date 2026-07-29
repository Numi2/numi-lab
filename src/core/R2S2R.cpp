#include "metalrobo/R2S2R.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <numbers>
#include <numeric>
#include <random>
#include <sstream>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace metalrobo {
namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

bool failReason(std::string* reason, std::string message) {
    if (reason != nullptr) {
        *reason = std::move(message);
    }
    return false;
}

R2S2RResult fail(const R2S2RStatus status, std::string message) {
    return {status, std::move(message)};
}

class HashBuilder {
public:
    template <typename T>
    void append(const T& value) noexcept {
        static_assert(std::is_trivially_copyable_v<T>);
        appendBytes(&value, sizeof(T));
    }

    void appendString(const std::string& value) noexcept {
        const std::uint64_t size = value.size();
        append(size);
        appendBytes(value.data(), value.size());
    }

    template <typename T>
    void appendSpan(const std::span<const T> values) noexcept {
        static_assert(std::is_trivially_copyable_v<T>);
        const std::uint64_t size = values.size();
        append(size);
        appendBytes(values.data(), values.size_bytes());
    }

    [[nodiscard]] std::uint64_t finish() const noexcept {
        return value_ == 0u ? 1u : value_;
    }

private:
    void appendBytes(const void* data, const std::size_t size) noexcept {
        const auto* bytes = static_cast<const unsigned char*>(data);
        for (std::size_t index = 0u; index < size; ++index) {
            value_ ^= bytes[index];
            value_ *= kFnvPrime;
        }
    }

    std::uint64_t value_ = kFnvOffset;
};

std::string hex64(const std::uint64_t value) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << value;
    return stream.str();
}

bool parseHex64(const unsigned char* text, std::uint64_t& value) {
    if (text == nullptr) {
        return false;
    }
    const std::string input{reinterpret_cast<const char*>(text)};
    if (input.size() != 16u) {
        return false;
    }
    std::istringstream stream{input};
    stream >> std::hex >> value;
    return !stream.fail() && stream.eof();
}

bool finite(const double value) noexcept {
    return std::isfinite(value);
}

bool finite(const float value) noexcept {
    return std::isfinite(value);
}

std::uint64_t scenarioSchemaFingerprint(const ScenarioSchema& schema) {
    HashBuilder hash;
    hash.append(MR_R2S2R_ABI_VERSION);
    hash.appendString(schema.id);
    hash.append(schema.familyFingerprint);
    hash.append(schema.programFingerprint);
    for (const ScenarioFeatureSpec& feature : schema.features) {
        hash.appendString(feature.id);
        hash.append(feature.axis);
        hash.append(feature.distribution);
        hash.append(feature.target);
        hash.appendString(feature.targetId);
        hash.append(feature.parameters);
        hash.appendSpan<std::uint32_t>(feature.categoricalValues);
        hash.append(feature.ordinal);
    }
    return hash.finish();
}

std::uint64_t alignmentFingerprint(
    const WorldAlignmentPopulation& population
) {
    HashBuilder hash;
    hash.append(MR_R2S2R_ABI_VERSION);
    hash.appendString(population.id);
    hash.append(population.schemaFingerprint);
    for (const WorldAlignmentParticle& particle : population.particles) {
        hash.appendSpan<float>(particle.quantiles);
        hash.append(particle.weight);
        hash.append(particle.replayResidual);
    }
    return hash.finish();
}

double huber(const double residual, const double delta) {
    const double magnitude = std::abs(residual);
    return magnitude <= delta
        ? 0.5 * residual * residual
        : delta * (magnitude - 0.5 * delta);
}

std::uint64_t feedbackFingerprint(
    const WorldFeedbackProgram& feedback
) {
    HashBuilder hash;
    hash.append(MR_R2S2R_ABI_VERSION);
    hash.appendString(feedback.id);
    hash.append(feedback.schemaFingerprint);
    hash.append(feedback.taskFingerprint);
    hash.append(feedback.policyFingerprint);
    hash.append(feedback.sourceModelFingerprint);
    hash.append(feedback.broadWeight);
    hash.append(feedback.failureWeight);
    hash.append(feedback.uncertaintyWeight);
    for (const FeedbackRegion& region : feedback.regions) {
        hash.append(region.kind);
        hash.append(region.weight);
        hash.appendSpan<float>(region.lowerQuantiles);
        hash.appendSpan<float>(region.upperQuantiles);
    }
    return hash.finish();
}

R2S2RResult databaseError(sqlite3* database, std::string context) {
    const char* message =
        database == nullptr ? nullptr : sqlite3_errmsg(database);
    if (message != nullptr) {
        context += ": ";
        context += message;
    }
    return fail(R2S2RStatus::databaseFailure, std::move(context));
}

R2S2RResult execSQL(sqlite3* database, const char* sql) {
    char* error = nullptr;
    const int status = sqlite3_exec(database, sql, nullptr, nullptr, &error);
    if (status == SQLITE_OK) {
        return {};
    }
    std::string message =
        error == nullptr ? "SQLite statement failed" : error;
    sqlite3_free(error);
    return fail(R2S2RStatus::databaseFailure, std::move(message));
}

bool bindText(sqlite3_stmt* statement, const int index,
              const std::string& value) {
    return sqlite3_bind_text(
               statement,
               index,
               value.c_str(),
               static_cast<int>(value.size()),
               SQLITE_TRANSIENT
           ) == SQLITE_OK;
}

template <typename T>
bool bindVector(sqlite3_stmt* statement, const int index,
                const std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (values.empty()) {
        return sqlite3_bind_blob(
                   statement,
                   index,
                   nullptr,
                   0,
                   SQLITE_TRANSIENT
               ) == SQLITE_OK;
    }
    if (values.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max()) /
            sizeof(T)) {
        return false;
    }
    return sqlite3_bind_blob(
               statement,
               index,
               values.data(),
               static_cast<int>(values.size() * sizeof(T)),
               SQLITE_TRANSIENT
           ) == SQLITE_OK;
}

template <typename T>
bool columnVector(sqlite3_stmt* statement, const int column,
                  std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    const int bytes = sqlite3_column_bytes(statement, column);
    if (bytes < 0 || bytes % static_cast<int>(sizeof(T)) != 0) {
        return false;
    }
    const auto* data = static_cast<const T*>(
        sqlite3_column_blob(statement, column)
    );
    if (bytes == 0) {
        values.clear();
        return true;
    }
    if (data == nullptr) {
        return false;
    }
    values.assign(data, data + bytes / static_cast<int>(sizeof(T)));
    return true;
}

std::string featureNameBlob(const ScenarioSchema& schema) {
    std::string output;
    for (const ScenarioFeatureSpec& feature : schema.features) {
        output += feature.id;
        output.push_back('\n');
    }
    return output;
}

R2S2RResult appendOutcomeRecord(
    sqlite3* database,
    const EpisodeOutcome& outcome
) {
    static constexpr const char* sql = R"SQL(
        INSERT INTO outcomes (
            id, run_id, scenario_key, episode_counter,
            family_fingerprint, alignment_fingerprint,
            feedback_fingerprint, policy_fingerprint,
            task_fingerprint, embodiment_fingerprint,
            source, termination, success, failure_mask,
            physics_status, step_count, episode_return,
            task_margin, safety_margin, duration_seconds,
            minimum_visibility, integrated_contact_load,
            peak_contact_load, scenario_values, scenario_mask
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?
        )
    )SQL";
    sqlite3_stmt* statement = nullptr;
    if (sqlite3_prepare_v2(database, sql, -1, &statement, nullptr) !=
        SQLITE_OK) {
        return databaseError(database, "could not prepare outcome insert");
    }
    const auto finalize = [&statement]() {
        sqlite3_finalize(statement);
    };
    bool okay =
        bindText(statement, 1, outcome.id) &&
        bindText(statement, 2, outcome.runId) &&
        bindText(statement, 3, hex64(outcome.scenarioKey)) &&
        bindText(statement, 4, hex64(outcome.episodeCounter)) &&
        bindText(statement, 5, hex64(outcome.familyFingerprint)) &&
        bindText(statement, 6, hex64(outcome.alignmentFingerprint)) &&
        bindText(statement, 7, hex64(outcome.feedbackFingerprint)) &&
        bindText(statement, 8, hex64(outcome.policyFingerprint)) &&
        bindText(statement, 9, hex64(outcome.taskFingerprint)) &&
        bindText(statement, 10, hex64(outcome.embodimentFingerprint)) &&
        sqlite3_bind_int(statement, 11, outcome.source) == SQLITE_OK &&
        sqlite3_bind_int(statement, 12, outcome.termination) == SQLITE_OK &&
        sqlite3_bind_int(statement, 13, outcome.success ? 1 : 0) ==
            SQLITE_OK &&
        bindText(statement, 14, hex64(outcome.failureMask)) &&
        sqlite3_bind_int64(statement, 15, outcome.physicsStatus) ==
            SQLITE_OK &&
        sqlite3_bind_int64(statement, 16, outcome.stepCount) == SQLITE_OK &&
        sqlite3_bind_double(statement, 17, outcome.episodeReturn) ==
            SQLITE_OK &&
        sqlite3_bind_double(statement, 18, outcome.taskMargin) == SQLITE_OK &&
        sqlite3_bind_double(statement, 19, outcome.safetyMargin) ==
            SQLITE_OK &&
        sqlite3_bind_double(statement, 20, outcome.durationSeconds) ==
            SQLITE_OK &&
        sqlite3_bind_double(statement, 21, outcome.minimumVisibility) ==
            SQLITE_OK &&
        sqlite3_bind_double(
            statement,
            22,
            outcome.integratedContactLoad
        ) == SQLITE_OK &&
        sqlite3_bind_double(statement, 23, outcome.peakContactLoad) ==
            SQLITE_OK &&
        bindVector(statement, 24, outcome.scenarioValues) &&
        bindVector(statement, 25, outcome.scenarioValueMask);
    if (!okay) {
        finalize();
        return databaseError(database, "could not bind outcome");
    }
    if (sqlite3_step(statement) != SQLITE_DONE) {
        finalize();
        return databaseError(database, "could not insert outcome");
    }
    finalize();

    static constexpr const char* artifactSQL = R"SQL(
        INSERT INTO outcome_artifacts (
            outcome_id, ordinal, kind, uri, content_hash
        ) VALUES (?, ?, ?, ?, ?)
    )SQL";
    for (std::size_t ordinal = 0u;
         ordinal < outcome.artifacts.size();
         ++ordinal) {
        sqlite3_stmt* artifactStatement = nullptr;
        if (sqlite3_prepare_v2(
                database,
                artifactSQL,
                -1,
                &artifactStatement,
                nullptr
            ) != SQLITE_OK) {
            return databaseError(
                database,
                "could not prepare outcome artifact insert"
            );
        }
        const OutcomeArtifact& artifact = outcome.artifacts[ordinal];
        okay =
            bindText(artifactStatement, 1, outcome.id) &&
            sqlite3_bind_int64(
                artifactStatement,
                2,
                static_cast<sqlite3_int64>(ordinal)
            ) == SQLITE_OK &&
            bindText(artifactStatement, 3, artifact.kind) &&
            bindText(artifactStatement, 4, artifact.uri) &&
            bindText(artifactStatement, 5, artifact.contentHash);
        if (!okay || sqlite3_step(artifactStatement) != SQLITE_DONE) {
            sqlite3_finalize(artifactStatement);
            return databaseError(
                database,
                "could not insert outcome artifact"
            );
        }
        sqlite3_finalize(artifactStatement);
    }
    return {};
}

} // namespace

bool ScenarioSchema::valid(std::string* reason) const {
    if (reason != nullptr) {
        reason->clear();
    }
    if (id.empty() || familyFingerprint == 0u ||
        programFingerprint == 0u || fingerprint == 0u) {
        return failReason(reason, "scenario schema identity is empty");
    }
    std::unordered_set<std::string> ids;
    for (std::size_t index = 0u; index < features.size(); ++index) {
        const ScenarioFeatureSpec& feature = features[index];
        if (feature.id.empty() ||
            !ids.insert(feature.id).second ||
            feature.ordinal != index ||
            feature.axis > MR_WORLD_VARIATION_CAMERA ||
            feature.distribution > MR_WORLD_DISTRIBUTION_CATEGORICAL ||
            feature.target >
                MR_WORLD_TARGET_ASSET_COLLISION_ALTERNATIVE) {
            return failReason(
                reason,
                "scenario schema contains an invalid feature"
            );
        }
        if (feature.distribution == MR_WORLD_DISTRIBUTION_CATEGORICAL &&
            feature.categoricalValues.empty()) {
            return failReason(
                reason,
                "categorical scenario feature has no alternatives"
            );
        }
    }
    if (scenarioSchemaFingerprint(*this) != fingerprint) {
        return failReason(
            reason,
            "scenario schema fingerprint does not match its content"
        );
    }
    return true;
}

std::uint32_t ScenarioSchema::featureIndex(
    const std::string& id
) const noexcept {
    for (std::uint32_t index = 0u; index < features.size(); ++index) {
        if (features[index].id == id) {
            return index;
        }
    }
    return MR_INVALID_INDEX;
}

ScenarioSchema compileScenarioSchema(const WorldFamily& family) {
    ScenarioSchema schema;
    if (family.fingerprint == 0u ||
        family.program.fingerprint == 0u ||
        family.program.variationIds.size() !=
            family.program.variations.size() ||
        family.program.variationTargetIds.size() !=
            family.program.variations.size()) {
        return schema;
    }
    schema.id = family.program.id + ".scenarios";
    schema.familyFingerprint = family.fingerprint;
    schema.programFingerprint = family.program.fingerprint;
    schema.features.reserve(family.program.variations.size());
    for (std::uint32_t ordinal = 0u;
         ordinal < family.program.variations.size();
         ++ordinal) {
        const MRWorldVariationGPU& descriptor =
            family.program.variations[ordinal];
        ScenarioFeatureSpec feature;
        feature.id = family.program.variationIds[ordinal];
        feature.axis = static_cast<MRWorldVariationAxis>(
            descriptor.binding.x
        );
        feature.distribution = static_cast<MRWorldDistributionKind>(
            descriptor.binding.y
        );
        feature.target = static_cast<MRWorldVariationTarget>(
            descriptor.binding.z
        );
        feature.targetId =
            family.program.variationTargetIds[ordinal];
        feature.parameters = descriptor.parameters;
        feature.ordinal = ordinal;
        if (feature.distribution ==
            MR_WORLD_DISTRIBUTION_CATEGORICAL) {
            const std::size_t first = descriptor.categorical.x;
            const std::size_t count = descriptor.categorical.y;
            if (first > family.program.categoricalValues.size() ||
                count > family.program.categoricalValues.size() - first) {
                return {};
            }
            feature.categoricalValues.assign(
                family.program.categoricalValues.begin() + first,
                family.program.categoricalValues.begin() + first + count
            );
        }
        schema.features.push_back(std::move(feature));
    }
    schema.fingerprint = scenarioSchemaFingerprint(schema);
    return schema;
}

bool WorldAlignmentPopulation::valid(
    const ScenarioSchema& schema,
    std::string* reason
) const {
    if (reason != nullptr) {
        reason->clear();
    }
    if (!schema.valid(reason)) {
        return false;
    }
    if (id.empty() || schemaFingerprint != schema.fingerprint ||
        fingerprint == 0u || particles.empty() ||
        particles.size() > kMaximumAlignmentParticles) {
        return failReason(
            reason,
            "alignment population identity or size is invalid"
        );
    }
    double weightSum = 0.0;
    for (const WorldAlignmentParticle& particle : particles) {
        if (particle.quantiles.size() != schema.features.size() ||
            !finite(particle.weight) || particle.weight < 0.0 ||
            !finite(particle.replayResidual) ||
            particle.replayResidual < 0.0) {
            return failReason(
                reason,
                "alignment particle dimensions or scores are invalid"
            );
        }
        for (const float quantile : particle.quantiles) {
            if (!finite(quantile) || quantile < 0.0f ||
                quantile > 1.0f) {
                return failReason(
                    reason,
                    "alignment particle quantile is outside [0,1]"
                );
            }
        }
        weightSum += particle.weight;
    }
    if (!finite(weightSum) || std::abs(weightSum - 1.0) > 1.0e-6) {
        return failReason(
            reason,
            "alignment particle weights are not normalized"
        );
    }
    if (alignmentFingerprint(*this) != fingerprint) {
        return failReason(
            reason,
            "alignment population fingerprint does not match its content"
        );
    }
    return true;
}

bool fitAlignmentPopulation(
    const ScenarioSchema& schema,
    const std::span<const float> candidateQuantiles,
    const std::span<const float> replayResiduals,
    const std::uint32_t candidateCount,
    const std::uint32_t residualCount,
    const ReplayAlignmentConfig& config,
    WorldAlignmentPopulation& output,
    std::string* reason
) {
    if (reason != nullptr) {
        reason->clear();
    }
    if (!schema.valid(reason) || candidateCount == 0u ||
        residualCount == 0u || config.rounds == 0u ||
        config.maximumParticles == 0u ||
        config.maximumParticles > kMaximumAlignmentParticles ||
        !finite(config.huberDelta) || !(config.huberDelta > 0.0) ||
        !finite(config.minimumEffectiveSampleFraction) ||
        !(config.minimumEffectiveSampleFraction > 0.0 &&
          config.minimumEffectiveSampleFraction <= 1.0) ||
        !finite(config.jitterScale) || config.jitterScale < 0.0) {
        return failReason(
            reason,
            "alignment configuration is invalid"
        );
    }
    const std::size_t featureCount = schema.features.size();
    if (featureCount == 0u ||
        candidateQuantiles.size() !=
            static_cast<std::size_t>(candidateCount) * featureCount ||
        replayResiduals.size() !=
            static_cast<std::size_t>(candidateCount) * residualCount) {
        return failReason(
            reason,
            "alignment candidate or residual tensor has the wrong shape"
        );
    }

    std::vector<double> losses(candidateCount, 0.0);
    for (std::uint32_t candidate = 0u;
         candidate < candidateCount;
         ++candidate) {
        for (std::size_t feature = 0u;
             feature < featureCount;
             ++feature) {
            const float value =
                candidateQuantiles[candidate * featureCount + feature];
            if (!finite(value) || value < 0.0f || value > 1.0f) {
                return failReason(
                    reason,
                    "alignment candidate quantile is outside [0,1]"
                );
            }
        }
        double loss = 0.0;
        for (std::uint32_t residual = 0u;
             residual < residualCount;
             ++residual) {
            const float value =
                replayResiduals[candidate * residualCount + residual];
            if (!finite(value)) {
                return failReason(
                    reason,
                    "alignment replay residual is non-finite"
                );
            }
            loss += huber(value, config.huberDelta);
        }
        losses[candidate] = loss;
    }

    std::vector<double> scaleValues = losses;
    const auto median = scaleValues.begin() + scaleValues.size() / 2u;
    std::nth_element(scaleValues.begin(), median, scaleValues.end());
    const double scale = std::max(*median, 1.0e-9);
    std::vector<double> logWeights(candidateCount, 0.0);
    std::vector<double> weights(candidateCount, 0.0);
    for (std::uint32_t round = 0u; round < config.rounds; ++round) {
        const double deltaBeta = 1.0 / config.rounds;
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::uint32_t candidate = 0u;
             candidate < candidateCount;
             ++candidate) {
            logWeights[candidate] -=
                deltaBeta * losses[candidate] / scale;
            maximum = std::max(maximum, logWeights[candidate]);
        }
        double sum = 0.0;
        for (std::uint32_t candidate = 0u;
             candidate < candidateCount;
             ++candidate) {
            weights[candidate] =
                std::exp(logWeights[candidate] - maximum);
            sum += weights[candidate];
        }
        if (!finite(sum) || !(sum > 0.0)) {
            return failReason(
                reason,
                "alignment likelihood normalization failed"
            );
        }
        for (double& weight : weights) {
            weight /= sum;
        }
    }

    std::vector<std::uint32_t> order(candidateCount);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(
        order.begin(),
        order.end(),
        [&weights, &losses](const std::uint32_t left,
                            const std::uint32_t right) {
            if (weights[left] != weights[right]) {
                return weights[left] > weights[right];
            }
            return losses[left] < losses[right];
        }
    );
    order.resize(std::min<std::size_t>(
        order.size(),
        config.maximumParticles
    ));
    double retainedWeight = 0.0;
    for (const std::uint32_t index : order) {
        retainedWeight += weights[index];
    }
    if (!(retainedWeight > 0.0)) {
        return failReason(
            reason,
            "alignment posterior retained no probability mass"
        );
    }

    WorldAlignmentPopulation staged;
    staged.schemaFingerprint = schema.fingerprint;
    staged.particles.reserve(order.size());
    for (const std::uint32_t index : order) {
        WorldAlignmentParticle particle;
        const auto begin =
            candidateQuantiles.begin() + index * featureCount;
        particle.quantiles.assign(begin, begin + featureCount);
        particle.weight = weights[index] / retainedWeight;
        particle.replayResidual = losses[index];
        staged.particles.push_back(std::move(particle));
    }
    HashBuilder idHash;
    idHash.append(schema.fingerprint);
    idHash.append(config.seed);
    idHash.append(candidateCount);
    idHash.append(residualCount);
    staged.id = "alignment." + hex64(idHash.finish());
    staged.fingerprint = alignmentFingerprint(staged);
    if (!staged.valid(schema, reason)) {
        return false;
    }
    output = std::move(staged);
    return true;
}

bool fitAlignmentPopulationSMC(
    const ScenarioSchema& schema,
    const std::span<const float> initialQuantiles,
    const std::uint32_t candidateCount,
    const std::uint32_t residualCount,
    const ReplayAlignmentConfig& config,
    const ReplayResidualEvaluator& evaluator,
    WorldAlignmentPopulation& output,
    std::string* reason
) {
    if (reason != nullptr) {
        reason->clear();
    }
    const std::size_t featureCount = schema.features.size();
    if (!schema.valid(reason) || candidateCount == 0u ||
        candidateCount > config.maximumParticles ||
        candidateCount > kMaximumAlignmentParticles ||
        residualCount == 0u || config.rounds == 0u ||
        config.maximumParticles == 0u ||
        config.maximumParticles > kMaximumAlignmentParticles ||
        !finite(config.huberDelta) || !(config.huberDelta > 0.0) ||
        !finite(config.minimumEffectiveSampleFraction) ||
        !(config.minimumEffectiveSampleFraction > 0.0 &&
          config.minimumEffectiveSampleFraction <= 1.0) ||
        !finite(config.jitterScale) || config.jitterScale < 0.0 ||
        !evaluator ||
        initialQuantiles.size() !=
            static_cast<std::size_t>(candidateCount) * featureCount) {
        return failReason(
            reason,
            "SMC alignment inputs or configuration are invalid"
        );
    }
    for (const float quantile : initialQuantiles) {
        if (!finite(quantile) || quantile < 0.0f ||
            quantile > 1.0f) {
            return failReason(
                reason,
                "SMC initial quantile is outside [0,1]"
            );
        }
    }

    std::vector<float> particles{
        initialQuantiles.begin(),
        initialQuantiles.end(),
    };
    std::vector<float> residuals;
    std::vector<double> losses(candidateCount, 0.0);
    std::vector<double> weights(
        candidateCount,
        1.0 / static_cast<double>(candidateCount)
    );
    std::vector<float> resampled(particles.size(), 0.0f);
    std::vector<double> cumulative(candidateCount, 0.0);
    std::mt19937_64 generator{config.seed};
    std::uniform_real_distribution<double> uniform{0.0, 1.0};

    for (std::uint32_t round = 0u;
         round < config.rounds;
         ++round) {
        residuals.clear();
        std::string evaluatorReason;
        if (!evaluator(
                particles,
                candidateCount,
                residualCount,
                residuals,
                &evaluatorReason
            ) ||
            residuals.size() !=
                static_cast<std::size_t>(candidateCount) *
                    residualCount) {
            return failReason(
                reason,
                evaluatorReason.empty()
                    ? "SMC replay evaluator returned the wrong shape"
                    : "SMC replay evaluator failed: " + evaluatorReason
            );
        }
        for (std::uint32_t candidate = 0u;
             candidate < candidateCount;
             ++candidate) {
            double loss = 0.0;
            for (std::uint32_t residual = 0u;
                 residual < residualCount;
                 ++residual) {
                const float value =
                    residuals[
                        static_cast<std::size_t>(candidate) *
                            residualCount +
                        residual
                    ];
                if (!finite(value)) {
                    return failReason(
                        reason,
                        "SMC replay residual is non-finite"
                    );
                }
                loss += huber(value, config.huberDelta);
            }
            losses[candidate] = loss;
        }

        std::vector<double> scaleValues = losses;
        auto median =
            scaleValues.begin() + scaleValues.size() / 2u;
        std::nth_element(
            scaleValues.begin(),
            median,
            scaleValues.end()
        );
        const double scale = std::max(*median, 1.0e-9);
        const double deltaBeta =
            1.0 / static_cast<double>(config.rounds);
        double maximumLogWeight =
            -std::numeric_limits<double>::infinity();
        std::vector<double> logWeights(candidateCount, 0.0);
        for (std::uint32_t candidate = 0u;
             candidate < candidateCount;
             ++candidate) {
            logWeights[candidate] =
                std::log(std::max(weights[candidate], 1.0e-300)) -
                deltaBeta * losses[candidate] / scale;
            maximumLogWeight = std::max(
                maximumLogWeight,
                logWeights[candidate]
            );
        }
        double weightSum = 0.0;
        for (std::uint32_t candidate = 0u;
             candidate < candidateCount;
             ++candidate) {
            weights[candidate] = std::exp(
                logWeights[candidate] - maximumLogWeight
            );
            weightSum += weights[candidate];
        }
        if (!finite(weightSum) || !(weightSum > 0.0)) {
            return failReason(
                reason,
                "SMC likelihood normalization failed"
            );
        }
        double squaredWeightSum = 0.0;
        for (double& weight : weights) {
            weight /= weightSum;
            squaredWeightSum += weight * weight;
        }
        const double effectiveSampleSize =
            1.0 / squaredWeightSum;
        const double resampleThreshold =
            config.minimumEffectiveSampleFraction *
            static_cast<double>(candidateCount);
        if (round + 1u == config.rounds ||
            effectiveSampleSize >= resampleThreshold) {
            continue;
        }

        std::partial_sum(
            weights.begin(),
            weights.end(),
            cumulative.begin()
        );
        cumulative.back() = 1.0;
        const double stride =
            1.0 / static_cast<double>(candidateCount);
        const double offset = uniform(generator) * stride;
        std::uint32_t ancestor = 0u;
        const double jitter =
            config.jitterScale /
            std::sqrt(static_cast<double>(round + 1u));
        for (std::uint32_t candidate = 0u;
             candidate < candidateCount;
             ++candidate) {
            const double selector =
                offset + static_cast<double>(candidate) * stride;
            while (ancestor + 1u < candidateCount &&
                   selector > cumulative[ancestor]) {
                ++ancestor;
            }
            for (std::size_t feature = 0u;
                 feature < featureCount;
                 ++feature) {
                const double u1 =
                    std::max(uniform(generator), 1.0e-12);
                const double u2 = uniform(generator);
                const double normal =
                    std::sqrt(-2.0 * std::log(u1)) *
                    std::cos(
                        2.0 * std::numbers::pi * u2
                    );
                const float source =
                    particles[
                        static_cast<std::size_t>(ancestor) *
                            featureCount +
                        feature
                    ];
                resampled[
                    static_cast<std::size_t>(candidate) *
                        featureCount +
                    feature
                ] = static_cast<float>(std::clamp(
                    static_cast<double>(source) + jitter * normal,
                    0.0,
                    1.0
                ));
            }
        }
        particles.swap(resampled);
        std::fill(
            weights.begin(),
            weights.end(),
            1.0 / static_cast<double>(candidateCount)
        );
    }

    std::vector<std::uint32_t> order(candidateCount);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(
        order.begin(),
        order.end(),
        [&weights, &losses](
            const std::uint32_t left,
            const std::uint32_t right
        ) {
            if (weights[left] != weights[right]) {
                return weights[left] > weights[right];
            }
            return losses[left] < losses[right];
        }
    );
    WorldAlignmentPopulation staged;
    staged.schemaFingerprint = schema.fingerprint;
    staged.particles.reserve(candidateCount);
    double normalizedSum = 0.0;
    for (const std::uint32_t index : order) {
        WorldAlignmentParticle particle;
        const auto begin =
            particles.begin() +
            static_cast<std::size_t>(index) * featureCount;
        particle.quantiles.assign(begin, begin + featureCount);
        particle.weight = weights[index];
        particle.replayResidual = losses[index];
        normalizedSum += particle.weight;
        staged.particles.push_back(std::move(particle));
    }
    if (!(normalizedSum > 0.0)) {
        return failReason(
            reason,
            "SMC posterior retained no probability mass"
        );
    }
    for (WorldAlignmentParticle& particle : staged.particles) {
        particle.weight /= normalizedSum;
    }
    HashBuilder idHash;
    idHash.append(schema.fingerprint);
    idHash.append(config.seed);
    idHash.append(config.rounds);
    idHash.append(candidateCount);
    idHash.append(residualCount);
    staged.id = "alignment.smc." + hex64(idHash.finish());
    staged.fingerprint = alignmentFingerprint(staged);
    if (!staged.valid(schema, reason)) {
        return false;
    }
    output = std::move(staged);
    return true;
}

bool WorldFeedbackProgram::valid(
    const ScenarioSchema& schema,
    std::string* reason
) const {
    if (reason != nullptr) {
        reason->clear();
    }
    if (!schema.valid(reason)) {
        return false;
    }
    if (id.empty() || schemaFingerprint != schema.fingerprint ||
        taskFingerprint == 0u || policyFingerprint == 0u ||
        sourceModelFingerprint == 0u || fingerprint == 0u ||
        regions.size() > kMaximumFeedbackRegions ||
        !finite(broadWeight) || !finite(failureWeight) ||
        !finite(uncertaintyWeight) || broadWeight < 0.0 ||
        failureWeight < 0.0 || uncertaintyWeight < 0.0 ||
        std::abs(
            broadWeight + failureWeight + uncertaintyWeight - 1.0
        ) > 1.0e-9) {
        return failReason(
            reason,
            "feedback program identity or mixture is invalid"
        );
    }
    double failureRegionWeight = 0.0;
    double uncertaintyRegionWeight = 0.0;
    for (const FeedbackRegion& region : regions) {
        if (region.kind > MR_FEEDBACK_REGION_UNCERTAINTY ||
            !finite(region.weight) || region.weight < 0.0 ||
            region.lowerQuantiles.size() != schema.features.size() ||
            region.upperQuantiles.size() != schema.features.size()) {
            return failReason(
                reason,
                "feedback region dimensions or weight are invalid"
            );
        }
        for (std::size_t feature = 0u;
             feature < schema.features.size();
             ++feature) {
            const float lower = region.lowerQuantiles[feature];
            const float upper = region.upperQuantiles[feature];
            if (!finite(lower) || !finite(upper) ||
                lower < 0.0f || upper > 1.0f || lower > upper) {
                return failReason(
                    reason,
                    "feedback region bounds are invalid"
                );
            }
        }
        if (region.kind == MR_FEEDBACK_REGION_FAILURE) {
            failureRegionWeight += region.weight;
        } else {
            uncertaintyRegionWeight += region.weight;
        }
    }
    if ((failureWeight > 0.0 && !(failureRegionWeight > 0.0)) ||
        (uncertaintyWeight > 0.0 &&
         !(uncertaintyRegionWeight > 0.0))) {
        return failReason(
            reason,
            "feedback mixture references an empty region class"
        );
    }
    if (feedbackFingerprint(*this) != fingerprint) {
        return failReason(
            reason,
            "feedback fingerprint does not match its content"
        );
    }
    return true;
}

bool finalizeWorldFeedbackProgram(
    const ScenarioSchema& schema,
    WorldFeedbackProgram& program,
    std::string* reason
) {
    if (reason != nullptr) {
        reason->clear();
    }
    if (!schema.valid(reason)) {
        return false;
    }
    program.schemaFingerprint = schema.fingerprint;
    program.fingerprint = feedbackFingerprint(program);
    return program.valid(schema, reason);
}

bool CompiledWorldSamplingProgram::valid(
    const ScenarioSchema& schema,
    std::string* reason
) const {
    if (reason != nullptr) {
        reason->clear();
    }
    if (!schema.valid(reason)) {
        return false;
    }
    if (schemaFingerprint != schema.fingerprint ||
        !finite(broadWeight) || !finite(failureWeight) ||
        !finite(uncertaintyWeight) || broadWeight < 0.0 ||
        failureWeight < 0.0 || uncertaintyWeight < 0.0 ||
        std::abs(
            broadWeight + failureWeight + uncertaintyWeight - 1.0
        ) > 1.0e-6 ||
        !finite(alignmentJitter) || alignmentJitter < 0.0f ||
        alignmentParticles.size() > kMaximumAlignmentParticles ||
        feedbackRegions.size() > kMaximumFeedbackRegions ||
        alignmentQuantiles.size() !=
            alignmentParticles.size() * schema.features.size() ||
        feedbackBounds.size() !=
            feedbackRegions.size() * schema.features.size()) {
        return failReason(
            reason,
            "compiled world sampling program layout is invalid"
        );
    }
    float previousAlignment = 0.0f;
    for (const MRWorldAlignmentParticleGPU& particle :
         alignmentParticles) {
        if (!finite(particle.statistics.x) ||
            !finite(particle.statistics.y) ||
            !finite(particle.statistics.z) ||
            particle.statistics.x < 0.0f ||
            particle.statistics.y < previousAlignment ||
            particle.statistics.y > 1.0f ||
            particle.statistics.z < 0.0f) {
            return failReason(
                reason,
                "compiled alignment particle is invalid"
            );
        }
        previousAlignment = particle.statistics.y;
    }
    if (!alignmentParticles.empty() &&
        std::abs(previousAlignment - 1.0f) > 1.0e-5f) {
        return failReason(
            reason,
            "compiled alignment CDF is not normalized"
        );
    }
    for (const float quantile : alignmentQuantiles) {
        if (!finite(quantile) || quantile < 0.0f ||
            quantile > 1.0f) {
            return failReason(
                reason,
                "compiled alignment quantile is invalid"
            );
        }
    }
    float previousFailure = 0.0f;
    float previousUncertainty = 0.0f;
    for (const MRWorldFeedbackRegionGPU& region : feedbackRegions) {
        if (region.identity.x > MR_FEEDBACK_REGION_UNCERTAINTY ||
            !finite(region.statistics.x) ||
            !finite(region.statistics.y) ||
            region.statistics.x < 0.0f ||
            region.statistics.y < 0.0f ||
            region.statistics.y > 1.0f) {
            return failReason(reason, "compiled feedback region is invalid");
        }
        float& previous =
            region.identity.x == MR_FEEDBACK_REGION_FAILURE
            ? previousFailure
            : previousUncertainty;
        if (region.statistics.y < previous) {
            return failReason(
                reason,
                "compiled feedback region CDF is not monotonic"
            );
        }
        previous = region.statistics.y;
    }
    if ((failureWeight > 0.0 &&
         std::abs(previousFailure - 1.0f) > 1.0e-5f) ||
        (uncertaintyWeight > 0.0 &&
         std::abs(previousUncertainty - 1.0f) > 1.0e-5f)) {
        return failReason(
            reason,
            "compiled feedback region CDF is not normalized"
        );
    }
    for (const mr_float4 bounds : feedbackBounds) {
        if (!finite(bounds.x) || !finite(bounds.y) ||
            bounds.x < 0.0f || bounds.y > 1.0f ||
            bounds.x > bounds.y) {
            return failReason(
                reason,
                "compiled feedback region bounds are invalid"
            );
        }
    }
    return true;
}

bool compileWorldSamplingProgram(
    const ScenarioSchema& schema,
    const WorldAlignmentPopulation* alignment,
    const WorldFeedbackProgram* feedback,
    CompiledWorldSamplingProgram& output,
    std::string* reason
) {
    if (reason != nullptr) {
        reason->clear();
    }
    if (!schema.valid(reason) ||
        (alignment != nullptr &&
         !alignment->valid(schema, reason)) ||
        (feedback != nullptr &&
         !feedback->valid(schema, reason))) {
        return false;
    }
    CompiledWorldSamplingProgram staged;
    staged.schemaFingerprint = schema.fingerprint;
    if (alignment != nullptr) {
        staged.alignmentFingerprint = alignment->fingerprint;
        staged.alignmentParticles.reserve(
            alignment->particles.size()
        );
        staged.alignmentQuantiles.reserve(
            alignment->particles.size() * schema.features.size()
        );
        double cumulative = 0.0;
        for (std::uint32_t ordinal = 0u;
             ordinal < alignment->particles.size();
             ++ordinal) {
            const WorldAlignmentParticle& particle =
                alignment->particles[ordinal];
            cumulative += particle.weight;
            MRWorldAlignmentParticleGPU compiled{};
            compiled.statistics = {
                static_cast<float>(particle.weight),
                static_cast<float>(
                    ordinal + 1u == alignment->particles.size()
                    ? 1.0
                    : cumulative
                ),
                static_cast<float>(particle.replayResidual),
                0.0f,
            };
            compiled.identity = {ordinal, 0u, 0u, 0u};
            staged.alignmentParticles.push_back(compiled);
            staged.alignmentQuantiles.insert(
                staged.alignmentQuantiles.end(),
                particle.quantiles.begin(),
                particle.quantiles.end()
            );
        }
    }
    if (feedback != nullptr) {
        staged.feedbackFingerprint = feedback->fingerprint;
        staged.broadWeight = feedback->broadWeight;
        staged.failureWeight = feedback->failureWeight;
        staged.uncertaintyWeight = feedback->uncertaintyWeight;
        double failureTotal = 0.0;
        double uncertaintyTotal = 0.0;
        for (const FeedbackRegion& region : feedback->regions) {
            if (region.kind == MR_FEEDBACK_REGION_FAILURE) {
                failureTotal += region.weight;
            } else {
                uncertaintyTotal += region.weight;
            }
        }
        double failureCumulative = 0.0;
        double uncertaintyCumulative = 0.0;
        for (std::uint32_t ordinal = 0u;
             ordinal < feedback->regions.size();
             ++ordinal) {
            const FeedbackRegion& region = feedback->regions[ordinal];
            const double total =
                region.kind == MR_FEEDBACK_REGION_FAILURE
                ? failureTotal
                : uncertaintyTotal;
            double& cumulative =
                region.kind == MR_FEEDBACK_REGION_FAILURE
                ? failureCumulative
                : uncertaintyCumulative;
            const double normalized =
                total > 0.0 ? region.weight / total : 0.0;
            cumulative += normalized;
            MRWorldFeedbackRegionGPU compiled{};
            compiled.statistics = {
                static_cast<float>(normalized),
                static_cast<float>(
                    std::abs(cumulative - 1.0) < 1.0e-12
                    ? 1.0
                    : cumulative
                ),
                0.0f,
                0.0f,
            };
            compiled.identity = {
                region.kind,
                ordinal,
                0u,
                0u,
            };
            staged.feedbackRegions.push_back(compiled);
            for (std::size_t feature = 0u;
                 feature < schema.features.size();
                 ++feature) {
                staged.feedbackBounds.push_back({
                    region.lowerQuantiles[feature],
                    region.upperQuantiles[feature],
                    0.0f,
                    0.0f,
                });
            }
        }
    }
    if (!staged.valid(schema, reason)) {
        return false;
    }
    output = std::move(staged);
    return true;
}

bool PolicyDescriptor::valid(std::string* reason) const {
    if (reason != nullptr) {
        reason->clear();
    }
    if (id.empty() || contentHash.empty() || fingerprint == 0u ||
        observationSchemaFingerprint == 0u ||
        actionSchemaFingerprint == 0u ||
        embodimentFingerprint == 0u) {
        return failReason(reason, "policy descriptor identity is empty");
    }
    return true;
}

bool TaskOutcomeSchema::valid(std::string* reason) const {
    if (reason != nullptr) {
        reason->clear();
    }
    if (id.empty() || fingerprint == 0u ||
        failureTags.size() > 64u) {
        return failReason(reason, "task outcome schema is invalid");
    }
    std::unordered_set<std::string> tags;
    for (const std::string& tag : failureTags) {
        if (tag.empty() || !tags.insert(tag).second) {
            return failReason(
                reason,
                "task outcome schema has an empty or duplicate tag"
            );
        }
    }
    return true;
}

bool EpisodeOutcome::valid(
    const ScenarioSchema* schema,
    std::string* reason
) const {
    if (reason != nullptr) {
        reason->clear();
    }
    if (id.empty() || runId.empty() ||
        familyFingerprint == 0u || policyFingerprint == 0u ||
        taskFingerprint == 0u || embodimentFingerprint == 0u ||
        source > MR_EPISODE_SOURCE_HARDWARE ||
        termination > MR_EPISODE_TERMINATION_EXTERNAL ||
        (source == MR_EPISODE_SOURCE_SIMULATION &&
         scenarioKey == 0u) ||
        !finite(episodeReturn) || !finite(taskMargin) ||
        !finite(safetyMargin) || !finite(durationSeconds) ||
        durationSeconds < 0.0 || !finite(minimumVisibility) ||
        !finite(integratedContactLoad) ||
        !finite(peakContactLoad)) {
        return failReason(reason, "episode outcome fields are invalid");
    }
    if (scenarioValues.size() != scenarioValueMask.size()) {
        return failReason(
            reason,
            "episode scenario values and mask have different lengths"
        );
    }
    if (schema != nullptr &&
        (scenarioValues.size() != schema->features.size() ||
         !schema->valid(reason))) {
        return failReason(
            reason,
            "episode scenario values do not match the schema"
        );
    }
    for (std::size_t index = 0u;
         index < scenarioValues.size();
         ++index) {
        if (scenarioValueMask[index] > 1u ||
            (scenarioValueMask[index] != 0u &&
             !finite(scenarioValues[index]))) {
            return failReason(
                reason,
                "episode scenario measurement is invalid"
            );
        }
    }
    for (const OutcomeArtifact& artifact : artifacts) {
        if (artifact.kind.empty() || artifact.uri.empty() ||
            artifact.contentHash.empty()) {
            return failReason(
                reason,
                "episode outcome artifact identity is empty"
            );
        }
    }
    return true;
}

PolicyEvaluationReport evaluatePolicies(
    const TaskOutcomeSchema& task,
    const ScenarioSchema& scenarios,
    const std::span<const EpisodeOutcome> outcomes
) {
    PolicyEvaluationReport report;
    if (!task.valid() || !scenarios.valid()) {
        return report;
    }
    report.taskFingerprint = task.fingerprint;
    report.scenarioSchemaFingerprint = scenarios.fingerprint;

    std::unordered_map<
        std::uint64_t,
        std::vector<const EpisodeOutcome*>
    > byPolicy;
    std::unordered_map<
        std::uint64_t,
        std::unordered_set<std::uint64_t>
    > scenarioPolicies;
    for (const EpisodeOutcome& outcome : outcomes) {
        if (outcome.taskFingerprint != task.fingerprint ||
            !outcome.valid(nullptr, nullptr)) {
            continue;
        }
        byPolicy[outcome.policyFingerprint].push_back(&outcome);
        if (outcome.source == MR_EPISODE_SOURCE_SIMULATION &&
            outcome.scenarioKey != 0u) {
            scenarioPolicies[outcome.scenarioKey].insert(
                outcome.policyFingerprint
            );
        }
    }
    for (const auto& [key, policies] : scenarioPolicies) {
        static_cast<void>(key);
        if (policies.size() >= 2u) {
            report.pairedScenarioCount += 1u;
        }
    }

    report.policies.reserve(byPolicy.size());
    for (const auto& [fingerprint, policyOutcomes] : byPolicy) {
        PolicyEvaluationSummary summary;
        summary.policyFingerprint = fingerprint;
        summary.failureRates.assign(task.failureTags.size(), 0.0);
        std::uint64_t simulationSuccess = 0u;
        std::uint64_t hardwareSuccess = 0u;
        std::vector<std::uint64_t> failureCounts(
            task.failureTags.size(),
            0u
        );
        for (const EpisodeOutcome* outcome : policyOutcomes) {
            if (outcome->source == MR_EPISODE_SOURCE_SIMULATION) {
                summary.simulationEpisodes += 1u;
                simulationSuccess += outcome->success ? 1u : 0u;
            } else {
                summary.hardwareEpisodes += 1u;
                hardwareSuccess += outcome->success ? 1u : 0u;
            }
            for (std::size_t tag = 0u;
                 tag < failureCounts.size();
                 ++tag) {
                failureCounts[tag] +=
                    (outcome->failureMask & (1ull << tag)) != 0u
                    ? 1u
                    : 0u;
            }
        }
        if (summary.simulationEpisodes != 0u) {
            summary.simulationSuccessRate =
                static_cast<double>(simulationSuccess) /
                summary.simulationEpisodes;
        }
        if (summary.hardwareEpisodes != 0u) {
            const double rate =
                static_cast<double>(hardwareSuccess) /
                summary.hardwareEpisodes;
            summary.hardwareSuccessRate = rate;
            summary.calibratedHardwareSuccessRate = rate;
        }
        const double total = static_cast<double>(policyOutcomes.size());
        for (std::size_t tag = 0u; tag < failureCounts.size(); ++tag) {
            summary.failureRates[tag] =
                total == 0.0 ? 0.0 : failureCounts[tag] / total;
        }
        report.policies.push_back(std::move(summary));
    }
    std::stable_sort(
        report.policies.begin(),
        report.policies.end(),
        [](const PolicyEvaluationSummary& left,
           const PolicyEvaluationSummary& right) {
            const double leftScore =
                left.calibratedHardwareSuccessRate.value_or(
                    left.simulationSuccessRate
                );
            const double rightScore =
                right.calibratedHardwareSuccessRate.value_or(
                    right.simulationSuccessRate
                );
            if (leftScore != rightScore) {
                return leftScore > rightScore;
            }
            return left.policyFingerprint < right.policyFingerprint;
        }
    );
    for (const PolicyEvaluationSummary& summary : report.policies) {
        report.relativeOrdering.push_back(summary.policyFingerprint);
    }
    return report;
}

struct R2S2RStore::Impl {
    explicit Impl(std::filesystem::path configured)
        : path(std::move(configured)) {}

    std::filesystem::path path;
    sqlite3* database = nullptr;
};

R2S2RStore::R2S2RStore(std::filesystem::path path)
    : impl_(std::make_unique<Impl>(std::move(path))) {}

R2S2RStore::~R2S2RStore() {
    if (impl_ != nullptr && impl_->database != nullptr) {
        sqlite3_close(impl_->database);
    }
}

R2S2RStore::R2S2RStore(R2S2RStore&&) noexcept = default;
R2S2RStore& R2S2RStore::operator=(R2S2RStore&&) noexcept = default;

R2S2RResult R2S2RStore::open() {
    if (impl_ == nullptr || impl_->path.empty()) {
        return fail(
            R2S2RStatus::invalidArgument,
            "R2S2R store path is empty"
        );
    }
    if (impl_->database != nullptr) {
        return {};
    }
    std::error_code error;
    const std::filesystem::path parent = impl_->path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, error);
        if (error) {
            return fail(
                R2S2RStatus::ioFailure,
                "could not create R2S2R store directory: " +
                    error.message()
            );
        }
    }
    sqlite3* database = nullptr;
    const int status = sqlite3_open_v2(
        impl_->path.string().c_str(),
        &database,
        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE |
            SQLITE_OPEN_FULLMUTEX,
        nullptr
    );
    if (status != SQLITE_OK) {
        const R2S2RResult result =
            databaseError(database, "could not open R2S2R store");
        if (database != nullptr) {
            sqlite3_close(database);
        }
        return result;
    }
    impl_->database = database;
    const R2S2RResult initialized = execSQL(
        database,
        R"SQL(
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE IF NOT EXISTS scenario_schemas (
                fingerprint TEXT PRIMARY KEY,
                id TEXT NOT NULL,
                family_fingerprint TEXT NOT NULL,
                program_fingerprint TEXT NOT NULL,
                feature_count INTEGER NOT NULL,
                feature_names TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS policies (
                fingerprint TEXT PRIMARY KEY,
                id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                observation_schema TEXT NOT NULL,
                action_schema TEXT NOT NULL,
                embodiment TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS outcomes (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                scenario_key TEXT NOT NULL,
                episode_counter TEXT NOT NULL,
                family_fingerprint TEXT NOT NULL,
                alignment_fingerprint TEXT NOT NULL,
                feedback_fingerprint TEXT NOT NULL,
                policy_fingerprint TEXT NOT NULL,
                task_fingerprint TEXT NOT NULL,
                embodiment_fingerprint TEXT NOT NULL,
                source INTEGER NOT NULL,
                termination INTEGER NOT NULL,
                success INTEGER NOT NULL,
                failure_mask TEXT NOT NULL,
                physics_status INTEGER NOT NULL,
                step_count INTEGER NOT NULL,
                episode_return REAL NOT NULL,
                task_margin REAL NOT NULL,
                safety_margin REAL NOT NULL,
                duration_seconds REAL NOT NULL,
                minimum_visibility REAL NOT NULL,
                integrated_contact_load REAL NOT NULL,
                peak_contact_load REAL NOT NULL,
                scenario_values BLOB NOT NULL,
                scenario_mask BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS outcomes_task_policy
                ON outcomes(task_fingerprint, policy_fingerprint);
            CREATE INDEX IF NOT EXISTS outcomes_scenario
                ON outcomes(scenario_key, source);
            CREATE TABLE IF NOT EXISTS outcome_artifacts (
                outcome_id TEXT NOT NULL
                    REFERENCES outcomes(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                kind TEXT NOT NULL,
                uri TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                PRIMARY KEY(outcome_id, ordinal)
            );
        )SQL"
    );
    if (!initialized.succeeded()) {
        sqlite3_close(impl_->database);
        impl_->database = nullptr;
    }
    return initialized;
}

R2S2RResult R2S2RStore::registerScenarioSchema(
    const ScenarioSchema& schema
) {
    std::string reason;
    if (!schema.valid(&reason)) {
        return fail(R2S2RStatus::invalidArgument, std::move(reason));
    }
    const R2S2RResult opened = open();
    if (!opened.succeeded()) {
        return opened;
    }
    static constexpr const char* sql = R"SQL(
        INSERT INTO scenario_schemas (
            fingerprint, id, family_fingerprint, program_fingerprint,
            feature_count, feature_names
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(fingerprint) DO UPDATE SET
            id=excluded.id,
            family_fingerprint=excluded.family_fingerprint,
            program_fingerprint=excluded.program_fingerprint,
            feature_count=excluded.feature_count,
            feature_names=excluded.feature_names
    )SQL";
    sqlite3_stmt* statement = nullptr;
    if (sqlite3_prepare_v2(
            impl_->database,
            sql,
            -1,
            &statement,
            nullptr
        ) != SQLITE_OK) {
        return databaseError(
            impl_->database,
            "could not prepare scenario schema registration"
        );
    }
    const bool okay =
        bindText(statement, 1, hex64(schema.fingerprint)) &&
        bindText(statement, 2, schema.id) &&
        bindText(statement, 3, hex64(schema.familyFingerprint)) &&
        bindText(statement, 4, hex64(schema.programFingerprint)) &&
        sqlite3_bind_int64(
            statement,
            5,
            static_cast<sqlite3_int64>(schema.features.size())
        ) == SQLITE_OK &&
        bindText(statement, 6, featureNameBlob(schema));
    const int status = okay ? sqlite3_step(statement) : SQLITE_ERROR;
    sqlite3_finalize(statement);
    return status == SQLITE_DONE
        ? R2S2RResult{}
        : databaseError(
              impl_->database,
              "could not register scenario schema"
          );
}

R2S2RResult R2S2RStore::registerPolicy(
    const PolicyDescriptor& policy
) {
    std::string reason;
    if (!policy.valid(&reason)) {
        return fail(R2S2RStatus::invalidArgument, std::move(reason));
    }
    const R2S2RResult opened = open();
    if (!opened.succeeded()) {
        return opened;
    }
    static constexpr const char* sql = R"SQL(
        INSERT INTO policies (
            fingerprint, id, content_hash, observation_schema,
            action_schema, embodiment
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(fingerprint) DO UPDATE SET
            id=excluded.id,
            content_hash=excluded.content_hash,
            observation_schema=excluded.observation_schema,
            action_schema=excluded.action_schema,
            embodiment=excluded.embodiment
    )SQL";
    sqlite3_stmt* statement = nullptr;
    if (sqlite3_prepare_v2(
            impl_->database,
            sql,
            -1,
            &statement,
            nullptr
        ) != SQLITE_OK) {
        return databaseError(
            impl_->database,
            "could not prepare policy registration"
        );
    }
    const bool okay =
        bindText(statement, 1, hex64(policy.fingerprint)) &&
        bindText(statement, 2, policy.id) &&
        bindText(statement, 3, policy.contentHash) &&
        bindText(
            statement,
            4,
            hex64(policy.observationSchemaFingerprint)
        ) &&
        bindText(statement, 5, hex64(policy.actionSchemaFingerprint)) &&
        bindText(statement, 6, hex64(policy.embodimentFingerprint));
    const int status = okay ? sqlite3_step(statement) : SQLITE_ERROR;
    sqlite3_finalize(statement);
    return status == SQLITE_DONE
        ? R2S2RResult{}
        : databaseError(impl_->database, "could not register policy");
}

R2S2RResult R2S2RStore::appendOutcome(
    const EpisodeOutcome& outcome
) {
    return appendOutcomes(std::span<const EpisodeOutcome>{&outcome, 1u});
}

R2S2RResult R2S2RStore::appendOutcomes(
    const std::span<const EpisodeOutcome> outcomes
) {
    if (outcomes.empty()) {
        return {};
    }
    for (const EpisodeOutcome& outcome : outcomes) {
        std::string reason;
        if (!outcome.valid(nullptr, &reason)) {
            return fail(R2S2RStatus::invalidArgument, std::move(reason));
        }
    }
    const R2S2RResult opened = open();
    if (!opened.succeeded()) {
        return opened;
    }
    R2S2RResult transaction = execSQL(impl_->database, "BEGIN IMMEDIATE");
    if (!transaction.succeeded()) {
        return transaction;
    }
    for (const EpisodeOutcome& outcome : outcomes) {
        transaction = appendOutcomeRecord(impl_->database, outcome);
        if (!transaction.succeeded()) {
            static_cast<void>(execSQL(impl_->database, "ROLLBACK"));
            return transaction;
        }
    }
    transaction = execSQL(impl_->database, "COMMIT");
    if (!transaction.succeeded()) {
        static_cast<void>(execSQL(impl_->database, "ROLLBACK"));
    }
    return transaction;
}

R2S2RResult R2S2RStore::loadOutcomes(
    const std::uint64_t taskFingerprint,
    std::vector<EpisodeOutcome>& output
) const {
    if (impl_ == nullptr || impl_->database == nullptr ||
        taskFingerprint == 0u) {
        return fail(
            R2S2RStatus::invalidArgument,
            "open the store and provide a task fingerprint"
        );
    }
    static constexpr const char* sql = R"SQL(
        SELECT
            id, run_id, scenario_key, episode_counter,
            family_fingerprint, alignment_fingerprint,
            feedback_fingerprint, policy_fingerprint,
            task_fingerprint, embodiment_fingerprint,
            source, termination, success, failure_mask,
            physics_status, step_count, episode_return,
            task_margin, safety_margin, duration_seconds,
            minimum_visibility, integrated_contact_load,
            peak_contact_load, scenario_values, scenario_mask
        FROM outcomes
        WHERE task_fingerprint = ?
        ORDER BY id
    )SQL";
    sqlite3_stmt* statement = nullptr;
    if (sqlite3_prepare_v2(
            impl_->database,
            sql,
            -1,
            &statement,
            nullptr
        ) != SQLITE_OK ||
        !bindText(statement, 1, hex64(taskFingerprint))) {
        sqlite3_finalize(statement);
        return databaseError(
            impl_->database,
            "could not prepare outcome query"
        );
    }
    std::vector<EpisodeOutcome> staged;
    int status = SQLITE_ROW;
    while ((status = sqlite3_step(statement)) == SQLITE_ROW) {
        EpisodeOutcome outcome;
        const auto text = [statement](const int column) {
            const unsigned char* value =
                sqlite3_column_text(statement, column);
            return value == nullptr
                ? std::string{}
                : std::string{reinterpret_cast<const char*>(value)};
        };
        outcome.id = text(0);
        outcome.runId = text(1);
        bool okay =
            parseHex64(sqlite3_column_text(statement, 2), outcome.scenarioKey) &&
            parseHex64(
                sqlite3_column_text(statement, 3),
                outcome.episodeCounter
            ) &&
            parseHex64(
                sqlite3_column_text(statement, 4),
                outcome.familyFingerprint
            ) &&
            parseHex64(
                sqlite3_column_text(statement, 5),
                outcome.alignmentFingerprint
            ) &&
            parseHex64(
                sqlite3_column_text(statement, 6),
                outcome.feedbackFingerprint
            ) &&
            parseHex64(
                sqlite3_column_text(statement, 7),
                outcome.policyFingerprint
            ) &&
            parseHex64(
                sqlite3_column_text(statement, 8),
                outcome.taskFingerprint
            ) &&
            parseHex64(
                sqlite3_column_text(statement, 9),
                outcome.embodimentFingerprint
            ) &&
            parseHex64(
                sqlite3_column_text(statement, 13),
                outcome.failureMask
            );
        outcome.source = static_cast<MREpisodeSource>(
            sqlite3_column_int(statement, 10)
        );
        outcome.termination = static_cast<MREpisodeTermination>(
            sqlite3_column_int(statement, 11)
        );
        outcome.success = sqlite3_column_int(statement, 12) != 0;
        outcome.physicsStatus = static_cast<std::uint32_t>(
            sqlite3_column_int64(statement, 14)
        );
        outcome.stepCount = static_cast<std::uint32_t>(
            sqlite3_column_int64(statement, 15)
        );
        outcome.episodeReturn = sqlite3_column_double(statement, 16);
        outcome.taskMargin = sqlite3_column_double(statement, 17);
        outcome.safetyMargin = sqlite3_column_double(statement, 18);
        outcome.durationSeconds = sqlite3_column_double(statement, 19);
        outcome.minimumVisibility = sqlite3_column_double(statement, 20);
        outcome.integratedContactLoad =
            sqlite3_column_double(statement, 21);
        outcome.peakContactLoad = sqlite3_column_double(statement, 22);
        okay = okay &&
            columnVector(statement, 23, outcome.scenarioValues) &&
            columnVector(statement, 24, outcome.scenarioValueMask);
        if (!okay || !outcome.valid(nullptr, nullptr)) {
            sqlite3_finalize(statement);
            return fail(
                R2S2RStatus::databaseFailure,
                "stored outcome is corrupt"
            );
        }
        staged.push_back(std::move(outcome));
    }
    sqlite3_finalize(statement);
    if (status != SQLITE_DONE) {
        return databaseError(
            impl_->database,
            "could not read outcomes"
        );
    }

    static constexpr const char* artifactSQL = R"SQL(
        SELECT kind, uri, content_hash
        FROM outcome_artifacts
        WHERE outcome_id = ?
        ORDER BY ordinal
    )SQL";
    for (EpisodeOutcome& outcome : staged) {
        sqlite3_stmt* artifactStatement = nullptr;
        if (sqlite3_prepare_v2(
                impl_->database,
                artifactSQL,
                -1,
                &artifactStatement,
                nullptr
            ) != SQLITE_OK ||
            !bindText(artifactStatement, 1, outcome.id)) {
            sqlite3_finalize(artifactStatement);
            return databaseError(
                impl_->database,
                "could not prepare artifact query"
            );
        }
        while ((status = sqlite3_step(artifactStatement)) ==
               SQLITE_ROW) {
            OutcomeArtifact artifact;
            const auto value = [artifactStatement](const int column) {
                const unsigned char* text =
                    sqlite3_column_text(artifactStatement, column);
                return text == nullptr
                    ? std::string{}
                    : std::string{
                          reinterpret_cast<const char*>(text)
                      };
            };
            artifact.kind = value(0);
            artifact.uri = value(1);
            artifact.contentHash = value(2);
            outcome.artifacts.push_back(std::move(artifact));
        }
        sqlite3_finalize(artifactStatement);
        if (status != SQLITE_DONE) {
            return databaseError(
                impl_->database,
                "could not read outcome artifacts"
            );
        }
    }
    output = std::move(staged);
    return {};
}

const std::filesystem::path& R2S2RStore::path() const noexcept {
    return impl_->path;
}

const char* r2s2rStatusName(const R2S2RStatus status) noexcept {
    switch (status) {
    case R2S2RStatus::success:
        return "success";
    case R2S2RStatus::invalidArgument:
        return "invalid_argument";
    case R2S2RStatus::invalidManifest:
        return "invalid_manifest";
    case R2S2RStatus::schemaMismatch:
        return "schema_mismatch";
    case R2S2RStatus::ioFailure:
        return "io_failure";
    case R2S2RStatus::databaseFailure:
        return "database_failure";
    case R2S2RStatus::unsupportedVersion:
        return "unsupported_version";
    case R2S2RStatus::internalFailure:
        return "internal_failure";
    }
    return "unknown";
}

} // namespace metalrobo
