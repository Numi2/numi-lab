#include "metalrobo/Franka.hpp"
#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/MetalWorldFamily.hpp"
#include "metalrobo/R2S2R.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

namespace {

constexpr std::uint32_t kWorldCount = 4096u;
constexpr std::uint32_t kResidualCount = 6u;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::uint64_t words64(const mr_uint4 words, const bool upper) {
    const std::uint32_t low = upper ? words.z : words.x;
    const std::uint32_t high = upper ? words.w : words.y;
    return (static_cast<std::uint64_t>(high) << 32u) | low;
}

std::string hex64(const std::uint64_t value) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << value;
    return stream.str();
}

std::vector<float> makeSpaceFillingCandidates(
    const std::size_t featureCount
) {
    static constexpr std::array<std::uint32_t, 16> primes{
        2u, 3u, 5u, 7u, 11u, 13u, 17u, 19u,
        23u, 29u, 31u, 37u, 41u, 43u, 47u, 53u,
    };
    std::vector<float> values(
        static_cast<std::size_t>(kWorldCount) * featureCount,
        0.5f
    );
    for (std::uint32_t environment = 0u;
         environment < kWorldCount;
         ++environment) {
        for (std::size_t feature = 0u;
             feature < featureCount;
             ++feature) {
            const std::uint32_t multiplier =
                primes[feature % primes.size()];
            const std::uint32_t permuted =
                (environment * multiplier +
                 static_cast<std::uint32_t>(feature * 131u)) %
                kWorldCount;
            values[
                static_cast<std::size_t>(environment) *
                    featureCount +
                feature
            ] = (
                static_cast<float>(permuted) + 0.5f
            ) / static_cast<float>(kWorldCount);
        }
    }
    return values;
}

double syntheticReplayLoss(
    const std::span<const float> quantiles
) {
    const auto modeLoss = [&quantiles](
        const std::array<float, kResidualCount>& target
    ) {
        double loss = 0.0;
        for (std::size_t index = 0u;
             index < target.size();
             ++index) {
            const double residual =
                static_cast<double>(
                    quantiles[index % quantiles.size()]
                ) -
                target[index];
            loss += residual * residual;
        }
        return loss;
    };
    // Two contact-consistent explanations remain intentionally distinct.
    static constexpr std::array<float, kResidualCount> modeA{
        0.18f, 0.74f, 0.42f, 0.31f, 0.63f, 0.27f,
    };
    static constexpr std::array<float, kResidualCount> modeB{
        0.79f, 0.24f, 0.58f, 0.72f, 0.36f, 0.81f,
    };
    return std::min(modeLoss(modeA), modeLoss(modeB));
}

metalrobo::FeedbackRegion makeRegion(
    const MRFeedbackRegionKind kind,
    const double weight,
    const std::size_t featureCount,
    const std::size_t focusedFeature,
    const float lower,
    const float upper
) {
    metalrobo::FeedbackRegion region;
    region.kind = kind;
    region.weight = weight;
    region.lowerQuantiles.assign(featureCount, 0.0f);
    region.upperQuantiles.assign(featureCount, 1.0f);
    region.lowerQuantiles[focusedFeature] = lower;
    region.upperQuantiles[focusedFeature] = upper;
    return region;
}

metalrobo::EpisodeOutcome makeOutcome(
    const metalrobo::WorldInstanceBatch& batch,
    const std::uint32_t environment,
    const std::uint64_t policy,
    const std::uint64_t task,
    const std::uint64_t embodiment,
    const bool hardware,
    const bool success,
    const std::string& id
) {
    const MRWorldScenarioHeaderGPU& header =
        batch.scenarioHeaders[environment];
    metalrobo::EpisodeOutcome outcome;
    outcome.id = id;
    outcome.runId = hardware ? "hardware.run.1" : "simulation.coverage.1";
    outcome.scenarioKey =
        (static_cast<std::uint64_t>(header.identity.y) << 32u) |
        header.identity.x;
    outcome.episodeCounter =
        (static_cast<std::uint64_t>(header.identity.w) << 32u) |
        header.identity.z;
    outcome.familyFingerprint = batch.familyFingerprint;
    outcome.alignmentFingerprint =
        words64(header.provenance, false);
    outcome.feedbackFingerprint =
        words64(header.provenance, true);
    outcome.policyFingerprint = policy;
    outcome.taskFingerprint = task;
    outcome.embodimentFingerprint = embodiment;
    outcome.source = hardware
        ? MR_EPISODE_SOURCE_HARDWARE
        : MR_EPISODE_SOURCE_SIMULATION;
    outcome.termination = success
        ? MR_EPISODE_TERMINATION_SUCCESS
        : MR_EPISODE_TERMINATION_POLICY;
    outcome.success = success;
    outcome.failureMask = success ? 0u : 1u;
    outcome.stepCount = success ? 180u : 73u;
    outcome.episodeReturn = success ? 1.0 : -0.25;
    outcome.taskMargin = success ? 0.12 : -0.04;
    outcome.safetyMargin = success ? 0.08 : 0.02;
    outcome.durationSeconds = 3.0;
    outcome.minimumVisibility = 0.72;
    outcome.integratedContactLoad = 4.5;
    outcome.peakContactLoad = 2.1;
    const std::size_t featureCount =
        batch.scenarioValues.size() / batch.instances.size();
    outcome.scenarioValues.reserve(featureCount);
    outcome.scenarioValueMask.assign(featureCount, 1u);
    for (std::size_t feature = 0u;
         feature < featureCount;
         ++feature) {
        outcome.scenarioValues.push_back(
            batch.scenarioValues[
                static_cast<std::size_t>(environment) *
                    featureCount +
                feature
            ].value.x
        );
    }
    return outcome;
}

} // namespace

int main() {
    try {
        metalrobo::WorldTemplate worldTemplate;
        const auto twin = metalrobo::compileEpisodeTwin(
            metalrobo::makeFrankaPickPlaceEpisodeTwin(),
            metalrobo::makeFrankaPickPlaceEngineModel(),
            worldTemplate
        );
        require(twin.succeeded(), twin.message);
        metalrobo::WorldFamily family;
        const auto compiledFamily = metalrobo::compileWorldFamily(
            worldTemplate,
            metalrobo::makeFrankaPickPlaceWorldProgram(),
            family
        );
        require(compiledFamily.succeeded(), compiledFamily.message);

        const metalrobo::ScenarioSchema schema =
            metalrobo::compileScenarioSchema(family);
        std::string reason;
        require(schema.valid(&reason), reason);
        const std::size_t featureCount = schema.features.size();
        require(featureCount >= kResidualCount, "scenario schema is too small");

        std::vector<float> candidates =
            makeSpaceFillingCandidates(featureCount);
        const double priorMeanLoss = [&]() {
            double sum = 0.0;
            for (std::uint32_t environment = 0u;
                 environment < kWorldCount;
                 ++environment) {
                sum += syntheticReplayLoss(std::span<const float>{
                    candidates.data() +
                        static_cast<std::size_t>(environment) *
                            featureCount,
                    featureCount,
                });
            }
            return sum / kWorldCount;
        }();
        metalrobo::ReplayAlignmentConfig alignmentConfig;
        alignmentConfig.rounds = 4u;
        alignmentConfig.maximumParticles = kWorldCount;
        alignmentConfig.minimumEffectiveSampleFraction = 0.72;
        alignmentConfig.jitterScale = 0.025;
        alignmentConfig.seed = 0x5eed1234ull;
        metalrobo::WorldAlignmentPopulation alignment;
        const metalrobo::ReplayResidualEvaluator replay =
            [featureCount](
                const std::span<const float> values,
                const std::uint32_t count,
                const std::uint32_t residualCount,
                std::vector<float>& residuals,
                std::string*
            ) {
                if (residualCount != kResidualCount) {
                    return false;
                }
                static constexpr std::array<float, kResidualCount> modeA{
                    0.18f, 0.74f, 0.42f, 0.31f, 0.63f, 0.27f,
                };
                static constexpr std::array<float, kResidualCount> modeB{
                    0.79f, 0.24f, 0.58f, 0.72f, 0.36f, 0.81f,
                };
                residuals.resize(
                    static_cast<std::size_t>(count) * residualCount
                );
                for (std::uint32_t environment = 0u;
                     environment < count;
                     ++environment) {
                    const auto row = std::span<const float>{
                        values.data() +
                            static_cast<std::size_t>(environment) *
                                featureCount,
                        featureCount,
                    };
                    const auto squared = [&row](
                        const auto& target
                    ) {
                        double result = 0.0;
                        for (std::size_t index = 0u;
                             index < target.size();
                             ++index) {
                            const double value =
                                row[index] - target[index];
                            result += value * value;
                        }
                        return result;
                    };
                    const auto& target =
                        squared(modeA) <= squared(modeB)
                        ? modeA
                        : modeB;
                    for (std::uint32_t residual = 0u;
                         residual < residualCount;
                         ++residual) {
                        residuals[
                            static_cast<std::size_t>(environment) *
                                residualCount +
                            residual
                        ] = row[residual] - target[residual];
                    }
                }
                return true;
            };
        require(
            metalrobo::fitAlignmentPopulationSMC(
                schema,
                candidates,
                kWorldCount,
                kResidualCount,
                alignmentConfig,
                replay,
                alignment,
                &reason
            ),
            reason
        );
        const double posteriorMeanLoss = std::accumulate(
            alignment.particles.begin(),
            alignment.particles.end(),
            0.0,
            [](const double sum,
               const metalrobo::WorldAlignmentParticle& particle) {
                return sum + particle.weight *
                    syntheticReplayLoss(particle.quantiles);
            }
        );
        require(
            posteriorMeanLoss < priorMeanLoss,
            "aligned posterior did not improve replay fit"
        );
        const bool hasModeA = std::any_of(
            alignment.particles.begin(),
            alignment.particles.end(),
            [](const auto& particle) {
                return particle.quantiles[0] < 0.4f &&
                    particle.weight > 0.0;
            }
        );
        const bool hasModeB = std::any_of(
            alignment.particles.begin(),
            alignment.particles.end(),
            [](const auto& particle) {
                return particle.quantiles[0] > 0.6f &&
                    particle.weight > 0.0;
            }
        );
        require(hasModeA && hasModeB, "SMC collapsed a contact mode");

        metalrobo::WorldFeedbackProgram feedback;
        feedback.id = "feedback.franka.pick_place.policy_a.v1";
        feedback.taskFingerprint = 0x741a5c001ull;
        feedback.policyFingerprint = 0xa110c001ull;
        feedback.sourceModelFingerprint = 0x5e11b1e001ull;
        feedback.regions.push_back(makeRegion(
            MR_FEEDBACK_REGION_FAILURE,
            1.0,
            featureCount,
            3u,
            0.82f,
            0.98f
        ));
        feedback.regions.push_back(makeRegion(
            MR_FEEDBACK_REGION_UNCERTAINTY,
            1.0,
            featureCount,
            4u,
            0.45f,
            0.55f
        ));
        require(
            metalrobo::finalizeWorldFeedbackProgram(
                schema,
                feedback,
                &reason
            ),
            reason
        );
        metalrobo::CompiledWorldSamplingProgram samplingProgram;
        require(
            metalrobo::compileWorldSamplingProgram(
                schema,
                &alignment,
                &feedback,
                samplingProgram,
                &reason
            ),
            reason
        );

        metalrobo::MetalWorldFamilyContext context;
        const auto compile = context.compile(family, kWorldCount);
        require(compile.succeeded(), compile.message);
        const auto configured =
            context.configureSamplingProgram(schema, samplingProgram);
        require(configured.succeeded(), configured.message);
        const auto coverageSample = context.sample(
            kWorldCount,
            0x123456789abcdef0ull,
            MR_WORLD_SAMPLING_COVERAGE,
            17u
        );
        require(coverageSample.succeeded(), coverageSample.message);
        metalrobo::WorldInstanceBatch coverage;
        const auto coverageReadback = context.readback(coverage);
        require(coverageReadback.succeeded(), coverageReadback.message);
        require(coverage.valid(&reason), reason);
        for (std::uint32_t environment = 0u;
             environment < kWorldCount;
             ++environment) {
            const MRWorldScenarioHeaderGPU& header =
                coverage.scenarioHeaders[environment];
            require(
                header.sampling.x == MR_WORLD_SAMPLING_COVERAGE &&
                    header.sampling.y == MR_WORLD_SAMPLE_ALIGNMENT &&
                    words64(header.identity, true) ==
                        17u + environment &&
                    words64(header.provenance, false) ==
                        alignment.fingerprint,
                "coverage provenance is not posterior-predictive"
            );
        }

        const auto curriculumSample = context.sample(
            kWorldCount,
            0x123456789abcdef0ull,
            MR_WORLD_SAMPLING_CURRICULUM,
            18u
        );
        require(curriculumSample.succeeded(), curriculumSample.message);
        metalrobo::WorldInstanceBatch curriculum;
        const auto curriculumReadback = context.readback(curriculum);
        require(
            curriculumReadback.succeeded(),
            curriculumReadback.message
        );
        require(curriculum.valid(&reason), reason);
        std::array<std::uint32_t, 4> sourceCounts{};
        for (std::uint32_t environment = 0u;
             environment < kWorldCount;
             ++environment) {
            const MRWorldScenarioHeaderGPU& header =
                curriculum.scenarioHeaders[environment];
            require(
                header.sampling.x == MR_WORLD_SAMPLING_CURRICULUM &&
                    header.sampling.y <= MR_WORLD_SAMPLE_UNCERTAINTY &&
                    words64(header.identity, true) ==
                        18u + environment,
                "curriculum scenario header is invalid"
            );
            sourceCounts[header.sampling.y] += 1u;
            if (header.sampling.y == MR_WORLD_SAMPLE_FAILURE) {
                const float quantile =
                    curriculum.scenarioValues[
                        static_cast<std::size_t>(environment) *
                            featureCount +
                        3u
                    ].value.y;
                require(
                    quantile >= 0.82f && quantile <= 0.98f,
                    "failure sample escaped its compiled region"
                );
            } else if (
                header.sampling.y == MR_WORLD_SAMPLE_UNCERTAINTY
            ) {
                const float quantile =
                    curriculum.scenarioValues[
                        static_cast<std::size_t>(environment) *
                            featureCount +
                        4u
                    ].value.y;
                require(
                    quantile >= 0.45f && quantile <= 0.55f,
                    "uncertainty sample escaped its compiled region"
                );
            }
        }

        const auto replaySample = context.sample(
            kWorldCount,
            0x0123456789abcdefull,
            MR_WORLD_SAMPLING_REPLAY,
            100u
        );
        require(replaySample.succeeded(), replaySample.message);
        metalrobo::WorldInstanceBatch replayWorlds;
        const auto replayReadback = context.readback(replayWorlds);
        require(replayReadback.succeeded(), replayReadback.message);
        for (std::uint32_t environment = 0u;
             environment < kWorldCount;
             ++environment) {
            const auto& header =
                replayWorlds.scenarioHeaders[environment];
            require(
                header.sampling.x == MR_WORLD_SAMPLING_REPLAY &&
                    header.sampling.y == MR_WORLD_SAMPLE_ALIGNMENT &&
                    header.sampling.z == environment &&
                    words64(header.identity, true) ==
                        100u + environment,
                "replay sampling did not preserve particle identity"
            );
            for (std::uint32_t feature = 0u;
                 feature < featureCount;
                 ++feature) {
                const float actual =
                    replayWorlds.scenarioValues[
                        static_cast<std::size_t>(environment) *
                            featureCount +
                        feature
                    ].value.y;
                const float expected =
                    alignment.particles[environment]
                        .quantiles[feature];
                require(
                    std::abs(actual - expected) <= 1.0e-6f,
                    "replay sampling changed a candidate quantile"
                );
            }
        }
        const double broadFraction =
            static_cast<double>(
                sourceCounts[MR_WORLD_SAMPLE_BROAD]
            ) / kWorldCount;
        const double failureFraction =
            static_cast<double>(
                sourceCounts[MR_WORLD_SAMPLE_FAILURE]
            ) / kWorldCount;
        const double uncertaintyFraction =
            static_cast<double>(
                sourceCounts[MR_WORLD_SAMPLE_UNCERTAINTY]
            ) / kWorldCount;
        require(
            broadFraction > 0.44 && broadFraction < 0.56 &&
                failureFraction > 0.24 && failureFraction < 0.36 &&
                uncertaintyFraction > 0.14 &&
                uncertaintyFraction < 0.26,
            "curriculum mixture is not 50/30/20"
        );

        constexpr std::uint64_t policyA = 0xa110c001ull;
        constexpr std::uint64_t policyB = 0xb220c002ull;
        constexpr std::uint64_t task = 0x741a5c001ull;
        constexpr std::uint64_t embodiment = 0xf4a9ca001ull;
        metalrobo::TaskOutcomeSchema taskSchema;
        taskSchema.id = "franka.pick_place.outcome.v1";
        taskSchema.fingerprint = task;
        taskSchema.failureTags = {"missed_grasp", "unsafe_contact"};
        std::vector<metalrobo::EpisodeOutcome> outcomes;
        for (std::uint32_t environment = 0u;
             environment < 16u;
             ++environment) {
            outcomes.push_back(makeOutcome(
                coverage,
                environment,
                policyA,
                task,
                embodiment,
                false,
                environment % 4u != 0u,
                "sim.a." + std::to_string(environment)
            ));
            outcomes.push_back(makeOutcome(
                coverage,
                environment,
                policyB,
                task,
                embodiment,
                false,
                environment % 2u != 0u,
                "sim.b." + std::to_string(environment)
            ));
        }
        const auto nonce = std::chrono::steady_clock::now()
            .time_since_epoch()
            .count();
        const std::filesystem::path hardwarePath =
            std::filesystem::temp_directory_path() /
            (
                "metalrobo-hardware-outcome-" +
                std::to_string(nonce) +
                ".json"
            );
        {
            std::ofstream hardware{hardwarePath};
            require(
                hardware.good(),
                "could not create hardware outcome manifest"
            );
            hardware
                << "{"
                << "\"schema_version\":1,"
                << "\"scenario_schema\":" << std::quoted(schema.id) << ","
                << "\"policy_id\":\"policy_a\","
                << "\"robot_id\":\"franka_panda\","
                << "\"task_id\":\"pick_place\","
                << "\"id\":\"real.a.0\","
                << "\"run_id\":\"hardware.run.1\","
                << "\"family_fingerprint\":\""
                << hex64(coverage.familyFingerprint) << "\","
                << "\"policy_fingerprint\":\""
                << hex64(policyA) << "\","
                << "\"task_fingerprint\":\""
                << hex64(task) << "\","
                << "\"embodiment_fingerprint\":\""
                << hex64(embodiment) << "\","
                << "\"termination\":\"policy\","
                << "\"success\":false,"
                << "\"failure_tags\":[\"missed_grasp\"],"
                << "\"task_margin\":-0.04,"
                << "\"scenario_values\":{";
            for (std::size_t feature = 0u;
                 feature < schema.features.size();
                 ++feature) {
                hardware
                    << (feature == 0u ? "" : ",")
                    << std::quoted(schema.features[feature].id)
                    << ":null";
            }
            hardware << "},\"missing_value_mask\":{";
            for (std::size_t feature = 0u;
                 feature < schema.features.size();
                 ++feature) {
                hardware
                    << (feature == 0u ? "" : ",")
                    << std::quoted(schema.features[feature].id)
                    << ":true";
            }
            hardware
                << "},\"artifacts\":[{"
                << "\"kind\":\"robot_telemetry\","
                << "\"uri\":\"artifact://sha256/telemetry\","
                << "\"content_hash\":\"sha256:telemetry\""
                << "}]}";
        }
        metalrobo::HardwareOutcomeManifest hardwareManifest;
        const metalrobo::R2S2RResult hardwareLoad =
            metalrobo::loadHardwareOutcomeManifestJSON(
                hardwarePath,
                schema,
                taskSchema,
                hardwareManifest
            );
        require(hardwareLoad.succeeded(), hardwareLoad.message);
        require(
            std::all_of(
                hardwareManifest.scenarioMissingMask.begin(),
                hardwareManifest.scenarioMissingMask.end(),
                [](const std::uint8_t missing) {
                    return missing == 1u;
                }
            ),
            "hardware missing-value mask did not round-trip"
        );
        outcomes.push_back(std::move(hardwareManifest.outcome));
        std::error_code removeError;
        std::filesystem::remove(hardwarePath, removeError);

        const metalrobo::PolicyEvaluationReport report =
            metalrobo::evaluatePolicies(
                taskSchema,
                schema,
                outcomes
            );
        require(
            report.pairedScenarioCount == 16u &&
                report.policies.size() == 2u,
            "paired policy evaluation did not use identical worlds"
        );

        const std::filesystem::path databasePath =
            std::filesystem::temp_directory_path() /
            (
                "metalrobo-r2s2r-" +
                std::to_string(nonce) +
                ".sqlite3"
            );
        metalrobo::R2S2RStore store{databasePath};
        require(store.open().succeeded(), "could not open outcome store");
        require(
            store.registerScenarioSchema(schema).succeeded(),
            "could not persist scenario schema"
        );
        for (const auto [id, fingerprint] :
             std::array{
                 std::pair{"policy_a", policyA},
                 std::pair{"policy_b", policyB},
             }) {
            metalrobo::PolicyDescriptor policy;
            policy.id = id;
            policy.contentHash =
                "sha256:0123456789abcdef0123456789abcdef";
            policy.fingerprint = fingerprint;
            policy.observationSchemaFingerprint = 0x0b5e001ull;
            policy.actionSchemaFingerprint = 0xac710001ull;
            policy.embodimentFingerprint = embodiment;
            require(
                store.registerPolicy(policy).succeeded(),
                "could not persist policy descriptor"
            );
        }
        require(
            store.appendOutcomes(outcomes).succeeded(),
            "could not persist episode outcomes"
        );
        std::vector<metalrobo::EpisodeOutcome> roundTrip;
        require(
            store.loadOutcomes(task, roundTrip).succeeded() &&
                roundTrip.size() == outcomes.size(),
            "outcome store did not round-trip records"
        );
        std::filesystem::remove(databasePath, removeError);
        std::filesystem::remove(
            databasePath.string() + "-wal",
            removeError
        );
        std::filesystem::remove(
            databasePath.string() + "-shm",
            removeError
        );

        std::cout
            << "device=\"" << curriculumSample.deviceName << "\""
            << " candidates=" << kWorldCount
            << " particles=" << alignment.particles.size()
            << " prior_replay_loss=" << priorMeanLoss
            << " posterior_replay_loss=" << posteriorMeanLoss
            << " curriculum="
            << sourceCounts[MR_WORLD_SAMPLE_BROAD] << "/"
            << sourceCounts[MR_WORLD_SAMPLE_FAILURE] << "/"
            << sourceCounts[MR_WORLD_SAMPLE_UNCERTAINTY]
            << " paired_worlds=" << report.pairedScenarioCount
            << " outcomes=" << roundTrip.size()
            << " hardware_manifest=yes"
            << " sqlite_wal=yes closed_loop=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_r2s2r_loop_probe: "
                  << error.what() << '\n';
        return 1;
    }
}
