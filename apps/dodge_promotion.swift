import Foundation
import Darwin

private enum PromotionError: Error, CustomStringConvertible {
    case invalid(String)
    case rollout(String)

    var description: String {
        switch self {
        case .invalid(let message), .rollout(let message):
            return message
        }
    }
}

private struct Options {
    var rolloutExecutable: String?
    var baselinePolicy: String?
    var candidatePolicy: String?
    var metallib: String?
    var robotVisualDirectory: String?
    var ballVisualDirectory: String?
    var seeds: [UInt64] = [880_055, 880_056, 880_057, 880_058, 880_059]
    var environments = 16
    var steps = 512
    var chunk = 1
    var scene = "terrain"

    init(_ arguments: [String]) throws {
        var index = 1
        while index < arguments.count {
            let option = arguments[index]
            guard index + 1 < arguments.count else {
                throw PromotionError.invalid("\(option) requires a value")
            }
            let value = arguments[index + 1]
            switch option {
            case "--rollout-executable": rolloutExecutable = value
            case "--baseline-policy": baselinePolicy = value
            case "--candidate-policy": candidatePolicy = value
            case "--metallib": metallib = value
            case "--g1-visual-pack-dir": robotVisualDirectory = value
            case "--ball-visual-pack-dir": ballVisualDirectory = value
            case "--seeds":
                seeds = value.split(separator: ",").compactMap {
                    UInt64($0)
                }
                guard !seeds.isEmpty else {
                    throw PromotionError.invalid(
                        "--seeds requires comma-separated unsigned integers"
                    )
                }
            case "--envs":
                guard let parsed = Int(value), parsed > 0 else {
                    throw PromotionError.invalid("--envs must be positive")
                }
                environments = parsed
            case "--steps":
                guard let parsed = Int(value), parsed > 0 else {
                    throw PromotionError.invalid("--steps must be positive")
                }
                steps = parsed
            case "--chunk":
                guard let parsed = Int(value), parsed > 0 else {
                    throw PromotionError.invalid("--chunk must be positive")
                }
                chunk = parsed
            case "--scene":
                guard value == "ground" || value == "terrain" else {
                    throw PromotionError.invalid(
                        "--scene must be ground or terrain"
                    )
                }
                scene = value
            default:
                throw PromotionError.invalid("unknown option \(option)")
            }
            index += 2
        }
        guard rolloutExecutable != nil,
              baselinePolicy != nil,
              candidatePolicy != nil,
              metallib != nil,
              robotVisualDirectory != nil,
              ballVisualDirectory != nil
        else {
            throw PromotionError.invalid(
                "rollout executable, both policies, metallib, and both visual-pack directories are required"
            )
        }
    }
}

private struct Aggregate {
    var hits = 0
    var misses = 0
    var balanceFailures = 0
    var failedEnvironmentSteps = 0
    var environmentSteps = 0
    var impactActiveSteps = 0
    var rewardStepSum = 0.0
    var taskRewardStepSum = 0.0
    var trackingStepSum = 0.0
    var rootHeightStepSum = 0.0
    var tiltStepSum = 0.0
    var maximumImpactTilt = 0.0
    var minimumImpactRootHeight = Double.infinity
    var perSeed: [[String: Any]] = []

    var trials: Int { hits + misses }
    var hitRate: Double {
        Double(hits) / Double(max(trials, 1))
    }

    var hitIncidence: Double {
        Double(hits) / Double(max(environmentSteps, 1))
    }

    var missIncidence: Double {
        Double(misses) / Double(max(environmentSteps, 1))
    }

    var balanceFailureIncidence: Double {
        Double(balanceFailures) / Double(max(environmentSteps, 1))
    }

    var impactActiveFraction: Double {
        Double(impactActiveSteps) /
            Double(max(environmentSteps, 1))
    }

    var meanReward: Double {
        rewardStepSum / Double(max(environmentSteps, 1))
    }

    var meanTaskReward: Double {
        taskRewardStepSum / Double(max(environmentSteps, 1))
    }

    var meanTracking: Double {
        trackingStepSum / Double(max(environmentSteps, 1))
    }

    var meanRootHeight: Double {
        rootHeightStepSum / Double(max(environmentSteps, 1))
    }

    var meanTilt: Double {
        tiltStepSum / Double(max(environmentSteps, 1))
    }

    var json: [String: Any] {
        [
            "any_link_projectile_hit_count": hits,
            "clean_projectile_miss_count": misses,
            "completed_projectile_trial_count": trials,
            "any_link_projectile_hit_rate": hitRate,
            "any_link_dodge_rate": 1.0 - hitRate,
            "height_or_tilt_termination_count": balanceFailures,
            "failed_environment_steps": failedEnvironmentSteps,
            "environment_steps": environmentSteps,
            "projectile_contacts_per_million_environment_steps":
                1.0e6 * hitIncidence,
            "clean_misses_per_million_environment_steps":
                1.0e6 * missIncidence,
            "balance_failures_per_million_environment_steps":
                1.0e6 * balanceFailureIncidence,
            "impact_active_fraction": impactActiveFraction,
            "mean_reward": meanReward,
            "mean_task_reward": meanTaskReward,
            "mean_tracking_score": meanTracking,
            "mean_root_height": meanRootHeight,
            "mean_tilt": meanTilt,
            "maximum_impact_tilt": maximumImpactTilt,
            "minimum_impact_root_height":
                minimumImpactRootHeight.isFinite
                ? minimumImpactRootHeight
                : 0.0,
            "per_seed": perSeed,
        ]
    }
}

private func integer(
    _ record: [String: Any],
    _ key: String
) throws -> Int {
    guard let value = record[key] as? NSNumber else {
        throw PromotionError.rollout("rollout omitted \(key)")
    }
    return value.intValue
}

private func double(
    _ record: [String: Any],
    _ key: String
) throws -> Double {
    guard let value = record[key] as? NSNumber else {
        throw PromotionError.rollout("rollout omitted \(key)")
    }
    return value.doubleValue
}

private func evaluate(
    policy: String,
    options: Options
) throws -> Aggregate {
    var aggregate = Aggregate()
    for seed in options.seeds {
        let process = Process()
        let output = Pipe()
        let errors = Pipe()
        process.executableURL = URL(
            fileURLWithPath: options.rolloutExecutable!
        )
        process.arguments = [
            "--metallib", options.metallib!,
            "--envs", String(options.environments),
            "--steps", String(options.steps),
            "--repeats", "1",
            "--chunk", String(options.chunk),
            "--task", "ball-dodge",
            "--scene", options.scene,
            "--seed", String(seed),
            "--policy-pack", policy,
            "--g1-visual-pack-dir", options.robotVisualDirectory!,
            "--ball-visual-pack-dir", options.ballVisualDirectory!,
        ]
        process.standardOutput = output
        process.standardError = errors
        try process.run()
        process.waitUntilExit()
        let data = output.fileHandleForReading.readDataToEndOfFile()
        let errorData = errors.fileHandleForReading.readDataToEndOfFile()
        guard process.terminationStatus == 0 else {
            throw PromotionError.rollout(
                String(data: errorData, encoding: .utf8) ??
                    "rollout failed for seed \(seed)"
            )
        }
        guard let record = try JSONSerialization.jsonObject(with: data)
                as? [String: Any]
        else {
            throw PromotionError.rollout(
                "rollout returned invalid JSON for seed \(seed)"
            )
        }
        let hits = try integer(record, "any_link_projectile_hit_count")
        let misses = try integer(record, "clean_projectile_miss_count")
        let balance = try integer(
            record,
            "height_or_tilt_termination_count"
        )
        let failed = try integer(record, "failed_environment_steps")
        let environmentSteps =
            try integer(record, "environments") *
            integer(record, "steps_per_repeat") *
            integer(record, "repeats")
        let impactActive = try integer(
            record,
            "impact_sequence_enabled_steps"
        )
        let meanReward = try double(record, "mean_reward")
        let meanTaskReward = try double(record, "mean_task_reward")
        let meanTracking = try double(record, "mean_tracking_score")
        let meanRootHeight = try double(record, "mean_root_height")
        let meanTilt = try double(record, "mean_tilt")
        guard let impactMetrics = record["impact_sequence_metrics"]
                as? [[String: Any]]
        else {
            throw PromotionError.rollout(
                "rollout omitted impact_sequence_metrics"
            )
        }
        var seedMaximumImpactTilt = 0.0
        var seedMinimumImpactRootHeight = Double.infinity
        for metric in impactMetrics {
            seedMaximumImpactTilt = max(
                seedMaximumImpactTilt,
                try double(metric, "peak_tilt")
            )
            seedMinimumImpactRootHeight = min(
                seedMinimumImpactRootHeight,
                try double(metric, "minimum_root_height")
            )
        }
        aggregate.hits += hits
        aggregate.misses += misses
        aggregate.balanceFailures += balance
        aggregate.failedEnvironmentSteps += failed
        aggregate.environmentSteps += environmentSteps
        aggregate.impactActiveSteps += impactActive
        aggregate.rewardStepSum += meanReward * Double(environmentSteps)
        aggregate.taskRewardStepSum +=
            meanTaskReward * Double(environmentSteps)
        aggregate.trackingStepSum +=
            meanTracking * Double(environmentSteps)
        aggregate.rootHeightStepSum +=
            meanRootHeight * Double(environmentSteps)
        aggregate.tiltStepSum += meanTilt * Double(environmentSteps)
        aggregate.maximumImpactTilt = max(
            aggregate.maximumImpactTilt,
            seedMaximumImpactTilt
        )
        aggregate.minimumImpactRootHeight = min(
            aggregate.minimumImpactRootHeight,
            seedMinimumImpactRootHeight
        )
        aggregate.perSeed.append([
            "seed": NSNumber(value: seed),
            "hits": hits,
            "misses": misses,
            "balance_failures": balance,
            "failed_environment_steps": failed,
            "environment_steps": environmentSteps,
            "projectile_contacts_per_million_environment_steps":
                1.0e6 * Double(hits) /
                Double(max(environmentSteps, 1)),
            "balance_failures_per_million_environment_steps":
                1.0e6 * Double(balance) /
                Double(max(environmentSteps, 1)),
            "impact_active_fraction":
                Double(impactActive) /
                Double(max(environmentSteps, 1)),
            "mean_reward": meanReward,
            "mean_task_reward": meanTaskReward,
            "mean_tracking_score": meanTracking,
            "mean_root_height": meanRootHeight,
            "mean_tilt": meanTilt,
            "maximum_impact_tilt": seedMaximumImpactTilt,
            "minimum_impact_root_height":
                seedMinimumImpactRootHeight.isFinite
                ? seedMinimumImpactRootHeight
                : 0.0,
        ])
    }
    return aggregate
}

@main
private enum DodgePromotionMain {
    static func main() {
        do {
            let options = try Options(CommandLine.arguments)
            let baseline = try evaluate(
                policy: options.baselinePolicy!,
                options: options
            )
            let candidate = try evaluate(
                policy: options.candidatePolicy!,
                options: options
            )
            let promoted =
                baseline.failedEnvironmentSteps == 0 &&
                candidate.failedEnvironmentSteps == 0 &&
                candidate.hits < baseline.hits &&
                candidate.hitRate < baseline.hitRate &&
                candidate.balanceFailures <= baseline.balanceFailures
            // Promotion remains a strict clean-outcome milestone. Progress
            // permits a two-percent companion balance regression because a
            // handful of events in a finite held-out sample is not evidence
            // that a policy with fewer contacts or more clean misses failed
            // to advance. The exact directional deltas remain published.
            let balanceRegressionTolerance =
                baseline.balanceFailureIncidence * 0.02
            let taskOutcomeImproved =
                candidate.hitIncidence < baseline.hitIncidence ||
                candidate.missIncidence > baseline.missIncidence
            let progressed =
                baseline.failedEnvironmentSteps == 0 &&
                candidate.failedEnvironmentSteps == 0 &&
                taskOutcomeImproved &&
                candidate.balanceFailureIncidence <=
                    baseline.balanceFailureIncidence +
                        balanceRegressionTolerance
            var improvedDimensions: [String] = []
            if candidate.hitIncidence < baseline.hitIncidence {
                improvedDimensions.append("projectile_contact_incidence")
            }
            if candidate.missIncidence > baseline.missIncidence {
                improvedDimensions.append("clean_miss_incidence")
            }
            if candidate.balanceFailureIncidence <
                baseline.balanceFailureIncidence {
                improvedDimensions.append("balance_failure_incidence")
            }
            if candidate.meanReward > baseline.meanReward {
                improvedDimensions.append("mean_reward")
            }
            if candidate.meanTaskReward > baseline.meanTaskReward {
                improvedDimensions.append("mean_task_reward")
            }
            if candidate.meanTracking > baseline.meanTracking {
                improvedDimensions.append("mean_tracking_score")
            }
            if candidate.meanRootHeight > baseline.meanRootHeight {
                improvedDimensions.append("mean_root_height")
            }
            if candidate.meanTilt < baseline.meanTilt {
                improvedDimensions.append("mean_tilt")
            }
            if candidate.impactActiveFraction >
                baseline.impactActiveFraction {
                improvedDimensions.append("impact_active_fraction")
            }
            if candidate.maximumImpactTilt < baseline.maximumImpactTilt {
                improvedDimensions.append("maximum_impact_tilt")
            }
            if candidate.minimumImpactRootHeight >
                baseline.minimumImpactRootHeight {
                improvedDimensions.append("minimum_impact_root_height")
            }
            var regressedDimensions: [String] = []
            if candidate.hitIncidence > baseline.hitIncidence {
                regressedDimensions.append("projectile_contact_incidence")
            }
            if candidate.missIncidence < baseline.missIncidence {
                regressedDimensions.append("clean_miss_incidence")
            }
            if candidate.balanceFailureIncidence >
                baseline.balanceFailureIncidence {
                regressedDimensions.append("balance_failure_incidence")
            }
            if candidate.meanReward < baseline.meanReward {
                regressedDimensions.append("mean_reward")
            }
            if candidate.meanTaskReward < baseline.meanTaskReward {
                regressedDimensions.append("mean_task_reward")
            }
            if candidate.meanTracking < baseline.meanTracking {
                regressedDimensions.append("mean_tracking_score")
            }
            if candidate.meanRootHeight < baseline.meanRootHeight {
                regressedDimensions.append("mean_root_height")
            }
            if candidate.meanTilt > baseline.meanTilt {
                regressedDimensions.append("mean_tilt")
            }
            if candidate.impactActiveFraction <
                baseline.impactActiveFraction {
                regressedDimensions.append("impact_active_fraction")
            }
            if candidate.maximumImpactTilt > baseline.maximumImpactTilt {
                regressedDimensions.append("maximum_impact_tilt")
            }
            if candidate.minimumImpactRootHeight <
                baseline.minimumImpactRootHeight {
                regressedDimensions.append("minimum_impact_root_height")
            }
            let result: [String: Any] = [
                "benchmark": "swift_native_multi_seed_dodge_promotion",
                "seeds": options.seeds.map { NSNumber(value: $0) },
                "baseline_policy": options.baselinePolicy!,
                "candidate_policy": options.candidatePolicy!,
                "baseline": baseline.json,
                "candidate": candidate.json,
                "hit_count_delta": candidate.hits - baseline.hits,
                "hit_rate_delta": candidate.hitRate - baseline.hitRate,
                "balance_failure_delta":
                    candidate.balanceFailures - baseline.balanceFailures,
                "projectile_contacts_per_million_environment_steps_delta":
                    1.0e6 *
                    (candidate.hitIncidence - baseline.hitIncidence),
                "clean_misses_per_million_environment_steps_delta":
                    1.0e6 *
                    (candidate.missIncidence - baseline.missIncidence),
                "balance_failures_per_million_environment_steps_delta":
                    1.0e6 *
                    (candidate.balanceFailureIncidence -
                     baseline.balanceFailureIncidence),
                "progress_balance_regression_tolerance_per_million_environment_steps":
                    1.0e6 * balanceRegressionTolerance,
                "mean_reward_delta":
                    candidate.meanReward - baseline.meanReward,
                "mean_task_reward_delta":
                    candidate.meanTaskReward - baseline.meanTaskReward,
                "mean_tracking_score_delta":
                    candidate.meanTracking - baseline.meanTracking,
                "mean_root_height_delta":
                    candidate.meanRootHeight - baseline.meanRootHeight,
                "mean_tilt_delta":
                    candidate.meanTilt - baseline.meanTilt,
                "impact_active_fraction_delta":
                    candidate.impactActiveFraction -
                    baseline.impactActiveFraction,
                "maximum_impact_tilt_delta":
                    candidate.maximumImpactTilt -
                    baseline.maximumImpactTilt,
                "minimum_impact_root_height_delta":
                    candidate.minimumImpactRootHeight -
                    baseline.minimumImpactRootHeight,
                "progress_dimensions_improved": improvedDimensions,
                "progress_dimensions_regressed": regressedDimensions,
                "progressed": progressed,
                "promoted": promoted,
            ]
            let data = try JSONSerialization.data(
                withJSONObject: result,
                options: [.prettyPrinted, .sortedKeys]
            )
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write(Data("\n".utf8))
        } catch {
            FileHandle.standardError.write(
                Data("dodge_promotion failed: \(error)\n".utf8)
            )
            Darwin.exit(1)
        }
    }
}
