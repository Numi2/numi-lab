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
    var perSeed: [[String: Any]] = []

    var trials: Int { hits + misses }
    var hitRate: Double {
        Double(hits) / Double(max(trials, 1))
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
        aggregate.hits += hits
        aggregate.misses += misses
        aggregate.balanceFailures += balance
        aggregate.failedEnvironmentSteps += failed
        aggregate.perSeed.append([
            "seed": NSNumber(value: seed),
            "hits": hits,
            "misses": misses,
            "balance_failures": balance,
            "failed_environment_steps": failed,
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
