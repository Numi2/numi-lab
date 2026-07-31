import Foundation
import Darwin

private struct SplitMix64 {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9e3779b97f4a7c15
        var value = state
        value =
            (value ^ (value >> 30)) &*
            0xbf58476d1ce4e5b9
        value =
            (value ^ (value >> 27)) &*
            0x94d049bb133111eb
        return value ^ (value >> 31)
    }

    mutating func signedUnit() -> Float {
        let value = UInt32(truncatingIfNeeded: next() >> 40)
        return 2 * Float(value) / Float(1 << 24) - 1
    }
}

private struct Options {
    var environments = 32
    var steps = 48
    var repeats = 20
    var chunk = 8
    var surface = MetalRoboLocomotionSurface.terrain
    var solver = MetalRoboTaskSolver.tgs
    var seed: UInt64 = 20_260_731
    var metallib = "build/shaders/MetalRobo.metallib"
    var verbose = false
    var nativePolicy = false
    var policyPack: String?

    init(arguments: [String]) throws {
        var index = 1
        while index < arguments.count {
            let option = arguments[index]
            func value() throws -> String {
                guard index + 1 < arguments.count else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "\(option) requires a value."
                    )
                }
                return arguments[index + 1]
            }
            switch option {
            case "--envs":
                environments = try Self.integer(value(), option)
                index += 1
            case "--steps":
                steps = try Self.integer(value(), option)
                index += 1
            case "--repeats":
                repeats = try Self.integer(value(), option)
                index += 1
            case "--chunk":
                chunk = try Self.integer(value(), option)
                index += 1
            case "--seed":
                guard let parsed = UInt64(try value()) else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--seed requires an unsigned integer."
                    )
                }
                seed = parsed
                index += 1
            case "--metallib":
                metallib = try value()
                index += 1
            case "--scene":
                switch try value() {
                case "ground":
                    surface = .ground
                case "terrain":
                    surface = .terrain
                default:
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--scene must be ground or terrain."
                    )
                }
                index += 1
            case "--solver-mode":
                switch try value() {
                case "pgs", "throughput_pgs":
                    solver = .pgs
                case "tgs", "throughput_tgs":
                    solver = .tgs
                default:
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--solver-mode must be pgs or tgs."
                    )
                }
                index += 1
            case "--verbose":
                verbose = true
            case "--native-policy":
                nativePolicy = true
            case "--policy-pack":
                policyPack = try value()
                index += 1
            default:
                throw MetalRoboTaskRolloutError.invalidShape(
                    "Unknown option \(option)."
                )
            }
            index += 1
        }
        guard environments > 0,
              steps > 0,
              repeats > 0,
              chunk > 0
        else {
            throw MetalRoboTaskRolloutError.invalidShape(
                "envs, steps, repeats, and chunk must be positive."
            )
        }
        if nativePolicy && policyPack != nil {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--native-policy and --policy-pack are mutually exclusive."
            )
        }
    }

    private static func integer(
        _ value: String,
        _ option: String
    ) throws -> Int {
        guard let parsed = Int(value) else {
            throw MetalRoboTaskRolloutError.invalidShape(
                "\(option) requires an integer."
            )
        }
        return parsed
    }
}

private func actions(
    startStep: Int,
    stepCount: Int,
    environmentCount: Int,
    actionCount: Int,
    generator: inout SplitMix64
) -> [Float] {
    var result = [Float](
        repeating: 0,
        count: stepCount * environmentCount * actionCount
    )
    for step in 0..<stepCount {
        let globalStep = startStep + step
        for environment in 0..<environmentCount {
            for joint in 0..<actionCount {
                let phase =
                    0.071 * Double(globalStep) +
                    0.113 * Double(environment) +
                    0.173 * Double(joint)
                let value =
                    0.28 * Float(sin(phase)) +
                    0.04 * generator.signedUnit()
                let index =
                    (step * environmentCount + environment) *
                    actionCount + joint
                result[index] = min(max(value, -1), 1)
            }
        }
    }
    return result
}

private func masks(
    startStep: Int,
    stepCount: Int,
    environmentCount: Int,
    pending: inout [Bool]
) -> [UInt32] {
    var result = [UInt32](
        repeating: 0,
        count: stepCount * environmentCount
    )
    for step in 0..<stepCount {
        let globalStep = startStep + step
        for environment in 0..<environmentCount {
            let requested =
                (step == 0 && pending[environment]) ||
                (
                    globalStep > 0 &&
                    (globalStep + 37 * environment) % 257 == 0
                )
            if requested {
                result[step * environmentCount + environment] = 1
                pending[environment] = false
            }
        }
    }
    return result
}

@main
private enum TaskRolloutMain {
    static func main() {
        do {
            let options = try Options(
                arguments: CommandLine.arguments
            )
            let context = try MetalRoboTaskRolloutContext(
                unitreeG1: .init(
                    environmentCount: UInt32(options.environments),
                    surface: options.surface,
                    solver: options.solver,
                    seed: options.seed
                ),
                metallibPath: options.metallib
            )
            let usesCompiledPolicy =
                options.nativePolicy || options.policyPack != nil
            if let policyPack = options.policyPack {
                try context.loadPolicy(
                    at: URL(fileURLWithPath: policyPack)
                )
            } else if options.nativePolicy {
                let layout = context.layout
                let hiddenCount = 64
                try context.setPolicy(
                    MetalRoboPolicyPack(
                        id: "native_rollout_zero_policy",
                        revision: 7,
                        layers: [
                            .init(
                                inputCount:
                                    layout.actorObservationCount,
                                outputCount: hiddenCount,
                                activation: .tanh,
                                weights: [Float](
                                    repeating: 0,
                                    count:
                                        layout.actorObservationCount *
                                        hiddenCount
                                ),
                                bias: [Float](
                                    repeating: 0,
                                    count: hiddenCount
                                )
                            ),
                            .init(
                                inputCount: hiddenCount,
                                outputCount: layout.actionCount,
                                activation: .tanh,
                                weights: [Float](
                                    repeating: 0,
                                    count:
                                        hiddenCount *
                                        layout.actionCount
                                ),
                                bias: [Float](
                                    repeating: 0,
                                    count: layout.actionCount
                                )
                            ),
                        ]
                    )
                )
            }
            var installedPolicyRevision: UInt64 = 0
            if usesCompiledPolicy {
                _ = try context.advanceWithPolicy(
                    resetMasks: [UInt32](
                        repeating: 0,
                        count: options.environments
                    ),
                    controlStepCount: 1
                )
                let revisions = try context.transitions(
                    controlStepCount: 1
                ).map(\.policyRevision)
                guard let revision = revisions.first,
                      revision != 0,
                      revisions.allSatisfy({ $0 == revision })
                else {
                    throw MetalRoboTaskRolloutError.native(
                        "Native policy revision was not attached to transitions."
                    )
                }
                installedPolicyRevision = revision
            } else {
                _ = try context.advance(
                    normalizedActions: [Float](
                        repeating: 0,
                        count:
                            options.environments *
                            context.layout.actionCount
                    ),
                    resetMasks: [UInt32](
                        repeating: 0,
                        count: options.environments
                    ),
                    controlStepCount: 1
                )
            }
            try context.reset(seed: options.seed)
            let baseline = context.layout
            var generator = SplitMix64(seed: options.seed)
            var pending = [Bool](
                repeating: false,
                count: options.environments
            )
            var globalStep = 0
            var totalResets = 0
            var maximumContacts = 0
            var maximumManifolds = 0
            var failedSteps = 0
            let clock = ContinuousClock()
            let start = clock.now

            for repeatIndex in 0..<options.repeats {
                let repeatStart = clock.now
                var completed = 0
                while completed < options.steps {
                    let stepCount = min(
                        options.chunk,
                        options.steps - completed
                    )
                    let resetBatch = masks(
                        startStep: globalStep,
                        stepCount: stepCount,
                        environmentCount: options.environments,
                        pending: &pending
                    )
                    let advance: MetalRoboTaskRolloutAdvance
                    if usesCompiledPolicy {
                        advance = try context.advanceWithPolicy(
                            resetMasks: resetBatch,
                            controlStepCount: stepCount,
                            policyRevision:
                                installedPolicyRevision
                        )
                    } else {
                        let actionBatch = actions(
                            startStep: globalStep,
                            stepCount: stepCount,
                            environmentCount:
                                options.environments,
                            actionCount:
                                context.layout.actionCount,
                            generator: &generator
                        )
                        advance = try context.advance(
                            normalizedActions: actionBatch,
                            resetMasks: resetBatch,
                            controlStepCount: stepCount,
                            policyRevision: 1
                        )
                    }
                    let statuses = try context.statusCodes(
                        controlStepCount: stepCount
                    )
                    guard statuses.allSatisfy({ $0 == 0 }) else {
                        throw MetalRoboTaskRolloutError.native(
                            "Task rollout returned a nonzero GPU status."
                        )
                    }
                    totalResets += advance.scheduledResets
                    maximumContacts = max(
                        maximumContacts,
                        advance.maximumActiveContacts
                    )
                    maximumManifolds = max(
                        maximumManifolds,
                        advance.maximumManifolds
                    )
                    failedSteps += advance.failedEnvironmentSteps
                    completed += stepCount
                    globalStep += stepCount
                }
                if options.verbose {
                    let elapsed =
                        repeatStart.duration(to: clock.now)
                    let measuredSubmissions =
                        context.layout.submissionCount -
                        baseline.submissionCount
                    print(
                        "repeat=\(repeatIndex + 1) " +
                        "seconds=\(elapsed.components.seconds) " +
                        "submissions=\(measuredSubmissions)"
                    )
                }
            }

            let elapsed = start.duration(to: clock.now)
            let seconds =
                Double(elapsed.components.seconds) +
                Double(elapsed.components.attoseconds) / 1e18
            let layout = context.layout
            let environmentSteps =
                options.environments *
                options.steps *
                options.repeats
            let output: [String: Any] = [
                "benchmark": "swift_native_task_rollout",
                "action_source":
                    options.policyPack != nil
                    ? "policy_pack"
                    : options.nativePolicy
                    ? "compiled_policy"
                    : "host_stream",
                "device": context.deviceName,
                "solver_mode":
                    options.solver == .tgs
                    ? "throughput_tgs"
                    : "throughput_pgs",
                "scene":
                    options.surface == .terrain
                    ? "terrain"
                    : "ground",
                "environments": options.environments,
                "steps_per_repeat": options.steps,
                "repeats": options.repeats,
                "control_steps_per_submission": options.chunk,
                "submission_count":
                    NSNumber(
                        value:
                            layout.submissionCount -
                            baseline.submissionCount
                    ),
                "environment_control_steps_per_second":
                    Double(environmentSteps) / seconds,
                "elapsed_seconds": seconds,
                "gpu_milliseconds":
                    layout.totalGPUMilliseconds -
                    baseline.totalGPUMilliseconds,
                "submission_milliseconds":
                    layout.totalSubmissionMilliseconds -
                    baseline.totalSubmissionMilliseconds,
                "retained_buffer_bytes":
                    layout.retainedBufferBytes,
                "immutable_private_bytes":
                    layout.immutablePrivateBytes,
                "persistent_state_private_bytes":
                    layout.persistentStatePrivateBytes,
                "transient_private_bytes":
                    layout.transientPrivateBytes,
                "shared_boundary_bytes":
                    layout.sharedBoundaryBytes,
                "peak_aliased_bytes":
                    layout.peakAliasedBytes,
                "scheduled_resets": totalResets,
                "maximum_active_contacts": maximumContacts,
                "maximum_manifolds": maximumManifolds,
                "failed_environment_steps": failedSteps,
            ]
            let data = try JSONSerialization.data(
                withJSONObject: output,
                options: [.prettyPrinted, .sortedKeys]
            )
            print(String(decoding: data, as: UTF8.self))
        } catch {
            FileHandle.standardError.write(
                Data("task_rollout failed: \(error)\n".utf8)
            )
            Darwin.exit(1)
        }
    }
}
