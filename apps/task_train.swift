import Foundation
import Darwin

private struct Options {
    var environments = 32
    var steps = 24
    var updates = 100
    var chunk = 8
    var surface = MetalRoboLocomotionSurface.terrain
    var solver = MetalRoboTaskSolver.tgs
    var seed: UInt64 = 20_260_731
    var metallib = "build/shaders/MetalRobo.metallib"
    var nativeLibrary = "build/lib/libmetalrobo.dylib"
    var mlxPython: String?
    var pythonRoot = "python"
    var policyPack: String?
    var initializePolicyID: String?
    var updatedPolicyPack: String?
    var deploymentPolicyPack: String?
    var rolloutPack: String?
    var learnerState: String?
    var worldPack: String?
    var taskPack: String?
    var urdf: String?
    var srdf: String?
    var updateEpochs = 5
    // Zero selects four minibatches per update, matching the bundled PPO
    // contract independently of environment and horizon counts.
    var minibatchSize = 0
    var learningRate = 1.0e-3
    var clipRatio = 0.2
    var valueCoefficient = 1.0
    var entropyCoefficient = 0.01
    var initialLogStandardDeviation = 0.0
    var maximumGradientNorm = 1.0
    var targetKL = 0.01
    var discount = 0.99
    var gaeLambda = 0.95
    var learnerSeed = 1
    var learnerTimeoutSeconds = 120.0
    var verbose = false

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
            case "--updates":
                updates = try Self.integer(value(), option)
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
            case "--native-library":
                nativeLibrary = try value()
                index += 1
            case "--mlx-python":
                mlxPython = try value()
                index += 1
            case "--python-root":
                pythonRoot = try value()
                index += 1
            case "--policy-pack":
                policyPack = try value()
                index += 1
            case "--initialize-policy":
                initializePolicyID = try value()
                index += 1
            case "--updated-policy-pack":
                updatedPolicyPack = try value()
                index += 1
            case "--deployment-policy-pack":
                deploymentPolicyPack = try value()
                index += 1
            case "--rollout-pack":
                rolloutPack = try value()
                index += 1
            case "--learner-state":
                learnerState = try value()
                index += 1
            case "--world-pack":
                worldPack = try value()
                index += 1
            case "--task-pack":
                taskPack = try value()
                index += 1
            case "--urdf":
                urdf = try value()
                index += 1
            case "--srdf":
                srdf = try value()
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
            case "--update-epochs":
                updateEpochs = try Self.integer(value(), option)
                index += 1
            case "--minibatch-size":
                minibatchSize = try Self.integer(value(), option)
                index += 1
            case "--learning-rate":
                learningRate = try Self.double(value(), option)
                index += 1
            case "--clip-ratio":
                clipRatio = try Self.double(value(), option)
                index += 1
            case "--value-coefficient":
                valueCoefficient = try Self.double(value(), option)
                index += 1
            case "--entropy-coefficient":
                entropyCoefficient = try Self.double(value(), option)
                index += 1
            case "--initial-log-standard-deviation":
                initialLogStandardDeviation =
                    try Self.double(value(), option)
                index += 1
            case "--maximum-gradient-norm":
                maximumGradientNorm =
                    try Self.double(value(), option)
                index += 1
            case "--target-kl":
                targetKL = try Self.double(value(), option)
                index += 1
            case "--discount":
                discount = try Self.double(value(), option)
                index += 1
            case "--gae-lambda":
                gaeLambda = try Self.double(value(), option)
                index += 1
            case "--learner-seed":
                learnerSeed = try Self.integer(value(), option)
                index += 1
            case "--learner-timeout-seconds":
                learnerTimeoutSeconds =
                    try Self.double(value(), option)
                index += 1
            case "--verbose":
                verbose = true
            default:
                throw MetalRoboTaskRolloutError.invalidShape(
                    "Unknown option \(option)."
                )
            }
            index += 1
        }
        guard environments > 0,
              steps > 0,
              updates > 0,
              chunk > 0,
              updateEpochs > 0,
              minibatchSize >= 0,
              learnerSeed >= 0,
              let mlxPython,
              !mlxPython.isEmpty,
              let policyPack,
              !policyPack.isEmpty,
              let updatedPolicyPack,
              !updatedPolicyPack.isEmpty,
              let rolloutPack,
              !rolloutPack.isEmpty
        else {
            throw MetalRoboTaskRolloutError.invalidShape(
                "Positive rollout sizes and all policy, rollout, and MLX paths are required."
            )
        }
        let (sampleCount, sampleOverflow) =
            environments.multipliedReportingOverflow(
                by: steps
            )
        guard !sampleOverflow else {
            throw MetalRoboTaskRolloutError.invalidShape(
                "Rollout sample count overflows Int."
            )
        }
        if minibatchSize == 0 {
            minibatchSize = max(
                sampleCount / 4 +
                    (sampleCount % 4 == 0 ? 0 : 1),
                1
            )
        }
        if learnerState == nil {
            learnerState =
                updatedPolicyPack + ".learner.safetensors"
        }
        if deploymentPolicyPack == nil {
            deploymentPolicyPack =
                updatedPolicyPack + ".deployment.policypack"
        }
        let ppoValues = [
            learningRate,
            clipRatio,
            valueCoefficient,
            entropyCoefficient,
            initialLogStandardDeviation,
            maximumGradientNorm,
            targetKL,
            discount,
            gaeLambda,
            learnerTimeoutSeconds,
        ]
        guard ppoValues.allSatisfy(\.isFinite),
              learningRate > 0,
              clipRatio > 0,
              valueCoefficient >= 0,
              entropyCoefficient >= 0,
              initialLogStandardDeviation >= -5,
              initialLogStandardDeviation <= 2,
              maximumGradientNorm > 0,
              targetKL >= 0,
              learnerTimeoutSeconds > 0,
              discount >= 0,
              discount <= 1,
              gaeLambda >= 0,
              gaeLambda <= 1
        else {
            throw MetalRoboTaskRolloutError.invalidShape(
                "PPO scalar options are invalid."
            )
        }
        if worldPack != nil && urdf != nil {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--world-pack and --urdf are mutually exclusive."
            )
        }
        if (worldPack != nil || urdf != nil) != (taskPack != nil) {
            throw MetalRoboTaskRolloutError.invalidShape(
                "Imported mechanics require exactly one --task-pack."
            )
        }
        if srdf != nil && urdf == nil {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--srdf requires --urdf."
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

    private static func double(
        _ value: String,
        _ option: String
    ) throws -> Double {
        guard let parsed = Double(value) else {
            throw MetalRoboTaskRolloutError.invalidShape(
                "\(option) requires a number."
            )
        }
        return parsed
    }
}

private func mlxEnvironment(options: Options) -> [String: String] {
    var environment = ProcessInfo.processInfo.environment
    let existing = environment["PYTHONPATH"] ?? ""
    environment["PYTHONPATH"] =
        existing.isEmpty
        ? options.pythonRoot
        : "\(options.pythonRoot):\(existing)"
    let libraryDirectory = URL(
        fileURLWithPath: options.nativeLibrary
    ).deletingLastPathComponent().path
    let existingLibraries =
        environment["DYLD_LIBRARY_PATH"] ?? ""
    environment["DYLD_LIBRARY_PATH"] =
        existingLibraries.isEmpty
        ? libraryDirectory
        : "\(libraryDirectory):\(existingLibraries)"
    return environment
}

private func initializePolicyIfRequested(
    options: Options,
    layout: MetalRoboTaskRolloutLayout
) throws {
    guard let identifier = options.initializePolicyID else {
        return
    }
    guard let executable = options.mlxPython,
          let policyPack = options.policyPack,
          !identifier.isEmpty
    else {
        throw MetalRoboTaskRolloutError.invalidShape(
            "Policy initialization paths and identity are incomplete."
        )
    }
    if FileManager.default.fileExists(atPath: policyPack) {
        throw MetalRoboTaskRolloutError.invalidShape(
            "--initialize-policy refuses to overwrite an existing PolicyPack."
        )
    }
    try FileManager.default.createDirectory(
        at: URL(fileURLWithPath: policyPack)
            .deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    let process = Process()
    let output = Pipe()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = [
        "-m",
        "metalrobo.mlx_policy_worker",
        "initialize",
        "--actor-observations",
        String(layout.actorObservationCount),
        "--critic-observations",
        String(layout.criticObservationCount),
        "--actions",
        String(layout.actionCount),
        "--policy-id",
        identifier,
        "--output",
        policyPack,
        "--native-library",
        options.nativeLibrary,
        "--seed",
        String(options.learnerSeed),
        "--initial-log-standard-deviation",
        String(options.initialLogStandardDeviation),
    ]
    process.environment = mlxEnvironment(options: options)
    process.standardOutput = output
    process.standardError = FileHandle.standardError
    try process.run()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
        throw MetalRoboTaskRolloutError.native(
            "MLX policy initialization exited with status "
            + "\(process.terminationStatus)."
        )
    }
    let data = output.fileHandleForReading.readDataToEndOfFile()
    guard let record = try JSONSerialization.jsonObject(
        with: data
    ) as? [String: Any],
          record["status"] as? String == "initialized"
    else {
        throw MetalRoboTaskRolloutError.native(
            "MLX policy initialization returned an invalid record."
        )
    }
}

private final class MLXLearnerWorker {
    private let process = Process()
    private let input = Pipe()
    private let output = Pipe()
    private var pendingOutput = Data()
    private let responseTimeoutMilliseconds: Int32
    private(set) var revision: UInt64
    private(set) var actorObservationCount: Int
    private(set) var criticObservationCount: Int
    private(set) var actionCount: Int
    private(set) var policyPackPath: String
    private(set) var deploymentPolicyPackPath: String
    private(set) var stateRestored: Bool
    private var closed = false

    init(options: Options) throws {
        guard let executable = options.mlxPython,
              let policyPack = options.policyPack,
              let outputPolicyPack = options.updatedPolicyPack,
              let deploymentPolicyPack =
                  options.deploymentPolicyPack,
              let learnerState = options.learnerState
        else {
            throw MetalRoboTaskRolloutError.invalidShape(
                "MLX learner paths are incomplete."
            )
        }
        process.executableURL = URL(fileURLWithPath: executable)
        responseTimeoutMilliseconds = Int32(
            min(
                ceil(options.learnerTimeoutSeconds * 1_000),
                Double(Int32.max)
            )
        )
        process.arguments = [
            "-m",
            "metalrobo.mlx_policy_worker",
            "serve",
            "--policy-pack",
            policyPack,
            "--output-policy-pack",
            outputPolicyPack,
            "--deployment-policy-pack",
            deploymentPolicyPack,
            "--learner-state",
            learnerState,
            "--native-library",
            options.nativeLibrary,
            "--update-epochs",
            String(options.updateEpochs),
            "--minibatch-size",
            String(options.minibatchSize),
            "--learning-rate",
            String(options.learningRate),
            "--clip-ratio",
            String(options.clipRatio),
            "--value-coefficient",
            String(options.valueCoefficient),
            "--entropy-coefficient",
            String(options.entropyCoefficient),
            "--maximum-gradient-norm",
            String(options.maximumGradientNorm),
            "--target-kl",
            String(options.targetKL),
            "--discount",
            String(options.discount),
            "--gae-lambda",
            String(options.gaeLambda),
            "--initial-log-standard-deviation",
            String(options.initialLogStandardDeviation),
            "--seed",
            String(options.learnerSeed),
        ]
        process.environment = mlxEnvironment(options: options)
        process.standardInput = input
        process.standardOutput = output
        process.standardError = FileHandle.standardError
        try process.run()

        revision = 0
        actorObservationCount = 0
        criticObservationCount = 0
        actionCount = 0
        policyPackPath = ""
        deploymentPolicyPackPath = ""
        stateRestored = false
        let ready = try response()
        guard ready["status"] as? String == "ready",
              let revisionValue =
                  ready["policy_revision"] as? NSNumber,
              let actorValue =
                  ready["actor_observation_count"] as? NSNumber,
              let criticValue =
                  ready["critic_observation_count"] as? NSNumber,
              let actionValue =
                  ready["action_count"] as? NSNumber,
              let readyPolicyPack =
                  ready["policy_pack"] as? String,
              let readyDeploymentPolicyPack =
                  ready["deployment_policy_pack"] as? String,
              let restored =
                  ready["learner_state_restored"] as? Bool
        else {
            throw MetalRoboTaskRolloutError.native(
                "MLX learner did not publish a valid ready record."
            )
        }
        revision = revisionValue.uint64Value
        actorObservationCount = actorValue.intValue
        criticObservationCount = criticValue.intValue
        actionCount = actionValue.intValue
        policyPackPath = readyPolicyPack
        deploymentPolicyPackPath =
            readyDeploymentPolicyPack
        stateRestored = restored
    }

    deinit {
        if process.isRunning {
            process.terminate()
        }
    }

    func update(
        rolloutPack: String
    ) throws -> [String: Any] {
        try request(
            [
                "operation": "update",
                "rollout_pack": rolloutPack,
            ]
        )
        let result = try response()
        guard result["status"] as? String == "updated",
              let before =
                  result["policy_revision_before"] as? NSNumber,
              let after =
                  result["policy_revision_after"] as? NSNumber,
              let deploymentPolicyPack =
                  result["deployment_policy_pack"] as? String,
              before.uint64Value == revision,
              after.uint64Value == revision + 1,
              deploymentPolicyPack ==
                  deploymentPolicyPackPath
        else {
            let message =
                result["error"] as? String ??
                "MLX learner returned an invalid update record."
            throw MetalRoboTaskRolloutError.native(message)
        }
        revision = after.uint64Value
        return result
    }

    func close() throws {
        guard !closed else {
            return
        }
        try request(["operation": "close"])
        let result = try response()
        guard result["status"] as? String == "closed" else {
            throw MetalRoboTaskRolloutError.native(
                "MLX learner did not close cleanly."
            )
        }
        closed = true
        try input.fileHandleForWriting.close()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw MetalRoboTaskRolloutError.native(
                "MLX learner exited with status \(process.terminationStatus)."
            )
        }
    }

    private func request(_ object: [String: Any]) throws {
        var data = try JSONSerialization.data(
            withJSONObject: object
        )
        data.append(0x0a)
        try input.fileHandleForWriting.write(contentsOf: data)
    }

    private func response() throws -> [String: Any] {
        while true {
            if let newline = pendingOutput.firstIndex(of: 0x0a) {
                let line = pendingOutput[..<newline]
                pendingOutput.removeSubrange(
                    ...newline
                )
                let value = try JSONSerialization.jsonObject(
                    with: Data(line)
                )
                guard let record = value as? [String: Any] else {
                    throw MetalRoboTaskRolloutError.native(
                        "MLX learner emitted a non-object JSON record."
                    )
                }
                return record
            }
            var descriptor = pollfd(
                fd: output.fileHandleForReading.fileDescriptor,
                events: Int16(POLLIN | POLLHUP | POLLERR),
                revents: 0
            )
            var pollStatus: Int32
            repeat {
                pollStatus = Darwin.poll(
                    &descriptor,
                    1,
                    responseTimeoutMilliseconds
                )
            } while pollStatus < 0 && errno == EINTR
            if pollStatus == 0 {
                throw MetalRoboTaskRolloutError.native(
                    "MLX learner did not answer within "
                    + "\(Double(responseTimeoutMilliseconds) / 1_000) seconds."
                )
            }
            guard pollStatus > 0,
                  descriptor.revents & Int16(POLLNVAL) == 0
            else {
                throw MetalRoboTaskRolloutError.native(
                    "MLX learner output polling failed."
                )
            }
            var bytes = [UInt8](
                repeating: 0,
                count: 64 * 1_024
            )
            let count = bytes.withUnsafeMutableBytes {
                storage -> Int in
                var result: Int
                repeat {
                    result = Darwin.read(
                        descriptor.fd,
                        storage.baseAddress,
                        storage.count
                    )
                } while result < 0 && errno == EINTR
                return result
            }
            guard count > 0 else {
                throw MetalRoboTaskRolloutError.native(
                    "MLX learner closed its output unexpectedly."
                )
            }
            pendingOutput.append(
                contentsOf: bytes.prefix(count)
            )
        }
    }
}

private func makeContext(
    options: Options
) throws -> (MetalRoboTaskRolloutContext, String) {
    let configuration = MetalRoboTaskRolloutConfiguration(
        environmentCount: UInt32(options.environments),
        surface: options.surface,
        solver: options.solver,
        seed: options.seed
    )
    if let worldPack = options.worldPack,
       let taskPack = options.taskPack
    {
        return (
            try MetalRoboTaskRolloutContext(
                worldPack: URL(fileURLWithPath: worldPack),
                taskPack: URL(fileURLWithPath: taskPack),
                configuration: configuration,
                metallibPath: options.metallib
            ),
            "world_pack"
        )
    }
    if let urdf = options.urdf,
       let taskPack = options.taskPack
    {
        return (
            try MetalRoboTaskRolloutContext(
                importedURDF: URL(fileURLWithPath: urdf),
                srdf: options.srdf.map {
                    URL(fileURLWithPath: $0)
                },
                taskPack: URL(fileURLWithPath: taskPack),
                configuration: configuration,
                metallibPath: options.metallib
            ),
            "urdf"
        )
    }
    return (
        try MetalRoboTaskRolloutContext(
            unitreeG1: configuration,
            metallibPath: options.metallib
        ),
        "bundled_g1"
    )
}

@main
private enum TaskTrainMain {
    static func main() {
        do {
            let options = try Options(
                arguments: CommandLine.arguments
            )
            let (context, worldSource) =
                try makeContext(options: options)
            try initializePolicyIfRequested(
                options: options,
                layout: context.layout
            )
            guard let updatedPolicyPack =
                      options.updatedPolicyPack,
                  let deploymentPolicyPack =
                      options.deploymentPolicyPack,
                  let rolloutPack = options.rolloutPack,
                  let learnerState = options.learnerState
            else {
                throw MetalRoboTaskRolloutError.invalidShape(
                    "Training artifact paths are incomplete."
                )
            }
            let learner = try MLXLearnerWorker(options: options)
            try context.loadPolicy(
                at: URL(fileURLWithPath: learner.policyPackPath)
            )
            let layout = context.layout
            guard learner.revision != 0,
                  learner.actorObservationCount ==
                      layout.actorObservationCount,
                  learner.criticObservationCount ==
                      layout.criticObservationCount,
                  learner.actionCount == layout.actionCount
            else {
                throw MetalRoboTaskRolloutError.invalidShape(
                    "PolicyPack dimensions do not match the compiled task."
                )
            }
            let initialRevision = learner.revision
            var installedRevision = learner.revision
            let warmup = try context.advanceWithPolicy(
                controlStepCount: 1,
                policyRevision: installedRevision
            )
            guard warmup.failedEnvironmentSteps == 0,
                  try context.transitions(
                      controlStepCount: 1
                  ).allSatisfy({
                      $0.policyRevision == installedRevision
                  })
            else {
                throw MetalRoboTaskRolloutError.native(
                    "Initial PolicyPack failed native warmup."
                )
            }
            try context.reset(seed: options.seed)

            let baseline = context.layout
            let clock = ContinuousClock()
            let start = clock.now
            var totalResets = 0
            var maximumContacts = 0
            var maximumManifolds = 0
            var stageHighWater: [String: Int] = [:]
            var failedSteps = 0
            var lastLearning: [String: Any] = [:]
            let samplesPerUpdate =
                options.environments * options.steps
            var actorObservations: [Float] = []
            var criticObservations: [Float] = []
            var latents: [Float] = []
            var logProbabilities: [Float] = []
            var values: [Float] = []
            var transitions: [MetalRoboTaskTransition] = []
            actorObservations.reserveCapacity(
                samplesPerUpdate *
                    layout.actorObservationCount
            )
            criticObservations.reserveCapacity(
                samplesPerUpdate *
                    layout.criticObservationCount
            )
            latents.reserveCapacity(
                samplesPerUpdate * layout.actionCount
            )
            logProbabilities.reserveCapacity(samplesPerUpdate)
            values.reserveCapacity(samplesPerUpdate)
            transitions.reserveCapacity(samplesPerUpdate)

            for updateIndex in 0..<options.updates {
                actorObservations.removeAll(
                    keepingCapacity: true
                )
                criticObservations.removeAll(
                    keepingCapacity: true
                )
                latents.removeAll(keepingCapacity: true)
                logProbabilities.removeAll(keepingCapacity: true)
                values.removeAll(keepingCapacity: true)
                transitions.removeAll(keepingCapacity: true)
                var completed = 0
                while completed < options.steps {
                    let stepCount = min(
                        options.chunk,
                        options.steps - completed
                    )
                    let advance = try context.advanceWithPolicy(
                        controlStepCount: stepCount,
                        policyRevision: installedRevision,
                        evaluateFinalPolicy:
                            completed + stepCount == options.steps
                    )
                    guard advance.failedEnvironmentSteps == 0 else {
                        throw MetalRoboTaskRolloutError.native(
                            "Native rollout returned a GPU failure."
                        )
                    }
                    let transitionOffset =
                        transitions.count
                    try context.appendCurrentPolicyRollout(
                        controlStepCount: stepCount,
                        actorObservations:
                            &actorObservations,
                        criticObservations:
                            &criticObservations,
                        latents: &latents,
                        logProbabilities:
                            &logProbabilities,
                        values: &values,
                        transitions: &transitions
                    )
                    guard transitions[
                        transitionOffset...
                    ].allSatisfy({
                        $0.policyRevision == installedRevision
                    }) else {
                        throw MetalRoboTaskRolloutError.native(
                            "Policy revision changed inside a rollout."
                        )
                    }
                    totalResets += advance.hostRequestedResets
                    maximumContacts = max(
                        maximumContacts,
                        advance.maximumActiveContacts
                    )
                    maximumManifolds = max(
                        maximumManifolds,
                        advance.maximumManifolds
                    )
                    for (stage, count) in
                        advance.stageHighWater
                    {
                        stageHighWater[stage] = max(
                            stageHighWater[stage, default: 0],
                            count
                        )
                    }
                    failedSteps += advance.failedEnvironmentSteps
                    completed += stepCount
                }
                let rollout = MetalRoboPolicyRolloutBatch(
                    controlStepCount: options.steps,
                    environmentCount: options.environments,
                    actorObservationCount:
                        layout.actorObservationCount,
                    criticObservationCount:
                        layout.criticObservationCount,
                    actionCount: layout.actionCount,
                    actorObservations: actorObservations,
                    criticObservations: criticObservations,
                    latents: latents,
                    logProbabilities: logProbabilities,
                    values: values,
                    transitions: transitions
                )
                let bootstrapValues =
                    try context.bootstrapPolicyValues()
                try context.writePolicyRolloutPack(
                    rollout,
                    bootstrapValues: bootstrapValues,
                    id: "swift_native_training_rollout",
                    to: URL(fileURLWithPath: rolloutPack)
                )
                lastLearning = try learner.update(
                    rolloutPack: rolloutPack
                )
                try context.loadPolicy(
                    at: URL(fileURLWithPath: updatedPolicyPack)
                )
                installedRevision = learner.revision
                if options.verbose {
                    let reward =
                        (lastLearning["mean_reward"] as? NSNumber)?
                            .doubleValue ?? 0
                    let tracking =
                        (
                            lastLearning[
                                "mean_tracking_score"
                            ] as? NSNumber
                        )?.doubleValue ?? 0
                    let height =
                        (
                            lastLearning[
                                "mean_root_height"
                            ] as? NSNumber
                        )?.doubleValue ?? 0
                    let tilt =
                        (lastLearning["mean_tilt"] as? NSNumber)?
                            .doubleValue ?? 0
                    let done =
                        (lastLearning["done_count"] as? NSNumber)?
                            .intValue ?? 0
                    let loss =
                        (lastLearning["loss"] as? NSNumber)?
                            .doubleValue ?? 0
                    let kl =
                        (lastLearning["kl_divergence"] as? NSNumber)?
                            .doubleValue ?? 0
                    let learningRate =
                        (lastLearning["learning_rate"] as? NSNumber)?
                            .doubleValue ?? 0
                    let actionStandardDeviation =
                        (
                            lastLearning[
                                "mean_action_standard_deviation"
                            ] as? NSNumber
                        )?.doubleValue ?? 0
                    FileHandle.standardError.write(
                        Data(
                            (
                                "update=\(updateIndex + 1) " +
                                "revision=\(installedRevision) " +
                                "reward=\(reward) " +
                                "tracking=\(tracking) " +
                                "height=\(height) " +
                                "tilt=\(tilt) " +
                                "done=\(done) loss=\(loss) " +
                                "kl=\(kl) lr=\(learningRate) " +
                                "std=\(actionStandardDeviation)\n"
                            ).utf8
                        )
                    )
                }
            }
            try learner.close()

            let end = clock.now
            let elapsed = start.duration(to: end)
            let seconds =
                Double(elapsed.components.seconds) +
                Double(elapsed.components.attoseconds) / 1e18
            let finalLayout = context.layout
            let sampleCount =
                options.environments *
                options.steps *
                options.updates
            let output: [String: Any] = [
                "operation": "swift_native_policy_training",
                "scheduler": "swift",
                "physics": "metal",
                "learner": "mlx",
                "world_source": worldSource,
                "device": context.deviceName,
                "environments": options.environments,
                "steps_per_update": options.steps,
                "updates": options.updates,
                "control_steps_per_submission": options.chunk,
                "training_samples": sampleCount,
                "initial_policy_revision": initialRevision,
                "final_policy_revision": installedRevision,
                "learner_state_restored":
                    learner.stateRestored,
                "native_submission_count": NSNumber(
                    value:
                        finalLayout.submissionCount -
                        baseline.submissionCount
                ),
                "end_to_end_environment_steps_per_second":
                    Double(sampleCount) / seconds,
                "elapsed_seconds": seconds,
                "gpu_milliseconds":
                    finalLayout.totalGPUMilliseconds -
                    baseline.totalGPUMilliseconds,
                "submission_milliseconds":
                    finalLayout.totalSubmissionMilliseconds -
                    baseline.totalSubmissionMilliseconds,
                "retained_buffer_bytes":
                    finalLayout.retainedBufferBytes,
                "transient_private_bytes":
                    finalLayout.transientPrivateBytes,
                "host_requested_resets": totalResets,
                "maximum_active_contacts": maximumContacts,
                "maximum_manifolds": maximumManifolds,
                "stage_high_water": stageHighWater,
                "failed_environment_steps": failedSteps,
                "rollout_pack": rolloutPack,
                "policy_pack": updatedPolicyPack,
                "deployment_policy_pack":
                    deploymentPolicyPack,
                "learner_state": learnerState,
                "last_learning_update": lastLearning,
            ]
            let data = try JSONSerialization.data(
                withJSONObject: output,
                options: [.prettyPrinted, .sortedKeys]
            )
            print(String(decoding: data, as: UTF8.self))
        } catch {
            FileHandle.standardError.write(
                Data("task_train failed: \(error)\n".utf8)
            )
            Darwin.exit(1)
        }
    }
}
