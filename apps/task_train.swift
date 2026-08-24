import Foundation
import Darwin
import AppKit

private struct Options {
    var environments = 32
    var steps = 24
    var stepsWereSpecified = false
    var updates = 100
    var chunk = 8
    var surface = MetalRoboLocomotionSurface.terrain
    var unitreeG1Task = MetalRoboUnitreeG1Task.velocity
    var seed: UInt64 = 20_260_731
    var metallib = "build/shaders/MetalRobo.metallib"
    var nativeLibrary = "build/lib/libmetalrobo.dylib"
    var mlxPython: String?
    var pythonRoot = "python"
    var policyPack: String?
    var basePolicyPack: String?
    var initializePolicyID: String?
    var zeroActorOutput = false
    var initializeActorPolicyPack: String?
    var initializeActorFreshCritic = false
    var actorObservationExtensionOffset: Int?
    var actorObservationExtensionMean: Double?
    var actorObservationExtensionInverseStandardDeviation = 1.0
    var updatedPolicyPack: String?
    var deploymentPolicyPack: String?
    var incumbentPolicyPack: String?
    var rolloutPack: String?
    var learnerState: String?
    var outputLearnerState: String?
    var checkpointDirectory: String?
    var checkpointInterval = 0
    var motionPack: String?
    var motionActivation = "impact"
    var interactionPack: String?
    var interactionClip: String?
    var interactionStudentAuthority: Float?
    var interactionResetPhaseFraction: Float?
    var interactionResetPhaseProbability: Float?
    var interactionResetMaximumPhase: Float?
    var interactionResetOnly = false
    var materializeArticulatedContactResponses = false
    var minimumDifficultyBand: Int?
    var maximumDifficultyBand: Int?
    var worldPack: String?
    var taskPack: String?
    var robotActuatorPack: String?
    var sensorProgramPack: String?
    var realityProgramPack: String?
    var urdf: String?
    var srdf: String?
    var g1VisualPackDirectory: String?
    var ballVisualPackDirectory: String?
    var visualEnvironmentPack: String?
    var visualObservationConfig: String?
    var birdFlowDove = false
    var birdFlowAmericanCrow = false
    var inspectionScene: String?
    var inspectionWidth = 640
    var inspectionHeight = 360
    var updateEpochs = 5
    // Zero derives the batch size from minibatchesPerEpoch so environment
    // scaling increases Apple-GPU matrix width instead of optimizer launches.
    var minibatchSize = 0
    var minibatchesPerEpoch = 0
    var motionMinibatchesPerEpoch = 8
    var motionUpdateEpochs = 1
    var motionMinibatchSize = 0
    var motionRewardCoefficient = 0.3
    var learningRate = 1.0e-3
    var minimumLearningRate = 1.0e-5
    var maximumLearningRate = 1.0e-2
    var fixedLearningRate = false
    var overrideResumedLearningRate = false
    var overrideResumedExploration = false
    var clipRatio = 0.2
    var valueCoefficient = 1.0
    var imaginationDistillationCoefficient = 1.0
    // The production default starts with a 0.2 standard deviation in policy
    // coordinates. With the bundled 0.25-radian action scale this is 0.05 rad
    // of target exploration, rather than a new order-one target every 20 ms.
    var entropyCoefficient = 0.001
    var initialLogStandardDeviation = -1.6094379124341003
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
                stepsWereSpecified = true
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
            case "--base-policy-pack":
                basePolicyPack = try value()
                index += 1
            case "--initialize-policy":
                initializePolicyID = try value()
                index += 1
            case "--zero-actor-output":
                zeroActorOutput = true
            case "--initialize-actor-policy-pack":
                initializeActorPolicyPack = try value()
                index += 1
            case "--initialize-actor-fresh-critic":
                initializeActorFreshCritic = true
            case "--actor-observation-extension-offset":
                actorObservationExtensionOffset = try Self.integer(
                    value(),
                    option
                )
                index += 1
            case "--actor-observation-extension-mean":
                guard let parsed = Double(try value()),
                      parsed.isFinite
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--actor-observation-extension-mean requires a finite value."
                    )
                }
                actorObservationExtensionMean = parsed
                index += 1
            case "--actor-observation-extension-inverse-standard-deviation":
                guard let parsed = Double(try value()),
                      parsed.isFinite,
                      parsed > 0
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--actor-observation-extension-inverse-standard-deviation requires a finite positive value."
                    )
                }
                actorObservationExtensionInverseStandardDeviation = parsed
                index += 1
            case "--updated-policy-pack":
                updatedPolicyPack = try value()
                index += 1
            case "--deployment-policy-pack":
                deploymentPolicyPack = try value()
                index += 1
            case "--incumbent-policy-pack":
                incumbentPolicyPack = try value()
                index += 1
            case "--rollout-pack":
                rolloutPack = try value()
                index += 1
            case "--learner-state":
                learnerState = try value()
                index += 1
            case "--output-learner-state":
                outputLearnerState = try value()
                index += 1
            case "--checkpoint-directory":
                checkpointDirectory = try value()
                index += 1
            case "--checkpoint-interval":
                checkpointInterval = try Self.integer(value(), option)
                index += 1
            case "--motion-pack":
                motionPack = try value()
                index += 1
            case "--motion-activation":
                motionActivation = try value()
                guard motionActivation == "impact" ||
                      motionActivation == "always"
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--motion-activation must be impact or always."
                    )
                }
                index += 1
            case "--interaction-pack":
                interactionPack = try value()
                index += 1
            case "--interaction-clip":
                interactionClip = try value()
                index += 1
            case "--interaction-student-authority":
                guard let parsed = Float(try value()),
                      parsed.isFinite,
                      parsed >= 0,
                      parsed <= 1
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--interaction-student-authority must be in [0, 1]."
                    )
                }
                interactionStudentAuthority = parsed
                index += 1
            case "--interaction-reset-phase-fraction":
                guard let parsed = Float(try value()),
                      parsed.isFinite,
                      parsed >= 0,
                      parsed <= 1
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--interaction-reset-phase-fraction must be in [0, 1]."
                    )
                }
                interactionResetPhaseFraction = parsed
                index += 1
            case "--interaction-reset-phase-probability":
                guard let parsed = Float(try value()),
                      parsed.isFinite,
                      parsed >= 0,
                      parsed <= 1
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--interaction-reset-phase-probability must be in [0, 1]."
                    )
                }
                interactionResetPhaseProbability = parsed
                index += 1
            case "--interaction-reset-maximum-phase":
                guard let parsed = Float(try value()),
                      parsed.isFinite,
                      parsed >= 0,
                      parsed <= 1
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--interaction-reset-maximum-phase must be in [0, 1]."
                    )
                }
                interactionResetMaximumPhase = parsed
                index += 1
            case "--interaction-reset-only":
                interactionResetOnly = true
            case "--materialize-articulated-contact-responses":
                materializeArticulatedContactResponses = true
            case "--minimum-difficulty-band":
                minimumDifficultyBand = try Self.integer(value(), option)
                index += 1
            case "--maximum-difficulty-band":
                maximumDifficultyBand = try Self.integer(value(), option)
                index += 1
            case "--world-pack":
                worldPack = try value()
                index += 1
            case "--task-pack":
                taskPack = try value()
                index += 1
            case "--robot-actuator-pack":
                robotActuatorPack = try value()
                index += 1
            case "--sensor-program-pack":
                sensorProgramPack = try value()
                index += 1
            case "--reality-program-pack":
                realityProgramPack = try value()
                index += 1
            case "--urdf":
                urdf = try value()
                index += 1
            case "--srdf":
                srdf = try value()
                index += 1
            case "--g1-visual-pack-dir":
                g1VisualPackDirectory = try value()
                index += 1
            case "--ball-visual-pack-dir":
                ballVisualPackDirectory = try value()
                index += 1
            case "--visual-environment-pack":
                visualEnvironmentPack = try value()
                index += 1
            case "--visual-observation-config":
                visualObservationConfig = try value()
                index += 1
            case "--birdflow-dove":
                birdFlowDove = true
            case "--birdflow-american-crow":
                birdFlowAmericanCrow = true
            case "--inspect-scene":
                inspectionScene = try value()
                index += 1
            case "--inspect-width":
                inspectionWidth = try Self.integer(value(), option)
                index += 1
            case "--inspect-height":
                inspectionHeight = try Self.integer(value(), option)
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
            case "--task":
                switch try value() {
                case "velocity":
                    unitreeG1Task = .velocity
                case "disturbance-recovery":
                    unitreeG1Task = .disturbanceRecovery
                case "supine-get-up":
                    unitreeG1Task = .supineGetUpDiscovery
                case "developmental-recovery":
                    unitreeG1Task = .developmentalRecovery
                case "adult-locomotion":
                    unitreeG1Task = .adultLocomotion
                case "g1-legs-locomotion":
                    unitreeG1Task = .g1LegsLocomotion
                case "ball-recovery":
                    unitreeG1Task = .ballDisturbanceRecovery
                case "ball-dodge":
                    unitreeG1Task = .ballDodge
                default:
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--task must be velocity, disturbance-recovery, supine-get-up, developmental-recovery, adult-locomotion, g1-legs-locomotion, ball-recovery, or ball-dodge."
                    )
                }
                index += 1
            case "--update-epochs":
                updateEpochs = try Self.integer(value(), option)
                index += 1
            case "--minibatch-size":
                minibatchSize = try Self.integer(value(), option)
                index += 1
            case "--minibatches-per-epoch":
                minibatchesPerEpoch = try Self.integer(value(), option)
                index += 1
            case "--motion-minibatches-per-epoch":
                motionMinibatchesPerEpoch =
                    try Self.integer(value(), option)
                index += 1
            case "--motion-update-epochs":
                motionUpdateEpochs = try Self.integer(value(), option)
                index += 1
            case "--motion-reward-coefficient":
                guard let parsed = Double(try value()),
                      parsed.isFinite,
                      parsed >= 0
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--motion-reward-coefficient must be finite and non-negative."
                    )
                }
                motionRewardCoefficient = parsed
                index += 1
            case "--learning-rate":
                learningRate = try Self.double(value(), option)
                index += 1
            case "--minimum-learning-rate":
                minimumLearningRate = try Self.double(value(), option)
                index += 1
            case "--maximum-learning-rate":
                maximumLearningRate = try Self.double(value(), option)
                index += 1
            case "--fixed-learning-rate":
                fixedLearningRate = true
            case "--override-resumed-learning-rate":
                overrideResumedLearningRate = true
            case "--override-resumed-exploration":
                overrideResumedExploration = true
            case "--clip-ratio":
                clipRatio = try Self.double(value(), option)
                index += 1
            case "--value-coefficient":
                valueCoefficient = try Self.double(value(), option)
                index += 1
            case "--imagination-distillation-coefficient":
                imaginationDistillationCoefficient =
                    try Self.double(value(), option)
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
        if unitreeG1Task == .ballDodge && !stepsWereSpecified {
            // 256 x 20 ms = 5.12 s: long enough for launch, interception,
            // avoidance, and settling instead of truncating the event.
            steps = 256
        }
        if (minimumDifficultyBand == nil) !=
            (maximumDifficultyBand == nil) ||
            (minimumDifficultyBand ?? 0) < 0 ||
            (minimumDifficultyBand ?? 0) >
                (maximumDifficultyBand ?? 0)
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "difficulty-band overrides require an ordered non-negative minimum and maximum."
            )
        }
        guard environments > 0,
              steps > 0,
              updates > 0,
              chunk > 0,
              updateEpochs > 0,
              minibatchSize >= 0,
              minibatchesPerEpoch >= 0,
              motionMinibatchesPerEpoch > 0,
              motionUpdateEpochs > 0,
              motionRewardCoefficient.isFinite,
              motionRewardCoefficient >= 0,
              checkpointInterval >= 0,
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
        if (checkpointDirectory == nil) != (checkpointInterval == 0) {
            throw MetalRoboTaskRolloutError.invalidShape(
                "Checkpoint snapshots require both --checkpoint-directory and a positive --checkpoint-interval."
            )
        }
        if let offset = actorObservationExtensionOffset,
           offset < 0
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--actor-observation-extension-offset must be non-negative."
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
            let targetBatchCount = minibatchesPerEpoch > 0
                ? minibatchesPerEpoch
                : unitreeG1Task == .ballDodge ? 32 : 4
            minibatchSize = max(
                sampleCount / targetBatchCount +
                    (sampleCount % targetBatchCount == 0 ? 0 : 1),
                1
            )
        }
        if motionPack != nil {
            motionMinibatchSize = max(
                sampleCount / motionMinibatchesPerEpoch +
                    (sampleCount % motionMinibatchesPerEpoch == 0
                        ? 0 : 1),
                1
            )
        }
        if learnerState == nil {
            learnerState =
                updatedPolicyPack + ".learner.safetensors"
        }
        if outputLearnerState == nil {
            outputLearnerState = learnerState
        }
        if deploymentPolicyPack == nil {
            deploymentPolicyPack =
                updatedPolicyPack + ".deployment.policypack"
        }
        if incumbentPolicyPack == nil {
            incumbentPolicyPack =
                updatedPolicyPack + ".incumbent.policypack"
        }
        if (g1VisualPackDirectory == nil) !=
            (ballVisualPackDirectory == nil)
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "Visual training requires both --g1-visual-pack-dir and --ball-visual-pack-dir."
            )
        }
        if visualObservationConfig != nil &&
            (g1VisualPackDirectory != nil ||
             ballVisualPackDirectory != nil ||
             visualEnvironmentPack != nil)
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--visual-observation-config cannot be combined with legacy visual preset options."
            )
        }
        if g1VisualPackDirectory != nil &&
            ((unitreeG1Task != .ballDisturbanceRecovery &&
              unitreeG1Task != .ballDodge) ||
             worldPack != nil || urdf != nil)
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "The bundled visual preset requires --task ball-recovery or ball-dodge."
            )
        }
        if unitreeG1Task == .ballDodge &&
            g1VisualPackDirectory == nil &&
            visualObservationConfig == nil
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "Ball-dodge training requires authored G1 and ball Visual Presentation packs."
            )
        }
        if inspectionScene != nil &&
            (inspectionWidth <= 0 || inspectionHeight <= 0) {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--inspect-width and --inspect-height must be positive."
            )
        }
        let ppoValues = [
            learningRate,
            minimumLearningRate,
            maximumLearningRate,
            clipRatio,
            valueCoefficient,
            imaginationDistillationCoefficient,
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
              minimumLearningRate > 0,
              maximumLearningRate >= minimumLearningRate,
              learningRate >= minimumLearningRate,
              learningRate <= maximumLearningRate,
              clipRatio > 0,
              valueCoefficient >= 0,
              imaginationDistillationCoefficient >= 0,
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
        if (interactionPack == nil) != (interactionClip == nil) {
            throw MetalRoboTaskRolloutError.invalidShape(
                "Interaction tracking requires both --interaction-pack and --interaction-clip."
            )
        }
        if let interactionPack,
           interactionPack.isEmpty || interactionClip?.isEmpty != false
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "InteractionPack path and clip identity cannot be empty."
            )
        }
        if interactionStudentAuthority != nil && interactionPack == nil {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--interaction-student-authority requires an InteractionPack."
            )
        }
        if interactionResetPhaseFraction != nil && interactionPack == nil {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--interaction-reset-phase-fraction requires an InteractionPack."
            )
        }
        if (interactionResetPhaseProbability != nil ||
            interactionResetMaximumPhase != nil) && interactionPack == nil
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "Interaction reset curriculum options require an InteractionPack."
            )
        }
        if interactionResetOnly && interactionPack == nil {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--interaction-reset-only requires an InteractionPack."
            )
        }
        if interactionPack != nil &&
            (worldPack != nil ||
             (unitreeG1Task != .velocity &&
              unitreeG1Task != .ballDodge &&
              unitreeG1Task != .supineGetUpDiscovery))
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "InteractionPack training supports imported URDF owner packs or bundled G1 velocity, ball-dodge, and supine-get-up mechanics; it cannot be combined with a WorldPack."
            )
        }
        let importing = worldPack != nil || urdf != nil
        let ownerArtifacts = [
            taskPack, robotActuatorPack, sensorProgramPack, realityProgramPack,
        ]
        if importing != ownerArtifacts.allSatisfy({ $0 != nil }) ||
            (!importing && ownerArtifacts.contains(where: { $0 != nil }))
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "Imported mechanics require --task-pack, --robot-actuator-pack, --sensor-program-pack, and --reality-program-pack."
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

private func makeVisualObservation(
    options: Options
) throws -> MetalRoboTaskVisualObservationConfiguration? {
    if let path = options.visualObservationConfig {
        return try MetalRoboTaskVisualObservationConfiguration
            .loadArtifact(at: URL(fileURLWithPath: path))
    }
    guard let robotDirectory = options.g1VisualPackDirectory,
          let ballDirectory = options.ballVisualPackDirectory
    else {
        return nil
    }
    return try .unitreeG1BallRecovery(
        robotPackDirectory: URL(
            fileURLWithPath: robotDirectory
        ),
        ballPackDirectory: URL(
            fileURLWithPath: ballDirectory
        ),
        environmentPackURL: options.visualEnvironmentPack.map {
            URL(fileURLWithPath: $0)
        }
    )
}

private func makeInspectionVisual(
    options: Options
) throws -> MetalRoboTaskVisualObservationConfiguration? {
    guard let path = options.inspectionScene else {
        return nil
    }
    var inspection = try MetalRoboTaskVisualObservationConfiguration.loadArtifact(
        at: URL(fileURLWithPath: path)
    )
    inspection.width = UInt32(options.inspectionWidth)
    inspection.height = UInt32(options.inspectionHeight)
    inspection.captureWidth = 0
    inspection.captureHeight = 0
    inspection.capturePolicyCamera = false
    return inspection
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
    layout: MetalRoboTaskRolloutLayout,
    actorObservationExtensionMean: Double
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
    var arguments = [
        "-m",
        "metalrobo.mlx_policy_worker",
        "initialize",
        "--actor-observations",
        String(layout.actorObservationCount),
        "--critic-observations",
        String(layout.criticObservationCount),
        "--actions",
        String(layout.actionCount),
        "--world-fingerprint",
        String(layout.worldFingerprint),
        "--task-fingerprint",
        String(layout.taskFingerprint),
        "--observation-fingerprint",
        String(layout.observationFingerprint),
        "--action-fingerprint",
        String(layout.actionFingerprint),
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
    if options.zeroActorOutput {
        arguments.append("--zero-actor-output")
    }
    if let actor = options.initializeActorPolicyPack {
        let extensionMean =
            options.actorObservationExtensionMean ??
            actorObservationExtensionMean
        arguments.append(contentsOf: [
            "--actor-policy-pack", actor,
            "--actor-observation-extension-mean",
            String(extensionMean),
            "--actor-observation-extension-inverse-standard-deviation",
            String(
                options
                    .actorObservationExtensionInverseStandardDeviation
            ),
        ])
        if options.initializeActorFreshCritic {
            arguments.append("--actor-fresh-critic")
        }
    }
    if let offset = options.actorObservationExtensionOffset {
        arguments.append(contentsOf: [
            "--actor-observation-extension-offset",
            String(offset),
        ])
    }
    process.arguments = arguments
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
    private(set) var motionFeatureCount: Int
    private(set) var policyPackPath: String
    private(set) var deploymentPolicyPackPath: String
    private(set) var stateRestored: Bool
    private var closed = false

    init(
        options: Options,
        layout: MetalRoboTaskRolloutLayout
    ) throws {
        guard let executable = options.mlxPython,
              let policyPack = options.policyPack,
              let outputPolicyPack = options.updatedPolicyPack,
              let deploymentPolicyPack =
                  options.deploymentPolicyPack,
              let incumbentPolicyPack =
                  options.incumbentPolicyPack,
              let learnerState = options.learnerState,
              let outputLearnerState = options.outputLearnerState
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
        var arguments = [
            "-m",
            "metalrobo.mlx_policy_worker",
            "serve",
            "--policy-pack",
            policyPack,
            "--output-policy-pack",
            outputPolicyPack,
            "--deployment-policy-pack",
            deploymentPolicyPack,
            "--incumbent-policy-pack",
            incumbentPolicyPack,
            "--learner-state",
            outputLearnerState,
            "--restore-learner-state",
            learnerState,
            "--native-library",
            options.nativeLibrary,
            "--update-epochs",
            String(options.updateEpochs),
            "--minibatch-size",
            String(options.minibatchSize),
            "--learning-rate",
            String(options.learningRate),
            "--minimum-learning-rate",
            String(options.minimumLearningRate),
            "--maximum-learning-rate",
            String(options.maximumLearningRate),
            "--clip-ratio",
            String(options.clipRatio),
            "--value-coefficient",
            String(options.valueCoefficient),
            "--imagination-distillation-coefficient",
            String(options.imaginationDistillationCoefficient),
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
            "--world-fingerprint",
            String(layout.worldFingerprint),
            "--task-fingerprint",
            String(layout.taskFingerprint),
            "--observation-fingerprint",
            String(layout.observationFingerprint),
            "--action-fingerprint",
            String(layout.actionFingerprint),
        ]
        if options.fixedLearningRate {
            arguments.append("--fixed-learning-rate")
        }
        if options.overrideResumedLearningRate {
            arguments.append("--override-resumed-learning-rate")
        }
        if options.overrideResumedExploration {
            arguments.append("--override-resumed-exploration")
        }
        if let offset = options.actorObservationExtensionOffset {
            arguments.append(contentsOf: [
                "--actor-observation-extension-offset",
                String(offset),
            ])
        }
        if let motionPack = options.motionPack {
            arguments.append(contentsOf: [
                "--motion-pack", motionPack,
                "--motion-minibatch-size",
                String(options.motionMinibatchSize),
                "--motion-update-epochs",
                String(options.motionUpdateEpochs),
                "--motion-activation",
                options.motionActivation,
                "--motion-reward-coefficient",
                String(options.motionRewardCoefficient),
            ])
        }
        process.arguments = arguments
        process.environment = mlxEnvironment(options: options)
        process.standardInput = input
        process.standardOutput = output
        process.standardError = FileHandle.standardError
        try process.run()

        revision = 0
        actorObservationCount = 0
        criticObservationCount = 0
        actionCount = 0
        motionFeatureCount = 0
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
              let motionValue =
                  ready["motion_feature_count"] as? NSNumber,
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
        motionFeatureCount = motionValue.intValue
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

    func update(rolloutPack: String) throws -> [String: Any] {
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
    let visualSensor = try makeVisualObservation(options: options)
    let inspectionVisual = try makeInspectionVisual(options: options)
    let dynamicSpheres: [MetalRoboDynamicSphere] =
        options.unitreeG1Task == .ballDodge
        ? MetalRoboDynamicSphere.g1BallDodgeDefaults
        : options.unitreeG1Task == .ballDisturbanceRecovery
            ? MetalRoboDynamicSphere.g1BallRecoveryDefaults
            : []
    let configuration = MetalRoboTaskRolloutConfiguration(
        environmentCount: UInt32(options.environments),
        surface: options.surface,
        seed: options.seed,
        dynamicSpheres: dynamicSpheres,
        materializeArticulatedContactResponses:
            options.materializeArticulatedContactResponses,
        difficultyBandRange:
            options.minimumDifficultyBand.map {
                UInt32($0)...UInt32(options.maximumDifficultyBand!)
            },
        interactionReferenceMode: options.interactionResetOnly
            ? .resetOnly
            : .taskDefault,
        interactionStudentAuthority:
            options.interactionStudentAuthority,
        interactionResetPhaseFraction:
            options.interactionResetPhaseFraction,
        interactionResetPhaseProbability:
            options.interactionResetPhaseProbability,
        interactionResetMaximumPhase:
            options.interactionResetMaximumPhase,
        unitreeG1Task: options.unitreeG1Task
    )
    if options.birdFlowDove && options.birdFlowAmericanCrow {
        throw MetalRoboTaskRolloutError.invalidShape(
            "--birdflow-dove and --birdflow-american-crow are mutually exclusive."
        )
    }
    if options.birdFlowDove {
        return (
            try MetalRoboTaskRolloutContext(
                manifest: MetalRoboRunManifest(
                    source: .birdFlowDove,
                    sensorsAndPhysics: configuration,
                    visualSensor: visualSensor
                ),
                metallibPath: options.metallib
            ),
            "birdflow_deetjen_dove_hybrid"
        )
    }
    if options.birdFlowAmericanCrow {
        return (
            try MetalRoboTaskRolloutContext(
                manifest: MetalRoboRunManifest(
                    source: .birdFlowAmericanCrow,
                    sensorsAndPhysics: configuration,
                    visualSensor: visualSensor
                ),
                metallibPath: options.metallib
            ),
            "birdflow_american_crow_estimated_hybrid"
        )
    }
    if let interactionPack = options.interactionPack,
       let interactionClip = options.interactionClip,
       options.urdf == nil
    {
        return (
            try MetalRoboTaskRolloutContext(
                manifest: MetalRoboRunManifest(
                    source: .unitreeG1,
                    sensorsAndPhysics: configuration,
                    visualSensor: visualSensor,
                    inspectionVisual: inspectionVisual,
                    teacher: MetalRoboTeacherSource(
                        pack: URL(fileURLWithPath: interactionPack),
                        clipID: interactionClip
                    )
                ),
                metallibPath: options.metallib
            ),
            "bundled_g1_interaction"
        )
    }
    if let worldPack = options.worldPack,
       let taskPack = options.taskPack,
       let actuatorPack = options.robotActuatorPack,
       let sensorPack = options.sensorProgramPack,
       let realityPack = options.realityProgramPack
    {
        return (
            try MetalRoboTaskRolloutContext(
                manifest: MetalRoboRunManifest(
                    source: .worldPack(
                        world: URL(fileURLWithPath: worldPack),
                        taskPack: URL(fileURLWithPath: taskPack),
                        actuatorPack: URL(fileURLWithPath: actuatorPack),
                        sensorPack: URL(fileURLWithPath: sensorPack),
                        realityPack: URL(fileURLWithPath: realityPack)
                    ),
                    sensorsAndPhysics: configuration,
                    visualSensor: visualSensor,
                    inspectionVisual: inspectionVisual
                ),
                metallibPath: options.metallib
            ),
            "world_pack"
        )
    }
    if let urdf = options.urdf,
       let taskPack = options.taskPack,
       let actuatorPack = options.robotActuatorPack,
       let sensorPack = options.sensorProgramPack,
       let realityPack = options.realityProgramPack
    {
        return (
            try MetalRoboTaskRolloutContext(
                manifest: MetalRoboRunManifest(
                    source: .importedURDF(
                        urdf: URL(fileURLWithPath: urdf),
                        srdf: options.srdf.map {
                            URL(fileURLWithPath: $0)
                        },
                        taskPack: URL(fileURLWithPath: taskPack),
                        actuatorPack: URL(fileURLWithPath: actuatorPack),
                        sensorPack: URL(fileURLWithPath: sensorPack),
                        realityPack: URL(fileURLWithPath: realityPack)
                    ),
                    sensorsAndPhysics: configuration,
                    visualSensor: visualSensor,
                    inspectionVisual: inspectionVisual,
                    teacher: options.interactionPack.flatMap { pack in
                        options.interactionClip.map {
                            MetalRoboTeacherSource(
                                pack: URL(fileURLWithPath: pack),
                                clipID: $0
                            )
                        }
                    }
                ),
                metallibPath: options.metallib
            ),
            "urdf"
        )
    }
    return (
        try MetalRoboTaskRolloutContext(
            manifest: MetalRoboRunManifest(
                source: .unitreeG1,
                sensorsAndPhysics: configuration,
                visualSensor: visualSensor,
                inspectionVisual: inspectionVisual
            ),
            metallibPath: options.metallib
        ),
        "bundled_g1"
    )
}

private func publishInspectionFrame(
    from context: MetalRoboTaskRolloutContext,
    to inspector: MetalRoboRunInspectorBridge?,
    loadPolicy: ((URL) throws -> UInt64)? = nil
) throws {
    guard let inspector else {
        return
    }
    if let enabled = inspector.takePendingNativeInspectionEnabled() {
        try context.setInspectionEnabled(enabled)
    }
    func install(_ url: URL) {
        guard let loadPolicy else {
            inspector.reportPolicyStatus("policy reload unavailable")
            return
        }
        do {
            let revision = try loadPolicy(url)
            inspector.reportPolicyInstalled(url, revision: revision)
        } catch {
            // A rejected replacement cannot change the compiled policy that
            // remains active for the current training submission.
            inspector.reportPolicyRejected()
        }
    }
    let latestRequested = inspector.takeLatestPolicyReloadRequest()
    if let selected = inspector.takePolicySelection() {
        install(selected)
    } else if latestRequested {
        guard let selected = inspector.selectedPolicy else {
            inspector.reportPolicyStatus("select a policy first")
            return
        }
        install(selected)
    }
    guard inspector.acceptsFrames else {
        if inspector.isClosed {
            throw MetalRoboRunInspectorError.closed
        }
        // Pause and window occlusion gate only presentation encoding at this
        // scheduler-owned boundary; training keeps its original cadence.
        return
    }
    guard let frame = try context.acquireInspectionFrame() else {
        return
    }
    inspector.publish(frame: frame, context: context)
}

@main
private enum TaskTrainMain {
    private static func run(
        options: Options,
        inspector: MetalRoboRunInspectorBridge?
    ) throws {
            let (context, worldSource) =
                try makeContext(options: options)
            try initializePolicyIfRequested(
                options: options,
                layout: context.layout,
                // Masked-depth device observations encode far/empty as 1.
                // Centering an appended visual suffix there preserves the
                // source actor on ball-free standing anchors.
                actorObservationExtensionMean:
                    context.visualSceneFingerprint == 0 ? 0.0 : 1.0
            )
            guard let updatedPolicyPack =
                      options.updatedPolicyPack,
                  let deploymentPolicyPack =
                      options.deploymentPolicyPack,
                  let rolloutPack = options.rolloutPack,
                  let learnerState = options.outputLearnerState
            else {
                throw MetalRoboTaskRolloutError.invalidShape(
                    "Training artifact paths are incomplete."
                )
            }
            let learner = try MLXLearnerWorker(
                options: options,
                layout: context.layout
            )
            try context.loadPolicy(
                at: URL(fileURLWithPath: learner.policyPackPath)
            )
            if let basePolicyPack = options.basePolicyPack {
                try context.loadBasePolicy(
                    at: URL(fileURLWithPath: basePolicyPack)
                )
            }
            let layout = context.layout
            guard learner.revision != 0,
                  learner.actorObservationCount ==
                      layout.actorObservationCount,
                  learner.criticObservationCount ==
                      layout.criticObservationCount,
                  learner.actionCount == layout.actionCount,
                  (
                      learner.motionFeatureCount == 0 ||
                      learner.motionFeatureCount ==
                          layout.motionFeatureCount
                  )
            else {
                throw MetalRoboTaskRolloutError.invalidShape(
                    "PolicyPack dimensions do not match the compiled task: "
                    + "actor \(learner.actorObservationCount)/"
                    + "\(layout.actorObservationCount), critic "
                    + "\(learner.criticObservationCount)/"
                    + "\(layout.criticObservationCount), actions "
                    + "\(learner.actionCount)/\(layout.actionCount), motion "
                    + "\(learner.motionFeatureCount)/"
                    + "\(layout.motionFeatureCount)."
                )
            }
            let initialRevision = learner.revision
            var installedRevision = learner.revision
            let warmup = try context.advanceWithPolicy(
                controlStepCount: 1,
                policyRevision: installedRevision
            )
            try publishInspectionFrame(
                from: context,
                to: inspector,
                loadPolicy: { url in
                    try context.loadPolicy(
                        at: url
                    )
                    installedRevision = context.installedPolicyRevision
                    return installedRevision
                }
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
            let warmupEvidenceTelemetry =
                try context.evidenceTelemetry()
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
            var lastEvidenceTelemetry =
                warmupEvidenceTelemetry
            let samplesPerUpdate =
                options.environments * options.steps
            let (trainingSamples, trainingSamplesOverflow) =
                samplesPerUpdate.multipliedReportingOverflow(
                    by: options.updates
                )
            guard !trainingSamplesOverflow else {
                throw MetalRoboTaskRolloutError.invalidShape(
                    "Total training sample count overflows Int."
                )
            }
            var actorObservations: [Float] = []
            var criticObservations: [Float] = []
            var motionFeatures: [Float] = []
            var teacherActions: [Float] = []
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
            motionFeatures.reserveCapacity(
                samplesPerUpdate * layout.motionFeatureCount
            )
            teacherActions.reserveCapacity(
                samplesPerUpdate * layout.actionCount
            )
            latents.reserveCapacity(
                samplesPerUpdate * layout.actionCount
            )
            logProbabilities.reserveCapacity(samplesPerUpdate)
            values.reserveCapacity(samplesPerUpdate)
            transitions.reserveCapacity(samplesPerUpdate)

            for updateIndex in 0..<options.updates {
                try autoreleasepool {
                    // Native Metal submissions and Foundation file bridges
                    // create Objective-C temporaries per update. Bound their
                    // lifetime to one rollout/learner transaction so a long
                    // 4K-environment run does not trade throughput for an
                    // ever-growing unified-memory footprint.
                    actorObservations.removeAll(
                        keepingCapacity: true
                    )
                criticObservations.removeAll(
                    keepingCapacity: true
                )
                motionFeatures.removeAll(keepingCapacity: true)
                teacherActions.removeAll(keepingCapacity: true)
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
                    try publishInspectionFrame(
                        from: context,
                        to: inspector,
                        loadPolicy: { url in
                            try context.loadPolicy(
                                at: url
                            )
                            installedRevision = context.installedPolicyRevision
                            return installedRevision
                        }
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
                        motionFeatures: &motionFeatures,
                        teacherActions: &teacherActions,
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
                    motionFeatureCount:
                        layout.motionFeatureCount,
                    actorObservations: actorObservations,
                    criticObservations: criticObservations,
                    motionFeatures: motionFeatures,
                    teacherActions: teacherActions,
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
                lastEvidenceTelemetry =
                    try context.evidenceTelemetry()
                lastLearning = try learner.update(
                    rolloutPack: rolloutPack
                )
                try context.loadPolicy(
                    at: URL(fileURLWithPath: updatedPolicyPack)
                )
                installedRevision = learner.revision
                if let checkpointDirectory =
                        options.checkpointDirectory,
                   (updateIndex + 1) % options.checkpointInterval == 0
                {
                    let directory = URL(
                        fileURLWithPath: checkpointDirectory,
                        isDirectory: true
                    )
                    try FileManager.default.createDirectory(
                        at: directory,
                        withIntermediateDirectories: true
                    )
                    let revision = String(
                        format: "%08llu",
                        learner.revision
                    )
                    let artifacts = [
                        (
                            updatedPolicyPack,
                            directory.appendingPathComponent(
                                "revision-\(revision).policypack"
                            )
                        ),
                        (
                            deploymentPolicyPack,
                            directory.appendingPathComponent(
                                "revision-\(revision).candidate.policypack"
                            )
                        ),
                        (
                            learnerState,
                            directory.appendingPathComponent(
                                "revision-\(revision).safetensors"
                            )
                        ),
                    ]
                    for (source, destination) in artifacts {
                        try FileManager.default.copyItem(
                            at: URL(fileURLWithPath: source),
                            to: destination
                        )
                    }
                }
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
                                "contact_rate=\(lastEvidenceTelemetry.lastContactRate) " +
                                "miss_rate=\(lastEvidenceTelemetry.lastCleanMissRate) " +
                                "balance_rate=\(lastEvidenceTelemetry.lastBalanceFailureRate) " +
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
            }
            try learner.close()

            let end = clock.now
            let elapsed = start.duration(to: end)
            let seconds =
                Double(elapsed.components.seconds) +
                Double(elapsed.components.attoseconds) / 1e18
            let finalLayout = context.layout
            let output: [String: Any] = [
                "operation": "swift_native_policy_training",
                "scheduler": "swift",
                "physics": "metal",
                "learner": "mlx",
                "world_source": worldSource,
                "action_carrier": options.birdFlowAmericanCrow
                    ? "stage1_crow_gait_plus_bounded_policy_residual_0.25_band_1;stage2_live_altitude_vertical_rate_and_airspeed_trim_plus_phase_calibrated_pronation_target_amplitude_0.20_phase_2.62_plus_bounded_residual_0.25_wing_sweep_pronation_and_leg_residual_0.25_tail_residual_0.10_band_2"
                    : "none",
                "device": context.deviceName,
                "visual_observation":
                    context.visualSceneFingerprint != 0,
                "visual_scene_fingerprint":
                    context.visualSceneFingerprint,
                "visual_observation_config":
                    options.visualObservationConfig ?? "",
                "environments": options.environments,
                "steps_per_update": options.steps,
                "maximum_episode_steps":
                    finalLayout.maximumEpisodeSteps,
                "updates": options.updates,
                "control_steps_per_submission": options.chunk,
                "minibatch_size": options.minibatchSize,
                "minibatches_per_epoch":
                    (options.environments * options.steps) /
                        options.minibatchSize +
                    ((options.environments * options.steps) %
                        options.minibatchSize == 0 ? 0 : 1),
                "motion_minibatch_size":
                    options.motionMinibatchSize,
                "checkpoint_directory":
                    options.checkpointDirectory ?? "",
                "checkpoint_interval": options.checkpointInterval,
                "samples_per_update": samplesPerUpdate,
                "training_samples": trainingSamples,
                "initial_policy_revision": initialRevision,
                "final_policy_revision": installedRevision,
                "final_physical_evidence": [
                    "control_steps":
                        lastEvidenceTelemetry.controlSteps,
                    "evidence_windows":
                        lastEvidenceTelemetry.evidenceWindows,
                    "pending_completed_episode_count":
                        lastEvidenceTelemetry
                            .pendingCompletedEpisodeCount,
                    "pending_timeout_episode_count":
                        lastEvidenceTelemetry
                            .pendingTimeoutEpisodeCount,
                    "last_completed_episode_count":
                        lastEvidenceTelemetry
                            .lastCompletedEpisodeCount,
                    "last_contact_rate":
                        lastEvidenceTelemetry.lastContactRate,
                    "last_clean_miss_rate":
                        lastEvidenceTelemetry.lastCleanMissRate,
                    "last_balance_failure_rate":
                        lastEvidenceTelemetry
                            .lastBalanceFailureRate,
                    "last_mean_tracking_per_million":
                        lastEvidenceTelemetry
                            .lastMeanTrackingPerMillion,
                ],
                "learner_state_restored":
                    learner.stateRestored,
                "resumed_learning_rate_overridden":
                    options.overrideResumedLearningRate &&
                    learner.stateRestored,
                "resumed_exploration_overridden":
                    options.overrideResumedExploration &&
                    learner.stateRestored,
                "native_submission_count": NSNumber(
                    value:
                        finalLayout.submissionCount -
                        baseline.submissionCount
                ),
                "end_to_end_environment_steps_per_second":
                    Double(trainingSamples) / seconds,
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
                "candidate_deployment_policy_pack":
                    deploymentPolicyPack,
                "deployment_selection_performed": false,
                "incumbent_policy_pack":
                    options.incumbentPolicyPack!,
                "learner_state": learnerState,
                "last_learning_update": lastLearning,
            ]
            let data = try JSONSerialization.data(
                withJSONObject: output,
                options: [.prettyPrinted, .sortedKeys]
            )
            print(String(decoding: data, as: UTF8.self))
    }

    static func main() {
        do {
            let options = try Options(arguments: CommandLine.arguments)
            guard options.inspectionScene != nil else {
                try run(options: options, inspector: nil)
                return
            }
            let policyChoices = metalRoboInspectorPolicyCatalog()
            let initialPolicyURL = (options.updatedPolicyPack ??
                options.policyPack).map {
                    URL(fileURLWithPath: $0)
                }
            let inspector = try MetalRoboRunInspectorBridge.launch(
                canReloadLatestPolicy:
                    initialPolicyURL != nil || !policyChoices.isEmpty,
                policyChoices: policyChoices,
                initialPolicyURL: initialPolicyURL
            )
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    try run(options: options, inspector: inspector)
                } catch {
                    if case MetalRoboRunInspectorError.closed = error {
                        // Closing the window ends its preview run cleanly.
                    } else {
                        FileHandle.standardError.write(
                            Data("task_train failed: \(error)\n".utf8)
                        )
                    }
                }
                DispatchQueue.main.async {
                    NSApplication.shared.terminate(nil)
                }
            }
            NSApplication.shared.run()
        } catch {
            FileHandle.standardError.write(
                Data("task_train failed: \(error)\n".utf8)
            )
            Darwin.exit(1)
        }
    }
}
