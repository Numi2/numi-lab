import Foundation
import Darwin

private struct Options {
    var environments = 32
    var steps = 24
    var updates = 100
    var chunk = 8
    var surface = MetalRoboBuiltinSurface.terrain
    var g1ActuatorPreset =
        MetalRoboG1ActuatorPreset.unitreeRLLab4960b84
    var numiIterationPolicy =
        MetalRoboNumiIterationPolicy.fixedBudget
    var seed: UInt64 = 20_260_731
    var metallib = "build/shaders/MetalRobo.metallib"
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
    var verbose = false

    init(arguments: [String]) throws {
        var index = 1
        while index < arguments.count {
            let option = arguments[index]
            func value() throws -> String {
                guard index + 1 < arguments.count else {
                    throw MetalRoboSimulationError.invalidShape(
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
                    throw MetalRoboSimulationError.invalidShape(
                        "--seed requires an unsigned integer."
                    )
                }
                seed = parsed
                index += 1
            case "--metallib":
                metallib = try value()
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
                    throw MetalRoboSimulationError.invalidShape(
                        "--scene must be ground or terrain."
                    )
                }
                index += 1
            case "--g1-actuator-preset":
                switch try value() {
                case "urdf", "unitree_urdf_rev_1_0":
                    g1ActuatorPreset = .unitreeURDFRev10
                case "mjcf", "unitree_mjcf_rev_1_0":
                    g1ActuatorPreset = .unitreeMJCFRev10
                case "rl_lab", "unitree_rl_lab_4960b84":
                    g1ActuatorPreset = .unitreeRLLab4960b84
                default:
                    throw MetalRoboSimulationError.invalidShape(
                        "--g1-actuator-preset must be urdf, mjcf, or rl_lab."
                    )
                }
                index += 1
            case "--numi-iteration-policy":
                switch try value() {
                case "fixed", "fixed_budget":
                    numiIterationPolicy = .fixedBudget
                case "residual", "residual_converged":
                    numiIterationPolicy = .residualConverged
                default:
                    throw MetalRoboSimulationError.invalidShape(
                        "--numi-iteration-policy must be fixed or residual."
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
            case "--verbose":
                verbose = true
            default:
                throw MetalRoboSimulationError.invalidShape(
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
              let policyPack,
              !policyPack.isEmpty,
              let updatedPolicyPack,
              !updatedPolicyPack.isEmpty,
              let rolloutPack,
              !rolloutPack.isEmpty
        else {
            throw MetalRoboSimulationError.invalidShape(
                "Positive rollout sizes and all native learning artifact paths are required."
            )
        }
        let (sampleCount, sampleOverflow) =
            environments.multipliedReportingOverflow(
                by: steps
            )
        guard !sampleOverflow else {
            throw MetalRoboSimulationError.invalidShape(
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
        ]
        guard ppoValues.allSatisfy(\.isFinite),
              learningRate > 0,
              clipRatio > 0,
              valueCoefficient >= 0,
              entropyCoefficient >= 0,
              initialLogStandardDeviation >= -5,
              initialLogStandardDeviation <= 2,
              maximumGradientNorm > 0,
              targetKL > 0,
              discount >= 0,
              discount <= 1,
              gaeLambda >= 0,
              gaeLambda <= 1
        else {
            throw MetalRoboSimulationError.invalidShape(
                "PPO scalar options are invalid."
            )
        }
        if worldPack != nil && urdf != nil {
            throw MetalRoboSimulationError.invalidShape(
                "--world-pack and --urdf are mutually exclusive."
            )
        }
        if (worldPack != nil || urdf != nil) != (taskPack != nil) {
            throw MetalRoboSimulationError.invalidShape(
                "Imported mechanics require exactly one --task-pack."
            )
        }
        if srdf != nil && urdf == nil {
            throw MetalRoboSimulationError.invalidShape(
                "--srdf requires --urdf."
            )
        }
    }

    private static func integer(
        _ value: String,
        _ option: String
    ) throws -> Int {
        guard let parsed = Int(value) else {
            throw MetalRoboSimulationError.invalidShape(
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
            throw MetalRoboSimulationError.invalidShape(
                "\(option) requires a number."
            )
        }
        return parsed
    }
}

private func makeContext(
    options: Options
) throws -> (MetalSimulationSession, String) {
    let configuration = MetalRoboSimulationConfiguration(
        environmentCount: UInt32(options.environments),
        numiSolver: .init(
            iterationPolicy: options.numiIterationPolicy
        ),
        seed: options.seed
    )
    if let worldPack = options.worldPack,
       let taskPack = options.taskPack
    {
        return (
            try MetalSimulationSession(
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
            try MetalSimulationSession(
                importedURDF: URL(fileURLWithPath: urdf),
                srdf: options.srdf.map {
                    URL(fileURLWithPath: $0)
                },
                taskPack: URL(fileURLWithPath: taskPack),
                configuration: configuration,
                surface: options.surface,
                metallibPath: options.metallib
            ),
            "urdf"
        )
    }
    return (
        try MetalSimulationSession(
            unitreeG1: configuration,
            actuatorPreset: options.g1ActuatorPreset,
            surface: options.surface,
            metallibPath: options.metallib
        ),
        "bundled_g1"
    )
}

@main
private enum TrainMain {
    static func main() {
        do {
            let options = try Options(
                arguments: CommandLine.arguments
            )
            let (context, worldSource) =
                try makeContext(options: options)
            guard let policyPack = options.policyPack,
                  let updatedPolicyPack =
                      options.updatedPolicyPack,
                  let deploymentPolicyPack =
                      options.deploymentPolicyPack,
                  let rolloutPack = options.rolloutPack,
                  let learnerState = options.learnerState
            else {
                throw MetalRoboSimulationError.invalidShape(
                    "Training artifact paths are incomplete."
                )
            }
            let policyURL = URL(fileURLWithPath: policyPack)
            let updatedPolicyURL = URL(
                fileURLWithPath: updatedPolicyPack
            )
            let deploymentPolicyURL = URL(
                fileURLWithPath: deploymentPolicyPack
            )
            let learnerStateURL = URL(
                fileURLWithPath: learnerState
            )
            let initializeFresh = options.initializePolicyID != nil
            if initializeFresh {
                guard !FileManager.default.fileExists(
                    atPath: policyURL.path
                ),
                !FileManager.default.fileExists(
                    atPath: learnerStateURL.path
                ) else {
                    throw MetalRoboSimulationError.invalidShape(
                        "--initialize-policy refuses to overwrite an existing policy or learner state."
                    )
                }
            } else {
                guard FileManager.default.fileExists(
                    atPath: learnerStateURL.path
                ) else {
                    throw MetalRoboSimulationError.invalidShape(
                        "Native training requires --initialize-policy for a fresh run or an existing --learner-state to resume."
                    )
                }
            }

            let layout = context.layout
            let learner = try MetalRoboMLXPPOTrainer(
                actorObservationCount:
                    layout.actorObservationCount,
                criticObservationCount:
                    layout.criticObservationCount,
                actionCount: layout.actionCount,
                policyID: options.initializePolicyID,
                trainingPolicyURL: updatedPolicyURL,
                deploymentPolicyURL: deploymentPolicyURL,
                learnerStateURL: learnerStateURL,
                taskFingerprint: context.taskFingerprint,
                configuration: MetalRoboMLXPPOConfiguration(
                    updateEpochs: options.updateEpochs,
                    minibatchSize: options.minibatchSize,
                    learningRate: Float(options.learningRate),
                    clipRatio: Float(options.clipRatio),
                    valueCoefficient:
                        Float(options.valueCoefficient),
                    entropyCoefficient:
                        Float(options.entropyCoefficient),
                    maximumGradientNorm:
                        Float(options.maximumGradientNorm),
                    targetKL: Float(options.targetKL),
                    discount: Float(options.discount),
                    gaeLambda: Float(options.gaeLambda),
                    initialLogStandardDeviation:
                        Float(options.initialLogStandardDeviation),
                    seed: UInt64(options.learnerSeed)
                ),
                resumeIfPresent: !initializeFresh
            )
            let initialPolicy = try learner.initialPolicyPack()
            if initializeFresh {
                try FileManager.default.createDirectory(
                    at: policyURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                try initialPolicy.write(to: policyURL)
            }
            try learner.publishCurrentPolicies()
            try context.setCurriculumLevel(
                learner.taskCurriculumLevel
            )
            try context.setPolicy(initialPolicy)
            let initialRevision = learner.revision
            let policyTopologyFingerprint =
                context.policyTopologyFingerprint
            let initialPolicyFingerprint =
                context.policyFingerprint
            guard policyTopologyFingerprint != 0,
                  initialPolicyFingerprint != 0,
                  context.policyRevision == initialRevision
            else {
                throw MetalRoboSimulationError.native(
                    "Native policy contract was not installed atomically."
                )
            }
            var stalePolicy = initialPolicy
            stalePolicy.layers[0].bias[0] += 1.0e-5
            var rejectedStaleRevision = false
            do {
                try context.setPolicy(stalePolicy)
            } catch {
                rejectedStaleRevision = true
            }
            var incompatiblePolicy = initialPolicy
            incompatiblePolicy.revision += 1
            incompatiblePolicy.layers[0].activation = .relu
            var rejectedTopologyChange = false
            do {
                try context.setPolicy(incompatiblePolicy)
            } catch {
                rejectedTopologyChange = true
            }
            guard rejectedStaleRevision,
                  rejectedTopologyChange,
                  context.policyFingerprint ==
                      initialPolicyFingerprint,
                  context.policyRevision == initialRevision
            else {
                throw MetalRoboSimulationError.native(
                    "Native policy revision/topology rejection was not transactional."
                )
            }
            let initialCurriculumLevel =
                learner.taskCurriculumLevel
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
                throw MetalRoboSimulationError.native(
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
            let rolloutRing = try MetalRoboRolloutBufferRing(
                layout: layout,
                controlStepCapacity: options.steps,
                slotCount: 3
            )

            for updateIndex in 0..<options.updates {
                let rollout = try rolloutRing.acquire(
                    policyRevision: installedRevision
                )
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
                        throw MetalRoboSimulationError.native(
                            "Native rollout returned a GPU failure."
                        )
                    }
                    try context.appendCurrentPolicyRollout(
                        controlStepCount: stepCount,
                        to: rollout,
                        includeBootstrapValues:
                            completed + stepCount == options.steps
                    )
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
                try context.sealPolicyRollout(rollout)
                try context.writePolicyRolloutPack(
                    rollout,
                    id: "swift_native_training_rollout",
                    to: URL(fileURLWithPath: rolloutPack)
                )
                let rolloutCurriculumLevel = try rollout.transition(
                    at: rollout.sampleCount - 1
                ).curriculumLevel
                guard
                      rolloutCurriculumLevel >=
                          learner.taskCurriculumLevel
                else {
                    throw MetalRoboSimulationError.native(
                        "Native rollout did not publish a monotonic task curriculum."
                    )
                }
                lastLearning = try learner.update(
                    rollout: rollout,
                    curriculumLevel: rolloutCurriculumLevel
                )
                guard learner.taskCurriculumLevel ==
                          rolloutCurriculumLevel
                else {
                    throw MetalRoboSimulationError.native(
                        "Learner checkpoint disagrees with the native rollout curriculum."
                    )
                }
                try context.setPolicy(
                    learner.initialPolicyPack()
                )
                installedRevision = learner.revision
                guard context.policyRevision == installedRevision,
                      context.policyTopologyFingerprint ==
                          policyTopologyFingerprint,
                      context.policyFingerprint != 0
                else {
                    throw MetalRoboSimulationError.native(
                        "Native policy bank violated its immutable topology or revision contract."
                    )
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
                                "curriculum=\(learner.taskCurriculumLevel) " +
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
                "learner": "mlx_swift",
                "world_source": worldSource,
                "device": context.deviceName,
                "environments": options.environments,
                "steps_per_update": options.steps,
                "updates": options.updates,
                "control_steps_per_submission": options.chunk,
                "training_samples": sampleCount,
                "initial_policy_revision": initialRevision,
                "final_policy_revision": installedRevision,
                "initial_policy_fingerprint": String(
                    initialPolicyFingerprint
                ),
                "final_policy_fingerprint": String(
                    context.policyFingerprint
                ),
                "policy_topology_fingerprint": String(
                    policyTopologyFingerprint
                ),
                "stale_policy_revision_rejected":
                    rejectedStaleRevision,
                "policy_topology_change_rejected":
                    rejectedTopologyChange,
                "initial_task_curriculum_level":
                    initialCurriculumLevel,
                "final_task_curriculum_level":
                    learner.taskCurriculumLevel,
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
                "rollout_ring_bytes": rolloutRing.retainedBytes,
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
                Data("training failed: \(error)\n".utf8)
            )
            Darwin.exit(1)
        }
    }
}
