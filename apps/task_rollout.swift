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

private func fingerprintWord(
    _ value: UInt64,
    byteCount: Int,
    into hash: inout UInt64
) {
    for byte in 0..<byteCount {
        hash ^= (value >> UInt64(8 * byte)) & 0xff
        hash &*= 1_099_511_628_211
    }
}

private func fingerprint(
    _ batch: MetalRoboPolicyRolloutBatch,
    into hash: inout UInt64
) {
    let floatTables = [
        batch.actorObservations,
        batch.criticObservations,
        batch.latents,
        batch.logProbabilities,
        batch.values,
    ]
    for table in floatTables {
        for value in table {
            fingerprintWord(
                UInt64(value.bitPattern),
                byteCount: 4,
                into: &hash
            )
        }
    }
    for transition in batch.transitions {
        for value in [
            transition.reward,
            transition.trackingScore,
            transition.rootHeight,
            transition.tilt,
            transition.taskReward,
            transition.baseReward,
            transition.jointVelocityReward,
            transition.jointAccelerationReward,
            transition.controlReward,
            transition.postureReward,
            transition.energyReward,
            transition.contactReward,
        ] {
            fingerprintWord(
                UInt64(value.bitPattern),
                byteCount: 4,
                into: &hash
            )
        }
        fingerprintWord(
            transition.done ? 1 : 0,
            byteCount: 4,
            into: &hash
        )
        fingerprintWord(
            transition.timeout ? 1 : 0,
            byteCount: 4,
            into: &hash
        )
        fingerprintWord(
            transition.physicsError ? 1 : 0,
            byteCount: 4,
            into: &hash
        )
        fingerprintWord(
            UInt64(transition.terminationReason),
            byteCount: 4,
            into: &hash
        )
        fingerprintWord(
            transition.policyRevision,
            byteCount: 8,
            into: &hash
        )
        fingerprintWord(
            UInt64(transition.timeoutBootstrapValue.bitPattern),
            byteCount: 4,
            into: &hash
        )
        fingerprintWord(
            UInt64(transition.episodeTrackingScore.bitPattern),
            byteCount: 4,
            into: &hash
        )
        fingerprintWord(
            UInt64(transition.curriculumLevel),
            byteCount: 4,
            into: &hash
        )
        fingerprintWord(
            UInt64(transition.terrainLevel),
            byteCount: 4,
            into: &hash
        )
    }
}

private struct Options {
    var environments = 32
    var steps = 48
    var repeats = 20
    var chunk = 8
    var physicsSubsteps = 4
    var velocityIterations = 4
    var finalVelocityIterations = 2
    var surface = MetalRoboLocomotionSurface.terrain
    var seed: UInt64 = 20_260_731
    var curriculumLevel: UInt32 = 0
    var metallib = "build/shaders/MetalRobo.metallib"
    var verbose = false
    var nativePolicy = false
    var zeroActions = false
    var scheduledResets = true
    var policyPack: String?
    var rolloutPack: String?
    var worldPack: String?
    var taskPack: String?
    var urdf: String?
    var srdf: String?
    var dynamicSpheres: [MetalRoboDynamicSphere] = []
    var disableTaskTerminations = false
    var stateTrace: String?

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
            case "--physics-substeps":
                physicsSubsteps = try Self.integer(value(), option)
                index += 1
            case "--velocity-iterations":
                velocityIterations = try Self.integer(value(), option)
                index += 1
            case "--final-velocity-iterations":
                finalVelocityIterations = try Self.integer(value(), option)
                index += 1
            case "--seed":
                guard let parsed = UInt64(try value()) else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--seed requires an unsigned integer."
                    )
                }
                seed = parsed
                index += 1
            case "--curriculum-level":
                guard let parsed = UInt32(try value()) else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--curriculum-level requires an unsigned 32-bit integer."
                    )
                }
                curriculumLevel = parsed
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
            case "--verbose":
                verbose = true
            case "--native-policy":
                nativePolicy = true
            case "--zero-actions":
                zeroActions = true
            case "--no-scheduled-resets":
                scheduledResets = false
            case "--continue-after-termination":
                disableTaskTerminations = true
            case "--policy-pack":
                policyPack = try value()
                index += 1
            case "--rollout-pack":
                rolloutPack = try value()
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
            case "--ball":
                let fields = try value().split(separator: ",")
                let values = fields.compactMap(Float.init)
                guard values.count == 8 || values.count == 9 else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--ball requires x,y,z,vx,vy,vz,radius,mass[,launch_step]."
                    )
                }
                guard values.allSatisfy(\.isFinite),
                      values[6] > 0,
                      values[7] > 0,
                      values.count == 8 ||
                        (values[8] >= 0 &&
                         values[8] <= Float(0x00ff_ffff) &&
                         values[8].rounded() == values[8])
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--ball values must be finite with positive radius and mass."
                    )
                }
                dynamicSpheres.append(
                    MetalRoboDynamicSphere(
                        position: SIMD3(values[0], values[1], values[2]),
                        linearVelocity: SIMD3(
                            values[3], values[4], values[5]
                        ),
                        radius: values[6],
                        mass: values[7],
                        launchStep:
                            values.count == 9
                            ? UInt32(values[8])
                            : 0
                    )
                )
                index += 1
            case "--state-trace":
                stateTrace = try value()
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
              chunk > 0,
              physicsSubsteps > 0,
              velocityIterations > 0,
              finalVelocityIterations >= 0
        else {
            throw MetalRoboTaskRolloutError.invalidShape(
                "envs, steps, repeats, chunk, physics substeps, and velocity iterations must be positive; final velocity iterations must be non-negative."
            )
        }
        if nativePolicy && policyPack != nil {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--native-policy and --policy-pack are mutually exclusive."
            )
        }
        if zeroActions && (nativePolicy || policyPack != nil) {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--zero-actions cannot be combined with a compiled policy."
            )
        }
        if rolloutPack != nil &&
            !nativePolicy && policyPack == nil
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--rollout-pack requires --native-policy or --policy-pack."
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
        if stateTrace != nil &&
            (environments != 1 || repeats != 1 || chunk != 1) {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--state-trace requires --envs 1 --repeats 1 --chunk 1."
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

private func makeContext(
    options: Options
) throws -> (MetalRoboTaskRolloutContext, String) {
    let configuration = MetalRoboTaskRolloutConfiguration(
        environmentCount: UInt32(options.environments),
        surface: options.surface,
        physicsSubsteps: UInt32(options.physicsSubsteps),
        velocityIterations: UInt32(options.velocityIterations),
        finalVelocityIterations:
            UInt32(options.finalVelocityIterations),
        seed: options.seed,
        dynamicSpheres: options.dynamicSpheres,
        disableTaskTerminations: options.disableTaskTerminations
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
            let (context, worldSource) =
                try makeContext(options: options)
            if options.stateTrace != nil {
                try context.setStateReadback(true)
            }
            try context.setCurriculumLevel(
                options.curriculumLevel
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
                                activation: .identity,
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
                        ],
                        criticLayers: [
                            .init(
                                inputCount:
                                    layout.criticObservationCount,
                                outputCount: hiddenCount,
                                activation: .elu,
                                weights: [Float](
                                    repeating: 0,
                                    count:
                                        layout.criticObservationCount *
                                        hiddenCount
                                ),
                                bias: [Float](
                                    repeating: 0,
                                    count: hiddenCount
                                )
                            ),
                            .init(
                                inputCount: hiddenCount,
                                outputCount: 1,
                                activation: .identity,
                                weights: [Float](
                                    repeating: 0,
                                    count: hiddenCount
                                ),
                                bias: [0]
                            ),
                        ],
                        actionLogStandardDeviation: [Float](
                            repeating: -2,
                            count: layout.actionCount
                        )
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
                let policyLogProbabilities =
                    try context.policyLogProbabilities(
                        controlStepCount: 1
                    )
                let policyValues =
                    try context.policyValues(
                        controlStepCount: 1
                    )
                let policyLatents =
                    try context.policyLatents(
                        controlStepCount: 1
                    )
                guard
                    policyLogProbabilities
                        .allSatisfy(\.isFinite),
                    policyValues.allSatisfy(\.isFinite),
                    policyLatents.allSatisfy(\.isFinite)
                else {
                    throw MetalRoboTaskRolloutError.native(
                        "Native policy produced non-finite PPO records."
                    )
                }
                if options.nativePolicy {
                    let currentLayout = context.layout
                    let logStandardDeviation = -2.0
                    let standardDeviation =
                        Foundation.exp(logStandardDeviation)
                    let halfLogTwoPi =
                        0.5 * Foundation.log(2 * Double.pi)
                    for environment in 0..<options.environments {
                        var expected = 0.0
                        let base =
                            environment *
                            currentLayout.actionCount
                        for action in
                            0..<currentLayout.actionCount
                        {
                            let latent = Double(
                                policyLatents[base + action]
                            )
                            let normal =
                                latent / standardDeviation
                            let gaussian =
                                -0.5 * normal * normal -
                                logStandardDeviation -
                                halfLogTwoPi
                            expected += gaussian
                        }
                        guard abs(
                            expected -
                            Double(
                                policyLogProbabilities[
                                    environment
                                ]
                            )
                        ) < 2.0e-3,
                              abs(policyValues[environment]) <
                                  1.0e-6
                        else {
                            throw MetalRoboTaskRolloutError.native(
                                "Native stochastic-policy record disagrees with the reference distribution."
                            )
                        }
                    }
                }
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
            var stageHighWater: [String: Int] = [:]
            var failedSteps = 0
            var policySampleCount = 0
            var transitionCount = 0
            var minimumCurriculumLevel = UInt32.max
            var finalCurriculumLevel = options.curriculumLevel
            var terminationCount = 0
            var rewardSum = 0.0
            var trackingSum = 0.0
            var rootHeightSum = 0.0
            var tiltSum = 0.0
            var maximumTilt = 0.0
            var taskRewardSum = 0.0
            var baseRewardSum = 0.0
            var jointVelocityRewardSum = 0.0
            var jointAccelerationRewardSum = 0.0
            var controlRewardSum = 0.0
            var postureRewardSum = 0.0
            var energyRewardSum = 0.0
            var contactRewardSum = 0.0
            var terminationReasonCounts: [String: Int] = [:]
            var policyRolloutFingerprint:
                UInt64 = 1_469_598_103_934_665_603
            var collectedPolicyBatches:
                [MetalRoboPolicyRolloutBatch] = []
            var stateTraceLines: [String] = []
            if options.stateTrace != nil {
                let traceLayout = context.layout
                stateTraceLines.append(
                    "# step nq=\(traceLayout.configurationCount) " +
                    "scene_bodies=\(traceLayout.sceneBodyCount) " +
                    "scene_stride=13 timestep=0.02"
                )
            }
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
                    let resetBatch = options.scheduledResets
                        ? masks(
                            startStep: globalStep,
                            stepCount: stepCount,
                            environmentCount: options.environments,
                            pending: &pending
                        )
                        : []
                    let advance: MetalRoboTaskRolloutAdvance
                    if usesCompiledPolicy {
                        advance = try context.advanceWithPolicy(
                            resetMasks: resetBatch,
                            controlStepCount: stepCount,
                            policyRevision:
                                installedPolicyRevision,
                            evaluateFinalPolicy:
                                options.rolloutPack != nil &&
                                repeatIndex + 1 ==
                                    options.repeats &&
                                completed + stepCount ==
                                    options.steps
                        )
                    } else {
                        let actionBatch: [Float]
                        if options.zeroActions {
                            actionBatch = [Float](
                                repeating: 0,
                                count:
                                    stepCount *
                                    options.environments *
                                    context.layout.actionCount
                            )
                        } else {
                            actionBatch = actions(
                                startStep: globalStep,
                                stepCount: stepCount,
                                environmentCount:
                                    options.environments,
                                actionCount:
                                    context.layout.actionCount,
                                generator: &generator
                            )
                        }
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
                    let observedTransitions:
                        [MetalRoboTaskTransition]
                    if usesCompiledPolicy {
                        let policyBatch =
                            try context.policyRolloutBatch(
                                controlStepCount: stepCount
                            )
                        guard policyBatch.transitions
                            .allSatisfy({
                                $0.policyRevision ==
                                    installedPolicyRevision
                            })
                        else {
                            throw MetalRoboTaskRolloutError.native(
                                "Policy revision changed inside a collected rollout batch."
                            )
                        }
                        fingerprint(
                            policyBatch,
                            into: &policyRolloutFingerprint
                        )
                        policySampleCount +=
                            policyBatch.sampleCount
                        if options.rolloutPack != nil {
                            collectedPolicyBatches.append(
                                policyBatch
                            )
                        }
                        observedTransitions =
                            policyBatch.transitions
                    } else {
                        observedTransitions =
                            try context.transitions(
                                controlStepCount: stepCount
                            )
                    }
                    for transition in observedTransitions {
                        transitionCount += 1
                        minimumCurriculumLevel = min(
                            minimumCurriculumLevel,
                            transition.curriculumLevel
                        )
                        guard transition.curriculumLevel >=
                                finalCurriculumLevel
                        else {
                            throw MetalRoboTaskRolloutError.native(
                                "Native task curriculum regressed."
                            )
                        }
                        finalCurriculumLevel =
                            transition.curriculumLevel
                        rewardSum += Double(transition.reward)
                        trackingSum +=
                            Double(transition.trackingScore)
                        rootHeightSum +=
                            Double(transition.rootHeight)
                        tiltSum += Double(transition.tilt)
                        maximumTilt = max(
                            maximumTilt,
                            Double(transition.tilt)
                        )
                        taskRewardSum +=
                            Double(transition.taskReward)
                        baseRewardSum +=
                            Double(transition.baseReward)
                        jointVelocityRewardSum +=
                            Double(transition.jointVelocityReward)
                        jointAccelerationRewardSum +=
                            Double(transition.jointAccelerationReward)
                        controlRewardSum +=
                            Double(transition.controlReward)
                        postureRewardSum +=
                            Double(transition.postureReward)
                        energyRewardSum +=
                            Double(transition.energyReward)
                        contactRewardSum +=
                            Double(transition.contactReward)
                        if transition.done {
                            terminationCount += 1
                            let reason = String(
                                transition.terminationReason
                            )
                            terminationReasonCounts[
                                reason,
                                default: 0
                            ] += 1
                        }
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
                    if options.stateTrace != nil {
                        let values =
                            try context.finalConfiguration() +
                            context.finalSceneStates()
                        let payload = values.map {
                            String(format: "%.9g", $0)
                        }.joined(separator: "\t")
                        stateTraceLines.append(
                            "\(globalStep + stepCount)\t\(payload)"
                        )
                    }
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

            let collectionEnd = clock.now
            let collectionLayout = context.layout
            if let stateTrace = options.stateTrace {
                try (stateTraceLines.joined(separator: "\n") + "\n")
                    .write(
                        to: URL(fileURLWithPath: stateTrace),
                        atomically: true,
                        encoding: .utf8
                    )
            }
            var rolloutPackBytes = 0
            if let rolloutPack = options.rolloutPack {
                let batch =
                    try MetalRoboPolicyRolloutBatch
                        .concatenating(
                            collectedPolicyBatches
                        )
                let bootstrapValues =
                    try context.bootstrapPolicyValues()
                let outputURL = URL(
                    fileURLWithPath: rolloutPack
                )
                try context.writePolicyRolloutPack(
                    batch,
                    bootstrapValues: bootstrapValues,
                    id: "swift_native_task_rollout",
                    to: outputURL
                )
                let attributes =
                    try FileManager.default.attributesOfItem(
                        atPath: outputURL.path
                    )
                rolloutPackBytes =
                    (attributes[.size] as? NSNumber)?
                        .intValue ?? 0
            }

            let elapsed = start.duration(to: collectionEnd)
            let seconds =
                Double(elapsed.components.seconds) +
                Double(elapsed.components.attoseconds) / 1e18
            let layout = collectionLayout
            let environmentSteps =
                options.environments *
                options.steps *
                options.repeats
            let output: [String: Any] = [
                "benchmark": "swift_native_task_rollout",
                "world_source": worldSource,
                "action_source":
                    options.policyPack != nil
                    ? "policy_pack"
                    : options.nativePolicy
                    ? "compiled_policy"
                    : options.zeroActions
                    ? "zero"
                    : "host_stream",
                "device": context.deviceName,
                "solver_mode": "temporal_cone",
                "scene":
                    options.surface == .terrain
                    ? "terrain"
                    : "ground",
                "dynamic_sphere_count": options.dynamicSpheres.count,
                "state_trace": options.stateTrace ?? "",
                "environments": options.environments,
                "steps_per_repeat": options.steps,
                "repeats": options.repeats,
                "control_steps_per_submission": options.chunk,
                "physics_substeps": options.physicsSubsteps,
                "velocity_iterations": options.velocityIterations,
                "final_velocity_iterations":
                    options.finalVelocityIterations,
                "initial_task_curriculum_level":
                    options.curriculumLevel,
                "minimum_task_curriculum_level":
                    minimumCurriculumLevel,
                "final_task_curriculum_level":
                    finalCurriculumLevel,
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
                "host_requested_resets": totalResets,
                "scheduled_resets": options.scheduledResets,
                "maximum_active_contacts": maximumContacts,
                "maximum_manifolds": maximumManifolds,
                "stage_high_water": stageHighWater,
                "failed_environment_steps": failedSteps,
                "termination_count": terminationCount,
                "termination_reason_counts":
                    terminationReasonCounts,
                "mean_reward":
                    rewardSum / Double(max(transitionCount, 1)),
                "mean_tracking_score":
                    trackingSum /
                    Double(max(transitionCount, 1)),
                "mean_root_height":
                    rootHeightSum /
                    Double(max(transitionCount, 1)),
                "mean_tilt":
                    tiltSum / Double(max(transitionCount, 1)),
                "maximum_tilt": maximumTilt,
                "mean_task_reward":
                    taskRewardSum /
                    Double(max(transitionCount, 1)),
                "mean_base_reward":
                    baseRewardSum /
                    Double(max(transitionCount, 1)),
                "mean_joint_velocity_reward":
                    jointVelocityRewardSum /
                    Double(max(transitionCount, 1)),
                "mean_joint_acceleration_reward":
                    jointAccelerationRewardSum /
                    Double(max(transitionCount, 1)),
                "mean_control_reward":
                    controlRewardSum /
                    Double(max(transitionCount, 1)),
                "mean_posture_reward":
                    postureRewardSum /
                    Double(max(transitionCount, 1)),
                "mean_energy_reward":
                    energyRewardSum /
                    Double(max(transitionCount, 1)),
                "mean_contact_reward":
                    contactRewardSum /
                    Double(max(transitionCount, 1)),
                "policy_sample_count": policySampleCount,
                "policy_rollout_fingerprint":
                    usesCompiledPolicy
                    ? String(policyRolloutFingerprint)
                    : "",
                "rollout_pack": options.rolloutPack ?? "",
                "rollout_pack_bytes": rolloutPackBytes,
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
