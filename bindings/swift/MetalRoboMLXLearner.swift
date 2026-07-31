import Darwin
import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// PPO configuration for the in-process Apple-native learning boundary.
/// Simulator state never enters MLX; only compact rollout tensors do.
struct MetalRoboMLXPPOConfiguration: Sendable {
    var updateEpochs: Int
    var minibatchSize: Int
    var hiddenSizes: [Int] = [512, 256, 128]
    var criticHiddenSizes: [Int]? = nil
    var learningRate: Float
    var clipRatio: Float
    var valueCoefficient: Float
    var entropyCoefficient: Float
    var maximumGradientNorm: Float
    var targetKL: Float
    var minimumLearningRate: Float = 1.0e-5
    var maximumLearningRate: Float = 1.0e-2
    var discount: Float
    var gaeLambda: Float
    var initialLogStandardDeviation: Float
    var observationClip: Float = 100
    var seed: UInt64

    func validate() throws {
        let finite = [
            learningRate,
            clipRatio,
            valueCoefficient,
            entropyCoefficient,
            maximumGradientNorm,
            targetKL,
            minimumLearningRate,
            maximumLearningRate,
            discount,
            gaeLambda,
            initialLogStandardDeviation,
            observationClip,
        ].allSatisfy(\.isFinite)
        guard updateEpochs > 0,
              minibatchSize > 0,
              !hiddenSizes.isEmpty,
              hiddenSizes.allSatisfy({ $0 > 0 }),
              criticHiddenSizes?.allSatisfy({ $0 > 0 }) ?? true,
              finite,
              minimumLearningRate > 0,
              minimumLearningRate <= learningRate,
              learningRate <= maximumLearningRate,
              clipRatio > 0,
              valueCoefficient >= 0,
              entropyCoefficient >= 0,
              maximumGradientNorm > 0,
              targetKL > 0,
              discount >= 0,
              discount <= 1,
              gaeLambda >= 0,
              gaeLambda <= 1,
              initialLogStandardDeviation >= -5,
              initialLogStandardDeviation <= 2,
              observationClip > 0
        else {
            throw MetalRoboSimulationError.invalidShape(
                "Native MLX PPO configuration is invalid."
            )
        }
    }

    var signature: String {
        [
            String(updateEpochs),
            String(minibatchSize),
            hiddenSizes.map(String.init).joined(separator: ","),
            (criticHiddenSizes ?? hiddenSizes)
                .map(String.init).joined(separator: ","),
            String(learningRate),
            String(clipRatio),
            String(valueCoefficient),
            String(entropyCoefficient),
            String(maximumGradientNorm),
            String(targetKL),
            String(minimumLearningRate),
            String(maximumLearningRate),
            String(discount),
            String(gaeLambda),
            String(initialLogStandardDeviation),
            String(observationClip),
            String(seed),
        ].joined(separator: "|")
    }
}

private final class MetalRoboStableELU: Module, UnaryLayer {
    func callAsFunction(_ value: MLXArray) -> MLXArray {
        let negative = MLX.exp(MLX.minimum(value, 0)) - 1
        return MLX.which(value .>= 0, value, negative)
    }
}

private func metalRoboMLP(
    inputCount: Int,
    outputCount: Int,
    hiddenSizes: [Int],
    outputGain: Float
) -> Sequential {
    var layers: [UnaryLayer] = []
    var previous = inputCount
    for width in hiddenSizes {
        layers.append(Linear(previous, width))
        layers.append(MetalRoboStableELU())
        previous = width
    }
    let outputBound = outputGain / sqrt(Float(previous))
    layers.append(
        Linear(
            weight: MLXRandom.uniform(
                -outputBound ..< outputBound,
                [outputCount, previous]
            ),
            bias: .zeros([outputCount])
        )
    )
    return Sequential(layers: layers)
}

private final class MetalRoboActorCritic: Module {
    @ModuleInfo var actor: Sequential
    @ModuleInfo var critic: Sequential
    let logStandardDeviation: MLXArray
    let observationClip: Float

    init(
        actorObservationCount: Int,
        criticObservationCount: Int,
        actionCount: Int,
        configuration: MetalRoboMLXPPOConfiguration
    ) {
        actor = metalRoboMLP(
            inputCount: actorObservationCount,
            outputCount: actionCount,
            hiddenSizes: configuration.hiddenSizes,
            outputGain: 0.01
        )
        critic = metalRoboMLP(
            inputCount: criticObservationCount,
            outputCount: 1,
            hiddenSizes:
                configuration.criticHiddenSizes ??
                configuration.hiddenSizes,
            outputGain: 1
        )
        logStandardDeviation = MLXArray.full(
            [actionCount],
            values: MLXArray(
                configuration.initialLogStandardDeviation
            )
        )
        observationClip = configuration.observationClip
        super.init()
    }

    func actorMean(_ observations: MLXArray) -> MLXArray {
        actor(
            MLX.clip(
                observations,
                min: -observationClip,
                max: observationClip
            )
        )
    }

    func value(_ observations: MLXArray) -> MLXArray {
        critic(
            MLX.clip(
                observations,
                min: -observationClip,
                max: observationClip
            )
        ).squeezed(axis: -1)
    }

    func evaluate(
        actorObservations: MLXArray,
        criticObservations: MLXArray,
        latents: MLXArray
    ) -> (logProbability: MLXArray, entropy: MLXArray, value: MLXArray) {
        let mean = actorMean(actorObservations)
        let logStandardDeviation = MLX.clip(
            logStandardDeviation,
            min: -5,
            max: 2
        )
        let standardized =
            (latents - mean) * MLX.exp(-logStandardDeviation)
        let gaussian =
            -0.5 * MLX.square(standardized) -
            logStandardDeviation -
            Float(0.5 * log(2.0 * Double.pi))
        let logProbability = MLX.sum(gaussian, axis: -1)
        let entropy = MLX.sum(
            logStandardDeviation +
                Float(0.5 * (1.0 + log(2.0 * Double.pi)))
        )
        return (
            logProbability,
            entropy,
            value(criticObservations)
        )
    }
}

private struct MetalRoboSplitMix64 {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }
}

/// In-process MLX Swift learner. Swift owns PPO scheduling and checkpoints;
/// MLX owns tensor algebra and automatic differentiation on Apple GPU.
final class MetalRoboMLXPPOTrainer {
    private static let checkpointSchema = "metalrobo.mlx-swift-ppo.1"
    private static let mlxSwiftVersion = "0.31.3"

    let actorObservationCount: Int
    let criticObservationCount: Int
    let actionCount: Int
    let policyID: String
    let trainingPolicyURL: URL
    let deploymentPolicyURL: URL
    let learnerStateURL: URL
    let taskFingerprint: UInt64
    let configuration: MetalRoboMLXPPOConfiguration

    private let model: MetalRoboActorCritic
    private let lossAndGradient: (
        MetalRoboActorCritic,
        [MLXArray]
    ) -> ([MLXArray], ModuleParameters)
    private var firstMoments: [String: MLXArray] = [:]
    private var secondMoments: [String: MLXArray] = [:]
    private var optimizerStep: UInt64 = 0
    private var currentLearningRate: Float

    private(set) var revision: UInt64 = 1
    private(set) var taskCurriculumLevel: UInt32 = 0
    private(set) var stateRestored = false

    init(
        actorObservationCount: Int,
        criticObservationCount: Int,
        actionCount: Int,
        policyID: String?,
        trainingPolicyURL: URL,
        deploymentPolicyURL: URL,
        learnerStateURL: URL,
        taskFingerprint: UInt64,
        configuration: MetalRoboMLXPPOConfiguration,
        resumeIfPresent: Bool
    ) throws {
        try configuration.validate()
        let checkpointExists = resumeIfPresent &&
            FileManager.default.fileExists(atPath: learnerStateURL.path)
        let restoredCheckpoint: (
            arrays: [String: MLXArray],
            metadata: [String: String]
        )? = checkpointExists
            ? try MLX.loadArraysAndMetadata(
                url: learnerStateURL,
                stream: .cpu
            )
            : nil
        guard let resolvedPolicyID =
                  policyID ?? restoredCheckpoint?.metadata["policy_id"],
              actorObservationCount > 0,
              criticObservationCount > 0,
              actionCount > 0,
              !resolvedPolicyID.isEmpty,
              taskFingerprint != 0
        else {
            throw MetalRoboSimulationError.invalidShape(
                "Native MLX learner identity or compiled dimensions are invalid."
            )
        }
        self.actorObservationCount = actorObservationCount
        self.criticObservationCount = criticObservationCount
        self.actionCount = actionCount
        self.policyID = resolvedPolicyID
        self.trainingPolicyURL = trainingPolicyURL
        self.deploymentPolicyURL = deploymentPolicyURL
        self.learnerStateURL = learnerStateURL
        self.taskFingerprint = taskFingerprint
        self.configuration = configuration
        currentLearningRate = configuration.learningRate

        MLXRandom.seed(configuration.seed)
        model = MetalRoboActorCritic(
            actorObservationCount: actorObservationCount,
            criticObservationCount: criticObservationCount,
            actionCount: actionCount,
            configuration: configuration
        )
        lossAndGradient = MLXNN.valueAndGrad(model: model) {
            model,
            arguments in
            Self.loss(
                model: model,
                arguments: arguments,
                configuration: configuration
            )
        }
        MLX.eval(model)

        if let restoredCheckpoint {
            try restoreCheckpoint(
                arrays: restoredCheckpoint.arrays,
                metadata: restoredCheckpoint.metadata
            )
            stateRestored = true
        }
        try requireFiniteTrainingState()
    }

    func initialPolicyPack() throws -> MetalRoboPolicyPack {
        try policyPack(stochastic: true)
    }

    func publishCurrentPolicies() throws {
        try policyPack(stochastic: true).write(
            to: trainingPolicyURL
        )
        try policyPack(stochastic: false).write(
            to: deploymentPolicyURL
        )
    }

    func update(
        rollout: MetalRoboPolicyRolloutBatch,
        bootstrapValues: [Float],
        curriculumLevel: UInt32
    ) throws -> [String: Any] {
        let sampleCount = try validate(
            rollout: rollout,
            bootstrapValues: bootstrapValues
        )
        guard curriculumLevel >= taskCurriculumLevel else {
            throw MetalRoboSimulationError.invalidShape(
                "Task curriculum cannot move backwards."
            )
        }

        let previousParameters = model.parameters()
        let previousFirstMoments = firstMoments
        let previousSecondMoments = secondMoments
        let previousOptimizerStep = optimizerStep
        let previousRevision = revision
        let previousCurriculum = taskCurriculumLevel
        let previousLearningRate = currentLearningRate

        do {
            let prepared = prepareBatch(
                rollout: rollout,
                bootstrapValues: bootstrapValues
            )
            let actorObservations = MLXArray(
                rollout.actorObservations,
                [sampleCount, actorObservationCount]
            )
            let criticObservations = MLXArray(
                rollout.criticObservations,
                [sampleCount, criticObservationCount]
            )
            let latents = MLXArray(
                rollout.latents,
                [sampleCount, actionCount]
            )
            let oldLogProbabilities = MLXArray(
                rollout.logProbabilities
            )
            let oldValues = MLXArray(rollout.values)
            let rawAdvantages = MLXArray(prepared.advantages)
            let advantages =
                (rawAdvantages - MLX.mean(rawAdvantages)) /
                (MLX.std(rawAdvantages) + 1.0e-8)
            let returns = MLXArray(prepared.returns)
            let oldMeans = model.actorMean(actorObservations)
            let oldLogStandardDeviation = MLX.clip(
                model.logStandardDeviation,
                min: -5,
                max: 2
            )
            MLX.eval(oldMeans, oldLogStandardDeviation, advantages)

            var totals = Array(repeating: 0.0, count: 7)
            var totalGradientNorm = 0.0
            var minibatchUpdates = 0
            let clock = ContinuousClock()
            let start = clock.now

            for epoch in 0..<configuration.updateEpochs {
                let permutation = shuffledIndices(
                    count: sampleCount,
                    revision: revision,
                    epoch: epoch
                )
                var offset = 0
                while offset < sampleCount {
                    let end = min(
                        offset + configuration.minibatchSize,
                        sampleCount
                    )
                    let indices = MLXArray(
                        Array(permutation[offset..<end])
                    )
                    let arguments = [
                        actorObservations.take(indices, axis: 0),
                        criticObservations.take(indices, axis: 0),
                        latents.take(indices, axis: 0),
                        oldLogProbabilities.take(indices, axis: 0),
                        oldMeans.take(indices, axis: 0),
                        oldLogStandardDeviation,
                        oldValues.take(indices, axis: 0),
                        advantages.take(indices, axis: 0),
                        returns.take(indices, axis: 0),
                    ]
                    let (metrics, gradients) = lossAndGradient(
                        model,
                        arguments
                    )
                    let (clippedGradients, gradientNorm) =
                        MLXOptimizers.clipGradNorm(
                            gradients: gradients,
                            maxNorm:
                                configuration.maximumGradientNorm
                        )
                    applyAdam(gradients: clippedGradients)
                    let evaluation: [Any] =
                        metrics + [gradientNorm] +
                        model.parameters().flattened().map(\.1) +
                        Array(firstMoments.values) +
                        Array(secondMoments.values)
                    MLX.eval(evaluation)

                    let numeric = metrics.map {
                        Double($0.item(Float.self))
                    }
                    let gradient = Double(
                        gradientNorm.item(Float.self)
                    )
                    guard numeric.allSatisfy(\.isFinite),
                          gradient.isFinite
                    else {
                        throw MetalRoboSimulationError.native(
                            "Native MLX PPO produced non-finite minibatch metrics."
                        )
                    }
                    for index in totals.indices {
                        totals[index] += numeric[index]
                    }
                    totalGradientNorm += gradient
                    minibatchUpdates += 1

                    let kl = numeric[4]
                    if kl > 2 * Double(configuration.targetKL) {
                        currentLearningRate = max(
                            configuration.minimumLearningRate,
                            currentLearningRate / 1.5
                        )
                    } else if kl > 0 &&
                        kl < 0.5 * Double(configuration.targetKL)
                    {
                        currentLearningRate = min(
                            configuration.maximumLearningRate,
                            currentLearningRate * 1.5
                        )
                    }
                    offset = end
                }
            }

            revision += 1
            taskCurriculumLevel = curriculumLevel
            try requireFiniteTrainingState()
            let trainingPack = try policyPack(stochastic: true)
            let deploymentPack = try policyPack(stochastic: false)
            try trainingPack.write(to: trainingPolicyURL)
            try deploymentPack.write(to: deploymentPolicyURL)
            try writeCheckpointAtomically()

            let elapsed = start.duration(to: clock.now)
            let seconds =
                Double(elapsed.components.seconds) +
                Double(elapsed.components.attoseconds) / 1e18
            let denominator = Double(max(minibatchUpdates, 1))
            let standardDeviation = MLX.mean(
                MLX.exp(
                    MLX.clip(
                        model.logStandardDeviation,
                        min: -5,
                        max: 2
                    )
                )
            )
            MLX.eval(standardDeviation)
            return metricsRecord(
                rollout: rollout,
                totals: totals,
                denominator: denominator,
                meanGradientNorm:
                    totalGradientNorm / denominator,
                updateSeconds: seconds,
                minibatchUpdates: minibatchUpdates,
                meanActionStandardDeviation: Double(
                    standardDeviation.item(Float.self)
                )
            )
        } catch {
            model.update(parameters: previousParameters)
            firstMoments = previousFirstMoments
            secondMoments = previousSecondMoments
            optimizerStep = previousOptimizerStep
            revision = previousRevision
            taskCurriculumLevel = previousCurriculum
            currentLearningRate = previousLearningRate
            MLX.eval(model)
            throw error
        }
    }

    private static func loss(
        model: MetalRoboActorCritic,
        arguments: [MLXArray],
        configuration: MetalRoboMLXPPOConfiguration
    ) -> [MLXArray] {
        precondition(arguments.count == 9)
        let evaluated = model.evaluate(
            actorObservations: arguments[0],
            criticObservations: arguments[1],
            latents: arguments[2]
        )
        let oldLogProbabilities = arguments[3]
        let oldMeans = arguments[4]
        let oldLogStandardDeviation = arguments[5]
        let oldValues = arguments[6]
        let advantages = arguments[7]
        let returns = arguments[8]
        let currentMeans = model.actorMean(arguments[0])
        let currentLogStandardDeviation = MLX.clip(
            model.logStandardDeviation,
            min: -5,
            max: 2
        )
        let inverseCurrentVariance = MLX.exp(
            -2 * currentLogStandardDeviation
        )
        let klDivergence = MLX.mean(
            MLX.sum(
                currentLogStandardDeviation -
                    oldLogStandardDeviation +
                    0.5 * (
                        MLX.exp(
                            2 * (
                                oldLogStandardDeviation -
                                currentLogStandardDeviation
                            )
                        ) +
                        MLX.square(oldMeans - currentMeans) *
                        inverseCurrentVariance - 1
                    ),
                axis: -1
            )
        )
        let logRatio =
            evaluated.logProbability - oldLogProbabilities
        let ratio = MLX.exp(logRatio)
        let unclipped = ratio * advantages
        let clipped = MLX.clip(
            ratio,
            min: 1 - configuration.clipRatio,
            max: 1 + configuration.clipRatio
        ) * advantages
        let policyLoss = -MLX.mean(
            MLX.minimum(unclipped, clipped)
        )
        let clippedValues = oldValues + MLX.clip(
            evaluated.value - oldValues,
            min: -configuration.clipRatio,
            max: configuration.clipRatio
        )
        let valueLoss = MLX.mean(
            MLX.maximum(
                MLX.square(evaluated.value - returns),
                MLX.square(clippedValues - returns)
            )
        )
        let entropy = MLX.mean(evaluated.entropy)
        let loss =
            policyLoss +
            configuration.valueCoefficient * valueLoss -
            configuration.entropyCoefficient * entropy
        let approximateKL = MLX.mean((ratio - 1) - logRatio)
        let clipFraction = MLX.mean(
            ((ratio - 1).abs() .> configuration.clipRatio)
                .asType(.float32)
        )
        return [
            loss,
            policyLoss,
            valueLoss,
            entropy,
            klDivergence,
            approximateKL,
            clipFraction,
        ]
    }

    private func applyAdam(gradients: ModuleParameters) {
        optimizerStep += 1
        let beta1: Float = 0.9
        let beta2: Float = 0.999
        let epsilon: Float = 1.0e-8
        let bias1 = Float(1 - pow(Double(beta1), Double(optimizerStep)))
        let bias2 = Float(1 - pow(Double(beta2), Double(optimizerStep)))
        let parameters = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened()
        )
        let gradientValues = Dictionary(
            uniqueKeysWithValues: gradients.flattened()
        )
        var updated: [(String, MLXArray)] = []
        updated.reserveCapacity(parameters.count)
        for key in parameters.keys.sorted() {
            guard let parameter = parameters[key],
                  let gradient = gradientValues[key]
            else {
                preconditionFailure(
                    "MLX parameter and gradient trees disagree."
                )
            }
            let first =
                beta1 * (firstMoments[key] ?? .zeros(like: parameter)) +
                (1 - beta1) * gradient
            let second =
                beta2 * (secondMoments[key] ?? .zeros(like: parameter)) +
                (1 - beta2) * MLX.square(gradient)
            firstMoments[key] = first
            secondMoments[key] = second
            let firstCorrected = first / bias1
            let secondCorrected = second / bias2
            updated.append(
                (
                    key,
                    parameter - currentLearningRate *
                        firstCorrected /
                        (MLX.sqrt(secondCorrected) + epsilon)
                )
            )
        }
        model.update(
            parameters: ModuleParameters.unflattened(updated)
        )
    }

    private func prepareBatch(
        rollout: MetalRoboPolicyRolloutBatch,
        bootstrapValues: [Float]
    ) -> (advantages: [Float], returns: [Float]) {
        let steps = rollout.controlStepCount
        let environments = rollout.environmentCount
        var advantages = Array(
            repeating: Float.zero,
            count: rollout.sampleCount
        )
        var running = Array(
            repeating: Float.zero,
            count: environments
        )
        for step in stride(from: steps - 1, through: 0, by: -1) {
            for environment in 0..<environments {
                let index = step * environments + environment
                let transition = rollout.transitions[index]
                let continuing: Float = transition.done ? 0 : 1
                let nextValue =
                    step + 1 == steps
                    ? bootstrapValues[environment]
                    : rollout.values[index + environments]
                let timeoutBootstrap: Float =
                    transition.timeout && !transition.physicsError
                    ? transition.timeoutBootstrapValue
                    : 0
                let delta =
                    transition.reward +
                    configuration.discount * timeoutBootstrap +
                    configuration.discount * continuing * nextValue -
                    rollout.values[index]
                running[environment] =
                    delta +
                    configuration.discount *
                    configuration.gaeLambda * continuing *
                    running[environment]
                advantages[index] = running[environment]
            }
        }
        let returns = zip(advantages, rollout.values).map(+)
        return (advantages, returns)
    }

    private func validate(
        rollout: MetalRoboPolicyRolloutBatch,
        bootstrapValues: [Float]
    ) throws -> Int {
        let samples = rollout.sampleCount
        guard rollout.controlStepCount > 0,
              rollout.environmentCount > 0,
              rollout.actorObservationCount == actorObservationCount,
              rollout.criticObservationCount == criticObservationCount,
              rollout.actionCount == actionCount,
              rollout.actorObservations.count ==
                  samples * actorObservationCount,
              rollout.criticObservations.count ==
                  samples * criticObservationCount,
              rollout.latents.count == samples * actionCount,
              rollout.logProbabilities.count == samples,
              rollout.values.count == samples,
              rollout.transitions.count == samples,
              bootstrapValues.count == rollout.environmentCount,
              rollout.actorObservations.allSatisfy(\.isFinite),
              rollout.criticObservations.allSatisfy(\.isFinite),
              rollout.latents.allSatisfy(\.isFinite),
              rollout.logProbabilities.allSatisfy(\.isFinite),
              rollout.values.allSatisfy(\.isFinite),
              bootstrapValues.allSatisfy(\.isFinite),
              rollout.transitions.allSatisfy({
                  $0.reward.isFinite &&
                      $0.timeoutBootstrapValue.isFinite
              })
        else {
            throw MetalRoboSimulationError.invalidShape(
                "Native rollout batch shape or values are invalid for PPO."
            )
        }
        return samples
    }

    private func shuffledIndices(
        count: Int,
        revision: UInt64,
        epoch: Int
    ) -> [Int] {
        var state =
            configuration.seed ^
            (revision &* 0xD2B7_4407_B1CE_6E93) ^
            (UInt64(epoch) &* 0xCA5A_8263_9512_1157)
        var mixer = MetalRoboSplitMix64(state: state)
        state = mixer.next()
        var generator = MetalRoboSplitMix64(state: state)
        var result = Array(0..<count)
        guard count > 1 else {
            return result
        }
        for index in stride(from: count - 1, through: 1, by: -1) {
            let selected = Int(generator.next() % UInt64(index + 1))
            result.swapAt(index, selected)
        }
        return result
    }

    private func policyPack(
        stochastic: Bool
    ) throws -> MetalRoboPolicyPack {
        MLX.eval(model)
        let actorLayers = try denseLayers(model.actor)
        let criticLayers = try denseLayers(model.critic)
        let logStandardDeviation = MLX.clip(
            model.logStandardDeviation,
            min: -5,
            max: 2
        )
        MLX.eval(logStandardDeviation)
        return MetalRoboPolicyPack(
            id: policyID,
            revision: revision,
            layers: actorLayers,
            criticLayers: criticLayers,
            actionLogStandardDeviation:
                stochastic
                ? logStandardDeviation.asArray(Float.self)
                : [],
            observationClip: configuration.observationClip,
            actionClip: Float.greatestFiniteMagnitude
        )
    }

    private func denseLayers(
        _ network: Sequential
    ) throws -> [MetalRoboPolicyDenseLayer] {
        let linears = network.layers.compactMap { $0 as? Linear }
        guard !linears.isEmpty else {
            throw MetalRoboSimulationError.native(
                "Native MLX policy network contains no dense layers."
            )
        }
        return linears.enumerated().map { index, layer in
            MLX.eval(layer.weight, layer.bias as Any)
            let (outputCount, inputCount) = layer.weight.shape2
            return MetalRoboPolicyDenseLayer(
                inputCount: inputCount,
                outputCount: outputCount,
                activation:
                    index + 1 == linears.count ? .identity : .elu,
                weights: layer.weight.asArray(Float.self),
                bias:
                    layer.bias?.asArray(Float.self) ??
                    Array(repeating: 0, count: outputCount)
            )
        }
    }

    private func requireFiniteTrainingState() throws {
        let named = model.parameters().flattened().map {
            ("model.\($0.0)", $0.1)
        } + firstMoments.map { ("adam.first.\($0.key)", $0.value) } +
            secondMoments.map { ("adam.second.\($0.key)", $0.value) }
        let checks = named.map { ($0.0, MLX.isFinite($0.1).all()) }
        MLX.eval(checks.map(\.1))
        let invalid = checks.compactMap {
            $0.1.item(Bool.self) ? nil : $0.0
        }
        guard invalid.isEmpty else {
            throw MetalRoboSimulationError.native(
                "Native MLX PPO contains non-finite state in " +
                    invalid.joined(separator: ", ")
            )
        }
    }

    private func writeCheckpointAtomically() throws {
        let directory = learnerStateURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        let temporaryURL = directory.appendingPathComponent(
            ".\(learnerStateURL.deletingPathExtension().lastPathComponent)." +
                "\(UUID().uuidString).safetensors"
        )
        var arrays: [String: MLXArray] = [:]
        for (key, value) in model.parameters().flattened() {
            arrays["model.\(key)"] = value
        }
        for (key, value) in firstMoments {
            arrays["adam.first.\(key)"] = value
        }
        for (key, value) in secondMoments {
            arrays["adam.second.\(key)"] = value
        }
        let metadata = checkpointMetadata()
        do {
            try MLX.save(
                arrays: arrays,
                metadata: metadata,
                url: temporaryURL
            )
            guard Darwin.rename(
                temporaryURL.path,
                learnerStateURL.path
            ) == 0 else {
                throw MetalRoboSimulationError.native(
                    "Atomic learner checkpoint rename failed: " +
                        String(cString: strerror(errno))
                )
            }
        } catch {
            try? FileManager.default.removeItem(at: temporaryURL)
            throw error
        }
    }

    private func checkpointMetadata() -> [String: String] {
        [
            "schema": Self.checkpointSchema,
            "mlx_swift_version": Self.mlxSwiftVersion,
            "policy_id": policyID,
            "revision": String(revision),
            "task_curriculum_level": String(taskCurriculumLevel),
            "task_fingerprint": String(taskFingerprint),
            "actor_observation_count": String(actorObservationCount),
            "critic_observation_count": String(criticObservationCount),
            "action_count": String(actionCount),
            "optimizer_step": String(optimizerStep),
            "learning_rate": String(currentLearningRate),
            "configuration": configuration.signature,
        ]
    }

    private func restoreCheckpoint(
        arrays: [String: MLXArray],
        metadata: [String: String]
    ) throws {
        func require(_ key: String, _ expected: String) throws {
            guard metadata[key] == expected else {
                throw MetalRoboSimulationError.invalidShape(
                    "Native MLX checkpoint \(key) does not match this task."
                )
            }
        }
        try require("schema", Self.checkpointSchema)
        try require("mlx_swift_version", Self.mlxSwiftVersion)
        try require("policy_id", policyID)
        try require("task_fingerprint", String(taskFingerprint))
        try require(
            "actor_observation_count",
            String(actorObservationCount)
        )
        try require(
            "critic_observation_count",
            String(criticObservationCount)
        )
        try require("action_count", String(actionCount))
        try require("configuration", configuration.signature)
        guard let restoredRevision = metadata["revision"].flatMap(UInt64.init),
              restoredRevision > 0,
              let restoredCurriculum =
                  metadata["task_curriculum_level"]
                    .flatMap(UInt32.init),
              let restoredStep =
                  metadata["optimizer_step"].flatMap(UInt64.init),
              let restoredLearningRate =
                  metadata["learning_rate"].flatMap(Float.init),
              restoredLearningRate.isFinite,
              restoredLearningRate >= configuration.minimumLearningRate,
              restoredLearningRate <= configuration.maximumLearningRate
        else {
            throw MetalRoboSimulationError.invalidShape(
                "Native MLX checkpoint counters are invalid."
            )
        }

        let current = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened()
        )
        var restoredParameters: [(String, MLXArray)] = []
        var restoredFirst: [String: MLXArray] = [:]
        var restoredSecond: [String: MLXArray] = [:]
        for key in current.keys.sorted() {
            guard let expected = current[key],
                  let parameter = arrays["model.\(key)"],
                  parameter.shape == expected.shape
            else {
                throw MetalRoboSimulationError.invalidShape(
                    "Native MLX checkpoint parameter \(key) is missing or has the wrong shape."
                )
            }
            restoredParameters.append((key, parameter))
            if restoredStep > 0 {
                guard let first = arrays["adam.first.\(key)"],
                      let second = arrays["adam.second.\(key)"],
                      first.shape == expected.shape,
                      second.shape == expected.shape
                else {
                    throw MetalRoboSimulationError.invalidShape(
                        "Native MLX checkpoint Adam state for \(key) is incomplete."
                    )
                }
                restoredFirst[key] = first
                restoredSecond[key] = second
            }
        }
        model.update(
            parameters: ModuleParameters.unflattened(
                restoredParameters
            )
        )
        firstMoments = restoredFirst
        secondMoments = restoredSecond
        optimizerStep = restoredStep
        revision = restoredRevision
        taskCurriculumLevel = restoredCurriculum
        currentLearningRate = restoredLearningRate
        MLX.eval(model)
    }

    private func metricsRecord(
        rollout: MetalRoboPolicyRolloutBatch,
        totals: [Double],
        denominator: Double,
        meanGradientNorm: Double,
        updateSeconds: Double,
        minibatchUpdates: Int,
        meanActionStandardDeviation: Double
    ) -> [String: Any] {
        let samples = Double(rollout.sampleCount)
        let sum: ((MetalRoboTaskTransition) -> Float) -> Double = {
            selector in
            rollout.transitions.reduce(0) {
                $0 + Double(selector($1))
            }
        }
        return [
            "loss": totals[0] / denominator,
            "policy_loss": totals[1] / denominator,
            "value_loss": totals[2] / denominator,
            "entropy": totals[3] / denominator,
            "kl_divergence": totals[4] / denominator,
            "approximate_kl": totals[5] / denominator,
            "clip_fraction": totals[6] / denominator,
            "gradient_norm": meanGradientNorm,
            "learning_rate": Double(currentLearningRate),
            "update_seconds": updateSeconds,
            "minibatch_updates": minibatchUpdates,
            "policy_revision": revision,
            "task_curriculum_level": taskCurriculumLevel,
            "mean_action_standard_deviation":
                meanActionStandardDeviation,
            "mean_reward": sum(\.reward) / samples,
            "mean_tracking_score": sum(\.trackingScore) / samples,
            "mean_root_height": sum(\.rootHeight) / samples,
            "mean_tilt": sum(\.tilt) / samples,
            "done_count": rollout.transitions.filter(\.done).count,
            "timeout_count":
                rollout.transitions.filter(\.timeout).count,
            "physics_error_count":
                rollout.transitions.filter(\.physicsError).count,
        ]
    }
}
