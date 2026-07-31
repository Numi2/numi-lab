import Foundation
import Metal
import MetalRoboC

private func withOptionalCString<Result>(
    _ value: String?,
    _ body: (UnsafePointer<CChar>?) -> Result
) -> Result {
    guard let value else {
        return body(nil)
    }
    return value.withCString(body)
}

private func withUnsafeFloatBuffers<Result>(
    _ arrays: [[Float]],
    _ body: ([UnsafeBufferPointer<Float>]) -> Result
) -> Result {
    var buffers: [UnsafeBufferPointer<Float>] = []
    buffers.reserveCapacity(arrays.count)
    func visit(_ index: Int) -> Result {
        guard index < arrays.count else {
            return body(buffers)
        }
        return arrays[index].withUnsafeBufferPointer { buffer in
            buffers.append(buffer)
            defer {
                buffers.removeLast()
            }
            return visit(index + 1)
        }
    }
    return visit(0)
}

public enum MetalRoboSimulationError:
    Error, CustomStringConvertible
{
    case native(String)
    case invalidShape(String)

    public var description: String {
        switch self {
        case .native(let message):
            return message
        case .invalidShape(let message):
            return message
        }
    }
}

public enum MetalRoboBuiltinSurface: UInt32, Sendable {
    case ground = 0
    case terrain = 1
}

public enum MetalRoboG1ActuatorPreset: UInt32, Sendable {
    case unitreeURDFRev10 = 0
    case unitreeMJCFRev10 = 1
    case unitreeRLLab4960b84 = 2
}

public enum MetalRoboNumiIterationPolicy: UInt32, Sendable {
    case fixedBudget = 0
    case residualConverged = 1
}

public struct MetalRoboNumiSolverSettings: Sendable {
    public var iterationPolicy: MetalRoboNumiIterationPolicy
    public var temporalSubsteps: UInt32

    public init(
        iterationPolicy: MetalRoboNumiIterationPolicy = .fixedBudget,
        temporalSubsteps: UInt32 = 4
    ) {
        self.iterationPolicy = iterationPolicy
        self.temporalSubsteps = temporalSubsteps
    }
}

public enum MetalRoboRobotRootMode: UInt32, Sendable {
    case fixed = 0
    case floating = 1
}

public enum MetalRoboPolicyActivation: UInt32, Sendable {
    case identity = 0
    case relu = 1
    case tanh = 2
    case elu = 3
    case silu = 4
}

public struct MetalRoboPolicyDenseLayer: Sendable {
    public var inputCount: Int
    public var outputCount: Int
    public var activation: MetalRoboPolicyActivation
    public var weights: [Float]
    public var bias: [Float]

    public init(
        inputCount: Int,
        outputCount: Int,
        activation: MetalRoboPolicyActivation,
        weights: [Float],
        bias: [Float]
    ) {
        self.inputCount = inputCount
        self.outputCount = outputCount
        self.activation = activation
        self.weights = weights
        self.bias = bias
    }
}

public struct MetalRoboPolicyPack: Sendable {
    public var id: String
    public var revision: UInt64
    public var observationMean: [Float]
    public var observationInverseStandardDeviation: [Float]
    public var layers: [MetalRoboPolicyDenseLayer]
    public var criticObservationMean: [Float]
    public var criticObservationInverseStandardDeviation: [Float]
    public var criticLayers: [MetalRoboPolicyDenseLayer]
    public var actionLogStandardDeviation: [Float]
    public var actionBias: [Float]
    public var actionScale: [Float]
    public var observationClip: Float
    public var actionClip: Float

    public init(
        id: String,
        revision: UInt64,
        observationMean: [Float] = [],
        observationInverseStandardDeviation: [Float] = [],
        layers: [MetalRoboPolicyDenseLayer],
        criticObservationMean: [Float] = [],
        criticObservationInverseStandardDeviation: [Float] = [],
        criticLayers: [MetalRoboPolicyDenseLayer] = [],
        actionLogStandardDeviation: [Float] = [],
        actionBias: [Float] = [],
        actionScale: [Float] = [],
        observationClip: Float = 100,
        actionClip: Float = Float.greatestFiniteMagnitude
    ) {
        self.id = id
        self.revision = revision
        self.observationMean = observationMean
        self.observationInverseStandardDeviation =
            observationInverseStandardDeviation
        self.layers = layers
        self.criticObservationMean = criticObservationMean
        self.criticObservationInverseStandardDeviation =
            criticObservationInverseStandardDeviation
        self.criticLayers = criticLayers
        self.actionLogStandardDeviation =
            actionLogStandardDeviation
        self.actionBias = actionBias
        self.actionScale = actionScale
        self.observationClip = observationClip
        self.actionClip = actionClip
    }

    /// Persists the canonical native PolicyPack consumed by Metal rollout and
    /// deployment. The write is validated and transactional in native code.
    public func write(to url: URL) throws {
        let status = try withNativePolicyPack(self) { native in
            url.path.withCString { path in
                mr_write_policy_pack(native, path)
            }
        }
        guard status == 0 else {
            throw MetalRoboSimulationError.native(
                String(cString: mr_last_error())
            )
        }
    }
}

private func withNativePolicyPack<Result>(
    _ policy: MetalRoboPolicyPack,
    _ body: (UnsafePointer<MRPolicyPackC>) -> Result
) throws -> Result {
    let validLayer: (MetalRoboPolicyDenseLayer) -> Bool = { layer in
        let product = layer.inputCount.multipliedReportingOverflow(
            by: layer.outputCount
        )
        return layer.inputCount > 0 &&
            layer.inputCount <= Int(UInt32.max) &&
            layer.outputCount > 0 &&
            layer.outputCount <= Int(UInt32.max) &&
            !product.overflow &&
            layer.weights.count == product.partialValue &&
            layer.bias.count == layer.outputCount
    }
    guard !policy.id.isEmpty,
          policy.revision != 0,
          !policy.layers.isEmpty,
          policy.layers.allSatisfy(validLayer),
          policy.criticLayers.allSatisfy(validLayer),
          policy.actionLogStandardDeviation.isEmpty ||
              !policy.criticLayers.isEmpty
    else {
        throw MetalRoboSimulationError.invalidShape(
            "Policy identity, revision, or dense layer shape is invalid."
        )
    }

    var allocations: [
        (pointer: UnsafeMutablePointer<Float>, count: Int)
    ] = []
    func copy(_ values: [Float]) -> UnsafePointer<Float>? {
        guard !values.isEmpty else {
            return nil
        }
        let pointer = UnsafeMutablePointer<Float>.allocate(
            capacity: values.count
        )
        values.withUnsafeBufferPointer { source in
            pointer.initialize(
                from: source.baseAddress!,
                count: values.count
            )
        }
        allocations.append((pointer, values.count))
        return UnsafePointer(pointer)
    }
    defer {
        for allocation in allocations {
            allocation.pointer.deinitialize(count: allocation.count)
            allocation.pointer.deallocate()
        }
    }

    let nativeLayers = policy.layers.map { layer in
        var native = MRPolicyDenseLayerC()
        native.input_count = UInt32(layer.inputCount)
        native.output_count = UInt32(layer.outputCount)
        native.activation = layer.activation.rawValue
        native.weights = copy(layer.weights)
        native.weight_count = layer.weights.count
        native.bias = copy(layer.bias)
        native.bias_count = layer.bias.count
        return native
    }
    let nativeCriticLayers = policy.criticLayers.map { layer in
        var native = MRPolicyDenseLayerC()
        native.input_count = UInt32(layer.inputCount)
        native.output_count = UInt32(layer.outputCount)
        native.activation = layer.activation.rawValue
        native.weights = copy(layer.weights)
        native.weight_count = layer.weights.count
        native.bias = copy(layer.bias)
        native.bias_count = layer.bias.count
        return native
    }
    var native = MRPolicyPackC()
    native.revision = policy.revision
    native.observation_mean = copy(policy.observationMean)
    native.observation_mean_count = policy.observationMean.count
    native.observation_inverse_standard_deviation = copy(
        policy.observationInverseStandardDeviation
    )
    native.observation_inverse_standard_deviation_count =
        policy.observationInverseStandardDeviation.count
    native.critic_observation_mean = copy(
        policy.criticObservationMean
    )
    native.critic_observation_mean_count =
        policy.criticObservationMean.count
    native.critic_observation_inverse_standard_deviation = copy(
        policy.criticObservationInverseStandardDeviation
    )
    native.critic_observation_inverse_standard_deviation_count =
        policy.criticObservationInverseStandardDeviation.count
    native.action_log_standard_deviation = copy(
        policy.actionLogStandardDeviation
    )
    native.action_log_standard_deviation_count =
        policy.actionLogStandardDeviation.count
    native.action_bias = copy(policy.actionBias)
    native.action_bias_count = policy.actionBias.count
    native.action_scale = copy(policy.actionScale)
    native.action_scale_count = policy.actionScale.count
    native.observation_clip = policy.observationClip
    native.action_clip = policy.actionClip

    return policy.id.withCString { identifier in
        native.id = identifier
        return nativeLayers.withUnsafeBufferPointer { layers in
            native.layers = layers.baseAddress
            native.layer_count = layers.count
            return nativeCriticLayers.withUnsafeBufferPointer {
                criticLayers in
                native.critic_layers = criticLayers.baseAddress
                native.critic_layer_count = criticLayers.count
                return withUnsafePointer(to: &native, body)
            }
        }
    }
}

public struct MetalRoboSimulationConfiguration: Sendable {
    public var environmentCount: UInt32
    public var numiSolver: MetalRoboNumiSolverSettings
    public var physicsSubsteps: UInt32
    public var controlTimestepSeconds: Float
    public var seed: UInt64

    public init(
        environmentCount: UInt32,
        numiSolver: MetalRoboNumiSolverSettings = .init(),
        physicsSubsteps: UInt32 = 1,
        controlTimestepSeconds: Float = 0.02,
        seed: UInt64 = 0
    ) {
        self.environmentCount = environmentCount
        self.numiSolver = numiSolver
        self.physicsSubsteps = physicsSubsteps
        self.controlTimestepSeconds = controlTimestepSeconds
        self.seed = seed
    }
}

public struct MetalRoboSimulationLayout: Sendable {
    public let environmentCount: Int
    public let configurationCount: Int
    public let velocityCount: Int
    public let actionCount: Int
    public let actorObservationCount: Int
    public let criticObservationCount: Int
    public let sceneBodyCount: Int
    public let submittedControlSteps: UInt64
    public let completedEnvironmentSteps: UInt64
    public let submissionCount: UInt64
    public let retainedBufferBytes: Int
    public let immutablePrivateBytes: Int
    public let persistentStatePrivateBytes: Int
    public let transientPrivateBytes: Int
    public let sharedBoundaryBytes: Int
    public let peakAliasedBytes: Int
    public let totalGPUMilliseconds: Double
    public let totalSubmissionMilliseconds: Double

    init(_ native: MRSimulationLayoutC) {
        environmentCount = Int(native.environment_count)
        configurationCount = Int(native.nq)
        velocityCount = Int(native.nv)
        actionCount = Int(native.action_count)
        actorObservationCount =
            Int(native.actor_observation_count)
        criticObservationCount =
            Int(native.critic_observation_count)
        sceneBodyCount = Int(native.scene_body_count)
        submittedControlSteps = native.submitted_control_steps
        completedEnvironmentSteps =
            native.completed_environment_steps
        submissionCount = native.submission_count
        retainedBufferBytes = native.retained_buffer_bytes
        immutablePrivateBytes = native.immutable_private_bytes
        persistentStatePrivateBytes =
            native.persistent_state_private_bytes
        transientPrivateBytes = native.transient_private_bytes
        sharedBoundaryBytes = native.shared_boundary_bytes
        peakAliasedBytes = native.peak_aliased_bytes
        totalGPUMilliseconds = native.total_gpu_milliseconds
        totalSubmissionMilliseconds =
            native.total_submission_milliseconds
    }
}

public struct MetalRoboSimulationAdvance: Sendable {
    public let controlStepCount: Int
    public let successfulEnvironmentSteps: Int
    public let failedEnvironmentSteps: Int
    public let firstFailingEnvironment: UInt32
    public let firstFailingControlStep: UInt32
    public let firstGPUStatusCode: UInt32
    public let hostRequestedResets: Int
    public let maximumActiveContacts: Int
    public let maximumManifolds: Int
    public let stageHighWater: [String: Int]
    public let gpuMilliseconds: Double
    public let submissionMilliseconds: Double

    init(_ native: MRSimulationAdvanceC) {
        controlStepCount = Int(native.control_step_count)
        successfulEnvironmentSteps =
            Int(native.successful_environment_steps)
        failedEnvironmentSteps =
            Int(native.failed_environment_steps)
        firstFailingEnvironment =
            native.first_failing_environment
        firstFailingControlStep =
            native.first_failing_control_step
        firstGPUStatusCode = native.first_gpu_status_code
        hostRequestedResets =
            Int(native.host_requested_resets)
        maximumActiveContacts =
            Int(native.maximum_active_contacts)
        maximumManifolds = Int(native.maximum_manifolds)
        let highWater = native.high_water
        stageHighWater = [
            "candidate_pairs": Int(highWater.candidate_pairs),
            "raw_contacts": Int(highWater.raw_contacts),
            "manifolds": Int(highWater.manifolds),
            "constraint_blocks":
                Int(highWater.constraint_blocks),
            "constraint_rows": Int(highWater.constraint_rows),
            "islands": Int(highWater.islands),
            "hard_convex_pairs":
                Int(highWater.hard_convex_pairs),
            "mesh_triangle_candidates":
                Int(highWater.mesh_triangle_candidates),
            "ccd_candidates": Int(highWater.ccd_candidates),
            "ccd_events": Int(highWater.ccd_events),
            "endpoint_runtime_records":
                Int(highWater.endpoint_runtime_records),
            "articulation_point_queries":
                Int(highWater.articulation_point_queries),
            "rod_candidate_pairs":
                Int(highWater.rod_candidate_pairs),
            "rod_raw_contacts":
                Int(highWater.rod_raw_contacts),
            "rod_manifolds": Int(highWater.rod_manifolds),
            "rod_ccd_events": Int(highWater.rod_ccd_events),
            "numi_generalized_velocities":
                Int(highWater.numi_generalized_velocities),
            "numi_rows": Int(highWater.numi_rows),
            "numi_krylov_vectors":
                Int(highWater.numi_krylov_vectors),
            "numi_direct_tiles":
                Int(highWater.numi_direct_tiles),
            "dynamic_nodes": Int(highWater.dynamic_nodes),
            "island_node_references":
                Int(highWater.island_node_references),
            "island_constraint_references":
                Int(highWater.island_constraint_references),
            "rod_factor_blocks":
                Int(highWater.rod_factor_blocks),
            "operator_velocity_elements":
                Int(highWater.operator_velocity_elements),
        ]
        gpuMilliseconds = native.gpu_milliseconds
        submissionMilliseconds = native.submission_milliseconds
    }
}

public struct MetalRoboTaskTransition: Sendable {
    public let reward: Float
    public let trackingScore: Float
    public let rootHeight: Float
    public let tilt: Float
    public let done: Bool
    public let timeout: Bool
    public let physicsError: Bool
    public let terminationReason: UInt32
    public let taskReward: Float
    public let baseReward: Float
    public let jointVelocityReward: Float
    public let jointAccelerationReward: Float
    public let controlReward: Float
    public let postureReward: Float
    public let energyReward: Float
    public let contactReward: Float
    public let policyRevision: UInt64
    public let timeoutBootstrapValue: Float
    public let episodeTrackingScore: Float
    public let curriculumLevel: UInt32
    public let terrainLevel: UInt32

    init(_ native: MRTaskTransitionC) {
        reward = native.reward
        trackingScore = native.tracking_score
        rootHeight = native.root_height
        tilt = native.tilt
        done = native.done != 0
        timeout = native.timeout != 0
        physicsError = native.physics_error != 0
        terminationReason = native.termination_reason
        taskReward = native.task_reward
        baseReward = native.base_reward
        jointVelocityReward = native.joint_velocity_reward
        jointAccelerationReward =
            native.joint_acceleration_reward
        controlReward = native.control_reward
        postureReward = native.posture_reward
        energyReward = native.energy_reward
        contactReward = native.contact_reward
        policyRevision = native.policy_revision
        timeoutBootstrapValue =
            native.timeout_bootstrap_value
        episodeTrackingScore =
            native.episode_tracking_score
        curriculumLevel = native.curriculum_level
        terrainLevel = native.terrain_level
    }
}

public struct MetalRoboPolicyRolloutBatch: Sendable {
    public let controlStepCount: Int
    public let environmentCount: Int
    public let actorObservationCount: Int
    public let criticObservationCount: Int
    public let actionCount: Int
    public let actorObservations: [Float]
    public let criticObservations: [Float]
    public let latents: [Float]
    public let logProbabilities: [Float]
    public let values: [Float]
    public let transitions: [MetalRoboTaskTransition]

    public var sampleCount: Int {
        controlStepCount * environmentCount
    }

    public static func concatenating(
        _ batches: [Self]
    ) throws -> Self {
        guard let first = batches.first else {
            throw MetalRoboSimulationError.invalidShape(
                "At least one policy rollout batch is required."
            )
        }
        guard batches.allSatisfy({
            $0.environmentCount == first.environmentCount &&
                $0.actorObservationCount ==
                    first.actorObservationCount &&
                $0.criticObservationCount ==
                    first.criticObservationCount &&
                $0.actionCount == first.actionCount
        }) else {
            throw MetalRoboSimulationError.invalidShape(
                "Policy rollout batches have incompatible dimensions."
            )
        }
        var totalSteps = 0
        for batch in batches {
            let sum = totalSteps.addingReportingOverflow(
                batch.controlStepCount
            )
            guard batch.controlStepCount > 0,
                  !sum.overflow,
                  sum.partialValue <= Int(UInt32.max)
            else {
                throw MetalRoboSimulationError.invalidShape(
                    "Policy rollout control-step count exceeds the artifact boundary."
                )
            }
            totalSteps = sum.partialValue
        }
        return Self(
            controlStepCount: totalSteps,
            environmentCount: first.environmentCount,
            actorObservationCount:
                first.actorObservationCount,
            criticObservationCount:
                first.criticObservationCount,
            actionCount: first.actionCount,
            actorObservations:
                batches.flatMap(\.actorObservations),
            criticObservations:
                batches.flatMap(\.criticObservations),
            latents: batches.flatMap(\.latents),
            logProbabilities:
                batches.flatMap(\.logProbabilities),
            values: batches.flatMap(\.values),
            transitions: batches.flatMap(\.transitions)
        )
    }
}

/// A fixed-capacity Apple-Silicon rollout boundary. Each slot is backed by
/// separate `storageModeShared` Metal buffers so MLX can wrap every tensor at
/// its page-aligned base address without copying. A slot is returned to the
/// ring only after its lifetime-scoped view (and every managed MLX view that
/// retained it) has been released.
public final class MetalRoboRolloutBufferRing: @unchecked Sendable {
    private enum SlotState {
        case available
        case collecting
        case sealed
    }

    private final class Slot {
        let actorObservations: any MTLBuffer
        let criticObservations: any MTLBuffer
        let latents: any MTLBuffer
        let logProbabilities: any MTLBuffer
        let values: any MTLBuffer
        let bootstrapValues: any MTLBuffer
        let transitions: any MTLBuffer
        var state = SlotState.available
        var generation: UInt64 = 0
        var writtenControlSteps = 0
        var bootstrapWritten = false
        var policyRevision: UInt64 = 0

        init(
            device: any MTLDevice,
            actorObservationBytes: Int,
            criticObservationBytes: Int,
            latentBytes: Int,
            scalarBytes: Int,
            bootstrapBytes: Int,
            transitionBytes: Int
        ) throws {
            func allocate(
                _ bytes: Int,
                label: String
            ) throws -> any MTLBuffer {
                guard bytes > 0,
                      let buffer = device.makeBuffer(
                          length: bytes,
                          options: .storageModeShared
                      )
                else {
                    throw MetalRoboSimulationError.native(
                        "Could not allocate shared rollout buffer: \(label)."
                    )
                }
                buffer.label = label
                return buffer
            }
            actorObservations = try allocate(
                actorObservationBytes,
                label: "MetalRobo rollout actor observations"
            )
            criticObservations = try allocate(
                criticObservationBytes,
                label: "MetalRobo rollout critic observations"
            )
            latents = try allocate(
                latentBytes,
                label: "MetalRobo rollout policy latents"
            )
            logProbabilities = try allocate(
                scalarBytes,
                label: "MetalRobo rollout log probabilities"
            )
            values = try allocate(
                scalarBytes,
                label: "MetalRobo rollout values"
            )
            bootstrapValues = try allocate(
                bootstrapBytes,
                label: "MetalRobo rollout bootstrap values"
            )
            transitions = try allocate(
                transitionBytes,
                label: "MetalRobo rollout transitions"
            )
        }
    }

    public let controlStepCapacity: Int
    public let environmentCount: Int
    public let actorObservationCount: Int
    public let criticObservationCount: Int
    public let actionCount: Int
    public let retainedBytes: Int

    private let lock = NSLock()
    private let slots: [Slot]
    private var nextSlot = 0

    public init(
        layout: MetalRoboSimulationLayout,
        controlStepCapacity: Int,
        slotCount: Int = 3,
        device: (any MTLDevice)? = MTLCreateSystemDefaultDevice()
    ) throws {
        guard controlStepCapacity > 0,
              controlStepCapacity <= Int(UInt32.max),
              (1...8).contains(slotCount),
              layout.environmentCount > 0,
              layout.actorObservationCount > 0,
              layout.criticObservationCount > 0,
              layout.actionCount > 0,
              let device
        else {
            throw MetalRoboSimulationError.invalidShape(
                "A rollout ring requires a Metal device, 1-8 slots, and positive compiled dimensions."
            )
        }

        func product(_ values: [Int], label: String) throws -> Int {
            var result = 1
            for value in values {
                let multiplication = result.multipliedReportingOverflow(
                    by: value
                )
                guard value > 0,
                      !multiplication.overflow
                else {
                    throw MetalRoboSimulationError.invalidShape(
                        "\(label) exceeds the native buffer boundary."
                    )
                }
                result = multiplication.partialValue
            }
            return result
        }

        let samples = try product(
            [controlStepCapacity, layout.environmentCount],
            label: "Rollout sample count"
        )
        let floatBytes = MemoryLayout<Float>.stride
        let actorBytes = try product(
            [samples, layout.actorObservationCount, floatBytes],
            label: "Actor rollout buffer"
        )
        let criticBytes = try product(
            [samples, layout.criticObservationCount, floatBytes],
            label: "Critic rollout buffer"
        )
        let latentBytes = try product(
            [samples, layout.actionCount, floatBytes],
            label: "Latent rollout buffer"
        )
        let scalarBytes = try product(
            [samples, floatBytes],
            label: "Scalar rollout buffer"
        )
        let bootstrapBytes = try product(
            [layout.environmentCount, floatBytes],
            label: "Bootstrap rollout buffer"
        )
        let transitionBytes = try product(
            [samples, MemoryLayout<MRTaskTransitionC>.stride],
            label: "Transition rollout buffer"
        )
        var bytesPerSlot = 0
        for bytes in [
            actorBytes,
            criticBytes,
            latentBytes,
            scalarBytes,
            scalarBytes,
            bootstrapBytes,
            transitionBytes,
        ] {
            let addition = bytesPerSlot.addingReportingOverflow(bytes)
            guard !addition.overflow else {
                throw MetalRoboSimulationError.invalidShape(
                    "Rollout slot retained bytes overflow Int."
                )
            }
            bytesPerSlot = addition.partialValue
        }
        let retained = bytesPerSlot.multipliedReportingOverflow(
            by: slotCount
        )
        guard !retained.overflow else {
            throw MetalRoboSimulationError.invalidShape(
                "Rollout ring retained bytes overflow Int."
            )
        }

        self.controlStepCapacity = controlStepCapacity
        environmentCount = layout.environmentCount
        actorObservationCount = layout.actorObservationCount
        criticObservationCount = layout.criticObservationCount
        actionCount = layout.actionCount
        retainedBytes = retained.partialValue
        var allocated: [Slot] = []
        allocated.reserveCapacity(slotCount)
        for _ in 0..<slotCount {
            allocated.append(
                try Slot(
                    device: device,
                    actorObservationBytes: actorBytes,
                    criticObservationBytes: criticBytes,
                    latentBytes: latentBytes,
                    scalarBytes: scalarBytes,
                    bootstrapBytes: bootstrapBytes,
                    transitionBytes: transitionBytes
                )
            )
        }
        slots = allocated
    }

    public func acquire(
        policyRevision: UInt64
    ) throws -> MetalRoboRolloutBufferView {
        lock.lock()
        defer { lock.unlock() }
        for offset in slots.indices {
            let index = (nextSlot + offset) % slots.count
            let slot = slots[index]
            guard slot.state == .available else {
                continue
            }
            slot.state = .collecting
            slot.generation &+= 1
            slot.writtenControlSteps = 0
            slot.bootstrapWritten = false
            slot.policyRevision = policyRevision
            nextSlot = (index + 1) % slots.count
            return MetalRoboRolloutBufferView(
                ring: self,
                slotIndex: index,
                generation: slot.generation,
                policyRevision: policyRevision
            )
        }
        throw MetalRoboSimulationError.native(
            "All rollout ring slots are leased by the learner."
        )
    }

    fileprivate func append(
        slotIndex: Int,
        generation: UInt64,
        controlStepCount: Int,
        actor: UnsafePointer<Float>,
        critic: UnsafePointer<Float>,
        latents: UnsafePointer<Float>,
        logProbabilities: UnsafePointer<Float>,
        values: UnsafePointer<Float>,
        transitions: UnsafePointer<MRTaskTransitionC>,
        bootstrap: UnsafePointer<Float>?
    ) throws {
        lock.lock()
        defer { lock.unlock() }
        guard slots.indices.contains(slotIndex) else {
            throw MetalRoboSimulationError.native(
                "Rollout lease refers to an invalid ring slot."
            )
        }
        let slot = slots[slotIndex]
        guard slot.state == .collecting,
              slot.generation == generation,
              controlStepCount > 0,
              slot.writtenControlSteps + controlStepCount <=
                  controlStepCapacity
        else {
            throw MetalRoboSimulationError.native(
                "Rollout append violates its slot lease or capacity."
            )
        }
        if bootstrap != nil {
            guard slot.writtenControlSteps + controlStepCount ==
                    controlStepCapacity,
                  !slot.bootstrapWritten
            else {
                throw MetalRoboSimulationError.native(
                    "Bootstrap values may be published only at the completed rollout boundary."
                )
            }
        }
        let sampleProduct = controlStepCount.multipliedReportingOverflow(
            by: environmentCount
        )
        guard !sampleProduct.overflow else {
            throw MetalRoboSimulationError.invalidShape(
                "Rollout chunk sample count overflows Int."
            )
        }
        let chunkSamples = sampleProduct.partialValue
        for index in 0..<chunkSamples where
            transitions[index].policy_revision != slot.policyRevision
        {
            throw MetalRoboSimulationError.native(
                "Policy revision changed inside a native rollout chunk."
            )
        }
        let destinationSample =
            slot.writtenControlSteps * environmentCount
        func copyFloats(
            _ source: UnsafePointer<Float>,
            to buffer: any MTLBuffer,
            destinationElement: Int,
            count: Int
        ) {
            buffer.contents()
                .advanced(
                    by: destinationElement *
                        MemoryLayout<Float>.stride
                )
                .copyMemory(
                    from: UnsafeRawPointer(source),
                    byteCount: count * MemoryLayout<Float>.stride
                )
        }
        copyFloats(
            actor,
            to: slot.actorObservations,
            destinationElement:
                destinationSample * actorObservationCount,
            count: chunkSamples * actorObservationCount
        )
        copyFloats(
            critic,
            to: slot.criticObservations,
            destinationElement:
                destinationSample * criticObservationCount,
            count: chunkSamples * criticObservationCount
        )
        copyFloats(
            latents,
            to: slot.latents,
            destinationElement: destinationSample * actionCount,
            count: chunkSamples * actionCount
        )
        copyFloats(
            logProbabilities,
            to: slot.logProbabilities,
            destinationElement: destinationSample,
            count: chunkSamples
        )
        copyFloats(
            values,
            to: slot.values,
            destinationElement: destinationSample,
            count: chunkSamples
        )
        slot.transitions.contents()
            .advanced(
                by: destinationSample *
                    MemoryLayout<MRTaskTransitionC>.stride
            )
            .copyMemory(
                from: UnsafeRawPointer(transitions),
                byteCount: chunkSamples *
                    MemoryLayout<MRTaskTransitionC>.stride
            )
        slot.writtenControlSteps += controlStepCount
        if let bootstrap {
            copyFloats(
                bootstrap,
                to: slot.bootstrapValues,
                destinationElement: 0,
                count: environmentCount
            )
            slot.bootstrapWritten = true
        }
    }

    fileprivate func seal(
        slotIndex: Int,
        generation: UInt64
    ) throws {
        lock.lock()
        defer { lock.unlock() }
        guard slots.indices.contains(slotIndex) else {
            throw MetalRoboSimulationError.native(
                "Rollout lease refers to an invalid ring slot."
            )
        }
        let slot = slots[slotIndex]
        guard slot.state == .collecting,
              slot.generation == generation,
              slot.writtenControlSteps == controlStepCapacity,
              slot.bootstrapWritten
        else {
            throw MetalRoboSimulationError.native(
                "A rollout slot cannot be sealed before all steps and bootstrap values are present."
            )
        }
        slot.state = .sealed
    }

    fileprivate func release(
        slotIndex: Int,
        generation: UInt64
    ) {
        lock.lock()
        defer { lock.unlock() }
        guard slots.indices.contains(slotIndex) else {
            return
        }
        let slot = slots[slotIndex]
        guard slot.generation == generation,
              slot.state != .available
        else {
            return
        }
        slot.state = .available
        slot.writtenControlSteps = 0
        slot.bootstrapWritten = false
        slot.policyRevision = 0
    }

    fileprivate func pointers(
        slotIndex: Int,
        generation: UInt64
    ) throws -> (
        actor: UnsafeMutableRawPointer,
        critic: UnsafeMutableRawPointer,
        latents: UnsafeMutableRawPointer,
        logProbabilities: UnsafeMutableRawPointer,
        values: UnsafeMutableRawPointer,
        bootstrap: UnsafeMutableRawPointer,
        transitions: UnsafeMutableRawPointer
    ) {
        lock.lock()
        defer { lock.unlock() }
        guard slots.indices.contains(slotIndex) else {
            throw MetalRoboSimulationError.native(
                "Rollout lease refers to an invalid ring slot."
            )
        }
        let slot = slots[slotIndex]
        guard slot.state == .sealed,
              slot.generation == generation
        else {
            throw MetalRoboSimulationError.native(
                "Rollout buffers are visible only through a sealed live lease."
            )
        }
        return (
            slot.actorObservations.contents(),
            slot.criticObservations.contents(),
            slot.latents.contents(),
            slot.logProbabilities.contents(),
            slot.values.contents(),
            slot.bootstrapValues.contents(),
            slot.transitions.contents()
        )
    }
}

/// Lifetime-scoped view of one complete rollout slot. Do not retain raw
/// pointers beyond a `with...` closure. MLX integration retains this object in
/// each managed-array finalizer, which prevents slot reuse while lazy work is
/// still outstanding.
public final class MetalRoboRolloutBufferView: @unchecked Sendable {
    fileprivate let ring: MetalRoboRolloutBufferRing
    fileprivate let slotIndex: Int
    fileprivate let generation: UInt64
    public let policyRevision: UInt64

    fileprivate init(
        ring: MetalRoboRolloutBufferRing,
        slotIndex: Int,
        generation: UInt64,
        policyRevision: UInt64
    ) {
        self.ring = ring
        self.slotIndex = slotIndex
        self.generation = generation
        self.policyRevision = policyRevision
    }

    deinit {
        ring.release(
            slotIndex: slotIndex,
            generation: generation
        )
    }

    public var controlStepCount: Int {
        ring.controlStepCapacity
    }

    public var environmentCount: Int {
        ring.environmentCount
    }

    public var actorObservationCount: Int {
        ring.actorObservationCount
    }

    public var criticObservationCount: Int {
        ring.criticObservationCount
    }

    public var actionCount: Int {
        ring.actionCount
    }

    public var sampleCount: Int {
        controlStepCount * environmentCount
    }

    fileprivate func seal() throws {
        try ring.seal(
            slotIndex: slotIndex,
            generation: generation
        )
    }

    func rawPointers() throws -> (
        actor: UnsafeMutableRawPointer,
        critic: UnsafeMutableRawPointer,
        latents: UnsafeMutableRawPointer,
        logProbabilities: UnsafeMutableRawPointer,
        values: UnsafeMutableRawPointer,
        bootstrap: UnsafeMutableRawPointer,
        transitions: UnsafeMutableRawPointer
    ) {
        try ring.pointers(
            slotIndex: slotIndex,
            generation: generation
        )
    }

    public func transition(
        at index: Int
    ) throws -> MetalRoboTaskTransition {
        guard (0..<sampleCount).contains(index) else {
            throw MetalRoboSimulationError.invalidShape(
                "Transition index is outside the rollout lease."
            )
        }
        let pointers = try rawPointers()
        return MetalRoboTaskTransition(
            pointers.transitions
                .assumingMemoryBound(to: MRTaskTransitionC.self)[index]
        )
    }

    public func value(at index: Int) throws -> Float {
        guard (0..<sampleCount).contains(index) else {
            throw MetalRoboSimulationError.invalidShape(
                "Value index is outside the rollout lease."
            )
        }
        return try rawPointers().values
            .assumingMemoryBound(to: Float.self)[index]
    }

    public func bootstrapValue(at environment: Int) throws -> Float {
        guard (0..<environmentCount).contains(environment) else {
            throw MetalRoboSimulationError.invalidShape(
                "Bootstrap index is outside the rollout lease."
            )
        }
        return try rawPointers().bootstrap
            .assumingMemoryBound(to: Float.self)[environment]
    }
}

/// Swift owns rollout chunking and policy revisions. Native code owns the
/// compiled world and task program, persistent private Metal state, reset
/// transaction, and compact learning publication.
public final class MetalSimulationSession {
    private var handle: OpaquePointer?

    /// Bundled G1 is a mechanics + TaskPack preset; execution uses the same
    /// generic compiled task path as imported robots.
    public init(
        unitreeG1 configuration: MetalRoboSimulationConfiguration,
        actuatorPreset: MetalRoboG1ActuatorPreset =
            .unitreeRLLab4960b84,
        surface: MetalRoboBuiltinSurface = .terrain,
        metallibPath: String? = nil
    ) throws {
        var native = Self.nativeConfiguration(configuration)

        let created: OpaquePointer? = withUnsafePointer(to: &native) {
            config in
            withOptionalCString(metallibPath) {
                metallib in
                mr_create_unitree_g1_simulation(
                    config,
                    actuatorPreset.rawValue,
                    surface.rawValue,
                    metallib
                )
            }
        }
        guard let created else {
            throw MetalRoboSimulationError.native(
                Self.lastError()
            )
        }
        handle = created
    }

    /// Imports mechanics from URDF/SRDF and task semantics from a persisted
    /// TaskPack. No robot-specific runtime or Metal shader is involved.
    public init(
        importedURDF urdfURL: URL,
        srdf srdfURL: URL? = nil,
        taskPack taskPackURL: URL,
        configuration: MetalRoboSimulationConfiguration,
        rootMode: MetalRoboRobotRootMode = .floating,
        surface: MetalRoboBuiltinSurface = .terrain,
        metallibPath: String? = nil
    ) throws {
        var native = Self.nativeConfiguration(configuration)
        let created: OpaquePointer? =
            urdfURL.path.withCString { urdf in
                taskPackURL.path.withCString { taskPack in
                    withOptionalCString(srdfURL?.path) {
                        srdf in
                        withOptionalCString(metallibPath) {
                            metallib in
                            withUnsafePointer(to: &native) {
                                config in
                                mr_create_urdf_simulation(
                                    urdf,
                                    srdf,
                                    taskPack,
                                    config,
                                    rootMode.rawValue,
                                    surface.rawValue,
                                    metallib
                                )
                            }
                        }
                    }
                }
            }
        guard let created else {
            throw MetalRoboSimulationError.native(
                Self.lastError()
            )
        }
        handle = created
    }

    /// Loads complete mechanics and scene composition from MRWorldPack, then
    /// resolves a separate TaskPack through the generic native executor.
    public init(
        worldPack worldPackURL: URL,
        taskPack taskPackURL: URL,
        configuration: MetalRoboSimulationConfiguration,
        metallibPath: String? = nil
    ) throws {
        var native = Self.nativeConfiguration(configuration)
        let created: OpaquePointer? =
            worldPackURL.path.withCString { worldPack in
                taskPackURL.path.withCString { taskPack in
                    withOptionalCString(metallibPath) {
                        metallib in
                        withUnsafePointer(to: &native) {
                            config in
                            mr_create_world_pack_simulation(
                                worldPack,
                                taskPack,
                                config,
                                metallib
                            )
                        }
                    }
                }
            }
        guard let created else {
            throw MetalRoboSimulationError.native(
                Self.lastError()
            )
        }
        handle = created
    }

    private static func nativeConfiguration(
        _ configuration: MetalRoboSimulationConfiguration
    ) -> MRSimulationConfigC {
        var native = MRSimulationConfigC()
        native.environment_count = configuration.environmentCount
        native.numi_iteration_policy =
            configuration.numiSolver.iterationPolicy.rawValue
        native.physics_substeps = configuration.physicsSubsteps
        native.temporal_substeps =
            configuration.numiSolver.temporalSubsteps
        native.control_timestep_seconds =
            configuration.controlTimestepSeconds
        native.seed = configuration.seed
        return native
    }

    deinit {
        if let handle {
            mr_simulation_destroy(handle)
        }
    }

    public var layout: MetalRoboSimulationLayout {
        MetalRoboSimulationLayout(
            mr_simulation_layout(handle)
        )
    }

    public var deviceName: String {
        String(cString: mr_simulation_device_name(handle))
    }

    public var taskFingerprint: UInt64 {
        mr_simulation_task_fingerprint(handle)
    }

    public var policyFingerprint: UInt64 {
        mr_simulation_policy_fingerprint(handle)
    }

    public var policyTopologyFingerprint: UInt64 {
        mr_simulation_policy_topology_fingerprint(handle)
    }

    public var policyRevision: UInt64 {
        mr_simulation_policy_revision(handle)
    }

    public func reset(seed: UInt64) throws {
        guard mr_simulation_reset(handle, seed) == 0 else {
            throw MetalRoboSimulationError.native(
                Self.lastError()
            )
        }
    }

    public func setCurriculumLevel(_ level: UInt32) throws {
        guard mr_simulation_set_curriculum_level(
            handle,
            level
        ) == 0 else {
            throw MetalRoboSimulationError.native(
                Self.lastError()
            )
        }
    }

    public func setPolicy(
        _ policy: MetalRoboPolicyPack
    ) throws {
        let status = try withNativePolicyPack(policy) { native in
            mr_simulation_set_policy(handle, native)
        }
        guard status == 0 else {
            throw MetalRoboSimulationError.native(
                Self.lastError()
            )
        }
    }

    public func loadPolicy(at url: URL) throws {
        let status = url.path.withCString {
            mr_simulation_load_policy_pack(handle, $0)
        }
        guard status == 0 else {
            throw MetalRoboSimulationError.native(
                Self.lastError()
            )
        }
    }

    public func clearPolicy() throws {
        guard mr_simulation_clear_policy(handle) == 0 else {
            throw MetalRoboSimulationError.native(
                Self.lastError()
            )
        }
    }

    public func advance(
        normalizedActions: [Float],
        resetMasks: [UInt32] = [],
        controlStepCount: Int,
        policyRevision: UInt64 = 0
    ) throws -> MetalRoboSimulationAdvance {
        let current = layout
        guard controlStepCount > 0 else {
            throw MetalRoboSimulationError.invalidShape(
                "controlStepCount must be positive."
            )
        }
        let actionElements =
            controlStepCount *
            current.environmentCount *
            current.actionCount
        guard normalizedActions.count == actionElements else {
            throw MetalRoboSimulationError.invalidShape(
                "Actions must contain \(actionElements) values."
            )
        }
        let maskElements =
            controlStepCount * current.environmentCount
        guard resetMasks.isEmpty ||
                resetMasks.count == maskElements
        else {
            throw MetalRoboSimulationError.invalidShape(
                "Reset masks must be empty or contain " +
                    "\(maskElements) values."
            )
        }

        var native = MRSimulationAdvanceC()
        let status = normalizedActions.withUnsafeBufferPointer {
            actions in
            if resetMasks.isEmpty {
                return mr_simulation_advance(
                    handle,
                    actions.baseAddress,
                    actions.count,
                    nil,
                    0,
                    UInt32(controlStepCount),
                    policyRevision,
                    0,
                    &native
                )
            }
            return resetMasks.withUnsafeBufferPointer { masks in
                mr_simulation_advance(
                    handle,
                    actions.baseAddress,
                    actions.count,
                    masks.baseAddress,
                    masks.count,
                    UInt32(controlStepCount),
                    policyRevision,
                    0,
                    &native
                )
            }
        }
        guard status == 0 else {
            throw MetalRoboSimulationError.native(
                Self.lastError()
            )
        }
        return MetalRoboSimulationAdvance(native)
    }

    public func advanceWithPolicy(
        resetMasks: [UInt32] = [],
        controlStepCount: Int,
        policyRevision: UInt64 = 0,
        evaluateFinalPolicy: Bool = false
    ) throws -> MetalRoboSimulationAdvance {
        let current = layout
        guard controlStepCount > 0 else {
            throw MetalRoboSimulationError.invalidShape(
                "controlStepCount must be positive."
            )
        }
        let maskElements =
            controlStepCount * current.environmentCount
        guard resetMasks.isEmpty ||
                resetMasks.count == maskElements
        else {
            throw MetalRoboSimulationError.invalidShape(
                "Reset masks must be empty or contain " +
                    "\(maskElements) values."
            )
        }
        var native = MRSimulationAdvanceC()
        if resetMasks.isEmpty {
            let status = mr_simulation_advance(
                handle,
                nil,
                0,
                nil,
                0,
                UInt32(controlStepCount),
                policyRevision,
                evaluateFinalPolicy ? 1 : 0,
                &native
            )
            guard status == 0 else {
                throw MetalRoboSimulationError.native(
                    Self.lastError()
                )
            }
            return MetalRoboSimulationAdvance(native)
        }
        let status = resetMasks.withUnsafeBufferPointer {
            masks in
            mr_simulation_advance(
                handle,
                nil,
                0,
                masks.baseAddress,
                masks.count,
                UInt32(controlStepCount),
                policyRevision,
                evaluateFinalPolicy ? 1 : 0,
                &native
            )
        }
        guard status == 0 else {
            throw MetalRoboSimulationError.native(
                Self.lastError()
            )
        }
        return MetalRoboSimulationAdvance(native)
    }

    public func statusCodes(
        controlStepCount: Int
    ) throws -> [UInt32] {
        let count = controlStepCount * layout.environmentCount
        guard let values =
                mr_simulation_status_codes(handle)
        else {
            throw MetalRoboSimulationError.native(
                "Task status stream is unavailable."
            )
        }
        return Array(
            UnsafeBufferPointer(start: values, count: count)
        )
    }

    public func activeContacts(
        controlStepCount: Int
    ) throws -> [UInt32] {
        let count = controlStepCount * layout.environmentCount
        guard let values =
                mr_simulation_active_contacts(handle)
        else {
            throw MetalRoboSimulationError.native(
                "Task contact stream is unavailable."
            )
        }
        return Array(
            UnsafeBufferPointer(start: values, count: count)
        )
    }

    public func actorObservations(
        controlStepCount: Int
    ) throws -> [Float] {
        let current = layout
        let count =
            controlStepCount *
            current.environmentCount *
            current.actorObservationCount
        guard let values =
                mr_simulation_actor_observations(handle)
        else {
            throw MetalRoboSimulationError.native(
                "Actor observation stream is unavailable."
            )
        }
        return Array(
            UnsafeBufferPointer(start: values, count: count)
        )
    }

    public func criticObservations(
        controlStepCount: Int
    ) throws -> [Float] {
        let current = layout
        let count =
            controlStepCount *
            current.environmentCount *
            current.criticObservationCount
        guard let values =
                mr_simulation_critic_observations(handle)
        else {
            throw MetalRoboSimulationError.native(
                "Critic observation stream is unavailable."
            )
        }
        return Array(
            UnsafeBufferPointer(start: values, count: count)
        )
    }

    public func transitions(
        controlStepCount: Int
    ) throws -> [MetalRoboTaskTransition] {
        let count = controlStepCount * layout.environmentCount
        guard let values =
                mr_simulation_transitions(handle)
        else {
            throw MetalRoboSimulationError.native(
                "Task transition stream is unavailable."
            )
        }
        return UnsafeBufferPointer(start: values, count: count)
            .map(MetalRoboTaskTransition.init)
    }

    public func policyLatents(
        controlStepCount: Int
    ) throws -> [Float] {
        let current = layout
        let count =
            controlStepCount *
            current.environmentCount *
            current.actionCount
        guard let values =
                mr_simulation_policy_latents(handle)
        else {
            throw MetalRoboSimulationError.native(
                "Policy latent stream is unavailable."
            )
        }
        return Array(
            UnsafeBufferPointer(start: values, count: count)
        )
    }

    public func policyLogProbabilities(
        controlStepCount: Int
    ) throws -> [Float] {
        let count =
            controlStepCount * layout.environmentCount
        guard let values =
                mr_simulation_policy_log_probabilities(handle)
        else {
            throw MetalRoboSimulationError.native(
                "Policy log-probability stream is unavailable."
            )
        }
        return Array(
            UnsafeBufferPointer(start: values, count: count)
        )
    }

    public func policyValues(
        controlStepCount: Int
    ) throws -> [Float] {
        let count =
            controlStepCount * layout.environmentCount
        guard let values =
                mr_simulation_policy_values(handle)
        else {
            throw MetalRoboSimulationError.native(
                "Policy value stream is unavailable."
            )
        }
        return Array(
            UnsafeBufferPointer(start: values, count: count)
        )
    }

    public func bootstrapPolicyValues() throws -> [Float] {
        let count = layout.environmentCount
        guard let values =
                mr_simulation_bootstrap_policy_values(handle)
        else {
            throw MetalRoboSimulationError.native(
                "Post-rollout policy values are unavailable."
            )
        }
        return Array(
            UnsafeBufferPointer(start: values, count: count)
        )
    }

    public func policyRolloutBatch(
        controlStepCount: Int
    ) throws -> MetalRoboPolicyRolloutBatch {
        let current = layout
        guard controlStepCount > 0 else {
            throw MetalRoboSimulationError.invalidShape(
                "Policy rollout batch must contain at least one control step."
            )
        }
        return MetalRoboPolicyRolloutBatch(
            controlStepCount: controlStepCount,
            environmentCount: current.environmentCount,
            actorObservationCount:
                current.actorObservationCount,
            criticObservationCount:
                current.criticObservationCount,
            actionCount: current.actionCount,
            actorObservations: try actorObservations(
                controlStepCount: controlStepCount
            ),
            criticObservations: try criticObservations(
                controlStepCount: controlStepCount
            ),
            latents: try policyLatents(
                controlStepCount: controlStepCount
            ),
            logProbabilities:
                try policyLogProbabilities(
                    controlStepCount: controlStepCount
                ),
            values: try policyValues(
                controlStepCount: controlStepCount
            ),
            transitions: try transitions(
                controlStepCount: controlStepCount
            )
        )
    }

    /// Copies one completed native chunk into its leased shared Metal slot.
    /// The destination is fixed-capacity and environment-major; no Swift
    /// arrays are grown or concatenated. On the last chunk, the accepted
    /// post-rollout bootstrap values are captured into the same lease.
    public func appendCurrentPolicyRollout(
        controlStepCount: Int,
        to rollout: MetalRoboRolloutBufferView,
        includeBootstrapValues: Bool = false
    ) throws {
        let current = layout
        guard controlStepCount > 0,
              rollout.environmentCount == current.environmentCount,
              rollout.actorObservationCount ==
                  current.actorObservationCount,
              rollout.criticObservationCount ==
                  current.criticObservationCount,
              rollout.actionCount == current.actionCount
        else {
            throw MetalRoboSimulationError.invalidShape(
                "Policy rollout lease does not match the installed simulation."
            )
        }
        guard let actor =
                  mr_simulation_actor_observations(handle),
              let critic =
                  mr_simulation_critic_observations(handle),
              let latent =
                  mr_simulation_policy_latents(handle),
              let logProbability =
                  mr_simulation_policy_log_probabilities(handle),
              let value =
                  mr_simulation_policy_values(handle),
              let transition =
                  mr_simulation_transitions(handle),
              !includeBootstrapValues ||
                  mr_simulation_bootstrap_policy_values(handle) != nil
        else {
            throw MetalRoboSimulationError.native(
                "Native policy rollout streams are unavailable."
            )
        }
        try rollout.ring.append(
            slotIndex: rollout.slotIndex,
            generation: rollout.generation,
            controlStepCount: controlStepCount,
            actor: actor,
            critic: critic,
            latents: latent,
            logProbabilities: logProbability,
            values: value,
            transitions: transition,
            bootstrap: includeBootstrapValues
                ? mr_simulation_bootstrap_policy_values(handle)
                : nil
        )
    }

    public func sealPolicyRollout(
        _ rollout: MetalRoboRolloutBufferView
    ) throws {
        try rollout.seal()
    }

    /// Writes a sealed shared-buffer rollout directly through the native pack
    /// serializer. The lease remains live for the duration of the call and no
    /// transition or float table is materialized as a Swift array.
    public func writePolicyRolloutPack(
        _ rollout: MetalRoboRolloutBufferView,
        id: String,
        to url: URL
    ) throws {
        let current = layout
        guard !id.isEmpty,
              rollout.controlStepCount > 0,
              rollout.controlStepCount <= Int(UInt32.max),
              rollout.environmentCount == current.environmentCount,
              rollout.actorObservationCount ==
                  current.actorObservationCount,
              rollout.criticObservationCount ==
                  current.criticObservationCount,
              rollout.actionCount == current.actionCount
        else {
            throw MetalRoboSimulationError.invalidShape(
                "Shared rollout dimensions do not match the installed task."
            )
        }
        let pointers = try rollout.rawPointers()
        let samples = rollout.sampleCount
        var native = MRPolicyRolloutBatchC()
        native.control_step_count = UInt32(rollout.controlStepCount)
        native.actor_observations = UnsafePointer(
            pointers.actor.assumingMemoryBound(to: Float.self)
        )
        native.actor_observation_count =
            samples * rollout.actorObservationCount
        native.critic_observations = UnsafePointer(
            pointers.critic.assumingMemoryBound(to: Float.self)
        )
        native.critic_observation_count =
            samples * rollout.criticObservationCount
        native.latents = UnsafePointer(
            pointers.latents.assumingMemoryBound(to: Float.self)
        )
        native.latent_count = samples * rollout.actionCount
        native.log_probabilities = UnsafePointer(
            pointers.logProbabilities.assumingMemoryBound(to: Float.self)
        )
        native.log_probability_count = samples
        native.values = UnsafePointer(
            pointers.values.assumingMemoryBound(to: Float.self)
        )
        native.value_count = samples
        native.bootstrap_values = UnsafePointer(
            pointers.bootstrap.assumingMemoryBound(to: Float.self)
        )
        native.bootstrap_value_count = rollout.environmentCount
        native.transitions = UnsafePointer(
            pointers.transitions.assumingMemoryBound(
                to: MRTaskTransitionC.self
            )
        )
        native.transition_count = samples
        let status = id.withCString { identifier in
            url.path.withCString { path in
                mr_simulation_write_policy_rollout_pack(
                    handle,
                    &native,
                    identifier,
                    path
                )
            }
        }
        guard status == 0 else {
            throw MetalRoboSimulationError.native(
                Self.lastError()
            )
        }
    }

    public func writePolicyRolloutPack(
        _ batch: MetalRoboPolicyRolloutBatch,
        bootstrapValues: [Float],
        id: String,
        to url: URL
    ) throws {
        let current = layout
        guard !id.isEmpty,
              batch.controlStepCount > 0,
              batch.controlStepCount <= Int(UInt32.max),
              batch.environmentCount ==
                  current.environmentCount,
              batch.actorObservationCount ==
                  current.actorObservationCount,
              batch.criticObservationCount ==
                  current.criticObservationCount,
              batch.actionCount == current.actionCount,
              bootstrapValues.count ==
                  current.environmentCount,
              batch.actorObservations.count ==
                  batch.sampleCount *
                  batch.actorObservationCount,
              batch.criticObservations.count ==
                  batch.sampleCount *
                  batch.criticObservationCount,
              batch.latents.count ==
                  batch.sampleCount * batch.actionCount,
              batch.logProbabilities.count ==
                  batch.sampleCount,
              batch.values.count == batch.sampleCount,
              batch.transitions.count == batch.sampleCount
        else {
            throw MetalRoboSimulationError.invalidShape(
                "Policy rollout pack dimensions do not match the installed task."
            )
        }
        var native = MRPolicyRolloutBatchC()
        native.control_step_count =
            UInt32(batch.controlStepCount)
        native.actor_observation_count =
            batch.actorObservations.count
        native.critic_observation_count =
            batch.criticObservations.count
        native.latent_count = batch.latents.count
        native.log_probability_count =
            batch.logProbabilities.count
        native.value_count = batch.values.count
        native.bootstrap_value_count =
            bootstrapValues.count
        let nativeTransitions = batch.transitions.map {
            transition in
            var value = MRTaskTransitionC()
            value.reward = transition.reward
            value.tracking_score =
                transition.trackingScore
            value.root_height = transition.rootHeight
            value.tilt = transition.tilt
            value.done = transition.done ? 1 : 0
            value.timeout = transition.timeout ? 1 : 0
            value.physics_error =
                transition.physicsError ? 1 : 0
            value.termination_reason =
                transition.terminationReason
            value.task_reward = transition.taskReward
            value.base_reward = transition.baseReward
            value.joint_velocity_reward =
                transition.jointVelocityReward
            value.joint_acceleration_reward =
                transition.jointAccelerationReward
            value.control_reward = transition.controlReward
            value.posture_reward = transition.postureReward
            value.energy_reward = transition.energyReward
            value.contact_reward = transition.contactReward
            value.policy_revision =
                transition.policyRevision
            value.timeout_bootstrap_value =
                transition.timeoutBootstrapValue
            value.episode_tracking_score =
                transition.episodeTrackingScore
            value.curriculum_level =
                transition.curriculumLevel
            value.terrain_level = transition.terrainLevel
            return value
        }
        let status = withUnsafeFloatBuffers(
            [
                batch.actorObservations,
                batch.criticObservations,
                batch.latents,
                batch.logProbabilities,
                batch.values,
                bootstrapValues,
            ]
        ) { buffers in
            native.actor_observations = buffers[0].baseAddress
            native.critic_observations = buffers[1].baseAddress
            native.latents = buffers[2].baseAddress
            native.log_probabilities = buffers[3].baseAddress
            native.values = buffers[4].baseAddress
            native.bootstrap_values = buffers[5].baseAddress
            return nativeTransitions.withUnsafeBufferPointer {
                transitions in
                native.transitions = transitions.baseAddress
                native.transition_count = transitions.count
                return id.withCString { identifier in
                    url.path.withCString { path in
                        mr_simulation_write_policy_rollout_pack(
                            handle,
                            &native,
                            identifier,
                            path
                        )
                    }
                }
            }
        }
        guard status == 0 else {
            throw MetalRoboSimulationError.native(
                Self.lastError()
            )
        }
    }

    private static func lastError() -> String {
        guard let pointer = mr_last_error() else {
            return "MetalRobo simulation operation failed."
        }
        return String(cString: pointer)
    }
}
