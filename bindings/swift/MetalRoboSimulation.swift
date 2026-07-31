import Foundation
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

/// A fixed-capacity native rollout boundary. Metal allocates and owns every
/// shared stream; Swift receives only opaque lifetime-scoped leases. A slot
/// returns to the ring after the view and all managed MLX arrays retaining it
/// have been released.
public final class MetalRoboRolloutBufferRing: @unchecked Sendable {
    fileprivate var handle: OpaquePointer?

    public let controlStepCapacity: Int
    public let environmentCount: Int
    public let actorObservationCount: Int
    public let criticObservationCount: Int
    public let actionCount: Int
    public let slotCount: Int
    public let retainedBytes: Int

    fileprivate init(
        session: MetalSimulationSession,
        controlStepCapacity: Int,
        slotCount: Int
    ) throws {
        guard controlStepCapacity > 0,
              controlStepCapacity <= Int(UInt32.max),
              (1...8).contains(slotCount)
        else {
            throw MetalRoboSimulationError.invalidShape(
                "A rollout ring requires 1-8 slots and a positive UInt32 control-step capacity."
            )
        }
        guard let created = mr_rollout_ring_create(
            session.handle,
            UInt32(controlStepCapacity),
            UInt32(slotCount)
        ) else {
            throw MetalRoboSimulationError.native(
                MetalSimulationSession.lastError()
            )
        }
        let native = mr_rollout_ring_layout(created)
        let simulation = session.layout
        guard native.environment_count ==
                  UInt32(simulation.environmentCount),
              native.control_step_capacity ==
                  UInt32(controlStepCapacity),
              native.actor_observation_count ==
                  UInt32(simulation.actorObservationCount),
              native.critic_observation_count ==
                  UInt32(simulation.criticObservationCount),
              native.action_count ==
                  UInt32(simulation.actionCount),
              native.slot_count == UInt32(slotCount)
        else {
            mr_rollout_ring_destroy(created)
            throw MetalRoboSimulationError.native(
                "Native rollout ring layout does not match the compiled simulation."
            )
        }

        handle = created
        self.controlStepCapacity =
            Int(native.control_step_capacity)
        environmentCount = Int(native.environment_count)
        actorObservationCount =
            Int(native.actor_observation_count)
        criticObservationCount =
            Int(native.critic_observation_count)
        actionCount = Int(native.action_count)
        self.slotCount = Int(native.slot_count)
        retainedBytes = Int(native.retained_bytes)
    }

    deinit {
        if let handle {
            mr_rollout_ring_destroy(handle)
        }
    }

    public func acquire(
        policyRevision: UInt64
    ) throws -> MetalRoboRolloutBufferView {
        guard policyRevision > 0 else {
            throw MetalRoboSimulationError.invalidShape(
                "A rollout lease requires a positive policy revision."
            )
        }
        guard let acquired = mr_rollout_ring_acquire(
            handle,
            policyRevision
        ) else {
            throw MetalRoboSimulationError.native(
                MetalSimulationSession.lastError()
            )
        }
        return MetalRoboRolloutBufferView(
            ring: self,
            handle: acquired
        )
    }
}

/// Lifetime-scoped view of one complete rollout slot. Raw pointers are visible
/// only after the native transaction has filled and sealed the lease.
public final class MetalRoboRolloutBufferView: @unchecked Sendable {
    fileprivate let ring: MetalRoboRolloutBufferRing
    fileprivate var handle: OpaquePointer?
    public let policyRevision: UInt64

    fileprivate init(
        ring: MetalRoboRolloutBufferRing,
        handle: OpaquePointer
    ) {
        self.ring = ring
        self.handle = handle
        policyRevision =
            mr_rollout_buffer_view_policy_revision(handle)
    }

    deinit {
        if let handle {
            mr_rollout_buffer_view_destroy(handle)
        }
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

    public var writtenControlSteps: Int {
        Int(mr_rollout_buffer_view_written_steps(handle))
    }

    fileprivate func seal() throws {
        guard mr_rollout_buffer_view_seal(handle) == 0 else {
            throw MetalRoboSimulationError.native(
                MetalSimulationSession.lastError()
            )
        }
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
        guard let actor =
                  mr_rollout_buffer_view_actor_observations(handle),
              let critic =
                  mr_rollout_buffer_view_critic_observations(handle),
              let latents =
                  mr_rollout_buffer_view_latents(handle),
              let logProbabilities =
                  mr_rollout_buffer_view_log_probabilities(handle),
              let values =
                  mr_rollout_buffer_view_values(handle),
              let bootstrap =
                  mr_rollout_buffer_view_bootstrap_values(handle),
              let transitions =
                  mr_rollout_buffer_view_transitions(handle)
        else {
            throw MetalRoboSimulationError.native(
                "Rollout buffers are visible only through a sealed live lease."
            )
        }
        return (
            UnsafeMutableRawPointer(actor),
            UnsafeMutableRawPointer(critic),
            UnsafeMutableRawPointer(latents),
            UnsafeMutableRawPointer(logProbabilities),
            UnsafeMutableRawPointer(values),
            UnsafeMutableRawPointer(bootstrap),
            UnsafeMutableRawPointer(transitions)
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

    public func bootstrapValue(
        at environment: Int
    ) throws -> Float {
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
    fileprivate var handle: OpaquePointer?

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

    public func makePolicyRolloutRing(
        controlStepCapacity: Int,
        slotCount: Int = 3
    ) throws -> MetalRoboRolloutBufferRing {
        try MetalRoboRolloutBufferRing(
            session: self,
            controlStepCapacity: controlStepCapacity,
            slotCount: slotCount
        )
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

    /// Advances one native policy chunk and publishes its compact learning
    /// streams directly into the leased shared-buffer slot in the same Metal
    /// command buffer. The rollout cursor advances only when world-state
    /// publication succeeds.
    public func advancePolicyRollout(
        into rollout: MetalRoboRolloutBufferView,
        resetMasks: [UInt32] = [],
        controlStepCount: Int,
        policyRevision: UInt64,
        includeBootstrapValues: Bool = false
    ) throws -> MetalRoboSimulationAdvance {
        let current = layout
        guard controlStepCount > 0,
              controlStepCount <= Int(UInt32.max),
              policyRevision > 0,
              rollout.policyRevision == policyRevision,
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
        let maskCount = controlStepCount * current.environmentCount
        guard resetMasks.isEmpty || resetMasks.count == maskCount
        else {
            throw MetalRoboSimulationError.invalidShape(
                "Reset masks must be empty or contain \(maskCount) values."
            )
        }
        var native = MRSimulationAdvanceC()
        let status: Int32
        if resetMasks.isEmpty {
            status = mr_simulation_advance_policy_rollout(
                handle,
                rollout.handle,
                nil,
                0,
                UInt32(controlStepCount),
                policyRevision,
                includeBootstrapValues ? 1 : 0,
                &native
            )
        } else {
            status = resetMasks.withUnsafeBufferPointer { masks in
                mr_simulation_advance_policy_rollout(
                    handle,
                    rollout.handle,
                    masks.baseAddress,
                    masks.count,
                    UInt32(controlStepCount),
                    policyRevision,
                    includeBootstrapValues ? 1 : 0,
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

    fileprivate static func lastError() -> String {
        guard let pointer = mr_last_error() else {
            return "MetalRobo simulation operation failed."
        }
        return String(cString: pointer)
    }
}
