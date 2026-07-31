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

public enum MetalRoboTaskRolloutError:
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

public enum MetalRoboLocomotionSurface: UInt32, Sendable {
    case ground = 0
    case terrain = 1
}

public enum MetalRoboUnitreeG1Task: UInt32, Sendable {
    case velocity = 0
    case disturbanceRecovery = 1
    case supineGetUpDiscovery = 2
}

public struct MetalRoboDynamicSphere: Sendable {
    public var position: SIMD3<Float>
    public var linearVelocity: SIMD3<Float>
    public var radius: Float
    public var mass: Float
    public var launchStep: UInt32

    public init(
        position: SIMD3<Float>,
        linearVelocity: SIMD3<Float>,
        radius: Float,
        mass: Float,
        launchStep: UInt32 = 0
    ) {
        self.position = position
        self.linearVelocity = linearVelocity
        self.radius = radius
        self.mass = mass
        self.launchStep = launchStep
    }
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
}

public struct MetalRoboTaskRolloutConfiguration: Sendable {
    public var environmentCount: UInt32
    public var surface: MetalRoboLocomotionSurface
    public var physicsSubsteps: UInt32
    public var velocityIterations: UInt32
    public var finalVelocityIterations: UInt32
    public var controlTimestepSeconds: Float
    public var seed: UInt64
    public var dynamicSpheres: [MetalRoboDynamicSphere]
    public var disableTaskTerminations: Bool
    public var unitreeG1Task: MetalRoboUnitreeG1Task

    public init(
        environmentCount: UInt32,
        surface: MetalRoboLocomotionSurface = .terrain,
        physicsSubsteps: UInt32 = 4,
        velocityIterations: UInt32 = 4,
        finalVelocityIterations: UInt32 = 2,
        controlTimestepSeconds: Float = 0.02,
        seed: UInt64 = 0,
        dynamicSpheres: [MetalRoboDynamicSphere] = [],
        disableTaskTerminations: Bool = false,
        unitreeG1Task: MetalRoboUnitreeG1Task = .velocity
    ) {
        self.environmentCount = environmentCount
        self.surface = surface
        self.physicsSubsteps = physicsSubsteps
        self.velocityIterations = velocityIterations
        self.finalVelocityIterations = finalVelocityIterations
        self.controlTimestepSeconds = controlTimestepSeconds
        self.seed = seed
        self.dynamicSpheres = dynamicSpheres
        self.disableTaskTerminations = disableTaskTerminations
        self.unitreeG1Task = unitreeG1Task
    }
}

public struct MetalRoboTaskRolloutLayout: Sendable {
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

    init(_ native: MRTaskRolloutLayoutC) {
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

public struct MetalRoboTaskRolloutAdvance: Sendable {
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

    init(_ native: MRTaskRolloutAdvanceC) {
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
            "solver_tiles": Int(highWater.solver_tiles),
            "spill_rows": Int(highWater.spill_rows),
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
            "quality_generalized_velocities":
                Int(highWater.quality_generalized_velocities),
            "quality_rows": Int(highWater.quality_rows),
            "quality_krylov_vectors":
                Int(highWater.quality_krylov_vectors),
            "quality_direct_tiles":
                Int(highWater.quality_direct_tiles),
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
            throw MetalRoboTaskRolloutError.invalidShape(
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
            throw MetalRoboTaskRolloutError.invalidShape(
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
                throw MetalRoboTaskRolloutError.invalidShape(
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

/// Swift owns rollout chunking and policy revisions. Native code owns the
/// compiled world and task program, persistent private Metal state, reset
/// transaction, and compact learning publication.
public final class MetalRoboTaskRolloutContext {
    private var handle: OpaquePointer?

    /// Bundled G1 is a mechanics + TaskPack preset; execution uses the same
    /// generic compiled task path as imported robots.
    public init(
        unitreeG1 configuration: MetalRoboTaskRolloutConfiguration,
        metallibPath: String? = nil
    ) throws {
        let created: OpaquePointer? =
            Self.withNativeConfiguration(configuration) { config in
                withOptionalCString(metallibPath) {
                    metallib in
                    mr_create_unitree_g1_task_rollout(
                        config,
                        configuration.surface.rawValue,
                        configuration.unitreeG1Task.rawValue,
                        metallib
                    )
                }
            }
        guard let created else {
            throw MetalRoboTaskRolloutError.native(
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
        configuration: MetalRoboTaskRolloutConfiguration,
        metallibPath: String? = nil
    ) throws {
        let created: OpaquePointer? =
            Self.withNativeConfiguration(configuration) { config in
                urdfURL.path.withCString { urdf in
                    taskPackURL.path.withCString { taskPack in
                        withOptionalCString(srdfURL?.path) {
                            srdf in
                            withOptionalCString(metallibPath) {
                                metallib in
                                mr_create_urdf_locomotion_rollout(
                                    urdf,
                                    srdf,
                                    taskPack,
                                    config,
                                    configuration.surface.rawValue,
                                    metallib
                                )
                            }
                        }
                    }
                }
            }
        guard let created else {
            throw MetalRoboTaskRolloutError.native(
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
        configuration: MetalRoboTaskRolloutConfiguration,
        metallibPath: String? = nil
    ) throws {
        let created: OpaquePointer? =
            Self.withNativeConfiguration(configuration) { config in
                worldPackURL.path.withCString { worldPack in
                    taskPackURL.path.withCString { taskPack in
                        withOptionalCString(metallibPath) {
                            metallib in
                            mr_create_world_pack_locomotion_rollout(
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
            throw MetalRoboTaskRolloutError.native(
                Self.lastError()
            )
        }
        handle = created
    }

    private static func nativeConfiguration(
        _ configuration: MetalRoboTaskRolloutConfiguration
    ) -> MRTaskRolloutConfigC {
        var native = MRTaskRolloutConfigC()
        native.environment_count = configuration.environmentCount
        native.physics_substeps = configuration.physicsSubsteps
        native.velocity_iterations =
            configuration.velocityIterations
        native.final_velocity_iterations =
            configuration.finalVelocityIterations
        native.control_timestep_seconds =
            configuration.controlTimestepSeconds
        native.seed = configuration.seed
        native.disable_task_terminations =
            configuration.disableTaskTerminations ? 1 : 0
        return native
    }

    private static func withNativeConfiguration<Result>(
        _ configuration: MetalRoboTaskRolloutConfiguration,
        _ body: (UnsafePointer<MRTaskRolloutConfigC>) -> Result
    ) -> Result {
        let spheres = configuration.dynamicSpheres.map { source in
            var sphere = MRTaskRolloutDynamicSphereC()
            sphere.position = (
                source.position.x,
                source.position.y,
                source.position.z
            )
            sphere.linear_velocity = (
                source.linearVelocity.x,
                source.linearVelocity.y,
                source.linearVelocity.z
            )
            sphere.radius = source.radius
            sphere.mass = source.mass
            sphere.launch_step = source.launchStep
            return sphere
        }
        return spheres.withUnsafeBufferPointer { buffer in
            var native = nativeConfiguration(configuration)
            native.dynamic_spheres = buffer.baseAddress
            native.dynamic_sphere_count = UInt32(buffer.count)
            return withUnsafePointer(to: &native, body)
        }
    }

    deinit {
        if let handle {
            mr_task_rollout_destroy(handle)
        }
    }

    public var layout: MetalRoboTaskRolloutLayout {
        MetalRoboTaskRolloutLayout(
            mr_task_rollout_layout(handle)
        )
    }

    public var deviceName: String {
        String(cString: mr_task_rollout_device_name(handle))
    }

    public func reset(seed: UInt64) throws {
        guard mr_task_rollout_reset(handle, seed) == 0 else {
            throw MetalRoboTaskRolloutError.native(
                Self.lastError()
            )
        }
    }

    public func setCurriculumLevel(_ level: UInt32) throws {
        guard mr_task_rollout_set_curriculum_level(
            handle,
            level
        ) == 0 else {
            throw MetalRoboTaskRolloutError.native(
                Self.lastError()
            )
        }
    }

    public func setStateReadback(_ enabled: Bool) throws {
        guard mr_task_rollout_set_state_readback(
            handle,
            enabled ? 1 : 0
        ) == 0 else {
            throw MetalRoboTaskRolloutError.native(Self.lastError())
        }
    }

    public func setPolicy(
        _ policy: MetalRoboPolicyPack
    ) throws {
        let validLayer: (
            MetalRoboPolicyDenseLayer
        ) -> Bool = { layer in
            let product =
                layer.inputCount.multipliedReportingOverflow(
                    by: layer.outputCount
                )
            return
                layer.inputCount > 0 &&
                layer.inputCount <= Int(UInt32.max) &&
                layer.outputCount > 0 &&
                layer.outputCount <= Int(UInt32.max) &&
                !product.overflow &&
                layer.weights.count == product.partialValue &&
                layer.bias.count == layer.outputCount
        }
        let validLayers = policy.layers.allSatisfy(validLayer)
        let validCriticLayers =
            policy.criticLayers.allSatisfy(validLayer)
        guard !policy.id.isEmpty,
              policy.revision != 0,
              !policy.layers.isEmpty,
              validLayers,
              validCriticLayers,
              policy.actionLogStandardDeviation.isEmpty ||
                  !policy.criticLayers.isEmpty
        else {
            throw MetalRoboTaskRolloutError.invalidShape(
                "Policy identity, revision, or dense layer shape is invalid."
            )
        }
        var allocations: [
            (pointer: UnsafeMutablePointer<Float>, count: Int)
        ] = []
        func copy(
            _ values: [Float]
        ) -> UnsafePointer<Float>? {
            guard !values.isEmpty else {
                return nil
            }
            let pointer =
                UnsafeMutablePointer<Float>.allocate(
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
                allocation.pointer.deinitialize(
                    count: allocation.count
                )
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
        let nativeCriticLayers =
            policy.criticLayers.map { layer in
                var native = MRPolicyDenseLayerC()
                native.input_count =
                    UInt32(layer.inputCount)
                native.output_count =
                    UInt32(layer.outputCount)
                native.activation = layer.activation.rawValue
                native.weights = copy(layer.weights)
                native.weight_count = layer.weights.count
                native.bias = copy(layer.bias)
                native.bias_count = layer.bias.count
                return native
            }
        var native = MRPolicyPackC()
        native.revision = policy.revision
        native.observation_mean = copy(
            policy.observationMean
        )
        native.observation_mean_count =
            policy.observationMean.count
        native.observation_inverse_standard_deviation =
            copy(
                policy.observationInverseStandardDeviation
            )
        native.observation_inverse_standard_deviation_count =
            policy.observationInverseStandardDeviation.count
        native.critic_observation_mean = copy(
            policy.criticObservationMean
        )
        native.critic_observation_mean_count =
            policy.criticObservationMean.count
        native.critic_observation_inverse_standard_deviation =
            copy(
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

        let status = policy.id.withCString { identifier in
            native.id = identifier
            return nativeLayers.withUnsafeBufferPointer {
                layers in
                native.layers = layers.baseAddress
                native.layer_count = layers.count
                return nativeCriticLayers
                    .withUnsafeBufferPointer { criticLayers in
                        native.critic_layers =
                            criticLayers.baseAddress
                        native.critic_layer_count =
                            criticLayers.count
                        return mr_task_rollout_set_policy(
                            handle,
                            &native
                        )
                    }
            }
        }
        guard status == 0 else {
            throw MetalRoboTaskRolloutError.native(
                Self.lastError()
            )
        }
    }

    public func loadPolicy(at url: URL) throws {
        let status = url.path.withCString {
            mr_task_rollout_load_policy_pack(handle, $0)
        }
        guard status == 0 else {
            throw MetalRoboTaskRolloutError.native(
                Self.lastError()
            )
        }
    }

    public func clearPolicy() throws {
        guard mr_task_rollout_clear_policy(handle) == 0 else {
            throw MetalRoboTaskRolloutError.native(
                Self.lastError()
            )
        }
    }

    public func advance(
        normalizedActions: [Float],
        resetMasks: [UInt32] = [],
        controlStepCount: Int,
        policyRevision: UInt64 = 0
    ) throws -> MetalRoboTaskRolloutAdvance {
        let current = layout
        guard controlStepCount > 0 else {
            throw MetalRoboTaskRolloutError.invalidShape(
                "controlStepCount must be positive."
            )
        }
        let actionElements =
            controlStepCount *
            current.environmentCount *
            current.actionCount
        guard normalizedActions.count == actionElements else {
            throw MetalRoboTaskRolloutError.invalidShape(
                "Actions must contain \(actionElements) values."
            )
        }
        let maskElements =
            controlStepCount * current.environmentCount
        guard resetMasks.isEmpty ||
                resetMasks.count == maskElements
        else {
            throw MetalRoboTaskRolloutError.invalidShape(
                "Reset masks must be empty or contain " +
                    "\(maskElements) values."
            )
        }

        var native = MRTaskRolloutAdvanceC()
        let status = normalizedActions.withUnsafeBufferPointer {
            actions in
            if resetMasks.isEmpty {
                return mr_task_rollout_advance(
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
                mr_task_rollout_advance(
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
            throw MetalRoboTaskRolloutError.native(
                Self.lastError()
            )
        }
        return MetalRoboTaskRolloutAdvance(native)
    }

    public func advanceWithPolicy(
        resetMasks: [UInt32] = [],
        controlStepCount: Int,
        policyRevision: UInt64 = 0,
        evaluateFinalPolicy: Bool = false
    ) throws -> MetalRoboTaskRolloutAdvance {
        let current = layout
        guard controlStepCount > 0 else {
            throw MetalRoboTaskRolloutError.invalidShape(
                "controlStepCount must be positive."
            )
        }
        let maskElements =
            controlStepCount * current.environmentCount
        guard resetMasks.isEmpty ||
                resetMasks.count == maskElements
        else {
            throw MetalRoboTaskRolloutError.invalidShape(
                "Reset masks must be empty or contain " +
                    "\(maskElements) values."
            )
        }
        var native = MRTaskRolloutAdvanceC()
        if resetMasks.isEmpty {
            let status = mr_task_rollout_advance(
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
                throw MetalRoboTaskRolloutError.native(
                    Self.lastError()
                )
            }
            return MetalRoboTaskRolloutAdvance(native)
        }
        let status = resetMasks.withUnsafeBufferPointer {
            masks in
            mr_task_rollout_advance(
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
            throw MetalRoboTaskRolloutError.native(
                Self.lastError()
            )
        }
        return MetalRoboTaskRolloutAdvance(native)
    }

    public func statusCodes(
        controlStepCount: Int
    ) throws -> [UInt32] {
        let count = controlStepCount * layout.environmentCount
        guard let values =
                mr_task_rollout_status_codes(handle)
        else {
            throw MetalRoboTaskRolloutError.native(
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
                mr_task_rollout_active_contacts(handle)
        else {
            throw MetalRoboTaskRolloutError.native(
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
                mr_task_rollout_actor_observations(handle)
        else {
            throw MetalRoboTaskRolloutError.native(
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
                mr_task_rollout_critic_observations(handle)
        else {
            throw MetalRoboTaskRolloutError.native(
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
                mr_task_rollout_transitions(handle)
        else {
            throw MetalRoboTaskRolloutError.native(
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
                mr_task_rollout_policy_latents(handle)
        else {
            throw MetalRoboTaskRolloutError.native(
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
                mr_task_rollout_policy_log_probabilities(handle)
        else {
            throw MetalRoboTaskRolloutError.native(
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
                mr_task_rollout_policy_values(handle)
        else {
            throw MetalRoboTaskRolloutError.native(
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
                mr_task_rollout_bootstrap_policy_values(handle)
        else {
            throw MetalRoboTaskRolloutError.native(
                "Post-rollout policy values are unavailable."
            )
        }
        return Array(
            UnsafeBufferPointer(start: values, count: count)
        )
    }

    public func finalConfiguration() throws -> [Float] {
        let count = layout.environmentCount * layout.configurationCount
        guard let values = mr_task_rollout_final_q(handle) else {
            throw MetalRoboTaskRolloutError.native(
                "Final configuration is unavailable."
            )
        }
        return Array(
            UnsafeBufferPointer(start: values, count: count)
        )
    }

    public func finalSceneStates() throws -> [Float] {
        let current = layout
        let count =
            current.environmentCount * current.sceneBodyCount * 13
        var output = [Float](repeating: 0, count: count)
        let status = output.withUnsafeMutableBufferPointer { buffer in
            mr_task_rollout_copy_final_scene_states(
                handle,
                buffer.baseAddress,
                buffer.count
            )
        }
        guard status == 0 else {
            throw MetalRoboTaskRolloutError.native(Self.lastError())
        }
        return output
    }

    public func policyRolloutBatch(
        controlStepCount: Int
    ) throws -> MetalRoboPolicyRolloutBatch {
        let current = layout
        guard controlStepCount > 0 else {
            throw MetalRoboTaskRolloutError.invalidShape(
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

    // Training hot path: append the current native readback directly into one
    // caller-owned update arena. This avoids materializing a second complete
    // Swift chunk before the append.
    public func appendCurrentPolicyRollout(
        controlStepCount: Int,
        actorObservations: inout [Float],
        criticObservations: inout [Float],
        latents: inout [Float],
        logProbabilities: inout [Float],
        values: inout [Float],
        transitions: inout [MetalRoboTaskTransition]
    ) throws {
        let current = layout
        guard controlStepCount > 0 else {
            throw MetalRoboTaskRolloutError.invalidShape(
                "Policy rollout append must contain at least one control step."
            )
        }
        let samples =
            controlStepCount * current.environmentCount
        let actorCount =
            samples * current.actorObservationCount
        let criticCount =
            samples * current.criticObservationCount
        let latentCount = samples * current.actionCount
        guard let actor =
                  mr_task_rollout_actor_observations(handle),
              let critic =
                  mr_task_rollout_critic_observations(handle),
              let latent =
                  mr_task_rollout_policy_latents(handle),
              let logProbability =
                  mr_task_rollout_policy_log_probabilities(handle),
              let value =
                  mr_task_rollout_policy_values(handle),
              let transition =
                  mr_task_rollout_transitions(handle)
        else {
            throw MetalRoboTaskRolloutError.native(
                "Native policy rollout streams are unavailable."
            )
        }
        actorObservations.append(
            contentsOf: UnsafeBufferPointer(
                start: actor,
                count: actorCount
            )
        )
        criticObservations.append(
            contentsOf: UnsafeBufferPointer(
                start: critic,
                count: criticCount
            )
        )
        latents.append(
            contentsOf: UnsafeBufferPointer(
                start: latent,
                count: latentCount
            )
        )
        logProbabilities.append(
            contentsOf: UnsafeBufferPointer(
                start: logProbability,
                count: samples
            )
        )
        values.append(
            contentsOf: UnsafeBufferPointer(
                start: value,
                count: samples
            )
        )
        for native in UnsafeBufferPointer(
            start: transition,
            count: samples
        ) {
            transitions.append(
                MetalRoboTaskTransition(native)
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
            throw MetalRoboTaskRolloutError.invalidShape(
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
                        mr_task_rollout_write_policy_rollout_pack(
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
            throw MetalRoboTaskRolloutError.native(
                Self.lastError()
            )
        }
    }

    private static func lastError() -> String {
        guard let pointer = mr_last_error() else {
            return "MetalRobo task-rollout operation failed."
        }
        return String(cString: pointer)
    }
}
