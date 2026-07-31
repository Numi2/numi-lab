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

public enum MetalRoboTaskSolver: UInt32, Sendable {
    case pgs = 0
    case tgs = 1
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
        actionBias: [Float] = [],
        actionScale: [Float] = [],
        observationClip: Float = 100,
        actionClip: Float = 1
    ) {
        self.id = id
        self.revision = revision
        self.observationMean = observationMean
        self.observationInverseStandardDeviation =
            observationInverseStandardDeviation
        self.layers = layers
        self.actionBias = actionBias
        self.actionScale = actionScale
        self.observationClip = observationClip
        self.actionClip = actionClip
    }
}

public struct MetalRoboTaskRolloutConfiguration: Sendable {
    public var environmentCount: UInt32
    public var surface: MetalRoboLocomotionSurface
    public var solver: MetalRoboTaskSolver
    public var physicsSubsteps: UInt32
    public var velocityIterations: UInt32
    public var finalVelocityIterations: UInt32
    public var controlTimestepSeconds: Float
    public var seed: UInt64

    public init(
        environmentCount: UInt32,
        surface: MetalRoboLocomotionSurface = .terrain,
        solver: MetalRoboTaskSolver = .tgs,
        physicsSubsteps: UInt32 = 4,
        velocityIterations: UInt32 = 4,
        finalVelocityIterations: UInt32 = 2,
        controlTimestepSeconds: Float = 0.02,
        seed: UInt64 = 0
    ) {
        self.environmentCount = environmentCount
        self.surface = surface
        self.solver = solver
        self.physicsSubsteps = physicsSubsteps
        self.velocityIterations = velocityIterations
        self.finalVelocityIterations = finalVelocityIterations
        self.controlTimestepSeconds = controlTimestepSeconds
        self.seed = seed
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
    public let scheduledResets: Int
    public let maximumActiveContacts: Int
    public let maximumManifolds: Int
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
        scheduledResets = Int(native.scheduled_resets)
        maximumActiveContacts =
            Int(native.maximum_active_contacts)
        maximumManifolds = Int(native.maximum_manifolds)
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
    public let policyRevision: UInt64

    init(_ native: MRTaskTransitionC) {
        reward = native.reward
        trackingScore = native.tracking_score
        rootHeight = native.root_height
        tilt = native.tilt
        done = native.done != 0
        timeout = native.timeout != 0
        physicsError = native.physics_error != 0
        terminationReason = native.termination_reason
        policyRevision = native.policy_revision
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
        var native = Self.nativeConfiguration(configuration)

        let created: OpaquePointer? = withUnsafePointer(to: &native) {
            config in
            withOptionalCString(metallibPath) {
                metallib in
                mr_create_unitree_g1_locomotion_rollout(
                    config,
                    configuration.surface.rawValue,
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

    private static func nativeConfiguration(
        _ configuration: MetalRoboTaskRolloutConfiguration
    ) -> MRTaskRolloutConfigC {
        var native = MRTaskRolloutConfigC()
        native.environment_count = configuration.environmentCount
        native.solver = configuration.solver.rawValue
        native.physics_substeps = configuration.physicsSubsteps
        native.velocity_iterations =
            configuration.velocityIterations
        native.final_velocity_iterations =
            configuration.finalVelocityIterations
        native.control_timestep_seconds =
            configuration.controlTimestepSeconds
        native.seed = configuration.seed
        return native
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

    public func setPolicy(
        _ policy: MetalRoboPolicyPack
    ) throws {
        let validLayers = policy.layers.allSatisfy { layer in
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
        guard !policy.id.isEmpty,
              policy.revision != 0,
              !policy.layers.isEmpty,
              validLayers
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
                return mr_task_rollout_set_policy(
                    handle,
                    &native
                )
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
            resetMasks.withUnsafeBufferPointer { masks in
                mr_task_rollout_advance(
                    handle,
                    actions.baseAddress,
                    actions.count,
                    masks.baseAddress,
                    masks.count,
                    UInt32(controlStepCount),
                    policyRevision,
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
        policyRevision: UInt64 = 0
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

    private static func lastError() -> String {
        guard let pointer = mr_last_error() else {
            return "MetalRobo task-rollout operation failed."
        }
        return String(cString: pointer)
    }
}
