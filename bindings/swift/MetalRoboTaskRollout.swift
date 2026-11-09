import Foundation
import Metal
import MetalRoboC

public enum MetalRoboTaskTerminationReason: UInt32, Sendable {
    case continuing = 0
    case height = 1
    case tilt = 2
    case contact = 3
    case timeout = 4
    case physicsError = 5
    case projectileContact = 6
    case navigationCompletion = 7
}

private func withOptionalCString<Result>(
    _ value: String?,
    _ body: (UnsafePointer<CChar>?) throws -> Result
) rethrows -> Result {
    guard let value else {
        return try body(nil)
    }
    return try value.withCString(body)
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

private func withUnsafeCStringBuffers<Result>(
    _ strings: [String],
    _ body: ([UnsafePointer<CChar>]) throws -> Result
) rethrows -> Result {
    let storage = strings.map { Array($0.utf8CString) }
    var pointers: [UnsafePointer<CChar>] = []
    pointers.reserveCapacity(storage.count)
    func visit(_ index: Int) throws -> Result {
        guard index < storage.count else {
            return try body(pointers)
        }
        return try storage[index].withUnsafeBufferPointer { buffer in
            pointers.append(buffer.baseAddress!)
            defer { pointers.removeLast() }
            return try visit(index + 1)
        }
    }
    return try visit(0)
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
    case ballDisturbanceRecovery = 3
    case ballDodge = 4
    case developmentalRecovery = 5
    case adultLocomotion = 6
    case g1LegsLocomotion = 7
}

public enum MetalRoboInteractionReferenceMode: UInt32, Sendable {
    case taskDefault = 0
    case guide = 1
    case resetOnly = 2
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

    public static let g1BallRecoveryDefaults: [Self] = [
        .init(
            position: SIMD3(-1.5, 0.0, 1.0),
            linearVelocity: SIMD3(5.0, 0.0, 2.5),
            radius: 0.10,
            mass: 1.00,
            launchStep: 100
        ),
        .init(
            position: SIMD3(1.5, 0.0, 1.0),
            linearVelocity: SIMD3(-5.0, 0.0, 2.5),
            radius: 0.12,
            mass: 2.00,
            launchStep: 200
        ),
        .init(
            position: SIMD3(0.0, -1.5, 1.0),
            linearVelocity: SIMD3(0.0, 5.0, 2.5),
            radius: 0.14,
            mass: 3.00,
            launchStep: 300
        ),
        .init(
            position: SIMD3(0.0, 1.5, 1.0),
            linearVelocity: SIMD3(0.0, -5.0, 2.5),
            radius: 0.16,
            mass: 4.00,
            launchStep: 400
        ),
        .init(
            position: SIMD3(-1.5, 0.25, 1.0),
            linearVelocity: SIMD3(5.0, 0.0, 2.5),
            radius: 0.18,
            mass: 5.00,
            launchStep: 500
        ),
        .init(
            position: SIMD3(1.5, -0.25, 1.0),
            linearVelocity: SIMD3(-5.0, 0.0, 2.5),
            radius: 0.20,
            mass: 6.00,
            launchStep: 600
        ),
    ]

    // Dodge training is geometric: all projectiles remain light so contact
    // avoidance is learned before impact-bracing dynamics are mixed in.
    public static let g1BallDodgeDefaults: [Self] = [
        .init(
            position: SIMD3(-1.5, 0.0, 1.0),
            linearVelocity: SIMD3(5.0, 0.0, 2.5),
            radius: 0.10, mass: 0.136, launchStep: 100
        ),
        .init(
            position: SIMD3(1.5, 0.0, 1.0),
            linearVelocity: SIMD3(-5.0, 0.0, 2.5),
            radius: 0.10, mass: 0.136, launchStep: 200
        ),
        .init(
            position: SIMD3(0.0, -1.5, 1.0),
            linearVelocity: SIMD3(0.0, 5.0, 2.5),
            radius: 0.10, mass: 0.136, launchStep: 300
        ),
        .init(
            position: SIMD3(0.0, 1.5, 1.0),
            linearVelocity: SIMD3(0.0, -5.0, 2.5),
            radius: 0.10, mass: 0.136, launchStep: 400
        ),
        .init(
            position: SIMD3(-1.5, 0.25, 1.0),
            linearVelocity: SIMD3(5.0, 0.0, 2.5),
            radius: 0.10, mass: 0.136, launchStep: 500
        ),
        .init(
            position: SIMD3(1.5, -0.25, 1.0),
            linearVelocity: SIMD3(-5.0, 0.0, 2.5),
            radius: 0.10, mass: 0.136, launchStep: 600
        ),
    ]
}

public struct MetalRoboTaskVisualPack: Sendable {
    public var url: URL
    public var assetID: String
    public var semanticID: UInt32
    public var instanceID: UInt32

    public init(
        url: URL,
        assetID: String,
        semanticID: UInt32,
        instanceID: UInt32
    ) {
        self.url = url
        self.assetID = assetID
        self.semanticID = semanticID
        self.instanceID = instanceID
    }
}

public struct MetalRoboRetinalObservationConfiguration: Sendable, Decodable {
    public var rightPosition: [Float]
    public var rightOrientation: [Float]
    public var width: UInt32
    public var height: UInt32
    public var sampleNumerator: UInt32
    public var sampleDenominator: UInt32
    public var latencyTicks: UInt32
    public var minimumDepthMeters: Float
    public var maximumDepthMeters: Float
    private enum CodingKeys: String, CodingKey {
        case rightPosition = "right_position", rightOrientation = "right_orientation"
        case width, height
        case sampleNumerator = "sample_numerator", sampleDenominator = "sample_denominator"
        case latencyTicks = "latency_ticks", minimumDepthMeters = "minimum_depth_meters"
        case maximumDepthMeters = "maximum_depth_meters"
    }
    fileprivate func native() throws -> MRRetinalSensorConfigC {
        guard rightPosition.count == 3, rightOrientation.count == 4,
              rightPosition.allSatisfy(\.isFinite), rightOrientation.allSatisfy(\.isFinite)
        else { throw MetalRoboTaskRolloutError.invalidShape("Retinal camera mount requires finite position3 and orientation4.") }
        var value = MRRetinalSensorConfigC()
        value.struct_size = UInt32(MemoryLayout<MRRetinalSensorConfigC>.size)
        value.abi_version = 1
        value.right_camera_position = (rightPosition[0], rightPosition[1], rightPosition[2])
        value.right_camera_orientation = (rightOrientation[0], rightOrientation[1], rightOrientation[2], rightOrientation[3])
        value.width = width; value.height = height
        value.sample_numerator = sampleNumerator; value.sample_denominator = sampleDenominator
        value.latency_ticks = latencyTicks; value.minimum_depth_meters = minimumDepthMeters
        value.maximum_depth_meters = maximumDepthMeters
        return value
    }
}

public struct MetalRoboCrowObjectPushCurriculum: Sendable, Decodable {
    public let schema: String
    public let episodeSteps: UInt32
    public let xRangeMeters: [Float]
    public let yRangeMeters: [Float]
    private enum CodingKeys: String, CodingKey {
        case schema
        case episodeSteps = "episode_steps"
        case xRangeMeters = "x_range_m", yRangeMeters = "y_range_m"
    }
    public static func load(from url: URL) throws -> Self {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard attributes[.type] as? FileAttributeType == .typeRegular,
              let bytes = attributes[.size] as? NSNumber, bytes.intValue > 0, bytes.intValue <= 16384 else {
            throw MetalRoboTaskRolloutError.invalidShape("Object curriculum must be a bounded regular JSON file.")
        }
        let result = try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
        _ = try result.native()
        return result
    }
    fileprivate func native() throws -> MRCrowObjectPushCurriculumC {
        guard schema == "numi.crow-object-push-curriculum.v1", (32...4096).contains(episodeSteps),
              xRangeMeters.count == 2, yRangeMeters.count == 2,
              (xRangeMeters + yRangeMeters).allSatisfy(\.isFinite),
              xRangeMeters[0] <= xRangeMeters[1], yRangeMeters[0] <= yRangeMeters[1],
              xRangeMeters[0] >= 0.222, xRangeMeters[1] <= 0.228,
              yRangeMeters[0] >= 0.026, yRangeMeters[1] <= 0.060 else {
            throw MetalRoboTaskRolloutError.invalidShape("Invalid object curriculum schema, lifetime or placement envelope.")
        }
        var c = MRCrowObjectPushCurriculumC()
        c.struct_size = UInt32(MemoryLayout<MRCrowObjectPushCurriculumC>.size)
        c.abi_version = 1; c.episode_steps = episodeSteps
        c.minimum_x = xRangeMeters[0]; c.maximum_x = xRangeMeters[1]
        c.minimum_y = yRangeMeters[0]; c.maximum_y = yRangeMeters[1]
        return c
    }
}

public struct MetalRoboCrowCuedForagingConfiguration: Sendable, Decodable {
    public var ports: [[Float]]
    public var cueMeshInstances: [UInt32]
    public var allocationSeed: UInt64
    public var memoryChoiceDecisionTick: UInt32?
    public var headYawHoldTicks: UInt32?
    public var beakHoldTicks: UInt32?
    public var occluded: Bool
    private enum CodingKeys: String, CodingKey {
        case ports, occluded
        case memoryChoiceDecisionTick = "memory_choice_decision_tick"
        case headYawHoldTicks = "head_yaw_hold_ticks", beakHoldTicks = "beak_hold_ticks"
        case cueMeshInstances = "cue_mesh_instances", allocationSeed = "allocation_seed"
    }
    public init(ports: [[Float]], cueMeshInstances: [UInt32], allocationSeed: UInt64, occluded: Bool = false, memoryChoiceDecisionTick: UInt32? = nil, headYawHoldTicks: UInt32? = nil, beakHoldTicks: UInt32? = nil) {
        self.ports = ports; self.cueMeshInstances = cueMeshInstances
        self.allocationSeed = allocationSeed; self.occluded = occluded
        self.memoryChoiceDecisionTick = memoryChoiceDecisionTick
        self.headYawHoldTicks = headYawHoldTicks; self.beakHoldTicks = beakHoldTicks
    }
    fileprivate func native() throws -> MRCrowCuedForagingConfigC {
        guard (headYawHoldTicks == nil && beakHoldTicks == nil) ||
              (memoryChoiceDecisionTick != nil && (1...512).contains(headYawHoldTicks ?? 0) && (1...512).contains(beakHoldTicks ?? 0))
        else { throw MetalRoboTaskRolloutError.invalidShape("Native holds require a decision tick and both periods in 1...512.") }
        guard ports.count == 2, ports.allSatisfy({ $0.count == 4 && $0.allSatisfy(\.isFinite) && $0[3] > 0 }),
              cueMeshInstances.count == 2, cueMeshInstances[0] != cueMeshInstances[1]
        else { throw MetalRoboTaskRolloutError.invalidShape("Cued foraging requires two finite ports and distinct mesh indices.") }
        var value = MRCrowCuedForagingConfigC()
        value.struct_size = UInt32(MemoryLayout<MRCrowCuedForagingConfigC>.size); value.abi_version = 1
        value.left_port = (ports[0][0], ports[0][1], ports[0][2], ports[0][3])
        value.right_port = (ports[1][0], ports[1][1], ports[1][2], ports[1][3])
        value.cue_mesh_instances = (cueMeshInstances[0], cueMeshInstances[1])
        value.allocation_seed = allocationSeed; value.occluded = occluded ? 1 : 0
        return value
    }
}

public struct MetalRoboTaskVisualObservationConfiguration:
    Sendable
{
    private struct Artifact: Decodable {
        struct Pack: Decodable {
            let path: String
            let assetID: String
            let semanticID: UInt32
            let instanceID: UInt32

            private enum CodingKeys: String, CodingKey {
                case path
                case assetID = "asset_id"
                case semanticID = "semantic_id"
                case instanceID = "instance_id"
            }
        }

        struct Camera: Decodable {
            let parentBody: String
            let position: [Float]
            let orientation: [Float]
            let width: UInt32
            let height: UInt32
            let minimumVisiblePixels: UInt32
            let verticalFieldOfViewDegrees: Float
            let nominalRateHz: Float
            let maximumRetainedBytes: UInt64

            private enum CodingKeys: String, CodingKey {
                case parentBody = "parent_body"
                case position, orientation, width, height
                case minimumVisiblePixels = "minimum_visible_pixels"
                case verticalFieldOfViewDegrees =
                    "vertical_field_of_view_degrees"
                case nominalRateHz = "nominal_rate_hz"
                case maximumRetainedBytes = "maximum_retained_bytes"
            }
        }

        let format: String
        let id: String
        let packs: [Pack]
        let environmentPack: String?
        let camera: Camera
        let retinal: MetalRoboRetinalObservationConfiguration?
        let cuedForaging: MetalRoboCrowCuedForagingConfiguration?

        private enum CodingKeys: String, CodingKey {
            case format, id, packs, camera, retinal
            case environmentPack = "environment_pack"
            case cuedForaging = "cued_foraging"
        }
    }

    public var packs: [MetalRoboTaskVisualPack]
    public var environmentPackURL: URL?
    public var cameraParentBody: String
    public var cameraPosition: SIMD3<Float>
    public var cameraOrientation: SIMD4<Float>
    public var width: UInt32
    public var height: UInt32
    public var minimumVisiblePixels: UInt32
    public var verticalFieldOfViewDegrees: Float
    public var nominalRateHz: Float
    public var maximumRetainedBytes: UInt64
    public var captureWidth: UInt32
    public var captureHeight: UInt32
    public var capturePolicyCamera: Bool
    public var retinal: MetalRoboRetinalObservationConfiguration?
    public var cuedForaging: MetalRoboCrowCuedForagingConfiguration?

    public init(
        packs: [MetalRoboTaskVisualPack],
        environmentPackURL: URL? = nil,
        cameraParentBody: String,
        cameraPosition: SIMD3<Float>,
        cameraOrientation: SIMD4<Float>,
        width: UInt32 = 160,
        height: UInt32 = 120,
        minimumVisiblePixels: UInt32 = 4,
        verticalFieldOfViewDegrees: Float = 0,
        nominalRateHz: Float = 15,
        maximumRetainedBytes: UInt64 = 0,
        captureWidth: UInt32 = 0,
        captureHeight: UInt32 = 0,
        capturePolicyCamera: Bool = false,
        retinal: MetalRoboRetinalObservationConfiguration? = nil,
        cuedForaging: MetalRoboCrowCuedForagingConfiguration? = nil
    ) {
        self.packs = packs
        self.environmentPackURL = environmentPackURL
        self.cameraParentBody = cameraParentBody
        self.cameraPosition = cameraPosition
        self.cameraOrientation = cameraOrientation
        self.width = width
        self.height = height
        self.minimumVisiblePixels = minimumVisiblePixels
        self.verticalFieldOfViewDegrees = verticalFieldOfViewDegrees
        self.nominalRateHz = nominalRateHz
        self.maximumRetainedBytes = maximumRetainedBytes
        self.captureWidth = captureWidth
        self.captureHeight = captureHeight
        self.capturePolicyCamera = capturePolicyCamera
        self.retinal = retinal
        self.cuedForaging = cuedForaging
    }

    public static func loadArtifact(at url: URL) throws -> Self {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        let artifact: Artifact
        do {
            artifact = try decoder.decode(Artifact.self, from: data)
        } catch {
            throw MetalRoboTaskRolloutError.invalidShape(
                "Visual observation artifact is invalid: \(error)"
            )
        }
        guard artifact.format == (artifact.retinal == nil ? "numi.visual-observation.v1" : "numi.retinal-observation.v1"),
              artifact.cuedForaging == nil || artifact.retinal != nil,
              !artifact.id.isEmpty,
              !artifact.packs.isEmpty,
              !artifact.camera.parentBody.isEmpty,
              artifact.camera.position.count == 3,
              artifact.camera.orientation.count == 4,
              artifact.camera.position.allSatisfy(\.isFinite),
              artifact.camera.orientation.allSatisfy(\.isFinite),
              artifact.camera.verticalFieldOfViewDegrees.isFinite,
              artifact.camera.nominalRateHz.isFinite
        else {
            throw MetalRoboTaskRolloutError.invalidShape(
                "Visual observation artifact has an invalid identity, camera, or pack table."
            )
        }
        let base = url.deletingLastPathComponent()
        func resolve(_ path: String) -> URL {
            URL(fileURLWithPath: path, relativeTo: base)
                .standardizedFileURL
        }
        return Self(
            packs: artifact.packs.map { pack in
                MetalRoboTaskVisualPack(
                    url: resolve(pack.path),
                    assetID: pack.assetID,
                    semanticID: pack.semanticID,
                    instanceID: pack.instanceID
                )
            },
            environmentPackURL: artifact.environmentPack.map(resolve),
            cameraParentBody: artifact.camera.parentBody,
            cameraPosition: SIMD3(
                artifact.camera.position[0],
                artifact.camera.position[1],
                artifact.camera.position[2]
            ),
            cameraOrientation: SIMD4(
                artifact.camera.orientation[0],
                artifact.camera.orientation[1],
                artifact.camera.orientation[2],
                artifact.camera.orientation[3]
            ),
            width: artifact.camera.width,
            height: artifact.camera.height,
            minimumVisiblePixels:
                artifact.camera.minimumVisiblePixels,
            verticalFieldOfViewDegrees:
                artifact.camera.verticalFieldOfViewDegrees,
            nominalRateHz: artifact.camera.nominalRateHz,
            maximumRetainedBytes:
                artifact.camera.maximumRetainedBytes,
            retinal: artifact.retinal,
            cuedForaging: artifact.cuedForaging
        )
    }

    public static func unitreeG1(
        robotPackDirectory: URL,
        environmentPackURL: URL? = nil
    ) throws -> Self {
        let robotPacks = try FileManager.default
            .contentsOfDirectory(
                at: robotPackDirectory,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )
            .filter { $0.pathExtension == "mrvpack" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !robotPacks.isEmpty else {
            throw MetalRoboTaskRolloutError.invalidShape(
                "The G1 visual-pack directory contains no .mrvpack files."
            )
        }
        let cameraHalfAngle = Float(35.0 * .pi / 180.0)
        return Self(
            packs: robotPacks.map {
                MetalRoboTaskVisualPack(
                    url: $0,
                    assetID: "robot",
                    semanticID: 1,
                    instanceID: 1
                )
            },
            environmentPackURL: environmentPackURL,
            cameraParentBody: "torso_link",
            cameraPosition: SIMD3(0.08, 0.0, 0.45),
            cameraOrientation: SIMD4(
                0.0,
                sin(cameraHalfAngle),
                0.0,
                cos(cameraHalfAngle)
            ),
            width: 16,
            height: 9,
            minimumVisiblePixels: 1,
            verticalFieldOfViewDegrees: 54,
            nominalRateHz: 50,
            maximumRetainedBytes: 4 * 1024 * 1024 * 1024
        )
    }

    public static func unitreeG1BallRecovery(
        robotPackDirectory: URL,
        ballPackDirectory: URL,
        environmentPackURL: URL? = nil
    ) throws -> Self {
        let robotPacks = try FileManager.default
            .contentsOfDirectory(
                at: robotPackDirectory,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )
            .filter { $0.pathExtension == "mrvpack" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !robotPacks.isEmpty else {
            throw MetalRoboTaskRolloutError.invalidShape(
                "The G1 visual-pack directory contains no .mrvpack files."
            )
        }
        var packs = robotPacks.map {
            MetalRoboTaskVisualPack(
                url: $0,
                assetID: "robot",
                semanticID: 1,
                instanceID: 1
            )
        }
        let ballPacks = try FileManager.default
            .contentsOfDirectory(
                at: ballPackDirectory,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )
            .filter { $0.pathExtension == "mrvpack" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard ballPacks.count == 6 else {
            throw MetalRoboTaskRolloutError.invalidShape(
                "The ball visual-pack directory must contain six body-bound .mrvpack files."
            )
        }
        for index in ballPacks.indices {
            packs.append(MetalRoboTaskVisualPack(
                url: ballPacks[index],
                assetID: "locomotion_dynamic_sphere_\(index)",
                semanticID: 2,
                instanceID: UInt32(100 + index)
            ))
        }
        // The optical +z axis is pitched 20 degrees above torso-forward.
        // This preserves the lower part of the image for descending throws
        // while extending useful coverage through the head-height band.
        let cameraHalfAngle = Float(35.0 * .pi / 180.0)
        return Self(
            packs: packs,
            environmentPackURL: environmentPackURL,
            cameraParentBody: "torso_link",
            cameraPosition: SIMD3(0.08, 0.0, 0.45),
            // A +70-degree local-y rotation maps optical +z to a forward
            // ray pitched +20 degrees in torso coordinates.
            cameraOrientation: SIMD4(
                0.0,
                sin(cameraHalfAngle),
                0.0,
                cos(cameraHalfAngle)
            ),
            width: 16,
            height: 9,
            minimumVisiblePixels: 1,
            verticalFieldOfViewDegrees: 54,
            // The task's sparse offsets (0, 3, 8, 18) assume distinct 50 Hz
            // camera samples. An implicit 15 Hz default aliases those frames
            // and weakens the looming cue used to infer approach velocity.
            nominalRateHz: 50,
            maximumRetainedBytes: 4 * 1024 * 1024 * 1024
        )
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
    public var contractVersion: UInt64
    public var worldFingerprint: UInt64
    public var taskFingerprint: UInt64
    public var observationFingerprint: UInt64
    public var actionFingerprint: UInt64
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
        contractVersion: UInt64 = 0,
        worldFingerprint: UInt64 = 0,
        taskFingerprint: UInt64 = 0,
        observationFingerprint: UInt64 = 0,
        actionFingerprint: UInt64 = 0,
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
        self.contractVersion = contractVersion
        self.worldFingerprint = worldFingerprint
        self.taskFingerprint = taskFingerprint
        self.observationFingerprint = observationFingerprint
        self.actionFingerprint = actionFingerprint
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

public enum MetalRoboBirdFlowJourneyVariant: UInt32, Sendable {
    case v7Hierarchical = 0
    case v8Neural = 1
    case v9VisualNeural = 2
    case v10WorldModelNavigation = 3
    case v11Embodied = 4
    case v12FeedingStation = 5
    case v13RetinalFeeding = 6
    case v14ObjectPush = 7
    case v15ActiveHeadNavigation = 8
    case v15HierarchicalNavigation = 9
    case v15WaypointExpertNavigation = 10
    case v15DualWaypointExpertNavigation = 11
    case v15TripleWaypointExpertNavigation = 12

    public var commandName: String {
        switch self {
        case .v7Hierarchical: "v7-hierarchical"
        case .v8Neural: "v8-neural"
        case .v9VisualNeural: "v9-visual-neural"
        case .v10WorldModelNavigation: "v10-world-model"
        case .v11Embodied: "v11-embodied"
        case .v12FeedingStation: "v12-feeding-station"
        case .v13RetinalFeeding: "v13-retinal-feeding"
        case .v14ObjectPush: "v14-object-push"
        case .v15ActiveHeadNavigation: "v15-active-head-navigation"
        case .v15HierarchicalNavigation: "v15-hierarchical-navigation"
        case .v15WaypointExpertNavigation: "v15-waypoint-expert-navigation"
        case .v15DualWaypointExpertNavigation: "v15-dual-waypoint-expert-navigation"
        case .v15TripleWaypointExpertNavigation: "v15-triple-waypoint-expert-navigation"
        }
    }

    public var taskIdentifier: String {
        switch self {
        case .v7Hierarchical: "birdflow_american_crow_journey_v7"
        case .v8Neural: "birdflow_american_crow_journey_v8_neural"
        case .v9VisualNeural: "birdflow_american_crow_journey_v9_visual_neural"
        case .v10WorldModelNavigation: "birdflow_american_crow_navigation_v10_world_model"
        case .v11Embodied: "american_crow_embodied_development_v11"
        case .v12FeedingStation: "american_crow_feeding_station_v12"
        case .v13RetinalFeeding: "american_crow_retinal_feeding_v13"
        case .v14ObjectPush: "american_crow_object_push_v14"
        case .v15ActiveHeadNavigation: "american_crow_active_head_navigation_development_v1"
        case .v15HierarchicalNavigation: "american_crow_v15_goal_memory_navigation_development_v1"
        case .v15WaypointExpertNavigation: "american_crow_v15_waypoint_expert_navigation_development_v1"
        case .v15DualWaypointExpertNavigation: "american_crow_v15_dual_waypoint_expert_navigation_development_v1"
        case .v15TripleWaypointExpertNavigation: "american_crow_v15_triple_waypoint_expert_navigation_development_v1"
        }
    }
}

public enum MetalRoboBirdFlowNavigationCourse: UInt32, Sendable {
    case training = 0
    case heldOutA = 1
    case heldOutB = 2
    case developmentReference = 3
    case reservedFinalC = 4
    case reservedFinalD = 5
    case reservedFinalE = 6
    case reservedFinalF = 7
    case reservedFinalG = 8
    case reservedFinalH = 9
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
    public var materializeArticulatedContactResponses: Bool
    public var difficultyBandRange: ClosedRange<UInt32>?
    public var difficultySamplingExponentOverride: Float
    public var interactionReferenceMode: MetalRoboInteractionReferenceMode
    public var interactionStudentAuthority: Float?
    public var interactionResetPhaseFraction: Float?
    public var interactionResetPhaseProbability: Float?
    public var interactionResetMaximumPhase: Float?
    public var birdFlowJourneyTeacher: Bool
    public var birdFlowJourneyStudentAuthority: Float
    public var birdFlowNavigationCurriculum: Bool
    public var birdFlowNavigationCurriculumStage: UInt32?
    public var birdFlowJourneyVariant: MetalRoboBirdFlowJourneyVariant
    public var birdFlowNavigationCourse: MetalRoboBirdFlowNavigationCourse
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
        materializeArticulatedContactResponses: Bool = false,
        difficultyBandRange: ClosedRange<UInt32>? = nil,
        difficultySamplingExponentOverride: Float = 0.0,
        interactionReferenceMode: MetalRoboInteractionReferenceMode =
            .taskDefault,
        interactionStudentAuthority: Float? = nil,
        interactionResetPhaseFraction: Float? = nil,
        interactionResetPhaseProbability: Float? = nil,
        interactionResetMaximumPhase: Float? = nil,
        birdFlowJourneyTeacher: Bool = false,
        birdFlowJourneyStudentAuthority: Float = 0.0,
        birdFlowNavigationCurriculum: Bool = false,
        birdFlowNavigationCurriculumStage: UInt32? = nil,
        birdFlowJourneyVariant: MetalRoboBirdFlowJourneyVariant =
            .v7Hierarchical,
        birdFlowNavigationCourse: MetalRoboBirdFlowNavigationCourse =
            .training,
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
        self.materializeArticulatedContactResponses =
            materializeArticulatedContactResponses
        self.difficultyBandRange = difficultyBandRange
        self.difficultySamplingExponentOverride =
            difficultySamplingExponentOverride
        self.interactionReferenceMode = interactionReferenceMode
        self.interactionStudentAuthority = interactionStudentAuthority
        self.interactionResetPhaseFraction = interactionResetPhaseFraction
        self.interactionResetPhaseProbability =
            interactionResetPhaseProbability
        self.interactionResetMaximumPhase = interactionResetMaximumPhase
        self.birdFlowJourneyTeacher = birdFlowJourneyTeacher
        self.birdFlowJourneyStudentAuthority =
            birdFlowJourneyStudentAuthority
        self.birdFlowNavigationCurriculum =
            birdFlowNavigationCurriculum
        self.birdFlowNavigationCurriculumStage =
            birdFlowNavigationCurriculumStage
        self.birdFlowJourneyVariant = birdFlowJourneyVariant
        self.birdFlowNavigationCourse = birdFlowNavigationCourse
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
    public let bodyCount: Int
    public let motionFeatureCount: Int
    public let maximumEpisodeSteps: Int
    public let worldFingerprint: UInt64
    public let taskFingerprint: UInt64
    public let observationFingerprint: UInt64
    public let actionFingerprint: UInt64
    public let runFingerprint: UInt64
    public let robotFingerprint: UInt64
    public let sensorFingerprint: UInt64
    public let realityFingerprint: UInt64
    public let teacherFingerprint: UInt64
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
        bodyCount = Int(native.body_count)
        motionFeatureCount = Int(native.motion_feature_count)
        maximumEpisodeSteps = Int(native.maximum_episode_steps)
        worldFingerprint = native.world_fingerprint
        taskFingerprint = native.task_fingerprint
        observationFingerprint = native.observation_fingerprint
        actionFingerprint = native.action_fingerprint
        runFingerprint = native.run_fingerprint
        robotFingerprint = native.robot_fingerprint
        sensorFingerprint = native.sensor_fingerprint
        realityFingerprint = native.reality_fingerprint
        teacherFingerprint = native.teacher_fingerprint
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
    // Generic fixed-width task-authored reward-contribution channels. Their
    // meanings come exclusively from outcomeSchema. Individual scalars keep
    // the transition four-byte aligned and avoid per-transition storage
    // padding in large Apple unified-memory rollout batches.
    public let taskOutcomeChannel0: Float
    public let taskOutcomeChannel1: Float
    public let taskOutcomeChannel2: Float
    public let taskOutcomeChannel3: Float
    public let taskOutcomeChannel4: Float
    public let taskOutcomeChannel5: Float
    public let taskOutcomeChannel6: Float
    public let taskOutcomeChannel7: Float
    public let policyRevision: UInt64
    public let timeoutBootstrapValue: Float
    public let episodeTrackingScore: Float
    public let difficultyBand: UInt32
    public let terrainLevel: UInt32
    public let impactSequenceIndex: UInt32
    public let impactEventFlags: UInt32
    // Sequential navigation state is part of the native transition ABI. Keep
    // it in the Swift value so PolicyRolloutPack serialization cannot silently
    // replace task-authored navigation outcomes with zeros.
    public let navigationStepProgress: Float
    public let navigationCumulativeProgress: Float
    public let navigationWaypointsReached: Float
    public let navigationCompletion: Float

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
        taskOutcomeChannel0 = native.task_outcome_channel_0
        taskOutcomeChannel1 = native.task_outcome_channel_1
        taskOutcomeChannel2 = native.task_outcome_channel_2
        taskOutcomeChannel3 = native.task_outcome_channel_3
        taskOutcomeChannel4 = native.task_outcome_channel_4
        taskOutcomeChannel5 = native.task_outcome_channel_5
        taskOutcomeChannel6 = native.task_outcome_channel_6
        taskOutcomeChannel7 = native.task_outcome_channel_7
        policyRevision = native.policy_revision
        timeoutBootstrapValue =
            native.timeout_bootstrap_value
        episodeTrackingScore =
            native.episode_tracking_score
        difficultyBand = native.difficulty_band
        terrainLevel = native.terrain_level
        impactSequenceIndex = native.impact_sequence_index
        impactEventFlags = native.impact_event_flags
        navigationStepProgress = native.navigation_step_progress
        navigationCumulativeProgress =
            native.navigation_cumulative_progress
        navigationWaypointsReached =
            native.navigation_waypoints_reached
        navigationCompletion = native.navigation_completion
    }
}

public struct MetalRoboTaskOutcomeDescriptor: Sendable {
    public let id: String
    public let unit: String
    public let direction: UInt32
}

public struct MetalRoboTaskEvidenceTelemetry: Sendable {
    public let controlSteps: UInt64
    public let evidenceWindows: UInt64
    public let pendingCompletedEpisodeCount: UInt32
    public let pendingTimeoutEpisodeCount: UInt32
    public let lastCompletedEpisodeCount: UInt32
    public let lastContactRate: UInt32
    public let lastCleanMissRate: UInt32
    public let lastBalanceFailureRate: UInt32
    public let lastMeanTrackingPerMillion: UInt32

    fileprivate init(_ native: MRTaskEvidenceTelemetryC) {
        controlSteps = native.control_steps
        evidenceWindows = native.evidence_windows
        pendingCompletedEpisodeCount =
            native.pending_completed_episode_count
        pendingTimeoutEpisodeCount =
            native.pending_timeout_episode_count
        lastCompletedEpisodeCount =
            native.last_completed_episode_count
        lastContactRate = native.last_contact_rate
        lastCleanMissRate = native.last_clean_miss_rate
        lastBalanceFailureRate = native.last_balance_failure_rate
        lastMeanTrackingPerMillion =
            native.last_mean_tracking_per_million
    }
}

public struct MetalRoboPolicyRolloutBatch: Sendable {
    public let controlStepCount: Int
    public let environmentCount: Int
    public let actorObservationCount: Int
    public let criticObservationCount: Int
    public let actionCount: Int
    public let motionFeatureCount: Int
    public let actorObservations: [Float]
    public let criticObservations: [Float]
    public let motionFeatures: [Float]
    public let teacherActions: [Float]
    public let latents: [Float]
    public let logProbabilities: [Float]
    public let values: [Float]
    public let transitions: [MetalRoboTaskTransition]
    public var policyMemoryWidth: Int = 0
    public var policyMemoryBefore: [Float] = []
    public var policyMemoryResets: [UInt32] = []

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
                $0.motionFeatureCount ==
                    first.motionFeatureCount &&
                $0.actionCount == first.actionCount &&
                $0.policyMemoryWidth == first.policyMemoryWidth &&
                $0.policyMemoryBefore.count == $0.sampleCount * $0.policyMemoryWidth &&
                $0.policyMemoryResets.count == ($0.policyMemoryWidth == 0 ? 0 : $0.sampleCount)
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
            motionFeatureCount: first.motionFeatureCount,
            actorObservations:
                batches.flatMap(\.actorObservations),
            criticObservations:
                batches.flatMap(\.criticObservations),
            motionFeatures: batches.flatMap(\.motionFeatures),
            teacherActions: batches.flatMap(\.teacherActions),
            latents: batches.flatMap(\.latents),
            logProbabilities:
                batches.flatMap(\.logProbabilities),
            values: batches.flatMap(\.values),
            transitions: batches.flatMap(\.transitions),
            policyMemoryWidth: first.policyMemoryWidth,
            policyMemoryBefore: batches.flatMap(\.policyMemoryBefore),
            policyMemoryResets: batches.flatMap(\.policyMemoryResets)
        )
    }
}

/// Swift owns rollout chunking and policy revisions. Native code owns the
/// compiled world and task program, persistent private Metal state, reset
/// transaction, and compact learning publication.
public enum MetalRoboRunSource: Sendable {
    case unitreeG1
    case frankaPickPlace
    case px4X500
    case birdFlowDove
    case birdFlowAmericanCrow
    case birdFlowAmericanCrowJourney
    case importedURDF(
        urdf: URL, srdf: URL?, taskPack: URL, actuatorPack: URL,
        sensorPack: URL, realityPack: URL
    )
    case worldPack(
        world: URL, taskPack: URL, actuatorPack: URL,
        sensorPack: URL, realityPack: URL
    )
}

public struct MetalRoboTeacherSource: Sendable {
    public let pack: URL
    public let clipID: String

    public init(pack: URL, clipID: String) {
        self.pack = pack
        self.clipID = clipID
    }
}

public struct MetalRoboRunManifest: Sendable {
    public let source: MetalRoboRunSource
    public let sensorsAndPhysics: MetalRoboTaskRolloutConfiguration
    public let visualSensor:
        MetalRoboTaskVisualObservationConfiguration?
    // Presentation-only renderer. This uses the same authored visual schema
    // without entering the SensorPack or policy observation contract.
    public let inspectionVisual:
        MetalRoboTaskVisualObservationConfiguration?
    public let teacher: MetalRoboTeacherSource?
    public let objectPushCurriculum: MetalRoboCrowObjectPushCurriculum?

    public init(
        source: MetalRoboRunSource,
        sensorsAndPhysics: MetalRoboTaskRolloutConfiguration,
        visualSensor: MetalRoboTaskVisualObservationConfiguration? = nil,
        inspectionVisual: MetalRoboTaskVisualObservationConfiguration? = nil,
        teacher: MetalRoboTeacherSource? = nil,
        objectPushCurriculum: MetalRoboCrowObjectPushCurriculum? = nil
    ) {
        self.source = source
        self.sensorsAndPhysics = sensorsAndPhysics
        self.visualSensor = visualSensor
        self.inspectionVisual = inspectionVisual
        self.teacher = teacher
        self.objectPushCurriculum = objectPushCurriculum
    }
}

public struct MetalRoboTaskInspectionFrame: @unchecked Sendable {
    public let rgbBuffer: MTLBuffer
    public let slotIndex: UInt32
    public let width: Int
    public let height: Int
    public let frameIndex: UInt64
    public let submissionIndex: UInt64
    public let environmentIndex: UInt32
    public let droppedFrames: UInt32
}

public final class MetalRoboTaskRolloutContext: @unchecked Sendable {
    private var handle: OpaquePointer?

    public init(
        manifest authored: MetalRoboRunManifest,
        metallibPath: String? = nil
    ) throws {
        if let teacher = authored.teacher, teacher.clipID.isEmpty {
            throw MetalRoboTaskRolloutError.invalidShape(
                "InteractionPack clip identity cannot be empty."
            )
        }
        var retinal = try authored.visualSensor?.retinal?.native()
        var foraging = try authored.visualSensor?.cuedForaging?.native()
        var objectCurriculum = try authored.objectPushCurriculum?.native()
        guard objectCurriculum == nil || (retinal != nil && foraging == nil) else {
            throw MetalRoboTaskRolloutError.invalidShape("Object curriculum requires a resource-free retinal sensor.")
        }
        guard foraging == nil || retinal != nil else {
            throw MetalRoboTaskRolloutError.invalidShape("Cued foraging requires an explicit retinal program.")
        }
        let configuration = authored.sensorsAndPhysics
        var source = MR_RUN_SOURCE_UNITREE_G1.rawValue
        var urdfPath: String?
        var srdfPath: String?
        var worldPackPath: String?
        var taskPackPath: String?
        var actuatorPackPath: String?
        var sensorPackPath: String?
        var realityPackPath: String?
        switch authored.source {
        case .unitreeG1:
            source = MR_RUN_SOURCE_UNITREE_G1.rawValue
        case .frankaPickPlace:
            source = MR_RUN_SOURCE_FRANKA_PICK_PLACE.rawValue
        case .px4X500:
            source = MR_RUN_SOURCE_PX4_X500.rawValue
        case .birdFlowDove:
            source = MR_RUN_SOURCE_BIRDFLOW_DOVE.rawValue
        case .birdFlowAmericanCrow:
            source = MR_RUN_SOURCE_BIRDFLOW_AMERICAN_CROW.rawValue
        case .birdFlowAmericanCrowJourney:
            source = MR_RUN_SOURCE_BIRDFLOW_AMERICAN_CROW_JOURNEY.rawValue
        case let .importedURDF(
            urdf, srdf, taskPack, actuatorPack, sensorPack, realityPack
        ):
            source = MR_RUN_SOURCE_IMPORTED_URDF.rawValue
            urdfPath = urdf.path
            srdfPath = srdf?.path
            taskPackPath = taskPack.path
            actuatorPackPath = actuatorPack.path
            sensorPackPath = sensorPack.path
            realityPackPath = realityPack.path
        case let .worldPack(
            world, taskPack, actuatorPack, sensorPack, realityPack
        ):
            source = MR_RUN_SOURCE_WORLD_PACK.rawValue
            worldPackPath = world.path
            taskPackPath = taskPack.path
            actuatorPackPath = actuatorPack.path
            sensorPackPath = sensorPack.path
            realityPackPath = realityPack.path
        }
        let created: OpaquePointer? = try Self.withNativeVisualSensor(
            authored.visualSensor
        ) { visualSensor in
            try Self.withNativeVisualSensor(
                authored.inspectionVisual
            ) { inspectionVisual in
            Self.withNativeConfiguration(configuration) { config in
                withOptionalCString(urdfPath) { urdf in
                    withOptionalCString(srdfPath) { srdf in
                        withOptionalCString(worldPackPath) { world in
                            withOptionalCString(taskPackPath) { task in
                                withOptionalCString(actuatorPackPath) {
                                    actuatorPack in
                                withOptionalCString(sensorPackPath) {
                                    sensorPack in
                                withOptionalCString(realityPackPath) {
                                    realityPack in
                                withOptionalCString(
                                    authored.teacher?.pack.path
                                ) { teacherPack in
                                    withOptionalCString(
                                        authored.teacher?.clipID
                                    ) { teacherClip in
                                        withOptionalCString(metallibPath) {
                                            metallib in
                                            var manifest = MRRunManifestC()
                                            manifest.profile = config.pointee
                                            manifest.source = source
                                            manifest.surface =
                                                configuration.surface.rawValue
                                            manifest.task = configuration
                                                .unitreeG1Task.rawValue
                                            manifest.urdf_path = urdf
                                            manifest.srdf_path = srdf
                                            manifest.world_pack_path = world
                                            manifest.task_pack_path = task
                                            manifest.robot_actuator_pack_path =
                                                actuatorPack
                                            manifest.sensor_program_pack_path =
                                                sensorPack
                                            manifest.reality_program_pack_path =
                                                realityPack
                                            manifest.teacher_pack_path =
                                                teacherPack
                                            manifest.teacher_clip_id =
                                                teacherClip
                                            manifest.visual_sensor_program =
                                                visualSensor
                                            manifest.inspection_visual_program =
                                                inspectionVisual
                                            manifest.metallib_path = metallib
                                            if retinal != nil {
                                                return withUnsafePointer(to: &retinal!) { sensor in
                                                    if objectCurriculum != nil {
                                                        return withUnsafePointer(to: &objectCurriculum!) { curriculum in
                                                            mr_create_task_rollout_object_push_curriculum(&manifest, sensor, curriculum)
                                                        }
                                                    }
                                                    if foraging != nil {
                                                        return withUnsafePointer(to: &foraging!) { resources in
                                                            if let decision = authored.visualSensor?.cuedForaging?.memoryChoiceDecisionTick {
                                                                if let head = authored.visualSensor?.cuedForaging?.headYawHoldTicks,
                                                                   let beak = authored.visualSensor?.cuedForaging?.beakHoldTicks {
                                                                    return mr_create_task_rollout_cued_memory_foraging_holds(&manifest, sensor, resources, decision, head, beak)
                                                                }
                                                                return mr_create_task_rollout_cued_memory_foraging(&manifest, sensor, resources, decision)
                                                            }
                                                            return mr_create_task_rollout_cued_foraging(&manifest, sensor, resources)
                                                        }
                                                    }
                                                    return mr_create_task_rollout_retinal(&manifest, sensor)
                                                }
                                            }
                                            return mr_create_task_rollout(&manifest)
                                        }
                                    }
                                }
                                }
                                }
                                }
                            }
                        }
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
        native.materialize_articulated_contact_responses =
            configuration.materializeArticulatedContactResponses ? 1 : 0
        if let range = configuration.difficultyBandRange {
            native.override_difficulty_band_range = 1
            native.minimum_difficulty_band = range.lowerBound
            native.maximum_difficulty_band = range.upperBound
        }
        native.interaction_reference_mode =
            configuration.interactionReferenceMode.rawValue
        if let authority = configuration.interactionStudentAuthority {
            native.interaction_student_authority = authority
            native.override_interaction_student_authority = 1
        }
        if let fraction = configuration.interactionResetPhaseFraction {
            native.interaction_reset_phase_fraction = fraction
            native.override_interaction_reset_phase_fraction = 1
        }
        if let probability =
            configuration.interactionResetPhaseProbability
        {
            native.interaction_reset_phase_probability = probability
            native.override_interaction_reset_phase_probability = 1
        }
        if let maximumPhase = configuration.interactionResetMaximumPhase {
            native.interaction_reset_maximum_phase = maximumPhase
            native.override_interaction_reset_maximum_phase = 1
        }
        native.birdflow_journey_teacher =
            configuration.birdFlowJourneyTeacher ? 1 : 0
        native.birdflow_journey_student_authority =
            configuration.birdFlowJourneyStudentAuthority
        if let stage = configuration.birdFlowNavigationCurriculumStage {
            native.birdflow_navigation_curriculum =
                stage <= 4 ? stage + 2 : UInt32.max
        } else {
            native.birdflow_navigation_curriculum =
                configuration.birdFlowNavigationCurriculum ? 1 : 0
        }
        native.birdflow_journey_variant =
            configuration.birdFlowJourneyVariant.rawValue
        native.birdflow_navigation_course =
            configuration.birdFlowNavigationCourse.rawValue
        native.difficulty_sampling_exponent_override =
            configuration.difficultySamplingExponentOverride
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

    private static func withNativeVisualSensor<Result>(
        _ configuration:
            MetalRoboTaskVisualObservationConfiguration?,
        _ body: (
            UnsafePointer<MRTaskVisualObservationConfigC>?
        ) throws -> Result
    ) throws -> Result {
        guard let configuration else {
            return try body(nil)
        }
        guard !configuration.packs.isEmpty,
              configuration.width > 0,
              configuration.height > 0,
              configuration.minimumVisiblePixels > 0
        else {
            throw MetalRoboTaskRolloutError.invalidShape(
                "Visual SensorPack requires packs and nonzero dimensions."
            )
        }
        var strings: [String] = []
        strings.reserveCapacity(2 * configuration.packs.count + 1)
        for pack in configuration.packs {
            strings.append(pack.url.path)
            strings.append(pack.assetID)
        }
        strings.append(configuration.cameraParentBody)
        return try withUnsafeCStringBuffers(strings) { pointers in
            var nativePacks: [MRTaskVisualPackC] = []
            nativePacks.reserveCapacity(configuration.packs.count)
            for index in configuration.packs.indices {
                var pack = MRTaskVisualPackC()
                pack.path = pointers[2 * index]
                pack.asset_id = pointers[2 * index + 1]
                pack.semantic_id = configuration.packs[index].semanticID
                pack.instance_id = configuration.packs[index].instanceID
                nativePacks.append(pack)
            }
            return try nativePacks.withUnsafeBufferPointer { packs in
                try withOptionalCString(
                    configuration.environmentPackURL?.path
                ) { environment in
                    try "sensor_fast".withCString { profile in
                        var native = MRTaskVisualObservationConfigC()
                        native.packs = packs.baseAddress
                        native.pack_count = UInt32(packs.count)
                        native.environment_pack_path = environment
                        native.renderer_profile = profile
                        native.camera_parent_body = pointers.last!
                        native.camera_position = (
                            configuration.cameraPosition.x,
                            configuration.cameraPosition.y,
                            configuration.cameraPosition.z
                        )
                        native.camera_orientation = (
                            configuration.cameraOrientation.x,
                            configuration.cameraOrientation.y,
                            configuration.cameraOrientation.z,
                            configuration.cameraOrientation.w
                        )
                        native.width = configuration.width
                        native.height = configuration.height
                        native.minimum_visible_pixels =
                            configuration.minimumVisiblePixels
                        native.vertical_field_of_view_degrees =
                            configuration.verticalFieldOfViewDegrees
                        native.nominal_rate_hz = configuration.nominalRateHz
                        native.maximum_retained_bytes =
                            configuration.maximumRetainedBytes
                        native.capture_width = configuration.captureWidth
                        native.capture_height = configuration.captureHeight
                        native.capture_policy_camera =
                            configuration.capturePolicyCamera ? 1 : 0
                        return try withUnsafePointer(to: &native, body)
                    }
                }
            }
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

    public var taskID: String {
        guard let value = mr_task_rollout_task_id(handle) else {
            return "unknown"
        }
        return String(cString: value)
    }

    public var deviceName: String {
        String(cString: mr_task_rollout_device_name(handle))
    }

    public var visualSceneFingerprint: UInt64 {
        mr_task_rollout_visual_scene_fingerprint(handle)
    }

    public var impactEventCount: UInt32 {
        mr_task_rollout_impact_event_count(handle)
    }

    public func reset(seed: UInt64) throws {
        guard mr_task_rollout_reset(handle, seed) == 0 else {
            throw MetalRoboTaskRolloutError.native(
                Self.lastError()
            )
        }
    }

    public func evidenceTelemetry() throws
        -> MetalRoboTaskEvidenceTelemetry
    {
        var native = MRTaskEvidenceTelemetryC()
        guard mr_task_rollout_evidence_telemetry(
            handle,
            &native
        ) == 0 else {
            throw MetalRoboTaskRolloutError.native(
                Self.lastError()
            )
        }
        return MetalRoboTaskEvidenceTelemetry(native)
    }

    public func setStateReadback(_ enabled: Bool) throws {
        guard mr_task_rollout_set_state_readback(
            handle,
            enabled ? 1 : 0
        ) == 0 else {
            throw MetalRoboTaskRolloutError.native(Self.lastError())
        }
    }

    // One publication readback for conservation/behavior evidence. Never part
    // of policy inference or per-control-step observation construction.
    public func acceptedLifetimeControlTicks() throws -> [UInt64] {
        var ticks = [UInt64](repeating: 0, count: layout.environmentCount)
        let status = ticks.withUnsafeMutableBufferPointer {
            mr_task_rollout_copy_lifetime_ticks(handle, $0.baseAddress, $0.count)
        }
        guard status == 0 else {
            throw MetalRoboTaskRolloutError.native(Self.lastError())
        }
        return ticks
    }

    public func finiteResourceAccounting() throws -> [MRTaskResourceAccountingC] {
        var records = [MRTaskResourceAccountingC](repeating: MRTaskResourceAccountingC(),
                                                count: layout.environmentCount)
        let status = records.withUnsafeMutableBufferPointer {
            mr_task_rollout_copy_resource_accounting(handle, $0.baseAddress, $0.count)
        }
        guard status == 0 else {
            throw MetalRoboTaskRolloutError.native(Self.lastError())
        }
        return records
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
        native.contract_version = policy.contractVersion
        native.world_fingerprint = policy.worldFingerprint
        native.task_fingerprint = policy.taskFingerprint
        native.observation_fingerprint =
            policy.observationFingerprint
        native.action_fingerprint = policy.actionFingerprint
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

    public func saveLifeCheckpoint(to url: URL) throws {
        let status = url.path.withCString { mr_task_rollout_save_life_checkpoint(handle, $0) }
        guard status == 0 else { throw MetalRoboTaskRolloutError.native(Self.lastError()) }
    }

    // Requires a fresh context with the exact saved native policy installed.
    public func restoreLifeCheckpoint(from url: URL) throws {
        let status = url.path.withCString { mr_task_rollout_restore_life_checkpoint(handle, $0) }
        guard status == 0 else { throw MetalRoboTaskRolloutError.native(Self.lastError()) }
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

    public func loadBasePolicy(at url: URL) throws {
        let status = url.path.withCString {
            mr_task_rollout_load_base_policy_pack(handle, $0)
        }
        guard status == 0 else {
            throw MetalRoboTaskRolloutError.native(
                Self.lastError()
            )
        }
    }

    public var installedPolicyRevision: UInt64 {
        mr_task_rollout_policy_revision(handle)
    }

    public func clearPolicy() throws {
        guard mr_task_rollout_clear_policy(handle) == 0 else {
            throw MetalRoboTaskRolloutError.native(
                Self.lastError()
            )
        }
    }

    public func visualRGBA() throws -> (
        width: Int,
        height: Int,
        values: [Float]
    ) {
        var width: UInt32 = 0
        var height: UInt32 = 0
        let count = mr_task_rollout_copy_visual_rgba(
            handle,
            nil,
            0,
            &width,
            &height
        )
        guard count > 0 else {
            throw MetalRoboTaskRolloutError.native(
                Self.lastError()
            )
        }
        var values = [Float](repeating: 0, count: count)
        let copied = values.withUnsafeMutableBufferPointer {
            mr_task_rollout_copy_visual_rgba(
                handle,
                $0.baseAddress,
                $0.count,
                &width,
                &height
            )
        }
        guard copied == count else {
            throw MetalRoboTaskRolloutError.native(
                Self.lastError()
            )
        }
        return (Int(width), Int(height), values)
    }

    // The returned buffer remains private GPU memory. Present it directly and
    // release the slot only from the presentation command buffer completion
    // handler; this never copies pixels through Swift or the CPU.
    public func acquireInspectionFrame() throws
        -> MetalRoboTaskInspectionFrame?
    {
        var native = MRTaskInspectionFrameC()
        let status = mr_task_rollout_acquire_inspection_frame(
            handle,
            &native
        )
        if status == 0 {
            return nil
        }
        guard status == 1, let opaque = native.rgb_buffer else {
            throw MetalRoboTaskRolloutError.native(Self.lastError())
        }
        let object = Unmanaged<AnyObject>.fromOpaque(opaque)
            .takeUnretainedValue()
        guard let buffer = object as? MTLBuffer else {
            throw MetalRoboTaskRolloutError.native(
                "inspection frame did not return a Metal buffer."
            )
        }
        return MetalRoboTaskInspectionFrame(
            rgbBuffer: buffer,
            slotIndex: native.slot_index,
            width: Int(native.width),
            height: Int(native.height),
            frameIndex: native.frame_index,
            submissionIndex: native.submission_index,
            environmentIndex: native.environment_index,
            droppedFrames: native.dropped_frames
        )
    }

    public func setInspectionEnabled(_ enabled: Bool) throws {
        guard mr_task_rollout_set_inspection_enabled(
            handle,
            enabled ? 1 : 0
        ) == 0 else {
            throw MetalRoboTaskRolloutError.native(Self.lastError())
        }
    }

    public func releaseInspectionFrame(slotIndex: UInt32) {
        _ = mr_task_rollout_release_inspection_frame(handle, slotIndex)
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

    public func motionFeatures(
        controlStepCount: Int
    ) throws -> [Float] {
        let current = layout
        let count = controlStepCount * current.environmentCount *
            current.motionFeatureCount
        guard count == 0 ||
                mr_task_rollout_motion_features(handle) != nil
        else {
            throw MetalRoboTaskRolloutError.native(
                "Task motion-feature stream is unavailable."
            )
        }
        guard count != 0 else { return [] }
        return Array(
            UnsafeBufferPointer(
                start: mr_task_rollout_motion_features(handle)!,
                count: count
            )
        )
    }

    public func teacherActions(
        controlStepCount: Int
    ) -> [Float] {
        let count = controlStepCount * layout.environmentCount *
            layout.actionCount
        guard let values =
                mr_task_rollout_teacher_actions(handle)
        else {
            return []
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

    public var outcomeSchema: [MetalRoboTaskOutcomeDescriptor] {
        let count = Int(mr_task_rollout_outcome_count(handle))
        return (0..<count).compactMap { index in
            guard let id = mr_task_rollout_outcome_id(
                    handle, UInt32(index)
                  ),
                  let unit = mr_task_rollout_outcome_unit(
                    handle, UInt32(index)
                  )
            else { return nil }
            return MetalRoboTaskOutcomeDescriptor(
                id: String(cString: id),
                unit: String(cString: unit),
                direction: mr_task_rollout_outcome_direction(
                    handle, UInt32(index)
                )
            )
        }
    }

    // Latest submission values packed [transition][outcome]. The schema is
    // compiled from the selected task; consumers do not assume G1 or dodge.
    public func outcomeValues(
        controlStepCount: Int
    ) throws -> [Float] {
        let count = controlStepCount * layout.environmentCount *
            outcomeSchema.count
        guard count == 0 || mr_task_rollout_outcome_values(handle) != nil
        else {
            throw MetalRoboTaskRolloutError.native(
                "Task outcome stream is unavailable."
            )
        }
        guard count != 0 else { return [] }
        return Array(UnsafeBufferPointer(
            start: mr_task_rollout_outcome_values(handle)!,
            count: count
        ))
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

    public func policyActions(
        controlStepCount: Int
    ) throws -> [Float] {
        let current = layout
        let count =
            controlStepCount *
            current.environmentCount *
            current.actionCount
        guard let values =
                mr_task_rollout_policy_actions(handle)
        else {
            throw MetalRoboTaskRolloutError.native(
                "Policy action stream is unavailable."
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

    public func finalVelocity() throws -> [Float] {
        let count = layout.environmentCount * layout.velocityCount
        guard let values = mr_task_rollout_final_v(handle) else {
            throw MetalRoboTaskRolloutError.native(
                "Final velocity is unavailable."
            )
        }
        return Array(UnsafeBufferPointer(start: values, count: count))
    }

    public var bodyNames: [String] {
        (0..<layout.bodyCount).compactMap { index in
            mr_task_rollout_body_name(handle, UInt32(index)).map(String.init(cString:))
        }
    }

    public func finalBodyStates() throws -> [Float] {
        let current = layout
        let count = current.environmentCount * current.bodyCount * 13
        var output = [Float](repeating: 0, count: count)
        let status = output.withUnsafeMutableBufferPointer { buffer in
            mr_task_rollout_copy_final_body_states(
                handle, buffer.baseAddress, buffer.count
            )
        }
        guard status == 0 else {
            throw MetalRoboTaskRolloutError.native(Self.lastError())
        }
        return output
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

    public var policyMemoryWidth: Int { Int(mr_task_rollout_policy_memory_width(handle)) }

    public func policyMemoryBefore(controlStepCount: Int) throws -> [Float] {
        if policyMemoryWidth == 0 { return [] }
        guard controlStepCount > 0,
              controlStepCount == Int(mr_task_rollout_policy_memory_control_step_count(handle)),
              let values = mr_task_rollout_policy_memory_before(handle) else {
            throw MetalRoboTaskRolloutError.invalidShape("Recurrent trace must cover exactly the latest real native control steps.")
        }
        return Array(UnsafeBufferPointer(start: values, count: controlStepCount * layout.environmentCount * policyMemoryWidth))
    }
    public func policyMemoryResets(controlStepCount: Int) throws -> [UInt32] {
        if policyMemoryWidth == 0 { return [] }
        guard controlStepCount > 0,
              controlStepCount == Int(mr_task_rollout_policy_memory_control_step_count(handle)),
              let values = mr_task_rollout_policy_memory_resets(handle) else {
            throw MetalRoboTaskRolloutError.invalidShape("Recurrent reset trace must cover exactly the latest real native control steps.")
        }
        return Array(UnsafeBufferPointer(start: values, count: controlStepCount * layout.environmentCount))
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
            motionFeatureCount: current.motionFeatureCount,
            actorObservations: try actorObservations(
                controlStepCount: controlStepCount
            ),
            criticObservations: try criticObservations(
                controlStepCount: controlStepCount
            ),
            motionFeatures: try motionFeatures(
                controlStepCount: controlStepCount
            ),
            teacherActions: teacherActions(
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
            ),
            policyMemoryWidth: policyMemoryWidth,
            policyMemoryBefore: try policyMemoryBefore(controlStepCount: controlStepCount),
            policyMemoryResets: try policyMemoryResets(controlStepCount: controlStepCount)
        )
    }

    // Training hot path: append the current native readback directly into one
    // caller-owned update arena. This avoids materializing a second complete
    // Swift chunk before the append.
    public func appendCurrentPolicyRollout(
        controlStepCount: Int,
        actorObservations: inout [Float],
        criticObservations: inout [Float],
        motionFeatures: inout [Float],
        teacherActions: inout [Float],
        latents: inout [Float],
        logProbabilities: inout [Float],
        values: inout [Float],
        transitions: inout [MetalRoboTaskTransition]
    ) throws {
        guard policyMemoryWidth == 0 else {
            throw MetalRoboTaskRolloutError.invalidShape("Recurrent rollout append requires explicit hidden/reset destinations.")
        }
        var before: [Float] = [], resets: [UInt32] = []
        try appendCurrentPolicyRollout(controlStepCount: controlStepCount,
            actorObservations: &actorObservations, criticObservations: &criticObservations,
            motionFeatures: &motionFeatures, teacherActions: &teacherActions, latents: &latents,
            logProbabilities: &logProbabilities, values: &values, transitions: &transitions,
            policyMemoryBefore: &before, policyMemoryResets: &resets)
    }

    public func appendCurrentPolicyRollout(
        controlStepCount: Int,
        actorObservations: inout [Float],
        criticObservations: inout [Float],
        motionFeatures: inout [Float],
        teacherActions: inout [Float],
        latents: inout [Float],
        logProbabilities: inout [Float],
        values: inout [Float],
        transitions: inout [MetalRoboTaskTransition],
        policyMemoryBefore: inout [Float],
        policyMemoryResets: inout [UInt32]
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
        let motionCount = samples * current.motionFeatureCount
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
        if policyMemoryWidth != 0 {
            guard controlStepCount == Int(mr_task_rollout_policy_memory_control_step_count(handle)),
                  let before = mr_task_rollout_policy_memory_before(handle),
                  let resets = mr_task_rollout_policy_memory_resets(handle) else {
                throw MetalRoboTaskRolloutError.invalidShape("Recurrent append must exclude previews and include all real native steps.")
            }
            policyMemoryBefore.append(contentsOf: UnsafeBufferPointer(start: before, count: samples * policyMemoryWidth))
            policyMemoryResets.append(contentsOf: UnsafeBufferPointer(start: resets, count: samples))
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
        if motionCount != 0 {
            guard let motion = mr_task_rollout_motion_features(handle)
            else {
                throw MetalRoboTaskRolloutError.native(
                    "Native motion-feature stream is unavailable."
                )
            }
            motionFeatures.append(
                contentsOf: UnsafeBufferPointer(
                    start: motion,
                    count: motionCount
                )
            )
        }
        if let teacher = mr_task_rollout_teacher_actions(handle) {
            teacherActions.append(
                contentsOf: UnsafeBufferPointer(
                    start: teacher,
                    count: latentCount
                )
            )
        }
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
              batch.motionFeatureCount ==
                  current.motionFeatureCount,
              bootstrapValues.count ==
                  current.environmentCount,
              batch.actorObservations.count ==
                  batch.sampleCount *
                  batch.actorObservationCount,
              batch.criticObservations.count ==
                  batch.sampleCount *
                  batch.criticObservationCount,
              batch.motionFeatures.count ==
                  batch.sampleCount *
                  batch.motionFeatureCount,
              batch.teacherActions.isEmpty ||
                  batch.teacherActions.count ==
                  batch.sampleCount * batch.actionCount,
              batch.latents.count ==
                  batch.sampleCount * batch.actionCount,
              batch.logProbabilities.count ==
                  batch.sampleCount,
              batch.values.count == batch.sampleCount,
              batch.transitions.count == batch.sampleCount,
              batch.policyMemoryWidth >= 0, batch.policyMemoryWidth <= 4096,
              batch.policyMemoryBefore.count == batch.sampleCount * batch.policyMemoryWidth,
              batch.policyMemoryResets.count == (batch.policyMemoryWidth == 0 ? 0 : batch.sampleCount)
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
        native.motion_feature_count = batch.motionFeatures.count
        native.teacher_action_count = batch.teacherActions.count
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
            value.task_outcome_channel_0 =
                transition.taskOutcomeChannel0
            value.task_outcome_channel_1 =
                transition.taskOutcomeChannel1
            value.task_outcome_channel_2 =
                transition.taskOutcomeChannel2
            value.task_outcome_channel_3 =
                transition.taskOutcomeChannel3
            value.task_outcome_channel_4 =
                transition.taskOutcomeChannel4
            value.task_outcome_channel_5 =
                transition.taskOutcomeChannel5
            value.task_outcome_channel_6 =
                transition.taskOutcomeChannel6
            value.task_outcome_channel_7 =
                transition.taskOutcomeChannel7
            value.policy_revision =
                transition.policyRevision
            value.timeout_bootstrap_value =
                transition.timeoutBootstrapValue
            value.episode_tracking_score =
                transition.episodeTrackingScore
            value.difficulty_band =
                transition.difficultyBand
            value.terrain_level = transition.terrainLevel
            value.impact_sequence_index =
                transition.impactSequenceIndex
            value.impact_event_flags =
                transition.impactEventFlags
            value.navigation_step_progress =
                transition.navigationStepProgress
            value.navigation_cumulative_progress =
                transition.navigationCumulativeProgress
            value.navigation_waypoints_reached =
                transition.navigationWaypointsReached
            value.navigation_completion =
                transition.navigationCompletion
            return value
        }
        let status = withUnsafeFloatBuffers(
            [
                batch.actorObservations,
                batch.criticObservations,
                batch.motionFeatures,
                batch.teacherActions,
                batch.latents,
                batch.logProbabilities,
                batch.values,
                bootstrapValues,
                batch.policyMemoryBefore,
            ]
        ) { buffers in
            native.actor_observations = buffers[0].baseAddress
            native.critic_observations = buffers[1].baseAddress
            native.motion_features = buffers[2].baseAddress
            native.teacher_actions = buffers[3].baseAddress
            native.latents = buffers[4].baseAddress
            native.log_probabilities = buffers[5].baseAddress
            native.values = buffers[6].baseAddress
            native.bootstrap_values = buffers[7].baseAddress
            return nativeTransitions.withUnsafeBufferPointer {
                transitions in
                native.transitions = transitions.baseAddress
                native.transition_count = transitions.count
                return id.withCString { identifier in
                    url.path.withCString { path in
                        if batch.policyMemoryWidth == 0 {
                            return mr_task_rollout_write_policy_rollout_pack(handle, &native, identifier, path)
                        }
                        return batch.policyMemoryResets.withUnsafeBufferPointer { resets in
                            var memory = MRPolicyMemoryBatchC()
                            memory.abi_version = 1
                            memory.struct_size = UInt32(MemoryLayout<MRPolicyMemoryBatchC>.size)
                            memory.hidden_count = UInt32(batch.policyMemoryWidth)
                            memory.before = buffers[8].baseAddress
                            memory.before_count = batch.policyMemoryBefore.count
                            memory.resets = resets.baseAddress
                            memory.reset_count = resets.count
                            return mr_task_rollout_write_recurrent_policy_rollout_pack(handle, &native, &memory, identifier, path)
                        }
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
