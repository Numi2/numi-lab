import Foundation
import Metal
import MetalRoboC

public enum MetalRoboTactileError: Error, CustomStringConvertible {
    case native(String)
    case invalidMetadata
    case unavailableBuffer(UInt32)

    public var description: String {
        switch self {
        case .native(let message):
            return message
        case .invalidMetadata:
            return "MetalRobo returned invalid tactile observation metadata."
        case .unavailableBuffer(let kind):
            return "MetalRobo tactile buffer \(kind) is unavailable."
        }
    }
}

public struct MetalRoboTactileLayout: Sendable {
    public let capacity: UInt32
    public let activeEnvironmentCount: UInt32
    public let bodyCount: UInt32
    public let sensorCount: UInt32
    public let sampleCount: UInt32
    public let contactCapacityPerEnvironment: UInt32
    public let retainedBytes: Int
    public let bytesPerEnvironment: Int
    public let hardwareRayQueriesAvailable: Bool

    init(_ value: MRTactileLayoutC) {
        capacity = value.capacity
        activeEnvironmentCount = value.active_environment_count
        bodyCount = value.body_count
        sensorCount = value.sensor_count
        sampleCount = value.sample_count
        contactCapacityPerEnvironment =
            value.contact_capacity_per_environment
        retainedBytes = value.retained_bytes
        bytesPerEnvironment = value.bytes_per_environment
        hardwareRayQueriesAvailable =
            value.hardware_ray_queries_available != 0
    }
}

public enum MetalRoboTactileBuffer: UInt32, Sendable {
    case penetrationDepth = 0
    case depthVelocity = 1
    case validity = 2
    case objectShapeIDs = 3
    case debugHits = 4
    case summaries = 5
    case statuses = 6
    case tangentialMotion = 7
}

/// Swift owner for the canonical Franka metric tactile observation context.
///
/// `encode` borrows all Metal objects. It performs no command-buffer commit,
/// synchronization, readback, or per-frame allocation.
public final class MetalRoboTactileContext {
    private var handle: OpaquePointer?

    public init(
        capacity: UInt32,
        contactCapacityPerEnvironment: UInt32 = 128,
        metallibPath: String? = nil
    ) throws {
        let created: OpaquePointer? = metallibPath?.withCString { path in
            mr_tactile_create_franka(
                capacity,
                contactCapacityPerEnvironment,
                path
            )
        } ?? mr_tactile_create_franka(
            capacity,
            contactCapacityPerEnvironment,
            nil
        )
        guard let created else {
            throw MetalRoboTactileError.native(Self.lastError())
        }
        handle = created
    }

    deinit {
        if let handle {
            mr_tactile_destroy(handle)
        }
    }

    public var layout: MetalRoboTactileLayout {
        MetalRoboTactileLayout(mr_tactile_layout(handle))
    }

    public var deviceName: String {
        String(cString: mr_tactile_device_name(handle))
    }

    public var observationMetadata: [String: Any] {
        get throws {
            guard
                let pointer = mr_tactile_observation_metadata_json(handle),
                let data = String(cString: pointer).data(using: .utf8),
                let object = try JSONSerialization.jsonObject(with: data)
                    as? [String: Any]
            else {
                throw MetalRoboTactileError.invalidMetadata
            }
            return object
        }
    }

    public func buffer(
        _ kind: MetalRoboTactileBuffer
    ) throws -> MTLBuffer {
        guard
            let pointer = mr_tactile_native_buffer(handle, kind.rawValue)
        else {
            throw MetalRoboTactileError.unavailableBuffer(kind.rawValue)
        }
        let object = Unmanaged<AnyObject>
            .fromOpaque(pointer)
            .takeUnretainedValue()
        guard let buffer = object as? MTLBuffer else {
            throw MetalRoboTactileError.unavailableBuffer(kind.rawValue)
        }
        return buffer
    }

    public func encode(
        bodyStates: MTLBuffer,
        contacts: MTLBuffer? = nil,
        contactCounts: MTLBuffer? = nil,
        resetMask: MTLBuffer? = nil,
        environmentCount: UInt32,
        observationTimestepSeconds: Float,
        contactImpulseTimestepSeconds: Float = 0,
        frameIndex: UInt64,
        timestampSeconds: Double,
        commandEncoder: MTLComputeCommandEncoder
    ) throws {
        guard (contacts == nil) == (contactCounts == nil) else {
            throw MetalRoboTactileError.native(
                "contacts and contactCounts must be supplied together."
            )
        }
        let currentLayout = layout
        let status = mr_tactile_encode(
            handle,
            Self.unretained(bodyStates),
            contacts.map(Self.unretained),
            contactCounts.map(Self.unretained),
            resetMask.map(Self.unretained),
            environmentCount,
            currentLayout.bodyCount,
            contacts == nil
                ? 0
                : currentLayout.contactCapacityPerEnvironment,
            observationTimestepSeconds,
            contactImpulseTimestepSeconds,
            frameIndex,
            timestampSeconds,
            Self.unretained(commandEncoder)
        )
        guard status == 0 else {
            throw MetalRoboTactileError.native(Self.lastError())
        }
    }

    private static func unretained(_ object: AnyObject) -> UnsafeMutableRawPointer {
        Unmanaged.passUnretained(object).toOpaque()
    }

    private static func lastError() -> String {
        guard let pointer = mr_last_error() else {
            return "MetalRobo tactile operation failed."
        }
        return String(cString: pointer)
    }
}
