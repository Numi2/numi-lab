import AppKit
import CryptoKit
import Metal
import MetalKit
import simd

private enum RobotError: Error, CustomStringConvertible {
    case invalid(String)
    var description: String { if case .invalid(let message) = self { return "Measured surface robot: \(message)" }; return "" }
}

private struct Component: Decodable {
    struct Topology: Decodable {
        let kind: String
        let radialCount: Int?
        let angularCount: Int?
        let firstCount: Int?
        let secondCount: Int?
    }
    let name: String
    let partIdentifier: UInt8
    let vertexOffset: Int
    let vertexCount: Int
    let triangleOffset: Int
    let triangleCount: Int
    let topology: Topology?
}

private struct Manifest: Decodable {
    struct Frames: Decodable {
        let count: Int
        let sampleRateHertz: Float
        let timesSeconds: [Float]
        let interpolation: String
        let endpointVelocity: String
        let periodic: Bool
    }
    struct Axes: Decodable { let x: String; let y: String; let z: String }
    struct Frame: Decodable { let units: String; let birdFlowAxes: Axes }
    struct Topology: Decodable {
        let vertexCount: Int
        let triangleCount: Int
        let indexType: String
        let fixedAcrossFrames: Bool
        let components: [Component]
    }
    struct Member: Decodable { let file: String; let format: String; let layout: String; let bytes: Int; let sha256: String }
    struct Binary: Decodable { let positions: Member; let triangles: Member }
    struct Readiness: Decodable { let completeBirdSurfaceReady: Bool; let cpuParityRequired: Bool; let quantitativeForceAcceptanceReady: Bool }
    let schemaVersion: Int
    let datasetIdentifier: String
    let scientificTier: String
    let frames: Frames
    let coordinateFrame: Frame
    let topology: Topology
    let binary: Binary
    let readiness: Readiness
}

private struct Dataset {
    let manifestURL: URL
    let manifest: Manifest
    let manifestHash: String
    let positions: Data
    let triangles: Data
    let vertexParts: [UInt8]
    let triangleParts: [UInt8]
    let minimum: SIMD3<Float>
    let maximum: SIMD3<Float>
    var center: SIMD3<Float> { (minimum + maximum) * 0.5 }
    var radius: Float { max(0.05, simd_length(maximum - minimum) * 0.5) }

    static func load(_ url: URL) throws -> Dataset {
        let manifestData = try Data(contentsOf: url)
        let value = try JSONDecoder().decode(Manifest.self, from: manifestData)
        guard value.schemaVersion == 1,
              value.scientificTier == "derived-measured-complete-surface",
              value.frames.count >= 2,
              value.frames.timesSeconds.count == value.frames.count,
              value.frames.sampleRateHertz > 0,
              value.frames.interpolation == "piecewise-linear-nonperiodic",
              value.frames.endpointVelocity == "one-sided-adjacent-frame",
              !value.frames.periodic,
              value.coordinateFrame.units == "meters",
              value.coordinateFrame.birdFlowAxes.x == "forward",
              value.coordinateFrame.birdFlowAxes.y == "left",
              value.coordinateFrame.birdFlowAxes.z == "up",
              value.topology.vertexCount > 0,
              value.topology.vertexCount <= Int(UInt16.max),
              value.topology.triangleCount > 0,
              value.topology.indexType == "uint16-little-endian",
              value.topology.fixedAcrossFrames,
              value.topology.components.map(\.name) == ["body", "leftWing", "rightWing", "tail"],
              value.topology.components.map(\.partIdentifier) == [1, 2, 3, 4],
              value.readiness.completeBirdSurfaceReady,
              value.readiness.cpuParityRequired,
              !value.readiness.quantitativeForceAcceptanceReady else {
            throw RobotError.invalid("manifest violates the provenance-locked surface contract")
        }
        let positions = try member(value.binary.positions, beside: url,
                                   format: "float32-little-endian",
                                   layout: "frame-major, vertex-major, xyz")
        let triangles = try member(value.binary.triangles, beside: url,
                                   format: "uint16-little-endian",
                                   layout: "triangle-major, three global vertex indices")
        guard positions.count == value.frames.count * value.topology.vertexCount * 12,
              triangles.count == value.topology.triangleCount * 6 else {
            throw RobotError.invalid("binary byte count disagrees with the manifest")
        }
        let indices = triangles.withUnsafeBytes { raw in
            stride(from: 0, to: raw.count, by: 2).map {
                UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: $0, as: UInt16.self))
            }
        }
        var vertexParts = [UInt8](repeating: 0, count: value.topology.vertexCount)
        var triangleParts = [UInt8](repeating: 0, count: value.topology.triangleCount)
        var nextVertex = 0, nextTriangle = 0
        for component in value.topology.components {
            let vertexEnd = component.vertexOffset + component.vertexCount
            let triangleEnd = component.triangleOffset + component.triangleCount
            guard component.vertexOffset == nextVertex, component.triangleOffset == nextTriangle,
                  component.vertexCount > 0, component.triangleCount > 0,
                  vertexEnd <= vertexParts.count, triangleEnd <= triangleParts.count else {
                throw RobotError.invalid("component ranges are not positive and contiguous")
            }
            vertexParts.replaceSubrange(component.vertexOffset..<vertexEnd,
                                        with: repeatElement(component.partIdentifier, count: component.vertexCount))
            triangleParts.replaceSubrange(component.triangleOffset..<triangleEnd,
                                          with: repeatElement(component.partIdentifier, count: component.triangleCount))
            for triangle in component.triangleOffset..<triangleEnd {
                for corner in 0..<3 {
                    let vertex = Int(indices[triangle * 3 + corner])
                    guard vertex >= component.vertexOffset, vertex < vertexEnd else {
                        throw RobotError.invalid("triangle crosses a component range")
                    }
                }
            }
            nextVertex = vertexEnd; nextTriangle = triangleEnd
        }
        guard nextVertex == vertexParts.count, nextTriangle == triangleParts.count,
              !vertexParts.contains(0), !triangleParts.contains(0) else {
            throw RobotError.invalid("component ranges do not cover the surface")
        }
        guard let tail = value.topology.components.last,
              tail.name == "tail",
              tail.topology?.kind == "polar-grid",
              tail.topology?.radialCount == 7,
              tail.topology?.angularCount == 17,
              tail.vertexCount == 1 + 7 * 17 else {
            throw RobotError.invalid(
                "tail presentation requires the provenance-locked 7x17 polar topology"
            )
        }
        for wing in value.topology.components[1...2] {
            guard wing.topology?.kind == "structured-grid",
                  wing.topology?.firstCount == 9,
                  wing.topology?.secondCount == 33,
                  wing.vertexCount == 9 * 33 else {
                throw RobotError.invalid(
                    "unified wing skin requires the provenance-locked 9x33 topology")
            }
        }
        var minimum = SIMD3<Float>(repeating: .infinity)
        var maximum = SIMD3<Float>(repeating: -.infinity)
        try positions.withUnsafeBytes { raw in
            for offset in stride(from: 0, to: raw.count, by: 12) {
                let point = SIMD3<Float>(decode(raw, offset), decode(raw, offset + 4), decode(raw, offset + 8))
                guard allFinite(point) else { throw RobotError.invalid("nonfinite measured position") }
                minimum = simd_min(minimum, point); maximum = simd_max(maximum, point)
            }
        }
        return Dataset(manifestURL: url, manifest: value, manifestHash: hash(manifestData), positions: positions,
                       triangles: triangles, vertexParts: vertexParts, triangleParts: triangleParts,
                       minimum: minimum, maximum: maximum)
    }

    func point(frame: Int, vertex: Int) -> SIMD3<Float> {
        positions.withUnsafeBytes { raw in
            let offset = (frame * manifest.topology.vertexCount + vertex) * 12
            return SIMD3<Float>(Self.decode(raw, offset), Self.decode(raw, offset + 4), Self.decode(raw, offset + 8))
        }
    }

    private static func member(_ item: Manifest.Member, beside url: URL, format: String, layout: String) throws -> Data {
        guard item.file == URL(fileURLWithPath: item.file).lastPathComponent,
              !item.file.contains(".."), item.format == format, item.layout == layout else {
            throw RobotError.invalid("unsafe or incompatible binary member")
        }
        let data = try Data(contentsOf: url.deletingLastPathComponent().appendingPathComponent(item.file))
        guard data.count == item.bytes, hash(data) == item.sha256 else {
            throw RobotError.invalid("hash lock failed for \(item.file)")
        }
        return data
    }
    private static func decode(_ raw: UnsafeRawBufferPointer, _ offset: Int) -> Float {
        Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self)))
    }
    private static func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    private static func allFinite(_ p: SIMD3<Float>) -> Bool { p.x.isFinite && p.y.isFinite && p.z.isFinite }
}

private func closestTriangleBarycentric(
    point: SIMD3<Float>,
    a: SIMD3<Float>,
    b: SIMD3<Float>,
    c: SIMD3<Float>
) -> SIMD3<Float> {
    let ab = b - a, ac = c - a, ap = point - a
    let d1 = simd_dot(ab, ap), d2 = simd_dot(ac, ap)
    if d1 <= 0, d2 <= 0 { return SIMD3<Float>(1, 0, 0) }
    let bp = point - b
    let d3 = simd_dot(ab, bp), d4 = simd_dot(ac, bp)
    if d3 >= 0, d4 <= d3 { return SIMD3<Float>(0, 1, 0) }
    let vc = d1 * d4 - d3 * d2
    if vc <= 0, d1 >= 0, d3 <= 0 {
        let v = d1 / max(1e-12, d1 - d3)
        return SIMD3<Float>(1 - v, v, 0)
    }
    let cp = point - c
    let d5 = simd_dot(ab, cp), d6 = simd_dot(ac, cp)
    if d6 >= 0, d5 <= d6 { return SIMD3<Float>(0, 0, 1) }
    let vb = d5 * d2 - d1 * d6
    if vb <= 0, d2 >= 0, d6 <= 0 {
        let w = d2 / max(1e-12, d2 - d6)
        return SIMD3<Float>(1 - w, 0, w)
    }
    let va = d3 * d6 - d5 * d4
    if va <= 0, d4 - d3 >= 0, d5 - d6 >= 0 {
        let w = (d4 - d3) / max(1e-12, (d4 - d3) + (d5 - d6))
        return SIMD3<Float>(0, 1 - w, w)
    }
    let denominator = max(1e-12, va + vb + vc)
    let v = vb / denominator, w = vc / denominator
    return SIMD3<Float>(1 - v - w, v, w)
}

private struct Uniforms {
    var viewProjection = matrix_identity_float4x4
    var eye = SIMD4<Float>(0, 0, 0, 1)
    var centerRadius = SIMD4<Float>(0, 0, 0, 1)
    var boundsMinimum = SIMD4<Float>(0, 0, 0, 0)
    var boundsMaximum = SIMD4<Float>(0, 0, 0, 0)
    var frameInfo = SIMD4<UInt32>(0, 1, 0, 0)
    var tailInfo = SIMD4<UInt32>(0, 0, 0, 0)
    var timing = SIMD4<Float>(0, 0.001, 0, 0)
    var foodTarget = SIMD4<Float>(0, 0, 0, 0)
}

private final class RobotEngine {
    let dataset: Dataset
    let device: MTLDevice
    private(set) var environmentIndex: Int
    private let queue: MTLCommandQueue
    private let integratePipeline: MTLComputePipelineState
    private let deformPipeline: MTLComputePipelineState
    private let rootPipeline: MTLComputePipelineState
    private let copyPipeline: MTLComputePipelineState
    private let smoothPipeline: MTLComputePipelineState
    private let normalPipeline: MTLComputePipelineState
    private let renderPipeline: MTLRenderPipelineState
    private let tailRenderPipeline: MTLRenderPipelineState
    private let groundRenderPipeline: MTLRenderPipelineState
    private let foodRenderPipeline: MTLRenderPipelineState
    private let depthStencilState: MTLDepthStencilState
    private let positions: MTLBuffer
    private let triangles: MTLBuffer
    private let vertexParts: MTLBuffer
    private let triangleParts: MTLBuffer
    private let triangleAdjacency: MTLBuffer
    private let adjacencyCounts: MTLBuffer
    private let surfaceNormals: MTLBuffer
    private let attachmentIndices: MTLBuffer
    private let attachmentWeights: MTLBuffer
    private let actuatorTargets: MTLBuffer
    private let actuatorState: MTLBuffer
    private let localSurface: MTLBuffer
    private let previousSurface: MTLBuffer
    private let visualSurfaceA: MTLBuffer
    private let visualSurfaceB: MTLBuffer
    private let rootState: MTLBuffer
    private let metrics: MTLBuffer
    private var initialized = false
    private(set) var simulationTime: Float = 0
    private(set) var sourceTime: Float = 0
    private var sourceDirection: Float = 1
    private(set) var terminated = false
    private var terminationHold: Float = 0
    private var groundTerminationEnabled = true
    private let policyContext: MetalRoboTaskRolloutContext?
    private let policyName: String?
    private var policyRevision: UInt64 = 0
    private var policyAccumulator: Float = 0
    private var policySeed: UInt64
    private let fixedReplaySeed: UInt64?
    private var latestTransition: MetalRoboTaskTransition?
    private let foodNavigation: Bool
    private var foodTarget = SIMD3<Float>(8, 0, 5)
    private var foodVisible = false
    private var foodConsumed = false
    var controllerEnabled = true
    var policyDriven: Bool { policyContext != nil }

    static let physicsTimestep: Float = 0.005
    private static let episodeDuration: Float = 4.8

    init(
        dataset: Dataset,
        environmentIndex: Int = 0,
        policyURL: URL? = nil,
        metallibPath: String? = nil,
        seed: UInt64? = nil,
        measuredSurfaceTask: MetalRoboMeasuredSurfaceTask = .fatalDropRecovery
    ) throws {
        self.dataset = dataset
        self.environmentIndex = environmentIndex
        fixedReplaySeed = seed
        policySeed = seed ?? UInt64(0xD0_0E_0000 + environmentIndex)
        switch measuredSurfaceTask {
        case .foodNavigation: foodNavigation = true
        default: foodNavigation = false
        }
        if let policyURL {
            let difficultyBandRange: ClosedRange<UInt32>
            switch measuredSurfaceTask {
            case .cruise: difficultyBandRange = 0...0
            // The viewer replays the learned first-stage food curriculum. The
            // broader 0...3 range remains an evaluation/training concern.
            case .foodNavigation: difficultyBandRange = 0...0
            default: difficultyBandRange = 4...4
            }
            let configuration = MetalRoboTaskRolloutConfiguration(
                environmentCount: 1,
                controlTimestepSeconds: 0.02,
                seed: policySeed,
                difficultyBandRange: difficultyBandRange
            )
            let context = try MetalRoboTaskRolloutContext(
                manifest: MetalRoboRunManifest(
                    source: .measuredDove(
                        manifest: dataset.manifestURL,
                        task: measuredSurfaceTask
                    ),
                    sensorsAndPhysics: configuration
                ),
                metallibPath: metallibPath
            )
            try context.setStateReadback(true)
            try context.loadPolicy(at: policyURL)
            guard context.installedPolicyRevision != 0 else {
                throw RobotError.invalid("PolicyPack installed without a revision")
            }
            policyContext = context
            policyName = policyURL.deletingLastPathComponent().lastPathComponent
            policyRevision = context.installedPolicyRevision
        } else {
            policyContext = nil
            policyName = nil
        }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw RobotError.invalid("Metal device unavailable")
        }
        self.device = device; self.queue = queue
        let library = try device.makeLibrary(source: Self.metalSource, options: nil)
        func compute(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else { throw RobotError.invalid("missing Metal kernel \(name)") }
            return try device.makeComputePipelineState(function: function)
        }
        integratePipeline = try compute("integrateActuators")
        deformPipeline = try compute("deformSurface")
        rootPipeline = try compute("integrateRoot")
        copyPipeline = try compute("copySurface")
        smoothPipeline = try compute("smoothVisualSurface")
        normalPipeline = try compute("computeSurfaceNormals")
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = library.makeFunction(name: "robotVertex")
        descriptor.fragmentFunction = library.makeFunction(name: "robotFragment")
        descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
        descriptor.depthAttachmentPixelFormat = .depth32Float
        renderPipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        let tailDescriptor = MTLRenderPipelineDescriptor()
        tailDescriptor.vertexFunction = library.makeFunction(name: "tailFeatherVertex")
        tailDescriptor.fragmentFunction = library.makeFunction(name: "tailFeatherFragment")
        tailDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
        tailDescriptor.depthAttachmentPixelFormat = .depth32Float
        tailRenderPipeline = try device.makeRenderPipelineState(
            descriptor: tailDescriptor)
        let groundDescriptor = MTLRenderPipelineDescriptor()
        groundDescriptor.vertexFunction = library.makeFunction(name: "groundVertex")
        groundDescriptor.fragmentFunction = library.makeFunction(name: "groundFragment")
        groundDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
        groundDescriptor.depthAttachmentPixelFormat = .depth32Float
        groundRenderPipeline = try device.makeRenderPipelineState(
            descriptor: groundDescriptor)
        let foodDescriptor = MTLRenderPipelineDescriptor()
        foodDescriptor.vertexFunction = library.makeFunction(name: "foodVertex")
        foodDescriptor.fragmentFunction = library.makeFunction(name: "foodFragment")
        foodDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
        foodDescriptor.depthAttachmentPixelFormat = .depth32Float
        foodRenderPipeline = try device.makeRenderPipelineState(
            descriptor: foodDescriptor)
        let depthDescriptor = MTLDepthStencilDescriptor()
        depthDescriptor.depthCompareFunction = .less
        depthDescriptor.isDepthWriteEnabled = true
        guard let depthState = device.makeDepthStencilState(
            descriptor: depthDescriptor) else {
            throw RobotError.invalid("could not create opaque depth state")
        }
        depthStencilState = depthState
        func buffer(_ data: Data, _ label: String) throws -> MTLBuffer {
            guard let result = data.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }) else {
                throw RobotError.invalid("could not allocate \(label)")
            }
            result.label = label; return result
        }
        positions = try buffer(dataset.positions, "provenance-locked positions")
        triangles = try buffer(dataset.triangles, "provenance-locked triangles")
        let meshIndices = dataset.triangles.withUnsafeBytes { raw in
            stride(from: 0, to: raw.count, by: 2).map {
                UInt16(littleEndian: raw.loadUnaligned(
                    fromByteOffset: $0, as: UInt16.self))
            }
        }
        let body = dataset.manifest.topology.components[0]
        let leftWing = dataset.manifest.topology.components[1]
        let rightWing = dataset.manifest.topology.components[2]
        let tail = dataset.manifest.topology.components[3]
        var attachmentIndexRecords: [SIMD4<UInt32>] = []
        var attachmentWeightRecords: [SIMD4<Float>] = []
        var maximumAttachmentGap: Float = 0
        func appendBodyAttachment(for point: SIMD3<Float>) {
            var bestIndices = SIMD4<UInt32>(0, 0, 0, 0)
            var bestWeights = SIMD4<Float>(1, 0, 0, 0)
            var bestDistance = Float.infinity
            for triangle in body.triangleOffset..<(body.triangleOffset + body.triangleCount) {
                let base = triangle * 3
                let ia = Int(meshIndices[base])
                let ib = Int(meshIndices[base + 1])
                let ic = Int(meshIndices[base + 2])
                let a = dataset.point(frame: 0, vertex: ia)
                let b = dataset.point(frame: 0, vertex: ib)
                let c = dataset.point(frame: 0, vertex: ic)
                let weights = closestTriangleBarycentric(
                    point: point, a: a, b: b, c: c)
                let closest = weights.x * a + weights.y * b + weights.z * c
                let distance = simd_length_squared(
                    closest - point)
                if distance < bestDistance {
                    bestDistance = distance
                    bestIndices = SIMD4<UInt32>(
                        UInt32(ia), UInt32(ib), UInt32(ic), 0)
                    bestWeights = SIMD4<Float>(weights, 0)
                }
            }
            maximumAttachmentGap = max(
                maximumAttachmentGap, sqrt(bestDistance))
            attachmentIndexRecords.append(bestIndices)
            attachmentWeightRecords.append(bestWeights)
        }
        for wing in [leftWing, rightWing] {
            for chord in 0..<33 {
                appendBodyAttachment(for: dataset.point(
                    frame: 0, vertex: wing.vertexOffset + chord))
            }
        }
        let tailRoot = dataset.point(frame: 0, vertex: tail.vertexOffset)
        let outerRing = tail.vertexOffset + 1 + 6 * 17
        let edgeA = dataset.point(frame: 0, vertex: outerRing)
        let edgeB = dataset.point(frame: 0, vertex: outerRing + 16)
        let distal = dataset.point(frame: 0, vertex: outerRing + 8)
        let lateralAxis = simd_normalize(edgeB - edgeA)
        let tailAxis = simd_normalize(distal - tailRoot)
        for feather in 0..<33 {
            let layerClass = feather < 12 ? 0 : (feather < 23 ? 1 : 2)
            let localFeather = layerClass == 0
                ? feather : (layerClass == 1 ? feather - 12 : feather - 23)
            let count = layerClass == 0 ? 12 : (layerClass == 1 ? 11 : 10)
            let lateral = -1 + 2 * Float(localFeather) / Float(max(1, count - 1))
            let socketSpan: Float = layerClass == 0 ? 0.028 :
                (layerClass == 1 ? 0.024 : 0.020)
            let socketSetback: Float = layerClass == 0 ? 0.003 :
                (layerClass == 1 ? 0.010 : 0.016)
            appendBodyAttachment(for: tailRoot + lateralAxis *
                (lateral * socketSpan) + tailAxis * socketSetback)
        }
        guard attachmentIndexRecords.count == 99,
              attachmentWeightRecords.count == 99,
              maximumAttachmentGap <= 0.04 else {
            throw RobotError.invalid(
                "unified surface attachment map is incomplete or discontinuous")
        }
        let adjacencyCapacity = 16
        var incident = [[UInt16]](
            repeating: [], count: dataset.manifest.topology.vertexCount)
        for triangle in 0..<dataset.manifest.topology.triangleCount {
            for corner in 0..<3 {
                incident[Int(meshIndices[triangle * 3 + corner])].append(
                    UInt16(triangle))
            }
        }
        guard incident.allSatisfy({ !$0.isEmpty && $0.count <= adjacencyCapacity }) else {
            throw RobotError.invalid(
                "surface adjacency exceeds the bounded smooth-normal contract")
        }
        var flattenedAdjacency = [UInt16](
            repeating: 0,
            count: dataset.manifest.topology.vertexCount * adjacencyCapacity)
        var incidentCounts = [UInt8](repeating: 0, count: incident.count)
        for vertex in incident.indices {
            incidentCounts[vertex] = UInt8(incident[vertex].count)
            flattenedAdjacency.replaceSubrange(
                vertex * adjacencyCapacity..<(vertex * adjacencyCapacity + incident[vertex].count),
                with: incident[vertex])
        }
        guard let vp = device.makeBuffer(bytes: dataset.vertexParts, length: dataset.vertexParts.count, options: .storageModeShared),
              let tp = device.makeBuffer(bytes: dataset.triangleParts, length: dataset.triangleParts.count, options: .storageModeShared),
              let adjacency = device.makeBuffer(bytes: flattenedAdjacency, length: flattenedAdjacency.count * MemoryLayout<UInt16>.stride, options: .storageModeShared),
              let counts = device.makeBuffer(bytes: incidentCounts, length: incidentCounts.count, options: .storageModeShared),
              let normals = device.makeBuffer(length: dataset.manifest.topology.vertexCount * MemoryLayout<SIMD4<Float>>.stride, options: .storageModePrivate),
              let attachments = device.makeBuffer(bytes: attachmentIndexRecords, length: attachmentIndexRecords.count * MemoryLayout<SIMD4<UInt32>>.stride, options: .storageModeShared),
              let weights = device.makeBuffer(bytes: attachmentWeightRecords, length: attachmentWeightRecords.count * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared),
              let targets = device.makeBuffer(length: 24 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let state = device.makeBuffer(length: 24 * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared),
              let local = device.makeBuffer(length: dataset.manifest.topology.vertexCount * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared),
              let previous = device.makeBuffer(length: dataset.manifest.topology.vertexCount * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared),
              let visualA = device.makeBuffer(length: dataset.manifest.topology.vertexCount * MemoryLayout<SIMD4<Float>>.stride, options: .storageModePrivate),
              let visualB = device.makeBuffer(length: dataset.manifest.topology.vertexCount * MemoryLayout<SIMD4<Float>>.stride, options: .storageModePrivate),
              let root = device.makeBuffer(length: 8 * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared),
              let metricBuffer = device.makeBuffer(length: 2 * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared) else {
            throw RobotError.invalid("could not allocate robot state")
        }
        vertexParts = vp; triangleParts = tp
        triangleAdjacency = adjacency; adjacencyCounts = counts; surfaceNormals = normals
        attachmentIndices = attachments; attachmentWeights = weights
        actuatorTargets = targets; actuatorState = state
        localSurface = local; previousSurface = previous
        visualSurfaceA = visualA; visualSurfaceB = visualB
        rootState = root; metrics = metricBuffer
        try reset()
    }

    func reset(dropEpisode: Bool = true) throws {
        actuatorTargets.contents().initializeMemory(as: UInt8.self, repeating: 0, count: actuatorTargets.length)
        actuatorState.contents().initializeMemory(as: UInt8.self, repeating: 0, count: actuatorState.length)
        rootState.contents().initializeMemory(as: UInt8.self, repeating: 0, count: rootState.length)
        metrics.contents().initializeMemory(as: UInt8.self, repeating: 0, count: metrics.length)
        let root = rootState.contents().bindMemory(to: SIMD4<Float>.self, capacity: 8)
        let lane = Float(environmentIndex % 4)
        let row = Float((environmentIndex / 4) % 4)
        root[0] = SIMD4<Float>(0, 0, dropEpisode ? 22 + 2 * lane : 0, 0.35)
        root[1] = SIMD4<Float>(
            dropEpisode ? (row - 1.5) * 0.4 : 0,
            dropEpisode ? (lane - 1.5) * 0.25 : 0,
            dropEpisode ? -6 - (4 / 3) * lane : 0, 0)
        if dropEpisode {
            let tilt = 0.18 + 0.28 * lane
            let yaw = -0.75 + 0.5 * row
            let orientation = simd_quatf(angle: yaw, axis: SIMD3<Float>(0, 0, 1)) *
                simd_quatf(angle: tilt, axis: simd_normalize(SIMD3<Float>(1, 0.35, 0)))
            root[2] = orientation.vector
        } else {
            root[2] = SIMD4<Float>(0, 0, 0, 1)
        }
        simulationTime = 0; sourceTime = 0; sourceDirection = 1
        terminated = false; terminationHold = 0
        groundTerminationEnabled = dropEpisode
        initialized = false
        policyAccumulator = 0
        latestTransition = nil
        foodVisible = false
        foodConsumed = false
        if let policyContext {
            try policyContext.reset(seed: fixedReplaySeed ?? policySeed)
            if fixedReplaySeed == nil { policySeed &+= 1 }
        }
    }

    func selectEnvironment(_ index: Int) throws {
        environmentIndex = max(0, index)
        try reset()
    }

    private func synchronizePolicyRoot() throws {
        guard let policyContext else { return }
        let q = try policyContext.finalConfiguration()
        guard q.count >= 7 else {
            throw RobotError.invalid("native policy configuration has \(q.count) values, expected at least 7")
        }
        let root = rootState.contents().bindMemory(to: SIMD4<Float>.self, capacity: 8)
        root[0] = SIMD4<Float>(q[0], q[1], q[2], 0.35)
        root[2] = SIMD4<Float>(q[3], q[4], q[5], q[6])
    }

    private func advancePolicy() throws {
        guard let policyContext else { return }
        let result = try policyContext.advanceWithPolicy(
            controlStepCount: 1,
            policyRevision: policyRevision
        )
        guard result.failedEnvironmentSteps == 0 else {
            throw RobotError.invalid(
                "native policy rollout failed with GPU status \(result.firstGPUStatusCode)"
            )
        }
        let latents = try policyContext.policyLatents(controlStepCount: 1)
        guard latents.count >= 24 else {
            throw RobotError.invalid("PolicyPack emitted \(latents.count) actions, expected 24")
        }
        let target = actuatorTargets.contents().bindMemory(to: Float.self, capacity: 24)
        for index in 0..<24 { target[index] = tanh(latents[index]) }
        latestTransition = try policyContext.transitions(controlStepCount: 1).last
        if foodNavigation {
            let sceneStates = try policyContext.finalSceneStates()
            // Scene bodies are packed as position(3), orientation(4), linear
            // velocity(3), angular velocity(3). Food is the final scene body.
            if sceneStates.count >= 13 {
                let base = sceneStates.count - 13
                let candidate = SIMD3<Float>(
                    sceneStates[base], sceneStates[base + 1],
                    sceneStates[base + 2])
                if candidate.x.isFinite && candidate.y.isFinite &&
                    candidate.z.isFinite {
                    foodTarget = candidate
                    foodVisible = true
                }
            }
        }
        foodConsumed = latestTransition?.terminationReason == 7
        try synchronizePolicyRoot()
        if latestTransition?.done == true {
            terminated = true
            terminationHold = 0.35
        }
    }

    private func frame(at time: Float) -> (Int, Int, Float, Float) {
        let times = dataset.manifest.frames.timesSeconds
        let clamped = min(max(time, times[0]), times[times.count - 1])
        let upper = times.firstIndex(where: { $0 >= clamped }) ?? times.count - 1
        let first = max(0, upper - (upper == 0 ? 0 : 1))
        let second = min(times.count - 1, max(first + 1, upper))
        let duration = max(1e-6, times[second] - times[first])
        return (first, second, first == second ? 0 : (clamped - times[first]) / duration, duration)
    }

    private func baseUniforms(time: Float, dt: Float) -> Uniforms {
        let f = frame(at: time)
        let tail = dataset.manifest.topology.components[3]
        let foodScale: Float = foodVisible
            ? (foodConsumed ? max(0.08, terminationHold / 0.35) : 1)
            : 0
        return Uniforms(centerRadius: SIMD4<Float>(dataset.center, dataset.radius),
                        boundsMinimum: SIMD4<Float>(dataset.minimum, 0),
                        boundsMaximum: SIMD4<Float>(dataset.maximum, 0),
                        frameInfo: SIMD4<UInt32>(UInt32(f.0), UInt32(f.1), UInt32(dataset.manifest.topology.vertexCount), UInt32(dataset.manifest.topology.triangleCount)),
                        tailInfo: SIMD4<UInt32>(UInt32(tail.vertexOffset), 7, 17, 12),
                        timing: SIMD4<Float>(f.2, dt, time, f.3),
                        foodTarget: SIMD4<Float>(foodTarget, 0.14 * foodScale))
    }

    private func setControllerTargets(time: Float) {
        guard policyContext == nil else { return }
        let target = actuatorTargets.contents().bindMemory(to: Float.self, capacity: 24)
        for i in 0..<24 { target[i] = 0 }
        guard controllerEnabled else { return }
        let wave = sin(time * 31.0)
        target[2] = 0.12 * wave
        target[5] = 0.32 * wave; target[13] = 0.32 * wave
        target[7] = 0.18 * cos(time * 31.0); target[15] = -0.18 * cos(time * 31.0)
        target[8] = 0.22 * wave; target[16] = -0.22 * wave
        target[10] = 0.10 * wave; target[18] = 0.10 * wave
        target[20] = 0.10; target[22] = 0.06 * sin(time * 12.0)
    }

    func step(count: Int = 1) throws {
        let dt = Self.physicsTimestep
        for _ in 0..<count {
            if terminated {
                terminationHold -= dt
                if terminationHold <= 0 { try reset() }
                continue
            }
            if policyContext == nil && simulationTime + dt > Self.episodeDuration {
                try reset()
            }
            if policyContext != nil {
                policyAccumulator += dt
                if policyAccumulator + 1e-6 >= 0.02 {
                    policyAccumulator -= 0.02
                    try advancePolicy()
                }
            }
            setControllerTargets(time: simulationTime)
            var uniforms = baseUniforms(time: sourceTime, dt: dt)
            guard let command = queue.makeCommandBuffer() else { throw RobotError.invalid("command allocation failed") }
            func dispatch(_ pipeline: MTLComputePipelineState, count: Int, buffers: [(MTLBuffer, Int)]) throws {
                guard let encoder = command.makeComputeCommandEncoder() else { throw RobotError.invalid("compute encoder failed") }
                encoder.setComputePipelineState(pipeline)
                for (buffer, index) in buffers { encoder.setBuffer(buffer, offset: 0, index: index) }
                encoder.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 8)
                let width = min(pipeline.maxTotalThreadsPerThreadgroup, max(1, pipeline.threadExecutionWidth * 2))
                encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
                encoder.endEncoding()
            }
            try dispatch(integratePipeline, count: 24, buffers: [(actuatorTargets, 0), (actuatorState, 1)])
            try dispatch(deformPipeline, count: dataset.manifest.topology.vertexCount,
                         buffers: [(positions, 0), (vertexParts, 1), (actuatorState, 2), (localSurface, 3)])
            if initialized && policyContext == nil {
                try dispatch(rootPipeline, count: 1,
                             buffers: [(localSurface, 0), (previousSurface, 1), (triangles, 2), (rootState, 3), (metrics, 4)])
            }
            try dispatch(copyPipeline, count: dataset.manifest.topology.vertexCount,
                         buffers: [(localSurface, 0), (previousSurface, 1)])
            command.commit(); command.waitUntilCompleted()
            guard command.status == .completed else { throw RobotError.invalid("Metal mechanics failed: \(command.error?.localizedDescription ?? "unknown")") }
            let actuator = actuatorState.contents().bindMemory(
                to: SIMD2<Float>.self, capacity: 24)
            let phaseRate = min(2.0, max(0.25, 1.0 + actuator[0].x))
            sourceTime += sourceDirection * phaseRate * dt
            let sourceEnd = dataset.manifest.frames.timesSeconds.last!
            while sourceTime > sourceEnd || sourceTime < 0 {
                if sourceTime > sourceEnd {
                    sourceTime = 2 * sourceEnd - sourceTime
                    sourceDirection = -1
                }
                if sourceTime < 0 {
                    sourceTime = -sourceTime
                    sourceDirection = 1
                }
            }
            initialized = true; simulationTime += dt
            let root = rootState.contents().bindMemory(
                to: SIMD4<Float>.self, capacity: 8)
            if policyContext == nil && groundTerminationEnabled && root[0].z <= 0.04 {
                terminated = true
                terminationHold = 0.35
            }
        }
    }

    func render(pass: MTLRenderPassDescriptor, drawable: MTLDrawable, size: CGSize,
                yaw: Float, pitch: Float, distanceScale: Float,
                groundAwareCamera: Bool) throws {
        let root = rootState.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: 8)
        let birdTarget = dataset.center +
            SIMD3<Float>(root[0].x, root[0].y, root[0].z)
        let altitude = max(0, root[0].z)
        let groundAware = policyContext != nil && groundAwareCamera
        let target = groundAware
            ? SIMD3<Float>(birdTarget.x, birdTarget.y,
                           max(dataset.radius, 0.5 * altitude))
            : birdTarget
        let closeDistance = dataset.radius * distanceScale
        let groundDistance = (0.92 * altitude + 4 * dataset.radius) *
            (distanceScale / 2.5)
        let cameraDistance = groundAware
            ? max(closeDistance, groundDistance)
            : closeDistance
        let eye = target + cameraDistance * SIMD3<Float>(
            cos(pitch) * cos(yaw), cos(pitch) * sin(yaw), sin(pitch))
        let fieldOfView: Float = groundAware ? 0.92 : 0.72
        let farDistance = max(dataset.radius * 20,
                              max(cameraDistance * 4 + altitude * 2,
                                  altitude * 3 + dataset.radius * 20))
        var uniforms = baseUniforms(time: sourceTime, dt: Self.physicsTimestep)
        uniforms.eye = SIMD4<Float>(eye, 1)
        uniforms.viewProjection = perspective(
            fieldOfView,
            Float(max(1, size.width) / max(1, size.height)),
            dataset.radius * 0.01,
            farDistance) * lookAt(eye, target, SIMD3<Float>(0, 0, 1))
        guard let command = queue.makeCommandBuffer() else {
            throw RobotError.invalid("render command allocation failed")
        }
        func dispatchPresentation(
            _ pipeline: MTLComputePipelineState,
            buffers: [(MTLBuffer, Int)],
            strength: Float? = nil
        ) throws {
            guard let computeEncoder = command.makeComputeCommandEncoder() else {
                throw RobotError.invalid("presentation compute encoder failed")
            }
            computeEncoder.setComputePipelineState(pipeline)
            for (buffer, index) in buffers {
                computeEncoder.setBuffer(buffer, offset: 0, index: index)
            }
            if var strength {
                computeEncoder.setBytes(
                    &strength, length: MemoryLayout<Float>.stride, index: 7)
            }
            computeEncoder.setBytes(
                &uniforms, length: MemoryLayout<Uniforms>.stride, index: 8)
            let width = min(
                pipeline.maxTotalThreadsPerThreadgroup,
                max(1, pipeline.threadExecutionWidth * 2))
            computeEncoder.dispatchThreads(
                MTLSize(width: dataset.manifest.topology.vertexCount,
                        height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(
                    width: width, height: 1, depth: 1))
            computeEncoder.endEncoding()
        }
        try dispatchPresentation(copyPipeline, buffers: [
            (localSurface, 0), (visualSurfaceA, 1)
        ])
        for _ in 0..<4 {
            try dispatchPresentation(smoothPipeline, buffers: [
                (visualSurfaceA, 0), (visualSurfaceB, 1),
                (triangles, 2), (triangleAdjacency, 3),
                (adjacencyCounts, 4), (vertexParts, 5)
            ], strength: 0.45)
            try dispatchPresentation(smoothPipeline, buffers: [
                (visualSurfaceB, 0), (visualSurfaceA, 1),
                (triangles, 2), (triangleAdjacency, 3),
                (adjacencyCounts, 4), (vertexParts, 5)
            ], strength: -0.47)
        }
        try dispatchPresentation(normalPipeline, buffers: [
            (visualSurfaceA, 0), (triangles, 1),
            (triangleAdjacency, 2), (adjacencyCounts, 3),
            (surfaceNormals, 4)
        ])
        guard let encoder = command.makeRenderCommandEncoder(descriptor: pass) else {
            throw RobotError.invalid("render encoder failed")
        }
        encoder.setDepthStencilState(depthStencilState)
        encoder.setRenderPipelineState(renderPipeline); encoder.setCullMode(.none)
        encoder.setVertexBuffer(visualSurfaceA, offset: 0, index: 0)
        encoder.setVertexBuffer(triangles, offset: 0, index: 1)
        encoder.setVertexBuffer(triangleParts, offset: 0, index: 2)
        encoder.setVertexBuffer(rootState, offset: 0, index: 3)
        encoder.setVertexBuffer(surfaceNormals, offset: 0, index: 4)
        encoder.setVertexBuffer(attachmentIndices, offset: 0, index: 5)
        encoder.setVertexBuffer(attachmentWeights, offset: 0, index: 6)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 8)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 8)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: dataset.manifest.topology.triangleCount * 3)
        encoder.setRenderPipelineState(tailRenderPipeline)
        encoder.setVertexBuffer(visualSurfaceA, offset: 0, index: 0)
        encoder.setVertexBuffer(rootState, offset: 0, index: 3)
        encoder.setVertexBuffer(surfaceNormals, offset: 0, index: 4)
        encoder.setVertexBuffer(attachmentIndices, offset: 0, index: 5)
        encoder.setVertexBuffer(attachmentWeights, offset: 0, index: 6)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 8)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 8)
        encoder.drawPrimitives(
            type: .triangle,
            vertexStart: 0,
            vertexCount: (12 + 11 + 10) * 14 * 8 * 6
        )
        encoder.setRenderPipelineState(groundRenderPipeline)
        encoder.setVertexBuffer(rootState, offset: 0, index: 3)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 8)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        if foodVisible {
            encoder.setRenderPipelineState(foodRenderPipeline)
            encoder.setCullMode(.none)
            encoder.setVertexBytes(
                &uniforms, length: MemoryLayout<Uniforms>.stride, index: 8)
            encoder.setFragmentBytes(
                &uniforms, length: MemoryLayout<Uniforms>.stride, index: 8)
            encoder.drawPrimitives(
                type: .triangle, vertexStart: 0,
                vertexCount: 5 * 28 * 16 * 6)
        }
        encoder.endEncoding(); command.present(drawable); command.commit()
    }

    func probe() throws -> (Float, Float, Float, SIMD4<Float>) {
        controllerEnabled = false; try reset(dropEpisode: false); try step()
        let gpu = localSurface.contents().bindMemory(to: SIMD4<Float>.self, capacity: dataset.manifest.topology.vertexCount)
        var exactError: Float = 0
        let f = frame(at: 0)
        for vertex in 0..<dataset.manifest.topology.vertexCount {
            let expected = simd_mix(dataset.point(frame: f.0, vertex: vertex), dataset.point(frame: f.1, vertex: vertex), SIMD3<Float>(repeating: f.2))
            exactError = max(exactError, simd_length(SIMD3<Float>(gpu[vertex].x, gpu[vertex].y, gpu[vertex].z) - expected))
        }
        guard exactError <= 1e-7 else { throw RobotError.invalid("zero-action geometry invariant failed at \(exactError) m") }
        controllerEnabled = true
        for _ in 0..<70 { try step() }
        var response: Float = 0
        let responseFrame = frame(at: sourceTime)
        for vertex in 0..<dataset.manifest.topology.vertexCount where dataset.vertexParts[vertex] != 1 {
            let reference = simd_mix(dataset.point(frame: responseFrame.0, vertex: vertex),
                                     dataset.point(frame: responseFrame.1, vertex: vertex),
                                     SIMD3<Float>(repeating: responseFrame.2))
            response = max(response, simd_length(SIMD3<Float>(gpu[vertex].x, gpu[vertex].y, gpu[vertex].z) - reference))
        }
        let root = rootState.contents().bindMemory(to: SIMD4<Float>.self, capacity: 8)
        let metric = metrics.contents().bindMemory(to: SIMD4<Float>.self, capacity: 2)[0]
        guard response > 1e-4, metric.x.isFinite, metric.y.isFinite, Self.allFinite(root[0]) else {
            throw RobotError.invalid("action-conditioned mechanics did not produce finite physical response")
        }
        return (exactError, response, simd_length(SIMD3<Float>(root[0].x, root[0].y, root[0].z)), metric)
    }

    var status: String {
        let rootStates = rootState.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: 8)
        let root = rootStates[0]
        let velocity = rootStates[1]
        let groundSpeed = hypot(velocity.x, velocity.y)
        let heading = atan2(velocity.y, velocity.x) * 180 / Float.pi
        let metric = metrics.contents().bindMemory(to: SIMD4<Float>.self, capacity: 2)[0]
        if let transition = latestTransition {
            let birdCenter = dataset.center + SIMD3<Float>(root.x, root.y, root.z)
            let foodDistance = foodVisible
                ? simd_length(foodTarget - birdCenter) : 0
            let outcome = transition.terminationReason == 7
                ? "  FOOD EATEN" : (terminated ? "  EPISODE END" : "")
            return String(
                format: "BEST POLICY  %@  native Metal r%llu  %.2f s  altitude %.2f m  speed %.2f m/s  heading %+.0f°  food %.2f m  tracking %.3f  tilt %.2f°%@",
                policyName ?? "PolicyPack", policyRevision, simulationTime,
                root.z, groundSpeed, heading, foodDistance,
                transition.trackingScore,
                transition.tilt * 180 / .pi,
                outcome
            )
        }
        if policyContext != nil {
            return String(
                format: "BEST POLICY  %@  native Metal r%llu  initializing  altitude %.2f m",
                policyName ?? "PolicyPack", policyRevision, root.z
            )
        }
        return String(
            format: "ENV %02d  %.2f / %.2f s  source %.3f s %@  demo %@  altitude %.2f m  aero %.3f N",
            environmentIndex + 1, simulationTime, Self.episodeDuration, sourceTime,
            sourceDirection > 0 ? "→" : "←",
            controllerEnabled ? "ON" : "OFF", root.z, metric.x) +
            (terminated ? "  IMPACT" : "")
    }

    private static func allFinite(_ p: SIMD4<Float>) -> Bool { p.x.isFinite && p.y.isFinite && p.z.isFinite && p.w.isFinite }

    private static let metalSource = #"""
    #include <metal_stdlib>
    using namespace metal;
    struct Uniforms { float4x4 vp; float4 eye; float4 centerRadius; float4 bmin; float4 bmax; uint4 frame; uint4 tail; float4 timing; float4 foodTarget; };
    float3 measured(device const packed_float3* p, uint frameIndex, uint vertexIndex, uint count) { return float3(p[frameIndex * count + vertexIndex]); }
    float3 rotateAxis(float3 p, float3 axis, float angle) { float s=sin(angle), c=cos(angle); return p*c + cross(axis,p)*s + axis*dot(axis,p)*(1-c); }
    kernel void integrateActuators(device const float* target [[buffer(0)]], device float2* state [[buffer(1)]], constant Uniforms& u [[buffer(8)]], uint id [[thread_position_in_grid]]) {
        if(id>=24) return; float hz=id<4?8.0f:14.0f, w=6.2831853f*hz, damping=id<4?1.0f:0.82f;
        float bound=id==0?0.35f:(id==1?0.25f:(id==2?0.45f:1.0f)); float lo=id==3?0.0f:-bound;
        float x=state[id].x, v=state[id].y, t=clamp(target[id],lo,bound); float a=w*w*(t-x)-2*damping*w*v;
        v+=u.timing.y*a; x+=u.timing.y*v; state[id]=float2(x,v);
    }
    kernel void deformSurface(device const packed_float3* source [[buffer(0)]], device const uchar* parts [[buffer(1)]], device const float2* q [[buffer(2)]], device float4* output [[buffer(3)]], constant Uniforms& u [[buffer(8)]], uint id [[thread_position_in_grid]]) {
        if(id>=u.frame.z) return; uint part=uint(parts[id]); uint actionBase=part==2?4u:(part==3?12u:4u); float phaseShift=0.10f*q[0].x+((part==2||part==3)?0.08f*q[actionBase].x:0.0f); float blend=clamp(u.timing.x+phaseShift,0.0f,1.0f); float3 p=mix(measured(source,u.frame.x,id,u.frame.z),measured(source,u.frame.y,id,u.frame.z),blend), base=p;
        float3 span=u.bmax.xyz-u.bmin.xyz; float nx=(p.x-u.bmin.x)/max(span.x,1e-6f); p.z+=0.004f*q[1].x*(part==1?0.25f:1.0f);
        if(part==2 || part==3) {
            uint a=part==2?4u:12u; float side=part==2?1.0f:-1.0f; float hingeY=side*0.020f;
            float s=clamp(abs(p.y-hingeY)/max(0.001f,0.5f*span.y),0.0f,1.0f); float3 r=p-float3(u.centerRadius.x,hingeY,u.centerRadius.z);
            r.y*=1.0f+0.18f*q[a+6].x*s; r=rotateAxis(r,float3(1,0,0),side*(0.28f*q[a+1].x+0.10f*q[2].x+0.08f*q[3].x)*s);
            r=rotateAxis(r,float3(0,0,1),-side*(0.22f*q[a+5].x+0.18f*q[a+2].x)*s); r=rotateAxis(r,normalize(float3(0,side,0.08f)),side*(0.34f*q[a+3].x+0.25f*q[a+4].x*s)*s);
            p=float3(u.centerRadius.x,hingeY,u.centerRadius.z)+r; p.x+=0.030f*q[a+5].x*s; p.z+=0.012f*q[a+7].x*sin(3.14159265f*nx)*s;
        } else if(part==4) {
            float3 pivot=mix(measured(source,u.frame.x,u.tail.x,u.frame.z),measured(source,u.frame.y,u.tail.x,u.frame.z),blend); pivot.z+=0.004f*q[1].x;
            float3 r=p-pivot; r=rotateAxis(r,float3(0,1,0),-0.30f*q[20].x); r=rotateAxis(r,float3(0,0,1),0.25f*q[21].x); r=rotateAxis(r,float3(1,0,0),0.28f*q[22].x); r.y*=1.0f+0.22f*q[23].x; p=pivot+r;
        }
        output[id]=float4(p, length(p-base));
    }
    kernel void integrateRoot(device const float4* current [[buffer(0)]], device const float4* previous [[buffer(1)]], device const ushort* indices [[buffer(2)]], device float4* root [[buffer(3)]], device float4* metrics [[buffer(4)]], constant Uniforms& u [[buffer(8)]], uint id [[thread_position_in_grid]]) {
        if(id) return; float3 force=float3(0), torque=float3(0); float areaSum=0, dt=max(u.timing.y,1e-6f);
        for(uint t=0;t<u.frame.w;t++){ uint i0=indices[3*t],i1=indices[3*t+1],i2=indices[3*t+2]; float3 a=current[i0].xyz,b=current[i1].xyz,c=current[i2].xyz; float3 nraw=cross(b-a,c-a); float twice=length(nraw); if(twice<1e-10f) continue; float3 n=nraw/twice; float area=0.5f*twice; float3 velocity=((a-previous[i0].xyz)+(b-previous[i1].xyz)+(c-previous[i2].xyz))/(3*dt)+root[1].xyz; float vn=dot(velocity,n); float3 f=-0.5f*1.225f*area*(1.15f*abs(vn)*vn*n+0.08f*length(velocity)*velocity); float3 centroid=(a+b+c)/3; force+=f; torque+=cross(centroid-u.centerRadius.xyz,f); areaSum+=area; }
        const float mass=0.35f; force+=float3(0,0,-9.80665f*mass); float3 velocity=root[1].xyz+dt*force/mass; float3 position=root[0].xyz+dt*velocity;
        float3 inertia=float3(0.0018f,0.0024f,0.0031f); float3 omega=root[3].xyz+dt*(torque-cross(root[3].xyz,inertia*root[3].xyz))/inertia; float4 q=root[2]; float4 dq=0.5f*float4(q.w*omega+cross(q.xyz,omega),-dot(q.xyz,omega)); q=normalize(q+dt*dq);
        bool ok=all(isfinite(position))&&all(isfinite(velocity))&&all(isfinite(q))&&all(isfinite(omega)); if(ok){root[0]=float4(position,mass);root[1]=float4(velocity,0);root[2]=q;root[3]=float4(omega,0);metrics[0]=float4(length(force),length(torque),areaSum,1);} else {metrics[1].x+=1;}
    }
    kernel void copySurface(device const float4* source [[buffer(0)]], device float4* target [[buffer(1)]], uint id [[thread_position_in_grid]]) { target[id]=source[id]; }
    kernel void smoothVisualSurface(device const float4* source [[buffer(0)]],device float4* target [[buffer(1)]],device const ushort* indices [[buffer(2)]],device const ushort* adjacency [[buffer(3)]],device const uchar* counts [[buffer(4)]],device const uchar* parts [[buffer(5)]],constant float& strength [[buffer(7)]],constant Uniforms& u [[buffer(8)]],uint id [[thread_position_in_grid]]) {
        if(id>=u.frame.z)return;if(parts[id]!=1u){target[id]=source[id];return;}float3 point=source[id].xyz,weighted=float3(0);float weightSum=0;uint incidentCount=uint(counts[id]);
        for(uint lane=0;lane<incidentCount;lane++){uint triangle=uint(adjacency[id*16u+lane]);for(uint corner=0;corner<3u;corner++){uint neighbor=uint(indices[triangle*3u+corner]);if(neighbor==id)continue;float3 sample=source[neighbor].xyz;float distanceSquared=dot(sample-point,sample-point);float weight=1.0f/(1.0f+distanceSquared/1.2e-4f);weighted+=weight*sample;weightSum+=weight;}}
        if(incidentCount<5u){target[id]=source[id];return;}float3 candidate=weightSum>1.0e-8f?point+strength*(weighted/weightSum-point):point;target[id]=float4(all(isfinite(candidate))?candidate:point,source[id].w);
    }
    kernel void computeSurfaceNormals(device const float4* surface [[buffer(0)]], device const ushort* indices [[buffer(1)]], device const ushort* adjacency [[buffer(2)]], device const uchar* counts [[buffer(3)]], device float4* normals [[buffer(4)]], constant Uniforms& u [[buffer(8)]], uint id [[thread_position_in_grid]]) {
        if(id>=u.frame.z) return; float3 accumulated=float3(0); uint count=uint(counts[id]);
        for(uint lane=0;lane<count;lane++){uint triangle=uint(adjacency[id*16u+lane]);ushort3 ix=ushort3(indices[triangle*3u],indices[triangle*3u+1u],indices[triangle*3u+2u]);float3 a=surface[ix.x].xyz,b=surface[ix.y].xyz,c=surface[ix.z].xyz;accumulated+=cross(b-a,c-a);}
        float magnitude=length(accumulated); normals[id]=float4(magnitude>1.0e-10f?accumulated/magnitude:float3(0,0,1),0);
    }
    struct Raster { float4 position [[position]]; float3 world; float3 normal; float4 color; float4 detail; };
    float3 quatRotate(float4 q,float3 p){return p+2*cross(q.xyz,cross(q.xyz,p)+q.w*p);}
    float3 attachedValue(device const float4* values,device const uint4* attachmentIndices,device const float4* attachmentWeights,uint attachment){uint4 ix=attachmentIndices[attachment];float4 w=attachmentWeights[attachment];return w.x*values[ix.x].xyz+w.y*values[ix.y].xyz+w.z*values[ix.z].xyz;}
    vertex Raster robotVertex(device const float4* surface [[buffer(0)]], device const ushort* indices [[buffer(1)]], device const uchar* parts [[buffer(2)]], device const float4* root [[buffer(3)]], device const float4* normals [[buffer(4)]],device const uint4* attachmentIndices [[buffer(5)]],device const float4* attachmentWeights [[buffer(6)]], constant Uniforms& u [[buffer(8)]], uint vid [[vertex_id]]) {
        uint tri=vid/3,corner=vid%3; ushort3 ix=ushort3(indices[tri*3],indices[tri*3+1],indices[tri*3+2]); ushort vertexIndex=corner==0?ix.x:(corner==1?ix.y:ix.z);uint part=parts[tri];float3 local=surface[vertexIndex].xyz,localNormal=normals[vertexIndex].xyz;float rootBlend=1.0f,colorBlend=1.0f;
        if(part==2u||part==3u){uint wingOffset=part==2u?u.tail.x-594u:u.tail.x-297u,wingLocal=uint(vertexIndex)-wingOffset,row=wingLocal/33u,column=wingLocal%33u,attachment=(part==2u?0u:33u)+column;rootBlend=smoothstep(0.0f,3.25f,float(row));colorBlend=smoothstep(2.0f,5.25f,float(row));float3 bodyPoint=attachedValue(surface,attachmentIndices,attachmentWeights,attachment),bodyNormal=normalize(attachedValue(normals,attachmentIndices,attachmentWeights,attachment));local=mix(bodyPoint,local,rootBlend);localNormal=normalize(mix(bodyNormal,localNormal,rootBlend));}
        float3 world=quatRotate(root[2],local-u.centerRadius.xyz)+u.centerRadius.xyz+root[0].xyz; float3 n=normalize(quatRotate(root[2],localNormal));
        float bodyBack=smoothstep(-.045f,.012f,local.z),bodyLength=clamp((local.x-u.bmin.x)/max(1.0e-6f,u.bmax.x-u.bmin.x),0.0f,1.0f);float3 bodyBase=mix(float3(.48f,.46f,.43f),float3(.30f,.32f,.35f),bodyBack);bodyBase=mix(bodyBase,float3(.27f,.29f,.32f),smoothstep(.64f,.90f,bodyLength)*bodyBack*.40f);
        float3 base=bodyBase;if(part==2u||part==3u){float3 wingColor=part==2u?float3(.03f,.58f,1.0f):float3(1.0f,.25f,.06f);base=mix(bodyBase,wingColor,colorBlend);} float act=surface[vertexIndex].w; Raster o;o.position=u.vp*float4(world,1);o.world=world;o.normal=n;o.color=float4(mix(base,float3(.2f,1.0f,.65f),clamp(act/.012f,0.0f,1.0f)*.7f),part==4?0.0f:1.0f);o.detail=float4(local,float(part));return o;
    }
    float3 tailPolarPoint(device const float4* surface, constant Uniforms& u, float angular, uint radial) {
        if(radial==0u) return surface[u.tail.x].xyz;
        float a=clamp(angular,0.0f,float(u.tail.z-1u)); uint a0=uint(floor(a)),a1=min(a0+1u,u.tail.z-1u); float f=a-float(a0);
        uint ring=min(radial,u.tail.y)-1u,base=u.tail.x+1u+ring*u.tail.z;
        return mix(surface[base+a0].xyz,surface[base+a1].xyz,f);
    }
    float3 tailSurfacePoint(device const float4* surface, constant Uniforms& u, float angular, float radialUnit) {
        float r=clamp(radialUnit,0.0f,1.0f)*float(u.tail.y); uint r0=uint(floor(r)),r1=min(r0+1u,u.tail.y); return mix(tailPolarPoint(surface,u,angular,r0),tailPolarPoint(surface,u,angular,r1),r-float(r0));
    }
    float3 tailFeatherCurve(float3 socket,float3 endpoint,float3 normal,float arch,float t) {
        float3 control=mix(socket,endpoint,.46f)+normal*arch;float s=1.0f-t;return s*s*socket+2.0f*s*t*control+t*t*endpoint;
    }
    vertex Raster tailFeatherVertex(device const float4* surface [[buffer(0)]], device const float4* root [[buffer(3)]],device const float4* normals [[buffer(4)]],device const uint4* attachmentIndices [[buffer(5)]],device const float4* attachmentWeights [[buffer(6)]], constant Uniforms& u [[buffer(8)]], uint vid [[vertex_id]]) {
        constexpr uint longitudinalSegments=14u,widthSegments=8u,verticesPerQuad=6u;
        const uint verticesPerFeather=longitudinalSegments*widthSegments*verticesPerQuad;
        uint feather=vid/verticesPerFeather,localVertex=vid%verticesPerFeather,quad=localVertex/verticesPerQuad,corner=localVertex%verticesPerQuad;
        uint longitudinal=quad/widthSegments,widthCell=quad%widthSegments;
        const float2 corners[6]={float2(0,0),float2(1,0),float2(0,1),float2(0,1),float2(1,0),float2(1,1)};
        float2 q=corners[corner]; float t=(float(longitudinal)+q.y)/float(longitudinalSegments); float across=-1.0f+2.0f*(float(widthCell)+q.x)/float(widthSegments);
        uint layerClass=feather<u.tail.w?0u:(feather<u.tail.w+11u?1u:2u);uint localFeather=layerClass==0u?feather:(layerClass==1u?feather-u.tail.w:feather-u.tail.w-11u);uint count=layerClass==0u?u.tail.w:(layerClass==1u?11u:10u);
        float lateral=-1.0f+2.0f*float(localFeather)/float(max(1u,count-1u));
        float angular=.40f+(float(u.tail.z-1u)-.80f)*(float(localFeather)+(layerClass==1u?.5f:0.0f))/float(max(1u,count-(layerClass==1u?0u:1u)));float angularHalfStep=.62f*float(u.tail.z-1u)/float(max(1u,count-1u));
        float3 tailRoot=surface[u.tail.x].xyz,distalCenter=tailPolarPoint(surface,u,.5f*float(u.tail.z-1u),u.tail.y),edgeA=tailPolarPoint(surface,u,0.0f,u.tail.y),edgeB=tailPolarPoint(surface,u,float(u.tail.z-1u),u.tail.y);
        float3 tailAxis=normalize(distalCenter-tailRoot+float3(-1.0e-8f,0,0)),lateralAxis=normalize(edgeB-edgeA+float3(0,1.0e-8f,0)),localNormal=normalize(cross(tailAxis,lateralAxis));
        float3 socket=attachedValue(surface,attachmentIndices,attachmentWeights,66u+feather);float3 socketNormal=normalize(attachedValue(normals,attachmentIndices,attachmentWeights,66u+feather));socket+=socketNormal*.00035f;
        float endT=layerClass==0u?(1.0f-.10f*pow(abs(lateral),1.35f)):(layerClass==1u?(.60f-.07f*abs(lateral)):(.39f-.045f*abs(lateral)));float3 endpoint=tailSurfacePoint(surface,u,angular,endT);
        float roundedT=clamp(t-(layerClass==0u?.070f:.050f)*across*across*smoothstep(.74f,1.0f,t),0.0f,1.0f),curveStep=.012f;float arch=layerClass==0u?.0015f:(layerClass==1u?.0024f:.0032f);
        float3 center=tailFeatherCurve(socket,endpoint,localNormal,arch,roundedT),previous=tailFeatherCurve(socket,endpoint,localNormal,arch,max(0.0f,roundedT-curveStep)),next=tailFeatherCurve(socket,endpoint,localNormal,arch,min(1.0f,roundedT+curveStep));
        float widthRadial=mix(.18f,endT,roundedT);float3 left=tailSurfacePoint(surface,u,angular-angularHalfStep,widthRadial),right=tailSurfacePoint(surface,u,angular+angularHalfStep,widthRadial);float3 surfaceSpan=normalize(right-left+float3(0,1.0e-8f,0)),spanDirection=normalize(mix(lateralAxis,surfaceSpan,smoothstep(.06f,.42f,t)));float3 radialDirection=normalize(next-previous+float3(1.0e-8f,0,0));localNormal=normalize(cross(radialDirection,spanDirection));
        float rootWidth=layerClass==0u?.0055f:(layerClass==1u?.0062f:.0056f),surfaceWidth=(layerClass==0u?.54f:(layerClass==1u?.66f:.72f))*.5f*length(right-left);float halfWidth=mix(rootWidth,surfaceWidth,smoothstep(0.0f,.28f,t))*mix(1.0f,.10f,smoothstep(.80f,1.0f,t));
        float camber=(layerClass==0u?.0017f:.0012f)*(1.0f-across*across)*sin(3.14159265f*t);float layer=.00035f+.00085f*float(layerClass)+.00012f*float(localFeather&1u);
        float3 local=center+spanDirection*(across*halfWidth)+localNormal*(camber+layer);
        float3 world=quatRotate(root[2],local-u.centerRadius.xyz)+u.centerRadius.xyz+root[0].xyz;
        float3 normal=normalize(quatRotate(root[2],localNormal)); float alternating=(localFeather&1u)?0.028f:-0.014f;
        float3 base=(layerClass==0u?float3(.38f,.43f,.52f):(layerClass==1u?float3(.46f,.50f,.58f):float3(.52f,.56f,.63f)))+alternating;base=mix(base,float3(.25f,.29f,.37f),smoothstep(.68f,1.0f,t)*(layerClass==0u?.42f:.22f));
        Raster o;o.position=u.vp*float4(world,1);o.world=world;o.normal=normal;o.color=float4(base,1);o.detail=float4(across,t,float(feather),float(layerClass));return o;
    }
    vertex Raster groundVertex(device const float4* root [[buffer(3)]], constant Uniforms& u [[buffer(8)]], uint vid [[vertex_id]]) {
        // The visible floor is the actual z=0 simulation ground. It follows
        // the dove only in XY to keep two triangles under the active view;
        // fragment coordinates stay world-locked and expose translation.
        const float2 corners[6] = {
            float2(-1,-1),float2(1,-1),float2(-1,1),
            float2(-1,1),float2(1,-1),float2(1,1)
        };
        float scale=max(u.centerRadius.w,1.0e-4f);
        float halfExtent=max(60.0f*scale,4.0f*max(root[0].z,scale));
        float3 world=float3(root[0].xy+corners[vid]*halfExtent,0.0f);
        Raster o;o.position=u.vp*float4(world,1);o.world=world;
        o.normal=float3(0,0,1);o.color=float4(1);
        o.detail=float4(world.xy/scale,0,0);return o;
    }
    uint numiGlyphRow(uint glyph, uint row) {
        // Five-by-seven uppercase glyphs: N U M I L A B.
        constexpr uint rows[49] = {
            17u,25u,21u,19u,17u,17u,17u,
            17u,17u,17u,17u,17u,17u,14u,
            17u,27u,21u,21u,17u,17u,17u,
            31u, 4u, 4u, 4u, 4u, 4u,31u,
            16u,16u,16u,16u,16u,16u,31u,
            14u,17u,17u,31u,17u,17u,17u,
            30u,17u,17u,30u,17u,17u,30u
        };
        return rows[min(glyph,6u)*7u+min(row,6u)];
    }
    uint numiPhraseGlyph(uint character) {
        // NUMI LAB; 7 is the blank separating the words.
        constexpr uint phrase[8] = {0u,1u,2u,3u,7u,4u,5u,6u};
        return phrase[min(character,7u)];
    }
    float groundLine(float coordinate, float spacing, float halfWidth) {
        float distanceToLine=abs(fract(coordinate/spacing+.5f)-.5f)*spacing;
        float antialias=max(fwidth(coordinate)*.85f,halfWidth*.35f);
        return 1.0f-smoothstep(halfWidth,halfWidth+antialias,distanceToLine);
    }
    fragment float4 groundFragment(Raster in [[stage_in]]) {
        const float2 tileSize=float2(6.0f,3.0f);
        float2 coordinate=in.detail.xy;
        float2 tile=floor(coordinate/tileSize);
        float checker=fmod(abs(tile.x+tile.y),2.0f);
        float3 base=mix(float3(.018f,.030f,.045f),float3(.022f,.037f,.053f),checker);

        float minor=max(groundLine(coordinate.x,.5f,.006f),groundLine(coordinate.y,.5f,.006f));
        float major=max(groundLine(coordinate.x,2.0f,.014f),groundLine(coordinate.y,2.0f,.014f));
        float3 color=mix(base,float3(.055f,.105f,.125f),minor*.58f);
        color=mix(color,float3(.075f,.205f,.235f),major*.78f);

        // The logo is a procedural bitmap repeated once per tile. It costs no
        // texture, buffer, draw call, or additional triangle.
        float2 local=fract(coordinate/tileSize)*tileSize;
        float2 pixel=(local-float2(.60f,1.15f))/.10f;
        if(all(pixel>=0.0f)&&pixel.x<48.0f&&pixel.y<7.0f) {
            uint character=min(uint(pixel.x)/6u,7u);
            uint column=uint(pixel.x)%6u;
            uint row=6u-min(uint(pixel.y),6u);
            uint glyph=numiPhraseGlyph(character);
            if(glyph<7u&&column<5u) {
                uint bits=numiGlyphRow(glyph,row);
                float ink=float((bits>>(4u-column))&1u);
                float2 cell=fract(pixel);
                float edge=min(min(cell.x,1.0f-cell.x),min(cell.y,1.0f-cell.y));
                float coverage=smoothstep(0.0f,max(fwidth(pixel.x),fwidth(pixel.y))*.7f,edge);
                color=mix(color,float3(.16f,.72f,.78f),ink*(.42f+.58f*coverage));
            }
        }
        return float4(1.0f-exp(-1.10f*color),1.0f);
    }
    vertex Raster foodVertex(constant Uniforms& u [[buffer(8)]], uint vid [[vertex_id]]) {
        constexpr uint longitudeSegments=28u,latitudeSegments=16u;
        constexpr uint verticesPerSphere=longitudeSegments*latitudeSegments*6u;
        const float2 corners[6]={float2(0,0),float2(1,0),float2(0,1),float2(0,1),float2(1,0),float2(1,1)};
        uint fruit=vid/verticesPerSphere,localVertex=vid%verticesPerSphere;
        uint quad=localVertex/6u,corner=localVertex%6u;
        uint latitude=quad/longitudeSegments,longitude=quad%longitudeSegments;
        float2 uv=(float2(float(longitude),float(latitude))+corners[corner])/float2(float(longitudeSegments),float(latitudeSegments));
        float azimuth=6.283185307f*uv.x,polar=3.141592654f*uv.y;
        float3 sphereNormal=float3(cos(azimuth)*sin(polar),sin(azimuth)*sin(polar),cos(polar));
        const float3 offsets[5]={
            float3(0,0,.10f),float3(.47f,.12f,-.12f),
            float3(-.31f,.39f,-.16f),float3(-.24f,-.43f,-.13f),
            float3(.18f,-.24f,.42f)
        };
        const float scales[5]={.70f,.55f,.53f,.54f,.40f};
        float radius=u.foodTarget.w;
        float breathing=1.0f+.025f*sin(4.0f*u.timing.z+float(fruit)*1.7f);
        float3 center=u.foodTarget.xyz+offsets[min(fruit,4u)]*radius;
        float3 world=center+sphereNormal*(radius*scales[min(fruit,4u)]*breathing);
        Raster o;o.position=u.vp*float4(world,1);o.world=world;o.normal=sphereNormal;
        float hue=float(fruit)/4.0f;
        o.color=float4(mix(float3(.98f,.38f,.035f),float3(1.0f,.72f,.08f),hue),1);
        o.detail=float4(uv,float(fruit),radius);return o;
    }
    fragment float4 foodFragment(Raster in [[stage_in]], bool frontFacing [[front_facing]], constant Uniforms& u [[buffer(8)]]) {
        float3 n=normalize(frontFacing?in.normal:-in.normal);
        float3 v=normalize(u.eye.xyz-in.world),l=normalize(float3(-.30f,-.42f,.86f));
        float diffuse=.30f+.70f*max(0.0f,dot(n,l));
        float rim=pow(1.0f-max(0.0f,dot(n,v)),2.1f);
        float gloss=pow(max(0.0f,dot(n,normalize(l+v))),42.0f);
        float pores=.94f+.06f*sin(73.0f*in.detail.x+47.0f*in.detail.y+11.0f*in.detail.z);
        float3 albedo=in.color.rgb*pores;
        float3 lit=albedo*diffuse+rim*.22f*float3(1.0f,.42f,.05f)+gloss*.42f*float3(1.0f,.86f,.58f);
        return float4(1.0f-exp(-1.22f*lit),1);
    }
    fragment float4 robotFragment(Raster in [[stage_in]],bool frontFacing [[front_facing]],constant Uniforms& u [[buffer(8)]]) {
        if(in.color.a<.5f) discard_fragment();float3 n=normalize(frontFacing?in.normal:-in.normal),v=normalize(u.eye.xyz-in.world),l=normalize(float3(.4,-.5,.8)),albedo=in.color.rgb;bool body=abs(in.detail.w-1.0f)<.25f;
        if(body){float dorsal=smoothstep(-.038f,.010f,in.detail.z),belly=smoothstep(-.018f,.070f,-in.detail.z),forward=smoothstep(-.105f,-.028f,in.detail.x);float3 slate=float3(.34f,.35f,.36f),back=float3(.235f,.255f,.285f),breast=float3(.57f,.535f,.485f);albedo=mix(slate,back,.72f*dorsal);albedo=mix(albedo,breast,.58f*belly);albedo=mix(albedo,float3(.285f,.30f,.325f),.26f*forward);
            float row=.5f+.5f*sin(285.0f*in.detail.x+132.0f*in.detail.z+24.0f*abs(in.detail.y));float barb=.5f+.5f*sin(690.0f*in.detail.x-94.0f*in.detail.z);albedo*=.925f+.055f*row+.020f*barb;
            float neck=forward*smoothstep(.004f,.032f,abs(in.detail.y))*(1.0f-smoothstep(.034f,.050f,abs(in.detail.y)));float grazing=pow(1.0f-clamp(abs(dot(n,v)),0.0f,1.0f),1.35f);float iridescentPhase=.5f+.5f*sin(19.0f*dot(n,float3(.3f,.7f,.2f))+7.0f*dot(v,float3(.6f,-.2f,.7f)));float3 iridescence=mix(float3(.10f,.31f,.235f),float3(.285f,.12f,.31f),iridescentPhase);albedo=mix(albedo,iridescence,.24f*neck*(.35f+.65f*grazing));}
        float ndotl=max(0.0f,dot(n,l)),wrap=.28f+.72f*ndotl,rim=pow(1.0f-max(0.0f,dot(n,v)),2.4f);float3 halfVector=normalize(l+v);float specular=pow(max(0.0f,dot(n,halfVector)),body?18.0f:30.0f)*(body?.055f:.10f);float3 lit=albedo*wrap+rim*(body?.10f:.16f)*albedo+specular*float3(.82f,.86f,.90f);return float4(1.0f-exp(-1.18f*lit),1.0f);
    }
    fragment float4 tailFeatherFragment(Raster in [[stage_in]], constant Uniforms& u [[buffer(8)]]) {
        float edge=smoothstep(.72f,1.0f,abs(in.detail.x)),shaft=1.0f-smoothstep(.025f,.095f,abs(in.detail.x));
        float barb=.5f+.5f*sin(115.0f*in.detail.y+18.0f*abs(in.detail.x)); float3 albedo=mix(in.color.rgb,float3(.17f,.20f,.27f),.42f*edge); albedo+=float3(.13f,.12f,.10f)*shaft*(.35f+.65f*in.detail.y); albedo*=.94f+.06f*barb*(1.0f-edge);
        float3 n=normalize(in.normal),v=normalize(u.eye.xyz-in.world),l=normalize(float3(.4,-.5,.8)); float d=.25f+.72f*abs(dot(n,l)),rim=pow(1.0f-abs(dot(n,v)),2.0f);
        return float4(1.0f-exp(-1.22f*(albedo*d+rim*.20f*albedo)),1);
    }
    """#
}

private final class RobotView: MTKView, MTKViewDelegate {
    let engine: RobotEngine
    private var yaw: Float = -0.78, pitch: Float = 0.28, distanceScale: Float = 2.5
    private var lastPoint: CGPoint?
    private(set) var simulationPaused = false
    private(set) var playbackRate: Float = 1
    private(set) var cameraFollowsBird = true
    private var simulationAccumulator: Float = 0
    private var lastDrawTime = CACurrentMediaTime()
    private let statusLabel = NSTextField(labelWithString: "")
    init(engine: RobotEngine, frame: CGRect) {
        self.engine = engine
        super.init(frame: frame, device: engine.device)
        colorPixelFormat = .bgra8Unorm_srgb; depthStencilPixelFormat = .depth32Float
        clearColor = MTLClearColorMake(0.006, 0.012, 0.025, 1); clearDepth = 1
        preferredFramesPerSecond = 60
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        statusLabel.textColor = .white
        statusLabel.font = .monospacedSystemFont(ofSize: 11, weight: .medium)
        statusLabel.backgroundColor = NSColor.black.withAlphaComponent(0.55)
        statusLabel.drawsBackground = true
        statusLabel.maximumNumberOfLines = 2
        statusLabel.lineBreakMode = .byTruncatingTail
        statusLabel.setContentCompressionResistancePriority(
            .defaultLow, for: .horizontal)
        statusLabel.setContentHuggingPriority(.defaultLow, for: .horizontal)
        addSubview(statusLabel)
        NSLayoutConstraint.activate([
            statusLabel.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 8),
            statusLabel.trailingAnchor.constraint(lessThanOrEqualTo: trailingAnchor, constant: -8),
            statusLabel.topAnchor.constraint(equalTo: topAnchor, constant: 8),
        ])
        delegate = self
    }
    required init(coder: NSCoder) { fatalError() }
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
    func draw(in view: MTKView) {
        do {
            let now = CACurrentMediaTime()
            let wallDelta = min(0.1, max(0, now - lastDrawTime))
            lastDrawTime = now
            if !simulationPaused {
                simulationAccumulator += Float(wallDelta) * playbackRate
                let steps = min(
                    20,
                    Int(simulationAccumulator / RobotEngine.physicsTimestep))
                if steps > 0 {
                    try engine.step(count: steps)
                    simulationAccumulator -=
                        Float(steps) * RobotEngine.physicsTimestep
                }
            }
            guard let pass = currentRenderPassDescriptor, let drawable = currentDrawable else { return }
            try engine.render(
                pass: pass, drawable: drawable, size: drawableSize,
                yaw: yaw, pitch: pitch, distanceScale: distanceScale,
                groundAwareCamera: !cameraFollowsBird)
            statusLabel.stringValue = String(
                format: "%.2fx  %@", playbackRate, engine.status)
        } catch {
            statusLabel.stringValue = "ENV \(engine.environmentIndex + 1) — \(error)"
            simulationPaused = true
        }
    }
    override var acceptsFirstResponder: Bool { true }
    override func keyDown(with event: NSEvent) {
        switch event.charactersIgnoringModifiers?.lowercased() {
        case " ": setPaused(!simulationPaused)
        case "r": stopAndReset()
        case "p":
            if !engine.policyDriven { engine.controllerEnabled.toggle() }
        case "[": setPlaybackRate(playbackRate * 0.5)
        case "]": setPlaybackRate(playbackRate * 2)
        case "c": setCameraFollowsBird(!cameraFollowsBird)
        default: super.keyDown(with: event)
        }
    }
    func setPaused(_ paused: Bool) {
        simulationPaused = paused
        lastDrawTime = CACurrentMediaTime()
    }
    func setPlaybackRate(_ rate: Float) {
        playbackRate = min(4, max(0.25, rate))
        lastDrawTime = CACurrentMediaTime()
    }
    func setCameraFollowsBird(_ follows: Bool) {
        cameraFollowsBird = follows
        pitch = follows ? 0.28 : 0.52
        distanceScale = 2.5
        lastDrawTime = CACurrentMediaTime()
    }
    func stopAndReset() {
        simulationPaused = true
        do {
            try engine.reset()
        } catch {
            statusLabel.stringValue = "ENV \(engine.environmentIndex + 1) — \(error)"
        }
        simulationAccumulator = 0
        lastDrawTime = CACurrentMediaTime()
    }
    func selectEnvironment(_ index: Int) {
        do {
            try engine.selectEnvironment(index)
        } catch {
            statusLabel.stringValue = "ENV \(engine.environmentIndex + 1) — \(error)"
            simulationPaused = true
        }
        simulationAccumulator = 0
        lastDrawTime = CACurrentMediaTime()
    }
    override func mouseDown(with event: NSEvent) { lastPoint = convert(event.locationInWindow, from: nil) }
    override func mouseDragged(with event: NSEvent) { let p=convert(event.locationInWindow,from:nil); if let q=lastPoint { yaw += Float(p.x-q.x)*0.008; pitch = min(1.3,max(-1.3,pitch+Float(p.y-q.y)*0.008)) }; lastPoint=p }
    override func mouseUp(with event: NSEvent) { lastPoint=nil }
    override func scrollWheel(with event: NSEvent) { distanceScale=min(8,max(1.4,distanceScale*exp(Float(event.scrollingDeltaY)*0.015))) }
}

@MainActor
private final class AppDelegate: NSObject, NSApplicationDelegate {
    private let views: [RobotView]
    private let gridContainer = NSView()
    private let pageLabel = NSTextField(labelWithString: "")
    private let runButton = NSButton(title: "Pause", target: nil, action: nil)
    private let cameraButton = NSButton(
        title: "Ground View", target: nil, action: nil)
    private var gridCount = 1
    private var firstEnvironment = 0
    private var paused = false
    private var playbackRate: Float = 1
    private let availableEnvironmentCount: Int
    private let supportsGrid: Bool
    var window: NSWindow?

    init(engines: [RobotEngine]) {
        views = engines.map { RobotView(engine: $0, frame: .zero) }
        supportsGrid = engines.count > 1
        availableEnvironmentCount = supportsGrid ? 16 : 1
        super.init()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1280, height: 820),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered, defer: false)
        window.title = views.first?.engine.policyDriven == true
            ? "Numi Dove — Best Policy"
            : "Numi Dove Recovery Viewer"

        let root = NSStackView()
        root.orientation = .vertical
        root.spacing = 0
        root.translatesAutoresizingMaskIntoConstraints = false
        let toolbar = makeToolbar()
        toolbar.setContentHuggingPriority(.required, for: .vertical)
        root.addArrangedSubview(toolbar)
        root.addArrangedSubview(gridContainer)
        let content = NSView()
        window.contentView = content
        content.addSubview(root)
        NSLayoutConstraint.activate([
            root.leadingAnchor.constraint(equalTo: content.leadingAnchor),
            root.trailingAnchor.constraint(equalTo: content.trailingAnchor),
            root.topAnchor.constraint(equalTo: content.topAnchor),
            root.bottomAnchor.constraint(equalTo: content.bottomAnchor),
            toolbar.heightAnchor.constraint(equalToConstant: 52),
        ])
        rebuildGrid()
        window.center()
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(views[0])
        self.window = window
        NSApp.activate(ignoringOtherApps: true)
    }

    private func makeToolbar() -> NSView {
        let bar = NSStackView()
        bar.orientation = .horizontal
        bar.alignment = .centerY
        bar.spacing = 8
        bar.edgeInsets = NSEdgeInsets(top: 8, left: 10, bottom: 8, right: 10)
        bar.wantsLayer = true
        bar.layer?.backgroundColor = NSColor.windowBackgroundColor.cgColor

        func button(_ title: String, _ action: Selector) -> NSButton {
            let value = NSButton(title: title, target: self, action: action)
            value.bezelStyle = .rounded
            return value
        }
        bar.addArrangedSubview(button("Stop", #selector(stop)))
        runButton.target = self
        runButton.action = #selector(toggleRun)
        runButton.bezelStyle = .rounded
        bar.addArrangedSubview(runButton)
        bar.addArrangedSubview(button("Slower", #selector(slower)))
        bar.addArrangedSubview(button("Faster", #selector(faster)))
        cameraButton.target = self
        cameraButton.action = #selector(toggleCamera)
        cameraButton.bezelStyle = .rounded
        bar.addArrangedSubview(cameraButton)
        if supportsGrid {
            bar.addArrangedSubview(button("Previous", #selector(previousPage)))
            bar.addArrangedSubview(button("Next", #selector(nextPage)))
            let gridSelector = NSSegmentedControl(
                labels: ["1 env", "4 envs", "9 envs"],
                trackingMode: .selectOne, target: self, action: #selector(changeGrid(_:)))
            gridSelector.selectedSegment = 0
            bar.addArrangedSubview(gridSelector)
        }
        pageLabel.font = .monospacedSystemFont(ofSize: 12, weight: .medium)
        pageLabel.alignment = .right
        pageLabel.setContentHuggingPriority(.defaultLow, for: .horizontal)
        bar.addArrangedSubview(pageLabel)
        return bar
    }

    private func rebuildGrid() {
        gridContainer.subviews.forEach { $0.removeFromSuperview() }
        let columns = gridCount == 1 ? 1 : (gridCount == 4 ? 2 : 3)
        let rows = Int(ceil(Double(gridCount) / Double(columns)))
        let outer = NSStackView()
        outer.orientation = .vertical
        outer.spacing = 2
        outer.distribution = .fillEqually
        outer.translatesAutoresizingMaskIntoConstraints = false
        for row in 0..<rows {
            let line = NSStackView()
            line.orientation = .horizontal
            line.spacing = 2
            line.distribution = .fillEqually
            for column in 0..<columns {
                let slot = row * columns + column
                if slot < min(gridCount, views.count) {
                    let view = views[slot]
                    view.selectEnvironment((firstEnvironment + slot) % availableEnvironmentCount)
                    view.setPlaybackRate(playbackRate)
                    view.setPaused(paused)
                    line.addArrangedSubview(view)
                } else {
                    line.addArrangedSubview(NSView())
                }
            }
            outer.addArrangedSubview(line)
        }
        gridContainer.addSubview(outer)
        NSLayoutConstraint.activate([
            outer.leadingAnchor.constraint(equalTo: gridContainer.leadingAnchor),
            outer.trailingAnchor.constraint(equalTo: gridContainer.trailingAnchor),
            outer.topAnchor.constraint(equalTo: gridContainer.topAnchor),
            outer.bottomAnchor.constraint(equalTo: gridContainer.bottomAnchor),
        ])
        updateToolbar()
    }

    private func updateToolbar() {
        let last = min(availableEnvironmentCount, firstEnvironment + gridCount)
        pageLabel.stringValue = String(
            format: "ENV %02d–%02d of %02d   %.2fx",
            firstEnvironment + 1, last, availableEnvironmentCount, playbackRate)
        runButton.title = paused ? "Run" : "Pause"
        cameraButton.title = views.first?.cameraFollowsBird == true
            ? "Ground View" : "Follow Bird"
    }

    @objc private func stop() {
        paused = true
        visibleViews.forEach { $0.stopAndReset() }
        updateToolbar()
    }
    @objc private func toggleRun() {
        paused.toggle()
        visibleViews.forEach { $0.setPaused(paused) }
        updateToolbar()
    }
    @objc private func slower() {
        playbackRate = max(0.25, playbackRate * 0.5)
        visibleViews.forEach { $0.setPlaybackRate(playbackRate) }
        updateToolbar()
    }
    @objc private func faster() {
        playbackRate = min(4, playbackRate * 2)
        visibleViews.forEach { $0.setPlaybackRate(playbackRate) }
        updateToolbar()
    }
    @objc private func toggleCamera() {
        let follows = !(views.first?.cameraFollowsBird ?? true)
        views.forEach { $0.setCameraFollowsBird(follows) }
        updateToolbar()
    }
    @objc private func previousPage() {
        firstEnvironment = max(0, firstEnvironment - gridCount)
        rebuildGrid()
    }
    @objc private func nextPage() {
        firstEnvironment = (firstEnvironment + gridCount) % availableEnvironmentCount
        rebuildGrid()
    }
    @objc private func changeGrid(_ sender: NSSegmentedControl) {
        gridCount = [1, 4, 9][max(0, sender.selectedSegment)]
        firstEnvironment = min(firstEnvironment, availableEnvironmentCount - 1)
        rebuildGrid()
    }
    private var visibleViews: ArraySlice<RobotView> {
        views.prefix(min(gridCount, views.count))
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

private func lookAt(_ eye: SIMD3<Float>, _ center: SIMD3<Float>, _ up: SIMD3<Float>) -> simd_float4x4 {
    let z=simd_normalize(eye-center), x=simd_normalize(simd_cross(up,z)), y=simd_cross(z,x)
    return simd_float4x4(SIMD4<Float>(x.x,y.x,z.x,0),SIMD4<Float>(x.y,y.y,z.y,0),SIMD4<Float>(x.z,y.z,z.z,0),SIMD4<Float>(-simd_dot(x,eye),-simd_dot(y,eye),-simd_dot(z,eye),1))
}
private func perspective(_ fovy: Float,_ aspect: Float,_ near: Float,_ far: Float)->simd_float4x4 { let y=1/tan(fovy/2),x=y/aspect,z=far/(near-far); return simd_float4x4(SIMD4<Float>(x,0,0,0),SIMD4<Float>(0,y,0,0),SIMD4<Float>(0,0,z,-1),SIMD4<Float>(0,0,z*near,0)) }

@MainActor
@main private enum Main {
    static func main() throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        let fallback = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("ValidationInputs/deetjen-ob-f03-surface-v1/manifest.json")
        var manifestURL = fallback
        var policyURL: URL?
        var metallibPath: String?
        var policySeed: UInt64?
        var measuredSurfaceTask: MetalRoboMeasuredSurfaceTask = .fatalDropRecovery
        var probe = false
        var index = 0
        while index < arguments.count {
            switch arguments[index] {
            case "--probe":
                probe = true
            case "--policy-pack":
                index += 1
                guard index < arguments.count else {
                    throw RobotError.invalid("--policy-pack requires a path")
                }
                policyURL = URL(fileURLWithPath: arguments[index])
            case "--metallib":
                index += 1
                guard index < arguments.count else {
                    throw RobotError.invalid("--metallib requires a path")
                }
                metallibPath = arguments[index]
            case "--seed":
                index += 1
                guard index < arguments.count,
                      let parsed = UInt64(arguments[index]) else {
                    throw RobotError.invalid("--seed requires an unsigned integer")
                }
                policySeed = parsed
            case "--task":
                index += 1
                guard index < arguments.count else {
                    throw RobotError.invalid("--task requires a value")
                }
                switch arguments[index] {
                case "dove-drop-recovery":
                    measuredSurfaceTask = .fatalDropRecovery
                case "dove-cruise":
                    measuredSurfaceTask = .cruise
                case "dove-food-navigation":
                    measuredSurfaceTask = .foodNavigation
                default:
                    throw RobotError.invalid(
                        "--task must be dove-drop-recovery, dove-cruise, or dove-food-navigation")
                }
            default:
                guard !arguments[index].hasPrefix("--") else {
                    throw RobotError.invalid("unknown option \(arguments[index])")
                }
                manifestURL = URL(fileURLWithPath: arguments[index])
            }
            index += 1
        }
        let dataset = try Dataset.load(manifestURL)
        let engine = try RobotEngine(
            dataset: dataset,
            policyURL: policyURL,
            metallibPath: metallibPath,
            seed: policySeed,
            measuredSurfaceTask: measuredSurfaceTask
        )
        if probe {
            let result = try engine.probe()
            print("Measured surface robot GPU probe passed")
            print("  device: \(engine.device.name)")
            print("  dataset: \(dataset.manifest.datasetIdentifier)")
            print("  manifest sha256: \(dataset.manifestHash)")
            print("  vertices: \(dataset.manifest.topology.vertexCount)")
            print("  triangles: \(dataset.manifest.topology.triangleCount)")
            print("  actions: 24")
            print("  zero-action maximum error: \(result.0) m")
            print("  action-conditioned maximum displacement: \(result.1) m")
            print("  free-root displacement: \(result.2) m")
            print("  net load magnitude: \(result.3.x) N")
            print("  net torque magnitude: \(result.3.y) N m")
            print("  transactional finite state: true")
            print("  source periodic: \(dataset.manifest.frames.periodic)")
            print("  quantitative force acceptance: \(dataset.manifest.readiness.quantitativeForceAcceptanceReady)")
            return
        }
        var engines = [engine]
        if policyURL == nil {
            for index in 1..<9 {
                engines.append(try RobotEngine(dataset: dataset, environmentIndex: index))
            }
        }
        let app=NSApplication.shared; let delegate=AppDelegate(engines: engines); app.delegate=delegate; app.setActivationPolicy(.regular); app.run()
        withExtendedLifetime(delegate) {}
    }
}
