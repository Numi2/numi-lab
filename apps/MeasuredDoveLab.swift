import AppKit
import CryptoKit
import Metal
import MetalKit
import simd

private enum DoveLabError: Error, CustomStringConvertible {
    case invalid(String)

    var description: String {
        switch self {
        case .invalid(let message): return "Measured dove: \(message)"
        }
    }
}

private struct DoveComponent: Decodable {
    let name: String
    let partIdentifier: UInt8
    let evidenceClass: String
    let vertexOffset: Int
    let vertexCount: Int
    let triangleOffset: Int
    let triangleCount: Int
}

private struct DoveManifest: Decodable {
    struct Source: Decodable {
        let datasetDOI: String
        let articleDOI: String
        let license: String
        let surfaceSHA256: String
        let muscleModelSHA256: String
    }
    struct Frames: Decodable {
        let count: Int
        let sampleRateHertz: Float
        let frameNumbers: [Int]
        let timesSeconds: [Float]
        let interpolation: String
        let endpointVelocity: String
        let periodic: Bool
    }
    struct Axes: Decodable { let x: String; let y: String; let z: String }
    struct CoordinateFrame: Decodable { let units: String; let birdFlowAxes: Axes }
    struct Topology: Decodable {
        let vertexCount: Int
        let triangleCount: Int
        let indexType: String
        let metalTriangleIdentifierLimit: Int
        let fixedAcrossFrames: Bool
        let components: [DoveComponent]
    }
    struct BinaryMember: Decodable {
        let file: String
        let format: String
        let layout: String
        let bytes: Int
        let sha256: String
    }
    struct Binary: Decodable { let positions: BinaryMember; let triangles: BinaryMember }
    struct Readiness: Decodable {
        let completeBirdSurfaceReady: Bool
        let cpuParityRequired: Bool
        let metalReplayReady: Bool
        let quantitativeForceAcceptanceReady: Bool
    }

    let schemaVersion: Int
    let datasetIdentifier: String
    let scientificTier: String
    let source: Source
    let frames: Frames
    let coordinateFrame: CoordinateFrame
    let topology: Topology
    let binary: Binary
    let readiness: Readiness
}

private struct DoveDataset {
    let manifest: DoveManifest
    let manifestHash: String
    let positions: Data
    let triangles: Data
    let triangleParts: [UInt8]
    let boundsMinimum: SIMD3<Float>
    let boundsMaximum: SIMD3<Float>

    var center: SIMD3<Float> { 0.5 * (boundsMinimum + boundsMaximum) }
    var radius: Float { max(0.05, 0.5 * simd_length(boundsMaximum - boundsMinimum)) }

    static func load(manifestURL: URL) throws -> DoveDataset {
        let manifestData = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(DoveManifest.self, from: manifestData)
        try validateManifest(manifest)
        let positions = try lockedMember(
            manifest.binary.positions,
            beside: manifestURL,
            format: "float32-little-endian",
            layout: "frame-major, vertex-major, xyz"
        )
        let triangles = try lockedMember(
            manifest.binary.triangles,
            beside: manifestURL,
            format: "uint16-little-endian",
            layout: "triangle-major, three global vertex indices"
        )
        let expectedPositions = manifest.frames.count * manifest.topology.vertexCount * 12
        let expectedTriangles = manifest.topology.triangleCount * 6
        guard positions.count == expectedPositions, triangles.count == expectedTriangles else {
            throw DoveLabError.invalid("binary counts disagree with the manifest")
        }
        let indices = triangles.withUnsafeBytes { bytes in
            stride(from: 0, to: triangles.count, by: 2).map {
                UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: $0, as: UInt16.self))
            }
        }
        guard indices.allSatisfy({ Int($0) < manifest.topology.vertexCount }) else {
            throw DoveLabError.invalid("triangle index is out of range")
        }
        var parts = [UInt8](repeating: 0, count: manifest.topology.triangleCount)
        var nextVertex = 0
        var nextTriangle = 0
        for component in manifest.topology.components {
            let end = component.triangleOffset + component.triangleCount
            let vertexEnd = component.vertexOffset + component.vertexCount
            guard component.vertexOffset == nextVertex,
                  component.triangleOffset == nextTriangle,
                  component.vertexCount > 0,
                  component.triangleCount > 0,
                  vertexEnd <= manifest.topology.vertexCount,
                  end <= parts.count else {
                throw DoveLabError.invalid("component ranges are not positive and contiguous")
            }
            for triangle in component.triangleOffset..<end {
                for corner in 0..<3 {
                    let vertex = Int(indices[triangle * 3 + corner])
                    guard vertex >= component.vertexOffset, vertex < vertexEnd else {
                        throw DoveLabError.invalid("triangle crosses a component vertex range")
                    }
                }
            }
            parts.replaceSubrange(component.triangleOffset..<end,
                                  with: repeatElement(component.partIdentifier,
                                                      count: component.triangleCount))
            nextVertex = vertexEnd
            nextTriangle = end
        }
        guard nextVertex == manifest.topology.vertexCount,
              nextTriangle == manifest.topology.triangleCount,
              !parts.contains(0) else {
            throw DoveLabError.invalid("component ranges do not cover every triangle")
        }
        var minimum = SIMD3<Float>(repeating: .infinity)
        var maximum = SIMD3<Float>(repeating: -.infinity)
        try positions.withUnsafeBytes { bytes in
            for offset in stride(from: 0, to: positions.count, by: 12) {
                let point = SIMD3<Float>(
                    decodeFloat(bytes, offset),
                    decodeFloat(bytes, offset + 4),
                    decodeFloat(bytes, offset + 8)
                )
                guard point.x.isFinite, point.y.isFinite, point.z.isFinite else {
                    throw DoveLabError.invalid("position stream contains a nonfinite value")
                }
                minimum = simd_min(minimum, point)
                maximum = simd_max(maximum, point)
            }
        }
        return DoveDataset(
            manifest: manifest,
            manifestHash: sha256(manifestData),
            positions: positions,
            triangles: triangles,
            triangleParts: parts,
            boundsMinimum: minimum,
            boundsMaximum: maximum
        )
    }

    private static func validateManifest(_ value: DoveManifest) throws {
        let expectedNames = ["body", "leftWing", "rightWing", "tail"]
        guard value.schemaVersion == 1,
              value.scientificTier == "derived-measured-complete-surface",
              value.coordinateFrame.units == "meters",
              value.coordinateFrame.birdFlowAxes.x == "forward",
              value.coordinateFrame.birdFlowAxes.y == "left",
              value.coordinateFrame.birdFlowAxes.z == "up",
              value.frames.count >= 2,
              value.frames.frameNumbers.count == value.frames.count,
              value.frames.timesSeconds.count == value.frames.count,
              value.frames.sampleRateHertz > 0,
              value.frames.interpolation == "piecewise-linear-nonperiodic",
              value.frames.endpointVelocity == "one-sided-adjacent-frame",
              !value.frames.periodic,
              zip(value.frames.timesSeconds, value.frames.timesSeconds.dropFirst())
                .allSatisfy({ $0.isFinite && $1.isFinite && $0 < $1 }),
              value.topology.vertexCount > 0,
              value.topology.vertexCount <= Int(UInt16.max),
              value.topology.triangleCount > 0,
              value.topology.triangleCount <= value.topology.metalTriangleIdentifierLimit,
              value.topology.metalTriangleIdentifierLimit <= 4096,
              value.topology.indexType == "uint16-little-endian",
              value.topology.fixedAcrossFrames,
              value.topology.components.map(\.name) == expectedNames,
              value.topology.components.map(\.partIdentifier) == [1, 2, 3, 4],
              value.readiness.completeBirdSurfaceReady,
              value.readiness.cpuParityRequired,
              !value.readiness.quantitativeForceAcceptanceReady else {
            throw DoveLabError.invalid("manifest violates the measured-surface contract")
        }
    }

    private static func lockedMember(
        _ member: DoveManifest.BinaryMember,
        beside manifestURL: URL,
        format: String,
        layout: String
    ) throws -> Data {
        guard !member.file.isEmpty,
              URL(fileURLWithPath: member.file).lastPathComponent == member.file,
              !member.file.contains(".."),
              member.format == format,
              member.layout == layout else {
            throw DoveLabError.invalid("unsafe or incompatible binary member")
        }
        let data = try Data(contentsOf: manifestURL.deletingLastPathComponent()
            .appendingPathComponent(member.file))
        guard data.count == member.bytes, sha256(data) == member.sha256 else {
            throw DoveLabError.invalid("hash lock failed for \(member.file)")
        }
        return data
    }

    private static func decodeFloat(_ bytes: UnsafeRawBufferPointer, _ offset: Int) -> Float {
        Float(bitPattern: UInt32(littleEndian:
            bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self)))
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    func point(frame: Int, vertex: Int) -> SIMD3<Float> {
        let offset = (frame * manifest.topology.vertexCount + vertex) * 12
        return positions.withUnsafeBytes { bytes in
            SIMD3<Float>(
                Self.decodeFloat(bytes, offset),
                Self.decodeFloat(bytes, offset + 4),
                Self.decodeFloat(bytes, offset + 8)
            )
        }
    }
}

private struct DoveUniforms {
    var viewProjection: simd_float4x4
    var eyeAndBlend: SIMD4<Float>
    var frameInfo: SIMD4<UInt32>
    var timingAndStyle: SIMD4<Float>
}

private final class DoveRenderer {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let dataset: DoveDataset
    private let surfacePipeline: MTLRenderPipelineState
    private let backgroundPipeline: MTLRenderPipelineState
    private let interpolationPipeline: MTLComputePipelineState
    private let depthWrite: MTLDepthStencilState
    private let positions: MTLBuffer
    private let triangles: MTLBuffer
    private let parts: MTLBuffer

    init(dataset: DoveDataset) throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw DoveLabError.invalid("Metal device or queue is unavailable")
        }
        self.device = device
        self.queue = queue
        self.dataset = dataset
        guard let positions = dataset.positions.withUnsafeBytes({ raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: raw.count,
                              options: .storageModeShared)
        }), let triangles = dataset.triangles.withUnsafeBytes({ raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: raw.count,
                              options: .storageModeShared)
        }), let parts = device.makeBuffer(bytes: dataset.triangleParts,
                                           length: dataset.triangleParts.count,
                                           options: .storageModeShared) else {
            throw DoveLabError.invalid("could not allocate measured surface buffers")
        }
        self.positions = positions
        self.triangles = triangles
        self.parts = parts
        let library = try device.makeLibrary(source: Self.shaderSource, options: nil)
        let surface = MTLRenderPipelineDescriptor()
        surface.label = "Numi measured dove surface"
        surface.vertexFunction = library.makeFunction(name: "doveVertex")
        surface.fragmentFunction = library.makeFunction(name: "doveFragment")
        surface.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
        surface.depthAttachmentPixelFormat = .depth32Float
        surfacePipeline = try device.makeRenderPipelineState(descriptor: surface)
        let background = MTLRenderPipelineDescriptor()
        background.label = "Numi measured dove laboratory background"
        background.vertexFunction = library.makeFunction(name: "backgroundVertex")
        background.fragmentFunction = library.makeFunction(name: "backgroundFragment")
        background.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
        background.depthAttachmentPixelFormat = .depth32Float
        backgroundPipeline = try device.makeRenderPipelineState(descriptor: background)
        guard let interpolation = library.makeFunction(name: "interpolateDove") else {
            throw DoveLabError.invalid("measured interpolation kernel is unavailable")
        }
        interpolationPipeline = try device.makeComputePipelineState(function: interpolation)
        let depth = MTLDepthStencilDescriptor()
        depth.depthCompareFunction = .less
        depth.isDepthWriteEnabled = true
        guard let depthState = device.makeDepthStencilState(descriptor: depth) else {
            throw DoveLabError.invalid("could not create depth state")
        }
        depthWrite = depthState
    }

    func encode(
        pass: MTLRenderPassDescriptor,
        command: MTLCommandBuffer,
        size: CGSize,
        time: Float,
        yaw: Float,
        pitch: Float,
        distanceScale: Float,
        background: Bool
    ) throws {
        let times = dataset.manifest.frames.timesSeconds
        let clamped = min(max(time, times[0]), times[times.count - 1])
        let upper = times.firstIndex(where: { $0 >= clamped }) ?? times.count - 1
        let first = max(0, upper - (upper == 0 ? 0 : 1))
        let second = min(times.count - 1, max(first + 1, upper))
        let duration = max(1e-6, times[second] - times[first])
        let blend = first == second ? 0 : (clamped - times[first]) / duration
        let target = dataset.center
        let distance = dataset.radius * max(1.5, distanceScale)
        let eye = target + distance * SIMD3<Float>(
            cos(pitch) * cos(yaw), cos(pitch) * sin(yaw), sin(pitch)
        )
        let aspect = Float(max(1, size.width) / max(1, size.height))
        var uniforms = DoveUniforms(
            viewProjection: perspective(fovy: 0.72, aspect: aspect,
                                        near: max(0.001, dataset.radius * 0.01),
                                        far: dataset.radius * 20)
                * lookAt(eye: eye, center: target, up: SIMD3<Float>(0, 0, 1)),
            eyeAndBlend: SIMD4<Float>(eye, blend),
            frameInfo: SIMD4<UInt32>(UInt32(first), UInt32(second),
                                     UInt32(dataset.manifest.topology.vertexCount),
                                     UInt32(dataset.manifest.topology.triangleCount)),
            timingAndStyle: SIMD4<Float>(duration, clamped,
                                         dataset.manifest.frames.sampleRateHertz, 0)
        )
        guard let encoder = command.makeRenderCommandEncoder(descriptor: pass) else {
            throw DoveLabError.invalid("could not create render encoder")
        }
        encoder.label = "Measured Deetjen dove replay"
        if background {
            encoder.setRenderPipelineState(backgroundPipeline)
            encoder.setFragmentBytes(&uniforms, length: MemoryLayout<DoveUniforms>.stride,
                                     index: 0)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        }
        encoder.setRenderPipelineState(surfacePipeline)
        encoder.setDepthStencilState(depthWrite)
        encoder.setCullMode(.none)
        encoder.setVertexBuffer(positions, offset: 0, index: 0)
        encoder.setVertexBuffer(triangles, offset: 0, index: 1)
        encoder.setVertexBuffer(parts, offset: 0, index: 2)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<DoveUniforms>.stride,
                               index: 3)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<DoveUniforms>.stride,
                                 index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0,
                               vertexCount: dataset.manifest.topology.triangleCount * 3)
        encoder.endEncoding()
    }

    func probe() throws -> (coveredPixels: Int, maximumPositionError: Float) {
        let width = 1024, height = 640
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm_srgb, width: width, height: height, mipmapped: false)
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = [.renderTarget]
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: width, height: height, mipmapped: false)
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = [.renderTarget]
        guard let color = device.makeTexture(descriptor: colorDescriptor),
              let depth = device.makeTexture(descriptor: depthDescriptor),
              let interpolated = device.makeBuffer(
                length: dataset.manifest.topology.vertexCount * MemoryLayout<SIMD4<Float>>.stride,
                options: .storageModeShared),
              let command = queue.makeCommandBuffer() else {
            throw DoveLabError.invalid("probe texture allocation failed")
        }
        let parityFrame = max(0, dataset.manifest.frames.count / 2 - 1)
        let parityBlend: Float = 0.371
        let parityDuration = dataset.manifest.frames.timesSeconds[parityFrame + 1]
            - dataset.manifest.frames.timesSeconds[parityFrame]
        var parityUniforms = DoveUniforms(
            viewProjection: matrix_identity_float4x4,
            eyeAndBlend: SIMD4<Float>(0, 0, 0, parityBlend),
            frameInfo: SIMD4<UInt32>(UInt32(parityFrame), UInt32(parityFrame + 1),
                                     UInt32(dataset.manifest.topology.vertexCount),
                                     UInt32(dataset.manifest.topology.triangleCount)),
            timingAndStyle: SIMD4<Float>(parityDuration, 0,
                                         dataset.manifest.frames.sampleRateHertz, 0)
        )
        guard let compute = command.makeComputeCommandEncoder() else {
            throw DoveLabError.invalid("could not create interpolation probe encoder")
        }
        compute.label = "Measured dove interpolation parity"
        compute.setComputePipelineState(interpolationPipeline)
        compute.setBuffer(positions, offset: 0, index: 0)
        compute.setBuffer(interpolated, offset: 0, index: 1)
        compute.setBytes(&parityUniforms, length: MemoryLayout<DoveUniforms>.stride, index: 2)
        let threads = min(interpolationPipeline.maxTotalThreadsPerThreadgroup,
                          max(1, interpolationPipeline.threadExecutionWidth * 4))
        compute.dispatchThreads(
            MTLSize(width: dataset.manifest.topology.vertexCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        compute.endEncoding()
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1)
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
        pass.depthAttachment.clearDepth = 1
        let times = dataset.manifest.frames.timesSeconds
        try encode(pass: pass, command: command,
                   size: CGSize(width: width, height: height),
                   time: times[times.count / 2], yaw: -0.78, pitch: 0.30,
                   distanceScale: 2.5, background: false)
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
            throw DoveLabError.invalid("Metal probe failed: \(command.error?.localizedDescription ?? "unknown error")")
        }
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        color.getBytes(&pixels, bytesPerRow: width * 4,
                       from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        let covered = stride(from: 0, to: pixels.count, by: 4).reduce(0) {
            $0 + ((pixels[$1] > 4 || pixels[$1 + 1] > 4 || pixels[$1 + 2] > 4) ? 1 : 0)
        }
        guard covered > 1_000 else {
            throw DoveLabError.invalid("Metal probe produced insufficient dove coverage (\(covered) pixels)")
        }
        let gpu = interpolated.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: dataset.manifest.topology.vertexCount)
        var maximumError: Float = 0
        for vertex in 0..<dataset.manifest.topology.vertexCount {
            let first = dataset.point(frame: parityFrame, vertex: vertex)
            let second = dataset.point(frame: parityFrame + 1, vertex: vertex)
            let expected = first + parityBlend * (second - first)
            maximumError = max(maximumError,
                               simd_length(SIMD3<Float>(gpu[vertex].x, gpu[vertex].y,
                                                       gpu[vertex].z) - expected))
        }
        guard maximumError <= 1e-6 else {
            throw DoveLabError.invalid("GPU interpolation parity failed (max error \(maximumError) m)")
        }
        return (covered, maximumError)
    }

    private static let shaderSource = #"""
    #include <metal_stdlib>
    using namespace metal;
    struct Uniforms { float4x4 viewProjection; float4 eyeAndBlend; uint4 frameInfo; float4 timingAndStyle; };
    struct Raster { float4 position [[position]]; float3 world; float3 normal; float4 color; };
    float3 samplePosition(device const packed_float3* p, uint frame, uint vertexIndex, uint count) {
        return float3(p[frame * count + vertexIndex]);
    }
    kernel void interpolateDove(device const packed_float3* positions [[buffer(0)]],
                                device float4* output [[buffer(1)]],
                                constant Uniforms& u [[buffer(2)]], uint id [[thread_position_in_grid]]) {
        if (id >= u.frameInfo.z) return;
        float3 first=samplePosition(positions,u.frameInfo.x,id,u.frameInfo.z);
        float3 second=samplePosition(positions,u.frameInfo.y,id,u.frameInfo.z);
        output[id]=float4(mix(first,second,u.eyeAndBlend.w),1);
    }
    vertex Raster doveVertex(device const packed_float3* positions [[buffer(0)]],
                             device const ushort* indices [[buffer(1)]],
                             device const uchar* parts [[buffer(2)]],
                             constant Uniforms& u [[buffer(3)]], uint vid [[vertex_id]]) {
        uint triangle = vid / 3u, corner = vid % 3u, count = u.frameInfo.z;
        ushort3 index = ushort3(indices[triangle * 3u], indices[triangle * 3u + 1u], indices[triangle * 3u + 2u]);
        float3 a0 = samplePosition(positions, u.frameInfo.x, index.x, count);
        float3 b0 = samplePosition(positions, u.frameInfo.x, index.y, count);
        float3 c0 = samplePosition(positions, u.frameInfo.x, index.z, count);
        float3 a1 = samplePosition(positions, u.frameInfo.y, index.x, count);
        float3 b1 = samplePosition(positions, u.frameInfo.y, index.y, count);
        float3 c1 = samplePosition(positions, u.frameInfo.y, index.z, count);
        float3 a = mix(a0, a1, u.eyeAndBlend.w), b = mix(b0, b1, u.eyeAndBlend.w), c = mix(c0, c1, u.eyeAndBlend.w);
        float3 point = corner == 0u ? a : (corner == 1u ? b : c);
        float3 normal = normalize(cross(b - a, c - a));
        float speed = (length(a1-a0) + length(b1-b0) + length(c1-c0)) / (3.0f * max(1e-6f, u.timingAndStyle.x));
        uint part = uint(parts[triangle]);
        float3 base = part == 1u ? float3(0.54f,0.62f,0.72f) : (part == 2u ? float3(0.08f,0.55f,0.96f) : (part == 3u ? float3(1.0f,0.30f,0.09f) : float3(0.76f,0.28f,0.90f)));
        base = mix(base, float3(0.88f,0.96f,1.0f), clamp(speed / 25.2305f, 0.0f, 1.0f) * 0.28f);
        Raster out; out.position = u.viewProjection * float4(point,1); out.world = point; out.normal = normal; out.color = float4(base,1); return out;
    }
    fragment float4 doveFragment(Raster in [[stage_in]], constant Uniforms& u [[buffer(0)]]) {
        float3 n=normalize(in.normal), v=normalize(u.eyeAndBlend.xyz-in.world), key=normalize(float3(0.38f,-0.48f,0.80f));
        float diffuse=0.24f+0.70f*abs(dot(n,key));
        float rim=pow(1.0f-abs(dot(n,v)),2.25f);
        float specular=pow(abs(dot(n,normalize(key+v))),42.0f);
        float3 color=in.color.rgb*diffuse + rim*mix(float3(0.03f,0.24f,0.46f),in.color.rgb,0.32f) + 0.24f*specular;
        return float4(1.0f-exp(-1.12f*color),1);
    }
    vertex Raster backgroundVertex(uint vid [[vertex_id]]) {
        float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)}; Raster o; o.position=float4(p[vid],0.999f,1); o.world=float3(0); o.normal=float3(0,0,1); o.color=float4(1); return o;
    }
    fragment float4 backgroundFragment(Raster in [[stage_in]], constant Uniforms& u [[buffer(0)]]) {
        float pulse=0.5f+0.5f*cos(6.2831853f*u.timingAndStyle.y*2.0f);
        return float4(float3(0.004f,0.009f,0.022f)+pulse*float3(0.003f,0.010f,0.016f),1);
    }
    """#
}

private final class DoveMetalView: MTKView {
    var drag: ((Float, Float) -> Void)?
    var zoom: ((Float) -> Void)?
    var key: ((Character) -> Void)?
    private var lastPoint = NSPoint.zero
    override var acceptsFirstResponder: Bool { true }
    override func mouseDown(with event: NSEvent) { lastPoint = convert(event.locationInWindow, from: nil) }
    override func mouseDragged(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        drag?(Float(point.x-lastPoint.x), Float(point.y-lastPoint.y)); lastPoint = point
    }
    override func scrollWheel(with event: NSEvent) { zoom?(Float(event.scrollingDeltaY)) }
    override func keyDown(with event: NSEvent) {
        if let character = event.charactersIgnoringModifiers?.first { key?(character) }
        else { super.keyDown(with: event) }
    }
}

@MainActor
private final class DoveViewController: NSObject, MTKViewDelegate {
    let renderer: DoveRenderer
    let view: DoveMetalView
    var updateTitle: ((String) -> Void)?
    private var sourceTime: Float
    private var lastWallTime = CACurrentMediaTime()
    private var playing = true
    private var yaw: Float = -0.78
    private var pitch: Float = 0.30
    private var distanceScale: Float = 2.5
    // The source clip is only 143 ms. Slow it without inventing frames or a
    // periodic closure so the measured deformation remains inspectable.
    private let playbackRate: Float = 0.02

    init(renderer: DoveRenderer) {
        self.renderer = renderer
        sourceTime = renderer.dataset.manifest.frames.timesSeconds[0]
        view = DoveMetalView(frame: .zero, device: renderer.device)
        super.init()
        view.delegate = self
        view.colorPixelFormat = .bgra8Unorm_srgb
        view.depthStencilPixelFormat = .depth32Float
        view.clearColor = MTLClearColorMake(0.004, 0.009, 0.022, 1)
        view.preferredFramesPerSecond = 60
        view.framebufferOnly = true
        view.drag = { [weak self] dx, dy in
            self?.yaw -= dx * 0.006; self?.pitch = min(1.25, max(-1.1, (self?.pitch ?? 0) + dy * 0.006))
        }
        view.zoom = { [weak self] delta in
            self?.distanceScale = min(7, max(1.5, (self?.distanceScale ?? 2.5) * exp(delta * 0.018)))
        }
        view.key = { [weak self] character in
            guard let self else { return }
            if character == " " { self.playing.toggle() }
            if character == "r" || character == "R" { self.restart() }
        }
    }

    func restart() {
        sourceTime = renderer.dataset.manifest.frames.timesSeconds[0]
        lastWallTime = CACurrentMediaTime(); playing = true
    }

    func draw(in view: MTKView) {
        let now = CACurrentMediaTime()
        if playing { sourceTime += Float(now-lastWallTime) * playbackRate }
        lastWallTime = now
        let times = renderer.dataset.manifest.frames.timesSeconds
        if sourceTime >= times.last! { sourceTime = times.last!; playing = false }
        guard let pass = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let command = renderer.queue.makeCommandBuffer() else { return }
        do {
            try renderer.encode(pass: pass, command: command, size: view.drawableSize,
                                time: sourceTime, yaw: yaw, pitch: pitch,
                                distanceScale: distanceScale, background: true)
            command.present(drawable); command.commit()
            let frame = min(times.count-1, max(0, Int(sourceTime * renderer.dataset.manifest.frames.sampleRateHertz)))
            updateTitle?("Numi Lab · measured dove · source frame \(renderer.dataset.manifest.frames.frameNumbers[frame]) · \(String(format: "%.3f", sourceTime)) s · \(playing ? "playing 0.02×" : "paused")")
        } catch { updateTitle?("Numi Lab · measured dove · \(error)") }
    }
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
}

@MainActor
private final class DoveAppDelegate: NSObject, NSApplicationDelegate {
    let renderer: DoveRenderer
    private var window: NSWindow?
    private var controller: DoveViewController?
    init(renderer: DoveRenderer) { self.renderer = renderer }
    func applicationDidFinishLaunching(_ notification: Notification) {
        let controller = DoveViewController(renderer: renderer)
        let window = NSWindow(contentRect: NSRect(x: 100, y: 100, width: 1100, height: 720),
                              styleMask: [.titled, .closable, .miniaturizable, .resizable],
                              backing: .buffered, defer: false)
        window.title = "Numi Lab · measured dove"
        window.contentView = controller.view
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(controller.view)
        controller.updateTitle = { [weak window] in window?.title = $0 }
        self.controller = controller; self.window = window
        NSApp.activate(ignoringOtherApps: true)
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

private func lookAt(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
    let forward = simd_normalize(center-eye)
    let right = simd_normalize(simd_cross(forward, up))
    let correctedUp = simd_cross(right, forward)
    return simd_float4x4(columns: (
        SIMD4<Float>(right.x, correctedUp.x, -forward.x, 0),
        SIMD4<Float>(right.y, correctedUp.y, -forward.y, 0),
        SIMD4<Float>(right.z, correctedUp.z, -forward.z, 0),
        SIMD4<Float>(-simd_dot(right, eye), -simd_dot(correctedUp, eye), simd_dot(forward, eye), 1)
    ))
}

private func perspective(fovy: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
    let y = 1 / tan(0.5*fovy), x = y/max(0.01, aspect)
    let z = far/(near-far), w = near*far/(near-far)
    return simd_float4x4(columns: (
        SIMD4<Float>(x,0,0,0), SIMD4<Float>(0,y,0,0),
        SIMD4<Float>(0,0,z,-1), SIMD4<Float>(0,0,w,0)
    ))
}

@main
private struct MeasuredDoveLabMain {
    @MainActor static func main() {
        do {
            let arguments = Array(CommandLine.arguments.dropFirst())
            let probe = arguments.first == "--probe"
            let path = probe ? arguments.dropFirst().first : arguments.first
            guard let path else {
                throw DoveLabError.invalid("usage: metalrobo_measured_dove_lab [--probe] MANIFEST")
            }
            let dataset = try DoveDataset.load(manifestURL: URL(fileURLWithPath: path))
            let renderer = try DoveRenderer(dataset: dataset)
            if probe {
                let result = try renderer.probe()
                let m = dataset.manifest
                print("device=\"\(renderer.device.name)\" dataset=\(m.datasetIdentifier) frames=\(m.frames.count) vertices=\(m.topology.vertexCount) triangles=\(m.topology.triangleCount) gpu_interpolation=passed maximum_position_error_m=\(result.maximumPositionError) covered_pixels=\(result.coveredPixels) manifest_sha256=\(dataset.manifestHash) positions_sha256=\(m.binary.positions.sha256) triangles_sha256=\(m.binary.triangles.sha256) source_periodic=\(m.frames.periodic) quantitative_force_acceptance=\(m.readiness.quantitativeForceAcceptanceReady)")
                return
            }
            let app = NSApplication.shared
            app.setActivationPolicy(.regular)
            let delegate = DoveAppDelegate(renderer: renderer)
            app.delegate = delegate
            app.run()
            withExtendedLifetime(delegate) {}
        } catch {
            fputs("metalrobo_measured_dove_lab: \(error)\n", stderr)
            Foundation.exit(1)
        }
    }
}
