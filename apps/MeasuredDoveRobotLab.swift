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
    let name: String
    let partIdentifier: UInt8
    let vertexOffset: Int
    let vertexCount: Int
    let triangleOffset: Int
    let triangleCount: Int
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
        var minimum = SIMD3<Float>(repeating: .infinity)
        var maximum = SIMD3<Float>(repeating: -.infinity)
        try positions.withUnsafeBytes { raw in
            for offset in stride(from: 0, to: raw.count, by: 12) {
                let point = SIMD3<Float>(decode(raw, offset), decode(raw, offset + 4), decode(raw, offset + 8))
                guard allFinite(point) else { throw RobotError.invalid("nonfinite measured position") }
                minimum = simd_min(minimum, point); maximum = simd_max(maximum, point)
            }
        }
        return Dataset(manifest: value, manifestHash: hash(manifestData), positions: positions,
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

private struct Uniforms {
    var viewProjection = matrix_identity_float4x4
    var eye = SIMD4<Float>(0, 0, 0, 1)
    var centerRadius = SIMD4<Float>(0, 0, 0, 1)
    var boundsMinimum = SIMD4<Float>(0, 0, 0, 0)
    var boundsMaximum = SIMD4<Float>(0, 0, 0, 0)
    var frameInfo = SIMD4<UInt32>(0, 1, 0, 0)
    var timing = SIMD4<Float>(0, 0.001, 0, 0)
}

private final class RobotEngine {
    let dataset: Dataset
    let device: MTLDevice
    private let queue: MTLCommandQueue
    private let integratePipeline: MTLComputePipelineState
    private let deformPipeline: MTLComputePipelineState
    private let rootPipeline: MTLComputePipelineState
    private let copyPipeline: MTLComputePipelineState
    private let renderPipeline: MTLRenderPipelineState
    private let positions: MTLBuffer
    private let triangles: MTLBuffer
    private let vertexParts: MTLBuffer
    private let triangleParts: MTLBuffer
    private let actuatorTargets: MTLBuffer
    private let actuatorState: MTLBuffer
    private let localSurface: MTLBuffer
    private let previousSurface: MTLBuffer
    private let rootState: MTLBuffer
    private let metrics: MTLBuffer
    private var initialized = false
    private(set) var simulationTime: Float = 0
    var controllerEnabled = true

    init(dataset: Dataset) throws {
        self.dataset = dataset
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
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = library.makeFunction(name: "robotVertex")
        descriptor.fragmentFunction = library.makeFunction(name: "robotFragment")
        descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
        descriptor.depthAttachmentPixelFormat = .depth32Float
        renderPipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        func buffer(_ data: Data, _ label: String) throws -> MTLBuffer {
            guard let result = data.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }) else {
                throw RobotError.invalid("could not allocate \(label)")
            }
            result.label = label; return result
        }
        positions = try buffer(dataset.positions, "provenance-locked positions")
        triangles = try buffer(dataset.triangles, "provenance-locked triangles")
        guard let vp = device.makeBuffer(bytes: dataset.vertexParts, length: dataset.vertexParts.count, options: .storageModeShared),
              let tp = device.makeBuffer(bytes: dataset.triangleParts, length: dataset.triangleParts.count, options: .storageModeShared),
              let targets = device.makeBuffer(length: 24 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let state = device.makeBuffer(length: 24 * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared),
              let local = device.makeBuffer(length: dataset.manifest.topology.vertexCount * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared),
              let previous = device.makeBuffer(length: dataset.manifest.topology.vertexCount * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared),
              let root = device.makeBuffer(length: 8 * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared),
              let metricBuffer = device.makeBuffer(length: 2 * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared) else {
            throw RobotError.invalid("could not allocate robot state")
        }
        vertexParts = vp; triangleParts = tp; actuatorTargets = targets; actuatorState = state
        localSurface = local; previousSurface = previous; rootState = root; metrics = metricBuffer
        reset()
    }

    func reset() {
        actuatorTargets.contents().initializeMemory(as: UInt8.self, repeating: 0, count: actuatorTargets.length)
        actuatorState.contents().initializeMemory(as: UInt8.self, repeating: 0, count: actuatorState.length)
        rootState.contents().initializeMemory(as: UInt8.self, repeating: 0, count: rootState.length)
        metrics.contents().initializeMemory(as: UInt8.self, repeating: 0, count: metrics.length)
        rootState.contents().bindMemory(to: SIMD4<Float>.self, capacity: 8)[2] = SIMD4<Float>(0, 0, 0, 1)
        simulationTime = 0; initialized = false
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
        return Uniforms(centerRadius: SIMD4<Float>(dataset.center, dataset.radius),
                        boundsMinimum: SIMD4<Float>(dataset.minimum, 0),
                        boundsMaximum: SIMD4<Float>(dataset.maximum, 0),
                        frameInfo: SIMD4<UInt32>(UInt32(f.0), UInt32(f.1), UInt32(dataset.manifest.topology.vertexCount), UInt32(dataset.manifest.topology.triangleCount)),
                        timing: SIMD4<Float>(f.2, dt, time, f.3))
    }

    private func setControllerTargets(time: Float) {
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
        let dt: Float = 0.001
        for _ in 0..<count {
            let end = dataset.manifest.frames.timesSeconds.last!
            if simulationTime + dt > end { reset() }
            setControllerTargets(time: simulationTime)
            var uniforms = baseUniforms(time: simulationTime, dt: dt)
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
            if initialized {
                try dispatch(rootPipeline, count: 1,
                             buffers: [(localSurface, 0), (previousSurface, 1), (triangles, 2), (rootState, 3), (metrics, 4)])
            }
            try dispatch(copyPipeline, count: dataset.manifest.topology.vertexCount,
                         buffers: [(localSurface, 0), (previousSurface, 1)])
            command.commit(); command.waitUntilCompleted()
            guard command.status == .completed else { throw RobotError.invalid("Metal mechanics failed: \(command.error?.localizedDescription ?? "unknown")") }
            initialized = true; simulationTime += dt
        }
    }

    func render(pass: MTLRenderPassDescriptor, drawable: MTLDrawable, size: CGSize,
                yaw: Float, pitch: Float, distanceScale: Float) throws {
        let target = dataset.center
        let eye = target + dataset.radius * distanceScale * SIMD3<Float>(cos(pitch) * cos(yaw), cos(pitch) * sin(yaw), sin(pitch))
        var uniforms = baseUniforms(time: simulationTime, dt: 0.001)
        uniforms.eye = SIMD4<Float>(eye, 1)
        uniforms.viewProjection = perspective(0.72, Float(max(1, size.width) / max(1, size.height)), dataset.radius * 0.01, dataset.radius * 20) * lookAt(eye, target, SIMD3<Float>(0, 0, 1))
        guard let command = queue.makeCommandBuffer(), let encoder = command.makeRenderCommandEncoder(descriptor: pass) else {
            throw RobotError.invalid("render encoder failed")
        }
        encoder.setRenderPipelineState(renderPipeline); encoder.setCullMode(.none)
        encoder.setVertexBuffer(localSurface, offset: 0, index: 0)
        encoder.setVertexBuffer(triangles, offset: 0, index: 1)
        encoder.setVertexBuffer(triangleParts, offset: 0, index: 2)
        encoder.setVertexBuffer(rootState, offset: 0, index: 3)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 8)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 8)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: dataset.manifest.topology.triangleCount * 3)
        encoder.endEncoding(); command.present(drawable); command.commit()
    }

    func probe() throws -> (Float, Float, Float, SIMD4<Float>) {
        controllerEnabled = false; reset(); try step()
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
        let responseFrame = frame(at: max(0, simulationTime - 0.001))
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
        let root = rootState.contents().bindMemory(to: SIMD4<Float>.self, capacity: 8)[0]
        let metric = metrics.contents().bindMemory(to: SIMD4<Float>.self, capacity: 2)[0]
        return String(format: "t %.3f s  controller %@  root %.3f m  aero %.3f N", simulationTime,
                      controllerEnabled ? "ON" : "OFF", simd_length(SIMD3<Float>(root.x, root.y, root.z)), metric.x)
    }

    private static func allFinite(_ p: SIMD4<Float>) -> Bool { p.x.isFinite && p.y.isFinite && p.z.isFinite && p.w.isFinite }

    private static let metalSource = #"""
    #include <metal_stdlib>
    using namespace metal;
    struct Uniforms { float4x4 vp; float4 eye; float4 centerRadius; float4 bmin; float4 bmax; uint4 frame; float4 timing; };
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
            float3 r=p-float3(u.bmin.x,u.centerRadius.y,u.centerRadius.z); r=rotateAxis(r,float3(0,1,0),-0.30f*q[20].x); r=rotateAxis(r,float3(0,0,1),0.25f*q[21].x); r=rotateAxis(r,float3(1,0,0),0.28f*q[22].x); r.y*=1.0f+0.22f*q[23].x; p=float3(u.bmin.x,u.centerRadius.y,u.centerRadius.z)+r;
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
    struct Raster { float4 position [[position]]; float3 world; float3 normal; float4 color; };
    float3 quatRotate(float4 q,float3 p){return p+2*cross(q.xyz,cross(q.xyz,p)+q.w*p);}
    vertex Raster robotVertex(device const float4* surface [[buffer(0)]], device const ushort* indices [[buffer(1)]], device const uchar* parts [[buffer(2)]], device const float4* root [[buffer(3)]], constant Uniforms& u [[buffer(8)]], uint vid [[vertex_id]]) {
        uint tri=vid/3,corner=vid%3; ushort3 ix=ushort3(indices[tri*3],indices[tri*3+1],indices[tri*3+2]); float3 a=surface[ix.x].xyz,b=surface[ix.y].xyz,c=surface[ix.z].xyz; float3 local=corner==0?a:(corner==1?b:c); float3 world=quatRotate(root[2],local-u.centerRadius.xyz)+u.centerRadius.xyz+root[0].xyz; float3 n=normalize(quatRotate(root[2],cross(b-a,c-a))); uint part=parts[tri]; float3 base=part==1?float3(.50f,.58f,.68f):(part==2?float3(.03f,.58f,1.0f):(part==3?float3(1.0f,.25f,.06f):float3(.72f,.22f,.92f))); float act=surface[corner==0?ix.x:(corner==1?ix.y:ix.z)].w; Raster o;o.position=u.vp*float4(world,1);o.world=world;o.normal=n;o.color=float4(mix(base,float3(.2f,1.0f,.65f),clamp(act/.012f,0.0f,1.0f)*.7f),1);return o;
    }
    fragment float4 robotFragment(Raster in [[stage_in]], constant Uniforms& u [[buffer(8)]]) { float3 n=normalize(in.normal),v=normalize(u.eye.xyz-in.world),l=normalize(float3(.4,-.5,.8));float d=.2+.75*abs(dot(n,l)),rim=pow(1-abs(dot(n,v)),2.2);return float4(1-exp(-1.15*(in.color.rgb*d+rim*.18*in.color.rgb)),1); }
    """#
}

private final class RobotView: MTKView, MTKViewDelegate {
    let engine: RobotEngine
    private var yaw: Float = -0.78, pitch: Float = 0.28, distanceScale: Float = 2.5
    private var lastPoint: CGPoint?
    private var simulationPaused = false
    init(engine: RobotEngine, frame: CGRect) {
        self.engine = engine
        super.init(frame: frame, device: engine.device)
        colorPixelFormat = .bgra8Unorm_srgb; depthStencilPixelFormat = .depth32Float
        clearColor = MTLClearColorMake(0.006, 0.012, 0.025, 1); preferredFramesPerSecond = 60
        delegate = self
    }
    required init(coder: NSCoder) { fatalError() }
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
    func draw(in view: MTKView) {
        do {
            if !simulationPaused { try engine.step(count: 2) }
            guard let pass = currentRenderPassDescriptor, let drawable = currentDrawable else { return }
            try engine.render(pass: pass, drawable: drawable, size: drawableSize, yaw: yaw, pitch: pitch, distanceScale: distanceScale)
            window?.title = "Numi Surface Robot — \(engine.status)"
        } catch { window?.title = "Numi Surface Robot — \(error)"; simulationPaused = true }
    }
    override var acceptsFirstResponder: Bool { true }
    override func keyDown(with event: NSEvent) {
        switch event.charactersIgnoringModifiers?.lowercased() {
        case " ": simulationPaused.toggle()
        case "r": engine.reset()
        case "p": engine.controllerEnabled.toggle()
        default: super.keyDown(with: event)
        }
    }
    override func mouseDown(with event: NSEvent) { lastPoint = convert(event.locationInWindow, from: nil) }
    override func mouseDragged(with event: NSEvent) { let p=convert(event.locationInWindow,from:nil); if let q=lastPoint { yaw += Float(p.x-q.x)*0.008; pitch = min(1.3,max(-1.3,pitch+Float(p.y-q.y)*0.008)) }; lastPoint=p }
    override func mouseUp(with event: NSEvent) { lastPoint=nil }
    override func scrollWheel(with event: NSEvent) { distanceScale=min(8,max(1.4,distanceScale*exp(Float(event.scrollingDeltaY)*0.015))) }
}

private final class AppDelegate: NSObject, NSApplicationDelegate {
    let engine: RobotEngine; var window: NSWindow?
    init(engine: RobotEngine) { self.engine = engine }
    func applicationDidFinishLaunching(_ notification: Notification) {
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1180, height: 760), styleMask: [.titled,.closable,.miniaturizable,.resizable], backing: .buffered, defer: false)
        let view = RobotView(engine: engine, frame: window.contentView!.bounds); view.autoresizingMask=[.width,.height]
        window.contentView = view; window.center(); window.makeKeyAndOrderFront(nil); window.makeFirstResponder(view)
        self.window=window; NSApp.activate(ignoringOtherApps: true)
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

private func lookAt(_ eye: SIMD3<Float>, _ center: SIMD3<Float>, _ up: SIMD3<Float>) -> simd_float4x4 {
    let z=simd_normalize(eye-center), x=simd_normalize(simd_cross(up,z)), y=simd_cross(z,x)
    return simd_float4x4(SIMD4<Float>(x.x,y.x,z.x,0),SIMD4<Float>(x.y,y.y,z.y,0),SIMD4<Float>(x.z,y.z,z.z,0),SIMD4<Float>(-simd_dot(x,eye),-simd_dot(y,eye),-simd_dot(z,eye),1))
}
private func perspective(_ fovy: Float,_ aspect: Float,_ near: Float,_ far: Float)->simd_float4x4 { let y=1/tan(fovy/2),x=y/aspect,z=far/(near-far); return simd_float4x4(SIMD4<Float>(x,0,0,0),SIMD4<Float>(0,y,0,0),SIMD4<Float>(0,0,z,-1),SIMD4<Float>(0,0,z*near,0)) }

@main private enum Main {
    static func main() throws {
        let arguments = CommandLine.arguments
        let fallback = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("ValidationInputs/deetjen-ob-f03-surface-v1/manifest.json")
        let path = arguments.dropFirst().first(where: { !$0.hasPrefix("--") }).map { URL(fileURLWithPath: $0) } ?? fallback
        let dataset = try Dataset.load(path); let engine = try RobotEngine(dataset: dataset)
        if arguments.contains("--probe") {
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
        let app=NSApplication.shared; let delegate=AppDelegate(engine: engine); app.delegate=delegate; app.setActivationPolicy(.regular); app.run()
        withExtendedLifetime(delegate) {}
    }
}
