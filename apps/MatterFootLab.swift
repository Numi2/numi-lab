import AppKit
import Metal
import MetalKit
import simd

private enum FootLabError: Error, CustomStringConvertible {
    case invalid(String)

    var description: String {
        switch self {
        case .invalid(let message): return "Matter foot lab: \(message)"
        }
    }
}

private struct FootSnapshot: Decodable {
    let schema: String
    let scene: String
    let frame: Int
    let frameCount: Int
    let contacts: Int
    let normalImpulse: Float
    let centerOfPressure: [Float]
    let padCompression: Float
    let porePressureMax: Float
    let rigidReactionZ: Float
    let acceptedFootVelocityZ: Float
    let maximumOffDiagonalResponse: Float
    let fixedBaseError: Float
    let kktResidual: Float
    let volumeResidual: Float
    let naturalResidual: Float
    let coneViolation: Float
    let complementarity: Float
    let transportResidual: Float
    let gpuMilliseconds: Float
    let footPosition: [Float]
    let padNodes: [[Float]]
    let contactPoints: [[Float]]

    enum CodingKeys: String, CodingKey {
        case schema, scene, frame, contacts
        case frameCount = "frame_count"
        case normalImpulse = "normal_impulse"
        case centerOfPressure = "center_of_pressure"
        case padCompression = "pad_compression"
        case porePressureMax = "pore_pressure_max"
        case rigidReactionZ = "rigid_reaction_z"
        case acceptedFootVelocityZ = "accepted_foot_velocity_z"
        case maximumOffDiagonalResponse = "maximum_off_diagonal_response"
        case fixedBaseError = "fixed_base_error"
        case kktResidual = "kkt_residual"
        case volumeResidual = "volume_residual"
        case naturalResidual = "natural_residual"
        case coneViolation = "cone_violation"
        case complementarity
        case transportResidual = "transport_residual"
        case gpuMilliseconds = "gpu_milliseconds"
        case footPosition = "foot_position"
        case padNodes = "pad_nodes"
        case contactPoints = "contact_points"
    }

    static func runCoupledProbe() throws -> [FootSnapshot] {
        guard let executable = Bundle.main.executableURL else {
            throw FootLabError.invalid("could not resolve the Numi executable")
        }
        let probe = executable.deletingLastPathComponent()
            .appendingPathComponent("metalrobo_matter_physics_probe")
        guard FileManager.default.isExecutableFile(atPath: probe.path) else {
            throw FootLabError.invalid("coupled Matter physics probe is unavailable beside the viewer")
        }
        let process = Process()
        let output = Pipe()
        process.executableURL = probe
        process.arguments = ["--articulated-foot-pad-sequence"]
        process.standardOutput = output
        process.standardError = output
        try process.run()
        let data = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        let text = String(decoding: data, as: UTF8.self)
        guard process.terminationStatus == 0 else {
            throw FootLabError.invalid("coupled run failed: \(text.trimmingCharacters(in: .whitespacesAndNewlines))")
        }
        var frames: [FootSnapshot] = []
        for line in text.split(separator: "\n") where line.first == "{" {
            guard let json = String(line).data(using: .utf8),
                  let snapshot = try? JSONDecoder().decode(FootSnapshot.self, from: json),
                  snapshot.scene == "articulated_foot_poroelastic_pad" else { continue }
            frames.append(snapshot)
        }
        guard !frames.isEmpty, frames.count == frames[0].frameCount else {
            throw FootLabError.invalid("the coupled probe did not publish a complete accepted sequence")
        }
        for (index, snapshot) in frames.enumerated() {
            guard snapshot.schema == "numi.matter.scene.v1",
                  snapshot.frame == index,
                  snapshot.frameCount == frames.count,
                  snapshot.padNodes.count == 8,
                  snapshot.footPosition.count == 3,
                  snapshot.centerOfPressure.count == 2,
                  snapshot.contacts > 0,
                  snapshot.kktResidual <= 1e-4,
                  snapshot.volumeResidual <= 1e-4,
                  snapshot.naturalResidual <= 1e-4,
                  snapshot.coneViolation <= 1e-5,
                  snapshot.fixedBaseError <= 1e-8 else {
                throw FootLabError.invalid("accepted frame \(index + 1) failed its scene or KKT contract")
            }
        }
        return frames
    }
}

private struct Vertex {
    var position: SIMD3<Float>
    var color: SIMD4<Float>
}

private struct Uniforms {
    var viewProjection: simd_float4x4
    var eye: SIMD4<Float>
}

private final class FootRenderer {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let frames: [FootSnapshot]
    var snapshot: FootSnapshot { frames[frameIndex] }
    private let pipeline: MTLRenderPipelineState
    private let depthState: MTLDepthStencilState
    private var vertexBuffer: MTLBuffer
    private var indexBuffer: MTLBuffer
    private var lineIndexBuffer: MTLBuffer
    private var indexCount = 0
    private var lineIndexCount = 0
    private(set) var deformationGain: Float = 1000
    private(set) var frameIndex = 0
    private(set) var isPlaying = true
    private var lastFrameTime: CFTimeInterval = 0

    init(frames: [FootSnapshot]) throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw FootLabError.invalid("Metal device or command queue is unavailable")
        }
        self.device = device
        self.queue = queue
        self.frames = frames
        let library = try device.makeLibrary(source: Self.shaderSource, options: nil)
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "Numi Matter accepted-state inspection"
        descriptor.vertexFunction = library.makeFunction(name: "sceneVertex")
        descriptor.fragmentFunction = library.makeFunction(name: "sceneFragment")
        descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
        descriptor.colorAttachments[0].isBlendingEnabled = true
        descriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        descriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        descriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        descriptor.depthAttachmentPixelFormat = .depth32Float
        pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        let depth = MTLDepthStencilDescriptor()
        depth.depthCompareFunction = .less
        depth.isDepthWriteEnabled = true
        guard let depthState = device.makeDepthStencilState(descriptor: depth),
              let emptyVertex = device.makeBuffer(length: 32),
              let emptyIndex = device.makeBuffer(length: 2),
              let emptyLineIndex = device.makeBuffer(length: 2) else {
            throw FootLabError.invalid("could not allocate Metal inspection resources")
        }
        self.depthState = depthState
        vertexBuffer = emptyVertex
        indexBuffer = emptyIndex
        lineIndexBuffer = emptyLineIndex
        rebuildMesh()
    }

    func toggleDeformationGain() {
        deformationGain = deformationGain > 1 ? 1 : 1000
        rebuildMesh()
    }

    func togglePlayback() { isPlaying.toggle() }

    func step(_ delta: Int) {
        isPlaying = false
        frameIndex = (frameIndex + delta + frames.count) % frames.count
        rebuildMesh()
    }

    func advanceAnimation(at time: CFTimeInterval) -> Bool {
        guard isPlaying else { return false }
        if lastFrameTime == 0 { lastFrameTime = time; return false }
        guard time - lastFrameTime >= 0.30 else { return false }
        lastFrameTime = time
        frameIndex = (frameIndex + 1) % frames.count
        rebuildMesh()
        return true
    }

    private func rebuildMesh() {
        var vertices: [Vertex] = []
        var indices: [UInt16] = []
        var lineIndices: [UInt16] = []

        func appendSolidBox(minimum: SIMD3<Float>, maximum: SIMD3<Float>,
                            color: SIMD4<Float>) {
            let base = UInt16(vertices.count)
            let points = [
                SIMD3<Float>(minimum.x, minimum.y, minimum.z),
                SIMD3<Float>(maximum.x, minimum.y, minimum.z),
                SIMD3<Float>(minimum.x, maximum.y, minimum.z),
                SIMD3<Float>(maximum.x, maximum.y, minimum.z),
                SIMD3<Float>(minimum.x, minimum.y, maximum.z),
                SIMD3<Float>(maximum.x, minimum.y, maximum.z),
                SIMD3<Float>(minimum.x, maximum.y, maximum.z),
                SIMD3<Float>(maximum.x, maximum.y, maximum.z),
            ]
            vertices.append(contentsOf: points.map { Vertex(position: $0, color: color) })
            let box: [UInt16] = [
                0,2,1, 1,2,3, 4,5,6, 5,7,6,
                0,1,4, 1,5,4, 2,6,3, 3,6,7,
                0,4,2, 2,4,6, 1,3,5, 3,7,5,
            ]
            indices.append(contentsOf: box.map { base + $0 })
        }

        func appendQuad(_ points: [SIMD3<Float>], color: SIMD4<Float>) {
            let base = UInt16(vertices.count)
            vertices.append(contentsOf: points.map { Vertex(position: $0, color: color) })
            indices.append(contentsOf: [base, base+1, base+2, base+1, base+3, base+2])
        }

        func appendWireBox(minimum: SIMD3<Float>, maximum: SIMD3<Float>,
                           color: SIMD4<Float>) {
            let base = UInt16(vertices.count)
            let points = [
                SIMD3<Float>(minimum.x, minimum.y, minimum.z),
                SIMD3<Float>(maximum.x, minimum.y, minimum.z),
                SIMD3<Float>(minimum.x, maximum.y, minimum.z),
                SIMD3<Float>(maximum.x, maximum.y, minimum.z),
                SIMD3<Float>(minimum.x, minimum.y, maximum.z),
                SIMD3<Float>(maximum.x, minimum.y, maximum.z),
                SIMD3<Float>(minimum.x, maximum.y, maximum.z),
                SIMD3<Float>(maximum.x, maximum.y, maximum.z),
            ]
            vertices.append(contentsOf: points.map { Vertex(position: $0, color: color) })
            let edges: [UInt16] = [
                0,1, 1,3, 3,2, 2,0,
                4,5, 5,7, 7,6, 6,4,
                0,4, 1,5, 2,6, 3,7,
            ]
            lineIndices.append(contentsOf: edges.map { base + $0 })
        }

        func appendOctahedron(center: SIMD3<Float>, radius: Float, color: SIMD4<Float>) {
            let base = UInt16(vertices.count)
            let points = [
                center + SIMD3<Float>( radius, 0, 0), center + SIMD3<Float>(-radius, 0, 0),
                center + SIMD3<Float>(0,  radius, 0), center + SIMD3<Float>(0, -radius, 0),
                center + SIMD3<Float>(0, 0,  radius), center + SIMD3<Float>(0, 0, -radius),
            ]
            vertices.append(contentsOf: points.map { Vertex(position: $0, color: color) })
            let octa: [UInt16] = [0,2,4, 2,1,4, 1,3,4, 3,0,4, 2,0,5, 1,2,5, 3,1,5, 0,3,5]
            indices.append(contentsOf: octa.map { base + $0 })
        }

        // The diagnostic reference is the first certified GPU state, not the
        // pre-solve authored mesh. This makes accepted step 1 coincide with
        // its pale reference and shows only subsequent accepted compression.
        let initial = (0..<8).map { node in
            SIMD3<Float>(frames[0].padNodes[node][0], frames[0].padNodes[node][1],
                         frames[0].padNodes[node][2])
        }
        let accepted = (0..<8).map { node in
            SIMD3<Float>(snapshot.padNodes[node][0], snapshot.padNodes[node][1],
                         snapshot.padNodes[node][2])
        }
        let displayedPad = (0..<8).map {
            initial[$0] + deformationGain * (accepted[$0] - initial[$0])
        }
        let diagnostic = deformationGain > 1
        let physicalSoleZ = snapshot.footPosition[2] - 0.0099
        let visualSoleZ = diagnostic
            ? (4..<8).map { displayedPad[$0].z }.max() ?? physicalSoleZ
            : physicalSoleZ

        var nodeLoads = [Float](repeating: 0, count: 8)
        for contact in snapshot.contactPoints where contact.count == 4 {
            let nearest = (4..<8).min { first, second in
                let a = SIMD2<Float>(accepted[first].x-contact[0], accepted[first].y-contact[1])
                let b = SIMD2<Float>(accepted[second].x-contact[0], accepted[second].y-contact[1])
                return simd_length_squared(a) < simd_length_squared(b)
            } ?? 4
            nodeLoads[nearest] += max(contact[3], 0)
        }

        let padBase = UInt16(vertices.count)
        for node in 0..<8 {
            let height = Float(node >= 4 ? 1 : 0)
            let pressure = min(1, snapshot.porePressureMax / 0.25)
            let load = min(1, nodeLoads[node] / max(snapshot.normalImpulse, 1e-9) * 4)
            let baseColor = SIMD3<Float>(0.05 + 0.10 * height,
                                         0.35 + 0.38 * pressure,
                                         0.62 + 0.28 * height)
            let color3 = simd_mix(baseColor, SIMD3<Float>(0.96, 0.20, 0.08),
                                  SIMD3<Float>(repeating: 0.55 * load))
            let color = SIMD4<Float>(color3, 1)
            vertices.append(Vertex(position: displayedPad[node], color: color))
        }
        let pad: [UInt16] = [
            0,2,1, 1,2,3, 4,5,6, 5,7,6,
            0,1,4, 1,5,4, 2,6,3, 3,6,7,
            0,4,2, 2,4,6, 1,3,5, 3,7,5,
        ]
        indices.append(contentsOf: pad.map { padBase + $0 })

        let referenceEdges: [UInt16] = [
            0,1, 1,3, 3,2, 2,0,
            4,5, 5,7, 7,6, 6,4,
            0,4, 1,5, 2,6, 3,7,
        ]
        let acceptedOutlineBase = UInt16(vertices.count)
        vertices.append(contentsOf: displayedPad.map {
            Vertex(position: $0, color: SIMD4<Float>(0.25, 0.84, 1.0, 1))
        })
        lineIndices.append(contentsOf: referenceEdges.map { acceptedOutlineBase + $0 })

        appendQuad([
            SIMD3<Float>(-0.036, -0.032, -0.00005),
            SIMD3<Float>( 0.036, -0.032, -0.00005),
            SIMD3<Float>(-0.036,  0.032, -0.00005),
            SIMD3<Float>( 0.036,  0.032, -0.00005),
        ], color: SIMD4<Float>(0.11, 0.14, 0.18, 1))

        if diagnostic {
            // The sole is an exact-position wireframe reference only. Keeping
            // it out of the filled geometry prevents amplified displacement
            // from being misread as physical penetration.
            let footX = snapshot.footPosition[0]
            let footY = snapshot.footPosition[1]
            appendSolidBox(
                minimum: SIMD3<Float>(footX - 0.011, footY - 0.009, visualSoleZ),
                maximum: SIMD3<Float>(footX + 0.011, footY + 0.009,
                                      visualSoleZ + 0.004),
                color: SIMD4<Float>(1.0, 0.68, 0.08, 0.22)
            )
            appendWireBox(
                minimum: SIMD3<Float>(footX - 0.011, footY - 0.009, visualSoleZ),
                maximum: SIMD3<Float>(footX + 0.011, footY + 0.009,
                                      visualSoleZ + 0.004),
                color: SIMD4<Float>(1.0, 0.78, 0.16, 1)
            )
            let unloadedBase = UInt16(vertices.count)
            vertices.append(contentsOf: initial.map {
                Vertex(position: $0, color: SIMD4<Float>(0.82, 0.90, 0.98, 1))
            })
            lineIndices.append(contentsOf: referenceEdges.map { unloadedBase + $0 })
            for node in 0..<8 {
                let vectorBase = UInt16(vertices.count)
                let vectorColor = node >= 4
                    ? SIMD4<Float>(1.0, 0.47, 0.10, 1)
                    : SIMD4<Float>(0.45, 0.70, 0.92, 1)
                vertices.append(Vertex(position: initial[node], color: vectorColor))
                vertices.append(Vertex(position: displayedPad[node], color: vectorColor))
                lineIndices.append(contentsOf: [vectorBase, vectorBase + 1])
            }
            for node in 4..<8 where nodeLoads[node] > 0 {
                appendOctahedron(center: displayedPad[node], radius: 0.00022,
                                 color: SIMD4<Float>(1, 0.28, 0.06, 1))
            }
        } else {
            let footX = snapshot.footPosition[0]
            let footY = snapshot.footPosition[1]
            appendSolidBox(
                minimum: SIMD3<Float>(footX - 0.011, footY - 0.009, visualSoleZ),
                maximum: SIMD3<Float>(footX + 0.011, footY + 0.009,
                                      visualSoleZ + 0.004),
                color: SIMD4<Float>(1.0, 0.68, 0.08, 0.22)
            )
            appendWireBox(
                minimum: SIMD3<Float>(footX - 0.011, footY - 0.009, visualSoleZ),
                maximum: SIMD3<Float>(footX + 0.011, footY + 0.009,
                                      visualSoleZ + 0.004),
                color: SIMD4<Float>(1.0, 0.78, 0.16, 1)
            )
            for contact in snapshot.contactPoints where contact.count == 4 {
                let loadScale = min(1, max(0.2, contact[3] /
                    max(snapshot.normalImpulse, 1e-9) * 4))
                appendOctahedron(center: SIMD3<Float>(contact[0], contact[1], contact[2]),
                                 radius: 0.00018,
                                 color: SIMD4<Float>(1, 0.18 + 0.50 * loadScale, 0.04, 1))
            }
            appendOctahedron(center: SIMD3<Float>(snapshot.centerOfPressure[0],
                                                  snapshot.centerOfPressure[1],
                                                  physicalSoleZ + 0.00005),
                             radius: 0.00028, color: SIMD4<Float>(1, 0.95, 0.16, 1))
        }

        vertexBuffer = device.makeBuffer(bytes: vertices,
                                         length: vertices.count * MemoryLayout<Vertex>.stride,
                                         options: .storageModeShared)!
        indexBuffer = device.makeBuffer(bytes: indices,
                                        length: indices.count * MemoryLayout<UInt16>.stride,
                                        options: .storageModeShared)!
        lineIndexBuffer = device.makeBuffer(bytes: lineIndices,
                                            length: lineIndices.count * MemoryLayout<UInt16>.stride,
                                            options: .storageModeShared)!
        indexCount = indices.count
        lineIndexCount = lineIndices.count
    }

    func draw(pass: MTLRenderPassDescriptor, drawable: CAMetalDrawable,
              size: CGSize, yaw: Float, pitch: Float, distance: Float) {
        guard let command = queue.makeCommandBuffer(),
              let encoder = command.makeRenderCommandEncoder(descriptor: pass) else { return }
        let target = SIMD3<Float>(0, 0, 0.010)
        let eye = target + distance * SIMD3<Float>(cos(pitch) * cos(yaw),
                                                   cos(pitch) * sin(yaw), sin(pitch))
        let aspect = Float(max(1, size.width) / max(1, size.height))
        var uniforms = Uniforms(
            viewProjection: perspective(fovy: 0.68, aspect: aspect, near: 0.001, far: 1)
                * lookAt(eye: eye, center: target, up: SIMD3<Float>(0,0,1)),
            eye: SIMD4<Float>(eye, 1)
        )
        encoder.label = "Numi Matter foot-pad accepted snapshot"
        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(depthState)
        encoder.setCullMode(.none)
        encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        encoder.drawIndexedPrimitives(type: .triangle, indexCount: indexCount,
                                      indexType: .uint16, indexBuffer: indexBuffer,
                                      indexBufferOffset: 0)
        encoder.setDepthBias(-1, slopeScale: 0, clamp: 0)
        encoder.drawIndexedPrimitives(type: .line, indexCount: lineIndexCount,
                                      indexType: .uint16, indexBuffer: lineIndexBuffer,
                                      indexBufferOffset: 0)
        encoder.endEncoding()
        command.present(drawable)
        command.commit()
    }

    private static let shaderSource = #"""
    #include <metal_stdlib>
    using namespace metal;
    struct Vertex { packed_float3 position; float4 color; };
    struct Uniforms { float4x4 viewProjection; float4 eye; };
    struct Raster { float4 position [[position]]; float3 world; float4 color; };
    vertex Raster sceneVertex(device const Vertex* vertices [[buffer(0)]],
                              constant Uniforms& u [[buffer(1)]], uint id [[vertex_id]]) {
        Raster out; out.world=float3(vertices[id].position); out.position=u.viewProjection*float4(out.world,1); out.color=vertices[id].color; return out;
    }
    fragment float4 sceneFragment(Raster in [[stage_in]], constant Uniforms& u [[buffer(1)]]) {
        float depth=clamp(1.0-length(u.eye.xyz-in.world)*3.0,0.65,1.0);
        float3 color=in.color.rgb*depth + 0.10*pow(clamp(in.world.z/0.025,0.0,1.0),2.0);
        return float4(1.0-exp(-1.35*color), in.color.a);
    }
    """#
}

private final class FootMetalView: MTKView {
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
private final class FootViewController: NSObject, MTKViewDelegate {
    let renderer: FootRenderer
    let view: FootMetalView
    var updateStatus: (() -> Void)?
    private var yaw: Float = -0.82
    private var pitch: Float = 0.48
    private var distance: Float = 0.070

    init(renderer: FootRenderer) {
        self.renderer = renderer
        view = FootMetalView(frame: .zero, device: renderer.device)
        super.init()
        view.delegate = self
        view.colorPixelFormat = .bgra8Unorm_srgb
        view.depthStencilPixelFormat = .depth32Float
        view.clearColor = MTLClearColorMake(0.008, 0.013, 0.024, 1)
        view.preferredFramesPerSecond = 30
        view.drag = { [weak self] dx, dy in
            self?.yaw -= dx * 0.006
            self?.pitch = min(1.25, max(-0.2, (self?.pitch ?? 0.48) + dy * 0.006))
        }
        view.zoom = { [weak self] delta in
            self?.distance = min(0.16, max(0.035, (self?.distance ?? 0.07) * exp(delta * 0.018)))
        }
        view.key = { [weak self] character in
            guard let self else { return }
            if character == " " || character == "p" || character == "P" {
                renderer.togglePlayback()
            }
            if character == "d" || character == "D" { renderer.toggleDeformationGain() }
            if character == "]" { renderer.step(1) }
            if character == "[" { renderer.step(-1) }
            updateStatus?()
        }
    }

    func draw(in view: MTKView) {
        guard let pass = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable else { return }
        if renderer.advanceAnimation(at: CACurrentMediaTime()) { updateStatus?() }
        renderer.draw(pass: pass, drawable: drawable, size: view.drawableSize,
                      yaw: yaw, pitch: pitch, distance: distance)
    }
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
}

@MainActor
private final class FootAppDelegate: NSObject, NSApplicationDelegate {
    let renderer: FootRenderer
    private var window: NSWindow?
    private var controller: FootViewController?
    private var status: NSTextField?

    init(renderer: FootRenderer) { self.renderer = renderer }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let controller = FootViewController(renderer: renderer)
        let status = NSTextField(wrappingLabelWithString: "")
        status.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        status.textColor = NSColor(calibratedWhite: 0.88, alpha: 1)
        status.translatesAutoresizingMaskIntoConstraints = false
        let sidebar = NSView()
        sidebar.wantsLayer = true
        sidebar.layer?.backgroundColor = NSColor(calibratedRed: 0.035, green: 0.05, blue: 0.075, alpha: 1).cgColor
        sidebar.addSubview(status)
        controller.view.translatesAutoresizingMaskIntoConstraints = false
        sidebar.translatesAutoresizingMaskIntoConstraints = false
        let root = NSView()
        root.addSubview(controller.view)
        root.addSubview(sidebar)
        NSLayoutConstraint.activate([
            controller.view.leadingAnchor.constraint(equalTo: root.leadingAnchor),
            controller.view.topAnchor.constraint(equalTo: root.topAnchor),
            controller.view.bottomAnchor.constraint(equalTo: root.bottomAnchor),
            controller.view.trailingAnchor.constraint(equalTo: sidebar.leadingAnchor),
            sidebar.trailingAnchor.constraint(equalTo: root.trailingAnchor),
            sidebar.topAnchor.constraint(equalTo: root.topAnchor),
            sidebar.bottomAnchor.constraint(equalTo: root.bottomAnchor),
            sidebar.widthAnchor.constraint(equalToConstant: 330),
            status.leadingAnchor.constraint(equalTo: sidebar.leadingAnchor, constant: 22),
            status.trailingAnchor.constraint(equalTo: sidebar.trailingAnchor, constant: -22),
            status.topAnchor.constraint(equalTo: sidebar.topAnchor, constant: 24),
        ])
        let window = NSWindow(contentRect: NSRect(x: 90, y: 80, width: 1180, height: 760),
                              styleMask: [.titled, .closable, .miniaturizable, .resizable],
                              backing: .buffered, defer: false)
        window.title = "Numi Lab · Matter · articulated foot on poroelastic pad"
        window.contentView = root
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(controller.view)
        controller.updateStatus = { [weak self] in self?.updateStatus() }
        self.controller = controller
        self.status = status
        self.window = window
        updateStatus()
        NSApp.activate(ignoringOtherApps: true)
    }

    private func updateStatus() {
        let s = renderer.snapshot
        let diagnostic = renderer.deformationGain > 1
        let displayedCompression = diagnostic
            ? max(0, s.padCompression - renderer.frames[0].padCompression) *
                renderer.deformationGain
            : s.padCompression
        let mode = diagnostic
            ? "DEFORMATION DIAGNOSTIC · 1000×"
            : "PHYSICAL ACCEPTED STATE · 1×"
        let geometryNote = diagnostic
            ? "  amplified FEM displacement\n  yellow sole is contact-aligned display"
            : "  exact accepted geometry\n  yellow sole uses accepted position"
        let legend = diagnostic
            ? "Gray plane            fixed ground support\nCyan/red              amplified FEM surface\nPale outline          first accepted pad\nYellow box/frame      contact-aligned sole\nOrange vectors        accepted displacement\nRed points            loaded FEM nodes"
            : "Gray plane            fixed ground support\nCyan/red              accepted FEM surface\nYellow box/frame      accepted articulated sole\nRed points            resolved contacts\nYellow point           center of pressure"
        status?.stringValue = """
        \(mode)

        \(geometryNote)

        Accepted step         \(renderer.frameIndex + 1) / \(renderer.frames.count)
        Playback              \(renderer.isPlaying ? "running" : "paused")

        Device
          \(renderer.device.name)

        Authority
          articulated finite box sole
          + mixed poroelastic FEM
          + sparse coupled contact

        Contacts              \(s.contacts)
        Normal impulse        \(format(s.normalImpulse)) N·s
        Rigid reaction z      \(format(s.rigidReactionZ)) N·s
        Center of pressure
          x \(format(s.centerOfPressure[0])) m
          y \(format(s.centerOfPressure[1])) m

        Pad compression       \(format(s.padCompression)) m
        Displayed compression \(format(displayedCompression)) m
        Fixed-base error      \(format(s.fixedBaseError)) m
        Max pore pressure     \(format(s.porePressureMax))
        Off-diagonal W        \(format(s.maximumOffDiagonalResponse))

        KKT residual          \(format(s.kktResidual))
        Volume residual       \(format(s.volumeResidual))
        Natural residual      \(format(s.naturalResidual))
        Cone violation        \(format(s.coneViolation))
        Complementarity       \(format(s.complementarity))
        Transport residual    \(format(s.transportResidual))

        GPU transaction       \(String(format: "%.2f", s.gpuMilliseconds)) ms

        \(legend)

        Drag to orbit · scroll to zoom
        Space: \(renderer.isPlaying ? "pause replay" : "resume replay")
        D: \(diagnostic ? "physical state 1×" : "deformation diagnostic 1000×")
        [ ]: accepted-state step
        """
    }

    private func format(_ value: Float) -> String { String(format: "%.6g", value) }
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
private struct MatterFootLabMain {
    @MainActor static func main() {
        do {
            let frames = try FootSnapshot.runCoupledProbe()
            let renderer = try FootRenderer(frames: frames)
            let app = NSApplication.shared
            app.setActivationPolicy(.regular)
            let delegate = FootAppDelegate(renderer: renderer)
            app.delegate = delegate
            app.run()
            withExtendedLifetime(delegate) {}
        } catch {
            let alert = NSAlert()
            alert.messageText = "Numi Matter foot lab could not open"
            alert.informativeText = String(describing: error)
            alert.runModal()
            Foundation.exit(1)
        }
    }
}
