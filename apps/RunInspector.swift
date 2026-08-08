import AppKit
import Metal
import MetalKit

private final class InspectionDelivery: @unchecked Sendable {
    let frame: MetalRoboTaskInspectionFrame
    let release: @Sendable () -> Void

    init(
        frame: MetalRoboTaskInspectionFrame,
        release: @escaping @Sendable () -> Void
    ) {
        self.frame = frame
        self.release = release
    }
}

enum MetalRoboRunInspectorError: Error {
    case closed
}

@MainActor
private final class RunInspectorWindow: NSObject, MTKViewDelegate,
    NSWindowDelegate
{
    private let window: NSWindow
    private let view: MTKView
    private let queue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private var pending: InspectionDelivery?
    private var closed = false
    var onClose: (() -> Void)?

    private init(
        window: NSWindow,
        view: MTKView,
        queue: MTLCommandQueue,
        pipeline: MTLRenderPipelineState
    ) {
        self.window = window
        self.view = view
        self.queue = queue
        self.pipeline = pipeline
        super.init()
        view.delegate = self
        window.contentView = view
        window.title = "Numi Lab Inspector — waiting for a frame"
        window.delegate = self
        window.makeKeyAndOrderFront(nil)
    }

    static func make() throws -> RunInspectorWindow {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue()
        else {
            throw MetalRoboTaskRolloutError.native(
                "Metal inspector could not create a display device."
            )
        }
        let library = try device.makeLibrary(
            source: Self.shaderSource,
            options: nil
        )
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = library.makeFunction(name: "inspectVertex")
        descriptor.fragmentFunction = library.makeFunction(
            name: "inspectFragment"
        )
        descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
        let pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        let view = MTKView(
            frame: NSRect(x: 0, y: 0, width: 960, height: 600),
            device: device
        )
        view.colorPixelFormat = .bgra8Unorm_srgb
        view.clearColor = MTLClearColor(red: 0.015, green: 0.017,
                                        blue: 0.024, alpha: 1)
        view.enableSetNeedsDisplay = true
        view.isPaused = true
        let window = NSWindow(
            contentRect: NSRect(x: 120, y: 120, width: 960, height: 600),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        return RunInspectorWindow(
            window: window,
            view: view,
            queue: queue,
            pipeline: pipeline
        )
    }

    func offer(_ delivery: InspectionDelivery) {
        guard !closed else {
            delivery.release()
            return
        }
        pending?.release()
        pending = delivery
        window.title = "Numi Lab Inspector — live · env \(delivery.frame.environmentIndex) · frame \(delivery.frame.frameIndex) · dropped \(delivery.frame.droppedFrames)"
        view.setNeedsDisplay(view.bounds)
    }

    func draw(in view: MTKView) {
        guard let delivery = pending,
              let descriptor = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let command = queue.makeCommandBuffer(),
              let encoder = command.makeRenderCommandEncoder(
                  descriptor: descriptor
              )
        else {
            return
        }
        pending = nil
        var dimensions = SIMD2<UInt32>(
            UInt32(delivery.frame.width),
            UInt32(delivery.frame.height)
        )
        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentBuffer(delivery.frame.rgbBuffer, offset: 0,
                                  index: 0)
        encoder.setFragmentBytes(&dimensions,
                                 length: MemoryLayout<SIMD2<UInt32>>.stride,
                                 index: 1)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0,
                               vertexCount: 3)
        encoder.endEncoding()
        command.addCompletedHandler { _ in
            delivery.release()
        }
        command.present(drawable)
        command.commit()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func windowWillClose(_ notification: Notification) {
        closed = true
        pending?.release()
        pending = nil
        onClose?()
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct VertexOut {
        float4 position [[position]];
        float2 uv;
    };

    vertex VertexOut inspectVertex(uint vertexID [[vertex_id]]) {
        constexpr float2 positions[3] = {
            float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)
        };
        VertexOut output;
        output.position = float4(positions[vertexID], 0.0, 1.0);
        output.uv = 0.5 * (positions[vertexID] + 1.0);
        return output;
    }

    fragment float4 inspectFragment(
        VertexOut input [[stage_in]],
        const device float4* source [[buffer(0)]],
        constant uint2& dimensions [[buffer(1)]]
    ) {
        const uint x = min(uint(clamp(input.uv.x, 0.0, 0.999999) * dimensions.x),
                           dimensions.x - 1u);
        const uint y = min(uint(clamp(1.0 - input.uv.y, 0.0, 0.999999) * dimensions.y),
                           dimensions.y - 1u);
        const float3 linear = max(source[y * dimensions.x + x].xyz, 0.0);
        // Bounded highlight rolloff keeps HDR scene lighting inspectable while
        // the sRGB drawable performs the final display conversion.
        return float4(linear / (1.0 + linear), 1.0);
    }
    """
}

// This bridge is intentionally sendable: the frame slot is exclusively owned
// by the delivery until the MTKView command-buffer completion releases it.
public final class MetalRoboRunInspectorBridge: @unchecked Sendable {
    private let window: RunInspectorWindow
    private let stateLock = NSLock()
    private var closed = false

    @MainActor
    private init(window: RunInspectorWindow) {
        self.window = window
        window.onClose = { [weak self] in
            self?.markClosed()
        }
    }

    @MainActor
    public static func launch() throws -> MetalRoboRunInspectorBridge {
        precondition(Thread.isMainThread)
        let application = NSApplication.shared
        application.setActivationPolicy(.regular)
        let bridge = MetalRoboRunInspectorBridge(
            window: try RunInspectorWindow.make()
        )
        application.activate(ignoringOtherApps: true)
        return bridge
    }

    var isClosed: Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return closed
    }

    private func markClosed() {
        stateLock.lock()
        closed = true
        stateLock.unlock()
    }

    public func publish(
        frame: MetalRoboTaskInspectionFrame,
        context: MetalRoboTaskRolloutContext
    ) {
        guard !isClosed else {
            context.releaseInspectionFrame(slotIndex: frame.slotIndex)
            return
        }
        let delivery = InspectionDelivery(
            frame: frame,
            release: {
                context.releaseInspectionFrame(slotIndex: frame.slotIndex)
            }
        )
        DispatchQueue.main.async { [weak self, window] in
            guard self?.isClosed == false else {
                delivery.release()
                return
            }
            window.offer(delivery)
        }
    }
}
