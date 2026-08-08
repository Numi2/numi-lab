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
    NSWindowDelegate, NSToolbarDelegate
{
    private static let pauseToolbarItem = NSToolbarItem.Identifier(
        "numi.inspector.pause"
    )
    private static let latestPolicyToolbarItem = NSToolbarItem.Identifier(
        "numi.inspector.latest-policy"
    )
    private let window: NSWindow
    private let view: MTKView
    private let queue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private var pending: InspectionDelivery?
    private var closed = false
    private var paused = false
    private var lastFrameSummary = "waiting for a frame"
    private weak var pauseItem: NSToolbarItem?
    private weak var latestPolicyItem: NSToolbarItem?
    private let canReloadLatestPolicy: Bool
    var onClose: (() -> Void)?
    var onPauseChanged: ((Bool) -> Void)?
    var onLatestPolicyRequested: (() -> Void)?

    private init(
        window: NSWindow,
        view: MTKView,
        queue: MTLCommandQueue,
        pipeline: MTLRenderPipelineState,
        canReloadLatestPolicy: Bool
    ) {
        self.window = window
        self.view = view
        self.queue = queue
        self.pipeline = pipeline
        self.canReloadLatestPolicy = canReloadLatestPolicy
        super.init()
        view.delegate = self
        window.contentView = view
        window.delegate = self
        let toolbar = NSToolbar(identifier: "numi.inspector.toolbar")
        toolbar.delegate = self
        toolbar.displayMode = .iconAndLabel
        toolbar.allowsUserCustomization = false
        toolbar.autosavesConfiguration = false
        window.toolbar = toolbar
        updateChrome()
        window.makeKeyAndOrderFront(nil)
    }

    static func make(
        canReloadLatestPolicy: Bool
    ) throws -> RunInspectorWindow {
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
            pipeline: pipeline,
            canReloadLatestPolicy: canReloadLatestPolicy
        )
    }

    func offer(_ delivery: InspectionDelivery) {
        guard !closed, !paused else {
            delivery.release()
            return
        }
        pending?.release()
        pending = delivery
        lastFrameSummary = "env \(delivery.frame.environmentIndex) · frame \(delivery.frame.frameIndex) · dropped \(delivery.frame.droppedFrames)"
        updateChrome()
        view.setNeedsDisplay(view.bounds)
    }

    func showPolicyStatus(_ summary: String) {
        guard !closed else {
            return
        }
        lastFrameSummary = summary
        updateChrome()
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
        var dimensions = SIMD4<UInt32>(
            UInt32(delivery.frame.width),
            UInt32(delivery.frame.height),
            UInt32(max(1, view.drawableSize.width.rounded(.down))),
            UInt32(max(1, view.drawableSize.height.rounded(.down)))
        )
        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentBuffer(delivery.frame.rgbBuffer, offset: 0,
                                  index: 0)
        encoder.setFragmentBytes(&dimensions,
                                 length: MemoryLayout<SIMD4<UInt32>>.stride,
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

    func toolbarAllowedItemIdentifiers(
        _ toolbar: NSToolbar
    ) -> [NSToolbarItem.Identifier] {
        [Self.pauseToolbarItem, Self.latestPolicyToolbarItem]
    }

    func toolbarDefaultItemIdentifiers(
        _ toolbar: NSToolbar
    ) -> [NSToolbarItem.Identifier] {
        [Self.pauseToolbarItem, Self.latestPolicyToolbarItem]
    }

    func toolbar(
        _ toolbar: NSToolbar,
        itemForItemIdentifier itemIdentifier: NSToolbarItem.Identifier,
        willBeInsertedIntoToolbar flag: Bool
    ) -> NSToolbarItem? {
        let item = NSToolbarItem(itemIdentifier: itemIdentifier)
        item.target = self
        if itemIdentifier == Self.pauseToolbarItem {
            item.action = #selector(togglePause)
            configurePauseItem(item)
            pauseItem = item
            return item
        }
        if itemIdentifier == Self.latestPolicyToolbarItem {
            item.action = #selector(requestLatestPolicy)
            configureLatestPolicyItem(item)
            latestPolicyItem = item
            return item
        }
        return nil
    }

    @objc private func togglePause() {
        paused.toggle()
        if paused {
            pending?.release()
            pending = nil
        }
        onPauseChanged?(paused)
        updateChrome()
    }

    @objc private func requestLatestPolicy() {
        guard canReloadLatestPolicy else {
            return
        }
        lastFrameSummary = "loading latest policy"
        updateChrome()
        onLatestPolicyRequested?()
    }

    private func configurePauseItem(_ item: NSToolbarItem) {
        item.label = paused ? "Resume" : "Pause"
        item.toolTip = paused
            ? "Resume live preview"
            : "Pause preview rendering; the run keeps going"
        item.image = NSImage(
            systemSymbolName: paused ? "play.fill" : "pause.fill",
            accessibilityDescription: item.label
        )
    }

    private func configureLatestPolicyItem(_ item: NSToolbarItem) {
        item.label = "Latest Policy"
        item.toolTip = canReloadLatestPolicy
            ? "Load the newest saved revision at the next rollout boundary"
            : "Start this run with --policy-pack to reload its newest revision"
        item.image = NSImage(
            systemSymbolName: "arrow.clockwise",
            accessibilityDescription: item.label
        )
        item.isEnabled = canReloadLatestPolicy
    }

    private func updateChrome() {
        window.title = "Numi Lab Inspector — \(paused ? "paused" : "live") · \(lastFrameSummary)"
        if let item = pauseItem {
            configurePauseItem(item)
        }
        if let item = latestPolicyItem {
            configureLatestPolicyItem(item)
        }
    }

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
        constant uint4& dimensions [[buffer(1)]]
    ) {
        const float sourceAspect = float(dimensions.x) / float(dimensions.y);
        const float destinationAspect = float(dimensions.z) / float(dimensions.w);
        float2 uv = input.uv;
        if (sourceAspect > destinationAspect) {
            const float height = destinationAspect / sourceAspect;
            uv.y = (uv.y - 0.5 * (1.0 - height)) / height;
        } else {
            const float width = sourceAspect / destinationAspect;
            uv.x = (uv.x - 0.5 * (1.0 - width)) / width;
        }
        const float vignette = 0.55 + 0.45 * input.uv.y;
        const float3 backdrop = float3(0.018, 0.030, 0.055) * vignette;
        if (any(uv < 0.0) || any(uv >= 1.0)) {
            return float4(backdrop, 1.0);
        }
        const uint x = min(uint(uv.x * dimensions.x),
                           dimensions.x - 1u);
        const uint y = min(uint((1.0 - uv.y) * dimensions.y),
                           dimensions.y - 1u);
        const float4 sample = source[y * dimensions.x + x];
        const float3 linear = max(sample.xyz, 0.0);
        // Bounded highlight rolloff keeps HDR scene lighting inspectable while
        // the sRGB drawable performs the final display conversion.
        const float3 shaded = linear / (1.0 + linear);
        // The sensor renderer marks background pixels transparent. Keep that
        // useful distinction in the presentation window: a missing/empty
        // camera image is a calm inspector backdrop, never an opaque black
        // result that looks like a rendering failure. Shaded geometry keeps
        // the renderer's own color unchanged.
        return float4(mix(backdrop, shaded, clamp(sample.w, 0.0, 1.0)), 1.0);
    }
    """
}

// This bridge is intentionally sendable: the frame slot is exclusively owned
// by the delivery until the MTKView command-buffer completion releases it.
public final class MetalRoboRunInspectorBridge: @unchecked Sendable {
    private let window: RunInspectorWindow
    private let stateLock = NSLock()
    private var closed = false
    private var acceptingFrames = true
    private var pendingNativeInspectionEnabled: Bool?
    private var latestPolicyReloadRequested = false

    @MainActor
    private init(window: RunInspectorWindow) {
        self.window = window
        window.onClose = { [weak self] in
            self?.markClosed()
        }
        window.onPauseChanged = { [weak self] paused in
            self?.setPresentationPaused(paused)
        }
        window.onLatestPolicyRequested = { [weak self] in
            self?.requestLatestPolicyReload()
        }
    }

    @MainActor
    public static func launch(
        canReloadLatestPolicy: Bool = false
    ) throws -> MetalRoboRunInspectorBridge {
        precondition(Thread.isMainThread)
        let application = NSApplication.shared
        application.setActivationPolicy(.regular)
        let bridge = MetalRoboRunInspectorBridge(
            window: try RunInspectorWindow.make(
                canReloadLatestPolicy: canReloadLatestPolicy
            )
        )
        application.activate(ignoringOtherApps: true)
        return bridge
    }

    var isClosed: Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return closed
    }

    var acceptsFrames: Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return !closed && acceptingFrames
    }

    private func markClosed() {
        stateLock.lock()
        closed = true
        acceptingFrames = false
        pendingNativeInspectionEnabled = false
        stateLock.unlock()
    }

    private func setPresentationPaused(_ paused: Bool) {
        stateLock.lock()
        let enabled = !paused && !closed
        if acceptingFrames != enabled {
            acceptingFrames = enabled
            pendingNativeInspectionEnabled = enabled
        }
        stateLock.unlock()
    }

    private func requestLatestPolicyReload() {
        stateLock.lock()
        if !closed {
            latestPolicyReloadRequested = true
        }
        stateLock.unlock()
    }

    // Consumed by the rollout scheduler at its normal publication boundary,
    // never from the AppKit action itself.
    func takePendingNativeInspectionEnabled() -> Bool? {
        stateLock.lock()
        defer { stateLock.unlock() }
        let value = pendingNativeInspectionEnabled
        pendingNativeInspectionEnabled = nil
        return value
    }

    func takeLatestPolicyReloadRequest() -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        let requested = latestPolicyReloadRequested
        latestPolicyReloadRequested = false
        return requested
    }

    func reportPolicyStatus(_ summary: String) {
        DispatchQueue.main.async { [window] in
            window.showPolicyStatus(summary)
        }
    }

    public func publish(
        frame: MetalRoboTaskInspectionFrame,
        context: MetalRoboTaskRolloutContext
    ) {
        guard acceptsFrames else {
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
            guard self?.acceptsFrames == true else {
                delivery.release()
                return
            }
            window.offer(delivery)
        }
    }
}
