import AppKit
import Metal
import MetalKit

public struct MetalRoboInspectorPolicyChoice: Sendable {
    public let robotID: String
    public let displayName: String
    public let policyURL: URL

    public init(robotID: String, displayName: String, policyURL: URL) {
        self.robotID = robotID
        self.displayName = displayName
        self.policyURL = policyURL
    }
}

public func metalRoboInspectorPolicyChoices(
    in directory: URL
) -> [MetalRoboInspectorPolicyChoice] {
    guard let entries = FileManager.default.enumerator(
        at: directory,
        includingPropertiesForKeys: [.isRegularFileKey],
        options: [.skipsHiddenFiles, .skipsPackageDescendants]
    ) else {
        return []
    }
    var choices: [MetalRoboInspectorPolicyChoice] = []
    for case let url as URL in entries {
        let regularFile =
            (try? url.resourceValues(forKeys: [.isRegularFileKey])
                .isRegularFile) ?? false
        guard url.pathExtension == "policypack", regularFile
        else {
            continue
        }
        let relative = url.path.replacingOccurrences(
            of: directory.path + "/",
            with: ""
        )
        let components = relative.split(separator: "/")
        let robotID = components.count > 1
            ? String(components[0])
            : "Uncategorized"
        choices.append(MetalRoboInspectorPolicyChoice(
            robotID: robotID,
            displayName: url.deletingPathExtension().lastPathComponent,
            policyURL: url
        ))
    }
    return choices.sorted {
        ($0.robotID.localizedStandardCompare($1.robotID) == .orderedAscending) ||
        ($0.robotID == $1.robotID &&
         $0.displayName.localizedStandardCompare($1.displayName) ==
            .orderedAscending)
    }
}

public func metalRoboInspectorPolicyCatalog() -> [
    MetalRoboInspectorPolicyChoice
] {
    let environment = ProcessInfo.processInfo.environment
    let directory: URL
    if let configured = environment["NUMI_WINDOW_POLICY_CATALOG"],
       !configured.isEmpty {
        directory = URL(fileURLWithPath: configured)
    } else {
        directory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".numi/policies", isDirectory: true)
    }
    return metalRoboInspectorPolicyChoices(in: directory)
}

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
    private static let policySelectorToolbarItem = NSToolbarItem.Identifier(
        "numi.inspector.policy-selector"
    )
    private let window: NSWindow
    private let view: MTKView
    private let queue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private var pending: InspectionDelivery?
    private var closed = false
    private var paused = false
    private var presentationVisible = true
    private var lastFrameSummary = "waiting for a frame"
    private var lastChromeUpdateSeconds: TimeInterval = 0
    private weak var pauseItem: NSToolbarItem?
    private weak var latestPolicyItem: NSToolbarItem?
    private weak var policySelector: NSPopUpButton?
    private let canReloadLatestPolicy: Bool
    private let policyChoices: [MetalRoboInspectorPolicyChoice]
    private let initialPolicyURL: URL?
    var onClose: (() -> Void)?
    var onPresentationEnabledChanged: ((Bool) -> Void)?
    var onLatestPolicyRequested: (() -> Void)?
    var onPolicySelected: ((URL) -> Void)?

    private init(
        window: NSWindow,
        view: MTKView,
        queue: MTLCommandQueue,
        pipeline: MTLRenderPipelineState,
        canReloadLatestPolicy: Bool,
        policyChoices: [MetalRoboInspectorPolicyChoice],
        initialPolicyURL: URL?
    ) {
        self.window = window
        self.view = view
        self.queue = queue
        self.pipeline = pipeline
        self.canReloadLatestPolicy = canReloadLatestPolicy
        self.policyChoices = policyChoices
        self.initialPolicyURL = initialPolicyURL
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
        selectDisplayedPolicy(initialPolicyURL)
        updateChrome(force: true)
        window.makeKeyAndOrderFront(nil)
    }

    static func make(
        canReloadLatestPolicy: Bool,
        policyChoices: [MetalRoboInspectorPolicyChoice],
        initialPolicyURL: URL?
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
        // The inspector writes the drawable exactly once and never samples or
        // copies it. Retain the render-target-only allocation path.
        view.framebufferOnly = true
        view.sampleCount = 1
        view.clearColor = MTLClearColor(red: 0.015, green: 0.017,
                                        blue: 0.024, alpha: 1)
        view.enableSetNeedsDisplay = true
        view.isPaused = true
        // The authored inspection frame is the useful resolution ceiling.
        // Avoid silently expanding a 960px preview into a 2x Retina drawable.
        view.autoResizeDrawable = false
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
            canReloadLatestPolicy: canReloadLatestPolicy,
            policyChoices: policyChoices,
            initialPolicyURL: initialPolicyURL
        )
    }

    func offer(_ delivery: InspectionDelivery) {
        guard presentationEnabled else {
            delivery.release()
            return
        }
        pending?.release()
        pending = delivery
        updateDrawableSize(
            sourceWidth: delivery.frame.width,
            sourceHeight: delivery.frame.height
        )
        lastFrameSummary = "env \(delivery.frame.environmentIndex) · frame \(delivery.frame.frameIndex) · dropped \(delivery.frame.droppedFrames)"
        updateChrome()
        view.setNeedsDisplay(view.bounds)
    }

    func showPolicyStatus(_ summary: String) {
        guard !closed else {
            return
        }
        lastFrameSummary = summary
        updateChrome(force: true)
    }

    func showInstalledPolicy(_ url: URL, revision: UInt64) {
        guard !closed else {
            return
        }
        selectDisplayedPolicy(url)
        lastFrameSummary =
            "\(url.deletingPathExtension().lastPathComponent) · revision \(revision)"
        updateChrome(force: true)
    }

    func showRejectedPolicy(activePolicyURL: URL?) {
        guard !closed else {
            return
        }
        selectDisplayedPolicy(activePolicyURL)
        lastFrameSummary = "policy rejected · unchanged"
        updateChrome(force: true)
    }

    func draw(in view: MTKView) {
        guard let delivery = pending,
              // Prepare all CPU-side work before taking a drawable. This
              // keeps CAMetalLayer ownership as short as possible.
              let command = queue.makeCommandBuffer(),
              let descriptor = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
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

    private func updateDrawableSize(sourceWidth: Int, sourceHeight: Int) {
        let points = view.bounds.size
        guard points.width > 0, points.height > 0,
              sourceWidth > 0, sourceHeight > 0 else {
            return
        }
        let usefulScale = min(
            1,
            min(
                CGFloat(sourceWidth) / points.width,
                CGFloat(sourceHeight) / points.height
            )
        )
        let size = CGSize(
            width: max(1, (points.width * usefulScale).rounded(.down)),
            height: max(1, (points.height * usefulScale).rounded(.down))
        )
        if view.drawableSize != size {
            view.drawableSize = size
        }
    }

    func toolbarAllowedItemIdentifiers(
        _ toolbar: NSToolbar
    ) -> [NSToolbarItem.Identifier] {
        [
            Self.pauseToolbarItem,
            Self.policySelectorToolbarItem,
            Self.latestPolicyToolbarItem,
        ]
    }

    func toolbarDefaultItemIdentifiers(
        _ toolbar: NSToolbar
    ) -> [NSToolbarItem.Identifier] {
        [
            Self.pauseToolbarItem,
            Self.policySelectorToolbarItem,
            Self.latestPolicyToolbarItem,
        ]
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
        if itemIdentifier == Self.policySelectorToolbarItem {
            let selector = NSPopUpButton(
                frame: NSRect(x: 0, y: 0, width: 220, height: 28),
                pullsDown: false
            )
            selector.target = self
            selector.action = #selector(selectPolicy)
            configurePolicySelector(selector)
            selector.setAccessibilityLabel("Policy")
            item.view = selector
            policySelector = selector
            selectDisplayedPolicy(initialPolicyURL)
            return item
        }
        return nil
    }

    @objc private func togglePause() {
        paused.toggle()
        updatePresentationState()
        updateChrome(force: true)
    }

    @objc private func requestLatestPolicy() {
        guard canReloadLatestPolicy else {
            return
        }
        lastFrameSummary = "loading latest policy"
        updateChrome(force: true)
        onLatestPolicyRequested?()
    }

    @objc private func selectPolicy(_ sender: NSPopUpButton) {
        guard let url = sender.selectedItem?.representedObject as? URL else {
            return
        }
        lastFrameSummary = "loading \(url.deletingPathExtension().lastPathComponent)"
        updateChrome(force: true)
        onPolicySelected?(url)
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

    private func configurePolicySelector(_ selector: NSPopUpButton) {
        selector.removeAllItems()
        if let initialPolicyURL,
           !policyChoices.contains(where: {
               $0.policyURL.standardizedFileURL ==
                   initialPolicyURL.standardizedFileURL
           }) {
            selector.addItem(
                withTitle: initialPolicyURL.deletingPathExtension()
                    .lastPathComponent
            )
            selector.lastItem?.representedObject = initialPolicyURL
            selector.lastItem?.toolTip = initialPolicyURL.path
            if !policyChoices.isEmpty {
                selector.menu?.addItem(.separator())
            }
        }
        guard !policyChoices.isEmpty || initialPolicyURL != nil else {
            selector.addItem(withTitle: "No policy catalog")
            selector.isEnabled = false
            return
        }
        var activeRobot = ""
        for choice in policyChoices {
            if choice.robotID != activeRobot {
                if !activeRobot.isEmpty {
                    selector.menu?.addItem(.separator())
                }
                let heading = NSMenuItem(
                    title: choice.robotID,
                    action: nil,
                    keyEquivalent: ""
                )
                heading.isEnabled = false
                selector.menu?.addItem(heading)
                activeRobot = choice.robotID
            }
            selector.addItem(withTitle: choice.displayName)
            guard let item = selector.lastItem else {
                continue
            }
            item.representedObject = choice.policyURL
            item.indentationLevel = 1
            item.toolTip = choice.policyURL.path
        }
    }

    private var presentationEnabled: Bool {
        !closed && !paused && presentationVisible
    }

    private func updatePresentationState() {
        if !presentationEnabled {
            pending?.release()
            pending = nil
        }
        onPresentationEnabledChanged?(presentationEnabled)
    }

    private func selectDisplayedPolicy(_ url: URL?) {
        guard let selector = policySelector, let url else {
            return
        }
        let selectedURL = url.standardizedFileURL
        guard let match = selector.itemArray.first(where: {
            ($0.representedObject as? URL)?.standardizedFileURL == selectedURL
        }) else {
            return
        }
        selector.select(match)
    }

    private func updateChrome(force: Bool = false) {
        let now = ProcessInfo.processInfo.systemUptime
        // Frame metadata can change much faster than a user can read it.
        // Throttle AppKit title invalidation while preserving immediate button
        // state and policy-result feedback below.
        if !force, now - lastChromeUpdateSeconds < 0.25 {
            return
        }
        lastChromeUpdateSeconds = now
        let state = paused ? "paused" : (presentationVisible ? "live" : "hidden")
        window.title = "Numi Lab Inspector — \(state) · \(lastFrameSummary)"
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

    func windowDidChangeOcclusionState(_ notification: Notification) {
        let visible = window.occlusionState.contains(.visible) &&
            !window.isMiniaturized
        guard presentationVisible != visible else {
            return
        }
        presentationVisible = visible
        updatePresentationState()
        updateChrome(force: true)
    }

    func windowDidMiniaturize(_ notification: Notification) {
        windowDidChangeOcclusionState(notification)
    }

    func windowDidDeminiaturize(_ notification: Notification) {
        windowDidChangeOcclusionState(notification)
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
        // A delivered frame is already valid. Alpha belongs to the authored
        // visual material and cannot be repurposed as frame validity: doing
        // so hides otherwise valid RGB from opaque cooked assets whose source
        // format does not carry an alpha channel.
        return float4(shaded, 1.0);
    }
    """
}

// This bridge is intentionally sendable: the frame slot is exclusively owned
// by the delivery until the MTKView command-buffer completion releases it.
public final class MetalRoboRunInspectorBridge: @unchecked Sendable {
    private let window: RunInspectorWindow
    private let stateLock = NSLock()
    private var closed = false
    private var presentationEnabled = true
    private var pendingNativeInspectionEnabled: Bool?
    private var latestPolicyReloadRequested = false
    private var pendingPolicySelection: URL?
    private var selectedPolicyURL: URL?

    @MainActor
    private init(window: RunInspectorWindow) {
        self.window = window
        window.onClose = { [weak self] in
            self?.markClosed()
        }
        window.onPresentationEnabledChanged = { [weak self] enabled in
            self?.setPresentationEnabled(enabled)
        }
        window.onLatestPolicyRequested = { [weak self] in
            self?.requestLatestPolicyReload()
        }
        window.onPolicySelected = { [weak self] url in
            self?.requestPolicySelection(url)
        }
    }

    @MainActor
    public static func launch(
        canReloadLatestPolicy: Bool = false,
        policyChoices: [MetalRoboInspectorPolicyChoice] = [],
        initialPolicyURL: URL? = nil
    ) throws -> MetalRoboRunInspectorBridge {
        precondition(Thread.isMainThread)
        let application = NSApplication.shared
        application.setActivationPolicy(.regular)
        let bridge = MetalRoboRunInspectorBridge(
            window: try RunInspectorWindow.make(
                canReloadLatestPolicy: canReloadLatestPolicy,
                policyChoices: policyChoices,
                initialPolicyURL: initialPolicyURL
            )
        )
        bridge.setInitialPolicy(initialPolicyURL)
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
        return !closed && presentationEnabled
    }

    private func markClosed() {
        stateLock.lock()
        closed = true
        presentationEnabled = false
        pendingNativeInspectionEnabled = false
        stateLock.unlock()
    }

    private func setPresentationEnabled(_ enabled: Bool) {
        stateLock.lock()
        let nextEnabled = enabled && !closed
        if presentationEnabled != nextEnabled {
            presentationEnabled = nextEnabled
            pendingNativeInspectionEnabled = nextEnabled
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

    private func requestPolicySelection(_ url: URL) {
        stateLock.lock()
        if !closed {
            pendingPolicySelection = url
        }
        stateLock.unlock()
    }

    private func setInitialPolicy(_ url: URL?) {
        stateLock.lock()
        selectedPolicyURL = url
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

    func takePolicySelection() -> URL? {
        stateLock.lock()
        defer { stateLock.unlock() }
        let selection = pendingPolicySelection
        pendingPolicySelection = nil
        return selection
    }

    var selectedPolicy: URL? {
        stateLock.lock()
        defer { stateLock.unlock() }
        return selectedPolicyURL
    }

    func reportPolicyInstalled(_ url: URL, revision: UInt64) {
        stateLock.lock()
        selectedPolicyURL = url
        stateLock.unlock()
        DispatchQueue.main.async { [window] in
            window.showInstalledPolicy(url, revision: revision)
        }
    }

    func reportPolicyRejected() {
        let activePolicyURL = selectedPolicy
        DispatchQueue.main.async { [window] in
            window.showRejectedPolicy(activePolicyURL: activePolicyURL)
        }
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
