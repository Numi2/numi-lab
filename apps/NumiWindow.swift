import AppKit
import Metal
import MetalKit
import simd

public struct NumiWindowPolicyChoice: Sendable {
    public let robotID: String
    public let displayName: String
    public let policyURL: URL

    public init(robotID: String, displayName: String, policyURL: URL) {
        self.robotID = robotID
        self.displayName = displayName
        self.policyURL = policyURL
    }
}

public struct NumiWindowSceneChoice: Sendable {
    public let id: String
    public let robotID: String
    public let robotName: String
    public let sceneID: String
    public let sceneName: String
    public let taskID: String
    public let taskName: String
    public let visualObservationURL: URL
    public let launchArguments: [String]
    public let trainingArguments: [String]

    public var isTrainable: Bool { !trainingArguments.isEmpty }

    public var policyURL: URL? {
        guard let option = launchArguments.firstIndex(of: "--policy-pack"),
              option + 1 < launchArguments.count
        else {
            return nil
        }
        return URL(fileURLWithPath: launchArguments[option + 1])
            .standardizedFileURL
    }

    fileprivate struct Record: Decodable {
        let format: String
        let id: String
        let robotID: String
        let robotName: String
        let sceneID: String
        let sceneName: String
        let taskID: String?
        let taskName: String?
        let visualObservation: String
        let arguments: [String]
        let trainingArguments: [String]?
        let available: Bool?

        enum CodingKeys: String, CodingKey {
            case format, id, arguments, available
            case robotID = "robot_id"
            case robotName = "robot_name"
            case sceneID = "scene_id"
            case sceneName = "scene_name"
            case taskID = "task_id"
            case taskName = "task_name"
            case visualObservation = "visual_observation"
            case trainingArguments = "training_arguments"
        }
    }
}

public func numiWindowPreferredSceneChoice(
    _ choices: [NumiWindowSceneChoice],
    robotID: String? = nil
) -> NumiWindowSceneChoice? {
    let candidates = choices.filter { choice in
        robotID == nil || choice.robotID == robotID
    }
    return candidates.first(where: {
        $0.isTrainable && $0.sceneID == "ground" && $0.taskID == "velocity"
    }) ?? candidates.first(where: { $0.isTrainable }) ?? candidates.first
}

private let numiWindowPathOptions: Set<String> = [
    "--urdf", "--srdf", "--world-pack", "--task-pack",
    "--robot-actuator-pack", "--sensor-program-pack",
    "--reality-program-pack", "--interaction-pack", "--policy-pack",
    "--action-stream", "--dove-manifest",
]

private func numiWindowResolvedArguments(
    _ arguments: [String],
    relativeTo directory: URL
) -> [String] {
    var result = arguments
    var index = 0
    while index + 1 < result.count {
        let option = result[index]
        if numiWindowPathOptions.contains(option),
           !result[index + 1].hasPrefix("/") {
            result[index + 1] = directory
                .appendingPathComponent(result[index + 1])
                .standardizedFileURL.path
            index += 1
        }
        index += 1
    }
    return result
}

private func numiWindowArgumentValue(
    _ option: String,
    in arguments: [String]
) -> String? {
    guard let index = arguments.firstIndex(of: option),
          index + 1 < arguments.count else { return nil }
    return arguments[index + 1]
}

public func numiWindowSceneChoices(in directory: URL) -> [NumiWindowSceneChoice] {
    guard let entries = FileManager.default.enumerator(
        at: directory,
        includingPropertiesForKeys: [.isRegularFileKey],
        // The production catalog normally lives under the workspace's
        // hidden `.numi` directory. Skipping hidden descendants here makes
        // every valid catalog appear empty.
        options: [.skipsPackageDescendants]
    ) else {
        return []
    }
    let decoder = JSONDecoder()
    var choices: [NumiWindowSceneChoice] = []
    var seen: Set<String> = []
    for case let url as URL in entries where url.lastPathComponent.hasSuffix(
        ".numi-window.json"
    ) {
        guard let data = try? Data(contentsOf: url),
              let record = try? decoder.decode(
                  NumiWindowSceneChoice.Record.self,
                  from: data
              ),
              record.format == "numi.window.scene.v1",
              !record.id.isEmpty,
              !record.robotID.isEmpty,
              !record.sceneID.isEmpty,
              record.available != false,
              seen.insert(record.id).inserted
        else {
            continue
        }
        let base = url.deletingLastPathComponent()
        let resolvedArguments = numiWindowResolvedArguments(
            record.arguments,
            relativeTo: base
        )
        let resolvedTrainingArguments = numiWindowResolvedArguments(
            record.trainingArguments ?? [],
            relativeTo: base
        )
        let legacyEnvironment =
            numiWindowArgumentValue("--scene", in: resolvedArguments) ??
            "studio"
        let environmentID = record.taskID == nil
            ? legacyEnvironment : record.sceneID
        let environmentName = record.taskID == nil
            ? legacyEnvironment.replacingOccurrences(of: "-", with: " ")
                .capitalized
            : record.sceneName
        let visualURL = URL(fileURLWithPath: record.visualObservation,
                            relativeTo: base).standardizedFileURL
        guard FileManager.default.fileExists(atPath: visualURL.path) else {
            continue
        }
        choices.append(NumiWindowSceneChoice(
            id: record.id,
            robotID: record.robotID,
            robotName: record.robotName,
            sceneID: environmentID,
            sceneName: environmentName,
            taskID: record.taskID ?? record.sceneID,
            taskName: record.taskName ?? record.sceneName,
            visualObservationURL: visualURL,
            launchArguments: resolvedArguments,
            trainingArguments: resolvedTrainingArguments
        ))
    }
    return choices.sorted {
        ($0.robotName.localizedStandardCompare($1.robotName) == .orderedAscending) ||
        ($0.robotName == $1.robotName &&
         $0.sceneName.localizedStandardCompare($1.sceneName) == .orderedAscending)
    }
}

public func numiWindowSceneCatalog() -> [NumiWindowSceneChoice] {
    let environment = ProcessInfo.processInfo.environment
    let directory = URL(fileURLWithPath:
        environment["NUMI_WINDOW_SCENE_CATALOG"] ??
        FileManager.default.currentDirectoryPath + "/.numi/runs",
        isDirectory: true
    )
    // Keep every authored robot/scene contract. The selectors already group
    // these by robot, and collapsing them to one synthetic "Studio" choice
    // hid the distinct locomotion, recovery, manipulation, and flight scenes.
    return numiWindowSceneChoices(in: directory)
}

public func numiWindowPolicyChoices(
    in directory: URL
) -> [NumiWindowPolicyChoice] {
    guard let entries = FileManager.default.enumerator(
        at: directory,
        includingPropertiesForKeys: [.isRegularFileKey],
        options: [.skipsPackageDescendants]
    ) else {
        return []
    }
    var choices: [NumiWindowPolicyChoice] = []
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
        choices.append(NumiWindowPolicyChoice(
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

public func numiWindowPolicyCatalog(
    sceneChoices: [NumiWindowSceneChoice] = []
) -> [
    NumiWindowPolicyChoice
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
    var choices = numiWindowPolicyChoices(in: directory)
    var seen = Set(choices.map { $0.policyURL.standardizedFileURL })
    let sceneDirectory = URL(fileURLWithPath:
        environment["NUMI_WINDOW_SCENE_CATALOG"] ??
        FileManager.default.currentDirectoryPath + "/.numi/runs",
        isDirectory: true
    )
    let policyPresets = numiWindowSceneChoices(in: sceneDirectory)
    for scene in sceneChoices + policyPresets {
        guard let option = scene.launchArguments.firstIndex(of: "--policy-pack"),
              option + 1 < scene.launchArguments.count
        else {
            continue
        }
        let url = URL(fileURLWithPath: scene.launchArguments[option + 1])
            .standardizedFileURL
        guard FileManager.default.fileExists(atPath: url.path),
              seen.insert(url).inserted
        else {
            continue
        }
        choices.append(NumiWindowPolicyChoice(
            robotID: scene.robotID,
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

private final class WindowFrameDelivery: @unchecked Sendable {
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

struct NumiWindowCameraControl: Sendable {
    let translation: SIMD3<Float>
    let orientation: SIMD4<Float>
}

@MainActor
private final class NumiWindowMetalView: MTKView {
    var onOrbit: ((CGFloat, CGFloat) -> Void)?
    var onPan: ((CGFloat, CGFloat) -> Void)?
    var onDolly: ((CGFloat) -> Void)?
    var onReset: (() -> Void)?
    var onPause: (() -> Void)?
    var onTrain: (() -> Void)?
    var onHelp: (() -> Void)?
    private var lastDragLocation: NSPoint?

    override var acceptsFirstResponder: Bool { true }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        lastDragLocation = convert(event.locationInWindow, from: nil)
    }

    override func mouseDragged(with event: NSEvent) {
        let location = convert(event.locationInWindow, from: nil)
        if let previous = lastDragLocation {
            let dx = location.x - previous.x
            let dy = location.y - previous.y
            if event.modifierFlags.contains(.shift) {
                onPan?(dx, dy)
            } else {
                onOrbit?(dx, dy)
            }
        }
        lastDragLocation = location
    }

    override func rightMouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        lastDragLocation = convert(event.locationInWindow, from: nil)
    }

    override func rightMouseDragged(with event: NSEvent) {
        let location = convert(event.locationInWindow, from: nil)
        if let previous = lastDragLocation {
            onPan?(location.x - previous.x, location.y - previous.y)
        }
        lastDragLocation = location
    }

    override func rightMouseUp(with event: NSEvent) {
        lastDragLocation = nil
    }

    override func scrollWheel(with event: NSEvent) {
        onDolly?(event.scrollingDeltaY)
    }

    override func mouseUp(with event: NSEvent) {
        if event.clickCount == 2 {
            onReset?()
        }
        lastDragLocation = nil
    }

    override func keyDown(with event: NSEvent) {
        switch event.charactersIgnoringModifiers?.lowercased() {
        case " ": onPause?()
        case "t": onTrain?()
        case "r": onReset?()
        case "?": onHelp?()
        default: super.keyDown(with: event)
        }
    }
}

enum NumiWindowError: Error {
    case closed
    case reconfigure(String)
    case train(NumiWindowTrainingRequest)
}

@MainActor
private final class NumiWindowController: NSObject, MTKViewDelegate,
    NSWindowDelegate, NSToolbarDelegate
{
    private static let pauseToolbarItem = NSToolbarItem.Identifier(
        "numi.window.pause"
    )
    private static let latestPolicyToolbarItem = NSToolbarItem.Identifier(
        "numi.window.latest-policy"
    )
    private static let policySelectorToolbarItem = NSToolbarItem.Identifier(
        "numi.window.policy-selector"
    )
    private static let robotSelectorToolbarItem = NSToolbarItem.Identifier(
        "numi.window.robot-selector"
    )
    private static let sceneSelectorToolbarItem = NSToolbarItem.Identifier(
        "numi.window.scene-selector"
    )
    private static let taskSelectorToolbarItem = NSToolbarItem.Identifier(
        "numi.window.task-selector"
    )
    private static let resetCameraToolbarItem = NSToolbarItem.Identifier(
        "numi.window.reset-camera"
    )
    private static let trainToolbarItem = NSToolbarItem.Identifier(
        "numi.window.train"
    )
    private let window: NSWindow
    private let view: NumiWindowMetalView
    private let queue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private var pending: WindowFrameDelivery?
    private var closed = false
    private var paused = false
    private var presentationVisible = true
    private var reconfiguring = false
    private var reconfigurationReenableScheduled = false
    private var statusSummary: String? = "waiting for a frame"
    private var latestEnvironmentIndex: UInt32 = 0
    private var latestFrameIndex: UInt64 = 0
    private var latestDroppedFrames: UInt32 = 0
    private var hasPresentedFrame = false
    private var lastChromeUpdateSeconds: TimeInterval = 0
    private var lastDrawableSource = SIMD2<Int32>(repeating: 0)
    private var lastDrawableBounds = CGSize.zero
    private weak var pauseItem: NSToolbarItem?
    private weak var latestPolicyItem: NSToolbarItem?
    private weak var policySelector: NSPopUpButton?
    private weak var robotSelector: NSPopUpButton?
    private weak var sceneSelector: NSPopUpButton?
    private weak var taskSelector: NSPopUpButton?
    private weak var trainItem: NSToolbarItem?
    private var trainingSetupController: NumiTrainingSetupController?
    private var trainingActive = false
    private let trainingBanner = NSVisualEffectView()
    private let trainingSpinner = NSProgressIndicator()
    private let trainingLabel = NSTextField(wrappingLabelWithString: "")
    private var trainingArtifactsURL: URL?
    private let coachCard = NSVisualEffectView()
    private let coachTitle = NSTextField(labelWithString: "Your first robot policy")
    private let coachDetail = NSTextField(wrappingLabelWithString: "")
    private var coachDismissed = false
    private let canReloadLatestPolicy: Bool
    private let policyChoices: [NumiWindowPolicyChoice]
    private let initialPolicyURL: URL?
    private let sceneChoices: [NumiWindowSceneChoice]
    private let initialSceneID: String?
    private var cameraYaw: Float = 0
    private var cameraPitch: Float = 0
    private var cameraPan = SIMD2<Float>(repeating: 0)
    private var cameraDolly: Float = 0
    private var activeRobotID: String
    private var activeSceneID: String
    var onClose: (() -> Void)?
    var onPresentationEnabledChanged: ((Bool) -> Void)?
    var onLatestPolicyRequested: (() -> Void)?
    var onPolicySelected: ((URL) -> Void)?
    var onSceneSelected: ((String) -> Void)?
    var onTrainingRequested: ((NumiWindowTrainingRequest) -> Void)?
    var onCameraChanged: ((NumiWindowCameraControl) -> Void)?

    private init(
        window: NSWindow,
        view: NumiWindowMetalView,
        queue: MTLCommandQueue,
        pipeline: MTLRenderPipelineState,
        canReloadLatestPolicy: Bool,
        policyChoices: [NumiWindowPolicyChoice],
        initialPolicyURL: URL?,
        sceneChoices: [NumiWindowSceneChoice],
        initialSceneID: String?
    ) {
        self.window = window
        self.view = view
        self.queue = queue
        self.pipeline = pipeline
        self.canReloadLatestPolicy = canReloadLatestPolicy
        self.policyChoices = policyChoices
        self.initialPolicyURL = initialPolicyURL
        self.sceneChoices = sceneChoices
        self.initialSceneID = initialSceneID
        self.activeRobotID = sceneChoices.first(where: {
            $0.id == initialSceneID
        })?.robotID ?? sceneChoices.first?.robotID ?? ""
        self.activeSceneID = sceneChoices.first(where: {
            $0.id == initialSceneID
        })?.sceneID ?? sceneChoices.first?.sceneID ?? ""
        super.init()
        view.delegate = self
        window.contentView = view
        window.delegate = self
        let toolbar = NSToolbar(identifier: "numi.window.toolbar")
        toolbar.delegate = self
        toolbar.displayMode = .iconAndLabel
        toolbar.allowsUserCustomization = false
        toolbar.autosavesConfiguration = false
        window.toolbar = toolbar
        configureTrainingBanner()
        configureCoachCard()
        view.onOrbit = { [weak self] dx, dy in
            self?.orbitCamera(dx: dx, dy: dy)
        }
        view.onPan = { [weak self] dx, dy in
            self?.panCamera(dx: dx, dy: dy)
        }
        view.onDolly = { [weak self] delta in
            self?.dollyCamera(delta: delta)
        }
        view.onReset = { [weak self] in self?.resetCamera() }
        view.onPause = { [weak self] in self?.togglePause() }
        view.onTrain = { [weak self] in self?.configureTraining() }
        view.onHelp = { [weak self] in self?.showCoach() }
        selectDisplayedPolicy(initialPolicyURL)
        updateChrome(force: true)
        updateCoachCard()
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(view)
    }

    static func make(
        canReloadLatestPolicy: Bool,
        policyChoices: [NumiWindowPolicyChoice],
        initialPolicyURL: URL?,
        sceneChoices: [NumiWindowSceneChoice] = [],
        initialSceneID: String? = nil
    ) throws -> NumiWindowController {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue()
        else {
            throw MetalRoboTaskRolloutError.native(
                "Metal window could not create a display device."
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
        let view = NumiWindowMetalView(
            frame: NSRect(x: 0, y: 0, width: 1180, height: 700),
            device: device
        )
        view.colorPixelFormat = .bgra8Unorm_srgb
        // The window writes the drawable exactly once and never samples or
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
            contentRect: NSRect(x: 120, y: 120, width: 1180, height: 700),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        return NumiWindowController(
            window: window,
            view: view,
            queue: queue,
            pipeline: pipeline,
            canReloadLatestPolicy: canReloadLatestPolicy,
            policyChoices: policyChoices,
            initialPolicyURL: initialPolicyURL,
            sceneChoices: sceneChoices,
            initialSceneID: initialSceneID
        )
    }

    func offer(_ delivery: WindowFrameDelivery) {
        guard presentationEnabled else {
            delivery.release()
            return
        }
        finishReconfiguration()
        pending?.release()
        pending = delivery
        updateDrawableSize(
            sourceWidth: delivery.frame.width,
            sourceHeight: delivery.frame.height
        )
        latestEnvironmentIndex = delivery.frame.environmentIndex
        latestFrameIndex = delivery.frame.frameIndex
        latestDroppedFrames = delivery.frame.droppedFrames
        hasPresentedFrame = true
        statusSummary = nil
        updateChrome()
        view.setNeedsDisplay(view.bounds)
    }

    func showPolicyStatus(_ summary: String) {
        guard !closed else {
            return
        }
        statusSummary = summary
        updateChrome(force: true)
    }

    func showInstalledPolicy(_ url: URL, revision: UInt64) {
        guard !closed else {
            return
        }
        selectDisplayedPolicy(url)
        statusSummary =
            "\(url.deletingPathExtension().lastPathComponent) · revision \(revision)"
        updateCoachCard()
        updateChrome(force: true)
    }

    func showRejectedPolicy(activePolicyURL: URL?) {
        guard !closed else {
            return
        }
        selectDisplayedPolicy(activePolicyURL)
        statusSummary = "policy rejected · unchanged"
        updateChrome(force: true)
    }

    func showRestoredScene(_ id: String, summary: String) {
        guard !closed else { return }
        finishReconfiguration()
        selectDisplayedScene(id)
        statusSummary = summary
        updateCoachCard()
        updateChrome(force: true)
    }

    func showTrainingStatus(
        _ summary: String,
        active: Bool,
        policyURL: URL? = nil,
        artifactsURL: URL? = nil
    ) {
        guard !closed else { return }
        trainingActive = active
        if let artifactsURL {
            trainingArtifactsURL = artifactsURL
        }
        trainingBanner.isHidden = false
        trainingLabel.stringValue = summary
        if active {
            trainingSpinner.isHidden = false
            trainingSpinner.startAnimation(nil)
        } else {
            trainingSpinner.stopAnimation(nil)
            trainingSpinner.isHidden = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 10) {
                [weak self] in
                guard let self, !self.trainingActive else { return }
                self.trainingBanner.isHidden = true
            }
        }
        if let policyURL, let policySelector {
            let standardized = policyURL.standardizedFileURL
            if !policySelector.itemArray.contains(where: {
                ($0.representedObject as? URL)?.standardizedFileURL ==
                    standardized
            }) {
                policySelector.addItem(
                    withTitle: policyURL.deletingPathExtension()
                        .lastPathComponent
                )
                policySelector.lastItem?.representedObject = policyURL
                policySelector.lastItem?.toolTip = policyURL.path
            }
            selectDisplayedPolicy(policyURL)
        }
        statusSummary = summary
        setConfigurationControlsEnabled(!active && !reconfiguring)
        updateCoachCard()
        updateChrome(force: true)
    }

    private func configureTrainingBanner() {
        trainingBanner.material = .hudWindow
        trainingBanner.blendingMode = .withinWindow
        trainingBanner.state = .active
        trainingBanner.wantsLayer = true
        trainingBanner.layer?.cornerRadius = 12
        trainingBanner.translatesAutoresizingMaskIntoConstraints = false
        trainingBanner.isHidden = true
        trainingSpinner.style = .spinning
        trainingSpinner.controlSize = .small
        trainingLabel.font = .systemFont(ofSize: 13, weight: .semibold)
        trainingLabel.textColor = .labelColor
        let reveal = NSButton(
            title: "Show Results",
            target: self,
            action: #selector(revealTrainingArtifacts)
        )
        reveal.bezelStyle = .rounded
        let stack = NSStackView(views: [
            trainingSpinner,
            trainingLabel,
            reveal,
        ])
        stack.orientation = .horizontal
        stack.alignment = .centerY
        stack.spacing = 12
        stack.translatesAutoresizingMaskIntoConstraints = false
        trainingBanner.addSubview(stack)
        view.addSubview(trainingBanner)
        NSLayoutConstraint.activate([
            trainingBanner.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            trainingBanner.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: -22),
            trainingBanner.widthAnchor.constraint(lessThanOrEqualToConstant: 760),
            trainingBanner.leadingAnchor.constraint(greaterThanOrEqualTo: view.leadingAnchor, constant: 24),
            trainingBanner.trailingAnchor.constraint(lessThanOrEqualTo: view.trailingAnchor, constant: -24),
            stack.leadingAnchor.constraint(equalTo: trainingBanner.leadingAnchor, constant: 16),
            stack.trailingAnchor.constraint(equalTo: trainingBanner.trailingAnchor, constant: -16),
            stack.topAnchor.constraint(equalTo: trainingBanner.topAnchor, constant: 10),
            stack.bottomAnchor.constraint(equalTo: trainingBanner.bottomAnchor, constant: -10),
            trainingLabel.widthAnchor.constraint(greaterThanOrEqualToConstant: 360),
        ])
    }

    private func configureCoachCard() {
        coachCard.material = .hudWindow
        coachCard.blendingMode = .withinWindow
        coachCard.state = .active
        coachCard.wantsLayer = true
        coachCard.layer?.cornerRadius = 14
        coachCard.translatesAutoresizingMaskIntoConstraints = false
        coachTitle.font = .systemFont(ofSize: 15, weight: .semibold)
        coachDetail.font = .systemFont(ofSize: 12)
        coachDetail.textColor = .secondaryLabelColor
        coachDetail.maximumNumberOfLines = 3
        coachDetail.stringValue =
            "Choose a robot, environment, and task above. Then train on native Metal; only policies that stay upright enter the Policy menu."
        let start = NSButton(
            title: "Configure Training",
            target: self,
            action: #selector(configureTraining)
        )
        start.bezelStyle = .rounded
        start.bezelColor = .systemGreen
        start.image = NSImage(
            systemSymbolName: "graduationcap.fill",
            accessibilityDescription: "Configure Training"
        )
        start.imagePosition = .imageLeading
        let dismiss = NSButton(
            image: NSImage(
                systemSymbolName: "xmark",
                accessibilityDescription: "Dismiss learning guide"
            ) ?? NSImage(),
            target: self,
            action: #selector(dismissCoach)
        )
        dismiss.bezelStyle = .inline
        dismiss.isBordered = false
        let header = NSStackView(views: [coachTitle, NSView(), dismiss])
        header.orientation = .horizontal
        header.alignment = .centerY
        header.spacing = 8
        let stack = NSStackView(views: [header, coachDetail, start])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 9
        stack.translatesAutoresizingMaskIntoConstraints = false
        coachCard.addSubview(stack)
        view.addSubview(coachCard)
        NSLayoutConstraint.activate([
            coachCard.leadingAnchor.constraint(
                equalTo: view.leadingAnchor,
                constant: 22
            ),
            coachCard.bottomAnchor.constraint(
                equalTo: view.bottomAnchor,
                constant: -22
            ),
            coachCard.widthAnchor.constraint(equalToConstant: 390),
            stack.leadingAnchor.constraint(
                equalTo: coachCard.leadingAnchor,
                constant: 16
            ),
            stack.trailingAnchor.constraint(
                equalTo: coachCard.trailingAnchor,
                constant: -12
            ),
            stack.topAnchor.constraint(
                equalTo: coachCard.topAnchor,
                constant: 12
            ),
            stack.bottomAnchor.constraint(
                equalTo: coachCard.bottomAnchor,
                constant: -12
            ),
            header.widthAnchor.constraint(equalTo: stack.widthAnchor),
            coachDetail.widthAnchor.constraint(equalToConstant: 350),
        ])
        coachCard.setAccessibilityLabel("Learn robotics guide")
    }

    private func updateCoachCard() {
        let hasPolicy = policySelector?.itemArray.contains(where: {
            $0.representedObject is URL
        }) ?? false
        coachCard.isHidden = coachDismissed || trainingActive || hasPolicy ||
            activeChoice?.isTrainable != true
    }

    @objc private func dismissCoach() {
        coachDismissed = true
        updateCoachCard()
    }

    private func showCoach() {
        coachDismissed = false
        updateCoachCard()
    }

    @objc private func revealTrainingArtifacts() {
        guard let trainingArtifactsURL else { return }
        NSWorkspace.shared.activateFileViewerSelecting([trainingArtifactsURL])
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
        let source = SIMD2(Int32(sourceWidth), Int32(sourceHeight))
        guard source != lastDrawableSource || points != lastDrawableBounds else {
            return
        }
        lastDrawableSource = source
        lastDrawableBounds = points
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
            Self.robotSelectorToolbarItem,
            Self.sceneSelectorToolbarItem,
            Self.taskSelectorToolbarItem,
            Self.policySelectorToolbarItem,
            Self.trainToolbarItem,
            Self.resetCameraToolbarItem,
        ]
    }

    func toolbarDefaultItemIdentifiers(
        _ toolbar: NSToolbar
    ) -> [NSToolbarItem.Identifier] {
        [
            Self.pauseToolbarItem,
            Self.robotSelectorToolbarItem,
            Self.sceneSelectorToolbarItem,
            Self.taskSelectorToolbarItem,
            Self.policySelectorToolbarItem,
            Self.trainToolbarItem,
            Self.resetCameraToolbarItem,
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
        if itemIdentifier == Self.resetCameraToolbarItem {
            item.label = "Reset View"
            item.toolTip =
                "R or double-click resets · drag orbits · Shift-drag pans · scroll zooms"
            item.image = NSImage(
                systemSymbolName: "view.3d",
                accessibilityDescription: item.label
            )
            item.action = #selector(resetCameraAction)
            return item
        }
        if itemIdentifier == Self.trainToolbarItem {
            item.action = #selector(configureTraining)
            configureTrainItem(item)
            trainItem = item
            return item
        }
        if itemIdentifier == Self.policySelectorToolbarItem {
            let selector = NSPopUpButton(
                frame: NSRect(x: 0, y: 0, width: 220, height: 28),
                pullsDown: false
            )
            selector.widthAnchor.constraint(equalToConstant: 180).isActive = true
            selector.target = self
            selector.action = #selector(selectPolicy)
            configurePolicySelector(
                selector,
                for: activeRobotID,
                preferred: initialPolicyURL
            )
            selector.setAccessibilityLabel("Policy")
            item.label = "Policy"
            item.toolTip = "Select a compatible saved PolicyPack"
            item.view = selector
            policySelector = selector
            selectDisplayedPolicy(initialPolicyURL)
            return item
        }
        if itemIdentifier == Self.robotSelectorToolbarItem {
            let selector = NSPopUpButton(
                frame: NSRect(x: 0, y: 0, width: 170, height: 28),
                pullsDown: false
            )
            selector.widthAnchor.constraint(equalToConstant: 150).isActive = true
            selector.target = self
            selector.action = #selector(selectRobot)
            selector.setAccessibilityLabel("Robot")
            item.label = "Robot"
            item.view = selector
            robotSelector = selector
            configureRobotSelector(selector)
            return item
        }
        if itemIdentifier == Self.sceneSelectorToolbarItem {
            let selector = NSPopUpButton(
                frame: NSRect(x: 0, y: 0, width: 170, height: 28),
                pullsDown: false
            )
            selector.widthAnchor.constraint(equalToConstant: 150).isActive = true
            selector.target = self
            selector.action = #selector(selectEnvironment)
            selector.setAccessibilityLabel("Environment")
            item.label = "Environment"
            item.toolTip = "Choose the visible physical world"
            item.view = selector
            sceneSelector = selector
            selectDisplayedScene(initialSceneID)
            return item
        }
        if itemIdentifier == Self.taskSelectorToolbarItem {
            let selector = NSPopUpButton(
                frame: NSRect(x: 0, y: 0, width: 190, height: 28),
                pullsDown: false
            )
            selector.widthAnchor.constraint(equalToConstant: 240).isActive = true
            selector.target = self
            selector.action = #selector(selectTask)
            selector.setAccessibilityLabel("Task")
            item.label = "Task"
            item.toolTip = "Choose what the robot should practice"
            item.view = selector
            taskSelector = selector
            selectDisplayedScene(initialSceneID)
            return item
        }
        return nil
    }

    @objc private func selectRobot(_ sender: NSPopUpButton) {
        guard let robotID = sender.selectedItem?.representedObject as? String,
              let choice = numiWindowPreferredSceneChoice(
                  sceneChoices,
                  robotID: robotID
              )
        else {
            return
        }
        configureEnvironmentSelector(for: robotID, selecting: choice.sceneID)
        configureTaskSelector(
            for: robotID,
            environmentID: choice.sceneID,
            selecting: choice.id
        )
        activeRobotID = robotID
        activeSceneID = choice.sceneID
        if let policySelector {
            configurePolicySelector(
                policySelector,
                for: robotID,
                preferred: choice.policyURL
            )
        }
        requestScene(choice)
    }

    @objc private func selectEnvironment(_ sender: NSPopUpButton) {
        let currentTaskID = (taskSelector?.selectedItem?.representedObject
            as? String).flatMap { selectedID in
                sceneChoices.first(where: { $0.id == selectedID })?.taskID
            }
        guard let environmentID = sender.selectedItem?.representedObject as? String,
              let choice = sceneChoices.first(where: {
                  $0.robotID == activeRobotID &&
                  $0.sceneID == environmentID &&
                  $0.taskID == currentTaskID
              }) ?? sceneChoices.first(where: {
                  $0.robotID == activeRobotID && $0.sceneID == environmentID
              })
        else {
            return
        }
        activeSceneID = environmentID
        configureTaskSelector(
            for: activeRobotID,
            environmentID: environmentID,
            selecting: choice.id
        )
        requestScene(choice)
    }

    @objc private func selectTask(_ sender: NSPopUpButton) {
        guard let id = sender.selectedItem?.representedObject as? String,
              let choice = sceneChoices.first(where: { $0.id == id })
        else { return }
        requestScene(choice)
    }

    private var activeChoice: NumiWindowSceneChoice? {
        if let id = taskSelector?.selectedItem?.representedObject as? String,
           let choice = sceneChoices.first(where: { $0.id == id }) {
            return choice
        }
        if let initialSceneID,
           let choice = sceneChoices.first(where: {
               $0.id == initialSceneID
           }) {
            return choice
        }
        return sceneChoices.first(where: {
            $0.robotID == activeRobotID && $0.sceneID == activeSceneID
        })
    }

    @objc private func configureTraining() {
        guard !trainingActive,
              trainingSetupController == nil,
              let choice = activeChoice,
              choice.isTrainable
        else { return }
        let selectedPolicy = policySelector?.selectedItem?
            .representedObject as? URL
        let setup = NumiTrainingSetupController(
            choice: choice,
            selectedPolicyURL: selectedPolicy
        ) { [weak self] request in
            guard let self else { return }
            self.trainingActive = true
            self.statusSummary =
                "preparing training · \(request.taskName)"
            self.setConfigurationControlsEnabled(false)
            self.updateChrome(force: true)
            self.onTrainingRequested?(request)
        }
        setup.onDismiss = { [weak self] in
            self?.trainingSetupController = nil
        }
        trainingSetupController = setup
        setup.present(over: window)
    }

    private func requestScene(_ choice: NumiWindowSceneChoice) {
        guard !reconfiguring else {
            return
        }
        reconfiguring = true
        let robotChanged = activeRobotID != choice.robotID
        robotSelector?.isEnabled = false
        sceneSelector?.isEnabled = false
        taskSelector?.isEnabled = false
        policySelector?.isEnabled = false
        trainItem?.isEnabled = false
        // Camera offsets are relative to each robot's authored camera. Never
        // carry an orbit from one robot into another robot's frame.
        resetCamera()
        activeRobotID = choice.robotID
        activeSceneID = choice.sceneID
        if robotChanged, let policySelector {
            configurePolicySelector(
                policySelector,
                for: choice.robotID,
                preferred: nil
            )
        }
        statusSummary =
            "loading \(choice.sceneName) · \(choice.taskName)"
        updateCoachCard()
        updateChrome(force: true)
        onSceneSelected?(choice.id)
    }

    private func finishReconfiguration() {
        guard reconfiguring, !reconfigurationReenableScheduled else {
            return
        }
        reconfigurationReenableScheduled = true
        // A newly displayed frame proves the replacement runtime is live,
        // but its predecessor can still have one presentation command in
        // flight. Keep selection serialized through a short retirement
        // window so rapid clicks cannot stack complete Metal worlds.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.75) { [weak self] in
            guard let self, !self.closed else {
                return
            }
            self.reconfigurationReenableScheduled = false
            self.reconfiguring = false
            self.setConfigurationControlsEnabled(!self.trainingActive)
        }
    }

    private func setConfigurationControlsEnabled(_ enabled: Bool) {
        robotSelector?.isEnabled = enabled && !sceneChoices.isEmpty
        sceneSelector?.isEnabled = enabled &&
            (sceneSelector?.numberOfItems ?? 0) > 0
        taskSelector?.isEnabled = enabled &&
            (taskSelector?.numberOfItems ?? 0) > 0
        policySelector?.isEnabled = enabled &&
            (policySelector?.itemArray.contains {
                $0.representedObject is URL
            } ?? false)
        if let trainItem {
            configureTrainItem(trainItem)
        }
    }

    private func configureRobotSelector(_ selector: NSPopUpButton) {
        selector.removeAllItems()
        var seen: Set<String> = []
        for choice in sceneChoices where seen.insert(choice.robotID).inserted {
            selector.addItem(withTitle: choice.robotName)
            selector.lastItem?.representedObject = choice.robotID
        }
        guard !sceneChoices.isEmpty else {
            selector.addItem(withTitle: "No robot catalog")
            selector.isEnabled = false
            configureEnvironmentSelector(for: "", selecting: nil)
            configureTaskSelector(
                for: "", environmentID: "", selecting: nil
            )
            return
        }
        let initial = sceneChoices.first(where: { $0.id == initialSceneID }) ??
            numiWindowPreferredSceneChoice(sceneChoices) ?? sceneChoices[0]
        selector.selectItem(withTitle: initial.robotName)
        configureEnvironmentSelector(
            for: initial.robotID,
            selecting: initial.sceneID
        )
        configureTaskSelector(
            for: initial.robotID,
            environmentID: initial.sceneID,
            selecting: initial.id
        )
    }

    private func configureEnvironmentSelector(
        for robotID: String,
        selecting selectedEnvironmentID: String?
    ) {
        guard let selector = sceneSelector else {
            return
        }
        selector.removeAllItems()
        var seen: Set<String> = []
        for choice in sceneChoices where
            choice.robotID == robotID && seen.insert(choice.sceneID).inserted {
            selector.addItem(withTitle: choice.sceneName)
            selector.lastItem?.representedObject = choice.sceneID
        }
        selector.isEnabled = selector.numberOfItems > 0
        if let selectedEnvironmentID {
            selector.itemArray.first(where: {
                ($0.representedObject as? String) == selectedEnvironmentID
            }).map { selector.select($0) }
        }
    }

    private func configureTaskSelector(
        for robotID: String,
        environmentID: String,
        selecting selectedID: String?
    ) {
        guard let selector = taskSelector else { return }
        selector.removeAllItems()
        for choice in sceneChoices where
            choice.robotID == robotID && choice.sceneID == environmentID {
            selector.addItem(withTitle: choice.taskName)
            selector.lastItem?.representedObject = choice.id
        }
        selector.isEnabled = selector.numberOfItems > 0
        if let selectedID {
            selector.itemArray.first(where: {
                ($0.representedObject as? String) == selectedID
            }).map { selector.select($0) }
        }
    }

    private func selectDisplayedScene(_ id: String?) {
        guard let id,
              let choice = sceneChoices.first(where: { $0.id == id })
        else {
            return
        }
        robotSelector?.itemArray.first(where: {
            ($0.representedObject as? String) == choice.robotID
        }).map { robotSelector?.select($0) }
        let robotChanged = activeRobotID != choice.robotID
        activeRobotID = choice.robotID
        activeSceneID = choice.sceneID
        if robotChanged, let policySelector {
            configurePolicySelector(
                policySelector,
                for: choice.robotID,
                preferred: nil
            )
        }
        configureEnvironmentSelector(
            for: choice.robotID,
            selecting: choice.sceneID
        )
        configureTaskSelector(
            for: choice.robotID,
            environmentID: choice.sceneID,
            selecting: choice.id
        )
        sceneSelector?.itemArray.first(where: {
            ($0.representedObject as? String) == choice.sceneID
        }).map { sceneSelector?.select($0) }
    }

    @objc private func togglePause() {
        paused.toggle()
        updatePresentationState()
        updateChrome(force: true)
    }

    @objc private func resetCameraAction() {
        resetCamera()
    }

    private func orbitCamera(dx: CGFloat, dy: CGFloat) {
        cameraYaw += Float(dx) * 0.006
        cameraPitch = min(1.2, max(-1.2,
            cameraPitch + Float(dy) * 0.006))
        publishCameraControl()
    }

    private func panCamera(dx: CGFloat, dy: CGFloat) {
        cameraPan.x -= Float(dx) * 0.0025
        cameraPan.y -= Float(dy) * 0.0025
        publishCameraControl()
    }

    private func dollyCamera(delta: CGFloat) {
        cameraDolly = min(1.2, max(-1.2,
            cameraDolly + Float(delta) * 0.012))
        publishCameraControl()
    }

    private func resetCamera() {
        cameraYaw = 0
        cameraPitch = 0
        cameraPan = .zero
        cameraDolly = 0
        publishCameraControl()
    }

    private func publishCameraControl() {
        let halfYaw = cameraYaw * 0.5
        let halfPitch = cameraPitch * 0.5
        let sy = sin(halfYaw)
        let cy = cos(halfYaw)
        let sx = sin(halfPitch)
        let cx = cos(halfPitch)
        let orientation = SIMD4<Float>(cy * sx, sy * cx, -sy * sx, cy * cx)
        let axis = SIMD3<Float>(
            orientation.x, orientation.y, orientation.z
        )
        let focus = SIMD3<Float>(0, 0, 1.55)
        let tangent = 2 * cross(axis, focus)
        let rotatedFocus = focus + orientation.w * tangent + cross(axis, tangent)
        let translation = focus - rotatedFocus + SIMD3<Float>(
            cameraPan.x, cameraPan.y, cameraDolly
        )
        onCameraChanged?(NumiWindowCameraControl(
            translation: translation,
            orientation: orientation
        ))
    }

    @objc private func requestLatestPolicy() {
        guard canReloadLatestPolicy else {
            return
        }
        statusSummary = "loading latest policy"
        updateChrome(force: true)
        onLatestPolicyRequested?()
    }

    @objc private func selectPolicy(_ sender: NSPopUpButton) {
        guard let url = sender.selectedItem?.representedObject as? URL else {
            return
        }
        statusSummary = "loading \(url.deletingPathExtension().lastPathComponent)"
        updateChrome(force: true)
        onPolicySelected?(url)
    }

    private func configurePauseItem(_ item: NSToolbarItem) {
        item.label = paused ? "Resume" : "Pause"
        item.toolTip = paused
            ? "Resume live preview (Space)"
            : "Pause preview rendering; simulation keeps running (Space)"
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

    private func configureTrainItem(_ item: NSToolbarItem) {
        let trainable = activeChoice?.isTrainable == true
        item.label = trainingActive ? "Training…" : "Train"
        item.paletteLabel = "Train Policy"
        item.toolTip = trainingActive
            ? "Training owns the Metal runtime; progress appears in the window title"
            : trainable
                ? "Configure and start policy training for this task (T)"
                : "This task does not publish a training contract yet"
        item.image = NSImage(
            systemSymbolName: trainingActive
                ? "brain.head.profile" : "play.circle.fill",
            accessibilityDescription: item.label
        )
        item.isEnabled = trainable && !trainingActive && !reconfiguring
    }

    private func configurePolicySelector(
        _ selector: NSPopUpButton,
        for robotID: String,
        preferred: URL?
    ) {
        selector.removeAllItems()
        let robotPolicies = policyChoices.filter { $0.robotID == robotID }
        if let initialPolicyURL,
           robotID == activeRobotID,
           !robotPolicies.contains(where: {
               $0.policyURL.standardizedFileURL ==
                   initialPolicyURL.standardizedFileURL
           }) {
            selector.addItem(
                withTitle: initialPolicyURL.deletingPathExtension()
                    .lastPathComponent
            )
            selector.lastItem?.representedObject = initialPolicyURL
            selector.lastItem?.toolTip = initialPolicyURL.path
        }
        for choice in robotPolicies {
            selector.addItem(withTitle: choice.displayName)
            selector.lastItem?.representedObject = choice.policyURL
            selector.lastItem?.toolTip = choice.policyURL.path
        }
        guard selector.numberOfItems > 0 else {
            selector.addItem(withTitle: "Untrained · no controller")
            selector.setAccessibilityHelp(
                "No qualified learned controller is loaded; a humanoid may fall until training succeeds"
            )
            selector.isEnabled = false
            return
        }
        selector.isEnabled = true
        if let preferred {
            let selected = preferred.standardizedFileURL
            if let item = selector.itemArray.first(where: {
                ($0.representedObject as? URL)?.standardizedFileURL == selected
            }) {
                selector.select(item)
                return
            }
        }
        selector.selectItem(at: 0)
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
        let frameSummary = hasPresentedFrame
            ? "env \(latestEnvironmentIndex) · frame \(latestFrameIndex) · dropped \(latestDroppedFrames)"
            : "waiting for a frame"
        window.title =
            "Numi Window — \(state) · \(statusSummary ?? frameSummary)"
        if force {
            if let item = pauseItem {
                configurePauseItem(item)
            }
            if let item = latestPolicyItem {
                configureLatestPolicyItem(item)
            }
            if let item = trainItem {
                configureTrainItem(item)
            }
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
        // The sensor renderer marks background pixels transparent. Keep that
        // useful distinction in the presentation window: a missing/empty
        // camera image is a calm window backdrop, never an opaque black
        // result that looks like a rendering failure. Shaded geometry keeps
        // the renderer's own color unchanged.
        return float4(mix(backdrop, shaded, clamp(sample.w, 0.0, 1.0)), 1.0);
    }
    """
}

// This bridge is intentionally sendable: the frame slot is exclusively owned
// by the delivery until the MTKView command-buffer completion releases it.
public final class NumiWindowBridge: @unchecked Sendable {
    private let window: NumiWindowController
    private let stateLock = NSLock()
    private var closed = false
    private var presentationEnabled = true
    private var pendingNativeInspectionEnabled: Bool?
    private var latestPolicyReloadRequested = false
    private var pendingPolicySelection: URL?
    private var pendingSceneSelection: String?
    private var pendingTrainingRequest: NumiWindowTrainingRequest?
    private var pendingCameraControl: NumiWindowCameraControl?
    private var selectedPolicyURL: URL?

    @MainActor
    private init(window: NumiWindowController) {
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
        window.onSceneSelected = { [weak self] id in
            self?.requestSceneSelection(id)
        }
        window.onTrainingRequested = { [weak self] request in
            self?.requestTraining(request)
        }
        window.onCameraChanged = { [weak self] control in
            self?.requestCameraControl(control)
        }
    }

    @MainActor
    public static func launch(
        canReloadLatestPolicy: Bool = false,
        policyChoices: [NumiWindowPolicyChoice] = [],
        initialPolicyURL: URL? = nil,
        sceneChoices: [NumiWindowSceneChoice] = [],
        initialSceneID: String? = nil
    ) throws -> NumiWindowBridge {
        precondition(Thread.isMainThread)
        let application = NSApplication.shared
        application.setActivationPolicy(.regular)
        let bridge = NumiWindowBridge(
            window: try NumiWindowController.make(
                canReloadLatestPolicy: canReloadLatestPolicy,
                policyChoices: policyChoices,
                initialPolicyURL: initialPolicyURL,
                sceneChoices: sceneChoices,
                initialSceneID: initialSceneID
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

    private func requestSceneSelection(_ id: String) {
        stateLock.lock()
        if !closed {
            pendingSceneSelection = id
        }
        stateLock.unlock()
    }

    private func requestTraining(_ request: NumiWindowTrainingRequest) {
        stateLock.lock()
        if !closed {
            pendingTrainingRequest = request
        }
        stateLock.unlock()
    }

    private func requestCameraControl(_ control: NumiWindowCameraControl) {
        stateLock.lock()
        if !closed {
            pendingCameraControl = control
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

    func takeCameraControl() -> NumiWindowCameraControl? {
        stateLock.lock()
        defer { stateLock.unlock() }
        let control = pendingCameraControl
        pendingCameraControl = nil
        return control
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

    func takeSceneSelection() -> String? {
        stateLock.lock()
        defer { stateLock.unlock() }
        let selection = pendingSceneSelection
        pendingSceneSelection = nil
        return selection
    }

    func takeTrainingRequest() -> NumiWindowTrainingRequest? {
        stateLock.lock()
        defer { stateLock.unlock() }
        let request = pendingTrainingRequest
        pendingTrainingRequest = nil
        return request
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

    func reportSceneRestored(_ id: String, summary: String) {
        DispatchQueue.main.async { [window] in
            window.showRestoredScene(id, summary: summary)
        }
    }

    func reportTrainingStatus(
        _ summary: String,
        active: Bool,
        policyURL: URL? = nil,
        artifactsURL: URL? = nil
    ) {
        if let policyURL {
            stateLock.lock()
            selectedPolicyURL = policyURL
            stateLock.unlock()
        }
        DispatchQueue.main.async { [window] in
            window.showTrainingStatus(
                summary,
                active: active,
                policyURL: policyURL,
                artifactsURL: artifactsURL
            )
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
        let delivery = WindowFrameDelivery(
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
