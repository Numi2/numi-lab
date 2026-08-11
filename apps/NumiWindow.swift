import AppKit
import Darwin
import Metal
import MetalKit
import simd

public struct NumiWindowPolicyContract: Sendable, Equatable {
    public let version: UInt64
    public let worldFingerprint: UInt64
    public let taskFingerprint: UInt64
    public let observationFingerprint: UInt64
    public let actionFingerprint: UInt64

    public var isExact: Bool {
        version != 0 && worldFingerprint != 0 && taskFingerprint != 0 &&
            observationFingerprint != 0 && actionFingerprint != 0
    }
}

public struct NumiWindowPolicyMetadata: Sendable {
    public let id: String
    public let revision: UInt64
    public let contract: NumiWindowPolicyContract
}

func numiWindowErrorSummary(_ error: Error) -> String {
    let collapsed = String(describing: error)
        .split(whereSeparator: { $0.isWhitespace })
        .joined(separator: " ")
    return String(collapsed.prefix(180))
}

public struct NumiWindowPolicyChoice: Sendable {
    public let robotID: String
    public let displayName: String
    public let policyURL: URL
    public let metadata: NumiWindowPolicyMetadata
    public let modificationDate: Date

    public init(
        robotID: String,
        displayName: String,
        policyURL: URL,
        metadata: NumiWindowPolicyMetadata,
        modificationDate: Date
    ) {
        self.robotID = robotID
        self.displayName = displayName
        self.policyURL = policyURL
        self.metadata = metadata
        self.modificationDate = modificationDate
    }
}

private extension Data {
    func numiLittleEndianInteger<T: FixedWidthInteger>(
        at offset: Int,
        as: T.Type = T.self
    ) -> T? {
        guard offset >= 0, count - offset >= MemoryLayout<T>.size else {
            return nil
        }
        var value: T = 0
        _ = withUnsafeBytes { bytes in
            memcpy(
                &value,
                bytes.baseAddress!.advanced(by: offset),
                MemoryLayout<T>.size
            )
        }
        return T(littleEndian: value)
    }
}

// Policy contracts are stored at the end of the deterministic v4 wire payload.
// Catalog discovery reads only the fixed header, identity prefix, and contract
// tail. It does not map or deserialize megabytes of weights merely to populate
// a menu; the native loader still performs the full content-hash and semantic
// validation before a selected policy can become executable.
public func numiWindowPolicyMetadata(
    at url: URL
) -> NumiWindowPolicyMetadata? {
    let headerBytes = 32
    let contractBytes = 5 * MemoryLayout<UInt64>.size
    let descriptor = url.withUnsafeFileSystemRepresentation { path in
        guard let path else { return Int32(-1) }
        return Darwin.open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK)
    }
    guard descriptor >= 0 else { return nil }
    defer { Darwin.close(descriptor) }
    var status = stat()
    guard fstat(descriptor, &status) == 0,
          status.st_mode & S_IFMT == S_IFREG,
          status.st_size >= headerBytes + 8 + 1 + 8 + contractBytes,
          status.st_size <= Int64.max
    else { return nil }
    let fileSize = Int(status.st_size)

    func read(_ count: Int, at offset: Int) -> Data? {
        guard count >= 0, offset >= 0, fileSize - offset >= count else {
            return nil
        }
        var data = Data(count: count)
        let amount = data.withUnsafeMutableBytes { bytes in
            pread(descriptor, bytes.baseAddress, count, off_t(offset))
        }
        return amount == count ? data : nil
    }

    guard let header = read(headerBytes, at: 0),
              String(decoding: header.prefix(7), as: UTF8.self) == "MRLEARN",
              header.numiLittleEndianInteger(at: 8, as: UInt32.self) == 4,
              header.numiLittleEndianInteger(at: 12, as: UInt32.self) == 2,
              let payloadBytes = header.numiLittleEndianInteger(
                  at: 16,
                  as: UInt64.self
              ),
              payloadBytes == UInt64(fileSize - headerBytes),
          let countData = read(8, at: headerBytes),
              let idCount = countData.numiLittleEndianInteger(
                  at: 0,
                  as: UInt64.self
              ),
              idCount > 0,
              idCount <= 16_384,
              idCount + 8 + UInt64(contractBytes) <= payloadBytes,
          let idData = read(Int(idCount), at: headerBytes + 8),
              let id = String(data: idData, encoding: .utf8),
              !id.isEmpty,
          let revisionData = read(
              8,
              at: headerBytes + 8 + Int(idCount)
          ),
              let revision = revisionData.numiLittleEndianInteger(
                  at: 0,
                  as: UInt64.self
              ),
              revision != 0
    else { return nil }
    guard let contractData = read(
              contractBytes,
              at: fileSize - contractBytes
          ),
              let version = contractData.numiLittleEndianInteger(
                  at: 0,
                  as: UInt64.self
              ),
              let world = contractData.numiLittleEndianInteger(
                  at: 8,
                  as: UInt64.self
              ),
              let task = contractData.numiLittleEndianInteger(
                  at: 16,
                  as: UInt64.self
              ),
              let observation = contractData.numiLittleEndianInteger(
                  at: 24,
                  as: UInt64.self
              ),
              let action = contractData.numiLittleEndianInteger(
                  at: 32,
                  as: UInt64.self
              )
    else { return nil }
    let contract = NumiWindowPolicyContract(
        version: version,
        worldFingerprint: world,
        taskFingerprint: task,
        observationFingerprint: observation,
        actionFingerprint: action
    )
    guard contract.isExact else { return nil }
    return NumiWindowPolicyMetadata(
        id: id,
        revision: revision,
        contract: contract
    )
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

private func numiWindowCatalogFiles(
    in directory: URL,
    matching predicate: (URL) -> Bool
) -> [URL] {
    let keys: Set<URLResourceKey> = [
        .isDirectoryKey,
        .isRegularFileKey,
        .contentModificationDateKey,
    ]
    guard let roots = try? FileManager.default.contentsOfDirectory(
        at: directory,
        includingPropertiesForKeys: Array(keys),
        options: []
    ) else {
        return []
    }
    var files: [URL] = []
    files.reserveCapacity(roots.count)
    for root in roots {
        let values = try? root.resourceValues(forKeys: keys)
        if values?.isRegularFile == true {
            if predicate(root) { files.append(root) }
            continue
        }
        guard values?.isDirectory == true,
              let children = try? FileManager.default.contentsOfDirectory(
                  at: root,
                  includingPropertiesForKeys: Array(keys),
                  options: []
              )
        else {
            continue
        }
        for child in children {
            let childValues = try? child.resourceValues(forKeys: keys)
            if childValues?.isRegularFile == true && predicate(child) {
                files.append(child)
            }
        }
    }
    return files
}

public func numiWindowSceneChoices(in directory: URL) -> [NumiWindowSceneChoice] {
    // Catalog entries are published either at the root or in one run folder.
    // Do not recursively walk checkpoints, rollout packs, captures, or other
    // multi-gigabyte run evidence just to build the toolbar.
    let entries = numiWindowCatalogFiles(in: directory) {
        $0.lastPathComponent.hasSuffix(".numi-window.json")
    }
    let decoder = JSONDecoder()
    var choices: [NumiWindowSceneChoice] = []
    var seen: Set<String> = []
    for url in entries {
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
    let entries = numiWindowCatalogFiles(in: directory) {
        $0.pathExtension == "policypack"
    }
    var choices: [NumiWindowPolicyChoice] = []
    for url in entries {
        let relative = url.path.replacingOccurrences(
            of: directory.path + "/",
            with: ""
        )
        let components = relative.split(separator: "/")
        let robotID = components.count > 1
            ? String(components[0])
            : "Uncategorized"
        guard let metadata = numiWindowPolicyMetadata(at: url) else {
            continue
        }
        let modificationDate = (try? url.resourceValues(
            forKeys: [.contentModificationDateKey]
        ).contentModificationDate) ?? .distantPast
        choices.append(NumiWindowPolicyChoice(
            robotID: robotID,
            displayName: url.deletingPathExtension().lastPathComponent,
            policyURL: url,
            metadata: metadata,
            modificationDate: modificationDate
        ))
    }
    return choices.sorted {
        let robotOrder = $0.robotID.localizedStandardCompare($1.robotID)
        guard robotOrder == .orderedSame else {
            return robotOrder == .orderedAscending
        }
        if $0.modificationDate != $1.modificationDate {
            return $0.modificationDate > $1.modificationDate
        }
        return $0.displayName.localizedStandardCompare($1.displayName) ==
            .orderedAscending
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
    for scene in sceneChoices {
        guard let option = scene.launchArguments.firstIndex(of: "--policy-pack"),
              option + 1 < scene.launchArguments.count
        else {
            continue
        }
        let url = URL(fileURLWithPath: scene.launchArguments[option + 1])
            .standardizedFileURL
        guard FileManager.default.fileExists(atPath: url.path),
              seen.insert(url).inserted,
              let metadata = numiWindowPolicyMetadata(at: url)
        else {
            continue
        }
        let modificationDate = (try? url.resourceValues(
            forKeys: [.contentModificationDateKey]
        ).contentModificationDate) ?? .distantPast
        choices.append(NumiWindowPolicyChoice(
            robotID: scene.robotID,
            displayName: url.deletingPathExtension().lastPathComponent,
            policyURL: url,
            metadata: metadata,
            modificationDate: modificationDate
        ))
    }
    // Training publishes qualified deployment candidates under .numi/runs;
    // requiring a second manual copy into .numi/policies made the newest
    // learned behavior disappear from the Window. Keep this bounded to each
    // run's final deployment artifact so checkpoints do not flood the menu.
    if let runEntries = try? FileManager.default.contentsOfDirectory(
        at: sceneDirectory,
        includingPropertiesForKeys: [
            .isDirectoryKey,
            .contentModificationDateKey,
        ],
        // sceneDirectory normally lives below `.numi`. Foundation treats
        // that hidden ancestor as a reason to return an empty direct listing
        // when `.skipsHiddenFiles` is set, even though the run folders
        // themselves are visible.
        options: []
    ) {
        // Deployment history can contain hundreds of heavyweight runs. The
        // selector is an interactive control, not an archive browser: inspect
        // only the newest direct run folders while scene-owned and promoted
        // policies above remain unconditional.
        let recentRuns = runEntries.compactMap { url -> (URL, Date)? in
            guard let values = try? url.resourceValues(forKeys: [
                .isDirectoryKey,
                .contentModificationDateKey,
            ]), values.isDirectory == true else { return nil }
            return (url, values.contentModificationDate ?? .distantPast)
        }.sorted { $0.1 > $1.1 }.prefix(32)
        for (runURL, _) in recentRuns {
            for filename in [
                "deployment.policypack",
                "candidate.deployment.policypack",
            ] {
                let url = runURL.appendingPathComponent(filename)
                    .standardizedFileURL
                guard FileManager.default.fileExists(atPath: url.path),
                      seen.insert(url).inserted,
                      let metadata = numiWindowPolicyMetadata(at: url)
                else { continue }
                let modificationDate = (try? url.resourceValues(
                    forKeys: [.contentModificationDateKey]
                ).contentModificationDate) ?? .distantPast
                choices.append(NumiWindowPolicyChoice(
                    // Run names are human labels, not ownership contracts.
                    // Exact live fingerprints below determine compatibility.
                    robotID: "",
                    displayName: runURL.lastPathComponent,
                    policyURL: url,
                    metadata: metadata,
                    modificationDate: modificationDate
                ))
                break
            }
        }
    }
    return choices.sorted {
        let robotOrder = $0.robotID.localizedStandardCompare($1.robotID)
        guard robotOrder == .orderedSame else {
            return robotOrder == .orderedAscending
        }
        if $0.modificationDate != $1.modificationDate {
            return $0.modificationDate > $1.modificationDate
        }
        return $0.displayName.localizedStandardCompare($1.displayName) ==
            .orderedAscending
    }
}

public func numiWindowCompatiblePolicyChoices(
    _ choices: [NumiWindowPolicyChoice],
    robotID: String,
    contract: NumiWindowPolicyContract?
) -> [NumiWindowPolicyChoice] {
    choices.filter { choice in
        if let contract {
            return contract.isExact && choice.metadata.contract == contract
        }
        // Robot labels are a presentation fallback only while the first live
        // world contract is not yet known. Executable selection is fingerprint
        // exact once the first native context has presented.
        return choice.robotID == robotID
    }
}

private final class WindowFrameDelivery: @unchecked Sendable {
    let frame: MetalRoboTaskInspectionFrame
    let release: @Sendable () -> Void
    let complete: @Sendable (Bool) -> Void

    init(
        frame: MetalRoboTaskInspectionFrame,
        release: @escaping @Sendable () -> Void,
        complete: @escaping @Sendable (Bool) -> Void
    ) {
        self.frame = frame
        self.release = release
        self.complete = complete
    }

    func discard() {
        release()
        complete(false)
    }

    func finish(_ succeeded: Bool) {
        release()
        complete(succeeded)
    }
}

struct NumiWindowPolicySelection: Sendable {
    let choiceID: String
    let policyURL: URL?
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
    case reconfigure(NumiWindowPolicySelection)
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
    private let loadingLabel = NSTextField(
        labelWithString: "Preparing the first simulation frame…"
    )
    private var trainingArtifactsURL: URL?
    private let coachCard = NSVisualEffectView()
    private let coachTitle = NSTextField(labelWithString: "Your first robot policy")
    private let coachDetail = NSTextField(wrappingLabelWithString: "")
    private var coachDismissed = false
    private let canReloadLatestPolicy: Bool
    private let policyChoices: [NumiWindowPolicyChoice]
    private var installedPolicyURL: URL?
    private var installedPolicyRobotID: String?
    private var activePolicyContract: NumiWindowPolicyContract?
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
    var onSimulationPausedChanged: ((Bool) -> Void)?
    var onLatestPolicyRequested: (() -> Void)?
    var onPolicySelected: ((NumiWindowPolicySelection) -> Void)?
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
        self.sceneChoices = sceneChoices
        self.initialSceneID = initialSceneID
        let startingRobotID = sceneChoices.first(where: {
            $0.id == initialSceneID
        })?.robotID ?? sceneChoices.first?.robotID ?? ""
        self.activeRobotID = startingRobotID
        self.installedPolicyURL = initialPolicyURL
        self.installedPolicyRobotID = initialPolicyURL == nil
            ? nil : startingRobotID
        self.activeSceneID = sceneChoices.first(where: {
            $0.id == initialSceneID
        })?.sceneID ?? sceneChoices.first?.sceneID ?? ""
        // Scene controls become valid only after the initial runtime has
        // produced a frame. Switching during native context construction used
        // to strand the window in a blank, apparently frozen rebuild.
        self.reconfiguring = true
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
        configureLoadingLabel()
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
        if reconfiguring {
            setConfigurationControlsEnabled(false)
        }
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
            delivery.discard()
            return
        }
        pending?.discard()
        pending = delivery
        updateDrawableSize(
            sourceWidth: delivery.frame.width,
            sourceHeight: delivery.frame.height
        )
        latestEnvironmentIndex = delivery.frame.environmentIndex
        latestFrameIndex = delivery.frame.frameIndex
        latestDroppedFrames = delivery.frame.droppedFrames
        updateChrome()
        view.setNeedsDisplay(view.bounds)
    }

    private func didPresent(frameIndex: UInt64) {
        guard !closed else { return }
        hasPresentedFrame = true
        loadingLabel.isHidden = true
        statusSummary = nil
        finishReconfiguration()
        updateChrome(force: true)
    }

    func showPolicyStatus(_ summary: String) {
        guard !closed else {
            return
        }
        statusSummary = summary
        updateChrome(force: true)
    }

    func closeWindow() {
        window.performClose(nil)
    }

    func showInstalledPolicy(_ url: URL, revision: UInt64) {
        guard !closed else {
            return
        }
        installedPolicyURL = url
        installedPolicyRobotID = activeRobotID
        selectDisplayedPolicy(url)
        statusSummary =
            "\(url.deletingPathExtension().lastPathComponent) · revision \(revision)"
        updateCoachCard()
        updateChrome(force: true)
    }

    func showRejectedPolicy(activePolicyURL: URL?, reason: String) {
        guard !closed else {
            return
        }
        selectDisplayedPolicy(activePolicyURL)
        statusSummary = "policy rejected · \(reason) · unchanged"
        updateChrome(force: true)
    }

    func showConfigurationInstalled(
        _ id: String,
        policyURL: URL?,
        contract: NumiWindowPolicyContract
    ) {
        guard !closed else { return }
        selectDisplayedScene(id)
        activePolicyContract = contract
        installedPolicyURL = policyURL
        installedPolicyRobotID = policyURL == nil ? nil : activeRobotID
        if let policySelector {
            configurePolicySelector(
                policySelector,
                for: activeRobotID,
                preferred: policyURL
            )
        }
        statusSummary = policyURL == nil
            ? "simulation ready · no controller"
            : "simulation ready · learned controller"
        updateCoachCard()
        updateChrome(force: true)
    }

    func showConfigurationRestored(
        _ id: String,
        policyURL: URL?,
        contract: NumiWindowPolicyContract?,
        summary: String
    ) {
        guard !closed else { return }
        reconfiguring = false
        selectDisplayedScene(id)
        activePolicyContract = contract
        installedPolicyURL = policyURL
        installedPolicyRobotID = policyURL == nil ? nil : activeRobotID
        if let policySelector {
            configurePolicySelector(
                policySelector,
                for: activeRobotID,
                preferred: policyURL
            )
        }
        loadingLabel.isHidden = true
        statusSummary = summary
        setConfigurationControlsEnabled(!trainingActive)
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

    private func configureLoadingLabel() {
        loadingLabel.font = .systemFont(ofSize: 15, weight: .semibold)
        loadingLabel.textColor = .secondaryLabelColor
        loadingLabel.alignment = .center
        loadingLabel.translatesAutoresizingMaskIntoConstraints = false
        loadingLabel.isHidden = !reconfiguring
        view.addSubview(loadingLabel)
        NSLayoutConstraint.activate([
            loadingLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            loadingLabel.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            loadingLabel.leadingAnchor.constraint(
                greaterThanOrEqualTo: view.leadingAnchor,
                constant: 32
            ),
            loadingLabel.trailingAnchor.constraint(
                lessThanOrEqualTo: view.trailingAnchor,
                constant: -32
            ),
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
        command.addCompletedHandler { [weak self] completed in
            let succeeded = completed.status == .completed
            delivery.finish(succeeded)
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                if succeeded {
                    self.didPresent(frameIndex: delivery.frame.frameIndex)
                } else {
                    self.statusSummary = "display command failed"
                    self.updateChrome(force: true)
                }
            }
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
                preferred: installedPolicyURL
            )
            selector.setAccessibilityLabel("Policy")
            item.label = "Policy"
            item.toolTip = "Select a compatible saved PolicyPack"
            item.view = selector
            policySelector = selector
            selectDisplayedPolicy(installedPolicyURL)
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
        guard hasPresentedFrame, !reconfiguring,
              let robotID = sender.selectedItem?.representedObject as? String,
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
        guard hasPresentedFrame, !reconfiguring else { return }
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
        guard hasPresentedFrame, !reconfiguring,
              let id = sender.selectedItem?.representedObject as? String,
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
        loadingLabel.stringValue = "Preparing \(choice.robotName)…"
        // Keep the last successfully rendered frame visible during a later
        // robot rebuild. Only the first launch needs the loading canvas.
        loadingLabel.isHidden = hasPresentedFrame
        updateCoachCard()
        updateChrome(force: true)
        onSceneSelected?(choice.id)
    }

    private func finishReconfiguration() {
        guard reconfiguring else { return }
        // The old world has already unwound before the scheduler constructs the
        // new one, and this callback follows successful display completion.
        // No heuristic retirement delay or overlapping Metal world is needed.
        reconfiguring = false
        setConfigurationControlsEnabled(!trainingActive)
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
        onSimulationPausedChanged?(paused)
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
        guard hasPresentedFrame, !reconfiguring,
              let url = sender.selectedItem?.representedObject as? URL,
              let choice = activeChoice else {
            return
        }
        reconfiguring = true
        setConfigurationControlsEnabled(false)
        statusSummary = "loading \(url.deletingPathExtension().lastPathComponent)"
        loadingLabel.stringValue = "Preparing learned controller…"
        loadingLabel.isHidden = hasPresentedFrame
        updateChrome(force: true)
        onPolicySelected?(NumiWindowPolicySelection(
            choiceID: choice.id,
            policyURL: url
        ))
    }

    private func configurePauseItem(_ item: NSToolbarItem) {
        item.label = paused ? "Resume" : "Pause"
        item.toolTip = paused
            ? "Resume simulation and live preview (Space)"
            : "Pause simulation and hold the current frame (Space)"
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
        let robotPolicies = numiWindowCompatiblePolicyChoices(
            policyChoices,
            robotID: robotID,
            contract: activePolicyContract
        )
        let installedPolicyURL = installedPolicyRobotID == robotID
            ? self.installedPolicyURL : nil
        if installedPolicyURL == nil {
            selector.addItem(withTitle: "Untrained · no controller")
            selector.lastItem?.toolTip =
                "No learned controller is active in this simulation"
        }
        if let installedPolicyURL,
           !robotPolicies.contains(where: {
               $0.policyURL.standardizedFileURL ==
                   installedPolicyURL.standardizedFileURL
           }) {
            selector.addItem(
                withTitle: installedPolicyURL.deletingPathExtension()
                    .lastPathComponent
            )
            selector.lastItem?.representedObject = installedPolicyURL
            selector.lastItem?.toolTip = installedPolicyURL.path
        }
        for choice in robotPolicies {
            selector.addItem(withTitle: choice.displayName)
            selector.lastItem?.representedObject = choice.policyURL
            selector.lastItem?.toolTip = choice.policyURL.path
        }
        guard selector.itemArray.contains(where: {
            $0.representedObject is URL
        }) else {
            if selector.numberOfItems == 0 {
                selector.addItem(withTitle: "Untrained · no controller")
            }
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
            pending?.discard()
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
        pending?.discard()
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
    // One condition owns cross-thread state. Pause can therefore sleep the
    // scheduler without polling, holding a command buffer, or advancing the
    // Metal world behind a frozen presentation.
    private let stateCondition = NSCondition()
    private var closed = false
    private var presentationEnabled = true
    private var simulationPaused = false
    private var pendingNativeInspectionEnabled: Bool?
    private var latestPolicyReloadRequested = false
    private var pendingPolicySelection: NumiWindowPolicySelection?
    private var pendingSceneSelection: String?
    private var pendingTrainingRequest: NumiWindowTrainingRequest?
    private var pendingCameraControl: NumiWindowCameraControl?
    private var selectedPolicyURL: URL?
    private var completedPresentationCount: UInt64 = 0
    private var failedPresentationCount: UInt64 = 0
    private var outstandingDeliveries = 0

    @MainActor
    private init(window: NumiWindowController) {
        self.window = window
        window.onClose = { [weak self] in self?.markClosed() }
        window.onPresentationEnabledChanged = { [weak self] enabled in
            self?.setPresentationEnabled(enabled)
        }
        window.onSimulationPausedChanged = { [weak self] paused in
            self?.setSimulationPaused(paused)
        }
        window.onLatestPolicyRequested = { [weak self] in
            self?.requestLatestPolicyReload()
        }
        window.onPolicySelected = { [weak self] selection in
            self?.requestPolicySelection(selection)
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
        stateCondition.withLock { closed }
    }

    var acceptsFrames: Bool {
        stateCondition.withLock { !closed && presentationEnabled }
    }

    var isSimulationPaused: Bool {
        stateCondition.withLock { simulationPaused }
    }

    var presentationCount: UInt64 {
        stateCondition.withLock { completedPresentationCount }
    }

    private var hasPendingSchedulerWork: Bool {
        pendingNativeInspectionEnabled != nil ||
            latestPolicyReloadRequested ||
            pendingPolicySelection != nil ||
            pendingSceneSelection != nil ||
            pendingTrainingRequest != nil ||
            pendingCameraControl != nil
    }

    func waitWhileSimulationPaused() {
        stateCondition.lock()
        while simulationPaused && !closed && !hasPendingSchedulerWork {
            stateCondition.wait()
        }
        stateCondition.unlock()
    }

    func waitForPresentation(
        after count: UInt64,
        timeout: TimeInterval
    ) -> Bool {
        let deadline = Date(timeIntervalSinceNow: timeout)
        stateCondition.lock()
        let startingFailures = failedPresentationCount
        while completedPresentationCount <= count &&
                failedPresentationCount == startingFailures && !closed {
            if !stateCondition.wait(until: deadline) { break }
        }
        let presented = completedPresentationCount > count
        stateCondition.unlock()
        return presented
    }

    func waitForDeliveryDrain(timeout: TimeInterval) -> Bool {
        let deadline = Date(timeIntervalSinceNow: timeout)
        stateCondition.lock()
        // Close also drains: a drawable already committed to Metal still owns
        // its inspection slot even after AppKit has closed the window.
        while outstandingDeliveries != 0 {
            if !stateCondition.wait(until: deadline) { break }
        }
        let drained = outstandingDeliveries == 0
        stateCondition.unlock()
        return drained
    }

    private func markClosed() {
        stateCondition.lock()
        closed = true
        presentationEnabled = false
        simulationPaused = false
        pendingNativeInspectionEnabled = false
        stateCondition.broadcast()
        stateCondition.unlock()
    }

    private func setPresentationEnabled(_ enabled: Bool) {
        stateCondition.lock()
        let nextEnabled = enabled && !closed
        if presentationEnabled != nextEnabled {
            presentationEnabled = nextEnabled
            pendingNativeInspectionEnabled = nextEnabled
        }
        stateCondition.broadcast()
        stateCondition.unlock()
    }

    private func setSimulationPaused(_ paused: Bool) {
        stateCondition.lock()
        simulationPaused = paused && !closed
        stateCondition.broadcast()
        stateCondition.unlock()
    }

    private func requestLatestPolicyReload() {
        stateCondition.lock()
        if !closed { latestPolicyReloadRequested = true }
        stateCondition.broadcast()
        stateCondition.unlock()
    }

    private func requestPolicySelection(_ selection: NumiWindowPolicySelection) {
        stateCondition.lock()
        if !closed { pendingPolicySelection = selection }
        stateCondition.broadcast()
        stateCondition.unlock()
    }

    private func requestSceneSelection(_ id: String) {
        stateCondition.lock()
        if !closed { pendingSceneSelection = id }
        stateCondition.broadcast()
        stateCondition.unlock()
    }

    // Runtime integration probes use the same scheduler mailbox as AppKit.
    // They never mutate the native world or controller directly.
    func requestSceneSelectionForRuntimeCheck(_ id: String) {
        requestSceneSelection(id)
    }

    func requestPolicySelectionForRuntimeCheck(
        _ selection: NumiWindowPolicySelection
    ) {
        requestPolicySelection(selection)
    }

    func closeAfterRuntimeCheck() {
        DispatchQueue.main.async { [window] in
            window.closeWindow()
        }
    }

    private func requestTraining(_ request: NumiWindowTrainingRequest) {
        stateCondition.lock()
        if !closed { pendingTrainingRequest = request }
        stateCondition.broadcast()
        stateCondition.unlock()
    }

    private func requestCameraControl(_ control: NumiWindowCameraControl) {
        stateCondition.lock()
        if !closed { pendingCameraControl = control }
        stateCondition.broadcast()
        stateCondition.unlock()
    }

    private func setInitialPolicy(_ url: URL?) {
        stateCondition.withLock { selectedPolicyURL = url }
    }

    func takePendingNativeInspectionEnabled() -> Bool? {
        stateCondition.withLock {
            defer { pendingNativeInspectionEnabled = nil }
            return pendingNativeInspectionEnabled
        }
    }

    func takeCameraControl() -> NumiWindowCameraControl? {
        stateCondition.withLock {
            defer { pendingCameraControl = nil }
            return pendingCameraControl
        }
    }

    func takeLatestPolicyReloadRequest() -> Bool {
        stateCondition.withLock {
            defer { latestPolicyReloadRequested = false }
            return latestPolicyReloadRequested
        }
    }

    func takePolicySelection() -> NumiWindowPolicySelection? {
        stateCondition.withLock {
            defer { pendingPolicySelection = nil }
            return pendingPolicySelection
        }
    }

    func takeSceneSelection() -> String? {
        stateCondition.withLock {
            defer { pendingSceneSelection = nil }
            return pendingSceneSelection
        }
    }

    func takeTrainingRequest() -> NumiWindowTrainingRequest? {
        stateCondition.withLock {
            defer { pendingTrainingRequest = nil }
            return pendingTrainingRequest
        }
    }

    var selectedPolicy: URL? {
        stateCondition.withLock { selectedPolicyURL }
    }

    func reportPolicyInstalled(_ url: URL, revision: UInt64) {
        stateCondition.withLock { selectedPolicyURL = url }
        DispatchQueue.main.async { [window] in
            window.showInstalledPolicy(url, revision: revision)
        }
    }

    func reportPolicyRejected(_ error: Error) {
        let activePolicyURL = selectedPolicy
        let reason = numiWindowErrorSummary(error)
        DispatchQueue.main.async { [window] in
            window.showRejectedPolicy(
                activePolicyURL: activePolicyURL,
                reason: reason
            )
        }
    }

    func reportPolicyStatus(_ summary: String) {
        DispatchQueue.main.async { [window] in
            window.showPolicyStatus(summary)
        }
    }

    func reportConfigurationInstalled(
        _ id: String,
        policyURL: URL?,
        contract: NumiWindowPolicyContract
    ) {
        stateCondition.withLock { selectedPolicyURL = policyURL }
        DispatchQueue.main.async { [window] in
            window.showConfigurationInstalled(
                id,
                policyURL: policyURL,
                contract: contract
            )
        }
    }

    func reportConfigurationRestored(
        _ id: String,
        policyURL: URL?,
        contract: NumiWindowPolicyContract?,
        summary: String
    ) {
        stateCondition.withLock { selectedPolicyURL = policyURL }
        DispatchQueue.main.async { [window] in
            window.showConfigurationRestored(
                id,
                policyURL: policyURL,
                contract: contract,
                summary: summary
            )
        }
    }

    func reportTrainingStatus(
        _ summary: String,
        active: Bool,
        policyURL: URL? = nil,
        artifactsURL: URL? = nil
    ) {
        if let policyURL {
            stateCondition.withLock { selectedPolicyURL = policyURL }
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
        stateCondition.withLock { outstandingDeliveries += 1 }
        let delivery = WindowFrameDelivery(
            frame: frame,
            release: {
                context.releaseInspectionFrame(slotIndex: frame.slotIndex)
            },
            complete: { [weak self] succeeded in
                guard let self else { return }
                self.stateCondition.lock()
                if succeeded {
                    self.completedPresentationCount &+= 1
                } else {
                    self.failedPresentationCount &+= 1
                }
                self.outstandingDeliveries -= 1
                self.stateCondition.broadcast()
                self.stateCondition.unlock()
            }
        )
        DispatchQueue.main.async { [weak self, window] in
            guard self?.acceptsFrames == true else {
                delivery.discard()
                return
            }
            window.offer(delivery)
        }
    }
}
