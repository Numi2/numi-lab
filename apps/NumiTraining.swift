import AppKit
import Foundation

public struct NumiWindowTrainingRequest: Sendable {
    public let choiceID: String
    public let robotID: String
    public let robotName: String
    public let environmentName: String
    public let taskID: String
    public let taskName: String
    public let trainingArguments: [String]
    public let startingPolicyURL: URL?
    public let environments: Int
    public let stepsPerUpdate: Int
    public let updates: Int
    public let chunk: Int
    public let seed: UInt64
    public let checkpointInterval: Int
    public let minimumDifficultyBand: Int
    public let maximumDifficultyBand: Int

    public var transitionCount: UInt64 {
        UInt64(environments) * UInt64(stepsPerUpdate) * UInt64(updates)
    }
}

@MainActor
final class NumiTrainingSetupController: NSWindowController,
    NSWindowDelegate
{
    private struct Profile {
        let title: String
        let detail: String
        let environments: Int
        let steps: Int
        let updates: Int
        let chunk: Int
        let checkpointInterval: Int
        let minimumDifficultyBand: Int
        let maximumDifficultyBand: Int
    }

    private static let profiles = [
        Profile(
            title: "Pipeline check",
            detail: "Checks the complete workflow; this is too short to teach a useful controller.",
            environments: 64,
            steps: 32,
            updates: 2,
            chunk: 8,
            checkpointInterval: 1,
            minimumDifficultyBand: 0,
            maximumDifficultyBand: 0
        ),
        Profile(
            title: "Learn standing · recommended",
            detail: "Starts with the easiest curriculum and gives balance time to emerge.",
            environments: 256,
            steps: 128,
            updates: 16,
            chunk: 8,
            checkpointInterval: 4,
            minimumDifficultyBand: 0,
            maximumDifficultyBand: 0
        ),
        Profile(
            title: "Build robust movement",
            detail: "Expands a standing policy into a wider, harder curriculum.",
            environments: 1024,
            steps: 128,
            updates: 48,
            chunk: 8,
            checkpointInterval: 8,
            minimumDifficultyBand: 0,
            maximumDifficultyBand: 2
        ),
    ]

    private let choice: NumiWindowSceneChoice
    private let selectedPolicyURL: URL?
    private let onStart: @MainActor (NumiWindowTrainingRequest) -> Void
    private let profileSelector = NSPopUpButton()
    private let profileDetail = NSTextField(wrappingLabelWithString: "")
    private let startingPolicySelector = NSPopUpButton()
    private let environmentsField = NSTextField()
    private let stepsField = NSTextField()
    private let updatesField = NSTextField()
    private let chunkField = NSTextField()
    private let seedField = NSTextField()
    private let checkpointField = NSTextField()
    private let advancedGrid = NSGridView()
    private let summaryLabel = NSTextField(wrappingLabelWithString: "")
    private let validationLabel = NSTextField(wrappingLabelWithString: "")
    private var startButton: NSButton!
    var onDismiss: (@MainActor () -> Void)?

    init(
        choice: NumiWindowSceneChoice,
        selectedPolicyURL: URL?,
        onStart: @escaping @MainActor (NumiWindowTrainingRequest) -> Void
    ) {
        self.choice = choice
        self.selectedPolicyURL = selectedPolicyURL
        self.onStart = onStart
        let panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 610, height: 480),
            styleMask: [.titled],
            backing: .buffered,
            defer: false
        )
        panel.title = "Configure Training"
        panel.isReleasedWhenClosed = false
        super.init(window: panel)
        panel.delegate = self
        buildInterface(in: panel)
        profileSelector.selectItem(at: 1)
        applyProfile(index: 1)
    }

    required init?(coder: NSCoder) { nil }

    func present(over parent: NSWindow) {
        guard let window else { return }
        parent.beginSheet(window)
    }

    private func buildInterface(in panel: NSPanel) {
        let title = NSTextField(labelWithString: "Teach \(choice.robotName)")
        title.font = .systemFont(ofSize: 24, weight: .semibold)
        let context = NSTextField(
            wrappingLabelWithString:
                "\(choice.taskName)  ·  \(choice.sceneName)"
        )
        context.font = .systemFont(ofSize: 14, weight: .medium)
        context.textColor = .secondaryLabelColor

        let explanation = NSTextField(
            wrappingLabelWithString:
                "Numi will run native Metal simulation, learn with MLX, save checkpoints, then physically evaluate the result. Falling policies stay as experiments and are not loaded as learned behavior."
        )
        explanation.textColor = .secondaryLabelColor

        for profile in Self.profiles {
            profileSelector.addItem(withTitle: profile.title)
        }
        profileSelector.target = self
        profileSelector.action = #selector(selectProfile)
        profileSelector.setAccessibilityLabel("Training run size")
        profileDetail.textColor = .secondaryLabelColor

        startingPolicySelector.addItem(withTitle: "New policy · recommended for learning")
        startingPolicySelector.lastItem?.representedObject = "new"
        if let selectedPolicyURL {
            startingPolicySelector.addItem(
                withTitle: "Continue \(selectedPolicyURL.deletingPathExtension().lastPathComponent)"
            )
            startingPolicySelector.lastItem?.representedObject = selectedPolicyURL
        }
        startingPolicySelector.setAccessibilityLabel("Starting policy")

        let advancedToggle = NSButton(
            checkboxWithTitle: "Customize advanced run settings",
            target: self,
            action: #selector(toggleAdvanced)
        )
        advancedToggle.setAccessibilityHelp(
            "Override environments, rollout length, updates, submission chunk, seed, and checkpoint cadence"
        )

        func configureNumberField(
            _ field: NSTextField,
            label: String
        ) {
            field.alignment = .right
            field.font = .monospacedDigitSystemFont(ofSize: 13, weight: .regular)
            field.setAccessibilityLabel(label)
            field.target = self
            field.action = #selector(updateSummary)
        }
        configureNumberField(environmentsField, label: "Parallel environments")
        configureNumberField(stepsField, label: "Steps per update")
        configureNumberField(updatesField, label: "Learning updates")
        configureNumberField(chunkField, label: "Control steps per GPU submission")
        configureNumberField(seedField, label: "Reproducible seed")
        configureNumberField(checkpointField, label: "Checkpoint interval")
        let seedFormatter = DateFormatter()
        seedFormatter.locale = Locale(identifier: "en_US_POSIX")
        seedFormatter.dateFormat = "yyyyMMdd"
        seedField.stringValue = seedFormatter.string(from: Date())

        advancedGrid.rowSpacing = 8
        advancedGrid.columnSpacing = 16
        let rows: [(String, NSTextField, String)] = [
            ("Parallel environments", environmentsField,
             "More worlds collect experience in parallel."),
            ("Steps per update", stepsField,
             "Experience gathered before each learner update."),
            ("Learning updates", updatesField,
             "How many times the policy is improved."),
            ("GPU submission chunk", chunkField,
             "Keep 8 unless profiling shows a better cadence."),
            ("Seed", seedField,
             "The same seed makes the run reproducible."),
            ("Checkpoint every", checkpointField,
             "Learning updates between recoverable snapshots."),
        ]
        for (name, field, help) in rows {
            let label = NSTextField(labelWithString: name)
            label.alignment = .right
            let helpLabel = NSTextField(wrappingLabelWithString: help)
            helpLabel.textColor = .tertiaryLabelColor
            helpLabel.font = .systemFont(ofSize: 11)
            advancedGrid.addRow(with: [label, field, helpLabel])
        }
        advancedGrid.column(at: 0).xPlacement = .trailing
        advancedGrid.column(at: 1).width = 92
        advancedGrid.column(at: 2).width = 255
        advancedGrid.isHidden = true

        summaryLabel.font = .systemFont(ofSize: 13, weight: .semibold)
        summaryLabel.textColor = .labelColor
        validationLabel.textColor = .systemRed
        validationLabel.font = .systemFont(ofSize: 12)

        let cancel = NSButton(title: "Cancel", target: self, action: #selector(cancel))
        cancel.keyEquivalent = "\u{1b}"
        startButton = NSButton(
            title: "Start Training",
            target: self,
            action: #selector(startTraining)
        )
        startButton.keyEquivalent = "\r"
        startButton.bezelStyle = .rounded
        startButton.bezelColor = .systemGreen
        startButton.image = NSImage(
            systemSymbolName: "play.fill",
            accessibilityDescription: "Start Training"
        )
        startButton.imagePosition = .imageLeading
        let buttons = NSStackView(views: [cancel, startButton])
        buttons.orientation = .horizontal
        buttons.alignment = .centerY
        buttons.spacing = 10

        let profileLabel = NSTextField(labelWithString: "Run size")
        profileLabel.font = .systemFont(ofSize: 12, weight: .semibold)
        let policyLabel = NSTextField(labelWithString: "Starting point")
        policyLabel.font = .systemFont(ofSize: 12, weight: .semibold)

        let stack = NSStackView(views: [
            title,
            context,
            explanation,
            profileLabel,
            profileSelector,
            profileDetail,
            policyLabel,
            startingPolicySelector,
            advancedToggle,
            advancedGrid,
            summaryLabel,
            validationLabel,
            buttons,
        ])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 10
        stack.setCustomSpacing(4, after: title)
        stack.setCustomSpacing(18, after: explanation)
        stack.setCustomSpacing(4, after: profileSelector)
        stack.setCustomSpacing(16, after: profileDetail)
        stack.setCustomSpacing(16, after: startingPolicySelector)
        stack.translatesAutoresizingMaskIntoConstraints = false
        panel.contentView = NSView()
        panel.contentView?.addSubview(stack)
        buttons.alignment = .trailing
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: panel.contentView!.leadingAnchor, constant: 28),
            stack.trailingAnchor.constraint(equalTo: panel.contentView!.trailingAnchor, constant: -28),
            stack.topAnchor.constraint(equalTo: panel.contentView!.topAnchor, constant: 24),
            profileSelector.widthAnchor.constraint(equalToConstant: 330),
            startingPolicySelector.widthAnchor.constraint(equalToConstant: 420),
            summaryLabel.widthAnchor.constraint(equalToConstant: 540),
            validationLabel.widthAnchor.constraint(equalToConstant: 540),
            buttons.trailingAnchor.constraint(equalTo: stack.trailingAnchor),
        ])
    }

    @objc private func selectProfile() {
        applyProfile(index: max(0, profileSelector.indexOfSelectedItem))
    }

    private func applyProfile(index: Int) {
        let profile = Self.profiles[index]
        environmentsField.integerValue = profile.environments
        stepsField.integerValue = profile.steps
        updatesField.integerValue = profile.updates
        chunkField.integerValue = profile.chunk
        checkpointField.integerValue = profile.checkpointInterval
        profileDetail.stringValue = profile.detail
        updateSummary()
    }

    @objc private func toggleAdvanced(_ sender: NSButton) {
        advancedGrid.isHidden = sender.state != .on
        window?.setContentSize(NSSize(
            width: 610,
            height: sender.state == .on ? 720 : 480
        ))
    }

    @objc private func updateSummary() {
        validationLabel.stringValue = ""
        let profile = Self.profiles[
            max(0, profileSelector.indexOfSelectedItem)
        ]
        guard let values = validatedValues(showError: false) else {
            summaryLabel.stringValue = "Enter positive whole numbers to configure this run."
            return
        }
        let transitions = UInt64(values.environments) *
            UInt64(values.steps) * UInt64(values.updates)
        let formatted = NumberFormatter.localizedString(
            from: NSNumber(value: transitions),
            number: .decimal
        )
        summaryLabel.stringValue =
            "\(formatted) simulated control transitions  ·  difficulty \(profile.minimumDifficultyBand)–\(profile.maximumDifficultyBand)  ·  checkpoints every \(values.checkpoint) updates"
    }

    private func validatedValues(
        showError: Bool
    ) -> (environments: Int, steps: Int, updates: Int, chunk: Int, seed: UInt64, checkpoint: Int)? {
        let environments = environmentsField.integerValue
        let steps = stepsField.integerValue
        let updates = updatesField.integerValue
        let chunk = chunkField.integerValue
        let checkpoint = checkpointField.integerValue
        guard let seed = UInt64(seedField.stringValue),
              environments > 0, steps > 0, updates > 0,
              chunk > 0, chunk <= steps,
              checkpoint > 0, checkpoint <= updates,
              UInt64(environments) <= UInt64.max / UInt64(steps),
              UInt64(environments) * UInt64(steps) <=
                UInt64.max / UInt64(updates)
        else {
            if showError {
                validationLabel.stringValue =
                    "Use positive whole numbers; chunk cannot exceed steps and checkpoint cannot exceed updates."
            }
            return nil
        }
        return (environments, steps, updates, chunk, seed, checkpoint)
    }

    @objc private func startTraining() {
        guard let values = validatedValues(showError: true) else { return }
        let profile = Self.profiles[
            max(0, profileSelector.indexOfSelectedItem)
        ]
        let startingPolicy = startingPolicySelector.selectedItem?
            .representedObject as? URL
        let request = NumiWindowTrainingRequest(
            choiceID: choice.id,
            robotID: choice.robotID,
            robotName: choice.robotName,
            environmentName: choice.sceneName,
            taskID: choice.taskID,
            taskName: choice.taskName,
            trainingArguments: choice.trainingArguments,
            startingPolicyURL: startingPolicy,
            environments: values.environments,
            stepsPerUpdate: values.steps,
            updates: values.updates,
            chunk: values.chunk,
            seed: values.seed,
            checkpointInterval: values.checkpoint,
            minimumDifficultyBand: profile.minimumDifficultyBand,
            maximumDifficultyBand: profile.maximumDifficultyBand
        )
        dismiss()
        onStart(request)
    }

    @objc private func cancel() { dismiss() }

    private func dismiss() {
        guard let window else { return }
        window.sheetParent?.endSheet(window)
        onDismiss?()
    }

    func windowWillClose(_ notification: Notification) { onDismiss?() }
}
