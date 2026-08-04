import RoomPlan
import SafariServices
import SwiftUI
import UIKit

struct RoomScannerView: UIViewControllerRepresentable {
    func makeUIViewController(context: Context) -> RoomScannerViewController {
        RoomScannerViewController()
    }

    func updateUIViewController(_ viewController: RoomScannerViewController, context: Context) {}
}

final class RoomScannerViewController: UIViewController, RoomCaptureViewDelegate {
    private let captureView = RoomCaptureView(frame: .zero)
    private let statusLabel = UILabel()
    private let privacyButton = UIButton(type: .system)
    private let stopButton = UIButton(type: .system)
    private let continueButton = UIButton(type: .system)
    private let sendButton = UIButton(type: .system)
    private let controlsCard = UIVisualEffectView(effect: UIBlurEffect(style: .systemUltraThinMaterial))

    private var capturedRooms: [CapturedRoom] = []
    private var isScanning = false
    private var isProcessing = false
    private var isExporting = false

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .clear
        view.isOpaque = false
        configureCaptureView()
        configureOverlay()
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        if !isScanning && capturedRooms.isEmpty {
            startScanning()
        }
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        guard isScanning else { return }
        isScanning = false
        captureView.captureSession.stop(pauseARSession: false)
        updateControls()
    }

    private func configureCaptureView() {
        captureView.translatesAutoresizingMaskIntoConstraints = false
        captureView.delegate = self
        captureView.isModelEnabled = true
        view.addSubview(captureView)
        NSLayoutConstraint.activate([
            captureView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            captureView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            captureView.topAnchor.constraint(equalTo: view.topAnchor),
            captureView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
    }

    private func configureOverlay() {
        statusLabel.text = "Move slowly through each room"
        statusLabel.font = .preferredFont(forTextStyle: .headline)
        statusLabel.textColor = .white
        statusLabel.textAlignment = .center
        statusLabel.numberOfLines = 0
        statusLabel.adjustsFontForContentSizeCategory = true
        statusLabel.layer.shadowColor = UIColor.black.cgColor
        statusLabel.layer.shadowOpacity = 0.8
        statusLabel.layer.shadowRadius = 3
        statusLabel.layer.shadowOffset = CGSize(width: 0, height: 1)
        statusLabel.accessibilityTraits = [.header, .updatesFrequently]
        view.addSubview(statusLabel)

        privacyButton.setImage(UIImage(systemName: "info.circle"), for: .normal)
        privacyButton.tintColor = .white
        privacyButton.accessibilityLabel = "Privacy information"
        privacyButton.accessibilityHint = "Explains how camera and LiDAR data is handled"
        privacyButton.addTarget(self, action: #selector(showPrivacyInformation), for: .touchUpInside)
        view.addSubview(privacyButton)

        configurePrimaryButton(stopButton, title: "Finish room", action: #selector(stopScanning))
        configureSecondaryButton(continueButton, title: "Continue scan", action: #selector(continueScanning))
        configurePrimaryButton(sendButton, title: "Send to Numi Lab", action: #selector(sendToNumiLab))

        let controls = UIStackView(arrangedSubviews: [stopButton, continueButton, sendButton])
        controls.axis = .vertical
        controls.spacing = 8
        controls.translatesAutoresizingMaskIntoConstraints = false
        controlsCard.translatesAutoresizingMaskIntoConstraints = false
        controlsCard.layer.cornerRadius = 22
        controlsCard.clipsToBounds = true
        controlsCard.contentView.addSubview(controls)
        view.addSubview(controlsCard)

        NSLayoutConstraint.activate([
            statusLabel.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 64),
            statusLabel.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -64),
            statusLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 18),
            privacyButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 12),
            privacyButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -16),
            privacyButton.widthAnchor.constraint(equalToConstant: 44),
            privacyButton.heightAnchor.constraint(equalToConstant: 44),
            controls.leadingAnchor.constraint(equalTo: controlsCard.contentView.leadingAnchor, constant: 12),
            controls.trailingAnchor.constraint(equalTo: controlsCard.contentView.trailingAnchor, constant: -12),
            controls.topAnchor.constraint(equalTo: controlsCard.contentView.topAnchor, constant: 12),
            controls.bottomAnchor.constraint(equalTo: controlsCard.contentView.bottomAnchor, constant: -12),
            controlsCard.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 20),
            controlsCard.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -20),
            controlsCard.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
            stopButton.heightAnchor.constraint(greaterThanOrEqualToConstant: 52),
            continueButton.heightAnchor.constraint(greaterThanOrEqualToConstant: 48),
            sendButton.heightAnchor.constraint(greaterThanOrEqualToConstant: 52)
        ])

        updateControls()
    }

    private func configurePrimaryButton(_ button: UIButton, title: String, action: Selector) {
        var configuration: UIButton.Configuration
        if #available(iOS 26.0, *) {
            configuration = .prominentClearGlass()
        } else {
            configuration = .tinted()
        }
        configuration.title = title
        configuration.buttonSize = .large
        configuration.cornerStyle = .capsule
        configuration.baseForegroundColor = .systemMint
        button.configuration = configuration
        button.titleLabel?.font = .preferredFont(forTextStyle: .headline)
        button.titleLabel?.adjustsFontForContentSizeCategory = true
        button.accessibilityLabel = title
        button.accessibilityHint = title == "Finish room"
            ? "Saves this room and prepares it for the property scan"
            : "Opens the standard share sheet for the property USDZ"
        button.addTarget(self, action: action, for: .touchUpInside)
    }

    private func configureSecondaryButton(_ button: UIButton, title: String, action: Selector) {
        var configuration: UIButton.Configuration
        if #available(iOS 26.0, *) {
            configuration = .clearGlass()
        } else {
            configuration = .borderless()
        }
        configuration.title = title
        configuration.buttonSize = .large
        configuration.baseForegroundColor = .label
        button.configuration = configuration
        button.titleLabel?.font = .preferredFont(forTextStyle: .body)
        button.titleLabel?.adjustsFontForContentSizeCategory = true
        button.accessibilityLabel = title
        button.accessibilityHint = "Starts another room scan"
        button.addTarget(self, action: action, for: .touchUpInside)
    }

    @objc private func startScanning() {
        guard RoomCaptureSession.isSupported else {
            statusLabel.text = "RoomPlan needs a LiDAR-capable iPhone or iPad."
            return
        }
        isScanning = true
        isProcessing = false
        statusLabel.text = capturedRooms.isEmpty
            ? "Move slowly through each room"
            : "Continue through the next room"
        updateControls()
        captureView.captureSession.run(configuration: RoomCaptureSession.Configuration())
    }

    @objc private func stopScanning() {
        guard isScanning else { return }
        isScanning = false
        isProcessing = true
        statusLabel.text = "Processing scan…"
        updateControls()
        captureView.captureSession.stop(pauseARSession: false)
    }

    @objc private func continueScanning() {
        guard !isScanning && !isProcessing && !isExporting else { return }
        startScanning()
    }

    @objc private func sendToNumiLab() {
        guard !capturedRooms.isEmpty, !isScanning, !isProcessing, !isExporting else { return }
        isExporting = true
        statusLabel.text = "Preparing property USDZ…"
        updateControls()

        let rooms = capturedRooms
        Task { @MainActor [weak self] in
            do {
                let url = try await Self.exportURL(for: rooms)
                guard let self else { return }
                self.isExporting = false
                self.statusLabel.text = self.summary(for: rooms)
                self.updateControls()

                let share = UIActivityViewController(activityItems: [url], applicationActivities: nil)
                if let popover = share.popoverPresentationController {
                    popover.sourceView = self.sendButton
                    popover.sourceRect = self.sendButton.bounds
                }
                self.present(share, animated: true)
            } catch {
                guard let self else { return }
                self.isExporting = false
                self.statusLabel.text = self.summary(for: rooms)
                self.updateControls()
                self.presentExportError(error)
            }
        }
    }

    func captureView(shouldPresent roomDataForProcessing: CapturedRoomData, error: Error?) -> Bool {
        guard let error else { return true }
        isProcessing = false
        statusLabel.text = "Room could not be processed. Try again."
        updateControls()
        presentProcessingError(error)
        return false
    }

    func captureView(didPresent room: CapturedRoom, error: Error?) {
        isProcessing = false
        if let error {
            statusLabel.text = "Room could not be processed. Try again."
            updateControls()
            presentProcessingError(error)
            return
        }
        capturedRooms.append(room)
        statusLabel.text = summary(for: capturedRooms)
        updateControls()
    }

    private func updateControls() {
        stopButton.isHidden = !isScanning
        let hasRooms = !capturedRooms.isEmpty
        continueButton.isHidden = isScanning || isProcessing || isExporting || !hasRooms
        sendButton.isHidden = isScanning || isProcessing || isExporting || !hasRooms
        controlsCard.isHidden = !isScanning && (isProcessing || isExporting || !hasRooms)
    }

    private func summary(for rooms: [CapturedRoom]) -> String {
        let wallCount = rooms.reduce(0) { $0 + $1.walls.count }
        let objectCount = rooms.reduce(0) { $0 + $1.objects.count }
        let openingCount = rooms.reduce(0) {
            $0 + $1.doors.count + $1.windows.count + $1.openings.count
        }
        let roomWord = rooms.count == 1 ? "room saved" : "rooms saved"
        return String(rooms.count) + " " + roomWord + "  ·  "
            + String(wallCount) + " walls  ·  " + String(objectCount) + " objects  ·  "
            + String(openingCount) + " openings"
    }

    @objc private func showPrivacyInformation() {
        let message = "RoomPlan uses the camera and LiDAR to process rooms on the device. Numi RoomPlan does not upload scans automatically. When you choose Send to Numi Lab, iOS opens the standard share sheet and you choose where the property USDZ file is sent."
        let alert = UIAlertController(title: "Privacy", message: message, preferredStyle: .alert)
        if let privacyPolicyURL {
            alert.addAction(UIAlertAction(title: "Privacy Policy", style: .default) { [weak self] _ in
                self?.present(SFSafariViewController(url: privacyPolicyURL), animated: true)
            })
        }
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }

    private var privacyPolicyURL: URL? {
        guard let value = Bundle.main.object(forInfoDictionaryKey: "PrivacyPolicyURL") as? String,
              let url = URL(string: value.trimmingCharacters(in: .whitespacesAndNewlines)),
              url.scheme?.lowercased() == "https",
              url.host != nil else {
            return nil
        }
        return url
    }

    private static func exportURL(for rooms: [CapturedRoom]) async throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("NumiProperty-\(UUID().uuidString).usdz")
        try? FileManager.default.removeItem(at: url)

        if rooms.count == 1, let room = rooms.first {
            try room.export(to: url, exportOptions: .parametric)
        } else {
            let structure = try await StructureBuilder(options: .beautifyObjects).capturedStructure(from: rooms)
            try structure.export(to: url, exportOptions: .parametric)
        }
        return url
    }

    private func presentExportError(_ error: Error) {
        let alert = UIAlertController(
            title: "Export failed",
            message: error.localizedDescription,
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }

    private func presentProcessingError(_ error: Error) {
        let alert = UIAlertController(
            title: "Room scan stopped",
            message: error.localizedDescription,
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "Try again", style: .default) { [weak self] _ in
            self?.startScanning()
        })
        alert.addAction(UIAlertAction(title: "Not now", style: .cancel))
        present(alert, animated: true)
    }

}
