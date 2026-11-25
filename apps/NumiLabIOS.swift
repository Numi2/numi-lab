import Foundation
import SwiftUI

private actor PendulumSimulation {
    private var engine: MetalRoboTaskRolloutContext?
    private var step = 0

    func reset() throws {
        if let engine {
            try engine.reset(seed: 0)
        } else {
            func asset(_ name: String, _ type: String) throws -> URL {
                guard let url = Bundle.main.url(forResource: name, withExtension: type) else {
                    throw CocoaError(.fileNoSuchFile, userInfo: [
                        NSLocalizedDescriptionKey: "The app is missing \(name).\(type).",
                    ])
                }
                return url
            }
            engine = try MetalRoboTaskRolloutContext(
                manifest: .init(
                    source: .worldPack(
                        world: asset("robot", "worldpack"),
                        taskPack: asset("robot", "taskpack"),
                        actuatorPack: asset("robot", "actuators"),
                        sensorPack: asset("robot", "sensors"),
                        realityPack: asset("robot", "reality")
                    ),
                    sensorsAndPhysics: .init(environmentCount: 1, disableTaskTerminations: true)
                ),
                metallibPath: asset("MetalRobo", "metallib").path
            )
        }
        step = 0
    }

    func advance(effort: Float) throws -> (Int, Float) {
        if engine == nil { try reset() }
        guard let engine else { throw CocoaError(.fileReadUnknown) }
        let phase = 2 * Float.pi * Float(step % 200) / 200
        let result = try engine.advance(normalizedActions: [effort * sin(phase)],
                                        controlStepCount: 1)
        guard result.failedEnvironmentSteps == 0,
              let angle = try engine.finalConfiguration().first,
              angle.isFinite else {
            throw MetalRoboTaskRolloutError.native("Pendulum physics failed at tick \(step + 1).")
        }
        step += 1
        return (step, angle)
    }
}

@MainActor
private struct PendulumView: View {
    @State private var simulation = PendulumSimulation()
    @State private var ticks = 0
    @State private var angle = 0.0
    @State private var effort = 0.1
    @State private var running = false
    @State private var resetting = false
    @State private var error: String?

    var body: some View {
        NavigationStack {
            ViewThatFits(in: .vertical) {
                controls
                ScrollView { controls }
            }
            .padding()
            .navigationTitle("Bench Pendulum")
        }
        .task(id: running) {
            guard running else { return }
            let clock = ContinuousClock()
            while !Task.isCancelled {
                let nextTick = clock.now + .milliseconds(20)
                do {
                    let (sample, position) = try await simulation.advance(effort: Float(effort))
                    guard !Task.isCancelled else { return }
                    ticks = sample
                    angle = Double(position)
                    try await clock.sleep(until: nextTick)
                } catch is CancellationError {
                    return
                } catch {
                    self.error = error.localizedDescription
                    running = false
                }
            }
        }

    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 20) {
            Canvas { context, size in
                let origin = CGPoint(x: size.width * 0.5, y: size.height * 0.29)
                let length = min(size.width * 0.42, size.height * 0.52)
                let tip = CGPoint(x: origin.x + length * sin(angle),
                                  y: origin.y + length * cos(angle))
                var stand = Path()
                stand.move(to: CGPoint(x: origin.x - length * 0.35, y: size.height * 0.89))
                stand.addLine(to: origin)
                stand.addLine(to: CGPoint(x: origin.x + length * 0.35, y: size.height * 0.89))
                context.stroke(stand, with: .color(.secondary.opacity(0.45)),
                               style: StrokeStyle(lineWidth: 6, lineCap: .round, lineJoin: .round))
                var rod = Path()
                rod.move(to: origin)
                rod.addLine(to: tip)
                context.stroke(rod, with: .color(.orange), style: StrokeStyle(lineWidth: 10, lineCap: .round))
                for (point, diameter, color) in [(origin, 19.0, Color.primary), (tip, 31.0, .orange)] {
                    context.fill(Path(ellipseIn: CGRect(x: point.x - diameter / 2, y: point.y - diameter / 2,
                                                        width: diameter, height: diameter)), with: .color(color))
                }
            }
            .frame(minHeight: 260)
            .background(.quaternary.opacity(0.6), in: RoundedRectangle(cornerRadius: 20))
            .accessibilityLabel("Pendulum diagram")
            HStack {
                Text("Tick \(ticks)")
                Spacer()
                Text("\(angle * 180 / .pi, format: .number.precision(.fractionLength(1)))°")
            }
            .font(.title3.monospacedDigit())
            LabeledContent("Effort amplitude", value: effort.formatted(.percent.precision(.fractionLength(0))))
            Slider(value: $effort, in: 0...0.35) {
                Text("Effort amplitude")
            }
            HStack {
                Button(running ? "Pause" : "Run", systemImage: running ? "pause.fill" : "play.fill") {
                    running.toggle()
                }
                .buttonStyle(.borderedProminent)
                Button("Reset", systemImage: "arrow.counterclockwise") {
                    running = false
                    resetting = true
                    Task {
                        defer { resetting = false }
                        do {
                            try await simulation.reset()
                            ticks = 0
                            angle = 0
                            error = nil
                        } catch {
                            self.error = error.localizedDescription
                        }
                    }
                }
                .buttonStyle(.bordered)
            }
            .disabled(resetting)
            if let error {
                Text(error).foregroundStyle(.red)
            }
        }
    }
}

@main
struct NumiLabIOS: App {
    init() {
        guard CommandLine.arguments.contains("--self-check") else { return }
        Task.detached {
            let simulation = PendulumSimulation()
            func report(_ message: String, status: Int32) -> Never {
                FileHandle.standardError.write(Data(("NUMI_IOS_CHECK " + message + "\n").utf8))
                exit(status)
            }
            do {
                var peak = Float.zero
                for _ in 0..<100 {
                    let (_, position) = try await simulation.advance(effort: 0.1)
                    peak = max(peak, abs(position))
                }
                guard peak > 0.2 else {
                    report("failed unexpectedly small pendulum motion", status: 1)
                }
                try await simulation.reset()
                let (sample, position) = try await simulation.advance(effort: 0.1)
                guard sample == 1, abs(position) < 1e-6 else {
                    report("failed reset did not restore the initial state", status: 1)
                }
                report("passed ticks=100 peak=\(peak)", status: 0)
            } catch {
                report("failed \(error)", status: 1)
            }
        }
    }

    var body: some Scene {
        WindowGroup { PendulumView() }
    }
}
