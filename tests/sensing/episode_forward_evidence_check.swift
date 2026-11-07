import Foundation

@main
struct EpisodeForwardEvidenceCheck {
    static func main() {
        var tracker = EpisodeForwardEvidence(environmentCount: 2)
        tracker.record(environment: 0, position: 10, done: false, physicsError: false)
        tracker.record(environment: 1, position: -2, done: false, physicsError: false)
        tracker.record(environment: 0, position: 12, done: true, physicsError: false)
        tracker.record(environment: 1, position: -1, done: false, physicsError: false)
        // Reset teleport and later episodes must not enter environment 0.
        tracker.record(environment: 0, position: 100, done: false, physicsError: false)
        tracker.record(environment: 1, position: -1.5, done: true, physicsError: false)
        precondition(tracker.available)
        precondition(tracker.peak == [2, 1] && tracker.final == [2, 0.5])
        precondition(!EpisodeForwardEvidence.supported(
            trace: true, scheduledResets: true, chunk: 1, repeats: 1))
        precondition(!EpisodeForwardEvidence.supported(
            trace: true, scheduledResets: false, chunk: 2, repeats: 1))
        precondition(!EpisodeForwardEvidence.supported(
            trace: false, scheduledResets: false, chunk: 1, repeats: 1))
        precondition(!EpisodeForwardEvidence.supported(
            trace: true, scheduledResets: false, chunk: 1, repeats: 2))
        precondition(EpisodeForwardEvidence.supported(
            trace: true, scheduledResets: false, chunk: 1, repeats: 1))
        var failed = EpisodeForwardEvidence(environmentCount: 1)
        failed.record(environment: 0, position: 0, done: false, physicsError: false)
        failed.record(environment: 0, position: 1, done: true, physicsError: true)
        precondition(!failed.available)
        var invalid = EpisodeForwardEvidence(environmentCount: 1)
        invalid.record(environment: 0, position: .nan, done: false, physicsError: false)
        precondition(!invalid.available)
        print("PASS first-episode progress, reset isolation, and failure gating")
    }
}
