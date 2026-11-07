import Foundation

@main
enum CrowEpisodeEvidenceTests {
    static func main() throws {
        // Failure followed by a successful retry must remain a first-attempt
        // failure. A reached goal followed by a crash must also fail.
        var evidence = CrowEpisodeEvidence(environmentCount: 3)
        try evidence.record(environment: 0, waypoints: 2, completion: 0,
                            done: true, physicalFailure: true, terminationReason: 3)
        try evidence.record(environment: 0, waypoints: 5, completion: 1,
                            done: true, physicalFailure: false, terminationReason: 7)
        try evidence.record(environment: 1, waypoints: 5, completion: 1,
                            done: false, physicalFailure: false, terminationReason: 0)
        try evidence.record(environment: 1, waypoints: 5, completion: 1,
                            done: true, physicalFailure: true, terminationReason: 2)
        try evidence.record(environment: 2, waypoints: 5, completion: 1,
                            done: true, physicalFailure: false, terminationReason: 4)
        precondition(evidence.lanes[0].steps == 1)
        precondition(evidence.lanes[0].maximumWaypoints == 2)
        precondition(!evidence.lanes[0].safeCompletion)
        precondition(!evidence.lanes[1].safeCompletion)
        precondition(evidence.lanes[2].safeCompletion)
        precondition(evidence.lanes[2].firstCompletionStep == 1)
        precondition(evidence.json["safe_completion_count"] as? Int == 1)

        // Changing submission partitioning must not change any episode or
        // count. Repeats are continuous execution, not independent replicates.
        func partitioned(_ chunk: Int) throws -> Data {
            var recorder = CrowEpisodeEvidence(environmentCount: 4)
            for start in stride(from: 0, to: 24, by: chunk) {
                for step in start..<min(start + chunk, 24) {
                    for environment in 0..<4 {
                        try recorder.record(
                            environment: environment,
                            waypoints: Double(min(step / 2, 5)),
                            completion: step >= 10 ? 1 : 0,
                            done: step == 7 + environment || step == 20,
                            physicalFailure: step == 7 + environment,
                            terminationReason: step == 7 + environment ? 3 : 0
                        )
                    }
                }
            }
            return try JSONSerialization.data(withJSONObject: recorder.json,
                                               options: [.sortedKeys])
        }
        let one = try partitioned(1)
        let eight = try partitioned(8)
        let seven = try partitioned(7)
        precondition(one == eight)
        precondition(one == seven)

        // Incomplete episodes and invalid instrument inputs stay explicit.
        var censored = CrowEpisodeEvidence(environmentCount: 1)
        try censored.record(environment: 0, waypoints: 3, completion: 0,
                            done: false, physicalFailure: false, terminationReason: 0)
        precondition(censored.json["horizon_censored_count"] as? Int == 1)
        precondition(!censored.lanes[0].safeCompletion)
        do {
            try censored.record(environment: 0, waypoints: .nan, completion: 0,
                                done: false, physicalFailure: false, terminationReason: 0)
            fatalError("Nonfinite instrument input was accepted")
        } catch {}
        print("Crow first-episode evidence: retry, post-goal crash, timeout, censoring, invalid input, and partition controls passed")
    }
}
