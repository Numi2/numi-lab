import Foundation

// An observer of accepted transitions. It never resets or controls a world.
// Each environment contributes exactly its first episode, even when the
// executor subsequently resets it or the submission/repeat boundary changes.
struct CrowEpisodeEvidence {
    struct Lane {
        var steps = 0
        var maximumWaypoints = 0.0
        var completionSeen = false
        var firstCompletionStep: Int?
        var closed = false
        var physicalFailure = false
        var terminationReason: UInt32 = 0

        var safeCompletion: Bool { completionSeen && !physicalFailure }
    }

    private(set) var lanes: [Lane]

    init(environmentCount: Int) {
        precondition(environmentCount > 0)
        lanes = Array(repeating: Lane(), count: environmentCount)
    }

    mutating func record(
        environment: Int,
        waypoints: Double,
        completion: Double,
        done: Bool,
        physicalFailure: Bool,
        terminationReason: UInt32
    ) throws {
        guard lanes.indices.contains(environment),
              waypoints.isFinite, completion.isFinite,
              (0.0...5.0).contains(waypoints),
              (0.0...1.0).contains(completion),
              !physicalFailure || done
        else {
            throw NSError(
                domain: "NumiCrowEpisodeEvidence", code: 1,
                userInfo: [NSLocalizedDescriptionKey:
                    "Invalid accepted navigation transition."]
            )
        }
        guard !lanes[environment].closed else { return }
        lanes[environment].steps += 1
        lanes[environment].maximumWaypoints = max(
            lanes[environment].maximumWaypoints, waypoints
        )
        if completion >= 1.0 {
            lanes[environment].completionSeen = true
            if lanes[environment].firstCompletionStep == nil {
                lanes[environment].firstCompletionStep =
                    lanes[environment].steps
            }
        }
        if done {
            lanes[environment].closed = true
            lanes[environment].physicalFailure = physicalFailure
            lanes[environment].terminationReason = terminationReason
        }
    }

    var json: [String: Any] {
        [
            "schema": "numi.crow-first-episode-evidence.v1",
            "unit": "first episode of each environment since context creation",
            "environment_count": lanes.count,
            "safe_completion_count": lanes.filter(\.safeCompletion).count,
            "completion_seen_count": lanes.filter(\.completionSeen).count,
            "physical_failure_count": lanes.filter(\.physicalFailure).count,
            "closed_episode_count": lanes.filter(\.closed).count,
            "horizon_censored_count": lanes.filter { !$0.closed }.count,
            "steps_by_environment": lanes.map(\.steps),
            "maximum_waypoints_by_environment": lanes.map(\.maximumWaypoints),
            "completion_seen_by_environment": lanes.map(\.completionSeen),
            "safe_completion_by_environment": lanes.map(\.safeCompletion),
            "physical_failure_by_environment": lanes.map(\.physicalFailure),
            "closed_by_environment": lanes.map(\.closed),
            "termination_reason_by_environment": lanes.map(\.terminationReason),
            "first_completion_step_by_environment": lanes.map {
                $0.firstCompletionStep.map { $0 as Any } ?? NSNull()
            },
            "boundary": "waypoint completion; does not establish supported landing",
        ]
    }
}
