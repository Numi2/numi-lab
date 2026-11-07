import Foundation

// First episode only, measured from the first accepted post-step sample.
// Later auto-resets never add discontinuous positions to this record.
struct EpisodeForwardEvidence {
    private(set) var origin: [Double]
    private(set) var peak: [Double]
    private(set) var final: [Double]
    private(set) var active: [Bool]
    private(set) var valid: [Bool]

    init(environmentCount: Int) {
        origin = Array(repeating: .nan, count: environmentCount)
        peak = Array(repeating: 0.0, count: environmentCount)
        final = Array(repeating: 0.0, count: environmentCount)
        active = Array(repeating: true, count: environmentCount)
        valid = Array(repeating: true, count: environmentCount)
    }

    mutating func record(environment: Int, position: Double,
                         done: Bool, physicsError: Bool) {
        guard active[environment] else { return }
        if physicsError || !position.isFinite {
            valid[environment] = false
            active[environment] = false
            return
        }
        if !origin[environment].isFinite {
            origin[environment] = position
        }
        let progress = position - origin[environment]
        peak[environment] = max(peak[environment], progress)
        final[environment] = progress
        if done { active[environment] = false }
    }

    var available: Bool {
        !origin.isEmpty && origin.allSatisfy(\.isFinite) &&
            valid.allSatisfy { $0 }
    }

    static func supported(trace: Bool, scheduledResets: Bool,
                          chunk: Int, repeats: Int) -> Bool {
        trace && !scheduledResets && chunk == 1 && repeats == 1
    }
}
