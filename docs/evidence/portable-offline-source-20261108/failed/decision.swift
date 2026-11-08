import Foundation
@main struct DecisionCheck {
    static func main() {
        let run = URL(fileURLWithPath: CommandLine.arguments[1])
        let policy = run.appendingPathComponent("deployment.policypack")
        let status = numiPolicyEvidenceStatus(policy: policy, run: run)
        let decision = numiGuidedPolicyDecision(policy: policy, run: run)
        print("\(status.balance.rawValue)|load=\(decision.shouldLoad)")
    }
}
