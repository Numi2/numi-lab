import Foundation
@main struct Check {
  static func main() {
    let original = URL(fileURLWithPath: "/Users/numi/numi-g1-training-200-20261108/seed-1/run")
    let synthetic = URL(fileURLWithPath: "/Users/numi/numi-policy-status-real-trial-20261108/synthetic-binding")
    let old = numiPolicyEvidenceStatus(policy: original.appendingPathComponent("deployment.policypack"), run: original)
    let test = numiPolicyEvidenceStatus(policy: synthetic.appendingPathComponent("deployment.policypack"), run: synthetic)
    print("Original legacy metadata: \(old.role); \(old.balance.rawValue)")
    print("Copied evidence with SYNTHETIC policy binding: \(test.role); \(test.balance.rawValue)")
    precondition(old.role == "selected candidate" && old.balance == .unknown)
    precondition(test.role == "selected candidate" && test.balance == .failed)
  }
}
