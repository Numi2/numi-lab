import Foundation
@main struct Check {
 static func main() {
  for (path, expected) in [
   ("/Users/numi/numi-integrated-installed-smoke-20261108/workspace/.numi/runs/window-train-smoke", "failed guided balance screen"),
   ("/Users/numi/numi-g1-training-200-20261108/seed-1/run", "balance evidence unavailable")
  ] {
   let run=URL(fileURLWithPath:path)
   let decision=numiGuidedPolicyDecision(policy:run.appendingPathComponent("deployment.policypack"),run:run)
   print("\(path): shouldLoad=\(decision.shouldLoad), \(decision.message)")
   precondition(!decision.shouldLoad && decision.message.contains(expected))
  }
 }
}
