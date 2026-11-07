import Foundation
@main struct DecisionCheck {
 static func main() throws {
  let run=URL(fileURLWithPath:CommandLine.arguments[1])
  let policy=run.appendingPathComponent("deployment.policypack")
  let status=numiPolicyEvidenceStatus(policy:policy,run:run)
  let decision=numiGuidedPolicyDecision(policy:policy,run:run)
  let data=try JSONSerialization.data(withJSONObject:["role":status.role,"balance":status.balance.rawValue,"should_load":decision.shouldLoad,"message":decision.message],options:[.sortedKeys])
  print(String(decoding:data,as:UTF8.self))
 }
}
