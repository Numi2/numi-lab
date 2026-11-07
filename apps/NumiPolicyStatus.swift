import CryptoKit
import Foundation

// These labels describe the retained held-out balance checks only. A hash-bound
// evaluation is required; compatibility and selector advancement are separate.
public enum NumiPolicyBalanceStatus: String, Sendable {
    case passed = "guided held-out balance screen passed"
    case failed = "guided held-out balance screen failed"
    case unknown = "guided held-out balance status unknown"
}

public struct NumiPolicyEvidenceStatus: Sendable {
    public let role: String
    public let balance: NumiPolicyBalanceStatus

    public init(role: String, balance: NumiPolicyBalanceStatus) {
        self.role = role
        self.balance = balance
    }
}

private func numiSHA256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

public func numiPolicyEvidenceStatus(policy: URL, run: URL) -> NumiPolicyEvidenceStatus {
    let unknown = NumiPolicyEvidenceStatus(role: "research policy", balance: .unknown)
    let directory = run.appendingPathComponent("selection", isDirectory: true)
    guard let selectionData = try? Data(contentsOf: directory.appendingPathComponent("selection.json")),
          let selection = (try? JSONSerialization.jsonObject(with: selectionData)) as? [String: Any],
          let selected = selection["selected"] as? String,
          let policyData = try? Data(contentsOf: policy) else { return unknown }
    let label: String
    let role: String
    if policy.lastPathComponent == "deployment.policypack" {
        if selected == "incumbent" {
            label = "incumbent"
            role = "selected incumbent"
        } else if selected == "candidate",
                  let candidate = selection["selected_candidate_label"] as? String,
                  candidate.hasPrefix("candidate"),
                  candidate.allSatisfy({ $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "-") }) {
            label = candidate
            role = "selected candidate"
        } else { return unknown }
    } else if policy.lastPathComponent == "candidate.deployment.policypack" {
        label = "candidate"
        role = "candidate experiment"
    } else { return unknown }
    let result = NumiPolicyEvidenceStatus(role: role, balance: .unknown)
    let evidenceURL = directory.appendingPathComponent("\(label).evidence.json")
    let metadataURL = directory.appendingPathComponent("\(label).evidence.json.meta.json")
    guard let evidenceData = try? Data(contentsOf: evidenceURL),
          let evidence = (try? JSONSerialization.jsonObject(with: evidenceData)) as? [String: Any],
          let metadataData = try? Data(contentsOf: metadataURL),
          let metadata = (try? JSONSerialization.jsonObject(with: metadataData)) as? [String: Any],
          metadata["schema"] as? String == "numi.policy-evidence-cache.v2",
          let evidenceManifest = metadata["evidence"] as? [String: Any],
          evidenceManifest["sha256"] as? String == numiSHA256(evidenceData),
          (evidenceManifest["size_bytes"] as? NSNumber)?.intValue == evidenceData.count,
          metadata["policy_sha256"] as? String == numiSHA256(policyData),
          let failed = (evidence["failed_environment_steps"] as? NSNumber)?.intValue,
          let falls = (evidence["height_or_tilt_termination_count"] as? NSNumber)?.intValue,
          let tilt = (evidence["mean_tilt"] as? NSNumber)?.doubleValue,
          let horizon = (evidence["clean_horizon_environment_rate"] as? NSNumber)?.doubleValue,
          let heights = evidence["minimum_root_height_by_environment"] as? [NSNumber],
          !heights.isEmpty, heights.allSatisfy({ $0.doubleValue.isFinite }),
          let minHeight = heights.map(\.doubleValue).min(),
          failed >= 0, falls >= 0, tilt.isFinite, horizon.isFinite,
          horizon >= 0, horizon <= 1, minHeight.isFinite else { return result }
    let passed = failed == 0 && falls == 0 && tilt <= 0.35 && horizon >= 0.5 && minHeight >= 0.55
    return NumiPolicyEvidenceStatus(role: role, balance: passed ? .passed : .failed)
}

// Automatic guided loading shares the catalog's evidence identity checks.
// Manual selection of research policies is independent of this decision.
public struct NumiGuidedPolicyDecision {
    public let shouldLoad: Bool
    public let message: String
}

public func numiGuidedPolicyDecision(policy: URL, run: URL) -> NumiGuidedPolicyDecision {
    let status = numiPolicyEvidenceStatus(policy: policy, run: run)
    switch status.balance {
    case .passed where status.role == "selected candidate":
        return NumiGuidedPolicyDecision(shouldLoad: true, message: "new policy passed guided held-out balance screen")
    case .passed where status.role == "selected incumbent":
        return NumiGuidedPolicyDecision(shouldLoad: true, message: "starting policy passed guided held-out balance screen")
    case .failed:
        return NumiGuidedPolicyDecision(shouldLoad: false, message: "experiment complete · policy failed guided balance screen · not loaded")
    default:
        return NumiGuidedPolicyDecision(shouldLoad: false, message: "experiment complete · balance evidence unavailable · not loaded")
    }
}
