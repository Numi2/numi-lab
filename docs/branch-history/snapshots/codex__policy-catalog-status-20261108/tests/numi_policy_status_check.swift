import CryptoKit
import Foundation

@main struct PolicyStatusCheck {
    static func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    static func main() throws {
        let fm = FileManager.default
        let run = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? fm.removeItem(at: run) }
        let selection = run.appendingPathComponent("selection")
        try fm.createDirectory(at: selection, withIntermediateDirectories: true)
        let pack = run.appendingPathComponent("deployment.policypack")
        let policy = Data("bound policy".utf8)
        try policy.write(to: pack)
        try JSONSerialization.data(withJSONObject: ["selected":"incumbent"]).write(to: selection.appendingPathComponent("selection.json"))
        let evidenceURL = selection.appendingPathComponent("incumbent.evidence.json")
        let metaURL = selection.appendingPathComponent("incumbent.evidence.json.meta.json")
        func store(_ falls: Int, _ includeHash: Bool = true) throws {
            let data = try JSONSerialization.data(withJSONObject: ["failed_environment_steps": 0, "height_or_tilt_termination_count": falls, "mean_tilt": 0.2, "clean_horizon_environment_rate": 0.75, "minimum_root_height_by_environment": [0.6]])
            try data.write(to: evidenceURL)
            var meta: [String: Any] = ["schema":"numi.policy-evidence-cache.v2", "evidence":["sha256":hash(data), "size_bytes":data.count]]
            if includeHash { meta["policy_sha256"] = hash(policy) }
            try JSONSerialization.data(withJSONObject: meta).write(to: metaURL)
        }
        precondition(numiPolicyEvidenceStatus(policy: pack, run: run).balance == .unknown)
        try store(0)
        let passed = numiPolicyEvidenceStatus(policy: pack, run: run)
        precondition(passed.balance == .passed && passed.role == "selected incumbent")
        try store(2)
        precondition(numiPolicyEvidenceStatus(policy: pack, run: run).balance == .failed)
        try store(0, false)
        precondition(numiPolicyEvidenceStatus(policy: pack, run: run).balance == .unknown)
        try store(0)
        try Data("changed".utf8).write(to: pack)
        precondition(numiPolicyEvidenceStatus(policy: pack, run: run).balance == .unknown)
        try policy.write(to: pack)
        try Data("{}".utf8).write(to: evidenceURL)
        precondition(numiPolicyEvidenceStatus(policy: pack, run: run).balance == .unknown)
        try store(0)
        try JSONSerialization.data(withJSONObject: ["selected":"candidate", "selected_candidate_label":"candidate"]).write(to: selection.appendingPathComponent("selection.json"))
        try fm.copyItem(at: evidenceURL, to: selection.appendingPathComponent("candidate.evidence.json"))
        try fm.copyItem(at: metaURL, to: selection.appendingPathComponent("candidate.evidence.json.meta.json"))
        let candidate = numiPolicyEvidenceStatus(policy: pack, run: run)
        precondition(candidate.role == "selected candidate" && candidate.balance == .passed)
        let researchPack = run.appendingPathComponent("candidate.deployment.policypack")
        try policy.write(to: researchPack)
        let research = numiPolicyEvidenceStatus(policy: researchPack, run: run)
        precondition(research.role == "candidate experiment" && research.balance == .passed)
        print("policy evidence status: passed, failed, missing binding, changed policy and changed evidence checks passed")
    }
}
