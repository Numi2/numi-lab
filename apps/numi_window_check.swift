import Foundation

private enum WindowCheckError: Error, CustomStringConvertible {
    case failed(String)

    var description: String {
        switch self {
        case .failed(let message): return message
        }
    }
}

private func require(_ condition: @autoclosure () -> Bool, _ message: String)
throws {
    guard condition() else { throw WindowCheckError.failed(message) }
}

private func appendLittleEndian<T: FixedWidthInteger>(
    _ value: T,
    to data: inout Data
) {
    var encoded = value.littleEndian
    withUnsafeBytes(of: &encoded) { data.append(contentsOf: $0) }
}

private func policyFixture(
    id: String,
    revision: UInt64,
    contract: NumiWindowPolicyContract
) -> Data {
    var payload = Data()
    let identifier = Data(id.utf8)
    appendLittleEndian(UInt64(identifier.count), to: &payload)
    payload.append(identifier)
    appendLittleEndian(revision, to: &payload)
    // The metadata path deliberately does not deserialize weight tensors. A
    // native install still validates this complete payload and content hash.
    payload.append(Data(repeating: 0x5a, count: 32))
    appendLittleEndian(contract.version, to: &payload)
    appendLittleEndian(contract.worldFingerprint, to: &payload)
    appendLittleEndian(contract.taskFingerprint, to: &payload)
    appendLittleEndian(contract.observationFingerprint, to: &payload)
    appendLittleEndian(contract.actionFingerprint, to: &payload)

    var result = Data("MRLEARN\0".utf8)
    appendLittleEndian(UInt32(4), to: &result)
    appendLittleEndian(UInt32(2), to: &result)
    appendLittleEndian(UInt64(payload.count), to: &result)
    appendLittleEndian(UInt64(0), to: &result)
    result.append(payload)
    return result
}

private func writeScene(
    _ url: URL,
    id: String,
    visual: URL
) throws {
    let record: [String: Any] = [
        "format": "numi.window.scene.v1",
        "id": id,
        "robot_id": "fixture-robot",
        "robot_name": "Fixture Robot",
        "scene_id": "studio",
        "scene_name": "Studio",
        "task_id": "hold",
        "task_name": "Hold",
        "visual_observation": visual.path,
        "arguments": ["--zero-actions"],
        "available": true,
    ]
    try JSONSerialization.data(withJSONObject: record).write(to: url)
}

@main
private enum NumiWindowCheck {
    static func main() throws {
        let manager = FileManager.default
        let root = manager.temporaryDirectory.appendingPathComponent(
            "numi-window-check-\(UUID().uuidString)",
            isDirectory: true
        )
        try manager.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: root) }

        let child = root.appendingPathComponent("fixture-robot", isDirectory: true)
        let deep = child.appendingPathComponent("checkpoints", isDirectory: true)
        try manager.createDirectory(at: deep, withIntermediateDirectories: true)
        let contract = NumiWindowPolicyContract(
            version: 1,
            worldFingerprint: 11,
            taskFingerprint: 22,
            observationFingerprint: 33,
            actionFingerprint: 44
        )
        let rootPolicy = root.appendingPathComponent("root.policypack")
        let childPolicy = child.appendingPathComponent("child.policypack")
        let hiddenCheckpoint = deep.appendingPathComponent("deep.policypack")
        try policyFixture(id: "root", revision: 7, contract: contract)
            .write(to: rootPolicy)
        try policyFixture(id: "child", revision: 8, contract: contract)
            .write(to: childPolicy)
        try policyFixture(id: "deep", revision: 9, contract: contract)
            .write(to: hiddenCheckpoint)

        let metadata = numiWindowPolicyMetadata(at: childPolicy)
        try require(metadata?.id == "child", "policy id metadata was not decoded")
        try require(metadata?.revision == 8, "policy revision metadata was not decoded")
        try require(metadata?.contract == contract, "policy contract metadata was not decoded")
        let policies = numiWindowPolicyChoices(in: root)
        try require(policies.count == 2, "catalog descended into checkpoint evidence")
        try require(
            numiWindowCompatiblePolicyChoices(
                policies,
                robotID: "wrong-label",
                contract: contract
            ).count == 2,
            "exact contract did not override presentation labels"
        )
        let incompatible = NumiWindowPolicyContract(
            version: 1,
            worldFingerprint: 99,
            taskFingerprint: 22,
            observationFingerprint: 33,
            actionFingerprint: 44
        )
        try require(
            numiWindowCompatiblePolicyChoices(
                policies,
                robotID: "fixture-robot",
                contract: incompatible
            ).isEmpty,
            "incompatible policy survived the exact contract gate"
        )

        let visual = root.appendingPathComponent("visual.bin")
        try Data([0]).write(to: visual)
        try writeScene(
            root.appendingPathComponent("root.numi-window.json"),
            id: "root-scene",
            visual: visual
        )
        try writeScene(
            child.appendingPathComponent("child.numi-window.json"),
            id: "child-scene",
            visual: visual
        )
        try writeScene(
            deep.appendingPathComponent("deep.numi-window.json"),
            id: "deep-scene",
            visual: visual
        )
        let scenes = numiWindowSceneChoices(in: root)
        try require(scenes.count == 2, "scene catalog recursively scanned run evidence")
        print("numi_window_check passed policies=\(policies.count) scenes=\(scenes.count)")
    }
}
