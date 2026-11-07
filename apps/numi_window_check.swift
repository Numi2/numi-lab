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

private func policyV5Fixture(id: String, contract: NumiWindowPolicyContract) -> Data {
    var payload = Data()
    let identifier = Data(id.utf8)
    appendLittleEndian(UInt64(identifier.count), to: &payload)
    payload.append(identifier)
    appendLittleEndian(UInt64(12), to: &payload)
    // Empty tensor vectors and layer lists, encoded with their real lengths.
    for _ in 0..<9 { appendLittleEndian(UInt64(0), to: &payload) }
    payload.append(Data(repeating: 0, count: 8))
    appendLittleEndian(contract.version, to: &payload)
    appendLittleEndian(contract.worldFingerprint, to: &payload)
    appendLittleEndian(contract.taskFingerprint, to: &payload)
    appendLittleEndian(contract.observationFingerprint, to: &payload)
    appendLittleEndian(contract.actionFingerprint, to: &payload)
    appendLittleEndian(UInt64(contract.compatibleTasks.count), to: &payload)
    for binding in contract.compatibleTasks {
        appendLittleEndian(binding.worldFingerprint, to: &payload)
        appendLittleEndian(binding.taskFingerprint, to: &payload)
    }
    var result = Data("MRLEARN\0".utf8)
    appendLittleEndian(UInt32(5), to: &result)
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

        let multiContract = NumiWindowPolicyContract(
            version: 2, worldFingerprint: 11, taskFingerprint: 22,
            observationFingerprint: 33, actionFingerprint: 44,
            compatibleTasks: [
                NumiWindowPolicyBinding(worldFingerprint: 11, taskFingerprint: 22),
                NumiWindowPolicyBinding(worldFingerprint: 11, taskFingerprint: 23),
            ])
        let multiURL = root.appendingPathComponent("multi.policypack")
        try policyV5Fixture(id: "multi", contract: multiContract).write(to: multiURL)
        let singleURL = root.appendingPathComponent("single-v5.policypack")
        try policyV5Fixture(id: "single", contract: contract).write(to: singleURL)
        try require(numiWindowPolicyMetadata(at: multiURL)?.contract == multiContract,
                    "version-5 multi-task contract was not decoded")
        try require(numiWindowPolicyMetadata(at: singleURL)?.contract == contract,
                    "version-5 single-task contract was not decoded")
        let secondary = NumiWindowPolicyContract(
            version: 1, worldFingerprint: 11, taskFingerprint: 23,
            observationFingerprint: 33, actionFingerprint: 44)
        let differentObservation = NumiWindowPolicyContract(
            version: 1, worldFingerprint: 11, taskFingerprint: 23,
            observationFingerprint: 34, actionFingerprint: 44)
        try require(multiContract.accepts(secondary), "secondary task not accepted")
        try require(!multiContract.accepts(differentObservation), "wrong observation accepted")
        try require(!multiContract.accepts(incompatible), "unlisted task accepted")
        let unsorted = NumiWindowPolicyContract(
            version: 2, worldFingerprint: 11, taskFingerprint: 22,
            observationFingerprint: 33, actionFingerprint: 44,
            compatibleTasks: [
                NumiWindowPolicyBinding(worldFingerprint: 11, taskFingerprint: 22),
                NumiWindowPolicyBinding(worldFingerprint: 11, taskFingerprint: 22),
            ])
        let invalidURL = root.appendingPathComponent("invalid.policypack")
        try policyV5Fixture(id: "invalid", contract: unsorted).write(to: invalidURL)
        try require(numiWindowPolicyMetadata(at: invalidURL) == nil,
                    "duplicate multi-task bindings accepted")
        var truncated = policyV5Fixture(id: "truncated", contract: multiContract)
        truncated.removeLast()
        try truncated.write(to: invalidURL)
        try require(numiWindowPolicyMetadata(at: invalidURL) == nil,
                    "truncated policy accepted")

        if CommandLine.arguments.count > 1 {
            let nativeURL = URL(fileURLWithPath: CommandLine.arguments[1])
            let native = numiWindowPolicyMetadata(at: nativeURL)
            try require(native?.id == "native-v5" && native?.revision == 12 &&
                        native?.contract == multiContract,
                        "native version-5 writer output was not decoded")
        }

        let tools = root.appendingPathComponent("tools", isDirectory: true)
        try manager.createDirectory(at: tools, withIntermediateDirectories: true)
        let dispatcher = tools.appendingPathComponent("numi")
        try Data("#!/bin/sh\n".utf8).write(to: dispatcher)
        try manager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: dispatcher.path)
        let trainingEnvironment = numiWindowTrainingEnvironment(
            ["NUMI_WINDOW_BUILD_DIR": "/tmp/window-build", "PATH": "/usr/bin"],
            workspace: root)
        try require(trainingEnvironment["NUMI_BUILD_DIR"] == "/tmp/window-build",
                    "window build override was not passed to training")
        try require(trainingEnvironment["PATH"] == tools.path + ":/usr/bin",
                    "workspace dispatcher was not added to training PATH")
        let defaultEnvironment = numiWindowTrainingEnvironment([:], workspace: root)
        try require(defaultEnvironment["NUMI_BUILD_DIR"] ==
                    root.appendingPathComponent(".numi/window-build").path,
                    "default window build path was not passed to training")

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
