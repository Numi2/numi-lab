import Foundation
import CryptoKit
import Darwin

// Publication-boundary I/O only. Physics and learner updates retain their owners.
struct MetalRoboTrainingLifeCheckpoint: Codable {
    struct FileRecord: Codable {
        let bytes: UInt64
        let sha256: String
    }
    let schema: String
    let policyRevision: UInt64
    let completedUpdates: UInt64
    let acceptedControlTicks: [UInt64]
    let trainingContract: [String: String]
    let learnerExecutionIdentity: String
    let files: [String: FileRecord]

    static let filenames = ["policy.policypack", "learner.safetensors", "life.native", "rollout.rolloutpack", "candidate.policypack"]
    static func require(_ condition: Bool, _ message: String) throws {
        guard condition else { throw MetalRoboTaskRolloutError.native(message) }
    }
    static func hash(_ url: URL) throws -> FileRecord {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        try require(attributes[.type] as? FileAttributeType == .typeRegular,
                    "Training checkpoint payload must be a regular file.")
        let bytes = (attributes[.size] as? NSNumber)?.uint64Value ?? 0
        try require(bytes > 0 && bytes <= 16 * 1024 * 1024 * 1024,
                    "Training checkpoint payload has an invalid size.")
        let input = try FileHandle(forReadingFrom: url)
        defer { try? input.close() }
        var digest = SHA256()
        var count: UInt64 = 0
        while let data = try input.read(upToCount: 1024 * 1024), !data.isEmpty {
            digest.update(data: data)
            count += UInt64(data.count)
        }
        try require(count == bytes, "Training checkpoint payload changed while hashing.")
        return FileRecord(bytes: bytes, sha256: digest.finalize().map { String(format: "%02x", $0) }.joined())
    }
    private static func sync(_ url: URL, directory: Bool = false) throws {
        let fd = open(url.path, O_RDONLY | (directory ? O_DIRECTORY : 0))
        try require(fd >= 0, "Cannot open training checkpoint for fsync.")
        let result = fsync(fd)
        let closed = close(fd)
        try require(result == 0 && closed == 0, "Training checkpoint fsync failed.")
    }
    private static func writeExclusive(_ data: Data, to url: URL) throws {
        let fd = open(url.path, O_WRONLY | O_CREAT | O_EXCL, 0o600)
        try require(fd >= 0, "Cannot exclusively create training checkpoint metadata.")
        var okay = true
        data.withUnsafeBytes { buffer in
            var offset = 0
            while offset < buffer.count {
                let count = Darwin.write(fd, buffer.baseAddress!.advanced(by: offset), buffer.count - offset)
                if count < 0 && errno == EINTR { continue }
                if count <= 0 { okay = false; break }
                offset += count
            }
        }
        if okay { okay = fsync(fd) == 0 }
        let closed = close(fd)
        try require(okay && closed == 0, "Training checkpoint metadata write failed; partial file retained.")
    }
    static func read(from directory: URL) throws -> Self {
        let manifest = directory.appendingPathComponent("manifest.json")
        let metadata = try FileManager.default.attributesOfItem(atPath: manifest.path)
        try require(metadata[.type] as? FileAttributeType == .typeRegular &&
                    ((metadata[.size] as? NSNumber)?.uint64Value ?? UInt64.max) <= 1024 * 1024,
                    "Training checkpoint manifest is missing or oversized.")
        let data = try Data(contentsOf: manifest)
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        let marker = directory.appendingPathComponent("COMMITTED")
        let markerAttributes = try FileManager.default.attributesOfItem(atPath: marker.path)
        try require(markerAttributes[.type] as? FileAttributeType == .typeRegular &&
                    (markerAttributes[.size] as? NSNumber)?.intValue == 64,
                    "Training checkpoint is not committed.")
        try require(try Data(contentsOf: marker) == Data(digest.utf8),
                    "Training checkpoint commit marker does not bind its manifest.")
        let result = try JSONDecoder().decode(Self.self, from: data)
        try require(result.schema == "numi.training-life-checkpoint.v1" && result.policyRevision > 0 &&
                    result.completedUpdates > 0 && !result.acceptedControlTicks.isEmpty &&
                    result.acceptedControlTicks.allSatisfy { $0 > 0 } &&
                    result.learnerExecutionIdentity.count == 64 &&
                    Set(result.files.keys) == Set(filenames), "Invalid training checkpoint contract.")
        for name in filenames {
            let saved = result.files[name]!
            let actual = try hash(directory.appendingPathComponent(name))
            try require(saved.bytes == actual.bytes && saved.sha256 == actual.sha256,
                        "Training checkpoint payload mismatch: \(name).")
        }
        return result
    }
    static func save(to directory: URL, context: MetalRoboTaskRolloutContext,
                     policyRevision: UInt64, completedUpdates: UInt64,
                     trainingContract: [String: String], learnerExecutionIdentity: String,
                     policy: String, learner: String, rollout: String, candidate: String) throws -> Self {
        try require(mkdir(directory.path, 0o700) == 0,
                    "Cannot exclusively create training checkpoint directory; existing evidence retained.")
        // No published checkpoint exists until every immutable payload and the
        // manifest are durable. Interrupted directories remain inadmissible.
        for (name, source) in [("policy.policypack", policy), ("learner.safetensors", learner),
                               ("rollout.rolloutpack", rollout), ("candidate.policypack", candidate)] {
            let destination = directory.appendingPathComponent(name)
            try FileManager.default.copyItem(at: URL(fileURLWithPath: source), to: destination)
            try sync(destination)
        }
        try context.saveLifeCheckpoint(to: directory.appendingPathComponent("life.native"))
        var files: [String: FileRecord] = [:]
        for name in filenames { files[name] = try hash(directory.appendingPathComponent(name)) }
        let ticks = try context.finiteResourceAccounting().map { $0.accepted_control_ticks }
        let result = Self(schema: "numi.training-life-checkpoint.v1", policyRevision: policyRevision,
                          completedUpdates: completedUpdates, acceptedControlTicks: ticks,
                          trainingContract: trainingContract, learnerExecutionIdentity: learnerExecutionIdentity,
                          files: files)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(result)
        try writeExclusive(data, to: directory.appendingPathComponent("manifest.json"))
        try sync(directory, directory: true)
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        let pending = directory.appendingPathComponent("commit.pending")
        try writeExclusive(Data(digest.utf8), to: pending)
        // Hard-link publication is atomic and refuses an existing destination.
        try require(link(pending.path, directory.appendingPathComponent("COMMITTED").path) == 0,
                    "Training checkpoint commit publication failed.")
        try sync(directory, directory: true)
        try require(unlink(pending.path) == 0, "Training checkpoint commit cleanup failed.")
        try sync(directory, directory: true)
        return result
    }
}
