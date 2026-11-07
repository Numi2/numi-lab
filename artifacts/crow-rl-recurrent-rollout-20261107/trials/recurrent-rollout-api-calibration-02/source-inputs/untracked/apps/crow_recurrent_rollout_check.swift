import Foundation

private func require(_ condition: Bool, _ reason: String) throws {
    if !condition { throw MetalRoboTaskRolloutError.invalidShape(reason) }
}
private func save(_ values: [Float], _ url: URL) throws {
    try values.withUnsafeBytes { try Data($0).write(to: url, options: .withoutOverwriting) }
}

@main
struct CrowRecurrentRolloutCheck {
    static func main() throws {
        try require(CommandLine.arguments.count == 3, "Expected NATIVE_EVIDENCE NEW_OUTPUT")
        let source = URL(fileURLWithPath: CommandLine.arguments[1]).standardizedFileURL
        let output = URL(fileURLWithPath: CommandLine.arguments[2]).standardizedFileURL
        try require(!FileManager.default.fileExists(atPath: output.path), "Output must be new")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let plan: [String: Any] = ["schema": "numi.crow-swift-recurrent-rollout-plan.v1",
            "prediction": "Swift chunk1 batch concatenation and chunk8 training append publish identical native format8 segment bytes, hidden/reset/learner tensors and final q/v; five rejection controls per arm preserve the subsequent life",
            "conditions": "two supported real native v13 lanes,40 ticks, lane1 reset9, next weights18; previews only at both segment endpoints; source native reference adds extra intermediate previews",
            "stopping": "exactly two bounded native instrument arms; preserve failures; no retries or tolerance changes",
            "limitations": "collection/publication engineering instrument, not sequence PPO or behavioral qualification"]
        try JSONSerialization.data(withJSONObject: plan, options: [.prettyPrinted, .sortedKeys]).write(to: output.appendingPathComponent("plan.json"), options: .withoutOverwriting)
        var negatives = 0
        func reject(_ phrase: String, _ body: () throws -> Void) throws {
            do { try body() } catch {
                try require(String(describing: error).contains(phrase), "Wrong rejection cause: \(error)")
                negatives += 1; return
            }
            throw MetalRoboTaskRolloutError.invalidShape("Invalid recurrent Swift use accepted")
        }
        for (chunk, append) in [(1, false), (8, true)] {
            let directory = output.appendingPathComponent(append ? "append-chunk8" : "concat-chunk1")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
            let visual = try MetalRoboTaskVisualObservationConfiguration.loadArtifact(at: source.appendingPathComponent("retinal-observation.json"))
            let configuration = MetalRoboTaskRolloutConfiguration(environmentCount: 2, seed: 2651109401,
                difficultyBandRange: 0...0, birdFlowJourneyVariant: .v13RetinalFeeding)
            let context = try MetalRoboTaskRolloutContext(manifest: .init(source: .birdFlowAmericanCrowJourney,
                sensorsAndPhysics: configuration, visualSensor: visual))
            try context.setStateReadback(true)
            for (begin, end, policy, name) in [(0,18,"first.policypack","segment-first"), (18,40,"next.policypack","segment-next")] {
                try context.loadPolicy(at: source.appendingPathComponent(policy))
                var chunks: [MetalRoboPolicyRolloutBatch] = []
                var actor: [Float] = [], critic: [Float] = [], motion: [Float] = [], teacher: [Float] = []
                var latents: [Float] = [], logp: [Float] = [], values: [Float] = [], before: [Float] = []
                var resets: [UInt32] = [], transitions: [MetalRoboTaskTransition] = []
                var start = begin
                var lastCount = 0
                while start < end {
                    let count = min(chunk, end-start);lastCount = count
                    var masks = [UInt32](repeating: 0, count: count*2)
                    if start <= 9 && 9 < start+count { masks[(9-start)*2+1] = 1 }
                    let advance = try context.advanceWithPolicy(resetMasks: masks, controlStepCount: count,
                        policyRevision: context.installedPolicyRevision, evaluateFinalPolicy: start+count == end)
                    try require(advance.failedEnvironmentSteps == 0 && advance.successfulEnvironmentSteps == count*2, "Native Swift recurrent step failed")
                    if append {
                        try context.appendCurrentPolicyRollout(controlStepCount: count, actorObservations: &actor,
                            criticObservations: &critic, motionFeatures: &motion, teacherActions: &teacher,
                            latents: &latents, logProbabilities: &logp, values: &values, transitions: &transitions,
                            policyMemoryBefore: &before, policyMemoryResets: &resets)
                    } else { chunks.append(try context.policyRolloutBatch(controlStepCount: count)) }
                    start += count
                }
                let batch: MetalRoboPolicyRolloutBatch
                if append {
                    batch = MetalRoboPolicyRolloutBatch(controlStepCount: end-begin, environmentCount: 2,
                        actorObservationCount: 1251, criticObservationCount: 97, actionCount: 18, motionFeatureCount: context.layout.motionFeatureCount,
                        actorObservations: actor, criticObservations: critic, motionFeatures: motion, teacherActions: teacher,
                        latents: latents, logProbabilities: logp, values: values, transitions: transitions,
                        policyMemoryWidth: context.policyMemoryWidth, policyMemoryBefore: before, policyMemoryResets: resets)
                } else { batch = try MetalRoboPolicyRolloutBatch.concatenating(chunks) }
                try require(batch.policyMemoryWidth == 7 && batch.policyMemoryResets.count == (end-begin)*2 && batch.policyMemoryBefore.count == (end-begin)*14, "Preview was included in training traces")
                let target = directory.appendingPathComponent(name+".rolloutpack")
                let bootstrap = try context.bootstrapPolicyValues()
                try context.writePolicyRolloutPack(batch, bootstrapValues: bootstrap, id: name, to: target)
                try require(try Data(contentsOf: target) == Data(contentsOf: source.appendingPathComponent(name+".rolloutpack")), "Swift/native recurrent rollout bytes differ")
                if begin == 0 {
                    try reject("exactly the latest real") { _ = try context.policyMemoryBefore(controlStepCount: lastCount+1) }
                    try reject("explicit hidden/reset destinations") {
                        var a: [Float] = [], c: [Float] = [], m: [Float] = [], t: [Float] = [], l: [Float] = [], p: [Float] = [], v: [Float] = []
                        var x: [MetalRoboTaskTransition] = []
                        try context.appendCurrentPolicyRollout(controlStepCount: lastCount, actorObservations: &a, criticObservations: &c,
                            motionFeatures: &m, teacherActions: &t, latents: &l, logProbabilities: &p, values: &v, transitions: &x)
                    }
                    var missing = batch;missing.policyMemoryWidth = 0;missing.policyMemoryBefore = [];missing.policyMemoryResets = []
                    try reject("explicit hidden/reset traces") {
                        try context.writePolicyRolloutPack(missing, bootstrapValues: bootstrap, id: name, to: directory.appendingPathComponent("missing.forbidden"))
                    }
                    try reject("incompatible dimensions") { _ = try MetalRoboPolicyRolloutBatch.concatenating([batch, missing]) }
                    var truncated = batch;truncated.policyMemoryBefore.removeLast()
                    try reject("dimensions") {
                        try context.writePolicyRolloutPack(truncated, bootstrapValues: bootstrap, id: name, to: directory.appendingPathComponent("truncated.forbidden"))
                    }
                }
            }
            try save(context.finalConfiguration(), directory.appendingPathComponent("q.f32"))
            try save(context.finalVelocity(), directory.appendingPathComponent("v.f32"))
            for name in ["q.f32", "v.f32"] {
                try require(try Data(contentsOf: directory.appendingPathComponent(name)) == Data(contentsOf: source.appendingPathComponent("tail/"+name)), "Swift/native final physical state differs")
            }
        }
        let result: [String: Any] = ["schema": "numi.crow-swift-recurrent-rollout-evidence.v1", "status": "passed",
            "native_swift_all_segment_bytes_exact": true, "concat_and_training_append_exact": true,
            "terminal_previews_excluded": true, "final_physics_exact": true, "rejection_controls": negatives,
            "recurrent_PPO_integrated": false, "embodied_memory_qualified": false]
        let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: output.appendingPathComponent("summary.json"), options: .withoutOverwriting)
        print(String(data: data, encoding: .utf8)!)
    }
}
