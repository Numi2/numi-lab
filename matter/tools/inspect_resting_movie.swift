// Run on the simulation Mac. Inspect the native recording without retiming it.
import Foundation
import AVFoundation
import AppKit

let root = URL(fileURLWithPath: CommandLine.arguments[1])
let asset = AVURLAsset(url: root.appendingPathComponent("native-viewer.mov"))
let generator = AVAssetImageGenerator(asset: asset)
generator.appliesPreferredTrackTransform = true
do {
    let csv = try String(contentsOf: root.appendingPathComponent("resting-surface-audit.csv"), encoding: .utf8)
    let simulatedTimes = try csv.split(separator: "\n").dropFirst().map { line -> Double in
        guard let first = line.split(separator: ",").first, let value = Double(first), value.isFinite else {
            throw NSError(domain: "invalid-surface-time", code: 1)
        }
        return value
    }
    let reader = try AVAssetReader(asset: asset)
    guard let track = asset.tracks(withMediaType: .video).first else {
        throw NSError(domain: "missing-video-track", code: 2)
    }
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
    reader.add(output)
    guard reader.startReading() else { throw reader.error! }
    var timestamps: [CMTime] = []
    var timingMarkers = 0
    while let sample = output.copyNextSampleBuffer() {
        // Zero-sample AV edit/timing markers are not image frames.
        if CMSampleBufferGetNumSamples(sample) == 0 { timingMarkers += 1; continue }
        guard CMSampleBufferGetNumSamples(sample) == 1 else {
            throw NSError(domain: "batched-video-samples", code: 3)
        }
        timestamps.append(CMSampleBufferGetOutputPresentationTimeStamp(sample))
    }
    guard reader.status == .completed && !timestamps.isEmpty else {
        throw reader.error ?? NSError(domain: "incomplete-recording", code: 4)
    }
    // Compressed H.264 samples can arrive in decode order. Check presentation
    // order while retaining every image sample, including non-keyframes.
    timestamps.sort { CMTimeCompare($0, $1) < 0 }
    guard timestamps.count == simulatedTimes.count else {
        throw NSError(domain: "movie-surface-frame-count-mismatch", code: 5)
    }
    var maxGap = 0.0
    for i in 1..<timestamps.count {
        let gap = timestamps[i].seconds - timestamps[i-1].seconds
        guard gap > 0 && simulatedTimes[i] > simulatedTimes[i-1] else {
            throw NSError(domain: "nonmonotonic-presentation", code: 6)
        }
        maxGap = max(maxGap, gap)
    }
    print("frames=\(timestamps.count) timing_markers=\(timingMarkers) first_wall_s=\(timestamps[0].seconds) last_wall_s=\(timestamps.last!.seconds) max_gap_wall_s=\(maxGap) duration_wall_s=\(asset.duration.seconds)")
    var selected = [("initial", 0), ("middle", timestamps.count/2), ("final", timestamps.count-1)]
    if CommandLine.arguments.count > 2 {
        let layerNames = CommandLine.arguments.count > 3 && CommandLine.arguments[3] == "7"
            ? ["skin", "muscles", "skeleton", "organs", "lungs", "heart", "vessels"]
            : ["skin", "muscles", "skeleton", "organs", "lungs", "heart"]
        guard let period = Double(CommandLine.arguments[2]), period.isFinite, period > 0,
              simulatedTimes.last! > (Double(layerNames.count)-0.5) * period else {
            throw NSError(domain: "incomplete-anatomical-layer-tour", code: 7)
        }
        for (layer, name) in layerNames.enumerated() {
            let target = (Double(layer) + 0.5) * period
            let index = simulatedTimes.indices.min { abs(simulatedTimes[$0]-target) < abs(simulatedTimes[$1]-target) }!
            selected.append((name, index))
        }
    }
    generator.requestedTimeToleranceBefore = .zero
    generator.requestedTimeToleranceAfter = .zero
    for (name, index) in selected {
        var actual = CMTime.zero
        let frame = try generator.copyCGImage(at: timestamps[index], actualTime: &actual)
        guard CMTimeCompare(actual, timestamps[index]) == 0 else {
            throw NSError(domain: "inexact-frame-extraction", code: 8)
        }
        let bitmap = NSBitmapImageRep(cgImage: frame)
        try bitmap.representation(using: .png, properties: [:])!.write(to: root.appendingPathComponent("frame-\(name).png"))
        print("frame=\(name) index=\(index) simulated_s=\(simulatedTimes[index]) wall_s=\(actual.seconds) width=\(frame.width) height=\(frame.height)")
    }
} catch { print(error); exit(1) }
