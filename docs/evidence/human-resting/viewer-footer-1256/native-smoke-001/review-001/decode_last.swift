import Foundation
import AVFoundation
import AppKit
import CoreVideo
import CoreGraphics

struct Saved {
    let index: Int
    let pts: CMTime
    let width: Int
    let height: Int
    let rowBytes: Int
    let pixels: Data
}

let movie = URL(fileURLWithPath: CommandLine.arguments[1])
let outDir = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
let asset = AVURLAsset(url: movie)
guard let track = asset.tracks(withMediaType: .video).first else { fatalError("missing video track") }
let reader = try AVAssetReader(asset: asset)
let settings: [String: Any] = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
output.alwaysCopiesSampleData = false
reader.add(output)
guard reader.startReading() else { throw reader.error ?? NSError(domain: "reader-start", code: 1) }
var count = 0
var lastPTS: CMTime? = nil
var nonmonotonic = 0
var tail: [Saved] = []
while let sample = output.copyNextSampleBuffer() {
    guard CMSampleBufferGetNumSamples(sample) == 1 else { fatalError("non-single sample") }
    let pts = CMSampleBufferGetOutputPresentationTimeStamp(sample)
    if let prior = lastPTS, CMTimeCompare(pts, prior) <= 0 { nonmonotonic += 1 }
    lastPTS = pts
    guard let pb = CMSampleBufferGetImageBuffer(sample) else { fatalError("sample lacks pixel buffer") }
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(pb) else { fatalError("pixel buffer lacks base") }
    let width = CVPixelBufferGetWidth(pb), height = CVPixelBufferGetHeight(pb)
    let rowBytes = CVPixelBufferGetBytesPerRow(pb)
    let pixels = Data(bytes: base, count: rowBytes * height)
    if true {
    tail.append(Saved(index: count, pts: pts, width: width, height: height, rowBytes: rowBytes, pixels: pixels))
    }
    if tail.count > 1 { tail.removeFirst() }
    count += 1
}
guard reader.status == .completed else { throw reader.error ?? NSError(domain: "reader-incomplete", code: 2) }
guard count > 0 else { fatalError("empty movie") }
let cs = CGColorSpaceCreateDeviceRGB()
for s in tail {
    let activeRGB = s.pixels.enumerated().filter { ($0.offset % 4) != 3 && $0.element != 0 }.count
    print("last_sample_nonzero_rgb_bytes=\(activeRGB)")
    let provider = CGDataProvider(data: s.pixels as CFData)!
    let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
    guard let cg = CGImage(width: s.width, height: s.height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: s.rowBytes, space: cs, bitmapInfo: info, provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { fatalError("CGImage creation failed") }
    let rep = NSBitmapImageRep(cgImage: cg)
    let name = String(format: "reader-sample-%03d.png", s.index)
    try rep.representation(using: .png, properties: [:])!.write(to: outDir.appendingPathComponent(name))
    print("index=\(s.index) pts=\(s.pts.seconds) size=\(s.width)x\(s.height) row_bytes=\(s.rowBytes) bytes=\(s.pixels.count) file=\(name)")
}
print("samples=\(count) nonmonotonic_or_duplicate_pts=\(nonmonotonic) last_pts=\(lastPTS!.seconds) status=completed")
