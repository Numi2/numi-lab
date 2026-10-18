import Foundation
import AVFoundation
import AppKit
@main struct Export {
 static func main() async throws {
  let args=CommandLine.arguments
  let asset=AVURLAsset(url:URL(fileURLWithPath:args[1]))
  let duration=try await asset.load(.duration)
  let tracks=try await asset.loadTracks(withMediaType:.video)
  let composition=AVMutableComposition()
  let track=composition.addMutableTrack(withMediaType:.video,preferredTrackID:kCMPersistentTrackID_Invalid)!
  try track.insertTimeRange(CMTimeRange(start:.zero,duration:duration),of:tracks[0],at:.zero)
  track.preferredTransform=try await tracks[0].load(.preferredTransform)
  let target=CMTime(seconds:24,preferredTimescale:60000)
  composition.scaleTimeRange(CMTimeRange(start:.zero,duration:duration),toDuration:target)
  let export=AVAssetExportSession(asset:composition,presetName:AVAssetExportPreset1280x720)!
  try await export.export(to:URL(fileURLWithPath:args[2]),as:.mp4)
  print("native_wall_duration_s=\(duration.seconds) accelerated_preview_duration_s=24")
  let generator=AVAssetImageGenerator(asset:asset);generator.appliesPreferredTrackTransform=true
  for (i,fraction) in [0.12,0.36,0.60,0.87].enumerated() {
   let result=try await generator.image(at:CMTime(seconds:duration.seconds*fraction,preferredTimescale:600))
   let bitmap=NSBitmapImageRep(cgImage:result.image)
   try bitmap.representation(using:.png,properties:[:])!.write(to:URL(fileURLWithPath:args[3]+"/view-\(i).png"))
  }
 }
}
