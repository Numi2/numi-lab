import Foundation
import CryptoKit

// Display-only splice: one original early frame and the last four original
// frames are non-emitted pre-roll; subsequent frames are the actual resumed
// trial. The branch moves at the seam. The output is not simulation evidence.
let args = CommandLine.arguments
guard args.count == 5, let limit = Int(args[4]), limit > 0 else { fatalError("usage: make_render_splice original.json trial.json output.json limit") }
func read(_ path: String) throws -> [String: Any] {
  guard let o = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath:path))) as? [String:Any] else { fatalError("bad envelope") }
  return o
}
let original = try read(args[1]), trial = try read(args[2])
guard let op = original["payload"] as? [String:Any], var tp = trial["payload"] as? [String:Any],
      let of = op["frames"] as? [[String:Any]], let tf = tp["frames"] as? [[String:Any]],
      let originalNames = op["body_names"] as? [String], let trialNames = tp["body_names"] as? [String],
      originalNames == trialNames, of.count >= 4, !tf.isEmpty else { fatalError("incompatible frames") }
// Truncate at first termination for this specific retained trial.
let firstEpisode = Array(tf.prefix(limit))
var frames = [of[0]] + Array(of.suffix(4)) + firstEpisode
for i in frames.indices { frames[i]["step"] = i+1 }
tp["frames"] = frames
tp["frame_count"] = frames.count
tp["presentation_splice_provenance"] = ["purpose":"render-only, five omitted pre-roll frames; not a continuous simulator record", "original_payload_sha256":original["payload_sha256"]!, "trial_payload_sha256":trial["payload_sha256"]!, "pre_roll_count":5, "trial_frame_limit":limit, "perch_was_relocated_at_restart":true]
let payload = try JSONSerialization.data(withJSONObject:tp, options:[.sortedKeys])
let hash = SHA256.hash(data:payload).map { String(format:"%02x",$0) }.joined()
let output = try JSONSerialization.data(withJSONObject:["schema":"numi.crow-replay.v1","payload_sha256":hash,"payload":tp],options:[.sortedKeys])
try output.write(to:URL(fileURLWithPath:args[3]), options:.atomic)
print("presentation splice: \(frames.count) frames, 5 pre-roll, payload \(hash)")
