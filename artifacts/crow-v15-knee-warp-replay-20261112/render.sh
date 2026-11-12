#!/bin/bash
set -euo pipefail
repo=/Users/n/Developer/numi-crow-rl-20261107
renderer=/Users/n/Developer/numi-crow-render-two-segment-20261111
out="$repo/artifacts/crow-v15-knee-warp-replay-20261112"
tmp=$(mktemp -d /tmp/crow-gait-preview.XXXXXX)
trap 'rm -rf "$tmp"' EXIT
baseline="$repo/artifacts/crow-v15-knee-phase-advance-screen-20261112/candidate-20261127-rebound-baseline/replay.json.gz"
candidate="$repo/artifacts/crow-v15-knee-phase-advance-screen-20261112/candidate-20261127-rebound-phase/replay.json.gz"
gzip -dc "$baseline" > "$tmp/baseline.json"
gzip -dc "$candidate" > "$tmp/candidate.json"
cd "$renderer"
for trial in baseline candidate; do
 for angle in side front; do
  if [[ "$angle" == side ]]; then yaw=-1.25; else yaw=-0.05; fi
  CROW_WORLD_UP_CAMERA=1 .build/arm64-apple-macosx/release/birdflow-capture --capture-crow-frames "$out/audit-$trial-$angle" --capture-crow-video "$out/$trial-$angle.mp4" --capture-crow-video-fps 50 --capture-frames 300 --capture-frame-start 200 --capture-frame-end 299 --capture-width 1280 --capture-height 720 --capture-crow-solid-course --capture-crow-research-narrow-perch --capture-crow-research-toe-hinges --capture-crow-presentation takeoff --capture-crow-replay-pack "$tmp/$trial.json" --capture-crow-lighting overcast --capture-crow-camera-yaw "$yaw" --capture-crow-camera-pitch 0.18 --capture-crow-camera-distance 1.6
done
done
