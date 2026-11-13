#!/bin/bash
set -euo pipefail
repo=/Users/n/Developer/numi-crow-rl-20261107
renderer=/Users/n/Developer/numi-crow-render-two-segment-20261111
out="$repo/artifacts/crow-v15-passive-aero-failure-video-20261113"
replay="$repo/build/crow-v15-energy-consistent-aero-20261113/push-wing-20261126-rebound-passive-replay.json"
mkdir -p "$out"
cd "$renderer"
for view in side front; do
 if [ "$view" = side ]; then yaw=-1.25; else yaw=0.0; fi
 CROW_WORLD_UP_CAMERA=1 .build/arm64-apple-macosx/release/birdflow-capture --capture-crow-frames "$out/audit-$view" --capture-crow-video "$out/$view.mp4" --capture-crow-video-fps 50 --capture-frames 400 --capture-width 1280 --capture-height 720 --capture-crow-solid-course --capture-crow-research-narrow-perch --capture-crow-research-toe-hinges --capture-crow-presentation takeoff --capture-crow-replay-pack "$replay" --capture-crow-lighting overcast --capture-crow-camera-yaw "$yaw" --capture-crow-camera-pitch 0.18 --capture-crow-camera-distance 1.60 > "$out/$view.log" 2>&1
done
shasum -a 256 "$replay" .build/arm64-apple-macosx/release/birdflow-capture "$out/side.mp4" "$out/front.mp4" > "$out/SHA256SUMS"
git rev-parse HEAD > "$out/renderer-commit.txt"
