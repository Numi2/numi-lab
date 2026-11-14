#!/bin/bash
set -euo pipefail
repo=/Users/n/Developer/numi-crow-rl-20261107
renderer=/Users/n/Developer/numi-crow-render-two-segment-20261111
out="$repo/artifacts/crow-v15-handoff-failure-video-corrected-20261114"
replay="$repo/build/crow-v15-handoff-continuity-20261114/push-20261126-rebound-continuity-replay.json"
cd "$renderer"
mkdir -p "$out"
for view in side front; do
 if [ "$view" = side ]; then yaw=-1.25; else yaw=0.0; fi
 CROW_WORLD_UP_CAMERA=1 .build/arm64-apple-macosx/release/birdflow-capture --capture-crow-frames "$out/frames-$view" --capture-crow-video "$out/$view.mp4" --capture-crow-video-fps 50 --capture-frames 800 --capture-frame-start 0 --capture-frame-end 667 --capture-width 1280 --capture-height 720 --capture-crow-solid-course --capture-crow-research-narrow-perch --capture-crow-research-toe-hinges --capture-crow-presentation takeoff --capture-crow-replay-pack "$replay" --capture-crow-lighting overcast --capture-crow-camera-yaw "$yaw" --capture-crow-camera-pitch 0.18 --capture-crow-camera-distance 2.50 > "$out/$view.log" 2>&1
done
shasum -a 256 "$replay" .build/arm64-apple-macosx/release/birdflow-capture "$out/side.mp4" "$out/front.mp4" > "$out/SHA256SUMS"
git rev-parse HEAD > "$out/renderer-commit.txt"
git status --porcelain=v1 > "$out/renderer-status.txt"
