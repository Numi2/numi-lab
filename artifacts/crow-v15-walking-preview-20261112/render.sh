#!/bin/bash
set -euo pipefail
repo=/Users/n/Developer/numi-crow-rl-20261107
renderer=/Users/n/Developer/numi-crow-render-two-segment-20261111
out="$repo/artifacts/crow-v15-walking-preview-20261112"
tmp=$(mktemp -d /tmp/crow-walk-preview.XXXXXX)
trap 'rm -rf "$tmp"' EXIT
cd "$renderer"
for seed in 20261126 20261127; do
 if [[ "$seed" == 20261126 ]]; then input="$repo/artifacts/crow-v15-toe-preload-second-seed-20261112/candidate-20261126-rebound-low/replay.json.gz"; else input="$repo/artifacts/crow-v15-toe-preload-matched-trial-20261112/candidate-20261127-rebound-low/replay.json.gz"; fi
 gzip -dc "$input" > "$tmp/replay-$seed.json"
 CROW_WORLD_UP_CAMERA=1 .build/arm64-apple-macosx/release/birdflow-capture --capture-crow-frames "$out/audit-$seed" --capture-crow-video "$out/walking-$seed.mp4" --capture-crow-video-fps 50 --capture-frames 300 --capture-frame-start 0 --capture-frame-end 299 --capture-width 1280 --capture-height 720 --capture-crow-solid-course --capture-crow-research-narrow-perch --capture-crow-research-toe-hinges --capture-crow-presentation takeoff --capture-crow-replay-pack "$tmp/replay-$seed.json" --capture-crow-lighting overcast --capture-crow-camera-yaw -1.25 --capture-crow-camera-pitch 0.20 --capture-crow-camera-distance 1.05
 done
