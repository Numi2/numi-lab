#!/bin/bash
set -euo pipefail
repo=/Users/n/Developer/numi-crow-rl-20261107
renderer=/Users/n/Developer/numi-crow-render-two-segment-20261111
out="$repo/artifacts/crow-v15-unassisted-launch-preview-20261112"
tmp=$(mktemp -d /tmp/crow-unassisted-launch-preview.XXXXXX)
trap 'rm -rf "$tmp"' EXIT
cd "$renderer"
for trial in ground earlywing; do
 if [[ "$trial" == ground ]]; then input="$repo/artifacts/crow-v15-launch-body-moment-ablation-20261112/candidate-20261127-rebound-flexed/replay.json.gz"; else input="$repo/artifacts/crow-v15-unassisted-push-wing-screen-20261112/candidate-20261127-rebound-flexed/replay.json.gz"; fi
 gzip -dc "$input" > "$tmp/$trial.json"
 CROW_WORLD_UP_CAMERA=1 .build/arm64-apple-macosx/release/birdflow-capture --capture-crow-frames "$out/audit-$trial" --capture-crow-video "$out/$trial.mp4" --capture-crow-video-fps 50 --capture-frames 500 --capture-frame-start 260 --capture-frame-end 499 --capture-width 1280 --capture-height 720 --capture-crow-solid-course --capture-crow-research-narrow-perch --capture-crow-research-toe-hinges --capture-crow-presentation takeoff --capture-crow-replay-pack "$tmp/$trial.json" --capture-crow-lighting overcast --capture-crow-camera-yaw -1.25 --capture-crow-camera-pitch 0.20 --capture-crow-camera-distance 1.9
done
