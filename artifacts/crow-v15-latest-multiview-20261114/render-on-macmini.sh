#!/bin/bash
# All five cameras replay the same previously recorded 2,400 states.
set -euo pipefail
REPO=/Users/n/Developer/numi-crow-rl-20261107
RENDERER=/Users/n/Developer/numi-crow-render-two-segment-20261111
OUT=/tmp/crow-48s-multiview-20261114
mkdir -p "$OUT"
gzip -dc "$REPO/artifacts/crow-v15-uniform-handoff-second-seed-20261114/push-20261127-rebound-uniform/replay.json.gz" > "$OUT/replay.json"
cd "$RENDERER"
while read -r view yaw pitch distance; do
  CROW_WORLD_UP_CAMERA=1 .build/arm64-apple-macosx/release/birdflow-capture \
    --capture-crow-frames "$OUT/frames-$view" --capture-crow-video "$OUT/$view.mp4" \
    --capture-crow-video-fps 50 --capture-frames 2400 --capture-width 640 --capture-height 480 \
    --capture-crow-inspection-hide-course --capture-crow-research-narrow-perch \
    --capture-crow-research-toe-hinges --capture-crow-presentation takeoff \
    --capture-crow-replay-pack "$OUT/replay.json" --capture-crow-lighting overcast \
    --capture-crow-camera-yaw "$yaw" --capture-crow-camera-pitch "$pitch" \
    --capture-crow-camera-distance "$distance" > "$OUT/$view.log" 2>&1
done <<'VIEWS'
front-close 0 0.18 1.35
rear-close 3.14159 0.18 1.35
top-close 0 1.32 1.35
below-close 0 -1.32 1.35
hipfeet-close -1.25 0.05 0.68
VIEWS
