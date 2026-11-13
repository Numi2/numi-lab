#!/bin/bash
set -euo pipefail
repo=/Users/n/Developer/numi-crow-rl-20261107
out="$repo/artifacts/crow-v15-long-walk-failure-video-20261113"
renderer=/Users/n/Developer/numi-crow-render-two-segment-20261111
mkdir -p "$out"
tmp=$(mktemp -d /tmp/crow-long-replay.XXXXXX)
trap 'rm -rf "$tmp"' EXIT
gzip -dc "$repo/artifacts/crow-v15-long-walk-failure-replay-20261113/autonomous-20261126-lane1/replay.json.gz" > "$tmp/lane1.json"
gzip -dc "$repo/artifacts/crow-v15-walk-hold-slower-cycle-20261111/autonomous-20261126/replay.json.gz" > "$tmp/lane5.json"
cd "$renderer"
for lane in lane1 lane5; do
 for angle in side front; do
  yaw=-1.25
  if [ "$angle" = front ]; then yaw=-0.05; fi
  printf 'Rendering %s %s\n' "$lane" "$angle"
  CROW_WORLD_UP_CAMERA=1 .build/arm64-apple-macosx/release/birdflow-capture \
   --capture-crow-frames "$out/audit-$lane-$angle" --capture-crow-video "$out/$lane-$angle.mp4" \
   --capture-crow-video-fps 50 --capture-frames 600 --capture-frame-start 200 --capture-frame-end 599 \
   --capture-width 1280 --capture-height 720 --capture-crow-solid-course \
   --capture-crow-research-narrow-perch --capture-crow-research-toe-hinges \
   --capture-crow-presentation takeoff --capture-crow-replay-pack "$tmp/$lane.json" \
   --capture-crow-lighting overcast --capture-crow-camera-yaw "$yaw" \
   --capture-crow-camera-pitch 0.18 --capture-crow-camera-distance 1.6 > "$out/$lane-$angle.log" 2>&1
  printf 'Finished %s %s\n' "$lane" "$angle"
 done
done
