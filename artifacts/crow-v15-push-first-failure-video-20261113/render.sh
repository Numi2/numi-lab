#!/bin/bash
set -euo pipefail
repo=/Users/n/Developer/numi-crow-rl-20261107
out="$repo/artifacts/crow-v15-push-first-failure-video-20261113"
renderer=/Users/n/Developer/numi-crow-render-two-segment-20261111
mkdir -p "$out"
tmp=$(mktemp -d /tmp/crow-pushfirst-video.XXXXXX)
trap 'rm -rf "$tmp"' EXIT
gzip -dc "$repo/artifacts/crow-v15-push-first-faststroke-20261113/push-wing-20261126-rebound-pushfirst/replay.json.gz" > "$tmp/replay.json"
cd "$renderer"
for angle in side front; do
 yaw=-1.25
 if [ "$angle" = front ]; then yaw=-0.05; fi
 CROW_WORLD_UP_CAMERA=1 .build/arm64-apple-macosx/release/birdflow-capture \
  --capture-crow-frames "$out/audit-$angle" --capture-crow-video "$out/$angle.mp4" \
  --capture-crow-video-fps 50 --capture-frames 400 --capture-frame-start 0 --capture-frame-end 399 \
  --capture-width 1280 --capture-height 720 --capture-crow-solid-course \
  --capture-crow-research-narrow-perch --capture-crow-research-toe-hinges \
  --capture-crow-presentation takeoff --capture-crow-replay-pack "$tmp/replay.json" \
  --capture-crow-lighting overcast --capture-crow-camera-yaw "$yaw" \
  --capture-crow-camera-pitch 0.18 --capture-crow-camera-distance 1.6 > "$out/$angle.log" 2>&1
 printf 'Completed %s\n' "$angle"
done
