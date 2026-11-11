#!/bin/bash
set -euo pipefail
cd /Users/n/Developer/numi-crow-render-two-segment-20261111
out=/tmp/crow-latest-failures-20261111
capture() {
  name=$1; first=$2; last=$3
  CROW_WORLD_UP_CAMERA=1 .build/arm64-apple-macosx/release/birdflow-capture --capture-crow-frames "$out/$name-upright-audit" --capture-crow-video "$out/$name-upright.mp4" --capture-crow-video-fps 50 --capture-frames 5000 --capture-frame-start "$first" --capture-frame-end "$last" --capture-width 1280 --capture-height 720 --capture-crow-solid-course --capture-crow-research-narrow-perch --capture-crow-research-toe-hinges --capture-crow-presentation takeoff --capture-crow-replay-pack "$out/$name.json" --capture-crow-lighting overcast --capture-crow-camera-yaw -1.25 --capture-crow-camera-pitch 0.20 --capture-crow-camera-distance 1.05
}
capture taper 3900 4128
capture recovery 3719 3841
