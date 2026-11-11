#!/bin/bash
set -euo pipefail
repo=/Users/n/Developer/numi-crow-rl-20261107
renderer=/Users/n/Developer/numi-crow-render-two-segment-20261111
out=/tmp/crow-current-development-video-20261112
mkdir -p "$out"
cd "$renderer"
CROW_WORLD_UP_CAMERA=1 .build/arm64-apple-macosx/release/birdflow-capture --capture-crow-frames "$out/takeoff-audit" --capture-crow-video "$out/takeoff.mp4" --capture-crow-video-fps 50 --capture-frames 500 --capture-frame-start 329 --capture-frame-end 370 --capture-width 1280 --capture-height 720 --capture-crow-solid-course --capture-crow-research-narrow-perch --capture-crow-research-toe-hinges --capture-crow-presentation takeoff --capture-crow-replay-pack "$repo/build/crow-v15-overlap-takeoff-screen-20261112/autonomous-20261126-replay.json" --capture-crow-lighting overcast --capture-crow-camera-yaw -1.25 --capture-crow-camera-pitch 0.20 --capture-crow-camera-distance 1.05
CROW_WORLD_UP_CAMERA=1 .build/arm64-apple-macosx/release/birdflow-capture --capture-crow-frames "$out/landing-audit" --capture-crow-video "$out/landing.mp4" --capture-crow-video-fps 50 --capture-frames 5000 --capture-frame-start 4000 --capture-frame-end 4282 --capture-width 1280 --capture-height 720 --capture-crow-solid-course --capture-crow-research-narrow-perch --capture-crow-research-toe-hinges --capture-crow-presentation takeoff --capture-crow-replay-pack "$repo/build/crow-v15-early-wing-sequence-screen-20261112/autonomous-20261126-replay.json" --capture-crow-lighting overcast --capture-crow-camera-yaw -1.25 --capture-crow-camera-pitch 0.20 --capture-crow-camera-distance 1.05
