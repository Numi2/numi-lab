#!/bin/bash
set -euo pipefail
repo=/Users/n/Developer/numi-crow-rl-20261107
renderer=/Users/n/Developer/numi-crow-render-two-segment-20261111
out=/tmp/crow-current-development-video-20261112
mkdir -p "$out"
cd "$renderer"
CROW_WORLD_UP_CAMERA=1 .build/arm64-apple-macosx/release/birdflow-capture --capture-crow-frames "$out/takeoff-audit" --capture-crow-video "$out/takeoff.mp4" --capture-crow-video-fps 50 --capture-frames 500 --capture-frame-start 329 --capture-frame-end 370 --capture-width 1280 --capture-height 720 --capture-crow-solid-course --capture-crow-research-narrow-perch --capture-crow-research-toe-hinges --capture-crow-presentation takeoff --capture-crow-replay-pack "$repo/build/crow-v15-overlap-takeoff-screen-20261112/autonomous-20261126-replay.json" --capture-crow-lighting overcast --capture-crow-camera-yaw -1.25 --capture-crow-camera-pitch 0.20 --capture-crow-camera-distance 1.05
CROW_WORLD_UP_CAMERA=1 .build/arm64-apple-macosx/release/birdflow-capture --capture-crow-frames "$out/landing-audit" --capture-crow-video "$out/landing.mp4" --capture-crow-video-fps 50 --capture-frames 5000 --capture-frame-start 4000 --capture-frame-end 4282 --capture-width 1280 --capture-height 720 --capture-crow-solid-course --capture-crow-research-narrow-perch --capture-crow-research-toe-hinges --capture-crow-presentation takeoff --capture-crow-replay-pack "$repo/build/crow-v15-early-wing-sequence-screen-20261112/autonomous-20261126-replay.json" --capture-crow-lighting overcast --capture-crow-camera-yaw -1.25 --capture-crow-camera-pitch 0.20 --capture-crow-camera-distance 1.05
cd "$repo"
python3 artifacts/crow-v15-current-development-video-20261112/labels.py
/opt/homebrew/bin/ffmpeg -hide_banner -loglevel error -y -i "$out/takeoff.mp4" -i "$out/landing.mp4" -i "$out/label-0.png" -i "$out/label-1.png" -filter_complex '[0:v]setpts=2*PTS[slow];[slow][2:v]overlay=0:0[a];[1:v][3:v]overlay=0:0[b];[a][b]concat=n=2:v=1:a=0[v]' -map '[v]' -an -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart "$repo/artifacts/crow-v15-current-development-video-20261112/crow-current-development.mp4"
