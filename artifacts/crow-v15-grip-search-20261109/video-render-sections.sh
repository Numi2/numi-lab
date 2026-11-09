#!/bin/bash
# Render complete v15 accepted-state replays into independently playable
# sections, without generating a large sequence of intermediate PNGs.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  printf "usage: %s OUTPUT_DIRECTORY I_REPLAY_JSON J_REPLAY_JSON\n" "$0" >&2
  exit 2
fi
output=$1
input_i=$2
input_j=$3
renderer=${CROW_CAPTURE_BINARY:-.build/release/birdflow-capture}
frames=${CROW_FRAME_COUNT:-3000}
fps=${CROW_VIDEO_FPS:-50}
section_frames=${CROW_SECTION_FRAMES:-600}
width=${CROW_VIDEO_WIDTH:-1600}
height=${CROW_VIDEO_HEIGHT:-900}
yaw=${CROW_CAMERA_YAW:--1.25}
pitch=${CROW_CAMERA_PITCH:-0.20}
distance=${CROW_CAMERA_DISTANCE:-1.05}
for path in "$input_i" "$input_j" "$renderer"; do
  if [ ! -f "$path" ]; then printf "missing input: %s\n" "$path" >&2; exit 2; fi
done
# Prevent a 50 Hz video from silently skipping or duplicating replay states.
python3 - "$input_i" "$input_j" "$frames" <<CHECK_REPLAY_COUNTS
import json
import sys
expected = int(sys.argv[3])
for name in sys.argv[1:3]:
    with open(name) as stream:
        payload = json.load(stream)["payload"]
    count = len(payload["frames"])
    if count != expected or payload["frame_count"] != count:
        raise SystemExit(f"{name}: expected {expected} replay frames, found {count}")
CHECK_REPLAY_COUNTS
mkdir -p "$output"
for course in i j; do
  if [ "$course" = i ]; then input=$input_i; else input=$input_j; fi
  for ((start=0; start<frames; start+=section_frames)); do
    end=$((start + section_frames - 1))
    if ((end >= frames)); then end=$((frames - 1)); fi
    label=$(printf "%s-%04d" "$course" "$start")
    "$renderer" --capture-crow-frames "$output/$label-audit" \
      --capture-crow-video "$output/$label.mp4" --capture-crow-video-fps "$fps" \
      --capture-frames "$frames" --capture-frame-start "$start" --capture-frame-end "$end" \
      --capture-width "$width" --capture-height "$height" --capture-crow-solid-course \
      --capture-crow-presentation takeoff --capture-crow-replay-pack "$input" \
      --capture-crow-lighting overcast --capture-crow-camera-yaw "$yaw" \
      --capture-crow-camera-pitch "$pitch" --capture-crow-camera-distance "$distance"
  done
done
