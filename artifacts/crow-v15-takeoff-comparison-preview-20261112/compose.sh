#!/bin/bash
set -euo pipefail
p=/Users/home/Documents/Codex/2026-11-07/c/crow-takeoff-comparison-20261112
python3 "$p/labels.py"
ffmpeg -v error -i "$p/before.mp4" -i "$p/after.mp4" -loop 1 -i "$p/labels.png" -filter_complex '[0:v]scale=640:360[l];[1:v]scale=640:360[r];[l][r]hstack=inputs=2[b];[b][2:v]overlay=shortest=1[v]' -map '[v]' -an -c:v libx264 -crf 19 -preset medium -movflags +faststart -y "$p/takeoff-comparison.mp4"
