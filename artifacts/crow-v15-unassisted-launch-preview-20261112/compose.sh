#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
python3 labels.py
ffmpeg -v error -i ground.mp4 -i earlywing.mp4 -loop 1 -i labels.png -filter_complex '[0:v]scale=640:360[l];[1:v]scale=640:360[r];[l][r]hstack=inputs=2[b];[b][2:v]overlay=shortest=1[v]' -map '[v]' -an -c:v libx264 -crf 20 -preset medium -movflags +faststart -y launch-comparison.mp4
