#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
python3 labels.py
ffmpeg -v error -i baseline-side.mp4 -i candidate-side.mp4 -i baseline-front.mp4 -i candidate-front.mp4 -loop 1 -i labels.png -filter_complex '[0:v]crop=640:360:(in_w-out_w)/2:(in_h-out_h)/2[a];[1:v]crop=640:360:(in_w-out_w)/2:(in_h-out_h)/2[b];[2:v]crop=640:360:(in_w-out_w)/2:(in_h-out_h)/2[c];[3:v]crop=640:360:(in_w-out_w)/2:(in_h-out_h)/2[d];[a][b]hstack[top];[c][d]hstack[bottom];[top][bottom]vstack[grid];[grid][4:v]overlay=shortest=1[v]' -map '[v]' -an -c:v libx264 -crf 20 -preset medium -movflags +faststart -y gait-comparison.mp4
