#!/bin/bash
set -euo pipefail
out=/tmp/crow-current-development-video-20261112
python3 labels.py
/opt/homebrew/bin/ffmpeg -hide_banner -loglevel error -y -i "$out/takeoff.mp4" -i "$out/landing.mp4" -i "$out/label-0.png" -i "$out/label-1.png" -filter_complex '[0:v]setpts=2*PTS[slow];[slow][2:v]overlay=0:0[a];[1:v][3:v]overlay=0:0[b];[a][b]concat=n=2:v=1:a=0[v]' -map '[v]' -an -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart "$out/crow-current-development.mp4"
