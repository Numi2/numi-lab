#!/bin/bash
set -euo pipefail
cd /tmp/crow-video-compose-20261111
/opt/homebrew/bin/ffmpeg -hide_banner -loglevel error -y \
 -i walk-reference.mp4 -i walk-reduced.mp4 -i taper.mp4 -i recovery.mp4 \
 -i label-0.png -i label-1.png -i label-2.png -i label-3.png \
 -filter_complex '[0:v][4:v]overlay=0:0[a];[1:v][5:v]overlay=0:0[b];[2:v][6:v]overlay=0:0[c];[3:v][7:v]overlay=0:0[d];[a][b][c][d]concat=n=4:v=1:a=0[v]' \
 -map '[v]' -an -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart /Users/home/Documents/Codex/2026-11-07/c/crow-latest-development-runs.mp4
