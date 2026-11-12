#!/bin/bash
set -euo pipefail
p=${1:-.}
python3 "$p/labels.py" "$p"
ffmpeg -v error -i "$p/walking-20261126.mp4" -i "$p/walking-20261127.mp4" -loop 1 -i "$p/label-20261126.png" -loop 1 -i "$p/label-20261127.png" -filter_complex '[0:v]crop=840:472:220:120,scale=1280:720[a];[a][2:v]overlay=0:0:shortest=1[v0];[1:v]crop=840:472:220:120,scale=1280:720[b];[b][3:v]overlay=0:0:shortest=1[v1];[v0][v1]concat=n=2:v=1:a=0[out]' -map '[out]' -c:v libx264 -crf 20 -preset medium -movflags +faststart -y "$p/development-walk-two-seeds.mp4"
