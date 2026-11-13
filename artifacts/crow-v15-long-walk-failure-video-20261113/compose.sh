#!/bin/sh
set -eu
python3 labels.py
ffmpeg -v error -y -i lane1-side.mp4 -i lane1-front.mp4 -i lane5-side.mp4 -i lane5-front.mp4 -loop 1 -i labels.png -filter_complex '[0:v]crop=480:270:400:225,scale=640:360[a];[1:v]crop=480:270:400:225,scale=640:360[b];[2:v]crop=480:270:400:225,scale=640:360[c];[3:v]crop=480:270:400:225,scale=640:360[d];[a][b]hstack[top];[c][d]hstack[bottom];[top][bottom]vstack,pad=1280:766:0:0:color=black[base];[base][4:v]overlay=shortest=1[out]' -map '[out]' -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart -t 8 comparison.mp4
