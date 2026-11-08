#!/bin/sh
set -eu
ROOT=/Users/numi/numi-lab-tracker-step-rollback-20261108
D="$ROOT/docs/evidence/rod-coupled-metal-resident-20261108"
OUT=/Users/numi/rod-coupled-metal-resident-20261108
mkdir -p "$OUT"
xcrun -sdk macosx metal -std=metal3.0 -fno-fast-math -I "$D" -c "$D/kernel.metal" -o "$OUT/kernel.air"
xcrun -sdk macosx metallib "$OUT/kernel.air" -o "$OUT/kernel.metallib"
clang++ -std=c++20 -O2 -fobjc-arc -I "$D" "$D/resident.mm" -framework Foundation -framework Metal -o "$OUT/resident"
