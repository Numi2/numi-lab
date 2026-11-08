#!/bin/bash
set -euo pipefail
python3 make-candidate.py
cat Collision.candidate.metal scalar-kernel.metal > Collision.scalar-probe.metal
xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math -I /Users/numi/numi-lab-policy-repair-main-20261108/include -c Collision.scalar-probe.metal -o scalar-probe.air
xcrun -sdk macosx metallib scalar-probe.air -o scalar-probe.metallib
clang++ -std=c++20 -O2 -fobjc-arc -I /Users/numi/numi-lab-policy-repair-main-20261108/include scalar_probe.mm -framework Foundation -framework Metal -o scalar_probe
