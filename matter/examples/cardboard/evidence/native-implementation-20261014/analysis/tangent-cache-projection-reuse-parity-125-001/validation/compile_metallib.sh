#!/bin/zsh
set -euo pipefail
PKG=/Users/home/cardboard-implementation-20261014/analysis/tangent-cache-projection-reuse-parity-125-001
BASE=/Users/home/cardboard-implementation-20261014/snapshots/source-status5-vtccd-diagnostics-125
OUT="$PKG/validation/metallib"
xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math \
  -I "$BASE/matter/include" -I "$BASE/include" -I "$BASE/matter/src" \
  -MMD -MF "$OUT/NumiMatter.d" -MT "$OUT/NumiMatter.metallib" \
  -c "$PKG/prototype/matter/src/matter_kernels.metal" -o "$OUT/NumiMatter.air"
xcrun -sdk macosx metallib "$OUT/NumiMatter.air" -o "$OUT/NumiMatter.metallib"
