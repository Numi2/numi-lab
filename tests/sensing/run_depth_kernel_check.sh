#!/bin/sh
set -eu
if [ "$#" -ne 1 ]; then
  echo "usage: $0 build-directory" >&2
  exit 2
fi
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
out=$1
mkdir -p "$out"
xcrun -sdk macosx metal -std=metal3.1 -I "$root/include" -I "$root/matter/include" -c "$root/src/metal/HybridRenderer.metal" -o "$out/HybridRenderer.air"
xcrun -sdk macosx metallib "$out/HybridRenderer.air" -o "$out/HybridRenderer.metallib"
xcrun clang++ -std=c++23 -fobjc-arc -I"$root/include" -I"$root/matter/include" "$root/tests/sensing/depth_kernel_check.mm" -framework Foundation -framework Metal -o "$out/depth_kernel_check"
"$out/depth_kernel_check" "$out/HybridRenderer.metallib"
