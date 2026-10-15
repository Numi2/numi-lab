#!/bin/bash
set -euo pipefail
PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE103_ROOT="${SOURCE103_ROOT:-/Users/home/cardboard-implementation-20261014/snapshots/source-whole-mortar-proof-103}"
OVERLAY_ROOT="$PACKAGE_DIR/source_overlay/matter/src"
COMPILE_DIR="$PACKAGE_DIR/compile"
mkdir -p "$COMPILE_DIR"

{
  echo "SOURCE103_ROOT=$SOURCE103_ROOT"
  echo "OVERLAY_ROOT=$OVERLAY_ROOT"
  echo "PACKAGE_DIR=$PACKAGE_DIR"
  echo "METAL_VERSION=$(xcrun --sdk macosx metal --version 2>&1 | head -1)"
  echo "CLANG_VERSION=$(xcrun --sdk macosx clang++ --version 2>&1 | head -1)"
  xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math \
    -I "$OVERLAY_ROOT" \
    -I "$SOURCE103_ROOT/matter/include" \
    -I "$SOURCE103_ROOT/include" \
    -I "$SOURCE103_ROOT/matter/src" \
    -c "$PACKAGE_DIR/diagnostic_pair.metal" \
    -o "$COMPILE_DIR/diagnostic_pair.air"
  xcrun -sdk macosx metallib "$COMPILE_DIR/diagnostic_pair.air" -o "$COMPILE_DIR/diagnostic_pair.metallib"
  xcrun --sdk macosx clang++ -std=c++20 -fobjc-arc \
    "$PACKAGE_DIR/pair_ccd_native_check.mm" -framework Foundation -framework Metal \
    -o "$COMPILE_DIR/pair_ccd_native_check"
} >"$COMPILE_DIR/build.log" 2>&1
echo "source103_pair_ccd_continuation_compile=PASS"
echo "metallib=$COMPILE_DIR/diagnostic_pair.metallib"
echo "host=$COMPILE_DIR/pair_ccd_native_check"
echo "No native executable was run by build.sh."
