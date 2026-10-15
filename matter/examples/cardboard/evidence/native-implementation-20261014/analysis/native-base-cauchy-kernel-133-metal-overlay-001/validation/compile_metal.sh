#!/bin/sh
set -eu
: "${SOURCE_ROOT:?set SOURCE_ROOT to the prepared source-125 overlay tree}"
PKG_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT_DIR=${OUT_DIR:-"$PKG_DIR/validation/compile"}
mkdir -p "$OUT_DIR"
for OPT in 0 3; do
  AIR="$OUT_DIR/matter-O$OPT.air"
  LOG="$OUT_DIR/matter-O$OPT.log"
  {
    printf '%s\n' "xcrun -sdk macosx metal -std=metal4.0 -O$OPT -fno-fast-math -I $SOURCE_ROOT/matter/include -I $SOURCE_ROOT/include -I $SOURCE_ROOT/matter/src -c $SOURCE_ROOT/matter/src/matter_kernels.metal -o $AIR"
    xcrun -sdk macosx metal -std=metal4.0 "-O$OPT" -fno-fast-math \
      -I "$SOURCE_ROOT/matter/include" -I "$SOURCE_ROOT/include" -I "$SOURCE_ROOT/matter/src" \
      -c "$SOURCE_ROOT/matter/src/matter_kernels.metal" -o "$AIR"
    shasum -a 256 "$AIR"
  } >"$LOG" 2>&1
done
