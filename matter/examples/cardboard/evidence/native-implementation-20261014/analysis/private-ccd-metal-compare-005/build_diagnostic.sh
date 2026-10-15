#!/bin/bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 SOURCE_099_ROOT OUTPUT_DIR CASES_FLOAT4_BIN" >&2
  exit 2
fi
SOURCE_ROOT="$(cd "$1" && pwd -P)"
OUTPUT_DIR="$2"
CASES_BIN="$3"
PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd -P)"
mkdir -p "$OUTPUT_DIR"

python3 "$PACKAGE_DIR/generate_metal.py" \
  --source-root "$SOURCE_ROOT" --output-dir "$OUTPUT_DIR"

xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math \
  -I "$SOURCE_ROOT/matter/src" \
  -I "$SOURCE_ROOT/matter/include" \
  -I "$SOURCE_ROOT/include" \
  -c "$OUTPUT_DIR/private_ee_ccd_diagnostic.metal" \
  -o "$OUTPUT_DIR/private_ee_ccd_diagnostic.air"
xcrun -sdk macosx metallib "$OUTPUT_DIR/private_ee_ccd_diagnostic.air" \
  -o "$OUTPUT_DIR/private_ee_ccd_diagnostic.metallib"

xcrun clang++ -std=c++20 -fobjc-arc \
  -framework Foundation -framework Metal \
  "$PACKAGE_DIR/run_native.mm" -o "$OUTPUT_DIR/run_native"

echo "Metal diagnostic compiled without dispatching GPU work: $OUTPUT_DIR/private_ee_ccd_diagnostic.metallib"
echo "Run only when the GPU worker is explicitly available:"
echo "  $OUTPUT_DIR/run_native --metallib $OUTPUT_DIR/private_ee_ccd_diagnostic.metallib --cases $CASES_BIN --cases-manifest $(dirname "$CASES_BIN")/captured-ee-cases.json --build-manifest $OUTPUT_DIR/build-manifest.json"
