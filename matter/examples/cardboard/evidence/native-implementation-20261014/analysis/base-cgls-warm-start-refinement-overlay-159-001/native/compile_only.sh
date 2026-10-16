#!/bin/bash
set -euo pipefail
PACKAGE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${PACKAGE_DIR}/native/build"
SOURCE_ROOT="/Users/home/cardboard-implementation-20261014/snapshots/source-right-grip-telemetry-157"
mkdir -p "${BUILD_DIR}"
xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math \
  -I "${SOURCE_ROOT}/matter/include" \
  -I "${SOURCE_ROOT}/matter/src" \
  -I "${PACKAGE_DIR}/candidate/matter/src/metal" \
  -I "${SOURCE_ROOT}/matter/src/metal" \
  -I "${SOURCE_ROOT}/include" \
  -c "${PACKAGE_DIR}/native/warm_start_check.metal" \
  -o "${BUILD_DIR}/warm_start_check.air"
xcrun -sdk macosx metallib "${BUILD_DIR}/warm_start_check.air" \
  -o "${BUILD_DIR}/WarmStartCheck.metallib"
clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Wpedantic -Werror \
  -I "${SOURCE_ROOT}/matter/include" \
  -DNUMI_WARM_START_METALLIB="\"${BUILD_DIR}/WarmStartCheck.metallib\"" \
  -fsyntax-only "${PACKAGE_DIR}/native/warm_start_check.mm"
