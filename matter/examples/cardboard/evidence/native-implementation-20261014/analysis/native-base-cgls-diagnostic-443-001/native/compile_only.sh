#!/bin/bash
set -euo pipefail
PACKAGE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${PACKAGE_DIR}/native/build"
mkdir -p "${BUILD_DIR}"
xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math \
  -I "${PACKAGE_DIR}/source/matter/src/metal" \
  -c "${PACKAGE_DIR}/native/base_cgls_test_actions.metal" \
  -o "${BUILD_DIR}/base_cgls_test_actions.air"
xcrun -sdk macosx metallib \
  "${BUILD_DIR}/base_cgls_test_actions.air" \
  -o "${BUILD_DIR}/BaseCGLSCheck.metallib"
clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Wpedantic -Werror \
  -DNUMI_BASE_CGLS_CHECK_METALLIB="\"${BUILD_DIR}/BaseCGLSCheck.metallib\"" \
  -fsyntax-only "${PACKAGE_DIR}/native/base_cgls_native_check.mm"
