#!/bin/bash
set -euo pipefail
PACKAGE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
"${PACKAGE_DIR}/native/compile_only.sh"
BUILD_DIR="${PACKAGE_DIR}/native/build"
clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Wpedantic -Werror \
  -I "${PACKAGE_DIR}/source/matter/include" \
  -DNUMI_BASE_CGLS_CHECK_METALLIB="\"${BUILD_DIR}/BaseCGLSRightScaleCheck.metallib\"" \
  "${PACKAGE_DIR}/native/base_cgls_identity_check.mm" \
  -framework Foundation -framework Metal \
  -o "${BUILD_DIR}/base_cgls_identity_check"
clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Wpedantic -Werror \
  -I "${PACKAGE_DIR}/source/matter/include" \
  -DNUMI_CGLS_RIGHT_SCALE_METALLIB="\"${BUILD_DIR}/BaseCGLSRightScaleCheck.metallib\"" \
  "${PACKAGE_DIR}/native/right_scale_check.mm" \
  -framework Foundation -framework Metal \
  -o "${BUILD_DIR}/base_cgls_right_scale_check"
