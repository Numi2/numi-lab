#!/bin/bash
set -euo pipefail
bundle_dir="$(cd "$(dirname "$0")/.." && pwd)"
NUMI_W_ROOT="${NUMI_W_ROOT:-/Users/n/numi-lab-native-replay-publish-20261017}"
expected_base="3059af430d0feaa7ce4af3c59fbc1ba9b04a214c"
expected_common_sha="16c90acd50ec4899a5fabd64341a93b0f7b232efce9d14d76b73bd33b1587dc6"
expected_tetgen_sha="1b5d04dd1cf5b5f3d1e83b7e8ff9696ce824c862f245c4b54875efa54fb1ea5e"
asset_dir="$bundle_dir/assets"
if ! git -C "$NUMI_W_ROOT" merge-base --is-ancestor "$expected_base" HEAD; then
  echo "Matter checkout is not descended from required base $expected_base" >&2
  exit 2
fi
for header in NumiHumanCoarseTissuePreparation.hpp NumiHumanTetGenOrganicFEM.hpp; do
  canonical="$NUMI_W_ROOT/apps/$header"
  bundled="$bundle_dir/source/apps/$header"
  if [[ ! -f "$canonical" ]] || ! cmp -s "$canonical" "$bundled"; then
    echo "Canonical W/apps/$header is missing or differs from this bundle" >&2
    exit 2
  fi
done
actual_common_sha="$(shasum -a 256 "$NUMI_W_ROOT/apps/NumiHumanCoarseTissuePreparation.hpp" | awk '{print $1}')"
actual_tetgen_sha="$(shasum -a 256 "$NUMI_W_ROOT/apps/NumiHumanTetGenOrganicFEM.hpp" | awk '{print $1}')"
if [[ "$actual_common_sha" != "$expected_common_sha" || "$actual_tetgen_sha" != "$expected_tetgen_sha" ]]; then
  echo "Canonical helper SHA mismatch: coarse=$actual_common_sha organic=$actual_tetgen_sha" >&2
  exit 2
fi
(
  cd "$bundle_dir"
  shasum -a 256 -c ASSETS.sha256
  shasum -a 256 -c BUNDLE_INPUTS.sha256
)
binary="$(mktemp "${TMPDIR:-/tmp}/test_tetgen_organic_fem_cpu.XXXXXX")"
trap 'rm -f "$binary"' EXIT
clang++ -std=c++20 -O0 \
  -I"$NUMI_W_ROOT/matter/include" \
  -I"$NUMI_W_ROOT/include" \
  -I"$NUMI_W_ROOT/apps" \
  "$bundle_dir/tests/test_tetgen_organic_fem.cpp" \
  -o "$binary"
"$binary" "$asset_dir"
