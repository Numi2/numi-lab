#!/bin/bash
set -euo pipefail
bundle_dir="$(cd "$(dirname "$0")/.." && pwd)"
NUMI_W_ROOT="/Users/n/numi-lab-native-replay-publish-20261017"
NUMI_S233_LIBRARY="/Users/n/numi-human-separation-20261018/C/deformable-contact-finalizer-233/build/libmetalrobo.dylib"
expected_base="3059af430d0feaa7ce4af3c59fbc1ba9b04a214c"
expected_s233_sha="71372f858367c5945eed34e7334f184f210f06c7cd21e6ec48cb340a4aebd60a"
expected_material_sha="f9fc009dc6ee8b110d475a87ea3ea3320c02095f07c8944e85b3c7f1d6967d33"
canonical_material="$NUMI_W_ROOT/matter/materials/human_optic_nerve_longitudinal_low_strain_isotropic_effective.nmatter"
bundled_material="$bundle_dir/source/matter/materials/human_optic_nerve_longitudinal_low_strain_isotropic_effective.nmatter"
if ! git -C "$NUMI_W_ROOT" merge-base --is-ancestor "$expected_base" HEAD; then
  echo "Matter checkout is not descended from required base $expected_base" >&2
  exit 2
fi
if [[ ! -f "$canonical_material" ]] || ! cmp -s "$canonical_material" "$bundled_material"; then
  echo "Canonical passive material is missing or differs from this bundle" >&2
  exit 2
fi
actual_material_sha="$(shasum -a 256 "$canonical_material" | cut -d " " -f1)"
if [[ "$actual_material_sha" != "$expected_material_sha" ]]; then
  echo "Canonical material SHA mismatch: $actual_material_sha" >&2
  exit 2
fi
actual_lib_sha="$(shasum -a 256 "$NUMI_S233_LIBRARY" | cut -d " " -f1)"
if [[ "$actual_lib_sha" != "$expected_s233_sha" ]]; then
  echo "S233 library SHA mismatch: $actual_lib_sha" >&2
  exit 2
fi
binary="$(mktemp /tmp/test_nmatter_cpu_compile.XXXXXX)"
trap "rm -f $binary" EXIT
clang++ -std=c++20 -O2 -I"$NUMI_W_ROOT/matter/include" "$bundle_dir/tests/test_nmatter_cpu_compile.cpp" -L"$(dirname "$NUMI_S233_LIBRARY")" -lmetalrobo -Wl,-rpath,"$(dirname "$NUMI_S233_LIBRARY")" -framework Foundation -framework Metal -framework QuartzCore -framework Accelerate -o "$binary"
"$binary" "$bundled_material"
