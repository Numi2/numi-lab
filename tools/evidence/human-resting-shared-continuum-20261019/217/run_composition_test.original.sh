#!/bin/bash
set -euo pipefail
stage=/Users/numi/numi-human-separation-20261018/C/shared-world-source-217
repo=/Users/numi/numi-lab-native-replay-publish-20261017
libdir=/Users/numi/numi-human-separation-20261018/C/coupling-adapter-composition-214/build
brain=/Users/numi/numi-human-resting-integration-20261005/numi-brain/Sources/NumiBrainABI/include
network=/Users/numi/numi-human-resting-evidence-20261005/reference-circulation-001/resting_reference_lv15.native.v3.json
material=$repo/matter/materials/blood_high_shear_newtonian_proxy.nmatter
fem_material=/Users/numi/numi-human-separation-20261018/C/fem-mpm-finalizer-213/source/fem_rollback_history.nmatter
mkdir -p "$stage/build"
ln -sf "$libdir/libmetalrobo-214.dylib" "$stage/build/libmetalrobo.dylib"
/usr/bin/clang++ \
  -I"$stage/matter/tools" -I"$stage/matter/src" \
  -I"$repo/matter/src" -I"$repo/src/metal" -I"$repo/apps" \
  -I"$repo/include" -I"$repo/matter/include" -I"$repo/src" \
  -I"$brain" \
  -x objective-c++ -std=c++23 -arch arm64 -fobjc-arc \
  -Wall -Wextra -Wpedantic \
  -L"$libdir" -lmetalrobo-214 \
  -Wl,-rpath,"$libdir" \
  -framework Foundation -framework Metal \
  "$stage/tests/resting_world_source_composition.mm" \
  -o "$stage/build/resting-world-source-composition-test-214"
DYLD_LIBRARY_PATH="$stage/build" "$stage/build/resting-world-source-composition-test-214"
DYLD_LIBRARY_PATH="$stage/build" "$stage/build/resting-world-source-composition-test-214" \
  "$network" "$material" "$fem_material"
