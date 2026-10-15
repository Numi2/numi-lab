#!/bin/sh
set -eu
base='/Users/home/cardboard-implementation-20261014/snapshots/source-stable-free-graph-and-general-arithmetic-076'
overlay='/Users/home/cardboard-implementation-20261014/analysis/stateless-generic-compensated-overlay-077'
build='/Users/home/numi-cardboard-20261006/build-cardboard-provenance-076/matter'
cxx='/usr/bin/c++'

"$cxx" -O3 -DNDEBUG -std=c++2b -arch arm64 -fPIC -Wall -Wextra -Wpedantic -Werror \
  -I"$base/matter/include" \
  -c "$overlay/overlay/matter/src/constitutive_impl.cpp" \
  -o "$overlay/evidence/constitutive_impl_overlay.o"
"$cxx" -O3 -DNDEBUG -std=c++2b -arch arm64 -fPIC -Wall -Wextra -Wpedantic -Werror \
  -I"$base/matter/include" -I"$base/matter/src" \
  -c "$overlay/overlay/matter/src/validation_impl.cpp" \
  -o "$overlay/evidence/validation_impl_overlay.o"
"$cxx" -O3 -DNDEBUG -std=c++2b -arch arm64 -fPIE -Wall -Wextra -Wpedantic -Werror \
  -I"$base/include" -I"$base/matter/include" -I"$base/matter/tools" \
  -c "$overlay/overlay/matter/tools/cardboard_generic_compensated_projection_check.cpp" \
  -o "$overlay/evidence/generic_compensated_projection_check_overlay.o"
"$cxx" -O3 -DNDEBUG -arch arm64 -Wl,-search_paths_first -Wl,-headerpad_max_install_names \
  "$overlay/evidence/generic_compensated_projection_check_overlay.o" \
  "$overlay/evidence/constitutive_impl_overlay.o" \
  "$overlay/evidence/validation_impl_overlay.o" \
  -o "$overlay/evidence/generic-compensated-overlay-check" \
  "$build/libnumi_matter_compiler.a"
"$overlay/evidence/generic-compensated-overlay-check" "$base/matter/materials"

"$cxx" -O3 -DNDEBUG -arch arm64 -Wl,-search_paths_first -Wl,-headerpad_max_install_names \
  "$overlay/evidence/generic_compensated_projection_check_overlay.o" \
  -o "$overlay/evidence/frozen076-negative-control" \
  "$build/libnumi_matter_compiler.a"
set +e
"$overlay/evidence/frozen076-negative-control" "$base/matter/materials"
status=$?
set -e
if [ "$status" -ne 1 ]; then
  echo "baseline negative control expected exit 1, got $status" >&2
  exit 2
fi
