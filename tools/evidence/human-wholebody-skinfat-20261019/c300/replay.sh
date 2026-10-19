#!/bin/sh
set -eu
ROOT=/Users/n/numi-human-separation-20261018/C/wholebody-closed-boundary-audit-300
C296=/Users/n/numi-human-separation-20261018/C/wholebody-skin-layer-fitting-296
NUMLAB=/Users/n/numi-lab-native-replay-publish-20261017
CANDIDATE=$ROOT/input/candidate.txt
REGISTERED=$ROOT/input/registered-skin-mechanical.csv
LAYERS=${1:?usage: replay.sh /absolute/path/to/admitted-layer-coordinates.csv /absolute/path/to/output-dir}
OUT=${2:?usage: replay.sh /absolute/path/to/admitted-layer-coordinates.csv /absolute/path/to/output-dir}
CANDIDATE_SHA=909d364267a09340ecd2038ed901efb6732b5d3b5a525f65221dbccc3cd40526
REGISTERED_SHA=c48e4d9076d90c6f8f4d308918e1ce6925739d1ee5e50b5affb234e8e54d4eb5
LAYERS_SHA=$(shasum -a 256 "$LAYERS" | awk {print })
clang++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror \
  -I"$ROOT/input" -I"$C296/apps" -I"$NUMLAB/apps" \
  -I"$NUMLAB/matter/include" -I"$NUMLAB/include" \
  "$ROOT/publication-final-032c/auditor.cpp" -o "$ROOT/build/compact_boundary_audit-publication"
"$ROOT/build/compact_boundary_audit-publication" \
  "$CANDIDATE" "$LAYERS" "$REGISTERED" "$OUT" \
  "$CANDIDATE_SHA" "$LAYERS_SHA" "$REGISTERED_SHA"
