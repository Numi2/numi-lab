#!/bin/sh
set -eu
REPO=/Users/n/numi-human-aortic-contact-opt-20261019
E="$REPO/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/arm-actuation-preparation-001/barycentric-embedding-owner-001"
cd "$REPO"
xcrun clang++ -std=c++20 -Wall -Wextra -Wpedantic -Werror \
  -MMD -MF "$E/test_source_embedding.d" \
  -I"$E/clean-head-apps" -Iapps -Iinclude -Imatter/include \
  "$E/test_source_embedding.cpp" -o "$E/test_source_embedding"
"$E/test_source_embedding" "$E/current-075-missing-palm-fixture.tsv"
