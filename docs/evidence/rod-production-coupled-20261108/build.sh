#!/bin/sh
set -eu
ROOT=/Users/numi/numi-lab-tracker-step-rollback-20261108
BUILD=/Users/numi/numi-tracker-step-rollback-build-20261108
OUT=/Users/numi/rod-production-coupled-20261108
mkdir -p "$OUT"
cmake --build "$BUILD" --target metalrobo -j 2
clang++ -std=c++20 -O2 -fobjc-arc -I "$ROOT/include" -I "$ROOT/matter/include" "$ROOT/docs/evidence/rod-production-coupled-20261108/probe.mm" -L "$BUILD/lib" -lmetalrobo -Wl,-rpath,"$BUILD/lib" -framework Foundation -framework Metal -o "$OUT/probe"
