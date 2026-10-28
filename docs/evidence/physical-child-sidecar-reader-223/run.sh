#!/bin/sh
set -eu
ROOT=/Users/numi/Work/skin-fat-native-integration-209
INPUT=/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/volume-worker/registered-surface-sidecar-222
OUT=/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/P/native-child-sidecar-reader-223
/usr/bin/c++ -I"$ROOT/apps" -I"$ROOT/include" -I"$ROOT/matter/include" -std=c++23 -O1 -Wall -Wextra -Wpedantic -Werror "$OUT/reader-check.cpp" -o "$OUT/reader-check" >"$OUT/compile.stdout.txt" 2>"$OUT/compile.stderr.txt"
/usr/bin/time -p "$OUT/reader-check" "$INPUT" >"$OUT/run.stdout.txt" 2>"$OUT/run.stderr.txt"
