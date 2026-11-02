#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CXX="${CXX:-/usr/bin/clang++}"
EVIDENCE="$ROOT/tools/evidence/current-carrier-witness-stream-032"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/current-carrier-witness-032.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$EVIDENCE"

FLAGS=(
  -std=c++20
  -O2
  -Wall
  -Wextra
  -Werror
  -I"$ROOT/apps"
  -I"$ROOT/include"
  -I"$ROOT/matter/include"
)

"$CXX" "${FLAGS[@]}" -MMD -MF "$EVIDENCE/gzip-check.d" \
  -MT current-carrier-witness-gzip-check-032 \
  "$ROOT/tests/CurrentCarrierWitnessGzipCheck_032.cpp" \
  -lz -o "$TMP/gzip-check"
"$CXX" "${FLAGS[@]}" -MMD -MF "$EVIDENCE/stream-check.d" \
  -MT current-carrier-witness-stream-check-032 \
  "$ROOT/tests/CurrentCarrierWitnessStreamBindingCheck_032.cpp" \
  -lz -o "$TMP/stream-check"

"$TMP/gzip-check" "$TMP/gzip-fixtures" \
  > "$EVIDENCE/gzip-check.stdout.txt" 2> "$EVIDENCE/gzip-check.stderr.txt"
"$TMP/stream-check" \
  > "$EVIDENCE/stream-check.stdout.txt" 2> "$EVIDENCE/stream-check.stderr.txt"

cat "$EVIDENCE/gzip-check.stdout.txt"
cat "$EVIDENCE/stream-check.stdout.txt"
