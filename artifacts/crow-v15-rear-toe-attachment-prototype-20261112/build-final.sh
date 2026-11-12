#!/bin/sh
set -eu
A=artifacts/crow-v15-rear-toe-attachment-prototype-20261112
B=build/crow-v15-rear-toe-attachment-prototype-20261112
L=build/crow-v15-hip-roll-journey-integration-20261111
for NAME in probe structure; do
 xcrun clang++ -std=c++20 -O2 -Iinclude -Ibuild/crow-v15-wide-contact-isolation-20261111/include "$A/$NAME.cpp" -L"$L" -lmetalrobo -Wl,-rpath,"$(pwd)/$L" -o "$B/$NAME"
 "$B/$NAME"
done
