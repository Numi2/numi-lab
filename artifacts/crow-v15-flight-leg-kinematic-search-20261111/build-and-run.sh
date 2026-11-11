#!/bin/bash
set -euo pipefail
root=/Users/n/Developer/numi-crow-rl-20261107
cd "$root"
mkdir -p build/crow-v15-flight-leg-kinematic-search-20261111
clang++ -std=c++20 -O2 -Iinclude -Isrc artifacts/crow-v15-flight-leg-kinematic-search-20261111/probe.cpp -Lbuild/crow-v15-hip-roll-journey-integration-20261111 -lmetalrobo -Wl,-rpath,/Users/n/Developer/numi-crow-rl-20261107/build/crow-v15-hip-roll-journey-integration-20261111 -o build/crow-v15-flight-leg-kinematic-search-20261111/probe
build/crow-v15-flight-leg-kinematic-search-20261111/probe
