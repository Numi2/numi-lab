#!/bin/bash
set -eu
cd /Users/n/Developer/numi-crow-rl-20261107
mkdir -p build/crow-v15-microstep-contact-observer-20261110
xcrun clang++ -std=c++20 -O2 -I include -I artifacts/crow-v15-wing-inertia-prototype-20261110 artifacts/crow-v15-microstep-contact-observer-20261110/perch_probe.cpp -L build/lib -lmetalrobo -Wl,-rpath,/Users/n/Developer/numi-crow-rl-20261107/build/lib -o build/crow-v15-microstep-contact-observer-20261110/probe
