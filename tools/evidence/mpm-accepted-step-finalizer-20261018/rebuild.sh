#!/bin/bash
set -euo pipefail
root=/Users/numi/numi-human-separation-20261018/C/external-mpm-finalizer-211
src=/Users/numi/numi-human-separation-20261018/C/current-operator-integration-207
repo=/Users/numi/numi-lab-native-replay-publish-20261017
mat=/Users/numi/numi-human-separation-20261018/C/bleeding-liquid-205/main-owner-206/matter/materials/blood_high_shear_newtonian_proxy.nmatter
clang++ -std=c++23 -O3 -arch arm64 -fobjc-arc -x objective-c++ -I"/Users/n/numi-human-separation-20261018/C/blood-transfer-208/matter/include" -I"$src/full-source/include" -I"$src/full-source/matter/include" -I"$repo/include" -I"$repo/matter/include" -DNUMI_MATTER_METALLIB=\"$src/shaders/MatterSource.metal\" -DNUMI_MATTER_MATERIAL=\"$mat\" -c "$root/source/finalizer.mm" -o "$root/build/finalizer.o" >"$root/build/compile-002.log" 2>&1
clang++ -arch arm64 "$root/build/finalizer.o" -L"/Users/n/numi-human-separation-20261018/C/blood-transfer-208/build" -lmetalrobo -Wl,-rpath,"/Users/n/numi-human-separation-20261018/C/blood-transfer-208/build" -framework Foundation -framework Metal -framework Accelerate -lz -o "$root/build/finalizer" >"$root/build/link-002.log" 2>&1
shasum -a 256 "$root/source/finalizer.mm" "$root/build/finalizer.o" "$root/build/finalizer"
