#!/bin/bash
set -euo pipefail
repo=/Users/numi/numi-lab-native-replay-publish-20261017
root="$repo/tools/evidence/contact-free-fem-mpm-finalizer-20261019"
sourceRoot=/Users/numi/numi-human-separation-20261018/C/current-operator-integration-207
runtimeLib="$(printenv NUMI_FEM_MPM_RUNTIME_LIBRARY || true)"
metalSource="$(printenv NUMI_MATTER_METALLIB || true)"
network="$(printenv NUMI_FEM_MPM_NETWORK || true)"
material="$(printenv NUMI_FEM_MPM_MATERIAL || true)"
out="$(printenv NUMI_FEM_MPM_OUTPUT || true)"
test -n "$runtimeLib" && test -n "$metalSource" && test -n "$network"
test -n "$material" || material="$root/source/fem_rollback_history.nmatter"
test -n "$out" || out="$root/finalizer"
clang++ -std=c++23 -O3 -arch arm64 -fobjc-arc -x objective-c++   -I"/Users/n/numi-human-separation-20261018/C/blood-transfer-208/matter/include"   -I"$sourceRoot/full-source/include" -I"$sourceRoot/full-source/matter/include"   -I"$repo/include" -I"$repo/matter/include"   -DNUMI_MATTER_METALLIB=\"$metalSource\"   -c "$root/source/finalizer.mm" -o "$root/finalizer.o"
clang++ -arch arm64 "$root/finalizer.o" "-L$(dirname "$runtimeLib")" -lmetalrobo   -Wl,-rpath,"$(dirname "$runtimeLib")" -framework Foundation -framework Metal   -framework Accelerate -lz -o "$out"
export NUMI_MATTER_TEST_SOURCE="$metalSource"
export NUMI_MATTER_TEST_SOURCE_SHA256="$(shasum -a 256 "$metalSource" | awk '{print $1}')"
export NUMI_FEM_MPM_NETWORK="$network" NUMI_FEM_MPM_MATERIAL="$material"
export DYLD_LIBRARY_PATH="$(dirname "$runtimeLib")"
exec /Users/numi/numi-human-prep-venv-20261005/bin/python3   /Users/numi/numi-human-final-integration-publish-1397/tools/run_resting_job_exclusive.py -- "$out"
