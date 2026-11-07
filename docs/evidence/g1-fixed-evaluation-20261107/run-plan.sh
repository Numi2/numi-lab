#!/bin/sh
set -u
base=/Users/numi/numi-heldout-g1-20261107
build=/Users/numi/numi-lab-heldout-old-build-20261107
pack=/Users/numi/numi-human-resting-integration-20261005/numi-lab/artifacts/g1-legs-locomotion-60min-20260805/retained-policy/candidate.policypack
bin=$build/bin/metalrobo_task_rollout
lib=$build/shaders/MetalRobo.metallib
run() {
  label=$1
  shift
  "$bin" --envs 8 --steps 1000 --repeats 1 --chunk 1 \
    --physics-substeps 4 --velocity-iterations 4 --final-velocity-iterations 2 \
    --scene ground --task g1-legs-locomotion --metallib "$lib" "$@" \
    > "$base/$label.stdout" 2> "$base/$label.stderr"
  result=$?
  printf '%s\n' "$result" > "$base/$label.exit"
  printf '%s exit=%s\n' "$label" "$result"
}
run policy-4081007 --seed 4081007 --policy-pack "$pack"
run policy-4081008 --seed 4081008 --policy-pack "$pack"
run zero-4081007 --seed 4081007 --zero-actions
run projectile-4081007 --seed 4081007 --policy-pack "$pack" --ball "-2,0,.8,3,0,0,.1,.1,100"
