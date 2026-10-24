#!/bin/zsh
set -eu
SCENE_DIR=/Users/n/numi-plastic-bag-franka-handoff-20261024
cd "$SCENE_DIR"
mkdir -p build run
xcrun -sdk macosx metal -std=metal3.1 -O3 -I source -I include -c source/PlasticBag.metal -o build/PlasticBag.air
xcrun -sdk macosx metallib build/PlasticBag.air -o build/PlasticBag.metallib
clang++ -std=c++20 -O3 -DNDEBUG -fobjc-arc -Wall -Wextra -I source -I include -DPLASTIC_BAG_METALLIB='"/Users/n/numi-plastic-bag-franka-handoff-20261024/build/PlasticBag.metallib"' source/plastic_bag_coins_scene.mm -framework Foundation -framework Metal -o build/numi_classic_bag_handoff
shasum -a 256 source/* include/metalrobo/gpu_types.h build/numi_classic_bag_handoff build/PlasticBag.metallib > run/runtime_sha256.txt
