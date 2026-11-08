#!/bin/zsh
set -eu
SCENE_DIR=${0:A:h}
cd "$SCENE_DIR"
mkdir -p build
xcrun -sdk macosx metal -std=metal3.1 -O3 -I source -I include -c source/PlasticBag.metal -o build/PlasticBag.air
xcrun -sdk macosx metallib build/PlasticBag.air -o build/PlasticBag.metallib
clang++ -std=c++20 -O3 -DNDEBUG -fobjc-arc -Wall -Wextra -I source -I include -DPLASTIC_BAG_METALLIB="\"$SCENE_DIR/build/PlasticBag.metallib\"" source/plastic_bag_coins_scene.mm -framework Foundation -framework Metal -o build/numi_bottom_seal_bag
shasum -a 256 source/* include/metalrobo/gpu_types.h build/numi_bottom_seal_bag build/PlasticBag.metallib > build/runtime_sha256.txt
