#!/bin/bash
set -euo pipefail
WORK=/Users/n/numi-human-retired-alias-visibility-018
E=/Users/n/numi-human-resting-evidence-20261005/native-retired-alias-visibility-018-attempt2
BUILD=/Users/n/numi-human-retired-alias-visibility-build-018-attempt2
FROZEN_SOURCE=/Users/n/numi-human-performance-source-014
FROZEN_BUILD=/Users/n/numi-human-performance-build-014
BRAIN=/Users/n/numi-human-resting-integration-20261005/numi-brain/Sources
SDK=/Applications/Xcode-26.6.0.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
CXX=/usr/bin/clang++
mkdir -p "$BUILD/obj" "$BUILD/bin" "$BUILD/matter/shaders"
EXPECTED_METALLIB=4b61361f513bf0996d687398498b85ba4e379edca91c36f134e1b0398f31c426
EXPECTED_DYLIB=6bccfc4044d825423e66bc2f60ba3cf59eaa9a4936773ab08182f58a09927092
[[ "$(git -C "$WORK" rev-parse HEAD)" == b091d7dcead509a325194563ed38261319118a88 ]]
[[ "$(shasum -a 256 "$WORK/matter/src/human_respiration.metal" | awk '{print $1}')" == 8ec33f7495e086618a294ecf900a571ed4f55091baae92989874c5dc1f045973 ]]
[[ "$(shasum -a 256 "$FROZEN_BUILD/lib/libmetalrobo.dylib" | awk '{print $1}')" == "$EXPECTED_DYLIB" ]]
cp /Users/n/numi-human-terminal-capture-build-017/matter/shaders/HumanRespiration.metallib "$BUILD/matter/shaders/HumanRespiration.metallib"
[[ "$(shasum -a 256 "$BUILD/matter/shaders/HumanRespiration.metallib" | awk '{print $1}')" == "$EXPECTED_METALLIB" ]]
DEFINES=(
  "-DMETALROBO_METALLIB=\"$FROZEN_BUILD/shaders/MetalRobo.metallib\""
  "-DNUMI_HUMAN_ANTERIOR_THORAX_MATERIAL=\"$FROZEN_SOURCE/matter/materials/human_anterior_abdominal_wall_composite_effective.nmatter\""
  "-DNUMI_HUMAN_OPEN_KNEE_LIGAMENT_MATERIAL=\"$FROZEN_SOURCE/matter/materials/open_knee_ligament_febio_exp_linear.nmatter\""
  "-DNUMI_HUMAN_PASSIVE_TISSUE_MATERIAL=\"$FROZEN_SOURCE/matter/materials/passive_skeletal_muscle_unqualified.nmatter\""
  "-DNUMI_HUMAN_PECTORALIS_FASCIA_MATERIAL=\"$FROZEN_SOURCE/matter/materials/human_pectoralis_fascia_goh_uniaxial.nmatter\""
  "-DNUMI_HUMAN_PLANTAR_FASCIA_MATERIAL=\"$FROZEN_SOURCE/matter/materials/human_plantar_fascia_reduced.nmatter\""
  "-DNUMI_HUMAN_REGIONAL_MYOFASCIA_MATERIAL=\"$FROZEN_SOURCE/matter/materials/human_myofascia_regional_transverse_isotropic.nmatter\""
  "-DNUMI_HUMAN_RESPIRATION_METALLIB=\"$BUILD/matter/shaders/HumanRespiration.metallib\""
  "-DNUMI_HUMAN_RESTING_SCENE=1"
  "-DNUMI_HUMAN_THORACOLUMBAR_FASCIA_MATERIAL=\"$FROZEN_SOURCE/matter/materials/human_thoracolumbar_fascia_effective_isotropic.nmatter\""
  "-DNUMI_MATTER_METALLIB=\"$FROZEN_BUILD/matter/shaders/NumiMatter.metallib\""
)
INCLUDES=("-I$BRAIN/NumiBrainABI/include" "-I$WORK/include" "-I$WORK/matter/include" "-F$SDK/System/Library/Frameworks")
FLAGS=(-O3 -DNDEBUG -std=c++23 -arch arm64 -isysroot "$SDK" -fPIE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
$CXX "${DEFINES[@]}" "${INCLUDES[@]}" "${FLAGS[@]}" -x objective-c++ -c \
  "$WORK/apps/numilab_human_myosim_visual_probe.mm" \
  -o "$BUILD/obj/numilab_human_myosim_visual_probe.mm.o"
$CXX -O3 -DNDEBUG -arch arm64 -isysroot "$SDK" -Wl,-search_paths_first \
  -Wl,-headerpad_max_install_names "$BUILD/obj/numilab_human_myosim_visual_probe.mm.o" \
  -o "$BUILD/bin/numi-human-native" -Wl,-rpath,"$FROZEN_BUILD/lib" \
  "$FROZEN_BUILD/lib/libmetalrobo.dylib" -framework Foundation -framework Metal \
  -framework CoreGraphics -framework ImageIO -framework AppKit -framework MetalKit \
  -framework AVFoundation -framework CoreVideo -framework CoreMedia -framework QuartzCore
shasum -a 256 "$BUILD/bin/numi-human-native" "$BUILD/matter/shaders/HumanRespiration.metallib" "$FROZEN_BUILD/lib/libmetalrobo.dylib"
otool -L "$BUILD/bin/numi-human-native"
