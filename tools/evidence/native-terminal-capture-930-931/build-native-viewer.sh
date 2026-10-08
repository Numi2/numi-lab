#!/bin/bash
set -euo pipefail
BUILD=/Users/n/numi-human-terminal-capture-build-017
SOURCE=/Users/n/numi-human-terminal-accepted-state-017
FROZEN_SOURCE=/Users/n/numi-human-performance-source-014
FROZEN_BUILD=/Users/n/numi-human-performance-build-014
BRAIN=/Users/n/numi-human-resting-integration-20261005/numi-brain/Sources
SDK=/Applications/Xcode-26.6.0.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
CXX=/usr/bin/clang++
xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math \
  -I "$SOURCE/matter/include" -I "$SOURCE/include" -I "$SOURCE/src/metal" \
  -I "$BRAIN/NumiBrainABI/include" -I "$BRAIN/NumiBrainMetal/Shaders" \
  -c "$SOURCE/matter/src/human_respiration.metal" -o "$BUILD/matter/shaders/HumanRespiration.air"
xcrun -sdk macosx metallib "$BUILD/matter/shaders/HumanRespiration.air" -o "$BUILD/matter/shaders/HumanRespiration.metallib"
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
INCLUDES=(
  "-I$BRAIN/NumiBrainABI/include"
  "-I$SOURCE/include"
  "-I$SOURCE/matter/include"
  "-F$SDK/System/Library/Frameworks"
)
FLAGS=(-O3 -DNDEBUG -std=c++23 -arch arm64 -isysroot "$SDK" -fPIE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
$CXX "${DEFINES[@]}" "${INCLUDES[@]}" "${FLAGS[@]}" -x objective-c++ -c \
  "$SOURCE/apps/numilab_human_myosim_visual_probe.mm" \
  -o "$BUILD/obj/numilab_human_myosim_visual_probe.mm.o"
$CXX -O3 -DNDEBUG -arch arm64 -isysroot "$SDK" -Wl,-search_paths_first \
  -Wl,-headerpad_max_install_names "$BUILD/obj/numilab_human_myosim_visual_probe.mm.o" \
  -o "$BUILD/bin/numi-human-native" -Wl,-rpath,"$FROZEN_BUILD/lib" \
  "$FROZEN_BUILD/lib/libmetalrobo.dylib" -framework Foundation -framework Metal \
  -framework CoreGraphics -framework ImageIO -framework AppKit -framework MetalKit \
  -framework AVFoundation -framework CoreVideo -framework CoreMedia -framework QuartzCore
