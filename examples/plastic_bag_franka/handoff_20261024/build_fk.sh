#!/bin/zsh
set -eu
cd "${0:A:h}"
mkdir -p build
clang++ -std=c++20 -O3 -DNDEBUG -I native_fk_owner/include source/franka_handoff_planner.cpp native_fk_owner/src/core/Franka.cpp native_fk_owner/src/core/FrankaEngine.cpp native_fk_owner/src/core/FrankaHand.cpp native_fk_owner/src/core/Model.cpp native_fk_owner/src/core/EngineModel.cpp native_fk_owner/src/core/ArticulatedDynamics.cpp native_fk_owner/src/core/ConstraintIR.cpp -o build/franka_handoff_planner
