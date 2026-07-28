#pragma once

#include "metalrobo/WorldCompiler.hpp"

namespace metalrobo {

// Canonical first world for the real-to-sim-to-real pipeline. Capture URIs are
// logical artifact locations that an EpisodeTwinCompiler adapter may replace
// with ARKit, RGB-D/ROS, or robot-log artifacts.
[[nodiscard]] EpisodeTwin makeFrankaPickPlaceEpisodeTwin();

// Runnable rigid/articulated topology matching the canonical episode:
// FER arm + Franka Hand, one dynamic pick object, static ground and target,
// and one dynamic clutter body.
[[nodiscard]] EngineModel makeFrankaPickPlaceEngineModel();

// Scene-state records ordered exactly as CompiledWorld::sceneBodyIndices().
[[nodiscard]] std::vector<MRBodyStateGPU>
makeFrankaPickPlaceSceneState();

// Covers appearance, object configuration, clutter, physics, robot state, and
// camera variation without changing the template topology.
[[nodiscard]] WorldProgram makeFrankaPickPlaceWorldProgram();

} // namespace metalrobo
