#pragma once

#include "metalrobo/WorldCompiler.hpp"
#include "metalrobo/TaskProgram.hpp"

namespace metalrobo {

// Canonical first world for the real-to-sim-to-real pipeline. Capture URIs are
// logical artifact locations that an EpisodeTwinCompiler adapter may replace
// with ARKit, RGB-D/ROS, or robot-log artifacts.
[[nodiscard]] EpisodeTwin makeFrankaPickPlaceEpisodeTwin();

// Runnable rigid/articulated topology matching the canonical episode:
// FER arm + Franka Hand, one dynamic pick object, static ground and target,
// and one dynamic clutter body.
[[nodiscard]] EngineModel makeFrankaPickPlaceEngineModel();

// Scene-only component used by CompiledRun. Body, shape, and material indices
// are local; composition owns every global rebase.
[[nodiscard]] EngineModel makeFrankaPickPlaceSceneEngineModel();

// Robot-independent pick/place objective bound to Franka semantics only when
// CompiledRun resolves the selected RobotPack.
[[nodiscard]] TaskPack makeFrankaPickPlaceTaskPack();

// Scene-state records ordered exactly as CompiledWorld::sceneBodyIndices().
[[nodiscard]] std::vector<MRBodyStateGPU>
makeFrankaPickPlaceSceneState();

// Covers appearance, object configuration, clutter, physics, robot state, and
// camera variation without changing the template topology.
[[nodiscard]] WorldProgram makeFrankaPickPlaceWorldProgram();

// Tacmap-style tactile variant. The two rubber pad colliders remain rigid
// backings; a 3 mm positive rest-offset shell plus bounded authored normal
// compliance creates bounded solver-visible compression. Dense flat atlases
// are cooked onto both articulated fingertips and target the manipulated
// object through the normal world pipeline.
[[nodiscard]] EngineModel makeFrankaTactileEngineModel();

[[nodiscard]] EpisodeTwin makeFrankaTactileEpisodeTwin();

} // namespace metalrobo
