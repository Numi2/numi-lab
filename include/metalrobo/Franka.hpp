#pragma once

#include "metalrobo/EngineModel.hpp"

#include <cstddef>

namespace metalrobo {

inline constexpr std::size_t kFrankaPandaBodyCount = 8u;
inline constexpr std::size_t kFrankaPandaJointCount = 7u;
inline constexpr std::size_t kFrankaPandaShapeCount = 22u;
inline constexpr std::size_t kFrankaHandBodyCount = 3u;
inline constexpr std::size_t kFrankaHandJointCount = 3u;
inline constexpr std::size_t kFrankaHandDofCount = 2u;
inline constexpr std::size_t kFrankaHandShapeCount = 10u;

// Compiles the pinned Franka FER/Panda records used by the compatibility
// runtime into the canonical engine ABI. This is a fixed-root articulation:
// q/v contain the seven arm coordinates and every body pose is COM-centred.
// makeFrankaPandaModel() remains the independent legacy-ABI entry point.
[[nodiscard]] EngineModel makeFrankaPandaEngineModel();

// Extends the pinned FER arm with the official Franka Hand topology,
// inertias, finger limits, and primitive collision proxies derived from the
// upstream hand collision records at the same pinned description commit.
// The two physical finger coordinates remain explicit; policy adapters may
// impose the upstream mimic relation without hiding it in engine topology.
[[nodiscard]] EngineModel makeFrankaPandaHandEngineModel();

} // namespace metalrobo
