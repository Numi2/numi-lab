#pragma once

#include "metalrobo/EngineModel.hpp"

#include <cstddef>

namespace metalrobo {

inline constexpr std::size_t kFrankaPandaBodyCount = 8u;
inline constexpr std::size_t kFrankaPandaJointCount = 7u;
inline constexpr std::size_t kFrankaPandaShapeCount = 22u;

// Compiles the pinned Franka FER/Panda records used by the compatibility
// runtime into the canonical engine ABI. This is a fixed-root articulation:
// q/v contain the seven arm coordinates and every body pose is COM-centred.
// makeFrankaPandaModel() remains the independent legacy-ABI entry point.
[[nodiscard]] EngineModel makeFrankaPandaEngineModel();

} // namespace metalrobo
