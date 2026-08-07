#pragma once

#include "numi/matter/matter.hpp"
#include "metalrobo/MetalWorld.hpp"

namespace numi::matter {

// Adapts a persistent Matter Runtime to MetalWorld's per-substep borrowed
// command-buffer boundary. The returned program does not own Runtime; the
// Runtime must outlive every MetalWorld submission that references it.
[[nodiscard]] metalrobo::MetalWorldDevicePhysicsProgram
makeMetalWorldDevicePhysicsProgram(Runtime& runtime) noexcept;

} // namespace numi::matter
