#pragma once

#include "metalrobo/MetalWorld.hpp"

#include <cstdint>
#include <vector>

namespace metalrobo {

enum class LocomotionSurface : std::uint32_t {
    ground = 0u,
    terrain = 1u,
};

struct LocomotionWorld {
    EngineModel model;
    std::vector<MRBodyStateGPU> sceneBodies;
    TaskPack task;
};

struct CompiledLocomotionWorld {
    CompiledWorld world;
    CompiledTaskProgram task;
};

struct LocomotionWorldCompileDiagnostics {
    MetalWorldCompileDiagnostics world;
    TaskCompileDiagnostics task;

    [[nodiscard]] bool succeeded() const noexcept {
        return world.succeeded() && task.succeeded();
    }
};

// Compiles an imported or bundled robot through the single native locomotion
// route. Both outputs publish transactionally only after the world's
// capacities and every named task binding have validated.
[[nodiscard]] LocomotionWorldCompileDiagnostics
compileLocomotionWorld(
    const LocomotionWorld& authored,
    std::uint32_t articulationIndex,
    CompiledLocomotionWorld& compiled
);

// Appends the current Z-up locomotion ground or procedural terrain to a
// mutable engine model. Scene-state records are appended in the same order as
// the new non-articulated bodies.
void appendLocomotionSurface(
    EngineModel& model,
    std::vector<MRBodyStateGPU>& sceneBodies,
    LocomotionSurface surface
);

// Official 29-DoF Unitree G1 mechanics plus the current locomotion surface.
[[nodiscard]] LocomotionWorld makeUnitreeG1LocomotionWorld(
    LocomotionSurface surface
);

// Bundled policy/task contract expressed entirely through the same authored
// TaskPack accepted for imported robots.
[[nodiscard]] TaskPack makeUnitreeG1LocomotionTaskPack(
    LocomotionSurface surface
);

} // namespace metalrobo
