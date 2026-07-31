#pragma once

#include "metalrobo/MetalWorld.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace metalrobo {

struct MRWorldPack;

enum class BuiltinSurface : std::uint32_t {
    ground = 0u,
    terrain = 1u,
};

struct SimulationDescription {
    EngineModel model;
    std::vector<MRBodyStateGPU> sceneBodies;
    TaskPack task;
    std::optional<PolicyPack> policy;
    std::uint32_t articulationIndex = 0u;
};

struct CompiledSimulation {
    CompiledWorld world;
    CompiledTaskProgram task;
    CompiledPolicyProgram policy;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint64_t fingerprint() const noexcept;
};

struct SimulationCompileDiagnostics {
    MetalWorldCompileDiagnostics world;
    TaskCompileDiagnostics task;
    PolicyCompileDiagnostics policy;
    bool policyRequested = false;

    [[nodiscard]] bool succeeded() const noexcept {
        return world.succeeded() && task.succeeded() &&
            (!policyRequested || policy.succeeded());
    }
};

// Compiles mechanics, task operators, and an optional deployment policy as
// one immutable program. The output publishes only after every capacity,
// semantic binding, and observation/action contract has validated.
[[nodiscard]] SimulationCompileDiagnostics
compileSimulation(
    const SimulationDescription& authored,
    std::uint32_t articulationIndex,
    CompiledSimulation& compiled
);

// Appends the current Z-up locomotion ground or procedural terrain to a
// mutable engine model. Scene-state records are appended in the same order as
// the new non-articulated bodies.
void appendBuiltinSurface(
    EngineModel& model,
    std::vector<MRBodyStateGPU>& sceneBodies,
    BuiltinSurface surface
);

// Materializes the authored base state from one complete MRWorldPack. The
// pack owns mechanics and scene composition; TaskPack remains a separate
// learning contract resolved against those exact semantic names.
[[nodiscard]] SimulationDescription makeWorldPackSimulation(
    const MRWorldPack& worldPack,
    TaskPack task
);

// Official 29-DoF Unitree G1 mechanics plus the current locomotion surface.
[[nodiscard]] SimulationDescription makeUnitreeG1Simulation(
    BuiltinSurface surface
);

// Bundled policy/task contract expressed entirely through the same authored
// TaskPack accepted for imported robots.
[[nodiscard]] TaskPack makeUnitreeG1TaskPack(
    BuiltinSurface surface
);

} // namespace metalrobo
