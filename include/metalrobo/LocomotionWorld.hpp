#pragma once

#include "metalrobo/MetalWorld.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace metalrobo {

struct MRWorldPack;

enum class LocomotionSurface : std::uint32_t {
    ground = 0u,
    terrain = 1u,
};

// Bundled G1 task presets compile through the same generic TaskPack route.
// The value selects authored data only; it does not select a robot-specific
// Metal kernel.
enum class UnitreeG1Task : std::uint32_t {
    velocity = 0u,
    disturbanceRecovery = 1u,
    supineGetUpDiscovery = 2u,
    ballDisturbanceRecovery = 3u,
    ballDodge = 4u,
};

struct LocomotionWorld {
    EngineModel model;
    std::vector<MRBodyStateGPU> sceneBodies;
    std::uint32_t articulationIndex = 0u;
};

struct LocomotionDynamicSphere {
    mr_float4 position{0.0f, 0.0f, 0.0f, 1.0f};
    mr_float4 linearVelocity{};
    float radius = 0.1f;
    float mass = 0.1f;
    std::uint32_t launchStep = 0u;
};

// A scene component owns both immutable mechanics and the reset state of each
// non-articulated body in that mechanics package. Body indices in reset state
// records are component-local; the run compiler rebases them during scene
// composition.
struct LocomotionSceneComponent {
    EngineModel mechanics;
    std::vector<MRBodyStateGPU> defaultBodyStates;
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
    const TaskPack& task,
    std::span<const RobotActuatorSpec> actuators,
    const TaskObservationProgram& observations,
    const TaskResetProgram& reset,
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

// Appends ordinary dynamic rigid bodies to any authored locomotion world.
// These bodies participate in the same broadphase, manifold, island, and
// temporal-cone solve as the robot and surface; no task-specific impulse path
// or robot-specific shader is involved.
void appendLocomotionDynamicSpheres(
    LocomotionWorld& world,
    std::span<const LocomotionDynamicSphere> spheres
);

// Package-oriented scene factories used by CompiledRun. These preserve the
// exact surface and rigid-body mechanics of the legacy mutable builders while
// keeping them outside the robot's mechanics package.
[[nodiscard]] LocomotionSceneComponent makeLocomotionSurfaceComponent(
    const EngineModel& referenceMechanics,
    LocomotionSurface surface
);
[[nodiscard]] LocomotionSceneComponent makeLocomotionDynamicSphereComponent(
    const EngineModel& referenceMechanics,
    std::span<const LocomotionDynamicSphere> spheres
);

// Materializes the authored base state from one complete MRWorldPack. The
// pack owns mechanics and scene composition; TaskPack remains a separate
// learning contract resolved against those exact semantic names.
[[nodiscard]] LocomotionWorld makeWorldPackLocomotionWorld(
    const MRWorldPack& worldPack
);

// Official 29-DoF Unitree G1 mechanics plus the current locomotion surface.
[[nodiscard]] LocomotionWorld makeUnitreeG1LocomotionWorld(
    LocomotionSurface surface
);

// Bundled policy/task contract expressed entirely through the same authored
// TaskPack accepted for imported robots.
[[nodiscard]] TaskPack makeUnitreeG1LocomotionTaskPack(
    LocomotionSurface surface,
    TaskObservationProgram& observations,
    TaskResetProgram& reset
);

// Canonical normalized-action radians used by the bundled G1 TaskPack and
// deployment adapters. Keeping one native source prevents teacher providers
// from silently targeting a different controller contract.
[[nodiscard]] std::span<const float>
unitreeG1LocomotionActionScales() noexcept;

// Zero-command standing and balance-recovery workload. Native randomized
// impulses are the high-throughput training proxy; ordinary dynamic bodies
// provide complementary physical-impact evidence.
[[nodiscard]] TaskPack makeUnitreeG1DisturbanceRecoveryTaskPack(
    LocomotionSurface surface,
    TaskObservationProgram& observations,
    TaskResetProgram& reset
);

// HumanUP-style Stage-I discovery task: fixed supine reset, full-body
// collision, dense height/upright/support shaping, and weak regularization.
[[nodiscard]] TaskPack makeUnitreeG1SupineGetUpDiscoveryTaskPack(
    LocomotionSurface surface,
    TaskObservationProgram& observations,
    TaskResetProgram& reset
);

// Balance recovery driven by randomized ordinary rigid spheres. Sphere
// mechanics are supplied by the world while this TaskPack only authors their
// per-episode launch envelopes through generic scene-body operators.
[[nodiscard]] TaskPack makeUnitreeG1BallDisturbanceRecoveryTaskPack(
    LocomotionSurface surface,
    TaskObservationProgram& observations,
    TaskResetProgram& reset
);

// Perception-conditioned, contact-free projectile avoidance. The actor uses
// ball-only masked depth while native privileged link-clearance operators
// shape the critic and task reward before physical contact.
[[nodiscard]] TaskPack makeUnitreeG1BallDodgeTaskPack(
    LocomotionSurface surface,
    TaskObservationProgram& observations,
    TaskResetProgram& reset
);

} // namespace metalrobo
