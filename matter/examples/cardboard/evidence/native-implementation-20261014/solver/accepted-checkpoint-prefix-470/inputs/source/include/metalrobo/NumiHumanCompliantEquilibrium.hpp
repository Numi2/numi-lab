#pragma once

#include "metalrobo/NumiHumanMuscleEquilibrium.hpp"

namespace metalrobo {

// Offline preparation uses the same source scalar law as NHEQ2/NHLIM1.
// The inverse weight is authored dof_invweight0, never inferred from M(q).
struct NumiHumanSourceScalarLaw {
    mr_float4 solref{};
    mr_float4 solimp0{};
    mr_float4 solimp1{};
    double inverseWeight = 0.0;
    bool referenceSafe = true;
};
struct NumiHumanCompliantEquality {
    MRNumiHumanJointEqualityGPU source{};
    double inverseWeight = 0.0;
};
struct NumiHumanCompliantLimit {
    std::uint32_t qIndex = MR_INVALID_INDEX;
    std::uint32_t dofIndex = MR_INVALID_INDEX;
    double lower = 0.0, upper = 0.0, margin = 0.0;
    NumiHumanSourceScalarLaw law{};
};

// Signed force at a stationary state, a_ref/R. For a unilateral row the
// caller must first apply strict source margin admission. No hard reaction,
// damping load, equality projection or artificial preload is introduced.
[[nodiscard]] bool evaluateNumiHumanSourceStaticForce(
    const NumiHumanSourceScalarLaw& law, double phi, double timestep,
    double& force);

struct NumiHumanCompliantEquilibriumConfig {
    double timestep = 1.0e-4;
    bool referenceSafe = true;
    std::uint32_t maximumIterations = 32u;
    bool optimizeActivation = false;
    bool optimizePose = true;
    double accelerationTolerance = 0.05;
    double supportGapTolerance = 1.0e-6;
    double maximumCoordinateDisplacement = 0.15;
};

struct NumiHumanCompliantEquilibriumResult {
    NumiHumanMuscleEquilibriumResult state;
    std::vector<double> initialAcceleration;
    std::vector<double> objectiveHistory;
    double maximumLoadedSupportGap = 0.0;
    double minimumSupportGap = 0.0;
    std::uint32_t rejectedEvaluations = 0u;
};

// Native FP64 initial-condition search. All scalar coordinates, including
// source dependents, may deform. Optional recruitment stays in [0,1]; otherwise
// activation is held at the supplied value.
// Search iterations are not physical time. Only `balanced` plus geometric
// admission certifies this offline force sum; runtime/tissue evidence is separate.
[[nodiscard]] NumiHumanMuscleEquilibriumDiagnostics
compileNumiHumanCompliantEquilibrium(
    const EngineModel& model, std::uint32_t articulationIndex,
    std::span<const double> q, std::span<const double> activation,
    std::span<const MujocoMuscleSite> sites,
    std::span<const MujocoWrapGeometry> wraps,
    std::span<const MujocoMuscleDefinition> muscles,
    std::span<const MujocoCompliantMuscleArchitecture> architectures,
    std::span<const NumiHumanCompliantEquality> equalities,
    std::span<const NumiHumanCompliantLimit> limits,
    std::span<const NumiHumanStaticSupportContact> supports,
    NumiHumanCompliantEquilibriumResult& result,
    const NumiHumanCompliantEquilibriumConfig& config = {});

} // namespace metalrobo
