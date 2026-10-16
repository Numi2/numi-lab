#pragma once

#include "metalrobo/engine_types.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace metalrobo {

struct ContactSolverConfig {
    double timestep = 1.0 / 240.0;
    double errorReduction = 0.2;
    double penetrationSlop = 1.0e-4;
    double maxDepenetrationVelocity = 2.0;
    double impulseTolerance = 1.0e-8;
    double warmStartScale = 1.0;
    // Thresholds apply to J M^-1 J^T, i.e. inverse effective mass.
    double minimumInverseLinearEffectiveMass = 1.0e-12;
    double minimumInverseAngularEffectiveMass = 1.0e-12;
    std::uint32_t velocityIterations = 32;
    bool enableWarmStart = true;
    bool enableEarlyExit = true;
    bool deterministic = true;
};

struct ContactSolverDiagnostics {
    MRStepStatusCode code = MR_STEP_SUCCESS;
    std::uint32_t iterations = 0;
    std::uint32_t activeContacts = 0;
    std::uint32_t islandCount = 0;
    double maximumImpulseDelta = 0.0;
    double maximumNormalResidual = 0.0;
    double maximumConeViolation = 0.0;
    // Dimensionless max/min spread across linear contact directions only.
    double inverseLinearEffectiveMassSpread = 1.0;
    std::uint32_t requiredPairs = 0;
    std::uint32_t requiredContacts = 0;
    std::uint32_t requiredConstraints = 0;
    std::uint32_t requiredIslands = 0;

    [[nodiscard]] bool succeeded() const noexcept {
        return code == MR_STEP_SUCCESS ||
            code == MR_STEP_FIXED_BUDGET_COMPLETE;
    }
};

struct ConstraintIsland {
    std::vector<std::uint32_t> dynamicBodies;
    std::vector<std::uint32_t> contacts;
};

[[nodiscard]] std::vector<ConstraintIsland> buildConstraintIslands(
    std::span<const MRBodyStateGPU> bodies,
    std::span<const MRContactConstraintGPU> contacts
);

// Persistent cache for the fixed-budget throughput solver. It rejects
// new/stale impacts and rotates cached tangent impulses between contact
// frames. The converged quality/reference solvers additionally compare the
// warm-start objective against the zero start.
class ContactImpulseCache {
public:
    void beginStep(std::uint64_t step);
    void seed(std::span<MRContactConstraintGPU> contacts) const;
    void commit(std::span<const MRContactConstraintGPU> contacts);
    void prune(std::uint64_t maximumAge);
    void clear();

    [[nodiscard]] std::size_t size() const noexcept;

private:
    struct Key {
        std::uint64_t pair = 0;
        std::uint64_t feature = 0;

        bool operator==(const Key&) const = default;
    };

    struct KeyHash {
        std::size_t operator()(const Key& key) const noexcept;
    };

    struct Entry {
        mr_float4 impulses{};
        mr_float4 normal{};
        mr_float4 tangentU{};
        std::uint64_t lastSeenStep = 0;
    };

    std::unordered_map<Key, Entry, KeyHash> entries_;
    std::uint64_t step_ = 0;
};

// `solveContactConstraints` operates on one connected or caller-selected
// batch and rejects more than MR_MAX_CONTACTS_PER_SOLVER_BATCH contacts.
// Scene-level callers must island first; one oversized connected island is an
// explicit current limitation, never silently truncated.
[[nodiscard]] ContactSolverDiagnostics solveContactConstraints(
    std::span<MRBodyStateGPU> bodies,
    std::span<MRContactConstraintGPU> contacts,
    const ContactSolverConfig& config
);

[[nodiscard]] MRSolverBatchGPU makeSolverBatch(
    std::uint32_t bodyOffset,
    std::uint32_t bodyCount,
    std::uint32_t contactOffset,
    std::uint32_t contactCount,
    const ContactSolverConfig& config
);

} // namespace metalrobo
