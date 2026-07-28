#pragma once

#include "metalrobo/ArticulatedContact.hpp"
#include "metalrobo/ConstraintIR.hpp"
#include "metalrobo/ConstraintSolver.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace metalrobo {

enum class ArticulatedCollisionFailure : std::uint32_t {
    none = 0u,
    invalidConfiguration,
    invalidConstraint,
    capacityOverflow,
    invalidBodyBinding,
    unboundDynamicBody,
    crossArticulationContact,
    unsupportedContactSemantics,
    nonfiniteResult,
};

struct ArticulatedCollisionAdapterConfig {
    ContactSolverConfig contact{};
    // Strictly positive floor that makes the exact-cone quality problem
    // strongly convex. Normal material compliance is converted from
    // position-level compliance to impulse regularization by compliance/dt^2
    // and added to this floor.
    double qualityTangentialRegularization = 1.0e-9;
    std::uint32_t contactCapacity =
        MR_MAX_CONTACTS_PER_SOLVER_BATCH;
};

struct ArticulatedCollisionDiagnostics {
    MRStepStatusCode code = MR_STEP_SUCCESS;
    ArticulatedCollisionFailure failure =
        ArticulatedCollisionFailure::none;
    std::uint32_t articulationIndex = 0u;
    std::uint32_t inputConstraintCount = 0u;
    std::uint32_t requiredContactCount = 0u;
    std::uint32_t adaptedContactCount = 0u;
    std::uint32_t swappedEndpointCount = 0u;
    std::uint32_t failedConstraintIndex = MR_INVALID_INDEX;
    std::uint32_t failedBlockIndex = MR_INVALID_INDEX;
    std::uint64_t semanticFingerprint = 0u;
    double maximumKinematicTargetCompensation = 0.0;
    double maximumNormalTargetVelocity = 0.0;

    [[nodiscard]] bool succeeded() const noexcept {
        return code == MR_STEP_SUCCESS;
    }
};

struct ArticulatedCollisionResult {
    ArticulatedCollisionDiagnostics diagnostics{};
    std::vector<ArticulatedContact> contacts;
    // Maps each output contact back to the common constraint span.
    std::vector<std::uint32_t> sourceConstraintIndices;
    // Maps each output contact back to an evaluated ConstraintIR block.
    std::vector<std::uint32_t> sourceBlockIndices;

    [[nodiscard]] bool succeeded() const noexcept {
        return diagnostics.succeeded();
    }
};

// Adapts active common contact ABI records into one articulation's analytic
// generalized-contact convention. MRBodyStateGPU flagsAndIndices bind a
// collision state to (articulation, model body/link). Maximal-coordinate
// dynamic bodies and contacts across articulations are rejected explicitly.
//
// Endpoint order is canonicalized so bodyA belongs to the selected
// articulation. When the common record has the articulated body at endpoint
// B, normal, tangent coordinates, surface target, and cached impulse are all
// transformed consistently. Static/kinematic counterparts become
// kArticulatedStaticWorld; their prescribed point velocity is moved into the
// target so the generalized solver sees the same physical relative velocity.
//
// The result is transactional: on any error or capacity overflow both payload
// vectors are empty and diagnostics identify the failing record/requirement.
[[nodiscard]] ArticulatedCollisionResult
adaptArticulatedContactConstraints(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<const MRContactConstraintGPU> constraints,
    std::span<const MRBodyStateGPU> bodyStates,
    const ArticulatedCollisionAdapterConfig& config = {}
);

// Consumes the already-evaluated ConstraintIR stream without re-deriving
// timestep-dependent targets, restitution, stiction selection, compliance,
// or warm-start projection. This is the authoritative quality-path adapter.
//
// The current exact-cone articulated solver supports active three-row,
// uncapped, isotropic contact blocks. Torsion, rolling, adhesion, anisotropy,
// scalar constraints, and zero-regularization rows are rejected explicitly.
// World-point endpoints bind through the supplied collision body states.
//
// The semantic fingerprint is validated before any output is produced and is
// echoed in diagnostics. On failure all result vectors are empty.
[[nodiscard]] ArticulatedCollisionResult
adaptEvaluatedArticulatedContacts(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    const ConstraintIREvaluationView& semantics,
    std::span<const MRBodyStateGPU> bodyStates,
    std::uint32_t contactCapacity =
        MR_MAX_CONTACTS_PER_SOLVER_BATCH
);

} // namespace metalrobo
