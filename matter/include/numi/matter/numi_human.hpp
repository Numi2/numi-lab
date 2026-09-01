#pragma once

#include "numi/matter/matter.hpp"
#include "numi/matter/numi_human_shared.h"
#include "metalrobo/MetalArticulatedOperator.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

namespace numi::matter {

enum class NumiHumanFEMPrestressStatus : std::uint8_t {
    success = 0u,
    invalidSnapshot,
    invalidFraction,
    invalidParameterArena,
    invalidMaterial,
    invalidParameter,
    duplicateParameter,
    invalidBounds,
};

// One environment-local material parameter that participates in a bounded
// prestress continuation. Indices are stable cooked material/local-parameter
// indices; callers must derive them from the admitted MaterialProgram rather
// than relying on a hard-coded global parameter offset.
struct NumiHumanFEMPrestressTarget {
    std::uint32_t materialIndex = NM_INVALID_INDEX;
    std::uint32_t localParameterIndex = NM_INVALID_INDEX;
    float neutralValue = 0.0f;
    float sourceValue = 0.0f;
};

struct NumiHumanFEMPrestressDiagnostics {
    NumiHumanFEMPrestressStatus status =
        NumiHumanFEMPrestressStatus::invalidSnapshot;
    std::uint32_t failingTarget = NM_INVALID_INDEX;
    std::uint32_t appliedParameterCount = 0u;
    float fraction = 0.0f;
    float maximumAbsoluteParameterDelta = 0.0f;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == NumiHumanFEMPrestressStatus::success;
    }
};

// Atomically prepares a completion-boundary snapshot for one continuation
// stage. This function only stages bounded material parameters; it does not
// claim equilibrium. The caller must restore the returned snapshot, execute
// an accepted FEM transaction, and establish its own convergence certificate.
// Validation is completed before the snapshot is mutated.
[[nodiscard]] NumiHumanFEMPrestressDiagnostics
prepareNumiHumanFEMPrestressStage(
    const CompiledWorld& world,
    std::span<const NumiHumanFEMPrestressTarget> targets,
    float fraction,
    RuntimeStateSnapshot& snapshot
);

// Complete load and bone-anchor maps for one Numi Human FEM consumer. Node
// records are immutable and indexed by cooked global FEM-node index.
//
// Active tendon replacement mode requires at least one endpoint replacement and
// a positive productionForceOwnerFraction. Endpoint mode removes the declared
// anchor-endpoint J^T share. Full-muscle-row mode removes the selected source
// muscle's complete generalized-force row and restores only its load-endpoint
// reaction, so internal source wrap forces cannot duplicate a continuum path.
// A declared distal force couple additionally maps the other terminal across
// two opposing exact attachment patches; its signed nodal weights sum to zero
// and its absolute weights sum to twice the owned terminal force.
// Both modes return solved fixed-node reactions through the owning body
// Jacobians.
//
// Passive attachment-only mode uses no endpoint replacements, no active node
// loads, and a zero productionForceOwnerFraction. Active anchors may reference
// different articulated bodies; their fixed-node reactions are returned through
// each owning body Jacobian in the same accepted command-buffer transaction.
struct NumiHumanTendonFEMLoadSource {
    std::span<const NMNumiHumanTendonFEMNodeLoadGPU> nodeLoads{};
    std::span<const NMNumiHumanTendonFEMNodeAnchorGPU> nodeAnchors{};
    std::span<const NMNumiHumanTendonFEMEndpointReplacementGPU>
        endpointReplacements{};
    // Optional internal cartilage/meniscus contact. Ranges exactly cover the
    // cooked FEM-node array and index contactContributions; every sample must
    // have one slave and three master contributions.
    std::span<const NMNumiHumanFEMContactSampleGPU> contactSamples{};
    std::span<const NMNumiHumanFEMContactContributionGPU>
        contactContributions{};
    std::span<const NMIncidenceRangeGPU> contactRanges{};
    // Optional exact-surface elastic-foundation contact scattered directly as
    // balanced articulated-body wrenches. This avoids duplicating the contact
    // compliance with a full-resolution cartilage volume solve.
    std::span<const NMNumiHumanArticularContactSampleGPU>
        articularContactSamples{};
    // Optional source-law passive ligament fibre families. These are reduced
    // force-transfer elements between exact enthesis attachment-node
    // centroids. They may coexist with neutral matrix-only FEM volumes without
    // duplicating axial fibre stiffness.
    std::span<const NMNumiHumanPassiveLigamentGPU> passiveLigaments{};
    std::uint32_t endpointCount = 0u;
    std::uint32_t environmentCount = 1u;
    float productionForceOwnerFraction = 0.0f;
};

struct NumiHumanTendonFEMLoadConfiguration {
    std::filesystem::path metallib;
};

struct NumiHumanTendonFEMLoadDiagnostics {
    bool initialized = false;
    std::uint32_t encodedPassCount = 0u;
    std::uint32_t abortCount = 0u;
    std::uint64_t fingerprint = 0u;
    // Valid after the enclosing borrowed command buffer has completed. L1 is
    // the sum of nodal force magnitudes; resultant is the magnitude of their
    // vector sum for the most recently encoded pass across all environments.
    double assembledExternalForceL1Newtons = 0.0;
    double assembledExternalForceResultantNewtons = 0.0;
    // Exact prescribed-node reactions captured before the same-command-buffer
    // J^T projection. Unlike Runtime scratch, this audit survives replay and
    // rollback bookkeeping and therefore describes the last encoded pass.
    double anchorReactionL1Newtons = 0.0;
    double anchorReactionResultantNewtons = 0.0;
    std::uint32_t anchorReactionAuditedStepCount = 0u;
    double anchorReactionTrajectoryMinimumL1Newtons = 0.0;
    double anchorReactionTrajectoryMaximumL1Newtons = 0.0;
    double anchorReactionTrajectoryMaximumResultantNewtons = 0.0;
    std::uint32_t contactSampleCount = 0u;
    std::uint32_t articularContactSampleCount = 0u;
    std::uint32_t articularMechanicalSampleCount = 0u;
    std::uint32_t articularInternalSameBodySampleCount = 0u;
    std::uint32_t articularClosedSampleCount = 0u;
    double articularContactAreaSquareMeters = 0.0;
    double articularNormalForceNewtons = 0.0;
    double articularMaximumPressurePascals = 0.0;
    double articularBodyForceL1Newtons = 0.0;
    double articularForceResidualNewtons = 0.0;
    double articularMomentResidualNewtonMeters = 0.0;
    double articularStoredEnergyJoules = 0.0;
    double articularMaximumNormalStrain = 0.0;
    double articularMaximumClosureMeters = 0.0;
    std::uint32_t articularAuditedStepCount = 0u;
    std::uint32_t articularTrajectoryMinimumClosedSampleCount = 0u;
    std::uint32_t articularTrajectoryMaximumClosedSampleCount = 0u;
    double articularTrajectoryMinimumNormalForceNewtons = 0.0;
    double articularTrajectoryMaximumNormalForceNewtons = 0.0;
    double articularTrajectoryMaximumPressurePascals = 0.0;
    double articularTrajectoryMaximumStoredEnergyJoules = 0.0;
    double articularTrajectoryMaximumNormalStrain = 0.0;
    double articularTrajectoryMaximumClosureMeters = 0.0;
    double articularTrajectoryMaximumForceResidualNewtons = 0.0;
    double articularTrajectoryMaximumMomentResidualNewtonMeters = 0.0;
    std::uint32_t passiveLigamentCount = 0u;
    bool passiveLigamentLatestTransactionAccepted = false;
    double passiveLigamentEndpointForceL1Newtons = 0.0;
    double passiveLigamentMaximumTensionNewtons = 0.0;
    double passiveLigamentMinimumEffectiveStretch = 0.0;
    double passiveLigamentMaximumEffectiveStretch = 0.0;
    double passiveLigamentForceResidualNewtons = 0.0;
    double passiveLigamentMomentResidualNewtonMeters = 0.0;
    std::string message;
};

struct NumiHumanPassiveLigamentFiberEvaluation {
    double effectiveStretch = 0.0;
    double fiberStressPascals = 0.0;
    double tensionNewtons = 0.0;
};

// CPU reference for the exact FEBio trans-iso fibre stress branch used by the
// reduced Metal ligament owner. Matrix stress is deliberately excluded.
[[nodiscard]] bool evaluateNumiHumanPassiveLigamentFiber(
    const NMNumiHumanPassiveLigamentGPU& ligament,
    double currentCentroidLengthMeters,
    NumiHumanPassiveLigamentFiberEvaluation& result
) noexcept;

class NumiHumanTendonFEMLoadAdapter {
public:
    NumiHumanTendonFEMLoadAdapter();
    ~NumiHumanTendonFEMLoadAdapter();
    NumiHumanTendonFEMLoadAdapter(NumiHumanTendonFEMLoadAdapter&&) noexcept;
    NumiHumanTendonFEMLoadAdapter& operator=(
        NumiHumanTendonFEMLoadAdapter&&
    ) noexcept;
    NumiHumanTendonFEMLoadAdapter(
        const NumiHumanTendonFEMLoadAdapter&
    ) = delete;
    NumiHumanTendonFEMLoadAdapter& operator=(
        const NumiHumanTendonFEMLoadAdapter&
    ) = delete;

    [[nodiscard]] bool initialize(
        Runtime& runtime,
        const NumiHumanTendonFEMLoadSource& source,
        const NumiHumanTendonFEMLoadConfiguration& configuration
    );
    [[nodiscard]] metalrobo::MetalNumiHumanTendonLoadProgram
    program() noexcept;
    [[nodiscard]] NumiHumanTendonFEMLoadDiagnostics diagnostics() const noexcept;

private:
    [[nodiscard]] bool encodePreDynamics(
        const metalrobo::MetalNumiHumanTendonLoadPass& pass
    );
    [[nodiscard]] bool encodePostValidation(
        const metalrobo::MetalNumiHumanTendonLoadPass& pass
    );
    void abort(void* commandBuffer) noexcept;
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace numi::matter
