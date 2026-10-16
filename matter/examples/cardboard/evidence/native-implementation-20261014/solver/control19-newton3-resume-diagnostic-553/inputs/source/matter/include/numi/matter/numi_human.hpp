#pragma once

#include "numi/matter/matter.hpp"
#include "numi/matter/numi_human_shared.h"
#include "metalrobo/MetalArticulatedOperator.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

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
    // Optional frictionless gliding contact from FEM surface nodes to
    // articulated body-following tangent planes. Every sample contributes an
    // equal-and-opposite body wrench in the same accepted transaction.
    std::span<const NMNumiHumanFEMBodyContactSampleGPU>
        femBodyContactSamples{};
    // Optional exact-surface elastic-foundation contact scattered directly as
    // balanced articulated-body wrenches. This avoids duplicating the contact
    // compliance with a full-resolution cartilage volume solve.
    std::span<const NMNumiHumanArticularContactSampleGPU>
        articularContactSamples{};
    // Optional contiguous source-order ranges over articularContactSamples.
    // When present, Matter records exact accepted FP32 normal force for every
    // pair plus the pair-summed aggregate under the same transaction gate as
    // the aggregate wrench audit.
    std::span<const NMIncidenceRangeGPU> articularContactPairRanges{};
    // Optional source-law passive ligament fibre families. These are reduced
    // force-transfer elements between exact enthesis attachment-node
    // centroids. They may coexist with neutral matrix-only FEM volumes without
    // duplicating axial fibre stiffness.
    std::span<const NMNumiHumanPassiveLigamentGPU> passiveLigaments{};
    // Optional three-point tension-only routes with a body-following pulley.
    // These own passive windlass force transfer without inventing an actuator
    // or requiring a compression-bearing volumetric fascia surrogate.
    std::span<const NMNumiHumanPassiveRoutedBandGPU> passiveRoutedBands{};
    std::uint32_t endpointCount = 0u;
    std::uint32_t environmentCount = 1u;
    float productionForceOwnerFraction = 0.0f;
};

enum class NumiHumanTendonFEMLoadExecutionMode : std::uint32_t {
    // The adapter opens and reconciles its own Runtime transaction. This is
    // the existing standalone Human tendon/FEM path.
    standaloneRuntime = 0u,
    // The adapter assembles the exact nodal field and source-force
    // corrections only. A bound NumanX Human/Matter owner consumes the field
    // in its one Runtime transaction and remains the sole prepare/apply owner.
    numanXDeferred = 1u,
};

struct NumiHumanTendonFEMLoadConfiguration {
    std::filesystem::path metallib;
    NumiHumanTendonFEMLoadExecutionMode executionMode =
        NumiHumanTendonFEMLoadExecutionMode::standaloneRuntime;
    std::uint32_t reserved0 = 0u;
};

struct NumiHumanTendonFEMLoadDiagnostics {
    bool initialized = false;
    std::uint32_t encodedPassCount = 0u;
    std::uint32_t abortCount = 0u;
    std::uint32_t preSourceCorrectionEncodeCount = 0u;
    std::uint32_t deferredExternalForceBorrowCount = 0u;
    std::uint32_t runtimePreDynamicsEncodeCount = 0u;
    std::uint32_t runtimePostCommitEncodeCount = 0u;
    NumiHumanTendonFEMLoadExecutionMode executionMode =
        NumiHumanTendonFEMLoadExecutionMode::standaloneRuntime;
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
    std::uint32_t femBodyContactSampleCount = 0u;
    std::uint32_t femBodyContactClosedSampleCount = 0u;
    std::uint32_t femBodyContactAuditedStepCount = 0u;
    double femBodyContactAreaSquareMeters = 0.0;
    double femBodyContactNormalForceNewtons = 0.0;
    double femBodyContactMaximumPressurePascals = 0.0;
    double femBodyContactForceResidualNewtons = 0.0;
    double femBodyContactMomentResidualNewtonMeters = 0.0;
    double femBodyContactStoredEnergyJoules = 0.0;
    double femBodyContactMaximumNormalStrain = 0.0;
    double femBodyContactMaximumClosureMeters = 0.0;
    double femBodyContactMaximumTangentialSlipMeters = 0.0;
    double femBodyContactTrajectoryMinimumNormalForceNewtons = 0.0;
    double femBodyContactTrajectoryMaximumNormalForceNewtons = 0.0;
    double femBodyContactTrajectoryMaximumForceResidualNewtons = 0.0;
    double femBodyContactTrajectoryMaximumMomentResidualNewtonMeters = 0.0;
    double femBodyContactTrajectoryMaximumTangentialSlipMeters = 0.0;
    std::uint32_t articularContactSampleCount = 0u;
    std::uint32_t articularContactPairCount = 0u;
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
    std::uint32_t passiveRoutedBandCount = 0u;
    bool passiveRoutedBandLatestTransactionAccepted = false;
    double passiveRoutedBandEndpointForceL1Newtons = 0.0;
    double passiveRoutedBandMaximumTensionNewtons = 0.0;
    double passiveRoutedBandMinimumStrain = 0.0;
    double passiveRoutedBandMaximumStrain = 0.0;
    double passiveRoutedBandForceResidualNewtons = 0.0;
    double passiveRoutedBandMomentResidualNewtonMeters = 0.0;
    double passiveRoutedBandStoredEnergyJoules = 0.0;
    double passiveRoutedBandMaximumExtensionMeters = 0.0;
    std::string message;
};

inline constexpr std::uint32_t
    kNumiHumanTendonFEMDeferredLoadABIVersion = 1u;

// Ephemeral, read-only view produced by a successful deferred pre-source
// callback for one exact command buffer and Human step. The adapter owns both
// Metal buffers. Consumers may bind them only on that command buffer; they may
// not retain, replace, mutate, commit, wait, or read them back. Validation
// statuses are the exact GPU admission result for the assembled force field.
struct NumiHumanTendonFEMDeferredExternalForceView {
    std::uint32_t abiVersion =
        kNumiHumanTendonFEMDeferredLoadABIVersion;
    std::uint32_t structSize = sizeof(
        NumiHumanTendonFEMDeferredExternalForceView);
    void* externalForces = nullptr;       // id<MTLBuffer>, float4
    void* validationStatuses = nullptr;   // id<MTLBuffer>, MRMetalWorldStatusGPU
    std::uint64_t externalForcesGPUAddress = 0u;
    std::uint64_t validationStatusesGPUAddress = 0u;
    std::uint64_t externalForceElementCount = 0u;
    std::uint64_t validationStatusElementCount = 0u;
    std::uint64_t deviceRegistryID = 0u;
    std::uint64_t programFingerprint = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t femNodeCount = 0u;
    std::uint32_t externalForceStride = 0u;
    std::uint32_t validationStatusStride = 0u;
    std::uint32_t stepIndex = 0u;
    std::uint32_t reserved0 = 0u;
};

using NumiHumanTendonFEMBorrowDeferredExternalForces = bool (*)(
    void* context,
    void* commandBuffer,
    std::uint32_t stepIndex,
    std::uint32_t environmentCount,
    NumiHumanTendonFEMDeferredExternalForceView& output
) noexcept;

// Immutable bridge capability shared by the enclosing Human owner and its
// NumanX Human/Matter adapter. The Human owner invokes
// encodePreSourceCorrections through MetalNumiHumanTendonLoadProgram before
// its free predictor. NumanX then borrows the resulting exact-command nodal
// field through borrowExternalForces. The callback never opens, prepares, or
// commits a Matter Runtime transaction.
struct NumiHumanTendonFEMDeferredLoadProgram {
    std::uint32_t abiVersion =
        kNumiHumanTendonFEMDeferredLoadABIVersion;
    std::uint32_t structSize = sizeof(
        NumiHumanTendonFEMDeferredLoadProgram);
    void* context = nullptr;
    metalrobo::MetalNumiHumanTendonLoadEncode
        encodePreSourceCorrections = nullptr;
    NumiHumanTendonFEMBorrowDeferredExternalForces
        borrowExternalForces = nullptr;
    std::uint64_t fingerprint = 0u;
    std::uint64_t runtimeDeviceProgramFingerprint = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t femNodeCount = 0u;
    std::uint32_t activeAnchorCount = 0u;
    std::uint32_t reserved0 = 0u;

    [[nodiscard]] bool valid() const noexcept {
        return abiVersion == kNumiHumanTendonFEMDeferredLoadABIVersion &&
            structSize == sizeof(NumiHumanTendonFEMDeferredLoadProgram) &&
            context != nullptr && encodePreSourceCorrections != nullptr &&
            borrowExternalForces != nullptr && fingerprint != 0u &&
            runtimeDeviceProgramFingerprint != 0u &&
            environmentCount != 0u && femNodeCount != 0u &&
            activeAnchorCount != 0u && reserved0 == 0u;
    }

    [[nodiscard]] bool configured() const noexcept {
        return abiVersion != kNumiHumanTendonFEMDeferredLoadABIVersion ||
            structSize != sizeof(NumiHumanTendonFEMDeferredLoadProgram) ||
            context != nullptr || encodePreSourceCorrections != nullptr ||
            borrowExternalForces != nullptr || fingerprint != 0u ||
            runtimeDeviceProgramFingerprint != 0u || environmentCount != 0u ||
            femNodeCount != 0u || activeAnchorCount != 0u || reserved0 != 0u;
    }
};

inline constexpr std::uint32_t
    kNumiHumanTendonFEMLoadAdapterSnapshotVersionV1 = 1u;

using NumiHumanTendonFEMLoadAdapterDigest =
    std::array<std::uint8_t, 32u>;

// Complete continuation authority owned by NumiHumanTendonFEMLoadAdapter.
// Immutable source mappings and Metal pipeline caches are bound by
// adapterFingerprint; provisional force/wrench/audit scratch is deliberately
// excluded. The vectors below are the accepted records that survive command
// completion and affect replay evidence or its accepted-history cursor.
//
// canonicalNumiHumanTendonFEMLoadAdapterSnapshotV1() emits a versioned,
// field-name- and element-size-framed byte stream. It is the only byte stream
// callers should embed in a larger owner digest. authoritySHA256 authenticates
// those exact bytes and is checked before restore mutates any adapter state.
struct NumiHumanTendonFEMLoadAdapterSnapshotV1 {
    bool available = false;
    std::uint32_t formatVersion =
        kNumiHumanTendonFEMLoadAdapterSnapshotVersionV1;
    std::uint64_t adapterFingerprint = 0u;
    std::uint64_t runtimeDeviceProgramFingerprint = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t endpointCount = 0u;
    std::uint32_t femNodeCount = 0u;
    std::uint32_t contactSampleCount = 0u;
    std::uint32_t femBodyContactSampleCount = 0u;
    std::uint32_t articularContactSampleCount = 0u;
    std::uint32_t articularContactPairCount = 0u;
    std::uint32_t passiveLigamentCount = 0u;
    std::uint32_t passiveRoutedBandCount = 0u;
    // Zero before the first borrowed pass and the exact bound pose stride
    // afterwards. This is restored with the accepted-state cursor.
    std::uint32_t boundBodyPoseStride = 0u;
    std::uint32_t encodedPassCount = 0u;
    std::uint32_t abortCount = 0u;
    std::uint32_t articularAttemptedStepCount = 0u;

    std::vector<nm_float4> anchorReactionAcceptedHistory;
    std::vector<NMNumiHumanArticularContactAuditGPU>
        femBodyContactAcceptedHistory;
    std::vector<NMNumiHumanArticularContactAuditGPU>
        articularContactAcceptedHistory;
    // [environment][accepted step][pair 0..pairCount-1, aggregate].
    std::vector<float> articularContactPairForceAcceptedHistory;
    std::vector<NMNumiHumanPassiveLigamentAuditGPU>
        passiveLigamentAcceptedState;
    std::vector<NMNumiHumanPassiveRoutedBandAuditGPU>
        passiveRoutedBandAcceptedState;

    NumiHumanTendonFEMLoadAdapterDigest authoritySHA256{};
    // Diagnostic only; excluded from canonical bytes and equality.
    std::string message;
};

struct NumiHumanTendonFEMLoadAdapterRestoreDiagnostics {
    bool restored = false;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept { return restored; }
};

// Emits the stable ABI-bound authority bytes used by authoritySHA256. The
// output is replaced only on success.
[[nodiscard]] bool canonicalNumiHumanTendonFEMLoadAdapterSnapshotV1(
    const NumiHumanTendonFEMLoadAdapterSnapshotV1& snapshot,
    std::vector<std::uint8_t>& output,
    std::string& error
) noexcept;

[[nodiscard]] bool digestNumiHumanTendonFEMLoadAdapterSnapshotV1(
    const NumiHumanTendonFEMLoadAdapterSnapshotV1& snapshot,
    NumiHumanTendonFEMLoadAdapterDigest& output,
    std::string& error
) noexcept;

[[nodiscard]] bool sameNumiHumanTendonFEMLoadAdapterSnapshotAuthorityV1(
    const NumiHumanTendonFEMLoadAdapterSnapshotV1& lhs,
    const NumiHumanTendonFEMLoadAdapterSnapshotV1& rhs
) noexcept;

struct NumiHumanPassiveLigamentFiberEvaluation {
    double effectiveStretch = 0.0;
    double fiberStressPascals = 0.0;
    double tensionNewtons = 0.0;
};

struct NumiHumanPassiveRoutedBandEvaluation {
    double routeLengthMeters = 0.0;
    double stretch = 0.0;
    double strain = 0.0;
    double tensionNewtons = 0.0;
    double storedEnergyJoules = 0.0;
};

// CPU reference for the exact FEBio trans-iso fibre stress branch used by the
// reduced Metal ligament owner. Matrix stress is deliberately excluded.
[[nodiscard]] bool evaluateNumiHumanPassiveLigamentFiber(
    const NMNumiHumanPassiveLigamentGPU& ligament,
    double currentCentroidLengthMeters,
    NumiHumanPassiveLigamentFiberEvaluation& result
) noexcept;

// CPU FP64 oracle for the exact tension law evaluated by the Metal routed-band
// owner. geometricLength excludes the pulley arc; signedWindingRadians is the
// live twist from the sample's neutral relative orientation.
[[nodiscard]] bool evaluateNumiHumanPassiveRoutedBand(
    const NMNumiHumanPassiveRoutedBandGPU& band,
    double geometricLengthMeters,
    double signedWindingRadians,
    NumiHumanPassiveRoutedBandEvaluation& result
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
    [[nodiscard]] NumiHumanTendonFEMDeferredLoadProgram
    deferredProgram() noexcept;
    [[nodiscard]] NumiHumanTendonFEMLoadDiagnostics diagnostics() const noexcept;
    // Snapshot/restore are completion-boundary operations: the caller must not
    // have an in-flight borrowed command buffer using this adapter. Restore
    // validates version, source/runtime identity, every vector extent, and the
    // canonical digest before the first counter or GPU byte is changed.
    [[nodiscard]] NumiHumanTendonFEMLoadAdapterSnapshotV1
    snapshot() const noexcept;
    [[nodiscard]] NumiHumanTendonFEMLoadAdapterRestoreDiagnostics restore(
        const NumiHumanTendonFEMLoadAdapterSnapshotV1& snapshot
    ) noexcept;

private:
    [[nodiscard]] bool encodePreDynamics(
        const metalrobo::MetalNumiHumanTendonLoadPass& pass
    );
    [[nodiscard]] bool encodePostValidation(
        const metalrobo::MetalNumiHumanTendonLoadPass& pass
    );
    [[nodiscard]] bool borrowDeferredExternalForces(
        void* commandBuffer,
        std::uint32_t stepIndex,
        std::uint32_t environmentCount,
        NumiHumanTendonFEMDeferredExternalForceView& output
    ) noexcept;
    void abort(void* commandBuffer) noexcept;
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace numi::matter
