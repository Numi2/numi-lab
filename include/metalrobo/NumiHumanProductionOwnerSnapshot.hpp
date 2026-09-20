#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

inline constexpr std::uint32_t
    kNumiHumanProductionOwnerSnapshotVersionV1 = 1u;
inline constexpr const char*
    kNumiHumanProductionOwnerSnapshotSchemaV1 =
        "persistent-production-owner-snapshot.v1";
inline constexpr const char*
    kNumiHumanProductionOwnerSnapshotEvidenceSchemaV1 =
        "persistent-production-owner-snapshot-evidence.v1";

enum class NumiHumanProductionOwnerTreatmentV1 : std::uint32_t {
    cold = 1u,
    seeded = 2u,
};

enum class NumiHumanProductionOwnerDispositionV1 : std::uint32_t {
    published = 1u,
    rejected = 2u,
};

enum NumiHumanProductionOwnerCoverageV1 : std::uint64_t {
    NumiHumanOwnerInitialStateV1 = 1ull << 0u,
    NumiHumanOwnerCheckpointStateV1 = 1ull << 1u,
    NumiHumanOwnerEffectiveTangentFactorV1 = 1ull << 2u,
    NumiHumanOwnerSourceGeneralizedForceV1 = 1ull << 3u,
    NumiHumanOwnerFreeVelocityV1 = 1ull << 4u,
    NumiHumanOwnerCandidateStateV1 = 1ull << 5u,
    NumiHumanOwnerMuscleStateV1 = 1ull << 6u,
    NumiHumanOwnerMuscleResultsV1 = 1ull << 7u,
    NumiHumanOwnerMuscleGeneralizedForcesV1 = 1ull << 8u,
    NumiHumanOwnerTendonV1 = 1ull << 9u,
    NumiHumanOwnerMatterReactionV1 = 1ull << 10u,
    NumiHumanOwnerSupportRowsV1 = 1ull << 11u,
    NumiHumanOwnerSupportHistoryV1 = 1ull << 12u,
    NumiHumanOwnerEqualityRowsV1 = 1ull << 13u,
    NumiHumanOwnerLimitRowsV1 = 1ull << 14u,
    NumiHumanOwnerRHSBiasAccelerationV1 = 1ull << 15u,
    NumiHumanOwnerWorkEnergyV1 = 1ull << 16u,
    NumiHumanOwnerPublicationIdentityV1 = 1ull << 17u,
    NumiHumanOwnerRollbackIdentityV1 = 1ull << 18u,
    NumiHumanOwnerContactSamplesV1 = 1ull << 19u,
    NumiHumanOwnerMuscleProgramV1 = 1ull << 20u,
    NumiHumanOwnerMatterIntegrationUpdateV1 = 1ull << 21u,
    NumiHumanOwnerStatusRecordsV1 = 1ull << 22u,
};

// Every captured array advertises its logical shape independently from the
// payload. A covered zero-row owner is available with expectedElementCount=0;
// an unavailable stage is available=false and has no payload. Serializers
// reject partial payloads instead of silently emitting prefixes.
struct NumiHumanProductionOwnerArrayV1 {
    bool available = false;
    std::uint64_t expectedElementCount = 0u;
    std::uint32_t elementBytes = 4u;
    std::vector<std::byte> bytes;
};

struct NumiHumanProductionOwnerSnapshotV1 {
    NumiHumanProductionOwnerSnapshotV1() noexcept;

    std::uint32_t formatVersion =
        kNumiHumanProductionOwnerSnapshotVersionV1;
    NumiHumanProductionOwnerTreatmentV1 treatment =
        NumiHumanProductionOwnerTreatmentV1::cold;
    NumiHumanProductionOwnerDispositionV1 disposition =
        NumiHumanProductionOwnerDispositionV1::published;

    // Cold/seeded treatment identity is deliberately separate from the base
    // state. The base fingerprint excludes accepted support history.
    std::uint64_t baseStateFingerprint = 0u;
    std::uint64_t treatmentHistoryFingerprint = 0u;
    std::uint64_t humanSourceFingerprint = 0u;
    std::uint64_t matterSourcePhysicsFingerprint = 0u;
    std::uint64_t matterDeviceProgramFingerprint = 0u;
    std::uint64_t ownerProgramFingerprint = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t previousTransactionFingerprint = 0u;
    std::uint64_t linearizationEpoch = 0u;
    std::uint64_t slotGeneration = 0u;
    std::uint64_t controlStep = 0u;
    std::uint64_t physicsGeneration = 0u;
    std::uint64_t previousPhysicsGeneration = 0u;
    std::uint64_t brainGeneration = 0u;
    std::uint64_t sensorGeneration = 0u;
    std::uint64_t humanIOProgramFingerprint = 0u;
    std::uint64_t sensorFingerprint = 0u;
    std::uint64_t transactionInstanceFingerprint = 0u;
    // Timestamp of the attempted candidate interval. A rejected disposition
    // must never relabel this as an accepted-state timestamp.
    std::uint64_t candidateTimestampNanoseconds = 0u;
    std::uint64_t publicationEpoch = 0u;
    std::uint64_t jointFenceFingerprint = 0u;
    std::uint64_t timestepNanoseconds = 0u;
    std::uint64_t equalityProgramFingerprint = 0u;
    std::uint64_t limitProgramFingerprint = 0u;
    std::uint64_t supportPayloadByteCount = 0u;
    std::uint32_t supportPayloadABI = 0u;
    std::vector<std::byte> supportPayloadSHA256;

    std::uint32_t qCoordinateCount = 0u;
    std::uint32_t dofCount = 0u;
    std::uint32_t muscleCount = 0u;
    // Independent producer-contract extents for the auxiliary muscle program.
    // These cannot be inferred from a captured prefix without turning coverage
    // into a caller assertion.
    std::uint64_t muscleSiteCount = 0u;
    std::uint64_t muscleWrapCount = 0u;
    std::uint64_t muscleRouteNodeCount = 0u;
    std::uint32_t supportRowCount = 0u;
    std::uint32_t equalityRowCount = 0u;
    std::uint32_t limitRowCount = 0u;
    std::uint32_t tendonRowCount = 0u;
    std::uint32_t matterControlStep = 0u;
    // These producer-contract counts are independent of the serialized array
    // payloads. The serializer requires exact equality and never infers a
    // complete logical shape from a captured prefix.
    std::uint64_t contactSampleCount = 0u;
    std::uint64_t terminalAcceptedMatterRigidGeneralizedStateCount = 0u;
    std::uint64_t terminalAcceptedMatterRigidReactionCount = 0u;

    // coverageMask must exactly match the available arrays/identity booleans.
    // requiredCoverageMask is the set the producing runtime promised for this
    // concrete program. Any truncation bit makes serialization fail closed.
    std::uint64_t coverageMask = 0u;
    std::uint64_t requiredCoverageMask = 0u;
    std::uint64_t truncationMask = 0u;
    bool publicationIdentityAvailable = false;
    bool rollbackIdentityAvailable = false;

    NumiHumanProductionOwnerArrayV1 initialQ;
    NumiHumanProductionOwnerArrayV1 initialV;
    NumiHumanProductionOwnerArrayV1 initialRoot;
    NumiHumanProductionOwnerArrayV1 initialMuscles;
    NumiHumanProductionOwnerArrayV1 muscleRecords;
    NumiHumanProductionOwnerArrayV1 muscleSites;
    NumiHumanProductionOwnerArrayV1 muscleWraps;
    NumiHumanProductionOwnerArrayV1 muscleRouteNodes;
    NumiHumanProductionOwnerArrayV1 checkpointQ;
    NumiHumanProductionOwnerArrayV1 checkpointV;
    NumiHumanProductionOwnerArrayV1 checkpointRoot;
    NumiHumanProductionOwnerArrayV1 checkpointMuscles;
    // Row-major nv*nv device storage: the lower triangle is Cholesky L and
    // the upper triangle retains the original source A0 bytes.
    NumiHumanProductionOwnerArrayV1 effectiveTangentFactorStorage;
    NumiHumanProductionOwnerArrayV1 sourceGeneralizedForce;
    NumiHumanProductionOwnerArrayV1 sourcePredictedVelocity;
    NumiHumanProductionOwnerArrayV1 candidateQ;
    NumiHumanProductionOwnerArrayV1 candidateV;
    NumiHumanProductionOwnerArrayV1 candidateRoot;
    NumiHumanProductionOwnerArrayV1 candidateMuscles;
    NumiHumanProductionOwnerArrayV1 muscleResults;
    NumiHumanProductionOwnerArrayV1 muscleGeneralizedForces;
    NumiHumanProductionOwnerArrayV1 reducedMuscleGeneralizedForce;
    NumiHumanProductionOwnerArrayV1 tendonTransfers;
    NumiHumanProductionOwnerArrayV1 tendonGeneralizedCorrections;
    NumiHumanProductionOwnerArrayV1 matterGeneralizedReaction;
    NumiHumanProductionOwnerArrayV1 supportRows;
    NumiHumanProductionOwnerArrayV1 supportPlane;
    // Realized initial Matter history. Cold treatment is represented by one
    // explicit zero float4 per support row; the separate treatment identity
    // preserves whether those zeros were implicit or source-seeded.
    NumiHumanProductionOwnerArrayV1 initialSupportHistories;
    NumiHumanProductionOwnerArrayV1 candidateSupportHistories;
    NumiHumanProductionOwnerArrayV1 candidateSupportConsequences;
    // Completion-boundary accepted context. On rejection these are the prior
    // accepted histories/consequences, not the rejected candidate's outcome.
    NumiHumanProductionOwnerArrayV1 terminalAcceptedSupportHistories;
    NumiHumanProductionOwnerArrayV1 terminalAcceptedSupportConsequences;
    NumiHumanProductionOwnerArrayV1 equalityRows;
    // Host reconstruction from the captured production q0/v0/vFree/delta-v
    // and immutable rows. Each 32-byte record contains the four production
    // linearization floats, signed row impulse, row velocity/violation,
    // active flag and reserved zero. This is not a device-buffer readback.
    NumiHumanProductionOwnerArrayV1 equalityLinearizationImpulses;
    NumiHumanProductionOwnerArrayV1 limitRows;
    NumiHumanProductionOwnerArrayV1 limitLinearizationImpulses;
    NumiHumanProductionOwnerArrayV1 contactSamples;
    // Completion-boundary accepted-state context. For a published attempt this
    // is the newly published integration state; after rejection it is prior
    // accepted-state context and is not the rejected attempt's integration.
    NumiHumanProductionOwnerArrayV1
        terminalAcceptedMatterRigidGeneralizedState;
    NumiHumanProductionOwnerArrayV1 terminalAcceptedMatterRigidReactions;
    NumiHumanProductionOwnerArrayV1 standStatus;
    NumiHumanProductionOwnerArrayV1 humanMatterOwnerStatus;
    NumiHumanProductionOwnerArrayV1 acceleration;
    NumiHumanProductionOwnerArrayV1 sourceRHS;
    NumiHumanProductionOwnerArrayV1 sourceBias;
    // Three FP32 words: source-force midpoint work, Matter-reaction midpoint
    // work, and effective-tangent delta-v energy. These are mechanical solver
    // components, not a physical whole-body energy-closure certificate.
    NumiHumanProductionOwnerArrayV1 workEnergyComponents;
};

[[nodiscard]] std::uint64_t numiHumanProductionOwnerBaseStateFingerprintV1(
    std::uint64_t humanSourceWithoutInitialHistory,
    std::uint64_t matterWorldFingerprint,
    std::uint64_t timestepNanoseconds,
    std::span<const std::byte> q,
    std::span<const std::byte> v,
    std::span<const std::byte> root,
    std::span<const std::byte> muscles) noexcept;

// The two 64-bit fingerprints above are deterministic joins for runtime
// comparison and routing, not collision-resistant accepted-state proofs.
// Evidence writers must SHA-256 the canonical serialized payload separately.

[[nodiscard]] std::uint64_t
numiHumanProductionOwnerTreatmentHistoryFingerprintV1(
    std::span<const std::byte> supportPayloadIdentity,
    std::span<const std::byte> acceptedSupportHistory) noexcept;

[[nodiscard]] std::uint64_t numiHumanProductionOwnerCoverageMaskV1(
    const NumiHumanProductionOwnerSnapshotV1& snapshot) noexcept;

// Canonical single-line JSON with fixed field order. All identities and every
// raw FP32/record word are lowercase fixed-width hexadecimal strings, so the
// result is independent of locale and floating-point text formatting.
[[nodiscard]] bool serializeNumiHumanProductionOwnerSnapshotV1(
    const NumiHumanProductionOwnerSnapshotV1& snapshot,
    std::string& output,
    std::string& error);

// Wraps the canonical payload in a collision-resistant evidence envelope.
// The SHA-256 digest is deliberately separate from every versioned FNV64
// runtime join and from the accepted-root publication proof.
[[nodiscard]] bool serializeNumiHumanProductionOwnerSnapshotEvidenceV1(
    const NumiHumanProductionOwnerSnapshotV1& snapshot,
    std::string& output,
    std::string& payloadSHA256,
    std::string& error);

} // namespace metalrobo
