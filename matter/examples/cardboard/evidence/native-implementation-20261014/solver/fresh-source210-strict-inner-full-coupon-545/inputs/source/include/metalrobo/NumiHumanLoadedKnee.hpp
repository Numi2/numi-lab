#pragma once

#include "metalrobo/EngineModel.hpp"
#include "metalrobo/NumiHumanKnee.hpp"
#include "metalrobo/NumiHumanTissueMass.hpp"
#include "numi/matter/matter.hpp"
#include "numi/matter/numi_human_shared.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace metalrobo {

using NumiHumanLoadedKneeDigest = std::array<std::uint8_t, 32u>;

inline constexpr std::uint32_t kNumiHumanLoadedKneeAuthoringVersionV1 = 1u;
inline constexpr std::uint32_t kNumiHumanLoadedKneeAcceptanceVersionV1 = 1u;
inline constexpr std::uint64_t kNumiHumanLoadedKneeStepNanoseconds = 50'000u;
inline constexpr std::uint32_t kNumiHumanLoadedKneeAcceptedStepCount = 8u;
inline constexpr std::uint32_t kNumiHumanLoadedKneeNodeCount = 248'236u;
inline constexpr std::uint32_t kNumiHumanLoadedKneeTetrahedronCount = 844'287u;
inline constexpr std::uint32_t kNumiHumanLoadedKneeSurfaceCount = 88u;
inline constexpr std::uint32_t kNumiHumanLoadedKneeSurfaceFaceCount = 729'068u;
inline constexpr std::uint32_t kNumiHumanLoadedKneeNodeSetCount = 42u;
inline constexpr std::uint32_t kNumiHumanLoadedKneeNodeSetMembershipCount = 43'260u;
inline constexpr std::uint32_t kNumiHumanLoadedKneeSourceSurfacePairCount = 19u;
inline constexpr std::uint32_t kNumiHumanLoadedKneeRegionCount = 6u;
inline constexpr std::uint32_t kNumiHumanLoadedKneeLoadedNodeCount = 62'402u;
inline constexpr std::uint32_t kNumiHumanLoadedKneeLoadedTetrahedronCount = 264'442u;
inline constexpr std::uint32_t kNumiHumanLoadedKneeContactPairCount = 7u;
inline constexpr std::uint32_t kNumiHumanLoadedKneeActiveReplacementCount = 4u;
inline constexpr std::uint32_t kNumiHumanLoadedKneePassiveOwnerCount = 5u;
// Stable identifiers for direct Matter Human attachments are the domain tag
// "KN" followed by the one-based executable FEM row. The executable row is
// profile/object/local order, not the sparse ABI3 source-global node index.
inline constexpr std::uint32_t
    kNumiHumanLoadedKneeAttachmentStableIdentifierBase = 0x4b4e0000u;
inline constexpr std::uint64_t kNumiHumanLoadedKneePayloadBytesV1 = 34'357'400u;
inline constexpr std::string_view kNumiHumanLoadedKneeLabAuthoringBoundaryV1 =
    "Candidate-only source/projected-reference and donor provenance export. "
    "Projected rest is not an unloaded or stress-free state; population "
    "material priors, prescribed contact coverage, and this export do not "
    "establish subject mechanics, production ownership, sustained tracking, "
    "mesh convergence, clinical validity, or a global seven-owner "
    "accepted-state root.";
inline constexpr std::string_view kNumiHumanLoadedKneeHumanPackBoundaryV1 =
    "Candidate-only left Open Knee authoring receipt. It binds source "
    "topology, projected-rest identity, population material priors, explicit "
    "donor subtraction, and candidate force/state ownership. It does not "
    "establish an unloaded or stress-free reference, prestress equilibrium, "
    "subject calibration, an accepted x_current value, production ownership, "
    "loaded motion, clinical validity, or integrated Human qualification.";

// HumanPack owns immutable authoring identity and policy only. In particular,
// it may require a runtime x_current SHA-256 but may not publish accepted
// x_current bytes or a runtime accepted-state count.
struct NumiHumanLoadedKneeImmutableIdentityV1 {
    std::string schema;
    NumiHumanLoadedKneeDigest fileSHA256{};
    NumiHumanLoadedKneeDigest identitySHA256{};
};

struct NumiHumanLoadedKneePayloadIdentityV1 {
    std::string schema;
    std::string magic;
    std::uint32_t abi = 0u;
    std::uint64_t byteCount = 0u;
    NumiHumanLoadedKneeDigest fileSHA256{};
};

struct NumiHumanLoadedKneeTopologyV1 {
    NumiHumanLoadedKneeDigest identitySHA256{};
    NumiHumanLoadedKneeDigest executableFEMTopologySHA256{};
    NumiHumanLoadedKneeDigest sourceGlobalNodeIndexSHA256{};
    NumiHumanLoadedKneeDigest anchorOwnershipSHA256{};
    std::uint32_t nodeCount = 0u;
    std::uint32_t tetrahedronCount = 0u;
    std::uint32_t surfaceCount = 0u;
    std::uint32_t surfaceFaceCount = 0u;
    std::uint32_t nodeSetCount = 0u;
    std::uint32_t nodeSetMembershipCount = 0u;
    std::uint32_t sourceSurfacePairCount = 0u;
    std::uint32_t loadedNodeCount = 0u;
    std::uint32_t loadedTetrahedronCount = 0u;
    std::array<std::string, kNumiHumanLoadedKneeRegionCount>
        loadedRegionNames{};
};

struct NumiHumanLoadedKneeCoordinateAuthoringV1 {
    NumiHumanLoadedKneeDigest sourceStateSHA256{};
    NumiHumanLoadedKneeDigest referenceStateSHA256{};
    std::uint32_t sourceStateNodeCount = 0u;
    std::uint32_t referenceStateNodeCount = 0u;
    std::string sourceStateEncoding;
    std::string referenceStateEncoding;
    std::string sourceFrameID;
    std::string referenceFrameID;
    std::string referenceStateClass;
    std::string constructionID;
    bool unloadedReferenceQualified = false;
    std::string prestrainResetStatus;
    std::string prestrainResetMethodID;
    std::string volumetricPrestressStatus;

    std::string currentStateOwner;
    std::string currentStateHashAlgorithm;
    std::string currentStateHashScope;
    bool currentStateRequired = false;
    // Must remain false/zero in HumanPack.loaded-anatomy-knee.v1.
    bool authoredAcceptedCurrentStatePresent = false;
    NumiHumanLoadedKneeDigest authoredAcceptedCurrentStateSHA256{};
};

struct NumiHumanLoadedKneeMaterialV1 {
    std::string sourceType;
    double c1Pascals = 0.0;
    double c2Pascals = 0.0;
    double c3Pascals = 0.0;
    double c4 = 0.0;
    double c5Pascals = 0.0;
    double lambdaMaximum = 0.0;
    double bulkModulusPascals = 0.0;
    double initialStretch = 0.0;
    std::array<double, 3u> homogeneousFiberWorld{};
    std::string calibrationStatus;
};

struct NumiHumanLoadedKneeRegionV1 {
    std::string semanticID;
    std::string sourceRegionName;
    std::uint32_t donorBodyIndex = NUMI_HUMAN_KNEE_INVALID_INDEX;
    NumiHumanLoadedKneeDigest topologyIdentitySHA256{};
    NumiHumanLoadedKneeMaterialV1 material;
    std::string physicalVolumeOwnerID;
    std::string mechanicalMassOwnerID;
    std::string materialOwnerID;
    std::string activeForceOwnerStatus;
    std::string activeForceOwnerID;
    std::string stateOwnerID;
};

struct NumiHumanLoadedKneeContactPairV1 {
    std::string semanticID;
    std::string sourcePairName;
    std::string masterSurface;
    std::string slaveSurface;
    std::string ownerID;
};

struct NumiHumanLoadedKneeActiveReplacementV1 {
    std::string semanticID;
    std::uint32_t sourceActuatorIndex = 0u;
    std::string sourceRouteName;
    std::uint32_t loadEndpointIndex = NUMI_HUMAN_KNEE_INVALID_INDEX;
    std::uint32_t loadRouteNodeIndex = NUMI_HUMAN_KNEE_INVALID_INDEX;
    std::uint32_t loadSourceSiteIndex = NUMI_HUMAN_KNEE_INVALID_INDEX;
    std::uint32_t loadBodyIndex = NUMI_HUMAN_KNEE_INVALID_INDEX;
    std::uint32_t anchorEndpointIndex = NUMI_HUMAN_KNEE_INVALID_INDEX;
    std::uint32_t anchorRouteNodeIndex = NUMI_HUMAN_KNEE_INVALID_INDEX;
    std::uint32_t anchorSourceSiteIndex = NUMI_HUMAN_KNEE_INVALID_INDEX;
    std::uint32_t anchorBodyIndex = NUMI_HUMAN_KNEE_INVALID_INDEX;
    std::string sourceOwnerID;
    std::string candidateOwnerID;
    std::string mode;
    double replacementFraction = 0.0;
};

struct NumiHumanLoadedKneePassiveOwnerV1 {
    std::string semanticID;
    std::string sourceRegionName;
    std::string ownerID;
};

struct NumiHumanLoadedKneeAuthoringV1 {
    std::uint32_t formatVersion = kNumiHumanLoadedKneeAuthoringVersionV1;
    std::string schema;
    std::string compiler;
    std::string status;
    std::string sourceOwnershipStatus;
    std::string side;
    bool productionPromotion = false;
    NumiHumanLoadedKneeDigest manifestSHA256{};
    NumiHumanLoadedKneeDigest ownershipManifestSHA256{};
    std::string manifestCanonicalization;
    std::string manifestHashExclusion;

    NumiHumanLoadedKneeImmutableIdentityV1 ownershipInput;
    NumiHumanLoadedKneeImmutableIdentityV1 authoringProfileInput;
    NumiHumanLoadedKneeImmutableIdentityV1 tendonPayloadInput;
    NumiHumanLoadedKneeImmutableIdentityV1 xReferenceInput;
    NumiHumanLoadedKneeImmutableIdentityV1 labExportInput;
    NumiHumanLoadedKneePayloadIdentityV1 kneePayload;
    NumiHumanLoadedKneePayloadIdentityV1 tendonPayload;
    // SHA-256 identity emitted by the Lab authoring export. The runtime's
    // compact EngineModel/MetalWorld fingerprints below are separate ABI-bound
    // execution identities and must never be substituted for this digest.
    NumiHumanLoadedKneeDigest sourceModelFingerprintSHA256{};
    NumiHumanLoadedKneeDigest sourceRigidPayloadSHA256{};
    NumiHumanLoadedKneeDigest equalityPayloadSHA256{};
    std::string sourceDefaultPoseID;
    NumiHumanLoadedKneeDigest sourceDefaultPoseSHA256{};
    std::string projectedReferencePoseID;
    NumiHumanLoadedKneeDigest projectedReferencePoseSHA256{};
    std::string sourceToReferenceMappingID;
    std::string sourceToReferenceMappingAlgorithm;
    NumiHumanLoadedKneeDigest sourceToReferenceMappingCodeSHA256{};

    std::string subjectID;
    std::vector<NumiHumanLoadedKneeDigest> coverageLeafSHA256s;
    std::string datasetID;
    NumiHumanLoadedKneeDigest licenseFileSHA256{};

    NumiHumanLoadedKneeTopologyV1 topology;
    NumiHumanLoadedKneeCoordinateAuthoringV1 coordinates;
    double densitySourceValue = 0.0;
    std::string densitySourceUnit;
    double densityConversionFactorToKgPerM3 = 0.0;
    double densityRuntimeKgPerM3 = 0.0;
    std::string densityCalibrationStatus;
    std::array<NumiHumanLoadedKneeRegionV1,
               kNumiHumanLoadedKneeRegionCount> regions{};
    std::array<NumiHumanLoadedKneeContactPairV1,
               kNumiHumanLoadedKneeContactPairCount> contactPairs{};
    std::array<NumiHumanLoadedKneeActiveReplacementV1,
               kNumiHumanLoadedKneeActiveReplacementCount>
        activeReplacements{};
    std::array<NumiHumanLoadedKneePassiveOwnerV1,
               kNumiHumanLoadedKneePassiveOwnerCount> passiveOwners{};
    NumiHumanLoadedKneeDigest authoredMassPartitionSHA256{};
    NumiHumanLoadedKneeDigest authoredRawF32NodeMassSHA256{};
    std::string fullStateSemanticID;
    std::string fullStateOwnerID;
    std::string fullStateSchema;
    NumiHumanLoadedKneeDigest fullStateIdentitySHA256{};
    std::vector<std::string> requiredSnapshotComponents;

    bool candidateOnly = false;
    bool productionQualified = false;
    bool unloadedReferenceQualified = false;
    bool subjectCalibratedMaterials = false;
    bool donorMassSubtractionRequired = false;
    bool massMomentClosureRequired = false;
    std::string boundary;
};

struct NumiHumanLoadedKneeMassEvidenceV1 {
    std::vector<NumiHumanTissueMassPartition> partitions;
    std::vector<NumiHumanTissueMassNode> cookedNodes;
    EngineModel rebasedModel;
    NumiHumanLoadedKneeDigest closureSHA256{};
    NumiHumanLoadedKneeDigest cookedNodeMassSHA256{};
    NumiHumanLoadedKneeDigest authoredRawF32NodeMassSHA256{};
    NumiHumanLoadedKneeDigest executedRawF32NodeMassSHA256{};
    NumiHumanLoadedKneeDigest referenceCoordinateSHA256{};
    NumiHumanLoadedKneeDigest sourceGlobalNodeIndexSHA256{};
    NumiHumanLoadedKneeDigest executableFEMTopologySHA256{};
    NumiHumanLoadedKneeDigest sourceAnchorOwnershipSHA256{};
    NumiHumanLoadedKneeDigest executedAnchorOwnershipSHA256{};
    // Exact source asset, compiled FEBio exp-linear programs/overlays, and
    // reduced passive-ligament rows admitted by the material binder below.
    NumiHumanLoadedKneeDigest materialExecutionSHA256{};
    std::uint64_t sourceRigidModelFingerprint = 0u;
    std::uint64_t executedSourcePhysicsFingerprint = 0u;
    std::uint64_t executedRigidModelFingerprint = 0u;
    double maximumPackedMomentRelativeError = 0.0;
};

// Exact immutable anchor table consumed by the Numi Human/Matter adapter.
// sourceGlobalNodeIndex prevents a profile-order row from being silently
// rebound to a different ABI3 source node. localPoint is the executed
// projected-reference/body-local point; the distinct source anchor digest in
// HumanPack retains the raw ABI3 authoring value.
struct NumiHumanLoadedKneeExecutedAnchorV1 {
    std::uint32_t sourceGlobalNodeIndex = NUMI_HUMAN_KNEE_INVALID_INDEX;
    std::uint32_t bodyIndex = NUMI_HUMAN_KNEE_INVALID_INDEX;
    std::uint32_t flags = 0u;
    std::array<float, 3u> localPoint{};
};

// Exact row-level admission used by the executable topology binder. Active
// anchors must retain the immutable ABI3 body and anchorLocal, translated only
// by the proven donor COM rebase. All comparisons are float-bit exact so a
// one-bit adapter-coordinate drift fails closed.
[[nodiscard]] bool validateNumiHumanLoadedKneeExecutedAnchorRowV1(
    std::uint32_t expectedSourceGlobalNodeIndex,
    const NumiHumanKneeNode& sourceNode,
    const std::array<double, 3u>& donorCOMOffsetM,
    const NMFEMNodeStateGPU& matterNode,
    const NumiHumanLoadedKneeExecutedAnchorV1& executedAnchor,
    std::string& error
);

// Proves that an active ABI3 anchor is represented by the exact native Matter
// attachment row used by the coupled Human solve. Marker 2, body/object/node,
// deterministic stable identifier, and rebased body-local point are all
// checked exactly. Inactive nodes have no compiled attachment row and are
// enforced by the enclosing topology binder.
[[nodiscard]] bool validateNumiHumanLoadedKneeCompiledAttachmentRowV1(
    std::uint32_t expectedExecutableNodeIndex,
    std::uint32_t expectedObjectIndex,
    const NumiHumanKneeNode& sourceNode,
    const NumiHumanLoadedKneeExecutedAnchorV1& executedAnchor,
    const NMFEMNodeStateGPU& matterNode,
    const NMFEMHumanAttachmentGPU& matterAttachment,
    std::string& error
);

[[nodiscard]] bool digestNumiHumanLoadedKneeExecutedAnchorsV1(
    std::span<const NumiHumanLoadedKneeExecutedAnchorV1> executedAnchors,
    NumiHumanLoadedKneeDigest& output,
    std::string& error
);

// Lab/runtime-owned accepted transaction evidence. This is deliberately not
// named or represented as the whole-Human seven-owner accepted-state root.
// Full Matter snapshot vectors are hashed as explicitly type-tagged,
// element-size-framed opaque bytes from the current Matter ABI. This is an
// ABI-bound replay identity, not a portable serialization. As with
// sameMatterSnapshotAuthority(), diagnostic deformableContactFailures are
// deliberately excluded because they do not own deterministic continuation.
struct NumiHumanLoadedKneeRuntimeEvidenceV1 {
    std::uint32_t formatVersion = kNumiHumanLoadedKneeAcceptanceVersionV1;
    std::uint32_t acceptedStepIndex = 0u;
    std::uint64_t acceptedTimestampNanoseconds = 0u;
    std::uint64_t timestepNanoseconds = 0u;
    std::uint32_t loadedFEMNodeFirst = 0u;
    std::uint32_t loadedFEMNodeCount = 0u;
    NumiHumanLoadedKneeDigest executedSourceCoordinateSHA256{};
    NumiHumanLoadedKneeDigest executedReferenceCoordinateSHA256{};
    NumiHumanLoadedKneeDigest executedInitialCoordinateSHA256{};
    NumiHumanLoadedKneeDigest referencePoseSHA256{};
    NumiHumanLoadedKneeDigest initialPoseSHA256{};
    NumiHumanLoadedKneeDigest coordinateDerivationSHA256{};
    NumiHumanLoadedKneeDigest executedMassClosureSHA256{};
    NumiHumanLoadedKneeDigest executedRawF32NodeMassSHA256{};
    NumiHumanLoadedKneeDigest executedSourceGlobalNodeIndexSHA256{};
    NumiHumanLoadedKneeDigest executedFEMTopologySHA256{};
    NumiHumanLoadedKneeDigest executedSourceAnchorOwnershipSHA256{};
    NumiHumanLoadedKneeDigest executedAnchorOwnershipSHA256{};
    NumiHumanLoadedKneeDigest executedMaterialExecutionSHA256{};
    std::uint64_t executedSourceRigidModelFingerprint = 0u;
    std::uint64_t executedSourcePhysicsFingerprint = 0u;
    std::uint64_t executedRigidModelFingerprint = 0u;
    std::array<double, kNumiHumanLoadedKneeContactPairCount>
        contactPairNormalForceNewtons{};
    double contactAggregateNormalForceNewtons = 0.0;
    // Independent prescribed-closure operator coverage. This is not the
    // actual-pose force vector and may not be used as physiology evidence.
    std::array<double, kNumiHumanLoadedKneeContactPairCount>
        prescribedClosurePairForceNewtons{};
    std::array<NumiHumanLoadedKneeContactPairV1,
               kNumiHumanLoadedKneeContactPairCount>
        executedContactPairs{};
    std::array<NumiHumanLoadedKneeActiveReplacementV1,
               kNumiHumanLoadedKneeActiveReplacementCount>
        executedActiveReplacements{};
    std::array<NumiHumanLoadedKneePassiveOwnerV1,
               kNumiHumanLoadedKneePassiveOwnerCount>
        executedPassiveOwners{};
    NumiHumanLoadedKneeDigest articulatedQVRootTimeSHA256{};
    NumiHumanLoadedKneeDigest muscleTendonStateSHA256{};
    NumiHumanLoadedKneeDigest adapterAcceptedStateSHA256{};
    numi::matter::RuntimeStateSnapshot snapshot;
};

struct NumiHumanLoadedKneeAcceptanceReceiptV1 {
    std::uint32_t formatVersion = kNumiHumanLoadedKneeAcceptanceVersionV1;
    std::string schema;
    // Propagated verbatim from the authenticated Human ownership artifact.
    // This scoped candidate receipt must not hide a blocked/partial source
    // registry behind the HumanPack manifest identity.
    std::string sourceOwnershipStatus;
    NumiHumanLoadedKneeDigest humanPackManifestSHA256{};
    NumiHumanLoadedKneeDigest massClosureSHA256{};
    NumiHumanLoadedKneeDigest rawF32NodeMassSHA256{};
    NumiHumanLoadedKneeDigest sourceGlobalNodeIndexSHA256{};
    NumiHumanLoadedKneeDigest executableFEMTopologySHA256{};
    NumiHumanLoadedKneeDigest sourceAnchorOwnershipSHA256{};
    NumiHumanLoadedKneeDigest executedAnchorOwnershipSHA256{};
    NumiHumanLoadedKneeDigest materialExecutionSHA256{};
    NumiHumanLoadedKneeDigest previousTransactionSHA256{};
    NumiHumanLoadedKneeDigest xSourceSHA256{};
    NumiHumanLoadedKneeDigest xReferenceSHA256{};
    NumiHumanLoadedKneeDigest xInitialSHA256{};
    NumiHumanLoadedKneeDigest coordinateDerivationSHA256{};
    NumiHumanLoadedKneeDigest xCurrentSHA256{};
    NumiHumanLoadedKneeDigest fullMatterSnapshotSHA256{};
    NumiHumanLoadedKneeDigest articulatedQVRootTimeSHA256{};
    NumiHumanLoadedKneeDigest muscleTendonStateSHA256{};
    NumiHumanLoadedKneeDigest adapterAcceptedStateSHA256{};
    NumiHumanLoadedKneeDigest fullAcceptedStateSHA256{};
    NumiHumanLoadedKneeDigest transactionSHA256{};
    std::uint64_t sourceRigidModelFingerprint = 0u;
    std::uint64_t executedRigidModelFingerprint = 0u;
    std::uint32_t acceptedStepIndex = 0u;
    std::uint64_t acceptedTimestampNanoseconds = 0u;
    std::uint64_t timestepNanoseconds = 0u;
    bool candidateOnly = true;
    bool productionQualified = false;
    bool globalSevenOwnerAcceptedStateRootAvailable = false;
};

[[nodiscard]] bool validateNumiHumanLoadedKneeAuthoringV1(
    const NumiHumanLoadedKneeAuthoringV1& authoring,
    const NumiHumanKneePayload* decodedPayload,
    std::string& error
);

[[nodiscard]] bool verifyNumiHumanLoadedKneeImmutableFileV1(
    const std::filesystem::path& path,
    const NumiHumanLoadedKneeDigest& expectedSHA256,
    std::uint64_t expectedByteCount,
    std::string& error
);

// Publishes bytes exactly once. An existing regular file is accepted only
// when its bytes are identical; symbolic links and conflicting contents are
// rejected.
// New files are staged and synchronized in the destination directory, then
// installed with a no-replace rename.
[[nodiscard]] bool writeNumiHumanLoadedKneeImmutableFileV1(
    const std::filesystem::path& path,
    std::span<const std::byte> bytes,
    std::string& error
);

// Exact ABI-bound SHA-256 over every immutable EngineModel field in the same
// semantic order used by WorldPack. This is stronger than the runtime's u64
// scheduling fingerprint and is the value carried by the Lab authoring export.
[[nodiscard]] bool digestNumiHumanLoadedKneeSourceModelV1(
    const EngineModel& model,
    NumiHumanLoadedKneeDigest& output,
    std::string& error
);

[[nodiscard]] bool prepareNumiHumanLoadedKneeMassV1(
    const NumiHumanLoadedKneeAuthoringV1& authoring,
    const EngineModel& source,
    std::span<const std::array<float, 3u>> referenceCoordinates,
    std::span<const MRBodyStateGPU> referenceBodies,
    std::span<const NumiHumanTissueMassNode> cookedNodes,
    NumiHumanLoadedKneeMassEvidenceV1& output,
    std::string& error
);

// Binds the exact compiled Matter world to the already-verified donor
// subtraction. Each cooked node mass is matched to the executable FEM arena,
// the donor moments are recomputed, and only then is the world's immutable
// authored-physics fingerprint admitted into the mass evidence.
[[nodiscard]] bool bindNumiHumanLoadedKneeMassToMatterWorldV1(
    const numi::matter::CompiledWorld& world,
    NumiHumanLoadedKneeMassEvidenceV1& evidence,
    std::string& error
);

// Recomputes the permanent profile-order ABI3 source-node, remapped-tet, and
// source-anchor identities; then proves that the compiled Matter rest-node and
// connectivity arenas plus the exact adapter anchor table use that ordering.
// Publication is atomic: evidence is unchanged on rejection.
[[nodiscard]] bool bindNumiHumanLoadedKneeExecutableTopologyV1(
    const NumiHumanLoadedKneeAuthoringV1& authoring,
    const NumiHumanKneePayload& decodedPayload,
    const numi::matter::CompiledWorld& world,
    std::span<const NumiHumanLoadedKneeExecutedAnchorV1> executedAnchors,
    NumiHumanLoadedKneeMassEvidenceV1& evidence,
    std::string& error
);

// Fail-closed material execution admission. The source path must contain the
// pinned exact FEBio exp-linear `.nmatter` asset. Each of the six compiled
// profile-order FEM programs must retain that asset's complete intrinsic IR,
// exact Human-authored material overlay, and the intended fibre ownership
// split (five neutral continua with fibre_scale=0, active QAT with
// fibre_scale=1). The five reduced rows must be in Human passive-owner order,
// carry exact c3/c4/c5/lambda/source-stretch float32 values, and retain exact
// enthesis-body ownership. Their local centroids, rest lengths, and areas are
// re-derived from the topology-authenticated adapter anchors plus executed
// x_ref/tetrahedra and donor COM offsets; finite but altered geometry is not
// accepted. Publication is atomic: evidence is unchanged on rejection.
[[nodiscard]] bool bindNumiHumanLoadedKneeMaterialExecutionV1(
    const NumiHumanLoadedKneeAuthoringV1& authoring,
    const std::filesystem::path& exactMatterSourcePath,
    const numi::matter::CompiledWorld& world,
    std::span<const NumiHumanLoadedKneeExecutedAnchorV1> executedAnchors,
    std::span<const NumiHumanLoadedKneePassiveOwnerV1>
        executedPassiveOwners,
    std::span<const NMNumiHumanPassiveLigamentGPU>
        executedPassiveLigaments,
    NumiHumanLoadedKneeMassEvidenceV1& evidence,
    std::string& error
);

// Separately binds the rigid donor subtraction to the exact EngineModel that
// completed the persistent articulated execution. Matter's physics
// fingerprint covers the continuum world and cannot stand in for this
// rigid-model identity. Call only after the articulated transaction succeeds.
[[nodiscard]] bool bindNumiHumanLoadedKneeMassToRigidExecutionV1(
    const EngineModel& executedModel,
    NumiHumanLoadedKneeMassEvidenceV1& evidence,
    std::string& error
);

[[nodiscard]] bool digestNumiHumanLoadedKneeXCurrentV1(
    const numi::matter::RuntimeStateSnapshot& snapshot,
    std::uint32_t firstNode,
    std::uint32_t nodeCount,
    NumiHumanLoadedKneeDigest& output,
    std::string& error
);

// Raw SHA-256 over exact float32 little-endian XYZ bytes in the permanent
// authoring region/source-node order. This is used for x_source, x_ref, and
// the separately Lab-owned runtime initialization state.
[[nodiscard]] bool digestNumiHumanLoadedKneeCoordinatesV1(
    std::span<const std::array<float, 3u>> coordinates,
    NumiHumanLoadedKneeDigest& output,
    std::string& error
);

[[nodiscard]] bool digestNumiHumanLoadedKneeCoordinateDerivationV1(
    const NumiHumanLoadedKneeAuthoringV1& authoring,
    const NumiHumanLoadedKneeDigest& initialCoordinateSHA256,
    const NumiHumanLoadedKneeDigest& referencePoseSHA256,
    const NumiHumanLoadedKneeDigest& initialPoseSHA256,
    NumiHumanLoadedKneeDigest& output,
    std::string& error
);

// ABI-bound exact bytes for non-Matter owners that are required to restore
// deterministic continuation. Callers must use one of the declared component
// IDs and provide the full accepted owner state, never a summary metric.
[[nodiscard]] bool digestNumiHumanLoadedKneeExternalComponentV1(
    std::string_view componentID,
    std::span<const std::byte> bytes,
    NumiHumanLoadedKneeDigest& output,
    std::string& error
);

[[nodiscard]] bool digestNumiHumanLoadedKneeMatterSnapshotV1(
    const numi::matter::RuntimeStateSnapshot& snapshot,
    NumiHumanLoadedKneeDigest& output,
    std::string& error
);

// Atomic publication: on every rejection, output is unchanged. The caller
// supplies the previously accepted receipt (or null for step one). Snapshot
// and x_current identities are direct-byte SHA-256 values, while the chained
// transaction root is domain-separated from the unavailable global root.
[[nodiscard]] bool acceptNumiHumanLoadedKneeRuntimeStateV1(
    const NumiHumanLoadedKneeAuthoringV1& authoring,
    const NumiHumanLoadedKneeMassEvidenceV1& mass,
    const NumiHumanLoadedKneeRuntimeEvidenceV1& evidence,
    const NumiHumanLoadedKneeAcceptanceReceiptV1* previous,
    NumiHumanLoadedKneeAcceptanceReceiptV1& output,
    std::string& error
);

[[nodiscard]] bool verifyNumiHumanLoadedKneeRestoreReplayV1(
    std::span<const NumiHumanLoadedKneeRuntimeEvidenceV1> accepted,
    std::span<const NumiHumanLoadedKneeRuntimeEvidenceV1> replayed,
    std::string& error
);

} // namespace metalrobo
