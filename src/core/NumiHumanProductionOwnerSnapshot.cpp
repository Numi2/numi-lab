#include "metalrobo/NumiHumanProductionOwnerSnapshot.hpp"

#include "metalrobo/compensated_translation_gpu.h"
#include "metalrobo/mujoco_muscle_gpu.h"
#include "metalrobo/numanx_human_matter_gpu.h"
#include "metalrobo/numi_human_stand_gpu.h"
#include "metalrobo/numi_human_tendon_gpu.h"
#include "numi/matter/human_limits_gpu.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string_view>

namespace metalrobo {
namespace {

constexpr std::uint64_t fnvOffset = 14695981039346656037ull;
constexpr std::uint64_t fnvPrime = 1099511628211ull;

// Persistent v1 stores the raw bytes of these already-versioned device ABIs.
// Keep their widths explicit here: accepting a caller-supplied width would let
// a type-confused payload receive a canonical field name and SHA-256 envelope.
constexpr std::uint32_t floatBytes = 4u;
constexpr std::uint32_t float4Bytes = 16u;
constexpr std::uint32_t compensatedRootBytes = 48u;
constexpr std::uint32_t muscleRecordBytes = 224u;
constexpr std::uint32_t muscleSiteBytes = 32u;
constexpr std::uint32_t muscleWrapBytes = 96u;
constexpr std::uint32_t muscleRouteNodeBytes = 16u;
constexpr std::uint32_t muscleStateBytes = 16u;
constexpr std::uint32_t muscleResultBytes = 96u;
constexpr std::uint32_t tendonTransferBytes = 112u;
constexpr std::uint32_t humanSupportRowBytes = 80u;
constexpr std::uint32_t humanSupportConsequenceBytes = 64u;
constexpr std::uint32_t equalityRowBytes = 112u;
constexpr std::uint32_t limitRowBytes = 80u;
constexpr std::uint32_t reconstructedConstraintWitnessBytes = 32u;
constexpr std::uint32_t contactSampleBytes = 160u;
constexpr std::uint32_t rigidReactionBytes = 32u;
constexpr std::uint32_t standStatusBytes = 272u;
constexpr std::uint32_t humanMatterOwnerStatusBytes = 96u;

static_assert(sizeof(float) == floatBytes);
static_assert(sizeof(mr_float4) == float4Bytes);
static_assert(sizeof(nm_float4) == float4Bytes);
static_assert(sizeof(MRCompensatedRootTranslationGPU) ==
    compensatedRootBytes);
static_assert(sizeof(MRMujocoMuscleGPU) == muscleRecordBytes);
static_assert(sizeof(MRMujocoMuscleSiteGPU) == muscleSiteBytes);
static_assert(sizeof(MRMujocoMuscleWrapGPU) == muscleWrapBytes);
static_assert(sizeof(MRMujocoMuscleRouteNodeGPU) == muscleRouteNodeBytes);
static_assert(sizeof(MRMujocoMuscleStateGPU) == muscleStateBytes);
static_assert(sizeof(MRMujocoMuscleResultGPU) == muscleResultBytes);
static_assert(sizeof(MRNumiHumanTendonTransferResultGPU) ==
    tendonTransferBytes);
static_assert(sizeof(NMHumanSupportContactGPU) == humanSupportRowBytes);
static_assert(sizeof(NMHumanSupportConsequenceGPU) ==
    humanSupportConsequenceBytes);
static_assert(sizeof(NMHumanJointEqualityGPU) == equalityRowBytes);
static_assert(sizeof(NMHumanJointLimitGPU) == limitRowBytes);
static_assert(sizeof(std::array<float, 8u>) ==
    reconstructedConstraintWitnessBytes);
static_assert(sizeof(NMContactSampleGPU) == contactSampleBytes);
static_assert(sizeof(NMRigidReactionGPU) == rigidReactionBytes);
static_assert(sizeof(MRNumiHumanStandStatusGPU) == standStatusBytes);
static_assert(sizeof(MRNumanXHumanMatterOwnerStatusGPU) ==
    humanMatterOwnerStatusBytes);

void append(std::uint64_t& hash, const std::span<const std::byte> bytes) {
    for (const auto value : bytes) {
        hash ^= std::to_integer<std::uint8_t>(value);
        hash *= fnvPrime;
    }
}

void appendU64(std::uint64_t& hash, const std::uint64_t value) {
    for (std::uint32_t index = 0u; index < 8u; ++index) {
        hash ^= static_cast<std::uint8_t>(value >> (index * 8u));
        hash *= fnvPrime;
    }
}

void appendPart(
    std::uint64_t& hash,
    const std::span<const std::byte> bytes) {
    appendU64(hash, static_cast<std::uint64_t>(bytes.size()));
    append(hash, bytes);
}

[[nodiscard]] bool arrayValid(
    const NumiHumanProductionOwnerArrayV1& array) noexcept {
    if (array.elementBytes == 0u) return false;
    if (!array.available) return array.bytes.empty();
    if (array.expectedElementCount >
        std::numeric_limits<std::uint64_t>::max() / array.elementBytes) {
        return false;
    }
    const std::uint64_t expectedBytes =
        array.expectedElementCount * array.elementBytes;
    return expectedBytes == array.bytes.size();
}

[[nodiscard]] bool allAvailable(
    const std::initializer_list<const NumiHumanProductionOwnerArrayV1*>
        arrays) noexcept {
    for (const auto* array : arrays) {
        if (array == nullptr || !array->available) return false;
    }
    return true;
}

void writeHex64(std::ostream& output, const std::uint64_t value) {
    output << '"' << std::hex << std::nouppercase << std::setfill('0')
           << std::setw(16) << value << std::dec << '"';
}

void writeArray(
    std::ostream& output,
    const char* name,
    const NumiHumanProductionOwnerArrayV1& array,
    const bool comma) {
    output << '"' << name << "\":{";
    output << "\"available\":" << (array.available ? "true" : "false")
           << ",\"expected_elements\":" << array.expectedElementCount
           << ",\"captured_elements\":"
           << (array.available ? array.expectedElementCount : 0u)
           << ",\"element_bytes\":" << array.elementBytes
           << ",\"words\":[";
    for (std::size_t offset = 0u; offset < array.bytes.size(); offset += 4u) {
        if (offset != 0u) output << ',';
        std::uint32_t word = 0u;
        const std::size_t remaining = array.bytes.size() - offset;
        const std::size_t count = remaining < 4u ? remaining : 4u;
        for (std::size_t byte = 0u; byte < count; ++byte) {
            word |= static_cast<std::uint32_t>(
                std::to_integer<std::uint8_t>(array.bytes[offset + byte]))
                << (8u * byte);
        }
        output << '"' << std::hex << std::nouppercase << std::setfill('0')
               << std::setw(8) << word << std::dec << '"';
    }
    output << "]}";
    if (comma) output << ',';
}

[[nodiscard]] const char* treatmentName(
    const NumiHumanProductionOwnerTreatmentV1 treatment) noexcept {
    switch (treatment) {
        case NumiHumanProductionOwnerTreatmentV1::cold: return "cold";
        case NumiHumanProductionOwnerTreatmentV1::seeded: return "seeded";
    }
    return nullptr;
}

[[nodiscard]] const char* dispositionName(
    const NumiHumanProductionOwnerDispositionV1 disposition) noexcept {
    switch (disposition) {
        case NumiHumanProductionOwnerDispositionV1::published:
            return "published";
        case NumiHumanProductionOwnerDispositionV1::rejected:
            return "rejected";
    }
    return nullptr;
}

} // namespace

NumiHumanProductionOwnerSnapshotV1::
NumiHumanProductionOwnerSnapshotV1() noexcept {
    initialQ.elementBytes = floatBytes;
    initialV.elementBytes = floatBytes;
    initialRoot.elementBytes = compensatedRootBytes;
    initialMuscles.elementBytes = muscleStateBytes;
    muscleRecords.elementBytes = muscleRecordBytes;
    muscleSites.elementBytes = muscleSiteBytes;
    muscleWraps.elementBytes = muscleWrapBytes;
    muscleRouteNodes.elementBytes = muscleRouteNodeBytes;
    checkpointQ.elementBytes = floatBytes;
    checkpointV.elementBytes = floatBytes;
    checkpointRoot.elementBytes = compensatedRootBytes;
    checkpointMuscles.elementBytes = muscleStateBytes;
    effectiveTangentFactorStorage.elementBytes = floatBytes;
    sourceGeneralizedForce.elementBytes = floatBytes;
    sourcePredictedVelocity.elementBytes = floatBytes;
    candidateQ.elementBytes = floatBytes;
    candidateV.elementBytes = floatBytes;
    candidateRoot.elementBytes = compensatedRootBytes;
    candidateMuscles.elementBytes = muscleStateBytes;
    muscleResults.elementBytes = muscleResultBytes;
    muscleGeneralizedForces.elementBytes = floatBytes;
    reducedMuscleGeneralizedForce.elementBytes = floatBytes;
    tendonTransfers.elementBytes = tendonTransferBytes;
    tendonGeneralizedCorrections.elementBytes = floatBytes;
    matterGeneralizedReaction.elementBytes = floatBytes;
    supportRows.elementBytes = humanSupportRowBytes;
    supportPlane.elementBytes = float4Bytes;
    initialSupportHistories.elementBytes = float4Bytes;
    candidateSupportHistories.elementBytes = float4Bytes;
    candidateSupportConsequences.elementBytes =
        humanSupportConsequenceBytes;
    terminalAcceptedSupportHistories.elementBytes = float4Bytes;
    terminalAcceptedSupportConsequences.elementBytes =
        humanSupportConsequenceBytes;
    equalityRows.elementBytes = equalityRowBytes;
    equalityLinearizationImpulses.elementBytes =
        reconstructedConstraintWitnessBytes;
    limitRows.elementBytes = limitRowBytes;
    limitLinearizationImpulses.elementBytes =
        reconstructedConstraintWitnessBytes;
    contactSamples.elementBytes = contactSampleBytes;
    terminalAcceptedMatterRigidGeneralizedState.elementBytes = floatBytes;
    terminalAcceptedMatterRigidReactions.elementBytes = rigidReactionBytes;
    standStatus.elementBytes = standStatusBytes;
    humanMatterOwnerStatus.elementBytes = humanMatterOwnerStatusBytes;
    acceleration.elementBytes = floatBytes;
    sourceRHS.elementBytes = floatBytes;
    sourceBias.elementBytes = floatBytes;
    workEnergyComponents.elementBytes = floatBytes;
}

std::uint64_t numiHumanProductionOwnerBaseStateFingerprintV1(
    const std::uint64_t humanSourceWithoutInitialHistory,
    const std::uint64_t matterWorldFingerprint,
    const std::uint64_t timestepNanoseconds,
    const std::span<const std::byte> q,
    const std::span<const std::byte> v,
    const std::span<const std::byte> root,
    const std::span<const std::byte> muscles) noexcept {
    constexpr std::string_view domain =
        "numi.human.production-owner.base-state.v1";
    std::uint64_t hash = fnvOffset;
    append(hash, std::as_bytes(std::span(domain.data(), domain.size())));
    appendU64(hash, humanSourceWithoutInitialHistory);
    appendU64(hash, matterWorldFingerprint);
    appendU64(hash, timestepNanoseconds);
    appendPart(hash, q);
    appendPart(hash, v);
    appendPart(hash, root);
    appendPart(hash, muscles);
    return hash == 0u ? fnvOffset : hash;
}

std::uint64_t numiHumanProductionOwnerTreatmentHistoryFingerprintV1(
    const std::span<const std::byte> supportPayloadIdentity,
    const std::span<const std::byte> acceptedSupportHistory) noexcept {
    constexpr std::string_view domain =
        "numi.human.production-owner.treatment-history.v1";
    std::uint64_t hash = fnvOffset;
    append(hash, std::as_bytes(std::span(domain.data(), domain.size())));
    appendPart(hash, supportPayloadIdentity);
    appendPart(hash, acceptedSupportHistory);
    return hash == 0u ? fnvOffset : hash;
}

std::uint64_t numiHumanProductionOwnerCoverageMaskV1(
    const NumiHumanProductionOwnerSnapshotV1& s) noexcept {
    std::uint64_t result = 0u;
    const auto cover = [&result](const bool present, const std::uint64_t bit) {
        if (present) result |= bit;
    };
    cover(allAvailable({&s.initialQ, &s.initialV, &s.initialRoot,
        &s.initialMuscles}), NumiHumanOwnerInitialStateV1);
    cover(allAvailable({&s.checkpointQ, &s.checkpointV, &s.checkpointRoot,
        &s.checkpointMuscles}), NumiHumanOwnerCheckpointStateV1);
    cover(s.effectiveTangentFactorStorage.available,
        NumiHumanOwnerEffectiveTangentFactorV1);
    cover(s.sourceGeneralizedForce.available,
        NumiHumanOwnerSourceGeneralizedForceV1);
    cover(s.sourcePredictedVelocity.available, NumiHumanOwnerFreeVelocityV1);
    cover(allAvailable({&s.candidateQ, &s.candidateV, &s.candidateRoot}),
        NumiHumanOwnerCandidateStateV1);
    cover(s.candidateMuscles.available, NumiHumanOwnerMuscleStateV1);
    cover(s.muscleResults.available, NumiHumanOwnerMuscleResultsV1);
    cover(allAvailable({&s.muscleGeneralizedForces,
        &s.reducedMuscleGeneralizedForce}),
        NumiHumanOwnerMuscleGeneralizedForcesV1);
    cover(allAvailable({&s.tendonTransfers,
        &s.tendonGeneralizedCorrections}), NumiHumanOwnerTendonV1);
    cover(s.matterGeneralizedReaction.available,
        NumiHumanOwnerMatterReactionV1);
    cover(allAvailable({&s.supportRows, &s.supportPlane}),
        NumiHumanOwnerSupportRowsV1);
    cover(allAvailable({&s.initialSupportHistories,
        &s.candidateSupportHistories, &s.candidateSupportConsequences,
        &s.terminalAcceptedSupportHistories,
        &s.terminalAcceptedSupportConsequences}),
        NumiHumanOwnerSupportHistoryV1);
    cover(allAvailable({&s.equalityRows,
        &s.equalityLinearizationImpulses}),
        NumiHumanOwnerEqualityRowsV1);
    cover(allAvailable({&s.limitRows, &s.limitLinearizationImpulses}),
        NumiHumanOwnerLimitRowsV1);
    cover(allAvailable({&s.acceleration, &s.sourceRHS, &s.sourceBias}),
        NumiHumanOwnerRHSBiasAccelerationV1);
    cover(s.workEnergyComponents.available, NumiHumanOwnerWorkEnergyV1);
    cover(s.publicationIdentityAvailable,
        NumiHumanOwnerPublicationIdentityV1);
    cover(s.rollbackIdentityAvailable, NumiHumanOwnerRollbackIdentityV1);
    cover(s.contactSamples.available, NumiHumanOwnerContactSamplesV1);
    cover(allAvailable({&s.muscleRecords, &s.muscleSites,
        &s.muscleWraps, &s.muscleRouteNodes}),
        NumiHumanOwnerMuscleProgramV1);
    cover(s.disposition ==
            NumiHumanProductionOwnerDispositionV1::published &&
        allAvailable({&s.terminalAcceptedMatterRigidGeneralizedState,
            &s.terminalAcceptedMatterRigidReactions}),
        NumiHumanOwnerMatterIntegrationUpdateV1);
    cover(allAvailable({&s.standStatus, &s.humanMatterOwnerStatus}),
        NumiHumanOwnerStatusRecordsV1);
    return result;
}

bool serializeNumiHumanProductionOwnerSnapshotV1(
    const NumiHumanProductionOwnerSnapshotV1& s,
    std::string& output,
    std::string& error) {
    const char* treatment = treatmentName(s.treatment);
    const char* disposition = dispositionName(s.disposition);
    if (s.formatVersion != kNumiHumanProductionOwnerSnapshotVersionV1 ||
        treatment == nullptr || disposition == nullptr) {
        error = "production-owner snapshot version/treatment/disposition invalid";
        return false;
    }
    if (s.baseStateFingerprint == 0u ||
        s.treatmentHistoryFingerprint == 0u ||
        s.humanSourceFingerprint == 0u ||
        s.matterSourcePhysicsFingerprint == 0u ||
        s.matterDeviceProgramFingerprint == 0u ||
        s.ownerProgramFingerprint == 0u || s.transactionFingerprint == 0u ||
        s.slotGeneration == 0u || s.physicsGeneration == 0u ||
        s.brainGeneration == 0u || s.sensorGeneration == 0u ||
        s.humanIOProgramFingerprint == 0u || s.sensorFingerprint == 0u ||
        s.transactionInstanceFingerprint == 0u ||
        s.timestepNanoseconds == 0u ||
        s.qCoordinateCount == 0u || s.dofCount == 0u ||
        s.muscleCount == 0u) {
        error = "production-owner snapshot identity/dimensions incomplete";
        return false;
    }
    if (s.supportPayloadSHA256.size() != 32u ||
        s.supportPayloadByteCount == 0u || s.supportPayloadABI == 0u ||
        std::all_of(s.supportPayloadSHA256.begin(),
            s.supportPayloadSHA256.end(),
            [](const std::byte value) { return value == std::byte{0}; }) ||
        ((s.equalityRowCount == 0u) !=
            (s.equalityProgramFingerprint == 0u)) ||
        ((s.limitRowCount == 0u) !=
            (s.limitProgramFingerprint == 0u))) {
        error = "production-owner row-program identity is incomplete";
        return false;
    }
    const std::array<const NumiHumanProductionOwnerArrayV1*, 45u> arrays{{
        &s.initialQ, &s.initialV, &s.initialRoot, &s.initialMuscles,
        &s.muscleRecords, &s.muscleSites, &s.muscleWraps,
        &s.muscleRouteNodes,
        &s.checkpointQ, &s.checkpointV, &s.checkpointRoot,
        &s.checkpointMuscles, &s.effectiveTangentFactorStorage,
        &s.sourceGeneralizedForce, &s.sourcePredictedVelocity,
        &s.candidateQ, &s.candidateV, &s.candidateRoot,
        &s.candidateMuscles, &s.muscleResults,
        &s.muscleGeneralizedForces, &s.reducedMuscleGeneralizedForce,
        &s.tendonTransfers, &s.tendonGeneralizedCorrections,
        &s.matterGeneralizedReaction, &s.supportRows, &s.supportPlane,
        &s.initialSupportHistories, &s.candidateSupportHistories,
        &s.candidateSupportConsequences,
        &s.terminalAcceptedSupportHistories,
        &s.terminalAcceptedSupportConsequences, &s.equalityRows,
        &s.equalityLinearizationImpulses, &s.limitRows,
        &s.limitLinearizationImpulses, &s.contactSamples,
        &s.terminalAcceptedMatterRigidGeneralizedState,
        &s.terminalAcceptedMatterRigidReactions,
        &s.standStatus, &s.humanMatterOwnerStatus,
        &s.acceleration, &s.sourceRHS, &s.sourceBias,
        &s.workEnergyComponents}};
    for (const auto* array : arrays) {
        if (!arrayValid(*array)) {
            error = "production-owner snapshot array is truncated or malformed";
            return false;
        }
    }
    const auto shape = [](
        const NumiHumanProductionOwnerArrayV1& array,
        const std::uint64_t expected,
        const std::uint32_t expectedElementBytes) noexcept {
        return array.elementBytes == expectedElementBytes &&
            (!array.available || array.expectedElementCount == expected);
    };
    const bool shapesValid =
        shape(s.initialQ, s.qCoordinateCount, floatBytes) &&
        shape(s.initialV, s.dofCount, floatBytes) &&
        shape(s.initialRoot, 1u, compensatedRootBytes) &&
        shape(s.initialMuscles, s.muscleCount, muscleStateBytes) &&
        shape(s.muscleRecords, s.muscleCount, muscleRecordBytes) &&
        shape(s.muscleSites, s.muscleSiteCount, muscleSiteBytes) &&
        shape(s.muscleWraps, s.muscleWrapCount, muscleWrapBytes) &&
        shape(s.muscleRouteNodes,
            s.muscleRouteNodeCount, muscleRouteNodeBytes) &&
        shape(s.checkpointQ, s.qCoordinateCount, floatBytes) &&
        shape(s.checkpointV, s.dofCount, floatBytes) &&
        shape(s.checkpointRoot, 1u, compensatedRootBytes) &&
        shape(s.checkpointMuscles, s.muscleCount, muscleStateBytes) &&
        shape(s.effectiveTangentFactorStorage,
            static_cast<std::uint64_t>(s.dofCount) * s.dofCount,
            floatBytes) &&
        shape(s.sourceGeneralizedForce, s.dofCount, floatBytes) &&
        shape(s.sourcePredictedVelocity, s.dofCount, floatBytes) &&
        shape(s.candidateQ, s.qCoordinateCount, floatBytes) &&
        shape(s.candidateV, s.dofCount, floatBytes) &&
        shape(s.candidateRoot, 1u, compensatedRootBytes) &&
        shape(s.candidateMuscles, s.muscleCount, muscleStateBytes) &&
        shape(s.muscleResults, s.muscleCount, muscleResultBytes) &&
        shape(s.muscleGeneralizedForces,
            static_cast<std::uint64_t>(s.muscleCount) * s.dofCount,
            floatBytes) &&
        shape(s.reducedMuscleGeneralizedForce, s.dofCount, floatBytes) &&
        shape(s.tendonTransfers, s.tendonRowCount, tendonTransferBytes) &&
        shape(s.tendonGeneralizedCorrections,
            static_cast<std::uint64_t>(s.tendonRowCount) * s.dofCount,
            floatBytes) &&
        shape(s.matterGeneralizedReaction, s.dofCount, floatBytes) &&
        shape(s.supportRows,
            s.supportRowCount, humanSupportRowBytes) &&
        shape(s.supportPlane, 2u, float4Bytes) &&
        shape(s.initialSupportHistories,
            s.supportRowCount, float4Bytes) &&
        shape(s.candidateSupportHistories,
            s.supportRowCount, float4Bytes) &&
        shape(s.candidateSupportConsequences,
            s.supportRowCount, humanSupportConsequenceBytes) &&
        shape(s.terminalAcceptedSupportHistories,
            s.supportRowCount, float4Bytes) &&
        shape(s.terminalAcceptedSupportConsequences,
            s.supportRowCount, humanSupportConsequenceBytes) &&
        shape(s.equalityRows, s.equalityRowCount, equalityRowBytes) &&
        shape(s.equalityLinearizationImpulses,
            s.equalityRowCount, reconstructedConstraintWitnessBytes) &&
        shape(s.limitRows, s.limitRowCount, limitRowBytes) &&
        shape(s.limitLinearizationImpulses,
            static_cast<std::uint64_t>(2u) * s.limitRowCount,
            reconstructedConstraintWitnessBytes) &&
        shape(s.contactSamples, s.contactSampleCount, contactSampleBytes) &&
        shape(s.terminalAcceptedMatterRigidGeneralizedState,
            s.terminalAcceptedMatterRigidGeneralizedStateCount,
            floatBytes) &&
        shape(s.terminalAcceptedMatterRigidReactions,
            s.terminalAcceptedMatterRigidReactionCount,
            rigidReactionBytes) &&
        shape(s.acceleration, s.dofCount, floatBytes) &&
        shape(s.sourceRHS, s.dofCount, floatBytes) &&
        shape(s.sourceBias, s.dofCount, floatBytes) &&
        shape(s.workEnergyComponents, 3u, floatBytes) &&
        shape(s.standStatus, 1u, standStatusBytes) &&
        shape(s.humanMatterOwnerStatus,
            1u, humanMatterOwnerStatusBytes);
    if (!shapesValid) {
        error = "production-owner snapshot logical shape mismatch";
        return false;
    }
    const std::uint64_t actualCoverage =
        numiHumanProductionOwnerCoverageMaskV1(s);
    if (s.truncationMask != 0u || s.coverageMask != actualCoverage ||
        (s.coverageMask & s.requiredCoverageMask) !=
            s.requiredCoverageMask) {
        error = "production-owner snapshot coverage/truncation contract failed";
        return false;
    }
    const bool publishedIdentity =
        s.publicationIdentityAvailable && s.publicationEpoch != 0u &&
        s.jointFenceFingerprint != 0u && !s.rollbackIdentityAvailable;
    const bool rejectedIdentity =
        s.rollbackIdentityAvailable && !s.publicationIdentityAvailable &&
        s.publicationEpoch == 0u && s.jointFenceFingerprint == 0u;
    if ((s.disposition ==
            NumiHumanProductionOwnerDispositionV1::published &&
         !publishedIdentity) ||
        (s.disposition ==
            NumiHumanProductionOwnerDispositionV1::rejected &&
         !rejectedIdentity)) {
        error = "production-owner terminal identity contradicts disposition";
        return false;
    }

    std::ostringstream json;
    json.imbue(std::locale::classic());
    json << "{\"schema\":\"" << kNumiHumanProductionOwnerSnapshotSchemaV1
         << "\",\"format_version\":" << s.formatVersion
         << ",\"comparison_identity_algorithm\":\"fnv1a64-domain-v1\""
         << ",\"comparison_identity_is_cryptographic_proof\":false"
         << ",\"rhs_bias_origin\":\"host-reconstructed-from-captured-A0-v0-vfree-tau\""
         << ",\"constraint_witness_origin\":\"host-reconstructed-from-captured-production-inputs\""
         << ",\"inertial_operator_capture\":\"source-effective-tangent-factor-storage\""
         << ",\"effective_tangent_factor_storage_layout\":\"row-major-lower-cholesky-upper-source-A0\""
         << ",\"rollback_identity_scope\":\"lifecycle-terminal-disposition-only-not-byte-restoration-proof\""
         << ",\"terminal_matter_state_role\":\""
         << (s.disposition ==
                 NumiHumanProductionOwnerDispositionV1::published
             ? "published-attempt-accepted-state"
             : "prior-accepted-state-context-after-rejection")
         << '"'
         << ",\"work_energy_scope\":\"solver-components-not-physical-energy-closure\""
         << ",\"treatment\":\"" << treatment
         << "\",\"disposition\":\"" << disposition << "\",";
    const auto identity = [&json](const char* name, const std::uint64_t value) {
        json << '"' << name << "\":";
        writeHex64(json, value);
        json << ',';
    };
    identity("base_state_fingerprint", s.baseStateFingerprint);
    identity("treatment_history_fingerprint",
        s.treatmentHistoryFingerprint);
    identity("human_source_fingerprint", s.humanSourceFingerprint);
    identity("matter_source_physics_fingerprint",
        s.matterSourcePhysicsFingerprint);
    identity("matter_device_program_fingerprint",
        s.matterDeviceProgramFingerprint);
    identity("owner_program_fingerprint", s.ownerProgramFingerprint);
    identity("transaction_fingerprint", s.transactionFingerprint);
    identity("previous_transaction_fingerprint",
        s.previousTransactionFingerprint);
    identity("linearization_epoch", s.linearizationEpoch);
    identity("slot_generation", s.slotGeneration);
    identity("physics_generation", s.physicsGeneration);
    identity("previous_physics_generation", s.previousPhysicsGeneration);
    identity("brain_generation", s.brainGeneration);
    identity("sensor_generation", s.sensorGeneration);
    identity("human_io_program_fingerprint",
        s.humanIOProgramFingerprint);
    identity("sensor_fingerprint", s.sensorFingerprint);
    identity("transaction_instance_fingerprint",
        s.transactionInstanceFingerprint);
    identity("joint_fence_fingerprint", s.jointFenceFingerprint);
    identity("equality_program_fingerprint",
        s.equalityProgramFingerprint);
    identity("limit_program_fingerprint", s.limitProgramFingerprint);
    identity("coverage_mask", s.coverageMask);
    identity("required_coverage_mask", s.requiredCoverageMask);
    identity("truncation_mask", s.truncationMask);
    json << "\"control_step\":" << s.controlStep
         << ",\"candidate_timestamp_nanoseconds\":"
         << s.candidateTimestampNanoseconds
         << ",\"publication_epoch\":" << s.publicationEpoch
         << ",\"timestep_nanoseconds\":" << s.timestepNanoseconds
         << ",\"support_payload_byte_count\":"
         << s.supportPayloadByteCount
         << ",\"support_payload_abi\":" << s.supportPayloadABI
         << ",\"support_payload_sha256\":\"";
    for (const auto byte : s.supportPayloadSHA256) {
        json << std::hex << std::nouppercase << std::setfill('0')
             << std::setw(2)
             << static_cast<unsigned>(std::to_integer<std::uint8_t>(byte));
    }
    json << std::dec << '"'
         << ",\"matter_control_step\":" << s.matterControlStep
         << ",\"q_coordinate_count\":" << s.qCoordinateCount
         << ",\"dof_count\":" << s.dofCount
         << ",\"muscle_count\":" << s.muscleCount
         << ",\"muscle_site_count\":" << s.muscleSiteCount
         << ",\"muscle_wrap_count\":" << s.muscleWrapCount
         << ",\"muscle_route_node_count\":" << s.muscleRouteNodeCount
         << ",\"support_row_count\":" << s.supportRowCount
         << ",\"equality_row_count\":" << s.equalityRowCount
         << ",\"limit_row_count\":" << s.limitRowCount
         << ",\"tendon_row_count\":" << s.tendonRowCount
         << ",\"contact_sample_count\":" << s.contactSampleCount
         << ",\"terminal_accepted_matter_rigid_generalized_state_count\":"
         << s.terminalAcceptedMatterRigidGeneralizedStateCount
         << ",\"terminal_accepted_matter_rigid_reaction_count\":"
         << s.terminalAcceptedMatterRigidReactionCount
         << ",\"publication_identity_available\":"
         << (s.publicationIdentityAvailable ? "true" : "false")
         << ",\"rollback_identity_available\":"
         << (s.rollbackIdentityAvailable ? "true" : "false")
         << ",\"arrays\":{";
    writeArray(json, "initial_q", s.initialQ, true);
    writeArray(json, "initial_v", s.initialV, true);
    writeArray(json, "initial_root", s.initialRoot, true);
    writeArray(json, "initial_muscles", s.initialMuscles, true);
    writeArray(json, "muscle_records", s.muscleRecords, true);
    writeArray(json, "muscle_sites", s.muscleSites, true);
    writeArray(json, "muscle_wraps", s.muscleWraps, true);
    writeArray(json, "muscle_route_nodes", s.muscleRouteNodes, true);
    writeArray(json, "checkpoint_q", s.checkpointQ, true);
    writeArray(json, "checkpoint_v", s.checkpointV, true);
    writeArray(json, "checkpoint_root", s.checkpointRoot, true);
    writeArray(json, "checkpoint_muscles", s.checkpointMuscles, true);
    writeArray(json, "effective_tangent_factor_storage",
        s.effectiveTangentFactorStorage, true);
    writeArray(json, "source_generalized_force",
        s.sourceGeneralizedForce, true);
    writeArray(json, "source_predicted_velocity",
        s.sourcePredictedVelocity, true);
    writeArray(json, "candidate_q", s.candidateQ, true);
    writeArray(json, "candidate_v", s.candidateV, true);
    writeArray(json, "candidate_root", s.candidateRoot, true);
    writeArray(json, "candidate_muscles", s.candidateMuscles, true);
    writeArray(json, "muscle_results", s.muscleResults, true);
    writeArray(json, "muscle_generalized_forces",
        s.muscleGeneralizedForces, true);
    writeArray(json, "reduced_muscle_generalized_force",
        s.reducedMuscleGeneralizedForce, true);
    writeArray(json, "tendon_transfers", s.tendonTransfers, true);
    writeArray(json, "tendon_generalized_corrections",
        s.tendonGeneralizedCorrections, true);
    writeArray(json, "matter_generalized_reaction",
        s.matterGeneralizedReaction, true);
    writeArray(json, "support_rows", s.supportRows, true);
    writeArray(json, "support_plane", s.supportPlane, true);
    writeArray(json, "initial_support_histories",
        s.initialSupportHistories, true);
    writeArray(json, "candidate_support_histories",
        s.candidateSupportHistories, true);
    writeArray(json, "candidate_support_consequences",
        s.candidateSupportConsequences, true);
    writeArray(json, "terminal_accepted_support_histories",
        s.terminalAcceptedSupportHistories, true);
    writeArray(json, "terminal_accepted_support_consequences",
        s.terminalAcceptedSupportConsequences, true);
    writeArray(json, "equality_rows", s.equalityRows, true);
    writeArray(json, "equality_reconstructed_linearization_impulses",
        s.equalityLinearizationImpulses, true);
    writeArray(json, "limit_rows", s.limitRows, true);
    writeArray(json, "limit_reconstructed_linearization_impulses",
        s.limitLinearizationImpulses, true);
    writeArray(json, "contact_samples", s.contactSamples, true);
    writeArray(json, "terminal_accepted_matter_rigid_generalized_state",
        s.terminalAcceptedMatterRigidGeneralizedState, true);
    writeArray(json, "terminal_accepted_matter_rigid_reactions",
        s.terminalAcceptedMatterRigidReactions, true);
    writeArray(json, "stand_status", s.standStatus, true);
    writeArray(json, "human_matter_owner_status",
        s.humanMatterOwnerStatus, true);
    writeArray(json, "acceleration", s.acceleration, true);
    writeArray(json, "source_rhs", s.sourceRHS, true);
    writeArray(json, "source_bias", s.sourceBias, true);
    writeArray(json, "work_energy_components",
        s.workEnergyComponents, false);
    json << "}}";
    output = json.str();
    error.clear();
    return true;
}

bool serializeNumiHumanProductionOwnerSnapshotEvidenceV1(
    const NumiHumanProductionOwnerSnapshotV1& snapshot,
    std::string& output,
    std::string& payloadSHA256,
    std::string& error) {
    std::string payload;
    if (!serializeNumiHumanProductionOwnerSnapshotV1(
            snapshot, payload, error)) {
        return false;
    }
    if (payload.size() > std::numeric_limits<CC_LONG>::max()) {
        error = "production-owner evidence exceeds SHA-256 input extent";
        return false;
    }
    std::array<std::uint8_t, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256(payload.data(), static_cast<CC_LONG>(payload.size()),
        digest.data());
    std::ostringstream digestHex;
    digestHex << std::hex << std::nouppercase << std::setfill('0');
    for (const auto byte : digest)
        digestHex << std::setw(2) << static_cast<unsigned>(byte);
    payloadSHA256 = digestHex.str();
    output = std::string{"{\"evidence_schema\":\""} +
        kNumiHumanProductionOwnerSnapshotEvidenceSchemaV1 +
        "\",\"payload_sha256\":\"" + payloadSHA256 +
        "\",\"payload\":" + payload + '}';
    error.clear();
    return true;
}

} // namespace metalrobo
