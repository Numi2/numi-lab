#include "metalrobo/NumiHumanProductionOwnerSnapshot.hpp"

#include "metalrobo/compensated_translation_gpu.h"
#include "metalrobo/mujoco_muscle_gpu.h"
#include "numi/matter/shared.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "Numi Human production-owner snapshot: %s\n", message);
    std::exit(1);
}

void require(const bool condition, const char* message) {
    if (!condition) fail(message);
}

template <typename T>
std::span<const std::byte> bytesOf(const std::vector<T>& values) {
    return std::as_bytes(std::span(values));
}

template <typename T>
std::span<const std::byte> bytesOf(const T& value) {
    return std::as_bytes(std::span(&value, 1u));
}

template <typename T>
metalrobo::NumiHumanProductionOwnerArrayV1 captured(
    const std::vector<T>& values) {
    metalrobo::NumiHumanProductionOwnerArrayV1 result;
    result.available = true;
    result.expectedElementCount = values.size();
    result.elementBytes = sizeof(T);
    const auto bytes = bytesOf(values);
    result.bytes.assign(bytes.begin(), bytes.end());
    return result;
}

metalrobo::NumiHumanProductionOwnerSnapshotV1 record() {
    using namespace metalrobo;
    const std::vector<float> q{1.0f, -0.0f, 3.5f};
    const std::vector<float> v{0.25f, -2.0f};
    MRCompensatedRootTranslationGPU root{};
    root.reference.x = 1.0f;
    std::vector<MRMujocoMuscleStateGPU> muscles(1u);
    muscles[0].excitationAndActivation = {0.1f, 0.2f, 0.3f, 0.4f};
    const std::vector<MRMujocoMuscleGPU> muscleRecords(1u);
    const std::vector<MRMujocoMuscleSiteGPU> muscleSites(2u);
    const std::vector<MRMujocoMuscleWrapGPU> muscleWraps(1u);
    const std::vector<MRMujocoMuscleRouteNodeGPU> muscleRouteNodes(3u);
    // Row-major storage: lower triangle is Cholesky L, upper triangle retains
    // the corresponding source A0 word.
    const std::vector<float> effectiveTangentFactorStorage{
        2.0f, 0.25f,
        0.5f, 1.5f};
    const std::vector<NMContactSampleGPU> contactSamples(1u);
    const std::vector<float> terminalMatterGeneralizedState{
        0.75f, -0.5f};
    const std::vector<NMRigidReactionGPU> terminalMatterReactions(1u);
    const std::array<std::uint8_t, 8u> supportIdentity{
        1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};
    const std::vector<std::array<float, 4u>> coldHistory{};

    NumiHumanProductionOwnerSnapshotV1 result;
    result.treatment = NumiHumanProductionOwnerTreatmentV1::cold;
    result.disposition = NumiHumanProductionOwnerDispositionV1::published;
    result.baseStateFingerprint =
        numiHumanProductionOwnerBaseStateFingerprintV1(
            0x101u, 0x202u, 12'500u, bytesOf(q), bytesOf(v),
            bytesOf(root), bytesOf(muscles));
    result.treatmentHistoryFingerprint =
        numiHumanProductionOwnerTreatmentHistoryFingerprintV1(
            bytesOf(supportIdentity), bytesOf(coldHistory));
    result.humanSourceFingerprint = 0x303u;
    result.matterSourcePhysicsFingerprint = 0x404u;
    result.matterDeviceProgramFingerprint = 0x505u;
    result.ownerProgramFingerprint = 0x606u;
    result.transactionFingerprint = 0x707u;
    result.previousTransactionFingerprint = 0x6060u;
    result.linearizationEpoch = 8u;
    result.slotGeneration = 9u;
    // Step zero is a valid first accepted interval in the exact-clock family.
    result.controlStep = 0u;
    result.physicsGeneration = 10u;
    result.previousPhysicsGeneration = 9u;
    result.brainGeneration = 12u;
    result.sensorGeneration = 13u;
    result.humanIOProgramFingerprint = 0x909u;
    result.sensorFingerprint = 0xa0au;
    result.transactionInstanceFingerprint = 0xb0bu;
    result.candidateTimestampNanoseconds = 12'500u;
    result.publicationEpoch = 11u;
    result.jointFenceFingerprint = 0x808u;
    result.timestepNanoseconds = 12'500u;
    result.supportPayloadByteCount = 84u;
    result.supportPayloadABI = 1u;
    result.supportPayloadSHA256.resize(32u, std::byte{1u});
    result.qCoordinateCount = q.size();
    result.dofCount = v.size();
    result.muscleCount = muscles.size();
    result.muscleSiteCount = muscleSites.size();
    result.muscleWrapCount = muscleWraps.size();
    result.muscleRouteNodeCount = muscleRouteNodes.size();
    result.contactSampleCount = contactSamples.size();
    result.terminalAcceptedMatterRigidGeneralizedStateCount =
        terminalMatterGeneralizedState.size();
    result.terminalAcceptedMatterRigidReactionCount =
        terminalMatterReactions.size();
    result.publicationIdentityAvailable = true;
    result.initialQ = captured(q);
    result.initialV = captured(v);
    result.initialRoot.available = true;
    result.initialRoot.expectedElementCount = 1u;
    result.initialRoot.elementBytes = sizeof(root);
    const auto rootBytes = bytesOf(root);
    result.initialRoot.bytes.assign(rootBytes.begin(), rootBytes.end());
    result.initialMuscles = captured(muscles);
    result.muscleRecords = captured(muscleRecords);
    result.muscleSites = captured(muscleSites);
    result.muscleWraps = captured(muscleWraps);
    result.muscleRouteNodes = captured(muscleRouteNodes);
    result.effectiveTangentFactorStorage =
        captured(effectiveTangentFactorStorage);
    result.tendonTransfers.available = true;
    result.tendonTransfers.expectedElementCount = 0u;
    result.tendonTransfers.elementBytes = 112u;
    result.tendonGeneralizedCorrections.available = true;
    result.tendonGeneralizedCorrections.expectedElementCount = 0u;
    result.tendonGeneralizedCorrections.elementBytes = sizeof(float);
    result.contactSamples = captured(contactSamples);
    result.terminalAcceptedMatterRigidGeneralizedState =
        captured(terminalMatterGeneralizedState);
    result.terminalAcceptedMatterRigidReactions =
        captured(terminalMatterReactions);
    result.coverageMask = numiHumanProductionOwnerCoverageMaskV1(result);
    result.requiredCoverageMask = NumiHumanOwnerInitialStateV1 |
        NumiHumanOwnerPublicationIdentityV1;
    return result;
}

} // namespace

int main() {
    using namespace metalrobo;
    auto source = record();
    require(source.baseStateFingerprint == 0x72655bd196980b6aull,
        "base-state fingerprint golden changed");
    require(source.treatmentHistoryFingerprint == 0x70a7461fffdcf6beull,
        "cold-history fingerprint golden changed");

    std::string first;
    std::string second;
    std::string error;
    require(serializeNumiHumanProductionOwnerSnapshotV1(
        source, first, error), "valid record did not serialize");
    require(serializeNumiHumanProductionOwnerSnapshotV1(
        source, second, error), "repeat serialization failed");
    require(first == second, "serialization is not deterministic");
    require(first.find("\"schema\":\"persistent-production-owner-snapshot.v1\"")
            != std::string::npos,
        "schema marker missing");
    require(first.find("\"control_step\":0") != std::string::npos,
        "control step zero was not preserved");
    require(first.find("\"muscle_site_count\":2") != std::string::npos &&
            first.find("\"muscle_wrap_count\":1") != std::string::npos &&
            first.find("\"muscle_route_node_count\":3") !=
                std::string::npos,
        "independent muscle-program counts were not serialized");
    require(first.find("\"candidate_timestamp_nanoseconds\":12500") !=
            std::string::npos &&
            first.find("\"accepted_timestamp_nanoseconds\"") ==
                std::string::npos,
        "candidate timestamp was mislabeled as accepted state");
    require(first.find("\"comparison_identity_is_cryptographic_proof\":false")
            != std::string::npos,
        "non-cryptographic comparison identity boundary missing");
    require(first.find(
            "\"effective_tangent_factor_storage_layout\":\"row-major-lower-cholesky-upper-source-A0\"") !=
            std::string::npos &&
            first.find(
                "\"inertial_operator_capture\":\"source-effective-tangent-factor-storage\"") !=
                std::string::npos &&
            first.find("\"effective_tangent_factor_storage\"") !=
                std::string::npos &&
            first.find("\"effective_tangent_lower\"") == std::string::npos,
        "effective-tangent factor storage layout is mislabeled");
    require(first.find(
            "\"terminal_matter_state_role\":\"published-attempt-accepted-state\"") !=
            std::string::npos &&
            first.find(
                "\"terminal_accepted_matter_rigid_generalized_state\"") !=
                std::string::npos &&
            first.find("\"terminal_accepted_matter_rigid_reactions\"") !=
                std::string::npos &&
            first.find("\"terminal_accepted_support_histories\"") !=
                std::string::npos &&
            first.find("\"terminal_accepted_support_consequences\"") !=
                std::string::npos &&
            first.find("\"matter_rigid_generalized_candidate\"") ==
                std::string::npos &&
            first.find("\"support_histories\"") == std::string::npos &&
            first.find("\"support_consequences\"") == std::string::npos,
        "published terminal Matter state role is ambiguous");
    require(first.find(
            "\"rollback_identity_scope\":\"lifecycle-terminal-disposition-only-not-byte-restoration-proof\"") !=
            std::string::npos &&
            first.find("rollback_restored_accepted_matter_state") ==
                std::string::npos,
        "rollback identity overclaims byte restoration");
    require((source.coverageMask &
            NumiHumanOwnerMatterIntegrationUpdateV1) != 0u,
        "published Matter integration update is not covered");
    require(first.find("\"words\":[\"3f800000\",\"80000000\",\"40600000\"]")
            != std::string::npos,
        "FP32 words were not serialized bit exactly");
    std::string evidence;
    std::string repeatedEvidence;
    std::string payloadSHA256;
    std::string repeatedSHA256;
    require(serializeNumiHumanProductionOwnerSnapshotEvidenceV1(
        source, evidence, payloadSHA256, error),
        "valid evidence envelope did not serialize");
    require(serializeNumiHumanProductionOwnerSnapshotEvidenceV1(
        source, repeatedEvidence, repeatedSHA256, error),
        "repeat evidence envelope did not serialize");
    require(evidence == repeatedEvidence &&
            payloadSHA256 == repeatedSHA256 &&
            payloadSHA256.size() == 64u,
        "SHA-256 evidence envelope is not deterministic");
    require(evidence.find(
            "\"evidence_schema\":\"persistent-production-owner-snapshot-evidence.v1\"")
            != std::string::npos &&
            evidence.find("\"payload_sha256\":\"" + payloadSHA256 + "\"")
            != std::string::npos &&
            evidence.find("\"payload\":" + first) != std::string::npos,
        "canonical payload SHA-256 envelope is incomplete");

    const std::array<std::uint8_t, 8u> supportIdentity{
        1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};
    const std::vector<std::array<float, 4u>> seededHistory{{
        {0.25f, 0.0f, -0.5f, 1.0f}}};
    auto seeded = source;
    seeded.treatment = NumiHumanProductionOwnerTreatmentV1::seeded;
    seeded.treatmentHistoryFingerprint =
        numiHumanProductionOwnerTreatmentHistoryFingerprintV1(
            bytesOf(supportIdentity), bytesOf(seededHistory));
    require(seeded.baseStateFingerprint == source.baseStateFingerprint,
        "treatment history contaminated the base-state fingerprint");
    require(seeded.treatmentHistoryFingerprint !=
            source.treatmentHistoryFingerprint,
        "cold and seeded treatment histories alias");

    auto truncated = source;
    truncated.initialQ.bytes.pop_back();
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        truncated, second, error), "truncated array was serialized");

    auto explicitTruncation = source;
    explicitTruncation.truncationMask = NumiHumanOwnerInitialStateV1;
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        explicitTruncation, second, error),
        "explicit truncation was serialized");

    auto falseCoverage = source;
    falseCoverage.coverageMask |= NumiHumanOwnerMatterReactionV1;
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        falseCoverage, second, error), "false coverage was serialized");

    auto wrongShape = source;
    wrongShape.initialQ.expectedElementCount =
        wrongShape.qCoordinateCount - 1u;
    wrongShape.initialQ.bytes.resize(
        wrongShape.initialQ.expectedElementCount * sizeof(float));
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        wrongShape, second, error), "wrong logical shape was serialized");
    require(error.find("logical shape mismatch (initial_q:") !=
            std::string::npos,
        "logical shape failure did not identify its field");

    auto leakedMatterReactionSentinel = source;
    leakedMatterReactionSentinel
        .terminalAcceptedMatterRigidReactionCount = 0u;
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        leakedMatterReactionSentinel, second, error),
        "physical zero-width Matter sentinel was serialized as a reaction");
    require(error.find(
            "logical shape mismatch ("
            "terminal_accepted_matter_rigid_reactions:") !=
            std::string::npos,
        "Matter sentinel leakage did not identify its field");

    auto zeroLogicalMatterReactions = leakedMatterReactionSentinel;
    zeroLogicalMatterReactions.terminalAcceptedMatterRigidReactions
        .expectedElementCount = 0u;
    zeroLogicalMatterReactions.terminalAcceptedMatterRigidReactions
        .bytes.clear();
    require(serializeNumiHumanProductionOwnerSnapshotV1(
        zeroLogicalMatterReactions, second, error),
        "zero logical Matter-reaction range did not serialize");
    require(second.find(
            "\"terminal_accepted_matter_rigid_reactions\":{"
            "\"available\":true,\"expected_elements\":0,"
            "\"captured_elements\":0,\"element_bytes\":32,"
            "\"words\":[]}") != std::string::npos,
        "zero logical Matter-reaction range changed representation");

    auto wrongFactorShape = source;
    wrongFactorShape.effectiveTangentFactorStorage.expectedElementCount--;
    wrongFactorShape.effectiveTangentFactorStorage.bytes.resize(
        wrongFactorShape.effectiveTangentFactorStorage.expectedElementCount *
        sizeof(float));
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        wrongFactorShape, second, error),
        "partial effective-tangent factor storage was serialized");

    auto wrongElementWidth = source;
    wrongElementWidth.initialQ.elementBytes = 8u;
    wrongElementWidth.initialQ.bytes.resize(
        wrongElementWidth.initialQ.expectedElementCount * 8u);
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        wrongElementWidth, second, error),
        "type-confused nonempty array was serialized");

    auto wrongZeroCountElementWidth = source;
    wrongZeroCountElementWidth.tendonTransfers.elementBytes = sizeof(float);
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        wrongZeroCountElementWidth, second, error),
        "type-confused zero-count array was serialized");

    require(!source.candidateRoot.available,
        "unavailable complex-field fixture unexpectedly became available");
    auto wrongUnavailableElementWidth = source;
    wrongUnavailableElementWidth.candidateRoot.elementBytes = sizeof(float);
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        wrongUnavailableElementWidth, second, error),
        "type-confused unavailable array was serialized");

    const auto resizeArray = [](
        NumiHumanProductionOwnerArrayV1& array,
        const std::uint64_t count) {
        array.expectedElementCount = count;
        array.bytes.resize(static_cast<std::size_t>(count) *
            array.elementBytes);
    };
    auto truncatedSites = source;
    resizeArray(truncatedSites.muscleSites,
        truncatedSites.muscleSiteCount - 1u);
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        truncatedSites, second, error),
        "truncated muscle-site program was serialized");

    auto overlongSites = source;
    resizeArray(overlongSites.muscleSites,
        overlongSites.muscleSiteCount + 1u);
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        overlongSites, second, error),
        "overlong muscle-site program was serialized");

    auto truncatedWraps = source;
    resizeArray(truncatedWraps.muscleWraps,
        truncatedWraps.muscleWrapCount - 1u);
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        truncatedWraps, second, error),
        "truncated muscle-wrap program was serialized");

    auto overlongWraps = source;
    resizeArray(overlongWraps.muscleWraps,
        overlongWraps.muscleWrapCount + 1u);
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        overlongWraps, second, error),
        "overlong muscle-wrap program was serialized");

    auto truncatedRouteNodes = source;
    resizeArray(truncatedRouteNodes.muscleRouteNodes,
        truncatedRouteNodes.muscleRouteNodeCount - 1u);
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        truncatedRouteNodes, second, error),
        "truncated muscle-route program was serialized");

    auto overlongRouteNodes = source;
    resizeArray(overlongRouteNodes.muscleRouteNodes,
        overlongRouteNodes.muscleRouteNodeCount + 1u);
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        overlongRouteNodes, second, error),
        "overlong muscle-route program was serialized");

    auto wrongContactCount = source;
    ++wrongContactCount.contactSampleCount;
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        wrongContactCount, second, error),
        "contact-sample logical-count mismatch was serialized");

    auto wrongMatterStateCount = source;
    wrongMatterStateCount.terminalAcceptedMatterRigidGeneralizedStateCount +=
        1u;
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        wrongMatterStateCount, second, error),
        "terminal Matter generalized-state count mismatch was serialized");

    auto wrongMatterReactionCount = source;
    ++wrongMatterReactionCount.terminalAcceptedMatterRigidReactionCount;
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        wrongMatterReactionCount, second, error),
        "terminal Matter reaction count mismatch was serialized");

    auto unmetCoverage = source;
    unmetCoverage.requiredCoverageMask |= NumiHumanOwnerMatterReactionV1;
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        unmetCoverage, second, error),
        "missing required coverage was serialized");

    auto rejected = source;
    rejected.disposition = NumiHumanProductionOwnerDispositionV1::rejected;
    rejected.publicationIdentityAvailable = false;
    rejected.publicationEpoch = 0u;
    rejected.jointFenceFingerprint = 0u;
    rejected.rollbackIdentityAvailable = true;
    rejected.coverageMask = numiHumanProductionOwnerCoverageMaskV1(rejected);
    rejected.requiredCoverageMask = NumiHumanOwnerInitialStateV1 |
        NumiHumanOwnerRollbackIdentityV1;
    require((rejected.coverageMask &
            NumiHumanOwnerMatterIntegrationUpdateV1) == 0u,
        "rejected attempt claimed a Matter integration update");
    require(serializeNumiHumanProductionOwnerSnapshotV1(
        rejected, second, error), "complete rollback record was rejected");
    require(second.find(
            "\"terminal_matter_state_role\":\"prior-accepted-state-context-after-rejection\"") !=
            std::string::npos &&
            second.find("rollback_restored_accepted_matter_state") ==
                std::string::npos,
        "rejected terminal Matter context overclaims attempted integration");

    auto rejectedIntegrationClaim = rejected;
    rejectedIntegrationClaim.requiredCoverageMask |=
        NumiHumanOwnerMatterIntegrationUpdateV1;
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        rejectedIntegrationClaim, second, error),
        "rejected attempt admitted a Matter integration-update claim");

    auto rejectedPublicationClaim = rejected;
    rejectedPublicationClaim.publicationIdentityAvailable = true;
    rejectedPublicationClaim.publicationEpoch = 11u;
    rejectedPublicationClaim.jointFenceFingerprint = 0x808u;
    rejectedPublicationClaim.coverageMask =
        numiHumanProductionOwnerCoverageMaskV1(rejectedPublicationClaim);
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        rejectedPublicationClaim, second, error),
        "rejected record with publication identity serialized");

    auto rejectedPublicationResidue = rejected;
    rejectedPublicationResidue.publicationEpoch = 11u;
    rejectedPublicationResidue.jointFenceFingerprint = 0x808u;
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        rejectedPublicationResidue, second, error),
        "rejected record with publication residue serialized");

    auto publishedRollbackClaim = source;
    publishedRollbackClaim.rollbackIdentityAvailable = true;
    publishedRollbackClaim.coverageMask =
        numiHumanProductionOwnerCoverageMaskV1(publishedRollbackClaim);
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        publishedRollbackClaim, second, error),
        "published record with rollback identity serialized");

    auto missingRollbackIdentity = rejected;
    missingRollbackIdentity.rollbackIdentityAvailable = false;
    missingRollbackIdentity.coverageMask =
        numiHumanProductionOwnerCoverageMaskV1(missingRollbackIdentity);
    missingRollbackIdentity.requiredCoverageMask = NumiHumanOwnerInitialStateV1;
    require(!serializeNumiHumanProductionOwnerSnapshotV1(
        missingRollbackIdentity, second, error),
        "rejected record without lifecycle rollback identity serialized");

    return 0;
}
