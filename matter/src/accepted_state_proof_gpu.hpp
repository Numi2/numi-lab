#pragma once

#include <cstdint>

namespace numi::matter::detail {

inline constexpr std::uint32_t kAcceptedStateProofSchemaVersion = 2u;
inline constexpr std::uint32_t kAcceptedStateProofManifestVersion = 3u;
inline constexpr std::uint32_t kAcceptedStateProofChunkBytes = 1024u;
inline constexpr std::uint32_t kAcceptedStateProofTargetHuman = 0u;
inline constexpr std::uint32_t kAcceptedStateProofTargetMatter = 1u;
inline constexpr std::uint32_t kAcceptedStateProofSourceShared = 1u;

enum class AcceptedStateProofSource : std::uint32_t {
    humanQ = 0x1001u,
    humanV = 0x1002u,
    humanMujoco = 0x1003u,
    matterParticles = 0x2001u,
    matterParticleMaterialState = 0x2002u,
    matterFEMNodes = 0x2003u,
    matterFEMMaterialState = 0x2004u,
    matterFEMFields = 0x2005u,
    matterFEMTetrahedra = 0x2006u,
    matterFEMTopologyNodes = 0x2007u,
    matterCohesiveFaces = 0x2008u,
    matterPunctureChannels = 0x2009u,
    matterTopologyStates = 0x200au,
    matterLearnedWeights = 0x200bu,
    matterLearnedRevision = 0x200cu,
    matterContactHistories = 0x200du,
    matterDeformableContactHistories = 0x200eu,
    matterGeneralizedCandidate = 0x200fu,
    matterFrameReactions = 0x2010u,
    matterAdaptiveState = 0x2011u,
    matterSchedulers = 0x2012u,
    matterIdentification = 0x2013u,
    matterEnvironmentParameters = 0x2014u,
    matterFEMNodeIncidence = 0x2015u,
    matterFEMNodeRanges = 0x2016u,
    matterRigidStates = 0x2017u,
};

struct alignas(16) AcceptedStateProofBeginGPU {
    std::uint32_t environmentCount = 0u;
    std::uint32_t schemaVersion = 0u;
    std::uint32_t reserved0 = 0u;
    std::uint32_t reserved1 = 0u;
    std::uint64_t transactionPolicyFingerprint = 0u;
    std::uint64_t reserved2 = 0u;
};

struct alignas(16) AcceptedStateProofChunkGPU {
    std::uint32_t environmentCount = 0u;
    std::uint32_t source = 0u;
    std::uint32_t flags = 0u;
    std::uint32_t chunkBytes = 0u;
    std::uint32_t chunkCount = 0u;
    std::uint32_t scratchStride = 0u;
    std::uint32_t reserved0 = 0u;
    std::uint32_t reserved1 = 0u;
    std::uint64_t bytesPerEnvironment = 0u;
    std::uint64_t sharedBytes = 0u;
};

struct alignas(16) AcceptedStateProofReduceGPU {
    std::uint32_t environmentCount = 0u;
    std::uint32_t source = 0u;
    std::uint32_t inputCount = 0u;
    std::uint32_t outputCount = 0u;
    std::uint32_t scratchStride = 0u;
    std::uint32_t level = 0u;
    std::uint32_t reserved0 = 0u;
    std::uint32_t reserved1 = 0u;
};

struct alignas(16) AcceptedStateProofFoldGPU {
    std::uint32_t environmentCount = 0u;
    std::uint32_t source = 0u;
    std::uint32_t target = 0u;
    std::uint32_t flags = 0u;
    std::uint32_t scratchStride = 0u;
    std::uint32_t reserved0 = 0u;
    std::uint32_t reserved1 = 0u;
    std::uint32_t reserved2 = 0u;
    std::uint64_t bytesPerEnvironment = 0u;
    std::uint64_t sharedBytes = 0u;
};

struct alignas(16) AcceptedStateProofFinalizeGPU {
    std::uint32_t abiVersion = 0u;
    std::uint32_t structSize = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t environmentIdentifierBase = 0u;
    std::uint32_t matterStatusStride = 0u;
    std::uint32_t acceptedStateProofStride = 0u;
    std::uint32_t controlStep = 0u;
    std::uint32_t physicsSubstep = 0u;
    std::uint32_t transactionSlot = 0u;
    std::uint32_t environmentStatusStride = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t substepFingerprint = 0u;
    std::uint64_t acceptedTimestampMicroseconds = 0u;
    std::uint64_t physicsGeneration = 0u;
    std::uint64_t matterSourcePhysicsFingerprint = 0u;
    std::uint64_t matterDeviceProgramFingerprint = 0u;
    std::uint64_t stateProofProgramFingerprint = 0u;
    std::uint64_t adapterProgramFingerprint = 0u;
    std::uint64_t transactionPolicyFingerprint = 0u;
    std::uint64_t linearizationEpoch = 0u;
    std::uint64_t slotGeneration = 0u;
};

static_assert(sizeof(AcceptedStateProofBeginGPU) == 32u);
static_assert(sizeof(AcceptedStateProofChunkGPU) == 48u);
static_assert(sizeof(AcceptedStateProofReduceGPU) == 32u);
static_assert(sizeof(AcceptedStateProofFoldGPU) == 48u);
static_assert(sizeof(AcceptedStateProofFinalizeGPU) == 128u);

} // namespace numi::matter::detail
