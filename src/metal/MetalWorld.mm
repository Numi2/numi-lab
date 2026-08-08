#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/ParallelABASchedule.hpp"
#include "metalrobo/unified_quality_shared.h"

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace metalrobo {
namespace {

constexpr std::size_t kRawBufferCount = 241u;
constexpr NSUInteger kABAThreadsPerThreadgroup = 32u;
constexpr NSUInteger kOperatorThreadsPerThreadgroup = 32u;
constexpr NSUInteger kWorldThreadsPerThreadgroup = 64u;
static_assert(
    kWorldThreadsPerThreadgroup ==
    MR_WORLD_QUEUE_THREADS_PER_THREADGROUP
);
constexpr mr_u32 kSmallABAMaxBodies = 12u;
constexpr mr_u32 kSmallABAMaxDofs = 16u;
constexpr mr_u32 kSmallABAMaxQ = 17u;
constexpr float kQuaternionHostTolerance = 1.9e-5f;
constexpr std::uint64_t kShaderAddressableElements =
    static_cast<std::uint64_t>(
        std::numeric_limits<mr_u32>::max()
    ) + 1u;
constexpr std::uint64_t kFNVOffset =
    14695981039346656037ull;
constexpr std::uint64_t kFNVPrime = 1099511628211ull;
const char kMetalRoboWorldImageAnchor = 0;

enum BufferIndex : std::size_t {
    kWorld = 0u,
    kArticulations = 1u,
    kJoints = 2u,
    kDofs = 3u,
    kBodies = 4u,
    kABADispatch = 5u,
    kStateQA = 6u,
    kStateVA = 7u,
    kWorkingEffort = 8u,
    kBodyWrenchPlaceholder = 9u,
    kCandidateAcceleration = 10u,
    kCandidateV = 11u,
    kCandidateQ = 12u,
    kABAStatuses = 13u,
    kStateQB = 14u,
    kStateVB = 15u,
    kEffortTrajectory = 16u,
    kResetMasks = 17u,
    kResetQ = 18u,
    kResetV = 19u,
    kObservations = 20u,
    kAccelerationTrajectory = 21u,
    kPublicStatuses = 22u,
    kWorldDispatch = 23u,
    kEnvironmentStatuses = 24u,
    kCheckpointQ = 25u,
    kCheckpointV = 26u,
    kShapes = 27u,
    kMaterials = 28u,
    kSceneBodyIndices = 29u,
    kEligiblePairs = 30u,
    kContactDispatch = 31u,
    kOperatorKinematicsDispatch = 32u,
    kOperatorFactorDispatch = 33u,
    kInitialSceneBodies = 34u,
    kResetSceneBodies = 35u,
    kKinematicTargets = 36u,
    kSceneBodiesA = 37u,
    kSceneBodiesB = 38u,
    kCheckpointSceneBodies = 39u,
    kBodyPoses = 40u,
    kPointWorld = 41u,
    kFactorMatrix = 42u,
    kPointJacobians = 43u,
    kGeneralizedImpulse = 44u,
    kDeltaVelocity = 45u,
    kOperatorStatuses = 46u,
    kCurrentBodies = 47u,
    kCandidateBodies = 48u,
    kManifoldHeadersA = 49u,
    kManifoldPointsA = 50u,
    kManifoldCountsA = 51u,
    kManifoldHeadersB = 52u,
    kManifoldPointsB = 53u,
    kManifoldCountsB = 54u,
    kCandidateManifoldHeaders = 55u,
    kCandidateManifoldPoints = 56u,
    kCandidateManifoldCounts = 57u,
    kCheckpointManifoldHeaders = 58u,
    kCheckpointManifoldPoints = 59u,
    kCheckpointManifoldCounts = 60u,
    kCandidatePairs = 61u,
    kRawContacts = 62u,
    kRawPairIndices = 63u,
    kContacts = 64u,
    kContactMetadata = 65u,
    kIRBlocks = 66u,
    kIREndpoints = 67u,
    kIRRows = 68u,
    kIRCones = 69u,
    kPointQueries = 70u,
    kEvaluatedRows = 71u,
    kEvaluatedCones = 72u,
    kFactorCaches = 73u,
    kIslands = 74u,
    kResponseColumns = 75u,
    kContactStatuses = 76u,
    kPublicContactStatuses = 77u,
    kActiveIndirectDispatch = 78u,
    kProjectedColliders = 79u,
    kPairOverlapFlags = 80u,
    kWorkQueueHeaders = 81u,
    kPairWorkQueue = 82u,
    kPairRawCounts = 83u,
    kCompactionOffsets = 84u,
    kCompactionScratch = 85u,
    kCompactionFlags = 86u,
    kIslandWorkQueue = 87u,
    kContactTiles = 88u,
    kTileConstraintIndices = 89u,
    kWave32ImpulseDeltas = 90u,
    kWave32IslandStatuses = 91u,
    kConvexCaches = 92u,
    kCCDPairs = 93u,
    kGeometryHeaders = 94u,
    kGeometryVertices = 95u,
    kGeometryIndices = 96u,
    kConvexFaces = 97u,
    kConvexHalfEdges = 98u,
    kMeshBvhNodes = 99u,
    kMeshTriangles = 100u,
    kPairRawContactStaging = 101u,
    kWave32Preconditioners = 102u,
    kIslandWorkDense = 103u,
    kFutureBodyPoses = 104u,
    kFutureProjectedColliders = 105u,
    kCandidateConvexCaches = 106u,
    kCCDEventStatesA = 107u,
    kCCDEventStatesB = 108u,
    kCCDImpactClusters = 109u,
    kWaveWorkPackets = 110u,
    kPairManifoldHeaders = 111u,
    kPairManifoldPoints = 112u,
    kManifoldIRScatter = 113u,
    kEndpointRuntime = 114u,
    kQualityDispatch = 115u,
    kQualityBlocks = 116u,
    kQualityDynamics = 117u,
    kQualityJacobian = 118u,
    kQualityBias = 119u,
    kQualityFreeVelocity = 120u,
    kQualityWarmVelocity = 121u,
    kQualityWarmImpulses = 122u,
    kQualityOutputVelocity = 123u,
    kQualityOutputImpulses = 124u,
    kQualityDerivatives = 125u,
    kQualityHessian = 126u,
    kQualityStatuses = 127u,
    kQualityWorkQueue = 128u,
    kQualityWorkPackets = 129u,
    kDynamicNodes = 130u,
    kBodyDynamicNodes = 131u,
    kIslandNodeReferences = 132u,
    kIslandConstraintReferences = 133u,
    kRodFactorCaches = 134u,
    kOperatorVelocityArena = 135u,
    kInitialRodNodes = 136u,
    kInitialRodEdges = 137u,
    kResetRodNodes = 138u,
    kResetRodEdges = 139u,
    kRodNodesA = 140u,
    kRodEdgesA = 141u,
    kRodNodesB = 142u,
    kRodEdgesB = 143u,
    kCheckpointRodNodes = 144u,
    kCheckpointRodEdges = 145u,
    kRodDispatches = 146u,
    kRodRestLengths = 147u,
    kRodRestTwists = 148u,
    kRodRestCurvatures = 149u,
    kRodInverseMasses = 150u,
    kRodInverseRotationalInertias = 151u,
    kRodStretchStiffness = 152u,
    kRodBendStiffness = 153u,
    kRodTwistStiffness = 154u,
    kRodInputPositions = 155u,
    kRodInputVelocities = 156u,
    kRodInputTwists = 157u,
    kRodInputTwistRates = 158u,
    kRodOutputPositions = 159u,
    kRodOutputVelocities = 160u,
    kRodOutputTwists = 161u,
    kRodOutputTwistRates = 162u,
    kRodStatuses = 163u,
    kRodAttachments = 164u,
    kRodReactions = 165u,
    kFactorMatrixStaging = 166u,
    kPointJacobiansStaging = 167u,
    kRodColliders = 168u,
    kRodShapeSources = 169u,
    kRodToolPairs = 170u,
    kRodWitnessCounts = 171u,
    kRodWitnessesA = 172u,
    kRodWitnessesB = 173u,
    kCandidateRodWitnesses = 174u,
    kCheckpointRodWitnesses = 175u,
    kRodConstraintWitnessIndices = 176u,
    kRodCollisionDispatches = 177u,
    kRodContactScratch = 178u,
    kAuthoredIRBlocks = 179u,
    kAuthoredIREndpoints = 180u,
    kAuthoredIRRows = 181u,
    kAuthoredIRCones = 182u,
    kAuthoredIRWarmImpulses = 183u,
    kCCDEventRodNodesA = 184u,
    kCCDEventRodEdgesA = 185u,
    kCCDEventRodWitnessesA = 186u,
    kCCDEventRodNodesB = 187u,
    kCCDEventRodEdgesB = 188u,
    kCCDEventRodWitnessesB = 189u,
    kProjectedRodColliders = 190u,
    kFutureProjectedRodColliders = 191u,
    kActuatorProfiles = 192u,
    kTaskDispatch = 193u,
    kTaskActions = 194u,
    kTaskState = 195u,
    kTaskActionHistory = 196u,
    kTaskActorHistory = 197u,
    kTaskCleanHistory = 198u,
    kTaskCriticHistory = 199u,
    kTaskPreviousJointVelocity = 200u,
    kTaskEncoderBias = 201u,
    kTaskBodyParameters = 202u,
    kTaskControllerParameters = 203u,
    kTaskActorObservations = 204u,
    kTaskCriticObservations = 205u,
    kTaskTransitions = 206u,
    kTaskContactCompact = 207u,
    kTaskDefaultQ = 208u,
    kTaskProgramHeader = 209u,
    kTaskProgramArena = 210u,
    kPolicyProgramHeader = 211u,
    kPolicyProgramArena = 212u,
    kPolicyScratchA = 213u,
    kPolicyScratchB = 214u,
    kPolicyActorMean = 215u,
    kPolicyLatents = 216u,
    kPolicyLogProbabilities = 217u,
    kPolicyValues = 218u,
    kTaskEvidenceState = 219u,
    kTaskMotionFeatures = 220u,
    kInverseMassStatuses = 221u,
    kInverseMassDispatch = 222u,
    kParallelScheduleArticulations = 223u,
    kParallelScheduleLevels = 224u,
    kParallelScheduleParentReductions = 225u,
    kParallelScheduleLevelBodies = 226u,
    kParallelScheduleParentLocal = 227u,
    kParallelScheduleInboundJoint = 228u,
    kParallelScheduleChildOffsets = 229u,
    kParallelScheduleChildIndices = 230u,
    kTaskTeacherActions = 231u,
    kMulticopterRotors = 232u,
    kMulticopterModel = 233u,
    kMulticopterMixer = 234u,
    kMulticopterStateA = 235u,
    kMulticopterStateB = 236u,
    kMulticopterCandidateState = 237u,
    kMulticopterDispatch = 238u,
    kFlappingWingSpecs = 239u,
    kFlappingWingDispatch = 240u,
};

struct BufferRequirement {
    const char* label = "";
    std::size_t logicalElements = 0u;
    std::size_t logicalBytes = 0u;
    std::size_t allocationBytes = 0u;
};

struct RequiredBuffers {
    std::array<BufferRequirement, kRawBufferCount> entries{};
};

std::uint64_t multicopterFingerprint(
    const MetalWorldMulticopterProgram& program
) {
    if (!program.valid()) {
        return 0u;
    }
    std::uint64_t hash = kFNVOffset;
    const auto append = [&](const void* bytes, const std::size_t count) {
        const auto* values = static_cast<const std::byte*>(bytes);
        for (std::size_t index = 0u; index < count; ++index) {
            hash ^= std::to_integer<std::uint8_t>(values[index]);
            hash *= kFNVPrime;
        }
    };
    append(&program, sizeof(program));
    return hash == 0u ? 1u : hash;
}

std::uint64_t flappingWingFingerprint(
    const MetalWorldFlappingWingProgram& program
) {
    if (!program.valid()) {
        return 0u;
    }
    std::uint64_t hash = kFNVOffset;
    const auto append = [&](const void* bytes, const std::size_t count) {
        const auto* values = static_cast<const std::byte*>(bytes);
        for (std::size_t index = 0u; index < count; ++index) {
            hash ^= std::to_integer<std::uint8_t>(values[index]);
            hash *= kFNVPrime;
        }
    };
    append(&program, sizeof(program));
    return hash == 0u ? 1u : hash;
}

bool privateTransientBuffer(std::size_t index);
bool privatePersistentBuffer(std::size_t index);
bool privatePersistentInputBuffer(std::size_t index);
bool privateImmutableBuffer(std::size_t index);

} // namespace

namespace detail {

struct MetalWorldContextState {
    explicit MetalWorldContextState(MetalWorldConfig configured)
        : config(std::move(configured)) {}

    MetalWorldConfig config;
    mutable std::mutex mutex;
    bool initialized = false;
    bool inFlight = false;
    __strong id<MTLDevice> device = nil;
    __strong id<MTLCommandQueue> queue = nil;
    __strong id<MTLLibrary> library = nil;
    __strong id<MTLComputePipelineState> abaPipeline = nil;
    __strong id<MTLComputePipelineState> parameterizedABAPipeline = nil;
    __strong id<MTLComputePipelineState> smallABAPipeline = nil;
    __strong id<MTLComputePipelineState> multiABAPipeline = nil;
    __strong id<MTLComputePipelineState> preparePipeline = nil;
    __strong id<MTLComputePipelineState> driveRefreshPipeline = nil;
    __strong id<MTLComputePipelineState> commitPipeline = nil;
    __strong id<MTLComputePipelineState> capturePipeline = nil;
    __strong id<MTLComputePipelineState> operatorPipeline = nil;
    __strong id<MTLComputePipelineState>
        parameterizedOperatorPipeline = nil;
    __strong id<MTLComputePipelineState> taskObservePipeline = nil;
    __strong id<MTLComputePipelineState> taskThreatSelectPipeline = nil;
    __strong id<MTLComputePipelineState> taskJointCbfPipeline = nil;
    __strong id<MTLComputePipelineState> taskMotionPipeline = nil;
    __strong id<MTLComputePipelineState> taskApplyPipeline = nil;
    __strong id<MTLComputePipelineState> taskNativeActuatorPipeline = nil;
    __strong id<MTLComputePipelineState> taskEffortPipeline = nil;
    __strong id<MTLComputePipelineState> taskImpactContactPipeline = nil;
    __strong id<MTLComputePipelineState> taskCompletePipeline = nil;
    __strong id<MTLComputePipelineState> taskEvidencePipeline = nil;
    __strong id<MTLComputePipelineState> multicopterPipeline = nil;
    __strong id<MTLComputePipelineState> multicopterCommitPipeline = nil;
    __strong id<MTLComputePipelineState> flappingWingPipeline = nil;
    __strong id<MTLComputePipelineState> policyDensePipeline = nil;
    __strong id<MTLComputePipelineState> policySamplePipeline = nil;
    __strong id<MTLComputePipelineState> contactPreparePipeline = nil;
    __strong id<MTLComputePipelineState>
        observationStateSelectPipeline = nil;
    __strong id<MTLComputePipelineState> bodyProjectionPipeline = nil;
    __strong id<MTLComputePipelineState> scenePredictionPipeline = nil;
    __strong id<MTLComputePipelineState> colliderProjectionPipeline = nil;
    __strong id<MTLComputePipelineState> sweptProjectionPipeline = nil;
    __strong id<MTLComputePipelineState> ccdPipeline = nil;
    __strong id<MTLComputePipelineState> ccdEventInitializePipeline = nil;
    __strong id<MTLComputePipelineState> ccdEventPreparePipeline = nil;
    __strong id<MTLComputePipelineState> ccdEventSelectPipeline = nil;
    __strong id<MTLComputePipelineState> ccdEventFinalizePipeline = nil;
    __strong id<MTLComputePipelineState> eventArticulationPipeline = nil;
    __strong id<MTLComputePipelineState> eventScenePredictionPipeline = nil;
    __strong id<MTLComputePipelineState> eventBodyOverlayPipeline = nil;
    __strong id<MTLComputePipelineState> jointLimitPipeline = nil;
    __strong id<MTLComputePipelineState> eventColliderProjectionPipeline = nil;
    __strong id<MTLComputePipelineState> inactiveEventRestorePipeline = nil;
    __strong id<MTLComputePipelineState> eventSegmentPublishPipeline = nil;
    __strong id<MTLComputePipelineState> rodEventInitializePipeline = nil;
    __strong id<MTLComputePipelineState> inactiveRodEventRestorePipeline = nil;
    __strong id<MTLComputePipelineState> rodEventSegmentPublishPipeline = nil;
    __strong id<MTLComputePipelineState> rodSweptProjectionPipeline = nil;
    __strong id<MTLComputePipelineState> rodCCDPipeline = nil;
    __strong id<MTLComputePipelineState> rodCCDWitnessTagPipeline = nil;
    __strong id<MTLComputePipelineState> rodContactLatchPipeline = nil;
    __strong id<MTLComputePipelineState> pairFlagPipeline = nil;
    __strong id<MTLComputePipelineState> scanBlocksPipeline = nil;
    __strong id<MTLComputePipelineState> scanAddPipeline = nil;
    __strong id<MTLComputePipelineState> pairClassFlagPipeline = nil;
    __strong id<MTLComputePipelineState> pairQueueScatterPipeline = nil;
    __strong id<MTLComputePipelineState> pairNarrowphasePipeline = nil;
    __strong id<MTLComputePipelineState> convexNarrowphasePipeline = nil;
    __strong id<MTLComputePipelineState> hullNarrowphasePipeline = nil;
    __strong id<MTLComputePipelineState> meshNarrowphasePipeline = nil;
    __strong id<MTLComputePipelineState> collisionCompilePipeline = nil;
    __strong id<MTLComputePipelineState> manifoldFinalizePipeline = nil;
    __strong id<MTLComputePipelineState> manifoldScanPipeline = nil;
    __strong id<MTLComputePipelineState> manifoldRecordScatterPipeline = nil;
    __strong id<MTLComputePipelineState> manifoldIRScatterPipeline = nil;
    __strong id<MTLComputePipelineState> multiQueryInitializePipeline = nil;
    __strong id<MTLComputePipelineState> multiOperatorComposePipeline = nil;
    __strong id<MTLComputePipelineState> factorDispatchPipeline = nil;
    __strong id<MTLComputePipelineState> pointQueryTailPipeline = nil;
    __strong id<MTLComputePipelineState> streamedInversePipeline = nil;
    __strong id<MTLComputePipelineState> evaluateIRPipeline = nil;
    __strong id<MTLComputePipelineState> islandPipeline = nil;
    __strong id<MTLComputePipelineState> buildTilesPipeline = nil;
    __strong id<MTLComputePipelineState> islandQueueScatterPipeline = nil;
    __strong id<MTLComputePipelineState> solverCohortPipeline = nil;
    __strong id<MTLComputePipelineState> distributedIslandFlagPipeline = nil;
    __strong id<MTLComputePipelineState> distributedIslandScatterPipeline = nil;
    __strong id<MTLComputePipelineState> distributedTileFlagPipeline = nil;
    __strong id<MTLComputePipelineState> distributedTileScatterPipeline = nil;
    __strong id<MTLComputePipelineState> contactSolvePipeline = nil;
    __strong id<MTLComputePipelineState> wave32SolvePipeline = nil;
    __strong id<MTLComputePipelineState> wave32DistributedPreparePipeline = nil;
    __strong id<MTLComputePipelineState> wave32DistributedDeltaPipeline = nil;
    __strong id<MTLComputePipelineState> wave32DistributedReducePipeline = nil;
    __strong id<MTLComputePipelineState> wave32ReducePipeline = nil;
    __strong id<MTLComputePipelineState> contactIntegratePipeline = nil;
    __strong id<MTLComputePipelineState> contactLatchPipeline = nil;
    __strong id<MTLComputePipelineState> contactCommitPipeline = nil;
    __strong id<MTLComputePipelineState> convexCachePublishPipeline = nil;
    __strong id<MTLComputePipelineState> contactCapturePipeline = nil;
    __strong id<MTLComputePipelineState> qualityPreparePipeline = nil;
    __strong id<MTLComputePipelineState> qualityWarmStartPipeline = nil;
    __strong id<MTLComputePipelineState> qualityQueuePipeline = nil;
    __strong id<MTLComputePipelineState> qualitySolvePipeline = nil;
    __strong id<MTLComputePipelineState> qualityApplyPipeline = nil;
    __strong id<MTLComputePipelineState> qualityQueueStatusPipeline = nil;
    __strong id<MTLComputePipelineState> rodPreparePipeline = nil;
    __strong id<MTLComputePipelineState>
        rodContactPreparePipeline = nil;
    __strong id<MTLComputePipelineState> rodPackPipeline = nil;
    __strong id<MTLComputePipelineState> rodStepPipeline = nil;
    __strong id<MTLComputePipelineState> rodFactorPipeline = nil;
    __strong id<MTLComputePipelineState> rodUnpackPipeline = nil;
    __strong id<MTLComputePipelineState> rodLatchPipeline = nil;
    __strong id<MTLComputePipelineState>
        rodToolNarrowphasePipeline = nil;
    __strong id<MTLComputePipelineState>
        rodContactScanPipeline = nil;
    __strong id<MTLComputePipelineState>
        rodContactScatterPipeline = nil;
    __strong id<MTLComputePipelineState>
        rodContactSolvePipeline = nil;
    __strong id<MTLComputePipelineState> rodCommitPipeline = nil;
    __strong id<MTLComputePipelineState>
        rodContactCommitPipeline = nil;
    __strong id<MTLComputePipelineState>
        authoredIRSeedPipeline = nil;
    __strong id<MTLComputePipelineState>
        generalizedConstraintSolvePipeline = nil;
    __strong id<MTLHeap> immutableHeap = nil;
    __strong id<MTLHeap> persistentHeap = nil;
    __strong id<MTLHeap> transientHeap = nil;
    __strong id<MTLBuffer> buffers[kRawBufferCount] = {};
    __strong id<MTLBuffer> uploadBuffers[kRawBufferCount] = {};
    __strong id<MTLBuffer> readbackBuffers[kRawBufferCount] = {};
    std::array<std::size_t, kRawBufferCount> capacities{};
    std::array<std::size_t, kRawBufferCount> uploadCapacities{};
    std::array<std::size_t, kRawBufferCount> readbackCapacities{};
    std::size_t readbackBytes = 0u;
    // Immutable host metadata used only while encoding the command graph.
    // Dynamic work counts and physical state remain entirely device-resident.
    std::vector<MRArticulationGPU> boundArticulations;
    std::vector<MRArticulatedOperatorDispatchGPU>
        boundFactorDispatches;
    MRMetalWorldContactDispatchGPU boundContactDispatch{};
    bool useTaskBodyParameters = false;
    std::uint64_t boundModelFingerprint = 0u;
    std::uint64_t boundTaskFingerprint = 0u;
    std::uint64_t boundPolicyFingerprint = 0u;
    std::uint64_t boundMulticopterFingerprint = 0u;
    std::uint64_t boundFlappingWingFingerprint = 0u;
    std::uint64_t stateArenaGeneration = 0u;
    std::weak_ptr<MetalWorldResidentStateData> residentOwner;
    MetalWorldContextStats stats{};
};

struct MetalWorldContextPool {
    mutable std::mutex mutex;
    std::vector<std::shared_ptr<MetalWorldContextState>> slots;
    std::size_t nextSlot = 0u;
};

struct MetalWorldSlotReservation {
    explicit MetalWorldSlotReservation(
        std::shared_ptr<MetalWorldContextState> selected
    )
        : state(std::move(selected)) {}

    ~MetalWorldSlotReservation() {
        if (!armed || state == nullptr) {
            return;
        }
        try {
            const std::lock_guard lock(state->mutex);
            state->inFlight = false;
            state->stats.hasInFlightSubmission = false;
        } catch (...) {
        }
    }

    void handoff() noexcept {
        armed = false;
    }

    std::shared_ptr<MetalWorldContextState> state;
    bool armed = true;
};

struct MetalWorldResidentStateData {
    mutable std::mutex mutex;
    std::weak_ptr<MetalWorldContextPool> ownerPool;
    std::shared_ptr<MetalWorldContextState> context;
    std::uint64_t worldFingerprint = 0u;
    std::uint64_t taskFingerprint = 0u;
    std::uint64_t taskSeed = 0u;
    std::uint64_t stateArenaGeneration = 0u;
    std::size_t environmentCount = 0u;
    std::size_t qBuffer = kStateQA;
    std::size_t vBuffer = kStateVA;
    std::size_t sceneBuffer = kSceneBodiesA;
    std::size_t manifoldHeaderBuffer = kManifoldHeadersA;
    std::size_t manifoldPointBuffer = kManifoldPointsA;
    std::size_t manifoldCountBuffer = kManifoldCountsA;
    std::size_t rodNodeBuffer = kRodNodesA;
    std::size_t rodEdgeBuffer = kRodEdgesA;
    std::size_t rodWitnessBuffer = kRodWitnessesA;
    bool supportsReset = false;
    bool initialized = false;
    bool pending = false;
};

struct MetalWorldResidentReservation {
    explicit MetalWorldResidentReservation(
        std::shared_ptr<MetalWorldResidentStateData> selected
    )
        : state(std::move(selected)) {}

    ~MetalWorldResidentReservation() {
        if (!armed || state == nullptr) {
            return;
        }
        try {
            const std::lock_guard lock(state->mutex);
            state->pending = false;
        } catch (...) {
        }
    }

    void handoff() noexcept {
        armed = false;
    }

    std::shared_ptr<MetalWorldResidentStateData> state;
    bool armed = true;
};

struct MetalWorldSubmissionState {
    void finishResident(const bool commandCompleted) noexcept {
        if (resident == nullptr) {
            return;
        }
        try {
            const std::lock_guard lock(resident->mutex);
            resident->pending = false;
            if (!commandCompleted || context == nullptr ||
                resident->context != context ||
                resident->stateArenaGeneration !=
                    context->stateArenaGeneration) {
                resident->initialized = false;
                return;
            }
            resident->qBuffer = finalQBuffer;
            resident->vBuffer = finalVBuffer;
            resident->sceneBuffer = finalSceneBuffer;
            resident->manifoldHeaderBuffer =
                finalManifoldHeaderBuffer;
            resident->manifoldPointBuffer =
                finalManifoldPointBuffer;
            resident->manifoldCountBuffer =
                finalManifoldCountBuffer;
            resident->rodNodeBuffer = finalRodNodeBuffer;
            resident->rodEdgeBuffer = finalRodEdgeBuffer;
            resident->rodWitnessBuffer =
                finalRodWitnessBuffer;
            resident->initialized = true;
        } catch (...) {
        }
    }

    ~MetalWorldSubmissionState() {
        if (!ownsInFlight || context == nullptr) {
            return;
        }
        @autoreleasepool {
            [commandBuffer waitUntilCompleted];
        }
        finishResident(
            commandBuffer.status ==
                MTLCommandBufferStatusCompleted
        );
        try {
            const std::lock_guard lock(context->mutex);
            context->inFlight = false;
            context->stats.hasInFlightSubmission = false;
            ++context->stats.completedSubmissionCount;
        } catch (...) {
            // Destruction cannot throw. Completion makes the shared arena
            // safe even if platform locking itself were to fail.
        }
        ownsInFlight = false;
    }

    std::shared_ptr<MetalWorldContextState> context;
    __strong id<MTLCommandBuffer> commandBuffer = nil;
    MetalWorldDiagnostics diagnostics{};
    std::chrono::steady_clock::time_point start{};
    MRArticulationGPU articulation{};
    std::size_t finalQBuffer = kStateQA;
    std::size_t finalVBuffer = kStateVA;
    std::size_t finalSceneBuffer = kSceneBodiesA;
    std::size_t finalManifoldHeaderBuffer = kManifoldHeadersA;
    std::size_t finalManifoldPointBuffer = kManifoldPointsA;
    std::size_t finalManifoldCountBuffer = kManifoldCountsA;
    std::size_t finalRodNodeBuffer = kRodNodesA;
    std::size_t finalRodEdgeBuffer = kRodEdgesA;
    std::size_t finalRodWitnessBuffer = kRodWitnessesA;
    std::shared_ptr<MetalWorldResidentStateData> resident;
    std::uint64_t policyRevision = 0u;
    bool hasRods = false;
    bool contactMode = false;
    bool nativeTask = false;
    bool captureContactEvidence = false;
    bool publishFinalState = true;
    bool publishStateTrajectory = true;
    bool ownsInFlight = false;
};

} // namespace detail

namespace {

std::string nsString(NSString* value) {
    if (value == nil || value.UTF8String == nullptr) {
        return {};
    }
    return std::string{value.UTF8String};
}

std::string describeError(NSError* error) {
    if (error == nil) {
        return "unknown Metal error";
    }
    std::string result = nsString(error.localizedDescription);
    if (result.empty()) {
        result = nsString(error.description);
    }
    NSArray<id<MTLCommandBufferEncoderInfo>>* encoders =
        error.userInfo[MTLCommandBufferEncoderInfoErrorKey];
    for (id<MTLCommandBufferEncoderInfo> encoder in encoders) {
        if (encoder.errorState == MTLCommandEncoderErrorStateFaulted ||
            encoder.errorState == MTLCommandEncoderErrorStateAffected) {
            result += " encoder=\"" + nsString(encoder.label) + "\"";
            result += encoder.errorState == MTLCommandEncoderErrorStateFaulted
                ? " state=faulted"
                : " state=affected";
        }
    }
    return result.empty() ? "unknown Metal error" : result;
}

std::string thermalStateName(
    const NSProcessInfoThermalState state
) {
    switch (state) {
    case NSProcessInfoThermalStateNominal:
        return "nominal";
    case NSProcessInfoThermalStateFair:
        return "fair";
    case NSProcessInfoThermalStateSerious:
        return "serious";
    case NSProcessInfoThermalStateCritical:
        return "critical";
    }
    return "unknown";
}

bool regularFile(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) &&
        !error;
}

std::string defaultMetallibPath() {
    Dl_info image{};
    if (dladdr(&kMetalRoboWorldImageAnchor, &image) != 0 &&
        image.dli_fname != nullptr) {
        const std::filesystem::path libraryDirectory =
            std::filesystem::path(image.dli_fname).parent_path();
        const std::array candidates{
            libraryDirectory / "metalrobo/MetalRobo.metallib",
            libraryDirectory.parent_path() /
                "shaders/MetalRobo.metallib",
        };
        for (const std::filesystem::path& candidate :
             candidates) {
            if (regularFile(candidate)) {
                return candidate.string();
            }
        }
    }

    const std::filesystem::path configured{
        METALROBO_DEFAULT_METALLIB
    };
    return regularFile(configured)
        ? configured.string()
        : std::string{};
}

MetalWorldDiagnostics reject(
    MetalWorldDiagnostics diagnostics,
    const MetalWorldHostStatus status,
    std::string message
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    return diagnostics;
}

MetalWorldCompileDiagnostics rejectCompile(
    MetalWorldCompileDiagnostics diagnostics,
    const MetalWorldHostStatus status,
    std::string message
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    return diagnostics;
}

bool checkedMultiply(
    const std::size_t left,
    const std::size_t right,
    std::size_t& result
) {
    if (left != 0u &&
        right >
            std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool checkedAdd(
    const std::size_t left,
    const std::size_t right,
    std::size_t& result
) {
    if (right >
        std::numeric_limits<std::size_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

std::uint32_t contactConstraintCapacity(
    const std::uint32_t constraintCapacity,
    const std::size_t authoredConstraintCount
) {
    const std::uint64_t authored = std::min<std::uint64_t>(
        authoredConstraintCount,
        constraintCapacity
    );
    return static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(constraintCapacity) -
        authored
    );
}

std::uint32_t requiredSolverTileCapacity(
    const std::uint32_t constraintCapacity,
    const std::size_t authoredConstraintCount,
    const std::uint32_t islandCapacity,
    const std::size_t dynamicNodeCount
) {
    const std::uint32_t contacts = contactConstraintCapacity(
        constraintCapacity,
        authoredConstraintCount
    );
    if (contacts == 0u) {
        // The contact dispatch ABI intentionally keeps a valid empty tile
        // arena so authored-only worlds use the same persistent graph.
        return 1u;
    }
    const std::uint32_t nonemptyIslands =
        std::min<std::uint32_t>(
            contacts,
            std::min<std::uint32_t>(
                islandCapacity,
                static_cast<std::uint32_t>(
                    std::min<std::size_t>(
                        dynamicNodeCount,
                        std::numeric_limits<std::uint32_t>::max()
                    )
                )
            )
        );
    // For C constraints distributed across K nonempty islands, the maximum
    // sum of ceil(c_i / 32) is K + floor((C - K) / 32).
    return std::max<std::uint32_t>(
        1u,
        nonemptyIslands +
            (contacts - nonemptyIslands) /
                MR_WAVE32_CONTACTS_PER_TILE
    );
}

std::uint32_t requiredSpillRowCapacity(
    const std::uint32_t constraintCapacity,
    const std::size_t authoredConstraintCount
) {
    const std::uint32_t contacts = contactConstraintCapacity(
        constraintCapacity,
        authoredConstraintCount
    );
    return contacts > MR_WAVE32_CONTACTS_PER_TILE
        ? 3u * (contacts - MR_WAVE32_CONTACTS_PER_TILE)
        : 0u;
}

template <typename T>
bool makeRequirement(
    const char* label,
    const std::size_t logicalElements,
    BufferRequirement& result
) {
    std::size_t logicalBytes = 0u;
    if (!checkedMultiply(
            logicalElements,
            sizeof(T),
            logicalBytes
        )) {
        return false;
    }
    result.label = label;
    result.logicalElements = logicalElements;
    result.logicalBytes = logicalBytes;
    result.allocationBytes =
        logicalBytes == 0u ? sizeof(T) : logicalBytes;
    return result.allocationBytes <=
        std::numeric_limits<NSUInteger>::max();
}

bool finite(const mr_float4 value) {
    return std::isfinite(value.x) &&
        std::isfinite(value.y) &&
        std::isfinite(value.z) &&
        std::isfinite(value.w);
}

bool finiteFloats(const std::span<const float> values) {
    return std::all_of(
        values.begin(),
        values.end(),
        [](const float value) {
            return std::isfinite(value);
        }
    );
}

using RodVec3 = std::array<double, 3u>;

RodVec3 rodAdd(const RodVec3& a, const RodVec3& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

RodVec3 rodSubtract(const RodVec3& a, const RodVec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

RodVec3 rodScale(const RodVec3& value, const double scale) {
    return {
        value[0] * scale,
        value[1] * scale,
        value[2] * scale,
    };
}

double rodDot(const RodVec3& a, const RodVec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

RodVec3 rodCross(const RodVec3& a, const RodVec3& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

bool rodNormalize(const RodVec3& input, RodVec3& output) {
    const double squared = rodDot(input, input);
    if (!(squared > 1.0e-28) || !std::isfinite(squared)) {
        return false;
    }
    output = rodScale(input, 1.0 / std::sqrt(squared));
    return std::all_of(
        output.begin(),
        output.end(),
        [](const double value) {
            return std::isfinite(value);
        }
    );
}

RodVec3 rodRotate(
    const RodVec3& vector,
    const RodVec3& axis,
    const double angle
) {
    return rodAdd(
        rodAdd(
            rodScale(vector, std::cos(angle)),
            rodScale(rodCross(axis, vector), std::sin(angle))
        ),
        rodScale(
            axis,
            rodDot(axis, vector) * (1.0 - std::cos(angle))
        )
    );
}

RodVec3 rodLeastAligned(const RodVec3& tangent) {
    const std::array<RodVec3, 3u> axes{{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    }};
    std::size_t selected = 0u;
    for (std::size_t axis = 1u;
         axis < axes.size();
         ++axis) {
        if (std::abs(rodDot(tangent, axes[axis])) <
            std::abs(rodDot(tangent, axes[selected]))) {
            selected = axis;
        }
    }
    RodVec3 result = rodSubtract(
        axes[selected],
        rodScale(
            tangent,
            rodDot(axes[selected], tangent)
        )
    );
    (void)rodNormalize(result, result);
    return result;
}

bool rodTransport(
    const RodVec3& director,
    const RodVec3& from,
    const RodVec3& to,
    RodVec3& output
) {
    const RodVec3 axis = rodCross(from, to);
    const double sine = std::sqrt(rodDot(axis, axis));
    const double cosine =
        std::clamp(rodDot(from, to), -1.0, 1.0);
    if (sine <= 1.0e-14) {
        if (cosine < 0.0) {
            return false;
        }
        output = director;
        return true;
    }
    output = rodRotate(
        director,
        rodScale(axis, 1.0 / sine),
        std::atan2(sine, cosine)
    );
    output = rodSubtract(
        output,
        rodScale(to, rodDot(output, to))
    );
    return rodNormalize(output, output);
}

bool rodRestCurvature(
    const DiscreteElasticRodModel& model,
    const std::size_t vertex,
    mr_float4& output
) {
    RodVec3 left;
    RodVec3 right;
    if (!rodNormalize(
            rodSubtract(
                model.restPositions[vertex + 1u],
                model.restPositions[vertex]
            ),
            left
        ) ||
        !rodNormalize(
            rodSubtract(
                model.restPositions[vertex + 2u],
                model.restPositions[vertex + 1u]
            ),
            right
        )) {
        return false;
    }
    const RodVec3 referenceLeft = rodLeastAligned(left);
    RodVec3 referenceRight;
    if (!rodTransport(
            referenceLeft,
            left,
            right,
            referenceRight
        )) {
        return false;
    }
    const RodVec3 directorLeft = rodRotate(
        referenceLeft,
        left,
        model.restTwists[vertex]
    );
    const RodVec3 directorRight = rodRotate(
        referenceRight,
        right,
        model.restTwists[vertex + 1u]
    );
    const RodVec3 secondLeft =
        rodCross(left, directorLeft);
    const RodVec3 secondRight =
        rodCross(right, directorRight);
    const double denominator = 1.0 + rodDot(left, right);
    if (!(denominator > 1.0e-8) ||
        !std::isfinite(denominator)) {
        return false;
    }
    const RodVec3 binormal =
        rodScale(rodCross(left, right), 2.0 / denominator);
    output = {
        static_cast<float>(
            0.5 * rodDot(
                binormal,
                rodAdd(secondLeft, secondRight)
            )
        ),
        static_cast<float>(
            -0.5 * rodDot(
                binormal,
                rodAdd(directorLeft, directorRight)
            )
        ),
        0.0f,
        0.0f,
    };
    return finite(output);
}

bool validRodStates(
    const CompiledWorld& world,
    const std::size_t environmentCount,
    const std::span<const MRRodNodeStateGPU> nodes,
    const std::span<const MRRodEdgeStateGPU> edges,
    const bool allowCompiledDefaults
) {
    if (world.rodCount() == 0u) {
        return nodes.empty() && edges.empty();
    }
    if (nodes.empty() && edges.empty() &&
        allowCompiledDefaults) {
        return true;
    }
    if (nodes.size() !=
            environmentCount * world.rodNodeCount() ||
        edges.size() !=
            environmentCount * world.rodEdgeCount()) {
        return false;
    }
    return
        std::all_of(
            nodes.begin(),
            nodes.end(),
            [](const MRRodNodeStateGPU& node) {
                return
                    finite(node.position) &&
                    finite(node.velocity) &&
                    node.position.w == 1.0f &&
                    node.velocity.w == 0.0f;
            }
        ) &&
        std::all_of(
            edges.begin(),
            edges.end(),
            [](const MRRodEdgeStateGPU& edge) {
                return finite(edge.twistAndRate);
            }
        );
}

bool validSceneStates(
    const CompiledWorld& world,
    const std::size_t environmentCount,
    const std::span<const MRBodyStateGPU> states
) {
    if (states.empty() && world.sceneBodyCount() == 0u) {
        return true;
    }
    if (states.size() !=
        environmentCount * world.sceneBodyCount()) {
        return false;
    }
    for (std::size_t environment = 0u;
         environment < environmentCount;
         ++environment) {
        std::size_t localScene = 0u;
        for (std::uint32_t globalBody = 0u;
             globalBody < world.model().bodies.size();
             ++globalBody) {
            if (world.model().bodies[globalBody].articulationIndex !=
                MR_INVALID_INDEX) {
                continue;
            }
            const MRBodyPropertiesGPU& properties =
                world.model().bodies[globalBody];
            const MRBodyStateGPU& state =
                states[
                    environment * world.sceneBodyCount() +
                    localScene
                ];
            const double quaternionNormSquared =
                static_cast<double>(state.orientation.x) *
                    state.orientation.x +
                static_cast<double>(state.orientation.y) *
                    state.orientation.y +
                static_cast<double>(state.orientation.z) *
                    state.orientation.z +
                static_cast<double>(state.orientation.w) *
                    state.orientation.w;
            if (!finite(state.position) ||
                !finite(state.orientation) ||
                !finite(state.linearVelocityAndInverseMass) ||
                !finite(state.angularVelocity) ||
                !finite(state.inverseInertiaWorldRow0) ||
                !finite(state.inverseInertiaWorldRow1) ||
                !finite(state.inverseInertiaWorldRow2) ||
                !(quaternionNormSquared > 1.0e-12) ||
                std::abs(std::sqrt(quaternionNormSquared) - 1.0) >
                    kQuaternionHostTolerance ||
                state.flagsAndIndices[0] != properties.motionType ||
                state.flagsAndIndices[1] != MR_INVALID_INDEX ||
                (state.flagsAndIndices[2] != globalBody &&
                 state.flagsAndIndices[2] != MR_INVALID_INDEX)) {
                return false;
            }
            ++localScene;
        }
    }
    return true;
}

bool supportedTopology(
    const EngineModel& model,
    const MRArticulationGPU& articulation,
    std::string& reason
) {
    if (articulation.bodyCount == 0u ||
        articulation.bodyCount >
            MR_ARTICULATED_ABA_MAX_BODIES ||
        articulation.nv == 0u ||
        articulation.nv > MR_ARTICULATED_ABA_MAX_DOFS ||
        articulation.nq == 0u ||
        articulation.nq > MR_ARTICULATED_ABA_MAX_Q) {
        reason =
            "articulation exceeds the Metal ABA body, DoF, or q bucket";
        return false;
    }
    if (articulation.rootType != MR_ROOT_FIXED &&
        articulation.rootType != MR_ROOT_FLOATING) {
        reason = "Metal world root type is unsupported";
        return false;
    }

    const std::size_t jointEnd =
        static_cast<std::size_t>(articulation.firstJoint) +
        articulation.jointCount;
    for (std::size_t jointIndex = articulation.firstJoint;
         jointIndex < jointEnd;
         ++jointIndex) {
        const MRJointDescriptorGPU& joint =
            model.joints[jointIndex];
        if (joint.flags != 0u) {
            reason =
                "Metal world joints require zero reserved flags";
            return false;
        }
        if (joint.jointType != MR_JOINT_REVOLUTE &&
            joint.jointType != MR_JOINT_CONTINUOUS &&
            joint.jointType != MR_JOINT_PRISMATIC &&
            joint.jointType != MR_JOINT_FIXED) {
            reason =
                "free-motion Metal world supports revolute, "
                "continuous, prismatic, and fixed joints";
            return false;
        }
    }

    const std::size_t bodyEnd =
        static_cast<std::size_t>(articulation.firstBody) +
        articulation.bodyCount;
    for (std::size_t bodyIndex = articulation.firstBody;
         bodyIndex < bodyEnd;
         ++bodyIndex) {
        if (model.bodies[bodyIndex].motionType !=
            MR_MOTION_DYNAMIC) {
            reason =
                "every body in the selected articulation must be dynamic";
            return false;
        }
    }
    return true;
}

bool validQ(
    const EngineModel& model,
    const std::size_t environmentCount,
    const std::span<const float> q
) {
    if (!finiteFloats(q)) {
        return false;
    }
    for (std::size_t environment = 0u;
         environment < environmentCount;
         ++environment) {
        for (const MRArticulationGPU& articulation :
             model.articulations) {
            if (articulation.rootType != MR_ROOT_FLOATING) {
                continue;
            }
            const std::size_t base =
                environment * model.world.nq +
                articulation.qOffset + 3u;
            const double x = q[base + 0u];
            const double y = q[base + 1u];
            const double z = q[base + 2u];
            const double w = q[base + 3u];
            const double normSquared =
                x * x + y * y + z * z + w * w;
            if (!(normSquared > 1.0e-12) ||
                !std::isfinite(normSquared)) {
                return false;
            }
            const double norm = std::sqrt(normSquared);
            if (!std::isfinite(norm) ||
                std::abs(norm - 1.0) >
                    kQuaternionHostTolerance) {
                return false;
            }
        }
    }
    return true;
}

void hashBytes(
    std::uint64_t& hash,
    const void* data,
    const std::size_t size
) {
    const auto* bytes =
        static_cast<const unsigned char*>(data);
    for (std::size_t index = 0u; index < size; ++index) {
        hash ^= bytes[index];
        hash *= kFNVPrime;
    }
}

template <typename T>
void hashValue(std::uint64_t& hash, const T& value) {
    hashBytes(hash, &value, sizeof(value));
}

template <typename T>
void hashVector(
    std::uint64_t& hash,
    const std::vector<T>& values
) {
    hashValue(hash, values.size());
    if (!values.empty()) {
        hashBytes(
            hash,
            values.data(),
            values.size() * sizeof(T)
        );
    }
}

void hashStrings(
    std::uint64_t& hash,
    const std::vector<std::string>& values
) {
    hashValue(hash, values.size());
    for (const std::string& value : values) {
        hashValue(hash, value.size());
        hashBytes(hash, value.data(), value.size());
    }
}

std::uint64_t fingerprint(const EngineModel& model) {
    std::uint64_t hash = kFNVOffset;
    hashValue(hash, MR_ENGINE_ABI_VERSION);
    hashValue(hash, model.world);
    hashVector(hash, model.articulations);
    hashVector(hash, model.joints);
    hashVector(hash, model.dofs);
    hashVector(hash, model.actuatorProfiles);
    hashVector(hash, model.bodies);
    hashVector(hash, model.shapes);
    hashVector(hash, model.materials);
    hashVector(hash, model.geometryHeaders);
    hashVector(hash, model.geometryVertices);
    hashVector(hash, model.geometryIndices);
    hashVector(hash, model.convexFaces);
    hashVector(hash, model.convexHalfEdges);
    hashVector(hash, model.meshBvhNodes);
    hashVector(hash, model.meshTriangles);
    hashVector(hash, model.collisionExclusions);
    hashValue(hash, model.constraintProgram.abiVersion);
    hashVector(hash, model.constraintProgram.blocks);
    hashVector(hash, model.constraintProgram.endpoints);
    hashVector(hash, model.constraintProgram.rows);
    hashVector(hash, model.constraintProgram.cones);
    hashVector(hash, model.constraintProgram.warmImpulses);
    hashVector(hash, model.defaultQ);
    hashVector(hash, model.defaultV);
    hashStrings(hash, model.bodyNames);
    hashStrings(hash, model.jointNames);
    hashStrings(hash, model.dofNames);
    hashStrings(hash, model.shapeNames);
    hashValue(hash, model.name.size());
    hashBytes(hash, model.name.data(), model.name.size());
    return hash == 0u ? 1u : hash;
}

bool taskHasActuatorKind(
    const CompiledTaskProgram& task,
    const std::uint32_t kind
) {
    return std::ranges::any_of(
        task.actionBindings(),
        [kind](const MRTaskActionBindingGPU& binding) {
            return binding.actuator.x == kind;
        }
    );
}

std::vector<MRActuatorProfileGPU> executionActuatorProfiles(
    const EngineModel& model
) {
    std::vector<MRActuatorProfileGPU> profiles =
        model.actuatorProfiles.empty()
        ? std::vector<MRActuatorProfileGPU>(model.dofs.size())
        : model.actuatorProfiles;
    for (std::size_t index = 0u;
         index < profiles.size();
         ++index) {
        const MRDofPropertiesGPU& dof = model.dofs[index];
        MRActuatorProfileGPU& profile = profiles[index];
        if ((profile.identity.y &
             MR_ACTUATOR_PROFILE_ACTIVE) != 0u) {
            continue;
        }
        profile.motorAndSpeed = {
            0.0f,
            0.0f,
            std::numeric_limits<float>::max(),
            1.0f,
        };
        profile.transmissionAndEnvelope = {};
        profile.transmissionAndEnvelope.z =
            (dof.flags & MR_DOF_FLAG_EFFORT_LIMIT) != 0u &&
                dof.limits.w > 0.0f
            ? dof.limits.w
            : std::numeric_limits<float>::max();
        profile.identity = {
            static_cast<mr_u32>(index),
            0u,
            0u,
            0u,
        };
    }
    return profiles;
}

std::uint32_t compiledPairClass(
    const std::uint32_t typeA,
    const std::uint32_t typeB
) {
    if (typeA == MR_SHAPE_SPHERE &&
        typeB == MR_SHAPE_SPHERE) {
        return MR_COLLISION_PAIR_SPHERE_SPHERE;
    }
    if ((typeA == MR_SHAPE_SPHERE &&
         typeB == MR_SHAPE_PLANE) ||
        (typeB == MR_SHAPE_SPHERE &&
         typeA == MR_SHAPE_PLANE)) {
        return MR_COLLISION_PAIR_SPHERE_PLANE;
    }
    if ((typeA == MR_SHAPE_CAPSULE &&
         typeB == MR_SHAPE_PLANE) ||
        (typeB == MR_SHAPE_CAPSULE &&
         typeA == MR_SHAPE_PLANE)) {
        return MR_COLLISION_PAIR_CAPSULE_PLANE;
    }
    if ((typeA == MR_SHAPE_BOX &&
         typeB == MR_SHAPE_PLANE) ||
        (typeB == MR_SHAPE_BOX &&
         typeA == MR_SHAPE_PLANE)) {
        return MR_COLLISION_PAIR_BOX_PLANE;
    }
    if ((typeA == MR_SHAPE_CYLINDER &&
         typeB == MR_SHAPE_PLANE) ||
        (typeB == MR_SHAPE_CYLINDER &&
         typeA == MR_SHAPE_PLANE)) {
        return MR_COLLISION_PAIR_CYLINDER_PLANE;
    }
    if ((typeA == MR_SHAPE_SPHERE &&
         typeB == MR_SHAPE_CAPSULE) ||
        (typeB == MR_SHAPE_SPHERE &&
         typeA == MR_SHAPE_CAPSULE)) {
        return MR_COLLISION_PAIR_SPHERE_CAPSULE;
    }
    if (typeA == MR_SHAPE_CAPSULE &&
        typeB == MR_SHAPE_CAPSULE) {
        return MR_COLLISION_PAIR_CAPSULE_CAPSULE;
    }
    if ((typeA == MR_SHAPE_SPHERE &&
         typeB == MR_SHAPE_BOX) ||
        (typeB == MR_SHAPE_SPHERE &&
         typeA == MR_SHAPE_BOX)) {
        return MR_COLLISION_PAIR_SPHERE_BOX;
    }
    if ((typeA == MR_SHAPE_CAPSULE &&
         typeB == MR_SHAPE_BOX) ||
        (typeB == MR_SHAPE_CAPSULE &&
         typeA == MR_SHAPE_BOX)) {
        return MR_COLLISION_PAIR_CAPSULE_BOX;
    }
    if (typeA == MR_SHAPE_BOX &&
        typeB == MR_SHAPE_BOX) {
        return MR_COLLISION_PAIR_BOX_BOX;
    }
    const bool surfaceA =
        typeA == MR_SHAPE_TRIANGLE_MESH ||
        typeA == MR_SHAPE_HEIGHTFIELD;
    const bool surfaceB =
        typeB == MR_SHAPE_TRIANGLE_MESH ||
        typeB == MR_SHAPE_HEIGHTFIELD;
    if (surfaceA || surfaceB) {
        return surfaceA && surfaceB
            ? MR_COLLISION_PAIR_UNSUPPORTED
            : MR_COLLISION_PAIR_MESH;
    }
    const bool supportMappedA =
        typeA == MR_SHAPE_SPHERE ||
        typeA == MR_SHAPE_CAPSULE ||
        typeA == MR_SHAPE_BOX ||
        typeA == MR_SHAPE_CYLINDER ||
        typeA == MR_SHAPE_CONVEX;
    const bool supportMappedB =
        typeB == MR_SHAPE_SPHERE ||
        typeB == MR_SHAPE_CAPSULE ||
        typeB == MR_SHAPE_BOX ||
        typeB == MR_SHAPE_CYLINDER ||
        typeB == MR_SHAPE_CONVEX;
    if ((supportMappedA && supportMappedB) ||
        ((typeA == MR_SHAPE_PLANE) && supportMappedB) ||
        ((typeB == MR_SHAPE_PLANE) && supportMappedA)) {
        return MR_COLLISION_PAIR_CONVEX;
    }
    return MR_COLLISION_PAIR_UNSUPPORTED;
}

std::uint64_t compiledFingerprint(
    const EngineModel& model,
    const MetalWorldCapacityProfile& capacities,
    const std::vector<std::uint32_t>& sceneBodyIndices,
    const std::vector<MRCompiledCollisionPairGPU>& eligiblePairs,
    const std::vector<MRWorldDynamicNodeGPU>& dynamicNodes,
    const std::vector<std::uint32_t>& bodyDynamicNodes
) {
    std::uint64_t hash = fingerprint(model);
    hashValue(hash, capacities);
    hashVector(hash, sceneBodyIndices);
    hashVector(hash, eligiblePairs);
    hashVector(hash, dynamicNodes);
    hashVector(hash, bodyDynamicNodes);
    return hash == 0u ? 1u : hash;
}

bool buildRequirements(
    const CompiledWorld& world,
    const MetalWorldLayout& layout,
    const CompiledTaskProgram& taskProgram,
    const CompiledPolicyProgram& policyProgram,
    const MetalWorldMulticopterProgram& multicopterProgram,
    const MetalWorldFlappingWingProgram& flappingWingProgram,
    RequiredBuffers& requirements,
    std::size_t& totalRequiredBytes
) {
    const EngineModel& model = world.model();
    ParallelABASchedule parallelSchedule;
    const bool streamedResponses =
        (layout.contactDispatch.flags &
         MR_METAL_WORLD_CONTACT_STREAMED_RESPONSES) != 0u;
    if (streamedResponses &&
        !compileParallelABASchedule(
             model,
             parallelSchedule
         ).succeeded()) {
        return false;
    }
    const std::size_t jointElements =
        std::max<std::size_t>(model.joints.size(), 1u);
    const std::size_t resetMaskElements =
        layout.resetMaskElements;
    const std::size_t resetQElements = layout.resetQElements;
    const std::size_t resetVElements = layout.resetVElements;
    const std::size_t environments =
        layout.dispatch.environmentCount;
    const std::size_t contactEnvironments =
        (layout.dispatch.flags & MR_METAL_WORLD_CONTACTS) != 0u
        ? environments
        : 0u;
    const bool nativeTask = taskProgram.valid();
    const bool nativePolicy = policyProgram.valid();
    const std::size_t taskEnvironments =
        nativeTask ? environments : 0u;
    const TaskProgramLayout& taskLayout =
        taskProgram.layout();
    std::size_t taskActionHistoryStride = 0u;
    std::size_t taskActionHistoryElements = 0u;
    std::size_t taskHistoryElements = 0u;
    std::size_t taskCriticHistoryElements = 0u;
    std::size_t taskBodyParameterElements = 0u;
    std::size_t taskContactElements = 0u;
    std::size_t policyScratchElements = 0u;
    std::size_t policyActorMeanElements = 0u;
    if (!checkedMultiply(
            taskLayout.delayStateCount,
            taskLayout.actionCount,
            taskActionHistoryStride
        ) ||
        !checkedMultiply(
            taskEnvironments,
            taskActionHistoryStride,
            taskActionHistoryElements
        ) ||
        !checkedMultiply(
            taskEnvironments,
            taskLayout.actorObservationSize,
            taskHistoryElements
        ) ||
        !checkedMultiply(
            taskEnvironments,
            static_cast<std::size_t>(
                taskLayout.criticFrameSize
            ) * taskLayout.criticHistoryLength,
            taskCriticHistoryElements
        ) ||
        !checkedMultiply(
            taskEnvironments,
            model.bodies.size(),
            taskBodyParameterElements
        ) ||
        !checkedMultiply(
            taskEnvironments,
            taskLayout.contactMetricCount,
            taskContactElements
        ) ||
        !checkedMultiply(
            nativePolicy ? taskEnvironments : 0u,
            policyProgram.layout().maximumHiddenCount,
            policyScratchElements
        ) ||
        !checkedMultiply(
            nativePolicy ? taskEnvironments : 0u,
            policyProgram.layout().actionCount,
            policyActorMeanElements
        )) {
        return false;
    }
    if (!makeRequirement<MRWorldGPU>(
            "runtime world",
            1u,
            requirements.entries[kWorld]
        ) ||
        !makeRequirement<MRArticulationGPU>(
            "articulations",
            model.articulations.size(),
            requirements.entries[kArticulations]
        ) ||
        !makeRequirement<MRJointDescriptorGPU>(
            "joints",
            jointElements,
            requirements.entries[kJoints]
        ) ||
        !makeRequirement<MRDofPropertiesGPU>(
            "DoF properties",
            model.dofs.size(),
            requirements.entries[kDofs]
        ) ||
        !makeRequirement<MRActuatorProfileGPU>(
            "actuator profiles",
            model.dofs.size(),
            requirements.entries[kActuatorProfiles]
        ) ||
        !makeRequirement<MRBodyPropertiesGPU>(
            "body properties",
            model.bodies.size(),
            requirements.entries[kBodies]
        ) ||
        !makeRequirement<MRMultiABADispatchGPU>(
            "multi-articulation ABA dispatches",
            model.articulations.size(),
            requirements.entries[kABADispatch]
        ) ||
        !makeRequirement<float>(
            "state q A",
            layout.initialQElements,
            requirements.entries[kStateQA]
        ) ||
        !makeRequirement<float>(
            "state v A",
            layout.initialVElements,
            requirements.entries[kStateVA]
        ) ||
        !makeRequirement<float>(
            "working effort",
            layout.initialVElements,
            requirements.entries[kWorkingEffort]
        ) ||
        !makeRequirement<MRABABodyWrenchGPU>(
            "external body wrenches",
            (layout.dispatch.flags &
             MR_METAL_WORLD_HAS_BODY_WRENCHES) != 0u
                ? layout.dispatch.environmentCount * model.bodies.size()
                : 0u,
            requirements.entries[kBodyWrenchPlaceholder]
        ) ||
        !makeRequirement<MRMulticopterRotorGPU>(
            "compiled multicopter rotors",
            multicopterProgram.valid()
                ? multicopterProgram.model.rotorCount
                : 0u,
            requirements.entries[kMulticopterRotors]
        ) ||
        !makeRequirement<MRMulticopterModelGPU>(
            "compiled multicopter model",
            multicopterProgram.valid() ? 1u : 0u,
            requirements.entries[kMulticopterModel]
        ) ||
        !makeRequirement<MRMulticopterMixerGPU>(
            "compiled multicopter mixer",
            multicopterProgram.valid() ? 1u : 0u,
            requirements.entries[kMulticopterMixer]
        ) ||
        !makeRequirement<MRMulticopterStateGPU>(
            "resident multicopter motor state A",
            multicopterProgram.valid()
                ? layout.dispatch.environmentCount
                : 0u,
            requirements.entries[kMulticopterStateA]
        ) ||
        !makeRequirement<MRMulticopterStateGPU>(
            "resident multicopter motor state B",
            multicopterProgram.valid()
                ? layout.dispatch.environmentCount
                : 0u,
            requirements.entries[kMulticopterStateB]
        ) ||
        !makeRequirement<MRMulticopterStateGPU>(
            "candidate multicopter motor state",
            multicopterProgram.valid()
                ? layout.dispatch.environmentCount
                : 0u,
            requirements.entries[kMulticopterCandidateState]
        ) ||
        !makeRequirement<MRCompiledMulticopterDispatchGPU>(
            "compiled multicopter dispatch",
            multicopterProgram.valid() ? 1u : 0u,
            requirements.entries[kMulticopterDispatch]
        ) ||
        !makeRequirement<MRFlappingWingGPU>(
            "compiled flapping-wing geometry",
            flappingWingProgram.valid() ? flappingWingProgram.wings.size() : 0u,
            requirements.entries[kFlappingWingSpecs]
        ) ||
        !makeRequirement<MRCompiledFlappingWingDispatchGPU>(
            "compiled flapping-wing dispatch",
            flappingWingProgram.valid() ? 1u : 0u,
            requirements.entries[kFlappingWingDispatch]
        ) ||
        !makeRequirement<float>(
            "candidate acceleration",
            layout.initialVElements,
            requirements.entries[kCandidateAcceleration]
        ) ||
        !makeRequirement<float>(
            "candidate v",
            layout.initialVElements,
            requirements.entries[kCandidateV]
        ) ||
        !makeRequirement<float>(
            "candidate q",
            layout.initialQElements,
            requirements.entries[kCandidateQ]
        ) ||
        !makeRequirement<MRABAStatusGPU>(
            "ABA statuses",
            layout.articulationStatusElements,
            requirements.entries[kABAStatuses]
        ) ||
        !makeRequirement<float>(
            "state q B",
            layout.initialQElements,
            requirements.entries[kStateQB]
        ) ||
        !makeRequirement<float>(
            "state v B",
            layout.initialVElements,
            requirements.entries[kStateVB]
        ) ||
        !makeRequirement<float>(
            "effort trajectory",
            layout.effortElements,
            requirements.entries[kEffortTrajectory]
        ) ||
        !makeRequirement<mr_u32>(
            "reset masks",
            resetMaskElements,
            requirements.entries[kResetMasks]
        ) ||
        !makeRequirement<float>(
            "reset q",
            resetQElements,
            requirements.entries[kResetQ]
        ) ||
        !makeRequirement<float>(
            "reset v",
            resetVElements,
            requirements.entries[kResetV]
        ) ||
        !makeRequirement<float>(
            "observations",
            layout.observationElements,
            requirements.entries[kObservations]
        ) ||
        !makeRequirement<float>(
            "acceleration trajectory",
            layout.accelerationElements,
            requirements.entries[kAccelerationTrajectory]
        ) ||
        !makeRequirement<MRMetalWorldStatusGPU>(
            "public statuses",
            layout.statusElements,
            requirements.entries[kPublicStatuses]
        ) ||
        !makeRequirement<MRMetalWorldDispatchGPU>(
            "world dispatch",
            1u,
            requirements.entries[kWorldDispatch]
        ) ||
        !makeRequirement<MRMetalWorldStatusGPU>(
            "environment statuses",
            layout.dispatch.environmentCount,
            requirements.entries[kEnvironmentStatuses]
        ) ||
        !makeRequirement<float>(
            "checkpoint q",
            layout.initialQElements,
            requirements.entries[kCheckpointQ]
        ) ||
        !makeRequirement<float>(
            "checkpoint v",
            layout.initialVElements,
            requirements.entries[kCheckpointV]
        )) {
        return false;
    }

    const MRMetalWorldContactDispatchGPU& contact =
        layout.contactDispatch;
    std::size_t bodyPoseElements = 0u;
    std::size_t bodyStateElements = 0u;
    std::size_t projectedColliderElements = 0u;
    std::size_t eligiblePairFlagElements = 0u;
    std::size_t pairElements = 0u;
    std::size_t rawContactElements = 0u;
    std::size_t pointQueryElements = 0u;
    std::size_t pointWorldElements = 0u;
    std::size_t factorElements = 0u;
    std::size_t pointJacobianElements = 0u;
    std::size_t endpointElements = 0u;
    std::size_t coneElements = 0u;
    std::size_t responseElements = 0u;
    std::size_t compactionElements = 0u;
    const std::size_t qualityEnvironments =
        contact.solverType == MR_SOLVER_QUALITY_NEWTON
        ? contactEnvironments
        : 0u;
    const std::size_t qualityNv =
        static_cast<std::size_t>(contact.nv) +
        6u * static_cast<std::size_t>(
            contact.sceneBodyCount
        ) +
        3u * static_cast<std::size_t>(
            contact.rodNodeCount
        ) +
        static_cast<std::size_t>(contact.rodEdgeCount);
    const std::size_t qualityRows =
        layout.qualityDispatch.rowCount;
    std::size_t qualityBlockElements = 0u;
    std::size_t qualityDynamicsElements = 0u;
    std::size_t qualityJacobianElements = 0u;
    std::size_t qualityVectorElements = 0u;
    std::size_t qualityDerivativeElements = 0u;
    std::size_t qualityHessianElements = 0u;
    std::size_t rodPairStateElements = 0u;
    std::size_t rodWitnessElements = 0u;
    if (qualityEnvironments != 0u &&
        (qualityNv >
             MR_UNIFIED_QUALITY_MAX_GENERALIZED_VELOCITIES ||
         qualityRows > MR_UNIFIED_QUALITY_MAX_ROWS ||
         layout.qualityDispatch.blockCount >
             MR_UNIFIED_QUALITY_MAX_BLOCKS)) {
        return false;
    }
    if (!checkedMultiply(
            contactEnvironments,
            world.rodToolPairs().size(),
            rodPairStateElements
        ) ||
        !checkedMultiply(
            rodPairStateElements,
            MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR,
            rodWitnessElements
        ) ||
        !checkedMultiply(
            contactEnvironments,
            world.bodyCount(),
            bodyPoseElements
        ) ||
        !checkedMultiply(
            contactEnvironments,
            model.bodies.size(),
            bodyStateElements
        ) ||
        !checkedMultiply(
            contactEnvironments,
            model.shapes.size(),
            projectedColliderElements
        ) ||
        !checkedMultiply(
            contactEnvironments,
            world.eligiblePairCount(),
            eligiblePairFlagElements
        ) ||
        !checkedMultiply(
            contactEnvironments,
            contact.pairStride,
            pairElements
        ) ||
        !checkedMultiply(
            contactEnvironments,
            contact.rawContactStride,
            rawContactElements
        ) ||
        !checkedMultiply(
            contactEnvironments,
            contact.pointQueryStride,
            pointWorldElements
        ) ||
        !checkedMultiply(
            pointWorldElements,
            world.articulationCount(),
            pointQueryElements
        ) ||
        !checkedMultiply(
            contactEnvironments,
            contact.factorStride,
            factorElements
        ) ||
        !checkedMultiply(
            pointWorldElements,
            3u * static_cast<std::size_t>(contact.nv),
            pointJacobianElements
        ) ||
        !checkedMultiply(
            layout.contactConstraintElements,
            2u,
            endpointElements
        ) ||
        !checkedMultiply(
            contactEnvironments,
            contact.constraintStride,
            coneElements
        ) ||
        !checkedMultiply(
            layout.contactConstraintElements,
            3u * static_cast<std::size_t>(contact.nv),
            responseElements
        ) ||
        !checkedMultiply(
            qualityEnvironments,
            static_cast<std::size_t>(
                layout.qualityDispatch.blockCount
            ),
            qualityBlockElements
        ) ||
        !checkedMultiply(
            qualityEnvironments,
            qualityNv * qualityNv,
            qualityDynamicsElements
        ) ||
        !checkedMultiply(
            qualityEnvironments,
            qualityRows * qualityNv,
            qualityJacobianElements
        ) ||
        !checkedMultiply(
            qualityEnvironments,
            std::max(qualityNv, qualityRows),
            qualityVectorElements
        ) ||
        !checkedMultiply(
            qualityBlockElements,
            36u,
            qualityDerivativeElements
        ) ||
        !checkedMultiply(
            qualityEnvironments,
            (
                qualityNv <=
                    layout.qualityDispatch
                        .directMaximumGeneralizedVelocities &&
                qualityRows <=
                    layout.qualityDispatch.directMaximumRows
                ? qualityNv * qualityNv
                : 1u
            ),
            qualityHessianElements
        )) {
        return false;
    }
    compactionElements = std::max(
        eligiblePairFlagElements,
        std::max(
            layout.islandElements,
            layout.contactTileElements
        )
    );
    const std::size_t immutableShapeElements =
        std::max<std::size_t>(model.shapes.size(), 1u);
    const std::size_t immutableMaterialElements =
        std::max<std::size_t>(model.materials.size(), 1u);
    const std::size_t immutableSceneIndexElements =
        std::max<std::size_t>(world.sceneBodyCount(), 1u);
    const std::size_t immutablePairElements =
        std::max<std::size_t>(world.eligiblePairCount(), 1u);
    if (!makeRequirement<MRShapeGPU>(
            "shapes",
            immutableShapeElements,
            requirements.entries[kShapes]
        ) ||
        !makeRequirement<MRMaterialGPU>(
            "materials",
            immutableMaterialElements,
            requirements.entries[kMaterials]
        ) ||
        !makeRequirement<mr_u32>(
            "scene body indices",
            immutableSceneIndexElements,
            requirements.entries[kSceneBodyIndices]
        ) ||
        !makeRequirement<MRCompiledCollisionPairGPU>(
            "eligible collision pairs",
            immutablePairElements,
            requirements.entries[kEligiblePairs]
        ) ||
        !makeRequirement<MRMetalWorldContactDispatchGPU>(
            "contact dispatch",
            1u,
            requirements.entries[kContactDispatch]
        ) ||
        !makeRequirement<MRInverseMassDispatchGPU>(
            "streamed inverse-mass dispatch",
            1u,
            requirements.entries[kInverseMassDispatch]
        ) ||
        !makeRequirement<MRParallelABAArticulationGPU>(
            "parallel ABA schedule articulations",
            streamedResponses
                ? parallelSchedule.articulations.size()
                : 0u,
            requirements.entries[kParallelScheduleArticulations]
        ) ||
        !makeRequirement<MRParallelABALevelGPU>(
            "parallel ABA schedule levels",
            streamedResponses ? parallelSchedule.levels.size() : 0u,
            requirements.entries[kParallelScheduleLevels]
        ) ||
        !makeRequirement<MRParallelABAParentReductionGPU>(
            "parallel ABA schedule parent reductions",
            streamedResponses
                ? parallelSchedule.parentReductions.size()
                : 0u,
            requirements.entries[kParallelScheduleParentReductions]
        ) ||
        !makeRequirement<mr_u32>(
            "parallel ABA schedule level bodies",
            streamedResponses
                ? parallelSchedule.levelBodies.size()
                : 0u,
            requirements.entries[kParallelScheduleLevelBodies]
        ) ||
        !makeRequirement<mr_u32>(
            "parallel ABA schedule parent indices",
            streamedResponses
                ? parallelSchedule.parentLocal.size()
                : 0u,
            requirements.entries[kParallelScheduleParentLocal]
        ) ||
        !makeRequirement<mr_u32>(
            "parallel ABA schedule inbound joints",
            streamedResponses
                ? parallelSchedule.inboundJoint.size()
                : 0u,
            requirements.entries[kParallelScheduleInboundJoint]
        ) ||
        !makeRequirement<mr_u32>(
            "parallel ABA schedule child offsets",
            streamedResponses
                ? parallelSchedule.childOffsets.size()
                : 0u,
            requirements.entries[kParallelScheduleChildOffsets]
        ) ||
        !makeRequirement<mr_u32>(
            "parallel ABA schedule child indices",
            streamedResponses
                ? parallelSchedule.childIndices.size()
                : 0u,
            requirements.entries[kParallelScheduleChildIndices]
        ) ||
        !makeRequirement<MRArticulatedOperatorDispatchGPU>(
            "multi-articulation kinematics dispatches",
            model.articulations.size(),
            requirements.entries[kOperatorKinematicsDispatch]
        ) ||
        !makeRequirement<MRArticulatedOperatorDispatchGPU>(
            "multi-articulation factor operator dispatches",
            model.articulations.size(),
            requirements.entries[kOperatorFactorDispatch]
        ) ||
        !makeRequirement<MRIndirectDispatchArgumentsGPU>(
            "active contact indirect dispatch arguments",
            contactEnvironments == 0u ? 0u : 2u,
            requirements.entries[kActiveIndirectDispatch]
        ) ||
        !makeRequirement<MRBodyStateGPU>(
            "initial scene bodies",
            layout.initialSceneBodyElements,
            requirements.entries[kInitialSceneBodies]
        ) ||
        !makeRequirement<MRBodyStateGPU>(
            "reset scene bodies",
            layout.resetSceneBodyElements,
            requirements.entries[kResetSceneBodies]
        ) ||
        !makeRequirement<MRBodyStateGPU>(
            "kinematic targets",
            layout.kinematicTargetElements,
            requirements.entries[kKinematicTargets]
        ) ||
        !makeRequirement<MRBodyStateGPU>(
            "scene bodies A",
            layout.initialSceneBodyElements,
            requirements.entries[kSceneBodiesA]
        ) ||
        !makeRequirement<MRBodyStateGPU>(
            "scene bodies B",
            layout.initialSceneBodyElements,
            requirements.entries[kSceneBodiesB]
        ) ||
        !makeRequirement<MRBodyStateGPU>(
            "checkpoint scene bodies",
            layout.initialSceneBodyElements,
            requirements.entries[kCheckpointSceneBodies]
        ) ||
        !makeRequirement<MRArticulatedBodyPoseGPU>(
            "articulation body poses",
            bodyPoseElements,
            requirements.entries[kBodyPoses]
        ) ||
        !makeRequirement<MRArticulatedBodyPoseGPU>(
            "unconstrained future articulation body poses",
            bodyPoseElements,
            requirements.entries[kFutureBodyPoses]
        ) ||
        !makeRequirement<MRArticulatedPointWorldGPU>(
            "articulated point world",
            pointWorldElements,
            requirements.entries[kPointWorld]
        ) ||
        !makeRequirement<float>(
            "articulation factor matrix",
            factorElements,
            requirements.entries[kFactorMatrix]
        ) ||
        !makeRequirement<float>(
            "point Jacobians",
            pointJacobianElements,
            requirements.entries[kPointJacobians]
        ) ||
        !makeRequirement<float>(
            "multi-articulation factor staging",
            factorElements,
            requirements.entries[kFactorMatrixStaging]
        ) ||
        !makeRequirement<float>(
            "multi-articulation Jacobian staging",
            pointJacobianElements,
            requirements.entries[kPointJacobiansStaging]
        ) ||
        !makeRequirement<float>(
            "generalized impulse",
            contactEnvironments == 0u
                ? 0u
                : layout.initialVElements,
            requirements.entries[kGeneralizedImpulse]
        ) ||
        !makeRequirement<float>(
            "operator delta velocity",
            contactEnvironments == 0u
                ? 0u
                : layout.initialVElements,
            requirements.entries[kDeltaVelocity]
        ) ||
        !makeRequirement<MRArticulatedOperatorStatusGPU>(
            "articulated operator statuses",
            contactEnvironments *
                model.articulations.size(),
            requirements.entries[kOperatorStatuses]
        ) ||
        !makeRequirement<MRInverseMassStatusGPU>(
            "streamed inverse-mass statuses",
            contactEnvironments,
            requirements.entries[kInverseMassStatuses]
        ) ||
        !makeRequirement<MRBodyStateGPU>(
            "current global body states",
            bodyStateElements,
            requirements.entries[kCurrentBodies]
        ) ||
        !makeRequirement<MRBodyStateGPU>(
            "candidate global body states",
            bodyStateElements,
            requirements.entries[kCandidateBodies]
        ) ||
        !makeRequirement<MRProjectedColliderGPU>(
            "projected colliders and AABBs",
            projectedColliderElements,
            requirements.entries[kProjectedColliders]
        ) ||
        !makeRequirement<MRProjectedColliderGPU>(
            "future projected colliders",
            projectedColliderElements,
            requirements.entries[kFutureProjectedColliders]
        ) ||
        !makeRequirement<mr_u32>(
            "eligible-pair overlap flags",
            eligiblePairFlagElements,
            requirements.entries[kPairOverlapFlags]
        ) ||
        !makeRequirement<MRManifoldHeaderGPU>(
            "manifold headers A",
            layout.manifoldHeaderElements,
            requirements.entries[kManifoldHeadersA]
        ) ||
        !makeRequirement<MRManifoldPointGPU>(
            "manifold points A",
            layout.manifoldPointElements,
            requirements.entries[kManifoldPointsA]
        ) ||
        !makeRequirement<mr_u32>(
            "manifold counts A",
            contactEnvironments,
            requirements.entries[kManifoldCountsA]
        ) ||
        !makeRequirement<MRManifoldHeaderGPU>(
            "manifold headers B",
            layout.manifoldHeaderElements,
            requirements.entries[kManifoldHeadersB]
        ) ||
        !makeRequirement<MRManifoldPointGPU>(
            "manifold points B",
            layout.manifoldPointElements,
            requirements.entries[kManifoldPointsB]
        ) ||
        !makeRequirement<mr_u32>(
            "manifold counts B",
            contactEnvironments,
            requirements.entries[kManifoldCountsB]
        ) ||
        !makeRequirement<MRManifoldHeaderGPU>(
            "candidate manifold headers",
            layout.manifoldHeaderElements,
            requirements.entries[kCandidateManifoldHeaders]
        ) ||
        !makeRequirement<MRManifoldPointGPU>(
            "candidate manifold points",
            layout.manifoldPointElements,
            requirements.entries[kCandidateManifoldPoints]
        ) ||
        !makeRequirement<mr_u32>(
            "candidate manifold counts",
            contactEnvironments,
            requirements.entries[kCandidateManifoldCounts]
        ) ||
        !makeRequirement<MRManifoldHeaderGPU>(
            "checkpoint manifold headers",
            layout.manifoldHeaderElements,
            requirements.entries[kCheckpointManifoldHeaders]
        ) ||
        !makeRequirement<MRManifoldPointGPU>(
            "checkpoint manifold points",
            layout.manifoldPointElements,
            requirements.entries[kCheckpointManifoldPoints]
        ) ||
        !makeRequirement<mr_u32>(
            "checkpoint manifold counts",
            contactEnvironments,
            requirements.entries[kCheckpointManifoldCounts]
        ) ||
        !makeRequirement<MRCandidatePairGPU>(
            "candidate pairs",
            pairElements,
            requirements.entries[kCandidatePairs]
        ) ||
        !makeRequirement<MRRawContactGPU>(
            "raw contacts",
            rawContactElements,
            requirements.entries[kRawContacts]
        ) ||
        !makeRequirement<mr_u32>(
            "raw contact pair indices",
            rawContactElements,
            requirements.entries[kRawPairIndices]
        ) ||
        !makeRequirement<MRContactConstraintGPU>(
            "contact constraints",
            layout.contactConstraintElements,
            requirements.entries[kContacts]
        ) ||
        !makeRequirement<MRContactPointMetaGPU>(
            "contact metadata",
            layout.contactConstraintElements,
            requirements.entries[kContactMetadata]
        ) ||
        !makeRequirement<MRConstraintIRBlockGPU>(
            "ConstraintIR blocks",
            layout.contactConstraintElements,
            requirements.entries[kIRBlocks]
        ) ||
        !makeRequirement<MRConstraintIREndpointGPU>(
            "ConstraintIR endpoints",
            endpointElements,
            requirements.entries[kIREndpoints]
        ) ||
        !makeRequirement<MRConstraintEndpointRuntimeGPU>(
            "ConstraintIR endpoint runtime bindings",
            endpointElements,
            requirements.entries[kEndpointRuntime]
        ) ||
        !makeRequirement<MRConstraintIRRowGPU>(
            "ConstraintIR rows",
            layout.constraintRowElements,
            requirements.entries[kIRRows]
        ) ||
        !makeRequirement<MRConstraintIRConeGPU>(
            "ConstraintIR cones",
            coneElements,
            requirements.entries[kIRCones]
        ) ||
        !makeRequirement<MRArticulatedPointImpulseGPU>(
            "articulated point queries",
            pointQueryElements,
            requirements.entries[kPointQueries]
        ) ||
        !makeRequirement<MREvaluatedConstraintIRRowGPU>(
            "evaluated ConstraintIR rows",
            layout.constraintRowElements,
            requirements.entries[kEvaluatedRows]
        ) ||
        !makeRequirement<MREvaluatedConstraintIRConeGPU>(
            "evaluated ConstraintIR cones",
            coneElements,
            requirements.entries[kEvaluatedCones]
        ) ||
        !makeRequirement<MRArticulationFactorCacheGPU>(
            "articulation factor cache",
            contactEnvironments * model.articulations.size(),
            requirements.entries[kFactorCaches]
        ) ||
        !makeRequirement<MRContactIslandGPU>(
            "contact islands",
            layout.islandElements,
            requirements.entries[kIslands]
        ) ||
        !makeRequirement<float>(
            "contact response columns",
            responseElements,
            requirements.entries[kResponseColumns]
        ) ||
        !makeRequirement<MRMetalWorldContactStatusGPU>(
            "contact statuses",
            contactEnvironments,
            requirements.entries[kContactStatuses]
        ) ||
        !makeRequirement<MRMetalWorldContactStatusGPU>(
            "public contact statuses",
            layout.contactStatusElements,
            requirements.entries[kPublicContactStatuses]
        ) ||
        !makeRequirement<MRWorkQueueHeaderGPU>(
            "work queue headers",
            layout.workQueueHeaderElements,
            requirements.entries[kWorkQueueHeaders]
        ) ||
        !makeRequirement<MRPairWorkGPU>(
            "compacted pair work",
            layout.pairWorkElements,
            requirements.entries[kPairWorkQueue]
        ) ||
        !makeRequirement<std::uint64_t>(
            "per-pair raw contact count and compact slot",
            eligiblePairFlagElements,
            requirements.entries[kPairRawCounts]
        ) ||
        !makeRequirement<MRManifoldHeaderGPU>(
            "active-pair manifold headers",
            layout.pairWorkElements,
            requirements.entries[kPairManifoldHeaders]
        ) ||
        !makeRequirement<MRManifoldPointGPU>(
            "active-pair manifold points",
            layout.pairWorkElements *
                MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY,
            requirements.entries[kPairManifoldPoints]
        ) ||
        !makeRequirement<MRManifoldIRScatterGPU>(
            "manifold-to-IR scatter records",
            eligiblePairFlagElements,
            requirements.entries[kManifoldIRScatter]
        ) ||
        !makeRequirement<mr_u32>(
            "compaction offsets",
            compactionElements,
            requirements.entries[kCompactionOffsets]
        ) ||
        !makeRequirement<mr_u32>(
            "hierarchical scan scratch",
            2u * compactionElements,
            requirements.entries[kCompactionScratch]
        ) ||
        !makeRequirement<mr_u32>(
            "compaction flags",
            compactionElements,
            requirements.entries[kCompactionFlags]
        ) ||
        !makeRequirement<MRIslandWorkGPU>(
            "compacted island work",
            2u * layout.islandWorkElements,
            requirements.entries[kIslandWorkQueue]
        ) ||
        !makeRequirement<MRContactTileGPU>(
            "contact tiles",
            2u * layout.contactTileElements,
            requirements.entries[kContactTiles]
        ) ||
        !makeRequirement<mr_u32>(
            "tile constraint indices",
            layout.contactConstraintElements,
            requirements.entries[kTileConstraintIndices]
        ) ||
        !makeRequirement<mr_float4>(
            "Wave32 impulse deltas",
            layout.contactConstraintElements,
            requirements.entries[kWave32ImpulseDeltas]
        ) ||
        !makeRequirement<MRWave32IslandStatusGPU>(
            "Wave32 island statuses",
            layout.islandWorkElements,
            requirements.entries[kWave32IslandStatuses]
        ) ||
        !makeRequirement<MRConvexQueryCacheGPU>(
            "persistent convex query cache",
            layout.convexCacheElements,
            requirements.entries[kConvexCaches]
        ) ||
        !makeRequirement<MRConvexQueryCacheGPU>(
            "candidate convex query cache",
            layout.convexCacheElements,
            requirements.entries[kCandidateConvexCaches]
        ) ||
        !makeRequirement<MRCCDPairGPU>(
            "CCD candidates",
            layout.ccdPairElements,
            requirements.entries[kCCDPairs]
        ) ||
        !makeRequirement<MRCCDEventStateGPU>(
            "CCD event states A",
            layout.ccdEventStateElements,
            requirements.entries[kCCDEventStatesA]
        ) ||
        !makeRequirement<MRCCDEventStateGPU>(
            "CCD event states B",
            layout.ccdEventStateElements,
            requirements.entries[kCCDEventStatesB]
        ) ||
        !makeRequirement<MRCCDImpactClusterGPU>(
            "CCD impact clusters",
            layout.ccdImpactClusterElements,
            requirements.entries[kCCDImpactClusters]
        ) ||
        !makeRequirement<MRWaveWorkPacketGPU>(
            "Wave32 work packets",
            layout.waveWorkPacketElements,
            requirements.entries[kWaveWorkPackets]
        ) ||
        !makeRequirement<MRGeometryHeaderGPU>(
            "geometry headers",
            std::max<std::size_t>(
                model.geometryHeaders.size(),
                1u
            ),
            requirements.entries[kGeometryHeaders]
        ) ||
        !makeRequirement<mr_float4>(
            "geometry vertices",
            std::max<std::size_t>(
                model.geometryVertices.size(),
                1u
            ),
            requirements.entries[kGeometryVertices]
        ) ||
        !makeRequirement<mr_u32>(
            "geometry indices",
            std::max<std::size_t>(
                model.geometryIndices.size(),
                1u
            ),
            requirements.entries[kGeometryIndices]
        ) ||
        !makeRequirement<MRConvexFaceGPU>(
            "convex faces",
            std::max<std::size_t>(
                model.convexFaces.size(),
                1u
            ),
            requirements.entries[kConvexFaces]
        ) ||
        !makeRequirement<MRConvexHalfEdgeGPU>(
            "convex half edges",
            std::max<std::size_t>(
                model.convexHalfEdges.size(),
                1u
            ),
            requirements.entries[kConvexHalfEdges]
        ) ||
        !makeRequirement<MRMeshBVHNodeGPU>(
            "mesh BVH nodes",
            std::max<std::size_t>(
                model.meshBvhNodes.size(),
                1u
            ),
            requirements.entries[kMeshBvhNodes]
        ) ||
        !makeRequirement<MRMeshTriangleGPU>(
            "mesh triangles",
            std::max<std::size_t>(
                model.meshTriangles.size(),
                1u
            ),
            requirements.entries[kMeshTriangles]
        ) ||
        !makeRequirement<MRRawContactGPU>(
            "pair raw contact staging",
            layout.pairRawStagingElements,
            requirements.entries[kPairRawContactStaging]
        ) ||
        !makeRequirement<MRWave32PreconditionerGPU>(
            "Wave32 coupled block preconditioners",
            layout.contactConstraintElements,
            requirements.entries[kWave32Preconditioners]
        ) ||
        !makeRequirement<MRIslandWorkGPU>(
            "dense island work staging",
            layout.islandWorkElements,
            requirements.entries[kIslandWorkDense]
        ) ||
        !makeRequirement<MRUnifiedQualityDispatchGPU>(
            "unified quality dispatch",
            qualityEnvironments == 0u ? 0u : 1u,
            requirements.entries[kQualityDispatch]
        ) ||
        !makeRequirement<MRUnifiedQualityBlockGPU>(
            "unified quality blocks",
            qualityBlockElements,
            requirements.entries[kQualityBlocks]
        ) ||
        !makeRequirement<float>(
            "unified quality dynamics",
            qualityDynamicsElements,
            requirements.entries[kQualityDynamics]
        ) ||
        !makeRequirement<float>(
            "unified quality Jacobian",
            qualityJacobianElements,
            requirements.entries[kQualityJacobian]
        ) ||
        !makeRequirement<float>(
            "unified quality bias",
            qualityVectorElements,
            requirements.entries[kQualityBias]
        ) ||
        !makeRequirement<float>(
            "unified quality free velocity",
            qualityVectorElements,
            requirements.entries[kQualityFreeVelocity]
        ) ||
        !makeRequirement<float>(
            "unified quality warm velocity",
            qualityVectorElements,
            requirements.entries[kQualityWarmVelocity]
        ) ||
        !makeRequirement<float>(
            "unified quality warm impulses",
            qualityVectorElements,
            requirements.entries[kQualityWarmImpulses]
        ) ||
        !makeRequirement<float>(
            "unified quality output velocity",
            qualityVectorElements,
            requirements.entries[kQualityOutputVelocity]
        ) ||
        !makeRequirement<float>(
            "unified quality output impulses",
            qualityVectorElements,
            requirements.entries[kQualityOutputImpulses]
        ) ||
        !makeRequirement<float>(
            "unified quality block derivatives",
            qualityDerivativeElements,
            requirements.entries[kQualityDerivatives]
        ) ||
        !makeRequirement<float>(
            "unified quality Hessian scratch",
            qualityHessianElements,
            requirements.entries[kQualityHessian]
        ) ||
        !makeRequirement<MRUnifiedQualityStatusGPU>(
            "unified quality statuses",
            qualityEnvironments,
            requirements.entries[kQualityStatuses]
        ) ||
        !makeRequirement<MRUnifiedQualityWorkQueueGPU>(
            "unified quality work queue",
            qualityEnvironments == 0u ? 0u : 1u,
            requirements.entries[kQualityWorkQueue]
        ) ||
        !makeRequirement<MRUnifiedQualityWorkPacketGPU>(
            "unified quality work packets",
            qualityEnvironments,
            requirements.entries[kQualityWorkPackets]
        ) ||
        !makeRequirement<MRWorldDynamicNodeGPU>(
            "typed dynamic nodes",
            world.dynamicNodes().size(),
            requirements.entries[kDynamicNodes]
        ) ||
        !makeRequirement<mr_u32>(
            "body to dynamic-node map",
            world.bodyDynamicNodes().size(),
            requirements.entries[kBodyDynamicNodes]
        ) ||
        !makeRequirement<MRIslandNodeRefGPU>(
            "island node references",
            layout.islandNodeReferenceElements,
            requirements.entries[kIslandNodeReferences]
        ) ||
        !makeRequirement<MRIslandConstraintRefGPU>(
            "island constraint references",
            layout.islandConstraintReferenceElements,
            requirements.entries[kIslandConstraintReferences]
        ) ||
        !makeRequirement<MRRodFactorCacheGPU>(
            "rod factor caches",
            layout.rodFactorCacheElements,
            requirements.entries[kRodFactorCaches]
        ) ||
        !makeRequirement<float>(
            "typed operator velocity arena",
            layout.operatorVelocityElements,
            requirements.entries[kOperatorVelocityArena]
        ) ||
        !makeRequirement<MRRodNodeStateGPU>(
            "initial rod nodes",
            layout.rodNodeStateElements,
            requirements.entries[kInitialRodNodes]
        ) ||
        !makeRequirement<MRRodEdgeStateGPU>(
            "initial rod edges",
            layout.rodEdgeStateElements,
            requirements.entries[kInitialRodEdges]
        ) ||
        !makeRequirement<MRRodNodeStateGPU>(
            "reset rod nodes",
            layout.resetRodNodeStateElements,
            requirements.entries[kResetRodNodes]
        ) ||
        !makeRequirement<MRRodEdgeStateGPU>(
            "reset rod edges",
            layout.resetRodEdgeStateElements,
            requirements.entries[kResetRodEdges]
        ) ||
        !makeRequirement<MRRodNodeStateGPU>(
            "rod node state A",
            layout.rodNodeStateElements,
            requirements.entries[kRodNodesA]
        ) ||
        !makeRequirement<MRRodEdgeStateGPU>(
            "rod edge state A",
            layout.rodEdgeStateElements,
            requirements.entries[kRodEdgesA]
        ) ||
        !makeRequirement<MRRodNodeStateGPU>(
            "rod node state B",
            layout.rodNodeStateElements,
            requirements.entries[kRodNodesB]
        ) ||
        !makeRequirement<MRRodEdgeStateGPU>(
            "rod edge state B",
            layout.rodEdgeStateElements,
            requirements.entries[kRodEdgesB]
        ) ||
        !makeRequirement<MRRodNodeStateGPU>(
            "checkpoint rod nodes",
            layout.rodNodeStateElements,
            requirements.entries[kCheckpointRodNodes]
        ) ||
        !makeRequirement<MRRodEdgeStateGPU>(
            "checkpoint rod edges",
            layout.rodEdgeStateElements,
            requirements.entries[kCheckpointRodEdges]
        ) ||
        !makeRequirement<MRRodGPUDispatch>(
            "rod component dispatches",
            world.rodCount(),
            requirements.entries[kRodDispatches]
        ) ||
        !makeRequirement<float>(
            "rod rest lengths",
            world.rodEdgeCount(),
            requirements.entries[kRodRestLengths]
        ) ||
        !makeRequirement<float>(
            "rod rest twists",
            world.rodEdgeCount(),
            requirements.entries[kRodRestTwists]
        ) ||
        !makeRequirement<mr_float4>(
            "rod rest curvatures",
            layout.rodBendStateElements,
            requirements.entries[kRodRestCurvatures]
        ) ||
        !makeRequirement<float>(
            "rod inverse masses",
            world.rodNodeCount(),
            requirements.entries[kRodInverseMasses]
        ) ||
        !makeRequirement<float>(
            "rod inverse rotational inertias",
            world.rodEdgeCount(),
            requirements.entries[kRodInverseRotationalInertias]
        ) ||
        !makeRequirement<float>(
            "rod stretch stiffness",
            world.rodEdgeCount(),
            requirements.entries[kRodStretchStiffness]
        ) ||
        !makeRequirement<float>(
            "rod bend stiffness",
            layout.rodBendStateElements,
            requirements.entries[kRodBendStiffness]
        ) ||
        !makeRequirement<float>(
            "rod twist stiffness",
            layout.rodBendStateElements,
            requirements.entries[kRodTwistStiffness]
        ) ||
        !makeRequirement<mr_float4>(
            "rod input positions",
            layout.rodNodeStateElements,
            requirements.entries[kRodInputPositions]
        ) ||
        !makeRequirement<mr_float4>(
            "rod input velocities",
            layout.rodNodeStateElements,
            requirements.entries[kRodInputVelocities]
        ) ||
        !makeRequirement<float>(
            "rod input twists",
            layout.rodEdgeStateElements,
            requirements.entries[kRodInputTwists]
        ) ||
        !makeRequirement<float>(
            "rod input twist rates",
            layout.rodEdgeStateElements,
            requirements.entries[kRodInputTwistRates]
        ) ||
        !makeRequirement<mr_float4>(
            "rod output positions",
            layout.rodNodeStateElements,
            requirements.entries[kRodOutputPositions]
        ) ||
        !makeRequirement<mr_float4>(
            "rod output velocities",
            layout.rodNodeStateElements,
            requirements.entries[kRodOutputVelocities]
        ) ||
        !makeRequirement<float>(
            "rod output twists",
            layout.rodEdgeStateElements,
            requirements.entries[kRodOutputTwists]
        ) ||
        !makeRequirement<float>(
            "rod output twist rates",
            layout.rodEdgeStateElements,
            requirements.entries[kRodOutputTwistRates]
        ) ||
        !makeRequirement<MRRodGPUStatus>(
            "rod component statuses",
            layout.rodStatusElements,
            requirements.entries[kRodStatuses]
        ) ||
        !makeRequirement<MRRodGPUAttachment>(
            "rod resolved attachments",
            0u,
            requirements.entries[kRodAttachments]
        ) ||
        !makeRequirement<MRRodGPUAttachmentReaction>(
            "rod attachment reactions",
            0u,
            requirements.entries[kRodReactions]
        ) ||
        !makeRequirement<MRRodColliderGPU>(
            "procedural rod colliders",
            world.rodColliders().size(),
            requirements.entries[kRodColliders]
        ) ||
        !makeRequirement<MRShapeGPU>(
            "rod collider shape sources",
            world.rodShapeSources().size(),
            requirements.entries[kRodShapeSources]
        ) ||
        !makeRequirement<MRRodToolPairGPU>(
            "compiled rod/tool pairs",
            world.rodToolPairs().size(),
            requirements.entries[kRodToolPairs]
        ) ||
        !makeRequirement<mr_u32>(
            "rod/tool witness counts",
            rodPairStateElements,
            requirements.entries[kRodWitnessCounts]
        ) ||
        !makeRequirement<MRRodToolWitnessGPU>(
            "published rod/tool witnesses A",
            rodWitnessElements,
            requirements.entries[kRodWitnessesA]
        ) ||
        !makeRequirement<MRRodToolWitnessGPU>(
            "published rod/tool witnesses B",
            rodWitnessElements,
            requirements.entries[kRodWitnessesB]
        ) ||
        !makeRequirement<MRRodToolWitnessGPU>(
            "candidate rod/tool witnesses",
            rodWitnessElements,
            requirements.entries[kCandidateRodWitnesses]
        ) ||
        !makeRequirement<MRRodToolWitnessGPU>(
            "checkpoint rod/tool witnesses",
            rodWitnessElements,
            requirements.entries[kCheckpointRodWitnesses]
        ) ||
        !makeRequirement<MRRodNodeStateGPU>(
            "CCD event rod nodes A",
            layout.rodNodeStateElements,
            requirements.entries[kCCDEventRodNodesA]
        ) ||
        !makeRequirement<MRRodEdgeStateGPU>(
            "CCD event rod edges A",
            layout.rodEdgeStateElements,
            requirements.entries[kCCDEventRodEdgesA]
        ) ||
        !makeRequirement<MRRodToolWitnessGPU>(
            "CCD event rod witnesses A",
            rodWitnessElements,
            requirements.entries[kCCDEventRodWitnessesA]
        ) ||
        !makeRequirement<MRRodNodeStateGPU>(
            "CCD event rod nodes B",
            layout.rodNodeStateElements,
            requirements.entries[kCCDEventRodNodesB]
        ) ||
        !makeRequirement<MRRodEdgeStateGPU>(
            "CCD event rod edges B",
            layout.rodEdgeStateElements,
            requirements.entries[kCCDEventRodEdgesB]
        ) ||
        !makeRequirement<MRRodToolWitnessGPU>(
            "CCD event rod witnesses B",
            rodWitnessElements,
            requirements.entries[kCCDEventRodWitnessesB]
        ) ||
        !makeRequirement<MRProjectedColliderGPU>(
            "projected procedural rod colliders",
            layout.rodEdgeStateElements,
            requirements.entries[kProjectedRodColliders]
        ) ||
        !makeRequirement<MRProjectedColliderGPU>(
            "future projected procedural rod colliders",
            layout.rodEdgeStateElements,
            requirements.entries[kFutureProjectedRodColliders]
        ) ||
        !makeRequirement<mr_u32>(
            "rod constraint/witness bindings",
            layout.contactConstraintElements,
            requirements.entries[kRodConstraintWitnessIndices]
        ) ||
        !makeRequirement<MRRodGPUDispatch>(
            "rod collision dispatches",
            world.rodCount(),
            requirements.entries[kRodCollisionDispatches]
        ) ||
        !makeRequirement<mr_u32>(
            "rod contact scan scratch",
            rodWitnessElements,
            requirements.entries[kRodContactScratch]
        ) ||
        !makeRequirement<MRConstraintIRBlockGPU>(
            "immutable authored ConstraintIR blocks",
            model.constraintProgram.blocks.size(),
            requirements.entries[kAuthoredIRBlocks]
        ) ||
        !makeRequirement<MRConstraintIREndpointGPU>(
            "immutable authored ConstraintIR endpoints",
            model.constraintProgram.endpoints.size(),
            requirements.entries[kAuthoredIREndpoints]
        ) ||
        !makeRequirement<MRConstraintIRRowGPU>(
            "immutable authored ConstraintIR rows",
            model.constraintProgram.rows.size(),
            requirements.entries[kAuthoredIRRows]
        ) ||
        !makeRequirement<MRConstraintIRConeGPU>(
            "immutable authored ConstraintIR cones",
            model.constraintProgram.cones.size(),
            requirements.entries[kAuthoredIRCones]
        ) ||
        !makeRequirement<float>(
            "immutable authored ConstraintIR warm impulses",
            model.constraintProgram.warmImpulses.size(),
            requirements.entries[kAuthoredIRWarmImpulses]
        ) ||
        !makeRequirement<MRTaskDispatchGPU>(
            "native task dispatch",
            nativeTask ? 1u : 0u,
            requirements.entries[kTaskDispatch]
        ) ||
        !makeRequirement<float>(
            "native task normalized actions",
            layout.actionElements,
            requirements.entries[kTaskActions]
        ) ||
        !makeRequirement<MRTaskStateGPU>(
            "native resident task state",
            taskEnvironments,
            requirements.entries[kTaskState]
        ) ||
        !makeRequirement<MRTaskEvidenceStateGPU>(
            "native resident task evidence state",
            nativeTask ? 1u : 0u,
            requirements.entries[kTaskEvidenceState]
        ) ||
        !makeRequirement<float>(
            "native delayed action history",
            taskActionHistoryElements,
            requirements.entries[kTaskActionHistory]
        ) ||
        !makeRequirement<float>(
            "native actor history",
            taskHistoryElements,
            requirements.entries[kTaskActorHistory]
        ) ||
        !makeRequirement<float>(
            "native clean observation history",
            taskHistoryElements,
            requirements.entries[kTaskCleanHistory]
        ) ||
        !makeRequirement<float>(
            "native critic observation history",
            taskCriticHistoryElements,
            requirements.entries[kTaskCriticHistory]
        ) ||
        !makeRequirement<float>(
            "native previous action velocity",
            taskEnvironments * taskLayout.actionCount * 2u,
            requirements.entries[kTaskPreviousJointVelocity]
        ) ||
        !makeRequirement<float>(
            "native task sensor bias",
            taskEnvironments * taskLayout.biasCount,
            requirements.entries[kTaskEncoderBias]
        ) ||
        !makeRequirement<mr_float4>(
            "native body domain parameters",
            taskBodyParameterElements,
            requirements.entries[kTaskBodyParameters]
        ) ||
        !makeRequirement<mr_float4>(
            "native controller domain parameters",
            taskEnvironments,
            requirements.entries[kTaskControllerParameters]
        ) ||
        !makeRequirement<float>(
            "native actor observations",
            layout.actorObservationElements,
            requirements.entries[kTaskActorObservations]
        ) ||
        !makeRequirement<float>(
            "native critic observations",
            layout.criticObservationElements,
            requirements.entries[kTaskCriticObservations]
        ) ||
        !makeRequirement<MRTaskTransitionGPU>(
            "native task transitions",
            layout.transitionElements,
            requirements.entries[kTaskTransitions]
        ) ||
        !makeRequirement<float>(
            "native motion-prior features",
            layout.motionFeatureElements,
            requirements.entries[kTaskMotionFeatures]
        ) ||
        !makeRequirement<float>(
            "native physically realized imagination actions",
            layout.teacherActionElements,
            requirements.entries[kTaskTeacherActions]
        ) ||
        !makeRequirement<float>(
            "native compact contact metrics",
            taskContactElements,
            requirements.entries[kTaskContactCompact]
        ) ||
        !makeRequirement<float>(
            "native task default configuration",
            nativeTask ? model.defaultQ.size() : 0u,
            requirements.entries[kTaskDefaultQ]
        ) ||
        !makeRequirement<MRTaskProgramHeaderGPU>(
            "compiled task header",
            nativeTask ? 1u : 0u,
            requirements.entries[kTaskProgramHeader]
        ) ||
        !makeRequirement<std::uint8_t>(
            "compiled task arena",
            taskProgram.arena().size(),
            requirements.entries[kTaskProgramArena]
        ) ||
        !makeRequirement<MRPolicyProgramHeaderGPU>(
            "compiled policy header",
            nativePolicy ? 1u : 0u,
            requirements.entries[kPolicyProgramHeader]
        ) ||
        !makeRequirement<std::uint8_t>(
            "compiled policy arena",
            policyProgram.arena().size(),
            requirements.entries[kPolicyProgramArena]
        ) ||
        !makeRequirement<float>(
            "native policy scratch A",
            policyScratchElements,
            requirements.entries[kPolicyScratchA]
        ) ||
        !makeRequirement<float>(
            "native policy scratch B",
            policyScratchElements,
            requirements.entries[kPolicyScratchB]
        ) ||
        !makeRequirement<float>(
            "native policy actor mean",
            policyActorMeanElements,
            requirements.entries[kPolicyActorMean]
        ) ||
        !makeRequirement<float>(
            "native policy Gaussian samples",
            layout.policyLatentElements,
            requirements.entries[kPolicyLatents]
        ) ||
        !makeRequirement<float>(
            "native policy log probabilities",
            layout.policyLogProbabilityElements,
            requirements.entries[kPolicyLogProbabilities]
        ) ||
        !makeRequirement<float>(
            "native policy values",
            layout.policyValueElements,
            requirements.entries[kPolicyValues]
        )) {
        return false;
    }

    totalRequiredBytes = 0u;
    for (const BufferRequirement& requirement :
         requirements.entries) {
        if (!checkedAdd(
                totalRequiredBytes,
                requirement.allocationBytes,
                totalRequiredBytes
            )) {
            return false;
        }
    }
    return true;
}

MetalWorldDiagnostics validateAndBuildLayout(
    const CompiledWorld& world,
    const MetalWorldBatch& batch,
    const MetalWorldStepConfig& config,
    const bool residentContinuation,
    RequiredBuffers& requirements
) {
    MetalWorldDiagnostics diagnostics{};
    if (!world.valid()) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidModel,
            "CompiledWorld is empty or invalid"
        );
    }
    if (config.solverMode !=
            MetalWorldSolverMode::freeMotionABA &&
        config.solverMode !=
            MetalWorldSolverMode::temporalCone &&
        config.solverMode !=
            MetalWorldSolverMode::qualityNewton) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::unsupportedSolverMode,
            "unknown MetalWorld solver mode"
        );
    }
    if (config.actuationMode !=
            MetalWorldActuationMode::effort &&
        config.actuationMode !=
            MetalWorldActuationMode::implicitPositionDrive) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidDimensions,
            "unknown MetalWorld actuation mode"
        );
    }
    const bool nativeTask = config.taskProgram.valid();
    const bool nativePolicy = config.policyProgram.valid();
    const bool hasBodyWrenches = config.multicopterProgram.valid() ||
        config.flappingWingProgram.valid() ||
        (nativeTask && taskHasActuatorKind(
            config.taskProgram,
            MR_TASK_ACTUATOR_BODY_WRENCH));
    if (config.multicopterProgram.valid()) {
        const auto& multicopter = config.multicopterProgram;
        if (!nativeTask ||
            multicopter.articulationIndex >= world.articulationCount() ||
            multicopter.bodyIndex >= world.model().bodies.size() ||
            world.model().bodies[multicopter.bodyIndex].articulationIndex !=
                multicopter.articulationIndex ||
            config.taskProgram.layout().actionCount <
                multicopter.firstAction + 4u ||
            !std::isfinite(multicopter.model.motorAndTimestep.w) ||
            std::abs(
                multicopter.model.motorAndTimestep.w -
                config.timestepSeconds /
                    static_cast<float>(config.physicsSubsteps)
            ) > 1.0e-8f) {
            return reject(
                std::move(diagnostics),
                MetalWorldHostStatus::invalidDimensions,
                "compiled multicopter program does not match the task, body, articulation, or physics cadence"
            );
        }
    }
    if (config.flappingWingProgram.valid()) {
        const auto& wings = config.flappingWingProgram;
        const EngineModel& model = world.model();
        if (wings.articulationIndex >= world.articulationCount() ||
            wings.rootBodyIndex >= model.bodies.size() ||
            model.bodies[wings.rootBodyIndex].articulationIndex !=
                wings.articulationIndex ||
            !std::isfinite(wings.windVelocityAndDensity.x) ||
            !std::isfinite(wings.windVelocityAndDensity.y) ||
            !std::isfinite(wings.windVelocityAndDensity.z) ||
            !std::isfinite(wings.windVelocityAndDensity.w) ||
            !(wings.windVelocityAndDensity.w > 0.0f)) {
            return reject(
                std::move(diagnostics),
                MetalWorldHostStatus::invalidDimensions,
                "compiled flapping-wing program does not match the selected articulation or atmosphere"
            );
        }
        for (const MRFlappingWingGPU& wing : wings.wings) {
            if (wing.bodyIndex >= model.bodies.size() ||
                model.bodies[wing.bodyIndex].articulationIndex !=
                    wings.articulationIndex ||
                wing.qIndex >= model.world.nq || wing.vIndex >= model.world.nv ||
                !(wing.rootToCenterAndArea.w > 0.0f) ||
                !(wing.hingeAxisAndChord.w > 0.0f) ||
                !(wing.coefficients.x > 0.0f) ||
                wing.coefficients.y < 0.0f || wing.coefficients.z < 0.0f ||
                !(wing.coefficients.w > 0.0f)) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::invalidDimensions,
                    "compiled flapping-wing geometry is invalid"
                );
            }
        }
    }
    const bool contactMode =
        config.solverMode != MetalWorldSolverMode::freeMotionABA;
    const bool qualityMode =
        config.solverMode == MetalWorldSolverMode::qualityNewton;
    if (config.deviceObservationProgram.configured() &&
        (!config.deviceObservationProgram.valid() ||
         !nativeTask || !nativePolicy)) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidDimensions,
            "device observation program requires a complete callback, native task, and native policy"
        );
    }
    if (!std::isfinite(config.timestepSeconds) ||
        !(config.timestepSeconds > 0.0f) ||
        config.physicsSubsteps == 0u ||
        config.physicsSubsteps >
            MR_METAL_WORLD_MAX_PHYSICS_SUBSTEPS ||
        (contactMode &&
         (config.velocityIterations == 0u ||
          config.velocityIterations > 128u ||
          config.finalVelocityIterations > 128u)) ||
        !std::isfinite(config.manifoldBreakingSeparation) ||
        !std::isfinite(config.manifoldBreakingTangential) ||
        !std::isfinite(config.manifoldMergeDistance) ||
        !std::isfinite(config.manifoldNormalCosine) ||
        !std::isfinite(config.ccdMinimumAdvance) ||
        !std::isfinite(config.ccdTimeTolerance) ||
        !std::isfinite(config.ccdSimultaneousTolerance) ||
        !std::isfinite(config.speculativeMarginScale) ||
        !std::isfinite(config.ccdSpeedEnvelope) ||
        config.manifoldBreakingSeparation < 0.0f ||
        config.manifoldBreakingTangential < 0.0f ||
        config.manifoldMergeDistance < 0.0f ||
        config.manifoldNormalCosine < -1.0f ||
        config.manifoldNormalCosine > 1.0f ||
        config.ccdMinimumAdvance <= 0.0f ||
        config.ccdTimeTolerance <= 0.0f ||
        config.ccdSimultaneousTolerance <= 0.0f ||
        config.speculativeMarginScale < 0.0f ||
        config.ccdSpeedEnvelope <= 0.0f ||
        config.maxCCDEvents == 0u ||
        config.maxCCDEvents > MR_CCD_MAX_EVENTS ||
        config.maxCCDAdvanceSolvePasses == 0u ||
        config.maxCCDAdvanceSolvePasses >
            MR_CCD_MAX_ADVANCE_SOLVE_PASSES ||
        config.maxCCDZeroTimeReplays >
            MR_CCD_MAX_ZERO_TIME_REPLAYS ||
        config.maxConservativeAdvancementIterations == 0u ||
        config.maxConservativeAdvancementIterations > 128u ||
        config.rodContactOuterIterations == 0u ||
        config.rodContactOuterIterations > 4u ||
        (qualityMode &&
         (
             config.quality.maximumNewtonIterations == 0u ||
             config.quality.maximumPCGIterations == 0u ||
             config.quality.maximumLineSearchIterations == 0u ||
             config.quality
                     .directMaximumGeneralizedVelocities ==
                 0u ||
             config.quality.directMaximumRows == 0u ||
             !std::isfinite(
                 config.quality.optimalityTolerance
             ) ||
             config.quality.optimalityTolerance <= 0.0f ||
             !std::isfinite(
                 config.quality.feasibilityTolerance
             ) ||
             config.quality.feasibilityTolerance <= 0.0f ||
             !std::isfinite(config.quality.armijoConstant) ||
             config.quality.armijoConstant <= 0.0f ||
             config.quality.armijoConstant >= 0.5f ||
             !std::isfinite(
                 config.quality.lineSearchContraction
             ) ||
             config.quality.lineSearchContraction <= 0.0f ||
             config.quality.lineSearchContraction >= 1.0f ||
             !std::isfinite(
                 config.quality.complianceFloorMultiplier
             ) ||
             config.quality.complianceFloorMultiplier <= 0.0f
         )) ||
        (config.ccdMode != MetalWorldCCDMode::disabled &&
         config.ccdMode != MetalWorldCCDMode::speculative &&
         config.ccdMode != MetalWorldCCDMode::hybrid)) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidDimensions,
            "timestepSeconds must be finite and positive and "
            "physicsSubsteps, solver iterations, or manifold "
            "thresholds are outside the supported range"
        );
    }
    if (nativeTask &&
        (
            !contactMode ||
            config.actuationMode !=
                MetalWorldActuationMode::
                    implicitPositionDrive ||
            world.rodCount() != 0u ||
            config.taskProgram.worldFingerprint() !=
                world.fingerprint()
        )) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::unsupportedTopology,
            "native locomotion requires an implicit-drive contact world "
            "matching the compiled task fingerprint"
        );
    }
    if (nativePolicy &&
        (
            !nativeTask ||
            config.policyProgram.taskFingerprint() !=
                config.taskProgram.fingerprint() ||
            config.policyProgram.layout()
                    .actorObservationCount !=
                config.taskProgram.layout()
                    .actorObservationSize ||
            (config.policyProgram.layout()
                     .criticLayerCount != 0u &&
             config.policyProgram.layout()
                     .criticObservationCount !=
                 config.taskProgram.layout()
                     .criticObservationSize) ||
            config.policyProgram.layout().actionCount !=
                config.taskProgram.layout().actionCount ||
            (batch.policyRevision != 0u &&
             batch.policyRevision !=
                 config.policyProgram.revision())
        )) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidDimensions,
            "compiled policy does not match the task observation/action contract or revision"
        );
    }
    if (qualityMode &&
        config.ccdMode == MetalWorldCCDMode::hybrid) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::unsupportedSolverMode,
            "qualityNewton currently requires disabled or speculative "
            "CCD; event-time quality re-solves are not silently routed "
            "through the temporal cone solver"
        );
    }
    const std::size_t qualityNv =
        static_cast<std::size_t>(world.nv()) +
        6u * world.sceneBodyCount() +
        3u * world.rodNodeCount() +
        world.rodEdgeCount();
    if (qualityMode &&
        qualityNv >
            MR_UNIFIED_QUALITY_MAX_GENERALIZED_VELOCITIES) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::unsupportedTopology,
            "qualityNewton island exceeds the compiled generalized "
            "velocity bucket"
        );
    }
    const float substepTimestep =
        config.timestepSeconds /
        static_cast<float>(config.physicsSubsteps);
    if (!std::isfinite(substepTimestep) ||
        substepTimestep <
            std::numeric_limits<float>::min()) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidDimensions,
            "derived physics substep timestep is not a positive "
            "normal FP32 value"
        );
    }
    if (batch.environmentCount == 0u ||
        batch.controlStepCount == 0u) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidDimensions,
            "environmentCount and controlStepCount must be nonzero"
        );
    }
    if (batch.environmentCount >
            std::numeric_limits<mr_u32>::max() ||
        batch.controlStepCount >
            std::numeric_limits<mr_u32>::max()) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::arithmeticOverflow,
            "batch dimensions do not fit the GPU ABI"
        );
    }

    MetalWorldLayout layout{};
    MRMetalWorldDispatchGPU& dispatch = layout.dispatch;
    dispatch.abiVersion = MR_METAL_WORLD_ABI_VERSION;
    dispatch.articulationIndex = world.articulationIndex();
    dispatch.environmentCount =
        static_cast<mr_u32>(batch.environmentCount);
    dispatch.controlStepCount =
        static_cast<mr_u32>(batch.controlStepCount);
    dispatch.physicsSubsteps = config.physicsSubsteps;
    dispatch.flags = (
        contactMode
            ? static_cast<mr_u32>(MR_METAL_WORLD_CONTACTS)
            : static_cast<mr_u32>(
                  MR_METAL_WORLD_FREE_MOTION_ONLY
              )
        ) |
        (config.applyBodyDamping
             ? static_cast<mr_u32>(
                   MR_METAL_WORLD_APPLY_BODY_DAMPING
               )
             : 0u) |
        (config.deterministic
             ? static_cast<mr_u32>(
                   MR_METAL_WORLD_DETERMINISTIC
               )
             : 0u) |
        (!batch.resetMasks.empty()
             ? static_cast<mr_u32>(
                   MR_METAL_WORLD_HAS_RESETS
               )
             : 0u) |
        (config.actuationMode ==
                 MetalWorldActuationMode::
                     implicitPositionDrive
             ? static_cast<mr_u32>(
                   MR_METAL_WORLD_IMPLICIT_POSITION_DRIVES
               )
             : 0u) |
        (nativeTask
             ? static_cast<mr_u32>(
                   MR_METAL_WORLD_NATIVE_TASK
               )
             : 0u) |
        (hasBodyWrenches
             ? static_cast<mr_u32>(
                   MR_METAL_WORLD_HAS_BODY_WRENCHES
               )
             : 0u);
    dispatch.nq = world.model().world.nq;
    dispatch.nv = world.model().world.nv;
    dispatch.qStride = world.model().world.nq;
    dispatch.vStride = world.model().world.nv;
    dispatch.effortEnvironmentStride = world.model().world.nv;
    const std::size_t observationEnvironmentStride =
        static_cast<std::size_t>(dispatch.nq) +
        dispatch.nv +
        (contactMode
             ? 13u * static_cast<std::size_t>(
                   world.sceneBodyCount()
               )
             : 0u);
    if (observationEnvironmentStride >
        std::numeric_limits<mr_u32>::max()) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::arithmeticOverflow,
            "observation environment stride exceeds the GPU ABI"
        );
    }
    dispatch.observationEnvironmentStride =
        static_cast<mr_u32>(observationEnvironmentStride);

    std::size_t effortStepStride = 0u;
    std::size_t resetMaskStepStride = 0u;
    std::size_t observationStepStride = 0u;
    std::size_t accelerationStepStride = 0u;
    if (!checkedMultiply(
            batch.environmentCount,
            dispatch.effortEnvironmentStride,
            effortStepStride
        ) ||
        !checkedMultiply(
            batch.environmentCount,
            dispatch.observationEnvironmentStride,
            observationStepStride
        ) ||
        !checkedMultiply(
            batch.environmentCount,
            dispatch.nv,
            accelerationStepStride
        )) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::arithmeticOverflow,
            "derived Metal world step-stride overflow"
        );
    }
    resetMaskStepStride = batch.resetMasks.empty()
        ? 0u
        : batch.environmentCount;
    const auto fitsU32 = [](const std::size_t value) {
        return value <= std::numeric_limits<mr_u32>::max();
    };
    if (!fitsU32(effortStepStride) ||
        !fitsU32(resetMaskStepStride) ||
        !fitsU32(observationStepStride) ||
        !fitsU32(accelerationStepStride)) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::arithmeticOverflow,
            "derived Metal world stride does not fit 32-bit ABI"
        );
    }
    dispatch.effortStepStride =
        static_cast<mr_u32>(effortStepStride);
    dispatch.resetMaskStepStride =
        static_cast<mr_u32>(resetMaskStepStride);
    dispatch.observationStepStride =
        static_cast<mr_u32>(observationStepStride);
    dispatch.accelerationStepStride =
        static_cast<mr_u32>(accelerationStepStride);

    MRABADispatchGPU& aba = layout.abaDispatch;
    aba.articulationIndex = world.articulationIndex();
    aba.environmentCount = dispatch.environmentCount;
    aba.flags =
        (config.applyBodyDamping
             ? static_cast<mr_u32>(
                   MR_ABA_APPLY_BODY_DAMPING
               )
             : 0u) |
        (config.actuationMode ==
                 MetalWorldActuationMode::
                     implicitPositionDrive
             ? static_cast<mr_u32>(
                   MR_ABA_IMPLICIT_DRIVES
               )
             : 0u) |
        (hasBodyWrenches
             ? static_cast<mr_u32>(MR_ABA_HAS_BODY_WRENCHES)
             : 0u);
    aba.qStride = dispatch.qStride;
    aba.vStride = dispatch.vStride;
    aba.effortStride = dispatch.effortEnvironmentStride;
    aba.wrenchStride = hasBodyWrenches
        ? static_cast<mr_u32>(world.model().bodies.size())
        : 0u;
    aba.accelerationStride = dispatch.vStride;
    aba.nextVStride = dispatch.vStride;
    aba.nextQStride = dispatch.qStride;
    layout.abaDispatches.reserve(world.articulationCount());
    layout.kinematicsDispatches.reserve(
        world.articulationCount()
    );
    layout.factorDispatches.reserve(
        world.articulationCount()
    );
    for (mr_u32 owner = 0u;
         owner < world.articulationCount();
         ++owner) {
        const MRArticulationGPU& owned =
            world.model().articulations[owner];
        MRMultiABADispatchGPU work{};
        work.dispatch = aba;
        work.dispatch.articulationIndex = owner;
        work.qBase = owned.qOffset;
        work.vBase = owned.vOffset;
        work.effortBase = owned.vOffset;
        work.wrenchBase = hasBodyWrenches
            ? owned.firstBody
            : 0u;
        work.accelerationBase = owned.vOffset;
        work.nextVBase = owned.vOffset;
        work.nextQBase = owned.qOffset;
        work.statusBase =
            owner * dispatch.environmentCount;
        layout.abaDispatches.push_back(work);

        MRArticulatedOperatorDispatchGPU kinematics{};
        kinematics.articulationIndex = owner;
        kinematics.environmentCount =
            dispatch.environmentCount;
        kinematics.flags =
            MR_ARTICULATED_OPERATOR_KINEMATICS_ONLY;
        kinematics.qStride = dispatch.qStride;
        kinematics.bodyPoseStride =
            static_cast<mr_u32>(world.bodyCount());
        kinematics.generalizedStride = dispatch.vStride;
        layout.kinematicsDispatches.push_back(kinematics);
    }

    MRMetalWorldContactDispatchGPU& contact =
        layout.contactDispatch;
    contact.abiVersion = MR_METAL_WORLD_CONTACT_ABI_VERSION;
    contact.environmentCount = dispatch.environmentCount;
    contact.articulationIndex = world.articulationIndex();
    contact.solverType =
        config.solverMode == MetalWorldSolverMode::qualityNewton
        ? MR_SOLVER_QUALITY_NEWTON
        : MR_SOLVER_TEMPORAL_CONE;
    contact.bodyCount =
        static_cast<mr_u32>(world.model().bodies.size());
    contact.sceneBodyCount = world.sceneBodyCount();
    contact.shapeCount = world.colliderCount();
    contact.eligiblePairCount = world.eligiblePairCount();
    contact.pairCapacity = world.capacities().candidatePairs;
    contact.rawContactCapacity = world.capacities().rawContacts;
    contact.manifoldCapacity = world.capacities().manifolds;
    contact.constraintCapacity =
        world.capacities().constraintBlocks;
    contact.rowCapacity = world.capacities().constraintRows;
    contact.islandCapacity = world.capacities().islands;
    contact.sceneBodyStride = world.sceneBodyCount();
    contact.bodyStateStride = contact.bodyCount;
    contact.pairStride = contact.pairCapacity;
    contact.rawContactStride = contact.rawContactCapacity;
    contact.manifoldStride = contact.manifoldCapacity;
    contact.constraintStride = contact.constraintCapacity;
    contact.rowStride = contact.rowCapacity;
    contact.islandStride = contact.islandCapacity;
    contact.pointQueryStride =
        2u * contact.constraintCapacity;
    contact.factorStride = dispatch.nv * dispatch.nv;
    contact.nv = dispatch.nv;
    contact.flags =
        (config.deterministic
             ? static_cast<mr_u32>(
                   MR_METAL_WORLD_CONTACT_DETERMINISTIC
               )
             : 0u) |
        (config.warmStart
             ? static_cast<mr_u32>(
                   MR_METAL_WORLD_CONTACT_WARM_START
               )
             : 0u) |
        (config.captureContactEvidence
             ? static_cast<mr_u32>(
                   MR_METAL_WORLD_CONTACT_CAPTURE_EVIDENCE
               )
             : 0u) |
        (!batch.kinematicTargets.empty()
             ? static_cast<mr_u32>(
                   MR_METAL_WORLD_CONTACT_HAS_KINEMATIC_TARGETS
               )
             : 0u) |
        (config.solverMode == MetalWorldSolverMode::qualityNewton
             ? static_cast<mr_u32>(
                   MR_METAL_WORLD_CONTACT_QUALITY
               )
             : 0u);
    contact.velocityIterations = config.velocityIterations;
    contact.finalVelocityIterations =
        config.finalVelocityIterations;
    contact.hardConvexCapacity =
        world.capacities().hardConvexPairs;
    contact.meshCandidateCapacity =
        world.capacities().meshTriangleCandidates;
    contact.solverTileCapacity =
        world.capacities().solverTiles;
    contact.spillRowCapacity = world.capacities().spillRows;
    contact.ccdCandidateCapacity =
        world.capacities().ccdCandidates;
    contact.ccdEventCapacity = world.capacities().ccdEvents;
    contact.ccdMode = static_cast<mr_u32>(config.ccdMode);
    contact.maxCCDEvents = config.maxCCDEvents;
    contact.maxConservativeAdvancementIterations =
        config.maxConservativeAdvancementIterations;
    contact.workQueueClassCount = MR_WORLD_WORK_CLASS_COUNT;
    contact.queueStride = std::max(
        contact.pairCapacity,
        std::max(
            contact.islandCapacity,
            contact.solverTileCapacity
        )
    );
    contact.convexCacheStride = static_cast<mr_u32>(
        std::count_if(
            world.eligiblePairs().begin(),
            world.eligiblePairs().end(),
            [](const MRCompiledCollisionPairGPU& pair) {
                return
                    pair.pairClass == MR_COLLISION_PAIR_CONVEX ||
                    pair.pairClass == MR_COLLISION_PAIR_MESH;
            }
        )
    );
    contact.maxCCDAdvanceSolvePasses =
        config.maxCCDAdvanceSolvePasses;
    contact.maxCCDZeroTimeReplays =
        config.maxCCDZeroTimeReplays;
    contact.waveWorkerGroupCount = 64u;
    contact.rodToolPairCount =
        static_cast<mr_u32>(world.rodToolPairs().size());
    contact.articulationCount = world.articulationCount();
    contact.dynamicNodeCount =
        static_cast<mr_u32>(world.dynamicNodes().size());
    contact.islandNodeReferenceCapacity =
        world.capacities().islandNodeReferences;
    contact.islandConstraintReferenceCapacity =
        world.capacities().islandConstraintReferences;
    contact.rodCount = world.rodCount();
    contact.rodNodeCount = world.rodNodeCount();
    contact.rodEdgeCount = world.rodEdgeCount();
    contact.operatorVelocityCapacity =
        world.capacities().operatorVelocityElements;
    contact.nq = dispatch.nq;
    contact.qStride = dispatch.qStride;
    contact.vStride = dispatch.vStride;
    contact.rodContactOuterIterations =
        config.rodContactOuterIterations;
    contact.authoredConstraintCount =
        static_cast<mr_u32>(
            world.model().constraintProgram.blocks.size()
        );
    contact.authoredEndpointCount =
        static_cast<mr_u32>(
            world.model().constraintProgram.endpoints.size()
        );
    contact.authoredRowCount =
        static_cast<mr_u32>(
            world.model().constraintProgram.rows.size()
        );
    contact.authoredConeCount =
        static_cast<mr_u32>(
            world.model().constraintProgram.cones.size()
        );
    if (config.matrixFreeArticulatedContact &&
        config.streamedArticulatedContactResponses &&
        nativeTask &&
        config.solverMode == MetalWorldSolverMode::temporalCone &&
        world.articulationCount() == 1u &&
        contact.rodNodeCount == 0u &&
        contact.authoredConstraintCount == 0u) {
        contact.flags |=
            MR_METAL_WORLD_CONTACT_STREAMED_RESPONSES;
    }
    MRInverseMassDispatchGPU& inverse =
        layout.inverseMassDispatch;
    inverse.articulationIndex = contact.articulationIndex;
    inverse.environmentCount = contact.environmentCount;
    inverse.rhsCount = 3u * contact.constraintStride;
    inverse.flags =
        config.actuationMode ==
            MetalWorldActuationMode::implicitPositionDrive
        ? MR_INVERSE_MASS_IMPLICIT_DRIVES
        : 0u;
    inverse.qStride = contact.qStride;
    inverse.rhsEnvironmentStride =
        inverse.rhsCount * contact.nv;
    inverse.rhsVectorStride = contact.nv;
    inverse.outputEnvironmentStride =
        inverse.rhsEnvironmentStride;
    inverse.outputVectorStride = contact.nv;
    contact.flags |= MR_METAL_WORLD_CONTACT_WAVE32;
    if (nativeTask) {
        contact.flags |=
            MR_METAL_WORLD_CONTACT_BODY_PARAMETERS;
    }
    if (config.ccdMode != MetalWorldCCDMode::disabled) {
        contact.flags |= MR_METAL_WORLD_CONTACT_CCD;
    }
    if (config.ccdMode == MetalWorldCCDMode::hybrid &&
        std::any_of(
            world.model().shapes.begin(),
            world.model().shapes.end(),
            [](const MRShapeGPU& shape) {
                return
                    (shape.flags & MR_SHAPE_FLAG_ENABLE_CCD) !=
                    0u;
            }
        )) {
        contact.flags |=
            MR_METAL_WORLD_CONTACT_HAS_FUTURE_KINEMATICS;
    }
    contact.timestepAndBias = {
        substepTimestep,
        world.model().world.solverScales.w,
        world.model().world.solverScales.z,
        config.warmStart ? 1.0f : 0.0f,
    };
    contact.manifoldThresholds = {
        config.manifoldBreakingSeparation,
        config.manifoldBreakingTangential,
        config.manifoldMergeDistance,
        config.manifoldNormalCosine,
    };
    contact.ccdParameters = {
        config.ccdMinimumAdvance,
        config.ccdTimeTolerance,
        config.speculativeMarginScale,
        config.ccdSpeedEnvelope,
    };
    contact.ccdEventParameters = {
        config.ccdSimultaneousTolerance,
        config.ccdTimeTolerance,
        0.0f,
        0.0f,
    };

    for (mr_u32 owner = 0u;
         owner < world.articulationCount();
         ++owner) {
        const MRArticulationGPU& owned =
            world.model().articulations[owner];
        MRArticulatedOperatorDispatchGPU factor{};
        factor.articulationIndex = owner;
        factor.environmentCount = dispatch.environmentCount;
        factor.pointCount = contact.pointQueryStride;
        factor.flags =
            (contact.flags &
             MR_METAL_WORLD_CONTACT_STREAMED_RESPONSES) != 0u
            ? MR_ARTICULATED_OPERATOR_KINEMATICS_JACOBIANS_ONLY
            : MR_ARTICULATED_OPERATOR_WRITE_CHOLESKY_FACTOR |
                  (
                      (dispatch.flags &
                       MR_METAL_WORLD_IMPLICIT_POSITION_DRIVES) != 0u
                      ? MR_ARTICULATED_OPERATOR_IMPLICIT_DRIVES
                      : 0u
                  );
        factor.qStride = dispatch.qStride;
        factor.pointStride = contact.pointQueryStride;
        factor.bodyPoseStride =
            static_cast<mr_u32>(world.bodyCount());
        factor.pointWorldStride =
            contact.pointQueryStride;
        factor.massMatrixStride = owned.nv * owned.nv;
        factor.pointJacobianStride =
            contact.pointQueryStride * 3u * owned.nv;
        factor.generalizedStride = dispatch.vStride;
        layout.factorDispatches.push_back(factor);
    }

    MRUnifiedQualityDispatchGPU& quality =
        layout.qualityDispatch;
    if (config.solverMode == MetalWorldSolverMode::qualityNewton) {
        const mr_u32 generalizedVelocityCount =
            contact.nv +
            6u * contact.sceneBodyCount +
            3u * contact.rodNodeCount +
            contact.rodEdgeCount;
        const mr_u32 requestedQualityBlocks =
            world.capacities().qualityRows / 3u;
        const mr_u32 qualityBlockCount = std::min(
            contact.constraintStride,
            std::min<mr_u32>(
                MR_UNIFIED_QUALITY_MAX_BLOCKS,
                requestedQualityBlocks == 0u
                ? MR_UNIFIED_QUALITY_MAX_BLOCKS
                : requestedQualityBlocks
            )
        );
        const mr_u32 qualityRowCount =
            3u * qualityBlockCount;
        quality.abiVersion = MR_UNIFIED_QUALITY_ABI_VERSION;
        quality.problemCount = dispatch.environmentCount;
        quality.generalizedVelocityCount =
            generalizedVelocityCount;
        quality.rowCount = qualityRowCount;
        quality.blockCount = qualityBlockCount;
        quality.dynamicsStride =
            generalizedVelocityCount *
            generalizedVelocityCount;
        quality.jacobianStride =
            qualityRowCount * generalizedVelocityCount;
        quality.vectorStride = std::max(
            generalizedVelocityCount,
            qualityRowCount
        );
        quality.maximumNewtonIterations =
            config.quality.maximumNewtonIterations;
        quality.maximumPCGIterations =
            config.quality.maximumPCGIterations;
        quality.maximumLineSearchIterations =
            config.quality.maximumLineSearchIterations;
        quality.directMaximumGeneralizedVelocities =
            config.quality.directMaximumGeneralizedVelocities;
        quality.directMaximumRows =
            config.quality.directMaximumRows;
        quality.derivativeStride =
            qualityBlockCount * 36u;
        quality.hessianStride =
            generalizedVelocityCount <=
                    config.quality
                        .directMaximumGeneralizedVelocities &&
                qualityRowCount <=
                    config.quality.directMaximumRows
            ? generalizedVelocityCount *
                  generalizedVelocityCount
            : 1u;
        quality.blockStride = qualityBlockCount;
        quality.tolerances = {
            config.quality.optimalityTolerance,
            config.quality.feasibilityTolerance,
            config.quality.armijoConstant,
            config.quality.lineSearchContraction,
        };
        quality.numerics = {
            config.quality.complianceFloorMultiplier,
            1.0e-10f,
            1.0e-20f,
            64.0f,
        };
    }

    if (!checkedMultiply(
            batch.environmentCount,
            dispatch.nq,
            layout.initialQElements
        ) ||
        !checkedMultiply(
            batch.environmentCount,
            dispatch.nv,
            layout.initialVElements
        ) ||
        !checkedMultiply(
            batch.controlStepCount,
            effortStepStride,
            layout.effortElements
        ) ||
        !checkedMultiply(
            batch.controlStepCount,
            resetMaskStepStride,
            layout.resetMaskElements
        ) ||
        !checkedMultiply(
            batch.controlStepCount,
            observationStepStride,
            layout.observationElements
        ) ||
        !checkedMultiply(
            batch.controlStepCount,
            accelerationStepStride,
            layout.accelerationElements
        ) ||
        !checkedMultiply(
            batch.controlStepCount,
            batch.environmentCount,
            layout.statusElements
        ) ||
        !checkedMultiply(
            batch.environmentCount,
            world.articulationCount(),
            layout.articulationStatusElements
        )) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::arithmeticOverflow,
            "derived Metal world element-count overflow"
        );
    }
    if (nativeTask) {
        const TaskProgramLayout& taskLayout =
            config.taskProgram.layout();
        const bool finalPolicyEvaluation =
            nativePolicy && config.evaluateFinalPolicy;
        std::size_t transitionCount = 0u;
        std::size_t actorEvaluationCount = 0u;
        std::size_t criticObservationCount = 0u;
        std::size_t policyScratchElements = 0u;
        if (!checkedMultiply(
                batch.controlStepCount,
                batch.environmentCount,
                transitionCount
            ) ||
            !checkedAdd(
                transitionCount,
                finalPolicyEvaluation
                    ? batch.environmentCount
                    : 0u,
                actorEvaluationCount
            ) ||
            !checkedAdd(
                transitionCount,
                nativePolicy &&
                        !config.policyProgram.criticLayers().empty()
                    ? batch.environmentCount
                    : 0u,
                criticObservationCount
            ) ||
            !checkedMultiply(
                actorEvaluationCount,
                taskLayout.actionCount,
                layout.actionElements
            ) ||
            !checkedMultiply(
                actorEvaluationCount,
                taskLayout.actorObservationSize,
                layout.actorObservationElements
            ) ||
            !checkedMultiply(
                criticObservationCount,
                taskLayout.criticObservationSize,
                layout.criticObservationElements
            ) ||
            !checkedMultiply(
                nativePolicy
                    ? batch.environmentCount
                    : 0u,
                config.policyProgram.layout()
                    .maximumHiddenCount,
                policyScratchElements
            ) ||
            !checkedMultiply(
                nativePolicy ? actorEvaluationCount : 0u,
                taskLayout.actionCount,
                layout.policyLatentElements
            ) ||
            !checkedMultiply(
                transitionCount,
                taskLayout.motionFeatureCount,
                layout.motionFeatureElements
            ) ||
            !checkedMultiply(
                taskLayout.interactionFrameCount != 0u
                    ? transitionCount
                    : 0u,
                taskLayout.actionCount,
                layout.teacherActionElements
            )) {
            return reject(
                std::move(diagnostics),
                MetalWorldHostStatus::arithmeticOverflow,
                "native task rollout element-count overflow"
            );
        }
        if (policyScratchElements >
            kShaderAddressableElements) {
            return reject(
                std::move(diagnostics),
                MetalWorldHostStatus::arithmeticOverflow,
                "native policy scratch exceeds the shader's 32-bit addressing contract"
            );
        }
        layout.transitionElements = transitionCount;
        layout.policyLogProbabilityElements =
            nativePolicy ? actorEvaluationCount : 0u;
        layout.policyValueElements =
            nativePolicy ? actorEvaluationCount : 0u;
    }
    layout.resetQElements = batch.resetMasks.empty()
        ? 0u
        : layout.initialQElements;
    layout.resetVElements = batch.resetMasks.empty()
        ? 0u
        : layout.initialVElements;
    if (!checkedMultiply(
            batch.environmentCount,
            world.rodNodeCount(),
            layout.rodNodeStateElements
        ) ||
        !checkedMultiply(
            batch.environmentCount,
            world.rodEdgeCount(),
            layout.rodEdgeStateElements
        )) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::arithmeticOverflow,
            "derived rod-state element-count overflow"
        );
    }
    layout.resetRodNodeStateElements =
        batch.resetMasks.empty()
        ? 0u
        : layout.rodNodeStateElements;
    layout.resetRodEdgeStateElements =
        batch.resetMasks.empty()
        ? 0u
        : layout.rodEdgeStateElements;
    layout.rodBendStateElements =
        static_cast<std::size_t>(world.rodEdgeCount()) -
        world.rodCount();
    if (!checkedMultiply(
            batch.environmentCount,
            world.rodCount(),
            layout.rodStatusElements
        )) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::arithmeticOverflow,
            "derived rod-status element-count overflow"
        );
    }
    layout.dynamicNodeElements = world.dynamicNodes().size();
    if (contactMode) {
        if (!checkedMultiply(
                batch.environmentCount,
                world.sceneBodyCount(),
                layout.initialSceneBodyElements
            ) ||
            !checkedMultiply(
                batch.controlStepCount,
                batch.environmentCount,
                layout.contactStatusElements
            ) ||
            !checkedMultiply(
                batch.environmentCount,
                contact.manifoldStride,
                layout.manifoldHeaderElements
            ) ||
            !checkedMultiply(
                layout.manifoldHeaderElements,
                MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY,
                layout.manifoldPointElements
            ) ||
            !checkedMultiply(
                batch.environmentCount,
                contact.constraintStride,
                layout.contactConstraintElements
            ) ||
            !checkedMultiply(
                batch.environmentCount,
                contact.rowStride,
                layout.constraintRowElements
            ) ||
            !checkedMultiply(
                batch.environmentCount,
                contact.islandStride,
                layout.islandElements
            ) ||
            !checkedMultiply(
                batch.environmentCount,
                contact.islandNodeReferenceCapacity,
                layout.islandNodeReferenceElements
            ) ||
            !checkedMultiply(
                batch.environmentCount,
                contact.islandConstraintReferenceCapacity,
                layout.islandConstraintReferenceElements
            ) ||
            !checkedMultiply(
                batch.environmentCount,
                world.rodCount(),
                layout.rodFactorCacheElements
            ) ||
            !checkedMultiply(
                batch.environmentCount,
                contact.operatorVelocityCapacity,
                layout.operatorVelocityElements
            ) ||
            !checkedMultiply(
                batch.environmentCount,
                contact.pairCapacity,
                layout.pairWorkElements
            ) ||
            !checkedMultiply(
                batch.environmentCount,
                static_cast<std::size_t>(
                    contact.pairCapacity
                ) * MR_METAL_WORLD_RAW_CONTACTS_PER_PAIR,
                layout.pairRawStagingElements
            ) ||
            !checkedMultiply(
                batch.environmentCount,
                contact.islandCapacity,
                layout.islandWorkElements
            ) ||
            !checkedMultiply(
                batch.environmentCount,
                contact.solverTileCapacity,
                layout.contactTileElements
            ) ||
            !checkedMultiply(
                batch.environmentCount,
                contact.convexCacheStride,
                layout.convexCacheElements
            ) ||
            !checkedMultiply(
                batch.environmentCount,
                contact.ccdCandidateCapacity,
                layout.ccdPairElements
            ) ||
            !checkedMultiply(
                batch.environmentCount,
                1u,
                layout.ccdEventStateElements
            ) ||
            !checkedMultiply(
                batch.environmentCount,
                1u,
                layout.ccdImpactClusterElements
            ) ||
            !checkedMultiply(
                batch.environmentCount,
                contact.islandCapacity,
                layout.waveWorkPacketElements
            )) {
            return reject(
                std::move(diagnostics),
                MetalWorldHostStatus::arithmeticOverflow,
                "derived contact-world element-count overflow"
            );
        }
        layout.resetSceneBodyElements =
            batch.resetMasks.empty()
            ? 0u
            : layout.initialSceneBodyElements;
        layout.workQueueHeaderElements =
            MR_WORLD_WORK_CLASS_COUNT;
        layout.manifoldScatterElements =
            batch.environmentCount *
            static_cast<std::size_t>(
                contact.eligiblePairCount
            );
        layout.endpointRuntimeElements =
            2u * layout.contactConstraintElements;
        if (!batch.kinematicTargets.empty() &&
            !checkedMultiply(
                batch.controlStepCount,
                layout.initialSceneBodyElements,
                layout.kinematicTargetElements
            )) {
            return reject(
                std::move(diagnostics),
                MetalWorldHostStatus::arithmeticOverflow,
                "kinematic-target element-count overflow"
            );
        }
    }

    const std::array shaderElementCounts{
        world.model().articulations.size(),
        world.model().joints.size(),
        world.model().dofs.size(),
        world.model().bodies.size(),
        layout.initialQElements,
        layout.initialVElements,
        layout.effortElements,
        layout.actionElements,
        layout.resetMaskElements,
        layout.resetQElements,
        layout.resetVElements,
        layout.observationElements,
        layout.actorObservationElements,
        layout.criticObservationElements,
        layout.transitionElements,
        layout.motionFeatureElements,
        layout.teacherActionElements,
        layout.policyLatentElements,
        layout.policyLogProbabilityElements,
        layout.policyValueElements,
        layout.accelerationElements,
        layout.statusElements,
        layout.initialSceneBodyElements,
        layout.resetSceneBodyElements,
        layout.kinematicTargetElements,
        layout.contactStatusElements,
        layout.manifoldHeaderElements,
        layout.manifoldPointElements,
        layout.contactConstraintElements,
        layout.constraintRowElements,
        layout.islandElements,
        layout.workQueueHeaderElements,
        layout.pairWorkElements,
        layout.pairRawStagingElements,
        layout.islandWorkElements,
        layout.contactTileElements,
        layout.convexCacheElements,
        layout.ccdPairElements,
        layout.ccdEventStateElements,
        layout.ccdImpactClusterElements,
        layout.waveWorkPacketElements,
        layout.rodNodeStateElements,
        layout.rodEdgeStateElements,
        layout.resetRodNodeStateElements,
        layout.resetRodEdgeStateElements,
        layout.dynamicNodeElements,
        layout.islandNodeReferenceElements,
        layout.islandConstraintReferenceElements,
        layout.rodFactorCacheElements,
        layout.operatorVelocityElements,
        layout.rodBendStateElements,
        layout.rodStatusElements,
    };
    if (std::any_of(
            shaderElementCounts.begin(),
            shaderElementCounts.end(),
            [](const std::size_t count) {
                return static_cast<std::uint64_t>(count) >
                    kShaderAddressableElements;
            }
        )) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::arithmeticOverflow,
            "compact buffer exceeds the shader's 32-bit "
            "element-addressing contract"
        );
    }

    std::size_t totalRequiredBytes = 0u;
    if (!buildRequirements(
            world,
            layout,
            config.taskProgram,
            config.policyProgram,
            config.multicopterProgram,
            config.flappingWingProgram,
            requirements,
            totalRequiredBytes
        )) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::arithmeticOverflow,
            "required persistent Metal arena byte-count overflow"
        );
    }
    layout.totalRequiredBytes = totalRequiredBytes;
    for (std::size_t index = 0u;
         index < kRawBufferCount;
         ++index) {
        const std::size_t bytes =
            requirements.entries[index].allocationBytes;
        if (privateImmutableBuffer(index)) {
            layout.memoryPlan.immutablePrivateBytes += bytes;
            // Standalone uploads are explicit, retained shared staging
            // boundaries; normal steps never map the private destination.
            layout.memoryPlan.sharedBoundaryBytes += bytes;
        } else if (privatePersistentBuffer(index)) {
            layout.memoryPlan.persistentStatePrivateBytes += bytes;
        } else if (privateTransientBuffer(index)) {
            layout.memoryPlan.transientPrivateBytes += bytes;
            layout.memoryPlan.peakAliasedBytes += bytes;
        } else {
            layout.memoryPlan.sharedBoundaryBytes += bytes;
        }
    }
    diagnostics.layout = layout;

    const std::size_t initialQElements =
        residentContinuation ? 0u : layout.initialQElements;
    const std::size_t initialVElements =
        residentContinuation ? 0u : layout.initialVElements;
    const std::size_t initialSceneBodyElements =
        residentContinuation
        ? 0u
        : layout.initialSceneBodyElements;
    const std::size_t expectedEffortElements =
        nativeTask ? 0u : layout.effortElements;
    const std::size_t expectedActionElements =
        nativeTask && !nativePolicy
        ? layout.actionElements
        : 0u;
    if (batch.initialQ.size() != initialQElements ||
        batch.initialV.size() != initialVElements ||
        batch.efforts.size() != expectedEffortElements ||
        batch.actions.size() != expectedActionElements ||
        batch.initialSceneBodies.size() !=
            initialSceneBodyElements ||
        (residentContinuation &&
         (!batch.initialRodNodes.empty() ||
          !batch.initialRodEdges.empty() ||
          !batch.resetRodNodes.empty() ||
          !batch.resetRodEdges.empty())) ||
        batch.kinematicTargets.size() !=
            layout.kinematicTargetElements) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidDimensions,
            "initial state, effort, or kinematic trajectory has the wrong "
            "packed element count"
        );
    }
    if (!contactMode &&
        (!batch.initialSceneBodies.empty() ||
         !batch.resetSceneBodies.empty() ||
         !batch.kinematicTargets.empty())) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidDimensions,
            "scene-body inputs require a contact solver mode"
        );
    }
    if (nativeTask && batch.resetMasks.empty()) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidReset,
            "native task requires a writable reset-mask stream"
        );
    }
    if (batch.resetMasks.empty()) {
        if (!batch.resetQ.empty() ||
            !batch.resetV.empty() ||
            !batch.resetSceneBodies.empty() ||
            !batch.resetRodNodes.empty() ||
            !batch.resetRodEdges.empty()) {
            return reject(
                std::move(diagnostics),
                MetalWorldHostStatus::invalidReset,
                "reset states require a control-step reset mask"
            );
        }
    } else {
        const std::size_t resetQElements =
            residentContinuation || nativeTask
            ? 0u
            : layout.resetQElements;
        const std::size_t resetVElements =
            residentContinuation || nativeTask
            ? 0u
            : layout.resetVElements;
        const std::size_t resetSceneBodyElements =
            residentContinuation || nativeTask
            ? 0u
            : layout.resetSceneBodyElements;
        if (batch.resetMasks.size() !=
                layout.resetMaskElements ||
            batch.resetQ.size() != resetQElements ||
            batch.resetV.size() != resetVElements ||
            batch.resetSceneBodies.size() !=
                resetSceneBodyElements) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidReset,
            "reset mask or environment reset state has the "
            "wrong packed element count"
        );
        }
    }
    if (!std::all_of(
            batch.resetMasks.begin(),
            batch.resetMasks.end(),
            [](const std::uint32_t value) {
                return value <= 1u;
            }
        )) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidReset,
            "reset masks must contain only zero or one"
        );
    }
    if ((!residentContinuation &&
         !validQ(
             world.model(),
             batch.environmentCount,
             batch.initialQ
         )) ||
        (!residentContinuation &&
         !finiteFloats(batch.initialV)) ||
        !finiteFloats(batch.efforts) ||
        !finiteFloats(batch.actions)) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::nonfiniteInput,
            "initial state or effort contains a non-finite value "
            "or invalid floating-root quaternion"
        );
    }
    if (!residentContinuation &&
        !nativeTask &&
        !batch.resetMasks.empty() &&
        (!validQ(
             world.model(),
             batch.environmentCount,
             batch.resetQ
         ) ||
         !finiteFloats(batch.resetV))) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidReset,
            "reset state is non-finite or has an invalid "
            "floating-root quaternion"
        );
    }
    if (!residentContinuation &&
        !validRodStates(
            world,
            batch.environmentCount,
            batch.initialRodNodes,
            batch.initialRodEdges,
            true
        )) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::nonfiniteInput,
            "initial rod state is non-finite or has the wrong "
            "environment-major packing"
        );
    }
    if (!residentContinuation &&
        !batch.resetMasks.empty() &&
        !validRodStates(
            world,
            batch.environmentCount,
            batch.resetRodNodes,
            batch.resetRodEdges,
            true
        )) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidReset,
            "reset rod state is non-finite or has the wrong "
            "environment-major packing"
        );
    }
    if (!residentContinuation &&
        contactMode &&
        (!validSceneStates(
             world,
             batch.environmentCount,
             batch.initialSceneBodies
         ) ||
         (!nativeTask &&
          !batch.resetMasks.empty() &&
          !validSceneStates(
              world,
              batch.environmentCount,
              batch.resetSceneBodies
          )))) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::nonfiniteInput,
            "scene-body state is non-finite, has an invalid "
            "quaternion, or does not match compiled body identity"
        );
    }
    if (contactMode && !batch.kinematicTargets.empty()) {
        const std::size_t stateCount =
            layout.initialSceneBodyElements;
        for (std::size_t controlStep = 0u;
             controlStep < batch.controlStepCount;
             ++controlStep) {
            if (!validSceneStates(
                    world,
                    batch.environmentCount,
                    batch.kinematicTargets.subspan(
                        controlStep * stateCount,
                        stateCount
                    )
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::nonfiniteInput,
                    "kinematic target state is invalid"
                );
            }
        }
    }
    return diagnostics;
}

NSString* bufferLabel(const std::size_t index) {
    switch (index) {
    case kWorld:
        return @"MetalWorld runtime world";
    case kArticulations:
        return @"MetalWorld articulations";
    case kJoints:
        return @"MetalWorld joints";
    case kDofs:
        return @"MetalWorld DoF properties";
    case kActuatorProfiles:
        return @"MetalWorld actuator profiles";
    case kBodies:
        return @"MetalWorld body properties";
    case kParallelScheduleArticulations:
        return @"MetalWorld parallel ABA schedule articulations";
    case kParallelScheduleLevels:
        return @"MetalWorld parallel ABA schedule levels";
    case kParallelScheduleParentReductions:
        return @"MetalWorld parallel ABA parent reductions";
    case kParallelScheduleLevelBodies:
        return @"MetalWorld parallel ABA level bodies";
    case kParallelScheduleParentLocal:
        return @"MetalWorld parallel ABA parent indices";
    case kParallelScheduleInboundJoint:
        return @"MetalWorld parallel ABA inbound joints";
    case kParallelScheduleChildOffsets:
        return @"MetalWorld parallel ABA child offsets";
    case kParallelScheduleChildIndices:
        return @"MetalWorld parallel ABA child indices";
    case kABADispatch:
        return @"MetalWorld ABA dispatch";
    case kStateQA:
        return @"MetalWorld state q A";
    case kStateVA:
        return @"MetalWorld state v A";
    case kWorkingEffort:
        return @"MetalWorld working effort";
    case kBodyWrenchPlaceholder:
        return @"MetalWorld external body wrenches";
    case kMulticopterRotors:
        return @"MetalWorld compiled multicopter rotors";
    case kMulticopterModel:
        return @"MetalWorld compiled multicopter model";
    case kMulticopterMixer:
        return @"MetalWorld compiled multicopter mixer";
    case kMulticopterStateA:
        return @"MetalWorld resident multicopter motor state A";
    case kMulticopterStateB:
        return @"MetalWorld resident multicopter motor state B";
    case kMulticopterCandidateState:
        return @"MetalWorld candidate multicopter motor state";
    case kMulticopterDispatch:
        return @"MetalWorld compiled multicopter dispatch";
    case kFlappingWingSpecs:
        return @"MetalWorld compiled flapping-wing geometry";
    case kFlappingWingDispatch:
        return @"MetalWorld compiled flapping-wing dispatch";
    case kCandidateAcceleration:
        return @"MetalWorld candidate acceleration";
    case kCandidateV:
        return @"MetalWorld candidate v";
    case kCandidateQ:
        return @"MetalWorld candidate q";
    case kABAStatuses:
        return @"MetalWorld ABA statuses";
    case kStateQB:
        return @"MetalWorld state q B";
    case kStateVB:
        return @"MetalWorld state v B";
    case kEffortTrajectory:
        return @"MetalWorld effort trajectory";
    case kResetMasks:
        return @"MetalWorld reset masks";
    case kResetQ:
        return @"MetalWorld reset q";
    case kResetV:
        return @"MetalWorld reset v";
    case kObservations:
        return @"MetalWorld observations";
    case kAccelerationTrajectory:
        return @"MetalWorld acceleration trajectory";
    case kPublicStatuses:
        return @"MetalWorld public statuses";
    case kWorldDispatch:
        return @"MetalWorld dispatch";
    case kEnvironmentStatuses:
        return @"MetalWorld environment statuses";
    case kCheckpointQ:
        return @"MetalWorld checkpoint q";
    case kCheckpointV:
        return @"MetalWorld checkpoint v";
    case kTaskDispatch:
        return @"MetalWorld task dispatch";
    case kTaskActions:
        return @"MetalWorld normalized task actions";
    case kTaskState:
        return @"MetalWorld resident task state";
    case kTaskEvidenceState:
        return @"MetalWorld resident task evidence state";
    case kTaskActionHistory:
        return @"MetalWorld resident action history";
    case kTaskActorHistory:
        return @"MetalWorld resident actor history";
    case kTaskCleanHistory:
        return @"MetalWorld resident clean history";
    case kTaskCriticHistory:
        return @"MetalWorld resident critic history";
    case kTaskPreviousJointVelocity:
        return @"MetalWorld resident previous action velocity";
    case kTaskEncoderBias:
        return @"MetalWorld resident task sensor bias";
    case kTaskBodyParameters:
        return @"MetalWorld resident body parameters";
    case kTaskControllerParameters:
        return @"MetalWorld resident controller parameters";
    case kTaskActorObservations:
        return @"MetalWorld actor observations";
    case kTaskCriticObservations:
        return @"MetalWorld critic observations";
    case kTaskTransitions:
        return @"MetalWorld task transitions";
    case kTaskMotionFeatures:
        return @"MetalWorld motion-prior features";
    case kTaskTeacherActions:
        return @"MetalWorld physically realized imagination actions";
    case kTaskContactCompact:
        return @"MetalWorld resident compact contact metrics";
    case kTaskDefaultQ:
        return @"MetalWorld task default configuration";
    case kTaskProgramHeader:
        return @"MetalWorld compiled task header";
    case kTaskProgramArena:
        return @"MetalWorld compiled task arena";
    case kPolicyProgramHeader:
        return @"MetalWorld compiled policy header";
    case kPolicyProgramArena:
        return @"MetalWorld compiled policy arena";
    case kPolicyScratchA:
        return @"MetalWorld policy scratch A";
    case kPolicyScratchB:
        return @"MetalWorld policy scratch B";
    case kPolicyActorMean:
        return @"MetalWorld policy actor mean";
    case kPolicyLatents:
        return @"MetalWorld policy Gaussian samples";
    case kPolicyLogProbabilities:
        return @"MetalWorld policy log probabilities";
    case kPolicyValues:
        return @"MetalWorld policy values";
    default:
        return @"MetalWorld buffer";
    }
}

id<MTLComputePipelineState> makePipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* functionName,
    NSError** error
) {
    id<MTLFunction> function =
        [library newFunctionWithName:functionName];
    if (function == nil) {
        if (error != nullptr) {
            *error = [NSError
                errorWithDomain:@"MetalRobo.MetalWorld"
                           code:1
                       userInfo:@{
                           NSLocalizedDescriptionKey:
                               [NSString
                                   stringWithFormat:
                                       @"metallib does not contain %@",
                                       functionName]
                       }];
        }
        return nil;
    }
    return [device
        newComputePipelineStateWithFunction:function
                                      error:error];
}

MetalWorldDiagnostics initializeContext(
    detail::MetalWorldContextState& context,
    MetalWorldDiagnostics diagnostics
) {
    if (context.initialized) {
        diagnostics.deviceName = nsString(context.device.name);
        diagnostics.thermalState = thermalStateName(
            [NSProcessInfo processInfo].thermalState
        );
        return diagnostics;
    }

    std::string metallibPath = context.config.metallibPath;
    if (metallibPath.empty()) {
        metallibPath = defaultMetallibPath();
    }
    if (metallibPath.empty()) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metallibUnavailable,
            "no MetalWorld metallib path is available"
        );
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalDeviceUnavailable,
            "no Metal-capable device is available"
        );
    }
    diagnostics.deviceName = nsString(device.name);
    diagnostics.thermalState = thermalStateName(
        [NSProcessInfo processInfo].thermalState
    );
    if (!device.hasUnifiedMemory) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalDeviceUnsupported,
            "MetalWorld requires unified-memory Metal"
        );
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalDeviceUnavailable,
            "failed to create MetalWorld command queue"
        );
    }
    queue.label = @"MetalRobo persistent world queue";

    NSString* path = [NSString
        stringWithUTF8String:metallibPath.c_str()];
    if (path == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metallibUnavailable,
            "metallib path is not valid UTF-8"
        );
    }
    NSError* error = nil;
    id<MTLLibrary> library = [device
        newLibraryWithURL:[NSURL fileURLWithPath:path]
                    error:&error];
    if (library == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalLibraryFailure,
            "failed to load MetalWorld metallib: " +
                describeError(error)
        );
    }

    error = nil;
    id<MTLComputePipelineState> aba = makePipeline(
        device,
        library,
        @"mr_articulated_aba_step",
        &error
    );
    if (aba == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalPipelineFailure,
            "failed to create ABA pipeline: " +
                describeError(error)
        );
    }
    error = nil;
    id<MTLComputePipelineState> parameterizedABA = makePipeline(
        device,
        library,
        @"mr_parameterized_articulated_aba_step",
        &error
    );
    if (parameterizedABA == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalPipelineFailure,
            "failed to create parameterized ABA pipeline: " +
                describeError(error)
        );
    }
    error = nil;
    id<MTLComputePipelineState> smallABA = makePipeline(
        device,
        library,
        @"mr_articulated_aba_step_small",
        &error
    );
    if (smallABA == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalPipelineFailure,
            "failed to create small-capacity ABA pipeline: " +
                describeError(error)
        );
    }
    error = nil;
    id<MTLComputePipelineState> multiABA = makePipeline(
        device,
        library,
        @"mr_multi_articulated_aba_step",
        &error
    );
    if (multiABA == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalPipelineFailure,
            "failed to create multi-articulation ABA pipeline: " +
                describeError(error)
        );
    }
    error = nil;
    id<MTLComputePipelineState> prepare = makePipeline(
        device,
        library,
        @"mr_metal_world_prepare",
        &error
    );
    if (prepare == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalPipelineFailure,
            "failed to create MetalWorld prepare pipeline: " +
                describeError(error)
        );
    }
    error = nil;
    id<MTLComputePipelineState> driveRefresh = makePipeline(
        device,
        library,
        @"mr_metal_world_refresh_drives",
        &error
    );
    if (driveRefresh == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalPipelineFailure,
            "failed to create MetalWorld drive-refresh pipeline: " +
                describeError(error)
        );
    }
    error = nil;
    id<MTLComputePipelineState> commit = makePipeline(
        device,
        library,
        @"mr_metal_world_commit",
        &error
    );
    if (commit == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalPipelineFailure,
            "failed to create MetalWorld commit pipeline: " +
                describeError(error)
        );
    }
    error = nil;
    id<MTLComputePipelineState> capture = makePipeline(
        device,
        library,
        @"mr_metal_world_capture",
        &error
    );
    if (capture == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalPipelineFailure,
            "failed to create MetalWorld capture pipeline: " +
                describeError(error)
        );
    }

    __strong id<MTLComputePipelineState> operatorPipeline = nil;
    __strong id<MTLComputePipelineState>
        parameterizedOperatorPipeline = nil;
    __strong id<MTLComputePipelineState> taskObserve = nil;
    __strong id<MTLComputePipelineState> taskThreatSelect = nil;
    __strong id<MTLComputePipelineState> taskJointCbf = nil;
    __strong id<MTLComputePipelineState> taskMotion = nil;
    __strong id<MTLComputePipelineState> taskApply = nil;
    __strong id<MTLComputePipelineState> taskNativeActuator = nil;
    __strong id<MTLComputePipelineState> taskEffort = nil;
    __strong id<MTLComputePipelineState> taskImpactContact = nil;
    __strong id<MTLComputePipelineState> taskComplete = nil;
    __strong id<MTLComputePipelineState> taskEvidence = nil;
    __strong id<MTLComputePipelineState> multicopter = nil;
    __strong id<MTLComputePipelineState> multicopterCommit = nil;
    __strong id<MTLComputePipelineState> flappingWing = nil;
    __strong id<MTLComputePipelineState> policyDense = nil;
    __strong id<MTLComputePipelineState> policySample = nil;
    __strong id<MTLComputePipelineState> contactPrepare = nil;
    __strong id<MTLComputePipelineState> observationStateSelect = nil;
    __strong id<MTLComputePipelineState> bodyProjection = nil;
    __strong id<MTLComputePipelineState> scenePrediction = nil;
    __strong id<MTLComputePipelineState> colliderProjection = nil;
    __strong id<MTLComputePipelineState> sweptProjection = nil;
    __strong id<MTLComputePipelineState> ccd = nil;
    __strong id<MTLComputePipelineState> ccdEventInitialize = nil;
    __strong id<MTLComputePipelineState> ccdEventPrepare = nil;
    __strong id<MTLComputePipelineState> ccdEventSelect = nil;
    __strong id<MTLComputePipelineState> ccdEventFinalize = nil;
    __strong id<MTLComputePipelineState> eventArticulation = nil;
    __strong id<MTLComputePipelineState> eventScenePrediction = nil;
    __strong id<MTLComputePipelineState> eventBodyOverlay = nil;
    __strong id<MTLComputePipelineState> jointLimits = nil;
    __strong id<MTLComputePipelineState> eventColliderProjection = nil;
    __strong id<MTLComputePipelineState> inactiveEventRestore = nil;
    __strong id<MTLComputePipelineState> eventSegmentPublish = nil;
    __strong id<MTLComputePipelineState> pairFlags = nil;
    __strong id<MTLComputePipelineState> scanBlocks = nil;
    __strong id<MTLComputePipelineState> scanAdd = nil;
    __strong id<MTLComputePipelineState> pairClassFlags = nil;
    __strong id<MTLComputePipelineState> pairQueueScatter = nil;
    __strong id<MTLComputePipelineState> pairNarrowphase = nil;
    __strong id<MTLComputePipelineState> convexNarrowphase = nil;
    __strong id<MTLComputePipelineState> hullNarrowphase = nil;
    __strong id<MTLComputePipelineState> meshNarrowphase = nil;
    __strong id<MTLComputePipelineState> collisionCompile = nil;
    __strong id<MTLComputePipelineState> manifoldFinalize = nil;
    __strong id<MTLComputePipelineState> manifoldScan = nil;
    __strong id<MTLComputePipelineState> manifoldRecordScatter = nil;
    __strong id<MTLComputePipelineState> manifoldIRScatter = nil;
    __strong id<MTLComputePipelineState> multiQueryInitialize = nil;
    __strong id<MTLComputePipelineState> multiOperatorCompose = nil;
    __strong id<MTLComputePipelineState> factorDispatch = nil;
    __strong id<MTLComputePipelineState> pointQueryTail = nil;
    __strong id<MTLComputePipelineState> streamedInverse = nil;
    __strong id<MTLComputePipelineState> evaluateIR = nil;
    __strong id<MTLComputePipelineState> islands = nil;
    __strong id<MTLComputePipelineState> buildTiles = nil;
    __strong id<MTLComputePipelineState> islandQueueScatter = nil;
    __strong id<MTLComputePipelineState> solverCohort = nil;
    __strong id<MTLComputePipelineState> distributedIslandFlags = nil;
    __strong id<MTLComputePipelineState> distributedIslandScatter = nil;
    __strong id<MTLComputePipelineState> distributedTileFlags = nil;
    __strong id<MTLComputePipelineState> distributedTileScatter = nil;
    __strong id<MTLComputePipelineState> contactSolve = nil;
    __strong id<MTLComputePipelineState> wave32Solve = nil;
    __strong id<MTLComputePipelineState> wave32DistributedPrepare = nil;
    __strong id<MTLComputePipelineState> wave32DistributedDelta = nil;
    __strong id<MTLComputePipelineState> wave32DistributedReduce = nil;
    __strong id<MTLComputePipelineState> wave32Reduce = nil;
    __strong id<MTLComputePipelineState> contactIntegrate = nil;
    __strong id<MTLComputePipelineState> contactLatch = nil;
    __strong id<MTLComputePipelineState> contactCommit = nil;
    __strong id<MTLComputePipelineState> convexCachePublish = nil;
    __strong id<MTLComputePipelineState> contactCapture = nil;
    __strong id<MTLComputePipelineState> qualityPrepare = nil;
    __strong id<MTLComputePipelineState> qualityWarmStart = nil;
    __strong id<MTLComputePipelineState> qualityQueue = nil;
    __strong id<MTLComputePipelineState> qualitySolve = nil;
    __strong id<MTLComputePipelineState> qualityApply = nil;
    __strong id<MTLComputePipelineState> qualityQueueStatus = nil;
    __strong id<MTLComputePipelineState> rodPrepare = nil;
    __strong id<MTLComputePipelineState> rodContactPrepare = nil;
    __strong id<MTLComputePipelineState> rodPack = nil;
    __strong id<MTLComputePipelineState> rodStep = nil;
    __strong id<MTLComputePipelineState> rodFactor = nil;
    __strong id<MTLComputePipelineState> rodUnpack = nil;
    __strong id<MTLComputePipelineState> rodLatch = nil;
    __strong id<MTLComputePipelineState> rodContactLatch = nil;
    __strong id<MTLComputePipelineState> rodToolNarrowphase = nil;
    __strong id<MTLComputePipelineState> rodContactScan = nil;
    __strong id<MTLComputePipelineState> rodContactScatter = nil;
    __strong id<MTLComputePipelineState> rodContactSolve = nil;
    __strong id<MTLComputePipelineState> rodCommit = nil;
    __strong id<MTLComputePipelineState> rodContactCommit = nil;
    __strong id<MTLComputePipelineState> rodEventInitialize = nil;
    __strong id<MTLComputePipelineState> inactiveRodEventRestore = nil;
    __strong id<MTLComputePipelineState> rodEventSegmentPublish = nil;
    __strong id<MTLComputePipelineState> rodSweptProjection = nil;
    __strong id<MTLComputePipelineState> rodCCD = nil;
    __strong id<MTLComputePipelineState> rodCCDWitnessTag = nil;
    __strong id<MTLComputePipelineState> authoredIRSeed = nil;
    __strong id<MTLComputePipelineState>
        generalizedConstraintSolve = nil;
    auto createContactPipeline = [&](
        NSString* functionName
    ) {
        error = nil;
        return makePipeline(
            device,
            library,
            functionName,
            &error
        );
    };
    operatorPipeline =
        createContactPipeline(@"mr_articulated_operator");
    parameterizedOperatorPipeline = createContactPipeline(
        @"mr_parameterized_articulated_operator"
    );
    taskObserve =
        createContactPipeline(@"mr_locomotion_task_observe");
    taskThreatSelect = createContactPipeline(
        @"mr_locomotion_task_select_threat_query"
    );
    taskJointCbf = createContactPipeline(
        @"mr_locomotion_task_joint_cbf_teacher"
    );
    taskMotion = createContactPipeline(
        @"mr_locomotion_task_motion_features"
    );
    taskApply =
        createContactPipeline(@"mr_locomotion_task_apply_actions");
    taskNativeActuator = createContactPipeline(
        @"mr_locomotion_task_apply_native_actuators"
    );
    taskEffort =
        createContactPipeline(@"mr_locomotion_task_measure_effort");
    multicopter = createContactPipeline(
        @"mr_step_compiled_multicopters"
    );
    multicopterCommit = createContactPipeline(
        @"mr_commit_compiled_multicopters"
    );
    flappingWing = createContactPipeline(
        @"mr_step_compiled_flapping_wings"
    );
    taskImpactContact = createContactPipeline(
        @"mr_locomotion_task_latch_impact_contact"
    );
    taskComplete =
        createContactPipeline(@"mr_locomotion_task_complete");
    taskEvidence = createContactPipeline(
        @"mr_locomotion_task_update_evidence"
    );
    policyDense =
        createContactPipeline(@"mr_policy_dense_layer");
    policySample = createContactPipeline(
        @"mr_policy_sample_and_score"
    );
    contactPrepare =
        createContactPipeline(@"mr_world_prepare_contact_step");
    observationStateSelect = createContactPipeline(
        @"mr_world_select_observation_state"
    );
    bodyProjection =
        createContactPipeline(@"mr_world_build_body_states");
    scenePrediction =
        createContactPipeline(@"mr_world_predict_scene");
    colliderProjection =
        createContactPipeline(@"mr_world_project_colliders");
    sweptProjection = createContactPipeline(
        @"mr_world_project_swept_colliders"
    );
    ccd = createContactPipeline(@"mr_world_resolve_ccd");
    ccdEventInitialize = createContactPipeline(
        @"mr_world_initialize_ccd_event_state"
    );
    ccdEventPrepare = createContactPipeline(
        @"mr_world_prepare_ccd_event_pass"
    );
    ccdEventSelect =
        createContactPipeline(@"mr_world_select_ccd_event_state");
    ccdEventFinalize =
        createContactPipeline(@"mr_world_finalize_ccd_event_state");
    eventArticulation = createContactPipeline(
        @"mr_world_materialize_event_articulation"
    );
    eventScenePrediction = createContactPipeline(
        @"mr_world_predict_scene_event"
    );
    eventBodyOverlay = createContactPipeline(
        @"mr_world_overlay_event_articulation_bodies"
    );
    jointLimits = createContactPipeline(
        @"mr_world_project_joint_limits"
    );
    eventColliderProjection = createContactPipeline(
        @"mr_world_project_event_colliders"
    );
    inactiveEventRestore = createContactPipeline(
        @"mr_world_restore_inactive_event_candidate"
    );
    eventSegmentPublish = createContactPipeline(
        @"mr_world_publish_event_segment"
    );
    rodEventInitialize = createContactPipeline(
        @"mr_world_initialize_rod_event_state"
    );
    inactiveRodEventRestore = createContactPipeline(
        @"mr_world_restore_inactive_rod_event_candidate"
    );
    rodEventSegmentPublish = createContactPipeline(
        @"mr_world_publish_rod_event_segment"
    );
    rodSweptProjection = createContactPipeline(
        @"mr_world_project_swept_rod_colliders"
    );
    rodCCD = createContactPipeline(
        @"mr_world_resolve_rod_ccd"
    );
    rodCCDWitnessTag = createContactPipeline(
        @"mr_world_tag_rod_ccd_witnesses"
    );
    pairFlags =
        createContactPipeline(@"mr_world_flag_eligible_pairs");
    scanBlocks =
        createContactPipeline(@"mr_world_scan_blocks");
    scanAdd =
        createContactPipeline(@"mr_world_scan_add_block_offsets");
    pairClassFlags =
        createContactPipeline(@"mr_world_flag_pair_work_class");
    pairQueueScatter =
        createContactPipeline(@"mr_world_scatter_pair_queue");
    pairNarrowphase =
        createContactPipeline(@"mr_world_narrowphase_pair_queue");
    convexNarrowphase =
        createContactPipeline(@"mr_world_narrowphase_convex_queue");
    hullNarrowphase =
        createContactPipeline(@"mr_world_narrowphase_hull_queue");
    meshNarrowphase =
        createContactPipeline(@"mr_world_narrowphase_mesh_queue");
    collisionCompile =
        createContactPipeline(@"mr_world_collide_compile");
    manifoldFinalize = createContactPipeline(
        @"mr_world_finalize_pair_manifold"
    );
    manifoldScan = createContactPipeline(
        @"mr_world_scan_manifold_ir"
    );
    manifoldRecordScatter = createContactPipeline(
        @"mr_world_scatter_manifold_records"
    );
    manifoldIRScatter = createContactPipeline(
        @"mr_world_scatter_manifold_ir"
    );
    multiQueryInitialize = createContactPipeline(
        @"mr_world_initialize_multi_articulation_queries"
    );
    multiOperatorCompose = createContactPipeline(
        @"mr_world_compose_multi_articulation_operator"
    );
    factorDispatch = createContactPipeline(
        @"mr_world_finalize_factor_dispatch"
    );
    pointQueryTail = createContactPipeline(
        @"mr_world_fill_point_query_tail"
    );
    streamedInverse = createContactPipeline(
        @"mr_world_parallel_streaming_articulated_inverse_mass"
    );
    evaluateIR =
        createContactPipeline(@"mr_world_evaluate_constraint_ir");
    islands =
        createContactPipeline(@"mr_world_build_contact_islands");
    buildTiles =
        createContactPipeline(@"mr_world_build_contact_tiles");
    islandQueueScatter =
        createContactPipeline(@"mr_world_scatter_island_queue");
    solverCohort =
        createContactPipeline(@"mr_world_select_solver_cohort");
    distributedIslandFlags = createContactPipeline(
        @"mr_world_flag_distributed_islands"
    );
    distributedIslandScatter = createContactPipeline(
        @"mr_world_scatter_distributed_island_queue"
    );
    distributedTileFlags = createContactPipeline(
        @"mr_world_flag_distributed_tiles"
    );
    distributedTileScatter = createContactPipeline(
        @"mr_world_scatter_distributed_tile_queue"
    );
    contactSolve =
        createContactPipeline(@"mr_world_solve_contact_islands");
    wave32Solve =
        createContactPipeline(@"mr_world_wave32_solve");
    wave32DistributedPrepare = createContactPipeline(
        @"mr_world_wave32_distributed_prepare"
    );
    wave32DistributedDelta = createContactPipeline(
        @"mr_world_wave32_distributed_delta"
    );
    wave32DistributedReduce = createContactPipeline(
        @"mr_world_wave32_distributed_reduce"
    );
    wave32Reduce =
        createContactPipeline(@"mr_world_reduce_wave32_status");
    contactIntegrate =
        createContactPipeline(@"mr_world_integrate_contact_state");
    contactLatch =
        createContactPipeline(@"mr_world_latch_contact_status");
    contactCommit =
        createContactPipeline(@"mr_world_commit_contact_state");
    convexCachePublish =
        createContactPipeline(@"mr_world_publish_convex_cache");
    contactCapture =
        createContactPipeline(@"mr_world_capture_contact");
    qualityPrepare = createContactPipeline(
        @"mr_world_prepare_unified_quality"
    );
    qualityWarmStart = createContactPipeline(
        @"mr_world_reconstruct_unified_quality_warm_start"
    );
    qualityQueue = createContactPipeline(
        @"mr_world_build_unified_quality_queue"
    );
    qualitySolve = createContactPipeline(
        @"mr_unified_quality_solve_queued"
    );
    qualityApply = createContactPipeline(
        @"mr_world_apply_unified_quality"
    );
    qualityQueueStatus = createContactPipeline(
        @"mr_world_publish_unified_quality_queue_status"
    );
    rodPrepare = createContactPipeline(
        @"mr_world_prepare_rod_state"
    );
    rodContactPrepare = createContactPipeline(
        @"mr_world_prepare_rod_contact_cache"
    );
    rodPack = createContactPipeline(
        @"mr_world_pack_rod_state"
    );
    rodStep = createContactPipeline(
        @"mr_discrete_elastic_rod_step"
    );
    rodFactor = createContactPipeline(
        @"mr_world_factor_rod_operator"
    );
    rodUnpack = createContactPipeline(
        @"mr_world_unpack_rod_state"
    );
    rodLatch = createContactPipeline(
        @"mr_world_latch_rod_status"
    );
    rodContactLatch = createContactPipeline(
        @"mr_world_latch_rod_contact_status"
    );
    rodToolNarrowphase = createContactPipeline(
        @"mr_rod_tool_narrowphase"
    );
    rodContactScan = createContactPipeline(
        @"mr_world_scan_rod_contact_ir"
    );
    rodContactScatter = createContactPipeline(
        @"mr_world_scatter_rod_contact_ir"
    );
    rodContactSolve = createContactPipeline(
        @"mr_world_solve_rod_contact_constraints"
    );
    rodCommit = createContactPipeline(
        @"mr_world_commit_rod_state"
    );
    rodContactCommit = createContactPipeline(
        @"mr_world_commit_rod_contact_cache"
    );
    authoredIRSeed = createContactPipeline(
        @"mr_world_seed_authored_constraint_ir"
    );
    generalizedConstraintSolve = createContactPipeline(
        @"mr_world_solve_generalized_constraints"
    );
    if (operatorPipeline == nil ||
        parameterizedOperatorPipeline == nil ||
        taskObserve == nil ||
        taskThreatSelect == nil ||
        taskJointCbf == nil ||
        taskMotion == nil ||
        taskApply == nil ||
        taskNativeActuator == nil ||
        taskEffort == nil ||
        taskImpactContact == nil ||
        taskComplete == nil ||
        taskEvidence == nil ||
        multicopter == nil ||
        multicopterCommit == nil ||
        flappingWing == nil ||
        policyDense == nil ||
        policySample == nil ||
        contactPrepare == nil ||
        observationStateSelect == nil ||
        bodyProjection == nil ||
        scenePrediction == nil ||
        colliderProjection == nil ||
        sweptProjection == nil ||
        ccd == nil ||
        ccdEventInitialize == nil ||
        ccdEventPrepare == nil ||
        ccdEventSelect == nil ||
        ccdEventFinalize == nil ||
        eventArticulation == nil ||
        eventScenePrediction == nil ||
        eventBodyOverlay == nil ||
        jointLimits == nil ||
        eventColliderProjection == nil ||
        inactiveEventRestore == nil ||
        eventSegmentPublish == nil ||
        rodEventInitialize == nil ||
        inactiveRodEventRestore == nil ||
        rodEventSegmentPublish == nil ||
        rodSweptProjection == nil ||
        rodCCD == nil ||
        rodCCDWitnessTag == nil ||
        pairFlags == nil ||
        scanBlocks == nil ||
        scanAdd == nil ||
        pairClassFlags == nil ||
        pairQueueScatter == nil ||
        pairNarrowphase == nil ||
        convexNarrowphase == nil ||
        hullNarrowphase == nil ||
        meshNarrowphase == nil ||
        collisionCompile == nil ||
        manifoldFinalize == nil ||
        manifoldScan == nil ||
        manifoldRecordScatter == nil ||
        manifoldIRScatter == nil ||
        multiQueryInitialize == nil ||
        multiOperatorCompose == nil ||
        factorDispatch == nil ||
        pointQueryTail == nil ||
        streamedInverse == nil ||
        evaluateIR == nil ||
        islands == nil ||
        buildTiles == nil ||
        islandQueueScatter == nil ||
        solverCohort == nil ||
        distributedIslandFlags == nil ||
        distributedIslandScatter == nil ||
        distributedTileFlags == nil ||
        distributedTileScatter == nil ||
        contactSolve == nil ||
        wave32Solve == nil ||
        wave32DistributedPrepare == nil ||
        wave32DistributedDelta == nil ||
        wave32DistributedReduce == nil ||
        wave32Reduce == nil ||
        contactIntegrate == nil ||
        contactLatch == nil ||
        contactCommit == nil ||
        convexCachePublish == nil ||
        contactCapture == nil ||
        qualityPrepare == nil ||
        qualityWarmStart == nil ||
        qualityQueue == nil ||
        qualitySolve == nil ||
        qualityApply == nil ||
        qualityQueueStatus == nil ||
        rodPrepare == nil ||
        rodContactPrepare == nil ||
        rodPack == nil ||
        rodStep == nil ||
        rodUnpack == nil ||
        rodLatch == nil ||
        rodContactLatch == nil ||
        rodToolNarrowphase == nil ||
        rodContactScan == nil ||
        rodContactScatter == nil ||
        rodContactSolve == nil ||
        rodCommit == nil ||
        rodContactCommit == nil ||
        authoredIRSeed == nil ||
        generalizedConstraintSolve == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalPipelineFailure,
            "failed to create device-resident contact pipeline: " +
                describeError(error)
        );
    }

    if (aba.maxTotalThreadsPerThreadgroup <
            kABAThreadsPerThreadgroup ||
        smallABA.maxTotalThreadsPerThreadgroup <
            kABAThreadsPerThreadgroup ||
        multiABA.maxTotalThreadsPerThreadgroup <
            kABAThreadsPerThreadgroup ||
        aba.staticThreadgroupMemoryLength >
            device.maxThreadgroupMemoryLength ||
        smallABA.staticThreadgroupMemoryLength >
            device.maxThreadgroupMemoryLength ||
        multiABA.staticThreadgroupMemoryLength >
            device.maxThreadgroupMemoryLength ||
        operatorPipeline.maxTotalThreadsPerThreadgroup <
            kOperatorThreadsPerThreadgroup ||
        operatorPipeline.staticThreadgroupMemoryLength >
            device.maxThreadgroupMemoryLength ||
        prepare.maxTotalThreadsPerThreadgroup == 0u ||
        commit.maxTotalThreadsPerThreadgroup == 0u ||
        capture.maxTotalThreadsPerThreadgroup == 0u ||
        taskObserve.maxTotalThreadsPerThreadgroup == 0u ||
        taskApply.maxTotalThreadsPerThreadgroup == 0u ||
        taskEffort.maxTotalThreadsPerThreadgroup == 0u ||
        taskComplete.maxTotalThreadsPerThreadgroup == 0u ||
        taskEvidence.maxTotalThreadsPerThreadgroup == 0u ||
        multicopter.maxTotalThreadsPerThreadgroup == 0u ||
        multicopterCommit.maxTotalThreadsPerThreadgroup == 0u ||
        policyDense.maxTotalThreadsPerThreadgroup == 0u ||
        policySample.maxTotalThreadsPerThreadgroup == 0u ||
        contactPrepare.maxTotalThreadsPerThreadgroup == 0u ||
        bodyProjection.maxTotalThreadsPerThreadgroup == 0u ||
        scenePrediction.maxTotalThreadsPerThreadgroup == 0u ||
        colliderProjection.maxTotalThreadsPerThreadgroup == 0u ||
        sweptProjection.maxTotalThreadsPerThreadgroup == 0u ||
        ccd.maxTotalThreadsPerThreadgroup == 0u ||
        ccdEventInitialize.maxTotalThreadsPerThreadgroup == 0u ||
        ccdEventPrepare.maxTotalThreadsPerThreadgroup == 0u ||
        ccdEventSelect.maxTotalThreadsPerThreadgroup == 0u ||
        ccdEventFinalize.maxTotalThreadsPerThreadgroup == 0u ||
        eventArticulation.maxTotalThreadsPerThreadgroup == 0u ||
        eventScenePrediction.maxTotalThreadsPerThreadgroup == 0u ||
        eventBodyOverlay.maxTotalThreadsPerThreadgroup == 0u ||
        jointLimits.maxTotalThreadsPerThreadgroup == 0u ||
        eventColliderProjection.maxTotalThreadsPerThreadgroup == 0u ||
        inactiveEventRestore.maxTotalThreadsPerThreadgroup == 0u ||
        eventSegmentPublish.maxTotalThreadsPerThreadgroup == 0u ||
        pairFlags.maxTotalThreadsPerThreadgroup == 0u ||
        scanBlocks.maxTotalThreadsPerThreadgroup <
            kWorldThreadsPerThreadgroup ||
        scanAdd.maxTotalThreadsPerThreadgroup <
            kWorldThreadsPerThreadgroup ||
        pairClassFlags.maxTotalThreadsPerThreadgroup <
            kWorldThreadsPerThreadgroup ||
        pairQueueScatter.maxTotalThreadsPerThreadgroup <
            kWorldThreadsPerThreadgroup ||
        pairNarrowphase.maxTotalThreadsPerThreadgroup <
            kWorldThreadsPerThreadgroup ||
        convexNarrowphase.maxTotalThreadsPerThreadgroup <
            kWorldThreadsPerThreadgroup ||
        hullNarrowphase.maxTotalThreadsPerThreadgroup <
            kWorldThreadsPerThreadgroup ||
        meshNarrowphase.maxTotalThreadsPerThreadgroup <
            kWorldThreadsPerThreadgroup ||
        collisionCompile.maxTotalThreadsPerThreadgroup == 0u ||
        manifoldFinalize.maxTotalThreadsPerThreadgroup == 0u ||
        manifoldScan.maxTotalThreadsPerThreadgroup <
            MR_SIMD_WIDTH ||
        manifoldRecordScatter.maxTotalThreadsPerThreadgroup == 0u ||
        manifoldIRScatter.maxTotalThreadsPerThreadgroup == 0u ||
        multiQueryInitialize.maxTotalThreadsPerThreadgroup == 0u ||
        multiOperatorCompose.maxTotalThreadsPerThreadgroup == 0u ||
        factorDispatch.maxTotalThreadsPerThreadgroup == 0u ||
        pointQueryTail.maxTotalThreadsPerThreadgroup <
            kWorldThreadsPerThreadgroup ||
        streamedInverse.maxTotalThreadsPerThreadgroup <
            kABAThreadsPerThreadgroup ||
        streamedInverse.staticThreadgroupMemoryLength >
            device.maxThreadgroupMemoryLength ||
        evaluateIR.maxTotalThreadsPerThreadgroup <
            kWorldThreadsPerThreadgroup ||
        islands.maxTotalThreadsPerThreadgroup <
            kWorldThreadsPerThreadgroup ||
        buildTiles.maxTotalThreadsPerThreadgroup <
            kWorldThreadsPerThreadgroup ||
        islandQueueScatter.maxTotalThreadsPerThreadgroup <
            kWorldThreadsPerThreadgroup ||
        solverCohort.maxTotalThreadsPerThreadgroup <
            MR_WAVE32_CONTACTS_PER_TILE ||
        distributedIslandFlags.maxTotalThreadsPerThreadgroup <
            kWorldThreadsPerThreadgroup ||
        distributedIslandScatter.maxTotalThreadsPerThreadgroup <
            kWorldThreadsPerThreadgroup ||
        distributedTileFlags.maxTotalThreadsPerThreadgroup <
            kWorldThreadsPerThreadgroup ||
        distributedTileScatter.maxTotalThreadsPerThreadgroup <
            kWorldThreadsPerThreadgroup ||
        contactSolve.maxTotalThreadsPerThreadgroup <
            kWorldThreadsPerThreadgroup ||
        wave32Solve.maxTotalThreadsPerThreadgroup <
            MR_WAVE32_CONTACTS_PER_TILE ||
        wave32DistributedPrepare.maxTotalThreadsPerThreadgroup <
            MR_WAVE32_CONTACTS_PER_TILE ||
        wave32DistributedDelta.maxTotalThreadsPerThreadgroup <
            MR_WAVE32_CONTACTS_PER_TILE ||
        wave32DistributedReduce.maxTotalThreadsPerThreadgroup <
            MR_WAVE32_CONTACTS_PER_TILE ||
        wave32Reduce.maxTotalThreadsPerThreadgroup <
            kWorldThreadsPerThreadgroup ||
        contactIntegrate.maxTotalThreadsPerThreadgroup == 0u ||
        contactLatch.maxTotalThreadsPerThreadgroup == 0u ||
        contactCommit.maxTotalThreadsPerThreadgroup == 0u ||
        convexCachePublish.maxTotalThreadsPerThreadgroup == 0u ||
        contactCapture.maxTotalThreadsPerThreadgroup == 0u ||
        qualityPrepare.maxTotalThreadsPerThreadgroup <
            MR_SIMD_WIDTH ||
        qualityWarmStart.maxTotalThreadsPerThreadgroup <
            MR_SIMD_WIDTH ||
        qualityQueue.maxTotalThreadsPerThreadgroup <
            MR_SIMD_WIDTH ||
        qualitySolve.maxTotalThreadsPerThreadgroup <
            MR_SIMD_WIDTH ||
        qualityApply.maxTotalThreadsPerThreadgroup == 0u ||
        qualityQueueStatus.maxTotalThreadsPerThreadgroup == 0u ||
        rodPrepare.maxTotalThreadsPerThreadgroup == 0u ||
        rodContactPrepare.maxTotalThreadsPerThreadgroup == 0u ||
        rodPack.maxTotalThreadsPerThreadgroup == 0u ||
        rodStep.maxTotalThreadsPerThreadgroup <
            MR_ROD_GPU_MAX_NODES ||
        rodFactor.maxTotalThreadsPerThreadgroup == 0u ||
        rodUnpack.maxTotalThreadsPerThreadgroup == 0u ||
        rodLatch.maxTotalThreadsPerThreadgroup == 0u ||
        rodContactLatch.maxTotalThreadsPerThreadgroup == 0u ||
        rodToolNarrowphase.maxTotalThreadsPerThreadgroup == 0u ||
        rodContactScan.maxTotalThreadsPerThreadgroup <
            MR_WAVE32_CONTACTS_PER_TILE ||
        rodContactScatter.maxTotalThreadsPerThreadgroup == 0u ||
        rodContactSolve.maxTotalThreadsPerThreadgroup == 0u ||
        rodCommit.maxTotalThreadsPerThreadgroup == 0u ||
        rodContactCommit.maxTotalThreadsPerThreadgroup == 0u ||
        authoredIRSeed.maxTotalThreadsPerThreadgroup == 0u ||
        generalizedConstraintSolve
                .maxTotalThreadsPerThreadgroup == 0u ||
        rodStep.staticThreadgroupMemoryLength >
            device.maxThreadgroupMemoryLength ||
        qualityPrepare.staticThreadgroupMemoryLength >
            device.maxThreadgroupMemoryLength ||
        qualityWarmStart.staticThreadgroupMemoryLength >
            device.maxThreadgroupMemoryLength ||
        qualityQueue.staticThreadgroupMemoryLength >
            device.maxThreadgroupMemoryLength ||
        qualitySolve.staticThreadgroupMemoryLength >
            device.maxThreadgroupMemoryLength) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalDeviceUnsupported,
            "device cannot execute the MetalWorld kernel geometry"
        );
    }
    if (pairNarrowphase.threadExecutionWidth !=
            MR_WAVE32_CONTACTS_PER_TILE ||
        manifoldScan.threadExecutionWidth !=
            MR_WAVE32_CONTACTS_PER_TILE ||
        solverCohort.threadExecutionWidth !=
            MR_WAVE32_CONTACTS_PER_TILE ||
        wave32Solve.threadExecutionWidth !=
            MR_WAVE32_CONTACTS_PER_TILE ||
        wave32DistributedPrepare.threadExecutionWidth !=
            MR_WAVE32_CONTACTS_PER_TILE ||
        wave32DistributedDelta.threadExecutionWidth !=
            MR_WAVE32_CONTACTS_PER_TILE ||
        wave32DistributedReduce.threadExecutionWidth !=
            MR_WAVE32_CONTACTS_PER_TILE ||
        qualityPrepare.threadExecutionWidth !=
            MR_WAVE32_CONTACTS_PER_TILE ||
        qualityWarmStart.threadExecutionWidth !=
            MR_WAVE32_CONTACTS_PER_TILE ||
        qualityQueue.threadExecutionWidth !=
            MR_WAVE32_CONTACTS_PER_TILE ||
        qualitySolve.threadExecutionWidth !=
            MR_WAVE32_CONTACTS_PER_TILE ||
        streamedInverse.threadExecutionWidth !=
            MR_WAVE32_CONTACTS_PER_TILE ||
        rodToolNarrowphase.threadExecutionWidth !=
            MR_WAVE32_CONTACTS_PER_TILE ||
        rodContactScan.threadExecutionWidth !=
            MR_WAVE32_CONTACTS_PER_TILE ||
        rodStep.threadExecutionWidth !=
            MR_WAVE32_CONTACTS_PER_TILE) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalDeviceUnsupported,
            "contact graph requires a queried SIMD execution width of 32"
        );
    }

    context.device = device;
    context.queue = queue;
    context.library = library;
    context.abaPipeline = aba;
    context.parameterizedABAPipeline = parameterizedABA;
    context.smallABAPipeline = smallABA;
    context.multiABAPipeline = multiABA;
    context.preparePipeline = prepare;
    context.driveRefreshPipeline = driveRefresh;
    context.commitPipeline = commit;
    context.capturePipeline = capture;
    context.operatorPipeline = operatorPipeline;
    context.parameterizedOperatorPipeline =
        parameterizedOperatorPipeline;
    context.taskObservePipeline = taskObserve;
    context.taskThreatSelectPipeline = taskThreatSelect;
    context.taskJointCbfPipeline = taskJointCbf;
    context.taskMotionPipeline = taskMotion;
    context.taskApplyPipeline = taskApply;
    context.taskNativeActuatorPipeline = taskNativeActuator;
    context.taskEffortPipeline = taskEffort;
    context.taskImpactContactPipeline = taskImpactContact;
    context.taskCompletePipeline = taskComplete;
    context.taskEvidencePipeline = taskEvidence;
    context.multicopterPipeline = multicopter;
    context.multicopterCommitPipeline = multicopterCommit;
    context.flappingWingPipeline = flappingWing;
    context.policyDensePipeline = policyDense;
    context.policySamplePipeline = policySample;
    context.contactPreparePipeline = contactPrepare;
    context.observationStateSelectPipeline =
        observationStateSelect;
    context.bodyProjectionPipeline = bodyProjection;
    context.scenePredictionPipeline = scenePrediction;
    context.colliderProjectionPipeline = colliderProjection;
    context.sweptProjectionPipeline = sweptProjection;
    context.ccdPipeline = ccd;
    context.ccdEventInitializePipeline = ccdEventInitialize;
    context.ccdEventPreparePipeline = ccdEventPrepare;
    context.ccdEventSelectPipeline = ccdEventSelect;
    context.ccdEventFinalizePipeline = ccdEventFinalize;
    context.eventArticulationPipeline = eventArticulation;
    context.eventScenePredictionPipeline = eventScenePrediction;
    context.eventBodyOverlayPipeline = eventBodyOverlay;
    context.jointLimitPipeline = jointLimits;
    context.eventColliderProjectionPipeline = eventColliderProjection;
    context.inactiveEventRestorePipeline = inactiveEventRestore;
    context.eventSegmentPublishPipeline = eventSegmentPublish;
    context.rodEventInitializePipeline = rodEventInitialize;
    context.inactiveRodEventRestorePipeline =
        inactiveRodEventRestore;
    context.rodEventSegmentPublishPipeline =
        rodEventSegmentPublish;
    context.rodSweptProjectionPipeline = rodSweptProjection;
    context.rodCCDPipeline = rodCCD;
    context.rodCCDWitnessTagPipeline = rodCCDWitnessTag;
    context.pairFlagPipeline = pairFlags;
    context.scanBlocksPipeline = scanBlocks;
    context.scanAddPipeline = scanAdd;
    context.pairClassFlagPipeline = pairClassFlags;
    context.pairQueueScatterPipeline = pairQueueScatter;
    context.pairNarrowphasePipeline = pairNarrowphase;
    context.convexNarrowphasePipeline = convexNarrowphase;
    context.hullNarrowphasePipeline = hullNarrowphase;
    context.meshNarrowphasePipeline = meshNarrowphase;
    context.collisionCompilePipeline = collisionCompile;
    context.manifoldFinalizePipeline = manifoldFinalize;
    context.manifoldScanPipeline = manifoldScan;
    context.manifoldRecordScatterPipeline =
        manifoldRecordScatter;
    context.manifoldIRScatterPipeline = manifoldIRScatter;
    context.multiQueryInitializePipeline =
        multiQueryInitialize;
    context.multiOperatorComposePipeline =
        multiOperatorCompose;
    context.factorDispatchPipeline = factorDispatch;
    context.pointQueryTailPipeline = pointQueryTail;
    context.streamedInversePipeline = streamedInverse;
    context.evaluateIRPipeline = evaluateIR;
    context.islandPipeline = islands;
    context.buildTilesPipeline = buildTiles;
    context.islandQueueScatterPipeline = islandQueueScatter;
    context.solverCohortPipeline = solverCohort;
    context.distributedIslandFlagPipeline =
        distributedIslandFlags;
    context.distributedIslandScatterPipeline =
        distributedIslandScatter;
    context.distributedTileFlagPipeline =
        distributedTileFlags;
    context.distributedTileScatterPipeline =
        distributedTileScatter;
    context.contactSolvePipeline = contactSolve;
    context.wave32SolvePipeline = wave32Solve;
    context.wave32DistributedPreparePipeline =
        wave32DistributedPrepare;
    context.wave32DistributedDeltaPipeline =
        wave32DistributedDelta;
    context.wave32DistributedReducePipeline =
        wave32DistributedReduce;
    context.wave32ReducePipeline = wave32Reduce;
    context.contactIntegratePipeline = contactIntegrate;
    context.contactLatchPipeline = contactLatch;
    context.contactCommitPipeline = contactCommit;
    context.convexCachePublishPipeline = convexCachePublish;
    context.contactCapturePipeline = contactCapture;
    context.qualityPreparePipeline = qualityPrepare;
    context.qualityWarmStartPipeline = qualityWarmStart;
    context.qualityQueuePipeline = qualityQueue;
    context.qualitySolvePipeline = qualitySolve;
    context.qualityApplyPipeline = qualityApply;
    context.qualityQueueStatusPipeline = qualityQueueStatus;
    context.rodPreparePipeline = rodPrepare;
    context.rodContactPreparePipeline = rodContactPrepare;
    context.rodPackPipeline = rodPack;
    context.rodStepPipeline = rodStep;
    context.rodFactorPipeline = rodFactor;
    context.rodUnpackPipeline = rodUnpack;
    context.rodLatchPipeline = rodLatch;
    context.rodContactLatchPipeline = rodContactLatch;
    context.rodToolNarrowphasePipeline = rodToolNarrowphase;
    context.rodContactScanPipeline = rodContactScan;
    context.rodContactScatterPipeline = rodContactScatter;
    context.rodContactSolvePipeline = rodContactSolve;
    context.rodCommitPipeline = rodCommit;
    context.rodContactCommitPipeline = rodContactCommit;
    context.authoredIRSeedPipeline = authoredIRSeed;
    context.generalizedConstraintSolvePipeline =
        generalizedConstraintSolve;
    context.stats.queriedThreadExecutionWidth =
        static_cast<std::uint32_t>(
            pairNarrowphase.threadExecutionWidth
        );
    context.initialized = true;
    context.stats.pipelineCreationCount += 85u;
    return diagnostics;
}

std::size_t growthCapacity(
    const std::size_t current,
    const std::size_t required,
    const std::size_t maximum
) {
    if (current >= required) {
        return current;
    }
    if (current == 0u) {
        return required;
    }
    const std::size_t half = current / 2u;
    const std::size_t grown =
        half <= maximum - current
            ? current + half
            : maximum;
    return std::max(required, grown);
}

bool privateTransientBuffer(const std::size_t index) {
    switch (index) {
    case kWorkingEffort:
    case kBodyWrenchPlaceholder:
    case kMulticopterCandidateState:
    case kCandidateAcceleration:
    case kCandidateV:
    case kCandidateQ:
    case kABAStatuses:
    case kBodyPoses:
    case kFutureBodyPoses:
    case kPointWorld:
    case kFactorMatrix:
    case kPointJacobians:
    case kGeneralizedImpulse:
    case kDeltaVelocity:
    case kOperatorStatuses:
    case kInverseMassStatuses:
    case kCurrentBodies:
    case kCandidateBodies:
    case kCandidateManifoldHeaders:
    case kCandidateManifoldPoints:
    case kCandidateManifoldCounts:
    case kCandidatePairs:
    case kRawContacts:
    case kRawPairIndices:
    case kContacts:
    case kContactMetadata:
    case kIRBlocks:
    case kIREndpoints:
    case kIRRows:
    case kIRCones:
    case kPointQueries:
    case kEvaluatedRows:
    case kEvaluatedCones:
    case kFactorCaches:
    case kIslands:
    case kResponseColumns:
    case kContactStatuses:
    case kActiveIndirectDispatch:
    case kProjectedColliders:
    case kFutureProjectedColliders:
    case kPairOverlapFlags:
    case kWorkQueueHeaders:
    case kPairWorkQueue:
    case kPairRawCounts:
    case kCompactionOffsets:
    case kCompactionScratch:
    case kCompactionFlags:
    case kIslandWorkQueue:
    case kContactTiles:
    case kTileConstraintIndices:
    case kWave32ImpulseDeltas:
    case kWave32IslandStatuses:
    case kCandidateConvexCaches:
    case kCCDPairs:
    case kCCDEventStatesA:
    case kCCDEventStatesB:
    case kCCDImpactClusters:
    case kWaveWorkPackets:
    case kPairRawContactStaging:
    case kPairManifoldHeaders:
    case kPairManifoldPoints:
    case kManifoldIRScatter:
    case kEndpointRuntime:
    case kWave32Preconditioners:
    case kIslandWorkDense:
    case kQualityBlocks:
    case kQualityDynamics:
    case kQualityJacobian:
    case kQualityBias:
    case kQualityFreeVelocity:
    case kQualityWarmVelocity:
    case kQualityWarmImpulses:
    case kQualityOutputVelocity:
    case kQualityOutputImpulses:
    case kQualityDerivatives:
    case kQualityHessian:
    case kQualityWorkQueue:
    case kQualityWorkPackets:
    case kRodFactorCaches:
    case kOperatorVelocityArena:
    case kIslandNodeReferences:
    case kIslandConstraintReferences:
    case kCheckpointRodNodes:
    case kCheckpointRodEdges:
    case kRodInputPositions:
    case kRodInputVelocities:
    case kRodInputTwists:
    case kRodInputTwistRates:
    case kRodOutputPositions:
    case kRodOutputVelocities:
    case kRodOutputTwists:
    case kRodOutputTwistRates:
    case kRodStatuses:
    case kRodAttachments:
    case kRodReactions:
    case kFactorMatrixStaging:
    case kPointJacobiansStaging:
    case kRodWitnessCounts:
    case kCandidateRodWitnesses:
    case kCheckpointRodWitnesses:
    case kRodConstraintWitnessIndices:
    case kRodContactScratch:
    case kCCDEventRodNodesA:
    case kCCDEventRodEdgesA:
    case kCCDEventRodWitnessesA:
    case kCCDEventRodNodesB:
    case kCCDEventRodEdgesB:
    case kCCDEventRodWitnessesB:
    case kProjectedRodColliders:
    case kFutureProjectedRodColliders:
    case kPolicyScratchA:
    case kPolicyScratchB:
    case kPolicyActorMean:
        return true;
    default:
        return false;
    }
}

bool privatePersistentBuffer(const std::size_t index) {
    switch (index) {
    case kStateQA:
    case kStateVA:
    case kStateQB:
    case kStateVB:
    case kResetQ:
    case kResetV:
    case kResetSceneBodies:
    case kSceneBodiesA:
    case kSceneBodiesB:
    case kManifoldHeadersA:
    case kManifoldPointsA:
    case kManifoldCountsA:
    case kManifoldHeadersB:
    case kManifoldPointsB:
    case kManifoldCountsB:
    case kConvexCaches:
    case kResetRodNodes:
    case kResetRodEdges:
    case kRodNodesA:
    case kRodEdgesA:
    case kRodNodesB:
    case kRodEdgesB:
    case kRodWitnessesA:
    case kRodWitnessesB:
    case kTaskState:
    case kTaskEvidenceState:
    case kTaskActionHistory:
    case kTaskActorHistory:
    case kTaskCleanHistory:
    case kTaskCriticHistory:
    case kTaskPreviousJointVelocity:
    case kTaskEncoderBias:
    case kTaskBodyParameters:
    case kTaskControllerParameters:
    case kTaskContactCompact:
    case kMulticopterStateA:
    case kMulticopterStateB:
        return true;
    default:
        return false;
    }
}

bool privatePersistentInputBuffer(const std::size_t index) {
    switch (index) {
    case kStateQA:
    case kStateVA:
    case kResetQ:
    case kResetV:
    case kResetSceneBodies:
    case kSceneBodiesA:
    case kResetRodNodes:
    case kResetRodEdges:
    case kRodNodesA:
    case kRodEdgesA:
    case kTaskEvidenceState:
        return true;
    default:
        return false;
    }
}

bool privateImmutableBuffer(const std::size_t index) {
    return
        (index >= kArticulations && index <= kBodies) ||
        index == kActuatorProfiles ||
        index == kTaskDefaultQ ||
        (index >= kTaskProgramHeader &&
         index <= kTaskProgramArena) ||
        (index >= kPolicyProgramHeader &&
         index <= kPolicyProgramArena) ||
        (index >= kParallelScheduleArticulations &&
         index <= kParallelScheduleChildIndices) ||
        index == kShapes ||
        index == kMaterials ||
        index == kSceneBodyIndices ||
        index == kEligiblePairs ||
        index == kDynamicNodes ||
        index == kBodyDynamicNodes ||
        index == kMulticopterRotors ||
        index == kMulticopterModel ||
        index == kMulticopterMixer ||
        index == kMulticopterDispatch ||
        index == kFlappingWingSpecs ||
        index == kFlappingWingDispatch ||
        index == kRodColliders ||
        index == kRodShapeSources ||
        index == kRodToolPairs ||
        (index >= kAuthoredIRBlocks &&
         index <= kAuthoredIRWarmImpulses) ||
        (index >= kRodRestLengths &&
         index <= kRodTwistStiffness) ||
        (index >= kGeometryHeaders &&
         index <= kMeshTriangles);
}

std::size_t alignUp(
    const std::size_t value,
    const std::size_t alignment
) {
    if (alignment <= 1u) {
        return value;
    }
    return (value + alignment - 1u) / alignment * alignment;
}

MetalWorldDiagnostics ensureBufferArena(
    detail::MetalWorldContextState& context,
    const RequiredBuffers& requirements,
    MetalWorldDiagnostics diagnostics
) {
    const std::size_t maximumBufferLength =
        static_cast<std::size_t>(
            context.device.maxBufferLength
        );
    std::array<std::size_t, kRawBufferCount> proposed =
        context.capacities;
    for (std::size_t index = 0u;
         index < kRawBufferCount;
         ++index) {
        const BufferRequirement& requirement =
            requirements.entries[index];
        if (requirement.allocationBytes >
            maximumBufferLength) {
            return reject(
                std::move(diagnostics),
                MetalWorldHostStatus::metalBufferFailure,
                std::string(requirement.label) +
                    " exceeds device.maxBufferLength"
            );
        }
        proposed[index] = growthCapacity(
            context.capacities[index],
            requirement.allocationBytes,
            maximumBufferLength
        );
    }

    std::size_t projectedBytes = 0u;
    for (const std::size_t capacity : proposed) {
        if (!checkedAdd(
                projectedBytes,
                capacity,
                projectedBytes
            )) {
            return reject(
                std::move(diagnostics),
                MetalWorldHostStatus::arithmeticOverflow,
                "persistent MetalWorld arena byte-count overflow"
            );
        }
    }
    const std::uint64_t recommendedWorkingSet =
        context.device.recommendedMaxWorkingSetSize;
    if (recommendedWorkingSet != 0u &&
        static_cast<std::uint64_t>(projectedBytes) >
            recommendedWorkingSet) {
        projectedBytes = 0u;
        for (std::size_t index = 0u;
             index < kRawBufferCount;
             ++index) {
            proposed[index] = std::max(
                context.capacities[index],
                requirements.entries[index].allocationBytes
            );
            if (!checkedAdd(
                    projectedBytes,
                    proposed[index],
                    projectedBytes
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::arithmeticOverflow,
                    "persistent MetalWorld arena byte-count overflow"
                );
            }
        }
    }
    if (recommendedWorkingSet != 0u &&
        static_cast<std::uint64_t>(projectedBytes) >
            recommendedWorkingSet) {
        std::vector<std::size_t> ranked;
        ranked.reserve(kRawBufferCount);
        for (std::size_t index = 0u;
             index < kRawBufferCount;
             ++index) {
            if (proposed[index] != 0u) {
                ranked.push_back(index);
            }
        }
        std::ranges::sort(
            ranked,
            [&](const std::size_t a, const std::size_t b) {
                return proposed[a] > proposed[b];
            }
        );
        std::string message =
            "persistent MetalWorld arena exceeds "
            "device.recommendedMaxWorkingSetSize: required=" +
            std::to_string(projectedBytes) +
            " recommended=" +
            std::to_string(recommendedWorkingSet) +
            " largest=";
        const std::size_t reported = std::min<std::size_t>(
            ranked.size(),
            6u
        );
        for (std::size_t rank = 0u; rank < reported; ++rank) {
            if (rank != 0u) {
                message += ",";
            }
            const std::size_t index = ranked[rank];
            message += requirements.entries[index].label;
            message += ":";
            message += std::to_string(proposed[index]);
        }
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalBufferFailure,
            std::move(message)
        );
    }

    const bool usePrivateHeap =
        context.config.preferPrivateHeaps;
    const auto needsHeapRebuild = [&](
        const auto predicate,
        id<MTLHeap> heap
    ) {
        if (!usePrivateHeap) {
            return false;
        }
        if (heap == nil) {
            return true;
        }
        for (std::size_t index = 0u;
             index < kRawBufferCount;
             ++index) {
            if (predicate(index) &&
                proposed[index] != context.capacities[index]) {
                return true;
            }
        }
        return false;
    };
    const bool rebuildImmutableHeap = needsHeapRebuild(
        privateImmutableBuffer,
        context.immutableHeap
    );
    const bool rebuildPersistentHeap = needsHeapRebuild(
        privatePersistentBuffer,
        context.persistentHeap
    );
    const bool rebuildTransientHeap = needsHeapRebuild(
        privateTransientBuffer,
        context.transientHeap
    );

    __strong id<MTLBuffer> replacements[kRawBufferCount] = {};
    __strong id<MTLBuffer> uploadReplacements[kRawBufferCount] = {};
    __strong id<MTLHeap> replacementImmutableHeap = nil;
    __strong id<MTLHeap> replacementPersistentHeap = nil;
    __strong id<MTLHeap> replacementTransientHeap = nil;
    std::string privateHeapFailure;
    const MTLResourceOptions privateOptions =
        MTLResourceStorageModePrivate |
        MTLResourceHazardTrackingModeTracked;
    const auto buildPlacementHeap = [&](
        const auto predicate,
        NSString* label,
        id<MTLHeap> __strong& replacement
    ) {
        std::array<NSUInteger, kRawBufferCount> offsets{};
        std::size_t heapBytes = 0u;
        for (std::size_t index = 0u;
             index < kRawBufferCount;
             ++index) {
            if (!predicate(index)) {
                continue;
            }
            const MTLSizeAndAlign sizeAndAlign =
                [context.device
                    heapBufferSizeAndAlignWithLength:
                        static_cast<NSUInteger>(proposed[index])
                                             options:privateOptions];
            const std::size_t aligned = alignUp(
                heapBytes,
                static_cast<std::size_t>(sizeAndAlign.align)
            );
            if (aligned > maximumBufferLength ||
                static_cast<std::size_t>(sizeAndAlign.size) >
                    maximumBufferLength - aligned) {
                privateHeapFailure =
                    "private placement-heap byte-count overflow";
                return false;
            }
            offsets[index] = static_cast<NSUInteger>(aligned);
            heapBytes =
                aligned +
                static_cast<std::size_t>(sizeAndAlign.size);
        }
        MTLHeapDescriptor* descriptor =
            [[MTLHeapDescriptor alloc] init];
        descriptor.type = MTLHeapTypePlacement;
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.hazardTrackingMode =
            MTLHazardTrackingModeTracked;
        descriptor.size = static_cast<NSUInteger>(heapBytes);
        replacement =
            [context.device newHeapWithDescriptor:descriptor];
        if (replacement == nil) {
            privateHeapFailure =
                "could not allocate a private placement heap";
            return false;
        }
        replacement.label = label;
        for (std::size_t index = 0u;
             index < kRawBufferCount;
             ++index) {
            if (!predicate(index)) {
                continue;
            }
            replacements[index] = [
                replacement
                newBufferWithLength:
                    static_cast<NSUInteger>(proposed[index])
                options:privateOptions
                offset:offsets[index]
            ];
            if (replacements[index] == nil ||
                replacements[index].length < proposed[index]) {
                privateHeapFailure =
                    std::string(
                        "private placement allocation failed for "
                    ) + requirements.entries[index].label;
                return false;
            }
            replacements[index].label = bufferLabel(index);
        }
        return true;
    };
    if ((rebuildImmutableHeap &&
         !buildPlacementHeap(
             privateImmutableBuffer,
             @"MetalWorld immutable private placement heap",
             replacementImmutableHeap
         )) ||
        (rebuildPersistentHeap &&
         !buildPlacementHeap(
             privatePersistentBuffer,
             @"MetalWorld persistent-state private placement heap",
             replacementPersistentHeap
         )) ||
        (rebuildTransientHeap &&
         !buildPlacementHeap(
             privateTransientBuffer,
             @"MetalWorld transient private placement heap",
             replacementTransientHeap
         ))) {
        return reject(
            std::move(diagnostics),
            privateHeapFailure.find("overflow") !=
                    std::string::npos
                ? MetalWorldHostStatus::arithmeticOverflow
                : MetalWorldHostStatus::metalBufferFailure,
            std::move(privateHeapFailure)
        );
    }
    if (rebuildImmutableHeap || rebuildPersistentHeap) {
        for (std::size_t index = 0u;
             index < kRawBufferCount;
             ++index) {
            const bool needsImmutableUpload =
                rebuildImmutableHeap &&
                privateImmutableBuffer(index);
            const bool needsPersistentUpload =
                rebuildPersistentHeap &&
                privatePersistentInputBuffer(index);
            if (!needsImmutableUpload &&
                !needsPersistentUpload) {
                continue;
            }
            uploadReplacements[index] = [context.device
                newBufferWithLength:static_cast<NSUInteger>(
                    proposed[index]
                )
                           options:MTLResourceStorageModeShared];
            if (uploadReplacements[index] == nil ||
                uploadReplacements[index].contents == nullptr ||
                uploadReplacements[index].length <
                    proposed[index]) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::metalBufferFailure,
                    std::string(
                        "immutable upload staging allocation failed for "
                    ) + requirements.entries[index].label
                );
            }
        }
    }
    for (std::size_t index = 0u;
         index < kRawBufferCount;
         ++index) {
        if (usePrivateHeap &&
            (
                privateImmutableBuffer(index) ||
                privatePersistentBuffer(index) ||
                privateTransientBuffer(index)
            )) {
            continue;
        }
        if (proposed[index] == context.capacities[index]) {
            continue;
        }
        replacements[index] = [context.device
            newBufferWithLength:static_cast<NSUInteger>(
                proposed[index]
            )
                       options:MTLResourceStorageModeShared];
        if (replacements[index] == nil ||
            replacements[index].contents == nullptr ||
            replacements[index].length < proposed[index]) {
            return reject(
                std::move(diagnostics),
                MetalWorldHostStatus::metalBufferFailure,
                std::string("persistent Metal buffer growth failed for ") +
                    requirements.entries[index].label
            );
        }
        replacements[index].label = bufferLabel(index);
    }

    if (replacementImmutableHeap != nil) {
        context.immutableHeap = replacementImmutableHeap;
    }
    if (replacementPersistentHeap != nil) {
        context.persistentHeap = replacementPersistentHeap;
    }
    if (replacementTransientHeap != nil) {
        context.transientHeap = replacementTransientHeap;
    }
    bool immutableBufferReplaced = false;
    bool persistentStateBufferReplaced = false;
    for (std::size_t index = 0u;
         index < kRawBufferCount;
         ++index) {
        if (replacements[index] == nil) {
            continue;
        }
        if (context.capacities[index] != 0u) {
            ++context.stats.bufferGrowthCount;
        }
        ++context.stats.bufferAllocationCount;
        context.buffers[index] = replacements[index];
        context.capacities[index] = proposed[index];
        immutableBufferReplaced =
            immutableBufferReplaced ||
            (index >= kArticulations && index <= kBodies) ||
            index == kShapes ||
            index == kMaterials ||
            index == kSceneBodyIndices ||
            index == kEligiblePairs ||
            index == kDynamicNodes ||
            index == kBodyDynamicNodes ||
            (index >= kAuthoredIRBlocks &&
             index <= kAuthoredIRWarmImpulses) ||
            (index >= kRodRestLengths &&
             index <= kRodTwistStiffness) ||
            index == kConvexCaches ||
            index == kCandidateConvexCaches ||
            (index >= kGeometryHeaders &&
             index <= kMeshTriangles);
        persistentStateBufferReplaced =
            persistentStateBufferReplaced ||
            privatePersistentBuffer(index);
    }
    std::size_t retainedBytes = 0u;
    if (!checkedAdd(
            projectedBytes,
            context.readbackBytes,
            retainedBytes
        )) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::arithmeticOverflow,
            "MetalWorld arena plus readback byte-count overflow"
        );
    }
    if (usePrivateHeap) {
        for (std::size_t index = 0u;
             index < kRawBufferCount;
             ++index) {
            if (uploadReplacements[index] != nil) {
                if (context.uploadCapacities[index] != 0u) {
                    ++context.stats.bufferGrowthCount;
                }
                ++context.stats.bufferAllocationCount;
                context.uploadBuffers[index] =
                    uploadReplacements[index];
                context.uploadCapacities[index] =
                    proposed[index];
            }
            if ((privateImmutableBuffer(index) ||
                 privatePersistentInputBuffer(index)) &&
                !checkedAdd(
                    retainedBytes,
                    proposed[index],
                    retainedBytes
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::arithmeticOverflow,
                    "retained immutable staging byte-count overflow"
                );
            }
        }
    }
    if (immutableBufferReplaced) {
        context.boundModelFingerprint = 0u;
        context.boundTaskFingerprint = 0u;
        context.boundPolicyFingerprint = 0u;
        context.boundMulticopterFingerprint = 0u;
        context.boundFlappingWingFingerprint = 0u;
    }
    if (persistentStateBufferReplaced) {
        ++context.stateArenaGeneration;
    }
    if (recommendedWorkingSet != 0u &&
        static_cast<std::uint64_t>(retainedBytes) >
            recommendedWorkingSet) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalBufferFailure,
            "private heaps plus immutable staging exceed "
            "device.recommendedMaxWorkingSetSize"
        );
    }
    context.stats.retainedBufferBytes = retainedBytes;
    context.stats.usingPrivateHeaps =
        usePrivateHeap &&
        context.immutableHeap != nil &&
        context.persistentHeap != nil &&
        context.transientHeap != nil;
    return diagnostics;
}

MetalWorldDiagnostics ensureReadbackBuffer(
    detail::MetalWorldContextState& context,
    const std::size_t index,
    const BufferRequirement& requirement,
    MetalWorldDiagnostics diagnostics
) {
    if (context.buffers[index].storageMode !=
            MTLStorageModePrivate ||
        requirement.logicalBytes == 0u ||
        context.readbackCapacities[index] >=
            requirement.logicalBytes) {
        return diagnostics;
    }
    const std::size_t maximumBufferLength =
        static_cast<std::size_t>(
            context.device.maxBufferLength
        );
    const std::size_t proposed = growthCapacity(
        context.readbackCapacities[index],
        requirement.logicalBytes,
        maximumBufferLength
    );
    if (proposed > maximumBufferLength) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalBufferFailure,
            std::string(requirement.label) +
                " readback exceeds device.maxBufferLength"
        );
    }
    const std::size_t oldCapacity =
        context.readbackCapacities[index];
    const std::size_t additional = proposed - oldCapacity;
    if (context.stats.retainedBufferBytes >
            std::numeric_limits<std::size_t>::max() -
                additional) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::arithmeticOverflow,
            "MetalWorld readback arena byte-count overflow"
        );
    }
    const std::size_t retained =
        context.stats.retainedBufferBytes + additional;
    const std::uint64_t recommendedWorkingSet =
        context.device.recommendedMaxWorkingSetSize;
    if (recommendedWorkingSet != 0u &&
        static_cast<std::uint64_t>(retained) >
            recommendedWorkingSet) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalBufferFailure,
            "MetalWorld arena plus requested readback exceeds "
            "device.recommendedMaxWorkingSetSize"
        );
    }
    id<MTLBuffer> replacement = [context.device
        newBufferWithLength:static_cast<NSUInteger>(proposed)
                   options:MTLResourceStorageModeShared];
    if (replacement == nil ||
        replacement.contents == nullptr ||
        replacement.length < proposed) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalBufferFailure,
            std::string(requirement.label) +
                " readback allocation failed"
        );
    }
    replacement.label = [
        bufferLabel(index)
        stringByAppendingString:@" readback"
    ];
    if (oldCapacity != 0u) {
        ++context.stats.bufferGrowthCount;
    }
    ++context.stats.bufferAllocationCount;
    context.readbackBuffers[index] = replacement;
    context.readbackCapacities[index] = proposed;
    context.readbackBytes =
        context.readbackBytes - oldCapacity + proposed;
    context.stats.retainedBufferBytes = retained;
    return diagnostics;
}

MetalWorldDiagnostics encodeReadbacks(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const RequiredBuffers& requirements,
    const std::span<const std::size_t> indices,
    MetalWorldDiagnostics diagnostics
) {
    bool needsReadback = false;
    for (const std::size_t index : indices) {
        if (context.buffers[index].storageMode !=
            MTLStorageModePrivate) {
            continue;
        }
        diagnostics = ensureReadbackBuffer(
            context,
            index,
            requirements.entries[index],
            std::move(diagnostics)
        );
        if (!diagnostics.succeeded()) {
            return diagnostics;
        }
        needsReadback =
            needsReadback ||
            requirements.entries[index].logicalBytes != 0u;
    }
    if (!needsReadback) {
        return diagnostics;
    }
    id<MTLBlitCommandEncoder> encoder =
        [commandBuffer blitCommandEncoder];
    if (encoder == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalCommandFailure,
            "failed to create MetalWorld readback encoder"
        );
    }
    encoder.label = @"MetalWorld explicit state readback";
    for (const std::size_t index : indices) {
        const std::size_t bytes =
            requirements.entries[index].logicalBytes;
        if (bytes == 0u ||
            context.buffers[index].storageMode !=
                MTLStorageModePrivate) {
            continue;
        }
        [encoder
            copyFromBuffer:context.buffers[index]
              sourceOffset:0u
                  toBuffer:context.readbackBuffers[index]
         destinationOffset:0u
                      size:static_cast<NSUInteger>(bytes)];
    }
    [encoder endEncoding];
    return diagnostics;
}

void copyToBuffer(
    id<MTLBuffer> destination,
    const void* source,
    const BufferRequirement& requirement
) {
    if (destination.storageMode == MTLStorageModePrivate ||
        destination.contents == nullptr) {
        throw std::runtime_error(
            std::string("CPU upload attempted for private buffer: ") +
            requirement.label
        );
    }
    if (requirement.logicalBytes == 0u) {
        std::memset(
            destination.contents,
            0,
            requirement.allocationBytes
        );
        return;
    }
    std::memcpy(
        destination.contents,
        source,
        requirement.logicalBytes
    );
}

void stagePrivateBuffer(
    detail::MetalWorldContextState& context,
    const std::size_t index,
    const void* source,
    const BufferRequirement& requirement,
    id<MTLBlitCommandEncoder> blit
) {
    id<MTLBuffer> destination = context.buffers[index];
    if (destination.storageMode != MTLStorageModePrivate) {
        copyToBuffer(destination, source, requirement);
        return;
    }
    id<MTLBuffer> staging = context.uploadBuffers[index];
    if (staging == nil ||
        staging.contents == nullptr ||
        staging.length < requirement.allocationBytes ||
        blit == nil) {
        throw std::runtime_error(
            std::string(
                "private upload staging is not prepared for "
            ) + requirement.label
        );
    }
    std::memset(
        staging.contents,
        0,
        requirement.allocationBytes
    );
    if (requirement.logicalBytes != 0u) {
        std::memcpy(
            staging.contents,
            source,
            requirement.logicalBytes
        );
    }
    [blit
        copyFromBuffer:staging
        sourceOffset:0u
        toBuffer:destination
        destinationOffset:0u
        size:requirement.allocationBytes];
}

void zeroBuffer(
    id<MTLBuffer> destination,
    const BufferRequirement& requirement
) {
    if (destination.storageMode == MTLStorageModePrivate ||
        destination.contents == nullptr) {
        return;
    }
    std::memset(
        destination.contents,
        0,
        requirement.allocationBytes
    );
}

void uploadBatch(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const CompiledWorld& world,
    const MetalWorldBatch& batch,
    const MetalWorldStepConfig& config,
    const MetalWorldLayout& layout,
    const RequiredBuffers& requirements,
    const bool uploadInitialState
) {
    const EngineModel& model = world.model();
    MRWorldGPU runtimeWorld = model.world;
    runtimeWorld.gravityAndTimestep.w =
        config.timestepSeconds /
        static_cast<float>(config.physicsSubsteps);
    copyToBuffer(
        context.buffers[kWorld],
        &runtimeWorld,
        requirements.entries[kWorld]
    );

    if (context.boundModelFingerprint !=
        world.fingerprint()) {
        id<MTLBlitCommandEncoder> immutableUpload = nil;
        if (context.config.preferPrivateHeaps) {
            immutableUpload =
                [commandBuffer blitCommandEncoder];
            if (immutableUpload == nil) {
                throw std::runtime_error(
                    "failed to create immutable upload encoder"
                );
            }
            immutableUpload.label =
                @"MetalWorld immutable model upload";
        }
        MRJointDescriptorGPU emptyJoint{};
        const std::array<const void*, 4u> sources{
            model.articulations.data(),
            model.joints.empty()
                ? static_cast<const void*>(&emptyJoint)
                : static_cast<const void*>(
                      model.joints.data()
                  ),
            model.dofs.data(),
            model.bodies.data(),
        };
        for (std::size_t offset = 0u;
             offset < sources.size();
             ++offset) {
            const std::size_t index =
                kArticulations + offset;
            stagePrivateBuffer(
                context,
                index,
                sources[offset],
                requirements.entries[index],
                immutableUpload
            );
        }
        ParallelABASchedule parallelSchedule;
        if ((layout.contactDispatch.flags &
             MR_METAL_WORLD_CONTACT_STREAMED_RESPONSES) != 0u) {
            const ParallelABAScheduleDiagnostics scheduleDiagnostics =
                compileParallelABASchedule(model, parallelSchedule);
            if (!scheduleDiagnostics.succeeded()) {
                throw std::runtime_error(
                    "failed to compile parallel ABA schedule: " +
                    scheduleDiagnostics.message
                );
            }
        }
        const std::array<std::pair<std::size_t, const void*>, 8u>
            scheduleSources{{
                {
                    kParallelScheduleArticulations,
                    parallelSchedule.articulations.data(),
                },
                {
                    kParallelScheduleLevels,
                    parallelSchedule.levels.data(),
                },
                {
                    kParallelScheduleParentReductions,
                    parallelSchedule.parentReductions.data(),
                },
                {
                    kParallelScheduleLevelBodies,
                    parallelSchedule.levelBodies.data(),
                },
                {
                    kParallelScheduleParentLocal,
                    parallelSchedule.parentLocal.data(),
                },
                {
                    kParallelScheduleInboundJoint,
                    parallelSchedule.inboundJoint.data(),
                },
                {
                    kParallelScheduleChildOffsets,
                    parallelSchedule.childOffsets.data(),
                },
                {
                    kParallelScheduleChildIndices,
                    parallelSchedule.childIndices.data(),
                },
            }};
        for (const auto& [index, source] : scheduleSources) {
            stagePrivateBuffer(
                context,
                index,
                source,
                requirements.entries[index],
                immutableUpload
            );
        }
        const std::vector<MRActuatorProfileGPU>
            actuatorProfiles =
                executionActuatorProfiles(model);
        stagePrivateBuffer(
            context,
            kActuatorProfiles,
            actuatorProfiles.data(),
            requirements.entries[kActuatorProfiles],
            immutableUpload
        );
        stagePrivateBuffer(
            context,
            kTaskDefaultQ,
            model.defaultQ.data(),
            requirements.entries[kTaskDefaultQ],
            immutableUpload
        );
        MRShapeGPU emptyShape{};
        MRMaterialGPU emptyMaterial{};
        mr_u32 emptySceneIndex = 0u;
        MRCompiledCollisionPairGPU emptyPair{};
        MRGeometryHeaderGPU emptyGeometry{};
        mr_float4 emptyVertex{};
        mr_u32 emptyGeometryIndex = 0u;
        MRConvexFaceGPU emptyFace{};
        MRConvexHalfEdgeGPU emptyHalfEdge{};
        MRMeshBVHNodeGPU emptyBvhNode{};
        MRMeshTriangleGPU emptyTriangle{};
        MRWorldDynamicNodeGPU emptyDynamicNode{};
        mr_u32 emptyDynamicNodeIndex = MR_INVALID_INDEX;
        MRRodColliderGPU emptyRodCollider{};
        MRRodToolPairGPU emptyRodToolPair{};
        MRConstraintIRBlockGPU emptyAuthoredBlock{};
        MRConstraintIREndpointGPU emptyAuthoredEndpoint{};
        MRConstraintIRRowGPU emptyAuthoredRow{};
        MRConstraintIRConeGPU emptyAuthoredCone{};
        float emptyRodScalar = 0.0f;
        mr_float4 emptyRodCurvature{};
        const std::array<std::pair<std::size_t, const void*>, 21u>
            contactSources{{
                {
                    kShapes,
                    model.shapes.empty()
                        ? static_cast<const void*>(&emptyShape)
                        : static_cast<const void*>(
                              model.shapes.data()
                          ),
                },
                {
                    kMaterials,
                    model.materials.empty()
                        ? static_cast<const void*>(&emptyMaterial)
                        : static_cast<const void*>(
                              model.materials.data()
                          ),
                },
                {
                    kSceneBodyIndices,
                    world.sceneBodyIndices().empty()
                        ? static_cast<const void*>(
                              &emptySceneIndex
                          )
                        : static_cast<const void*>(
                              world.sceneBodyIndices().data()
                          ),
                },
                {
                    kEligiblePairs,
                    world.eligiblePairs().empty()
                        ? static_cast<const void*>(&emptyPair)
                        : static_cast<const void*>(
                              world.eligiblePairs().data()
                          ),
                },
                {
                    kGeometryHeaders,
                    model.geometryHeaders.empty()
                        ? static_cast<const void*>(&emptyGeometry)
                        : static_cast<const void*>(
                              model.geometryHeaders.data()
                          ),
                },
                {
                    kGeometryVertices,
                    model.geometryVertices.empty()
                        ? static_cast<const void*>(&emptyVertex)
                        : static_cast<const void*>(
                              model.geometryVertices.data()
                          ),
                },
                {
                    kGeometryIndices,
                    model.geometryIndices.empty()
                        ? static_cast<const void*>(&emptyGeometryIndex)
                        : static_cast<const void*>(
                              model.geometryIndices.data()
                          ),
                },
                {
                    kConvexFaces,
                    model.convexFaces.empty()
                        ? static_cast<const void*>(&emptyFace)
                        : static_cast<const void*>(
                              model.convexFaces.data()
                          ),
                },
                {
                    kConvexHalfEdges,
                    model.convexHalfEdges.empty()
                        ? static_cast<const void*>(&emptyHalfEdge)
                        : static_cast<const void*>(
                              model.convexHalfEdges.data()
                          ),
                },
                {
                    kMeshBvhNodes,
                    model.meshBvhNodes.empty()
                        ? static_cast<const void*>(&emptyBvhNode)
                        : static_cast<const void*>(
                              model.meshBvhNodes.data()
                          ),
                },
                {
                    kMeshTriangles,
                    model.meshTriangles.empty()
                        ? static_cast<const void*>(&emptyTriangle)
                        : static_cast<const void*>(
                              model.meshTriangles.data()
                          ),
                },
                {
                    kDynamicNodes,
                    world.dynamicNodes().empty()
                        ? static_cast<const void*>(
                              &emptyDynamicNode
                          )
                        : static_cast<const void*>(
                              world.dynamicNodes().data()
                          ),
                },
                {
                    kBodyDynamicNodes,
                    world.bodyDynamicNodes().empty()
                        ? static_cast<const void*>(
                              &emptyDynamicNodeIndex
                          )
                        : static_cast<const void*>(
                              world.bodyDynamicNodes().data()
                          ),
                },
                {
                    kRodColliders,
                    world.rodColliders().empty()
                        ? static_cast<const void*>(
                              &emptyRodCollider
                          )
                        : static_cast<const void*>(
                              world.rodColliders().data()
                          ),
                },
                {
                    kRodShapeSources,
                    world.rodShapeSources().empty()
                        ? static_cast<const void*>(&emptyShape)
                        : static_cast<const void*>(
                              world.rodShapeSources().data()
                          ),
                },
                {
                    kRodToolPairs,
                    world.rodToolPairs().empty()
                        ? static_cast<const void*>(
                              &emptyRodToolPair
                          )
                        : static_cast<const void*>(
                              world.rodToolPairs().data()
                          ),
                },
                {
                    kAuthoredIRBlocks,
                    model.constraintProgram.blocks.empty()
                        ? static_cast<const void*>(
                              &emptyAuthoredBlock
                          )
                        : static_cast<const void*>(
                              model.constraintProgram.blocks.data()
                          ),
                },
                {
                    kAuthoredIREndpoints,
                    model.constraintProgram.endpoints.empty()
                        ? static_cast<const void*>(
                              &emptyAuthoredEndpoint
                          )
                        : static_cast<const void*>(
                              model.constraintProgram.endpoints.data()
                          ),
                },
                {
                    kAuthoredIRRows,
                    model.constraintProgram.rows.empty()
                        ? static_cast<const void*>(
                              &emptyAuthoredRow
                          )
                        : static_cast<const void*>(
                              model.constraintProgram.rows.data()
                          ),
                },
                {
                    kAuthoredIRCones,
                    model.constraintProgram.cones.empty()
                        ? static_cast<const void*>(
                              &emptyAuthoredCone
                          )
                        : static_cast<const void*>(
                              model.constraintProgram.cones.data()
                          ),
                },
                {
                    kAuthoredIRWarmImpulses,
                    model.constraintProgram.warmImpulses.empty()
                        ? static_cast<const void*>(
                              &emptyRodScalar
                          )
                        : static_cast<const void*>(
                              model.constraintProgram
                                  .warmImpulses.data()
                          ),
                },
            }};
        for (const auto& [index, source] : contactSources) {
            stagePrivateBuffer(
                context,
                index,
                source,
                requirements.entries[index],
                immutableUpload
            );
        }

        std::vector<float> rodRestLengths;
        std::vector<float> rodRestTwists;
        std::vector<mr_float4> rodRestCurvatures;
        std::vector<float> rodInverseMasses;
        std::vector<float> rodInverseRotationalInertias;
        std::vector<float> rodStretchStiffness;
        std::vector<float> rodBendStiffness;
        std::vector<float> rodTwistStiffness;
        rodRestLengths.reserve(world.rodEdgeCount());
        rodRestTwists.reserve(world.rodEdgeCount());
        rodRestCurvatures.reserve(
            layout.rodBendStateElements
        );
        rodInverseMasses.reserve(world.rodNodeCount());
        rodInverseRotationalInertias.reserve(
            world.rodEdgeCount()
        );
        rodStretchStiffness.reserve(world.rodEdgeCount());
        rodBendStiffness.reserve(
            layout.rodBendStateElements
        );
        rodTwistStiffness.reserve(
            layout.rodBendStateElements
        );
        for (const HeterogeneousRodProgram& program :
             world.rodPrograms()) {
            const auto& rod = program.model;
            for (const double value : rod.restLengths) {
                rodRestLengths.push_back(
                    static_cast<float>(value)
                );
            }
            for (const double value : rod.restTwists) {
                rodRestTwists.push_back(
                    static_cast<float>(value)
                );
            }
            for (const double value : rod.nodeMasses) {
                rodInverseMasses.push_back(
                    static_cast<float>(1.0 / value)
                );
            }
            for (const double value :
                 rod.edgeRotationalInertias) {
                rodInverseRotationalInertias.push_back(
                    static_cast<float>(1.0 / value)
                );
            }
            for (const double value :
                 rod.stretchStiffness) {
                rodStretchStiffness.push_back(
                    static_cast<float>(value)
                );
            }
            for (std::size_t bend = 0u;
                 bend + 1u < rod.restLengths.size();
                 ++bend) {
                mr_float4 curvature{};
                if (!rodRestCurvature(
                        rod,
                        bend,
                        curvature
                    )) {
                    throw std::runtime_error(
                        "compiled rod has degenerate rest "
                        "curvature"
                    );
                }
                rodRestCurvatures.push_back(curvature);
                rodBendStiffness.push_back(
                    static_cast<float>(
                        rod.bendStiffness[bend]
                    )
                );
                rodTwistStiffness.push_back(
                    static_cast<float>(
                        rod.twistStiffness[bend]
                    )
                );
            }
        }
        const std::array<
            std::pair<std::size_t, const void*>,
            8u
        > rodSources{{
            {
                kRodRestLengths,
                rodRestLengths.empty()
                    ? static_cast<const void*>(&emptyRodScalar)
                    : static_cast<const void*>(
                          rodRestLengths.data()
                      ),
            },
            {
                kRodRestTwists,
                rodRestTwists.empty()
                    ? static_cast<const void*>(&emptyRodScalar)
                    : static_cast<const void*>(
                          rodRestTwists.data()
                      ),
            },
            {
                kRodRestCurvatures,
                rodRestCurvatures.empty()
                    ? static_cast<const void*>(
                          &emptyRodCurvature
                      )
                    : static_cast<const void*>(
                          rodRestCurvatures.data()
                      ),
            },
            {
                kRodInverseMasses,
                rodInverseMasses.empty()
                    ? static_cast<const void*>(&emptyRodScalar)
                    : static_cast<const void*>(
                          rodInverseMasses.data()
                      ),
            },
            {
                kRodInverseRotationalInertias,
                rodInverseRotationalInertias.empty()
                    ? static_cast<const void*>(&emptyRodScalar)
                    : static_cast<const void*>(
                          rodInverseRotationalInertias.data()
                      ),
            },
            {
                kRodStretchStiffness,
                rodStretchStiffness.empty()
                    ? static_cast<const void*>(&emptyRodScalar)
                    : static_cast<const void*>(
                          rodStretchStiffness.data()
                      ),
            },
            {
                kRodBendStiffness,
                rodBendStiffness.empty()
                    ? static_cast<const void*>(&emptyRodScalar)
                    : static_cast<const void*>(
                          rodBendStiffness.data()
                      ),
            },
            {
                kRodTwistStiffness,
                rodTwistStiffness.empty()
                    ? static_cast<const void*>(&emptyRodScalar)
                    : static_cast<const void*>(
                          rodTwistStiffness.data()
                      ),
            },
        }};
        for (const auto& [index, source] : rodSources) {
            stagePrivateBuffer(
                context,
                index,
                source,
                requirements.entries[index],
                immutableUpload
            );
        }
        if (immutableUpload != nil) {
            [immutableUpload endEncoding];
        }
        context.boundModelFingerprint = world.fingerprint();
        context.boundArticulations.assign(
            model.articulations.begin(),
            model.articulations.end()
        );
        ++context.stats.modelUploadCount;
    }

    const bool nativeTask = config.taskProgram.valid();
    const bool nativePolicy = config.policyProgram.valid();
    if (nativeTask &&
        context.boundTaskFingerprint !=
            config.taskProgram.fingerprint()) {
        id<MTLBlitCommandEncoder> taskUpload = nil;
        if (context.config.preferPrivateHeaps) {
            taskUpload = [commandBuffer blitCommandEncoder];
            if (taskUpload == nil) {
                throw std::runtime_error(
                    "failed to create compiled-task upload encoder"
                );
            }
            taskUpload.label =
                @"MetalWorld immutable compiled task upload";
        }
        const auto stageTask =
            [&](const std::size_t index, const void* source) {
                stagePrivateBuffer(
                    context,
                    index,
                    source,
                    requirements.entries[index],
                    taskUpload
                );
            };
        stageTask(
            kTaskProgramHeader,
            &config.taskProgram.header()
        );
        stageTask(
            kTaskProgramArena,
            config.taskProgram.arena().data()
        );
        if (taskUpload != nil) {
            [taskUpload endEncoding];
        }
        context.boundTaskFingerprint =
            config.taskProgram.fingerprint();
    }
    const std::uint64_t multicopterHash =
        multicopterFingerprint(config.multicopterProgram);
    if (config.multicopterProgram.valid() &&
        context.boundMulticopterFingerprint != multicopterHash) {
        id<MTLBlitCommandEncoder> actuatorUpload = nil;
        if (context.config.preferPrivateHeaps) {
            actuatorUpload = [commandBuffer blitCommandEncoder];
            if (actuatorUpload == nil) {
                throw std::runtime_error(
                    "failed to create compiled-actuator upload encoder"
                );
            }
            actuatorUpload.label =
                @"MetalWorld immutable multicopter program upload";
        }
        const auto& multicopter = config.multicopterProgram;
        const MRArticulationGPU& articulation =
            model.articulations[multicopter.articulationIndex];
        MRCompiledMulticopterDispatchGPU dispatch{};
        dispatch.environmentCount = layout.dispatch.environmentCount;
        dispatch.qStride = layout.dispatch.qStride;
        dispatch.vStride = layout.dispatch.vStride;
        dispatch.bodyStride = static_cast<mr_u32>(model.bodies.size());
        dispatch.qOffset = articulation.qOffset;
        dispatch.vOffset = articulation.vOffset;
        dispatch.bodyIndex = multicopter.bodyIndex;
        dispatch.localBodyIndex =
            multicopter.bodyIndex - articulation.firstBody;
        dispatch.actionCount = config.taskProgram.layout().actionCount;
        dispatch.actionHistoryStride =
            config.taskProgram.layout().delayStateCount *
            config.taskProgram.layout().actionCount;
        dispatch.filterSlot =
            config.taskProgram.layout().delayStateCount - 1u;
        dispatch.firstAction = multicopter.firstAction;
        dispatch.windVelocity = multicopter.windVelocity;
        const std::array<std::pair<std::size_t, const void*>, 4u>
            sources{{
                {kMulticopterRotors, multicopter.rotors.data()},
                {kMulticopterModel, &multicopter.model},
                {kMulticopterMixer, &multicopter.mixer},
                {kMulticopterDispatch, &dispatch},
            }};
        for (const auto& [index, source] : sources) {
            stagePrivateBuffer(
                context,
                index,
                source,
                requirements.entries[index],
                actuatorUpload
            );
        }
        if (actuatorUpload != nil) {
            [actuatorUpload endEncoding];
        }
        context.boundMulticopterFingerprint = multicopterHash;
    }
    const std::uint64_t flappingWingHash =
        flappingWingFingerprint(config.flappingWingProgram);
    if (config.flappingWingProgram.valid() &&
        context.boundFlappingWingFingerprint != flappingWingHash) {
        id<MTLBlitCommandEncoder> actuatorUpload = nil;
        if (context.config.preferPrivateHeaps) {
            actuatorUpload = [commandBuffer blitCommandEncoder];
            if (actuatorUpload == nil) {
                throw std::runtime_error(
                    "failed to create flapping-wing program upload encoder"
                );
            }
            actuatorUpload.label =
                @"MetalWorld immutable flapping-wing program upload";
        }
        const auto& wings = config.flappingWingProgram;
        const MRArticulationGPU& articulation =
            model.articulations[wings.articulationIndex];
        MRCompiledFlappingWingDispatchGPU dispatch{};
        dispatch.environmentCount = layout.dispatch.environmentCount;
        dispatch.qStride = layout.dispatch.qStride;
        dispatch.vStride = layout.dispatch.vStride;
        dispatch.bodyStride = static_cast<mr_u32>(model.bodies.size());
        dispatch.qOffset = articulation.qOffset;
        dispatch.vOffset = articulation.vOffset;
        dispatch.rootBodyIndex = wings.rootBodyIndex;
        dispatch.windVelocityAndDensity = wings.windVelocityAndDensity;
        stagePrivateBuffer(
            context, kFlappingWingSpecs, wings.wings.data(),
            requirements.entries[kFlappingWingSpecs], actuatorUpload
        );
        stagePrivateBuffer(
            context, kFlappingWingDispatch, &dispatch,
            requirements.entries[kFlappingWingDispatch], actuatorUpload
        );
        if (actuatorUpload != nil) {
            [actuatorUpload endEncoding];
        }
        context.boundFlappingWingFingerprint = flappingWingHash;
    }
    if (nativePolicy &&
        context.boundPolicyFingerprint !=
            config.policyProgram.fingerprint()) {
        id<MTLBlitCommandEncoder> policyUpload = nil;
        if (context.config.preferPrivateHeaps) {
            policyUpload = [commandBuffer blitCommandEncoder];
            if (policyUpload == nil) {
                throw std::runtime_error(
                    "failed to create compiled-policy upload encoder"
                );
            }
            policyUpload.label =
                @"MetalWorld immutable compiled policy upload";
        }
        const auto stagePolicy =
            [&](const std::size_t index, const void* source) {
                stagePrivateBuffer(
                    context,
                    index,
                    source,
                    requirements.entries[index],
                    policyUpload
                );
            };
        stagePolicy(
            kPolicyProgramHeader,
            &config.policyProgram.header()
        );
        stagePolicy(
            kPolicyProgramArena,
            config.policyProgram.arena().data()
        );
        if (policyUpload != nil) {
            [policyUpload endEncoding];
        }
        context.boundPolicyFingerprint =
            config.policyProgram.fingerprint();
    }

    copyToBuffer(
        context.buffers[kABADispatch],
        layout.abaDispatches.data(),
        requirements.entries[kABADispatch]
    );
    std::vector<MRRodGPUDispatch> rodDispatches;
    rodDispatches.reserve(world.rodCount());
    std::vector<MRRodGPUDispatch> rodCollisionDispatches;
    rodCollisionDispatches.reserve(world.rodCount());
    std::size_t rodToolPairCursor = 0u;
    for (std::size_t rodIndex = 0u;
         rodIndex < world.rodPrograms().size();
         ++rodIndex) {
        const HeterogeneousRodProgram& program =
            world.rodPrograms()[rodIndex];
        const std::uint32_t nodeCount =
            world.rodNodeOffsets()[rodIndex + 1u] -
            world.rodNodeOffsets()[rodIndex];
        const std::uint32_t edgeCount =
            world.rodEdgeOffsets()[rodIndex + 1u] -
            world.rodEdgeOffsets()[rodIndex];
        const std::uint32_t nodeBase =
            world.rodNodeOffsets()[rodIndex];
        const std::uint32_t edgeBase =
            world.rodEdgeOffsets()[rodIndex];
        while (rodToolPairCursor < world.rodToolPairs().size() &&
               world.rodToolPairs()[rodToolPairCursor]
                       .rodCollider < edgeBase) {
            ++rodToolPairCursor;
        }
        const std::size_t pairBase = rodToolPairCursor;
        while (rodToolPairCursor < world.rodToolPairs().size() &&
               world.rodToolPairs()[rodToolPairCursor]
                       .rodCollider <
                   edgeBase + edgeCount) {
            ++rodToolPairCursor;
        }
        const std::size_t pairCount =
            rodToolPairCursor - pairBase;
        MRRodGPUDispatch dispatch{};
        dispatch.abiVersion = MR_ROD_GPU_ABI_VERSION;
        dispatch.environmentCount =
            layout.dispatch.environmentCount;
        dispatch.nodeCount = nodeCount;
        dispatch.edgeCount = edgeCount;
        dispatch.attachmentCount = 0u;
        dispatch.solverIterations =
            program.stepConfig.solverIterations;
        dispatch.stateNodeStride = world.rodNodeCount();
        dispatch.stateEdgeStride = world.rodEdgeCount();
        dispatch.rigidBodyCount = world.bodyCount();
        dispatch.stateBodyStride = world.bodyCount();
        dispatch.flags =
            program.stepConfig.enableSelfCollision
            ? MR_ROD_GPU_FLAG_SELF_COLLISION
            : 0u;
        dispatch.toolShapeCount = world.colliderCount();
        dispatch.toolPairCount =
            static_cast<mr_u32>(pairCount);
        dispatch.toolContactStride =
            static_cast<mr_u32>(
                world.rodToolPairs().size() *
                MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR
            );
        dispatch.toolContactIterations =
            pairCount == 0u
            ? 0u
            : config.rodContactOuterIterations;
        dispatch.rodMaterialIndex =
            program.collision.materialIndex;
        dispatch.rodNodeBase = nodeBase;
        dispatch.rodEdgeBase = edgeBase;
        dispatch.toolPairBase =
            static_cast<mr_u32>(pairBase);
        dispatch.toolPairWorldStride =
            static_cast<mr_u32>(
                world.rodToolPairs().size()
            );
        mr_u32 collisionFlags = dispatch.flags;
        if (pairCount != 0u) {
            collisionFlags |=
                MR_ROD_GPU_FLAG_TOOL_COLLISION |
                MR_ROD_GPU_FLAG_TOOL_WARM_START;
            const bool enableCCD = std::any_of(
                world.rodToolPairs().begin() +
                    static_cast<std::ptrdiff_t>(pairBase),
                world.rodToolPairs().begin() +
                    static_cast<std::ptrdiff_t>(
                        rodToolPairCursor
                    ),
                [](const MRRodToolPairGPU& pair) {
                    return
                        (pair.flags &
                         MR_ROD_TOOL_PAIR_ENABLE_CCD) != 0u;
                }
            );
            if (enableCCD) {
                collisionFlags |=
                    MR_ROD_GPU_FLAG_ENABLE_CCD;
            }
        }
        dispatch.gravityAndTimestep = {
            static_cast<float>(
                program.stepConfig.gravity[0]
            ),
            static_cast<float>(
                program.stepConfig.gravity[1]
            ),
            static_cast<float>(
                program.stepConfig.gravity[2]
            ),
            runtimeWorld.gravityAndTimestep.w,
        };
        dispatch.dampingDerivativeTolerance = {
            static_cast<float>(
                program.stepConfig.linearDamping
            ),
            static_cast<float>(
                program.stepConfig.twistDamping
            ),
            static_cast<float>(std::max(
                program.stepConfig.derivativeStep,
                3.5e-4
            )),
            static_cast<float>(
                program.stepConfig.constraintTolerance
            ),
        };
        dispatch.selfCollision = {
            static_cast<float>(program.model.radius),
            static_cast<float>(
                program.stepConfig.selfCollisionMargin
            ),
            static_cast<float>(
                program.stepConfig.selfCollisionCompliance
            ),
            0.0f,
        };
        dispatch.toolContact = {
            static_cast<float>(
                program.collision.contactOffset
            ),
            static_cast<float>(
                program.collision.restOffset
            ),
            0.0f,
            0.0f,
        };
        dispatch.toolResponse = {
            0.0f,
            1.0f,
            1.0f,
            10.0f,
        };
        rodDispatches.push_back(dispatch);
        MRRodGPUDispatch collisionDispatch = dispatch;
        collisionDispatch.flags = collisionFlags;
        rodCollisionDispatches.push_back(collisionDispatch);
    }
    MRRodGPUDispatch emptyRodDispatch{};
    copyToBuffer(
        context.buffers[kRodDispatches],
        rodDispatches.empty()
            ? static_cast<const void*>(&emptyRodDispatch)
            : static_cast<const void*>(
                  rodDispatches.data()
              ),
        requirements.entries[kRodDispatches]
    );
    copyToBuffer(
        context.buffers[kRodCollisionDispatches],
        rodCollisionDispatches.empty()
            ? static_cast<const void*>(&emptyRodDispatch)
            : static_cast<const void*>(
                  rodCollisionDispatches.data()
              ),
        requirements.entries[kRodCollisionDispatches]
    );
    id<MTLBlitCommandEncoder> stateUpload = nil;
    if (uploadInitialState &&
        context.config.preferPrivateHeaps) {
        stateUpload = [commandBuffer blitCommandEncoder];
        if (stateUpload == nil) {
            throw std::runtime_error(
                "failed to create resident-state upload encoder"
            );
        }
        stateUpload.label =
            @"MetalWorld initialize resident state";
    }
    if (uploadInitialState) {
        stagePrivateBuffer(
            context,
            kStateQA,
            batch.initialQ.data(),
            requirements.entries[kStateQA],
            stateUpload
        );
        stagePrivateBuffer(
            context,
            kStateVA,
            batch.initialV.data(),
            requirements.entries[kStateVA],
            stateUpload
        );
    }
    if (!nativeTask) {
        copyToBuffer(
            context.buffers[kEffortTrajectory],
            batch.efforts.data(),
            requirements.entries[kEffortTrajectory]
        );
    }
    copyToBuffer(
        context.buffers[kResetMasks],
        batch.resetMasks.data(),
        requirements.entries[kResetMasks]
    );
    if (uploadInitialState && !nativeTask) {
        stagePrivateBuffer(
            context,
            kResetQ,
            batch.resetQ.data(),
            requirements.entries[kResetQ],
            stateUpload
        );
        stagePrivateBuffer(
            context,
            kResetV,
            batch.resetV.data(),
            requirements.entries[kResetV],
            stateUpload
        );
    }
    copyToBuffer(
        context.buffers[kWorldDispatch],
        &layout.dispatch,
        requirements.entries[kWorldDispatch]
    );
    copyToBuffer(
        context.buffers[kContactDispatch],
        &layout.contactDispatch,
        requirements.entries[kContactDispatch]
    );
    copyToBuffer(
        context.buffers[kInverseMassDispatch],
        &layout.inverseMassDispatch,
        requirements.entries[kInverseMassDispatch]
    );
    if (layout.contactDispatch.solverType ==
        MR_SOLVER_QUALITY_NEWTON) {
        copyToBuffer(
            context.buffers[kQualityDispatch],
            &layout.qualityDispatch,
            requirements.entries[kQualityDispatch]
        );
    }
    copyToBuffer(
        context.buffers[kOperatorKinematicsDispatch],
        layout.kinematicsDispatches.data(),
        requirements.entries[kOperatorKinematicsDispatch]
    );
    copyToBuffer(
        context.buffers[kOperatorFactorDispatch],
        layout.factorDispatches.data(),
        requirements.entries[kOperatorFactorDispatch]
    );
    context.boundFactorDispatches = layout.factorDispatches;
    context.boundContactDispatch = layout.contactDispatch;
    if (uploadInitialState) {
        copyToBuffer(
            context.buffers[kInitialSceneBodies],
            batch.initialSceneBodies.data(),
            requirements.entries[kInitialSceneBodies]
        );
        stagePrivateBuffer(
            context,
            kSceneBodiesA,
            batch.initialSceneBodies.data(),
            requirements.entries[kSceneBodiesA],
            stateUpload
        );
        if (!nativeTask) {
            stagePrivateBuffer(
                context,
                kResetSceneBodies,
                batch.resetSceneBodies.data(),
                requirements.entries[kResetSceneBodies],
                stateUpload
            );
        }
    }
    copyToBuffer(
        context.buffers[kKinematicTargets],
        batch.kinematicTargets.data(),
        requirements.entries[kKinematicTargets]
    );
    if (nativeTask) {
        const TaskProgramLayout& taskLayout =
            config.taskProgram.layout();
        const std::size_t actionStepStride =
            batch.environmentCount * taskLayout.actionCount;
        MRTaskDispatchGPU task{};
        task.counts = {
            static_cast<mr_u32>(batch.environmentCount),
            static_cast<mr_u32>(batch.controlStepCount),
            layout.dispatch.nq,
            layout.dispatch.nv,
        };
        task.strides = {
            static_cast<mr_u32>(actionStepStride),
            layout.contactDispatch.bodyStateStride,
            layout.contactDispatch.shapeCount,
            layout.contactDispatch.sceneBodyStride,
        };
        task.outputs = {
            static_cast<mr_u32>(
                batch.environmentCount *
                taskLayout.actorObservationSize
            ),
            static_cast<mr_u32>(
                batch.environmentCount *
                taskLayout.criticObservationSize
            ),
            static_cast<mr_u32>(batch.environmentCount),
            static_cast<mr_u32>(batch.environmentCount),
        };
        task.timing = {
            config.timestepSeconds,
            config.timestepSeconds /
                static_cast<float>(config.physicsSubsteps),
            config.evaluateFinalPolicy ? 1.0f : 0.0f,
            nativePolicy &&
                    !config.policyProgram.criticLayers().empty()
                ? 1.0f
                : 0.0f,
        };
        task.sampling = {
            config.minimumDifficultyBand,
            config.maximumDifficultyBand,
            static_cast<mr_u32>(model.bodies.size()),
            0u,
        };
        task.seed = config.taskSeed;
        task.policyRevision =
            nativePolicy
            ? config.policyProgram.revision()
            : batch.policyRevision;
        task.taskFingerprint =
            config.taskProgram.fingerprint();
        task.worldFingerprint = world.fingerprint();
        copyToBuffer(
            context.buffers[kTaskDispatch],
            &task,
            requirements.entries[kTaskDispatch]
        );
        if (!nativePolicy) {
            copyToBuffer(
                context.buffers[kTaskActions],
                batch.actions.data(),
                requirements.entries[kTaskActions]
            );
        }
    }

    const auto expandRodNodes =
        [&](const std::span<const MRRodNodeStateGPU> authored) {
            std::vector<MRRodNodeStateGPU> expanded;
            if (!authored.empty()) {
                expanded.assign(
                    authored.begin(),
                    authored.end()
                );
                return expanded;
            }
            expanded.reserve(layout.rodNodeStateElements);
            for (std::size_t environment = 0u;
                 environment < batch.environmentCount;
                 ++environment) {
                expanded.insert(
                    expanded.end(),
                    world.defaultRodNodes().begin(),
                    world.defaultRodNodes().end()
                );
            }
            return expanded;
        };
    const auto expandRodEdges =
        [&](const std::span<const MRRodEdgeStateGPU> authored) {
            std::vector<MRRodEdgeStateGPU> expanded;
            if (!authored.empty()) {
                expanded.assign(
                    authored.begin(),
                    authored.end()
                );
                return expanded;
            }
            expanded.reserve(layout.rodEdgeStateElements);
            for (std::size_t environment = 0u;
                 environment < batch.environmentCount;
                 ++environment) {
                expanded.insert(
                    expanded.end(),
                    world.defaultRodEdges().begin(),
                    world.defaultRodEdges().end()
                );
            }
            return expanded;
        };
    const std::vector<MRRodNodeStateGPU> initialRodNodes =
        uploadInitialState
        ? expandRodNodes(batch.initialRodNodes)
        : std::vector<MRRodNodeStateGPU>{};
    const std::vector<MRRodEdgeStateGPU> initialRodEdges =
        uploadInitialState
        ? expandRodEdges(batch.initialRodEdges)
        : std::vector<MRRodEdgeStateGPU>{};
    const std::vector<MRRodNodeStateGPU> resetRodNodes =
        uploadInitialState && !batch.resetMasks.empty()
        ? expandRodNodes(batch.resetRodNodes)
        : std::vector<MRRodNodeStateGPU>{};
    const std::vector<MRRodEdgeStateGPU> resetRodEdges =
        uploadInitialState && !batch.resetMasks.empty()
        ? expandRodEdges(batch.resetRodEdges)
        : std::vector<MRRodEdgeStateGPU>{};
    MRRodNodeStateGPU emptyRodNode{};
    MRRodEdgeStateGPU emptyRodEdge{};
    if (uploadInitialState) {
        copyToBuffer(
            context.buffers[kInitialRodNodes],
            initialRodNodes.empty()
                ? static_cast<const void*>(&emptyRodNode)
                : static_cast<const void*>(
                      initialRodNodes.data()
                  ),
            requirements.entries[kInitialRodNodes]
        );
        copyToBuffer(
            context.buffers[kInitialRodEdges],
            initialRodEdges.empty()
                ? static_cast<const void*>(&emptyRodEdge)
                : static_cast<const void*>(
                      initialRodEdges.data()
                  ),
            requirements.entries[kInitialRodEdges]
        );
        stagePrivateBuffer(
            context,
            kRodNodesA,
            initialRodNodes.empty()
                ? static_cast<const void*>(&emptyRodNode)
                : static_cast<const void*>(
                      initialRodNodes.data()
                  ),
            requirements.entries[kRodNodesA],
            stateUpload
        );
        stagePrivateBuffer(
            context,
            kRodEdgesA,
            initialRodEdges.empty()
                ? static_cast<const void*>(&emptyRodEdge)
                : static_cast<const void*>(
                      initialRodEdges.data()
                  ),
            requirements.entries[kRodEdgesA],
            stateUpload
        );
        stagePrivateBuffer(
            context,
            kResetRodNodes,
            resetRodNodes.empty()
                ? static_cast<const void*>(&emptyRodNode)
                : static_cast<const void*>(
                      resetRodNodes.data()
                  ),
            requirements.entries[kResetRodNodes],
            stateUpload
        );
        stagePrivateBuffer(
            context,
            kResetRodEdges,
            resetRodEdges.empty()
                ? static_cast<const void*>(&emptyRodEdge)
                : static_cast<const void*>(
                      resetRodEdges.data()
                  ),
            requirements.entries[kResetRodEdges],
            stateUpload
        );
    }
    if (stateUpload != nil) {
        [stateUpload endEncoding];
    }

    const std::array scratch{
        kWorkingEffort,
        kBodyWrenchPlaceholder,
        kCandidateAcceleration,
        kCandidateV,
        kCandidateQ,
        kABAStatuses,
        kStateQB,
        kStateVB,
        kObservations,
        kAccelerationTrajectory,
        kPublicStatuses,
        kEnvironmentStatuses,
        kCheckpointQ,
        kSceneBodiesB,
        kCheckpointSceneBodies,
        kBodyPoses,
        kFutureBodyPoses,
        kPointWorld,
        kFactorMatrix,
        kPointJacobians,
        kGeneralizedImpulse,
        kDeltaVelocity,
        kOperatorStatuses,
        kCurrentBodies,
        kCandidateBodies,
        kManifoldHeadersA,
        kManifoldPointsA,
        kManifoldCountsA,
        kManifoldHeadersB,
        kManifoldPointsB,
        kManifoldCountsB,
        kCandidateManifoldHeaders,
        kCandidateManifoldPoints,
        kCandidateManifoldCounts,
        kCheckpointManifoldHeaders,
        kCheckpointManifoldPoints,
        kCheckpointManifoldCounts,
        kCandidatePairs,
        kRawContacts,
        kRawPairIndices,
        kContacts,
        kIREndpoints,
        kIRBlocks,
        kIREndpoints,
        kIRRows,
        kIRCones,
        kPointQueries,
        kEvaluatedRows,
        kEvaluatedCones,
        kFactorCaches,
        kIslands,
        kResponseColumns,
        kContactStatuses,
        kPublicContactStatuses,
        kActiveIndirectDispatch,
        kProjectedColliders,
        kFutureProjectedColliders,
        kPairOverlapFlags,
        kWorkQueueHeaders,
        kPairWorkQueue,
        kPairRawCounts,
        kCompactionOffsets,
        kCompactionScratch,
        kCompactionFlags,
        kIslandWorkQueue,
        kContactTiles,
        kTileConstraintIndices,
        kWave32ImpulseDeltas,
        kWave32IslandStatuses,
        kConvexCaches,
        kCandidateConvexCaches,
        kCCDPairs,
        kPairRawContactStaging,
        kPairManifoldHeaders,
        kPairManifoldPoints,
        kManifoldIRScatter,
        kEndpointRuntime,
        kWave32Preconditioners,
        kIslandWorkDense,
        kQualityStatuses,
        kRodNodesB,
        kRodEdgesB,
        kRodInputPositions,
        kRodInputVelocities,
        kRodInputTwists,
        kRodInputTwistRates,
        kRodOutputPositions,
        kRodOutputVelocities,
        kRodOutputTwists,
        kRodOutputTwistRates,
        kRodStatuses,
        kRodReactions,
    };
    for (const std::size_t index : scratch) {
        zeroBuffer(
            context.buffers[index],
            requirements.entries[index]
        );
    }
    zeroBuffer(
        context.buffers[kCheckpointV],
        requirements.entries[kCheckpointV]
    );
}

bool encodeResidentStateInitialization(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const RequiredBuffers& requirements,
    const bool contactMode,
    const bool hasRods,
    const bool nativeTask
) {
    if (nativeTask) {
        const MRTaskEvidenceStateGPU initialEvidence{
            .controlSteps = 0u,
            .evidenceWindows = 0u,
            .completedEpisodeCount = 0u,
            .timeoutEpisodeCount = 0u,
            .impactContactCount = 0u,
            .impactCleanMissCount = 0u,
            .balanceFailureCount = 0u,
            .trackingScoreSum = 0.0f,
            .lastCompletedEpisodeCount = 0u,
            .reserved = 0u,
            .lastWindow = {},
        };
        copyToBuffer(
            context.uploadBuffers[kTaskEvidenceState],
            &initialEvidence,
            requirements.entries[kTaskEvidenceState]
        );
    }
    id<MTLBlitCommandEncoder> encoder =
        [commandBuffer blitCommandEncoder];
    if (encoder == nil) {
        return false;
    }
    encoder.label = @"MetalWorld initialize resident caches";
    const auto clear = [&](const std::size_t index) {
        [encoder
            fillBuffer:context.buffers[index]
                 range:NSMakeRange(
                     0u,
                     requirements.entries[index].allocationBytes
                 )
                 value:0u];
    };
    clear(kStateQB);
    clear(kStateVB);
    if (contactMode) {
        clear(kSceneBodiesB);
        clear(kManifoldHeadersA);
        clear(kManifoldPointsA);
        clear(kManifoldCountsA);
        clear(kManifoldHeadersB);
        clear(kManifoldPointsB);
        clear(kManifoldCountsB);
        clear(kConvexCaches);
        clear(kCandidateConvexCaches);
    }
    if (hasRods) {
        clear(kRodNodesB);
        clear(kRodEdgesB);
        clear(kRodWitnessesA);
        clear(kRodWitnessesB);
        clear(kCandidateRodWitnesses);
        clear(kCheckpointRodWitnesses);
        clear(kRodWitnessCounts);
    }
    if (nativeTask) {
        clear(kTaskState);
        clear(kTaskEvidenceState);
        clear(kTaskActionHistory);
        clear(kTaskActorHistory);
        clear(kTaskCleanHistory);
        clear(kTaskCriticHistory);
        clear(kTaskPreviousJointVelocity);
        clear(kTaskEncoderBias);
        clear(kTaskBodyParameters);
        clear(kTaskControllerParameters);
        clear(kTaskContactCompact);
        clear(kMulticopterStateA);
        clear(kMulticopterStateB);
        clear(kMulticopterCandidateState);
        [encoder
            copyFromBuffer:
                context.uploadBuffers[kTaskEvidenceState]
              sourceOffset:0u
                  toBuffer:
                context.buffers[kTaskEvidenceState]
         destinationOffset:0u
                     size:requirements
                              .entries[kTaskEvidenceState]
                              .logicalBytes];
    }
    [encoder endEncoding];
    return true;
}

NSUInteger worldThreadWidth(
    id<MTLComputePipelineState> pipeline
) {
    return std::max<NSUInteger>(
        1u,
        std::min<NSUInteger>(
            kWorldThreadsPerThreadgroup,
            pipeline.maxTotalThreadsPerThreadgroup
        )
    );
}

void dispatchWorldThreads(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    const std::size_t environmentCount
) {
    [encoder
        dispatchThreads:MTLSizeMake(
            static_cast<NSUInteger>(environmentCount),
            1u,
            1u
        )
        threadsPerThreadgroup:MTLSizeMake(
            worldThreadWidth(pipeline),
            1u,
            1u
    )];
}

struct MetalBufferBinding {
    NSUInteger argument = 0u;
    std::size_t buffer = 0u;
};

bool encodeContactThreadKernel(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    NSString* label,
    const std::initializer_list<MetalBufferBinding> bindings,
    const MRMetalWorldPassGPU* pass,
    const NSUInteger passArgument,
    const std::size_t environmentCount,
    const bool indirectDispatch = false,
    const NSUInteger indirectOffset = 0u,
    const void* secondaryBytes = nullptr,
    const NSUInteger secondaryLength = 0u,
    const NSUInteger secondaryArgument = 0u
) {
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        return false;
    }
    encoder.label = label;
    [encoder setComputePipelineState:pipeline];
    for (const MetalBufferBinding& binding : bindings) {
        [encoder setBuffer:context.buffers[binding.buffer]
                     offset:0u
                    atIndex:binding.argument];
    }
    if (pass != nullptr) {
        [encoder setBytes:pass
                   length:sizeof(*pass)
                  atIndex:passArgument];
    }
    if (secondaryBytes != nullptr && secondaryLength != 0u) {
        [encoder setBytes:secondaryBytes
                   length:secondaryLength
                  atIndex:secondaryArgument];
    }
    if (indirectDispatch) {
        [encoder
            dispatchThreadgroupsWithIndirectBuffer:
                context.buffers[kActiveIndirectDispatch]
            indirectBufferOffset:indirectOffset
            threadsPerThreadgroup:MTLSizeMake(
                kWorldThreadsPerThreadgroup,
                1u,
                1u
            )];
    } else {
        dispatchWorldThreads(
            encoder,
            pipeline,
            environmentCount
        );
    }
    [encoder endEncoding];
    return true;
}

bool encodeScanBlocks(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const std::size_t inputBuffer,
    const std::size_t outputBuffer,
    const MRScanLevelGPU& level
) {
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        return false;
    }
    encoder.label = @"MetalWorld hierarchical pair scan";
    [encoder setComputePipelineState:context.scanBlocksPipeline];
    [encoder setBuffer:context.buffers[inputBuffer]
                 offset:0u
                atIndex:0u];
    [encoder setBuffer:context.buffers[outputBuffer]
                 offset:0u
                atIndex:1u];
    [encoder setBuffer:context.buffers[kCompactionScratch]
                 offset:0u
                atIndex:2u];
    [encoder setBytes:&level
               length:sizeof(level)
              atIndex:3u];
    [encoder
        dispatchThreadgroups:MTLSizeMake(
            static_cast<NSUInteger>(level.blockCount),
            1u,
            1u
        )
        threadsPerThreadgroup:MTLSizeMake(
            MR_BROADPHASE_SCAN_BLOCK_SIZE,
            1u,
            1u
        )];
    [encoder endEncoding];
    return true;
}

bool encodeScanAdd(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const std::size_t outputBuffer,
    const MRScanLevelGPU& level
) {
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        return false;
    }
    encoder.label = @"MetalWorld hierarchical scan propagation";
    [encoder setComputePipelineState:context.scanAddPipeline];
    [encoder setBuffer:context.buffers[outputBuffer]
                 offset:0u
                atIndex:0u];
    [encoder setBuffer:context.buffers[kCompactionScratch]
                 offset:0u
                atIndex:1u];
    [encoder setBytes:&level
               length:sizeof(level)
              atIndex:2u];
    [encoder
        dispatchThreads:MTLSizeMake(
            static_cast<NSUInteger>(level.elementCount),
            1u,
            1u
        )
        threadsPerThreadgroup:MTLSizeMake(
            kWorldThreadsPerThreadgroup,
            1u,
            1u
        )];
    [encoder endEncoding];
    return true;
}

bool encodeStableBooleanScan(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const std::size_t inputBufferIndex,
    const std::size_t elementCountInput,
    const mr_u32 workClass
) {
    std::vector<MRScanLevelGPU> levels;
    if (elementCountInput != 0u) {
        std::size_t elementCount = elementCountInput;
        std::size_t scratchCursor = 0u;
        std::size_t inputOffset = 0u;
        std::size_t inputBuffer = inputBufferIndex;
        std::size_t outputBuffer = kCompactionOffsets;
        while (true) {
            const std::size_t blockCount =
                (elementCount +
                 MR_BROADPHASE_SCAN_BLOCK_SIZE - 1u) /
                MR_BROADPHASE_SCAN_BLOCK_SIZE;
            if (elementCount >
                    std::numeric_limits<mr_u32>::max() ||
                blockCount >
                    std::numeric_limits<mr_u32>::max() ||
                inputOffset >
                    std::numeric_limits<mr_u32>::max() ||
                scratchCursor >
                    std::numeric_limits<mr_u32>::max()) {
                return false;
            }
            MRScanLevelGPU level{};
            level.elementCount =
                static_cast<mr_u32>(elementCount);
            level.blockCount =
                static_cast<mr_u32>(blockCount);
            level.inputOffset =
                static_cast<mr_u32>(inputOffset);
            level.outputOffset =
                levels.empty()
                ? 0u
                : static_cast<mr_u32>(scratchCursor);
            if (!levels.empty()) {
                scratchCursor += elementCount;
            }
            level.blockSumOffset =
                static_cast<mr_u32>(scratchCursor);
            level.workClass = workClass;
            level.flags = levels.empty()
                ? static_cast<mr_u32>(MR_SCAN_BOOLEAN_INPUT)
                : 0u;
            scratchCursor += blockCount;
            levels.push_back(level);
            if (!encodeScanBlocks(
                    context,
                    commandBuffer,
                    inputBuffer,
                    outputBuffer,
                    level
                )) {
                return false;
            }
            if (blockCount <= 1u) {
                break;
            }
            inputBuffer = kCompactionScratch;
            outputBuffer = kCompactionScratch;
            inputOffset = level.blockSumOffset;
            elementCount = blockCount;
        }
        for (std::size_t levelIndex = levels.size();
             levelIndex-- > 1u;) {
            MRScanLevelGPU child = levels[levelIndex - 1u];
            child.parentOffset =
                levels[levelIndex].outputOffset;
            if (!encodeScanAdd(
                    context,
                    commandBuffer,
                    levelIndex == 1u
                        ? kCompactionOffsets
                        : kCompactionScratch,
                    child
                )) {
                return false;
            }
        }
    }
    return true;
}

[[maybe_unused]] bool encodeCompactedPairNarrowphase(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const std::size_t pairElementCount,
    const bool hasConvexPairs,
    const bool hasMeshPairs
) {
    if (!encodeStableBooleanScan(
            context,
            commandBuffer,
            kPairOverlapFlags,
            pairElementCount,
            MR_WORLD_WORK_ANALYTIC
        )) {
        return false;
    }
    id<MTLComputeCommandEncoder> scatter =
        [commandBuffer computeCommandEncoder];
    if (scatter == nil) {
        return false;
    }
    scatter.label = @"MetalWorld stable compact pair scatter";
    [scatter
        setComputePipelineState:
            context.pairQueueScatterPipeline];
    const std::array<std::size_t, 7u> scatterBuffers{{
        kContactDispatch,
        kShapes,
        kEligiblePairs,
        kPairOverlapFlags,
        kCompactionOffsets,
        kPairWorkQueue,
        kWorkQueueHeaders,
    }};
    for (NSUInteger argument = 0u;
         argument < scatterBuffers.size();
         ++argument) {
        [scatter setBuffer:context.buffers[
                               scatterBuffers[argument]
                           ]
                    offset:0u
                   atIndex:argument];
    }
    [scatter
        dispatchThreads:MTLSizeMake(
            static_cast<NSUInteger>(
                std::max<std::size_t>(pairElementCount, 1u)
            ),
            1u,
            1u
        )
        threadsPerThreadgroup:MTLSizeMake(
            kWorldThreadsPerThreadgroup,
            1u,
            1u
        )];
    [scatter endEncoding];

    id<MTLComputeCommandEncoder> narrowphase =
        [commandBuffer computeCommandEncoder];
    if (narrowphase == nil) {
        return false;
    }
    narrowphase.label =
        @"MetalWorld compact SIMD32 narrowphase";
    [narrowphase
        setComputePipelineState:
            context.pairNarrowphasePipeline];
    const std::array<std::size_t, 8u> narrowphaseBuffers{{
        kContactDispatch,
        kShapes,
        kEligiblePairs,
        kProjectedColliders,
        kPairWorkQueue,
        kWorkQueueHeaders,
        kPairRawContactStaging,
        kPairRawCounts,
    }};
    for (NSUInteger argument = 0u;
         argument < narrowphaseBuffers.size();
         ++argument) {
        [narrowphase setBuffer:context.buffers[
                                   narrowphaseBuffers[argument]
                               ]
                        offset:0u
                       atIndex:argument];
    }
    [narrowphase
        dispatchThreadgroupsWithIndirectBuffer:
            context.buffers[kWorkQueueHeaders]
        indirectBufferOffset:
            MR_WORLD_WORK_ANALYTIC *
                sizeof(MRWorkQueueHeaderGPU) +
            offsetof(MRWorkQueueHeaderGPU, indirect)
        threadsPerThreadgroup:MTLSizeMake(
            kWorldThreadsPerThreadgroup,
            1u,
            1u
        )];
    [narrowphase endEncoding];
    if (hasConvexPairs) {
        id<MTLComputeCommandEncoder> convex =
            [commandBuffer computeCommandEncoder];
        if (convex == nil) {
            return false;
        }
        convex.label =
            @"MetalWorld certified convex GJK MPR EPA";
        [convex
            setComputePipelineState:
                context.convexNarrowphasePipeline];
        const std::array<std::size_t, 11u> convexBuffers{{
            kContactDispatch,
            kShapes,
            kEligiblePairs,
            kProjectedColliders,
            kPairWorkQueue,
            kWorkQueueHeaders,
            kGeometryHeaders,
            kGeometryVertices,
            kPairRawContactStaging,
            kPairRawCounts,
            kConvexCaches,
        }};
        for (NSUInteger argument = 0u;
             argument < convexBuffers.size();
             ++argument) {
            [convex setBuffer:context.buffers[
                                  convexBuffers[argument]
                              ]
                       offset:0u
                      atIndex:argument];
        }
        [convex
            dispatchThreadgroupsWithIndirectBuffer:
                context.buffers[kWorkQueueHeaders]
            indirectBufferOffset:
                MR_WORLD_WORK_ANALYTIC *
                    sizeof(MRWorkQueueHeaderGPU) +
                offsetof(MRWorkQueueHeaderGPU, indirect)
            threadsPerThreadgroup:MTLSizeMake(
                kWorldThreadsPerThreadgroup,
                1u,
                1u
            )];
        [convex endEncoding];
    }
    if (hasMeshPairs) {
        id<MTLComputeCommandEncoder> mesh =
            [commandBuffer computeCommandEncoder];
        if (mesh == nil) {
            return false;
        }
        mesh.label =
            @"MetalWorld stackless BVH4 mesh narrowphase";
        [mesh
            setComputePipelineState:
                context.meshNarrowphasePipeline];
        const std::array<std::size_t, 13u> meshBuffers{{
            kContactDispatch,
            kShapes,
            kEligiblePairs,
            kProjectedColliders,
            kPairWorkQueue,
            kWorkQueueHeaders,
            kGeometryHeaders,
            kGeometryVertices,
            kMeshBvhNodes,
            kMeshTriangles,
            kPairRawContactStaging,
            kPairRawCounts,
            kConvexCaches,
        }};
        for (NSUInteger argument = 0u;
             argument < meshBuffers.size();
             ++argument) {
            [mesh setBuffer:context.buffers[
                                meshBuffers[argument]
                            ]
                     offset:0u
                    atIndex:argument];
        }
        [mesh
            dispatchThreadgroupsWithIndirectBuffer:
                context.buffers[kWorkQueueHeaders]
            indirectBufferOffset:
                MR_WORLD_WORK_ANALYTIC *
                    sizeof(MRWorkQueueHeaderGPU) +
                offsetof(MRWorkQueueHeaderGPU, indirect)
            threadsPerThreadgroup:MTLSizeMake(
                kWorldThreadsPerThreadgroup,
                1u,
                1u
            )];
        [mesh endEncoding];
    }
    return true;
}

bool encodeClassCompactedPairNarrowphase(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const std::size_t pairElementCount,
    const mr_u32 activeClassMask
) {
    constexpr mr_u32 kFirstPairClass =
        MR_WORLD_WORK_ANALYTIC;
    constexpr mr_u32 kLastPairClass =
        MR_WORLD_WORK_MESH;
    for (mr_u32 workClass = kFirstPairClass;
         workClass <= kLastPairClass;
         ++workClass) {
        if ((activeClassMask & (1u << workClass)) == 0u) {
            continue;
        }
        const mr_uint4 classConfig{
            workClass,
            activeClassMask,
            0u,
            0u,
        };
        id<MTLComputeCommandEncoder> classify =
            [commandBuffer computeCommandEncoder];
        if (classify == nil) {
            return false;
        }
        classify.label =
            @"MetalWorld pair algorithm classification";
        [classify
            setComputePipelineState:
                context.pairClassFlagPipeline];
        const std::array<std::size_t, 5u> classifyBuffers{{
            kContactDispatch,
            kShapes,
            kEligiblePairs,
            kPairOverlapFlags,
            kCompactionFlags,
        }};
        for (NSUInteger argument = 0u;
             argument < classifyBuffers.size();
             ++argument) {
            [classify setBuffer:context.buffers[
                                    classifyBuffers[argument]
                                ]
                         offset:0u
                        atIndex:argument];
        }
        [classify setBytes:&classConfig
                    length:sizeof(classConfig)
                   atIndex:5u];
        [classify
            dispatchThreads:MTLSizeMake(
                static_cast<NSUInteger>(
                    std::max<std::size_t>(
                        pairElementCount,
                        1u
                    )
                ),
                1u,
                1u
            )
            threadsPerThreadgroup:MTLSizeMake(
                kWorldThreadsPerThreadgroup,
                1u,
                1u
            )];
        [classify endEncoding];
        if (!encodeStableBooleanScan(
                context,
                commandBuffer,
                kCompactionFlags,
                pairElementCount,
                workClass
            )) {
            return false;
        }

        id<MTLComputeCommandEncoder> scatter =
            [commandBuffer computeCommandEncoder];
        if (scatter == nil) {
            return false;
        }
        scatter.label =
            @"MetalWorld stable class pair scatter";
        [scatter
            setComputePipelineState:
                context.pairQueueScatterPipeline];
        const std::array<std::size_t, 7u> buffers{{
            kContactDispatch,
            kShapes,
            kEligiblePairs,
            kCompactionFlags,
            kCompactionOffsets,
            kPairWorkQueue,
            kWorkQueueHeaders,
        }};
        for (NSUInteger argument = 0u;
             argument < buffers.size();
             ++argument) {
            [scatter setBuffer:context.buffers[
                                   buffers[argument]
                               ]
                        offset:0u
                       atIndex:argument];
        }
        [scatter setBytes:&classConfig
                   length:sizeof(classConfig)
                  atIndex:7u];
        [scatter
            dispatchThreads:MTLSizeMake(
                static_cast<NSUInteger>(
                    std::max<std::size_t>(
                        pairElementCount,
                        1u
                    )
                ),
                1u,
                1u
            )
            threadsPerThreadgroup:MTLSizeMake(
                kWorldThreadsPerThreadgroup,
                1u,
                1u
            )];
        [scatter endEncoding];
    }

    for (mr_u32 workClass = kFirstPairClass;
         workClass <= kLastPairClass;
         ++workClass) {
        if ((activeClassMask & (1u << workClass)) == 0u) {
            continue;
        }
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return false;
        }
        const mr_uint4 classConfig{
            workClass,
            activeClassMask,
            0u,
            0u,
        };
        NSUInteger classConfigIndex = 0u;
        if (workClass == MR_WORLD_WORK_ANALYTIC ||
            workClass == MR_WORLD_WORK_SAT_CLIP) {
            encoder.label =
                @"MetalWorld compact analytic/SAT queue";
            [encoder
                setComputePipelineState:
                    context.pairNarrowphasePipeline];
            const std::array<std::size_t, 8u> buffers{{
                kContactDispatch,
                kShapes,
                kEligiblePairs,
                kProjectedColliders,
                kPairWorkQueue,
                kWorkQueueHeaders,
                kPairRawContactStaging,
                kPairRawCounts,
            }};
            for (NSUInteger argument = 0u;
                 argument < buffers.size();
                 ++argument) {
                [encoder setBuffer:context.buffers[
                                       buffers[argument]
                                   ]
                            offset:0u
                           atIndex:argument];
            }
            classConfigIndex = 8u;
        } else if (
            workClass == MR_WORLD_WORK_PRIMITIVE_GJK ||
            workClass == MR_WORLD_WORK_HULL_GJK ||
            workClass == MR_WORLD_WORK_HARD_CONVEX
        ) {
            encoder.label =
                workClass == MR_WORLD_WORK_HULL_GJK
                ? @"MetalWorld compact authored-hull MPR queue"
                : @"MetalWorld certified convex GJK MPR EPA queue";
            [encoder
                setComputePipelineState:
                    workClass == MR_WORLD_WORK_HULL_GJK
                    ? context.hullNarrowphasePipeline
                    : context.convexNarrowphasePipeline];
            const std::array<std::size_t, 12u> buffers{{
                kContactDispatch,
                kShapes,
                kEligiblePairs,
                kProjectedColliders,
                kPairWorkQueue,
                kWorkQueueHeaders,
                kGeometryHeaders,
                kGeometryVertices,
                kPairRawContactStaging,
                kPairRawCounts,
                kConvexCaches,
                kCandidateConvexCaches,
            }};
            for (NSUInteger argument = 0u;
                 argument < buffers.size();
                 ++argument) {
                [encoder setBuffer:context.buffers[
                                       buffers[argument]
                                   ]
                            offset:0u
                           atIndex:argument];
            }
            classConfigIndex = 12u;
        } else {
            encoder.label =
                @"MetalWorld compact stackless mesh queue";
            [encoder
                setComputePipelineState:
                    context.meshNarrowphasePipeline];
            const std::array<std::size_t, 13u> buffers{{
                kContactDispatch,
                kShapes,
                kEligiblePairs,
                kProjectedColliders,
                kPairWorkQueue,
                kWorkQueueHeaders,
                kGeometryHeaders,
                kGeometryVertices,
                kMeshBvhNodes,
                kMeshTriangles,
                kPairRawContactStaging,
                kPairRawCounts,
                kCandidateConvexCaches,
            }};
            for (NSUInteger argument = 0u;
                 argument < buffers.size();
                 ++argument) {
                [encoder setBuffer:context.buffers[
                                       buffers[argument]
                                   ]
                            offset:0u
                           atIndex:argument];
            }
            classConfigIndex = 13u;
        }
        [encoder setBytes:&classConfig
                   length:sizeof(classConfig)
                  atIndex:classConfigIndex];
        [encoder
            dispatchThreadgroupsWithIndirectBuffer:
                context.buffers[kWorkQueueHeaders]
            indirectBufferOffset:
                workClass * sizeof(MRWorkQueueHeaderGPU) +
                offsetof(MRWorkQueueHeaderGPU, indirect)
            threadsPerThreadgroup:MTLSizeMake(
                kWorldThreadsPerThreadgroup,
                1u,
                1u
            )];
        [encoder endEncoding];
    }
    return true;
}

bool encodeArticulatedOperator(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const std::size_t dispatchBuffer,
    const std::size_t qBuffer,
    const std::size_t pointBuffer,
    const std::size_t bodyPoseBuffer,
    const std::size_t environmentCount,
    NSString* label,
    const bool indirectDispatch
) {
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        return false;
    }
    encoder.label = label;
    id<MTLComputePipelineState> pipeline =
        context.useTaskBodyParameters
        ? context.parameterizedOperatorPipeline
        : context.operatorPipeline;
    [encoder setComputePipelineState:pipeline];
    const std::array<std::size_t, 15u> buffers{{
        kWorld,
        kArticulations,
        kJoints,
        kDofs,
        kBodies,
        dispatchBuffer,
        qBuffer,
        pointBuffer,
        bodyPoseBuffer,
        kPointWorld,
        kFactorMatrix,
        kPointJacobians,
        kGeneralizedImpulse,
        kDeltaVelocity,
        kOperatorStatuses,
    }};
    const MTLSize threadgroupSize = MTLSizeMake(
        kOperatorThreadsPerThreadgroup,
        1u,
        1u
    );
    if (context.useTaskBodyParameters) {
        [encoder setBuffer:context.buffers[kTaskBodyParameters]
                     offset:0u
                    atIndex:15u];
        [encoder setBuffer:
                     context.buffers[kTaskControllerParameters]
                     offset:0u
                    atIndex:16u];
    }

    // Kinematics is a composed world operation: encode one dispatch for every
    // cooked articulation, writing into disjoint slices of a world-global
    // pose tensor and an articulation-major status tensor. This preserves the
    // existing optimized single-articulation operator while eliminating the
    // old primary-articulation projection boundary.
    if (dispatchBuffer == kOperatorKinematicsDispatch) {
        if (indirectDispatch ||
            context.boundArticulations.empty()) {
            [encoder endEncoding];
            return false;
        }
        for (std::size_t owner = 0u;
             owner < context.boundArticulations.size();
             ++owner) {
            const MRArticulationGPU& articulation =
                context.boundArticulations[owner];
            for (NSUInteger argument = 0u;
                 argument < buffers.size();
                 ++argument) {
                NSUInteger offset = 0u;
                if (argument == 5u) {
                    offset = owner *
                        sizeof(
                            MRArticulatedOperatorDispatchGPU
                        );
                } else if (argument == 6u) {
                    offset = articulation.qOffset *
                        sizeof(float);
                } else if (argument == 8u) {
                    offset = articulation.firstBody *
                        sizeof(MRArticulatedBodyPoseGPU);
                } else if (argument == 14u) {
                    offset =
                        owner * environmentCount *
                        sizeof(
                            MRArticulatedOperatorStatusGPU
                        );
                }
                [encoder
                    setBuffer:context.buffers[
                                  buffers[argument]
                              ]
                       offset:offset
                      atIndex:argument];
            }
            [encoder
                setThreadgroupMemoryLength:
                    detail::articulatedOperatorThreadgroupBytes(
                        articulation.bodyCount,
                        articulation.nv
                    )
                atIndex:0u];
            [encoder
                dispatchThreadgroups:MTLSizeMake(
                    static_cast<NSUInteger>(
                        environmentCount
                    ),
                    1u,
                    1u
                )
                threadsPerThreadgroup:threadgroupSize];
        }
        [encoder endEncoding];
        return true;
    }

    if (dispatchBuffer == kOperatorFactorDispatch &&
        context.boundArticulations.size() > 1u) {
        if (context.boundFactorDispatches.size() !=
                context.boundArticulations.size()) {
            [encoder endEncoding];
            return false;
        }
        std::size_t factorPrefixElements = 0u;
        std::size_t jacobianPrefixElements = 0u;
        for (std::size_t owner = 0u;
             owner < context.boundArticulations.size();
             ++owner) {
            const MRArticulationGPU& articulation =
                context.boundArticulations[owner];
            const MRArticulatedOperatorDispatchGPU& local =
                context.boundFactorDispatches[owner];
            for (NSUInteger argument = 0u;
                 argument < buffers.size();
                 ++argument) {
                std::size_t resource = buffers[argument];
                NSUInteger offset = 0u;
                if (argument == 5u) {
                    offset = owner *
                        sizeof(
                            MRArticulatedOperatorDispatchGPU
                        );
                } else if (argument == 6u) {
                    offset = articulation.qOffset *
                        sizeof(float);
                } else if (argument == 7u) {
                    offset =
                        owner * environmentCount *
                        static_cast<std::size_t>(
                            local.pointStride
                        ) *
                        sizeof(MRArticulatedPointImpulseGPU);
                } else if (argument == 8u) {
                    offset = articulation.firstBody *
                        sizeof(MRArticulatedBodyPoseGPU);
                } else if (argument == 10u) {
                    resource = kFactorMatrixStaging;
                    offset = factorPrefixElements *
                        sizeof(float);
                } else if (argument == 11u) {
                    resource = kPointJacobiansStaging;
                    offset = jacobianPrefixElements *
                        sizeof(float);
                } else if (
                    argument == 12u ||
                    argument == 13u
                ) {
                    offset = articulation.vOffset *
                        sizeof(float);
                } else if (argument == 14u) {
                    offset =
                        owner * environmentCount *
                        sizeof(
                            MRArticulatedOperatorStatusGPU
                        );
                }
                [encoder setBuffer:context.buffers[resource]
                             offset:offset
                            atIndex:argument];
            }
            [encoder
                setThreadgroupMemoryLength:
                    detail::articulatedOperatorThreadgroupBytes(
                        articulation.bodyCount,
                        articulation.nv
                    )
                atIndex:0u];
            [encoder
                dispatchThreadgroups:MTLSizeMake(
                    static_cast<NSUInteger>(
                        environmentCount
                    ),
                    1u,
                    1u
                )
                threadsPerThreadgroup:threadgroupSize];
            factorPrefixElements +=
                environmentCount *
                static_cast<std::size_t>(articulation.nv) *
                articulation.nv;
            jacobianPrefixElements +=
                environmentCount *
                static_cast<std::size_t>(
                    local.pointJacobianStride
                );
        }
        [encoder endEncoding];

        id<MTLComputeCommandEncoder> compose =
            [commandBuffer computeCommandEncoder];
        if (compose == nil) {
            return false;
        }
        compose.label =
            @"MetalWorld compose multi-articulation factor/Jacobians";
        [compose
            setComputePipelineState:
                context.multiOperatorComposePipeline];
        const std::array<std::size_t, 9u> composeBuffers{{
            kContactDispatch,
            kArticulations,
            kOperatorFactorDispatch,
            kOperatorStatuses,
            kFactorMatrixStaging,
            kPointJacobiansStaging,
            kFactorMatrix,
            kPointJacobians,
            kContactStatuses,
        }};
        for (NSUInteger argument = 0u;
             argument < composeBuffers.size();
             ++argument) {
            [compose
                setBuffer:context.buffers[
                              composeBuffers[argument]
                          ]
                   offset:0u
                  atIndex:argument];
        }
        dispatchWorldThreads(
            compose,
            context.multiOperatorComposePipeline,
            environmentCount
        );
        [compose endEncoding];
        return true;
    }

    for (NSUInteger argument = 0u;
         argument < buffers.size();
         ++argument) {
        [encoder setBuffer:context.buffers[buffers[argument]]
                     offset:0u
                    atIndex:argument];
    }
    if (context.boundArticulations.empty()) {
        [encoder endEncoding];
        return false;
    }
    const MRArticulationGPU& articulation =
        context.boundArticulations.front();
    [encoder
        setThreadgroupMemoryLength:
            detail::articulatedOperatorThreadgroupBytes(
                articulation.bodyCount,
                articulation.nv
            )
        atIndex:0u];
    if (indirectDispatch) {
        [encoder
            dispatchThreadgroupsWithIndirectBuffer:
                context.buffers[kActiveIndirectDispatch]
            indirectBufferOffset:0u
            threadsPerThreadgroup:threadgroupSize];
    } else {
        [encoder
            dispatchThreadgroups:MTLSizeMake(
                static_cast<NSUInteger>(environmentCount),
                1u,
                1u
            )
            threadsPerThreadgroup:threadgroupSize];
    }
    [encoder endEncoding];
    return true;
}

bool encodeStreamedArticulatedResponses(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const std::size_t sourceQ,
    const std::size_t environmentCount
) {
    const MRMetalWorldContactDispatchGPU& contact =
        context.boundContactDispatch;
    if ((contact.flags &
         MR_METAL_WORLD_CONTACT_STREAMED_RESPONSES) == 0u) {
        return true;
    }
    if (context.boundArticulations.size() != 1u ||
        context.boundFactorDispatches.size() != 1u ||
        !context.useTaskBodyParameters) {
        return false;
    }
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        return false;
    }
    encoder.label = @"MetalWorld streamed articulated responses";
    [encoder setComputePipelineState:context.streamedInversePipeline];
    const std::array<std::size_t, 26u> buffers{{
        kWorld,
        kArticulations,
        kJoints,
        kDofs,
        kBodies,
        kInverseMassDispatch,
        sourceQ,
        kResponseColumns,
        kResponseColumns,
        kInverseMassStatuses,
        kParallelScheduleArticulations,
        kParallelScheduleLevels,
        kParallelScheduleParentReductions,
        kParallelScheduleLevelBodies,
        kParallelScheduleParentLocal,
        kParallelScheduleInboundJoint,
        kParallelScheduleChildOffsets,
        kParallelScheduleChildIndices,
        kTaskBodyParameters,
        kTaskControllerParameters,
        kContactStatuses,
        kContactDispatch,
        kPointJacobians,
        kCandidateBodies,
        kContacts,
        kEvaluatedRows,
    }};
    for (NSUInteger argument = 0u;
         argument < buffers.size();
         ++argument) {
        NSUInteger offset = 0u;
        if (argument == 6u) {
            offset = context.boundArticulations.front().qOffset *
                sizeof(float);
        }
        [encoder setBuffer:context.buffers[buffers[argument]]
                    offset:offset
                   atIndex:argument];
    }
    [encoder
        dispatchThreadgroups:MTLSizeMake(environmentCount, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder endEncoding];
    return true;
}

bool encodePrepare(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t sourceQ,
    const std::size_t sourceV,
    const std::size_t environmentCount
) {
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        return false;
    }
    encoder.label = @"MetalWorld prepare/reset";
    [encoder setComputePipelineState:context.preparePipeline];
    [encoder setBuffer:context.buffers[kWorldDispatch]
                 offset:0u
                atIndex:0u];
    [encoder setBytes:&pass length:sizeof(pass) atIndex:1u];
    [encoder setBuffer:context.buffers[kEffortTrajectory]
                 offset:0u
                atIndex:2u];
    [encoder setBuffer:context.buffers[kResetMasks]
                 offset:0u
                atIndex:3u];
    [encoder setBuffer:context.buffers[kResetQ]
                 offset:0u
                atIndex:4u];
    [encoder setBuffer:context.buffers[kResetV]
                 offset:0u
                atIndex:5u];
    [encoder setBuffer:context.buffers[sourceQ]
                 offset:0u
                atIndex:6u];
    [encoder setBuffer:context.buffers[sourceV]
                 offset:0u
                atIndex:7u];
    [encoder setBuffer:context.buffers[kCheckpointQ]
                 offset:0u
                atIndex:8u];
    [encoder setBuffer:context.buffers[kCheckpointV]
                 offset:0u
                atIndex:9u];
    [encoder setBuffer:context.buffers[kWorkingEffort]
                 offset:0u
                atIndex:10u];
    [encoder setBuffer:context.buffers[kEnvironmentStatuses]
                 offset:0u
                atIndex:11u];
    [encoder setBuffer:context.buffers[kWorld]
                 offset:0u
                atIndex:12u];
    [encoder setBuffer:context.buffers[kArticulations]
                 offset:0u
                atIndex:13u];
    [encoder setBuffer:context.buffers[kDofs]
                 offset:0u
                atIndex:14u];
    [encoder setBuffer:context.buffers[kActuatorProfiles]
                 offset:0u
                atIndex:15u];
    [encoder setBuffer:context.buffers[kTaskControllerParameters]
                 offset:0u
                atIndex:16u];
    dispatchWorldThreads(
        encoder,
        context.preparePipeline,
        environmentCount
    );
    [encoder endEncoding];
    return true;
}

bool encodeDriveRefresh(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t sourceQ,
    const std::size_t sourceV,
    const std::size_t environmentCount
) {
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        return false;
    }
    encoder.label = @"MetalWorld microstep drive refresh";
    [encoder setComputePipelineState:context.driveRefreshPipeline];
    [encoder setBuffer:context.buffers[kWorldDispatch]
                 offset:0u
                atIndex:0u];
    [encoder setBytes:&pass length:sizeof(pass) atIndex:1u];
    [encoder setBuffer:context.buffers[kEffortTrajectory]
                 offset:0u
                atIndex:2u];
    [encoder setBuffer:context.buffers[sourceQ]
                 offset:0u
                atIndex:3u];
    [encoder setBuffer:context.buffers[sourceV]
                 offset:0u
                atIndex:4u];
    [encoder setBuffer:context.buffers[kWorkingEffort]
                 offset:0u
                atIndex:5u];
    [encoder setBuffer:context.buffers[kDofs]
                 offset:0u
                atIndex:6u];
    [encoder setBuffer:context.buffers[kActuatorProfiles]
                 offset:0u
                atIndex:7u];
    [encoder setBuffer:context.buffers[kTaskControllerParameters]
                 offset:0u
                atIndex:8u];
    [encoder setBuffer:context.buffers[kWorld]
                 offset:0u
                atIndex:9u];
    dispatchWorldThreads(
        encoder,
        context.driveRefreshPipeline,
        environmentCount
    );
    [encoder endEncoding];
    return true;
}

bool encodeTaskObserve(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t sourceQ,
    const std::size_t sourceV,
    const std::size_t sourceScene,
    const std::size_t environmentCount
) {
    return encodeContactThreadKernel(
        context,
        commandBuffer,
        context.taskObservePipeline,
        @"compiled task reset/observation",
        {
            {0u, kTaskDispatch},
            {1u, kTaskProgramHeader},
            {2u, kTaskProgramArena},
            {3u, kWorldDispatch},
            {6u, kResetMasks},
            {7u, kResetQ},
            {8u, kResetV},
            {9u, kResetSceneBodies},
            {10u, sourceQ},
            {11u, sourceV},
            {12u, kInitialSceneBodies},
            {13u, kTaskCriticHistory},
            {15u, sourceScene},
            {16u, kTaskDefaultQ},
            {17u, kTaskState},
            {18u, kTaskActionHistory},
            {19u, kTaskActorHistory},
            {20u, kTaskCleanHistory},
            {21u, kTaskPreviousJointVelocity},
            {22u, kTaskEncoderBias},
            {23u, kTaskBodyParameters},
            {24u, kTaskControllerParameters},
            {25u, kTaskActorObservations},
            {26u, kTaskCriticObservations},
            {27u, kTaskContactCompact},
            {28u, kShapes},
            {29u, kGeometryHeaders},
            {30u, kGeometryVertices},
            {5u, kTaskEvidenceState},
        },
        &pass,
        4u,
        environmentCount
    );
}

bool encodeDeviceObservationBodies(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t sourceQ,
    const std::size_t observationQ,
    const std::size_t sourceScene,
    const std::size_t observationScene,
    const std::size_t environmentCount
) {
    return encodeContactThreadKernel(
               context,
               commandBuffer,
               context.observationStateSelectPipeline,
               @"MetalWorld observation-time state selection",
               {
                   {0u, kWorldDispatch},
                   {1u, kContactDispatch},
                   {3u, kResetMasks},
                   {4u, kResetQ},
                   {5u, kResetSceneBodies},
                   {6u, sourceQ},
                   {7u, sourceScene},
                   {8u, observationQ},
                   {9u, observationScene},
                   {10u, kContactStatuses},
               },
               &pass,
               2u,
               environmentCount
           ) &&
        encodeArticulatedOperator(
            context,
            commandBuffer,
            kOperatorKinematicsDispatch,
            observationQ,
            kPointQueries,
            kBodyPoses,
            environmentCount,
            @"MetalWorld observation-time articulation kinematics",
            false
        ) &&
        encodeContactThreadKernel(
            context,
            commandBuffer,
            context.bodyProjectionPipeline,
            @"MetalWorld observation-time global body projection",
            {
                {0u, kContactDispatch},
                {1u, kArticulations},
                {2u, kBodies},
                {3u, kSceneBodyIndices},
                {4u, kBodyPoses},
                {5u, kOperatorStatuses},
                {6u, observationScene},
                {7u, kCurrentBodies},
                {8u, kContactStatuses},
            },
            nullptr,
            0u,
            environmentCount
        );
}

bool encodeTaskThreatSelect(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t environmentCount
) {
    return encodeContactThreadKernel(
        context,
        commandBuffer,
        context.taskThreatSelectPipeline,
        @"compiled task predictive threat selection",
        {
            {0u, kTaskDispatch},
            {1u, kTaskProgramHeader},
            {2u, kTaskProgramArena},
            {3u, kContactDispatch},
            {5u, kCurrentBodies},
            {6u, kTaskState},
            {7u, kPointQueries},
        },
        &pass,
        4u,
        environmentCount
    );
}

bool encodeTaskThreatJacobians(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const std::size_t qBuffer,
    const std::size_t environmentCount
) {
    if (context.boundArticulations.size() != 1u ||
        context.boundFactorDispatches.size() != 1u) {
        return false;
    }
    MRArticulatedOperatorDispatchGPU dispatch =
        context.boundFactorDispatches.front();
    dispatch.pointCount = 1u;
    dispatch.flags =
        MR_ARTICULATED_OPERATOR_KINEMATICS_JACOBIANS_ONLY;
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        return false;
    }
    encoder.label = @"compiled task privileged threat Jacobian";
    id<MTLComputePipelineState> pipeline =
        context.useTaskBodyParameters
        ? context.parameterizedOperatorPipeline
        : context.operatorPipeline;
    [encoder setComputePipelineState:pipeline];
    const std::array<std::size_t, 15u> buffers{{
        kWorld,
        kArticulations,
        kJoints,
        kDofs,
        kBodies,
        kOperatorFactorDispatch,
        qBuffer,
        kPointQueries,
        kBodyPoses,
        kPointWorld,
        kFactorMatrix,
        kPointJacobians,
        kGeneralizedImpulse,
        kDeltaVelocity,
        kOperatorStatuses,
    }};
    for (NSUInteger argument = 0u;
         argument < buffers.size();
         ++argument) {
        if (argument == 5u) {
            [encoder setBytes:&dispatch
                       length:sizeof(dispatch)
                      atIndex:argument];
        } else {
            [encoder setBuffer:context.buffers[buffers[argument]]
                         offset:0u
                        atIndex:argument];
        }
    }
    if (context.useTaskBodyParameters) {
        [encoder setBuffer:context.buffers[kTaskBodyParameters]
                     offset:0u
                    atIndex:15u];
        [encoder setBuffer:context.buffers[kTaskControllerParameters]
                     offset:0u
                    atIndex:16u];
    }
    const MRArticulationGPU& articulation =
        context.boundArticulations.front();
    [encoder setThreadgroupMemoryLength:
                 detail::articulatedOperatorThreadgroupBytes(
                     articulation.bodyCount,
                     articulation.nv
                 )
                               atIndex:0u];
    [encoder dispatchThreadgroups:MTLSizeMake(
                                        environmentCount,
                                        1u,
                                        1u
                                    )
             threadsPerThreadgroup:MTLSizeMake(
                                        kOperatorThreadsPerThreadgroup,
                                        1u,
                                        1u
                                    )];
    [encoder endEncoding];
    return true;
}

bool encodeTaskJointCbf(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t qState,
    const std::size_t vState,
    const std::size_t environmentCount
) {
    return encodeContactThreadKernel(
        context,
        commandBuffer,
        context.taskJointCbfPipeline,
        @"compiled task privileged Joint-CBF teacher",
        {
            {0u, kTaskDispatch},
            {1u, kTaskProgramHeader},
            {2u, kTaskProgramArena},
            {3u, kContactDispatch},
            {5u, qState},
            {6u, vState},
            {7u, kTaskActions},
            {8u, kTaskDefaultQ},
            {9u, kCurrentBodies},
            {10u, kPointJacobians},
            {11u, kOperatorStatuses},
            {12u, kTaskState},
        },
        &pass,
        4u,
        environmentCount
    );
}

bool encodeTaskMotionFeatures(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t environmentCount
) {
    return encodeContactThreadKernel(
        context,
        commandBuffer,
        context.taskMotionPipeline,
        @"compiled task tracked-link motion features",
        {
            {0u, kTaskDispatch},
            {1u, kTaskProgramHeader},
            {2u, kTaskProgramArena},
            {3u, kContactDispatch},
            {5u, kCurrentBodies},
            {6u, kTaskMotionFeatures},
        },
        &pass,
        4u,
        environmentCount
    );
}

bool encodePolicyInference(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const CompiledPolicyProgram& program,
    const MRMetalWorldPassGPU& pass,
    const std::size_t environmentCount,
    const bool terminalValueOnly = false,
    const mr_u32 terminalObservationStep = 0u
) {
    const auto& header = program.header();
    const auto encodeNetwork = [&](
        const std::span<const MRPolicyDenseLayerGPU> layers,
        const std::size_t observationBuffer,
        const std::size_t finalBuffer,
        const mr_u32 observationCount,
        const mr_u32 finalStride,
        const mr_u32 finalBase,
        const mr_u32 meanOffset,
        const mr_u32 inverseStdOffset,
        const mr_u32 observationStep,
        NSString* label
    ) {
        for (std::size_t layerIndex = 0u;
             layerIndex < layers.size();
             ++layerIndex) {
            const MRPolicyDenseLayerGPU& layer =
                layers[layerIndex];
            const bool first = layerIndex == 0u;
            const bool final =
                layerIndex + 1u == layers.size();
            const std::size_t source =
                first
                ? observationBuffer
                : ((layerIndex - 1u) & 1u) == 0u
                ? kPolicyScratchA
                : kPolicyScratchB;
            const std::size_t destination =
                final
                ? finalBuffer
                : (layerIndex & 1u) == 0u
                ? kPolicyScratchA
                : kPolicyScratchB;
            MRPolicyDenseDispatchGPU dispatch{};
            dispatch.counts = {
                static_cast<mr_u32>(environmentCount),
                layer.counts.x,
                layer.counts.y,
                layer.counts.z,
            };
            dispatch.strides = {
                layer.counts.x,
                final ? finalStride : layer.counts.y,
                first
                    ? static_cast<mr_u32>(
                          observationStep *
                          environmentCount *
                          observationCount
                      )
                    : 0u,
                final ? finalBase : 0u,
            };
            dispatch.offsets0 = {
                layer.offsets.x,
                layer.offsets.y,
                meanOffset,
                inverseStdOffset,
            };
            dispatch.offsets1 = {
                terminalValueOnly
                    ? static_cast<mr_u32>(
                          pass.controlStep * environmentCount
                      )
                    : 0u,
                0u,
                layer.counts.w |
                    (terminalValueOnly
                         ? MR_POLICY_DENSE_TIMEOUT_ONLY
                         : 0u),
                0u,
            };
            dispatch.limits = header.limits;
            dispatch.policyFingerprint =
                program.fingerprint();
            dispatch.taskFingerprint =
                program.taskFingerprint();
            if (!encodeContactThreadKernel(
                    context,
                    commandBuffer,
                    context.policyDensePipeline,
                    label,
                    {
                        {0u, kPolicyProgramHeader},
                        {1u, kPolicyProgramArena},
                        {3u, source},
                        {4u, destination},
                        {5u, kTaskTransitions},
                    },
                    nullptr,
                    0u,
                    environmentCount * layer.counts.y,
                    false,
                    0u,
                    &dispatch,
                    sizeof(dispatch),
                    2u
                )) {
                return false;
            }
        }
        return true;
    };

    if (!terminalValueOnly && !encodeNetwork(
            program.actorLayers(),
            kTaskActorObservations,
            kPolicyActorMean,
            header.counts0.z,
            header.counts1.x,
            0u,
            header.offsets0.z,
            header.offsets0.w,
            pass.controlStep,
            @"compiled actor inference"
        )) {
        return false;
    }
    if (!program.criticLayers().empty() &&
        !encodeNetwork(
            program.criticLayers(),
            kTaskCriticObservations,
            terminalValueOnly
                ? kTaskTransitions
                : kPolicyValues,
            header.counts0.w,
            terminalValueOnly
                ? static_cast<mr_u32>(
                      sizeof(MRTaskTransitionGPU) /
                      sizeof(float)
                  )
                : 1u,
            terminalValueOnly
                ? static_cast<mr_u32>(
                      pass.controlStep *
                          environmentCount *
                          (
                              sizeof(MRTaskTransitionGPU) /
                              sizeof(float)
                          ) +
                      offsetof(
                          MRTaskTransitionGPU,
                          timeoutBootstrapValue
                      ) /
                          sizeof(float)
                  )
                : static_cast<mr_u32>(
                      pass.controlStep * environmentCount
                  ),
            header.offsets1.x,
            header.offsets1.y,
            terminalValueOnly
                ? terminalObservationStep
                : pass.controlStep,
            terminalValueOnly
                ? @"compiled timeout bootstrap value"
                : @"compiled critic inference"
        )) {
        return false;
    }
    if (terminalValueOnly) {
        return true;
    }

    MRPolicySampleDispatchGPU dispatch{};
    dispatch.counts = {
        static_cast<mr_u32>(environmentCount),
        header.counts1.x,
        pass.controlStep,
        header.counts1.z,
    };
    dispatch.strides = {
        static_cast<mr_u32>(
            environmentCount * header.counts1.x
        ),
        static_cast<mr_u32>(environmentCount),
        header.counts1.x,
        0u,
    };
    dispatch.policyFingerprint = program.fingerprint();
    dispatch.taskFingerprint = program.taskFingerprint();
    return encodeContactThreadKernel(
        context,
        commandBuffer,
        context.policySamplePipeline,
        @"compiled policy sampling and scoring",
        {
            {0u, kPolicyProgramHeader},
            {1u, kPolicyProgramArena},
            {3u, kTaskDispatch},
            {4u, kTaskState},
            {5u, kPolicyActorMean},
            {6u, kTaskActions},
            {7u, kPolicyLatents},
            {8u, kPolicyLogProbabilities},
            {9u, kPolicyValues},
        },
        nullptr,
        0u,
        environmentCount,
        false,
        0u,
        &dispatch,
        sizeof(dispatch),
        2u
    );
}

bool encodeTaskApplyActions(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t environmentCount
) {
    return encodeContactThreadKernel(
        context,
        commandBuffer,
        context.taskApplyPipeline,
        @"compiled task action application",
        {
            {0u, kTaskDispatch},
            {1u, kTaskProgramHeader},
            {2u, kTaskProgramArena},
            {3u, kWorldDispatch},
            {5u, kTaskActions},
            {6u, kEffortTrajectory},
            {7u, kTaskDefaultQ},
            {8u, kTaskState},
            {9u, kTaskActionHistory},
            {10u, kTaskTeacherActions},
        },
        &pass,
        4u,
        environmentCount
    );
}

bool encodeMulticopterActuation(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t qState,
    const std::size_t vState,
    const std::size_t sourceMotorState,
    const std::size_t environmentCount
) {
    return encodeContactThreadKernel(
        context,
        commandBuffer,
        context.multicopterPipeline,
        @"compiled multicopter actuator program",
        {
            {0u, kMulticopterRotors},
            {1u, kMulticopterModel},
            {2u, kMulticopterMixer},
            {3u, kMulticopterDispatch},
            {4u, kTaskActionHistory},
            {5u, kResetMasks},
            {6u, qState},
            {7u, vState},
            {8u, sourceMotorState},
            {9u, kMulticopterCandidateState},
            {10u, kBodyWrenchPlaceholder},
        },
        &pass,
        11u,
        environmentCount
    );
}

bool encodeFlappingWingActuation(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const std::size_t qState,
    const std::size_t vState,
    const std::size_t environmentCount
) {
    return encodeContactThreadKernel(
        context,
        commandBuffer,
        context.flappingWingPipeline,
        @"compiled articulated flapping-wing load program",
        {
            {0u, kFlappingWingSpecs},
            {1u, kFlappingWingDispatch},
            {2u, qState},
            {3u, vState},
            {4u, kBodyWrenchPlaceholder},
        },
        nullptr,
        0u,
        environmentCount
    );
}

bool encodeTaskNativeActuators(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t qState,
    const std::size_t vState,
    const std::size_t environmentCount
) {
    return encodeContactThreadKernel(
        context,
        commandBuffer,
        context.taskNativeActuatorPipeline,
        @"compiled typed robot actuator program",
        {
            {0u, kTaskDispatch},
            {1u, kTaskProgramHeader},
            {2u, kTaskProgramArena},
            {3u, kWorldDispatch},
            {5u, qState},
            {6u, vState},
            {7u, kTaskActionHistory},
            {8u, kDofs},
            {9u, kActuatorProfiles},
            {10u, kBodyPoses},
            {11u, kWorkingEffort},
            {12u, kBodyWrenchPlaceholder},
        },
        &pass,
        4u,
        environmentCount
    );
}

bool encodeMulticopterCommit(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const std::size_t sourceMotorState,
    const std::size_t destinationMotorState,
    const std::size_t environmentCount
) {
    return encodeContactThreadKernel(
        context,
        commandBuffer,
        context.multicopterCommitPipeline,
        @"transactional multicopter motor-state commit",
        {
            {0u, sourceMotorState},
            {1u, kMulticopterCandidateState},
            {2u, destinationMotorState},
            {3u, kEnvironmentStatuses},
            {4u, kMulticopterDispatch},
        },
        nullptr,
        0u,
        environmentCount
    );
}

bool encodeTaskEffort(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t vState,
    const std::size_t environmentCount
) {
    return encodeContactThreadKernel(
        context,
        commandBuffer,
        context.taskEffortPipeline,
        @"compiled task applied-effort measurement",
        {
            {0u, kTaskDispatch},
            {1u, kTaskProgramHeader},
            {2u, kTaskProgramArena},
            {4u, vState},
            {5u, kWorkingEffort},
            {6u, kTaskState},
        },
        &pass,
        3u,
        environmentCount
    );
}

bool encodeTaskImpactContact(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t environmentCount
) {
    return encodeContactThreadKernel(
        context,
        commandBuffer,
        context.taskImpactContactPipeline,
        @"compiled task projectile-contact latch",
        {
            {0u, kTaskDispatch},
            {1u, kTaskProgramHeader},
            {2u, kTaskProgramArena},
            {3u, kContactDispatch},
            {4u, kContacts},
            {5u, kContactStatuses},
            {6u, kTaskState},
        },
        &pass,
        7u,
        environmentCount
    );
}

bool encodeTaskComplete(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t qState,
    const std::size_t vState,
    const std::size_t sceneState,
    const std::size_t environmentCount
) {
    return encodeContactThreadKernel(
        context,
        commandBuffer,
        context.taskCompletePipeline,
        @"compiled task contacts/reward/termination",
        {
            {0u, kTaskDispatch},
            {1u, kTaskProgramHeader},
            {2u, kTaskProgramArena},
            {3u, kTaskEvidenceState},
            {4u, kContactDispatch},
            {6u, qState},
            {7u, vState},
            {8u, kCandidateBodies},
            {9u, kContacts},
            {10u, kContactStatuses},
            {11u, kEnvironmentStatuses},
            {12u, sceneState},
            {13u, kDofs},
            {14u, kTaskDefaultQ},
            {15u, kTaskState},
            {16u, kTaskActionHistory},
            {17u, kTaskActorHistory},
            {18u, kTaskCleanHistory},
            {19u, kTaskPreviousJointVelocity},
            {20u, kTaskEncoderBias},
            {21u, kTaskBodyParameters},
            {22u, kTaskControllerParameters},
            {23u, kTaskContactCompact},
            {24u, kTaskTransitions},
            {25u, kShapes},
            {26u, kGeometryHeaders},
            {27u, kGeometryVertices},
            {28u, kTaskActorObservations},
            {29u, kTaskCriticObservations},
            {30u, kTaskCriticHistory},
        },
        &pass,
        5u,
        environmentCount
    );
}

bool encodeTaskEvidence(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass
) {
    return encodeContactThreadKernel(
        context,
        commandBuffer,
        context.taskEvidencePipeline,
        @"compiled task evidence reduction",
        {
            {0u, kTaskDispatch},
            {1u, kTaskProgramHeader},
            {2u, kTaskEvidenceState},
            {3u, kTaskTransitions},
        },
        &pass,
        4u,
        1u
    );
}

bool encodeRodControlPrepare(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t sourceNodes,
    const std::size_t sourceEdges,
    const std::size_t sourceWitnesses,
    const std::size_t witnessCount,
    const std::size_t environmentCount
) {
    if (!encodeContactThreadKernel(
        context,
        commandBuffer,
        context.rodPreparePipeline,
        @"MetalWorld rod checkpoint/reset",
        {
            {0u, kWorldDispatch},
            {1u, kContactDispatch},
            {3u, kResetMasks},
            {4u, kResetRodNodes},
            {5u, kResetRodEdges},
            {6u, sourceNodes},
            {7u, sourceEdges},
            {8u, kCheckpointRodNodes},
            {9u, kCheckpointRodEdges},
        },
        &pass,
        2u,
        environmentCount
    )) {
        return false;
    }
    return
        witnessCount == 0u ||
        encodeContactThreadKernel(
            context,
            commandBuffer,
            context.rodContactPreparePipeline,
            @"MetalWorld rod contact cache checkpoint/reset",
            {
                {0u, kWorldDispatch},
                {1u, kContactDispatch},
                {3u, kResetMasks},
                {4u, sourceWitnesses},
                {5u, kCheckpointRodWitnesses},
                {6u, kCandidateRodWitnesses},
            },
            &pass,
            2u,
            witnessCount
        );
}

bool encodeRodSubstep(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const CompiledWorld& world,
    const std::size_t sourceNodes,
    const std::size_t sourceEdges,
    const std::size_t candidateNodes,
    const std::size_t candidateEdges,
    const std::size_t eventStates,
    const mr_u32 eventSegmentMode,
    const std::size_t environmentCount
) {
    if (world.rodCount() == 0u) {
        return true;
    }
    if (!encodeContactThreadKernel(
            context,
            commandBuffer,
            context.rodPackPipeline,
            @"MetalWorld pack explicit rod state",
            {
                {0u, kContactDispatch},
                {1u, sourceNodes},
                {2u, sourceEdges},
                {3u, kRodInputPositions},
                {4u, kRodInputVelocities},
                {5u, kRodInputTwists},
                {6u, kRodInputTwistRates},
            },
            nullptr,
            0u,
            environmentCount
        )) {
        return false;
    }

    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        return false;
    }
    encoder.label =
        @"MetalWorld persistent SIMD32 DER factors/free motion";
    [encoder setComputePipelineState:context.rodStepPipeline];
    for (std::size_t rod = 0u;
         rod < world.rodCount();
         ++rod) {
        const std::size_t nodeOffset =
            world.rodNodeOffsets()[rod];
        const std::size_t edgeOffset =
            world.rodEdgeOffsets()[rod];
        const std::size_t bendOffset = edgeOffset - rod;
        const std::array<std::pair<std::size_t, NSUInteger>, 20u>
            bindings{{
                {kRodDispatches,
                 rod * sizeof(MRRodGPUDispatch)},
                {kRodRestLengths,
                 edgeOffset * sizeof(float)},
                {kRodRestTwists,
                 edgeOffset * sizeof(float)},
                {kRodRestCurvatures,
                 bendOffset * sizeof(mr_float4)},
                {kRodInverseMasses,
                 nodeOffset * sizeof(float)},
                {kRodInverseRotationalInertias,
                 edgeOffset * sizeof(float)},
                {kRodStretchStiffness,
                 edgeOffset * sizeof(float)},
                {kRodBendStiffness,
                 bendOffset * sizeof(float)},
                {kRodTwistStiffness,
                 bendOffset * sizeof(float)},
                {kRodInputPositions,
                 nodeOffset * sizeof(mr_float4)},
                {kRodInputVelocities,
                 nodeOffset * sizeof(mr_float4)},
                {kRodInputTwists,
                 edgeOffset * sizeof(float)},
                {kRodInputTwistRates,
                 edgeOffset * sizeof(float)},
                {kRodAttachments, 0u},
                {kRodOutputPositions,
                 nodeOffset * sizeof(mr_float4)},
                {kRodOutputVelocities,
                 nodeOffset * sizeof(mr_float4)},
                {kRodOutputTwists,
                 edgeOffset * sizeof(float)},
                {kRodOutputTwistRates,
                 edgeOffset * sizeof(float)},
                {kRodStatuses,
                 rod * environmentCount *
                     sizeof(MRRodGPUStatus)},
                {kRodReactions, 0u},
            }};
        for (NSUInteger argument = 0u;
             argument < bindings.size();
             ++argument) {
            [encoder
                setBuffer:context.buffers[
                              bindings[argument].first
                          ]
                   offset:bindings[argument].second
                  atIndex:argument];
        }
        [encoder setBuffer:context.buffers[eventStates]
                    offset:0u
                   atIndex:20u];
        [encoder setBytes:&eventSegmentMode
                   length:sizeof(eventSegmentMode)
                  atIndex:21u];
        [encoder
            dispatchThreadgroups:MTLSizeMake(
                static_cast<NSUInteger>(environmentCount),
                1u,
                1u
            )
            threadsPerThreadgroup:MTLSizeMake(
                MR_ROD_GPU_MAX_NODES,
                1u,
                1u
            )];
    }
    [encoder endEncoding];

    if (!encodeContactThreadKernel(
            context,
            commandBuffer,
            context.rodUnpackPipeline,
            @"MetalWorld unpack candidate rod state",
            {
                {0u, kContactDispatch},
                {1u, kRodOutputPositions},
                {2u, kRodOutputVelocities},
                {3u, kRodOutputTwists},
                {4u, kRodOutputTwistRates},
                {5u, candidateNodes},
                {6u, candidateEdges},
            },
            nullptr,
            0u,
            environmentCount
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.rodLatchPipeline,
            @"MetalWorld rod failure latch",
            {
                {0u, kWorldDispatch},
                {1u, kContactDispatch},
                {3u, kRodStatuses},
                {4u, kEnvironmentStatuses},
            },
            &pass,
            2u,
            environmentCount
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.rodContactLatchPipeline,
            @"MetalWorld rod/contact transaction latch",
            {
                {0u, kContactDispatch},
                {2u, kRodStatuses},
                {3u, kContactStatuses},
            },
            &pass,
            1u,
            environmentCount
        )) {
        return false;
    }

    id<MTLComputeCommandEncoder> factorEncoder =
        [commandBuffer computeCommandEncoder];
    if (factorEncoder == nil) {
        return false;
    }
    factorEncoder.label =
        @"MetalWorld retained banded rod operator";
    [factorEncoder
        setComputePipelineState:context.rodFactorPipeline];
    for (std::size_t rod = 0u;
         rod < world.rodCount();
         ++rod) {
        const std::array<std::size_t, 12u> bindings{{
            kContactDispatch,
            kRodDispatches,
            candidateNodes,
            kRodInverseMasses,
            kRodInverseRotationalInertias,
            kRodColliders,
            kRodRestLengths,
            kRodStretchStiffness,
            kRodBendStiffness,
            kRodTwistStiffness,
            kRodFactorCaches,
            kOperatorVelocityArena,
        }};
        for (NSUInteger argument = 0u;
             argument < bindings.size();
             ++argument) {
            const NSUInteger offset =
                argument == 1u
                ? rod * sizeof(MRRodGPUDispatch)
                : 0u;
            [factorEncoder
                setBuffer:context.buffers[bindings[argument]]
                   offset:offset
                  atIndex:argument];
        }
        const std::uint32_t rodIndex =
            static_cast<std::uint32_t>(rod);
        [factorEncoder
            setBytes:&rodIndex
              length:sizeof(rodIndex)
             atIndex:12u];
        [factorEncoder
            setBytes:&pass
              length:sizeof(pass)
             atIndex:13u];
        const NSUInteger threadgroupWidth = std::min<NSUInteger>(
            std::max<NSUInteger>(
                context.rodFactorPipeline.threadExecutionWidth,
                1u
            ),
            context.rodFactorPipeline
                .maxTotalThreadsPerThreadgroup
        );
        [factorEncoder
            dispatchThreads:MTLSizeMake(
                static_cast<NSUInteger>(environmentCount),
                1u,
                1u
            )
            threadsPerThreadgroup:MTLSizeMake(
                threadgroupWidth,
                1u,
                1u
            )];
    }
    [factorEncoder endEncoding];
    return true;
}

bool encodeRodToolNarrowphase(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const CompiledWorld& world,
    const std::size_t previousWitnesses,
    const std::size_t environmentCount
) {
    if (world.rodToolPairs().empty()) {
        return true;
    }
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        return false;
    }
    encoder.label =
        @"MetalWorld procedural rod/tool narrowphase";
    [encoder
        setComputePipelineState:
            context.rodToolNarrowphasePipeline];
    for (std::size_t rod = 0u;
         rod < world.rodCount();
         ++rod) {
        const MRRodGPUDispatch& dispatch =
            reinterpret_cast<const MRRodGPUDispatch*>(
                context.buffers[kRodCollisionDispatches].contents
            )[rod];
        if (dispatch.toolPairCount == 0u) {
            continue;
        }
        const std::array<
            std::pair<std::size_t, NSUInteger>,
            17u
        > bindings{{
            {kRodCollisionDispatches,
             rod * sizeof(MRRodGPUDispatch)},
            {kRodColliders, 0u},
            {kRodShapeSources, 0u},
            {kRodToolPairs, 0u},
            {kShapes, 0u},
            {kCandidateBodies, 0u},
            {kGeometryHeaders, 0u},
            {kGeometryVertices, 0u},
            {kMeshBvhNodes, 0u},
            {kMeshTriangles, 0u},
            {kRodOutputPositions, 0u},
            {kRodOutputVelocities, 0u},
            {kRodOutputTwistRates, 0u},
            {previousWitnesses, 0u},
            {kRodWitnessCounts, 0u},
            {kCandidateRodWitnesses, 0u},
            {kRodStatuses,
             rod * environmentCount *
                 sizeof(MRRodGPUStatus)},
        }};
        for (NSUInteger argument = 0u;
             argument < bindings.size();
             ++argument) {
            [encoder
                setBuffer:context.buffers[
                              bindings[argument].first
                          ]
                   offset:bindings[argument].second
                  atIndex:argument];
        }
        dispatchWorldThreads(
            encoder,
            context.rodToolNarrowphasePipeline,
            environmentCount * dispatch.toolPairCount
        );
    }
    [encoder endEncoding];
    return true;
}

bool encodeRodContactSolve(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t candidateRodNodes,
    const std::size_t candidateRodEdges,
    const std::size_t rodWitnessCount,
    const std::size_t environmentCount
) {
    return
        rodWitnessCount == 0u ||
        encodeContactThreadKernel(
            context,
            commandBuffer,
            context.rodContactSolvePipeline,
            @"MetalWorld typed rod/contact operator solve",
            {
                {0u, kContactDispatch},
                {1u, kFactorMatrix},
                {2u, kPointJacobians},
                {3u, kCandidateV},
                {4u, kCandidateBodies},
                {5u, candidateRodNodes},
                {6u, candidateRodEdges},
                {7u, kRodInverseMasses},
                {8u, kRodInverseRotationalInertias},
                {9u, kRodColliders},
                {10u, kContacts},
                {11u, kIRBlocks},
                {12u, kEvaluatedRows},
                {13u, kEvaluatedCones},
                {14u, kResponseColumns},
                {15u, kCandidateRodWitnesses},
                {16u, kContactStatuses},
            },
            &pass,
            17u,
            environmentCount
        );
}

bool encodeRodCommit(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t candidateNodes,
    const std::size_t candidateEdges,
    const std::size_t destinationNodes,
    const std::size_t destinationEdges,
    const std::size_t destinationWitnesses,
    const std::size_t witnessCount,
    const std::size_t environmentCount
) {
    if (!encodeContactThreadKernel(
        context,
        commandBuffer,
        context.rodCommitPipeline,
        @"MetalWorld transactional rod publication",
        {
            {0u, kWorldDispatch},
            {1u, kContactDispatch},
            {3u, kEnvironmentStatuses},
            {4u, candidateNodes},
            {5u, candidateEdges},
            {6u, kCheckpointRodNodes},
            {7u, kCheckpointRodEdges},
            {8u, destinationNodes},
            {9u, destinationEdges},
        },
        &pass,
        2u,
        environmentCount
    )) {
        return false;
    }
    return
        witnessCount == 0u ||
        encodeContactThreadKernel(
            context,
            commandBuffer,
            context.rodContactCommitPipeline,
            @"MetalWorld transactional rod contact publication",
            {
                {0u, kWorldDispatch},
                {1u, kContactDispatch},
                {3u, kEnvironmentStatuses},
                {4u, kCandidateRodWitnesses},
                {5u, kCheckpointRodWitnesses},
                {6u, destinationWitnesses},
            },
            &pass,
            2u,
            witnessCount
        );
}

bool encodeABA(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    const std::size_t sourceQ,
    const std::size_t sourceV,
    const std::size_t articulationCount,
    const std::size_t environmentCount
) {
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        return false;
    }
    encoder.label = @"MetalWorld ABA";
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:context.buffers[kWorld]
                 offset:0u
                atIndex:0u];
    [encoder setBuffer:context.buffers[kArticulations]
                 offset:0u
                atIndex:1u];
    [encoder setBuffer:context.buffers[kJoints]
                 offset:0u
                atIndex:2u];
    [encoder setBuffer:context.buffers[kDofs]
                 offset:0u
                atIndex:3u];
    [encoder setBuffer:context.buffers[kBodies]
                 offset:0u
                atIndex:4u];
    [encoder setBuffer:context.buffers[kABADispatch]
                 offset:0u
                atIndex:5u];
    [encoder setBuffer:context.buffers[sourceQ]
                 offset:0u
                atIndex:6u];
    [encoder setBuffer:context.buffers[sourceV]
                 offset:0u
                atIndex:7u];
    [encoder setBuffer:context.buffers[kWorkingEffort]
                 offset:0u
                atIndex:8u];
    [encoder setBuffer:context.buffers[kBodyWrenchPlaceholder]
                 offset:0u
                atIndex:9u];
    [encoder setBuffer:context.buffers[kCandidateAcceleration]
                 offset:0u
                atIndex:10u];
    [encoder setBuffer:context.buffers[kCandidateV]
                 offset:0u
                atIndex:11u];
    [encoder setBuffer:context.buffers[kCandidateQ]
                 offset:0u
                atIndex:12u];
    [encoder setBuffer:context.buffers[kABAStatuses]
                 offset:0u
                atIndex:13u];
    if (context.useTaskBodyParameters) {
        [encoder setBuffer:context.buffers[kTaskBodyParameters]
                     offset:0u
                    atIndex:14u];
        [encoder setBuffer:
                     context.buffers[kTaskControllerParameters]
                     offset:0u
                    atIndex:15u];
    }
    [encoder
        dispatchThreadgroups:MTLSizeMake(
            static_cast<NSUInteger>(environmentCount),
            static_cast<NSUInteger>(
                std::max<std::size_t>(articulationCount, 1u)
            ),
            1u
        )
        threadsPerThreadgroup:MTLSizeMake(
            kABAThreadsPerThreadgroup,
            1u,
            1u
        )];
    [encoder endEncoding];
    return true;
}

bool encodeCommit(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t destinationQ,
    const std::size_t destinationV,
    const std::size_t environmentCount
) {
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        return false;
    }
    encoder.label = @"MetalWorld transactional commit";
    [encoder setComputePipelineState:context.commitPipeline];
    [encoder setBuffer:context.buffers[kWorldDispatch]
                 offset:0u
                atIndex:0u];
    [encoder setBytes:&pass length:sizeof(pass) atIndex:1u];
    [encoder setBuffer:context.buffers[kABAStatuses]
                 offset:0u
                atIndex:2u];
    [encoder setBuffer:context.buffers[kCandidateQ]
                 offset:0u
                atIndex:3u];
    [encoder setBuffer:context.buffers[kCandidateV]
                 offset:0u
                atIndex:4u];
    [encoder setBuffer:context.buffers[destinationQ]
                 offset:0u
                atIndex:5u];
    [encoder setBuffer:context.buffers[destinationV]
                 offset:0u
                atIndex:6u];
    [encoder setBuffer:context.buffers[kEnvironmentStatuses]
                 offset:0u
                atIndex:7u];
    [encoder setBuffer:context.buffers[kCheckpointQ]
                 offset:0u
                atIndex:8u];
    [encoder setBuffer:context.buffers[kCheckpointV]
                 offset:0u
                atIndex:9u];
    [encoder setBuffer:context.buffers[kWorld]
                 offset:0u
                atIndex:10u];
    dispatchWorldThreads(
        encoder,
        context.commitPipeline,
        environmentCount
    );
    [encoder endEncoding];
    return true;
}

bool encodeCapture(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t sourceQ,
    const std::size_t sourceV,
    const std::size_t environmentCount
) {
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        return false;
    }
    encoder.label = @"MetalWorld observation capture";
    [encoder setComputePipelineState:context.capturePipeline];
    [encoder setBuffer:context.buffers[kWorldDispatch]
                 offset:0u
                atIndex:0u];
    [encoder setBytes:&pass length:sizeof(pass) atIndex:1u];
    [encoder setBuffer:context.buffers[sourceQ]
                 offset:0u
                atIndex:2u];
    [encoder setBuffer:context.buffers[sourceV]
                 offset:0u
                atIndex:3u];
    [encoder setBuffer:context.buffers[kCandidateAcceleration]
                 offset:0u
                atIndex:4u];
    [encoder setBuffer:context.buffers[kEnvironmentStatuses]
                 offset:0u
                atIndex:5u];
    [encoder setBuffer:context.buffers[kObservations]
                 offset:0u
                atIndex:6u];
    [encoder setBuffer:context.buffers[kAccelerationTrajectory]
                 offset:0u
                atIndex:7u];
    [encoder setBuffer:context.buffers[kPublicStatuses]
                 offset:0u
                atIndex:8u];
    dispatchWorldThreads(
        encoder,
        context.capturePipeline,
        environmentCount
    );
    [encoder endEncoding];
    return true;
}

bool encodeContactControlPrepare(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t sourceScene,
    const std::size_t sourceManifoldHeaders,
    const std::size_t sourceManifoldPoints,
    const std::size_t sourceManifoldCounts,
    const std::size_t environmentCount
) {
    return encodeContactThreadKernel(
        context,
        commandBuffer,
        context.contactPreparePipeline,
        @"MetalWorld contact checkpoint/reset",
        {
            {0u, kWorldDispatch},
            {1u, kContactDispatch},
            {3u, kResetMasks},
            {4u, kResetSceneBodies},
            {5u, kKinematicTargets},
            {6u, kBodies},
            {7u, kSceneBodyIndices},
            {8u, sourceScene},
            {9u, kCheckpointSceneBodies},
            {10u, sourceManifoldHeaders},
            {11u, sourceManifoldPoints},
            {12u, sourceManifoldCounts},
            {13u, kCheckpointManifoldHeaders},
            {14u, kCheckpointManifoldPoints},
            {15u, kCheckpointManifoldCounts},
            {16u, kContactStatuses},
            {17u, kConvexCaches},
        },
        &pass,
        2u,
        environmentCount
    );
}

bool encodeWave32ContactSolve(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& solverPass,
    const mr_u32 solverIterationCount,
    const bool enableDistributed,
    const std::size_t candidateRodNodes,
    const std::size_t candidateRodEdges,
    const std::size_t environmentCount,
    const std::size_t islandWorkCount,
    const std::size_t tileWorkCount
) {
    if (!encodeContactThreadKernel(
            context,
            commandBuffer,
            context.buildTilesPipeline,
            @"MetalWorld 32-contact tile construction",
            {
                {0u, kContactDispatch},
                {1u, kCandidateBodies},
                {2u, kContacts},
                {3u, kIslands},
                {4u, kIslandWorkDense},
                {5u, kContactTiles},
                {6u, kTileConstraintIndices},
                {7u, kContactStatuses},
                {8u, kCompactionFlags},
            },
            nullptr,
            0u,
            environmentCount
        )) {
        return false;
    }
    if (!encodeStableBooleanScan(
            context,
            commandBuffer,
            kCompactionFlags,
            islandWorkCount,
            MR_WORLD_WORK_SOLVER
        )) {
        return false;
    }

    id<MTLComputeCommandEncoder> scatter =
        [commandBuffer computeCommandEncoder];
    if (scatter == nil) {
        return false;
    }
    scatter.label = @"MetalWorld stable compact island scatter";
    [scatter
        setComputePipelineState:
            context.islandQueueScatterPipeline];
    const std::array<std::size_t, 6u> scatterBuffers{{
        kContactDispatch,
        kCompactionFlags,
        kCompactionOffsets,
        kIslandWorkDense,
        kIslandWorkQueue,
        kWorkQueueHeaders,
    }};
    for (NSUInteger argument = 0u;
         argument < scatterBuffers.size();
         ++argument) {
        [scatter setBuffer:context.buffers[
                               scatterBuffers[argument]
                           ]
                    offset:0u
                   atIndex:argument];
    }
    [scatter
        dispatchThreads:MTLSizeMake(
            static_cast<NSUInteger>(
                std::max<std::size_t>(islandWorkCount, 1u)
            ),
            1u,
            1u
        )
        threadsPerThreadgroup:MTLSizeMake(
            kWorldThreadsPerThreadgroup,
            1u,
            1u
        )];
    [scatter endEncoding];

    id<MTLComputeCommandEncoder> cohort =
        [commandBuffer computeCommandEncoder];
    if (cohort == nil) {
        return false;
    }
    cohort.label = @"MetalWorld homogeneous solver cohort selection";
    [cohort setComputePipelineState:context.solverCohortPipeline];
    [cohort setBuffer:context.buffers[kIslandWorkQueue]
               offset:0u
              atIndex:0u];
    [cohort setBuffer:context.buffers[kWorkQueueHeaders]
               offset:0u
              atIndex:1u];
    [cohort setBuffer:context.buffers[kWaveWorkPackets]
               offset:0u
              atIndex:2u];
    [cohort
        dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(
            MR_WAVE32_CONTACTS_PER_TILE,
            1u,
            1u
        )];
    [cohort endEncoding];

    if (enableDistributed) {
        if (!encodeContactThreadKernel(
            context,
            commandBuffer,
            context.distributedIslandFlagPipeline,
            @"MetalWorld distributed-island flags",
            {
                {0u, kContactDispatch},
                {1u, kIslandWorkDense},
                {2u, kCompactionFlags},
            },
            nullptr,
            0u,
            islandWorkCount
        ) ||
        !encodeStableBooleanScan(
            context,
            commandBuffer,
            kCompactionFlags,
            islandWorkCount,
            MR_WORLD_WORK_SOLVER_DISTRIBUTED
        )) {
            return false;
        }
    id<MTLComputeCommandEncoder> distributedIslandScatter =
        [commandBuffer computeCommandEncoder];
    if (distributedIslandScatter == nil) {
        return false;
    }
    distributedIslandScatter.label =
        @"MetalWorld compact distributed-island scatter";
    [distributedIslandScatter
        setComputePipelineState:
            context.distributedIslandScatterPipeline];
    const std::array<std::size_t, 6u>
        distributedIslandScatterBuffers{{
            kContactDispatch,
            kCompactionFlags,
            kCompactionOffsets,
            kIslandWorkDense,
            kIslandWorkQueue,
            kWorkQueueHeaders,
        }};
    for (NSUInteger argument = 0u;
         argument < distributedIslandScatterBuffers.size();
         ++argument) {
        [distributedIslandScatter
            setBuffer:context.buffers[
                          distributedIslandScatterBuffers[
                              argument
                          ]
                      ]
            offset:0u
            atIndex:argument];
    }
    [distributedIslandScatter
        dispatchThreads:MTLSizeMake(
            static_cast<NSUInteger>(
                std::max<std::size_t>(
                    islandWorkCount,
                    1u
                )
            ),
            1u,
            1u
        )
        threadsPerThreadgroup:MTLSizeMake(
            kWorldThreadsPerThreadgroup,
            1u,
            1u
        )];
        [distributedIslandScatter endEncoding];

    if (!encodeContactThreadKernel(
            context,
            commandBuffer,
            context.distributedTileFlagPipeline,
            @"MetalWorld distributed-tile flags",
            {
                {0u, kContactDispatch},
                {1u, kContactStatuses},
                {2u, kIslandWorkDense},
                {3u, kContactTiles},
                {4u, kCompactionFlags},
            },
            nullptr,
            0u,
            tileWorkCount
        ) ||
        !encodeStableBooleanScan(
            context,
            commandBuffer,
            kCompactionFlags,
            tileWorkCount,
            MR_WORLD_WORK_SOLVER_SPILL
        )) {
            return false;
        }
    id<MTLComputeCommandEncoder> distributedTileScatter =
        [commandBuffer computeCommandEncoder];
    if (distributedTileScatter == nil) {
        return false;
    }
    distributedTileScatter.label =
        @"MetalWorld compact distributed-tile scatter";
    [distributedTileScatter
        setComputePipelineState:
            context.distributedTileScatterPipeline];
    const std::array<std::size_t, 6u>
        distributedTileScatterBuffers{{
            kContactDispatch,
            kCompactionFlags,
            kCompactionOffsets,
            kContactTiles,
            kContactTiles,
            kWorkQueueHeaders,
        }};
    for (NSUInteger argument = 0u;
         argument < distributedTileScatterBuffers.size();
         ++argument) {
        [distributedTileScatter
            setBuffer:context.buffers[
                          distributedTileScatterBuffers[
                              argument
                          ]
                      ]
            offset:0u
            atIndex:argument];
    }
    [distributedTileScatter
        dispatchThreads:MTLSizeMake(
            static_cast<NSUInteger>(
                std::max<std::size_t>(tileWorkCount, 1u)
            ),
            1u,
            1u
        )
        threadsPerThreadgroup:MTLSizeMake(
            kWorldThreadsPerThreadgroup,
            1u,
            1u
        )];
        [distributedTileScatter endEncoding];

    const auto encodeDistributedPrepare = [&]() {
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return false;
        }
        encoder.label =
            @"MetalWorld distributed Wave32 block preparation";
        [encoder
            setComputePipelineState:
                context.wave32DistributedPreparePipeline];
        const std::array<std::size_t, 15u> buffers{{
            kContactDispatch,
            kFactorMatrix,
            kPointJacobians,
            kCandidateBodies,
            kContacts,
            kEvaluatedRows,
            kEvaluatedCones,
            kResponseColumns,
            kWave32ImpulseDeltas,
            kWave32Preconditioners,
            kContactStatuses,
            kIslandWorkDense,
            kContactTiles,
            kTileConstraintIndices,
            kWorkQueueHeaders,
        }};
        for (NSUInteger argument = 0u;
             argument < buffers.size();
             ++argument) {
            [encoder setBuffer:context.buffers[
                                   buffers[argument]
                               ]
                        offset:0u
                       atIndex:argument];
        }
        [encoder
            dispatchThreadgroupsWithIndirectBuffer:
                context.buffers[kWorkQueueHeaders]
            indirectBufferOffset:
                MR_WORLD_WORK_SOLVER_SPILL *
                    sizeof(MRWorkQueueHeaderGPU) +
                offsetof(
                    MRWorkQueueHeaderGPU,
                    indirect
                )
            threadsPerThreadgroup:MTLSizeMake(
                MR_WAVE32_CONTACTS_PER_TILE,
                1u,
                1u
            )];
        [encoder endEncoding];
        return true;
    };
    const auto encodeDistributedDelta = [&]() {
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return false;
        }
        encoder.label =
            @"MetalWorld distributed Wave32 cone update";
        [encoder
            setComputePipelineState:
                context.wave32DistributedDeltaPipeline];
        const std::array<std::size_t, 13u> buffers{{
            kContactDispatch,
            kPointJacobians,
            kCandidateV,
            kCandidateBodies,
            kContacts,
            kEvaluatedRows,
            kEvaluatedCones,
            kWave32Preconditioners,
            kWave32ImpulseDeltas,
            kContactStatuses,
            kContactTiles,
            kTileConstraintIndices,
            kWorkQueueHeaders,
        }};
        for (NSUInteger argument = 0u;
             argument < buffers.size();
             ++argument) {
            [encoder setBuffer:context.buffers[
                                   buffers[argument]
                               ]
                        offset:0u
                       atIndex:argument];
        }
        [encoder
            dispatchThreadgroupsWithIndirectBuffer:
                context.buffers[kWorkQueueHeaders]
            indirectBufferOffset:
                MR_WORLD_WORK_SOLVER_SPILL *
                    sizeof(MRWorkQueueHeaderGPU) +
                offsetof(
                    MRWorkQueueHeaderGPU,
                    indirect
                )
            threadsPerThreadgroup:MTLSizeMake(
                MR_WAVE32_CONTACTS_PER_TILE,
                1u,
                1u
            )];
        [encoder endEncoding];
        return true;
    };
    const auto encodeDistributedReduce = [&](
        const mr_u32 mode
    ) {
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return false;
        }
        encoder.label =
            @"MetalWorld deterministic distributed reduction";
        [encoder
            setComputePipelineState:
                context.wave32DistributedReducePipeline];
        const std::array<std::size_t, 18u> buffers{{
            kContactDispatch,
            kPointJacobians,
            kCandidateV,
            kCandidateBodies,
            kContacts,
            kContactMetadata,
            kEvaluatedRows,
            kEvaluatedCones,
            kResponseColumns,
            kWave32ImpulseDeltas,
            kWave32Preconditioners,
            kCandidateManifoldPoints,
            kContactStatuses,
            kIslandWorkQueue,
            kContactTiles,
            kTileConstraintIndices,
            kWave32IslandStatuses,
            kWorkQueueHeaders,
        }};
        for (NSUInteger argument = 0u;
             argument < buffers.size();
             ++argument) {
            [encoder setBuffer:context.buffers[
                                   buffers[argument]
                               ]
                        offset:0u
                       atIndex:argument];
        }
        MRMetalWorldPassGPU distributedPass = solverPass;
        distributedPass.reserved0 = solverIterationCount;
        distributedPass.reserved1 = mode;
        [encoder setBytes:&distributedPass
                   length:sizeof(distributedPass)
                  atIndex:18u];
        [encoder
            dispatchThreadgroupsWithIndirectBuffer:
                context.buffers[kWorkQueueHeaders]
            indirectBufferOffset:
                MR_WORLD_WORK_SOLVER_DISTRIBUTED *
                    sizeof(MRWorkQueueHeaderGPU) +
                offsetof(
                    MRWorkQueueHeaderGPU,
                    indirect
                )
            threadsPerThreadgroup:MTLSizeMake(
                MR_WAVE32_CONTACTS_PER_TILE,
                1u,
                1u
            )];
        [encoder endEncoding];
        return true;
    };
        if (!encodeDistributedPrepare() ||
            !encodeDistributedReduce(0u)) {
            return false;
        }
        for (mr_u32 iteration = 0u;
             iteration < solverIterationCount;
             ++iteration) {
            if (!encodeDistributedDelta() ||
                !encodeDistributedReduce(
                    iteration + 1u ==
                            solverIterationCount
                    ? 2u
                    : 1u
                )) {
                return false;
            }
        }
    }

    id<MTLComputeCommandEncoder> wave =
        [commandBuffer computeCommandEncoder];
    if (wave == nil) {
        return false;
    }
    wave.label = @"MetalWorld Wave32 temporal cone solve";
    [wave setComputePipelineState:context.wave32SolvePipeline];
    const std::array<std::size_t, 19u> buffers{{
        kContactDispatch,
        kFactorMatrix,
        kPointJacobians,
        kCandidateV,
        kCandidateBodies,
        kContacts,
        kContactMetadata,
        kEvaluatedRows,
        kEvaluatedCones,
        kResponseColumns,
        kCandidateManifoldPoints,
        kContactStatuses,
        kIslandWorkQueue,
        kContactTiles,
        kTileConstraintIndices,
        kWave32ImpulseDeltas,
        kWave32Preconditioners,
        kWave32IslandStatuses,
        kWorkQueueHeaders,
    }};
    for (NSUInteger argument = 0u;
         argument < buffers.size();
         ++argument) {
        [wave setBuffer:context.buffers[buffers[argument]]
                 offset:0u
                atIndex:argument];
    }
    [wave setBytes:&solverPass
            length:sizeof(solverPass)
           atIndex:19u];
    [wave setBuffer:context.buffers[kWaveWorkPackets]
             offset:0u
            atIndex:20u];
    const std::array<std::size_t, 7u> rodBuffers{{
        candidateRodNodes,
        candidateRodEdges,
        kRodInverseMasses,
        kRodInverseRotationalInertias,
        kRodColliders,
        kCandidateRodWitnesses,
        kRodConstraintWitnessIndices,
    }};
    for (NSUInteger argument = 0u;
         argument < rodBuffers.size();
         ++argument) {
        [wave setBuffer:context.buffers[rodBuffers[argument]]
                 offset:0u
                atIndex:21u + argument];
    }
    [wave setBuffer:context.buffers[kRodFactorCaches]
             offset:0u
            atIndex:28u];
    [wave setBuffer:context.buffers[kOperatorVelocityArena]
             offset:0u
            atIndex:29u];
    [wave
        dispatchThreadgroupsWithIndirectBuffer:
            context.buffers[kWorkQueueHeaders]
        indirectBufferOffset:
            MR_WORLD_WORK_SOLVER *
                sizeof(MRWorkQueueHeaderGPU) +
            offsetof(MRWorkQueueHeaderGPU, indirect)
        threadsPerThreadgroup:MTLSizeMake(
            MR_WAVE32_CONTACTS_PER_TILE,
            1u,
            1u
        )];
    [wave endEncoding];

    if (!encodeContactThreadKernel(
            context,
            commandBuffer,
            context.wave32ReducePipeline,
            @"MetalWorld Wave32 island status reduction",
            {
                {0u, kContactDispatch},
                {1u, kWave32IslandStatuses},
                {2u, kContactStatuses},
                {3u, kWorkQueueHeaders},
            },
            nullptr,
            0u,
            environmentCount
        )) {
        return false;
    }
    MRMetalWorldPassGPU replayPass = solverPass;
    replayPass.reserved1 = 1u;
    return encodeContactThreadKernel(
        context,
        commandBuffer,
        context.contactSolvePipeline,
        @"MetalWorld stiff-island ordered cone replay",
        {
            {0u, kContactDispatch},
            {1u, kFactorMatrix},
            {2u, kPointJacobians},
            {3u, kCandidateV},
            {4u, kCandidateBodies},
            {5u, kContacts},
            {6u, kContactMetadata},
            {7u, kEvaluatedRows},
            {8u, kEvaluatedCones},
            {9u, kResponseColumns},
            {10u, kCandidateManifoldPoints},
            {11u, kContactStatuses},
        },
        &replayPass,
        12u,
        environmentCount
    );
}

bool encodeParallelManifoldCompile(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t sourceManifoldHeaders,
    const std::size_t sourceManifoldPoints,
    const std::size_t sourceManifoldCounts,
    const std::size_t candidateRodNodes,
    const std::size_t environmentCount,
    const std::size_t pairFlagThreadCount,
    const std::size_t rodWitnessThreadCount
) {
    const auto encodeScan = [&]() {
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return false;
        }
        encoder.label =
            @"MetalWorld SIMD32 manifold segmented scan";
        [encoder
            setComputePipelineState:
                context.manifoldScanPipeline];
        [encoder setBuffer:context.buffers[kContactDispatch]
                    offset:0u
                   atIndex:0u];
        [encoder setBuffer:context.buffers[kManifoldIRScatter]
                    offset:0u
                   atIndex:1u];
        [encoder
            setBuffer:context.buffers[kCandidateManifoldCounts]
              offset:0u
             atIndex:2u];
        [encoder setBuffer:context.buffers[kContactStatuses]
                    offset:0u
                   atIndex:3u];
        [encoder setBytes:&pass
                   length:sizeof(pass)
                  atIndex:4u];
        [encoder
            dispatchThreadgroups:MTLSizeMake(
                static_cast<NSUInteger>(
                    std::max<std::size_t>(
                        environmentCount,
                        1u
                    )
                ),
                1u,
                1u
            )
            threadsPerThreadgroup:MTLSizeMake(
                MR_SIMD_WIDTH,
                1u,
                1u
            )];
        [encoder endEncoding];
        return true;
    };
    const auto encodeRodScan = [&]() {
        if (rodWitnessThreadCount == 0u) {
            return true;
        }
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return false;
        }
        encoder.label =
            @"MetalWorld SIMD32 rod witness segmented scan";
        [encoder
            setComputePipelineState:
                context.rodContactScanPipeline];
        [encoder setBuffer:context.buffers[kContactDispatch]
                    offset:0u
                   atIndex:0u];
        [encoder setBuffer:context.buffers[kRodWitnessCounts]
                    offset:0u
                   atIndex:1u];
        [encoder
            setBuffer:context.buffers[kCandidateRodWitnesses]
              offset:0u
             atIndex:2u];
        [encoder setBuffer:context.buffers[kRodContactScratch]
                    offset:0u
                   atIndex:3u];
        [encoder setBuffer:context.buffers[kContactStatuses]
                    offset:0u
                   atIndex:4u];
        [encoder
            dispatchThreadgroups:MTLSizeMake(
                static_cast<NSUInteger>(environmentCount),
                1u,
                1u
            )
            threadsPerThreadgroup:MTLSizeMake(
                MR_SIMD_WIDTH,
                1u,
                1u
            )];
        [encoder endEncoding];
        return true;
    };
    return
        encodeContactThreadKernel(
            context,
            commandBuffer,
            context.multiQueryInitializePipeline,
            @"MetalWorld articulation-major point-query initialization",
            {
                {0u, kContactDispatch},
                {1u, kArticulations},
                {2u, kPointQueries},
                {3u, kContactStatuses},
            },
            nullptr,
            0u,
            environmentCount
        ) &&
        encodeContactThreadKernel(
            context,
            commandBuffer,
            context.authoredIRSeedPipeline,
            @"MetalWorld immutable mechanism ConstraintIR seed",
            {
                {0u, kContactDispatch},
                {1u, kAuthoredIRBlocks},
                {2u, kAuthoredIREndpoints},
                {3u, kAuthoredIRRows},
                {4u, kAuthoredIRCones},
                {5u, kAuthoredIRWarmImpulses},
                {6u, kDynamicNodes},
                {7u, kContacts},
                {8u, kContactMetadata},
                {9u, kIRBlocks},
                {10u, kIREndpoints},
                {11u, kEndpointRuntime},
                {12u, kIRRows},
                {13u, kIRCones},
                {14u, kContactStatuses},
                {15u, kBodyDynamicNodes},
                {16u, kCandidateBodies},
                {17u, candidateRodNodes},
            },
            nullptr,
            0u,
            environmentCount
        ) &&
        encodeContactThreadKernel(
            context,
            commandBuffer,
            context.manifoldFinalizePipeline,
            @"MetalWorld pair-parallel manifold finalization",
            {
                {0u, kContactDispatch},
                {1u, kShapes},
                {2u, kCurrentBodies},
                {3u, kEligiblePairs},
                {4u, sourceManifoldCounts},
                {5u, sourceManifoldHeaders},
                {6u, sourceManifoldPoints},
                {7u, kPairOverlapFlags},
                {8u, kPairRawCounts},
                {9u, kPairRawContactStaging},
                {10u, kCandidateConvexCaches},
                {11u, kCCDPairs},
                {12u, kPairManifoldHeaders},
                {13u, kPairManifoldPoints},
                {14u, kManifoldIRScatter},
                {15u, kContactStatuses},
            },
            &pass,
            16u,
            pairFlagThreadCount
        ) &&
        encodeScan() &&
        encodeRodScan() &&
        encodeContactThreadKernel(
            context,
            commandBuffer,
            context.manifoldRecordScatterPipeline,
            @"MetalWorld stable manifold record scatter",
            {
                {0u, kContactDispatch},
                {1u, kEligiblePairs},
                {2u, kPairRawContactStaging},
                {3u, kPairManifoldHeaders},
                {4u, kPairManifoldPoints},
                {5u, kManifoldIRScatter},
                {6u, kContactStatuses},
                {7u, kCandidatePairs},
                {8u, kRawContacts},
                {9u, kRawPairIndices},
                {10u, kCandidateManifoldHeaders},
                {11u, kCandidateManifoldPoints},
            },
            nullptr,
            0u,
            pairFlagThreadCount
        ) &&
        encodeContactThreadKernel(
            context,
            commandBuffer,
            context.manifoldIRScatterPipeline,
            @"MetalWorld manifold-to-ConstraintIR scatter",
            {
                {0u, kContactDispatch},
                {1u, kShapes},
                {2u, kMaterials},
                {3u, kCurrentBodies},
                {4u, kArticulations},
                {5u, kEligiblePairs},
                {6u, kPairManifoldHeaders},
                {7u, kPairManifoldPoints},
                {8u, kManifoldIRScatter},
                {9u, kContactStatuses},
                {10u, kContacts},
                {11u, kContactMetadata},
                {12u, kIRBlocks},
                {13u, kIREndpoints},
                {14u, kEndpointRuntime},
                {15u, kIRRows},
                {16u, kIRCones},
                {17u, kPointQueries},
                {18u, kBodyDynamicNodes},
                {19u,
                 context.useTaskBodyParameters
                     ? kTaskBodyParameters
                     : kMaterials},
            },
            nullptr,
            0u,
            pairFlagThreadCount *
                MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY
        ) &&
        (
            rodWitnessThreadCount == 0u ||
            encodeContactThreadKernel(
                context,
                commandBuffer,
                context.rodContactScatterPipeline,
                @"MetalWorld rod witness-to-ConstraintIR scatter",
                {
                    {0u, kContactDispatch},
                    {1u, kRodColliders},
                    {2u, kRodToolPairs},
                    {3u, kShapes},
                    {4u, kMaterials},
                    {5u, kCandidateBodies},
                    {6u, kRodWitnessCounts},
                    {7u, kCandidateRodWitnesses},
                    {8u, kRodContactScratch},
                    {9u, kContactStatuses},
                    {10u, kContacts},
                    {11u, kContactMetadata},
                    {12u, kIRBlocks},
                    {13u, kIREndpoints},
                    {14u, kEndpointRuntime},
                    {15u, kIRRows},
                    {16u, kIRCones},
                    {17u, kPointQueries},
                    {18u, kBodyDynamicNodes},
                    {19u, kRodConstraintWitnessIndices},
                },
                nullptr,
                0u,
                rodWitnessThreadCount
            )
        );
}

bool encodeContactCollisionAndSolve(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const MRMetalWorldPassGPU& solverPass,
    const std::size_t factorQ,
    const std::size_t sourceManifoldHeaders,
    const std::size_t sourceManifoldPoints,
    const std::size_t sourceManifoldCounts,
    const std::size_t candidateRodNodes,
    const std::size_t candidateRodEdges,
    const std::size_t rodWitnessThreadCount,
    const std::size_t rodWitnessCount,
    const bool useWave32,
    const mr_u32 activePairClassMask,
    const mr_u32 solverIterationCount,
    const bool enableDistributed,
    const std::size_t environmentCount,
    const std::size_t islandWorkCount,
    const std::size_t tileWorkCount,
    const std::size_t pairFlagThreadCount
) {
    return
        encodeClassCompactedPairNarrowphase(
            context,
            commandBuffer,
            pairFlagThreadCount,
            activePairClassMask
        ) &&
        encodeParallelManifoldCompile(
            context,
            commandBuffer,
            pass,
            sourceManifoldHeaders,
            sourceManifoldPoints,
            sourceManifoldCounts,
            candidateRodNodes,
            environmentCount,
            pairFlagThreadCount,
            rodWitnessThreadCount
        ) &&
        encodeContactThreadKernel(
            context,
            commandBuffer,
            context.factorDispatchPipeline,
            @"MetalWorld active point-query reduction",
            {
                {0u, kContactDispatch},
                {1u, kContactStatuses},
                {2u, kOperatorFactorDispatch},
                {3u, kActiveIndirectDispatch},
            },
            nullptr,
            0u,
            1u
        ) &&
        encodeContactThreadKernel(
            context,
            commandBuffer,
            context.pointQueryTailPipeline,
            @"MetalWorld point-query tail fill",
            {
                {0u, kContactDispatch},
                {1u, kOperatorFactorDispatch},
                {2u, kArticulations},
                {3u, kContactStatuses},
                {4u, kPointQueries},
            },
            nullptr,
            0u,
            environmentCount,
            true,
            sizeof(MRIndirectDispatchArgumentsGPU)
        ) &&
        encodeArticulatedOperator(
            context,
            commandBuffer,
            kOperatorFactorDispatch,
            factorQ,
            kPointQueries,
            kBodyPoses,
            environmentCount,
            @"MetalWorld articulated factor/Jacobians",
            true
        ) &&
        encodeContactThreadKernel(
            context,
            commandBuffer,
            context.evaluateIRPipeline,
            @"MetalWorld ConstraintIR evaluation",
            {
                {0u, kContactDispatch},
                {1u, kContacts},
                {2u, kContacts},
                {3u, kIRBlocks},
                {4u, kIRRows},
                {5u, kIRCones},
                {6u, kCandidateBodies},
                {7u, kCandidateV},
                {8u, kPointJacobians},
                {9u, kOperatorStatuses},
                {10u, kEvaluatedRows},
                {11u, kEvaluatedCones},
                {12u, kFactorCaches},
                {13u, kContactStatuses},
                {14u, kIREndpoints},
                {15u, candidateRodNodes},
            },
            nullptr,
            0u,
            environmentCount,
            true,
            sizeof(MRIndirectDispatchArgumentsGPU)
        ) &&
        encodeStreamedArticulatedResponses(
            context,
            commandBuffer,
            factorQ,
            environmentCount
        ) &&
        encodeContactThreadKernel(
            context,
            commandBuffer,
            context.islandPipeline,
            @"MetalWorld mixed contact islands",
            {
                {0u, kContactDispatch},
                {1u, kCandidateBodies},
                {2u, kContacts},
                {3u, kIRBlocks},
                {4u, kIslands},
                {5u, kContactStatuses},
                {6u, kDynamicNodes},
                {7u, kBodyDynamicNodes},
                {8u, kEndpointRuntime},
                {9u, kIslandNodeReferences},
                {10u, kIslandConstraintReferences},
            },
            nullptr,
            0u,
            environmentCount,
            true,
            sizeof(MRIndirectDispatchArgumentsGPU)
        ) &&
        (
            useWave32
            ? encodeWave32ContactSolve(
                  context,
                  commandBuffer,
                  solverPass,
                  solverIterationCount,
                  enableDistributed,
                  candidateRodNodes,
                  candidateRodEdges,
                  environmentCount,
                  islandWorkCount,
                  tileWorkCount
              )
            : encodeContactThreadKernel(
                  context,
                  commandBuffer,
                  context.contactSolvePipeline,
                  @"MetalWorld exact-cone contact solve",
                  {
                      {0u, kContactDispatch},
                      {1u, kFactorMatrix},
                      {2u, kPointJacobians},
                      {3u, kCandidateV},
                      {4u, kCandidateBodies},
                      {5u, kContacts},
                      {6u, kContactMetadata},
                      {7u, kEvaluatedRows},
                      {8u, kEvaluatedCones},
                      {9u, kResponseColumns},
                      {10u, kCandidateManifoldPoints},
                      {11u, kContactStatuses},
                  },
                  &solverPass,
                  12u,
                  environmentCount,
                  true,
                  sizeof(MRIndirectDispatchArgumentsGPU)
              ) &&
                  encodeRodContactSolve(
                      context,
                      commandBuffer,
                      solverPass,
                      candidateRodNodes,
                      candidateRodEdges,
                      rodWitnessCount,
                      environmentCount
                  )
        );
}

bool encodeHybridContactSubstep(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> abaPipeline,
    const CompiledWorld& world,
    const MRMetalWorldPassGPU& pass,
    const bool finalPhysicsSubstep,
    const std::size_t sourceQ,
    const std::size_t sourceV,
    const std::size_t destinationQ,
    const std::size_t destinationV,
    const std::size_t sourceScene,
    const std::size_t destinationScene,
    const std::size_t sourceManifoldHeaders,
    const std::size_t sourceManifoldPoints,
    const std::size_t sourceManifoldCounts,
    const std::size_t destinationManifoldHeaders,
    const std::size_t destinationManifoldPoints,
    const std::size_t destinationManifoldCounts,
    const std::size_t sourceRodNodes,
    const std::size_t sourceRodEdges,
    const std::size_t candidateRodNodes,
    const std::size_t candidateRodEdges,
    const std::size_t sourceRodWitnesses,
    const std::size_t rodWitnessCount,
    const bool useWave32,
    const mr_u32 activePairClassMask,
    const mr_u32 solverIterationCount,
    const bool enableDistributed,
    const mr_u32 eventPassCount,
    const std::size_t articulationCount,
    const std::size_t environmentCount,
    const std::size_t islandWorkCount,
    const std::size_t tileWorkCount,
    const std::size_t colliderThreadCount,
    const std::size_t pairFlagThreadCount
) {
    MRMetalWorldPassGPU solverPass = pass;
    solverPass.reserved0 = finalPhysicsSubstep ? 1u : 0u;
    if (!encodeContactThreadKernel(
            context,
            commandBuffer,
            context.ccdEventInitializePipeline,
            @"MetalWorld initialize CCD event cursor",
            {
                {0u, kContactDispatch},
                {1u, kCCDEventStatesA},
                {2u, kCCDEventStatesB},
                {3u, kCCDImpactClusters},
                {4u, kContactStatuses},
            },
            &pass,
            5u,
            environmentCount
        ) ||
        (
            world.rodCount() != 0u &&
            !encodeContactThreadKernel(
                context,
                commandBuffer,
                context.rodEventInitializePipeline,
                @"MetalWorld initialize rod CCD event state",
                {
                    {0u, kContactDispatch},
                    {1u, sourceRodNodes},
                    {2u, sourceRodEdges},
                    {3u, sourceRodWitnesses},
                    {4u, kCCDEventRodNodesA},
                    {5u, kCCDEventRodEdgesA},
                    {6u, kCCDEventRodWitnessesA},
                    {7u, kCCDEventRodNodesB},
                    {8u, kCCDEventRodEdgesB},
                    {9u, kCCDEventRodWitnessesB},
                },
                nullptr,
                0u,
                environmentCount
            )
        )) {
        return false;
    }

    std::size_t eventSourceQ = sourceQ;
    std::size_t eventSourceV = sourceV;
    std::size_t eventSourceScene = sourceScene;
    std::size_t eventSourceHeaders = sourceManifoldHeaders;
    std::size_t eventSourcePoints = sourceManifoldPoints;
    std::size_t eventSourceCounts = sourceManifoldCounts;
    std::size_t eventDestinationQ = destinationQ;
    std::size_t eventDestinationV = destinationV;
    std::size_t eventDestinationScene = destinationScene;
    std::size_t eventDestinationHeaders =
        destinationManifoldHeaders;
    std::size_t eventDestinationPoints =
        destinationManifoldPoints;
    std::size_t eventDestinationCounts =
        destinationManifoldCounts;
    std::size_t eventStateIn = kCCDEventStatesA;
    std::size_t eventStateOut = kCCDEventStatesB;
    std::size_t eventSourceRodNodes = kCCDEventRodNodesA;
    std::size_t eventSourceRodEdges = kCCDEventRodEdgesA;
    std::size_t eventSourceRodWitnesses =
        kCCDEventRodWitnessesA;
    std::size_t eventDestinationRodNodes =
        kCCDEventRodNodesB;
    std::size_t eventDestinationRodEdges =
        kCCDEventRodEdgesB;
    std::size_t eventDestinationRodWitnesses =
        kCCDEventRodWitnessesB;
    const mr_u32 remainingMode = MR_CCD_SEGMENT_REMAINING;
    const mr_u32 selectedMode = MR_CCD_SEGMENT_SELECTED;

    for (mr_u32 eventPass = 0u;
         eventPass < eventPassCount;
         ++eventPass) {
        if ((eventPass != 0u &&
             !encodeContactThreadKernel(
                 context,
                 commandBuffer,
                 context.ccdEventPreparePipeline,
                 @"MetalWorld prepare active CCD environments",
                 {
                     {0u, kContactDispatch},
                     {1u, eventStateIn},
                     {2u, kContactStatuses},
                 },
                 nullptr,
                 0u,
                 environmentCount
             )) ||
            !encodeABA(
                context,
                commandBuffer,
                abaPipeline,
                eventSourceQ,
                eventSourceV,
                articulationCount,
                environmentCount
            ) ||
            !encodeArticulatedOperator(
                context,
                commandBuffer,
                kOperatorKinematicsDispatch,
                eventSourceQ,
                kPointQueries,
                kBodyPoses,
                environmentCount,
                @"MetalWorld event-source articulation kinematics",
                false
            ) ||
            !encodeContactThreadKernel(
                context,
                commandBuffer,
                context.bodyProjectionPipeline,
                @"MetalWorld event-source body projection",
                {
                    {0u, kContactDispatch},
                    {1u, kArticulations},
                    {2u, kBodies},
                    {3u, kSceneBodyIndices},
                    {4u, kBodyPoses},
                    {5u, kOperatorStatuses},
                    {6u, eventSourceScene},
                    {7u, kCurrentBodies},
                    {8u, kContactStatuses},
                },
                nullptr,
                0u,
                environmentCount
            ) ||
            !encodeContactThreadKernel(
                context,
                commandBuffer,
                context.eventArticulationPipeline,
                @"MetalWorld remaining-time articulation prediction",
                {
                    {0u, kContactDispatch},
                    {1u, kArticulations},
                    {2u, kJoints},
                    {3u, eventSourceQ},
                    {4u, eventSourceV},
                    {5u, kCandidateAcceleration},
                    {6u, eventStateIn},
                    {7u, kCandidateQ},
                    {8u, kCandidateV},
                    {9u, kABAStatuses},
                    {10u, kContactStatuses},
                },
                nullptr,
                0u,
                environmentCount,
                false,
                0u,
                &remainingMode,
                sizeof(remainingMode),
                11u
            ) ||
            !encodeArticulatedOperator(
                context,
                commandBuffer,
                kOperatorKinematicsDispatch,
                kCandidateQ,
                kPointQueries,
                kFutureBodyPoses,
                environmentCount,
                @"MetalWorld remaining-time articulation prediction",
                false
            ) ||
            !encodeContactThreadKernel(
                context,
                commandBuffer,
                context.eventScenePredictionPipeline,
                @"MetalWorld remaining-time scene prediction",
                {
                    {0u, kWorld},
                    {1u, kContactDispatch},
                    {2u, kBodies},
                    {3u, kSceneBodyIndices},
                    {4u, kCurrentBodies},
                    {5u, eventStateIn},
                    {6u, kCandidateBodies},
                    {7u, kContactStatuses},
                },
                nullptr,
                0u,
                environmentCount,
                false,
                0u,
	                &remainingMode,
	                sizeof(remainingMode),
	                8u
	            ) ||
	            (
	                world.rodCount() != 0u &&
	                !encodeRodSubstep(
	                    context,
	                    commandBuffer,
	                    pass,
	                    world,
	                    eventSourceRodNodes,
	                    eventSourceRodEdges,
	                    candidateRodNodes,
	                    candidateRodEdges,
	                    eventStateIn,
	                    remainingMode,
	                    environmentCount
	                )
	            ) ||
	            !encodeContactThreadKernel(
	                context,
                commandBuffer,
                context.sweptProjectionPipeline,
                @"MetalWorld event swept collider projection",
                {
                    {0u, kContactDispatch},
                    {1u, kShapes},
                    {2u, kArticulations},
                    {3u, kCurrentBodies},
                    {4u, kCandidateBodies},
                    {5u, kFutureBodyPoses},
                    {6u, kCandidateV},
                    {7u, kGeometryHeaders},
                    {8u, kProjectedColliders},
                    {9u, kFutureProjectedColliders},
                    {10u, kContactStatuses},
                    {11u, eventStateIn},
                },
                nullptr,
                0u,
	                colliderThreadCount
	            ) ||
	            (
	                world.rodCount() != 0u &&
	                !encodeContactThreadKernel(
	                    context,
	                    commandBuffer,
	                    context.rodSweptProjectionPipeline,
	                    @"MetalWorld event swept rod projection",
	                    {
	                        {0u, kContactDispatch},
	                        {1u, kRodColliders},
	                        {2u, eventSourceRodNodes},
	                        {3u, candidateRodNodes},
	                        {4u, kProjectedRodColliders},
	                        {5u, kFutureProjectedRodColliders},
	                        {6u, kContactStatuses},
	                    },
	                    nullptr,
	                    0u,
	                    environmentCount *
	                        world.rodEdgeCount()
	                )
	            ) ||
	            !encodeContactThreadKernel(
                context,
                commandBuffer,
                context.pairFlagPipeline,
                @"MetalWorld swept broadphase flags",
                {
                    {0u, kContactDispatch},
                    {1u, kShapes},
                    {2u, kEligiblePairs},
                    {3u, kProjectedColliders},
                    {4u, kPairOverlapFlags},
                },
                nullptr,
                0u,
                pairFlagThreadCount
            ) ||
            !encodeContactThreadKernel(
                context,
                commandBuffer,
                context.ccdPipeline,
                @"MetalWorld exact CCD candidate query",
                {
                    {0u, kContactDispatch},
                    {1u, kShapes},
                    {2u, kEligiblePairs},
                    {3u, kPairOverlapFlags},
                    {4u, kProjectedColliders},
                    {5u, kFutureProjectedColliders},
                    {6u, kGeometryHeaders},
                    {7u, kGeometryVertices},
                    {8u, kMeshBvhNodes},
                    {9u, kMeshTriangles},
                    {10u, kCCDPairs},
                    {11u, kContactStatuses},
                    {12u, eventStateIn},
                },
                nullptr,
                0u,
                environmentCount
            ) ||
            (
                world.rodCount() != 0u &&
                !encodeContactThreadKernel(
                    context,
                    commandBuffer,
                    context.rodCCDPipeline,
                    @"MetalWorld rod/tool conservative advancement",
                    {
                        {0u, kContactDispatch},
                        {1u, kRodColliders},
                        {2u, kRodShapeSources},
                        {3u, kRodToolPairs},
                        {4u, kShapes},
                        {5u, kProjectedRodColliders},
                        {6u, kFutureProjectedRodColliders},
                        {7u, kProjectedColliders},
                        {8u, kFutureProjectedColliders},
                        {9u, kGeometryHeaders},
                        {10u, kGeometryVertices},
                        {11u, kMeshBvhNodes},
                        {12u, kMeshTriangles},
                        {13u, kCCDPairs},
                        {14u, kContactStatuses},
                        {15u, eventStateIn},
                    },
                    nullptr,
                    0u,
                    environmentCount
                )
            ) ||
            !encodeContactThreadKernel(
                context,
                commandBuffer,
                context.ccdEventSelectPipeline,
                @"MetalWorld deterministic CCD event cluster",
                {
                    {0u, kContactDispatch},
                    {1u, kCCDPairs},
                    {2u, eventStateIn},
                    {3u, eventStateOut},
                    {4u, kCCDImpactClusters},
                    {5u, kContactStatuses},
                },
                &pass,
                6u,
                environmentCount,
                false,
                0u,
                &eventPass,
                sizeof(eventPass),
                7u
            ) ||
            !encodeContactThreadKernel(
                context,
                commandBuffer,
                context.eventArticulationPipeline,
                @"MetalWorld selected-TOI articulation advance",
                {
                    {0u, kContactDispatch},
                    {1u, kArticulations},
                    {2u, kJoints},
                    {3u, eventSourceQ},
                    {4u, eventSourceV},
                    {5u, kCandidateAcceleration},
                    {6u, eventStateOut},
                    {7u, kCandidateQ},
                    {8u, kCandidateV},
                    {9u, kABAStatuses},
                    {10u, kContactStatuses},
                },
                nullptr,
                0u,
                environmentCount,
                false,
                0u,
                &selectedMode,
                sizeof(selectedMode),
                11u
            ) ||
            !encodeArticulatedOperator(
                context,
                commandBuffer,
                kOperatorKinematicsDispatch,
                kCandidateQ,
                kPointQueries,
                kBodyPoses,
                environmentCount,
                @"MetalWorld selected-TOI articulation kinematics",
                false
            ) ||
            !encodeContactThreadKernel(
                context,
                commandBuffer,
                context.eventScenePredictionPipeline,
                @"MetalWorld selected-TOI scene advance",
                {
                    {0u, kWorld},
                    {1u, kContactDispatch},
                    {2u, kBodies},
                    {3u, kSceneBodyIndices},
                    {4u, kCurrentBodies},
                    {5u, eventStateOut},
                    {6u, kCandidateBodies},
                    {7u, kContactStatuses},
                },
                nullptr,
                0u,
                environmentCount,
                false,
                0u,
                &selectedMode,
                sizeof(selectedMode),
                8u
            ) ||
            (
                world.rodCount() != 0u &&
                !encodeRodSubstep(
                    context,
                    commandBuffer,
                    pass,
                    world,
                    eventSourceRodNodes,
                    eventSourceRodEdges,
                    candidateRodNodes,
                    candidateRodEdges,
                    eventStateOut,
                    selectedMode,
                    environmentCount
                )
            ) ||
            !encodeContactThreadKernel(
                context,
                commandBuffer,
                context.eventBodyOverlayPipeline,
                @"MetalWorld selected-TOI articulation overlay",
                {
                    {0u, kContactDispatch},
                    {1u, kArticulations},
                    {2u, kBodies},
                    {3u, kBodyPoses},
                    {4u, kOperatorStatuses},
                    {5u, kCandidateBodies},
                    {6u, kContactStatuses},
                },
                nullptr,
                0u,
                environmentCount
            ) ||
            !encodeContactThreadKernel(
                context,
                commandBuffer,
                context.eventColliderProjectionPipeline,
                @"MetalWorld selected-TOI collider projection",
                {
                    {0u, kContactDispatch},
                    {1u, kShapes},
                    {2u, kGeometryHeaders},
                    {3u, kCandidateBodies},
                    {4u, kProjectedColliders},
                    {5u, kContactStatuses},
                },
                nullptr,
                0u,
                colliderThreadCount
            ) ||
            !encodeContactThreadKernel(
                context,
                commandBuffer,
                context.pairFlagPipeline,
                @"MetalWorld selected-TOI broadphase flags",
                {
                    {0u, kContactDispatch},
                    {1u, kShapes},
                    {2u, kEligiblePairs},
                    {3u, kProjectedColliders},
                    {4u, kPairOverlapFlags},
                },
                nullptr,
                0u,
                pairFlagThreadCount
            ) ||
            (
                world.rodCount() != 0u &&
                !encodeRodToolNarrowphase(
                    context,
                    commandBuffer,
                    world,
                    eventSourceRodWitnesses,
                    environmentCount
                )
            ) ||
            (
                rodWitnessCount != 0u &&
                !encodeContactThreadKernel(
                    context,
                    commandBuffer,
                    context.rodCCDWitnessTagPipeline,
                    @"MetalWorld tag rod CCD impact witnesses",
                    {
                        {0u, kContactDispatch},
                        {1u, kCCDPairs},
                        {2u, kContactStatuses},
                        {3u, kCandidateRodWitnesses},
                    },
                    nullptr,
                    0u,
                    rodWitnessCount
                )
            ) ||
            !encodeContactCollisionAndSolve(
                context,
                commandBuffer,
                pass,
                solverPass,
                kCandidateQ,
                eventSourceHeaders,
                eventSourcePoints,
                eventSourceCounts,
                candidateRodNodes,
                candidateRodEdges,
                rodWitnessCount,
                rodWitnessCount,
                useWave32,
                activePairClassMask,
                solverIterationCount,
                enableDistributed,
                environmentCount,
                islandWorkCount,
                tileWorkCount,
                pairFlagThreadCount
            ) ||
            !encodeContactThreadKernel(
                context,
                commandBuffer,
                context.jointLimitPipeline,
                @"MetalWorld selected-TOI joint limits",
                {
                    {0u, kContactDispatch},
                    {1u, kArticulations},
                    {2u, kDofs},
                    {3u, eventSourceQ},
                    {4u, kCandidateQ},
                    {5u, kCandidateV},
                    {6u, kContactStatuses},
                },
                nullptr,
                0u,
                environmentCount,
                false,
                0u,
                &selectedMode,
                sizeof(selectedMode),
                7u
            ) ||
            !encodeContactThreadKernel(
                context,
                commandBuffer,
                context.inactiveEventRestorePipeline,
                @"MetalWorld restore finished CCD environments",
                {
                    {0u, kContactDispatch},
                    {1u, kArticulations},
                    {2u, kSceneBodyIndices},
                    {3u, eventSourceQ},
                    {4u, eventSourceV},
                    {5u, eventSourceScene},
                    {6u, eventSourceHeaders},
                    {7u, eventSourcePoints},
                    {8u, eventSourceCounts},
                    {9u, kCandidateQ},
                    {10u, kCandidateV},
                    {11u, kCandidateBodies},
                    {12u, kCandidateManifoldHeaders},
                    {13u, kCandidateManifoldPoints},
                    {14u, kCandidateManifoldCounts},
                    {15u, kContactStatuses},
                },
                nullptr,
                0u,
                environmentCount
            ) ||
            (
                world.rodCount() != 0u &&
                !encodeContactThreadKernel(
                    context,
                    commandBuffer,
                    context.inactiveRodEventRestorePipeline,
                    @"MetalWorld restore finished rod CCD environments",
                    {
                        {0u, kContactDispatch},
                        {1u, eventSourceRodNodes},
                        {2u, eventSourceRodEdges},
                        {3u, eventSourceRodWitnesses},
                        {4u, candidateRodNodes},
                        {5u, candidateRodEdges},
                        {6u, kCandidateRodWitnesses},
                        {7u, kContactStatuses},
                    },
                    nullptr,
                    0u,
                    environmentCount
                )
            ) ||
            !encodeContactThreadKernel(
                context,
                commandBuffer,
                context.ccdEventFinalizePipeline,
                @"MetalWorld CCD event-time closeout",
                {
                    {0u, kContactDispatch},
                    {1u, eventStateOut},
                    {2u, kContactStatuses},
                },
                nullptr,
                0u,
                environmentCount,
                false,
                0u,
                &eventPass,
                sizeof(eventPass),
                3u
            )) {
            return false;
        }

        const bool finalEventPass =
            eventPass + 1u == eventPassCount;
        if (!finalEventPass) {
            if (!encodeContactThreadKernel(
                    context,
                    commandBuffer,
                    context.eventSegmentPublishPipeline,
                    @"MetalWorld publish accepted CCD segment",
                    {
                        {0u, kContactDispatch},
                        {1u, kArticulations},
                        {2u, kSceneBodyIndices},
                        {3u, eventSourceQ},
                        {4u, eventSourceV},
                        {5u, eventSourceScene},
                        {6u, eventSourceHeaders},
                        {7u, eventSourcePoints},
                        {8u, eventSourceCounts},
                        {9u, kCandidateQ},
                        {10u, kCandidateV},
                        {11u, kCandidateBodies},
                        {12u, kCandidateManifoldHeaders},
                        {13u, kCandidateManifoldPoints},
                        {14u, kCandidateManifoldCounts},
                        {15u, kContactStatuses},
                        {16u, eventDestinationQ},
                        {17u, eventDestinationV},
                        {18u, eventDestinationScene},
                        {19u, eventDestinationHeaders},
                        {20u, eventDestinationPoints},
                        {21u, eventDestinationCounts},
                    },
                    nullptr,
                    0u,
                    environmentCount
                ) ||
                (
                    world.rodCount() != 0u &&
                    !encodeContactThreadKernel(
                        context,
                        commandBuffer,
                        context.rodEventSegmentPublishPipeline,
                        @"MetalWorld publish accepted rod CCD segment",
                        {
                            {0u, kContactDispatch},
                            {1u, eventSourceRodNodes},
                            {2u, eventSourceRodEdges},
                            {3u, eventSourceRodWitnesses},
                            {4u, candidateRodNodes},
                            {5u, candidateRodEdges},
                            {6u, kCandidateRodWitnesses},
                            {7u, kContactStatuses},
                            {8u, eventDestinationRodNodes},
                            {9u, eventDestinationRodEdges},
                            {10u, eventDestinationRodWitnesses},
                        },
                        nullptr,
                        0u,
                        environmentCount
                    )
                )) {
                return false;
            }
            std::swap(eventSourceQ, eventDestinationQ);
            std::swap(eventSourceV, eventDestinationV);
            std::swap(eventSourceScene, eventDestinationScene);
            std::swap(
                eventSourceHeaders,
                eventDestinationHeaders
            );
            std::swap(
                eventSourcePoints,
                eventDestinationPoints
            );
            std::swap(
                eventSourceCounts,
                eventDestinationCounts
            );
            std::swap(
                eventSourceRodNodes,
                eventDestinationRodNodes
            );
            std::swap(
                eventSourceRodEdges,
                eventDestinationRodEdges
            );
            std::swap(
                eventSourceRodWitnesses,
                eventDestinationRodWitnesses
            );
            std::swap(eventStateIn, eventStateOut);
        }
    }

    if (!encodeContactThreadKernel(
            context,
            commandBuffer,
            context.contactLatchPipeline,
            @"MetalWorld contact failure latch",
            {
                {0u, kWorldDispatch},
                {2u, kContactStatuses},
                {3u, kEnvironmentStatuses},
            },
            &pass,
            1u,
            environmentCount
        ) ||
        !encodeCommit(
            context,
            commandBuffer,
            pass,
            destinationQ,
            destinationV,
            environmentCount
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.contactCommitPipeline,
            @"MetalWorld contact transactional commit",
            {
                {0u, kWorldDispatch},
                {1u, kContactDispatch},
                {3u, kEnvironmentStatuses},
                {4u, kSceneBodyIndices},
                {5u, kCandidateBodies},
                {6u, kCheckpointSceneBodies},
                {7u, destinationScene},
                {8u, kCandidateManifoldHeaders},
                {9u, kCandidateManifoldPoints},
                {10u, kCandidateManifoldCounts},
                {11u, kCheckpointManifoldHeaders},
                {12u, kCheckpointManifoldPoints},
                {13u, kCheckpointManifoldCounts},
                {14u, destinationManifoldHeaders},
                {15u, destinationManifoldPoints},
                {16u, destinationManifoldCounts},
            },
            &pass,
            2u,
            environmentCount
        )) {
        return false;
    }
    return !finalPhysicsSubstep ||
        encodeContactThreadKernel(
            context,
            commandBuffer,
            context.convexCachePublishPipeline,
            @"MetalWorld transactional convex-cache publication",
            {
                {0u, kContactDispatch},
                {1u, kEligiblePairs},
                {2u, kPairOverlapFlags},
                {3u, kContactStatuses},
                {4u, kCandidateConvexCaches},
                {5u, kConvexCaches},
            },
            nullptr,
            0u,
            pairFlagThreadCount
        );
}

bool encodeUnifiedQualitySolve(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const std::size_t candidateRodNodes,
    const std::size_t candidateRodEdges,
    const std::size_t environmentCount
) {
    id<MTLComputeCommandEncoder> prepare =
        [commandBuffer computeCommandEncoder];
    if (prepare == nil) {
        return false;
    }
    prepare.label =
        @"MetalWorld unified quality operator preparation";
    [prepare
        setComputePipelineState:
            context.qualityPreparePipeline];
    const std::array<std::size_t, 29u> prepareBuffers{{
        kContactDispatch,
        kQualityDispatch,
        kSceneBodyIndices,
        kFactorMatrix,
        kPointJacobians,
        kCandidateV,
        kCandidateBodies,
        kContacts,
        kContactMetadata,
        kIRBlocks,
        kEvaluatedRows,
        kEvaluatedCones,
        kContactStatuses,
        kQualityBlocks,
        kQualityDynamics,
        kQualityJacobian,
        kQualityBias,
        kQualityFreeVelocity,
        kQualityWarmVelocity,
        kQualityWarmImpulses,
        candidateRodNodes,
        candidateRodEdges,
        kRodInverseMasses,
        kRodInverseRotationalInertias,
        kRodColliders,
        kCandidateRodWitnesses,
        kRodConstraintWitnessIndices,
        kRodFactorCaches,
        kOperatorVelocityArena,
    }};
    for (NSUInteger argument = 0u;
         argument < prepareBuffers.size();
         ++argument) {
        [prepare setBuffer:context.buffers[
                               prepareBuffers[argument]
                           ]
                    offset:0u
                   atIndex:argument];
    }
    [prepare
        dispatchThreadgroups:MTLSizeMake(
            static_cast<NSUInteger>(environmentCount),
            1u,
            1u
        )
        threadsPerThreadgroup:MTLSizeMake(
            MR_SIMD_WIDTH,
            1u,
            1u
        )];
    [prepare endEncoding];

    id<MTLComputeCommandEncoder> warmStart =
        [commandBuffer computeCommandEncoder];
    if (warmStart == nil) {
        return false;
    }
    warmStart.label =
        @"MetalWorld persistent-impulse quality warm start";
    [warmStart
        setComputePipelineState:
            context.qualityWarmStartPipeline];
    const std::array<std::size_t, 11u> warmStartBuffers{{
        kContactDispatch,
        kQualityDispatch,
        kFactorMatrix,
        kQualityDynamics,
        kQualityJacobian,
        kQualityFreeVelocity,
        kQualityWarmImpulses,
        kQualityWarmVelocity,
        kContactStatuses,
        kRodFactorCaches,
        kOperatorVelocityArena,
    }};
    for (NSUInteger argument = 0u;
         argument < warmStartBuffers.size();
         ++argument) {
        [warmStart setBuffer:context.buffers[
                                  warmStartBuffers[argument]
                              ]
                       offset:0u
                      atIndex:argument];
    }
    [warmStart
        dispatchThreadgroups:MTLSizeMake(
            static_cast<NSUInteger>(environmentCount),
            1u,
            1u
        )
        threadsPerThreadgroup:MTLSizeMake(
            MR_SIMD_WIDTH,
            1u,
            1u
        )];
    [warmStart endEncoding];

    id<MTLComputeCommandEncoder> queue =
        [commandBuffer computeCommandEncoder];
    if (queue == nil) {
        return false;
    }
    queue.label =
        @"MetalWorld stable unified-quality packet compaction";
    [queue setComputePipelineState:context.qualityQueuePipeline];
    const std::array<std::size_t, 5u> queueBuffers{{
        kContactDispatch,
        kQualityDispatch,
        kContactStatuses,
        kQualityWorkQueue,
        kQualityWorkPackets,
    }};
    for (NSUInteger argument = 0u;
         argument < queueBuffers.size();
         ++argument) {
        [queue setBuffer:context.buffers[
                             queueBuffers[argument]
                         ]
                  offset:0u
                 atIndex:argument];
    }
    [queue
        dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(
            MR_SIMD_WIDTH,
            1u,
            1u
        )];
    [queue endEncoding];

    id<MTLComputeCommandEncoder> solve =
        [commandBuffer computeCommandEncoder];
    if (solve == nil) {
        return false;
    }
    solve.label =
        @"MetalWorld unified matrix-free quality Newton";
    [solve setComputePipelineState:context.qualitySolvePipeline];
    const std::array<std::size_t, 15u> solveBuffers{{
        kQualityDispatch,
        kQualityBlocks,
        kQualityDynamics,
        kQualityJacobian,
        kQualityBias,
        kQualityFreeVelocity,
        kQualityWarmVelocity,
        kQualityWarmImpulses,
        kQualityOutputVelocity,
        kQualityOutputImpulses,
        kQualityDerivatives,
        kQualityHessian,
        kQualityStatuses,
        kQualityWorkPackets,
        kQualityWorkQueue,
    }};
    for (NSUInteger argument = 0u;
         argument < solveBuffers.size();
         ++argument) {
        [solve setBuffer:context.buffers[
                             solveBuffers[argument]
                         ]
                  offset:0u
                 atIndex:argument];
    }
    [solve
        dispatchThreadgroupsWithIndirectBuffer:
            context.buffers[kQualityWorkQueue]
        indirectBufferOffset:
            offsetof(MRUnifiedQualityWorkQueueGPU, indirect)
        threadsPerThreadgroup:MTLSizeMake(
            MR_SIMD_WIDTH,
            1u,
            1u
        )];
    [solve endEncoding];

    if (!encodeContactThreadKernel(
            context,
            commandBuffer,
            context.qualityApplyPipeline,
            @"MetalWorld unified quality transactional application",
            {
                {0u, kContactDispatch},
                {1u, kQualityDispatch},
                {2u, kSceneBodyIndices},
                {3u, kQualityStatuses},
                {4u, kQualityOutputVelocity},
                {5u, kQualityOutputImpulses},
                {6u, kCandidateV},
                {7u, kCandidateBodies},
                {8u, kContacts},
                {9u, kContactMetadata},
                {10u, kCandidateManifoldPoints},
                {11u, kContactStatuses},
                {12u, candidateRodNodes},
                {13u, candidateRodEdges},
                {14u, kCandidateRodWitnesses},
                {15u, kRodConstraintWitnessIndices},
            },
            nullptr,
            0u,
            environmentCount
        )) {
        return false;
    }
    return encodeContactThreadKernel(
        context,
        commandBuffer,
        context.qualityQueueStatusPipeline,
        @"MetalWorld unified quality queue evidence",
        {
            {0u, kContactDispatch},
            {1u, kQualityWorkQueue},
            {2u, kContactStatuses},
        },
        nullptr,
        0u,
        environmentCount
    );
}

bool encodeContactSubstep(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const CompiledWorld& world,
    const MRMetalWorldPassGPU& pass,
    const bool finalPhysicsSubstep,
    const std::size_t sourceQ,
    const std::size_t destinationQ,
    const std::size_t destinationV,
    const std::size_t sourceScene,
    const std::size_t destinationScene,
    const std::size_t sourceManifoldHeaders,
    const std::size_t sourceManifoldPoints,
    const std::size_t sourceManifoldCounts,
    const std::size_t sourceRodWitnesses,
    const std::size_t candidateRodNodes,
    const std::size_t candidateRodEdges,
    const std::size_t rodWitnessCount,
    const std::size_t destinationManifoldHeaders,
    const std::size_t destinationManifoldPoints,
    const std::size_t destinationManifoldCounts,
    const bool useWave32,
    const bool useQuality,
    const mr_u32 activePairClassMask,
    const bool hasFutureKinematics,
    const bool useHybridCCD,
    const mr_u32 solverIterationCount,
    const bool enableDistributed,
    const std::size_t environmentCount,
    const std::size_t islandWorkCount,
    const std::size_t tileWorkCount,
    const std::size_t colliderThreadCount,
    const std::size_t pairFlagThreadCount
) {
    MRMetalWorldPassGPU solverPass = pass;
    solverPass.reserved0 = finalPhysicsSubstep ? 1u : 0u;
    const mr_u32 eventPass = 0u;
    const mr_u32 stateNotIntegrated = 0u;
    if ((useHybridCCD &&
         !encodeContactThreadKernel(
             context,
             commandBuffer,
             context.ccdEventInitializePipeline,
             @"MetalWorld initialize CCD event cursor",
             {
                 {0u, kContactDispatch},
                 {1u, kCCDEventStatesA},
                 {2u, kCCDEventStatesB},
                 {3u, kCCDImpactClusters},
                 {4u, kContactStatuses},
             },
             &pass,
             5u,
             environmentCount
         )) ||
        !encodeArticulatedOperator(
            context,
            commandBuffer,
            kOperatorKinematicsDispatch,
            sourceQ,
            kPointQueries,
            kBodyPoses,
            environmentCount,
            @"MetalWorld articulation kinematics",
            false
        ) ||
        (hasFutureKinematics &&
         !encodeArticulatedOperator(
             context,
             commandBuffer,
             kOperatorKinematicsDispatch,
             kCandidateQ,
             kPointQueries,
             kFutureBodyPoses,
             environmentCount,
             @"MetalWorld unconstrained future kinematics",
             false
         )) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.bodyProjectionPipeline,
            @"MetalWorld body/collider projection",
            {
                {0u, kContactDispatch},
                {1u, kArticulations},
                {2u, kBodies},
                {3u, kSceneBodyIndices},
                {4u, kBodyPoses},
                {5u, kOperatorStatuses},
                {6u, sourceScene},
                {7u, kCurrentBodies},
                {8u, kContactStatuses},
            },
            nullptr,
            0u,
            environmentCount
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.scenePredictionPipeline,
            @"MetalWorld scene prediction",
            {
                {0u, kWorld},
                {1u, kContactDispatch},
                {2u, kBodies},
                {3u, kSceneBodyIndices},
                {4u, kCurrentBodies},
                {5u, kCandidateBodies},
                {6u, kContactStatuses},
            },
            nullptr,
            0u,
            environmentCount
        ) ||
        !encodeRodToolNarrowphase(
            context,
            commandBuffer,
            world,
            sourceRodWitnesses,
            environmentCount
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.sweptProjectionPipeline,
            @"MetalWorld swept/speculative collider projection",
            {
                {0u, kContactDispatch},
                {1u, kShapes},
                {2u, kArticulations},
                {3u, kCurrentBodies},
                {4u, kCandidateBodies},
                {5u, kFutureBodyPoses},
                {6u, kCandidateV},
                {7u, kGeometryHeaders},
                {8u, kProjectedColliders},
                {9u, kFutureProjectedColliders},
                {10u, kContactStatuses},
                {11u, kCCDEventStatesA},
            },
            nullptr,
            0u,
            colliderThreadCount
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.pairFlagPipeline,
            @"MetalWorld compiled-pair broadphase flags",
            {
                {0u, kContactDispatch},
                {1u, kShapes},
                {2u, kEligiblePairs},
                {3u, kProjectedColliders},
                {4u, kPairOverlapFlags},
            },
            nullptr,
            0u,
            pairFlagThreadCount
        ) ||
        (useHybridCCD &&
         !encodeContactThreadKernel(
             context,
             commandBuffer,
             context.ccdPipeline,
             @"MetalWorld hybrid conservative advancement",
             {
                 {0u, kContactDispatch},
                 {1u, kShapes},
                 {2u, kEligiblePairs},
                 {3u, kPairOverlapFlags},
                 {4u, kProjectedColliders},
                 {5u, kFutureProjectedColliders},
                 {6u, kGeometryHeaders},
                 {7u, kGeometryVertices},
                 {8u, kMeshBvhNodes},
                 {9u, kMeshTriangles},
                 {10u, kCCDPairs},
                 {11u, kContactStatuses},
                 {12u, kCCDEventStatesA},
             },
             nullptr,
             0u,
             environmentCount
         )) ||
        (useHybridCCD &&
         !encodeContactThreadKernel(
             context,
             commandBuffer,
             context.ccdEventSelectPipeline,
             @"MetalWorld deterministic CCD event cluster",
             {
                 {0u, kContactDispatch},
                 {1u, kCCDPairs},
                 {2u, kCCDEventStatesA},
                 {3u, kCCDEventStatesB},
                 {4u, kCCDImpactClusters},
                 {5u, kContactStatuses},
             },
             &pass,
             6u,
             environmentCount,
             false,
             0u,
             &eventPass,
             sizeof(eventPass),
             7u
         )) ||
        !encodeClassCompactedPairNarrowphase(
            context,
            commandBuffer,
            pairFlagThreadCount,
            activePairClassMask
        ) ||
        !encodeParallelManifoldCompile(
            context,
            commandBuffer,
            pass,
            sourceManifoldHeaders,
            sourceManifoldPoints,
            sourceManifoldCounts,
            candidateRodNodes,
            environmentCount,
            pairFlagThreadCount,
            environmentCount *
                world.rodToolPairs().size() *
                MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.factorDispatchPipeline,
            @"MetalWorld active point-query reduction",
            {
                {0u, kContactDispatch},
                {1u, kContactStatuses},
                {2u, kOperatorFactorDispatch},
                {3u, kActiveIndirectDispatch},
            },
            nullptr,
            0u,
            1u
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.pointQueryTailPipeline,
            @"MetalWorld point-query tail fill",
            {
                {0u, kContactDispatch},
                {1u, kOperatorFactorDispatch},
                {2u, kArticulations},
                {3u, kContactStatuses},
                {4u, kPointQueries},
            },
            nullptr,
            0u,
            environmentCount,
            true,
            sizeof(MRIndirectDispatchArgumentsGPU)
        ) ||
        !encodeArticulatedOperator(
            context,
            commandBuffer,
            kOperatorFactorDispatch,
            sourceQ,
            kPointQueries,
            kBodyPoses,
            environmentCount,
            @"MetalWorld articulated factor/Jacobians",
            true
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.evaluateIRPipeline,
            @"MetalWorld ConstraintIR evaluation",
            {
                {0u, kContactDispatch},
                {1u, kContacts},
                {2u, kContacts},
                {3u, kIRBlocks},
                {4u, kIRRows},
                {5u, kIRCones},
                {6u, kCandidateBodies},
                {7u, kCandidateV},
                {8u, kPointJacobians},
                {9u, kOperatorStatuses},
                {10u, kEvaluatedRows},
                {11u, kEvaluatedCones},
                {12u, kFactorCaches},
                {13u, kContactStatuses},
                {14u, kIREndpoints},
                {15u, candidateRodNodes},
            },
            nullptr,
            0u,
            environmentCount,
            true,
            sizeof(MRIndirectDispatchArgumentsGPU)
        ) ||
        !encodeStreamedArticulatedResponses(
            context,
            commandBuffer,
            sourceQ,
            environmentCount
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.islandPipeline,
            @"MetalWorld mixed contact islands",
            {
                {0u, kContactDispatch},
                {1u, kCandidateBodies},
                {2u, kContacts},
                {3u, kIRBlocks},
                {4u, kIslands},
                {5u, kContactStatuses},
                {6u, kDynamicNodes},
                {7u, kBodyDynamicNodes},
                {8u, kEndpointRuntime},
                {9u, kIslandNodeReferences},
                {10u, kIslandConstraintReferences},
            },
            nullptr,
            0u,
            environmentCount,
            true,
            sizeof(MRIndirectDispatchArgumentsGPU)
        ) ||
        (
            !useQuality &&
            !encodeContactThreadKernel(
                context,
                commandBuffer,
                context.generalizedConstraintSolvePipeline,
                @"MetalWorld pre-contact typed scalar IR sweep",
                {
                    {0u, kContactDispatch},
                    {1u, kFactorMatrix},
                    {2u, kCandidateV},
                    {3u, kContacts},
                    {4u, kIRBlocks},
                    {5u, kIREndpoints},
                    {6u, kEvaluatedRows},
                    {7u, kContactStatuses},
                    {9u, kCandidateBodies},
                    {10u, candidateRodNodes},
                    {11u, kRodInverseMasses},
                },
                &solverPass,
                8u,
                environmentCount
            )
        ) ||
        !(useQuality
              ? encodeUnifiedQualitySolve(
                    context,
                    commandBuffer,
                    candidateRodNodes,
                    candidateRodEdges,
                    environmentCount
                )
              : useWave32
              ? encodeWave32ContactSolve(
                    context,
                    commandBuffer,
                    solverPass,
                    solverIterationCount,
                    enableDistributed,
                    candidateRodNodes,
                    candidateRodEdges,
                    environmentCount,
                    islandWorkCount,
                    tileWorkCount
                )
              : encodeContactThreadKernel(
                    context,
                    commandBuffer,
                    context.contactSolvePipeline,
                    @"MetalWorld exact-cone contact solve",
                    {
                        {0u, kContactDispatch},
                        {1u, kFactorMatrix},
                        {2u, kPointJacobians},
                        {3u, kCandidateV},
                        {4u, kCandidateBodies},
                        {5u, kContacts},
                        {6u, kContactMetadata},
                        {7u, kEvaluatedRows},
                        {8u, kEvaluatedCones},
                        {9u, kResponseColumns},
                        {10u, kCandidateManifoldPoints},
                        {11u, kContactStatuses},
                    },
                    &solverPass,
                    12u,
                    environmentCount,
                    true,
                    sizeof(MRIndirectDispatchArgumentsGPU)
                )) ||
        (
            !useQuality &&
            !useWave32 &&
            !encodeRodContactSolve(
                context,
                commandBuffer,
                solverPass,
                candidateRodNodes,
                candidateRodEdges,
                rodWitnessCount,
                environmentCount
            )
        ) ||
        (
            !useQuality &&
            !encodeContactThreadKernel(
                context,
                commandBuffer,
                context.generalizedConstraintSolvePipeline,
                @"MetalWorld canonical generalized ConstraintIR solve",
                {
                    {0u, kContactDispatch},
                    {1u, kFactorMatrix},
                    {2u, kCandidateV},
                    {3u, kContacts},
                    {4u, kIRBlocks},
                    {5u, kIREndpoints},
                    {6u, kEvaluatedRows},
                    {7u, kContactStatuses},
                    {9u, kCandidateBodies},
                    {10u, candidateRodNodes},
                    {11u, kRodInverseMasses},
                },
                &solverPass,
                8u,
                environmentCount
            )
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.jointLimitPipeline,
            @"MetalWorld constrained joint limits",
            {
                {0u, kContactDispatch},
                {1u, kArticulations},
                {2u, kDofs},
                {3u, sourceQ},
                {4u, kCandidateQ},
                {5u, kCandidateV},
                {6u, kContactStatuses},
            },
            nullptr,
            0u,
            environmentCount,
            false,
            0u,
            &stateNotIntegrated,
            sizeof(stateNotIntegrated),
            7u
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.contactIntegratePipeline,
            @"MetalWorld constrained integration",
            {
                {0u, kContactDispatch},
                {1u, kArticulations},
                {2u, kJoints},
                {3u, kBodies},
                {4u, kSceneBodyIndices},
                {5u, sourceQ},
                {6u, kCandidateV},
                {7u, kCandidateQ},
                {8u, kCandidateBodies},
                {9u, kContactStatuses},
            },
            nullptr,
            0u,
            environmentCount
        ) ||
        (useHybridCCD &&
         !encodeContactThreadKernel(
             context,
             commandBuffer,
             context.ccdEventFinalizePipeline,
             @"MetalWorld CCD event-time closeout",
             {
                 {0u, kContactDispatch},
                 {1u, kCCDEventStatesB},
                 {2u, kContactStatuses},
             },
             nullptr,
             0u,
             environmentCount,
             false,
             0u,
             &eventPass,
             sizeof(eventPass),
             3u
         )) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.contactLatchPipeline,
            @"MetalWorld contact failure latch",
            {
                {0u, kWorldDispatch},
                {2u, kContactStatuses},
                {3u, kEnvironmentStatuses},
            },
            &pass,
            1u,
            environmentCount
        ) ||
        !encodeCommit(
            context,
            commandBuffer,
            pass,
            destinationQ,
            destinationV,
            environmentCount
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.contactCommitPipeline,
            @"MetalWorld contact transactional commit",
            {
                {0u, kWorldDispatch},
                {1u, kContactDispatch},
                {3u, kEnvironmentStatuses},
                {4u, kSceneBodyIndices},
                {5u, kCandidateBodies},
                {6u, kCheckpointSceneBodies},
                {7u, destinationScene},
                {8u, kCandidateManifoldHeaders},
                {9u, kCandidateManifoldPoints},
                {10u, kCandidateManifoldCounts},
                {11u, kCheckpointManifoldHeaders},
                {12u, kCheckpointManifoldPoints},
                {13u, kCheckpointManifoldCounts},
                {14u, destinationManifoldHeaders},
                {15u, destinationManifoldPoints},
                {16u, destinationManifoldCounts},
            },
            &pass,
            2u,
            environmentCount
        )) {
        return false;
    }
    if (finalPhysicsSubstep &&
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.convexCachePublishPipeline,
            @"MetalWorld transactional convex-cache publication",
            {
                {0u, kContactDispatch},
                {1u, kEligiblePairs},
                {2u, kPairOverlapFlags},
                {3u, kContactStatuses},
                {4u, kCandidateConvexCaches},
                {5u, kConvexCaches},
            },
            nullptr,
            0u,
            pairFlagThreadCount
        )) {
        return false;
    }
    return true;
}

bool encodeContactCapture(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t sceneState,
    const std::size_t environmentCount
) {
    return encodeContactThreadKernel(
        context,
        commandBuffer,
        context.contactCapturePipeline,
        @"MetalWorld contact observation/status capture",
        {
            {0u, kWorldDispatch},
            {1u, kContactDispatch},
            {3u, sceneState},
            {4u, kContactStatuses},
            {5u, kObservations},
            {6u, kPublicContactStatuses},
        },
        &pass,
        2u,
        environmentCount
    );
}

template <typename T>
void copyOutput(
    std::vector<T>& destination,
    id<MTLBuffer> source
) {
    if (!destination.empty()) {
        std::memcpy(
            destination.data(),
            source.contents,
            destination.size() * sizeof(T)
        );
    }
}

bool finiteRange(
    const std::vector<float>& values,
    const std::size_t begin,
    const std::size_t count
) {
    return std::all_of(
        values.begin() + static_cast<std::ptrdiff_t>(begin),
        values.begin() +
            static_cast<std::ptrdiff_t>(begin + count),
        [](const float value) {
            return std::isfinite(value);
        }
    );
}

bool zeroRange(
    const std::vector<float>& values,
    const std::size_t begin,
    const std::size_t count
) {
    return std::all_of(
        values.begin() + static_cast<std::ptrdiff_t>(begin),
        values.begin() +
            static_cast<std::ptrdiff_t>(begin + count),
        [](const float value) {
            return value == 0.0f;
        }
    );
}

MetalWorldDiagnostics validateAndPublish(
    MetalWorldResult&& staged,
    MetalWorldDiagnostics diagnostics,
    const bool publishFinalState,
    const bool publishStateTrajectory,
    const std::uint64_t expectedPolicyRevision,
    MetalWorldResult& result
) {
    const MRMetalWorldDispatchGPU& dispatch =
        diagnostics.layout.dispatch;
    const bool contactMode =
        (dispatch.flags & MR_METAL_WORLD_CONTACTS) != 0u;
    const MRMetalWorldContactDispatchGPU& contactDispatch =
        diagnostics.layout.contactDispatch;
    const std::size_t observationWidth =
        dispatch.observationEnvironmentStride;
    staged.environmentStatuses.resize(
        dispatch.environmentCount
    );
    std::vector<std::uint64_t> retainedPointTotals(
        dispatch.environmentCount,
        0u
    );
    std::vector<std::uint64_t> observedPointTotals(
        dispatch.environmentCount,
        0u
    );
    for (std::uint32_t environment = 0u;
         environment < dispatch.environmentCount;
         ++environment) {
        staged.environmentStatuses[environment].environment =
            environment;
    }
    for (std::size_t controlStep = 0u;
         controlStep < dispatch.controlStepCount;
         ++controlStep) {
        for (std::size_t environment = 0u;
             environment < dispatch.environmentCount;
             ++environment) {
            const std::size_t statusIndex =
                controlStep * dispatch.environmentCount +
                environment;
            const MRMetalWorldStatusGPU& status =
                staged.statuses[statusIndex];
            const MRMetalWorldContactStatusGPU* contactStatus =
                contactMode
                ? &staged.contactStatuses[statusIndex]
                : nullptr;
            if (status.environment != environment ||
                status.controlStep != controlStep ||
                status.code >= MR_STEP_STATUS_COUNT ||
                status.abaCode >
                    MR_ABA_UNSUPPORTED_TOPOLOGY ||
                status.flags != dispatch.flags ||
                status.successfulSubsteps >
                    dispatch.physicsSubsteps ||
                !finite(status.diagnostics)) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::internalFailure,
                    "GPU returned a malformed MetalWorld status record"
                );
            }
            if (contactStatus != nullptr &&
                (
                    contactStatus->environment != environment ||
                    contactStatus->controlStep != controlStep ||
                    (contactStatus->code == MR_STEP_SUCCESS &&
                     contactStatus->physicsSubstep >=
                         dispatch.physicsSubsteps) ||
                    contactStatus->code >=
                        MR_STEP_STATUS_COUNT ||
                    !finite(contactStatus->residuals) ||
                    !finite(contactStatus->diagnostics) ||
                    !finite(contactStatus->eventTimes) ||
                    !finite(contactStatus->qualityCertificates) ||
                    !finite(contactStatus->qualityDiagnostics) ||
                    contactStatus->activePairs >
                        contactStatus->requiredPairs ||
                    contactStatus->activeContacts >
                        contactStatus->requiredConstraints ||
                    contactStatus->islandCount >
                        contactStatus->requiredIslands
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::internalFailure,
                    "GPU returned a malformed contact-world status record"
                    " at environment " + std::to_string(environment) +
                    ", control step " + std::to_string(controlStep) +
                    ": environment=" +
                    std::to_string(contactStatus->environment) +
                    ", controlStep=" +
                    std::to_string(contactStatus->controlStep) +
                    ", physicsSubstep=" +
                    std::to_string(contactStatus->physicsSubstep) +
                    ", code=" + std::to_string(contactStatus->code) +
                    ", activePairs=" +
                    std::to_string(contactStatus->activePairs) + "/" +
                    std::to_string(contactStatus->requiredPairs) +
                    ", activeContacts=" +
                    std::to_string(contactStatus->activeContacts) + "/" +
                    std::to_string(contactStatus->requiredConstraints) +
                    ", islands=" +
                    std::to_string(contactStatus->islandCount) + "/" +
                    std::to_string(contactStatus->requiredIslands) +
                    ", eventTimes=[" +
                    std::to_string(contactStatus->eventTimes.x) + "," +
                    std::to_string(contactStatus->eventTimes.y) + "," +
                    std::to_string(contactStatus->eventTimes.z) + "," +
                    std::to_string(contactStatus->eventTimes.w) + "]"
                );
            }
            MetalWorldStatus& summary =
                staged.environmentStatuses[environment];
            if (status.code == MR_STEP_SUCCESS) {
                ++summary.successfulControlSteps;
            } else {
                ++summary.failedControlSteps;
                if (summary.code == MR_STEP_SUCCESS) {
                    summary.code = status.code;
                    summary.firstFailingControlStep =
                        static_cast<std::uint32_t>(controlStep);
                    if (contactStatus != nullptr) {
                        summary.firstFailingPair =
                            contactStatus->firstFailingPair;
                        summary.firstFailingConstraint =
                            contactStatus
                                ->firstFailingConstraint;
                        const std::uint64_t reportedStableKey =
                            (
                                static_cast<std::uint64_t>(
                                    contactStatus
                                        ->firstFailingStableKeyHigh
                                ) << 32u
                            ) |
                            contactStatus
                                ->firstFailingStableKeyLow;
                        if (reportedStableKey != 0u &&
                            reportedStableKey !=
                                std::numeric_limits<
                                    std::uint64_t
                                >::max()) {
                            summary.firstFailingStableKey =
                                reportedStableKey;
                        } else if (summary.firstFailingPair !=
                            MR_INVALID_INDEX) {
                            summary.firstFailingStableKey =
                                summary.firstFailingPair;
                        } else if (
                            summary.firstFailingConstraint !=
                            MR_INVALID_INDEX
                        ) {
                            summary.firstFailingStableKey =
                                (std::uint64_t{1} << 63u) |
                                summary.firstFailingConstraint;
                        }
                    }
                }
            }
            if (contactStatus != nullptr) {
                auto updateMaximum = [](
                    std::uint32_t& target,
                    const std::uint32_t value
                ) {
                    target = std::max(target, value);
                };
                updateMaximum(
                    summary.required.candidatePairs,
                    contactStatus->requiredPairs
                );
                updateMaximum(
                    summary.required.rawContacts,
                    contactStatus->requiredRawContacts
                );
                updateMaximum(
                    summary.required.manifolds,
                    contactStatus->requiredManifolds
                );
                updateMaximum(
                    summary.required.constraintBlocks,
                    contactStatus->requiredConstraints
                );
                updateMaximum(
                    summary.required.constraintRows,
                    contactStatus->requiredRows
                );
                updateMaximum(
                    summary.required.islands,
                    contactStatus->requiredIslands
                );
                updateMaximum(
                    summary.required.hardConvexPairs,
                    contactStatus->requiredHardConvexPairs
                );
                updateMaximum(
                    summary.required.meshTriangleCandidates,
                    contactStatus->requiredMeshCandidates
                );
                updateMaximum(
                    summary.required.solverTiles,
                    contactStatus->requiredSolverTiles
                );
                updateMaximum(
                    summary.required.spillRows,
                    contactStatus->requiredSpillRows
                );
                updateMaximum(
                    summary.required.ccdCandidates,
                    contactStatus->requiredCCDCandidates
                );
                updateMaximum(
                    summary.required.ccdEvents,
                    contactStatus->requiredCCDEvents
                );
                updateMaximum(
                    summary.highWater.candidatePairs,
                    std::min(
                        contactStatus->activePairs,
                        contactDispatch.pairCapacity
                    )
                );
                updateMaximum(
                    summary.highWater.rawContacts,
                    std::min(
                        contactStatus->requiredRawContacts,
                        contactDispatch.rawContactCapacity
                    )
                );
                updateMaximum(
                    summary.highWater.manifolds,
                    std::min(
                        contactStatus->requiredManifolds,
                        contactDispatch.manifoldCapacity
                    )
                );
                updateMaximum(
                    summary.highWater.constraintBlocks,
                    std::min(
                        contactStatus->activeContacts,
                        contactDispatch.constraintCapacity
                    )
                );
                updateMaximum(
                    summary.highWater.constraintRows,
                    std::min(
                        contactStatus->requiredRows,
                        contactDispatch.rowCapacity
                    )
                );
                updateMaximum(
                    summary.highWater.islands,
                    std::min(
                        contactStatus->islandCount,
                        contactDispatch.islandCapacity
                    )
                );
                updateMaximum(
                    summary.highWater.hardConvexPairs,
                    std::min(
                        contactStatus->hardConvexPairs,
                        contactDispatch.hardConvexCapacity
                    )
                );
                updateMaximum(
                    summary.highWater.meshTriangleCandidates,
                    std::min(
                        contactStatus->meshCandidates,
                        contactDispatch.meshCandidateCapacity
                    )
                );
                updateMaximum(
                    summary.highWater.solverTiles,
                    std::min(
                        contactStatus->solverTiles,
                        contactDispatch.solverTileCapacity
                    )
                );
                updateMaximum(
                    summary.highWater.spillRows,
                    contactStatus->spillRows
                );
                updateMaximum(
                    summary.highWater.ccdCandidates,
                    std::min(
                        contactStatus->ccdCandidates,
                        contactDispatch.ccdCandidateCapacity
                    )
                );
                updateMaximum(
                    summary.highWater.ccdEvents,
                    std::min(
                        contactStatus->ccdEvents,
                        contactDispatch.ccdEventCapacity
                    )
                );
                summary.hardConvexFallbacks = std::max(
                    summary.hardConvexFallbacks,
                    contactStatus->hardFallbacks
                );
                summary.unresolvedCCDCount = std::max(
                    summary.unresolvedCCDCount,
                    contactStatus->unresolvedCCDCount
                );
                updateMaximum(
                    summary.maximumCCDAdvanceCount,
                    contactStatus->ccdAdvanceCount
                );
                updateMaximum(
                    summary.maximumClusteredCCDImpacts,
                    contactStatus->clusteredCCDImpacts
                );
                updateMaximum(
                    summary.maximumZeroTimeCCDReplays,
                    contactStatus->zeroTimeCCDReplays
                );
                updateMaximum(
                    summary.maximumWorkerPackets,
                    contactStatus->workerPackets
                );
                summary.maximumUnconsumedCCDTime = std::max(
                    summary.maximumUnconsumedCCDTime,
                    std::abs(contactStatus->eventTimes.y)
                );
                updateMaximum(
                    summary.maximumSolverIterations,
                    contactStatus->solverIterations
                );
                updateMaximum(
                    summary.maximumQualityNewtonIterations,
                    contactStatus->qualityNewtonIterations
                );
                updateMaximum(
                    summary.maximumQualityPCGIterations,
                    contactStatus->qualityPCGIterations
                );
                updateMaximum(
                    summary.maximumQualityLineSearchBacktracks,
                    contactStatus->qualityLineSearchBacktracks
                );
                const std::array residuals{
                    contactStatus->residuals.x,
                    contactStatus->residuals.y,
                    contactStatus->residuals.z,
                    contactStatus->residuals.w,
                };
                for (std::size_t index = 0u;
                     index < residuals.size();
                     ++index) {
                    summary.maximumResiduals[index] = std::max(
                        summary.maximumResiduals[index],
                        std::abs(residuals[index])
                    );
                }
                const std::array qualityCertificates{
                    contactStatus->qualityCertificates.x,
                    contactStatus->qualityCertificates.y,
                    contactStatus->qualityCertificates.z,
                    contactStatus->qualityCertificates.w,
                };
                for (std::size_t index = 0u;
                     index < qualityCertificates.size();
                     ++index) {
                    summary.maximumQualityCertificates[index] =
                        std::max(
                            summary
                                .maximumQualityCertificates[index],
                            std::abs(qualityCertificates[index])
                        );
                }
                retainedPointTotals[environment] +=
                    contactStatus->retainedPoints;
                observedPointTotals[environment] +=
                    static_cast<std::uint64_t>(
                        contactStatus->retainedPoints
                    ) + contactStatus->newPoints;
            }
            const std::size_t observationBase =
                controlStep * dispatch.observationStepStride +
                environment *
                    dispatch.observationEnvironmentStride;
            const std::size_t accelerationBase =
                controlStep * dispatch.accelerationStepStride +
                environment * dispatch.nv;
            if (publishStateTrajectory &&
                !finiteRange(
                    staged.observations,
                    observationBase,
                    observationWidth
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::internalFailure,
                    "GPU published a non-finite MetalWorld observation"
                );
            }

            if (status.code == MR_STEP_SUCCESS) {
                if (status.successfulSubsteps !=
                        dispatch.physicsSubsteps ||
                    status.abaCode != MR_ABA_SUCCESS ||
                    (contactStatus != nullptr &&
                     (
                         contactStatus->code != MR_STEP_SUCCESS ||
                         contactStatus->requiredPairs >
                             contactDispatch.pairCapacity ||
                         contactStatus->requiredRawContacts >
                             contactDispatch.rawContactCapacity ||
                         contactStatus->requiredManifolds >
                             contactDispatch.manifoldCapacity ||
                         contactStatus->requiredConstraints >
                             contactDispatch.constraintCapacity ||
                         contactStatus->requiredRows >
                             contactDispatch.rowCapacity ||
                         contactStatus->requiredIslands >
                             contactDispatch.islandCapacity
                     )) ||
                    status.failingSubstep != MR_INVALID_INDEX ||
                    status.failingIndex != MR_INVALID_INDEX ||
                    (publishStateTrajectory &&
                     !finiteRange(
                        staged.accelerations,
                        accelerationBase,
                        dispatch.nv
                    ))) {
                    return reject(
                        std::move(diagnostics),
                        MetalWorldHostStatus::internalFailure,
                        "GPU success status has invalid substep "
                        "accounting or payload"
                    );
                }
                ++diagnostics.successfulStepCount;
            } else {
                if (status.successfulSubsteps >=
                        dispatch.physicsSubsteps ||
                    (status.abaCode == MR_ABA_SUCCESS &&
                     (contactStatus == nullptr ||
                      contactStatus->code == MR_STEP_SUCCESS)) ||
                    status.failingSubstep >=
                        dispatch.physicsSubsteps ||
                    (publishStateTrajectory &&
                     !zeroRange(
                        staged.accelerations,
                        accelerationBase,
                        dispatch.nv
                    ))) {
                    return reject(
                        std::move(diagnostics),
                        MetalWorldHostStatus::internalFailure,
                        "failed GPU step violated rollback or "
                        "failure-accounting semantics: status=" +
                            std::to_string(status.code) +
                            " aba=" +
                            std::to_string(status.abaCode) +
                            " successful_substeps=" +
                            std::to_string(
                                status.successfulSubsteps
                            ) +
                            " failing_substep=" +
                            std::to_string(
                                status.failingSubstep
                            ) +
                            " failing_index=" +
                            std::to_string(status.failingIndex)
                    );
                }
                if (diagnostics.failedStepCount == 0u) {
                    diagnostics.firstFailingEnvironment =
                        static_cast<std::uint32_t>(environment);
                    diagnostics.firstFailingControlStep =
                        static_cast<std::uint32_t>(controlStep);
                    diagnostics.firstGPUStatusCode = status.code;
                }
                ++diagnostics.failedStepCount;
            }
        }
    }
    for (std::size_t environment = 0u;
         environment < dispatch.environmentCount;
         ++environment) {
        staged.environmentStatuses[environment]
            .manifoldRetention =
            observedPointTotals[environment] == 0u
            ? 1.0f
            : static_cast<float>(
                  retainedPointTotals[environment]
              ) /
                static_cast<float>(
                    observedPointTotals[environment]
                );
    }

    if (publishFinalState &&
        (!finiteFloats(staged.finalQ) ||
         !finiteFloats(staged.finalV))) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::internalFailure,
            "GPU published a non-finite final MetalWorld state"
        );
    }
    if (publishFinalState &&
        contactMode &&
        !std::all_of(
            staged.finalSceneBodies.begin(),
            staged.finalSceneBodies.end(),
            [](const MRBodyStateGPU& state) {
                return finite(state.position) &&
                    finite(state.orientation) &&
                    finite(
                        state.linearVelocityAndInverseMass
                    ) &&
                    finite(state.angularVelocity) &&
                    finite(state.inverseInertiaWorldRow0) &&
                    finite(state.inverseInertiaWorldRow1) &&
                    finite(state.inverseInertiaWorldRow2);
            }
        )) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::internalFailure,
            "GPU published a non-finite final scene-body state"
        );
    }
    if (publishFinalState && publishStateTrajectory) {
        const std::size_t lastStep =
            static_cast<std::size_t>(
                dispatch.controlStepCount
            ) - 1u;
        for (std::size_t environment = 0u;
             environment < dispatch.environmentCount;
             ++environment) {
            const std::size_t observationBase =
                lastStep * dispatch.observationStepStride +
                environment *
                    dispatch.observationEnvironmentStride;
            const std::size_t qBase =
                environment * dispatch.qStride;
            const std::size_t vBase =
                environment * dispatch.vStride;
            if (std::memcmp(
                    staged.finalQ.data() + qBase,
                    staged.observations.data() +
                        observationBase,
                    dispatch.nq * sizeof(float)
                ) != 0 ||
                std::memcmp(
                    staged.finalV.data() + vBase,
                    staged.observations.data() +
                        observationBase + dispatch.nq,
                    dispatch.nv * sizeof(float)
                ) != 0) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::internalFailure,
                    "final state does not match the last captured "
                    "observation"
                );
            }
        }
    }
    if (diagnostics.layout.actionElements != 0u) {
        if (!finiteFloats(staged.actorObservations) ||
            !finiteFloats(staged.criticObservations) ||
            !finiteFloats(staged.motionFeatures) ||
            !finiteFloats(staged.teacherActions) ||
            !finiteFloats(staged.policyLatents) ||
            !finiteFloats(
                staged.policyLogProbabilities
            ) ||
            !finiteFloats(staged.policyValues) ||
            !std::all_of(
                staged.transitions.begin(),
                staged.transitions.end(),
                [expectedPolicyRevision](
                    const MRTaskTransitionGPU& transition
                ) {
                    return
                        finite(transition.rewardAndState) &&
                        finite(transition.rewardBreakdown0) &&
                        finite(transition.rewardBreakdown1) &&
                        finite(transition.outcomeChannels0) &&
                        finite(transition.outcomeChannels1) &&
                        transition.policyRevision ==
                            expectedPolicyRevision &&
                        transition.termination.x <= 1u &&
                        transition.termination.y <= 1u &&
                        transition.termination.z <= 1u &&
                        transition.termination.w <=
                            MR_TASK_TERMINATION_PROJECTILE_CONTACT &&
                        (
                            transition.termination.x != 0u ||
                            transition.termination.w ==
                                MR_TASK_TERMINATION_CONTINUING
                        );
                }
            )) {
            return reject(
                std::move(diagnostics),
                MetalWorldHostStatus::internalFailure,
                "GPU published a malformed native task transition"
            );
        }
    }

    result = std::move(staged);
    diagnostics.published = true;
    if (diagnostics.failedStepCount != 0u) {
        std::string contactDetail;
        if (contactMode &&
            diagnostics.firstFailingEnvironment <
                dispatch.environmentCount &&
            diagnostics.firstFailingControlStep <
                dispatch.controlStepCount) {
            const std::size_t index =
                static_cast<std::size_t>(
                    diagnostics.firstFailingControlStep
                ) * dispatch.environmentCount +
                diagnostics.firstFailingEnvironment;
            const MRMetalWorldContactStatusGPU& contact =
                result.contactStatuses[index];
            const MRMetalWorldStatusGPU& worldStatus =
                result.statuses[index];
            contactDetail =
                " aba_status=" +
                std::to_string(worldStatus.abaCode) +
                " failing_substep=" +
                std::to_string(worldStatus.failingSubstep) +
                " failing_index=" +
                std::to_string(worldStatus.failingIndex) +
                " aba_diagnostic_0=" +
                std::to_string(worldStatus.diagnostics.x) +
                " aba_diagnostic_1=" +
                std::to_string(worldStatus.diagnostics.y) +
                " aba_diagnostic_2=" +
                std::to_string(worldStatus.diagnostics.z) +
                " aba_diagnostic_3=" +
                std::to_string(worldStatus.diagnostics.w) +
                " contact_status=" +
                std::to_string(contact.code) +
                " required_pairs=" +
                std::to_string(contact.requiredPairs) +
                " pair_capacity=" +
                std::to_string(contactDispatch.pairCapacity) +
                " required_raw_contacts=" +
                std::to_string(contact.requiredRawContacts) +
                " raw_contact_capacity=" +
                std::to_string(
                    contactDispatch.rawContactCapacity
                ) +
                " required_manifolds=" +
                std::to_string(contact.requiredManifolds) +
                " manifold_capacity=" +
                std::to_string(
                    contactDispatch.manifoldCapacity
                ) +
                " required_constraints=" +
                std::to_string(contact.requiredConstraints) +
                " constraint_capacity=" +
                std::to_string(
                    contactDispatch.constraintCapacity
                ) +
                " failing_constraint=" +
                std::to_string(
                    contact.firstFailingConstraint
                ) +
                " diagnostic_0=" +
                std::to_string(contact.diagnostics.x) +
                " diagnostic_1=" +
                std::to_string(contact.diagnostics.y) +
                " diagnostic_2=" +
                std::to_string(contact.diagnostics.z) +
                " diagnostic_3=" +
                std::to_string(contact.diagnostics.w);
            if (index < result.qualityStatuses.size()) {
                const MRUnifiedQualityStatusGPU& quality =
                    result.qualityStatuses[index];
                contactDetail +=
                    " quality_status=" +
                    std::to_string(quality.code) +
                    " quality_path=" +
                    std::to_string(quality.solvePath) +
                    " quality_block=" +
                    std::to_string(quality.failingBlock) +
                    " quality_newton=" +
                    std::to_string(quality.newtonIterations) +
                    " quality_pcg=" +
                    std::to_string(quality.pcgIterations) +
                    " quality_backtracks=" +
                    std::to_string(
                        quality.lineSearchBacktracks
                    ) +
                    " quality_retries=" +
                    std::to_string(quality.regularizationRetries) +
                    " quality_cert0=" +
                    std::to_string(quality.certificates0.x) +
                    "," +
                    std::to_string(quality.certificates0.y) +
                    "," +
                    std::to_string(quality.certificates0.z) +
                    "," +
                    std::to_string(quality.certificates0.w) +
                    " quality_cert1=" +
                    std::to_string(quality.certificates1.x) +
                    "," +
                    std::to_string(quality.certificates1.y) +
                    "," +
                    std::to_string(quality.certificates1.z) +
                    "," +
                    std::to_string(quality.certificates1.w) +
                    " quality_numerics=" +
                    std::to_string(quality.numerics.x) +
                    "," +
                    std::to_string(quality.numerics.y) +
                    "," +
                    std::to_string(quality.numerics.z) +
                    "," +
                    std::to_string(quality.numerics.w);
            }
        }
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::gpuEnvironmentFailure,
            "one or more MetalWorld control steps rolled back "
            "after a GPU environment failure: status=" +
                std::to_string(diagnostics.firstGPUStatusCode) +
                " environment=" +
                std::to_string(
                    diagnostics.firstFailingEnvironment
                ) +
                " control_step=" +
                std::to_string(
                    diagnostics.firstFailingControlStep
                ) +
                contactDetail
        );
    }
    diagnostics.status = MetalWorldHostStatus::success;
    diagnostics.message.clear();
    return diagnostics;
}

} // namespace

bool CompiledWorld::valid() const noexcept {
    return fingerprint_ != 0u &&
        modelFingerprint_ != 0u &&
        articulationIndex_ < model_.articulations.size() &&
        capacityClass_ != MetalWorldCapacityClass::uncompiled;
}

const EngineModel& CompiledWorld::model() const noexcept {
    return model_;
}

std::uint32_t CompiledWorld::articulationIndex() const noexcept {
    return articulationIndex_;
}

std::uint32_t CompiledWorld::articulationCount() const noexcept {
    return valid()
        ? static_cast<std::uint32_t>(model_.articulations.size())
        : 0u;
}

std::uint32_t CompiledWorld::nq() const noexcept {
    return valid()
        ? model_.world.nq
        : 0u;
}

std::uint32_t CompiledWorld::nv() const noexcept {
    return valid()
        ? model_.world.nv
        : 0u;
}

std::uint32_t CompiledWorld::bodyCount() const noexcept {
    return valid()
        ? static_cast<std::uint32_t>(model_.bodies.size())
        : 0u;
}

std::uint32_t CompiledWorld::sceneBodyCount() const noexcept {
    return valid()
        ? static_cast<std::uint32_t>(sceneBodyIndices_.size())
        : 0u;
}

std::uint32_t CompiledWorld::colliderCount() const noexcept {
    return valid()
        ? static_cast<std::uint32_t>(model_.shapes.size())
        : 0u;
}

std::uint32_t CompiledWorld::eligiblePairCount() const noexcept {
    return valid()
        ? static_cast<std::uint32_t>(eligiblePairs_.size())
        : 0u;
}

std::uint32_t CompiledWorld::rodCount() const noexcept {
    return valid() && !rodNodeOffsets_.empty()
        ? static_cast<std::uint32_t>(
              rodNodeOffsets_.size() - 1u
          )
        : 0u;
}

std::uint32_t CompiledWorld::rodNodeCount() const noexcept {
    return valid()
        ? static_cast<std::uint32_t>(
              defaultRodNodes_.size()
          )
        : 0u;
}

std::uint32_t CompiledWorld::rodEdgeCount() const noexcept {
    return valid()
        ? static_cast<std::uint32_t>(
              defaultRodEdges_.size()
          )
        : 0u;
}

std::span<const std::uint32_t>
CompiledWorld::sceneBodyIndices() const noexcept {
    return sceneBodyIndices_;
}

std::span<const MRCompiledCollisionPairGPU>
CompiledWorld::eligiblePairs() const noexcept {
    return eligiblePairs_;
}

std::span<const MRGeometryHeaderGPU>
CompiledWorld::geometryHeaders() const noexcept {
    return model_.geometryHeaders;
}

std::span<const mr_float4>
CompiledWorld::geometryVertices() const noexcept {
    return model_.geometryVertices;
}

std::span<const std::uint32_t>
CompiledWorld::geometryIndices() const noexcept {
    return model_.geometryIndices;
}

std::span<const MRConvexFaceGPU>
CompiledWorld::convexFaces() const noexcept {
    return model_.convexFaces;
}

std::span<const MRConvexHalfEdgeGPU>
CompiledWorld::convexHalfEdges() const noexcept {
    return model_.convexHalfEdges;
}

std::span<const MRMeshBVHNodeGPU>
CompiledWorld::meshBvhNodes() const noexcept {
    return model_.meshBvhNodes;
}

std::span<const MRMeshTriangleGPU>
CompiledWorld::meshTriangles() const noexcept {
    return model_.meshTriangles;
}

std::span<const MRRodColliderGPU>
CompiledWorld::rodColliders() const noexcept {
    return rodColliders_;
}

std::span<const MRShapeGPU>
CompiledWorld::rodShapeSources() const noexcept {
    return rodShapeSources_;
}

std::span<const MRRodToolPairGPU>
CompiledWorld::rodToolPairs() const noexcept {
    return rodToolPairs_;
}

std::span<const std::uint32_t>
CompiledWorld::rodNodeOffsets() const noexcept {
    return rodNodeOffsets_;
}

std::span<const std::uint32_t>
CompiledWorld::rodEdgeOffsets() const noexcept {
    return rodEdgeOffsets_;
}

std::span<const MRRodNodeStateGPU>
CompiledWorld::defaultRodNodes() const noexcept {
    return defaultRodNodes_;
}

std::span<const MRRodEdgeStateGPU>
CompiledWorld::defaultRodEdges() const noexcept {
    return defaultRodEdges_;
}

std::span<const HeterogeneousRodProgram>
CompiledWorld::rodPrograms() const noexcept {
    return rodPrograms_;
}

std::span<const std::uint32_t>
CompiledWorld::articulationQOffsets() const noexcept {
    return articulationQOffsets_;
}

std::span<const std::uint32_t>
CompiledWorld::articulationVOffsets() const noexcept {
    return articulationVOffsets_;
}

std::span<const MRWorldDynamicNodeGPU>
CompiledWorld::dynamicNodes() const noexcept {
    return dynamicNodes_;
}

std::span<const std::uint32_t>
CompiledWorld::bodyDynamicNodes() const noexcept {
    return bodyDynamicNodes_;
}

std::span<const std::uint32_t>
CompiledWorld::sceneBodyDynamicNodes() const noexcept {
    return sceneBodyDynamicNodes_;
}

std::span<const std::uint32_t>
CompiledWorld::rodDynamicNodes() const noexcept {
    return rodDynamicNodes_;
}

const MetalWorldCapacityProfile& CompiledWorld::capacities()
    const noexcept {
    return capacities_;
}

const MetalWorldCapacityProfile&
CompiledWorld::minimumCapacities() const noexcept {
    return minimumCapacities_;
}

MetalWorldCapacityClass CompiledWorld::capacityClass()
    const noexcept {
    return capacityClass_;
}

std::uint64_t CompiledWorld::fingerprint() const noexcept {
    return fingerprint_;
}

std::uint64_t CompiledWorld::modelFingerprint() const noexcept {
    return valid() ? modelFingerprint_ : 0u;
}

std::uint64_t engineModelFingerprint(
    const EngineModel& model
) noexcept {
    return model.valid(nullptr) ? fingerprint(model) : 0u;
}

MetalWorldCompileDiagnostics compileMetalWorld(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    CompiledWorld& compiled,
    const MetalWorldCapacityProfile& requestedCapacities
) {
    MetalWorldCompileDiagnostics diagnostics{};
    try {
        std::string modelReason;
        if (!model.valid(&modelReason)) {
            return rejectCompile(
                std::move(diagnostics),
                MetalWorldHostStatus::invalidModel,
                "invalid EngineModel: " + modelReason
            );
        }
        if (model.articulations.empty() ||
            articulationIndex >= model.articulations.size()) {
            return rejectCompile(
                std::move(diagnostics),
                MetalWorldHostStatus::unsupportedTopology,
                "MetalWorld requires at least one valid primary "
                "articulation"
            );
        }
        for (std::uint32_t owner = 0u;
             owner < model.articulations.size();
             ++owner) {
            std::string topologyReason;
            if (supportedTopology(
                    model,
                    model.articulations[owner],
                    topologyReason
                )) {
                continue;
            }
            const MRArticulationGPU& candidate =
                model.articulations[owner];
            const bool capacity =
                candidate.bodyCount >
                    MR_ARTICULATED_ABA_MAX_BODIES ||
                candidate.nv >
                    MR_ARTICULATED_ABA_MAX_DOFS ||
                candidate.nq >
                    MR_ARTICULATED_ABA_MAX_Q;
            return rejectCompile(
                std::move(diagnostics),
                capacity
                    ? MetalWorldHostStatus::capacityOverflow
                    : MetalWorldHostStatus::unsupportedTopology,
                "articulation " + std::to_string(owner) +
                    ": " + topologyReason
            );
        }

        CompiledWorld staged;
        staged.model_ = model;
        staged.articulationIndex_ = articulationIndex;
        staged.capacityClass_ =
            std::all_of(
                staged.model_.articulations.begin(),
                staged.model_.articulations.end(),
                [](const MRArticulationGPU& candidate) {
                    return
                        candidate.bodyCount <=
                            kSmallABAMaxBodies &&
                        candidate.nv <= kSmallABAMaxDofs &&
                        candidate.nq <= kSmallABAMaxQ;
                }
            )
            ? MetalWorldCapacityClass::compactABA12
            : MetalWorldCapacityClass::fullABA32;

        for (std::uint32_t collider = 0u;
             collider < staged.model_.shapes.size();
             ++collider) {
            const MRShapeGPU& shape =
                staged.model_.shapes[collider];
            if ((shape.flags &
                 MR_SHAPE_FLAG_SIMULATION_DISABLED) != 0u) {
                continue;
            }
            if ((shape.shapeType ==
                     MR_SHAPE_TRIANGLE_MESH ||
                 shape.shapeType ==
                     MR_SHAPE_HEIGHTFIELD) &&
                staged.model_.bodies[shape.bodyIndex]
                        .motionType ==
                    MR_MOTION_DYNAMIC) {
                return rejectCompile(
                    std::move(diagnostics),
                    MetalWorldHostStatus::unsupportedTopology,
                    "dynamic concave surface collider " +
                        std::to_string(collider) +
                        " must be replaced by convex decomposition"
                );
            }
        }

        for (std::uint32_t body = 0u;
             body < staged.model_.bodies.size();
             ++body) {
            if (staged.model_.bodies[body].articulationIndex ==
                MR_INVALID_INDEX) {
                staged.sceneBodyIndices_.push_back(body);
            }
        }

        // Compile one immutable typed dynamic-node table. Runtime islands
        // reference this table instead of inferring ownership from body
        // indices, which is the key representation needed by mixed
        // articulation/free-body/rod operators.
        staged.bodyDynamicNodes_.assign(
            staged.model_.bodies.size(),
            MR_INVALID_INDEX
        );
        staged.sceneBodyDynamicNodes_.assign(
            staged.sceneBodyIndices_.size(),
            MR_INVALID_INDEX
        );
        staged.articulationQOffsets_.reserve(
            staged.model_.articulations.size() + 1u
        );
        staged.articulationVOffsets_.reserve(
            staged.model_.articulations.size() + 1u
        );
        for (std::uint32_t owner = 0u;
             owner < staged.model_.articulations.size();
             ++owner) {
            const MRArticulationGPU& owned =
                staged.model_.articulations[owner];
            staged.articulationQOffsets_.push_back(owned.qOffset);
            staged.articulationVOffsets_.push_back(owned.vOffset);
            const std::uint32_t nodeIndex =
                static_cast<std::uint32_t>(
                    staged.dynamicNodes_.size()
                );
            MRWorldDynamicNodeGPU node{};
            node.environment = MR_INVALID_INDEX;
            node.stableId = nodeIndex;
            node.kind = MR_WORLD_DYNAMIC_NODE_ARTICULATION;
            node.ownerIndex = owner;
            node.velocityOffset = owned.vOffset;
            node.velocityCount = owned.nv;
            node.configurationOffset = owned.qOffset;
            node.configurationCount = owned.nq;
            node.factorIndex = owner;
            node.generation = 1u;
            node.operatorBucket =
                owned.nv <= 8u ? 8u :
                owned.nv <= 16u ? 16u :
                owned.nv <= 32u ? 32u : 64u;
            node.flags =
                MR_WORLD_DYNAMIC_NODE_VALID |
                MR_WORLD_DYNAMIC_NODE_HAS_IMPLICIT_FACTOR |
                (
                    owned.rootType == MR_ROOT_FLOATING
                    ? MR_WORLD_DYNAMIC_NODE_FLOATING
                    : 0u
                );
            staged.dynamicNodes_.push_back(node);
            const std::uint64_t bodyEnd =
                static_cast<std::uint64_t>(owned.firstBody) +
                owned.bodyCount;
            for (std::uint64_t body = owned.firstBody;
                 body < bodyEnd;
                 ++body) {
                staged.bodyDynamicNodes_[body] = nodeIndex;
            }
        }
        staged.articulationQOffsets_.push_back(
            staged.model_.world.nq
        );
        staged.articulationVOffsets_.push_back(
            staged.model_.world.nv
        );
        std::uint32_t dynamicSceneOrdinal = 0u;
        for (std::uint32_t localScene = 0u;
             localScene < staged.sceneBodyIndices_.size();
             ++localScene) {
            const std::uint32_t body =
                staged.sceneBodyIndices_[localScene];
            if (staged.model_.bodies[body].motionType !=
                MR_MOTION_DYNAMIC) {
                continue;
            }
            const std::uint32_t nodeIndex =
                static_cast<std::uint32_t>(
                    staged.dynamicNodes_.size()
                );
            MRWorldDynamicNodeGPU node{};
            node.environment = MR_INVALID_INDEX;
            node.stableId = nodeIndex;
            node.kind = MR_WORLD_DYNAMIC_NODE_FREE_BODY;
            node.ownerIndex = localScene;
            node.velocityOffset =
                staged.model_.world.nv +
                6u * dynamicSceneOrdinal;
            node.velocityCount = 6u;
            node.configurationOffset = MR_INVALID_INDEX;
            node.configurationCount = 0u;
            node.factorIndex = localScene;
            node.generation = 1u;
            node.operatorBucket = 8u;
            node.flags =
                MR_WORLD_DYNAMIC_NODE_VALID |
                MR_WORLD_DYNAMIC_NODE_FLOATING |
                MR_WORLD_DYNAMIC_NODE_HAS_IMPLICIT_FACTOR;
            staged.dynamicNodes_.push_back(node);
            staged.bodyDynamicNodes_[body] = nodeIndex;
            staged.sceneBodyDynamicNodes_[localScene] = nodeIndex;
            ++dynamicSceneOrdinal;
        }
        const auto collisionExcluded = [&staged](
            const std::uint32_t colliderA,
            const std::uint32_t colliderB
        ) {
            const std::uint64_t key =
                (static_cast<std::uint64_t>(colliderA) << 32u) |
                colliderB;
            const auto& exclusions =
                staged.model_.collisionExclusions;
            const auto found = std::lower_bound(
                exclusions.begin(),
                exclusions.end(),
                key,
                [](const CollisionPairExclusion& exclusion,
                   const std::uint64_t candidate) {
                    const std::uint64_t exclusionKey =
                        (
                            static_cast<std::uint64_t>(
                                exclusion.colliderA
                            ) << 32u
                        ) |
                        exclusion.colliderB;
                    return exclusionKey < candidate;
                }
            );
            return
                found != exclusions.end() &&
                found->colliderA == colliderA &&
                found->colliderB == colliderB;
        };
        std::uint32_t convexCacheSlot = 0u;
        for (std::uint32_t colliderA = 0u;
             colliderA < staged.model_.shapes.size();
             ++colliderA) {
            const MRShapeGPU& shapeA =
                staged.model_.shapes[colliderA];
            if ((shapeA.flags &
                 MR_SHAPE_FLAG_SIMULATION_DISABLED) != 0u) {
                continue;
            }
            for (std::uint32_t colliderB = colliderA + 1u;
                 colliderB < staged.model_.shapes.size();
                 ++colliderB) {
                const MRShapeGPU& shapeB =
                    staged.model_.shapes[colliderB];
                if ((shapeB.flags &
                     MR_SHAPE_FLAG_SIMULATION_DISABLED) != 0u ||
                    shapeA.bodyIndex == shapeB.bodyIndex ||
                    collisionExcluded(colliderA, colliderB) ||
                    (shapeA.collisionGroup &
                     shapeB.collisionMask) == 0u ||
                    (shapeB.collisionGroup &
                     shapeA.collisionMask) == 0u) {
                    continue;
                }
                const MRBodyPropertiesGPU& bodyA =
                    staged.model_.bodies[shapeA.bodyIndex];
                const MRBodyPropertiesGPU& bodyB =
                    staged.model_.bodies[shapeB.bodyIndex];
                if (bodyA.motionType != MR_MOTION_DYNAMIC &&
                    bodyB.motionType != MR_MOTION_DYNAMIC) {
                    continue;
                }
                // Directly joint-connected links are a canonical cooker
                // exclusion. Solving their intentionally overlapping
                // collision proxies creates rank-deficient self constraints
                // and fights the authored joint.
                if (bodyA.articulationIndex != MR_INVALID_INDEX &&
                    bodyA.articulationIndex ==
                        bodyB.articulationIndex &&
                    (bodyA.parentBody == shapeB.bodyIndex ||
                     bodyB.parentBody == shapeA.bodyIndex)) {
                    continue;
                }
                const std::uint32_t pairClass =
                    compiledPairClass(
                        shapeA.shapeType,
                        shapeB.shapeType
                    );
                if (pairClass ==
                    MR_COLLISION_PAIR_UNSUPPORTED) {
                    return rejectCompile(
                        std::move(diagnostics),
                        MetalWorldHostStatus::unsupportedTopology,
                        "active collider pair " +
                            std::to_string(colliderA) +
                            "/" +
                            std::to_string(colliderB) +
                            " has no device narrowphase"
                    );
                }
                staged.eligiblePairs_.push_back({
                    .colliderA = colliderA,
                    .colliderB = colliderB,
                    .pairClass = pairClass,
                    .convexCacheSlot =
                        pairClass == MR_COLLISION_PAIR_CONVEX ||
                            pairClass == MR_COLLISION_PAIR_MESH
                        ? convexCacheSlot++
                        : MR_INVALID_INDEX,
                });
            }
        }

        const auto inferred = [](
            const std::uint32_t requested,
            const std::uint32_t fallback
        ) {
            return requested == 0u ? fallback : requested;
        };
        const std::uint64_t eligibleCount =
            staged.eligiblePairs_.size();
        const std::uint32_t eligibleU32 =
            static_cast<std::uint32_t>(eligibleCount);
        const std::uint32_t authoredBlockCount =
            static_cast<std::uint32_t>(
                staged.model_.constraintProgram.blocks.size()
            );
        for (std::uint32_t authored = 0u;
             authored < authoredBlockCount;
             ++authored) {
            const MRConstraintIRBlockGPU& block =
                staged.model_.constraintProgram.blocks[authored];
            if (block.dimension == 0u ||
                block.dimension > 3u ||
                block.endpointCount == 0u ||
                block.endpointCount > 2u) {
                return rejectCompile(
                    std::move(diagnostics),
                    MetalWorldHostStatus::unsupportedTopology,
                    "persistent MetalWorld currently requires authored "
                    "mechanism blocks with one-to-three rows and at most "
                    "two sparse endpoints"
                );
            }
            for (std::uint32_t local = 0u;
                 local < block.endpointCount;
                 ++local) {
                const MRConstraintIREndpointGPU& endpoint =
                    staged.model_.constraintProgram.endpoints[
                        block.endpointOffset + local
                    ];
                if (endpoint.jacobianKind !=
                    MR_CONSTRAINT_IR_JACOBIAN_GENERALIZED) {
                    return rejectCompile(
                        std::move(diagnostics),
                        MetalWorldHostStatus::unsupportedTopology,
                        "persistent authored mechanism constraints must "
                        "use sparse generalized endpoints"
                    );
                }
            }
        }
        const std::uint64_t pointJacobianWordsPerConstraint =
            6ull * staged.model_.world.nv;
        const std::uint32_t maximumConstraintCapacity =
            pointJacobianWordsPerConstraint == 0u
            ? 0u
            : static_cast<std::uint32_t>(
                  std::numeric_limits<std::uint32_t>::max() /
                  pointJacobianWordsPerConstraint
              );
        if (authoredBlockCount > maximumConstraintCapacity) {
            return rejectCompile(
                std::move(diagnostics),
                MetalWorldHostStatus::capacityOverflow,
                "authored mechanism program exceeds Metal "
                "constraint addressing"
            );
        }
        const std::uint32_t defaultRaw =
            static_cast<std::uint32_t>(
                std::max<std::uint64_t>(
                    1u,
                    std::min<std::uint64_t>(
                        MR_METAL_WORLD_RAW_CONTACTS_PER_PAIR *
                            eligibleCount,
                        std::numeric_limits<std::uint32_t>::max()
                    )
                )
            );
        const std::uint32_t defaultContactConstraints =
            static_cast<std::uint32_t>(
                std::min<std::uint64_t>(
                    4u * eligibleCount,
                    maximumConstraintCapacity -
                        authoredBlockCount
                )
            );
        const std::uint32_t defaultConstraints =
            std::max<std::uint32_t>(
                authoredBlockCount +
                    defaultContactConstraints,
                1u
            );
        staged.capacities_.candidatePairs = inferred(
            requestedCapacities.candidatePairs,
            std::max(eligibleU32, 1u)
        );
        staged.capacities_.rawContacts = inferred(
            requestedCapacities.rawContacts,
            defaultRaw
        );
        staged.capacities_.manifolds = inferred(
            requestedCapacities.manifolds,
            std::max(eligibleU32, 1u)
        );
        staged.capacities_.constraintBlocks = inferred(
            requestedCapacities.constraintBlocks,
            defaultConstraints
        );
        if (staged.capacities_.constraintBlocks <
            authoredBlockCount) {
            return rejectCompile(
                std::move(diagnostics),
                MetalWorldHostStatus::capacityOverflow,
                "constraint block capacity cannot hold the "
                "authored mechanism program"
            );
        }
        const std::uint64_t requiredRows =
            3ull * staged.capacities_.constraintBlocks;
        if (requiredRows >
            std::numeric_limits<std::uint32_t>::max()) {
            return rejectCompile(
                std::move(diagnostics),
                MetalWorldHostStatus::arithmeticOverflow,
                "contact row capacity overflows the GPU ABI"
            );
        }
        staged.capacities_.constraintRows = inferred(
            requestedCapacities.constraintRows,
            static_cast<std::uint32_t>(requiredRows)
        );
        staged.capacities_.islands = inferred(
            requestedCapacities.islands,
            std::max(
                staged.model_.world.islandCapacity,
                static_cast<std::uint32_t>(
                    std::min<std::size_t>(
                        staged.model_.bodies.size(),
                        MR_ARTICULATED_OPERATOR_MAX_BODIES
                    )
                )
            )
        );
        staged.capacities_.hardConvexPairs = inferred(
            requestedCapacities.hardConvexPairs,
            eligibleU32
        );
        staged.capacities_.meshTriangleCandidates = inferred(
            requestedCapacities.meshTriangleCandidates,
            defaultRaw
        );
        const std::uint32_t minimumSolverTiles =
            requiredSolverTileCapacity(
                staged.capacities_.constraintBlocks,
                authoredBlockCount,
                staged.capacities_.islands,
                staged.dynamicNodes_.size()
            );
        staged.capacities_.solverTiles = inferred(
            requestedCapacities.solverTiles,
            minimumSolverTiles
        );
        const std::uint32_t defaultSpillRows =
            requiredSpillRowCapacity(
                staged.capacities_.constraintBlocks,
                authoredBlockCount
            );
        staged.capacities_.spillRows = inferred(
            requestedCapacities.spillRows,
            defaultSpillRows
        );
        staged.capacities_.ccdCandidates = inferred(
            requestedCapacities.ccdCandidates,
            eligibleU32
        );
        staged.capacities_.ccdEvents = inferred(
            requestedCapacities.ccdEvents,
            std::max<std::uint32_t>(
                1u,
                std::min<std::uint32_t>(
                    eligibleU32,
                    MR_CCD_DEFAULT_MAX_EVENTS
                )
            )
        );
        const std::uint32_t endpointRecords =
            staged.capacities_.constraintBlocks >
                    0x7fffffffu
            ? 0xffffffffu
            : 2u * staged.capacities_.constraintBlocks;
        const std::uint32_t minimumEndpointRecords =
            defaultConstraints > 0x7fffffffu
            ? 0xffffffffu
            : 2u * defaultConstraints;
        std::uint32_t dynamicSceneBodies = 0u;
        for (const std::uint32_t body :
             staged.sceneBodyIndices_) {
            if (staged.model_.bodies[body].motionType ==
                MR_MOTION_DYNAMIC) {
                ++dynamicSceneBodies;
            }
        }
        const std::uint64_t qualityVelocity64 =
            static_cast<std::uint64_t>(
                staged.model_.world.nv
            ) +
            6ull * dynamicSceneBodies;
        const std::uint32_t qualityVelocities =
            static_cast<std::uint32_t>(
                std::min<std::uint64_t>(
                    qualityVelocity64,
                    0xffffffffull
                )
            );
        staged.capacities_.endpointRuntimeRecords = inferred(
            requestedCapacities.endpointRuntimeRecords,
            endpointRecords
        );
        staged.capacities_.articulationPointQueries = inferred(
            requestedCapacities.articulationPointQueries,
            endpointRecords
        );
        if (staged.capacities_.endpointRuntimeRecords !=
                endpointRecords ||
            staged.capacities_.articulationPointQueries !=
                endpointRecords) {
            return rejectCompile(
                std::move(diagnostics),
                MetalWorldHostStatus::capacityOverflow,
                "endpoint-runtime and articulation-query capacities must "
                "equal twice the constraint-block capacity"
            );
        }
        staged.capacities_.rodCandidatePairs =
            requestedCapacities.rodCandidatePairs;
        staged.capacities_.rodRawContacts =
            requestedCapacities.rodRawContacts;
        staged.capacities_.rodManifolds =
            requestedCapacities.rodManifolds;
        staged.capacities_.rodCCDEvents =
            requestedCapacities.rodCCDEvents;
        staged.capacities_.qualityGeneralizedVelocities = inferred(
            requestedCapacities.qualityGeneralizedVelocities,
            qualityVelocities
        );
        staged.capacities_.qualityRows = inferred(
            requestedCapacities.qualityRows,
            staged.capacities_.constraintRows
        );
        staged.capacities_.qualityKrylovVectors = inferred(
            requestedCapacities.qualityKrylovVectors,
            8u
        );
        staged.capacities_.qualityDirectTiles = inferred(
            requestedCapacities.qualityDirectTiles,
            std::max<std::uint32_t>(
                1u,
                (
                    qualityVelocities +
                    MR_SIMD_WIDTH - 1u
                ) / MR_SIMD_WIDTH
            )
        );
        const std::uint32_t dynamicNodeCount =
            static_cast<std::uint32_t>(
                staged.dynamicNodes_.size()
            );
        staged.capacities_.dynamicNodes = inferred(
            requestedCapacities.dynamicNodes,
            std::max(dynamicNodeCount, 1u)
        );
        staged.capacities_.islandNodeReferences = inferred(
            requestedCapacities.islandNodeReferences,
            std::max(dynamicNodeCount, 1u)
        );
        staged.capacities_.islandConstraintReferences = inferred(
            requestedCapacities.islandConstraintReferences,
            staged.capacities_.constraintBlocks
        );
        staged.capacities_.rodFactorBlocks =
            requestedCapacities.rodFactorBlocks;
        staged.capacities_.operatorVelocityElements = inferred(
            requestedCapacities.operatorVelocityElements,
            std::max(qualityVelocities, 1u)
        );
        staged.minimumCapacities_ = {
            .candidatePairs = std::max(eligibleU32, 1u),
            .rawContacts = defaultRaw,
            .manifolds = std::max(eligibleU32, 1u),
            .constraintBlocks = defaultConstraints,
            .constraintRows = 3u * defaultConstraints,
            .islands = std::max(
                staged.model_.world.islandCapacity,
                static_cast<std::uint32_t>(
                    std::min<std::size_t>(
                        staged.model_.bodies.size(),
                        MR_ARTICULATED_OPERATOR_MAX_BODIES
                    )
                )
            ),
            .hardConvexPairs = eligibleU32,
            .meshTriangleCandidates = defaultRaw,
            .solverTiles = minimumSolverTiles,
            .spillRows = defaultSpillRows,
            .ccdCandidates = eligibleU32,
            .ccdEvents = std::max<std::uint32_t>(
                1u,
                std::min<std::uint32_t>(
                    eligibleU32,
                    MR_CCD_DEFAULT_MAX_EVENTS
                )
            ),
            .endpointRuntimeRecords = minimumEndpointRecords,
            .articulationPointQueries = minimumEndpointRecords,
            .qualityGeneralizedVelocities =
                qualityVelocities,
            .qualityRows = 3u * defaultConstraints,
            .qualityKrylovVectors = 8u,
            .qualityDirectTiles = std::max<std::uint32_t>(
                1u,
                (
                    qualityVelocities +
                    MR_SIMD_WIDTH - 1u
                ) / MR_SIMD_WIDTH
            ),
            .dynamicNodes = std::max(dynamicNodeCount, 1u),
            .islandNodeReferences =
                std::max(dynamicNodeCount, 1u),
            .islandConstraintReferences =
                defaultConstraints,
            .rodFactorBlocks = 0u,
            .operatorVelocityElements =
                std::max(qualityVelocities, 1u),
        };
        if (staged.capacities_.candidatePairs == 0u ||
            staged.capacities_.rawContacts == 0u ||
            staged.capacities_.manifolds == 0u ||
            staged.capacities_.constraintBlocks == 0u ||
            staged.capacities_.constraintBlocks >
                maximumConstraintCapacity ||
            staged.capacities_.constraintRows <
                requiredRows ||
            staged.capacities_.islands == 0u ||
            staged.capacities_.solverTiles <
                minimumSolverTiles ||
            staged.capacities_.ccdEvents == 0u ||
            staged.capacities_.dynamicNodes <
                dynamicNodeCount ||
            staged.capacities_.islandNodeReferences <
                dynamicNodeCount ||
            staged.capacities_.islandConstraintReferences <
                staged.capacities_.constraintBlocks ||
            staged.capacities_.operatorVelocityElements <
                qualityVelocities ||
            staged.dynamicNodes_.size() >
                MR_WORLD_MAX_DYNAMIC_NODES ||
            staged.model_.bodies.size() >
                MR_ARTICULATED_OPERATOR_MAX_BODIES ||
            staged.model_.shapes.size() >
                std::numeric_limits<std::uint32_t>::max() ||
            staged.sceneBodyIndices_.size() >
                std::numeric_limits<std::uint32_t>::max() ||
            staged.eligiblePairs_.size() >
                std::numeric_limits<std::uint32_t>::max()) {
            return rejectCompile(
                std::move(diagnostics),
                MetalWorldHostStatus::capacityOverflow,
                "contact capacity profile is empty, internally "
                "inconsistent, or exceeds checked GPU ABI strides"
            );
        }

        staged.modelFingerprint_ = fingerprint(staged.model_);
        staged.fingerprint_ = compiledFingerprint(
            staged.model_,
            staged.capacities_,
            staged.sceneBodyIndices_,
            staged.eligiblePairs_,
            staged.dynamicNodes_,
            staged.bodyDynamicNodes_
        );
        diagnostics.fingerprint = staged.fingerprint_;
        compiled = std::move(staged);
        diagnostics.status = MetalWorldHostStatus::success;
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return rejectCompile(
            std::move(diagnostics),
            MetalWorldHostStatus::metalBufferFailure,
            "host allocation failed while compiling MetalWorld"
        );
    } catch (const std::exception& exception) {
        return rejectCompile(
            std::move(diagnostics),
            MetalWorldHostStatus::internalFailure,
            exception.what()
        );
    }
}

MetalWorldCompileDiagnostics compileMetalWorld(
    const HeterogeneousWorld& world,
    CompiledWorld& compiled,
    const MetalWorldCapacityProfile& requestedCapacities
) {
    MetalWorldCompileDiagnostics diagnostics{};
    try {
        std::string reason;
        if (!world.valid(&reason)) {
            return rejectCompile(
                std::move(diagnostics),
                MetalWorldHostStatus::invalidModel,
                "invalid HeterogeneousWorld: " + reason
            );
        }
        CompiledWorld staged;
        diagnostics = compileMetalWorld(
            world.model,
            0u,
            staged,
            requestedCapacities
        );
        if (!diagnostics.succeeded()) {
            return diagnostics;
        }
        staged.rodPrograms_ = world.rods;
        staged.rodNodeOffsets_.push_back(0u);
        staged.rodEdgeOffsets_.push_back(0u);

        std::uint64_t rodPairCount = 0u;
        std::uint64_t rodNodeCount = 0u;
        std::uint64_t rodEdgeCount = 0u;
        std::uint64_t rodCCDPairCount = 0u;
        std::uint64_t rodHardPairCount = 0u;
        std::uint64_t rodMeshPairCount = 0u;
        std::uint64_t rodAttachmentConstraintCount = 0u;
        std::uint64_t rodVelocityCursor =
            staged.minimumCapacities_
                .qualityGeneralizedVelocities;
        staged.rodDynamicNodes_.reserve(world.rods.size());
        for (std::uint32_t rodIndex = 0u;
             rodIndex < world.rods.size();
             ++rodIndex) {
            const HeterogeneousRodProgram& program =
                world.rods[rodIndex];
            const std::uint32_t nodeOffset =
                static_cast<std::uint32_t>(rodNodeCount);
            const std::uint32_t edgeOffset =
                static_cast<std::uint32_t>(rodEdgeCount);
            const std::uint32_t nodes =
                static_cast<std::uint32_t>(
                    program.model.restPositions.size()
                );
            const std::uint32_t edges = nodes - 1u;
            const std::uint64_t rodVelocityCount =
                3ull * nodes + edges;
            const std::uint64_t rodVelocityEnd =
                rodVelocityCursor + rodVelocityCount;
            rodNodeCount += nodes;
            rodEdgeCount += edges;
            if (rodNodeCount >
                    std::numeric_limits<std::uint32_t>::max() ||
                rodEdgeCount >
                    std::numeric_limits<std::uint32_t>::max() ||
                rodVelocityEnd >
                    std::numeric_limits<std::uint32_t>::max()) {
                return rejectCompile(
                    std::move(diagnostics),
                    MetalWorldHostStatus::capacityOverflow,
                    "flattened rod topology or operator state exceeds "
                    "32-bit indexing"
                );
            }

            const std::uint32_t dynamicNode =
                static_cast<std::uint32_t>(
                    staged.dynamicNodes_.size()
                );
            MRWorldDynamicNodeGPU node{};
            node.environment = MR_INVALID_INDEX;
            node.stableId = dynamicNode;
            node.kind = MR_WORLD_DYNAMIC_NODE_ROD;
            node.ownerIndex = rodIndex;
            node.velocityOffset =
                static_cast<std::uint32_t>(rodVelocityCursor);
            node.velocityCount =
                static_cast<std::uint32_t>(rodVelocityCount);
            node.configurationOffset = MR_INVALID_INDEX;
            node.configurationCount = 0u;
            node.factorIndex = rodIndex;
            node.generation =
                program.collision.topologyGeneration;
            node.operatorBucket =
                rodVelocityCount <= 32u ? 32u :
                rodVelocityCount <= 64u ? 64u : 128u;
            node.flags =
                MR_WORLD_DYNAMIC_NODE_VALID |
                MR_WORLD_DYNAMIC_NODE_HAS_IMPLICIT_FACTOR;
            staged.dynamicNodes_.push_back(node);
            staged.rodDynamicNodes_.push_back(dynamicNode);
            rodVelocityCursor = rodVelocityEnd;

            std::set<
                std::pair<std::uint32_t, std::uint32_t>
            > attachmentExclusions;
            for (std::size_t attachment = 0u;
                 attachment < program.attachments.size();
                 ++attachment) {
                const DiscreteRodAttachment& attachmentRecord =
                    program.attachments[attachment];
                const auto& binding =
                    program.rigidBindings[attachment];
                const std::uint32_t globalNode =
                    nodeOffset + attachmentRecord.nodeIndex;
                const std::uint32_t globalBody =
                    binding.bodyIndex ==
                            kDiscreteRodNoRigidBody
                    ? MR_INVALID_INDEX
                    : world.sceneBodyIndices[binding.bodyIndex];
                for (std::uint32_t axis = 0u;
                     axis < 3u;
                     ++axis) {
                    const std::uint32_t endpointOffset =
                        static_cast<std::uint32_t>(
                            staged.model_.constraintProgram
                                .endpoints.size()
                        );
                    const std::uint32_t rowOffset =
                        static_cast<std::uint32_t>(
                            staged.model_.constraintProgram
                                .rows.size()
                        );

                    MRConstraintIRBlockGPU block{};
                    block.key.words[0] = 0x52415454u;
                    block.key.words[1] = rodIndex;
                    block.key.words[2] =
                        static_cast<std::uint32_t>(attachment);
                    block.key.words[3] = axis;
                    block.type = MR_CONSTRAINT_BILATERAL;
                    block.dimension = 1u;
                    block.flags =
                        MR_CONSTRAINT_IR_BLOCK_ROD_ATTACHMENT;
                    block.islandIndex = MR_INVALID_INDEX;
                    block.endpointOffset = endpointOffset;
                    block.endpointCount = 2u;
                    block.rowOffset = rowOffset;
                    block.impulseOffset = rowOffset;
                    block.coneIndex =
                        MR_CONSTRAINT_IR_INVALID_INDEX;
                    block.eventSlot =
                        MR_CONSTRAINT_IR_INVALID_INDEX;

                    MRConstraintIREndpointGPU rodEndpoint{};
                    rodEndpoint.objectIndex = globalNode;
                    rodEndpoint.articulationIndex = rodIndex;
                    rodEndpoint.linkIndex =
                        MR_CONSTRAINT_IR_INVALID_INDEX;
                    rodEndpoint.role =
                        MR_CONSTRAINT_IR_ENDPOINT_A;
                    rodEndpoint.jacobianKind =
                        MR_CONSTRAINT_IR_JACOBIAN_ROD_NODE;

                    MRConstraintIREndpointGPU targetEndpoint{};
                    targetEndpoint.objectIndex = globalBody;
                    targetEndpoint.articulationIndex =
                        MR_CONSTRAINT_IR_INVALID_INDEX;
                    targetEndpoint.linkIndex =
                        MR_CONSTRAINT_IR_INVALID_INDEX;
                    targetEndpoint.role =
                        globalBody == MR_INVALID_INDEX
                        ? MR_CONSTRAINT_IR_ENDPOINT_WORLD
                        : MR_CONSTRAINT_IR_ENDPOINT_B;
                    targetEndpoint.jacobianKind =
                        globalBody == MR_INVALID_INDEX
                        ? MR_CONSTRAINT_IR_JACOBIAN_WORLD_POINT
                        : MR_CONSTRAINT_IR_JACOBIAN_BODY_LOCAL_POINT;
                    targetEndpoint.anchor = {
                        static_cast<float>(
                            globalBody == MR_INVALID_INDEX
                            ? attachmentRecord.targetPosition[0]
                            : binding.localAnchor[0]
                        ),
                        static_cast<float>(
                            globalBody == MR_INVALID_INDEX
                            ? attachmentRecord.targetPosition[1]
                            : binding.localAnchor[1]
                        ),
                        static_cast<float>(
                            globalBody == MR_INVALID_INDEX
                            ? attachmentRecord.targetPosition[2]
                            : binding.localAnchor[2]
                        ),
                        1.0f,
                    };

                    MRConstraintIRRowGPU row{};
                    row.direction = {
                        axis == 0u ? 1.0f : 0.0f,
                        axis == 1u ? 1.0f : 0.0f,
                        axis == 2u ? 1.0f : 0.0f,
                        0.0f,
                    };
                    row.positionError = 0.0f;
                    row.targetVelocity =
                        globalBody == MR_INVALID_INDEX
                        ? static_cast<float>(
                              attachmentRecord
                                  .targetVelocity[axis]
                          )
                        : 0.0f;
                    row.compliance = static_cast<float>(
                        attachmentRecord.compliance
                    );
                    row.timeConstant = std::max(
                        static_cast<float>(
                            2.0 * program.stepConfig.timestep
                        ),
                        1.0e-5f
                    );
                    row.dampingRatio = 1.0f;
                    row.impulseLower =
                        -MR_CONSTRAINT_IR_UNBOUNDED;
                    row.impulseUpper =
                        MR_CONSTRAINT_IR_UNBOUNDED;
                    row.flags =
                        MR_CONSTRAINT_IR_ROW_POSITION_STABILIZED;

                    staged.model_.constraintProgram.blocks
                        .push_back(block);
                    staged.model_.constraintProgram.endpoints
                        .push_back(rodEndpoint);
                    staged.model_.constraintProgram.endpoints
                        .push_back(targetEndpoint);
                    staged.model_.constraintProgram.rows
                        .push_back(row);
                    staged.model_.constraintProgram.warmImpulses
                        .push_back(0.0f);
                    ++rodAttachmentConstraintCount;
                }
                if (binding.bodyIndex ==
                    kDiscreteRodNoRigidBody) {
                    continue;
                }
                const std::uint32_t node =
                    attachmentRecord.nodeIndex;
                if (node > 0u) {
                    attachmentExclusions.emplace(
                        node - 1u,
                        globalBody
                    );
                }
                if (node < edges) {
                    attachmentExclusions.emplace(
                        node,
                        globalBody
                    );
                }
            }

            for (std::uint32_t node = 0u;
                 node < nodes;
                 ++node) {
                const auto& position =
                    program.defaultState.positions[node];
                const auto& velocity =
                    program.defaultState.velocities[node];
                staged.defaultRodNodes_.push_back({
                    .position = {
                        static_cast<float>(position[0]),
                        static_cast<float>(position[1]),
                        static_cast<float>(position[2]),
                        1.0f,
                    },
                    .velocity = {
                        static_cast<float>(velocity[0]),
                        static_cast<float>(velocity[1]),
                        static_cast<float>(velocity[2]),
                        0.0f,
                    },
                });
            }
            for (std::uint32_t edge = 0u;
                 edge < edges;
                 ++edge) {
                staged.defaultRodEdges_.push_back({
                    .twistAndRate = {
                        static_cast<float>(
                            program.defaultState.twists[edge]
                        ),
                        static_cast<float>(
                            program.defaultState
                                .twistRates[edge]
                        ),
                        0.0f,
                        0.0f,
                    },
                });
                MRRodColliderGPU collider{};
                collider.rodIndex = rodIndex;
                collider.edgeIndex = edgeOffset + edge;
                collider.nodeA = nodeOffset + edge;
                collider.nodeB = nodeOffset + edge + 1u;
                collider.materialIndex =
                    program.collision.materialIndex;
                collider.collisionGroup =
                    program.collision.collisionGroup;
                collider.collisionMask =
                    program.collision.collisionMask;
                collider.topologyGeneration =
                    program.collision.topologyGeneration;
                collider.radiusAndOffsets = {
                    static_cast<float>(program.model.radius),
                    static_cast<float>(
                        program.collision.contactOffset
                    ),
                    static_cast<float>(
                        program.collision.restOffset
                    ),
                    static_cast<float>(
                        program.model.radius +
                        0.5 *
                            program.model.restLengths[edge]
                    ),
                };
                collider.flagsAndExclusions.x =
                    (
                        program.collision.enableToolCollision
                        ? MR_ROD_GPU_FLAG_TOOL_COLLISION
                        : 0u
                    ) |
                    (
                        program.collision.enableCCD
                        ? MR_ROD_GPU_FLAG_ENABLE_CCD
                        : 0u
                    );
                collider.flagsAndExclusions.y = dynamicNode;
                staged.rodColliders_.push_back(collider);

                MRShapeGPU source{};
                source.bodyIndex = MR_INVALID_INDEX;
                source.shapeType = MR_SHAPE_CAPSULE;
                source.materialIndex =
                    program.collision.materialIndex;
                source.collisionGroup =
                    program.collision.collisionGroup;
                source.collisionMask =
                    program.collision.collisionMask;
                source.slotGeneration =
                    program.collision.topologyGeneration;
                source.localRotation = {
                    0.0f,
                    0.0f,
                    0.0f,
                    1.0f,
                };
                source.dimensions = {
                    static_cast<float>(program.model.radius),
                    static_cast<float>(
                        0.5 *
                        program.model.restLengths[edge]
                    ),
                    0.0f,
                    0.0f,
                };
                source.contactRestAndBoundingRadius =
                    collider.radiusAndOffsets;
                staged.rodShapeSources_.push_back(source);

                if (!program.collision.enableToolCollision) {
                    continue;
                }
                for (std::uint32_t rigidCollider = 0u;
                     rigidCollider <
                         world.model.shapes.size();
                     ++rigidCollider) {
                    const MRShapeGPU& rigid =
                        world.model.shapes[rigidCollider];
                    if ((rigid.flags &
                         MR_SHAPE_FLAG_SIMULATION_DISABLED) !=
                            0u ||
                        (collider.collisionGroup &
                         rigid.collisionMask) == 0u ||
                        (rigid.collisionGroup &
                         collider.collisionMask) == 0u ||
                        attachmentExclusions.contains({
                            edge,
                            rigid.bodyIndex,
                        })) {
                        continue;
                    }
                    const std::uint32_t pairClass =
                        compiledPairClass(
                            MR_SHAPE_CAPSULE,
                            rigid.shapeType
                        );
                    if (pairClass ==
                        MR_COLLISION_PAIR_UNSUPPORTED) {
                        return rejectCompile(
                            std::move(diagnostics),
                            MetalWorldHostStatus::
                                unsupportedTopology,
                            "rod edge/tool collider has no Metal "
                            "narrowphase"
                        );
                    }
                    const bool enablePairCCD =
                        program.collision.enableCCD &&
                        (
                            rigid.flags &
                            MR_SHAPE_FLAG_ENABLE_CCD
                        ) != 0u;
                    staged.rodToolPairs_.push_back({
                        .rodCollider = edgeOffset + edge,
                        .rigidCollider = rigidCollider,
                        .pairClass = pairClass,
                        .flags =
                            MR_ROD_TOOL_PAIR_VALID |
                            (
                                enablePairCCD
                                ? MR_ROD_TOOL_PAIR_ENABLE_CCD
                                : 0u
                            ),
                    });
                    ++rodPairCount;
                    rodCCDPairCount +=
                        enablePairCCD ? 1u : 0u;
                    rodHardPairCount +=
                        pairClass == MR_COLLISION_PAIR_CONVEX
                        ? 1u
                        : 0u;
                    if (pairClass == MR_COLLISION_PAIR_MESH) {
                        const MRShapeGPU& meshShape =
                            rigid.shapeType ==
                                MR_SHAPE_TRIANGLE_MESH
                            ? rigid
                            : source;
                        if (meshShape.shapeType ==
                                MR_SHAPE_TRIANGLE_MESH &&
                            meshShape.geometryOffset <
                                world.model
                                    .geometryHeaders.size()) {
                            rodMeshPairCount +=
                                world.model.geometryHeaders[
                                    meshShape.geometryOffset
                                ].triangleCount;
                        }
                    }
                }
            }
            staged.rodNodeOffsets_.push_back(
                static_cast<std::uint32_t>(rodNodeCount)
            );
            staged.rodEdgeOffsets_.push_back(
                static_cast<std::uint32_t>(rodEdgeCount)
            );
        }

        const std::uint64_t rodRaw =
            rodPairCount *
            MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
        const std::uint64_t rodConstraints =
            rodRaw + rodAttachmentConstraintCount;
        const std::uint64_t rodRows = 3u * rodConstraints;
        const std::uint64_t rodVelocities =
            3u * rodNodeCount + rodEdgeCount;
        const std::uint64_t rodFactorNumerics =
            static_cast<std::uint64_t>(
                MR_ROD_FACTOR_TRANSLATION_FLOATS_PER_NODE
            ) * rodNodeCount +
            static_cast<std::uint64_t>(
                MR_ROD_FACTOR_TWIST_FLOATS_PER_EDGE
            ) * rodEdgeCount;
        const std::uint64_t rodOperatorElements =
            rodFactorNumerics + rodVelocities;
        const auto checkedU32 = [&diagnostics](
            const std::uint64_t value,
            const char* label,
            std::uint32_t& output
        ) {
            if (value >
                std::numeric_limits<std::uint32_t>::max()) {
                diagnostics.status =
                    MetalWorldHostStatus::capacityOverflow;
                diagnostics.message =
                    std::string(label) +
                    " exceeds 32-bit world capacity";
                return false;
            }
            output = static_cast<std::uint32_t>(value);
            return true;
        };
        std::uint32_t requiredRodPairs = 0u;
        std::uint32_t requiredRodRaw = 0u;
        std::uint32_t requiredRodConstraints = 0u;
        std::uint32_t requiredRodRows = 0u;
        std::uint32_t requiredRodVelocities = 0u;
        std::uint32_t requiredRodOperatorElements = 0u;
        std::uint32_t requiredRodNodes = 0u;
        std::uint32_t requiredRodComponents = 0u;
        std::uint32_t requiredRodCCDPairs = 0u;
        if (!checkedU32(
                rodPairCount,
                "rod pair count",
                requiredRodPairs
            ) ||
            !checkedU32(
                rodRaw,
                "rod raw-contact count",
                requiredRodRaw
            ) ||
            !checkedU32(
                rodConstraints,
                "rod constraint count",
                requiredRodConstraints
            ) ||
            !checkedU32(
                rodRows,
                "rod row count",
                requiredRodRows
            ) ||
            !checkedU32(
                rodVelocities,
                "rod generalized velocity count",
                requiredRodVelocities
            ) ||
            !checkedU32(
                rodOperatorElements,
                "rod retained-factor/operator element count",
                requiredRodOperatorElements
            ) ||
            !checkedU32(
                rodNodeCount,
                "rod factor block count",
                requiredRodNodes
            ) ||
            !checkedU32(
                world.rods.size(),
                "rod dynamic-node count",
                requiredRodComponents
            ) ||
            !checkedU32(
                rodCCDPairCount,
                "rod CCD pair count",
                requiredRodCCDPairs
            )) {
            return diagnostics;
        }

        const auto addCapacity = [&diagnostics](
            const char* label,
            const std::uint32_t requested,
            const std::uint32_t base,
            const std::uint32_t additional,
            std::uint32_t& capacity,
            std::uint32_t& minimum
        ) {
            const std::uint64_t required =
                static_cast<std::uint64_t>(base) +
                additional;
            if (required >
                std::numeric_limits<std::uint32_t>::max()) {
                diagnostics.status =
                    MetalWorldHostStatus::capacityOverflow;
                diagnostics.message =
                    std::string(label) +
                    " exceeds 32-bit world capacity";
                return false;
            }
            minimum = static_cast<std::uint32_t>(required);
            if (requested != 0u && requested < minimum) {
                diagnostics.status =
                    MetalWorldHostStatus::capacityOverflow;
                diagnostics.message =
                    std::string(label) +
                    " is below the heterogeneous minimum";
                return false;
            }
            capacity =
                requested == 0u ? minimum : requested;
            return true;
        };
        if (!addCapacity(
                "raw-contact capacity",
                requestedCapacities.rawContacts,
                staged.minimumCapacities_.rawContacts,
                requiredRodRaw,
                staged.capacities_.rawContacts,
                staged.minimumCapacities_.rawContacts
            ) ||
            !addCapacity(
                "manifold capacity",
                requestedCapacities.manifolds,
                staged.minimumCapacities_.manifolds,
                requiredRodPairs,
                staged.capacities_.manifolds,
                staged.minimumCapacities_.manifolds
            ) ||
            !addCapacity(
                "constraint-block capacity",
                requestedCapacities.constraintBlocks,
                staged.minimumCapacities_.constraintBlocks,
                requiredRodConstraints,
                staged.capacities_.constraintBlocks,
                staged.minimumCapacities_.constraintBlocks
            ) ||
            !addCapacity(
                "constraint-row capacity",
                requestedCapacities.constraintRows,
                staged.minimumCapacities_.constraintRows,
                requiredRodRows,
                staged.capacities_.constraintRows,
                staged.minimumCapacities_.constraintRows
            ) ||
            !addCapacity(
                "quality generalized velocity capacity",
                requestedCapacities
                    .qualityGeneralizedVelocities,
                staged.minimumCapacities_
                    .qualityGeneralizedVelocities,
                requiredRodVelocities,
                staged.capacities_
                    .qualityGeneralizedVelocities,
                staged.minimumCapacities_
                    .qualityGeneralizedVelocities
            ) ||
            !addCapacity(
                "quality row capacity",
                requestedCapacities.qualityRows,
                staged.minimumCapacities_.qualityRows,
                requiredRodRows,
                staged.capacities_.qualityRows,
                staged.minimumCapacities_.qualityRows
            ) ||
            !addCapacity(
                "dynamic-node capacity",
                requestedCapacities.dynamicNodes,
                staged.minimumCapacities_.dynamicNodes,
                requiredRodComponents,
                staged.capacities_.dynamicNodes,
                staged.minimumCapacities_.dynamicNodes
            ) ||
            !addCapacity(
                "island node-reference capacity",
                requestedCapacities.islandNodeReferences,
                staged.minimumCapacities_
                    .islandNodeReferences,
                requiredRodComponents,
                staged.capacities_.islandNodeReferences,
                staged.minimumCapacities_
                    .islandNodeReferences
            ) ||
            !addCapacity(
                "island constraint-reference capacity",
                requestedCapacities.islandConstraintReferences,
                staged.minimumCapacities_
                    .islandConstraintReferences,
                requiredRodConstraints,
                staged.capacities_
                    .islandConstraintReferences,
                staged.minimumCapacities_
                    .islandConstraintReferences
            ) ||
            !addCapacity(
                "rod factor-block capacity",
                requestedCapacities.rodFactorBlocks,
                staged.minimumCapacities_.rodFactorBlocks,
                requiredRodNodes,
                staged.capacities_.rodFactorBlocks,
                staged.minimumCapacities_.rodFactorBlocks
            ) ||
            !addCapacity(
                "operator velocity capacity",
                requestedCapacities.operatorVelocityElements,
                staged.minimumCapacities_
                    .operatorVelocityElements,
                requiredRodOperatorElements,
                staged.capacities_.operatorVelocityElements,
                staged.minimumCapacities_
                    .operatorVelocityElements
            ) ||
            !addCapacity(
                "CCD candidate capacity",
                requestedCapacities.ccdCandidates,
                staged.minimumCapacities_.ccdCandidates,
                requiredRodCCDPairs,
                staged.capacities_.ccdCandidates,
                staged.minimumCapacities_.ccdCandidates
            ) ||
            !addCapacity(
                "CCD event capacity",
                requestedCapacities.ccdEvents,
                staged.minimumCapacities_.ccdEvents,
                requiredRodCCDPairs,
                staged.capacities_.ccdEvents,
                staged.minimumCapacities_.ccdEvents
            )) {
            return diagnostics;
        }
        const std::uint64_t endpointCapacity =
            2ull * staged.capacities_.constraintBlocks;
        const std::uint64_t minimumEndpointCapacity =
            2ull * staged.minimumCapacities_.constraintBlocks;
        if (endpointCapacity >
                std::numeric_limits<std::uint32_t>::max() ||
            minimumEndpointCapacity >
                std::numeric_limits<std::uint32_t>::max() ||
            (requestedCapacities.endpointRuntimeRecords != 0u &&
             requestedCapacities.endpointRuntimeRecords !=
                 endpointCapacity) ||
            (requestedCapacities.articulationPointQueries != 0u &&
             requestedCapacities.articulationPointQueries !=
                 endpointCapacity)) {
            return rejectCompile(
                std::move(diagnostics),
                MetalWorldHostStatus::capacityOverflow,
                "endpoint-runtime and articulation-query capacities must "
                "equal twice the final constraint-block capacity"
            );
        }
        staged.capacities_.endpointRuntimeRecords =
            static_cast<std::uint32_t>(endpointCapacity);
        staged.capacities_.articulationPointQueries =
            static_cast<std::uint32_t>(endpointCapacity);
        staged.minimumCapacities_.endpointRuntimeRecords =
            static_cast<std::uint32_t>(
                minimumEndpointCapacity
            );
        staged.minimumCapacities_.articulationPointQueries =
            static_cast<std::uint32_t>(
                minimumEndpointCapacity
            );
        staged.capacities_.rodCandidatePairs =
            requestedCapacities.rodCandidatePairs == 0u
            ? requiredRodPairs
            : requestedCapacities.rodCandidatePairs;
        staged.capacities_.rodRawContacts =
            requestedCapacities.rodRawContacts == 0u
            ? requiredRodRaw
            : requestedCapacities.rodRawContacts;
        staged.capacities_.rodManifolds =
            requestedCapacities.rodManifolds == 0u
            ? requiredRodPairs
            : requestedCapacities.rodManifolds;
        staged.capacities_.rodCCDEvents =
            requestedCapacities.rodCCDEvents == 0u
            ? requiredRodCCDPairs
            : requestedCapacities.rodCCDEvents;
        staged.minimumCapacities_.rodCandidatePairs =
            requiredRodPairs;
        staged.minimumCapacities_.rodRawContacts =
            requiredRodRaw;
        staged.minimumCapacities_.rodManifolds =
            requiredRodPairs;
        staged.minimumCapacities_.rodCCDEvents =
            requiredRodCCDPairs;
        if (staged.capacities_.rodCandidatePairs <
                requiredRodPairs ||
            staged.capacities_.rodRawContacts <
                requiredRodRaw ||
            staged.capacities_.rodManifolds <
                requiredRodPairs ||
            staged.capacities_.rodCCDEvents <
                rodCCDPairCount) {
            return rejectCompile(
                std::move(diagnostics),
                MetalWorldHostStatus::capacityOverflow,
                "rod collision capacity is below the cooked minimum"
            );
        }
        const std::uint64_t hardRequired =
            staged.minimumCapacities_.hardConvexPairs +
            rodHardPairCount;
        const std::uint64_t meshRequired =
            staged.minimumCapacities_
                .meshTriangleCandidates +
            rodMeshPairCount;
        if (hardRequired >
                std::numeric_limits<std::uint32_t>::max() ||
            meshRequired >
                std::numeric_limits<std::uint32_t>::max()) {
            return rejectCompile(
                std::move(diagnostics),
                MetalWorldHostStatus::capacityOverflow,
                "heterogeneous hard-query capacity overflows"
            );
        }
        staged.minimumCapacities_.hardConvexPairs =
            static_cast<std::uint32_t>(hardRequired);
        staged.minimumCapacities_.meshTriangleCandidates =
            static_cast<std::uint32_t>(meshRequired);
        if ((requestedCapacities.hardConvexPairs != 0u &&
             requestedCapacities.hardConvexPairs <
                 staged.minimumCapacities_.hardConvexPairs) ||
            (requestedCapacities.meshTriangleCandidates !=
                 0u &&
             requestedCapacities.meshTriangleCandidates <
                 staged.minimumCapacities_
                     .meshTriangleCandidates)) {
            return rejectCompile(
                std::move(diagnostics),
                MetalWorldHostStatus::capacityOverflow,
                "convex or mesh query capacity is below the "
                "heterogeneous cooked minimum"
            );
        }
        staged.capacities_.hardConvexPairs =
            requestedCapacities.hardConvexPairs == 0u
            ? staged.minimumCapacities_.hardConvexPairs
            : requestedCapacities.hardConvexPairs;
        staged.capacities_.meshTriangleCandidates =
            requestedCapacities.meshTriangleCandidates == 0u
            ? staged.minimumCapacities_
                  .meshTriangleCandidates
            : requestedCapacities.meshTriangleCandidates;
        const std::uint32_t solverTiles =
            requiredSolverTileCapacity(
                staged.capacities_.constraintBlocks,
                staged.model_.constraintProgram.blocks.size(),
                staged.capacities_.islands,
                staged.dynamicNodes_.size()
            );
        if (requestedCapacities.solverTiles != 0u &&
            requestedCapacities.solverTiles < solverTiles) {
            return rejectCompile(
                std::move(diagnostics),
                MetalWorldHostStatus::capacityOverflow,
                "solver-tile capacity is below the heterogeneous "
                "topology minimum"
            );
        }
        staged.minimumCapacities_.solverTiles = solverTiles;
        staged.capacities_.solverTiles =
            requestedCapacities.solverTiles == 0u
            ? solverTiles
            : requestedCapacities.solverTiles;
        const std::uint32_t spillRows =
            requiredSpillRowCapacity(
                staged.capacities_.constraintBlocks,
                staged.model_.constraintProgram.blocks.size()
            );
        if (requestedCapacities.spillRows != 0u &&
            requestedCapacities.spillRows < spillRows) {
            return rejectCompile(
                std::move(diagnostics),
                MetalWorldHostStatus::capacityOverflow,
                "spill-row capacity is below the heterogeneous "
                "topology minimum"
            );
        }
        staged.minimumCapacities_.spillRows = spillRows;
        staged.capacities_.spillRows =
            requestedCapacities.spillRows == 0u
            ? spillRows
            : requestedCapacities.spillRows;
        if (staged.dynamicNodes_.size() >
            MR_WORLD_MAX_DYNAMIC_NODES) {
            return rejectCompile(
                std::move(diagnostics),
                MetalWorldHostStatus::capacityOverflow,
                "heterogeneous dynamic-node count exceeds the "
                "configured Metal island bucket"
            );
        }

        std::uint64_t hash = staged.fingerprint_;
        hashValue(hash, world.fingerprint);
        hashVector(hash, staged.rodColliders_);
        hashVector(hash, staged.rodShapeSources_);
        hashVector(hash, staged.rodToolPairs_);
        hashVector(hash, staged.rodNodeOffsets_);
        hashVector(hash, staged.rodEdgeOffsets_);
        hashVector(hash, staged.defaultRodNodes_);
        hashVector(hash, staged.defaultRodEdges_);
        hashVector(hash, staged.dynamicNodes_);
        hashVector(hash, staged.rodDynamicNodes_);
        hashVector(
            hash,
            staged.model_.constraintProgram.blocks
        );
        hashVector(
            hash,
            staged.model_.constraintProgram.endpoints
        );
        hashVector(
            hash,
            staged.model_.constraintProgram.rows
        );
        hashVector(
            hash,
            staged.model_.constraintProgram.warmImpulses
        );
        hashValue(hash, staged.capacities_);
        staged.fingerprint_ = hash == 0u ? 1u : hash;
        diagnostics.fingerprint = staged.fingerprint_;
        diagnostics.status = MetalWorldHostStatus::success;
        diagnostics.message.clear();
        compiled = std::move(staged);
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return rejectCompile(
            std::move(diagnostics),
            MetalWorldHostStatus::metalBufferFailure,
            "host allocation failed while compiling heterogeneous world"
        );
    } catch (const std::exception& exception) {
        return rejectCompile(
            std::move(diagnostics),
            MetalWorldHostStatus::internalFailure,
            exception.what()
        );
    }
}

MetalWorldResidentState::MetalWorldResidentState() noexcept = default;

MetalWorldResidentState::~MetalWorldResidentState() = default;

MetalWorldResidentState::MetalWorldResidentState(
    MetalWorldResidentState&& other
) noexcept = default;

MetalWorldResidentState& MetalWorldResidentState::operator=(
    MetalWorldResidentState&& other
) noexcept = default;

bool MetalWorldResidentState::valid() const noexcept {
    if (state_ == nullptr) {
        return false;
    }
    try {
        const std::lock_guard lock(state_->mutex);
        return state_->initialized &&
            !state_->pending &&
            state_->context != nullptr &&
            state_->stateArenaGeneration ==
                state_->context->stateArenaGeneration;
    } catch (...) {
        return false;
    }
}

MetalWorldSubmission::MetalWorldSubmission() noexcept = default;

MetalWorldSubmission::~MetalWorldSubmission() = default;

MetalWorldSubmission::MetalWorldSubmission(
    MetalWorldSubmission&& other
) noexcept = default;

MetalWorldSubmission& MetalWorldSubmission::operator=(
    MetalWorldSubmission&& other
) noexcept = default;

bool MetalWorldSubmission::valid() const noexcept {
    return state_ != nullptr;
}

MetalWorldDiagnostics MetalWorldSubmission::wait(
    MetalWorldResult& result
) {
    if (state_ == nullptr) {
        return reject(
            {},
            MetalWorldHostStatus::metalCommandFailure,
            "MetalWorld submission is empty or already consumed"
        );
    }

    std::unique_ptr<detail::MetalWorldSubmissionState> pending =
        std::move(state_);
    MetalWorldDiagnostics diagnostics = pending->diagnostics;
    try {
        MetalWorldResult staged{};
        @autoreleasepool {
            [pending->commandBuffer waitUntilCompleted];
            const auto end = std::chrono::steady_clock::now();
            diagnostics.submissionElapsedMilliseconds =
                std::chrono::duration<double, std::milli>(
                    end - pending->start
                ).count();
            diagnostics.thermalState = thermalStateName(
                [NSProcessInfo processInfo].thermalState
            );
            if (pending->commandBuffer.status !=
                MTLCommandBufferStatusCompleted) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::metalCommandFailure,
                    "MetalWorld command failed: " +
                        describeError(
                            pending->commandBuffer.error
                        )
                );
            }
            const CFTimeInterval gpuStart =
                pending->commandBuffer.GPUStartTime;
            const CFTimeInterval gpuEnd =
                pending->commandBuffer.GPUEndTime;
            if (std::isfinite(gpuStart) &&
                std::isfinite(gpuEnd) &&
                gpuEnd >= gpuStart) {
                diagnostics.gpuElapsedMilliseconds =
                    1000.0 * (gpuEnd - gpuStart);
            }

            staged.layout = diagnostics.layout;
            if (pending->publishFinalState) {
                staged.finalQ.resize(
                    staged.layout.initialQElements
                );
                staged.finalV.resize(
                    staged.layout.initialVElements
                );
            }
            if (pending->publishStateTrajectory) {
                staged.observations.resize(
                    staged.layout.observationElements
                );
                staged.accelerations.resize(
                    staged.layout.accelerationElements
                );
            }
            staged.statuses.resize(
                staged.layout.statusElements
            );
            if (staged.layout.actionElements != 0u) {
                staged.actorObservations.resize(
                    staged.layout.actorObservationElements
                );
                staged.criticObservations.resize(
                    staged.layout.criticObservationElements
                );
                staged.transitions.resize(
                    staged.layout.transitionElements
                );
                staged.motionFeatures.resize(
                    staged.layout.motionFeatureElements
                );
                staged.teacherActions.resize(
                    staged.layout.teacherActionElements
                );
                staged.policyLatents.resize(
                    staged.layout.policyLatentElements
                );
                staged.policyLogProbabilities.resize(
                    staged.layout
                        .policyLogProbabilityElements
                );
                staged.policyValues.resize(
                    staged.layout.policyValueElements
                );
            }
            if (pending->hasRods) {
                if (pending->publishFinalState) {
                    staged.finalRodNodes.resize(
                        staged.layout.rodNodeStateElements
                    );
                    staged.finalRodEdges.resize(
                        staged.layout.rodEdgeStateElements
                    );
                }
            }
            if (pending->contactMode) {
                if (pending->publishFinalState) {
                    staged.finalSceneBodies.resize(
                        staged.layout.initialSceneBodyElements
                    );
                }
                staged.contactStatuses.resize(
                    staged.layout.contactStatusElements
                );
                if (staged.layout.contactDispatch.solverType ==
                    MR_SOLVER_QUALITY_NEWTON) {
                    staged.qualityStatuses.resize(
                        staged.layout.dispatch.environmentCount
                    );
                }
                if (pending->captureContactEvidence) {
                    auto& evidence = staged.contactEvidence;
                    evidence.manifoldHeaders.resize(
                        staged.layout.manifoldHeaderElements
                    );
                    evidence.manifoldPoints.resize(
                        staged.layout.manifoldPointElements
                    );
                    evidence.manifoldCounts.resize(
                        staged.layout.dispatch.environmentCount
                    );
                    evidence.contacts.resize(
                        staged.layout.contactConstraintElements
                    );
                    evidence.contactMetadata.resize(
                        staged.layout.contactConstraintElements
                    );
                    evidence.blocks.resize(
                        staged.layout.contactConstraintElements
                    );
                    evidence.endpoints.resize(
                        2u *
                            staged.layout
                                .contactConstraintElements
                    );
                    evidence.endpointRuntime.resize(
                        staged.layout.endpointRuntimeElements
                    );
                    evidence.rows.resize(
                        staged.layout.constraintRowElements
                    );
                    evidence.cones.resize(
                        staged.layout.contactConstraintElements
                    );
                    evidence.evaluatedRows.resize(
                        staged.layout.constraintRowElements
                    );
                    evidence.evaluatedCones.resize(
                        staged.layout.contactConstraintElements
                    );
                    evidence.islands.resize(
                        staged.layout.islandElements
                    );
                    evidence.islandNodes.resize(
                        staged.layout
                            .islandNodeReferenceElements
                    );
                    evidence.islandConstraints.resize(
                        staged.layout
                            .islandConstraintReferenceElements
                    );
                }
            }
            const auto& buffers = pending->context->buffers;
            const auto stateOutputBuffer =
                [&](const std::size_t index) -> id<MTLBuffer> {
                    return buffers[index].storageMode ==
                            MTLStorageModePrivate
                        ? pending->context
                              ->readbackBuffers[index]
                        : buffers[index];
                };
            if (pending->publishFinalState) {
                copyOutput(
                    staged.finalQ,
                    stateOutputBuffer(pending->finalQBuffer)
                );
                copyOutput(
                    staged.finalV,
                    stateOutputBuffer(pending->finalVBuffer)
                );
            }
            if (pending->publishStateTrajectory) {
                copyOutput(
                    staged.observations,
                    buffers[kObservations]
                );
                copyOutput(
                    staged.accelerations,
                    buffers[kAccelerationTrajectory]
                );
            }
            copyOutput(
                staged.statuses,
                buffers[kPublicStatuses]
            );
            if (pending->nativeTask) {
                id<MTLBuffer> evidenceBuffer =
                    stateOutputBuffer(kTaskEvidenceState);
                if (evidenceBuffer == nil ||
                    evidenceBuffer.contents == nullptr) {
                    return reject(
                        std::move(diagnostics),
                        MetalWorldHostStatus::metalBufferFailure,
                        "task evidence readback is unavailable"
                    );
                }
                std::memcpy(
                    &staged.evidenceState,
                    evidenceBuffer.contents,
                    sizeof(staged.evidenceState)
                );
            }
            if (staged.layout.actionElements != 0u) {
                copyOutput(
                    staged.actorObservations,
                    buffers[kTaskActorObservations]
                );
                copyOutput(
                    staged.criticObservations,
                    buffers[kTaskCriticObservations]
                );
                copyOutput(
                    staged.transitions,
                    buffers[kTaskTransitions]
                );
                copyOutput(
                    staged.motionFeatures,
                    buffers[kTaskMotionFeatures]
                );
                copyOutput(
                    staged.teacherActions,
                    buffers[kTaskTeacherActions]
                );
                if (staged.layout.policyLatentElements != 0u) {
                    copyOutput(
                        staged.policyLatents,
                        buffers[kPolicyLatents]
                    );
                    copyOutput(
                        staged.policyLogProbabilities,
                        buffers[kPolicyLogProbabilities]
                    );
                    copyOutput(
                        staged.policyValues,
                        buffers[kPolicyValues]
                    );
                }
            }
            if (pending->hasRods &&
                pending->publishFinalState) {
                copyOutput(
                    staged.finalRodNodes,
                    stateOutputBuffer(
                        pending->finalRodNodeBuffer
                    )
                );
                copyOutput(
                    staged.finalRodEdges,
                    stateOutputBuffer(
                        pending->finalRodEdgeBuffer
                    )
                );
            }
            if (pending->contactMode) {
                if (pending->publishFinalState) {
                    copyOutput(
                        staged.finalSceneBodies,
                        stateOutputBuffer(
                            pending->finalSceneBuffer
                        )
                    );
                }
                copyOutput(
                    staged.contactStatuses,
                    buffers[kPublicContactStatuses]
                );
                copyOutput(
                    staged.qualityStatuses,
                    buffers[kQualityStatuses]
                );
                if (pending->captureContactEvidence) {
                    auto& evidence = staged.contactEvidence;
                    copyOutput(
                        evidence.manifoldHeaders,
                        stateOutputBuffer(
                            pending->finalManifoldHeaderBuffer
                        )
                    );
                    copyOutput(
                        evidence.manifoldPoints,
                        stateOutputBuffer(
                            pending->finalManifoldPointBuffer
                        )
                    );
                    copyOutput(
                        evidence.manifoldCounts,
                        stateOutputBuffer(
                            pending->finalManifoldCountBuffer
                        )
                    );
                    copyOutput(
                        evidence.contacts,
                        stateOutputBuffer(kContacts)
                    );
                    copyOutput(
                        evidence.contactMetadata,
                        stateOutputBuffer(kContactMetadata)
                    );
                    copyOutput(
                        evidence.blocks,
                        stateOutputBuffer(kIRBlocks)
                    );
                    copyOutput(
                        evidence.endpoints,
                        stateOutputBuffer(kIREndpoints)
                    );
                    copyOutput(
                        evidence.endpointRuntime,
                        stateOutputBuffer(kEndpointRuntime)
                    );
                    copyOutput(
                        evidence.rows,
                        stateOutputBuffer(kIRRows)
                    );
                    copyOutput(
                        evidence.cones,
                        stateOutputBuffer(kIRCones)
                    );
                    copyOutput(
                        evidence.evaluatedRows,
                        stateOutputBuffer(kEvaluatedRows)
                    );
                    copyOutput(
                        evidence.evaluatedCones,
                        stateOutputBuffer(kEvaluatedCones)
                    );
                    copyOutput(
                        evidence.islands,
                        stateOutputBuffer(kIslands)
                    );
                    copyOutput(
                        evidence.islandNodes,
                        stateOutputBuffer(
                            kIslandNodeReferences
                        )
                    );
                    copyOutput(
                        evidence.islandConstraints,
                        stateOutputBuffer(
                            kIslandConstraintReferences
                        )
                    );
                }
            }
        }

        return validateAndPublish(
            std::move(staged),
            std::move(diagnostics),
            pending->publishFinalState,
            pending->publishStateTrajectory,
            pending->policyRevision,
            result
        );
    } catch (const std::bad_alloc&) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalBufferFailure,
            "host allocation failed while publishing MetalWorld results"
        );
    } catch (const std::exception& exception) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::internalFailure,
            exception.what()
        );
    }
}

MetalWorldContext::MetalWorldContext(MetalWorldConfig config)
    : pool_(std::make_shared<detail::MetalWorldContextPool>()) {
    const std::uint32_t slotCount = std::clamp(
        config.maximumInFlightSubmissions,
        1u,
        8u
    );
    pool_->slots.reserve(slotCount);
    for (std::uint32_t slot = 0u; slot < slotCount; ++slot) {
        pool_->slots.push_back(
            std::make_shared<detail::MetalWorldContextState>(
                config
            )
        );
    }
}

MetalWorldContext::~MetalWorldContext() = default;

MetalWorldContext::MetalWorldContext(
    MetalWorldContext&& other
) noexcept = default;

MetalWorldContext& MetalWorldContext::operator=(
    MetalWorldContext&& other
) noexcept = default;

MetalWorldDiagnostics MetalWorldContext::submit(
    const CompiledWorld& world,
    const MetalWorldBatch& batch,
    const MetalWorldStepConfig& config,
    MetalWorldSubmission& submission
) {
    return submitImpl(
        world,
        batch,
        config,
        nullptr,
        false,
        submission
    );
}

MetalWorldDiagnostics MetalWorldContext::initializeResidentState(
    const CompiledWorld& world,
    const MetalWorldBatch& batch,
    const MetalWorldStepConfig& config,
    MetalWorldResidentState& state,
    MetalWorldSubmission& submission
) {
    return submitImpl(
        world,
        batch,
        config,
        &state,
        true,
        submission
    );
}

MetalWorldDiagnostics MetalWorldContext::submitResident(
    const CompiledWorld& world,
    const MetalWorldBatch& batch,
    const MetalWorldStepConfig& config,
    MetalWorldResidentState& state,
    MetalWorldSubmission& submission
) {
    return submitImpl(
        world,
        batch,
        config,
        &state,
        false,
        submission
    );
}

MetalWorldDiagnostics MetalWorldContext::submitImpl(
    const CompiledWorld& world,
    const MetalWorldBatch& batch,
    const MetalWorldStepConfig& config,
    MetalWorldResidentState* residentState,
    const bool initializeResidentState,
    MetalWorldSubmission& submission
) {
    MetalWorldDiagnostics diagnostics{};
    if (pool_ == nullptr) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::internalFailure,
            "MetalWorld context was moved from"
        );
    }
    if (submission.valid()) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::contextBusy,
            "submission output already owns a live MetalWorld batch"
        );
    }
    if ((residentState == nullptr) &&
        initializeResidentState) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::internalFailure,
            "resident initialization is missing its ownership token"
        );
    }

    RequiredBuffers requirements{};
    try {
        const bool residentContinuation =
            residentState != nullptr &&
            !initializeResidentState;
        std::shared_ptr<detail::MetalWorldResidentStateData>
            residentData;
        std::shared_ptr<detail::MetalWorldContextState>
            selectedState;
        std::size_t residentQ = kStateQA;
        std::size_t residentV = kStateVA;
        std::size_t residentScene = kSceneBodiesA;
        std::size_t residentManifoldHeaders =
            kManifoldHeadersA;
        std::size_t residentManifoldPoints =
            kManifoldPointsA;
        std::size_t residentManifoldCounts =
            kManifoldCountsA;
        std::size_t residentRodNodes = kRodNodesA;
        std::size_t residentRodEdges = kRodEdgesA;
        std::size_t residentRodWitnesses =
            kRodWitnessesA;
        if (initializeResidentState &&
            residentState->state_ != nullptr) {
            return reject(
                std::move(diagnostics),
                MetalWorldHostStatus::contextBusy,
                "resident state token is already initialized or pending"
            );
        }
        if (residentContinuation) {
            residentData = residentState->state_;
            if (residentData == nullptr) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::invalidDimensions,
                    "resident state token is empty"
                );
            }
            const std::lock_guard residentLock(
                residentData->mutex
            );
            if (residentData->ownerPool.lock() != pool_) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::invalidDimensions,
                    "resident state belongs to another MetalWorld "
                    "context"
                );
            }
            if (!residentData->initialized ||
                residentData->pending ||
                residentData->context == nullptr) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::contextBusy,
                    "resident state is not ready for continuation"
                );
            }
            if (residentData->worldFingerprint !=
                    world.fingerprint() ||
                residentData->taskFingerprint !=
                    config.taskProgram.fingerprint() ||
                residentData->taskSeed != config.taskSeed ||
                residentData->environmentCount !=
                    batch.environmentCount) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::invalidDimensions,
                    "resident world, task, seed, or environment count "
                    "changed"
                );
            }
            if (!batch.resetMasks.empty() &&
                !residentData->supportsReset) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::invalidReset,
                    "resident state was initialized without a reset "
                    "contract"
                );
            }
            selectedState = residentData->context;
            residentQ = residentData->qBuffer;
            residentV = residentData->vBuffer;
            residentScene = residentData->sceneBuffer;
            residentManifoldHeaders =
                residentData->manifoldHeaderBuffer;
            residentManifoldPoints =
                residentData->manifoldPointBuffer;
            residentManifoldCounts =
                residentData->manifoldCountBuffer;
            residentRodNodes =
                residentData->rodNodeBuffer;
            residentRodEdges =
                residentData->rodEdgeBuffer;
            residentRodWitnesses =
                residentData->rodWitnessBuffer;
            residentData->pending = true;
        }
        auto residentReservation =
            residentContinuation
            ? std::make_unique<
                  detail::MetalWorldResidentReservation
              >(residentData)
            : nullptr;
        diagnostics = validateAndBuildLayout(
            world,
            batch,
            config,
            residentContinuation,
            requirements
        );
        if (!diagnostics.succeeded()) {
            return diagnostics;
        }
        if (residentContinuation) {
            const std::lock_guard slotLock(
                selectedState->mutex
            );
            if (selectedState->inFlight) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::contextBusy,
                    "resident MetalWorld arena is in flight"
                );
            }
            selectedState->inFlight = true;
            selectedState->stats.hasInFlightSubmission = true;
        } else {
            const std::lock_guard poolLock(pool_->mutex);
            const std::size_t slotCount = pool_->slots.size();
            for (std::size_t offset = 0u;
                 offset < slotCount;
                 ++offset) {
                const std::size_t slot =
                    (pool_->nextSlot + offset) % slotCount;
                const auto& candidate = pool_->slots[slot];
                const std::lock_guard slotLock(
                    candidate->mutex
                );
                if (candidate->inFlight ||
                    !candidate->residentOwner.expired()) {
                    continue;
                }
                candidate->inFlight = true;
                candidate->stats.hasInFlightSubmission = true;
                selectedState = candidate;
                // Reuse the warm slot for sequential runs. If it is still
                // busy, the same ordered search naturally advances through
                // the remaining ring slots without waiting.
                pool_->nextSlot = slot;
                break;
            }
            if (selectedState != nullptr &&
                initializeResidentState) {
                residentData = std::make_shared<
                    detail::MetalWorldResidentStateData
                >();
                residentData->ownerPool = pool_;
                residentData->context = selectedState;
                residentData->worldFingerprint =
                    world.fingerprint();
                residentData->taskFingerprint =
                    config.taskProgram.fingerprint();
                residentData->taskSeed = config.taskSeed;
                residentData->environmentCount =
                    batch.environmentCount;
                residentData->supportsReset =
                    !batch.resetMasks.empty();
                residentData->pending = true;
                {
                    const std::lock_guard slotLock(
                        selectedState->mutex
                    );
                    selectedState->residentOwner =
                        residentData;
                }
                residentReservation = std::make_unique<
                    detail::MetalWorldResidentReservation
                >(residentData);
            }
        }
        if (selectedState == nullptr) {
            return reject(
                std::move(diagnostics),
                MetalWorldHostStatus::contextBusy,
                "all MetalWorld asynchronous arena slots are in flight"
            );
        }
        detail::MetalWorldSlotReservation reservation(
            selectedState
        );
        const std::lock_guard lock(selectedState->mutex);

        @autoreleasepool {
            diagnostics = initializeContext(
                *selectedState,
                std::move(diagnostics)
            );
            if (!diagnostics.succeeded()) {
                return diagnostics;
            }
            diagnostics = ensureBufferArena(
                *selectedState,
                requirements,
                std::move(diagnostics)
            );
            if (!diagnostics.succeeded()) {
                return diagnostics;
            }
            if (residentContinuation &&
                (
                    residentData->stateArenaGeneration !=
                        selectedState->stateArenaGeneration ||
                    selectedState->boundModelFingerprint !=
                        world.fingerprint()
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::invalidDimensions,
                    "resident state arena or compiled world changed"
                );
            }
            const bool contactMode =
                config.solverMode !=
                    MetalWorldSolverMode::freeMotionABA;
            const bool uploadInitialState =
                !residentContinuation;
            const bool nativeTask = config.taskProgram.valid();
            const bool taskTracksImpactContacts =
                nativeTask &&
                !config.taskProgram.impactEvents().empty();
            const bool taskHasThreatTeacher =
                nativeTask &&
                (config.taskProgram.header().schedule.w &
                 MR_TASK_PROGRAM_THREAT_TEACHER) != 0u;
            const bool taskHasMotionPrior =
                nativeTask &&
                config.taskProgram.layout().motionFeatureCount != 0u;
            selectedState->useTaskBodyParameters = nativeTask;
            selectedState->stats.memoryPlan =
                diagnostics.layout.memoryPlan;
            MTLCommandBufferDescriptor* commandDescriptor =
                [MTLCommandBufferDescriptor new];
            commandDescriptor.errorOptions =
                MTLCommandBufferErrorOptionEncoderExecutionStatus;
            id<MTLCommandBuffer> commandBuffer =
                [selectedState->queue
                    commandBufferWithDescriptor:commandDescriptor];
            if (commandBuffer == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::metalCommandFailure,
                    "failed to create MetalWorld command buffer"
                );
            }
            commandBuffer.label =
                @"MetalRobo persistent batched world graph";
            uploadBatch(
                *selectedState,
                commandBuffer,
                world,
                batch,
                config,
                diagnostics.layout,
                requirements,
                uploadInitialState
            );
            if (uploadInitialState &&
                !encodeResidentStateInitialization(
                    *selectedState,
                    commandBuffer,
                    requirements,
                    contactMode,
                    world.rodCount() != 0u,
                    nativeTask
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::metalCommandFailure,
                    "failed to initialize resident MetalWorld state"
                );
            }

            id<MTLComputePipelineState> selectedABAPipeline =
                nativeTask
                ? selectedState->parameterizedABAPipeline
                : world.articulationCount() > 1u
                ? selectedState->multiABAPipeline
                : world.capacityClass() ==
                    MetalWorldCapacityClass::compactABA12
                ? selectedState->smallABAPipeline
                : selectedState->abaPipeline;
            std::size_t sourceQ =
                residentContinuation ? residentQ : kStateQA;
            std::size_t sourceV =
                residentContinuation ? residentV : kStateVA;
            std::size_t destinationQ =
                sourceQ == kStateQA ? kStateQB : kStateQA;
            std::size_t destinationV =
                sourceV == kStateVA ? kStateVB : kStateVA;
            std::size_t sourceMotorState =
                sourceQ == kStateQA
                ? kMulticopterStateA
                : kMulticopterStateB;
            std::size_t destinationMotorState =
                sourceMotorState == kMulticopterStateA
                ? kMulticopterStateB
                : kMulticopterStateA;
            mr_u32 activePairClassMask = 0u;
            for (const MRCompiledCollisionPairGPU& pair :
                 world.eligiblePairs()) {
                mr_u32 workClass = MR_WORLD_WORK_ANALYTIC;
                if (pair.pairClass ==
                    MR_COLLISION_PAIR_BOX_BOX) {
                    workClass = MR_WORLD_WORK_SAT_CLIP;
                } else if (
                    pair.pairClass == MR_COLLISION_PAIR_MESH
                ) {
                    workClass = MR_WORLD_WORK_MESH;
                } else if (
                    pair.pairClass == MR_COLLISION_PAIR_CONVEX
                ) {
                    const auto& shapes = world.model().shapes;
                    workClass =
                        shapes[pair.colliderA].shapeType ==
                                MR_SHAPE_CONVEX ||
                            shapes[pair.colliderB].shapeType ==
                                MR_SHAPE_CONVEX
                        ? MR_WORLD_WORK_HULL_GJK
                        : MR_WORLD_WORK_PRIMITIVE_GJK;
                }
                activePairClassMask |= 1u << workClass;
            }
            std::size_t sourceScene =
                residentContinuation
                ? residentScene
                : kSceneBodiesA;
            std::size_t destinationScene =
                sourceScene == kSceneBodiesA
                ? kSceneBodiesB
                : kSceneBodiesA;
            std::size_t sourceManifoldHeaders =
                residentContinuation
                ? residentManifoldHeaders
                : kManifoldHeadersA;
            std::size_t sourceManifoldPoints =
                residentContinuation
                ? residentManifoldPoints
                : kManifoldPointsA;
            std::size_t sourceManifoldCounts =
                residentContinuation
                ? residentManifoldCounts
                : kManifoldCountsA;
            std::size_t destinationManifoldHeaders =
                sourceManifoldHeaders == kManifoldHeadersA
                ? kManifoldHeadersB
                : kManifoldHeadersA;
            std::size_t destinationManifoldPoints =
                sourceManifoldPoints == kManifoldPointsA
                ? kManifoldPointsB
                : kManifoldPointsA;
            std::size_t destinationManifoldCounts =
                sourceManifoldCounts == kManifoldCountsA
                ? kManifoldCountsB
                : kManifoldCountsA;
            std::size_t sourceRodNodes =
                residentContinuation
                ? residentRodNodes
                : kRodNodesA;
            std::size_t sourceRodEdges =
                residentContinuation
                ? residentRodEdges
                : kRodEdgesA;
            std::size_t destinationRodNodes =
                sourceRodNodes == kRodNodesA
                ? kRodNodesB
                : kRodNodesA;
            std::size_t destinationRodEdges =
                sourceRodEdges == kRodEdgesA
                ? kRodEdgesB
                : kRodEdgesA;
            std::size_t sourceRodWitnesses =
                residentContinuation
                ? residentRodWitnesses
                : kRodWitnessesA;
            std::size_t destinationRodWitnesses =
                sourceRodWitnesses == kRodWitnessesA
                ? kRodWitnessesB
                : kRodWitnessesA;
            const std::size_t rodWitnessCount =
                requirements.entries[kRodWitnessesA]
                    .logicalElements;
            for (std::uint32_t controlStep = 0u;
                 controlStep <
                     diagnostics.layout.dispatch.controlStepCount;
                 ++controlStep) {
                MRMetalWorldPassGPU pass{};
                pass.controlStep = controlStep;
                pass.physicsSubstep = MR_INVALID_INDEX;
                if (nativeTask) {
                    if (!encodeTaskObserve(
                            *selectedState,
                            commandBuffer,
                            pass,
                            sourceQ,
                            sourceV,
                            sourceScene,
                            batch.environmentCount
                        )) {
                        return reject(
                            std::move(diagnostics),
                            MetalWorldHostStatus::metalCommandFailure,
                            "failed to encode native observation pass"
                        );
                    }
                    const bool needsObservationBodies =
                        config.deviceObservationProgram.valid() ||
                        taskHasThreatTeacher ||
                        taskHasMotionPrior;
                    if (needsObservationBodies &&
                        !encodeDeviceObservationBodies(
                                *selectedState,
                                commandBuffer,
                                pass,
                                sourceQ,
                                destinationQ,
                                sourceScene,
                                destinationScene,
                                batch.environmentCount
                            )) {
                            return reject(
                                std::move(diagnostics),
                                MetalWorldHostStatus::metalCommandFailure,
                                "failed to encode observation-time body projection"
                            );
                    }
                    if (taskHasThreatTeacher &&
                        (!encodeTaskThreatSelect(
                             *selectedState,
                             commandBuffer,
                             pass,
                             batch.environmentCount
                         ) ||
                         !encodeTaskThreatJacobians(
                             *selectedState,
                             commandBuffer,
                             sourceQ,
                             batch.environmentCount
                         ))) {
                        return reject(
                            std::move(diagnostics),
                            MetalWorldHostStatus::metalCommandFailure,
                            "failed to encode predictive threat/Jacobian pass"
                        );
                    }
                    if (taskHasMotionPrior &&
                        !encodeTaskMotionFeatures(
                            *selectedState,
                            commandBuffer,
                            pass,
                            batch.environmentCount
                        )) {
                        return reject(
                            std::move(diagnostics),
                            MetalWorldHostStatus::metalCommandFailure,
                            "failed to encode tracked-link motion features"
                        );
                    }
                    if (config.deviceObservationProgram.valid()) {
                        const TaskProgramLayout& taskLayout =
                            config.taskProgram.layout();
                        const MetalWorldDeviceObservationPass observation{
                            .commandBuffer = (__bridge void*)commandBuffer,
                            .q = (__bridge void*)selectedState->buffers[sourceQ],
                            .v = (__bridge void*)selectedState->buffers[sourceV],
                            .sceneBodies = (__bridge void*)selectedState->buffers[sourceScene],
                            .currentBodies = (__bridge void*)selectedState->buffers[kCurrentBodies],
                            .resetMasks = (__bridge void*)selectedState->buffers[kResetMasks],
                            .taskStates = (__bridge void*)selectedState->buffers[kTaskState],
                            .actorHistory = (__bridge void*)selectedState->buffers[kTaskActorHistory],
                            .actorObservations = (__bridge void*)selectedState->buffers[kTaskActorObservations],
                            .actorObservationOffsetElements =
                                static_cast<std::size_t>(controlStep) *
                                batch.environmentCount *
                                taskLayout.actorObservationSize,
                            .seed = config.taskSeed,
                            .controlStep = controlStep,
                            .environmentCount = static_cast<std::uint32_t>(batch.environmentCount),
                            .bodyCount = static_cast<std::uint32_t>(world.model().bodies.size()),
                            .sceneBodyCount = static_cast<std::uint32_t>(world.sceneBodyCount()),
                            .nq = diagnostics.layout.dispatch.nq,
                            .nv = diagnostics.layout.dispatch.nv,
                            .actorFrameSize = taskLayout.actorFrameSize,
                            .actorHistoryLength = taskLayout.actorHistoryLength,
                            .actorObservationSize = taskLayout.actorObservationSize,
                        };
                        if (!config.deviceObservationProgram.encode(
                                config.deviceObservationProgram.context,
                                observation
                            )) {
                            return reject(
                                std::move(diagnostics),
                                MetalWorldHostStatus::metalCommandFailure,
                                "device observation program rejected the rollout pass"
                            );
                        }
                    }
                    if (config.policyProgram.valid() &&
                        !encodePolicyInference(
                                *selectedState,
                                commandBuffer,
                                config.policyProgram,
                                pass,
                                batch.environmentCount
                            )) {
                        return reject(
                            std::move(diagnostics),
                            MetalWorldHostStatus::metalCommandFailure,
                            "failed to encode native policy inference pass"
                        );
                    }
                    if (taskHasThreatTeacher &&
                        !encodeTaskJointCbf(
                            *selectedState,
                            commandBuffer,
                            pass,
                            sourceQ,
                            sourceV,
                            batch.environmentCount
                        )) {
                        return reject(
                            std::move(diagnostics),
                            MetalWorldHostStatus::metalCommandFailure,
                            "failed to encode privileged Joint-CBF teacher"
                        );
                    }
                    if (!encodeTaskApplyActions(
                            *selectedState,
                            commandBuffer,
                            pass,
                            batch.environmentCount
                        )) {
                        return reject(
                            std::move(diagnostics),
                            MetalWorldHostStatus::metalCommandFailure,
                            "failed to encode native action pass"
                        );
                    }
                }
                if (!encodePrepare(
                        *selectedState,
                        commandBuffer,
                        pass,
                        sourceQ,
                        sourceV,
                        batch.environmentCount
                     )) {
                    return reject(
                        std::move(diagnostics),
                        MetalWorldHostStatus::metalCommandFailure,
                        "failed to encode MetalWorld prepare pass"
                    );
                }
                if (contactMode &&
                    !encodeContactControlPrepare(
                        *selectedState,
                        commandBuffer,
                        pass,
                        sourceScene,
                        sourceManifoldHeaders,
                        sourceManifoldPoints,
                        sourceManifoldCounts,
                        batch.environmentCount
                    )) {
                    return reject(
                        std::move(diagnostics),
                        MetalWorldHostStatus::metalCommandFailure,
                        "failed to encode contact checkpoint/reset pass"
                    );
                }
                if (world.rodCount() != 0u &&
                    !encodeRodControlPrepare(
                        *selectedState,
                        commandBuffer,
                        pass,
                        sourceRodNodes,
                        sourceRodEdges,
                        sourceRodWitnesses,
                        rodWitnessCount,
                        batch.environmentCount
                    )) {
                    return reject(
                        std::move(diagnostics),
                        MetalWorldHostStatus::metalCommandFailure,
                        "failed to encode rod checkpoint/reset pass"
                    );
                }

                for (std::uint32_t physicsSubstep = 0u;
                     physicsSubstep < config.physicsSubsteps;
                     ++physicsSubstep) {
                    pass.physicsSubstep = physicsSubstep;
                    const bool useHybridContact =
                        contactMode &&
                        config.ccdMode ==
                            MetalWorldCCDMode::hybrid;
                    const bool encodedABA =
                        (
                            config.actuationMode !=
                                MetalWorldActuationMode::
                                    implicitPositionDrive ||
                            encodeDriveRefresh(
                                *selectedState,
                                commandBuffer,
                                pass,
                                sourceQ,
                                sourceV,
                                batch.environmentCount
                            )
                        ) &&
                        (
                            !nativeTask ||
                            encodeTaskNativeActuators(
                                *selectedState,
                                commandBuffer,
                                pass,
                                sourceQ,
                                sourceV,
                                batch.environmentCount
                            )
                        ) &&
                        (
                            !config.multicopterProgram.valid() ||
                            encodeMulticopterActuation(
                                *selectedState,
                                commandBuffer,
                                pass,
                                sourceQ,
                                sourceV,
                                sourceMotorState,
                                batch.environmentCount
                            )
                        ) &&
                        (
                            !config.flappingWingProgram.valid() ||
                            encodeFlappingWingActuation(
                                *selectedState,
                                commandBuffer,
                                sourceQ,
                                sourceV,
                                batch.environmentCount
                            )
                        ) &&
                        (
                            useHybridContact ||
                            encodeABA(
                                *selectedState,
                                commandBuffer,
                                selectedABAPipeline,
                                sourceQ,
                                sourceV,
                                world.articulationCount(),
                                batch.environmentCount
                            )
                        );
                    const bool encodedRod =
                        encodedABA &&
                        (
                            useHybridContact ||
                            encodeRodSubstep(
                                *selectedState,
                                commandBuffer,
                                pass,
                                world,
                                sourceRodNodes,
                                sourceRodEdges,
                                destinationRodNodes,
                                destinationRodEdges,
                                kCCDEventStatesA,
                                MR_CCD_SEGMENT_FULL_MICROSTEP,
                                batch.environmentCount
                            )
                        );
                    const bool encodedPublication =
                        encodedRod &&
                        (
                        useHybridContact
	                        ? encodeHybridContactSubstep(
	                              *selectedState,
	                              commandBuffer,
	                              selectedABAPipeline,
	                              world,
	                              pass,
                              physicsSubstep + 1u ==
                                  config.physicsSubsteps,
                              sourceQ,
                              sourceV,
                              destinationQ,
                              destinationV,
                              sourceScene,
                              destinationScene,
                              sourceManifoldHeaders,
                              sourceManifoldPoints,
                              sourceManifoldCounts,
	                              destinationManifoldHeaders,
	                              destinationManifoldPoints,
	                              destinationManifoldCounts,
	                              sourceRodNodes,
	                              sourceRodEdges,
	                              destinationRodNodes,
	                              destinationRodEdges,
	                              physicsSubstep == 0u
	                                  ? kCheckpointRodWitnesses
	                                  : sourceRodWitnesses,
	                              rodWitnessCount,
                              config.solverMode ==
                                  MetalWorldSolverMode::
                                      temporalCone,
                              activePairClassMask,
                              config.velocityIterations +
                                  (
                                      physicsSubstep + 1u ==
                                              config.physicsSubsteps
                                      ? config.finalVelocityIterations
                                      : 0u
                                  ),
                              diagnostics.layout.contactDispatch
                                      .constraintCapacity > 256u,
                              config.maxCCDAdvanceSolvePasses,
                              world.articulationCount(),
                              batch.environmentCount,
                              diagnostics.layout
                                  .islandWorkElements,
                              diagnostics.layout
                                  .contactTileElements,
                              requirements.entries[
                                  kProjectedColliders
                              ].logicalElements,
                              std::max<std::size_t>(
                                  requirements.entries[
                                      kPairOverlapFlags
                                  ].logicalElements,
                                  1u
                              )
                          )
                        : contactMode
                        ? encodeContactSubstep(
                              *selectedState,
                              commandBuffer,
                              world,
                              pass,
                              physicsSubstep + 1u ==
                                  config.physicsSubsteps,
                              sourceQ,
                              destinationQ,
                              destinationV,
                              sourceScene,
                              destinationScene,
                              sourceManifoldHeaders,
                              sourceManifoldPoints,
                              sourceManifoldCounts,
                              physicsSubstep == 0u
                                  ? kCheckpointRodWitnesses
                                  : sourceRodWitnesses,
                              destinationRodNodes,
                              destinationRodEdges,
                              rodWitnessCount,
                              destinationManifoldHeaders,
                              destinationManifoldPoints,
                              destinationManifoldCounts,
                              config.solverMode ==
                                  MetalWorldSolverMode::
                                      temporalCone,
                              config.solverMode ==
                                  MetalWorldSolverMode::
                                      qualityNewton,
                              activePairClassMask,
                              (
                                  diagnostics.layout
                                      .contactDispatch.flags &
                                  MR_METAL_WORLD_CONTACT_HAS_FUTURE_KINEMATICS
                              ) != 0u,
                              false,
                              config.velocityIterations +
                                  (
                                      physicsSubstep + 1u ==
                                              config.physicsSubsteps
                                      ? config.finalVelocityIterations
                                      : 0u
                                  ),
                              diagnostics.layout.contactDispatch
                                      .constraintCapacity > 256u,
                              batch.environmentCount,
                              diagnostics.layout
                                  .islandWorkElements,
                              diagnostics.layout
                                  .contactTileElements,
                              requirements.entries[
                                  kProjectedColliders
                              ].logicalElements,
                              std::max<std::size_t>(
                                  requirements.entries[
                                      kPairOverlapFlags
                                  ].logicalElements,
                                  1u
                              )
                          )
                        : encodeCommit(
                              *selectedState,
                              commandBuffer,
                              pass,
                              destinationQ,
                              destinationV,
                              batch.environmentCount
                          )
                        );
                    const bool encodedRodPublication =
                        encodedPublication &&
                        (
                            world.rodCount() == 0u ||
                            encodeRodCommit(
                                *selectedState,
                                commandBuffer,
                                pass,
                                destinationRodNodes,
                                destinationRodEdges,
                                destinationRodNodes,
                                destinationRodEdges,
                                destinationRodWitnesses,
                                rodWitnessCount,
                                batch.environmentCount
                            )
                        );
                    const bool encodedTaskImpactContact =
                        encodedRodPublication &&
                        (
                            !config.multicopterProgram.valid() ||
                            encodeMulticopterCommit(
                                *selectedState,
                                commandBuffer,
                                sourceMotorState,
                                destinationMotorState,
                                batch.environmentCount
                            )
                        ) &&
                        (
                            !taskTracksImpactContacts ||
                            !contactMode ||
                            encodeTaskImpactContact(
                                *selectedState,
                                commandBuffer,
                                pass,
                                batch.environmentCount
                            )
                        );
                    if (!encodedABA ||
                        !encodedRod ||
                        !encodedPublication ||
                        !encodedRodPublication ||
                        !encodedTaskImpactContact) {
                        return reject(
                            std::move(diagnostics),
                            MetalWorldHostStatus::metalCommandFailure,
                            "failed to encode MetalWorld substep graph"
                        );
                    }
                    std::swap(sourceQ, destinationQ);
                    std::swap(sourceV, destinationV);
                    if (config.multicopterProgram.valid()) {
                        std::swap(
                            sourceMotorState,
                            destinationMotorState
                        );
                    }
                    if (world.rodCount() != 0u) {
                        std::swap(
                            sourceRodNodes,
                            destinationRodNodes
                        );
                        std::swap(
                            sourceRodEdges,
                            destinationRodEdges
                        );
                        std::swap(
                            sourceRodWitnesses,
                            destinationRodWitnesses
                        );
                    }
                    if (contactMode) {
                        std::swap(sourceScene, destinationScene);
                        std::swap(
                            sourceManifoldHeaders,
                            destinationManifoldHeaders
                        );
                        std::swap(
                            sourceManifoldPoints,
                            destinationManifoldPoints
                        );
                        std::swap(
                            sourceManifoldCounts,
                            destinationManifoldCounts
                        );
                    }
                }

                pass.physicsSubstep = MR_INVALID_INDEX;
                if ((nativeTask &&
                     (
                         !encodeTaskEffort(
                             *selectedState,
                             commandBuffer,
                             pass,
                             sourceV,
                             batch.environmentCount
                         ) ||
                         !encodeTaskComplete(
                             *selectedState,
                             commandBuffer,
                             pass,
                             sourceQ,
                             sourceV,
                             sourceScene,
                             batch.environmentCount
                         ) ||
                         !encodeTaskEvidence(
                             *selectedState,
                             commandBuffer,
                             pass
                         )
                     )) ||
                    !encodeCapture(
                        *selectedState,
                        commandBuffer,
                        pass,
                        sourceQ,
                        sourceV,
                        batch.environmentCount
                    ) ||
                    (contactMode &&
                     !encodeContactCapture(
                         *selectedState,
                         commandBuffer,
                         pass,
                         sourceScene,
                         batch.environmentCount
                    ))) {
                    return reject(
                        std::move(diagnostics),
                        MetalWorldHostStatus::metalCommandFailure,
                        "failed to encode native task or MetalWorld "
                        "capture pass"
                    );
                }
                if (nativeTask &&
                    config.policyProgram.valid() &&
                    !config.policyProgram.criticLayers().empty() &&
                    !encodePolicyInference(
                        *selectedState,
                        commandBuffer,
                        config.policyProgram,
                        pass,
                        batch.environmentCount,
                        true,
                        diagnostics.layout.dispatch.controlStepCount
                    )) {
                    return reject(
                        std::move(diagnostics),
                        MetalWorldHostStatus::metalCommandFailure,
                        "failed to encode timeout bootstrap value"
                    );
                }
            }

            if (nativeTask &&
                config.policyProgram.valid() &&
                config.evaluateFinalPolicy) {
                MRMetalWorldPassGPU bootstrapPass{};
                bootstrapPass.controlStep =
                    diagnostics.layout.dispatch.controlStepCount;
                bootstrapPass.physicsSubstep = MR_INVALID_INDEX;
                if (!encodePolicyInference(
                        *selectedState,
                        commandBuffer,
                        config.policyProgram,
                        bootstrapPass,
                        batch.environmentCount
                    )) {
                    return reject(
                        std::move(diagnostics),
                        MetalWorldHostStatus::metalCommandFailure,
                        "failed to encode terminal native policy evaluation"
                    );
                }
            }

            std::vector<std::size_t> readbackIndices;
            if (nativeTask) {
                readbackIndices.push_back(kTaskEvidenceState);
            }
            if (config.publishFinalState) {
                readbackIndices.push_back(sourceQ);
                readbackIndices.push_back(sourceV);
                if (contactMode) {
                    readbackIndices.push_back(sourceScene);
                }
                if (world.rodCount() != 0u) {
                    readbackIndices.push_back(sourceRodNodes);
                    readbackIndices.push_back(sourceRodEdges);
                }
            }
            if (config.captureContactEvidence) {
                readbackIndices.push_back(
                    sourceManifoldHeaders
                );
                readbackIndices.push_back(
                    sourceManifoldPoints
                );
                readbackIndices.push_back(
                    sourceManifoldCounts
                );
                readbackIndices.push_back(kContacts);
                readbackIndices.push_back(kContactMetadata);
                readbackIndices.push_back(kIRBlocks);
                readbackIndices.push_back(kIREndpoints);
                readbackIndices.push_back(kEndpointRuntime);
                readbackIndices.push_back(kIRRows);
                readbackIndices.push_back(kIRCones);
                readbackIndices.push_back(kEvaluatedRows);
                readbackIndices.push_back(kEvaluatedCones);
                readbackIndices.push_back(kIslands);
                readbackIndices.push_back(
                    kIslandNodeReferences
                );
                readbackIndices.push_back(
                    kIslandConstraintReferences
                );
            }
            diagnostics = encodeReadbacks(
                *selectedState,
                commandBuffer,
                requirements,
                readbackIndices,
                std::move(diagnostics)
            );
            if (!diagnostics.succeeded()) {
                return diagnostics;
            }

            auto pending =
                std::make_unique<
                    detail::MetalWorldSubmissionState
                >();
            diagnostics.dispatched = true;
            pending->context = selectedState;
            pending->commandBuffer = commandBuffer;
            pending->diagnostics = diagnostics;
            pending->articulation =
                world.model().articulations[
                    world.articulationIndex()
                ];
            pending->finalQBuffer = sourceQ;
            pending->finalVBuffer = sourceV;
            pending->finalSceneBuffer = sourceScene;
            pending->finalManifoldHeaderBuffer =
                sourceManifoldHeaders;
            pending->finalManifoldPointBuffer =
                sourceManifoldPoints;
            pending->finalManifoldCountBuffer =
                sourceManifoldCounts;
            pending->finalRodNodeBuffer = sourceRodNodes;
            pending->finalRodEdgeBuffer = sourceRodEdges;
            pending->finalRodWitnessBuffer =
                sourceRodWitnesses;
            pending->resident = residentData;
            pending->policyRevision =
                config.policyProgram.valid()
                ? config.policyProgram.revision()
                : batch.policyRevision;
            pending->hasRods = world.rodCount() != 0u;
            pending->contactMode = contactMode;
            pending->nativeTask = nativeTask;
            pending->captureContactEvidence =
                config.captureContactEvidence;
            pending->publishFinalState =
                config.publishFinalState;
            pending->publishStateTrajectory =
                config.publishStateTrajectory;
            pending->start = std::chrono::steady_clock::now();
            pending->ownsInFlight = true;

            if (initializeResidentState) {
                const std::lock_guard residentLock(
                    residentData->mutex
                );
                residentData->stateArenaGeneration =
                    selectedState->stateArenaGeneration;
            }
            ++selectedState->stats.submissionCount;
            [commandBuffer commit];
            submission.state_ = std::move(pending);
            if (initializeResidentState) {
                residentState->state_ = residentData;
            }
            reservation.handoff();
            if (residentReservation != nullptr) {
                residentReservation->handoff();
            }
        }
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalBufferFailure,
            "host allocation failed while preparing MetalWorld submission"
        );
    } catch (const std::exception& exception) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::internalFailure,
            exception.what()
        );
    }
}

MetalWorldDiagnostics MetalWorldContext::run(
    const CompiledWorld& world,
    const MetalWorldBatch& batch,
    const MetalWorldStepConfig& config,
    MetalWorldResult& result
) {
    MetalWorldSubmission submission;
    MetalWorldDiagnostics diagnostics = submit(
        world,
        batch,
        config,
        submission
    );
    if (!diagnostics.succeeded()) {
        return diagnostics;
    }
    return submission.wait(result);
}

MetalWorldContextStats MetalWorldContext::stats()
    const noexcept {
    if (pool_ == nullptr) {
        return {};
    }
    try {
        MetalWorldContextStats aggregate{};
        const std::lock_guard poolLock(pool_->mutex);
        for (const auto& slot : pool_->slots) {
            const std::lock_guard slotLock(slot->mutex);
            const MetalWorldContextStats& source = slot->stats;
            aggregate.pipelineCreationCount +=
                source.pipelineCreationCount;
            aggregate.modelUploadCount +=
                source.modelUploadCount;
            aggregate.bufferAllocationCount +=
                source.bufferAllocationCount;
            aggregate.bufferGrowthCount +=
                source.bufferGrowthCount;
            aggregate.submissionCount += source.submissionCount;
            aggregate.completedSubmissionCount +=
                source.completedSubmissionCount;
            aggregate.retainedBufferBytes +=
                source.retainedBufferBytes;
            aggregate.memoryPlan.immutablePrivateBytes +=
                source.memoryPlan.immutablePrivateBytes;
            aggregate.memoryPlan.persistentStatePrivateBytes +=
                source.memoryPlan.persistentStatePrivateBytes;
            aggregate.memoryPlan.transientPrivateBytes +=
                source.memoryPlan.transientPrivateBytes;
            aggregate.memoryPlan.sharedBoundaryBytes +=
                source.memoryPlan.sharedBoundaryBytes;
            aggregate.memoryPlan.peakAliasedBytes +=
                source.memoryPlan.peakAliasedBytes;
            aggregate.queriedThreadExecutionWidth = std::max(
                aggregate.queriedThreadExecutionWidth,
                source.queriedThreadExecutionWidth
            );
            aggregate.usingPrivateHeaps =
                aggregate.usingPrivateHeaps ||
                source.usingPrivateHeaps;
            aggregate.hasInFlightSubmission =
                aggregate.hasInFlightSubmission ||
                source.hasInFlightSubmission;
        }
        return aggregate;
    } catch (...) {
        return {};
    }
}

const char* metalWorldHostStatusName(
    const MetalWorldHostStatus status
) noexcept {
    switch (status) {
    case MetalWorldHostStatus::success:
        return "success";
    case MetalWorldHostStatus::invalidModel:
        return "invalid_model";
    case MetalWorldHostStatus::unsupportedTopology:
        return "unsupported_topology";
    case MetalWorldHostStatus::unsupportedSolverMode:
        return "unsupported_solver_mode";
    case MetalWorldHostStatus::invalidDimensions:
        return "invalid_dimensions";
    case MetalWorldHostStatus::capacityOverflow:
        return "capacity_overflow";
    case MetalWorldHostStatus::arithmeticOverflow:
        return "arithmetic_overflow";
    case MetalWorldHostStatus::nonfiniteInput:
        return "nonfinite_input";
    case MetalWorldHostStatus::invalidReset:
        return "invalid_reset";
    case MetalWorldHostStatus::metallibUnavailable:
        return "metallib_unavailable";
    case MetalWorldHostStatus::metalDeviceUnavailable:
        return "metal_device_unavailable";
    case MetalWorldHostStatus::metalDeviceUnsupported:
        return "metal_device_unsupported";
    case MetalWorldHostStatus::metalLibraryFailure:
        return "metal_library_failure";
    case MetalWorldHostStatus::metalPipelineFailure:
        return "metal_pipeline_failure";
    case MetalWorldHostStatus::metalBufferFailure:
        return "metal_buffer_failure";
    case MetalWorldHostStatus::metalCommandFailure:
        return "metal_command_failure";
    case MetalWorldHostStatus::gpuEnvironmentFailure:
        return "gpu_environment_failure";
    case MetalWorldHostStatus::internalFailure:
        return "internal_failure";
    case MetalWorldHostStatus::contextBusy:
        return "context_busy";
    }
    return "unknown";
}

} // namespace metalrobo
