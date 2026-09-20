#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/EngineModel.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifndef METALROBO_DEFAULT_METALLIB
#error "METALROBO_DEFAULT_METALLIB must name the build-tree Metal library"
#endif

namespace {

constexpr std::uint32_t kStepCount = 2u;
constexpr float kTimestepSeconds = 1.0e-4f;
constexpr std::uint64_t kProgramFingerprint = 0x4e554d414e585458ull;
constexpr std::uint8_t kMarkerSentinel = 0xa5u;
constexpr std::size_t kPhaseCount = 3u;
constexpr std::size_t kMarkerCount = kStepCount * kPhaseCount;

using Phase = metalrobo::MetalNumanXTransactionPhase;
using Pass = metalrobo::MetalNumanXTransactionPass;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct BorrowedResources {
    void* q = nullptr;
    void* v = nullptr;
    void* bodyPoses = nullptr;
    void* pointWorld = nullptr;
    void* pointJacobians = nullptr;
    void* mujocoMuscles = nullptr;
    void* mujocoStates = nullptr;
    void* mujocoSites = nullptr;
    void* mujocoWraps = nullptr;
    void* mujocoRouteNodes = nullptr;
    void* mujocoResults = nullptr;
    void* mujocoGeneralizedForceArena = nullptr;
    void* tendonBindings = nullptr;
    void* tendonEnvelopes = nullptr;
    void* tendonTransfers = nullptr;
    void* tendonGeneralizedCorrections = nullptr;
    void* standStatuses = nullptr;
};

BorrowedResources borrowedResources(const Pass& pass) noexcept {
    return {
        .q = pass.q,
        .v = pass.v,
        .bodyPoses = pass.bodyPoses,
        .pointWorld = pass.pointWorld,
        .pointJacobians = pass.pointJacobians,
        .mujocoMuscles = pass.mujocoMuscles,
        .mujocoStates = pass.mujocoStates,
        .mujocoSites = pass.mujocoSites,
        .mujocoWraps = pass.mujocoWraps,
        .mujocoRouteNodes = pass.mujocoRouteNodes,
        .mujocoResults = pass.mujocoResults,
        .mujocoGeneralizedForceArena = pass.mujocoGeneralizedForceArena,
        .tendonBindings = pass.tendonBindings,
        .tendonEnvelopes = pass.tendonEnvelopes,
        .tendonTransfers = pass.tendonTransfers,
        .tendonGeneralizedCorrections = pass.tendonGeneralizedCorrections,
        .standStatuses = pass.standStatuses,
    };
}

bool sameResources(
    const BorrowedResources& lhs,
    const BorrowedResources& rhs
) noexcept {
    return lhs.q == rhs.q && lhs.v == rhs.v &&
        lhs.bodyPoses == rhs.bodyPoses && lhs.pointWorld == rhs.pointWorld &&
        lhs.pointJacobians == rhs.pointJacobians &&
        lhs.mujocoMuscles == rhs.mujocoMuscles &&
        lhs.mujocoStates == rhs.mujocoStates &&
        lhs.mujocoSites == rhs.mujocoSites &&
        lhs.mujocoWraps == rhs.mujocoWraps &&
        lhs.mujocoRouteNodes == rhs.mujocoRouteNodes &&
        lhs.mujocoResults == rhs.mujocoResults &&
        lhs.mujocoGeneralizedForceArena == rhs.mujocoGeneralizedForceArena &&
        lhs.tendonBindings == rhs.tendonBindings &&
        lhs.tendonEnvelopes == rhs.tendonEnvelopes &&
        lhs.tendonTransfers == rhs.tendonTransfers &&
        lhs.tendonGeneralizedCorrections ==
            rhs.tendonGeneralizedCorrections &&
        lhs.standStatuses == rhs.standStatuses;
}

bool borrowedBufferOnDevice(
    void* opaque,
    const std::uint64_t registryID
) noexcept {
    if (opaque == nullptr) {
        return false;
    }
    __unsafe_unretained id<MTLBuffer> buffer =
        (__bridge id<MTLBuffer>)opaque;
    return buffer != nil && buffer.device != nil &&
        buffer.device.registryID == registryID && buffer.length > 0u;
}

bool validBorrowedResources(
    const BorrowedResources& resources,
    const std::uint64_t registryID
) noexcept {
    return borrowedBufferOnDevice(resources.q, registryID) &&
        borrowedBufferOnDevice(resources.v, registryID) &&
        borrowedBufferOnDevice(resources.bodyPoses, registryID) &&
        borrowedBufferOnDevice(resources.pointWorld, registryID) &&
        borrowedBufferOnDevice(resources.pointJacobians, registryID) &&
        borrowedBufferOnDevice(resources.mujocoMuscles, registryID) &&
        borrowedBufferOnDevice(resources.mujocoStates, registryID) &&
        borrowedBufferOnDevice(resources.mujocoSites, registryID) &&
        borrowedBufferOnDevice(resources.mujocoWraps, registryID) &&
        borrowedBufferOnDevice(resources.mujocoRouteNodes, registryID) &&
        borrowedBufferOnDevice(resources.mujocoResults, registryID) &&
        borrowedBufferOnDevice(
            resources.mujocoGeneralizedForceArena, registryID
        ) &&
        borrowedBufferOnDevice(resources.tendonBindings, registryID) &&
        borrowedBufferOnDevice(resources.tendonEnvelopes, registryID) &&
        borrowedBufferOnDevice(resources.tendonTransfers, registryID) &&
        borrowedBufferOnDevice(
            resources.tendonGeneralizedCorrections, registryID
        ) &&
        borrowedBufferOnDevice(resources.standStatuses, registryID);
}

struct PhaseRecord {
    Phase phase = Phase::beginStep;
    std::uint32_t stepIndex = 0u;
};

struct TransactionAudit {
    bool rollback = false;
    const char* failure = nullptr;
    void* commandBuffer = nullptr;
    BorrowedResources resources{};
    std::array<PhaseRecord, kMarkerCount> records{};
    std::size_t recordCount = 0u;
    std::uint32_t rejectionCount = 0u;
    std::uint32_t abortCount = 0u;
    id<MTLBuffer> markers = nil;

    bool fail(const char* message) noexcept {
        if (failure == nullptr) {
            failure = message;
        }
        return false;
    }
};

Phase expectedPhase(const std::size_t ordinal) noexcept {
    switch (ordinal % kPhaseCount) {
    case 0u:
        return Phase::beginStep;
    case 1u:
        return Phase::preDynamics;
    default:
        return Phase::postDynamics;
    }
}

bool validatePassMetadata(const Pass& pass) noexcept {
    const std::uint32_t expectedAccessFlags =
        metalrobo::MetalNumanXTransactionReadBorrowedState |
        (pass.phase == Phase::beginStep
             ? metalrobo::MetalNumanXTransactionWriteMujocoExcitation
             : 0u) |
        (pass.phase == Phase::postDynamics
             ? metalrobo::MetalNumanXTransactionWriteStandFailure
             : 0u);
    return pass.abiVersion == metalrobo::kMetalNumanXTransactionABIVersion &&
        pass.structSize == sizeof(Pass) &&
        pass.accessFlags == expectedAccessFlags && pass.reserved0 == 0u &&
        pass.programFingerprint == kProgramFingerprint &&
        pass.stepCount == kStepCount &&
        pass.timestepSeconds == kTimestepSeconds &&
        pass.articulationFirstBody == 1u &&
        pass.bodyJacobianPointOffset == 0u &&
        pass.environmentCount == 1u &&
        pass.qCoordinateCount == 7u && pass.qElementCount == 7u &&
        pass.qStride == 7u && pass.dofCount == 6u &&
        pass.vElementCount == 6u && pass.vStride == 6u &&
        pass.bodyCount == 1u && pass.bodyPoseElementCount == 1u &&
        pass.bodyPoseStride == 1u && pass.pointCount == 4u &&
        pass.pointWorldElementCount == 4u && pass.pointWorldStride == 4u &&
        pass.pointJacobianElementCount == 72u &&
        pass.pointJacobianStride == 72u &&
        pass.mujocoMuscleCount == 1u &&
        pass.mujocoStateElementCount == 1u &&
        pass.mujocoStateStride == 1u && pass.mujocoSiteCount == 2u &&
        pass.mujocoWrapCount == 0u && pass.mujocoRouteNodeCount == 2u &&
        pass.mujocoResultElementCount == 1u &&
        pass.mujocoResultStride == 1u &&
        pass.mujocoMuscleGeneralizedForceElementCount == 6u &&
        pass.mujocoMuscleGeneralizedForceRowStride == 6u &&
        pass.mujocoMuscleGeneralizedForceEnvironmentStride == 6u &&
        pass.mujocoGeneralizedForceElementCount == 6u &&
        pass.mujocoGeneralizedForceOffset == 6u &&
        pass.mujocoGeneralizedForceStride == 6u &&
        pass.mujocoGeneralizedForceArenaElementCount == 12u &&
        pass.tendonBindingCount == 0u && pass.tendonEnvelopeCount == 0u &&
        pass.tendonTransferElementCount == 0u &&
        pass.tendonTransferStride == 0u &&
        pass.tendonCorrectionElementCount == 0u &&
        pass.tendonCorrectionStride == 0u &&
        pass.standStatusElementCount == 1u && pass.standStatusStride == 1u;
}

bool encodeTransaction(void* context, const Pass& pass) noexcept {
    auto& audit = *static_cast<TransactionAudit*>(context);
    if (pass.commandBuffer == nullptr || audit.markers == nil) {
        return audit.fail("transaction pass lacks its command buffer or marker");
    }
    if (audit.recordCount >= audit.records.size()) {
        return audit.fail("transaction offered more than three phases per step");
    }

    const std::size_t ordinal = audit.recordCount;
    const Phase phase = expectedPhase(ordinal);
    const std::uint32_t stepIndex = static_cast<std::uint32_t>(
        ordinal / kPhaseCount
    );
    if (pass.phase != phase || pass.stepIndex != stepIndex) {
        return audit.fail("transaction phases are not in canonical step order");
    }
    if (!validatePassMetadata(pass)) {
        return audit.fail("transaction pass metadata disagrees with the fixture");
    }

    __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    if (commandBuffer == nil || commandBuffer.device == nil) {
        return audit.fail("transaction command buffer is not a Metal object");
    }
    const std::uint64_t registryID = commandBuffer.device.registryID;
    const BorrowedResources resources = borrowedResources(pass);
    if (!validBorrowedResources(resources, registryID)) {
        return audit.fail("transaction did not borrow every expected Metal buffer");
    }
    if (audit.commandBuffer == nullptr) {
        audit.commandBuffer = pass.commandBuffer;
        audit.resources = resources;
    } else if (audit.commandBuffer != pass.commandBuffer ||
               !sameResources(audit.resources, resources)) {
        return audit.fail("transaction changed command buffer or resources mid-horizon");
    }

    audit.records[ordinal] = {.phase = pass.phase, .stepIndex = pass.stepIndex};
    ++audit.recordCount;
    if (audit.rollback && pass.phase == Phase::preDynamics &&
        pass.stepIndex == 0u) {
        ++audit.rejectionCount;
        return false;
    }

    id<MTLBlitCommandEncoder> encoder = [commandBuffer blitCommandEncoder];
    if (encoder == nil) {
        return audit.fail("transaction could not create its Metal blit encoder");
    }
    [encoder fillBuffer:audit.markers
                  range:NSMakeRange(ordinal, 1u)
                  value:static_cast<std::uint8_t>(0x20u + ordinal)];
    [encoder endEncoding];
    return true;
}

void abortTransaction(void* context, void* commandBuffer) noexcept {
    auto& audit = *static_cast<TransactionAudit*>(context);
    ++audit.abortCount;
    if (audit.commandBuffer == nullptr ||
        audit.commandBuffer != commandBuffer) {
        audit.fail("transaction abort did not return the borrowed command buffer");
    }
}

std::vector<MRArticulatedPointImpulseGPU> makeBodyProbes() {
    constexpr std::array<mr_float4, 4u> points{{
        {0.0f, 0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
    }};
    std::vector<MRArticulatedPointImpulseGPU> result;
    result.reserve(points.size());
    for (const mr_float4 point : points) {
        MRArticulatedPointImpulseGPU query{};
        query.bodyIndex = 1u;
        query.localPoint = point;
        result.push_back(query);
    }
    return result;
}

struct SyntheticHumanFixture {
    metalrobo::EngineModel model = metalrobo::makeFreeSphereEngineModel();
    std::vector<float> q = model.defaultQ;
    std::vector<float> v = model.defaultV;
    std::vector<MRArticulatedPointImpulseGPU> points = makeBodyProbes();
    std::vector<MRMujocoMuscleGPU> muscles;
    std::vector<MRMujocoMuscleStateGPU> states;
    std::vector<MRMujocoMuscleSiteGPU> sites;
    std::vector<MRMujocoMuscleRouteNodeGPU> routeNodes;

    SyntheticHumanFixture() {
        MRMujocoMuscleSiteGPU origin{};
        origin.bodyIndex = 1u;
        origin.localPoint = {-0.05f, 0.0f, 0.0f, 0.0f};
        sites.push_back(origin);
        MRMujocoMuscleSiteGPU insertion{};
        insertion.bodyIndex = 1u;
        insertion.localPoint = {0.05f, 0.0f, 0.0f, 0.0f};
        sites.push_back(insertion);

        MRMujocoMuscleRouteNodeGPU first{};
        first.type = MR_MUJOCO_MUSCLE_ROUTE_SITE;
        first.targetIndex = 0u;
        first.sideSiteIndex = MR_INVALID_INDEX;
        routeNodes.push_back(first);
        MRMujocoMuscleRouteNodeGPU second{};
        second.type = MR_MUJOCO_MUSCLE_ROUTE_SITE;
        second.targetIndex = 1u;
        second.sideSiteIndex = MR_INVALID_INDEX;
        routeNodes.push_back(second);

        MRMujocoMuscleGPU muscle{};
        muscle.route = {0u, 2u, 0u, 0u};
        muscle.lengthRangeAndAcceleration = {0.05f, 0.15f, 1.0f, 0.0f};
        muscle.controlRange = {0.0f, 1.0f, 0.0f, 0.0f};
        muscles.push_back(muscle);

        MRMujocoMuscleStateGPU state{};
        state.excitationAndActivation = {0.0f, 0.0f, 0.0f, 0.0f};
        states.push_back(state);
    }

    metalrobo::MetalArticulatedOperatorInput input(
        TransactionAudit& audit
    ) const {
        metalrobo::MetalArticulatedOperatorInput result{
            .articulationIndex = 0u,
            .environmentCount = 1u,
            .pointCount = points.size(),
            .q = q,
            .v = v,
            .points = points,
            .mujoco = {
                .muscles = muscles,
                .states = states,
                .sites = sites,
                .wraps = {},
                .routeNodes = routeNodes,
                .bodyJacobianPointOffset = 0u,
            },
            .stand = {
                .v = v,
                .contacts = {},
                .jointEqualities = {},
                .tendonBindings = {},
                .tendonEnvelopes = {},
                .tendonLoadProgram = {},
                .numanXTransactionProgram = {
                    .context = &audit,
                    .encode = &encodeTransaction,
                    .abort = &abortTransaction,
                    .fingerprint = kProgramFingerprint,
                },
                .stepCount = kStepCount,
                .contactIterationCount = 1u,
                .enableContact = false,
                .enableRootAssistance = false,
            },
        };
        return result;
    }
};

template <typename T>
void seed(std::vector<T>& values, const std::uint8_t byte) {
    static_assert(std::is_trivially_copyable_v<T>);
    values.resize(2u);
    std::memset(values.data(), byte, values.size() * sizeof(T));
}

void seedResult(metalrobo::MetalArticulatedOperatorResult& result) {
    result.layout.dispatch.flags = 0x13579bdfu;
    result.layout.qElements = 17u;
    result.layout.totalAllocatedBytes = 0x2468u;
    seed(result.bodyPoses, 0x10u);
    seed(result.pointWorld, 0x11u);
    seed(result.diagnosticMassMatrix, 0x12u);
    seed(result.pointJacobians, 0x13u);
    seed(result.generalizedImpulse, 0x14u);
    seed(result.deltaVelocity, 0x15u);
    seed(result.statuses, 0x16u);
    seed(result.millardResults, 0x17u);
    seed(result.millardGeneralizedForces, 0x18u);
    seed(result.mujocoResults, 0x19u);
    seed(result.mujocoActivationStates, 0x1au);
    seed(result.mujocoMuscleGeneralizedForces, 0x1bu);
    seed(result.mujocoGeneralizedForces, 0x1cu);
    seed(result.standQ, 0x1du);
    seed(result.standV, 0x1eu);
    seed(result.standStatuses, 0x1fu);
    seed(result.standTendonTransfers, 0x20u);
    seed(result.standTendonGeneralizedCorrections, 0x21u);
}

template <typename T>
void appendObject(std::vector<std::byte>& bytes, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* first = reinterpret_cast<const std::byte*>(&value);
    bytes.insert(bytes.end(), first, first + sizeof(T));
}

template <typename T>
void appendVector(std::vector<std::byte>& bytes, const std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    appendObject(bytes, values.size());
    const auto* first = reinterpret_cast<const std::byte*>(values.data());
    bytes.insert(bytes.end(), first, first + values.size() * sizeof(T));
}

std::vector<std::byte> resultBytes(
    const metalrobo::MetalArticulatedOperatorResult& result
) {
    std::vector<std::byte> bytes;
    appendObject(bytes, result.layout);
    appendVector(bytes, result.bodyPoses);
    appendVector(bytes, result.pointWorld);
    appendVector(bytes, result.diagnosticMassMatrix);
    appendVector(bytes, result.pointJacobians);
    appendVector(bytes, result.generalizedImpulse);
    appendVector(bytes, result.deltaVelocity);
    appendVector(bytes, result.statuses);
    appendVector(bytes, result.millardResults);
    appendVector(bytes, result.millardGeneralizedForces);
    appendVector(bytes, result.mujocoResults);
    appendVector(bytes, result.mujocoActivationStates);
    appendVector(bytes, result.mujocoMuscleGeneralizedForces);
    appendVector(bytes, result.mujocoGeneralizedForces);
    appendVector(bytes, result.standQ);
    appendVector(bytes, result.standV);
    appendVector(bytes, result.standStatuses);
    appendVector(bytes, result.standTendonTransfers);
    appendVector(bytes, result.standTendonGeneralizedCorrections);
    return bytes;
}

TransactionAudit makeAudit(const bool rollback) {
    TransactionAudit audit;
    audit.rollback = rollback;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    require(device != nil, "NumanX transaction probe requires a local Metal device");
    audit.markers = [device newBufferWithLength:kMarkerCount
                                       options:MTLResourceStorageModeShared];
    require(audit.markers != nil, "could not allocate transaction marker buffer");
    std::memset(audit.markers.contents, kMarkerSentinel, kMarkerCount);
    return audit;
}

metalrobo::MetalArticulatedOperatorContext makeContext() {
    const metalrobo::MetalArticulatedOperatorConfig config{
        .pointJacobiansOnly = true,
        .mujocoActivationTimestepSeconds = kTimestepSeconds,
        .metallibPath = METALROBO_DEFAULT_METALLIB,
    };
    return metalrobo::MetalArticulatedOperatorContext(config);
}

void verifyMarkers(const TransactionAudit& audit, const bool committed) {
    const auto* markers = static_cast<const std::uint8_t*>(
        audit.markers.contents
    );
    for (std::size_t index = 0u; index < kMarkerCount; ++index) {
        const std::uint8_t expected = committed
            ? static_cast<std::uint8_t>(0x20u + index)
            : kMarkerSentinel;
        require(markers[index] == expected,
                "transaction marker publication disagrees with commit state");
    }
}

void runAccept() {
    SyntheticHumanFixture fixture;
    TransactionAudit audit = makeAudit(false);
    auto context = makeContext();
    const auto input = fixture.input(audit);
    metalrobo::MetalArticulatedOperatorResult result;
    const auto diagnostics = context.run(fixture.model, input, result);

    require(audit.failure == nullptr,
            audit.failure == nullptr ? "" : audit.failure);
    require(diagnostics.succeeded() && diagnostics.dispatched &&
                diagnostics.published &&
                diagnostics.completedStandSteps == kStepCount &&
                diagnostics.numanXProgramFingerprint ==
                    kProgramFingerprint &&
                diagnostics.commandBufferIdentity ==
                    reinterpret_cast<std::uintptr_t>(audit.commandBuffer),
            "accepted NumanX transaction did not dispatch and publish: " +
                diagnostics.message);
    require(audit.recordCount == kMarkerCount && audit.rejectionCount == 0u &&
                audit.abortCount == 0u,
            "accepted NumanX transaction did not offer exactly three phases per step");
    require(result.standQ.size() == 7u && result.standV.size() == 6u &&
                result.standStatuses.size() == 1u &&
                result.standStatuses.front().code == MR_NUMI_HUMAN_STAND_SUCCESS &&
                result.standStatuses.front().completedSteps == kStepCount &&
                result.mujocoResults.size() == 1u &&
                result.mujocoResults.front().status ==
                    MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS,
            "accepted NumanX transaction published an incoherent Human result");
    verifyMarkers(audit, true);
    std::cout << "numanx human transaction accept passed device=\""
              << diagnostics.deviceName << "\" phases=" << audit.recordCount
              << " steps=" << diagnostics.completedStandSteps << '\n';
}

void runRollback() {
    SyntheticHumanFixture fixture;
    TransactionAudit audit = makeAudit(true);
    auto context = makeContext();
    const auto input = fixture.input(audit);
    metalrobo::MetalArticulatedOperatorResult result;
    seedResult(result);
    const std::vector<std::byte> sentinel = resultBytes(result);
    const auto diagnostics = context.run(fixture.model, input, result);

    require(audit.failure == nullptr,
            audit.failure == nullptr ? "" : audit.failure);
    require(!diagnostics.succeeded() && !diagnostics.dispatched &&
                !diagnostics.published &&
                diagnostics.status ==
                    metalrobo::MetalArticulatedOperatorHostStatus::
                        externalProgramFailure &&
                std::string(
                    metalrobo::metalArticulatedOperatorHostStatusName(
                        diagnostics.status
                    )
                ) == "external_program_failure",
            "rejected NumanX transaction was dispatched or published");
    require(audit.recordCount == 2u &&
                audit.records[0].phase == Phase::beginStep &&
                audit.records[1].phase == Phase::preDynamics &&
                audit.rejectionCount == 1u && audit.abortCount == 1u,
            "rejected NumanX transaction did not abort exactly once");
    require(resultBytes(result) == sentinel,
            "rejected NumanX transaction changed the caller result sentinel");
    verifyMarkers(audit, false);
    std::cout << "numanx human transaction rollback passed phases="
              << audit.recordCount << " aborts=" << audit.abortCount << '\n';
}

void runABIContract() {
    metalrobo::MetalNumanXTransactionProgram empty;
    require(!empty.configured() && !empty.valid(),
            "default NumanX program is not an empty configuration");

    TransactionAudit audit = makeAudit(false);
    metalrobo::MetalNumanXTransactionProgram malformed{
        .context = &audit,
        .encode = &encodeTransaction,
        .abort = &abortTransaction,
        .fingerprint = kProgramFingerprint,
    };
    require(malformed.configured() && malformed.valid(),
            "well-formed NumanX program did not validate");
    malformed.abiVersion =
        metalrobo::kMetalNumanXTransactionABIVersion + 1u;
    require(malformed.configured() && !malformed.valid(),
            "unknown NumanX program ABI did not fail closed");

    SyntheticHumanFixture fixture;
    auto input = fixture.input(audit);
    input.stand.numanXTransactionProgram = malformed;
    auto context = makeContext();
    metalrobo::MetalArticulatedOperatorResult result;
    seedResult(result);
    const std::vector<std::byte> sentinel = resultBytes(result);
    const auto diagnostics = context.run(fixture.model, input, result);
    require(!diagnostics.succeeded() && !diagnostics.dispatched &&
                !diagnostics.published && audit.recordCount == 0u &&
                audit.abortCount == 0u && resultBytes(result) == sentinel,
            "unknown NumanX program ABI reached encoding or publication");

    malformed.abiVersion = metalrobo::kMetalNumanXTransactionABIVersion;
    malformed.structSize = sizeof(malformed) - 1u;
    require(malformed.configured() && !malformed.valid(),
            "truncated NumanX program ABI did not fail closed");
    std::cout << "numanx human transaction ABI contract passed version="
              << metalrobo::kMetalNumanXTransactionABIVersion
              << " pass_bytes=" << sizeof(Pass)
              << " program_bytes=" << sizeof(malformed) << '\n';
}

} // namespace

int main(const int argc, const char* const argv[]) {
    @autoreleasepool {
        try {
            require(argc == 2,
                    "usage: metalrobo_numanx_transaction_probe --accept|--rollback|--abi-contract");
            const std::string mode = argv[1];
            if (mode == "--accept") {
                runAccept();
            } else if (mode == "--rollback") {
                runRollback();
            } else if (mode == "--abi-contract") {
                runABIContract();
            } else {
                throw std::runtime_error("unknown NumanX transaction probe mode");
            }
            return 0;
        } catch (const std::exception& exception) {
            std::cerr << "numanx transaction probe failed: "
                      << exception.what() << '\n';
            return 1;
        }
    }
}
