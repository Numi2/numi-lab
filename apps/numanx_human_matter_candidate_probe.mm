#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/EngineModel.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kEnvironmentCount = 1u;
constexpr std::uint32_t kNq = 161u;
constexpr std::uint32_t kNv = 160u;
constexpr std::uint32_t kBodyCount = 155u;
constexpr std::uint32_t kFirstBody = 1u;
constexpr std::uint32_t kWorldBodyCount = kFirstBody + kBodyCount;
constexpr float kTimestep = 1.0e-3f;
constexpr float kFiniteDifferenceEpsilon = 2.0e-1f;
constexpr std::uint32_t kFiniteDifferenceDof = 80u;
constexpr std::uint64_t kProgramFingerprint = 0x484d43414e443031ull;
constexpr std::uint64_t kTransactionFingerprint = 0x484d54584e303031ull;
constexpr std::uint64_t kLinearizationEpoch = 0x484d45504f434831ull;
constexpr std::uint64_t kSlotGeneration = 0x484d47454e303031ull;
constexpr std::uint32_t kControlStep = 37u;

using Phase = metalrobo::MetalNumanXHumanMatterPhase;
using Pass = metalrobo::MetalNumanXHumanMatterPass;
using Query = metalrobo::MetalNumanXHumanMatterCandidateQuery;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

mr_float4 f4(
    const float x,
    const float y,
    const float z,
    const float w = 0.0f
) {
    return {x, y, z, w};
}

template <typename T>
id<MTLBuffer> makeBuffer(
    id<MTLDevice> device,
    const std::size_t count,
    NSString* label
) {
    const std::size_t bytes = std::max<std::size_t>(sizeof(T), count * sizeof(T));
    id<MTLBuffer> buffer = [device
        newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    require(buffer != nil && buffer.contents != nullptr,
            "failed to allocate Metal probe buffer");
    buffer.label = label;
    std::memset(buffer.contents, 0, buffer.length);
    return buffer;
}

template <typename T>
T* contents(id<MTLBuffer> buffer) {
    return static_cast<T*>(buffer.contents);
}

MRBodyPropertiesGPU dynamicBody(
    const std::uint32_t parent,
    const std::uint32_t inbound
) {
    MRBodyPropertiesGPU body{};
    body.articulationIndex = 0u;
    body.parentBody = parent;
    body.inboundJoint = inbound;
    body.motionType = MR_MOTION_DYNAMIC;
    body.massAndInverseMass = f4(0.25f, 4.0f, 0.0f, 0.0f);
    body.centerOfMass = f4(0.0f, 0.0f, 0.0f, 0.0f);
    body.inertiaRow0 = f4(0.01f, 0.0f, 0.0f, 0.0f);
    body.inertiaRow1 = f4(0.0f, 0.012f, 0.0f, 0.0f);
    body.inertiaRow2 = f4(0.0f, 0.0f, 0.014f, 0.0f);
    body.inverseInertiaRow0 = f4(100.0f, 0.0f, 0.0f, 0.0f);
    body.inverseInertiaRow1 = f4(0.0f, 1.0f / 0.012f, 0.0f, 0.0f);
    body.inverseInertiaRow2 = f4(0.0f, 0.0f, 1.0f / 0.014f, 0.0f);
    body.dampingAndSpeedLimits = f4(0.0f, 0.0f, 1.0e6f, 1.0e6f);
    return body;
}

metalrobo::EngineModel makeHumanModel() {
    metalrobo::EngineModel model = metalrobo::makeFreeSphereEngineModel();
    model.name = "numanx_human_matter_candidate_160";
    model.world.bodyCount = kWorldBodyCount;
    model.world.jointCount = kBodyCount - 1u;
    model.world.nq = kNq;
    model.world.nv = kNv;
    MRArticulationGPU& articulation = model.articulations.front();
    articulation.bodyCount = kBodyCount;
    articulation.jointCount = kBodyCount - 1u;
    articulation.nq = kNq;
    articulation.nv = kNv;

    model.joints.reserve(kBodyCount - 1u);
    model.bodies.reserve(kWorldBodyCount);
    model.dofs.reserve(kNv);
    for (std::uint32_t jointIndex = 0u;
         jointIndex < kBodyCount - 1u; ++jointIndex) {
        const std::uint32_t parent = kFirstBody + jointIndex;
        const std::uint32_t child = parent + 1u;
        MRJointDescriptorGPU joint{};
        joint.parentBody = parent;
        joint.childBody = child;
        joint.jointType = MR_JOINT_REVOLUTE;
        joint.qOffset = 7u + jointIndex;
        joint.nq = 1u;
        joint.vOffset = 6u + jointIndex;
        joint.nv = 1u;
        switch (jointIndex % 3u) {
        case 0u:
            joint.axis0 = f4(0.0f, 0.0f, 1.0f, 0.0f);
            break;
        case 1u:
            joint.axis0 = f4(0.0f, 1.0f, 0.0f, 0.0f);
            break;
        default:
            joint.axis0 = f4(1.0f, 0.0f, 0.0f, 0.0f);
            break;
        }
        joint.parentAnchor = f4(0.025f, 0.003f, 0.0f, 0.0f);
        joint.childAnchor = f4(-0.025f, -0.003f, 0.0f, 0.0f);
        joint.parentRotation = f4(0.0f, 0.0f, 0.0f, 1.0f);
        joint.childRotation = f4(0.0f, 0.0f, 0.0f, 1.0f);
        model.joints.push_back(joint);
        model.bodies.push_back(dynamicBody(parent, jointIndex));

        MRDofPropertiesGPU dof{};
        dof.articulationIndex = 0u;
        dof.jointIndex = jointIndex;
        dof.qIndex = joint.qOffset;
        dof.vIndex = joint.vOffset;
        dof.localDof = 0u;
        dof.drive = f4(0.0f, 0.03f, 0.4f, 0.0f);
        model.dofs.push_back(dof);
    }
    model.defaultQ.resize(kNq, 0.0f);
    model.defaultQ[0] = 0.1f;
    model.defaultQ[1] = 1.0f;
    model.defaultQ[2] = -0.2f;
    model.defaultQ[6] = 1.0f;
    for (std::uint32_t coordinate = 7u; coordinate < kNq; ++coordinate) {
        model.defaultQ[coordinate] =
            0.12f * std::sin(0.17f * static_cast<float>(coordinate));
    }
    model.defaultV.resize(kNv, 0.0f);
    for (std::uint32_t dof = 0u; dof < kNv; ++dof) {
        model.defaultV[dof] =
            0.02f * std::cos(0.11f * static_cast<float>(dof));
    }
    std::string reason;
    require(model.valid(&reason), "160-DoF Human fixture is invalid: " + reason);
    return model;
}

std::vector<MRArticulatedPointImpulseGPU> bodyProbes() {
    constexpr std::array<mr_float4, 4u> locals{{
        {0.0f, 0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
    }};
    std::vector<MRArticulatedPointImpulseGPU> result;
    result.reserve(4u * kBodyCount);
    for (std::uint32_t localBody = 0u;
         localBody < kBodyCount; ++localBody) {
        for (const mr_float4 local : locals) {
            MRArticulatedPointImpulseGPU query{};
            query.bodyIndex = kFirstBody + localBody;
            query.localPoint = local;
            result.push_back(query);
        }
    }
    return result;
}

struct CandidateArena {
    id<MTLBuffer> delta = nil;
    id<MTLBuffer> q = nil;
    id<MTLBuffer> bodies = nil;
    id<MTLBuffer> pointWorld = nil;
    id<MTLBuffer> pointJacobians = nil;
};

CandidateArena makeCandidateArena(
    id<MTLDevice> device,
    NSString* label,
    const bool withPoint
) {
    CandidateArena result;
    result.delta = makeBuffer<float>(device, kNv,
        [label stringByAppendingString:@" delta"]);
    result.q = makeBuffer<float>(device, kNq,
        [label stringByAppendingString:@" q"]);
    result.bodies = makeBuffer<MRBodyStateGPU>(device, kWorldBodyCount,
        [label stringByAppendingString:@" bodies"]);
    if (withPoint) {
        result.pointWorld = makeBuffer<MRArticulatedPointWorldGPU>(
            device, 1u, [label stringByAppendingString:@" point world"]);
        result.pointJacobians = makeBuffer<float>(
            device, 3u * kNv,
            [label stringByAppendingString:@" point Jacobians"]);
    }
    return result;
}

struct CandidateAudit {
    id<MTLDevice> device = nil;
    id<MTLBuffer> reaction = nil;
    id<MTLBuffer> joint = nil;
    id<MTLBuffer> token = nil;
    id<MTLBuffer> matterApplyOutcome = nil;
    id<MTLBuffer> standStatusSnapshot = nil;
    id<MTLBuffer> acceptJoint = nil;
    id<MTLBuffer> acceptToken = nil;
    id<MTLBuffer> attachment = nil;
    id<MTLBuffer> factorBefore = nil;
    id<MTLBuffer> factorAfter = nil;
    CandidateArena base{};
    CandidateArena plus{};
    CandidateArena minus{};
    CandidateArena bodyOnly{};
    CandidateArena jacobianOnly{};
    std::vector<MRMujocoMuscleGPU> muscles;
    std::vector<MRMujocoMuscleStateGPU> muscleStates;
    std::vector<MRMujocoMuscleSiteGPU> muscleSites;
    std::vector<MRMujocoMuscleRouteNodeGPU> muscleRoutes;
    const char* failure = nullptr;
    std::uint32_t phaseCount = 0u;
    std::uint32_t abortCount = 0u;
    std::uint32_t acquireLeaseCount = 0u;
    std::uint32_t releaseLeaseCount = 0u;
    bool malformedAddressRejected = false;
    bool malformedStrideRejected = false;
    bool aliasRejected = false;
    bool bodyOnlyAccepted = false;
    bool optionalPointWorldAccepted = false;

    bool fail(const char* message) noexcept {
        if (failure == nullptr) failure = message;
        return false;
    }
};

Query candidateQuery(
    const CandidateAudit& audit,
    const CandidateArena& arena,
    const bool withPoint
) noexcept {
    Query query{};
    query.accessFlags =
        metalrobo::MetalNumanXHumanMatterCandidateReadDeltaVelocity |
        metalrobo::MetalNumanXHumanMatterCandidateWriteQ |
        metalrobo::MetalNumanXHumanMatterCandidateWriteBodies;
    query.deltaVelocity = (__bridge void*)arena.delta;
    query.candidateQ = (__bridge void*)arena.q;
    query.candidateBodies = (__bridge void*)arena.bodies;
    query.deltaVelocityGPUAddress = arena.delta.gpuAddress;
    query.candidateQGPUAddress = arena.q.gpuAddress;
    query.candidateBodiesGPUAddress = arena.bodies.gpuAddress;
    query.deltaVelocityStride = kNv;
    query.candidateQStride = kNq;
    query.candidateBodyStride = kWorldBodyCount;
    if (withPoint) {
        query.accessFlags |=
            metalrobo::MetalNumanXHumanMatterCandidateReadPointQueries |
            metalrobo::MetalNumanXHumanMatterCandidateWritePointWorld |
            metalrobo::MetalNumanXHumanMatterCandidateWritePointJacobians;
        query.pointQueries = (__bridge void*)audit.attachment;
        query.pointWorld = (__bridge void*)arena.pointWorld;
        query.pointJacobians = (__bridge void*)arena.pointJacobians;
        query.pointQueriesGPUAddress = audit.attachment.gpuAddress;
        query.pointWorldGPUAddress = arena.pointWorld.gpuAddress;
        query.pointJacobiansGPUAddress = arena.pointJacobians.gpuAddress;
        query.pointCount = 1u;
        query.pointStride = 1u;
        query.pointWorldStride = 1u;
        query.pointJacobianStride = 3u * kNv;
    }
    query.programFingerprint = kProgramFingerprint;
    query.transactionFingerprint = kTransactionFingerprint;
    query.linearizationEpoch = kLinearizationEpoch;
    query.slotGeneration = kSlotGeneration;
    query.physicsSubstepCount = 1u;
    query.controlStep = kControlStep;
    return query;
}

bool encodeProgram(void* raw, const Pass& pass) noexcept {
    auto& audit = *static_cast<CandidateAudit*>(raw);
    ++audit.phaseCount;
    if (pass.commandBuffer == nullptr || pass.environmentCount != 1u ||
        pass.qCoordinateCount != kNq || pass.dofCount != kNv ||
        pass.bodyCount != kBodyCount ||
        pass.articulationFirstBody != kFirstBody ||
        pass.programFingerprint != kProgramFingerprint ||
        pass.transactionFingerprint != kTransactionFingerprint ||
        pass.linearizationEpoch != kLinearizationEpoch ||
        pass.slotGeneration != kSlotGeneration ||
        pass.physicsSubstepCount != 1u ||
        pass.controlStep != kControlStep) {
        return audit.fail("owner pass metadata mismatch");
    }
    __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    if (pass.phase == Phase::beginStep) {
        if (pass.encodeExactCandidate != nullptr ||
            pass.exactCandidateContext != nullptr) {
            return audit.fail("exact candidate escaped preDynamics");
        }
        return true;
    }
    if (pass.phase == Phase::preDynamics) {
        if (pass.encodeExactCandidate == nullptr ||
            pass.exactCandidateContext == nullptr) {
            return audit.fail("preDynamics lacks exact candidate callback");
        }
        id<MTLBlitCommandEncoder> before =
            [commandBuffer blitCommandEncoder];
        if (before == nil) return audit.fail("factor-before blit failed");
        [before copyFromBuffer:(__bridge id<MTLBuffer>)
                                   pass.sourceEffectiveTangentFactor
                  sourceOffset:0u
                      toBuffer:audit.factorBefore
             destinationOffset:0u
                          size:kNv * kNv * sizeof(float)];
        [before endEncoding];

        Query malformedAddress = candidateQuery(audit, audit.base, true);
        malformedAddress.candidateQGPUAddress += sizeof(float);
        audit.malformedAddressRejected = !pass.encodeExactCandidate(
            pass.exactCandidateContext, pass, malformedAddress);
        Query malformedStride = candidateQuery(audit, audit.base, true);
        malformedStride.candidateQStride = kNq - 1u;
        audit.malformedStrideRejected = !pass.encodeExactCandidate(
            pass.exactCandidateContext, pass, malformedStride);
        Query alias = candidateQuery(audit, audit.base, true);
        alias.candidateQ = pass.qCheckpoint;
        alias.candidateQGPUAddress = pass.qCheckpointGPUAddress;
        audit.aliasRejected = !pass.encodeExactCandidate(
            pass.exactCandidateContext, pass, alias);
        if (!audit.malformedAddressRejected ||
            !audit.malformedStrideRejected || !audit.aliasRejected) {
            return audit.fail("malformed or alias candidate was admitted");
        }

        const Query base = candidateQuery(audit, audit.base, true);
        const Query plus = candidateQuery(audit, audit.plus, true);
        const Query minus = candidateQuery(audit, audit.minus, true);
        const Query bodyOnly = candidateQuery(audit, audit.bodyOnly, false);
        Query jacobianOnly = candidateQuery(
            audit, audit.jacobianOnly, true);
        jacobianOnly.accessFlags &=
            ~metalrobo::MetalNumanXHumanMatterCandidateWritePointWorld;
        jacobianOnly.pointWorld = nullptr;
        jacobianOnly.pointWorldGPUAddress = 0u;
        jacobianOnly.pointWorldStride = 0u;
        if (!pass.encodeExactCandidate(
                pass.exactCandidateContext, pass, base) ||
            !pass.encodeExactCandidate(
                pass.exactCandidateContext, pass, plus) ||
            !pass.encodeExactCandidate(
                pass.exactCandidateContext, pass, minus)) {
            return audit.fail("valid exact candidate encoding rejected");
        }
        audit.bodyOnlyAccepted = pass.encodeExactCandidate(
            pass.exactCandidateContext, pass, bodyOnly);
        audit.optionalPointWorldAccepted = pass.encodeExactCandidate(
            pass.exactCandidateContext, pass, jacobianOnly);
        if (!audit.bodyOnlyAccepted || !audit.optionalPointWorldAccepted) {
            return audit.fail("optional exact-candidate outputs were rejected");
        }

        id<MTLBlitCommandEncoder> after =
            [commandBuffer blitCommandEncoder];
        if (after == nil) return audit.fail("factor-after blit failed");
        [after copyFromBuffer:(__bridge id<MTLBuffer>)
                                  pass.sourceEffectiveTangentFactor
                 sourceOffset:0u
                     toBuffer:audit.factorAfter
            destinationOffset:0u
                         size:kNv * kNv * sizeof(float)];
        [after endEncoding];
        return true;
    }
    if (pass.phase == Phase::postDynamics) {
        id<MTLBlitCommandEncoder> publish =
            [commandBuffer blitCommandEncoder];
        if (publish == nil) return audit.fail("joint accept blit failed");
        [publish copyFromBuffer:(__bridge id<MTLBuffer>)pass.standStatuses
                   sourceOffset:0u toBuffer:audit.standStatusSnapshot
              destinationOffset:0u size:sizeof(MRNumiHumanStandStatusGPU)];
        [publish copyFromBuffer:audit.acceptJoint sourceOffset:0u
                       toBuffer:audit.joint destinationOffset:0u
                            size:sizeof(MRNumanXCoupledHumanStatusGPU)];
        [publish copyFromBuffer:audit.acceptToken sourceOffset:0u
                       toBuffer:audit.token destinationOffset:0u
                            size:MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES];
        [publish endEncoding];
        return true;
    }
    return audit.fail("unknown owner phase");
}

void abortProgram(void* raw, void*) noexcept {
    ++static_cast<CandidateAudit*>(raw)->abortCount;
}

bool acquireCandidateLease(
    void* raw,
    const metalrobo::MetalNumanXHumanMatterPrepareLease& lease
) noexcept {
    auto& audit = *static_cast<CandidateAudit*>(raw);
    if (lease.abiVersion != metalrobo::kMetalNumanXHumanMatterABIVersion ||
        lease.structSize != sizeof(lease) || lease.environmentCount != 1u ||
        lease.transactionSlot != 0u || lease.stepIndex != 0u ||
        lease.substepIndex != 0u || lease.physicsSubstepCount != 1u ||
        lease.controlStep != kControlStep ||
        lease.programFingerprint != kProgramFingerprint ||
        lease.transactionFingerprint != kTransactionFingerprint ||
        lease.linearizationEpoch != kLinearizationEpoch ||
        lease.slotGeneration != kSlotGeneration ||
        lease.preparedPhysicsStateTokens != (__bridge void*)audit.token ||
        lease.matterApplyOutcomes !=
            (__bridge void*)audit.matterApplyOutcome ||
        lease.proposals == nullptr ||
        lease.proposedPhysicsStateTokens == nullptr ||
        lease.applyActions == nullptr || lease.appliedOutcomes == nullptr ||
        lease.finalAcceptedPhysicsStateTokens == nullptr ||
        lease.publicationFences == nullptr ||
        lease.physicalPreparedEvent == nullptr ||
        lease.physicalPreparedEventValue == 0u ||
        lease.proposalEventValue <= lease.physicalPreparedEventValue ||
        lease.appliedEventValue <= lease.proposalEventValue) {
        return audit.fail("ABI4 candidate prepare lease is malformed");
    }
    if (lease.humanIOCandidate.configured()) {
        return audit.fail("pre-command candidate lease carried HumanIO publication authority");
    }
    ++audit.acquireLeaseCount;
    return true;
}

bool rejectCandidateHumanIOBind(
    void*,
    const metalrobo::MetalNumanXHumanMatterPrepareLease&,
    const metalrobo::MetalNumanXHumanIOCandidatePublicationProgram&
) noexcept {
    return false;
}

metalrobo::MetalNumanXHumanMatterPrepareLeaseDisposition
releaseCandidateLease(
    void* raw,
    const metalrobo::MetalNumanXHumanMatterPrepareLease&,
    void*,
    bool
) noexcept {
    ++static_cast<CandidateAudit*>(raw)->releaseLeaseCount;
    // This focused probe intentionally stops at physical prepare. Context
    // teardown therefore quarantines the un-applied generation rather than
    // pretending proposal/ACK/apply completed.
    return metalrobo::MetalNumanXHumanMatterPrepareLeaseDisposition::
        terminalNoTouch;
}

bool rejectCandidateApplication(
    void*,
    const metalrobo::MetalNumanXHumanMatterPrepareLease&,
    const metalrobo::MetalNumanXHumanMatterProposalView&,
    const metalrobo::MetalNumanXHumanMatterBrainPreflightView&
) noexcept {
    return false;
}

bool rejectCandidateApply(
    void*,
    const metalrobo::MetalNumanXHumanMatterPrepareLease&,
    const metalrobo::MetalNumanXHumanMatterApplyPass&
) noexcept {
    return false;
}

void abortCandidateApply(
    void*,
    const metalrobo::MetalNumanXHumanMatterPrepareLease&,
    const metalrobo::MetalNumanXHumanMatterApplyPass&
) noexcept {}

bool rejectCandidatePublication(
    void*,
    const metalrobo::MetalNumanXHumanMatterPrepareLease&,
    const metalrobo::MetalNumanXHumanMatterPublicationReservationView&
) noexcept {
    return false;
}

metalrobo::MetalNumanXHumanMatterPrepareLeaseDisposition
rejectCandidatePublicationRelease(
    void*,
    const metalrobo::MetalNumanXHumanMatterPrepareLease&,
    const metalrobo::MetalNumanXHumanMatterPublicationFenceView&
) noexcept {
    return metalrobo::MetalNumanXHumanMatterPrepareLeaseDisposition::
        terminalNoTouch;
}

void initializeAudit(CandidateAudit& audit, id<MTLDevice> device) {
    audit.device = device;
    audit.reaction = makeBuffer<float>(device, kNv, @"Matter reaction");
    audit.joint = makeBuffer<MRNumanXCoupledHumanStatusGPU>(
        device, 1u, @"joint status");
    audit.token = makeBuffer<std::uint8_t>(
        device, MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES,
        @"accepted physics token");
    audit.matterApplyOutcome = makeBuffer<
        MRNumanXHumanMatterMatterApplyOutcomeGPU>(
            device, 1u, @"unused ABI4 Matter apply outcome");
    audit.standStatusSnapshot = makeBuffer<MRNumiHumanStandStatusGPU>(
        device, 1u, @"stand status snapshot");
    audit.acceptJoint = makeBuffer<MRNumanXCoupledHumanStatusGPU>(
        device, 1u, @"accepted joint source");
    audit.acceptToken = makeBuffer<std::uint8_t>(
        device, MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES,
        @"accepted token source");
    audit.attachment = makeBuffer<MRArticulatedPointImpulseGPU>(
        device, 1u, @"Matter attachment");
    audit.factorBefore = makeBuffer<float>(
        device, kNv * kNv, @"source factor before candidate");
    audit.factorAfter = makeBuffer<float>(
        device, kNv * kNv, @"source factor after candidate");
    audit.base = makeCandidateArena(device, @"base candidate", true);
    audit.plus = makeCandidateArena(device, @"plus candidate", true);
    audit.minus = makeCandidateArena(device, @"minus candidate", true);
    audit.bodyOnly = makeCandidateArena(device, @"body-only candidate", false);
    audit.jacobianOnly = makeCandidateArena(
        device, @"Jacobian-only attachment candidate", true);

    MRNumanXCoupledHumanStatusGPU pending{};
    pending.abiVersion = MR_NUMANX_COUPLED_HUMAN_ABI_VERSION;
    pending.decision = MR_NUMANX_COUPLED_HUMAN_PENDING;
    pending.environment = 0u;
    pending.stepIndex = 0u;
    pending.humanCode = MR_NUMI_HUMAN_STAND_SUCCESS;
    pending.matterCompletedMicrosteps = 0u;
    contents<MRNumanXCoupledHumanStatusGPU>(audit.joint)[0] = pending;
    MRNumanXCoupledHumanStatusGPU accepted = pending;
    accepted.decision = MR_NUMANX_COUPLED_HUMAN_ACCEPT;
    accepted.humanCompletedSteps = 1u;
    accepted.matterCompletedMicrosteps = 1u;
    contents<MRNumanXCoupledHumanStatusGPU>(audit.acceptJoint)[0] = accepted;
    contents<std::uint64_t>(audit.acceptToken)[7] =
        0x4143434550544544ull;

    MRArticulatedPointImpulseGPU attachment{};
    attachment.bodyIndex = kFirstBody + kBodyCount - 1u;
    attachment.localPoint = f4(0.13f, -0.07f, 0.09f, 0.0f);
    contents<MRArticulatedPointImpulseGPU>(audit.attachment)[0] = attachment;

    const auto fillDelta = [] (id<MTLBuffer> buffer) {
        float* delta = contents<float>(buffer);
        for (std::uint32_t dof = 0u; dof < kNv; ++dof) {
            delta[dof] = 0.08f * std::sin(
                0.07f * static_cast<float>(dof + 1u));
        }
    };
    fillDelta(audit.base.delta);
    fillDelta(audit.plus.delta);
    fillDelta(audit.minus.delta);
    fillDelta(audit.bodyOnly.delta);
    fillDelta(audit.jacobianOnly.delta);
    contents<float>(audit.plus.delta)[kFiniteDifferenceDof] +=
        kFiniteDifferenceEpsilon;
    contents<float>(audit.minus.delta)[kFiniteDifferenceDof] -=
        kFiniteDifferenceEpsilon;

    audit.muscleSites.resize(2u);
    audit.muscleSites[0].bodyIndex = kFirstBody;
    audit.muscleSites[0].localPoint =
        f4(-0.02f, 0.0f, 0.0f, 0.0f);
    audit.muscleSites[1].bodyIndex = kFirstBody;
    audit.muscleSites[1].localPoint =
        f4(0.02f, 0.0f, 0.0f, 0.0f);
    audit.muscleRoutes.resize(2u);
    audit.muscleRoutes[0].type = MR_MUJOCO_MUSCLE_ROUTE_SITE;
    audit.muscleRoutes[0].targetIndex = 0u;
    audit.muscleRoutes[0].sideSiteIndex = MR_INVALID_INDEX;
    audit.muscleRoutes[1].type = MR_MUJOCO_MUSCLE_ROUTE_SITE;
    audit.muscleRoutes[1].targetIndex = 1u;
    audit.muscleRoutes[1].sideSiteIndex = MR_INVALID_INDEX;
    audit.muscles.resize(1u);
    audit.muscles[0].route = {0u, 2u, 0u, 0u};
    audit.muscles[0].lengthRangeAndAcceleration =
        {0.02f, 0.08f, 1.0f, 0.0f};
    audit.muscles[0].controlRange = {0.0f, 1.0f, 0.0f, 0.0f};
    audit.muscleStates.resize(1u);
}

metalrobo::MetalArticulatedOperatorInput makeInput(
    const metalrobo::EngineModel& model,
    const std::vector<MRArticulatedPointImpulseGPU>& points,
    CandidateAudit& audit
) {
    metalrobo::MetalArticulatedOperatorInput input{
        .articulationIndex = 0u,
        .environmentCount = kEnvironmentCount,
        .pointCount = points.size(),
        .q = model.defaultQ,
        .v = model.defaultV,
        .points = points,
        .mujoco = {
            .muscles = audit.muscles,
            .states = audit.muscleStates,
            .sites = audit.muscleSites,
            .wraps = {},
            .routeNodes = audit.muscleRoutes,
            .bodyJacobianPointOffset = 0u,
        },
        .stand = {
            .v = model.defaultV,
            .contacts = {},
            .jointEqualities = {},
            .tendonBindings = {},
            .tendonEnvelopes = {},
            .tendonLoadProgram = {},
            .numanXTransactionProgram = {},
            .numanXHumanMatterProgram = {
                .capabilities =
                    metalrobo::MetalNumanXHumanMatterExactCandidateKinematics |
                    metalrobo::MetalNumanXHumanMatterSourceEffectiveTangent |
                    metalrobo::MetalNumanXHumanMatterStagedReaction |
                    metalrobo::MetalNumanXHumanMatterJointDecision |
                    metalrobo::MetalNumanXHumanMatterPreparedPhysicsGate,
                .accessFlags =
                    metalrobo::MetalNumanXHumanMatterReadLiveHumanState |
                    metalrobo::MetalNumanXHumanMatterReadHumanCheckpoints |
                    metalrobo::MetalNumanXHumanMatterReadSourceEffectiveTangent |
                    metalrobo::MetalNumanXHumanMatterMayEncodeExactCandidate |
                    metalrobo::MetalNumanXHumanMatterWriteStagedReaction |
                    metalrobo::MetalNumanXHumanMatterWriteJointStatus |
                    metalrobo::MetalNumanXHumanMatterWritePreparedPhysicsToken,
                .context = &audit,
                .encode = &encodeProgram,
                .abort = &abortProgram,
                .acquirePrepareLease = &acquireCandidateLease,
                .bindHumanIOCandidatePublication =
                    &rejectCandidateHumanIOBind,
                .releasePrepareLease = &releaseCandidateLease,
                .reservePreparedApplication = &rejectCandidateApplication,
                .encodePreparedApply = &rejectCandidateApply,
                .abortPreparedApply = &abortCandidateApply,
                .reservePublishedRoot = &rejectCandidatePublication,
                .releasePublishedRoot =
                    &rejectCandidatePublicationRelease,
                .fingerprint = kProgramFingerprint,
                .matterGeneralizedReaction = (__bridge void*)audit.reaction,
                .jointStatuses = (__bridge void*)audit.joint,
                .acceptedPhysicsStateTokens = (__bridge void*)audit.token,
                .matterApplyOutcomes =
                    (__bridge void*)audit.matterApplyOutcome,
                .matterGeneralizedReactionGPUAddress =
                    audit.reaction.gpuAddress,
                .jointStatusesGPUAddress = audit.joint.gpuAddress,
                .acceptedPhysicsStateTokensGPUAddress =
                    audit.token.gpuAddress,
                .matterApplyOutcomesGPUAddress =
                    audit.matterApplyOutcome.gpuAddress,
                .matterGeneralizedReactionElementCount = kNv,
                .jointStatusElementCount = 1u,
                .acceptedPhysicsStateTokenByteCount =
                    MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES,
                .matterApplyOutcomeElementCount = 1u,
                .environmentCount = kEnvironmentCount,
                .reactionStride = kNv,
                .jointStatusStride = 1u,
                .acceptedTokenStrideBytes =
                    MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES,
                .matterApplyOutcomeStride = 1u,
                .transactionSlot = 0u,
                .substepIndex = 0u,
                .physicsSubstepCount = 1u,
                .candidatePointCapacity = 1u,
                .controlStep = kControlStep,
                .qCoordinateCount = kNq,
                .dofCount = kNv,
                .dofLayoutVersion =
                    metalrobo::kMetalNumanXHumanMatterDofLayoutVersion,
                .transactionFingerprint = kTransactionFingerprint,
                .linearizationEpoch = kLinearizationEpoch,
                .slotGeneration = kSlotGeneration,
            },
            .stepCount = 1u,
            .contactIterationCount = 1u,
            .enableContact = false,
            .enableRootAssistance = false,
        },
    };
    return input;
}

float maximumFactorDifference(const CandidateAudit& audit) {
    const float* before = contents<float>(audit.factorBefore);
    const float* after = contents<float>(audit.factorAfter);
    float maximum = 0.0f;
    for (std::size_t index = 0u; index < kNv * kNv; ++index) {
        maximum = std::max(maximum, std::abs(before[index] - after[index]));
    }
    return maximum;
}

void verifyCandidate(
    const metalrobo::EngineModel& model,
    const CandidateAudit& audit
) {
    require(audit.failure == nullptr,
            audit.failure == nullptr ? "" : audit.failure);
    require(audit.phaseCount == 3u && audit.abortCount == 0u &&
                audit.acquireLeaseCount == 1u &&
                audit.releaseLeaseCount == 0u,
            "owner phase/abort count mismatch");
    require(audit.malformedAddressRejected &&
                audit.malformedStrideRejected && audit.aliasRejected &&
                audit.bodyOnlyAccepted && audit.optionalPointWorldAccepted,
            "candidate admission evidence is incomplete");
    require(std::isfinite(
                contents<float>(audit.jacobianOnly.pointJacobians)[
                    kFiniteDifferenceDof]),
            "Jacobian-only attachment candidate was not materialized");
    const auto& stand = contents<MRNumiHumanStandStatusGPU>(
        audit.standStatusSnapshot)[0];
    require(stand.code ==
                    MR_NUMI_HUMAN_STAND_SUCCESS &&
                stand.completedSteps == 1u,
            "Human stand did not accept the coupled step");
    require(maximumFactorDifference(audit) == 0.0f,
            "candidate kinematics modified frozen A0 bytes");

    const float* delta = contents<float>(audit.base.delta);
    const float* q = contents<float>(audit.base.q);
    for (std::uint32_t dof = 6u; dof < kNv; ++dof) {
        const std::uint32_t qIndex = dof + 1u;
        const float expected = model.defaultQ[qIndex] + kTimestep *
            (model.defaultV[dof] + delta[dof]);
        require(std::abs(q[qIndex] - expected) <= 2.0e-6f,
                "candidate scalar q integration is not exact");
    }
    require(std::abs(q[0] - (model.defaultQ[0] + kTimestep *
                (model.defaultV[0] + delta[0]))) <= 2.0e-6f,
            "candidate root translation is not exact");

    const MRBodyStateGPU* bodies = contents<MRBodyStateGPU>(audit.base.bodies);
    const MRBodyStateGPU& distal = bodies[kWorldBodyCount - 1u];
    require(distal.flagsAndIndices[0] == MR_MOTION_DYNAMIC &&
                distal.flagsAndIndices[1] == 0u &&
                distal.flagsAndIndices[2] == kWorldBodyCount - 1u &&
                std::isfinite(distal.position.x) &&
                std::isfinite(distal.linearVelocityAndInverseMass.x) &&
                std::isfinite(distal.angularVelocity.z) &&
                distal.linearVelocityAndInverseMass.w == 0.0f,
            "candidate MRBodyStateGPU materialization is malformed");

    const MRArticulatedPointWorldGPU baseWorld =
        contents<MRArticulatedPointWorldGPU>(audit.base.pointWorld)[0];
    const MRArticulatedPointWorldGPU plusWorld =
        contents<MRArticulatedPointWorldGPU>(audit.plus.pointWorld)[0];
    const MRArticulatedPointWorldGPU minusWorld =
        contents<MRArticulatedPointWorldGPU>(audit.minus.pointWorld)[0];
    const float* jacobian = contents<float>(audit.base.pointJacobians);
    require(std::isfinite(baseWorld.position.x) &&
                std::isfinite(baseWorld.position.y) &&
                std::isfinite(baseWorld.position.z) &&
                baseWorld.position.w == 1.0f,
            "candidate point-world materialization is malformed");
    const std::array<float, 3u> finiteDifference{{
        (plusWorld.position.x - minusWorld.position.x) /
            (2.0f * kFiniteDifferenceEpsilon * kTimestep),
        (plusWorld.position.y - minusWorld.position.y) /
            (2.0f * kFiniteDifferenceEpsilon * kTimestep),
        (plusWorld.position.z - minusWorld.position.z) /
            (2.0f * kFiniteDifferenceEpsilon * kTimestep),
    }};
    float maximumError = 0.0f;
    float maximumMagnitude = 0.0f;
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        const float analytic = jacobian[axis * kNv + kFiniteDifferenceDof];
        maximumError = std::max(
            maximumError, std::abs(finiteDifference[axis] - analytic));
        maximumMagnitude = std::max(maximumMagnitude, std::abs(analytic));
    }
    require(maximumMagnitude > 1.0e-4f && maximumError < 3.0e-2f,
            "nonlinear attachment finite difference rejected analytic J: " +
                std::to_string(maximumError) + " magnitude=" +
                std::to_string(maximumMagnitude) + " fd=" +
                std::to_string(finiteDifference[0]) + "," +
                std::to_string(finiteDifference[1]) + "," +
                std::to_string(finiteDifference[2]) + " J=" +
                std::to_string(jacobian[kFiniteDifferenceDof]) + "," +
                std::to_string(jacobian[kNv + kFiniteDifferenceDof]) + "," +
                std::to_string(jacobian[2u * kNv + kFiniteDifferenceDof]));
}

} // namespace

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        try {
            require(argc == 2,
                    "usage: numanx_human_matter_candidate_probe <metallib>");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "NumanX candidate probe requires Metal");
            metalrobo::EngineModel model = makeHumanModel();
            const std::vector<MRArticulatedPointImpulseGPU> points =
                bodyProbes();
            CandidateAudit audit;
            initializeAudit(audit, device);
            const auto input = makeInput(model, points, audit);
            const metalrobo::MetalArticulatedOperatorConfig config{
                .pointJacobiansOnly = true,
                .mujocoActivationTimestepSeconds = kTimestep,
                .metallibPath = argv[1],
            };
            metalrobo::MetalArticulatedOperatorContext context(config);
            metalrobo::MetalArticulatedOperatorSubmission submission;
            const auto diagnostics = context.submit(model, input, submission);
            require(diagnostics.succeeded() && diagnostics.dispatched &&
                        submission.valid(),
                    "exact candidate owner submit failed: " +
                        diagnostics.message);
            metalrobo::MetalNumanXHumanMatterPrepared prepared;
            require(submission.extractPreparedHumanMatter(prepared) &&
                        !submission.valid() && prepared.valid(),
                    "exact candidate submission did not enter ABI4 quarantine");
            id<MTLCommandQueue> waitQueue = [device newCommandQueue];
            id<MTLCommandBuffer> waitCommand = [waitQueue commandBuffer];
            require(waitQueue != nil && waitCommand != nil &&
                        prepared.encodeWaitForPhysicalPrepare(
                            (__bridge void*)waitCommand),
                    "exact candidate physical-prepare wait encode failed");
            [waitCommand commit];
            [waitCommand waitUntilCompleted];
            require(waitCommand.status == MTLCommandBufferStatusCompleted,
                    "exact candidate physical-prepare wait failed");
            verifyCandidate(model, audit);
            std::cout
                << "PASS device=\"" << diagnostics.deviceName
                << "\" dofs=160 q=161 exact_candidate=generic_analytic"
                << " nonlinear_fd=passed malformed=fail_closed"
                << " alias=fail_closed point_world=materialized"
                << " body_only=accepted A0=frozen\n";
            return 0;
        } catch (const std::exception& exception) {
            std::cerr << "numanx_human_matter_candidate_probe: "
                      << exception.what() << '\n';
            return 1;
        }
    }
}
