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
#include <limits>
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
constexpr double kSourceDiagonalRelativeTolerance = 2.0e-4;
constexpr double kSourceForceClosureRelativeTolerance = 2.0e-5;
constexpr double kSourcePredictorAbsoluteTolerance = 2.0e-5;
constexpr double kSourceRHSULPAllowance = 8.0;

using Phase = metalrobo::MetalNumanXHumanMatterPhase;
using Pass = metalrobo::MetalNumanXHumanMatterPass;
using Query = metalrobo::MetalNumanXHumanMatterCandidateQuery;

static_assert(
    metalrobo::kMetalNumanXHumanMatterPassABIVersionV6 == 6u &&
    metalrobo::kMetalNumanXHumanMatterPassABIVersion == 7u);
static_assert(offsetof(Pass, sourceDynamicsWitness) == 672u,
    "v7 source witness must be a strict tail addition to the 672-byte v6 pass");
static_assert(sizeof(Pass) == 704u,
    "unexpected MetalNumanXHumanMatterPass v7 layout");

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void accumulateBoundRatio(
    const double error,
    const double scale,
    const double tolerance,
    double& maximumRatio
) {
    require(std::isfinite(error) && error >= 0.0 &&
                std::isfinite(scale) && scale >= 0.0 &&
                std::isfinite(tolerance) && tolerance > 0.0,
            "source-dynamics bound input is invalid");
    maximumRatio = std::max(
        maximumRatio, error / (tolerance * (1.0 + scale)));
}

double sourceResidualGamma(const std::size_t dimension) {
    const double operationError =
        (static_cast<double>(dimension) + 2.0) *
        std::numeric_limits<float>::epsilon();
    require(operationError < 1.0,
            "source residual dimension exceeds FP32 audit bound");
    return operationError / (1.0 - operationError);
}

double floatULP(const float value) {
    require(std::isfinite(value), "FP32 ULP input is nonfinite");
    const float magnitude = std::abs(value);
    const float next = std::nextafterf(
        magnitude, std::numeric_limits<float>::infinity());
    if (std::isfinite(next)) {
        return static_cast<double>(next) - magnitude;
    }
    const float previous = std::nextafterf(magnitude, 0.0f);
    return static_cast<double>(magnitude) - previous;
}

double sourceResidualBound(
    const std::size_t dimension,
    const float rhs,
    const double productScale
) {
    require(std::isfinite(rhs) && std::isfinite(productScale) &&
                productScale >= 0.0,
            "source residual bound input is invalid");
    return sourceResidualGamma(dimension) * productScale +
        kSourceRHSULPAllowance * floatULP(rhs);
}

void accumulateAbsoluteBoundRatio(
    const double error,
    const double bound,
    double& maximumRatio
) {
    require(std::isfinite(error) && error >= 0.0 &&
                std::isfinite(bound) && bound >= 0.0,
            "source-dynamics absolute bound input is invalid");
    const double ratio = bound > 0.0
        ? error / bound
        : (error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
    maximumRatio = std::max(maximumRatio, ratio);
}

void verifyHeterogeneousScaleGate() {
    // A large, well-closed row must never donate its scale to a corrupted
    // zero/small row. This specifically guards against the old
    // max(error) <= tolerance * max(scale) formulation.
    const std::array<double, 2u> errors{{0.01, 0.0}};
    const std::array<double, 2u> scales{{0.0, 1.0e8}};
    for (const double tolerance : {
             kSourceDiagonalRelativeTolerance,
             kSourceForceClosureRelativeTolerance}) {
        double maximumRatio = 0.0;
        for (std::size_t row = 0u; row < errors.size(); ++row) {
            accumulateBoundRatio(
                errors[row], scales[row], tolerance, maximumRatio);
        }
        require(maximumRatio > 1.0,
                "heterogeneous-scale source corruption escaped its row bound");
    }
    double residualRatio = 0.0;
    accumulateAbsoluteBoundRatio(
        errors[0], sourceResidualBound(errors.size(), 0.0f, scales[0]),
        residualRatio);
    accumulateAbsoluteBoundRatio(
        errors[1], sourceResidualBound(errors.size(), 1.0e8f, scales[1]),
        residualRatio);
    require(residualRatio > 1.0,
            "heterogeneous-scale residual corruption escaped its row bound");
    const double maximumFiniteRHSBound = sourceResidualBound(
        errors.size(), std::numeric_limits<float>::max(), 0.0);
    require(std::isfinite(maximumFiniteRHSBound) &&
                maximumFiniteRHSBound > 0.0,
            "maximum finite RHS produced an unbounded residual allowance");
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
    id<MTLBuffer> rootTranslation = nil;
    id<MTLBuffer> bodyPositionLow = nil;
    id<MTLBuffer> pointPositionLow = nil;
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
    result.rootTranslation = makeBuffer<MRCompensatedRootTranslationGPU>(
        device, kEnvironmentCount, [label stringByAppendingString:@" root translation"]);
    result.bodyPositionLow = makeBuffer<mr_float4>(device, kWorldBodyCount,
        [label stringByAppendingString:@" body position low"]);
    const float poison = std::numeric_limits<float>::quiet_NaN();
    std::fill_n(contents<mr_float4>(result.bodyPositionLow), kWorldBodyCount,
                f4(poison, poison, poison, poison));
    if (withPoint) {
        result.pointPositionLow = makeBuffer<mr_float4>(device, 1u,
            [label stringByAppendingString:@" point position low"]);
        contents<mr_float4>(result.pointPositionLow)[0] = f4(poison, poison, poison, poison);
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
    id<MTLBuffer> predictorSnapshot = nil;
    id<MTLBuffer> sourceDynamicsBefore = nil;
    id<MTLBuffer> sourceDynamicsAfter = nil;
    id<MTLBuffer> sourceForceSnapshot = nil;
    id<MTLBuffer> finalQSnapshot = nil;
    id<MTLBuffer> finalVSnapshot = nil;
    id<MTLBuffer> shortRootTranslation = nil;
    id<MTLBuffer> shortBodyPositionLow = nil;
    id<MTLBuffer> shortPointPositionLow = nil;
    std::uint32_t companionNegativeCount = 0u;
    CandidateArena freeMotion{};
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
    bool witnessAliasRejected = false;
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
    query.candidateRootTranslation = (__bridge void*)arena.rootTranslation;
    query.candidateRootTranslationGPUAddress = arena.rootTranslation.gpuAddress;
    query.candidateRootTranslationElementCount = kEnvironmentCount;
    query.candidateBodyPositionLow = (__bridge void*)arena.bodyPositionLow;
    query.candidateBodyPositionLowGPUAddress = arena.bodyPositionLow.gpuAddress;
    query.candidateBodyPositionLowElementCount = kWorldBodyCount;
    if (withPoint) {
        query.accessFlags |=
            metalrobo::MetalNumanXHumanMatterCandidateReadPointQueries |
            metalrobo::MetalNumanXHumanMatterCandidateWritePointWorld |
            metalrobo::MetalNumanXHumanMatterCandidateWritePointJacobians;
        query.pointQueries = (__bridge void*)audit.attachment;
        query.pointWorld = (__bridge void*)arena.pointWorld;
        query.pointPositionLow = (__bridge void*)arena.pointPositionLow;
        query.pointPositionLowGPUAddress = arena.pointPositionLow.gpuAddress;
        query.pointPositionLowElementCount = 1u;
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
        pass.controlStep != kControlStep ||
        (pass.capabilities &
         metalrobo::MetalNumanXHumanMatterSourceDynamicsWitness) == 0u ||
        (pass.accessFlags &
         metalrobo::MetalNumanXHumanMatterReadSourceDynamicsWitness) == 0u ||
        pass.sourceDynamicsWitness == nullptr ||
        pass.sourceDynamicsWitnessGPUAddress !=
            [(__bridge id<MTLBuffer>)pass.sourceDynamicsWitness gpuAddress] ||
        pass.sourceDynamicsWitnessElementCount != 3u * kNv ||
        pass.sourceDynamicsWitnessStride != 3u * kNv) {
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
        [before copyFromBuffer:(__bridge id<MTLBuffer>)pass.sourcePredictedVelocity
            sourceOffset:0u toBuffer:audit.predictorSnapshot destinationOffset:0u
            size:kNv * sizeof(float)];
        [before copyFromBuffer:(__bridge id<MTLBuffer>)pass.sourceDynamicsWitness
            sourceOffset:0u toBuffer:audit.sourceDynamicsBefore
            destinationOffset:0u size:3u * kNv * sizeof(float)];
        [before copyFromBuffer:(__bridge id<MTLBuffer>)
                                   pass.mujocoGeneralizedForceArena
            sourceOffset:pass.generalizedForceOffset * sizeof(float)
            toBuffer:audit.sourceForceSnapshot destinationOffset:0u
            size:kNv * sizeof(float)];
        [before endEncoding];

        Pass aliasPredictor = pass;
        aliasPredictor.sourcePredictedVelocity = pass.vCheckpoint;
        aliasPredictor.sourcePredictedVelocityGPUAddress = pass.vCheckpointGPUAddress;
        if (pass.encodeExactCandidate(pass.exactCandidateContext, aliasPredictor,
                candidateQuery(audit, audit.freeMotion, false)))
            return audit.fail("checkpoint alias admitted as the free predictor");
        Pass aliasWitness = pass;
        aliasWitness.sourceDynamicsWitness =
            pass.sourceEffectiveTangentFactor;
        aliasWitness.sourceDynamicsWitnessGPUAddress =
            pass.sourceEffectiveTangentFactorGPUAddress;
        audit.witnessAliasRejected = !pass.encodeExactCandidate(
            pass.exactCandidateContext, aliasWitness,
            candidateQuery(audit, audit.freeMotion, false));
        if (!audit.witnessAliasRejected)
            return audit.fail("source factor alias admitted as dynamics witness");
        Pass staleABI = pass;
        staleABI.abiVersion = metalrobo::kMetalNumanXHumanMatterABIVersion;
        if (pass.encodeExactCandidate(pass.exactCandidateContext, staleABI,
                candidateQuery(audit, audit.freeMotion, false)))
            return audit.fail("legacy predictor-free pass admitted");
        if (!pass.encodeExactCandidate(pass.exactCandidateContext, pass,
                candidateQuery(audit, audit.freeMotion, false)))
            return audit.fail("free-motion candidate was rejected");

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

        // Each companion is independently admitted as a complete, correctly
        // sized, non-aliasing output arena before any candidate dispatch.
        struct CompanionFields {
            void* Query::* buffer;
            std::uint64_t Query::* address;
            std::uint64_t Query::* count;
            id<MTLBuffer> shortBuffer;
            void* aliasBuffer;
            std::uint64_t aliasAddress;
            const char* missingMessage;
            const char* countMessage;
            const char* addressMessage;
            const char* extentMessage;
            const char* aliasMessage;
        };
        const std::array<CompanionFields, 3u> companions{{
            {&Query::candidateRootTranslation, &Query::candidateRootTranslationGPUAddress,
             &Query::candidateRootTranslationElementCount, audit.shortRootTranslation,
             pass.rootTranslationCheckpoint, pass.rootTranslationCheckpointGPUAddress,
             "missing root companion admitted", "wrong root companion count admitted",
             "wrong root companion address admitted", "short root companion admitted",
             "checkpoint root substituted as candidate output"},
            {&Query::candidateBodyPositionLow, &Query::candidateBodyPositionLowGPUAddress,
             &Query::candidateBodyPositionLowElementCount, audit.shortBodyPositionLow,
             (__bridge void*)audit.base.bodies, audit.base.bodies.gpuAddress,
             "missing body-low companion admitted", "wrong body-low count admitted",
             "wrong body-low address admitted", "short body-low companion admitted",
             "body high substituted as body-low output"},
            {&Query::pointPositionLow, &Query::pointPositionLowGPUAddress,
             &Query::pointPositionLowElementCount, audit.shortPointPositionLow,
             (__bridge void*)audit.base.pointWorld, audit.base.pointWorld.gpuAddress,
             "missing point-low companion admitted", "wrong point-low count admitted",
             "wrong point-low address admitted", "short point-low companion admitted",
             "point high substituted as point-low output"},
        }};
        const auto rejectCompanion = [&](const Query& query, const char* message) noexcept {
            if (pass.encodeExactCandidate(pass.exactCandidateContext, pass, query))
                return audit.fail(message);
            ++audit.companionNegativeCount;
            return true;
        };
        for (std::size_t index = 0u; index < companions.size(); ++index) {
            const auto& fields = companions[index];
            Query changed = candidateQuery(audit, audit.base, true);
            changed.*(fields.buffer) = nullptr;
            changed.*(fields.address) = 0u;
            changed.*(fields.count) = 0u;
            if (!rejectCompanion(changed, fields.missingMessage)) return false;
            changed = candidateQuery(audit, audit.base, true);
            changed.*(fields.count) = index == 0u ? kEnvironmentCount + 1u
                                                  : changed.*(fields.count) - 1u;
            if (!rejectCompanion(changed, fields.countMessage)) return false;
            changed = candidateQuery(audit, audit.base, true);
            changed.*(fields.address) += sizeof(float);
            if (!rejectCompanion(changed, fields.addressMessage)) return false;
            changed = candidateQuery(audit, audit.base, true);
            changed.*(fields.buffer) = (__bridge void*)fields.shortBuffer;
            changed.*(fields.address) = fields.shortBuffer.gpuAddress;
            if (!rejectCompanion(changed, fields.extentMessage)) return false;
            changed = candidateQuery(audit, audit.base, true);
            changed.*(fields.buffer) = fields.aliasBuffer;
            changed.*(fields.address) = fields.aliasAddress;
            if (!rejectCompanion(changed, fields.aliasMessage)) return false;
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
        jacobianOnly.pointPositionLow = nullptr;
        jacobianOnly.pointPositionLowGPUAddress = 0u;
        jacobianOnly.pointPositionLowElementCount = 0u;
        Query stalePointLow = jacobianOnly;
        stalePointLow.pointPositionLow = (__bridge void*)audit.jacobianOnly.pointPositionLow;
        if (!rejectCompanion(stalePointLow, "unused point-low pointer admitted")) return false;
        stalePointLow = jacobianOnly;
        stalePointLow.pointPositionLowGPUAddress = audit.jacobianOnly.pointPositionLow.gpuAddress;
        if (!rejectCompanion(stalePointLow, "unused point-low address admitted")) return false;
        stalePointLow = jacobianOnly;
        stalePointLow.pointPositionLowElementCount = 1u;
        if (!rejectCompanion(stalePointLow, "unused point-low count admitted")) return false;
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
        [publish copyFromBuffer:(__bridge id<MTLBuffer>)pass.q sourceOffset:0u
            toBuffer:audit.finalQSnapshot destinationOffset:0u size:kNq * sizeof(float)];
        [publish copyFromBuffer:(__bridge id<MTLBuffer>)pass.v sourceOffset:0u
            toBuffer:audit.finalVSnapshot destinationOffset:0u size:kNv * sizeof(float)];
        [publish copyFromBuffer:(__bridge id<MTLBuffer>)pass.standStatuses
                   sourceOffset:0u toBuffer:audit.standStatusSnapshot
              destinationOffset:0u size:sizeof(MRNumiHumanStandStatusGPU)];
        [publish copyFromBuffer:(__bridge id<MTLBuffer>)
                                    pass.sourceDynamicsWitness
                   sourceOffset:0u toBuffer:audit.sourceDynamicsAfter
              destinationOffset:0u size:3u * kNv * sizeof(float)];
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
    audit.shortRootTranslation = makeBuffer<std::uint8_t>(device,
        sizeof(MRCompensatedRootTranslationGPU) - 1u, @"short root companion");
    audit.shortBodyPositionLow = makeBuffer<std::uint8_t>(device,
        kWorldBodyCount * sizeof(mr_float4) - 1u, @"short body-low companion");
    audit.shortPointPositionLow = makeBuffer<std::uint8_t>(device,
        sizeof(mr_float4) - 1u, @"short point-low companion");
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

    audit.freeMotion = makeCandidateArena(device, @"zero-delta candidate", false);
    audit.predictorSnapshot = makeBuffer<float>(device, kNv, @"free predictor snapshot");
    audit.sourceDynamicsBefore = makeBuffer<float>(
        device, 3u * kNv, @"source dynamics before physical stand");
    audit.sourceDynamicsAfter = makeBuffer<float>(
        device, 3u * kNv, @"source dynamics after physical stand");
    audit.sourceForceSnapshot = makeBuffer<float>(
        device, kNv, @"source generalized force snapshot");
    audit.finalQSnapshot = makeBuffer<float>(device, kNq, @"actual Human q");
    audit.finalVSnapshot = makeBuffer<float>(device, kNv, @"actual Human v");

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
                    metalrobo::MetalNumanXHumanMatterSourceDynamicsWitness |
                    metalrobo::MetalNumanXHumanMatterStagedReaction |
                    metalrobo::MetalNumanXHumanMatterJointDecision |
                    metalrobo::MetalNumanXHumanMatterPreparedPhysicsGate,
                .accessFlags =
                    metalrobo::MetalNumanXHumanMatterReadLiveHumanState |
                    metalrobo::MetalNumanXHumanMatterReadHumanCheckpoints |
                    metalrobo::MetalNumanXHumanMatterReadSourceEffectiveTangent |
                    metalrobo::MetalNumanXHumanMatterReadSourceDynamicsWitness |
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

void verifySourceDynamicsWitness(
    const metalrobo::EngineModel& model,
    const CandidateAudit& audit
) {
    require(std::memcmp(
                audit.sourceDynamicsBefore.contents,
                audit.sourceDynamicsAfter.contents,
                3u * kNv * sizeof(float)) == 0,
            "source-dynamics witness changed after preDynamics");
    const float* witness = contents<float>(audit.sourceDynamicsBefore);
    const float* diagonal = witness;
    const float* bias = witness + kNv;
    const float* rhs = witness + 2u * kNv;
    const float* factor = contents<float>(audit.factorBefore);
    const float* sourceForce = contents<float>(audit.sourceForceSnapshot);
    const float* predictor = contents<float>(audit.predictorSnapshot);

    std::vector<double> forward(kNv, 0.0);
    std::vector<double> acceleration(kNv, 0.0);
    double maximumDiagonalBoundRatio = 0.0;
    double maximumForceClosureBoundRatio = 0.0;
    double witnessMagnitude = 0.0;
    for (std::uint32_t row = 0u; row < kNv; ++row) {
        require(std::isfinite(diagonal[row]) && std::isfinite(bias[row]) &&
                    std::isfinite(rhs[row]),
                "source-dynamics witness contains nonfinite values");
        witnessMagnitude = std::max(witnessMagnitude,
            std::max({std::abs(static_cast<double>(diagonal[row])),
                      std::abs(static_cast<double>(bias[row])),
                      std::abs(static_cast<double>(rhs[row]))}));
        double reconstructedDiagonal = 0.0;
        for (std::uint32_t column = 0u; column <= row; ++column) {
            const double value = factor[row * kNv + column];
            reconstructedDiagonal += value * value;
        }
        const double diagonalError =
            std::abs(reconstructedDiagonal - diagonal[row]);
        const double diagonalScale = std::max(
            std::abs(reconstructedDiagonal),
            std::abs(static_cast<double>(diagonal[row])));
        accumulateBoundRatio(
            diagonalError, diagonalScale,
            kSourceDiagonalRelativeTolerance,
            maximumDiagonalBoundRatio);
        const double forceClosure =
            std::abs(static_cast<double>(bias[row]) + rhs[row] -
                     sourceForce[row]);
        const double forceScale =
            std::abs(static_cast<double>(bias[row])) +
            std::abs(static_cast<double>(rhs[row])) +
            std::abs(static_cast<double>(sourceForce[row]));
        accumulateBoundRatio(
            forceClosure, forceScale,
            kSourceForceClosureRelativeTolerance,
            maximumForceClosureBoundRatio);
    }
    require(witnessMagnitude > 0.0, "source-dynamics witness remained zero");
    require(maximumDiagonalBoundRatio <= 1.0,
            "A0 diagonal witness does not close against source Cholesky: " +
                std::to_string(maximumDiagonalBoundRatio));
    require(maximumForceClosureBoundRatio <= 1.0,
            "raw bias/RHS witness does not close against source force: " +
                std::to_string(maximumForceClosureBoundRatio));

    for (std::uint32_t row = 0u; row < kNv; ++row) {
        double value = rhs[row];
        for (std::uint32_t column = 0u; column < row; ++column) {
            value -= factor[row * kNv + column] * forward[column];
        }
        const double pivot = factor[row * kNv + row];
        require(std::isfinite(pivot) && pivot > 0.0,
                "source Cholesky has an invalid pivot");
        forward[row] = value / pivot;
    }
    for (std::uint32_t reverse = 0u; reverse < kNv; ++reverse) {
        const std::uint32_t row = kNv - 1u - reverse;
        double value = forward[row];
        for (std::uint32_t column = row + 1u; column < kNv; ++column) {
            value -= factor[column * kNv + row] * acceleration[column];
        }
        acceleration[row] = value / factor[row * kNv + row];
    }

    double maximumResidualBoundRatio = 0.0;
    double maximumPredictorBoundRatio = 0.0;
    for (std::uint32_t row = 0u; row < kNv; ++row) {
        double product = 0.0;
        double productScale = 0.0;
        for (std::uint32_t column = 0u; column < kNv; ++column) {
            const double a0 = row == column
                ? diagonal[row]
                : factor[std::min(row, column) * kNv +
                         std::max(row, column)];
            product += a0 * acceleration[column];
            productScale += std::abs(a0 * acceleration[column]);
        }
        accumulateAbsoluteBoundRatio(
            std::abs(product - rhs[row]),
            sourceResidualBound(kNv, rhs[row], productScale),
            maximumResidualBoundRatio);
        const double expectedVelocity = model.defaultV[row] +
            static_cast<double>(kTimestep) * acceleration[row];
        accumulateBoundRatio(
            std::abs(expectedVelocity - predictor[row]), 0.0,
            kSourcePredictorAbsoluteTolerance,
            maximumPredictorBoundRatio);
    }
    require(maximumResidualBoundRatio <= 1.0,
            "source A0/RHS witness fails the CPU residual oracle: " +
                std::to_string(maximumResidualBoundRatio));
    require(maximumPredictorBoundRatio <= 1.0,
            "source witness solve does not reproduce free predictor: " +
                std::to_string(maximumPredictorBoundRatio));
    std::cout << "SOURCE_DYNAMICS immutable=yes diagonal_bound_ratio="
              << maximumDiagonalBoundRatio << " force_closure_bound_ratio="
              << maximumForceClosureBoundRatio << " residual_bound_ratio="
              << maximumResidualBoundRatio << " predictor_bound_ratio="
              << maximumPredictorBoundRatio << '\n';
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
                audit.witnessAliasRejected &&
                audit.bodyOnlyAccepted && audit.optionalPointWorldAccepted,
            "candidate admission evidence is incomplete");
    require(audit.companionNegativeCount == 18u,
            "compensated candidate companion controls are incomplete");
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
    verifySourceDynamicsWitness(model, audit);

    const float* predictor = contents<float>(audit.predictorSnapshot);
    const float* actualV = contents<float>(audit.finalVSnapshot);
    const float* actualQ = contents<float>(audit.finalQSnapshot);
    const float* freeQ = contents<float>(audit.freeMotion.q);
    float predictorChange = 0.0f;
    float velocityClosure = 0.0f;
    float positionClosure = 0.0f;
    for (std::uint32_t dof = 0u; dof < kNv; ++dof) {
        predictorChange = std::max(predictorChange, std::abs(predictor[dof] - model.defaultV[dof]));
        velocityClosure = std::max(velocityClosure, std::abs(predictor[dof] - actualV[dof]));
    }
    for (std::uint32_t qIndex = 0u; qIndex < kNq; ++qIndex)
        positionClosure = std::max(positionClosure, std::abs(freeQ[qIndex] - actualQ[qIndex]));
    float legacyPositionMismatch = 0.0f;
    for (std::uint32_t axis = 0u; axis < 3u; ++axis)
        legacyPositionMismatch = std::max(legacyPositionMismatch,
            std::abs(model.defaultQ[axis] + kTimestep * model.defaultV[axis] - actualQ[axis]));
    require(legacyPositionMismatch > 2.0e-6f,
            "legacy v0 candidate is not a discriminating negative control");
    require(predictorChange > 1.0e-4f,
            "fixture did not exercise source Human acceleration");
    require(velocityClosure <= 2.0e-6f && positionClosure <= 2.0e-6f,
            "free candidate differs from the actual Human step: q=" +
                std::to_string(positionClosure) + " v=" + std::to_string(velocityClosure));
    std::cout << "PREDICTOR source_acceleration_dv=" << predictorChange
              << " q_closure=" << positionClosure << " v_closure=" << velocityClosure
              << " legacy_q_mismatch=" << legacyPositionMismatch << '\n';

    const float* delta = contents<float>(audit.base.delta);
    const float* q = contents<float>(audit.base.q);
    for (std::uint32_t dof = 6u; dof < kNv; ++dof) {
        const std::uint32_t qIndex = dof + 1u;
        const float expected = model.defaultQ[qIndex] + kTimestep *
            (predictor[dof] + delta[dof]);
        require(std::abs(q[qIndex] - expected) <= 2.0e-6f,
                "candidate scalar q integration is not exact");
    }
    require(std::abs(q[0] - (model.defaultQ[0] + kTimestep *
                (predictor[0] + delta[0]))) <= 2.0e-6f,
            "candidate root translation is not exact");

    const auto validLow = [](const mr_float4& low) {
        return std::isfinite(low.x) && std::isfinite(low.y) &&
               std::isfinite(low.z) && low.w == 0.0f;
    };
    for (const CandidateArena* arena : {&audit.freeMotion, &audit.base, &audit.plus,
                                      &audit.minus, &audit.bodyOnly, &audit.jacobianOnly}) {
        const auto& root = contents<MRCompensatedRootTranslationGPU>(arena->rootTranslation)[0];
        require(validLow(root.reference) && validLow(root.displacement) && validLow(root.correction),
                "candidate compensated root is nonfinite or noncanonical");
        require(root.reference.x == model.defaultQ[0] && root.reference.y == model.defaultQ[1] &&
                    root.reference.z == model.defaultQ[2],
                "candidate root lost immutable initial reference");
        for (std::uint32_t body = kFirstBody; body < kWorldBodyCount; ++body)
            require(validLow(contents<mr_float4>(arena->bodyPositionLow)[body]),
                    "candidate body-low companion was not materialized canonically");
    }
    for (const CandidateArena* arena : {&audit.base, &audit.plus, &audit.minus})
        require(validLow(contents<mr_float4>(arena->pointPositionLow)[0]),
                "candidate point-low companion was not materialized canonically");

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
    const auto& plusLow = contents<mr_float4>(audit.plus.pointPositionLow)[0];
    const auto& minusLow = contents<mr_float4>(audit.minus.pointPositionLow)[0];
    const auto pairedDifference = [](float highPlus, float lowPlus, float highMinus, float lowMinus) {
        return static_cast<float>(((static_cast<double>(highPlus) + lowPlus) -
                                   (static_cast<double>(highMinus) + lowMinus)) /
            (2.0 * kFiniteDifferenceEpsilon * kTimestep));
    };
    const std::array<float, 3u> finiteDifference{{
        pairedDifference(plusWorld.position.x, plusLow.x, minusWorld.position.x, minusLow.x),
        pairedDifference(plusWorld.position.y, plusLow.y, minusWorld.position.y, minusLow.y),
        pairedDifference(plusWorld.position.z, plusLow.z, minusWorld.position.z, minusLow.z),
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
            verifyHeterogeneousScaleGate();
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
                << " body_only=accepted A0=frozen"
                << " source_dynamics=immutable_3xnv"
                << " compensated_companion_negative=" << audit.companionNegativeCount
                << " optional_point_low=absent root_reference=preserved\n";
            return 0;
        } catch (const std::exception& exception) {
            std::cerr << "numanx_human_matter_candidate_probe: "
                      << exception.what() << '\n';
            return 1;
        }
    }
}
