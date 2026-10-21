#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/numi_human_stand_gpu.h"
#include "metalrobo/EngineModel.hpp"
#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/compensated_translation_gpu.h"
#include "metalrobo/numi_human_tendon_gpu.h"
#include "metalrobo/numi_human_joint_equality_gpu.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
constexpr unsigned nv = 6u;
constexpr unsigned nq = 7u;
constexpr unsigned bodyCount = 1u;
constexpr unsigned pointCount = 5u;
using Buffers = std::array<id<MTLBuffer>, 26u>;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

Buffers allocate(id<MTLDevice> device, const float timestep) {
    const std::array<std::size_t, 26u> sizes{{
        sizeof(MRWorldGPU), sizeof(MRArticulationGPU),
        nv * sizeof(MRDofPropertiesGPU), bodyCount * sizeof(MRBodyPropertiesGPU),
        sizeof(MRNumiHumanStandDispatchGPU), nq * sizeof(float), nv * sizeof(float),
        bodyCount * sizeof(MRArticulatedBodyPoseGPU),
        pointCount * sizeof(MRArticulatedPointWorldGPU),
        pointCount * 3u * nv * sizeof(float), nv * sizeof(float),
        sizeof(MRNumiHumanStandContactGPU),
        bodyCount * MR_NUMI_HUMAN_STAND_SPATIAL_SCRATCH_ROWS * nv * sizeof(float),
        bodyCount * 2u * sizeof(mr_float4), nv * nv * sizeof(float),
        4u * nv * sizeof(float), 12u * nv * sizeof(float),
        sizeof(MRNumiHumanStandStatusGPU), sizeof(MRNumiHumanTendonBindingGPU),
        sizeof(MRNumiHumanTendonTransferResultGPU), sizeof(MRNumiHumanJointEqualityGPU),
        sizeof(MRCompensatedRootTranslationGPU), bodyCount * sizeof(mr_float4),
        pointCount * sizeof(mr_float4), nv * (nv + 1u) * sizeof(float),
        3u * nv * sizeof(float)
    }};
    Buffers buffers{};
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        buffers[i] = [device newBufferWithLength:sizes[i]
                                          options:MTLResourceStorageModeShared];
        require(buffers[i] != nil, "could not allocate stand fixture buffer");
        std::memset(buffers[i].contents, 0, sizes[i]);
    }

    auto* world = static_cast<MRWorldGPU*>(buffers[0].contents);
    world->abiVersion = MR_ENGINE_ABI_VERSION;
    world->bodyCount = bodyCount;
    world->articulationCount = 1u;
    world->nq = nq;
    world->nv = nv;
    world->gravityAndTimestep = {0.0f, 0.0f, 0.0f, timestep};

    auto* articulation = static_cast<MRArticulationGPU*>(buffers[1].contents);
    articulation->bodyCount = bodyCount;
    articulation->rootType = MR_ROOT_FLOATING;
    articulation->nq = nq;
    articulation->nv = nv;

    auto* dofs = static_cast<MRDofPropertiesGPU*>(buffers[2].contents);
    for (unsigned i = 0; i < nv; ++i) {
        dofs[i].vIndex = i;
        dofs[i].qIndex = i < 3u ? i : MR_INVALID_INDEX;
    }
    auto* bodies = static_cast<MRBodyPropertiesGPU*>(buffers[3].contents);
    for (unsigned i = 0; i < bodyCount; ++i) {
        bodies[i].massAndInverseMass = {2.5f, 0.4f, 0.0f, 0.0f};
        bodies[i].inertiaRow0 = {0.1f, 0.0f, 0.0f, 0.0f};
        bodies[i].inertiaRow1 = {0.0f, 1.0f, 0.0f, 0.0f};
        bodies[i].inertiaRow2 = {0.0f, 0.0f, 1.0f, 0.0f};
    }

    auto* dispatch = static_cast<MRNumiHumanStandDispatchGPU*>(buffers[4].contents);
    dispatch->abiVersion = MR_NUMI_HUMAN_STAND_ABI_VERSION;
    dispatch->environmentCount = 1u;
    dispatch->stepCount = 1u;
    dispatch->qStride = nq;
    dispatch->vStride = nv;
    dispatch->pointWorldStride = pointCount;
    dispatch->pointJacobianStride = pointCount * 3u * nv;
    dispatch->bodyPoseStride = bodyCount;
    dispatch->generalizedForceStride = nv;
    dispatch->supportContactCount = 1u;
    dispatch->flags = MR_NUMI_HUMAN_STAND_ENABLE_CONTACT;
    dispatch->contactIterationCount = 8u;
    dispatch->groundPointAndTimestep = {0.0f, 0.0f, 0.0f, timestep};
    dispatch->groundNormal = {0.0f, 0.0f, 1.0f, 0.0f};
    dispatch->targetRootOrientation = {0.0f, 0.0f, 0.0f, 1.0f};

    auto* contact = static_cast<MRNumiHumanStandContactGPU*>(buffers[11].contents);
    contact->bodyIndex = 0u;
    contact->pointQueryIndex = 4u;
    contact->sourceGeometryIndex = 0u;
    contact->frictionSlopAndStabilization = {0.5f, 0.001f, 1.0f, 0.0f};
    contact->planePoint = {0.0f, 0.0f, 0.0f, 0.0f};
    contact->planeNormal = {0.0f, 1.0f, 0.0f, 0.0f};

    auto* q = static_cast<float*>(buffers[5].contents);
    q[2u] = 0.5f;
    q[6u] = 1.0f;
    auto* v = static_cast<float*>(buffers[6].contents);
    v[0u] = -0.05f;
    v[1u] = -1.0f;
    auto* poses = static_cast<MRArticulatedBodyPoseGPU*>(buffers[7].contents);
    auto* points = static_cast<MRArticulatedPointWorldGPU*>(buffers[8].contents);
    auto* jacobians = static_cast<float*>(buffers[9].contents);
    const std::array<std::array<float, 3u>, pointCount> offsets{{
        {{0.0f, 0.0f, 0.0f}}, {{1.0f, 0.0f, 0.0f}},
        {{0.0f, 1.0f, 0.0f}}, {{0.0f, 0.0f, 1.0f}},
        {{0.0f, 0.0f, -0.5f}}
    }};
    poses[0].orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    auto cross = [](const std::array<float, 3u>& a, const std::array<float, 3u>& b) {
        return std::array<float, 3u>{a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
    };
    for (unsigned point = 0u; point < pointCount; ++point) {
        const auto& r = offsets[point];
        points[point].position = {r[0], r[1], 0.5f + r[2], 0.0f};
        for (unsigned axis = 0u; axis < 3u; ++axis)
            jacobians[point * 3u * nv + axis * nv + axis] = 1.0f;
        for (unsigned dof = 3u; dof < 6u; ++dof) {
            std::array<float, 3u> axis{};
            axis[dof - 3u] = 1.0f;
            const auto angular = cross(axis, r);
            for (unsigned component = 0u; component < 3u; ++component)
                jacobians[point * 3u * nv + component * nv + dof] = angular[component];
        }
    }
    return buffers;
}

void run(id<MTLComputePipelineState> pipeline,
         id<MTLCommandQueue> queue, const Buffers& buffers) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    require(command != nil && encoder != nil, "could not encode stand fixture");
    [encoder setComputePipelineState:pipeline];
    for (unsigned i = 0u; i < buffers.size(); ++i)
        [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    [encoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
           threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1u, 1u)];
    [encoder endEncoding];
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [command addCompletedHandler:^(id<MTLCommandBuffer>) {
        dispatch_semaphore_signal(done);
    }];
    [command commit];
    require(dispatch_semaphore_wait(done,
        dispatch_time(DISPATCH_TIME_NOW, 30 * NSEC_PER_SEC)) == 0,
        "per-contact support-plane fixture timed out");
    require(command.status == MTLCommandBufferStatusCompleted,
        "per-contact support-plane Metal command failed");
}

template <typename T>
bool exactBytes(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
        (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

struct GeometryCaptureProbe {
    unsigned encodeCalls = 0u;
    unsigned abortCalls = 0u;
};

bool encodeCapturePlane(void* context,
                        const metalrobo::MetalNumiHumanSupportGeometryPass& pass) {
    auto* probe = static_cast<GeometryCaptureProbe*>(context);
    if (probe == nullptr || pass.phase !=
            metalrobo::MetalNumiHumanSupportGeometryPhase::currentPose ||
        pass.commandBuffer == nullptr || pass.bodyPoses == nullptr ||
        pass.pointWorld == nullptr || pass.pointPositionLow == nullptr ||
        pass.pointJacobians == nullptr || pass.standContacts == nullptr ||
        pass.environmentCount != 1u ||
        pass.pointCount != 1u || pass.standContactCount != 1u ||
        pass.dofCount != 6u) {
        return false;
    }
    id<MTLBuffer> contactBuffer = (__bridge id<MTLBuffer>)pass.standContacts;
    if (contactBuffer.length < sizeof(MRNumiHumanStandContactGPU)) return false;
    auto* contact = static_cast<MRNumiHumanStandContactGPU*>(
        contactBuffer.contents);
    contact->planePoint = {0.0f, 0.0f, -0.25f, 0.0f};
    contact->planeNormal = {0.0f, 0.0f, 1.0f, 0.0f};
    ++probe->encodeCalls;
    return true;
}

bool encodeCaptureGlobalPlane(
    void* context, const metalrobo::MetalNumiHumanSupportGeometryPass& pass) {
    auto* probe = static_cast<GeometryCaptureProbe*>(context);
    if (probe == nullptr || pass.phase !=
            metalrobo::MetalNumiHumanSupportGeometryPhase::currentPose ||
        pass.commandBuffer == nullptr || pass.bodyPoses == nullptr ||
        pass.pointWorld == nullptr || pass.pointPositionLow == nullptr ||
        pass.pointJacobians == nullptr || pass.standContacts == nullptr ||
        pass.environmentCount != 1u || pass.pointCount != 1u ||
        pass.standContactCount != 1u || pass.dofCount != 6u ||
        pass.groundPoint.x != 0.0f || pass.groundPoint.y != 0.0f ||
        pass.groundPoint.z != -0.25f || pass.groundPoint.w != 0.0f ||
        pass.groundNormal.x != 0.0f || pass.groundNormal.y != 0.0f ||
        pass.groundNormal.z != 1.0f || pass.groundNormal.w != 0.0f) {
        return false;
    }
    id<MTLBuffer> contactBuffer = (__bridge id<MTLBuffer>)pass.standContacts;
    if (contactBuffer.length < sizeof(MRNumiHumanStandContactGPU)) return false;
    const auto* contact = static_cast<const MRNumiHumanStandContactGPU*>(
        contactBuffer.contents);
    if (std::memcmp(&contact->planePoint, &pass.groundPoint,
                    sizeof(mr_float4)) != 0 ||
        std::memcmp(&contact->planeNormal, &pass.groundNormal,
                    sizeof(mr_float4)) != 0) {
        return false;
    }
    ++probe->encodeCalls;
    return true;
}

bool encodeCaptureMutatingGlobalPlane(
    void* context, const metalrobo::MetalNumiHumanSupportGeometryPass& pass) {
    if (!encodeCaptureGlobalPlane(context, pass)) return false;
    id<MTLBuffer> contactBuffer = (__bridge id<MTLBuffer>)pass.standContacts;
    auto* contact = static_cast<MRNumiHumanStandContactGPU*>(
        contactBuffer.contents);
    contact->planePoint.z += 0.001f;
    return true;
}

void abortCapturePlane(void* context, void*) {
    auto* probe = static_cast<GeometryCaptureProbe*>(context);
    if (probe != nullptr) ++probe->abortCalls;
}

metalrobo::EngineModel makeSourceMass128Model() {
    using namespace metalrobo;
    constexpr std::uint32_t dofs = 128u;
    constexpr std::uint32_t joints = dofs - 6u;
    constexpr std::uint32_t bodies = joints + 1u;
    constexpr std::uint32_t nq = 7u + joints;

    EngineModel model;
    model.name = "source_mass_128_prismatic_chain";
    model.world.abiVersion = MR_ENGINE_ABI_VERSION;
    model.world.bodyCount = bodies;
    model.world.articulationCount = 1u;
    model.world.jointCount = joints;
    model.world.nq = nq;
    model.world.nv = dofs;
    model.world.shapeCount = 0u;
    model.world.materialCount = 0u;
    model.world.pairCapacity = 1u;
    model.world.contactCapacity = 1u;
    model.world.constraintCapacity = 1u;
    model.world.islandCapacity = 1u;
    model.world.solverType = MR_SOLVER_REFERENCE_FP64;
    model.world.frictionConeType = MR_FRICTION_CONE_ELLIPTIC;
    model.world.gravityAndTimestep = {0.0f, 0.0f, 0.0f, 1.0f / 240.0f};
    model.world.solverScales = {1.0e-8f, 1.0e-9f, 2.0f, 1.0e-4f};

    MRArticulationGPU articulation{};
    articulation.rootBody = 0u;
    articulation.rootType = MR_ROOT_FLOATING;
    articulation.firstBody = 0u;
    articulation.bodyCount = bodies;
    articulation.firstJoint = 0u;
    articulation.jointCount = joints;
    articulation.qOffset = 0u;
    articulation.nq = nq;
    articulation.vOffset = 0u;
    articulation.nv = dofs;
    model.articulations.push_back(articulation);

    model.bodies.resize(bodies);
    for (std::uint32_t bodyIndex = 0u; bodyIndex < bodies; ++bodyIndex) {
        MRBodyPropertiesGPU body{};
        body.articulationIndex = 0u;
        body.parentBody = bodyIndex == 0u ? MR_INVALID_INDEX : bodyIndex - 1u;
        body.inboundJoint = bodyIndex == 0u ? MR_INVALID_INDEX : bodyIndex - 1u;
        body.motionType = MR_MOTION_DYNAMIC;
        body.massAndInverseMass = {1.0f, 1.0f, 0.0f, 0.0f};
        body.inertiaRow0 = {0.1f, 0.0f, 0.0f, 0.0f};
        body.inertiaRow1 = {0.0f, 0.1f, 0.0f, 0.0f};
        body.inertiaRow2 = {0.0f, 0.0f, 0.1f, 0.0f};
        body.inverseInertiaRow0 = {10.0f, 0.0f, 0.0f, 0.0f};
        body.inverseInertiaRow1 = {0.0f, 10.0f, 0.0f, 0.0f};
        body.inverseInertiaRow2 = {0.0f, 0.0f, 10.0f, 0.0f};
        model.bodies[bodyIndex] = body;
    }

    model.joints.resize(joints);
    for (std::uint32_t jointIndex = 0u; jointIndex < joints; ++jointIndex) {
        MRJointDescriptorGPU joint{};
        joint.parentBody = jointIndex;
        joint.childBody = jointIndex + 1u;
        joint.jointType = MR_JOINT_PRISMATIC;
        joint.qOffset = 7u + jointIndex;
        joint.nq = 1u;
        joint.vOffset = 6u + jointIndex;
        joint.nv = 1u;
        joint.axis0 = {0.0f, 0.0f, 1.0f, 0.0f};
        joint.parentRotation = {0.0f, 0.0f, 0.0f, 1.0f};
        joint.childRotation = {0.0f, 0.0f, 0.0f, 1.0f};
        model.joints[jointIndex] = joint;
    }

    model.dofs.resize(dofs);
    for (std::uint32_t dofIndex = 0u; dofIndex < dofs; ++dofIndex) {
        MRDofPropertiesGPU dof{};
        dof.articulationIndex = 0u;
        dof.jointIndex = dofIndex < 6u ? MR_INVALID_INDEX : dofIndex - 6u;
        dof.qIndex = dofIndex < 3u ? dofIndex :
            (dofIndex < 6u ? MR_INVALID_INDEX : 7u + dofIndex - 6u);
        dof.vIndex = dofIndex;
        dof.localDof = dofIndex < 6u ? dofIndex : 0u;
        dof.flags = dofIndex < 6u ? MR_DOF_FLAG_ROOT : 0u;
        model.dofs[dofIndex] = dof;
    }

    model.defaultQ.assign(nq, 0.0f);
    model.defaultQ[6u] = 1.0f;
    model.defaultV.assign(dofs, 0.0f);
    std::string reason;
    if (!model.valid(&reason)) throw std::runtime_error(reason);
    return model;
}

struct SourceMass128Probe {
    void* input = nullptr;
    void* action = nullptr;
    void* solution = nullptr;
    void* repeatedSolution = nullptr;
    void* actionStatus = nullptr;
    void* solveStatus = nullptr;
    void* repeatedStatus = nullptr;
    std::uint32_t preDynamicsCalls = 0u;
    bool operationsEncoded = false;
    std::uint32_t abortCalls = 0u;
};

bool encodeSourceMass128(void* raw,
                         const metalrobo::MetalNumanXTransactionPass& pass) noexcept {
    auto* probe = static_cast<SourceMass128Probe*>(raw);
    if (probe == nullptr) return false;
    if (pass.phase != metalrobo::MetalNumanXTransactionPhase::preDynamics)
        return true;
    ++probe->preDynamicsCalls;
    if (probe->preDynamicsCalls != 1u ||
        pass.encodeSourceMassOperation == nullptr ||
        pass.sourceMassOperationContext == nullptr ||
        pass.dofCount != 128u || pass.environmentCount != 1u)
        return false;

    const auto encode = [&](const std::uint32_t operation,
                            void* input, void* output, void* statuses) noexcept {
        metalrobo::MetalNumanXTransactionSourceMassQuery query{};
        query.operation = operation;
        query.input = input;
        query.output = output;
        query.statuses = statuses;
        query.inputGPUAddress = ((__bridge id<MTLBuffer>)input).gpuAddress;
        query.outputGPUAddress = ((__bridge id<MTLBuffer>)output).gpuAddress;
        query.statusesGPUAddress = ((__bridge id<MTLBuffer>)statuses).gpuAddress;
        query.inputElementCount = 128u;
        query.outputElementCount = 128u;
        query.statusElementCount = 1u;
        query.generalizedVectorStride = 128u;
        query.statusStride = 1u;
        return pass.encodeSourceMassOperation(
            pass.sourceMassOperationContext, pass, query);
    };

    const bool ok =
        encode(MR_ARTICULATED_SOURCE_MASS_ACTION,
               probe->input, probe->action, probe->actionStatus) &&
        encode(MR_ARTICULATED_SOURCE_MASS_SOLVE,
               probe->action, probe->solution, probe->solveStatus) &&
        encode(MR_ARTICULATED_SOURCE_MASS_SOLVE,
               probe->action, probe->repeatedSolution,
               probe->repeatedStatus);
    probe->operationsEncoded = ok;
    return ok;
}

void abortSourceMass128(void* raw, void*) noexcept {
    auto* probe = static_cast<SourceMass128Probe*>(raw);
    if (probe != nullptr) ++probe->abortCalls;
}

id<MTLBuffer> sharedBuffer(id<MTLDevice> device, std::size_t bytes,
                           const char* label) {
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes
        options:MTLResourceStorageModeShared];
    require(buffer != nil, label);
    std::memset(buffer.contents, 0, bytes);
    return buffer;
}

void testSourceMass128(id<MTLDevice> device, const char* metallibPath) {
    using namespace metalrobo;
    constexpr std::size_t nv128 = 128u;
    const EngineModel model = makeSourceMass128Model();

    std::vector<double> q(model.defaultQ.begin(), model.defaultQ.end());
    std::vector<double> mass(nv128 * nv128, 0.0);
    ArticulatedDynamicsConfig referenceConfig;
    referenceConfig.gravity = {0.0, 0.0, 0.0};
    const auto reference = computeArticulatedMassMatrix(
        model, 0u, q, mass, referenceConfig);
    require(reference.succeeded(),
            "FP64 128-DoF mass-matrix oracle rejected the fixture model");

    id<MTLBuffer> inputBuffer = sharedBuffer(
        device, nv128 * sizeof(float), "could not allocate 128-DoF source-mass input");
    id<MTLBuffer> actionBuffer = sharedBuffer(
        device, nv128 * sizeof(float), "could not allocate 128-DoF source-mass action");
    id<MTLBuffer> solutionBuffer = sharedBuffer(
        device, nv128 * sizeof(float), "could not allocate 128-DoF source-mass solution");
    id<MTLBuffer> repeatedBuffer = sharedBuffer(
        device, nv128 * sizeof(float), "could not allocate repeated 128-DoF solution");
    id<MTLBuffer> actionStatus = sharedBuffer(
        device, sizeof(std::uint32_t), "could not allocate source-mass ACTION status");
    id<MTLBuffer> solveStatus = sharedBuffer(
        device, sizeof(std::uint32_t), "could not allocate source-mass SOLVE status");
    id<MTLBuffer> repeatedStatus = sharedBuffer(
        device, sizeof(std::uint32_t), "could not allocate repeated SOLVE status");

    std::vector<float> x(nv128);
    for (std::size_t i = 0u; i < nv128; ++i) {
        const int centered = static_cast<int>((i * 37u) % 101u) - 50;
        x[i] = 0.01f * static_cast<float>(centered);
    }
    std::memcpy(inputBuffer.contents, x.data(), x.size() * sizeof(float));

    SourceMass128Probe probe;
    probe.input = (__bridge void*)inputBuffer;
    probe.action = (__bridge void*)actionBuffer;
    probe.solution = (__bridge void*)solutionBuffer;
    probe.repeatedSolution = (__bridge void*)repeatedBuffer;
    probe.actionStatus = (__bridge void*)actionStatus;
    probe.solveStatus = (__bridge void*)solveStatus;
    probe.repeatedStatus = (__bridge void*)repeatedStatus;

    MetalArticulatedOperatorConfig config;
    config.pointJacobiansOnly = true;
    config.metallibPath = metallibPath;
    config.mujocoActivationTimestepSeconds = 1.0f / 240.0f;
    MetalArticulatedOperatorContext context(config);

    // Use the same minimal, zero-excitation MyoSim owner as the public
    // transaction probe. A stand transaction requires activation stepping;
    // omitting it would reject before exercising source-mass assembly.
    std::vector<MRArticulatedPointImpulseGPU> bodyProbes;
    for (std::uint32_t body = 0u; body < model.bodies.size(); ++body) {
        for (std::uint32_t axis = 0u; axis < 4u; ++axis) {
            MRArticulatedPointImpulseGPU point{};
            point.bodyIndex = body;
            if (axis != 0u) {
                if (axis == 1u) point.localPoint.x = 1.0f;
                if (axis == 2u) point.localPoint.y = 1.0f;
                if (axis == 3u) point.localPoint.z = 1.0f;
            }
            bodyProbes.push_back(point);
        }
    }
    std::array<MRMujocoMuscleSiteGPU, 2u> sites{};
    sites[0].bodyIndex = 0u;
    sites[1].bodyIndex = 0u;
    sites[0].localPoint = {-0.05f, 0.0f, 0.0f, 0.0f};
    sites[1].localPoint = {0.05f, 0.0f, 0.0f, 0.0f};
    std::array<MRMujocoMuscleRouteNodeGPU, 2u> routes{};
    for (std::uint32_t i = 0u; i < 2u; ++i) {
        routes[i].type = MR_MUJOCO_MUSCLE_ROUTE_SITE;
        routes[i].targetIndex = i;
        routes[i].sideSiteIndex = MR_INVALID_INDEX;
    }
    std::array<MRMujocoMuscleGPU, 1u> muscles{};
    muscles[0].route = {0u, 2u, 0u, 0u};
    muscles[0].lengthRangeAndAcceleration = {0.05f, 0.15f, 1.0f, 0.0f};
    muscles[0].controlRange = {0.0f, 1.0f, 0.0f, 0.0f};
    std::array<MRMujocoMuscleStateGPU, 1u> muscleStates{};

    MetalNumanXTransactionProgram transaction;
    transaction.abiVersion = kMetalNumanXTransactionSourceMassABIVersion;
    transaction.structSize = sizeof(transaction);
    transaction.context = &probe;
    transaction.encode = encodeSourceMass128;
    transaction.abort = abortSourceMass128;
    transaction.fingerprint = 0x534f555243453132ull;
    transaction.writeExactCandidateKinematics = true;
    transaction.candidatePointCapacity = 1u;

    MetalArticulatedOperatorInput input;
    input.articulationIndex = 0u;
    input.environmentCount = 1u;
    input.q = model.defaultQ;
    input.v = model.defaultV;
    input.points = bodyProbes;
    input.pointCount = bodyProbes.size();
    input.mujoco.muscles = muscles;
    input.mujoco.states = muscleStates;
    input.mujoco.sites = sites;
    input.mujoco.routeNodes = routes;
    input.mujoco.bodyJacobianPointOffset = 0u;
    input.stand.v = model.defaultV;
    input.stand.stepCount = 1u;
    input.stand.enableContact = false;
    input.stand.numanXTransactionProgram = transaction;
    MetalArticulatedOperatorResult result;
    const auto diagnostics = context.run(model, input, result);
    if (!diagnostics.succeeded()) {
        throw std::runtime_error(
            std::string("128-DoF production source-mass transaction failed: status=") +
            std::to_string(static_cast<unsigned>(diagnostics.status)) +
            " message=" + diagnostics.message +
            " dispatched=" + std::to_string(diagnostics.dispatched ? 1u : 0u) +
            " published=" + std::to_string(diagnostics.published ? 1u : 0u) +
            " failed_envs=" + std::to_string(diagnostics.failedEnvironmentCount) +
            " first_gpu_status=" + std::to_string(diagnostics.firstGPUStatusCode) +
            " stand_steps=" + std::to_string(diagnostics.completedStandSteps) +
            " pre_dynamics_calls=" + std::to_string(probe.preDynamicsCalls) +
            " operations_encoded=" + std::to_string(probe.operationsEncoded ? 1u : 0u) +
            " abort_calls=" + std::to_string(probe.abortCalls));
    }
    if (probe.preDynamicsCalls != 1u || !probe.operationsEncoded ||
        probe.abortCalls != 0u) {
        throw std::runtime_error(
            std::string("128-DoF source-mass callback mismatch: pre_dynamics_calls=") +
            std::to_string(probe.preDynamicsCalls) +
            " operations_encoded=" + std::to_string(probe.operationsEncoded ? 1u : 0u) +
            " abort_calls=" + std::to_string(probe.abortCalls));
    }

    const auto* action = static_cast<const float*>(actionBuffer.contents);
    const auto* solution = static_cast<const float*>(solutionBuffer.contents);
    const auto* repeated = static_cast<const float*>(repeatedBuffer.contents);
    const auto* actionCode = static_cast<const std::uint32_t*>(actionStatus.contents);
    const auto* solveCode = static_cast<const std::uint32_t*>(solveStatus.contents);
    const auto* repeatedCode = static_cast<const std::uint32_t*>(repeatedStatus.contents);
    require(actionCode[0] == MR_ARTICULATED_SOURCE_MASS_SUCCESS &&
                solveCode[0] == MR_ARTICULATED_SOURCE_MASS_SUCCESS &&
                repeatedCode[0] == MR_ARTICULATED_SOURCE_MASS_SUCCESS,
            "128-DoF source-mass operation status was not success");

    double actionErrorSquared = 0.0;
    double actionReferenceSquared = 0.0;
    double solveErrorSquared = 0.0;
    double vectorSquared = 0.0;
    double solveResidualSquared = 0.0;
    double rhsSquared = 0.0;
    double reuseErrorSquared = 0.0;
    for (std::size_t row = 0u; row < nv128; ++row) {
        double expectedAction = 0.0;
        double rowScale = 0.0;
        double solvedAction = 0.0;
        for (std::size_t column = 0u; column < nv128; ++column) {
            const double coefficient = mass[row * nv128 + column];
            expectedAction += coefficient * static_cast<double>(x[column]);
            rowScale += std::abs(coefficient * static_cast<double>(x[column]));
            solvedAction += coefficient * static_cast<double>(solution[column]);
        }
        const double actionError = static_cast<double>(action[row]) - expectedAction;
        require(std::abs(actionError) <= 3.0e-5 * rowScale + 2.0e-6,
                "128-DoF production ACTION disagrees with FP64 mass oracle");
        actionErrorSquared += actionError * actionError;
        actionReferenceSquared += expectedAction * expectedAction;
        const double solutionError = static_cast<double>(solution[row]) - x[row];
        solveErrorSquared += solutionError * solutionError;
        vectorSquared += static_cast<double>(x[row]) * x[row];
        const double residual = solvedAction - static_cast<double>(action[row]);
        solveResidualSquared += residual * residual;
        rhsSquared += static_cast<double>(action[row]) * action[row];
        const double reuseError = static_cast<double>(repeated[row]) - solution[row];
        reuseErrorSquared += reuseError * reuseError;
    }
    require(std::sqrt(actionErrorSquared) /
                std::max(std::sqrt(actionReferenceSquared), 1.0e-12) < 2.0e-5,
            "128-DoF ACTION relative error exceeded the FP64 oracle bound");
    require(std::sqrt(solveErrorSquared) /
                std::max(std::sqrt(vectorSquared), 1.0e-12) < 5.0e-3,
            "128-DoF SOLVE did not recover the known source vector");
    require(std::sqrt(solveResidualSquared) /
                std::max(std::sqrt(rhsSquared), 1.0e-12) < 2.0e-4,
            "128-DoF SOLVE residual against production ACTION was too large");
    require(std::sqrt(reuseErrorSquared) /
                std::max(std::sqrt(vectorSquared), 1.0e-12) < 1.0e-6,
            "128-DoF repeated SOLVE changed the cached source-mass factor result");
}

void testNoncommittingSupportCapture(const char* metallibPath) {
    using namespace metalrobo;
    const EngineModel model = makeFreeSphereEngineModel();
    MetalArticulatedOperatorConfig config;
    config.pointJacobiansOnly = true;
    config.metallibPath = metallibPath;
    MetalArticulatedOperatorContext context(config);

    std::array<MRArticulatedPointImpulseGPU, 1u> points{};
    points[0].bodyIndex = model.articulations[0].rootBody;
    points[0].localPoint = {0.125f, -0.25f, 0.375f, 0.0f};
    std::array<MRCompensatedRootTranslationGPU, 1u> roots{};
    roots[0].reference = {model.defaultQ[0], model.defaultQ[1],
                          model.defaultQ[2], 0.0f};
    const auto baselineInput = [&]() {
        MetalArticulatedOperatorInput input;
        input.articulationIndex = 0u;
        input.environmentCount = 1u;
        input.pointCount = points.size();
        input.q = model.defaultQ;
        input.rootTranslations = roots;
        input.points = points;
        input.mujoco.bodyJacobianPointOffset = 0u;
        input.collectFullResultToHost = true;
        return input;
    };
    MetalArticulatedOperatorResult before;
    const auto beforeDiagnostics = context.run(model, baselineInput(), before);
    require(beforeDiagnostics.succeeded(),
            "baseline paired FK/J query failed before capture");

    GeometryCaptureProbe probe;
    std::array<MRNumiHumanStandContactGPU, 1u> contacts{};
    contacts[0].bodyIndex = points[0].bodyIndex;
    contacts[0].pointQueryIndex = 0u;
    contacts[0].sourceGeometryIndex = 7u;
    contacts[0].frictionSlopAndStabilization = {0.5f, 0.001f, 0.0f, 0.0f};
    contacts[0].planeNormal = {0.0f, 0.0f, 1.0f, 0.0f};
    const MetalNumiHumanSupportGeometryProgram geometryProgram{
        .context = &probe,
        .encodePreDynamics = encodeCapturePlane,
        .abort = abortCapturePlane,
        .fingerprint = 0x5343415054555245ull,
        .usePerContactSupportPlanes = true,
    };
    auto captureInput = baselineInput();
    captureInput.stand.contacts = contacts;
    captureInput.stand.usePerContactSupportPlanes = true;
    captureInput.stand.supportGeometryProgram = geometryProgram;
    captureInput.stand.enableContact = true;
    captureInput.stand.groundNormal = {0.0f, 0.0f, 1.0f, 0.0f};
    captureInput.captureSupportGeometryOnly = true;
    MetalArticulatedOperatorResult captured;
    const auto captureDiagnostics = context.run(model, captureInput, captured);
    require(captureDiagnostics.succeeded() && probe.encodeCalls == 1u &&
                probe.abortCalls == 0u,
            "registered support callback was not run exactly once");
    const bool captureStateWasNoncommitting =
        captureDiagnostics.completedStandSteps == 0u &&
        captureDiagnostics.published &&
        captureDiagnostics.residentStateTransactionFingerprint == 0u &&
        captureDiagnostics.residentStateGeneration == 0u &&
        captured.standQ.empty() && captured.standV.empty() &&
        captured.standStatuses.empty() &&
        captured.standContactImpulses.empty() &&
        std::all_of(captured.generalizedImpulse.begin(),
            captured.generalizedImpulse.end(),
            [](float value) { return value == 0.0f; }) &&
        std::all_of(captured.deltaVelocity.begin(),
            captured.deltaVelocity.end(),
            [](float value) { return value == 0.0f; });
    if (!captureStateWasNoncommitting) {
        const auto printFloatSummary = [](
            const char* label, const std::vector<float>& values) {
            std::cerr << " " << label << "Count=" << values.size();
            if (values.empty()) {
                std::cerr << " min=NA max=NA";
                return;
            }
            const auto [minimum, maximum] = std::minmax_element(
                values.begin(), values.end());
            std::cerr << " min=" << *minimum << " max=" << *maximum;
        };
        std::cerr << "support-capture noncommit failure: completedStandSteps="
                  << captureDiagnostics.completedStandSteps
                  << " hostResultPublished=" << captureDiagnostics.published
                  << " residentTransactionFingerprint="
                  << captureDiagnostics.residentStateTransactionFingerprint
                  << " residentStateGeneration="
                  << captureDiagnostics.residentStateGeneration
                  << " standQCount=" << captured.standQ.size()
                  << " standVCount=" << captured.standV.size()
                  << " standStatusCount=" << captured.standStatuses.size()
                  << " contactImpulseCount=" << captured.standContactImpulses.size()
                  << " equalityImpulseCount=" << captured.standJointEqualityImpulses.size()
                  << " sourceLimitImpulseCount=" << captured.standSourceLimitImpulses.size()
                  << " generalizedImpulseCount=" << captured.generalizedImpulse.size()
                  << " deltaVelocityCount=" << captured.deltaVelocity.size();
        for (std::size_t index = 0u;
             index < std::min<std::size_t>(captured.standStatuses.size(), 2u);
             ++index) {
            const auto& status = captured.standStatuses[index];
            std::cerr << " status[" << index << "].code=" << status.code
                      << ".completedSteps=" << status.completedSteps
                      << ".failingIndex=" << status.failingIndex;
        }
        printFloatSummary("standQ", captured.standQ);
        printFloatSummary("standV", captured.standV);
        printFloatSummary("generalizedImpulse", captured.generalizedImpulse);
        printFloatSummary("deltaVelocity", captured.deltaVelocity);
        std::cerr << '\n';
    }
    require(captureStateWasNoncommitting,
            "support capture advanced or published physical state");
    const bool supportContactCountMatches =
        captured.supportGeometryContacts.size() == contacts.size();
    const bool supportBodyIdentityMatches = supportContactCountMatches &&
        captured.supportGeometryContacts[0].bodyIndex == contacts[0].bodyIndex;
    const bool supportPointIdentityMatches = supportContactCountMatches &&
        captured.supportGeometryContacts[0].pointQueryIndex == 0u;
    const bool supportGeometryIdentityMatches = supportContactCountMatches &&
        captured.supportGeometryContacts[0].sourceGeometryIndex == 7u;
    const bool supportPlanePointMatches = supportContactCountMatches &&
        captured.supportGeometryContacts[0].planePoint.z == -0.25f;
    const bool supportPlaneNormalMatches = supportContactCountMatches &&
        captured.supportGeometryContacts[0].planeNormal.z == 1.0f;
    const bool supportFrictionMatches = supportContactCountMatches &&
        captured.supportGeometryContacts[0].frictionSlopAndStabilization.w == 0.0f;
    const bool pointWorldMatches = exactBytes(captured.pointWorld, before.pointWorld);
    const bool pointPositionLowMatches =
        exactBytes(captured.pointPositionLow, before.pointPositionLow);
    const bool pointJacobiansMatch =
        exactBytes(captured.pointJacobians, before.pointJacobians);
    const bool bodyPosesMatch = exactBytes(captured.bodyPoses, before.bodyPoses);
    const bool rootTranslationsMatch =
        exactBytes(captured.rootTranslations, before.rootTranslations);
    const bool captureRowsAndGeometryValid =
        supportContactCountMatches && supportBodyIdentityMatches &&
        supportPointIdentityMatches && supportGeometryIdentityMatches &&
        supportPlanePointMatches && supportPlaneNormalMatches &&
        supportFrictionMatches && captured.pointWorld.size() == 1u &&
        captured.pointPositionLow.size() == 1u &&
        captured.pointJacobians.size() == 18u &&
        captured.bodyPoses.size() == model.articulations[0].bodyCount &&
        captured.bodyPoses.size() == before.bodyPoses.size() &&
        captured.generalizedImpulse.size() == model.world.nv &&
        captured.deltaVelocity.size() == model.world.nv && pointWorldMatches &&
        pointPositionLowMatches && pointJacobiansMatch && bodyPosesMatch &&
        rootTranslationsMatch;
    if (!captureRowsAndGeometryValid) {
        const auto maxFloat4Difference = [](const mr_float4& a, const mr_float4& b) {
            return std::max({
                std::abs(static_cast<double>(a.x) - b.x),
                std::abs(static_cast<double>(a.y) - b.y),
                std::abs(static_cast<double>(a.z) - b.z),
                std::abs(static_cast<double>(a.w) - b.w)});
        };
        const auto maxFloatArrayDifference = [](
            const std::vector<float>& a, const std::vector<float>& b) {
            double maximum = 0.0;
            for (std::size_t index = 0u; index < std::min(a.size(), b.size()); ++index) {
                maximum = std::max(maximum,
                    std::abs(static_cast<double>(a[index]) - b[index]));
            }
            return maximum;
        };
        double pointWorldMaximumDifference = 0.0;
        for (std::size_t index = 0u;
             index < std::min(captured.pointWorld.size(), before.pointWorld.size());
             ++index) {
            pointWorldMaximumDifference = std::max(pointWorldMaximumDifference,
                maxFloat4Difference(captured.pointWorld[index].position,
                                    before.pointWorld[index].position));
        }
        double bodyPoseMaximumDifference = 0.0;
        for (std::size_t index = 0u;
             index < std::min(captured.bodyPoses.size(), before.bodyPoses.size());
             ++index) {
            bodyPoseMaximumDifference = std::max(bodyPoseMaximumDifference,
                maxFloat4Difference(captured.bodyPoses[index].position,
                                    before.bodyPoses[index].position));
            bodyPoseMaximumDifference = std::max(bodyPoseMaximumDifference,
                maxFloat4Difference(captured.bodyPoses[index].orientation,
                                    before.bodyPoses[index].orientation));
        }
        double pointPositionLowMaximumDifference = 0.0;
        for (std::size_t index = 0u;
             index < std::min(captured.pointPositionLow.size(), before.pointPositionLow.size());
             ++index) {
            pointPositionLowMaximumDifference = std::max(pointPositionLowMaximumDifference,
                maxFloat4Difference(captured.pointPositionLow[index],
                                    before.pointPositionLow[index]));
        }
        double rootTranslationMaximumDifference = 0.0;
        for (std::size_t index = 0u;
             index < std::min(captured.rootTranslations.size(), before.rootTranslations.size());
             ++index) {
            rootTranslationMaximumDifference = std::max(rootTranslationMaximumDifference,
                maxFloat4Difference(captured.rootTranslations[index].reference,
                                    before.rootTranslations[index].reference));
            rootTranslationMaximumDifference = std::max(rootTranslationMaximumDifference,
                maxFloat4Difference(captured.rootTranslations[index].displacement,
                                    before.rootTranslations[index].displacement));
            rootTranslationMaximumDifference = std::max(rootTranslationMaximumDifference,
                maxFloat4Difference(captured.rootTranslations[index].correction,
                                    before.rootTranslations[index].correction));
        }
        std::cerr << "support-capture geometry failure: contactCount="
                  << captured.supportGeometryContacts.size() << "/" << contacts.size()
                  << " bodyIdentity=" << supportBodyIdentityMatches
                  << " pointIdentity=" << supportPointIdentityMatches
                  << " sourceGeometryIdentity=" << supportGeometryIdentityMatches
                  << " planePoint=" << supportPlanePointMatches
                  << " planeNormal=" << supportPlaneNormalMatches
                  << " frictionRow=" << supportFrictionMatches
                  << " pointWorldCount=" << captured.pointWorld.size()
                  << "/" << before.pointWorld.size()
                  << " pointLowCount=" << captured.pointPositionLow.size()
                  << "/" << before.pointPositionLow.size()
                  << " pointJCount=" << captured.pointJacobians.size()
                  << "/" << before.pointJacobians.size()
                  << " bodyPoseCount=" << captured.bodyPoses.size()
                  << "/" << before.bodyPoses.size()
                  << " expected=" << model.articulations[0].bodyCount
                  << " rootCount=" << captured.rootTranslations.size()
                  << "/" << before.rootTranslations.size()
                  << " generalizedImpulseCount=" << captured.generalizedImpulse.size()
                  << " deltaVelocityCount=" << captured.deltaVelocity.size()
                  << " pointWorldEqual=" << pointWorldMatches
                  << " pointLowEqual=" << pointPositionLowMatches
                  << " pointJEqual=" << pointJacobiansMatch
                  << " bodyPoseEqual=" << bodyPosesMatch
                  << " rootEqual=" << rootTranslationsMatch
                  << " maxPointDiff=" << pointWorldMaximumDifference
                  << " maxPointLowDiff=" << pointPositionLowMaximumDifference
                  << " maxPointJDiff="
                  << maxFloatArrayDifference(captured.pointJacobians, before.pointJacobians)
                  << " maxBodyDiff=" << bodyPoseMaximumDifference
                  << " maxRootDiff=" << rootTranslationMaximumDifference
                  << std::endl;
    }
    require(captureRowsAndGeometryValid,
            "capture lost callback rows or changed paired FK/J geometry");

    MetalArticulatedOperatorResult after;
    const auto afterDiagnostics = context.run(model, baselineInput(), after);
    require(afterDiagnostics.succeeded() &&
                exactBytes(after.pointWorld, before.pointWorld) &&
                exactBytes(after.pointPositionLow, before.pointPositionLow) &&
                exactBytes(after.pointJacobians, before.pointJacobians) &&
                exactBytes(after.bodyPoses, before.bodyPoses) &&
                exactBytes(after.rootTranslations, before.rootTranslations),
            "same-context geometry capture changed subsequent physical query results");

    GeometryCaptureProbe globalProbe;
    const MetalNumiHumanSupportGeometryProgram globalGeometryProgram{
        .context = &globalProbe,
        .encodePreDynamics = encodeCaptureGlobalPlane,
        .abort = abortCapturePlane,
        .fingerprint = 0x474c4f42414c504cull,
        .usePerContactSupportPlanes = false,
    };
    auto globalInput = baselineInput();
    contacts[0].planePoint = {0.0f, 0.0f, -0.25f, 0.0f};
    contacts[0].planeNormal = {0.0f, 0.0f, 1.0f, 0.0f};
    globalInput.stand.contacts = contacts;
    globalInput.stand.usePerContactSupportPlanes = false;
    globalInput.stand.supportGeometryProgram = globalGeometryProgram;
    globalInput.stand.enableContact = true;
    globalInput.stand.groundPoint = {0.0f, 0.0f, -0.25f, 0.0f};
    globalInput.stand.groundNormal = {0.0f, 0.0f, 1.0f, 0.0f};
    globalInput.captureSupportGeometryOnly = true;
    MetalArticulatedOperatorResult globalCaptured;
    const auto globalDiagnostics = context.run(
        model, globalInput, globalCaptured);
    const bool globalPlaneRowsPreserved =
        globalCaptured.supportGeometryContacts.size() == 1u &&
        globalCaptured.supportGeometryContacts[0].bodyIndex == contacts[0].bodyIndex &&
        globalCaptured.supportGeometryContacts[0].pointQueryIndex ==
            contacts[0].pointQueryIndex &&
        globalCaptured.supportGeometryContacts[0].sourceGeometryIndex ==
            contacts[0].sourceGeometryIndex &&
        globalCaptured.supportGeometryContacts[0].boundaryNormalVelocity ==
            contacts[0].boundaryNormalVelocity &&
        std::memcmp(
            &globalCaptured.supportGeometryContacts[0].frictionSlopAndStabilization,
            &contacts[0].frictionSlopAndStabilization, sizeof(mr_float4)) == 0 &&
        std::memcmp(&globalCaptured.supportGeometryContacts[0].planePoint,
                    &contacts[0].planePoint, sizeof(mr_float4)) == 0 &&
        std::memcmp(&globalCaptured.supportGeometryContacts[0].planeNormal,
                    &contacts[0].planeNormal, sizeof(mr_float4)) == 0;
    require(globalDiagnostics.succeeded() && globalProbe.encodeCalls == 1u &&
                globalProbe.abortCalls == 0u && globalPlaneRowsPreserved &&
                globalDiagnostics.completedStandSteps == 0u &&
                globalDiagnostics.residentStateTransactionFingerprint == 0u &&
                globalDiagnostics.residentStateGeneration == 0u &&
                globalCaptured.standQ.empty() && globalCaptured.standV.empty() &&
                globalCaptured.standStatuses.empty() &&
                globalCaptured.standContactImpulses.empty() &&
                exactBytes(globalCaptured.pointWorld, before.pointWorld) &&
                exactBytes(globalCaptured.pointPositionLow, before.pointPositionLow) &&
                exactBytes(globalCaptured.pointJacobians, before.pointJacobians) &&
                exactBytes(globalCaptured.bodyPoses, before.bodyPoses) &&
                exactBytes(globalCaptured.rootTranslations, before.rootTranslations) &&
                std::all_of(globalCaptured.generalizedImpulse.begin(),
                    globalCaptured.generalizedImpulse.end(),
                    [](float value) { return value == 0.0f; }) &&
                std::all_of(globalCaptured.deltaVelocity.begin(),
                    globalCaptured.deltaVelocity.end(),
                    [](float value) { return value == 0.0f; }),
            "global-plane callback capture must preserve the common plane and remain noncommitting");

    GeometryCaptureProbe mismatchProbe;
    const MetalNumiHumanSupportGeometryProgram mismatchGeometryProgram{
        .context = &mismatchProbe,
        .encodePreDynamics = encodeCaptureGlobalPlane,
        .abort = abortCapturePlane,
        .fingerprint = 0x474c4f42414c4d49ull,
        .usePerContactSupportPlanes = false,
    };
    auto mismatchedPlaneInput = globalInput;
    contacts[0].planePoint.z = -0.24f;
    mismatchedPlaneInput.stand.contacts = contacts;
    mismatchedPlaneInput.stand.supportGeometryProgram = mismatchGeometryProgram;
    MetalArticulatedOperatorResult mismatchedPlaneResult;
    const auto mismatchedPlaneDiagnostics = context.run(
        model, mismatchedPlaneInput, mismatchedPlaneResult);
    require(!mismatchedPlaneDiagnostics.succeeded() &&
                mismatchProbe.encodeCalls == 0u && mismatchProbe.abortCalls == 0u,
            "global-plane capture must reject a row off the registered plane before callback execution");

    contacts[0].planePoint.z = -0.25f;
    GeometryCaptureProbe modeProbe;
    const MetalNumiHumanSupportGeometryProgram modeGeometryProgram{
        .context = &modeProbe,
        .encodePreDynamics = encodeCaptureGlobalPlane,
        .abort = abortCapturePlane,
        .fingerprint = 0x474c4f42414c4d4full,
        .usePerContactSupportPlanes = false,
    };
    auto mismatchedModeInput = globalInput;
    mismatchedModeInput.stand.usePerContactSupportPlanes = true;
    mismatchedModeInput.stand.supportGeometryProgram = modeGeometryProgram;
    MetalArticulatedOperatorResult mismatchedModeResult;
    const auto mismatchedModeDiagnostics = context.run(
        model, mismatchedModeInput, mismatchedModeResult);
    require(!mismatchedModeDiagnostics.succeeded() &&
                modeProbe.encodeCalls == 0u && modeProbe.abortCalls == 0u,
            "support capture must reject input/provider plane-mode mismatch before callback execution");

    GeometryCaptureProbe mutationProbe;
    const MetalNumiHumanSupportGeometryProgram mutationGeometryProgram{
        .context = &mutationProbe,
        .encodePreDynamics = encodeCaptureMutatingGlobalPlane,
        .abort = abortCapturePlane,
        .fingerprint = 0x474c4f42414c4d55ull,
        .usePerContactSupportPlanes = false,
    };
    auto mutatedPlaneInput = globalInput;
    mutatedPlaneInput.stand.supportGeometryProgram = mutationGeometryProgram;
    MetalArticulatedOperatorResult mutatedPlaneResult;
    const auto mutatedPlaneDiagnostics = context.run(
        model, mutatedPlaneInput, mutatedPlaneResult);
    require(!mutatedPlaneDiagnostics.succeeded() &&
                mutationProbe.encodeCalls == 1u,
            "global-plane capture must reject callback mutation during result collection");
}
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            const bool sourceMass128Only = argc == 3 &&
                std::strcmp(argv[1], "--source-mass-128") == 0;
            require(argc == 2 || sourceMass128Only,
                "usage: per-contact-plane-test [--source-mass-128] /path/MetalRobo.metallib");
            const char* metallibPath = sourceMass128Only ? argv[2] : argv[1];
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil) {
                std::cout << "gpu_available=false execution=not_run\n";
                return 77;
            }
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithURL:
                [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallibPath]]
                error:&error];
            require(library != nil, "cannot load production MetalRobo metallib");
            if (sourceMass128Only) {
                testSourceMass128(device, metallibPath);
                std::cout << "source_mass_128=passed action_fp64_oracle=passed"
                    << " solve_roundtrip=passed repeated_factor_solve=passed\n";
                return 0;
            }
            MTLFunctionConstantValues* constants =
                [[MTLFunctionConstantValues alloc] init];
            const bool disabled = false;
            [constants setConstantValue:&disabled type:MTLDataTypeBool atIndex:2];
            [constants setConstantValue:&disabled type:MTLDataTypeBool atIndex:3];
            id<MTLFunction> function = [library newFunctionWithName:
                @"mr_numi_human_stand_step" constantValues:constants error:&error];
            require(function != nil, "production stand-step kernel is absent");
            id<MTLComputePipelineState> pipeline =
                [device newComputePipelineStateWithFunction:function error:&error];
            require(pipeline != nil, "cannot compile production stand-step pipeline");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(queue != nil, "cannot create fixture command queue");

            auto perContact = allocate(device, 0.002f);
            auto* perDispatch = static_cast<MRNumiHumanStandDispatchGPU*>(perContact[4].contents);
            perDispatch->supportContactPlaneMode =
                MR_NUMI_HUMAN_STAND_SUPPORT_PLANE_PER_CONTACT;
            perDispatch->supportContactPlaneCount = 1u;
            run(pipeline, queue, perContact);
            const auto* perStatus = static_cast<const MRNumiHumanStandStatusGPU*>(perContact[17].contents);
            const auto* perVelocity = static_cast<const float*>(perContact[6].contents);
            const float contactVelocityX = perVelocity[0] - 0.5f * perVelocity[4];
            const float contactVelocityY = perVelocity[1] + 0.5f * perVelocity[3];
            require(perStatus[0].code == MR_NUMI_HUMAN_STAND_SUCCESS &&
                    perStatus[0].completedSteps == 1u &&
                    std::abs(contactVelocityY) < 2.0e-4f &&
                    std::abs(contactVelocityX) < 2.0e-4f &&
                    perStatus[0].constraintImpulseDiagnostics.x > 0.3f &&
                    perStatus[0].constraintImpulseDiagnostics.y > 0.05f,
                    "per-contact plane did not resolve normal and tangent point velocity in its own basis");

            auto global = allocate(device, 0.002f);
            auto* globalDispatch = static_cast<MRNumiHumanStandDispatchGPU*>(global[4].contents);
            globalDispatch->supportContactPlaneMode =
                MR_NUMI_HUMAN_STAND_SUPPORT_PLANE_GLOBAL;
            globalDispatch->supportContactPlaneCount = 0u;
            run(pipeline, queue, global);
            const auto* globalStatus = static_cast<const MRNumiHumanStandStatusGPU*>(global[17].contents);
            const auto* globalVelocity = static_cast<const float*>(global[6].contents);
            require(globalStatus[0].code == MR_NUMI_HUMAN_STAND_SUCCESS &&
                    globalStatus[0].completedSteps == 1u &&
                    std::abs(globalVelocity[1] + 1.0f) < 2.0e-4f &&
                    std::abs(globalVelocity[0] + 0.05f) < 2.0e-4f,
                    "legacy global plane changed orthogonal source velocities");

            auto invalid = allocate(device, 0.002f);
            auto* invalidDispatch = static_cast<MRNumiHumanStandDispatchGPU*>(invalid[4].contents);
            invalidDispatch->supportContactPlaneMode =
                MR_NUMI_HUMAN_STAND_SUPPORT_PLANE_PER_CONTACT;
            invalidDispatch->supportContactPlaneCount = 1u;
            auto* invalidContact = static_cast<MRNumiHumanStandContactGPU*>(invalid[11].contents);
            invalidContact->planeNormal = {0.0f, 2.0f, 0.0f, 0.0f};
            run(pipeline, queue, invalid);
            const auto* invalidStatus = static_cast<const MRNumiHumanStandStatusGPU*>(invalid[17].contents);
            const auto* invalidVelocity = static_cast<const float*>(invalid[6].contents);
            require(invalidStatus[0].code == MR_NUMI_HUMAN_STAND_INVALID_DISPATCH &&
                    invalidStatus[0].completedSteps == 0u &&
                    std::abs(invalidVelocity[0] + 0.05f) < 1.0e-7f &&
                    std::abs(invalidVelocity[1] + 1.0f) < 1.0e-7f,
                    "malformed per-contact normal was not rejected atomically");

            auto countMismatch = allocate(device, 0.002f);
            auto* mismatchDispatch = static_cast<MRNumiHumanStandDispatchGPU*>(countMismatch[4].contents);
            mismatchDispatch->supportContactPlaneMode =
                MR_NUMI_HUMAN_STAND_SUPPORT_PLANE_PER_CONTACT;
            mismatchDispatch->supportContactPlaneCount = 0u;
            run(pipeline, queue, countMismatch);
            const auto* mismatchStatus = static_cast<const MRNumiHumanStandStatusGPU*>(countMismatch[17].contents);
            require(mismatchStatus[0].code == MR_NUMI_HUMAN_STAND_INVALID_DISPATCH &&
                    mismatchStatus[0].completedSteps == 0u,
                    "per-contact mode accepted a mismatched plane count");

            testNoncommittingSupportCapture(metallibPath);
            testSourceMass128(device, metallibPath);

            std::cout << "gpu_available=true device=\"" << device.name.UTF8String
                << "\" kernel=mr_numi_human_stand_step checks=13"
                << " per_contact_normal_tangent_response=passed global_plane_parity=passed"
                << " invalid_plane_atomic_reject=passed mode_count_reject=passed"
                << " noncommitting_callback_capture=passed same_context_fk_j_parity=passed"
                << " global_plane_capture_passthrough=passed global_plane_mismatch_reject=passed"
                << " capture_mode_mismatch_reject=passed global_callback_plane_mutation_reject=passed"
                << " source_mass_128_action_solve=passed\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
