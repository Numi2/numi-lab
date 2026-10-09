#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/numi_human_stand_gpu.h"
#include "metalrobo/compensated_translation_gpu.h"
#include "metalrobo/numi_human_tendon_gpu.h"
#include "metalrobo/numi_human_joint_equality_gpu.h"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

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
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            require(argc == 2, "usage: per-contact-plane-test /path/MetalRobo.metallib");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil) {
                std::cout << "gpu_available=false execution=not_run\n";
                return 77;
            }
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithURL:
                [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]
                error:&error];
            require(library != nil, "cannot load production MetalRobo metallib");
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

            std::cout << "gpu_available=true device=\"" << device.name.UTF8String
                << "\" kernel=mr_numi_human_stand_step checks=4"
                << " per_contact_normal_tangent_response=passed global_plane_parity=passed"
                << " invalid_plane_atomic_reject=passed mode_count_reject=passed\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
