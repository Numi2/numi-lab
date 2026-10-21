#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/numanx_coupled_human_gpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void near(float actual, double expected, double tolerance, const char* label) {
    if (!std::isfinite(actual) ||
        std::abs(static_cast<double>(actual) - expected) > tolerance)
        throw std::runtime_error(label);
}

id<MTLBuffer> buffer(id<MTLDevice> device, const void* source,
    NSUInteger length) {
    return [device newBufferWithBytes:source length:length
        options:MTLResourceStorageModeShared];
}

id<MTLBuffer> scratch(id<MTLDevice> device, NSUInteger length) {
    return [device newBufferWithLength:length
        options:MTLResourceStorageModeShared];
}

void runKernel(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
    id<MTLBuffer> __strong const* buffers, NSUInteger count) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    require(command && encoder, "cannot create Metal command");
    [encoder setComputePipelineState:pipeline];
    for (NSUInteger index = 0; index < count; ++index)
        [encoder setBuffer:buffers[index] offset:0 atIndex:index];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
        "Metal KKT dispatch failed");
}

MRNumanXCoupledHumanDispatchGPU makeDispatch() {
    MRNumanXCoupledHumanDispatchGPU value{};
    value.abiVersion = MR_NUMANX_COUPLED_HUMAN_ABI_VERSION;
    value.operation = MR_NUMANX_COUPLED_HUMAN_INVERSE_MASS_PRECONDITIONER;
    value.environmentCount = 1u;
    value.flags = 0u;
    value.nq = 3u;
    value.nv = 2u;
    value.sourceQStride = 3u;
    value.sourceVStride = 2u;
    value.generalizedVectorStride = 2u;
    value.candidateQStride = 3u;
    value.candidateBodyStride = 2u;
    value.statusStride = 1u;
    value.factorStride = 4u;
    value.reactionStride = 2u;
    value.transactionSlot = 1u;
    value.stepIndex = 3u;
    value.bodyCount = 1u;
    value.timestepAndInverse = {0.01f, 100.0f, 0.0f, 0.0f};
    value.programFingerprint = 101u;
    value.transactionFingerprint = 202u;
    value.linearizationEpoch = 303u;
    value.slotGeneration = 404u;
    return value;
}

MRNumanXCoupledHumanSlotMetadataGPU makeMetadata(
    const MRNumanXCoupledHumanDispatchGPU& dispatch) {
    MRNumanXCoupledHumanSlotMetadataGPU value{};
    value.abiVersion = MR_NUMANX_COUPLED_HUMAN_ABI_VERSION;
    value.structSize = sizeof(value);
    value.stage = MR_NUMANX_COUPLED_HUMAN_STAGE_CANDIDATE_VALIDATED;
    value.transactionSlot = dispatch.transactionSlot;
    value.stepIndex = dispatch.stepIndex;
    value.environmentCount = dispatch.environmentCount;
    value.operationWitnesses = MR_NUMANX_COUPLED_HUMAN_WITNESS_CANDIDATE;
    value.qCoordinateCount = dispatch.nq;
    value.dofCount = dispatch.nv;
    value.sourceQStride = dispatch.sourceQStride;
    value.sourceVStride = dispatch.sourceVStride;
    value.factorStride = dispatch.factorStride;
    value.reactionStride = dispatch.reactionStride;
    value.programFingerprint = dispatch.programFingerprint;
    value.transactionFingerprint = dispatch.transactionFingerprint;
    value.linearizationEpoch = dispatch.linearizationEpoch;
    value.slotGeneration = dispatch.slotGeneration;
    return value;
}

MRNumanXCoupledHumanStatusGPU makeStatus(
    const MRNumanXCoupledHumanDispatchGPU& dispatch) {
    MRNumanXCoupledHumanStatusGPU value{};
    value.abiVersion = MR_NUMANX_COUPLED_HUMAN_ABI_VERSION;
    value.decision = MR_NUMANX_COUPLED_HUMAN_PENDING;
    value.environment = 0u;
    value.stepIndex = dispatch.stepIndex;
    return value;
}
}

int main(int argc, char** argv) {
    try {
        @autoreleasepool {
            require(argc == 2,
                "usage: numanx-human-bilateral-kkt-check METALLIB");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "Metal device unavailable");
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithURL:
                [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]
                error:&error];
            require(library != nil, "cannot load bilateral KKT metallib");
            id<MTLFunction> prepareFunction = [library newFunctionWithName:
                @"numanx_human_bilateral_kkt_prepare_fixture"];
            id<MTLFunction> kktFunction = [library newFunctionWithName:
                @"numanx_coupled_human_bilateral_kkt_inverse"];
            require(prepareFunction && kktFunction,
                "bilateral KKT test or production kernel missing");
            id<MTLComputePipelineState> prepare = [device
                newComputePipelineStateWithFunction:prepareFunction error:&error];
            id<MTLComputePipelineState> kkt = [device
                newComputePipelineStateWithFunction:kktFunction error:&error];
            require(prepare && kkt, "cannot compile bilateral KKT pipelines");

            // A0 = L L^T = [[4,1],[1,3]], J=[1,1], S=J A0^-1 J^T=5/11.
            const std::array<float, 4> a0Factor{
                2.0f, 0.0f, 0.5f, std::sqrt(2.75f)
            };
            const std::array<float, 2> forceRhs{1.0f, -2.0f};
            const std::array<float, 1> constraintRhs{0.2f};
            id<MTLBuffer> a0 = buffer(device, a0Factor.data(), sizeof(a0Factor));
            id<MTLBuffer> force = buffer(device, forceRhs.data(), sizeof(forceRhs));
            id<MTLBuffer> constraint = buffer(
                device, constraintRhs.data(), sizeof(constraintRhs));
            id<MTLBuffer> rows = scratch(device, 2u * sizeof(float));
            id<MTLBuffer> responses = scratch(device, 2u * sizeof(float));
            id<MTLBuffer> schur = scratch(device, sizeof(float));
            id<MTLBuffer> scale = scratch(device, sizeof(float));
            id<MTLBuffer> pivots = scratch(device, sizeof(float));
            id<MTLBuffer> prepareWitness = scratch(device, sizeof(float));
            require(a0 && force && constraint && rows && responses && schur &&
                scale && pivots && prepareWitness, "Metal test buffer unavailable");

            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(queue != nil, "Metal command queue unavailable");
            id<MTLBuffer> prepareBuffers[] = {
                a0, rows, responses, schur, scale, pivots, prepareWitness
            };
            runKernel(queue, prepare, prepareBuffers, 7u);
            near(static_cast<const float*>(prepareWitness.contents)[0],
                5.0 / 11.0, 2.0e-6, "source Delassus scalar mismatch");

            const auto dispatchValue = makeDispatch();
            const auto kktValue = MRNumanXCoupledHumanBilateralKKTDispatchGPU{
                MR_NUMANX_COUPLED_HUMAN_BILATERAL_KKT_ABI_VERSION,
                sizeof(MRNumanXCoupledHumanBilateralKKTDispatchGPU),
                1u, 2u, 2u, 1u, 1u, 1u, 2u, 1u, 2u, 1u,
                {0u, 0u, 0u, 0u}
            };
            const auto metadataValue = makeMetadata(dispatchValue);
            auto statusValue = makeStatus(dispatchValue);
            id<MTLBuffer> dispatch = buffer(
                device, &dispatchValue, sizeof(dispatchValue));
            id<MTLBuffer> kktDispatch = buffer(
                device, &kktValue, sizeof(kktValue));
            id<MTLBuffer> metadata = buffer(
                device, &metadataValue, sizeof(metadataValue));
            id<MTLBuffer> status = buffer(device, &statusValue, sizeof(statusValue));
            id<MTLBuffer> deltaV = scratch(device, 2u * sizeof(float));
            id<MTLBuffer> multipliers = scratch(device, sizeof(float));
            require(dispatch && kktDispatch && metadata && status &&
                deltaV && multipliers, "KKT dispatch buffers unavailable");

            id<MTLBuffer> kktBuffers[] = {
                dispatch, kktDispatch, a0, rows, responses, schur, scale,
                pivots, force, constraint, deltaV, multipliers, status, metadata
            };
            runKernel(queue, kkt, kktBuffers, 14u);
            const auto* actual = static_cast<const float*>(deltaV.contents);
            const auto* actualLambda =
                static_cast<const float*>(multipliers.contents);
            const auto* statusAfterSolve =
                static_cast<const MRNumanXCoupledHumanStatusGPU*>(status.contents);
            near(actual[0], 0.68, 3.0e-6, "KKT delta-v[0] mismatch");
            near(actual[1], -0.48, 3.0e-6, "KKT delta-v[1] mismatch");
            near(actualLambda[0], -1.24, 3.0e-6, "KKT multiplier mismatch");
            const double forceResidual0 =
                4.0 * actual[0] + actual[1] + actualLambda[0] - forceRhs[0];
            const double forceResidual1 =
                actual[0] + 3.0 * actual[1] + actualLambda[0] - forceRhs[1];
            const double equalityResidual =
                actual[0] + actual[1] - constraintRhs[0];
            require(std::max({std::abs(forceResidual0), std::abs(forceResidual1),
                std::abs(equalityResidual)}) <= 3.0e-6,
                "production KKT action/reaction residual exceeded tolerance");
            require(statusAfterSolve->decision ==
                    MR_NUMANX_COUPLED_HUMAN_PENDING,
                "successful KKT solve changed transaction acceptance state");

            // A singular Delassus scale rejects the candidate and leaves the
            // caller-owned delta-v/multiplier sentinels intact.
            const std::array<float, 1> invalidScale{0.0f};
            id<MTLBuffer> badScale = buffer(
                device, invalidScale.data(), sizeof(invalidScale));
            const std::array<float, 2> deltaSentinel{123.0f, -456.0f};
            const std::array<float, 1> lambdaSentinel{789.0f};
            std::memcpy(deltaV.contents, deltaSentinel.data(), sizeof(deltaSentinel));
            std::memcpy(multipliers.contents, lambdaSentinel.data(),
                sizeof(lambdaSentinel));
            *static_cast<MRNumanXCoupledHumanStatusGPU*>(status.contents) =
                makeStatus(dispatchValue);
            kktBuffers[6] = badScale;
            runKernel(queue, kkt, kktBuffers, 14u);
            const auto* rejected =
                static_cast<const MRNumanXCoupledHumanStatusGPU*>(status.contents);
            near(static_cast<const float*>(deltaV.contents)[0],
                123.0, 0.0, "failed solve modified delta-v");
            near(static_cast<const float*>(deltaV.contents)[1],
                -456.0, 0.0, "failed solve modified delta-v");
            near(static_cast<const float*>(multipliers.contents)[0],
                789.0, 0.0, "failed solve modified reaction multiplier");
            require(rejected->decision == MR_NUMANX_COUPLED_HUMAN_REJECT_HUMAN &&
                rejected->humanCode ==
                    MR_NUMANX_COUPLED_HUMAN_SERVICE_INVALID_FACTOR,
                "singular Schur block was not rejected transactionally");
            std::printf(
                "{\"passed\":true,\"mode\":\"metal\",\"rows\":1,"
                "\"dofs\":2,\"production_kernel\":true,"
                "\"residual\":true,\"rollback\":true,"
                "\"unilateral\":false}\n");
        }
        return 0;
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "bilateral KKT check failed: %s\n",
            exception.what());
        return 1;
    }
}
