#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/shared.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef NUMI_MATTER_METALLIB
#error "CMake must define NUMI_MATTER_METALLIB"
#endif

namespace {

struct alignas(8) NativeFloat2 { float x; float y; };

unsigned checks = 0u;

void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
id<MTLBuffer> makeBuffer(id<MTLDevice> device, const std::vector<T>& values) {
    require(!values.empty(), "native fixture attempted an empty Metal buffer");
    id<MTLBuffer> buffer = [device newBufferWithBytes:values.data()
        length:values.size() * sizeof(T)
        options:MTLResourceStorageModeShared];
    require(buffer != nil, "Metal buffer allocation failed");
    return buffer;
}

template <typename T>
std::vector<T> readBuffer(id<MTLBuffer> buffer, std::size_t count) {
    require(buffer != nil && [buffer length] >= count * sizeof(T),
        "Metal readback extent is invalid");
    const T* values = static_cast<const T*>([buffer contents]);
    return std::vector<T>(values, values + count);
}

id<MTLComputePipelineState> pipeline(
    id<MTLDevice> device, id<MTLLibrary> library, const char* name) {
    NSString* functionName = [NSString stringWithUTF8String:name];
    id<MTLFunction> function = [library newFunctionWithName:functionName];
    require(function != nil, std::string("missing Metal function: ") + name);
    NSError* error = nil;
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:&error];
    require(result != nil, error != nil
        ? std::string("pipeline creation failed: ") +
            [[error localizedDescription] UTF8String]
        : std::string("pipeline creation failed: ") + name);
    return result;
}

void dispatch(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> state,
    const std::vector<std::pair<NSUInteger, id<MTLBuffer>>>& bindings,
    const NSUInteger groups,
    const NSUInteger threads) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        command != nil ? [command computeCommandEncoder] : nil;
    require(command != nil && encoder != nil,
        "cannot allocate Metal command encoder");
    [encoder setComputePipelineState:state];
    for (const auto& [index, buffer] : bindings)
        [encoder setBuffer:buffer offset:0u atIndex:index];
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(threads, 1u, 1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
        command.error != nil
            ? std::string("Metal command failed: ") +
                [[command.error localizedDescription] UTF8String]
            : "Metal command did not complete");
    (void)device;
}

void resetStatuses(
    id<MTLBuffer> diagnosticBuffer,
    id<MTLBuffer> productionBuffer,
    const NMMatterStatusGPU& diagnostic,
    const NMMatterStatusGPU& production) {
    *static_cast<NMMatterStatusGPU*>([diagnosticBuffer contents]) = diagnostic;
    *static_cast<NMMatterStatusGPU*>([productionBuffer contents]) = production;
}

bool close(float a, float b, float tolerance = 1.0e-6f) {
    return std::isfinite(a) && std::isfinite(b) &&
        std::abs(a - b) <= tolerance;
}

} // namespace

int main() {
    @autoreleasepool {
        try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "Metal device unavailable");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(queue != nil, "Metal command queue unavailable");

            NSError* error = nil;
            NSURL* libraryURL = [NSURL fileURLWithPath:
                [NSString stringWithUTF8String:NUMI_MATTER_METALLIB]];
            id<MTLLibrary> library =
                [device newLibraryWithURL:libraryURL error:&error];
            require(library != nil, error != nil
                ? std::string("cannot load Matter metallib: ") +
                    [[error localizedDescription] UTF8String]
                : "cannot load Matter metallib");

            auto finalize = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_finalize_cycle");
            auto accumulate = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_accumulate");
            auto accumulateFields = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_accumulate_fields");
            auto measure = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_measure_true_residual");

            NMMatterDispatchGPU dispatchData{};
            dispatchData.abiVersion = NM_MATTER_ABI_VERSION;
            dispatchData.environmentCount = 1u;
            dispatchData.objectCount = 1u;
            dispatchData.femNodeCount = 1u;
            NMFGMRESLayoutGPU layout{};
            layout.unknownCount = 2u;
            NMMixedSolverGPU solver{};
            solver.nonlinearIterations.y = 1u;
            solver.residualTolerances.x = 1.0e-5f;
            solver.regularization.x = 1.0e-6f;
            NMIncidenceRangeGPU nodeRange{};
            nodeRange.objectIndex = 0u;

            const std::size_t hessenbergCount =
                std::size_t(NM_MIXED_FGMRES_RESTART + 1u) *
                std::size_t(NM_MIXED_FGMRES_RESTART);
            std::vector<float> hessenberg(hessenbergCount, 0.0f);
            std::vector<float> leastSquares(
                NM_MIXED_FGMRES_RESTART + 1u, 0.0f);
            std::vector<NativeFloat2> rotations(NM_MIXED_FGMRES_RESTART);
            std::vector<float> restartCoefficients(
                NM_MIXED_FGMRES_RESTART + 1u, 0.0f);
            hessenberg[0] = 1.0f;
            leastSquares[0] = 0.5f;

            NMFGMRESStateGPU initialState{};
            initialState.diagnostics = {1.0e-3f, 2.23606798f, 0.0f, 1.0f};
            initialState.nonlinear = {1.0f, 1.0f, 0.0f, 0.0f};
            NMMatterStatusGPU healthy{};
            healthy.code = NM_STATUS_SUCCESS;
            healthy.environment = 0u;

            auto dispatchBuffer = makeBuffer(device,
                std::vector<NMMatterDispatchGPU>{dispatchData});
            auto layoutBuffer = makeBuffer(device,
                std::vector<NMFGMRESLayoutGPU>{layout});
            auto solverBuffer = makeBuffer(device,
                std::vector<NMMixedSolverGPU>{solver});
            auto hessenbergBuffer = makeBuffer(device, hessenberg);
            auto leastSquaresBuffer = makeBuffer(device, leastSquares);
            auto rotationsBuffer = makeBuffer(device, rotations);
            auto restartCoefficientsBuffer =
                makeBuffer(device, restartCoefficients);
            auto stateBuffer = makeBuffer(device,
                std::vector<NMFGMRESStateGPU>{initialState});
            auto statusBuffer = makeBuffer(device,
                std::vector<NMMatterStatusGPU>{healthy});
            auto rangesBuffer = makeBuffer(device,
                std::vector<NMIncidenceRangeGPU>{nodeRange});
            std::vector<nm_float4> basis(
                2u * std::size_t(NM_MIXED_FGMRES_RESTART + 1u),
                nm_float4{0.0f, 0.0f, 0.0f, 0.0f});
            basis[0] = {2.0f, 0.0f, 0.0f, 0.0f};
            basis[1] = {4.0f, 0.0f, 0.0f, 0.0f};
            auto basisBuffer = makeBuffer(device, basis);
            std::vector<nm_float4> solutionValues(
                2u, nm_float4{0.0f, 0.0f, 0.0f, 0.0f});
            auto solutionBuffer = makeBuffer(device, solutionValues);
            std::vector<nm_float4> rhsValues{
                {2.0f, 0.0f, 0.0f, 0.0f},
                {4.0f, 0.0f, 0.0f, 0.0f}};
            auto rhsBuffer = makeBuffer(device, rhsValues);
            std::vector<nm_float4> operatorValues(
                2u, nm_float4{0.0f, 0.0f, 0.0f, 0.0f});
            auto operatorBuffer = makeBuffer(device, operatorValues);
            std::vector<nm_float4> residualValues(
                2u, nm_float4{-7.0f, -7.0f, -7.0f, -7.0f});
            auto residualBuffer = makeBuffer(device, residualValues);
            auto metricsBuffer = makeBuffer(device,
                std::vector<nm_float4>{nm_float4{-1.0f, -1.0f, -1.0f, -1.0f}});
            auto diagnosticBuffer = makeBuffer(device,
                std::vector<NMMatterStatusGPU>{healthy});

            const std::uint32_t offset = 0u;
            const std::uint32_t finalCycle = 1u;
            const float forcing = 0.1f;
            const std::uint32_t defer = 1u;
            dispatch(device, queue, finalize, {
                {0u, dispatchBuffer}, {1u, solverBuffer},
                {2u, hessenbergBuffer}, {3u, leastSquaresBuffer},
                {4u, stateBuffer}, {5u, statusBuffer},
                {6u, makeBuffer(device, std::vector<std::uint32_t>{offset})},
                {7u, makeBuffer(device, std::vector<std::uint32_t>{finalCycle})},
                {8u, makeBuffer(device, std::vector<float>{forcing})},
                {9u, rotationsBuffer}, {10u, restartCoefficientsBuffer},
                {11u, makeBuffer(device, std::vector<std::uint32_t>{defer})}},
                1u, 32u);
            auto afterFinalize = readBuffer<NMFGMRESStateGPU>(stateBuffer, 1u);
            require(afterFinalize[0].diagnostics.z == 0.0f,
                "deferred mode latched Arnoldi-only completion");

            const std::uint32_t firstCycle = 0u;
            auto cycleBuffer = makeBuffer(device,
                std::vector<std::uint32_t>{firstCycle});
            dispatch(device, queue, accumulate, {
                {0u, dispatchBuffer}, {30u, layoutBuffer},
                {1u, solverBuffer}, {2u, rangesBuffer},
                {3u, basisBuffer}, {4u, leastSquaresBuffer},
                {5u, stateBuffer}, {6u, solutionBuffer}, {7u, cycleBuffer}},
                1u, 1u);
            dispatch(device, queue, accumulateFields, {
                {0u, dispatchBuffer}, {30u, layoutBuffer},
                {1u, solverBuffer}, {2u, rangesBuffer},
                {3u, basisBuffer}, {4u, leastSquaresBuffer},
                {5u, stateBuffer}, {6u, solutionBuffer}, {7u, cycleBuffer}},
                1u, 1u);
            solutionValues = readBuffer<nm_float4>(solutionBuffer, 2u);
            require(close(solutionValues[0].x, 1.0f) &&
                    close(solutionValues[1].x, 2.0f),
                "production accumulation did not form the authored correction");

            // The focused algebraic fixture uses A=I: operatorValue is the
            // actual accumulated production correction. This isolates the
            // native completion/replacement kernels; it is not a full FEM
            // operator qualification.
            operatorValues = solutionValues;
            operatorValues[0].y = operatorValues[0].z = operatorValues[0].w = 0.0f;
            operatorValues[1].y = operatorValues[1].z = operatorValues[1].w = 0.0f;
            std::memcpy([operatorBuffer contents], operatorValues.data(),
                operatorValues.size() * sizeof(nm_float4));
            const std::uint32_t replace = 1u;
            const std::uint32_t propagate = 1u;
            auto replaceBuffer = makeBuffer(device,
                std::vector<std::uint32_t>{replace});
            auto propagateBuffer = makeBuffer(device,
                std::vector<std::uint32_t>{propagate});

            dispatch(device, queue, measure, {
                {0u, dispatchBuffer}, {30u, layoutBuffer},
                {1u, solverBuffer}, {2u, rhsBuffer},
                {3u, operatorBuffer}, {4u, residualBuffer},
                {5u, stateBuffer}, {6u, metricsBuffer},
                {7u, makeBuffer(device, std::vector<float>{forcing})},
                {8u, diagnosticBuffer}, {9u, replaceBuffer},
                {10u, statusBuffer}, {11u, propagateBuffer}},
                1u, 32u);
            auto rejected = readBuffer<NMFGMRESStateGPU>(stateBuffer, 1u);
            auto replacedResidual = readBuffer<nm_float4>(residualBuffer, 2u);
            require(rejected[0].diagnostics.z == 0.0f,
                "true residual incorrectly accepted false Arnoldi convergence");
            require(close(replacedResidual[0].x, 1.0f) &&
                    close(replacedResidual[1].x, 2.0f),
                "same-operator true residual did not replace the restart vector");

            // A zero true residual meets the unchanged absolute/forcing gate.
            rhsValues = solutionValues;
            std::memcpy([rhsBuffer contents], rhsValues.data(),
                rhsValues.size() * sizeof(nm_float4));
            auto beforeSuccess = rejected[0];
            beforeSuccess.diagnostics.z = 0.0f;
            std::memcpy([stateBuffer contents], &beforeSuccess,
                sizeof(beforeSuccess));
            std::memcpy([residualBuffer contents], residualValues.data(),
                residualValues.size() * sizeof(nm_float4));
            dispatch(device, queue, measure, {
                {0u, dispatchBuffer}, {30u, layoutBuffer},
                {1u, solverBuffer}, {2u, rhsBuffer},
                {3u, operatorBuffer}, {4u, residualBuffer},
                {5u, stateBuffer}, {6u, metricsBuffer},
                {7u, makeBuffer(device, std::vector<float>{forcing})},
                {8u, diagnosticBuffer}, {9u, replaceBuffer},
                {10u, statusBuffer}, {11u, propagateBuffer}},
                1u, 32u);
            auto accepted = readBuffer<NMFGMRESStateGPU>(stateBuffer, 1u);
            require(accepted[0].diagnostics.z == 1.0f,
                "resolved zero true residual did not latch completion");

            // Native regression for the actual coupon's inner absolute
            // shortcut: outer tolerance is not a sufficient inexact solve
            // when the current nonlinear RHS remains above that tolerance.
            auto strictTrial = [&](float rhsNorm, float trueNorm,
                                   std::uint32_t mode, bool expected,
                                   const char* label) {
                NMMixedSolverGPU tight = solver;
                tight.residualTolerances.x = 5.0e-7f;
                tight.regularization.x = 1.0e-12f;
                std::memcpy([solverBuffer contents], &tight, sizeof(tight));
                NMFGMRESStateGPU trial{};
                trial.nonlinear = {1.0f,1.0f,0.0f,0.0f};
                trial.diagnostics = {0.0f,rhsNorm,1.0f,0.0f};
                std::memcpy([stateBuffer contents], &trial, sizeof(trial));
                std::vector<nm_float4> trhs(2u, {0.0f,0.0f,0.0f,0.0f});
                std::vector<nm_float4> top(2u, {0.0f,0.0f,0.0f,0.0f});
                trhs[0].x = rhsNorm;
                top[0].x = rhsNorm - trueNorm;
                std::memcpy([rhsBuffer contents], trhs.data(), 2u*sizeof(nm_float4));
                std::memcpy([operatorBuffer contents], top.data(), 2u*sizeof(nm_float4));
                std::memcpy([residualBuffer contents], residualValues.data(), 2u*sizeof(nm_float4));
                std::memcpy([replaceBuffer contents], &mode, sizeof(mode));
                resetStatuses(diagnosticBuffer,statusBuffer,healthy,healthy);
                const float tightForcing = 0.015625f;
                dispatch(device,queue,measure,{
                    {0u,dispatchBuffer},{30u,layoutBuffer},
                    {1u,solverBuffer},{2u,rhsBuffer},{3u,operatorBuffer},
                    {4u,residualBuffer},{5u,stateBuffer},{6u,metricsBuffer},
                    {7u,makeBuffer(device,std::vector<float>{tightForcing})},
                    {8u,diagnosticBuffer},{9u,replaceBuffer},
                    {10u,statusBuffer},{11u,propagateBuffer}},1u,32u);
                const auto got = readBuffer<NMFGMRESStateGPU>(stateBuffer,1u);
                const auto rr = readBuffer<nm_float4>(residualBuffer,2u);
                require((got[0].diagnostics.z==1.0f)==expected,label);
                require(std::abs(rr[0].x-trueNorm)<=2.0e-13f &&
                        rr[1].x==0.0f,"strict mode did not replace exact tested residual");
                const auto st=readBuffer<NMMatterStatusGPU>(statusBuffer,1u);
                require(st[0].code==NM_STATUS_SUCCESS,"strict forcing poisoned production status");
            };
            strictTrial(6.3580319e-7f,4.55182175e-7f,1u,true,
                        "legacy absolute shortcut control changed");
            strictTrial(6.3580319e-7f,4.55182175e-7f,3u,false,
                        "strict forcing accepted active RHS solely at outer tolerance");
            strictTrial(6.3580319e-7f,8.0e-9f,3u,true,
                        "strict forcing rejected resolved forcing target");
            strictTrial(3.39000259e-8f,3.39000259e-8f,3u,true,
                        "strict forcing reactivated already acceptable held environment");
            strictTrial(6.3580319e-7f,0.0f,3u,true,
                        "strict forcing rejected zero residual");
            std::memcpy([solverBuffer contents], &solver, sizeof(solver));
            std::memcpy([replaceBuffer contents], &replace, sizeof(replace));

            // Nonfinite action is reported and must not publish a partial
            // residual replacement.
            auto nanOperator = operatorValues;
            nanOperator[0].x = NAN;
            std::memcpy([operatorBuffer contents], nanOperator.data(),
                nanOperator.size() * sizeof(nm_float4));
            auto freshState = initialState;
            freshState.diagnostics.z = 0.0f;
            std::memcpy([stateBuffer contents], &freshState, sizeof(freshState));
            std::memcpy([residualBuffer contents], residualValues.data(),
                residualValues.size() * sizeof(nm_float4));
            resetStatuses(diagnosticBuffer, statusBuffer, healthy, healthy);
            dispatch(device, queue, measure, {
                {0u, dispatchBuffer}, {30u, layoutBuffer},
                {1u, solverBuffer}, {2u, rhsBuffer},
                {3u, operatorBuffer}, {4u, residualBuffer},
                {5u, stateBuffer}, {6u, metricsBuffer},
                {7u, makeBuffer(device, std::vector<float>{forcing})},
                {8u, diagnosticBuffer}, {9u, replaceBuffer},
                {10u, statusBuffer}, {11u, propagateBuffer}},
                1u, 32u);
            auto failedDiagnostic =
                readBuffer<NMMatterStatusGPU>(diagnosticBuffer, 1u);
            auto failedProduction =
                readBuffer<NMMatterStatusGPU>(statusBuffer, 1u);
            auto untouched = readBuffer<nm_float4>(residualBuffer, 2u);
            require(failedDiagnostic[0].code == NM_STATUS_NONFINITE_RESULT &&
                    failedProduction[0].code == NM_STATUS_NONFINITE_RESULT,
                "nonfinite action did not propagate to diagnostic and production status");
            require(untouched[0].x == -7.0f && untouched[1].x == -7.0f,
                "nonfinite action partially replaced the restart residual");

            // Early unhealthy diagnostic status publishes once to a healthy
            // production status, but never overwrites an existing failure.
            NMMatterStatusGPU diagnosticFailure = healthy;
            diagnosticFailure.code = NM_STATUS_CONTACT_FAILURE;
            diagnosticFailure.failingIndex = 9u;
            resetStatuses(diagnosticBuffer, statusBuffer, diagnosticFailure, healthy);
            dispatch(device, queue, measure, {
                {0u, dispatchBuffer}, {30u, layoutBuffer},
                {1u, solverBuffer}, {2u, rhsBuffer},
                {3u, operatorBuffer}, {4u, residualBuffer},
                {5u, stateBuffer}, {6u, metricsBuffer},
                {7u, makeBuffer(device, std::vector<float>{forcing})},
                {8u, diagnosticBuffer}, {9u, replaceBuffer},
                {10u, statusBuffer}, {11u, propagateBuffer}},
                1u, 32u);
            auto propagated =
                readBuffer<NMMatterStatusGPU>(statusBuffer, 1u);
            require(propagated[0].code == NM_STATUS_CONTACT_FAILURE &&
                    propagated[0].failingIndex == 9u,
                "unhealthy diagnostic status was not propagated to healthy production status");

            NMMatterStatusGPU existingFailure = healthy;
            existingFailure.code = NM_STATUS_INVALID_DISPATCH;
            existingFailure.failingIndex = 4u;
            resetStatuses(diagnosticBuffer, statusBuffer, diagnosticFailure,
                          existingFailure);
            dispatch(device, queue, measure, {
                {0u, dispatchBuffer}, {30u, layoutBuffer},
                {1u, solverBuffer}, {2u, rhsBuffer},
                {3u, operatorBuffer}, {4u, residualBuffer},
                {5u, stateBuffer}, {6u, metricsBuffer},
                {7u, makeBuffer(device, std::vector<float>{forcing})},
                {8u, diagnosticBuffer}, {9u, replaceBuffer},
                {10u, statusBuffer}, {11u, propagateBuffer}},
                1u, 32u);
            auto preserved =
                readBuffer<NMMatterStatusGPU>(statusBuffer, 1u);
            require(preserved[0].code == NM_STATUS_INVALID_DISPATCH &&
                    preserved[0].failingIndex == 4u,
                "diagnostic status overwrote an existing production failure");

            std::cout << "PASS native FGMRES true-residual kernel contract "
                      << checks
                      << " checks; correction_accumulated=true; unit_action=true; "
                         "full_FEM_operator_qualification=false\n";
            return 0;
        } catch (const std::exception& exception) {
            std::cerr << "FAIL native FGMRES true-residual kernel contract: "
                      << exception.what() << '\n';
            return 1;
        }
    }
}

