#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/shared.h"

#include <cmath>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
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

// The production runtime explicitly specializes FC34 to false unless the
// relative-inner-forcing experiment is enabled. Keep this native checker on
// that same legacy default so kernels that inline the FC34 helper are buildable.
id<MTLFunction> functionWithRelativeInnerForcingDisabled(
    id<MTLDevice> device, id<MTLLibrary> library, const char* name) {
    NSString* functionName = [NSString stringWithUTF8String:name];
    MTLFunctionConstantValues* values =
        [[MTLFunctionConstantValues alloc] init];
    const BOOL enabled = NO;
    [values setConstantValue:&enabled type:MTLDataTypeBool atIndex:34u];
    NSError* error = nil;
    id<MTLFunction> result = [library newFunctionWithName:functionName
        constantValues:values error:&error];
    require(result != nil, error != nil
        ? std::string("cannot specialize Metal function: ") +
            [[error localizedDescription] UTF8String]
        : std::string("cannot specialize Metal function: ") + name);
    (void)device;
    return result;
}

id<MTLComputePipelineState> pipelineWithRelativeInnerForcingDisabled(
    id<MTLDevice> device, id<MTLLibrary> library, const char* name) {
    id<MTLFunction> function =
        functionWithRelativeInnerForcingDisabled(device, library, name);
    NSError* error = nil;
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:&error];
    require(result != nil, error != nil
        ? std::string("pipeline creation failed: ") +
            [[error localizedDescription] UTF8String]
        : std::string("pipeline creation failed: ") + name);
    return result;
}

id<MTLComputePipelineState> indirectPipelineWithRelativeInnerForcingDisabled(
    id<MTLDevice> device, id<MTLLibrary> library, const char* name) {
    MTLComputePipelineDescriptor* descriptor =
        [MTLComputePipelineDescriptor new];
    descriptor.computeFunction =
        functionWithRelativeInnerForcingDisabled(device, library, name);
    descriptor.supportIndirectCommandBuffers = YES;
    NSError* error = nil;
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithDescriptor:descriptor
            options:MTLPipelineOptionNone reflection:nil error:&error];
    require(result != nil, error != nil
        ? std::string("indirect pipeline creation failed: ") +
            [[error localizedDescription] UTF8String]
        : std::string("indirect pipeline creation failed: ") + name);
    return result;
}

id<MTLComputePipelineState> indirectPipeline(
    id<MTLDevice> device, id<MTLLibrary> library, const char* name) {
    NSString* functionName = [NSString stringWithUTF8String:name];
    id<MTLFunction> function = [library newFunctionWithName:functionName];
    require(function != nil, std::string("missing Metal function: ") + name);
    MTLComputePipelineDescriptor* descriptor =
        [MTLComputePipelineDescriptor new];
    descriptor.computeFunction = function;
    descriptor.supportIndirectCommandBuffers = YES;
    NSError* error = nil;
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithDescriptor:descriptor
            options:MTLPipelineOptionNone reflection:nil error:&error];
    require(result != nil, error != nil
        ? std::string("indirect pipeline creation failed: ") +
            [[error localizedDescription] UTF8String]
        : std::string("indirect pipeline creation failed: ") + name);
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


void checkEmptyFieldPreconditioner(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> legacyPrecondition,
    id<MTLComputePipelineState> emptyFieldPrecondition,
    id<MTLComputePipelineState> fieldInitialize,
    id<MTLComputePipelineState> fieldExport,
    id<MTLComputePipelineState> crossPrecondition,
    id<MTLComputePipelineState> accumulateMechanical,
    id<MTLComputePipelineState> accumulateFields
) {
    constexpr std::uint32_t environments = 2u;
    constexpr std::uint32_t nodesPerEnvironment = 4u;
    constexpr std::uint32_t totalNodes = environments * nodesPerEnvironment;
    NMMatterDispatchGPU dispatchData{};
    dispatchData.abiVersion = NM_MATTER_ABI_VERSION;
    dispatchData.environmentCount = environments;
    dispatchData.femNodeCount = nodesPerEnvironment;
    dispatchData.objectCount = 1u;
    dispatchData.mixedMaterialCount = 1u;

    NMFGMRESLayoutGPU layout{};
    layout.unknownCount = 2u * totalNodes;

    NMMixedSolverGPU solver{};
    solver.nonlinearIterations.y = 1u;
    std::vector<NMFGMRESStateGPU> states(environments);
    for (auto& state : states)
        state.diagnostics = {1.0f, 1.0f, 0.0f, 1.0f};
    // A converged environment retains the old output tail exactly: both
    // legacy export and the replacement kernel leave its field rows untouched.
    states[1].diagnostics.z = 1.0f;
    std::vector<NMIncidenceRangeGPU> ranges(totalNodes);
    for (auto& range : ranges) {
        range.objectIndex = 0u;
        range.count = 0u;
    }
    NMContinuumObjectGPU object{};
    object.materialIndex = 0u;
    object.flags = 0u; // no multiphysics cross block
    std::vector<nm_float4> inverse(totalNodes);
    std::vector<nm_float4> basis(2u * totalNodes);
    for (std::uint32_t i = 0u; i < totalNodes; ++i) {
        inverse[i] = {1.25f + 0.125f * float(i),
                      0.75f + 0.0625f * float(i),
                      1.5f + 0.03125f * float(i),
                      0.5f + 0.25f * float(i)};
        basis[i] = {0.25f + float(i), -0.5f * float(i),
                    0.125f * float(i), 1.0f + 0.5f * float(i)};
        basis[totalNodes + i] = {2.0f + float(i), 1.0f, -3.0f, 4.0f};
    }
    constexpr nm_float4 poison{91.0f, -92.0f, 93.0f, -94.0f};
    std::vector<nm_float4> initial(2u * totalNodes, poison);
    auto dispatchBuffer = makeBuffer(device,
        std::vector<NMMatterDispatchGPU>{dispatchData});
    auto layoutBuffer = makeBuffer(device, std::vector<NMFGMRESLayoutGPU>{layout});
    auto solverBuffer = makeBuffer(device, std::vector<NMMixedSolverGPU>{solver});
    auto rangesBuffer = makeBuffer(device, ranges);
    auto inverseBuffer = makeBuffer(device, inverse);
    auto basisBuffer = makeBuffer(device, basis);
    auto stateBuffer = makeBuffer(device, states);
    auto objectsBuffer = makeBuffer(device, std::vector<NMContinuumObjectGPU>{object});
    auto materialBuffer = makeBuffer(device, std::vector<NMMixedMaterialGPU>(1u));
    auto legacyOutput = makeBuffer(device, initial);
    auto emptyOutput = makeBuffer(device, initial);

    auto precondition = [&](id<MTLComputePipelineState> kernel,
                            id<MTLBuffer> output) {
        dispatch(device, queue, kernel, {
            {0u, dispatchBuffer}, {1u, rangesBuffer}, {2u, inverseBuffer},
            {3u, basisBuffer}, {4u, output}, {5u, stateBuffer},
            {6u, solverBuffer}, {7u, objectsBuffer}}, 1u, 32u);
    };
    precondition(legacyPrecondition, legacyOutput);
    precondition(emptyFieldPrecondition, emptyOutput);

    auto fieldSolution = makeBuffer(device,
        std::vector<nm_float4>(totalNodes, poison));
    auto fieldResidual = makeBuffer(device,
        std::vector<nm_float4>(totalNodes, poison));
    auto fieldPreconditioned = makeBuffer(device,
        std::vector<nm_float4>(totalNodes, poison));
    auto fieldDirection = makeBuffer(device,
        std::vector<nm_float4>(totalNodes, poison));
    dispatch(device, queue, fieldInitialize, {
        {0u, dispatchBuffer}, {1u, rangesBuffer}, {2u, basisBuffer},
        {3u, inverseBuffer}, {4u, fieldSolution}, {5u, fieldResidual},
        {6u, fieldPreconditioned}, {7u, fieldDirection}, {8u, stateBuffer}},
        1u, 32u);
    dispatch(device, queue, fieldExport, {
        {0u, dispatchBuffer}, {1u, rangesBuffer}, {2u, fieldSolution},
        {3u, legacyOutput}, {4u, stateBuffer}}, 1u, 32u);
    dispatch(device, queue, crossPrecondition, {
        {0u, dispatchBuffer}, {1u, rangesBuffer}, {2u, objectsBuffer},
        {3u, materialBuffer}, {4u, inverseBuffer}, {5u, legacyOutput},
        {6u, stateBuffer}}, 1u, 32u);

    const auto legacy = readBuffer<nm_float4>(legacyOutput, 2u * totalNodes);
    const auto optimized = readBuffer<nm_float4>(emptyOutput, 2u * totalNodes);
    require(std::memcmp(legacy.data(), optimized.data(),
                2u * totalNodes * sizeof(nm_float4)) == 0,
        "empty-field precondition path changed legacy vector values");
    for (std::uint32_t node = 0u; node < totalNodes; ++node) {
        const auto& mechanical = optimized[node];
        if (node < nodesPerEnvironment) {
            require(std::abs(mechanical.w -
                    inverse[node].w * basis[node].w) <= 1.0e-6f,
                "mechanical pressure lane was not preserved");
            const auto& field = optimized[totalNodes + node];
            require(field.x == 0.0f && field.y == 0.0f &&
                    field.z == 0.0f && field.w == 0.0f,
                "active empty-field row was not cleared");
        } else {
            const auto& field = optimized[totalNodes + node];
            require(std::memcmp(&field, &poison, sizeof(poison)) == 0,
                "converged environment field tail was unexpectedly overwritten");
        }
    }

    // Exercise the production accumulation tail with the legacy and optimized
    // preconditioned columns. Both mechanical and field solutions must agree.
    std::vector<float> coefficients(
        environments * (NM_MIXED_FGMRES_RESTART + 1u), 0.0f);
    coefficients[0u] = 0.75f;
    coefficients[NM_MIXED_FGMRES_RESTART + 1u] = -0.5f;
    auto coefficientsBuffer = makeBuffer(device, coefficients);
    auto restartBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto legacySolution = makeBuffer(device,
        std::vector<nm_float4>(2u * totalNodes, poison));
    auto optimizedSolution = makeBuffer(device,
        std::vector<nm_float4>(2u * totalNodes, poison));
    auto accumulate = [&](id<MTLComputePipelineState> kernel,
                          id<MTLBuffer> columns,
                          id<MTLBuffer> solution) {
        dispatch(device, queue, kernel, {
            {0u, dispatchBuffer}, {30u, layoutBuffer}, {1u, solverBuffer},
            {2u, rangesBuffer}, {3u, columns}, {4u, coefficientsBuffer},
            {5u, stateBuffer}, {6u, solution}, {7u, restartBuffer}},
            1u, 32u);
    };
    accumulate(accumulateMechanical, legacyOutput, legacySolution);
    accumulate(accumulateMechanical, emptyOutput, optimizedSolution);
    accumulate(accumulateFields, legacyOutput, legacySolution);
    accumulate(accumulateFields, emptyOutput, optimizedSolution);
    const auto legacyTail = readBuffer<nm_float4>(legacySolution, 2u * totalNodes);
    const auto optimizedTail = readBuffer<nm_float4>(optimizedSolution, 2u * totalNodes);
    require(std::memcmp(legacyTail.data(), optimizedTail.data(),
                2u * totalNodes * sizeof(nm_float4)) == 0,
        "empty-field precondition path changed the completed accumulation tail");
    std::cout << "empty_field_preconditioner=true poisoned_field=true "
                 "pressure_w_preserved=true converged_rows_preserved=true "
                 "accumulation_tail_parity=true result=PASS" << std::endl;
}

void checkOperatorICBRanges(
    id<MTLDevice> device, id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> prepareRange
) {
    constexpr std::uint32_t prefixCommands = 5u, tailCommands = 9u;
    auto dispatchCase = [&](std::uint32_t environmentCount, float converged,
                            std::uint32_t statusCode, bool expectedActive,
                            const char* label) {
        NMMatterDispatchGPU dispatchData{};
        dispatchData.abiVersion = NM_MATTER_ABI_VERSION;
        dispatchData.environmentCount = environmentCount;
        std::vector<NMFGMRESStateGPU> states(environmentCount);
        std::vector<NMMatterStatusGPU> statuses(environmentCount);
        for (auto& status : statuses) status.code = NM_STATUS_SUCCESS;
        states[0].diagnostics = {1.0f, 1.0f, converged, 0.0f};
        statuses[0].code = statusCode;
        const auto beforeState = states[0];
        const auto beforeStatus = statuses[0];
        const auto dispatchBuffer = makeBuffer(device,
            std::vector<NMMatterDispatchGPU>{dispatchData});
        const auto stateBuffer = makeBuffer(device, states);
        const auto statusBuffer = makeBuffer(device, statuses);
        const auto prefixBuffer =
            makeBuffer(device, std::vector<std::uint32_t>{prefixCommands});
        const auto tailBuffer =
            makeBuffer(device, std::vector<std::uint32_t>{tailCommands});
        const auto rangesBuffer = makeBuffer(device,
            std::vector<std::uint32_t>{77u, 77u, 77u, 77u});
        const std::vector<std::uint32_t> sourceWords{
            0x01234567u, 0x89abcdefu, prefixCommands, tailCommands};
        const auto sourceWordsBuffer = makeBuffer(device, sourceWords);
        const auto destinationWordsBuffer = makeBuffer(device,
            std::vector<std::uint32_t>(sourceWords.size(), 0u));
        const auto wordCountBuffer = makeBuffer(device,
            std::vector<std::uint32_t>{
                static_cast<std::uint32_t>(sourceWords.size())});
        dispatch(device, queue, prepareRange, {
            {0u, dispatchBuffer}, {1u, stateBuffer}, {2u, statusBuffer},
            {3u, prefixBuffer}, {4u, tailBuffer}, {5u, rangesBuffer},
            {6u, sourceWordsBuffer}, {7u, destinationWordsBuffer},
            {8u, wordCountBuffer}}, 1u, 1u);
        const auto ranges = readBuffer<std::uint32_t>(rangesBuffer, 4u);
        require(readBuffer<std::uint32_t>(destinationWordsBuffer,
                    sourceWords.size()) == sourceWords,
            std::string("operator ICB constant refresh failed for ") + label);
        const std::uint32_t expectedPrefix = expectedActive ? prefixCommands : 0u;
        const std::uint32_t expectedTail = expectedActive ? tailCommands : 0u;
        require(ranges[0] == 0u && ranges[1] == expectedPrefix &&
                ranges[2] == 0u && ranges[3] == expectedTail,
            std::string("operator ICB range selection failed for ") + label);
        const auto stateAfter = readBuffer<NMFGMRESStateGPU>(stateBuffer, 1u)[0];
        const auto statusAfter = readBuffer<NMMatterStatusGPU>(statusBuffer, 1u)[0];
        require(std::memcmp(&beforeState, &stateAfter, sizeof(beforeState)) == 0 &&
                std::memcmp(&beforeStatus, &statusAfter, sizeof(beforeStatus)) == 0,
            std::string("operator ICB range selector mutated certification state for ") + label);
    };
    dispatchCase(1u, 0.0f, NM_STATUS_SUCCESS, true, "healthy active");
    dispatchCase(1u, 1.0f, NM_STATUS_SUCCESS, false, "converged");
    dispatchCase(1u, 0.0f, NM_STATUS_INVALID_DISPATCH, false, "unhealthy");
    dispatchCase(2u, 0.0f, NM_STATUS_SUCCESS, false, "multi-environment");
    std::cout << "operator_icb_gpu_ranges=true selected=5/9 "
                 "converged_unhealthy_multienv=0/0 certification_state_unchanged=true "
                 "result=PASS" << std::endl;
}

void checkTelemetryExhaustion(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> state
) {
    NMMatterDispatchGPU dispatchData{};
    dispatchData.abiVersion = NM_MATTER_ABI_VERSION;
    dispatchData.environmentCount = 1u;
    NMFGMRESStateGPU fgmres{};
    fgmres.diagnostics = {1.0f, 1.0f, 0.0f, 2.0f};
    NMMixedSolverGPU solver{};
    solver.nonlinearIterations.y = 2u;
    NMFGMRESTelemetryGPU telemetry{};
    NMMatterStatusGPU status{};
    const std::vector<NMMatterDispatchGPU> dispatchValues{dispatchData};
    const std::vector<NMFGMRESStateGPU> stateValues{fgmres};
    const std::vector<NMMixedSolverGPU> solverValues{solver};
    const std::vector<NMFGMRESTelemetryGPU> telemetryValues{telemetry};
    const std::vector<NMMatterStatusGPU> statusValues{status};
    id<MTLBuffer> dispatchBuffer = makeBuffer(device, dispatchValues);
    id<MTLBuffer> fgmresBuffer = makeBuffer(device, stateValues);
    id<MTLBuffer> solverBuffer = makeBuffer(device, solverValues);
    id<MTLBuffer> telemetryBuffer = makeBuffer(device, telemetryValues);
    id<MTLBuffer> statusBuffer = makeBuffer(device, statusValues);
    auto run = [&](const std::uint32_t iterationOffset,
                   const std::uint32_t totalBudget,
                   const std::uint32_t used,
                   const float converged,
                   const std::uint32_t statusCode) {
        auto* stateValue = static_cast<NMFGMRESStateGPU*>(
            [fgmresBuffer contents]);
        stateValue[0].diagnostics = {1.0f, 1.0f, converged,
                                     static_cast<float>(used)};
        auto* telemetryValue = static_cast<NMFGMRESTelemetryGPU*>(
            [telemetryBuffer contents]);
        telemetryValue[0] = NMFGMRESTelemetryGPU{};
        static_cast<NMMatterStatusGPU*>([statusBuffer contents])[0].code =
            statusCode;
        const std::vector<std::uint32_t> offsetValues{iterationOffset};
        const std::vector<std::uint32_t> budgetValues{totalBudget};
        id<MTLBuffer> offsetBuffer = makeBuffer(device, offsetValues);
        id<MTLBuffer> budgetBuffer = makeBuffer(device, budgetValues);
        dispatch(device, queue, state, {
            {0u, dispatchBuffer}, {1u, fgmresBuffer},
            {2u, telemetryBuffer}, {3u, solverBuffer},
            {4u, offsetBuffer}, {5u, budgetBuffer}, {6u, statusBuffer}},
            1u, 1u);
        return static_cast<NMFGMRESTelemetryGPU*>(
            [telemetryBuffer contents])[0].linear.w;
    };
    require(run(2u, 4u, 2u, 0.0f, 0u) == 1u,
        "terminal unconverged solve did not record total-budget exhaustion");
    require(run(0u, 4u, 2u, 0.0f, 0u) == 0u,
        "nonterminal restart was counted as total-budget exhaustion");
    require(run(2u, 4u, 2u, 1.0f, 0u) == 0u,
        "converged terminal solve was counted as budget exhaustion");
    require(run(2u, 4u, 0u, 0.0f, 0u) == 0u,
        "empty terminal restart was counted as budget exhaustion");
    require(run(2u, 4u, 2u, 0.0f, NM_STATUS_INVALID_DISPATCH) == 0u,
        "unhealthy terminal solve was counted as budget exhaustion");
    std::cout << "fgmres_budget_exhaustion=true total_budget=true "
                 "terminal_only=true converged_or_unhealthy_excluded=true result=PASS"
              << std::endl;
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

void checkTelemetryDomainPartitions(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> partialPipeline,
    id<MTLComputePipelineState> finishPipeline
) {
    constexpr std::uint32_t environments = 2u;
    constexpr std::uint32_t nodes = 300u;
    constexpr std::uint32_t mpmCapacity = 3u;
    constexpr std::uint32_t rigidCapacity = 8u;
    constexpr std::uint32_t supportCount = 5u;

    auto run = [&](const std::uint32_t vascularCount) {
        NMMatterDispatchGPU dispatchData{};
        dispatchData.abiVersion = NM_MATTER_ABI_VERSION;
        dispatchData.environmentCount = environments;
        dispatchData.femNodeCount = nodes;
        dispatchData.mpmActiveNodeCapacity = mpmCapacity;
        dispatchData.rigidGeneralizedCapacity = rigidCapacity;

        NMFGMRESLayoutGPU layout{};
        layout.reserved = 1u;
        layout.supportContactCount = supportCount;
        const std::uint32_t mpmBase = 2u * environments * nodes;
        const std::uint32_t rigidBase = mpmBase + environments * mpmCapacity;
        layout.supportBase = rigidBase + environments * rigidCapacity;
        layout.vascularUnknownCount = vascularCount;
        layout.vascularBase = layout.supportBase + environments * supportCount;
        layout.unknownCount = layout.vascularBase + environments * vascularCount;
        layout.vascularResidualTolerance =
            vascularCount == 0u ? 0.75f : 0.125f;

        const std::size_t physicalTransportBase =
            std::size_t(environments) * nodes;
        const std::size_t totalRows =
            std::size_t(layout.vascularBase) +
            std::size_t(environments) * vascularCount;
        require(totalRows > physicalTransportBase + std::size_t(environments) * nodes,
            "domain partition fixture does not retain the inactive transport span");

        std::vector<nm_float4> residual(totalRows, nm_float4{});
        std::array<double, environments> expectedMechanical{};
        std::array<double, environments> expectedVascular{};
        auto addSquares = [](double& sum, const nm_float4& value) {
            sum += double(value.x) * value.x;
            sum += double(value.y) * value.y;
            sum += double(value.z) * value.z;
            sum += double(value.w) * value.w;
        };
        auto valueFor = [](const std::uint32_t environment,
                           const std::uint32_t item,
                           const float scale) {
            const float e = float(environment + 1u);
            const float i = float(item % 37u);
            return nm_float4{
                scale * (0.2f + 0.013f * e + 0.001f * i),
                scale * (-0.04f + 0.0007f * i),
                scale * (0.03f - 0.0003f * i),
                scale * (0.01f + 0.0002f * i)};
        };

        for (std::uint32_t environment = 0u;
             environment < environments; ++environment) {
            for (std::uint32_t node = 0u; node < nodes; ++node) {
                const nm_float4 value = valueFor(environment, node, 1.0f);
                const std::size_t row = std::size_t(environment) * nodes + node;
                residual[row] = value;
                addSquares(expectedMechanical[environment], value);
            }
            for (std::uint32_t node = 0u; node < nodes; ++node) {
                const std::size_t row = physicalTransportBase +
                    std::size_t(environment) * nodes + node;
                residual[row] = nm_float4{
                    1.0e10f, -2.0e10f, 3.0e10f, -4.0e10f};
            }
            for (std::uint32_t item = 0u; item < mpmCapacity; ++item) {
                const nm_float4 value = valueFor(environment, item + 401u, 0.4f);
                const std::size_t row = std::size_t(mpmBase) +
                    std::size_t(environment) * mpmCapacity + item;
                residual[row] = value;
                addSquares(expectedMechanical[environment], value);
            }
            for (std::uint32_t dof = 0u; dof < rigidCapacity; ++dof) {
                const std::size_t row = std::size_t(rigidBase) +
                    std::size_t(environment) * rigidCapacity + dof;
                const nm_float4 value{
                    0.05f * float(environment + 1u) + 0.002f * float(dof),
                    8.0e9f, -9.0e9f, 1.0e10f};
                residual[row] = value;
                expectedMechanical[environment] += double(value.x) * value.x;
            }
            for (std::uint32_t item = 0u; item < supportCount; ++item) {
                const nm_float4 value = valueFor(environment, item + 701u, 0.2f);
                const std::size_t row = std::size_t(layout.supportBase) +
                    std::size_t(environment) * supportCount + item;
                residual[row] = value;
                addSquares(expectedMechanical[environment], value);
            }
            for (std::uint32_t item = 0u; item < vascularCount; ++item) {
                const float value = 0.3f * float(environment + 1u) +
                    0.01f * float(item);
                const std::size_t row = std::size_t(layout.vascularBase) +
                    std::size_t(environment) * vascularCount + item;
                residual[row] = nm_float4{value, 0.0f, 0.0f, 0.0f};
                expectedVascular[environment] += double(value) * value;
            }
        }

        NMMixedSolverGPU solver{};
        solver.residualTolerances.x = 0.01f;
        solver.regularization.x = 0.02f;
        std::vector<NMFGMRESStateGPU> states(environments);
        states[0].nonlinear = {2.0f, 1.5f, 0.0f, 0.0f};
        states[1].nonlinear = {4.0f, 3.0f, 0.0f, 0.0f};
        std::vector<NMFGMRESTelemetryGPU> telemetry(environments);
        const std::uint32_t itemCount = nodes + mpmCapacity + rigidCapacity +
            supportCount + vascularCount;
        const std::uint32_t tiles = (itemCount + 255u) / 256u;
        require(tiles > 1u,
            "domain partition fixture must exercise more than one 256-item tile");

        auto dispatchBuffer = makeBuffer(device,
            std::vector<NMMatterDispatchGPU>{dispatchData});
        auto layoutBuffer = makeBuffer(device,
            std::vector<NMFGMRESLayoutGPU>{layout});
        auto residualBuffer = makeBuffer(device, residual);
        auto partialBuffer = makeBuffer(device,
            std::vector<nm_float4>(std::size_t(environments) * tiles,
                nm_float4{-17.0f, -19.0f, -23.0f, -29.0f}));
        auto telemetryBuffer = makeBuffer(device, telemetry);
        auto stateBuffer = makeBuffer(device, states);
        auto solverBuffer = makeBuffer(device,
            std::vector<NMMixedSolverGPU>{solver});
        std::vector<NMMatterStatusGPU> statuses(environments);
        for (auto& status : statuses) status.code = NM_STATUS_SUCCESS;
        auto statusBuffer = makeBuffer(device, statuses);
        const std::vector<std::pair<NSUInteger, id<MTLBuffer>>> finishBindings{
            {0u, dispatchBuffer}, {30u, layoutBuffer},
            {1u, partialBuffer}, {2u, telemetryBuffer},
            {3u, stateBuffer}, {4u, solverBuffer}, {5u, statusBuffer}};

        dispatch(device, queue, partialPipeline, {
            {0u, dispatchBuffer}, {30u, layoutBuffer},
            {1u, residualBuffer}, {2u, partialBuffer}},
            std::size_t(environments) * tiles, 256u);
        dispatch(device, queue, finishPipeline, finishBindings,
            environments, 32u);

        const auto actual = readBuffer<NMFGMRESTelemetryGPU>(
            telemetryBuffer, environments);
        for (std::uint32_t environment = 0u;
             environment < environments; ++environment) {
            const float expectedNonvascular =
                static_cast<float>(std::sqrt(expectedMechanical[environment]));
            const float expectedVascularNorm =
                static_cast<float>(std::sqrt(expectedVascular[environment]));
            const float nonvascularTolerance =
                std::max(1.0e-5f, expectedNonvascular * 3.0e-5f);
            const float vascularNormTolerance =
                std::max(1.0e-5f, expectedVascularNorm * 3.0e-5f);
            require(close(actual[environment].domains.x, expectedNonvascular,
                          nonvascularTolerance),
                "domain telemetry nonvascular norm differs from exact FGMRES row mask");
            require(close(actual[environment].domains.y, expectedVascularNorm,
                          vascularNormTolerance),
                "domain telemetry vascular norm differs from vascular row mask");
            const float expectedThreshold = std::max(
                solver.regularization.x,
                solver.residualTolerances.x *
                    std::max(std::min(states[environment].nonlinear.x,
                                      states[environment].nonlinear.y), 1.0f));
            require(close(actual[environment].domains.z, expectedThreshold),
                "domain telemetry mechanical threshold mismatch");
            require(actual[environment].domains.w ==
                    layout.vascularResidualTolerance,
                "domain telemetry vascular threshold mismatch");
        }

        if (vascularCount != 0u) {
            auto* statusValues =
                static_cast<NMMatterStatusGPU*>([statusBuffer contents]);
            statusValues[1].code = NM_STATUS_NONLINEAR_SOLVER_FAILURE;
            auto* telemetryValues =
                static_cast<NMFGMRESTelemetryGPU*>([telemetryBuffer contents]);
            const nm_float4 sentinel{91.25f, -13.5f, 7.75f, 3.125f};
            telemetryValues[1].domains = sentinel;
            dispatch(device, queue, finishPipeline, finishBindings,
                environments, 32u);
            const auto guarded = readBuffer<NMFGMRESTelemetryGPU>(
                telemetryBuffer, environments);
            require(guarded[1].domains.x == sentinel.x &&
                    guarded[1].domains.y == sentinel.y &&
                    guarded[1].domains.z == sentinel.z &&
                    guarded[1].domains.w == sentinel.w,
                "failed environment overwrote previously published domain telemetry");
        }

        std::cout << "telemetry_domain_case vascular_rows=" << vascularCount
                  << " environments=" << environments
                  << " fem_nodes=" << nodes << " tiles_per_environment=" << tiles
                  << " inactive_transport_sentinels=true rigid_scalar_mask=true"
                  << " result=PASS" << '\n';
    };

    run(4u);
    run(0u);
}

void checkLargeNodeResidualMeasurement(
    id<MTLDevice> device, id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> measure,
    id<MTLComputePipelineState> distributedPartial,
    id<MTLComputePipelineState> distributedFinish,
    id<MTLComputePipelineState> distributedReplace,
    const std::uint32_t nodeCount) {
    const std::size_t unknownCount = 2u * std::size_t(nodeCount);
    NMMatterDispatchGPU dispatchData{};
    dispatchData.abiVersion = NM_MATTER_ABI_VERSION;
    dispatchData.environmentCount = 1u;
    dispatchData.objectCount = 1u;
    dispatchData.femNodeCount = nodeCount;
    NMFGMRESLayoutGPU layout{};
    layout.unknownCount = static_cast<std::uint32_t>(unknownCount);
    NMMixedSolverGPU solver{};
    solver.residualTolerances.x = 1.0e-8f;
    solver.regularization.x = 1.0e-12f;
    NMFGMRESStateGPU state{};
    state.diagnostics = {0.0f, 1.0f, 0.0f, 0.0f};
    state.nonlinear = {0.0f, 0.0f, 0.0f, 0.0f};
    NMMatterStatusGPU healthy{};
    healthy.code = NM_STATUS_SUCCESS;
    healthy.environment = 0u;
    std::vector<nm_float4> rhs(unknownCount, nm_float4{0.0f,0.0f,0.0f,0.0f});
    std::vector<nm_float4> action(unknownCount, nm_float4{0.0f,0.0f,0.0f,0.0f});
    std::vector<nm_float4> expectedResidual(unknownCount, nm_float4{0.0f,0.0f,0.0f,0.0f});
    double expectedNormSquared = 0.0;
    float expectedMaximum = 0.0f;
    std::uint32_t fixedMechanical = 0u, fixedField = 0u;
    for (std::uint32_t node = 0u; node < nodeCount; ++node) {
        const bool fixedMechanicalRow = node % 257u == 0u;
        const bool fixedFieldRow = node % 263u == 0u;
        fixedMechanical += fixedMechanicalRow ? 1u : 0u;
        fixedField += fixedFieldRow ? 1u : 0u;
        const nm_float4 mechanicalRHS{
            0.001f + float(node % 11u) * 0.00001f,
            -0.0003f - float(node % 5u) * 0.00001f,
            0.0002f + float(node % 7u) * 0.00001f, 0.0f};
        const nm_float4 fieldRHS{-0.0004f - float(node % 13u) * 0.00001f,0.0f,0.0f,0.0f};
        const std::size_t mechanical=node, field=std::size_t(nodeCount)+node;
        rhs[mechanical]=mechanicalRHS; rhs[field]=fieldRHS;
        action[mechanical]=fixedMechanicalRow?mechanicalRHS:
            nm_float4{0.25f*mechanicalRHS.x,0.25f*mechanicalRHS.y,0.25f*mechanicalRHS.z,0.0f};
        action[field]=fixedFieldRow?fieldRHS:nm_float4{-0.5f*fieldRHS.x,0.0f,0.0f,0.0f};
        expectedResidual[mechanical]={mechanicalRHS.x-action[mechanical].x,
            mechanicalRHS.y-action[mechanical].y,mechanicalRHS.z-action[mechanical].z,0.0f};
        expectedResidual[field]={fieldRHS.x-action[field].x,0.0f,0.0f,0.0f};
        for(const auto& r:{expectedResidual[mechanical],expectedResidual[field]}){
            expectedNormSquared+=double(r.x)*r.x+double(r.y)*r.y+double(r.z)*r.z+double(r.w)*r.w;
            expectedMaximum=std::max(expectedMaximum,std::max(std::max(std::abs(r.x),std::abs(r.y)),std::max(std::abs(r.z),std::abs(r.w))));
        }
    }
    require(fixedMechanical>0u&&fixedField>0u,"large-row fixture lacks constrained identity rows");
    auto dispatchBuffer=makeBuffer(device,std::vector<NMMatterDispatchGPU>{dispatchData});
    auto layoutBuffer=makeBuffer(device,std::vector<NMFGMRESLayoutGPU>{layout});
    auto solverBuffer=makeBuffer(device,std::vector<NMMixedSolverGPU>{solver});
    auto rhsBuffer=makeBuffer(device,rhs), actionBuffer=makeBuffer(device,action);
    auto residualBuffer=makeBuffer(device,std::vector<nm_float4>(unknownCount,nm_float4{-9.0f,-9.0f,-9.0f,-9.0f}));
    auto stateBuffer=makeBuffer(device,std::vector<NMFGMRESStateGPU>{state});
    auto metricsBuffer=makeBuffer(device,std::vector<nm_float4>{nm_float4{-1.0f,-1.0f,-1.0f,-1.0f}});
    auto diagnosticBuffer=makeBuffer(device,std::vector<NMMatterStatusGPU>{healthy});
    auto productionBuffer=makeBuffer(device,std::vector<NMMatterStatusGPU>{healthy});
    auto forcingBuffer=makeBuffer(device,std::vector<float>{1.0e-5f});
    auto replaceBuffer=makeBuffer(device,std::vector<std::uint32_t>{1u});
    auto propagateBuffer=makeBuffer(device,std::vector<std::uint32_t>{0u});
    dispatch(device,queue,measure,{{0u,dispatchBuffer},{30u,layoutBuffer},{1u,solverBuffer},
        {2u,rhsBuffer},{3u,actionBuffer},{4u,residualBuffer},{5u,stateBuffer},
        {6u,metricsBuffer},{7u,forcingBuffer},{8u,diagnosticBuffer},{9u,replaceBuffer},
        {10u,productionBuffer},{11u,propagateBuffer}},1u,32u);
    const auto actualResidual=readBuffer<nm_float4>(residualBuffer,unknownCount);
    const auto actualMetrics=readBuffer<nm_float4>(metricsBuffer,1u)[0];
    const auto actualState=readBuffer<NMFGMRESStateGPU>(stateBuffer,1u)[0];
    const auto diagnostic=readBuffer<NMMatterStatusGPU>(diagnosticBuffer,1u)[0];
    const auto production=readBuffer<NMMatterStatusGPU>(productionBuffer,1u)[0];
    const double expectedNorm=std::sqrt(expectedNormSquared);
    const double normTolerance=std::max(1.0e-7,expectedNorm*3.0e-5);
    require(diagnostic.code==NM_STATUS_SUCCESS&&production.code==NM_STATUS_SUCCESS,
        "large-row residual measurement changed healthy status");
    require(std::abs(double(actualMetrics.x)-expectedNorm)<=normTolerance,
        "large-row full mechanical+field residual norm mismatch");
    require(std::abs(actualMetrics.z-expectedMaximum)<=1.0e-7f,
        "large-row residual maximum mismatch");
    require(actualState.diagnostics.z==0.0f,
        "large-row residual unexpectedly passed its intentionally tighter gate");
    for(std::size_t row=0u;row<unknownCount;++row){
        const auto&a=actualResidual[row];const auto&e=expectedResidual[row];
        require(a.x==e.x&&a.y==e.y&&a.z==e.z&&a.w==e.w,
            "large-row same-operator residual replacement mismatch");
    }
    // Run the distributed opt-in path on the same assembled rows and compare
    // against the unchanged legacy kernel, including the exact replacement.
    const std::uint32_t tileCount =
        static_cast<std::uint32_t>((unknownCount + 255u) / 256u);
    auto partialBuffer = makeBuffer(device,
        std::vector<nm_float4>(tileCount, nm_float4{-7.0f,-7.0f,-7.0f,-7.0f}));
    std::memcpy([stateBuffer contents], &state, sizeof(state));
    std::memcpy([residualBuffer contents],
        std::vector<nm_float4>(unknownCount,
            nm_float4{-9.0f,-9.0f,-9.0f,-9.0f}).data(),
        unknownCount * sizeof(nm_float4));
    const nm_float4 sentinelMetrics{-1.0f,-1.0f,-1.0f,-1.0f};
    std::memcpy([metricsBuffer contents], &sentinelMetrics, sizeof(sentinelMetrics));
    resetStatuses(diagnosticBuffer, productionBuffer, healthy, healthy);
    const std::vector<std::pair<NSUInteger,id<MTLBuffer>>> distributedBindings{
        {0u,dispatchBuffer},{30u,layoutBuffer},{1u,solverBuffer},
        {2u,rhsBuffer},{3u,actionBuffer},{4u,residualBuffer},
        {5u,stateBuffer},{6u,metricsBuffer},{7u,forcingBuffer},
        {8u,diagnosticBuffer},{9u,replaceBuffer},{10u,productionBuffer},
        {11u,propagateBuffer},{24u,partialBuffer}};
    dispatch(device,queue,distributedPartial,distributedBindings,tileCount,256u);
    dispatch(device,queue,distributedFinish,distributedBindings,1u,32u);
    dispatch(device,queue,distributedReplace,distributedBindings,
        static_cast<NSUInteger>((nodeCount+255u)/256u),256u);
    const auto distributedResidual=readBuffer<nm_float4>(residualBuffer,unknownCount);
    const auto distributedMetrics=readBuffer<nm_float4>(metricsBuffer,1u)[0];
    const auto distributedState=readBuffer<NMFGMRESStateGPU>(stateBuffer,1u)[0];
    const auto distributedDiagnostic=readBuffer<NMMatterStatusGPU>(diagnosticBuffer,1u)[0];
    const auto distributedProduction=readBuffer<NMMatterStatusGPU>(productionBuffer,1u)[0];
    require(distributedDiagnostic.code==NM_STATUS_SUCCESS&&
            distributedProduction.code==NM_STATUS_SUCCESS,
        "distributed true-residual measurement changed healthy status");
    require(close(distributedMetrics.x,actualMetrics.x,
                  std::max(1.0e-7f,std::abs(actualMetrics.x)*3.0e-5f))&&
            close(distributedMetrics.z,actualMetrics.z,1.0e-7f)&&
            close(distributedMetrics.y,actualMetrics.y)&&
            close(distributedMetrics.w,actualMetrics.w),
        "distributed true-residual metrics differ from legacy control");
    require(distributedState.diagnostics.z==actualState.diagnostics.z&&
            close(distributedState.diagnostics.x,actualState.diagnostics.x)&&
            close(distributedState.diagnostics.y,actualState.diagnostics.y),
        "distributed true-residual convergence state differs from legacy control");
    for(std::size_t row=0u;row<unknownCount;++row){
        const auto&a=distributedResidual[row];const auto&e=actualResidual[row];
        require(a.x==e.x&&a.y==e.y&&a.z==e.z&&a.w==e.w,
            "distributed residual replacement differs from legacy control");
    }

    // Invalid free-rigid shape must reject before either path replaces the
    // residual and must propagate the same status to production.
    NMMatterDispatchGPU invalidDispatch=dispatchData;
    invalidDispatch.rigidGeneralizedCapacity=3u;
    std::memcpy([dispatchBuffer contents],&invalidDispatch,sizeof(invalidDispatch));
    std::vector<nm_float4> sentinelResidual(unknownCount,
        nm_float4{19.0f,20.0f,21.0f,22.0f});
    auto invalidPartials=makeBuffer(device,std::vector<nm_float4>(
        static_cast<std::size_t>((2u*nodeCount+3u+255u)/256u),
        nm_float4{-3.0f,-3.0f,-3.0f,-3.0f}));
    auto invalidPropagateBuffer=makeBuffer(device,std::vector<std::uint32_t>{1u});
    auto runInvalid=[&](const bool distributed){
        std::memcpy([stateBuffer contents],&state,sizeof(state));
        std::memcpy([residualBuffer contents],sentinelResidual.data(),
            sentinelResidual.size()*sizeof(nm_float4));
        resetStatuses(diagnosticBuffer,productionBuffer,healthy,healthy);
        if(distributed){
            const std::vector<std::pair<NSUInteger,id<MTLBuffer>>> bindings{
                {0u,dispatchBuffer},{30u,layoutBuffer},{1u,solverBuffer},
                {2u,rhsBuffer},{3u,actionBuffer},{4u,residualBuffer},
                {5u,stateBuffer},{6u,metricsBuffer},{7u,forcingBuffer},
                {8u,diagnosticBuffer},{9u,replaceBuffer},{10u,productionBuffer},
                {11u,invalidPropagateBuffer},{24u,invalidPartials}};
            dispatch(device,queue,distributedPartial,bindings,
                static_cast<NSUInteger>((2u*nodeCount+3u+255u)/256u),256u);
            dispatch(device,queue,distributedFinish,bindings,1u,32u);
            dispatch(device,queue,distributedReplace,bindings,
                static_cast<NSUInteger>((nodeCount+255u)/256u),256u);
        }else{
            dispatch(device,queue,measure,{
                {0u,dispatchBuffer},{30u,layoutBuffer},{1u,solverBuffer},
                {2u,rhsBuffer},{3u,actionBuffer},{4u,residualBuffer},
                {5u,stateBuffer},{6u,metricsBuffer},{7u,forcingBuffer},
                {8u,diagnosticBuffer},{9u,replaceBuffer},
                {10u,productionBuffer},{11u,invalidPropagateBuffer}},1u,32u);
        }
        const auto diagnosticStatus=readBuffer<NMMatterStatusGPU>(diagnosticBuffer,1u)[0];
        const auto productionStatus=readBuffer<NMMatterStatusGPU>(productionBuffer,1u)[0];
        const auto after=readBuffer<nm_float4>(residualBuffer,unknownCount);
        require(diagnosticStatus.code==NM_STATUS_INVALID_DISPATCH&&
                productionStatus.code==NM_STATUS_INVALID_DISPATCH,
            "invalid rigid-tail rejection/status propagation differs by path");
        require(std::memcmp(after.data(),sentinelResidual.data(),
                    sentinelResidual.size()*sizeof(nm_float4))==0,
            "invalid rigid-tail path partially replaced the residual");
    };
    runInvalid(false);
    runInvalid(true);

    std::cout << "true_residual_measurement_nodes=" << nodeCount
              << " unknown_rows=" << unknownCount
              << " constrained_mechanical_rows=" << fixedMechanical
              << " constrained_field_rows=" << fixedField
              << " distributed_vs_legacy=true invalid_rigid_tail_rejected=true"
              << " measurement_only=true full_FEM_operator_qualified=false"
              << " result=PASS" << '\n';
}

void runTailTrueResidualDistributedCase(id<MTLDevice> device,
    id<MTLCommandQueue> queue, id<MTLComputePipelineState> partialPipeline,
    id<MTLComputePipelineState> finishPipeline,
    id<MTLComputePipelineState> replacePipeline,
    const std::uint32_t q, const std::uint32_t reserved,
    const char* caseName) {
    constexpr std::uint32_t n=4u, rigid=9u, support=2u, vascular=3u;
    const std::uint32_t rows=2u*n+rigid+support+vascular;
    NMMatterDispatchGPU d{}; d.abiVersion=NM_MATTER_ABI_VERSION;
    d.environmentCount=1u; d.objectCount=1u; d.femNodeCount=n;
    d.rigidGeneralizedCapacity=rigid;
    NMFGMRESLayoutGPU l{}; l.unknownCount=rows;
    l.supportContactCount=support; l.supportBase=2u*n+rigid;
    l.vascularUnknownCount=vascular; l.vascularBase=l.supportBase+support;
    l.vascularResidualTolerance=1.0f; l.vascularReserved=q; l.reserved=reserved;
    NMMixedSolverGPU solver{}; solver.residualTolerances.x=0.25f;
    solver.regularization.x=1.0e-8f;
    NMFGMRESStateGPU initial{}; initial.diagnostics={0.125f,2.0f,1.0f,0.0f};
    initial.nonlinear={4.0f,0.0f,0.0f,0.0f};
    NMMatterStatusGPU healthy{}; healthy.code=NM_STATUS_SUCCESS;
    healthy.environment=0u;
    std::vector<nm_float4> rhs(rows,nm_float4{}), action(rows,nm_float4{});
    const nm_float4 sentinel{91.25f,-13.5f,7.75f,3.125f};
    std::vector<nm_float4> residual(rows,sentinel), expected(rows,sentinel);
    double norm2=0.0; float maximum=0.0f;
    auto setRow=[&](std::uint32_t row,nm_float4 b,nm_float4 ax,bool scalar){
        rhs[row]=b; action[row]=ax;
        nm_float4 r{b.x-ax.x,b.y-ax.y,b.z-ax.z,b.w-ax.w};
        if(scalar) r.y=r.z=r.w=0.0f;
        expected[row]=r;
        norm2+=double(r.x)*r.x+double(r.y)*r.y+
            double(r.z)*r.z+double(r.w)*r.w;
        maximum=std::max(maximum,std::max(std::max(std::abs(r.x),std::abs(r.y)),
            std::max(std::abs(r.z),std::abs(r.w))));
    };
    for(std::uint32_t i=0;i<n;++i){float f=float(i+1u);
        setRow(i,{.018f*f,-.011f*f,.007f*f,.004f*f},
            {.003f*f,.001f*f,-.002f*f,.001f*f},false);}
    for(std::uint32_t i=0;i<q;++i){float f=float(i+1u);
        setRow(n+i,{.09f*f,-.035f*f,.022f*f,.013f*f},
            {.012f*f,.005f*f,-.004f*f,.003f*f},false);}
    // Poison every unused row in the inactive field span.
    for(std::uint32_t i=n+q;i<2u*n;++i){
        rhs[i]={500.0f,-400.0f,300.0f,-200.0f};
        action[i]={-100.0f,80.0f,-60.0f,40.0f};}
    for(std::uint32_t i=0;i<rigid;++i){float f=float(i+1u);
        setRow(2u*n+i,{.021f*f,700.0f,-800.0f,900.0f},
            {.004f*f,-300.0f,200.0f,-100.0f},true);}
    for(std::uint32_t i=0;i<support;++i){float f=float(i+1u);
        setRow(l.supportBase+i,{.052f*f,-.027f*f,.019f*f,.008f*f},
            {.006f*f,.003f*f,-.002f*f,.001f*f},false);}
    for(std::uint32_t i=0;i<vascular;++i){float f=float(i+1u);
        setRow(l.vascularBase+i,{-.061f*f,.031f*f,.024f*f,-.017f*f},
            {-.008f*f,.004f*f,.005f*f,-.003f*f},false);}
    const float expectedNorm=float(std::sqrt(norm2));
    require(expectedNorm<1.0f,"tail fixture should pass unchanged completion gate");

    auto db=makeBuffer(device,std::vector<NMMatterDispatchGPU>{d});
    auto lb=makeBuffer(device,std::vector<NMFGMRESLayoutGPU>{l});
    auto sb=makeBuffer(device,std::vector<NMMixedSolverGPU>{solver});
    auto rb=makeBuffer(device,rhs), ab=makeBuffer(device,action);
    auto xb=makeBuffer(device,residual);
    auto states=makeBuffer(device,std::vector<NMFGMRESStateGPU>{initial});
    auto metrics=makeBuffer(device,std::vector<nm_float4>{{-1,-1,-1,-1}});
    auto diag=makeBuffer(device,std::vector<NMMatterStatusGPU>{healthy});
    auto prod=makeBuffer(device,std::vector<NMMatterStatusGPU>{healthy});
    auto forcing=makeBuffer(device,std::vector<float>{0.5f});
    auto replace=makeBuffer(device,std::vector<std::uint32_t>{1u});
    auto propagate=makeBuffer(device,std::vector<std::uint32_t>{0u});
    auto partials=makeBuffer(device,std::vector<nm_float4>{{-7,-7,-7,-7}});
    const std::vector<std::pair<NSUInteger,id<MTLBuffer>>> binds{
        {0u,db},{30u,lb},{1u,sb},{2u,rb},{3u,ab},{4u,xb},{5u,states},
        {6u,metrics},{7u,forcing},{8u,diag},{9u,replace},{10u,prod},
        {11u,propagate},{24u,partials}};
    dispatch(device,queue,partialPipeline,binds,1u,256u);
    dispatch(device,queue,finishPipeline,binds,1u,32u);
    dispatch(device,queue,replacePipeline,binds,1u,256u);
    const auto got=readBuffer<nm_float4>(xb,rows);
    const auto m=readBuffer<nm_float4>(metrics,1u)[0];
    const auto st=readBuffer<NMFGMRESStateGPU>(states,1u)[0];
    require(readBuffer<NMMatterStatusGPU>(diag,1u)[0].code==NM_STATUS_SUCCESS &&
            readBuffer<NMMatterStatusGPU>(prod,1u)[0].code==NM_STATUS_SUCCESS,
        "tail distributed residual changed healthy status");
    require(close(m.x,expectedNorm,std::max(1.0e-7f,expectedNorm*3.0e-5f)),
        "tail norm omitted/included an unexpected row or padding");
    require(close(m.z,maximum,1.0e-7f)&&close(m.y,initial.diagnostics.x)&&
            close(m.w,initial.diagnostics.y),"tail residual metrics mismatch");
    require(st.diagnostics.z==1.0f,
        "tail measurement lost the pre-existing completion latch");
    for(std::uint32_t i=0;i<rows;++i){
        const auto&a=got[i]; const auto&e=expected[i];
        require(a.x==e.x&&a.y==e.y&&a.z==e.z&&a.w==e.w,
            (i>=n+q && i<2u*n)
                ? "tail replacement changed ignored field padding"
                : "tail replacement changed an active vector row");
    }
    if (reserved==9u) {
        auto runInvalidLayout = [&](NMMatterDispatchGPU invalidDispatch,
                                    NMFGMRESLayoutGPU invalidLayout,
                                    const char* failureMessage) {
            const std::uint32_t propagateFailure = 1u;
            const nm_float4 metricSentinel{-1.0f,-1.0f,-1.0f,-1.0f};
            std::memcpy([db contents],&invalidDispatch,sizeof(invalidDispatch));
            std::memcpy([lb contents],&invalidLayout,sizeof(invalidLayout));
            std::memcpy([states contents],&initial,sizeof(initial));
            std::memcpy([diag contents],&healthy,sizeof(healthy));
            std::memcpy([prod contents],&healthy,sizeof(healthy));
            std::memcpy([propagate contents],&propagateFailure,sizeof(propagateFailure));
            std::memcpy([xb contents],residual.data(),residual.size()*sizeof(nm_float4));
            std::memcpy([metrics contents],&metricSentinel,sizeof(metricSentinel));
            dispatch(device,queue,partialPipeline,binds,1u,256u);
            dispatch(device,queue,finishPipeline,binds,1u,32u);
            dispatch(device,queue,replacePipeline,binds,1u,256u);
            require(readBuffer<NMMatterStatusGPU>(diag,1u)[0].code==NM_STATUS_INVALID_DISPATCH &&
                    readBuffer<NMMatterStatusGPU>(prod,1u)[0].code==NM_STATUS_INVALID_DISPATCH,
                failureMessage);
            const auto unchangedResidual=readBuffer<nm_float4>(xb,rows);
            const auto unchangedMetrics=readBuffer<nm_float4>(metrics,1u);
            require(std::memcmp(unchangedResidual.data(),residual.data(),
                        residual.size()*sizeof(nm_float4))==0 &&
                    std::memcmp(unchangedMetrics.data(),&metricSentinel,sizeof(metricSentinel))==0 &&
                    readBuffer<NMFGMRESStateGPU>(states,1u)[0].diagnostics.z==initial.diagnostics.z,
                "invalid full-tail layout partially replaced residual, metrics, or latch");
        };
        NMFGMRESLayoutGPU badOffset=l;
        badOffset.supportBase += 1u;
        runInvalidLayout(d,badOffset,
            "ordinary full-tail certificate accepted a malformed support offset");
        NMMatterDispatchGPU badCapacity=d;
        badCapacity.rigidGeneralizedCapacity -= 1u;
        runInvalidLayout(badCapacity,l,
            "ordinary full-tail certificate accepted a mismatched rigid capacity");
        NMFGMRESLayoutGPU badCount=l;
        badCount.unknownCount -= 1u;
        runInvalidLayout(d,badCount,
            "ordinary full-tail certificate accepted an undersized unknown arena");
    }
    std::cout<<caseName<<"_tail_true_residual=true rigid=9 q="<<q<<" support=2 "
                "vascular=3 inactive_field_padding="<<(n-q)<<" prelatched=true "
                "replacement=true result=PASS\\n";
}

void checkHybridTailTrueResidualDistributed(id<MTLDevice> device,
    id<MTLCommandQueue> queue, id<MTLComputePipelineState> partialPipeline,
    id<MTLComputePipelineState> finishPipeline,
    id<MTLComputePipelineState> replacePipeline) {
    runTailTrueResidualDistributedCase(device,queue,partialPipeline,finishPipeline,
        replacePipeline,2u,5u,"hybrid");
    runTailTrueResidualDistributedCase(device,queue,partialPipeline,finishPipeline,
        replacePipeline,0u,9u,"ordinary");
}

void checkHybridFullPhysicalResidualDiagnostic(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> preservePipeline,
    id<MTLComputePipelineState> partialPipeline,
    id<MTLComputePipelineState> finishPipeline) {
    NMMatterDispatchGPU dispatchValue{};
    dispatchValue.environmentCount = 1u;
    dispatchValue.femNodeCount = 4u;
    dispatchValue.rigidGeneralizedCapacity = 2u;
    NMFGMRESLayoutGPU layout{
        2u, 10u, 15u, 12u, 3u, 12u, 1.0e-5f, 1u};

    std::vector<nm_float4> rhs(layout.unknownCount, nm_float4{});
    std::vector<nm_float4> action(layout.unknownCount, nm_float4{});
    // The selected-row residual [1,-1] is orthogonal to the single q mode
    // [1,1], so q alone sees zero there. Tails remain physical unknowns.
    rhs[0] = {1.0f, 0.25f, 0.0f, 0.0f};
    rhs[1] = {-1.0f, 0.0f, 0.0f, 0.0f};
    action[0] = {0.0f, 0.25f, 0.0f, 0.0f};
    // The inactive field span is deliberately poisoned and must be excluded.
    for (std::uint32_t row = 4u; row < 8u; ++row) {
        rhs[row] = {1.0e6f, -2.0e6f, 3.0e6f, -4.0e6f};
        action[row] = {-1.0e6f, 2.0e6f, -3.0e6f, 4.0e6f};
    }
    rhs[8] = {0.5f, 100.0f, 200.0f, 300.0f};
    rhs[9] = {-0.5f, -100.0f, -200.0f, -300.0f};
    rhs[10] = {0.25f, 0.5f, 0.0f, 0.0f};
    rhs[12] = {0.5f, -0.25f, 0.0f, 0.0f};

    auto dispatchBuffer = makeBuffer(device,
        std::vector<NMMatterDispatchGPU>{dispatchValue});
    auto layoutBuffer = makeBuffer(device,
        std::vector<NMFGMRESLayoutGPU>{layout});
    auto rhsBuffer = makeBuffer(device, rhs);
    auto actionBuffer = makeBuffer(device, action);
    auto preserved = makeBuffer(device,
        std::vector<nm_float4>(layout.unknownCount, nm_float4{}));
    auto statuses = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
    auto partials = makeBuffer(device,
        std::vector<nm_float4>(1u, nm_float4{-7.0f,-7.0f,-7.0f,-7.0f}));
    auto metric = makeBuffer(device,
        std::vector<nm_float4>(1u, nm_float4{-1.0f,-1.0f,-1.0f,-1.0f}));

    dispatch(device, queue, preservePipeline, {
        {30u, layoutBuffer}, {1u, rhsBuffer}, {2u, preserved}},
        1u, 256u);
    const auto saved = readBuffer<nm_float4>(preserved, layout.unknownCount);
    require(std::memcmp(saved.data(), rhs.data(),
                rhs.size() * sizeof(nm_float4)) == 0,
        "hybrid diagnostic did not preserve the original unprojected RHS");

    dispatch(device, queue, partialPipeline, {
        {0u, dispatchBuffer}, {30u, layoutBuffer}, {1u, preserved},
        {2u, actionBuffer}, {3u, statuses}, {4u, partials}},
        1u, 256u);
    dispatch(device, queue, finishPipeline, {
        {0u, dispatchBuffer}, {30u, layoutBuffer}, {1u, partials},
        {2u, statuses}, {3u, metric}},
        1u, 32u);
    const auto got = readBuffer<nm_float4>(metric, 1u)[0];
    const float physicalNorm = std::sqrt(3.125f);
    const float physicalRHSNorm = std::sqrt(3.1875f);
    const float actionNorm = 0.25f;
    const float projectedNorm = std::sqrt(1.125f);
    require(close(got.x, physicalNorm, 2.0e-6f) &&
            close(got.y, physicalRHSNorm, 2.0e-6f) &&
            close(got.z, actionNorm, 2.0e-6f) && got.w == 1.0f,
        "full physical hybrid residual included padding or omitted active rows");
    const float qModeResidual = rhs[0].x + rhs[1].x;
    require(qModeResidual == 0.0f && projectedNorm < got.x,
        "adversarial residual did not expose the q-projection blind component");
    std::cout << "hybrid_full_physical_linear_residual=true projected_q_l2="
              << projectedNorm << " full_physical_l2=" << got.x
              << " acceptance_gate_changed=false result=PASS\n";
}

void checkDistributedBeginAndCorrection(
    id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library,
    id<MTLComputePipelineState> legacyBegin,
    id<MTLComputePipelineState> beginPartial,
    id<MTLComputePipelineState> beginFinish,
    id<MTLComputePipelineState> beginBasis0,
    id<MTLComputePipelineState> legacyCorrection,
    id<MTLComputePipelineState> correctionPartial,
    id<MTLComputePipelineState> correctionFinish,
    id<MTLComputePipelineState> trueResidualPartial,
    id<MTLComputePipelineState> trueResidualFinish) {
    constexpr std::uint32_t nodes=300u;
    id<MTLFunction> argumentLayout=[library newFunctionWithName:
        @"numi_matter_metal::nm_primal_contact_argument_layout"];
    require(argumentLayout!=nil,"primal-contact argument layout missing");
    id<MTLArgumentEncoder> contactEncoder=
        [argumentLayout newArgumentEncoderWithBufferIndex:0u];
    require(contactEncoder!=nil,"cannot create primal-contact argument encoder");
    auto zeroPointer=makeBuffer(device,std::vector<std::uint32_t>{0u});
    id<MTLBuffer> contactArguments=[device newBufferWithLength:
        contactEncoder.encodedLength options:MTLResourceStorageModeShared];
    require(contactArguments!=nil,"cannot allocate primal-contact arguments");
    [contactEncoder setArgumentBuffer:contactArguments offset:0u];
    for(NSUInteger index=0u;index<20u;++index)
        [contactEncoder setBuffer:zeroPointer offset:0u atIndex:index];

    NMMatterDispatchGPU dispatchData{};
    dispatchData.abiVersion=NM_MATTER_ABI_VERSION;
    dispatchData.environmentCount=1u;
    dispatchData.objectCount=1u;
    dispatchData.femNodeCount=nodes;
    dispatchData.tetrahedronCount=257u;
    NMFGMRESLayoutGPU layout{};
    layout.unknownCount=2u*nodes;
    layout.reserved=1u;
    NMMixedSolverGPU solver{};
    solver.residualTolerances.x=1.0e-6f;
    solver.regularization.x=1.0e-12f;
    NMFGMRESStateGPU initialState{};
    std::vector<nm_float4> residual(2u*nodes,
        nm_float4{1000.0f,1000.0f,1000.0f,1000.0f});
    for(std::uint32_t node=0u;node<nodes;++node)
        residual[node]=nm_float4{1.0f,0.0f,0.0f,0.0f};
    std::vector<NMFEMNodeStateGPU> candidate(nodes);
    NMContinuumObjectGPU object{};
    NMMicrostepGPU microstep{};
    NMSchedulerStateGPU scheduler{};
    NMAdaptiveStateGPU adaptive{};
    NMHumanSupportDispatchGPU support{};
    NMHumanSupportContactGPU supportContact{};
    NMMatterStatusGPU healthy{};
    healthy.code=NM_STATUS_SUCCESS;
    const std::uint32_t restartCycle=0u, nonlinearIteration=1u;
    const float forcing=0.01f;
    const std::size_t matrixCount=std::size_t(NM_MIXED_FGMRES_RESTART+1u)*
        NM_MIXED_FGMRES_RESTART;
    auto dispatchBuffer=makeBuffer(device,std::vector<NMMatterDispatchGPU>{dispatchData});
    auto layoutBuffer=makeBuffer(device,std::vector<NMFGMRESLayoutGPU>{layout});
    auto solverBuffer=makeBuffer(device,std::vector<NMMixedSolverGPU>{solver});
    auto objectsBuffer=makeBuffer(device,std::vector<NMContinuumObjectGPU>{object});
    auto residualBuffer=makeBuffer(device,residual);
    auto basisBuffer=makeBuffer(device,std::vector<nm_float4>(2u*nodes,
        nm_float4{-5.0f,-5.0f,-5.0f,-5.0f}));
    auto hessenbergBuffer=makeBuffer(device,std::vector<float>(matrixCount,7.0f));
    auto rotationsBuffer=makeBuffer(device,std::vector<NativeFloat2>(
        NM_MIXED_FGMRES_RESTART,NativeFloat2{8.0f,9.0f}));
    auto leastSquaresBuffer=makeBuffer(device,std::vector<float>(
        NM_MIXED_FGMRES_RESTART+1u,10.0f));
    auto stateBuffer=makeBuffer(device,std::vector<NMFGMRESStateGPU>{initialState});
    auto statusBuffer=makeBuffer(device,std::vector<NMMatterStatusGPU>{healthy});
    auto microstepBuffer=makeBuffer(device,std::vector<NMMicrostepGPU>{microstep});
    auto schedulerBuffer=makeBuffer(device,std::vector<NMSchedulerStateGPU>{scheduler});
    auto adaptiveBuffer=makeBuffer(device,std::vector<NMAdaptiveStateGPU>{adaptive});
    auto restartBuffer=makeBuffer(device,std::vector<std::uint32_t>{restartCycle});
    auto forcingBuffer=makeBuffer(device,std::vector<float>{forcing});
    auto iterationBuffer=makeBuffer(device,std::vector<std::uint32_t>{nonlinearIteration});
    auto candidateBuffer=makeBuffer(device,candidate);
    auto rigidCandidateBuffer=makeBuffer(device,std::vector<float>{0.0f});
    auto supportHistoryBuffer=makeBuffer(device,std::vector<nm_float4>{nm_float4{}});
    auto supportBuffer=makeBuffer(device,std::vector<NMHumanSupportDispatchGPU>{support});
    auto supportContactBuffer=makeBuffer(device,std::vector<NMHumanSupportContactGPU>{supportContact});
    auto nodeIndirectBuffer=makeBuffer(device,std::vector<NMIndirectDispatchGPU>(1u));
    auto elementIndirectBuffer=makeBuffer(device,std::vector<NMIndirectDispatchGPU>(1u));
    const std::vector<std::pair<NSUInteger,id<MTLBuffer>>> beginBindings{
        {0u,dispatchBuffer},{30u,layoutBuffer},{1u,solverBuffer},
        {2u,objectsBuffer},{3u,residualBuffer},{4u,basisBuffer},
        {5u,hessenbergBuffer},{6u,rotationsBuffer},{7u,leastSquaresBuffer},
        {8u,stateBuffer},{9u,statusBuffer},{10u,microstepBuffer},
        {11u,schedulerBuffer},{12u,adaptiveBuffer},{13u,contactArguments},
        {14u,restartBuffer},{15u,forcingBuffer},{16u,iterationBuffer},
        {17u,candidateBuffer},{18u,rigidCandidateBuffer},
        {19u,supportHistoryBuffer},{20u,supportBuffer},
        {21u,supportContactBuffer},{22u,nodeIndirectBuffer},
        {23u,elementIndirectBuffer}};
    dispatch(device,queue,legacyBegin,beginBindings,1u,32u);
    const auto legacyBeginState=readBuffer<NMFGMRESStateGPU>(stateBuffer,1u)[0];
    const auto legacyBasis=readBuffer<nm_float4>(basisBuffer,2u*nodes);
    std::memcpy([stateBuffer contents],&initialState,sizeof(initialState));
    std::vector<nm_float4> basisSentinel(2u*nodes,
        nm_float4{-5.0f,-5.0f,-5.0f,-5.0f});
    std::memcpy([basisBuffer contents],basisSentinel.data(),
        basisSentinel.size()*sizeof(nm_float4));
    const std::uint32_t beginTiles=(nodes+255u)/256u;
    auto beginPartials=makeBuffer(device,std::vector<nm_float4>(beginTiles));
    auto tiledBeginBindings=beginBindings;
    auto unusedDomainPartials=makeBuffer(device,
        std::vector<NativeFloat2>(beginTiles));
    tiledBeginBindings.emplace_back(24u,beginPartials);
    tiledBeginBindings.emplace_back(25u,unusedDomainPartials);
    dispatch(device,queue,beginPartial,tiledBeginBindings,beginTiles,256u);
    dispatch(device,queue,beginFinish,tiledBeginBindings,1u,32u);
    dispatch(device,queue,beginBasis0,tiledBeginBindings,beginTiles,256u);
    const auto distributedBeginState=readBuffer<NMFGMRESStateGPU>(stateBuffer,1u)[0];
    const auto distributedBasis=readBuffer<nm_float4>(basisBuffer,2u*nodes);
    require(close(distributedBeginState.diagnostics.x,
                  legacyBeginState.diagnostics.x,2.0e-5f)&&
            distributedBeginState.diagnostics.z==legacyBeginState.diagnostics.z,
        "distributed begin changed the inactive-transport residual gate");
    for(std::uint32_t node=0u;node<nodes;++node){
        require(close(distributedBasis[node].x,1.0f/std::sqrt(float(nodes)),2.0e-5f)&&
                close(distributedBasis[node].x,legacyBasis[node].x,2.0e-5f),
            "distributed begin mechanical basis differs from legacy control");
        const auto field=distributedBasis[nodes+node];
        require(field.x==0.0f&&field.y==0.0f&&field.z==0.0f&&field.w==0.0f,
            "distributed begin retained an inactive transport basis row");
    }

    // Predictor Arnoldi residual may be tiny while the assembled nonlinear
    // RHS remains large. Prove begin uses b-A*x0 for the Krylov basis but the
    // untouched physical RHS for the outer nonlinear latch and history.
    NMFGMRESLayoutGPU predictorLayout = layout;
    predictorLayout.reserved = 1u | (1u << 22u);
    std::memcpy([layoutBuffer contents], &predictorLayout,
        sizeof(predictorLayout));
    std::vector<nm_float4> warmResidual(2u * nodes, nm_float4{});
    std::vector<nm_float4> originalPhysicalResidual(2u * nodes,
        nm_float4{1.0e20f, -1.0e20f, 1.0e20f, -1.0e20f});
    for (std::uint32_t node = 0u; node < nodes; ++node) {
        warmResidual[node] = nm_float4{1.0e-8f, 0.0f, 0.0f, 0.0f};
        originalPhysicalResidual[node] = nm_float4{0.1f, 0.0f, 0.0f, 0.0f};
    }
    auto warmResidualBuffer = makeBuffer(device, warmResidual);
    auto originalPhysicalBuffer = makeBuffer(device, originalPhysicalResidual);
    auto predictorPartialBindings = beginBindings;
    for (auto& binding : predictorPartialBindings)
        if (binding.first == 3u) binding.second = warmResidualBuffer;
    predictorPartialBindings.emplace_back(24u, beginPartials);
    predictorPartialBindings.emplace_back(25u, unusedDomainPartials);
    predictorPartialBindings.emplace_back(26u, originalPhysicalBuffer);
    auto predictorFinishBindings = predictorPartialBindings;
    predictorFinishBindings.pop_back(); // b26 belongs only to the partial kernel.
    const float originalNorm = 0.1f * std::sqrt(float(nodes));
    NMFGMRESStateGPU predictorInitialState = initialState;
    predictorInitialState.nonlinear = {3.0f * originalNorm, 2.0f * originalNorm,
                                      0.0f, 0.0f};
    std::memcpy([stateBuffer contents], &predictorInitialState,
        sizeof(predictorInitialState));
    *static_cast<std::uint32_t*>([iterationBuffer contents]) = 2u;
    dispatch(device, queue, beginPartial, predictorPartialBindings,
        beginTiles, 256u);
    dispatch(device, queue, beginFinish, predictorFinishBindings, 1u, 32u);
    const auto predictorBeginState =
        readBuffer<NMFGMRESStateGPU>(stateBuffer, 1u)[0];
    const float warmNorm = 1.0e-8f * std::sqrt(float(nodes));
    require(close(predictorBeginState.diagnostics.x, warmNorm, 2.0e-5f),
        "predictor begin must use exact current residual for Arnoldi");
    require(close(predictorBeginState.diagnostics.y, originalNorm, 2.0e-5f),
        "relative forcing reference must retain the original nonlinear RHS norm");
    require(close(predictorBeginState.nonlinear.x, 3.0f * originalNorm, 2.0e-5f) &&
            close(predictorBeginState.nonlinear.y, originalNorm, 2.0e-5f),
        "predictor begin must retain original assembled nonlinear RHS norms");
    require(predictorBeginState.nonlinear.w < 0.5f,
        "small predictor residual must not falsely latch nonlinear convergence");
    const auto preservedPhysicalResidual =
        readBuffer<nm_float4>(originalPhysicalBuffer, 2u * nodes);
    require(close(preservedPhysicalResidual[0].x, 0.1f) &&
            close(preservedPhysicalResidual[nodes].x, 1.0e20f),
        "begin reduction modified the original nonlinear RHS buffer");

    // Schur Arnoldi uses the condensed RHS, but its stopping target must use
    // the original full-space b norm. The chosen true residual lies between
    // the two possible targets, so this checks the actual acceptance decision.
    auto runSchurOriginalNormCase = [&](const bool preserveFullNorm) {
        NMFGMRESLayoutGPU schurLayout{};
        schurLayout.unknownCount = 2u * nodes;
        schurLayout.supportBase = 2u * nodes;
        schurLayout.vascularBase = 2u * nodes;
        schurLayout.reserved = 1u | 8u |
            (preserveFullNorm ? (1u << 22u) : 0u);
        std::memcpy([layoutBuffer contents], &schurLayout,
            sizeof(schurLayout));
        *static_cast<float*>([forcingBuffer contents]) = 0.2f;
        std::vector<nm_float4> condensedRHS(2u * nodes, nm_float4{});
        std::vector<nm_float4> physicalRHS(2u * nodes, nm_float4{});
        const float fullPerRow = 10.0f / std::sqrt(float(nodes));
        const float condensedPerRow = 0.01f;
        for (std::uint32_t node = 0u; node < nodes; ++node) {
            condensedRHS[node] = nm_float4{condensedPerRow, 0.0f, 0.0f, 0.0f};
            physicalRHS[node] = nm_float4{fullPerRow, 0.0f, 0.0f, 0.0f};
        }
        auto condensedBuffer = makeBuffer(device, condensedRHS);
        auto physicalBuffer = makeBuffer(device, physicalRHS);
        auto partialBindings = beginBindings;
        for (auto& binding : partialBindings)
            if (binding.first == 3u) binding.second = condensedBuffer;
        partialBindings.emplace_back(24u, beginPartials);
        partialBindings.emplace_back(25u, unusedDomainPartials);
        partialBindings.emplace_back(26u, physicalBuffer);
        auto finishBindings = partialBindings;
        finishBindings.erase(std::remove_if(finishBindings.begin(),
            finishBindings.end(), [](const auto& binding) {
                return binding.first == 26u;
            }), finishBindings.end());
        std::memcpy([stateBuffer contents], &initialState, sizeof(initialState));
        *static_cast<std::uint32_t*>([iterationBuffer contents]) = 1u;
        dispatch(device, queue, beginPartial, partialBindings,
            beginTiles, 256u);
        dispatch(device, queue, beginFinish, finishBindings, 1u, 32u);
        const auto beginState =
            readBuffer<NMFGMRESStateGPU>(stateBuffer, 1u)[0];
        const float condensedNorm = condensedPerRow * std::sqrt(float(nodes));
        const float fullNorm = 10.0f;
        require(close(beginState.diagnostics.x, condensedNorm, 2.0e-5f),
            "Schur Arnoldi norm changed while retaining physical RHS scale");
        require(close(beginState.diagnostics.y,
                preserveFullNorm ? fullNorm : condensedNorm, 2.0e-5f),
            "Schur begin did not select the requested RHS normalization");

        const float truePerRow = 1.0f / std::sqrt(float(nodes));
        std::vector<nm_float4> operatorValue(2u * nodes, nm_float4{});
        for (std::uint32_t node = 0u; node < nodes; ++node)
            operatorValue[node] = nm_float4{
                physicalRHS[node].x - truePerRow, 0.0f, 0.0f, 0.0f};
        auto operatorBuffer = makeBuffer(device, operatorValue);
        auto residualOut = makeBuffer(device,
            std::vector<nm_float4>(2u * nodes, nm_float4{}));
        auto metricsBuffer = makeBuffer(device,
            std::vector<nm_float4>(1u, nm_float4{}));
        auto replaceBuffer = makeBuffer(device, std::vector<std::uint32_t>{3u});
        auto propagateBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
        auto truePartials = makeBuffer(device,
            std::vector<nm_float4>(beginTiles, nm_float4{}));
        const std::vector<std::pair<NSUInteger, id<MTLBuffer>>> trueBindings{
            {0u, dispatchBuffer}, {30u, layoutBuffer}, {1u, solverBuffer},
            {2u, physicalBuffer}, {3u, operatorBuffer}, {4u, residualOut},
            {5u, stateBuffer}, {6u, metricsBuffer}, {7u, forcingBuffer},
            {8u, statusBuffer}, {9u, replaceBuffer}, {10u, statusBuffer},
            {11u, propagateBuffer}, {24u, truePartials}};
        dispatch(device, queue, trueResidualPartial, trueBindings,
            beginTiles, 256u);
        dispatch(device, queue, trueResidualFinish, trueBindings, 1u, 32u);
        const auto resultState =
            readBuffer<NMFGMRESStateGPU>(stateBuffer, 1u)[0];
        const auto resultMetrics = readBuffer<nm_float4>(metricsBuffer, 1u)[0];
        require(close(resultMetrics.x, 1.0f, 2.0e-5f),
            "Schur certificate fixture did not produce its chosen full residual");
        require(close(resultMetrics.w,
                preserveFullNorm ? fullNorm : condensedNorm, 2.0e-5f),
            "Schur certificate used a different normalization than begin");
        return resultState.diagnostics.z;
    };
    require(runSchurOriginalNormCase(true) > 0.5f,
        "full-space Schur residual was rejected against the condensed RHS target");
    require(runSchurOriginalNormCase(false) < 0.5f,
        "Schur normalization fixture does not distinguish full and condensed RHS targets");
    std::cout << "schur_original_rhs_norm=true condensed_arnoldi_norm=true "
                 "full_residual_target=true result=PASS\n";

    *static_cast<float*>([forcingBuffer contents]) = forcing;
    *static_cast<std::uint32_t*>([iterationBuffer contents]) = nonlinearIteration;

    // The private domain latch keeps the full b24 norm untouched while
    // gating mechanical/pressure and vascular rows independently. A large
    // inactive transport sentinel must remain masked by reserved bit zero.
    std::vector<nm_float4> domainResidual(2u*nodes+2u,
        nm_float4{1.0e20f,1.0e20f,1.0e20f,1.0e20f});
    auto domainResidualBuffer=makeBuffer(device,domainResidual);
    auto domainPartialsBuffer=makeBuffer(device,
        std::vector<NativeFloat2>(beginTiles));
    auto domainBindings=beginBindings;
    for (auto& binding:domainBindings)
        if (binding.first==3u) binding.second=domainResidualBuffer;
    domainBindings.emplace_back(24u,beginPartials);
    domainBindings.emplace_back(25u,domainPartialsBuffer);
    auto runDomainLatch=[&](const float mechanicalNorm,
                            const float vascularNorm,
                            const bool pressureFailure,
                            const bool supportFailure,
                            const bool vascularNaN,
                            const std::uint32_t reserved) {
        const std::uint32_t supportCount=supportFailure?1u:0u;
        NMFGMRESLayoutGPU domainLayout{};
        domainLayout.supportContactCount=supportCount;
        domainLayout.supportBase=2u*nodes;
        domainLayout.unknownCount=2u*nodes+supportCount+1u;
        domainLayout.vascularUnknownCount=1u;
        domainLayout.vascularBase=2u*nodes+supportCount;
        domainLayout.vascularResidualTolerance=1.0e-5f;
        domainLayout.reserved=reserved;
        NMMixedSolverGPU domainSolver{};
        domainSolver.residualTolerances.x=1.0e-4f;
        domainSolver.regularization.x=1.0e-12f;
        NMMatterDispatchGPU domainDispatch=dispatchData;
        domainDispatch.abiVersion=NM_MATTER_ABI_VERSION;
        domainDispatch.environmentCount=1u;
        domainDispatch.femNodeCount=nodes;
        std::vector<nm_float4> rows(
            2u*nodes+supportCount+1u,nm_float4{});
        const float perNode=mechanicalNorm/std::sqrt(float(nodes));
        for (std::uint32_t node=0u;node<nodes;++node) {
            rows[node]=nm_float4{perNode,0.0f,0.0f,0.0f};
            rows[nodes+node]=nm_float4{1.0e20f,-1.0e20f,1.0e20f,-1.0e20f};
        }
        if (pressureFailure)
            rows[0].w=1.1e-4f;
        rows[domainLayout.vascularBase]=nm_float4{
            vascularNaN?std::numeric_limits<float>::quiet_NaN():vascularNorm,
            0.0f,0.0f,0.0f};
        std::memcpy([dispatchBuffer contents],&domainDispatch,sizeof(domainDispatch));
        std::memcpy([layoutBuffer contents],&domainLayout,sizeof(domainLayout));
        std::memcpy([solverBuffer contents],&domainSolver,sizeof(domainSolver));
        std::memcpy([domainResidualBuffer contents],rows.data(),
            rows.size()*sizeof(nm_float4));
        NMFGMRESStateGPU domainState{};
        domainState.nonlinear=nm_float4{1.0f,0.0f,0.0f,0.0f};
        std::memcpy([stateBuffer contents],&domainState,sizeof(domainState));
        const std::uint32_t domainIteration=2u;
        std::memcpy([iterationBuffer contents],&domainIteration,
            sizeof(domainIteration));
        NMHumanSupportDispatchGPU domainSupport{};
        domainSupport.contactCount=supportCount;
        domainSupport.groundNormal=nm_float4{0.0f,1.0f,0.0f,0.0f};
        std::memcpy([supportBuffer contents],&domainSupport,sizeof(domainSupport));
        NMHumanSupportContactGPU domainContact{};
        domainContact.frictionSlopAndStabilization.x=0.5f;
        std::memcpy([supportContactBuffer contents],&domainContact,
            sizeof(domainContact));
        const nm_float4 history=supportFailure
            ? nm_float4{0.0f,0.0f,0.0f,-1.0f} : nm_float4{};
        std::memcpy([supportHistoryBuffer contents],&history,sizeof(history));
        dispatch(device,queue,beginPartial,domainBindings,beginTiles,256u);
        dispatch(device,queue,beginFinish,domainBindings,1u,32u);
        const auto result=readBuffer<NMFGMRESStateGPU>(stateBuffer,1u)[0];
        return std::pair<float,float>{result.nonlinear.w,result.diagnostics.x};
    };
    const auto acceptedDomains=runDomainLatch(
        5.0e-5f,0.5e-5f,false,false,false,3u);
    require(acceptedDomains.first>0.5f,
        "domain latch rejected separately passing mechanical and vascular norms");
    require(acceptedDomains.second>1.0e-5f,
        "domain-latch fixture does not exceed the legacy vascular cap");
    require(close(acceptedDomains.second,
        std::sqrt(5.0e-5f*5.0e-5f+0.5e-5f*0.5e-5f),2.0e-8f),
        "domain latch changed the full FGMRES residual norm");
    require(runDomainLatch(5.0e-5f,0.5e-5f,false,false,false,1u).first<0.5f,
        "default legacy full-norm latch changed without the feature bit");
    require(runDomainLatch(1.1e-4f,0.5e-5f,false,false,false,3u).first<0.5f,
        "domain latch accepted a failed mechanical block");
    require(runDomainLatch(5.0e-5f,1.1e-5f,false,false,false,3u).first<0.5f,
        "domain latch accepted a failed vascular block");
    require(runDomainLatch(5.0e-5f,0.0f,false,false,true,3u).first<0.5f,
        "domain latch accepted a nonfinite vascular residual");
    require(runDomainLatch(5.0e-5f,0.5e-5f,true,false,false,3u).first<0.5f,
        "domain latch excluded mixed pressure from the mechanical block");
    require(runDomainLatch(5.0e-5f,0.5e-5f,false,true,false,3u).first<0.5f,
        "domain latch bypassed support-history admissibility");
    require(runDomainLatch(3.0e-5f,0.5e-5f,false,false,false,
        3u | (2u << 16u)).first<0.5f,
        "tightened domain gate accepted a mechanical norm above its bound");
    require(runDomainLatch(2.0e-5f,0.5e-5f,false,false,false,
        3u | (2u << 16u)).first>0.5f,
        "tightened domain gate rejected independently passing blocks");
    require(runDomainLatch(3.0e-5f,0.5e-5f,false,false,false,
        1u | (2u << 16u)).first<0.5f,
        "tightening metadata bypassed the legacy full-norm gate");
    // In Schur mode, buffer 3 is the condensed Arnoldi RHS while b26 is the
    // preserved physical RHS. The domain latch must use the latter as well;
    // otherwise a small condensed vector can falsely certify a larger full
    // mechanical residual.
    const std::uint32_t schurDomainTiles = (nodes + 1u + 255u) / 256u;
    NMFGMRESLayoutGPU schurDomainLayout{};
    schurDomainLayout.unknownCount = 2u * nodes + 1u;
    schurDomainLayout.supportBase = 2u * nodes;
    schurDomainLayout.vascularBase = 2u * nodes;
    schurDomainLayout.vascularUnknownCount = 1u;
    schurDomainLayout.vascularResidualTolerance = 1.0e-5f;
    schurDomainLayout.reserved = 1u | 2u | (1u << 22u);
    NMMixedSolverGPU schurDomainSolver{};
    schurDomainSolver.residualTolerances.x = 1.0e-4f;
    schurDomainSolver.regularization.x = 1.0e-12f;
    NMMatterDispatchGPU schurDomainDispatch = dispatchData;
    schurDomainDispatch.abiVersion = NM_MATTER_ABI_VERSION;
    schurDomainDispatch.environmentCount = 1u;
    schurDomainDispatch.femNodeCount = nodes;
    std::vector<nm_float4> schurDomainRows(
        2u * nodes + 1u, nm_float4{});
    std::vector<nm_float4> schurPhysicalRows(
        2u * nodes + 1u, nm_float4{});
    const auto runSchurDomainLatch = [&](const float physicalMechanicalNorm) {
        const float condensedMechanicalPerNode =
            4.0e-5f / std::sqrt(float(nodes));
        const float physicalMechanicalPerNode =
            physicalMechanicalNorm / std::sqrt(float(nodes));
        for (std::uint32_t node = 0u; node < nodes; ++node) {
            schurDomainRows[node] =
                nm_float4{condensedMechanicalPerNode, 0.0f, 0.0f, 0.0f};
            // Inactive transport rows are deliberately huge and remain masked.
            schurDomainRows[nodes + node] =
                nm_float4{1.0e20f, -1.0e20f, 1.0e20f, -1.0e20f};
            schurPhysicalRows[node] =
                nm_float4{physicalMechanicalPerNode, 0.0f, 0.0f, 0.0f};
            schurPhysicalRows[nodes + node] =
                nm_float4{1.0e20f, -1.0e20f, 1.0e20f, -1.0e20f};
        }
        schurDomainRows[2u * nodes] =
            nm_float4{0.5e-5f, 0.0f, 0.0f, 0.0f};
        schurPhysicalRows[2u * nodes] =
            nm_float4{0.5e-5f, 0.0f, 0.0f, 0.0f};
        std::memcpy([dispatchBuffer contents], &schurDomainDispatch,
            sizeof(schurDomainDispatch));
        std::memcpy([layoutBuffer contents], &schurDomainLayout,
            sizeof(schurDomainLayout));
        std::memcpy([solverBuffer contents], &schurDomainSolver,
            sizeof(schurDomainSolver));
        std::memcpy([domainResidualBuffer contents], schurDomainRows.data(),
            schurDomainRows.size() * sizeof(nm_float4));
        auto schurDomainBindings = beginBindings;
        for (auto& binding : schurDomainBindings)
            if (binding.first == 3u) binding.second = domainResidualBuffer;
        auto schurPhysicalBuffer = makeBuffer(device, schurPhysicalRows);
        schurDomainBindings.emplace_back(24u, beginPartials);
        schurDomainBindings.emplace_back(25u, domainPartialsBuffer);
        schurDomainBindings.emplace_back(26u, schurPhysicalBuffer);
        NMFGMRESStateGPU schurDomainState{};
        schurDomainState.nonlinear =
            nm_float4{1.0f, 0.0f, 0.0f, 0.0f};
        std::memcpy([stateBuffer contents], &schurDomainState,
            sizeof(schurDomainState));
        const std::uint32_t iteration = 2u;
        std::memcpy([iterationBuffer contents], &iteration, sizeof(iteration));
        dispatch(device, queue, beginPartial, schurDomainBindings,
            schurDomainTiles, 256u);
        auto schurFinishBindings = schurDomainBindings;
        schurFinishBindings.pop_back(); // b26 is only used by begin_partial.
        dispatch(device, queue, beginFinish, schurFinishBindings, 1u, 32u);
        return readBuffer<NMFGMRESStateGPU>(stateBuffer, 1u)[0];
    };
    const auto condensedPassFullFail = runSchurDomainLatch(1.5e-4f);
    require(condensedPassFullFail.nonlinear.w < 0.5f,
        "Schur domain latch accepted condensed-pass/full-mechanical-fail RHS");
    require(close(condensedPassFullFail.diagnostics.x,
        std::sqrt(4.0e-5f * 4.0e-5f + 0.5e-5f * 0.5e-5f), 2.0e-5f),
        "Schur domain fix changed the condensed Arnoldi norm");
    require(close(condensedPassFullFail.diagnostics.y,
        std::sqrt(1.5e-4f * 1.5e-4f + 0.5e-5f * 0.5e-5f), 2.0e-5f),
        "Schur domain latch did not retain full physical RHS norm");
    require(runSchurDomainLatch(8.0e-5f).nonlinear.w > 0.5f,
        "Schur domain latch rejected a full mechanical RHS below its threshold");
    std::cout<<"schur_domain_latch_uses_full_rhs=true "
                "condensed_arnoldi_norm_preserved=true result=PASS\n";
    std::cout<<"domain_nonlinear_latch=true separate_mechanical_vascular=true "
                "legacy_full_norm=true pressure_and_support_gates=true "
                "nonfinite_rejected=true result=PASS\n";

    dispatchData={};
    dispatchData.abiVersion=NM_MATTER_ABI_VERSION;
    dispatchData.environmentCount=1u;
    dispatchData.objectCount=1u;
    dispatchData.femNodeCount=nodes;
    dispatchData.mixedMaterialCount=1u;
    layout={};
    layout.unknownCount=2u*nodes;
    object={};
    object.flags=NM_OBJECT_MIXED_FEM;
    object.materialIndex=0u;
    NMMixedMaterialGPU material{};
    material.mechanics.x=4.0f;
    std::vector<NMFEMNodeStateGPU> correctionCandidate(nodes);
    std::vector<NMFEMFieldStateGPU> fields(nodes);
    std::vector<nm_float4> solution(2u*nodes,nm_float4{});
    std::vector<NMIncidenceRangeGPU> nodeRanges(nodes);
    for(std::uint32_t node=0u;node<nodes;++node){
        correctionCandidate[node].velocityAndInverseMass={2.0f,0.0f,0.0f,1.0f};
        fields[node].primary={2.0f,3.0f,4.0f,0.0f};
        solution[node]={4.0f,0.0f,0.0f,2.0f};
        solution[nodes+node]={3.0f,4.0f,0.0f,0.0f};
        nodeRanges[node].objectIndex=0u;
    }
    NMFGMRESStateGPU correctionState{};
    correctionState.nonlinear.z=-99.0f;
    auto correctionDispatchBuffer=makeBuffer(device,
        std::vector<NMMatterDispatchGPU>{dispatchData});
    auto correctionLayoutBuffer=makeBuffer(device,
        std::vector<NMFGMRESLayoutGPU>{layout});
    auto correctionObjectsBuffer=makeBuffer(device,
        std::vector<NMContinuumObjectGPU>{object});
    auto correctionCandidateBuffer=makeBuffer(device,correctionCandidate);
    auto solutionBuffer=makeBuffer(device,solution);
    auto correctionStateBuffer=makeBuffer(device,
        std::vector<NMFGMRESStateGPU>{correctionState});
    auto fieldsBuffer=makeBuffer(device,fields);
    auto gridBuffer=makeBuffer(device,std::vector<NMGridNodeStateGPU>(1u));
    auto activeIndicesBuffer=makeBuffer(device,std::vector<std::uint32_t>{0u});
    auto activeCountsBuffer=makeBuffer(device,std::vector<std::uint32_t>{0u});
    auto rangesBuffer=makeBuffer(device,nodeRanges);
    auto materialsBuffer=makeBuffer(device,std::vector<NMMixedMaterialGPU>{material});
    auto vascularCandidateBuffer=makeBuffer(device,std::vector<nm_float4>{nm_float4{}});
    const std::vector<std::pair<NSUInteger,id<MTLBuffer>>> correctionBindings{
        {0u,correctionDispatchBuffer},{30u,correctionLayoutBuffer},
        {1u,correctionObjectsBuffer},{2u,correctionCandidateBuffer},
        {3u,solutionBuffer},{4u,correctionStateBuffer},{5u,fieldsBuffer},
        {6u,contactArguments},{7u,gridBuffer},{8u,activeIndicesBuffer},
        {9u,activeCountsBuffer},{10u,rangesBuffer},{11u,materialsBuffer},
        {12u,vascularCandidateBuffer}};
    dispatch(device,queue,legacyCorrection,correctionBindings,1u,32u);
    const float legacyRatio=
        readBuffer<NMFGMRESStateGPU>(correctionStateBuffer,1u)[0].nonlinear.z;
    correctionState.nonlinear.z=-99.0f;
    std::memcpy([correctionStateBuffer contents],&correctionState,
        sizeof(correctionState));
    const std::uint32_t correctionTiles=(nodes+255u)/256u;
    auto correctionPartials=makeBuffer(device,
        std::vector<nm_float4>(correctionTiles));
    auto tiledCorrectionBindings=correctionBindings;
    tiledCorrectionBindings.emplace_back(24u,correctionPartials);
    dispatch(device,queue,correctionPartial,tiledCorrectionBindings,
        correctionTiles,256u);
    dispatch(device,queue,correctionFinish,tiledCorrectionBindings,1u,32u);
    const float distributedRatio=
        readBuffer<NMFGMRESStateGPU>(correctionStateBuffer,1u)[0].nonlinear.z;
    require(close(legacyRatio,4.0f)&&close(distributedRatio,legacyRatio),
        "distributed correction changed mixed pressure/field row ratios");
    std::cout<<"distributed_begin=true inactive_transport_mask=true "
                "distributed_correction=true mixed_pressure_field=true "
                "legacy_controls=true result=PASS\n";
}


void checkInexactNewtonForcing(
    id<MTLDevice> device, id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> prepare,
    id<MTLComputePipelineState> commit) {
    NMMatterDispatchGPU dispatchData{};
    dispatchData.abiVersion = NM_MATTER_ABI_VERSION;
    dispatchData.environmentCount = 1u;
    dispatchData.objectCount = 1u;
    dispatchData.femNodeCount = 1u;
    NMFGMRESLayoutGPU layout{};
    layout.unknownCount = 2u;
    layout.reserved = 1u;
    NMMixedSolverGPU solver{};
    solver.residualTolerances.x = 1.0e-8f;
    NMContinuumObjectGPU object{};
    object.representation = NM_REPRESENTATION_FEM;
    object.flags = NM_OBJECT_ACTIVE;
    NMSchedulerStateGPU scheduler{};
    scheduler.numerical.w = 1.0f;
    NMAdaptiveStateGPU adaptive{};
    adaptive.activeRepresentation = NM_REPRESENTATION_FEM;
    NMMicrostepGPU microstep{};
    NMMatterStatusGPU healthy{};
    healthy.code = NM_STATUS_SUCCESS;
    NMFGMRESStateGPU state{};
    state.nonlinear.w = 0.0f;

    auto dispatchBuffer = makeBuffer(device,
        std::vector<NMMatterDispatchGPU>{dispatchData});
    auto layoutBuffer = makeBuffer(device,
        std::vector<NMFGMRESLayoutGPU>{layout});
    auto solverBuffer = makeBuffer(device,
        std::vector<NMMixedSolverGPU>{solver});
    auto restartBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto iterationBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u});
    auto forcingBuffer = makeBuffer(device, std::vector<float>{0.1f});
    auto partialBuffer = makeBuffer(device,
        std::vector<nm_float4>{nm_float4{100.0f, 0.0f, 1.0f, 0.0f}});
    auto historyBuffer = makeBuffer(device,
        std::vector<nm_float4>{nm_float4{77.0f, 88.0f, 1.0f, 1.0f}});
    auto etaBuffer = makeBuffer(device, std::vector<float>{-1.0f});
    auto objectsBuffer = makeBuffer(device,
        std::vector<NMContinuumObjectGPU>{object});
    auto schedulersBuffer = makeBuffer(device,
        std::vector<NMSchedulerStateGPU>{scheduler});
    auto adaptiveBuffer = makeBuffer(device,
        std::vector<NMAdaptiveStateGPU>{adaptive});
    auto microstepBuffer = makeBuffer(device,
        std::vector<NMMicrostepGPU>{microstep});
    auto lineSearchBuffer = makeBuffer(device,
        std::vector<nm_float4>{nm_float4{1.0f, 0.0f, 0.0f, 0.0f}});
    auto environmentAlphaBuffer = makeBuffer(device,
        std::vector<float>{1.0f});
    auto statusBuffer = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{healthy});
    auto stateBuffer = makeBuffer(device,
        std::vector<NMFGMRESStateGPU>{state});

    const auto runPrepare = [&] {
        dispatch(device, queue, prepare, {
            {0u, dispatchBuffer}, {1u, solverBuffer},
            {14u, restartBuffer}, {15u, forcingBuffer},
            {16u, iterationBuffer}, {24u, partialBuffer},
            {25u, historyBuffer}, {26u, etaBuffer},
            {30u, layoutBuffer}}, 1u, 32u);
    };
    const auto runCommit = [&] {
        dispatch(device, queue, commit, {
            {0u, dispatchBuffer}, {1u, objectsBuffer},
            {2u, schedulersBuffer}, {3u, adaptiveBuffer},
            {4u, microstepBuffer}, {5u, lineSearchBuffer},
            {6u, environmentAlphaBuffer}, {7u, statusBuffer},
            {8u, stateBuffer}, {9u, historyBuffer}}, 1u, 32u);
    };
    const auto setIteration = [&](const std::uint32_t value) {
        *static_cast<std::uint32_t*>([iterationBuffer contents]) = value;
    };
    const auto setRestart = [&](const std::uint32_t value) {
        *static_cast<std::uint32_t*>([restartBuffer contents]) = value;
    };
    const auto setPartialNormSquared = [&](const float value) {
        static_cast<nm_float4*>([partialBuffer contents])[0].x = value;
    };
    const auto setPartialSupportAdmissible = [&](const float value) {
        static_cast<nm_float4*>([partialBuffer contents])[0].z = value;
    };
    const auto readHistory = [&] {
        return readBuffer<nm_float4>(historyBuffer, 1u)[0];
    };
    const auto readEta = [&] {
        return readBuffer<float>(etaBuffer, 1u)[0];
    };

    // A new root discards stale history and preserves its legacy schedule.
    setIteration(0u);
    setRestart(0u);
    runPrepare();
    auto h = readHistory();
    require(close(readEta(), 0.1f) && h.x == 0.0f && h.z == 0.0f &&
            h.w == 1.0f && close(h.y, 10.0f),
        "EW initial/root-reset preparation did not use legacy forcing");

    // Only a committed positive line-search candidate advances the base norm.
    runCommit();
    h = readHistory();
    require(close(h.x, 10.0f) && h.z == 1.0f && h.w == 0.0f,
        "accepted FEM candidate did not commit its staged base norm");

    setPartialNormSquared(25.0f);
    setIteration(1u);
    runPrepare();
    require(close(readEta(), 0.225f, 2.0e-6f),
        "EW ratio forcing did not use the consecutive full RHS norms");

    for (std::uint32_t exponent = 1u; exponent <= 4u; ++exponent) {
        auto* selectedLayout = static_cast<NMFGMRESLayoutGPU*>(layoutBuffer.contents);
        selectedLayout->reserved = 1u | (exponent << 19u);
        runPrepare();
        require(close(readEta(), std::ldexp(0.225f, -int(exponent)), 2.0e-6f),
            "conservative EW tightening did not scale accepted-history eta");
    }
    static_cast<NMFGMRESLayoutGPU*>(layoutBuffer.contents)->reserved = 1u;
    runPrepare();

    // Restart tails must reuse the prepared scalar and must not restage history.
    const nm_float4 beforeRestart = readHistory();
    const float etaBeforeRestart = readEta();
    setPartialNormSquared(400.0f);
    setRestart(1u);
    runPrepare();
    h = readHistory();
    require(close(readEta(), etaBeforeRestart) &&
            std::memcmp(&h, &beforeRestart, sizeof(h)) == 0,
        "restart-cycle preparation changed EW eta or accepted history");

    // A second accepted candidate commits the new base norm.
    setRestart(0u);
    runCommit();
    h = readHistory();
    require(close(h.x, 5.0f) && h.z == 1.0f && h.w == 0.0f,
        "second accepted FEM candidate did not advance EW history");

    // A base already complete is a no-op and does not advance the ratio.
    setPartialNormSquared(4.0f);
    setIteration(2u);
    runPrepare();
    *static_cast<NMFGMRESStateGPU*>([stateBuffer contents]) =
        NMFGMRESStateGPU{};
    static_cast<NMFGMRESStateGPU*>([stateBuffer contents])->nonlinear.w = 1.0f;
    runCommit();
    h = readHistory();
    require(close(h.x, 5.0f) && h.z == 1.0f && h.w == 0.0f,
        "already-complete nonlinear base advanced EW history");

    // Rejected line search invalidates the ratio so the next solve falls back.
    setPartialNormSquared(9.0f);
    setIteration(2u);
    *static_cast<NMFGMRESStateGPU*>([stateBuffer contents]) =
        NMFGMRESStateGPU{};
    runPrepare();
    static_cast<nm_float4*>([lineSearchBuffer contents])[0].x = 0.0f;
    runCommit();
    h = readHistory();
    require(h.z == 0.0f && h.w == 0.0f,
        "rejected line-search candidate retained EW history");

    // An unhealthy environment also invalidates the ratio.
    static_cast<nm_float4*>([lineSearchBuffer contents])[0].x = 1.0f;
    setPartialNormSquared(16.0f);
    setIteration(3u);
    runPrepare();
    auto badStatus = healthy;
    badStatus.code = NM_STATUS_CONTACT_FAILURE;
    *static_cast<NMMatterStatusGPU*>([statusBuffer contents]) = badStatus;
    runCommit();
    h = readHistory();
    require(h.z == 0.0f && h.w == 0.0f,
        "unhealthy environment retained EW history");
    *static_cast<NMMatterStatusGPU*>([statusBuffer contents]) = healthy;

    // Nonfinite reduction input cannot be hidden by sqrt(max(norm2, 0)).
    setPartialNormSquared(NAN);
    setIteration(3u);
    runPrepare();
    h = readHistory();
    require(close(readEta(), 0.1f) && h.z == 0.0f && h.w == 0.0f,
        "nonfinite full RHS partial did not invalidate EW history/fallback");

    // With no due FEM object there is no accepted material candidate to commit.
    setPartialNormSquared(9.0f);
    setIteration(0u);
    runPrepare();
    static_cast<NMSchedulerStateGPU*>([schedulersBuffer contents])[0]
        .numerical.w = 0.0f;
    runCommit();
    h = readHistory();
    require(h.x == 0.0f && h.z == 0.0f && h.w == 1.0f,
        "no-op environment without due FEM objects advanced EW history");

    // Support inadmissibility is the begin reduction's validity gate. Even
    // with a finite full RHS, it must invalidate old EW history and use the
    // legacy forcing for this solve.
    static_cast<NMSchedulerStateGPU*>([schedulersBuffer contents])[0]
        .numerical.w = 1.0f;
    setPartialSupportAdmissible(1.0f);
    setPartialNormSquared(9.0f);
    setIteration(0u);
    *static_cast<NMFGMRESStateGPU*>([stateBuffer contents]) =
        NMFGMRESStateGPU{};
    runPrepare();
    runCommit();
    h = readHistory();
    require(close(h.x, 3.0f) && h.z == 1.0f,
        "support-gate fixture did not establish accepted EW history");
    setPartialNormSquared(16.0f);
    setPartialSupportAdmissible(0.0f);
    setIteration(1u);
    runPrepare();
    h = readHistory();
    require(close(readEta(), 0.1f) && h.z == 0.0f && h.w == 0.0f,
        "inadmissible support partial retained EW history or skipped fallback");

    std::cout << "inexact_newton_forcing=true full_rhs=true accepted_history=true "
                 "restart_preserved=true reject_status_reset=true NaN_fallback=true "
                 "support_gate=true no_op_preserved=true result=PASS\n";
}

void checkDistributedCGS2(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> partial,
    id<MTLComputePipelineState> reduceRow,
    id<MTLComputePipelineState> update,
    id<MTLComputePipelineState> measureNorm,
    id<MTLComputePipelineState> finishColumn,
    id<MTLComputePipelineState> normalize) {
    // Two environments share every tiled launch. Environment zero starts
    // nearly parallel to q and stresses cancellation after the first CGS
    // update; environment one has a zero residual and must take the breakdown
    // path without contaminating its neighbor.
    NMMatterDispatchGPU dispatchData{};
    dispatchData.abiVersion = NM_MATTER_ABI_VERSION;
    dispatchData.environmentCount = 2u;
    dispatchData.objectCount = 1u;
    dispatchData.femNodeCount = 2u;
    NMFGMRESLayoutGPU layout{};
    layout.unknownCount = 8u;
    NMMixedSolverGPU solver{};
    solver.nonlinearIterations.y = 1u;
    solver.residualTolerances.x = 1.0e-6f;
    solver.regularization.x = 1.0e-12f;
    constexpr std::uint32_t column = 0u;
    constexpr std::uint32_t partialCount = 8u;
    constexpr std::uint32_t itemTileCount = 1u;

    std::vector<nm_float4> basis(
        std::size_t(NM_MIXED_FGMRES_RESTART + 1u) * layout.unknownCount,
        nm_float4{0.0f, 0.0f, 0.0f, 0.0f});
    const float scale = 1.0f + 5.0e-7f;
    const float q = (1.0f / std::sqrt(3.0f)) * scale;
    basis[0].x = q;
    basis[1].x = q;
    basis[4].x = q;
    basis[2].x = 1.0f;

    std::vector<nm_float4> work(8u,
        nm_float4{0.0f, 0.0f, 0.0f, 0.0f});
    const float alpha = 1.0e6f;
    work[0].x = alpha * q + 1.0f;
    work[1].x = alpha * q - 1.0f;
    work[4].x = alpha * q;

    std::vector<NMFGMRESStateGPU> states(2u);
    states[0].diagnostics = {0.0f, alpha, 0.0f, 0.0f};
    states[1].diagnostics = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> hessenberg(
        2u * std::size_t(NM_MIXED_FGMRES_RESTART + 1u) *
            NM_MIXED_FGMRES_RESTART, 0.0f);
    std::vector<NativeFloat2> rotations(
        2u * std::size_t(NM_MIXED_FGMRES_RESTART),
        NativeFloat2{1.0f, 0.0f});
    std::vector<float> leastSquares(
        2u * std::size_t(NM_MIXED_FGMRES_RESTART + 1u), 0.0f);
    leastSquares[0] = alpha;
    auto nodeActive = makeBuffer(device,
        std::vector<NMIndirectDispatchGPU>{NMIndirectDispatchGPU{}});
    auto elementActive = makeBuffer(device,
        std::vector<NMIndirectDispatchGPU>{NMIndirectDispatchGPU{}});
    auto dispatchBuffer = makeBuffer(device,
        std::vector<NMMatterDispatchGPU>{dispatchData});
    auto layoutBuffer = makeBuffer(device,
        std::vector<NMFGMRESLayoutGPU>{layout});
    auto solverBuffer = makeBuffer(device,
        std::vector<NMMixedSolverGPU>{solver});
    auto basisBuffer = makeBuffer(device, basis);
    auto workBuffer = makeBuffer(device, work);
    auto nextBasisBuffer = makeBuffer(device,
        std::vector<nm_float4>(8u, nm_float4{0.0f,0.0f,0.0f,0.0f}));
    auto partialBuffer = makeBuffer(device,
        std::vector<float>(2u * 2u * (NM_MIXED_FGMRES_RESTART + 1u) *
            partialCount, 0.0f));
    auto hessenbergBuffer = makeBuffer(device, hessenberg);
    auto rotationsBuffer = makeBuffer(device, rotations);
    auto leastSquaresBuffer = makeBuffer(device, leastSquares);
    auto stateBuffer = makeBuffer(device, states);
    auto columnBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{column});
    auto passBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u});
    auto forcingBuffer = makeBuffer(device, std::vector<float>{0.1f});

    auto runPartial = [&](std::uint32_t passIndex) {
        std::memcpy([passBuffer contents], &passIndex, sizeof(passIndex));
        dispatch(device, queue, partial, {
            {0u, dispatchBuffer}, {1u, columnBuffer}, {2u, passBuffer},
            {3u, basisBuffer}, {4u, workBuffer}, {5u, partialBuffer},
            {6u, stateBuffer}, {30u, layoutBuffer}},
            2u * itemTileCount, 256u);
    };
    auto runReduceAndUpdate = [&](std::uint32_t passIndex) {
        std::memcpy([passBuffer contents], &passIndex, sizeof(passIndex));
        dispatch(device, queue, reduceRow, {
            {0u, dispatchBuffer}, {30u, layoutBuffer}, {1u, columnBuffer},
            {2u, passBuffer}, {3u, hessenbergBuffer},
            {4u, partialBuffer}, {5u, stateBuffer}},
            2u, 32u);
        dispatch(device, queue, update, {
            {0u, dispatchBuffer}, {30u, layoutBuffer}, {1u, columnBuffer},
            {2u, passBuffer}, {3u, basisBuffer}, {4u, workBuffer},
            {5u, hessenbergBuffer}, {6u, partialBuffer}, {7u, stateBuffer}},
            2u * itemTileCount, 256u);
    };

    auto scalarNorm = [](const std::vector<nm_float4>& values,
                         const std::size_t mechanicalBase,
                         const std::size_t fieldBase) {
        double sum = 0.0;
        for (std::size_t node = 0u; node < 2u; ++node) {
            const auto& mechanical = values[mechanicalBase + node];
            const auto& field = values[fieldBase + node];
            sum += double(mechanical.x) * mechanical.x;
            sum += double(field.x) * field.x;
        }
        return std::sqrt(sum);
    };

    const double initialNorm = scalarNorm(work, 0u, 4u);
    runPartial(0u);
    runReduceAndUpdate(0u);
    const auto afterFirstPass = readBuffer<nm_float4>(workBuffer, 8u);
    const double beforeReorthogonalization =
        scalarNorm(afterFirstPass, 0u, 4u);
    runPartial(1u);
    runReduceAndUpdate(1u);
    const auto afterSecondPass = readBuffer<nm_float4>(workBuffer, 8u);
    const double postCGS2Norm = scalarNorm(afterSecondPass, 0u, 4u);

    require(initialNorm > 9.0e5,
        "near-dependent fixture did not begin at its large scale");
    require(beforeReorthogonalization < initialNorm * 1.0e-4,
        "first CGS update did not expose the cancellation residual");
    require(std::abs(beforeReorthogonalization - postCGS2Norm) >
                0.05 * beforeReorthogonalization,
        "second CGS pass did not materially correct the cancellation residual");

    dispatch(device, queue, measureNorm, {
        {0u, dispatchBuffer}, {30u, layoutBuffer}, {1u, workBuffer},
        {2u, partialBuffer}, {3u, stateBuffer}},
        2u * itemTileCount, 256u);
    dispatch(device, queue, finishColumn, {
        {0u, dispatchBuffer}, {30u, layoutBuffer}, {1u, solverBuffer},
        {2u, columnBuffer}, {3u, hessenbergBuffer},
        {4u, rotationsBuffer}, {5u, leastSquaresBuffer},
        {6u, stateBuffer}, {7u, forcingBuffer}, {8u, partialBuffer},
        {9u, nodeActive}, {10u, elementActive}},
        2u, 32u);
    dispatch(device, queue, normalize, {
        {0u, dispatchBuffer}, {30u, layoutBuffer}, {1u, columnBuffer},
        {2u, workBuffer}, {3u, nextBasisBuffer}, {4u, partialBuffer},
        {5u, stateBuffer}},
        2u * itemTileCount, 256u);

    const auto nextBasis = readBuffer<nm_float4>(nextBasisBuffer, 8u);
    const auto updatedStates = readBuffer<NMFGMRESStateGPU>(stateBuffer, 2u);
    const auto finalH = readBuffer<float>(hessenbergBuffer, hessenberg.size());
    const auto finalPartials = readBuffer<float>(
        partialBuffer, 2u * 2u * (NM_MIXED_FGMRES_RESTART + 1u) *
            partialCount);
    const auto actualWork = readBuffer<nm_float4>(workBuffer, 8u);
    const double storedNorm = finalPartials[
        (1u * (NM_MIXED_FGMRES_RESTART + 1u) + NM_MIXED_FGMRES_RESTART) *
        partialCount];
    const double actualPostNorm = scalarNorm(actualWork, 0u, 4u);
    require(std::isfinite(storedNorm) && storedNorm > 0.0,
        "finish did not preserve a positive raw Arnoldi norm");
    require(std::abs(storedNorm - actualPostNorm) <=
                std::max(1.0e-5, actualPostNorm * 2.0e-5),
        "stored post-CGS2 norm does not match the updated vector");
    const std::size_t hSubdiagonal = NM_MIXED_FGMRES_RESTART;
    require(finalH[hSubdiagonal] == 0.0f,
        "Givens finish did not zero the Hessenberg subdiagonal");

    const double nextNorm = scalarNorm(nextBasis, 0u, 4u);
    const double orthogonalDot =
        double(basis[0].x) * nextBasis[0].x +
        double(basis[1].x) * nextBasis[1].x +
        double(basis[4].x) * nextBasis[4].x;
    require(std::abs(nextNorm - 1.0) < 2.0e-4,
        "distributed normalizer did not produce a unit basis vector");
    require(std::abs(orthogonalDot) < 2.0e-3,
        "CGS2 output is not orthogonal to the near-dependent basis");
    require(updatedStates[0].diagnostics.w == 1.0f &&
            updatedStates[1].diagnostics.w == 1.0f,
        "scalar finish did not publish a column count for both environments");
    for (const std::size_t index : {2u, 3u, 6u, 7u})
        require(nextBasis[index].x == 0.0f &&
                nextBasis[index].y == 0.0f &&
                nextBasis[index].z == 0.0f &&
                nextBasis[index].w == 0.0f,
            "zero-residual environment did not normalize to a zero breakdown vector");
    require(updatedStates[1].diagnostics.x == 0.0f &&
            updatedStates[1].diagnostics.z == 1.0f,
        "zero-residual environment did not take the converged breakdown path");

    std::cout << "distributed_CGS2=true environments=2 cancellation=true "
                 "post_update_norm=true zero_breakdown=true result=PASS"
              << std::endl;
}


void checkInactiveTransportSpan(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> partial,
    id<MTLComputePipelineState> reduceRow,
    id<MTLComputePipelineState> update,
    id<MTLComputePipelineState> measureNorm,
    id<MTLComputePipelineState> finishColumn,
    id<MTLComputePipelineState> normalize) {
    // Reserved bit zero omits only the second FEM span from the tile stream.
    // Physical offsets remain fixed. Large poison values prove that projection
    // and norm skip that span; pressure in mechanical .w remains live.
    constexpr std::uint32_t kSkipInactiveTransportSpan = 1u << 0u;
    NMMatterDispatchGPU dispatchData{};
    dispatchData.abiVersion = NM_MATTER_ABI_VERSION;
    dispatchData.environmentCount = 2u;
    dispatchData.objectCount = 1u;
    dispatchData.femNodeCount = 2u;
    NMFGMRESLayoutGPU layout{};
    layout.unknownCount = 8u;
    layout.reserved = kSkipInactiveTransportSpan;
    NMMixedSolverGPU solver{};
    solver.nonlinearIterations.y = 1u;
    solver.residualTolerances.x = 1.0e-6f;
    solver.regularization.x = 1.0e-12f;
    constexpr std::uint32_t column = 0u;
    constexpr std::uint32_t partialCount = 8u;

    std::vector<nm_float4> basis(
        std::size_t(NM_MIXED_FGMRES_RESTART + 1u) * layout.unknownCount,
        nm_float4{0.0f, 0.0f, 0.0f, 0.0f});
    basis[0].x = 1.0f;
    basis[4].x = 1.0e6f;
    std::vector<nm_float4> work(8u,
        nm_float4{0.0f, 0.0f, 0.0f, 0.0f});
    work[0].x = 3.0f;
    work[1].w = 4.0f;
    work[4].x = 1.0e6f;
    work[6].x = -2.0e6f;
    work[7].y = 3.0e6f;

    std::vector<NMFGMRESStateGPU> states(2u);
    states[0].diagnostics = {0.0f, 5.0f, 0.0f, 0.0f};
    states[1].diagnostics = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> hessenberg(
        2u * std::size_t(NM_MIXED_FGMRES_RESTART + 1u) *
            NM_MIXED_FGMRES_RESTART, 0.0f);
    std::vector<NativeFloat2> rotations(
        2u * std::size_t(NM_MIXED_FGMRES_RESTART),
        NativeFloat2{1.0f, 0.0f});
    std::vector<float> leastSquares(
        2u * std::size_t(NM_MIXED_FGMRES_RESTART + 1u), 0.0f);
    leastSquares[0] = 5.0f;
    auto nodeActive = makeBuffer(device,
        std::vector<NMIndirectDispatchGPU>{NMIndirectDispatchGPU{}});
    auto elementActive = makeBuffer(device,
        std::vector<NMIndirectDispatchGPU>{NMIndirectDispatchGPU{}});
    auto dispatchBuffer = makeBuffer(device,
        std::vector<NMMatterDispatchGPU>{dispatchData});
    auto layoutBuffer = makeBuffer(device,
        std::vector<NMFGMRESLayoutGPU>{layout});
    auto solverBuffer = makeBuffer(device,
        std::vector<NMMixedSolverGPU>{solver});
    auto basisBuffer = makeBuffer(device, basis);
    auto workBuffer = makeBuffer(device, work);
    auto nextBasisBuffer = makeBuffer(device,
        std::vector<nm_float4>(8u, nm_float4{9.0f,9.0f,9.0f,9.0f}));
    auto partialBuffer = makeBuffer(device,
        std::vector<float>(2u * 2u * (NM_MIXED_FGMRES_RESTART + 1u) *
            partialCount, 0.0f));
    auto hessenbergBuffer = makeBuffer(device, hessenberg);
    auto rotationsBuffer = makeBuffer(device, rotations);
    auto leastSquaresBuffer = makeBuffer(device, leastSquares);
    auto stateBuffer = makeBuffer(device, states);
    auto columnBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{column});
    auto passBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto forcingBuffer = makeBuffer(device, std::vector<float>{0.1f});

    const auto runPass = [&](const std::uint32_t passIndex) {
        std::memcpy([passBuffer contents], &passIndex, sizeof(passIndex));
        dispatch(device, queue, partial, {
            {0u,dispatchBuffer},{1u,columnBuffer},{2u,passBuffer},
            {3u,basisBuffer},{4u,workBuffer},{5u,partialBuffer},
            {6u,stateBuffer},{30u,layoutBuffer}}, 2u, 256u);
        dispatch(device, queue, reduceRow, {
            {0u,dispatchBuffer},{30u,layoutBuffer},{1u,columnBuffer},
            {2u,passBuffer},{3u,hessenbergBuffer},{4u,partialBuffer},
            {5u,stateBuffer}}, 2u, 32u);
        dispatch(device, queue, update, {
            {0u,dispatchBuffer},{30u,layoutBuffer},{1u,columnBuffer},
            {2u,passBuffer},{3u,basisBuffer},{4u,workBuffer},
            {5u,hessenbergBuffer},{6u,partialBuffer},{7u,stateBuffer}},
            2u, 256u);
    };
    runPass(0u);
    runPass(1u);
    dispatch(device, queue, measureNorm, {
        {0u,dispatchBuffer},{30u,layoutBuffer},{1u,workBuffer},
        {2u,partialBuffer},{3u,stateBuffer}}, 2u, 256u);
    dispatch(device, queue, finishColumn, {
        {0u,dispatchBuffer},{30u,layoutBuffer},{1u,solverBuffer},
        {2u,columnBuffer},{3u,hessenbergBuffer},{4u,rotationsBuffer},
        {5u,leastSquaresBuffer},{6u,stateBuffer},{7u,forcingBuffer},
        {8u,partialBuffer},{9u,nodeActive},{10u,elementActive}}, 2u, 32u);
    dispatch(device, queue, normalize, {
        {0u,dispatchBuffer},{30u,layoutBuffer},{1u,columnBuffer},
        {2u,workBuffer},{3u,nextBasisBuffer},{4u,partialBuffer},
        {5u,stateBuffer}}, 2u, 256u);

    const auto updatedWork = readBuffer<nm_float4>(workBuffer, 8u);
    const auto nextBasis = readBuffer<nm_float4>(nextBasisBuffer, 8u);
    const auto updatedStates = readBuffer<NMFGMRESStateGPU>(stateBuffer, 2u);
    const auto partials = readBuffer<float>(partialBuffer,
        2u * 2u * (NM_MIXED_FGMRES_RESTART + 1u) * partialCount);
    const std::size_t normIndex = ((NM_MIXED_FGMRES_RESTART + 1u) + NM_MIXED_FGMRES_RESTART) * partialCount;
    require(std::abs(partials[normIndex] - 4.0f) < 2.0e-5f,
        "inactive field poison entered the norm or mechanical pressure lane was dropped");
    require(std::abs(nextBasis[1].w - 1.0f) < 2.0e-5f,
        "mechanical pressure lane was not normalized with the active Krylov vector");
    require(nextBasis[4].x == 0.0f && nextBasis[5].x == 0.0f &&
            nextBasis[6].x == 0.0f && nextBasis[7].y == 0.0f,
        "inactive field rows were not cleared from the next basis");
    require(updatedWork[4].x == 1.0e6f && updatedWork[6].x == -2.0e6f,
        "orthogonalization modified the physically retained inactive field span");
    require(std::abs(nextBasis[0].x) < 2.0e-5f &&
            std::abs(nextBasis[1].w - 1.0f) < 2.0e-5f,
        "inactive transport changed the mechanical-only Arnoldi direction");
    require(updatedStates[0].diagnostics.w == 1.0f,
        "inactive-transport fixture did not finish its active Arnoldi column");
    require(updatedStates[1].diagnostics.z == 1.0f &&
            updatedStates[1].diagnostics.w == 1.0f,
        "poisoned inactive transport activated the zero-residual environment");
    std::cout << "inactive_transport_tile_omission=true physical_offsets_preserved=true "
                 "poison_excluded=true pressure_lane_retained=true result=PASS"
              << std::endl;
}


struct CGS2IndirectConstants {
    NMMatterDispatchGPU dispatch;
    NMFGMRESLayoutGPU layout;
    std::uint32_t column;
    std::uint32_t pass0;
    std::uint32_t pass1;
    float forcing;
};

void checkCGS2IndirectRange(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> prepareRange,
    id<MTLComputePipelineState> partial,
    id<MTLComputePipelineState> reduceRow,
    id<MTLComputePipelineState> update,
    id<MTLComputePipelineState> measureNorm,
    id<MTLComputePipelineState> finishColumn,
    id<MTLComputePipelineState> normalize) {
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
    solver.regularization.x = 1.0e-12f;
    std::vector<nm_float4> basis(
        std::size_t(NM_MIXED_FGMRES_RESTART + 1u) * layout.unknownCount,
        nm_float4{0.0f,0.0f,0.0f,0.0f});
    basis[0].x = 1.0f;
    std::vector<nm_float4> work = {
        nm_float4{3.0f,0.0f,0.0f,0.0f},
        nm_float4{0.0f,0.0f,0.0f,1.0f}};
    std::vector<NMFGMRESStateGPU> states(1u);
    states[0].diagnostics = {0.0f,std::sqrt(10.0f),0.0f,0.0f};
    std::vector<float> hessenberg(
        std::size_t(NM_MIXED_FGMRES_RESTART + 1u) *
            NM_MIXED_FGMRES_RESTART, 0.0f);
    std::vector<NativeFloat2> rotations(NM_MIXED_FGMRES_RESTART,
        NativeFloat2{1.0f,0.0f});
    std::vector<float> leastSquares(
        NM_MIXED_FGMRES_RESTART + 1u, 0.0f);
    leastSquares[0] = std::sqrt(10.0f);
    std::vector<float> partials(2u * 2u *
        (NM_MIXED_FGMRES_RESTART + 1u) * 8u, 0.0f);
    auto dispatchBuffer = makeBuffer(device,
        std::vector<NMMatterDispatchGPU>{dispatchData});
    auto layoutBuffer = makeBuffer(device,
        std::vector<NMFGMRESLayoutGPU>{layout});
    auto solverBuffer = makeBuffer(device,
        std::vector<NMMixedSolverGPU>{solver});
    auto basisBuffer = makeBuffer(device,basis);
    auto workBuffer = makeBuffer(device,work);
    auto nextBasisBuffer = makeBuffer(device,
        std::vector<nm_float4>(2u,nm_float4{0.0f,0.0f,0.0f,0.0f}));
    auto stateBuffer = makeBuffer(device,states);
    auto hessenbergBuffer = makeBuffer(device,hessenberg);
    auto rotationsBuffer = makeBuffer(device,rotations);
    auto leastSquaresBuffer = makeBuffer(device,leastSquares);
    auto partialBuffer = makeBuffer(device,partials);
    auto nodeActive = makeBuffer(device,
        std::vector<NMIndirectDispatchGPU>{NMIndirectDispatchGPU{}});
    auto elementActive = makeBuffer(device,
        std::vector<NMIndirectDispatchGPU>{NMIndirectDispatchGPU{}});
    auto commandCountBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{9u});
    auto executionRange = makeBuffer(device,
        std::vector<std::uint32_t>{0u,0u,0u,0u});
    const std::uint32_t column = 0u;
    const std::uint32_t pass0 = 0u;
    const std::uint32_t pass1 = 1u;
    const float forcing = 0.5f;
    const CGS2IndirectConstants constants{
        dispatchData, layout, column, pass0, pass1, forcing};
    id<MTLBuffer> constantsBuffer = [device newBufferWithBytes:&constants
        length:sizeof(constants) options:MTLResourceStorageModeShared];
    require(constantsBuffer != nil,
        "CGS2 indirect constants allocation failed");
    MTLIndirectCommandBufferDescriptor* descriptor =
        [MTLIndirectCommandBufferDescriptor new];
    descriptor.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
    descriptor.inheritBuffers = NO;
    descriptor.inheritPipelineState = NO;
    descriptor.maxKernelBufferBindCount = 31u;
    id<MTLIndirectCommandBuffer> graph =
        [device newIndirectCommandBufferWithDescriptor:descriptor
            maxCommandCount:9u options:MTLResourceStorageModePrivate];
    require(graph != nil, "CGS2 indirect command buffer allocation failed");

    NSUInteger ordinal = 0u;
    const auto beginCommand = [&](id<MTLComputePipelineState> pipelineState) {
        id<MTLIndirectComputeCommand> cmd =
            [graph indirectComputeCommandAtIndex:ordinal++];
        [cmd setComputePipelineState:pipelineState];
        [cmd setKernelBuffer:constantsBuffer
            offset:offsetof(CGS2IndirectConstants, dispatch) atIndex:0u];
        [cmd setKernelBuffer:constantsBuffer
            offset:offsetof(CGS2IndirectConstants, layout) atIndex:30u];
        return cmd;
    };
    const auto bindBuffer = [&](id<MTLIndirectComputeCommand> cmd,
        const NSUInteger index, id<MTLBuffer> buffer,
        const NSUInteger offset = 0u) {
        [cmd setKernelBuffer:buffer offset:offset atIndex:index];
    };
    const auto dispatchCommand = [&](id<MTLIndirectComputeCommand> cmd,
        const NSUInteger groups, const NSUInteger width) {
        [cmd concurrentDispatchThreadgroups:MTLSizeMake(groups,1u,1u)
            threadsPerThreadgroup:MTLSizeMake(width,1u,1u)];
        [cmd setBarrier];
    };
    for (std::uint32_t pass = 0u; pass < 2u; ++pass) {
        const NSUInteger passOffset = pass == 0u
            ? offsetof(CGS2IndirectConstants, pass0)
            : offsetof(CGS2IndirectConstants, pass1);
        auto cmd = beginCommand(partial);
        bindBuffer(cmd,1u,constantsBuffer,
            offsetof(CGS2IndirectConstants,column));
        bindBuffer(cmd,2u,constantsBuffer,passOffset);
        bindBuffer(cmd,3u,basisBuffer);
        bindBuffer(cmd,4u,workBuffer);
        bindBuffer(cmd,5u,partialBuffer);
        bindBuffer(cmd,6u,stateBuffer);
        dispatchCommand(cmd,1u,256u);

        cmd = beginCommand(reduceRow);
        bindBuffer(cmd,1u,constantsBuffer,
            offsetof(CGS2IndirectConstants,column));
        bindBuffer(cmd,2u,constantsBuffer,passOffset);
        bindBuffer(cmd,3u,hessenbergBuffer);
        bindBuffer(cmd,4u,partialBuffer);
        bindBuffer(cmd,5u,stateBuffer);
        dispatchCommand(cmd,1u,32u);

        cmd = beginCommand(update);
        bindBuffer(cmd,1u,constantsBuffer,
            offsetof(CGS2IndirectConstants,column));
        bindBuffer(cmd,2u,constantsBuffer,passOffset);
        bindBuffer(cmd,3u,basisBuffer);
        bindBuffer(cmd,4u,workBuffer);
        bindBuffer(cmd,5u,hessenbergBuffer);
        bindBuffer(cmd,6u,partialBuffer);
        bindBuffer(cmd,7u,stateBuffer);
        dispatchCommand(cmd,1u,256u);
    }
    auto cmd = beginCommand(measureNorm);
    bindBuffer(cmd,1u,workBuffer);
    bindBuffer(cmd,2u,partialBuffer);
    bindBuffer(cmd,3u,stateBuffer);
    dispatchCommand(cmd,1u,256u);

    cmd = beginCommand(finishColumn);
    bindBuffer(cmd,1u,solverBuffer);
    bindBuffer(cmd,2u,constantsBuffer,
        offsetof(CGS2IndirectConstants,column));
    bindBuffer(cmd,3u,hessenbergBuffer);
    bindBuffer(cmd,4u,rotationsBuffer);
    bindBuffer(cmd,5u,leastSquaresBuffer);
    bindBuffer(cmd,6u,stateBuffer);
    bindBuffer(cmd,7u,constantsBuffer,
        offsetof(CGS2IndirectConstants,forcing));
    bindBuffer(cmd,8u,partialBuffer);
    bindBuffer(cmd,9u,nodeActive);
    bindBuffer(cmd,10u,elementActive);
    dispatchCommand(cmd,1u,32u);

    cmd = beginCommand(normalize);
    bindBuffer(cmd,1u,constantsBuffer,
        offsetof(CGS2IndirectConstants,column));
    bindBuffer(cmd,2u,workBuffer);
    bindBuffer(cmd,3u,nextBasisBuffer);
    bindBuffer(cmd,4u,partialBuffer);
    bindBuffer(cmd,5u,stateBuffer);
    dispatchCommand(cmd,1u,256u);
    require(ordinal == 9u, "CGS2 ICB did not encode nine pipeline commands");

    const auto execute = [&]() {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            command != nil ? [command computeCommandEncoder] : nil;
        require(command != nil && encoder != nil,
            "cannot allocate CGS2 ICB command encoder");
        [encoder setComputePipelineState:prepareRange];
        [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
        [encoder setBuffer:stateBuffer offset:0u atIndex:1u];
        [encoder setBuffer:commandCountBuffer offset:0u atIndex:2u];
        [encoder setBuffer:executionRange offset:0u atIndex:3u];
        [encoder dispatchThreadgroups:MTLSizeMake(1u,1u,1u)
            threadsPerThreadgroup:MTLSizeMake(1u,1u,1u)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        for (id<MTLBuffer> buffer : {dispatchBuffer,layoutBuffer,solverBuffer,
                basisBuffer,workBuffer,nextBasisBuffer,stateBuffer,
                hessenbergBuffer,rotationsBuffer,leastSquaresBuffer,
                partialBuffer,nodeActive,elementActive,constantsBuffer})
            [encoder useResource:buffer
                usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        [encoder useResource:graph usage:MTLResourceUsageRead];
        [encoder useResource:executionRange
            usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        [encoder executeCommandsInBuffer:graph indirectBuffer:executionRange
            indirectBufferOffset:0u];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
            command.error != nil
                ? std::string("CGS2 ICB execution failed: ") +
                    [[command.error localizedDescription] UTF8String]
                : "CGS2 ICB command buffer did not complete");
    };

    execute();
    auto range = readBuffer<std::uint32_t>(executionRange,2u);
    auto activeState = readBuffer<NMFGMRESStateGPU>(stateBuffer,1u);
    auto activeNextBasis = readBuffer<nm_float4>(nextBasisBuffer,2u);
    require(range[0] == 0u && range[1] == 9u,
        "GPU range helper did not enable all nine active ICB commands");
    require(activeState[0].diagnostics.w == 1.0f &&
            activeState[0].diagnostics.z == 1.0f,
        "active ICB graph did not complete the Arnoldi scalar finish");
    require(std::abs(activeNextBasis[1].w - 1.0f) < 2.0e-5f,
        "reserved-bit-clear ICB graph did not retain its active transport row");

    // Poison writable solver buffers before requesting an empty range. The
    // only expected write is the range itself; normalize would alter sentinels.
    states[0].diagnostics = {17.0f,23.0f,1.0f,1.0f};
    std::memcpy([stateBuffer contents],states.data(),sizeof(NMFGMRESStateGPU));
    for (auto& value : basis) value = nm_float4{2.0f,3.0f,4.0f,5.0f};
    for (auto& value : work) value = nm_float4{6.0f,7.0f,8.0f,9.0f};
    for (auto& value : hessenberg) value = 10.0f;
    for (auto& value : rotations) value = NativeFloat2{11.0f,12.0f};
    for (auto& value : leastSquares) value = 13.0f;
    for (auto& value : partials) value = 14.0f;
    std::memcpy([basisBuffer contents],basis.data(),basis.size()*sizeof(nm_float4));
    std::memcpy([workBuffer contents],work.data(),work.size()*sizeof(nm_float4));
    std::memcpy([hessenbergBuffer contents],hessenberg.data(),hessenberg.size()*sizeof(float));
    std::memcpy([rotationsBuffer contents],rotations.data(),rotations.size()*sizeof(NativeFloat2));
    std::memcpy([leastSquaresBuffer contents],leastSquares.data(),leastSquares.size()*sizeof(float));
    std::memcpy([partialBuffer contents],partials.data(),partials.size()*sizeof(float));
    std::vector<nm_float4> nextSentinel(2u,nm_float4{15.0f,16.0f,17.0f,18.0f});
    std::memcpy([nextBasisBuffer contents],nextSentinel.data(),nextSentinel.size()*sizeof(nm_float4));
    auto nodeSentinel = std::vector<NMIndirectDispatchGPU>{
        NMIndirectDispatchGPU{19u,20u,21u,22u}};
    auto elementSentinel = std::vector<NMIndirectDispatchGPU>{
        NMIndirectDispatchGPU{23u,24u,25u,26u}};
    std::memcpy([nodeActive contents],nodeSentinel.data(),sizeof(NMIndirectDispatchGPU));
    std::memcpy([elementActive contents],elementSentinel.data(),sizeof(NMIndirectDispatchGPU));

    const auto beforeStates = readBuffer<NMFGMRESStateGPU>(stateBuffer,1u);
    const auto beforeBasis = readBuffer<nm_float4>(basisBuffer,basis.size());
    const auto beforeWork = readBuffer<nm_float4>(workBuffer,work.size());
    const auto beforeNext = readBuffer<nm_float4>(nextBasisBuffer,2u);
    const auto beforeH = readBuffer<float>(hessenbergBuffer,hessenberg.size());
    const auto beforeRotations = readBuffer<NativeFloat2>(rotationsBuffer,rotations.size());
    const auto beforeLS = readBuffer<float>(leastSquaresBuffer,leastSquares.size());
    const auto beforePartials = readBuffer<float>(partialBuffer,partials.size());
    execute();
    range = readBuffer<std::uint32_t>(executionRange,2u);
    require(range[0] == 0u && range[1] == 0u,
        "GPU range helper did not suppress a converged ICB tail");
    const auto same = [](const auto& a, const auto& b) {
        return a.size() == b.size() &&
            std::memcmp(a.data(),b.data(),a.size()*sizeof(a[0])) == 0;
    };
    require(same(beforeStates,readBuffer<NMFGMRESStateGPU>(stateBuffer,1u)) &&
            same(beforeBasis,readBuffer<nm_float4>(basisBuffer,basis.size())) &&
            same(beforeWork,readBuffer<nm_float4>(workBuffer,work.size())) &&
            same(beforeNext,readBuffer<nm_float4>(nextBasisBuffer,2u)) &&
            same(beforeH,readBuffer<float>(hessenbergBuffer,hessenberg.size())) &&
            same(beforeRotations,readBuffer<NativeFloat2>(rotationsBuffer,rotations.size())) &&
            same(beforeLS,readBuffer<float>(leastSquaresBuffer,leastSquares.size())) &&
            same(beforePartials,readBuffer<float>(partialBuffer,partials.size())) &&
            same(nodeSentinel,readBuffer<NMIndirectDispatchGPU>(nodeActive,1u)) &&
            same(elementSentinel,readBuffer<NMIndirectDispatchGPU>(elementActive,1u)),
        "zero-range CGS2 ICB changed solver state or output buffers");
    std::cout << "cgs2_ICB_range=true active_commands=9 converged_commands=0 "
                 "active_transport_row_retained=true zero_range_state_unchanged=true result=PASS"
              << std::endl;
}

} // namespace


struct NativeZ0OwnerGate {
    std::uint32_t environmentCount;
    std::uint32_t externalStatusStrideWords;
    std::uint32_t acceptedStatusCode;
    std::uint32_t environmentIndexOffsetWords;
    std::uint32_t expectedEnvironmentBase;
    std::uint32_t completedCountOffsetWords;
    std::uint32_t expectedCompletedCount;
    std::uint32_t reserved0;
};

void checkAcceptedZ0Recycle(
    id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library) {
    auto reset = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_accepted_z0_reset_candidate");
    auto resetCapture = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_accepted_z0_reset_candidate_for_capture");
    auto capture = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_accepted_z0_capture_candidate");
    auto promote = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_accepted_z0_promote_owner_candidate");
    auto snapshot = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_accepted_z0_snapshot_owner_contact_keys");
    auto validate = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_accepted_z0_validate_constraint_mask");
    auto add = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_accepted_z0_add_to_preconditioned_basis");
    MTLFunctionConstantValues* femConstants = [MTLFunctionConstantValues new];
    const BOOL disabled = NO;
    const std::uint32_t noMaterial = NM_INVALID_INDEX;
    for (NSUInteger index = 30u; index <= 34u; ++index)
        [femConstants setConstantValue:&disabled type:MTLDataTypeBool atIndex:index];
    for (NSUInteger index : {35u, 36u})
        [femConstants setConstantValue:&noMaterial type:MTLDataTypeUInt atIndex:index];
    NSError* femError = nil;
    id<MTLFunction> femFunction = [library newFunctionWithName:
        @"numi_matter_metal::nm_fem_apply_operator_elements"
        constantValues:femConstants error:&femError];
    require(femFunction != nil, "recycle fixture cannot specialize current FEM operator");
    id<MTLComputePipelineState> femApply =
        [device newComputePipelineStateWithFunction:femFunction error:&femError];
    require(femApply != nil, "recycle fixture cannot create current FEM operator");

    NMMatterDispatchGPU d{};
    d.abiVersion = NM_MATTER_ABI_VERSION;
    d.environmentCount = 1u;
    d.objectCount = 1u;
    d.femNodeCount = 4u;
    d.tetrahedronCount = 1u;
    d.contactPairCount = 1u;
    d.parameterCount = 1u;
    d.materialStateStride = 1u;
    d.gravityAndTimestep.w = 0.25f;
    NMMatterStatusGPU healthy{};
    healthy.code = NM_STATUS_SUCCESS;
    healthy.environment = 0u;
    NMMicrostepGPU micro{};
    micro.flags = NM_MICROSTEP_FGMRES_OPERATOR;
    micro.time = {0.25f, 4.0f, 0.0f, 0.25f};
    NMFGMRESStateGPU state{};
    state.nonlinear = {1.0f, 1.0f, 0.0f, 0.0f};

    auto dispatchBuffer = makeBuffer(device, std::vector<NMMatterDispatchGPU>{d});
    auto statusBuffer = makeBuffer(device, std::vector<NMMatterStatusGPU>{healthy});
    auto stateBuffer = makeBuffer(device, std::vector<NMFGMRESStateGPU>{state});
    auto flagsBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto validBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto candidateDtBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto acceptedDtBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
    std::vector<nm_float4> candidateValues(4u, nm_float4{});
    candidateValues[1].x = 0.25f;
    candidateValues[2].y = -0.125f;
    candidateValues[3].z = 0.0625f;
    auto solutionBuffer = makeBuffer(device, candidateValues);
    std::vector<NMFEMNodeStateGPU> nodes(4u);
    for (auto& node : nodes) node.restAndFixed.w = 0.0f;
    auto nodeBuffer = makeBuffer(device, nodes);
    auto candidateZBuffer = makeBuffer(device, std::vector<nm_float4>(4u, nm_float4{}));
    auto acceptedZBuffer = makeBuffer(device, std::vector<nm_float4>(4u, nm_float4{}));
    auto acceptedMaskBuffer = makeBuffer(device, std::vector<std::uint32_t>(4u, 0u));

    NMContactSampleGPU sample{};
    sample.identity = {11u, 12u, 13u, 14u};
    auto sampleBuffer = makeBuffer(device, std::vector<NMContactSampleGPU>{sample});
    auto zeroBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto oneBuffer = makeBuffer(device, std::vector<std::uint32_t>{1u});
    id<MTLFunction> contactLayout = [library newFunctionWithName:
        @"numi_matter_metal::nm_primal_contact_argument_layout"];
    require(contactLayout != nil, "recycle fixture contact layout missing");
    id<MTLArgumentEncoder> contactEncoder =
        [contactLayout newArgumentEncoderWithBufferIndex:0u];
    require(contactEncoder != nil, "recycle fixture contact encoder missing");
    id<MTLBuffer> contactArguments = [device newBufferWithLength:
        contactEncoder.encodedLength options:MTLResourceStorageModeShared];
    require(contactArguments != nil, "recycle fixture contact buffer allocation failed");
    [contactEncoder setArgumentBuffer:contactArguments offset:0u];
    for (NSUInteger i = 0u; i < 20u; ++i)
        [contactEncoder setBuffer:zeroBuffer offset:0u atIndex:i];
    [contactEncoder setBuffer:sampleBuffer offset:0u atIndex:1u];
    [contactEncoder setBuffer:zeroBuffer offset:0u atIndex:15u];
    [contactEncoder setBuffer:oneBuffer offset:0u atIndex:16u];

    NativeZ0OwnerGate owner{1u, 4u, 0x51u, 1u, 0u, 2u, 7u, 0u};
    // Metal receives this owner record separately as constant buffer(3).
    std::vector<std::uint32_t> externalWords{0x50u, 0u, 7u, 0u};
    auto externalBuffer = makeBuffer(device, externalWords);
    auto contactKeysBuffer = makeBuffer(device, std::vector<nm_uint4>{
        nm_uint4{0u, 0u, 0u, 0u}});

    auto runCapture = [&] {
        dispatch(device, queue, resetCapture, {
            {0u, dispatchBuffer}, {1u, stateBuffer}, {2u, statusBuffer},
            {3u, flagsBuffer}}, 1u, 1u);
        dispatch(device, queue, capture, {
            {0u, dispatchBuffer}, {1u, solutionBuffer}, {2u, nodeBuffer},
            {3u, candidateZBuffer}, {4u, flagsBuffer},
            {5u, makeBuffer(device, std::vector<NMMicrostepGPU>{micro})},
            {6u, candidateDtBuffer}, {7u, stateBuffer}, {8u, statusBuffer}},
            1u, 256u);
    };
    auto runOwnerPromotion = [&] {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            command != nil ? [command computeCommandEncoder] : nil;
        require(command != nil && encoder != nil,
            "recycle fixture cannot allocate owner encoder");
        [encoder setComputePipelineState:promote];
        [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
        [encoder setBuffer:statusBuffer offset:0u atIndex:1u];
        [encoder setBuffer:externalBuffer offset:0u atIndex:2u];
        [encoder setBytes:&owner length:sizeof(owner) atIndex:3u];
        [encoder setBuffer:candidateZBuffer offset:0u atIndex:4u];
        [encoder setBuffer:acceptedZBuffer offset:0u atIndex:5u];
        [encoder setBuffer:flagsBuffer offset:0u atIndex:6u];
        [encoder setBuffer:validBuffer offset:0u atIndex:7u];
        [encoder setBuffer:nodeBuffer offset:0u atIndex:8u];
        [encoder setBuffer:acceptedMaskBuffer offset:0u atIndex:9u];
        [encoder setBuffer:candidateDtBuffer offset:0u atIndex:10u];
        [encoder setBuffer:acceptedDtBuffer offset:0u atIndex:11u];
        [encoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
            "recycle fixture owner promotion failed");
    };
    auto runSnapshot = [&] {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            command != nil ? [command computeCommandEncoder] : nil;
        require(command != nil && encoder != nil,
            "recycle fixture cannot allocate contact snapshot encoder");
        [encoder setComputePipelineState:snapshot];
        [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
        [encoder setBuffer:statusBuffer offset:0u atIndex:1u];
        [encoder setBuffer:externalBuffer offset:0u atIndex:2u];
        [encoder setBytes:&owner length:sizeof(owner) atIndex:3u];
        [encoder setBuffer:contactArguments offset:0u atIndex:4u];
        [encoder setBuffer:flagsBuffer offset:0u atIndex:5u];
        [encoder setBuffer:validBuffer offset:0u atIndex:6u];
        [encoder setBuffer:contactKeysBuffer offset:0u atIndex:7u];
        [encoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
            "recycle fixture contact-key snapshot failed");
    };

    dispatch(device, queue, reset,
        {{0u, dispatchBuffer}, {1u, flagsBuffer}}, 1u, 1u);
    runCapture();
    auto capturedFlags = readBuffer<std::uint32_t>(flagsBuffer, 1u)[0];
    require(capturedFlags == 5u,
        "nonzero finite Newton correction was not captured as a fresh candidate");

    // A completed nonlinear tail may replay the capture graph after the
    // solution scratch has become zero. Neither the per-capture reset nor the
    // capture kernel may erase the last active Newton correction.
    NMFGMRESStateGPU completedTail = state;
    completedTail.nonlinear.w = 1.0f;
    std::memcpy([stateBuffer contents], &completedTail, sizeof(completedTail));
    std::fill(candidateValues.begin(), candidateValues.end(), nm_float4{});
    std::memcpy([solutionBuffer contents], candidateValues.data(),
        candidateValues.size() * sizeof(nm_float4));
    runCapture();
    require(readBuffer<std::uint32_t>(flagsBuffer, 1u)[0] == 5u,
        "completed Newton tail reset a fresh correction candidate");
    const auto tailPreserved = readBuffer<nm_float4>(candidateZBuffer, 4u);
    require(tailPreserved[1].x == 0.25f && tailPreserved[2].y == -0.125f &&
            tailPreserved[3].z == 0.0625f,
        "completed Newton tail overwrote the last active correction");
    std::memcpy([stateBuffer contents], &state, sizeof(state));
    std::memcpy([solutionBuffer contents], candidateValues.data(),
        candidateValues.size() * sizeof(nm_float4));
    candidateValues[1].x = 0.25f;
    candidateValues[2].y = -0.125f;
    candidateValues[3].z = 0.0625f;
    std::memcpy([solutionBuffer contents], candidateValues.data(),
        candidateValues.size() * sizeof(nm_float4));

    // Rejected/failing owner status must never promote the candidate.
    runOwnerPromotion();
    require(readBuffer<std::uint32_t>(validBuffer, 1u)[0] == 0u,
        "rejected owner transaction promoted a solver direction");
    for (const auto& v : readBuffer<nm_float4>(acceptedZBuffer, 4u))
        require(v.x == 0.0f && v.y == 0.0f && v.z == 0.0f,
            "rejected owner transaction left a promoted direction component");

    externalWords[0] = owner.acceptedStatusCode;
    std::memcpy([externalBuffer contents], externalWords.data(),
        externalWords.size() * sizeof(std::uint32_t));
    runOwnerPromotion();
    require(readBuffer<std::uint32_t>(validBuffer, 1u)[0] == 1u,
        "successful owner transaction did not promote a nonzero candidate");
    const auto promoted = readBuffer<nm_float4>(acceptedZBuffer, 4u);
    require(promoted[1].x == 0.25f && promoted[2].y == -0.125f &&
            promoted[3].z == 0.0625f,
        "owner promotion changed the accepted correction vector");
    std::uint32_t expectedDtBits = 0u;
    std::memcpy(&expectedDtBits, &micro.time.x, sizeof(expectedDtBits));
    require(readBuffer<std::uint32_t>(acceptedDtBuffer, 1u)[0] ==
            expectedDtBits,
        "owner promotion did not store the candidate timestep");

    // The accepted direction is injected as a preconditioned basis vector,
    // then the current FEM operator is applied to it. Changing the current
    // tangent cache changes A z, proving only z is recycled.
    auto basisBuffer = makeBuffer(device, std::vector<nm_float4>(4u, nm_float4{}));
    auto useCountsBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
    dispatch(device, queue, add, {
        {0u, dispatchBuffer}, {1u, acceptedZBuffer}, {2u, validBuffer},
        {3u, basisBuffer}, {4u, useCountsBuffer}}, 1u, 256u);
    require(readBuffer<std::uint32_t>(useCountsBuffer, 1u)[0] == 1u,
        "accepted Z0 actual injection counter was not incremented");
    require(readBuffer<nm_float4>(basisBuffer, 4u)[1].x == 0.25f,
        "valid accepted direction was not added to the first preconditioned basis");
    NMContinuumObjectGPU object{};
    object.representation = NM_REPRESENTATION_FEM;
    object.flags = NM_OBJECT_ACTIVE;
    object.solver.y = 0u;
    NMSchedulerStateGPU scheduler{};
    scheduler.numerical.w = 1.0f;
    NMAdaptiveStateGPU adaptive{};
    adaptive.activeRepresentation = NM_REPRESENTATION_FEM;
    NMTetrahedronGPU tet{};
    tet.nodes = {0u, 1u, 2u, 3u};
    tet.inverseRestRow0 = {1.0f, 0.0f, 0.0f, 1.0f};
    tet.inverseRestRow1 = {0.0f, 1.0f, 0.0f, 0.0f};
    tet.inverseRestRow2 = {0.0f, 0.0f, 1.0f, 0.0f};
    tet.identity = {0u, 0u, 0u, NM_OBJECT_ACTIVE};
    auto microBuffer = makeBuffer(device, std::vector<NMMicrostepGPU>{micro});
    auto objectBuffer = makeBuffer(device, std::vector<NMContinuumObjectGPU>{object});
    auto materialBuffer = makeBuffer(device, std::vector<NMMaterialGPU>{NMMaterialGPU{}});
    auto programBuffer = makeBuffer(device, std::vector<NMScalarProgramGPU>{NMScalarProgramGPU{}});
    auto instructionBuffer = makeBuffer(device, std::vector<NMExpressionInstructionGPU>{NMExpressionInstructionGPU{}});
    auto parameterBuffer = makeBuffer(device, std::vector<float>{0.0f});
    auto tetrahedronBuffer = makeBuffer(device, std::vector<NMTetrahedronGPU>{tet});
    auto acceptedMaterialStateBuffer = makeBuffer(device, std::vector<float>{0.0f});
    auto schedulerBuffer = makeBuffer(device, std::vector<NMSchedulerStateGPU>{scheduler});
    auto adaptiveBuffer = makeBuffer(device, std::vector<NMAdaptiveStateGPU>{adaptive});
    auto elementBuffer = makeBuffer(device, std::vector<NMFEMElementVectorGPU>{NMFEMElementVectorGPU{}});
    auto mixedBuffer = makeBuffer(device, std::vector<NMMixedMaterialGPU>{NMMixedMaterialGPU{}});
    auto fieldBuffer = makeBuffer(device, std::vector<NMFEMFieldStateGPU>(4u));
    auto learnedMaterialBuffer = makeBuffer(device, std::vector<NMLearnedMaterialGPU>{NMLearnedMaterialGPU{}});
    auto learnedLayerBuffer = makeBuffer(device, std::vector<NMLearnedLayerGPU>{NMLearnedLayerGPU{}});
    auto learnedWeightBuffer = makeBuffer(device, std::vector<float>{0.0f});
    auto solverBuffer = makeBuffer(device, std::vector<NMMixedSolverGPU>{NMMixedSolverGPU{}});
    auto tangentValidBuffer = makeBuffer(device, std::vector<std::uint32_t>{1u});
    auto cacheModeBuffer = makeBuffer(device, std::vector<std::uint32_t>{1u});
    auto fallbackMetricsBuffer = makeBuffer(device, std::vector<nm_float4>{nm_float4{}});
    std::vector<float> tangentCache(81u, 0.0f);
    for (std::uint32_t i = 0u; i < 9u; ++i) tangentCache[i * 9u + i] = 1.0f;
    auto tangentCacheBuffer = makeBuffer(device, tangentCache);
    auto runCurrentOperator = [&] {
        dispatch(device, queue, femApply, {
            {0u, dispatchBuffer}, {1u, microBuffer}, {2u, objectBuffer},
            {3u, materialBuffer}, {4u, programBuffer}, {5u, instructionBuffer},
            {6u, parameterBuffer}, {7u, nodeBuffer}, {8u, tetrahedronBuffer},
            {9u, acceptedMaterialStateBuffer}, {10u, basisBuffer},
            {11u, schedulerBuffer}, {12u, adaptiveBuffer}, {13u, elementBuffer},
            {14u, statusBuffer}, {16u, mixedBuffer}, {17u, fieldBuffer},
            {18u, learnedMaterialBuffer}, {19u, learnedLayerBuffer},
            {20u, learnedWeightBuffer}, {21u, solverBuffer}, {22u, stateBuffer},
            {23u, tangentCacheBuffer}, {24u, tangentValidBuffer},
            {25u, cacheModeBuffer}, {26u, fallbackMetricsBuffer}}, 1u, 256u);
        const auto result = readBuffer<NMFEMElementVectorGPU>(elementBuffer, 1u)[0];
        return std::abs(result.node0.x) + std::abs(result.node1.x) +
            std::abs(result.node2.x) + std::abs(result.node3.x) +
            std::abs(result.node0.y) + std::abs(result.node0.z);
    };
    const float action1 = runCurrentOperator();
    require(std::isfinite(action1) && action1 > 0.0f,
        "current FEM operator did not act on the recycled direction");
    for (std::uint32_t i = 0u; i < 9u; ++i) tangentCache[i * 9u + i] = 2.0f;
    std::memcpy([tangentCacheBuffer contents], tangentCache.data(),
        tangentCache.size() * sizeof(float));
    const float action2 = runCurrentOperator();
    require(std::isfinite(action2) && std::abs(action2 - 2.0f * action1) <=
            2.0e-6f * std::max(1.0f, action2),
        "recycled z reused stale A z instead of applying the current operator");

    // A zero correction is a fallback, not a replacement. It must preserve
    // the last useful vector while still allowing physical owner promotion.
    std::fill(candidateValues.begin(), candidateValues.end(), nm_float4{});
    std::memcpy([solutionBuffer contents], candidateValues.data(),
        candidateValues.size() * sizeof(nm_float4));
    runCapture();
    require(readBuffer<std::uint32_t>(flagsBuffer, 1u)[0] == 1u,
        "zero correction did not remain a nonreplacement candidate");
    runOwnerPromotion();
    const auto afterZero = readBuffer<nm_float4>(acceptedZBuffer, 4u);
    require(readBuffer<std::uint32_t>(validBuffer, 1u)[0] == 1u &&
            afterZero[1].x == promoted[1].x &&
            afterZero[2].y == promoted[2].y &&
            afterZero[3].z == promoted[3].z,
        "zero correction replaced the useful accepted direction");

    // Refresh the key snapshot with the unchanged contact identity. A changed
    // identity on the following zero-correction step must invalidate Z0.
    std::fill(candidateValues.begin(), candidateValues.end(), nm_float4{});
    candidateValues[1].x = 0.25f;
    std::memcpy([solutionBuffer contents], candidateValues.data(),
        candidateValues.size() * sizeof(nm_float4));
    runCapture();
    runOwnerPromotion();
    runSnapshot();
    sample.identity.x = 99u;
    std::memcpy([sampleBuffer contents], &sample, sizeof(sample));
    std::fill(candidateValues.begin(), candidateValues.end(), nm_float4{});
    std::memcpy([solutionBuffer contents], candidateValues.data(),
        candidateValues.size() * sizeof(nm_float4));
    runCapture();
    runOwnerPromotion();
    runSnapshot();
    require(readBuffer<std::uint32_t>(validBuffer, 1u)[0] == 0u,
        "contact identity change did not invalidate accepted Z0");

    // Re-establish a valid direction, then bitwise-change dt and ensure the
    // pre-use validator rejects it without altering any physical state.
    std::fill(candidateValues.begin(), candidateValues.end(), nm_float4{});
    candidateValues[1].x = 0.25f;
    std::memcpy([solutionBuffer contents], candidateValues.data(),
        candidateValues.size() * sizeof(nm_float4));
    runCapture();
    runOwnerPromotion();
    micro.time.x = 0.5f;
    micro.time.y = 2.0f;
    auto changedMicroBuffer = makeBuffer(device, std::vector<NMMicrostepGPU>{micro});
    dispatch(device, queue, validate, {
        {0u, dispatchBuffer}, {1u, nodeBuffer}, {2u, acceptedMaskBuffer},
        {3u, validBuffer}, {4u, changedMicroBuffer}, {5u, acceptedDtBuffer}},
        1u, 256u);
    require(readBuffer<std::uint32_t>(validBuffer, 1u)[0] == 0u,
        "timestep change did not invalidate accepted Z0");

    std::cout << "accepted_z0_owner_gate=true zero_fallback=true "
                 "contact_invalidation=true dt_invalidation=true "
                 "current_A_applied=true result=PASS\n";
}

void checkRegionalSchurInteriorExtraction(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLLibrary> library) {
    NMMatterDispatchGPU dispatchData{};
    dispatchData.abiVersion = NM_MATTER_ABI_VERSION;
    dispatchData.environmentCount = 1u;
    dispatchData.femNodeCount = 4u;
    const std::uint32_t nodeCount = 4u;
    const std::uint32_t patchDofs = 6u;
    std::vector<float> graphMatrix(patchDofs * patchDofs);
    for (std::uint32_t row = 0u; row < patchDofs; ++row)
        for (std::uint32_t column = 0u; column < patchDofs; ++column)
            graphMatrix[row * patchDofs + column] =
                float(1u + row * 97u + column * 3u);
    // Use a nonzero patch offset and a permuted global-node map. The matrix
    // indices must follow the patch slots, not assume patch offset zero or
    // global node IDs equal local matrix indices.
    const std::vector<nm_uint4> graphPatches{{1u, 2u, 0u, 0u}};
    const std::vector<std::uint32_t> graphPatchNodes{0u,3u,1u,2u};
    // The nonzero offset distinguishes extraction descriptors from LU
    // descriptors, which begin with this offset and then carry the node count.
    const std::vector<std::uint32_t> interiorValues{99u,1u,2u};
    // Two source descriptors target disjoint output windows. The GPU launch
    // assigns one threadgroup to each {environment, source}, so this catches
    // accidental all-source scans and source/output aliasing.
    const std::vector<nm_uint4> schurSources{
        {0u,1u,2u,0u}, {0u,1u,2u,36u}};
    std::vector<float> interiorMatrix(72u, -1.0f);
    std::vector<NMMatterStatusGPU> status(1u);
    const auto dispatchBuffer = makeBuffer(device,
        std::vector<NMMatterDispatchGPU>{dispatchData});
    const auto graphPatchBuffer = makeBuffer(device, graphPatches);
    const auto graphNodeBuffer = makeBuffer(device, graphPatchNodes);
    const auto interiorValueBuffer = makeBuffer(device, interiorValues);
    const auto sourceBuffer = makeBuffer(device, schurSources);
    const auto matrixBuffer = makeBuffer(device, graphMatrix);
    const auto outputBuffer = makeBuffer(device, interiorMatrix);
    const auto statusBuffer = makeBuffer(device, status);
    const std::uint32_t patchCount = 1u;
    const std::uint32_t patchNodeCount = nodeCount;
    const std::uint32_t matrixFloatCount =
        static_cast<std::uint32_t>(graphMatrix.size());
    const std::uint32_t sourceCount =
        static_cast<std::uint32_t>(schurSources.size());
    const std::uint32_t outputFloatCount = 72u;
    const std::uint32_t interiorPatchValueCount =
        static_cast<std::uint32_t>(interiorValues.size());
    const auto extract = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_graph_extract_schur_interior");
    dispatch(device, queue, extract, {
        {0u, dispatchBuffer}, {1u, graphPatchBuffer}, {2u, graphNodeBuffer},
        {3u, interiorValueBuffer}, {4u, sourceBuffer}, {5u, matrixBuffer},
        {6u, outputBuffer}, {7u, statusBuffer},
        {8u, makeBuffer(device, std::vector<std::uint32_t>{patchCount})},
        {9u, makeBuffer(device, std::vector<std::uint32_t>{patchNodeCount})},
        {10u, makeBuffer(device, std::vector<std::uint32_t>{matrixFloatCount})},
        {11u, makeBuffer(device, std::vector<std::uint32_t>{sourceCount})},
        {12u, makeBuffer(device, std::vector<std::uint32_t>{outputFloatCount})},
        {13u, makeBuffer(device,
            std::vector<std::uint32_t>{interiorPatchValueCount})}
    }, sourceCount, 256u);
    const auto actual = readBuffer<float>(outputBuffer, outputFloatCount);
    const auto actualStatus = readBuffer<NMMatterStatusGPU>(statusBuffer, 1u)[0];
    require(actualStatus.code == NM_STATUS_SUCCESS,
        "Schur A_ii extraction reported an invalid graph descriptor");
    // The selected patch-local nodes are slots 0 and 1 (absolute graph patch
    // values 1 and 2), independent of their global IDs {3,1}.
    for (std::uint32_t source = 0u; source < sourceCount; ++source) {
        for (std::uint32_t row = 0u; row < 6u; ++row) {
            const std::uint32_t fullRow = row;
            for (std::uint32_t column = 0u; column < 6u; ++column) {
                const std::uint32_t fullColumn = column;
                require(actual[source * 36u + row * 6u + column] ==
                        graphMatrix[fullRow * patchDofs + fullColumn],
                    "Schur A_ii extraction changed a graph patch matrix entry");
            }
        }
    }
    // An out-of-range flattened interior descriptor must be rejected before
    // reading its value from the map.
    const std::vector<nm_uint4> invalidSchurSources{{0u,3u,2u,0u}};
    const auto invalidSourceBuffer = makeBuffer(device, invalidSchurSources);
    const auto invalidOutputBuffer = makeBuffer(device,
        std::vector<float>(outputFloatCount, -1.0f));
    NMMatterStatusGPU healthyStatus{};
    const auto healthyStatusBuffer = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{healthyStatus});
    const std::uint32_t invalidSourceCount = 1u;
    const std::uint32_t invalidOutputFloatCount = 36u;
    dispatch(device, queue, extract, {
        {0u, dispatchBuffer}, {1u, graphPatchBuffer}, {2u, graphNodeBuffer},
        {3u, interiorValueBuffer}, {4u, invalidSourceBuffer},
        {5u, matrixBuffer}, {6u, invalidOutputBuffer},
        {7u, healthyStatusBuffer},
        {8u, makeBuffer(device, std::vector<std::uint32_t>{patchCount})},
        {9u, makeBuffer(device, std::vector<std::uint32_t>{patchNodeCount})},
        {10u, makeBuffer(device, std::vector<std::uint32_t>{matrixFloatCount})},
        {11u, makeBuffer(device, std::vector<std::uint32_t>{invalidSourceCount})},
        {12u, makeBuffer(device, std::vector<std::uint32_t>{invalidOutputFloatCount})},
        {13u, makeBuffer(device,
            std::vector<std::uint32_t>{interiorPatchValueCount})}
    }, invalidSourceCount, 256u);
    const auto invalidStatus =
        readBuffer<NMMatterStatusGPU>(healthyStatusBuffer, 1u)[0];
    require(invalidStatus.code == NM_STATUS_INVALID_DISPATCH,
        "out-of-range Schur interior map was not rejected");
    // LU descriptors use {interior-value offset, node count, matrix offset,
    // pivot offset}. Feeding that record to extraction must fail closed.
    const std::vector<nm_uint4> factorDescriptorLookalike{{1u,2u,0u,0u}};
    const auto factorDescriptorBuffer = makeBuffer(device, factorDescriptorLookalike);
    const auto factorShapeStatusBuffer = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{healthyStatus});
    const auto twoPatchCountBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{2u});
    dispatch(device, queue, extract, {
        {0u, dispatchBuffer}, {1u, graphPatchBuffer}, {2u, graphNodeBuffer},
        {3u, interiorValueBuffer}, {4u, factorDescriptorBuffer},
        {5u, matrixBuffer}, {6u, invalidOutputBuffer},
        {7u, factorShapeStatusBuffer}, {8u, twoPatchCountBuffer},
        {9u, makeBuffer(device, std::vector<std::uint32_t>{patchNodeCount})},
        {10u, makeBuffer(device, std::vector<std::uint32_t>{matrixFloatCount})},
        {11u, makeBuffer(device, std::vector<std::uint32_t>{invalidSourceCount})},
        {12u, makeBuffer(device, std::vector<std::uint32_t>{invalidOutputFloatCount})},
        {13u, makeBuffer(device,
            std::vector<std::uint32_t>{interiorPatchValueCount})}
    }, invalidSourceCount, 256u);
    require(readBuffer<NMMatterStatusGPU>(factorShapeStatusBuffer, 1u)[0].code ==
                NM_STATUS_INVALID_DISPATCH,
        "factor descriptor was accepted as an extraction descriptor");
    std::cout << "regional_schur_Aii_extraction=exact descriptor_shapes=distinct bounds=guarded result=PASS\\n";
}


void checkRegionalSchurTransformsAndRecovery(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLLibrary> library) {
    // Two 3-DOF nodes: node 0 is an interior block and node 1 is Gamma.
    // Aii is deliberately nonsymmetric and forces a row pivot.
    NMMatterDispatchGPU shape{};
    shape.abiVersion = NM_MATTER_ABI_VERSION;
    shape.environmentCount = 1u;
    shape.femNodeCount = 2u;
    NMFGMRESLayoutGPU layout{};
    // FGMRES stores one float4 row per FEM node; xyz are the three scalar
    // coordinates used by the dense 6x6 oracle below.
    layout.unknownCount = 2u;
    const auto dispatchBuffer = makeBuffer(device,
        std::vector<NMMatterDispatchGPU>{shape});
    const auto layoutBuffer = makeBuffer(device,
        std::vector<NMFGMRESLayoutGPU>{layout});

    const std::array<double, 9> aii{
        0.25, 2.0, 0.0,
        3.0, 1.0, 1.0,
        1.0, 0.0, 4.0};
    const std::array<double, 9> aiGamma{
        1.0, 0.2, 0.0,
        -0.3, 0.1, 0.7,
        -0.2, 0.4, 0.5};
    const std::array<double, 9> aGammaI{
        0.4, 0.1, 0.0,
        -0.2, 0.3, 0.1,
        0.5, 0.0, 0.2};
    const std::array<double, 9> aGammaGamma{
        5.0, 0.1, 0.0,
        0.0, 4.0, 0.2,
        0.3, 0.0, 3.0};
    std::array<double, 36> matrix{};
    for (std::uint32_t row = 0u; row < 3u; ++row) {
        for (std::uint32_t column = 0u; column < 3u; ++column) {
            matrix[row * 6u + column] = aii[row * 3u + column];
            matrix[row * 6u + 3u + column] =
                aiGamma[row * 3u + column];
            matrix[(3u + row) * 6u + column] =
                aGammaI[row * 3u + column];
            matrix[(3u + row) * 6u + 3u + column] =
                aGammaGamma[row * 3u + column];
        }
    }
    const auto multiply6 = [](const std::array<double, 36>& a,
                              const std::array<double, 6>& x) {
        std::array<double, 6> result{};
        for (std::uint32_t row = 0u; row < 6u; ++row)
            for (std::uint32_t column = 0u; column < 6u; ++column)
                result[row] += a[row * 6u + column] * x[column];
        return result;
    };
    const auto solveDense = [](std::vector<double> a,
                               std::vector<double> b,
                               std::uint32_t n) {
        for (std::uint32_t column = 0u; column < n; ++column) {
            std::uint32_t pivot = column;
            for (std::uint32_t row = column + 1u; row < n; ++row)
                if (std::abs(a[row * n + column]) >
                    std::abs(a[pivot * n + column])) pivot = row;
            require(std::abs(a[pivot * n + column]) > 1.0e-12,
                "Schur reference matrix is singular");
            for (std::uint32_t k = column; k < n; ++k)
                std::swap(a[column * n + k], a[pivot * n + k]);
            std::swap(b[column], b[pivot]);
            const double diagonal = a[column * n + column];
            for (std::uint32_t k = column; k < n; ++k)
                a[column * n + k] /= diagonal;
            b[column] /= diagonal;
            for (std::uint32_t row = 0u; row < n; ++row) {
                if (row == column) continue;
                const double scale = a[row * n + column];
                for (std::uint32_t k = column; k < n; ++k)
                    a[row * n + k] -= scale * a[column * n + k];
                b[row] -= scale * b[column];
            }
        }
        return b;
    };
    const auto toFloat4 = [](const std::array<double, 6>& values) {
        return std::vector<nm_float4>{
            nm_float4{static_cast<float>(values[0]),
                static_cast<float>(values[1]), static_cast<float>(values[2]),
                0.0f},
            nm_float4{static_cast<float>(values[3]),
                static_cast<float>(values[4]), static_cast<float>(values[5]),
                0.0f}};
    };
    const auto fromFloat4 = [](const std::vector<nm_float4>& values) {
        return std::array<double, 6>{values[0].x, values[0].y, values[0].z,
            values[1].x, values[1].y, values[1].z};
    };

    const std::uint32_t patchCount = 1u;
    const std::uint32_t patchNodeCount = 1u;
    const std::uint32_t matrixFloatCount = 9u;
    const std::uint32_t pivotCount = 9u;
    const std::uint32_t sourceCount = 1u;
    const std::uint32_t schurMatrixFloatCount = 9u;
    const std::uint32_t interiorValueCount = 1u;
    const auto patchBuffer = makeBuffer(device,
        std::vector<nm_uint4>{{0u, 1u, 0u, 0u}});
    const auto patchNodeBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u});
    const auto interiorValueBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u});
    const auto sourceBuffer = makeBuffer(device,
        std::vector<nm_uint4>{{0u, 0u, 1u, 0u}});
    std::vector<float> graphMatrix(9u);
    for (std::uint32_t i = 0u; i < 9u; ++i)
        graphMatrix[i] = static_cast<float>(aii[i]);
    const auto graphMatrixBuffer = makeBuffer(device, graphMatrix);
    auto extractedBuffer = makeBuffer(device, std::vector<float>(9u, 0.0f));
    auto statusBuffer = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
    const auto extract = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_graph_extract_schur_interior");
    const auto u32 = [&](std::uint32_t value) {
        return makeBuffer(device, std::vector<std::uint32_t>{value});
    };
    dispatch(device, queue, extract, {
        {0u, dispatchBuffer}, {1u, patchBuffer}, {2u, patchNodeBuffer},
        {3u, interiorValueBuffer}, {4u, sourceBuffer},
        {5u, graphMatrixBuffer}, {6u, extractedBuffer}, {7u, statusBuffer},
        {8u, u32(patchCount)}, {9u, u32(patchNodeCount)},
        {10u, u32(matrixFloatCount)}, {11u, u32(sourceCount)},
        {12u, u32(schurMatrixFloatCount)}, {13u, u32(interiorValueCount)}},
        1u, 256u);
    require(readBuffer<NMMatterStatusGPU>(statusBuffer, 1u)[0].code ==
            NM_STATUS_SUCCESS,
        "Schur fixture Aii extraction failed");

    auto pivotsBuffer = makeBuffer(device, std::vector<std::uint32_t>(9u));
    auto factorValidBuffer = makeBuffer(device, std::vector<std::uint32_t>(1u));
    auto originalMatrixBuffer = makeBuffer(device, std::vector<float>(9u));
    auto statesBuffer = makeBuffer(device,
        std::vector<NMFGMRESStateGPU>{NMFGMRESStateGPU{}});
    const auto factor = pipelineWithRelativeInnerForcingDisabled(device, library,
        "numi_matter_metal::nm_fgmres_graph_factor_patches");
    dispatch(device, queue, factor, {
        {0u, dispatchBuffer}, {1u, patchBuffer}, {2u, extractedBuffer},
        {3u, pivotsBuffer}, {4u, factorValidBuffer}, {5u, statusBuffer},
        {6u, u32(patchCount)}, {7u, u32(matrixFloatCount)},
        {8u, u32(pivotCount)}, {9u, originalMatrixBuffer},
        {10u, statesBuffer}, {11u, u32(0u)},
        {12u, u32(0u)}, {13u, u32(1u)},
        {14u, u32(0u)}, {15u, u32(0u)},
        {16u, u32(NM_INVALID_INDEX)}}, 1u, 256u);
    require(readBuffer<std::uint32_t>(factorValidBuffer, 1u)[0] == 1u &&
            readBuffer<NMMatterStatusGPU>(statusBuffer, 1u)[0].code ==
                NM_STATUS_SUCCESS,
        "nonsymmetric Schur Aii factorization failed");

    std::vector<NMFEMNodeStateGPU> candidateValues(2u);
    candidateValues[0].restAndFixed.w = 0.0f;
    candidateValues[1].restAndFixed.w = 0.0f;
    const auto candidateBuffer = makeBuffer(device, candidateValues);
    const auto regionalBasisBuffer = makeBuffer(device,
        std::vector<nm_float4>(2u));
    const auto regionalAPBuffer = makeBuffer(device,
        std::vector<nm_float4>(2u));
    const auto coefficientsBuffer = makeBuffer(device,
        std::vector<float>{0.0f});
    const auto solve = pipelineWithRelativeInnerForcingDisabled(device, library,
        "numi_matter_metal::nm_fgmres_graph_solve_patches");
    const auto solveAii = [&](const std::array<double, 6>& fullRhs) {
        std::vector<nm_float4> basisValues(2u);
        basisValues[0].x = static_cast<float>(fullRhs[0]);
        basisValues[0].y = static_cast<float>(fullRhs[1]);
        basisValues[0].z = static_cast<float>(fullRhs[2]);
        const auto basisBuffer = makeBuffer(device, basisValues);
        auto correctionBuffer = makeBuffer(device,
            std::vector<nm_float4>(1u));
        auto solveStatusBuffer = makeBuffer(device,
            std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
        dispatch(device, queue, solve, {
            {0u, dispatchBuffer}, {1u, patchBuffer}, {2u, patchNodeBuffer},
            {3u, basisBuffer}, {4u, regionalBasisBuffer},
            {5u, regionalAPBuffer}, {6u, coefficientsBuffer},
            {7u, extractedBuffer}, {8u, pivotsBuffer},
            {9u, factorValidBuffer}, {10u, correctionBuffer},
            {11u, solveStatusBuffer}, {12u, u32(patchCount)},
            {13u, u32(patchNodeCount)}, {14u, u32(matrixFloatCount)},
            {15u, u32(pivotCount)}, {16u, u32(0u)}, {17u, statesBuffer},
            {18u, originalMatrixBuffer}, {19u, u32(1u)},
            {20u, candidateBuffer}, {21u, u32(0u)},
            {22u, u32(0u)}, {23u, u32(0u)}}, 1u, 256u);
        require(readBuffer<NMMatterStatusGPU>(solveStatusBuffer, 1u)[0].code ==
                NM_STATUS_SUCCESS,
            "Schur Aii solve failed");
        const auto packed = readBuffer<nm_float4>(correctionBuffer, 1u)[0];
        return std::array<double, 3>{packed.x, packed.y, packed.z};
    };

    const auto interiorMapBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u, NM_INVALID_INDEX});
    const auto transform = pipelineWithRelativeInnerForcingDisabled(
        device, library, "numi_matter_metal::nm_fgmres_schur_transform");
    const auto runTransform = [&](std::uint32_t mode,
                                  const std::array<double, 6>& a,
                                  const std::array<double, 6>& b,
                                  const std::array<double, 3>& packed) {
        const auto aBuffer = makeBuffer(device, toFloat4(a));
        const auto bBuffer = makeBuffer(device, toFloat4(b));
        std::vector<nm_float4> packedValues(1u);
        packedValues[0] = nm_float4{static_cast<float>(packed[0]),
            static_cast<float>(packed[1]), static_cast<float>(packed[2]), 0.0f};
        const auto packedBuffer = makeBuffer(device, packedValues);
        auto outputBuffer = makeBuffer(device,
            std::vector<nm_float4>(2u));
        auto transformStatus = makeBuffer(device,
            std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
        dispatch(device, queue, transform, {
            {0u, dispatchBuffer}, {30u, layoutBuffer},
            {1u, interiorMapBuffer}, {2u, aBuffer}, {3u, bBuffer},
            {4u, packedBuffer}, {5u, outputBuffer}, {6u, u32(mode)},
            {7u, u32(1u)}, {8u, transformStatus},
            {9u, u32(0u)}, {10u, u32(0u)},
            {11u, u32(0u)}, {12u, u32(0u)}}, 1u, 256u);
        require(readBuffer<NMMatterStatusGPU>(transformStatus, 1u)[0].code ==
                NM_STATUS_SUCCESS,
            "Schur vector transform rejected a valid 2-node fixture");
        return fromFloat4(readBuffer<nm_float4>(outputBuffer, 2u));
    };

    const std::array<double, 6> rhs{1.0, -2.0, 0.5, 0.3, 1.2, -0.4};
    const auto yi = solveAii(rhs);
    const std::array<double, 6> yiPacked{yi[0], yi[1], yi[2], 0.0, 0.0, 0.0};
    const std::array<double, 3> zeroPacked{};
    const auto ayi = multiply6(matrix, yiPacked);
    const auto condensedRhs = runTransform(2u, rhs, ayi, zeroPacked);
    for (std::uint32_t i = 0u; i < 3u; ++i)
        require(std::abs(condensedRhs[i]) < 1.0e-6,
            "condensed RHS retained an interior row");
    const auto expectedRhsAction = multiply6(matrix, yiPacked);
    for (std::uint32_t i = 3u; i < 6u; ++i)
        require(std::abs(condensedRhs[i] - (rhs[i] - expectedRhsAction[i])) <=
                2.0e-5 * std::max(1.0, std::abs(rhs[i])),
            "condensed RHS disagrees with b_Gamma - A_Gamma_i A_ii^-1 b_i");

    // Solve the independent dense reference system for a full solution. Its
    // interface entries exercise projection, Schur action, and recovery.
    std::vector<double> denseMatrix(matrix.begin(), matrix.end());
    std::vector<double> denseRhs(rhs.begin(), rhs.end());
    const auto referenceSolutionVector = solveDense(denseMatrix, denseRhs, 6u);
    std::array<double, 6> referenceSolution{};
    for (std::uint32_t i = 0u; i < 6u; ++i)
        referenceSolution[i] = referenceSolutionVector[i];
    const auto referenceGamma = runTransform(0u, referenceSolution, rhs,
                                               zeroPacked);
    for (std::uint32_t i = 0u; i < 3u; ++i)
        require(std::abs(referenceGamma[i]) < 1.0e-7 &&
                std::abs(referenceGamma[i + 3u] - referenceSolution[i + 3u]) <
                    1.0e-7,
            "Gamma projection did not preserve exactly the interface block");
    const auto action = multiply6(matrix, referenceGamma);
    const std::array<double, 6> zeroFull{};
    const auto interiorAction = runTransform(5u, action, zeroFull,
                                               zeroPacked);
    const auto actionInterior = solveAii(interiorAction);
    const auto liftedAction = runTransform(1u, zeroFull, zeroFull,
                                             actionInterior);
    const auto aLiftedAction = multiply6(matrix, liftedAction);
    const auto schurAction = runTransform(3u, action, aLiftedAction,
                                            zeroPacked);
    const auto rhsGamma = runTransform(2u, rhs, ayi, zeroPacked);
    for (std::uint32_t i = 3u; i < 6u; ++i)
        require(std::abs(schurAction[i] - rhsGamma[i]) <=
                2.0e-4 * std::max(1.0, std::abs(rhsGamma[i])),
            "Schur action on reference solution does not equal condensed RHS");

    const auto recoveryRhs = runTransform(5u, rhs, action, zeroPacked);
    const auto recoveredInterior = solveAii(recoveryRhs);
    const auto recovered = runTransform(4u, referenceGamma, zeroFull,
                                         recoveredInterior);
    const auto recoveredAction = multiply6(matrix, recovered);
    for (std::uint32_t i = 0u; i < 6u; ++i)
        require(std::abs(recovered[i] - referenceSolution[i]) <=
                    2.0e-5 * std::max(1.0, std::abs(referenceSolution[i])) &&
                std::abs(recoveredAction[i] - rhs[i]) <=
                    2.0e-4 * std::max(1.0, std::abs(rhs[i])),
            "Schur interior recovery disagrees with the full solve");

    // Aii with nineteen orders of magnitude of row/column variation. The
    // reference matrix is nonsymmetric and well conditioned after diagonal
    // equilibration; without stored Dr/Dc the unchanged relative pivot floor
    // rejects its first pivot. Verify both physical-coordinate solution and
    // residual after the production two-word refinement pass.
    const std::array<double, 9> balancedIllScaled{
        4.0, 1.0, 0.2,
        2.0, 5.0, 1.0,
        0.3, 2.0, 3.0};
    const std::array<double, 3> rowFactors{1.0e-8, 1.0e4, 1.0};
    const std::array<double, 3> columnFactors{1.0e5, 1.0e-3, 1.0e2};
    std::array<double, 9> illScaledA{};
    std::vector<float> illScaledAFloats(9u);
    for (std::uint32_t row = 0u; row < 3u; ++row)
        for (std::uint32_t column = 0u; column < 3u; ++column) {
            const std::uint32_t index = row * 3u + column;
            illScaledA[index] = balancedIllScaled[index] /
                (rowFactors[row] * columnFactors[column]);
            illScaledAFloats[index] = static_cast<float>(illScaledA[index]);
        }
    auto illScaledMatrixBuffer = makeBuffer(device, illScaledAFloats);
    auto illScaledOriginalBuffer = makeBuffer(device, std::vector<float>(9u));
    auto illScaledPivotsBuffer = makeBuffer(device,
        std::vector<std::uint32_t>(9u));
    auto illScaledValidBuffer = makeBuffer(device, std::vector<std::uint32_t>(1u));
    auto illScaledFactorStatus = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
    dispatch(device, queue, factor, {
        {0u, dispatchBuffer}, {1u, patchBuffer}, {2u, illScaledMatrixBuffer},
        {3u, illScaledPivotsBuffer}, {4u, illScaledValidBuffer},
        {5u, illScaledFactorStatus}, {6u, u32(patchCount)},
        {7u, u32(matrixFloatCount)}, {8u, u32(pivotCount)},
        {9u, illScaledOriginalBuffer}, {10u, statesBuffer},
        {11u, u32(0u)}, {12u, u32(0u)}, {13u, u32(1u)},
        {14u, u32(0u)}, {15u, u32(0u)},
        {16u, u32(NM_INVALID_INDEX)}}, 1u, 256u);
    require(readBuffer<std::uint32_t>(illScaledValidBuffer, 1u)[0] == 1u &&
            readBuffer<NMMatterStatusGPU>(illScaledFactorStatus, 1u)[0].code ==
                NM_STATUS_SUCCESS,
        "row/column equilibration rejected nonsymmetric ill-scaled Aii");
    const std::array<double, 3> knownX{0.75, -1.25, 2.0};
    std::array<double, 3> illScaledRhs{};
    std::array<float, 3> illScaledRhsFloat{};
    std::vector<nm_float4> illScaledBasis(2u);
    for (std::uint32_t row = 0u; row < 3u; ++row) {
        for (std::uint32_t column = 0u; column < 3u; ++column)
            illScaledRhs[row] += illScaledAFloats[row * 3u + column] *
                knownX[column];
        illScaledRhsFloat[row] = static_cast<float>(illScaledRhs[row]);
    }
    illScaledBasis[0] = nm_float4{illScaledRhsFloat[0],
        illScaledRhsFloat[1], illScaledRhsFloat[2], 0.0f};
    auto illScaledBasisBuffer = makeBuffer(device, illScaledBasis);
    auto illScaledCorrectionBuffer = makeBuffer(device,
        std::vector<nm_float4>(1u));
    auto illScaledSolveStatus = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
    dispatch(device, queue, solve, {
        {0u, dispatchBuffer}, {1u, patchBuffer}, {2u, patchNodeBuffer},
        {3u, illScaledBasisBuffer}, {4u, regionalBasisBuffer},
        {5u, regionalAPBuffer}, {6u, coefficientsBuffer},
        {7u, illScaledMatrixBuffer}, {8u, illScaledPivotsBuffer},
        {9u, illScaledValidBuffer}, {10u, illScaledCorrectionBuffer},
        {11u, illScaledSolveStatus}, {12u, u32(patchCount)},
        {13u, u32(patchNodeCount)}, {14u, u32(matrixFloatCount)},
        {15u, u32(pivotCount)}, {16u, u32(0u)}, {17u, statesBuffer},
        {18u, illScaledOriginalBuffer}, {19u, u32(1u)},
        {20u, candidateBuffer}, {21u, u32(0u)},
        {22u, u32(0u)}, {23u, u32(0u)}}, 1u, 256u);
    require(readBuffer<NMMatterStatusGPU>(illScaledSolveStatus, 1u)[0].code ==
                NM_STATUS_SUCCESS,
        "row/column equilibrated Aii solve failed");
    const auto illScaledResultPacked =
        readBuffer<nm_float4>(illScaledCorrectionBuffer, 1u)[0];
    const std::array<double, 3> illScaledResult{
        illScaledResultPacked.x, illScaledResultPacked.y,
        illScaledResultPacked.z};
    const std::vector<double> illScaledDense(
        illScaledAFloats.begin(), illScaledAFloats.end());
    const std::vector<double> illScaledReferenceVector = solveDense(
        illScaledDense,
        std::vector<double>(illScaledRhsFloat.begin(),
                            illScaledRhsFloat.end()), 3u);
    double maximumSolutionError = 0.0;
    double maximumRelativeResidual = 0.0;
    for (std::uint32_t row = 0u; row < 3u; ++row) {
        double product = 0.0;
        double productMagnitude = 0.0;
        for (std::uint32_t column = 0u; column < 3u; ++column) {
            const double term = illScaledAFloats[row * 3u + column] *
                illScaledResult[column];
            product += term;
            productMagnitude += std::abs(term);
        }
        maximumSolutionError = std::max(maximumSolutionError,
            std::abs(illScaledResult[row] - illScaledReferenceVector[row]) /
                std::max(1.0, std::abs(illScaledReferenceVector[row])));
        maximumRelativeResidual = std::max(maximumRelativeResidual,
            std::abs(product - illScaledRhsFloat[row]) /
                std::max<double>({1.0e-30, std::abs(illScaledRhsFloat[row]),
                          productMagnitude}));
    }
    require(maximumSolutionError <= 2.0e-4 &&
            maximumRelativeResidual <= 2.0e-5,
        "row/column equilibration failed the ill-scaled original-Aii residual");

    std::cout << "regional_schur_Aii_pivot=passed gamma_rhs=passed "
                 "schur_action=passed recovery=passed "
                 "equilibrated_ill_scaled=passed result=PASS\n";
}

void checkActiveContactSchurEligibility(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLLibrary> library) {
    NMMatterDispatchGPU shape{};
    shape.abiVersion = NM_MATTER_ABI_VERSION;
    shape.environmentCount = 1u;
    shape.femNodeCount = 6u;
    shape.deformableContactCapacity = 3u;
    NMFGMRESLayoutGPU layout{};
    layout.unknownCount = 6u;
    const auto dispatchBuffer = makeBuffer(device,
        std::vector<NMMatterDispatchGPU>{shape});
    const auto layoutBuffer = makeBuffer(device,
        std::vector<NMFGMRESLayoutGPU>{layout});
    const auto u32 = [&](std::uint32_t value) {
        return makeBuffer(device, std::vector<std::uint32_t>{value});
    };

    auto reset = pipelineWithRelativeInnerForcingDisabled(device, library,
        "numi_matter_metal::nm_fgmres_schur_reset_patch_eligibility");
    auto classify = pipelineWithRelativeInnerForcingDisabled(device, library,
        "numi_matter_metal::nm_fgmres_schur_classify_active_contacts");
    auto transform = pipelineWithRelativeInnerForcingDisabled(device, library,
        "numi_matter_metal::nm_fgmres_schur_transform");
    auto factor = pipelineWithRelativeInnerForcingDisabled(device, library,
        "numi_matter_metal::nm_fgmres_graph_factor_patches");
    auto solve = pipelineWithRelativeInnerForcingDisabled(device, library,
        "numi_matter_metal::nm_fgmres_graph_solve_patches");

    std::vector<NMDeformableContactGPU> contacts(3u);
    contacts[0].identity.w = NM_CONTACT_VALID;
    contacts[0].nodes = nm_uint4{0u, NM_INVALID_INDEX,
        NM_INVALID_INDEX, NM_INVALID_INDEX};
    contacts[0].extraNodes = nm_uint4{NM_INVALID_INDEX, 1u, 0u, 0u};
    // Valid but absent from the compact active list in phases A and B.
    contacts[1].identity.w = NM_CONTACT_VALID;
    contacts[1].nodes = nm_uint4{1u, NM_INVALID_INDEX,
        NM_INVALID_INDEX, NM_INVALID_INDEX};
    contacts[1].extraNodes = nm_uint4{NM_INVALID_INDEX,
        NM_INVALID_INDEX, 0u, 0u};
    // Active surface-only contact; its endpoint has no interior-patch map.
    contacts[2].identity.w = NM_CONTACT_VALID;
    contacts[2].nodes = nm_uint4{5u, NM_INVALID_INDEX,
        NM_INVALID_INDEX, NM_INVALID_INDEX};
    contacts[2].extraNodes = nm_uint4{NM_INVALID_INDEX,
        NM_INVALID_INDEX, 0u, 0u};
    const auto contactsBuffer = makeBuffer(device, contacts);
    const auto contactFailureWitness = makeBuffer(device,
        std::vector<std::uint32_t>{NM_INVALID_INDEX});
    const auto patchByNodeBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u, 1u,
            NM_INVALID_INDEX, NM_INVALID_INDEX, NM_INVALID_INDEX,
            NM_INVALID_INDEX});
    auto eligibilityBuffer = makeBuffer(device,
        std::vector<std::uint32_t>(2u, 0u));

    const auto classifyActive = [&](const std::vector<std::uint32_t>& indices,
                                    const std::uint32_t activeCount) {
        require(indices.size() == 3u,
            "active Schur fixture index capacity mismatch");
        dispatch(device, queue, reset, {
            {0u, dispatchBuffer}, {1u, eligibilityBuffer}, {2u, u32(2u)},
            {3u, contactFailureWitness}}, 2u, 32u);
        auto indicesBuffer = makeBuffer(device, indices);
        auto countsBuffer = makeBuffer(device,
            std::vector<std::uint32_t>{activeCount});
        auto statusBuffer = makeBuffer(device,
            std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
        dispatch(device, queue, classify, {
            {0u, dispatchBuffer}, {1u, contactsBuffer},
            {2u, indicesBuffer}, {3u, countsBuffer},
            {4u, patchByNodeBuffer}, {5u, eligibilityBuffer},
            {6u, statusBuffer}, {7u, u32(2u)}, {8u, u32(3u)},
            {9u, contactFailureWitness}}, 3u, 32u);
        require(readBuffer<NMMatterStatusGPU>(statusBuffer, 1u)[0].code ==
                NM_STATUS_SUCCESS,
            "active Schur contact classifier rejected a valid fixture");
        require(readBuffer<std::uint32_t>(contactFailureWitness, 1u)[0] ==
                NM_INVALID_INDEX,
            "valid active Schur contacts wrote a malformed-contact witness");
        return readBuffer<std::uint32_t>(eligibilityBuffer, 2u);
    };

    // Corner 0 touches core 0; corner 5 touches core 1. Force/impulse are
    // exactly zero, so classification must follow active constraints only.
    require(classifyActive({0u, 2u, 0u}, 2u) ==
            std::vector<std::uint32_t>{0u, 0u},
        "active zero-force contact did not promote every touched core");
    require(classifyActive({2u, 0u, 0u}, 1u) ==
            std::vector<std::uint32_t>{1u, 1u},
        "inactive or surface-only contact incorrectly promoted an interior core");
    require(classifyActive({1u, 0u, 0u}, 1u) ==
            std::vector<std::uint32_t>{1u, 0u},
        "active contact failed to promote the core it touches");
    require(classifyActive({NM_INVALID_INDEX, NM_INVALID_INDEX,
                            NM_INVALID_INDEX}, 0u) ==
            std::vector<std::uint32_t>{1u, 1u},
        "empty active set did not leave every interior core eligible");

    // Malformed compact active lists and malformed nonzero contact corners
    // are reported after classification by the factor dispatch. The witness
    // is an explicit per-environment atomic minimum, so classifier lanes never
    // race the status record and the chosen slot/corner is deterministic.
    const auto finalizeMalformedContact = [&](
        const std::vector<NMDeformableContactGPU>& fixtureContacts,
        const std::vector<std::uint32_t>& fixtureActiveIndices,
        const std::uint32_t activeCount,
        const std::vector<std::uint32_t>& fixturePatchByNode) {
        auto localContacts = makeBuffer(device, fixtureContacts);
        auto localEligibility = makeBuffer(device,
            std::vector<std::uint32_t>(2u, 0u));
        auto localWitness = makeBuffer(device,
            std::vector<std::uint32_t>{NM_INVALID_INDEX});
        auto localStatus = makeBuffer(device,
            std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
        auto activeIndices = makeBuffer(device, fixtureActiveIndices);
        auto activeCounts = makeBuffer(device,
            std::vector<std::uint32_t>{activeCount});
        auto patchByNode = makeBuffer(device, fixturePatchByNode);
        dispatch(device, queue, reset, {
            {0u, dispatchBuffer}, {1u, localEligibility}, {2u, u32(2u)},
            {3u, localWitness}}, 2u, 32u);
        dispatch(device, queue, classify, {
            {0u, dispatchBuffer}, {1u, localContacts},
            {2u, activeIndices}, {3u, activeCounts},
            {4u, patchByNode}, {5u, localEligibility},
            {6u, localStatus}, {7u, u32(2u)}, {8u, u32(3u)},
            {9u, localWitness}}, 3u, 32u);
        require(readBuffer<NMMatterStatusGPU>(localStatus, 1u)[0].code ==
                    NM_STATUS_SUCCESS,
            "malformed contact classification wrote status concurrently");
        const auto witness = readBuffer<std::uint32_t>(localWitness, 1u)[0];
        require(witness != NM_INVALID_INDEX,
            "malformed active contact did not preserve a failure witness");

        auto localPatches = makeBuffer(device,
            std::vector<nm_uint4>{{0u, 1u, 0u, 0u},
                                  {1u, 1u, 9u, 9u}});
        auto localMatrices = makeBuffer(device, std::vector<float>(18u, 1.0f));
        auto localPivots = makeBuffer(device, std::vector<std::uint32_t>(18u));
        auto localFactorValid = makeBuffer(device, std::vector<std::uint32_t>(2u, 1u));
        auto localOriginal = makeBuffer(device, std::vector<float>(18u));
        auto localStates = makeBuffer(device,
            std::vector<NMFGMRESStateGPU>{NMFGMRESStateGPU{}});
        dispatch(device, queue, factor, {
            {0u, dispatchBuffer}, {1u, localPatches}, {2u, localMatrices},
            {3u, localPivots}, {4u, localFactorValid}, {5u, localStatus},
            {6u, u32(2u)}, {7u, u32(18u)}, {8u, u32(18u)},
            {9u, localOriginal}, {10u, localStates}, {11u, u32(0u)},
            {12u, makeBuffer(device, std::vector<std::uint32_t>(2u, 0u))},
            {13u, u32(0u)}, {14u, localEligibility}, {15u, u32(1u)},
            {16u, localWitness}}, 2u, 256u);
        const auto finalStatus =
            readBuffer<NMMatterStatusGPU>(localStatus, 1u)[0];
        require(finalStatus.code == NM_STATUS_INVALID_DISPATCH &&
                (readBuffer<std::uint32_t>(localFactorValid, 2u) ==
                    std::vector<std::uint32_t>{0u, 0u}),
            "Schur factor did not finalize malformed contact as failure");
        return finalStatus;
    };
    {
        const auto result = finalizeMalformedContact(
            contacts, {2u, 3u, 0u}, 2u,
            {0u, 1u, NM_INVALID_INDEX, NM_INVALID_INDEX,
             NM_INVALID_INDEX, NM_INVALID_INDEX});
        require(result.failingIndex == 1u &&
                result.diagnostics.x == 1.0f &&
                result.diagnostics.y == 14.0f,
            "invalid compact contact at nonzero active slot lost its witness");
    }
    {
        auto malformed = contacts;
        malformed[0].extraNodes.y = 99u; // local corner 5, out of FEM range.
        const auto result = finalizeMalformedContact(
            malformed, {0u, 0u, 0u}, 1u,
            {0u, 1u, NM_INVALID_INDEX, NM_INVALID_INDEX,
             NM_INVALID_INDEX, NM_INVALID_INDEX});
        require(result.failingIndex == 0u &&
                result.diagnostics.x == 0.0f &&
                result.diagnostics.y == 5.0f,
            "invalid nonzero FEM contact corner lost its witness");
    }
    {
        const auto result = finalizeMalformedContact(
            contacts, {0u, 0u, 0u}, 1u,
            {0u, 3u, NM_INVALID_INDEX, NM_INVALID_INDEX,
             NM_INVALID_INDEX, NM_INVALID_INDEX});
        require(result.failingIndex == 0u &&
                result.diagnostics.x == 0.0f &&
                result.diagnostics.y == 13.0f,
            "invalid patch map at nonzero contact corner lost its witness");
    }
    {
        const auto result = finalizeMalformedContact(
            contacts, {2u, 0u, 0u}, 4u,
            {0u, 1u, NM_INVALID_INDEX, NM_INVALID_INDEX,
             NM_INVALID_INDEX, NM_INVALID_INDEX});
        require(result.failingIndex == 0u &&
                result.diagnostics.x == 0.0f &&
                result.diagnostics.y == 15.0f,
            "active contact count overflow lost its deterministic witness");
    }
    {
        auto malformed = contacts;
        malformed[0].identity.w = 0u;
        const auto result = finalizeMalformedContact(
            malformed, {0u, 3u, 0u}, 2u,
            {0u, 1u, NM_INVALID_INDEX, NM_INVALID_INDEX,
             NM_INVALID_INDEX, NM_INVALID_INDEX});
        require(result.failingIndex == 0u &&
                result.diagnostics.x == 0.0f &&
                result.diagnostics.y == 14.0f,
            "multiple malformed contacts did not choose the lowest active slot");
    }

    // Feed exactly that eligibility into local factor/solve and the full-vector
    // transform. Promoted core 1 is an interface row; core 0 remains local.
    eligibilityBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{1u, 0u});
    const auto patchesBuffer = makeBuffer(device,
        std::vector<nm_uint4>{{0u, 1u, 0u, 0u},
                              {1u, 1u, 9u, 9u}});
    const auto patchNodesBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u, 1u});
    std::vector<float> matrixValues(18u, 0.0f);
    for (std::uint32_t patch = 0u; patch < 2u; ++patch)
        for (std::uint32_t axis = 0u; axis < 3u; ++axis)
            matrixValues[patch * 9u + axis * 3u + axis] = 1.0f;
    auto matrixBuffer = makeBuffer(device, matrixValues);
    auto pivotsBuffer = makeBuffer(device, std::vector<std::uint32_t>(18u));
    auto factorValidBuffer = makeBuffer(device, std::vector<std::uint32_t>(2u));
    auto originalMatrixBuffer = makeBuffer(device, std::vector<float>(18u));
    auto statesBuffer = makeBuffer(device,
        std::vector<NMFGMRESStateGPU>{NMFGMRESStateGPU{}});
    auto factorStatusBuffer = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
    dispatch(device, queue, factor, {
        {0u, dispatchBuffer}, {1u, patchesBuffer}, {2u, matrixBuffer},
        {3u, pivotsBuffer}, {4u, factorValidBuffer}, {5u, factorStatusBuffer},
        {6u, u32(2u)}, {7u, u32(18u)}, {8u, u32(18u)},
        {9u, originalMatrixBuffer}, {10u, statesBuffer}, {11u, u32(0u)},
        {12u, makeBuffer(device, std::vector<std::uint32_t>(2u, 0u))},
        {13u, u32(0u)}, {14u, eligibilityBuffer}, {15u, u32(1u)},
        {16u, u32(NM_INVALID_INDEX)}}, 2u, 256u);
    require(readBuffer<std::uint32_t>(factorValidBuffer, 2u) ==
                std::vector<std::uint32_t>{1u, 0u} &&
            readBuffer<NMMatterStatusGPU>(factorStatusBuffer, 1u)[0].code ==
                NM_STATUS_SUCCESS,
        "ineligible active-contact core was factored instead of promoted");

    std::vector<nm_float4> rhs(6u);
    rhs[0].x = 3.0f;
    rhs[1].x = 7.0f;
    auto rhsBuffer = makeBuffer(device, rhs);
    auto correctionBuffer = makeBuffer(device, std::vector<nm_float4>(2u));
    std::vector<NMFEMNodeStateGPU> candidateValues(6u);
    for (auto& value : candidateValues) value.restAndFixed.w = 0.0f;
    auto candidateBuffer = makeBuffer(device, candidateValues);
    auto solveStatusBuffer = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
    auto zeros = makeBuffer(device, std::vector<nm_float4>(6u));
    auto zeroCoefficients = makeBuffer(device, std::vector<float>{0.0f});
    dispatch(device, queue, solve, {
        {0u, dispatchBuffer}, {1u, patchesBuffer}, {2u, patchNodesBuffer},
        {3u, rhsBuffer}, {4u, zeros}, {5u, zeros}, {6u, zeroCoefficients},
        {7u, matrixBuffer}, {8u, pivotsBuffer}, {9u, factorValidBuffer},
        {10u, correctionBuffer}, {11u, solveStatusBuffer},
        {12u, u32(2u)}, {13u, u32(2u)}, {14u, u32(18u)}, {15u, u32(18u)},
        {16u, u32(0u)}, {17u, statesBuffer}, {18u, originalMatrixBuffer},
        {19u, u32(0u)}, {20u, candidateBuffer}, {21u, u32(0u)},
        {22u, eligibilityBuffer}, {23u, u32(1u)}}, 2u, 256u);
    require(readBuffer<NMMatterStatusGPU>(solveStatusBuffer, 1u)[0].code ==
                NM_STATUS_SUCCESS,
        "eligible-core solve reported failure");
    const auto corrections = readBuffer<nm_float4>(correctionBuffer, 2u);
    require(corrections[0].x == 3.0f && corrections[1].x == 0.0f,
        "Schur solve failed to zero the promoted core correction");

    const auto interiorMap = makeBuffer(device,
        std::vector<std::uint32_t>{0u, 1u, NM_INVALID_INDEX,
            NM_INVALID_INDEX, NM_INVALID_INDEX, NM_INVALID_INDEX});
    const auto aBuffer = makeBuffer(device, std::vector<nm_float4>{
        nm_float4{10.0f, 0.0f, 0.0f, 0.0f},
        nm_float4{20.0f, 0.0f, 0.0f, 0.0f},
        nm_float4{30.0f, 0.0f, 0.0f, 0.0f},
        nm_float4{40.0f, 0.0f, 0.0f, 0.0f},
        nm_float4{}, nm_float4{}});
    const auto zeroInput = makeBuffer(device, std::vector<nm_float4>(6u));
    auto transformOutput = makeBuffer(device, std::vector<nm_float4>(6u));
    auto transformStatus = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
    dispatch(device, queue, transform, {
        {0u, dispatchBuffer}, {30u, layoutBuffer}, {1u, interiorMap},
        {2u, aBuffer}, {3u, zeroInput}, {4u, correctionBuffer},
        {5u, transformOutput}, {6u, u32(2u)}, {7u, u32(2u)},
        {8u, transformStatus}, {9u, patchByNodeBuffer},
        {10u, eligibilityBuffer}, {11u, u32(2u)}, {12u, u32(1u)}},
        1u, 32u);
    const auto transformed = readBuffer<nm_float4>(transformOutput, 6u);
    require(readBuffer<NMMatterStatusGPU>(transformStatus, 1u)[0].code ==
                NM_STATUS_SUCCESS &&
            transformed[0].x == 0.0f && transformed[1].x == 20.0f &&
            transformed[2].x == 30.0f,
        "promoted core was not treated as an exact Schur interface row");
    std::cout << "schur_active_contact_cores=true zero_force=true "
                 "all_six_corners=true active_list_only=true "
                 "factor_solve_transform=true result=PASS\n";
}

void checkDueAwareSchurPatchAssembly(
    id<MTLDevice> device, id<MTLCommandQueue> queue,
    id<MTLLibrary> library) {
    // The production row owner is due every tick; the second FEM object's
    // period is two ticks, so it is inactive at microtick 1. No tets are
    // needed here: the exact row action is its mass term for the due object
    // and zero for the dormant object. The Schur factor uses identity padding
    // for that zero row, with a zero RHS, to keep a bounded solve slot.
    NMMatterDispatchGPU shape{};
    shape.abiVersion = NM_MATTER_ABI_VERSION;
    shape.environmentCount = 1u;
    shape.objectCount = 2u;
    shape.femNodeCount = 3u;
    shape.tetrahedronCount = 0u;
    NMFGMRESLayoutGPU layout{};
    layout.unknownCount = 3u;

    std::vector<NMContinuumObjectGPU> objects(2u);
    for (auto& object : objects) {
        object.representation = NM_REPRESENTATION_FEM;
        object.flags = NM_OBJECT_ACTIVE;
        object.solver.y = 1u;
    }
    std::vector<NMSchedulerStateGPU> schedulers(2u);
    schedulers[0].activeExponent = 1u;
    schedulers[1].activeExponent = 0u;
    schedulers[0].numerical.w = schedulers[1].numerical.w = 1.0f;
    std::vector<NMAdaptiveStateGPU> adaptive(2u);
    adaptive[0].activeRepresentation = NM_REPRESENTATION_FEM;
    adaptive[1].activeRepresentation = NM_REPRESENTATION_FEM;
    NMMicrostepGPU microstep{};
    microstep.microtick = 1u;

    std::vector<NMFEMNodeStateGPU> candidate(3u);
    const std::array<float, 3> masses{2.0f, 2.5f, 4.0f};
    for (std::uint32_t node = 0u; node < 3u; ++node) {
        candidate[node].positionAndMass.w = masses[node];
        candidate[node].velocityAndInverseMass.w = 1.0f / masses[node];
    }
    const std::vector<NMIncidenceRangeGPU> ranges{
        {0u, 0u, 0u, 0u}, {0u, 0u, 1u, 0u},
        {0u, 0u, 0u, 0u}};
    const std::vector<std::uint32_t> coverage{1u, 1u, 1u};
    const std::uint32_t rowCount = 6u;
    const std::uint32_t patchCount = 2u;
    const std::uint32_t matrixFloatCount = 18u;
    const std::vector<nm_uint4> patches{
        nm_uint4{0u, 1u, 0u, 0u}, nm_uint4{1u, 1u, 9u, 9u}};
    const std::vector<std::uint32_t> patchNodes{0u, 1u};
    const std::vector<nm_uint4> rows{
        nm_uint4{0u, 0u, 0u, 0u}, nm_uint4{0u, 0u, 1u, 0u},
        nm_uint4{0u, 0u, 2u, 0u}, nm_uint4{1u, 0u, 0u, 1u},
        nm_uint4{1u, 0u, 1u, 1u}, nm_uint4{1u, 0u, 2u, 1u}};
    const auto dispatchBuffer = makeBuffer(device,
        std::vector<NMMatterDispatchGPU>{shape});
    const auto objectsBuffer = makeBuffer(device, objects);
    const auto candidateBuffer = makeBuffer(device, candidate);
    const auto tetBuffer = makeBuffer(device,
        std::vector<NMTetrahedronGPU>{NMTetrahedronGPU{}});
    const auto incidenceBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u});
    const auto rangesBuffer = makeBuffer(device, ranges);
    const auto schedulerBuffer = makeBuffer(device, schedulers);
    const auto adaptiveBuffer = makeBuffer(device, adaptive);
    const auto microstepBuffer = makeBuffer(device,
        std::vector<NMMicrostepGPU>{microstep});
    const auto materialBuffer = makeBuffer(device,
        std::vector<NMMaterialGPU>{NMMaterialGPU{}});
    const auto programBuffer = makeBuffer(device,
        std::vector<NMScalarProgramGPU>{NMScalarProgramGPU{}});
    const auto instructionBuffer = makeBuffer(device,
        std::vector<NMExpressionInstructionGPU>{NMExpressionInstructionGPU{}});
    const auto parameterBuffer = makeBuffer(device, std::vector<float>{0.0f});
    const auto stateBuffer = makeBuffer(device, std::vector<float>{0.0f});
    const auto tangentBuffer = makeBuffer(device, std::vector<float>(81u, 0.0f));
    const auto tangentValidBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u});
    const auto patchesBuffer = makeBuffer(device, patches);
    const auto patchNodesBuffer = makeBuffer(device, patchNodes);
    const auto rowsBuffer = makeBuffer(device, rows);
    const auto coverageBuffer = makeBuffer(device, coverage);
    const auto statesBuffer = makeBuffer(device,
        std::vector<NMFGMRESStateGPU>{NMFGMRESStateGPU{}});
    const auto priorValidBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u, 0u});
    const auto cacheModeBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u});
    const auto graphBuild = pipelineWithRelativeInnerForcingDisabled(
        device, library,
        "numi_matter_metal::nm_fgmres_graph_build_patch_matrices");
    const auto build = [&](id<MTLBuffer> matrix, std::uint32_t dueAware) {
        auto status = makeBuffer(device,
            std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
        dispatch(device, queue, graphBuild, {
            {0u, dispatchBuffer}, {1u, objectsBuffer},
            {2u, candidateBuffer}, {3u, tetBuffer}, {4u, incidenceBuffer},
            {5u, rangesBuffer}, {6u, schedulerBuffer}, {7u, materialBuffer},
            {8u, programBuffer}, {9u, instructionBuffer},
            {10u, parameterBuffer}, {11u, stateBuffer},
            {12u, tangentBuffer}, {13u, tangentValidBuffer},
            {14u, cacheModeBuffer}, {15u, patchesBuffer},
            {16u, patchNodesBuffer}, {17u, rowsBuffer},
            {18u, coverageBuffer}, {19u, matrix}, {20u, status},
            {21u, makeBuffer(device, std::vector<std::uint32_t>{rowCount})},
            {22u, makeBuffer(device, std::vector<std::uint32_t>{patchCount})},
            {23u, makeBuffer(device,
                std::vector<std::uint32_t>{matrixFloatCount})},
            {24u, statesBuffer},
            {25u, makeBuffer(device, std::vector<std::uint32_t>{0u})},
            {26u, priorValidBuffer}, {27u, adaptiveBuffer},
            {28u, microstepBuffer},
            {29u, makeBuffer(device,
                std::vector<std::uint32_t>{dueAware})}}, 1u, rowCount);
        require(readBuffer<NMMatterStatusGPU>(status, 1u)[0].code ==
                NM_STATUS_SUCCESS,
            "due-aware Schur graph matrix assembly failed");
    };

    auto legacyMatrix = makeBuffer(device, std::vector<float>(matrixFloatCount));
    build(legacyMatrix, 0u);
    const auto legacy = readBuffer<float>(legacyMatrix, matrixFloatCount);
    require(legacy[0] == 2.0f && legacy[4] == 2.0f && legacy[8] == 2.0f &&
            legacy[9] == 2.5f && legacy[13] == 2.5f && legacy[17] == 2.5f,
        "Schur graph default matrix path changed without due-aware opt-in");

    auto dueMatrix = makeBuffer(device, std::vector<float>(matrixFloatCount));
    build(dueMatrix, 1u);
    const auto assembled = readBuffer<float>(dueMatrix, matrixFloatCount);
    require(assembled[0] == 2.0f && assembled[4] == 2.0f &&
            assembled[8] == 2.0f && assembled[9] == 1.0f &&
            assembled[13] == 1.0f && assembled[17] == 1.0f,
        "due-aware Aii must retain active mass and identity-pad inactive rows");

    // Factor both local blocks, solve the condensed RHS, then recover the
    // full solution. The full system is nonsymmetric: the active interface
    // row receives a coupling from the interior, while the inactive object's
    // entire row and column stay zero. That dormant coordinate has zero RHS,
    // so the identity-padded Aii solve selects the zero representative.
    const auto pivots = makeBuffer(device, std::vector<std::uint32_t>(18u));
    const auto factorValid = makeBuffer(device, std::vector<std::uint32_t>(2u));
    const auto originalMatrices = makeBuffer(device, std::vector<float>(18u));
    const auto factorStatus = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
    const auto factor = pipelineWithRelativeInnerForcingDisabled(device, library,
        "numi_matter_metal::nm_fgmres_graph_factor_patches");
    dispatch(device, queue, factor, {
        {0u, dispatchBuffer}, {1u, patchesBuffer}, {2u, dueMatrix},
        {3u, pivots}, {4u, factorValid}, {5u, factorStatus},
        {6u, makeBuffer(device, std::vector<std::uint32_t>{patchCount})},
        {7u, makeBuffer(device,
            std::vector<std::uint32_t>{matrixFloatCount})},
        {8u, makeBuffer(device, std::vector<std::uint32_t>{18u})},
        {9u, originalMatrices}, {10u, statesBuffer},
        {11u, makeBuffer(device, std::vector<std::uint32_t>{0u})},
        {12u, priorValidBuffer},
        {13u, makeBuffer(device, std::vector<std::uint32_t>{0u})},
        {14u, makeBuffer(device, std::vector<std::uint32_t>{0u})},
        {15u, makeBuffer(device, std::vector<std::uint32_t>{0u})},
        {16u, makeBuffer(device,
            std::vector<std::uint32_t>{NM_INVALID_INDEX})}},
        patchCount, 256u);
    require(readBuffer<std::uint32_t>(factorValid, 2u) ==
                std::vector<std::uint32_t>{1u, 1u} &&
            readBuffer<NMMatterStatusGPU>(factorStatus, 1u)[0].code ==
                NM_STATUS_SUCCESS,
        "due-aware active/inactive Aii blocks did not factor");

    const std::array<nm_float4, 3> rhs{
        nm_float4{6.0f, -4.0f, 2.0f, 0.0f},
        nm_float4{0.0f, 0.0f, 0.0f, 0.0f},
        nm_float4{8.0f, 4.0f, -4.0f, 0.0f}};
    auto rhsBuffer = makeBuffer(device,
        std::vector<nm_float4>(rhs.begin(), rhs.end()));
    auto regionalBasis = makeBuffer(device, std::vector<nm_float4>(3u));
    auto regionalAP = makeBuffer(device, std::vector<nm_float4>(3u));
    auto coefficients = makeBuffer(device, std::vector<float>{0.0f});
    auto patchCorrections = makeBuffer(device, std::vector<nm_float4>(2u));
    auto solveStatus = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
    const auto solve = pipelineWithRelativeInnerForcingDisabled(device, library,
        "numi_matter_metal::nm_fgmres_graph_solve_patches");
    dispatch(device, queue, solve, {
        {0u, dispatchBuffer}, {1u, patchesBuffer}, {2u, patchNodesBuffer},
        {3u, rhsBuffer}, {4u, regionalBasis}, {5u, regionalAP},
        {6u, coefficients}, {7u, dueMatrix}, {8u, pivots},
        {9u, factorValid}, {10u, patchCorrections}, {11u, solveStatus},
        {12u, makeBuffer(device, std::vector<std::uint32_t>{patchCount})},
        {13u, makeBuffer(device, std::vector<std::uint32_t>{2u})},
        {14u, makeBuffer(device,
            std::vector<std::uint32_t>{matrixFloatCount})},
        {15u, makeBuffer(device, std::vector<std::uint32_t>{18u})},
        {16u, makeBuffer(device, std::vector<std::uint32_t>{0u})},
        {17u, statesBuffer}, {18u, originalMatrices},
        {19u, makeBuffer(device, std::vector<std::uint32_t>{1u})},
        {20u, candidateBuffer},
        {21u, makeBuffer(device, std::vector<std::uint32_t>{0u})},
        {22u, makeBuffer(device, std::vector<std::uint32_t>{0u})},
        {23u, makeBuffer(device, std::vector<std::uint32_t>{0u})}},
        patchCount, 256u);
    require(readBuffer<NMMatterStatusGPU>(solveStatus, 1u)[0].code ==
            NM_STATUS_SUCCESS,
        "due-aware Schur local solve failed");
    const auto packed = readBuffer<nm_float4>(patchCorrections, 2u);
    const std::array<float, 3> expectedInterior{3.0f, -2.0f, 1.0f};
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        const float actual = axis == 0u ? packed[0].x :
            (axis == 1u ? packed[0].y : packed[0].z);
        require(std::abs(actual - expectedInterior[axis]) <= 1.0e-6f,
            "due FEM interior solve disagrees with its active mass row");
    }
    require(packed[1].x == 0.0f && packed[1].y == 0.0f &&
            packed[1].z == 0.0f,
        "inactive-object identity padding must not invent a correction");

    // Compute b_Gamma - A_Gamma_i A_ii^-1 b_i using the fixture's full
    // matrix, then recover the interface and verify every full row.
    auto interiorMap = makeBuffer(device,
        std::vector<std::uint32_t>{0u, 1u, NM_INVALID_INDEX});
    const auto transform = pipelineWithRelativeInnerForcingDisabled(
        device, library, "numi_matter_metal::nm_fgmres_schur_transform");
    std::array<nm_float4, 3> aTimesYi{
        nm_float4{12.0f, -8.0f, 4.0f, 0.0f},
        nm_float4{0.0f, 0.0f, 0.0f, 0.0f},
        nm_float4{1.5f, -1.0f, 0.5f, 0.0f}};
    auto aTimesYiBuffer = makeBuffer(device,
        std::vector<nm_float4>(aTimesYi.begin(), aTimesYi.end()));
    auto transformStatus = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
    const auto transformArgs = [&](std::uint32_t mode,
                                   id<MTLBuffer> inputA,
                                   id<MTLBuffer> inputB,
                                   id<MTLBuffer> output) {
        dispatch(device, queue, transform, {
            {0u, dispatchBuffer},
            {30u, makeBuffer(device, std::vector<NMFGMRESLayoutGPU>{layout})},
            {1u, interiorMap}, {2u, inputA}, {3u, inputB},
            {4u, patchCorrections}, {5u, output},
            {6u, makeBuffer(device, std::vector<std::uint32_t>{mode})},
            {7u, makeBuffer(device, std::vector<std::uint32_t>{2u})},
            {8u, transformStatus},
            {9u, makeBuffer(device, std::vector<std::uint32_t>{0u})},
            {10u, makeBuffer(device, std::vector<std::uint32_t>{0u})},
            {11u, makeBuffer(device, std::vector<std::uint32_t>{0u})},
            {12u, makeBuffer(device, std::vector<std::uint32_t>{0u})}},
            1u, 32u);
        require(readBuffer<NMMatterStatusGPU>(transformStatus, 1u)[0].code ==
                NM_STATUS_SUCCESS,
            "due-aware Schur transform rejected the two-object fixture");
    };
    auto condensedRhs = makeBuffer(device, std::vector<nm_float4>(3u));
    transformArgs(2u, rhsBuffer, aTimesYiBuffer, condensedRhs);
    const auto condensed = readBuffer<nm_float4>(condensedRhs, 3u);
    require(condensed[0].x == 0.0f && condensed[1].x == 0.0f &&
            std::abs(condensed[2].x - 6.5f) <= 1.0e-6f &&
            std::abs(condensed[2].y - 5.0f) <= 1.0e-6f &&
            std::abs(condensed[2].z + 4.5f) <= 1.0e-6f,
        "due-aware condensed RHS disagrees with full A and local solve");
    // The interface Schur diagonal is 4I. Solve it directly, then add the
    // GPU local recovery into that full interface vector.
    std::array<nm_float4, 3> interfaceSolution{};
    interfaceSolution[2] = nm_float4{1.625f, 1.25f, -1.125f, 0.0f};
    auto interfaceBuffer = makeBuffer(device,
        std::vector<nm_float4>(interfaceSolution.begin(),
                               interfaceSolution.end()));
    auto recoveredBuffer = makeBuffer(device, std::vector<nm_float4>(3u));
    transformArgs(4u, interfaceBuffer, rhsBuffer, recoveredBuffer);
    const auto recovered = readBuffer<nm_float4>(recoveredBuffer, 3u);
    const std::array<nm_float4, 3> expected{
        nm_float4{3.0f, -2.0f, 1.0f, 0.0f},
        nm_float4{0.0f, 0.0f, 0.0f, 0.0f},
        nm_float4{1.625f, 1.25f, -1.125f, 0.0f}};
    for (std::uint32_t node = 0u; node < 3u; ++node)
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            const float actual = axis == 0u ? recovered[node].x :
                (axis == 1u ? recovered[node].y : recovered[node].z);
            const float reference = axis == 0u ? expected[node].x :
                (axis == 1u ? expected[node].y : expected[node].z);
            require(std::abs(actual - reference) <= 1.0e-6f,
                "due-aware recovered full vector differs from reference");
        }
    const std::array<float, 3> activeFullResidual{
        2.0f * recovered[0].x - rhs[0].x,
        2.0f * recovered[0].y - rhs[0].y,
        2.0f * recovered[0].z - rhs[0].z};
    const std::array<float, 3> gammaFullResidual{
        0.5f * recovered[0].x + 4.0f * recovered[2].x - rhs[2].x,
        0.5f * recovered[0].y + 4.0f * recovered[2].y - rhs[2].y,
        0.5f * recovered[0].z + 4.0f * recovered[2].z - rhs[2].z};
    for (const float value : activeFullResidual)
        require(std::abs(value) <= 1.0e-6f,
            "due-aware recovered active full-system residual is nonzero");
    for (const float value : gammaFullResidual)
        require(std::abs(value) <= 1.0e-6f,
            "due-aware recovered interface full-system residual is nonzero");
    require(rhs[1].x == 0.0f && rhs[1].y == 0.0f && rhs[1].z == 0.0f,
        "inactive due object must have zero full-operator RHS");
    std::cout << "regional_schur_due_rows=active_and_inactive_pass "
                 "condensed_rhs=passed recovery=passed result=PASS\n";
}

void checkAcceptedSubspacePredictor(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLLibrary> library
) {
    auto reset = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_accepted_subspace_reset");
    auto projectPartial = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_accepted_subspace_project_partial");
    auto projectFinish = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_accepted_subspace_project_finish");
    auto buildGuess = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_accepted_subspace_build_guess");
    auto formResidual = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_accepted_subspace_form_residual");
    auto addGuess = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_accepted_subspace_add_guess");
    auto commitVectors = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_accepted_subspace_commit_vectors");
    auto commitCount = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_accepted_subspace_commit_count");

    NMMatterDispatchGPU dispatchData{};
    dispatchData.abiVersion = NM_MATTER_ABI_VERSION;
    dispatchData.environmentCount = 1u;
    dispatchData.femNodeCount = 4u;
    NMFGMRESLayoutGPU layout{};
    layout.unknownCount = 4u;
    layout.reserved = 1u;
    auto dispatchBuffer = makeBuffer(device,
        std::vector<NMMatterDispatchGPU>{dispatchData});
    auto layoutBuffer = makeBuffer(device,
        std::vector<NMFGMRESLayoutGPU>{layout});

    // C0=[1,2,0,0], C1=[0,1,1,0], and b=3*C0+2*C1.
    // The old images are predictor data only; the current action below is
    // supplied separately and is the sole source for the rebuilt residual.
    auto vectorsBuffer = makeBuffer(device, std::vector<nm_float4>{
        {1,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0},
        {0,0,0,0}, {1,0,0,0}, {0,0,0,0}, {0,0,0,0}});
    auto imagesBuffer = makeBuffer(device, std::vector<nm_float4>{
        {1,0,0,0}, {2,0,0,0}, {0,0,0,0}, {0,0,0,0},
        {0,0,0,0}, {1,0,0,0}, {1,0,0,0}, {0,0,0,0}});
    auto rhsBuffer = makeBuffer(device, std::vector<nm_float4>{
        {3,0,0,0}, {8,0,0,0}, {2,0,0,0}, {0,0,0,0}});
    auto countsBuffer = makeBuffer(device, std::vector<std::uint32_t>{2u});
    auto partialsBuffer = makeBuffer(device, std::vector<float>(40u, 0.0f));
    auto coefficientsBuffer = makeBuffer(device,
        std::vector<nm_float4>{{0,0,0,0}});
    auto guessBuffer = makeBuffer(device, std::vector<nm_float4>(4u));
    dispatch(device, queue, projectPartial,
        {{0u, dispatchBuffer}, {1u, rhsBuffer}, {2u, imagesBuffer},
         {3u, countsBuffer}, {4u, partialsBuffer}, {30u, layoutBuffer}},
        1u, 256u);
    dispatch(device, queue, projectFinish,
        {{0u, dispatchBuffer}, {1u, countsBuffer}, {2u, partialsBuffer},
         {3u, coefficientsBuffer}, {30u, layoutBuffer}}, 1u, 32u);
    auto coefficients = readBuffer<nm_float4>(coefficientsBuffer, 1u);
    require(close(coefficients[0].x, 3.0f, 2.0e-5f) &&
            close(coefficients[0].y, 2.0f, 2.0e-5f) &&
            close(coefficients[0].z, 1.0f) &&
            close(coefficients[0].w, 2.0f),
        "accepted-subspace least-squares projection must recover both coefficients");
    // Three tiles exercise every SIMD-group partial writer, the partial-tile
    // tail, cancellation, and a safely near-dependent image pair.
    auto verifyWideProjection = [&](const bool nearDependent) {
        constexpr std::uint32_t rows = 600u;
        constexpr std::uint32_t tiles = (rows + 255u) / 256u;
        NMMatterDispatchGPU wideDispatch = dispatchData;
        wideDispatch.femNodeCount = rows;
        NMFGMRESLayoutGPU wideLayout = layout;
        wideLayout.unknownCount = rows;
        std::vector<nm_float4> wideImages(2u * rows, nm_float4{});
        std::vector<nm_float4> wideRhs(rows, nm_float4{});
        double g00 = 0.0, g01 = 0.0, g11 = 0.0, q0 = 0.0, q1 = 0.0;
        for (std::uint32_t i = 0u; i < rows; ++i) {
            const float centered = float(int(i % 13u) - 6);
            const float c0 = 1.0f + 0.001f * centered;
            const float sign = (i & 1u) == 0u ? -1.0f : 1.0f;
            const float c1 = nearDependent
                ? c0 + 0.045f * sign + 0.001f * float(int(i % 7u) - 3)
                : sign * (1.0f + 0.002f * float(i % 11u));
            const float noise = nearDependent
                ? 0.0001f * float(int(i % 3u) - 1)
                : 0.003f * float(int(i % 5u) - 2);
            const float b = 2.75f * c0 - 1.25f * c1 + noise;
            wideImages[i].x = c0;
            wideImages[rows + i].x = c1;
            wideRhs[i].x = b;
            g00 += double(c0) * c0;
            g01 += double(c0) * c1;
            g11 += double(c1) * c1;
            q0 += double(c0) * b;
            q1 += double(c1) * b;
        }
        const double determinant = g00 * g11 - g01 * g01;
        require(determinant > 0.0, "wide projection oracle is singular");
        const double beta0 = (q0 * g11 - q1 * g01) / determinant;
        const double beta1 = (q1 * g00 - q0 * g01) / determinant;
        auto wideDispatchBuffer = makeBuffer(device,
            std::vector<NMMatterDispatchGPU>{wideDispatch});
        auto wideLayoutBuffer = makeBuffer(device,
            std::vector<NMFGMRESLayoutGPU>{wideLayout});
        auto wideImagesBuffer = makeBuffer(device, wideImages);
        auto wideRhsBuffer = makeBuffer(device, wideRhs);
        auto widePartialsBuffer = makeBuffer(device,
            std::vector<float>(std::size_t(tiles) * 8u * 5u, 0.0f));
        auto wideCountsBuffer = makeBuffer(device,
            std::vector<std::uint32_t>{2u});
        auto wideCoefficientsBuffer = makeBuffer(device,
            std::vector<nm_float4>(1u));
        dispatch(device, queue, projectPartial,
            {{0u, wideDispatchBuffer}, {1u, wideRhsBuffer},
             {2u, wideImagesBuffer}, {3u, wideCountsBuffer},
             {4u, widePartialsBuffer}, {30u, wideLayoutBuffer}}, tiles, 256u);
        dispatch(device, queue, projectFinish,
            {{0u, wideDispatchBuffer}, {1u, wideCountsBuffer},
             {2u, widePartialsBuffer}, {3u, wideCoefficientsBuffer},
             {30u, wideLayoutBuffer}}, 1u, 32u);
        const auto wideCoefficients = readBuffer<nm_float4>(
            wideCoefficientsBuffer, 1u)[0];
        const float tolerance = nearDependent ? 1.5e-3f : 4.0e-4f;
        require(wideCoefficients.z > 0.5f && wideCoefficients.w == 2.0f &&
                std::abs(double(wideCoefficients.x) - beta0) < tolerance &&
                std::abs(double(wideCoefficients.y) - beta1) < tolerance,
            nearDependent
                ? "near-dependent multi-SIMD projection differs from double oracle"
                : "multi-tile cancellation projection differs from double oracle");
    };
    verifyWideProjection(false);
    verifyWideProjection(true);

    dispatch(device, queue, buildGuess,
        {{0u, dispatchBuffer}, {1u, vectorsBuffer},
         {2u, coefficientsBuffer}, {3u, guessBuffer}, {30u, layoutBuffer}},
        1u, 4u);
    auto guess = readBuffer<nm_float4>(guessBuffer, 4u);
    require(close(guess[0].x, 3.0f) && close(guess[1].x, 2.0f) &&
            close(guess[2].x, 0.0f) && close(guess[3].x, 0.0f),
        "accepted-subspace guess must map coefficients through stored corrections");

    auto currentAction = makeBuffer(device, std::vector<nm_float4>{
        {7,0,0,0}, {-1,0,0,0}, {8,0,0,0}, {0,0,0,0}});
    auto residual = makeBuffer(device, std::vector<nm_float4>(4u));
    std::vector<NMMatterStatusGPU> healthy(1u);
    healthy[0].code = NM_STATUS_SUCCESS;
    auto statuses = makeBuffer(device, healthy);
    dispatch(device, queue, formResidual,
        {{0u, dispatchBuffer}, {1u, rhsBuffer}, {2u, currentAction},
         {3u, residual}, {4u, statuses}, {30u, layoutBuffer}}, 1u, 4u);
    auto residualValues = readBuffer<nm_float4>(residual, 4u);
    require(close(residualValues[0].x, -4.0f) &&
            close(residualValues[1].x, 9.0f) &&
            close(residualValues[2].x, -6.0f) &&
            close(residualValues[3].x, 0.0f),
        "accepted-subspace path must rebuild residual from current operator action");

    auto solutionBuffer = makeBuffer(device, std::vector<nm_float4>{
        {10,0,0,0}, {20,0,0,0}, {30,0,0,0}, {40,0,0,0}});
    dispatch(device, queue, addGuess,
        {{0u, dispatchBuffer}, {1u, guessBuffer}, {2u, solutionBuffer},
         {30u, layoutBuffer}}, 1u, 4u);
    auto solution = readBuffer<nm_float4>(solutionBuffer, 4u);
    require(close(solution[0].x, 13.0f) && close(solution[1].x, 22.0f) &&
            close(solution[2].x, 30.0f) && close(solution[3].x, 40.0f),
        "accepted-subspace initial guess must be added once to FGMRES correction");

    auto alphaBuffer = makeBuffer(device, std::vector<float>{0.5f});
    auto fullCurrentAction = makeBuffer(device, std::vector<nm_float4>{
        {57,0,0,0}, {9,0,0,0}, {48,0,0,0}, {40,0,0,0}});
    dispatch(device, queue, commitVectors,
        {{0u, dispatchBuffer}, {1u, solutionBuffer}, {2u, fullCurrentAction},
         {3u, vectorsBuffer}, {4u, imagesBuffer}, {5u, countsBuffer},
         {6u, alphaBuffer}, {7u, statuses}, {30u, layoutBuffer}}, 1u, 4u);
    dispatch(device, queue, commitCount,
        {{0u, alphaBuffer}, {1u, statuses}, {2u, countsBuffer}}, 1u, 32u);
    auto committedVectors = readBuffer<nm_float4>(vectorsBuffer, 8u);
    auto committedImages = readBuffer<nm_float4>(imagesBuffer, 8u);
    auto committedCount = readBuffer<std::uint32_t>(countsBuffer, 1u);
    require(close(committedVectors[0].x, 6.5f) &&
            close(committedVectors[1].x, 11.0f) &&
            close(committedVectors[2].x, 15.0f) &&
            close(committedVectors[3].x, 20.0f) &&
            close(committedVectors[4].x, 1.0f) &&
            close(committedImages[0].x, 28.5f) &&
            close(committedImages[1].x, 4.5f) &&
            close(committedImages[2].x, 24.0f) &&
            close(committedImages[3].x, 20.0f) &&
            close(committedImages[4].x, 1.0f) &&
            close(committedImages[5].x, 2.0f) && committedCount[0] == 2u,
        "accepted history must shift and store alpha-scaled correction/current image");

    // Rejection and root reset must not carry stale subspace vectors forward.
    *static_cast<float*>([alphaBuffer contents]) = 0.0f;
    dispatch(device, queue, commitCount,
        {{0u, alphaBuffer}, {1u, statuses}, {2u, countsBuffer}}, 1u, 32u);
    committedCount = readBuffer<std::uint32_t>(countsBuffer, 1u);
    require(committedCount[0] == 0u,
        "rejected Newton step must clear accepted-subspace history");
    *static_cast<std::uint32_t*>([countsBuffer contents]) = 2u;
    dispatch(device, queue, reset, {{0u, countsBuffer}}, 1u, 1u);
    committedCount = readBuffer<std::uint32_t>(countsBuffer, 1u);
    require(committedCount[0] == 0u,
        "new physical root must clear accepted-subspace history");
}

void checkCertificationStateCopy(id<MTLDevice> device,
    id<MTLCommandQueue> queue, id<MTLLibrary> library) {
    auto kernel = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_prepare_certification_operator_states");
    NMMatterDispatchGPU shape{}; shape.environmentCount = 2u;
    std::vector<NMFGMRESStateGPU> original(2u);
    original[0].diagnostics = {3.0f, 4.0f, 1.0f, 6.0f};
    original[1].diagnostics = {7.0f, 8.0f, 1.0f, 9.0f};
    std::vector<NMMatterStatusGPU> health(2u);
    health[0].code = NM_STATUS_SUCCESS;
    health[1].code = NM_STATUS_NONLINEAR_SOLVER_FAILURE;
    auto source = makeBuffer(device, original);
    auto result = makeBuffer(device, std::vector<NMFGMRESStateGPU>(2u));
    auto copiedHealth = makeBuffer(device, std::vector<NMMatterStatusGPU>(2u));
    dispatch(device, queue, kernel, {{0u, makeBuffer(device, std::vector{shape})},
        {1u, source}, {2u, result}, {3u, makeBuffer(device, health)},
        {4u, copiedHealth}}, 1u, 32u);
    auto values = readBuffer<NMFGMRESStateGPU>(result, 2u);
    auto actualHealth = readBuffer<NMMatterStatusGPU>(copiedHealth, 2u);
    require(std::memcmp(actualHealth.data(), health.data(), sizeof(NMMatterStatusGPU)*2u) == 0,
        "certification must preserve source failure status");
    auto expected = original; expected[0].diagnostics.z = 0.0f;
    require(std::memcmp(values.data(), expected.data(), sizeof(NMFGMRESStateGPU)*2u) == 0,
        "certification must clear only a healthy copied completion latch");
    auto unchanged = readBuffer<NMFGMRESStateGPU>(source, 2u);
    require(std::memcmp(unchanged.data(), original.data(), sizeof(NMFGMRESStateGPU)*2u) == 0,
        "certification must preserve production solver state");
}

int main(int argc, const char* argv[]) {
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
            checkAcceptedSubspacePredictor(device, queue, library);
            auto legacyPrecondition = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_precondition");
            auto emptyFieldPrecondition = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_precondition_empty_fields");
            auto fieldInitialize = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_field_smoother_initialize");
            auto fieldExport = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_field_smoother_export");
            auto crossPrecondition = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_precondition_cross");
            auto accumulateMechanical = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_accumulate");
            auto emptyFieldAccumulate = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_accumulate_fields");
            checkEmptyFieldPreconditioner(device, queue, legacyPrecondition,
                emptyFieldPrecondition, fieldInitialize, fieldExport,
                crossPrecondition, accumulateMechanical, emptyFieldAccumulate);
            checkCertificationStateCopy(device, queue, library);
            checkAcceptedZ0Recycle(device, queue, library);
            checkRegionalSchurInteriorExtraction(device, queue, library);
            checkRegionalSchurTransformsAndRecovery(device, queue, library);
            checkDueAwareSchurPatchAssembly(device, queue, library);
            checkActiveContactSchurEligibility(device, queue, library);
            auto operatorICBRanges = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_prepare_operator_icb_ranges");
            checkOperatorICBRanges(device, queue, operatorICBRanges);

            auto finalize = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_finalize_cycle");
            auto telemetryFinalize = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_telemetry_finalize");
            auto telemetryExhaustion = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_telemetry_exhaustion");
            checkTelemetryExhaustion(device, queue, telemetryExhaustion);
            auto telemetryDomainsPartial =
                pipelineWithRelativeInnerForcingDisabled(device, library,
                    "numi_matter_metal::nm_fgmres_telemetry_domains_partial");
            auto telemetryDomainsFinish =
                pipelineWithRelativeInnerForcingDisabled(device, library,
                    "numi_matter_metal::nm_fgmres_telemetry_domains_finish");
            checkTelemetryDomainPartitions(device, queue,
                telemetryDomainsPartial, telemetryDomainsFinish);
            auto accumulate = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_accumulate");
            auto accumulateFields = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_accumulate_fields");
            auto measure = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_measure_true_residual");
            auto beginLegacy = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_begin");
            auto beginPartial = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_begin_distributed_partial");
            auto beginFinish = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_begin_distributed_finish");
            auto beginBasis0 = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_begin_distributed_basis0");
            auto ewPrepare = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_prepare_inexact_newton_forcing");
            auto ewCommit = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_commit_inexact_newton_history");
            auto correctionLegacy = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_measure_correction");
            auto correctionPartial = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_correction_distributed_partial");
            auto correctionFinish = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_correction_distributed_finish");
            auto trueResidualPartial = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_true_residual_distributed_partial");
            auto trueResidualFinish = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_true_residual_distributed_finish");
            auto trueResidualReplace = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_true_residual_distributed_replace");
            auto hybridPhysicalPreserve = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_hybrid_preserve_physical_rhs");
            auto hybridPhysicalPartial = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_hybrid_physical_residual_partial");
            auto hybridPhysicalFinish = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_hybrid_physical_residual_finish");
            checkDistributedBeginAndCorrection(
                device,queue,library,beginLegacy,beginPartial,beginFinish,
                beginBasis0,correctionLegacy,correctionPartial,correctionFinish,
                trueResidualPartial,trueResidualFinish);
            checkInexactNewtonForcing(device, queue, ewPrepare, ewCommit);
            auto cgs2Partial = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_cgs2_partial");
            auto cgs2ReduceRow = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_cgs2_reduce_row");
            auto cgs2Update = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_cgs2_update");
            auto cgs2MeasureNorm = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_cgs2_measure_norm");
            auto cgs2FinishColumn =
                pipelineWithRelativeInnerForcingDisabled(device, library,
                    "numi_matter_metal::nm_fgmres_cgs2_finish_column");
            auto cgs2Normalize = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_cgs2_normalize");
            checkDistributedCGS2(
                device, queue, cgs2Partial, cgs2ReduceRow, cgs2Update,
                cgs2MeasureNorm, cgs2FinishColumn, cgs2Normalize);
            checkInactiveTransportSpan(
                device, queue, cgs2Partial, cgs2ReduceRow, cgs2Update,
                cgs2MeasureNorm, cgs2FinishColumn, cgs2Normalize);
            auto cgs2PrepareRange = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_cgs2_prepare_range");
            auto cgs2PartialICB = indirectPipeline(device, library,
                "numi_matter_metal::nm_fgmres_cgs2_partial");
            auto cgs2ReduceRowICB = indirectPipeline(device, library,
                "numi_matter_metal::nm_fgmres_cgs2_reduce_row");
            auto cgs2UpdateICB = indirectPipeline(device, library,
                "numi_matter_metal::nm_fgmres_cgs2_update");
            auto cgs2MeasureNormICB = indirectPipeline(device, library,
                "numi_matter_metal::nm_fgmres_cgs2_measure_norm");
            auto cgs2FinishColumnICB =
                indirectPipelineWithRelativeInnerForcingDisabled(device,
                    library, "numi_matter_metal::nm_fgmres_cgs2_finish_column");
            auto cgs2NormalizeICB = indirectPipeline(device, library,
                "numi_matter_metal::nm_fgmres_cgs2_normalize");
            checkCGS2IndirectRange(device, queue, cgs2PrepareRange,
                cgs2PartialICB, cgs2ReduceRowICB, cgs2UpdateICB,
                cgs2MeasureNormICB, cgs2FinishColumnICB, cgs2NormalizeICB);

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
            auto telemetryBuffer = makeBuffer(device,
                std::vector<NMFGMRESTelemetryGPU>{NMFGMRESTelemetryGPU{}});
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
            dispatch(device, queue, telemetryFinalize, {
                {0u, dispatchBuffer}, {1u, solverBuffer},
                {2u, stateBuffer}, {3u, telemetryBuffer}}, 1u, 1u);
            dispatch(device, queue, finalize, {
                {0u, dispatchBuffer}, {30u, layoutBuffer}, {1u, solverBuffer},
                {2u, hessenbergBuffer}, {3u, leastSquaresBuffer},
                {4u, stateBuffer}, {5u, statusBuffer},
                {6u, makeBuffer(device, std::vector<std::uint32_t>{offset})},
                {7u, makeBuffer(device, std::vector<std::uint32_t>{finalCycle})},
                {8u, makeBuffer(device, std::vector<float>{forcing})},
                {9u, rotationsBuffer}, {10u, restartCoefficientsBuffer},
                {11u, makeBuffer(device, std::vector<std::uint32_t>{defer})}},
                1u, 32u);
            const auto telemetryAfterFinalize =
                readBuffer<NMFGMRESTelemetryGPU>(telemetryBuffer, 1u);
            require(telemetryAfterFinalize[0].linear.x == 1u &&
                    telemetryAfterFinalize[0].linear.y == 1u &&
                    telemetryAfterFinalize[0].linear.z == 1u,
                "standalone telemetry did not count the encoded cycle and used column");
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

            // A vascular-coupled Arnoldi estimate must meet the authored
            // normalized absolute row tolerance before cycle finalization.
            // This checks the estimated Krylov stopping gate, not a measured
            // true residual; the optional true-residual fixture above remains
            // limited to its supported FEM-only operator scope.
            NMFGMRESLayoutGPU vascularLayout = layout;
            vascularLayout.unknownCount = 3u;
            vascularLayout.vascularUnknownCount = 1u;
            vascularLayout.vascularBase = 2u;
            vascularLayout.vascularResidualTolerance = 1.0e-5f;
            auto vascularLayoutBuffer = makeBuffer(device,
                std::vector<NMFGMRESLayoutGPU>{vascularLayout});
            auto checkVascularCycleTarget = [&](const float arnoldiEstimate,
                                                const bool expectedConverged) {
                std::fill(hessenberg.begin(), hessenberg.end(), 0.0f);
                hessenberg[0] = 1.0f;
                std::fill(leastSquares.begin(), leastSquares.end(), 0.0f);
                leastSquares[0] = 0.5f;
                std::memcpy([hessenbergBuffer contents], hessenberg.data(),
                    hessenberg.size()*sizeof(float));
                std::memcpy([leastSquaresBuffer contents], leastSquares.data(),
                    leastSquares.size()*sizeof(float));
                NMFGMRESStateGPU vascularState{};
                vascularState.diagnostics = {
                    arnoldiEstimate, 1.0e-3f, 0.0f, 1.0f};
                vascularState.nonlinear = {1.0e-3f, 1.0e-3f, 0.0f, 0.0f};
                std::memcpy([stateBuffer contents], &vascularState,
                    sizeof(vascularState));
                resetStatuses(diagnosticBuffer, statusBuffer, healthy, healthy);
                const std::uint32_t doNotDefer = 0u;
                const float vascularForcing = 0.25f;
                dispatch(device, queue, finalize, {
                    {0u, dispatchBuffer}, {30u, vascularLayoutBuffer},
                    {1u, solverBuffer}, {2u, hessenbergBuffer},
                    {3u, leastSquaresBuffer}, {4u, stateBuffer},
                    {5u, statusBuffer},
                    {6u, makeBuffer(device,std::vector<std::uint32_t>{offset})},
                    {7u, makeBuffer(device,std::vector<std::uint32_t>{finalCycle})},
                    {8u, makeBuffer(device,std::vector<float>{vascularForcing})},
                    {9u, rotationsBuffer}, {10u, restartCoefficientsBuffer},
                    {11u, makeBuffer(device,std::vector<std::uint32_t>{doNotDefer})}},
                    1u, 32u);
                const auto measured = readBuffer<NMFGMRESStateGPU>(stateBuffer,1u)[0];
                require((measured.diagnostics.z == 1.0f) == expectedConverged,
                    "vascular Arnoldi estimate bypassed normalized absolute tolerance: estimate=" +
                    std::to_string(arnoldiEstimate) + " converged=" +
                    std::to_string(measured.diagnostics.z));
            };
            checkVascularCycleTarget(5.0e-5f, false);
            checkVascularCycleTarget(5.0e-6f, true);
            std::cout << "vascular_arnoldi_absolute_target=true estimates=5e-5,5e-6 result=PASS\n";

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

            require(argc==1||(argc==2&&std::string(argv[1])=="--refined-coupon"),"usage: true-residual check [--refined-coupon]");
            if(argc==2){checkLargeNodeResidualMeasurement(device,queue,measure,trueResidualPartial,trueResidualFinish,trueResidualReplace,20682u);checkLargeNodeResidualMeasurement(device,queue,measure,trueResidualPartial,trueResidualFinish,trueResidualReplace,32768u);}
            checkLargeNodeResidualMeasurement(device,queue,measure,trueResidualPartial,trueResidualFinish,trueResidualReplace,4096u);
            checkLargeNodeResidualMeasurement(device,queue,measure,trueResidualPartial,trueResidualFinish,trueResidualReplace,8192u);
            checkHybridTailTrueResidualDistributed(device, queue, trueResidualPartial, trueResidualFinish, trueResidualReplace);
            checkHybridFullPhysicalResidualDiagnostic(device, queue,
                hybridPhysicalPreserve, hybridPhysicalPartial, hybridPhysicalFinish);
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

