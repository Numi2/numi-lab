#include "numi/matter/partitioned_reduction_projection.h"
#include "numi/matter/partitioned_reduction_hybrid_gs.hpp"
#include "numi/matter/human_limits_gpu.h"
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
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <type_traits>
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

void dispatchGrid3D(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> state,
    const std::vector<std::pair<NSUInteger, id<MTLBuffer>>>& bindings,
    const NSUInteger groupsX, const NSUInteger groupsY,
    const NSUInteger groupsZ, const NSUInteger threads) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        command != nil ? [command computeCommandEncoder] : nil;
    require(command != nil && encoder != nil,
        "cannot allocate 3D Metal command encoder");
    [encoder setComputePipelineState:state];
    for (const auto& [index, buffer] : bindings)
        [encoder setBuffer:buffer offset:0u atIndex:index];
    [encoder dispatchThreadgroups:MTLSizeMake(groupsX, groupsY, groupsZ)
        threadsPerThreadgroup:MTLSizeMake(threads, 1u, 1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
        command.error != nil
            ? std::string("3D Metal command failed: ") +
                [[command.error localizedDescription] UTF8String]
            : "3D Metal command did not complete");
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


struct Selector6ICBConstants {
    NMMatterDispatchGPU dispatch;
    NMFGMRESLayoutGPU layout;
    NMMicrostepGPU operatorMicro;
    NMMicrostepGPU micro;
    NMVascularLayoutGPU vascularLayout;
    NMHumanSupportDispatchGPU supportDispatch;
    NMHumanEqualityDispatchGPU equalityDispatch;
    std::uint32_t articulatedNv, useCache, mixedAllowConverged, allowConverged;
    std::uint32_t springCount, useWorkingSet, currentBodyCount, bodyStride;
    std::uint32_t compensatedTranslation, assembledXYZNNZ;
    NMPartitionedReductionBlockDispatchGPU partitionedDispatch;
    std::uint32_t partitionedBlockModeCount, partitionedRegionEnvironmentCount;
    std::uint32_t partitionedSparseTileCount, graphPatches, graphPatchNodes;
    std::uint32_t graphMatrixFloats, graphPivots, graphModeCount;
    std::uint32_t graphModeStride, graphRefinementMode;
    std::uint32_t graphNoActiveContactEligibility;
};
static_assert(std::is_trivially_copyable_v<Selector6ICBConstants>);

void checkSelector6PrefixICB(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLLibrary> library
) {
    constexpr std::uint32_t commandCount = 6u;
    NMMatterDispatchGPU matter{};
    matter.abiVersion = NM_MATTER_ABI_VERSION;
    matter.environmentCount = 1u;
    matter.femNodeCount = 1u;
    NMFGMRESLayoutGPU layout{};
    layout.unknownCount = 4u;
    NMMicrostepGPU operatorMicro{};
    NMMicrostepGPU micro{};
    micro.solverIteration = 1u;
    NMPartitionedReductionBlockDispatchGPU block{};
    block.globalNodeCount = 1u;
    block.basisNodeCount = 1u;
    block.modeCount = 1u;
    block.tileCount = 1u;
    block.environmentCount = 1u;
    block.regionCount = 1u;
    block.blockModeCount = 1u;
    block.reserved = 1u;

    Selector6ICBConstants constants{};
    constants.dispatch = matter;
    constants.layout = layout;
    constants.operatorMicro = operatorMicro;
    constants.micro = micro;
    constants.partitionedDispatch = block;
    constants.partitionedBlockModeCount = 1u;
    constants.partitionedRegionEnvironmentCount = 1u;
    constants.partitionedSparseTileCount = 1u;
    constants.graphPatches = 1u;
    constants.graphPatchNodes = 1u;
    constants.graphMatrixFloats = 9u;
    constants.graphPivots = 9u;
    constants.graphModeCount = 0u;
    constants.graphModeStride = 0u;
    constants.graphRefinementMode = 1u;
    constants.graphNoActiveContactEligibility = 0u;

    auto matterBuffer = makeBuffer(device,
        std::vector<NMMatterDispatchGPU>{matter});
    auto stateBuffer = makeBuffer(device,
        std::vector<NMFGMRESStateGPU>{NMFGMRESStateGPU{}});
    auto statusBuffer = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
    auto prefixCountBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{commandCount});
    auto tailCountBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u});
    auto rangeBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u,0u,0u,0u});
    auto sourceConstantsBuffer = makeBuffer(device,
        std::vector<Selector6ICBConstants>{constants});
    auto destinationConstantsBuffer = makeBuffer(device,
        std::vector<Selector6ICBConstants>{constants});
    const std::uint32_t constantWordCount =
        static_cast<std::uint32_t>(sizeof(constants) / sizeof(std::uint32_t));

    const auto reduceTiles = pipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_reduce_sparse_mode_tiles");
    const auto reduceTilesICB = indirectPipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_reduce_sparse_mode_tiles");
    const auto reducePartials = pipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_reduce_sparse_mode_partials");
    const auto reducePartialsICB = indirectPipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_reduce_sparse_mode_partials");
    const auto solveBlocks = pipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_solve_region_blocks");
    const auto solveBlocksICB = indirectPipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_solve_region_blocks");
    const auto addCorrection = pipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_add_sparse_region_correction");
    const auto addCorrectionICB = indirectPipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_add_sparse_region_correction");
    const auto solvePatches = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_graph_solve_patches");
    const auto solvePatchesICB = indirectPipeline(device, library,
        "numi_matter_metal::nm_fgmres_graph_solve_patches");
    const auto applyPatches = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_graph_apply_patches");
    const auto applyPatchesICB = indirectPipeline(device, library,
        "numi_matter_metal::nm_fgmres_graph_apply_patches");

    auto globalNodeIndices = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto basisModes = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto regionSupportRanges = makeBuffer(device,
        std::vector<NMPartitionedReductionSparseRangeGPU>{
            NMPartitionedReductionSparseRangeGPU{0u,1u}});
    auto regionSupportRows = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto nodeModeRanges = makeBuffer(device,
        std::vector<NMPartitionedReductionSparseRangeGPU>{
            NMPartitionedReductionSparseRangeGPU{0u,1u}});
    auto nodeModeSlots = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto mass = makeBuffer(device, std::vector<float>{1.0f});
    auto partials = makeBuffer(device, std::vector<float>{-99.0f});
    auto reduced = makeBuffer(device, std::vector<float>{-99.0f});
    auto lu = makeBuffer(device, std::vector<float>{2.0f});
    auto pivots = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto regionValid = makeBuffer(device, std::vector<std::uint32_t>{1u});
    auto factorValid = makeBuffer(device, std::vector<std::uint32_t>{1u});
    auto coefficients = makeBuffer(device, std::vector<float>{-99.0f});
    auto solveValid = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto regionalBasis = makeBuffer(device,
        std::vector<nm_float4>{nm_float4{1.0f,0.0f,0.0f,0.0f}});
    auto regionalAP = makeBuffer(device,
        std::vector<nm_float4>{nm_float4{0.0f,0.0f,0.0f,0.0f}});
    auto graphPatches = makeBuffer(device,
        std::vector<nm_uint4>{nm_uint4{0u,1u,0u,0u}});
    auto graphPatchNodes = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto graphFactors = makeBuffer(device,
        std::vector<float>{1.0f,0.0f,0.0f, 0.0f,1.0f,0.0f, 0.0f,0.0f,1.0f});
    auto graphOriginal = makeBuffer(device,
        std::vector<float>{1.0f,0.0f,0.0f, 0.0f,1.0f,0.0f, 0.0f,0.0f,1.0f});
    float oneFloat = 1.0f;
    std::uint32_t oneBits = 0u;
    std::memcpy(&oneBits, &oneFloat, sizeof(oneBits));
    auto graphPivots = makeBuffer(device,
        std::vector<std::uint32_t>{0u,1u,2u,oneBits,oneBits,oneBits,
                                   oneBits,oneBits,oneBits});
    auto graphFactorValid = makeBuffer(device, std::vector<std::uint32_t>{1u});
    auto graphCorrections = makeBuffer(device,
        std::vector<nm_float4>{nm_float4{-77.0f,-77.0f,-77.0f,-77.0f}});
    auto nodePatchOffsets = makeBuffer(device, std::vector<std::uint32_t>{0u,1u});
    auto nodePatchValues = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto nodeCoverage = makeBuffer(device, std::vector<std::uint32_t>{1u});
    auto candidate = makeBuffer(device,
        std::vector<NMFEMNodeStateGPU>{NMFEMNodeStateGPU{}});
    auto dummy = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto prepareRange = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_prepare_operator_icb_ranges");

    MTLIndirectCommandBufferDescriptor* descriptor =
        [MTLIndirectCommandBufferDescriptor new];
    descriptor.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
    descriptor.inheritBuffers = NO;
    descriptor.inheritPipelineState = NO;
    descriptor.maxKernelBufferBindCount = 31u;

    struct RunBuffers {
        id<MTLBuffer> input;
        id<MTLBuffer> output;
        id<MTLIndirectCommandBuffer> graph;
    };
    std::array<RunBuffers,2> runs{};
    const std::array<nm_float4,2> inputs{{
        nm_float4{2.0f,3.0f,4.0f,0.0f},
        nm_float4{1.25f,-0.5f,2.25f,0.0f}}};
    for (std::size_t runIndex = 0u; runIndex < runs.size(); ++runIndex) {
        runs[runIndex].input = makeBuffer(device,
            std::vector<nm_float4>{inputs[runIndex]});
        runs[runIndex].output = makeBuffer(device,
            std::vector<nm_float4>{nm_float4{0.25f,0.5f,0.75f,0.0f}});
        runs[runIndex].graph = [device
            newIndirectCommandBufferWithDescriptor:descriptor
                maxCommandCount:commandCount
                options:MTLResourceStorageModePrivate];
        require(runs[runIndex].graph != nil,
            "selector-6 prefix ICB allocation failed");

        auto runConstants = constants;
        runConstants.micro.solverIteration =
            static_cast<std::uint32_t>(runIndex + 1u);
        auto* source = static_cast<Selector6ICBConstants*>(
            [sourceConstantsBuffer contents]);
        source[0] = runConstants;

        NSUInteger ordinal = 0u;
        const auto beginCommand = [&](id<MTLComputePipelineState> p) {
            require(ordinal < commandCount,
                "selector-6 prefix ICB command capacity exceeded before recording");
            id<MTLIndirectComputeCommand> cmd =
                [runs[runIndex].graph indirectComputeCommandAtIndex:ordinal++];
            [cmd setComputePipelineState:p];
            [cmd setKernelBuffer:destinationConstantsBuffer
                offset:offsetof(Selector6ICBConstants, dispatch) atIndex:0u];
            [cmd setKernelBuffer:destinationConstantsBuffer
                offset:offsetof(Selector6ICBConstants, layout) atIndex:30u];
            return cmd;
        };
        const auto bind = [&](id<MTLIndirectComputeCommand> cmd,
            NSUInteger index, id<MTLBuffer> buffer, NSUInteger offset=0u) {
            [cmd setKernelBuffer:buffer offset:offset atIndex:index];
        };
        const auto bindScalar = [&](id<MTLIndirectComputeCommand> cmd,
            NSUInteger index, NSUInteger offset) {
            bind(cmd,index,destinationConstantsBuffer,offset);
        };
        const auto dispatch = [&](id<MTLIndirectComputeCommand> cmd,
            NSUInteger gx, NSUInteger gy, NSUInteger gz, NSUInteger width) {
            [cmd concurrentDispatchThreadgroups:MTLSizeMake(gx,gy,gz)
                threadsPerThreadgroup:MTLSizeMake(width,1u,1u)];
            [cmd setBarrier];
        };

        auto cmd = beginCommand(reduceTilesICB);
        bindScalar(cmd,0u,offsetof(Selector6ICBConstants,partitionedDispatch));
        bind(cmd,1u,runs[runIndex].input); bind(cmd,2u,regionalBasis);
        bind(cmd,3u,globalNodeIndices); bind(cmd,4u,mass);
        bind(cmd,5u,partials); bind(cmd,6u,basisModes);
        bind(cmd,7u,regionSupportRanges); bind(cmd,8u,regionSupportRows);
        dispatch(cmd,1u,1u,1u,256u);

        cmd = beginCommand(reducePartialsICB);
        bindScalar(cmd,0u,offsetof(Selector6ICBConstants,partitionedDispatch));
        bind(cmd,1u,partials); bind(cmd,2u,reduced); bind(cmd,3u,basisModes);
        dispatch(cmd,1u,1u,1u,256u);

        cmd = beginCommand(solveBlocksICB);
        bindScalar(cmd,0u,offsetof(Selector6ICBConstants,partitionedDispatch));
        bind(cmd,1u,reduced); bind(cmd,2u,basisModes); bind(cmd,3u,lu);
        bind(cmd,4u,pivots); bind(cmd,5u,regionValid); bind(cmd,6u,factorValid);
        bind(cmd,7u,coefficients); bind(cmd,8u,solveValid);
        dispatch(cmd,1u,1u,1u,256u);

        cmd = beginCommand(addCorrectionICB);
        bindScalar(cmd,0u,offsetof(Selector6ICBConstants,partitionedDispatch));
        bind(cmd,1u,globalNodeIndices); bind(cmd,2u,regionalBasis);
        bind(cmd,3u,basisModes); bind(cmd,4u,coefficients);
        bind(cmd,5u,solveValid); bind(cmd,6u,runs[runIndex].output);
        bind(cmd,7u,nodeModeRanges); bind(cmd,8u,nodeModeSlots);
        dispatch(cmd,1u,1u,1u,256u);

        cmd = beginCommand(solvePatchesICB);
        bind(cmd,1u,graphPatches); bind(cmd,2u,graphPatchNodes);
        bind(cmd,3u,runs[runIndex].input); bind(cmd,4u,regionalBasis);
        bind(cmd,5u,regionalAP); bind(cmd,6u,coefficients);
        bind(cmd,7u,graphFactors); bind(cmd,8u,graphPivots);
        bind(cmd,9u,graphFactorValid); bind(cmd,10u,graphCorrections);
        bind(cmd,11u,statusBuffer);
        bindScalar(cmd,12u,offsetof(Selector6ICBConstants,graphPatches));
        bindScalar(cmd,13u,offsetof(Selector6ICBConstants,graphPatchNodes));
        bindScalar(cmd,14u,offsetof(Selector6ICBConstants,graphMatrixFloats));
        bindScalar(cmd,15u,offsetof(Selector6ICBConstants,graphPivots));
        bindScalar(cmd,16u,offsetof(Selector6ICBConstants,graphModeCount));
        bind(cmd,17u,stateBuffer); bind(cmd,18u,graphOriginal);
        bindScalar(cmd,19u,offsetof(Selector6ICBConstants,graphRefinementMode));
        bind(cmd,20u,candidate);
        bindScalar(cmd,21u,offsetof(Selector6ICBConstants,graphModeStride));
        bind(cmd,22u,dummy);
        bindScalar(cmd,23u,offsetof(Selector6ICBConstants,graphNoActiveContactEligibility));
        dispatch(cmd,1u,1u,1u,256u);

        cmd = beginCommand(applyPatchesICB);
        bind(cmd,1u,graphPatchNodes); bind(cmd,2u,nodePatchOffsets);
        bind(cmd,3u,nodePatchValues); bind(cmd,4u,nodeCoverage);
        bind(cmd,5u,regionalBasis); bind(cmd,6u,coefficients);
        bind(cmd,7u,graphCorrections); bind(cmd,8u,runs[runIndex].output);
        bind(cmd,9u,stateBuffer);
        bindScalar(cmd,10u,offsetof(Selector6ICBConstants,graphPatchNodes));
        bindScalar(cmd,11u,offsetof(Selector6ICBConstants,graphModeCount));
        bind(cmd,12u,candidate);
        bindScalar(cmd,13u,offsetof(Selector6ICBConstants,graphModeStride));
        dispatch(cmd,1u,1u,1u,256u);
        require(ordinal == commandCount,
            "selector-6 prefix ICB did not record exactly six commands");
    }

    const auto resetBuffer = [&](id<MTLBuffer> buffer, const auto& values) {
        require([buffer contents] != nullptr &&
                [buffer length] == values.size()*sizeof(values[0]),
            "selector-6 fixture reset buffer shape mismatch");
        std::memcpy([buffer contents], values.data(), [buffer length]);
    };
    const auto direct1D = [&](id<MTLComputePipelineState> p,
        const std::vector<std::pair<NSUInteger,id<MTLBuffer>>>& bindings,
        NSUInteger groups, NSUInteger width) {
        dispatch(device, queue, p, bindings, groups, width);
    };
    const auto runDirect = [&](std::size_t runIndex) {
        auto blockBuffer = makeBuffer(device,
            std::vector<NMPartitionedReductionBlockDispatchGPU>{block});
        dispatchGrid3D(device, queue, reduceTiles,
            {{0u,blockBuffer},{1u,runs[runIndex].input},{2u,regionalBasis},
             {3u,globalNodeIndices},{4u,mass},{5u,partials},{6u,basisModes},
             {7u,regionSupportRanges},{8u,regionSupportRows}},1u,1u,1u,256u);
        dispatchGrid3D(device, queue, reducePartials,
            {{0u,blockBuffer},{1u,partials},{2u,reduced},{3u,basisModes}},
            1u,1u,1u,256u);
        direct1D(solveBlocks,
            {{0u,blockBuffer},{1u,reduced},{2u,basisModes},{3u,lu},
             {4u,pivots},{5u,regionValid},{6u,factorValid},
             {7u,coefficients},{8u,solveValid}},1u,256u);
        direct1D(addCorrection,
            {{0u,blockBuffer},{1u,globalNodeIndices},{2u,regionalBasis},
             {3u,basisModes},{4u,coefficients},{5u,solveValid},
             {6u,runs[runIndex].output},{7u,nodeModeRanges},
             {8u,nodeModeSlots}},1u,256u);
        direct1D(solvePatches,
            {{0u,matterBuffer},{1u,graphPatches},{2u,graphPatchNodes},
             {3u,runs[runIndex].input},{4u,regionalBasis},{5u,regionalAP},
             {6u,coefficients},{7u,graphFactors},{8u,graphPivots},
             {9u,graphFactorValid},{10u,graphCorrections},{11u,statusBuffer},
             {12u,makeBuffer(device,std::vector<std::uint32_t>{1u})},
             {13u,makeBuffer(device,std::vector<std::uint32_t>{1u})},
             {14u,makeBuffer(device,std::vector<std::uint32_t>{9u})},
             {15u,makeBuffer(device,std::vector<std::uint32_t>{9u})},
             {16u,makeBuffer(device,std::vector<std::uint32_t>{0u})},
             {17u,stateBuffer},{18u,graphOriginal},
             {19u,makeBuffer(device,std::vector<std::uint32_t>{0u})},
             {20u,candidate},{21u,makeBuffer(device,std::vector<std::uint32_t>{0u})},
             {22u,dummy},{23u,makeBuffer(device,std::vector<std::uint32_t>{0u})}},
            1u,256u);
        direct1D(applyPatches,
            {{0u,matterBuffer},{1u,graphPatchNodes},{2u,nodePatchOffsets},
             {3u,nodePatchValues},{4u,nodeCoverage},{5u,regionalBasis},
             {6u,coefficients},{7u,graphCorrections},{8u,runs[runIndex].output},
             {9u,stateBuffer},{10u,makeBuffer(device,std::vector<std::uint32_t>{1u})},
             {11u,makeBuffer(device,std::vector<std::uint32_t>{0u})},
             {12u,candidate},{13u,makeBuffer(device,std::vector<std::uint32_t>{0u})}},
            1u,256u);
    };
    const auto snapshot = [&](id<MTLBuffer> buffer) {
        std::vector<std::uint8_t> bytes([buffer length]);
        std::memcpy(bytes.data(), [buffer contents], bytes.size());
        return bytes;
    };
    const auto prepareAndExecute = [&](std::size_t runIndex) {
        auto sourceConstants = constants;
        sourceConstants.micro.solverIteration =
            static_cast<std::uint32_t>(runIndex + 1u);
        resetBuffer(sourceConstantsBuffer,
            std::vector<Selector6ICBConstants>{sourceConstants});
        const auto words = std::vector<std::uint32_t>{constantWordCount};
        auto wordsBuffer = makeBuffer(device,words);
        dispatch(device, queue, prepareRange, {
            {0u,matterBuffer},{1u,stateBuffer},{2u,statusBuffer},
            {3u,prefixCountBuffer},{4u,tailCountBuffer},{5u,rangeBuffer},
            {6u,sourceConstantsBuffer},{7u,destinationConstantsBuffer},
            {8u,wordsBuffer}},1u,1u);
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            command != nil ? [command computeCommandEncoder] : nil;
        require(command != nil && encoder != nil,
            "cannot allocate selector-6 prefix ICB encoder");
        for (id<MTLBuffer> resource : {
            matterBuffer,stateBuffer,statusBuffer,rangeBuffer,
            sourceConstantsBuffer,destinationConstantsBuffer,
            runs[runIndex].input,runs[runIndex].output,
            globalNodeIndices,basisModes,regionSupportRanges,regionSupportRows,
            nodeModeRanges,nodeModeSlots,mass,partials,reduced,lu,pivots,
            regionValid,factorValid,coefficients,solveValid,regionalBasis,
            regionalAP,graphPatches,graphPatchNodes,graphFactors,graphOriginal,
            graphPivots,graphFactorValid,graphCorrections,nodePatchOffsets,
            nodePatchValues,nodeCoverage,candidate,dummy})
            [encoder useResource:resource
                usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        [encoder useResource:runs[runIndex].graph usage:MTLResourceUsageRead];
        [encoder executeCommandsInBuffer:runs[runIndex].graph
            indirectBuffer:rangeBuffer indirectBufferOffset:0u];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
            command.error != nil
                ? std::string("selector-6 prefix ICB execution failed: ") +
                    [[command.error localizedDescription] UTF8String]
                : "selector-6 prefix ICB command buffer did not complete");
    };

    struct Snapshot {
        std::vector<std::uint8_t> partials,reduced,coefficients,solveValid,
            output,corrections,status,state;
    };
    std::array<Snapshot,2> directSnapshots{};
    for (std::size_t runIndex = 0u; runIndex < runs.size(); ++runIndex) {
        resetBuffer(partials,std::vector<float>{-99.0f});
        resetBuffer(reduced,std::vector<float>{-99.0f});
        resetBuffer(coefficients,std::vector<float>{-99.0f});
        resetBuffer(solveValid,std::vector<std::uint32_t>{0u});
        resetBuffer(graphCorrections,std::vector<nm_float4>{
            nm_float4{-77.0f,-77.0f,-77.0f,-77.0f}});
        resetBuffer(stateBuffer,std::vector<NMFGMRESStateGPU>{NMFGMRESStateGPU{}});
        resetBuffer(statusBuffer,std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
        resetBuffer(runs[runIndex].output,std::vector<nm_float4>{
            nm_float4{0.25f,0.5f,0.75f,0.0f}});
        runDirect(runIndex);
        directSnapshots[runIndex] = Snapshot{
            snapshot(partials),snapshot(reduced),snapshot(coefficients),
            snapshot(solveValid),snapshot(runs[runIndex].output),
            snapshot(graphCorrections),snapshot(statusBuffer),snapshot(stateBuffer)};

        resetBuffer(partials,std::vector<float>{-99.0f});
        resetBuffer(reduced,std::vector<float>{-99.0f});
        resetBuffer(coefficients,std::vector<float>{-99.0f});
        resetBuffer(solveValid,std::vector<std::uint32_t>{0u});
        resetBuffer(graphCorrections,std::vector<nm_float4>{
            nm_float4{-77.0f,-77.0f,-77.0f,-77.0f}});
        resetBuffer(stateBuffer,std::vector<NMFGMRESStateGPU>{NMFGMRESStateGPU{}});
        resetBuffer(statusBuffer,std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
        resetBuffer(runs[runIndex].output,std::vector<nm_float4>{
            nm_float4{0.25f,0.5f,0.75f,0.0f}});
        prepareAndExecute(runIndex);
        const auto range = readBuffer<std::uint32_t>(rangeBuffer,4u);
        require(range[0]==0u && range[1]==commandCount &&
                range[2]==0u && range[3]==0u,
            "selector-6 ICB did not execute the six-command active prefix");
        require(snapshot(partials)==directSnapshots[runIndex].partials &&
                snapshot(reduced)==directSnapshots[runIndex].reduced &&
                snapshot(coefficients)==directSnapshots[runIndex].coefficients &&
                snapshot(solveValid)==directSnapshots[runIndex].solveValid &&
                snapshot(runs[runIndex].output)==directSnapshots[runIndex].output &&
                snapshot(graphCorrections)==directSnapshots[runIndex].corrections &&
                snapshot(statusBuffer)==directSnapshots[runIndex].status &&
                snapshot(stateBuffer)==directSnapshots[runIndex].state,
            "selector-6 prefix ICB differs from direct dispatch for a basis column");
        require(readBuffer<std::uint32_t>(solveValid,1u)[0] == 1u &&
                std::abs(readBuffer<float>(coefficients,1u)[0]) > 0.1f &&
                std::abs(readBuffer<nm_float4>(runs[runIndex].output,1u)[0].x -
                         0.25f) > 0.1f,
            "selector-6 prefix ICB parity passed without running its sparse/graph work");
        const auto copied = readBuffer<Selector6ICBConstants>(
            destinationConstantsBuffer,1u)[0];
        require(copied.micro.solverIteration ==
                    static_cast<std::uint32_t>(runIndex+1u),
            "selector-6 ICB did not refresh per-iteration constants");
    }

    resetBuffer(stateBuffer,std::vector<NMFGMRESStateGPU>{
        NMFGMRESStateGPU{{1.0f,1.0f,1.0f,0.0f},{},{}}});
    resetBuffer(statusBuffer,std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
    resetBuffer(partials,std::vector<float>{123.0f});
    resetBuffer(reduced,std::vector<float>{124.0f});
    resetBuffer(coefficients,std::vector<float>{125.0f});
    resetBuffer(solveValid,std::vector<std::uint32_t>{126u});
    resetBuffer(graphCorrections,std::vector<nm_float4>{
        nm_float4{127.0f,128.0f,129.0f,130.0f}});
    resetBuffer(runs[1].output,std::vector<nm_float4>{
        nm_float4{131.0f,132.0f,133.0f,134.0f}});
    const auto beforePartial = snapshot(partials);
    const auto beforeReduced = snapshot(reduced);
    const auto beforeCoeff = snapshot(coefficients);
    const auto beforeSolveValid = snapshot(solveValid);
    const auto beforeCorrection = snapshot(graphCorrections);
    const auto beforeOutput = snapshot(runs[1].output);
    prepareAndExecute(1u);
    const auto emptyRange = readBuffer<std::uint32_t>(rangeBuffer,4u);
    require(emptyRange[0]==0u && emptyRange[1]==0u &&
            emptyRange[2]==0u && emptyRange[3]==0u,
        "converged selector-6 prefix ICB did not select an empty range");
    require(snapshot(partials)==beforePartial && snapshot(reduced)==beforeReduced &&
            snapshot(coefficients)==beforeCoeff &&
            snapshot(solveValid)==beforeSolveValid &&
            snapshot(graphCorrections)==beforeCorrection &&
            snapshot(runs[1].output)==beforeOutput,
        "converged selector-6 ICB range modified scratch buffers");
    std::cout << "selector6_prefix_icb=true commands=6 columns=2 "
                 "iteration_refresh=true converged_range=empty direct_parity=bitwise "
                 "result=PASS\n";
}



void checkSelector6VascularPrefixICB(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLLibrary> library
) {
    constexpr std::uint32_t columnCount = 2u;
    constexpr std::uint32_t vectorCount = 8u;
    constexpr std::uint32_t vascularCount = 3u;
    constexpr std::uint32_t vascularBase = 2u;
    constexpr std::uint32_t commandCapacity = 2u;
    const NSUInteger vectorBytes = vectorCount * sizeof(nm_float4);

    NMMatterDispatchGPU matter{};
    matter.abiVersion = NM_MATTER_ABI_VERSION;
    matter.environmentCount = 1u;
    matter.femNodeCount = 2u;
    NMFGMRESLayoutGPU layout{};
    layout.unknownCount = vectorCount;
    layout.vascularBase = vascularBase;
    layout.vascularUnknownCount = vascularCount;
    NMVascularLayoutGPU vascular{};
    vascular.counts = {1u,1u,0u,0u};
    vascular.ranges = {0u,0u,vascularCount,0u};
    vascular.offsets = {0u,1u,2u,2u};
    vascular.cavities = {1u,0u,2u,0u};

    const auto precondition = pipeline(device, library,
        "numi_matter_metal::nm_vascular_precondition");
    const auto preconditionICB = indirectPipeline(device, library,
        "numi_matter_metal::nm_vascular_precondition");
    const auto principal = pipeline(device, library,
        "numi_matter_metal::nm_vascular_apply_principal_block");
    const auto principalICB = indirectPipeline(device, library,
        "numi_matter_metal::nm_vascular_apply_principal_block");
    const auto prepareRanges = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_prepare_operator_icb_ranges");

    const std::vector<NMVascularUnknownGPU> unknownValues{
        NMVascularUnknownGPU{{0.0f,4.0f,2.0f,0.0f}},
        NMVascularUnknownGPU{{0.0f,3.0f,1.0f,0.0f}},
        NMVascularUnknownGPU{{0.0f,1.0f,1.0f,0.0f}}};
    const std::vector<NMVascularConnectionGPU> connectionValues{
        NMVascularConnectionGPU{{0u,0u,0u,0u},{2.0f,0.5f,0.0f,0.0f},{}}};
    const auto dispatchBuffer = makeBuffer(device,
        std::vector<NMMatterDispatchGPU>{matter});
    const auto vascularLayoutBuffer = makeBuffer(device,
        std::vector<NMVascularLayoutGPU>{vascular});
    const auto fgmresLayoutBuffer = makeBuffer(device,
        std::vector<NMFGMRESLayoutGPU>{layout});
    const auto unknownBuffer = makeBuffer(device, unknownValues);
    const auto connectionBuffer = makeBuffer(device, connectionValues);
    const auto inputBuffer = makeBuffer(device,
        std::vector<nm_float4>(columnCount * vectorCount));
    auto* input = static_cast<nm_float4*>([inputBuffer contents]);
    for (std::uint32_t column = 0u; column < columnCount; ++column) {
        const auto base = std::size_t(column) * vectorCount;
        input[base+0u] = {101.0f+float(column),102.0f,103.0f,104.0f};
        input[base+1u] = {201.0f,202.0f+float(column),203.0f,204.0f};
        input[base+vascularBase+0u] = {1.25f+float(column),0.0f,0.0f,0.0f};
        input[base+vascularBase+1u] = {-2.5f+0.5f*float(column),0.0f,0.0f,0.0f};
        input[base+vascularBase+2u] = {3.75f-float(column),0.0f,0.0f,0.0f};
        input[base+5u] = {301.0f,302.0f,303.0f,304.0f};
        input[base+6u] = {401.0f,402.0f,403.0f,404.0f};
        input[base+7u] = {501.0f,502.0f,503.0f,504.0f};
    }
    const nm_float4 sentinel{71.0f,72.0f,73.0f,74.0f};
    const auto sentinelValues = [&] {
        return std::vector<nm_float4>(columnCount * vectorCount, sentinel);
    };
    const auto outputBuffer = makeBuffer(device, sentinelValues());
    const auto stateBuffer = makeBuffer(device,
        std::vector<NMFGMRESStateGPU>{NMFGMRESStateGPU{}});
    const auto statusBuffer = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
    const auto candidateBuffer = makeBuffer(device,
        std::vector<nm_float4>(vascularCount, nm_float4{}));
    const auto tissueBuffer = makeBuffer(device,
        std::vector<NMVascularTissueGPU>{NMVascularTissueGPU{}});
    const auto exchangeBuffer = makeBuffer(device,
        std::vector<NMVascularExchangeGPU>{NMVascularExchangeGPU{}});
    const auto incidenceBuffer = makeBuffer(device,std::vector<std::uint32_t>{0u});
    const auto rangesBuffer = makeBuffer(device,
        std::vector<NMVascularRangeGPU>{NMVascularRangeGPU{}});
    const auto compartmentBuffer = makeBuffer(device,
        std::vector<NMVascularCompartmentGPU>{NMVascularCompartmentGPU{}});
    const auto elastanceBuffer = makeBuffer(device,std::vector<float>{0.0f});
    const auto workingSetBuffer = makeBuffer(device,
        std::vector<std::uint32_t>(vascularCount,0u));
    const auto cavityBuffer = makeBuffer(device,
        std::vector<NMVascularCavityGPU>{NMVascularCavityGPU{}});
    const auto compartmentCavityBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{NM_INVALID_INDEX});
    const auto cavityDiagonalBuffer = makeBuffer(device,std::vector<float>{4.0f});

    // A nontrivial 3x3 LU with identity pivots is the same kind of factor
    // buffer produced once per Newton by nm_vascular_factor_principal_block.
    std::vector<float> factors(64u*64u,0.0f);
    factors[0u] = 2.0f; factors[1u] = 1.0f;
    factors[64u] = 0.5f; factors[65u] = 2.5f; factors[66u] = 1.0f;
    factors[129u] = 0.4f; factors[130u] = 3.6f;
    const auto factorBuffer = makeBuffer(device,factors);
    std::vector<std::uint32_t> pivots(64u);
    for (std::uint32_t i=0u;i<64u;++i) pivots[i]=i;
    const auto pivotBuffer = makeBuffer(device,pivots);
    const auto factorValidBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{1u});
    const auto prefixCountBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{1u});
    const auto tailCountBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u});
    const auto rangeBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u,0u,0u,0u});

    Selector6ICBConstants constants{};
    constants.dispatch = matter;
    constants.layout = layout;
    constants.vascularLayout = vascular;
    constants.useWorkingSet = 1u;
    const auto sourceConstantsBuffer = makeBuffer(device,
        std::vector<Selector6ICBConstants>{constants});
    const auto destinationConstantsBuffer = makeBuffer(device,
        std::vector<Selector6ICBConstants>{constants});
    const std::uint32_t constantWords =
        static_cast<std::uint32_t>(sizeof(constants)/sizeof(std::uint32_t));
    const auto constantWordsBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{constantWords});

    MTLIndirectCommandBufferDescriptor* descriptor =
        [MTLIndirectCommandBufferDescriptor new];
    descriptor.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
    descriptor.inheritBuffers = NO;
    descriptor.inheritPipelineState = NO;
    descriptor.maxKernelBufferBindCount = 31u;

    auto encodeCase = [&](bool usePrincipal, std::uint32_t column) {
        const NSUInteger commandCount = usePrincipal ? 2u : 1u;
        auto graph = [device newIndirectCommandBufferWithDescriptor:descriptor
            maxCommandCount:commandCapacity options:MTLResourceStorageModePrivate];
        require(graph != nil, "vascular prefix ICB allocation failed");
        const NSUInteger byteOffset = std::size_t(column) * vectorBytes;
        NSUInteger ordinal = 0u;
        auto begin = [&](id<MTLComputePipelineState> p) {
            require(ordinal < commandCapacity,
                "vascular prefix ICB command capacity exceeded");
            auto cmd = [graph indirectComputeCommandAtIndex:ordinal++];
            [cmd setComputePipelineState:p];
            [cmd setKernelBuffer:destinationConstantsBuffer
                offset:offsetof(Selector6ICBConstants,dispatch) atIndex:0u];
            [cmd setKernelBuffer:destinationConstantsBuffer
                offset:offsetof(Selector6ICBConstants,layout) atIndex:30u];
            return cmd;
        };
        auto bind = [&](id<MTLIndirectComputeCommand> cmd, NSUInteger index,
                        id<MTLBuffer> buffer, NSUInteger offset=0u) {
            [cmd setKernelBuffer:buffer offset:offset atIndex:index];
        };
        auto bindScalar = [&](id<MTLIndirectComputeCommand> cmd,
                              NSUInteger index, NSUInteger offset) {
            bind(cmd,index,destinationConstantsBuffer,offset);
        };
        auto cmd = begin(preconditionICB);
        bindScalar(cmd,1u,offsetof(Selector6ICBConstants,vascularLayout));
        bindScalar(cmd,2u,offsetof(Selector6ICBConstants,micro));
        bind(cmd,3u,unknownBuffer); bind(cmd,4u,connectionBuffer);
        bind(cmd,5u,inputBuffer,byteOffset); bind(cmd,6u,outputBuffer,byteOffset);
        bind(cmd,7u,stateBuffer); bind(cmd,8u,candidateBuffer);
        bind(cmd,9u,tissueBuffer); bind(cmd,10u,exchangeBuffer);
        bind(cmd,11u,incidenceBuffer); bind(cmd,12u,rangesBuffer);
        bind(cmd,13u,incidenceBuffer); bind(cmd,14u,rangesBuffer);
        bind(cmd,15u,incidenceBuffer); bind(cmd,16u,rangesBuffer);
        bind(cmd,17u,statusBuffer); bind(cmd,18u,compartmentBuffer);
        bind(cmd,19u,elastanceBuffer); bind(cmd,20u,workingSetBuffer);
        bind(cmd,21u,cavityBuffer); bind(cmd,22u,compartmentCavityBuffer);
        bind(cmd,23u,cavityDiagonalBuffer);
        [cmd concurrentDispatchThreadgroups:MTLSizeMake(1u,1u,1u)
            threadsPerThreadgroup:MTLSizeMake(32u,1u,1u)];
        [cmd setBarrier];
        if (usePrincipal) {
            cmd = begin(principalICB);
            bindScalar(cmd,1u,offsetof(Selector6ICBConstants,vascularLayout));
            bindScalar(cmd,2u,offsetof(Selector6ICBConstants,layout));
            bind(cmd,3u,inputBuffer,byteOffset); bind(cmd,4u,outputBuffer,byteOffset);
            bind(cmd,5u,stateBuffer); bind(cmd,6u,workingSetBuffer);
            bind(cmd,7u,factorBuffer); bind(cmd,8u,pivotBuffer);
            bind(cmd,9u,factorValidBuffer);
            bindScalar(cmd,10u,offsetof(Selector6ICBConstants,useWorkingSet));
            [cmd concurrentDispatchThreadgroups:MTLSizeMake(1u,1u,1u)
                threadsPerThreadgroup:MTLSizeMake(32u,1u,1u)];
            [cmd setBarrier];
        }
        require(ordinal==commandCount,
            "vascular prefix ICB recorded the wrong number of per-column consumers");
        return graph;
    };

    auto resetState = [&](float converged) {
        NMFGMRESStateGPU state{};
        state.diagnostics.z=converged;
        std::memcpy([stateBuffer contents],&state,sizeof(state));
        NMMatterStatusGPU status{};
        status.code=NM_STATUS_SUCCESS;
        std::memcpy([statusBuffer contents],&status,sizeof(status));
    };
    auto runDirect = [&](bool usePrincipal, std::uint32_t column,
                         const NMMicrostepGPU& micro) {
        const NSUInteger byteOffset = std::size_t(column) * vectorBytes;
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=
            command!=nil?[command computeCommandEncoder]:nil;
        require(command!=nil && encoder!=nil,
            "cannot allocate direct vascular preconditioner encoder");
        [encoder setComputePipelineState:precondition];
        [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
        [encoder setBuffer:vascularLayoutBuffer offset:0u atIndex:1u];
        [encoder setBytes:&micro length:sizeof(micro) atIndex:2u];
        [encoder setBuffer:unknownBuffer offset:0u atIndex:3u];
        [encoder setBuffer:connectionBuffer offset:0u atIndex:4u];
        [encoder setBuffer:inputBuffer offset:byteOffset atIndex:5u];
        [encoder setBuffer:outputBuffer offset:byteOffset atIndex:6u];
        [encoder setBuffer:stateBuffer offset:0u atIndex:7u];
        [encoder setBuffer:candidateBuffer offset:0u atIndex:8u];
        [encoder setBuffer:tissueBuffer offset:0u atIndex:9u];
        [encoder setBuffer:exchangeBuffer offset:0u atIndex:10u];
        [encoder setBuffer:incidenceBuffer offset:0u atIndex:11u];
        [encoder setBuffer:rangesBuffer offset:0u atIndex:12u];
        [encoder setBuffer:incidenceBuffer offset:0u atIndex:13u];
        [encoder setBuffer:rangesBuffer offset:0u atIndex:14u];
        [encoder setBuffer:incidenceBuffer offset:0u atIndex:15u];
        [encoder setBuffer:rangesBuffer offset:0u atIndex:16u];
        [encoder setBuffer:statusBuffer offset:0u atIndex:17u];
        [encoder setBuffer:compartmentBuffer offset:0u atIndex:18u];
        [encoder setBuffer:elastanceBuffer offset:0u atIndex:19u];
        [encoder setBuffer:workingSetBuffer offset:0u atIndex:20u];
        [encoder setBuffer:cavityBuffer offset:0u atIndex:21u];
        [encoder setBuffer:compartmentCavityBuffer offset:0u atIndex:22u];
        [encoder setBuffer:cavityDiagonalBuffer offset:0u atIndex:23u];
        [encoder setBuffer:fgmresLayoutBuffer offset:0u atIndex:30u];
        [encoder dispatchThreads:MTLSizeMake(vascularCount,1u,1u)
            threadsPerThreadgroup:MTLSizeMake(32u,1u,1u)];
        if (usePrincipal) {
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            const std::uint32_t useWorkingSet=1u;
            [encoder setComputePipelineState:principal];
            [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
            [encoder setBuffer:vascularLayoutBuffer offset:0u atIndex:1u];
            [encoder setBuffer:fgmresLayoutBuffer offset:0u atIndex:2u];
            [encoder setBuffer:inputBuffer offset:byteOffset atIndex:3u];
            [encoder setBuffer:outputBuffer offset:byteOffset atIndex:4u];
            [encoder setBuffer:stateBuffer offset:0u atIndex:5u];
            [encoder setBuffer:workingSetBuffer offset:0u atIndex:6u];
            [encoder setBuffer:factorBuffer offset:0u atIndex:7u];
            [encoder setBuffer:pivotBuffer offset:0u atIndex:8u];
            [encoder setBuffer:factorValidBuffer offset:0u atIndex:9u];
            [encoder setBytes:&useWorkingSet length:sizeof(useWorkingSet) atIndex:10u];
            [encoder dispatchThreadgroups:MTLSizeMake(1u,1u,1u)
                threadsPerThreadgroup:MTLSizeMake(32u,1u,1u)];
        }
        [encoder endEncoding];
        [command commit]; [command waitUntilCompleted];
        require(command.status==MTLCommandBufferStatusCompleted,
            "direct vascular preconditioner command failed");
    };

    auto runICB = [&](id<MTLIndirectCommandBuffer> graph, std::uint32_t count,
                      std::uint32_t column, const NMMicrostepGPU& micro) {
        auto updated=constants;
        updated.micro=micro;
        auto* source=static_cast<Selector6ICBConstants*>(
            [sourceConstantsBuffer contents]);
        source[0]=updated;
        const auto countValues=std::vector<std::uint32_t>{count};
        std::memcpy([prefixCountBuffer contents],countValues.data(),sizeof(std::uint32_t));
        const auto words=std::vector<std::uint32_t>{constantWords};
        dispatch(device,queue,prepareRanges,{
            {0u,dispatchBuffer},{1u,stateBuffer},{2u,statusBuffer},
            {3u,prefixCountBuffer},{4u,tailCountBuffer},{5u,rangeBuffer},
            {6u,sourceConstantsBuffer},{7u,destinationConstantsBuffer},
            {8u,constantWordsBuffer}},1u,1u);
        const auto activeRange=readBuffer<std::uint32_t>(rangeBuffer,4u);
        require(activeRange[0]==0u && activeRange[1]==count &&
                activeRange[2]==0u && activeRange[3]==0u,
            "vascular prefix ICB did not publish the expected active range");
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=
            command!=nil?[command computeCommandEncoder]:nil;
        require(command!=nil && encoder!=nil,
            "cannot allocate vascular prefix ICB execution encoder");
        for (id<MTLBuffer> resource : {dispatchBuffer,vascularLayoutBuffer,fgmresLayoutBuffer,unknownBuffer,
             connectionBuffer,inputBuffer,outputBuffer,stateBuffer,candidateBuffer,
             tissueBuffer,exchangeBuffer,incidenceBuffer,rangesBuffer,statusBuffer,
             compartmentBuffer,elastanceBuffer,workingSetBuffer,cavityBuffer,
             compartmentCavityBuffer,cavityDiagonalBuffer,factorBuffer,pivotBuffer,
             factorValidBuffer,destinationConstantsBuffer,rangeBuffer})
            [encoder useResource:resource
                usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        [encoder useResource:graph usage:MTLResourceUsageRead];
        [encoder executeCommandsInBuffer:graph
            indirectBuffer:rangeBuffer indirectBufferOffset:0u];
        [encoder endEncoding];
        [command commit]; [command waitUntilCompleted];
        require(command.status==MTLCommandBufferStatusCompleted,
            "vascular prefix ICB execution failed");
        (void)column;
        (void)words;
    };

    for (bool usePrincipal : {false,true}) {
        for (std::uint32_t column=0u;column<columnCount;++column) {
            resetState(0.0f);
            auto outputValues=sentinelValues();
            std::memcpy([outputBuffer contents],outputValues.data(),
                outputValues.size()*sizeof(nm_float4));
            NMMicrostepGPU micro{};
            micro.time={column==0u?0.125f:0.25f,column==0u?8.0f:4.0f,0.0f,0.0f};
            const float cavitySchur=column==0u?4.0f:8.0f;
            std::memcpy([cavityDiagonalBuffer contents],&cavitySchur,sizeof(float));
            runDirect(usePrincipal,column,micro);
            const auto direct=readBuffer<nm_float4>(outputBuffer,columnCount*vectorCount);

            resetState(0.0f);
            std::memcpy([outputBuffer contents],outputValues.data(),
                outputValues.size()*sizeof(nm_float4));
            auto graph=encodeCase(usePrincipal,column);
            runICB(graph,usePrincipal?2u:1u,column,micro);
            const auto indirect=readBuffer<nm_float4>(outputBuffer,columnCount*vectorCount);
            require(std::memcmp(direct.data(),indirect.data(),
                    direct.size()*sizeof(nm_float4))==0,
                "vascular preconditioner ICB differs from direct per-column correction");
            const std::size_t base=std::size_t(column)*vectorCount+vascularBase;
            if (!usePrincipal) {
                const float flowDiagonal=3.0f*(2.0f+0.5f/micro.time.x);
                require(std::abs(indirect[base+1u].x-
                        input[base+1u].x/flowDiagonal)<1e-6f,
                    "vascular preconditioner ignored timestep-dependent flow scaling");
                require(std::abs(indirect[base+2u].x-
                        input[base+2u].x/cavitySchur)<1e-6f,
                    "vascular preconditioner ignored the prepared cavity-Schur diagonal");
            } else {
                const float b0=input[base+0u].x;
                const float b1=input[base+1u].x;
                const float b2=input[base+2u].x;
                const float y0=b0, y1=b1-0.5f*y0, y2=b2-0.4f*y1;
                const float x2=y2/3.6f, x1=(y1-x2)/2.5f, x0=(y0-x1)/2.0f;
                require(std::abs(indirect[base+0u].x-x0)<2e-6f &&
                        std::abs(indirect[base+1u].x-x1)<2e-6f &&
                        std::abs(indirect[base+2u].x-x2)<2e-6f,
                    "vascular ICB did not consume the current principal factors and pivots");
            }
            for (std::uint32_t row=0u;row<vectorCount;++row) {
                if (row>=vascularBase && row<vascularBase+vascularCount) continue;
                const auto at=std::size_t(column)*vectorCount+row;
                require(std::memcmp(&indirect[at],&sentinel,sizeof(sentinel))==0,
                    "vascular preconditioner ICB modified a nonvascular row");
            }
        }

        // A completed FGMRES range is empty before executing either vascular
        // consumer, so stale or cached factors cannot touch the output vector.
        resetState(1.0f);
        auto outputValues=sentinelValues();
        std::memcpy([outputBuffer contents],outputValues.data(),
            outputValues.size()*sizeof(nm_float4));
        NMMicrostepGPU micro{};
        micro.time={0.125f,8.0f,0.0f,0.0f};
        auto graph=encodeCase(usePrincipal,0u);
        runICB(graph,0u,0u,micro);
        require(std::memcmp([outputBuffer contents],outputValues.data(),
                outputValues.size()*sizeof(nm_float4))==0,
            "completed vascular preconditioner range modified output");
    }
    std::cout << "selector6_vascular_prefix_icb=true "
                 "cavity_diagonal_dependency=true principal_factor_dependency=true "
                 "timestep_and_column_refresh=true completed_range=empty "
                 "direct_parity=bitwise result=PASS\n";
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


void checkAssembledXYZOperator(
    id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library) {
    auto specializedPipeline = [&](const char* name) {
        MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
        const BOOL disabled = NO;
        const std::uint32_t noMaterial = NM_INVALID_INDEX;
        for (NSUInteger index = 30u; index <= 34u; ++index)
            [constants setConstantValue:&disabled type:MTLDataTypeBool
                               atIndex:index];
        for (NSUInteger index : {35u, 36u})
            [constants setConstantValue:&noMaterial type:MTLDataTypeUInt
                               atIndex:index];
        NSError* error = nil;
        NSString* functionName = [NSString stringWithUTF8String:name];
        id<MTLFunction> function = [library newFunctionWithName:functionName
            constantValues:constants error:&error];
        require(function != nil, error != nil
            ? std::string("assembled checker cannot specialize ") + name +
                ": " + [[error localizedDescription] UTF8String]
            : std::string("assembled checker cannot find ") + name);
        id<MTLComputePipelineState> result =
            [device newComputePipelineStateWithFunction:function error:&error];
        require(result != nil, error != nil
            ? std::string("assembled checker cannot create ") + name +
                ": " + [[error localizedDescription] UTF8String]
            : std::string("assembled checker cannot create ") + name);
        return result;
    };
    auto tangentBuild = specializedPipeline(
        "numi_matter_metal::nm_fem_build_algorithmic_tangent_cache");
    auto tangentPrepareColumns = pipeline(device, library,
        "numi_matter_metal::nm_fem_prepare_algorithmic_tangent_columns");
    auto tangentBuildColumns = specializedPipeline(
        "numi_matter_metal::nm_fem_build_algorithmic_tangent_cache_columns");
    auto tangentFinalizeColumns = specializedPipeline(
        "numi_matter_metal::nm_fem_finalize_algorithmic_tangent_columns");
    auto directOperator = specializedPipeline(
        "numi_matter_metal::nm_fem_apply_operator_elements");
    auto assembleTet = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_assemble_tet_xyz_blocks");
    auto assembleCSR = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_assemble_xyz_block_csr");
    auto assembleCSREdges = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_assemble_xyz_block_csr_edges");
    auto applyCSR = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_apply_xyz_block_csr");
    auto applyCSRSimdRow = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_apply_xyz_block_csr_simdrow");

    constexpr std::uint32_t environments = 2u;
    constexpr std::uint32_t nodesPerEnvironment = 5u;
    constexpr std::uint32_t tetrahedraPerEnvironment = 2u;
    constexpr std::uint32_t incidencePerEnvironment = 8u;
    NMMatterDispatchGPU shape{};
    shape.abiVersion = NM_MATTER_ABI_VERSION;
    shape.environmentCount = environments;
    shape.objectCount = 1u;
    shape.materialCount = 1u;
    shape.parameterCount = 1u;
    shape.femNodeCount = nodesPerEnvironment;
    shape.tetrahedronCount = tetrahedraPerEnvironment;
    shape.maximumRateExponent = 1u;
    shape.materialStateStride = 1u;
    shape.gravityAndTimestep.w = 0.08f;

    NMContinuumObjectGPU object{};
    object.representation = NM_REPRESENTATION_FEM;
    object.materialIndex = 0u;
    object.flags = NM_OBJECT_ACTIVE | NM_OBJECT_FEM_MATERIAL_FRAME;
    object.solver.y = 0u;

    NMMaterialGPU material{};
    material.constitutiveKind = NM_CONSTITUTIVE_BYTECODE;
    material.parameterOffset = 0u;
    material.parameterCount = 1u;
    material.stateCount = 0u;
    material.stressProgramOffset = 0u;
    material.tangentProgramOffset = 9u;
    material.validityProgram = NM_INVALID_INDEX;
    material.stateInitialOffset = NM_INVALID_INDEX;
    material.stateUpdateProgramOffset = NM_INVALID_INDEX;
    material.dissipationProgram = NM_INVALID_INDEX;
    material.projectionKind = NM_MATERIAL_PROJECTION_GENERIC;
    material.viscousStressProgramOffset = NM_INVALID_INDEX;
    material.viscousTangentProgramOffset = NM_INVALID_INDEX;
    material.implicitResidualProgramOffset = NM_INVALID_INDEX;
    material.implicitJacobianProgramOffset = NM_INVALID_INDEX;
    material.implicitDeformationProgramOffset = NM_INVALID_INDEX;
    material.stressStateDerivativeProgramOffset = NM_INVALID_INDEX;
    material.bulk = {1.0f, 293.0f, 0.0f, 0.0f};

    // A compact bytecode law makes the production cache builder produce a
    // current, deformation-dependent, nonsymmetric material Jacobian. Each
    // tangent row uses a distinct F coefficient and cyclic DF input.
    std::vector<NMScalarProgramGPU> programs(18u);
    std::vector<NMExpressionInstructionGPU> instructions;
    instructions.reserve(36u);
    for (std::uint32_t row = 0u; row < 9u; ++row) {
        NMScalarProgramGPU& stress = programs[row];
        stress.firstInstruction = static_cast<std::uint32_t>(instructions.size());
        stress.instructionCount = 1u;
        stress.maximumStack = 1u;
        NMExpressionInstructionGPU f{};
        f.opcode = NM_EXPR_F;
        f.index = row;
        instructions.push_back(f);
    }
    for (std::uint32_t row = 0u; row < 9u; ++row) {
        NMScalarProgramGPU& tangent = programs[9u + row];
        tangent.firstInstruction = static_cast<std::uint32_t>(instructions.size());
        tangent.instructionCount = 3u;
        tangent.maximumStack = 2u;
        NMExpressionInstructionGPU f{};
        f.opcode = NM_EXPR_F;
        f.index = (row + 2u) % 9u;
        instructions.push_back(f);
        NMExpressionInstructionGPU df{};
        df.opcode = NM_EXPR_DF;
        df.index = (row + 1u) % 9u;
        instructions.push_back(df);
        NMExpressionInstructionGPU multiply{};
        multiply.opcode = NM_EXPR_MULTIPLY;
        instructions.push_back(multiply);
    }

    std::vector<NMSchedulerStateGPU> schedulers(environments);
    std::vector<NMAdaptiveStateGPU> adaptive(environments);
    std::vector<NMFGMRESStateGPU> solverStates(environments);
    std::vector<NMMatterStatusGPU> statuses(environments);
    std::vector<NMFEMNodeStateGPU> candidate(
        environments * nodesPerEnvironment);
    std::vector<NMTetrahedronGPU> tetrahedra(
        environments * tetrahedraPerEnvironment);
    std::vector<NMIncidenceRangeGPU> nodeRanges(
        environments * nodesPerEnvironment);
    std::vector<std::uint32_t> nodeIncidence(
        environments * incidencePerEnvironment);
    std::vector<nm_float4> direction(environments * nodesPerEnvironment);
    const std::array<nm_float4, nodesPerEnvironment> positions{{
        {0.00f, 0.00f, 0.00f, 0.0f},
        {1.10f, 0.05f, 0.00f, 0.0f},
        {0.10f, 0.90f, 0.02f, 0.0f},
        {0.05f, 0.05f, 1.20f, 0.0f},
        {1.00f, 0.95f, 1.00f, 0.0f},
    }};
    const std::array<std::array<std::uint32_t, 4u>,
                     tetrahedraPerEnvironment> tetNodes{{
        {{0u, 1u, 2u, 3u}},
        {{1u, 2u, 3u, 4u}},
    }};
    const std::array<std::uint32_t, nodesPerEnvironment> incidenceFirst{
        {0u, 1u, 3u, 5u, 7u}};
    const std::array<std::uint32_t, nodesPerEnvironment> incidenceCount{
        {1u, 2u, 2u, 2u, 1u}};
    const std::array<std::uint32_t, incidencePerEnvironment> incidence{
        {0u, 0u, 1u, 0u, 1u, 0u, 1u, 1u}};
    for (std::uint32_t environment = 0u; environment < environments;
         ++environment) {
        schedulers[environment].activeExponent = environment;
        schedulers[environment].numerical.w = 1.0f;
        adaptive[environment].activeRepresentation = NM_REPRESENTATION_FEM;
        statuses[environment].code = NM_STATUS_SUCCESS;
        statuses[environment].environment = environment;
        for (std::uint32_t node = 0u; node < nodesPerEnvironment; ++node) {
            const std::uint32_t global = environment * nodesPerEnvironment + node;
            NMFEMNodeStateGPU& value = candidate[global];
            value.positionAndMass = positions[node];
            value.positionAndMass.x += 0.015f * float(environment) *
                float(node + 1u);
            value.positionAndMass.y -= 0.009f * float(environment) *
                float(node + 2u);
            value.positionAndMass.w = 1.0f + 0.1f * float(node);
            value.velocityAndInverseMass.w = 1.0f /
                value.positionAndMass.w;
            value.restAndFixed.w = node == 0u ? 1.0f : 0.0f;
            nodeRanges[global] = {
                environment * incidencePerEnvironment + incidenceFirst[node],
                incidenceCount[node], 0u, 0u};
        }
        for (std::uint32_t i = 0u; i < incidencePerEnvironment; ++i)
            nodeIncidence[environment * incidencePerEnvironment + i] =
                incidence[i];
        for (std::uint32_t tetIndex = 0u;
             tetIndex < tetrahedraPerEnvironment; ++tetIndex) {
            NMTetrahedronGPU& tet = tetrahedra[
                environment * tetrahedraPerEnvironment + tetIndex];
            const auto& n = tetNodes[tetIndex];
            tet.nodes = {n[0], n[1], n[2], n[3]};
            if (tetIndex == 0u) {
                tet.inverseRestRow0 = {1.0f, 0.0f, 0.0f, 0.16666667f};
                tet.inverseRestRow1 = {0.0f, 1.0f, 0.0f, 0.0f};
                tet.inverseRestRow2 = {0.0f, 0.0f, 1.0f, 0.0f};
            } else {
                tet.inverseRestRow0 = {0.8f, 0.1f, 0.0f, 0.21f};
                tet.inverseRestRow1 = {0.05f, 1.1f, 0.1f, 0.0f};
                tet.inverseRestRow2 = {0.0f, 0.2f, 0.9f, 0.0f};
            }
            tet.identity = {0u, 0u, 0u, NM_OBJECT_ACTIVE};
            const float angle = tetIndex == 0u ? 0.31f : -0.27f;
            tet.materialFrameRotation = {
                0.0f, 0.0f, std::sin(0.5f * angle),
                std::cos(0.5f * angle)};
        }
        for (std::uint32_t node = 0u; node < nodesPerEnvironment; ++node) {
            const float e = float(environment + 1u);
            const float n = float(node + 1u);
            direction[environment * nodesPerEnvironment + node] = {
                std::sin(0.71f * e * n),
                std::cos(0.37f * (e + 2.0f) * n),
                std::sin(0.29f * (e + n) * (n + 1.0f)),
                0.0f};
        }
    }

    NMMicrostepGPU micro{};
    micro.microtick = 1u;
    micro.flags = NM_MICROSTEP_FGMRES_OPERATOR;
    micro.time = {shape.gravityAndTimestep.w,
                  1.0f / shape.gravityAndTimestep.w,
                  0.0f, shape.gravityAndTimestep.w};
    std::vector<float> parameters(environments * shape.parameterCount, 0.0f);
    std::vector<float> acceptedState(
        environments * tetrahedraPerEnvironment * shape.materialStateStride,
        0.0f);
    std::vector<float> tangentCache(
        environments * tetrahedraPerEnvironment * 81u, 0.0f);
    std::vector<std::uint32_t> cacheValid(
        environments * tetrahedraPerEnvironment, 0u);
    auto dispatchBuffer = makeBuffer(device, std::vector{shape});
    auto microBuffer = makeBuffer(device, std::vector{micro});
    auto objectsBuffer = makeBuffer(device, std::vector{object});
    auto materialsBuffer = makeBuffer(device, std::vector{material});
    auto programsBuffer = makeBuffer(device, programs);
    auto instructionsBuffer = makeBuffer(device, instructions);
    auto parametersBuffer = makeBuffer(device, parameters);
    auto candidateBuffer = makeBuffer(device, candidate);
    auto tetrahedraBuffer = makeBuffer(device, tetrahedra);
    auto acceptedStateBuffer = makeBuffer(device, acceptedState);
    auto schedulerBuffer = makeBuffer(device, schedulers);
    auto adaptiveBuffer = makeBuffer(device, adaptive);
    auto tangentCacheBuffer = makeBuffer(device, tangentCache);
    auto cacheValidBuffer = makeBuffer(device, cacheValid);
    auto statusBuffer = makeBuffer(device, statuses);
    auto solverStateBuffer = makeBuffer(device, solverStates);
    auto forceRebuildBuffer = makeBuffer(device, std::vector<std::uint32_t>{1u});
    auto includeMixedGraphXYZBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u});
    auto graphOnlyMixedCacheBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{1u});
    auto fullSourceSelectionBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u});
    auto selectedTetrahedronBaseBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u});
    auto fullSourceSelectionCountBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u});
    dispatch(device, queue, tangentBuild, {
        {0u, dispatchBuffer}, {1u, microBuffer}, {2u, objectsBuffer},
        {3u, materialsBuffer}, {4u, programsBuffer}, {5u, instructionsBuffer},
        {6u, parametersBuffer}, {7u, candidateBuffer}, {8u, tetrahedraBuffer},
        {9u, acceptedStateBuffer}, {10u, schedulerBuffer}, {11u, adaptiveBuffer},
        {12u, tangentCacheBuffer}, {13u, cacheValidBuffer}, {14u, statusBuffer},
        {15u, solverStateBuffer}, {16u, forceRebuildBuffer}},
        environments * tetrahedraPerEnvironment, 1u);
    cacheValid = readBuffer<std::uint32_t>(cacheValidBuffer,
        cacheValid.size());
    for (std::uint32_t valid : cacheValid)
        require(valid == 1u,
            "current bytecode tangent cache failed to build for active FEM tet");

    auto incidenceBuffer = makeBuffer(device, nodeIncidence);
    auto nodeRangesBuffer = makeBuffer(device, nodeRanges);
    auto cacheModeBuffer = makeBuffer(device, std::vector<std::uint32_t>{2u});
    const std::uint32_t elementBlockFloatCount =
        environments * tetrahedraPerEnvironment * 144u;
    auto elementBlocksBuffer = makeBuffer(device,
        std::vector<float>(elementBlockFloatCount, -91.0f));
    dispatch(device, queue, assembleTet, {
        {0u, dispatchBuffer}, {1u, microBuffer}, {2u, objectsBuffer},
        {3u, tetrahedraBuffer}, {4u, tangentCacheBuffer},
        {5u, cacheValidBuffer}, {6u, elementBlocksBuffer}, {7u, statusBuffer},
        {8u, schedulerBuffer}, {9u, adaptiveBuffer}, {10u, solverStateBuffer},
        {11u, cacheModeBuffer}, {12u, forceRebuildBuffer},
        {13u, includeMixedGraphXYZBuffer}},
        1u, 256u);

    // The tets share three vertices, so many element edges map to the same
    // global CSR entries. Upload compact, sorted unique adjacency as runtime
    // does; the GPU kernel fills each environment's nine-float edge blocks.
    std::array<std::set<std::uint32_t>, nodesPerEnvironment> neighbors;
    for (const auto& tet : tetNodes)
        for (std::uint32_t row : tet)
            for (std::uint32_t column : tet) neighbors[row].insert(column);
    std::uint32_t nnz = 0u;
    std::vector<NMIncidenceRangeGPU> csrRows(
        environments * nodesPerEnvironment);
    std::vector<std::uint32_t> csrColumns;
    for (std::uint32_t node = 0u; node < nodesPerEnvironment; ++node) {
        const NMIncidenceRangeGPU row{nnz,
            static_cast<std::uint32_t>(neighbors[node].size()), node, 0u};
        for (std::uint32_t environment = 0u; environment < environments;
             ++environment)
            csrRows[environment * nodesPerEnvironment + node] = row;
        nnz += static_cast<std::uint32_t>(neighbors[node].size());
        csrColumns.insert(csrColumns.end(), neighbors[node].begin(),
                          neighbors[node].end());
    }
    auto csrRowsBuffer = makeBuffer(device, csrRows);
    auto csrColumnsBuffer = makeBuffer(device, csrColumns);
    auto csrValuesBuffer = makeBuffer(device,
        std::vector<float>(environments * nnz * 9u, -73.0f));
    auto nnzBuffer = makeBuffer(device, std::vector<std::uint32_t>{nnz});
    std::vector<std::vector<std::uint32_t>> edgeLists(nnz);
    std::vector<std::uint32_t> edgeOwner(nnz, NM_INVALID_INDEX);
    std::vector<std::uint32_t> edgeNode(nnz, NM_INVALID_INDEX);
    for (std::uint32_t node = 0u; node < nodesPerEnvironment; ++node) {
        const NMIncidenceRangeGPU row = csrRows[node];
        const NMIncidenceRangeGPU range = nodeRanges[node];
        for (std::uint32_t incident = 0u; incident < range.count; ++incident) {
            const std::uint32_t tetIndex =
                nodeIncidence[range.first + incident];
            require(tetIndex < tetrahedraPerEnvironment,
                "edge map encountered an invalid incidence tet");
            std::uint32_t localRow = 4u;
            for (std::uint32_t slot = 0u; slot < 4u; ++slot)
                if (tetNodes[tetIndex][slot] == node) localRow = slot;
            require(localRow < 4u,
                "edge map incidence omits its node row");
            for (std::uint32_t localColumn = 0u; localColumn < 4u;
                 ++localColumn) {
                bool lastOccurrence = true;
                for (std::uint32_t later = localColumn + 1u; later < 4u;
                     ++later)
                    if (tetNodes[tetIndex][later] ==
                        tetNodes[tetIndex][localColumn]) lastOccurrence = false;
                if (!lastOccurrence) continue;
                const std::uint32_t columnNode = tetNodes[tetIndex][localColumn];
                auto first = csrColumns.begin() + row.first;
                auto last = first + row.count;
                const auto found = std::lower_bound(first, last, columnNode);
                require(found != last && *found == columnNode,
                    "edge map contribution is absent from its CSR row");
                const std::size_t edge =
                    static_cast<std::size_t>(found - csrColumns.begin());
                require(edge < nnz &&
                        (edgeOwner[edge] == NM_INVALID_INDEX ||
                         (edgeOwner[edge] == range.objectIndex &&
                          edgeNode[edge] == node)),
                    "edge map has conflicting owner or row metadata");
                edgeOwner[edge] = range.objectIndex;
                edgeNode[edge] = node;
                require(tetIndex <=
                    (std::numeric_limits<std::uint32_t>::max() -
                     localRow * 4u - localColumn) / 16u,
                    "edge contribution encoding overflowed");
                edgeLists[edge].push_back(
                    tetIndex * 16u + localRow * 4u + localColumn);
            }
        }
    }
    std::vector<NMIncidenceRangeGPU> edgeRanges(nnz);
    std::vector<std::uint32_t> edgeContributions;
    for (std::uint32_t edge = 0u; edge < nnz; ++edge) {
        require(!edgeLists[edge].empty() &&
                edgeOwner[edge] != NM_INVALID_INDEX &&
                edgeNode[edge] != NM_INVALID_INDEX,
            "edge map left a CSR entry without contributions");
        require(edgeContributions.size() <=
                    std::numeric_limits<std::uint32_t>::max() &&
                edgeLists[edge].size() <=
                    std::numeric_limits<std::uint32_t>::max() -
                        edgeContributions.size(),
            "edge contribution prefix overflowed uint32");
        edgeRanges[edge] = NMIncidenceRangeGPU{
            static_cast<std::uint32_t>(edgeContributions.size()),
            static_cast<std::uint32_t>(edgeLists[edge].size()),
            edgeOwner[edge], edgeNode[edge]};
        edgeContributions.insert(edgeContributions.end(),
            edgeLists[edge].begin(), edgeLists[edge].end());
    }
    require(edgeContributions.size() <=
                environments * tetrahedraPerEnvironment * 16u,
        "edge contribution topology exceeds the fixture tet bound");
    auto edgeRangesBuffer = makeBuffer(device, edgeRanges);
    auto edgeContributionsBuffer = makeBuffer(device, edgeContributions);
    auto edgeContributionCountBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{
            static_cast<std::uint32_t>(edgeContributions.size())});
    auto edgeCSRValuesBuffer = makeBuffer(device,
        std::vector<float>(environments * nnz * 9u, -117.0f));
    auto assembleEdges = [&] {
        const std::uint32_t total = environments * nnz;
        dispatch(device, queue, assembleCSREdges, {
            {0u, dispatchBuffer}, {1u, elementBlocksBuffer},
            {2u, edgeRangesBuffer}, {3u, edgeContributionsBuffer},
            {4u, edgeCSRValuesBuffer}, {5u, nnzBuffer},
            {6u, edgeContributionCountBuffer}, {7u, statusBuffer},
            {8u, solverStateBuffer}, {9u, forceRebuildBuffer},
            {10u, microBuffer}}, (total + 255u) / 256u, 256u);
    };
    dispatch(device, queue, assembleCSR, {
        {0u, dispatchBuffer}, {1u, tetrahedraBuffer},
        {2u, incidenceBuffer}, {3u, nodeRangesBuffer},
        {4u, elementBlocksBuffer}, {5u, csrRowsBuffer},
        {6u, csrColumnsBuffer}, {7u, csrValuesBuffer}, {8u, nnzBuffer},
        {9u, statusBuffer}, {10u, solverStateBuffer},
        {11u, forceRebuildBuffer}, {12u, microBuffer}},
        1u, 256u);
    for (std::uint32_t node = 0u; node < nodesPerEnvironment; ++node)
        require(csrRows[node].count == neighbors[node].size() &&
                csrRows[node].objectIndex == node,
            "host CSR fixture did not retain exact graph structure");

    const auto referenceElementBlocks = readBuffer<float>(
        elementBlocksBuffer, elementBlockFloatCount);
    const auto referenceCSRValues = readBuffer<float>(
        csrValuesBuffer, environments * nnz * 9u);
    assembleEdges();
    const auto edgeAssembledReference = readBuffer<float>(
        edgeCSRValuesBuffer, environments * nnz * 9u);
    require(edgeAssembledReference == referenceCSRValues,
        "edge contribution assembler changed ordered CSR accumulation");
    require(edgeRanges[0].count > 0u,
        "edge malformed-record fixture needs a nonempty contribution range");
    auto malformedEdgeContributions = edgeContributions;
    malformedEdgeContributions[edgeRanges[0].first] =
        tetrahedraPerEnvironment << 4u;
    auto malformedEdgeContributionsBuffer = makeBuffer(device,
        malformedEdgeContributions);
    auto malformedEdgeStatusBuffer = makeBuffer(device, statuses);
    const std::vector<float> malformedEdgeSentinel(
        environments * nnz * 9u, -131.0f);
    auto malformedEdgeValuesBuffer = makeBuffer(device,
        malformedEdgeSentinel);
    const std::uint32_t edgeTotal = environments * nnz;
    dispatch(device, queue, assembleCSREdges, {
        {0u, dispatchBuffer}, {1u, elementBlocksBuffer},
        {2u, edgeRangesBuffer}, {3u, malformedEdgeContributionsBuffer},
        {4u, malformedEdgeValuesBuffer}, {5u, nnzBuffer},
        {6u, edgeContributionCountBuffer}, {7u, malformedEdgeStatusBuffer},
        {8u, solverStateBuffer}, {9u, forceRebuildBuffer},
        {10u, microBuffer}}, (edgeTotal + 255u) / 256u, 256u);
    const auto malformedEdgeStatuses = readBuffer<NMMatterStatusGPU>(
        malformedEdgeStatusBuffer, environments);
    require(malformedEdgeStatuses[0u].code == NM_STATUS_NONFINITE_RESULT &&
            malformedEdgeStatuses[0u].environment == 0u &&
            malformedEdgeStatuses[0u].failingIndex == edgeNode[0] &&
            malformedEdgeStatuses[1u].code == NM_STATUS_NONFINITE_RESULT &&
            malformedEdgeStatuses[1u].environment == 1u &&
            malformedEdgeStatuses[1u].failingIndex == edgeNode[0],
        "shared malformed edge contribution did not fail both owning environments");
    const auto malformedEdgeValues = readBuffer<float>(
        malformedEdgeValuesBuffer, environments * nnz * 9u);
    for (std::size_t i = 0u; i < 9u; ++i)
        require(malformedEdgeValues[i] == -131.0f,
            "malformed edge contribution partially wrote its CSR edge");
    require(std::any_of(referenceElementBlocks.begin(),
                        referenceElementBlocks.end(),
                        [](float value) { return value != 0.0f; }) &&
            std::any_of(referenceCSRValues.begin(),
                        referenceCSRValues.end(),
                        [](float value) { return value != 0.0f; }),
        "cache-lifecycle fixture requires nonzero initial block and CSR data");
    auto assembleCurrentCache = [&] {
        dispatch(device, queue, assembleTet, {
            {0u, dispatchBuffer}, {1u, microBuffer}, {2u, objectsBuffer},
            {3u, tetrahedraBuffer}, {4u, tangentCacheBuffer},
            {5u, cacheValidBuffer}, {6u, elementBlocksBuffer},
            {7u, statusBuffer}, {8u, schedulerBuffer}, {9u, adaptiveBuffer},
            {10u, solverStateBuffer}, {11u, cacheModeBuffer},
            {12u, forceRebuildBuffer},
        {13u, includeMixedGraphXYZBuffer}}, 1u, 256u);
        dispatch(device, queue, assembleCSR, {
            {0u, dispatchBuffer}, {1u, tetrahedraBuffer},
            {2u, incidenceBuffer}, {3u, nodeRangesBuffer},
            {4u, elementBlocksBuffer}, {5u, csrRowsBuffer},
            {6u, csrColumnsBuffer}, {7u, csrValuesBuffer},
            {8u, nnzBuffer}, {9u, statusBuffer},
            {10u, solverStateBuffer}, {11u, forceRebuildBuffer},
            {12u, microBuffer}}, 1u, 256u);
        assembleEdges();
    };

    // A stale previous-convergence diagnostic must not suppress the new
    // Newton assembly when the current nonlinear latch is clear.
    auto staleConvergedStates = solverStates;
    for (auto& state : staleConvergedStates) {
        state.diagnostics.z = 1.0f;
        state.nonlinear.w = 0.0f;
    }
    std::memcpy([solverStateBuffer contents], staleConvergedStates.data(),
        staleConvergedStates.size() * sizeof(NMFGMRESStateGPU));
    NMMicrostepGPU currentNewtonMicro = micro;
    currentNewtonMicro.solverIteration = 1u;
    std::memcpy([microBuffer contents], &currentNewtonMicro,
        sizeof(currentNewtonMicro));
    const std::uint32_t noForcedRebuild = 0u;
    std::memcpy([forceRebuildBuffer contents], &noForcedRebuild,
        sizeof(noForcedRebuild));
    std::vector<float> sentinelBlocks(elementBlockFloatCount, -91.0f);
    std::vector<float> sentinelCSR(environments * nnz * 9u, -73.0f);
    std::memcpy([elementBlocksBuffer contents], sentinelBlocks.data(),
        sentinelBlocks.size() * sizeof(float));
    std::memcpy([csrValuesBuffer contents], sentinelCSR.data(),
        sentinelCSR.size() * sizeof(float));
    assembleCurrentCache();
    const auto rebuiltForCurrentNewton = readBuffer<float>(
        elementBlocksBuffer, elementBlockFloatCount);
    const auto rebuiltCSRForCurrentNewton = readBuffer<float>(
        csrValuesBuffer, environments * nnz * 9u);
    const auto rebuiltEdgesForCurrentNewton = readBuffer<float>(
        edgeCSRValuesBuffer, environments * nnz * 9u);
    require(rebuiltForCurrentNewton == referenceElementBlocks &&
            rebuiltCSRForCurrentNewton == referenceCSRValues &&
            rebuiltEdgesForCurrentNewton == referenceCSRValues,
        "stale prior diagnostics suppressed the current Newton CSR rebuild");

    // A completed current Newton with forceRebuild disabled preserves both
    // scratch arenas; it cannot overwrite the accepted operator snapshot.
    auto completedNewtonStates = staleConvergedStates;
    for (auto& state : completedNewtonStates) state.nonlinear.w = 1.0f;
    std::memcpy([solverStateBuffer contents], completedNewtonStates.data(),
        completedNewtonStates.size() * sizeof(NMFGMRESStateGPU));
    std::memcpy([elementBlocksBuffer contents], sentinelBlocks.data(),
        sentinelBlocks.size() * sizeof(float));
    std::memcpy([csrValuesBuffer contents], sentinelCSR.data(),
        sentinelCSR.size() * sizeof(float));
    const std::vector<float> sentinelEdges(environments * nnz * 9u, -119.0f);
    std::memcpy([edgeCSRValuesBuffer contents], sentinelEdges.data(),
        sentinelEdges.size() * sizeof(float));
    assembleCurrentCache();
    const auto preservedBlocks = readBuffer<float>(
        elementBlocksBuffer, elementBlockFloatCount);
    const auto preservedCSR = readBuffer<float>(
        csrValuesBuffer, environments * nnz * 9u);
    const auto preservedEdges = readBuffer<float>(
        edgeCSRValuesBuffer, environments * nnz * 9u);
    require(preservedBlocks == sentinelBlocks && preservedCSR == sentinelCSR &&
            preservedEdges == sentinelEdges,
        "completed Newton rewrote assembled operator scratch with forceRebuild off");

    // Restore a fresh active Newton and its bit-identical reference matrix for
    // the action parity probes below.
    std::memcpy([solverStateBuffer contents], solverStates.data(),
        solverStates.size() * sizeof(NMFGMRESStateGPU));
    micro.solverIteration = 0u;
    std::memcpy([microBuffer contents], &micro, sizeof(micro));
    const std::uint32_t forcedRebuild = 1u;
    std::memcpy([forceRebuildBuffer contents], &forcedRebuild,
        sizeof(forcedRebuild));
    assembleCurrentCache();

    auto directElementBuffer = makeBuffer(device,
        std::vector<NMFEMElementVectorGPU>(
            environments * tetrahedraPerEnvironment));
    auto mixedBuffer = makeBuffer(device,
        std::vector<NMMixedMaterialGPU>{NMMixedMaterialGPU{}});
    auto fieldBuffer = makeBuffer(device,
        std::vector<NMFEMFieldStateGPU>(
            environments * nodesPerEnvironment));
    auto learnedMaterialsBuffer = makeBuffer(device,
        std::vector<NMLearnedMaterialGPU>{NMLearnedMaterialGPU{}});
    auto learnedLayersBuffer = makeBuffer(device,
        std::vector<NMLearnedLayerGPU>{NMLearnedLayerGPU{}});
    auto learnedWeightsBuffer = makeBuffer(device, std::vector<float>{0.0f});
    auto solverBuffer = makeBuffer(device, std::vector<NMMixedSolverGPU>{NMMixedSolverGPU{}});
    auto fallbackMetricsBuffer = makeBuffer(device,
        std::vector<nm_float4>(environments, nm_float4{}));
    auto directionBuffer = makeBuffer(device, direction);
    auto runDirect = [&](const std::uint32_t assembledFlag) {
        micro.flags = NM_MICROSTEP_FGMRES_OPERATOR | assembledFlag;
        std::memcpy([microBuffer contents], &micro, sizeof(micro));
        dispatch(device, queue, directOperator, {
            {0u, dispatchBuffer}, {1u, microBuffer}, {2u, objectsBuffer},
            {3u, materialsBuffer}, {4u, programsBuffer},
            {5u, instructionsBuffer}, {6u, parametersBuffer},
            {7u, candidateBuffer}, {8u, tetrahedraBuffer},
            {9u, acceptedStateBuffer}, {10u, directionBuffer},
            {11u, schedulerBuffer}, {12u, adaptiveBuffer},
            {13u, directElementBuffer}, {14u, statusBuffer},
            {16u, mixedBuffer}, {17u, fieldBuffer},
            {18u, learnedMaterialsBuffer}, {19u, learnedLayersBuffer},
            {20u, learnedWeightsBuffer}, {21u, solverBuffer},
            {22u, solverStateBuffer}, {23u, tangentCacheBuffer},
            {24u, cacheValidBuffer}, {25u, cacheModeBuffer},
            {26u, fallbackMetricsBuffer}},
            1u, 256u);
        return readBuffer<NMFEMElementVectorGPU>(directElementBuffer,
            environments * tetrahedraPerEnvironment);
    };
    auto gatherElementActions = [&](const std::vector<NMFEMElementVectorGPU>& values) {
        std::vector<nm_float4> result(environments * nodesPerEnvironment,
                                      nm_float4{});
        for (std::uint32_t environment = 0u; environment < environments;
             ++environment) {
            const float dt = shape.gravityAndTimestep.w /
                float(1u << schedulers[environment].activeExponent);
            for (std::uint32_t tet = 0u; tet < tetrahedraPerEnvironment; ++tet) {
                const NMFEMElementVectorGPU value =
                    values[environment * tetrahedraPerEnvironment + tet];
                const std::array<nm_float4, 4u> local{
                    value.node0, value.node1, value.node2, value.node3};
                for (std::uint32_t slot = 0u; slot < 4u; ++slot) {
                    nm_float4& target = result[
                        environment * nodesPerEnvironment + tetNodes[tet][slot]];
                    target.x += -dt * dt * local[slot].x;
                    target.y += -dt * dt * local[slot].y;
                    target.z += -dt * dt * local[slot].z;
                }
            }
        }
        for (std::uint32_t environment = 0u; environment < environments;
             ++environment)
            for (std::uint32_t node = 0u; node < nodesPerEnvironment; ++node) {
                if (candidate[environment * nodesPerEnvironment + node].
                        restAndFixed.w == 0.0f)
                    continue;
                nm_float4& value =
                    result[environment * nodesPerEnvironment + node];
                value.x = value.y = value.z = 0.0f;
            }
        return result;
    };
    auto runSparseAction = [&](const bool zeroInitialWork = false,
                               const bool simdRow = false) {
        std::vector<nm_float4> initialWork(environments * nodesPerEnvironment);
        if (!zeroInitialWork)
            for (std::uint32_t index = 0u; index < initialWork.size(); ++index)
                initialWork[index] = {
                    0.25f + 0.1f * float(index), -0.5f, 0.75f,
                    17.0f + float(index)};
        auto workBuffer = makeBuffer(device, initialWork);
        const NSUInteger rows = environments * nodesPerEnvironment;
        dispatch(device, queue, simdRow ? applyCSRSimdRow : applyCSR, {
            {0u, dispatchBuffer}, {1u, microBuffer}, {2u, objectsBuffer},
            {3u, candidateBuffer}, {4u, nodeRangesBuffer},
            {5u, schedulerBuffer}, {6u, adaptiveBuffer},
            {7u, directionBuffer}, {8u, workBuffer}, {9u, csrRowsBuffer},
            {10u, csrColumnsBuffer}, {11u, csrValuesBuffer},
            {12u, nnzBuffer}, {13u, solverStateBuffer},
            {14u, statusBuffer}},
            simdRow ? rows : (rows + 255u) / 256u,
            simdRow ? 32u : 256u);
        auto after = readBuffer<nm_float4>(workBuffer, initialWork.size());
        for (std::uint32_t i = 0u; i < after.size(); ++i) {
            after[i].x -= initialWork[i].x;
            after[i].y -= initialWork[i].y;
            after[i].z -= initialWork[i].z;
            after[i].w -= initialWork[i].w;
        }
        return after;
    };
    auto compareAction = [&](const std::vector<NMFEMElementVectorGPU>& full,
                             const std::vector<NMFEMElementVectorGPU>& directRemainder,
                             const std::string& caseName) {
        const auto sparse = runSparseAction();
        const auto sparseSIMD = runSparseAction(false, true);
        const auto expected = gatherElementActions(full);
        const auto remainder = gatherElementActions(directRemainder);
        for (std::uint32_t index = 0u; index < sparse.size(); ++index) {
            const float actualX = sparse[index].x + remainder[index].x;
            const float actualY = sparse[index].y + remainder[index].y;
            const float actualZ = sparse[index].z + remainder[index].z;
            const float simdX = sparseSIMD[index].x + remainder[index].x;
            const float simdY = sparseSIMD[index].y + remainder[index].y;
            const float simdZ = sparseSIMD[index].z + remainder[index].z;
            const float scale = std::max(1.0f, std::max(
                std::abs(expected[index].x),
                std::max(std::abs(expected[index].y),
                         std::abs(expected[index].z))));
            const float tolerance = 3.0e-5f * scale;
            require(std::abs(actualX - expected[index].x) <= tolerance &&
                    std::abs(actualY - expected[index].y) <= tolerance &&
                    std::abs(actualZ - expected[index].z) <= tolerance &&
                    std::abs(simdX - expected[index].x) <= tolerance &&
                    std::abs(simdY - expected[index].y) <= tolerance &&
                    std::abs(simdZ - expected[index].z) <= tolerance &&
                    sparse[index].w == 0.0f && sparseSIMD[index].w == 0.0f &&
                    remainder[index].w == 0.0f,
                "assembled xyz action differs from direct FEM action in " +
                    caseName + " at row " + std::to_string(index));
        }
    };
    auto compareActionWithRoundoffBound = [&] (
        const std::vector<NMFEMElementVectorGPU>& full,
        const std::vector<NMFEMElementVectorGPU>& directRemainder,
        const std::string& caseName,
        const bool zeroInitialWork = false) {
        const auto sparse = runSparseAction(zeroInitialWork);
        const auto sparseSIMD = runSparseAction(zeroInitialWork, true);
        const auto expected = gatherElementActions(full);
        const auto remainder = gatherElementActions(directRemainder);
        const auto values = readBuffer<float>(csrValuesBuffer,
            environments * nnz * 9u);
        constexpr double roundoffFactor =
            64.0 * double(std::numeric_limits<float>::epsilon());
        for (std::uint32_t environment = 0u; environment < environments;
             ++environment) {
            const float dt = shape.gravityAndTimestep.w /
                float(1u << schedulers[environment].activeExponent);
            for (std::uint32_t node = 0u; node < nodesPerEnvironment; ++node) {
                const std::uint32_t rowIndex =
                    environment * nodesPerEnvironment + node;
                const NMIncidenceRangeGPU row = csrRows[rowIndex];
                for (std::uint32_t outAxis = 0u; outAxis < 3u; ++outAxis) {
                    double absoluteProducts = 0.0;
                    for (std::uint32_t edge = 0u; edge < row.count; ++edge) {
                        const std::uint32_t column =
                            csrColumns[row.first + edge];
                        const nm_float4 x =
                            direction[environment * nodesPerEnvironment + column];
                        for (std::uint32_t inAxis = 0u; inAxis < 3u; ++inAxis) {
                            const float input = inAxis == 0u ? x.x :
                                (inAxis == 1u ? x.y : x.z);
                            const float coefficient = values[
                                (environment * nnz + row.first + edge) * 9u +
                                outAxis * 3u + inAxis];
                            absoluteProducts += std::abs(double(coefficient) *
                                double(input) * double(dt) * double(dt));
                        }
                    }
                    const float sparseValue = outAxis == 0u ? sparse[rowIndex].x :
                        (outAxis == 1u ? sparse[rowIndex].y : sparse[rowIndex].z);
                    const float expectedValue = outAxis == 0u ? expected[rowIndex].x :
                        (outAxis == 1u ? expected[rowIndex].y : expected[rowIndex].z);
                    const float remainderValue = outAxis == 0u ? remainder[rowIndex].x :
                        (outAxis == 1u ? remainder[rowIndex].y : remainder[rowIndex].z);
                    const double actual = double(sparseValue) +
                        double(remainderValue);
                    const float simdValue = outAxis == 0u ? sparseSIMD[rowIndex].x :
                        (outAxis == 1u ? sparseSIMD[rowIndex].y :
                         sparseSIMD[rowIndex].z);
                    const double simdActual = double(simdValue) +
                        double(remainderValue);
                    const double error = std::abs(actual - double(expectedValue));
                    const double simdError =
                        std::abs(simdActual - double(expectedValue));
                    const double scale = absoluteProducts +
                        std::abs(double(expectedValue)) +
                        std::abs(double(remainderValue));
                    const double tolerance = roundoffFactor * scale + 1.0e-7;
                    require(error <= tolerance && simdError <= tolerance,
                        "assembled xyz action exceeded dot-product roundoff bound in " +
                            caseName + " at environment " +
                            std::to_string(environment) + " node " +
                            std::to_string(node) + " axis " +
                            std::to_string(outAxis) + " errors " +
                            std::to_string(error) + "," +
                            std::to_string(simdError) + " bound " +
                            std::to_string(tolerance));
                    require(sparse[rowIndex].w == 0.0f &&
                            sparseSIMD[rowIndex].w == 0.0f &&
                            remainder[rowIndex].w == 0.0f,
                        "assembled xyz action changed the pressure lane in " +
                            caseName);
                }
            }
        }
    };

    // Each deterministic random vector tests the actual cached tangent on the
    // current candidate, material-frame transforms, both object timesteps,
    // shared-edge graph reduction, and the fixed row.
    for (std::uint32_t sample = 0u; sample < 3u; ++sample) {
        std::uint32_t seed = 0x91e10da5u + sample * 0x9e3779b9u;
        for (auto& value : direction) {
            auto next = [&] {
                seed ^= seed << 13u;
                seed ^= seed >> 17u;
                seed ^= seed << 5u;
                return (float(seed & 0xffffu) / 32767.5f) - 1.0f;
            };
            value = {next(), next(), next(), 0.0f};
        }
        for (std::uint32_t environment = 0u; environment < environments;
             ++environment)
            direction[environment * nodesPerEnvironment] = nm_float4{};
        std::memcpy([directionBuffer contents], direction.data(),
            direction.size() * sizeof(nm_float4));
        const auto full = runDirect(0u);
        const auto skipped = runDirect(1u << 4u);
        for (const auto& value : skipped)
            require(std::abs(value.node0.x) + std::abs(value.node0.y) +
                    std::abs(value.node0.z) + std::abs(value.node1.x) +
                    std::abs(value.node1.y) + std::abs(value.node1.z) +
                    std::abs(value.node2.x) + std::abs(value.node2.y) +
                    std::abs(value.node2.z) + std::abs(value.node3.x) +
                    std::abs(value.node3.y) + std::abs(value.node3.z) == 0.0f,
                "assembled cache-valid tet was not skipped by direct FEM pass");
        const std::vector<NMFEMElementVectorGPU> noRemainder(
            environments * tetrahedraPerEnvironment, NMFEMElementVectorGPU{});
        compareAction(full, noRemainder, "valid-cache random direction");
    }

    // A constant displacement is an exact null mode of a nonmixed FEM
    // deformation-gradient tangent. Keep every row unconstrained here so the
    // test measures the assembled operator's partition-of-unity cancellation.
    for (auto& node : candidate) node.restAndFixed.w = 0.0f;
    std::memcpy([candidateBuffer contents], candidate.data(),
        candidate.size() * sizeof(NMFEMNodeStateGPU));
    for (std::uint32_t environment = 0u; environment < environments;
         ++environment)
        for (std::uint32_t node = 0u; node < nodesPerEnvironment; ++node)
            direction[environment * nodesPerEnvironment + node] =
                {8192.0f * float(environment + 1u),
                 -4096.0f * float(environment + 1u),
                 2048.0f * float(environment + 1u), 0.0f};
    std::memcpy([directionBuffer contents], direction.data(),
        direction.size() * sizeof(nm_float4));
    const auto translatedDirect = runDirect(0u);
    const auto noRemainder = std::vector<NMFEMElementVectorGPU>(
        environments * tetrahedraPerEnvironment, NMFEMElementVectorGPU{});
    const auto translatedExpected = gatherElementActions(translatedDirect);
    for (const auto& value : translatedExpected)
        require(std::abs(value.x) + std::abs(value.y) + std::abs(value.z) < 1.0e-6f,
            "direct FEM tangent failed exact uniform-translation null mode");
    const auto translatedSparse = runSparseAction(true);
    const auto translatedSparseSIMD = runSparseAction(true, true);
    for (const nm_float4 value : translatedSparse)
            require(std::abs(value.x) < 3.0e-5f &&
                std::abs(value.y) < 3.0e-5f &&
                std::abs(value.z) < 3.0e-5f,
            "assembled FEM operator did not preserve uniform-translation null mode");
    for (const nm_float4 value : translatedSparseSIMD)
        require(std::abs(value.x) < 3.0e-5f &&
                std::abs(value.y) < 3.0e-5f &&
                std::abs(value.z) < 3.0e-5f,
            "SIMD-row FEM operator did not preserve uniform-translation null mode");
    compareActionWithRoundoffBound(translatedDirect, noRemainder,
                                   "uniform translation cancellation", true);

    // Large shared translation plus a small nonuniform component stresses
    // cancellation in K*x while retaining a well-resolved direct dF.
    for (std::uint32_t environment = 0u; environment < environments;
         ++environment) {
        std::uint32_t seed = 0xa511e9b3u + environment * 0x63d83595u;
        for (std::uint32_t node = 0u; node < nodesPerEnvironment; ++node) {
            seed = seed * 1664525u + 1013904223u;
            const float x = float(seed >> 8u) / float(0x00ffffffu);
            seed = seed * 1664525u + 1013904223u;
            const float y = float(seed >> 8u) / float(0x00ffffffu);
            seed = seed * 1664525u + 1013904223u;
            const float z = float(seed >> 8u) / float(0x00ffffffu);
            direction[environment * nodesPerEnvironment + node] = {
                8192.0f * float(environment + 1u) + 0.25f * (2.0f * x - 1.0f),
                -4096.0f * float(environment + 1u) + 0.25f * (2.0f * y - 1.0f),
                2048.0f * float(environment + 1u) + 0.25f * (2.0f * z - 1.0f),
                0.0f};
        }
    }
    std::memcpy([directionBuffer contents], direction.data(),
        direction.size() * sizeof(nm_float4));
    const auto cancellationDirect = runDirect(0u);
    compareActionWithRoundoffBound(cancellationDirect, noRemainder,
                                   "large-translation cancellation", true);

    // Internal FEM forces close to zero over all unconstrained nodes. Check
    // both paths with double-precision host accumulation and an explicit
    // float summation bound derived from the accumulated magnitudes.
    const auto cancellationExpected = gatherElementActions(cancellationDirect);
    const auto cancellationSparse = runSparseAction(true);
    const auto cancellationSparseSIMD = runSparseAction(true, true);
    for (std::uint32_t environment = 0u; environment < environments;
         ++environment)
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            double directSum = 0.0, directAbs = 0.0;
            double sparseSum = 0.0, sparseAbs = 0.0;
            double sparseSIMDSum = 0.0, sparseSIMDAbs = 0.0;
            for (std::uint32_t node = 0u; node < nodesPerEnvironment; ++node) {
                const nm_float4 directValue = cancellationExpected[
                    environment * nodesPerEnvironment + node];
                const nm_float4 sparseValue = cancellationSparse[
                    environment * nodesPerEnvironment + node];
                const double a = axis == 0u ? directValue.x :
                    (axis == 1u ? directValue.y : directValue.z);
                const double b = axis == 0u ? sparseValue.x :
                    (axis == 1u ? sparseValue.y : sparseValue.z);
                const nm_float4 simdValue = cancellationSparseSIMD[
                    environment * nodesPerEnvironment + node];
                const double c = axis == 0u ? simdValue.x :
                    (axis == 1u ? simdValue.y : simdValue.z);
                directSum += a;
                directAbs += std::abs(a);
                sparseSum += b;
                sparseAbs += std::abs(b);
                sparseSIMDSum += c;
                sparseSIMDAbs += std::abs(c);
            }
            const double tolerance = 96.0 *
                double(std::numeric_limits<float>::epsilon()) *
                std::max(directAbs, sparseAbs) + 1.0e-7;
            if (!(std::abs(directSum) <= tolerance &&
                  std::abs(sparseSum) <= tolerance &&
                  std::abs(sparseSIMDSum) <= 96.0 *
                      double(std::numeric_limits<float>::epsilon()) *
                      std::max(directAbs, sparseSIMDAbs) + 1.0e-7)) {
                std::cerr << "force_closure_failure env=" << environment
                          << " axis=" << axis
                          << " direct_sum=" << directSum
                          << " sparse_sum=" << sparseSum
                          << " simd_sum=" << sparseSIMDSum
                          << " tolerance=" << tolerance
                          << " direct_abs=" << directAbs
                          << " sparse_abs=" << sparseAbs << std::endl;
                require(false,
                    "nonmixed FEM internal tangent force did not close under "
                    "global xyz summation");
            }
        }

    // A nonfinite CSR coefficient must fail its owning environment and report
    // the output-node witness, rather than silently poisoning Krylov work.
    const std::uint32_t witnessEnvironment = 1u;
    const std::uint32_t witnessNode = nodesPerEnvironment - 1u;
    const NMIncidenceRangeGPU witnessRow = csrRows[
        witnessEnvironment * nodesPerEnvironment + witnessNode];
    require(witnessRow.count > 0u,
        "nonfinite CSR witness row unexpectedly has no incident edges");
    auto nonfiniteValues = readBuffer<float>(csrValuesBuffer,
        environments * nnz * 9u);
    const std::size_t badValue =
        (witnessEnvironment * nnz + witnessRow.first) * 9u;
    nonfiniteValues[badValue] = std::numeric_limits<float>::quiet_NaN();
    auto nonfiniteValuesBuffer = makeBuffer(device, nonfiniteValues);
    auto nonfiniteStatusBuffer = makeBuffer(device, statuses);
    auto nonfiniteWorkBuffer = makeBuffer(device,
        std::vector<nm_float4>(environments * nodesPerEnvironment,
                               nm_float4{}));
    dispatch(device, queue, applyCSR, {
        {0u, dispatchBuffer}, {1u, microBuffer}, {2u, objectsBuffer},
        {3u, candidateBuffer}, {4u, nodeRangesBuffer},
        {5u, schedulerBuffer}, {6u, adaptiveBuffer},
        {7u, directionBuffer}, {8u, nonfiniteWorkBuffer},
        {9u, csrRowsBuffer}, {10u, csrColumnsBuffer},
        {11u, nonfiniteValuesBuffer}, {12u, nnzBuffer},
        {13u, solverStateBuffer}, {14u, nonfiniteStatusBuffer}},
        1u, 256u);
    const auto nonfiniteStatus = readBuffer<NMMatterStatusGPU>(
        nonfiniteStatusBuffer, environments);
    auto simdNonfiniteStatusBuffer = makeBuffer(device, statuses);
    auto simdNonfiniteWorkBuffer = makeBuffer(device,
        std::vector<nm_float4>(environments * nodesPerEnvironment,
                               nm_float4{}));
    dispatch(device, queue, applyCSRSimdRow, {
        {0u, dispatchBuffer}, {1u, microBuffer}, {2u, objectsBuffer},
        {3u, candidateBuffer}, {4u, nodeRangesBuffer},
        {5u, schedulerBuffer}, {6u, adaptiveBuffer},
        {7u, directionBuffer}, {8u, simdNonfiniteWorkBuffer},
        {9u, csrRowsBuffer}, {10u, csrColumnsBuffer},
        {11u, nonfiniteValuesBuffer}, {12u, nnzBuffer},
        {13u, solverStateBuffer}, {14u, simdNonfiniteStatusBuffer}},
        environments * nodesPerEnvironment, 32u);
    const auto simdNonfiniteStatus = readBuffer<NMMatterStatusGPU>(
        simdNonfiniteStatusBuffer, environments);
    require(simdNonfiniteStatus[witnessEnvironment].code ==
                NM_STATUS_NONFINITE_RESULT &&
            simdNonfiniteStatus[witnessEnvironment].failingIndex == witnessNode &&
            simdNonfiniteStatus[0u].code == NM_STATUS_SUCCESS,
        "SIMD-row nonfinite action did not report its environment/node");
    require(nonfiniteStatus[witnessEnvironment].code ==
                NM_STATUS_NONFINITE_RESULT &&
            nonfiniteStatus[witnessEnvironment].environment ==
                witnessEnvironment &&
            nonfiniteStatus[witnessEnvironment].objectIndex == 0u &&
            nonfiniteStatus[witnessEnvironment].failingIndex == witnessNode &&
            nonfiniteStatus[0u].code == NM_STATUS_SUCCESS,
        "nonfinite assembled FEM action did not report its environment/node");

    // An invalid cache entry emits no block and remains in the direct fallback.
    for (std::uint32_t environment = 0u; environment < environments;
         ++environment)
        cacheValid[environment * tetrahedraPerEnvironment + 1u] = 0u;
    std::memcpy([cacheValidBuffer contents], cacheValid.data(),
        cacheValid.size() * sizeof(std::uint32_t));
    dispatch(device, queue, assembleTet, {
        {0u, dispatchBuffer}, {1u, microBuffer}, {2u, objectsBuffer},
        {3u, tetrahedraBuffer}, {4u, tangentCacheBuffer},
        {5u, cacheValidBuffer}, {6u, elementBlocksBuffer}, {7u, statusBuffer},
        {8u, schedulerBuffer}, {9u, adaptiveBuffer}, {10u, solverStateBuffer},
        {11u, cacheModeBuffer}, {12u, forceRebuildBuffer},
        {13u, includeMixedGraphXYZBuffer}},
        1u, 256u);
    dispatch(device, queue, assembleCSR, {
        {0u, dispatchBuffer}, {1u, tetrahedraBuffer},
        {2u, incidenceBuffer}, {3u, nodeRangesBuffer},
        {4u, elementBlocksBuffer}, {5u, csrRowsBuffer},
        {6u, csrColumnsBuffer}, {7u, csrValuesBuffer}, {8u, nnzBuffer},
        {9u, statusBuffer}, {10u, solverStateBuffer},
        {11u, forceRebuildBuffer}, {12u, microBuffer}},
        1u, 256u);
    const auto invalidBlocks = readBuffer<float>(
        elementBlocksBuffer, elementBlockFloatCount);
    for (std::uint32_t environment = 0u; environment < environments;
         ++environment) {
        const std::size_t base =
            (environment * tetrahedraPerEnvironment + 1u) * 144u;
        for (std::size_t i = 0u; i < 144u; ++i)
            require(invalidBlocks[base + i] == 0.0f,
                "invalid tangent-cache tet must emit a zero assembled block");
    }
    std::uint32_t seed = 0x1234567u;
    for (auto& value : direction) {
        seed = seed * 1664525u + 1013904223u;
        const float x = float(seed >> 8u) / float(0x00ffffffu);
        seed = seed * 1664525u + 1013904223u;
        const float y = float(seed >> 8u) / float(0x00ffffffu);
        seed = seed * 1664525u + 1013904223u;
        const float z = float(seed >> 8u) / float(0x00ffffffu);
        value = {2.0f * x - 1.0f, 2.0f * y - 1.0f, 2.0f * z - 1.0f, 0.0f};
    }
    std::memcpy([directionBuffer contents], direction.data(),
        direction.size() * sizeof(nm_float4));
    const auto fullWithFallback = runDirect(0u);
    const auto directRemainder = runDirect(1u << 4u);
    compareAction(fullWithFallback, directRemainder,
                  "invalid-cache direct fallback");

    // A zero-element graph is a no-op for sparse action and output state.
    NMMatterDispatchGPU emptyShape = shape;
    emptyShape.femNodeCount = 3u;
    emptyShape.tetrahedronCount = 0u;
    auto emptyDispatch = makeBuffer(device, std::vector{emptyShape});
    std::vector<NMIncidenceRangeGPU> emptyRanges(
        environments * emptyShape.femNodeCount, NMIncidenceRangeGPU{});
    for (auto& range : emptyRanges) range.objectIndex = 0u;
    auto emptyRangesBuffer = makeBuffer(device, emptyRanges);
    auto emptyIncidenceBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
    std::vector<NMIncidenceRangeGPU> emptyRows(
        environments * emptyShape.femNodeCount);
    for (std::uint32_t environment = 0u; environment < environments;
         ++environment)
        for (std::uint32_t node = 0u; node < emptyShape.femNodeCount; ++node)
            emptyRows[environment * emptyShape.femNodeCount + node] =
                {0u, 0u, node, 0u};
    auto emptyRowsBuffer = makeBuffer(device, emptyRows);
    auto emptyColumnsBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto emptyValuesBuffer = makeBuffer(device,
        std::vector<float>(environments * 9u, 0.0f));
    auto emptyNNZBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto emptyBlocksBuffer = makeBuffer(device, std::vector<float>{0.0f});
    auto emptyTetsBuffer = makeBuffer(device,
        std::vector<NMTetrahedronGPU>{NMTetrahedronGPU{}});
    auto emptyCandidateBuffer = makeBuffer(device,
        std::vector<NMFEMNodeStateGPU>(environments * emptyShape.femNodeCount));
    std::vector<nm_float4> emptyDirection(
        environments * emptyShape.femNodeCount,
        nm_float4{0.25f, -0.5f, 0.75f, 4.0f});
    auto emptyDirectionBuffer = makeBuffer(device, emptyDirection);
    std::vector<nm_float4> emptyWork(
        environments * emptyShape.femNodeCount,
        nm_float4{3.0f, 4.0f, 5.0f, 6.0f});
    auto emptyWorkBuffer = makeBuffer(device, emptyWork);
    auto emptyTangentCache = makeBuffer(device, std::vector<float>{0.0f});
    auto emptyValid = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto emptyState = makeBuffer(device,
        std::vector<NMFGMRESStateGPU>(environments));
    auto emptyStatuses = makeBuffer(device, statuses);
    dispatch(device, queue, assembleTet, {
        {0u, emptyDispatch}, {1u, microBuffer}, {2u, objectsBuffer},
        {3u, emptyTetsBuffer}, {4u, emptyTangentCache}, {5u, emptyValid},
        {6u, emptyBlocksBuffer}, {7u, emptyStatuses}, {8u, schedulerBuffer},
        {9u, adaptiveBuffer}, {10u, emptyState}, {11u, cacheModeBuffer},
        {12u, forceRebuildBuffer},
        {13u, includeMixedGraphXYZBuffer}}, 1u, 256u);
    dispatch(device, queue, assembleCSR, {
        {0u, emptyDispatch}, {1u, emptyTetsBuffer},
        {2u, emptyIncidenceBuffer}, {3u, emptyRangesBuffer},
        {4u, emptyBlocksBuffer}, {5u, emptyRowsBuffer},
        {6u, emptyColumnsBuffer}, {7u, emptyValuesBuffer},
        {8u, emptyNNZBuffer}, {9u, emptyStatuses}, {10u, emptyState},
        {11u, forceRebuildBuffer}, {12u, microBuffer}}, 1u, 256u);
    dispatch(device, queue, applyCSR, {
        {0u, emptyDispatch}, {1u, microBuffer}, {2u, objectsBuffer},
        {3u, emptyCandidateBuffer}, {4u, emptyRangesBuffer},
        {5u, schedulerBuffer}, {6u, adaptiveBuffer},
        {7u, emptyDirectionBuffer}, {8u, emptyWorkBuffer},
        {9u, emptyRowsBuffer}, {10u, emptyColumnsBuffer},
        {11u, emptyValuesBuffer}, {12u, emptyNNZBuffer},
        {13u, emptyState}, {14u, emptyStatuses}}, 1u, 256u);
    const auto emptyResult = readBuffer<nm_float4>(emptyWorkBuffer,
        emptyWork.size());
    require(std::memcmp(emptyResult.data(), emptyWork.data(),
            emptyWork.size() * sizeof(nm_float4)) == 0,
        "empty assembled FEM domain changed sparse-action output");
    auto emptySIMDWorkBuffer = makeBuffer(device, emptyWork);
    dispatch(device, queue, applyCSRSimdRow, {
        {0u, emptyDispatch}, {1u, microBuffer}, {2u, objectsBuffer},
        {3u, emptyCandidateBuffer}, {4u, emptyRangesBuffer},
        {5u, schedulerBuffer}, {6u, adaptiveBuffer},
        {7u, emptyDirectionBuffer}, {8u, emptySIMDWorkBuffer},
        {9u, emptyRowsBuffer}, {10u, emptyColumnsBuffer},
        {11u, emptyValuesBuffer}, {12u, emptyNNZBuffer},
        {13u, emptyState}, {14u, emptyStatuses}},
        environments * emptyShape.femNodeCount, 32u);
    const auto emptySIMDResult = readBuffer<nm_float4>(emptySIMDWorkBuffer,
        emptyWork.size());
    require(std::memcmp(emptySIMDResult.data(), emptyWork.data(),
            emptyWork.size() * sizeof(nm_float4)) == 0,
        "empty assembled SIMD-row FEM domain changed output");

    // Build a coupled q+fine operator from the production FEM tangent action.
    constexpr std::uint32_t hybridModes = 3u;
    constexpr std::uint32_t hybridDofs = 12u;
    auto hybridLift = pipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_hybrid_lift");
    NMPartitionedReductionBlockDispatchGPU hybridBlock{};
    hybridBlock.globalNodeCount = nodesPerEnvironment;
    hybridBlock.basisNodeCount = nodesPerEnvironment;
    hybridBlock.modeCount = hybridModes;
    hybridBlock.tileCount = 1u;
    hybridBlock.environmentCount = 1u;
    hybridBlock.regionCount = 1u;
    hybridBlock.blockModeCount = hybridModes;
    hybridBlock.reserved = 1u;
    std::vector<std::uint32_t> hybridGlobalNodes{0u, 1u, 2u, 3u, 4u};
    std::vector<nm_float4> hybridBasis(hybridModes * nodesPerEnvironment,
        nm_float4{});
    const std::array<std::array<float, 3u>, 3u> hybridTransform{{
        {{1.0f, 0.20f, -0.10f}},
        {{0.15f, 0.90f, 0.25f}},
        {{-0.05f, 0.10f, 1.10f}},
    }};
    for (std::uint32_t mode = 0u; mode < hybridModes; ++mode)
        hybridBasis[mode * nodesPerEnvironment + 1u] = {
            hybridTransform[0u][mode], hybridTransform[1u][mode],
            hybridTransform[2u][mode], 0.0f};
    std::vector<std::uint32_t> hybridBlockModes{0u, 1u, 2u};
    std::vector<NMPartitionedReductionSparseRangeGPU> hybridNodeModes(
        nodesPerEnvironment, NMPartitionedReductionSparseRangeGPU{0u, 0u});
    hybridNodeModes[1u] = {0u, hybridModes};
    auto hybridBlockBuffer = makeBuffer(device,
        std::vector<NMPartitionedReductionBlockDispatchGPU>{hybridBlock});
    auto hybridGlobalNodesBuffer = makeBuffer(device, hybridGlobalNodes);
    auto hybridBasisBuffer = makeBuffer(device, hybridBasis);
    auto hybridBlockModesBuffer = makeBuffer(device, hybridBlockModes);
    auto hybridNodeModesBuffer = makeBuffer(device, hybridNodeModes);
    auto hybridNodeSlotsBuffer = makeBuffer(device, hybridBlockModes);
    auto hybridSparseTiles = pipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_reduce_sparse_mode_tiles");
    auto hybridSparsePartials = pipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_reduce_sparse_mode_partials");
    auto hybridZeroSelected = pipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_hybrid_zero_selected");
    auto hybridPackModes = pipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_hybrid_pack_modes");
    auto hybridClearInactive = pipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_hybrid_clear_inactive_field");
    auto hybridGatherQRHS = pipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_hybrid_gather_q_rhs");
    auto hybridLiftCopy = pipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_hybrid_copy");

    // Exercise the hybrid global coarse bridge on a nonidentity B. Coarse
    // probes are written only into q, lifted by the production B map, and
    // coarse corrections are scattered only back into q while preserving the
    // fine complement and the other lanes of each float4.
    auto hybridCoarsePrepare = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_human_global_coarse_hybrid_prepare_input");
    auto hybridCoarseProject = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_human_global_coarse_hybrid_project_rhs");
    auto hybridCoarseAdd = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_human_global_coarse_hybrid_add");
    constexpr std::uint32_t hybridCoarseModes = 6u;
    constexpr std::uint32_t hybridCoordinateCount = hybridModes;
    NMMatterDispatchGPU hybridCoarseDispatch{};
    hybridCoarseDispatch.environmentCount = 1u;
    hybridCoarseDispatch.femNodeCount = nodesPerEnvironment;
    NMFGMRESLayoutGPU hybridCoarseLayout{};
    hybridCoarseLayout.unknownCount = 2u * nodesPerEnvironment;
    auto hybridCoarseDispatchBuffer = makeBuffer(device,
        std::vector<NMMatterDispatchGPU>{hybridCoarseDispatch});
    auto hybridCoarseLayoutBuffer = makeBuffer(device,
        std::vector<NMFGMRESLayoutGPU>{hybridCoarseLayout});
    std::vector<float> hybridCoarseQCoordinates(
        hybridCoarseModes * hybridCoordinateCount);
    for (std::uint32_t coarse = 0u; coarse < hybridCoarseModes; ++coarse)
        for (std::uint32_t q = 0u; q < hybridCoordinateCount; ++q)
            hybridCoarseQCoordinates[coarse * hybridCoordinateCount + q] =
                0.11f * float(coarse + 1u) - 0.07f * float(q + 1u) +
                0.013f * float((coarse + 2u) * (q + 1u));
    auto hybridCoarseQBuffer = makeBuffer(device, hybridCoarseQCoordinates);
    const std::uint32_t hybridCoarseMode = 4u;
    auto hybridCoarseModeBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{hybridCoarseMode});
    auto hybridCoordinateCountBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{hybridCoordinateCount});
    std::vector<nm_float4> hybridCoarseInput(
        hybridCoarseLayout.unknownCount,
        nm_float4{91.0f, 92.0f, 93.0f, 94.0f});
    auto hybridCoarseInputBuffer = makeBuffer(device, hybridCoarseInput);
    dispatch(device, queue, hybridCoarsePrepare, {
        {0u, hybridCoarseDispatchBuffer}, {1u, hybridCoarseQBuffer},
        {2u, hybridCoarseInputBuffer}, {3u, hybridCoarseModeBuffer},
        {4u, hybridCoordinateCountBuffer}, {30u, hybridCoarseLayoutBuffer}},
        (hybridCoarseLayout.unknownCount + 255u) / 256u, 256u);
    hybridCoarseInput = readBuffer<nm_float4>(hybridCoarseInputBuffer,
        hybridCoarseLayout.unknownCount);
    for (std::uint32_t i = 0u; i < hybridCoarseLayout.unknownCount; ++i) {
        const bool qSlot = i >= nodesPerEnvironment &&
            i < nodesPerEnvironment + hybridCoordinateCount;
        if (!qSlot) {
            require(hybridCoarseInput[i].x == 0.0f &&
                    hybridCoarseInput[i].y == 0.0f &&
                    hybridCoarseInput[i].z == 0.0f &&
                    hybridCoarseInput[i].w == 0.0f,
                "hybrid coarse probe did not clear non-q coordinates");
        } else {
            const std::uint32_t q = i - nodesPerEnvironment;
            require(std::abs(hybridCoarseInput[i].x -
                    hybridCoarseQCoordinates[
                        hybridCoarseMode * hybridCoordinateCount + q]) < 1.0e-7f &&
                    hybridCoarseInput[i].y == 0.0f &&
                    hybridCoarseInput[i].z == 0.0f &&
                    hybridCoarseInput[i].w == 0.0f,
                "hybrid coarse probe did not pack tracked q coefficients");
        }
    }
    std::vector<nm_float4> coarseLiftSource = hybridCoarseInput;
    coarseLiftSource[0u] = {1.25f, -2.5f, 3.75f, 71.0f};
    coarseLiftSource[1u] = {9.0f, 8.0f, 7.0f, 73.0f};
    coarseLiftSource[2u] = {-4.0f, 5.0f, -6.0f, 79.0f};
    auto coarseLiftSourceBuffer = makeBuffer(device, coarseLiftSource);
    auto coarseLiftOutputBuffer = makeBuffer(device, coarseLiftSource);
    dispatch(device, queue, hybridLiftCopy, {
        {0u, hybridCoarseLayoutBuffer}, {1u, coarseLiftSourceBuffer},
        {2u, coarseLiftOutputBuffer}},
        (hybridCoarseLayout.unknownCount + 255u) / 256u, 256u);
    dispatch(device, queue, hybridLift, {
        {0u, hybridBlockBuffer}, {1u, hybridGlobalNodesBuffer},
        {2u, hybridBasisBuffer}, {3u, hybridBlockModesBuffer},
        {4u, hybridNodeModesBuffer}, {5u, hybridNodeSlotsBuffer},
        {6u, coarseLiftSourceBuffer}, {7u, coarseLiftOutputBuffer}},
        nodesPerEnvironment, 256u);
    dispatch(device, queue, hybridClearInactive, {
        {0u, hybridBlockBuffer}, {1u, coarseLiftOutputBuffer}},
        nodesPerEnvironment, 256u);
    const auto coarseLifted = readBuffer<nm_float4>(coarseLiftOutputBuffer,
        hybridCoarseLayout.unknownCount);
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        double expected = 0.0;
        for (std::uint32_t q = 0u; q < hybridCoordinateCount; ++q)
            expected += double(hybridTransform[axis][q]) *
                hybridCoarseQCoordinates[
                    hybridCoarseMode * hybridCoordinateCount + q];
        const double actual = axis == 0u ? coarseLifted[1u].x :
            (axis == 1u ? coarseLifted[1u].y : coarseLifted[1u].z);
        require(std::abs(actual - expected) <= 4.0e-5 *
                std::max(1.0, std::abs(expected)),
            "hybrid coarse probe lift did not reconstruct P=Bq");
    }
    require(coarseLifted[1u].w == 73.0f &&
            coarseLifted[2u].x == -4.0f &&
            coarseLifted[2u].y == 5.0f &&
            coarseLifted[2u].z == -6.0f &&
            coarseLifted[2u].w == 79.0f,
        "hybrid coarse lift changed selected w or fine-complement coordinates");

    std::vector<nm_float4> hybridCoarseResidual(
        hybridCoarseLayout.unknownCount, nm_float4{});
    std::array<float, hybridCoordinateCount> qResidual{
        0.4f, -0.3f, 0.8f};
    for (std::uint32_t q = 0u; q < hybridCoordinateCount; ++q)
        hybridCoarseResidual[nodesPerEnvironment + q].x = qResidual[q];
    auto hybridCoarseResidualBuffer = makeBuffer(device, hybridCoarseResidual);
    auto hybridCoarseRHSBuffer = makeBuffer(device,
        std::vector<float>(hybridCoarseModes, -99.0f));
    auto hybridCoarseModeCountBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{hybridCoarseModes});
    dispatch(device, queue, hybridCoarseProject, {
        {0u, hybridCoarseDispatchBuffer}, {1u, hybridCoarseQBuffer},
        {2u, hybridCoarseResidualBuffer}, {3u, hybridCoarseRHSBuffer},
        {4u, hybridCoordinateCountBuffer}, {5u, hybridCoarseModeCountBuffer},
        {30u, hybridCoarseLayoutBuffer}}, hybridCoarseModes, 32u);
    const auto hybridCoarseRHS = readBuffer<float>(
        hybridCoarseRHSBuffer, hybridCoarseModes);
    for (std::uint32_t coarse = 0u; coarse < hybridCoarseModes; ++coarse) {
        double expected = 0.0;
        for (std::uint32_t q = 0u; q < hybridCoordinateCount; ++q)
            expected += double(hybridCoarseQCoordinates[
                coarse * hybridCoordinateCount + q]) * qResidual[q];
        require(std::abs(double(hybridCoarseRHS[coarse]) - expected) < 2.0e-6,
            "hybrid global coarse RHS did not project the q covector");
    }
    std::vector<float> hybridCoarseCoefficients{
        0.2f, -0.1f, 0.3f, -0.25f, 0.05f, 0.15f};
    auto hybridCoarseCoefficientsBuffer = makeBuffer(device,
        hybridCoarseCoefficients);
    auto hybridCoarseValidBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{1u});
    NMFGMRESStateGPU hybridCoarseState{};
    hybridCoarseState.diagnostics.z = 0.0f;
    auto hybridCoarseStateBuffer = makeBuffer(device,
        std::vector<NMFGMRESStateGPU>{hybridCoarseState});
    auto hybridCoarseUseCountBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u});
    std::vector<nm_float4> coarsePreconditioned(
        hybridCoarseLayout.unknownCount, nm_float4{});
    for (std::uint32_t q = 0u; q < hybridCoordinateCount; ++q)
        coarsePreconditioned[nodesPerEnvironment + q] =
            {0.5f + float(q), -2.0f, 3.0f, 17.0f + float(q)};
    coarsePreconditioned[2u] = {-1.0f, 2.0f, 4.0f, 23.0f};
    auto coarsePreconditionedBuffer = makeBuffer(device, coarsePreconditioned);
    dispatch(device, queue, hybridCoarseAdd, {
        {0u, hybridCoarseDispatchBuffer}, {1u, hybridCoarseQBuffer},
        {2u, hybridCoarseCoefficientsBuffer}, {3u, hybridCoarseValidBuffer},
        {4u, coarsePreconditionedBuffer}, {5u, hybridCoarseStateBuffer},
        {6u, hybridCoordinateCountBuffer}, {7u, hybridCoarseUseCountBuffer}},
        (hybridCoordinateCount + 255u) / 256u, 256u);
    const auto coarseCorrected = readBuffer<nm_float4>(
        coarsePreconditionedBuffer, hybridCoarseLayout.unknownCount);
    for (std::uint32_t q = 0u; q < hybridCoordinateCount; ++q) {
        double expected = coarsePreconditioned[
            nodesPerEnvironment + q].x;
        for (std::uint32_t coarse = 0u; coarse < hybridCoarseModes; ++coarse)
            expected += double(hybridCoarseQCoordinates[
                coarse * hybridCoordinateCount + q]) *
                hybridCoarseCoefficients[coarse];
        const nm_float4 actual = coarseCorrected[nodesPerEnvironment + q];
        require(std::abs(double(actual.x) - expected) < 2.0e-6 &&
                actual.y == -2.0f && actual.z == 3.0f &&
                actual.w == 17.0f + float(q),
            "hybrid coarse correction did not scatter only q.x");
    }
    require(coarseCorrected[2u].x == -1.0f &&
            coarseCorrected[2u].y == 2.0f &&
            coarseCorrected[2u].z == 4.0f &&
            coarseCorrected[2u].w == 23.0f,
        "hybrid coarse correction changed fine-complement coordinates");
    require(readBuffer<std::uint32_t>(hybridCoarseUseCountBuffer, 1u)[0] == 1u,
        "hybrid coarse q correction did not report its actual nonzero use");

    std::vector<NMPartitionedReductionSparseRangeGPU> hybridRegionSupport{{0u, 1u}};
    std::vector<std::uint32_t> hybridRegionSupportRows{1u};
    auto hybridRegionSupportBuffer = makeBuffer(device, hybridRegionSupport);
    auto hybridRegionSupportRowsBuffer = makeBuffer(device, hybridRegionSupportRows);
    auto hybridUnitMassBuffer = makeBuffer(device,
        std::vector<float>(nodesPerEnvironment, 1.0f));
    auto hybridProjectionPartialsBuffer = makeBuffer(device,
        std::vector<float>(hybridModes, 0.0f));
    auto hybridProjectedValuesBuffer = makeBuffer(device,
        std::vector<float>(hybridModes, 0.0f));
    auto projectHybridAction = [&](const std::vector<nm_float4>& physicalAction) {
        require(physicalAction.size() == nodesPerEnvironment,
            "hybrid restriction received an invalid physical span");
        std::vector<nm_float4> coordinates(2u * nodesPerEnvironment, nm_float4{});
        std::copy(physicalAction.begin(), physicalAction.end(), coordinates.begin());
        auto coordinatesBuffer = makeBuffer(device, coordinates);
        dispatchGrid3D(device, queue, hybridSparseTiles, {
            {0u, hybridBlockBuffer}, {1u, coordinatesBuffer},
            {2u, hybridBasisBuffer}, {3u, hybridGlobalNodesBuffer},
            {4u, hybridUnitMassBuffer}, {5u, hybridProjectionPartialsBuffer},
            {6u, hybridBlockModesBuffer}, {7u, hybridRegionSupportBuffer},
            {8u, hybridRegionSupportRowsBuffer}}, 1u, hybridModes, 1u, 256u);
        dispatchGrid3D(device, queue, hybridSparsePartials, {
            {0u, hybridBlockBuffer}, {1u, hybridProjectionPartialsBuffer},
            {2u, hybridProjectedValuesBuffer}, {3u, hybridBlockModesBuffer}},
            1u, hybridModes, 1u, 256u);
        dispatch(device, queue, hybridZeroSelected, {
            {0u, hybridBlockBuffer}, {1u, hybridGlobalNodesBuffer},
            {2u, hybridNodeModesBuffer}, {3u, coordinatesBuffer}}, 1u, 256u);
        dispatch(device, queue, hybridPackModes, {
            {0u, hybridBlockBuffer}, {1u, hybridProjectedValuesBuffer},
            {2u, coordinatesBuffer}}, 1u, 256u);
        const auto projected = readBuffer<nm_float4>(coordinatesBuffer,
            coordinates.size());
        const nm_float4 physical = physicalAction[1u];
        const std::array<float, 3u> physicalXYZ{physical.x, physical.y, physical.z};
        for (std::uint32_t mode = 0u; mode < hybridModes; ++mode) {
            double expected = 0.0;
            for (std::uint32_t axis = 0u; axis < 3u; ++axis)
                expected += double(hybridTransform[axis][mode]) * physicalXYZ[axis];
            const double actual = projected[nodesPerEnvironment + mode].x;
            require(std::abs(actual - expected) <= 4.0e-5 *
                    std::max(1.0, std::abs(expected)),
                "production sparse restriction did not apply the nonidentity B transpose");
        }
        return projected;
    };
    std::vector<double> hybridMatrix(hybridDofs * hybridDofs, 0.0);
    auto hybridCoordinateNode = [](std::uint32_t coordinate) {
        return 2u + (coordinate - hybridModes) / 3u;
    };
    auto hybridCoordinateAxis = [](std::uint32_t coordinate) {
        return (coordinate - hybridModes) % 3u;
    };
    for (std::uint32_t column = 0u; column < hybridDofs; ++column) {
        std::vector<nm_float4> hybridSource(2u * nodesPerEnvironment,
            nm_float4{});
        if (column < hybridModes) {
            hybridSource[nodesPerEnvironment + column].x = 1.0f;
        } else {
            nm_float4& value = hybridSource[hybridCoordinateNode(column)];
            if (hybridCoordinateAxis(column) == 0u) value.x = 1.0f;
            if (hybridCoordinateAxis(column) == 1u) value.y = 1.0f;
            if (hybridCoordinateAxis(column) == 2u) value.z = 1.0f;
        }
        auto hybridSourceBuffer = makeBuffer(device, hybridSource);
        auto hybridLiftedBuffer = makeBuffer(device, hybridSource);
        dispatch(device, queue, hybridLift, {
            {0u, hybridBlockBuffer}, {1u, hybridGlobalNodesBuffer},
            {2u, hybridBasisBuffer}, {3u, hybridBlockModesBuffer},
            {4u, hybridNodeModesBuffer}, {5u, hybridNodeSlotsBuffer},
            {6u, hybridSourceBuffer}, {7u, hybridLiftedBuffer}},
            nodesPerEnvironment, 256u);
        const auto lifted = readBuffer<nm_float4>(hybridLiftedBuffer,
            hybridSource.size());
        const nm_float4 selectedLift = lifted[1u];
        const std::array<float, 3u> selectedLiftXYZ{
            selectedLift.x, selectedLift.y, selectedLift.z};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            double expected = 0.0;
            for (std::uint32_t mode = 0u; mode < hybridModes; ++mode)
                expected += double(hybridTransform[axis][mode]) *
                    hybridSource[nodesPerEnvironment + mode].x;
            require(std::abs(double(selectedLiftXYZ[axis]) - expected) <=
                    4.0e-5 * std::max(1.0, std::abs(expected)),
                "production hybrid lift did not apply the nonidentity B map");
        }
        std::vector<nm_float4> physicalDirection(
            environments * nodesPerEnvironment, nm_float4{});
        for (std::uint32_t node = 0u; node < nodesPerEnvironment; ++node)
            physicalDirection[node] = lifted[node];
        std::memcpy([directionBuffer contents], physicalDirection.data(),
            physicalDirection.size() * sizeof(nm_float4));
        const auto elementAction = runDirect(0u);
        const auto physicalAction = gatherElementActions(elementAction);
        std::vector<nm_float4> action(nodesPerEnvironment, nm_float4{});
        for (std::uint32_t node = 1u; node < nodesPerEnvironment; ++node) {
            action[node] = physicalAction[node];
            action[node].x += candidate[node].positionAndMass.w * lifted[node].x;
            action[node].y += candidate[node].positionAndMass.w * lifted[node].y;
            action[node].z += candidate[node].positionAndMass.w * lifted[node].z;
        }
        const auto restricted = projectHybridAction(action);
        for (std::uint32_t mode = 0u; mode < hybridModes; ++mode)
            hybridMatrix[mode * hybridDofs + column] =
                restricted[nodesPerEnvironment + mode].x;
        for (std::uint32_t node = 2u; node < nodesPerEnvironment; ++node)
            for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
                const std::uint32_t row = hybridModes + (node - 2u) * 3u + axis;
                const nm_float4 value = restricted[node];
                const double component = axis == 0u ? value.x :
                    (axis == 1u ? value.y : value.z);
                hybridMatrix[row * hybridDofs + column] = component;
            }
    }
    double qToFine = 0.0;
    double fineToQ = 0.0;
    for (std::uint32_t q = 0u; q < hybridModes; ++q)
        for (std::uint32_t fine = hybridModes; fine < hybridDofs; ++fine) {
            qToFine += std::abs(hybridMatrix[fine * hybridDofs + q]);
            fineToQ += std::abs(hybridMatrix[q * hybridDofs + fine]);
        }
    require(qToFine > 1.0e-7 && fineToQ > 1.0e-7,
        "manufactured hybrid operator omitted q/fine off-diagonal coupling");

    std::vector<double> hybridExact(hybridDofs);
    for (std::uint32_t i = 0u; i < hybridDofs; ++i)
        hybridExact[i] = 0.2 + 0.031 * double(i + 1u);
    std::vector<double> hybridRHS(hybridDofs, 0.0);
    for (std::uint32_t row = 0u; row < hybridDofs; ++row)
        for (std::uint32_t column = 0u; column < hybridDofs; ++column)
            hybridRHS[row] += hybridMatrix[row * hybridDofs + column] *
                hybridExact[column];
    auto hybridSolve = [&](std::vector<double> matrix,
                           std::vector<double> rhs) {
        for (std::uint32_t pivot = 0u; pivot < hybridDofs; ++pivot) {
            std::uint32_t best = pivot;
            for (std::uint32_t row = pivot + 1u; row < hybridDofs; ++row)
                if (std::abs(matrix[row * hybridDofs + pivot]) >
                    std::abs(matrix[best * hybridDofs + pivot])) best = row;
            require(std::abs(matrix[best * hybridDofs + pivot]) > 1.0e-12,
                "hybrid dense oracle encountered a singular pivot");
            if (best != pivot) {
                for (std::uint32_t column = pivot; column < hybridDofs; ++column)
                    std::swap(matrix[pivot * hybridDofs + column],
                              matrix[best * hybridDofs + column]);
                std::swap(rhs[pivot], rhs[best]);
            }
            const double diagonal = matrix[pivot * hybridDofs + pivot];
            for (std::uint32_t column = pivot; column < hybridDofs; ++column)
                matrix[pivot * hybridDofs + column] /= diagonal;
            rhs[pivot] /= diagonal;
            for (std::uint32_t row = 0u; row < hybridDofs; ++row) {
                if (row == pivot) continue;
                const double factor = matrix[row * hybridDofs + pivot];
                for (std::uint32_t column = pivot; column < hybridDofs; ++column)
                    matrix[row * hybridDofs + column] -=
                        factor * matrix[pivot * hybridDofs + column];
                rhs[row] -= factor * rhs[pivot];
            }
        }
        return rhs;
    };
    const auto hybridOracle = hybridSolve(hybridMatrix, hybridRHS);
    double hybridOracleError = 0.0;
    for (std::uint32_t i = 0u; i < hybridDofs; ++i)
        hybridOracleError = std::max(hybridOracleError,
            std::abs(hybridOracle[i] - hybridExact[i]));
    require(hybridOracleError < 2.0e-4,
        "full q+fine dense oracle did not recover manufactured solution");

    auto toHybridVector = [&](const std::vector<double>& coordinates) {
        std::vector<nm_float4> vector(2u * nodesPerEnvironment, nm_float4{});
        for (std::uint32_t mode = 0u; mode < hybridModes; ++mode)
            vector[nodesPerEnvironment + mode].x =
                static_cast<float>(coordinates[mode]);
        for (std::uint32_t node = 2u; node < nodesPerEnvironment; ++node)
            for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
                nm_float4& value = vector[node];
                const float component = static_cast<float>(coordinates[
                    hybridModes + (node - 2u) * 3u + axis]);
                if (axis == 0u) value.x = component;
                if (axis == 1u) value.y = component;
                if (axis == 2u) value.z = component;
            }
        return vector;
    };
    auto fromHybridVector = [&](const std::vector<nm_float4>& vector) {
        std::vector<double> coordinates(hybridDofs);
        for (std::uint32_t mode = 0u; mode < hybridModes; ++mode)
            coordinates[mode] = vector[nodesPerEnvironment + mode].x;
        for (std::uint32_t node = 2u; node < nodesPerEnvironment; ++node)
            for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
                const nm_float4& value = vector[node];
                coordinates[hybridModes + (node - 2u) * 3u + axis] =
                    axis == 0u ? value.x : (axis == 1u ? value.y : value.z);
            }
        return coordinates;
    };
    auto multiplyHybrid = [&](const std::vector<double>& value) {
        std::vector<double> result(hybridDofs, 0.0);
        for (std::uint32_t row = 0u; row < hybridDofs; ++row)
            for (std::uint32_t column = 0u; column < hybridDofs; ++column)
                result[row] += hybridMatrix[row * hybridDofs + column] *
                    value[column];
        return result;
    };
    double rhsNorm2 = 0.0;
    for (double value : hybridRHS) rhsNorm2 += value * value;
    const double rhsNorm = std::sqrt(rhsNorm2);
    std::vector<double> firstBasis(hybridDofs);
    for (std::uint32_t i = 0u; i < hybridDofs; ++i)
        firstBasis[i] = hybridRHS[i] / rhsNorm;
    const auto firstAction = multiplyHybrid(firstBasis);
    NMMatterDispatchGPU hybridDispatch{};
    hybridDispatch.abiVersion = NM_MATTER_ABI_VERSION;
    hybridDispatch.environmentCount = 1u;
    hybridDispatch.objectCount = 1u;
    hybridDispatch.femNodeCount = nodesPerEnvironment;
    hybridDispatch.tetrahedronCount = tetrahedraPerEnvironment;
    auto hybridDispatchBuffer = makeBuffer(device,
        std::vector<NMMatterDispatchGPU>{hybridDispatch});
    NMFGMRESLayoutGPU hybridLayout{};
    hybridLayout.unknownCount = 2u * nodesPerEnvironment;
    hybridLayout.reserved = 5u;
    hybridLayout.vascularReserved = hybridModes;
    auto hybridLayoutBuffer = makeBuffer(device,
        std::vector<NMFGMRESLayoutGPU>{hybridLayout});
    NMMixedSolverGPU hybridSolver{};
    hybridSolver.nonlinearIterations.y = NM_MIXED_FGMRES_RESTART;
    hybridSolver.residualTolerances.x = 1.0e-7f;
    hybridSolver.regularization.x = 1.0e-12f;
    auto hybridSolverBuffer = makeBuffer(device,
        std::vector<NMMixedSolverGPU>{hybridSolver});
    auto hybridCGSPartial = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_cgs2_partial");
    auto hybridCGSRow = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_cgs2_reduce_row");
    auto hybridCGSUpdate = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_cgs2_update");
    auto hybridMeasureNorm = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_cgs2_measure_norm");
    auto hybridFinishColumn = pipelineWithRelativeInnerForcingDisabled(
        device, library, "numi_matter_metal::nm_fgmres_cgs2_finish_column");
    auto hybridNormalize = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_cgs2_normalize");
    auto hybridColumnBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto hybridPassBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto hybridForcingBuffer = makeBuffer(device, std::vector<float>{0.05f});
    auto hybridNodeActive = makeBuffer(device,
        std::vector<NMIndirectDispatchGPU>{NMIndirectDispatchGPU{}});
    auto hybridElementActive = makeBuffer(device,
        std::vector<NMIndirectDispatchGPU>{NMIndirectDispatchGPU{}});
    NMFGMRESStateGPU hybridStateInitial{};
    hybridStateInitial.diagnostics = {
        static_cast<float>(rhsNorm), static_cast<float>(rhsNorm), 0.0f, 0.0f};
    auto hybridStates = makeBuffer(device,
        std::vector<NMFGMRESStateGPU>{hybridStateInitial});
    const std::uint32_t hybridPartialCount = 8u;
    auto hybridPartialsBuffer = makeBuffer(device, std::vector<float>(
        2u * (NM_MIXED_FGMRES_RESTART + 1u) * hybridPartialCount, 0.0f));
    auto hybridHessenbergBuffer = makeBuffer(device, std::vector<float>(
        (NM_MIXED_FGMRES_RESTART + 1u) * NM_MIXED_FGMRES_RESTART, 0.0f));
    auto hybridRotationsBuffer = makeBuffer(device, std::vector<NativeFloat2>(
        NM_MIXED_FGMRES_RESTART, NativeFloat2{1.0f, 0.0f}));
    std::vector<float> hybridLeastSquares(NM_MIXED_FGMRES_RESTART + 1u, 0.0f);
    hybridLeastSquares[0] = static_cast<float>(rhsNorm);
    auto hybridLeastSquaresBuffer = makeBuffer(device, hybridLeastSquares);
    auto hybridBasisValues = toHybridVector(firstBasis);
    auto hybridActionValues = toHybridVector(firstAction);
    auto hybridArnoldiBasisBuffer = makeBuffer(device, std::vector<nm_float4>(
        (NM_MIXED_FGMRES_RESTART + 1u) * hybridLayout.unknownCount,
        nm_float4{}));
    std::memcpy([hybridArnoldiBasisBuffer contents], hybridBasisValues.data(),
        hybridBasisValues.size() * sizeof(nm_float4));
    auto hybridWorkBuffer = makeBuffer(device, hybridActionValues);
    auto hybridNextBasisBuffer = makeBuffer(device,
        std::vector<nm_float4>(hybridLayout.unknownCount, nm_float4{}));
    auto runHybridPass = [&](std::uint32_t passIndex) {
        std::memcpy([hybridPassBuffer contents], &passIndex, sizeof(passIndex));
        dispatch(device, queue, hybridCGSPartial, {
            {0u, hybridDispatchBuffer}, {1u, hybridColumnBuffer},
            {2u, hybridPassBuffer}, {3u, hybridArnoldiBasisBuffer},
            {4u, hybridWorkBuffer}, {5u, hybridPartialsBuffer},
            {6u, hybridStates}, {30u, hybridLayoutBuffer}}, 1u, 256u);
        dispatch(device, queue, hybridCGSRow, {
            {0u, hybridDispatchBuffer}, {30u, hybridLayoutBuffer},
            {1u, hybridColumnBuffer}, {2u, hybridPassBuffer},
            {3u, hybridHessenbergBuffer}, {4u, hybridPartialsBuffer},
            {5u, hybridStates}}, 1u, 32u);
        dispatch(device, queue, hybridCGSUpdate, {
            {0u, hybridDispatchBuffer}, {30u, hybridLayoutBuffer},
            {1u, hybridColumnBuffer}, {2u, hybridPassBuffer},
            {3u, hybridArnoldiBasisBuffer}, {4u, hybridWorkBuffer},
            {5u, hybridHessenbergBuffer}, {6u, hybridPartialsBuffer},
            {7u, hybridStates}}, 1u, 256u);
    };
    runHybridPass(0u);
    runHybridPass(1u);
    dispatch(device, queue, hybridMeasureNorm, {
        {0u, hybridDispatchBuffer}, {30u, hybridLayoutBuffer},
        {1u, hybridWorkBuffer}, {2u, hybridPartialsBuffer},
        {3u, hybridStates}}, 1u, 256u);
    dispatch(device, queue, hybridFinishColumn, {
        {0u, hybridDispatchBuffer}, {30u, hybridLayoutBuffer},
        {1u, hybridSolverBuffer}, {2u, hybridColumnBuffer},
        {3u, hybridHessenbergBuffer}, {4u, hybridRotationsBuffer},
        {5u, hybridLeastSquaresBuffer}, {6u, hybridStates},
        {7u, hybridForcingBuffer}, {8u, hybridPartialsBuffer},
        {9u, hybridNodeActive}, {10u, hybridElementActive}}, 1u, 32u);
    dispatch(device, queue, hybridNormalize, {
        {0u, hybridDispatchBuffer}, {30u, hybridLayoutBuffer},
        {1u, hybridColumnBuffer}, {2u, hybridWorkBuffer},
        {3u, hybridNextBasisBuffer}, {4u, hybridPartialsBuffer},
        {5u, hybridStates}}, 1u, 256u);
    const auto hybridNext = readBuffer<nm_float4>(hybridNextBasisBuffer,
        hybridLayout.unknownCount);
    const auto hybridNextCoordinates = fromHybridVector(hybridNext);
    double hybridNextNorm2 = 0.0;
    double hybridOrthogonalDot = 0.0;
    for (std::uint32_t i = 0u; i < hybridDofs; ++i) {
        hybridNextNorm2 += hybridNextCoordinates[i] * hybridNextCoordinates[i];
        hybridOrthogonalDot += hybridNextCoordinates[i] * firstBasis[i];
    }
    require(std::isfinite(hybridNextNorm2) &&
            std::abs(std::sqrt(hybridNextNorm2) - 1.0) < 2.0e-4 &&
            std::abs(hybridOrthogonalDot) < 2.0e-4,
        "production CGS2 failed to orthogonalize the coupled q+fine operator image");
    // Complete a production FGMRES cycle on the manufactured q+fine operator.
    // Arnoldi images use the GPU FEM tangent, hybrid lift, and sparse
    // restriction. The dense matrix is a separate small oracle; a single
    // CGS2 step cannot catch cycle/correction layout bugs.
    const std::uint32_t cycleLimit = hybridDofs;
    const std::uint32_t cycleUnknowns = hybridLayout.unknownCount;
    hybridSolver.nonlinearIterations.y = cycleLimit;
    hybridSolver.residualTolerances.x = 1.0e-8f; // Roundoff-floor one-cycle control, not the certified restart test.
    auto cycleSolverBuffer = makeBuffer(device,
        std::vector<NMMixedSolverGPU>{hybridSolver});
    std::vector<nm_float4> cycleBasis(
        (NM_MIXED_FGMRES_RESTART + 1u) * cycleUnknowns, nm_float4{});
    const auto initialHybridBasis = toHybridVector(firstBasis);
    std::memcpy(cycleBasis.data(), initialHybridBasis.data(),
        cycleUnknowns * sizeof(nm_float4));
    auto cycleBasisBuffer = makeBuffer(device, cycleBasis);
    std::vector<nm_float4> cyclePreconditionedBasis(
        NM_MIXED_FGMRES_RESTART * cycleUnknowns, nm_float4{});
    auto cyclePreconditionedBasisBuffer = makeBuffer(device,
        cyclePreconditionedBasis);
    auto cycleWorkBuffer = makeBuffer(device, std::vector<nm_float4>(
        cycleUnknowns, nm_float4{}));
    auto cycleNextBasisBuffer = makeBuffer(device, std::vector<nm_float4>(
        cycleUnknowns, nm_float4{}));
    auto cyclePartialsBuffer = makeBuffer(device, std::vector<float>(
        2u * (NM_MIXED_FGMRES_RESTART + 1u) * hybridPartialCount, 0.0f));
    auto cycleHessenbergBuffer = makeBuffer(device, std::vector<float>(
        (NM_MIXED_FGMRES_RESTART + 1u) * NM_MIXED_FGMRES_RESTART, 0.0f));
    auto cycleRotationsBuffer = makeBuffer(device, std::vector<NativeFloat2>(
        NM_MIXED_FGMRES_RESTART, NativeFloat2{1.0f, 0.0f}));
    std::vector<float> cycleLeastSquares(
        NM_MIXED_FGMRES_RESTART + 1u, 0.0f);
    cycleLeastSquares[0] = static_cast<float>(rhsNorm);
    auto cycleLeastSquaresBuffer = makeBuffer(device, cycleLeastSquares);
    auto cycleStates = makeBuffer(device,
        std::vector<NMFGMRESStateGPU>{hybridStateInitial});
    auto cycleStatuses = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
    auto cycleColumnBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto cyclePassBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto cycleForcingBuffer = makeBuffer(device, std::vector<float>{1.0e-8f});
    auto cycleNodeActive = makeBuffer(device,
        std::vector<NMIndirectDispatchGPU>{NMIndirectDispatchGPU{}});
    auto cycleElementActive = makeBuffer(device,
        std::vector<NMIndirectDispatchGPU>{NMIndirectDispatchGPU{}});
    auto cycleFinePrecondition = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_precondition");
    auto cycleSolveRegionBlocks = pipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_solve_region_blocks");
    auto cyclePackUniqueQ = pipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_hybrid_pack_unique_preconditioned_q");
    std::vector<nm_float4> inverseDiagonal(nodesPerEnvironment, nm_float4{1.0f, 1.0f, 1.0f, 1.0f});
    for (std::uint32_t node = 2u; node < nodesPerEnvironment; ++node)
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            const std::uint32_t coordinate =
                hybridModes + (node - 2u) * 3u + axis;
            const float diagonal = static_cast<float>(
                hybridMatrix[coordinate * hybridDofs + coordinate]);
            require(std::isfinite(diagonal) && std::abs(diagonal) > 1.0e-10f,
                "hybrid fine diagonal preconditioner has a singular row");
            float& inverse = axis == 0u ? inverseDiagonal[node].x :
                (axis == 1u ? inverseDiagonal[node].y : inverseDiagonal[node].z);
            inverse = 1.0f / diagonal;
        }
    auto cycleInverseDiagonalBuffer = makeBuffer(device, inverseDiagonal);
    std::vector<float> qFactors(hybridModes * hybridModes);
    std::vector<std::uint32_t> qPivots(hybridModes);
    for (std::uint32_t row = 0u; row < hybridModes; ++row)
        for (std::uint32_t column = 0u; column < hybridModes; ++column)
            qFactors[row * hybridModes + column] = static_cast<float>(
                hybridMatrix[row * hybridDofs + column]);
    for (std::uint32_t pivot = 0u; pivot < hybridModes; ++pivot) {
        std::uint32_t best = pivot;
        for (std::uint32_t row = pivot + 1u; row < hybridModes; ++row)
            if (std::abs(qFactors[row * hybridModes + pivot]) >
                std::abs(qFactors[best * hybridModes + pivot])) best = row;
        require(std::isfinite(qFactors[best * hybridModes + pivot]) &&
                std::abs(qFactors[best * hybridModes + pivot]) > 1.0e-10f,
            "hybrid q block preconditioner has a singular pivot");
        qPivots[pivot] = best;
        if (best != pivot)
            for (std::uint32_t column = 0u; column < hybridModes; ++column)
                std::swap(qFactors[pivot * hybridModes + column],
                          qFactors[best * hybridModes + column]);
        for (std::uint32_t row = pivot + 1u; row < hybridModes; ++row) {
            qFactors[row * hybridModes + pivot] /=
                qFactors[pivot * hybridModes + pivot];
            for (std::uint32_t column = pivot + 1u; column < hybridModes; ++column)
                qFactors[row * hybridModes + column] -=
                    qFactors[row * hybridModes + pivot] *
                    qFactors[pivot * hybridModes + column];
        }
    }
    auto cycleQFactorsBuffer = makeBuffer(device, qFactors);
    auto cycleQPivotsBuffer = makeBuffer(device, qPivots);
    auto cycleRegionValidBuffer = makeBuffer(device, std::vector<std::uint32_t>{1u});
    auto cycleFactorValidBuffer = makeBuffer(device, std::vector<std::uint32_t>{1u});
    auto cycleQCoefficientsBuffer = makeBuffer(device, std::vector<float>(hybridModes, 0.0f));
    auto cycleSolveValidBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto cycleQInputBuffer = makeBuffer(device, std::vector<float>(hybridModes, 0.0f));
    auto cycleInputBuffer = makeBuffer(device, std::vector<nm_float4>(
        cycleUnknowns, nm_float4{}));
    auto cyclePreconditionedColumnBuffer = makeBuffer(device, std::vector<nm_float4>(
        cycleUnknowns, nm_float4{}));
    auto cycleLiftedBuffer = makeBuffer(device, std::vector<nm_float4>(
        cycleUnknowns, nm_float4{}));
    bool sawNonzeroReducedPreconditionerInput = false;
    double maximumReducedPreconditionerChange = 0.0;
    auto preconditionHybrid = [&](const std::vector<nm_float4>& input) {
        require(input.size() == cycleUnknowns,
            "hybrid preconditioner received an invalid vector extent");
        std::vector<nm_float4> zero(cycleUnknowns, nm_float4{});
        std::memcpy([cycleInputBuffer contents], input.data(),
            input.size() * sizeof(nm_float4));
        std::memcpy([cyclePreconditionedColumnBuffer contents], zero.data(),
            zero.size() * sizeof(nm_float4));
        dispatch(device, queue, cycleFinePrecondition, {
            {0u, hybridDispatchBuffer}, {1u, nodeRangesBuffer},
            {2u, cycleInverseDiagonalBuffer}, {3u, cycleInputBuffer},
            {4u, cyclePreconditionedColumnBuffer}, {5u, cycleStates},
            {6u, cycleSolverBuffer}, {7u, objectsBuffer}},
            1u, 256u);
        dispatch(device, queue, hybridClearInactive, {
            {0u, hybridBlockBuffer}, {1u, cyclePreconditionedColumnBuffer}},
            1u, 256u);
        dispatch(device, queue, hybridGatherQRHS, {
            {0u, hybridBlockBuffer}, {1u, cycleInputBuffer},
            {2u, cycleQInputBuffer}}, 1u, 256u);
        dispatch(device, queue, cycleSolveRegionBlocks, {
            {0u, hybridBlockBuffer}, {1u, cycleQInputBuffer},
            {2u, hybridBlockModesBuffer}, {3u, cycleQFactorsBuffer},
            {4u, cycleQPivotsBuffer}, {5u, cycleRegionValidBuffer},
            {6u, cycleFactorValidBuffer}, {7u, cycleQCoefficientsBuffer},
            {8u, cycleSolveValidBuffer}}, 1u, 32u);
        dispatch(device, queue, cyclePackUniqueQ, {
            {0u, hybridBlockBuffer}, {1u, hybridBlockModesBuffer},
            {2u, cycleQCoefficientsBuffer}, {3u, cycleSolveValidBuffer},
            {4u, cycleInputBuffer}, {5u, cyclePreconditionedColumnBuffer}},
            1u, 32u);
        const auto qSolveValid = readBuffer<std::uint32_t>(
            cycleSolveValidBuffer, 1u);
        require(qSolveValid[0] == 1u,
            "production hybrid q-block solve did not publish a valid correction");
        const auto qPreconditioned = readBuffer<nm_float4>(
            cyclePreconditionedColumnBuffer, cycleUnknowns);
        for (std::uint32_t mode = 0u; mode < hybridModes; ++mode) {
            const float before = input[nodesPerEnvironment + mode].x;
            const float after = qPreconditioned[nodesPerEnvironment + mode].x;
            if (std::abs(before) > 1.0e-7f)
                sawNonzeroReducedPreconditionerInput = true;
            maximumReducedPreconditionerChange = std::max(
                maximumReducedPreconditionerChange,
                std::abs(double(after) - before));
        }
        dispatch(device, queue, hybridZeroSelected, {
            {0u, hybridBlockBuffer}, {1u, hybridGlobalNodesBuffer},
            {2u, hybridNodeModesBuffer}, {3u, cyclePreconditionedColumnBuffer}},
            1u, 256u);
        return readBuffer<nm_float4>(cyclePreconditionedColumnBuffer,
            cycleUnknowns);
    };
    auto applyHybridOperator = [&](const std::vector<nm_float4>& input) {
        require(input.size() == cycleUnknowns,
            "hybrid operator received an invalid vector extent");
        std::memcpy([cycleInputBuffer contents], input.data(),
            input.size() * sizeof(nm_float4));
        std::memcpy([cycleLiftedBuffer contents], input.data(),
            input.size() * sizeof(nm_float4));
        dispatch(device, queue, hybridLiftCopy, {
            {0u, hybridLayoutBuffer}, {1u, cycleInputBuffer},
            {2u, cycleLiftedBuffer}}, 1u, 256u);
        dispatch(device, queue, hybridLift, {
            {0u, hybridBlockBuffer}, {1u, hybridGlobalNodesBuffer},
            {2u, hybridBasisBuffer}, {3u, hybridBlockModesBuffer},
            {4u, hybridNodeModesBuffer}, {5u, hybridNodeSlotsBuffer},
            {6u, cycleInputBuffer}, {7u, cycleLiftedBuffer}}, 1u, 256u);
        dispatch(device, queue, hybridClearInactive, {
            {0u, hybridBlockBuffer}, {1u, cycleLiftedBuffer}}, 1u, 256u);
        const auto lifted = readBuffer<nm_float4>(cycleLiftedBuffer, cycleUnknowns);
        std::vector<nm_float4> physicalDirection(
            environments * nodesPerEnvironment, nm_float4{});
        std::copy(lifted.begin(), lifted.begin() + nodesPerEnvironment,
                  physicalDirection.begin());
        std::memcpy([directionBuffer contents], physicalDirection.data(),
            physicalDirection.size() * sizeof(nm_float4));
        const auto elementValues = runDirect(0u);
        const auto physicalForces = gatherElementActions(elementValues);
        std::vector<nm_float4> action(nodesPerEnvironment, nm_float4{});
        for (std::uint32_t node = 1u; node < nodesPerEnvironment; ++node) {
            action[node] = physicalForces[node];
            action[node].x += candidate[node].positionAndMass.w * lifted[node].x;
            action[node].y += candidate[node].positionAndMass.w * lifted[node].y;
            action[node].z += candidate[node].positionAndMass.w * lifted[node].z;
        }
        return projectHybridAction(action);
    };
    bool sawNonzeroFinePreconditionerInput = false;
    for (std::uint32_t column = 0u; column < cycleLimit; ++column) {
        std::vector<nm_float4> current(cycleUnknowns);
        std::copy(cycleBasis.begin() + std::size_t(column) * cycleUnknowns,
                  cycleBasis.begin() + std::size_t(column + 1u) * cycleUnknowns,
                  current.begin());
        const auto preconditioned = preconditionHybrid(current);
        const std::size_t preconditionedBase =
            std::size_t(column) * cycleUnknowns;
        std::memcpy(cyclePreconditionedBasis.data() + preconditionedBase,
            preconditioned.data(), cycleUnknowns * sizeof(nm_float4));
        std::memcpy(static_cast<char*>([cyclePreconditionedBasisBuffer contents]) +
                preconditionedBase * sizeof(nm_float4),
            preconditioned.data(), cycleUnknowns * sizeof(nm_float4));
        double fineInputNorm2 = 0.0;
        double finePreconditionedNorm2 = 0.0;
        double finePreconditionerDifference2 = 0.0;
        for (std::uint32_t node = 2u; node < nodesPerEnvironment; ++node) {
            const nm_float4& before = current[node];
            const nm_float4& after = preconditioned[node];
            fineInputNorm2 += double(before.x) * before.x +
                double(before.y) * before.y + double(before.z) * before.z;
            finePreconditionedNorm2 += double(after.x) * after.x +
                double(after.y) * after.y + double(after.z) * after.z;
            const double dx = double(after.x) - before.x;
            const double dy = double(after.y) - before.y;
            const double dz = double(after.z) - before.z;
            finePreconditionerDifference2 += dx * dx + dy * dy + dz * dz;
        }
        if (fineInputNorm2 > 1.0e-12) {
            require(finePreconditionedNorm2 > 1.0e-12 &&
                    finePreconditionerDifference2 > 1.0e-12,
                "production fine preconditioner did not transform its nonzero input");
            sawNonzeroFinePreconditionerInput = true;
        }
        const auto imageVector = applyHybridOperator(preconditioned);
        const auto imageCoordinates = fromHybridVector(imageVector);
        (void)imageCoordinates;
        std::memcpy([cycleWorkBuffer contents], imageVector.data(),
            cycleUnknowns * sizeof(nm_float4));
        std::memcpy([cycleColumnBuffer contents], &column, sizeof(column));
        for (std::uint32_t passIndex = 0u; passIndex < 2u; ++passIndex) {
            std::memcpy([cyclePassBuffer contents], &passIndex,
                sizeof(passIndex));
            dispatch(device, queue, hybridCGSPartial, {
                {0u, hybridDispatchBuffer}, {1u, cycleColumnBuffer},
                {2u, cyclePassBuffer}, {3u, cycleBasisBuffer},
                {4u, cycleWorkBuffer}, {5u, cyclePartialsBuffer},
                {6u, cycleStates}, {30u, hybridLayoutBuffer}}, 1u, 256u);
            dispatch(device, queue, hybridCGSRow, {
                {0u, hybridDispatchBuffer}, {30u, hybridLayoutBuffer},
                {1u, cycleColumnBuffer}, {2u, cyclePassBuffer},
                {3u, cycleHessenbergBuffer}, {4u, cyclePartialsBuffer},
                {5u, cycleStates}}, 1u, 32u);
            dispatch(device, queue, hybridCGSUpdate, {
                {0u, hybridDispatchBuffer}, {30u, hybridLayoutBuffer},
                {1u, cycleColumnBuffer}, {2u, cyclePassBuffer},
                {3u, cycleBasisBuffer}, {4u, cycleWorkBuffer},
                {5u, cycleHessenbergBuffer}, {6u, cyclePartialsBuffer},
                {7u, cycleStates}}, 1u, 256u);
        }
        dispatch(device, queue, hybridMeasureNorm, {
            {0u, hybridDispatchBuffer}, {30u, hybridLayoutBuffer},
            {1u, cycleWorkBuffer}, {2u, cyclePartialsBuffer},
            {3u, cycleStates}}, 1u, 256u);
        dispatch(device, queue, hybridFinishColumn, {
            {0u, hybridDispatchBuffer}, {30u, hybridLayoutBuffer},
            {1u, cycleSolverBuffer}, {2u, cycleColumnBuffer},
            {3u, cycleHessenbergBuffer}, {4u, cycleRotationsBuffer},
            {5u, cycleLeastSquaresBuffer}, {6u, cycleStates},
            {7u, cycleForcingBuffer}, {8u, cyclePartialsBuffer},
            {9u, cycleNodeActive}, {10u, cycleElementActive}},
            1u, 32u);
        dispatch(device, queue, hybridNormalize, {
            {0u, hybridDispatchBuffer}, {30u, hybridLayoutBuffer},
            {1u, cycleColumnBuffer}, {2u, cycleWorkBuffer},
            {3u, cycleNextBasisBuffer}, {4u, cyclePartialsBuffer},
            {5u, cycleStates}}, 1u, 256u);
        const auto cycleNext = readBuffer<nm_float4>(
            cycleNextBasisBuffer, cycleUnknowns);
        std::memcpy(cycleBasis.data() +
                std::size_t(column + 1u) * cycleUnknowns,
            cycleNext.data(), cycleUnknowns * sizeof(nm_float4));
        std::memcpy([cycleBasisBuffer contents], cycleBasis.data(),
            cycleBasis.size() * sizeof(nm_float4));
        const auto cycleState = readBuffer<NMFGMRESStateGPU>(cycleStates, 1u)[0];
        if (cycleState.diagnostics.z > 0.5f ||
            cycleState.diagnostics.w != float(column + 1u)) break;
    }
    require(sawNonzeroFinePreconditionerInput,
        "full hybrid FGMRES cycle never exercised the fine preconditioner on nonzero data");
    require(sawNonzeroReducedPreconditionerInput &&
            maximumReducedPreconditionerChange > 1.0e-7,
        "full hybrid FGMRES cycle did not exercise a nontrivial q-block solve");
    const std::uint32_t cycleIterationOffset = 0u;
    const std::uint32_t finalCycle = 1u;
    const std::uint32_t deferCompletion = 0u;
    auto restartCoefficientsBuffer = makeBuffer(device, std::vector<float>(
        NM_MIXED_FGMRES_RESTART + 1u, 0.0f));
    auto finalizeCycle = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_finalize_cycle");
    dispatch(device, queue, finalizeCycle, {
        {0u, hybridDispatchBuffer}, {30u, hybridLayoutBuffer},
        {1u, cycleSolverBuffer}, {2u, cycleHessenbergBuffer},
        {3u, cycleLeastSquaresBuffer}, {4u, cycleStates},
        {5u, cycleStatuses},
        {6u, makeBuffer(device, std::vector<std::uint32_t>{cycleIterationOffset})},
        {7u, makeBuffer(device, std::vector<std::uint32_t>{finalCycle})},
        {8u, cycleForcingBuffer}, {9u, cycleRotationsBuffer},
        {10u, restartCoefficientsBuffer},
        {11u, makeBuffer(device, std::vector<std::uint32_t>{deferCompletion})}},
        1u, 32u);
    auto cycleSolutionBuffer = makeBuffer(device,
        std::vector<nm_float4>(cycleUnknowns, nm_float4{}));
    auto cycleRestartBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto cycleAccumulate = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_accumulate");
    auto cycleAccumulateFields = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_accumulate_fields");
    dispatch(device, queue, cycleAccumulate, {
        {0u, hybridDispatchBuffer}, {30u, hybridLayoutBuffer},
        {1u, cycleSolverBuffer}, {2u, nodeRangesBuffer},
        {3u, cyclePreconditionedBasisBuffer}, {4u, cycleLeastSquaresBuffer},
        {5u, cycleStates}, {6u, cycleSolutionBuffer},
        {7u, cycleRestartBuffer}}, nodesPerEnvironment, 256u);
    dispatch(device, queue, cycleAccumulateFields, {
        {0u, hybridDispatchBuffer}, {30u, hybridLayoutBuffer},
        {1u, cycleSolverBuffer}, {2u, nodeRangesBuffer},
        {3u, cyclePreconditionedBasisBuffer}, {4u, cycleLeastSquaresBuffer},
        {5u, cycleStates}, {6u, cycleSolutionBuffer},
        {7u, cycleRestartBuffer}}, nodesPerEnvironment, 256u);
    const auto cycleStateFinal =
        readBuffer<NMFGMRESStateGPU>(cycleStates, 1u)[0];
    const auto cycleStatusFinal =
        readBuffer<NMMatterStatusGPU>(cycleStatuses, 1u)[0];
    const auto cycleSolution =
        fromHybridVector(readBuffer<nm_float4>(cycleSolutionBuffer,
            cycleUnknowns));
    const auto cycleImage = multiplyHybrid(cycleSolution);
    double cycleResidual2 = 0.0;
    double cycleSolutionError2 = 0.0;
    double cycleOracleNorm2 = 0.0;
    for (std::uint32_t i = 0u; i < hybridDofs; ++i) {
        const double residual = cycleImage[i] - hybridRHS[i];
        cycleResidual2 += residual * residual;
        const double error = cycleSolution[i] - hybridOracle[i];
        cycleSolutionError2 += error * error;
        cycleOracleNorm2 += hybridOracle[i] * hybridOracle[i];
    }
    const double cycleRelativeResidual = std::sqrt(cycleResidual2) / rhsNorm;
    const double cycleRelativeSolutionError =
        std::sqrt(cycleSolutionError2) /
        std::max(1.0e-30, std::sqrt(cycleOracleNorm2));
    require(cycleStatusFinal.code == NM_STATUS_SUCCESS &&
            cycleStateFinal.diagnostics.w >= 2.0f &&
            cycleRelativeResidual < 5.0e-4 &&
            cycleRelativeSolutionError < 5.0e-3,
        "production full FGMRES cycle failed the dense q+fine oracle");
    std::cout << "hybrid_q_fine_roundoff_control=true forcing=1e-8 "
                 "single_cycle_only=true certified=false relative_residual="
              << cycleRelativeResidual << " result=CONTROL\n";
    std::cout << "hybrid_q_fine_full_FGMRES=true gpu_lift=true "
                 "gpu_sparse_restrict=true gpu_q_solve=true "
                 "gpu_fine_precondition=true gpu_q_block_solve=true "
                 "host_element_gather=true columns="
              << cycleStateFinal.diagnostics.w
              << " relative_residual=" << cycleRelativeResidual
              << " relative_solution_error=" << cycleRelativeSolutionError
              << " result=PASS" << std::endl;


    // Keep the one-cycle solve above as a control, then force production
    // distributed-begin and true-residual restarts on the same q+fine map.
    const std::uint32_t restartedLimit=1u,restartedMaximumCycles=32u;
    hybridSolver.nonlinearIterations.y=restartedLimit;
    hybridSolver.residualTolerances.x=1.0e-6f;
    NMFGMRESLayoutGPU restartedLayout=hybridLayout;
    restartedLayout.supportBase=2u*nodesPerEnvironment;
    restartedLayout.vascularBase=2u*nodesPerEnvironment;
    auto restartedLayoutBuffer=makeBuffer(device,
        std::vector<NMFGMRESLayoutGPU>{restartedLayout});
    auto restartedSolverBuffer=makeBuffer(device,std::vector<NMMixedSolverGPU>{hybridSolver});
    std::memcpy([cycleSolverBuffer contents],&hybridSolver,sizeof(hybridSolver));
    auto restartedResidualBuffer=makeBuffer(device,toHybridVector(hybridRHS));
    auto restartedRHSBuffer=makeBuffer(device,toHybridVector(hybridRHS));
    auto restartedOperatorBuffer=makeBuffer(device,std::vector<nm_float4>(cycleUnknowns,nm_float4{}));
    auto restartedMetricsBuffer=makeBuffer(device,std::vector<nm_float4>(1u,nm_float4{}));
    auto restartedTruePartialsBuffer=makeBuffer(device,std::vector<nm_float4>(1u,nm_float4{}));
    auto restartedReplaceBuffer=makeBuffer(device,std::vector<std::uint32_t>{3u});
    auto restartedPropagateBuffer=makeBuffer(device,std::vector<std::uint32_t>{0u});
    auto restartedRestartCycleBuffer=makeBuffer(device,std::vector<std::uint32_t>{0u});
    auto restartedIterationBuffer=makeBuffer(device,std::vector<std::uint32_t>{1u});
    auto restartedBeginForcingBuffer=makeBuffer(device,std::vector<float>{1.0e-6f});
    auto restartedBeginPartialsBuffer=makeBuffer(device,std::vector<nm_float4>(1u,nm_float4{}));
    auto restartedDomainPartialsBuffer=makeBuffer(device,std::vector<NativeFloat2>(1u,NativeFloat2{}));
    auto restartedObjectsBuffer=makeBuffer(device,std::vector{object});
    auto restartedCandidateBuffer=makeBuffer(device,std::vector<NMFEMNodeStateGPU>(
        candidate.begin(),candidate.begin()+nodesPerEnvironment));
    auto restartedSchedulerBuffer=makeBuffer(device,std::vector<NMSchedulerStateGPU>{schedulers[0]});
    auto restartedAdaptiveBuffer=makeBuffer(device,std::vector<NMAdaptiveStateGPU>{adaptive[0]});
    auto restartedMicroBuffer=makeBuffer(device,std::vector{micro});
    auto restartedRigidCandidateBuffer=makeBuffer(device,std::vector<float>{0.0f});
    auto restartedSupportHistoryBuffer=makeBuffer(device,std::vector<nm_float4>{nm_float4{}});
    auto restartedSupportBuffer=makeBuffer(device,
        std::vector<NMHumanSupportDispatchGPU>{NMHumanSupportDispatchGPU{}});
    auto restartedSupportContactsBuffer=makeBuffer(device,
        std::vector<NMHumanSupportContactGPU>{NMHumanSupportContactGPU{}});
    id<MTLFunction> restartContactLayout=[library newFunctionWithName:
        @"numi_matter_metal::nm_primal_contact_argument_layout"];
    require(restartContactLayout!=nil,"hybrid restart contact layout missing");
    id<MTLArgumentEncoder> restartContactEncoder=
        [restartContactLayout newArgumentEncoderWithBufferIndex:0u];
    require(restartContactEncoder!=nil,"hybrid restart contact encoder missing");
    id<MTLBuffer> restartContactArguments=[device newBufferWithLength:
        restartContactEncoder.encodedLength options:MTLResourceStorageModeShared];
    require(restartContactArguments!=nil,"hybrid restart contact buffer allocation failed");
    auto restartZeroPointer=makeBuffer(device,std::vector<std::uint32_t>{0u});
    [restartContactEncoder setArgumentBuffer:restartContactArguments offset:0u];
    for(NSUInteger index=0u;index<20u;++index)
        [restartContactEncoder setBuffer:restartZeroPointer offset:0u atIndex:index];
    auto beginRestartPartial=pipeline(device,library,
        "numi_matter_metal::nm_fgmres_begin_distributed_partial");
    auto beginRestartFinish=pipeline(device,library,
        "numi_matter_metal::nm_fgmres_begin_distributed_finish");
    auto beginRestartBasis0=pipeline(device,library,
        "numi_matter_metal::nm_fgmres_begin_distributed_basis0");
    auto trueRestartPartial=pipeline(device,library,
        "numi_matter_metal::nm_fgmres_true_residual_distributed_partial");
    auto trueRestartFinish=pipeline(device,library,
        "numi_matter_metal::nm_fgmres_true_residual_distributed_finish");
    auto trueRestartReplace=pipeline(device,library,
        "numi_matter_metal::nm_fgmres_true_residual_distributed_replace");
    auto restartedSolutionBuffer=makeBuffer(device,std::vector<nm_float4>(cycleUnknowns,nm_float4{}));
    auto restartedBasisBuffer=makeBuffer(device,std::vector<nm_float4>(
        (NM_MIXED_FGMRES_RESTART+1u)*cycleUnknowns,nm_float4{}));
    auto restartedPreconditionedBasisBuffer=makeBuffer(device,std::vector<nm_float4>(
        NM_MIXED_FGMRES_RESTART*cycleUnknowns,nm_float4{}));
    auto restartBeginBindings=std::vector<std::pair<NSUInteger,id<MTLBuffer>>>{
        {0u,hybridDispatchBuffer},{30u,restartedLayoutBuffer},
        {1u,restartedSolverBuffer},{2u,restartedObjectsBuffer},
        {3u,restartedResidualBuffer},{4u,restartedBasisBuffer},
        {5u,cycleHessenbergBuffer},{6u,cycleRotationsBuffer},
        {7u,cycleLeastSquaresBuffer},{8u,cycleStates},{9u,cycleStatuses},
        {10u,restartedMicroBuffer},{11u,restartedSchedulerBuffer},
        {12u,restartedAdaptiveBuffer},{13u,restartContactArguments},
        {14u,restartedRestartCycleBuffer},{15u,restartedBeginForcingBuffer},
        {16u,restartedIterationBuffer},{17u,restartedCandidateBuffer},
        {18u,restartedRigidCandidateBuffer},{19u,restartedSupportHistoryBuffer},
        {20u,restartedSupportBuffer},{21u,restartedSupportContactsBuffer},
        {22u,cycleNodeActive},{23u,cycleElementActive},
        {24u,restartedBeginPartialsBuffer},{25u,restartedDomainPartialsBuffer},
        {26u,restartedRHSBuffer}};
    auto restartBeginFinishBindings=restartBeginBindings;
    restartBeginFinishBindings.erase(std::remove_if(
        restartBeginFinishBindings.begin(),restartBeginFinishBindings.end(),
        [](const auto& binding){return binding.first==26u;}),
        restartBeginFinishBindings.end());
    auto restartTrueBindings=std::vector<std::pair<NSUInteger,id<MTLBuffer>>>{
        {0u,hybridDispatchBuffer},{30u,restartedLayoutBuffer},
        {1u,restartedSolverBuffer},{2u,restartedRHSBuffer},
        {3u,restartedOperatorBuffer},{4u,restartedResidualBuffer},
        {5u,cycleStates},{6u,restartedMetricsBuffer},
        {7u,restartedBeginForcingBuffer},{8u,cycleStatuses},
        {9u,restartedReplaceBuffer},{10u,cycleStatuses},
        {11u,restartedPropagateBuffer},{24u,restartedTruePartialsBuffer}};
    std::memcpy([cycleStates contents],&hybridStateInitial,sizeof(hybridStateInitial));
    NMMatterStatusGPU restartHealthy{}; restartHealthy.code=NM_STATUS_SUCCESS;
    std::memcpy([cycleStatuses contents],&restartHealthy,sizeof(restartHealthy));
    std::uint32_t restartedCycles=0u,restartedTotalColumns=0u;
    float restartedRelativeResidual=std::numeric_limits<float>::infinity();
    float restartedRelativeError=std::numeric_limits<float>::infinity();
    bool restartedCertified=false;
    for(std::uint32_t cycle=0u;cycle<restartedMaximumCycles;++cycle){
        const std::uint32_t restartValue=cycle==0u?0u:1u;
        std::memcpy([restartedRestartCycleBuffer contents],&restartValue,sizeof(restartValue));
        std::fill(static_cast<char*>([restartedBasisBuffer contents]),
            static_cast<char*>([restartedBasisBuffer contents])+restartedBasisBuffer.length,0);
        std::fill(static_cast<char*>([restartedPreconditionedBasisBuffer contents]),
            static_cast<char*>([restartedPreconditionedBasisBuffer contents])+
                restartedPreconditionedBasisBuffer.length,0);
        dispatch(device,queue,beginRestartPartial,restartBeginBindings,1u,256u);
        dispatch(device,queue,beginRestartFinish,restartBeginFinishBindings,1u,32u);
        dispatch(device,queue,beginRestartBasis0,restartBeginBindings,1u,256u);
        const auto begun=readBuffer<NMFGMRESStateGPU>(cycleStates,1u)[0];
        require(begun.diagnostics.z<=0.5f,
            "hybrid true residual latched before restarted basis initialization");
        std::uint32_t usedColumns=0u;
        for(std::uint32_t column=0u;column<restartedLimit;++column){
            const auto basisValues=readBuffer<nm_float4>(restartedBasisBuffer,
                std::size_t(NM_MIXED_FGMRES_RESTART+1u)*cycleUnknowns);
            std::vector<nm_float4> current(cycleUnknowns);
            std::copy(basisValues.begin()+std::size_t(column)*cycleUnknowns,
                basisValues.begin()+std::size_t(column+1u)*cycleUnknowns,current.begin());
            const auto preconditioned=preconditionHybrid(current);
            std::memcpy(static_cast<char*>([restartedPreconditionedBasisBuffer contents])+
                    std::size_t(column)*cycleUnknowns*sizeof(nm_float4),
                preconditioned.data(),cycleUnknowns*sizeof(nm_float4));
            const auto image=applyHybridOperator(preconditioned);
            std::memcpy([cycleWorkBuffer contents],image.data(),cycleUnknowns*sizeof(nm_float4));
            std::memcpy([cycleColumnBuffer contents],&column,sizeof(column));
            for(std::uint32_t pass=0u;pass<2u;++pass){
                std::memcpy([cyclePassBuffer contents],&pass,sizeof(pass));
                dispatch(device,queue,hybridCGSPartial,{
                    {0u,hybridDispatchBuffer},{1u,cycleColumnBuffer},{2u,cyclePassBuffer},
                    {3u,restartedBasisBuffer},{4u,cycleWorkBuffer},{5u,cyclePartialsBuffer},
                    {6u,cycleStates},{30u,restartedLayoutBuffer}},1u,256u);
                dispatch(device,queue,hybridCGSRow,{
                    {0u,hybridDispatchBuffer},{30u,restartedLayoutBuffer},
                    {1u,cycleColumnBuffer},{2u,cyclePassBuffer},
                    {3u,cycleHessenbergBuffer},{4u,cyclePartialsBuffer},{5u,cycleStates}},1u,32u);
                dispatch(device,queue,hybridCGSUpdate,{
                    {0u,hybridDispatchBuffer},{30u,restartedLayoutBuffer},
                    {1u,cycleColumnBuffer},{2u,cyclePassBuffer},
                    {3u,restartedBasisBuffer},{4u,cycleWorkBuffer},
                    {5u,cycleHessenbergBuffer},{6u,cyclePartialsBuffer},{7u,cycleStates}},1u,256u);
            }
            dispatch(device,queue,hybridMeasureNorm,{
                {0u,hybridDispatchBuffer},{30u,restartedLayoutBuffer},
                {1u,cycleWorkBuffer},{2u,cyclePartialsBuffer},{3u,cycleStates}},1u,256u);
            dispatch(device,queue,hybridFinishColumn,{
                {0u,hybridDispatchBuffer},{30u,restartedLayoutBuffer},
                {1u,restartedSolverBuffer},{2u,cycleColumnBuffer},
                {3u,cycleHessenbergBuffer},{4u,cycleRotationsBuffer},
                {5u,cycleLeastSquaresBuffer},{6u,cycleStates},
                {7u,restartedBeginForcingBuffer},{8u,cyclePartialsBuffer},
                {9u,cycleNodeActive},{10u,cycleElementActive}},1u,32u);
            dispatch(device,queue,hybridNormalize,{
                {0u,hybridDispatchBuffer},{30u,restartedLayoutBuffer},
                {1u,cycleColumnBuffer},{2u,cycleWorkBuffer},
                {3u,cycleNextBasisBuffer},{4u,cyclePartialsBuffer},{5u,cycleStates}},1u,256u);
            const auto next=readBuffer<nm_float4>(cycleNextBasisBuffer,cycleUnknowns);
            std::memcpy(static_cast<char*>([restartedBasisBuffer contents])+
                    std::size_t(column+1u)*cycleUnknowns*sizeof(nm_float4),
                next.data(),cycleUnknowns*sizeof(nm_float4));
            ++usedColumns;
            const auto state=readBuffer<NMFGMRESStateGPU>(cycleStates,1u)[0];
            if(state.diagnostics.z>0.5f||state.diagnostics.w!=float(column+1u))break;
        }
        require(usedColumns>0u,"hybrid restarted FGMRES emitted an empty Arnoldi cycle");
        restartedTotalColumns+=usedColumns;
        const std::uint32_t offset=cycle*restartedLimit;
        const std::uint32_t last=cycle+1u==restartedMaximumCycles?1u:0u;
        const std::uint32_t defer=1u;
        auto offsetBuffer=makeBuffer(device,std::vector<std::uint32_t>{offset});
        auto lastBuffer=makeBuffer(device,std::vector<std::uint32_t>{last});
        auto deferBuffer=makeBuffer(device,std::vector<std::uint32_t>{defer});
        auto coefficients=makeBuffer(device,std::vector<float>(NM_MIXED_FGMRES_RESTART+1u,0.0f));
        dispatch(device,queue,finalizeCycle,{
            {0u,hybridDispatchBuffer},{30u,restartedLayoutBuffer},
            {1u,restartedSolverBuffer},{2u,cycleHessenbergBuffer},
            {3u,cycleLeastSquaresBuffer},{4u,cycleStates},{5u,cycleStatuses},
            {6u,offsetBuffer},{7u,lastBuffer},{8u,restartedBeginForcingBuffer},
            {9u,cycleRotationsBuffer},{10u,coefficients},{11u,deferBuffer}},1u,32u);
        const std::uint32_t add=cycle==0u?0u:1u;
        std::memcpy([cycleRestartBuffer contents],&add,sizeof(add));
        dispatch(device,queue,cycleAccumulate,{
            {0u,hybridDispatchBuffer},{30u,restartedLayoutBuffer},
            {1u,restartedSolverBuffer},{2u,nodeRangesBuffer},
            {3u,restartedPreconditionedBasisBuffer},{4u,cycleLeastSquaresBuffer},
            {5u,cycleStates},{6u,restartedSolutionBuffer},{7u,cycleRestartBuffer}},
            nodesPerEnvironment,256u);
        dispatch(device,queue,cycleAccumulateFields,{
            {0u,hybridDispatchBuffer},{30u,restartedLayoutBuffer},
            {1u,restartedSolverBuffer},{2u,nodeRangesBuffer},
            {3u,restartedPreconditionedBasisBuffer},{4u,cycleLeastSquaresBuffer},
            {5u,cycleStates},{6u,restartedSolutionBuffer},{7u,cycleRestartBuffer}},
            nodesPerEnvironment,256u);
        const auto solutionVector=readBuffer<nm_float4>(restartedSolutionBuffer,cycleUnknowns);
        const auto action=applyHybridOperator(solutionVector);
        std::memcpy([restartedOperatorBuffer contents],action.data(),cycleUnknowns*sizeof(nm_float4));
        dispatch(device,queue,trueRestartPartial,restartTrueBindings,1u,256u);
        dispatch(device,queue,trueRestartFinish,restartTrueBindings,1u,32u);
        dispatch(device,queue,trueRestartReplace,restartTrueBindings,
            hybridModes+nodesPerEnvironment,256u);
        const auto state=readBuffer<NMFGMRESStateGPU>(cycleStates,1u)[0];
        const auto metrics=readBuffer<nm_float4>(restartedMetricsBuffer,1u)[0];
        restartedCertified=state.diagnostics.z>0.5f;
        require(readBuffer<NMMatterStatusGPU>(cycleStatuses,1u)[0].code==NM_STATUS_SUCCESS&&
                std::isfinite(metrics.x),
            "hybrid production true-residual restart failed status/finite checks");
        restartedRelativeResidual=metrics.x/rhsNorm;
        const auto coordinates=fromHybridVector(solutionVector);
        double error2=0.0;
        for(std::uint32_t i=0u;i<hybridDofs;++i){
            const double e=coordinates[i]-hybridOracle[i];error2+=e*e;}
        restartedRelativeError=static_cast<float>(std::sqrt(error2)/
            std::max(1.0e-30,std::sqrt(cycleOracleNorm2)));
        ++restartedCycles;
        if(state.diagnostics.z>0.5f)break;
    }
    std::cout<<"hybrid_restart_diagnostic cycles="<<restartedCycles
    <<" total_columns="<<restartedTotalColumns
    <<" certified="<<(restartedCertified?1:0)
    <<" relative_residual="<<restartedRelativeResidual
    <<" relative_error="<<restartedRelativeError
    <<" initial_rhs_norm="<<rhsNorm<<" forcing=1e-6\n";
    require(restartedCycles>=2u&&restartedTotalColumns>=2u&&
            restartedCertified&&restartedRelativeResidual<1.0e-6f&&
            restartedRelativeError<1.0e-5f,
        "production hybrid restarted FGMRES failed the dense q+fine oracle");
    std::cout<<"hybrid_q_fine_production_restarts=true restart=1 cycles="
        <<restartedCycles<<" total_columns="<<restartedTotalColumns
        <<" relative_residual="<<restartedRelativeResidual
        <<" relative_solution_error="<<restartedRelativeError
        <<" forcing=1e-6 certified=true roundoff_control_forcing=1e-8 "
        <<"distributed_begin=true full_true_residual=true result=PASS\n";

    std::cout << "hybrid_q_fine_dense_oracle=true dofs=" << hybridDofs
              << " q_to_fine=" << qToFine << " fine_to_q=" << fineToQ
              << " oracle_error=" << hybridOracleError
              << " production_CGS2=true next_norm="
              << std::sqrt(hybridNextNorm2) << " q_fine_dot="
              << hybridOrthogonalDot << " result=PASS" << std::endl;

    // Verify the complete production path, not just reuse of injected blocks:
    // build the nonlinear tangent cache, generate exact cached tet blocks,
    // and assemble one graph patch matrix. Compare its cached-block mode with
    // direct material evaluation and cached tangent modes 1/2.
    {
        constexpr std::uint32_t graphDofs = nodesPerEnvironment * 3u;
        constexpr std::uint32_t graphMatrixFloats = graphDofs * graphDofs;
        const auto graphBuild = pipelineWithRelativeInnerForcingDisabled(
            device, library,
            "numi_matter_metal::nm_fgmres_graph_build_patch_matrices");
        const auto graphPatches = makeBuffer(device,
            std::vector<nm_uint4>{nm_uint4{0u, nodesPerEnvironment, 0u, 0u}});
        std::vector<std::uint32_t> graphPatchNodes(nodesPerEnvironment);
        std::iota(graphPatchNodes.begin(), graphPatchNodes.end(), 0u);
        const auto graphPatchNodesBuffer = makeBuffer(device, graphPatchNodes);
        std::vector<nm_uint4> graphRows;
        for (std::uint32_t node = 0u; node < nodesPerEnvironment; ++node)
            for (std::uint32_t axis = 0u; axis < 3u; ++axis)
                graphRows.push_back(nm_uint4{0u, node, axis, node});
        const auto graphRowsBuffer = makeBuffer(device, graphRows);
        const auto graphCoverage = makeBuffer(device,
            std::vector<std::uint32_t>(nodesPerEnvironment, 1u));
        const auto graphReuse = makeBuffer(device, std::vector<std::uint32_t>{0u});
        const auto graphPriorValid = makeBuffer(device,
            std::vector<std::uint32_t>(environments, 0u));
        const auto graphRowCount = makeBuffer(device,
            std::vector<std::uint32_t>{graphDofs});
        const auto graphPatchCount = makeBuffer(device,
            std::vector<std::uint32_t>{1u});
        const auto graphMatrixCount = makeBuffer(device,
            std::vector<std::uint32_t>{graphMatrixFloats});
        const auto graphDue = makeBuffer(device, std::vector<std::uint32_t>{1u});
        NMMicrostepGPU graphMicro = micro;
        graphMicro.flags = NM_MICROSTEP_FGMRES_OPERATOR;
        graphMicro.solverIteration = 0u;
        graphMicro.microtick = 1u;
        const auto graphMicroBuffer = makeBuffer(device,
            std::vector<NMMicrostepGPU>{graphMicro});
        auto buildGraph = [&](std::uint32_t mode) {
            auto modeBuffer = makeBuffer(device, std::vector<std::uint32_t>{mode});
            auto output = makeBuffer(device,
                std::vector<float>(environments * graphMatrixFloats, 0.0f));
            std::vector<NMMatterStatusGPU> health(environments);
            for (std::uint32_t e = 0u; e < environments; ++e) {
                health[e].code = NM_STATUS_SUCCESS;
                health[e].environment = e;
            }
            auto healthBuffer = makeBuffer(device, health);
            auto states = makeBuffer(device, solverStates);
            dispatch(device, queue, graphBuild, {
                {0u, dispatchBuffer}, {1u, objectsBuffer},
                {2u, candidateBuffer}, {3u, tetrahedraBuffer},
                {4u, incidenceBuffer}, {5u, nodeRangesBuffer},
                {6u, schedulerBuffer}, {7u, materialsBuffer},
                {8u, programsBuffer}, {9u, instructionsBuffer},
                {10u, parametersBuffer}, {11u, acceptedStateBuffer},
                {12u, tangentCacheBuffer}, {13u, cacheValidBuffer},
                {14u, modeBuffer}, {15u, graphPatches},
                {16u, graphPatchNodesBuffer}, {17u, graphRowsBuffer},
                {18u, graphCoverage}, {19u, output}, {20u, healthBuffer},
                {21u, graphRowCount}, {22u, graphPatchCount},
                {23u, graphMatrixCount}, {24u, states},
                {25u, graphReuse}, {26u, graphPriorValid},
                {27u, adaptiveBuffer}, {28u, graphMicroBuffer},
                {29u, graphDue}, {30u, elementBlocksBuffer}},
                environments * graphDofs, 256u);
            const auto graphStatuses = readBuffer<NMMatterStatusGPU>(
                healthBuffer, environments);
            for (std::size_t e = 0; e < graphStatuses.size(); ++e) {
                const auto& value = graphStatuses[e];
                if (value.code != NM_STATUS_SUCCESS) {
                    std::cerr << "generated_graph_matrix_status environment=" << e
                              << " code=" << value.code
                              << " object=" << value.objectIndex
                              << " element=" << value.failingIndex
                              << " diagnostics=[" << value.diagnostics.x << ","
                              << value.diagnostics.y << ","
                              << value.diagnostics.z << ","
                              << value.diagnostics.w << "]" << std::endl;
                }
                require(value.code == NM_STATUS_SUCCESS,
                    "generated tet-block graph matrix build failed status validation");
            }
            return readBuffer<float>(output,
                environments * graphMatrixFloats);
        };
        for (std::uint32_t sample = 0u; sample < 3u; ++sample) {
            std::vector<NMFEMNodeStateGPU> varied = candidate;
            for (std::uint32_t e = 0u; e < environments; ++e)
                for (std::uint32_t node = 0u; node < nodesPerEnvironment; ++node) {
                    auto& value = varied[e * nodesPerEnvironment + node];
                    const float g = float(sample + 1u);
                    const float en = float(e + 1u);
                    const float n = float(node + 1u);
                    value.positionAndMass.x = positions[node].x +
                        0.015f * float(e) * float(node + 1u) +
                        0.006f * std::sin(0.37f * g * n + en);
                    value.positionAndMass.y = positions[node].y -
                        0.009f * float(e) * float(node + 2u) +
                        0.005f * std::cos(0.29f * g * (n + en));
                    value.positionAndMass.z = positions[node].z +
                        0.004f * std::sin(0.41f * (g + n) * en);
                    value.restAndFixed.w = node == 0u ? 1.0f : 0.0f;
                    if (node == 0u) value.velocityAndInverseMass.w = 0.0f;
                }
            std::memcpy([candidateBuffer contents], varied.data(),
                varied.size() * sizeof(NMFEMNodeStateGPU));
            std::fill(cacheValid.begin(), cacheValid.end(), 0u);
            std::memcpy([cacheValidBuffer contents], cacheValid.data(),
                cacheValid.size() * sizeof(std::uint32_t));
            std::vector<NMMatterStatusGPU> healthy(environments);
            for (std::uint32_t e = 0u; e < environments; ++e) {
                healthy[e].code = NM_STATUS_SUCCESS;
                healthy[e].environment = e;
            }
            std::memcpy([statusBuffer contents], healthy.data(),
                healthy.size() * sizeof(NMMatterStatusGPU));
            dispatch(device, queue, tangentBuild, {
                {0u, dispatchBuffer}, {1u, graphMicroBuffer},
                {2u, objectsBuffer}, {3u, materialsBuffer},
                {4u, programsBuffer}, {5u, instructionsBuffer},
                {6u, parametersBuffer}, {7u, candidateBuffer},
                {8u, tetrahedraBuffer}, {9u, acceptedStateBuffer},
                {10u, schedulerBuffer}, {11u, adaptiveBuffer},
                {12u, tangentCacheBuffer}, {13u, cacheValidBuffer},
                {14u, statusBuffer}, {15u, solverStateBuffer},
                {16u, forceRebuildBuffer}},
                environments * tetrahedraPerEnvironment, 1u);
            const auto valid = readBuffer<std::uint32_t>(
                cacheValidBuffer, cacheValid.size());
            require(std::all_of(valid.begin(), valid.end(),
                                [](std::uint32_t v) { return v == 1u; }),
                "random graph geometry did not produce current tangent caches");
            const std::uint32_t mode2 = 2u;
            std::memcpy([cacheModeBuffer contents], &mode2, sizeof(mode2));
            dispatch(device, queue, assembleTet, {
                {0u, dispatchBuffer}, {1u, graphMicroBuffer},
                {2u, objectsBuffer}, {3u, tetrahedraBuffer},
                {4u, tangentCacheBuffer}, {5u, cacheValidBuffer},
                {6u, elementBlocksBuffer}, {7u, statusBuffer},
                {8u, schedulerBuffer}, {9u, adaptiveBuffer},
                {10u, solverStateBuffer}, {11u, cacheModeBuffer},
                {12u, forceRebuildBuffer},
        {13u, includeMixedGraphXYZBuffer}}, 1u, 256u);
            for (const auto& value : readBuffer<NMMatterStatusGPU>(
                     statusBuffer, environments))
                require(value.code == NM_STATUS_SUCCESS,
                    "production cached tet-block assembly failed status validation");
            const auto generated = buildGraph(3u);
            for (const std::uint32_t mode : {0u, 1u, 2u}) {
                const auto direct = buildGraph(mode);
                double scale = 1.0;
                double maxError = 0.0;
                std::size_t worst = 0u;
                for (std::size_t i = 0u; i < generated.size(); ++i) {
                    scale = std::max(scale, std::abs(double(generated[i])));
                    scale = std::max(scale, std::abs(double(direct[i])));
                    const double error = std::abs(
                        double(generated[i]) - double(direct[i]));
                    if (error > maxError) { maxError = error; worst = i; }
                }
                const double tolerance = 32.0 *
                    double(std::numeric_limits<float>::epsilon()) * scale + 1.0e-7;
                if (maxError > tolerance)
                    std::cerr << "generated_graph_matrix_parity sample="
                              << sample << " mode=" << mode
                              << " max_error=" << maxError
                              << " bound=" << tolerance
                              << " worst_index=" << worst << std::endl;
                require(maxError <= tolerance,
                    "graph matrix from generated tet blocks differs from direct/cached mode " +
                        std::to_string(mode));
            }

            // Mixed FEM's graph patch retains the exact constitutive xyz block
            // as a preconditioner contribution. The pressure/field blocks and
            // their cross-couplings remain owned by the mixed operator path.
            auto mixedObject = object;
            mixedObject.flags |= NM_OBJECT_MIXED_FEM;
            std::memcpy([objectsBuffer contents], &mixedObject,
                sizeof(mixedObject));
            std::fill(healthy.begin(), healthy.end(), NMMatterStatusGPU{});
            for (std::uint32_t e = 0u; e < environments; ++e) {
                healthy[e].code = NM_STATUS_SUCCESS;
                healthy[e].environment = e;
            }
            std::memcpy([statusBuffer contents], healthy.data(),
                healthy.size() * sizeof(NMMatterStatusGPU));
            std::fill(cacheValid.begin(), cacheValid.end(), 0u);
            std::memcpy([cacheValidBuffer contents], cacheValid.data(),
                cacheValid.size() * sizeof(std::uint32_t));
            dispatch(device, queue, tangentPrepareColumns, {
                {0u, dispatchBuffer}, {1u, graphMicroBuffer},
                {13u, cacheValidBuffer}, {15u, solverStateBuffer},
                {16u, forceRebuildBuffer}},
                environments * tetrahedraPerEnvironment, 256u);
            dispatch(device, queue, tangentBuildColumns, {
                {0u, dispatchBuffer}, {1u, graphMicroBuffer},
                {2u, objectsBuffer}, {3u, materialsBuffer},
                {4u, programsBuffer}, {5u, instructionsBuffer},
                {6u, parametersBuffer}, {7u, candidateBuffer},
                {8u, tetrahedraBuffer}, {9u, acceptedStateBuffer},
                {10u, schedulerBuffer}, {11u, adaptiveBuffer},
                {12u, tangentCacheBuffer}, {13u, cacheValidBuffer},
                {14u, statusBuffer}, {15u, solverStateBuffer},
                {16u, forceRebuildBuffer},
                {17u, fullSourceSelectionCountBuffer},
                {18u, fullSourceSelectionBuffer},
                {19u, selectedTetrahedronBaseBuffer},
                {20u, graphOnlyMixedCacheBuffer}},
                environments * tetrahedraPerEnvironment * 9u, 256u);
            dispatch(device, queue, tangentFinalizeColumns, {
                {0u, dispatchBuffer}, {1u, graphMicroBuffer},
                {13u, cacheValidBuffer}, {14u, statusBuffer},
                {15u, solverStateBuffer}, {16u, forceRebuildBuffer}},
                environments * tetrahedraPerEnvironment, 256u);
            const auto mixedCacheValid = readBuffer<std::uint32_t>(
                cacheValidBuffer, cacheValid.size());
            require(std::all_of(mixedCacheValid.begin(), mixedCacheValid.end(),
                                [](std::uint32_t v) { return v == 2u; }),
                "mixed graph-only cache was not published with marker 2");

            const std::uint32_t omitMixedXYZ = 0u;
            std::memcpy([includeMixedGraphXYZBuffer contents], &omitMixedXYZ,
                sizeof(omitMixedXYZ));
            dispatch(device, queue, assembleTet, {
                {0u, dispatchBuffer}, {1u, graphMicroBuffer},
                {2u, objectsBuffer}, {3u, tetrahedraBuffer},
                {4u, tangentCacheBuffer}, {5u, cacheValidBuffer},
                {6u, elementBlocksBuffer}, {7u, statusBuffer},
                {8u, schedulerBuffer}, {9u, adaptiveBuffer},
                {10u, solverStateBuffer}, {11u, cacheModeBuffer},
                {12u, forceRebuildBuffer},
                {13u, includeMixedGraphXYZBuffer}}, 1u, 256u);
            const auto fullOperatorBlocks = readBuffer<float>(
                elementBlocksBuffer, elementBlockFloatCount);
            require(std::all_of(fullOperatorBlocks.begin(),
                                fullOperatorBlocks.end(),
                                [](float value) { return value == 0.0f; }),
                "full assembled-operator eligibility accidentally accepted mixed FEM xyz blocks");

            const std::uint32_t graphOnlyMixedXYZ = 1u;
            std::memcpy([includeMixedGraphXYZBuffer contents],
                &graphOnlyMixedXYZ, sizeof(graphOnlyMixedXYZ));
            dispatch(device, queue, assembleTet, {
                {0u, dispatchBuffer}, {1u, graphMicroBuffer},
                {2u, objectsBuffer}, {3u, tetrahedraBuffer},
                {4u, tangentCacheBuffer}, {5u, cacheValidBuffer},
                {6u, elementBlocksBuffer}, {7u, statusBuffer},
                {8u, schedulerBuffer}, {9u, adaptiveBuffer},
                {10u, solverStateBuffer}, {11u, cacheModeBuffer},
                {12u, forceRebuildBuffer},
                {13u, includeMixedGraphXYZBuffer}}, 1u, 256u);
            const auto mixedElementBlocks = readBuffer<float>(
                elementBlocksBuffer, elementBlockFloatCount);
            require(std::any_of(mixedElementBlocks.begin(),
                                mixedElementBlocks.end(),
                                [](float value) { return value != 0.0f; }),
                "graph-only mixed FEM xyz block assembly produced no contribution");
            const auto mixedGenerated = buildGraph(3u);
            const auto mixedDirect = buildGraph(0u);
            double mixedScale = 1.0;
            double mixedMaxError = 0.0;
            for (std::size_t i = 0u; i < mixedGenerated.size(); ++i) {
                mixedScale = std::max(mixedScale,
                    std::abs(double(mixedGenerated[i])));
                mixedScale = std::max(mixedScale,
                    std::abs(double(mixedDirect[i])));
                mixedMaxError = std::max(mixedMaxError,
                    std::abs(double(mixedGenerated[i]) -
                             double(mixedDirect[i])));
            }
            const double mixedTolerance = 32.0 *
                double(std::numeric_limits<float>::epsilon()) * mixedScale + 1.0e-7;
            require(mixedMaxError <= mixedTolerance,
                "mixed graph-only cached xyz matrix differs from direct constitutive preconditioner matrix");
            std::memcpy([objectsBuffer contents], &object, sizeof(object));
            std::memcpy([includeMixedGraphXYZBuffer contents], &omitMixedXYZ,
                sizeof(omitMixedXYZ));
        }
        std::cout << "graph_generated_tet_block_parity=true modes=0,1,2,3 "
                     "geometry_samples=3 two_env_dt=true rotated_tets=true "
                     "result=PASS\n";
    }

    std::cout << "assembled_xyz_operator=true edge_assembly=true "
                 "simd_row_apply=true current_bytecode_tangent=true "
                 "two_env_dt=true fixed_rows=true shared_edge_csr=true "
                 "cache_lifecycle=true "
                 "translation_null=true cancellation_bound=true "
                 "force_closure=true nonfinite_witness=true "
                 "invalid_cache_fallback=true "
                 "empty_domain=true result=PASS"
              << std::endl;
}



void checkHybridBlockGaussSeidel(
    id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library) {
    constexpr std::uint32_t nodes = 4u;
    NMMatterDispatchGPU dispatchData{};
    dispatchData.abiVersion = NM_MATTER_ABI_VERSION;
    dispatchData.environmentCount = 1u;
    dispatchData.objectCount = 1u;
    dispatchData.femNodeCount = nodes;
    dispatchData.tetrahedronCount = 1u;
    auto dispatchBuffer = makeBuffer(device, std::vector{dispatchData});

    NMPartitionedReductionFEMTangentDispatchGPU tangent{};
    tangent.globalNodeCount = nodes;
    tangent.basisNodeCount = nodes;
    tangent.modeCount = 1u;
    tangent.tetrahedronCount = 1u;
    tangent.environmentCount = 1u;
    tangent.regionCount = 1u;
    tangent.blockModeCount = 1u;
    tangent.supportTetrahedronCount = 1u;
    auto tangentBuffer = makeBuffer(device, std::vector{tangent});
    auto regionRangesBuffer = makeBuffer(device,
        std::vector<NMPartitionedReductionSparseRangeGPU>{{0u, 1u}});
    auto regionIndicesBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto basisRowsBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u, 1u, 2u, 3u});
    auto blockModesBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto nodeModeRangesBuffer = makeBuffer(device,
        std::vector<NMPartitionedReductionSparseRangeGPU>{
            {0u, 1u}, {0u, 0u}, {0u, 0u}, {0u, 0u}});
    auto globalNodesBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u, 1u, 2u, 3u});

    const auto qRhsKernel = pipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_hybrid_gs_correct_q_rhs");
    const auto fineKernel = pipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_hybrid_gs_add_fine_correction");

    // A_fq is the three-by-one block C. The selected q-row at corner zero
    // carries a large sentinel action and must be excluded by nodeModeRanges.
    NMPartitionedReductionFEMTangentActionGPU action{};
    action.full.node0 = nm_float4{80.0f, -40.0f, 20.0f, 0.0f};
    action.full.node1 = nm_float4{0.1f, 0.2f, -0.1f, 0.0f};
    action.full.node2 = nm_float4{-0.15f, 0.05f, 0.2f, 0.0f};
    action.full.node3 = nm_float4{0.2f, -0.1f, 0.05f, 0.0f};
    auto actionBuffer = makeBuffer(device,
        std::vector<NMPartitionedReductionFEMTangentActionGPU>{action});

    const std::array<nm_float4, nodes> fineInput{{
        nm_float4{5.0f, 6.0f, 7.0f, 0.0f},
        nm_float4{0.5f, -1.0f, 2.0f, 0.0f},
        nm_float4{-0.25f, 0.75f, 1.25f, 0.0f},
        nm_float4{1.5f, 0.5f, -0.5f, 0.0f}}};
    const float initialQRhs = 1.25f;
    auto runCase = [&](const bool nonsymmetric) {
        auto fineBuffer = makeBuffer(device,
            std::vector<nm_float4>(fineInput.begin(), fineInput.end()));
        auto rhsBuffer = makeBuffer(device, std::vector<float>{initialQRhs});
        dispatch(device, queue, qRhsKernel, {
            {0u, dispatchBuffer}, {1u, regionRangesBuffer},
            {2u, regionIndicesBuffer}, {3u, basisRowsBuffer},
            {4u, actionBuffer}, {5u, blockModesBuffer},
            {6u, nodeModeRangesBuffer}, {7u, globalNodesBuffer},
            {8u, fineBuffer}, {9u, rhsBuffer}, {30u, tangentBuffer}}, 1u, 32u);
        const float projectedQRhs = readBuffer<float>(rhsBuffer, 1u)[0];
        // C^T z_f = -0.35 + 0.325 + 0.225 = 0.20. The selected-corner
        // sentinel would alter this if the support mask were ignored.
        require(close(projectedQRhs, 1.05f, 2.0e-6f),
            "hybrid GS q-RHS kernel did not apply the fine-to-q block exactly");
        const float qCoefficient = projectedQRhs / 2.0f;
        auto coefficientsBuffer = makeBuffer(device,
            std::vector<float>{qCoefficient});
        auto termRangesBuffer = makeBuffer(device,
            std::vector<NMPartitionedReductionSparseRangeGPU>{
                {0u, 0u}, {0u, 1u}, {1u, 1u}, {2u, 1u}});
        auto termsBuffer = makeBuffer(device,
            std::vector<NMPartitionedReductionHybridGSTermGPU>{
                {0u, 0u, 1u, 0u}, {0u, 0u, 2u, 0u},
                {0u, 0u, 3u, 0u}});
        std::vector<nm_float4> inverseRows(nodes * 3u, nm_float4{});
        for (std::uint32_t node = 0u; node < nodes; ++node) {
            inverseRows[node * 3u] = nm_float4{1.0f, 0.0f, 0.0f, 1.0f};
            inverseRows[node * 3u + 1u] = nm_float4{0.0f, 1.0f, 0.0f, 0.0f};
            inverseRows[node * 3u + 2u] = nm_float4{0.0f, 0.0f, 1.0f, 0.0f};
        }
        auto inverseBuffer = makeBuffer(device, inverseRows);
        auto solveValidBuffer = makeBuffer(device, std::vector<std::uint32_t>{1u});
        dispatch(device, queue, fineKernel, {
            {1u, termRangesBuffer}, {2u, termsBuffer}, {3u, inverseBuffer},
            {4u, actionBuffer}, {5u, coefficientsBuffer},
            {6u, solveValidBuffer}, {7u, fineBuffer}, {30u, tangentBuffer}},
            1u, 32u);
        const auto actual = readBuffer<nm_float4>(fineBuffer, nodes);
        require(close(actual[0].x, fineInput[0].x) &&
                close(actual[0].y, fineInput[0].y) &&
                close(actual[0].z, fineInput[0].z),
            "hybrid GS fine correction modified a selected q row");
        const float fineZ[3][3]{{0.5f, -1.0f, 2.0f},
            {-0.25f, 0.75f, 1.25f}, {1.5f, 0.5f, -0.5f}};
        const float C[3][3]{{0.1f, 0.2f, -0.1f},
            {-0.15f, 0.05f, 0.2f}, {0.2f, -0.1f, 0.05f}};
        for (std::uint32_t row = 0u; row < 3u; ++row) {
            const float target[3]{
                fineZ[row][0] - C[row][0] * qCoefficient,
                fineZ[row][1] - C[row][1] * qCoefficient,
                fineZ[row][2] - C[row][2] * qCoefficient};
            require(close(actual[row + 1u].x, target[0], 2.0e-6f) &&
                    close(actual[row + 1u].y, target[1], 2.0e-6f) &&
                    close(actual[row + 1u].z, target[2], 2.0e-6f),
                "hybrid GS fine correction disagrees with dense block oracle");
        }
        if (nonsymmetric) {
            const float G[3][3]{{0.4f, 0.1f, 0.0f},
                {0.0f, -0.3f, 0.2f}, {-0.2f, 0.5f, 0.4f}};
            const float z[3][3]{{0.5f, -1.0f, 2.0f},
                {-0.25f, 0.75f, 1.25f}, {1.5f, 0.5f, -0.5f}};
            double exactCoupling = 0.0;
            for (std::uint32_t row = 0u; row < 3u; ++row)
                for (std::uint32_t axis = 0u; axis < 3u; ++axis)
                    exactCoupling += double(G[row][axis]) * z[row][axis];
            const float exactNonsymmetricQ =
                static_cast<float>((initialQRhs - exactCoupling) / 2.0);
            require(std::abs(exactNonsymmetricQ - qCoefficient) > 0.1f,
                "nonsymmetric GS fixture does not distinguish exact A_qf from A_fq transpose");
        } else {
            require(close(qCoefficient, 0.525f, 2.0e-6f),
                "symmetric hybrid GS q coefficient is not the dense exact result");
        }
    };
    runCase(false);
    runCase(true);
    std::cout << "hybrid_block_GS=true SPD_exact=true "
                 "nonsymmetric_transpose_approximation=true "
                 "selected_rows_preserved=true result=PASS\n";
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


void checkParallelGraphPatchTriangularSolve(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLLibrary> library) {
    // Exercise the maximum admitted patch width (64 nodes / 192 scalar DOFs)
    // so each pivot update spans most of the 256-lane group and crosses the
    // lane-stride boundary. The nonsymmetric first column forces a row pivot.
    constexpr std::uint32_t nodeCount = 64u;
    constexpr std::uint32_t n = nodeCount * 3u;
    constexpr std::uint32_t patchCount = 1u;
    constexpr std::uint32_t patchNodeCount = nodeCount;
    constexpr std::uint32_t matrixFloatCount = n * n;
    constexpr std::uint32_t pivotCount = 3u * n;

    NMMatterDispatchGPU shape{};
    shape.abiVersion = NM_MATTER_ABI_VERSION;
    shape.environmentCount = 1u;
    shape.femNodeCount = nodeCount;
    const auto dispatchBuffer = makeBuffer(device,
        std::vector<NMMatterDispatchGPU>{shape});
    auto patches = makeBuffer(device,
        std::vector<nm_uint4>{nm_uint4{0u, nodeCount, 0u, 0u}});
    std::vector<std::uint32_t> patchNodeValues(nodeCount);
    for (std::uint32_t node = 0u; node < nodeCount; ++node)
        patchNodeValues[node] = node;
    auto patchNodes = makeBuffer(device, patchNodeValues);
    auto u32 = [&](std::uint32_t value) {
        return makeBuffer(device, std::vector<std::uint32_t>{value});
    };

    std::vector<float> original(matrixFloatCount, 0.0f);
    for (std::uint32_t row = 0u; row < n; ++row) {
        original[row * n + row] =
            3.0f + 0.001f * float(row % 37u);
        for (std::uint32_t column = 0u; column < n; ++column) {
            if (row == column) continue;
            const std::uint32_t distance = row > column
                ? row - column : column - row;
            float value = 0.0f;
            if (distance <= 3u) {
                const float sign = row < column ? 1.0f : -1.0f;
                value = sign * 0.003f / float(distance);
            } else if ((row * 17u + column * 13u) % 97u == 0u) {
                value = 0.0005f * (row < column ? 1.0f : -1.0f);
            }
            original[row * n + column] = value;
        }
    }
    // Force deterministic partial pivoting at column zero without making the
    // dense reference ill-conditioned.
    original[0] = 0.75f;
    original[1] = 2.0f;
    original[n] = 1.75f;
    auto matrix = makeBuffer(device, original);
    auto originalMatrix = makeBuffer(device,
        std::vector<float>(matrixFloatCount, 0.0f));
    auto pivots = makeBuffer(device, std::vector<std::uint32_t>(pivotCount));
    auto factorValid = makeBuffer(device, std::vector<std::uint32_t>(patchCount));
    auto states = makeBuffer(device,
        std::vector<NMFGMRESStateGPU>{NMFGMRESStateGPU{}});
    auto factorStatus = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
    const auto factor = pipelineWithRelativeInnerForcingDisabled(device,
        library, "numi_matter_metal::nm_fgmres_graph_factor_patches");
    dispatch(device, queue, factor, {
        {0u, dispatchBuffer}, {1u, patches}, {2u, matrix},
        {3u, pivots}, {4u, factorValid}, {5u, factorStatus},
        {6u, u32(patchCount)}, {7u, u32(matrixFloatCount)},
        {8u, u32(pivotCount)}, {9u, originalMatrix}, {10u, states},
        {11u, u32(0u)}, {12u, u32(0u)}, {13u, u32(1u)},
        {14u, u32(0u)}, {15u, u32(0u)}, {16u, u32(NM_INVALID_INDEX)}},
        patchCount, 256u);
    require(readBuffer<std::uint32_t>(factorValid, patchCount)[0] == 1u &&
            readBuffer<NMMatterStatusGPU>(factorStatus, 1u)[0].code ==
                NM_STATUS_SUCCESS,
        "wide graph patch factorization failed");
    require(readBuffer<std::uint32_t>(pivots, pivotCount)[0] == 1u,
        "wide graph fixture did not exercise a row pivot");

    std::vector<NMFEMNodeStateGPU> candidateValues(nodeCount);
    for (auto& candidate : candidateValues)
        candidate.restAndFixed.w = 0.0f;
    auto candidate = makeBuffer(device, candidateValues);
    auto regionalBasis = makeBuffer(device,
        std::vector<nm_float4>(nodeCount));
    auto regionalAP = makeBuffer(device,
        std::vector<nm_float4>(nodeCount));
    auto coefficients = makeBuffer(device, std::vector<float>{0.0f});
    const auto solve = pipelineWithRelativeInnerForcingDisabled(device,
        library, "numi_matter_metal::nm_fgmres_graph_solve_patches");

    const auto luValues = readBuffer<float>(matrix, matrixFloatCount);
    const auto pivotValues = readBuffer<std::uint32_t>(pivots, pivotCount);
    const auto floatFromBits = [](std::uint32_t bits) {
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    };
    auto runSolve = [&](std::uint32_t trial, std::uint32_t refinementMode) {
        std::vector<float> known(n);
        for (std::uint32_t scalar = 0u; scalar < n; ++scalar) {
            const float phase = float((scalar * 29u + trial * 11u) % 101u);
            known[scalar] = 0.2f * std::sin(phase * 0.071f) +
                0.1f * std::cos(phase * 0.037f);
        }
        std::vector<nm_float4> rhs(nodeCount);
        std::vector<float> rhsScalar(n, 0.0f);
        for (std::uint32_t row = 0u; row < n; ++row) {
            float value = 0.0f;
            for (std::uint32_t column = 0u; column < n; ++column)
                value += original[row * n + column] * known[column];
            rhsScalar[row] = value;
            const std::uint32_t node = row / 3u;
            const std::uint32_t axis = row % 3u;
            if (axis == 0u) rhs[node].x = value;
            else if (axis == 1u) rhs[node].y = value;
            else rhs[node].z = value;
        }
        auto rhsBuffer = makeBuffer(device, rhs);
        auto corrections = makeBuffer(device,
            std::vector<nm_float4>(nodeCount));
        auto solveStatus = makeBuffer(device,
            std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
        dispatch(device, queue, solve, {
            {0u, dispatchBuffer}, {1u, patches}, {2u, patchNodes},
            {3u, rhsBuffer}, {4u, regionalBasis}, {5u, regionalAP},
            {6u, coefficients}, {7u, matrix}, {8u, pivots},
            {9u, factorValid}, {10u, corrections}, {11u, solveStatus},
            {12u, u32(patchCount)}, {13u, u32(patchNodeCount)},
            {14u, u32(matrixFloatCount)}, {15u, u32(pivotCount)},
            {16u, u32(0u)}, {17u, states}, {18u, originalMatrix},
            {19u, u32(refinementMode)}, {20u, candidate}, {21u, u32(0u)},
            {22u, u32(0u)}, {23u, u32(0u)}, {24u, u32(0u)},
            {25u, u32(0u)}}, patchCount, 256u);
        require(readBuffer<NMMatterStatusGPU>(solveStatus, 1u)[0].code ==
                NM_STATUS_SUCCESS,
            "wide parallel graph patch solve reported failure");
        const auto solution = readBuffer<nm_float4>(corrections, nodeCount);
        std::vector<float> scalarSolution(n);
        for (std::uint32_t row = 0u; row < n; ++row) {
            const auto& node = solution[row / 3u];
            scalarSolution[row] = row % 3u == 0u ? node.x :
                (row % 3u == 1u ? node.y : node.z);
            require(std::isfinite(scalarSolution[row]),
                "wide parallel graph patch solve wrote a nonfinite result");
        }
        std::vector<float> leftLookingReference(n);
        for (std::uint32_t scalar = 0u; scalar < n; ++scalar) {
            const std::uint32_t originalRow = pivotValues[scalar];
            const float rowScale = floatFromBits(pivotValues[n + originalRow]);
            leftLookingReference[scalar] = rhsScalar[originalRow] * rowScale;
        }
        for (std::uint32_t row = 0u; row < n; ++row) {
            float value = leftLookingReference[row];
            for (std::uint32_t column = 0u; column < row; ++column)
                value -= luValues[row * n + column] *
                    leftLookingReference[column];
            leftLookingReference[row] = value;
        }
        for (std::int32_t reverse = std::int32_t(n) - 1;
             reverse >= 0; --reverse) {
            const std::uint32_t row = std::uint32_t(reverse);
            float value = leftLookingReference[row];
            for (std::uint32_t column = row + 1u; column < n; ++column)
                value -= luValues[row * n + column] *
                    leftLookingReference[column];
            leftLookingReference[row] = value / luValues[row * n + row];
        }
        for (std::uint32_t scalar = 0u; scalar < n; ++scalar)
            leftLookingReference[scalar] *=
                floatFromBits(pivotValues[2u * n + scalar]);

        // Independent emulation of the legacy compensated refinement solve.
        // It deliberately uses the original left-looking row accumulation
        // order, including the same fma product-error recovery as the shader.
        auto accumulateProduct = [](std::array<float, 2> accumulated,
                                    float left, float right) {
            const float product = left * right;
            const float productError = std::fma(left, right, -product);
            const float term = -product;
            const float termError = -productError;
            const float high = accumulated[0] + term;
            const float virtualTerm = high - accumulated[0];
            const float sumError =
                (accumulated[0] - (high - virtualTerm)) +
                (term - virtualTerm);
            const float tail = accumulated[1] + sumError + termError;
            const float normalizedHigh = high + tail;
            const float normalizedLow = tail - (normalizedHigh - high);
            return std::array<float, 2>{normalizedHigh, normalizedLow};
        };
        std::vector<float> legacyRefinedSolution = leftLookingReference;
        std::vector<float> refinementResidual(n, 0.0f);
        for (std::uint32_t row = 0u; row < n; ++row) {
            std::array<float, 2> accumulated{rhsScalar[row], 0.0f};
            for (std::uint32_t column = 0u; column < n; ++column)
                accumulated = accumulateProduct(accumulated,
                    original[row * n + column],
                    leftLookingReference[column]);
            refinementResidual[row] = accumulated[0] + accumulated[1];
        }
        std::vector<float> correction(n, 0.0f);
        for (std::uint32_t scalar = 0u; scalar < n; ++scalar) {
            const std::uint32_t originalRow = pivotValues[scalar];
            const float rowScale = floatFromBits(
                pivotValues[n + originalRow]);
            correction[scalar] = refinementResidual[originalRow] * rowScale;
        }
        for (std::uint32_t row = 0u; row < n; ++row) {
            std::array<float, 2> accumulated{correction[row], 0.0f};
            for (std::uint32_t column = 0u; column < row; ++column)
                accumulated = accumulateProduct(accumulated,
                    luValues[row * n + column], correction[column]);
            correction[row] = accumulated[0] + accumulated[1];
        }
        for (std::int32_t reverse = std::int32_t(n) - 1;
             reverse >= 0; --reverse) {
            const std::uint32_t row = std::uint32_t(reverse);
            std::array<float, 2> accumulated{correction[row], 0.0f};
            for (std::uint32_t column = row + 1u; column < n; ++column)
                accumulated = accumulateProduct(accumulated,
                    luValues[row * n + column], correction[column]);
            correction[row] = (accumulated[0] + accumulated[1]) /
                luValues[row * n + row];
        }
        for (std::uint32_t scalar = 0u; scalar < n; ++scalar) {
            correction[scalar] *=
                floatFromBits(pivotValues[2u * n + scalar]);
            legacyRefinedSolution[scalar] += correction[scalar];
        }

        double maxScaledResidual = 0.0;
        double maxSolutionError = 0.0;
        double maxLeftLookingDifference = 0.0;
        double maxOrderedReferenceDifference = 0.0;
        for (std::uint32_t row = 0u; row < n; ++row) {
            double product = 0.0;
            double scale = std::abs(double(rhsScalar[row]));
            for (std::uint32_t column = 0u; column < n; ++column) {
                const double term = double(original[row * n + column]) *
                    double(scalarSolution[column]);
                product += term;
                scale += std::abs(term);
            }
            maxScaledResidual = std::max(maxScaledResidual,
                std::abs(product - double(rhsScalar[row])) /
                    std::max(scale, 1.0e-30));
            maxSolutionError = std::max(maxSolutionError,
                std::abs(double(scalarSolution[row]) -
                         double(known[row])) /
                    std::max(1.0, std::abs(double(known[row]))));
            maxLeftLookingDifference = std::max(maxLeftLookingDifference,
                std::abs(double(scalarSolution[row]) -
                         double(leftLookingReference[row])) /
                    std::max(1.0,
                        std::abs(double(leftLookingReference[row]))));
            const auto& orderedReference = refinementMode == 0u
                ? leftLookingReference : legacyRefinedSolution;
            maxOrderedReferenceDifference =
                std::max(maxOrderedReferenceDifference,
                    std::abs(double(scalarSolution[row]) -
                        double(orderedReference[row])) /
                    std::max(1.0, std::abs(double(orderedReference[row]))));
        }
        require(maxScaledResidual <= 2.0e-5 &&
                maxSolutionError <= 2.0e-4 &&
                maxOrderedReferenceDifference <= 2.0e-6,
            "wide parallel graph patch solve disagrees with its independent known-solution oracle");
        if (refinementMode == 0u)
            require(maxLeftLookingDifference <= 2.0e-6,
                "parallel unrefined graph solve differs from the legacy left-looking solve");
        else
            require(maxOrderedReferenceDifference <= 2.0e-6,
                "parallel compensated graph solve differs from the legacy ordered refinement");
    };
    runSolve(0u, 0u);
    runSolve(1u, 0u);
    runSolve(0u, 1u);
    runSolve(1u, 1u);
    std::cout << "graph_patch_parallel_triangular_solve=192dof "
                 "partial_pivot=true refined=true trials=2 "
                 "result=PASS\n";
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
            {22u, u32(0u)}, {23u, u32(0u)},
            {24u, u32(0u)}, {25u, u32(0u)}}, 1u, 256u);
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

    // Reproduce the restart path from a nonzero accumulated interface
    // solution. The physical RHS is b=A*x_old+r, so the next Schur cycle
    // must condense the freshly measured full residual r, not reuse b or the
    // original cycle-zero condensed RHS.
    const std::array<double, 6> priorRestartSolution{
        0.0, 0.0, 0.0, 0.35, -0.2, 0.1};
    const std::array<double, 6> restartResidual{
        0.4, -0.75, 0.3, -0.2, 0.55, -0.45};
    const auto priorRestartAction = multiply6(matrix, priorRestartSolution);
    std::array<double, 6> restartPhysicalRhs{};
    for (std::uint32_t i = 0u; i < 6u; ++i)
        restartPhysicalRhs[i] =
            priorRestartAction[i] + restartResidual[i];
    const auto restartInteriorParticular = solveAii(restartResidual);
    const auto restartInteriorLift = runTransform(1u, zeroFull, zeroFull,
        restartInteriorParticular);
    const auto restartInteriorAction = multiply6(matrix, restartInteriorLift);
    const auto recondensedRestartRhs = runTransform(2u, restartResidual,
        restartInteriorAction, zeroPacked);
    for (std::uint32_t i = 0u; i < 3u; ++i)
        require(std::abs(recondensedRestartRhs[i]) < 1.0e-6,
            "restart Schur RHS retained a nonzero local residual");
    for (std::uint32_t i = 3u; i < 6u; ++i)
        require(std::abs(recondensedRestartRhs[i] -
                         (restartResidual[i] - restartInteriorAction[i])) <=
                    2.0e-5 * std::max(1.0,
                                      std::abs(restartResidual[i])),
            "restart Schur RHS did not recondense r_Gamma - A_Gamma_i Aii^-1 r_i");

    const auto incorrectlyReusedPhysicalRhsInterior =
        solveAii(restartPhysicalRhs);
    const auto incorrectlyReusedPhysicalRhsLift = runTransform(
        1u, zeroFull, zeroFull, incorrectlyReusedPhysicalRhsInterior);
    const auto incorrectlyReusedPhysicalRhsAction =
        multiply6(matrix, incorrectlyReusedPhysicalRhsLift);
    const auto incorrectlyReusedPhysicalRhs = runTransform(2u,
        restartPhysicalRhs, incorrectlyReusedPhysicalRhsAction, zeroPacked);
    double staleRhsSeparation = 0.0;
    for (std::uint32_t i = 3u; i < 6u; ++i)
        staleRhsSeparation = std::max(staleRhsSeparation,
            std::abs(incorrectlyReusedPhysicalRhs[i] -
                     recondensedRestartRhs[i]));
    require(staleRhsSeparation > 1.0e-3,
        "restart fixture did not distinguish the measured residual from the old physical RHS");

    // Build the actual 3x3 Schur operator S=A_GG-A_Gi*A_ii^-1*A_iG
    // using the same native transform and Aii solve pipelines, then solve
    // S*deltaGamma=r_Gamma-A_Gi*A_ii^-1*r_i. Do not seed recovery with the
    // already-known full dense solution: this verifies the restart input path.
    std::array<double, 9> restartSchurMatrix{};
    for (std::uint32_t column = 0u; column < 3u; ++column) {
        std::array<double, 6> gammaBasis{};
        gammaBasis[3u + column] = 1.0;
        const auto fullGammaAction = multiply6(matrix, gammaBasis);
        const auto localAction = runTransform(5u, fullGammaAction,
            zeroFull, zeroPacked);
        const auto localSolve = solveAii(localAction);
        const auto localLift = runTransform(1u, zeroFull, zeroFull, localSolve);
        const auto fullLocalAction = multiply6(matrix, localLift);
        const auto schurColumn = runTransform(3u, fullGammaAction,
            fullLocalAction, zeroPacked);
        for (std::uint32_t row = 0u; row < 3u; ++row)
            restartSchurMatrix[row * 3u + column] = schurColumn[3u + row];
    }
    const std::vector<double> condensedGammaRhs{
        recondensedRestartRhs[3], recondensedRestartRhs[4],
        recondensedRestartRhs[5]};
    const auto restartGammaCorrection = solveDense(
        std::vector<double>(restartSchurMatrix.begin(),
                            restartSchurMatrix.end()),
        condensedGammaRhs, 3u);
    std::array<double, 6> restartSchurSolution{};
    for (std::uint32_t i = 0u; i < 3u; ++i)
        restartSchurSolution[3u + i] = restartGammaCorrection[i];
    const auto restartGammaAction = multiply6(matrix, restartSchurSolution);
    const auto restartRecoveryRhs = runTransform(5u, restartResidual,
        restartGammaAction, zeroPacked);
    const auto restartRecoveryInterior = solveAii(restartRecoveryRhs);
    const auto restartRecovered = runTransform(4u, restartSchurSolution,
        zeroFull, restartRecoveryInterior);
    const auto restartRecoveredAction = multiply6(matrix, restartRecovered);

    const auto restartReferenceVector = solveDense(
        std::vector<double>(matrix.begin(), matrix.end()),
        std::vector<double>(restartResidual.begin(), restartResidual.end()), 6u);
    std::array<double, 6> restartReference{};
    for (std::uint32_t i = 0u; i < 6u; ++i)
        restartReference[i] = restartReferenceVector[i];
    const auto restartAccumulatedAction = [&] {
        std::array<double, 6> accumulated{};
        for (std::uint32_t i = 0u; i < 6u; ++i)
            accumulated[i] = priorRestartSolution[i] + restartRecovered[i];
        return multiply6(matrix, accumulated);
    }();
    for (std::uint32_t i = 0u; i < 6u; ++i) {
        require(std::abs(restartRecovered[i] - restartReference[i]) <=
                    2.0e-5 * std::max(1.0, std::abs(restartReference[i])) &&
                std::abs(restartRecoveredAction[i] - restartResidual[i]) <=
                    2.0e-4 * std::max(1.0, std::abs(restartResidual[i])) &&
                std::abs(restartAccumulatedAction[i] -
                         restartPhysicalRhs[i]) <=
                    2.0e-4 * std::max(1.0, std::abs(restartPhysicalRhs[i])),
            "solved condensed restart plus recovered correction disagrees with full-system residual");
    }

    // Negative control: omit -A_Gi*Aii^-1*r_i from the restart RHS. Recovery
    // still solves the local block exactly, but the resulting full correction
    // must fail the original residual equation by a visible margin.
    const auto uncondensedGammaCorrection = solveDense(
        std::vector<double>(restartSchurMatrix.begin(),
                            restartSchurMatrix.end()),
        std::vector<double>{restartResidual[3], restartResidual[4],
                           restartResidual[5]}, 3u);
    std::array<double, 6> uncondensedSchurSolution{};
    for (std::uint32_t i = 0u; i < 3u; ++i)
        uncondensedSchurSolution[3u + i] = uncondensedGammaCorrection[i];
    const auto uncondensedGammaAction =
        multiply6(matrix, uncondensedSchurSolution);
    const auto uncondensedRecoveryRhs = runTransform(5u, restartResidual,
        uncondensedGammaAction, zeroPacked);
    const auto uncondensedRecoveryInterior = solveAii(uncondensedRecoveryRhs);
    const auto uncondensedRecovered = runTransform(4u,
        uncondensedSchurSolution, zeroFull, uncondensedRecoveryInterior);
    const auto uncondensedAction = multiply6(matrix, uncondensedRecovered);
    double uncondensedResidualError = 0.0;
    for (std::uint32_t i = 0u; i < 6u; ++i)
        uncondensedResidualError = std::max(uncondensedResidualError,
            std::abs(uncondensedAction[i] - restartResidual[i]));
    require(uncondensedResidualError > 1.0e-2,
        "uncondensed restart RHS unexpectedly satisfied the full residual equation");
    std::cout << "schur_restart_recondensation=nonzero_local_residual "
                 "prior_interface_solution=true physical_rhs_preserved=true "
                 "result=PASS\n";

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
        {22u, u32(0u)}, {23u, u32(0u)},
        {24u, u32(0u)}, {25u, u32(0u)}}, 1u, 256u);
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
            {9u, contactFailureWitness},
            {10u, makeBuffer(device, std::vector<std::uint32_t>(6u, NM_INVALID_INDEX))},
            {11u, u32(0u)}}, 3u, 32u);
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
            {9u, localWitness},
            {10u, makeBuffer(device, std::vector<std::uint32_t>(6u, NM_INVALID_INDEX))},
            {11u, u32(0u)}}, 3u, 32u);
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
        {22u, eligibilityBuffer}, {23u, u32(1u)},
        {24u, u32(0u)}, {25u, u32(0u)}}, 2u, 256u);
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

    // Node-level promotion keeps untouched xyz triples in Aii while moving
    // only an actively contacted endpoint into the Schur interface.
    const auto resetNodes = pipelineWithRelativeInnerForcingDisabled(device,
        library, "numi_matter_metal::nm_fgmres_schur_reset_active_node_index");
    const auto maskNodes = pipelineWithRelativeInnerForcingDisabled(device,
        library, "numi_matter_metal::nm_fgmres_schur_mask_promoted_node_blocks");
    auto nodeStatus = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
    const auto staticNodeMap = makeBuffer(device,
        std::vector<std::uint32_t>{0u, 1u, NM_INVALID_INDEX,
            NM_INVALID_INDEX, NM_INVALID_INDEX, NM_INVALID_INDEX});
    auto activeNodeMap = makeBuffer(device,
        std::vector<std::uint32_t>(6u, NM_INVALID_INDEX));
    dispatch(device, queue, resetNodes, {
        {0u, dispatchBuffer}, {1u, staticNodeMap}, {2u, activeNodeMap},
        {3u, nodeStatus}}, 6u, 32u);
    require(readBuffer<std::uint32_t>(activeNodeMap, 6u) ==
            std::vector<std::uint32_t>{0u, 1u, NM_INVALID_INDEX,
                NM_INVALID_INDEX, NM_INVALID_INDEX, NM_INVALID_INDEX},
        "dynamic node map did not reset from its immutable closed core");

    std::vector<NMDeformableContactGPU> nodeContacts(3u);
    nodeContacts[0].identity.w = NM_CONTACT_VALID;
    nodeContacts[0].nodes = nm_uint4{0u, NM_INVALID_INDEX,
        NM_INVALID_INDEX, NM_INVALID_INDEX};
    nodeContacts[0].extraNodes = nm_uint4{NM_INVALID_INDEX,
        NM_INVALID_INDEX, 0u, 0u};
    for (std::uint32_t i = 1u; i < 3u; ++i) {
        nodeContacts[i].identity.w = NM_CONTACT_VALID;
        nodeContacts[i].nodes = nm_uint4{1u, NM_INVALID_INDEX,
            NM_INVALID_INDEX, NM_INVALID_INDEX};
        nodeContacts[i].extraNodes = nm_uint4{NM_INVALID_INDEX,
            NM_INVALID_INDEX, 0u, 0u};
    }
    auto nodeContactsBuffer = makeBuffer(device, nodeContacts);
    auto nodeActiveIndices = makeBuffer(device,
        std::vector<std::uint32_t>{0u, 0u, 0u});
    auto nodeActiveCounts = makeBuffer(device, std::vector<std::uint32_t>{1u});
    auto nodePatchByNode = makeBuffer(device,
        std::vector<std::uint32_t>{0u, 0u, NM_INVALID_INDEX,
            NM_INVALID_INDEX, NM_INVALID_INDEX, NM_INVALID_INDEX});
    auto nodeEligibility = makeBuffer(device, std::vector<std::uint32_t>{1u});
    auto nodeWitness = makeBuffer(device,
        std::vector<std::uint32_t>{NM_INVALID_INDEX});
    dispatch(device, queue, classify, {
        {0u, dispatchBuffer}, {1u, nodeContactsBuffer},
        {2u, nodeActiveIndices}, {3u, nodeActiveCounts},
        {4u, nodePatchByNode}, {5u, nodeEligibility}, {6u, nodeStatus},
        {7u, u32(1u)}, {8u, u32(3u)}, {9u, nodeWitness},
        {10u, activeNodeMap}, {11u, u32(1u)}}, 3u, 32u);
    const auto nodeMap = readBuffer<std::uint32_t>(activeNodeMap, 6u);
    require(readBuffer<NMMatterStatusGPU>(nodeStatus, 1u)[0].code ==
                NM_STATUS_SUCCESS && nodeMap[0] == NM_INVALID_INDEX &&
            nodeMap[1] == 1u &&
            readBuffer<std::uint32_t>(nodeEligibility, 1u)[0] == 1u &&
            readBuffer<std::uint32_t>(nodeWitness, 1u)[0] == NM_INVALID_INDEX,
        "active contact must promote only its endpoint, retaining core peers");

    std::vector<float> nodeA(36u);
    for (std::uint32_t row = 0u; row < 6u; ++row)
        for (std::uint32_t column = 0u; column < 6u; ++column)
            nodeA[row * 6u + column] =
                0.03f * float(1u + row * 6u + column);
    const std::array<float, 9> retainedBlock{
        4.0f, 1.0f, 0.5f, 0.25f, 3.0f, -0.2f, -0.1f, 0.3f, 2.0f};
    for (std::uint32_t row = 0u; row < 3u; ++row)
        for (std::uint32_t column = 0u; column < 3u; ++column)
            nodeA[(row + 3u) * 6u + column + 3u] =
                retainedBlock[row * 3u + column];
    auto nodeMatrices = makeBuffer(device, nodeA);
    auto nodeSources = makeBuffer(device,
        std::vector<nm_uint4>{nm_uint4{0u, 0u, 2u, 0u}});
    auto nodeGlobals = makeBuffer(device, std::vector<std::uint32_t>{0u, 1u});
    dispatch(device, queue, maskNodes, {
        {0u, dispatchBuffer}, {1u, nodeSources}, {2u, nodeGlobals},
        {3u, activeNodeMap}, {4u, nodeMatrices}, {5u, u32(1u)},
        {6u, u32(36u)}, {7u, u32(2u)}, {8u, nodeStatus}}, 1u, 256u);
    const auto maskedNodeA = readBuffer<float>(nodeMatrices, 36u);
    require(readBuffer<NMMatterStatusGPU>(nodeStatus, 1u)[0].code ==
                NM_STATUS_SUCCESS,
        "node-level Aii mask failed");
    for (std::uint32_t row = 0u; row < 6u; ++row) {
        for (std::uint32_t column = 0u; column < 6u; ++column) {
            float expected = nodeA[row * 6u + column];
            if (row < 3u || column < 3u)
                expected = row == column ? 1.0f : 0.0f;
            require(maskedNodeA[row * 6u + column] == expected,
                "promotion must identity-mask exactly one xyz triple");
        }
    }

    const auto nodePatches = makeBuffer(device,
        std::vector<nm_uint4>{nm_uint4{0u, 2u, 0u, 0u}});
    const auto nodePatchValues = makeBuffer(device,
        std::vector<std::uint32_t>{0u, 1u});
    auto nodePivots = makeBuffer(device, std::vector<std::uint32_t>(18u));
    auto nodeFactorValid = makeBuffer(device, std::vector<std::uint32_t>(1u));
    auto nodeOriginal = makeBuffer(device, std::vector<float>(36u));
    dispatch(device, queue, factor, {
        {0u, dispatchBuffer}, {1u, nodePatches}, {2u, nodeMatrices},
        {3u, nodePivots}, {4u, nodeFactorValid}, {5u, nodeStatus},
        {6u, u32(1u)}, {7u, u32(36u)}, {8u, u32(18u)},
        {9u, nodeOriginal}, {10u, statesBuffer}, {11u, u32(0u)},
        {12u, makeBuffer(device, std::vector<std::uint32_t>{0u})},
        {13u, u32(0u)}, {14u, u32(0u)}, {15u, u32(0u)},
        {16u, nodeWitness}}, 1u, 256u);
    require(readBuffer<std::uint32_t>(nodeFactorValid, 1u)[0] == 1u &&
            readBuffer<NMMatterStatusGPU>(nodeStatus, 1u)[0].code ==
                NM_STATUS_SUCCESS,
        "untouched Aii block did not factor after endpoint promotion");

    // Basis stores one float4 per FEM node; xyz are that node's DOFs.
    std::vector<nm_float4> nodeRhs(2u);
    nodeRhs[0] = nm_float4{99.0f, -88.0f, 77.0f, 0.0f};
    nodeRhs[1] = nm_float4{6.0f, -3.0f, 1.0f, 0.0f};
    auto nodeRhsBuffer = makeBuffer(device, nodeRhs);
    auto nodeCorrections = makeBuffer(device, std::vector<nm_float4>(2u));
    auto zeroRegional = makeBuffer(device, std::vector<nm_float4>(6u));
    auto nodeZeroCoefficients = makeBuffer(device, std::vector<float>{0.0f});
    dispatch(device, queue, solve, {
        {0u, dispatchBuffer}, {1u, nodePatches}, {2u, nodePatchValues},
        {3u, nodeRhsBuffer}, {4u, zeroRegional}, {5u, zeroRegional},
        {6u, nodeZeroCoefficients}, {7u, nodeMatrices}, {8u, nodePivots},
        {9u, nodeFactorValid}, {10u, nodeCorrections}, {11u, nodeStatus},
        {12u, u32(1u)}, {13u, u32(2u)}, {14u, u32(36u)},
        {15u, u32(18u)}, {16u, u32(0u)}, {17u, statesBuffer},
        {18u, nodeOriginal}, {19u, u32(0u)}, {20u, candidateBuffer},
        {21u, u32(0u)}, {22u, u32(0u)}, {23u, u32(0u)},
        {24u, activeNodeMap}, {25u, u32(1u)}}, 1u, 256u);
    const auto nodeSolution = readBuffer<nm_float4>(nodeCorrections, 2u);
    const std::vector<double> retainedDense(retainedBlock.begin(),
                                             retainedBlock.end());
    const auto solveNodeDense = [](std::vector<double> matrix,
                                   std::vector<double> rhs,
                                   std::uint32_t dimension) {
        for (std::uint32_t column = 0u; column < dimension; ++column) {
            std::uint32_t pivot = column;
            for (std::uint32_t row = column + 1u; row < dimension; ++row)
                if (std::abs(matrix[row * dimension + column]) >
                    std::abs(matrix[pivot * dimension + column])) pivot = row;
            require(std::abs(matrix[pivot * dimension + column]) > 1.0e-12,
                    "node Schur reference matrix is singular");
            for (std::uint32_t k = column; k < dimension; ++k)
                std::swap(matrix[column * dimension + k],
                          matrix[pivot * dimension + k]);
            std::swap(rhs[column], rhs[pivot]);
            const double diagonal = matrix[column * dimension + column];
            for (std::uint32_t k = column; k < dimension; ++k)
                matrix[column * dimension + k] /= diagonal;
            rhs[column] /= diagonal;
            for (std::uint32_t row = 0u; row < dimension; ++row) {
                if (row == column) continue;
                const double scale = matrix[row * dimension + column];
                for (std::uint32_t k = column; k < dimension; ++k)
                    matrix[row * dimension + k] -=
                        scale * matrix[column * dimension + k];
                rhs[row] -= scale * rhs[column];
            }
        }
        return rhs;
    };
    const auto retainedReference = solveNodeDense(retainedDense,
        std::vector<double>{6.0, -3.0, 1.0}, 3u);
    require(readBuffer<NMMatterStatusGPU>(nodeStatus, 1u)[0].code ==
                NM_STATUS_SUCCESS &&
            nodeSolution[0].x == 0.0f && nodeSolution[0].y == 0.0f &&
            nodeSolution[0].z == 0.0f &&
            std::abs(nodeSolution[1].x - retainedReference[0]) < 2.0e-5 &&
            std::abs(nodeSolution[1].y - retainedReference[1]) < 2.0e-5 &&
            std::abs(nodeSolution[1].z - retainedReference[2]) < 2.0e-5,
        "promoted RHS leaked into Aii solve or retained solve changed");

    auto nodeInput = makeBuffer(device, std::vector<nm_float4>{
        nm_float4{2.0f, 3.0f, 4.0f, 0.0f},
        nm_float4{5.0f, 6.0f, 7.0f, 0.0f},
        nm_float4{}, nm_float4{}, nm_float4{}, nm_float4{}});
    auto nodeProjectionOutput = makeBuffer(device,
        std::vector<nm_float4>(6u));
    dispatch(device, queue, transform, {
        {0u, dispatchBuffer}, {30u, layoutBuffer}, {1u, activeNodeMap},
        {2u, nodeInput}, {3u, zeroInput}, {4u, nodeCorrections},
        {5u, nodeProjectionOutput}, {6u, u32(0u)}, {7u, u32(2u)},
        {8u, nodeStatus}, {9u, nodePatchByNode}, {10u, nodeEligibility},
        {11u, u32(1u)}, {12u, u32(0u)}}, 1u, 32u);
    const auto nodeProjection =
        readBuffer<nm_float4>(nodeProjectionOutput, 6u);
    require(readBuffer<NMMatterStatusGPU>(nodeStatus, 1u)[0].code ==
                NM_STATUS_SUCCESS &&
            nodeProjection[0].x == 2.0f && nodeProjection[0].y == 3.0f &&
            nodeProjection[0].z == 4.0f && nodeProjection[1].x == 0.0f &&
            nodeProjection[1].y == 0.0f && nodeProjection[1].z == 0.0f,
        "node-level transform did not retain exactly the active interface node");

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
    const auto emptyElementBlocksBuffer = makeBuffer(device,
        std::vector<float>(144u, 0.0f));
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
                std::vector<std::uint32_t>{dueAware})},
            {30u, emptyElementBlocksBuffer}}, 1u, rowCount);
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
        {23u, makeBuffer(device, std::vector<std::uint32_t>{0u})},
        {24u, makeBuffer(device, std::vector<std::uint32_t>{0u})},
        {25u, makeBuffer(device, std::vector<std::uint32_t>{0u})}},
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

void checkSchurGraphInterfaceAction(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLLibrary> library) {
    NMMatterDispatchGPU shape{};
    shape.abiVersion = NM_MATTER_ABI_VERSION;
    shape.environmentCount = 1u;
    shape.femNodeCount = 4u;
    const auto dispatchBuffer = makeBuffer(device,
        std::vector<NMMatterDispatchGPU>{shape});
    // Two Schur extraction blocks map to nonidentity graph patch indices;
    // gaps catch accidental comparisons between these distinct index spaces.
    std::vector<nm_uint4> patches(10u, nm_uint4{});
    patches[4] = nm_uint4{0u, 2u, 0u, 0u};
    patches[9] = nm_uint4{2u, 2u, 36u, 0u};
    const std::vector<nm_uint4> extractionSources{
        nm_uint4{4u, 0u, 1u, 0u}, nm_uint4{9u, 1u, 1u, 36u}};
    const std::vector<std::uint32_t> patchNodes{0u, 1u, 0u, 2u};
    std::vector<nm_uint4> rows;
    const std::array<std::uint32_t, 2> graphPatchIDs{4u, 9u};
    for (std::uint32_t p = 0u; p < 2u; ++p) {
        for (std::uint32_t local = 0u; local < 2u; ++local) {
            const std::uint32_t node = patchNodes[p * 2u + local];
            for (std::uint32_t axis = 0u; axis < 3u; ++axis)
                rows.push_back(nm_uint4{graphPatchIDs[p], local, axis, node});
        }
    }
    const std::vector<std::uint32_t> nodePatchOffsets{0u, 2u, 3u, 4u, 4u};
    const std::vector<std::uint32_t> nodePatchValues{0u, 2u, 1u, 3u};
    const std::vector<std::uint32_t> interiorByNode{
        NM_INVALID_INDEX, 0u, 1u, NM_INVALID_INDEX};
    const std::vector<std::uint32_t> interiorPatchByNode{
        NM_INVALID_INDEX, 0u, 1u, NM_INVALID_INDEX};
    std::vector<float> matrices(72u, 0.0f);
    const std::array<float, 9> a01{
        2.0f, -1.0f, 0.5f,
        0.25f, 3.0f, -0.75f,
        -1.5f, 0.125f, 4.0f};
    const std::array<float, 9> a02{
        -0.5f, 2.25f, 1.0f,
        1.75f, -3.0f, 0.25f,
        0.125f, 0.75f, -2.0f};
    const auto placeBlock = [&](const std::uint32_t patchBase,
                                const std::array<float, 9>& block) {
        for (std::uint32_t out = 0u; out < 3u; ++out)
            for (std::uint32_t in = 0u; in < 3u; ++in)
                matrices[patchBase + out * 6u + 3u + in] =
                    block[out * 3u + in];
    };
    placeBlock(0u, a01);
    placeBlock(36u, a02);

    const auto patchesBuffer = makeBuffer(device, patches);
    const auto extractionSourcesBuffer = makeBuffer(device, extractionSources);
    const auto patchNodesBuffer = makeBuffer(device, patchNodes);
    const auto rowsBuffer = makeBuffer(device, rows);
    const auto nodePatchOffsetsBuffer = makeBuffer(device, nodePatchOffsets);
    const auto nodePatchValuesBuffer = makeBuffer(device, nodePatchValues);
    const auto interiorByNodeBuffer = makeBuffer(device, interiorByNode);
    const auto interiorPatchByNodeBuffer = makeBuffer(device, interiorPatchByNode);
    const auto eligibilityBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{1u, 1u});
    const auto matricesBuffer = makeBuffer(device, matrices);
    const std::vector<nm_float4> input{
        nm_float4{9.0f, -7.0f, 2.0f, 0.0f},
        nm_float4{0.25f, -1.0f, 2.0f, 0.0f},
        nm_float4{-0.75f, 0.5f, 1.25f, 0.0f},
        nm_float4{4.0f, 3.0f, -2.0f, 0.0f}};
    const auto inputBuffer = makeBuffer(device, input);
    const auto outputBuffer = makeBuffer(device,
        std::vector<nm_float4>(4u, nm_float4{0.0f, 0.0f, 0.0f, 7.0f}));
    const auto statusBuffer = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
    const auto u32 = [&](std::uint32_t value) {
        return makeBuffer(device, std::vector<std::uint32_t>{value});
    };
    auto action = pipelineWithRelativeInnerForcingDisabled(device, library,
        "numi_matter_metal::nm_fgmres_schur_graph_interface_action");
    const auto run = [&] {
        dispatch(device, queue, action, {
            {0u, dispatchBuffer}, {1u, patchesBuffer},
            {2u, patchNodesBuffer}, {3u, rowsBuffer},
            {4u, nodePatchOffsetsBuffer}, {5u, nodePatchValuesBuffer},
            {6u, interiorByNodeBuffer}, {7u, interiorPatchByNodeBuffer},
            {8u, eligibilityBuffer}, {9u, matricesBuffer},
            {10u, inputBuffer}, {11u, outputBuffer}, {12u, statusBuffer},
            {13u, u32(10u)}, {14u, u32(4u)}, {15u, u32(12u)},
            {16u, u32(4u)}, {17u, u32(72u)}, {18u, u32(2u)},
            {19u, extractionSourcesBuffer}, {20u, u32(2u)}},
            1u, 32u);
        require(readBuffer<NMMatterStatusGPU>(statusBuffer, 1u)[0].code ==
                NM_STATUS_SUCCESS,
            "raw Schur interface action rejected a valid patch layout");
    };
    const auto setOutput = [&](const nm_float4 value) {
        std::vector<nm_float4> values(4u, value);
        std::memcpy([outputBuffer contents], values.data(),
                    values.size() * sizeof(nm_float4));
    };
    const auto multiply = [](const std::array<float, 9>& matrix,
                             const nm_float4 vector) {
        return std::array<float, 3>{
            matrix[0] * vector.x + matrix[1] * vector.y + matrix[2] * vector.z,
            matrix[3] * vector.x + matrix[4] * vector.y + matrix[5] * vector.z,
            matrix[6] * vector.x + matrix[7] * vector.y + matrix[8] * vector.z};
    };
    const auto expected01 = multiply(a01, input[1]);
    const auto expected02 = multiply(a02, input[2]);
    const std::array<float, 3> expected{
        expected01[0] + expected02[0],
        expected01[1] + expected02[1],
        expected01[2] + expected02[2]};
    run();
    auto output = readBuffer<nm_float4>(outputBuffer, 4u);
    const std::array<float, 3> actual{
        output[0].x, output[0].y, output[0].z};
    for (std::uint32_t axis = 0u; axis < 3u; ++axis)
        require(std::abs(actual[axis] - expected[axis]) < 2.0e-6f,
            "shared interface row did not sum nonsymmetric A_Gamma,i blocks");
    require(output[0].w == 7.0f && output[1].w == 7.0f &&
            output[2].w == 7.0f && output[3].w == 7.0f,
        "raw Schur interface action changed pressure/preserved components");
    for (std::uint32_t node = 1u; node < 4u; ++node)
        require(output[node].x == 0.0f && output[node].y == 0.0f &&
                output[node].z == 0.0f,
            "raw Schur action wrote outside the interface row");

    // Dynamic contact promotes one core into Gamma. Its former interior input
    // is ignored, while the still-eligible neighboring core contributes.
    auto eligibility = std::vector<std::uint32_t>{0u, 1u};
    std::memcpy([eligibilityBuffer contents], eligibility.data(),
                eligibility.size() * sizeof(std::uint32_t));
    setOutput(nm_float4{0.0f, 0.0f, 0.0f, 7.0f});
    run();
    output = readBuffer<nm_float4>(outputBuffer, 4u);
    const std::array<float, 3> promotedActual{
        output[0].x, output[0].y, output[0].z};
    for (std::uint32_t axis = 0u; axis < 3u; ++axis)
        require(std::abs(promotedActual[axis] - expected02[axis]) < 2.0e-6f,
            "promoted Schur core still contributed an eliminated column");

    // An all-zero due-aware/raw off-diagonal matrix and a zero interior vector
    // each produce exactly zero, including a layout with no interior nodes.
    std::fill(matrices.begin(), matrices.end(), 0.0f);
    std::memcpy([matricesBuffer contents], matrices.data(),
                matrices.size() * sizeof(float));
    eligibility = {1u, 1u};
    std::memcpy([eligibilityBuffer contents], eligibility.data(),
                eligibility.size() * sizeof(std::uint32_t));
    setOutput(nm_float4{0.0f, 0.0f, 0.0f, 7.0f});
    run();
    output = readBuffer<nm_float4>(outputBuffer, 4u);
    require(output[0].x == 0.0f && output[0].y == 0.0f &&
            output[0].z == 0.0f,
        "zero/due-padded Schur matrix produced a nonzero interface action");
    auto noInterior = std::vector<std::uint32_t>(4u, NM_INVALID_INDEX);
    std::memcpy([interiorByNodeBuffer contents], noInterior.data(),
                noInterior.size() * sizeof(std::uint32_t));
    std::memcpy([interiorPatchByNodeBuffer contents], noInterior.data(),
                noInterior.size() * sizeof(std::uint32_t));
    setOutput(nm_float4{0.0f, 0.0f, 0.0f, 7.0f});
    run();
    output = readBuffer<nm_float4>(outputBuffer, 4u);
    require(std::all_of(output.begin(), output.end(), [](const nm_float4 value) {
                return value.x == 0.0f && value.y == 0.0f && value.z == 0.0f;
            }),
        "zero-interior Schur layout produced a nonzero interface action");

    // Failed/completed Newton tails must never look like a valid action audit.
    const auto auditResult = makeBuffer(device, std::vector<std::uint32_t>(24u, 99u));
    const auto liveStatus = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
    const auto liveState = makeBuffer(device,
        std::vector<NMFGMRESStateGPU>{NMFGMRESStateGPU{}});
    NMMicrostepGPU auditMicro{};
    auditMicro.solverIteration = 1u;
    const auto microBuffer = makeBuffer(device, std::vector<NMMicrostepGPU>{auditMicro});
    auto compare = pipelineWithRelativeInnerForcingDisabled(device, library,
        "numi_matter_metal::nm_fgmres_schur_raw_audit_compare");
    const auto compareAudit = [&] {
        dispatch(device, queue, compare, {
            {0u, dispatchBuffer}, {1u, interiorByNodeBuffer},
            {2u, interiorPatchByNodeBuffer}, {3u, eligibilityBuffer},
            {4u, inputBuffer}, {5u, inputBuffer},
            {6u, statusBuffer}, {7u, statusBuffer}, {8u, auditResult},
            {9u, u32(4u)}, {10u, u32(0u)}, {11u, u32(2u)},
            {12u, u32(0u)}, {13u, liveStatus}, {14u, liveState},
            {15u, microBuffer}}, 1u, 256u);
        return readBuffer<std::uint32_t>(auditResult, 24u);
    };
    auto audit = compareAudit();
    require(audit[7] == 0u && audit[3] == 0u,
        "healthy equal Schur actions did not produce a valid zero-error audit");
    NMFGMRESStateGPU latched{};
    latched.nonlinear.w = 1.0f;
    std::memcpy([liveState contents], &latched, sizeof(latched));
    audit = compareAudit();
    require(audit[7] == 2u && audit[10] == 1u,
        "completed Newton tail was reported as a numerical action mismatch");
    latched.nonlinear.w = 0.0f;
    std::memcpy([liveState contents], &latched, sizeof(latched));
    NMMatterStatusGPU failed{};
    failed.code = NM_STATUS_NONLINEAR_SOLVER_FAILURE;
    std::memcpy([liveStatus contents], &failed, sizeof(failed));
    audit = compareAudit();
    require(audit[7] == 2u && audit[8] == failed.code,
        "unhealthy Newton tail was reported as a numerical action mismatch");
}

void checkHybridGlobalCoarseRAPOracle(
    id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library) {
    constexpr std::uint32_t nodes = 8u;
    constexpr std::uint32_t modes = 6u;
    constexpr std::uint32_t physicalDofs = 3u * nodes;
    constexpr std::uint32_t unknownCount = 2u * nodes;
    NMMatterDispatchGPU matter{};
    matter.abiVersion = NM_MATTER_ABI_VERSION;
    matter.environmentCount = 1u;
    matter.objectCount = 1u;
    matter.femNodeCount = nodes;
    NMPartitionedReductionBlockDispatchGPU block{};
    block.globalNodeCount = nodes;
    block.basisNodeCount = nodes;
    block.modeCount = modes;
    block.tileCount = 1u;
    block.environmentCount = 1u;
    block.regionCount = 1u;
    block.blockModeCount = modes;
    block.reserved = 1u;
    NMFGMRESLayoutGPU layout{};
    layout.unknownCount = unknownCount;

    // Six nonidentity B modes on nodes 0..5; nodes 6..7 retain their full
    // fine xyz complement. Each reduced mode couples two physical nodes.
    std::vector<nm_float4> basis(modes * nodes, nm_float4{});
    for (std::uint32_t mode = 0u; mode < modes; ++mode) {
        const std::uint32_t neighbor = (mode + 1u) % modes;
        basis[mode * nodes + mode] = {
            1.0f, 0.03f * float(mode + 1u), -0.02f * float(mode), 0.0f};
        basis[mode * nodes + neighbor] = {
            0.12f, -0.02f * float(mode + 1u), 0.015f, 0.0f};
    }
    std::vector<std::uint32_t> globalRows(nodes);
    std::vector<std::uint32_t> modeMap(modes);
    std::vector<NMPartitionedReductionSparseRangeGPU> nodeRanges(nodes);
    std::vector<std::uint32_t> nodeSlots;
    for (std::uint32_t row = 0u; row < nodes; ++row) {
        globalRows[row] = row;
        if (row < modes) {
            nodeRanges[row].first = static_cast<std::uint32_t>(nodeSlots.size());
            for (std::uint32_t mode = 0u; mode < modes; ++mode) {
                const nm_float4 value = basis[mode * nodes + row];
                if (value.x != 0.0f || value.y != 0.0f || value.z != 0.0f)
                    nodeSlots.push_back(mode);
            }
            nodeRanges[row].count = static_cast<std::uint32_t>(nodeSlots.size()) -
                nodeRanges[row].first;
        }
    }
    for (std::uint32_t mode = 0u; mode < modes; ++mode) modeMap[mode] = mode;

    // A dense symmetric, diagonally dominant physical operator with explicit
    // selected/fine off-diagonal coupling. This supplies a nonzero q/fine
    // block while keeping an independent double-precision oracle.
    std::vector<double> physicalA(physicalDofs * physicalDofs, 0.0);
    double qFineCoupling = 0.0;
    for (std::uint32_t row = 0u; row < physicalDofs; ++row) {
        for (std::uint32_t column = 0u; column < physicalDofs; ++column) {
            double value = 0.0;
            if (row == column) {
                value = 3.0 + 0.01 * double(row);
            } else {
                const std::uint32_t lo = std::min(row, column);
                const std::uint32_t hi = std::max(row, column);
                const int code = int((lo * 13u + hi * 7u) % 11u) - 5;
                value = 0.003 * double(code);
                const bool rowSelected = row / 3u < modes;
                const bool colSelected = column / 3u < modes;
                if (rowSelected != colSelected) qFineCoupling += std::abs(value);
            }
            physicalA[row * physicalDofs + column] = value;
        }
    }
    require(qFineCoupling > 0.1,
        "hybrid RAP oracle omitted selected/fine operator coupling");

    // Nonidentity but well-conditioned q coordinates define six physical
    // coarse vectors P=BQ. The same Q is consumed by production prepare,
    // project, and add kernels.
    std::vector<float> qCoordinates(modes * modes);
    for (std::uint32_t coarse = 0u; coarse < modes; ++coarse)
        for (std::uint32_t q = 0u; q < modes; ++q) {
            const std::uint32_t code =
                ((coarse + 3u) * (q + 5u) * 7u) % 13u;
            qCoordinates[coarse * modes + q] =
                (coarse == q ? 1.0f : 0.0f) +
                0.003f * float(int(code) - 6);
        }
    std::vector<nm_float4> physicalP(modes * nodes, nm_float4{});
    for (std::uint32_t coarse = 0u; coarse < modes; ++coarse)
        for (std::uint32_t row = 0u; row < nodes; ++row)
            for (std::uint32_t q = 0u; q < modes; ++q) {
                const float coefficient = qCoordinates[coarse * modes + q];
                const nm_float4 b = basis[q * nodes + row];
                nm_float4& p = physicalP[coarse * nodes + row];
                p.x += coefficient * b.x;
                p.y += coefficient * b.y;
                p.z += coefficient * b.z;
            }

    double projectedQFineCoupling = 0.0;
    for (std::uint32_t coarse = 0u; coarse < modes; ++coarse)
        for (std::uint32_t fineNode = modes; fineNode < nodes; ++fineNode)
            for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
                const std::uint32_t fineDof = 3u * fineNode + axis;
                for (std::uint32_t rowNode = 0u; rowNode < nodes; ++rowNode) {
                    const auto p = physicalP[coarse * nodes + rowNode];
                    const double pAxis[3] = {p.x, p.y, p.z};
                    for (std::uint32_t rowAxis = 0u; rowAxis < 3u; ++rowAxis)
                        projectedQFineCoupling += std::abs(
                            pAxis[rowAxis] * physicalA[
                                (3u * rowNode + rowAxis) * physicalDofs + fineDof]);
                }
            }
    require(projectedQFineCoupling > 0.01,
        "hybrid RAP oracle did not couple coarse q modes to fine complement");

    auto matterBuffer = makeBuffer(device, std::vector{matter});
    auto blockBuffer = makeBuffer(device, std::vector{block});
    auto layoutBuffer = makeBuffer(device, std::vector{layout});
    auto basisBuffer = makeBuffer(device, basis);
    auto rowsBuffer = makeBuffer(device, globalRows);
    auto modeMapBuffer = makeBuffer(device, modeMap);
    auto nodeRangesBuffer = makeBuffer(device, nodeRanges);
    auto nodeSlotsBuffer = makeBuffer(device, nodeSlots);
    auto qBuffer = makeBuffer(device, qCoordinates);
    auto physicalBasisBuffer = makeBuffer(device, physicalP);

    auto prepare = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_human_global_coarse_hybrid_prepare_input");
    auto copy = pipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_hybrid_copy");
    auto lift = pipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_hybrid_lift");
    auto clearInactive = pipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_hybrid_clear_inactive_field");
    auto buildColumn = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_human_global_coarse_build_column");
    auto project = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_human_global_coarse_hybrid_project_rhs");
    auto factor = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_regional_factor_rap");
    auto solve = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_regional_solve_rap");
    auto add = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_human_global_coarse_hybrid_add");

    // Confirm the production hybrid input/lift gives exactly P=BQ, before
    // sending each column through the independently supplied full operator.
    auto probe = makeBuffer(device, std::vector<nm_float4>(unknownCount, nm_float4{}));
    auto lifted = makeBuffer(device, std::vector<nm_float4>(unknownCount, nm_float4{}));
    auto columnBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto coordinateCountBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{modes});
    for (std::uint32_t column = 0u; column < modes; ++column) {
        static_cast<std::uint32_t*>([columnBuffer contents])[0] = column;
        dispatch(device, queue, prepare, {
            {0u, matterBuffer}, {1u, qBuffer}, {2u, probe},
            {3u, columnBuffer}, {4u, coordinateCountBuffer}, {30u, layoutBuffer}},
            1u, 256u);
        dispatch(device, queue, copy, {
            {0u, layoutBuffer}, {1u, probe}, {2u, lifted}}, 1u, 256u);
        dispatch(device, queue, lift, {
            {0u, blockBuffer}, {1u, rowsBuffer}, {2u, basisBuffer},
            {3u, modeMapBuffer}, {4u, nodeRangesBuffer}, {5u, nodeSlotsBuffer},
            {6u, probe}, {7u, lifted}}, 1u, 256u);
        dispatch(device, queue, clearInactive, {
            {0u, blockBuffer}, {1u, lifted}}, 1u, 256u);
        const auto actual = readBuffer<nm_float4>(lifted, unknownCount);
        for (std::uint32_t row = 0u; row < nodes; ++row) {
            const auto& expected = physicalP[column * nodes + row];
            require(std::abs(actual[row].x - expected.x) < 2.0e-6f &&
                    std::abs(actual[row].y - expected.y) < 2.0e-6f &&
                    std::abs(actual[row].z - expected.z) < 2.0e-6f,
                "production hybrid lift did not construct P=BQ");
        }
    }

    // Full dense physical A actions are applied to the lifted coarse vectors.
    // The production RAP-column kernel performs the actual P^T AP contraction.
    std::vector<NMMatterStatusGPU> healthy(1u);
    healthy[0].code = NM_STATUS_SUCCESS;
    auto statusBuffer = makeBuffer(device, healthy);
    auto rapBuffer = makeBuffer(device, std::vector<float>(modes * modes, 0.0f));
    auto operatorValue = makeBuffer(device,
        std::vector<nm_float4>(unknownCount, nm_float4{}));
    std::vector<double> expectedRap(modes * modes, 0.0);
    for (std::uint32_t column = 0u; column < modes; ++column) {
        std::vector<double> p(physicalDofs, 0.0);
        for (std::uint32_t node = 0u; node < nodes; ++node) {
            const auto v = physicalP[column * nodes + node];
            p[3u * node + 0u] = v.x;
            p[3u * node + 1u] = v.y;
            p[3u * node + 2u] = v.z;
        }
        std::vector<double> ap(physicalDofs, 0.0);
        for (std::uint32_t row = 0u; row < physicalDofs; ++row)
            for (std::uint32_t k = 0u; k < physicalDofs; ++k)
                ap[row] += physicalA[row * physicalDofs + k] * p[k];
        std::vector<nm_float4> packed(unknownCount, nm_float4{});
        for (std::uint32_t node = 0u; node < nodes; ++node)
            packed[node] = {float(ap[3u * node]), float(ap[3u * node + 1u]),
                            float(ap[3u * node + 2u]), 0.0f};
        std::memcpy([operatorValue contents], packed.data(),
            packed.size() * sizeof(nm_float4));
        for (std::uint32_t row = 0u; row < modes; ++row) {
            double value = 0.0;
            for (std::uint32_t node = 0u; node < nodes; ++node) {
                const auto basisRow = physicalP[row * nodes + node];
                value += double(basisRow.x) * ap[3u * node] +
                    double(basisRow.y) * ap[3u * node + 1u] +
                    double(basisRow.z) * ap[3u * node + 2u];
            }
            expectedRap[row * modes + column] = value;
        }
        static_cast<std::uint32_t*>([columnBuffer contents])[0] = column;
        dispatch(device, queue, buildColumn, {
            {0u, matterBuffer}, {1u, physicalBasisBuffer}, {2u, operatorValue},
            {3u, rapBuffer}, {4u, coordinateCountBuffer}, {5u, columnBuffer},
            {6u, statusBuffer}}, modes, 32u);
    }
    const auto rap = readBuffer<float>(rapBuffer, modes * modes);
    for (std::uint32_t i = 0u; i < modes * modes; ++i)
        require(std::abs(double(rap[i]) - expectedRap[i]) <
                    2.0e-5 * std::max(1.0, std::abs(expectedRap[i])),
            "production RAP column disagrees with independent dense P^T A P");

    // Project a nonzero physical force through B^T and the production q
    // projector, then factor/solve E c=P^T f and compare to dense double LU.
    std::vector<double> force(physicalDofs);
    for (std::uint32_t i = 0u; i < physicalDofs; ++i)
        force[i] = 0.17 * double(int((i * 9u + 4u) % 17u) - 8);
    std::vector<nm_float4> hybridResidual(unknownCount, nm_float4{});
    for (std::uint32_t mode = 0u; mode < modes; ++mode) {
        double restricted = 0.0;
        for (std::uint32_t node = 0u; node < nodes; ++node) {
            const auto b = basis[mode * nodes + node];
            restricted += double(b.x) * force[3u * node] +
                double(b.y) * force[3u * node + 1u] +
                double(b.z) * force[3u * node + 2u];
        }
        hybridResidual[nodes + mode].x = float(restricted);
    }
    for (std::uint32_t node = modes; node < nodes; ++node)
        hybridResidual[node] = {0.3f * float(node + 1u),
            -0.2f * float(node), 0.11f, 55.0f + float(node)};
    auto residualBuffer = makeBuffer(device, hybridResidual);
    auto rhsBuffer = makeBuffer(device, std::vector<float>(modes, 0.0f));
    dispatch(device, queue, project, {
        {0u, matterBuffer}, {1u, qBuffer}, {2u, residualBuffer},
        {3u, rhsBuffer}, {4u, coordinateCountBuffer}, {5u, coordinateCountBuffer},
        {30u, layoutBuffer}}, modes, 32u);
    const auto projectedRHS = readBuffer<float>(rhsBuffer, modes);
    std::vector<double> expectedRHS(modes, 0.0);
    for (std::uint32_t coarse = 0u; coarse < modes; ++coarse)
        for (std::uint32_t node = 0u; node < nodes; ++node) {
            const auto p = physicalP[coarse * nodes + node];
            expectedRHS[coarse] += double(p.x) * force[3u * node] +
                double(p.y) * force[3u * node + 1u] +
                double(p.z) * force[3u * node + 2u];
        }
    for (std::uint32_t i = 0u; i < modes; ++i)
        require(std::abs(double(projectedRHS[i]) - expectedRHS[i]) <
                    2.0e-6 * std::max(1.0, std::abs(expectedRHS[i])),
            "hybrid production RHS projection disagrees with P^T f");

    std::vector<double> denseMatrix = expectedRap;
    std::vector<double> denseRHS = expectedRHS;
    for (std::uint32_t pivot = 0u; pivot < modes; ++pivot) {
        std::uint32_t best = pivot;
        for (std::uint32_t row = pivot + 1u; row < modes; ++row)
            if (std::abs(denseMatrix[row * modes + pivot]) >
                std::abs(denseMatrix[best * modes + pivot])) best = row;
        require(std::abs(denseMatrix[best * modes + pivot]) > 1.0e-10,
            "independent dense RAP oracle found a singular pivot");
        if (best != pivot) {
            for (std::uint32_t column = pivot; column < modes; ++column)
                std::swap(denseMatrix[pivot * modes + column],
                          denseMatrix[best * modes + column]);
            std::swap(denseRHS[pivot], denseRHS[best]);
        }
        const double diagonal = denseMatrix[pivot * modes + pivot];
        for (std::uint32_t row = pivot + 1u; row < modes; ++row) {
            const double multiplier = denseMatrix[row * modes + pivot] / diagonal;
            for (std::uint32_t column = pivot + 1u; column < modes; ++column)
                denseMatrix[row * modes + column] -=
                    multiplier * denseMatrix[pivot * modes + column];
            denseMatrix[row * modes + pivot] = multiplier;
            denseRHS[row] -= multiplier * denseRHS[pivot];
        }
    }
    std::vector<double> expectedCoefficients(modes, 0.0);
    for (int reverse = int(modes) - 1; reverse >= 0; --reverse) {
        double value = denseRHS[std::uint32_t(reverse)];
        for (std::uint32_t column = std::uint32_t(reverse) + 1u;
             column < modes; ++column)
            value -= denseMatrix[std::uint32_t(reverse) * modes + column] *
                expectedCoefficients[column];
        expectedCoefficients[std::uint32_t(reverse)] =
            value / denseMatrix[std::uint32_t(reverse) * modes +
                                std::uint32_t(reverse)];
    }

    auto pivotsBuffer = makeBuffer(device, std::vector<std::uint32_t>(modes));
    auto validBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto coefficientsBuffer = makeBuffer(device, std::vector<float>(modes, 0.0f));
    auto modeCountBuffer = makeBuffer(device, std::vector<std::uint32_t>{modes});
    dispatch(device, queue, factor, {
        {0u, matterBuffer}, {1u, rapBuffer}, {2u, pivotsBuffer},
        {3u, validBuffer}, {4u, modeCountBuffer}}, 1u, 32u);
    require(readBuffer<std::uint32_t>(validBuffer, 1u)[0] == 1u,
        "production RAP factor rejected the positive definite oracle");
    dispatch(device, queue, solve, {
        {0u, matterBuffer}, {1u, rapBuffer}, {2u, pivotsBuffer},
        {3u, validBuffer}, {4u, rhsBuffer}, {5u, coefficientsBuffer},
        {6u, modeCountBuffer}}, 1u, 32u);
    const auto coefficients = readBuffer<float>(coefficientsBuffer, modes);
    for (std::uint32_t i = 0u; i < modes; ++i)
        require(std::abs(double(coefficients[i]) - expectedCoefficients[i]) <
                    2.0e-5 * std::max(1.0, std::abs(expectedCoefficients[i])),
            "production RAP solve disagrees with independent dense coefficients");

    // Production scatter adds q-space Pc to existing local z. It must preserve
    // fine-complement xyz and every other lane; the following production lift
    // turns that q correction back into physical xyz.
    std::vector<nm_float4> preconditioned(unknownCount, nm_float4{});
    for (std::uint32_t node = 0u; node < nodes; ++node)
        preconditioned[node] = {0.02f * float(node + 1u), -0.03f, 0.04f,
                                80.0f + float(node)};
    for (std::uint32_t q = 0u; q < modes; ++q)
        preconditioned[nodes + q] = {0.0f, -0.5f, 0.75f, 90.0f + float(q)};
    auto preconditionedBuffer = makeBuffer(device, preconditioned);
    NMFGMRESStateGPU fgmresState{};
    fgmresState.diagnostics.z = 0.0f;
    auto statesBuffer = makeBuffer(device, std::vector{fgmresState});
    auto useCountBuffer = makeBuffer(device, std::vector<std::uint32_t>{0u});
    dispatch(device, queue, add, {
        {0u, matterBuffer}, {1u, qBuffer}, {2u, coefficientsBuffer},
        {3u, validBuffer}, {4u, preconditionedBuffer}, {5u, statesBuffer},
        {6u, coordinateCountBuffer}, {7u, useCountBuffer}}, 1u, 256u);
    const auto scattered = readBuffer<nm_float4>(preconditionedBuffer, unknownCount);
    for (std::uint32_t q = 0u; q < modes; ++q) {
        double expectedQ = 0.0;
        for (std::uint32_t coarse = 0u; coarse < modes; ++coarse)
            expectedQ += double(qCoordinates[coarse * modes + q]) *
                expectedCoefficients[coarse];
        require(std::abs(double(scattered[nodes + q].x) - expectedQ) < 2.0e-5 &&
                scattered[nodes + q].y == -0.5f &&
                scattered[nodes + q].z == 0.75f &&
                scattered[nodes + q].w == 90.0f + float(q),
            "production RAP q scatter changed the wrong coordinates or lanes");
    }
    for (std::uint32_t node = modes; node < nodes; ++node)
        require(std::memcmp(&scattered[node], &preconditioned[node],
                            sizeof(nm_float4)) == 0,
            "production RAP q scatter changed a fine-complement row");
    require(readBuffer<std::uint32_t>(useCountBuffer, 1u)[0] == 1u,
        "production RAP correction-use counter did not record the solve");

    auto liftedCorrectionBuffer = makeBuffer(device, preconditioned);
    dispatch(device, queue, copy, {
        {0u, layoutBuffer}, {1u, preconditionedBuffer}, {2u, liftedCorrectionBuffer}},
        1u, 256u);
    dispatch(device, queue, lift, {
        {0u, blockBuffer}, {1u, rowsBuffer}, {2u, basisBuffer},
        {3u, modeMapBuffer}, {4u, nodeRangesBuffer}, {5u, nodeSlotsBuffer},
        {6u, preconditionedBuffer}, {7u, liftedCorrectionBuffer}}, 1u, 256u);
    dispatch(device, queue, clearInactive, {
        {0u, blockBuffer}, {1u, liftedCorrectionBuffer}}, 1u, 256u);
    const auto liftedCorrection = readBuffer<nm_float4>(
        liftedCorrectionBuffer, unknownCount);
    for (std::uint32_t node = 0u; node < nodes; ++node) {
        nm_float4 expected = node < modes
            ? nm_float4{0.0f, 0.0f, 0.0f, preconditioned[node].w}
            : preconditioned[node];
        if (node < modes) {
            for (std::uint32_t coarse = 0u; coarse < modes; ++coarse) {
                const double coefficient = expectedCoefficients[coarse];
                const auto p = physicalP[coarse * nodes + node];
                expected.x += float(coefficient * p.x);
                expected.y += float(coefficient * p.y);
                expected.z += float(coefficient * p.z);
            }
        }
        require(std::abs(liftedCorrection[node].x - expected.x) < 3.0e-5f &&
                std::abs(liftedCorrection[node].y - expected.y) < 3.0e-5f &&
                std::abs(liftedCorrection[node].z - expected.z) < 3.0e-5f &&
                liftedCorrection[node].w == preconditioned[node].w,
            "production RAP lift did not add P c and preserve fine rows/w");
    }

    // Exercise the multiplicative hybrid coarse path end to end: cache the
    // full current AP images in hybrid coordinates, form r-APc over every
    // physical/q/tail slot, apply an explicit diagonal local M^{-1}, then
    // add P c with the production scatter. The dense CPU calculation is
    // independent of the GPU restriction/cache/deflation kernels.
    constexpr std::uint32_t tailCount = 4u;
    constexpr std::uint32_t hybridUnknownCount =
        2u * nodes + tailCount;
    NMFGMRESLayoutGPU hybridLayout = layout;
    hybridLayout.unknownCount = hybridUnknownCount;
    NMPartitionedReductionBlockDispatchGPU sparseDispatch = block;
    sparseDispatch.reserved = 1u;
    NMPartitionedReductionSparseRangeGPU supportRange{};
    supportRange.first = 0u;
    supportRange.count = modes;
    std::vector<std::uint32_t> supportRows(modes);
    for (std::uint32_t row = 0u; row < modes; ++row) supportRows[row] = row;
    std::vector<float> unitMass(nodes, 1.0f);
    auto sparseDispatchBuffer = makeBuffer(device, std::vector{sparseDispatch});
    auto hybridLayoutBuffer = makeBuffer(device, std::vector{hybridLayout});
    auto supportRangeBuffer = makeBuffer(device, std::vector{supportRange});
    auto supportRowsBuffer = makeBuffer(device, supportRows);
    auto unitMassBuffer = makeBuffer(device, unitMass);
    auto projectedBuffer = makeBuffer(device, std::vector<float>(modes, 0.0f));
    auto sparsePartialBuffer = makeBuffer(device, std::vector<float>(modes, 0.0f));
    auto restrictTiles = pipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_reduce_sparse_mode_tiles");
    auto restrictPartials = pipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_reduce_sparse_mode_partials");
    auto zeroSelected = pipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_hybrid_zero_selected");
    auto packModes = pipeline(device, library,
        "numi_matter_metal::nm_partitioned_reduction_hybrid_pack_modes");
    auto cacheColumn = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_human_global_coarse_cache_operator_column");
    auto deflateRHS = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_human_global_coarse_deflate_rhs");

    std::vector<std::vector<double>> oracleHybridAP(
        modes, std::vector<double>(hybridUnknownCount * 4u, 0.0));
    auto cachedHybridAP = makeBuffer(device,
        std::vector<nm_float4>(modes * hybridUnknownCount, nm_float4{}));
    auto rawOperatorOutput = makeBuffer(device,
        std::vector<nm_float4>(hybridUnknownCount, nm_float4{}));
    for (std::uint32_t coarse = 0u; coarse < modes; ++coarse) {
        std::vector<double> p(physicalDofs, 0.0);
        for (std::uint32_t node = 0u; node < nodes; ++node) {
            const nm_float4 value = physicalP[coarse * nodes + node];
            p[3u * node] = value.x;
            p[3u * node + 1u] = value.y;
            p[3u * node + 2u] = value.z;
        }
        std::vector<double> ap(physicalDofs, 0.0);
        for (std::uint32_t row = 0u; row < physicalDofs; ++row)
            for (std::uint32_t k = 0u; k < physicalDofs; ++k)
                ap[row] += physicalA[row * physicalDofs + k] * p[k];

        std::vector<nm_float4> raw(hybridUnknownCount, nm_float4{});
        for (std::uint32_t node = 0u; node < nodes; ++node) {
            raw[node] = {float(ap[3u * node]), float(ap[3u * node + 1u]),
                         float(ap[3u * node + 2u]),
                         120.0f + float(10u * coarse + node)};
            raw[nodes + node] = {91.0f, -92.0f, 93.0f, -94.0f};
        }
        for (std::uint32_t tail = 0u; tail < tailCount; ++tail) {
            double tailValue = 0.0;
            for (std::uint32_t dof = 0u; dof < physicalDofs; ++dof) {
                const int code = int((dof * 5u + tail * 3u) % 7u) - 3;
                tailValue += 0.013 * double(tail + 1u) *
                    double(code) * p[dof];
            }
            const std::uint32_t index = 2u * nodes + tail;
            raw[index] = {float(tailValue), -0.2f * float(tail + 1u),
                          0.07f * float(coarse + 1u),
                          210.0f + float(10u * coarse + tail)};
            oracleHybridAP[coarse][4u * index] = tailValue;
            oracleHybridAP[coarse][4u * index + 1u] =
                -0.2 * double(tail + 1u);
            oracleHybridAP[coarse][4u * index + 2u] =
                0.07 * double(coarse + 1u);
            oracleHybridAP[coarse][4u * index + 3u] =
                210.0 + double(10u * coarse + tail);
        }
        for (std::uint32_t mode = 0u; mode < modes; ++mode) {
            double qValue = 0.0;
            for (std::uint32_t node = 0u; node < nodes; ++node) {
                const nm_float4 b = basis[mode * nodes + node];
                qValue += double(b.x) * ap[3u * node] +
                    double(b.y) * ap[3u * node + 1u] +
                    double(b.z) * ap[3u * node + 2u];
            }
            const std::uint32_t index = nodes + mode;
            raw[index] = {91.0f, -92.0f, 93.0f, -94.0f};
            oracleHybridAP[coarse][4u * index] = qValue;
            for (std::uint32_t axis = 1u; axis < 4u; ++axis)
                oracleHybridAP[coarse][4u * index + axis] = 0.0;
        }
        for (std::uint32_t node = 0u; node < nodes; ++node) {
            const std::uint32_t index = node;
            oracleHybridAP[coarse][4u * index] =
                node < modes ? 0.0 : ap[3u * node];
            oracleHybridAP[coarse][4u * index + 1u] =
                node < modes ? 0.0 : ap[3u * node + 1u];
            oracleHybridAP[coarse][4u * index + 2u] =
                node < modes ? 0.0 : ap[3u * node + 2u];
            oracleHybridAP[coarse][4u * index + 3u] =
                120.0 + double(10u * coarse + node);
        }
        std::memcpy([rawOperatorOutput contents], raw.data(),
            raw.size() * sizeof(nm_float4));
        dispatchGrid3D(device, queue, restrictTiles, {
            {0u, sparseDispatchBuffer}, {1u, rawOperatorOutput},
            {2u, basisBuffer}, {3u, rowsBuffer}, {4u, unitMassBuffer},
            {5u, sparsePartialBuffer}, {6u, modeMapBuffer},
            {7u, supportRangeBuffer}, {8u, supportRowsBuffer}},
            1u, modes, 1u, 256u);
        dispatchGrid3D(device, queue, restrictPartials, {
            {0u, sparseDispatchBuffer}, {1u, sparsePartialBuffer},
            {2u, projectedBuffer}, {3u, modeMapBuffer}},
            1u, modes, 1u, 256u);
        dispatch(device, queue, zeroSelected, {
            {0u, sparseDispatchBuffer}, {1u, rowsBuffer},
            {2u, nodeRangesBuffer}, {3u, rawOperatorOutput}},
            nodes, 256u);
        dispatch(device, queue, packModes, {
            {0u, sparseDispatchBuffer}, {1u, projectedBuffer},
            {2u, rawOperatorOutput}}, nodes, 256u);
        const std::uint32_t coarseColumn = coarse;
        dispatch(device, queue, cacheColumn, {
            {0u, matterBuffer}, {1u, rawOperatorOutput},
            {2u, cachedHybridAP}, {3u, makeBuffer(device,
                std::vector<std::uint32_t>{coarseColumn})},
            {30u, hybridLayoutBuffer}}, hybridUnknownCount, 256u);
        const auto restricted = readBuffer<nm_float4>(
            rawOperatorOutput, hybridUnknownCount);
        for (std::uint32_t index = 0u; index < hybridUnknownCount; ++index) {
            const nm_float4 value = restricted[index];
            for (std::uint32_t axis = 0u; axis < 4u; ++axis) {
                const double got = axis == 0u ? value.x : axis == 1u ? value.y :
                    axis == 2u ? value.z : value.w;
                require(std::abs(got -
                    oracleHybridAP[coarse][4u * index + axis]) < 2.0e-5,
                    "hybrid AP restriction changed a selected/complement/tail value");
            }
        }
    }
    const auto cached = readBuffer<nm_float4>(
        cachedHybridAP, modes * hybridUnknownCount);
    for (std::uint32_t mode = 0u; mode < modes; ++mode)
        for (std::uint32_t index = 0u; index < hybridUnknownCount; ++index)
            for (std::uint32_t axis = 0u; axis < 4u; ++axis) {
                const nm_float4 value =
                    cached[mode * hybridUnknownCount + index];
                const double got = axis == 0u ? value.x : axis == 1u ? value.y :
                    axis == 2u ? value.z : value.w;
                require(std::abs(got -
                    oracleHybridAP[mode][4u * index + axis]) < 2.0e-5,
                    "cached hybrid AP column differs from full A P oracle");
            }

    std::vector<nm_float4> hybridSource(hybridUnknownCount, nm_float4{});
    for (std::uint32_t node = 0u; node < nodes; ++node) {
        hybridSource[node] = node < modes
            ? nm_float4{0.0f, 0.0f, 0.0f, 310.0f + float(node)}
            : hybridResidual[node];
    }
    for (std::uint32_t q = 0u; q < nodes; ++q) {
        hybridSource[nodes + q] = q < modes
            ? hybridResidual[nodes + q]
            : nm_float4{0.0f, 0.0f, 0.0f, 410.0f + float(q)};
    }
    for (std::uint32_t tail = 0u; tail < tailCount; ++tail) {
        const std::uint32_t index = 2u * nodes + tail;
        hybridSource[index] = {0.19f * float(tail + 1u),
            -0.07f * float(tail + 2u), 0.03f * float(tail + 3u),
            510.0f + float(tail)};
    }
    auto hybridSourceBuffer = makeBuffer(device, hybridSource);
    auto deflatedBuffer = makeBuffer(device,
        std::vector<nm_float4>(hybridUnknownCount, nm_float4{}));
    dispatch(device, queue, deflateRHS, {
        {0u, matterBuffer}, {1u, hybridSourceBuffer},
        {2u, cachedHybridAP}, {3u, coefficientsBuffer}, {4u, validBuffer},
        {5u, statesBuffer}, {6u, deflatedBuffer}, {30u, hybridLayoutBuffer}},
        hybridUnknownCount, 256u);
    const auto deflated = readBuffer<nm_float4>(
        deflatedBuffer, hybridUnknownCount);
    std::vector<nm_float4> oracleDeflated(hybridUnknownCount);
    for (std::uint32_t index = 0u; index < hybridUnknownCount; ++index) {
        oracleDeflated[index] = hybridSource[index];
        for (std::uint32_t mode = 0u; mode < modes; ++mode) {
            // Use the production float coefficient after the independent
            // dense RAP solve comparison above. This isolates deflate algebra
            // from the separately bounded float-LU coefficient roundoff.
            const float c = coefficients[mode];
            oracleDeflated[index].x -= c * float(oracleHybridAP[mode][4u * index]);
            oracleDeflated[index].y -= c * float(oracleHybridAP[mode][4u * index + 1u]);
            oracleDeflated[index].z -= c * float(oracleHybridAP[mode][4u * index + 2u]);
            oracleDeflated[index].w -= c * float(oracleHybridAP[mode][4u * index + 3u]);
        }
        const float deflateError[4] = {
            std::abs(deflated[index].x - oracleDeflated[index].x),
            std::abs(deflated[index].y - oracleDeflated[index].y),
            std::abs(deflated[index].z - oracleDeflated[index].z),
            std::abs(deflated[index].w - oracleDeflated[index].w)};
        const float deflateExpected[4] = {oracleDeflated[index].x,
            oracleDeflated[index].y, oracleDeflated[index].z,
            oracleDeflated[index].w};
        const float deflateGot[4] = {deflated[index].x, deflated[index].y,
            deflated[index].z, deflated[index].w};
        for (std::uint32_t axis = 0u; axis < 4u; ++axis)
            if (deflateError[axis] >= 3.0e-5f) {
                std::cerr << "hybrid_deflate_witness index=" << index
                    << " lane=" << axis << " got=" << deflateGot[axis]
                    << " expected=" << deflateExpected[axis]
                    << " abs_error=" << deflateError[axis] << std::endl;
                break;
            }
        require(deflateError[0] < 3.0e-5f && deflateError[1] < 3.0e-5f &&
                deflateError[2] < 3.0e-5f && deflateError[3] < 3.0e-5f,
            "production multiplicative deflation disagrees with r-APc");
    }

    // The local M^{-1} here is a deliberately explicit diagonal oracle. The
    // production coarse add then composes P c on q only; the other physical
    // fine rows, pressure/tail-like lanes, and w sentinels must stay local.
    std::vector<nm_float4> localZ(hybridUnknownCount);
    std::vector<nm_float4> oracleLocalZ(hybridUnknownCount);
    for (std::uint32_t index = 0u; index < hybridUnknownCount; ++index) {
        const float scale = 0.4f + 0.015f * float(index);
        localZ[index] = {scale * deflated[index].x,
            (scale + 0.03f) * deflated[index].y,
            (scale + 0.06f) * deflated[index].z,
            (scale + 0.09f) * deflated[index].w};
        oracleLocalZ[index] = {scale * oracleDeflated[index].x,
            (scale + 0.03f) * oracleDeflated[index].y,
            (scale + 0.06f) * oracleDeflated[index].z,
            (scale + 0.09f) * oracleDeflated[index].w};
        require(std::abs(localZ[index].x - oracleLocalZ[index].x) < 3.0e-5f &&
                std::abs(localZ[index].y - oracleLocalZ[index].y) < 3.0e-5f &&
                std::abs(localZ[index].z - oracleLocalZ[index].z) < 3.0e-5f &&
                std::abs(localZ[index].w - oracleLocalZ[index].w) < 3.0e-5f,
            "diagonal local M oracle disagrees with r-APc");
    }
    auto localZBuffer = makeBuffer(device, localZ);
    auto coarseUseCountBuffer =
        makeBuffer(device, std::vector<std::uint32_t>{0u});
    dispatch(device, queue, add, {
        {0u, matterBuffer}, {1u, qBuffer}, {2u, coefficientsBuffer},
        {3u, validBuffer}, {4u, localZBuffer}, {5u, statesBuffer},
        {6u, coordinateCountBuffer}, {7u, coarseUseCountBuffer}}, modes, 32u);
    const auto composed = readBuffer<nm_float4>(
        localZBuffer, hybridUnknownCount);
    for (std::uint32_t index = 0u; index < hybridUnknownCount; ++index) {
        nm_float4 expected = oracleLocalZ[index];
        if (index >= nodes && index < nodes + modes) {
            const std::uint32_t q = index - nodes;
            for (std::uint32_t coarse = 0u; coarse < modes; ++coarse)
                expected.x += qCoordinates[coarse * modes + q] *
                    float(expectedCoefficients[coarse]);
        }
        require(std::abs(composed[index].x - expected.x) < 3.0e-5f &&
                std::abs(composed[index].y - expected.y) < 3.0e-5f &&
                std::abs(composed[index].z - expected.z) < 3.0e-5f &&
                std::abs(composed[index].w - expected.w) < 3.0e-5f,
            "multiplicative localM plus Pc disagrees with independent oracle");
    }
    require(readBuffer<std::uint32_t>(coarseUseCountBuffer, 1u)[0] == 1u,
        "multiplicative coarse correction was not counted");

    // Invalid coarse factor and a converged/no-op state must preserve exact
    // source values and must not add a stale coarse correction.
    auto invalidCoarseBuffer =
        makeBuffer(device, std::vector<std::uint32_t>{0u});
    auto noOpOutput = makeBuffer(device,
        std::vector<nm_float4>(hybridUnknownCount, nm_float4{}));
    dispatch(device, queue, deflateRHS, {
        {0u, matterBuffer}, {1u, hybridSourceBuffer},
        {2u, cachedHybridAP}, {3u, coefficientsBuffer},
        {4u, invalidCoarseBuffer}, {5u, statesBuffer},
        {6u, noOpOutput}, {30u, hybridLayoutBuffer}},
        hybridUnknownCount, 256u);
    auto passThrough = readBuffer<nm_float4>(noOpOutput, hybridUnknownCount);
    for (std::uint32_t i = 0u; i < hybridUnknownCount; ++i)
        require(std::memcmp(&passThrough[i], &hybridSource[i],
                            sizeof(nm_float4)) == 0,
            "invalid coarse factor did not preserve the input vector");
    auto invalidAddOutput = makeBuffer(device, localZ);
    auto invalidAddUseCount = makeBuffer(device, std::vector<std::uint32_t>{0u});
    dispatch(device, queue, add, {
        {0u, matterBuffer}, {1u, qBuffer}, {2u, coefficientsBuffer},
        {3u, invalidCoarseBuffer}, {4u, invalidAddOutput}, {5u, statesBuffer},
        {6u, coordinateCountBuffer}, {7u, invalidAddUseCount}}, modes, 32u);
    const auto invalidAdd = readBuffer<nm_float4>(
        invalidAddOutput, hybridUnknownCount);
    for (std::uint32_t i = 0u; i < hybridUnknownCount; ++i)
        require(std::memcmp(&invalidAdd[i], &localZ[i], sizeof(nm_float4)) == 0,
            "invalid coarse factor added a stale correction");
    require(readBuffer<std::uint32_t>(invalidAddUseCount, 1u)[0] == 0u,
        "invalid coarse factor incremented the use counter");
    std::vector<NMFGMRESStateGPU> noOpStates(1u);
    noOpStates[0].diagnostics.z = 1.0f;
    auto noOpStateBuffer = makeBuffer(device, noOpStates);
    dispatch(device, queue, deflateRHS, {
        {0u, matterBuffer}, {1u, hybridSourceBuffer},
        {2u, cachedHybridAP}, {3u, coefficientsBuffer},
        {4u, validBuffer}, {5u, noOpStateBuffer},
        {6u, noOpOutput}, {30u, hybridLayoutBuffer}},
        hybridUnknownCount, 256u);
    passThrough = readBuffer<nm_float4>(noOpOutput, hybridUnknownCount);
    for (std::uint32_t i = 0u; i < hybridUnknownCount; ++i)
        require(std::memcmp(&passThrough[i], &hybridSource[i],
                            sizeof(nm_float4)) == 0,
            "converged-state deflation was not a byte-exact no-op");
    std::vector<nm_float4> noOpZ = localZ;
    auto noOpZBuffer = makeBuffer(device, noOpZ);
    dispatch(device, queue, add, {
        {0u, matterBuffer}, {1u, qBuffer}, {2u, coefficientsBuffer},
        {3u, validBuffer}, {4u, noOpZBuffer}, {5u, noOpStateBuffer},
        {6u, coordinateCountBuffer}, {7u, coarseUseCountBuffer}}, modes, 32u);
    const auto noOpAdded = readBuffer<nm_float4>(noOpZBuffer, hybridUnknownCount);
    for (std::uint32_t i = 0u; i < hybridUnknownCount; ++i)
        require(std::memcmp(&noOpAdded[i], &noOpZ[i], sizeof(nm_float4)) == 0,
            "converged-state coarse add changed the local preconditioner result");
    std::cout << "hybrid_multiplicative_coarse=true AP=full-hybrid "
              << "deflate=PASS localM_plus_Pc=PASS invalid=noop "
              << "tail=preserved result=PASS" << std::endl;

    std::cout << "hybrid_global_coarse_rap=true modes=6 q_fine_coupling="
              << qFineCoupling << " projected_q_fine="
              << projectedQFineCoupling
              << " lift=PASS RAP=PASS factor=PASS solve=PASS scatter=PASS "
                 "fine_complement=PASS result=PASS" << std::endl;
}



void checkLeftRowEquilibration(
    id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library) {
    auto buildScale = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_build_left_row_equilibration_scale");
    auto applyScale = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_apply_left_row_equilibration");

    NMMatterDispatchGPU dispatchShape{};
    dispatchShape.abiVersion = NM_MATTER_ABI_VERSION;
    dispatchShape.environmentCount = 1u;
    dispatchShape.femNodeCount = 3u;
    NMFGMRESLayoutGPU layout{};
    layout.unknownCount = 6u; // three FEM float4 rows plus three identity tails

    std::vector<nm_float4> inverseBlocks(9u, nm_float4{});
    const float inverseDiagonal[3] = {50.0f, 10.0f, 0.1f};
    for (std::uint32_t node = 0u; node < 3u; ++node) {
        for (std::uint32_t row = 0u; row < 3u; ++row) {
            inverseBlocks[node * 3u + row] = nm_float4{
                row == 0u ? inverseDiagonal[node] : 0.0f,
                row == 1u ? inverseDiagonal[node] : 0.0f,
                row == 2u ? inverseDiagonal[node] : 0.0f,
                row == 0u ? 1.0f : 0.0f};
        }
    }
    auto dispatchBuffer = makeBuffer(device, std::vector{dispatchShape});
    auto layoutBuffer = makeBuffer(device, std::vector{layout});
    auto inverseBlockBuffer = makeBuffer(device, inverseBlocks);
    auto scaleBuffer = makeBuffer(device, std::vector<float>(3u, 0.0f));
    dispatch(device, queue, buildScale,
        {{0u, dispatchBuffer}, {1u, inverseBlockBuffer}, {2u, scaleBuffer}},
        1u, 32u);
    const auto scales = readBuffer<float>(scaleBuffer, 3u);
    const double expectedScales[3] = {50.0, 10.0, 0.1};
    for (std::uint32_t node = 0u; node < 3u; ++node)
        require(std::abs(double(scales[node]) - expectedScales[node]) <
                1.0e-5 * expectedScales[node],
            "row scale did not recover the isotropic inverse-block magnitude");

    std::vector<nm_float4> source{
        {1.25f, -2.5f, 3.75f, 11.0f},
        {-4.5f, 5.25f, -6.0f, 12.0f},
        {7.0f, -8.0f, 9.0f, 13.0f},
        {0.125f, -0.25f, 0.5f, 17.0f},
        {-1.0f, 2.0f, -3.0f, 19.0f},
        {4.0f, -5.0f, 6.0f, 23.0f}};
    auto sourceBuffer = makeBuffer(device, source);
    auto scaledBuffer = makeBuffer(device, std::vector<nm_float4>(source.size()));
    dispatch(device, queue, applyScale,
        {{0u, dispatchBuffer}, {30u, layoutBuffer}, {1u, scaleBuffer},
         {2u, sourceBuffer}, {3u, scaledBuffer}}, source.size(), 32u);
    const auto scaled = readBuffer<nm_float4>(scaledBuffer, source.size());
    for (std::uint32_t node = 0u; node < 3u; ++node) {
        for (std::uint32_t component = 0u; component < 3u; ++component) {
            const float value = component == 0u ? source[node].x :
                (component == 1u ? source[node].y : source[node].z);
            const float scaledValue = component == 0u ? scaled[node].x :
                (component == 1u ? scaled[node].y : scaled[node].z);
            require(std::abs(scaledValue - value * scales[node]) <=
                    2.0e-6f * std::max(1.0f, std::abs(scaledValue)),
                "native row scaling changed the wrong value");
        }
        require(scaled[node].w == source[node].w,
            "row scaling modified FEM float4 padding");
    }
    for (std::uint32_t row = 3u; row < 6u; ++row)
        require(std::memcmp(&scaled[row], &source[row], sizeof(nm_float4)) == 0,
            "row scaling modified an identity tail row");

    // Dense nonsymmetric oracle for mode 1: solve S*A*M^-1 u=S*b,
    // then recover the physical correction x=M^-1*u.
    constexpr std::size_t n = 12u; // nine mechanical coordinates + three tails
    std::vector<double> rowS(n, 1.0);
    for (std::size_t node = 0; node < 3u; ++node)
        for (std::size_t component = 0; component < 3u; ++component)
            rowS[node * 3u + component] = scales[node];
    std::vector<double> a(n * n, 0.0), minv(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const double scaledA = i == j
                ? 2.0 + 0.05 * double(i)
                : 0.006 * double((i + 1u) * (j + 2u) % 7u + 1u) /
                    double(j + 1u);
            a[i * n + j] = scaledA / rowS[i];
            minv[i * n + j] = i == j
                ? 0.8 + 0.025 * double(i)
                : 0.004 * double((i + 3u) * (j + 1u) % 5u + 1u) /
                    double(j + 2u);
        }
    }
    const auto multiply = [](const std::vector<double>& lhs,
        const std::vector<double>& rhs) {
        const std::size_t n = static_cast<std::size_t>(std::sqrt(lhs.size()));
        std::vector<double> out(n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t k = 0; k < n; ++k)
                for (std::size_t j = 0; j < n; ++j)
                    out[i * n + j] += lhs[i * n + k] * rhs[k * n + j];
        return out;
    };
    const auto solve = [](std::vector<double> matrix,
        std::vector<double> rhs) {
        const std::size_t n = rhs.size();
        for (std::size_t pivot = 0; pivot < n; ++pivot) {
            std::size_t best = pivot;
            for (std::size_t row = pivot + 1u; row < n; ++row)
                if (std::abs(matrix[row * n + pivot]) >
                    std::abs(matrix[best * n + pivot])) best = row;
            if (!(std::abs(matrix[best * n + pivot]) > 1.0e-14))
                throw std::runtime_error("left scaling dense oracle is singular");
            if (best != pivot) {
                for (std::size_t column = pivot; column < n; ++column)
                    std::swap(matrix[pivot * n + column],
                              matrix[best * n + column]);
                std::swap(rhs[pivot], rhs[best]);
            }
            const double diagonal = matrix[pivot * n + pivot];
            for (std::size_t column = pivot; column < n; ++column)
                matrix[pivot * n + column] /= diagonal;
            rhs[pivot] /= diagonal;
            for (std::size_t row = 0; row < n; ++row) {
                if (row == pivot) continue;
                const double factor = matrix[row * n + pivot];
                for (std::size_t column = pivot; column < n; ++column)
                    matrix[row * n + column] -=
                        factor * matrix[pivot * n + column];
                rhs[row] -= factor * rhs[pivot];
            }
        }
        return rhs;
    };
    const std::vector<double> xExpected{
        0.2, -0.4, 0.7, -0.1, 0.3, -0.8,
        0.9, -0.2, 0.5, 1.1, -0.6, 0.4};
    std::vector<double> b(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            b[i] += a[i * n + j] * xExpected[j];
    const auto am = multiply(a, minv);
    std::vector<double> scaledRHS(n);
    for (std::size_t i = 0; i < n; ++i) scaledRHS[i] = rowS[i] * b[i];
    auto mode1 = am;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            mode1[i * n + j] *= rowS[i];
    const auto u1 = solve(mode1, scaledRHS);
    std::vector<double> x1(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            x1[i] += minv[i * n + j] * u1[j];

    double maxSolutionDifference = 0.0, maxPhysicalResidual = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        maxSolutionDifference = std::max(maxSolutionDifference,
                                         std::abs(x1[i] - xExpected[i]));
        double residual = -b[i];
        for (std::size_t j = 0; j < n; ++j)
            residual += a[i * n + j] * x1[j];
        maxPhysicalResidual = std::max(maxPhysicalResidual,
                                       std::abs(residual));
    }
    require(maxSolutionDifference < 2.0e-10,
        "equilibrated coordinate variants did not recover the same physical solution");
    require(maxPhysicalResidual < 2.0e-10,
        "equilibrated coordinate variant failed the raw physical residual");

    // Arnoldi residuals are measured after S, so the inner target must use
    // ||S*b||. diagnostics.y remains the raw ||b|| for the physical certificate.
    NMMatterDispatchGPU targetDispatch{};
    targetDispatch.abiVersion = NM_MATTER_ABI_VERSION;
    targetDispatch.environmentCount = 1u;
    targetDispatch.femNodeCount = 1u;
    NMFGMRESLayoutGPU targetLayout{};
    targetLayout.unknownCount = 2u;
    targetLayout.reserved = 1u << 23u;
    NMMixedSolverGPU targetSolver{};
    targetSolver.nonlinearIterations.y = NM_MIXED_FGMRES_RESTART;
    targetSolver.residualTolerances.x = 0.1f;
    targetSolver.regularization.x = 1.0e-12f;
    NMFGMRESStateGPU targetState{};
    targetState.diagnostics = {0.01f, 1.0f, 0.0f, 1.0f};
    targetState.nonlinear = {1.0f, 1.0f, 0.0f, 0.0f};
    targetState.rowEquilibration = {0.001f, 0.0f, 0.0f, 0.0f};
    NMMatterStatusGPU targetStatus{};
    targetStatus.code = NM_STATUS_SUCCESS;
    std::vector<float> targetHessenberg(
        (NM_MIXED_FGMRES_RESTART + 1u) * NM_MIXED_FGMRES_RESTART, 0.0f);
    targetHessenberg[0] = 1.0f;
    std::vector<float> targetLeastSquares(NM_MIXED_FGMRES_RESTART + 1u, 0.0f);
    std::vector<NativeFloat2> targetRotations(
        NM_MIXED_FGMRES_RESTART, NativeFloat2{1.0f, 0.0f});
    std::vector<float> targetRestartCoefficients(
        NM_MIXED_FGMRES_RESTART + 1u, 0.0f);
    const std::uint32_t zeroIterationOffset = 0u;
    const std::uint32_t finalCycle = 1u;
    const std::uint32_t doNotDefer = 0u;
    const float targetForcing = 0.1f;
    auto targetDispatchBuffer = makeBuffer(device,
        std::vector<NMMatterDispatchGPU>{targetDispatch});
    auto targetLayoutBuffer = makeBuffer(device,
        std::vector<NMFGMRESLayoutGPU>{targetLayout});
    auto targetSolverBuffer = makeBuffer(device,
        std::vector<NMMixedSolverGPU>{targetSolver});
    auto targetHessenbergBuffer = makeBuffer(device, targetHessenberg);
    auto targetLeastSquaresBuffer = makeBuffer(device, targetLeastSquares);
    auto targetRotationsBuffer = makeBuffer(device, targetRotations);
    auto targetRestartCoefficientsBuffer = makeBuffer(device,
        targetRestartCoefficients);
    auto targetStateBuffer = makeBuffer(device,
        std::vector<NMFGMRESStateGPU>{targetState});
    auto targetStatusBuffer = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{targetStatus});
    auto targetFinalizer = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_finalize_cycle");
    const auto runTargetFinalizer = [&] {
        dispatch(device, queue, targetFinalizer, {
            {0u, targetDispatchBuffer}, {30u, targetLayoutBuffer},
            {1u, targetSolverBuffer}, {2u, targetHessenbergBuffer},
            {3u, targetLeastSquaresBuffer}, {4u, targetStateBuffer},
            {5u, targetStatusBuffer},
            {6u, makeBuffer(device,
                std::vector<std::uint32_t>{zeroIterationOffset})},
            {7u, makeBuffer(device, std::vector<std::uint32_t>{finalCycle})},
            {8u, makeBuffer(device, std::vector<float>{targetForcing})},
            {9u, targetRotationsBuffer},
            {10u, targetRestartCoefficientsBuffer},
            {11u, makeBuffer(device, std::vector<std::uint32_t>{doNotDefer})}},
            1u, 32u);
    };
    runTargetFinalizer();
    auto targetStateAfter = readBuffer<NMFGMRESStateGPU>(
        targetStateBuffer, 1u)[0];
    require(targetStateAfter.diagnostics.z == 0.0f &&
            targetStateAfter.diagnostics.y == 1.0f &&
            targetStateAfter.rowEquilibration.x == 0.001f,
        "Arnoldi target compared S*r with raw ||b||");
    targetStateAfter.diagnostics.x = 0.00005f;
    targetStateAfter.diagnostics.z = 0.0f;
    std::memcpy(targetStateBuffer.contents, &targetStateAfter,
        sizeof(targetStateAfter));
    runTargetFinalizer();
    targetStateAfter = readBuffer<NMFGMRESStateGPU>(
        targetStateBuffer, 1u)[0];
    require(targetStateAfter.diagnostics.z == 1.0f &&
            targetStateAfter.diagnostics.y == 1.0f,
        "scaled Arnoldi target did not accept a residual below eta*||S*b||");

    // The physical true-residual certificate deliberately keeps using raw
    // diagnostics.y, even when the separately stored scaled reference is much
    // smaller. A raw residual of 0.05 is below eta*||b||=0.1, but not below
    // eta*||S*b||=0.0001.
    targetLayout.reserved = 1u << 23u;
    auto certificateLayoutBuffer = makeBuffer(device,
        std::vector<NMFGMRESLayoutGPU>{targetLayout});
    std::vector<nm_float4> certificateRHS{
        {0.05f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f}};
    std::vector<nm_float4> certificateOperator(2u, nm_float4{});
    auto certificateRHSBuffer = makeBuffer(device, certificateRHS);
    auto certificateOperatorBuffer = makeBuffer(device, certificateOperator);
    auto certificateResidualBuffer = makeBuffer(device,
        std::vector<nm_float4>(2u, nm_float4{}));
    auto certificateMetricsBuffer = makeBuffer(device,
        std::vector<nm_float4>(1u, nm_float4{}));
    auto certificateDiagnosticStatus = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{targetStatus});
    auto certificateProductionStatus = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{targetStatus});
    auto certificateReplaceBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{3u});
    auto certificatePropagateBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u});
    auto rawCertificate = pipeline(device, library,
        "numi_matter_metal::nm_fgmres_measure_true_residual");
    dispatch(device, queue, rawCertificate, {
        {0u, targetDispatchBuffer}, {30u, certificateLayoutBuffer},
        {1u, targetSolverBuffer}, {2u, certificateRHSBuffer},
        {3u, certificateOperatorBuffer}, {4u, certificateResidualBuffer},
        {5u, targetStateBuffer}, {6u, certificateMetricsBuffer},
        {7u, makeBuffer(device, std::vector<float>{targetForcing})},
        {8u, certificateDiagnosticStatus},
        {9u, certificateReplaceBuffer},
        {10u, certificateProductionStatus},
        {11u, certificatePropagateBuffer}}, 1u, 32u);
    targetStateAfter = readBuffer<NMFGMRESStateGPU>(
        targetStateBuffer, 1u)[0];
    const auto rawCertificateMetrics = readBuffer<nm_float4>(
        certificateMetricsBuffer, 1u)[0];
    require(targetStateAfter.diagnostics.z == 1.0f &&
            std::abs(rawCertificateMetrics.x - 0.05f) < 1.0e-7f &&
            targetStateAfter.diagnostics.y == 1.0f,
        "physical true-residual certificate used the scaled Arnoldi reference");

    std::cout << "left_row_scale_native=true equation=SA_Minv raw_physical_residual="
              << maxPhysicalResidual << " arnoldi_reference=scaled raw_certificate=preserved result=PASS\n";
}


void checkGraphTetBlockReuse(
    id<MTLDevice> device, id<MTLCommandQueue> queue,
    id<MTLLibrary> library) {
    NMMatterDispatchGPU shape{};
    shape.abiVersion = NM_MATTER_ABI_VERSION;
    shape.environmentCount = 1u;
    shape.objectCount = 1u;
    shape.materialCount = 1u;
    shape.parameterCount = 1u;
    shape.femNodeCount = 4u;
    shape.tetrahedronCount = 1u;
    shape.gravityAndTimestep.w = 0.125f;

    NMContinuumObjectGPU object{};
    object.representation = NM_REPRESENTATION_FEM;
    object.flags = NM_OBJECT_ACTIVE;
    object.solver.y = 0u;
    NMTetrahedronGPU tet{};
    tet.identity = nm_uint4{0u, 0u, 0u, NM_OBJECT_ACTIVE};
    tet.nodes = nm_uint4{0u, 1u, 2u, 3u};

    std::vector<NMFEMNodeStateGPU> nodes(4u);
    const std::array<float, 4u> masses{1.5f, 2.0f, 2.5f, 3.0f};
    for (std::uint32_t node = 0u; node < 4u; ++node) {
        nodes[node].positionAndMass.w = masses[node];
        nodes[node].velocityAndInverseMass.w = 1.0f / masses[node];
        nodes[node].restAndFixed.w = 0.0f;
    }
    std::vector<NMIncidenceRangeGPU> nodeRanges;
    for (std::uint32_t node = 0u; node < 4u; ++node)
        nodeRanges.push_back(NMIncidenceRangeGPU{node, 1u, 0u, 0u});
    const std::vector<std::uint32_t> incidence{0u, 0u, 0u, 0u};
    NMSchedulerStateGPU scheduler{};
    scheduler.numerical.w = 1.0f;
    NMAdaptiveStateGPU adaptive{};
    adaptive.activeRepresentation = NM_REPRESENTATION_FEM;
    NMMicrostepGPU micro{};
    micro.microtick = 0u;

    const auto dispatchBuffer = makeBuffer(device,
        std::vector<NMMatterDispatchGPU>{shape});
    const auto objectsBuffer = makeBuffer(device,
        std::vector<NMContinuumObjectGPU>{object});
    const auto candidateBuffer = makeBuffer(device, nodes);
    const auto tetBuffer = makeBuffer(device,
        std::vector<NMTetrahedronGPU>{tet});
    const auto incidenceBuffer = makeBuffer(device, incidence);
    const auto rangesBuffer = makeBuffer(device, nodeRanges);
    const auto schedulerBuffer = makeBuffer(device,
        std::vector<NMSchedulerStateGPU>{scheduler});
    const auto adaptiveBuffer = makeBuffer(device,
        std::vector<NMAdaptiveStateGPU>{adaptive});
    const auto materialBuffer = makeBuffer(device,
        std::vector<NMMaterialGPU>{NMMaterialGPU{}});
    const auto programBuffer = makeBuffer(device,
        std::vector<NMScalarProgramGPU>{NMScalarProgramGPU{}});
    const auto instructionBuffer = makeBuffer(device,
        std::vector<NMExpressionInstructionGPU>{NMExpressionInstructionGPU{}});
    const auto parameterBuffer = makeBuffer(device, std::vector<float>{0.0f});
    const auto stateBuffer = makeBuffer(device, std::vector<float>{0.0f});
    const auto tangentBuffer = makeBuffer(device, std::vector<float>(81u));
    const auto tangentValid = makeBuffer(device,
        std::vector<std::uint32_t>{1u});
    std::vector<float> elementBlocks(144u);
    for (std::uint32_t row = 0u; row < 12u; ++row)
        for (std::uint32_t column = 0u; column < 12u; ++column)
            elementBlocks[row * 12u + column] =
                0.015625f * float(1u + row * 12u + column);
    const auto elementBlocksBuffer = makeBuffer(device, elementBlocks);
    const auto patchesBuffer = makeBuffer(device,
        std::vector<nm_uint4>{nm_uint4{0u, 4u, 0u, 0u}});
    const auto patchNodesBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u, 1u, 2u, 3u});
    std::vector<nm_uint4> rows;
    rows.reserve(12u);
    for (std::uint32_t node = 0u; node < 4u; ++node)
        for (std::uint32_t axis = 0u; axis < 3u; ++axis)
            rows.push_back(nm_uint4{0u, node, axis, node});
    const auto rowsBuffer = makeBuffer(device, rows);
    const auto coverageBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{1u, 1u, 1u, 1u});
    const auto statesBuffer = makeBuffer(device,
        std::vector<NMFGMRESStateGPU>{NMFGMRESStateGPU{}});
    const auto priorValidBuffer = makeBuffer(device,
        std::vector<std::uint32_t>{0u});
    const auto graphBuild = pipelineWithRelativeInnerForcingDisabled(
        device, library,
        "numi_matter_metal::nm_fgmres_graph_build_patch_matrices");
    auto matrix = makeBuffer(device, std::vector<float>(144u));
    auto status = makeBuffer(device,
        std::vector<NMMatterStatusGPU>{NMMatterStatusGPU{}});
    const auto u32 = [&](const std::uint32_t value) {
        return makeBuffer(device, std::vector<std::uint32_t>{value});
    };
    const std::uint32_t useCachedBlocks = 3u;
    const std::uint32_t rowCount = 12u;
    const std::uint32_t patchCount = 1u;
    const std::uint32_t matrixFloats = 144u;
    const std::uint32_t reuseTail = 0u;
    const std::uint32_t dueAwareRows = 0u;
    const auto microBuffer = makeBuffer(device,
        std::vector<NMMicrostepGPU>{micro});
    dispatch(device, queue, graphBuild, {
        {0u, dispatchBuffer}, {1u, objectsBuffer},
        {2u, candidateBuffer}, {3u, tetBuffer}, {4u, incidenceBuffer},
        {5u, rangesBuffer}, {6u, schedulerBuffer},
        {7u, materialBuffer}, {8u, programBuffer},
        {9u, instructionBuffer}, {10u, parameterBuffer},
        {11u, stateBuffer}, {12u, tangentBuffer}, {13u, tangentValid},
        {14u, u32(useCachedBlocks)}, {15u, patchesBuffer},
        {16u, patchNodesBuffer}, {17u, rowsBuffer},
        {18u, coverageBuffer}, {19u, matrix}, {20u, status},
        {21u, u32(rowCount)}, {22u, u32(patchCount)},
        {23u, u32(matrixFloats)}, {24u, statesBuffer},
        {25u, u32(reuseTail)}, {26u, priorValidBuffer},
        {27u, adaptiveBuffer}, {28u, microBuffer},
        {29u, u32(dueAwareRows)}, {30u, elementBlocksBuffer}},
        1u, rowCount);
    const auto actual = readBuffer<float>(matrix, matrixFloats);
    require(readBuffer<NMMatterStatusGPU>(status, 1u)[0].code ==
                NM_STATUS_SUCCESS,
        "graph tet-block reuse kernel reported a failure");
    const float dt = shape.gravityAndTimestep.w;
    for (std::uint32_t rowNode = 0u; rowNode < 4u; ++rowNode) {
        for (std::uint32_t rowAxis = 0u; rowAxis < 3u; ++rowAxis) {
            const std::uint32_t rowDof = rowNode * 3u + rowAxis;
            for (std::uint32_t columnNode = 0u; columnNode < 4u;
                 ++columnNode) {
                for (std::uint32_t columnAxis = 0u; columnAxis < 3u;
                     ++columnAxis) {
                    const std::uint32_t columnDof =
                        columnNode * 3u + columnAxis;
                    const std::uint32_t index = rowDof * 12u + columnDof;
                    float expected = -dt * dt *
                        elementBlocks[rowDof * 12u + columnDof];
                    if (rowDof == columnDof) expected += masses[rowNode];
                    require(std::abs(actual[index] - expected) <= 1.0e-7f,
                        "graph builder did not reuse the matching cached element block");
                }
            }
        }
    }
    std::cout << "graph_tet_block_reuse=true rows=12 tet=1 result=PASS\n";
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
            if (argc == 2 && std::string(argv[1]) == "--left-row-equilibration-only") {
                checkLeftRowEquilibration(device, queue, library);
                std::cout << "PASS left row equilibration " << checks << " checks" << std::endl;
                return 0;
            }
            if (argc == 2 && std::string(argv[1]) == "--selector6-prefix-icb-only") {
                checkSelector6PrefixICB(device, queue, library);
                std::cout << "PASS selector-6 prefix ICB " << checks << " checks" << std::endl;
                return 0;
            }
            if (argc == 2 && std::string(argv[1]) == "--selector6-vascular-prefix-icb-only") {
                checkSelector6VascularPrefixICB(device, queue, library);
                std::cout << "PASS selector-6 vascular prefix ICB " << checks << " checks" << std::endl;
                return 0;
            }
            if (argc == 2 && std::string(argv[1]) == "--assembled-xyz-only") {
                checkAssembledXYZOperator(device, queue, library);
                std::cout << "PASS assembled xyz operator " << checks << " checks" << std::endl;
                return 0;
            }
            if (argc == 2 && std::string(argv[1]) == "--graph-tet-block-reuse-only") {
                checkGraphTetBlockReuse(device, queue, library);
                std::cout << "PASS graph tet block reuse " << checks << " checks" << std::endl;
                return 0;
            }
            if (argc == 2 && std::string(argv[1]) == "--hybrid-rap-only") {
                checkHybridGlobalCoarseRAPOracle(device, queue, library);
                std::cout << "PASS hybrid global coarse RAP " << checks << " checks" << std::endl;
                return 0;
            }
            if (argc == 2 && std::string(argv[1]) == "--hybrid-gs-only") {
                checkHybridBlockGaussSeidel(device, queue, library);
                std::cout << "PASS hybrid block GS " << checks << " checks" << std::endl;
                return 0;
            }
            if (argc == 2 && std::string(argv[1]) == "--schur-interface-only") {
                checkSchurGraphInterfaceAction(device, queue, library);
                std::cout << "PASS raw Schur interface action "
                          << checks << " checks" << std::endl;
                return 0;
            }
            require(argc == 1, "unknown native-check option; use a declared focused checker or no arguments");
            checkAcceptedSubspacePredictor(device, queue, library);
            checkHybridBlockGaussSeidel(device, queue, library);
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
            checkParallelGraphPatchTriangularSolve(device, queue, library);
            checkRegionalSchurTransformsAndRecovery(device, queue, library);
            checkDueAwareSchurPatchAssembly(device, queue, library);
            checkActiveContactSchurEligibility(device, queue, library);
            checkSchurGraphInterfaceAction(device, queue, library);
            auto operatorICBRanges = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_prepare_operator_icb_ranges");
            checkOperatorICBRanges(device, queue, operatorICBRanges);
            checkSelector6PrefixICB(device, queue, library);
            checkSelector6VascularPrefixICB(device, queue, library);

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
