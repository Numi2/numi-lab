#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/shared.h"

#include <cmath>
#include <algorithm>
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

void checkDistributedBeginAndCorrection(
    id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library,
    id<MTLComputePipelineState> legacyBegin,
    id<MTLComputePipelineState> beginPartial,
    id<MTLComputePipelineState> beginFinish,
    id<MTLComputePipelineState> beginBasis0,
    id<MTLComputePipelineState> legacyCorrection,
    id<MTLComputePipelineState> correctionPartial,
    id<MTLComputePipelineState> correctionFinish) {
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
    tiledBeginBindings.emplace_back(24u,beginPartials);
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

            auto finalize = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_finalize_cycle");
            auto telemetryFinalize = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_telemetry_finalize");
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
            checkDistributedBeginAndCorrection(
                device,queue,library,beginLegacy,beginPartial,beginFinish,
                beginBasis0,correctionLegacy,correctionPartial,correctionFinish);
            checkInexactNewtonForcing(device, queue, ewPrepare, ewCommit);
            auto cgs2Partial = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_cgs2_partial");
            auto cgs2ReduceRow = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_cgs2_reduce_row");
            auto cgs2Update = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_cgs2_update");
            auto cgs2MeasureNorm = pipeline(device, library,
                "numi_matter_metal::nm_fgmres_cgs2_measure_norm");
            auto cgs2FinishColumn = pipeline(device, library,
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
            auto cgs2FinishColumnICB = indirectPipeline(device, library,
                "numi_matter_metal::nm_fgmres_cgs2_finish_column");
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

