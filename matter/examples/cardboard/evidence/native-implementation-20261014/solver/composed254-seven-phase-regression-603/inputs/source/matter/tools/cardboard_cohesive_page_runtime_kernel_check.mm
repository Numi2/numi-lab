#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/detail/shipping_contact_page_plan.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace numi::matter::detail;

static void req(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class T>
static id<MTLBuffer> mkbuf(id<MTLDevice> device, const std::vector<T>& values) {
    const NSUInteger length = std::max<NSUInteger>(
        sizeof(T), values.size() * sizeof(T));
    id<MTLBuffer> buffer = [device newBufferWithLength:length
        options:MTLResourceStorageModeShared];
    req(buffer != nil, "buffer allocation failed");
    std::memset(buffer.contents, 0, buffer.length);
    if (!values.empty()) {
        std::memcpy(buffer.contents, values.data(),
            values.size() * sizeof(T));
    }
    return buffer;
}

static id<MTLComputePipelineState> pipeline(
    id<MTLDevice> device, id<MTLLibrary> library, const char* name
) {
    const std::string qualified = std::string("numi_matter_metal::") + name;
    id<MTLFunction> function = [library newFunctionWithName:
        [NSString stringWithUTF8String:qualified.c_str()]];
    req(function != nil, "missing production cohesive page kernel");
    NSError* error = nil;
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:&error];
    req(result != nil, "pipeline creation failed");
    return result;
}

static void dispatch(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> state,
    NSUInteger threadCount
) {
    if (!threadCount) return;
    [encoder setComputePipelineState:state];
    [encoder dispatchThreads:MTLSizeMake(threadCount, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(
            std::min<NSUInteger>(128, state.maxTotalThreadsPerThreadgroup),
            1, 1)];
}

static void run(id<MTLDevice> device, id<MTLLibrary> library) {
    constexpr uint32_t E = 2;
    constexpr uint32_t N = 7;
    constexpr uint32_t C = 5;
    constexpr uint32_t capacity = 2;

    ShippingCohesivePagePlanInput input{};
    input.environmentCount = E;
    input.femNodeCount = N;
    input.cohesiveFaceCount = C;
    input.pageBufferBudgetBytes =
        E * capacity * sizeof(NMCohesiveFaceVectorGPU);
    input.maxBufferLength = 1u << 20;
    const auto plan = makeShippingCohesivePagePlan(input);
    req(plan.admitted() && plan.paged &&
        plan.facesPerPage == capacity && plan.pageCount == 3 &&
        plan.lastPageFaces == 1, "bounded test plan mismatch");

    NMMatterDispatchGPU dispatchData{};
    dispatchData.environmentCount = E;
    dispatchData.objectCount = 1;
    dispatchData.femNodeCount = N;
    dispatchData.cohesiveFaceCount = C;
    dispatchData.gravityAndTimestep.w = 0.25f;
    NMMicrostepGPU microstep{};
    microstep.microtickCount = 1;
    microstep.time = {0.25f, 4, 0, 0.25f};
    NMContinuumObjectGPU object{};
    object.representation = NM_REPRESENTATION_FEM;
    object.flags = NM_OBJECT_ACTIVE;
    object.schedulerIndex = 0;

    std::vector<NMSchedulerStateGPU> schedulers(E);
    std::vector<NMAdaptiveStateGPU> adaptive(E);
    for (uint32_t env = 0; env < E; ++env) {
        schedulers[env].numerical.w = 1;
        adaptive[env].activeRepresentation = NM_REPRESENTATION_FEM;
    }

    std::vector<std::vector<uint32_t>> entries(N);
    for (uint32_t face = 0; face < C; ++face) {
        for (uint32_t slot = 0; slot < 6; ++slot) {
            entries[(face + slot) % N].push_back(face * 6 + slot);
        }
    }
    std::vector<uint32_t> incidence;
    std::vector<NMIncidenceRangeGPU> ranges(E * N);
    for (uint32_t node = 0; node < N; ++node) {
        const uint32_t first = static_cast<uint32_t>(incidence.size());
        incidence.insert(incidence.end(), entries[node].begin(),
            entries[node].end());
        const NMIncidenceRangeGPU range{
            first, static_cast<uint32_t>(entries[node].size()), 0, 0};
        for (uint32_t env = 0; env < E; ++env) ranges[env * N + node] = range;
    }

    std::vector<NMFEMNodeStateGPU> nodes(E * N);
    for (auto& node : nodes) {
        node.positionAndMass.w = 1.0f;
        node.velocityAndInverseMass.w = 0.5f;
    }
    std::vector<NMMatterStatusGPU> statuses(E);

    auto dispatchBuffer = mkbuf(device,
        std::vector<NMMatterDispatchGPU>{dispatchData});
    auto microstepBuffer = mkbuf(device,
        std::vector<NMMicrostepGPU>{microstep});
    auto objectBuffer = mkbuf(device,
        std::vector<NMContinuumObjectGPU>{object});
    auto schedulerBuffer = mkbuf(device, schedulers);
    auto adaptiveBuffer = mkbuf(device, adaptive);
    auto incidenceBuffer = mkbuf(device, incidence);
    auto rangeBuffer = mkbuf(device, ranges);
    auto nodeBuffer = mkbuf(device, nodes);
    auto statusBuffer = mkbuf(device, statuses);
    auto accumulator = [device newBufferWithLength:plan.accumulatorBytes
        options:MTLResourceStorageModeShared];
    req(accumulator != nil, "accumulator allocation failed");

    uint32_t starts[3] = {0, 2, 4};
    uint32_t counts[3] = {2, 2, 1};
    id<MTLBuffer> pages[3] = {nil, nil, nil};
    for (uint32_t pageIndex = 0; pageIndex < plan.pageCount; ++pageIndex) {
        pages[pageIndex] = [device newBufferWithLength:plan.facePageBytes
            options:MTLResourceStorageModeShared];
        req(pages[pageIndex] != nil, "page allocation failed");
        std::memset(pages[pageIndex].contents, 0, pages[pageIndex].length);
        auto* values = static_cast<NMCohesiveFaceVectorGPU*>(
            pages[pageIndex].contents);
        for (uint32_t env = 0; env < E; ++env) {
            for (uint32_t local = 0; local < counts[pageIndex]; ++local) {
                const uint32_t face = starts[pageIndex] + local;
                for (uint32_t slot = 0; slot < 6; ++slot) {
                    const float value = static_cast<float>(
                        1 + env * 100 + face * 10 + slot);
                    values[env * counts[pageIndex] + local].node[slot] =
                        {value, 2 * value, -value, 0};
                }
            }
        }
    }

    auto residual = mkbuf(device, std::vector<nm_float4>(E * N));
    auto preconditioned = mkbuf(device, std::vector<nm_float4>(E * N));
    auto direction = mkbuf(device, std::vector<nm_float4>(E * N));
    auto reactions = mkbuf(device, std::vector<nm_float4>(E * N));
    auto work = mkbuf(device, std::vector<nm_float4>(E * N));

    auto clear = pipeline(device, library,
        "nm_fem_clear_cohesive_force_accumulator");
    auto accumulate = pipeline(device, library,
        "nm_fem_accumulate_cohesive_page_forces");
    auto applyForce = pipeline(device, library,
        "nm_fem_apply_cohesive_force_accumulator");
    auto applyOperator = pipeline(device, library,
        "nm_fem_apply_cohesive_operator_accumulator");

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    req(queue && command && encoder, "command setup failed");

    auto clearAccumulator = [&]() {
        [encoder setBuffer:dispatchBuffer offset:0 atIndex:0];
        [encoder setBuffer:accumulator offset:0 atIndex:1];
        dispatch(encoder, clear, E * N);
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    };

    auto accumulatePages = [&]() {
        for (uint32_t pageIndex = 0; pageIndex < plan.pageCount; ++pageIndex) {
            const uint32_t start = starts[pageIndex];
            const uint32_t count = counts[pageIndex];
            [encoder setBuffer:dispatchBuffer offset:0 atIndex:0];
            [encoder setBuffer:microstepBuffer offset:0 atIndex:1];
            [encoder setBuffer:objectBuffer offset:0 atIndex:2];
            [encoder setBuffer:incidenceBuffer offset:0 atIndex:3];
            [encoder setBuffer:rangeBuffer offset:0 atIndex:4];
            [encoder setBuffer:schedulerBuffer offset:0 atIndex:5];
            [encoder setBuffer:adaptiveBuffer offset:0 atIndex:6];
            [encoder setBuffer:pages[pageIndex] offset:0 atIndex:7];
            [encoder setBuffer:accumulator offset:0 atIndex:8];
            [encoder setBuffer:statusBuffer offset:0 atIndex:9];
            [encoder setBytes:&start length:sizeof(start) atIndex:10];
            [encoder setBytes:&count length:sizeof(count) atIndex:11];
            dispatch(encoder, accumulate, E * N);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
    };

    clearAccumulator();
    accumulatePages();

    [encoder setBuffer:dispatchBuffer offset:0 atIndex:0];
    [encoder setBuffer:microstepBuffer offset:0 atIndex:1];
    [encoder setBuffer:objectBuffer offset:0 atIndex:2];
    [encoder setBuffer:nodeBuffer offset:0 atIndex:3];
    [encoder setBuffer:rangeBuffer offset:0 atIndex:4];
    [encoder setBuffer:schedulerBuffer offset:0 atIndex:5];
    [encoder setBuffer:adaptiveBuffer offset:0 atIndex:6];
    [encoder setBuffer:accumulator offset:0 atIndex:7];
    [encoder setBuffer:residual offset:0 atIndex:8];
    [encoder setBuffer:preconditioned offset:0 atIndex:9];
    [encoder setBuffer:direction offset:0 atIndex:10];
    [encoder setBuffer:reactions offset:0 atIndex:11];
    [encoder setBuffer:statusBuffer offset:0 atIndex:12];
    dispatch(encoder, applyForce, E * N);

    clearAccumulator();
    accumulatePages();

    [encoder setBuffer:dispatchBuffer offset:0 atIndex:0];
    [encoder setBuffer:microstepBuffer offset:0 atIndex:1];
    [encoder setBuffer:objectBuffer offset:0 atIndex:2];
    [encoder setBuffer:nodeBuffer offset:0 atIndex:3];
    [encoder setBuffer:rangeBuffer offset:0 atIndex:4];
    [encoder setBuffer:schedulerBuffer offset:0 atIndex:5];
    [encoder setBuffer:adaptiveBuffer offset:0 atIndex:6];
    [encoder setBuffer:accumulator offset:0 atIndex:7];
    [encoder setBuffer:work offset:0 atIndex:8];
    [encoder setBuffer:statusBuffer offset:0 atIndex:9];
    dispatch(encoder, applyOperator, E * N);

    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    req(command.status == MTLCommandBufferStatusCompleted,
        "page consumer dispatch failed");

    const auto* force = static_cast<const nm_float4*>(accumulator.contents);
    const auto* residualValues =
        static_cast<const nm_float4*>(residual.contents);
    const auto* preconditionedValues =
        static_cast<const nm_float4*>(preconditioned.contents);
    const auto* directionValues =
        static_cast<const nm_float4*>(direction.contents);
    const auto* workValues = static_cast<const nm_float4*>(work.contents);

    for (uint32_t env = 0; env < E; ++env) {
        for (uint32_t node = 0; node < N; ++node) {
            float x = 0;
            float y = 0;
            float z = 0;
            for (uint32_t face = 0; face < C; ++face) {
                for (uint32_t slot = 0; slot < 6; ++slot) {
                    if ((face + slot) % N != node) continue;
                    const float value = static_cast<float>(
                        1 + env * 100 + face * 10 + slot);
                    x += value;
                    y += 2 * value;
                    z -= value;
                }
            }
            const uint32_t index = env * N + node;
            req(force[index].x == x && force[index].y == y &&
                force[index].z == z, "ordered page accumulation mismatch");
            req(residualValues[index].x == 0.25f * x &&
                residualValues[index].y == 0.25f * y &&
                residualValues[index].z == 0.25f * z,
                "force apply scale mismatch");
            req(preconditionedValues[index].x == (0.5f * 0.25f) * x &&
                directionValues[index].y == (0.5f * 0.25f) * y,
                "force correction mismatch");
            req(workValues[index].x == -(0.25f * 0.25f) * x &&
                workValues[index].z == -(0.25f * 0.25f) * z,
                "operator apply scale mismatch");
        }
    }

    std::cout << "production_cohesive_page_consumers=PASS E=" << E
        << " N=" << N << " C=" << C
        << " page_capacity=" << capacity
        << " pages=" << plan.pageCount
        << " acc_bytes=" << plan.accumulatorBytes
        << " physical_transactions=0\n";
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            req(argc == 2, "usage: checker <NumiMatter.metallib>");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            req(device != nil, "no Metal device");
            NSError* error = nil;
            NSString* path = [NSString stringWithUTF8String:argv[1]];
            id<MTLLibrary> library = [device newLibraryWithURL:
                [NSURL fileURLWithPath:path] error:&error];
            req(library != nil, "cannot load source-bound NumiMatter.metallib");
            run(device, library);
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "cohesive page checker failed: " << error.what() << '\n';
            return 1;
        }
    }
}
