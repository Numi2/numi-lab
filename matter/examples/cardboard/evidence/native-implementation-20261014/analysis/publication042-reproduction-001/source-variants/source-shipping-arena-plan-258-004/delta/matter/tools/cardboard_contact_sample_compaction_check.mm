#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/shared.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class T>
id<MTLBuffer> makeBuffer(id<MTLDevice> device, const std::vector<T>& values) {
    const NSUInteger bytes = std::max<NSUInteger>(
        sizeof(T), values.size() * sizeof(T));
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes
        options:MTLResourceStorageModeShared];
    require(buffer != nil, "Metal buffer allocation failed");
    std::memset(buffer.contents, 0, buffer.length);
    if (!values.empty())
        std::memcpy(buffer.contents, values.data(), values.size() * sizeof(T));
    return buffer;
}

id<MTLComputePipelineState> pipeline(
    id<MTLDevice> device, id<MTLLibrary> library, const char* name) {
    const std::string qualified = std::string("numi_matter_metal::") + name;
    id<MTLFunction> function = [library newFunctionWithName:
        [NSString stringWithUTF8String:qualified.c_str()]];
    require(function != nil, "required production/test kernel is missing");
    NSError* error = nil;
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:&error];
    require(result != nil, "compute pipeline creation failed");
    return result;
}

void dispatchThreads(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> kernel,
    NSUInteger count) {
    if (count == 0u) return;
    [encoder setComputePipelineState:kernel];
    [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
}

void dispatchGroups32(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> kernel,
    NSUInteger groups) {
    if (groups == 0u) return;
    [encoder setComputePipelineState:kernel];
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
}

struct Fixture {
    static constexpr uint32_t E = 2u;
    static constexpr uint32_t P = 7u;
    static constexpr uint32_t N = 3u;
    static constexpr uint32_t R = 2u;
    static constexpr uint32_t Capacity = 4u;

    NMMatterDispatchGPU dispatch{};
    std::vector<NMContactPairGPU> pairs;
    std::vector<uint32_t> activeMask;
    std::vector<NMContactSampleGPU> denseSamples;
    std::vector<uint32_t> nodeIncidence;
    std::vector<NMIncidenceRangeGPU> nodeRanges;
    std::vector<uint32_t> rigidIncidence;
    std::vector<NMIncidenceRangeGPU> rigidRanges;

    Fixture() {
        dispatch.environmentCount = E;
        dispatch.objectCount = 1u;
        dispatch.gridNodeCount = 0u;
        dispatch.femNodeCount = N;
        dispatch.rigidProxyCount = R;
        dispatch.contactPairCount = P;
        dispatch.reservedMixed1 = Capacity;

        const uint32_t node[P]  = {0u, 1u, 0u, 2u, 0u, 1u, 2u};
        const uint32_t proxy[P] = {0u, 0u, 1u, 1u, 1u, 0u, 1u};
        pairs.resize(P);
        for (uint32_t i = 0u; i < P; ++i)
            pairs[i] = NMContactPairGPU{node[i], proxy[i], 0u, 0u};

        activeMask.assign(E * P, 0u);
        for (uint32_t pair : {0u, 2u, 4u, 5u})
            activeMask[pair] = 1u;
        for (uint32_t pair : {1u, 2u, 3u, 6u})
            activeMask[P + pair] = 1u;

        denseSamples.resize(E * P);
        for (uint32_t env = 0u; env < E; ++env) {
            for (uint32_t pair = 0u; pair < P; ++pair) {
                NMContactSampleGPU sample{};
                sample.identity = {
                    pairs[pair].continuumNode,
                    pairs[pair].rigidProxy,
                    pairs[pair].objectIndex,
                    activeMask[env * P + pair] ? NM_CONTACT_VALID : 0u,
                };
                float x = static_cast<float>((env + 1u) * (pair + 2u));
                if (env == 0u && pair == 0u) x = 1.0e20f;
                if (env == 0u && pair == 2u) x = -1.0e20f;
                if (env == 0u && pair == 4u) x = 3.25f;
                sample.impulseAndNormal = {
                    x, static_cast<float>(pair + 1u),
                    -static_cast<float>(env + pair + 1u), 0.5f};
                sample.angularImpulseAndTangent = {
                    static_cast<float>(env + pair + 2u),
                    static_cast<float>(2u * pair + 1u),
                    -static_cast<float>(env + 3u * pair + 1u), 0.25f};
                denseSamples[env * P + pair] =
                    activeMask[env * P + pair] ? sample : NMContactSampleGPU{};
            }
        }

        std::vector<std::vector<uint32_t>> nodeRows(N);
        std::vector<std::vector<uint32_t>> rigidRows(R);
        for (uint32_t pair = 0u; pair < P; ++pair) {
            nodeRows[pairs[pair].continuumNode].push_back(pair);
            rigidRows[pairs[pair].rigidProxy].push_back(pair);
        }
        nodeRanges.resize(N);
        for (uint32_t n = 0u; n < N; ++n) {
            const uint32_t first =
                static_cast<uint32_t>(nodeIncidence.size());
            nodeIncidence.insert(nodeIncidence.end(),
                nodeRows[n].begin(), nodeRows[n].end());
            nodeRanges[n] = NMIncidenceRangeGPU{
                first, static_cast<uint32_t>(nodeRows[n].size()), 0u, 0u};
        }
        rigidRanges.resize(R);
        for (uint32_t r = 0u; r < R; ++r) {
            const uint32_t first =
                static_cast<uint32_t>(rigidIncidence.size());
            rigidIncidence.insert(rigidIncidence.end(),
                rigidRows[r].begin(), rigidRows[r].end());
            rigidRanges[r] = NMIncidenceRangeGPU{
                first, static_cast<uint32_t>(rigidRows[r].size()), 0u, 0u};
        }
    }
};

struct Buffers {
    id<MTLBuffer> dispatch = nil;
    id<MTLBuffer> pairs = nil;
    id<MTLBuffer> activeMask = nil;
    id<MTLBuffer> sourceSamples = nil;
    id<MTLBuffer> compactSamples = nil;
    id<MTLBuffer> slotsByPair = nil;
    id<MTLBuffer> activeCounts = nil;
    id<MTLBuffer> pageComplete = nil;
    id<MTLBuffer> activePairs = nil;
    id<MTLBuffer> activeSlots = nil;
    id<MTLBuffer> statuses = nil;
    id<MTLBuffer> config = nil;
    id<MTLBuffer> denseConfig = nil;
    id<MTLBuffer> densePage = nil;
    id<MTLBuffer> denseSlots = nil;
    id<MTLBuffer> denseCounts = nil;
    id<MTLBuffer> nodeIncidence = nil;
    id<MTLBuffer> nodeRanges = nil;
    id<MTLBuffer> rigidIncidence = nil;
    id<MTLBuffer> rigidRanges = nil;
    id<MTLBuffer> compactResidual = nil;
    id<MTLBuffer> denseResidual = nil;
    id<MTLBuffer> compactReactions = nil;
    id<MTLBuffer> denseReactions = nil;
    id<MTLBuffer> compactHistory = nil;
    id<MTLBuffer> denseHistory = nil;
};

Buffers makeBuffers(id<MTLDevice> device, const Fixture& f,
                    uint32_t capacity, bool overflow) {
    Buffers b{};
    b.dispatch = makeBuffer(device,
        std::vector<NMMatterDispatchGPU>{f.dispatch});
    b.pairs = makeBuffer(device, f.pairs);
    b.activeMask = makeBuffer(device, f.activeMask);
    b.sourceSamples = makeBuffer(device, f.denseSamples);
    b.compactSamples = makeBuffer(device,
        std::vector<NMContactSampleGPU>(Fixture::E * capacity));
    b.slotsByPair = makeBuffer(device,
        std::vector<uint32_t>(Fixture::E * Fixture::P, NM_INVALID_INDEX));
    b.activeCounts = makeBuffer(device, std::vector<uint32_t>(Fixture::E));
    b.pageComplete = makeBuffer(device, std::vector<uint32_t>(Fixture::E));
    b.activePairs = makeBuffer(device,
        std::vector<uint32_t>(Fixture::E * capacity, NM_INVALID_INDEX));
    b.activeSlots = makeBuffer(device,
        std::vector<uint32_t>(Fixture::E * Fixture::P, NM_INVALID_INDEX));
    b.statuses = makeBuffer(device,
        std::vector<NMMatterStatusGPU>(Fixture::E));
    const std::vector<uint32_t> config{1u, capacity};
    const std::vector<uint32_t> denseConfig{0u, Fixture::P};
    b.config = makeBuffer(device, config);
    b.denseConfig = makeBuffer(device, denseConfig);
    b.densePage = makeBuffer(device, std::vector<uint32_t>(Fixture::E, 0u));
    b.denseSlots = makeBuffer(device,
        std::vector<uint32_t>(Fixture::E * Fixture::P, NM_INVALID_INDEX));
    b.denseCounts = makeBuffer(device, std::vector<uint32_t>(Fixture::E, 0u));
    b.nodeIncidence = makeBuffer(device, f.nodeIncidence);
    b.nodeRanges = makeBuffer(device, f.nodeRanges);
    b.rigidIncidence = makeBuffer(device, f.rigidIncidence);
    b.rigidRanges = makeBuffer(device, f.rigidRanges);
    const float sentinel = overflow ? 17.0f : 0.0f;
    b.compactResidual = makeBuffer(device,
        std::vector<nm_float4>(Fixture::E * Fixture::N,
            nm_float4{sentinel, sentinel + 1.0f, sentinel + 2.0f, 0.0f}));
    b.denseResidual = makeBuffer(device,
        std::vector<nm_float4>(Fixture::E * Fixture::N));
    b.compactReactions = makeBuffer(device,
        std::vector<NMRigidReactionGPU>(Fixture::E * Fixture::R));
    b.denseReactions = makeBuffer(device,
        std::vector<NMRigidReactionGPU>(Fixture::E * Fixture::R));
    b.compactHistory = makeBuffer(device,
        std::vector<nm_float4>(Fixture::E * Fixture::P));
    b.denseHistory = makeBuffer(device,
        std::vector<nm_float4>(Fixture::E * Fixture::P));
    return b;
}

void run(id<MTLDevice> device, id<MTLLibrary> library) {
    static_assert(sizeof(NMContactPairGPU) == 16u);
    static_assert(sizeof(NMContactSampleGPU) == 160u);
    static_assert(sizeof(NMIncidenceRangeGPU) == 16u);
    constexpr uint32_t capacity = Fixture::Capacity;
    Fixture f;
    Buffers b = makeBuffers(device, f, capacity, false);

    const auto begin = pipeline(device, library,
        "nm_contact_begin_compact_samples");
    const auto append = pipeline(device, library,
        "nm_contact_sample_compaction_fixture_append");
    const auto finish = pipeline(device, library,
        "nm_contact_finish_compact_samples");
    const auto femResidual = pipeline(device, library,
        "nm_contact_accumulate_fem_residual");
    const auto rigidReduce = pipeline(device, library,
        "nm_contact_reduce_rigid");
    const auto publishHistory = pipeline(device, library,
        "nm_contact_publish_primal_history");

    id<MTLCommandQueue> queue = [device newCommandQueue];
    require(queue != nil, "command queue creation failed");
    id<MTLCommandBuffer> command = [queue commandBuffer];
    require(command != nil, "command buffer creation failed");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    require(encoder != nil, "compute encoder creation failed");

    [encoder setBuffer:b.dispatch offset:0u atIndex:0u];
    [encoder setBytes:&capacity length:sizeof(capacity) atIndex:1u];
    [encoder setBuffer:b.slotsByPair offset:0u atIndex:2u];
    [encoder setBuffer:b.activeCounts offset:0u atIndex:3u];
    [encoder setBuffer:b.pageComplete offset:0u atIndex:4u];
    [encoder setBuffer:b.statuses offset:0u atIndex:5u];
    [encoder setBuffer:b.activeSlots offset:0u atIndex:6u];
    dispatchGroups32(encoder, begin, Fixture::E);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setBuffer:b.dispatch offset:0u atIndex:0u];
    [encoder setBytes:&capacity length:sizeof(capacity) atIndex:1u];
    [encoder setBuffer:b.activeMask offset:0u atIndex:2u];
    [encoder setBuffer:b.pairs offset:0u atIndex:3u];
    [encoder setBuffer:b.sourceSamples offset:0u atIndex:4u];
    [encoder setBuffer:b.compactSamples offset:0u atIndex:5u];
    [encoder setBuffer:b.slotsByPair offset:0u atIndex:6u];
    [encoder setBuffer:b.activeCounts offset:0u atIndex:7u];
    [encoder setBuffer:b.statuses offset:0u atIndex:8u];
    dispatchThreads(encoder, append, Fixture::E * Fixture::P);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setBuffer:b.dispatch offset:0u atIndex:0u];
    [encoder setBytes:&capacity length:sizeof(capacity) atIndex:1u];
    [encoder setBuffer:b.pairs offset:0u atIndex:2u];
    [encoder setBuffer:b.compactSamples offset:0u atIndex:3u];
    [encoder setBuffer:b.activePairs offset:0u atIndex:4u];
    [encoder setBuffer:b.slotsByPair offset:0u atIndex:5u];
    [encoder setBuffer:b.activeCounts offset:0u atIndex:6u];
    [encoder setBuffer:b.pageComplete offset:0u atIndex:7u];
    [encoder setBuffer:b.statuses offset:0u atIndex:8u];
    [encoder setBuffer:b.activeSlots offset:0u atIndex:9u];
    dispatchGroups32(encoder, finish, Fixture::E);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    auto runFem = [&](id<MTLBuffer> samples, id<MTLBuffer> config,
                      id<MTLBuffer> page, id<MTLBuffer> slots,
                      id<MTLBuffer> counts, id<MTLBuffer> residual) {
        [encoder setBuffer:b.dispatch offset:0u atIndex:0u];
        [encoder setBuffer:b.nodeIncidence offset:0u atIndex:1u];
        [encoder setBuffer:b.nodeRanges offset:0u atIndex:2u];
        [encoder setBuffer:samples offset:0u atIndex:3u];
        [encoder setBuffer:residual offset:0u atIndex:4u];
        [encoder setBuffer:config offset:0u atIndex:5u];
        [encoder setBuffer:page offset:0u atIndex:6u];
        [encoder setBuffer:slots offset:0u atIndex:7u];
        [encoder setBuffer:counts offset:0u atIndex:8u];
        [encoder setBuffer:config offset:sizeof(uint32_t) atIndex:9u];
        dispatchThreads(encoder, femResidual, Fixture::E * Fixture::N);
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    };
    runFem(b.compactSamples, b.config, b.pageComplete, b.slotsByPair,
           b.activeCounts, b.compactResidual);
    runFem(b.sourceSamples, b.denseConfig, b.densePage, b.denseSlots,
           b.denseCounts, b.denseResidual);

    auto runRigid = [&](id<MTLBuffer> samples, id<MTLBuffer> config,
                        id<MTLBuffer> page, id<MTLBuffer> slots,
                        id<MTLBuffer> counts, id<MTLBuffer> reactions) {
        [encoder setBuffer:b.dispatch offset:0u atIndex:0u];
        [encoder setBuffer:b.rigidIncidence offset:0u atIndex:1u];
        [encoder setBuffer:b.rigidRanges offset:0u atIndex:2u];
        [encoder setBuffer:samples offset:0u atIndex:3u];
        [encoder setBuffer:reactions offset:0u atIndex:4u];
        [encoder setBuffer:config offset:0u atIndex:5u];
        [encoder setBuffer:page offset:0u atIndex:6u];
        [encoder setBuffer:slots offset:0u atIndex:7u];
        [encoder setBuffer:counts offset:0u atIndex:8u];
        [encoder setBuffer:config offset:sizeof(uint32_t) atIndex:9u];
        dispatchThreads(encoder, rigidReduce, Fixture::E * Fixture::R);
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    };
    runRigid(b.compactSamples, b.config, b.pageComplete, b.slotsByPair,
             b.activeCounts, b.compactReactions);
    runRigid(b.sourceSamples, b.denseConfig, b.densePage, b.denseSlots,
             b.denseCounts, b.denseReactions);

    auto runHistory = [&](id<MTLBuffer> samples, id<MTLBuffer> config,
                          id<MTLBuffer> page, id<MTLBuffer> slots,
                          id<MTLBuffer> counts, id<MTLBuffer> history) {
        [encoder setBuffer:b.dispatch offset:0u atIndex:0u];
        [encoder setBuffer:samples offset:0u atIndex:1u];
        [encoder setBuffer:history offset:0u atIndex:2u];
        [encoder setBuffer:config offset:0u atIndex:3u];
        [encoder setBuffer:page offset:0u atIndex:4u];
        [encoder setBuffer:slots offset:0u atIndex:5u];
        [encoder setBuffer:counts offset:0u atIndex:6u];
        [encoder setBuffer:config offset:sizeof(uint32_t) atIndex:7u];
        dispatchThreads(encoder, publishHistory, Fixture::E * Fixture::P);
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    };
    runHistory(b.compactSamples, b.config, b.pageComplete, b.slotsByPair,
               b.activeCounts, b.compactHistory);
    runHistory(b.sourceSamples, b.denseConfig, b.densePage, b.denseSlots,
               b.denseCounts, b.denseHistory);

    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
        "native compaction production-kernel command failed");

    const auto* status = static_cast<const NMMatterStatusGPU*>(b.statuses.contents);
    const auto* counts = static_cast<const uint32_t*>(b.activeCounts.contents);
    const auto* complete = static_cast<const uint32_t*>(b.pageComplete.contents);
    const auto* pairMap = static_cast<const uint32_t*>(b.slotsByPair.contents);
    const auto* active = static_cast<const uint32_t*>(b.activePairs.contents);
    const auto* activeSlots = static_cast<const uint32_t*>(b.activeSlots.contents);
    const auto* compactSamples =
        static_cast<const NMContactSampleGPU*>(b.compactSamples.contents);
    const auto* denseSamples =
        static_cast<const NMContactSampleGPU*>(b.sourceSamples.contents);
    require(status[0].code == 0u && status[1].code == 0u,
        "valid compact page raised a status");
    for (uint32_t env = 0u; env < Fixture::E; ++env) {
        require(complete[env] == 1u, "valid page did not complete");
        require(counts[env] == Fixture::Capacity,
            "valid page tuple count changed");
        uint32_t slot = 0u;
        for (uint32_t pair = 0u; pair < Fixture::P; ++pair) {
            const bool expected = f.activeMask[env * Fixture::P + pair] != 0u;
            const uint32_t map = pairMap[env * Fixture::P + pair];
            if (!expected) {
                require(map == NM_INVALID_INDEX,
                    "inactive canonical pair received a compact slot");
                continue;
            }
            require(map < capacity, "active pair has an out-of-range compact slot");
            require(active[env * capacity + slot] == pair,
                "active pair list is not canonical pair-index order");
            require(activeSlots[env * Fixture::P + pair] == slot,
                "canonical active slot map disagrees with compact list");
            const NMContactSampleGPU& compact =
                compactSamples[env * capacity + map];
            const NMContactSampleGPU& dense =
                denseSamples[env * Fixture::P + pair];
            require(std::memcmp(&compact, &dense, sizeof(compact)) == 0,
                "compact tuple differs from its canonical dense tuple");
            ++slot;
        }
        require(slot == Fixture::Capacity,
            "active pair enumeration omitted or duplicated a tuple");
    }

    const auto* compactResidual =
        static_cast<const nm_float4*>(b.compactResidual.contents);
    const auto* denseResidual =
        static_cast<const nm_float4*>(b.denseResidual.contents);
    const auto* compactReactions =
        static_cast<const NMRigidReactionGPU*>(b.compactReactions.contents);
    const auto* denseReactions =
        static_cast<const NMRigidReactionGPU*>(b.denseReactions.contents);
    const auto* compactHistory =
        static_cast<const nm_float4*>(b.compactHistory.contents);
    const auto* denseHistory =
        static_cast<const nm_float4*>(b.denseHistory.contents);
    require(std::memcmp(compactResidual, denseResidual,
                        Fixture::E * Fixture::N * sizeof(nm_float4)) == 0,
        "compact FEM force gather changed canonical accumulation");
    require(std::memcmp(compactReactions, denseReactions,
                        Fixture::E * Fixture::R * sizeof(NMRigidReactionGPU)) == 0,
        "compact rigid reaction gather changed canonical accumulation");
    require(std::memcmp(compactHistory, denseHistory,
                        Fixture::E * Fixture::P * sizeof(nm_float4)) == 0,
        "compact per-pair history publication changed tuple values");

    // Overflow must poison the whole sample page. The consumer is run against
    // a sentinel residual to prove it does not silently apply a valid prefix.
    constexpr uint32_t smallCapacity = 2u;
    Buffers overflow = makeBuffers(device, f, smallCapacity, true);
    id<MTLCommandBuffer> overflowCommand = [queue commandBuffer];
    require(overflowCommand != nil, "overflow command buffer creation failed");
    id<MTLComputeCommandEncoder> overflowEncoder =
        [overflowCommand computeCommandEncoder];
    require(overflowEncoder != nil, "overflow encoder creation failed");
    [overflowEncoder setBuffer:overflow.dispatch offset:0u atIndex:0u];
    [overflowEncoder setBytes:&smallCapacity length:sizeof(smallCapacity) atIndex:1u];
    [overflowEncoder setBuffer:overflow.slotsByPair offset:0u atIndex:2u];
    [overflowEncoder setBuffer:overflow.activeCounts offset:0u atIndex:3u];
    [overflowEncoder setBuffer:overflow.pageComplete offset:0u atIndex:4u];
    [overflowEncoder setBuffer:overflow.statuses offset:0u atIndex:5u];
    [overflowEncoder setBuffer:overflow.activeSlots offset:0u atIndex:6u];
    dispatchGroups32(overflowEncoder, begin, Fixture::E);
    [overflowEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [overflowEncoder setBuffer:overflow.dispatch offset:0u atIndex:0u];
    [overflowEncoder setBytes:&smallCapacity length:sizeof(smallCapacity) atIndex:1u];
    [overflowEncoder setBuffer:overflow.activeMask offset:0u atIndex:2u];
    [overflowEncoder setBuffer:overflow.pairs offset:0u atIndex:3u];
    [overflowEncoder setBuffer:overflow.sourceSamples offset:0u atIndex:4u];
    [overflowEncoder setBuffer:overflow.compactSamples offset:0u atIndex:5u];
    [overflowEncoder setBuffer:overflow.slotsByPair offset:0u atIndex:6u];
    [overflowEncoder setBuffer:overflow.activeCounts offset:0u atIndex:7u];
    [overflowEncoder setBuffer:overflow.statuses offset:0u atIndex:8u];
    dispatchThreads(overflowEncoder, append, Fixture::E * Fixture::P);
    [overflowEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [overflowEncoder setBuffer:overflow.dispatch offset:0u atIndex:0u];
    [overflowEncoder setBytes:&smallCapacity length:sizeof(smallCapacity) atIndex:1u];
    [overflowEncoder setBuffer:overflow.pairs offset:0u atIndex:2u];
    [overflowEncoder setBuffer:overflow.compactSamples offset:0u atIndex:3u];
    [overflowEncoder setBuffer:overflow.activePairs offset:0u atIndex:4u];
    [overflowEncoder setBuffer:overflow.slotsByPair offset:0u atIndex:5u];
    [overflowEncoder setBuffer:overflow.activeCounts offset:0u atIndex:6u];
    [overflowEncoder setBuffer:overflow.pageComplete offset:0u atIndex:7u];
    [overflowEncoder setBuffer:overflow.statuses offset:0u atIndex:8u];
    [overflowEncoder setBuffer:overflow.activeSlots offset:0u atIndex:9u];
    dispatchGroups32(overflowEncoder, finish, Fixture::E);
    [overflowEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [overflowEncoder setBuffer:overflow.dispatch offset:0u atIndex:0u];
    [overflowEncoder setBuffer:overflow.nodeIncidence offset:0u atIndex:1u];
    [overflowEncoder setBuffer:overflow.nodeRanges offset:0u atIndex:2u];
    [overflowEncoder setBuffer:overflow.compactSamples offset:0u atIndex:3u];
    [overflowEncoder setBuffer:overflow.compactResidual offset:0u atIndex:4u];
    [overflowEncoder setBuffer:overflow.config offset:0u atIndex:5u];
    [overflowEncoder setBuffer:overflow.pageComplete offset:0u atIndex:6u];
    [overflowEncoder setBuffer:overflow.slotsByPair offset:0u atIndex:7u];
    [overflowEncoder setBuffer:overflow.activeCounts offset:0u atIndex:8u];
    [overflowEncoder setBuffer:overflow.config offset:sizeof(uint32_t) atIndex:9u];
    dispatchThreads(overflowEncoder, femResidual, Fixture::E * Fixture::N);
    [overflowEncoder endEncoding];
    [overflowCommand commit];
    [overflowCommand waitUntilCompleted];
    require(overflowCommand.status == MTLCommandBufferStatusCompleted,
        "overflow fail-closed command did not complete");
    const auto* overflowStatus =
        static_cast<const NMMatterStatusGPU*>(overflow.statuses.contents);
    const auto* overflowPage =
        static_cast<const uint32_t*>(overflow.pageComplete.contents);
    const auto* overflowCount =
        static_cast<const uint32_t*>(overflow.activeCounts.contents);
    const auto* sentinelResidual =
        static_cast<const nm_float4*>(overflow.compactResidual.contents);
    for (uint32_t env = 0u; env < Fixture::E; ++env) {
        require(overflowStatus[env].code == NM_STATUS_CAPACITY_OVERFLOW,
            "overflow did not preserve an explicit capacity failure");
        require(overflowCount[env] == Fixture::Capacity,
            "overflow fixture did not observe the full active tuple count");
        require(overflowPage[env] == 0u,
            "overflow page was incorrectly marked complete");
        for (uint32_t node = 0u; node < Fixture::N; ++node) {
            const nm_float4 value = sentinelResidual[env * Fixture::N + node];
            require(value.x == 17.0f && value.y == 18.0f &&
                    value.z == 19.0f && value.w == 0.0f,
                "incomplete page applied a prefix force");
        }
    }
    std::cout << "production_contact_sample_compaction=PASS"
        << " E=" << Fixture::E << " P=" << Fixture::P
        << " active_per_env=" << Fixture::Capacity
        << " FEM/rigid/history_bitwise_parity=PASS"
        << " overflow_prefix_rejected=PASS physical_transactions=0\n";
}
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            require(argc == 2, "usage: checker <NumiMatterContactSampleCompaction.metallib>");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "no Metal device");
            NSError* error = nil;
            NSString* path = [NSString stringWithUTF8String:argv[1]];
            id<MTLLibrary> library = [device newLibraryWithURL:
                [NSURL fileURLWithPath:path] error:&error];
            require(library != nil, "cannot load contact compaction metallib");
            run(device, library);
            return 0;
        } catch (const std::exception& ex) {
            std::cerr << "contact sample compaction checker failed: "
                      << ex.what() << '\n';
            return 1;
        }
    }
}
