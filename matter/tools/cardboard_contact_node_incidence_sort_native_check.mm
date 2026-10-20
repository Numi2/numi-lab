#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/shared.h"
#include "cardboard_contact_node_incidence_sort_reference.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NUMI_MATTER_METALLIB
#error "CMake must define NUMI_MATTER_METALLIB"
#endif

namespace {
using namespace cardboard::contact_incidence_sort;
constexpr std::uint32_t kEnvironments = 2u;
constexpr std::uint32_t kNodes = 6u;
constexpr std::uint32_t kCapacity = 4u;
unsigned checks = 0u;

void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

Contact contact(std::array<std::uint32_t, 6> nodes, bool valid = true) {
    return {nodes, valid ? kValid : 0u};
}

template<class T>
id<MTLBuffer> inputBuffer(id<MTLDevice> device, const std::vector<T>& values) {
    require(!values.empty(), "test input buffer cannot be empty");
    id<MTLBuffer> buffer = [device newBufferWithBytes:values.data()
        length:values.size() * sizeof(T) options:MTLResourceStorageModeShared];
    require(buffer != nil, "Metal input buffer allocation failed");
    return buffer;
}

template<class T>
id<MTLBuffer> zeroBuffer(id<MTLDevice> device, std::size_t count,
                         T initial = T{}) {
    require(count != 0u, "test output buffer cannot be empty");
    std::vector<T> values(count, initial);
    id<MTLBuffer> buffer = [device newBufferWithBytes:values.data()
        length:values.size() * sizeof(T) options:MTLResourceStorageModeShared];
    require(buffer != nil, "Metal output buffer allocation failed");
    return buffer;
}

id<MTLComputePipelineState> makePipeline(id<MTLDevice> device,
                                        id<MTLLibrary> library,
                                        const char* name) {
    NSString* qualified = [NSString stringWithFormat:@"numi_matter_metal::%s", name];
    id<MTLFunction> function = [library newFunctionWithName:qualified];
    require(function != nil, std::string("Metal incidence kernel missing: ") + name);
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    require(pipeline != nil, error != nil
        ? std::string("Metal incidence pipeline failed: ") +
            [[error localizedDescription] UTF8String]
        : std::string("Metal incidence pipeline failed: ") + name);
    return pipeline;
}

std::vector<NMDeformableContactGPU> packContacts(
    const std::vector<Environment>& environments
) {
    std::vector<NMDeformableContactGPU> packed;
    packed.reserve(kEnvironments * kCapacity);
    for (std::uint32_t environment = 0u; environment < kEnvironments;
         ++environment) {
        for (std::uint32_t slot = 0u; slot < kCapacity; ++slot) {
            const Contact& source = environments[environment].contacts[slot];
            NMDeformableContactGPU contact{};
            contact.nodes = {source.nodes[0], source.nodes[1],
                             source.nodes[2], source.nodes[3]};
            contact.extraNodes = {source.nodes[4], source.nodes[5], 0u, 0u};
            contact.identity.w = source.flags;
            packed.push_back(contact);
        }
    }
    return packed;
}

void executeCase(id<MTLDevice> device,
                 const std::vector<Environment>& environments,
                 id<MTLComputePipelineState> emit,
                 id<MTLComputePipelineState> bitonic,
                 id<MTLComputePipelineState> countSorted,
                 id<MTLComputePipelineState> countLegacy,
                 id<MTLComputePipelineState> scan,
                 id<MTLComputePipelineState> scatter,
                 id<MTLComputePipelineState> scatterLegacy,
                 id<MTLComputePipelineState> radixHistogram,
                 id<MTLComputePipelineState> radixPrefix,
                 id<MTLComputePipelineState> radixScatter,
                 id<MTLComputePipelineState> countRadix,
                 id<MTLComputePipelineState> copyRadix,
                 id<MTLComputePipelineState> scanTiles,
                 id<MTLComputePipelineState> scanTileOffsets,
                 id<MTLComputePipelineState> scanAddTileOffsets,
                 const std::uint32_t radixPassCount) {
    const WorkspaceLayout workspace = checkedWorkspaceLayout(
        kCapacity, kEnvironments, kNodes);
    require(workspace.valid && workspace.paddedPerEnvironment == 32u,
            "test sort workspace layout is not the expected 32 records/environment");
    NMMatterDispatchGPU dispatch{};
    dispatch.environmentCount = kEnvironments;
    dispatch.gridNodeCount = 2u;
    dispatch.femNodeCount = kNodes - dispatch.gridNodeCount;
    dispatch.deformableContactCapacity = kCapacity;
    const std::uint32_t padded = workspace.paddedPerEnvironment;
    const std::uint32_t logicalCapacity = 6u * kCapacity;
    const std::uint32_t maxTileCount = (padded + 31u) / 32u;
    const std::uint64_t scratchStrideWords = std::uint64_t(padded) * 2u;
    const std::uint32_t useHierarchicalPrefix = 0u;
    const std::uint32_t scanTileCount = (kNodes + 255u) / 256u;
    require(radixPassCount >= 1u && radixPassCount <= 8u,
            "native radix check requires one through eight complete key passes");
    const Result expected = buildLegacyReference(kNodes, kCapacity, environments);

    const auto contacts = packContacts(environments);
    std::vector<std::uint32_t> activeIndices(kEnvironments * kCapacity);
    std::vector<std::uint32_t> activeCounts(kEnvironments);
    for (std::uint32_t environment = 0u; environment < kEnvironments;
         ++environment) {
        require(environments[environment].activeIndices.size() == kCapacity,
                "fixture active index row must equal capacity");
        activeCounts[environment] = environments[environment].activeCount;
        std::copy(environments[environment].activeIndices.begin(),
                  environments[environment].activeIndices.end(),
                  activeIndices.begin() + environment * kCapacity);
    }
    id<MTLBuffer> dispatchBuffer = inputBuffer(device,
        std::vector<NMMatterDispatchGPU>{dispatch});
    id<MTLBuffer> contactsBuffer = inputBuffer(device, contacts);
    id<MTLBuffer> activeIndicesBuffer = inputBuffer(device, activeIndices);
    id<MTLBuffer> activeCountsBuffer = inputBuffer(device, activeCounts);
    id<MTLBuffer> sortKeys = zeroBuffer<std::uint64_t>(
        device, workspace.totalRecords, ~std::uint64_t{0});
    id<MTLBuffer> sortPayloads = zeroBuffer<std::uint32_t>(
        device, workspace.totalRecords, 0u);
    id<MTLBuffer> ranges = zeroBuffer<NMIncidenceRangeGPU>(
        device, kEnvironments * kNodes);
    id<MTLBuffer> legacyRanges = zeroBuffer<NMIncidenceRangeGPU>(
        device, kEnvironments * kNodes);
    id<MTLBuffer> incidence = zeroBuffer<std::uint32_t>(
        device, kEnvironments * kCapacity * 6u, kInvalid);
    id<MTLBuffer> legacyIncidence = zeroBuffer<std::uint32_t>(
        device, kEnvironments * kCapacity * 6u, kInvalid);
    id<MTLBuffer> statuses = zeroBuffer<NMMatterStatusGPU>(
        device, kEnvironments);
    id<MTLBuffer> legacyStatuses = zeroBuffer<NMMatterStatusGPU>(
        device, kEnvironments);
    id<MTLBuffer> radixRanges = zeroBuffer<NMIncidenceRangeGPU>(
        device, kEnvironments * kNodes);
    id<MTLBuffer> radixIncidence = zeroBuffer<std::uint32_t>(
        device, kEnvironments * logicalCapacity, kInvalid);
    id<MTLBuffer> radixStatuses = zeroBuffer<NMMatterStatusGPU>(
        device, kEnvironments);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    require(queue != nil, "Metal command queue allocation failed");
    id<MTLCommandBuffer> command = [queue commandBuffer];
    require(command != nil, "Metal command buffer allocation failed");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    require(encoder != nil, "Metal incidence encoder allocation failed");
    const NSUInteger sortThreads = workspace.totalRecords;
    const auto bindDispatch = [&] {
        [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
    };

    [encoder setComputePipelineState:emit];
    bindDispatch();
    [encoder setBuffer:contactsBuffer offset:0u atIndex:1u];
    [encoder setBuffer:activeIndicesBuffer offset:0u atIndex:2u];
    [encoder setBuffer:activeCountsBuffer offset:0u atIndex:3u];
    [encoder setBuffer:sortKeys offset:0u atIndex:4u];
    [encoder setBuffer:sortPayloads offset:0u atIndex:5u];
    [encoder setBytes:&padded length:sizeof(padded) atIndex:6u];
    [encoder dispatchThreads:MTLSizeMake(sortThreads, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    for (std::uint32_t width = 2u;; width *= 2u) {
        for (std::uint32_t distance = width / 2u; distance != 0u;
             distance /= 2u) {
            [encoder setComputePipelineState:bitonic];
            bindDispatch();
            [encoder setBytes:&padded length:sizeof(padded) atIndex:1u];
            [encoder setBytes:&width length:sizeof(width) atIndex:2u];
            [encoder setBytes:&distance length:sizeof(distance) atIndex:3u];
            [encoder setBuffer:sortKeys offset:0u atIndex:4u];
            [encoder setBuffer:sortPayloads offset:0u atIndex:5u];
            [encoder dispatchThreads:MTLSizeMake(sortThreads, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        if (width == padded) break;
    }

    [encoder setComputePipelineState:countSorted];
    bindDispatch();
    [encoder setBytes:&padded length:sizeof(padded) atIndex:1u];
    [encoder setBuffer:sortKeys offset:0u atIndex:2u];
    [encoder setBuffer:ranges offset:0u atIndex:3u];
    [encoder dispatchThreads:MTLSizeMake(kEnvironments * kNodes, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:scan];
    bindDispatch();
    [encoder setBuffer:ranges offset:0u atIndex:1u];
    [encoder setBuffer:statuses offset:0u atIndex:2u];
    [encoder dispatchThreadgroups:MTLSizeMake(kEnvironments, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:scatter];
    bindDispatch();
    [encoder setBytes:&padded length:sizeof(padded) atIndex:1u];
    [encoder setBuffer:sortKeys offset:0u atIndex:2u];
    [encoder setBuffer:sortPayloads offset:0u atIndex:3u];
    [encoder setBuffer:incidence offset:0u atIndex:4u];
    [encoder dispatchThreads:MTLSizeMake(sortThreads, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Re-emit the compact active prefix before radix sorting; the prior
    // bitonic pass has permuted the payload records in place.
    [encoder setComputePipelineState:emit];
    bindDispatch();
    [encoder setBuffer:contactsBuffer offset:0u atIndex:1u];
    [encoder setBuffer:activeIndicesBuffer offset:0u atIndex:2u];
    [encoder setBuffer:activeCountsBuffer offset:0u atIndex:3u];
    [encoder setBuffer:sortKeys offset:0u atIndex:4u];
    [encoder setBuffer:sortPayloads offset:0u atIndex:5u];
    [encoder setBytes:&padded length:sizeof(padded) atIndex:6u];
    [encoder dispatchThreads:MTLSizeMake(sortThreads, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    const NSUInteger radixGroups =
        NSUInteger(kEnvironments) * maxTileCount;
    for (std::uint32_t pass = 0u; pass < radixPassCount; ++pass) {
        [encoder setComputePipelineState:radixHistogram];
        bindDispatch();
        [encoder setBuffer:contactsBuffer offset:0u atIndex:1u];
        [encoder setBuffer:sortPayloads offset:0u atIndex:2u];
        [encoder setBuffer:radixIncidence offset:0u atIndex:3u];
        [encoder setBuffer:activeCountsBuffer offset:0u atIndex:4u];
        [encoder setBuffer:sortKeys offset:0u atIndex:5u];
        [encoder setBytes:&padded length:sizeof(padded) atIndex:6u];
        [encoder setBytes:&maxTileCount length:sizeof(maxTileCount) atIndex:7u];
        [encoder setBytes:&pass length:sizeof(pass) atIndex:8u];
        [encoder setBytes:&scratchStrideWords
                  length:sizeof(scratchStrideWords) atIndex:9u];
        [encoder dispatchThreadgroups:MTLSizeMake(radixGroups, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

        [encoder setComputePipelineState:radixPrefix];
        bindDispatch();
        [encoder setBuffer:activeCountsBuffer offset:0u atIndex:1u];
        [encoder setBuffer:sortKeys offset:0u atIndex:2u];
        [encoder setBytes:&maxTileCount length:sizeof(maxTileCount) atIndex:3u];
        [encoder setBytes:&scratchStrideWords
                  length:sizeof(scratchStrideWords) atIndex:4u];
        [encoder setBytes:&useHierarchicalPrefix
                  length:sizeof(useHierarchicalPrefix) atIndex:5u];
        [encoder dispatchThreadgroups:MTLSizeMake(kEnvironments, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

        [encoder setComputePipelineState:radixScatter];
        bindDispatch();
        [encoder setBuffer:contactsBuffer offset:0u atIndex:1u];
        [encoder setBuffer:sortPayloads offset:0u atIndex:2u];
        [encoder setBuffer:radixIncidence offset:0u atIndex:3u];
        [encoder setBuffer:activeCountsBuffer offset:0u atIndex:4u];
        [encoder setBuffer:sortKeys offset:0u atIndex:5u];
        [encoder setBuffer:sortPayloads offset:0u atIndex:6u];
        [encoder setBytes:&padded length:sizeof(padded) atIndex:7u];
        [encoder setBytes:&maxTileCount length:sizeof(maxTileCount) atIndex:8u];
        [encoder setBytes:&pass length:sizeof(pass) atIndex:9u];
        [encoder setBytes:&scratchStrideWords
                  length:sizeof(scratchStrideWords) atIndex:10u];
        [encoder setBytes:&useHierarchicalPrefix
                  length:sizeof(useHierarchicalPrefix) atIndex:11u];
        [encoder dispatchThreadgroups:MTLSizeMake(radixGroups, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    id<MTLBuffer> sortedRadixPayloads = (radixPassCount & 1u) != 0u
        ? radixIncidence : sortPayloads;
    [encoder setComputePipelineState:countRadix];
    bindDispatch();
    [encoder setBuffer:contactsBuffer offset:0u atIndex:1u];
    [encoder setBuffer:sortedRadixPayloads offset:0u atIndex:2u];
    [encoder setBuffer:activeCountsBuffer offset:0u atIndex:3u];
    [encoder setBuffer:radixRanges offset:0u atIndex:4u];
    const std::uint32_t sortedStride = (radixPassCount & 1u) != 0u
        ? logicalCapacity : padded;
    [encoder setBytes:&sortedStride length:sizeof(sortedStride) atIndex:5u];
    [encoder dispatchThreads:MTLSizeMake(kEnvironments * kNodes, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    const NSUInteger scanGroups =
        NSUInteger(kEnvironments) * scanTileCount;
    [encoder setComputePipelineState:scanTiles];
    bindDispatch();
    [encoder setBuffer:radixRanges offset:0u atIndex:1u];
    [encoder setBuffer:sortKeys offset:0u atIndex:2u];
    [encoder setBytes:&scanTileCount length:sizeof(scanTileCount) atIndex:3u];
    [encoder dispatchThreadgroups:MTLSizeMake(scanGroups, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:scanTileOffsets];
    bindDispatch();
    [encoder setBuffer:sortKeys offset:0u atIndex:1u];
    [encoder setBuffer:radixStatuses offset:0u atIndex:2u];
    [encoder setBytes:&scanTileCount length:sizeof(scanTileCount) atIndex:3u];
    [encoder dispatchThreadgroups:MTLSizeMake(kEnvironments, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:scanAddTileOffsets];
    bindDispatch();
    [encoder setBuffer:radixRanges offset:0u atIndex:1u];
    [encoder setBuffer:sortKeys offset:0u atIndex:2u];
    [encoder setBytes:&scanTileCount length:sizeof(scanTileCount) atIndex:3u];
    [encoder dispatchThreads:MTLSizeMake(kEnvironments * kNodes, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    if ((radixPassCount & 1u) == 0u) {
        [encoder setComputePipelineState:copyRadix];
        bindDispatch();
        [encoder setBuffer:sortPayloads offset:0u atIndex:1u];
        [encoder setBuffer:activeCountsBuffer offset:0u atIndex:2u];
        [encoder setBuffer:radixIncidence offset:0u atIndex:3u];
        [encoder setBytes:&padded length:sizeof(padded) atIndex:4u];
        [encoder dispatchThreads:MTLSizeMake(sortThreads, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    // Build the same prepared fixture with the production legacy GPU count /scan/
    // scatter pipeline, then compare both GPU outputs and the CPU legacy oracle.
    [encoder setComputePipelineState:countLegacy];
    bindDispatch();
    [encoder setBuffer:contactsBuffer offset:0u atIndex:1u];
    [encoder setBuffer:activeIndicesBuffer offset:0u atIndex:2u];
    [encoder setBuffer:activeCountsBuffer offset:0u atIndex:3u];
    [encoder setBuffer:legacyRanges offset:0u atIndex:4u];
    [encoder dispatchThreads:MTLSizeMake(kEnvironments * kNodes, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:scan];
    bindDispatch();
    [encoder setBuffer:legacyRanges offset:0u atIndex:1u];
    [encoder setBuffer:legacyStatuses offset:0u atIndex:2u];
    [encoder dispatchThreadgroups:MTLSizeMake(kEnvironments, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:scatterLegacy];
    bindDispatch();
    [encoder setBuffer:contactsBuffer offset:0u atIndex:1u];
    [encoder setBuffer:activeIndicesBuffer offset:0u atIndex:2u];
    [encoder setBuffer:activeCountsBuffer offset:0u atIndex:3u];
    [encoder setBuffer:legacyRanges offset:0u atIndex:4u];
    [encoder setBuffer:legacyIncidence offset:0u atIndex:5u];
    [encoder dispatchThreads:MTLSizeMake(kEnvironments * kNodes, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
            "native sorted incidence command buffer failed");

    Result actual;
    actual.ranges.resize(kEnvironments * kNodes);
    actual.incidence.resize(kEnvironments * kCapacity * 6u);
    const auto* gpuRanges = static_cast<const NMIncidenceRangeGPU*>(ranges.contents);
    const auto* gpuLegacyRanges = static_cast<const NMIncidenceRangeGPU*>(
        legacyRanges.contents);
    const auto* gpuIncidence = static_cast<const std::uint32_t*>(incidence.contents);
    const auto* gpuLegacyIncidence = static_cast<const std::uint32_t*>(
        legacyIncidence.contents);
    const auto* gpuStatuses = static_cast<const NMMatterStatusGPU*>(statuses.contents);
    const auto* gpuRadixRanges = static_cast<const NMIncidenceRangeGPU*>(
        radixRanges.contents);
    const auto* gpuRadixIncidence = static_cast<const std::uint32_t*>(
        radixIncidence.contents);
    const auto* gpuRadixStatuses = static_cast<const NMMatterStatusGPU*>(
        radixStatuses.contents);
    Result radixGPU;
    radixGPU.ranges.resize(kEnvironments * kNodes);
    radixGPU.incidence.resize(kEnvironments * logicalCapacity);
    Result legacyGPU;
    legacyGPU.ranges.resize(kEnvironments * kNodes);
    legacyGPU.incidence.resize(kEnvironments * kCapacity * 6u);
    for (std::size_t i = 0u; i < actual.ranges.size(); ++i) {
        actual.ranges[i] = {gpuRanges[i].first, gpuRanges[i].count};
        legacyGPU.ranges[i] = {gpuLegacyRanges[i].first,
                               gpuLegacyRanges[i].count};
        radixGPU.ranges[i] = {gpuRadixRanges[i].first,
                              gpuRadixRanges[i].count};
        require(gpuRanges[i].objectIndex == NM_INVALID_INDEX &&
                    gpuRanges[i].reserved == 0u &&
                    gpuLegacyRanges[i].objectIndex == NM_INVALID_INDEX &&
                    gpuLegacyRanges[i].reserved == 0u &&
                    gpuRadixRanges[i].objectIndex == NM_INVALID_INDEX &&
                    gpuRadixRanges[i].reserved == 0u,
                "native incidence range metadata differs from the legacy initializer");
    }
    std::copy(gpuIncidence, gpuIncidence + actual.incidence.size(),
              actual.incidence.begin());
    std::copy(gpuLegacyIncidence,
              gpuLegacyIncidence + legacyGPU.incidence.size(),
              legacyGPU.incidence.begin());
    std::copy(gpuRadixIncidence,
              gpuRadixIncidence + radixGPU.incidence.size(),
              radixGPU.incidence.begin());
    require(actual.ranges == expected.ranges &&
                actual.incidence == expected.incidence,
            "native sorted CSR differs from the complete CPU legacy output");
    require(legacyGPU.ranges == expected.ranges &&
                legacyGPU.incidence == expected.incidence,
            "native legacy CSR differs from the complete CPU reference output");
    require(radixGPU.ranges == expected.ranges &&
                radixGPU.incidence == expected.incidence,
            "native radix CSR differs from the complete CPU legacy output");
    require(radixGPU.ranges == legacyGPU.ranges &&
                radixGPU.incidence == legacyGPU.incidence,
            "native radix CSR differs from native legacy CSR");
    for (const NMMatterStatusGPU& status :
         std::span<const NMMatterStatusGPU>(gpuStatuses, kEnvironments))
        require(status.code == 0u,
                "native sorted incidence reported a capacity/status failure");
    for (const NMMatterStatusGPU& status :
         std::span<const NMMatterStatusGPU>(gpuRadixStatuses, kEnvironments))
        require(status.code == 0u,
                "native radix incidence reported a capacity/status failure");
    const auto* gpuLegacyStatuses = static_cast<const NMMatterStatusGPU*>(
        legacyStatuses.contents);
    for (const NMMatterStatusGPU& status :
         std::span<const NMMatterStatusGPU>(gpuLegacyStatuses, kEnvironments))
        require(status.code == 0u,
                "native legacy incidence reported a capacity/status failure");
}


void executeHierarchicalRadixCase(
    id<MTLDevice> device,
    const std::vector<std::uint32_t>& requestedActiveCounts,
    id<MTLComputePipelineState> emit,
    id<MTLComputePipelineState> histogram,
    id<MTLComputePipelineState> smallPrefix,
    id<MTLComputePipelineState> preparePrefix,
    id<MTLComputePipelineState> prefixBlocks,
    id<MTLComputePipelineState> prefixChunkOffsets,
    id<MTLComputePipelineState> prepareTiles,
    id<MTLComputePipelineState> scatter,
    id<MTLComputePipelineState> countRadix,
    id<MTLComputePipelineState> scanTiles,
    id<MTLComputePipelineState> scanTileOffsets,
    id<MTLComputePipelineState> scanAddTileOffsets
) {
    constexpr std::uint32_t capacity = 1366u;
    constexpr std::uint32_t nodeCount = 6u;
    constexpr std::uint32_t bucketCount = 16u;
    constexpr std::uint32_t chunkTiles = 256u;
    const std::uint32_t environmentCount =
        static_cast<std::uint32_t>(requestedActiveCounts.size());
    require(environmentCount == 1u || environmentCount == 2u,
            "hierarchical radix fixture requires one or two environments");
    std::vector<Environment> environments(environmentCount);
    std::vector<NMDeformableContactGPU> packedContacts(
        std::size_t(environmentCount) * capacity);
    std::vector<std::uint32_t> activeIndices(
        std::size_t(environmentCount) * capacity);
    for (std::uint32_t environment = 0u;
         environment < environmentCount; ++environment) {
        Environment& fixture = environments[environment];
        fixture.contacts.resize(capacity);
        fixture.activeIndices.resize(capacity);
        fixture.activeCount = requestedActiveCounts[environment];
        for (std::uint32_t slot = 0u; slot < capacity; ++slot) {
            std::array<std::uint32_t, 6> nodes{};
            for (std::uint32_t local = 0u; local < 6u; ++local) {
                const std::uint32_t pattern =
                    slot * 13u + local * 5u + environment * 3u;
                nodes[local] = pattern % 31u == 0u
                    ? kInvalid : (pattern % nodeCount);
            }
            const bool valid = (slot + 7u * environment) % 43u != 0u;
            fixture.contacts[slot] = contact(nodes, valid);
            const std::uint32_t activeSlot = slot;
            fixture.activeIndices[slot] =
                slot > 0u && slot % 37u == 0u ? slot - 1u : activeSlot;
            NMDeformableContactGPU& packed =
                packedContacts[std::size_t(environment) * capacity + slot];
            packed.nodes = {nodes[0], nodes[1], nodes[2], nodes[3]};
            packed.extraNodes = {nodes[4], nodes[5], 0u, 0u};
            packed.identity.w = valid ? kValid : 0u;
            activeIndices[std::size_t(environment) * capacity + slot] =
                fixture.activeIndices[slot];
        }
    }

    const std::uint64_t logicalRecords = std::uint64_t(6u) * capacity;
    const std::uint32_t padded = static_cast<std::uint32_t>(
        std::bit_ceil(logicalRecords));
    const std::uint32_t maxTileCount = (padded + 31u) / 32u;
    const std::uint32_t maxChunkCount =
        (maxTileCount + chunkTiles - 1u) / chunkTiles;
    const std::uint64_t histogramWords =
        std::uint64_t(maxTileCount) * bucketCount;
    const std::uint64_t requiredScratchWords =
        histogramWords + std::uint64_t(bucketCount) * maxChunkCount +
        bucketCount;
    const std::uint64_t keyStride = std::max<std::uint64_t>(
        padded, (requiredScratchWords + 1u) / 2u);
    const std::uint64_t scratchStrideWords = keyStride * 2u;
    require(scratchStrideWords >= requiredScratchWords &&
                scratchStrideWords == keyStride * 2u,
            "hierarchical radix per-environment scratch stride is too small");

    NMMatterDispatchGPU dispatch{};
    dispatch.environmentCount = environmentCount;
    dispatch.gridNodeCount = 2u;
    dispatch.femNodeCount = nodeCount - dispatch.gridNodeCount;
    dispatch.deformableContactCapacity = capacity;
    id<MTLBuffer> dispatchBuffer = inputBuffer(device,
        std::vector<NMMatterDispatchGPU>{dispatch});
    id<MTLBuffer> contactsBuffer = inputBuffer(device, packedContacts);
    id<MTLBuffer> activeIndicesBuffer = inputBuffer(device, activeIndices);
    id<MTLBuffer> activeCountsBuffer = inputBuffer(device, requestedActiveCounts);
    id<MTLBuffer> sortKeys = zeroBuffer<std::uint64_t>(
        device, keyStride * environmentCount, ~std::uint64_t{0});
    require(sortKeys.length / sizeof(std::uint32_t) >=
                scratchStrideWords * environmentCount,
            "hierarchical radix scratch writes exceed the allocated per-environment key buffer");
    id<MTLBuffer> sortPayloads = zeroBuffer<std::uint32_t>(
        device, padded * environmentCount, kInvalid);
    id<MTLBuffer> incidence = zeroBuffer<std::uint32_t>(
        device, logicalRecords * environmentCount, kInvalid);
    id<MTLBuffer> ranges = zeroBuffer<NMIncidenceRangeGPU>(
        device, std::size_t(nodeCount) * environmentCount);
    id<MTLBuffer> statuses = zeroBuffer<NMMatterStatusGPU>(
        device, environmentCount);
    id<MTLBuffer> prefixDispatch = zeroBuffer<NMIndirectDispatchGPU>(device, 1u);
    id<MTLBuffer> scatterDispatch = zeroBuffer<NMIndirectDispatchGPU>(device, 1u);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    require(queue != nil, "hierarchical radix command queue allocation failed");
    id<MTLCommandBuffer> command = [queue commandBuffer];
    require(command != nil, "hierarchical radix command buffer allocation failed");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    require(encoder != nil, "hierarchical radix encoder allocation failed");
    const auto bindDispatch = [&] {
        [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
    };
    const NSUInteger totalSortThreads =
        NSUInteger(padded) * environmentCount;
    const std::uint32_t radixPass = 0u;
    const std::uint32_t useHierarchical = 1u;
    const std::uint32_t scanTileCount =
        (nodeCount + 255u) / 256u;
    const std::uint32_t activeLimit = capacity;
    const std::uint32_t recordsPerActive = 6u;
    const std::uint32_t sortedStride =
        static_cast<std::uint32_t>(logicalRecords);

    [encoder setComputePipelineState:emit];
    bindDispatch();
    [encoder setBuffer:contactsBuffer offset:0u atIndex:1u];
    [encoder setBuffer:activeIndicesBuffer offset:0u atIndex:2u];
    [encoder setBuffer:activeCountsBuffer offset:0u atIndex:3u];
    [encoder setBuffer:sortKeys offset:0u atIndex:4u];
    [encoder setBuffer:sortPayloads offset:0u atIndex:5u];
    [encoder setBytes:&padded length:sizeof(padded) atIndex:6u];
    [encoder dispatchThreads:MTLSizeMake(totalSortThreads, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    const NSUInteger histogramGroups =
        NSUInteger(environmentCount) * maxTileCount;
    [encoder setComputePipelineState:histogram];
    bindDispatch();
    [encoder setBuffer:contactsBuffer offset:0u atIndex:1u];
    [encoder setBuffer:sortPayloads offset:0u atIndex:2u];
    [encoder setBuffer:incidence offset:0u atIndex:3u];
    [encoder setBuffer:activeCountsBuffer offset:0u atIndex:4u];
    [encoder setBuffer:sortKeys offset:0u atIndex:5u];
    [encoder setBytes:&padded length:sizeof(padded) atIndex:6u];
    [encoder setBytes:&maxTileCount length:sizeof(maxTileCount) atIndex:7u];
    [encoder setBytes:&radixPass length:sizeof(radixPass) atIndex:8u];
    [encoder setBytes:&scratchStrideWords
              length:sizeof(scratchStrideWords) atIndex:9u];
    [encoder dispatchThreadgroups:MTLSizeMake(histogramGroups, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:smallPrefix];
    bindDispatch();
    [encoder setBuffer:activeCountsBuffer offset:0u atIndex:1u];
    [encoder setBuffer:sortKeys offset:0u atIndex:2u];
    [encoder setBytes:&maxTileCount length:sizeof(maxTileCount) atIndex:3u];
    [encoder setBytes:&scratchStrideWords
              length:sizeof(scratchStrideWords) atIndex:4u];
    [encoder setBytes:&useHierarchical
              length:sizeof(useHierarchical) atIndex:5u];
    [encoder dispatchThreadgroups:MTLSizeMake(environmentCount, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    if (environmentCount == 1u) {
        [encoder setComputePipelineState:preparePrefix];
        [encoder setBuffer:activeCountsBuffer offset:0u atIndex:0u];
        [encoder setBuffer:prefixDispatch offset:0u atIndex:1u];
        [encoder setBytes:&activeLimit length:sizeof(activeLimit) atIndex:2u];
        [encoder setBytes:&recordsPerActive
                  length:sizeof(recordsPerActive) atIndex:3u];
        [encoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    [encoder setComputePipelineState:prefixBlocks];
    bindDispatch();
    [encoder setBuffer:activeCountsBuffer offset:0u atIndex:1u];
    [encoder setBuffer:sortKeys offset:0u atIndex:2u];
    [encoder setBytes:&maxTileCount length:sizeof(maxTileCount) atIndex:3u];
    [encoder setBytes:&scratchStrideWords
              length:sizeof(scratchStrideWords) atIndex:4u];
    if (environmentCount == 1u) {
        [encoder dispatchThreadgroupsWithIndirectBuffer:prefixDispatch
            indirectBufferOffset:0u
            threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    } else {
        const NSUInteger blockGroups =
            NSUInteger(environmentCount) * bucketCount * maxChunkCount;
        [encoder dispatchThreadgroups:MTLSizeMake(blockGroups, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    }
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:prefixChunkOffsets];
    bindDispatch();
    [encoder setBuffer:activeCountsBuffer offset:0u atIndex:1u];
    [encoder setBuffer:sortKeys offset:0u atIndex:2u];
    [encoder setBytes:&maxTileCount length:sizeof(maxTileCount) atIndex:3u];
    [encoder setBytes:&scratchStrideWords
              length:sizeof(scratchStrideWords) atIndex:4u];
    [encoder dispatchThreadgroups:MTLSizeMake(environmentCount, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    if (environmentCount == 1u) {
        [encoder setComputePipelineState:prepareTiles];
        [encoder setBuffer:activeCountsBuffer offset:0u atIndex:0u];
        [encoder setBuffer:scatterDispatch offset:0u atIndex:1u];
        [encoder setBytes:&activeLimit length:sizeof(activeLimit) atIndex:2u];
        [encoder setBytes:&recordsPerActive
                  length:sizeof(recordsPerActive) atIndex:3u];
        [encoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    [encoder setComputePipelineState:scatter];
    bindDispatch();
    [encoder setBuffer:contactsBuffer offset:0u atIndex:1u];
    [encoder setBuffer:sortPayloads offset:0u atIndex:2u];
    [encoder setBuffer:incidence offset:0u atIndex:3u];
    [encoder setBuffer:activeCountsBuffer offset:0u atIndex:4u];
    [encoder setBuffer:sortKeys offset:0u atIndex:5u];
    [encoder setBuffer:sortPayloads offset:0u atIndex:6u];
    [encoder setBytes:&padded length:sizeof(padded) atIndex:7u];
    [encoder setBytes:&maxTileCount length:sizeof(maxTileCount) atIndex:8u];
    [encoder setBytes:&radixPass length:sizeof(radixPass) atIndex:9u];
    [encoder setBytes:&scratchStrideWords
              length:sizeof(scratchStrideWords) atIndex:10u];
    [encoder setBytes:&useHierarchical
              length:sizeof(useHierarchical) atIndex:11u];
    if (environmentCount == 1u) {
        [encoder dispatchThreadgroupsWithIndirectBuffer:scatterDispatch
            indirectBufferOffset:0u
            threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    } else {
        [encoder dispatchThreadgroups:MTLSizeMake(histogramGroups, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    }
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:countRadix];
    bindDispatch();
    [encoder setBuffer:contactsBuffer offset:0u atIndex:1u];
    [encoder setBuffer:incidence offset:0u atIndex:2u];
    [encoder setBuffer:activeCountsBuffer offset:0u atIndex:3u];
    [encoder setBuffer:ranges offset:0u atIndex:4u];
    [encoder setBytes:&sortedStride length:sizeof(sortedStride) atIndex:5u];
    [encoder dispatchThreads:MTLSizeMake(
        NSUInteger(environmentCount) * nodeCount, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:scanTiles];
    bindDispatch();
    [encoder setBuffer:ranges offset:0u atIndex:1u];
    [encoder setBuffer:sortKeys offset:0u atIndex:2u];
    [encoder setBytes:&scanTileCount length:sizeof(scanTileCount) atIndex:3u];
    [encoder dispatchThreadgroups:MTLSizeMake(
        NSUInteger(environmentCount) * scanTileCount, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:scanTileOffsets];
    bindDispatch();
    [encoder setBuffer:sortKeys offset:0u atIndex:1u];
    [encoder setBuffer:statuses offset:0u atIndex:2u];
    [encoder setBytes:&scanTileCount length:sizeof(scanTileCount) atIndex:3u];
    [encoder dispatchThreadgroups:MTLSizeMake(environmentCount, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:scanAddTileOffsets];
    bindDispatch();
    [encoder setBuffer:ranges offset:0u atIndex:1u];
    [encoder setBuffer:sortKeys offset:0u atIndex:2u];
    [encoder setBytes:&scanTileCount length:sizeof(scanTileCount) atIndex:3u];
    [encoder dispatchThreads:MTLSizeMake(
        NSUInteger(environmentCount) * nodeCount, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
            "hierarchical radix command buffer failed");

    const Result expected = buildLegacyReference(
        nodeCount, capacity, environments);
    Result actual;
    actual.ranges.resize(std::size_t(environmentCount) * nodeCount);
    actual.incidence.resize(std::size_t(environmentCount) * logicalRecords);
    const auto* gpuRanges = static_cast<const NMIncidenceRangeGPU*>(ranges.contents);
    const auto* gpuIncidence = static_cast<const std::uint32_t*>(incidence.contents);
    for (std::size_t index = 0u; index < actual.ranges.size(); ++index) {
        actual.ranges[index] = {gpuRanges[index].first, gpuRanges[index].count};
        require(gpuRanges[index].objectIndex == NM_INVALID_INDEX &&
                    gpuRanges[index].reserved == 0u,
                "hierarchical radix range metadata changed");
    }
    std::copy(gpuIncidence, gpuIncidence + actual.incidence.size(),
              actual.incidence.begin());
    require(actual.ranges == expected.ranges &&
                actual.incidence == expected.incidence,
            "hierarchical radix ranges or duplicate payload ordering differ from the CPU legacy oracle");
    const auto* gpuStatuses = static_cast<const NMMatterStatusGPU*>(statuses.contents);
    for (const NMMatterStatusGPU& status :
         std::span<const NMMatterStatusGPU>(gpuStatuses, environmentCount))
        require(status.code == 0u,
                "hierarchical radix scan reported capacity/status failure");

    if (environmentCount == 1u) {
        const std::uint32_t activeCount =
            std::min(requestedActiveCounts[0], capacity);
        const std::uint32_t activeTiles =
            (6u * activeCount + 31u) / 32u;
        const std::uint32_t expectedPrefixGroups = activeTiles > chunkTiles
            ? bucketCount * ((activeTiles + chunkTiles - 1u) / chunkTiles)
            : 0u;
        const std::uint32_t expectedScatterGroups = activeTiles;
        const auto* gpuPrefixDispatch =
            static_cast<const NMIndirectDispatchGPU*>(prefixDispatch.contents);
        const auto* gpuScatterDispatch =
            static_cast<const NMIndirectDispatchGPU*>(scatterDispatch.contents);
        require(gpuPrefixDispatch[0].threadgroupsX == expectedPrefixGroups &&
                    gpuPrefixDispatch[0].threadgroupsY == 1u &&
                    gpuPrefixDispatch[0].threadgroupsZ == 1u,
                "GPU hierarchical prefix dispatch threshold/count is incorrect");
        require(gpuScatterDispatch[0].threadgroupsX == expectedScatterGroups &&
                    gpuScatterDispatch[0].threadgroupsY == 1u &&
                    gpuScatterDispatch[0].threadgroupsZ == 1u,
                "GPU active scatter dispatch differs from the active tile count");
    }
}

std::vector<Environment> explicitCase() {
    std::vector<Environment> environments(kEnvironments);
    environments[0].contacts = {
        contact({1u, 1u, 2u, 0u, 5u, kInvalid}),
        contact({0u, 0u, 0u, 0u, 0u, 0u}, false),
        contact({2u, 0u, 3u, 1u, 2u, 4u}),
        contact({4u, 4u, 4u, 4u, 4u, 4u}),
    };
    environments[0].activeIndices = {2u, 0u, 2u, 3u};
    environments[0].activeCount = 4u;
    environments[1].contacts = {
        contact({0u, 1u, 2u, 3u, 4u, 5u}, false),
        contact({0u, 2u, 2u, 3u, 4u, 4u}),
        contact({5u, 5u, 5u, 5u, 5u, 5u}),
        contact({1u, 3u, 1u, 3u, kInvalid, 0u}),
    };
    environments[1].activeIndices = {1u, 0u, 3u, 2u};
    environments[1].activeCount = 2u;
    return environments;
}

std::vector<Environment> zeroActiveCase() {
    std::vector<Environment> environments = explicitCase();
    for (Environment& environment : environments) environment.activeCount = 0u;
    return environments;
}

std::vector<Environment> invalidSlotCase() {
    std::vector<Environment> environments = explicitCase();
    environments[0].activeIndices = {kInvalid, kInvalid, kInvalid, kInvalid};
    environments[0].activeCount = 99u;
    environments[1].activeIndices = {3u, 0u, 1u, 2u};
    environments[1].activeCount = 99u;
    return environments;
}
std::uint32_t spreadMorton10(const std::uint32_t value) {
    std::uint32_t spread = value & 0x000003ffu;
    spread = (spread | (spread << 16u)) & 0x030000ffu;
    spread = (spread | (spread << 8u)) & 0x0300f00fu;
    spread = (spread | (spread << 4u)) & 0x030c30c3u;
    spread = (spread | (spread << 2u)) & 0x09249249u;
    return spread;
}

std::uint32_t surfaceMortonKeyCPU(
    const std::array<float, 3>& center,
    const std::array<float, 3>& minimum,
    const std::array<float, 3>& maximum
) {
    std::array<std::uint32_t, 3> quantized{};
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
        const float inverseExtent = 1.0f /
            std::max(maximum[axis] - minimum[axis], 1.0e-12f);
        const float normalized = (center[axis] - minimum[axis]) * inverseExtent;
        quantized[axis] = static_cast<std::uint32_t>(std::clamp(
            normalized * 1023.0f, 0.0f, 1023.0f));
    }
    return spreadMorton10(quantized[0]) |
        (spreadMorton10(quantized[1]) << 1u) |
        (spreadMorton10(quantized[2]) << 2u);
}

std::vector<NMContinuumSurfacePrimitiveGPU> makeSurfaceFixture(
    const bool allInactive
) {
    constexpr std::uint32_t primitiveCount = 70u;
    std::vector<NMContinuumSurfacePrimitiveGPU> primitives(
        kEnvironments * primitiveCount);
    for (std::uint32_t environment = 0u; environment < kEnvironments;
         ++environment) {
        for (std::uint32_t local = 0u; local < primitiveCount; ++local) {
            const std::uint32_t pattern = (local + 3u * environment) % 9u;
            const float x = float(pattern % 3u);
            const float y = float(pattern / 3u);
            const float z = float((2u * pattern + environment) % 5u);
            const bool active = !allInactive &&
                (environment == 0u ? local % 5u != 0u : local % 4u != 0u);
            NMContinuumSurfacePrimitiveGPU& primitive =
                primitives[environment * primitiveCount + local];
            primitive.nodesAndObject = {local, local + 1u, local + 2u, 0u};
            primitive.identity = {
                local, NM_CONTINUUM_SURFACE_TRIANGLE, 0u,
                active ? NM_TOPOLOGY_ACTIVE : 0u};
            primitive.boundsMinimum = {x - 0.5f, y - 0.25f, z - 0.75f, 0.0f};
            primitive.boundsMaximum = {x + 0.5f, y + 0.25f, z + 0.75f, 0.0f};
        }
    }
    return primitives;
}

void executeSurfaceRadixCase(
    id<MTLDevice> device,
    const bool allInactive,
    id<MTLComputePipelineState> legacySort,
    id<MTLComputePipelineState> emit,
    id<MTLComputePipelineState> tileBounds,
    id<MTLComputePipelineState> boundsReduce,
    id<MTLComputePipelineState> scanTileCounts,
    id<MTLComputePipelineState> scanBlockOffsets,
    id<MTLComputePipelineState> scanAddBlockOffsets,
    id<MTLComputePipelineState> emitParallel,
    id<MTLComputePipelineState> prepareRadixDispatch,
    id<MTLComputePipelineState> histogram,
    id<MTLComputePipelineState> prefix,
    id<MTLComputePipelineState> scatter
) {
    constexpr std::uint32_t primitiveCount = 70u;
    constexpr std::uint32_t radixBucketCount = 16u;
    constexpr std::uint32_t radixPassCount = 8u;
    NMMatterDispatchGPU dispatch{};
    dispatch.environmentCount = kEnvironments;
    dispatch.gridNodeCount = 2u;
    dispatch.tetrahedronCount = (primitiveCount - dispatch.gridNodeCount) / 4u;
    const std::uint32_t maxTileCount = primitiveCount / 32u +
        (primitiveCount % 32u != 0u ? 1u : 0u);
    const std::uint32_t scanBlockCount = maxTileCount / 256u +
        (maxTileCount % 256u != 0u ? 1u : 0u);
    const std::uint32_t scratchStride = std::max(
        primitiveCount, maxTileCount * radixBucketCount);
    const std::uint32_t useArmijoSurfaceCache = 0u;
    const auto primitives = makeSurfaceFixture(allInactive);
    std::vector<std::uint32_t> cacheFlags(kEnvironments, 0u);

    id<MTLBuffer> dispatchBuffer = inputBuffer(device,
        std::vector<NMMatterDispatchGPU>{dispatch});
    id<MTLBuffer> primitivesBuffer = inputBuffer(device, primitives);
    id<MTLBuffer> cacheFlagsBuffer = inputBuffer(device, cacheFlags);
    id<MTLBuffer> legacyKeysA = zeroBuffer<std::uint32_t>(
        device, kEnvironments * primitiveCount, kInvalid);
    id<MTLBuffer> legacyKeysB = zeroBuffer<std::uint32_t>(
        device, kEnvironments * primitiveCount, kInvalid);
    id<MTLBuffer> legacyIndicesA = zeroBuffer<std::uint32_t>(
        device, kEnvironments * primitiveCount, kInvalid);
    id<MTLBuffer> legacyIndicesB = zeroBuffer<std::uint32_t>(
        device, kEnvironments * primitiveCount, kInvalid);
    id<MTLBuffer> legacyCounts = zeroBuffer<std::uint32_t>(device, kEnvironments);
    id<MTLBuffer> keysA = zeroBuffer<std::uint32_t>(
        device, kEnvironments * primitiveCount, kInvalid);
    id<MTLBuffer> scratch = zeroBuffer<std::uint32_t>(
        device, kEnvironments * scratchStride, 0u);
    id<MTLBuffer> indicesA = zeroBuffer<std::uint32_t>(
        device, kEnvironments * primitiveCount, kInvalid);
    id<MTLBuffer> indicesB = zeroBuffer<std::uint32_t>(
        device, kEnvironments * primitiveCount, kInvalid);
    id<MTLBuffer> activeCounts = zeroBuffer<std::uint32_t>(
        device, kEnvironments);
    id<MTLBuffer> parallelKeysA = zeroBuffer<std::uint32_t>(
        device, kEnvironments * primitiveCount, kInvalid);
    id<MTLBuffer> parallelScratch = zeroBuffer<std::uint32_t>(
        device, kEnvironments * scratchStride, 0u);
    id<MTLBuffer> parallelBounds = zeroBuffer<nm_float4>(
        device, kEnvironments * (2u * maxTileCount + 2u));
    id<MTLBuffer> parallelIndicesA = zeroBuffer<std::uint32_t>(
        device, kEnvironments * primitiveCount, kInvalid);
    id<MTLBuffer> parallelIndicesB = zeroBuffer<std::uint32_t>(
        device, kEnvironments * primitiveCount, kInvalid);
    id<MTLBuffer> parallelActiveCounts = zeroBuffer<std::uint32_t>(
        device, kEnvironments);
    id<MTLBuffer> preparedDispatch = zeroBuffer<NMIndirectDispatchGPU>(
        device, 1u);

    id<MTLCommandQueue> queue = [device newCommandQueue];
    require(queue != nil, "surface radix queue allocation failed");
    id<MTLCommandBuffer> command = [queue commandBuffer];
    require(command != nil, "surface radix command buffer allocation failed");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    require(encoder != nil, "surface radix encoder allocation failed");
    const auto bindDispatch = [&] {
        [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
    };

    [encoder setComputePipelineState:legacySort];
    bindDispatch();
    [encoder setBuffer:primitivesBuffer offset:0u atIndex:1u];
    [encoder setBuffer:legacyKeysA offset:0u atIndex:2u];
    [encoder setBuffer:legacyKeysB offset:0u atIndex:3u];
    [encoder setBuffer:legacyIndicesA offset:0u atIndex:4u];
    [encoder setBuffer:legacyIndicesB offset:0u atIndex:5u];
    [encoder setBuffer:legacyCounts offset:0u atIndex:6u];
    [encoder setBuffer:cacheFlagsBuffer offset:0u atIndex:7u];
    [encoder setBytes:&useArmijoSurfaceCache
              length:sizeof(useArmijoSurfaceCache) atIndex:8u];
    [encoder dispatchThreadgroups:MTLSizeMake(kEnvironments, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:emit];
    bindDispatch();
    [encoder setBuffer:primitivesBuffer offset:0u atIndex:1u];
    [encoder setBuffer:keysA offset:0u atIndex:2u];
    [encoder setBuffer:scratch offset:0u atIndex:3u];
    [encoder setBuffer:indicesA offset:0u atIndex:4u];
    [encoder setBuffer:indicesB offset:0u atIndex:5u];
    [encoder setBuffer:activeCounts offset:0u atIndex:6u];
    [encoder setBuffer:cacheFlagsBuffer offset:0u atIndex:7u];
    [encoder setBytes:&useArmijoSurfaceCache
              length:sizeof(useArmijoSurfaceCache) atIndex:8u];
    [encoder dispatchThreadgroups:MTLSizeMake(kEnvironments, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    const NSUInteger radixGroups =
        NSUInteger(kEnvironments) * maxTileCount;
    for (std::uint32_t pass = 0u; pass < radixPassCount; ++pass) {
        [encoder setComputePipelineState:histogram];
        bindDispatch();
        [encoder setBuffer:keysA offset:0u atIndex:1u];
        [encoder setBuffer:indicesA offset:0u atIndex:2u];
        [encoder setBuffer:indicesB offset:0u atIndex:3u];
        [encoder setBuffer:activeCounts offset:0u atIndex:4u];
        [encoder setBuffer:scratch offset:0u atIndex:5u];
        [encoder setBuffer:cacheFlagsBuffer offset:0u atIndex:6u];
        [encoder setBytes:&useArmijoSurfaceCache
                  length:sizeof(useArmijoSurfaceCache) atIndex:7u];
        [encoder setBytes:&maxTileCount length:sizeof(maxTileCount) atIndex:8u];
        [encoder setBytes:&scratchStride length:sizeof(scratchStride) atIndex:9u];
        [encoder setBytes:&pass length:sizeof(pass) atIndex:10u];
        [encoder dispatchThreadgroups:MTLSizeMake(radixGroups, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

        [encoder setComputePipelineState:prefix];
        bindDispatch();
        [encoder setBuffer:activeCounts offset:0u atIndex:1u];
        [encoder setBuffer:scratch offset:0u atIndex:2u];
        [encoder setBuffer:cacheFlagsBuffer offset:0u atIndex:3u];
        [encoder setBytes:&useArmijoSurfaceCache
                  length:sizeof(useArmijoSurfaceCache) atIndex:4u];
        [encoder setBytes:&maxTileCount length:sizeof(maxTileCount) atIndex:5u];
        [encoder setBytes:&scratchStride length:sizeof(scratchStride) atIndex:6u];
        [encoder dispatchThreadgroups:MTLSizeMake(kEnvironments, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

        [encoder setComputePipelineState:scatter];
        bindDispatch();
        [encoder setBuffer:keysA offset:0u atIndex:1u];
        [encoder setBuffer:indicesA offset:0u atIndex:2u];
        [encoder setBuffer:indicesB offset:0u atIndex:3u];
        [encoder setBuffer:scratch offset:0u atIndex:4u];
        [encoder setBuffer:activeCounts offset:0u atIndex:5u];
        [encoder setBuffer:cacheFlagsBuffer offset:0u atIndex:6u];
        [encoder setBytes:&useArmijoSurfaceCache
                  length:sizeof(useArmijoSurfaceCache) atIndex:7u];
        [encoder setBytes:&maxTileCount length:sizeof(maxTileCount) atIndex:8u];
        [encoder setBytes:&scratchStride length:sizeof(scratchStride) atIndex:9u];
        [encoder setBytes:&pass length:sizeof(pass) atIndex:10u];
        [encoder dispatchThreadgroups:MTLSizeMake(radixGroups, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    const NSUInteger tileGroups = NSUInteger(kEnvironments) * maxTileCount;
    [encoder setComputePipelineState:tileBounds];
    bindDispatch();
    [encoder setBuffer:primitivesBuffer offset:0u atIndex:1u];
    [encoder setBuffer:parallelBounds offset:0u atIndex:2u];
    [encoder setBuffer:parallelScratch offset:0u atIndex:3u];
    [encoder setBuffer:cacheFlagsBuffer offset:0u atIndex:4u];
    [encoder setBytes:&useArmijoSurfaceCache
              length:sizeof(useArmijoSurfaceCache) atIndex:5u];
    [encoder setBytes:&maxTileCount length:sizeof(maxTileCount) atIndex:6u];
    [encoder setBytes:&scratchStride length:sizeof(scratchStride) atIndex:7u];
    [encoder dispatchThreadgroups:MTLSizeMake(tileGroups, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:boundsReduce];
    bindDispatch();
    [encoder setBuffer:parallelBounds offset:0u atIndex:1u];
    [encoder setBuffer:cacheFlagsBuffer offset:0u atIndex:2u];
    [encoder setBytes:&useArmijoSurfaceCache
              length:sizeof(useArmijoSurfaceCache) atIndex:3u];
    [encoder setBytes:&maxTileCount length:sizeof(maxTileCount) atIndex:4u];
    [encoder dispatchThreadgroups:MTLSizeMake(kEnvironments, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:scanTileCounts];
    bindDispatch();
    [encoder setBuffer:parallelScratch offset:0u atIndex:1u];
    [encoder setBuffer:cacheFlagsBuffer offset:0u atIndex:2u];
    [encoder setBytes:&useArmijoSurfaceCache
              length:sizeof(useArmijoSurfaceCache) atIndex:3u];
    [encoder setBytes:&maxTileCount length:sizeof(maxTileCount) atIndex:4u];
    [encoder setBytes:&scanBlockCount length:sizeof(scanBlockCount) atIndex:5u];
    [encoder setBytes:&scratchStride length:sizeof(scratchStride) atIndex:6u];
    [encoder dispatchThreadgroups:MTLSizeMake(
            NSUInteger(kEnvironments) * scanBlockCount, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:scanBlockOffsets];
    bindDispatch();
    [encoder setBuffer:parallelScratch offset:0u atIndex:1u];
    [encoder setBuffer:parallelActiveCounts offset:0u atIndex:2u];
    [encoder setBuffer:cacheFlagsBuffer offset:0u atIndex:3u];
    [encoder setBytes:&useArmijoSurfaceCache
              length:sizeof(useArmijoSurfaceCache) atIndex:4u];
    [encoder setBytes:&maxTileCount length:sizeof(maxTileCount) atIndex:5u];
    [encoder setBytes:&scanBlockCount length:sizeof(scanBlockCount) atIndex:6u];
    [encoder setBytes:&scratchStride length:sizeof(scratchStride) atIndex:7u];
    [encoder dispatchThreadgroups:MTLSizeMake(kEnvironments, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:scanAddBlockOffsets];
    bindDispatch();
    [encoder setBuffer:parallelScratch offset:0u atIndex:1u];
    [encoder setBuffer:cacheFlagsBuffer offset:0u atIndex:2u];
    [encoder setBytes:&useArmijoSurfaceCache
              length:sizeof(useArmijoSurfaceCache) atIndex:3u];
    [encoder setBytes:&maxTileCount length:sizeof(maxTileCount) atIndex:4u];
    [encoder setBytes:&scanBlockCount length:sizeof(scanBlockCount) atIndex:5u];
    [encoder setBytes:&scratchStride length:sizeof(scratchStride) atIndex:6u];
    [encoder dispatchThreads:MTLSizeMake(tileGroups, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:emitParallel];
    bindDispatch();
    [encoder setBuffer:primitivesBuffer offset:0u atIndex:1u];
    [encoder setBuffer:parallelKeysA offset:0u atIndex:2u];
    [encoder setBuffer:parallelBounds offset:0u atIndex:3u];
    [encoder setBuffer:parallelScratch offset:0u atIndex:4u];
    [encoder setBuffer:parallelIndicesA offset:0u atIndex:5u];
    [encoder setBuffer:cacheFlagsBuffer offset:0u atIndex:6u];
    [encoder setBytes:&useArmijoSurfaceCache
              length:sizeof(useArmijoSurfaceCache) atIndex:7u];
    [encoder setBytes:&maxTileCount length:sizeof(maxTileCount) atIndex:8u];
    [encoder setBytes:&scratchStride length:sizeof(scratchStride) atIndex:9u];
    [encoder dispatchThreadgroups:MTLSizeMake(tileGroups, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    const std::uint32_t activeCountLimit = primitiveCount;
    const std::uint32_t recordsPerActive = 1u;
    [encoder setComputePipelineState:prepareRadixDispatch];
    [encoder setBuffer:parallelActiveCounts offset:0u atIndex:0u];
    [encoder setBuffer:preparedDispatch offset:0u atIndex:1u];
    [encoder setBytes:&activeCountLimit length:sizeof(activeCountLimit) atIndex:2u];
    [encoder setBytes:&recordsPerActive length:sizeof(recordsPerActive) atIndex:3u];
    [encoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    for (std::uint32_t pass = 0u; pass < radixPassCount; ++pass) {
        [encoder setComputePipelineState:histogram];
        bindDispatch();
        [encoder setBuffer:parallelKeysA offset:0u atIndex:1u];
        [encoder setBuffer:parallelIndicesA offset:0u atIndex:2u];
        [encoder setBuffer:parallelIndicesB offset:0u atIndex:3u];
        [encoder setBuffer:parallelActiveCounts offset:0u atIndex:4u];
        [encoder setBuffer:parallelScratch offset:0u atIndex:5u];
        [encoder setBuffer:cacheFlagsBuffer offset:0u atIndex:6u];
        [encoder setBytes:&useArmijoSurfaceCache
                  length:sizeof(useArmijoSurfaceCache) atIndex:7u];
        [encoder setBytes:&maxTileCount length:sizeof(maxTileCount) atIndex:8u];
        [encoder setBytes:&scratchStride length:sizeof(scratchStride) atIndex:9u];
        [encoder setBytes:&pass length:sizeof(pass) atIndex:10u];
        [encoder dispatchThreadgroups:MTLSizeMake(tileGroups, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

        [encoder setComputePipelineState:prefix];
        bindDispatch();
        [encoder setBuffer:parallelActiveCounts offset:0u atIndex:1u];
        [encoder setBuffer:parallelScratch offset:0u atIndex:2u];
        [encoder setBuffer:cacheFlagsBuffer offset:0u atIndex:3u];
        [encoder setBytes:&useArmijoSurfaceCache
                  length:sizeof(useArmijoSurfaceCache) atIndex:4u];
        [encoder setBytes:&maxTileCount length:sizeof(maxTileCount) atIndex:5u];
        [encoder setBytes:&scratchStride length:sizeof(scratchStride) atIndex:6u];
        [encoder dispatchThreadgroups:MTLSizeMake(kEnvironments, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

        [encoder setComputePipelineState:scatter];
        bindDispatch();
        [encoder setBuffer:parallelKeysA offset:0u atIndex:1u];
        [encoder setBuffer:parallelIndicesA offset:0u atIndex:2u];
        [encoder setBuffer:parallelIndicesB offset:0u atIndex:3u];
        [encoder setBuffer:parallelScratch offset:0u atIndex:4u];
        [encoder setBuffer:parallelActiveCounts offset:0u atIndex:5u];
        [encoder setBuffer:cacheFlagsBuffer offset:0u atIndex:6u];
        [encoder setBytes:&useArmijoSurfaceCache
                  length:sizeof(useArmijoSurfaceCache) atIndex:7u];
        [encoder setBytes:&maxTileCount length:sizeof(maxTileCount) atIndex:8u];
        [encoder setBytes:&scratchStride length:sizeof(scratchStride) atIndex:9u];
        [encoder setBytes:&pass length:sizeof(pass) atIndex:10u];
        [encoder dispatchThreadgroups:MTLSizeMake(tileGroups, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
            "native surface Morton radix command buffer failed");

    const auto* oldCount = static_cast<const std::uint32_t*>(legacyCounts.contents);
    const auto* newCount = static_cast<const std::uint32_t*>(activeCounts.contents);
    const auto* oldIndices = static_cast<const std::uint32_t*>(legacyIndicesA.contents);
    const auto* newIndices = static_cast<const std::uint32_t*>(indicesA.contents);
    const auto* parallelCount = static_cast<const std::uint32_t*>(
        parallelActiveCounts.contents);
    const auto* parallelIndices = static_cast<const std::uint32_t*>(
        parallelIndicesA.contents);
    const auto* prepared = static_cast<const NMIndirectDispatchGPU*>(
        preparedDispatch.contents);
    require(prepared[0].threadgroupsX ==
                (parallelCount[0] + 31u) / 32u &&
                prepared[0].threadgroupsY == 1u &&
                prepared[0].threadgroupsZ == 1u &&
                prepared[0].reserved == parallelCount[0],
            "GPU radix indirect group preparation differs from active surface count");
    for (std::uint32_t environment = 0u; environment < kEnvironments;
         ++environment) {
        const std::size_t base = std::size_t(environment) * primitiveCount;
        std::array<float, 3> minimum{
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity()};
        std::array<float, 3> maximum{
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity()};
        std::vector<std::pair<std::uint32_t, std::uint32_t>> keyed;
        for (std::uint32_t local = 0u; local < primitiveCount; ++local) {
            const NMContinuumSurfacePrimitiveGPU& primitive = primitives[base + local];
            if ((primitive.identity.w & NM_TOPOLOGY_ACTIVE) == 0u) continue;
            const std::array<float, 3> center{
                0.5f * (primitive.boundsMinimum.x + primitive.boundsMaximum.x),
                0.5f * (primitive.boundsMinimum.y + primitive.boundsMaximum.y),
                0.5f * (primitive.boundsMinimum.z + primitive.boundsMaximum.z)};
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                minimum[axis] = std::min(minimum[axis], center[axis]);
                maximum[axis] = std::max(maximum[axis], center[axis]);
            }
        }
        for (std::uint32_t local = 0u; local < primitiveCount; ++local) {
            const NMContinuumSurfacePrimitiveGPU& primitive = primitives[base + local];
            if ((primitive.identity.w & NM_TOPOLOGY_ACTIVE) == 0u) continue;
            const std::array<float, 3> center{
                0.5f * (primitive.boundsMinimum.x + primitive.boundsMaximum.x),
                0.5f * (primitive.boundsMinimum.y + primitive.boundsMaximum.y),
                0.5f * (primitive.boundsMinimum.z + primitive.boundsMaximum.z)};
            keyed.emplace_back(surfaceMortonKeyCPU(center, minimum, maximum), local);
        }
        std::stable_sort(keyed.begin(), keyed.end(),
            [](const auto& left, const auto& right) {
                return left.first < right.first;
            });
        require(oldCount[environment] == keyed.size() &&
                    newCount[environment] == keyed.size() &&
                    parallelCount[environment] == keyed.size(),
                "native surface active count differs from fixture activity");
        for (std::size_t index = 0u; index < keyed.size(); ++index) {
            const std::uint32_t expected = keyed[index].second;
            require(oldIndices[base + index] == expected,
                    "legacy native surface Morton order differs from stable CPU oracle");
            require(newIndices[base + index] == expected,
                    "radix native surface Morton order differs from stable CPU oracle");
            require(newIndices[base + index] == oldIndices[base + index],
                    "radix native surface order differs from legacy Morton order");
            require(parallelIndices[base + index] == expected,
                    "tiled radix native surface order differs from stable CPU oracle");
        }
    }
}

}

int main() {
    @autoreleasepool {
        try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "Apple Metal device is unavailable");
            NSError* error = nil;
            NSURL* url = [NSURL fileURLWithPath:
                [NSString stringWithUTF8String:NUMI_MATTER_METALLIB]];
            id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
            require(library != nil, error != nil
                ? std::string("cannot load Matter metallib: ") +
                    [[error localizedDescription] UTF8String]
                : "cannot load Matter metallib");
            const auto emit = makePipeline(device, library,
                "nm_contact_emit_deformable_node_corners");
            const auto bitonic = makePipeline(device, library,
                "nm_contact_bitonic_sort_deformable_node_corners");
            const auto countSorted = makePipeline(device, library,
                "nm_contact_count_deformable_node_incidence_sorted");
            const auto countLegacy = makePipeline(device, library,
                "nm_contact_count_deformable_node_incidence");
            const auto scan = makePipeline(device, library,
                "nm_contact_scan_deformable_node_incidence");
            const auto scatter = makePipeline(device, library,
                "nm_contact_scatter_deformable_node_incidence_sorted");
            const auto scatterLegacy = makePipeline(device, library,
                "nm_contact_scatter_deformable_node_incidence");
            const auto radixHistogram = makePipeline(device, library,
                "nm_contact_deformable_node_radix_histogram");
            const auto radixPrefix = makePipeline(device, library,
                "nm_contact_deformable_node_radix_prefix");
            const auto radixPrefixPrepare = makePipeline(device, library,
                "nm_contact_prepare_radix_prefix_dispatch");
            const auto radixPrefixBlocks = makePipeline(device, library,
                "nm_contact_deformable_node_radix_prefix_blocks");
            const auto radixPrefixChunkOffsets = makePipeline(device, library,
                "nm_contact_deformable_node_radix_prefix_chunk_offsets");
            const auto radixScatter = makePipeline(device, library,
                "nm_contact_deformable_node_radix_scatter");
            const auto countRadix = makePipeline(device, library,
                "nm_contact_count_deformable_node_incidence_radix");
            const auto copyRadix = makePipeline(device, library,
                "nm_contact_copy_deformable_node_radix_result");
            const auto scanTiles = makePipeline(device, library,
                "nm_contact_scan_deformable_node_incidence_tiles");
            const auto scanTileOffsets = makePipeline(device, library,
                "nm_contact_scan_deformable_node_incidence_tile_offsets");
            const auto scanAddTileOffsets = makePipeline(device, library,
                "nm_contact_scan_deformable_node_incidence_add_tile_offsets");
            const auto surfaceLegacySort = makePipeline(device, library,
                "nm_contact_sort_surface_primitives");
            const auto surfaceEmit = makePipeline(device, library,
                "nm_contact_emit_surface_morton_records");
            const auto surfaceTileBounds = makePipeline(device, library,
                "nm_contact_surface_morton_tile_bounds");
            const auto surfaceBoundsReduce = makePipeline(device, library,
                "nm_contact_surface_morton_bounds_reduce");
            const auto surfaceScanTileCounts = makePipeline(device, library,
                "nm_contact_surface_morton_scan_tile_counts");
            const auto surfaceScanBlockOffsets = makePipeline(device, library,
                "nm_contact_surface_morton_scan_block_offsets");
            const auto surfaceScanAddBlockOffsets = makePipeline(device, library,
                "nm_contact_surface_morton_scan_add_block_offsets");
            const auto surfaceEmitParallel = makePipeline(device, library,
                "nm_contact_emit_surface_morton_records_parallel");
            const auto prepareRadixDispatch = makePipeline(device, library,
                "nm_contact_prepare_radix_tile_dispatch");
            const auto surfaceRadixHistogram = makePipeline(device, library,
                "nm_contact_surface_morton_radix_histogram");
            const auto surfaceRadixPrefix = makePipeline(device, library,
                "nm_contact_surface_morton_radix_prefix");
            const auto surfaceRadixScatter = makePipeline(device, library,
                "nm_contact_surface_morton_radix_scatter");

            executeSurfaceRadixCase(device, false, surfaceLegacySort,
                surfaceEmit, surfaceTileBounds, surfaceBoundsReduce,
                surfaceScanTileCounts, surfaceScanBlockOffsets,
                surfaceScanAddBlockOffsets, surfaceEmitParallel,
                prepareRadixDispatch, surfaceRadixHistogram, surfaceRadixPrefix, surfaceRadixScatter);
            ++checks;
            executeSurfaceRadixCase(device, true, surfaceLegacySort,
                surfaceEmit, surfaceTileBounds, surfaceBoundsReduce,
                surfaceScanTileCounts, surfaceScanBlockOffsets,
                surfaceScanAddBlockOffsets, surfaceEmitParallel,
                prepareRadixDispatch, surfaceRadixHistogram, surfaceRadixPrefix, surfaceRadixScatter);
            ++checks;

            executeCase(device, explicitCase(), emit, bitonic, countSorted,
                        countLegacy, scan, scatter, scatterLegacy,
                        radixHistogram, radixPrefix, radixScatter, countRadix,
                        copyRadix, scanTiles, scanTileOffsets, scanAddTileOffsets,
                        stableRadixPassCount(kNodes));
            ++checks;
            // Preserve the previous seven-pass control for parity coverage;
            // the dynamic case above now exercises the production width.
            executeCase(device, explicitCase(), emit, bitonic, countSorted,
                        countLegacy, scan, scatter, scatterLegacy,
                        radixHistogram, radixPrefix, radixScatter, countRadix,
                        copyRadix, scanTiles, scanTileOffsets, scanAddTileOffsets,
                        7u);
            ++checks;
            // Force the full-width fallback on the small fixture. The high
            // key digit is zero here, while this exercises the even-pass
            // ping-pong parity and final copy used for large node domains.
            executeCase(device, explicitCase(), emit, bitonic, countSorted,
                        countLegacy, scan, scatter, scatterLegacy,
                        radixHistogram, radixPrefix, radixScatter, countRadix,
                        copyRadix, scanTiles, scanTileOffsets, scanAddTileOffsets,
                        8u);
            ++checks;
            executeCase(device, zeroActiveCase(), emit, bitonic, countSorted,
                        countLegacy, scan, scatter, scatterLegacy,
                        radixHistogram, radixPrefix, radixScatter, countRadix,
                        copyRadix, scanTiles, scanTileOffsets, scanAddTileOffsets,
                        7u);
            ++checks;
            executeCase(device, invalidSlotCase(), emit, bitonic, countSorted,
                        countLegacy, scan, scatter, scatterLegacy,
                        radixHistogram, radixPrefix, radixScatter, countRadix,
                        copyRadix, scanTiles, scanTileOffsets, scanAddTileOffsets,
                        7u);
            ++checks;
            executeHierarchicalRadixCase(device, {1365u}, emit, radixHistogram,
                radixPrefix, radixPrefixPrepare, radixPrefixBlocks,
                radixPrefixChunkOffsets, prepareRadixDispatch, radixScatter,
                countRadix, scanTiles, scanTileOffsets, scanAddTileOffsets);
            ++checks;
            executeHierarchicalRadixCase(device, {1366u}, emit, radixHistogram,
                radixPrefix, radixPrefixPrepare, radixPrefixBlocks,
                radixPrefixChunkOffsets, prepareRadixDispatch, radixScatter,
                countRadix, scanTiles, scanTileOffsets, scanAddTileOffsets);
            ++checks;
            executeHierarchicalRadixCase(device, {1365u, 1366u}, emit,
                radixHistogram, radixPrefix, radixPrefixPrepare,
                radixPrefixBlocks, radixPrefixChunkOffsets,
                prepareRadixDispatch, radixScatter, countRadix, scanTiles,
                scanTileOffsets, scanAddTileOffsets);
            ++checks;
            executeHierarchicalRadixCase(device, {0u, 1366u}, emit,
                radixHistogram, radixPrefix, radixPrefixPrepare,
                radixPrefixBlocks, radixPrefixChunkOffsets,
                prepareRadixDispatch, radixScatter, countRadix, scanTiles,
                scanTileOffsets, scanAddTileOffsets);
            ++checks;
            std::cout << "PASS native deterministic radix contact incidence and surface Morton sort; checks="
                      << checks << " ; incidence_runs=8; surface_runs=2; hierarchical_runs=4; environments=2; physical_steps=0\n";
            return 0;
        } catch (const std::exception& exception) {
            std::cerr << "FAIL native deterministic sorted contact incidence: "
                      << exception.what() << '\n';
            return 1;
        }
    }
}
