#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/shared.h"
#include "cardboard_contact_node_incidence_sort_reference.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
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
                 id<MTLComputePipelineState> scatterLegacy) {
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

    const Result expected = buildLegacyReference(kNodes, kCapacity, environments);
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
    Result legacyGPU;
    legacyGPU.ranges.resize(kEnvironments * kNodes);
    legacyGPU.incidence.resize(kEnvironments * kCapacity * 6u);
    for (std::size_t i = 0u; i < actual.ranges.size(); ++i) {
        actual.ranges[i] = {gpuRanges[i].first, gpuRanges[i].count};
        legacyGPU.ranges[i] = {gpuLegacyRanges[i].first,
                               gpuLegacyRanges[i].count};
        require(gpuRanges[i].objectIndex == NM_INVALID_INDEX &&
                    gpuRanges[i].reserved == 0u &&
                    gpuLegacyRanges[i].objectIndex == NM_INVALID_INDEX &&
                    gpuLegacyRanges[i].reserved == 0u,
                "native incidence range metadata differs from the legacy initializer");
    }
    std::copy(gpuIncidence, gpuIncidence + actual.incidence.size(),
              actual.incidence.begin());
    std::copy(gpuLegacyIncidence,
              gpuLegacyIncidence + legacyGPU.incidence.size(),
              legacyGPU.incidence.begin());
    require(actual.ranges == expected.ranges &&
                actual.incidence == expected.incidence,
            "native sorted CSR differs from the complete CPU legacy output");
    require(legacyGPU.ranges == expected.ranges &&
                legacyGPU.incidence == expected.incidence,
            "native legacy CSR differs from the complete CPU reference output");
    require(actual.ranges == legacyGPU.ranges &&
                actual.incidence == legacyGPU.incidence,
            "native sorted CSR differs from native legacy CSR");
    for (const NMMatterStatusGPU& status :
         std::span<const NMMatterStatusGPU>(gpuStatuses, kEnvironments))
        require(status.code == 0u,
                "native sorted incidence reported a capacity/status failure");
    const auto* gpuLegacyStatuses = static_cast<const NMMatterStatusGPU*>(
        legacyStatuses.contents);
    for (const NMMatterStatusGPU& status :
         std::span<const NMMatterStatusGPU>(gpuLegacyStatuses, kEnvironments))
        require(status.code == 0u,
                "native legacy incidence reported a capacity/status failure");
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

            executeCase(device, explicitCase(), emit, bitonic, countSorted,
                        countLegacy, scan, scatter, scatterLegacy);
            ++checks;
            executeCase(device, zeroActiveCase(), emit, bitonic, countSorted,
                        countLegacy, scan, scatter, scatterLegacy);
            ++checks;
            executeCase(device, invalidSlotCase(), emit, bitonic, countSorted,
                        countLegacy, scan, scatter, scatterLegacy);
            ++checks;
            std::cout << "PASS native deterministic sorted contact incidence; checks="
                      << checks << "; cases=3; environments=2; physical_steps=0\n";
            return 0;
        } catch (const std::exception& exception) {
            std::cerr << "FAIL native deterministic sorted contact incidence: "
                      << exception.what() << '\n';
            return 1;
        }
    }
}
