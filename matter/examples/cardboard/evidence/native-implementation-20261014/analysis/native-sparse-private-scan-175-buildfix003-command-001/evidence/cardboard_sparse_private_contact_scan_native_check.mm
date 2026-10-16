#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/shared.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NUMI_MATTER_METALLIB
#error "CMake must define NUMI_MATTER_METALLIB"
#endif

namespace {
using U = std::uint32_t;
constexpr U kPairsPerBlock = 256u;
constexpr U kInvalid = NM_INVALID_INDEX;
unsigned checks = 0u;

void require(const bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

id<MTLBuffer> makeBuffer(id<MTLDevice> device, const void* bytes,
                         const std::size_t size, const char* name) {
    require(bytes != nullptr && size > 0u,
            std::string("invalid input buffer: ") + name);
    id<MTLBuffer> result = [device newBufferWithBytes:bytes length:size
        options:MTLResourceStorageModeShared];
    require(result != nil, std::string("allocation failed: ") + name);
    return result;
}

template <typename T>
id<MTLBuffer> makeBuffer(id<MTLDevice> device,
                         const std::vector<T>& values, const char* name) {
    return makeBuffer(device, values.data(), values.size() * sizeof(T), name);
}

id<MTLComputePipelineState> makePipeline(id<MTLDevice> device,
                                          id<MTLLibrary> library,
                                          const char* name) {
    const std::string qualified = std::string("numi_matter_metal::") + name;
    id<MTLFunction> function = [library
        newFunctionWithName:[NSString stringWithUTF8String:qualified.c_str()]];
    require(function != nil, std::string("missing Metal function: ") + name);
    NSError* error = nil;
    id<MTLComputePipelineState> result = [device
        newComputePipelineStateWithFunction:function error:&error];
    require(result != nil, std::string("pipeline creation failed: ") + name +
        ": " + (error.localizedDescription.UTF8String != nullptr
            ? error.localizedDescription.UTF8String : "unknown"));
    return result;
}

void bind(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> buffer,
          const NSUInteger index) {
    [encoder setBuffer:buffer offset:0u atIndex:index];
}

template <typename T>
void bindBytes(id<MTLComputeCommandEncoder> encoder, const T& value,
               const NSUInteger index) {
    [encoder setBytes:&value length:sizeof(value) atIndex:index];
}

void finishCommand(id<MTLCommandBuffer> command) {
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
        std::string("Metal command failed: ") +
        (command.error.localizedDescription.UTF8String != nullptr
            ? command.error.localizedDescription.UTF8String : "unknown"));
}

U blocksFor(const U pairCount) {
    return pairCount / kPairsPerBlock +
        (pairCount % kPairsPerBlock != 0u ? 1u : 0u);
}

struct Pattern {
    std::vector<U> selected;
};

Pattern makePattern(const std::string& kind, const U pairCount,
                    const U seed) {
    Pattern result;
    result.selected.reserve(pairCount);
    std::uint32_t state = seed == 0u ? 1u : seed;
    for (U pair = 0u; pair < pairCount; ++pair) {
        bool selected = false;
        if (kind == "all") selected = true;
        else if (kind == "none") selected = false;
        else if (kind == "alternating") selected = (pair & 1u) == 0u;
        else if (kind == "block-tail")
            selected = pair % kPairsPerBlock >= 248u;
        else if (kind == "random") {
            state = state * 1664525u + 1013904223u;
            selected = (state >> 27u) < 3u;
        } else {
            throw std::runtime_error("unknown test pattern");
        }
        if (selected) result.selected.push_back(pair);
    }
    return result;
}

struct Case {
    std::string name;
    U environments = 1u;
    U pairCount = 0u;
    U capacity = 0u;
    std::vector<std::vector<U>> selectedByEnvironment;
    std::vector<bool> unhealthy;
    bool partialFlagsForUnhealthy = false;
};

struct Outputs {
    std::vector<U> slots;
    std::vector<U> ids;
    std::vector<U> counts;
    std::vector<NMMatterStatusGPU> statuses;
};

Outputs initialOutputs(const Case& test) {
    Outputs output;
    const std::size_t pairTotal =
        std::size_t(test.environments) * test.pairCount;
    const std::size_t candidateTotal =
        std::size_t(test.environments) * test.capacity;
    output.slots.resize(pairTotal, kInvalid);
    output.ids.resize(candidateTotal);
    output.counts.resize(test.environments);
    output.statuses.resize(test.environments);
    for (std::size_t index = 0u; index < candidateTotal; ++index)
        output.ids[index] = 0xA5000000u + static_cast<U>(index);
    for (U environment = 0u; environment < test.environments; ++environment) {
        output.counts[environment] = 0x12340000u + environment;
        auto& status = output.statuses[environment];
        std::memset(&status, 0, sizeof(status));
        status.code = test.unhealthy[environment]
            ? NM_STATUS_CONTACT_FAILURE : NM_STATUS_SUCCESS;
        status.environment = environment;
        status.objectIndex = test.unhealthy[environment] ? 17u : NM_INVALID_INDEX;
        status.failingIndex = test.unhealthy[environment] ? 23u : NM_INVALID_INDEX;
        status.completedMicrosteps = 41u + environment;
        status.fgmresIterations = 5u + environment;
        status.contactCount = 7u + environment;
        status.eventCount = 9u + environment;
        status.diagnostics = {1.25f, 2.5f, 3.75f, 4.5f};
        const std::size_t pairBase = std::size_t(environment) * test.pairCount;
        if (test.partialFlagsForUnhealthy && test.unhealthy[environment]) {
            // An earlier producer can fail after writing only part of its
            // flags. The legacy scan preserves these words and the old IDs.
            for (U pair = 0u; pair < test.pairCount; ++pair) {
                output.slots[pairBase + pair] = pair % 5u == 0u ? 1u :
                    (pair % 5u == 1u ? 7u : kInvalid);
            }
        } else {
            for (const U pair : test.selectedByEnvironment[environment])
                output.slots[pairBase + pair] = 1u;
        }
    }
    return output;
}

void validateOracle(const Case& test, const Outputs& output,
                    const Outputs& initial) {
    for (U environment = 0u; environment < test.environments; ++environment) {
        const std::size_t pairBase = std::size_t(environment) * test.pairCount;
        const std::size_t candidateBase =
            std::size_t(environment) * test.capacity;
        if (test.unhealthy[environment]) {
            require(output.counts[environment] == 0u,
                test.name + ": unhealthy count is not zero");
            require(std::equal(output.slots.begin() + pairBase,
                    output.slots.begin() + pairBase + test.pairCount,
                    initial.slots.begin() + pairBase),
                test.name + ": unhealthy pair flags were rewritten");
            require(std::equal(output.ids.begin() + candidateBase,
                    output.ids.begin() + candidateBase + test.capacity,
                    initial.ids.begin() + candidateBase),
                test.name + ": unhealthy candidate IDs were rewritten");
            require(std::memcmp(&output.statuses[environment],
                    &initial.statuses[environment], sizeof(NMMatterStatusGPU)) == 0,
                test.name + ": unhealthy status was changed");
            continue;
        }

        const auto& selected = test.selectedByEnvironment[environment];
        const U expectedCount = std::min<U>(
            static_cast<U>(selected.size()), test.capacity);
        require(output.counts[environment] == expectedCount,
            test.name + ": candidate count differs from independent oracle");
        const bool overflow = selected.size() > test.capacity;
        require(output.statuses[environment].code ==
                (overflow ? NM_STATUS_CAPACITY_OVERFLOW : NM_STATUS_SUCCESS),
            test.name + ": overflow status differs from independent oracle");
        for (U rank = 0u; rank < test.capacity; ++rank) {
            const U expectedID = rank < expectedCount
                ? selected[rank] : initial.ids[candidateBase + rank];
            require(output.ids[candidateBase + rank] == expectedID,
                test.name + ": candidate ID order/tail differs from oracle");
        }
        std::size_t selectedCursor = 0u;
        for (U pair = 0u; pair < test.pairCount; ++pair) {
            U expectedSlot = kInvalid;
            if (selectedCursor < selected.size() && selected[selectedCursor] == pair) {
                if (selectedCursor < test.capacity)
                    expectedSlot = static_cast<U>(selectedCursor);
                ++selectedCursor;
            }
            require(output.slots[pairBase + pair] == expectedSlot,
                test.name + ": canonical pair-to-slot map differs from oracle");
        }
    }
}

Outputs runLegacy(id<MTLDevice> device, id<MTLCommandQueue> queue,
                  id<MTLComputePipelineState> pipeline,
                  const NMMatterDispatchGPU& dispatch, const Case& test,
                  const Outputs& initial) {
    Outputs output = initial;
    auto dispatchBuffer = makeBuffer(device, &dispatch, sizeof(dispatch), "dispatch");
    auto capacityBuffer = makeBuffer(device, &test.capacity,
        sizeof(test.capacity), "capacity");
    auto slotsBuffer = makeBuffer(device, output.slots, "legacy slots");
    auto idsBuffer = makeBuffer(device, output.ids, "legacy IDs");
    auto countsBuffer = makeBuffer(device, output.counts, "legacy counts");
    auto statusBuffer = makeBuffer(device, output.statuses, "legacy statuses");
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    require(command != nil && encoder != nil, "legacy encoder allocation failed");
    [encoder setComputePipelineState:pipeline];
    bind(encoder, dispatchBuffer, 0u);
    bind(encoder, capacityBuffer, 1u);
    bind(encoder, slotsBuffer, 2u);
    bind(encoder, idsBuffer, 3u);
    bind(encoder, countsBuffer, 4u);
    bind(encoder, statusBuffer, 5u);
    [encoder dispatchThreadgroups:MTLSizeMake(test.environments, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder endEncoding];
    finishCommand(command);
    std::memcpy(output.slots.data(), slotsBuffer.contents,
        output.slots.size() * sizeof(U));
    std::memcpy(output.ids.data(), idsBuffer.contents,
        output.ids.size() * sizeof(U));
    std::memcpy(output.counts.data(), countsBuffer.contents,
        output.counts.size() * sizeof(U));
    std::memcpy(output.statuses.data(), statusBuffer.contents,
        output.statuses.size() * sizeof(NMMatterStatusGPU));
    return output;
}

Outputs runHierarchical(id<MTLDevice> device, id<MTLCommandQueue> queue,
                        id<MTLComputePipelineState> blocksPipeline,
                        id<MTLComputePipelineState> offsetsPipeline,
                        id<MTLComputePipelineState> scatterPipeline,
                        const NMMatterDispatchGPU& dispatch, const Case& test,
                        const Outputs& initial) {
    Outputs output = initial;
    const U blocks = blocksFor(test.pairCount);
    std::vector<U> offsets(std::size_t(test.environments) * blocks,
        0xCCCCCCCCu);
    auto dispatchBuffer = makeBuffer(device, &dispatch, sizeof(dispatch), "dispatch");
    auto capacityBuffer = makeBuffer(device, &test.capacity,
        sizeof(test.capacity), "capacity");
    auto blocksBuffer = makeBuffer(device, &blocks, sizeof(blocks), "blocks");
    auto slotsBuffer = makeBuffer(device, output.slots, "hierarchical slots");
    auto offsetsBuffer = makeBuffer(device, offsets, "block offsets");
    auto idsBuffer = makeBuffer(device, output.ids, "hierarchical IDs");
    auto countsBuffer = makeBuffer(device, output.counts, "hierarchical counts");
    auto statusBuffer = makeBuffer(device, output.statuses, "hierarchical statuses");

    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    require(command != nil && encoder != nil, "hierarchical encoder allocation failed");

    [encoder setComputePipelineState:blocksPipeline];
    bind(encoder, dispatchBuffer, 0u);
    bind(encoder, slotsBuffer, 1u);
    bind(encoder, offsetsBuffer, 2u);
    bind(encoder, statusBuffer, 3u);
    [encoder dispatchThreadgroups:MTLSizeMake(
        std::size_t(test.environments) * blocks, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:offsetsPipeline];
    bind(encoder, dispatchBuffer, 0u);
    bind(encoder, capacityBuffer, 1u);
    bind(encoder, blocksBuffer, 2u);
    bind(encoder, offsetsBuffer, 3u);
    bind(encoder, countsBuffer, 4u);
    bind(encoder, statusBuffer, 5u);
    [encoder dispatchThreadgroups:MTLSizeMake(test.environments, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:scatterPipeline];
    bind(encoder, dispatchBuffer, 0u);
    bind(encoder, capacityBuffer, 1u);
    bind(encoder, blocksBuffer, 2u);
    bind(encoder, slotsBuffer, 3u);
    bind(encoder, offsetsBuffer, 4u);
    bind(encoder, idsBuffer, 5u);
    const NSUInteger width = std::max<NSUInteger>(
        1u, std::min<NSUInteger>(256u, scatterPipeline.maxTotalThreadsPerThreadgroup));
    [encoder dispatchThreads:MTLSizeMake(
        std::size_t(test.environments) * test.pairCount, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
    [encoder endEncoding];
    finishCommand(command);

    std::memcpy(output.slots.data(), slotsBuffer.contents,
        output.slots.size() * sizeof(U));
    std::memcpy(output.ids.data(), idsBuffer.contents,
        output.ids.size() * sizeof(U));
    std::memcpy(output.counts.data(), countsBuffer.contents,
        output.counts.size() * sizeof(U));
    std::memcpy(output.statuses.data(), statusBuffer.contents,
        output.statuses.size() * sizeof(NMMatterStatusGPU));
    return output;
}

void runCase(id<MTLDevice> device, id<MTLCommandQueue> queue,
             id<MTLComputePipelineState> legacy,
             id<MTLComputePipelineState> blockScan,
             id<MTLComputePipelineState> offsetScan,
             id<MTLComputePipelineState> scatter,
             const Case& test) {
    require(test.environments > 0u && test.environments <= 4u &&
            test.pairCount > 0u && test.capacity > 0u &&
            test.capacity <= test.pairCount &&
            test.selectedByEnvironment.size() == test.environments &&
            test.unhealthy.size() == test.environments,
        test.name + ": malformed test dimensions");
    require(std::uint64_t(test.environments) * test.pairCount <=
            std::numeric_limits<U>::max(),
        test.name + ": flattened pair domain exceeds uint32");
    NMMatterDispatchGPU dispatch{};
    dispatch.environmentCount = test.environments;
    dispatch.contactPairCount = test.pairCount;
    const Outputs initial = initialOutputs(test);
    const Outputs oldResult = runLegacy(
        device, queue, legacy, dispatch, test, initial);
    const Outputs newResult = runHierarchical(device, queue,
        blockScan, offsetScan, scatter, dispatch, test, initial);
    validateOracle(test, oldResult, initial);
    validateOracle(test, newResult, initial);
    require(oldResult.slots == newResult.slots,
        test.name + ": pair slot map differs from legacy scan");
    require(oldResult.ids == newResult.ids,
        test.name + ": compact IDs/tail differ from legacy scan");
    require(oldResult.counts == newResult.counts,
        test.name + ": candidate counts differ from legacy scan");
    require(std::memcmp(oldResult.statuses.data(), newResult.statuses.data(),
                oldResult.statuses.size() * sizeof(NMMatterStatusGPU)) == 0,
        test.name + ": status records differ from legacy scan");
    std::cout << "case=" << test.name << " env=" << test.environments
              << " pair_count=" << test.pairCount << " capacity="
              << test.capacity << " result=exact_match\n";
}

void runCpuContracts() {
    constexpr U pairCount = 257u;
    constexpr U capacity = 128u;
    for (const U selectedCount : {capacity - 1u, capacity + 1u}) {
        std::vector<U> inputSlots(pairCount, kInvalid);
        std::vector<U> actualSlots(pairCount, kInvalid);
        std::vector<U> inputIDs(capacity);
        std::vector<U> actualIDs(capacity);
        std::vector<U> expectedIDs(capacity);
        for (U slot = 0u; slot < capacity; ++slot)
            inputIDs[slot] = actualIDs[slot] =
                expectedIDs[slot] = 0xB1000000u + slot;
        for (U rank = 0u; rank < selectedCount; ++rank)
            inputSlots[rank * 2u] = actualSlots[rank * 2u] = 1u;
        std::vector<U> expectedSlots(pairCount, kInvalid);
        const U expectedCount = std::min(selectedCount, capacity);
        for (U rank = 0u; rank < expectedCount; ++rank) {
            expectedIDs[rank] = rank * 2u;
            expectedSlots[rank * 2u] = rank;
        }
        U actualCount = 0u;
        for (U pair = 0u; pair < pairCount; ++pair) {
            if (inputSlots[pair] == 1u) {
                if (actualCount < capacity) {
                    actualIDs[actualCount] = pair;
                    actualSlots[pair] = actualCount;
                } else {
                    actualSlots[pair] = kInvalid;
                }
                ++actualCount;
            } else {
                actualSlots[pair] = kInvalid;
            }
        }
        const U actualStatus = actualCount > capacity
            ? NM_STATUS_CAPACITY_OVERFLOW : NM_STATUS_SUCCESS;
        actualCount = std::min(actualCount, capacity);
        require(actualCount == expectedCount && actualStatus ==
                    (selectedCount == capacity + 1u
                        ? NM_STATUS_CAPACITY_OVERFLOW : NM_STATUS_SUCCESS) &&
                actualIDs == expectedIDs && actualSlots == expectedSlots,
            "CPU scan contract differs from exact capacity boundary oracle");
    }

    std::vector<U> partialSlots{1u, 7u, kInvalid, 1u, 0u};
    const std::vector<U> originalSlots = partialSlots;
    std::vector<U> partialIDs{91u, 92u, 93u};
    const std::vector<U> originalIDs = partialIDs;
    U count = 77u;
    NMMatterStatusGPU status{};
    status.code = NM_STATUS_CONTACT_FAILURE;
    status.environment = 0u;
    status.objectIndex = 3u;
    status.failingIndex = 4u;
    status.completedMicrosteps = 9u;
    const NMMatterStatusGPU originalStatus = status;
    // The legacy unhealthy path resets only the count; flags, IDs, and status
    // are preserved, even when a prior producer left mixed partial flags.
    if (status.code != NM_STATUS_SUCCESS) count = 0u;
    require(count == 0u && partialSlots == originalSlots &&
            partialIDs == originalIDs &&
            std::memcmp(&status, &originalStatus, sizeof(status)) == 0,
        "prefailed partial-map contract must preserve flags, IDs, and failure status");
    std::cout << "cpu_contracts=pass selected_count=127/128/129 prefailed_partial_preserved=true\n";
}

Case singleCase(const std::string& name, const U pairCount, const U capacity,
                const std::string& pattern, const bool overflow = false) {
    Case result;
    result.name = name;
    result.pairCount = pairCount;
    result.capacity = capacity;
    result.selectedByEnvironment = {
        makePattern(pattern, pairCount, 0x12345678u).selected};
    result.unhealthy = {false};
    if (overflow && result.selectedByEnvironment[0].size() <= capacity) {
        result.selectedByEnvironment[0].clear();
        for (U pair = 0u; pair < pairCount; ++pair)
            if ((pair % 7u) == 0u) result.selectedByEnvironment[0].push_back(pair);
        require(result.selectedByEnvironment[0].size() > capacity,
            name + ": requested overflow pattern does not overflow");
    }
    return result;
}

Case exactBoundaryCase(const std::string& name, const U selectedCount) {
    constexpr U pairCount = 257u;
    constexpr U capacity = 128u;
    require(selectedCount <= pairCount,
        name + ": selected count exceeds pair domain");
    Case result;
    result.name = name;
    result.pairCount = pairCount;
    result.capacity = capacity;
    std::vector<U> selected;
    selected.reserve(selectedCount);
    for (U pair = 0u; pair < selectedCount; ++pair)
        selected.push_back(pair);
    result.selectedByEnvironment = {std::move(selected)};
    result.unhealthy = {false};
    return result;
}

} // namespace

int main(const int argc, const char* const* argv) {
    @autoreleasepool {
        try {
            runCpuContracts();
            if (argc == 2 && std::strcmp(argv[1], "--cpu-only") == 0) {
                std::cout << "sparse_scan_cpu_checks=" << checks
                          << " metal_kernels_executed=0\n";
                return 0;
            }
            require(argc == 1, "usage: checker [--cpu-only]");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "no default Metal device");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(queue != nil, "Metal command queue allocation failed");
            NSError* error = nil;
            NSString* libraryPath = [NSString stringWithUTF8String:NUMI_MATTER_METALLIB];
            id<MTLLibrary> library = [device
                newLibraryWithURL:[NSURL fileURLWithPath:libraryPath] error:&error];
            require(library != nil, std::string("cannot load Matter metallib: ") +
                (error.localizedDescription.UTF8String != nullptr
                    ? error.localizedDescription.UTF8String : "unknown"));
            auto legacy = makePipeline(device, library,
                "nm_contact_scan_armijo_sparse_pair_flags");
            auto blockScan = makePipeline(device, library,
                "nm_contact_scan_armijo_sparse_pair_blocks");
            auto offsetScan = makePipeline(device, library,
                "nm_contact_scan_armijo_sparse_pair_block_offsets");
            auto scatter = makePipeline(device, library,
                "nm_contact_scatter_armijo_sparse_pair_blocks");

            for (const U size : {1u, 31u, 32u, 33u, 255u, 256u, 257u,
                                 511u, 512u, 513u, 767u, 768u, 769u, 1025u}) {
                const U cap = std::max(1u, size / 2u);
                runCase(device, queue, legacy, blockScan, offsetScan, scatter,
                    singleCase(std::string(size == 1u ? "all_fit_" :
                        "all_overflow_") + std::to_string(size),
                        size, cap, "all", size > 1u));
                runCase(device, queue, legacy, blockScan, offsetScan, scatter,
                    singleCase("none_" + std::to_string(size),
                        size, cap, "none"));
                runCase(device, queue, legacy, blockScan, offsetScan, scatter,
                    singleCase("block_tail_" + std::to_string(size),
                        size, size, "block-tail"));
            }
            runCase(device, queue, legacy, blockScan, offsetScan, scatter,
                exactBoundaryCase("capacity_minus_one", 127u));
            runCase(device, queue, legacy, blockScan, offsetScan, scatter,
                exactBoundaryCase("capacity_exact", 128u));
            runCase(device, queue, legacy, blockScan, offsetScan, scatter,
                exactBoundaryCase("capacity_plus_one", 129u));
            runCase(device, queue, legacy, blockScan, offsetScan, scatter,
                singleCase("alternating_multi_block", 4097u, 2048u,
                    "alternating", true));
            runCase(device, queue, legacy, blockScan, offsetScan, scatter,
                singleCase("deterministic_random", 8193u, 8193u, "random"));

            Case mixed;
            mixed.name = "two_environment_healthy_overflow_and_prefailed_partial";
            mixed.environments = 2u;
            mixed.pairCount = 513u;
            mixed.capacity = 64u;
            mixed.selectedByEnvironment = {
                makePattern("all", mixed.pairCount, 1u).selected,
                {1u, 257u, 512u}};
            mixed.unhealthy = {false, true};
            mixed.partialFlagsForUnhealthy = true;
            runCase(device, queue, legacy, blockScan, offsetScan, scatter, mixed);

            constexpr U shippingPairs = 6335568u;
            Case shipping = singleCase("shipping_pair_domain_no_overflow",
                shippingPairs, 65536u, "random");
            // Deterministic sparse occupancy below the fixed candidate cap,
            // spread across all block boundaries of the full shipping domain.
            shipping.selectedByEnvironment[0].clear();
            for (U pair = 0u; pair < shippingPairs; pair += 97u)
                shipping.selectedByEnvironment[0].push_back(pair);
            require(shipping.selectedByEnvironment[0].size() <= shipping.capacity,
                "shipping sparse fixture unexpectedly overflows capacity");
            runCase(device, queue, legacy, blockScan, offsetScan, scatter, shipping);

            std::cout << "hierarchical_sparse_scan_native_checks=" << checks
                      << " kernels_executed=1 physical_transactions=0\n";
            return 0;
        } catch (const std::exception& exception) {
            std::cerr << "sparse scan checker error: " << exception.what() << '\n';
            return 1;
        }
    }
}
