#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/detail/shipping_contact_page_plan.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace numi::matter::detail;

void require(bool okay, const char* message) {
    if (!okay) throw std::runtime_error(message);
}

constexpr const char* kSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void shipping_contact_page_mark(
    device atomic_uint* visits [[buffer(0)]],
    constant uint4& page [[buffer(1)]],
    device atomic_uint* errors [[buffer(2)]],
    uint local [[thread_position_in_grid]]) {
    if (local >= page.y) return;
    const uint id = page.x + local;
    if (id < page.x || id >= page.z) {
        atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
        return;
    }
    atomic_fetch_add_explicit(&visits[id], 1u, memory_order_relaxed);
}
)MSL";

struct DomainResult {
    std::uint64_t totalRows = 0u;
    std::uint32_t pageRows = 0u;
    std::uint32_t pageCount = 0u;
    std::uint32_t lastRows = 0u;
    std::uint64_t missing = 0u;
    std::uint64_t duplicate = 0u;
    std::uint64_t visitsBytes = 0u;
};

DomainResult checkDomain(id<MTLDevice> device,
                         id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> pipeline,
                         std::uint64_t totalRows,
                         std::uint32_t pageRows,
                         std::uint32_t pageCount,
                         std::uint32_t lastRows) {
    DomainResult result{totalRows, pageRows, pageCount, lastRows};
    require(totalRows != 0u &&
            totalRows <= std::numeric_limits<std::uint32_t>::max(),
            "native page-domain is outside the uint32 canonical-ID range");
    result.visitsBytes = totalRows * sizeof(std::uint32_t);
    require(result.visitsBytes <= device.maxBufferLength,
            "native coverage counter buffer exceeds Metal maxBufferLength");
    id<MTLBuffer> visits = [device newBufferWithLength:
        static_cast<NSUInteger>(result.visitsBytes)
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> errors = [device newBufferWithLength:16u
        options:MTLResourceStorageModeShared];
    require(visits != nil && errors != nil,
            "native coverage buffers failed allocation");
    std::memset(visits.contents, 0, visits.length);
    std::memset(errors.contents, 0, errors.length);

    id<MTLCommandBuffer> command = [queue commandBuffer];
    require(command != nil, "cannot create native page command buffer");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    require(encoder != nil, "cannot create native page encoder");
    [encoder setComputePipelineState:pipeline];

    std::uint64_t enumerated = 0u;
    for (std::uint32_t pageIndex = 0u; pageIndex < pageCount; ++pageIndex) {
        ShippingContactPageRange range{};
        require(shippingContactPageRange(totalRows, pageRows, pageIndex, range),
                "CPU range generator rejected an in-range page");
        const std::uint32_t end = range.firstRow + range.rowCount;
        require(end >= range.firstRow && end <= totalRows,
                "page end overflow or out-of-domain range");
        const nm_uint4 page{
            range.firstRow, range.rowCount,
            static_cast<std::uint32_t>(totalRows), pageIndex};
        [encoder setBuffer:visits offset:0u atIndex:0u];
        [encoder setBytes:&page length:sizeof(page) atIndex:1u];
        [encoder setBuffer:errors offset:0u atIndex:2u];
        [encoder dispatchThreads:MTLSizeMake(range.rowCount, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(64u, 1u, 1u)];
        enumerated += range.rowCount;
        if (pageIndex + 1u < pageCount) {
            ShippingContactPageRange next{};
            require(shippingContactPageRange(totalRows, pageRows,
                                              pageIndex + 1u, next) &&
                    next.firstRow == end,
                    "canonical page intervals have a gap or overlap");
        } else {
            require(range.rowCount == lastRows && end == totalRows,
                    "final page is truncated or extends beyond the immutable ID domain");
        }
    }
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
            "native page coverage dispatch failed");
    require(enumerated == totalRows,
            "native dispatch plan did not enumerate the complete domain");
    require(static_cast<const std::uint32_t*>(errors.contents)[0] == 0u,
            "Metal page kernel observed an out-of-domain index");

    const auto* counts = static_cast<const std::uint32_t*>(visits.contents);
    for (std::uint64_t row = 0u; row < totalRows; ++row) {
        if (counts[row] == 0u) ++result.missing;
        else if (counts[row] != 1u) ++result.duplicate;
    }
    require(result.missing == 0u && result.duplicate == 0u,
            "canonical immutable row IDs were omitted or visited more than once");
    return result;
}
}

int main() {
    @autoreleasepool {
        try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "no Metal device");
            NSError* error = nil;
            NSString* source = [NSString stringWithUTF8String:kSource];
            id<MTLLibrary> library =
                [device newLibraryWithSource:source options:nil error:&error];
            require(library != nil, "cannot compile the bounded page coverage kernel");
            id<MTLFunction> function = [library newFunctionWithName:
                @"shipping_contact_page_mark"];
            require(function != nil, "page coverage function is missing");
            id<MTLComputePipelineState> pipeline =
                [device newComputePipelineStateWithFunction:function error:&error];
            require(pipeline != nil, "cannot create page coverage pipeline");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(queue != nil, "cannot create page command queue");

            ShippingContactPagePlanInput input;
            input.environmentCount = 1u;
            input.contactPairCount = 6335568u;
            input.cohesiveFaceCount = 260835u;
            input.femNodeCount = 528444u;
            input.retainedGlobalUnknownBufferCount = 2u;
            input.retainedGlobalUnknownBytesPerBuffer = 287473536u;
            input.maxBufferLength = device.maxBufferLength;
            const auto plan = makeShippingContactPagePlan(input);
            require(plan.admitted(), shippingContactPagePlanErrorText(plan.error));

            id<MTLBuffer> samplePage = [device newBufferWithLength:
                static_cast<NSUInteger>(plan.contactSamplePageBytes)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> candidateIds = [device newBufferWithLength:
                static_cast<NSUInteger>(plan.contactCandidateIdPageBytes)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> slots = [device newBufferWithLength:
                static_cast<NSUInteger>(plan.contactSlotPageBytes)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> pageCounts = [device newBufferWithLength:
                static_cast<NSUInteger>(plan.contactPageCountsBytes)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> cohesiveForcePage = [device newBufferWithLength:
                static_cast<NSUInteger>(plan.cohesiveForcePageBytes)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> cohesiveOperatorPage = [device newBufferWithLength:
                static_cast<NSUInteger>(plan.cohesiveOperatorPageBytes)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> cohesiveAccumulator = [device newBufferWithLength:
                static_cast<NSUInteger>(plan.cohesiveAccumulatorBytes)
                options:MTLResourceStorageModeShared];
            require(samplePage != nil && candidateIds != nil && slots != nil &&
                    pageCounts != nil && cohesiveForcePage != nil &&
                    cohesiveOperatorPage != nil && cohesiveAccumulator != nil,
                    "bounded page workspaces failed native allocation");
            require(samplePage.length == plan.contactSamplePageBytes &&
                    candidateIds.length == plan.contactCandidateIdPageBytes &&
                    slots.length == plan.contactSlotPageBytes &&
                    pageCounts.length == plan.contactPageCountsBytes &&
                    cohesiveForcePage.length == plan.cohesiveForcePageBytes &&
                    cohesiveOperatorPage.length == plan.cohesiveOperatorPageBytes &&
                    cohesiveAccumulator.length == plan.cohesiveAccumulatorBytes,
                    "native page allocations disagree with the checked plan");

            const DomainResult pairs = checkDomain(device, queue, pipeline,
                input.contactPairCount, plan.contactPageRows,
                plan.contactPageCount, plan.contactLastPageRows);
            const DomainResult faces = checkDomain(device, queue, pipeline,
                input.cohesiveFaceCount, plan.cohesivePageFaces,
                plan.cohesivePageCount, plan.cohesiveLastPageFaces);

            std::cout << "shipping_contact_page_native_checks=PASS"
                      << " pair_rows=" << pairs.totalRows
                      << " pair_pages=" << pairs.pageCount
                      << " pair_last=" << pairs.lastRows
                      << " pair_visits_bytes=" << pairs.visitsBytes
                      << " cohesive_faces=" << faces.totalRows
                      << " cohesive_pages=" << faces.pageCount
                      << " cohesive_last=" << faces.lastRows
                      << " cohesive_visits_bytes=" << faces.visitsBytes
                      << " device_max_buffer_bytes="
                      << static_cast<unsigned long long>(device.maxBufferLength)
                      << " contact_page_bytes=" << plan.contactPageScratchBytes
                      << " cohesive_page_bytes=" << plan.cohesivePageScratchBytes
                      << " id_basis=canonical_compiled_row_ordinal"
                      << " production_paging_enabled=false"
                      << " whole_process_admission=false"
                      << " physical_transactions=0\n";
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "shipping contact page native check failed: "
                      << e.what() << '\n';
            return 1;
        }
    }
}
