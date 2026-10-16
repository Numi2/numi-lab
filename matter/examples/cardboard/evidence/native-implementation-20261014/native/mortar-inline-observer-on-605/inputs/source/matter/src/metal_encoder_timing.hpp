#pragma once
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace numi::matter::detail {
// Opt-in, read-only stage timestamps on the existing borrowed command buffer.
// Sampling one accepted root bounds counter allocation and does not read any
// physical buffer. Unavailable timestamps are never reported as zero time.
inline id<MTLComputeCommandEncoder> timedEncoder(
    id<MTLCommandBuffer> command, id<MTLDevice> device,
    const char* stage, unsigned root
) {
    const char* requested = std::getenv("NUMI_MATTER_GPU_TIMING");
    const char* selected = std::getenv("NUMI_MATTER_GPU_TIMING_STAGE");
    if (requested == nullptr || std::strcmp(requested, "1") != 0 ||
        root != 100u ||
        (selected != nullptr && std::strcmp(selected, stage) != 0) ||
        ![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])
        return [command computeCommandEncoder];
    id<MTLCounterSampleBuffer> timing = nil;
    for (id<MTLCounterSet> set in device.counterSets) {
        if (![set.name isEqualToString:MTLCommonCounterSetTimestamp]) continue;
        auto* descriptor = [MTLCounterSampleBufferDescriptor new];
        descriptor.counterSet = set;
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.sampleCount = 2u;
        timing = [device newCounterSampleBufferWithDescriptor:descriptor error:nil];
        break;
    }
    if (timing == nil) return [command computeCommandEncoder];
    auto* pass = [MTLComputePassDescriptor computePassDescriptor];
    pass.sampleBufferAttachments[0].sampleBuffer = timing;
    pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = 0u;
    pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = 1u;
    [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        NSData* data = [timing resolveCounterRange:NSMakeRange(0u, 2u)];
        if (completed.status != MTLCommandBufferStatusCompleted ||
            data.length != 2u * sizeof(MTLCounterResultTimestamp)) return;
        const auto* sample = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
        if (sample[0].timestamp == 0u || sample[1].timestamp < sample[0].timestamp ||
            sample[0].timestamp == MTLCounterErrorValue ||
            sample[1].timestamp == MTLCounterErrorValue) return;
        std::fprintf(stderr, "matter_gpu_stage=%s step=%u elapsed_ns=%llu\n",
            stage, root, static_cast<unsigned long long>(sample[1].timestamp - sample[0].timestamp));
    }];
    return [command computeCommandEncoderWithDescriptor:pass];
}
} // namespace numi::matter::detail
