#pragma once
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace numi::matter::detail {
struct CounterTimingRecord {
    std::string stage;
    unsigned root = 0u;
    NSUInteger begin = 0u;
    NSUInteger end = 0u;
};

struct CounterTimingSession {
    __strong id<MTLCounterSampleBuffer> buffer = nil;
    NSUInteger capacity = 0u;
    NSUInteger nextSample = 0u;
    std::size_t dropped = 0u;
    std::mutex mutex;
    std::vector<CounterTimingRecord> records;
};

inline std::mutex& counterTimingMapMutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::unordered_map<const void*, std::shared_ptr<CounterTimingSession>>&
counterTimingSessions() {
    static std::unordered_map<const void*, std::shared_ptr<CounterTimingSession>> sessions;
    return sessions;
}

inline void eraseCounterTimingSession(const void* commandKey) {
    std::lock_guard<std::mutex> lock(counterTimingMapMutex());
    counterTimingSessions().erase(commandKey);
}

inline std::shared_ptr<CounterTimingSession> counterTimingSession(
    id<MTLCommandBuffer> command, id<MTLDevice> device
) {
    const void* commandKey = (__bridge const void*)command;
    std::lock_guard<std::mutex> mapLock(counterTimingMapMutex());
    auto& sessions = counterTimingSessions();
    if (const auto existing = sessions.find(commandKey); existing != sessions.end())
        return existing->second;

    id<MTLCounterSet> timestampSet = nil;
    for (id<MTLCounterSet> set in device.counterSets) {
        if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) {
            timestampSet = set;
            break;
        }
    }
    if (timestampSet == nil) return {};

    // One shared sample buffer avoids allocating one Metal counter resource
    // per kernel, which previously stopped capture before iterative solves.
    constexpr NSUInteger sampleCapacity = 4096u;
    auto* descriptor = [MTLCounterSampleBufferDescriptor new];
    descriptor.counterSet = timestampSet;
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.sampleCount = sampleCapacity;
    NSError* error = nil;
    id<MTLCounterSampleBuffer> buffer =
        [device newCounterSampleBufferWithDescriptor:descriptor error:&error];
    if (buffer == nil) {
        std::fprintf(stderr,
            "matter_gpu_timing_buffer_unavailable capacity=%lu error=%s\n",
            static_cast<unsigned long>(sampleCapacity),
            (error.localizedDescription.UTF8String != nullptr)
                ? error.localizedDescription.UTF8String : "unknown");
        return {};
    }

    auto session = std::make_shared<CounterTimingSession>();
    session->buffer = buffer;
    session->capacity = sampleCapacity;
    sessions.emplace(commandKey, session);
    [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        std::vector<CounterTimingRecord> records;
        NSUInteger sampleCount = 0u;
        std::size_t dropped = 0u;
        {
            std::lock_guard<std::mutex> lock(session->mutex);
            records = session->records;
            sampleCount = session->nextSample;
            dropped = session->dropped;
        }
        if (completed.status == MTLCommandBufferStatusCompleted && sampleCount > 0u) {
            NSData* data = [session->buffer resolveCounterRange:NSMakeRange(0u, sampleCount)];
            if (data.length == sampleCount * sizeof(MTLCounterResultTimestamp)) {
                const auto* samples =
                    static_cast<const MTLCounterResultTimestamp*>(data.bytes);
                for (const auto& record : records) {
                    const auto begin = samples[record.begin].timestamp;
                    const auto end = samples[record.end].timestamp;
                    if (begin == 0u || end < begin ||
                        begin == MTLCounterErrorValue || end == MTLCounterErrorValue)
                        continue;
                    std::fprintf(stderr,
                        "matter_gpu_stage=%s step=%u elapsed_ns=%llu\n",
                        record.stage.c_str(), record.root,
                        static_cast<unsigned long long>(end - begin));
                }
            } else {
                std::fprintf(stderr,
                    "matter_gpu_timing_resolution_failed samples=%lu bytes=%lu\n",
                    static_cast<unsigned long>(sampleCount),
                    static_cast<unsigned long>(data.length));
            }
        } else if (completed.status != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr,
                "matter_gpu_timing_command_failed status=%lu\n",
                static_cast<unsigned long>(completed.status));
        }
        if (dropped > 0u) {
            std::fprintf(stderr,
                "matter_gpu_timing_samples_dropped count=%lu capacity=%lu\n",
                static_cast<unsigned long>(dropped),
                static_cast<unsigned long>(session->capacity));
        }
        eraseCounterTimingSession(commandKey);
    }];
    return session;
}

// Opt-in, read-only stage timestamps on the existing borrowed command buffer.
// Sampling one accepted root bounds counter allocation and does not read any
// physical buffer. Unavailable timestamps are never reported as zero time.
inline id<MTLComputeCommandEncoder> timedEncoder(
    id<MTLCommandBuffer> command, id<MTLDevice> device,
    const char* stage, unsigned root
) {
    const char* requested = std::getenv("NUMI_MATTER_GPU_TIMING");
    const char* selected = std::getenv("NUMI_MATTER_GPU_TIMING_STAGE");
    if (requested == nullptr || std::strcmp(requested, "1") != 0)
        return [command computeCommandEncoder];
    // Keep the historical root-100 default while allowing an exact native
    // warmup/first-step sample without replaying 100 physical steps.
    unsigned sampledRoot = 100u;
    if (const char* rootSetting =
            std::getenv("NUMI_MATTER_GPU_TIMING_ROOT")) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(rootSetting, &end, 10);
        if (end == rootSetting || *end != '\0' ||
            parsed > static_cast<unsigned long>(~0u))
            return [command computeCommandEncoder];
        sampledRoot = static_cast<unsigned>(parsed);
    }
    if (root != sampledRoot ||
        (selected != nullptr && std::strcmp(selected, stage) != 0) ||
        ![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])
        return [command computeCommandEncoder];

    auto session = counterTimingSession(command, device);
    if (!session) return [command computeCommandEncoder];
    NSUInteger begin = 0u;
    NSUInteger end = 0u;
    {
        std::lock_guard<std::mutex> lock(session->mutex);
        if (session->nextSample + 2u > session->capacity) {
            ++session->dropped;
            return [command computeCommandEncoder];
        }
        begin = session->nextSample++;
        end = session->nextSample++;
        session->records.push_back(CounterTimingRecord{stage, root, begin, end});
    }
    auto* pass = [MTLComputePassDescriptor computePassDescriptor];
    pass.sampleBufferAttachments[0].sampleBuffer = session->buffer;
    pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = begin;
    pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = end;
    return [command computeCommandEncoderWithDescriptor:pass];
}
} // namespace numi::matter::detail
