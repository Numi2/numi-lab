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

// A stage span starts at one encoder's start timestamp and closes at a
// later encoder's start timestamp. Runtime may split the work into multiple
// encoders between these boundaries; all remain inside the measured interval.
struct TimingSpan {
    std::shared_ptr<CounterTimingSession> session;
    const void* commandKey = nullptr;
    std::string stage;
    unsigned root = 0u;
    NSUInteger begin = 0u;
    bool active = false;
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
    NSUInteger sampleCapacity = 4096u;
    if (const char* setting = std::getenv("NUMI_MATTER_GPU_TIMING_CAPACITY")) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(setting, &end, 10);
        if (*setting < '0' || *setting > '9' || end == setting ||
            *end != '\0' || parsed < 1024u || parsed > 262144u) {
            std::fprintf(stderr,
                "matter_gpu_timing_buffer_unavailable reason=invalid_capacity expected=1024..262144\n");
            return {};
        }
        sampleCapacity = static_cast<NSUInteger>(parsed);
    }
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
                std::size_t invalidRecords = 0u;
                for (const auto& record : records) {
                    const auto begin = samples[record.begin].timestamp;
                    const auto end = samples[record.end].timestamp;
                    if (begin == 0u || end < begin ||
                        begin == MTLCounterErrorValue || end == MTLCounterErrorValue) {
                        if (invalidRecords++ < 8u)
                            std::fprintf(stderr,
                                "matter_gpu_timing_record_unavailable stage=%s step=%u begin=%llu end=%llu\n",
                                record.stage.c_str(), record.root,
                                static_cast<unsigned long long>(begin),
                                static_cast<unsigned long long>(end));
                        continue;
                    }
                    std::fprintf(stderr,
                        "matter_gpu_stage=%s step=%u elapsed_ns=%llu\n",
                        record.stage.c_str(), record.root,
                        static_cast<unsigned long long>(end - begin));
                }
                if (invalidRecords != 0u)
                    std::fprintf(stderr,
                        "matter_gpu_timing_invalid_records count=%lu\n",
                        static_cast<unsigned long>(invalidRecords));
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

// Close the current GPU-time span at the start of the returned encoder, then
// optionally open stage at that same timestamp. The returned encoder is the
// next encoder in the caller's original command-buffer timeline, so it can
// immediately receive the stage's normal work. Pass nullptr to close a span
// at a final marker encoder. If sampling is disabled or unavailable this
// returns a plain encoder and emits no fabricated elapsed time.
inline id<MTLComputeCommandEncoder> timingSpanBoundary(
    id<MTLCommandBuffer> command, id<MTLDevice> device,
    unsigned root, const char* stage, TimingSpan& span,
    bool* sampledBoundary = nullptr
) {
    if (sampledBoundary != nullptr) *sampledBoundary = false;
    if (command == nil) return nil;
    const void* commandKey = (__bridge const void*)command;
    const bool sameCommand = span.active && span.commandKey == commandKey;
    if (span.active && !sameCommand) {
        std::fprintf(stderr,
            "matter_gpu_stage_incomplete stage=%s step=%u reason=command_buffer_changed\n",
            span.stage.c_str(), span.root);
        if (span.session) {
            std::lock_guard<std::mutex> lock(span.session->mutex);
            ++span.session->dropped;
        }
        span.session.reset();
        span.commandKey = nullptr;
        span.stage.clear();
        span.active = false;
    }

    const char* requested = std::getenv("NUMI_MATTER_GPU_TIMING");
    const char* selected = std::getenv("NUMI_MATTER_GPU_TIMING_STAGE");
    bool rootConfigValid = true;
    unsigned sampledRoot = 100u;
    if (const char* rootSetting =
            std::getenv("NUMI_MATTER_GPU_TIMING_ROOT")) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(rootSetting, &end, 10);
        if (end == rootSetting || *end != 0 ||
            parsed > static_cast<unsigned long>(~0u)) {
            rootConfigValid = false;
        } else {
            sampledRoot = static_cast<unsigned>(parsed);
        }
    }
    bool openRequested = requested != nullptr &&
        std::strcmp(requested, "1") == 0 && rootConfigValid &&
        root == sampledRoot && stage != nullptr &&
        (selected == nullptr || std::strcmp(selected, stage) == 0);
    if (openRequested &&
        (device == nil || ![device supportsCounterSampling:
            MTLCounterSamplingPointAtStageBoundary])) {
        std::fprintf(stderr,
            "matter_gpu_timing_span_unavailable stage=%s step=%u reason=counter_sampling_unsupported\n",
            stage, root);
        openRequested = false;
    }

    std::shared_ptr<CounterTimingSession> session =
        sameCommand ? span.session : std::shared_ptr<CounterTimingSession>{};
    if (!session && openRequested)
        session = counterTimingSession(command, device);
    if (!session) {
        if (openRequested)
            std::fprintf(stderr,
                "matter_gpu_timing_span_unavailable stage=%s step=%u reason=counter_session_unavailable\n",
                stage, root);
        // An active span always owns its session; if it was lost, report and
        // reset rather than creating a cross-command range.
        if (span.active) {
            span.active = false;
            span.session.reset();
            span.commandKey = nullptr;
            span.stage.clear();
        }
        return [command computeCommandEncoder];
    }

    const bool closesActive = span.active && sameCommand;
    const bool opensNext = openRequested;
    if (!closesActive && !opensNext)
        return [command computeCommandEncoder];

    NSUInteger boundarySample = 0u;
    bool sampleAvailable = false;
    {
        std::lock_guard<std::mutex> lock(session->mutex);
        if (session->nextSample < session->capacity) {
            boundarySample = session->nextSample++;
            sampleAvailable = true;
        } else {
            if (closesActive) ++session->dropped;
            if (opensNext) ++session->dropped;
        }
    }
    if (!sampleAvailable) {
        if (closesActive)
            std::fprintf(stderr,
                "matter_gpu_stage_incomplete stage=%s step=%u reason=counter_capacity\n",
                span.stage.c_str(), span.root);
        if (opensNext)
            std::fprintf(stderr,
                "matter_gpu_timing_span_unavailable stage=%s step=%u reason=counter_capacity\n",
                stage, root);
        if (closesActive) {
            span.active = false;
            span.session.reset();
            span.commandKey = nullptr;
            span.stage.clear();
        }
        return [command computeCommandEncoder];
    }
    if (sampledBoundary != nullptr) *sampledBoundary = true;

    if (closesActive) {
        std::lock_guard<std::mutex> lock(session->mutex);
        session->records.push_back(CounterTimingRecord{
            span.stage, span.root, span.begin, boundarySample});
        span.active = false;
        span.session.reset();
        span.commandKey = nullptr;
        span.stage.clear();
    }
    if (opensNext) {
        span.session = session;
        span.commandKey = commandKey;
        span.stage = stage;
        span.root = root;
        span.begin = boundarySample;
        span.active = true;
    }

    auto* pass = [MTLComputePassDescriptor computePassDescriptor];
    pass.sampleBufferAttachments[0].sampleBuffer = session->buffer;
    pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = boundarySample;
    pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = MTLCounterDontSample;
    return [command computeCommandEncoderWithDescriptor:pass];
}
} // namespace numi::matter::detail
