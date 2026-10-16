#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalTactile.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <new>
#include <ranges>
#include <span>
#include <string>
#include <system_error>
#include <utility>

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace metalrobo {
namespace {

constexpr NSUInteger kMinimumAllocationBytes = 16u;
const char kMetalTactileImageAnchor = 0;

std::string nsString(NSString* value) {
    if (value == nil || value.UTF8String == nullptr) {
        return {};
    }
    return std::string{value.UTF8String};
}

std::string describeError(NSError* error) {
    if (error == nil) {
        return "unknown Metal error";
    }
    const std::string description =
        nsString(error.localizedDescription);
    return description.empty()
        ? nsString(error.description)
        : description;
}

bool regularFile(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) && !error;
}

std::string defaultMetallibPath() {
    Dl_info image{};
    if (dladdr(&kMetalTactileImageAnchor, &image) != 0 &&
        image.dli_fname != nullptr) {
        const std::filesystem::path libraryDirectory =
            std::filesystem::path(image.dli_fname).parent_path();
        const std::array candidates{
            libraryDirectory / "metalrobo/MetalRobo.metallib",
            libraryDirectory.parent_path() /
                "shaders/MetalRobo.metallib",
        };
        for (const std::filesystem::path& candidate : candidates) {
            if (regularFile(candidate)) {
                return candidate.string();
            }
        }
    }
    const std::filesystem::path configured{
        METALROBO_DEFAULT_METALLIB
    };
    return regularFile(configured)
        ? configured.string()
        : std::string{};
}

MetalTactileDiagnostics reject(
    MetalTactileDiagnostics diagnostics,
    const MetalTactileStatus status,
    std::string message
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    return diagnostics;
}

bool checkedMultiply(
    const std::size_t left,
    const std::size_t right,
    std::size_t& result
) {
    if (left != 0u &&
        right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool checkedAdd(
    const std::size_t left,
    const std::size_t right,
    std::size_t& result
) {
    if (right >
        std::numeric_limits<std::size_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

template <typename Value>
bool checkedBytes(
    const std::size_t count,
    std::size_t& result
) {
    return checkedMultiply(count, sizeof(Value), result) &&
        result <= std::numeric_limits<NSUInteger>::max();
}

id<MTLBuffer> makeBuffer(
    id<MTLDevice> device,
    const std::size_t logicalBytes,
    const MTLResourceOptions options,
    NSString* label
) {
    const NSUInteger bytes = static_cast<NSUInteger>(
        std::max<std::size_t>(
            logicalBytes,
            kMinimumAllocationBytes
        )
    );
    id<MTLBuffer> buffer =
        [device newBufferWithLength:bytes options:options];
    buffer.label = label;
    return buffer;
}

id<MTLBuffer> makePrivateBuffer(
    id<MTLDevice> device,
    const std::size_t bytes,
    NSString* label
) {
    return makeBuffer(
        device,
        bytes,
        MTLResourceStorageModePrivate,
        label
    );
}

id<MTLBuffer> makeSharedBuffer(
    id<MTLDevice> device,
    const std::size_t bytes,
    NSString* label
) {
    return makeBuffer(
        device,
        bytes,
        MTLResourceStorageModeShared,
        label
    );
}

template <typename Value>
id<MTLBuffer> makeStaticBuffer(
    id<MTLDevice> device,
    const std::span<const Value> values,
    NSString* label
) {
    std::size_t bytes = 0u;
    if (!checkedBytes<Value>(values.size(), bytes)) {
        return nil;
    }
    id<MTLBuffer> buffer =
        makeSharedBuffer(device, bytes, label);
    if (buffer != nil && !values.empty()) {
        std::memcpy(buffer.contents, values.data(), bytes);
    }
    return buffer;
}

id<MTLComputePipelineState> makePipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* name,
    const bool specializeDebugHits,
    const bool enableDebugHits,
    NSError** error
) {
    id<MTLFunction> function = nil;
    if (specializeDebugHits) {
        MTLFunctionConstantValues* constants =
            [MTLFunctionConstantValues new];
        [constants setConstantValue:&enableDebugHits
                              type:MTLDataTypeBool
                           atIndex:0u];
        function = [library
            newFunctionWithName:name
                 constantValues:constants
                          error:error];
    } else {
        function = [library newFunctionWithName:name];
    }
    if (function == nil) {
        return nil;
    }
    return [device
        newComputePipelineStateWithFunction:function
        error:error];
}

void dispatch1D(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    const NSUInteger count
) {
    if (count == 0u) {
        return;
    }
    const NSUInteger width = std::min<NSUInteger>(
        256u,
        pipeline.maxTotalThreadsPerThreadgroup
    );
    [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
       threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
}

template <typename Value>
void copyShared(
    id<MTLBuffer> buffer,
    const std::span<const Value> values
) {
    if (!values.empty()) {
        std::memcpy(
            buffer.contents,
            values.data(),
            values.size_bytes()
        );
    }
}

} // namespace

namespace detail {

struct MetalTactileBuffers {
    __strong id<MTLBuffer> sensors = nil;
    __strong id<MTLBuffer> samples = nil;
    __strong id<MTLBuffer> targets = nil;
    __strong id<MTLBuffer> shapeToSensor = nil;
    __strong id<MTLBuffer> shapes = nil;
    __strong id<MTLBuffer> geometryHeaders = nil;
    __strong id<MTLBuffer> geometryVertices = nil;
    __strong id<MTLBuffer> convexFaces = nil;
    __strong id<MTLBuffer> meshNodes = nil;
    __strong id<MTLBuffer> meshTriangles = nil;

    __strong id<MTLBuffer> hostBodies = nil;
    __strong id<MTLBuffer> hostContacts = nil;
    __strong id<MTLBuffer> hostContactCounts = nil;
    __strong id<MTLBuffer> hostResetMask = nil;
    __strong id<MTLBuffer> hostFrameIndices = nil;

    __strong id<MTLBuffer> depth = nil;
    __strong id<MTLBuffer> depthVelocity = nil;
    __strong id<MTLBuffer> tangentialMotion = nil;
    __strong id<MTLBuffer> targetLocalAnchor = nil;
    __strong id<MTLBuffer> validity = nil;
    __strong id<MTLBuffer> objectShape = nil;
    __strong id<MTLBuffer> hits = nil;
    __strong id<MTLBuffer> summaries = nil;
    __strong id<MTLBuffer> statuses = nil;

    __strong id<MTLBuffer> previousDepth = nil;
    __strong id<MTLBuffer> previousValidity = nil;
    __strong id<MTLBuffer> previousObject = nil;
    __strong id<MTLBuffer> previousHits = nil;
    __strong id<MTLBuffer> previousTangentialMotion = nil;
    __strong id<MTLBuffer> previousTargetLocalAnchor = nil;

    __strong id<MTLBuffer> readbackDepth = nil;
    __strong id<MTLBuffer> readbackDepthVelocity = nil;
    __strong id<MTLBuffer> readbackTangentialMotion = nil;
    __strong id<MTLBuffer> readbackValidity = nil;
    __strong id<MTLBuffer> readbackObject = nil;
    __strong id<MTLBuffer> readbackHits = nil;
    __strong id<MTLBuffer> readbackSummaries = nil;
    __strong id<MTLBuffer> readbackStatuses = nil;
};

struct MetalTactileState {
    MetalTactileConfig config;
    MetalTactileLayout layout;
    CookedTactileSystem tactile;
    EngineModel model;
    __strong id<MTLDevice> device = nil;
    __strong id<MTLCommandQueue> queue = nil;
    __strong id<MTLLibrary> library = nil;
    __strong id<MTLComputePipelineState> samplePipeline = nil;
    __strong id<MTLComputePipelineState> reducePipeline = nil;
    __strong id<MTLComputePipelineState> commitPipeline = nil;
    MetalTactileBuffers buffers;
    std::uint32_t activeEnvironmentCount = 0u;
    std::uint64_t activeFrameIndex = 0u;
    double activeTimestampSeconds = 0.0;
    bool compiled = false;
    mutable std::mutex mutex;
};

} // namespace detail

namespace {

MetalTactileDiagnostics baseDiagnostics(
    const detail::MetalTactileState& state
) {
    MetalTactileDiagnostics result;
    result.layout = state.layout;
    result.deviceName = nsString(state.device.name);
    return result;
}

MRTactileDispatchGPU makeDispatch(
    const detail::MetalTactileState& state,
    const MetalTactileDeviceFrame& frame
) {
    MRTactileDispatchGPU dispatch{};
    dispatch.counts = {
        frame.environmentCount,
        state.layout.bodyCount,
        state.layout.sensorCount,
        state.layout.sampleCount,
    };
    dispatch.geometryCounts = {
        state.layout.shapeCount,
        static_cast<std::uint32_t>(
            state.model.geometryHeaders.size()
        ),
        static_cast<std::uint32_t>(
            state.model.geometryVertices.size()
        ),
        static_cast<std::uint32_t>(
            state.model.convexFaces.size()
        ),
    };
    dispatch.queryCounts = {
        static_cast<std::uint32_t>(
            state.model.meshBvhNodes.size()
        ),
        static_cast<std::uint32_t>(
            state.model.meshTriangles.size()
        ),
        state.layout.targetCount,
        frame.contacts != nullptr
            ? state.layout.contactCapacityPerEnvironment
            : 0u,
    };
    dispatch.frameAndAbi = {
        static_cast<std::uint32_t>(frame.frameIndex),
        static_cast<std::uint32_t>(frame.frameIndex >> 32u),
        MR_TACTILE_ABI_VERSION,
        0u,
    };
    dispatch.timing = {
        frame.observationTimestepSeconds,
        1.0f / frame.observationTimestepSeconds,
        static_cast<float>(frame.timestampSeconds),
        1.0f / (
            frame.contacts != nullptr
            ? frame.contactImpulseTimestepSeconds
            : frame.observationTimestepSeconds
        ),
    };
    return dispatch;
}

MetalTactileDiagnostics encodeLocked(
    detail::MetalTactileState& state,
    const MetalTactileDeviceFrame& frame,
    id<MTLComputeCommandEncoder> encoder
) {
    MetalTactileDiagnostics diagnostics =
        baseDiagnostics(state);
    if (!state.compiled) {
        return reject(
            std::move(diagnostics),
            MetalTactileStatus::notCompiled,
            "tactile context has not been compiled"
        );
    }
    if (encoder == nil || frame.bodyStates == nullptr ||
        frame.environmentCount == 0u ||
        frame.environmentCount >
            state.layout.environmentCapacity ||
        frame.bodyCount != state.layout.bodyCount ||
        !std::isfinite(frame.observationTimestepSeconds) ||
        !(frame.observationTimestepSeconds > 0.0f) ||
        (frame.contacts != nullptr &&
         (
             !std::isfinite(
                 frame.contactImpulseTimestepSeconds
             ) ||
             !(frame.contactImpulseTimestepSeconds > 0.0f)
         )) ||
        !std::isfinite(frame.timestampSeconds) ||
        (frame.contacts != nullptr &&
         frame.contactCapacityPerEnvironment !=
            state.layout.contactCapacityPerEnvironment) ||
        (frame.contactCounts == nullptr &&
         frame.contacts != nullptr)) {
        return reject(
            std::move(diagnostics),
            MetalTactileStatus::incompatibleState,
            "tactile device frame extents, buffers, or timing are invalid"
        );
    }

    id<MTLBuffer> bodyStates =
        (__bridge id<MTLBuffer>)frame.bodyStates;
    if (frame.resetMask == nullptr) {
        std::memset(
            state.buffers.hostResetMask.contents,
            0,
            static_cast<std::size_t>(frame.environmentCount) *
                sizeof(std::uint32_t)
        );
    }
    id<MTLBuffer> contacts = frame.contacts == nullptr
        ? state.buffers.hostContacts
        : (__bridge id<MTLBuffer>)frame.contacts;
    id<MTLBuffer> contactCounts =
        frame.contactCounts == nullptr
        ? state.buffers.hostContactCounts
        : (__bridge id<MTLBuffer>)frame.contactCounts;
    id<MTLBuffer> resetMask = frame.resetMask == nullptr
        ? state.buffers.hostResetMask
        : (__bridge id<MTLBuffer>)frame.resetMask;
    auto* frameIndices = static_cast<std::uint64_t*>(
        state.buffers.hostFrameIndices.contents
    );
    std::fill_n(
        frameIndices,
        frame.environmentCount,
        frame.frameIndex
    );
    std::size_t requiredBodyBytes = 0u;
    std::size_t requiredContactBytes = 0u;
    std::size_t requiredCountBytes = 0u;
    if (!checkedBytes<MRBodyStateGPU>(
            static_cast<std::size_t>(frame.environmentCount) *
                state.layout.bodyCount,
            requiredBodyBytes
        ) ||
        !checkedBytes<MRTactileContactGPU>(
            static_cast<std::size_t>(frame.environmentCount) *
                state.layout.contactCapacityPerEnvironment,
            requiredContactBytes
        ) ||
        !checkedBytes<std::uint32_t>(
            frame.environmentCount,
            requiredCountBytes
        ) ||
        bodyStates.length < requiredBodyBytes ||
        contacts.length <
            std::max<std::size_t>(
                requiredContactBytes,
                kMinimumAllocationBytes
            ) ||
        contactCounts.length < requiredCountBytes ||
        resetMask.length < requiredCountBytes) {
        return reject(
            std::move(diagnostics),
            MetalTactileStatus::incompatibleState,
            "borrowed tactile buffer is shorter than its declared shape"
        );
    }

    const MRTactileDispatchGPU dispatch =
        makeDispatch(state, frame);
    [encoder setComputePipelineState:state.samplePipeline];
    [encoder setBytes:&dispatch
               length:sizeof(dispatch)
              atIndex:0u];
    [encoder setBuffer:state.buffers.sensors offset:0u atIndex:1u];
    [encoder setBuffer:state.buffers.samples offset:0u atIndex:2u];
    [encoder setBuffer:state.buffers.targets offset:0u atIndex:3u];
    [encoder setBuffer:state.buffers.shapes offset:0u atIndex:4u];
    [encoder setBuffer:state.buffers.geometryHeaders
                 offset:0u
                atIndex:5u];
    [encoder setBuffer:state.buffers.geometryVertices
                 offset:0u
                atIndex:6u];
    [encoder setBuffer:state.buffers.convexFaces
                 offset:0u
                atIndex:7u];
    [encoder setBuffer:state.buffers.meshNodes offset:0u atIndex:8u];
    [encoder setBuffer:state.buffers.meshTriangles
                 offset:0u
                atIndex:9u];
    [encoder setBuffer:bodyStates offset:0u atIndex:10u];
    [encoder setBuffer:resetMask offset:0u atIndex:11u];
    [encoder setBuffer:state.buffers.previousDepth
                 offset:0u
                atIndex:12u];
    [encoder setBuffer:state.buffers.previousValidity
                 offset:0u
                atIndex:13u];
    [encoder setBuffer:state.buffers.previousObject
                 offset:0u
                atIndex:14u];
    [encoder setBuffer:state.buffers.previousHits
                 offset:0u
                atIndex:15u];
    [encoder setBuffer:state.buffers.previousTangentialMotion
                 offset:0u
                atIndex:16u];
    [encoder setBuffer:state.buffers.previousTargetLocalAnchor
                 offset:0u
                atIndex:17u];
    [encoder setBuffer:state.buffers.depth offset:0u atIndex:18u];
    [encoder setBuffer:state.buffers.depthVelocity
                 offset:0u
                atIndex:19u];
    [encoder setBuffer:state.buffers.tangentialMotion
                 offset:0u
                atIndex:20u];
    [encoder setBuffer:state.buffers.targetLocalAnchor
                 offset:0u
                atIndex:21u];
    [encoder setBuffer:state.buffers.validity offset:0u atIndex:22u];
    [encoder setBuffer:state.buffers.objectShape
                 offset:0u
                atIndex:23u];
    [encoder setBuffer:state.buffers.hits offset:0u atIndex:24u];
    [encoder setBuffer:state.buffers.hostFrameIndices
                 offset:0u
                atIndex:25u];
    dispatch1D(
        encoder,
        state.samplePipeline,
        static_cast<NSUInteger>(frame.environmentCount) *
            state.layout.sampleCount
    );
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:state.reducePipeline];
    [encoder setBytes:&dispatch
               length:sizeof(dispatch)
              atIndex:0u];
    [encoder setBuffer:state.buffers.sensors offset:0u atIndex:1u];
    [encoder setBuffer:state.buffers.samples offset:0u atIndex:2u];
    [encoder setBuffer:bodyStates offset:0u atIndex:3u];
    [encoder setBuffer:contacts offset:0u atIndex:4u];
    [encoder setBuffer:contactCounts offset:0u atIndex:5u];
    [encoder setBuffer:resetMask offset:0u atIndex:6u];
    [encoder setBuffer:state.buffers.depth offset:0u atIndex:7u];
    [encoder setBuffer:state.buffers.validity offset:0u atIndex:8u];
    [encoder setBuffer:state.buffers.objectShape offset:0u atIndex:9u];
    [encoder setBuffer:state.buffers.tangentialMotion
                 offset:0u
                atIndex:10u];
    [encoder setBuffer:state.buffers.summaries offset:0u atIndex:11u];
    [encoder setBuffer:state.buffers.statuses offset:0u atIndex:12u];
    [encoder setBuffer:state.buffers.hostFrameIndices
                 offset:0u
                atIndex:13u];
    [encoder setBuffer:state.buffers.shapeToSensor
                 offset:0u
                atIndex:14u];
    dispatch1D(
        encoder,
        state.reducePipeline,
        static_cast<NSUInteger>(frame.environmentCount) *
            state.layout.sensorCount
    );
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:state.commitPipeline];
    [encoder setBytes:&dispatch
               length:sizeof(dispatch)
              atIndex:0u];
    [encoder setBuffer:state.buffers.depth offset:0u atIndex:1u];
    [encoder setBuffer:state.buffers.validity offset:0u atIndex:2u];
    [encoder setBuffer:state.buffers.objectShape offset:0u atIndex:3u];
    [encoder setBuffer:state.buffers.hits offset:0u atIndex:4u];
    [encoder setBuffer:state.buffers.tangentialMotion
                 offset:0u
                atIndex:5u];
    [encoder setBuffer:state.buffers.targetLocalAnchor
                 offset:0u
                atIndex:6u];
    [encoder setBuffer:state.buffers.previousDepth
                 offset:0u
                atIndex:7u];
    [encoder setBuffer:state.buffers.previousValidity
                 offset:0u
                atIndex:8u];
    [encoder setBuffer:state.buffers.previousObject
                 offset:0u
                atIndex:9u];
    [encoder setBuffer:state.buffers.previousHits
                 offset:0u
                atIndex:10u];
    [encoder setBuffer:state.buffers.previousTangentialMotion
                 offset:0u
                atIndex:11u];
    [encoder setBuffer:state.buffers.previousTargetLocalAnchor
                 offset:0u
                atIndex:12u];
    dispatch1D(
        encoder,
        state.commitPipeline,
        static_cast<NSUInteger>(frame.environmentCount) *
            state.layout.sampleCount
    );

    state.activeEnvironmentCount = frame.environmentCount;
    state.activeFrameIndex = frame.frameIndex;
    state.activeTimestampSeconds = frame.timestampSeconds;
    diagnostics.message = "ok";
    return diagnostics;
}

bool allBuffersAllocated(const detail::MetalTactileBuffers& buffers) {
    return buffers.sensors != nil &&
        buffers.samples != nil &&
        buffers.targets != nil &&
        buffers.shapeToSensor != nil &&
        buffers.shapes != nil &&
        buffers.geometryHeaders != nil &&
        buffers.geometryVertices != nil &&
        buffers.convexFaces != nil &&
        buffers.meshNodes != nil &&
        buffers.meshTriangles != nil &&
        buffers.hostBodies != nil &&
        buffers.hostContacts != nil &&
        buffers.hostContactCounts != nil &&
        buffers.hostResetMask != nil &&
        buffers.hostFrameIndices != nil &&
        buffers.depth != nil &&
        buffers.depthVelocity != nil &&
        buffers.tangentialMotion != nil &&
        buffers.targetLocalAnchor != nil &&
        buffers.validity != nil &&
        buffers.objectShape != nil &&
        buffers.hits != nil &&
        buffers.summaries != nil &&
        buffers.statuses != nil &&
        buffers.previousDepth != nil &&
        buffers.previousValidity != nil &&
        buffers.previousObject != nil &&
        buffers.previousHits != nil &&
        buffers.previousTangentialMotion != nil &&
        buffers.previousTargetLocalAnchor != nil &&
        buffers.readbackDepth != nil &&
        buffers.readbackDepthVelocity != nil &&
        buffers.readbackTangentialMotion != nil &&
        buffers.readbackValidity != nil &&
        buffers.readbackObject != nil &&
        buffers.readbackHits != nil &&
        buffers.readbackSummaries != nil &&
        buffers.readbackStatuses != nil;
}

} // namespace

MetalTactileContext::MetalTactileContext(MetalTactileConfig config)
    : state_(std::make_shared<detail::MetalTactileState>()) {
    state_->config = std::move(config);
}

MetalTactileContext::~MetalTactileContext() = default;

MetalTactileContext::MetalTactileContext(
    MetalTactileContext&& other
) noexcept = default;

MetalTactileContext& MetalTactileContext::operator=(
    MetalTactileContext&& other
) noexcept = default;

MetalTactileDiagnostics MetalTactileContext::compile(
    const CookedTactileSystem& tactile,
    const EngineModel& model,
    const std::uint32_t environmentCapacity
) {
    if (state_ == nullptr) {
        return reject(
            {},
            MetalTactileStatus::internalFailure,
            "tactile context state is missing"
        );
    }
    const std::shared_ptr<detail::MetalTactileState> prior = state_;
    std::lock_guard lock(prior->mutex);
    std::string reason;
    if (environmentCapacity == 0u ||
        !tactile.valid(model, &reason)) {
        return reject(
            {},
            MetalTactileStatus::invalidSystem,
            environmentCapacity == 0u
                ? "tactile environment capacity is zero"
                : std::move(reason)
        );
    }
    if (model.bodies.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        model.shapes.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        model.geometryHeaders.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        model.geometryVertices.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        model.convexFaces.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        model.meshBvhNodes.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        model.meshTriangles.size() >
            std::numeric_limits<std::uint32_t>::max()) {
        return reject(
            {},
            MetalTactileStatus::capacityOverflow,
            "tactile model arenas exceed the pointer-free ABI"
        );
    }

    auto candidate =
        std::make_shared<detail::MetalTactileState>();
    candidate->config = prior->config;
    candidate->tactile = tactile;
    candidate->model = model;
    candidate->layout.environmentCapacity = environmentCapacity;
    candidate->layout.bodyCount =
        static_cast<std::uint32_t>(model.bodies.size());
    candidate->layout.shapeCount =
        static_cast<std::uint32_t>(model.shapes.size());
    candidate->layout.sensorCount =
        static_cast<std::uint32_t>(tactile.sensors.size());
    candidate->layout.sampleCount =
        static_cast<std::uint32_t>(tactile.samples.size());
    candidate->layout.targetCount =
        static_cast<std::uint32_t>(
            tactile.targetShapeIndices.size()
        );
    candidate->layout.contactCapacityPerEnvironment =
        candidate->config.contactCapacityPerEnvironment;
    candidate->layout.queryBackend =
        MR_TACTILE_QUERY_METAL_ANALYTIC_BVH4;

    candidate->device = MTLCreateSystemDefaultDevice();
    if (candidate->device == nil) {
        return reject(
            {},
            MetalTactileStatus::metalDeviceUnavailable,
            "no default Metal device is available"
        );
    }
    candidate->queue = [candidate->device newCommandQueue];
    if (candidate->queue == nil) {
        return reject(
            baseDiagnostics(*candidate),
            MetalTactileStatus::metalCommandFailure,
            "could not create tactile Metal command queue"
        );
    }
    if (@available(macOS 13.0, *)) {
        candidate->layout.hardwareRayQueriesAvailable =
            candidate->device.supportsRaytracing;
    }
    const std::string path =
        candidate->config.metallibPath.empty()
        ? defaultMetallibPath()
        : candidate->config.metallibPath;
    if (path.empty() || !regularFile(path)) {
        return reject(
            baseDiagnostics(*candidate),
            MetalTactileStatus::metallibUnavailable,
            "MetalRobo metallib containing tactile kernels is unavailable"
        );
    }
    NSError* error = nil;
    candidate->library = [candidate->device
        newLibraryWithURL:
            [NSURL fileURLWithPath:
                [NSString stringWithUTF8String:path.c_str()]]
        error:&error];
    if (candidate->library == nil) {
        return reject(
            baseDiagnostics(*candidate),
            MetalTactileStatus::metalLibraryFailure,
            describeError(error)
        );
    }
    candidate->samplePipeline = makePipeline(
        candidate->device,
        candidate->library,
        @"mr_tactile_sample",
        true,
        candidate->config.enableDebugHits,
        &error
    );
    candidate->reducePipeline = makePipeline(
        candidate->device,
        candidate->library,
        @"mr_tactile_reduce",
        false,
        false,
        &error
    );
    candidate->commitPipeline = makePipeline(
        candidate->device,
        candidate->library,
        @"mr_tactile_commit_history",
        true,
        candidate->config.enableDebugHits,
        &error
    );
    if (candidate->samplePipeline == nil ||
        candidate->reducePipeline == nil ||
        candidate->commitPipeline == nil) {
        return reject(
            baseDiagnostics(*candidate),
            MetalTactileStatus::metalPipelineFailure,
            error == nil
                ? "one or more tactile Metal functions are missing"
                : describeError(error)
        );
    }

    std::size_t denseCount = 0u;
    std::size_t summaryCount = 0u;
    std::size_t bodyCount = 0u;
    std::size_t contactCount = 0u;
    if (!checkedMultiply(
            environmentCapacity,
            tactile.samples.size(),
            denseCount
        ) ||
        !checkedMultiply(
            environmentCapacity,
            tactile.sensors.size(),
            summaryCount
        ) ||
        !checkedMultiply(
            environmentCapacity,
            model.bodies.size(),
            bodyCount
        ) ||
        !checkedMultiply(
            environmentCapacity,
            candidate->layout.contactCapacityPerEnvironment,
            contactCount
        )) {
        return reject(
            baseDiagnostics(*candidate),
            MetalTactileStatus::capacityOverflow,
            "tactile environment capacity overflows host size_t"
        );
    }
    std::size_t denseFloatBytes = 0u;
    std::size_t denseUintBytes = 0u;
    std::size_t tangentialMotionBytes = 0u;
    std::size_t anchorBytes = 0u;
    std::size_t hitBytes = 0u;
    std::size_t summaryBytes = 0u;
    std::size_t statusBytes = 0u;
    std::size_t bodyBytes = 0u;
    std::size_t contactBytes = 0u;
    std::size_t countBytes = 0u;
    std::size_t frameIndexBytes = 0u;
    if (!checkedBytes<float>(denseCount, denseFloatBytes) ||
        !checkedBytes<std::uint32_t>(denseCount, denseUintBytes) ||
        !checkedBytes<MRTactileTangentialMotionGPU>(
            denseCount,
            tangentialMotionBytes
        ) ||
        !checkedBytes<mr_float4>(denseCount, anchorBytes) ||
        !checkedBytes<MRTactileHitGPU>(denseCount, hitBytes) ||
        !checkedBytes<MRTactileSummaryGPU>(
            summaryCount,
            summaryBytes
        ) ||
        !checkedBytes<MRTactileStatusGPU>(
            summaryCount,
            statusBytes
        ) ||
        !checkedBytes<MRBodyStateGPU>(bodyCount, bodyBytes) ||
        !checkedBytes<MRTactileContactGPU>(
            contactCount,
            contactBytes
        ) ||
        !checkedBytes<std::uint32_t>(
            environmentCapacity,
            countBytes
        ) ||
        !checkedBytes<std::uint64_t>(
            environmentCapacity,
            frameIndexBytes
        )) {
        return reject(
            baseDiagnostics(*candidate),
            MetalTactileStatus::capacityOverflow,
            "tactile buffer byte size exceeds Metal limits"
        );
    }

    detail::MetalTactileBuffers& buffers = candidate->buffers;
    buffers.sensors = makeStaticBuffer(
        candidate->device,
        std::span<const MRTactileSensorGPU>{
            tactile.sensors
        },
        @"MetalRobo tactile sensors"
    );
    buffers.samples = makeStaticBuffer(
        candidate->device,
        std::span<const MRTactileSampleGPU>{
            tactile.samples
        },
        @"MetalRobo tactile samples"
    );
    buffers.targets = makeStaticBuffer(
        candidate->device,
        std::span<const std::uint32_t>{
            tactile.targetShapeIndices
        },
        @"MetalRobo tactile targets"
    );
    buffers.shapeToSensor = makeStaticBuffer(
        candidate->device,
        std::span<const std::uint32_t>{
            tactile.shapeToSensor
        },
        @"MetalRobo tactile shape ownership"
    );
    buffers.shapes = makeStaticBuffer(
        candidate->device,
        std::span<const MRShapeGPU>{model.shapes},
        @"MetalRobo tactile shapes"
    );
    buffers.geometryHeaders = makeStaticBuffer(
        candidate->device,
        std::span<const MRGeometryHeaderGPU>{
            model.geometryHeaders
        },
        @"MetalRobo tactile geometry headers"
    );
    buffers.geometryVertices = makeStaticBuffer(
        candidate->device,
        std::span<const mr_float4>{model.geometryVertices},
        @"MetalRobo tactile geometry vertices"
    );
    buffers.convexFaces = makeStaticBuffer(
        candidate->device,
        std::span<const MRConvexFaceGPU>{model.convexFaces},
        @"MetalRobo tactile convex faces"
    );
    buffers.meshNodes = makeStaticBuffer(
        candidate->device,
        std::span<const MRMeshBVHNodeGPU>{model.meshBvhNodes},
        @"MetalRobo tactile BVH4"
    );
    buffers.meshTriangles = makeStaticBuffer(
        candidate->device,
        std::span<const MRMeshTriangleGPU>{model.meshTriangles},
        @"MetalRobo tactile mesh triangles"
    );
    buffers.hostBodies = makeSharedBuffer(
        candidate->device,
        bodyBytes,
        @"MetalRobo tactile host bodies"
    );
    buffers.hostContacts = makeSharedBuffer(
        candidate->device,
        contactBytes,
        @"MetalRobo tactile host contacts"
    );
    buffers.hostContactCounts = makeSharedBuffer(
        candidate->device,
        countBytes,
        @"MetalRobo tactile host contact counts"
    );
    buffers.hostResetMask = makeSharedBuffer(
        candidate->device,
        countBytes,
        @"MetalRobo tactile host reset mask"
    );
    buffers.hostFrameIndices = makeSharedBuffer(
        candidate->device,
        frameIndexBytes,
        @"MetalRobo tactile host frame indices"
    );
    buffers.depth = makePrivateBuffer(
        candidate->device,
        denseFloatBytes,
        @"MetalRobo tactile depth"
    );
    buffers.depthVelocity = makePrivateBuffer(
        candidate->device,
        denseFloatBytes,
        @"MetalRobo tactile depth velocity"
    );
    buffers.tangentialMotion = makePrivateBuffer(
        candidate->device,
        tangentialMotionBytes,
        @"MetalRobo tactile tangential motion"
    );
    buffers.targetLocalAnchor = makePrivateBuffer(
        candidate->device,
        anchorBytes,
        @"MetalRobo tactile target-local anchors"
    );
    buffers.validity = makePrivateBuffer(
        candidate->device,
        denseUintBytes,
        @"MetalRobo tactile validity"
    );
    buffers.objectShape = makePrivateBuffer(
        candidate->device,
        denseUintBytes,
        @"MetalRobo tactile object IDs"
    );
    const std::size_t debugHitBytes =
        candidate->config.enableDebugHits
        ? hitBytes
        : 0u;
    buffers.hits = makePrivateBuffer(
        candidate->device,
        debugHitBytes,
        @"MetalRobo tactile debug hits"
    );
    buffers.summaries = makePrivateBuffer(
        candidate->device,
        summaryBytes,
        @"MetalRobo tactile summaries"
    );
    buffers.statuses = makePrivateBuffer(
        candidate->device,
        statusBytes,
        @"MetalRobo tactile statuses"
    );
    buffers.previousDepth = makePrivateBuffer(
        candidate->device,
        denseFloatBytes,
        @"MetalRobo tactile history depth"
    );
    buffers.previousValidity = makePrivateBuffer(
        candidate->device,
        denseUintBytes,
        @"MetalRobo tactile history validity"
    );
    buffers.previousObject = makePrivateBuffer(
        candidate->device,
        denseUintBytes,
        @"MetalRobo tactile history object IDs"
    );
    buffers.previousHits = makePrivateBuffer(
        candidate->device,
        debugHitBytes,
        @"MetalRobo tactile history hits"
    );
    buffers.previousTangentialMotion = makePrivateBuffer(
        candidate->device,
        tangentialMotionBytes,
        @"MetalRobo tactile history tangential motion"
    );
    buffers.previousTargetLocalAnchor = makePrivateBuffer(
        candidate->device,
        anchorBytes,
        @"MetalRobo tactile history target-local anchors"
    );
    buffers.readbackDepth = makeSharedBuffer(
        candidate->device,
        denseFloatBytes,
        @"MetalRobo tactile readback depth"
    );
    buffers.readbackDepthVelocity = makeSharedBuffer(
        candidate->device,
        denseFloatBytes,
        @"MetalRobo tactile readback velocity"
    );
    buffers.readbackTangentialMotion = makeSharedBuffer(
        candidate->device,
        tangentialMotionBytes,
        @"MetalRobo tactile readback tangential motion"
    );
    buffers.readbackValidity = makeSharedBuffer(
        candidate->device,
        denseUintBytes,
        @"MetalRobo tactile readback validity"
    );
    buffers.readbackObject = makeSharedBuffer(
        candidate->device,
        denseUintBytes,
        @"MetalRobo tactile readback object IDs"
    );
    buffers.readbackHits = makeSharedBuffer(
        candidate->device,
        debugHitBytes,
        @"MetalRobo tactile readback hits"
    );
    buffers.readbackSummaries = makeSharedBuffer(
        candidate->device,
        summaryBytes,
        @"MetalRobo tactile readback summaries"
    );
    buffers.readbackStatuses = makeSharedBuffer(
        candidate->device,
        statusBytes,
        @"MetalRobo tactile readback statuses"
    );
    if (!allBuffersAllocated(buffers)) {
        return reject(
            baseDiagnostics(*candidate),
            MetalTactileStatus::metalBufferFailure,
            "could not allocate fixed-capacity tactile buffers"
        );
    }
    std::memset(
        buffers.hostContactCounts.contents,
        0,
        countBytes
    );
    std::memset(buffers.hostResetMask.contents, 0, countBytes);

    std::size_t retainedBytes = 0u;
    const std::array<id<MTLBuffer>, 38u> retained{
        buffers.sensors,
        buffers.samples,
        buffers.targets,
        buffers.shapeToSensor,
        buffers.shapes,
        buffers.geometryHeaders,
        buffers.geometryVertices,
        buffers.convexFaces,
        buffers.meshNodes,
        buffers.meshTriangles,
        buffers.hostBodies,
        buffers.hostContacts,
        buffers.hostContactCounts,
        buffers.hostResetMask,
        buffers.hostFrameIndices,
        buffers.depth,
        buffers.depthVelocity,
        buffers.tangentialMotion,
        buffers.targetLocalAnchor,
        buffers.validity,
        buffers.objectShape,
        buffers.hits,
        buffers.summaries,
        buffers.statuses,
        buffers.previousDepth,
        buffers.previousValidity,
        buffers.previousObject,
        buffers.previousHits,
        buffers.previousTangentialMotion,
        buffers.previousTargetLocalAnchor,
        buffers.readbackDepth,
        buffers.readbackDepthVelocity,
        buffers.readbackTangentialMotion,
        buffers.readbackValidity,
        buffers.readbackObject,
        buffers.readbackHits,
        buffers.readbackSummaries,
        buffers.readbackStatuses,
    };
    for (id<MTLBuffer> buffer : retained) {
        if (!checkedAdd(
                retainedBytes,
                buffer.length,
                retainedBytes
            )) {
            return reject(
                baseDiagnostics(*candidate),
                MetalTactileStatus::capacityOverflow,
                "tactile retained-byte sum overflowed"
            );
        }
    }
    if (retainedBytes > candidate->config.maximumRetainedBytes) {
        return reject(
            baseDiagnostics(*candidate),
            MetalTactileStatus::capacityOverflow,
            "tactile retained buffers exceed the configured budget"
        );
    }
    candidate->layout.retainedBytes = retainedBytes;
    candidate->layout.bytesPerEnvironment =
        retainedBytes / environmentCapacity;
    candidate->compiled = true;

    id<MTLCommandBuffer> commandBuffer =
        [candidate->queue commandBuffer];
    id<MTLBlitCommandEncoder> blit =
        [commandBuffer blitCommandEncoder];
    if (commandBuffer == nil || blit == nil) {
        return reject(
            baseDiagnostics(*candidate),
            MetalTactileStatus::metalCommandFailure,
            "could not initialize tactile temporal history"
        );
    }
    [blit fillBuffer:buffers.previousDepth
               range:NSMakeRange(0u, buffers.previousDepth.length)
               value:0u];
    [blit fillBuffer:buffers.previousValidity
               range:NSMakeRange(0u, buffers.previousValidity.length)
               value:0u];
    [blit fillBuffer:buffers.previousObject
               range:NSMakeRange(0u, buffers.previousObject.length)
               value:0xffu];
    [blit fillBuffer:buffers.previousHits
               range:NSMakeRange(0u, buffers.previousHits.length)
               value:0u];
    [blit fillBuffer:buffers.previousTangentialMotion
               range:NSMakeRange(
                   0u,
                   buffers.previousTangentialMotion.length
               )
               value:0u];
    [blit fillBuffer:buffers.previousTargetLocalAnchor
               range:NSMakeRange(
                   0u,
                   buffers.previousTargetLocalAnchor.length
               )
               value:0u];
    [blit endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        return reject(
            baseDiagnostics(*candidate),
            MetalTactileStatus::metalCommandFailure,
            describeError(commandBuffer.error)
        );
    }
    state_ = std::move(candidate);
    MetalTactileDiagnostics diagnostics =
        baseDiagnostics(*state_);
    diagnostics.message = "ok";
    return diagnostics;
}

MetalTactileDiagnostics MetalTactileContext::observe(
    const MetalTactileHostFrame& frame
) {
    if (state_ == nullptr) {
        return reject(
            {},
            MetalTactileStatus::internalFailure,
            "tactile context state is missing"
        );
    }
    std::lock_guard lock(state_->mutex);
    MetalTactileDiagnostics diagnostics =
        baseDiagnostics(*state_);
    if (!state_->compiled) {
        return reject(
            std::move(diagnostics),
            MetalTactileStatus::notCompiled,
            "tactile context has not been compiled"
        );
    }
    const std::size_t expectedBodies =
        static_cast<std::size_t>(frame.environmentCount) *
        state_->layout.bodyCount;
    const std::size_t expectedContacts =
        static_cast<std::size_t>(frame.environmentCount) *
        state_->layout.contactCapacityPerEnvironment;
    if (frame.environmentCount == 0u ||
        frame.environmentCount >
            state_->layout.environmentCapacity ||
        frame.bodies.size() != expectedBodies ||
        (!frame.contacts.empty() &&
         frame.contacts.size() != expectedContacts) ||
        (!frame.contactCounts.empty() &&
         frame.contactCounts.size() != frame.environmentCount) ||
        (frame.contacts.empty() != frame.contactCounts.empty()) ||
        (!frame.resetMask.empty() &&
         frame.resetMask.size() != frame.environmentCount) ||
        !std::isfinite(frame.observationTimestepSeconds) ||
        !(frame.observationTimestepSeconds > 0.0f) ||
        (!frame.contacts.empty() &&
         (
             !std::isfinite(
                 frame.contactImpulseTimestepSeconds
             ) ||
             !(frame.contactImpulseTimestepSeconds > 0.0f)
         )) ||
        !std::isfinite(frame.timestampSeconds)) {
        return reject(
            std::move(diagnostics),
            MetalTactileStatus::incompatibleState,
            "host tactile frame extents are invalid"
        );
    }
    if (std::ranges::any_of(
            frame.contactCounts,
            [&](const std::uint32_t count) {
                return count >
                    state_->layout.contactCapacityPerEnvironment;
            }
        )) {
        return reject(
            std::move(diagnostics),
            MetalTactileStatus::capacityOverflow,
            "host tactile contact count exceeds fixed capacity"
        );
    }
    copyShared(state_->buffers.hostBodies, frame.bodies);
    if (frame.contacts.empty()) {
        std::memset(
            state_->buffers.hostContactCounts.contents,
            0,
            static_cast<std::size_t>(frame.environmentCount) *
                sizeof(std::uint32_t)
        );
    } else {
        copyShared(state_->buffers.hostContacts, frame.contacts);
        copyShared(
            state_->buffers.hostContactCounts,
            frame.contactCounts
        );
    }
    if (frame.resetMask.empty()) {
        std::memset(
            state_->buffers.hostResetMask.contents,
            0,
            static_cast<std::size_t>(frame.environmentCount) *
                sizeof(std::uint32_t)
        );
    } else {
        copyShared(
            state_->buffers.hostResetMask,
            frame.resetMask
        );
    }

    id<MTLCommandBuffer> commandBuffer =
        [state_->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (commandBuffer == nil || encoder == nil) {
        return reject(
            std::move(diagnostics),
            MetalTactileStatus::metalCommandFailure,
            "could not create tactile command encoder"
        );
    }
    MetalTactileDeviceFrame deviceFrame;
    deviceFrame.bodyStates =
        (__bridge void*)state_->buffers.hostBodies;
    deviceFrame.contacts = frame.contacts.empty()
        ? nullptr
        : (__bridge void*)state_->buffers.hostContacts;
    deviceFrame.contactCounts = frame.contacts.empty()
        ? nullptr
        : (__bridge void*)state_->buffers.hostContactCounts;
    deviceFrame.resetMask =
        (__bridge void*)state_->buffers.hostResetMask;
    deviceFrame.environmentCount = frame.environmentCount;
    deviceFrame.bodyCount = state_->layout.bodyCount;
    deviceFrame.contactCapacityPerEnvironment =
        state_->layout.contactCapacityPerEnvironment;
    deviceFrame.observationTimestepSeconds =
        frame.observationTimestepSeconds;
    deviceFrame.contactImpulseTimestepSeconds =
        frame.contacts.empty()
        ? frame.observationTimestepSeconds
        : frame.contactImpulseTimestepSeconds;
    deviceFrame.frameIndex = frame.frameIndex;
    deviceFrame.timestampSeconds = frame.timestampSeconds;
    const auto start = std::chrono::steady_clock::now();
    diagnostics = encodeLocked(*state_, deviceFrame, encoder);
    if (!diagnostics.succeeded()) {
        [encoder endEncoding];
        return diagnostics;
    }
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    const auto end = std::chrono::steady_clock::now();
    diagnostics.elapsedMilliseconds =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        return reject(
            std::move(diagnostics),
            MetalTactileStatus::metalCommandFailure,
            describeError(commandBuffer.error)
        );
    }
    diagnostics.message = "ok";
    return diagnostics;
}

MetalTactileDiagnostics MetalTactileContext::encode(
    const MetalTactileDeviceFrame& frame,
    void* metalComputeCommandEncoder
) {
    if (state_ == nullptr) {
        return reject(
            {},
            MetalTactileStatus::internalFailure,
            "tactile context state is missing"
        );
    }
    std::lock_guard lock(state_->mutex);
    return encodeLocked(
        *state_,
        frame,
        (__bridge id<MTLComputeCommandEncoder>)
            metalComputeCommandEncoder
    );
}

MetalTactileDiagnostics MetalTactileContext::readback(
    const std::uint32_t environmentCount,
    TactileObservationBatch& output
) {
    if (state_ == nullptr) {
        return reject(
            {},
            MetalTactileStatus::internalFailure,
            "tactile context state is missing"
        );
    }
    std::lock_guard lock(state_->mutex);
    MetalTactileDiagnostics diagnostics =
        baseDiagnostics(*state_);
    if (!state_->compiled) {
        return reject(
            std::move(diagnostics),
            MetalTactileStatus::notCompiled,
            "tactile context has not been compiled"
        );
    }
    if (environmentCount == 0u ||
        environmentCount > state_->activeEnvironmentCount) {
        return reject(
            std::move(diagnostics),
            MetalTactileStatus::incompatibleState,
            "readback environment count exceeds the active frame"
        );
    }
    const std::size_t denseCount =
        static_cast<std::size_t>(environmentCount) *
        state_->layout.sampleCount;
    const std::size_t summaryCount =
        static_cast<std::size_t>(environmentCount) *
        state_->layout.sensorCount;
    const std::size_t depthBytes = denseCount * sizeof(float);
    const std::size_t tangentialMotionBytes =
        denseCount * sizeof(MRTactileTangentialMotionGPU);
    const std::size_t uintBytes =
        denseCount * sizeof(std::uint32_t);
    const std::size_t hitBytes =
        state_->config.enableDebugHits
        ? denseCount * sizeof(MRTactileHitGPU)
        : 0u;
    const std::size_t summaryBytes =
        summaryCount * sizeof(MRTactileSummaryGPU);
    const std::size_t statusBytes =
        summaryCount * sizeof(MRTactileStatusGPU);
    id<MTLCommandBuffer> commandBuffer =
        [state_->queue commandBuffer];
    id<MTLBlitCommandEncoder> blit =
        [commandBuffer blitCommandEncoder];
    if (commandBuffer == nil || blit == nil) {
        return reject(
            std::move(diagnostics),
            MetalTactileStatus::metalCommandFailure,
            "could not create tactile readback blit"
        );
    }
    [blit copyFromBuffer:state_->buffers.depth
            sourceOffset:0u
                toBuffer:state_->buffers.readbackDepth
       destinationOffset:0u
                    size:depthBytes];
    [blit copyFromBuffer:state_->buffers.depthVelocity
            sourceOffset:0u
                toBuffer:state_->buffers.readbackDepthVelocity
       destinationOffset:0u
                    size:depthBytes];
    [blit copyFromBuffer:state_->buffers.tangentialMotion
            sourceOffset:0u
                toBuffer:state_->buffers.readbackTangentialMotion
       destinationOffset:0u
                    size:tangentialMotionBytes];
    [blit copyFromBuffer:state_->buffers.validity
            sourceOffset:0u
                toBuffer:state_->buffers.readbackValidity
       destinationOffset:0u
                    size:uintBytes];
    [blit copyFromBuffer:state_->buffers.objectShape
            sourceOffset:0u
                toBuffer:state_->buffers.readbackObject
       destinationOffset:0u
                    size:uintBytes];
    if (hitBytes > 0u) {
        [blit copyFromBuffer:state_->buffers.hits
                sourceOffset:0u
                    toBuffer:state_->buffers.readbackHits
           destinationOffset:0u
                        size:hitBytes];
    }
    [blit copyFromBuffer:state_->buffers.summaries
            sourceOffset:0u
                toBuffer:state_->buffers.readbackSummaries
       destinationOffset:0u
                    size:summaryBytes];
    [blit copyFromBuffer:state_->buffers.statuses
            sourceOffset:0u
                toBuffer:state_->buffers.readbackStatuses
       destinationOffset:0u
                    size:statusBytes];
    [blit endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        return reject(
            std::move(diagnostics),
            MetalTactileStatus::metalCommandFailure,
            describeError(commandBuffer.error)
        );
    }

    TactileObservationBatch candidate;
    try {
        candidate.environmentCount = environmentCount;
        candidate.sensorCount = state_->layout.sensorCount;
        candidate.sampleCount = state_->layout.sampleCount;
        candidate.frameIndex = state_->activeFrameIndex;
        candidate.timestampSeconds = state_->activeTimestampSeconds;
        candidate.penetrationDepthMeters.resize(denseCount);
        candidate.depthVelocityMetersPerSecond.resize(denseCount);
        candidate.tangentialMotion.resize(denseCount);
        candidate.validity.resize(denseCount);
        candidate.objectShapeIds.resize(denseCount);
        if (state_->config.enableDebugHits) {
            candidate.debugHits.resize(denseCount);
        }
        candidate.summaries.resize(summaryCount);
        candidate.statuses.resize(summaryCount);
    } catch (const std::bad_alloc&) {
        return reject(
            std::move(diagnostics),
            MetalTactileStatus::capacityOverflow,
            "host allocation failed during tactile readback"
        );
    }
    std::memcpy(
        candidate.penetrationDepthMeters.data(),
        state_->buffers.readbackDepth.contents,
        depthBytes
    );
    std::memcpy(
        candidate.depthVelocityMetersPerSecond.data(),
        state_->buffers.readbackDepthVelocity.contents,
        depthBytes
    );
    std::memcpy(
        candidate.tangentialMotion.data(),
        state_->buffers.readbackTangentialMotion.contents,
        tangentialMotionBytes
    );
    std::memcpy(
        candidate.validity.data(),
        state_->buffers.readbackValidity.contents,
        uintBytes
    );
    std::memcpy(
        candidate.objectShapeIds.data(),
        state_->buffers.readbackObject.contents,
        uintBytes
    );
    if (hitBytes > 0u) {
        std::memcpy(
            candidate.debugHits.data(),
            state_->buffers.readbackHits.contents,
            hitBytes
        );
    }
    std::memcpy(
        candidate.summaries.data(),
        state_->buffers.readbackSummaries.contents,
        summaryBytes
    );
    std::memcpy(
        candidate.statuses.data(),
        state_->buffers.readbackStatuses.contents,
        statusBytes
    );
    output = std::move(candidate);
    diagnostics.message = "ok";
    return diagnostics;
}

MetalTactileDiagnostics MetalTactileContext::clearHistory() {
    if (state_ == nullptr) {
        return reject(
            {},
            MetalTactileStatus::internalFailure,
            "tactile context state is missing"
        );
    }
    std::lock_guard lock(state_->mutex);
    MetalTactileDiagnostics diagnostics =
        baseDiagnostics(*state_);
    if (!state_->compiled) {
        return reject(
            std::move(diagnostics),
            MetalTactileStatus::notCompiled,
            "tactile context has not been compiled"
        );
    }
    id<MTLCommandBuffer> commandBuffer =
        [state_->queue commandBuffer];
    id<MTLBlitCommandEncoder> blit =
        [commandBuffer blitCommandEncoder];
    if (commandBuffer == nil || blit == nil) {
        return reject(
            std::move(diagnostics),
            MetalTactileStatus::metalCommandFailure,
            "could not create tactile history-clear blit"
        );
    }
    [blit fillBuffer:state_->buffers.previousDepth
               range:NSMakeRange(
                   0u,
                   state_->buffers.previousDepth.length
               )
               value:0u];
    [blit fillBuffer:state_->buffers.previousValidity
               range:NSMakeRange(
                   0u,
                   state_->buffers.previousValidity.length
               )
               value:0u];
    [blit fillBuffer:state_->buffers.previousObject
               range:NSMakeRange(
                   0u,
                   state_->buffers.previousObject.length
               )
               value:0xffu];
    [blit fillBuffer:state_->buffers.previousHits
               range:NSMakeRange(
                   0u,
                   state_->buffers.previousHits.length
               )
               value:0u];
    [blit fillBuffer:state_->buffers.previousTangentialMotion
               range:NSMakeRange(
                   0u,
                   state_->buffers.previousTangentialMotion.length
               )
               value:0u];
    [blit fillBuffer:state_->buffers.previousTargetLocalAnchor
               range:NSMakeRange(
                   0u,
                   state_->buffers.previousTargetLocalAnchor.length
               )
               value:0u];
    [blit endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        return reject(
            std::move(diagnostics),
            MetalTactileStatus::metalCommandFailure,
            describeError(commandBuffer.error)
        );
    }
    diagnostics.message = "ok";
    return diagnostics;
}

MetalTactileLayout MetalTactileContext::layout() const noexcept {
    if (state_ == nullptr) {
        return {};
    }
    std::lock_guard lock(state_->mutex);
    return state_->layout;
}

void* MetalTactileContext::nativeBuffer(
    const MetalTactileBuffer buffer
) const noexcept {
    if (state_ == nullptr) {
        return nullptr;
    }
    std::lock_guard lock(state_->mutex);
    id<MTLBuffer> result = nil;
    switch (buffer) {
    case MetalTactileBuffer::penetrationDepth:
        result = state_->buffers.depth;
        break;
    case MetalTactileBuffer::depthVelocity:
        result = state_->buffers.depthVelocity;
        break;
    case MetalTactileBuffer::validity:
        result = state_->buffers.validity;
        break;
    case MetalTactileBuffer::objectShapeIds:
        result = state_->buffers.objectShape;
        break;
    case MetalTactileBuffer::debugHits:
        result = state_->config.enableDebugHits
            ? state_->buffers.hits
            : nil;
        break;
    case MetalTactileBuffer::summaries:
        result = state_->buffers.summaries;
        break;
    case MetalTactileBuffer::statuses:
        result = state_->buffers.statuses;
        break;
    case MetalTactileBuffer::tangentialMotion:
        result = state_->buffers.tangentialMotion;
        break;
    }
    return result == nil ? nullptr : (__bridge void*)result;
}

const char*
metalTactileStatusName(const MetalTactileStatus status) noexcept {
    switch (status) {
    case MetalTactileStatus::success:
        return "success";
    case MetalTactileStatus::invalidSystem:
        return "invalid_system";
    case MetalTactileStatus::invalidConfiguration:
        return "invalid_configuration";
    case MetalTactileStatus::capacityOverflow:
        return "capacity_overflow";
    case MetalTactileStatus::metallibUnavailable:
        return "metallib_unavailable";
    case MetalTactileStatus::metalDeviceUnavailable:
        return "metal_device_unavailable";
    case MetalTactileStatus::metalLibraryFailure:
        return "metal_library_failure";
    case MetalTactileStatus::metalPipelineFailure:
        return "metal_pipeline_failure";
    case MetalTactileStatus::metalBufferFailure:
        return "metal_buffer_failure";
    case MetalTactileStatus::metalCommandFailure:
        return "metal_command_failure";
    case MetalTactileStatus::incompatibleState:
        return "incompatible_state";
    case MetalTactileStatus::notCompiled:
        return "not_compiled";
    case MetalTactileStatus::internalFailure:
        return "internal_failure";
    }
    return "unknown";
}

} // namespace metalrobo
