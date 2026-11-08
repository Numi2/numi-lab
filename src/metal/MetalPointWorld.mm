#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalPointWorld.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <ranges>
#include <utility>

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace metalrobo {
namespace {

constexpr std::uint32_t kWidth = 320u;
constexpr std::uint32_t kHeight = 180u;
constexpr std::size_t kPixels = static_cast<std::size_t>(kWidth) * kHeight;

std::string stringValue(NSString* value) {
    return value == nil || value.UTF8String == nullptr ? std::string{} : std::string{value.UTF8String};
}

MetalPointWorldDiagnostics reject(MetalPointWorldStatus status, std::string message) {
    MetalPointWorldDiagnostics result;
    result.status = status;
    result.message = std::move(message);
    return result;
}

id<MTLBuffer> buffer(id<MTLDevice> device, const NSUInteger length, NSString* label) {
    id<MTLBuffer> result = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
    result.label = label;
    return result;
}

} // namespace

namespace detail {

struct MetalPointWorldPreprocessorState {
    __strong id<MTLDevice> device = nil;
    __strong id<MTLCommandQueue> queue = nil;
    __strong id<MTLLibrary> library = nil;
    __strong id<MTLComputePipelineState> normalize = nil;
    __strong id<MTLComputePipelineState> backproject = nil;
    __strong id<MTLBuffer> rgba = nil;
    __strong id<MTLBuffer> depth = nil;
    __strong id<MTLBuffer> validity = nil;
    __strong id<MTLBuffer> normalized = nil;
    __strong id<MTLBuffer> points = nil;
    std::string initializationError;
    std::size_t retainedBytes = 0u;
    std::mutex mutex;
};

} // namespace detail

MetalPointWorldPreprocessor::MetalPointWorldPreprocessor(std::string metallibPath)
    : state_(std::make_shared<detail::MetalPointWorldPreprocessorState>()) {
    auto& state = *state_;
    state.device = MTLCreateSystemDefaultDevice();
    if (state.device == nil) {
        state.initializationError = "no Apple Metal device is available";
        return;
    }
    state.queue = [state.device newCommandQueue];
    if (state.queue == nil) {
        state.initializationError = "could not create persistent PointWorld command queue";
        return;
    }
    if (metallibPath.empty()) {
        metallibPath = METALROBO_DEFAULT_METALLIB;
    }
    NSError* error = nil;
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallibPath.c_str()]];
    state.library = [state.device newLibraryWithURL:url error:&error];
    if (state.library == nil) {
        state.initializationError = error == nil ? "PointWorld metallib is unavailable" : stringValue(error.localizedDescription);
        return;
    }
    id<MTLFunction> normalizeFunction = [state.library newFunctionWithName:@"mr_pointworld_normalize_rgb"];
    id<MTLFunction> backprojectFunction = [state.library newFunctionWithName:@"mr_pointworld_backproject_depth"];
    state.normalize = normalizeFunction == nil ? nil : [state.device newComputePipelineStateWithFunction:normalizeFunction error:&error];
    state.backproject = backprojectFunction == nil ? nil : [state.device newComputePipelineStateWithFunction:backprojectFunction error:&error];
    if (state.normalize == nil || state.backproject == nil) {
        state.initializationError = error == nil ? "PointWorld preprocessing functions are missing" : stringValue(error.localizedDescription);
        return;
    }
    state.rgba = buffer(state.device, kPixels * 4u, @"PointWorld RGBA staging");
    state.depth = buffer(state.device, kPixels * sizeof(float), @"PointWorld depth staging");
    state.validity = buffer(state.device, kPixels, @"PointWorld validity staging");
    state.normalized = buffer(state.device, kPixels * sizeof(float) * 4u, @"PointWorld normalized RGB");
    state.points = buffer(state.device, kPixels * sizeof(float) * 4u, @"PointWorld camera points");
    if (state.rgba == nil || state.depth == nil || state.validity == nil || state.normalized == nil || state.points == nil) {
        state.initializationError = "could not allocate persistent PointWorld preprocessing buffers";
        return;
    }
    state.retainedBytes = kPixels * (4u + sizeof(float) + 1u + 2u * sizeof(float) * 4u);
}

MetalPointWorldPreprocessor::~MetalPointWorldPreprocessor() = default;
MetalPointWorldPreprocessor::MetalPointWorldPreprocessor(MetalPointWorldPreprocessor&&) noexcept = default;
MetalPointWorldPreprocessor& MetalPointWorldPreprocessor::operator=(MetalPointWorldPreprocessor&&) noexcept = default;

MetalPointWorldDiagnostics MetalPointWorldPreprocessor::encode(
    const PointWorldPreprocessDeviceFrame& frame,
    void* metalComputeCommandEncoder
) {
    if (!state_) {
        return reject(MetalPointWorldStatus::metalUnavailable, "PointWorld preprocessing state is unavailable");
    }
    auto& state = *state_;
    if (!state.initializationError.empty()) {
        return reject(MetalPointWorldStatus::pipelineFailure, state.initializationError);
    }
    if (frame.width != kWidth || frame.height != kHeight || frame.rgba == nullptr || frame.metricDepth == nullptr ||
        frame.depthValidity == nullptr || frame.normalizedRGBA == nullptr || frame.cameraPoints == nullptr ||
        !(frame.intrinsics[0] > 0.0f) || !(frame.intrinsics[1] > 0.0f) ||
        !std::ranges::all_of(frame.intrinsics, [](const float value) { return std::isfinite(value); }) ||
        metalComputeCommandEncoder == nullptr) {
        return reject(MetalPointWorldStatus::invalidInput, "PointWorld device frame must be complete calibrated 320x180 RGB-D");
    }
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)metalComputeCommandEncoder;
    id<MTLBuffer> rgba = (__bridge id<MTLBuffer>)frame.rgba;
    id<MTLBuffer> depth = (__bridge id<MTLBuffer>)frame.metricDepth;
    id<MTLBuffer> validity = (__bridge id<MTLBuffer>)frame.depthValidity;
    id<MTLBuffer> normalized = (__bridge id<MTLBuffer>)frame.normalizedRGBA;
    id<MTLBuffer> points = (__bridge id<MTLBuffer>)frame.cameraPoints;
    if (rgba.length < kPixels * 4u || depth.length < kPixels * sizeof(float) ||
        validity.length < kPixels || normalized.length < kPixels * sizeof(float) * 4u ||
        points.length < kPixels * sizeof(float) * 4u) {
        return reject(MetalPointWorldStatus::invalidInput, "PointWorld borrowed Metal buffer capacity is insufficient");
    }
    const std::uint32_t pixelCount = static_cast<std::uint32_t>(kPixels);
    const std::array<std::uint32_t, 2u> imageSize{kWidth, kHeight};
    MTLSize grid = MTLSizeMake(kPixels, 1u, 1u);
    const NSUInteger normalizeWidth = std::min<NSUInteger>(state.normalize.maxTotalThreadsPerThreadgroup, 256u);
    MTLSize group = MTLSizeMake(normalizeWidth, 1u, 1u);
    [encoder setComputePipelineState:state.normalize];
    [encoder setBuffer:rgba offset:0 atIndex:0];
    [encoder setBuffer:normalized offset:0 atIndex:1];
    [encoder setBytes:&pixelCount length:sizeof(pixelCount) atIndex:2];
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    [encoder setComputePipelineState:state.backproject];
    [encoder setBuffer:depth offset:0 atIndex:0];
    [encoder setBuffer:validity offset:0 atIndex:1];
    [encoder setBuffer:points offset:0 atIndex:2];
    [encoder setBytes:frame.intrinsics.data() length:sizeof(float) * frame.intrinsics.size() atIndex:3];
    [encoder setBytes:&imageSize length:sizeof(imageSize) atIndex:4];
    const NSUInteger backprojectWidth = std::min<NSUInteger>(state.backproject.maxTotalThreadsPerThreadgroup, 256u);
    group = MTLSizeMake(backprojectWidth, 1u, 1u);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    MetalPointWorldDiagnostics result;
    result.deviceName = stringValue(state.device.name);
    result.retainedBytes = state.retainedBytes;
    return result;
}

MetalPointWorldDiagnostics MetalPointWorldPreprocessor::run(
    const PointWorldPreprocessHostInput& input,
    PointWorldPreprocessResult& output
) {
    if (!state_) {
        return reject(MetalPointWorldStatus::metalUnavailable, "PointWorld preprocessing state is unavailable");
    }
    auto& state = *state_;
    std::scoped_lock lock{state.mutex};
    if (input.width != kWidth || input.height != kHeight || input.rgba.size() != kPixels * 4u ||
        input.metricDepth.size() != kPixels || input.depthValidity.size() != kPixels) {
        return reject(MetalPointWorldStatus::invalidInput, "PointWorld host input extents are invalid");
    }
    if (!state.initializationError.empty()) {
        return reject(MetalPointWorldStatus::pipelineFailure, state.initializationError);
    }
    std::memcpy(state.rgba.contents, input.rgba.data(), input.rgba.size());
    std::memcpy(state.depth.contents, input.metricDepth.data(), input.metricDepth.size_bytes());
    std::memcpy(state.validity.contents, input.depthValidity.data(), input.depthValidity.size());
    id<MTLCommandBuffer> command = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (command == nil || encoder == nil) {
        return reject(MetalPointWorldStatus::commandFailure, "could not create PointWorld preprocessing command encoder");
    }
    const auto begin = std::chrono::steady_clock::now();
    PointWorldPreprocessDeviceFrame frame{
        .rgba = (__bridge void*)state.rgba, .metricDepth = (__bridge void*)state.depth,
        .depthValidity = (__bridge void*)state.validity, .normalizedRGBA = (__bridge void*)state.normalized,
        .cameraPoints = (__bridge void*)state.points, .intrinsics = input.intrinsics,
        .width = input.width, .height = input.height,
    };
    MetalPointWorldDiagnostics diagnostics = encode(frame, (__bridge void*)encoder);
    if (!diagnostics.succeeded()) {
        [encoder endEncoding];
        return diagnostics;
    }
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted || command.error != nil) {
        return reject(MetalPointWorldStatus::commandFailure, command.error == nil ? "PointWorld Metal command failed" : stringValue(command.error.localizedDescription));
    }
    output.normalizedRGBA.resize(kPixels);
    output.cameraPoints.resize(kPixels);
    std::memcpy(output.normalizedRGBA.data(), state.normalized.contents, kPixels * sizeof(float) * 4u);
    std::memcpy(output.cameraPoints.data(), state.points.contents, kPixels * sizeof(float) * 4u);
    diagnostics.elapsedMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    return diagnostics;
}

const char* metalPointWorldStatusName(const MetalPointWorldStatus status) noexcept {
    switch (status) {
    case MetalPointWorldStatus::success: return "success";
    case MetalPointWorldStatus::invalidInput: return "invalid_input";
    case MetalPointWorldStatus::metalUnavailable: return "metal_unavailable";
    case MetalPointWorldStatus::libraryFailure: return "library_failure";
    case MetalPointWorldStatus::pipelineFailure: return "pipeline_failure";
    case MetalPointWorldStatus::bufferFailure: return "buffer_failure";
    case MetalPointWorldStatus::commandFailure: return "command_failure";
    }
    return "unknown";
}

} // namespace metalrobo
