#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/Runtime.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace metalrobo {
namespace {

constexpr NSUInteger kThreadsPerEnvironment = MR_SIMD_WIDTH;
constexpr NSUInteger kBufferCount = 18u;

std::string nsString(const NSString* value) {
    if (value == nil) {
        return {};
    }
    const char* utf8 = value.UTF8String;
    return utf8 == nullptr ? std::string{} : std::string{utf8};
}

std::string describeError(const NSError* error) {
    if (error == nil) {
        return "unknown Metal error";
    }
    std::string description = nsString(error.localizedDescription);
    if (description.empty()) {
        description = nsString(error.description);
    }
    return description.empty() ? "unknown Metal error" : description;
}

std::uint64_t checkedMultiply(
    const std::string& label,
    const std::initializer_list<std::uint64_t> factors
) {
    std::uint64_t value = 1u;
    for (const std::uint64_t factor : factors) {
        if (factor != 0u &&
            value > std::numeric_limits<std::uint64_t>::max() / factor) {
            throw std::overflow_error(
                "MetalRobo buffer size overflow for " + label
            );
        }
        value *= factor;
    }
    return value;
}

struct BufferRequirement {
    const char* label;
    std::uint64_t bytes;
};

class MetalRuntime final : public Runtime {
public:
    MetalRuntime(Model model, const RuntimeDescriptor& descriptor)
        : model_(std::move(model)),
          descriptor_(descriptor),
          environmentCount_(descriptor.environmentCount) {
        @autoreleasepool {
            initialize();
            reset(descriptor_.seed);
        }
    }

    ~MetalRuntime() override = default;

    [[nodiscard]] const Model& model() const noexcept override {
        return model_;
    }

    [[nodiscard]] std::uint32_t environmentCount() const noexcept override {
        return environmentCount_;
    }

    [[nodiscard]] std::span<const float> observations() const noexcept override {
        return {
            static_cast<const float*>(observationBuffer_.contents),
            observationElementCount_
        };
    }

    [[nodiscard]] std::span<const float> rewards() const noexcept override {
        return {
            static_cast<const float*>(rewardBuffer_.contents),
            environmentCount_
        };
    }

    [[nodiscard]] std::span<const std::uint8_t>
    terminated() const noexcept override {
        return {
            static_cast<const std::uint8_t*>(terminatedBuffer_.contents),
            environmentCount_
        };
    }

    [[nodiscard]] std::span<const float>
    bodyPositions() const noexcept override {
        if (!descriptor_.captureBodyPoses) {
            return {};
        }
        return {
            static_cast<const float*>(bodyPositionBuffer_.contents),
            bodyPoseFloatCount_
        };
    }

    [[nodiscard]] std::span<const float>
    bodyRotations() const noexcept override {
        if (!descriptor_.captureBodyPoses) {
            return {};
        }
        return {
            static_cast<const float*>(bodyRotationBuffer_.contents),
            bodyPoseFloatCount_
        };
    }

    [[nodiscard]] RuntimeStats stats() const noexcept override {
        return stats_;
    }

    [[nodiscard]] std::string deviceName() const override {
        return nsString(device_.name);
    }

    void reset(const std::uint64_t seed) override {
        @autoreleasepool {
            descriptor_.seed = seed;
            updateUniforms(seed, true);
            encodeAndWait(resetPipeline_, @"MetalRobo reset");
            stats_.controlSteps = 0u;
            stats_.physicsSteps = 0u;
        }
    }

    void step(const std::span<const float> normalizedActions) override {
        const std::size_t requiredActions = checkedSizeT(
            "action element count",
            checkedMultiply(
                "actions",
                {
                    environmentCount_,
                    model_.gpu.actionCount,
                }
            )
        );
        if (normalizedActions.size() != requiredActions) {
            std::ostringstream message;
            message
                << "MetalRobo expected "
                << requiredActions
                << " normalized actions (["
                << environmentCount_
                << ", "
                << model_.gpu.actionCount
                << "]), received "
                << normalizedActions.size();
            throw std::invalid_argument(message.str());
        }

        @autoreleasepool {
            if (!normalizedActions.empty()) {
                std::memcpy(
                    actionBuffer_.contents,
                    normalizedActions.data(),
                    normalizedActions.size_bytes()
                );
            }
            // Mix the control-step index into the dispatch seed so automatic
            // resets sample a new deterministic target each episode.
            const std::uint64_t dispatchSeed =
                descriptor_.seed +
                (stats_.controlSteps + 1u) * 0x9e3779b97f4a7c15ull;
            updateUniforms(dispatchSeed, false);
            encodeAndWait(stepPipeline_, @"MetalRobo step");
            ++stats_.controlSteps;
            stats_.physicsSteps += std::max(model_.gpu.substeps, 1u);
        }
    }

private:
    static std::size_t checkedSizeT(
        const std::string& label,
        const std::uint64_t value
    ) {
        if (value > std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error(label + " does not fit size_t");
        }
        return static_cast<std::size_t>(value);
    }

    void initialize() {
        std::string reason;
        if (!model_.valid(&reason)) {
            throw std::invalid_argument(
                "cannot create Metal runtime for invalid model: " + reason
            );
        }
        if (environmentCount_ == 0u) {
            throw std::invalid_argument(
                "Metal runtime environmentCount must be greater than zero"
            );
        }

        device_ = MTLCreateSystemDefaultDevice();
        if (device_ == nil) {
            throw std::runtime_error(
                "MetalRobo could not find a Metal-capable GPU"
            );
        }
        if (!device_.hasUnifiedMemory) {
            throw std::runtime_error(
                "MetalRobo v0.1 requires an Apple-silicon GPU with unified "
                "memory; selected device is '" + nsString(device_.name) + "'"
            );
        }
        queue_ = [device_ newCommandQueue];
        if (queue_ == nil) {
            throw std::runtime_error(
                "MetalRobo failed to create a command queue for '" +
                nsString(device_.name) + "'"
            );
        }
        queue_.label = @"MetalRobo command queue";

        loadPipelines();
        allocateBuffers();
        bindBufferTable();
    }

    void loadPipelines() {
        std::string path = descriptor_.metallibPath;
        if (path.empty()) {
            path = METALROBO_DEFAULT_METALLIB;
        }
        if (path.empty()) {
            throw std::runtime_error(
                "no MetalRobo metallib path was supplied and this build has "
                "no METALROBO_DEFAULT_METALLIB"
            );
        }

        NSString* pathString =
            [NSString stringWithUTF8String:path.c_str()];
        if (pathString == nil) {
            throw std::runtime_error(
                "MetalRobo metallib path is not valid UTF-8"
            );
        }
        NSURL* url = [NSURL fileURLWithPath:pathString];
        NSError* error = nil;
        library_ = [device_ newLibraryWithURL:url error:&error];
        if (library_ == nil) {
            throw std::runtime_error(
                "failed to load MetalRobo metallib '" + path +
                "': " + describeError(error)
            );
        }
        library_.label = @"MetalRobo physics library";

        resetPipeline_ = makePipeline(@"metalrobo_reset");
        stepPipeline_ = makePipeline(@"metalrobo_step");

        const NSUInteger maximumThreadgroupMemory =
            device_.maxThreadgroupMemoryLength;
        const NSUInteger resetMemory =
            resetPipeline_.staticThreadgroupMemoryLength;
        const NSUInteger stepMemory =
            stepPipeline_.staticThreadgroupMemoryLength;
        if (resetMemory > maximumThreadgroupMemory ||
            stepMemory > maximumThreadgroupMemory) {
            std::ostringstream message;
            message
                << "MetalRobo physics kernels require "
                << std::max(resetMemory, stepMemory)
                << " bytes of threadgroup memory, but device '"
                << nsString(device_.name)
                << "' exposes "
                << maximumThreadgroupMemory
                << " bytes";
            throw std::runtime_error(message.str());
        }
    }

    id<MTLComputePipelineState> makePipeline(NSString* functionName) {
        id<MTLFunction> function =
            [library_ newFunctionWithName:functionName];
        if (function == nil) {
            throw std::runtime_error(
                "MetalRobo metallib does not contain kernel '" +
                nsString(functionName) + "'"
            );
        }

        NSError* error = nil;
        id<MTLComputePipelineState> pipeline =
            [device_ newComputePipelineStateWithFunction:function
                                                   error:&error];
        if (pipeline == nil) {
            throw std::runtime_error(
                "failed to build Metal pipeline '" +
                nsString(functionName) + "': " + describeError(error)
            );
        }
        if (pipeline.maxTotalThreadsPerThreadgroup <
            kThreadsPerEnvironment) {
            std::ostringstream message;
            message
                << "Metal pipeline '"
                << nsString(functionName)
                << "' permits only "
                << pipeline.maxTotalThreadsPerThreadgroup
                << " threads per threadgroup; MetalRobo requires "
                << kThreadsPerEnvironment;
            throw std::runtime_error(message.str());
        }
        return pipeline;
    }

    void allocateBuffers() {
        const std::uint64_t environmentCount = environmentCount_;
        const std::uint64_t dofCount = model_.gpu.dofCount;
        const std::uint64_t actionCount = model_.gpu.actionCount;
        const std::uint64_t observationCount =
            model_.gpu.observationCount;
        const std::uint64_t linkCount = model_.gpu.linkCount;

        const std::uint64_t stateBytes = checkedMultiply(
            "articulation state",
            {environmentCount, dofCount, sizeof(float)}
        );
        const std::uint64_t actionBytes = checkedMultiply(
            "actions",
            {environmentCount, actionCount, sizeof(float)}
        );
        const std::uint64_t targetBytes = checkedMultiply(
            "targets",
            {environmentCount, sizeof(mr_float4)}
        );
        const std::uint64_t stepBytes = checkedMultiply(
            "episode steps",
            {environmentCount, sizeof(std::uint32_t)}
        );
        const std::uint64_t observationBytes = checkedMultiply(
            "observations",
            {environmentCount, observationCount, sizeof(float)}
        );
        const std::uint64_t rewardBytes = checkedMultiply(
            "rewards",
            {environmentCount, sizeof(float)}
        );
        const std::uint64_t terminatedBytes = environmentCount;
        const std::uint64_t fullBodyPoseBytes = checkedMultiply(
            "body poses",
            {environmentCount, linkCount, sizeof(mr_float4)}
        );
        const std::uint64_t bodyPoseBytes =
            descriptor_.captureBodyPoses
            ? fullBodyPoseBytes
            : sizeof(mr_float4);

        observationElementCount_ = checkedSizeT(
            "observation element count",
            checkedMultiply(
                "observation element count",
                {environmentCount, observationCount}
            )
        );
        bodyPoseFloatCount_ = checkedSizeT(
            "body pose float count",
            checkedMultiply(
                "body pose float count",
                {environmentCount, linkCount, 4u}
            )
        );

        const std::array<BufferRequirement, kBufferCount> requirements{{
            {"model", sizeof(MRModelGPU)},
            {
                "joints",
                checkedMultiply(
                    "joints",
                    {model_.joints.size(), sizeof(MRJointGPU)}
                )
            },
            {
                "links",
                checkedMultiply(
                    "links",
                    {model_.links.size(), sizeof(MRLinkGPU)}
                )
            },
            {
                "colliders",
                checkedMultiply(
                    "colliders",
                    {model_.colliders.size(), sizeof(MRColliderGPU)}
                )
            },
            {
                "home position",
                checkedMultiply(
                    "home position",
                    {model_.homePosition.size(), sizeof(float)}
                )
            },
            {"step uniforms", sizeof(MRStepUniformsGPU)},
            {"actions", actionBytes},
            {"joint positions", stateBytes},
            {"joint velocities", stateBytes},
            {"joint accelerations", stateBytes},
            {"joint torques", stateBytes},
            {"targets", targetBytes},
            {"episode steps", stepBytes},
            {"observations", observationBytes},
            {"rewards", rewardBytes},
            {"termination flags", terminatedBytes},
            {"body positions", bodyPoseBytes},
            {"body rotations", bodyPoseBytes},
        }};

        std::uint64_t totalBytes = 0u;
        for (const BufferRequirement& requirement : requirements) {
            const std::uint64_t allocationBytes =
                std::max<std::uint64_t>(requirement.bytes, 1u);
            if (allocationBytes >
                static_cast<std::uint64_t>(device_.maxBufferLength)) {
                std::ostringstream message;
                message
                    << "MetalRobo "
                    << requirement.label
                    << " buffer requires "
                    << allocationBytes
                    << " bytes, exceeding device maxBufferLength "
                    << device_.maxBufferLength;
                throw std::length_error(message.str());
            }
            if (totalBytes >
                std::numeric_limits<std::uint64_t>::max() -
                    allocationBytes) {
                throw std::overflow_error(
                    "MetalRobo total buffer working-set size overflow"
                );
            }
            totalBytes += allocationBytes;
        }
        const std::uint64_t recommendedWorkingSet =
            device_.recommendedMaxWorkingSetSize;
        if (recommendedWorkingSet != 0u &&
            totalBytes > recommendedWorkingSet) {
            std::ostringstream message;
            message
                << "MetalRobo requested "
                << totalBytes
                << " bytes of shared buffers, exceeding device '"
                << nsString(device_.name)
                << "' recommendedMaxWorkingSetSize of "
                << recommendedWorkingSet
                << " bytes; reduce environmentCount";
            throw std::length_error(message.str());
        }

        modelBuffer_ = makeInitializedBuffer(
            "model",
            &model_.gpu,
            sizeof(MRModelGPU)
        );
        jointBuffer_ = makeInitializedBuffer(
            "joints",
            model_.joints.data(),
            checkedSizeT("joints", requirements[1].bytes)
        );
        linkBuffer_ = makeInitializedBuffer(
            "links",
            model_.links.data(),
            checkedSizeT("links", requirements[2].bytes)
        );
        colliderBuffer_ = makeInitializedBuffer(
            "colliders",
            model_.colliders.data(),
            checkedSizeT("colliders", requirements[3].bytes)
        );
        homePositionBuffer_ = makeInitializedBuffer(
            "home positions",
            model_.homePosition.data(),
            checkedSizeT("home positions", requirements[4].bytes)
        );
        uniformBuffer_ = makeZeroedBuffer(
            "step uniforms",
            sizeof(MRStepUniformsGPU)
        );
        actionBuffer_ = makeZeroedBuffer(
            "actions",
            checkedSizeT("actions", actionBytes)
        );
        positionBuffer_ = makeZeroedBuffer(
            "joint positions",
            checkedSizeT("joint positions", stateBytes)
        );
        velocityBuffer_ = makeZeroedBuffer(
            "joint velocities",
            checkedSizeT("joint velocities", stateBytes)
        );
        accelerationBuffer_ = makeZeroedBuffer(
            "joint accelerations",
            checkedSizeT("joint accelerations", stateBytes)
        );
        torqueBuffer_ = makeZeroedBuffer(
            "joint torques",
            checkedSizeT("joint torques", stateBytes)
        );
        targetBuffer_ = makeZeroedBuffer(
            "targets",
            checkedSizeT("targets", targetBytes)
        );
        episodeStepBuffer_ = makeZeroedBuffer(
            "episode steps",
            checkedSizeT("episode steps", stepBytes)
        );
        observationBuffer_ = makeZeroedBuffer(
            "observations",
            checkedSizeT("observations", observationBytes)
        );
        rewardBuffer_ = makeZeroedBuffer(
            "rewards",
            checkedSizeT("rewards", rewardBytes)
        );
        terminatedBuffer_ = makeZeroedBuffer(
            "termination flags",
            checkedSizeT("termination flags", terminatedBytes)
        );
        bodyPositionBuffer_ = makeZeroedBuffer(
            "body positions",
            checkedSizeT("body positions", bodyPoseBytes)
        );
        bodyRotationBuffer_ = makeZeroedBuffer(
            "body rotations",
            checkedSizeT("body rotations", bodyPoseBytes)
        );
    }

    id<MTLBuffer> makeZeroedBuffer(
        const char* label,
        const std::size_t requestedBytes
    ) {
        const std::size_t bytes = std::max<std::size_t>(
            requestedBytes,
            1u
        );
        id<MTLBuffer> buffer = [
            device_
            newBufferWithLength:bytes
            options:MTLResourceStorageModeShared
        ];
        if (buffer == nil) {
            std::ostringstream message;
            message
                << "MetalRobo failed to allocate shared "
                << label
                << " buffer ("
                << bytes
                << " bytes) on '"
                << nsString(device_.name)
                << "'";
            throw std::runtime_error(message.str());
        }
        buffer.label = [NSString stringWithUTF8String:label];
        std::memset(buffer.contents, 0, bytes);
        return buffer;
    }

    id<MTLBuffer> makeInitializedBuffer(
        const char* label,
        const void* source,
        const std::size_t requestedBytes
    ) {
        id<MTLBuffer> buffer = makeZeroedBuffer(label, requestedBytes);
        if (requestedBytes != 0u) {
            if (source == nullptr) {
                throw std::invalid_argument(
                    std::string{"null initialization data for "} + label
                );
            }
            std::memcpy(buffer.contents, source, requestedBytes);
        }
        return buffer;
    }

    void bindBufferTable() {
        buffers_[0] = modelBuffer_;
        buffers_[1] = jointBuffer_;
        buffers_[2] = linkBuffer_;
        buffers_[3] = colliderBuffer_;
        buffers_[4] = homePositionBuffer_;
        buffers_[5] = uniformBuffer_;
        buffers_[6] = actionBuffer_;
        buffers_[7] = positionBuffer_;
        buffers_[8] = velocityBuffer_;
        buffers_[9] = accelerationBuffer_;
        buffers_[10] = torqueBuffer_;
        buffers_[11] = targetBuffer_;
        buffers_[12] = episodeStepBuffer_;
        buffers_[13] = observationBuffer_;
        buffers_[14] = rewardBuffer_;
        buffers_[15] = terminatedBuffer_;
        buffers_[16] = bodyPositionBuffer_;
        buffers_[17] = bodyRotationBuffer_;
    }

    void updateUniforms(
        const std::uint64_t seed,
        const bool resetAll
    ) {
        auto* uniforms =
            static_cast<MRStepUniformsGPU*>(uniformBuffer_.contents);
        uniforms->environmentCount = environmentCount_;
        uniforms->seedLo = static_cast<std::uint32_t>(seed);
        uniforms->seedHi = static_cast<std::uint32_t>(seed >> 32u);
        uniforms->resetAll = resetAll ? 1u : 0u;
        uniforms->autoReset = descriptor_.autoReset ? 1u : 0u;
        uniforms->captureBodyPoses =
            descriptor_.captureBodyPoses ? 1u : 0u;
        uniforms->reserved0 = 0u;
        uniforms->reserved1 = 0u;
    }

    void encodeAndWait(
        id<MTLComputePipelineState> pipeline,
        NSString* label
    ) {
        id<MTLCommandBuffer> commandBuffer = [queue_ commandBuffer];
        if (commandBuffer == nil) {
            throw std::runtime_error(
                "MetalRobo failed to create command buffer for " +
                nsString(label)
            );
        }
        commandBuffer.label = label;

        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            throw std::runtime_error(
                "MetalRobo failed to create compute encoder for " +
                nsString(label)
            );
        }
        encoder.label = label;
        [encoder setComputePipelineState:pipeline];
        for (NSUInteger index = 0u; index < kBufferCount; ++index) {
            [encoder setBuffer:buffers_[index] offset:0 atIndex:index];
        }
        [encoder
            dispatchThreadgroups:MTLSizeMake(environmentCount_, 1u, 1u)
            threadsPerThreadgroup:
                MTLSizeMake(kThreadsPerEnvironment, 1u, 1u)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.status == MTLCommandBufferStatusError) {
            throw std::runtime_error(
                "MetalRobo GPU command '" +
                nsString(label) +
                "' failed: " +
                describeError(commandBuffer.error)
            );
        }
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            std::ostringstream message;
            message
                << "MetalRobo GPU command '"
                << nsString(label)
                << "' ended with unexpected status "
                << static_cast<long>(commandBuffer.status);
            throw std::runtime_error(message.str());
        }

        const CFTimeInterval start = commandBuffer.GPUStartTime;
        const CFTimeInterval end = commandBuffer.GPUEndTime;
        const double milliseconds =
            end >= start && start > 0.0
            ? (end - start) * 1000.0
            : 0.0;
        stats_.lastGpuMilliseconds = milliseconds;
        stats_.totalGpuMilliseconds += milliseconds;
    }

    Model model_;
    RuntimeDescriptor descriptor_;
    std::uint32_t environmentCount_ = 0u;
    std::size_t observationElementCount_ = 0u;
    std::size_t bodyPoseFloatCount_ = 0u;
    RuntimeStats stats_{};

    __strong id<MTLDevice> device_;
    __strong id<MTLCommandQueue> queue_;
    __strong id<MTLLibrary> library_;
    __strong id<MTLComputePipelineState> resetPipeline_;
    __strong id<MTLComputePipelineState> stepPipeline_;

    __strong id<MTLBuffer> modelBuffer_;
    __strong id<MTLBuffer> jointBuffer_;
    __strong id<MTLBuffer> linkBuffer_;
    __strong id<MTLBuffer> colliderBuffer_;
    __strong id<MTLBuffer> homePositionBuffer_;
    __strong id<MTLBuffer> uniformBuffer_;
    __strong id<MTLBuffer> actionBuffer_;
    __strong id<MTLBuffer> positionBuffer_;
    __strong id<MTLBuffer> velocityBuffer_;
    __strong id<MTLBuffer> accelerationBuffer_;
    __strong id<MTLBuffer> torqueBuffer_;
    __strong id<MTLBuffer> targetBuffer_;
    __strong id<MTLBuffer> episodeStepBuffer_;
    __strong id<MTLBuffer> observationBuffer_;
    __strong id<MTLBuffer> rewardBuffer_;
    __strong id<MTLBuffer> terminatedBuffer_;
    __strong id<MTLBuffer> bodyPositionBuffer_;
    __strong id<MTLBuffer> bodyRotationBuffer_;
    __strong id<MTLBuffer> buffers_[kBufferCount];
};

} // namespace

std::unique_ptr<Runtime> makeMetalRuntime(
    Model model,
    const RuntimeDescriptor& descriptor
) {
    return std::make_unique<MetalRuntime>(
        std::move(model),
        descriptor
    );
}

} // namespace metalrobo
