#include "metalrobo/MetalHyperPolicy.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

std::string nsString(NSString* value) {
    return value == nil ? std::string{} : std::string{value.UTF8String};
}

std::string defaultMetallibPath() {
#ifdef METALROBO_DEFAULT_METALLIB
    return METALROBO_DEFAULT_METALLIB;
#else
    return {};
#endif
}

MetalHyperPolicyDiagnostics reject(
    MetalHyperPolicyStatus status,
    std::string message,
    std::string device = {}
) {
    return {
        .status = status,
        .message = std::move(message),
        .deviceName = std::move(device),
    };
}

id<MTLComputePipelineState> makePipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* name,
    NSError** error
) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) {
        if (error != nullptr) {
            *error = [NSError
                errorWithDomain:@"MetalRobo.HyperPolicy"
                           code:1
                       userInfo:@{
                           NSLocalizedDescriptionKey:
                               [NSString stringWithFormat:
                                   @"metallib does not contain %@", name]
                       }];
        }
        return nil;
    }
    return [device newComputePipelineStateWithFunction:function error:error];
}

id<MTLBuffer> immutableBuffer(
    id<MTLDevice> device,
    const void* bytes,
    const std::size_t byteCount,
    NSString* label
) {
    const std::size_t allocation = std::max<std::size_t>(byteCount, 16u);
    id<MTLBuffer> buffer = [device
        newBufferWithLength:allocation
                    options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        return nil;
    }
    buffer.label = label;
    if (byteCount != 0u) {
        std::memcpy(buffer.contents, bytes, byteCount);
    } else {
        std::memset(buffer.contents, 0, allocation);
    }
    return buffer;
}

void dispatchThreads(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    const std::size_t count
) {
    if (count == 0u) {
        return;
    }
    const NSUInteger width = std::min<NSUInteger>(
        std::max<NSUInteger>(pipeline.threadExecutionWidth, 1u),
        pipeline.maxTotalThreadsPerThreadgroup
    );
    [encoder
        dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(count), 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
}

void dispatchSimdMatrix(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    const std::size_t columns,
    const std::size_t rows
) {
    if (columns == 0u || rows == 0u) {
        return;
    }
    const NSUInteger width = std::max<NSUInteger>(
        pipeline.threadExecutionWidth, 1u
    );
    [encoder
        dispatchThreadgroups:MTLSizeMake(
            static_cast<NSUInteger>(columns),
            static_cast<NSUInteger>(rows),
            1u
        )
        threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
}

bool sameDevice(id<MTLDevice> device, id<MTLBuffer> buffer) {
    return buffer != nil && buffer.device.registryID == device.registryID;
}

} // namespace

struct MetalHyperPolicyRuntime::State {
    std::mutex mutex;
    id<MTLDevice> device = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> signaturePipeline = nil;
    id<MTLComputePipelineState> phasePipeline = nil;
    id<MTLComputePipelineState> adapterDownPipeline = nil;
    id<MTLComputePipelineState> adapterUpPipeline = nil;
    id<MTLComputePipelineState> finalizePipeline = nil;

    id<MTLBuffer> programHeader = nil;
    id<MTLBuffer> programArena = nil;
    id<MTLBuffer> actionLower = nil;
    id<MTLBuffer> actionUpper = nil;
    id<MTLBuffer> maximumActionRate = nil;

    id<MTLBuffer> phaseStates = nil;
    id<MTLBuffer> signatures = nil;
    id<MTLBuffer> contactMasks = nil;
    id<MTLBuffer> rankWorkspace = nil;
    id<MTLBuffer> scratchA = nil;
    id<MTLBuffer> scratchB = nil;
    id<MTLBuffer> previousActions = nil;
    id<MTLBuffer> phaseTrajectory = nil;
    id<MTLBuffer> actionTrajectory = nil;
    id<MTLBuffer> latentTrajectory = nil;
    id<MTLBuffer> logProbabilityTrajectory = nil;

    MRHyperPolicyProgramHeaderGPU header{};
    std::vector<MRHyperPolicyLayerGPU> layers;
    std::uint64_t taskFingerprint = 0u;
    std::uint32_t forwardSearchFrames = 12u;
    std::uint32_t maximumActivationWidth = 0u;
    std::uint32_t environmentCapacity = 0u;
    std::uint32_t controlStepCapacity = 0u;
    std::string deviceName;

    [[nodiscard]] bool valid() const noexcept {
        return device != nil && library != nil &&
            signaturePipeline != nil && phasePipeline != nil &&
            adapterDownPipeline != nil && adapterUpPipeline != nil &&
            finalizePipeline != nil && programHeader != nil &&
            programArena != nil && actionLower != nil &&
            actionUpper != nil && maximumActionRate != nil &&
            header.policyFingerprint != 0u &&
            taskFingerprint == header.taskFingerprint &&
            !layers.empty();
    }

    bool ensureCapacity(
        id<MTLCommandBuffer> commandBuffer,
        const std::uint32_t environmentCount,
        const std::uint32_t controlStepCount
    ) {
        if (environmentCount <= environmentCapacity &&
            controlStepCount <= controlStepCapacity) {
            return true;
        }
        const std::uint32_t nextEnvironmentCapacity = std::max(
            environmentCount, environmentCapacity
        );
        const std::uint32_t nextControlStepCapacity = std::max(
            controlStepCount, controlStepCapacity
        );
        std::uint64_t signatureElements =
            static_cast<std::uint64_t>(nextEnvironmentCapacity) * header.counts1.z;
        std::uint64_t workspaceElements =
            static_cast<std::uint64_t>(nextEnvironmentCapacity) *
            2u * header.counts1.w;
        std::uint64_t scratchElements =
            static_cast<std::uint64_t>(nextEnvironmentCapacity) *
            maximumActivationWidth;
        std::uint64_t actionElements =
            static_cast<std::uint64_t>(nextEnvironmentCapacity) * header.counts0.z;
        const std::uint64_t phaseElements =
            static_cast<std::uint64_t>(nextEnvironmentCapacity) *
            nextControlStepCapacity;
        const std::uint64_t actionTrajectoryElements = phaseElements *
            header.counts0.z;
        constexpr std::uint64_t maximum =
            std::numeric_limits<NSUInteger>::max() / sizeof(float);
        if (signatureElements > maximum || workspaceElements > maximum ||
            scratchElements > maximum || actionElements > maximum ||
            phaseElements > maximum || actionTrajectoryElements > maximum) {
            return false;
        }
        auto makePrivate = [&](std::size_t bytes, NSString* label) {
            id<MTLBuffer> buffer = [device
                newBufferWithLength:std::max<std::size_t>(bytes, 16u)
                            options:MTLResourceStorageModePrivate];
            if (buffer != nil) {
                buffer.label = label;
            }
            return buffer;
        };
        phaseStates = makePrivate(
            nextEnvironmentCapacity * sizeof(MRHyperPolicyPhaseStateGPU),
            @"HyperPolicy phase states"
        );
        signatures = makePrivate(
            signatureElements * sizeof(float),
            @"HyperPolicy live phase signatures"
        );
        contactMasks = makePrivate(
            nextEnvironmentCapacity * sizeof(std::uint32_t),
            @"HyperPolicy measured contact masks"
        );
        rankWorkspace = makePrivate(
            workspaceElements * sizeof(float),
            @"HyperPolicy low-rank workspace"
        );
        scratchA = makePrivate(
            scratchElements * sizeof(float),
            @"HyperPolicy activation scratch A"
        );
        scratchB = makePrivate(
            scratchElements * sizeof(float),
            @"HyperPolicy activation scratch B"
        );
        previousActions = makePrivate(
            actionElements * sizeof(float),
            @"HyperPolicy previous task actions"
        );
        phaseTrajectory = [device
            newBufferWithLength:std::max<std::size_t>(
                phaseElements * sizeof(float), 16u
            )
            options:MTLResourceStorageModeShared];
        actionTrajectory = [device
            newBufferWithLength:std::max<std::size_t>(
                actionTrajectoryElements * sizeof(float), 16u
            )
            options:MTLResourceStorageModeShared];
        latentTrajectory = [device
            newBufferWithLength:std::max<std::size_t>(
                actionTrajectoryElements * sizeof(float), 16u
            )
            options:MTLResourceStorageModeShared];
        logProbabilityTrajectory = [device
            newBufferWithLength:std::max<std::size_t>(
                phaseElements * sizeof(float), 16u
            )
            options:MTLResourceStorageModeShared];
        phaseTrajectory.label = @"HyperPolicy rollout phases";
        actionTrajectory.label = @"HyperPolicy executed actor-coordinate teachers";
        latentTrajectory.label = @"HyperPolicy sampled actor latents";
        logProbabilityTrajectory.label = @"HyperPolicy action log probabilities";
        if (phaseStates == nil || signatures == nil ||
            contactMasks == nil || rankWorkspace == nil ||
            scratchA == nil || scratchB == nil ||
            previousActions == nil || phaseTrajectory == nil ||
            actionTrajectory == nil || latentTrajectory == nil ||
            logProbabilityTrajectory == nil) {
            return false;
        }
        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
        if (blit == nil) {
            return false;
        }
        for (id<MTLBuffer> buffer in @[
                phaseStates, signatures, contactMasks, rankWorkspace,
                scratchA, scratchB, previousActions,
                phaseTrajectory, actionTrajectory, latentTrajectory,
                logProbabilityTrajectory
             ]) {
            [blit fillBuffer:buffer
                       range:NSMakeRange(0u, buffer.length)
                       value:0u];
        }
        [blit endEncoding];
        environmentCapacity = nextEnvironmentCapacity;
        controlStepCapacity = nextControlStepCapacity;
        return true;
    }

    bool encode(const MetalWorldDeviceActionPass& pass) {
        const std::lock_guard lock(mutex);
        if (!valid() || pass.commandBuffer == nullptr ||
            pass.q == nullptr || pass.v == nullptr ||
            pass.resetMasks == nullptr || pass.taskActions == nullptr ||
            pass.taskContactCompact == nullptr ||
            pass.taskProgramHeader == nullptr ||
            pass.taskProgramArena == nullptr ||
            pass.taskStates == nullptr ||
            pass.actorObservations == nullptr ||
            pass.environmentCount == 0u ||
            pass.controlStepCount == 0u ||
            pass.controlStep >= pass.controlStepCount ||
            pass.actionCount != header.counts0.z ||
            pass.actorObservationSize != header.counts0.y ||
            pass.taskFingerprint != taskFingerprint ||
            pass.policyRevision != header.revision ||
            pass.contactMetricCount == 0u) {
            return false;
        }
        id<MTLCommandBuffer> commandBuffer =
            (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
        id<MTLBuffer> q = (__bridge id<MTLBuffer>)pass.q;
        id<MTLBuffer> v = (__bridge id<MTLBuffer>)pass.v;
        id<MTLBuffer> resetMasks =
            (__bridge id<MTLBuffer>)pass.resetMasks;
        id<MTLBuffer> actions =
            (__bridge id<MTLBuffer>)pass.taskActions;
        id<MTLBuffer> compactContact =
            (__bridge id<MTLBuffer>)pass.taskContactCompact;
        id<MTLBuffer> taskHeader =
            (__bridge id<MTLBuffer>)pass.taskProgramHeader;
        id<MTLBuffer> taskArena =
            (__bridge id<MTLBuffer>)pass.taskProgramArena;
        id<MTLBuffer> taskStates =
            (__bridge id<MTLBuffer>)pass.taskStates;
        id<MTLBuffer> actorObservations =
            (__bridge id<MTLBuffer>)pass.actorObservations;
        const std::array<id<MTLBuffer>, 9u> borrowed{
            q, v, resetMasks, actions, compactContact,
            taskHeader, taskArena, taskStates, actorObservations,
        };
        if (std::any_of(
                borrowed.begin(), borrowed.end(),
                [&](id<MTLBuffer> buffer) {
                    return !sameDevice(device, buffer);
                }
            ) || !ensureCapacity(
                commandBuffer,
                pass.environmentCount,
                pass.controlStepCount
            )) {
            return false;
        }

        MRHyperPolicySignatureDispatchGPU signatureDispatch{};
        signatureDispatch.counts = {
            pass.environmentCount,
            pass.actionCount,
            header.counts1.z,
            header.counts2.y,
        };
        signatureDispatch.strides = {
            pass.nq,
            pass.nv,
            pass.contactMetricCount,
            1u,
        };
        signatureDispatch.offsets = {
            0u,
            0u,
            0u,
            static_cast<std::uint32_t>(pass.resetMaskOffsetElements),
        };
        signatureDispatch.policyFingerprint = header.policyFingerprint;
        signatureDispatch.taskFingerprint = header.taskFingerprint;

        id<MTLComputeCommandEncoder> signatureEncoder =
            [commandBuffer computeCommandEncoder];
        if (signatureEncoder == nil) {
            return false;
        }
        signatureEncoder.label = @"HyperPolicy physical signature";
        [signatureEncoder setComputePipelineState:signaturePipeline];
        [signatureEncoder setBuffer:programHeader offset:0u atIndex:0u];
        [signatureEncoder setBuffer:programArena offset:0u atIndex:1u];
        [signatureEncoder setBuffer:taskHeader offset:0u atIndex:2u];
        [signatureEncoder setBuffer:taskArena offset:0u atIndex:3u];
        [signatureEncoder setBytes:&signatureDispatch
                            length:sizeof(signatureDispatch)
                           atIndex:4u];
        [signatureEncoder setBuffer:q offset:0u atIndex:5u];
        [signatureEncoder setBuffer:v offset:0u atIndex:6u];
        [signatureEncoder setBuffer:compactContact offset:0u atIndex:7u];
        [signatureEncoder setBuffer:resetMasks offset:0u atIndex:8u];
        [signatureEncoder setBuffer:phaseStates offset:0u atIndex:9u];
        [signatureEncoder setBuffer:signatures offset:0u atIndex:10u];
        [signatureEncoder setBuffer:contactMasks offset:0u atIndex:11u];
        dispatchThreads(
            signatureEncoder, signaturePipeline, pass.environmentCount
        );
        [signatureEncoder endEncoding];

        MRHyperPolicyPhaseDispatchGPU phaseDispatch{};
        phaseDispatch.counts = {
            pass.environmentCount,
            header.counts1.z,
            forwardSearchFrames,
            header.counts2.y,
        };
        phaseDispatch.strides = {
            header.counts1.z,
            1u,
            0u,
            0u,
        };
        phaseDispatch.policyFingerprint = header.policyFingerprint;
        phaseDispatch.taskFingerprint = header.taskFingerprint;
        id<MTLComputeCommandEncoder> phaseEncoder =
            [commandBuffer computeCommandEncoder];
        if (phaseEncoder == nil) {
            return false;
        }
        phaseEncoder.label = @"HyperPolicy event-synchronised phase";
        [phaseEncoder setComputePipelineState:phasePipeline];
        [phaseEncoder setBuffer:programHeader offset:0u atIndex:0u];
        [phaseEncoder setBuffer:programArena offset:0u atIndex:1u];
        [phaseEncoder setBytes:&phaseDispatch
                        length:sizeof(phaseDispatch)
                       atIndex:2u];
        [phaseEncoder setBuffer:signatures offset:0u atIndex:3u];
        [phaseEncoder setBuffer:contactMasks offset:0u atIndex:4u];
        [phaseEncoder setBuffer:phaseStates offset:0u atIndex:5u];
        dispatchThreads(phaseEncoder, phasePipeline, pass.environmentCount);
        [phaseEncoder endEncoding];

        id<MTLBuffer> input = actorObservations;
        NSUInteger inputOffset = pass.actorObservationOffsetElements
            * sizeof(float);
        std::uint32_t inputStride = pass.actorObservationSize;
        id<MTLBuffer> output = scratchA;
        for (std::uint32_t layerIndex = 0u;
             layerIndex < layers.size(); ++layerIndex) {
            const MRHyperPolicyLayerGPU& layer = layers[layerIndex];
            output = (layerIndex & 1u) == 0u ? scratchA : scratchB;
            MRHyperPolicyLayerDispatchGPU layerDispatch{};
            layerDispatch.counts = {
                pass.environmentCount,
                layerIndex,
                layer.counts.x,
                layer.counts.y,
            };
            layerDispatch.strides = {
                inputStride,
                layer.counts.y,
                2u * header.counts1.w,
                0u,
            };
            layerDispatch.offsets = {0u, 0u, 0u, 0u};
            layerDispatch.policyFingerprint = header.policyFingerprint;
            layerDispatch.taskFingerprint = header.taskFingerprint;

            id<MTLComputeCommandEncoder> down =
                [commandBuffer computeCommandEncoder];
            if (down == nil) {
                return false;
            }
            down.label = @"HyperPolicy low-rank down/gate";
            [down setComputePipelineState:adapterDownPipeline];
            [down setBuffer:programHeader offset:0u atIndex:0u];
            [down setBuffer:programArena offset:0u atIndex:1u];
            [down setBytes:&layerDispatch
                    length:sizeof(layerDispatch)
                   atIndex:2u];
            [down setBuffer:phaseStates offset:0u atIndex:3u];
            [down setBuffer:input offset:inputOffset atIndex:4u];
            [down setBuffer:rankWorkspace offset:0u atIndex:5u];
            dispatchSimdMatrix(
                down,
                adapterDownPipeline,
                layer.counts.z,
                pass.environmentCount
            );
            [down endEncoding];

            id<MTLComputeCommandEncoder> up =
                [commandBuffer computeCommandEncoder];
            if (up == nil) {
                return false;
            }
            up.label = @"HyperPolicy fused base and adapter-up";
            [up setComputePipelineState:adapterUpPipeline];
            [up setBuffer:programHeader offset:0u atIndex:0u];
            [up setBuffer:programArena offset:0u atIndex:1u];
            [up setBytes:&layerDispatch
                  length:sizeof(layerDispatch)
                 atIndex:2u];
            [up setBuffer:input offset:inputOffset atIndex:3u];
            [up setBuffer:rankWorkspace offset:0u atIndex:4u];
            [up setBuffer:output offset:0u atIndex:5u];
            dispatchSimdMatrix(
                up,
                adapterUpPipeline,
                layer.counts.y,
                pass.environmentCount
            );
            [up endEncoding];
            input = output;
            inputOffset = 0u;
            inputStride = layer.counts.y;
        }

        MRHyperPolicyActionDispatchGPU actionDispatch{};
        actionDispatch.counts = {
            pass.environmentCount,
            pass.actionCount,
            pass.controlStep,
            header.counts2.z,
        };
        actionDispatch.strides = {
            pass.actionCount,
            pass.environmentCount * pass.actionCount,
            pass.environmentCount,
            pass.actionCount,
        };
        actionDispatch.offsets = {
            0u,
            0u,
            0u,
            0u,
        };
        actionDispatch.bounds = {0u, 0u, 0u, 0u};
        actionDispatch.reset = {
            static_cast<std::uint32_t>(pass.resetMaskOffsetElements),
            1u,
            0u,
            0u,
        };
        actionDispatch.policyFingerprint = header.policyFingerprint;
        actionDispatch.taskFingerprint = header.taskFingerprint;
        actionDispatch.randomSeed = pass.seed;

        id<MTLComputeCommandEncoder> finalize =
            [commandBuffer computeCommandEncoder];
        if (finalize == nil) {
            return false;
        }
        finalize.label = @"HyperPolicy final physical action";
        [finalize setComputePipelineState:finalizePipeline];
        [finalize setBuffer:programHeader offset:0u atIndex:0u];
        [finalize setBuffer:programArena offset:0u atIndex:1u];
        [finalize setBytes:&actionDispatch
                    length:sizeof(actionDispatch)
                   atIndex:2u];
        [finalize setBuffer:phaseStates offset:0u atIndex:3u];
        [finalize setBuffer:input offset:0u atIndex:4u];
        [finalize setBuffer:actions offset:0u atIndex:5u];
        [finalize setBuffer:latentTrajectory offset:0u atIndex:6u];
        [finalize setBuffer:logProbabilityTrajectory offset:0u atIndex:7u];
        [finalize setBuffer:previousActions offset:0u atIndex:8u];
        [finalize setBuffer:actionLower offset:0u atIndex:9u];
        [finalize setBuffer:actionUpper offset:0u atIndex:10u];
        [finalize setBuffer:maximumActionRate offset:0u atIndex:11u];
        [finalize setBuffer:resetMasks offset:0u atIndex:12u];
        [finalize setBuffer:phaseTrajectory offset:0u atIndex:13u];
        [finalize setBuffer:actionTrajectory offset:0u atIndex:14u];
        [finalize setBuffer:taskStates offset:0u atIndex:15u];
        dispatchThreads(finalize, finalizePipeline, pass.environmentCount);
        [finalize endEncoding];

        return true;
    }
};


bool MetalHyperPolicyRuntime::encodeCallback(
    void* context,
    const MetalWorldDeviceActionPass& pass
) noexcept {
    if (context == nullptr) {
        return false;
    }
    @autoreleasepool {
        try {
            return static_cast<State*>(context)->encode(pass);
        } catch (...) {
            return false;
        }
    }
}


MetalHyperPolicyRuntime::~MetalHyperPolicyRuntime() = default;
MetalHyperPolicyRuntime::MetalHyperPolicyRuntime(
    MetalHyperPolicyRuntime&&
) noexcept = default;
MetalHyperPolicyRuntime& MetalHyperPolicyRuntime::operator=(
    MetalHyperPolicyRuntime&&
) noexcept = default;

bool MetalHyperPolicyRuntime::valid() const noexcept {
    return state_ != nullptr && state_->valid();
}

const std::string& MetalHyperPolicyRuntime::deviceName() const noexcept {
    static const std::string empty;
    return valid() ? state_->deviceName : empty;
}

void MetalHyperPolicyRuntime::reset() noexcept {
    if (!valid()) {
        return;
    }
    try {
        const std::lock_guard lock(state_->mutex);
        state_->phaseStates = nil;
        state_->signatures = nil;
        state_->contactMasks = nil;
        state_->rankWorkspace = nil;
        state_->scratchA = nil;
        state_->scratchB = nil;
        state_->previousActions = nil;
        state_->phaseTrajectory = nil;
        state_->actionTrajectory = nil;
        state_->latentTrajectory = nil;
        state_->logProbabilityTrajectory = nil;
        state_->environmentCapacity = 0u;
        state_->controlStepCapacity = 0u;
    } catch (...) {
    }
}

MetalWorldDeviceActionProgram
MetalHyperPolicyRuntime::actionProgram() noexcept {
    if (!valid()) {
        return {};
    }
    return {
        .context = state_.get(),
        .encode = &MetalHyperPolicyRuntime::encodeCallback,
        .policyRevision = state_->header.revision,
        .taskFingerprint = state_->header.taskFingerprint,
        .actionCount = state_->header.counts0.z,
        .stochastic =
            (state_->header.counts2.z & MR_HYPER_POLICY_STOCHASTIC) != 0u,
    };
}

bool MetalHyperPolicyRuntime::copyRolloutTrace(
    const std::uint32_t controlStepCount,
    const std::uint32_t environmentCount,
    std::vector<float>& phases,
    std::vector<float>& teacherActions,
    std::vector<float>& latents,
    std::vector<float>& logProbabilities
) {
    if (!valid() || controlStepCount == 0u || environmentCount == 0u) {
        return false;
    }
    const std::lock_guard lock(state_->mutex);
    if (environmentCount > state_->environmentCapacity ||
        controlStepCount > state_->controlStepCapacity ||
        state_->phaseTrajectory == nil || state_->actionTrajectory == nil ||
        state_->latentTrajectory == nil ||
        state_->logProbabilityTrajectory == nil) {
        return false;
    }
    const std::size_t phaseCount =
        static_cast<std::size_t>(controlStepCount) * environmentCount;
    const std::size_t actionCount = phaseCount * state_->header.counts0.z;
    phases.resize(phaseCount);
    teacherActions.resize(actionCount);
    latents.resize(actionCount);
    logProbabilities.resize(phaseCount);
    std::memcpy(
        phases.data(), state_->phaseTrajectory.contents,
        phaseCount * sizeof(float)
    );
    std::memcpy(
        teacherActions.data(), state_->actionTrajectory.contents,
        actionCount * sizeof(float)
    );
    std::memcpy(
        latents.data(), state_->latentTrajectory.contents,
        actionCount * sizeof(float)
    );
    std::memcpy(
        logProbabilities.data(), state_->logProbabilityTrajectory.contents,
        phaseCount * sizeof(float)
    );
    return std::all_of(phases.begin(), phases.end(), [](const float value) {
        return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
    }) && std::all_of(
        teacherActions.begin(), teacherActions.end(),
        [](const float value) { return std::isfinite(value); }
    ) && std::all_of(
        latents.begin(), latents.end(),
        [](const float value) { return std::isfinite(value); }
    ) && std::all_of(
        logProbabilities.begin(), logProbabilities.end(),
        [](const float value) { return std::isfinite(value); }
    );
}


MetalHyperPolicyDiagnostics createMetalHyperPolicyRuntime(
    const CompiledHyperPolicyProgram& program,
    const CompiledTaskProgram& task,
    const MetalHyperPolicyConfiguration& configuration,
    MetalHyperPolicyRuntime& output
) {
    if (!program.valid()) {
        return reject(
            MetalHyperPolicyStatus::invalidProgram,
            "compiled hyper-policy program is invalid"
        );
    }
    if (!task.valid() || program.taskFingerprint() != task.fingerprint() ||
        program.layout().actorObservationCount !=
            task.layout().actorObservationSize ||
        program.layout().actionCount != task.layout().actionCount ||
        program.layout().signatureCount !=
            2u * program.layout().actionCount + 9u +
                program.layout().contactTrackCount ||
        configuration.forwardSearchFrames == 0u) {
        return reject(
            MetalHyperPolicyStatus::incompatibleTask,
            "hyper-policy runtime does not match the compiled task or signature contract"
        );
    }
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return reject(
                MetalHyperPolicyStatus::metalUnavailable,
                "no Metal device is available"
            );
        }
        std::string path = configuration.metallibPath;
        if (path.empty()) {
            path = defaultMetallibPath();
        }
        if (!path.empty()) {
            const std::filesystem::path requested{path};
            path = requested.filename() == "MetalRoboHyperPolicy.metallib"
                ? requested.string()
                : (requested.parent_path() /
                   "MetalRoboHyperPolicy.metallib").string();
        }
        if (path.empty()) {
            return reject(
                MetalHyperPolicyStatus::metallibUnavailable,
                "no HyperPolicy metallib path is available",
                nsString(device.name)
            );
        }
        NSError* error = nil;
        id<MTLLibrary> library = [device
            newLibraryWithURL:[NSURL fileURLWithPath:
                [NSString stringWithUTF8String:path.c_str()]]
                        error:&error];
        if (library == nil) {
            return reject(
                MetalHyperPolicyStatus::metallibUnavailable,
                error == nil ? "failed to load HyperPolicy metallib"
                    : nsString(error.localizedDescription),
                nsString(device.name)
            );
        }
        auto staged = std::make_shared<MetalHyperPolicyRuntime::State>();
        staged->device = device;
        staged->library = library;
        staged->header = program.header();
        staged->layers.assign(program.layers().begin(), program.layers().end());
        staged->taskFingerprint = task.fingerprint();
        staged->forwardSearchFrames = configuration.forwardSearchFrames;
        staged->deviceName = nsString(device.name);
        for (const MRHyperPolicyLayerGPU& layer : staged->layers) {
            staged->maximumActivationWidth = std::max(
                staged->maximumActivationWidth,
                std::max(layer.counts.x, layer.counts.y)
            );
        }
        staged->signaturePipeline = makePipeline(
            device, library, @"mr_hyper_policy_build_signature", &error
        );
        staged->phasePipeline = makePipeline(
            device, library, @"mr_hyper_policy_update_phase", &error
        );
        staged->adapterDownPipeline = makePipeline(
            device, library, @"mr_hyper_policy_adapter_down", &error
        );
        staged->adapterUpPipeline = makePipeline(
            device, library, @"mr_hyper_policy_adapter_up", &error
        );
        staged->finalizePipeline = makePipeline(
            device, library, @"mr_hyper_policy_sample_and_finalize", &error
        );
        if (staged->signaturePipeline == nil || staged->phasePipeline == nil ||
            staged->adapterDownPipeline == nil ||
            staged->adapterUpPipeline == nil ||
            staged->finalizePipeline == nil) {
            return reject(
                MetalHyperPolicyStatus::pipelineFailure,
                error == nil ? "failed to compile HyperPolicy pipelines"
                    : nsString(error.localizedDescription),
                staged->deviceName
            );
        }
        staged->programHeader = immutableBuffer(
            device,
            &staged->header,
            sizeof(staged->header),
            @"HyperPolicy program header"
        );
        staged->programArena = immutableBuffer(
            device,
            program.arena().data(),
            program.arena().size_bytes(),
            @"HyperPolicy immutable arena"
        );
        staged->actionLower = immutableBuffer(
            device,
            program.actionLower().data(),
            program.actionLower().size_bytes(),
            @"HyperPolicy action lower bounds"
        );
        staged->actionUpper = immutableBuffer(
            device,
            program.actionUpper().data(),
            program.actionUpper().size_bytes(),
            @"HyperPolicy action upper bounds"
        );
        staged->maximumActionRate = immutableBuffer(
            device,
            program.maximumActionRate().data(),
            program.maximumActionRate().size_bytes(),
            @"HyperPolicy action rate bounds"
        );
        if (!staged->valid()) {
            return reject(
                MetalHyperPolicyStatus::bufferFailure,
                "failed to allocate immutable HyperPolicy buffers",
                staged->deviceName
            );
        }
        output.state_ = std::move(staged);
        return {
            .status = MetalHyperPolicyStatus::success,
            .message = {},
            .deviceName = output.deviceName(),
        };
    }
}

const char* metalHyperPolicyStatusName(
    const MetalHyperPolicyStatus status
) noexcept {
    switch (status) {
    case MetalHyperPolicyStatus::success: return "success";
    case MetalHyperPolicyStatus::invalidProgram: return "invalid-program";
    case MetalHyperPolicyStatus::incompatibleTask: return "incompatible-task";
    case MetalHyperPolicyStatus::metallibUnavailable: return "metallib-unavailable";
    case MetalHyperPolicyStatus::metalUnavailable: return "metal-unavailable";
    case MetalHyperPolicyStatus::pipelineFailure: return "pipeline-failure";
    case MetalHyperPolicyStatus::bufferFailure: return "buffer-failure";
    case MetalHyperPolicyStatus::internalFailure: return "internal-failure";
    }
    return "unknown";
}

} // namespace metalrobo
