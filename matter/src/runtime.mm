#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef NUMI_MATTER_DEFAULT_METALLIB
#define NUMI_MATTER_DEFAULT_METALLIB ""
#endif

namespace numi::matter {
namespace {

const char kImageAnchor = 0;

[[nodiscard]] std::string nsString(NSString* value) {
    return value != nil && value.UTF8String != nullptr
        ? std::string(value.UTF8String)
        : std::string{};
}

[[nodiscard]] std::string errorString(NSError* error) {
    return error == nil
        ? std::string{"unknown Metal error"}
        : nsString(error.localizedDescription);
}

[[nodiscard]] bool regularFile(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) && !error;
}

[[nodiscard]] std::filesystem::path defaultMetallib() {
    Dl_info image{};
    if (dladdr(&kImageAnchor, &image) != 0 && image.dli_fname != nullptr) {
        const std::filesystem::path directory =
            std::filesystem::path(image.dli_fname).parent_path();
        const std::array candidates{
            directory / "numi-matter/NumiMatter.metallib",
            directory.parent_path() / "shaders/NumiMatter.metallib",
            directory / "NumiMatter.metallib",
        };
        for (const auto& candidate : candidates) {
            if (regularFile(candidate)) {
                return candidate;
            }
        }
    }
    const std::filesystem::path configured{NUMI_MATTER_DEFAULT_METALLIB};
    return regularFile(configured) ? configured : std::filesystem::path{};
}

[[nodiscard]] NSUInteger checkedBytes(
    const std::size_t count,
    const std::size_t elementSize,
    bool& valid
) {
    if (count != 0u && elementSize >
        std::numeric_limits<std::size_t>::max() / count) {
        valid = false;
        return 0u;
    }
    const std::size_t bytes = count * elementSize;
    if (bytes > std::numeric_limits<NSUInteger>::max()) {
        valid = false;
        return 0u;
    }
    return static_cast<NSUInteger>(std::max<std::size_t>(bytes, 16u));
}

class UploadPlan {
public:
    UploadPlan(id<MTLDevice> device, id<MTLCommandQueue> queue)
        : device_(device), commandBuffer_([queue commandBuffer]),
          staging_([NSMutableArray array]) {
        blit_ = [commandBuffer_ blitCommandEncoder];
    }

    template <typename T>
    [[nodiscard]] id<MTLBuffer> repeated(
        const std::span<const T> values,
        const std::size_t repeat,
        bool& valid,
        std::size_t& residentBytes
    ) {
        const NSUInteger oneBytes = checkedBytes(values.size(), sizeof(T), valid);
        const NSUInteger totalBytes = checkedBytes(values.size() * repeat, sizeof(T), valid);
        if (!valid) {
            return nil;
        }
        id<MTLBuffer> output = [device_
            newBufferWithLength:totalBytes
                        options:MTLResourceStorageModePrivate];
        if (output == nil) {
            valid = false;
            return nil;
        }
        residentBytes += totalBytes;
        if (values.empty() || repeat == 0u) {
            return output;
        }
        id<MTLBuffer> staging = [device_
            newBufferWithBytes:values.data()
                       length:oneBytes
                      options:MTLResourceStorageModeShared];
        if (staging == nil) {
            valid = false;
            return nil;
        }
        [staging_ addObject:staging];
        const NSUInteger logicalBytes = values.size_bytes();
        for (std::size_t index = 0u; index < repeat; ++index) {
            [blit_ copyFromBuffer:staging
                    sourceOffset:0u
                        toBuffer:output
               destinationOffset:static_cast<NSUInteger>(index) * logicalBytes
                            size:logicalBytes];
        }
        return output;
    }

    template <typename T>
    [[nodiscard]] id<MTLBuffer> one(
        const std::span<const T> values,
        bool& valid,
        std::size_t& residentBytes
    ) {
        return repeated(values, 1u, valid, residentBytes);
    }

    [[nodiscard]] bool finish(std::string& error) {
        [blit_ endEncoding];
        [commandBuffer_ commit];
        [commandBuffer_ waitUntilCompleted];
        if (commandBuffer_.status == MTLCommandBufferStatusError) {
            error = errorString(commandBuffer_.error);
            return false;
        }
        return true;
    }

private:
    id<MTLDevice> device_ = nil;
    id<MTLCommandBuffer> commandBuffer_ = nil;
    id<MTLBlitCommandEncoder> blit_ = nil;
    NSMutableArray<id<MTLBuffer>>* staging_ = nil;
};

template <typename T>
[[nodiscard]] id<MTLBuffer> privateScratch(
    id<MTLDevice> device,
    const std::size_t count,
    bool& valid,
    std::size_t& residentBytes
) {
    const NSUInteger bytes = checkedBytes(count, sizeof(T), valid);
    if (!valid) {
        return nil;
    }
    id<MTLBuffer> buffer = [device
        newBufferWithLength:bytes
                    options:MTLResourceStorageModePrivate];
    if (buffer == nil) {
        valid = false;
        return nil;
    }
    residentBytes += bytes;
    return buffer;
}

template <typename T>
[[nodiscard]] id<MTLBuffer> sharedScratch(
    id<MTLDevice> device,
    const std::size_t count,
    bool& valid,
    std::size_t& residentBytes
) {
    const NSUInteger bytes = checkedBytes(count, sizeof(T), valid);
    if (!valid) {
        return nil;
    }
    id<MTLBuffer> buffer = [device
        newBufferWithLength:bytes
                    options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        valid = false;
        return nil;
    }
    std::memset(buffer.contents, 0, bytes);
    residentBytes += bytes;
    return buffer;
}

[[nodiscard]] bool expectedWorldLayout(const CompiledWorld& world) {
    const NMMatterDispatchGPU& dispatch = world.dispatch;
    return
        dispatch.abiVersion == NM_MATTER_ABI_VERSION &&
        dispatch.materialCount == world.materials.size() &&
        dispatch.parameterCount == world.parameters.size() &&
        dispatch.objectCount == world.objects.size() &&
        dispatch.particleCount == world.mpm.particles.size() &&
        dispatch.gridNodeCount == world.mpm.nodes.size() &&
        dispatch.femNodeCount == world.fem.nodes.size() &&
        dispatch.tetrahedronCount == world.fem.tetrahedra.size() &&
        dispatch.rigidProxyCount == world.contact.rigidProxies.size() &&
        dispatch.contactPairCount == world.contact.pairs.size() &&
        world.mpm.nodeRanges.size() == world.mpm.nodes.size() &&
        world.fem.nodeRanges.size() == world.fem.nodes.size() &&
        world.contact.nodeRanges.size() ==
            world.mpm.nodes.size() + world.fem.nodes.size() &&
        world.contact.rigidRanges.size() == world.contact.rigidProxies.size() &&
        world.adaptive.size() == world.objects.size() &&
        world.schedulers.size() == world.objects.size() &&
        world.fingerprint != 0u;
}

} // namespace

struct Runtime::State {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

    NMMatterDispatchGPU dispatch{};
    std::uint64_t worldFingerprint = 0u;
    std::size_t residentBytes = 0u;
    std::uint32_t identificationGeneration = 0u;
    std::uint32_t identificationDistributionCount = 0u;
    bool captureEvents = true;
    bool requiresCurrentBodies = false;
    bool requiresArticulatedWrenches = false;
    bool requiresSceneBodies = false;
    bool hasAdaptive = false;

    id<MTLBuffer> dispatchBuffer = nil;
    id<MTLBuffer> materials = nil;
    id<MTLBuffer> parameterDefaults = nil;
    id<MTLBuffer> instructions = nil;
    id<MTLBuffer> scalarPrograms = nil;
    id<MTLBuffer> objects = nil;
    id<MTLBuffer> mpmStencils = nil;
    id<MTLBuffer> mpmNodeIncidence = nil;
    id<MTLBuffer> mpmNodeRanges = nil;
    id<MTLBuffer> femTetrahedra = nil;
    id<MTLBuffer> femNodeIncidence = nil;
    id<MTLBuffer> femNodeRanges = nil;
    id<MTLBuffer> rigidProxies = nil;
    id<MTLBuffer> contactPairs = nil;
    id<MTLBuffer> contactNodeIncidence = nil;
    id<MTLBuffer> contactNodeRanges = nil;
    id<MTLBuffer> rigidIncidence = nil;
    id<MTLBuffer> rigidRanges = nil;

    id<MTLBuffer> environmentParameters = nil;
    id<MTLBuffer> particleAccepted = nil;
    id<MTLBuffer> particleCandidate = nil;
    id<MTLBuffer> particleCheckpoint = nil;
    id<MTLBuffer> gridNodes = nil;
    id<MTLBuffer> femAccepted = nil;
    id<MTLBuffer> femCandidate = nil;
    id<MTLBuffer> femCheckpoint = nil;
    id<MTLBuffer> rigidStates = nil;
    id<MTLBuffer> contactSamples = nil;
    id<MTLBuffer> microstepReactions = nil;
    id<MTLBuffer> frameReactions = nil;
    id<MTLBuffer> adaptive = nil;
    id<MTLBuffer> schedulers = nil;
    id<MTLBuffer> schedulerPrevious = nil;
    id<MTLBuffer> statuses = nil;
    id<MTLBuffer> events = nil;

    id<MTLBuffer> elementForces = nil;
    id<MTLBuffer> elementOperator = nil;
    id<MTLBuffer> femSolution = nil;
    id<MTLBuffer> femResidual = nil;
    id<MTLBuffer> femPreconditioned = nil;
    id<MTLBuffer> femDirection = nil;
    id<MTLBuffer> femOperatorValue = nil;
    id<MTLBuffer> pcgScalars = nil;

    id<MTLBuffer> identificationDistributions = nil;
    id<MTLBuffer> identificationCandidates = nil;
    id<MTLBuffer> identificationLosses = nil;
    id<MTLBuffer> dummy = nil;

    [[nodiscard]] id<MTLComputePipelineState> pipeline(
        const std::string_view name
    ) const {
        const auto iterator = pipelines.find(std::string(name));
        return iterator == pipelines.end() ? nil : iterator->second;
    }
};

Runtime::Runtime() = default;
Runtime::~Runtime() = default;
Runtime::Runtime(Runtime&&) noexcept = default;
Runtime& Runtime::operator=(Runtime&&) noexcept = default;

RuntimeDiagnostics Runtime::initialize(
    const CompiledWorld& world,
    const RuntimeConfiguration& configuration
) {
    @autoreleasepool {
        RuntimeDiagnostics diagnostics;
        if (!expectedWorldLayout(world)) {
            diagnostics.message = "compiled matter world has an inconsistent fixed-capacity layout";
            return diagnostics;
        }
        if (configuration.environmentCount != 0u &&
            configuration.environmentCount != world.dispatch.environmentCount) {
            diagnostics.message = "runtime environment count must match the cooked matter package";
            return diagnostics;
        }
        auto candidate = std::make_unique<State>();
        candidate->dispatch = world.dispatch;
        candidate->worldFingerprint = world.fingerprint;
        candidate->captureEvents = configuration.captureEvents;
        candidate->identificationDistributionCount =
            static_cast<std::uint32_t>(world.identification.size());
        candidate->hasAdaptive =
            (world.dispatch.flags & NM_MATTER_ADAPTIVE) != 0u;
        candidate->device = MTLCreateSystemDefaultDevice();
        if (candidate->device == nil) {
            diagnostics.message = "no Metal device is available";
            return diagnostics;
        }
        candidate->queue = [candidate->device newCommandQueue];
        if (candidate->queue == nil) {
            diagnostics.message = "failed to create Numi Matter command queue";
            return diagnostics;
        }
        std::filesystem::path metallib = configuration.metallib;
        if (metallib.empty()) {
            metallib = defaultMetallib();
        }
        if (!regularFile(metallib)) {
            diagnostics.message = "NumiMatter.metallib is unavailable";
            return diagnostics;
        }
        NSError* libraryError = nil;
        candidate->library = [candidate->device
            newLibraryWithFile:[NSString stringWithUTF8String:metallib.string().c_str()]
                         error:&libraryError];
        if (candidate->library == nil) {
            diagnostics.message = "failed to load Numi Matter Metal library: " +
                errorString(libraryError);
            return diagnostics;
        }

        const std::array<const char*, 37> kernelNames{
            "nm_prepare_status",
            "nm_prepare_events",
            "nm_prepare_reactions",
            "nm_prepare_scheduler",
            "nm_reset_parameter_overlay",
            "nm_identification_update",
            "nm_identification_sample",
            "nm_mpm_checkpoint",
            "nm_fem_checkpoint",
            "nm_project_rigid_states",
            "nm_mpm_p2g",
            "nm_fem_internal_forces",
            "nm_fem_pcg_initialize",
            "nm_fem_apply_operator_elements",
            "nm_fem_apply_operator_nodes",
            "nm_fem_reduce_pap",
            "nm_fem_pcg_step_xr",
            "nm_fem_precondition",
            "nm_fem_reduce_new_rz",
            "nm_fem_pcg_update_direction",
            "nm_fem_apply_solution",
            "nm_contact_evaluate",
            "nm_contact_apply_nodes",
            "nm_contact_reduce_rigid",
            "nm_accumulate_rigid_reactions",
            "nm_mpm_g2p",
            "nm_fem_integrate",
            "nm_fem_validate",
            "nm_mpm_commit_microstep",
            "nm_fem_commit_microstep",
            "nm_scheduler_observe",
            "nm_complete_microstep",
            "nm_mpm_rollback_frame",
            "nm_fem_rollback_frame",
            "nm_adaptive_measure",
            "nm_adaptive_decide",
            "nm_adaptive_demote_to_rigid",
        };
        const std::array<const char*, 4> finalKernelNames{
            "nm_adaptive_promote_mpm",
            "nm_adaptive_promote_fem",
            "nm_adaptive_finish_promotion",
            "nm_scheduler_finalize",
        };
        for (const char* name : kernelNames) {
            NSString* functionName = [NSString stringWithUTF8String:name];
            id<MTLFunction> function = [candidate->library newFunctionWithName:functionName];
            if (function == nil) {
                diagnostics.message = std::string("missing Metal function ") + name;
                return diagnostics;
            }
            NSError* pipelineError = nil;
            id<MTLComputePipelineState> pipeline = [candidate->device
                newComputePipelineStateWithFunction:function
                                               error:&pipelineError];
            if (pipeline == nil) {
                diagnostics.message = std::string("failed to compile pipeline ") + name +
                    ": " + errorString(pipelineError);
                return diagnostics;
            }
            candidate->pipelines.emplace(name, pipeline);
        }
        for (const char* name : finalKernelNames) {
            id<MTLFunction> function = [candidate->library
                newFunctionWithName:[NSString stringWithUTF8String:name]];
            NSError* pipelineError = nil;
            id<MTLComputePipelineState> pipeline = function == nil
                ? nil
                : [candidate->device newComputePipelineStateWithFunction:function
                                                                   error:&pipelineError];
            if (pipeline == nil) {
                diagnostics.message = std::string("failed to compile pipeline ") + name +
                    ": " + errorString(pipelineError);
                return diagnostics;
            }
            candidate->pipelines.emplace(name, pipeline);
        }
        {
            const char* name = "nm_bridge_rigid_reactions";
            id<MTLFunction> function = [candidate->library
                newFunctionWithName:[NSString stringWithUTF8String:name]];
            NSError* pipelineError = nil;
            id<MTLComputePipelineState> pipeline = function == nil
                ? nil
                : [candidate->device newComputePipelineStateWithFunction:function
                                                                   error:&pipelineError];
            if (pipeline == nil) {
                diagnostics.message = "failed to compile rigid bridge pipeline: " +
                    errorString(pipelineError);
                return diagnostics;
            }
            candidate->pipelines.emplace(name, pipeline);
        }
        for (const char* name : {
                "nm_fem_reduce_pap",
                "nm_fem_reduce_new_rz",
                "nm_scheduler_observe",
                "nm_adaptive_measure",
            }) {
            if (candidate->pipeline(name).threadExecutionWidth != 32u) {
                diagnostics.message = std::string(name) +
                    " requires the Apple SIMD32 execution contract";
                return diagnostics;
            }
        }

        bool valid = true;
        UploadPlan uploads(candidate->device, candidate->queue);
        const std::size_t environments = world.dispatch.environmentCount;
        candidate->dispatchBuffer = uploads.one(
            std::span<const NMMatterDispatchGPU>(&world.dispatch, 1u),
            valid, candidate->residentBytes);
        candidate->materials = uploads.one(
            std::span<const NMMaterialGPU>(world.materials),
            valid, candidate->residentBytes);
        candidate->parameterDefaults = uploads.one(
            std::span<const NMParameterRangeGPU>(world.parameters),
            valid, candidate->residentBytes);
        candidate->instructions = uploads.one(
            std::span<const NMExpressionInstructionGPU>(world.instructions),
            valid, candidate->residentBytes);
        candidate->scalarPrograms = uploads.one(
            std::span<const NMScalarProgramGPU>(world.scalarPrograms),
            valid, candidate->residentBytes);
        candidate->objects = uploads.one(
            std::span<const NMContinuumObjectGPU>(world.objects),
            valid, candidate->residentBytes);
        candidate->mpmStencils = uploads.one(
            std::span<const NMMPMStencilGPU>(world.mpm.stencils),
            valid, candidate->residentBytes);
        candidate->mpmNodeIncidence = uploads.one(
            std::span<const std::uint32_t>(world.mpm.nodeIncidence),
            valid, candidate->residentBytes);
        candidate->mpmNodeRanges = uploads.one(
            std::span<const NMIncidenceRangeGPU>(world.mpm.nodeRanges),
            valid, candidate->residentBytes);
        candidate->femTetrahedra = uploads.one(
            std::span<const NMTetrahedronGPU>(world.fem.tetrahedra),
            valid, candidate->residentBytes);
        candidate->femNodeIncidence = uploads.one(
            std::span<const std::uint32_t>(world.fem.nodeIncidence),
            valid, candidate->residentBytes);
        candidate->femNodeRanges = uploads.one(
            std::span<const NMIncidenceRangeGPU>(world.fem.nodeRanges),
            valid, candidate->residentBytes);
        candidate->rigidProxies = uploads.one(
            std::span<const NMRigidProxyGPU>(world.contact.rigidProxies),
            valid, candidate->residentBytes);
        candidate->contactPairs = uploads.one(
            std::span<const NMContactPairGPU>(world.contact.pairs),
            valid, candidate->residentBytes);
        candidate->contactNodeIncidence = uploads.one(
            std::span<const std::uint32_t>(world.contact.nodeIncidence),
            valid, candidate->residentBytes);
        candidate->contactNodeRanges = uploads.one(
            std::span<const NMIncidenceRangeGPU>(world.contact.nodeRanges),
            valid, candidate->residentBytes);
        candidate->rigidIncidence = uploads.one(
            std::span<const std::uint32_t>(world.contact.rigidIncidence),
            valid, candidate->residentBytes);
        candidate->rigidRanges = uploads.one(
            std::span<const NMIncidenceRangeGPU>(world.contact.rigidRanges),
            valid, candidate->residentBytes);

        std::vector<float> defaultParameters;
        defaultParameters.reserve(world.parameters.size());
        for (const NMParameterRangeGPU parameter : world.parameters) {
            defaultParameters.push_back(parameter.valueAndBounds.x);
        }
        candidate->environmentParameters = uploads.repeated(
            std::span<const float>(defaultParameters),
            environments, valid, candidate->residentBytes);
        candidate->particleAccepted = uploads.repeated(
            std::span<const NMParticleStateGPU>(world.mpm.particles),
            environments, valid, candidate->residentBytes);
        candidate->particleCandidate = uploads.repeated(
            std::span<const NMParticleStateGPU>(world.mpm.particles),
            environments, valid, candidate->residentBytes);
        candidate->particleCheckpoint = uploads.repeated(
            std::span<const NMParticleStateGPU>(world.mpm.particles),
            environments, valid, candidate->residentBytes);
        candidate->gridNodes = uploads.repeated(
            std::span<const NMGridNodeStateGPU>(world.mpm.nodes),
            environments, valid, candidate->residentBytes);
        candidate->femAccepted = uploads.repeated(
            std::span<const NMFEMNodeStateGPU>(world.fem.nodes),
            environments, valid, candidate->residentBytes);
        candidate->femCandidate = uploads.repeated(
            std::span<const NMFEMNodeStateGPU>(world.fem.nodes),
            environments, valid, candidate->residentBytes);
        candidate->femCheckpoint = uploads.repeated(
            std::span<const NMFEMNodeStateGPU>(world.fem.nodes),
            environments, valid, candidate->residentBytes);
        candidate->adaptive = uploads.repeated(
            std::span<const NMAdaptiveStateGPU>(world.adaptive),
            environments, valid, candidate->residentBytes);
        candidate->schedulers = uploads.repeated(
            std::span<const NMSchedulerStateGPU>(world.schedulers),
            environments, valid, candidate->residentBytes);
        candidate->schedulerPrevious = uploads.repeated(
            std::span<const NMSchedulerStateGPU>(world.schedulers),
            environments, valid, candidate->residentBytes);
        candidate->identificationDistributions = uploads.one(
            std::span<const NMIdentificationDistributionGPU>(world.identification),
            valid, candidate->residentBytes);
        std::string uploadError;
        if (!valid || !uploads.finish(uploadError)) {
            diagnostics.message = valid
                ? "failed to upload immutable matter state: " + uploadError
                : "matter buffer size or allocation overflow";
            return diagnostics;
        }

        const auto multiplied = [&](const std::size_t perEnvironment) {
            return environments * perEnvironment;
        };
        candidate->rigidStates = privateScratch<NMRigidStateGPU>(
            candidate->device, multiplied(world.dispatch.rigidProxyCount),
            valid, candidate->residentBytes);
        candidate->contactSamples = privateScratch<NMContactSampleGPU>(
            candidate->device, multiplied(world.dispatch.contactPairCount),
            valid, candidate->residentBytes);
        candidate->microstepReactions = privateScratch<NMRigidReactionGPU>(
            candidate->device, multiplied(world.dispatch.rigidProxyCount),
            valid, candidate->residentBytes);
        candidate->frameReactions = privateScratch<NMRigidReactionGPU>(
            candidate->device, multiplied(world.dispatch.rigidProxyCount),
            valid, candidate->residentBytes);
        candidate->elementForces = privateScratch<NMFEMElementVectorGPU>(
            candidate->device, multiplied(world.dispatch.tetrahedronCount),
            valid, candidate->residentBytes);
        candidate->elementOperator = privateScratch<NMFEMElementVectorGPU>(
            candidate->device, multiplied(world.dispatch.tetrahedronCount),
            valid, candidate->residentBytes);
        candidate->femSolution = privateScratch<simd_float4>(
            candidate->device, multiplied(world.dispatch.femNodeCount),
            valid, candidate->residentBytes);
        candidate->femResidual = privateScratch<simd_float4>(
            candidate->device, multiplied(world.dispatch.femNodeCount),
            valid, candidate->residentBytes);
        candidate->femPreconditioned = privateScratch<simd_float4>(
            candidate->device, multiplied(world.dispatch.femNodeCount),
            valid, candidate->residentBytes);
        candidate->femDirection = privateScratch<simd_float4>(
            candidate->device, multiplied(world.dispatch.femNodeCount),
            valid, candidate->residentBytes);
        candidate->femOperatorValue = privateScratch<simd_float4>(
            candidate->device, multiplied(world.dispatch.femNodeCount),
            valid, candidate->residentBytes);
        candidate->pcgScalars = privateScratch<NMPCGScalarGPU>(
            candidate->device, multiplied(world.dispatch.objectCount),
            valid, candidate->residentBytes);
        candidate->identificationCandidates =
            privateScratch<NMIdentificationCandidateGPU>(
                candidate->device,
                static_cast<std::size_t>(world.dispatch.identificationCandidateCount) *
                    world.identification.size(),
                valid,
                candidate->residentBytes
            );
        candidate->identificationLosses = sharedScratch<float>(
            candidate->device,
            world.dispatch.identificationCandidateCount,
            valid,
            candidate->residentBytes
        );
        candidate->statuses = sharedScratch<NMMatterStatusGPU>(
            candidate->device,
            environments,
            valid,
            candidate->residentBytes
        );
        candidate->events = sharedScratch<NMEventTokenGPU>(
            candidate->device,
            multiplied(world.dispatch.eventStride),
            valid,
            candidate->residentBytes
        );
        candidate->dummy = sharedScratch<std::uint8_t>(
            candidate->device, 256u, valid, candidate->residentBytes);
        if (!valid) {
            diagnostics.message = "failed to allocate persistent matter runtime state";
            return diagnostics;
        }

        std::set<std::pair<bool, std::uint32_t>> dynamicBindings;
        for (const NMRigidProxyGPU proxy : world.contact.rigidProxies) {
            if (proxy.bodyIndex != NM_INVALID_INDEX) {
                candidate->requiresCurrentBodies = true;
            }
            if ((proxy.flags & NM_RIGID_ARTICULATED) != 0u) {
                candidate->requiresArticulatedWrenches = true;
            } else if ((proxy.flags & NM_RIGID_DYNAMIC) != 0u) {
                candidate->requiresSceneBodies = true;
            }
            if ((proxy.flags & (NM_RIGID_ARTICULATED | NM_RIGID_DYNAMIC)) != 0u &&
                !dynamicBindings.insert({
                    (proxy.flags & NM_RIGID_ARTICULATED) != 0u,
                    proxy.bodyIndex,
                }).second) {
                diagnostics.message =
                    "dynamic rigid proxy targets are not unique; deterministic bridge writes would alias";
                return diagnostics;
            }
        }

        diagnostics.encoded = true;
        diagnostics.residentBytes = candidate->residentBytes;
        diagnostics.device = nsString(candidate->device.name);
        diagnostics.message = "Numi Matter runtime initialized";
        state_ = std::move(candidate);
        return diagnostics;
    }
}

RuntimeDiagnostics Runtime::encode(const EncodeRequest& request) {
    @autoreleasepool {
        RuntimeDiagnostics diagnostics;
        if (!state_) {
            diagnostics.message = "Numi Matter runtime is not initialized";
            return diagnostics;
        }
        if (request.commandBuffer == nullptr) {
            diagnostics.message = "encode requires a borrowed Metal command buffer";
            return diagnostics;
        }
        State& state = *state_;
        if (state.requiresCurrentBodies && request.rigid.currentBodies == nullptr) {
            diagnostics.message = "body-backed matter proxies require the current body arena";
            return diagnostics;
        }
        if (state.requiresArticulatedWrenches &&
            request.rigid.articulatedWrenches == nullptr) {
            diagnostics.message = "articulated matter coupling requires the ABA wrench arena";
            return diagnostics;
        }
        if (state.requiresSceneBodies && request.rigid.sceneBodies == nullptr) {
            diagnostics.message = "dynamic matter coupling requires the candidate scene-body arena";
            return diagnostics;
        }
        if (request.rigid.currentBodyCount > request.rigid.currentBodyStride ||
            request.rigid.articulatedBodyCount > request.rigid.articulatedStride ||
            request.rigid.sceneBodyCount > request.rigid.sceneStride) {
            diagnostics.message = "borrowed rigid-world buffer strides are invalid";
            return diagnostics;
        }

        id<MTLCommandBuffer> commandBuffer =
            (__bridge id<MTLCommandBuffer>)request.commandBuffer;
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            diagnostics.message = "failed to create borrowed matter compute encoder";
            return diagnostics;
        }
        [encoder setLabel:@"Numi Matter"];

        const auto buffer = [&](void* value) -> id<MTLBuffer> {
            return value == nullptr
                ? state.dummy
                : (__bridge id<MTLBuffer>)value;
        };
        id<MTLBuffer> currentBodies = buffer(request.rigid.currentBodies);
        id<MTLBuffer> articulatedWrenches = buffer(request.rigid.articulatedWrenches);
        id<MTLBuffer> sceneBodies = buffer(request.rigid.sceneBodies);

        NMBridgeDispatchGPU bridge{};
        bridge.environmentCount = state.dispatch.environmentCount;
        bridge.rigidProxyCount = state.dispatch.rigidProxyCount;
        bridge.currentBodyCount = request.rigid.currentBodyCount;
        bridge.currentBodyStride = request.rigid.currentBodyStride;
        bridge.articulatedBodyCount = request.rigid.articulatedBodyCount;
        bridge.sceneBodyCount = request.rigid.sceneBodyCount;
        bridge.articulatedStride = request.rigid.articulatedStride;
        bridge.sceneStride = request.rigid.sceneStride;
        bridge.reactionStride = state.dispatch.rigidProxyCount;
        bridge.time = {
            1.0f / state.dispatch.gravityAndTimestep.w,
            state.dispatch.gravityAndTimestep.w,
            0.0f,
            0.0f,
        };

        const auto dispatchThreads = [&](
            const char* name,
            const NSUInteger count,
            const auto& bind
        ) {
            if (count == 0u) {
                return;
            }
            id<MTLComputePipelineState> pipeline = state.pipeline(name);
            [encoder setComputePipelineState:pipeline];
            bind();
            const NSUInteger width = std::min<NSUInteger>(
                256u,
                pipeline.maxTotalThreadsPerThreadgroup
            );
            [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
        };
        const auto dispatchGroups32 = [&](
            const char* name,
            const NSUInteger groupCount,
            const auto& bind
        ) {
            if (groupCount == 0u) {
                return;
            }
            [encoder setComputePipelineState:state.pipeline(name)];
            bind();
            [encoder dispatchThreadgroups:MTLSizeMake(groupCount, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        };
        const auto setDispatch = [&]() {
            [encoder setBuffer:state.dispatchBuffer offset:0u atIndex:0u];
        };
        const NSUInteger environments = state.dispatch.environmentCount;
        const NSUInteger objects = state.dispatch.objectCount;
        const NSUInteger particleTotal = environments * state.dispatch.particleCount;
        const NSUInteger gridTotal = environments * state.dispatch.gridNodeCount;
        const NSUInteger femNodeTotal = environments * state.dispatch.femNodeCount;
        const NSUInteger tetrahedronTotal = environments * state.dispatch.tetrahedronCount;
        const NSUInteger pairTotal = environments * state.dispatch.contactPairCount;
        const NSUInteger proxyTotal = environments * state.dispatch.rigidProxyCount;
        const NSUInteger objectTotal = environments * objects;

        dispatchThreads("nm_prepare_status", environments, [&] {
            setDispatch();
            [encoder setBuffer:state.statuses offset:0u atIndex:1u];
        });
        if (state.captureEvents) {
            dispatchThreads(
                "nm_prepare_events",
                environments * state.dispatch.eventStride,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.events offset:0u atIndex:1u];
                }
            );
        }
        dispatchThreads("nm_prepare_reactions", proxyTotal, [&] {
            setDispatch();
            [encoder setBuffer:state.frameReactions offset:0u atIndex:1u];
        });
        dispatchThreads("nm_prepare_scheduler", objectTotal, [&] {
            setDispatch();
            [encoder setBuffer:state.schedulers offset:0u atIndex:1u];
            [encoder setBuffer:state.schedulerPrevious offset:0u atIndex:2u];
        });
        dispatchThreads(
            "nm_reset_parameter_overlay",
            environments * state.dispatch.parameterCount,
            [&] {
                setDispatch();
                [encoder setBuffer:state.parameterDefaults offset:0u atIndex:1u];
                [encoder setBuffer:state.environmentParameters offset:0u atIndex:2u];
            }
        );

        if (request.runIdentification &&
            state.dispatch.identificationCandidateCount != 0u &&
            state.identificationDistributionCount != 0u) {
            NMIdentificationPassGPU pass{};
            pass.candidateCount = state.dispatch.identificationCandidateCount;
            pass.distributionCount = state.identificationDistributionCount;
            pass.generation = state.identificationGeneration;
            pass.seedLo = static_cast<std::uint32_t>(request.seed);
            pass.seedHi = static_cast<std::uint32_t>(request.seed >> 32u);
            if (state.identificationGeneration != 0u) {
                dispatchThreads(
                    "nm_identification_update",
                    pass.distributionCount,
                    [&] {
                        setDispatch();
                        [encoder setBytes:&pass length:sizeof(pass) atIndex:1u];
                        [encoder setBuffer:state.identificationDistributions offset:0u atIndex:2u];
                        [encoder setBuffer:state.identificationCandidates offset:0u atIndex:3u];
                        [encoder setBuffer:state.identificationLosses offset:0u atIndex:4u];
                    }
                );
            }
            dispatchThreads(
                "nm_identification_sample",
                static_cast<NSUInteger>(pass.candidateCount) * pass.distributionCount,
                [&] {
                    setDispatch();
                    [encoder setBytes:&pass length:sizeof(pass) atIndex:1u];
                    [encoder setBuffer:state.identificationDistributions offset:0u atIndex:2u];
                    [encoder setBuffer:state.identificationCandidates offset:0u atIndex:3u];
                    [encoder setBuffer:state.environmentParameters offset:0u atIndex:4u];
                }
            );
            ++state.identificationGeneration;
        }

        dispatchThreads("nm_mpm_checkpoint", particleTotal, [&] {
            setDispatch();
            [encoder setBuffer:state.particleAccepted offset:0u atIndex:1u];
            [encoder setBuffer:state.particleCheckpoint offset:0u atIndex:2u];
        });
        dispatchThreads("nm_fem_checkpoint", femNodeTotal, [&] {
            setDispatch();
            [encoder setBuffer:state.femAccepted offset:0u atIndex:1u];
            [encoder setBuffer:state.femCheckpoint offset:0u atIndex:2u];
        });
        dispatchThreads("nm_project_rigid_states", proxyTotal, [&] {
            setDispatch();
            [encoder setBytes:&bridge length:sizeof(bridge) atIndex:1u];
            [encoder setBuffer:state.rigidProxies offset:0u atIndex:2u];
            [encoder setBuffer:currentBodies offset:0u atIndex:3u];
            [encoder setBuffer:state.rigidStates offset:0u atIndex:4u];
        });

        const std::uint32_t microtickCount =
            1u << state.dispatch.maximumRateExponent;
        for (std::uint32_t microtick = 0u;
             microtick < microtickCount;
             ++microtick) {
            NMMicrostepGPU micro{};
            micro.controlStep = request.controlStep;
            micro.microtick = microtick;
            micro.microtickCount = microtickCount;
            micro.seedLo = static_cast<std::uint32_t>(request.seed);
            micro.seedHi = static_cast<std::uint32_t>(request.seed >> 32u);
            micro.runIdentification = request.runIdentification ? 1u : 0u;
            micro.runAdaptiveTransfer = request.runAdaptiveTransfer ? 1u : 0u;
            const float globalDt =
                state.dispatch.gravityAndTimestep.w / float(microtickCount);
            micro.time = {
                globalDt,
                1.0f / globalDt,
                float(microtick) * globalDt,
                0.0f,
            };

            dispatchThreads("nm_mpm_p2g", gridTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.materials offset:0u atIndex:3u];
                [encoder setBuffer:state.scalarPrograms offset:0u atIndex:4u];
                [encoder setBuffer:state.instructions offset:0u atIndex:5u];
                [encoder setBuffer:state.environmentParameters offset:0u atIndex:6u];
                [encoder setBuffer:state.particleAccepted offset:0u atIndex:7u];
                [encoder setBuffer:state.gridNodes offset:0u atIndex:8u];
                [encoder setBuffer:state.mpmStencils offset:0u atIndex:9u];
                [encoder setBuffer:state.mpmNodeIncidence offset:0u atIndex:10u];
                [encoder setBuffer:state.mpmNodeRanges offset:0u atIndex:11u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:12u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:13u];
                [encoder setBuffer:state.statuses offset:0u atIndex:14u];
            });
            dispatchThreads("nm_fem_internal_forces", tetrahedronTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.materials offset:0u atIndex:3u];
                [encoder setBuffer:state.scalarPrograms offset:0u atIndex:4u];
                [encoder setBuffer:state.instructions offset:0u atIndex:5u];
                [encoder setBuffer:state.environmentParameters offset:0u atIndex:6u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:7u];
                [encoder setBuffer:state.femTetrahedra offset:0u atIndex:8u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:9u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:10u];
                [encoder setBuffer:state.elementForces offset:0u atIndex:11u];
                [encoder setBuffer:state.statuses offset:0u atIndex:12u];
            });
            dispatchThreads("nm_fem_pcg_initialize", femNodeTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:3u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:4u];
                [encoder setBuffer:state.femTetrahedra offset:0u atIndex:5u];
                [encoder setBuffer:state.elementForces offset:0u atIndex:6u];
                [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:7u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:8u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:9u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:10u];
                [encoder setBuffer:state.femSolution offset:0u atIndex:11u];
                [encoder setBuffer:state.femResidual offset:0u atIndex:12u];
                [encoder setBuffer:state.femPreconditioned offset:0u atIndex:13u];
                [encoder setBuffer:state.femDirection offset:0u atIndex:14u];
                [encoder setBuffer:state.femOperatorValue offset:0u atIndex:15u];
                [encoder setBuffer:state.statuses offset:0u atIndex:16u];
            });

            for (std::uint32_t iteration = 0u;
                 iteration < state.dispatch.femPCGIterations;
                 ++iteration) {
                micro.pcgIteration = iteration;
                dispatchThreads("nm_fem_apply_operator_elements", tetrahedronTotal, [&] {
                    setDispatch();
                    [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.materials offset:0u atIndex:3u];
                    [encoder setBuffer:state.scalarPrograms offset:0u atIndex:4u];
                    [encoder setBuffer:state.instructions offset:0u atIndex:5u];
                    [encoder setBuffer:state.environmentParameters offset:0u atIndex:6u];
                    [encoder setBuffer:state.femAccepted offset:0u atIndex:7u];
                    [encoder setBuffer:state.femTetrahedra offset:0u atIndex:8u];
                    [encoder setBuffer:state.femDirection offset:0u atIndex:9u];
                    [encoder setBuffer:state.schedulers offset:0u atIndex:10u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:11u];
                    [encoder setBuffer:state.elementOperator offset:0u atIndex:12u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:13u];
                });
                dispatchThreads("nm_fem_apply_operator_nodes", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.femAccepted offset:0u atIndex:3u];
                    [encoder setBuffer:state.femTetrahedra offset:0u atIndex:4u];
                    [encoder setBuffer:state.elementOperator offset:0u atIndex:5u];
                    [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:6u];
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:7u];
                    [encoder setBuffer:state.schedulers offset:0u atIndex:8u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:9u];
                    [encoder setBuffer:state.femDirection offset:0u atIndex:10u];
                    [encoder setBuffer:state.femOperatorValue offset:0u atIndex:11u];
                });
                dispatchGroups32("nm_fem_reduce_pap", objectTotal, [&] {
                    setDispatch();
                    [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.schedulers offset:0u atIndex:3u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:4u];
                    [encoder setBuffer:state.femResidual offset:0u atIndex:5u];
                    [encoder setBuffer:state.femPreconditioned offset:0u atIndex:6u];
                    [encoder setBuffer:state.femDirection offset:0u atIndex:7u];
                    [encoder setBuffer:state.femOperatorValue offset:0u atIndex:8u];
                    [encoder setBuffer:state.pcgScalars offset:0u atIndex:9u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:10u];
                });
                dispatchThreads("nm_fem_pcg_step_xr", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:1u];
                    [encoder setBuffer:state.pcgScalars offset:0u atIndex:2u];
                    [encoder setBuffer:state.femSolution offset:0u atIndex:3u];
                    [encoder setBuffer:state.femResidual offset:0u atIndex:4u];
                    [encoder setBuffer:state.femDirection offset:0u atIndex:5u];
                    [encoder setBuffer:state.femOperatorValue offset:0u atIndex:6u];
                });
                dispatchThreads("nm_fem_precondition", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.femAccepted offset:0u atIndex:1u];
                    [encoder setBuffer:state.femResidual offset:0u atIndex:2u];
                    [encoder setBuffer:state.femPreconditioned offset:0u atIndex:3u];
                });
                dispatchGroups32("nm_fem_reduce_new_rz", objectTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.objects offset:0u atIndex:1u];
                    [encoder setBuffer:state.femResidual offset:0u atIndex:2u];
                    [encoder setBuffer:state.femPreconditioned offset:0u atIndex:3u];
                    [encoder setBuffer:state.pcgScalars offset:0u atIndex:4u];
                });
                dispatchThreads("nm_fem_pcg_update_direction", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:1u];
                    [encoder setBuffer:state.pcgScalars offset:0u atIndex:2u];
                    [encoder setBuffer:state.femPreconditioned offset:0u atIndex:3u];
                    [encoder setBuffer:state.femDirection offset:0u atIndex:4u];
                });
            }

            dispatchThreads("nm_fem_apply_solution", femNodeTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:3u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:4u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:5u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:6u];
                [encoder setBuffer:state.femSolution offset:0u atIndex:7u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:8u];
            });
            dispatchThreads("nm_contact_evaluate", pairTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.materials offset:0u atIndex:3u];
                [encoder setBuffer:state.gridNodes offset:0u atIndex:4u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:5u];
                [encoder setBuffer:state.rigidProxies offset:0u atIndex:6u];
                [encoder setBuffer:state.rigidStates offset:0u atIndex:7u];
                [encoder setBuffer:state.contactPairs offset:0u atIndex:8u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:9u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:10u];
                [encoder setBuffer:state.contactSamples offset:0u atIndex:11u];
                [encoder setBuffer:state.statuses offset:0u atIndex:12u];
            });
            dispatchThreads(
                "nm_contact_apply_nodes",
                environments * (state.dispatch.gridNodeCount + state.dispatch.femNodeCount),
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.contactNodeIncidence offset:0u atIndex:1u];
                    [encoder setBuffer:state.contactNodeRanges offset:0u atIndex:2u];
                    [encoder setBuffer:state.contactSamples offset:0u atIndex:3u];
                    [encoder setBuffer:state.gridNodes offset:0u atIndex:4u];
                    [encoder setBuffer:state.femCandidate offset:0u atIndex:5u];
                }
            );
            dispatchThreads("nm_contact_reduce_rigid", proxyTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.rigidIncidence offset:0u atIndex:1u];
                [encoder setBuffer:state.rigidRanges offset:0u atIndex:2u];
                [encoder setBuffer:state.contactSamples offset:0u atIndex:3u];
                [encoder setBuffer:state.microstepReactions offset:0u atIndex:4u];
            });
            dispatchThreads("nm_accumulate_rigid_reactions", proxyTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.microstepReactions offset:0u atIndex:1u];
                [encoder setBuffer:state.frameReactions offset:0u atIndex:2u];
            });
            dispatchThreads("nm_mpm_g2p", particleTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.materials offset:0u atIndex:3u];
                [encoder setBuffer:state.scalarPrograms offset:0u atIndex:4u];
                [encoder setBuffer:state.instructions offset:0u atIndex:5u];
                [encoder setBuffer:state.environmentParameters offset:0u atIndex:6u];
                [encoder setBuffer:state.particleAccepted offset:0u atIndex:7u];
                [encoder setBuffer:state.particleCandidate offset:0u atIndex:8u];
                [encoder setBuffer:state.gridNodes offset:0u atIndex:9u];
                [encoder setBuffer:state.mpmStencils offset:0u atIndex:10u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:11u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:12u];
                [encoder setBuffer:state.statuses offset:0u atIndex:13u];
            });
            dispatchThreads("nm_fem_integrate", femNodeTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:3u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:4u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:5u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:6u];
            });
            dispatchThreads("nm_fem_validate", tetrahedronTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.materials offset:0u atIndex:1u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:2u];
                [encoder setBuffer:state.femTetrahedra offset:0u atIndex:3u];
                [encoder setBuffer:state.statuses offset:0u atIndex:4u];
            });
            dispatchThreads("nm_mpm_commit_microstep", particleTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.particleAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.particleCandidate offset:0u atIndex:3u];
            });
            dispatchThreads("nm_fem_commit_microstep", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:3u];
            });
            dispatchGroups32("nm_scheduler_observe", objectTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.objects offset:0u atIndex:1u];
                [encoder setBuffer:state.contactSamples offset:0u atIndex:2u];
                [encoder setBuffer:state.gridNodes offset:0u atIndex:3u];
                [encoder setBuffer:state.mpmNodeRanges offset:0u atIndex:4u];
                [encoder setBuffer:state.particleAccepted offset:0u atIndex:5u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:6u];
                [encoder setBuffer:state.femTetrahedra offset:0u atIndex:7u];
                [encoder setBuffer:state.statuses offset:0u atIndex:8u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:9u];
            });
            dispatchThreads("nm_complete_microstep", environments, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
            });
        }

        dispatchThreads("nm_mpm_rollback_frame", particleTotal, [&] {
            setDispatch();
            [encoder setBuffer:state.statuses offset:0u atIndex:1u];
            [encoder setBuffer:state.particleAccepted offset:0u atIndex:2u];
            [encoder setBuffer:state.particleCheckpoint offset:0u atIndex:3u];
        });
        dispatchThreads("nm_fem_rollback_frame", femNodeTotal, [&] {
            setDispatch();
            [encoder setBuffer:state.statuses offset:0u atIndex:1u];
            [encoder setBuffer:state.femAccepted offset:0u atIndex:2u];
            [encoder setBuffer:state.femCheckpoint offset:0u atIndex:3u];
        });

        if (request.runAdaptiveTransfer && state.hasAdaptive) {
            dispatchGroups32("nm_adaptive_measure", objectTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.objects offset:0u atIndex:1u];
                [encoder setBuffer:state.particleAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:3u];
                [encoder setBuffer:state.femTetrahedra offset:0u atIndex:4u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:5u];
            });
            dispatchThreads("nm_adaptive_decide", objectTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.objects offset:0u atIndex:1u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:2u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:3u];
            });
            dispatchThreads("nm_adaptive_demote_to_rigid", objectTotal, [&] {
                setDispatch();
                [encoder setBytes:&bridge length:sizeof(bridge) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.rigidProxies offset:0u atIndex:3u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:4u];
                [encoder setBuffer:state.rigidStates offset:0u atIndex:5u];
                [encoder setBuffer:sceneBodies offset:0u atIndex:6u];
            });
            dispatchThreads("nm_adaptive_promote_mpm", particleTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.objects offset:0u atIndex:1u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:2u];
                [encoder setBuffer:state.rigidStates offset:0u atIndex:3u];
                [encoder setBuffer:state.particleAccepted offset:0u atIndex:4u];
            });
            dispatchThreads("nm_adaptive_promote_fem", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.objects offset:0u atIndex:1u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:3u];
                [encoder setBuffer:state.rigidStates offset:0u atIndex:4u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:5u];
            });
            dispatchThreads("nm_adaptive_finish_promotion", objectTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.objects offset:0u atIndex:1u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:2u];
            });
        }

        if (state.captureEvents) {
            dispatchThreads("nm_scheduler_finalize", objectTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.objects offset:0u atIndex:1u];
                [encoder setBuffer:state.schedulerPrevious offset:0u atIndex:2u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:3u];
                [encoder setBuffer:state.events offset:0u atIndex:4u];
            });
        }
        dispatchThreads("nm_bridge_rigid_reactions", proxyTotal, [&] {
            setDispatch();
            [encoder setBytes:&bridge length:sizeof(bridge) atIndex:1u];
            [encoder setBuffer:state.rigidProxies offset:0u atIndex:2u];
            [encoder setBuffer:state.frameReactions offset:0u atIndex:3u];
            [encoder setBuffer:articulatedWrenches offset:0u atIndex:4u];
            [encoder setBuffer:sceneBodies offset:0u atIndex:5u];
        });

        [encoder endEncoding];
        diagnostics.encoded = true;
        diagnostics.residentBytes = state.residentBytes;
        diagnostics.device = nsString(state.device.name);
        diagnostics.message = "Numi Matter graph encoded into borrowed command buffer";
        return diagnostics;
    }
}

bool Runtime::valid() const noexcept {
    return state_ != nullptr;
}

std::uint64_t Runtime::fingerprint() const noexcept {
    return state_ ? state_->worldFingerprint : 0u;
}

void* Runtime::eventBuffer() const noexcept {
    return state_ ? (__bridge void*)state_->events : nullptr;
}

void* Runtime::statusBuffer() const noexcept {
    return state_ ? (__bridge void*)state_->statuses : nullptr;
}

void* Runtime::parameterBuffer() const noexcept {
    return state_ ? (__bridge void*)state_->environmentParameters : nullptr;
}

void* Runtime::identificationLossBuffer() const noexcept {
    return state_ ? (__bridge void*)state_->identificationLosses : nullptr;
}

} // namespace numi::matter
