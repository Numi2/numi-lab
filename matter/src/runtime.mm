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
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
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

[[nodiscard]] bool fileFingerprint(
    const std::filesystem::path& path,
    std::uint64_t& output
) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return false;
    }
    std::uint64_t hash = 14695981039346656037ull;
    std::array<char, 64u * 1024u> buffer{};
    while (stream) {
        stream.read(
            buffer.data(),
            static_cast<std::streamsize>(buffer.size())
        );
        const std::streamsize count = stream.gcount();
        for (std::streamsize index = 0; index < count; ++index) {
            hash ^= static_cast<unsigned char>(buffer[
                static_cast<std::size_t>(index)
            ]);
            hash *= 1099511628211ull;
        }
    }
    if (!stream.eof()) {
        return false;
    }
    output = hash == 0u ? 1u : hash;
    return true;
}

[[nodiscard]] std::uint64_t mixFingerprint(
    std::uint64_t hash,
    const std::uint64_t value
) noexcept {
    constexpr std::uint64_t prime = 1099511628211ull;
    for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
        hash ^= (value >> (8u * byte)) & 0xffu;
        hash *= prime;
    }
    return hash;
}

[[nodiscard]] std::uint64_t makeDeviceProgramFingerprint(
    const std::uint64_t worldFingerprint,
    const std::uint64_t metallibFingerprint,
    const RuntimeConfiguration& configuration
) noexcept {
    std::uint64_t hash = 14695981039346656037ull;
    hash = mixFingerprint(hash, worldFingerprint);
    hash = mixFingerprint(hash, NM_MATTER_ABI_VERSION);
    hash = mixFingerprint(hash, metallibFingerprint);
    hash = mixFingerprint(hash, configuration.captureEvents ? 1u : 0u);
    hash = mixFingerprint(hash, configuration.captureDiagnostics ? 1u : 0u);
    hash = mixFingerprint(hash, configuration.automaticIdentification ? 1u : 0u);
    hash = mixFingerprint(hash, configuration.adaptiveTransfer ? 1u : 0u);
    return hash == 0u ? 1u : hash;
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
    const bool mpmRangesValid = std::ranges::all_of(
        world.objects,
        [&](const NMContinuumObjectGPU& object) {
            return object.representation != NM_REPRESENTATION_MPM ||
                (object.auxiliaryOffset <= world.mpm.nodes.size() &&
                 object.auxiliaryCount <=
                    world.mpm.nodes.size() - object.auxiliaryOffset);
        }
    );
    return
        dispatch.abiVersion == NM_MATTER_ABI_VERSION &&
        dispatch.materialCount == world.materials.size() &&
        dispatch.parameterCount == world.parameters.size() &&
        dispatch.objectCount == world.objects.size() &&
        dispatch.particleCount == world.mpm.particles.size() &&
        dispatch.gridNodeCount == world.mpm.nodes.size() &&
        dispatch.mpmGridCount == world.mpm.grids.size() &&
        dispatch.mpmBlockCount == world.mpm.blocks.size() &&
        dispatch.mpmBlockLookupCount == world.mpm.blockLookup.size() &&
        dispatch.maximumParticlesPerBlock ==
            NM_MPM_MAX_PARTICLES_PER_BLOCK &&
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
        mpmRangesValid &&
        world.fingerprint != 0u &&
        world.fingerprint == compiledWorldFingerprint(world);
}

} // namespace

struct Runtime::State {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

    NMMatterDispatchGPU dispatch{};
    std::uint64_t worldFingerprint = 0u;
    std::uint64_t executionFingerprint = 0u;
    std::size_t residentBytes = 0u;
    std::uint32_t identificationDistributionCount = 0u;
    std::uint32_t requiredCurrentBodyCount = 0u;
    std::uint32_t requiredBodyWrenchCount = 0u;
    std::uint32_t requiredSceneBodyCount = 0u;
    std::uint32_t reactionBodyCount = 0u;
    struct CommandOwnership {
        std::mutex mutex;
        void* activeCommandBuffer = nullptr;
        bool preDynamicsOpen = false;
        std::uint32_t controlStep = 0u;
        std::uint32_t physicsSubstep = 0u;
        std::uint32_t identificationGeneration = 0u;
        std::uint32_t identificationCheckpoint = 0u;
        bool identificationAdvanced = false;
    };

    bool captureEvents = true;
    bool automaticIdentification = false;
    bool adaptiveTransfer = true;
    bool requiresCurrentBodies = false;
    bool requiresBodyWrenches = false;
    bool requiresSceneBodies = false;
    bool hasAdaptive = false;
    std::shared_ptr<CommandOwnership> commandOwnership =
        std::make_shared<CommandOwnership>();

    id<MTLBuffer> dispatchBuffer = nil;
    id<MTLBuffer> materials = nil;
    id<MTLBuffer> parameterDefaults = nil;
    id<MTLBuffer> instructions = nil;
    id<MTLBuffer> scalarPrograms = nil;
    id<MTLBuffer> objects = nil;
    id<MTLBuffer> mpmGrids = nil;
    id<MTLBuffer> mpmBlocks = nil;
    id<MTLBuffer> mpmBlockLookup = nil;
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
    // Runtime-derived body-owned proxy incidence. It permits multiple contact
    // proxies per rigid body while retaining one deterministic wrench writer.
    id<MTLBuffer> bodyProxyIncidence = nil;
    id<MTLBuffer> bodyProxyRanges = nil;

    id<MTLBuffer> environmentParameters = nil;
    id<MTLBuffer> particleDefaults = nil;
    id<MTLBuffer> femDefaults = nil;
    id<MTLBuffer> adaptiveDefaults = nil;
    id<MTLBuffer> schedulerDefaults = nil;
    id<MTLBuffer> particleAccepted = nil;
    id<MTLBuffer> particleCandidate = nil;
    id<MTLBuffer> particleCheckpoint = nil;
    id<MTLBuffer> gridNodes = nil;
    id<MTLBuffer> mpmNodeGenerations = nil;
    id<MTLBuffer> mpmParticleKeys = nil;
    id<MTLBuffer> mpmBlockCounts = nil;
    id<MTLBuffer> mpmBlockOffsets = nil;
    id<MTLBuffer> mpmBlockCursors = nil;
    id<MTLBuffer> mpmBlockActiveFlags = nil;
    id<MTLBuffer> mpmBlockActiveLocalOffsets = nil;
    id<MTLBuffer> mpmEnvironmentActiveCounts = nil;
    id<MTLBuffer> mpmEnvironmentActiveOffsets = nil;
    id<MTLBuffer> mpmActiveBlocks = nil;
    id<MTLBuffer> mpmSortedParticleIndices = nil;
    id<MTLBuffer> mpmActiveDispatch = nil;
    id<MTLBuffer> femAccepted = nil;
    id<MTLBuffer> femCandidate = nil;
    id<MTLBuffer> femCheckpoint = nil;
    id<MTLBuffer> rigidStates = nil;
    id<MTLBuffer> contactSamples = nil;
    id<MTLBuffer> microstepReactions = nil;
    id<MTLBuffer> frameReactions = nil;
    id<MTLBuffer> adaptive = nil;
    id<MTLBuffer> schedulers = nil;
    // Per-substep snapshot used only for event edge detection.
    id<MTLBuffer> schedulerPrevious = nil;
    // Immutable control-step-start snapshot used for transactional rollback.
    id<MTLBuffer> schedulerCheckpoint = nil;
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
        candidate->automaticIdentification =
            configuration.automaticIdentification;
        candidate->adaptiveTransfer = configuration.adaptiveTransfer;
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
        std::uint64_t metallibFingerprint = 0u;
        if (!fileFingerprint(metallib, metallibFingerprint)) {
            diagnostics.message =
                "failed to fingerprint NumiMatter.metallib";
            return diagnostics;
        }
        candidate->executionFingerprint = makeDeviceProgramFingerprint(
            world.fingerprint,
            metallibFingerprint,
            configuration
        );
        NSError* libraryError = nil;
        NSString* metallibPath = [NSString
            stringWithUTF8String:metallib.string().c_str()];
        candidate->library = metallibPath == nil
            ? nil
            : [candidate->device
                  newLibraryWithURL:[NSURL fileURLWithPath:metallibPath]
                              error:&libraryError];
        if (candidate->library == nil) {
            diagnostics.message = "failed to load Numi Matter Metal library: " +
                errorString(libraryError);
            return diagnostics;
        }

        const std::array<const char*, 51> kernelNames{
            "nm_prepare_status",
            "nm_prepare_events",
            "nm_prepare_reactions",
            "nm_checkpoint_scheduler",
            "nm_prepare_scheduler",
            "nm_apply_episode_resets",
            "nm_identification_update",
            "nm_publish_identification_means",
            "nm_identification_sample",
            "nm_mpm_checkpoint",
            "nm_mpm_prepare_sparse",
            "nm_mpm_classify_particles",
            "nm_mpm_scan_environment_blocks",
            "nm_mpm_scan_environment_counts",
            "nm_mpm_scatter_active_blocks",
            "nm_mpm_scatter_particles",
            "nm_mpm_sort_block_particles",
            "nm_fem_checkpoint",
            "nm_project_rigid_states",
            "nm_publish_adaptive_rigid_ownership",
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
            "nm_latch_matter_status_into_rigid_world",
            "nm_reconcile_rigid_world_status",
            "nm_scheduler_reconcile",
            "nm_adaptive_measure",
            "nm_adaptive_observe_rigid_contacts",
            "nm_adaptive_decide",
            "nm_adaptive_demote_to_rigid",
        };
        const std::array<const char*, 5> finalKernelNames{
            "nm_adaptive_promote_mpm",
            "nm_adaptive_promote_fem",
            "nm_adaptive_finish_promotion",
            "nm_scheduler_finalize",
            "nm_finalize_status",
        };
        const auto kernel = [&](const char* name) {
            const std::string qualified =
                std::string("numi_matter_metal::") + name;
            return [candidate->library newFunctionWithName:
                [NSString stringWithUTF8String:qualified.c_str()]];
        };
        for (const char* name : kernelNames) {
            id<MTLFunction> function = kernel(name);
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
            id<MTLFunction> function = kernel(name);
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
            id<MTLFunction> function = kernel(name);
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
                "nm_mpm_scan_environment_blocks",
                "nm_mpm_scan_environment_counts",
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
        candidate->mpmGrids = uploads.one(
            std::span<const NMMPMGridGPU>(world.mpm.grids),
            valid, candidate->residentBytes);
        candidate->mpmBlocks = uploads.one(
            std::span<const NMMPMBlockGPU>(world.mpm.blocks),
            valid, candidate->residentBytes);
        candidate->mpmBlockLookup = uploads.one(
            std::span<const std::uint32_t>(world.mpm.blockLookup),
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
        candidate->particleDefaults = uploads.repeated(
            std::span<const NMParticleStateGPU>(world.mpm.particles),
            environments, valid, candidate->residentBytes);
        candidate->femDefaults = uploads.repeated(
            std::span<const NMFEMNodeStateGPU>(world.fem.nodes),
            environments, valid, candidate->residentBytes);
        candidate->adaptiveDefaults = uploads.repeated(
            std::span<const NMAdaptiveStateGPU>(world.adaptive),
            environments, valid, candidate->residentBytes);
        candidate->schedulerDefaults = uploads.repeated(
            std::span<const NMSchedulerStateGPU>(world.schedulers),
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
        candidate->schedulerCheckpoint = uploads.repeated(
            std::span<const NMSchedulerStateGPU>(world.schedulers),
            environments, valid, candidate->residentBytes);
        candidate->identificationDistributions = uploads.one(
            std::span<const NMIdentificationDistributionGPU>(world.identification),
            valid, candidate->residentBytes);
        const std::vector<NMRigidStateGPU> initialRigidStates(
            world.dispatch.rigidProxyCount
        );
        candidate->rigidStates = uploads.repeated(
            std::span<const NMRigidStateGPU>(initialRigidStates),
            environments, valid, candidate->residentBytes);
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
        candidate->mpmNodeGenerations = privateScratch<std::uint32_t>(
            candidate->device, multiplied(world.dispatch.gridNodeCount),
            valid, candidate->residentBytes);
        candidate->mpmParticleKeys = privateScratch<NMMPMParticleKeyGPU>(
            candidate->device, multiplied(world.dispatch.particleCount),
            valid, candidate->residentBytes);
        candidate->mpmBlockCounts = privateScratch<std::uint32_t>(
            candidate->device, multiplied(world.dispatch.mpmBlockCount),
            valid, candidate->residentBytes);
        candidate->mpmBlockOffsets = privateScratch<std::uint32_t>(
            candidate->device, multiplied(world.dispatch.mpmBlockCount),
            valid, candidate->residentBytes);
        candidate->mpmBlockCursors = privateScratch<std::uint32_t>(
            candidate->device, multiplied(world.dispatch.mpmBlockCount),
            valid, candidate->residentBytes);
        candidate->mpmBlockActiveFlags = privateScratch<std::uint32_t>(
            candidate->device, multiplied(world.dispatch.mpmBlockCount),
            valid, candidate->residentBytes);
        candidate->mpmBlockActiveLocalOffsets =
            privateScratch<std::uint32_t>(
                candidate->device,
                multiplied(world.dispatch.mpmBlockCount),
                valid,
                candidate->residentBytes
            );
        candidate->mpmEnvironmentActiveCounts =
            privateScratch<std::uint32_t>(
                candidate->device, environments,
                valid, candidate->residentBytes);
        candidate->mpmEnvironmentActiveOffsets =
            privateScratch<std::uint32_t>(
                candidate->device, environments,
                valid, candidate->residentBytes);
        candidate->mpmActiveBlocks = privateScratch<NMActiveMPMBlockGPU>(
            candidate->device, multiplied(world.dispatch.mpmBlockCount),
            valid, candidate->residentBytes);
        candidate->mpmSortedParticleIndices = privateScratch<std::uint32_t>(
            candidate->device, multiplied(world.dispatch.particleCount),
            valid, candidate->residentBytes);
        candidate->mpmActiveDispatch = privateScratch<NMIndirectDispatchGPU>(
            candidate->device, 1u, valid, candidate->residentBytes);
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
        candidate->femSolution = privateScratch<nm_float4>(
            candidate->device, multiplied(world.dispatch.femNodeCount),
            valid, candidate->residentBytes);
        candidate->femResidual = privateScratch<nm_float4>(
            candidate->device, multiplied(world.dispatch.femNodeCount),
            valid, candidate->residentBytes);
        candidate->femPreconditioned = privateScratch<nm_float4>(
            candidate->device, multiplied(world.dispatch.femNodeCount),
            valid, candidate->residentBytes);
        candidate->femDirection = privateScratch<nm_float4>(
            candidate->device, multiplied(world.dispatch.femNodeCount),
            valid, candidate->residentBytes);
        candidate->femOperatorValue = privateScratch<nm_float4>(
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

        for (const NMRigidProxyGPU& proxy : world.contact.rigidProxies) {
            if (proxy.bodyIndex != NM_INVALID_INDEX) {
                candidate->requiresCurrentBodies = true;
                candidate->requiredCurrentBodyCount = std::max(
                    candidate->requiredCurrentBodyCount,
                    proxy.bodyIndex + 1u
                );
            }
            if ((proxy.flags &
                 (NM_RIGID_ARTICULATED | NM_RIGID_DYNAMIC)) != 0u) {
                candidate->requiresBodyWrenches = true;
                candidate->requiredBodyWrenchCount = std::max(
                    candidate->requiredBodyWrenchCount,
                    proxy.bodyIndex + 1u
                );
            }
            if ((proxy.flags & NM_RIGID_DYNAMIC) != 0u) {
                candidate->requiresSceneBodies = true;
                candidate->requiredSceneBodyCount = std::max(
                    candidate->requiredSceneBodyCount,
                    proxy.sceneBodyIndex + 1u
                );
                if (proxy.sceneBodyIndex == NM_INVALID_INDEX) {
                    diagnostics.message =
                        "dynamic rigid proxies require a scene-body index";
                    return diagnostics;
                }
            }
        }

        candidate->reactionBodyCount =
            candidate->requiredBodyWrenchCount;
        std::vector<std::vector<std::uint32_t>> bodyProxyOwners(
            candidate->reactionBodyCount
        );
        for (std::uint32_t proxyIndex = 0u;
             proxyIndex < world.contact.rigidProxies.size();
             ++proxyIndex) {
            const NMRigidProxyGPU& proxy =
                world.contact.rigidProxies[proxyIndex];
            if (proxy.bodyIndex != NM_INVALID_INDEX &&
                (proxy.flags &
                 (NM_RIGID_ARTICULATED | NM_RIGID_DYNAMIC)) != 0u) {
                bodyProxyOwners[proxy.bodyIndex].push_back(proxyIndex);
            }
        }

        std::set<std::uint32_t> adaptiveProxyBindings;
        std::set<std::uint32_t> adaptiveBodyBindings;
        std::set<std::uint32_t> adaptiveSceneBindings;
        for (const NMContinuumObjectGPU& object : world.objects) {
            if ((object.flags & NM_OBJECT_ADAPTIVE) == 0u) {
                continue;
            }
            if (object.rigidBinding >= world.contact.rigidProxies.size()) {
                diagnostics.message =
                    "adaptive Matter object has no valid rigid proxy";
                return diagnostics;
            }
            const NMRigidProxyGPU& proxy =
                world.contact.rigidProxies[object.rigidBinding];
            if ((proxy.flags & NM_RIGID_DYNAMIC) == 0u ||
                (proxy.flags & NM_RIGID_ARTICULATED) != 0u ||
                proxy.bodyIndex == NM_INVALID_INDEX ||
                proxy.sceneBodyIndex == NM_INVALID_INDEX ||
                !adaptiveProxyBindings.insert(object.rigidBinding).second ||
                !adaptiveBodyBindings.insert(proxy.bodyIndex).second ||
                !adaptiveSceneBindings.insert(proxy.sceneBodyIndex).second) {
                diagnostics.message =
                    "adaptive Matter objects require unique free-dynamic proxy and scene-body bindings";
                return diagnostics;
            }
        }

        std::vector<std::uint32_t> bodyProxyIncidence;
        std::vector<NMIncidenceRangeGPU> bodyProxyRanges(
            candidate->reactionBodyCount
        );
        for (std::uint32_t bodyIndex = 0u;
             bodyIndex < candidate->reactionBodyCount;
             ++bodyIndex) {
            NMIncidenceRangeGPU range{};
            range.first = static_cast<nm_u32>(bodyProxyIncidence.size());
            range.count = static_cast<nm_u32>(bodyProxyOwners[bodyIndex].size());
            range.objectIndex = bodyIndex;
            bodyProxyIncidence.insert(
                bodyProxyIncidence.end(),
                bodyProxyOwners[bodyIndex].begin(),
                bodyProxyOwners[bodyIndex].end()
            );
            bodyProxyRanges[bodyIndex] = range;
        }
        UploadPlan bridgeUploads(candidate->device, candidate->queue);
        candidate->bodyProxyIncidence = bridgeUploads.one(
            std::span<const std::uint32_t>(bodyProxyIncidence),
            valid,
            candidate->residentBytes
        );
        candidate->bodyProxyRanges = bridgeUploads.one(
            std::span<const NMIncidenceRangeGPU>(bodyProxyRanges),
            valid,
            candidate->residentBytes
        );
        std::string bridgeUploadError;
        if (!valid || !bridgeUploads.finish(bridgeUploadError)) {
            diagnostics.message = valid
                ? "failed to upload body-owned Matter coupling incidence: " +
                    bridgeUploadError
                : "failed to allocate body-owned Matter coupling incidence";
            return diagnostics;
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
        if (request.phase != EncodePhase::preDynamics &&
            request.phase != EncodePhase::postCommit) {
            diagnostics.message = "unknown Matter encode phase";
            return diagnostics;
        }
        if (request.physicsSubsteps == 0u ||
            request.physicsSubstep >= request.physicsSubsteps) {
            diagnostics.message = "matter physics-substep coordinates are invalid";
            return diagnostics;
        }
        if (state.requiresCurrentBodies &&
            request.rigid.currentBodies == nullptr) {
            diagnostics.message =
                "body-backed matter proxies require the current body arena";
            return diagnostics;
        }
        if (request.phase == EncodePhase::preDynamics &&
            state.requiresBodyWrenches &&
            request.rigid.bodyWrenches == nullptr) {
            diagnostics.message =
                "two-way matter coupling requires the global body-wrench arena";
            return diagnostics;
        }
        if (state.requiresSceneBodies &&
            request.rigid.sceneBodies == nullptr) {
            diagnostics.message =
                "dynamic matter coupling requires the scene-body arena";
            return diagnostics;
        }
        if (request.phase == EncodePhase::postCommit &&
            request.environmentStatuses == nullptr) {
            diagnostics.message =
                "post-commit matter reconciliation requires MetalWorld status";
            return diagnostics;
        }
        if (request.environmentStatuses == nullptr) {
            diagnostics.message =
                "coupled Matter execution requires MetalWorld environment status";
            return diagnostics;
        }
        if (request.rigid.currentBodyCount >
                request.rigid.currentBodyStride ||
            request.rigid.bodyWrenchCount >
                request.rigid.bodyWrenchStride ||
            request.rigid.sceneBodyCount >
                request.rigid.sceneStride) {
            diagnostics.message =
                "borrowed rigid-world buffer strides are invalid";
            return diagnostics;
        }
        if (request.rigid.currentBodyCount <
                state.requiredCurrentBodyCount ||
            request.rigid.bodyWrenchCount <
                state.requiredBodyWrenchCount ||
            request.rigid.sceneBodyCount <
                state.requiredSceneBodyCount) {
            diagnostics.message =
                "borrowed rigid-world buffers do not cover every compiled Matter proxy";
            return diagnostics;
        }
        if (request.runIdentification &&
            (request.phase != EncodePhase::preDynamics ||
             request.physicsSubstep != 0u)) {
            diagnostics.message =
                "inverse material identification may run only in the first pre-dynamics substep";
            return diagnostics;
        }
        if (request.runAdaptiveTransfer &&
            (request.phase != EncodePhase::postCommit ||
             request.physicsSubstep + 1u != request.physicsSubsteps)) {
            diagnostics.message =
                "adaptive representation transfer may run only after the final rigid substep commits";
            return diagnostics;
        }
        if (request.resetMaskStepStride != 0u &&
            (request.resetMasks == nullptr ||
             request.resetMaskStepStride < state.dispatch.environmentCount)) {
            diagnostics.message =
                "matter reset-mask stream is missing or undersized";
            return diagnostics;
        }
        const float cookedTimestep =
            state.dispatch.gravityAndTimestep.w;
        const float frameTimestep = request.timestepSeconds > 0.0f
            ? request.timestepSeconds
            : cookedTimestep;
        if (!std::isfinite(frameTimestep) || !(frameTimestep > 0.0f)) {
            diagnostics.message =
                "matter frame timestep must be finite and positive";
            return diagnostics;
        }
        if (frameTimestep != cookedTimestep) {
            diagnostics.message =
                "Matter runtime timestep differs from the cooked execution graph";
            return diagnostics;
        }

        id<MTLCommandBuffer> commandBuffer =
            (__bridge id<MTLCommandBuffer>)request.commandBuffer;
        if (commandBuffer.commandQueue == nil ||
            commandBuffer.commandQueue.device.registryID !=
                state.device.registryID) {
            diagnostics.message =
                "borrowed command buffer does not belong to the initialized Matter device";
            return diagnostics;
        }

        const auto ownership = state.commandOwnership;
        std::unique_lock ownershipLock(ownership->mutex);
        if (ownership->activeCommandBuffer != nullptr &&
            ownership->activeCommandBuffer != request.commandBuffer) {
            diagnostics.message =
                "Numi Matter runtime already has another command buffer in flight";
            return diagnostics;
        }
        if (request.phase == EncodePhase::preDynamics &&
            ownership->preDynamicsOpen) {
            diagnostics.message =
                "a Matter pre-dynamics pass is still awaiting post-commit reconciliation";
            return diagnostics;
        }
        if (request.phase == EncodePhase::postCommit &&
            (!ownership->preDynamicsOpen ||
             ownership->controlStep != request.controlStep ||
             ownership->physicsSubstep != request.physicsSubstep)) {
            diagnostics.message =
                "Matter post-commit pass does not match its pre-dynamics transaction";
            return diagnostics;
        }
        if (request.runIdentification &&
            ownership->identificationAdvanced) {
            diagnostics.message =
                "inverse material identification may advance only once per command buffer";
            return diagnostics;
        }
        const bool firstEncodeForCommandBuffer =
            ownership->activeCommandBuffer == nullptr;

        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            diagnostics.message =
                "failed to create borrowed matter compute encoder";
            return diagnostics;
        }
        [encoder setLabel:request.phase == EncodePhase::preDynamics
            ? @"Numi Matter pre-dynamics"
            : @"Numi Matter post-commit"];

        const auto buffer = [&](void* value) -> id<MTLBuffer> {
            return value == nullptr
                ? state.dummy
                : (__bridge id<MTLBuffer>)value;
        };
        id<MTLBuffer> currentBodies = buffer(request.rigid.currentBodies);
        id<MTLBuffer> bodyWrenches = buffer(request.rigid.bodyWrenches);
        id<MTLBuffer> sceneBodies = buffer(request.rigid.sceneBodies);
        id<MTLBuffer> resetMasks = buffer(request.resetMasks);
        id<MTLBuffer> worldStatuses = buffer(request.environmentStatuses);

        NMMatterDispatchGPU frameDispatch = state.dispatch;
        frameDispatch.gravityAndTimestep.w = frameTimestep;
        NMBridgeDispatchGPU bridge{};
        bridge.environmentCount = state.dispatch.environmentCount;
        bridge.rigidProxyCount = state.dispatch.rigidProxyCount;
        bridge.currentBodyCount = request.rigid.currentBodyCount;
        bridge.currentBodyStride = request.rigid.currentBodyStride;
        bridge.bodyWrenchCount = request.rigid.bodyWrenchCount;
        bridge.sceneBodyCount = request.rigid.sceneBodyCount;
        bridge.bodyWrenchStride = request.rigid.bodyWrenchStride;
        bridge.sceneStride = request.rigid.sceneStride;
        bridge.reactionStride = state.dispatch.rigidProxyCount;
        bridge.reactionBodyCount = state.reactionBodyCount;
        bridge.time = {
            1.0f / frameTimestep,
            frameTimestep,
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
        const auto dispatchIndirect = [&](
            const char* name,
            id<MTLBuffer> indirect,
            const NSUInteger width,
            const auto& bind
        ) {
            id<MTLComputePipelineState> pipeline = state.pipeline(name);
            if (pipeline == nil || indirect == nil ||
                width == 0u ||
                width > pipeline.maxTotalThreadsPerThreadgroup) {
                return false;
            }
            [encoder setComputePipelineState:pipeline];
            bind();
            [encoder
                dispatchThreadgroupsWithIndirectBuffer:indirect
                indirectBufferOffset:0u
                threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
            return true;
        };
        const auto setDispatch = [&]() {
            [encoder setBytes:&frameDispatch
                       length:sizeof(frameDispatch)
                      atIndex:0u];
        };
        const NSUInteger environments = state.dispatch.environmentCount;
        const NSUInteger objects = state.dispatch.objectCount;
        const NSUInteger particleTotal =
            environments * state.dispatch.particleCount;
        const NSUInteger femNodeTotal =
            environments * state.dispatch.femNodeCount;
        const NSUInteger tetrahedronTotal =
            environments * state.dispatch.tetrahedronCount;
        const NSUInteger pairTotal =
            environments * state.dispatch.contactPairCount;
        const NSUInteger proxyTotal =
            environments * state.dispatch.rigidProxyCount;
        const NSUInteger bodyWrenchTotal =
            environments * state.reactionBodyCount;
        const NSUInteger objectTotal = environments * objects;

        if (request.phase == EncodePhase::postCommit) {
            dispatchThreads(
                "nm_reconcile_rigid_world_status",
                environments,
                [&] {
                    setDispatch();
                    [encoder setBuffer:worldStatuses offset:0u atIndex:1u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:2u];
                }
            );
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
            dispatchThreads("nm_scheduler_reconcile", objectTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.schedulerCheckpoint offset:0u atIndex:2u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:3u];
                [encoder setBuffer:state.events offset:0u atIndex:4u];
            });
            dispatchThreads("nm_project_rigid_states", proxyTotal, [&] {
                setDispatch();
                [encoder setBytes:&bridge length:sizeof(bridge) atIndex:1u];
                [encoder setBuffer:state.rigidProxies offset:0u atIndex:2u];
                [encoder setBuffer:currentBodies offset:0u atIndex:3u];
                [encoder setBuffer:state.rigidStates offset:0u atIndex:4u];
            });

            if (request.runAdaptiveTransfer && state.hasAdaptive) {
                dispatchGroups32("nm_adaptive_measure", objectTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.objects offset:0u atIndex:1u];
                    [encoder setBuffer:state.particleAccepted offset:0u atIndex:2u];
                    [encoder setBuffer:state.femAccepted offset:0u atIndex:3u];
                    [encoder setBuffer:state.femTetrahedra offset:0u atIndex:4u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:5u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:6u];
                });
                if (request.rigidContactConstraints != nullptr &&
                    request.rigidContactStatuses != nullptr &&
                    request.rigidContactConstraintStride != 0u) {
                    const std::uint32_t contactConstraintStride =
                        request.rigidContactConstraintStride;
                    dispatchThreads(
                        "nm_adaptive_observe_rigid_contacts",
                        objectTotal,
                        [&] {
                            setDispatch();
                            [encoder setBuffer:state.objects offset:0u atIndex:1u];
                            [encoder setBuffer:state.rigidProxies offset:0u atIndex:2u];
                            [encoder
                                setBuffer:(__bridge id<MTLBuffer>)request.rigidContactConstraints
                                offset:0u atIndex:3u];
                            [encoder
                                setBuffer:(__bridge id<MTLBuffer>)request.rigidContactStatuses
                                offset:0u atIndex:4u];
                            [encoder setBytes:&contactConstraintStride
                                       length:sizeof(contactConstraintStride)
                                      atIndex:5u];
                            [encoder setBuffer:state.schedulers offset:0u atIndex:6u];
                            [encoder setBuffer:state.statuses offset:0u atIndex:7u];
                        }
                    );
                }
                dispatchThreads("nm_adaptive_decide", objectTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.objects offset:0u atIndex:1u];
                    [encoder setBuffer:state.schedulers offset:0u atIndex:2u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:3u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:4u];
                });
                dispatchThreads("nm_adaptive_demote_to_rigid", objectTotal, [&] {
                    setDispatch();
                    [encoder setBytes:&bridge length:sizeof(bridge) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.rigidProxies offset:0u atIndex:3u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:4u];
                    [encoder setBuffer:state.rigidStates offset:0u atIndex:5u];
                    [encoder setBuffer:sceneBodies offset:0u atIndex:6u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:7u];
                });
                dispatchThreads("nm_adaptive_promote_mpm", particleTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.objects offset:0u atIndex:1u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:2u];
                    [encoder setBuffer:state.rigidStates offset:0u atIndex:3u];
                    [encoder setBuffer:state.particleAccepted offset:0u atIndex:4u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:5u];
                });
                dispatchThreads("nm_adaptive_promote_fem", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.objects offset:0u atIndex:1u];
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:3u];
                    [encoder setBuffer:state.rigidStates offset:0u atIndex:4u];
                    [encoder setBuffer:state.femAccepted offset:0u atIndex:5u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:6u];
                });
                dispatchThreads("nm_adaptive_finish_promotion", objectTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.objects offset:0u atIndex:1u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:2u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:3u];
                });
            }

            if (state.hasAdaptive) {
                dispatchThreads(
                    "nm_publish_adaptive_rigid_ownership",
                    objectTotal,
                    [&] {
                        setDispatch();
                        [encoder setBytes:&bridge length:sizeof(bridge) atIndex:1u];
                        [encoder setBuffer:state.objects offset:0u atIndex:2u];
                        [encoder setBuffer:state.rigidProxies offset:0u atIndex:3u];
                        [encoder setBuffer:state.adaptive offset:0u atIndex:4u];
                        [encoder setBuffer:currentBodies offset:0u atIndex:5u];
                        [encoder setBuffer:sceneBodies offset:0u atIndex:6u];
                        [encoder setBuffer:state.statuses offset:0u atIndex:7u];
                    }
                );
            }

            NMMicrostepGPU eventFrame{};
            eventFrame.controlStep = request.controlStep;
            eventFrame.microtick = request.physicsSubstep;
            eventFrame.microtickCount = request.physicsSubsteps;
            eventFrame.flags = state.captureEvents
                ? NM_MICROSTEP_CAPTURE_EVENTS
                : 0u;
            const std::uint64_t completedSubsteps =
                static_cast<std::uint64_t>(request.controlStep) *
                    request.physicsSubsteps +
                request.physicsSubstep;
            eventFrame.time = {
                frameTimestep,
                1.0f / frameTimestep,
                static_cast<float>(completedSubsteps) * frameTimestep,
                frameTimestep,
            };
            dispatchThreads("nm_scheduler_finalize", objectTotal, [&] {
                setDispatch();
                [encoder setBytes:&eventFrame length:sizeof(eventFrame) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.schedulerPrevious offset:0u atIndex:3u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:4u];
                [encoder setBuffer:state.statuses offset:0u atIndex:5u];
                [encoder setBuffer:state.events offset:0u atIndex:6u];
            });
            dispatchThreads("nm_finalize_status", environments, [&] {
                setDispatch();
                [encoder setBuffer:state.objects offset:0u atIndex:1u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:2u];
                [encoder setBuffer:state.contactSamples offset:0u atIndex:3u];
                [encoder setBuffer:state.statuses offset:0u atIndex:4u];
            });

            [encoder endEncoding];
            ownership->preDynamicsOpen = false;
            diagnostics.encoded = true;
            diagnostics.residentBytes = state.residentBytes;
            diagnostics.device = nsString(state.device.name);
            diagnostics.message =
                "Numi Matter post-commit reconciliation encoded";
            return diagnostics;
        }

        const bool firstPrePass = request.physicsSubstep == 0u;
        if (firstPrePass && request.resetMaskStepStride != 0u) {
            const std::uint32_t stateWidth = std::max({
                state.dispatch.particleCount,
                state.dispatch.femNodeCount,
                state.dispatch.objectCount,
                state.dispatch.parameterCount,
                state.dispatch.rigidProxyCount,
            });
            NMResetPassGPU reset{};
            reset.controlStep = request.controlStep;
            reset.resetMaskStepStride = request.resetMaskStepStride;
            reset.stateWidth = stateWidth;
            reset.flags = NM_RESET_ENABLED | NM_RESET_PARAMETERS;
            dispatchThreads(
                "nm_apply_episode_resets",
                environments * stateWidth,
                [&] {
                    setDispatch();
                    [encoder setBytes:&reset length:sizeof(reset) atIndex:1u];
                    [encoder setBuffer:resetMasks offset:0u atIndex:2u];
                    [encoder setBuffer:state.particleDefaults offset:0u atIndex:3u];
                    [encoder setBuffer:state.femDefaults offset:0u atIndex:4u];
                    [encoder setBuffer:state.adaptiveDefaults offset:0u atIndex:5u];
                    [encoder setBuffer:state.schedulerDefaults offset:0u atIndex:6u];
                    [encoder setBuffer:state.parameterDefaults offset:0u atIndex:7u];
                    [encoder setBuffer:state.particleAccepted offset:0u atIndex:8u];
                    [encoder setBuffer:state.particleCandidate offset:0u atIndex:9u];
                    [encoder setBuffer:state.particleCheckpoint offset:0u atIndex:10u];
                    [encoder setBuffer:state.femAccepted offset:0u atIndex:11u];
                    [encoder setBuffer:state.femCandidate offset:0u atIndex:12u];
                    [encoder setBuffer:state.femCheckpoint offset:0u atIndex:13u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:14u];
                    [encoder setBuffer:state.schedulers offset:0u atIndex:15u];
                    [encoder setBuffer:state.schedulerPrevious offset:0u atIndex:16u];
                    [encoder setBuffer:state.environmentParameters offset:0u atIndex:17u];
                    [encoder setBuffer:state.rigidStates offset:0u atIndex:18u];
                }
            );
        }

        if (firstPrePass) {
            // MetalWorld's rollback authority is the control-step checkpoint,
            // not the current physics substep. Keep this snapshot immutable
            // until the final post-commit reconciliation has completed.
            dispatchThreads("nm_checkpoint_scheduler", objectTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.schedulers offset:0u atIndex:1u];
                [encoder setBuffer:state.schedulerCheckpoint offset:0u atIndex:2u];
            });
        }

        dispatchThreads("nm_prepare_status", environments, [&] {
            setDispatch();
            [encoder setBuffer:worldStatuses offset:0u atIndex:1u];
            [encoder setBuffer:state.statuses offset:0u atIndex:2u];
        });
        if (firstPrePass && state.captureEvents) {
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
        if (state.hasAdaptive) {
            dispatchThreads(
                "nm_publish_adaptive_rigid_ownership",
                objectTotal,
                [&] {
                    setDispatch();
                    [encoder setBytes:&bridge length:sizeof(bridge) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.rigidProxies offset:0u atIndex:3u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:4u];
                    [encoder setBuffer:currentBodies offset:0u atIndex:5u];
                    [encoder setBuffer:sceneBodies offset:0u atIndex:6u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:7u];
                }
            );
        }

        if (firstPrePass && state.identificationDistributionCount != 0u) {
            NMIdentificationPassGPU identification{};
            identification.candidateCount =
                state.dispatch.identificationCandidateCount;
            identification.distributionCount =
                state.identificationDistributionCount;
            identification.generation =
                ownership->identificationGeneration;
            identification.seedLo = static_cast<std::uint32_t>(request.seed);
            identification.seedHi =
                static_cast<std::uint32_t>(request.seed >> 32u);
            if (request.runIdentification &&
                ownership->identificationGeneration != 0u) {
                dispatchThreads(
                    "nm_identification_update",
                    identification.distributionCount,
                    [&] {
                        setDispatch();
                        [encoder setBytes:&identification
                                   length:sizeof(identification)
                                  atIndex:1u];
                        [encoder setBuffer:state.identificationDistributions offset:0u atIndex:2u];
                        [encoder setBuffer:state.identificationCandidates offset:0u atIndex:3u];
                        [encoder setBuffer:state.identificationLosses offset:0u atIndex:4u];
                    }
                );
            }
            dispatchThreads(
                "nm_publish_identification_means",
                environments * identification.distributionCount,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.identificationDistributions offset:0u atIndex:1u];
                    [encoder setBuffer:state.environmentParameters offset:0u atIndex:2u];
                    [encoder setBytes:&identification.distributionCount
                               length:sizeof(identification.distributionCount)
                              atIndex:3u];
                }
            );
            if (request.runIdentification &&
                identification.candidateCount != 0u) {
                dispatchThreads(
                    "nm_identification_sample",
                    static_cast<NSUInteger>(identification.candidateCount) *
                        identification.distributionCount,
                    [&] {
                        setDispatch();
                        [encoder setBytes:&identification
                                   length:sizeof(identification)
                                  atIndex:1u];
                        [encoder setBuffer:state.identificationDistributions offset:0u atIndex:2u];
                        [encoder setBuffer:state.identificationCandidates offset:0u atIndex:3u];
                        [encoder setBuffer:state.environmentParameters offset:0u atIndex:4u];
                    }
                );
                ownership->identificationCheckpoint =
                    ownership->identificationGeneration;
                ownership->identificationAdvanced = true;
                ++ownership->identificationGeneration;
            }
        }

        if (firstPrePass) {
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
        }
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
            micro.flags = 0u;
            const std::uint64_t generation64 =
                (static_cast<std::uint64_t>(request.controlStep) *
                     request.physicsSubsteps +
                 request.physicsSubstep) *
                    microtickCount +
                microtick + 1u;
            micro.reserved = static_cast<std::uint32_t>(generation64);
            if (micro.reserved == 0u) {
                micro.reserved = 1u;
            }
            const float globalDt =
                frameTimestep / float(microtickCount);
            micro.time = {
                globalDt,
                1.0f / globalDt,
                float(request.physicsSubstep) * frameTimestep +
                float(microtick) * globalDt,
                0.0f,
            };

            const NSUInteger sparsePrepareCount = std::max<NSUInteger>(
                environments * state.dispatch.mpmBlockCount,
                environments
            );
            dispatchThreads("nm_mpm_prepare_sparse", sparsePrepareCount, [&] {
                setDispatch();
                [encoder setBuffer:state.mpmBlockCounts offset:0u atIndex:1u];
                [encoder setBuffer:state.mpmBlockOffsets offset:0u atIndex:2u];
                [encoder setBuffer:state.mpmBlockCursors offset:0u atIndex:3u];
                [encoder setBuffer:state.mpmBlockActiveFlags offset:0u atIndex:4u];
                [encoder setBuffer:state.mpmBlockActiveLocalOffsets offset:0u atIndex:5u];
                [encoder setBuffer:state.mpmEnvironmentActiveCounts offset:0u atIndex:6u];
                [encoder setBuffer:state.mpmEnvironmentActiveOffsets offset:0u atIndex:7u];
                [encoder setBuffer:state.mpmActiveDispatch offset:0u atIndex:8u];
            });
            dispatchThreads("nm_mpm_classify_particles", particleTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.mpmGrids offset:0u atIndex:3u];
                [encoder setBuffer:state.mpmBlockLookup offset:0u atIndex:4u];
                [encoder setBuffer:state.particleAccepted offset:0u atIndex:5u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:6u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:7u];
                [encoder setBuffer:state.statuses offset:0u atIndex:8u];
                [encoder setBuffer:state.mpmParticleKeys offset:0u atIndex:9u];
                [encoder setBuffer:state.mpmBlockCounts offset:0u atIndex:10u];
                [encoder setBuffer:state.mpmBlockActiveFlags offset:0u atIndex:11u];
            });
            dispatchGroups32("nm_mpm_scan_environment_blocks", environments, [&] {
                setDispatch();
                [encoder setBuffer:state.mpmBlockCounts offset:0u atIndex:1u];
                [encoder setBuffer:state.mpmBlockOffsets offset:0u atIndex:2u];
                [encoder setBuffer:state.mpmBlockActiveFlags offset:0u atIndex:3u];
                [encoder setBuffer:state.mpmBlockActiveLocalOffsets offset:0u atIndex:4u];
                [encoder setBuffer:state.mpmEnvironmentActiveCounts offset:0u atIndex:5u];
                [encoder setBuffer:state.statuses offset:0u atIndex:6u];
            });
            dispatchGroups32("nm_mpm_scan_environment_counts", 1u, [&] {
                setDispatch();
                [encoder setBuffer:state.mpmEnvironmentActiveCounts offset:0u atIndex:1u];
                [encoder setBuffer:state.mpmEnvironmentActiveOffsets offset:0u atIndex:2u];
                [encoder setBuffer:state.mpmActiveDispatch offset:0u atIndex:3u];
            });
            dispatchThreads(
                "nm_mpm_scatter_active_blocks",
                environments * state.dispatch.mpmBlockCount,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.mpmBlockActiveFlags offset:0u atIndex:1u];
                    [encoder setBuffer:state.mpmBlockActiveLocalOffsets offset:0u atIndex:2u];
                    [encoder setBuffer:state.mpmEnvironmentActiveOffsets offset:0u atIndex:3u];
                    [encoder setBuffer:state.mpmBlockOffsets offset:0u atIndex:4u];
                    [encoder setBuffer:state.mpmBlockCounts offset:0u atIndex:5u];
                    [encoder setBuffer:state.mpmActiveBlocks offset:0u atIndex:6u];
                }
            );
            dispatchThreads("nm_mpm_scatter_particles", particleTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.mpmParticleKeys offset:0u atIndex:1u];
                [encoder setBuffer:state.mpmBlockOffsets offset:0u atIndex:2u];
                [encoder setBuffer:state.mpmBlockCursors offset:0u atIndex:3u];
                [encoder setBuffer:state.mpmSortedParticleIndices offset:0u atIndex:4u];
            });
            if (!dispatchIndirect(
                    "nm_mpm_sort_block_particles",
                    state.mpmActiveDispatch,
                    NM_MPM_MAX_PARTICLES_PER_BLOCK,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.mpmActiveBlocks offset:0u atIndex:1u];
                        [encoder setBuffer:state.mpmParticleKeys offset:0u atIndex:2u];
                        [encoder setBuffer:state.mpmSortedParticleIndices offset:0u atIndex:3u];
                    }
                ) ||
                !dispatchIndirect(
                    "nm_mpm_p2g",
                    state.mpmActiveDispatch,
                    256u,
                    [&] {
                        setDispatch();
                        [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                        [encoder setBuffer:state.objects offset:0u atIndex:2u];
                        [encoder setBuffer:state.materials offset:0u atIndex:3u];
                        [encoder setBuffer:state.scalarPrograms offset:0u atIndex:4u];
                        [encoder setBuffer:state.instructions offset:0u atIndex:5u];
                        [encoder setBuffer:state.environmentParameters offset:0u atIndex:6u];
                        [encoder setBuffer:state.particleAccepted offset:0u atIndex:7u];
                        [encoder setBuffer:state.gridNodes offset:0u atIndex:8u];
                        [encoder setBuffer:state.mpmGrids offset:0u atIndex:9u];
                        [encoder setBuffer:state.mpmBlocks offset:0u atIndex:10u];
                        [encoder setBuffer:state.mpmBlockLookup offset:0u atIndex:11u];
                        [encoder setBuffer:state.mpmActiveBlocks offset:0u atIndex:12u];
                        [encoder setBuffer:state.mpmBlockCounts offset:0u atIndex:13u];
                        [encoder setBuffer:state.mpmBlockOffsets offset:0u atIndex:14u];
                        [encoder setBuffer:state.mpmSortedParticleIndices offset:0u atIndex:15u];
                        [encoder setBuffer:state.schedulers offset:0u atIndex:16u];
                        [encoder setBuffer:state.adaptive offset:0u atIndex:17u];
                        [encoder setBuffer:state.mpmNodeGenerations offset:0u atIndex:18u];
                        [encoder setBuffer:state.statuses offset:0u atIndex:19u];
                    }
                )) {
                [encoder endEncoding];
                ownership->preDynamicsOpen = false;
                diagnostics.message =
                    "failed to encode sparse MPM indirect work";
                return diagnostics;
            }
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
                    [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.schedulers offset:0u atIndex:3u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:4u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:5u];
                    [encoder setBuffer:state.femResidual offset:0u atIndex:6u];
                    [encoder setBuffer:state.femPreconditioned offset:0u atIndex:7u];
                    [encoder setBuffer:state.pcgScalars offset:0u atIndex:8u];
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
                [encoder setBuffer:state.mpmNodeGenerations offset:0u atIndex:12u];
                [encoder setBuffer:state.statuses offset:0u atIndex:13u];
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
                [encoder setBuffer:state.mpmGrids offset:0u atIndex:10u];
                [encoder setBuffer:state.mpmNodeGenerations offset:0u atIndex:11u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:12u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:13u];
                [encoder setBuffer:state.statuses offset:0u atIndex:14u];
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
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.contactSamples offset:0u atIndex:3u];
                [encoder setBuffer:state.gridNodes offset:0u atIndex:4u];
                [encoder setBuffer:state.mpmNodeRanges offset:0u atIndex:5u];
                [encoder setBuffer:state.mpmNodeGenerations offset:0u atIndex:6u];
                [encoder setBuffer:state.particleAccepted offset:0u atIndex:7u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:8u];
                [encoder setBuffer:state.femTetrahedra offset:0u atIndex:9u];
                [encoder setBuffer:state.statuses offset:0u atIndex:10u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:11u];
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

        dispatchThreads(
            "nm_latch_matter_status_into_rigid_world",
            environments,
            [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:worldStatuses offset:0u atIndex:2u];
                [encoder setBytes:&request.physicsSubstep
                           length:sizeof(request.physicsSubstep)
                          atIndex:3u];
            }
        );

        dispatchThreads("nm_bridge_rigid_reactions", bodyWrenchTotal, [&] {
            setDispatch();
            [encoder setBytes:&bridge length:sizeof(bridge) atIndex:1u];
            [encoder setBuffer:state.rigidProxies offset:0u atIndex:2u];
            [encoder setBuffer:state.frameReactions offset:0u atIndex:3u];
            [encoder setBuffer:bodyWrenches offset:0u atIndex:4u];
            [encoder setBuffer:state.statuses offset:0u atIndex:5u];
            [encoder setBuffer:state.bodyProxyIncidence offset:0u atIndex:6u];
            [encoder setBuffer:state.bodyProxyRanges offset:0u atIndex:7u];
        });

        [encoder endEncoding];
        if (firstEncodeForCommandBuffer) {
            ownership->activeCommandBuffer = request.commandBuffer;
            const std::weak_ptr<State::CommandOwnership> weakOwnership =
                ownership;
            void* const borrowedIdentity = request.commandBuffer;
            [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                if (const auto locked = weakOwnership.lock()) {
                    const std::lock_guard lock(locked->mutex);
                    if (locked->activeCommandBuffer == borrowedIdentity) {
                        if (completed.status !=
                                MTLCommandBufferStatusCompleted &&
                            locked->identificationAdvanced) {
                            locked->identificationGeneration =
                                locked->identificationCheckpoint;
                        }
                        locked->identificationAdvanced = false;
                        locked->activeCommandBuffer = nullptr;
                        locked->preDynamicsOpen = false;
                    }
                }
            }];
        }
        ownership->preDynamicsOpen = true;
        ownership->controlStep = request.controlStep;
        ownership->physicsSubstep = request.physicsSubstep;
        diagnostics.encoded = true;
        diagnostics.residentBytes = state.residentBytes;
        diagnostics.device = nsString(state.device.name);
        diagnostics.message =
            "Numi Matter pre-dynamics graph encoded into borrowed command buffer";
        return diagnostics;
    }
}

void Runtime::cancel(void* commandBuffer) noexcept {
    if (state_ == nullptr || commandBuffer == nullptr) {
        return;
    }
    try {
        const auto ownership = state_->commandOwnership;
        const std::lock_guard lock(ownership->mutex);
        if (ownership->activeCommandBuffer == commandBuffer) {
            if (ownership->identificationAdvanced) {
                ownership->identificationGeneration =
                    ownership->identificationCheckpoint;
            }
            ownership->identificationAdvanced = false;
            ownership->activeCommandBuffer = nullptr;
            ownership->preDynamicsOpen = false;
            ownership->controlStep = 0u;
            ownership->physicsSubstep = 0u;
        }
    } catch (...) {
    }
}

bool Runtime::valid() const noexcept {
    return state_ != nullptr;
}

std::uint64_t Runtime::fingerprint() const noexcept {
    return state_ ? state_->worldFingerprint : 0u;
}

std::uint64_t Runtime::deviceProgramFingerprint() const noexcept {
    return state_ ? state_->executionFingerprint : 0u;
}

bool Runtime::automaticIdentificationEnabled() const noexcept {
    return state_ != nullptr && state_->automaticIdentification;
}

bool Runtime::adaptiveTransferEnabled() const noexcept {
    return state_ != nullptr && state_->adaptiveTransfer;
}

float Runtime::timestepSeconds() const noexcept {
    return state_ ? state_->dispatch.gravityAndTimestep.w : 0.0f;
}

RuntimeStateSnapshot Runtime::snapshot() const {
    RuntimeStateSnapshot snapshot;
    if (state_ == nullptr) {
        snapshot.message = "Matter runtime is not initialized";
        return snapshot;
    }
    const auto ownership = state_->commandOwnership;
    {
        const std::lock_guard lock(ownership->mutex);
        if (ownership->activeCommandBuffer != nullptr) {
            snapshot.message =
                "Matter state cannot be read while a borrowed transaction is active";
            return snapshot;
        }
    }
    @autoreleasepool {
        const auto copy = [&](id<MTLBuffer> source) -> id<MTLBuffer> {
            return [state_->device newBufferWithLength:source.length
                                               options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> particles = copy(state_->particleAccepted);
        id<MTLBuffer> femNodes = copy(state_->femAccepted);
        id<MTLBuffer> adaptive = copy(state_->adaptive);
        id<MTLBuffer> schedulers = copy(state_->schedulers);
        id<MTLBuffer> reactions = copy(state_->frameReactions);
        id<MTLBuffer> identification =
            copy(state_->identificationDistributions);
        id<MTLBuffer> environmentParameters =
            copy(state_->environmentParameters);
        if (particles == nil || femNodes == nil || adaptive == nil ||
            schedulers == nil || reactions == nil ||
            identification == nil || environmentParameters == nil) {
            snapshot.message = "failed to allocate Matter diagnostic readback";
            return snapshot;
        }
        id<MTLCommandBuffer> commandBuffer = [state_->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
        if (commandBuffer == nil || blit == nil) {
            snapshot.message = "failed to create Matter diagnostic readback command";
            return snapshot;
        }
        const auto encodeCopy = [&](id<MTLBuffer> source, id<MTLBuffer> target) {
            if (source.length != 0u) {
                [blit copyFromBuffer:source sourceOffset:0u
                              toBuffer:target destinationOffset:0u
                                  size:source.length];
            }
        };
        encodeCopy(state_->particleAccepted, particles);
        encodeCopy(state_->femAccepted, femNodes);
        encodeCopy(state_->adaptive, adaptive);
        encodeCopy(state_->schedulers, schedulers);
        encodeCopy(state_->frameReactions, reactions);
        encodeCopy(state_->identificationDistributions, identification);
        encodeCopy(state_->environmentParameters, environmentParameters);
        [blit endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            snapshot.message = "Matter diagnostic readback command failed: " +
                errorString(commandBuffer.error);
            return snapshot;
        }
        const auto read = [](id<MTLBuffer> buffer, auto& values) {
            using Value = typename std::decay_t<decltype(values)>::value_type;
            values.resize(buffer.length / sizeof(Value));
            if (!values.empty()) {
                std::memcpy(
                    values.data(),
                    buffer.contents,
                    values.size() * sizeof(Value)
                );
            }
        };
        read(particles, snapshot.particles);
        read(femNodes, snapshot.femNodes);
        read(adaptive, snapshot.adaptive);
        read(schedulers, snapshot.schedulers);
        read(reactions, snapshot.reactions);
        read(identification, snapshot.identification);
        read(environmentParameters, snapshot.environmentParameters);
        snapshot.available = true;
    }
    return snapshot;
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
