#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "numi/matter/detail.hpp"
#include "metalrobo/engine_types.h"

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
#include <numeric>
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


} // namespace

struct Runtime::State {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

    NMMatterDispatchGPU dispatch{};
    NMMixedSolverGPU mixedSolverValue{};
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
    bool captureDiagnostics = true;
    bool automaticIdentification = false;
    bool adaptiveTransfer = true;
    bool requiresCurrentBodies = false;
    bool requiresBodyWrenches = false;
    bool requiresArticulatedResponses = false;
    bool requiresSceneBodies = false;
    bool hasAdaptive = false;
    std::shared_ptr<CommandOwnership> commandOwnership =
        std::make_shared<CommandOwnership>();

    id<MTLBuffer> dispatchBuffer = nil;
    id<MTLBuffer> mixedSolver = nil;
    id<MTLBuffer> materials = nil;
    id<MTLBuffer> mixedMaterials = nil;
    id<MTLBuffer> learnedMaterials = nil;
    id<MTLBuffer> learnedLayers = nil;
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
    id<MTLBuffer> femTetrahedraAccepted = nil;
    id<MTLBuffer> femTetrahedraCandidate = nil;
    id<MTLBuffer> femTetrahedraCheckpoint = nil;
    id<MTLBuffer> femSurfaceFaces = nil;
    id<MTLBuffer> femSurfacePrimitives = nil;
    id<MTLBuffer> femSurfaceSortKeysA = nil;
    id<MTLBuffer> femSurfaceSortKeysB = nil;
    id<MTLBuffer> femSurfaceSortIndicesA = nil;
    id<MTLBuffer> femSurfaceSortIndicesB = nil;
    id<MTLBuffer> femSurfaceActiveCounts = nil;
    id<MTLBuffer> deformableContactCandidates = nil;
    id<MTLBuffer> deformableContactCandidateCounts = nil;
    id<MTLBuffer> deformableContacts = nil;
    id<MTLBuffer> deformableWarmstartsAccepted = nil;
    id<MTLBuffer> deformableWarmstartsCandidate = nil;
    id<MTLBuffer> deformableWarmstartsCheckpoint = nil;
    id<MTLBuffer> deformableContactActiveIndices = nil;
    id<MTLBuffer> deformableContactActiveCounts = nil;
    id<MTLBuffer> deformableContactNodeIncidence = nil;
    id<MTLBuffer> deformableContactNodeRanges = nil;
    id<MTLBuffer> deformableContactActiveOffsets = nil;
    id<MTLBuffer> deformableContactGlobalActiveIndices = nil;
    id<MTLBuffer> deformableContactActiveDispatch = nil;
    id<MTLBuffer> femTopologyNodesAccepted = nil;
    id<MTLBuffer> femTopologyNodesCandidate = nil;
    id<MTLBuffer> femTopologyNodesCheckpoint = nil;
    id<MTLBuffer> cohesiveFacesAccepted = nil;
    id<MTLBuffer> cohesiveFacesCandidate = nil;
    id<MTLBuffer> cohesiveFacesCheckpoint = nil;
    id<MTLBuffer> punctureChannelsAccepted = nil;
    id<MTLBuffer> punctureChannelsCandidate = nil;
    id<MTLBuffer> punctureChannelsCheckpoint = nil;
    id<MTLBuffer> topologyStatesAccepted = nil;
    id<MTLBuffer> topologyStatesCandidate = nil;
    id<MTLBuffer> topologyStatesCheckpoint = nil;
    id<MTLBuffer> femCapacities = nil;
    id<MTLBuffer> cookedMutationCommands = nil;
    id<MTLBuffer> triggeredMutationCommands = nil;
    id<MTLBuffer> femNodeIncidence = nil;
    id<MTLBuffer> femNodeRanges = nil;
    id<MTLBuffer> femNodeIncidenceCheckpoint = nil;
    id<MTLBuffer> femNodeRangesCheckpoint = nil;
    id<MTLBuffer> topologyIncidenceCursors = nil;
    id<MTLBuffer> fieldBoundaries = nil;
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
    // Fixed contact-space sparsity. Rows are contact pairs; columns include
    // every pair sharing a continuum node, a rigid body, or conservative
    // articulated ownership. The borrowed MetalWorld response leaves entries
    // across independent articulations at exact zero. Connected components
    // provide race-free device work ownership for the PGS solve.
    id<MTLBuffer> contactResponseColumns = nil;
    id<MTLBuffer> contactResponseRows = nil;
    id<MTLBuffer> contactResponseRanges = nil;
    id<MTLBuffer> contactComponentIncidence = nil;
    id<MTLBuffer> contactComponentRanges = nil;
    id<MTLBuffer> contactActivePairs = nil;
    id<MTLBuffer> contactActiveSlotsByPair = nil;
    id<MTLBuffer> contactActiveCounts = nil;
    std::uint32_t contactResponseEntryCount = 0u;
    std::uint32_t contactComponentCount = 0u;
    std::uint32_t contactActiveCapacity = 0u;

    id<MTLBuffer> environmentParameters = nil;
    id<MTLBuffer> particleDefaults = nil;
    id<MTLBuffer> particleMaterialStateDefaults = nil;
    id<MTLBuffer> femDefaults = nil;
    id<MTLBuffer> femMaterialStateDefaults = nil;
    id<MTLBuffer> adaptiveDefaults = nil;
    id<MTLBuffer> schedulerDefaults = nil;
    id<MTLBuffer> particleAccepted = nil;
    id<MTLBuffer> particleCandidate = nil;
    id<MTLBuffer> particleCheckpoint = nil;
    id<MTLBuffer> particleMaterialStateAccepted = nil;
    id<MTLBuffer> particleMaterialStateCandidate = nil;
    id<MTLBuffer> particleMaterialStateCheckpoint = nil;
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
    id<MTLBuffer> femMaterialStateAccepted = nil;
    id<MTLBuffer> femMaterialStateCandidate = nil;
    id<MTLBuffer> femMaterialStateCheckpoint = nil;
    id<MTLBuffer> femFieldsAccepted = nil;
    id<MTLBuffer> femFieldsCandidate = nil;
    id<MTLBuffer> femFieldsCheckpoint = nil;
    id<MTLBuffer> femFieldsWork = nil;
    id<MTLBuffer> mixedFieldSolution = nil;
    id<MTLBuffer> mixedFieldResidual = nil;
    id<MTLBuffer> mixedFieldPreconditioned = nil;
    id<MTLBuffer> mixedFieldDirection = nil;
    id<MTLBuffer> mixedFieldInverseDiagonal = nil;
    id<MTLBuffer> mixedFieldOperator = nil;
    id<MTLBuffer> solverCertificates = nil;
    id<MTLBuffer> learnedWeightDefaults = nil;
    id<MTLBuffer> learnedWeightsAccepted = nil;
    id<MTLBuffer> learnedWeightsCandidate = nil;
    id<MTLBuffer> learnedWeightsCheckpoint = nil;
    id<MTLBuffer> learnedRevisionAccepted = nil;
    id<MTLBuffer> learnedRevisionCandidate = nil;
    id<MTLBuffer> learnedRevisionCheckpoint = nil;
    std::uint64_t mutationFingerprint = 0u;
    std::uint64_t learnedFingerprint = 0u;
    id<MTLBuffer> rigidStates = nil;
    id<MTLBuffer> contactSamples = nil;
    id<MTLBuffer> contactWarmstartsAccepted = nil;
    id<MTLBuffer> contactWarmstartsCandidate = nil;
    id<MTLBuffer> contactWarmstartsCheckpoint = nil;
    id<MTLBuffer> contactResponseValues = nil;
    id<MTLBuffer> articulatedPointQueries = nil;
    id<MTLBuffer> articulatedPointWorld = nil;
    id<MTLBuffer> articulatedPointJacobians = nil;
    id<MTLBuffer> articulatedRightHandSides = nil;
    id<MTLBuffer> articulatedResponseColumns = nil;
    id<MTLBuffer> articulatedInverseStatuses = nil;
    std::uint32_t articulatedInverseStatusStride = 0u;
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
    id<MTLBuffer> femLineSearch = nil;
    id<MTLBuffer> fgmresBasis = nil;
    id<MTLBuffer> fgmresPreconditionedBasis = nil;
    id<MTLBuffer> fgmresHessenberg = nil;
    id<MTLBuffer> fgmresRotations = nil;
    id<MTLBuffer> fgmresLeastSquares = nil;
    id<MTLBuffer> fgmresRestartCoefficients = nil;
    id<MTLBuffer> fgmresStates = nil;
    id<MTLBuffer> fgmresContactArguments = nil;

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
        std::string layoutError;
        if (!validateCompiledWorldLayout(world, &layoutError)) {
            diagnostics.message =
                "compiled Matter world has an invalid fixed-capacity layout: " +
                layoutError;
            return diagnostics;
        }
        if (configuration.environmentCount != 0u &&
            configuration.environmentCount != world.dispatch.environmentCount) {
            diagnostics.message = "runtime environment count must match the cooked matter package";
            return diagnostics;
        }
        auto candidate = std::make_unique<State>();
        candidate->dispatch = world.dispatch;
        candidate->mixedSolverValue = world.mixedSolver;
        candidate->mutationFingerprint = detail::hashBytes(
            world.fem.mutationCommands.data(),
            world.fem.mutationCommands.size() * sizeof(NMMutationCommandGPU)
        );
        candidate->learnedFingerprint = detail::hashBytes(
            world.learnedWeights.data(),
            world.learnedWeights.size() * sizeof(float)
        );
        candidate->worldFingerprint = world.fingerprint;
        candidate->captureEvents = configuration.captureEvents;
        candidate->captureDiagnostics = configuration.captureDiagnostics;
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

        const char* kernelNames[]{
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
            "nm_fem_prepare_candidate",
            "nm_topology_checkpoint",
            "nm_topology_detect_puncture",
            "nm_topology_execute_transaction",
            "nm_topology_rebuild_clear",
            "nm_topology_rebuild_count_mass",
            "nm_topology_rebuild_prefix",
            "nm_topology_rebuild_scatter",
            "nm_topology_rebuild_finalize",
            "nm_topology_commit",
            "nm_topology_rollback",
            "nm_mixed_checkpoint",
            "nm_mixed_prepare_microstep",
            "nm_mixed_prepare_residual",
            "nm_mixed_apply_operator_elements",
            "nm_mixed_apply_operator_nodes",
            "nm_mixed_build_residual_diagonal",
            "nm_mixed_apply_kkt_solution",
            "nm_mixed_certify",
            "nm_mixed_commit",
            "nm_mixed_rollback",
            "nm_learned_stage_weights",
            "nm_learned_checkpoint",
            "nm_learned_commit",
            "nm_learned_rollback",
            "nm_project_rigid_states",
            "nm_publish_adaptive_rigid_ownership",
            "nm_mpm_p2g",
            "nm_fem_internal_forces",
            "nm_fem_build_mechanical_residual",
            "nm_fem_apply_solution",
            "nm_fem_select_backtracking",
            "nm_fem_synchronize_environment_line_search",
            "nm_fgmres_begin",
            "nm_fgmres_measure_correction",
            "nm_fgmres_build_preconditioner",
            "nm_fgmres_precondition",
            "nm_fgmres_precondition_patches",
            "nm_fgmres_precondition_coarse",
            "nm_fgmres_import_field_residual",
            "nm_fgmres_field_smoother_initialize",
            "nm_fgmres_field_smoother_export",
            "nm_fgmres_field_smooth",
            "nm_fgmres_precondition_cross",
            "nm_fgmres_precondition_contacts",
            "nm_fgmres_precondition_deformable_contacts",
            "nm_fgmres_precondition_contact_cross",
            "nm_fem_apply_operator_elements",
            "nm_fgmres_gather_nodes",
            "nm_fgmres_apply_contacts",
            "nm_fgmres_apply_deformable_contacts",
            "nm_fgmres_orthogonalize_column",
            "nm_fgmres_finish_column",
            "nm_fgmres_form_restart_coefficients",
            "nm_fgmres_backsolve",
            "nm_fgmres_accumulate",
            "nm_fgmres_accumulate_fields",
            "nm_fgmres_accumulate_contacts",
            "nm_fgmres_accumulate_deformable_contacts",
            "nm_fgmres_restart_residual_nodes",
            "nm_fgmres_restart_residual_contacts",
            "nm_fgmres_restart_residual_deformable_contacts",
            "nm_contact_clear_samples",
            "nm_contact_build_surface_primitives",
            "nm_contact_sort_surface_primitives",
            "nm_contact_build_deformable_candidates",
            "nm_contact_narrowphase_deformable",
            "nm_contact_compact_deformable",
            "nm_contact_scan_deformable_active_counts",
            "nm_contact_scatter_deformable_active_work",
            "nm_contact_count_deformable_node_incidence",
            "nm_contact_scan_deformable_node_incidence",
            "nm_contact_scatter_deformable_node_incidence",
            "nm_contact_evaluate",
            "nm_contact_compact_active",
            "nm_contact_checkpoint_warmstarts",
            "nm_contact_commit_warmstarts",
            "nm_contact_rollback_warmstarts",
            "nm_contact_checkpoint_deformable_warmstarts",
            "nm_contact_commit_deformable_warmstarts",
            "nm_contact_rollback_deformable_warmstarts",
            "nm_contact_prepare_articulated_queries",
            "nm_contact_gather_response",
            "nm_contact_validate_articulated_response",
            "nm_contact_solve_coupled",
            "nm_contact_certify_natural_map",
            "nm_contact_accumulate_fem_residual",
            "nm_contact_accumulate_deformable_fem_residual",
            "nm_contact_build_kkt_residual",
            "nm_contact_build_deformable_kkt_residual",
            "nm_contact_apply_kkt_solution",
            "nm_contact_limit_deformable_line_search",
            "nm_contact_apply_deformable_kkt_solution",
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

        id<MTLFunction> fgmresFunction = [candidate->library
            newFunctionWithName:
                @"numi_matter_metal::nm_fgmres_begin"];
        id<MTLArgumentEncoder> fgmresContactEncoder = fgmresFunction == nil
            ? nil
            : [fgmresFunction newArgumentEncoderWithBufferIndex:13u];
        if (fgmresContactEncoder == nil) {
            diagnostics.message =
                "failed to create monolithic KKT contact argument encoder";
            return diagnostics;
        }

        bool valid = true;
        const std::size_t contactPairCount = world.contact.pairs.size();
        candidate->contactActiveCapacity = std::min<std::uint32_t>(
            candidate->dispatch.reservedMixed1,
            candidate->dispatch.contactPairCount
        );
        if (contactPairCount != 0u && candidate->contactActiveCapacity == 0u) {
            diagnostics.message =
                "Matter active contact capacity is zero for a contact world";
            return diagnostics;
        }
        std::vector<std::uint32_t> responseRows;
        std::vector<std::uint32_t> responseColumns;
        std::vector<NMIncidenceRangeGPU> responseRanges(
            candidate->contactActiveCapacity
        );
        responseRows.reserve(
            static_cast<std::size_t>(candidate->contactActiveCapacity) *
            candidate->contactActiveCapacity
        );
        responseColumns.reserve(responseRows.capacity());
        for (std::uint32_t row = 0u;
             row < candidate->contactActiveCapacity; ++row) {
            NMIncidenceRangeGPU range{};
            range.first = static_cast<nm_u32>(responseColumns.size());
            range.objectIndex = row;
            for (std::uint32_t column = 0u;
                 column < candidate->contactActiveCapacity;
                 ++column) {
                responseRows.push_back(row);
                responseColumns.push_back(column);
                ++range.count;
            }
            responseRanges[row] = range;
        }
        if (responseColumns.size() >
            std::numeric_limits<std::uint32_t>::max()) {
            diagnostics.message =
                "Matter contact response CSR exceeds 32-bit capacity";
            return diagnostics;
        }
        std::vector<std::uint32_t> componentIncidence(
            candidate->contactActiveCapacity
        );
        std::iota(componentIncidence.begin(), componentIncidence.end(), 0u);
        std::vector<NMIncidenceRangeGPU> componentRanges;
        if (candidate->contactActiveCapacity != 0u) {
            componentRanges.push_back({
                0u, candidate->contactActiveCapacity, 0u, 0u
            });
        }
        candidate->contactResponseEntryCount =
            static_cast<std::uint32_t>(responseColumns.size());
        candidate->dispatch.reservedMixed0 =
            candidate->contactResponseEntryCount;
        candidate->contactComponentCount =
            static_cast<std::uint32_t>(componentRanges.size());
        UploadPlan uploads(candidate->device, candidate->queue);
        const std::size_t environments = world.dispatch.environmentCount;
        candidate->dispatchBuffer = uploads.one(
            std::span<const NMMatterDispatchGPU>(&candidate->dispatch, 1u),
            valid, candidate->residentBytes);
        candidate->mixedSolver = uploads.one(
            std::span<const NMMixedSolverGPU>(&world.mixedSolver, 1u),
            valid, candidate->residentBytes);
        candidate->materials = uploads.one(
            std::span<const NMMaterialGPU>(world.materials),
            valid, candidate->residentBytes);
        candidate->mixedMaterials = uploads.one(
            std::span<const NMMixedMaterialGPU>(world.mixedMaterials),
            valid, candidate->residentBytes);
        candidate->learnedMaterials = uploads.one(
            std::span<const NMLearnedMaterialGPU>(world.learnedMaterials),
            valid, candidate->residentBytes);
        candidate->learnedLayers = uploads.one(
            std::span<const NMLearnedLayerGPU>(world.learnedLayers),
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
        candidate->femCapacities = uploads.one(
            std::span<const NMFEMCapacityGPU>(world.fem.capacities),
            valid, candidate->residentBytes);
        candidate->cookedMutationCommands = uploads.one(
            std::span<const NMMutationCommandGPU>(world.fem.mutationCommands),
            valid, candidate->residentBytes);
        std::vector<NMFEMTopologyStateGPU> initialTopologyStates(
            world.objects.size()
        );
        for (std::size_t objectIndex = 0u;
             objectIndex < world.objects.size(); ++objectIndex) {
            const NMContinuumObjectGPU& object = world.objects[objectIndex];
            if (object.representation != NM_REPRESENTATION_FEM) continue;
            NMFEMTopologyStateGPU topology{};
            for (std::uint32_t local = 0u; local < object.stateCount; ++local) {
                const std::size_t node = object.stateOffset + local;
                if ((world.fem.topologyNodes[node].identity.w &
                     NM_TOPOLOGY_ACTIVE) != 0u) {
                    ++topology.counts.x;
                    topology.accounting.x +=
                        world.fem.nodes[node].positionAndMass.w;
                }
            }
            for (std::uint32_t local = 0u; local < object.elementCount; ++local)
                if ((world.fem.tetrahedra[object.elementOffset + local]
                     .identity.w & NM_OBJECT_ACTIVE) != 0u)
                    ++topology.counts.y;
            for (const NMCohesiveFaceGPU& face : world.fem.cohesiveFaces)
                if (face.adjacency.y == objectIndex &&
                    (face.adjacency.w & NM_TOPOLOGY_ACTIVE) != 0u)
                    ++topology.counts.z;
            for (const NMPunctureChannelGPU& channel : world.fem.punctureChannels)
                if (channel.identity.x == objectIndex &&
                    (channel.identity.w & NM_TOPOLOGY_ACTIVE) != 0u)
                    ++topology.counts.w;
            topology.roles = {0u, 1u, 2u, object.topologyGeneration};
            initialTopologyStates[objectIndex] = topology;
        }
        candidate->topologyStatesAccepted = uploads.repeated(
            std::span<const NMFEMTopologyStateGPU>(initialTopologyStates),
            environments, valid, candidate->residentBytes);
        candidate->topologyStatesCandidate = uploads.repeated(
            std::span<const NMFEMTopologyStateGPU>(initialTopologyStates),
            environments, valid, candidate->residentBytes);
        candidate->topologyStatesCheckpoint = uploads.repeated(
            std::span<const NMFEMTopologyStateGPU>(initialTopologyStates),
            environments, valid, candidate->residentBytes);
        std::vector<std::uint32_t> environmentFEMIncidence;
        std::vector<NMIncidenceRangeGPU> environmentFEMRanges;
        environmentFEMIncidence.reserve(
            world.fem.nodeIncidence.size() * environments
        );
        environmentFEMRanges.reserve(
            world.fem.nodeRanges.size() * environments
        );
        for (std::size_t environment = 0u; environment < environments;
             ++environment) {
            environmentFEMIncidence.insert(
                environmentFEMIncidence.end(),
                world.fem.nodeIncidence.begin(),
                world.fem.nodeIncidence.end()
            );
            const std::uint32_t incidenceBase = static_cast<std::uint32_t>(
                environment * world.fem.nodeIncidence.size()
            );
            for (NMIncidenceRangeGPU range : world.fem.nodeRanges) {
                range.first += incidenceBase;
                environmentFEMRanges.push_back(range);
            }
        }
        candidate->femNodeIncidence = uploads.one(
            std::span<const std::uint32_t>(environmentFEMIncidence),
            valid, candidate->residentBytes);
        candidate->femNodeRanges = uploads.one(
            std::span<const NMIncidenceRangeGPU>(environmentFEMRanges),
            valid, candidate->residentBytes);
        candidate->femNodeIncidenceCheckpoint = privateScratch<std::uint32_t>(
            candidate->device, environmentFEMIncidence.size(),
            valid, candidate->residentBytes);
        candidate->femNodeRangesCheckpoint = privateScratch<NMIncidenceRangeGPU>(
            candidate->device, environmentFEMRanges.size(),
            valid, candidate->residentBytes);
        candidate->topologyIncidenceCursors = privateScratch<std::uint32_t>(
            candidate->device,
            static_cast<std::size_t>(environments) * world.dispatch.femNodeCount,
            valid, candidate->residentBytes);
        candidate->fieldBoundaries = uploads.one(
            std::span<const NMFieldBoundaryGPU>(world.fem.fieldBoundaries),
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
        candidate->contactResponseColumns = uploads.one(
            std::span<const std::uint32_t>(responseColumns),
            valid, candidate->residentBytes);
        candidate->contactResponseRows = uploads.one(
            std::span<const std::uint32_t>(responseRows),
            valid, candidate->residentBytes);
        candidate->contactResponseRanges = uploads.one(
            std::span<const NMIncidenceRangeGPU>(responseRanges),
            valid, candidate->residentBytes);
        candidate->contactComponentIncidence = uploads.one(
            std::span<const std::uint32_t>(componentIncidence),
            valid, candidate->residentBytes);
        candidate->contactComponentRanges = uploads.one(
            std::span<const NMIncidenceRangeGPU>(componentRanges),
            valid, candidate->residentBytes);

        std::vector<float> defaultParameters;
        defaultParameters.reserve(world.parameters.size());
        for (const NMParameterRangeGPU parameter : world.parameters) {
            defaultParameters.push_back(parameter.valueAndBounds.x);
        }
        const std::size_t materialStateStride =
            world.dispatch.materialStateStride;
        std::vector<float> particleMaterialStateDefaults(
            world.mpm.particles.size() * materialStateStride,
            0.0f
        );
        for (std::size_t particleIndex = 0u;
             particleIndex < world.mpm.particles.size();
             ++particleIndex) {
            const std::uint32_t materialIndex =
                world.mpm.particles[particleIndex].identity.y;
            if (materialIndex >= world.materials.size()) {
                diagnostics.message =
                    "MPM particle references an invalid stateful material";
                return diagnostics;
            }
            const NMMaterialGPU& material = world.materials[materialIndex];
            for (std::uint32_t stateIndex = 0u;
                 stateIndex < material.stateCount;
                 ++stateIndex) {
                particleMaterialStateDefaults[
                    particleIndex * materialStateStride + stateIndex
                ] = world.stateInitials[
                    material.stateInitialOffset + stateIndex
                ];
            }
        }
        std::vector<float> femMaterialStateDefaults(
            world.fem.tetrahedra.size() * materialStateStride,
            0.0f
        );
        for (std::size_t tetrahedronIndex = 0u;
             tetrahedronIndex < world.fem.tetrahedra.size();
             ++tetrahedronIndex) {
            const std::uint32_t materialIndex =
                world.fem.tetrahedra[tetrahedronIndex].identity.x;
            if (materialIndex >= world.materials.size()) {
                diagnostics.message =
                    "FEM element references an invalid stateful material";
                return diagnostics;
            }
            const NMMaterialGPU& material = world.materials[materialIndex];
            for (std::uint32_t stateIndex = 0u;
                 stateIndex < material.stateCount;
                 ++stateIndex) {
                femMaterialStateDefaults[
                    tetrahedronIndex * materialStateStride + stateIndex
                ] = world.stateInitials[
                    material.stateInitialOffset + stateIndex
                ];
            }
        }
        candidate->environmentParameters = uploads.repeated(
            std::span<const float>(defaultParameters),
            environments, valid, candidate->residentBytes);
        candidate->particleDefaults = uploads.repeated(
            std::span<const NMParticleStateGPU>(world.mpm.particles),
            environments, valid, candidate->residentBytes);
        candidate->particleMaterialStateDefaults = uploads.repeated(
            std::span<const float>(particleMaterialStateDefaults),
            environments, valid, candidate->residentBytes);
        candidate->femDefaults = uploads.repeated(
            std::span<const NMFEMNodeStateGPU>(world.fem.nodes),
            environments, valid, candidate->residentBytes);
        candidate->femMaterialStateDefaults = uploads.repeated(
            std::span<const float>(femMaterialStateDefaults),
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
        candidate->particleMaterialStateAccepted = uploads.repeated(
            std::span<const float>(particleMaterialStateDefaults),
            environments, valid, candidate->residentBytes);
        candidate->particleMaterialStateCandidate = uploads.repeated(
            std::span<const float>(particleMaterialStateDefaults),
            environments, valid, candidate->residentBytes);
        candidate->particleMaterialStateCheckpoint = uploads.repeated(
            std::span<const float>(particleMaterialStateDefaults),
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
        candidate->femTetrahedraAccepted = uploads.repeated(
            std::span<const NMTetrahedronGPU>(world.fem.tetrahedra),
            environments, valid, candidate->residentBytes);
        candidate->femTetrahedraCandidate = uploads.repeated(
            std::span<const NMTetrahedronGPU>(world.fem.tetrahedra),
            environments, valid, candidate->residentBytes);
        candidate->femTetrahedraCheckpoint = uploads.repeated(
            std::span<const NMTetrahedronGPU>(world.fem.tetrahedra),
            environments, valid, candidate->residentBytes);
        candidate->femSurfaceFaces = uploads.one(
            std::span<const NMFEMSurfaceFaceGPU>(world.fem.surfaceFaces),
            valid, candidate->residentBytes);
        candidate->femTopologyNodesAccepted = uploads.repeated(
            std::span<const NMFEMTopologyNodeGPU>(world.fem.topologyNodes),
            environments, valid, candidate->residentBytes);
        candidate->femTopologyNodesCandidate = uploads.repeated(
            std::span<const NMFEMTopologyNodeGPU>(world.fem.topologyNodes),
            environments, valid, candidate->residentBytes);
        candidate->femTopologyNodesCheckpoint = uploads.repeated(
            std::span<const NMFEMTopologyNodeGPU>(world.fem.topologyNodes),
            environments, valid, candidate->residentBytes);
        candidate->cohesiveFacesAccepted = uploads.repeated(
            std::span<const NMCohesiveFaceGPU>(world.fem.cohesiveFaces),
            environments, valid, candidate->residentBytes);
        candidate->cohesiveFacesCandidate = uploads.repeated(
            std::span<const NMCohesiveFaceGPU>(world.fem.cohesiveFaces),
            environments, valid, candidate->residentBytes);
        candidate->cohesiveFacesCheckpoint = uploads.repeated(
            std::span<const NMCohesiveFaceGPU>(world.fem.cohesiveFaces),
            environments, valid, candidate->residentBytes);
        candidate->punctureChannelsAccepted = uploads.repeated(
            std::span<const NMPunctureChannelGPU>(world.fem.punctureChannels),
            environments, valid, candidate->residentBytes);
        candidate->punctureChannelsCandidate = uploads.repeated(
            std::span<const NMPunctureChannelGPU>(world.fem.punctureChannels),
            environments, valid, candidate->residentBytes);
        candidate->punctureChannelsCheckpoint = uploads.repeated(
            std::span<const NMPunctureChannelGPU>(world.fem.punctureChannels),
            environments, valid, candidate->residentBytes);
        candidate->femMaterialStateAccepted = uploads.repeated(
            std::span<const float>(femMaterialStateDefaults),
            environments, valid, candidate->residentBytes);
        candidate->femMaterialStateCandidate = uploads.repeated(
            std::span<const float>(femMaterialStateDefaults),
            environments, valid, candidate->residentBytes);
        candidate->femMaterialStateCheckpoint = uploads.repeated(
            std::span<const float>(femMaterialStateDefaults),
            environments, valid, candidate->residentBytes);
        candidate->femFieldsAccepted = uploads.repeated(
            std::span<const NMFEMFieldStateGPU>(world.fem.fields),
            environments, valid, candidate->residentBytes);
        candidate->femFieldsCandidate = uploads.repeated(
            std::span<const NMFEMFieldStateGPU>(world.fem.fields),
            environments, valid, candidate->residentBytes);
        candidate->femFieldsCheckpoint = uploads.repeated(
            std::span<const NMFEMFieldStateGPU>(world.fem.fields),
            environments, valid, candidate->residentBytes);
        candidate->femFieldsWork = uploads.repeated(
            std::span<const NMFEMFieldStateGPU>(world.fem.fields),
            environments, valid, candidate->residentBytes);
        const std::size_t mixedNodeTotal =
            environments * world.dispatch.femNodeCount;
        candidate->mixedFieldSolution = privateScratch<nm_float4>(
            candidate->device, mixedNodeTotal,
            valid, candidate->residentBytes);
        candidate->mixedFieldResidual = privateScratch<nm_float4>(
            candidate->device, mixedNodeTotal,
            valid, candidate->residentBytes);
        candidate->mixedFieldPreconditioned = privateScratch<nm_float4>(
            candidate->device, mixedNodeTotal,
            valid, candidate->residentBytes);
        candidate->mixedFieldDirection = privateScratch<nm_float4>(
            candidate->device, mixedNodeTotal,
            valid, candidate->residentBytes);
        candidate->mixedFieldInverseDiagonal = privateScratch<nm_float4>(
            candidate->device, mixedNodeTotal,
            valid, candidate->residentBytes);
        candidate->mixedFieldOperator = privateScratch<nm_float4>(
            candidate->device, mixedNodeTotal,
            valid, candidate->residentBytes);
        candidate->learnedWeightDefaults = uploads.one(
            std::span<const float>(world.learnedWeights),
            valid, candidate->residentBytes);
        candidate->learnedWeightsAccepted = uploads.one(
            std::span<const float>(world.learnedWeights),
            valid, candidate->residentBytes);
        candidate->learnedWeightsCandidate = uploads.one(
            std::span<const float>(world.learnedWeights),
            valid, candidate->residentBytes);
        candidate->learnedWeightsCheckpoint = uploads.one(
            std::span<const float>(world.learnedWeights),
            valid, candidate->residentBytes);
        const std::uint32_t initialLearnedRevision = 0u;
        candidate->learnedRevisionAccepted = uploads.one(
            std::span<const std::uint32_t>(&initialLearnedRevision, 1u),
            valid, candidate->residentBytes);
        candidate->learnedRevisionCandidate = uploads.one(
            std::span<const std::uint32_t>(&initialLearnedRevision, 1u),
            valid, candidate->residentBytes);
        candidate->learnedRevisionCheckpoint = uploads.one(
            std::span<const std::uint32_t>(&initialLearnedRevision, 1u),
            valid, candidate->residentBytes);
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
        const std::vector<nm_float4> initialContactWarmstarts(
            world.dispatch.contactPairCount
        );
        candidate->contactWarmstartsAccepted = uploads.repeated(
            std::span<const nm_float4>(initialContactWarmstarts),
            environments, valid, candidate->residentBytes);
        const std::vector<NMDeformableWarmstartGPU>
            initialDeformableWarmstarts(
                world.dispatch.deformableContactCapacity);
        candidate->deformableWarmstartsAccepted = uploads.repeated(
            std::span<const NMDeformableWarmstartGPU>(
                initialDeformableWarmstarts),
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
        candidate->femSurfacePrimitives =
            privateScratch<NMFEMSurfacePrimitiveGPU>(
                candidate->device,
                multiplied(2u * world.dispatch.surfaceFaceCount),
                valid,
                candidate->residentBytes);
        const std::size_t surfacePrimitiveTotal =
            multiplied(2u * world.dispatch.surfaceFaceCount);
        candidate->femSurfaceSortKeysA = privateScratch<std::uint32_t>(
            candidate->device, surfacePrimitiveTotal,
            valid, candidate->residentBytes);
        candidate->femSurfaceSortKeysB = privateScratch<std::uint32_t>(
            candidate->device, surfacePrimitiveTotal,
            valid, candidate->residentBytes);
        candidate->femSurfaceSortIndicesA = privateScratch<std::uint32_t>(
            candidate->device, surfacePrimitiveTotal,
            valid, candidate->residentBytes);
        candidate->femSurfaceSortIndicesB = privateScratch<std::uint32_t>(
            candidate->device, surfacePrimitiveTotal,
            valid, candidate->residentBytes);
        candidate->femSurfaceActiveCounts = privateScratch<std::uint32_t>(
            candidate->device, environments,
            valid, candidate->residentBytes);
        candidate->deformableContactCandidates =
            privateScratch<NMDeformableContactCandidateGPU>(
                candidate->device,
                multiplied(world.dispatch.deformableContactCapacity),
                valid,
                candidate->residentBytes);
        candidate->deformableContactCandidateCounts =
            privateScratch<std::uint32_t>(
                candidate->device, environments,
                valid, candidate->residentBytes);
        candidate->deformableContacts = privateScratch<NMDeformableContactGPU>(
            candidate->device,
            multiplied(world.dispatch.deformableContactCapacity),
            valid,
            candidate->residentBytes);
        candidate->deformableWarmstartsCandidate =
            privateScratch<NMDeformableWarmstartGPU>(
                candidate->device,
                multiplied(world.dispatch.deformableContactCapacity),
                valid,
                candidate->residentBytes);
        candidate->deformableWarmstartsCheckpoint =
            privateScratch<NMDeformableWarmstartGPU>(
                candidate->device,
                multiplied(world.dispatch.deformableContactCapacity),
                valid,
                candidate->residentBytes);
        candidate->deformableContactActiveIndices =
            privateScratch<std::uint32_t>(
                candidate->device,
                multiplied(world.dispatch.deformableContactCapacity),
                valid,
                candidate->residentBytes);
        candidate->deformableContactActiveCounts =
            privateScratch<std::uint32_t>(
                candidate->device, environments,
                valid, candidate->residentBytes);
        candidate->deformableContactNodeIncidence =
            privateScratch<std::uint32_t>(
                candidate->device,
                4u * multiplied(world.dispatch.deformableContactCapacity),
                valid,
                candidate->residentBytes);
        candidate->deformableContactNodeRanges =
            privateScratch<NMIncidenceRangeGPU>(
                candidate->device,
                multiplied(world.dispatch.femNodeCount),
                valid,
                candidate->residentBytes);
        candidate->deformableContactActiveOffsets =
            privateScratch<std::uint32_t>(
                candidate->device,
                environments,
                valid,
                candidate->residentBytes);
        candidate->deformableContactGlobalActiveIndices =
            privateScratch<std::uint32_t>(
                candidate->device,
                multiplied(world.dispatch.deformableContactCapacity),
                valid,
                candidate->residentBytes);
        candidate->deformableContactActiveDispatch =
            privateScratch<NMIndirectDispatchGPU>(
                candidate->device,
                1u,
                valid,
                candidate->residentBytes);
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
        candidate->contactActivePairs = privateScratch<std::uint32_t>(
            candidate->device, multiplied(candidate->contactActiveCapacity),
            valid, candidate->residentBytes);
        candidate->contactActiveSlotsByPair = privateScratch<std::uint32_t>(
            candidate->device, multiplied(world.dispatch.contactPairCount),
            valid, candidate->residentBytes);
        candidate->contactActiveCounts = privateScratch<std::uint32_t>(
            candidate->device, environments,
            valid, candidate->residentBytes);
        candidate->triggeredMutationCommands =
            privateScratch<NMMutationCommandGPU>(
                candidate->device,
                multiplied(world.dispatch.objectCount),
                valid,
                candidate->residentBytes
            );
        candidate->contactWarmstartsCandidate = privateScratch<nm_float4>(
            candidate->device, multiplied(world.dispatch.contactPairCount),
            valid, candidate->residentBytes);
        candidate->contactWarmstartsCheckpoint = privateScratch<nm_float4>(
            candidate->device, multiplied(world.dispatch.contactPairCount),
            valid, candidate->residentBytes);
        candidate->contactResponseValues = privateScratch<float>(
            candidate->device,
            multiplied(candidate->contactResponseEntryCount),
            valid,
            candidate->residentBytes
        );
        candidate->articulatedInverseStatusStride =
            static_cast<std::uint32_t>(
                (candidate->contactActiveCapacity +
                 MR_ARTICULATED_INVERSE_MASS_MAX_RHS - 1u) /
                MR_ARTICULATED_INVERSE_MASS_MAX_RHS
            );
        candidate->articulatedPointQueries =
            privateScratch<MRArticulatedPointImpulseGPU>(
                candidate->device,
                multiplied(candidate->contactActiveCapacity),
                valid,
                candidate->residentBytes
            );
        candidate->articulatedPointWorld = privateScratch<nm_float4>(
            candidate->device,
            multiplied(candidate->contactActiveCapacity),
            valid,
            candidate->residentBytes
        );
        candidate->articulatedPointJacobians = privateScratch<float>(
            candidate->device,
            multiplied(candidate->contactActiveCapacity) * 3u *
                MR_ARTICULATED_ABA_MAX_DOFS,
            valid,
            candidate->residentBytes
        );
        const std::size_t articulatedColumnElements =
            multiplied(candidate->contactActiveCapacity) *
            MR_ARTICULATED_ABA_MAX_DOFS;
        candidate->articulatedRightHandSides = privateScratch<float>(
            candidate->device,
            articulatedColumnElements,
            valid,
            candidate->residentBytes
        );
        candidate->articulatedResponseColumns = privateScratch<float>(
            candidate->device,
            articulatedColumnElements,
            valid,
            candidate->residentBytes
        );
        candidate->articulatedInverseStatuses =
            privateScratch<MRInverseMassStatusGPU>(
                candidate->device,
                multiplied(candidate->articulatedInverseStatusStride),
                valid,
                candidate->residentBytes
            );
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
        const std::size_t mixedUnknownWidth =
            2u * static_cast<std::size_t>(world.dispatch.femNodeCount) +
            world.dispatch.contactPairCount +
            world.dispatch.deformableContactCapacity;
        const std::size_t mixedUnknownTotal = multiplied(mixedUnknownWidth);
        candidate->femSolution = privateScratch<nm_float4>(
            candidate->device, mixedUnknownTotal,
            valid, candidate->residentBytes);
        candidate->femResidual = privateScratch<nm_float4>(
            candidate->device, mixedUnknownTotal,
            valid, candidate->residentBytes);
        candidate->femPreconditioned = privateScratch<nm_float4>(
            candidate->device, multiplied(world.dispatch.femNodeCount),
            valid, candidate->residentBytes);
        candidate->femDirection = privateScratch<nm_float4>(
            candidate->device, multiplied(world.dispatch.femNodeCount),
            valid, candidate->residentBytes);
        candidate->femOperatorValue = privateScratch<nm_float4>(
            candidate->device, mixedUnknownTotal,
            valid, candidate->residentBytes);
        candidate->pcgScalars = privateScratch<NMPCGScalarGPU>(
            candidate->device, multiplied(world.dispatch.objectCount),
            valid, candidate->residentBytes);
        candidate->femLineSearch = privateScratch<nm_float4>(
            candidate->device, multiplied(world.dispatch.objectCount),
            valid, candidate->residentBytes);
        const std::size_t fgmresUnknownTotal = mixedUnknownTotal;
        const std::size_t fgmresSystemTotal = multiplied(1u);
        const std::size_t configuredRestart = std::clamp<std::size_t>(
            world.mixedSolver.nonlinearIterations.y,
            1u,
            NM_MIXED_FGMRES_RESTART);
        candidate->fgmresBasis = privateScratch<nm_float4>(
            candidate->device,
            fgmresUnknownTotal * (configuredRestart + 1u),
            valid, candidate->residentBytes);
        candidate->fgmresPreconditionedBasis = privateScratch<nm_float4>(
            candidate->device,
            fgmresUnknownTotal * configuredRestart,
            valid, candidate->residentBytes);
        candidate->fgmresHessenberg = privateScratch<float>(
            candidate->device,
            fgmresSystemTotal * (NM_MIXED_FGMRES_RESTART + 1u) *
                NM_MIXED_FGMRES_RESTART,
            valid, candidate->residentBytes);
        candidate->fgmresRotations = privateScratch<float>(
            candidate->device,
            fgmresSystemTotal * NM_MIXED_FGMRES_RESTART * 2u,
            valid, candidate->residentBytes);
        candidate->fgmresLeastSquares = privateScratch<float>(
            candidate->device,
            fgmresSystemTotal * (NM_MIXED_FGMRES_RESTART + 1u),
            valid, candidate->residentBytes);
        candidate->fgmresRestartCoefficients = privateScratch<float>(
            candidate->device,
            fgmresSystemTotal * (NM_MIXED_FGMRES_RESTART + 1u),
            valid, candidate->residentBytes);
        candidate->fgmresStates = privateScratch<NMFGMRESStateGPU>(
            candidate->device, fgmresSystemTotal,
            valid, candidate->residentBytes);
        candidate->solverCertificates = privateScratch<NMSolverCertificateGPU>(
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

        candidate->fgmresContactArguments = [candidate->device
            newBufferWithLength:fgmresContactEncoder.encodedLength
                         options:MTLResourceStorageModeShared];
        if (candidate->fgmresContactArguments == nil) {
            diagnostics.message =
                "failed to allocate monolithic KKT contact argument table";
            return diagnostics;
        }
        [fgmresContactEncoder
            setArgumentBuffer:candidate->fgmresContactArguments offset:0u];
        [fgmresContactEncoder setBuffer:candidate->contactPairs
                                 offset:0u atIndex:0u];
        [fgmresContactEncoder setBuffer:candidate->contactSamples
                                 offset:0u atIndex:1u];
        [fgmresContactEncoder setBuffer:candidate->contactResponseRanges
                                 offset:0u atIndex:2u];
        [fgmresContactEncoder setBuffer:candidate->contactResponseColumns
                                 offset:0u atIndex:3u];
        [fgmresContactEncoder setBuffer:candidate->contactResponseValues
                                 offset:0u atIndex:4u];
        [fgmresContactEncoder setBuffer:candidate->contactNodeIncidence
                                 offset:0u atIndex:5u];
        [fgmresContactEncoder setBuffer:candidate->contactNodeRanges
                                 offset:0u atIndex:6u];
        [fgmresContactEncoder setBuffer:candidate->contactActivePairs
                                 offset:0u atIndex:7u];
        [fgmresContactEncoder setBuffer:candidate->contactActiveSlotsByPair
                                 offset:0u atIndex:8u];
        [fgmresContactEncoder setBuffer:candidate->contactActiveCounts
                                 offset:0u atIndex:9u];
        [fgmresContactEncoder setBuffer:candidate->deformableContacts
                                 offset:0u atIndex:10u];
        [fgmresContactEncoder
            setBuffer:candidate->deformableContactActiveIndices
               offset:0u atIndex:11u];
        [fgmresContactEncoder
            setBuffer:candidate->deformableContactActiveCounts
               offset:0u atIndex:12u];
        [fgmresContactEncoder
            setBuffer:candidate->deformableContactNodeIncidence
               offset:0u atIndex:13u];
        [fgmresContactEncoder
            setBuffer:candidate->deformableContactNodeRanges
               offset:0u atIndex:14u];
        [fgmresContactEncoder
            setBuffer:candidate->deformableContactGlobalActiveIndices
               offset:0u atIndex:15u];
        [fgmresContactEncoder
            setBuffer:candidate->deformableContactActiveDispatch
               offset:0u atIndex:16u];
        candidate->residentBytes += fgmresContactEncoder.encodedLength;

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
            if ((proxy.flags & NM_RIGID_ARTICULATED) != 0u) {
                candidate->requiresArticulatedResponses = true;
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
        if (request.phase == EncodePhase::preDynamics &&
            state.requiresArticulatedResponses &&
            (request.encodeArticulatedResponses == nullptr ||
             request.articulatedResponseContext == nullptr)) {
            diagnostics.message =
                "articulated Matter contact requires a compatible borrowed inverse-ABA response service";
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
        if (request.learnedWeightUpdate != nullptr &&
            request.phase == EncodePhase::preDynamics &&
            (request.physicsSubstep != 0u ||
             request.learnedWeightCount != state.dispatch.learnedWeightCount ||
             request.learnedWeightRevision == 0u ||
             (request.expectedLearnedFingerprint != 0u &&
              request.expectedLearnedFingerprint != state.learnedFingerprint))) {
            diagnostics.message =
                "learned weights require a complete, newer first-substep candidate";
            return diagnostics;
        }
        if (request.learnedWeightUpdate == nullptr &&
            request.learnedWeightCount != 0u) {
            diagnostics.message = "learned weight count has no borrowed buffer";
            return diagnostics;
        }
        if (request.mutationCommandCount != 0u &&
            request.phase == EncodePhase::preDynamics &&
            (request.mutationCommands == nullptr ||
             request.physicsSubstep != 0u ||
             request.mutationCommandStride < sizeof(NMMutationCommandGPU))) {
            diagnostics.message =
                "mutation commands require a fixed-stride first-substep device stream";
            return diagnostics;
        }
        if (request.mutationCommandCount != 0u &&
            request.expectedMutationFingerprint != 0u &&
            request.expectedMutationFingerprint != state.mutationFingerprint) {
            diagnostics.message = "borrowed mutation fingerprint is stale";
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
        const NSUInteger femTransactionalTotal = environments * std::max(
            state.dispatch.femNodeCount,
            state.dispatch.tetrahedronCount
        );
        const NSUInteger pairTotal =
            environments * state.dispatch.contactPairCount;
        const NSUInteger deformableWarmstartTotal =
            environments * state.dispatch.deformableContactCapacity;
        const NSUInteger proxyTotal =
            environments * state.dispatch.rigidProxyCount;
        const NSUInteger bodyWrenchTotal =
            environments * state.reactionBodyCount;
        const NSUInteger objectTotal = environments * objects;
        const NSUInteger mixedTransactionalTotal = std::max(
            femNodeTotal, objectTotal
        );
        const NSUInteger topologyTransactionalTotal = environments * std::max({
            static_cast<NSUInteger>(state.dispatch.femNodeCount),
            static_cast<NSUInteger>(state.dispatch.tetrahedronCount),
            static_cast<NSUInteger>(state.dispatch.cohesiveFaceCount),
            static_cast<NSUInteger>(state.dispatch.punctureChannelCount),
            static_cast<NSUInteger>(state.dispatch.objectCount),
            static_cast<NSUInteger>(state.dispatch.tetrahedronCount) * 4u
        });

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
                [encoder setBuffer:state.particleMaterialStateAccepted offset:0u atIndex:4u];
                [encoder setBuffer:state.particleMaterialStateCheckpoint offset:0u atIndex:5u];
            });
            dispatchThreads("nm_fem_rollback_frame", femTransactionalTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.femCheckpoint offset:0u atIndex:3u];
                [encoder setBuffer:state.femMaterialStateAccepted offset:0u atIndex:4u];
                [encoder setBuffer:state.femMaterialStateCheckpoint offset:0u atIndex:5u];
            });
            dispatchThreads("nm_topology_rollback", topologyTransactionalTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.femTetrahedraAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.femTetrahedraCheckpoint offset:0u atIndex:3u];
                [encoder setBuffer:state.femTopologyNodesAccepted offset:0u atIndex:4u];
                [encoder setBuffer:state.femTopologyNodesCheckpoint offset:0u atIndex:5u];
                [encoder setBuffer:state.cohesiveFacesAccepted offset:0u atIndex:6u];
                [encoder setBuffer:state.cohesiveFacesCheckpoint offset:0u atIndex:7u];
                [encoder setBuffer:state.punctureChannelsAccepted offset:0u atIndex:8u];
                [encoder setBuffer:state.punctureChannelsCheckpoint offset:0u atIndex:9u];
                [encoder setBuffer:state.topologyStatesAccepted offset:0u atIndex:10u];
                [encoder setBuffer:state.topologyStatesCheckpoint offset:0u atIndex:11u];
                [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:12u];
                [encoder setBuffer:state.femNodeIncidenceCheckpoint offset:0u atIndex:13u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:14u];
                [encoder setBuffer:state.femNodeRangesCheckpoint offset:0u atIndex:15u];
            });
            dispatchThreads("nm_mixed_rollback", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.femFieldsAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.femFieldsCheckpoint offset:0u atIndex:3u];
            });
            dispatchThreads("nm_learned_rollback", state.dispatch.learnedWeightCount, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.learnedWeightsAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.learnedWeightsCheckpoint offset:0u atIndex:3u];
                [encoder setBuffer:state.learnedRevisionAccepted offset:0u atIndex:4u];
                [encoder setBuffer:state.learnedRevisionCheckpoint offset:0u atIndex:5u];
            });
            dispatchThreads("nm_contact_rollback_warmstarts", pairTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.contactWarmstartsAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.contactWarmstartsCheckpoint offset:0u atIndex:3u];
            });
            dispatchThreads(
                "nm_contact_rollback_deformable_warmstarts",
                deformableWarmstartTotal,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                    [encoder setBuffer:state.deformableWarmstartsAccepted
                                 offset:0u atIndex:2u];
                    [encoder setBuffer:state.deformableWarmstartsCandidate
                                 offset:0u atIndex:3u];
                    [encoder setBuffer:state.deformableWarmstartsCheckpoint
                                 offset:0u atIndex:4u];
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
                    [encoder setBuffer:state.femTetrahedraAccepted offset:0u atIndex:4u];
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
                [encoder setBuffer:state.pcgScalars offset:0u atIndex:5u];
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
                state.dispatch.tetrahedronCount,
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
                    [encoder setBuffer:state.particleMaterialStateDefaults offset:0u atIndex:19u];
                    [encoder setBuffer:state.particleMaterialStateAccepted offset:0u atIndex:20u];
                    [encoder setBuffer:state.particleMaterialStateCandidate offset:0u atIndex:21u];
                    [encoder setBuffer:state.particleMaterialStateCheckpoint offset:0u atIndex:22u];
                    [encoder setBuffer:state.femMaterialStateDefaults offset:0u atIndex:23u];
                    [encoder setBuffer:state.femMaterialStateAccepted offset:0u atIndex:24u];
                    [encoder setBuffer:state.femMaterialStateCandidate offset:0u atIndex:25u];
                    [encoder setBuffer:state.femMaterialStateCheckpoint offset:0u atIndex:26u];
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
                [encoder setBuffer:state.particleMaterialStateAccepted offset:0u atIndex:3u];
                [encoder setBuffer:state.particleMaterialStateCheckpoint offset:0u atIndex:4u];
            });
            dispatchThreads("nm_fem_checkpoint", femTransactionalTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.femAccepted offset:0u atIndex:1u];
                [encoder setBuffer:state.femCheckpoint offset:0u atIndex:2u];
                [encoder setBuffer:state.femMaterialStateAccepted offset:0u atIndex:3u];
                [encoder setBuffer:state.femMaterialStateCheckpoint offset:0u atIndex:4u];
            });
            dispatchThreads("nm_topology_checkpoint", topologyTransactionalTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.femTetrahedraAccepted offset:0u atIndex:1u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:2u];
                [encoder setBuffer:state.femTetrahedraCheckpoint offset:0u atIndex:3u];
                [encoder setBuffer:state.femTopologyNodesAccepted offset:0u atIndex:4u];
                [encoder setBuffer:state.femTopologyNodesCandidate offset:0u atIndex:5u];
                [encoder setBuffer:state.femTopologyNodesCheckpoint offset:0u atIndex:6u];
                [encoder setBuffer:state.cohesiveFacesAccepted offset:0u atIndex:7u];
                [encoder setBuffer:state.cohesiveFacesCandidate offset:0u atIndex:8u];
                [encoder setBuffer:state.cohesiveFacesCheckpoint offset:0u atIndex:9u];
                [encoder setBuffer:state.punctureChannelsAccepted offset:0u atIndex:10u];
                [encoder setBuffer:state.punctureChannelsCandidate offset:0u atIndex:11u];
                [encoder setBuffer:state.punctureChannelsCheckpoint offset:0u atIndex:12u];
                [encoder setBuffer:state.topologyStatesAccepted offset:0u atIndex:13u];
                [encoder setBuffer:state.topologyStatesCandidate offset:0u atIndex:14u];
                [encoder setBuffer:state.topologyStatesCheckpoint offset:0u atIndex:15u];
                [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:16u];
                [encoder setBuffer:state.femNodeIncidenceCheckpoint offset:0u atIndex:17u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:18u];
                [encoder setBuffer:state.femNodeRangesCheckpoint offset:0u atIndex:19u];
            });
            dispatchThreads("nm_mixed_checkpoint", mixedTransactionalTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.femFieldsAccepted offset:0u atIndex:1u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:2u];
                [encoder setBuffer:state.femFieldsCheckpoint offset:0u atIndex:3u];
                [encoder setBuffer:state.solverCertificates offset:0u atIndex:4u];
            });
            dispatchThreads("nm_learned_checkpoint", state.dispatch.learnedWeightCount, [&] {
                setDispatch();
                [encoder setBuffer:state.learnedWeightsAccepted offset:0u atIndex:1u];
                [encoder setBuffer:state.learnedWeightsCandidate offset:0u atIndex:2u];
                [encoder setBuffer:state.learnedWeightsCheckpoint offset:0u atIndex:3u];
                [encoder setBuffer:state.learnedRevisionAccepted offset:0u atIndex:4u];
                [encoder setBuffer:state.learnedRevisionCandidate offset:0u atIndex:5u];
                [encoder setBuffer:state.learnedRevisionCheckpoint offset:0u atIndex:6u];
            });
            dispatchThreads("nm_contact_checkpoint_warmstarts", pairTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.contactWarmstartsAccepted offset:0u atIndex:1u];
                [encoder setBuffer:state.contactWarmstartsCandidate offset:0u atIndex:2u];
                [encoder setBuffer:state.contactWarmstartsCheckpoint offset:0u atIndex:3u];
            });
            dispatchThreads(
                "nm_contact_checkpoint_deformable_warmstarts",
                deformableWarmstartTotal,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.deformableWarmstartsAccepted
                                 offset:0u atIndex:1u];
                    [encoder setBuffer:state.deformableWarmstartsCandidate
                                 offset:0u atIndex:2u];
                    [encoder setBuffer:state.deformableWarmstartsCheckpoint
                                 offset:0u atIndex:3u];
                });
            if (request.learnedWeightUpdate != nullptr) {
                id<MTLBuffer> learnedUpdate =
                    (__bridge id<MTLBuffer>)request.learnedWeightUpdate;
                dispatchThreads("nm_learned_stage_weights", state.dispatch.learnedWeightCount, [&] {
                    setDispatch();
                    [encoder setBuffer:learnedUpdate offset:0u atIndex:1u];
                    [encoder setBuffer:state.learnedWeightsCandidate offset:0u atIndex:2u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:3u];
                    [encoder setBuffer:state.learnedLayers offset:0u atIndex:4u];
                    [encoder setBuffer:state.learnedRevisionAccepted offset:0u atIndex:5u];
                    [encoder setBuffer:state.learnedRevisionCandidate offset:0u atIndex:6u];
                    [encoder setBytes:&request.learnedWeightCount
                               length:sizeof(request.learnedWeightCount)
                              atIndex:7u];
                    [encoder setBytes:&request.learnedWeightRevision
                               length:sizeof(request.learnedWeightRevision)
                              atIndex:8u];
                });
            }
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
        id<MTLBuffer> borrowedMutations = request.mutationCommands == nullptr
            ? state.dummy
            : (__bridge id<MTLBuffer>)request.mutationCommands;
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

            const auto encodeTopologyRebuild = [&](id<MTLBuffer> nodes) {
                dispatchThreads("nm_topology_rebuild_clear", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:nodes offset:0u atIndex:1u];
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                    [encoder setBuffer:state.topologyIncidenceCursors offset:0u atIndex:3u];
                });
                dispatchThreads("nm_topology_rebuild_count_mass", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.materials offset:0u atIndex:1u];
                    [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:2u];
                    [encoder setBuffer:nodes offset:0u atIndex:3u];
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:4u];
                });
                dispatchGroups32("nm_topology_rebuild_prefix", environments, [&] {
                    setDispatch();
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:1u];
                    [encoder setBuffer:state.topologyIncidenceCursors offset:0u atIndex:2u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:3u];
                });
                dispatchThreads("nm_topology_rebuild_scatter", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:1u];
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                    [encoder setBuffer:state.topologyIncidenceCursors offset:0u atIndex:3u];
                    [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:4u];
                });
                dispatchThreads("nm_topology_rebuild_finalize", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:nodes offset:0u atIndex:1u];
                    [encoder setBuffer:state.femTopologyNodesCandidate offset:0u atIndex:2u];
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:3u];
                    [encoder setBuffer:state.topologyStatesCandidate offset:0u atIndex:4u];
                });
            };

            if (firstPrePass && microtick == 0u) {
                dispatchThreads("nm_topology_detect_puncture", objectTotal, [&] {
                    setDispatch();
                    [encoder setBytes:&request.controlStep
                               length:sizeof(request.controlStep) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.femCapacities offset:0u atIndex:3u];
                    [encoder setBuffer:state.rigidProxies offset:0u atIndex:4u];
                    [encoder setBuffer:state.rigidStates offset:0u atIndex:5u];
                    [encoder setBuffer:state.femAccepted offset:0u atIndex:6u];
                    [encoder setBuffer:state.femTetrahedraAccepted offset:0u
                               atIndex:7u];
                    [encoder setBuffer:state.punctureChannelsAccepted offset:0u
                               atIndex:8u];
                    [encoder setBuffer:state.contactPairs offset:0u atIndex:9u];
                    [encoder setBuffer:state.contactWarmstartsAccepted offset:0u
                               atIndex:10u];
                    [encoder setBuffer:state.triggeredMutationCommands offset:0u
                               atIndex:11u];
                });
                const std::uint32_t triggeredMutationCount =
                    state.dispatch.objectCount;
                const std::uint32_t cookedMutationCount =
                    state.dispatch.mutationCommandCount;
                dispatchThreads("nm_topology_execute_transaction", environments, [&] {
                    setDispatch();
                    [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.femCapacities offset:0u atIndex:3u];
                    [encoder setBuffer:state.materials offset:0u atIndex:4u];
                    [encoder setBuffer:state.cookedMutationCommands offset:0u atIndex:5u];
                    [encoder setBuffer:borrowedMutations offset:0u atIndex:6u];
                    [encoder setBytes:&request.mutationCommandCount
                               length:sizeof(request.mutationCommandCount) atIndex:7u];
                    [encoder setBytes:&request.mutationCommandStride
                               length:sizeof(request.mutationCommandStride) atIndex:8u];
                    [encoder setBuffer:state.femAccepted offset:0u atIndex:9u];
                    [encoder setBuffer:state.femFieldsAccepted offset:0u atIndex:10u];
                    [encoder setBuffer:state.femTopologyNodesCandidate offset:0u atIndex:11u];
                    [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:12u];
                    [encoder setBuffer:state.cohesiveFacesCandidate offset:0u atIndex:13u];
                    [encoder setBuffer:state.punctureChannelsCandidate offset:0u atIndex:14u];
                    [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:15u];
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:16u];
                    [encoder setBuffer:state.topologyStatesCandidate offset:0u atIndex:17u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:18u];
                    [encoder setBuffer:state.triggeredMutationCommands offset:0u
                               atIndex:19u];
                    [encoder setBytes:&triggeredMutationCount
                               length:sizeof(triggeredMutationCount) atIndex:20u];
                    [encoder setBytes:&cookedMutationCount
                               length:sizeof(cookedMutationCount) atIndex:21u];
                });
                encodeTopologyRebuild(state.femAccepted);
            }

            dispatchThreads("nm_mixed_prepare_microstep", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.femFieldsAccepted offset:0u atIndex:1u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:2u];
            });

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
                        [encoder setBuffer:state.particleMaterialStateAccepted offset:0u atIndex:8u];
                        [encoder setBuffer:state.gridNodes offset:0u atIndex:9u];
                        [encoder setBuffer:state.mpmGrids offset:0u atIndex:10u];
                        [encoder setBuffer:state.mpmBlocks offset:0u atIndex:11u];
                        [encoder setBuffer:state.mpmBlockLookup offset:0u atIndex:12u];
                        [encoder setBuffer:state.mpmActiveBlocks offset:0u atIndex:13u];
                        [encoder setBuffer:state.mpmBlockCounts offset:0u atIndex:14u];
                        [encoder setBuffer:state.mpmBlockOffsets offset:0u atIndex:15u];
                        [encoder setBuffer:state.mpmSortedParticleIndices offset:0u atIndex:16u];
                        [encoder setBuffer:state.schedulers offset:0u atIndex:17u];
                        [encoder setBuffer:state.adaptive offset:0u atIndex:18u];
                        [encoder setBuffer:state.mpmNodeGenerations offset:0u atIndex:19u];
                        [encoder setBuffer:state.statuses offset:0u atIndex:20u];
                    }
                )) {
                [encoder endEncoding];
                ownership->preDynamicsOpen = false;
                diagnostics.message =
                    "failed to encode sparse MPM indirect work";
                return diagnostics;
            }
            dispatchThreads("nm_fem_prepare_candidate", femNodeTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:3u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:4u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:5u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:6u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:7u];
            });
            dispatchThreads("nm_contact_clear_samples", pairTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.contactSamples offset:0u atIndex:1u];
                [encoder setBuffer:state.contactWarmstartsCandidate offset:0u atIndex:2u];
            });
            // Contact is a block of the nonlinear KKT residual, not a
            // post-FEM correction. Re-evaluate candidate geometry and stream
            // inverse-ABA columns on every Newton candidate, then accumulate
            // J^T lambda before the single fused FGMRES solve below. The
            // callback only encodes into the borrowed command buffer; it
            // never commits, waits, or opens another queue.
            const auto encodeCoupledKKTContact = [&] (
                const bool certify,
                id<MTLBuffer> warmstarts,
                id<MTLBuffer> deformableWarmstarts
            ) -> bool {
                dispatchThreads(
                    "nm_contact_build_surface_primitives",
                    2u * environments * state.dispatch.surfaceFaceCount,
                    [&] {
                        setDispatch();
                        [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                        [encoder setBuffer:state.objects offset:0u atIndex:2u];
                        [encoder setBuffer:state.femSurfaceFaces offset:0u atIndex:3u];
                        [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:4u];
                        [encoder setBuffer:state.femCandidate offset:0u atIndex:5u];
                        [encoder setBuffer:state.schedulers offset:0u atIndex:6u];
                        [encoder setBuffer:state.femSurfacePrimitives offset:0u atIndex:7u];
                    }
                );
                dispatchGroups32(
                    "nm_contact_sort_surface_primitives",
                    environments,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.femSurfacePrimitives offset:0u atIndex:1u];
                        [encoder setBuffer:state.femSurfaceSortKeysA offset:0u atIndex:2u];
                        [encoder setBuffer:state.femSurfaceSortKeysB offset:0u atIndex:3u];
                        [encoder setBuffer:state.femSurfaceSortIndicesA offset:0u atIndex:4u];
                        [encoder setBuffer:state.femSurfaceSortIndicesB offset:0u atIndex:5u];
                        [encoder setBuffer:state.femSurfaceActiveCounts offset:0u atIndex:6u];
                    }
                );
                dispatchGroups32(
                    "nm_contact_build_deformable_candidates",
                    environments,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.femSurfacePrimitives offset:0u atIndex:1u];
                        [encoder setBuffer:state.femSurfaceSortIndicesA offset:0u atIndex:2u];
                        [encoder setBuffer:state.femSurfaceActiveCounts offset:0u atIndex:3u];
                        [encoder setBuffer:state.deformableContactCandidates offset:0u atIndex:4u];
                        [encoder setBuffer:state.deformableContactCandidateCounts offset:0u atIndex:5u];
                    }
                );
                dispatchThreads(
                    "nm_contact_narrowphase_deformable",
                    environments * state.dispatch.deformableContactCapacity,
                    [&] {
                        setDispatch();
                        [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                        [encoder setBuffer:state.objects offset:0u atIndex:2u];
                        [encoder setBuffer:state.materials offset:0u atIndex:3u];
                        [encoder setBuffer:state.femCandidate offset:0u atIndex:4u];
                        [encoder setBuffer:state.schedulers offset:0u atIndex:5u];
                        [encoder setBuffer:state.femSurfacePrimitives offset:0u atIndex:6u];
                        [encoder setBuffer:state.deformableContactCandidates offset:0u atIndex:7u];
                        [encoder setBuffer:state.deformableContactCandidateCounts offset:0u atIndex:8u];
                        [encoder setBuffer:deformableWarmstarts offset:0u atIndex:9u];
                        [encoder setBuffer:state.deformableContacts offset:0u atIndex:10u];
                    }
                );
                dispatchGroups32(
                    "nm_contact_compact_deformable",
                    environments,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.deformableContacts
                                   offset:0u atIndex:1u];
                        [encoder setBuffer:state.deformableContactActiveIndices
                                   offset:0u atIndex:2u];
                        [encoder setBuffer:state.deformableContactActiveCounts
                                   offset:0u atIndex:3u];
                    }
                );
                dispatchGroups32(
                    "nm_contact_scan_deformable_active_counts",
                    1u,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.deformableContactActiveCounts
                                   offset:0u atIndex:1u];
                        [encoder setBuffer:state.deformableContactActiveOffsets
                                   offset:0u atIndex:2u];
                        [encoder setBuffer:state.deformableContactActiveDispatch
                                   offset:0u atIndex:3u];
                    });
                dispatchThreads(
                    "nm_contact_scatter_deformable_active_work",
                    environments * state.dispatch.deformableContactCapacity,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.deformableContactActiveIndices
                                   offset:0u atIndex:1u];
                        [encoder setBuffer:state.deformableContactActiveCounts
                                   offset:0u atIndex:2u];
                        [encoder setBuffer:state.deformableContactActiveOffsets
                                   offset:0u atIndex:3u];
                        [encoder setBuffer:state.deformableContactGlobalActiveIndices
                                   offset:0u atIndex:4u];
                    });
                dispatchThreads(
                    "nm_contact_count_deformable_node_incidence",
                    femNodeTotal,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.deformableContacts
                                   offset:0u atIndex:1u];
                        [encoder setBuffer:state.deformableContactActiveIndices
                                   offset:0u atIndex:2u];
                        [encoder setBuffer:state.deformableContactActiveCounts
                                   offset:0u atIndex:3u];
                        [encoder setBuffer:state.deformableContactNodeRanges
                                   offset:0u atIndex:4u];
                    });
                dispatchGroups32(
                    "nm_contact_scan_deformable_node_incidence",
                    environments,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.deformableContactNodeRanges
                                   offset:0u atIndex:1u];
                        [encoder setBuffer:state.statuses
                                   offset:0u atIndex:2u];
                    });
                dispatchThreads(
                    "nm_contact_scatter_deformable_node_incidence",
                    femNodeTotal,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.deformableContacts
                                   offset:0u atIndex:1u];
                        [encoder setBuffer:state.deformableContactActiveIndices
                                   offset:0u atIndex:2u];
                        [encoder setBuffer:state.deformableContactActiveCounts
                                   offset:0u atIndex:3u];
                        [encoder setBuffer:state.deformableContactNodeRanges
                                   offset:0u atIndex:4u];
                        [encoder setBuffer:state.deformableContactNodeIncidence
                                   offset:0u atIndex:5u];
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
                    [encoder setBuffer:warmstarts offset:0u atIndex:14u];
                });
                dispatchGroups32("nm_contact_compact_active", environments, [&] {
                    setDispatch();
                    [encoder setBytes:&state.contactActiveCapacity
                               length:sizeof(state.contactActiveCapacity)
                              atIndex:1u];
                    [encoder setBuffer:state.contactSamples offset:0u atIndex:2u];
                    [encoder setBuffer:state.contactActivePairs offset:0u atIndex:3u];
                    [encoder setBuffer:state.contactActiveSlotsByPair offset:0u atIndex:4u];
                    [encoder setBuffer:state.contactActiveCounts offset:0u atIndex:5u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:6u];
                });
                if (state.requiresArticulatedResponses) {
                    dispatchThreads(
                        "nm_contact_prepare_articulated_queries",
                        environments * state.contactActiveCapacity,
                        [&] {
                            setDispatch();
                            [encoder setBytes:&bridge length:sizeof(bridge) atIndex:1u];
                            [encoder setBytes:&request.articulationRootBody
                                       length:sizeof(request.articulationRootBody)
                                      atIndex:2u];
                            [encoder setBytes:&state.contactActiveCapacity
                                       length:sizeof(state.contactActiveCapacity)
                                      atIndex:3u];
                            [encoder setBuffer:state.contactActivePairs offset:0u atIndex:4u];
                            [encoder setBuffer:state.contactActiveCounts offset:0u atIndex:5u];
                            [encoder setBuffer:state.rigidProxies offset:0u atIndex:6u];
                            [encoder setBuffer:currentBodies offset:0u atIndex:7u];
                            [encoder setBuffer:state.contactPairs offset:0u atIndex:8u];
                            [encoder setBuffer:state.contactSamples offset:0u atIndex:9u];
                            [encoder setBuffer:state.articulatedPointQueries offset:0u atIndex:10u];
                            [encoder setBuffer:state.statuses offset:0u atIndex:11u];
                        }
                    );
                }
                dispatchThreads("nm_contact_gather_response",
                    environments * state.contactActiveCapacity, [&] {
                    setDispatch();
                    [encoder setBuffer:state.gridNodes offset:0u atIndex:1u];
                    [encoder setBuffer:state.femCandidate offset:0u atIndex:2u];
                    [encoder setBuffer:state.rigidProxies offset:0u atIndex:3u];
                    [encoder setBuffer:state.rigidStates offset:0u atIndex:4u];
                    [encoder setBuffer:state.contactPairs offset:0u atIndex:5u];
                    [encoder setBuffer:state.contactSamples offset:0u atIndex:6u];
                    [encoder setBuffer:state.contactResponseRanges offset:0u atIndex:7u];
                    [encoder setBuffer:state.contactResponseColumns offset:0u atIndex:8u];
                    [encoder setBuffer:state.contactResponseValues offset:0u atIndex:9u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:10u];
                    [encoder setBytes:&state.contactResponseEntryCount
                               length:sizeof(state.contactResponseEntryCount)
                              atIndex:11u];
                    [encoder setBytes:&state.contactActiveCapacity
                               length:sizeof(state.contactActiveCapacity)
                              atIndex:12u];
                    [encoder setBuffer:state.contactActivePairs offset:0u atIndex:13u];
                    [encoder setBuffer:state.contactActiveCounts offset:0u atIndex:14u];
                });
                if (state.requiresArticulatedResponses) {
                    [encoder endEncoding];
                    const ArticulatedResponseQuery articulatedQuery{
                        .pointQueries = (__bridge void*)state.articulatedPointQueries,
                        .pointWorld = (__bridge void*)state.articulatedPointWorld,
                        .pointJacobians = (__bridge void*)state.articulatedPointJacobians,
                        .rightHandSides = (__bridge void*)state.articulatedRightHandSides,
                        .responseColumns = (__bridge void*)state.articulatedResponseColumns,
                        .inverseMassStatuses = (__bridge void*)state.articulatedInverseStatuses,
                        .csrRows = (__bridge void*)state.contactResponseRows,
                        .csrColumns = (__bridge void*)state.contactResponseColumns,
                        .csrValues = (__bridge void*)state.contactResponseValues,
                        .pointCount = state.contactActiveCapacity,
                        .responseEntryCount = state.contactResponseEntryCount,
                        .generalizedVectorStride = MR_ARTICULATED_ABA_MAX_DOFS,
                        .inverseMassStatusStride = state.articulatedInverseStatusStride,
                    };
                    if (!request.encodeArticulatedResponses(
                            request.articulatedResponseContext,
                            articulatedQuery
                        )) {
                        diagnostics.message =
                            "MetalWorld failed to encode borrowed inverse-ABA contact responses";
                        return false;
                    }
                    encoder = [commandBuffer computeCommandEncoder];
                    if (encoder == nil) {
                        diagnostics.message =
                            "failed to resume Matter KKT solve after inverse ABA";
                        return false;
                    }
                    [encoder setLabel:@"Numi Matter monolithic KKT continuation"];
                    dispatchThreads(
                        "nm_contact_validate_articulated_response",
                        environments,
                        [&] {
                            setDispatch();
                            [encoder setBuffer:state.contactResponseValues offset:0u atIndex:1u];
                            [encoder setBuffer:state.articulatedInverseStatuses offset:0u atIndex:2u];
                            [encoder setBytes:&state.articulatedInverseStatusStride
                                       length:sizeof(state.articulatedInverseStatusStride)
                                      atIndex:3u];
                            [encoder setBytes:&state.contactResponseEntryCount
                                       length:sizeof(state.contactResponseEntryCount)
                                      atIndex:4u];
                            [encoder setBuffer:state.statuses offset:0u atIndex:5u];
                        }
                    );
                }
                dispatchThreads(
                    "nm_contact_solve_coupled",
                    environments * state.contactComponentCount,
                    [&] {
                        setDispatch();
                        [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                        [encoder setBuffer:state.objects offset:0u atIndex:2u];
                        [encoder setBuffer:state.materials offset:0u atIndex:3u];
                        [encoder setBuffer:state.rigidStates offset:0u atIndex:4u];
                        [encoder setBuffer:state.contactPairs offset:0u atIndex:5u];
                        [encoder setBuffer:state.schedulers offset:0u atIndex:6u];
                        [encoder setBuffer:state.contactSamples offset:0u atIndex:7u];
                        [encoder setBuffer:state.statuses offset:0u atIndex:8u];
                        [encoder setBuffer:state.contactResponseColumns offset:0u atIndex:9u];
                        [encoder setBuffer:state.contactResponseRanges offset:0u atIndex:10u];
                        [encoder setBuffer:state.contactResponseValues offset:0u atIndex:11u];
                        [encoder setBuffer:state.contactComponentIncidence offset:0u atIndex:12u];
                        [encoder setBuffer:state.contactComponentRanges offset:0u atIndex:13u];
                        [encoder setBytes:&state.contactResponseEntryCount
                                   length:sizeof(state.contactResponseEntryCount)
                                  atIndex:14u];
                        [encoder setBytes:&state.contactComponentCount
                                   length:sizeof(state.contactComponentCount)
                                  atIndex:15u];
                        [encoder setBuffer:state.contactWarmstartsCandidate
                                    offset:0u atIndex:16u];
                        [encoder setBytes:&state.contactActiveCapacity
                                   length:sizeof(state.contactActiveCapacity)
                                  atIndex:17u];
                        [encoder setBuffer:state.contactActivePairs offset:0u atIndex:18u];
                        [encoder setBuffer:state.contactActiveCounts offset:0u atIndex:19u];
                    }
                );
                if (certify) {
                    dispatchThreads("nm_contact_certify_natural_map", objectTotal, [&] {
                        setDispatch();
                        [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                        [encoder setBuffer:state.mixedSolver offset:0u atIndex:2u];
                        [encoder setBuffer:state.objects offset:0u atIndex:3u];
                        [encoder setBuffer:state.materials offset:0u atIndex:4u];
                        [encoder setBuffer:state.contactPairs offset:0u atIndex:5u];
                        [encoder setBuffer:state.schedulers offset:0u atIndex:6u];
                        [encoder setBuffer:state.contactSamples offset:0u atIndex:7u];
                        [encoder setBuffer:state.contactResponseRanges offset:0u atIndex:8u];
                        [encoder setBuffer:state.contactResponseColumns offset:0u atIndex:9u];
                        [encoder setBuffer:state.contactResponseValues offset:0u atIndex:10u];
                        [encoder setBuffer:state.femCandidate offset:0u atIndex:11u];
                        [encoder setBytes:&state.contactResponseEntryCount
                                   length:sizeof(state.contactResponseEntryCount) atIndex:12u];
                        [encoder setBuffer:state.solverCertificates offset:0u atIndex:13u];
                        [encoder setBuffer:state.statuses offset:0u atIndex:14u];
                        [encoder setBytes:&state.contactActiveCapacity
                                   length:sizeof(state.contactActiveCapacity)
                                  atIndex:15u];
                        [encoder setBuffer:state.contactActivePairs offset:0u atIndex:16u];
                        [encoder setBuffer:state.contactActiveCounts offset:0u atIndex:17u];
                    });
                }
                dispatchThreads("nm_contact_accumulate_fem_residual", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.contactNodeIncidence offset:0u atIndex:1u];
                    [encoder setBuffer:state.contactNodeRanges offset:0u atIndex:2u];
                    [encoder setBuffer:state.contactSamples offset:0u atIndex:3u];
                    [encoder setBuffer:state.femResidual offset:0u atIndex:4u];
                });
                dispatchThreads(
                    "nm_contact_accumulate_deformable_fem_residual",
                    femNodeTotal,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.deformableContacts
                                   offset:0u atIndex:1u];
                        [encoder setBuffer:state.deformableContactNodeIncidence
                                   offset:0u atIndex:2u];
                        [encoder setBuffer:state.deformableContactNodeRanges
                                   offset:0u atIndex:3u];
                        [encoder setBuffer:state.femResidual
                                   offset:0u atIndex:4u];
                    });
                return true;
            };
            for (std::uint32_t nonlinearIteration = 0u;
                 nonlinearIteration <
                    state.mixedSolverValue.nonlinearIterations.x;
                 ++nonlinearIteration) {
            micro.pcgIteration = nonlinearIteration;
            // Reassemble the backward-Euler field residual at this Newton
            // candidate. These kernels no longer iterate or publish a field
            // solution; they provide the field residual and block diagonal to
            // the single generalized KKT solve below.
            dispatchThreads("nm_mixed_prepare_residual", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.objects offset:0u atIndex:1u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                [encoder setBuffer:state.fieldBoundaries offset:0u atIndex:3u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:4u];
                [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:5u];
            });
            dispatchThreads("nm_mixed_apply_operator_elements", tetrahedronTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:3u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:4u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:5u];
                [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:6u];
                [encoder setBuffer:state.elementOperator offset:0u atIndex:7u];
                [encoder setBuffer:state.femDirection offset:0u atIndex:8u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:9u];
            });
            dispatchThreads("nm_mixed_apply_operator_nodes", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:1u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:3u];
                [encoder setBuffer:state.elementOperator offset:0u atIndex:4u];
                [encoder setBuffer:state.fieldBoundaries offset:0u atIndex:5u];
                [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:6u];
                [encoder setBuffer:state.mixedFieldOperator offset:0u atIndex:7u];
            });
            dispatchThreads("nm_mixed_build_residual_diagonal", femNodeTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:3u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:4u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:5u];
                [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:6u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:7u];
                [encoder setBuffer:state.fieldBoundaries offset:0u atIndex:8u];
                [encoder setBuffer:state.femFieldsAccepted offset:0u atIndex:9u];
                [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:10u];
                [encoder setBuffer:state.mixedFieldOperator offset:0u atIndex:11u];
                [encoder setBuffer:state.mixedFieldResidual offset:0u atIndex:12u];
                [encoder setBuffer:state.mixedFieldPreconditioned offset:0u atIndex:13u];
                [encoder setBuffer:state.mixedFieldInverseDiagonal offset:0u atIndex:14u];
            });
            // Mechanical pressure is a KKT unknown. Its normalized Schur
            // diagonal is applied only by nm_fgmres_precondition; there is no
            // standalone pressure correction or independently failing solve.
            dispatchThreads("nm_fem_internal_forces", tetrahedronTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.materials offset:0u atIndex:3u];
                [encoder setBuffer:state.scalarPrograms offset:0u atIndex:4u];
                [encoder setBuffer:state.instructions offset:0u atIndex:5u];
                [encoder setBuffer:state.environmentParameters offset:0u atIndex:6u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:7u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:8u];
                [encoder setBuffer:state.femMaterialStateAccepted offset:0u atIndex:9u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:10u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:11u];
                [encoder setBuffer:state.elementForces offset:0u atIndex:12u];
                [encoder setBuffer:state.statuses offset:0u atIndex:13u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:14u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:15u];
                [encoder setBuffer:state.learnedMaterials offset:0u atIndex:16u];
                [encoder setBuffer:state.learnedLayers offset:0u atIndex:17u];
                [encoder setBuffer:state.learnedWeightsCandidate offset:0u atIndex:18u];
            });
            dispatchThreads("nm_fem_build_mechanical_residual", femNodeTotal, [&] {
                const std::uint32_t preserveSolution = 0u;
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:3u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:4u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:5u];
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
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:17u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:18u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:19u];
                [encoder setBytes:&preserveSolution
                           length:sizeof(preserveSolution) atIndex:20u];
            });

            id<MTLBuffer> nonlinearWarmstarts = nonlinearIteration == 0u
                ? state.contactWarmstartsAccepted
                : state.contactWarmstartsCandidate;
            id<MTLBuffer> nonlinearDeformableWarmstarts =
                nonlinearIteration == 0u
                ? state.deformableWarmstartsAccepted
                : state.deformableWarmstartsCandidate;
            if (!encodeCoupledKKTContact(
                    false,
                    nonlinearWarmstarts,
                    nonlinearDeformableWarmstarts)) {
                [encoder endEncoding];
                ownership->preDynamicsOpen = false;
                return diagnostics;
            }

            dispatchThreads("nm_contact_build_kkt_residual",
                environments * state.contactActiveCapacity, [&] {
                setDispatch();
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.materials offset:0u atIndex:3u];
                [encoder setBuffer:state.contactPairs offset:0u atIndex:4u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:5u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:6u];
                [encoder setBuffer:state.contactSamples offset:0u atIndex:7u];
                [encoder setBuffer:state.contactResponseRanges offset:0u atIndex:8u];
                [encoder setBuffer:state.contactResponseColumns offset:0u atIndex:9u];
                [encoder setBuffer:state.contactResponseValues offset:0u atIndex:10u];
                [encoder setBytes:&state.contactResponseEntryCount
                           length:sizeof(state.contactResponseEntryCount)
                          atIndex:11u];
                [encoder setBuffer:state.femResidual offset:0u atIndex:12u];
                [encoder setBuffer:state.statuses offset:0u atIndex:13u];
                [encoder setBytes:&state.contactActiveCapacity
                           length:sizeof(state.contactActiveCapacity)
                          atIndex:14u];
                [encoder setBuffer:state.contactActivePairs offset:0u atIndex:15u];
                [encoder setBuffer:state.contactActiveCounts offset:0u atIndex:16u];
            });
            (void)dispatchIndirect(
                "nm_contact_build_deformable_kkt_residual",
                state.deformableContactActiveDispatch,
                256u,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.mixedSolver offset:0u atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.schedulers offset:0u atIndex:3u];
                    [encoder setBuffer:state.deformableContacts offset:0u atIndex:4u];
                    [encoder setBuffer:state.deformableContactGlobalActiveIndices offset:0u atIndex:5u];
                    [encoder setBuffer:state.deformableContactActiveDispatch offset:0u atIndex:6u];
                    [encoder setBuffer:state.femResidual offset:0u atIndex:7u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:8u];
                });
            dispatchThreads("nm_fgmres_import_field_residual", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.mixedFieldResidual offset:0u atIndex:1u];
                [encoder setBuffer:state.femResidual offset:0u atIndex:2u];
            });

            // Staged, GPU-resident FGMRES: expensive constitutive and contact
            // operator work is distributed over nodes/active contacts. The
            // SIMD32 object kernels only perform bounded reductions and the
            // tiny Hessenberg solve.
            const NSUInteger deformableContactTotal = environments *
                state.dispatch.deformableContactCapacity;
            const NSUInteger kktUnknownTotal =
                2u * femNodeTotal + pairTotal + deformableContactTotal;
            const NSUInteger activeContactTotal =
                environments * state.contactActiveCapacity;
            const NSUInteger vectorBytes =
                kktUnknownTotal * sizeof(nm_float4);
            [encoder useResource:state.contactPairs usage:MTLResourceUsageRead];
            [encoder useResource:state.contactSamples usage:MTLResourceUsageRead];
            [encoder useResource:state.contactResponseRanges usage:MTLResourceUsageRead];
            [encoder useResource:state.contactResponseColumns usage:MTLResourceUsageRead];
            [encoder useResource:state.contactResponseValues usage:MTLResourceUsageRead];
            [encoder useResource:state.contactNodeIncidence usage:MTLResourceUsageRead];
            [encoder useResource:state.contactNodeRanges usage:MTLResourceUsageRead];
            [encoder useResource:state.contactActivePairs usage:MTLResourceUsageRead];
            [encoder useResource:state.contactActiveSlotsByPair usage:MTLResourceUsageRead];
            [encoder useResource:state.contactActiveCounts usage:MTLResourceUsageRead];
            [encoder useResource:state.deformableContacts
                         usage:MTLResourceUsageRead | MTLResourceUsageWrite];
            [encoder useResource:state.deformableContactActiveIndices
                         usage:MTLResourceUsageRead];
            [encoder useResource:state.deformableContactActiveCounts
                         usage:MTLResourceUsageRead];
            [encoder useResource:state.deformableContactNodeIncidence
                         usage:MTLResourceUsageRead];
            [encoder useResource:state.deformableContactNodeRanges
                         usage:MTLResourceUsageRead];
            [encoder useResource:state.deformableContactGlobalActiveIndices
                         usage:MTLResourceUsageRead];
            [encoder useResource:state.deformableContactActiveDispatch
                         usage:MTLResourceUsageRead];
            const std::uint32_t restart = std::min(
                state.mixedSolverValue.nonlinearIterations.y,
                static_cast<std::uint32_t>(NM_MIXED_FGMRES_RESTART));
            const std::uint32_t linearIterationBudget = std::max(
                restart, state.mixedSolverValue.nonlinearIterations.z);
            const std::uint32_t restartCycleCount =
                (linearIterationBudget + restart - 1u) / restart;
            const float nonlinearTolerance = std::max(
                state.mixedSolverValue.nonlinearTolerances.x, 0.0f);
            const float forcingFloor = std::sqrt(nonlinearTolerance);
            const float scheduledForcing = std::ldexp(
                0.25f,
                -static_cast<int>(std::min(nonlinearIteration, 4u)));
            const float linearForcing = std::clamp(
                std::max(forcingFloor, scheduledForcing),
                nonlinearTolerance,
                0.25f);
            NMMicrostepGPU operatorMicro = micro;
            operatorMicro.flags |= NM_MICROSTEP_FGMRES_OPERATOR;
            for (std::uint32_t restartCycle = 0u;
                 restartCycle < restartCycleCount;
                 ++restartCycle) {
            const std::uint32_t iterationOffset = restartCycle * restart;
            const std::uint32_t columnsThisCycle = std::min(
                restart, linearIterationBudget - iterationOffset);
            dispatchGroups32("nm_fgmres_begin", environments, [&] {
                setDispatch();
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.femResidual offset:0u atIndex:3u];
                [encoder setBuffer:state.fgmresBasis offset:0u atIndex:4u];
                [encoder setBuffer:state.fgmresHessenberg offset:0u atIndex:5u];
                [encoder setBuffer:state.fgmresRotations offset:0u atIndex:6u];
                [encoder setBuffer:state.fgmresLeastSquares offset:0u atIndex:7u];
                [encoder setBuffer:state.fgmresStates offset:0u atIndex:8u];
                [encoder setBuffer:state.statuses offset:0u atIndex:9u];
                [encoder setBytes:&micro length:sizeof(micro) atIndex:10u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:11u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:12u];
                [encoder setBuffer:state.fgmresContactArguments offset:0u atIndex:13u];
                [encoder setBytes:&restartCycle
                           length:sizeof(restartCycle) atIndex:14u];
                [encoder setBytes:&linearForcing
                           length:sizeof(linearForcing) atIndex:15u];
                [encoder setBytes:&nonlinearIteration
                           length:sizeof(nonlinearIteration) atIndex:16u];
            });
            dispatchThreads("nm_fgmres_build_preconditioner", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:3u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:4u];
                [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:5u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:6u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:7u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:8u];
                [encoder setBuffer:state.femPreconditioned offset:0u atIndex:9u];
            });
            for (std::uint32_t column = 0u; column < columnsThisCycle; ++column) {
                const NSUInteger columnOffset = vectorBytes * column;
                dispatchThreads("nm_fgmres_precondition", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:1u];
                    [encoder setBuffer:state.femPreconditioned offset:0u atIndex:2u];
                    [encoder setBuffer:state.fgmresBasis offset:columnOffset atIndex:3u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis offset:columnOffset atIndex:4u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:5u];
                    [encoder setBuffer:state.mixedSolver offset:0u atIndex:6u];
                    [encoder setBuffer:state.objects offset:0u atIndex:7u];
                });
                dispatchThreads("nm_fgmres_precondition_patches", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.femTetrahedraCandidate
                                 offset:0u atIndex:1u];
                    [encoder setBuffer:state.femNodeIncidence
                                 offset:0u atIndex:2u];
                    [encoder setBuffer:state.femNodeRanges
                                 offset:0u atIndex:3u];
                    [encoder setBuffer:state.fgmresBasis
                                 offset:columnOffset atIndex:4u];
                    [encoder setBuffer:state.femPreconditioned
                                 offset:0u atIndex:5u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis
                                 offset:columnOffset atIndex:6u];
                    [encoder setBuffer:state.fgmresStates
                                 offset:0u atIndex:7u];
                });
                dispatchGroups32("nm_fgmres_precondition_coarse", objectTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.objects offset:0u atIndex:1u];
                    [encoder setBuffer:state.fgmresBasis
                                 offset:columnOffset atIndex:2u];
                    [encoder setBuffer:state.femPreconditioned
                                 offset:0u atIndex:3u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis
                                 offset:columnOffset atIndex:4u];
                    [encoder setBuffer:state.fgmresStates
                                 offset:0u atIndex:5u];
                });
                dispatchThreads("nm_fgmres_field_smoother_initialize", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:1u];
                    [encoder setBuffer:state.fgmresBasis offset:columnOffset atIndex:2u];
                    [encoder setBuffer:state.mixedFieldInverseDiagonal offset:0u atIndex:3u];
                    [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:4u];
                    [encoder setBuffer:state.mixedFieldResidual offset:0u atIndex:5u];
                    [encoder setBuffer:state.mixedFieldPreconditioned offset:0u atIndex:6u];
                    [encoder setBuffer:state.mixedFieldDirection offset:0u atIndex:7u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:8u];
                });
                // Apply a fixed matrix-free polynomial smoother to the four
                // packed transport blocks. The former nested PCG launched up
                // to seven kernels per inner iteration for every Arnoldi
                // column. This bounded right preconditioner keeps the outer
                // FGMRES authoritative while removing reductions, scalar
                // dependencies and most command traffic from the hot loop.
                NMMicrostepGPU fieldPreconditionerMicro = micro;
                fieldPreconditionerMicro.flags |=
                    NM_MICROSTEP_FIELD_PRECONDITIONER;
                constexpr std::array<float, 3> fieldSmootherDamping{
                    0.72f, 0.86f, 0.72f
                };
                const std::uint32_t fieldSmoothingPasses = std::min(
                    state.mixedSolverValue.blockIterations.z,
                    static_cast<std::uint32_t>(fieldSmootherDamping.size()));
                for (std::uint32_t fieldPass = 0u;
                     fieldPass < fieldSmoothingPasses; ++fieldPass) {
                    fieldPreconditionerMicro.pcgIteration = fieldPass;
                    dispatchThreads("nm_mixed_apply_operator_elements", tetrahedronTotal, [&] {
                        setDispatch();
                        [encoder setBytes:&fieldPreconditionerMicro
                                   length:sizeof(fieldPreconditionerMicro) atIndex:1u];
                        [encoder setBuffer:state.objects offset:0u atIndex:2u];
                        [encoder setBuffer:state.mixedMaterials offset:0u atIndex:3u];
                        [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:4u];
                        [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:5u];
                        [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:6u];
                        [encoder setBuffer:state.elementOperator offset:0u atIndex:7u];
                        [encoder setBuffer:state.femDirection offset:0u atIndex:8u];
                        [encoder setBuffer:state.femCandidate offset:0u atIndex:9u];
                    });
                    dispatchThreads("nm_mixed_apply_operator_nodes", femNodeTotal, [&] {
                        setDispatch();
                        [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:1u];
                        [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                        [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:3u];
                        [encoder setBuffer:state.elementOperator offset:0u atIndex:4u];
                        [encoder setBuffer:state.fieldBoundaries offset:0u atIndex:5u];
                        [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:6u];
                        [encoder setBuffer:state.mixedFieldOperator offset:0u atIndex:7u];
                    });
                    dispatchThreads("nm_fgmres_field_smooth", femNodeTotal, [&] {
                        setDispatch();
                        [encoder setBuffer:state.femNodeRanges offset:0u atIndex:1u];
                        [encoder setBuffer:state.fgmresBasis
                                     offset:columnOffset atIndex:2u];
                        [encoder setBuffer:state.mixedFieldInverseDiagonal
                                     offset:0u atIndex:3u];
                        [encoder setBuffer:state.mixedFieldOperator
                                     offset:0u atIndex:4u];
                        [encoder setBuffer:state.mixedFieldSolution
                                     offset:0u atIndex:5u];
                        [encoder setBuffer:state.fgmresStates
                                     offset:0u atIndex:6u];
                        const float damping = fieldSmootherDamping[fieldPass];
                        [encoder setBytes:&damping length:sizeof(damping)
                                   atIndex:7u];
                    });
                }
                dispatchThreads("nm_fgmres_field_smoother_export", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:1u];
                    [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:2u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis
                                 offset:columnOffset atIndex:3u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:4u];
                });
                dispatchThreads("nm_fgmres_precondition_cross", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.mixedMaterials offset:0u atIndex:3u];
                    [encoder setBuffer:state.femPreconditioned offset:0u atIndex:4u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis
                                 offset:columnOffset atIndex:5u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:6u];
                });
                dispatchThreads("nm_fgmres_precondition_contacts", activeContactTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.objects offset:0u atIndex:1u];
                    [encoder setBuffer:state.fgmresBasis offset:columnOffset atIndex:2u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis offset:columnOffset atIndex:3u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:4u];
                    [encoder setBuffer:state.fgmresContactArguments offset:0u atIndex:5u];
                });
                (void)dispatchIndirect(
                    "nm_fgmres_precondition_deformable_contacts",
                    state.deformableContactActiveDispatch,
                    256u,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.fgmresBasis
                                     offset:columnOffset atIndex:1u];
                        [encoder setBuffer:state.fgmresPreconditionedBasis
                                     offset:columnOffset atIndex:2u];
                        [encoder setBuffer:state.fgmresStates
                                     offset:0u atIndex:3u];
                        [encoder setBuffer:state.fgmresContactArguments
                                     offset:0u atIndex:4u];
                    });
                dispatchThreads(
                    "nm_fgmres_precondition_contact_cross",
                    femNodeTotal,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.femPreconditioned
                                     offset:0u atIndex:1u];
                        [encoder setBuffer:state.fgmresPreconditionedBasis
                                     offset:columnOffset atIndex:2u];
                        [encoder setBuffer:state.fgmresContactArguments
                                     offset:0u atIndex:3u];
                        [encoder setBuffer:state.fgmresStates
                                     offset:0u atIndex:4u];
                    });
                dispatchThreads("nm_fem_apply_operator_elements", tetrahedronTotal, [&] {
                    setDispatch();
                    [encoder setBytes:&operatorMicro length:sizeof(operatorMicro) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.materials offset:0u atIndex:3u];
                    [encoder setBuffer:state.scalarPrograms offset:0u atIndex:4u];
                    [encoder setBuffer:state.instructions offset:0u atIndex:5u];
                    [encoder setBuffer:state.environmentParameters offset:0u atIndex:6u];
                    [encoder setBuffer:state.femCandidate offset:0u atIndex:7u];
                    [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:8u];
                    [encoder setBuffer:state.femMaterialStateAccepted offset:0u atIndex:9u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis offset:columnOffset atIndex:10u];
                    [encoder setBuffer:state.schedulers offset:0u atIndex:11u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:12u];
                    [encoder setBuffer:state.elementOperator offset:0u atIndex:13u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:14u];
                    [encoder setBuffer:state.pcgScalars offset:0u atIndex:15u];
                    [encoder setBuffer:state.mixedMaterials offset:0u atIndex:16u];
                    [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:17u];
                    [encoder setBuffer:state.learnedMaterials offset:0u atIndex:18u];
                    [encoder setBuffer:state.learnedLayers offset:0u atIndex:19u];
                    [encoder setBuffer:state.learnedWeightsCandidate offset:0u atIndex:20u];
                    [encoder setBuffer:state.mixedSolver offset:0u atIndex:21u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:22u];
                });
                dispatchThreads("nm_fgmres_gather_nodes", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBytes:&operatorMicro length:sizeof(operatorMicro) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.femCandidate offset:0u atIndex:3u];
                    [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:4u];
                    [encoder setBuffer:state.elementOperator offset:0u atIndex:5u];
                    [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:6u];
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:7u];
                    [encoder setBuffer:state.schedulers offset:0u atIndex:8u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:9u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis offset:columnOffset atIndex:10u];
                    [encoder setBuffer:state.femOperatorValue offset:0u atIndex:11u];
                    [encoder setBuffer:state.fgmresContactArguments offset:0u atIndex:12u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:13u];
                });
                const NSUInteger fieldVectorOffset =
                    columnOffset + femNodeTotal * sizeof(nm_float4);
                const NSUInteger fieldWorkOffset =
                    femNodeTotal * sizeof(nm_float4);
                dispatchThreads("nm_mixed_apply_operator_elements", tetrahedronTotal, [&] {
                    setDispatch();
                    [encoder setBytes:&operatorMicro length:sizeof(operatorMicro) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.mixedMaterials offset:0u atIndex:3u];
                    [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:4u];
                    [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:5u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis
                                 offset:fieldVectorOffset atIndex:6u];
                    [encoder setBuffer:state.elementOperator offset:0u atIndex:7u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis
                                 offset:columnOffset atIndex:8u];
                    [encoder setBuffer:state.femCandidate offset:0u atIndex:9u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:10u];
                });
                dispatchThreads("nm_mixed_apply_operator_nodes", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:1u];
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                    [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:3u];
                    [encoder setBuffer:state.elementOperator offset:0u atIndex:4u];
                    [encoder setBuffer:state.fieldBoundaries offset:0u atIndex:5u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis
                                 offset:fieldVectorOffset atIndex:6u];
                    [encoder setBuffer:state.femOperatorValue
                                 offset:fieldWorkOffset atIndex:7u];
                });
                dispatchThreads("nm_fgmres_apply_contacts", activeContactTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.mixedSolver offset:0u atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.materials offset:0u atIndex:3u];
                    [encoder setBuffer:state.schedulers offset:0u atIndex:4u];
                    [encoder setBuffer:state.femCandidate offset:0u atIndex:5u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis offset:columnOffset atIndex:6u];
                    [encoder setBuffer:state.femOperatorValue offset:0u atIndex:7u];
                    [encoder setBuffer:state.fgmresContactArguments offset:0u atIndex:8u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:9u];
                });
                (void)dispatchIndirect(
                    "nm_fgmres_apply_deformable_contacts",
                    state.deformableContactActiveDispatch,
                    256u,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.mixedSolver offset:0u atIndex:1u];
                        [encoder setBuffer:state.schedulers offset:0u atIndex:2u];
                        [encoder setBuffer:state.fgmresPreconditionedBasis
                                     offset:columnOffset atIndex:3u];
                        [encoder setBuffer:state.femOperatorValue
                                     offset:0u atIndex:4u];
                        [encoder setBuffer:state.fgmresContactArguments
                                     offset:0u atIndex:5u];
                        [encoder setBuffer:state.fgmresStates
                                     offset:0u atIndex:6u];
                    });
                dispatchGroups32("nm_fgmres_orthogonalize_column", environments, [&] {
                    setDispatch();
                    [encoder setBytes:&column length:sizeof(column) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.fgmresBasis offset:0u atIndex:3u];
                    [encoder setBuffer:state.femOperatorValue offset:0u atIndex:4u];
                    [encoder setBuffer:state.fgmresHessenberg offset:0u atIndex:5u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:6u];
                    [encoder setBuffer:state.fgmresContactArguments offset:0u atIndex:7u];
                });
                dispatchGroups32("nm_fgmres_finish_column", environments, [&] {
                    setDispatch();
                    [encoder setBuffer:state.mixedSolver offset:0u atIndex:1u];
                    [encoder setBytes:&column length:sizeof(column) atIndex:2u];
                    [encoder setBuffer:state.objects offset:0u atIndex:3u];
                    [encoder setBuffer:state.femOperatorValue offset:0u atIndex:4u];
                    [encoder setBuffer:state.fgmresBasis offset:vectorBytes * (column + 1u) atIndex:5u];
                    [encoder setBuffer:state.fgmresHessenberg offset:0u atIndex:6u];
                    [encoder setBuffer:state.fgmresRotations offset:0u atIndex:7u];
                    [encoder setBuffer:state.fgmresLeastSquares offset:0u atIndex:8u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:9u];
                    [encoder setBuffer:state.fgmresContactArguments offset:0u atIndex:10u];
                    [encoder setBytes:&linearForcing
                               length:sizeof(linearForcing) atIndex:11u];
                });
            }
            dispatchThreads("nm_fgmres_form_restart_coefficients", environments, [&] {
                setDispatch();
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:1u];
                [encoder setBuffer:state.fgmresRotations offset:0u atIndex:2u];
                [encoder setBuffer:state.fgmresLeastSquares offset:0u atIndex:3u];
                [encoder setBuffer:state.fgmresStates offset:0u atIndex:4u];
                [encoder setBuffer:state.fgmresRestartCoefficients
                             offset:0u atIndex:5u];
            });
            const std::uint32_t finalRestartCycle =
                restartCycle + 1u == restartCycleCount ? 1u : 0u;
            dispatchThreads("nm_fgmres_backsolve", environments, [&] {
                setDispatch();
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:1u];
                [encoder setBuffer:state.fgmresHessenberg offset:0u atIndex:2u];
                [encoder setBuffer:state.fgmresLeastSquares offset:0u atIndex:3u];
                [encoder setBuffer:state.fgmresStates offset:0u atIndex:4u];
                [encoder setBuffer:state.statuses offset:0u atIndex:5u];
                [encoder setBuffer:state.pcgScalars offset:0u atIndex:6u];
                [encoder setBytes:&iterationOffset
                           length:sizeof(iterationOffset) atIndex:7u];
                [encoder setBytes:&finalRestartCycle
                           length:sizeof(finalRestartCycle) atIndex:8u];
                [encoder setBytes:&linearForcing
                           length:sizeof(linearForcing) atIndex:9u];
            });
            dispatchThreads("nm_fgmres_accumulate", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:1u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                [encoder setBuffer:state.fgmresPreconditionedBasis offset:0u atIndex:3u];
                [encoder setBuffer:state.fgmresLeastSquares offset:0u atIndex:4u];
                [encoder setBuffer:state.fgmresStates offset:0u atIndex:5u];
                [encoder setBuffer:state.femSolution offset:0u atIndex:6u];
                [encoder setBytes:&restartCycle
                           length:sizeof(restartCycle) atIndex:7u];
            });
            dispatchThreads("nm_fgmres_accumulate_fields", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:1u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                [encoder setBuffer:state.fgmresPreconditionedBasis offset:0u atIndex:3u];
                [encoder setBuffer:state.fgmresLeastSquares offset:0u atIndex:4u];
                [encoder setBuffer:state.fgmresStates offset:0u atIndex:5u];
                [encoder setBuffer:state.femSolution offset:0u atIndex:6u];
                [encoder setBytes:&restartCycle
                           length:sizeof(restartCycle) atIndex:7u];
            });
            dispatchThreads("nm_fgmres_accumulate_contacts", activeContactTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:1u];
                [encoder setBuffer:state.fgmresPreconditionedBasis offset:0u atIndex:2u];
                [encoder setBuffer:state.fgmresLeastSquares offset:0u atIndex:3u];
                [encoder setBuffer:state.fgmresStates offset:0u atIndex:4u];
                [encoder setBuffer:state.femSolution offset:0u atIndex:5u];
                [encoder setBuffer:state.fgmresContactArguments offset:0u atIndex:6u];
                [encoder setBytes:&restartCycle
                           length:sizeof(restartCycle) atIndex:7u];
            });
            (void)dispatchIndirect(
                "nm_fgmres_accumulate_deformable_contacts",
                state.deformableContactActiveDispatch,
                256u,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.mixedSolver offset:0u atIndex:1u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis offset:0u atIndex:2u];
                    [encoder setBuffer:state.fgmresLeastSquares offset:0u atIndex:3u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:4u];
                    [encoder setBuffer:state.femSolution offset:0u atIndex:5u];
                    [encoder setBuffer:state.fgmresContactArguments offset:0u atIndex:6u];
                    [encoder setBytes:&restartCycle
                               length:sizeof(restartCycle) atIndex:7u];
                });
            if (finalRestartCycle == 0u) {
                dispatchThreads("nm_fgmres_restart_residual_nodes", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:1u];
                    [encoder setBuffer:state.fgmresBasis offset:0u atIndex:2u];
                    [encoder setBuffer:state.fgmresRestartCoefficients
                                 offset:0u atIndex:3u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:4u];
                    [encoder setBuffer:state.femResidual offset:0u atIndex:5u];
                });
                dispatchThreads("nm_fgmres_restart_residual_contacts",
                    activeContactTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.fgmresBasis offset:0u atIndex:1u];
                    [encoder setBuffer:state.fgmresRestartCoefficients
                                 offset:0u atIndex:2u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:3u];
                    [encoder setBuffer:state.femResidual offset:0u atIndex:4u];
                    [encoder setBuffer:state.fgmresContactArguments
                                 offset:0u atIndex:5u];
                });
                (void)dispatchIndirect(
                    "nm_fgmres_restart_residual_deformable_contacts",
                    state.deformableContactActiveDispatch,
                    256u,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.fgmresBasis offset:0u atIndex:1u];
                        [encoder setBuffer:state.fgmresRestartCoefficients
                                     offset:0u atIndex:2u];
                        [encoder setBuffer:state.fgmresStates offset:0u atIndex:3u];
                        [encoder setBuffer:state.femResidual offset:0u atIndex:4u];
                        [encoder setBuffer:state.fgmresContactArguments
                                     offset:0u atIndex:5u];
                    });
            }
            }

            dispatchGroups32("nm_fgmres_measure_correction", environments, [&] {
                setDispatch();
                [encoder setBuffer:state.objects offset:0u atIndex:1u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:2u];
                [encoder setBuffer:state.femSolution offset:0u atIndex:3u];
                [encoder setBuffer:state.fgmresStates offset:0u atIndex:4u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:5u];
                [encoder setBuffer:state.fgmresContactArguments offset:0u atIndex:6u];
            });

            micro.pcgIteration = nonlinearIteration;
            dispatchThreads("nm_fem_select_backtracking", objectTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:2u];
                [encoder setBuffer:state.objects offset:0u atIndex:3u];
                [encoder setBuffer:state.materials offset:0u atIndex:4u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:5u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:6u];
                [encoder setBuffer:state.femSolution offset:0u atIndex:7u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:8u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:9u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:10u];
                [encoder setBuffer:state.femLineSearch offset:0u atIndex:11u];
                [encoder setBuffer:state.statuses offset:0u atIndex:12u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:13u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:14u];
            });
            dispatchGroups32(
                "nm_fem_synchronize_environment_line_search",
                environments,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.femLineSearch
                                 offset:0u atIndex:1u];
                });
            dispatchGroups32(
                "nm_contact_limit_deformable_line_search",
                environments,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.objects offset:0u atIndex:1u];
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                    [encoder setBuffer:state.schedulers offset:0u atIndex:3u];
                    [encoder setBuffer:state.deformableContacts offset:0u atIndex:4u];
                    [encoder setBuffer:state.deformableContactActiveIndices offset:0u atIndex:5u];
                    [encoder setBuffer:state.deformableContactActiveCounts offset:0u atIndex:6u];
                    [encoder setBuffer:state.femCandidate offset:0u atIndex:7u];
                    [encoder setBuffer:state.femSolution offset:0u atIndex:8u];
                    [encoder setBuffer:state.femLineSearch offset:0u atIndex:9u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:10u];
                });
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
                [encoder setBuffer:state.femLineSearch offset:0u atIndex:9u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:10u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:11u];
            });
            dispatchThreads("nm_mixed_apply_kkt_solution", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:1u];
                [encoder setBuffer:state.femSolution offset:0u atIndex:2u];
                [encoder setBuffer:state.femLineSearch offset:0u atIndex:3u];
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:4u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:5u];
                [encoder setBuffer:state.statuses offset:0u atIndex:6u];
            });
            dispatchThreads("nm_contact_apply_kkt_solution", pairTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.objects offset:0u atIndex:1u];
                [encoder setBuffer:state.materials offset:0u atIndex:2u];
                [encoder setBuffer:state.contactPairs offset:0u atIndex:3u];
                [encoder setBuffer:state.contactSamples offset:0u atIndex:4u];
                [encoder setBuffer:state.femSolution offset:0u atIndex:5u];
                [encoder setBuffer:state.femLineSearch offset:0u atIndex:6u];
                [encoder setBuffer:state.contactWarmstartsCandidate
                             offset:0u atIndex:7u];
            });
            (void)dispatchIndirect(
                "nm_contact_apply_deformable_kkt_solution",
                state.deformableContactActiveDispatch,
                256u,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.deformableContacts offset:0u atIndex:1u];
                    [encoder setBuffer:state.deformableContactGlobalActiveIndices offset:0u atIndex:2u];
                    [encoder setBuffer:state.deformableContactActiveDispatch offset:0u atIndex:3u];
                    [encoder setBuffer:state.femSolution offset:0u atIndex:4u];
                    [encoder setBuffer:state.femLineSearch offset:0u atIndex:5u];
                    [encoder setBuffer:state.deformableWarmstartsCandidate
                                 offset:0u atIndex:6u];
                });
            // Physics-triggered puncture is an active-set change of this same
            // nonlinear transaction. Apply it to candidate topology/state,
            // then let the following Newton iteration rebuild contact and the
            // KKT operator. No intermediate commit is permitted.
            if (nonlinearIteration <
                state.mixedSolverValue.blockIterations.w) {
                dispatchThreads("nm_topology_detect_puncture", objectTotal, [&] {
                    setDispatch();
                    [encoder setBytes:&request.controlStep
                               length:sizeof(request.controlStep) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.femCapacities offset:0u atIndex:3u];
                    [encoder setBuffer:state.rigidProxies offset:0u atIndex:4u];
                    [encoder setBuffer:state.rigidStates offset:0u atIndex:5u];
                    [encoder setBuffer:state.femCandidate offset:0u atIndex:6u];
                    [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:7u];
                    [encoder setBuffer:state.punctureChannelsCandidate offset:0u atIndex:8u];
                    [encoder setBuffer:state.contactPairs offset:0u atIndex:9u];
                    [encoder setBuffer:state.contactWarmstartsCandidate offset:0u atIndex:10u];
                    [encoder setBuffer:state.triggeredMutationCommands offset:0u atIndex:11u];
                });
                const std::uint32_t triggeredMutationCount =
                    state.dispatch.objectCount;
                const std::uint32_t zeroMutationCount = 0u;
                dispatchThreads("nm_topology_execute_transaction", environments, [&] {
                    setDispatch();
                    [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.femCapacities offset:0u atIndex:3u];
                    [encoder setBuffer:state.materials offset:0u atIndex:4u];
                    [encoder setBuffer:state.cookedMutationCommands offset:0u atIndex:5u];
                    [encoder setBuffer:borrowedMutations offset:0u atIndex:6u];
                    [encoder setBytes:&zeroMutationCount
                               length:sizeof(zeroMutationCount) atIndex:7u];
                    [encoder setBytes:&request.mutationCommandStride
                               length:sizeof(request.mutationCommandStride) atIndex:8u];
                    [encoder setBuffer:state.femCandidate offset:0u atIndex:9u];
                    [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:10u];
                    [encoder setBuffer:state.femTopologyNodesCandidate offset:0u atIndex:11u];
                    [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:12u];
                    [encoder setBuffer:state.cohesiveFacesCandidate offset:0u atIndex:13u];
                    [encoder setBuffer:state.punctureChannelsCandidate offset:0u atIndex:14u];
                    [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:15u];
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:16u];
                    [encoder setBuffer:state.topologyStatesCandidate offset:0u atIndex:17u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:18u];
                    [encoder setBuffer:state.triggeredMutationCommands offset:0u atIndex:19u];
                    [encoder setBytes:&triggeredMutationCount
                               length:sizeof(triggeredMutationCount) atIndex:20u];
                    [encoder setBytes:&zeroMutationCount
                               length:sizeof(zeroMutationCount) atIndex:21u];
                });
                encodeTopologyRebuild(state.femCandidate);
            }
            }
            // Reassemble the accepted Newton candidate once, then certify the
            // same generalized residual. This is a certificate pass only: no
            // post-contact FEM correction or second linear solve is encoded.
            dispatchThreads("nm_mixed_prepare_residual", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.objects offset:0u atIndex:1u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                [encoder setBuffer:state.fieldBoundaries offset:0u atIndex:3u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:4u];
                [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:5u];
            });
            dispatchThreads("nm_mixed_apply_operator_elements", tetrahedronTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:3u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:4u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:5u];
                [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:6u];
                [encoder setBuffer:state.elementOperator offset:0u atIndex:7u];
                [encoder setBuffer:state.femDirection offset:0u atIndex:8u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:9u];
            });
            dispatchThreads("nm_mixed_apply_operator_nodes", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:1u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:3u];
                [encoder setBuffer:state.elementOperator offset:0u atIndex:4u];
                [encoder setBuffer:state.fieldBoundaries offset:0u atIndex:5u];
                [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:6u];
                [encoder setBuffer:state.mixedFieldOperator offset:0u atIndex:7u];
            });
            dispatchThreads("nm_mixed_build_residual_diagonal", femNodeTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:3u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:4u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:5u];
                [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:6u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:7u];
                [encoder setBuffer:state.fieldBoundaries offset:0u atIndex:8u];
                [encoder setBuffer:state.femFieldsAccepted offset:0u atIndex:9u];
                [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:10u];
                [encoder setBuffer:state.mixedFieldOperator offset:0u atIndex:11u];
                [encoder setBuffer:state.mixedFieldResidual offset:0u atIndex:12u];
                [encoder setBuffer:state.mixedFieldPreconditioned offset:0u atIndex:13u];
                [encoder setBuffer:state.mixedFieldInverseDiagonal offset:0u atIndex:14u];
            });
            dispatchThreads("nm_fem_internal_forces", tetrahedronTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.materials offset:0u atIndex:3u];
                [encoder setBuffer:state.scalarPrograms offset:0u atIndex:4u];
                [encoder setBuffer:state.instructions offset:0u atIndex:5u];
                [encoder setBuffer:state.environmentParameters offset:0u atIndex:6u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:7u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:8u];
                [encoder setBuffer:state.femMaterialStateAccepted offset:0u atIndex:9u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:10u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:11u];
                [encoder setBuffer:state.elementForces offset:0u atIndex:12u];
                [encoder setBuffer:state.statuses offset:0u atIndex:13u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:14u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:15u];
                [encoder setBuffer:state.learnedMaterials offset:0u atIndex:16u];
                [encoder setBuffer:state.learnedLayers offset:0u atIndex:17u];
                [encoder setBuffer:state.learnedWeightsCandidate offset:0u atIndex:18u];
            });
            dispatchThreads("nm_fem_build_mechanical_residual", femNodeTotal, [&] {
                const std::uint32_t preserveSolution = 1u;
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:3u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:4u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:5u];
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
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:17u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:18u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:19u];
                [encoder setBytes:&preserveSolution
                           length:sizeof(preserveSolution) atIndex:20u];
            });
            if (!encodeCoupledKKTContact(
                    true,
                    state.contactWarmstartsCandidate,
                    state.deformableWarmstartsCandidate
                )) {
                [encoder endEncoding];
                ownership->preDynamicsOpen = false;
                return diagnostics;
            }
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
                [encoder setBuffer:state.particleMaterialStateAccepted offset:0u atIndex:9u];
                [encoder setBuffer:state.particleMaterialStateCandidate offset:0u atIndex:10u];
                [encoder setBuffer:state.gridNodes offset:0u atIndex:11u];
                [encoder setBuffer:state.mpmGrids offset:0u atIndex:12u];
                [encoder setBuffer:state.mpmNodeGenerations offset:0u atIndex:13u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:14u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:15u];
                [encoder setBuffer:state.statuses offset:0u atIndex:16u];
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
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.materials offset:0u atIndex:3u];
                [encoder setBuffer:state.scalarPrograms offset:0u atIndex:4u];
                [encoder setBuffer:state.instructions offset:0u atIndex:5u];
                [encoder setBuffer:state.environmentParameters offset:0u atIndex:6u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:7u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:8u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:9u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:10u];
                [encoder setBuffer:state.femMaterialStateAccepted offset:0u atIndex:11u];
                [encoder setBuffer:state.femMaterialStateCandidate offset:0u atIndex:12u];
                [encoder setBuffer:state.statuses offset:0u atIndex:13u];
            });
            dispatchGroups32("nm_mixed_certify", objectTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:3u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:4u];
                [encoder setBuffer:state.femFieldsAccepted offset:0u atIndex:5u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:6u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:7u];
                [encoder setBuffer:state.solverCertificates offset:0u atIndex:8u];
                [encoder setBuffer:state.statuses offset:0u atIndex:9u];
                [encoder setBuffer:state.femResidual offset:0u atIndex:10u];
                [encoder setBuffer:state.fgmresStates offset:0u atIndex:11u];
                [encoder setBuffer:state.mixedFieldResidual offset:0u atIndex:12u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:13u];
                [encoder setBuffer:state.femSolution offset:0u atIndex:14u];
            });
            dispatchThreads("nm_mpm_commit_microstep", particleTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.particleAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.particleCandidate offset:0u atIndex:3u];
                [encoder setBuffer:state.particleMaterialStateAccepted offset:0u atIndex:4u];
                [encoder setBuffer:state.particleMaterialStateCandidate offset:0u atIndex:5u];
            });
            dispatchThreads("nm_fem_commit_microstep", femTransactionalTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:3u];
                [encoder setBuffer:state.femMaterialStateAccepted offset:0u atIndex:4u];
                [encoder setBuffer:state.femMaterialStateCandidate offset:0u atIndex:5u];
            });
            dispatchThreads("nm_mixed_commit", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.femFieldsAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:3u];
            });
            dispatchThreads("nm_learned_commit", state.dispatch.learnedWeightCount, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.learnedWeightsAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.learnedWeightsCandidate offset:0u atIndex:3u];
                [encoder setBuffer:state.learnedRevisionAccepted offset:0u atIndex:4u];
                [encoder setBuffer:state.learnedRevisionCandidate offset:0u atIndex:5u];
            });
            dispatchThreads("nm_contact_commit_warmstarts", pairTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.contactWarmstartsAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.contactWarmstartsCandidate offset:0u atIndex:3u];
            });
            dispatchThreads(
                "nm_contact_commit_deformable_warmstarts",
                deformableWarmstartTotal,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                    [encoder setBuffer:state.deformableWarmstartsAccepted
                                 offset:0u atIndex:2u];
                    [encoder setBuffer:state.deformableWarmstartsCandidate
                                 offset:0u atIndex:3u];
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
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:9u];
                [encoder setBuffer:state.statuses offset:0u atIndex:10u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:11u];
            });
            dispatchThreads("nm_complete_microstep", environments, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
            });
        }

        dispatchThreads("nm_topology_commit", topologyTransactionalTotal, [&] {
            setDispatch();
            [encoder setBuffer:state.statuses offset:0u atIndex:1u];
            [encoder setBuffer:state.femTetrahedraAccepted offset:0u atIndex:2u];
            [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:3u];
            [encoder setBuffer:state.femTopologyNodesAccepted offset:0u atIndex:4u];
            [encoder setBuffer:state.femTopologyNodesCandidate offset:0u atIndex:5u];
            [encoder setBuffer:state.cohesiveFacesAccepted offset:0u atIndex:6u];
            [encoder setBuffer:state.cohesiveFacesCandidate offset:0u atIndex:7u];
            [encoder setBuffer:state.punctureChannelsAccepted offset:0u atIndex:8u];
            [encoder setBuffer:state.punctureChannelsCandidate offset:0u atIndex:9u];
            [encoder setBuffer:state.topologyStatesAccepted offset:0u atIndex:10u];
            [encoder setBuffer:state.topologyStatesCandidate offset:0u atIndex:11u];
        });

        dispatchThreads("nm_mpm_rollback_frame", particleTotal, [&] {
            setDispatch();
            [encoder setBuffer:state.statuses offset:0u atIndex:1u];
            [encoder setBuffer:state.particleAccepted offset:0u atIndex:2u];
            [encoder setBuffer:state.particleCheckpoint offset:0u atIndex:3u];
            [encoder setBuffer:state.particleMaterialStateAccepted offset:0u atIndex:4u];
            [encoder setBuffer:state.particleMaterialStateCheckpoint offset:0u atIndex:5u];
        });
        dispatchThreads("nm_fem_rollback_frame", femTransactionalTotal, [&] {
            setDispatch();
            [encoder setBuffer:state.statuses offset:0u atIndex:1u];
            [encoder setBuffer:state.femAccepted offset:0u atIndex:2u];
            [encoder setBuffer:state.femCheckpoint offset:0u atIndex:3u];
            [encoder setBuffer:state.femMaterialStateAccepted offset:0u atIndex:4u];
            [encoder setBuffer:state.femMaterialStateCheckpoint offset:0u atIndex:5u];
        });
        dispatchThreads("nm_topology_rollback", topologyTransactionalTotal, [&] {
            setDispatch();
            [encoder setBuffer:state.statuses offset:0u atIndex:1u];
            [encoder setBuffer:state.femTetrahedraAccepted offset:0u atIndex:2u];
            [encoder setBuffer:state.femTetrahedraCheckpoint offset:0u atIndex:3u];
            [encoder setBuffer:state.femTopologyNodesAccepted offset:0u atIndex:4u];
            [encoder setBuffer:state.femTopologyNodesCheckpoint offset:0u atIndex:5u];
            [encoder setBuffer:state.cohesiveFacesAccepted offset:0u atIndex:6u];
            [encoder setBuffer:state.cohesiveFacesCheckpoint offset:0u atIndex:7u];
            [encoder setBuffer:state.punctureChannelsAccepted offset:0u atIndex:8u];
            [encoder setBuffer:state.punctureChannelsCheckpoint offset:0u atIndex:9u];
            [encoder setBuffer:state.topologyStatesAccepted offset:0u atIndex:10u];
            [encoder setBuffer:state.topologyStatesCheckpoint offset:0u atIndex:11u];
            [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:12u];
            [encoder setBuffer:state.femNodeIncidenceCheckpoint offset:0u atIndex:13u];
            [encoder setBuffer:state.femNodeRanges offset:0u atIndex:14u];
            [encoder setBuffer:state.femNodeRangesCheckpoint offset:0u atIndex:15u];
        });
        dispatchThreads("nm_mixed_rollback", femNodeTotal, [&] {
            setDispatch();
            [encoder setBuffer:state.statuses offset:0u atIndex:1u];
            [encoder setBuffer:state.femFieldsAccepted offset:0u atIndex:2u];
            [encoder setBuffer:state.femFieldsCheckpoint offset:0u atIndex:3u];
        });
        dispatchThreads("nm_learned_rollback", state.dispatch.learnedWeightCount, [&] {
            setDispatch();
            [encoder setBuffer:state.statuses offset:0u atIndex:1u];
            [encoder setBuffer:state.learnedWeightsAccepted offset:0u atIndex:2u];
            [encoder setBuffer:state.learnedWeightsCheckpoint offset:0u atIndex:3u];
            [encoder setBuffer:state.learnedRevisionAccepted offset:0u atIndex:4u];
            [encoder setBuffer:state.learnedRevisionCheckpoint offset:0u atIndex:5u];
        });
        dispatchThreads("nm_contact_rollback_warmstarts", pairTotal, [&] {
            setDispatch();
            [encoder setBuffer:state.statuses offset:0u atIndex:1u];
            [encoder setBuffer:state.contactWarmstartsAccepted offset:0u atIndex:2u];
            [encoder setBuffer:state.contactWarmstartsCheckpoint offset:0u atIndex:3u];
        });
        dispatchThreads(
            "nm_contact_rollback_deformable_warmstarts",
            deformableWarmstartTotal,
            [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.deformableWarmstartsAccepted
                             offset:0u atIndex:2u];
                [encoder setBuffer:state.deformableWarmstartsCandidate
                             offset:0u atIndex:3u];
                [encoder setBuffer:state.deformableWarmstartsCheckpoint
                             offset:0u atIndex:4u];
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

bool Runtime::requiresBodyWrenches() const noexcept {
    return state_ != nullptr && state_->requiresBodyWrenches;
}

bool Runtime::requiresArticulatedResponses() const noexcept {
    return state_ != nullptr && state_->requiresArticulatedResponses;
}

bool Runtime::requiresRigidContactEvidence() const noexcept {
    return state_ != nullptr && state_->adaptiveTransfer &&
        state_->hasAdaptive;
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
        id<MTLBuffer> particleMaterialState =
            copy(state_->particleMaterialStateAccepted);
        id<MTLBuffer> femMaterialState =
            copy(state_->femMaterialStateAccepted);
        id<MTLBuffer> femFields = copy(state_->femFieldsAccepted);
        id<MTLBuffer> solverCertificates = copy(state_->solverCertificates);
        id<MTLBuffer> learnedWeights = copy(state_->learnedWeightsAccepted);
        id<MTLBuffer> learnedRevision = copy(state_->learnedRevisionAccepted);
        id<MTLBuffer> topologyNodes = copy(state_->femTopologyNodesAccepted);
        id<MTLBuffer> topologyTetrahedra = copy(state_->femTetrahedraAccepted);
        id<MTLBuffer> cohesiveFaces = copy(state_->cohesiveFacesAccepted);
        id<MTLBuffer> punctureChannels = copy(state_->punctureChannelsAccepted);
        id<MTLBuffer> topologyStates = copy(state_->topologyStatesAccepted);
        id<MTLBuffer> contactWarmstarts =
            copy(state_->contactWarmstartsAccepted);
        id<MTLBuffer> deformableContactWarmstarts =
            copy(state_->deformableWarmstartsAccepted);
        id<MTLBuffer> adaptive = copy(state_->adaptive);
        id<MTLBuffer> schedulers = copy(state_->schedulers);
        id<MTLBuffer> reactions = copy(state_->frameReactions);
        id<MTLBuffer> contactSamples = state_->captureDiagnostics
            ? copy(state_->contactSamples)
            : nil;
        id<MTLBuffer> contactResponseRows = state_->captureDiagnostics
            ? copy(state_->contactResponseRows)
            : nil;
        id<MTLBuffer> contactResponseColumns = state_->captureDiagnostics
            ? copy(state_->contactResponseColumns)
            : nil;
        id<MTLBuffer> contactResponseValues = state_->captureDiagnostics
            ? copy(state_->contactResponseValues)
            : nil;
        id<MTLBuffer> contactActivePairs = state_->captureDiagnostics
            ? copy(state_->contactActivePairs)
            : nil;
        id<MTLBuffer> contactActiveCounts = state_->captureDiagnostics
            ? copy(state_->contactActiveCounts)
            : nil;
        id<MTLBuffer> identification =
            copy(state_->identificationDistributions);
        id<MTLBuffer> environmentParameters =
            copy(state_->environmentParameters);
        if (particles == nil || femNodes == nil ||
            particleMaterialState == nil || femMaterialState == nil ||
            femFields == nil || solverCertificates == nil ||
            learnedWeights == nil || learnedRevision == nil ||
            topologyNodes == nil || topologyTetrahedra == nil ||
            cohesiveFaces == nil || punctureChannels == nil ||
            topologyStates == nil || contactWarmstarts == nil ||
            deformableContactWarmstarts == nil ||
            adaptive == nil || schedulers == nil || reactions == nil ||
            (state_->captureDiagnostics &&
             (contactSamples == nil || contactResponseRows == nil ||
              contactResponseColumns == nil ||
              contactResponseValues == nil || contactActivePairs == nil ||
              contactActiveCounts == nil)) ||
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
        encodeCopy(
            state_->particleMaterialStateAccepted,
            particleMaterialState
        );
        encodeCopy(state_->femMaterialStateAccepted, femMaterialState);
        encodeCopy(state_->femFieldsAccepted, femFields);
        encodeCopy(state_->solverCertificates, solverCertificates);
        encodeCopy(state_->learnedWeightsAccepted, learnedWeights);
        encodeCopy(state_->learnedRevisionAccepted, learnedRevision);
        encodeCopy(state_->femTopologyNodesAccepted, topologyNodes);
        encodeCopy(state_->femTetrahedraAccepted, topologyTetrahedra);
        encodeCopy(state_->cohesiveFacesAccepted, cohesiveFaces);
        encodeCopy(state_->punctureChannelsAccepted, punctureChannels);
        encodeCopy(state_->topologyStatesAccepted, topologyStates);
        encodeCopy(state_->contactWarmstartsAccepted, contactWarmstarts);
        encodeCopy(
            state_->deformableWarmstartsAccepted,
            deformableContactWarmstarts);
        encodeCopy(state_->adaptive, adaptive);
        encodeCopy(state_->schedulers, schedulers);
        encodeCopy(state_->frameReactions, reactions);
        if (state_->captureDiagnostics) {
            encodeCopy(state_->contactSamples, contactSamples);
            encodeCopy(state_->contactResponseRows, contactResponseRows);
            encodeCopy(state_->contactResponseColumns, contactResponseColumns);
            encodeCopy(state_->contactResponseValues, contactResponseValues);
            encodeCopy(state_->contactActivePairs, contactActivePairs);
            encodeCopy(state_->contactActiveCounts, contactActiveCounts);
        }
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
        const auto readCount = [](id<MTLBuffer> buffer, auto& values,
                                  const std::size_t count) {
            using Value = typename std::decay_t<decltype(values)>::value_type;
            values.resize(count);
            if (count != 0u) {
                const std::size_t bytes = count * sizeof(Value);
                if (bytes > buffer.length) {
                    values.clear();
                    return;
                }
                std::memcpy(values.data(), buffer.contents, bytes);
            }
        };
        read(particles, snapshot.particles);
        read(femNodes, snapshot.femNodes);
        snapshot.materialStateStride = state_->dispatch.materialStateStride;
        read(particleMaterialState, snapshot.particleMaterialState);
        read(femMaterialState, snapshot.femMaterialState);
        read(femFields, snapshot.femFields);
        read(solverCertificates, snapshot.solverCertificates);
        read(learnedWeights, snapshot.learnedWeights);
        snapshot.learnedWeightRevision =
            *static_cast<const std::uint32_t*>(learnedRevision.contents);
        read(topologyNodes, snapshot.femTopologyNodes);
        read(topologyTetrahedra, snapshot.femTopologyTetrahedra);
        read(cohesiveFaces, snapshot.cohesiveFaces);
        read(punctureChannels, snapshot.punctureChannels);
        read(topologyStates, snapshot.topologyStates);
        const std::size_t logicalContactCount =
            static_cast<std::size_t>(state_->dispatch.environmentCount) *
            state_->dispatch.contactPairCount;
        readCount(contactWarmstarts, snapshot.contactWarmstarts,
                  logicalContactCount);
        const std::size_t logicalDeformableContactCount =
            static_cast<std::size_t>(state_->dispatch.environmentCount) *
            state_->dispatch.deformableContactCapacity;
        readCount(
            deformableContactWarmstarts,
            snapshot.deformableContactWarmstarts,
            logicalDeformableContactCount);
        read(adaptive, snapshot.adaptive);
        read(schedulers, snapshot.schedulers);
        read(reactions, snapshot.reactions);
        if (state_->captureDiagnostics) {
            readCount(contactSamples, snapshot.contactSamples,
                      logicalContactCount);
            const std::size_t logicalResponseCount =
                static_cast<std::size_t>(state_->dispatch.environmentCount) *
                state_->contactResponseEntryCount;
            readCount(contactResponseValues, snapshot.contactResponseValues,
                      logicalResponseCount);
            std::vector<std::uint32_t> responseSlotRows;
            std::vector<std::uint32_t> responseSlotColumns;
            std::vector<std::uint32_t> activePairs;
            std::vector<std::uint32_t> activeCounts;
            readCount(contactResponseRows, responseSlotRows,
                      state_->contactResponseEntryCount);
            readCount(contactResponseColumns, responseSlotColumns,
                      state_->contactResponseEntryCount);
            readCount(contactActivePairs, activePairs,
                      static_cast<std::size_t>(state_->dispatch.environmentCount) *
                          state_->contactActiveCapacity);
            readCount(contactActiveCounts, activeCounts,
                      state_->dispatch.environmentCount);
            snapshot.contactResponseRows.resize(
                logicalResponseCount, std::numeric_limits<std::uint32_t>::max());
            snapshot.contactResponseColumns.resize(
                logicalResponseCount, std::numeric_limits<std::uint32_t>::max());
            for (std::uint32_t environment = 0u;
                 environment < state_->dispatch.environmentCount; ++environment) {
                const std::uint32_t activeCount = std::min(
                    activeCounts[environment], state_->contactActiveCapacity);
                const std::size_t activeBase =
                    static_cast<std::size_t>(environment) *
                    state_->contactActiveCapacity;
                const std::size_t responseBase =
                    static_cast<std::size_t>(environment) *
                    state_->contactResponseEntryCount;
                for (std::size_t entry = 0u;
                     entry < state_->contactResponseEntryCount; ++entry) {
                    const std::uint32_t rowSlot = responseSlotRows[entry];
                    const std::uint32_t columnSlot = responseSlotColumns[entry];
                    if (rowSlot < activeCount && columnSlot < activeCount) {
                        snapshot.contactResponseRows[responseBase + entry] =
                            activePairs[activeBase + rowSlot];
                        snapshot.contactResponseColumns[responseBase + entry] =
                            activePairs[activeBase + columnSlot];
                    }
                }
            }
        }
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
