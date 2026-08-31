#include "numi/matter/detail.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace numi::matter {
namespace {

enum class FingerprintSection : std::uint32_t {
    dispatch = 1u,
    mixedSolver,
    materials,
    mixedMaterials,
    parameters,
    stateInitials,
    instructions,
    scalarPrograms,
    objects,
    mpmParticles,
    mpmNodes,
    mpmGrids,
    mpmBlocks,
    mpmBlockLookup,
    mpmStencils,
    mpmNodeIncidence,
    mpmNodeRanges,
    femNodes,
    femTetrahedra,
    femSurfaceFaces,
    femNodeIncidence,
    femNodeRanges,
    femCapacities,
    femFields,
    femFieldBoundaries,
    femTopologyNodes,
    femCohesiveFaces,
    femMutationCommands,
    femPunctureChannels,
    femHumanAttachments,
    rigidProxies,
    contactPairs,
    contactNodeIncidence,
    contactNodeRanges,
    rigidIncidence,
    rigidRanges,
    adaptive,
    schedulers,
    identification,
    learnedMaterials,
    learnedLayers,
    learnedWeights,
};

template <typename T>
void hashSection(
    std::uint64_t& fingerprint,
    const FingerprintSection section,
    const std::span<const T> values
) noexcept {
    const std::uint32_t id = static_cast<std::uint32_t>(section);
    const std::uint32_t elementSize = sizeof(T);
    const std::uint64_t elementCount = values.size();
    fingerprint = detail::hashBytes(&id, sizeof(id), fingerprint);
    fingerprint = detail::hashBytes(
        &elementSize,
        sizeof(elementSize),
        fingerprint
    );
    fingerprint = detail::hashBytes(
        &elementCount,
        sizeof(elementCount),
        fingerprint
    );
    fingerprint = detail::hashBytes(
        values.data(),
        values.size_bytes(),
        fingerprint
    );
}

} // namespace

std::uint64_t compiledWorldFingerprint(
    const CompiledWorld& world
) noexcept {
    std::uint64_t fingerprint = 1469598103934665603ull;
    hashSection(
        fingerprint,
        FingerprintSection::dispatch,
        std::span<const NMMatterDispatchGPU>(&world.dispatch, 1u)
    );
    hashSection(
        fingerprint,
        FingerprintSection::mixedSolver,
        std::span<const NMMixedSolverGPU>(&world.mixedSolver, 1u)
    );
    hashSection(fingerprint, FingerprintSection::materials,
        std::span<const NMMaterialGPU>(world.materials));
    hashSection(fingerprint, FingerprintSection::mixedMaterials,
        std::span<const NMMixedMaterialGPU>(world.mixedMaterials));
    hashSection(fingerprint, FingerprintSection::parameters,
        std::span<const NMParameterRangeGPU>(world.parameters));
    hashSection(fingerprint, FingerprintSection::stateInitials,
        std::span<const float>(world.stateInitials));
    hashSection(fingerprint, FingerprintSection::instructions,
        std::span<const NMExpressionInstructionGPU>(world.instructions));
    hashSection(fingerprint, FingerprintSection::scalarPrograms,
        std::span<const NMScalarProgramGPU>(world.scalarPrograms));
    hashSection(fingerprint, FingerprintSection::objects,
        std::span<const NMContinuumObjectGPU>(world.objects));
    hashSection(fingerprint, FingerprintSection::mpmParticles,
        std::span<const NMParticleStateGPU>(world.mpm.particles));
    hashSection(fingerprint, FingerprintSection::mpmNodes,
        std::span<const NMGridNodeStateGPU>(world.mpm.nodes));
    hashSection(fingerprint, FingerprintSection::mpmGrids,
        std::span<const NMMPMGridGPU>(world.mpm.grids));
    hashSection(fingerprint, FingerprintSection::mpmBlocks,
        std::span<const NMMPMBlockGPU>(world.mpm.blocks));
    hashSection(fingerprint, FingerprintSection::mpmBlockLookup,
        std::span<const std::uint32_t>(world.mpm.blockLookup));
    hashSection(fingerprint, FingerprintSection::mpmStencils,
        std::span<const NMMPMStencilGPU>(world.mpm.stencils));
    hashSection(fingerprint, FingerprintSection::mpmNodeIncidence,
        std::span<const std::uint32_t>(world.mpm.nodeIncidence));
    hashSection(fingerprint, FingerprintSection::mpmNodeRanges,
        std::span<const NMIncidenceRangeGPU>(world.mpm.nodeRanges));
    hashSection(fingerprint, FingerprintSection::femNodes,
        std::span<const NMFEMNodeStateGPU>(world.fem.nodes));
    hashSection(fingerprint, FingerprintSection::femTetrahedra,
        std::span<const NMTetrahedronGPU>(world.fem.tetrahedra));
    hashSection(fingerprint, FingerprintSection::femSurfaceFaces,
        std::span<const NMFEMSurfaceFaceGPU>(world.fem.surfaceFaces));
    hashSection(fingerprint, FingerprintSection::femNodeIncidence,
        std::span<const std::uint32_t>(world.fem.nodeIncidence));
    hashSection(fingerprint, FingerprintSection::femNodeRanges,
        std::span<const NMIncidenceRangeGPU>(world.fem.nodeRanges));
    hashSection(fingerprint, FingerprintSection::femCapacities,
        std::span<const NMFEMCapacityGPU>(world.fem.capacities));
    hashSection(fingerprint, FingerprintSection::femFields,
        std::span<const NMFEMFieldStateGPU>(world.fem.fields));
    hashSection(fingerprint, FingerprintSection::femFieldBoundaries,
        std::span<const NMFieldBoundaryGPU>(world.fem.fieldBoundaries));
    hashSection(fingerprint, FingerprintSection::femTopologyNodes,
        std::span<const NMFEMTopologyNodeGPU>(world.fem.topologyNodes));
    hashSection(fingerprint, FingerprintSection::femCohesiveFaces,
        std::span<const NMCohesiveFaceGPU>(world.fem.cohesiveFaces));
    hashSection(fingerprint, FingerprintSection::femMutationCommands,
        std::span<const NMMutationCommandGPU>(world.fem.mutationCommands));
    hashSection(fingerprint, FingerprintSection::femPunctureChannels,
        std::span<const NMPunctureChannelGPU>(world.fem.punctureChannels));
    hashSection(fingerprint, FingerprintSection::femHumanAttachments,
        std::span<const NMFEMHumanAttachmentGPU>(
            world.fem.humanAttachments));
    hashSection(fingerprint, FingerprintSection::rigidProxies,
        std::span<const NMRigidProxyGPU>(world.contact.rigidProxies));
    hashSection(fingerprint, FingerprintSection::contactPairs,
        std::span<const NMContactPairGPU>(world.contact.pairs));
    hashSection(fingerprint, FingerprintSection::contactNodeIncidence,
        std::span<const std::uint32_t>(world.contact.nodeIncidence));
    hashSection(fingerprint, FingerprintSection::contactNodeRanges,
        std::span<const NMIncidenceRangeGPU>(world.contact.nodeRanges));
    hashSection(fingerprint, FingerprintSection::rigidIncidence,
        std::span<const std::uint32_t>(world.contact.rigidIncidence));
    hashSection(fingerprint, FingerprintSection::rigidRanges,
        std::span<const NMIncidenceRangeGPU>(world.contact.rigidRanges));
    hashSection(fingerprint, FingerprintSection::adaptive,
        std::span<const NMAdaptiveStateGPU>(world.adaptive));
    hashSection(fingerprint, FingerprintSection::schedulers,
        std::span<const NMSchedulerStateGPU>(world.schedulers));
    hashSection(fingerprint, FingerprintSection::identification,
        std::span<const NMIdentificationDistributionGPU>(world.identification));
    hashSection(fingerprint, FingerprintSection::learnedMaterials,
        std::span<const NMLearnedMaterialGPU>(world.learnedMaterials));
    hashSection(fingerprint, FingerprintSection::learnedLayers,
        std::span<const NMLearnedLayerGPU>(world.learnedLayers));
    hashSection(fingerprint, FingerprintSection::learnedWeights,
        std::span<const float>(world.learnedWeights));
    return fingerprint == 0u ? 1u : fingerprint;
}

} // namespace numi::matter
