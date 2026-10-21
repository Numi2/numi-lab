#pragma once

#include "numi/matter/shared.h"

#include <cstdint>
#include <span>

namespace numi::matter::detail {

struct AcceptedZ0StaticSourceView {
    const NMMatterDispatchGPU& dispatch;
    std::span<const NMContinuumObjectGPU> objects;
    std::span<const NMTetrahedronGPU> tetrahedra;
    std::span<const NMMaterialGPU> materials;
    bool identificationEnabled = false;
    bool fullCoupledTrueResidual = false;
};

// Z0 is only a search-space seed: every use recomputes the current full
// operator action, and the owner publishes it only after its accepted gate.
// Keep cross-transaction reuse limited to source layouts whose topology and
// constitutive state cannot change in place. Runtime device checks separately
// invalidate it when contact identities, FEM constraints, or dt change.
[[nodiscard]] inline const char* acceptedZ0StaticSourceIneligibility(
    const AcceptedZ0StaticSourceView& source
) noexcept {
    const auto& dispatch = source.dispatch;
    if (!source.fullCoupledTrueResidual)
        return "full_coupled_true_residual_required";
    if (dispatch.environmentCount != 1u || dispatch.femNodeCount == 0u ||
        dispatch.tetrahedronCount == 0u || dispatch.objectCount == 0u ||
        dispatch.objectCount != source.objects.size() ||
        dispatch.tetrahedronCount != source.tetrahedra.size() ||
        dispatch.materialCount != source.materials.size())
        return "compiled_source_extent_mismatch";
    if (dispatch.mpmActiveNodeCapacity != 0u ||
        dispatch.cohesiveFaceCount != 0u || dispatch.fieldBoundaryCount != 0u)
        return "unsupported_coupled_source_layout";
    // Static compiled meshes also have topology identity storage. Only
    // spare growth capacity is ineligible; existing rows are not mutation.
    // Identifiable parameter distributions may be resident but dormant.
    // Automatic/manual updates are excluded by admission and encode guards.
    constexpr std::uint32_t mutableMatterFlags =
        NM_MATTER_ADAPTIVE |
        NM_MATTER_MUTATION | NM_MATTER_LEARNED_MATERIAL;
    if ((dispatch.flags & mutableMatterFlags) != 0u ||
        dispatch.mutationCommandCount != 0u ||
        dispatch.identificationCandidateCount != 0u ||
        dispatch.learnedMaterialCount != 0u ||
        dispatch.learnedLayerCount != 0u ||
        dispatch.learnedWeightCount != 0u ||
        dispatch.topologyNodeCapacity > dispatch.femNodeCount ||
        dispatch.topologyTetrahedronCapacity > dispatch.tetrahedronCount ||
        source.identificationEnabled)
        return "mutable_topology_or_material_source";

    constexpr std::uint32_t mutableObjectFlags =
        NM_OBJECT_ADAPTIVE | NM_OBJECT_IDENTIFIABLE |
        NM_OBJECT_MUTABLE_TOPOLOGY;
    for (const NMContinuumObjectGPU& object : source.objects) {
        if ((object.flags & mutableObjectFlags) != 0u)
            return "mutable_object_source";
    }

    // HAS_DISSIPATION denotes a compiled, memoryless viscous potential; it
    // depends on the current deformation rate and is reevaluated with the
    // current operator action. Persistent material history is represented by
    // stateCount / HAS_STATE / HAS_IMPLICIT_STATE and remains ineligible.
    constexpr std::uint32_t statefulMaterialFlags =
        NM_MATERIAL_HAS_STATE | NM_MATERIAL_HAS_IMPLICIT_STATE;
    std::uint32_t activeFemTetrahedra = 0u;
    for (const NMTetrahedronGPU& tet : source.tetrahedra) {
        if ((tet.identity.w & NM_OBJECT_ACTIVE) == 0u) continue;
        const std::uint32_t materialIndex = tet.identity.x;
        const std::uint32_t objectIndex = tet.identity.y;
        if (materialIndex >= source.materials.size() ||
            objectIndex >= source.objects.size() ||
            source.objects[objectIndex].representation != NM_REPRESENTATION_FEM ||
            (source.objects[objectIndex].flags & NM_OBJECT_ACTIVE) == 0u)
            return "active_fem_element_source_invalid";
        ++activeFemTetrahedra;
        const NMMaterialGPU& material = source.materials[materialIndex];
        if (material.stateCount != 0u ||
            (material.flags & statefulMaterialFlags) != 0u)
            return "stateful_material_source";
    }
    if (activeFemTetrahedra == 0u)
        return "no_active_fem_elements";
    return nullptr;
}

} // namespace numi::matter::detail
