#pragma once

// CPU-side extraction of an anatomically partitioned FEM basis from the
// compiler's immutable CompiledWorld. This preserves the compiled source mesh
// and provenance. It does not enable runtime coordinates x = Bq.

#include "numi/matter/matter.hpp"
#include "numi/matter/partitioned_reduction_basis.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <limits>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace numi::matter {

enum CompiledReductionAnchorReason : std::uint32_t {
    compiledReductionAnchorAttachment = 1u << 0u,
    compiledReductionAnchorContact = 1u << 1u,
    compiledReductionAnchorThorax = 1u << 2u,
    compiledReductionAnchorVascularBinding = 1u << 3u,
    compiledReductionAnchorVascularCavity = 1u << 4u,
};

struct CompiledWorldPartitionedReductionOptions {
    std::uint32_t minimumCoordinateReductionFactor = 4u;
    // Output is a prerequisite/mapping artifact, not the runtime's existing
    // 32-mode preconditioner ABI. Callers can retain larger candidate bases,
    // while production must validate its consumer's mode limit independently.
    std::uint32_t maximumModeCount = 128u;
    // Contact pairs are eligibility candidates, not current active contacts.
    // The runtime contact owner supplies only presently active global FEM nodes.
    std::span<const std::uint32_t> activeContactAnchorGlobalFEMNodes;
    // Rows whose xyz coordinates stay on the exact fine complement path.
    // They are zeroed in B, but are not promoted to one-hot anchor modes.
    std::span<const std::uint32_t> fineCoordinateOnlyGlobalFEMNodes;
    // CompiledWorld has no semantic thorax selector. These must be explicit
    // global FEM node IDs selected by an anatomy-aware source owner.
    std::span<const std::uint32_t> thoraxAnchorGlobalFEMNodes;
    // Selectors 0 and 6 do not retain source region IDs in CompiledWorld.
    // The source owner may pass the exact authored label sequence, in source
    // tetrahedron order, bound to this object and this physical world.
    std::span<const std::uint32_t> explicitAuthoredTetrahedronRegionLabels;
    std::uint32_t explicitRegionLabelObjectIndex = NM_INVALID_INDEX;
    std::uint64_t explicitRegionLabelWorldPhysicsFingerprint = 0u;
};

enum class CompiledReductionRegionLabelSource : std::uint8_t {
    compiledTetrahedronFlags,
    callerBoundSourcePayload,
};

struct CompiledWorldPartitionedReductionBasis {
    PartitionedReductionBasis basis;
    std::uint32_t objectIndex = NM_INVALID_INDEX;
    CompiledReductionRegionLabelSource regionLabelSource =
        CompiledReductionRegionLabelSource::compiledTetrahedronFlags;
    std::uint64_t sourceWorldPhysicsFingerprint = 0u;
    // One entry per basis source row, retaining compiled and authored-source
    // identities for a future runtime prolongation map.
    std::vector<std::uint32_t> globalFEMNodeIndices;
    std::vector<std::uint32_t> sourceNodeIndices;
    // Aligned with the active input tetrahedra in the exact source order.
    std::vector<std::uint32_t> globalFEMTetrahedronIndices;
    std::vector<std::uint32_t> sourceTetrahedronIndices;
    std::vector<std::uint32_t> tetrahedronRegionLabels;
    // Aligned with basis.anchorNodes, which are object-local row indices.
    std::vector<std::uint32_t> anchorReasonMasks;
    std::string error;

    [[nodiscard]] bool valid() const noexcept {
        const std::size_t tetrahedronCount = tetrahedronRegionLabels.size();
        return error.empty() && basis.valid() &&
            globalFEMNodeIndices.size() == basis.nodeCount &&
            sourceNodeIndices.size() == basis.nodeCount &&
            globalFEMTetrahedronIndices.size() == tetrahedronCount &&
            sourceTetrahedronIndices.size() == tetrahedronCount &&
            anchorReasonMasks.size() == basis.anchorNodes.size();
    }
};

namespace compiled_reduction_detail {

[[nodiscard]] inline std::array<double, 3> affineValue(
    const std::array<double, 3>& translation,
    const std::array<double, 9>& gradient,
    const std::array<double, 3>& position
) noexcept {
    return {
        translation[0] + gradient[0] * position[0] +
            gradient[1] * position[1] + gradient[2] * position[2],
        translation[1] + gradient[3] * position[0] +
            gradient[4] * position[1] + gradient[5] * position[2],
        translation[2] + gradient[6] * position[0] +
            gradient[7] * position[1] + gradient[8] * position[2],
    };
}

[[nodiscard]] inline bool finalizeAffineModes(
    PartitionedReductionBasis& basis,
    const std::uint32_t minimumCoordinateReductionFactor,
    const std::uint32_t maximumModeCount,
    const std::span<const std::array<double, 3>> positions,
    const std::span<const double> masses,
    const std::span<const std::uint8_t> fixedMask
) {
    if (positions.size() != basis.nodeCount || masses.size() != basis.nodeCount ||
        fixedMask.size() != basis.nodeCount || basis.regions.empty()) {
        basis.error = "compiled affine basis source arrays do not match the extracted mesh";
        return false;
    }

    basis.modes.clear();
    basis.values.clear();
    basis.projectedMass.clear();
    const std::size_t nodeCount = basis.nodeCount;
    const std::size_t regionCount = basis.regions.size();
    std::size_t freeAnchorCount = 0u;
    for (const std::uint32_t node : basis.anchorNodes) {
        if (node < nodeCount && fixedMask[node] == 0u) ++freeAnchorCount;
    }
    const std::uint64_t requestedModes =
        static_cast<std::uint64_t>(regionCount) * 12u +
        static_cast<std::uint64_t>(freeAnchorCount) * 3u;
    if (requestedModes == 0u || requestedModes > maximumModeCount ||
        requestedModes > std::numeric_limits<std::uint32_t>::max()) {
        basis.error = "compiled affine basis exceeds its configured mode limit after sensitive-anchor preservation";
        return false;
    }
    if (requestedModes * minimumCoordinateReductionFactor >
        basis.freeCoordinateCount) {
        basis.error = "compiled affine basis misses the requested minimum coordinate reduction after sensitive-anchor preservation";
        return false;
    }

    const auto append = [&](const PartitionedReductionMode mode,
                            const std::vector<std::array<double, 3>>& field) {
        basis.modes.push_back(mode);
        for (const auto& value : field) {
            basis.values.push_back({static_cast<float>(value[0]),
                                    static_cast<float>(value[1]),
                                    static_cast<float>(value[2]), 0.0f});
        }
    };
    for (std::size_t region = 0u; region < regionCount; ++region) {
        const PartitionedReductionRegion& metadata = basis.regions[region];
        for (std::uint32_t outputAxis = 0u; outputAxis < 3u; ++outputAxis) {
            std::vector<std::array<double, 3>> field(nodeCount);
            for (std::size_t node = 0u; node < nodeCount; ++node) {
                if (fixedMask[node] != 0u) continue;
                const double weight = basis.partitionWeights[
                    node * basis.partitionCount + region];
                field[node][outputAxis] = weight;
            }
            append({PartitionedReductionModeKind::regionalTranslation,
                    metadata.sourceLabel, outputAxis, NM_INVALID_INDEX, 1.0},
                   field);
        }
        for (std::uint32_t outputAxis = 0u; outputAxis < 3u; ++outputAxis) {
            for (std::uint32_t inputAxis = 0u; inputAxis < 3u; ++inputAxis) {
                std::vector<std::array<double, 3>> field(nodeCount);
                for (std::size_t node = 0u; node < nodeCount; ++node) {
                    if (fixedMask[node] != 0u) continue;
                    const double weight = basis.partitionWeights[
                        node * basis.partitionCount + region];
                    field[node][outputAxis] = weight *
                        (positions[node][inputAxis] - metadata.center[inputAxis]) /
                        metadata.rotationRadius;
                }
                append({PartitionedReductionModeKind::regionalAffineGradient,
                        metadata.sourceLabel, outputAxis * 3u + inputAxis,
                        NM_INVALID_INDEX, metadata.rotationRadius}, field);
            }
        }
    }
    for (const std::uint32_t anchorNode : basis.anchorNodes) {
        if (anchorNode >= nodeCount || fixedMask[anchorNode] != 0u) continue;
        for (std::uint32_t component = 0u; component < 3u; ++component) {
            std::vector<std::array<double, 3>> field(nodeCount);
            field[anchorNode][component] = 1.0;
            append({PartitionedReductionModeKind::anchorTranslation,
                    NM_INVALID_INDEX, component, anchorNode, 1.0}, field);
        }
    }
    basis.modeCount = static_cast<std::uint32_t>(basis.modes.size());
    if (basis.modeCount != requestedModes) {
        basis.error = "compiled affine basis mode accounting is inconsistent";
        return false;
    }

    basis.projectedMass.assign(
        static_cast<std::size_t>(basis.modeCount) * basis.modeCount, 0.0);
    std::vector<std::uint32_t> activeModes;
    for (std::size_t node = 0u; node < nodeCount; ++node) {
        activeModes.clear();
        for (std::uint32_t mode = 0u; mode < basis.modeCount; ++mode) {
            const auto& value = basis.values[
                static_cast<std::size_t>(mode) * nodeCount + node];
            if (value.x != 0.0f || value.y != 0.0f || value.z != 0.0f)
                activeModes.push_back(mode);
        }
        for (const std::uint32_t row : activeModes) {
            const auto& a = basis.values[
                static_cast<std::size_t>(row) * nodeCount + node];
            for (const std::uint32_t column : activeModes) {
                const auto& b = basis.values[
                    static_cast<std::size_t>(column) * nodeCount + node];
                basis.projectedMass[static_cast<std::size_t>(row) *
                                    basis.modeCount + column] +=
                    masses[node] *
                    (static_cast<double>(a.x) * b.x +
                     static_cast<double>(a.y) * b.y +
                     static_cast<double>(a.z) * b.z);
            }
        }
    }
    if (!partitioned_reduction_detail::choleskyPositiveDefinite(
            basis.projectedMass, basis.modeCount)) {
        basis.error = "compiled affine basis is rank deficient under the exact cooked nodal masses";
        return false;
    }
    return true;
}

} // namespace compiled_reduction_detail

// Extracts one immutable, non-mutable FEM object's exact compiled source rows.
// Region labels come from compiler-retained flags or a caller-bound source
// payload; this function never infers anatomy from pooled materials.
// Attachments, caller-selected currently active contact nodes, every vascular
// tissue/cavity node on the object, and caller-selected thorax nodes are held
// as one-hot fine anchor partitions.
[[nodiscard]] inline CompiledWorldPartitionedReductionBasis
buildCompiledWorldPartitionedReductionBasis(
    const CompiledWorld& world,
    const std::uint32_t objectIndex,
    const CompiledWorldPartitionedReductionOptions& options = {}
) {
    using namespace compiled_reduction_detail;
    CompiledWorldPartitionedReductionBasis output;
    output.objectIndex = objectIndex;
    const auto fail = [&](const char* message) {
        output.error = message;
        output.basis.error = message;
        output.basis.values.clear();
        output.basis.modes.clear();
        output.basis.projectedMass.clear();
        return output;
    };
    if (objectIndex >= world.objects.size())
        return fail("compiled reduction object index is out of range");
    if (options.minimumCoordinateReductionFactor < 2u ||
        options.maximumModeCount == 0u || options.maximumModeCount > kPartitionedReductionMaximumConstructionModeCount)
        return fail("compiled reduction factor or mode limit is invalid");
    const NMContinuumObjectGPU& object = world.objects[objectIndex];
    if (object.representation != NM_REPRESENTATION_FEM ||
        object.stateCount == 0u || object.elementCount == 0u)
        return fail("compiled reduction requires a nonempty FEM object");
    if ((object.flags & NM_OBJECT_MUTABLE_TOPOLOGY) != 0u)
        return fail("compiled reduction refuses mutable FEM topology because its source-node map is not immutable");
    const std::uint32_t regionalSelector =
        world.mixedSolver.executionBudgets.z;
    const bool compiledLabelsAvailable =
        regionalSelector >= 2u && regionalSelector <= 5u;
    output.sourceWorldPhysicsFingerprint = world.physicsFingerprint;

    const std::uint64_t stateEnd =
        static_cast<std::uint64_t>(object.stateOffset) + object.stateCount;
    const std::uint64_t elementEnd =
        static_cast<std::uint64_t>(object.elementOffset) + object.elementCount;
    if (stateEnd > world.fem.nodes.size() ||
        stateEnd > world.fem.topologyNodes.size() ||
        elementEnd > world.fem.tetrahedra.size() ||
        stateEnd > std::numeric_limits<std::uint32_t>::max() ||
        elementEnd > std::numeric_limits<std::uint32_t>::max())
        return fail("compiled FEM object node or tetrahedron range exceeds its cooked arena");

    std::vector<std::array<double, 3>> positions;
    std::vector<double> masses;
    std::vector<std::uint8_t> fixedMask;
    std::vector<std::uint32_t> globalNodeIndices;
    std::vector<std::uint32_t> sourceNodeIndices;
    std::vector<std::int64_t> localIndex(world.fem.nodes.size(), -1);
    std::set<std::uint32_t> sourceNodeSet;
    for (std::uint32_t global = object.stateOffset; global < stateEnd; ++global) {
        const NMFEMTopologyNodeGPU& topology = world.fem.topologyNodes[global];
        if (topology.identity.y != objectIndex ||
            (topology.identity.w & NM_TOPOLOGY_ACTIVE) == 0u) continue;
        if (topology.identity.z != object.topologyGeneration ||
            !sourceNodeSet.insert(topology.identity.x).second)
            return fail("compiled FEM source ancestry is duplicated or has a stale topology generation");
        const NMFEMNodeStateGPU& node = world.fem.nodes[global];
        const nm_float4 position = node.restAndFixed;
        const nm_float4 mass = node.positionAndMass;
        if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z) || !std::isfinite(position.w) ||
            !std::isfinite(mass.w) || !(mass.w > 0.0f))
            return fail("compiled FEM source rest position or cooked nodal mass is non-finite or non-positive");
        std::uint8_t fixed = 0u;
        if (position.w == 1.0f || position.w == 3.0f) fixed = 1u;
        else if (position.w != 0.0f && position.w != 2.0f)
            return fail("compiled FEM source contains an unknown constraint tag");
        localIndex[global] = static_cast<std::int64_t>(positions.size());
        positions.push_back({position.x, position.y, position.z});
        masses.push_back(mass.w);
        fixedMask.push_back(fixed);
        globalNodeIndices.push_back(global);
        sourceNodeIndices.push_back(topology.identity.x);
    }
    if (positions.empty() ||
        positions.size() > std::numeric_limits<std::uint32_t>::max())
        return fail("compiled FEM object has no active source nodes or exceeds the index range");
    std::set<std::uint32_t> fineCoordinateOnly;
    for (const std::uint32_t global : options.fineCoordinateOnlyGlobalFEMNodes) {
        if (global >= localIndex.size() || localIndex[global] < 0 ||
            !fineCoordinateOnly.insert(global).second)
            return fail("fine-coordinate-only source row is inactive, foreign, or duplicated");
        fixedMask[static_cast<std::size_t>(localIndex[global])] = 1u;
    }

    std::vector<std::array<std::uint32_t, 4>> tetrahedra;
    std::vector<std::uint32_t> compiledRegionLabels;
    std::vector<std::uint32_t> globalTetrahedronIndices;
    std::vector<std::uint32_t> sourceTetrahedronIndices;
    for (std::uint32_t index = object.elementOffset; index < elementEnd; ++index) {
        const NMTetrahedronGPU& tet = world.fem.tetrahedra[index];
        if ((tet.identity.w & NM_OBJECT_ACTIVE) == 0u) continue;
        if (tet.identity.y != objectIndex)
            return fail("active compiled FEM tetrahedron range contains a foreign object");
        const std::array<std::uint32_t, 4> globalNodes{
            tet.nodes.x, tet.nodes.y, tet.nodes.z, tet.nodes.w};
        std::array<std::uint32_t, 4> localNodes{};
        for (std::size_t corner = 0u; corner < 4u; ++corner) {
            const std::uint32_t global = globalNodes[corner];
            if (global >= localIndex.size() || localIndex[global] < 0)
                return fail("active compiled FEM tetrahedron references an omitted or foreign source node");
            localNodes[corner] =
                static_cast<std::uint32_t>(localIndex[global]);
        }
        tetrahedra.push_back(localNodes);
        compiledRegionLabels.push_back((tet.identity.w &
            NM_TETRAHEDRON_PRECONDITIONER_REGION_MASK) >>
            NM_TETRAHEDRON_PRECONDITIONER_REGION_SHIFT);
        globalTetrahedronIndices.push_back(index);
        sourceTetrahedronIndices.push_back(index - object.elementOffset);
    }
    if (tetrahedra.empty())
        return fail("compiled FEM object has no active source tetrahedra");

    std::vector<std::uint32_t> regionLabels;
    if (compiledLabelsAvailable) {
        output.regionLabelSource =
            CompiledReductionRegionLabelSource::compiledTetrahedronFlags;
        regionLabels = compiledRegionLabels;
        if (!options.explicitAuthoredTetrahedronRegionLabels.empty() &&
            !std::equal(regionLabels.begin(), regionLabels.end(),
                        options.explicitAuthoredTetrahedronRegionLabels.begin(),
                        options.explicitAuthoredTetrahedronRegionLabels.end()))
            return fail("explicit source region labels disagree with the compiled source-bound labels");
    } else {
        if (options.explicitRegionLabelObjectIndex != objectIndex ||
            options.explicitRegionLabelWorldPhysicsFingerprint == 0u ||
            options.explicitRegionLabelWorldPhysicsFingerprint !=
                world.physicsFingerprint ||
            options.explicitAuthoredTetrahedronRegionLabels.size() !=
                tetrahedra.size())
            return fail("selector 0/6 requires exact source labels bound to the selected FEM object and compiled-world physics fingerprint");
        output.regionLabelSource =
            CompiledReductionRegionLabelSource::callerBoundSourcePayload;
        regionLabels.assign(
            options.explicitAuthoredTetrahedronRegionLabels.begin(),
            options.explicitAuthoredTetrahedronRegionLabels.end());
        if (std::any_of(regionLabels.begin(), regionLabels.end(),
                        [](const std::uint32_t label) { return label == NM_INVALID_INDEX; }))
            return fail("explicit source region label contains the invalid sentinel");
    }

    std::map<std::uint32_t, std::uint32_t> anchorReasons;
    const auto addAnchor = [&](const std::uint32_t global,
                               const std::uint32_t reason) -> bool {
        if (global >= localIndex.size() || localIndex[global] < 0) return false;
        anchorReasons[global] |= reason;
        return true;
    };
    for (const NMFEMHumanAttachmentGPU& attachment : world.fem.humanAttachments) {
        if (attachment.identity.z != objectIndex) continue;
        if (!addAnchor(attachment.identity.x,
                       compiledReductionAnchorAttachment))
            return fail("compiled Human attachment references an inactive or foreign FEM source node");
    }
    std::set<std::uint32_t> compiledContactCandidates;
    for (const NMContactPairGPU& pair : world.contact.pairs) {
        if (pair.objectIndex != objectIndex) continue;
        if (pair.continuumNode < world.dispatch.gridNodeCount)
            return fail("compiled FEM contact candidate uses an MPM rather than FEM unified-node index");
        const std::uint64_t global =
            static_cast<std::uint64_t>(pair.continuumNode) -
            world.dispatch.gridNodeCount;
        if (global >= world.fem.nodes.size() || global > UINT32_MAX)
            return fail("compiled FEM contact candidate is outside the FEM node arena");
        compiledContactCandidates.insert(static_cast<std::uint32_t>(global));
    }
    for (const std::uint32_t global :
         options.activeContactAnchorGlobalFEMNodes) {
        if (!compiledContactCandidates.contains(global) ||
            !addAnchor(global, compiledReductionAnchorContact))
            return fail("active-contact anchor is not an eligible compiled contact node of the selected FEM object");
    }
    for (const NMVascularTissueBindingGPU& binding :
         world.vascular.tissueBindings) {
        const std::uint32_t global = binding.identity.y;
        if (global < localIndex.size() && localIndex[global] >= 0)
            anchorReasons[global] |= compiledReductionAnchorVascularBinding;
    }
    for (const NMVascularCavityFaceGPU& face : world.vascular.cavityFaces) {
        if (face.identity.x >= world.vascular.cavities.size())
            return fail("compiled vascular cavity face references a missing cavity");
        if (world.vascular.cavities[face.identity.x].identity.z != objectIndex)
            continue;
        for (const std::uint32_t global :
             {face.nodes.x, face.nodes.y, face.nodes.z}) {
            if (!addAnchor(global, compiledReductionAnchorVascularCavity))
                return fail("compiled vascular cavity face references an inactive or foreign FEM source node");
        }
    }
    for (const std::uint32_t global : options.thoraxAnchorGlobalFEMNodes) {
        if (!addAnchor(global, compiledReductionAnchorThorax))
            return fail("caller-selected thorax anchor is not an active node of the selected compiled FEM object");
    }

    std::vector<std::uint32_t> localAnchors;
    std::vector<std::uint32_t> anchorReasonMasks;
    localAnchors.reserve(anchorReasons.size());
    anchorReasonMasks.reserve(anchorReasons.size());
    for (const auto& [global, reason] : anchorReasons) {
        localAnchors.push_back(static_cast<std::uint32_t>(localIndex[global]));
        anchorReasonMasks.push_back(reason);
    }

    PartitionedReductionBasisInput input{
        positions, masses, fixedMask, tetrahedra, regionLabels, localAnchors};
    PartitionedReductionBasisOptions baseOptions;
    baseOptions.minimumCoordinateReductionFactor =
        options.minimumCoordinateReductionFactor;
    // Reuse the existing helper's validated source partition construction.
    // The future runtime preconditioner still has its own smaller ABI limit.
    baseOptions.maximumModeCount = kPartitionedReductionMaximumConstructionModeCount;
    output.basis = buildPartitionedReductionBasis(input, baseOptions);
    if (!output.basis.valid()) {
        output.error = output.basis.error;
        return output;
    }
    if (!finalizeAffineModes(output.basis,
                             options.minimumCoordinateReductionFactor,
                             options.maximumModeCount, positions, masses,
                             fixedMask)) {
        output.error = output.basis.error;
        return output;
    }
    output.globalFEMNodeIndices = std::move(globalNodeIndices);
    output.sourceNodeIndices = std::move(sourceNodeIndices);
    output.globalFEMTetrahedronIndices = std::move(globalTetrahedronIndices);
    output.sourceTetrahedronIndices = std::move(sourceTetrahedronIndices);
    output.tetrahedronRegionLabels = std::move(regionLabels);
    output.anchorReasonMasks = std::move(anchorReasonMasks);
    return output;
}

} // namespace numi::matter
