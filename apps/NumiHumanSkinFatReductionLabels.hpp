#pragma once

// Source-bound anatomical labels for the registered Human skin/fat shell.
// Labels retain the original Human donor-body IDs. This is compiler input
// provenance only; it does not enable reduced coordinates in the runtime.
#include "NumiHumanCoarseTissuePreparation.hpp"
#include "numi/matter/compiled_world_partitioned_reduction.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace metalrobo::human::coarse_tissue {

struct RegisteredSkinFatReductionSourceLabels {
    // Compiled source tetrahedron order: one original donor-body ID per tet.
    std::vector<std::uint32_t> tetrahedronRegionLabels;
    // Original registered source vertex and layer for each authored shell node.
    std::vector<std::uint32_t> sourceNodeVertexIndices;
    std::vector<std::uint32_t> sourceNodeLayerIndices;
    // Original registered skin face identity and dominant donor body for each tet.
    std::vector<std::uint32_t> tetrahedronSourceFaceIndices;
    std::vector<std::uint32_t> tetrahedronSourceDonorBodyIndices;
    // Immediate parent for the dominant donor body; root bodies retain INVALID.
    std::vector<std::uint32_t> tetrahedronSourceParentBodyIndices;
    // Complete original donor-body ancestry, body first and root last.
    std::map<std::uint32_t, std::vector<std::uint32_t>> donorBodyAncestry;
    std::size_t registeredVertexCount = 0u;
    std::size_t registeredFaceCount = 0u;
    std::size_t sourceTetrahedronCount = 0u;
};

// Verifies that a later shared-world compile contains the exact shell source
// for which the anatomical labels were authored. The physics fingerprint is
// deliberately not reused here: the final world may include an additional
// vascular payload and must receive a freshly built reduction basis.
[[nodiscard]] inline bool verifyRegisteredSkinFatCompiledShell(
    const numi::matter::CompiledWorld& world,
    const std::uint32_t objectIndex,
    const numi::matter::ObjectSource& sourceShell,
    const RegisteredSkinFatReductionSourceLabels& labels,
    std::string& error
) {
    const auto fail = [&](const char* message) {
        error = message;
        return false;
    };
    if (objectIndex >= world.objects.size())
        return fail("shared compiled world lost the registered skin/fat object index");
    const auto& object = world.objects[objectIndex];
    if (object.representation != NM_REPRESENTATION_FEM ||
        object.stateCount != sourceShell.femNodes.size() ||
        object.elementCount != sourceShell.tetrahedra.size() ||
        object.topologyGeneration == 0u)
        return fail("shared compiled skin/fat object dimensions or topology generation differ from the labeled source");
    const std::uint64_t nodeEnd =
        static_cast<std::uint64_t>(object.stateOffset) + object.stateCount;
    const std::uint64_t tetEnd =
        static_cast<std::uint64_t>(object.elementOffset) + object.elementCount;
    if (nodeEnd > world.fem.nodes.size() ||
        nodeEnd > world.fem.topologyNodes.size() ||
        tetEnd > world.fem.tetrahedra.size())
        return fail("shared compiled skin/fat object range escapes its FEM arena");
    if (labels.registeredVertexCount == 0u ||
        sourceShell.femNodes.size() != 3u * labels.registeredVertexCount ||
        labels.sourceNodeVertexIndices.size() != sourceShell.femNodes.size() ||
        labels.sourceNodeLayerIndices.size() != sourceShell.femNodes.size() ||
        labels.sourceTetrahedronCount != sourceShell.tetrahedra.size() ||
        labels.tetrahedronRegionLabels.size() != sourceShell.tetrahedra.size() ||
        labels.tetrahedronSourceFaceIndices.size() != sourceShell.tetrahedra.size() ||
        labels.tetrahedronSourceDonorBodyIndices.size() != sourceShell.tetrahedra.size() ||
        labels.tetrahedronSourceParentBodyIndices.size() != sourceShell.tetrahedra.size())
        return fail("retained skin/fat source labels no longer cover the exact source mesh");

    for (std::uint32_t local = 0u; local < object.stateCount; ++local) {
        const std::uint32_t global = object.stateOffset + local;
        const auto& topology = world.fem.topologyNodes[global];
        if (topology.identity.x != local || topology.identity.y != objectIndex ||
            topology.identity.z != object.topologyGeneration ||
            (topology.identity.w & NM_TOPOLOGY_ACTIVE) == 0u)
            return fail("shared compiled skin/fat node order or topology ancestry differs from the labeled source");
        const auto& cooked = world.fem.nodes[global].positionAndMass;
        const auto& source = sourceShell.femNodes[local];
        const float expectedX = static_cast<float>(source[0]);
        const float expectedY = static_cast<float>(source[1]);
        const float expectedZ = static_cast<float>(source[2]);
        if (!std::isfinite(expectedX) || !std::isfinite(expectedY) ||
            !std::isfinite(expectedZ) || cooked.x != expectedX ||
            cooked.y != expectedY || cooked.z != expectedZ)
            return fail("shared compiled skin/fat Float32 source geometry differs from the registered labeled geometry");
    }
    for (std::uint32_t local = 0u; local < object.elementCount; ++local) {
        const auto& cooked = world.fem.tetrahedra[object.elementOffset + local];
        const auto& source = sourceShell.tetrahedra[local].nodes;
        if (cooked.identity.y != objectIndex ||
            (cooked.identity.w & NM_OBJECT_ACTIVE) == 0u ||
            cooked.nodes.x != object.stateOffset + source[0] ||
            cooked.nodes.y != object.stateOffset + source[1] ||
            cooked.nodes.z != object.stateOffset + source[2] ||
            cooked.nodes.w != object.stateOffset + source[3])
            return fail("shared compiled skin/fat tetrahedron order or source connectivity differs from the anatomical labels");
    }
    error.clear();
    return true;
}

// Build the basis and, when requested, install a fresh runtime payload bound
// to this exact compiled world's physics fingerprint. This single function is
// shared by preparation diagnostics and the final RestingRun handoff.
[[nodiscard]] inline numi::matter::CompiledWorldPartitionedReductionBasis
buildAndInstallRegisteredSkinFatPreconditioner(
    numi::matter::CompiledWorld& world,
    const std::uint32_t objectIndex,
    const RegisteredSkinFatReductionSourceLabels& labels,
    const numi::matter::ObjectSource& sourceShell,
    const bool installPayload,
    std::string& error
) {
    using Candidate = numi::matter::CompiledWorldPartitionedReductionBasis;
    Candidate failed;
    failed.objectIndex = objectIndex;
    const auto fail = [&](const std::string& message) {
        error = message;
        failed.error = message;
        failed.basis.error = message;
        return failed;
    };
    if (world.physicsFingerprint == 0u)
        return fail("shared compiled world has no physics fingerprint for source-label binding");
    if (!verifyRegisteredSkinFatCompiledShell(
            world, objectIndex, sourceShell, labels, error))
        return fail(error);

    numi::matter::CompiledWorldPartitionedReductionOptions options;
    options.minimumCoordinateReductionFactor = 4u;
    options.maximumModeCount = 512u;
    options.explicitAuthoredTetrahedronRegionLabels =
        std::span<const std::uint32_t>(labels.tetrahedronRegionLabels);
    options.explicitRegionLabelObjectIndex = objectIndex;
    options.explicitRegionLabelWorldPhysicsFingerprint = world.physicsFingerprint;
    Candidate candidate =
        numi::matter::buildCompiledWorldPartitionedReductionBasis(
            world, objectIndex, options);
    if (!candidate.valid()) {
        error = candidate.error.empty()
            ? "compiled skin/fat basis failed its completeness contract"
            : candidate.error;
        return candidate;
    }
    if (candidate.tetrahedronRegionLabels != labels.tetrahedronRegionLabels ||
        candidate.sourceTetrahedronIndices.size() != sourceShell.tetrahedra.size() ||
        candidate.sourceNodeIndices.size() != sourceShell.femNodes.size())
        return fail("fresh compiled skin/fat basis lost exact labeled source order");
    for (std::size_t tet = 0u; tet < candidate.sourceTetrahedronIndices.size(); ++tet) {
        if (candidate.sourceTetrahedronIndices[tet] != tet)
            return fail("fresh compiled skin/fat basis tetrahedron order differs from registered source labels");
    }
    const auto& descriptor = world.objects[objectIndex];
    for (std::size_t row = 0u; row < candidate.sourceNodeIndices.size(); ++row) {
        if (candidate.sourceNodeIndices[row] != row ||
            candidate.globalFEMNodeIndices[row] != descriptor.stateOffset + row)
            return fail("fresh compiled skin/fat basis node rows differ from the exact source object order");
    }
    if (!installPayload) {
        error.clear();
        return candidate;
    }

    if (candidate.basis.regions.empty() ||
        candidate.basis.regions.size() > 42u ||
        candidate.basis.regions.size() >
            std::numeric_limits<std::uint32_t>::max() / 12u)
        return fail("fresh skin/fat basis region count exceeds the runtime block limit");
    numi::matter::detail::HumanPartitionedPreconditionerSource payload;
    payload.objectIndex = objectIndex;
    payload.basisNodeCount = candidate.basis.nodeCount;
    payload.regionCount = static_cast<std::uint32_t>(candidate.basis.regions.size());
    payload.blockModeCount = 12u;
    payload.modeCount = payload.regionCount * payload.blockModeCount;
    payload.freeCoordinateCount = candidate.basis.freeCoordinateCount;
    payload.sensitiveAnchorCount = static_cast<std::uint32_t>(candidate.basis.anchorNodes.size());
    payload.minimumCoordinateReductionFactor = 4u;
    payload.sourceWorldPhysicsFingerprint = world.physicsFingerprint;
    payload.globalFEMNodeIndices = candidate.globalFEMNodeIndices;
    payload.regionLabels.reserve(payload.regionCount);
    payload.blockModeIndices.reserve(payload.modeCount);
    payload.basisValues.reserve(static_cast<std::size_t>(payload.modeCount) *
                                 payload.basisNodeCount);
    if (payload.regionCount == 0u || payload.modeCount > 512u ||
        static_cast<std::uint64_t>(payload.modeCount) * 4u >
            payload.freeCoordinateCount)
        return fail("fresh skin/fat basis does not satisfy the runtime regional-block admission bounds");
    for (std::uint32_t region = 0u; region < payload.regionCount; ++region) {
        const std::uint32_t regionLabel = candidate.basis.regions[region].sourceLabel;
        payload.regionLabels.push_back(regionLabel);
        std::uint32_t retainedModes = 0u;
        for (std::uint32_t modeIndex = 0u;
             modeIndex < candidate.basis.modeCount; ++modeIndex) {
            const auto& mode = candidate.basis.modes[modeIndex];
            if (mode.regionLabel != regionLabel ||
                mode.kind == numi::matter::PartitionedReductionModeKind::anchorTranslation)
                continue;
            if (retainedModes >= payload.blockModeCount)
                return fail("fresh skin/fat source region has more than twelve retained affine modes");
            payload.blockModeIndices.push_back(
                region * payload.blockModeCount + retainedModes);
            const auto sourceOffset = static_cast<std::size_t>(modeIndex) *
                                      payload.basisNodeCount;
            if (sourceOffset + payload.basisNodeCount > candidate.basis.values.size())
                return fail("fresh skin/fat basis mode row escapes its coefficient storage");
            payload.basisValues.insert(
                payload.basisValues.end(),
                candidate.basis.values.begin() + static_cast<std::ptrdiff_t>(sourceOffset),
                candidate.basis.values.begin() + static_cast<std::ptrdiff_t>(sourceOffset +
                                                                              payload.basisNodeCount));
            ++retainedModes;
        }
        if (retainedModes != payload.blockModeCount)
            return fail("fresh skin/fat source region does not have exactly twelve retained affine modes");
    }
    if (payload.basisValues.size() !=
            static_cast<std::size_t>(payload.modeCount) * payload.basisNodeCount ||
        payload.blockModeIndices.size() != payload.modeCount ||
        payload.regionLabels.size() != payload.regionCount)
        return fail("fresh Human partitioned preconditioner payload packing is inconsistent");
    payload.enabled = true;
    world.humanPartitionedPreconditioner = std::move(payload);
    error.clear();
    return candidate;
}

// The Human shell has three nodal layers [outer skin, bonded interface, deep
// fat]. Each triangular registered source face owns three prism tets per
// layer. A tet's four nodes therefore map modulo layerWidth to exactly the
// three source vertices of its registered face.
template<class VerifyBoundBody, class ParentBodyAt>
[[nodiscard]] inline bool buildRegisteredSkinFatReductionSourceLabels(
    const RegisteredSurface& surface,
    const numi::matter::ObjectSource& shell,
    const std::uint32_t layerWidth,
    const std::size_t sourceBodyCount,
    VerifyBoundBody&& verifyBoundBody,
    ParentBodyAt&& parentBodyAt,
    RegisteredSkinFatReductionSourceLabels& output,
    std::string& error
) {
    const auto fail = [&](const char* message) {
        error = message;
        return false;
    };
    if (layerWidth == 0u || sourceBodyCount == 0u ||
        surface.faces.size() > std::numeric_limits<std::size_t>::max() / 6u ||
        surface.worldPoints.size() != layerWidth ||
        surface.faces.empty() || surface.donorBodyIndices.size() != layerWidth ||
        surface.sourceVertexIndices.size() != layerWidth ||
        (!surface.sourceFaceIndices.empty() &&
         surface.sourceFaceIndices.size() != surface.faces.size()) ||
        shell.representation != numi::matter::Representation::fem ||
        shell.femNodes.size() != 3u * static_cast<std::size_t>(layerWidth) ||
        shell.tetrahedra.size() != 6u * surface.faces.size()) {
        return fail("registered Human skin/fat source arrays do not match the three-layer prism mesh");
    }

    using FaceKey = std::array<std::uint32_t, 3u>;
    std::map<FaceKey, std::size_t> faceByVertices;
    std::set<std::uint32_t> sourceVertices;
    std::set<std::uint32_t> sourceFaceIdentities;
    for (std::size_t vertex = 0u; vertex < layerWidth; ++vertex) {
        const std::uint32_t sourceVertex = surface.sourceVertexIndices[vertex];
        const std::uint32_t donorBody = surface.donorBodyIndices[vertex];
        if (sourceVertex == NM_INVALID_INDEX || donorBody == NM_INVALID_INDEX ||
            donorBody >= sourceBodyCount || !sourceVertices.insert(sourceVertex).second ||
            !verifyBoundBody(sourceVertex, donorBody)) {
            return fail("registered skin vertex has invalid or unverified original source/donor identity");
        }
    }
    for (std::size_t face = 0u; face < surface.faces.size(); ++face) {
        FaceKey key = surface.faces[face];
        if (key[0] >= layerWidth || key[1] >= layerWidth || key[2] >= layerWidth ||
            key[0] == key[1] || key[0] == key[2] || key[1] == key[2]) {
            return fail("registered skin source face contains an invalid local vertex index");
        }
        std::sort(key.begin(), key.end());
        if (!faceByVertices.emplace(key, face).second) {
            return fail("registered skin source has duplicate faces after canonical vertex ordering");
        }
        if (!surface.sourceFaceIndices.empty()) {
            const std::uint32_t sourceFace = surface.sourceFaceIndices[face];
            if (sourceFace == NM_INVALID_INDEX ||
                !sourceFaceIdentities.insert(sourceFace).second) {
                return fail("registered skin source face identity is invalid or duplicated");
            }
        } else if (face > std::numeric_limits<std::uint32_t>::max()) {
            return fail("registered skin source face index exceeds the retained source identity range");
        }
    }

    RegisteredSkinFatReductionSourceLabels next;
    next.registeredVertexCount = layerWidth;
    next.registeredFaceCount = surface.faces.size();
    next.sourceTetrahedronCount = shell.tetrahedra.size();
    next.sourceNodeVertexIndices.reserve(shell.femNodes.size());
    next.sourceNodeLayerIndices.reserve(shell.femNodes.size());
    for (std::uint32_t layer = 0u; layer < 3u; ++layer) {
        for (std::uint32_t vertex = 0u; vertex < layerWidth; ++vertex) {
            next.sourceNodeVertexIndices.push_back(surface.sourceVertexIndices[vertex]);
            next.sourceNodeLayerIndices.push_back(layer);
        }
    }
    next.tetrahedronRegionLabels.reserve(shell.tetrahedra.size());
    next.tetrahedronSourceFaceIndices.reserve(shell.tetrahedra.size());
    next.tetrahedronSourceDonorBodyIndices.reserve(shell.tetrahedra.size());
    next.tetrahedronSourceParentBodyIndices.reserve(shell.tetrahedra.size());
    std::vector<std::array<std::uint32_t, 2u>> tetsPerFaceLayer(surface.faces.size());

    for (const auto& tet : shell.tetrahedra) {
        FaceKey sourceFace{};
        std::array<std::uint32_t, 4u> layers{};
        std::size_t uniqueCount = 0u;
        for (std::size_t corner = 0u; corner < tet.nodes.size(); ++corner) {
            const std::uint32_t node = tet.nodes[corner];
            if (node >= shell.femNodes.size())
                return fail("registered skin/fat tetrahedron references a node outside its authored object");
            layers[corner] = node / layerWidth;
            if (layers[corner] > 2u)
                return fail("registered skin/fat tetrahedron references an unknown nodal layer");
            const std::uint32_t sourceVertex = node % layerWidth;
            if (std::find(sourceFace.begin(),
                          sourceFace.begin() + static_cast<std::ptrdiff_t>(uniqueCount),
                          sourceVertex) ==
                sourceFace.begin() + static_cast<std::ptrdiff_t>(uniqueCount)) {
                if (uniqueCount >= sourceFace.size())
                    return fail("registered prism tetrahedron does not map to one triangular source face");
                sourceFace[uniqueCount++] = sourceVertex;
            }
        }
        if (uniqueCount != 3u)
            return fail("registered prism tetrahedron does not have exactly three source-face vertices");
        std::sort(sourceFace.begin(), sourceFace.end());
        const auto faceIt = faceByVertices.find(sourceFace);
        if (faceIt == faceByVertices.end())
            return fail("registered prism tetrahedron has no exact registered source-face ancestry");
        const bool hasOuterLayer = std::find(layers.begin(), layers.end(), 0u) != layers.end();
        const bool hasInterfaceLayer = std::find(layers.begin(), layers.end(), 1u) != layers.end();
        const bool hasFatLayer = std::find(layers.begin(), layers.end(), 2u) != layers.end();
        const bool skinPrism = hasOuterLayer && hasInterfaceLayer && !hasFatLayer;
        const bool fatPrism = !hasOuterLayer && hasInterfaceLayer && hasFatLayer;
        if (skinPrism == fatPrism)
            return fail("registered prism tetrahedron crosses an invalid skin/fat layer pair");
        ++tetsPerFaceLayer[faceIt->second][skinPrism ? 0u : 1u];

        std::array<std::uint32_t, 3u> donorBodies{};
        for (std::size_t corner = 0u; corner < sourceFace.size(); ++corner)
            donorBodies[corner] = surface.donorBodyIndices[sourceFace[corner]];
        std::sort(donorBodies.begin(), donorBodies.end());
        std::uint32_t dominantBody = donorBodies[0];
        std::size_t bestCount = 0u;
        for (std::size_t begin = 0u; begin < donorBodies.size();) {
            std::size_t end = begin + 1u;
            while (end < donorBodies.size() && donorBodies[end] == donorBodies[begin]) ++end;
            const std::size_t count = end - begin;
            if (count > bestCount) {
                dominantBody = donorBodies[begin];
                bestCount = count;
            }
            begin = end;
        }
        // Sorted input and strict count comparison break ties by lowest stable body ID.
        const std::uint32_t parent = parentBodyAt(dominantBody);
        if ((parent != NM_INVALID_INDEX && parent >= sourceBodyCount) ||
            parent == dominantBody)
            return fail("dominant registered donor body has an invalid source parent identity");
        auto ancestry = std::vector<std::uint32_t>{dominantBody};
        std::set<std::uint32_t> seen{dominantBody};
        std::uint32_t cursor = parent;
        while (cursor != NM_INVALID_INDEX) {
            if (cursor >= sourceBodyCount || !seen.insert(cursor).second)
                return fail("registered Human donor-body ancestry contains an invalid index or cycle");
            ancestry.push_back(cursor);
            cursor = parentBodyAt(cursor);
        }
        const auto [ancestryIt, inserted] = next.donorBodyAncestry.emplace(dominantBody, ancestry);
        if (!inserted && ancestryIt->second != ancestry)
            return fail("registered donor-body ancestry changed while labeling source tetrahedra");
        const std::uint32_t sourceFaceIndex = surface.sourceFaceIndices.empty()
            ? static_cast<std::uint32_t>(faceIt->second)
            : surface.sourceFaceIndices[faceIt->second];
        next.tetrahedronRegionLabels.push_back(dominantBody);
        next.tetrahedronSourceFaceIndices.push_back(sourceFaceIndex);
        next.tetrahedronSourceDonorBodyIndices.push_back(dominantBody);
        next.tetrahedronSourceParentBodyIndices.push_back(parent);
    }
    for (const auto& counts : tetsPerFaceLayer) {
        if (counts[0] != 3u || counts[1] != 3u)
            return fail("registered skin source face does not own exactly three skin and three fat prism tetrahedra");
    }
    output = std::move(next);
    error.clear();
    return true;
}

} // namespace metalrobo::human::coarse_tissue
