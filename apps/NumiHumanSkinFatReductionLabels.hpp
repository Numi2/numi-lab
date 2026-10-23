#pragma once

// Source-bound anatomical labels for the registered Human skin/fat shell.
// Labels retain the original Human donor-body IDs. This is compiler input
// provenance only; it does not enable reduced coordinates in the runtime.
#include "NumiHumanCoarseTissuePreparation.hpp"
#include "NumiHumanOpticNerveTetGenAssets.hpp"
#include "numi/matter/compiled_world_partitioned_reduction.hpp"
#include "numi/matter/partitioned_reduction_mass_whitening.hpp"
#include "numi/matter/partitioned_reduction_sparse_support.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iterator>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace metalrobo::human::coarse_tissue {

struct RegisteredSkinFatReductionSourceLabels {
    // Original donor-body identity per compiled source tetrahedron.
    std::vector<std::uint32_t> tetrahedronRegionLabels;
    // Basis-only local refinement labels. Original donor and face ancestry
    // remain in the source arrays above/below and in tetrahedronSource*.
    std::vector<std::uint32_t> tetrahedronBasisRegionLabels;
    // Original registered source vertex and layer for each authored shell node.
    std::vector<std::uint32_t> sourceNodeVertexIndices;
    // Projected nodes retain INVALID direct IDs plus their exact source-face embedding.
    std::vector<std::uint32_t> sourceNodeEmbeddingFaceIndices;
    std::vector<std::array<double, 3>> sourceNodeEmbeddingBarycentric;
    std::vector<std::uint32_t> sourceNodeDonorBodyIndices;
    std::vector<std::uint32_t> sourceNodeLayerIndices;
    // Original registered skin face identity and dominant donor body for each tet.
    // Solver face identities. Optional local identities distinguish remeshed
    // children while the independent parent array retains detailed provenance.
    std::vector<std::uint32_t> tetrahedronSourceFaceIndices;
    std::vector<std::uint32_t> tetrahedronOriginalSourceFaceIndices;
    std::vector<std::uint32_t> registeredMechanicalFaceIndices;
    std::vector<std::uint32_t> tetrahedronSourceDonorBodyIndices;
    // Immediate parent for the dominant donor body; root bodies retain INVALID.
    std::vector<std::uint32_t> tetrahedronSourceParentBodyIndices;
    // Explicit child topology retains the exact parent source tetrahedron.
    // Empty on the legacy identity-topology path.
    std::vector<std::uint32_t> tetrahedronParentTetrahedronIndices;
    // Complete original donor-body ancestry, body first and root last.
    std::map<std::uint32_t, std::vector<std::uint32_t>> donorBodyAncestry;
    std::size_t registeredVertexCount = 0u;
    std::size_t registeredFaceCount = 0u;
    std::size_t sourceTetrahedronCount = 0u;
};

struct RegisteredSkinFatBasisRegionAncestry {
    std::map<std::uint32_t, std::uint32_t> donorBodyByBasisRegion;
    std::map<std::uint32_t, std::size_t> tetrahedronCountByBasisRegion;
};

// Basis-local refinements replace donor IDs in compiled region labels. Recover
// source ancestry only through the exact registered source-tet ordering.
[[nodiscard]] inline bool mapRegisteredSkinFatBasisRegionsToDonors(
    const RegisteredSkinFatReductionSourceLabels& labels,
    const std::span<const std::uint32_t> compiledBasisRegionLabels,
    const std::span<const std::uint32_t> compiledSourceTetrahedronIndices,
    const std::size_t skinTetrahedronCount,
    RegisteredSkinFatBasisRegionAncestry& output,
    std::string& error
) {
    const auto fail = [&](const char* message) {
        error = message;
        output = {};
        return false;
    };
    const auto count = labels.sourceTetrahedronCount;
    if (count != skinTetrahedronCount ||
        labels.tetrahedronRegionLabels.size() != count ||
        labels.tetrahedronBasisRegionLabels.size() != count ||
        labels.tetrahedronSourceDonorBodyIndices.size() != count ||
        compiledBasisRegionLabels.size() < count ||
        compiledSourceTetrahedronIndices.size() < count)
        return fail("compiled skin/fat basis labels do not cover registered source tetrahedra");
    RegisteredSkinFatBasisRegionAncestry next;
    for (std::size_t tet = 0u; tet < count; ++tet) {
        if (compiledSourceTetrahedronIndices[tet] != tet ||
            compiledBasisRegionLabels[tet] != labels.tetrahedronBasisRegionLabels[tet])
            return fail("compiled skin/fat basis labels lost exact source-tet order");
        const std::uint32_t donor =
            labels.tetrahedronSourceDonorBodyIndices[tet];
        const auto ancestry = labels.donorBodyAncestry.find(donor);
        if (donor == NM_INVALID_INDEX ||
            donor != labels.tetrahedronRegionLabels[tet] ||
            ancestry == labels.donorBodyAncestry.end() ||
            ancestry->second.empty() || ancestry->second.front() != donor)
            return fail("registered skin/fat source tetrahedron lost original donor ancestry");
        const std::uint32_t basisRegion = compiledBasisRegionLabels[tet];
        const auto [region, inserted] =
            next.donorBodyByBasisRegion.emplace(basisRegion, donor);
        if (!inserted && region->second != donor)
            return fail("one compiled skin/fat basis region maps to multiple donor bodies");
        ++next.tetrahedronCountByBasisRegion[basisRegion];
    }
    output = std::move(next);
    error.clear();
    return true;
}

// Deterministic spatial refinement of source-face regions. Each face's six
// prism tetrahedra keep one basis label; original donor and face ancestry stay
// in the independent source arrays. Morton chunks are local to each donor.
[[nodiscard]] inline bool buildRegisteredSkinFatBasisRegionLabels(
    const RegisteredSurface& surface,
    const numi::matter::ObjectSource& shell,
    const RegisteredSkinFatReductionSourceLabels& labels,
    std::vector<std::uint32_t>& output,
    std::string& error,
    const std::uint32_t maximumFacesPerBasisRegion,
    const std::span<const std::uint32_t> splitSourceFaceIndices,
    const std::uint32_t maximumRefinementPatchCount = 0u
) {
    constexpr std::uint32_t maximumSkinBasisRegions =
        numi::matter::kHumanPartitionedBasisMaximumRegionCount;
    constexpr std::uint32_t refinementLabelBase = 0x53460001u;
    const auto fail = [&](const char* message) {
        error = message;
        output.clear();
        return false;
    };
    if (maximumFacesPerBasisRegion == 0u ||
        labels.tetrahedronRegionLabels.size() != shell.tetrahedra.size() ||
        labels.tetrahedronSourceFaceIndices.size() != shell.tetrahedra.size() ||
        labels.tetrahedronSourceDonorBodyIndices.size() != shell.tetrahedra.size() ||
        surface.faces.empty() || surface.worldPoints.empty() ||
        (!surface.sourceFaceIndices.empty() &&
         surface.sourceFaceIndices.size() != surface.faces.size()))
        return fail("skin/fat basis refinement source-face ancestry is incomplete");

    if (!labels.registeredMechanicalFaceIndices.empty() &&
        labels.registeredMechanicalFaceIndices.size() != surface.faces.size())
        return fail("skin/fat mechanical face identities do not cover the source surface");
    std::map<std::uint32_t, std::array<double, 3>> faceCenterByIdentity;
    for (std::size_t face = 0u; face < surface.faces.size(); ++face) {
        const std::uint32_t identity = !labels.registeredMechanicalFaceIndices.empty()
            ? labels.registeredMechanicalFaceIndices[face]
            : (surface.sourceFaceIndices.empty()
                ? static_cast<std::uint32_t>(face) : surface.sourceFaceIndices[face]);
        std::array<double, 3> center{};
        for (const std::uint32_t node : surface.faces[face]) {
            if (node >= surface.worldPoints.size())
                return fail("skin/fat basis refinement face references an absent registered vertex");
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                const double value = surface.worldPoints[node][axis];
                if (!std::isfinite(value))
                    return fail("skin/fat basis refinement has a non-finite source coordinate");
                center[axis] += value / 3.0;
            }
        }
        if (!faceCenterByIdentity.emplace(identity, center).second)
            return fail("skin/fat basis refinement face identity is duplicated");
    }

    std::set<std::uint32_t> targetFaces;
    for (const std::uint32_t face : splitSourceFaceIndices) {
        if (!faceCenterByIdentity.contains(face))
            return fail("requested skin/fat subdivision source face is absent");
        if (!targetFaces.insert(face).second)
            return fail("requested skin/fat subdivision repeats a source face");
    }

    std::map<std::uint32_t, std::uint32_t> donorByFace;
    std::map<std::uint32_t, std::uint32_t> tetsPerFace;
    std::map<std::uint32_t, std::vector<std::uint32_t>> facesByDonor;
    std::set<std::uint32_t> originalDonorLabels;
    for (std::size_t tet = 0u; tet < labels.tetrahedronRegionLabels.size(); ++tet) {
        const std::uint32_t donor = labels.tetrahedronSourceDonorBodyIndices[tet];
        const std::uint32_t face = labels.tetrahedronSourceFaceIndices[tet];
        if (donor == NM_INVALID_INDEX || !faceCenterByIdentity.contains(face))
            return fail("skin/fat basis refinement lost original donor or registered face ancestry");
        const auto [entry, inserted] = donorByFace.emplace(face, donor);
        if (!inserted && entry->second != donor)
            return fail("one registered source face maps to multiple original donor bodies");
        ++tetsPerFace[face];
        originalDonorLabels.insert(donor);
        facesByDonor[donor].push_back(face);
    }
    for (const auto& [face, count] : tetsPerFace) {
        (void)face;
        if (count != 6u)
            return fail("registered source face no longer owns exactly six skin/fat prism tetrahedra");
    }

    std::map<std::uint32_t, std::uint32_t> targetCountByDonor;
    if (maximumRefinementPatchCount != 0u) {
        if (targetFaces.empty() || maximumRefinementPatchCount > maximumSkinBasisRegions ||
            maximumRefinementPatchCount > targetFaces.size())
            return fail("skin/fat refinement patch count must fit the distinct selected source faces and Human region budget");
        for (const std::uint32_t face : targetFaces) {
            const auto donor = donorByFace.find(face);
            if (donor == donorByFace.end())
                return fail("selected skin/fat refinement face has no original donor");
            ++targetCountByDonor[donor->second];
        }
        if (maximumRefinementPatchCount < targetCountByDonor.size())
            return fail("skin/fat refinement patch count cannot leave a targeted donor without a local patch");
    }

    std::map<std::uint32_t, std::uint32_t> patchesByDonor;
    for (const auto& [donor, count] : targetCountByDonor) {
        (void)count;
        patchesByDonor.emplace(donor, 1u);
    }
    std::uint32_t unassignedPatches = maximumRefinementPatchCount == 0u
        ? 0u
        : maximumRefinementPatchCount - static_cast<std::uint32_t>(patchesByDonor.size());
    while (unassignedPatches > 0u) {
        auto best = targetCountByDonor.end();
        for (auto candidate = targetCountByDonor.begin();
             candidate != targetCountByDonor.end(); ++candidate) {
            const std::uint32_t assigned = patchesByDonor.at(candidate->first);
            if (assigned >= candidate->second) continue;
            if (best == targetCountByDonor.end()) {
                best = candidate;
                continue;
            }
            const std::uint32_t bestAssigned = patchesByDonor.at(best->first);
            const std::uint64_t lhs =
                static_cast<std::uint64_t>(candidate->second) * bestAssigned;
            const std::uint64_t rhs =
                static_cast<std::uint64_t>(best->second) * assigned;
            if (lhs > rhs || (lhs == rhs && candidate->first < best->first))
                best = candidate;
        }
        if (best == targetCountByDonor.end())
            return fail("skin/fat refinement patch allocation exceeded selected source faces");
        ++patchesByDonor.at(best->first);
        --unassignedPatches;
    }

    std::map<std::uint32_t, std::uint32_t> basisLabelByFace;
    std::uint64_t totalRegionCount = 0u;
    std::uint32_t nextRefinementLabel = refinementLabelBase;
    std::set<std::uint32_t> processedTargetFaces;
    const auto allocateRefinementLabel = [&]() {
        while (originalDonorLabels.contains(nextRefinementLabel) ||
               nextRefinementLabel == NM_INVALID_INDEX) {
            if (nextRefinementLabel == NM_INVALID_INDEX - 1u)
                return NM_INVALID_INDEX;
            ++nextRefinementLabel;
        }
        if (nextRefinementLabel == NM_INVALID_INDEX - 1u)
            return NM_INVALID_INDEX;
        return nextRefinementLabel++;
    };
    const auto morton3D = [](const std::array<std::uint32_t, 3>& coordinate) {
        std::uint32_t key = 0u;
        for (std::uint32_t bit = 0u; bit < 10u; ++bit) {
            key |= ((coordinate[0] >> bit) & 1u) << (3u * bit);
            key |= ((coordinate[1] >> bit) & 1u) << (3u * bit + 1u);
            key |= ((coordinate[2] >> bit) & 1u) << (3u * bit + 2u);
        }
        return key;
    };

    for (auto& [donor, donorFaces] : facesByDonor) {
        std::sort(donorFaces.begin(), donorFaces.end());
        donorFaces.erase(std::unique(donorFaces.begin(), donorFaces.end()),
                         donorFaces.end());
        std::array<double, 3> low{
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()};
        std::array<double, 3> high{
            -std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity()};
        for (const std::uint32_t face : donorFaces) {
            const auto& center = faceCenterByIdentity.at(face);
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                low[axis] = std::min(low[axis], center[axis]);
                high[axis] = std::max(high[axis], center[axis]);
            }
        }
        struct FaceOrder {
            std::uint32_t identity;
            std::uint32_t key;
        };
        std::vector<FaceOrder> ordered;
        ordered.reserve(donorFaces.size());
        for (const std::uint32_t face : donorFaces) {
            const auto& center = faceCenterByIdentity.at(face);
            std::array<std::uint32_t, 3> quantized{};
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                const double extent = high[axis] - low[axis];
                if (!std::isfinite(extent))
                    return fail("skin/fat basis refinement coordinate extent overflowed");
                if (extent > 0.0) {
                    const double scaled = std::clamp(
                        (center[axis] - low[axis]) / extent * 1023.0,
                        0.0, 1023.0);
                    quantized[axis] = static_cast<std::uint32_t>(
                        std::llround(scaled));
                }
            }
            ordered.push_back({face, morton3D(quantized)});
        }
        std::sort(ordered.begin(), ordered.end(),
            [](const FaceOrder& a, const FaceOrder& b) {
                return a.key != b.key ? a.key < b.key :
                    a.identity < b.identity;
            });
        const std::size_t chunkCount =
            (ordered.size() - 1u) / maximumFacesPerBasisRegion + 1u;
        if (chunkCount == 0u)
            return fail("skin/fat basis refinement produced an empty donor partition");

        if (maximumRefinementPatchCount != 0u) {
            const std::uint32_t patchCount = patchesByDonor.contains(donor)
                ? patchesByDonor.at(donor) : 0u;
            const std::uint32_t targetCount = targetCountByDonor.contains(donor)
                ? targetCountByDonor.at(donor) : 0u;
            std::vector<std::size_t> untargetedPerChunk(chunkCount, 0u);
            for (std::size_t position = 0u; position < ordered.size(); ++position) {
                if (!targetFaces.contains(ordered[position].identity))
                    ++untargetedPerChunk[position / maximumFacesPerBasisRegion];
            }
            std::size_t emittedBaseRegions = 0u;
            for (const std::size_t count : untargetedPerChunk)
                emittedBaseRegions += count != 0u ? 1u : 0u;
            totalRegionCount += emittedBaseRegions + patchCount;
            if (totalRegionCount > maximumSkinBasisRegions)
                return fail("spatial skin/fat refinement exceeds the compiled Human basis region budget");

            std::vector<std::uint32_t> baseLabels(chunkCount, NM_INVALID_INDEX);
            for (std::size_t chunk = 0u; chunk < chunkCount; ++chunk) {
                if (untargetedPerChunk[chunk] == 0u) continue;
                const std::uint32_t label = chunkCount == 1u
                    ? donor : allocateRefinementLabel();
                if (label == NM_INVALID_INDEX)
                    return fail("spatial skin/fat refinement exhausted its deterministic label range");
                baseLabels[chunk] = label;
            }
            std::vector<std::uint32_t> patchLabels;
            patchLabels.reserve(patchCount);
            for (std::uint32_t patch = 0u; patch < patchCount; ++patch) {
                const std::uint32_t label = allocateRefinementLabel();
                if (label == NM_INVALID_INDEX)
                    return fail("spatial skin/fat refinement exhausted its deterministic label range");
                patchLabels.push_back(label);
            }
            std::uint32_t targetOrdinal = 0u;
            for (std::size_t position = 0u; position < ordered.size(); ++position) {
                const std::uint32_t face = ordered[position].identity;
                std::uint32_t label = NM_INVALID_INDEX;
                if (targetFaces.contains(face)) {
                    if (patchCount == 0u || targetCount == 0u)
                        return fail("selected skin/fat refinement face lost its donor patch allocation");
                    const std::uint64_t scaledOrdinal =
                        static_cast<std::uint64_t>(targetOrdinal) * patchCount;
                    const std::uint32_t patch = static_cast<std::uint32_t>(
                        std::min<std::uint64_t>(patchCount - 1u,
                                                scaledOrdinal / targetCount));
                    label = patchLabels[patch];
                    ++targetOrdinal;
                    processedTargetFaces.insert(face);
                } else {
                    label = baseLabels[position / maximumFacesPerBasisRegion];
                    if (label == NM_INVALID_INDEX)
                        return fail("skin/fat base partition emitted an unlabelled source face");
                }
                if (!basisLabelByFace.emplace(face, label).second)
                    return fail("spatial skin/fat refinement assigned one source face more than once");
            }
            if (targetOrdinal != targetCount)
                return fail("skin/fat donor patch allocation did not cover every selected face");
            continue;
        }

        std::vector<std::vector<std::size_t>> targetedPositions(chunkCount);
        for (std::size_t position = 0u; position < ordered.size(); ++position) {
            if (targetFaces.contains(ordered[position].identity)) {
                targetedPositions[position / maximumFacesPerBasisRegion].push_back(position);
                processedTargetFaces.insert(ordered[position].identity);
            }
        }
        totalRegionCount += chunkCount;
        for (const auto& positions : targetedPositions) {
            if (positions.size() == 1u) ++totalRegionCount;
            else if (positions.size() > 1u) totalRegionCount += positions.size() - 1u;
        }
        if (totalRegionCount > maximumSkinBasisRegions)
            return fail("spatial skin/fat refinement exceeds the compiled Human basis region budget");

        for (std::size_t chunk = 0u; chunk < chunkCount; ++chunk) {
            const std::size_t begin = chunk * maximumFacesPerBasisRegion;
            const std::size_t finish = std::min(
                ordered.size(), begin + maximumFacesPerBasisRegion);
            if (begin >= finish)
                return fail("spatial skin/fat refinement emitted an empty Morton chunk");
            const auto& targets = targetedPositions[chunk];
            if (targets.empty()) {
                const std::uint32_t label = chunkCount == 1u
                    ? donor : allocateRefinementLabel();
                if (label == NM_INVALID_INDEX)
                    return fail("spatial skin/fat refinement exhausted its deterministic label range");
                for (std::size_t i = begin; i < finish; ++i)
                    basisLabelByFace.emplace(ordered[i].identity, label);
                continue;
            }

            std::vector<std::size_t> boundaries{begin};
            if (targets.size() == 1u) {
                const std::size_t middle = begin + (finish - begin) / 2u;
                if (middle == begin || middle == finish)
                    return fail("requested skin/fat source-face region cannot be subdivided into two nonempty bins");
                boundaries.push_back(middle);
            } else {
                for (std::size_t i = 0u; i + 1u < targets.size(); ++i) {
                    const std::size_t left = targets[i];
                    const std::size_t right = targets[i + 1u];
                    const std::size_t split = left + (right - left + 1u) / 2u;
                    if (split <= boundaries.back() || split >= finish)
                        return fail("targeted skin/fat subdivisions cannot form nonempty Morton bins");
                    boundaries.push_back(split);
                }
            }
            boundaries.push_back(finish);
            for (std::size_t bin = 0u; bin + 1u < boundaries.size(); ++bin) {
                const std::uint32_t label = allocateRefinementLabel();
                if (label == NM_INVALID_INDEX)
                    return fail("spatial skin/fat refinement exhausted its deterministic label range");
                for (std::size_t i = boundaries[bin]; i < boundaries[bin + 1u]; ++i)
                    basisLabelByFace.emplace(ordered[i].identity, label);
            }
        }
    }
    if (processedTargetFaces.size() != targetFaces.size())
        return fail("requested skin/fat subdivision source face has no donor partition");
    if (basisLabelByFace.size() != faceCenterByIdentity.size())
        return fail("spatial skin/fat refinement did not assign every registered source face exactly once");

    output.resize(labels.tetrahedronRegionLabels.size());
    for (std::size_t tet = 0u; tet < output.size(); ++tet) {
        const auto label = basisLabelByFace.find(
            labels.tetrahedronSourceFaceIndices[tet]);
        if (label == basisLabelByFace.end())
            return fail("spatial skin/fat refinement lost a source tetrahedron face");
        output[tet] = label->second;
    }
    error.clear();
    return true;
}

[[nodiscard]] inline bool buildRegisteredSkinFatBasisRegionLabels(
    const RegisteredSurface& surface,
    const numi::matter::ObjectSource& shell,
    const RegisteredSkinFatReductionSourceLabels& labels,
    std::vector<std::uint32_t>& output,
    std::string& error,
    const std::uint32_t maximumFacesPerBasisRegion = 36u,
    const std::uint32_t splitSourceFaceIndex = NM_INVALID_INDEX
) {
    std::array<std::uint32_t, 1u> target{splitSourceFaceIndex};
    const std::span<const std::uint32_t> targets =
        splitSourceFaceIndex == NM_INVALID_INDEX
            ? std::span<const std::uint32_t>{}
            : std::span<const std::uint32_t>(target);
    return buildRegisteredSkinFatBasisRegionLabels(
        surface, shell, labels, output, error,
        maximumFacesPerBasisRegion, targets);
}

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
        labels.tetrahedronBasisRegionLabels.size() != sourceShell.tetrahedra.size() ||
        labels.tetrahedronSourceFaceIndices.size() != sourceShell.tetrahedra.size() ||
        labels.tetrahedronSourceDonorBodyIndices.size() != sourceShell.tetrahedra.size() ||
        labels.tetrahedronSourceParentBodyIndices.size() != sourceShell.tetrahedra.size() ||
        (!labels.tetrahedronParentTetrahedronIndices.empty() &&
         labels.tetrahedronParentTetrahedronIndices.size() != sourceShell.tetrahedra.size()))
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


// Source-bound, longitudinal partitions for the two pinned TetGen optic nerves.
// These are deterministic geometric bins derived from the exact source mesh,
// not authored clinical segment labels.
struct RegisteredOpticNerveReductionSourceLabels {
    std::vector<std::uint32_t> tetrahedronRegionLabels;
    std::vector<std::uint32_t> sourceNodeVertexIndices;
    std::vector<std::uint32_t> tetrahedronSourceIndices;
    std::vector<std::uint32_t> boundarySourceFaceIndices;
    std::vector<std::uint32_t> boundarySourceNodeIndices;
    // Left optic tet 92 and its source-mesh one-ring stay on the fine complement.
    std::vector<std::uint32_t> witnessSourceNodeIndices;
    std::string member;
    std::string meshSHA256;
    std::string payloadSHA256;
};

[[nodiscard]] inline bool buildRegisteredOpticNerveReductionSourceLabels(
    const numi::human::optic_nerve::tetgen_assets::AssetView& asset,
    const numi::matter::ObjectSource& source,
    RegisteredOpticNerveReductionSourceLabels& output,
    std::string& error,
    const std::uint32_t longitudinalBinCount = 5u
) {
    using namespace numi::human::optic_nerve::tetgen_assets;
    const auto fail = [&](const char* message) {
        error = message;
        return false;
    };
    const auto samePinnedAsset = [&](const AssetView& pinned) {
        return asset.member == pinned.member &&
            asset.anatomicalConcept == pinned.anatomicalConcept &&
            asset.meshSHA256 == pinned.meshSHA256 &&
            asset.payloadSHA256 == pinned.payloadSHA256 &&
            asset.sourceVertexCount == pinned.sourceVertexCount &&
            asset.sourceFaceCount == pinned.sourceFaceCount &&
            asset.sourceVolumeM3 == pinned.sourceVolumeM3 &&
            asset.donorLocalPoints.data() == pinned.donorLocalPoints.data() &&
            asset.donorLocalPoints.size() == pinned.donorLocalPoints.size() &&
            asset.sourceVertex.data() == pinned.sourceVertex.data() &&
            asset.sourceVertex.size() == pinned.sourceVertex.size() &&
            asset.tetrahedra.data() == pinned.tetrahedra.data() &&
            asset.tetrahedra.size() == pinned.tetrahedra.size() &&
            asset.boundaryFaces.data() == pinned.boundaryFaces.data() &&
            asset.boundaryFaces.size() == pinned.boundaryFaces.size() &&
            asset.boundarySourceFace.data() == pinned.boundarySourceFace.data() &&
            asset.boundarySourceFace.size() == pinned.boundarySourceFace.size();
    };
    if ((!samePinnedAsset(kLeftAsset) && !samePinnedAsset(kRightAsset)) ||
        asset.donorLocalPoints.size() < 4u ||
        asset.sourceVertex.size() != asset.donorLocalPoints.size() ||
        asset.tetrahedra.empty() ||
        asset.boundaryFaces.empty() ||
        asset.boundarySourceFace.size() != asset.boundaryFaces.size() ||
        source.representation != numi::matter::Representation::fem ||
        source.femNodes.size() != asset.donorLocalPoints.size() ||
        source.tetrahedra.size() != asset.tetrahedra.size())
        return fail("optic reduction source differs from a complete pinned bilateral TetGen mesh");
    for (std::size_t i = 0u; i < asset.tetrahedra.size(); ++i) {
        if (source.tetrahedra[i].nodes != asset.tetrahedra[i])
            return fail("optic reduction source tetrahedron order/connectivity differs from pinned ancestry");
    }
    std::set<std::uint32_t> boundaryNodes;
    std::set<std::uint32_t> boundaryFaces;
    for (std::size_t face = 0u; face < asset.boundaryFaces.size(); ++face) {
        if (asset.boundarySourceFace[face] == NM_INVALID_INDEX ||
            !boundaryFaces.insert(asset.boundarySourceFace[face]).second)
            return fail("optic reduction source boundary face ancestry is invalid or duplicated");
        for (const std::uint32_t node : asset.boundaryFaces[face]) {
            if (node >= asset.donorLocalPoints.size())
                return fail("optic reduction source boundary face references an absent vertex");
            boundaryNodes.insert(node);
        }
    }
    std::vector<std::uint32_t> compiledBoundary = source.femContactNodes;
    std::sort(compiledBoundary.begin(), compiledBoundary.end());
    if (std::adjacent_find(compiledBoundary.begin(), compiledBoundary.end()) !=
            compiledBoundary.end() ||
        std::vector<std::uint32_t>(boundaryNodes.begin(), boundaryNodes.end()) !=
            compiledBoundary)
        return fail("optic reduction fine contact rows differ from the pinned source boundary vertex set");

    std::set<std::uint32_t> witnessFineNodes;
    if (asset.member == "FJ1772") {
        constexpr std::uint32_t witnessTetIndex = 92u;
        if (witnessTetIndex >= asset.tetrahedra.size())
            return fail("left optic retained fine witness tetrahedron 92 is absent");
        std::set<std::uint32_t> witnessVertices;
        for (const std::uint32_t node : asset.tetrahedra[witnessTetIndex]) {
            if (node >= asset.donorLocalPoints.size() ||
                !witnessVertices.insert(node).second)
                return fail("left optic tet 92 has invalid or repeated source vertices");
        }
        if (witnessVertices.size() != 4u)
            return fail("left optic tet 92 does not have four distinct source vertices");
        for (const auto& tet : asset.tetrahedra) {
            bool incident = false;
            for (const std::uint32_t node : tet)
                incident = incident || witnessVertices.contains(node);
            if (!incident) continue;
            for (const std::uint32_t node : tet) {
                if (node >= asset.donorLocalPoints.size())
                    return fail("left optic tet 92 one-ring references an absent source vertex");
                witnessFineNodes.insert(node);
            }
        }
        for (const std::uint32_t node : witnessVertices)
            if (!witnessFineNodes.contains(node))
                return fail("left optic tet 92 witness vertex was lost from its fine one-ring");
    }

    std::array<double, 3> center{};
    for (const auto& point : asset.donorLocalPoints) {
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            if (!std::isfinite(point[axis]))
                return fail("optic reduction source point is non-finite");
            center[axis] += point[axis] / double(asset.donorLocalPoints.size());
        }
    }
    std::array<std::array<double, 3>, 3> covariance{};
    for (const auto& point : asset.donorLocalPoints) {
        for (std::size_t r = 0u; r < 3u; ++r)
            for (std::size_t c = 0u; c < 3u; ++c)
                covariance[r][c] += (point[r] - center[r]) *
                    (point[c] - center[c]);
    }
    std::array<std::array<double, 3>, 3> eigenvectors{{
        {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}};
    for (unsigned sweep = 0u; sweep < 24u; ++sweep) {
        std::size_t p = 0u, q = 1u;
        double largest = std::abs(covariance[p][q]);
        for (const auto pair : {std::array<std::size_t, 2>{0u, 2u},
                                std::array<std::size_t, 2>{1u, 2u}}) {
            if (std::abs(covariance[pair[0]][pair[1]]) > largest) {
                p = pair[0]; q = pair[1];
                largest = std::abs(covariance[p][q]);
            }
        }
        if (largest <= 1.0e-14 * std::max({std::abs(covariance[0][0]),
                std::abs(covariance[1][1]), std::abs(covariance[2][2]), 1.0e-30}))
            break;
        const double tau = (covariance[q][q] - covariance[p][p]) /
            (2.0 * covariance[p][q]);
        const double tangent = std::copysign(1.0, tau) /
            (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
        const double sine = tangent * cosine;
        const double app = covariance[p][p], aqq = covariance[q][q];
        const double apq = covariance[p][q];
        covariance[p][p] = app - tangent * apq;
        covariance[q][q] = aqq + tangent * apq;
        covariance[p][q] = covariance[q][p] = 0.0;
        for (std::size_t k = 0u; k < 3u; ++k) {
            if (k == p || k == q) continue;
            const double akp = covariance[k][p], akq = covariance[k][q];
            covariance[k][p] = covariance[p][k] = cosine * akp - sine * akq;
            covariance[k][q] = covariance[q][k] = sine * akp + cosine * akq;
        }
        for (std::size_t k = 0u; k < 3u; ++k) {
            const double vkp = eigenvectors[k][p], vkq = eigenvectors[k][q];
            eigenvectors[k][p] = cosine * vkp - sine * vkq;
            eigenvectors[k][q] = sine * vkp + cosine * vkq;
        }
    }
    std::array<std::size_t, 3> order{0u, 1u, 2u};
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return covariance[a][a] > covariance[b][b];
    });
    if (!(covariance[order[0]][order[0]] >
          1.15 * covariance[order[1]][order[1]]) ||
        !std::isfinite(covariance[order[0]][order[0]]))
        return fail("optic source mesh has no stable longitudinal principal axis for local partitioning");
    std::array<double, 3> axis{};
    for (std::size_t k = 0u; k < 3u; ++k)
        axis[k] = eigenvectors[k][order[0]];
    std::size_t signAxis = 0u;
    for (std::size_t k = 1u; k < 3u; ++k)
        if (std::abs(axis[k]) > std::abs(axis[signAxis])) signAxis = k;
    if (axis[signAxis] < 0.0)
        for (double& value : axis) value = -value;

    struct TetOrder { double projection; double volume; std::uint32_t index; };
    std::vector<TetOrder> ordered;
    ordered.reserve(asset.tetrahedra.size());
    std::vector<std::array<double, 3>> points(
        asset.donorLocalPoints.begin(), asset.donorLocalPoints.end());
    double totalVolume = 0.0;
    for (std::uint32_t tetIndex = 0u;
         tetIndex < asset.tetrahedra.size(); ++tetIndex) {
        const auto& tet = asset.tetrahedra[tetIndex];
        const double volume = numi::matter::partitioned_reduction_detail::
            tetrahedronVolume(points, tet);
        std::array<double, 3> centroid{};
        for (const std::uint32_t node : tet) {
            if (node >= points.size())
                return fail("optic reduction source tet references an absent vertex");
            for (std::size_t k = 0u; k < 3u; ++k)
                centroid[k] += points[node][k] * 0.25;
        }
        const double projection = centroid[0] * axis[0] +
            centroid[1] * axis[1] + centroid[2] * axis[2];
        if (!std::isfinite(volume) || !(volume > 0.0) ||
            !std::isfinite(projection))
            return fail("optic reduction source tet has non-positive volume or non-finite longitudinal coordinate");
        ordered.push_back({projection, volume, tetIndex});
        totalVolume += volume;
    }
    if (!std::isfinite(totalVolume) || !(totalVolume > 0.0))
        return fail("optic reduction source has no positive total tetrahedron volume");
    std::sort(ordered.begin(), ordered.end(), [](const TetOrder& a, const TetOrder& b) {
        return a.projection != b.projection ? a.projection < b.projection :
            a.index < b.index;
    });
    // Default five-bin layout preserves the original first-quartile split.
    // The opt-in ten-bin layout bisects each existing volume bin, preserving
    // every old boundary while adding one affine block per optic subregion.
    constexpr std::array<double, 5> defaultCumulativeFractions{
        0.125, 0.25, 0.5, 0.75, 1.0};
    constexpr std::array<double, 10> subdividedCumulativeFractions{
        0.0625, 0.125, 0.1875, 0.25, 0.375,
        0.5, 0.625, 0.75, 0.875, 1.0};
    if (longitudinalBinCount != 5u && longitudinalBinCount != 10u)
        return fail("optic longitudinal basis must use the unchanged five-bin or opt-in ten-bin layout");
    const std::span<const double> cumulativeFractions =
        longitudinalBinCount == 5u
            ? std::span<const double>(defaultCumulativeFractions)
            : std::span<const double>(subdividedCumulativeFractions);
    const std::uint32_t bins = longitudinalBinCount;
    const std::uint32_t labelBase = asset.member == "FJ1772"
        ? 0xF1772001u : 0xF1819001u;
    RegisteredOpticNerveReductionSourceLabels next;
    next.member = std::string(asset.member);
    next.meshSHA256 = std::string(asset.meshSHA256);
    next.payloadSHA256 = std::string(asset.payloadSHA256);
    next.tetrahedronRegionLabels.assign(asset.tetrahedra.size(), NM_INVALID_INDEX);
    next.sourceNodeVertexIndices.assign(asset.sourceVertex.begin(), asset.sourceVertex.end());
    next.tetrahedronSourceIndices.resize(asset.tetrahedra.size());
    next.boundarySourceFaceIndices.assign(asset.boundarySourceFace.begin(),
                                               asset.boundarySourceFace.end());
    next.boundarySourceNodeIndices.assign(boundaryNodes.begin(), boundaryNodes.end());
    next.witnessSourceNodeIndices.assign(
        witnessFineNodes.begin(), witnessFineNodes.end());
    double cumulative = 0.0;
    std::size_t cursor = 0u;
    for (std::uint32_t bin = 0u; bin < bins; ++bin) {
        const double upper = totalVolume * cumulativeFractions[bin];
        if (cursor >= ordered.size())
            return fail("optic longitudinal partition has fewer nonempty tet groups than bins");
        do {
            const auto& tet = ordered[cursor];
            next.tetrahedronRegionLabels[tet.index] = labelBase + bin;
            next.tetrahedronSourceIndices[tet.index] = tet.index;
            cumulative += tet.volume;
            ++cursor;
        } while (cursor < ordered.size() &&
                 cumulative + 0.5 * ordered[cursor].volume < upper);
    }
    if (cursor != ordered.size() ||
        std::find(next.tetrahedronRegionLabels.begin(),
                  next.tetrahedronRegionLabels.end(), NM_INVALID_INDEX) !=
            next.tetrahedronRegionLabels.end())
        return fail("optic longitudinal source partition did not assign every source tet exactly once");
    output = std::move(next);
    error.clear();
    return true;
}

struct RegisteredOpticNerveReductionObject {
    std::uint32_t objectIndex = NM_INVALID_INDEX;
    const numi::matter::ObjectSource* source = nullptr;
    const numi::human::optic_nerve::tetgen_assets::AssetView* asset = nullptr;
};

// This opt-in selects optic blocks for the existing full-coordinate
// preconditioner. It never changes the physical operator or certificate.
[[nodiscard]] inline bool parseHumanOpticOnlyPartitionedPreconditionerSetting(
    const char* setting,
    bool& enabled,
    std::string& error
) {
    enabled = false;
    if (setting == nullptr) {
        error.clear();
        return true;
    }
    const std::string_view value(setting);
    if (value == "0") {
        error.clear();
        return true;
    }
    if (value == "1") {
        enabled = true;
        error.clear();
        return true;
    }
    error = "NM_FEM_HUMAN_PARTITIONED_OPTIC_ONLY must be exactly 0 or 1";
    return false;
}

[[nodiscard]] inline bool parseHumanSparseCoefficientPreconditionerSetting(
    const char* setting,
    bool& enabled,
    std::string& error
) {
    enabled = false;
    if (setting == nullptr || std::string_view(setting) == "0") {
        error.clear();
        return true;
    }
    if (std::string_view(setting) == "1") {
        enabled = true;
        error.clear();
        return true;
    }
    error = "NM_FEM_HUMAN_PARTITIONED_SPARSE_BASIS must be exactly 0 or 1";
    return false;
}

// Select the complete bilateral five- or ten-bin optic family. Skin/fat
// labels are intentionally ignored; callers retain their complete source-row
// map and simply leave those rows on the full-coordinate fine path.
[[nodiscard]] inline bool selectHumanOpticOnlyPartitionedRegionLabels(
    const std::span<const std::uint32_t> availableLabels,
    const std::uint32_t binCount,
    std::vector<std::uint32_t>& selectedLabels,
    std::string& error
) {
    selectedLabels.clear();
    if (binCount != 5u && binCount != 10u) {
        error = "optic-only preconditioner requires the registered five- or ten-bin layout";
        return false;
    }
    std::array<std::uint8_t, 10u> leftSeen{};
    std::array<std::uint8_t, 10u> rightSeen{};
    const std::uint32_t leftBase = 0xF1772001u;
    const std::uint32_t rightBase = 0xF1819001u;
    for (const std::uint32_t label : availableLabels) {
        if (label >= leftBase && label < leftBase + 10u) {
            if (label >= leftBase + binCount) {
                error = "left optic source bin count differs from the selected payload layout";
                return false;
            }
            auto& seen = leftSeen[label - leftBase];
            if (seen != 0u) {
                error = "optic-only source repeats a left optic region label";
                return false;
            }
            seen = 1u;
        } else if (label >= rightBase && label < rightBase + 10u) {
            if (label >= rightBase + binCount) {
                error = "right optic source bin count differs from the selected payload layout";
                return false;
            }
            auto& seen = rightSeen[label - rightBase];
            if (seen != 0u) {
                error = "optic-only source repeats a right optic region label";
                return false;
            }
            seen = 1u;
        }
    }
    for (std::uint32_t bin = 0u; bin < binCount; ++bin) {
        if (leftSeen[bin] == 0u || rightSeen[bin] == 0u) {
            error = "optic-only source is missing a registered left or right longitudinal region";
            return false;
        }
        selectedLabels.push_back(leftBase + bin);
        selectedLabels.push_back(rightBase + bin);
    }
    std::sort(selectedLabels.begin(), selectedLabels.end());
    error.clear();
    return true;
}

[[nodiscard]] inline bool appendHybridResidualWitnessTraceLine(
    const std::string_view line,
    std::vector<numi::matter::PartitionedReductionResidualWitness>& witnesses,
    std::string& error
) {
    if (line.find("\"hybrid_physical_residual_witness\"") == std::string_view::npos)
        return true;
    if (witnesses.size() >= 2u) {
        error = "hybrid residual refinement accepts at most two sequential final-cycle witnesses";
        return false;
    }
    numi::matter::PartitionedReductionResidualWitness parsed;
    if (!numi::matter::parsePartitionedReductionResidualWitnessTrace(
            line, parsed, error)) return false;
    if (!witnesses.empty() &&
        (parsed.sourceWorldPhysicsFingerprint !=
             witnesses.front().sourceWorldPhysicsFingerprint ||
         parsed.objectIndex != witnesses.front().objectIndex)) {
        error = "sequential residual witnesses changed their source physics or object owner";
        return false;
    }
    witnesses.push_back(std::move(parsed));
    error.clear();
    return true;
}

[[nodiscard]] inline bool loadHybridResidualWitnessTrace(
    const char* path,
    std::vector<numi::matter::PartitionedReductionResidualWitness>& witnesses,
    std::string& error
) {
    if (path == nullptr || path[0] == '\0') {
        error = "hybrid residual refinement trace path is empty";
        return false;
    }
    std::ifstream input(path);
    if (!input) {
        error = "cannot open the requested hybrid residual witness trace";
        return false;
    }
    std::string line;
    std::size_t totalBytes = 0u;
    witnesses.clear();
    while (std::getline(input, line)) {
        totalBytes += line.size() + 1u;
        if (totalBytes > 64u * 1024u * 1024u || line.size() > 1024u * 1024u) {
            error = "hybrid residual witness trace exceeds its bounded input size";
            return false;
        }
        if (!appendHybridResidualWitnessTraceLine(line, witnesses, error))
            return false;
    }
    if (witnesses.empty()) {
        error = "hybrid residual witness trace has no final witness record";
        return false;
    }
    error.clear();
    return true;
}

[[nodiscard]] inline bool requestedHybridResidualReductionFactor(
    const std::uint32_t defaultFactor,
    std::uint32_t& requestedFactor,
    std::string& error
) {
    requestedFactor = defaultFactor;
    const char* setting = std::getenv("NM_FEM_HYBRID_RESIDUAL_REFINEMENT_MIN_RATIO");
    if (setting == nullptr) return true;
    std::uint32_t parsed = 0u;
    const char* end = setting;
    while (*end != '\0') ++end;
    const auto result = std::from_chars(setting, end, parsed);
    if (result.ec != std::errc{} || result.ptr != end || parsed < 2u || parsed > 64u) {
        error = "NM_FEM_HYBRID_RESIDUAL_REFINEMENT_MIN_RATIO must be an integer from 2 through 64";
        return false;
    }
    requestedFactor = parsed;
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
    const bool massWhiten,
    std::string& error,
    const bool keepLayerZeroFine = false,
    const std::span<const std::uint32_t> residualFineGlobalFEMNodes = {},
    const std::uint32_t minimumCoordinateReductionFactor = 4u,
    const bool promoteRankDeficientRegions = false
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

    if (minimumCoordinateReductionFactor < 2u ||
        minimumCoordinateReductionFactor > 64u)
        return fail("skin/fat requested reduction target is outside the checked range");
    numi::matter::CompiledWorldPartitionedReductionOptions options;
    options.minimumCoordinateReductionFactor = minimumCoordinateReductionFactor;
    options.promoteRankDeficientRegionsToFine =
        promoteRankDeficientRegions || !residualFineGlobalFEMNodes.empty();
    options.maximumModeCount = numi::matter::kHumanPartitionedBasisMaximumModeCount;
    options.explicitAuthoredTetrahedronRegionLabels =
        std::span<const std::uint32_t>(labels.tetrahedronBasisRegionLabels);
    options.explicitRegionLabelObjectIndex = objectIndex;
    options.explicitRegionLabelWorldPhysicsFingerprint = world.physicsFingerprint;
    std::vector<std::uint32_t> layerZeroFineRows;
    if (keepLayerZeroFine) {
        if (labels.sourceNodeLayerIndices.size() != sourceShell.femNodes.size() ||
            labels.registeredVertexCount == 0u)
            return fail("layer-zero fine policy lacks exact source-node layer ancestry");
        const auto& object = world.objects[objectIndex];
        layerZeroFineRows.reserve(labels.registeredVertexCount);
        for (std::size_t local = 0u;
             local < labels.sourceNodeLayerIndices.size(); ++local) {
            if (labels.sourceNodeLayerIndices[local] > 2u)
                return fail("skin/fat source node has an invalid layer identity");
            if (labels.sourceNodeLayerIndices[local] == 0u) {
                if (local > std::numeric_limits<std::uint32_t>::max() ||
                    object.stateOffset > std::numeric_limits<std::uint32_t>::max() -
                        static_cast<std::uint32_t>(local))
                    return fail("layer-zero fine source row exceeds the compiled FEM index range");
                layerZeroFineRows.push_back(object.stateOffset +
                    static_cast<std::uint32_t>(local));
            }
        }
        if (layerZeroFineRows.size() != labels.registeredVertexCount)
            return fail("layer-zero fine policy does not cover exactly one registered surface layer");
    }
    std::vector<std::uint32_t> fineRows = layerZeroFineRows;
    std::set<std::uint32_t> selectedResidualRows;
    for (const std::uint32_t global : residualFineGlobalFEMNodes) {
        if (!selectedResidualRows.insert(global).second ||
            std::find(layerZeroFineRows.begin(), layerZeroFineRows.end(), global) !=
                layerZeroFineRows.end())
            return fail("residual refinement repeats or overlaps a protected skin row");
        fineRows.push_back(global);
    }
    std::sort(fineRows.begin(), fineRows.end());
    if (!fineRows.empty())
        options.fineCoordinateOnlyGlobalFEMNodes =
            std::span<const std::uint32_t>(fineRows);
    Candidate candidate =
        numi::matter::buildCompiledWorldPartitionedReductionBasis(
            world, objectIndex, options);
    if (!candidate.valid()) {
        error = candidate.error.empty()
            ? "compiled skin/fat basis failed its completeness contract"
            : candidate.error;
        return candidate;
    }
    if (candidate.tetrahedronRegionLabels != labels.tetrahedronBasisRegionLabels ||
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
    if (massWhiten) {
        std::vector<double> sourceMass;
        sourceMass.reserve(candidate.globalFEMNodeIndices.size());
        for (const std::uint32_t globalNode : candidate.globalFEMNodeIndices) {
            if (globalNode >= world.fem.nodes.size())
                return fail("mass-whitening source row escaped the compiled FEM mass array");
            sourceMass.push_back(static_cast<double>(
                world.fem.nodes[globalNode].positionAndMass.w));
        }
        const auto whitening =
            numi::matter::massWhitenPartitionedReductionBasis(
                candidate.basis, sourceMass);
        if (!whitening.valid())
            return fail("compiled skin/fat mass whitening failed: " + whitening.error);
    }
    if (!installPayload) {
        error.clear();
        return candidate;
    }

    if (candidate.basis.regions.empty() ||
        candidate.basis.regions.size() > numi::matter::kHumanPartitionedBasisMaximumRegionCount ||
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
    payload.minimumCoordinateReductionFactor = minimumCoordinateReductionFactor;
    payload.sourceWorldPhysicsFingerprint = world.physicsFingerprint;
    payload.globalFEMNodeIndices = candidate.globalFEMNodeIndices;
    const std::uint64_t payloadValues =
        static_cast<std::uint64_t>(payload.modeCount) * payload.basisNodeCount;
    if (payloadValues >
            numi::matter::kHumanPartitionedBasisCoefficientByteBudget /
                sizeof(nm_float4) ||
        !numi::matter::humanPartitionedBasisCoefficientExtentWithinBudget(
            payload.modeCount, payload.basisNodeCount))
        return fail("skin/fat Human basis exceeds its checked 384 MiB coefficient budget");
    payload.regionLabels.reserve(payload.regionCount);
    payload.blockModeIndices.reserve(payload.modeCount);
    payload.basisValues.reserve(static_cast<std::size_t>(payloadValues));
    if (payload.regionCount == 0u || payload.modeCount > numi::matter::kHumanPartitionedBasisMaximumModeCount ||
        static_cast<std::uint64_t>(payload.modeCount) *
                minimumCoordinateReductionFactor > payload.freeCoordinateCount)
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
    if (massWhiten) {
        // Rebuild deterministic row/mode support from the exact transformed
        // Float32 payload. Runtime repeats this same builder after handoff.
        const auto support = numi::matter::buildPartitionedReductionSparseSupport(
            std::span<const nm_float4>(payload.basisValues),
            payload.basisNodeCount, payload.modeCount,
            std::span<const std::uint32_t>(payload.blockModeIndices),
            payload.regionCount, payload.blockModeCount);
        if (!support.valid())
            return fail("mass-whitened Human payload sparse support is invalid: " +
                        support.error);
    }
    payload.enabled = true;
    world.humanPartitionedPreconditioner = std::move(payload);
    error.clear();
    return candidate;
}

// The Human shell has three nodal layers [outer skin, bonded interface, deep
// fat]. Each triangular registered source face owns three prism tets per
// layer. A tet's four nodes therefore map modulo layerWidth to exactly the
// three source vertices of its registered face.

[[nodiscard]] inline bool verifyRegisteredOpticNerveCompiledSource(
    const numi::matter::CompiledWorld& world,
    const RegisteredOpticNerveReductionObject& nerve,
    const RegisteredOpticNerveReductionSourceLabels& labels,
    std::string& error
) {
    using namespace numi::human::optic_nerve::tetgen_assets;
    const auto fail = [&](const char* message) { error = message; return false; };
    if (nerve.source == nullptr || nerve.asset == nullptr ||
        nerve.objectIndex >= world.objects.size())
        return fail("optic reduction handoff has a missing source or compiled object");
    const auto& object = world.objects[nerve.objectIndex];
    const auto& source = *nerve.source;
    const auto& asset = *nerve.asset;
    if (object.representation != NM_REPRESENTATION_FEM ||
        object.stateCount != source.femNodes.size() ||
        object.elementCount != source.tetrahedra.size() ||
        object.stateCount != asset.donorLocalPoints.size() ||
        object.elementCount != asset.tetrahedra.size() ||
        object.topologyGeneration == 0u)
        return fail("compiled optic object dimensions differ from pinned source ancestry");
    const std::uint64_t nodeEnd = std::uint64_t(object.stateOffset) + object.stateCount;
    const std::uint64_t tetEnd = std::uint64_t(object.elementOffset) + object.elementCount;
    if (nodeEnd > world.fem.nodes.size() ||
        nodeEnd > world.fem.topologyNodes.size() ||
        tetEnd > world.fem.tetrahedra.size())
        return fail("compiled optic object escapes its FEM arena");
    if (labels.sourceNodeVertexIndices.size() != object.stateCount ||
        labels.tetrahedronRegionLabels.size() != object.elementCount ||
        labels.tetrahedronSourceIndices.size() != object.elementCount)
        return fail("optic source labels no longer cover the exact source mesh");
    for (std::uint32_t local = 0u; local < object.stateCount; ++local) {
        const auto& topology = world.fem.topologyNodes[object.stateOffset + local];
        if (topology.identity.x != local || topology.identity.y != nerve.objectIndex ||
            topology.identity.z != object.topologyGeneration ||
            (topology.identity.w & NM_TOPOLOGY_ACTIVE) == 0u)
            return fail("compiled optic node order or object ancestry differs from the pinned source");
        const auto& cooked = world.fem.nodes[object.stateOffset + local].positionAndMass;
        const auto& authored = source.femNodes[local];
        if (cooked.x != static_cast<float>(authored[0]) ||
            cooked.y != static_cast<float>(authored[1]) ||
            cooked.z != static_cast<float>(authored[2]))
            return fail("compiled optic Float32 geometry differs from the registered source object");
    }
    for (std::uint32_t local = 0u; local < object.elementCount; ++local) {
        const auto& cooked = world.fem.tetrahedra[object.elementOffset + local];
        const auto& tet = asset.tetrahedra[local];
        if (cooked.identity.y != nerve.objectIndex ||
            (cooked.identity.w & NM_OBJECT_ACTIVE) == 0u ||
            cooked.nodes.x != object.stateOffset + tet[0] ||
            cooked.nodes.y != object.stateOffset + tet[1] ||
            cooked.nodes.z != object.stateOffset + tet[2] ||
            cooked.nodes.w != object.stateOffset + tet[3])
            return fail("compiled optic tetrahedron order differs from pinned source connectivity");
    }
    error.clear();
    return true;
}

[[nodiscard]] inline numi::matter::CompiledWorldPartitionedReductionBasis
buildAndInstallRegisteredSkinFatOpticPreconditioner(
    numi::matter::CompiledWorld& world,
    const std::uint32_t skinObjectIndex,
    const RegisteredSkinFatReductionSourceLabels& skinLabels,
    const numi::matter::ObjectSource& skinSource,
    const std::span<const RegisteredOpticNerveReductionObject> opticObjects,
    const bool installPayload,
    const bool massWhiten,
    std::string& error,
    const bool keepSkinLayerZeroFine = false,
    const std::span<const std::uint32_t> residualFineGlobalFEMNodes = {},
    const std::uint32_t minimumCoordinateReductionFactor = 4u,
    const bool allowTraceRefinement = true,
    const bool embedContactBoundary = false,
    numi::matter::detail::HumanPartitionedPreconditionerSource* generatedPayload = nullptr,
    const std::uint32_t opticLongitudinalBinCount = 5u
) {
    using Candidate = numi::matter::CompiledWorldPartitionedReductionBasis;
    const std::uint32_t buildReductionFactor = minimumCoordinateReductionFactor;
    bool opticOnlyPreconditioner = false;
    bool sparseCoefficientEncoding = false;
    if (!parseHumanSparseCoefficientPreconditionerSetting(
            std::getenv("NM_FEM_HUMAN_PARTITIONED_SPARSE_BASIS"),
            sparseCoefficientEncoding, error)) {
        Candidate failed;
        failed.objectIndex = skinObjectIndex;
        failed.error = error;
        failed.basis.error = error;
        return failed;
    }
    if (!parseHumanOpticOnlyPartitionedPreconditionerSetting(
            std::getenv("NM_FEM_HUMAN_PARTITIONED_OPTIC_ONLY"),
            opticOnlyPreconditioner, error)) {
        Candidate failed;
        failed.objectIndex = skinObjectIndex;
        failed.error = error;
        failed.basis.error = error;
        return failed;
    }
    if (opticOnlyPreconditioner && opticObjects.size() != 2u) {
        Candidate failed;
        failed.objectIndex = skinObjectIndex;
        error = "optic-only preconditioner requires both registered optic source objects";
        failed.error = error;
        failed.basis.error = error;
        return failed;
    }
    if (opticLongitudinalBinCount != 5u && opticLongitudinalBinCount != 10u) {
        Candidate failed;
        failed.objectIndex = skinObjectIndex;
        error = "Human optic basis subdivision must be five bins by default or ten bins when opted in";
        failed.error = error;
        failed.basis.error = error;
        return failed;
    }
    // Validate a trace against the original source basis. A relaxed target
    // applies only when rebuilding the enriched basis, not to its provenance.
    std::uint32_t refinementReductionFactor = buildReductionFactor;
    const char* witnessTracePath = std::getenv(
        "NM_FEM_HYBRID_RESIDUAL_REFINEMENT_TRACE");
    if (buildReductionFactor < 2u || buildReductionFactor > 64u) {
        Candidate failed;
        failed.objectIndex = skinObjectIndex;
        error = "combined Human requested reduction target is outside the checked range";
        failed.error = error;
        failed.basis.error = error;
        return failed;
    }
    if (installPayload && allowTraceRefinement && witnessTracePath != nullptr &&
        !requestedHybridResidualReductionFactor(
            buildReductionFactor, refinementReductionFactor, error)) {
        Candidate failed;
        failed.objectIndex = skinObjectIndex;
        failed.error = error;
        failed.basis.error = error;
        return failed;
    }
    if (opticObjects.empty()) {
        if (opticOnlyPreconditioner) {
            Candidate failed;
            failed.objectIndex = skinObjectIndex;
            error = "optic-only preconditioner cannot fall back to a skin/fat-only source";
            failed.error = error;
            failed.basis.error = error;
            return failed;
        }
        if (installPayload && allowTraceRefinement && witnessTracePath != nullptr) {
            Candidate failed;
            failed.objectIndex = skinObjectIndex;
            error = "residual refinement trace requires the combined skin/fat and bilateral optic owner";
            failed.error = error;
            failed.basis.error = error;
            return failed;
        }
        return buildAndInstallRegisteredSkinFatPreconditioner(
            world, skinObjectIndex, skinLabels, skinSource,
            installPayload, massWhiten, error,
            keepSkinLayerZeroFine && !embedContactBoundary,
            residualFineGlobalFEMNodes, buildReductionFactor,
            embedContactBoundary);
    }
    std::map<std::uint32_t, std::vector<std::uint32_t>> residualRowsByObject;
    std::set<std::uint32_t> uniqueResidualRows;
    std::set<std::uint32_t> combinedObjectIndices{skinObjectIndex};
    for (const auto& nerve : opticObjects) {
        if (!combinedObjectIndices.insert(nerve.objectIndex).second) {
            Candidate failed;
            failed.objectIndex = skinObjectIndex;
            error = "combined Human source repeats an optic object owner";
            failed.error = error;
            failed.basis.error = error;
            return failed;
        }
    }
    for (const std::uint32_t global : residualFineGlobalFEMNodes) {
        if (!uniqueResidualRows.insert(global).second ||
            global >= world.fem.topologyNodes.size()) {
            Candidate failed;
            failed.objectIndex = skinObjectIndex;
            error = "residual refinement repeats or escapes the compiled FEM node map";
            failed.error = error;
            failed.basis.error = error;
            return failed;
        }
        const auto& topology = world.fem.topologyNodes[global];
        if ((topology.identity.w & NM_TOPOLOGY_ACTIVE) == 0u ||
            combinedObjectIndices.count(topology.identity.y) == 0u) {
            Candidate failed;
            failed.objectIndex = skinObjectIndex;
            error = "residual refinement node is not an active selected Human source row";
            failed.error = error;
            failed.basis.error = error;
            return failed;
        }
        residualRowsByObject[topology.identity.y].push_back(global);
    }
    const auto& skinResidualRows = residualRowsByObject[skinObjectIndex];
    Candidate skin = buildAndInstallRegisteredSkinFatPreconditioner(
        world, skinObjectIndex, skinLabels, skinSource, false,
        massWhiten, error, keepSkinLayerZeroFine && !embedContactBoundary,
        skinResidualRows, buildReductionFactor, embedContactBoundary);
    if (!skin.valid()) return skin;
    struct Piece {
        Candidate candidate;
        RegisteredOpticNerveReductionSourceLabels labels;
        std::uint32_t objectIndex = NM_INVALID_INDEX;
    };
    std::vector<Piece> pieces;
    pieces.push_back({std::move(skin), {}, skinObjectIndex});
    for (const auto& nerve : opticObjects) {
        if (nerve.source == nullptr || nerve.asset == nullptr) {
            error = "combined Human source has a null optic source owner";
            Candidate failed;
            failed.objectIndex = skinObjectIndex;
            failed.error = error;
            failed.basis.error = error;
            return failed;
        }
        Piece piece;
        piece.objectIndex = nerve.objectIndex;
        if (!buildRegisteredOpticNerveReductionSourceLabels(
                *nerve.asset, *nerve.source, piece.labels, error,
                opticLongitudinalBinCount) ||
            !verifyRegisteredOpticNerveCompiledSource(
                world, nerve, piece.labels, error)) {
            Candidate failed;
            failed.objectIndex = skinObjectIndex;
            failed.error = error;
            failed.basis.error = error;
            return failed;
        }
        numi::matter::CompiledWorldPartitionedReductionOptions options;
        options.minimumCoordinateReductionFactor = buildReductionFactor;
        options.maximumModeCount = numi::matter::kHumanPartitionedBasisMaximumModeCount;
        options.promoteRankDeficientRegionsToFine = embedContactBoundary ||
            !residualRowsByObject[nerve.objectIndex].empty();
        options.explicitAuthoredTetrahedronRegionLabels =
            std::span<const std::uint32_t>(piece.labels.tetrahedronRegionLabels);
        options.explicitRegionLabelObjectIndex = nerve.objectIndex;
        options.explicitRegionLabelWorldPhysicsFingerprint = world.physicsFingerprint;
        std::vector<std::uint32_t> fineRows;
        fineRows.reserve((embedContactBoundary ? 0u :
                         piece.labels.boundarySourceNodeIndices.size()) +
                         piece.labels.witnessSourceNodeIndices.size() +
                         residualRowsByObject[nerve.objectIndex].size());
        const std::uint32_t stateOffset = world.objects[nerve.objectIndex].stateOffset;
        if (!embedContactBoundary) {
            for (const std::uint32_t local : piece.labels.boundarySourceNodeIndices)
                fineRows.push_back(stateOffset + local);
        }
        for (const std::uint32_t local : piece.labels.witnessSourceNodeIndices) {
            if (local >= nerve.source->femNodes.size()) {
                error = "optic fine witness source node escapes its exact source object";
                piece.candidate.error = error;
                piece.candidate.basis.error = error;
                return piece.candidate;
            }
            fineRows.push_back(stateOffset + local);
        }
        std::sort(fineRows.begin(), fineRows.end());
        fineRows.erase(std::unique(fineRows.begin(), fineRows.end()), fineRows.end());
        const std::size_t protectedFineCount = fineRows.size();
        for (const std::uint32_t global : residualRowsByObject[nerve.objectIndex]) {
            if (std::binary_search(fineRows.begin(), fineRows.begin() +
                    static_cast<std::ptrdiff_t>(protectedFineCount), global)) {
                // The residual request is already represented exactly on the
                // fine complement. Keep the protected source row and avoid
                // turning a redundant promotion into a failed rebuild.
                continue;
            }
            fineRows.push_back(global);
        }
        std::sort(fineRows.begin(), fineRows.end());
        options.fineCoordinateOnlyGlobalFEMNodes =
            std::span<const std::uint32_t>(fineRows);
        piece.candidate =
            numi::matter::buildCompiledWorldPartitionedReductionBasis(
                world, nerve.objectIndex, options);
        if (!piece.candidate.valid()) {
            error = piece.candidate.error.empty()
                ? "pinned optic source basis failed completeness checks"
                : piece.candidate.error;
            piece.candidate.error = error;
            return piece.candidate;
        }
        if (piece.candidate.tetrahedronRegionLabels !=
                piece.labels.tetrahedronRegionLabels ||
            piece.candidate.sourceNodeIndices.size() != nerve.source->femNodes.size() ||
            piece.candidate.sourceTetrahedronIndices.size() !=
                nerve.source->tetrahedra.size())
        {
            error = "optic compiled basis lost source labels or exact object-local ancestry";
            piece.candidate.error = error;
            piece.candidate.basis.error = error;
            return piece.candidate;
        }
        for (std::size_t row = 0u; row < piece.candidate.sourceNodeIndices.size(); ++row) {
            if (piece.candidate.sourceNodeIndices[row] != row ||
                piece.candidate.globalFEMNodeIndices[row] != stateOffset + row) {
                error = "optic basis rows are not the exact source node order";
                piece.candidate.error = error;
                piece.candidate.basis.error = error;
                return piece.candidate;
            }
        }
        for (std::size_t tet = 0u; tet < piece.candidate.sourceTetrahedronIndices.size(); ++tet) {
            if (piece.candidate.sourceTetrahedronIndices[tet] != tet) {
                error = "optic basis tetrahedra are not the exact source order";
                piece.candidate.error = error;
                piece.candidate.basis.error = error;
                return piece.candidate;
            }
        }
        if (massWhiten) {
            std::vector<double> masses;
            masses.reserve(piece.candidate.globalFEMNodeIndices.size());
            for (const auto global : piece.candidate.globalFEMNodeIndices)
                masses.push_back(world.fem.nodes[global].positionAndMass.w);
            const auto whitening = numi::matter::massWhitenPartitionedReductionBasis(
                piece.candidate.basis, masses);
            if (!whitening.valid()) {
                error = "optic source mass whitening failed: " + whitening.error;
                piece.candidate.error = error;
                piece.candidate.basis.error = error;
                return piece.candidate;
            }
        }
        pieces.push_back(std::move(piece));
    }
    if (opticObjects.size() != 2u ||
        opticObjects[0].asset->member == opticObjects[1].asset->member ||
        opticObjects[0].objectIndex == opticObjects[1].objectIndex) {
        error = "combined Human optic preconditioner requires distinct left and right source objects";
        Candidate failed;
        failed.objectIndex = skinObjectIndex;
        failed.error = error;
        failed.basis.error = error;
        return failed;
    }

    std::vector<std::uint32_t> globalRows;
    std::map<std::uint32_t, std::pair<std::size_t, std::uint32_t>> rowOwner;
    std::map<std::uint32_t, std::uint32_t> anchorReasonByGlobal;
    std::vector<std::uint32_t> allGlobalTets, allSourceTets, allTetLabels;
    std::set<std::uint32_t> uniqueLabels;
    std::uint64_t freeCoordinates = 0u;
    for (std::size_t p = 0u; p < pieces.size(); ++p) {
        const auto& candidate = pieces[p].candidate;
        if (!candidate.valid()) {
            error = candidate.error.empty() ? "one Human source piece is invalid" : candidate.error;
            Candidate failed;
            failed.objectIndex = skinObjectIndex;
            failed.error = error;
            failed.basis.error = error;
            return failed;
        }
        if (freeCoordinates >
            std::numeric_limits<std::uint64_t>::max() -
                candidate.basis.freeCoordinateCount) {
            error = "combined Human free-coordinate count exceeds its checked range";
            Candidate failed;
            failed.objectIndex = skinObjectIndex;
            failed.error = error;
            failed.basis.error = error;
            return failed;
        }
        freeCoordinates += candidate.basis.freeCoordinateCount;
        for (std::size_t row = 0u; row < candidate.globalFEMNodeIndices.size(); ++row) {
            const std::uint32_t global = candidate.globalFEMNodeIndices[row];
            if (!rowOwner.emplace(global, std::pair{p, static_cast<std::uint32_t>(row)}).second) {
                error = "skin/fat and optic source objects overlap a compiled FEM node row";
                Candidate failed;
                failed.objectIndex = skinObjectIndex;
                failed.error = error;
                failed.basis.error = error;
                return failed;
            }
            globalRows.push_back(global);
        }
        for (std::size_t tet = 0u; tet < candidate.globalFEMTetrahedronIndices.size(); ++tet) {
            allGlobalTets.push_back(candidate.globalFEMTetrahedronIndices[tet]);
            allSourceTets.push_back(candidate.sourceTetrahedronIndices[tet]);
            allTetLabels.push_back(candidate.tetrahedronRegionLabels[tet]);
        }
        for (std::size_t a = 0u; a < candidate.basis.anchorNodes.size(); ++a) {
            const auto local = candidate.basis.anchorNodes[a];
            if (local >= candidate.globalFEMNodeIndices.size()) {
                error = "source-piece anchor escapes its own node map";
                Candidate failed;
                failed.objectIndex = skinObjectIndex;
                failed.error = error;
                failed.basis.error = error;
                return failed;
            }
            const std::uint32_t global = candidate.globalFEMNodeIndices[local];
            const std::uint32_t reason = a < candidate.anchorReasonMasks.size()
                ? candidate.anchorReasonMasks[a] : 0u;
            anchorReasonByGlobal[global] |= reason;
        }
    }
    std::sort(globalRows.begin(), globalRows.end());
    std::vector<std::uint32_t> sourceRows;
    sourceRows.reserve(globalRows.size());
    for (const std::uint32_t global : globalRows) {
        const auto it = rowOwner.find(global);
        if (it == rowOwner.end()) {
            error = "combined Human source row lost its ancestry map";
            Candidate failed;
            failed.objectIndex = skinObjectIndex;
            failed.error = error;
            failed.basis.error = error;
            return failed;
        }
        sourceRows.push_back(pieces[it->second.first].candidate.sourceNodeIndices[it->second.second]);
    }
    struct RegionRef { std::uint32_t label; std::size_t piece; std::uint32_t region; };
    std::vector<RegionRef> regions;
    for (std::size_t p = 0u; p < pieces.size(); ++p) {
        const auto& basis = pieces[p].candidate.basis;
        for (std::uint32_t r = 0u; r < basis.regions.size(); ++r) {
            const auto label = basis.regions[r].sourceLabel;
            if (!uniqueLabels.insert(label).second) {
                error = "skin/fat and optic source region labels collide";
                Candidate failed;
                failed.objectIndex = skinObjectIndex;
                failed.error = error;
                failed.basis.error = error;
                return failed;
            }
            regions.push_back({label, p, r});
        }
    }
    std::sort(regions.begin(), regions.end(), [](const RegionRef& a, const RegionRef& b) {
        return a.label < b.label;
    });
    if (opticOnlyPreconditioner) {
        std::vector<std::uint32_t> availableLabels;
        availableLabels.reserve(regions.size());
        for (const RegionRef& region : regions)
            availableLabels.push_back(region.label);
        std::vector<std::uint32_t> opticLabels;
        if (!selectHumanOpticOnlyPartitionedRegionLabels(
                std::span<const std::uint32_t>(availableLabels),
                opticLongitudinalBinCount, opticLabels, error)) {
            Candidate failed;
            failed.objectIndex = skinObjectIndex;
            failed.error = error;
            failed.basis.error = error;
            return failed;
        }
        std::vector<RegionRef> opticRegions;
        opticRegions.reserve(opticLabels.size());
        for (const RegionRef& region : regions) {
            if (std::binary_search(opticLabels.begin(), opticLabels.end(), region.label)) {
                if (region.piece == 0u) {
                    Candidate failed;
                    failed.objectIndex = skinObjectIndex;
                    error = "optic-only region label is owned by the skin/fat source";
                    failed.error = error;
                    failed.basis.error = error;
                    return failed;
                }
                opticRegions.push_back(region);
            }
        }
        if (opticRegions.size() != opticLabels.size()) {
            Candidate failed;
            failed.objectIndex = skinObjectIndex;
            error = "optic-only source region selection did not match both registered optic meshes";
            failed.error = error;
            failed.basis.error = error;
            return failed;
        }
        regions = std::move(opticRegions);
    }
    constexpr std::uint32_t blockModes = 12u;
    if (regions.empty() || regions.size() > numi::matter::kHumanPartitionedBasisMaximumRegionCount ||
        regions.size() * blockModes > numi::matter::kHumanPartitionedBasisMaximumModeCount ||
        freeCoordinates > std::numeric_limits<std::uint32_t>::max()) {
        error = "combined skin/fat and bilateral optic basis exceeds runtime regional or coordinate limits";
        Candidate failed;
        failed.objectIndex = skinObjectIndex;
        failed.error = error;
        failed.basis.error = error;
        return failed;
    }
    const std::uint32_t rowCount = static_cast<std::uint32_t>(globalRows.size());
    const std::uint32_t regionCount = static_cast<std::uint32_t>(regions.size());
    const std::uint32_t modeCount = regionCount * blockModes;
    const std::uint64_t maxBasisBytes =
        numi::matter::kHumanPartitionedBasisCoefficientByteBudget;
    if (!numi::matter::humanPartitionedBasisCoefficientExtentWithinBudget(
            modeCount, rowCount) ||
        static_cast<std::uint64_t>(modeCount) >
            maxBasisBytes / sizeof(nm_float4) / rowCount) {
        error = "combined Human basis exceeds its checked 384 MiB coefficient budget";
        Candidate failed;
        failed.objectIndex = skinObjectIndex;
        failed.error = error;
        failed.basis.error = error;
        return failed;
    }
    if (std::uint64_t(modeCount) * buildReductionFactor > freeCoordinates) {
        error = "combined source basis misses its requested free-coordinate reduction target";
        Candidate failed;
        failed.objectIndex = skinObjectIndex;
        failed.error = error;
        failed.basis.error = error;
        return failed;
    }

    Candidate combined;
    combined.objectIndex = skinObjectIndex;
    combined.regionLabelSource =
        numi::matter::CompiledReductionRegionLabelSource::callerBoundSourcePayload;
    combined.sourceWorldPhysicsFingerprint = world.physicsFingerprint;
    combined.globalFEMNodeIndices = globalRows;
    combined.sourceNodeIndices = sourceRows;
    combined.globalFEMTetrahedronIndices = std::move(allGlobalTets);
    combined.sourceTetrahedronIndices = std::move(allSourceTets);
    combined.tetrahedronRegionLabels = std::move(allTetLabels);
    combined.basis.nodeCount = rowCount;
    combined.basis.freeCoordinateCount = static_cast<std::uint32_t>(freeCoordinates);
    combined.basis.modeCount = modeCount;
    combined.basis.minimumCoordinateReductionFactor = buildReductionFactor;
    combined.basis.regions.reserve(regionCount);
    combined.basis.values.assign(std::size_t(modeCount) * rowCount, nm_float4{});
    combined.basis.projectedMass.assign(std::size_t(modeCount) * modeCount, 0.0);
    struct ModeRef { std::size_t piece; std::uint32_t mode; };
    std::vector<ModeRef> modes;
    modes.reserve(modeCount);
    for (const auto& region : regions) {
        const auto& basis = pieces[region.piece].candidate.basis;
        combined.basis.regions.push_back(basis.regions[region.region]);
        std::vector<std::uint32_t> localModes;
        for (std::uint32_t m = 0u; m < basis.modeCount; ++m) {
            const auto& mode = basis.modes[m];
            if (mode.regionLabel == region.label &&
                mode.kind != numi::matter::PartitionedReductionModeKind::anchorTranslation)
                localModes.push_back(m);
        }
        if (localModes.size() != blockModes) {
            error = "combined source region does not contain exactly twelve affine coordinates";
            combined.error = error;
            combined.basis.error = error;
            return combined;
        }
        for (const auto localMode : localModes) {
            modes.push_back({region.piece, localMode});
            combined.basis.modes.push_back(basis.modes[localMode]);
        }
    }
    if (modes.size() != modeCount) {
        error = "combined Human mode accounting is inconsistent";
        combined.error = error;
        combined.basis.error = error;
        return combined;
    }
    std::map<std::uint32_t, std::uint32_t> combinedRow;
    for (std::uint32_t row = 0u; row < rowCount; ++row)
        combinedRow.emplace(globalRows[row], row);
    for (std::uint32_t mode = 0u; mode < modeCount; ++mode) {
        const auto ref = modes[mode];
        const auto& piece = pieces[ref.piece].candidate;
        const std::size_t sourceOffset =
            std::size_t(ref.mode) * piece.basis.nodeCount;
        const std::size_t destOffset = std::size_t(mode) * rowCount;
        for (std::uint32_t local = 0u; local < piece.basis.nodeCount; ++local) {
            const auto global = piece.globalFEMNodeIndices[local];
            const auto dest = combinedRow.at(global);
            combined.basis.values[destOffset + dest] =
                piece.basis.values[sourceOffset + local];
        }
        // Different source objects have disjoint node rows, so exact
        // cross-source projected-mass blocks are zero. Copy the source-piece
        // Gram blocks in their selected affine-mode order without a large
        // dense B^T M B recomputation.
    }
    std::uint32_t modeBase = 0u;
    for (const auto& region : regions) {
        const auto& basis = pieces[region.piece].candidate.basis;
        std::vector<std::uint32_t> localModes;
        for (std::uint32_t m = 0u; m < basis.modeCount; ++m) {
            const auto& mode = basis.modes[m];
            if (mode.regionLabel == region.label &&
                mode.kind != numi::matter::PartitionedReductionModeKind::anchorTranslation)
                localModes.push_back(m);
        }
        for (const auto localMode : localModes) {
            std::uint32_t columnBase = 0u;
            for (const auto& columnRegion : regions) {
                if (columnRegion.piece == region.piece) {
                    const auto& columnBasis = pieces[columnRegion.piece].candidate.basis;
                    std::vector<std::uint32_t> columnModes;
                    for (std::uint32_t m = 0u; m < columnBasis.modeCount; ++m) {
                        const auto& mode = columnBasis.modes[m];
                        if (mode.regionLabel == columnRegion.label &&
                            mode.kind != numi::matter::PartitionedReductionModeKind::anchorTranslation)
                            columnModes.push_back(m);
                    }
                    for (const auto columnMode : columnModes) {
                        combined.basis.projectedMass[
                            std::size_t(modeBase) * modeCount + columnBase] =
                            basis.projectedMass[
                                std::size_t(localMode) * basis.modeCount + columnMode];
                        ++columnBase;
                    }
                } else {
                    columnBase += blockModes;
                }
            }
            ++modeBase;
        }
    }
    // Each piece candidate has already passed exact cooked-mass Cholesky.
    // Source-row ownership above proves cross-piece Gram blocks are zero, and
    // this matrix is copied piece-by-piece, so the combined matrix is their
    // direct sum; refactoring a 2k-mode dense matrix here would duplicate work.
    std::map<std::uint32_t, std::uint32_t> anchorRows;
    for (const auto& [global, reason] : anchorReasonByGlobal) {
        const auto row = combinedRow.find(global);
        if (row == combinedRow.end()) {
            error = "combined source anchor is absent from the exact row map";
            combined.error = error;
            combined.basis.error = error;
            return combined;
        }
        anchorRows.emplace(row->second, reason);
    }
    for (const auto& [row, reason] : anchorRows) {
        combined.basis.anchorNodes.push_back(row);
        combined.anchorReasonMasks.push_back(reason);
    }
    std::uint32_t piecePartitionTotal = 0u;
    std::vector<std::uint32_t> partitionBase(pieces.size(), 0u);
    for (std::size_t p = 0u; p < pieces.size(); ++p) {
        partitionBase[p] = piecePartitionTotal;
        piecePartitionTotal += pieces[p].candidate.basis.partitionCount;
    }
    combined.basis.partitionCount = piecePartitionTotal;
    combined.basis.partitionWeights.assign(
        std::size_t(rowCount) * piecePartitionTotal, 0.0);
    for (std::size_t p = 0u; p < pieces.size(); ++p) {
        const auto& basis = pieces[p].candidate.basis;
        for (std::uint32_t local = 0u; local < basis.nodeCount; ++local) {
            const auto row = combinedRow.at(
                pieces[p].candidate.globalFEMNodeIndices[local]);
            for (std::uint32_t part = 0u; part < basis.partitionCount; ++part)
                combined.basis.partitionWeights[
                    std::size_t(row) * piecePartitionTotal +
                    partitionBase[p] + part] =
                    basis.partitionWeights[
                        std::size_t(local) * basis.partitionCount + part];
        }
    }
    combined.basis.error.clear();
    combined.error.clear();
    if (!installPayload && generatedPayload == nullptr) {
        error.clear();
        return combined;
    }
    numi::matter::detail::HumanPartitionedPreconditionerSource payload;
    payload.objectIndex = skinObjectIndex; // provenance primary; row owners are validated from topology.
    payload.opticOnly = opticOnlyPreconditioner;
    payload.basisNodeCount = rowCount;
    payload.regionCount = regionCount;
    payload.blockModeCount = blockModes;
    payload.modeCount = modeCount;
    payload.freeCoordinateCount = static_cast<std::uint32_t>(freeCoordinates);
    payload.sensitiveAnchorCount = static_cast<std::uint32_t>(combined.basis.anchorNodes.size());
    payload.minimumCoordinateReductionFactor = buildReductionFactor;
    payload.sourceWorldPhysicsFingerprint = world.physicsFingerprint;
    payload.globalFEMNodeIndices = combined.globalFEMNodeIndices;
    payload.regionLabels.reserve(regionCount);
    for (const auto& region : combined.basis.regions)
        payload.regionLabels.push_back(region.sourceLabel);
    payload.blockModeIndices.resize(modeCount);
    for (std::uint32_t mode = 0u; mode < modeCount; ++mode)
        payload.blockModeIndices[mode] = mode;
    const std::uint64_t payloadBytes =
        static_cast<std::uint64_t>(modeCount) * rowCount * sizeof(nm_float4);
    if (payloadBytes > numi::matter::kHumanPartitionedBasisCoefficientByteBudget) {
        error = "combined Human basis exceeds its checked 384 MiB coefficient budget";
        combined.error = error;
        combined.basis.error = error;
        return combined;
    }
    payload.basisValues = combined.basis.values;
    const auto support = numi::matter::buildPartitionedReductionSparseSupport(
        std::span<const nm_float4>(payload.basisValues), rowCount, modeCount,
        std::span<const std::uint32_t>(payload.blockModeIndices),
        regionCount, blockModes);
    if (!support.valid()) {
        error = "combined bilateral optic sparse support is invalid: " + support.error;
        combined.error = error;
        combined.basis.error = error;
        return combined;
    }
    if (sparseCoefficientEncoding) {
        const auto sparse =
            numi::matter::buildPartitionedReductionSparseCoefficients(
                std::span<const nm_float4>(payload.basisValues), rowCount,
                modeCount,
                std::span<const std::uint32_t>(payload.blockModeIndices),
                regionCount, blockModes, support);
        if (!sparse.valid()) {
            error = "Human sparse basis encoding failed: " + sparse.error;
            combined.error = error;
            combined.basis.error = error;
            return combined;
        }
        payload.sparseCoefficientEncoding = true;
        payload.sparseRegionSupportOffsets.assign(
            static_cast<std::size_t>(regionCount) + 1u, 0u);
        for (std::uint32_t region = 0u; region < regionCount; ++region) {
            const auto range = support.regionSupportRanges[region];
            if (range.first != payload.sparseRegionSupportOffsets[region] ||
                static_cast<std::uint64_t>(range.first) + range.count >
                    std::numeric_limits<std::uint32_t>::max()) {
                error = "Human sparse region support ranges are not contiguous and checked";
                combined.error = error;
                combined.basis.error = error;
                return combined;
            }
            payload.sparseRegionSupportOffsets[region + 1u] =
                range.first + range.count;
        }
        payload.sparseRegionSupportRows = support.regionSupportRows;
        payload.sparseModeValueOffsets = sparse.modeValueOffsets;
        payload.sparseBasisValues = sparse.values;
        payload.sparseNodeModeValueIndices = sparse.nodeModeValueIndices;
    }
    payload.enabled = true;
    if (installPayload && allowTraceRefinement && witnessTracePath != nullptr) {
        const std::uint32_t requestedFactor = refinementReductionFactor;
        std::vector<numi::matter::PartitionedReductionResidualWitness> witnesses;
        if (!loadHybridResidualWitnessTrace(witnessTracePath, witnesses, error)) {
            combined.error = error;
            combined.basis.error = error;
            return combined;
        }
        Candidate activeCandidate = std::move(combined);
        auto activePayload = std::move(payload);
        std::vector<std::uint32_t> existingFineRows;
        std::size_t cumulativeAddedRows = 0u;
        for (std::size_t stage = 0u; stage < witnesses.size(); ++stage) {
            existingFineRows.clear();
            existingFineRows.reserve(activePayload.basisNodeCount);
            for (std::uint32_t row = 0u; row < activePayload.basisNodeCount; ++row) {
                bool hasBasisSupport = false;
                for (std::uint32_t mode = 0u; mode < activePayload.modeCount; ++mode) {
                    const auto value = activePayload.basisValues[
                        std::size_t(mode) * activePayload.basisNodeCount + row];
                    if (!std::isfinite(value.x) || !std::isfinite(value.y) ||
                        !std::isfinite(value.z) || !std::isfinite(value.w)) {
                        error = "current Human basis contains a non-finite coefficient during residual refinement";
                        activeCandidate.error = error;
                        activeCandidate.basis.error = error;
                        return activeCandidate;
                    }
                    hasBasisSupport = hasBasisSupport || value.x != 0.0f ||
                        value.y != 0.0f || value.z != 0.0f;
                }
                if (!hasBasisSupport)
                    existingFineRows.push_back(activePayload.globalFEMNodeIndices[row]);
            }
            std::vector<std::uint32_t> selectedFineRows;
            numi::matter::PartitionedReductionResidualEnrichmentSelection selection;
            const std::uint32_t maximumFineRows = activePayload.regionCount * 4u;
            const bool selected =
                numi::matter::selectPartitionedResidualEnrichmentFineNodes(
                    activePayload, witnesses[stage], existingFineRows,
                    requestedFactor, 4u, maximumFineRows,
                    selectedFineRows, error, &selection);
            std::fprintf(stderr,
                "human_hybrid_residual_selection={\"schema\":1,\"stage\":%zu,\"stage_limit\":2,\"witness_candidates\":%zu,\"eligible_candidates\":%zu,\"after_region_cap\":%zu,\"selected_candidates\":%zu,\"dropped_by_global_cap\":%zu,\"global_cap\":%u,\"available_fine_row_budget\":%llu,\"global_cap_applied\":%s,\"success\":%s}\n",
                stage + 1u, selection.candidateCount,
                selection.eligibleCandidateCount, selection.perRegionCandidateCount,
                selection.selectedCandidateCount, selection.droppedByGlobalCap,
                selection.globalMaximum,
                static_cast<unsigned long long>(selection.availableFineRowBudget),
                selection.globalCapApplied ? "true" : "false",
                selected ? "true" : "false");
            if (!selected) {
                activeCandidate.error = error;
                activeCandidate.basis.error = error;
                return activeCandidate;
            }
            if (selectedFineRows == existingFineRows) {
                error = "the captured residual witness selected no new supported Human rows";
                activeCandidate.error = error;
                activeCandidate.basis.error = error;
                return activeCandidate;
            }
            std::vector<std::uint32_t> addedFineRows;
            std::set_difference(selectedFineRows.begin(), selectedFineRows.end(),
                                existingFineRows.begin(), existingFineRows.end(),
                                std::back_inserter(addedFineRows));
            if (addedFineRows.empty()) {
                error = "sequential residual refinement stage added no new fine rows";
                activeCandidate.error = error;
                activeCandidate.basis.error = error;
                return activeCandidate;
            }

            const std::uint64_t sourceFingerprint =
                activePayload.sourceWorldPhysicsFingerprint;
            // Release the previous dense basis copies before rebuilding the
            // next stage; the coefficient arena is bounded but large.
            activeCandidate = Candidate{};
            activePayload = numi::matter::detail::HumanPartitionedPreconditionerSource{};

            numi::matter::detail::HumanPartitionedPreconditionerSource rebuiltPayload;
            Candidate refined = buildAndInstallRegisteredSkinFatOpticPreconditioner(
                world, skinObjectIndex, skinLabels, skinSource, opticObjects,
                false, massWhiten, error, keepSkinLayerZeroFine, selectedFineRows,
                requestedFactor, false, embedContactBoundary, &rebuiltPayload);
            if (!refined.valid()) return refined;
            if (!rebuiltPayload.enabled ||
                rebuiltPayload.sourceWorldPhysicsFingerprint != sourceFingerprint) {
                error = "rebuilt residual stage lost its source-bound payload identity";
                refined.error = error;
                refined.basis.error = error;
                return refined;
            }
            std::vector<std::uint32_t> rebuiltFineRows;
            rebuiltFineRows.reserve(rebuiltPayload.basisNodeCount);
            for (std::uint32_t row = 0u; row < rebuiltPayload.basisNodeCount; ++row) {
                bool hasBasisSupport = false;
                for (std::uint32_t mode = 0u; mode < rebuiltPayload.modeCount; ++mode) {
                    const auto value = rebuiltPayload.basisValues[
                        std::size_t(mode) * rebuiltPayload.basisNodeCount + row];
                    if (!std::isfinite(value.x) || !std::isfinite(value.y) ||
                        !std::isfinite(value.z) || !std::isfinite(value.w)) {
                        error = "rebuilt residual stage contains a non-finite basis coefficient";
                        refined.error = error;
                        refined.basis.error = error;
                        return refined;
                    }
                    hasBasisSupport = hasBasisSupport || value.x != 0.0f ||
                        value.y != 0.0f || value.z != 0.0f;
                }
                if (!hasBasisSupport)
                    rebuiltFineRows.push_back(rebuiltPayload.globalFEMNodeIndices[row]);
            }
            if (!std::includes(rebuiltFineRows.begin(), rebuiltFineRows.end(),
                               selectedFineRows.begin(), selectedFineRows.end())) {
                error = "rebuilt residual stage did not preserve every selected fine row";
                refined.error = error;
                refined.basis.error = error;
                return refined;
            }
            const std::uint64_t requiredCoordinates =
                std::uint64_t(rebuiltPayload.modeCount) * requestedFactor;
            if (rebuiltPayload.freeCoordinateCount < requiredCoordinates ||
                rebuiltFineRows.size() >
                    (rebuiltPayload.freeCoordinateCount - requiredCoordinates) / 3u) {
                error = "rank-safe residual promotion exceeds the requested coordinate-reduction target";
                refined.error = error;
                refined.basis.error = error;
                return refined;
            }
            std::vector<std::uint32_t> actualAddedFineRows;
            std::set_difference(rebuiltFineRows.begin(), rebuiltFineRows.end(),
                                existingFineRows.begin(), existingFineRows.end(),
                                std::back_inserter(actualAddedFineRows));
            cumulativeAddedRows += actualAddedFineRows.size();
            std::fprintf(stderr,
                "human_hybrid_residual_refinement={\"schema\":1,\"stage\":%zu,\"stage_limit\":2,\"selected_new_rows\":%zu,\"added_fine_rows\":%zu,\"cumulative_added_fine_rows\":%zu,\"source_physics_fingerprint\":\"%016llx\",\"witness_layout_fingerprint\":\"%016llx\",\"rebuilt_minimum_reduction_factor\":%u}\n",
                stage + 1u, addedFineRows.size(), actualAddedFineRows.size(),
                cumulativeAddedRows,
                static_cast<unsigned long long>(
                    witnesses[stage].sourceWorldPhysicsFingerprint),
                static_cast<unsigned long long>(
                    witnesses[stage].sourceLayoutFingerprint),
                requestedFactor);
            activeCandidate = std::move(refined);
            activePayload = std::move(rebuiltPayload);
        }
        world.humanPartitionedPreconditioner = std::move(activePayload);
        error.clear();
        return activeCandidate;
    }
    if (installPayload)
        world.humanPartitionedPreconditioner = std::move(payload);
    else if (generatedPayload != nullptr)
        *generatedPayload = std::move(payload);
    error.clear();
    return combined;
}

template<class VerifyBoundBody, class ParentBodyAt>
[[nodiscard]] inline bool buildRegisteredSkinFatReductionSourceLabels(
    const RegisteredSurface& surface,
    const numi::matter::ObjectSource& shell,
    const std::uint32_t layerWidth,
    const std::size_t sourceBodyCount,
    VerifyBoundBody&& verifyBoundBody,
    ParentBodyAt&& parentBodyAt,
    RegisteredSkinFatReductionSourceLabels& output,
    std::string& error,
    const std::uint32_t maximumFacesPerBasisRegion,
    const std::span<const std::uint32_t> splitSourceTetrahedronIndices,
    const std::uint32_t maximumRefinementPatchCount = 0u,
    const std::function<bool(std::uint32_t, std::uint32_t)>& verifyEmbeddedBody = {},
    const std::span<const std::uint32_t> mechanicalFaceIdentities = {},
    const ExplicitPairedLayerTopology* explicitTopology = nullptr,
    const std::uint32_t skinMaterialIndex = NM_INVALID_INDEX,
    const std::uint32_t fatMaterialIndex = NM_INVALID_INDEX,
    const bool buildSpatialBasisRegions = true
) {
    const auto fail = [&](const char* message) {
        error = message;
        return false;
    };
    if (layerWidth == 0u || sourceBodyCount == 0u ||
        (!explicitTopology &&
         surface.faces.size() > std::numeric_limits<std::size_t>::max() / 6u) ||
        surface.worldPoints.size() != layerWidth ||
        surface.faces.empty() || surface.donorBodyIndices.size() != layerWidth ||
        surface.sourceVertexIndices.size() != layerWidth ||
        (!surface.sourceFaceIndices.empty() &&
         surface.sourceFaceIndices.size() != surface.faces.size()) ||
        shell.representation != numi::matter::Representation::fem ||
        shell.femNodes.size() != 3u * static_cast<std::size_t>(layerWidth) ||
        (!explicitTopology && shell.tetrahedra.size() != 6u * surface.faces.size())) {
        return fail("registered Human skin/fat source arrays do not match the three-layer source mesh");
    }
    if (!mechanicalFaceIdentities.empty() && mechanicalFaceIdentities.size() != surface.faces.size())
        return fail("registered mechanical face identity count differs from the source topology");
    if (!validateRegisteredSourceEmbedding(surface, 0u, error)) return false;
    const bool embedded = !surface.sourceEmbeddingFaceIndices.empty();
    std::set<std::uint32_t> uniqueSplitTetrahedra;
    for (const std::uint32_t tet : splitSourceTetrahedronIndices) {
        if (tet >= shell.tetrahedra.size())
            return fail("requested skin/fat subdivision tetrahedron is outside the exact source mesh");
        if (!uniqueSplitTetrahedra.insert(tet).second)
            return fail("requested skin/fat subdivision repeats a source tetrahedron");
    }

    using FaceKey = std::array<std::uint32_t, 3u>;
    std::map<FaceKey, std::size_t> faceByVertices;
    std::set<std::uint32_t> sourceVertices;
    std::set<std::uint32_t> sourceFaceIdentities;
    for (std::size_t vertex = 0u; vertex < layerWidth; ++vertex) {
        const std::uint32_t sourceVertex = surface.sourceVertexIndices[vertex];
        const std::uint32_t donorBody = surface.donorBodyIndices[vertex];
        if (donorBody == NM_INVALID_INDEX || donorBody >= sourceBodyCount)
            return fail("registered skin vertex has invalid donor identity");
        if (sourceVertex == NM_INVALID_INDEX) {
            // A real face interpolation is not an invented source vertex. The caller
            // certifies the full source-weight interpolation against this exact row.
            if (!embedded || !verifyEmbeddedBody ||
                !verifyEmbeddedBody(static_cast<std::uint32_t>(vertex), donorBody))
                return fail("registered skin projected vertex has unverified source embedding/donor identity");
        } else if (!sourceVertices.insert(sourceVertex).second ||
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
        const std::uint32_t originalParent = surface.sourceFaceIndices.empty()
            ? static_cast<std::uint32_t>(face) : surface.sourceFaceIndices[face];
        const std::uint32_t mechanicalIdentity = mechanicalFaceIdentities.empty()
            ? originalParent : mechanicalFaceIdentities[face];
        if (face > std::numeric_limits<std::uint32_t>::max() ||
            originalParent == NM_INVALID_INDEX || mechanicalIdentity == NM_INVALID_INDEX ||
            !sourceFaceIdentities.insert(mechanicalIdentity).second)
            return fail("registered skin mechanical face identity is invalid/duplicated or source parent is missing");
    }

    RegisteredSkinFatReductionSourceLabels next;
    next.registeredMechanicalFaceIndices.assign(mechanicalFaceIdentities.begin(), mechanicalFaceIdentities.end());
    next.registeredVertexCount = layerWidth;
    next.registeredFaceCount = surface.faces.size();
    next.sourceTetrahedronCount = shell.tetrahedra.size();
    next.sourceNodeVertexIndices.reserve(shell.femNodes.size());
    next.sourceNodeLayerIndices.reserve(shell.femNodes.size());
    next.sourceNodeDonorBodyIndices.reserve(shell.femNodes.size());
    if (embedded) {
        next.sourceNodeEmbeddingFaceIndices.reserve(shell.femNodes.size());
        next.sourceNodeEmbeddingBarycentric.reserve(shell.femNodes.size());
    }
    for (std::uint32_t layer = 0u; layer < 3u; ++layer) {
        for (std::uint32_t vertex = 0u; vertex < layerWidth; ++vertex) {
            next.sourceNodeVertexIndices.push_back(surface.sourceVertexIndices[vertex]);
            next.sourceNodeDonorBodyIndices.push_back(surface.donorBodyIndices[vertex]);
            if (embedded) {
                next.sourceNodeEmbeddingFaceIndices.push_back(surface.sourceEmbeddingFaceIndices[vertex]);
                next.sourceNodeEmbeddingBarycentric.push_back(surface.sourceEmbeddingBarycentric[vertex]);
            }
            next.sourceNodeLayerIndices.push_back(layer);
        }
    }
    next.tetrahedronRegionLabels.reserve(shell.tetrahedra.size());
    next.tetrahedronBasisRegionLabels.reserve(shell.tetrahedra.size());
    next.tetrahedronSourceFaceIndices.reserve(shell.tetrahedra.size());
    next.tetrahedronOriginalSourceFaceIndices.reserve(shell.tetrahedra.size());
    next.tetrahedronSourceDonorBodyIndices.reserve(shell.tetrahedra.size());
    next.tetrahedronSourceParentBodyIndices.reserve(shell.tetrahedra.size());
    std::size_t explicitParentWidth = 0u;
    std::vector<std::uint32_t> explicitParentDonor;
    std::vector<std::uint32_t> explicitParentStableFace;
    if (explicitTopology != nullptr) {
        const auto& topology = *explicitTopology;
        const std::size_t childCount = shell.tetrahedra.size();
        const std::size_t parentCount = topology.parentTetrahedra.size();
        if (skinMaterialIndex == NM_INVALID_INDEX || fatMaterialIndex == NM_INVALID_INDEX ||
            skinMaterialIndex == fatMaterialIndex ||
            topology.childTetrahedra.size() != childCount ||
            topology.childMaterialIndices.size() != childCount ||
            topology.childParentTetrahedra.size() != childCount ||
            topology.childParentSurfaceFaces.size() != childCount ||
            topology.childSourceFaceIdentities.size() != childCount ||
            shell.femMaterialIndices.size() != childCount ||
            parentCount == 0u || topology.parentMaterialIndices.size() != parentCount ||
            topology.parentWorldPoints.empty() || topology.parentWorldPoints.size() % 3u != 0u ||
            mechanicalFaceIdentities.size() != surface.faces.size())
            return fail("explicit skin/fat source-label topology metadata is incomplete");

        const std::size_t parentWidth = topology.parentWorldPoints.size() / 3u;
        explicitParentWidth = parentWidth;
        if (parentWidth == 0u || parentWidth > layerWidth ||
            parentCount > std::numeric_limits<std::uint32_t>::max() ||
            childCount > std::numeric_limits<std::uint32_t>::max())
            return fail("explicit skin/fat source-label parent dimensions are invalid");
        for (const auto& point : topology.parentWorldPoints)
            if (!std::all_of(point.begin(), point.end(),
                             [](double value) { return std::isfinite(value); }))
                return fail("explicit skin/fat source-label parent geometry is non-finite");

        explicitParentDonor.assign(parentCount, NM_INVALID_INDEX);
        explicitParentStableFace.assign(parentCount, NM_INVALID_INDEX);
        auto& parentDonor = explicitParentDonor;
        auto& parentStableFace = explicitParentStableFace;
        std::vector<std::uint32_t> parentChildCount(parentCount, 0u);
        std::vector<std::uint32_t> parentSkinChildCount(parentCount, 0u);
        std::vector<std::uint32_t> parentFatChildCount(parentCount, 0u);
        std::vector<std::uint32_t> faceChildCount(surface.faces.size(), 0u);
        std::vector<std::uint32_t> faceDonor(surface.faces.size(), NM_INVALID_INDEX);
        std::size_t totalSkinChildren = 0u;
        std::size_t totalFatChildren = 0u;

        // Parent tetrahedra retain the original 3-layer coarse source topology.
        // Recover each parent's original face donor once from its three normalized
        // old source vertices; children inherit that identity strictly by parent ID.
        for (std::size_t parent = 0u; parent < parentCount; ++parent) {
            const std::uint32_t parentMaterial = topology.parentMaterialIndices[parent];
            if (parentMaterial != skinMaterialIndex && parentMaterial != fatMaterialIndex)
                return fail("explicit skin/fat parent tetrahedron has an unknown material identity");
            std::array<std::uint32_t, 3u> sourceVertices{};
            std::size_t sourceVertexCount = 0u;
            for (const std::uint32_t node : topology.parentTetrahedra[parent]) {
                if (node >= topology.parentWorldPoints.size())
                    return fail("explicit skin/fat parent tetrahedron references a missing parent node");
                const std::uint32_t sourceVertex =
                    static_cast<std::uint32_t>(node % parentWidth);
                if (sourceVertex >= parentWidth || sourceVertex >= layerWidth)
                    return fail("explicit skin/fat parent tetrahedron escapes the original vertex prefix");
                if (std::find(sourceVertices.begin(), sourceVertices.begin() +
                              static_cast<std::ptrdiff_t>(sourceVertexCount), sourceVertex) ==
                    sourceVertices.begin() + static_cast<std::ptrdiff_t>(sourceVertexCount)) {
                    if (sourceVertexCount == sourceVertices.size())
                        return fail("explicit parent tetrahedron does not retain a triangular source-face ancestry");
                    sourceVertices[sourceVertexCount++] = sourceVertex;
                }
            }
            if (sourceVertexCount != 3u)
                return fail("explicit parent tetrahedron does not normalize to exactly three original skin vertices");
            std::array<std::uint32_t, 3u> donors{};
            for (std::size_t corner = 0u; corner < sourceVertices.size(); ++corner)
                donors[corner] = surface.donorBodyIndices[sourceVertices[corner]];
            std::sort(donors.begin(), donors.end());
            std::uint32_t dominant = donors[0];
            std::size_t bestCount = 0u;
            for (std::size_t begin = 0u; begin < donors.size();) {
                std::size_t finish = begin + 1u;
                while (finish < donors.size() && donors[finish] == donors[begin]) ++finish;
                if (finish - begin > bestCount) {
                    dominant = donors[begin];
                    bestCount = finish - begin;
                }
                begin = finish;
            }
            if (dominant == NM_INVALID_INDEX || dominant >= sourceBodyCount)
                return fail("explicit parent tetrahedron has no valid original donor-body identity");
            parentDonor[parent] = dominant;
        }

        std::set<std::uint32_t> usedMechanicalFaces;
        for (std::size_t face = 0u; face < surface.faces.size(); ++face) {
            const std::uint32_t identity = mechanicalFaceIdentities[face];
            if (identity == NM_INVALID_INDEX || !usedMechanicalFaces.insert(identity).second)
                return fail("explicit refined surface face identities are invalid or duplicated");
        }
        for (std::size_t tet = 0u; tet < childCount; ++tet) {
            const auto parent = topology.childParentTetrahedra[tet];
            const auto face = topology.childParentSurfaceFaces[tet];
            const auto material = topology.childMaterialIndices[tet];
            const auto stableFace = topology.childSourceFaceIdentities[tet];
            if (parent >= parentCount || face >= surface.faces.size() ||
                stableFace == NM_INVALID_INDEX ||
                (material != skinMaterialIndex && material != fatMaterialIndex) ||
                material != topology.parentMaterialIndices[parent] ||
                shell.femMaterialIndices[tet] != material)
                return fail("explicit child skin/fat material, parent, or registered-face identity is inconsistent");
            const std::uint32_t registeredStableFace = surface.sourceFaceIndices.empty()
                ? face : surface.sourceFaceIndices[face];
            if (stableFace != registeredStableFace)
                return fail("explicit child source-face identity does not match the exact registered surface ancestry");

            auto expectedNodes = topology.childTetrahedra[tet];
            auto actualNodes = shell.tetrahedra[tet].nodes;
            std::sort(expectedNodes.begin(), expectedNodes.end());
            std::sort(actualNodes.begin(), actualNodes.end());
            if (expectedNodes != actualNodes)
                return fail("explicit label topology child order differs from the appended FEM shell topology");
            for (const std::uint32_t node : topology.childTetrahedra[tet])
                if (node >= shell.femNodes.size())
                    return fail("explicit child skin/fat tetrahedron references an absent shell node");

            if (parentStableFace[parent] == NM_INVALID_INDEX)
                parentStableFace[parent] = stableFace;
            else if (parentStableFace[parent] != stableFace)
                return fail("explicit children of one parent tetrahedron disagree on original source-face ancestry");
            if (faceDonor[face] == NM_INVALID_INDEX)
                faceDonor[face] = parentDonor[parent];
            else if (faceDonor[face] != parentDonor[parent])
                return fail("one refined registered face maps to multiple original parent donor bodies");
            ++parentChildCount[parent];
            if (material == skinMaterialIndex) {
                ++parentSkinChildCount[parent];
                ++totalSkinChildren;
            } else {
                ++parentFatChildCount[parent];
                ++totalFatChildren;
            }
            ++faceChildCount[face];
        }
        if (totalSkinChildren == 0u || totalFatChildren == 0u)
            return fail("explicit skin/fat label topology omits one tissue material");
        for (std::size_t parent = 0u; parent < parentCount; ++parent) {
            if (parentChildCount[parent] == 0u || parentStableFace[parent] == NM_INVALID_INDEX ||
                (topology.parentMaterialIndices[parent] == skinMaterialIndex &&
                 parentSkinChildCount[parent] == 0u) ||
                (topology.parentMaterialIndices[parent] == fatMaterialIndex &&
                 parentFatChildCount[parent] == 0u))
                return fail("explicit skin/fat child topology does not completely cover every material parent");
        }
        for (std::size_t face = 0u; face < surface.faces.size(); ++face)
            if (faceChildCount[face] == 0u || faceDonor[face] == NM_INVALID_INDEX)
                return fail("explicit skin/fat child topology leaves a registered boundary face unlabeled");

        const auto ensureDonorAncestry = [&](const std::uint32_t donor) -> bool {
            const std::uint32_t parent = parentBodyAt(donor);
            if ((parent != NM_INVALID_INDEX && parent >= sourceBodyCount) || parent == donor)
                return false;
            std::vector<std::uint32_t> ancestry{donor};
            std::set<std::uint32_t> seen{donor};
            std::uint32_t cursor = parent;
            while (cursor != NM_INVALID_INDEX) {
                if (cursor >= sourceBodyCount || !seen.insert(cursor).second) return false;
                ancestry.push_back(cursor);
                cursor = parentBodyAt(cursor);
            }
            const auto [entry, inserted] = next.donorBodyAncestry.emplace(donor, ancestry);
            return inserted || entry->second == ancestry;
        };
        for (std::size_t tet = 0u; tet < childCount; ++tet) {
            const std::uint32_t parent = topology.childParentTetrahedra[tet];
            const std::uint32_t face = topology.childParentSurfaceFaces[tet];
            const std::uint32_t donor = parentDonor[parent];
            if (!ensureDonorAncestry(donor))
                return fail("explicit parent donor-body ancestry contains an invalid index or cycle");
            const std::uint32_t parentBody = parentBodyAt(donor);
            const std::uint32_t stableFace = topology.childSourceFaceIdentities[tet];
            next.tetrahedronRegionLabels.push_back(donor);
            next.tetrahedronBasisRegionLabels.push_back(NM_INVALID_INDEX);
            next.tetrahedronSourceFaceIndices.push_back(mechanicalFaceIdentities[face]);
            next.tetrahedronOriginalSourceFaceIndices.push_back(stableFace);
            next.tetrahedronSourceDonorBodyIndices.push_back(donor);
            next.tetrahedronSourceParentBodyIndices.push_back(parentBody);
            next.tetrahedronParentTetrahedronIndices.push_back(parent);
        }
    } else {
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
            next.tetrahedronSourceFaceIndices.push_back(mechanicalFaceIdentities.empty()
                ? sourceFaceIndex : mechanicalFaceIdentities[faceIt->second]);
            next.tetrahedronOriginalSourceFaceIndices.push_back(sourceFaceIndex);
            next.tetrahedronSourceDonorBodyIndices.push_back(dominantBody);
            next.tetrahedronSourceParentBodyIndices.push_back(parent);
        }
        for (const auto& counts : tetsPerFaceLayer) {
            if (counts[0] != 3u || counts[1] != 3u)
                return fail("registered skin source face does not own exactly three skin and three fat prism tetrahedra");
        }
    }
    // Endpoint capture needs exact anatomical source ancestry, not an unused
    // preconditioner basis. Keep every source/tet/donor validation above.
    if (!buildSpatialBasisRegions && explicitTopology == nullptr) {
        if (!uniqueSplitTetrahedra.empty() || maximumRefinementPatchCount != 0u)
            return fail("ancestry-only capture cannot request basis refinement");
        next.tetrahedronBasisRegionLabels = next.tetrahedronRegionLabels;
        output = std::move(next);
        error.clear();
        return true;
    }
    if (explicitTopology != nullptr) {
        const auto& topology = *explicitTopology;
        const std::size_t parentWidth = explicitParentWidth;
        const auto& parentDonor = explicitParentDonor;
        const auto& parentStableFace = explicitParentStableFace;
        using FaceKey = std::array<std::uint32_t, 3u>;
        std::map<FaceKey, std::uint32_t> parentFaceByVertices;
        std::vector<FaceKey> parentFaceVertices;
        std::vector<std::uint32_t> parentStableIdentity;
        std::vector<std::uint32_t> parentFaceDonor;
        std::vector<std::uint32_t> parentFaceTetCount;
        std::vector<std::uint32_t> parentTetToFace(topology.parentTetrahedra.size(), NM_INVALID_INDEX);
        for (std::size_t parent = 0u; parent < topology.parentTetrahedra.size(); ++parent) {
            FaceKey key{};
            std::size_t count = 0u;
            for (const std::uint32_t node : topology.parentTetrahedra[parent]) {
                const std::uint32_t vertex = static_cast<std::uint32_t>(node % parentWidth);
                if (std::find(key.begin(), key.begin() + static_cast<std::ptrdiff_t>(count), vertex) ==
                    key.begin() + static_cast<std::ptrdiff_t>(count)) {
                    if (count == key.size())
                        return fail("explicit parent tetrahedron no longer has one triangular source-face key");
                    key[count++] = vertex;
                }
            }
            if (count != 3u)
                return fail("explicit parent tetrahedron lost its three-vertex source-face key");
            std::sort(key.begin(), key.end());
            auto [entry, inserted] = parentFaceByVertices.emplace(
                key, static_cast<std::uint32_t>(parentFaceVertices.size()));
            if (inserted) {
                parentFaceVertices.push_back(key);
                parentStableIdentity.push_back(parentStableFace[parent]);
                parentFaceDonor.push_back(parentDonor[parent]);
                parentFaceTetCount.push_back(0u);
            } else {
                const std::size_t face = entry->second;
                if (parentStableIdentity[face] != parentStableFace[parent] ||
                    parentFaceDonor[face] != parentDonor[parent])
                    return fail("original parent face changed stable source or donor ancestry across its tetrahedra");
            }
            const std::size_t face = entry->second;
            parentTetToFace[parent] = static_cast<std::uint32_t>(face);
            ++parentFaceTetCount[face];
        }
        for (const std::uint32_t count : parentFaceTetCount)
            if (count != 6u)
                return fail("explicit parent source face does not preserve its original six-tet prism basis ancestry");
        if (parentFaceVertices.empty() || parentFaceVertices.size() >
            std::numeric_limits<std::uint32_t>::max())
            return fail("explicit parent source-face topology is empty or exceeds supported identity range");

        RegisteredSurface parentSurface;
        parentSurface.worldPoints.assign(surface.worldPoints.begin(),
            surface.worldPoints.begin() + static_cast<std::ptrdiff_t>(parentWidth));
        parentSurface.sourceVertexIndices.assign(surface.sourceVertexIndices.begin(),
            surface.sourceVertexIndices.begin() + static_cast<std::ptrdiff_t>(parentWidth));
        parentSurface.donorBodyIndices.assign(surface.donorBodyIndices.begin(),
            surface.donorBodyIndices.begin() + static_cast<std::ptrdiff_t>(parentWidth));
        parentSurface.faces = parentFaceVertices;
        parentSurface.sourceFaceIndices = parentStableIdentity;

        numi::matter::ObjectSource parentShell;
        parentShell.representation = numi::matter::Representation::fem;
        parentShell.femNodes.assign(topology.parentWorldPoints.begin(), topology.parentWorldPoints.end());
        parentShell.tetrahedra.reserve(topology.parentTetrahedra.size());
        for (const auto& nodes : topology.parentTetrahedra)
            parentShell.tetrahedra.push_back({nodes});

        RegisteredSkinFatReductionSourceLabels parentLabels;
        parentLabels.registeredVertexCount = parentWidth;
        parentLabels.registeredFaceCount = parentFaceVertices.size();
        parentLabels.sourceTetrahedronCount = topology.parentTetrahedra.size();
        parentLabels.registeredMechanicalFaceIndices.resize(parentFaceVertices.size());
        for (std::size_t face = 0u; face < parentFaceVertices.size(); ++face)
            parentLabels.registeredMechanicalFaceIndices[face] = static_cast<std::uint32_t>(face);
        parentLabels.tetrahedronRegionLabels = parentDonor;
        parentLabels.tetrahedronSourceDonorBodyIndices = parentDonor;
        parentLabels.tetrahedronSourceFaceIndices = parentTetToFace;
        parentLabels.tetrahedronOriginalSourceFaceIndices.resize(topology.parentTetrahedra.size());
        parentLabels.tetrahedronSourceParentBodyIndices.resize(topology.parentTetrahedra.size());
        for (std::size_t parent = 0u; parent < topology.parentTetrahedra.size(); ++parent) {
            parentLabels.tetrahedronOriginalSourceFaceIndices[parent] = parentStableFace[parent];
            const auto donor = parentDonor[parent];
            const auto bodyParent = parentBodyAt(donor);
            if ((bodyParent != NM_INVALID_INDEX && bodyParent >= sourceBodyCount) || bodyParent == donor)
                return fail("explicit original parent donor has invalid parent-body identity");
            parentLabels.tetrahedronSourceParentBodyIndices[parent] = bodyParent;
        }
        std::vector<std::uint32_t> parentSplitFaces;
        parentSplitFaces.reserve(uniqueSplitTetrahedra.size());
        for (const std::uint32_t child : uniqueSplitTetrahedra) {
            const std::uint32_t parent = topology.childParentTetrahedra[child];
            parentSplitFaces.push_back(parentTetToFace[parent]);
        }
        std::sort(parentSplitFaces.begin(), parentSplitFaces.end());
        parentSplitFaces.erase(std::unique(parentSplitFaces.begin(), parentSplitFaces.end()),
                               parentSplitFaces.end());
        // All explicit parent/child/material/source validation above is still
        // mandatory for capture. Only the unused basis construction is skipped.
        if (!buildSpatialBasisRegions) {
            if (!uniqueSplitTetrahedra.empty() || maximumRefinementPatchCount != 0u)
                return fail("ancestry-only capture cannot request basis refinement");
            next.tetrahedronBasisRegionLabels = next.tetrahedronRegionLabels;
            output = std::move(next);
            error.clear();
            return true;
        }
        std::vector<std::uint32_t> parentBasisLabels;
        if (!buildRegisteredSkinFatBasisRegionLabels(
                parentSurface, parentShell, parentLabels, parentBasisLabels, error,
                maximumFacesPerBasisRegion,
                std::span<const std::uint32_t>(parentSplitFaces),
                maximumRefinementPatchCount))
            return false;
        if (parentBasisLabels.size() != topology.parentTetrahedra.size())
            return fail("original parent basis-label prolongation lost a parent tetrahedron");
        next.tetrahedronBasisRegionLabels.resize(shell.tetrahedra.size());
        for (std::size_t child = 0u; child < shell.tetrahedra.size(); ++child) {
            const std::uint32_t parent = topology.childParentTetrahedra[child];
            if (parent >= parentBasisLabels.size() || parentBasisLabels[parent] == NM_INVALID_INDEX)
                return fail("explicit child cannot inherit its original parent basis-region identity");
            next.tetrahedronBasisRegionLabels[child] = parentBasisLabels[parent];
        }
    } else {
        std::vector<std::uint32_t> splitSourceFaceIndices;
        splitSourceFaceIndices.reserve(uniqueSplitTetrahedra.size());
        for (const std::uint32_t tet : uniqueSplitTetrahedra)
            splitSourceFaceIndices.push_back(next.tetrahedronSourceFaceIndices[tet]);
        std::sort(splitSourceFaceIndices.begin(), splitSourceFaceIndices.end());
        splitSourceFaceIndices.erase(
            std::unique(splitSourceFaceIndices.begin(), splitSourceFaceIndices.end()),
            splitSourceFaceIndices.end());
        if (!buildRegisteredSkinFatBasisRegionLabels(
                surface, shell, next, next.tetrahedronBasisRegionLabels, error,
                maximumFacesPerBasisRegion,
                std::span<const std::uint32_t>(splitSourceFaceIndices),
                maximumRefinementPatchCount))
            return false;
    }
    output = std::move(next);
    error.clear();
    return true;
}

template<class VerifyBoundBody, class ParentBodyAt>
[[nodiscard]] inline bool buildRegisteredSkinFatReductionSourceLabels(
    const RegisteredSurface& surface,
    const numi::matter::ObjectSource& shell,
    const std::uint32_t layerWidth,
    const std::size_t sourceBodyCount,
    VerifyBoundBody&& verifyBoundBody,
    ParentBodyAt&& parentBodyAt,
    RegisteredSkinFatReductionSourceLabels& output,
    std::string& error,
    const std::uint32_t maximumFacesPerBasisRegion = 36u,
    const std::uint32_t splitSourceTetrahedronIndex = NM_INVALID_INDEX
) {
    std::array<std::uint32_t, 1u> target{splitSourceTetrahedronIndex};
    const std::span<const std::uint32_t> targets =
        splitSourceTetrahedronIndex == NM_INVALID_INDEX
            ? std::span<const std::uint32_t>{}
            : std::span<const std::uint32_t>(target);
    return buildRegisteredSkinFatReductionSourceLabels(
        surface, shell, layerWidth, sourceBodyCount,
        std::forward<VerifyBoundBody>(verifyBoundBody),
        std::forward<ParentBodyAt>(parentBodyAt), output, error,
        maximumFacesPerBasisRegion, targets);
}

} // namespace metalrobo::human::coarse_tissue
