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
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
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

// Adds one deterministic source-face-local subdivision for donor body 20,
// the largest skin-shell strain-sensitivity region in the retained endpoint.
// This changes basis partitions only; all original donor IDs and face IDs stay
// available unchanged for capture and source ancestry.
[[nodiscard]] inline bool buildRegisteredSkinFatBasisRegionLabels(
    const RegisteredSurface& surface,
    const numi::matter::ObjectSource& shell,
    const RegisteredSkinFatReductionSourceLabels& labels,
    std::vector<std::uint32_t>& output,
    std::string& error
) {
    constexpr std::uint32_t refinedDonor = 20u;
    constexpr std::uint32_t firstRefinementLabel = 0x53460001u;
    constexpr std::uint32_t secondRefinementLabel = 0x53460002u;
    const auto fail = [&](const char* message) {
        error = message;
        return false;
    };
    if (labels.tetrahedronRegionLabels.size() != shell.tetrahedra.size() ||
        labels.tetrahedronSourceFaceIndices.size() != shell.tetrahedra.size() ||
        labels.tetrahedronSourceDonorBodyIndices.size() != shell.tetrahedra.size() ||
        surface.faces.empty() || surface.worldPoints.empty() ||
        (!surface.sourceFaceIndices.empty() &&
         surface.sourceFaceIndices.size() != surface.faces.size()))
        return fail("skin/fat basis refinement source-face ancestry is incomplete");

    output = labels.tetrahedronRegionLabels;
    std::map<std::uint32_t, std::array<double, 3>> faceCenterByIdentity;
    for (std::size_t face = 0u; face < surface.faces.size(); ++face) {
        const std::uint32_t identity = surface.sourceFaceIndices.empty()
            ? static_cast<std::uint32_t>(face) : surface.sourceFaceIndices[face];
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

    std::set<std::uint32_t> selectedFaces;
    for (std::size_t tet = 0u; tet < labels.tetrahedronRegionLabels.size(); ++tet) {
        if (labels.tetrahedronSourceDonorBodyIndices[tet] == refinedDonor) {
            if (!faceCenterByIdentity.contains(labels.tetrahedronSourceFaceIndices[tet]))
                return fail("skin/fat refined donor tetrahedron lost its registered face");
            selectedFaces.insert(labels.tetrahedronSourceFaceIndices[tet]);
        }
    }
    if (selectedFaces.size() < 2u) {
        error.clear();
        return true;
    }

    std::array<double, 3> mean{};
    for (const std::uint32_t face : selectedFaces) {
        const auto& center = faceCenterByIdentity.at(face);
        for (std::size_t axis = 0u; axis < 3u; ++axis)
            mean[axis] += center[axis] / double(selectedFaces.size());
    }
    std::array<double, 3> variance{};
    for (const std::uint32_t face : selectedFaces) {
        const auto& center = faceCenterByIdentity.at(face);
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            const double delta = center[axis] - mean[axis];
            variance[axis] += delta * delta;
        }
    }
    std::size_t splitAxis = 0u;
    for (std::size_t axis = 1u; axis < 3u; ++axis)
        if (variance[axis] > variance[splitAxis]) splitAxis = axis;
    std::vector<std::uint32_t> orderedFaces(selectedFaces.begin(), selectedFaces.end());
    std::sort(orderedFaces.begin(), orderedFaces.end(),
        [&](const std::uint32_t a, const std::uint32_t b) {
            const double x = faceCenterByIdentity.at(a)[splitAxis];
            const double y = faceCenterByIdentity.at(b)[splitAxis];
            return x != y ? x < y : a < b;
        });
    const std::size_t split = orderedFaces.size() / 2u;
    if (split == 0u || split == orderedFaces.size())
        return fail("skin/fat basis refinement could not form two nonempty source-face regions");
    std::map<std::uint32_t, std::uint32_t> refinedLabel;
    for (std::size_t i = 0u; i < orderedFaces.size(); ++i)
        refinedLabel.emplace(orderedFaces[i],
            i < split ? firstRefinementLabel : secondRefinementLabel);
    for (std::size_t tet = 0u; tet < output.size(); ++tet) {
        if (labels.tetrahedronSourceDonorBodyIndices[tet] != refinedDonor) continue;
        output[tet] = refinedLabel.at(labels.tetrahedronSourceFaceIndices[tet]);
    }
    if (std::find(output.begin(), output.end(), firstRefinementLabel) == output.end() ||
        std::find(output.begin(), output.end(), secondRefinementLabel) == output.end())
        return fail("skin/fat basis refinement did not retain both nonempty labels");
    error.clear();
    return true;
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


// Source-bound, longitudinal partitions for the two pinned TetGen optic nerves.
// These are deterministic geometric bins derived from the exact source mesh,
// not authored clinical segment labels.
struct RegisteredOpticNerveReductionSourceLabels {
    std::vector<std::uint32_t> tetrahedronRegionLabels;
    std::vector<std::uint32_t> sourceNodeVertexIndices;
    std::vector<std::uint32_t> tetrahedronSourceIndices;
    std::vector<std::uint32_t> boundarySourceFaceIndices;
    std::vector<std::uint32_t> boundarySourceNodeIndices;
    std::string member;
    std::string meshSHA256;
    std::string payloadSHA256;
};

[[nodiscard]] inline bool buildRegisteredOpticNerveReductionSourceLabels(
    const numi::human::optic_nerve::tetgen_assets::AssetView& asset,
    const numi::matter::ObjectSource& source,
    RegisteredOpticNerveReductionSourceLabels& output,
    std::string& error
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
    // The first quartile contained both retained maximum-error optic tets.
    // Split it in half and retain the other three volume quartiles unchanged.
    constexpr std::array<double, 5> cumulativeFractions{
        0.125, 0.25, 0.5, 0.75, 1.0};
    constexpr std::uint32_t bins =
        static_cast<std::uint32_t>(cumulativeFractions.size());
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
        std::span<const std::uint32_t>(labels.tetrahedronBasisRegionLabels);
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
    std::string& error
) {
    using Candidate = numi::matter::CompiledWorldPartitionedReductionBasis;
    if (opticObjects.empty())
        return buildAndInstallRegisteredSkinFatPreconditioner(
            world, skinObjectIndex, skinLabels, skinSource,
            installPayload, massWhiten, error);
    Candidate skin = buildAndInstallRegisteredSkinFatPreconditioner(
        world, skinObjectIndex, skinLabels, skinSource, false,
        massWhiten, error);
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
                *nerve.asset, *nerve.source, piece.labels, error) ||
            !verifyRegisteredOpticNerveCompiledSource(
                world, nerve, piece.labels, error)) {
            Candidate failed;
            failed.objectIndex = skinObjectIndex;
            failed.error = error;
            failed.basis.error = error;
            return failed;
        }
        numi::matter::CompiledWorldPartitionedReductionOptions options;
        options.minimumCoordinateReductionFactor = 4u;
        options.maximumModeCount = 512u;
        options.explicitAuthoredTetrahedronRegionLabels =
            std::span<const std::uint32_t>(piece.labels.tetrahedronRegionLabels);
        options.explicitRegionLabelObjectIndex = nerve.objectIndex;
        options.explicitRegionLabelWorldPhysicsFingerprint = world.physicsFingerprint;
        std::vector<std::uint32_t> fineRows;
        fineRows.reserve(piece.labels.boundarySourceNodeIndices.size());
        const std::uint32_t stateOffset = world.objects[nerve.objectIndex].stateOffset;
        for (const std::uint32_t local : piece.labels.boundarySourceNodeIndices)
            fineRows.push_back(stateOffset + local);
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
            const auto& node = world.fem.nodes[global];
            if (node.restAndFixed.w != 1.0f && node.restAndFixed.w != 3.0f)
                freeCoordinates += 3u;
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
    constexpr std::uint32_t blockModes = 12u;
    if (regions.empty() || regions.size() > 42u ||
        regions.size() * blockModes > 512u ||
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
    if (std::uint64_t(modeCount) * 4u > freeCoordinates) {
        error = "combined source basis misses the four-to-one free-coordinate reduction prerequisite";
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
    combined.basis.minimumCoordinateReductionFactor = 4u;
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
    if (!numi::matter::partitioned_reduction_detail::choleskyPositiveDefinite(
            combined.basis.projectedMass, modeCount)) {
        error = "combined exact cooked-mass projection is not positive definite";
        combined.error = error;
        combined.basis.error = error;
        return combined;
    }
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
    if (!installPayload) {
        error.clear();
        return combined;
    }
    numi::matter::detail::HumanPartitionedPreconditionerSource payload;
    payload.objectIndex = skinObjectIndex; // provenance primary; row owners are validated from topology.
    payload.basisNodeCount = rowCount;
    payload.regionCount = regionCount;
    payload.blockModeCount = blockModes;
    payload.modeCount = modeCount;
    payload.freeCoordinateCount = static_cast<std::uint32_t>(freeCoordinates);
    payload.sensitiveAnchorCount = static_cast<std::uint32_t>(combined.basis.anchorNodes.size());
    payload.minimumCoordinateReductionFactor = 4u;
    payload.sourceWorldPhysicsFingerprint = world.physicsFingerprint;
    payload.globalFEMNodeIndices = combined.globalFEMNodeIndices;
    payload.regionLabels.reserve(regionCount);
    for (const auto& region : combined.basis.regions)
        payload.regionLabels.push_back(region.sourceLabel);
    payload.blockModeIndices.resize(modeCount);
    for (std::uint32_t mode = 0u; mode < modeCount; ++mode)
        payload.blockModeIndices[mode] = mode;
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
    payload.enabled = true;
    world.humanPartitionedPreconditioner = std::move(payload);
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
    next.tetrahedronBasisRegionLabels.reserve(shell.tetrahedra.size());
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
    if (!buildRegisteredSkinFatBasisRegionLabels(
            surface, shell, next, next.tetrahedronBasisRegionLabels, error))
        return false;
    output = std::move(next);
    error.clear();
    return true;
}

} // namespace metalrobo::human::coarse_tissue
