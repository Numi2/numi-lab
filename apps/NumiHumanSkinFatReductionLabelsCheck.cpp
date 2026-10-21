#include "NumiHumanSkinFatReductionLabels.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <map>
#include <span>
#include <cstdint>
#include <iostream>
#include <string>
#include <set>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

void assertProjectedMassParity(
    const numi::matter::CompiledWorldPartitionedReductionBasis& candidate,
    const numi::matter::CompiledWorld& world
) {
    const auto& basis = candidate.basis;
    assert(candidate.valid());
    assert(candidate.globalFEMNodeIndices.size() == basis.nodeCount);
    assert(basis.projectedMass.size() ==
           static_cast<std::size_t>(basis.modeCount) * basis.modeCount);
    std::vector<std::size_t> modesPerRegion(basis.regions.size(), 0u);
    std::map<std::uint32_t, std::size_t> regionByLabel;
    for (std::size_t region = 0u; region < basis.regions.size(); ++region)
        assert(regionByLabel.emplace(basis.regions[region].sourceLabel, region).second);
    for (const auto& mode : basis.modes) {
        if (mode.kind == numi::matter::PartitionedReductionModeKind::anchorTranslation)
            continue;
        const auto found = regionByLabel.find(mode.regionLabel);
        assert(found != regionByLabel.end());
        ++modesPerRegion[found->second];
    }
    for (const auto count : modesPerRegion) assert(count == 12u);
    for (std::uint32_t row = 0u; row < basis.nodeCount; ++row) {
        const auto global = candidate.globalFEMNodeIndices[row];
        assert(global < world.fem.nodes.size());
        assert(std::isfinite(world.fem.nodes[global].positionAndMass.w));
        assert(world.fem.nodes[global].positionAndMass.w > 0.0f);
    }
    for (std::uint32_t i = 0u; i < basis.modeCount; ++i) {
        for (std::uint32_t j = 0u; j < basis.modeCount; ++j) {
            double expected = 0.0;
            for (std::uint32_t row = 0u; row < basis.nodeCount; ++row) {
                const auto global = candidate.globalFEMNodeIndices[row];
                const auto& a = basis.values[
                    static_cast<std::size_t>(i) * basis.nodeCount + row];
                const auto& b = basis.values[
                    static_cast<std::size_t>(j) * basis.nodeCount + row];
                expected += static_cast<double>(world.fem.nodes[global].positionAndMass.w) *
                    (static_cast<double>(a.x) * b.x +
                     static_cast<double>(a.y) * b.y +
                     static_cast<double>(a.z) * b.z);
            }
            const double actual = basis.projectedMass[
                static_cast<std::size_t>(i) * basis.modeCount + j];
            assert(std::isfinite(actual));
            assert(std::abs(actual - expected) <=
                   2.0e-8 * std::max(1.0, std::abs(expected)));
            const double transpose = basis.projectedMass[
                static_cast<std::size_t>(j) * basis.modeCount + i];
            assert(std::abs(actual - transpose) <=
                   2.0e-8 * std::max({1.0, std::abs(actual), std::abs(transpose)}));
        }
    }
}

void runCombinedOpticConsumerTest() {
    using namespace metalrobo::human::coarse_tissue;
    using numi::matter::ObjectSource;
    using numi::matter::Representation;
    using numi::human::optic_nerve::tetgen_assets::AssetView;

    RegisteredSurface surface;
    constexpr std::uint32_t side = 3u;
    for (std::uint32_t y = 0u; y < side; ++y) {
        for (std::uint32_t x = 0u; x < side; ++x) {
            const std::uint32_t vertex = x + side * y;
            surface.worldPoints.push_back({{0.01 * x, 0.01 * y, 0.0}});
            surface.sourceVertexIndices.push_back(1000u + vertex);
            surface.donorBodyIndices.push_back(7u);
        }
    }
    for (std::uint32_t y = 0u; y + 1u < side; ++y) {
        for (std::uint32_t x = 0u; x + 1u < side; ++x) {
            const std::uint32_t a = x + side * y;
            const std::uint32_t b = a + 1u;
            const std::uint32_t c = a + side;
            const std::uint32_t d = c + 1u;
            surface.faces.push_back({a, b, d});
            surface.faces.push_back({a, d, c});
        }
    }
    for (std::uint32_t face = 0u; face < surface.faces.size(); ++face)
        surface.sourceFaceIndices.push_back(2000u + face);

    ObjectSource skin;
    skin.name = "synthetic_registered_skin_fat";
    skin.representation = Representation::fem;
    skin.femNodes.resize(3u * surface.worldPoints.size());
    for (std::uint32_t layer = 0u; layer < 3u; ++layer) {
        for (std::uint32_t vertex = 0u; vertex < surface.worldPoints.size(); ++vertex) {
            skin.femNodes[layer * surface.worldPoints.size() + vertex] = {
                surface.worldPoints[vertex][0], surface.worldPoints[vertex][1],
                0.01 * layer};
        }
    }
    const auto appendTet = [&](std::uint32_t a, std::uint32_t b,
                               std::uint32_t c, std::uint32_t d) {
        skin.tetrahedra.push_back({{{a, b, c, d}}});
    };
    const std::uint32_t layerWidth =
        static_cast<std::uint32_t>(surface.worldPoints.size());
    for (const auto& face : surface.faces) {
        const std::uint32_t a = face[0], b = face[1], c = face[2];
        appendTet(a, b, c, layerWidth + c);
        appendTet(a, b, layerWidth + b, layerWidth + c);
        appendTet(a, layerWidth + a, layerWidth + b, layerWidth + c);
        appendTet(layerWidth + a, layerWidth + b, layerWidth + c,
                  2u * layerWidth + c);
        appendTet(layerWidth + a, layerWidth + b, 2u * layerWidth + b,
                  2u * layerWidth + c);
        appendTet(layerWidth + a, 2u * layerWidth + a, 2u * layerWidth + b,
                  2u * layerWidth + c);
    }
    const auto verifyBinding = [=](std::uint32_t sourceVertex, std::uint32_t donor) {
        return sourceVertex >= 1000u && sourceVertex < 1000u + side * side && donor == 7u;
    };
    const auto parentBody = [](std::uint32_t body) {
        return body == 7u ? 0u : NM_INVALID_INDEX;
    };
    RegisteredSkinFatReductionSourceLabels skinLabels;
    std::string error;
    const bool skinLabelsBuilt = buildRegisteredSkinFatReductionSourceLabels(
        surface, skin, static_cast<std::uint32_t>(surface.worldPoints.size()),
        8u, verifyBinding, parentBody, skinLabels, error);
    if (!skinLabelsBuilt)
        std::cerr << error << std::endl;
    assert(skinLabelsBuilt);
    assert(skinLabels.tetrahedronRegionLabels ==
           skinLabels.tetrahedronBasisRegionLabels);

    // Refinement is basis-local: donor IDs and source-face ancestry remain
    // unchanged while body 20 maps to two nonempty local support labels.
    RegisteredSurface refinedSurface = surface;
    std::fill(refinedSurface.donorBodyIndices.begin(),
              refinedSurface.donorBodyIndices.end(), 20u);
    const auto verifyRefinedBinding = [&](std::uint32_t sourceVertex,
                                          std::uint32_t donorBody) {
        return sourceVertex >= 1000u && sourceVertex < 1000u + side * side &&
            donorBody == 20u;
    };
    const auto refinedParent = [](std::uint32_t) { return NM_INVALID_INDEX; };
    RegisteredSkinFatReductionSourceLabels refinedLabels;
    assert(buildRegisteredSkinFatReductionSourceLabels(
        refinedSurface, skin, static_cast<std::uint32_t>(refinedSurface.worldPoints.size()),
        21u, verifyRefinedBinding, refinedParent, refinedLabels, error, 4u));
    assert(std::all_of(refinedLabels.tetrahedronRegionLabels.begin(),
                       refinedLabels.tetrahedronRegionLabels.end(),
                       [](std::uint32_t label) { return label == 20u; }));
    assert(refinedLabels.tetrahedronSourceFaceIndices ==
           skinLabels.tetrahedronSourceFaceIndices);
    std::map<std::uint32_t, std::size_t> refinedSkinRegions;
    for (const auto label : refinedLabels.tetrahedronBasisRegionLabels)
        ++refinedSkinRegions[label];
    assert(refinedSkinRegions.size() == 2u);
    assert(refinedSkinRegions.contains(0x53460001u));
    assert(refinedSkinRegions.contains(0x53460002u));
    std::vector<std::uint32_t> sourceTetOrder(skin.tetrahedra.size());
    for (std::uint32_t tet = 0u; tet < sourceTetOrder.size(); ++tet)
        sourceTetOrder[tet] = tet;
    RegisteredSkinFatBasisRegionAncestry refinementAncestry;
    assert(mapRegisteredSkinFatBasisRegionsToDonors(
        refinedLabels, refinedLabels.tetrahedronBasisRegionLabels,
        sourceTetOrder, skin.tetrahedra.size(), refinementAncestry, error));
    assert(refinementAncestry.donorBodyByBasisRegion.size() == 2u);
    assert(refinementAncestry.donorBodyByBasisRegion.at(0x53460001u) == 20u);
    assert(refinementAncestry.donorBodyByBasisRegion.at(0x53460002u) == 20u);
    assert(refinementAncestry.tetrahedronCountByBasisRegion.at(0x53460001u) > 0u);
    assert(refinementAncestry.tetrahedronCountByBasisRegion.at(0x53460002u) > 0u);
    std::map<std::uint32_t, std::uint32_t> labelByFace;
    for (std::size_t tet = 0u;
         tet < refinedLabels.tetrahedronBasisRegionLabels.size(); ++tet) {
        const auto [entry, inserted] = labelByFace.emplace(
            refinedLabels.tetrahedronSourceFaceIndices[tet],
            refinedLabels.tetrahedronBasisRegionLabels[tet]);
        assert(inserted || entry->second ==
            refinedLabels.tetrahedronBasisRegionLabels[tet]);
        assert(refinedLabels.tetrahedronRegionLabels[tet] == 20u);
        assert(refinedLabels.tetrahedronSourceDonorBodyIndices[tet] == 20u);
    }
    assert(labelByFace.size() == refinedSurface.faces.size());
    RegisteredSkinFatReductionSourceLabels splitFaceLabels;
    assert(buildRegisteredSkinFatReductionSourceLabels(
        refinedSurface, skin, static_cast<std::uint32_t>(side * side),
        21u, verifyRefinedBinding, refinedParent, splitFaceLabels, error,
        4u, 0u));
    assert(splitFaceLabels.tetrahedronRegionLabels ==
           refinedLabels.tetrahedronRegionLabels);
    assert(splitFaceLabels.tetrahedronSourceFaceIndices ==
           refinedLabels.tetrahedronSourceFaceIndices);
    assert(splitFaceLabels.tetrahedronSourceDonorBodyIndices ==
           refinedLabels.tetrahedronSourceDonorBodyIndices);
    std::map<std::uint32_t, std::set<std::uint32_t>> splitLabelsByFace;
    for (std::size_t tet = 0u;
         tet < splitFaceLabels.tetrahedronBasisRegionLabels.size(); ++tet)
        splitLabelsByFace[splitFaceLabels.tetrahedronSourceFaceIndices[tet]].insert(
            splitFaceLabels.tetrahedronBasisRegionLabels[tet]);
    assert(splitLabelsByFace.at(2000u).size() == 1u);
    assert(splitLabelsByFace.at(2001u).size() == 1u);
    assert(splitLabelsByFace.at(2002u).size() == 1u);
    assert(splitLabelsByFace.at(2003u).size() == 1u);
    std::set<std::uint32_t> splitRegionLabels;
    for (const auto& [face, labelsForFace] : splitLabelsByFace) {
        (void)face;
        splitRegionLabels.insert(labelsForFace.begin(), labelsForFace.end());
    }
    assert(splitRegionLabels.size() == 3u);

    // A multi-witness refinement isolates multiple source-face neighborhoods
    // while keeping donor IDs and exact source-tet/source-face provenance.
    const std::array<std::uint32_t, 3u> shellStrainWitnessTets{0u, 6u, 18u};
    RegisteredSkinFatReductionSourceLabels multiTargetLabels;
    assert(buildRegisteredSkinFatReductionSourceLabels(
        refinedSurface, skin, static_cast<std::uint32_t>(refinedSurface.worldPoints.size()),
        21u, verifyRefinedBinding, refinedParent, multiTargetLabels, error,
        4u, std::span<const std::uint32_t>(shellStrainWitnessTets)));
    assert(multiTargetLabels.tetrahedronRegionLabels ==
           refinedLabels.tetrahedronRegionLabels);
    assert(multiTargetLabels.tetrahedronSourceFaceIndices ==
           refinedLabels.tetrahedronSourceFaceIndices);
    assert(multiTargetLabels.tetrahedronSourceDonorBodyIndices ==
           refinedLabels.tetrahedronSourceDonorBodyIndices);
    std::map<std::uint32_t, std::uint32_t> multiLabelByFace;
    std::set<std::uint32_t> multiBasisLabels;
    for (std::size_t tet = 0u;
         tet < multiTargetLabels.tetrahedronBasisRegionLabels.size(); ++tet) {
        const auto face = multiTargetLabels.tetrahedronSourceFaceIndices[tet];
        const auto basisLabel = multiTargetLabels.tetrahedronBasisRegionLabels[tet];
        const auto [entry, inserted] = multiLabelByFace.emplace(face, basisLabel);
        assert(inserted || entry->second == basisLabel);
        multiBasisLabels.insert(basisLabel);
    }
    assert(multiBasisLabels.size() >= 3u && multiBasisLabels.size() <= 5u);
    const std::array<std::uint32_t, 3u> targetSourceFaces{
        multiTargetLabels.tetrahedronSourceFaceIndices[0u],
        multiTargetLabels.tetrahedronSourceFaceIndices[6u],
        multiTargetLabels.tetrahedronSourceFaceIndices[18u]};
    std::set<std::uint32_t> targetBasisLabels;
    for (const auto face : targetSourceFaces)
        targetBasisLabels.insert(multiLabelByFace.at(face));
    assert(targetBasisLabels.size() == targetSourceFaces.size());
    RegisteredSkinFatBasisRegionAncestry multiAncestry;
    assert(mapRegisteredSkinFatBasisRegionsToDonors(
        multiTargetLabels, multiTargetLabels.tetrahedronBasisRegionLabels,
        sourceTetOrder, skin.tetrahedra.size(), multiAncestry, error));
    assert(multiAncestry.donorBodyByBasisRegion.size() ==
           multiBasisLabels.size());
    for (const auto label : multiBasisLabels)
        assert(multiAncestry.donorBodyByBasisRegion.at(label) == 20u);

    // The producer may use the supported 170-region/2040-mode boundary, but
    // must reject one more region instead of truncating or overflowing it.
    const auto buildCapacitySource = [&](std::uint32_t faceCount,
                                         RegisteredSurface& capacitySurface,
                                         ObjectSource& capacitySkin) {
        const std::uint32_t width = 3u * faceCount;
        for (std::uint32_t face = 0u; face < faceCount; ++face) {
            const std::uint32_t first = 3u * face;
            const double x = static_cast<double>(face);
            capacitySurface.worldPoints.push_back({{x, 0.0, 0.0}});
            capacitySurface.worldPoints.push_back({{x + 0.25, 0.0, 0.0}});
            capacitySurface.worldPoints.push_back({{x, 0.25, 0.0}});
            for (std::uint32_t corner = 0u; corner < 3u; ++corner) {
                capacitySurface.sourceVertexIndices.push_back(6000u + first + corner);
                capacitySurface.donorBodyIndices.push_back(20u);
            }
            capacitySurface.faces.push_back({first, first + 1u, first + 2u});
            capacitySurface.sourceFaceIndices.push_back(50000u + face);
        }
        capacitySkin.name = "synthetic_basis_capacity_skin";
        capacitySkin.representation = Representation::fem;
        capacitySkin.femNodes.resize(3u * width);
        for (std::uint32_t layer = 0u; layer < 3u; ++layer) {
            for (std::uint32_t vertex = 0u; vertex < width; ++vertex) {
                const auto& point = capacitySurface.worldPoints[vertex];
                capacitySkin.femNodes[layer * width + vertex] = {
                    point[0], point[1], 0.01 * static_cast<double>(layer)};
            }
        }
        const auto appendCapacityTet = [&](std::uint32_t a, std::uint32_t b,
                                           std::uint32_t c, std::uint32_t d) {
            capacitySkin.tetrahedra.push_back({{{a, b, c, d}}});
        };
        for (const auto& face : capacitySurface.faces) {
            const auto a = face[0], b = face[1], c = face[2];
            appendCapacityTet(a, b, c, width + c);
            appendCapacityTet(a, b, width + b, width + c);
            appendCapacityTet(a, width + a, width + b, width + c);
            appendCapacityTet(width + a, width + b, width + c, 2u * width + c);
            appendCapacityTet(width + a, width + b, 2u * width + b, 2u * width + c);
            appendCapacityTet(width + a, 2u * width + a, 2u * width + b, 2u * width + c);
        }
    };
    const auto verifyCapacityBinding = [](std::uint32_t sourceVertex,
                                          std::uint32_t donor) {
        return sourceVertex >= 6000u && donor == 20u;
    };
    const auto noCapacityParent = [](std::uint32_t) { return NM_INVALID_INDEX; };
    RegisteredSurface capacitySurface;
    ObjectSource capacitySkin;
    buildCapacitySource(
        numi::matter::kHumanPartitionedBasisMaximumRegionCount,
        capacitySurface, capacitySkin);
    RegisteredSkinFatReductionSourceLabels capacityLabels;
    assert(buildRegisteredSkinFatReductionSourceLabels(
        capacitySurface, capacitySkin, 3u * static_cast<std::uint32_t>(
            numi::matter::kHumanPartitionedBasisMaximumRegionCount),
        21u, verifyCapacityBinding, noCapacityParent, capacityLabels, error,
        1u, NM_INVALID_INDEX));
    std::set<std::uint32_t> capacityRegions(
        capacityLabels.tetrahedronBasisRegionLabels.begin(),
        capacityLabels.tetrahedronBasisRegionLabels.end());
    assert(capacityRegions.size() ==
           numi::matter::kHumanPartitionedBasisMaximumRegionCount);
    assert(numi::matter::kHumanPartitionedBasisMaximumModeCount == 2040u);
    RegisteredSurface overCapacitySurface;
    ObjectSource overCapacitySkin;
    buildCapacitySource(
        numi::matter::kHumanPartitionedBasisMaximumRegionCount + 1u,
        overCapacitySurface, overCapacitySkin);
    RegisteredSkinFatReductionSourceLabels overCapacityLabels;
    assert(!buildRegisteredSkinFatReductionSourceLabels(
        overCapacitySurface, overCapacitySkin, 3u * (
            numi::matter::kHumanPartitionedBasisMaximumRegionCount + 1u),
        21u, verifyCapacityBinding, noCapacityParent, overCapacityLabels, error,
        1u, NM_INVALID_INDEX));
    assert(error.find("compiled Human basis region budget") != std::string::npos);

    // Multiple target tetrahedra from one prism face request one subdivision,
    // not duplicate source-face work. Duplicate tetrahedra still fail closed.
    const std::array<std::uint32_t, 2u> sameFaceTets{0u, 1u};
    RegisteredSkinFatReductionSourceLabels sameFaceLabels;
    assert(buildRegisteredSkinFatReductionSourceLabels(
        refinedSurface, skin, static_cast<std::uint32_t>(refinedSurface.worldPoints.size()),
        21u, verifyRefinedBinding, refinedParent, sameFaceLabels, error,
        4u, std::span<const std::uint32_t>(sameFaceTets)));
    const std::array<std::uint32_t, 2u> repeatedTargetTet{0u, 0u};
    RegisteredSkinFatReductionSourceLabels rejectedMultiLabels;
    assert(!buildRegisteredSkinFatReductionSourceLabels(
        refinedSurface, skin, static_cast<std::uint32_t>(refinedSurface.worldPoints.size()),
        21u, verifyRefinedBinding, refinedParent, rejectedMultiLabels, error,
        4u, std::span<const std::uint32_t>(repeatedTargetTet)));
    assert(error.find("repeats a source tetrahedron") != std::string::npos);
    const std::array<std::uint32_t, 1u> outsideTargetTet{
        static_cast<std::uint32_t>(skin.tetrahedra.size())};
    assert(!buildRegisteredSkinFatReductionSourceLabels(
        refinedSurface, skin, static_cast<std::uint32_t>(refinedSurface.worldPoints.size()),
        21u, verifyRefinedBinding, refinedParent, rejectedMultiLabels, error,
        4u, std::span<const std::uint32_t>(outsideTargetTet)));
    assert(error.find("outside the exact source mesh") != std::string::npos);

    std::vector<std::uint32_t> repeatedSpatialLabels;
    assert(buildRegisteredSkinFatBasisRegionLabels(
        refinedSurface, skin, refinedLabels, repeatedSpatialLabels, error, 4u));
    assert(repeatedSpatialLabels == refinedLabels.tetrahedronBasisRegionLabels);
    auto wrongSourceOrder = sourceTetOrder;
    std::swap(wrongSourceOrder[0], wrongSourceOrder[1]);
    assert(!mapRegisteredSkinFatBasisRegionsToDonors(
        refinedLabels, refinedLabels.tetrahedronBasisRegionLabels,
        wrongSourceOrder, skin.tetrahedra.size(), refinementAncestry, error));
    assert(error.find("source-tet order") != std::string::npos);

    // Spatial chunks never merge original donor identities, even when both
    // donors are subdivided in the same registered surface.
    RegisteredSurface mixedDonorSurface = surface;
    for (std::uint32_t vertex = 0u; vertex < side * side; ++vertex)
        mixedDonorSurface.donorBodyIndices[vertex] =
            (vertex % side == 0u) ? 20u : 7u;
    const auto verifyMixedDonor = [&](std::uint32_t sourceVertex,
                                      std::uint32_t donorBody) {
        if (sourceVertex < 1000u || sourceVertex >= 1000u + side * side)
            return false;
        const std::uint32_t local = sourceVertex - 1000u;
        return donorBody == mixedDonorSurface.donorBodyIndices[local];
    };
    const auto noMixedParent = [](std::uint32_t) { return NM_INVALID_INDEX; };
    RegisteredSkinFatReductionSourceLabels mixedLabels;
    assert(buildRegisteredSkinFatReductionSourceLabels(
        mixedDonorSurface, skin, static_cast<std::uint32_t>(side * side),
        21u, verifyMixedDonor, noMixedParent, mixedLabels, error, 2u));
    RegisteredSkinFatBasisRegionAncestry mixedAncestry;
    assert(mapRegisteredSkinFatBasisRegionsToDonors(
        mixedLabels, mixedLabels.tetrahedronBasisRegionLabels,
        sourceTetOrder, skin.tetrahedra.size(), mixedAncestry, error));
    assert(mixedAncestry.donorBodyByBasisRegion.size() >= 2u);
    for (std::size_t tet = 0u; tet < mixedLabels.tetrahedronRegionLabels.size(); ++tet) {
        const auto basisRegion = mixedLabels.tetrahedronBasisRegionLabels[tet];
        assert(mixedAncestry.donorBodyByBasisRegion.at(basisRegion) ==
               mixedLabels.tetrahedronSourceDonorBodyIndices[tet]);
        assert(mixedLabels.tetrahedronRegionLabels[tet] ==
               mixedLabels.tetrahedronSourceDonorBodyIndices[tet]);
    }

    using namespace numi::human::optic_nerve::tetgen_assets;
    const std::array<AssetView, 2> assets{kLeftAsset, kRightAsset};
    std::array<ObjectSource, 2> opticSources;
    for (std::size_t sideIndex = 0u; sideIndex < assets.size(); ++sideIndex) {
        auto& source = opticSources[sideIndex];
        source.name = std::string(assets[sideIndex].member);
        source.representation = Representation::fem;
        source.femNodes.assign(assets[sideIndex].donorLocalPoints.begin(),
                               assets[sideIndex].donorLocalPoints.end());
        source.tetrahedra.reserve(assets[sideIndex].tetrahedra.size());
        for (const auto& tet : assets[sideIndex].tetrahedra)
            source.tetrahedra.push_back({tet});
        std::set<std::uint32_t> boundary;
        for (const auto& face : assets[sideIndex].boundaryFaces)
            boundary.insert(face.begin(), face.end());
        source.femContactNodes.assign(boundary.begin(), boundary.end());
        // Model the three sparse donor-reference support anchors used by the
        // registered optic source; they remain physical fixed tags regardless
        // of whether the rest of the contact boundary is represented in B.
        assert(source.femContactNodes.size() >= 3u);
        source.femFixedNodes.assign(source.femContactNodes.begin(),
                                    source.femContactNodes.begin() + 3);
    }

    numi::matter::CompiledWorld world;
    world.physicsFingerprint = 0xabcdef0123456789ULL;
    world.mixedSolver.executionBudgets.z = 6u;
    world.objects.resize(3u);
    const auto appendObject = [&](const std::uint32_t objectIndex,
                                  const ObjectSource& source) {
        auto& object = world.objects[objectIndex];
        object.representation = NM_REPRESENTATION_FEM;
        object.stateOffset = static_cast<std::uint32_t>(world.fem.nodes.size());
        object.stateCount = static_cast<std::uint32_t>(source.femNodes.size());
        object.elementOffset = static_cast<std::uint32_t>(world.fem.tetrahedra.size());
        object.elementCount = static_cast<std::uint32_t>(source.tetrahedra.size());
        object.topologyGeneration = 1u;
        for (std::uint32_t local = 0u; local < object.stateCount; ++local) {
            const auto& p = source.femNodes[local];
            const float mass = 1.0f + 1.0e-5f *
                static_cast<float>(world.fem.nodes.size());
            NMFEMNodeStateGPU node{};
            node.positionAndMass = {static_cast<float>(p[0]), static_cast<float>(p[1]),
                                    static_cast<float>(p[2]), mass};
            const bool physicallyFixed = std::binary_search(
                source.femFixedNodes.begin(), source.femFixedNodes.end(), local);
            node.restAndFixed = {static_cast<float>(p[0]), static_cast<float>(p[1]),
                                 static_cast<float>(p[2]), physicallyFixed ? 1.0f : 0.0f};
            world.fem.nodes.push_back(node);
            NMFEMTopologyNodeGPU topology{};
            topology.identity = {local, objectIndex, 1u, NM_TOPOLOGY_ACTIVE};
            world.fem.topologyNodes.push_back(topology);
        }
        for (std::uint32_t local = 0u; local < object.elementCount; ++local) {
            const auto& sourceNodes = source.tetrahedra[local].nodes;
            NMTetrahedronGPU tet{};
            tet.nodes = {object.stateOffset + sourceNodes[0],
                         object.stateOffset + sourceNodes[1],
                         object.stateOffset + sourceNodes[2],
                         object.stateOffset + sourceNodes[3]};
            tet.identity = {local, objectIndex, 1u, NM_OBJECT_ACTIVE};
            world.fem.tetrahedra.push_back(tet);
        }
    };
    appendObject(0u, skin);
    appendObject(1u, opticSources[0]);
    appendObject(2u, opticSources[1]);
    world.dispatch.objectCount = 3u;
    world.dispatch.femNodeCount = static_cast<std::uint32_t>(world.fem.nodes.size());
    world.dispatch.tetrahedronCount = static_cast<std::uint32_t>(world.fem.tetrahedra.size());

    NMFEMHumanAttachmentGPU attachment{};
    attachment.identity = {world.objects[0].stateOffset, 0u, 0u, 1u};
    world.fem.humanAttachments.push_back(attachment);
    std::vector<float> sourceMasses;
    sourceMasses.reserve(world.fem.nodes.size());
    for (const auto& node : world.fem.nodes)
        sourceMasses.push_back(node.positionAndMass.w);

    const std::array<RegisteredOpticNerveReductionObject, 2> opticObjects{{
        {1u, &opticSources[0], &assets[0]},
        {2u, &opticSources[1], &assets[1]},
    }};
    const auto combined = buildAndInstallRegisteredSkinFatOpticPreconditioner(
        world, 0u, skinLabels, skin,
        std::span<const RegisteredOpticNerveReductionObject>(opticObjects),
        true, false, error);
    assert(combined.valid() && error.empty());
    const auto& payload = world.humanPartitionedPreconditioner;
    assert(payload.enabled && payload.objectIndex == 0u);
    assert(payload.basisNodeCount == world.fem.nodes.size());
    assert(payload.regionCount == 11u && payload.modeCount == 132u);
    assert(payload.globalFEMNodeIndices.size() == world.fem.nodes.size());
    assert(std::is_sorted(payload.globalFEMNodeIndices.begin(),
                          payload.globalFEMNodeIndices.end()));
    assert(payload.blockModeIndices.size() == payload.modeCount);
    assert(payload.basisValues.size() ==
           static_cast<std::size_t>(payload.modeCount) * payload.basisNodeCount);
    assert(payload.sensitiveAnchorCount > 0u);

    RegisteredSkinFatReductionSourceLabels targetedSkinLabels;
    assert(buildRegisteredSkinFatReductionSourceLabels(
        surface, skin, static_cast<std::uint32_t>(surface.worldPoints.size()),
        8u, verifyBinding, parentBody, targetedSkinLabels, error, 36u, 0u));
    const auto targetedSkin = buildAndInstallRegisteredSkinFatPreconditioner(
        world, 0u, targetedSkinLabels, skin, false, true, error,
        false, {}, 2u, false);
    assert(targetedSkin.valid() && error.empty());
    assert(targetedSkin.basis.regions.size() == 2u);
    assert(targetedSkin.basis.freeCoordinateCount >=
           2u * targetedSkin.basis.modeCount);
    assert(targetedSkin.globalFEMNodeIndices.size() == skin.femNodes.size());
    for (std::uint32_t row = 0u;
         row < targetedSkin.globalFEMNodeIndices.size(); ++row)
        assert(targetedSkin.globalFEMNodeIndices[row] ==
               world.objects[0u].stateOffset + row);
    assert(targetedSkin.sourceTetrahedronIndices.size() == skin.tetrahedra.size());
    for (std::uint32_t tet = 0u;
         tet < targetedSkin.sourceTetrahedronIndices.size(); ++tet)
        assert(targetedSkin.sourceTetrahedronIndices[tet] == tet);
    assertProjectedMassParity(targetedSkin, world);

    const auto tenBinCombined = buildAndInstallRegisteredSkinFatOpticPreconditioner(
        world, 0u, skinLabels, skin,
        std::span<const RegisteredOpticNerveReductionObject>(opticObjects),
        false, true, error, true, {}, 2u, false, true, nullptr, 10u);
    assert(tenBinCombined.valid() && error.empty());
    assert(tenBinCombined.basis.regions.size() == 21u);
    assert(tenBinCombined.basis.modeCount >= 21u * 12u);
    assert(tenBinCombined.globalFEMNodeIndices == combined.globalFEMNodeIndices);
    assert(tenBinCombined.sourceNodeIndices == combined.sourceNodeIndices);
    assert(tenBinCombined.sourceTetrahedronIndices ==
           combined.sourceTetrahedronIndices);
    assert(tenBinCombined.tetrahedronRegionLabels.size() ==
           combined.tetrahedronRegionLabels.size());
    assertProjectedMassParity(tenBinCombined, world);

    // Every source row keeps its compiled owner and source mass. The map is
    // complete, globally sorted, and has no cross-object or duplicate rows.
    for (std::uint32_t row = 0u; row < payload.basisNodeCount; ++row) {
        assert(payload.globalFEMNodeIndices[row] == row);
        assert(world.fem.topologyNodes[row].identity.y < 3u);
        assert(world.fem.nodes[row].positionAndMass.w == sourceMasses[row]);
    }
    for (std::size_t region = 0u; region < payload.regionLabels.size(); ++region) {
        const std::uint32_t label = payload.regionLabels[region];
        const std::uint32_t expectedOwner = label == 7u ? 0u :
            label >= 0xF1772001u && label <= 0xF1772005u ? 1u :
            label >= 0xF1819001u && label <= 0xF1819005u ? 2u : NM_INVALID_INDEX;
        assert(expectedOwner != NM_INVALID_INDEX);
        bool supported = false;
        for (std::uint32_t localMode = 0u; localMode < 12u; ++localMode) {
            const std::uint32_t mode = payload.blockModeIndices[region * 12u + localMode];
            assert(mode < payload.modeCount);
            bool modeSupported = false;
            for (std::uint32_t row = 0u; row < payload.basisNodeCount; ++row) {
                const auto& value = payload.basisValues[
                    static_cast<std::size_t>(mode) * payload.basisNodeCount + row];
                if (value.x != 0.0f || value.y != 0.0f || value.z != 0.0f) {
                    assert(world.fem.topologyNodes[row].identity.y == expectedOwner);
                    modeSupported = true;
                }
            }
            supported = supported || modeSupported;
            assert(modeSupported);
        }
        assert(supported);
    }

    // Pinned optic boundary/contact rows remain in the fine complement: the
    // payload includes them, but every coarse xyz coefficient is exactly zero.
    for (std::size_t optic = 0u; optic < opticObjects.size(); ++optic) {
        const std::uint32_t offset = world.objects[opticObjects[optic].objectIndex].stateOffset;
        std::set<std::uint32_t> boundary;
        for (const auto& face : opticObjects[optic].asset->boundaryFaces)
            for (const auto local : face) boundary.insert(offset + local);
        for (const auto global : boundary) {
            const auto found = std::lower_bound(payload.globalFEMNodeIndices.begin(),
                                                payload.globalFEMNodeIndices.end(), global);
            assert(found != payload.globalFEMNodeIndices.end() && *found == global);
            const std::size_t row = static_cast<std::size_t>(
                found - payload.globalFEMNodeIndices.begin());
            for (std::uint32_t mode = 0u; mode < payload.modeCount; ++mode) {
                const auto& value = payload.basisValues[
                    static_cast<std::size_t>(mode) * payload.basisNodeCount + row];
                assert(value.x == 0.0f && value.y == 0.0f && value.z == 0.0f);
            }
        }
    }

    // The opt-in embedded-contact policy leaves source contact geometry and
    // boundary eligibility intact, but lets the existing B map carry surface
    // motion. Contact forces are still full-row forces and must be restricted
    // by the adjoint B^T, including forces applied only at boundary nodes.
    const auto embeddedBoundary = buildAndInstallRegisteredSkinFatOpticPreconditioner(
        world, 0u, skinLabels, skin,
        std::span<const RegisteredOpticNerveReductionObject>(opticObjects),
        false, false, error, true, {}, 4u, false, true);
    assert(embeddedBoundary.valid() && error.empty());
    assert(embeddedBoundary.globalFEMNodeIndices == combined.globalFEMNodeIndices);
    assert(embeddedBoundary.sourceNodeIndices == combined.sourceNodeIndices);
    assert(embeddedBoundary.tetrahedronRegionLabels == combined.tetrahedronRegionLabels);
    assert(embeddedBoundary.basis.anchorNodes == combined.basis.anchorNodes);
    assert(embeddedBoundary.anchorReasonMasks == combined.anchorReasonMasks);
    assert(embeddedBoundary.basis.freeCoordinateCount > combined.basis.freeCoordinateCount);
    assert(embeddedBoundary.basis.values.size() ==
           static_cast<std::size_t>(embeddedBoundary.basis.modeCount) *
               embeddedBoundary.basis.nodeCount);
    const std::uint32_t protectedBoundaryGlobal =
        world.objects[opticObjects[0].objectIndex].stateOffset +
        opticSources[0].femContactNodes.front();
    const std::array<std::uint32_t, 1> alreadyProtectedResidual{{
        protectedBoundaryGlobal}};
    const auto overlapRefinement = buildAndInstallRegisteredSkinFatOpticPreconditioner(
        world, 0u, skinLabels, skin,
        std::span<const RegisteredOpticNerveReductionObject>(opticObjects),
        false, false, error, false,
        std::span<const std::uint32_t>(alreadyProtectedResidual),
        4u, false, false);
    assert(overlapRefinement.valid() && error.empty());
    const auto protectedRow = std::lower_bound(
        overlapRefinement.globalFEMNodeIndices.begin(),
        overlapRefinement.globalFEMNodeIndices.end(), protectedBoundaryGlobal);
    assert(protectedRow != overlapRefinement.globalFEMNodeIndices.end() &&
           *protectedRow == protectedBoundaryGlobal);
    const auto protectedRowIndex = static_cast<std::size_t>(
        protectedRow - overlapRefinement.globalFEMNodeIndices.begin());
    for (std::uint32_t mode = 0u;
         mode < overlapRefinement.basis.modeCount; ++mode) {
        const auto& value = overlapRefinement.basis.values[
            static_cast<std::size_t>(mode) * overlapRefinement.basis.nodeCount +
            protectedRowIndex];
        assert(value.x == 0.0f && value.y == 0.0f && value.z == 0.0f);
    }
    assert(opticSources[0].deformableContact && opticSources[0].deformableSelfContact);
    assert(opticSources[1].deformableContact && opticSources[1].deformableSelfContact);
    for (std::size_t optic = 0u; optic < assets.size(); ++optic) {
        std::set<std::uint32_t> authoredBoundary;
        for (const auto& face : assets[optic].boundaryFaces)
            authoredBoundary.insert(face.begin(), face.end());
        assert(opticSources[optic].femContactNodes.size() == authoredBoundary.size());
    }

    std::set<std::uint32_t> contactBoundaryRows;
    for (std::size_t optic = 0u; optic < opticObjects.size(); ++optic) {
        const std::uint32_t offset = world.objects[opticObjects[optic].objectIndex].stateOffset;
        for (const std::uint32_t fixedLocal : opticSources[optic].femFixedNodes) {
            const std::uint32_t global = offset + fixedLocal;
            assert(world.fem.nodes[global].restAndFixed.w == 1.0f);
            const auto fixedRow = std::lower_bound(
                embeddedBoundary.globalFEMNodeIndices.begin(),
                embeddedBoundary.globalFEMNodeIndices.end(), global);
            assert(fixedRow != embeddedBoundary.globalFEMNodeIndices.end() && *fixedRow == global);
            const std::size_t row = static_cast<std::size_t>(
                fixedRow - embeddedBoundary.globalFEMNodeIndices.begin());
            for (std::uint32_t mode = 0u; mode < embeddedBoundary.basis.modeCount; ++mode) {
                const auto& value = embeddedBoundary.basis.values[
                    static_cast<std::size_t>(mode) * embeddedBoundary.basis.nodeCount + row];
                assert(value.x == 0.0f && value.y == 0.0f && value.z == 0.0f);
            }
        }
        std::set<std::uint32_t> boundary;
        for (const auto& face : opticObjects[optic].asset->boundaryFaces)
            for (const auto local : face) boundary.insert(offset + local);
        std::size_t embeddedRows = 0u;
        for (const std::uint32_t global : boundary) {
            contactBoundaryRows.insert(global);
            const auto found = std::lower_bound(
                embeddedBoundary.globalFEMNodeIndices.begin(),
                embeddedBoundary.globalFEMNodeIndices.end(), global);
            assert(found != embeddedBoundary.globalFEMNodeIndices.end() && *found == global);
            const std::size_t row = static_cast<std::size_t>(
                found - embeddedBoundary.globalFEMNodeIndices.begin());
            bool supported = false;
            for (std::uint32_t mode = 0u; mode < embeddedBoundary.basis.modeCount; ++mode) {
                const auto& value = embeddedBoundary.basis.values[
                    static_cast<std::size_t>(mode) * embeddedBoundary.basis.nodeCount + row];
                supported = supported || value.x != 0.0f || value.y != 0.0f || value.z != 0.0f;
            }
            embeddedRows += supported ? 1u : 0u;
        }
        assert(embeddedRows > 0u);
        assert(embeddedRows < boundary.size()); // fixed anchors/witnesses remain protected.
    }
    std::size_t supportedSkinBoundaryRows = 0u;
    for (std::size_t local = 0u; local < skinLabels.registeredVertexCount; ++local) {
        const std::uint32_t global = world.objects[0u].stateOffset +
            static_cast<std::uint32_t>(local);
        contactBoundaryRows.insert(global);
        const auto found = std::lower_bound(
            embeddedBoundary.globalFEMNodeIndices.begin(),
            embeddedBoundary.globalFEMNodeIndices.end(), global);
        assert(found != embeddedBoundary.globalFEMNodeIndices.end() && *found == global);
        const std::size_t row = static_cast<std::size_t>(
            found - embeddedBoundary.globalFEMNodeIndices.begin());
        for (std::uint32_t mode = 0u; mode < embeddedBoundary.basis.modeCount; ++mode) {
            const auto& value = embeddedBoundary.basis.values[
                static_cast<std::size_t>(mode) * embeddedBoundary.basis.nodeCount + row];
            if (value.x != 0.0f || value.y != 0.0f || value.z != 0.0f) {
                ++supportedSkinBoundaryRows;
                break;
            }
        }
    }
    assert(supportedSkinBoundaryRows > 0u);
    assert(!contactBoundaryRows.empty());
    std::vector<double> boundaryQ(embeddedBoundary.basis.modeCount);
    std::vector<double> boundaryBTForce(embeddedBoundary.basis.modeCount, 0.0);
    for (std::uint32_t mode = 0u; mode < embeddedBoundary.basis.modeCount; ++mode)
        boundaryQ[mode] = (mode % 2u == 0u ? 1.0 : -1.0) * 1.0e-4 * (mode + 1u);
    double boundaryContactWork = 0.0;
    for (const std::uint32_t global : contactBoundaryRows) {
        const auto found = std::lower_bound(
            embeddedBoundary.globalFEMNodeIndices.begin(),
            embeddedBoundary.globalFEMNodeIndices.end(), global);
        assert(found != embeddedBoundary.globalFEMNodeIndices.end() && *found == global);
        const std::size_t row = static_cast<std::size_t>(
            found - embeddedBoundary.globalFEMNodeIndices.begin());
        const std::array<double, 3> contactForce{
            0.25 + 1.0e-5 * global, -0.5 + 2.0e-5 * global,
            0.75 - 1.0e-5 * global};
        std::array<double, 3> lifted{};
        for (std::uint32_t mode = 0u; mode < embeddedBoundary.basis.modeCount; ++mode) {
            const auto& basis = embeddedBoundary.basis.values[
                static_cast<std::size_t>(mode) * embeddedBoundary.basis.nodeCount + row];
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                const double coefficient = axis == 0u ? basis.x : axis == 1u ? basis.y : basis.z;
                lifted[axis] += coefficient * boundaryQ[mode];
                boundaryBTForce[mode] += coefficient * contactForce[axis];
            }
        }
        for (std::size_t axis = 0u; axis < 3u; ++axis)
            boundaryContactWork += contactForce[axis] * lifted[axis];
    }
    double boundaryReducedWork = 0.0;
    for (std::size_t mode = 0u; mode < boundaryQ.size(); ++mode)
        boundaryReducedWork += boundaryQ[mode] * boundaryBTForce[mode];
    assert(std::abs(boundaryContactWork - boundaryReducedWork) <=
           2.0e-6 * std::max(1.0, std::abs(boundaryContactWork)));

    // Hybrid coordinates keep the registered outer skin layer in the exact
    // fine complement while retaining the existing regional basis on the
    // remaining skin/fat layers and optic interiors.
    const auto layerZeroFine = buildAndInstallRegisteredSkinFatOpticPreconditioner(
        world, 0u, skinLabels, skin,
        std::span<const RegisteredOpticNerveReductionObject>(opticObjects),
        false, false, error, true);
    assert(layerZeroFine.valid() && error.empty());
    assert(layerZeroFine.globalFEMNodeIndices == combined.globalFEMNodeIndices);
    assert(layerZeroFine.sourceNodeIndices == combined.sourceNodeIndices);
    assert(layerZeroFine.tetrahedronRegionLabels == combined.tetrahedronRegionLabels);
    assert(layerZeroFine.basis.modeCount == combined.basis.modeCount);
    assert(layerZeroFine.basis.freeCoordinateCount < combined.basis.freeCoordinateCount);
    assert(layerZeroFine.basis.partitionWeights.size() ==
           static_cast<std::size_t>(layerZeroFine.basis.nodeCount) *
               layerZeroFine.basis.partitionCount);
    bool laterSkinLayerHasSupport = false;
    for (std::size_t local = 0u;
         local < skinLabels.sourceNodeLayerIndices.size(); ++local) {
        const std::uint32_t global = world.objects[0u].stateOffset +
            static_cast<std::uint32_t>(local);
        const auto found = std::lower_bound(
            layerZeroFine.globalFEMNodeIndices.begin(),
            layerZeroFine.globalFEMNodeIndices.end(), global);
        assert(found != layerZeroFine.globalFEMNodeIndices.end() && *found == global);
        const std::size_t row = static_cast<std::size_t>(
            found - layerZeroFine.globalFEMNodeIndices.begin());
        bool rowHasSupport = false;
        for (std::uint32_t mode = 0u; mode < layerZeroFine.basis.modeCount; ++mode) {
            const auto& value = layerZeroFine.basis.values[
                static_cast<std::size_t>(mode) * layerZeroFine.basis.nodeCount + row];
            rowHasSupport = rowHasSupport || value.x != 0.0f ||
                value.y != 0.0f || value.z != 0.0f;
        }
        if (skinLabels.sourceNodeLayerIndices[local] == 0u) {
            assert(!rowHasSupport);
            for (std::uint32_t partition = 0u;
                 partition < layerZeroFine.basis.partitionCount; ++partition)
                assert(layerZeroFine.basis.partitionWeights[
                    row * layerZeroFine.basis.partitionCount + partition] == 0.0);
        } else {
            laterSkinLayerHasSupport = laterSkinLayerHasSupport || rowHasSupport;
        }
    }
    assert(laterSkinLayerHasSupport);
    assert(layerZeroFine.basis.freeCoordinateCount >=
           4u * layerZeroFine.basis.modeCount);

    // Left optic tet 92 and all tetrahedra sharing one of its vertices stay
    // on the exact fine complement, while source ancestry remains unchanged.
    {
        const auto& left = *opticObjects[0].asset;
        assert(left.member == "FJ1772" && left.tetrahedra.size() > 92u);
        RegisteredOpticNerveReductionSourceLabels leftLabels;
        assert(buildRegisteredOpticNerveReductionSourceLabels(
            left, *opticObjects[0].source, leftLabels, error));
        std::set<std::uint32_t> witnessVertices(
            left.tetrahedra[92].begin(), left.tetrahedra[92].end());
        assert(witnessVertices.size() == 4u);
        std::set<std::uint32_t> expectedFine;
        for (const auto& tet : left.tetrahedra) {
            bool incident = false;
            for (const auto node : tet)
                incident = incident || witnessVertices.contains(node);
            if (incident) expectedFine.insert(tet.begin(), tet.end());
        }
        assert(std::vector<std::uint32_t>(
            expectedFine.begin(), expectedFine.end()) ==
            leftLabels.witnessSourceNodeIndices);
        const std::uint32_t offset =
            world.objects[opticObjects[0].objectIndex].stateOffset;
        for (const auto local : expectedFine) {
            const std::uint32_t global = offset + local;
            const auto found = std::lower_bound(
                payload.globalFEMNodeIndices.begin(),
                payload.globalFEMNodeIndices.end(), global);
            assert(found != payload.globalFEMNodeIndices.end() && *found == global);
            const std::size_t row = static_cast<std::size_t>(
                found - payload.globalFEMNodeIndices.begin());
            for (std::uint32_t mode = 0u; mode < payload.modeCount; ++mode) {
                const auto& value = payload.basisValues[
                    static_cast<std::size_t>(mode) * payload.basisNodeCount + row];
                assert(value.x == 0.0f && value.y == 0.0f && value.z == 0.0f);
            }
            for (std::uint32_t mode = 0u;
                 mode < embeddedBoundary.basis.modeCount; ++mode) {
                const auto& embeddedValue = embeddedBoundary.basis.values[
                    static_cast<std::size_t>(mode) * embeddedBoundary.basis.nodeCount + row];
                assert(embeddedValue.x == 0.0f && embeddedValue.y == 0.0f &&
                       embeddedValue.z == 0.0f);
            }
        }
    }

    // Independently reconstruct B^T M B from the installed float payload and
    // check virtual work for an arbitrary fine force and reduced coordinate.
    std::vector<double> q(payload.modeCount), coarseForce(payload.modeCount, 0.0);
    for (std::uint32_t mode = 0u; mode < payload.modeCount; ++mode)
        q[mode] = (mode % 2u == 0u ? 1.0 : -1.0) * 1.0e-3 * (mode + 1u);
    double fineWork = 0.0, coarseWork = 0.0;
    for (std::uint32_t row = 0u; row < payload.basisNodeCount; ++row) {
        const std::array<double, 3> force{
            0.1 + 1.0e-4 * row, -0.2 + 2.0e-4 * row,
            0.3 - 1.0e-4 * row};
        std::array<double, 3> lifted{};
        for (std::uint32_t mode = 0u; mode < payload.modeCount; ++mode) {
            const auto& value = payload.basisValues[
                static_cast<std::size_t>(mode) * payload.basisNodeCount + row];
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                const double b = axis == 0u ? value.x : axis == 1u ? value.y : value.z;
                lifted[axis] += b * q[mode];
                coarseForce[mode] += b * force[axis];
            }
        }
        for (std::size_t axis = 0u; axis < 3u; ++axis)
            fineWork += force[axis] * lifted[axis];
    }
    for (std::uint32_t mode = 0u; mode < payload.modeCount; ++mode)
        coarseWork += q[mode] * coarseForce[mode];
    assert(std::abs(fineWork - coarseWork) <=
           2.0e-5 * std::max({1.0, std::abs(fineWork), std::abs(coarseWork)}));
    for (std::uint32_t a = 0u; a < payload.modeCount; ++a) {
        for (std::uint32_t b = 0u; b < payload.modeCount; ++b) {
            double projectedMass = 0.0;
            for (std::uint32_t row = 0u; row < payload.basisNodeCount; ++row) {
                const auto& va = payload.basisValues[
                    static_cast<std::size_t>(a) * payload.basisNodeCount + row];
                const auto& vb = payload.basisValues[
                    static_cast<std::size_t>(b) * payload.basisNodeCount + row];
                projectedMass += sourceMasses[row] *
                    (double(va.x) * vb.x + double(va.y) * vb.y + double(va.z) * vb.z);
            }
            const double scale = std::max({1.0, std::abs(projectedMass),
                std::abs(combined.basis.projectedMass[
                    static_cast<std::size_t>(a) * payload.modeCount + b])});
            assert(std::abs(projectedMass - combined.basis.projectedMass[
                static_cast<std::size_t>(a) * payload.modeCount + b]) <= 2.0e-5 * scale);
        }
    }
    auto opticOnlyWorld = world;
    const char* previousOpticOnlySetting =
        std::getenv("NM_FEM_HUMAN_PARTITIONED_OPTIC_ONLY");
    const std::string savedOpticOnlySetting = previousOpticOnlySetting == nullptr
        ? std::string{} : std::string(previousOpticOnlySetting);
    const bool hadPreviousOpticOnlySetting = previousOpticOnlySetting != nullptr;
    assert(::setenv("NM_FEM_HUMAN_PARTITIONED_OPTIC_ONLY", "1", 1) == 0);
    const auto opticOnly = buildAndInstallRegisteredSkinFatOpticPreconditioner(
        opticOnlyWorld, 0u, skinLabels, skin,
        std::span<const RegisteredOpticNerveReductionObject>(opticObjects),
        true, false, error);
    if (hadPreviousOpticOnlySetting)
        assert(::setenv("NM_FEM_HUMAN_PARTITIONED_OPTIC_ONLY",
                        savedOpticOnlySetting.c_str(), 1) == 0);
    else
        assert(::unsetenv("NM_FEM_HUMAN_PARTITIONED_OPTIC_ONLY") == 0);
    assert(opticOnly.valid() && error.empty());
    const auto& opticPayload = opticOnlyWorld.humanPartitionedPreconditioner;
    assert(opticPayload.enabled && opticPayload.opticOnly &&
           opticPayload.objectIndex == 0u);
    assert(opticPayload.regionCount == 10u && opticPayload.modeCount == 120u);
    assert(opticOnly.basis.regions.size() == 10u &&
           opticOnly.basis.modeCount == 120u);
    assert(opticPayload.basisNodeCount == payload.basisNodeCount);
    assert(opticPayload.globalFEMNodeIndices == payload.globalFEMNodeIndices);
    assert(opticPayload.sourceWorldPhysicsFingerprint == world.physicsFingerprint);
    assert(opticPayload.regionLabels.size() == 10u);
    for (const auto label : opticPayload.regionLabels)
        assert((label >= 0xF1772001u && label <= 0xF1772005u) ||
               (label >= 0xF1819001u && label <= 0xF1819005u));
    for (std::uint32_t row = 0u; row < opticPayload.basisNodeCount; ++row) {
        const std::uint32_t owner =
            opticOnlyWorld.fem.topologyNodes[
                opticPayload.globalFEMNodeIndices[row]].identity.y;
        if (owner == 0u) {
            for (std::uint32_t mode = 0u; mode < opticPayload.modeCount; ++mode) {
                const auto& value = opticPayload.basisValues[
                    static_cast<std::size_t>(mode) *
                        opticPayload.basisNodeCount + row];
                assert(value.x == 0.0f && value.y == 0.0f && value.z == 0.0f);
            }
        }
    }
    const auto opticSupport =
        numi::matter::buildPartitionedReductionSparseSupport(
            std::span<const nm_float4>(opticPayload.basisValues),
            opticPayload.basisNodeCount, opticPayload.modeCount,
            std::span<const std::uint32_t>(opticPayload.blockModeIndices),
            opticPayload.regionCount, opticPayload.blockModeCount);
    assert(opticSupport.valid());
    assertProjectedMassParity(opticOnly, opticOnlyWorld);
    std::array<bool, 3u> opticOwnerHasSupport{};
    for (std::uint32_t mode = 0u; mode < opticPayload.modeCount; ++mode) {
        bool modeSupported = false;
        for (std::uint32_t row = 0u; row < opticPayload.basisNodeCount; ++row) {
            const auto& value = opticPayload.basisValues[
                static_cast<std::size_t>(mode) * opticPayload.basisNodeCount + row];
            if (value.x != 0.0f || value.y != 0.0f || value.z != 0.0f) {
                const std::uint32_t owner = opticOnlyWorld.fem.topologyNodes[
                    opticPayload.globalFEMNodeIndices[row]].identity.y;
                assert(owner == 1u || owner == 2u);
                opticOwnerHasSupport[owner] = true;
                modeSupported = true;
            }
        }
        assert(modeSupported);
    }
    assert(opticOwnerHasSupport[1u] && opticOwnerHasSupport[2u]);
    assert(numi::matter::partitionedReductionSourceLayoutFingerprint(opticPayload) !=
           numi::matter::partitionedReductionSourceLayoutFingerprint(payload));

}

void runHumanOpticOnlySelectorChecks() {
    bool enabled = true;
    std::string error;
    assert(parseHumanOpticOnlyPartitionedPreconditionerSetting(
        nullptr, enabled, error) && !enabled);
    assert(parseHumanOpticOnlyPartitionedPreconditionerSetting(
        "0", enabled, error) && !enabled);
    assert(parseHumanOpticOnlyPartitionedPreconditionerSetting(
        "1", enabled, error) && enabled);
    assert(!parseHumanOpticOnlyPartitionedPreconditionerSetting(
        "true", enabled, error));
    assert(error.find("exactly 0 or 1") != std::string::npos);

    const std::array<std::uint32_t, 12u> available{{
        7u, 8u,
        0xF1772001u, 0xF1772002u, 0xF1772003u,
        0xF1772004u, 0xF1772005u,
        0xF1819001u, 0xF1819002u, 0xF1819003u,
        0xF1819004u, 0xF1819005u}};
    std::vector<std::uint32_t> selected;
    assert(selectHumanOpticOnlyPartitionedRegionLabels(
        available, 5u, selected, error));
    assert((selected == std::vector<std::uint32_t>{
        0xF1772001u, 0xF1772002u, 0xF1772003u, 0xF1772004u, 0xF1772005u,
        0xF1819001u, 0xF1819002u, 0xF1819003u, 0xF1819004u, 0xF1819005u}));

    std::array<std::uint32_t, 22u> tenBins{};
    tenBins[0] = 9u;
    for (std::uint32_t bin = 0u; bin < 10u; ++bin) {
        tenBins[1u + bin] = 0xF1772001u + bin;
        tenBins[11u + bin] = 0xF1819001u + bin;
    }
    assert(selectHumanOpticOnlyPartitionedRegionLabels(
        tenBins, 10u, selected, error));
    assert(selected.size() == 20u);
    assert(!selectHumanOpticOnlyPartitionedRegionLabels(
        available, 10u, selected, error));
    assert(!selectHumanOpticOnlyPartitionedRegionLabels(
        std::span<const std::uint32_t>(available.data(), available.size() - 1u),
        5u, selected, error));
    assert(!selectHumanOpticOnlyPartitionedRegionLabels(
        available, 6u, selected, error));
    assert(selected.empty());
    auto duplicate = available;
    duplicate[1] = duplicate[2];
    assert(!selectHumanOpticOnlyPartitionedRegionLabels(
        duplicate, 5u, selected, error));


}

void runResidualEnrichmentWitnessChecks() {
    using namespace numi::matter;
    detail::HumanPartitionedPreconditionerSource source;
    source.enabled = true;
    source.objectIndex = 3u;
    source.basisNodeCount = 4u;
    source.regionCount = 2u;
    source.blockModeCount = 2u;
    source.modeCount = 4u;
    source.freeCoordinateCount = 27u;
    source.minimumCoordinateReductionFactor = 4u;
    source.sourceWorldPhysicsFingerprint = 0x123456789abcdef0ull;
    source.regionLabels = {100u, 200u};
    source.globalFEMNodeIndices = {10u, 11u, 12u, 13u};
    source.blockModeIndices = {0u, 1u, 2u, 3u};
    source.basisValues.assign(16u, nm_float4{});
    source.basisValues[0u * 4u + 0u] = {1,0,0,0};
    source.basisValues[1u * 4u + 1u] = {1,0,0,0};
    source.basisValues[2u * 4u + 2u] = {1,0,0,0};
    source.basisValues[3u * 4u + 3u] = {1,0,0,0};

    PartitionedReductionResidualWitness witness;
    witness.sourceWorldPhysicsFingerprint = source.sourceWorldPhysicsFingerprint;
    witness.sourceLayoutFingerprint = partitionedReductionSourceLayoutFingerprint(source);
    witness.objectIndex = source.objectIndex;
    witness.basisNodeCount = source.basisNodeCount;
    witness.regionCount = source.regionCount;
    witness.blockModeCount = source.blockModeCount;
    witness.modeCount = source.modeCount;
    witness.candidates = {
        {10u,100u,0.8f,0.7f}, {11u,100u,0.7f,0.6f},
        {12u,200u,0.4f,0.3f}, {13u,200u,0.9f,0.8f}};
    std::vector<std::uint32_t> result;
    std::string error;
    const std::array<std::uint32_t,1> preservedFine{10u};
    PartitionedReductionResidualEnrichmentSelection initialSelection;
    assert(selectPartitionedResidualEnrichmentFineNodes(
        source,witness,preservedFine,4u,1u,8u,result,error,&initialSelection));
    assert((result==std::vector<std::uint32_t>{10u,11u,13u}));
    assert(initialSelection.availableFineRowBudget == 2u);

    const std::array<std::uint32_t, 4> overBudgetExisting{10u,11u,12u,13u};
    assert(!selectPartitionedResidualEnrichmentFineNodes(
        source,witness,overBudgetExisting,4u,1u,8u,result,error,&initialSelection));
    assert(error.find("existing residual fine rows exceed") != std::string::npos);
    assert(result.empty());
    const std::array<std::uint32_t, 2> duplicateExisting{10u,10u};
    assert(!selectPartitionedResidualEnrichmentFineNodes(
        source,witness,duplicateExisting,4u,1u,8u,result,error));
    assert(error.find("duplicates") != std::string::npos);
    const std::array<std::uint32_t, 1> outsideExisting{99u};
    assert(!selectPartitionedResidualEnrichmentFineNodes(
        source,witness,outsideExisting,4u,1u,8u,result,error));
    assert(error.find("outside the exact source map") != std::string::npos);

    // A second witness is accepted only against the layout rebuilt from the
    // first stage, and all previously promoted rows remain in the fine map.
    auto sequentialSource = source;
    sequentialSource.freeCoordinateCount = 36u;
    auto firstStageWitness = witness;
    firstStageWitness.sourceLayoutFingerprint =
        partitionedReductionSourceLayoutFingerprint(sequentialSource);
    std::vector<std::uint32_t> firstStageFine;
    assert(selectPartitionedResidualEnrichmentFineNodes(
        sequentialSource,firstStageWitness,preservedFine,4u,1u,8u,
        firstStageFine,error));
    assert((firstStageFine==std::vector<std::uint32_t>{10u,11u,13u}));
    auto secondStageSource = sequentialSource;
    for (const std::uint32_t global : firstStageFine) {
        const auto row = static_cast<std::size_t>(
            std::lower_bound(secondStageSource.globalFEMNodeIndices.begin(),
                             secondStageSource.globalFEMNodeIndices.end(), global) -
            secondStageSource.globalFEMNodeIndices.begin());
        for (std::uint32_t mode = 0u; mode < secondStageSource.modeCount; ++mode)
            secondStageSource.basisValues[
                std::size_t(mode) * secondStageSource.basisNodeCount + row] = {};
    }
    auto secondStageWitness = firstStageWitness;
    secondStageWitness.sourceLayoutFingerprint =
        partitionedReductionSourceLayoutFingerprint(secondStageSource);
    secondStageWitness.candidates = {{12u,200u,0.5f,0.2f}};
    assert(!selectPartitionedResidualEnrichmentFineNodes(
        secondStageSource,firstStageWitness,firstStageFine,4u,1u,8u,
        result,error));
    assert(error.find("layout") != std::string::npos);
    assert(selectPartitionedResidualEnrichmentFineNodes(
        secondStageSource,secondStageWitness,firstStageFine,4u,1u,8u,
        result,error));
    assert((result==std::vector<std::uint32_t>{10u,11u,12u,13u}));

    auto stale = witness;
    ++stale.sourceLayoutFingerprint;
    assert(!selectPartitionedResidualEnrichmentFineNodes(
        source,stale,preservedFine,4u,1u,8u,result,error));
    assert(error.find("layout")!=std::string::npos);

    auto wrongRegion = witness;
    wrongRegion.candidates[1].regionLabel = 200u;
    assert(!selectPartitionedResidualEnrichmentFineNodes(
        source,wrongRegion,preservedFine,4u,1u,8u,result,error));

    auto impossible = source;
    impossible.freeCoordinateCount = impossible.modeCount *
        impossible.minimumCoordinateReductionFactor;
    auto boundedWitness = witness;
    boundedWitness.sourceLayoutFingerprint =
        partitionedReductionSourceLayoutFingerprint(impossible);
    assert(!selectPartitionedResidualEnrichmentFineNodes(
        impossible,boundedWitness,preservedFine,4u,1u,8u,result,error));
    assert(error.find("budget") != std::string::npos);

    // The target is an explicit caller choice, not a fixed anatomical rule.
    assert(selectPartitionedResidualEnrichmentFineNodes(
        source,witness,preservedFine,3u,1u,8u,result,error));
    assert((result==std::vector<std::uint32_t>{10u,11u,13u}));
    assert(!selectPartitionedResidualEnrichmentFineNodes(
        source,witness,preservedFine,4u,2u,8u,result,error));
    assert(error.find("budget") != std::string::npos);

    // Many-region captures must honor both caps deterministically.
    detail::HumanPartitionedPreconditionerSource largeSource;
    largeSource.enabled = true;
    largeSource.objectIndex = 8u;
    largeSource.basisNodeCount = 130u;
    largeSource.regionCount = 130u;
    largeSource.blockModeCount = 1u;
    largeSource.modeCount = 130u;
    largeSource.minimumCoordinateReductionFactor = 4u;
    largeSource.freeCoordinateCount = 904u; // exactly 128 fine rows remain
    largeSource.sourceWorldPhysicsFingerprint = 0x8877665544332211ull;
    largeSource.basisValues.assign(130u * 130u, nm_float4{});
    PartitionedReductionResidualWitness manyRegions;
    manyRegions.sourceWorldPhysicsFingerprint =
        largeSource.sourceWorldPhysicsFingerprint;
    manyRegions.objectIndex = largeSource.objectIndex;
    manyRegions.basisNodeCount = largeSource.basisNodeCount;
    manyRegions.regionCount = largeSource.regionCount;
    manyRegions.blockModeCount = largeSource.blockModeCount;
    manyRegions.modeCount = largeSource.modeCount;
    for (std::uint32_t row = 0u; row < 130u; ++row) {
        const auto node = 10000u + row;
        const auto label = 2000u + row;
        largeSource.globalFEMNodeIndices.push_back(node);
        largeSource.regionLabels.push_back(label);
        largeSource.blockModeIndices.push_back(row);
        largeSource.basisValues[std::size_t(row) * 130u + row] = {1,0,0,0};
        manyRegions.candidates.push_back({node,label,float(row + 1u),0.5f});
    }
    manyRegions.sourceLayoutFingerprint =
        partitionedReductionSourceLayoutFingerprint(largeSource);
    PartitionedReductionResidualEnrichmentSelection selectedSummary;
    assert(selectPartitionedResidualEnrichmentFineNodes(
        largeSource,manyRegions,{},4u,4u,128u,result,error,&selectedSummary));
    assert(result.size() == 128u);
    assert(result.front() == 10002u && result.back() == 10129u);
    assert(selectedSummary.candidateCount == 130u &&
           selectedSummary.eligibleCandidateCount == 130u &&
           selectedSummary.perRegionCandidateCount == 130u &&
           selectedSummary.selectedCandidateCount == 128u &&
           selectedSummary.droppedByGlobalCap == 2u &&
           selectedSummary.globalCapApplied &&
           selectedSummary.availableFineRowBudget == 128u);

    // Equal residuals use global node order, not incoming trace order.
    std::reverse(manyRegions.candidates.begin(), manyRegions.candidates.end());
    for (auto& candidate : manyRegions.candidates)
        candidate.residualL2 = 1.0f;
    assert(selectPartitionedResidualEnrichmentFineNodes(
        largeSource,manyRegions,{},4u,4u,128u,result,error,&selectedSummary));
    assert(result.front() == 10000u && result.back() == 10127u);
    const auto deterministicResult = result;
    std::reverse(manyRegions.candidates.begin(), manyRegions.candidates.end());
    assert(selectPartitionedResidualEnrichmentFineNodes(
        largeSource,manyRegions,{},4u,4u,128u,result,error,&selectedSummary));
    assert(result == deterministicResult);

    // The explicit global cap still fails closed if the caller's target has
    // room for fewer fine rows than the selected 128.
    auto insufficientTarget = largeSource;
    insufficientTarget.freeCoordinateCount = 901u; // 127 available rows
    manyRegions.sourceLayoutFingerprint =
        partitionedReductionSourceLayoutFingerprint(insufficientTarget);
    assert(!selectPartitionedResidualEnrichmentFineNodes(
        insufficientTarget,manyRegions,{},4u,4u,128u,result,error,&selectedSummary));
    assert(error.find("available fine-row budget") != std::string::npos);
    assert(result.empty() && selectedSummary.selectedCandidateCount == 128u &&
           selectedSummary.availableFineRowBudget == 127u &&
           selectedSummary.globalCapApplied);

    // A final-cycle trace may carry more than 128 candidates. Allow four
    // rows per source region, while keeping the fine-row budget independent.
    detail::HumanPartitionedPreconditionerSource sampledSource;
    sampledSource.enabled = true;
    sampledSource.objectIndex = 15u;
    sampledSource.basisNodeCount = 605u;
    sampledSource.regionCount = 156u;
    sampledSource.blockModeCount = 12u;
    sampledSource.modeCount = sampledSource.regionCount * sampledSource.blockModeCount;
    sampledSource.minimumCoordinateReductionFactor = 4u;
    sampledSource.freeCoordinateCount = sampledSource.modeCount * 4u +
        sampledSource.basisNodeCount * 3u;
    sampledSource.sourceWorldPhysicsFingerprint = 0x4a9c0212871356bfull;
    sampledSource.regionLabels.resize(sampledSource.regionCount);
    sampledSource.globalFEMNodeIndices.resize(sampledSource.basisNodeCount);
    sampledSource.blockModeIndices.resize(sampledSource.modeCount);
    sampledSource.basisValues.assign(
        std::size_t(sampledSource.modeCount) * sampledSource.basisNodeCount,
        nm_float4{});
    for (std::uint32_t region = 0u; region < sampledSource.regionCount; ++region)
        sampledSource.regionLabels[region] = 0x10000u + region;
    for (std::uint32_t mode = 0u; mode < sampledSource.modeCount; ++mode)
        sampledSource.blockModeIndices[mode] = mode;
    PartitionedReductionResidualWitness sampledWitness;
    sampledWitness.sourceWorldPhysicsFingerprint =
        sampledSource.sourceWorldPhysicsFingerprint;
    sampledWitness.objectIndex = sampledSource.objectIndex;
    sampledWitness.basisNodeCount = sampledSource.basisNodeCount;
    sampledWitness.regionCount = sampledSource.regionCount;
    sampledWitness.blockModeCount = sampledSource.blockModeCount;
    sampledWitness.modeCount = sampledSource.modeCount;
    for (std::uint32_t row = 0u; row < sampledSource.basisNodeCount; ++row) {
        const std::uint32_t owner = row < 548u
            ? row / 4u : 137u + (row - 548u) / 3u;
        const std::uint32_t node = 20000u + row;
        sampledSource.globalFEMNodeIndices[row] = node;
        sampledSource.basisValues[
            std::size_t(owner * sampledSource.blockModeCount) *
                sampledSource.basisNodeCount + row] = {1,0,0,0};
        sampledWitness.candidates.push_back({
            node, sampledSource.regionLabels[owner],
            1.0f + float(row) * 0.001f, 0.5f});
    }
    sampledWitness.sourceLayoutFingerprint =
        partitionedReductionSourceLayoutFingerprint(sampledSource);
    const std::uint32_t sampledGlobalCap = sampledSource.regionCount * 4u;
    assert(sampledGlobalCap == 624u && sampledWitness.candidates.size() == 605u);
    assert(selectPartitionedResidualEnrichmentFineNodes(
        sampledSource, sampledWitness, {}, 4u, 4u, sampledGlobalCap,
        result, error, &selectedSummary));
    assert(result.size() == 605u && result.front() == 20000u &&
           result.back() == 20604u);
    assert(selectedSummary.candidateCount == 605u &&
           selectedSummary.eligibleCandidateCount == 605u &&
           selectedSummary.perRegionCandidateCount == 605u &&
           selectedSummary.selectedCandidateCount == 605u &&
           selectedSummary.droppedByGlobalCap == 0u &&
           !selectedSummary.globalCapApplied &&
           selectedSummary.globalMaximum == 624u &&
           selectedSummary.availableFineRowBudget == 605u);

    assert(selectPartitionedResidualEnrichmentFineNodes(
        sampledSource, sampledWitness, {}, 4u, 4u, 600u,
        result, error, &selectedSummary));
    assert(result.size() == 600u && selectedSummary.selectedCandidateCount == 600u &&
           selectedSummary.droppedByGlobalCap == 5u &&
           selectedSummary.globalCapApplied &&
           selectedSummary.availableFineRowBudget == 605u);

    auto sampledInsufficient = sampledSource;
    sampledInsufficient.freeCoordinateCount -= 3u;
    auto insufficientSampledWitness = sampledWitness;
    insufficientSampledWitness.sourceLayoutFingerprint =
        partitionedReductionSourceLayoutFingerprint(sampledInsufficient);
    assert(!selectPartitionedResidualEnrichmentFineNodes(
        sampledInsufficient, insufficientSampledWitness, {}, 4u, 4u,
        sampledGlobalCap, result, error, &selectedSummary));
    assert(error.find("available fine-row budget") != std::string::npos &&
           result.empty() && selectedSummary.selectedCandidateCount == 605u &&
           selectedSummary.availableFineRowBudget == 604u);

    assert(!selectPartitionedResidualEnrichmentFineNodes(
        sampledSource, sampledWitness, {}, 4u, 4u,
        kHumanPartitionedBasisMaximumRegionCount * 4u + 1u,
        result, error, &selectedSummary));
    assert(error.find("bounded selection dimensions") != std::string::npos &&
           result.empty());

    const std::string traceLine = R"json({"schema":2,"root":1,"control_step":17,"newton_iteration":8,"restart_cycle":2,"final_cycle":true,"hybrid_physical_residual_witness":{"source_physics_fingerprint":"0x123456789abcdef0","basis_layout_fingerprint_fnv1a64":"0x1234abcd","object_index":3,"basis_node_count":4,"region_count":2,"block_mode_count":2,"mode_count":4,"top_per_region_limit":4,"regions":[{"region_label":100,"nodes":[{"global_node":10,"residual_l2":0.8,"residual_max_abs":0.7}]},{"region_label":200,"nodes":[{"global_node":13,"residual_l2":0.9,"residual_max_abs":0.8}]}]}})json";
    PartitionedReductionResidualWitness parsed;
    assert(parsePartitionedReductionResidualWitnessTrace(traceLine, parsed, error));
    assert(parsed.sourceWorldPhysicsFingerprint == source.sourceWorldPhysicsFingerprint);
    assert(parsed.objectIndex == source.objectIndex && parsed.candidates.size() == 2u);
    assert(parsed.candidates[1].globalFEMNodeIndex == 13u);
    std::vector<PartitionedReductionResidualWitness> stagedWitnesses;
    assert(appendHybridResidualWitnessTraceLine(traceLine, stagedWitnesses, error));
    assert(stagedWitnesses.size() == 1u);
    assert(appendHybridResidualWitnessTraceLine(traceLine, stagedWitnesses, error));
    assert(stagedWitnesses.size() == 2u);
    auto secondStageTrace = traceLine;
    secondStageTrace.replace(
        secondStageTrace.find("\"basis_layout_fingerprint_fnv1a64\":\"0x1234abcd\""),
        std::string("\"basis_layout_fingerprint_fnv1a64\":\"0x1234abcd\"").size(),
        "\"basis_layout_fingerprint_fnv1a64\":\"0x99887766\"");
    std::vector<PartitionedReductionResidualWitness> sequentialTrace;
    assert(appendHybridResidualWitnessTraceLine(traceLine, sequentialTrace, error));
    assert(appendHybridResidualWitnessTraceLine(secondStageTrace, sequentialTrace, error));
    assert(sequentialTrace.size() == 2u &&
           sequentialTrace[0].sourceLayoutFingerprint !=
               sequentialTrace[1].sourceLayoutFingerprint);
    auto wrongOwnerTrace = traceLine;
    wrongOwnerTrace.replace(
        wrongOwnerTrace.find("\"object_index\":3"),
        std::string("\"object_index\":3").size(),
        "\"object_index\":4");
    std::vector<PartitionedReductionResidualWitness> wrongOwnerSequence;
    assert(appendHybridResidualWitnessTraceLine(traceLine, wrongOwnerSequence, error));
    assert(!appendHybridResidualWitnessTraceLine(
        wrongOwnerTrace, wrongOwnerSequence, error));
    assert(error.find("source physics or object owner") != std::string::npos);
    assert(wrongOwnerSequence.size() == 1u);
    assert(!appendHybridResidualWitnessTraceLine(traceLine, stagedWitnesses, error));
    assert(error.find("at most two sequential") != std::string::npos);
    const auto nonFinalTrace = std::string(traceLine).replace(
        traceLine.find("\"final_cycle\":true"),
        std::string("\"final_cycle\":true").size(),
        "\"final_cycle\":false");
    assert(!parsePartitionedReductionResidualWitnessTrace(nonFinalTrace, parsed, error));

    auto badMetric = witness;
    badMetric.candidates[0].residualMaxAbs = 2.0f;
    assert(!selectPartitionedResidualEnrichmentFineNodes(
        source,badMetric,preservedFine,4u,1u,8u,result,error));
}

int main() {
    runHumanOpticOnlySelectorChecks();
    static_assert(numi::matter::kHumanPartitionedBasisMaximumRegionCount == 170u);
    static_assert(numi::matter::kHumanPartitionedBasisMaximumModeCount == 2040u);
    assert(numi::matter::humanPartitionedBasisCoefficientExtentWithinBudget(
        2040u, 10425u));
    assert(!numi::matter::humanPartitionedBasisCoefficientExtentWithinBudget(
        2414u, 10425u));
    assert(!numi::matter::humanPartitionedBasisCoefficientExtentWithinBudget(
        std::numeric_limits<std::uint32_t>::max(),
        std::numeric_limits<std::uint32_t>::max()));
    RegisteredSurface surface;
    surface.worldPoints = {{{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}};
    surface.faces = {{{0u, 1u, 2u}}};
    surface.sourceVertexIndices = {101u, 202u, 303u};
    surface.sourceFaceIndices = {909u};
    surface.donorBodyIndices = {2u, 1u, 2u};

    numi::matter::ObjectSource shell;
    shell.representation = numi::matter::Representation::fem;
    shell.femNodes.resize(9u);
    shell.tetrahedra = {
        {{{0u, 1u, 2u, 5u}}},
        {{{0u, 1u, 4u, 5u}}},
        {{{0u, 3u, 4u, 5u}}},
        {{{3u, 4u, 5u, 8u}}},
        {{{3u, 4u, 7u, 8u}}},
        {{{3u, 6u, 7u, 8u}}},
    };
    const auto verifiesBinding = [&](std::uint32_t sourceVertex, std::uint32_t donorBody) {
        if (sourceVertex == 101u) return donorBody == surface.donorBodyIndices[0];
        if (sourceVertex == 202u) return donorBody == surface.donorBodyIndices[1];
        if (sourceVertex == 303u) return donorBody == surface.donorBodyIndices[2];
        return false;
    };
    const auto parentBody = [](std::uint32_t body) {
        if (body == 0u) return NM_INVALID_INDEX;
        if (body == 1u || body == 3u) return 0u;
        if (body == 2u) return 1u;
        return NM_INVALID_INDEX;
    };

    RegisteredSkinFatReductionSourceLabels labels;
    std::string error;
    assert(buildRegisteredSkinFatReductionSourceLabels(
        surface, shell, 3u, 4u, verifiesBinding, parentBody, labels, error));
    assert(labels.tetrahedronRegionLabels == std::vector<std::uint32_t>(6u, 2u));
    assert(labels.tetrahedronSourceFaceIndices == std::vector<std::uint32_t>(6u, 909u));
    assert(labels.tetrahedronSourceParentBodyIndices == std::vector<std::uint32_t>(6u, 1u));
    assert(labels.donorBodyAncestry.at(2u) == std::vector<std::uint32_t>({2u, 1u, 0u}));
    assert(labels.registeredVertexCount == 3u && labels.registeredFaceCount == 1u);
    assert(labels.sourceTetrahedronCount == 6u);
    assert(labels.sourceNodeVertexIndices ==
           std::vector<std::uint32_t>({101u, 202u, 303u,
                                       101u, 202u, 303u,
                                       101u, 202u, 303u}));
    assert(labels.sourceNodeLayerIndices ==
           std::vector<std::uint32_t>({0u, 0u, 0u, 1u, 1u, 1u,
                                       2u, 2u, 2u}));

    surface.donorBodyIndices = {3u, 1u, 2u};
    assert(buildRegisteredSkinFatReductionSourceLabels(
        surface, shell, 3u, 4u, verifiesBinding, parentBody, labels, error));
    assert(labels.tetrahedronRegionLabels == std::vector<std::uint32_t>(6u, 1u));
    assert(labels.donorBodyAncestry.at(1u) == std::vector<std::uint32_t>({1u, 0u}));

    const auto cycleParent = [](std::uint32_t body) {
        if (body == 1u) return 2u;
        if (body == 2u) return 1u;
        return NM_INVALID_INDEX;
    };
    assert(!buildRegisteredSkinFatReductionSourceLabels(
        surface, shell, 3u, 4u, verifiesBinding, cycleParent, labels, error));
    assert(error.find("cycle") != std::string::npos);

    shell.tetrahedra.pop_back();
    assert(!buildRegisteredSkinFatReductionSourceLabels(
        surface, shell, 3u, 4u, verifiesBinding, parentBody, labels, error));
    assert(error.find("three-layer prism mesh") != std::string::npos);

    using numi::human::optic_nerve::tetgen_assets::kLeftAsset;
    using numi::human::optic_nerve::tetgen_assets::kRightAsset;
    for (const auto* asset : {&kLeftAsset, &kRightAsset}) {
        numi::matter::ObjectSource nerve;
        nerve.representation = numi::matter::Representation::fem;
        nerve.femNodes.resize(asset->donorLocalPoints.size());
        nerve.tetrahedra.reserve(asset->tetrahedra.size());
        for (const auto& tet : asset->tetrahedra)
            nerve.tetrahedra.push_back({tet});
        std::set<std::uint32_t> boundary;
        for (const auto& face : asset->boundaryFaces)
            boundary.insert(face.begin(), face.end());
        nerve.femContactNodes.assign(boundary.begin(), boundary.end());
        RegisteredOpticNerveReductionSourceLabels optic;
        assert(buildRegisteredOpticNerveReductionSourceLabels(
            *asset, nerve, optic, error));
        assert(optic.sourceNodeVertexIndices.size() ==
               asset->donorLocalPoints.size());
        assert(optic.tetrahedronRegionLabels.size() ==
               asset->tetrahedra.size());
        assert(optic.tetrahedronSourceIndices.size() ==
               asset->tetrahedra.size());
        assert(optic.boundarySourceFaceIndices ==
               std::vector<std::uint32_t>(
                   asset->boundarySourceFace.begin(),
                   asset->boundarySourceFace.end()));
        assert(optic.boundarySourceNodeIndices ==
               std::vector<std::uint32_t>(boundary.begin(), boundary.end()));
        assert(optic.member == asset->member);
        assert(optic.meshSHA256 == asset->meshSHA256);
        std::map<std::uint32_t, std::size_t> perRegion;
        for (std::size_t tet = 0u; tet < asset->tetrahedra.size(); ++tet) {
            assert(optic.tetrahedronSourceIndices[tet] == tet);
            ++perRegion[optic.tetrahedronRegionLabels[tet]];
        }
        assert(perRegion.size() == 5u);
        for (const auto& [region, count] : perRegion) {
            assert(region != NM_INVALID_INDEX && count > 0u);
        }
        RegisteredOpticNerveReductionSourceLabels tenBinLabels;
        assert(buildRegisteredOpticNerveReductionSourceLabels(
            *asset, nerve, tenBinLabels, error, 10u));
        std::map<std::uint32_t, std::size_t> tenBinCounts;
        for (const auto label : tenBinLabels.tetrahedronRegionLabels)
            ++tenBinCounts[label];
        assert(tenBinCounts.size() == 10u);
        const std::uint32_t tenBinBase = asset->member == "FJ1772"
            ? 0xF1772001u : 0xF1819001u;
        for (std::uint32_t bin = 0u; bin < 10u; ++bin)
            assert(tenBinCounts.contains(tenBinBase + bin));
        assert(tenBinLabels.tetrahedronSourceIndices.size() ==
               nerve.tetrahedra.size());
        for (std::uint32_t tet = 0u;
             tet < tenBinLabels.tetrahedronSourceIndices.size(); ++tet)
            assert(tenBinLabels.tetrahedronSourceIndices[tet] == tet);
        assert(tenBinLabels.boundarySourceFaceIndices ==
               optic.boundarySourceFaceIndices);
        assert(tenBinLabels.boundarySourceNodeIndices ==
               optic.boundarySourceNodeIndices);
        assert(tenBinLabels.witnessSourceNodeIndices ==
               optic.witnessSourceNodeIndices);
    }
    assert(kLeftAsset.member != kRightAsset.member);
    assert(kLeftAsset.meshSHA256 != kRightAsset.meshSHA256);

    runCombinedOpticConsumerTest();
    runResidualEnrichmentWitnessChecks();
    std::cout << "registered_skin_fat_reduction_labels_check=pass\n";
    return 0;
}
