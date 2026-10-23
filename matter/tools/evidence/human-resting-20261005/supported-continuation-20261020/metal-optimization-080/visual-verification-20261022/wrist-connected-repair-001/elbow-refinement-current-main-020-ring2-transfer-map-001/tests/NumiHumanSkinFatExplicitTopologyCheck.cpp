#include "NumiHumanSkinFatReductionLabels.hpp"
#include "NumiHumanWholeBodySkinTransferAncestry.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

int main() {
    constexpr std::uint32_t parentWidth = 2332u;
    constexpr std::uint32_t addedWidth = 79u;
    constexpr std::uint32_t layerWidth = parentWidth + addedWidth;
    constexpr std::uint32_t parentFaceCount = 4619u;
    constexpr std::uint32_t surfaceFaceCount = 4777u;
    constexpr std::uint32_t parentTetCount = 27714u;
    constexpr std::uint32_t childTetCount = 28632u;
    constexpr std::uint32_t skinMaterial = 1u;
    constexpr std::uint32_t fatMaterial = 2u;

    RegisteredSurface surface;
    surface.worldPoints.resize(layerWidth);
    surface.sourceVertexIndices.resize(layerWidth);
    surface.donorBodyIndices.assign(layerWidth, 7u);
    surface.sourceEmbeddingFaceIndices.assign(layerWidth, NM_INVALID_INDEX);
    surface.sourceEmbeddingBarycentric.assign(layerWidth, {{1.0, 0.0, 0.0}});
    surface.sourceFaceIndices.reserve(surfaceFaceCount);
    surface.faces.reserve(surfaceFaceCount);
    for (std::uint32_t v = 0u; v < parentWidth; ++v) {
        const double angle = 6.2831853071795864769 * static_cast<double>(v) / parentWidth;
        surface.worldPoints[v] = {{std::cos(angle), std::sin(angle), 0.0}};
        surface.sourceVertexIndices[v] = 100000u + v;
    }
    for (std::uint32_t v = 0u; v < addedWidth; ++v) {
        surface.worldPoints[parentWidth + v] = {{0.0, 0.0, 0.0}};
        surface.sourceVertexIndices[parentWidth + v] = NM_INVALID_INDEX;
        surface.sourceEmbeddingFaceIndices[parentWidth + v] = 0u;
        surface.sourceEmbeddingBarycentric[parentWidth + v] = {{0.5, 0.5, 0.0}};
    }
    // Unique parent triangles: two deterministic local triples per start ID,
    // then one final triple. They stand in for the accepted coarse boundary.
    for (std::uint32_t v = 0u; v <= 2308u; ++v) {
        surface.faces.push_back({v, v + 1u, v + 2u});
        surface.faces.push_back({v, v + 1u, v + 3u});
    }
    surface.faces.push_back({2309u, 2310u, 2311u});
    assert(surface.faces.size() == parentFaceCount);
    std::vector<std::uint32_t> appendedFaceParent;
    appendedFaceParent.reserve(addedWidth * 2u);
    for (std::uint32_t v = 0u; v < addedWidth; ++v) {
        const std::uint32_t m0 = parentWidth + v;
        const std::uint32_t m1 = parentWidth + ((v + 1u) % addedWidth);
        const std::uint32_t p0 = (v * 2u) % parentFaceCount;
        const std::uint32_t p1 = (v * 2u + 1u) % parentFaceCount;
        surface.faces.push_back({0u, m0, m1});
        appendedFaceParent.push_back(p0);
        surface.faces.push_back({1u, m0, m1});
        appendedFaceParent.push_back(p1);
    }
    assert(surface.faces.size() == surfaceFaceCount);
    for (std::uint32_t face = 0u; face < parentFaceCount; ++face)
        surface.sourceFaceIndices.push_back(1000000u + face);
    for (const auto parentFace : appendedFaceParent)
        surface.sourceFaceIndices.push_back(1000000u + parentFace);
    static_assert(kWholeBodySkinMechanicalFaceIdentities.size() == surfaceFaceCount);
    for (std::uint32_t face = 0u; face < surfaceFaceCount; ++face)
        assert(kWholeBodySkinMechanicalFaceIdentities[face] == face);

    numi::matter::ObjectSource shell;
    shell.name = "explicit_parent_child_label_fixture";
    shell.representation = numi::matter::Representation::fem;
    shell.femNodes.resize(3u * layerWidth);
    for (std::uint32_t layer = 0u; layer < 3u; ++layer) {
        for (std::uint32_t v = 0u; v < layerWidth; ++v) {
            const auto p = surface.worldPoints[v];
            shell.femNodes[layer * layerWidth + v] = {p[0], p[1], p[2] + 0.002 * layer};
        }
    }
    const auto parentNodeToChild = [&](const std::uint32_t node) {
        const std::uint32_t layer = node / parentWidth;
        const std::uint32_t vertex = node % parentWidth;
        return layer * layerWidth + vertex;
    };
    std::vector<V3> parentWorldPoints;
    std::vector<std::array<std::uint32_t, 4u>> parentTetrahedra;
    std::vector<std::uint32_t> parentMaterialIndices;
    std::vector<std::array<std::uint32_t, 4u>> childTetrahedra;
    std::vector<std::uint32_t> childMaterialIndices;
    std::vector<std::uint32_t> childParentTetrahedra;
    std::vector<std::uint32_t> childParentSurfaceFaces;
    std::vector<std::uint32_t> childSourceFaceIdentities;
    parentWorldPoints.reserve(3u * parentWidth);
    parentTetrahedra.reserve(parentTetCount);
    parentMaterialIndices.reserve(parentTetCount);
    const auto appendParentTet = [&](std::uint32_t a, std::uint32_t b,
                                     std::uint32_t c, std::uint32_t d,
                                     std::uint32_t material) {
        parentTetrahedra.push_back({a, b, c, d});
        parentMaterialIndices.push_back(material);
    };
    for (std::uint32_t layer = 0u; layer < 3u; ++layer)
        for (std::uint32_t v = 0u; v < parentWidth; ++v) {
            auto p = surface.worldPoints[v];
            p[2] = 0.002 * layer;
            parentWorldPoints.push_back(p);
        }
    for (const auto& f : std::span<const Face>(surface.faces.data(), parentFaceCount)) {
        const auto [a, b, c] = f;
        appendParentTet(a, b, c, parentWidth + c, skinMaterial);
        appendParentTet(a, b, parentWidth + b, parentWidth + c, skinMaterial);
        appendParentTet(a, parentWidth + a, parentWidth + b, parentWidth + c, skinMaterial);
        appendParentTet(parentWidth + a, parentWidth + b, parentWidth + c, 2u * parentWidth + c, fatMaterial);
        appendParentTet(parentWidth + a, parentWidth + b, 2u * parentWidth + b, 2u * parentWidth + c, fatMaterial);
        appendParentTet(parentWidth + a, 2u * parentWidth + a, 2u * parentWidth + b, 2u * parentWidth + c, fatMaterial);
    }
    assert(parentTetrahedra.size() == parentTetCount);

    childTetrahedra.reserve(childTetCount);
    childMaterialIndices.reserve(childTetCount);
    childParentTetrahedra.reserve(childTetCount);
    childParentSurfaceFaces.reserve(childTetCount);
    childSourceFaceIdentities.reserve(childTetCount);
    shell.tetrahedra.reserve(childTetCount);
    shell.femMaterialIndices.reserve(childTetCount);
    const auto appendChild = [&](const std::uint32_t parent,
                                 const std::uint32_t surfaceFace) {
        auto child = parentTetrahedra[parent];
        for (auto& node : child) node = parentNodeToChild(node);
        const auto material = parentMaterialIndices[parent];
        childTetrahedra.push_back(child);
        childMaterialIndices.push_back(material);
        childParentTetrahedra.push_back(parent);
        childParentSurfaceFaces.push_back(surfaceFace);
        childSourceFaceIdentities.push_back(surface.sourceFaceIndices[surfaceFace]);
        shell.tetrahedra.push_back({child});
        shell.femMaterialIndices.push_back(material);
    };
    for (std::uint32_t parent = 0u; parent < parentTetCount; ++parent)
        appendChild(parent, parent / 6u);
    for (std::uint32_t extra = 0u; extra < childTetCount - parentTetCount; ++extra) {
        const std::uint32_t faceOffset = extra % static_cast<std::uint32_t>(appendedFaceParent.size());
        const std::uint32_t surfaceFace = parentFaceCount + faceOffset;
        const std::uint32_t parentFace = appendedFaceParent[faceOffset];
        const std::uint32_t parent = parentFace * 6u + (extra % 6u);
        appendChild(parent, surfaceFace);
    }
    assert(shell.tetrahedra.size() == childTetCount);
    assert(childTetrahedra.size() == childTetCount);

    const ExplicitPairedLayerTopology topology{
        std::span<const std::array<std::uint32_t, 4u>>(childTetrahedra),
        std::span<const std::uint32_t>(childMaterialIndices),
        std::span<const std::uint32_t>(childParentTetrahedra),
        std::span<const std::array<std::uint32_t, 4u>>(parentTetrahedra),
        std::span<const V3>(parentWorldPoints),
        std::span<const std::uint32_t>(parentMaterialIndices),
        std::span<const std::uint32_t>(childParentSurfaceFaces),
        std::span<const std::uint32_t>(childSourceFaceIdentities)};

    const auto verifyDirect = [](std::uint32_t source, std::uint32_t donor) {
        return source >= 100000u && source < 100000u + parentWidth && donor == 7u;
    };
    const auto verifyProjected = [](std::uint32_t row, std::uint32_t donor) {
        return row >= parentWidth && row < layerWidth && donor == 7u;
    };
    const auto parentBody = [](std::uint32_t donor) {
        return donor == 7u ? 0u : NM_INVALID_INDEX;
    };
    RegisteredSkinFatReductionSourceLabels labels;
    std::string error;
    const std::span<const std::uint32_t> noSplitTets{};
    assert(buildRegisteredSkinFatReductionSourceLabels(
        surface, shell, layerWidth, 8u, verifyDirect, parentBody, labels, error,
        36u, noSplitTets, 0u, verifyProjected,
        std::span<const std::uint32_t>(kWholeBodySkinMechanicalFaceIdentities), &topology,
        skinMaterial, fatMaterial));
    assert(labels.registeredVertexCount == layerWidth);
    assert(labels.registeredFaceCount == surfaceFaceCount);
    assert(labels.sourceTetrahedronCount == childTetCount);
    assert(labels.tetrahedronRegionLabels.size() == childTetCount);
    assert(labels.tetrahedronBasisRegionLabels.size() == childTetCount);
    assert(labels.tetrahedronSourceFaceIndices.size() == childTetCount);
    assert(labels.tetrahedronOriginalSourceFaceIndices.size() == childTetCount);
    assert(std::equal(labels.tetrahedronParentTetrahedronIndices.begin(),
                      labels.tetrahedronParentTetrahedronIndices.end(),
                      topology.childParentTetrahedra.begin(),
                      topology.childParentTetrahedra.end()));
    for (std::uint32_t tet = 0u; tet < childTetCount; ++tet) {
        const auto parent = topology.childParentTetrahedra[tet];
        assert(labels.tetrahedronRegionLabels[tet] == 7u);
        assert(labels.tetrahedronBasisRegionLabels[tet] != NM_INVALID_INDEX);
        assert(labels.tetrahedronSourceFaceIndices[tet] ==
               kWholeBodySkinMechanicalFaceIdentities[topology.childParentSurfaceFaces[tet]]);
        assert(labels.tetrahedronOriginalSourceFaceIndices[tet] ==
               topology.childSourceFaceIdentities[tet]);
        assert(labels.tetrahedronBasisRegionLabels[tet] ==
               labels.tetrahedronBasisRegionLabels[parent]);
    }
    assert(labels.donorBodyAncestry.at(7u) == std::vector<std::uint32_t>({7u, 0u}));

    // Independently run the legacy six-prism-per-parent-face label path on
    // the original 27,714-tet / 4,619-face parent, then compare every explicit
    // child against the legacy label of its declared parent tetrahedron.
    RegisteredSurface parentSurface;
    parentSurface.worldPoints.assign(surface.worldPoints.begin(),
                                     surface.worldPoints.begin() + parentWidth);
    parentSurface.sourceVertexIndices.assign(surface.sourceVertexIndices.begin(),
                                             surface.sourceVertexIndices.begin() + parentWidth);
    parentSurface.donorBodyIndices.assign(surface.donorBodyIndices.begin(),
                                          surface.donorBodyIndices.begin() + parentWidth);
    parentSurface.sourceEmbeddingFaceIndices.assign(
        surface.sourceEmbeddingFaceIndices.begin(),
        surface.sourceEmbeddingFaceIndices.begin() + parentWidth);
    parentSurface.sourceEmbeddingBarycentric.assign(
        surface.sourceEmbeddingBarycentric.begin(),
        surface.sourceEmbeddingBarycentric.begin() + parentWidth);
    parentSurface.faces.assign(surface.faces.begin(), surface.faces.begin() + parentFaceCount);
    parentSurface.sourceFaceIndices.assign(surface.sourceFaceIndices.begin(),
                                           surface.sourceFaceIndices.begin() + parentFaceCount);
    numi::matter::ObjectSource parentShell;
    parentShell.name = "legacy_parent_label_fixture";
    parentShell.representation = numi::matter::Representation::fem;
    parentShell.femNodes.assign(shell.femNodes.begin(), shell.femNodes.begin() + 3u * parentWidth);
    parentShell.tetrahedra.reserve(parentTetCount);
    parentShell.femMaterialIndices = parentMaterialIndices;
    for (const auto& tet : parentTetrahedra) parentShell.tetrahedra.push_back({tet});
    RegisteredSkinFatReductionSourceLabels parentLabels;
    assert(buildRegisteredSkinFatReductionSourceLabels(
        parentSurface, parentShell, parentWidth, 8u, verifyDirect, parentBody,
        parentLabels, error, 36u, noSplitTets));
    assert(parentLabels.registeredFaceCount == parentFaceCount);
    assert(parentLabels.sourceTetrahedronCount == parentTetCount);
    for (std::uint32_t tet = 0u; tet < childTetCount; ++tet) {
        const auto parent = topology.childParentTetrahedra[tet];
        assert(labels.tetrahedronRegionLabels[tet] ==
               parentLabels.tetrahedronRegionLabels[parent]);
        assert(labels.tetrahedronBasisRegionLabels[tet] ==
               parentLabels.tetrahedronBasisRegionLabels[parent]);
        assert(labels.tetrahedronOriginalSourceFaceIndices[tet] ==
               parentLabels.tetrahedronOriginalSourceFaceIndices[parent]);
    }

    auto badChildParents = childParentTetrahedra;
    badChildParents[0] = parentTetCount;
    auto malformed = topology;
    malformed.childParentTetrahedra = std::span<const std::uint32_t>(badChildParents);
    RegisteredSkinFatReductionSourceLabels rejected;
    assert(!buildRegisteredSkinFatReductionSourceLabels(
        surface, shell, layerWidth, 8u, verifyDirect, parentBody, rejected, error,
        36u, noSplitTets, 0u, verifyProjected,
        std::span<const std::uint32_t>(kWholeBodySkinMechanicalFaceIdentities), &malformed,
        skinMaterial, fatMaterial));
    assert(error.find("explicit child") != std::string::npos);

    auto badChildFaceIds = childSourceFaceIdentities;
    badChildFaceIds[1] += 1u;
    malformed = topology;
    malformed.childSourceFaceIdentities = std::span<const std::uint32_t>(badChildFaceIds);
    assert(!buildRegisteredSkinFatReductionSourceLabels(
        surface, shell, layerWidth, 8u, verifyDirect, parentBody, rejected, error,
        36u, noSplitTets, 0u, verifyProjected,
        std::span<const std::uint32_t>(kWholeBodySkinMechanicalFaceIdentities), &malformed,
        skinMaterial, fatMaterial));
    assert(error.find("source-face identity") != std::string::npos);
    std::cout << "explicit_parent_child_skin_fat_labels=pass legacy_parent_label_parity=pass parents=" << parentTetCount
              << " children=" << childTetCount << " faces=" << surfaceFaceCount << "\n";
}
