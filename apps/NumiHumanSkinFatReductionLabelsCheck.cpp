#include "NumiHumanSkinFatReductionLabels.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

int main() {
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

    std::cout << "registered_skin_fat_reduction_labels_check=pass\n";
    return 0;
}
