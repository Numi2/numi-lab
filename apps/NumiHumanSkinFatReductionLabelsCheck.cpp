#include "NumiHumanSkinFatReductionLabels.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <span>
#include <cstdint>
#include <iostream>
#include <string>
#include <set>
#include <vector>

using namespace metalrobo::human::coarse_tissue;


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
        21u, verifyRefinedBinding, refinedParent, refinedLabels, error));
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
    auto wrongSourceOrder = sourceTetOrder;
    std::swap(wrongSourceOrder[0], wrongSourceOrder[1]);
    assert(!mapRegisteredSkinFatBasisRegionsToDonors(
        refinedLabels, refinedLabels.tetrahedronBasisRegionLabels,
        wrongSourceOrder, skin.tetrahedra.size(), refinementAncestry, error));
    assert(error.find("source-tet order") != std::string::npos);

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
            node.restAndFixed = {static_cast<float>(p[0]), static_cast<float>(p[1]),
                                 static_cast<float>(p[2]), 0.0f};
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
}

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
    }
    assert(kLeftAsset.member != kRightAsset.member);
    assert(kLeftAsset.meshSHA256 != kRightAsset.meshSHA256);

    runCombinedOpticConsumerTest();
    std::cout << "registered_skin_fat_reduction_labels_check=pass\n";
    return 0;
}
