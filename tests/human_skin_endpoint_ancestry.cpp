#include "NumiHumanSkinFatReductionLabels.hpp"
#include <cassert>
#include <iostream>
using namespace metalrobo::human::coarse_tissue;
using numi::matter::ObjectSource;
using numi::matter::Representation;
int main() {
    RegisteredSurface surface;
    constexpr std::uint32_t side = 12u;
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

    RegisteredSkinFatReductionSourceLabels labels;
    std::string error;
    const auto build=[&](bool spatial) {
        return buildRegisteredSkinFatReductionSourceLabels(
            surface,skin,layerWidth,8u,verifyBinding,parentBody,labels,error,1u,
            std::span<const std::uint32_t>{},0u,{}, {},nullptr,
            NM_INVALID_INDEX,NM_INVALID_INDEX,spatial);
    };
    assert(!build(true));
    assert(error.find("compiled Human basis region budget")!=std::string::npos);
    assert(build(false));
    assert(labels.sourceNodeVertexIndices.size()==skin.femNodes.size());
    assert(labels.tetrahedronSourceFaceIndices.size()==skin.tetrahedra.size());
    assert(labels.tetrahedronRegionLabels==labels.tetrahedronBasisRegionLabels);
    assert(labels.donorBodyAncestry.at(7u)==(std::vector<std::uint32_t>{7u,0u}));
    for(auto x:labels.tetrahedronSourceDonorBodyIndices)assert(x==7u);
    std::vector<std::array<std::uint32_t,4>> tets;
    std::vector<std::uint32_t> materials,parents,parentFaces,stableFaces,mechanicalFaces;
    for(std::uint32_t i=0;i<skin.tetrahedra.size();++i) {
        tets.push_back(skin.tetrahedra[i].nodes);
        materials.push_back(i%6u<3u?0u:1u);
        parents.push_back(i);parentFaces.push_back(i/6u);
        stableFaces.push_back(surface.sourceFaceIndices[i/6u]);
    }
    for(std::uint32_t i=0;i<surface.faces.size();++i)mechanicalFaces.push_back(i);
    skin.femMaterialIndices=materials;
    ExplicitPairedLayerTopology topology{tets,materials,parents,tets,skin.femNodes,
                                         materials,parentFaces,stableFaces};
    assert(buildRegisteredSkinFatReductionSourceLabels(
        surface,skin,layerWidth,8u,verifyBinding,parentBody,labels,error,1u,
        std::span<const std::uint32_t>{},0u,{},mechanicalFaces,&topology,0u,1u,false));
    assert(labels.tetrahedronBasisRegionLabels==labels.tetrahedronRegionLabels);
    assert(labels.tetrahedronParentTetrahedronIndices==parents);
    auto oldLabels=labels.tetrahedronSourceFaceIndices;
    parents[0]=static_cast<std::uint32_t>(parents.size());
    assert(!buildRegisteredSkinFatReductionSourceLabels(
        surface,skin,layerWidth,8u,verifyBinding,parentBody,labels,error,1u,
        std::span<const std::uint32_t>{},0u,{},mechanicalFaces,&topology,0u,1u,false));
    assert(error.find("parent")!=std::string::npos);
    assert(labels.tetrahedronSourceFaceIndices==oldLabels);
    parents[0]=0u;
    assert(build(false));
    auto expected=labels.tetrahedronSourceFaceIndices;
    skin.tetrahedra[0].nodes[0]=static_cast<std::uint32_t>(skin.femNodes.size());
    assert(!build(false));
    assert(error.find("outside its authored object")!=std::string::npos);
    assert(labels.tetrahedronSourceFaceIndices==expected);
    std::cout<<"endpoint_ancestry=PASS unused_basis=NOT_BUILT default_capacity_gate=UNCHANGED invalid_topology=REJECTED transactionally\n";
}
