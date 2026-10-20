#include "../src/regional_schur_condensation.hpp"

#include <algorithm>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

using numi::matter::detail::RegionalGraphSchwarzLayout;
using numi::matter::detail::RegionalGraphSchwarzPatch;
using numi::matter::detail::buildRegionalSchurGraphLayout;

namespace {
bool require(const bool value, const char* message, std::uint32_t& checks) {
    ++checks;
    if (!value) std::cerr << "regional_schur_condensation_check: " << message << '\n';
    return value;
}
} // namespace

int main() {
    std::uint32_t checks=0u;
    bool ok=true;
    const std::array<std::array<std::uint32_t,4>,2> tets{{
        {{0u,1u,2u,3u}}, {{2u,3u,4u,5u}}
    }};
    const std::array<std::array<std::uint32_t,2>,2> sources{{
        {{0u,10u}}, {{0u,20u}}
    }};
    const std::array<std::uint32_t,1> externalAnchors{{5u}};
    RegionalGraphSchwarzLayout graph;
    graph.nodeCount=6u;
    graph.environmentCount=1u;
    graph.patchNodes={0u,1u,2u,3u,4u,5u};
    graph.patches.push_back(RegionalGraphSchwarzPatch{0u,6u,0u,0u});
    const auto layout=buildRegionalSchurGraphLayout(
        graph,tets,sources,externalAnchors);
    ok &= require(layout.valid(),"valid exact source labels were rejected",checks);
    ok &= require(layout.patches.size()==1u && layout.patches[0].eligible,
        "bounded Human-style patch did not admit a nonempty interior/interface",checks);
    ok &= require(layout.patches[0].interiorCount==3u &&
                  layout.patches[0].interfaceCount==3u,
        "cross-region and external endpoints were not separated",checks);
    const std::vector<std::uint32_t> expectedInterior{0u,1u,4u};
    const std::vector<std::uint32_t> expectedInterface{2u,3u,5u};
    for (std::size_t i=0u; i<expectedInterior.size(); ++i)
        ok &= require(graph.patchNodes[layout.interiorPatchValues[i]]==expectedInterior[i],
            "interior ordering is not stable in graph patch order",checks);
    for (std::size_t i=0u; i<expectedInterface.size(); ++i)
        ok &= require(graph.patchNodes[layout.interfacePatchValues[i]]==expectedInterface[i],
            "interface ordering is not stable in graph patch order",checks);

    // A smaller patch has to push nodes with an out-of-patch tetrahedral
    // neighbor onto its interface. The remaining interior's A row is then
    // fully present in the already-built graph patch principal matrix.
    RegionalGraphSchwarzLayout bounded=graph;
    bounded.patchNodes={0u,1u,2u,3u,4u,5u};
    bounded.patches={{0u,4u,0u,0u},{4u,2u,48u,48u}};
    const auto boundedLayout=buildRegionalSchurGraphLayout(
        bounded,tets,sources,externalAnchors);
    ok &= require(boundedLayout.valid() && boundedLayout.patches[0].eligible &&
                  boundedLayout.patches[0].interiorCount==2u &&
                  boundedLayout.patches[0].interfaceCount==2u,
        "out-of-patch FEM neighbors were not retained as Schur interface",checks);

    // Existing graph Schwarz patches overlap. The Schur metadata assigns
    // each node a deterministic first owner and moves every cross-owner
    // tetrahedron onto the interface, so two patches never eliminate the
    // same node or omit an element edge.
    RegionalGraphSchwarzLayout overlapping;
    overlapping.nodeCount=6u;
    overlapping.environmentCount=1u;
    overlapping.patchNodes={0u,1u,2u,3u,2u,3u,4u,5u};
    overlapping.patches={{0u,4u,0u,0u},{4u,4u,48u,48u}};
    const auto overlapLayout=buildRegionalSchurGraphLayout(
        overlapping,tets,sources,externalAnchors);
    ok &= require(overlapLayout.valid() && overlapLayout.patches.size()==2u &&
                  overlapLayout.patches[0].interiorCount==2u &&
                  overlapLayout.patches[1].interiorCount==0u,
        "overlapping graph patches did not produce disjoint interiors",checks);

    const std::array<std::array<std::uint32_t,2>,2> missingSource{{
        {{0u,10u}}, {{UINT32_MAX,20u}}
    }};
    ok &= require(!buildRegionalSchurGraphLayout(
        graph,tets,missingSource,externalAnchors).valid(),
        "invalid compiled source identity did not fail closed",checks);
    const std::array<std::array<std::uint32_t,2>,2> sameRegionOtherObject{{
        {{0u,10u}}, {{1u,10u}}
    }};
    const auto objectBoundary = buildRegionalSchurGraphLayout(
        graph,tets,sameRegionOtherObject,externalAnchors);
    ok &= require(objectBoundary.valid() &&
                  graph.patchNodes[objectBoundary.interfacePatchValues[0]]==2u &&
                  graph.patchNodes[objectBoundary.interfacePatchValues[1]]==3u,
        "shared nodes across same-tag objects were not retained on interface",checks);
    // Every node of a tetrahedron adjacent to a compiled dynamic-contact
    // surface is an interface node, including vertices that are not current
    // pair endpoints. A remote interior remains eligible.
    const std::array<std::array<std::uint32_t,4>,3> surfaceTets{{
        {{0u,1u,2u,3u}}, {{1u,2u,3u,4u}}, {{2u,3u,4u,5u}}
    }};
    const std::array<std::array<std::uint32_t,2>,1> dynamicSurface{{{{0u,1u}}}};
    std::vector<std::uint32_t> surfaceAnchors;
    ok &= require(numi::matter::detail::appendRegionalSchurSurfaceTetAnchors(
            surfaceTets, dynamicSurface, surfaceAnchors),
        "valid dynamic surface adjacency failed to produce interface anchors",checks);
    std::sort(surfaceAnchors.begin(), surfaceAnchors.end());
    surfaceAnchors.erase(std::unique(surfaceAnchors.begin(), surfaceAnchors.end()),
                         surfaceAnchors.end());
    RegionalGraphSchwarzLayout surfaceGraph;
    surfaceGraph.nodeCount=6u;
    surfaceGraph.environmentCount=1u;
    surfaceGraph.patchNodes={0u,1u,2u,3u,4u,5u};
    surfaceGraph.patches={{0u,6u,0u,0u}};
    const std::array<std::array<std::uint32_t,2>,3> surfaceSources{{
        {{0u,10u}}, {{0u,10u}}, {{0u,10u}}
    }};
    const auto surfaceLayout=buildRegionalSchurGraphLayout(
        surfaceGraph,surfaceTets,surfaceSources,surfaceAnchors);
    ok &= require(surfaceLayout.valid() &&
                  surfaceLayout.patches[0].interiorCount==1u &&
                  surfaceLayout.patches[0].interfaceCount==5u &&
                  surfaceGraph.patchNodes[surfaceLayout.interiorPatchValues[0]]==5u,
        "dynamic surface closure did not anchor both adjacent tets",checks);
    const std::array<std::array<std::uint32_t,2>,1> invalidSurface{{{{3u,UINT32_MAX}}}};
    ok &= require(!numi::matter::detail::appendRegionalSchurSurfaceTetAnchors(
            surfaceTets, invalidSurface, surfaceAnchors),
        "out-of-range dynamic surface adjacency was not rejected",checks);
    // A runtime implicit spring endpoint must also be supplied as an external
    // anchor; if it consumes the last interior, exact condensation is disabled.
    surfaceAnchors.push_back(5u);
    ok &= require(!buildRegionalSchurGraphLayout(
            surfaceGraph,surfaceTets,surfaceSources,surfaceAnchors).valid(),
        "spring endpoint that covers the last interior did not fail closed",checks);

    const std::array<std::uint32_t,6> allAnchors{{0u,1u,2u,3u,4u,5u}};
    ok &= require(!buildRegionalSchurGraphLayout(
        graph,tets,sources,allAnchors).valid(),
        "partition without an eliminable interior was admitted",checks);
    const std::array<std::uint32_t,1> foreignAnchor{{6u}};
    ok &= require(!buildRegionalSchurGraphLayout(
        graph,tets,sources,foreignAnchor).valid(),
        "foreign non-FEM coupling anchor did not fail closed",checks);
    const std::array<std::array<std::uint32_t,4>,2> orphanTets{{
        {{0u,1u,2u,3u}}, {{0u,1u,2u,3u}}
    }};
    ok &= require(!buildRegionalSchurGraphLayout(
        graph,orphanTets,sources,externalAnchors).valid(),
        "unreferenced unknown could not be left outside the partition",checks);

    std::cout << "regional_schur_partition_checks=" << checks
              << " interior=3 interface=3 result=" << (ok ? "PASS" : "FAIL") << '\n';
    return ok ? 0 : 1;
}
