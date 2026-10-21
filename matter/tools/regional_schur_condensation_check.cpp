#include "../src/regional_schur_condensation.hpp"

#include <algorithm>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

using numi::matter::detail::RegionalGraphSchwarzLayout;
using numi::matter::detail::RegionalGraphSchwarzPatch;
using numi::matter::detail::RegionalSchurPatchRange;
using numi::matter::detail::buildRegionalSchurGraphLayout;
using numi::matter::detail::appendRegionalSchurSurfaceVertexAnchors;
using numi::matter::detail::RegionalSchurSurfaceFace;
using numi::matter::detail::regionalSchurContactTopologyIsImmutable;

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
                  overlapLayout.patches[1].interiorCount==1u,
        "overlapping graph patches did not retain their disjoint interface",checks);

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
    // In the immutable compact surface map, contact primitives use the
    // first adjacency/side only. Each active contact corner is sourced from
    // the triangle's three primitive vertices (mortar features remain
    // barycentric on those same vertices); the opposite tetrahedron vertex
    // and the non-emitted second adjacency must not be anchored.
    constexpr std::uint32_t disabledContactFace = 1u << 31u;
    const std::array<std::array<std::uint32_t,4>,3> surfaceVertexTets{{
        {{0u,1u,2u,3u}}, {{1u,2u,3u,6u}}, {{2u,3u,4u,5u}}
    }};
    const std::array<RegionalSchurSurfaceFace,3> emittedSurfaceFaces{{
        {{{0u,1u}}, {{0u,3u}}, 0u}, // emitted tet 0 side 0 => {1,2,3}
        {{{2u,UINT32_MAX}}, {{2u,UINT32_MAX}}, disabledContactFace},
        {{{2u,UINT32_MAX}}, {{1u,UINT32_MAX}}, 0u}, // {2,4,5}, excludes 3
    }};
    std::vector<std::uint32_t> surfaceAnchors;
    ok &= require(appendRegionalSchurSurfaceVertexAnchors(
            surfaceVertexTets, emittedSurfaceFaces, disabledContactFace,
            surfaceAnchors),
        "valid emitted static surface faces failed to produce vertex anchors",checks);
    std::sort(surfaceAnchors.begin(), surfaceAnchors.end());
    surfaceAnchors.erase(std::unique(surfaceAnchors.begin(), surfaceAnchors.end()),
                         surfaceAnchors.end());
    const std::vector<std::uint32_t> expectedSurfaceVertices{1u,2u,3u,4u,5u};
    ok &= require(surfaceAnchors == expectedSurfaceVertices,
        "surface anchors included an opposite/disabled/non-emitted node",checks);
    RegionalGraphSchwarzLayout surfaceGraph;
    surfaceGraph.nodeCount=7u;
    surfaceGraph.environmentCount=1u;
    surfaceGraph.patchNodes={0u,1u,2u,3u,4u,5u,6u};
    surfaceGraph.patches={{0u,7u,0u,0u}};
    const std::array<std::array<std::uint32_t,2>,3> surfaceSources{{
        {{0u,10u}}, {{0u,10u}}, {{0u,10u}}
    }};
    const auto surfaceLayout=buildRegionalSchurGraphLayout(
        surfaceGraph,surfaceVertexTets,surfaceSources,surfaceAnchors);
    ok &= require(surfaceLayout.valid() &&
                  surfaceLayout.patches[0].interiorCount==2u &&
                  surfaceLayout.patches[0].interfaceCount==5u &&
                  surfaceGraph.patchNodes[surfaceLayout.interiorPatchValues[0]]==0u &&
                  surfaceGraph.patchNodes[surfaceLayout.interiorPatchValues[1]]==6u,
        "three-vertex contact superset did not retain both adjacent tet opposite vertices",checks);

    // A disabled emitted record is ignored before touching its invalid
    // geometry, matching nm_contact_build_surface_primitives' disabled-face
    // early return. An enabled invalid descriptor fails without appending a
    // partial anchor prefix.
    const std::array<RegionalSchurSurfaceFace,1> disabledInvalidFace{{
        {{{UINT32_MAX,UINT32_MAX}}, {{UINT32_MAX,UINT32_MAX}},
         disabledContactFace}
    }};
    const std::size_t anchorCountBeforeDisabled = surfaceAnchors.size();
    ok &= require(appendRegionalSchurSurfaceVertexAnchors(
            surfaceVertexTets, disabledInvalidFace, disabledContactFace,
            surfaceAnchors) &&
          surfaceAnchors.size()==anchorCountBeforeDisabled,
        "disabled surface face was not skipped safely",checks);
    const std::array<RegionalSchurSurfaceFace,2> malformedEnabledFaces{{
        {{{0u,UINT32_MAX}}, {{0u,UINT32_MAX}}, 0u},
        {{{99u,UINT32_MAX}}, {{2u,UINT32_MAX}}, 0u}
    }};
    const std::size_t anchorCountBeforeMalformed = surfaceAnchors.size();
    ok &= require(!appendRegionalSchurSurfaceVertexAnchors(
            surfaceVertexTets, malformedEnabledFaces, disabledContactFace,
            surfaceAnchors) &&
          surfaceAnchors.size()==anchorCountBeforeMalformed,
        "malformed enabled surface map did not fail atomically",checks);

    // Mutation and puncture topology remain fail-closed for this static
    // surface-map proof; it cannot be reused after a topology-changing edit.
    ok &= require(regionalSchurContactTopologyIsImmutable(true,true) &&
                  !regionalSchurContactTopologyIsImmutable(false,true) &&
                  !regionalSchurContactTopologyIsImmutable(true,false) &&
                  !regionalSchurContactTopologyIsImmutable(false,false),
        "mutable or punctured FEM topology was admitted for static anchors",checks);

    // The non-compact dynamic-map fallback retains the previous conservative
    // behavior and validates both adjacency indices.
    const std::array<std::array<std::uint32_t,4>,3> surfaceTets{{
        {{0u,1u,2u,3u}}, {{1u,2u,3u,4u}}, {{2u,3u,4u,5u}}
    }};
    const std::array<std::array<std::uint32_t,2>,1> dynamicSurface{{{{0u,1u}}}};
    RegionalGraphSchwarzLayout dynamicGraph;
    dynamicGraph.nodeCount=6u;
    dynamicGraph.environmentCount=1u;
    dynamicGraph.patchNodes={0u,1u,2u,3u,4u,5u};
    dynamicGraph.patches={{0u,6u,0u,0u}};
    std::vector<std::uint32_t> dynamicAnchors;
    ok &= require(numi::matter::detail::appendRegionalSchurSurfaceTetAnchors(
            surfaceTets, dynamicSurface, dynamicAnchors),
        "valid dynamic surface adjacency failed conservative fallback",checks);
    const std::array<std::array<std::uint32_t,2>,1> invalidSurface{{{{3u,UINT32_MAX}}}};
    const std::size_t dynamicAnchorCountBeforeInvalid = dynamicAnchors.size();
    ok &= require(!numi::matter::detail::appendRegionalSchurSurfaceTetAnchors(
            surfaceTets, invalidSurface, dynamicAnchors) &&
          dynamicAnchors.size()==dynamicAnchorCountBeforeInvalid,
        "out-of-range dynamic surface adjacency was not rejected atomically",checks);
    // A runtime implicit spring endpoint must also be supplied as an external
    // anchor; if it consumes the last interior, exact condensation is disabled.
    dynamicAnchors.push_back(5u);
    ok &= require(!buildRegionalSchurGraphLayout(
            dynamicGraph,surfaceTets,surfaceSources,dynamicAnchors).valid(),
        "spring endpoint that covers the last interior did not fail closed",checks);

    // A 64-node cap creates dense local arenas larger than this fixture's
    // budget. The production builder must repartition into smaller bounded
    // cores while retaining every node and a nonempty exact interior.
    constexpr std::uint32_t fanGroups = 5u;
    constexpr std::uint32_t tetsPerFan = 50u;
    const std::uint32_t fanNodeCount =
        fanGroups * (1u + 3u * tetsPerFan);
    std::vector<std::array<std::uint32_t, 4>> fanTets;
    std::vector<std::array<std::uint32_t, 2>> fanSources;
    std::vector<std::uint8_t> fanFreeMask(fanNodeCount, 1u);
    fanTets.reserve(fanGroups * tetsPerFan);
    fanSources.reserve(fanGroups * tetsPerFan);
    for (std::uint32_t group = 0u; group < fanGroups; ++group) {
        const std::uint32_t groupBase = group * (1u + 3u * tetsPerFan);
        for (std::uint32_t tet = 0u; tet < tetsPerFan; ++tet) {
            const std::uint32_t firstLeaf = groupBase + 1u + 3u * tet;
            fanTets.push_back({groupBase, firstLeaf, firstLeaf + 1u,
                               firstLeaf + 2u});
            fanSources.push_back({0u, 10u});
        }
    }
    constexpr std::uint64_t boundedBytes = 5ull * 1024ull * 1024ull +
                                           512ull * 1024ull;
    const auto oversizedCore =
        numi::matter::detail::buildRegionalSchurElementCoreLayoutAttempt(
            fanNodeCount, fanTets, fanSources, fanFreeMask, {}, 1u, 64u,
            16384u, boundedBytes);
    ok &= require(!oversizedCore.valid() &&
                  oversizedCore.graph.deviceBytes > boundedBytes,
        "oversized 64-node core fixture did not exercise the arena bound",checks);
    const auto adaptiveCore =
        numi::matter::detail::buildRegionalSchurElementCoreLayout(
            fanNodeCount, fanTets, fanSources, fanFreeMask, {}, 1u, 64u,
            16384u, boundedBytes);
    ok &= require(adaptiveCore.valid() &&
                  adaptiveCore.graph.deviceBytes <= boundedBytes,
        "adaptive exact partition did not fit the unchanged arena bound",checks);
    std::uint32_t largestAdaptivePatch = 0u;
    for (const RegionalGraphSchwarzPatch& patch : adaptiveCore.graph.patches)
        largestAdaptivePatch = std::max(largestAdaptivePatch, patch.nodeCount);
    ok &= require(largestAdaptivePatch <= 56u &&
                  !adaptiveCore.extractionSources.empty(),
        "arena fallback did not choose bounded cores with eliminable interiors",checks);
    bool fullNodeCoverage = adaptiveCore.graph.nodeCoverage.size() == fanNodeCount;
    for (const std::uint32_t coverage : adaptiveCore.graph.nodeCoverage)
        fullNodeCoverage = fullNodeCoverage && coverage != 0u;
    for (const RegionalSchurPatchRange& patch : adaptiveCore.partition.patches)
        fullNodeCoverage = fullNodeCoverage &&
            patch.interiorCount + patch.interfaceCount > 0u;
    ok &= require(fullNodeCoverage,
        "adaptive partition dropped nodes or left a patch unclassified",checks);

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
