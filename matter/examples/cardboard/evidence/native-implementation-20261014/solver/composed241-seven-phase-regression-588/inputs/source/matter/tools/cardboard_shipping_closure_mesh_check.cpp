#include "cardboard_shipping_closure_mesh.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using namespace numi::cardboard;

void check(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

ConformingGlueMesh simpleSection(const double length, const double caliper) {
    ConformingGlueMesh section;
    section.config.length = length;
    section.config.pitch = 0.0079;
    section.config.caliper = caliper;
    section.config.linerThickness = caliper;
    section.config.mediumThickness = 0.0;
    section.points = {{0.0,0.0},{length,0.0},{length,caliper},{0.0,caliper}};
    section.triangles = {
        {{0u,1u,2u},CellMaterial::liner,0.0},
        {{0u,2u,3u},CellMaterial::liner,0.0},
    };
    section.crossSectionAreaByMaterial[0] = length*caliper;
    return section;
}

void testAllNineDeclaredClosureFootprints() {
    constexpr double length=0.325;
    constexpr double width=0.200;
    constexpr double height=0.200;
    constexpr double joint=0.025;
    constexpr double kerf=0.001;
    constexpr double caliper=0.004210;
    constexpr double strip=0.010;
    auto section=simpleSection(joint+2.0*(length+width),caliper);
    FEFCO0201BlankConfig config{length,width,height,joint,kerf,16u};
    const auto blank=buildFEFCO0201Blank(config,section);
    std::vector<BoxBlankPoint> nodes=blank.points;
    std::vector<ShippingClosureTet> tets;
    tets.reserve(blank.tetrahedra.size());
    for (const auto& tet:blank.tetrahedra)
        tets.push_back({tet.nodes,static_cast<std::uint32_t>(tet.material)});
    const auto prepared=buildShippingClosureMortarMesh(
        blank,nodes,tets,strip,0.0001,0u,100u);
    check(prepared.groups.size()==9u,"shipping closure mesh must emit lap plus eight end patches");
    std::size_t patchCount=0u;
    for (const auto& group:prepared.groups) {
        check(!group.stableIdentifiers.empty(),"declared adhesive region has no mortar triangles");
        check(std::abs(group.preparedAdhesiveAreaM2-group.expectedAdhesiveAreaM2)
                <=2.0e-8*group.expectedAdhesiveAreaM2,
              "mortar overlay did not conserve adhesive footprint area");
        patchCount+=group.stableIdentifiers.size();
    }
    check(patchCount==prepared.patches.size(),"group ID accounting differs from patch table");
    check(patchCount>9u,"nonmatching mesh overlay should subdivide at least one declared footprint");
    for (std::size_t index=0u;index<prepared.patches.size();++index) {
        const auto& patch=prepared.patches[index];
        check(patch.stableIdentifier==100u+index,"stable patch IDs are not deterministic and contiguous");
        check(patch.firstTetrahedron<tets.size() && patch.secondTetrahedron<tets.size(),
              "mortar patch parent tet is invalid");
        check(patch.cohesiveMaterialIndex==0u,"mortar patch law table index changed");
        for (const auto& row:patch.sideAVertexBarycentric) {
            double sum=0.0;
            for (const double value:row) { check(value>=0.0 && value<=1.0,"invalid side A barycentric coordinate"); sum+=value; }
            check(std::abs(sum-1.0)<1.0e-12,"side A barycentric row does not sum to one");
        }
        for (const auto& row:patch.sideBVertexBarycentric) {
            double sum=0.0;
            for (const double value:row) { check(value>=0.0 && value<=1.0,"invalid side B barycentric coordinate"); sum+=value; }
            check(std::abs(sum-1.0)<1.0e-12,"side B barycentric row does not sum to one");
        }
        for (const auto a:patch.sideANodes)
            for (const auto b:patch.sideBNodes)
                check(a!=b,"paired HMA faces unexpectedly share node identity");
    }
}

void testExplicitFailureOnMissingFootprintGeometry() {
    constexpr double length=0.325;
    constexpr double width=0.200;
    constexpr double joint=0.025;
    constexpr double caliper=0.004210;
    auto section=simpleSection(joint+2.0*(length+width),caliper);
    const auto blank=buildFEFCO0201Blank(
        FEFCO0201BlankConfig{length,width,0.200,joint,0.001,4u},section);
    std::vector<BoxBlankPoint> nodes=blank.points;
    std::vector<ShippingClosureTet> tets;
    for (const auto& tet:blank.tetrahedra)
    {
        double centerX=0.0;
        for (const auto node:tet.nodes) centerX+=blank.points.at(node).x/4.0;
        // Remove one whole wall region so the manufacturer-lap side has no
        // source faces. The overlay must reject that missing geometry.
        if (centerX < blank.panels.at(3u).xStart)
            tets.push_back({tet.nodes,static_cast<std::uint32_t>(tet.material)});
    }
    bool rejected=false;
    try { (void)buildShippingClosureMortarMesh(blank,nodes,tets,0.010,0.0001,0u,1u); }
    catch (const std::invalid_argument&) { rejected=true; }
    check(rejected,"mesh span mismatch must fail closed before generating closure patches");
}

} // namespace

int main() {
    try {
        testAllNineDeclaredClosureFootprints();
        testExplicitFailureOnMissingFootprintGeometry();
        std::cout << "cardboard shipping closure mesh checks passed: 2\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cardboard shipping closure mesh checks failed: " << error.what() << '\n';
        return 1;
    }
}
