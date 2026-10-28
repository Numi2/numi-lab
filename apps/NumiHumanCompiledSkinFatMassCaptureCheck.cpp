#include "NumiHumanCompiledSkinFatMassCapture.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

int main() {
    numi::matter::WorldSource source;
    source.materials.resize(1);
    source.materials[0].name = "skin";
    source.materials[0].parameters.push_back(
        {.name="density", .defaultValue=1000.0, .lower=1.0, .upper=5000.0});
    source.objects.resize(1);
    auto& object = source.objects[0];
    object.name = "compiled skin/fat parent fixture";
    object.representation = numi::matter::Representation::fem;
    object.mixedFEM = false;
    object.materialIndex = 0u;
    object.femNodes = {
        V3{0.0, 0.0, 0.0}, V3{1.0, 0.0, 0.0},
        V3{0.0, 1.0, 0.0}, V3{0.0, 0.0, 1.0}};
    object.femReferenceNodes = object.femNodes;
    object.femReferenceSourceIdentity[0] = 1u;
    object.femMaterialIndices = {0u};
    object.femMaterialSourceIdentity[0] = 2u;
    object.tetrahedra.push_back({{0u, 1u, 2u, 3u}});

    numi::matter::CompiledWorld compiled;
    compiled.objects.resize(1);
    compiled.objects[0].stateOffset = 0u;
    compiled.objects[0].stateCount = 4u;
    compiled.objects[0].elementOffset = 0u;
    compiled.objects[0].elementCount = 1u;
    compiled.materials.resize(1);
    compiled.materials[0].bulk.x = 1000.0f;
    compiled.fem.nodes.resize(4);
    compiled.fem.tetrahedra.resize(1);
    auto& cookedTet = compiled.fem.tetrahedra[0];
    cookedTet.nodes = {0u, 1u, 2u, 3u};
    cookedTet.inverseRestRow0 = {0.0f, 0.0f, 0.0f, 1.0f / 6.0f};
    cookedTet.identity = {0u, 0u, 0u, 0u};
    const double parentMass =
        static_cast<double>(compiled.materials[0].bulk.x) *
        static_cast<double>(cookedTet.inverseRestRow0.w);
    const float nodeMass = static_cast<float>(parentMass * 0.25);
    const std::array<std::uint32_t, 4> donors{11u, 11u, 22u, 22u};
    const std::array<V3, 4> localPoints{
        V3{0.01, 0.00, 0.00}, V3{0.02, 0.00, 0.00},
        V3{0.00, 0.01, 0.00}, V3{0.00, 0.02, 0.00}};
    std::vector<metalrobo::NumiHumanTissueMassNode> prepared;
    for (std::uint32_t node = 0; node < 4u; ++node) {
        const auto& p = object.femNodes[node];
        compiled.fem.nodes[node].restAndFixed =
            {static_cast<float>(p[0]), static_cast<float>(p[1]),
             static_cast<float>(p[2]), 0.0f};
        compiled.fem.nodes[node].positionAndMass =
            {static_cast<float>(p[0]), static_cast<float>(p[1]),
             static_cast<float>(p[2]), nodeMass};
        prepared.push_back({.nodeIndex=node, .donorBody=donors[node],
                            .massKg=static_cast<double>(nodeMass),
                            .localPosition=localPoints[node]});
    }

    std::vector<double> parentMasses;
    std::vector<ExplicitUnpairedVolumeSourceMassContribution> sourceMass;
    CompiledSkinFatMassCaptureReport report;
    std::string error;
    if (!captureCompiledSkinFatParentMassLedger(
            source, compiled, 0u, donors, localPoints, prepared,
            parentMasses, sourceMass, report, error)) {
        std::cerr << "density-derived capture failed: " << error << '\n';
        return 1;
    }
    if (parentMasses.size() != 1u || parentMasses[0] != parentMass ||
        sourceMass.size() != 4u ||
        std::abs(report.parentMassKg - parentMass) > 1.0e-12 ||
        report.usesAuthoredPerTetrahedronMasses ||
        report.maximumCompiledNodalMassClosureErrorKg != 0.0) {
        std::cerr << "density-derived captured mass or source rows do not close\n";
        return 2;
    }
    double attributed = 0.0;
    for (const auto& row : sourceMass) attributed += row.massKg;
    if (std::abs(attributed - parentMass) > 1.0e-12 ||
        report.massByMaterialDonor.size() != 2u) {
        std::cerr << "density-derived donor attribution does not close\n";
        return 3;
    }

    object.femTetrahedronMassesKg = {0.2};
    object.femTetrahedronMassSourceIdentity[0] = 9u;
    compiled.femTetrahedronMassSourceIdentities = {{9u, 0u, 0u, 0u}};
    compiled.femTetrahedronMassesKg = {0.2};
    for (auto& node : compiled.fem.nodes) node.positionAndMass.w = 0.05f;
    for (auto& row : prepared) row.massKg = 0.05;
    if (!captureCompiledSkinFatParentMassLedger(
            source, compiled, 0u, donors, localPoints, prepared,
            parentMasses, sourceMass, report, error)) {
        std::cerr << "authored-mass capture failed: " << error << '\n';
        return 4;
    }
    if (parentMasses.size() != 1u || parentMasses[0] != 0.2 ||
        !report.usesAuthoredPerTetrahedronMasses) {
        std::cerr << "authored per-tet mass was not captured exactly\n";
        return 5;
    }

    std::cout << "compiled skin/fat mass capture PASS"
              << " density_parent_kg=" << parentMass
              << " attributed_kg=" << attributed
              << " explicit_parent_kg=" << parentMasses[0] << '\n';
    return 0;
}
