#pragma once

// Captures the source skin/fat object's actual cooked FEM cell masses as the
// parent ledger for a new child-volume transfer. This is read-only: it neither
// compiles nor mutates the accepted source world, and it does not require an
// ancestry map for the already-existing parent mesh.
#include "NumiHumanUnpairedSkinFatVolume.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace metalrobo::human::coarse_tissue {

struct CompiledSkinFatDonorMassTotal {
    std::uint32_t materialIndex = NM_INVALID_INDEX;
    std::uint32_t donorBodyIndex = NM_INVALID_INDEX;
    double massKg = 0.0;
};

struct CompiledSkinFatMassCaptureReport {
    bool usesAuthoredPerTetrahedronMasses = false;
    std::size_t objectIndex = 0u;
    std::size_t nodeCount = 0u;
    std::size_t tetrahedronCount = 0u;
    double parentMassKg = 0.0;
    double maximumCompiledNodalMassClosureErrorKg = 0.0;
    double maximumCompiledNodalMassClosureRelativeError = 0.0;
    double maximumPreparedDonorMassClosureErrorKg = 0.0;
    double maximumPreparedDonorMassClosureRelativeError = 0.0;
    std::vector<CompiledSkinFatDonorMassTotal> massByMaterialDonor;
};

// The output mass vector is indexed by the baseline ObjectSource tet row.
// Source-contribution parentNodeIndex values are object-local node rows, as
// required by ExplicitUnpairedVolumeTopology. Each cell contributes exactly
// one quarter of its cooked mass to each corner; the corner's existing donor
// frame owner receives that source-mass share. This matches the compiler's
// lumped nodal assembly and preserves material and donor mass totals.
[[nodiscard]] inline bool captureCompiledSkinFatParentMassLedger(
        const numi::matter::WorldSource& source,
        const numi::matter::CompiledWorld& compiled,
        const std::size_t objectIndex,
        const std::span<const std::uint32_t> nodeDonorBodyIndices,
        const std::span<const V3> nodeDonorLocalPoints,
        const std::span<const NumiHumanTissueMassNode> preparedDonorMassNodes,
        std::vector<double>& parentTetrahedronMassKg,
        std::vector<ExplicitUnpairedVolumeSourceMassContribution>& parentNodeSourceMass,
        CompiledSkinFatMassCaptureReport& report,
        std::string& error) {
    const auto fail = [&](std::string message) {
        error = std::move(message);
        return false;
    };
    if (objectIndex >= source.objects.size() ||
        objectIndex >= compiled.objects.size())
        return fail("compiled skin/fat mass capture object index is out of range");
    const auto& object = source.objects[objectIndex];
    const auto& descriptor = compiled.objects[objectIndex];
    if (object.representation != numi::matter::Representation::fem ||
        object.femNodes.size() < 4u || object.tetrahedra.empty())
        return fail("compiled skin/fat mass capture requires a nonempty FEM ObjectSource");
    const std::size_t nodeCount = object.femNodes.size();
    const std::size_t tetCount = object.tetrahedra.size();
    if (descriptor.stateCount != nodeCount || descriptor.elementCount != tetCount ||
        static_cast<std::size_t>(descriptor.stateOffset) + nodeCount >
            compiled.fem.nodes.size() ||
        static_cast<std::size_t>(descriptor.elementOffset) + tetCount >
            compiled.fem.tetrahedra.size())
        return fail("compiled skin/fat FEM ranges do not match the source object");
    if (nodeDonorBodyIndices.size() != nodeCount ||
        nodeDonorLocalPoints.size() != nodeCount ||
        preparedDonorMassNodes.size() != nodeCount)
        return fail("compiled skin/fat donor body, local point, and prepared mass rows must cover every source node");
    if (!object.femReferenceNodes.empty() &&
        object.femReferenceNodes.size() != nodeCount)
        return fail("compiled skin/fat reference coordinate rows do not cover every source node");
    if (!object.femMaterialIndices.empty() &&
        object.femMaterialIndices.size() != tetCount)
        return fail("compiled skin/fat material rows do not cover every source tetrahedron");

    const auto hasIdentity = [](const auto& identity) {
        return std::any_of(identity.begin(), identity.end(),
                           [](std::uint64_t word) { return word != 0u; });
    };
    const bool authoredMasses = !object.femTetrahedronMassesKg.empty();
    const bool compiledMassIdentity =
        objectIndex < compiled.femTetrahedronMassSourceIdentities.size() &&
        hasIdentity(compiled.femTetrahedronMassSourceIdentities[objectIndex]);
    if (authoredMasses != compiledMassIdentity ||
        (authoredMasses &&
         (object.femTetrahedronMassesKg.size() != tetCount ||
          !hasIdentity(object.femTetrahedronMassSourceIdentity) ||
          compiled.femTetrahedronMassSourceIdentities[objectIndex] !=
              object.femTetrahedronMassSourceIdentity ||
          static_cast<std::size_t>(descriptor.elementOffset) + tetCount >
              compiled.femTetrahedronMassesKg.size())))
        return fail("compiled skin/fat explicit cell-mass identity or rows do not match the source object");

    std::vector<const NumiHumanTissueMassNode*> donorRowByLocalNode(nodeCount, nullptr);
    for (const auto& row : preparedDonorMassNodes) {
        if (row.nodeIndex < descriptor.stateOffset ||
            static_cast<std::size_t>(row.nodeIndex - descriptor.stateOffset) >= nodeCount ||
            row.donorBody == NM_INVALID_INDEX ||
            !std::isfinite(row.massKg) || row.massKg <= 0.0 ||
            !std::all_of(row.localPosition.begin(), row.localPosition.end(),
                         [](double x) { return std::isfinite(x); }))
            return fail("prepared skin/fat donor mass row has an invalid node, donor, mass, or local point");
        const auto local = static_cast<std::size_t>(row.nodeIndex - descriptor.stateOffset);
        if (donorRowByLocalNode[local] != nullptr)
            return fail("prepared skin/fat donor mass rows duplicate a source node");
        if (row.donorBody != nodeDonorBodyIndices[local])
            return fail("prepared skin/fat donor row disagrees with the source node frame owner");
        V3 delta{};
        for (std::size_t axis = 0; axis < 3u; ++axis) {
            if (!std::isfinite(nodeDonorLocalPoints[local][axis]))
                return fail("compiled skin/fat donor local point is nonfinite");
            delta[axis] = row.localPosition[axis] - nodeDonorLocalPoints[local][axis];
        }
        if (norm(delta) > 1.0e-7)
            return fail("prepared skin/fat donor local point differs from the compiled node frame");
        donorRowByLocalNode[local] = &row;
    }
    if (std::any_of(donorRowByLocalNode.begin(), donorRowByLocalNode.end(),
                    [](const auto* row) { return row == nullptr; }))
        return fail("prepared skin/fat donor mass rows omit a source node");

    const auto sourceDensity = [&](const std::uint32_t materialIndex,
                                   double& density) {
        if (materialIndex >= source.materials.size() ||
            materialIndex >= compiled.materials.size()) return false;
        const auto& parameters = source.materials[materialIndex].parameters;
        const auto found = std::find_if(parameters.begin(), parameters.end(),
            [](const auto& parameter) { return parameter.name == "density"; });
        if (found == parameters.end() || !std::isfinite(found->defaultValue) ||
            found->defaultValue <= 0.0) return false;
        density = found->defaultValue;
        return true;
    };

    const bool referenced = !object.femReferenceNodes.empty();
    const bool regional = !object.femMaterialIndices.empty();
    std::vector<double> massByTet(tetCount, 0.0);
    std::vector<double> reassembledNodalMass(nodeCount, 0.0);
    using NodeMaterialDonor = std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>;
    std::map<NodeMaterialDonor, double> massByNodeMaterialDonor;
    std::map<std::pair<std::uint32_t, std::uint32_t>, double> massByMaterialDonor;

    const auto position = [&](const std::uint32_t globalNode) {
        const auto& state = compiled.fem.nodes[globalNode];
        const auto p = referenced ? state.restAndFixed : state.positionAndMass;
        return V3{static_cast<double>(p.x), static_cast<double>(p.y),
                  static_cast<double>(p.z)};
    };

    for (std::size_t tetRow = 0; tetRow < tetCount; ++tetRow) {
        const auto& sourceTet = object.tetrahedra[tetRow].nodes;
        const auto& cookedTet =
            compiled.fem.tetrahedra[descriptor.elementOffset + tetRow];
        const std::uint32_t materialIndex =
            regional ? object.femMaterialIndices[tetRow] : object.materialIndex;
        if (materialIndex >= compiled.materials.size() ||
            cookedTet.identity.x != materialIndex ||
            cookedTet.identity.y != objectIndex)
            return fail("compiled skin/fat tetrahedron material or object identity differs from its source row");
        const auto cookedNode = [&](const std::size_t corner) {
            switch (corner) {
                case 0u: return cookedTet.nodes.x;
                case 1u: return cookedTet.nodes.y;
                case 2u: return cookedTet.nodes.z;
                default: return cookedTet.nodes.w;
            }
        };
        std::array<std::uint32_t, 4> localNodes{};
        for (std::size_t corner = 0; corner < 4u; ++corner) {
            const auto local = sourceTet[corner];
            if (local >= nodeCount ||
                cookedNode(corner) != descriptor.stateOffset + local)
                return fail("compiled skin/fat tetrahedron node order differs from its source row");
            localNodes[corner] = local;
        }

        const auto p0 = position(cookedNode(0u));
        const auto p1 = position(cookedNode(1u));
        const auto p2 = position(cookedNode(2u));
        const auto p3 = position(cookedNode(3u));
        const V3 a{p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]};
        const V3 b{p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]};
        const V3 c{p3[0]-p0[0], p3[1]-p0[1], p3[2]-p0[2]};
        const double signedVolume =
            (a[0] * (b[1]*c[2] - b[2]*c[1]) -
             b[0] * (a[1]*c[2] - a[2]*c[1]) +
             c[0] * (a[1]*b[2] - a[2]*b[1])) / 6.0;
        if (!std::isfinite(signedVolume) || signedVolume <= 0.0 ||
            !std::isfinite(cookedTet.inverseRestRow0.w) ||
            cookedTet.inverseRestRow0.w <= 0.0f ||
            static_cast<float>(signedVolume) != cookedTet.inverseRestRow0.w)
            return fail("compiled skin/fat cooked rest volume is invalid or differs from its exact FP32 rest nodes");

        double cellMass = 0.0;
        if (authoredMasses) {
            cellMass = compiled.femTetrahedronMassesKg[
                descriptor.elementOffset + tetRow];
            if (!std::isfinite(cellMass) || cellMass <= 0.0 ||
                cellMass != object.femTetrahedronMassesKg[tetRow])
                return fail("compiled explicit skin/fat cell mass differs from its authored source row");
        } else if (regional || referenced) {
            const double density =
                static_cast<double>(compiled.materials[materialIndex].bulk.x);
            if (!std::isfinite(density) || density <= 0.0)
                return fail("compiled skin/fat material density is invalid");
            cellMass = density * static_cast<double>(cookedTet.inverseRestRow0.w);
        } else {
            double density = 0.0;
            if (!sourceDensity(materialIndex, density))
                return fail("legacy compiled skin/fat source has no fixed positive density");
            cellMass = density * signedVolume;
        }
        if (!std::isfinite(cellMass) || cellMass <= 0.0)
            return fail("compiled skin/fat cell mass is nonpositive or nonfinite");
        massByTet[tetRow] = cellMass;
        const double cornerMass = cellMass * 0.25;
        for (const auto local : localNodes) {
            reassembledNodalMass[local] += cornerMass;
            const auto donor = nodeDonorBodyIndices[local];
            massByNodeMaterialDonor[{local, materialIndex, donor}] += cornerMass;
            massByMaterialDonor[{materialIndex, donor}] += cornerMass;
        }
    }

    CompiledSkinFatMassCaptureReport stagedReport;
    stagedReport.usesAuthoredPerTetrahedronMasses = authoredMasses;
    stagedReport.objectIndex = objectIndex;
    stagedReport.nodeCount = nodeCount;
    stagedReport.tetrahedronCount = tetCount;
    std::vector<ExplicitUnpairedVolumeSourceMassContribution> sourceRows;
    sourceRows.reserve(massByNodeMaterialDonor.size());
    for (const auto& [key, mass] : massByNodeMaterialDonor) {
        const auto [node, materialIndex, donor] = key;
        const auto* donorRow = donorRowByLocalNode[node];
        sourceRows.push_back({
            .parentNodeIndex = node,
            .materialIndex = materialIndex,
            .donorBodyIndex = donor,
            .massKg = mass,
            .donorLocalPosition = donorRow->localPosition});
    }

    for (const auto mass : massByTet) stagedReport.parentMassKg += mass;
    for (const auto& [key, mass] : massByMaterialDonor)
        stagedReport.massByMaterialDonor.push_back({
            .materialIndex = key.first, .donorBodyIndex = key.second, .massKg = mass});

    for (std::size_t node = 0; node < nodeCount; ++node) {
        const auto& compiledNode = compiled.fem.nodes[descriptor.stateOffset + node];
        const float cookedExpectedMass =
            static_cast<float>(reassembledNodalMass[node]);
        const double compiledMass = static_cast<double>(compiledNode.positionAndMass.w);
        const double compiledError =
            std::abs(static_cast<double>(cookedExpectedMass) - compiledMass);
        const double compiledRelative = compiledError /
            std::max({std::abs(compiledMass),
                      std::abs(static_cast<double>(cookedExpectedMass)), 1.0e-30});
        stagedReport.maximumCompiledNodalMassClosureErrorKg =
            std::max(stagedReport.maximumCompiledNodalMassClosureErrorKg, compiledError);
        stagedReport.maximumCompiledNodalMassClosureRelativeError =
            std::max(stagedReport.maximumCompiledNodalMassClosureRelativeError,
                     compiledRelative);
        if (cookedExpectedMass != compiledNode.positionAndMass.w)
            return fail("captured skin/fat cell masses do not reproduce the compiled FEM nodal mass");

        const double preparedMass = donorRowByLocalNode[node]->massKg;
        const double preparedError =
            std::abs(reassembledNodalMass[node] - preparedMass);
        const double preparedRelative = preparedError /
            std::max({std::abs(reassembledNodalMass[node]), std::abs(preparedMass), 1.0e-30});
        stagedReport.maximumPreparedDonorMassClosureErrorKg =
            std::max(stagedReport.maximumPreparedDonorMassClosureErrorKg, preparedError);
        stagedReport.maximumPreparedDonorMassClosureRelativeError =
            std::max(stagedReport.maximumPreparedDonorMassClosureRelativeError,
                     preparedRelative);
        if (preparedError >
            std::max(1.0e-10, 5.0e-4 *
                std::max(std::abs(reassembledNodalMass[node]), std::abs(preparedMass))))
            return fail("captured compiled node mass differs from its registered donor debit beyond the existing FP32 cook bound");
    }

    parentTetrahedronMassKg = std::move(massByTet);
    parentNodeSourceMass = std::move(sourceRows);
    report = std::move(stagedReport);
    error.clear();
    return true;
}

} // namespace metalrobo::human::coarse_tissue
