#include "cardboard_starch_interface_mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using namespace numi::cardboard;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool near(double actual, double expected, double absTol = 1.0e-13,
          double relTol = 1.0e-11) {
    return std::abs(actual - expected) <= absTol + relTol * std::abs(expected);
}

template<class F>
void expectInvalid(F&& action, const std::string& message) {
    bool rejected = false;
    try { action(); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, message);
}

StarchInterfaceTet tet(std::array<std::size_t, 4> nodes,
                       std::uint8_t material, double density) {
    StarchInterfaceTet result;
    result.nodeIds = nodes;
    result.materialIndex = material;
    result.densityKgPerM3 = density;
    return result;
}

StarchInterfaceMesh sampleMesh() {
    StarchInterfaceMesh mesh;
    mesh.referenceNodesM = {
        {0.0, 0.0, 0.0},   // A
        {1.0, 0.0, 0.0},   // B
        {0.0, 1.0, 0.0},   // C
        {0.0, 0.0, 1.0},   // D
        {2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0}, // E, outside face BCD
        {0.0, 0.0, -1.0},  // F, outside face ABC
        {-1.0, 0.0, 0.0},  // G, outside face ACD
    };
    auto glueA = tet({0u, 1u, 2u, 3u}, kStarchMaterialGlue, 760.0);
    // Source material frame must survive topology preparation byte-for-byte.
    glueA.materialFrameToReference = {
        0.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
        -1.0, 0.0, 0.0};
    glueA.boundarySlotByFace[2] = 23; // exterior face ABD, arbitrary cut slot id
    const auto glueB = tet({1u, 2u, 3u, 4u}, kStarchMaterialGlue, 780.0);
    const auto liner = tet({0u, 1u, 2u, 5u}, kStarchMaterialLiner, 840.0);
    const auto medium = tet({0u, 2u, 3u, 6u}, kStarchMaterialMedium, 720.0);
    mesh.tetrahedra = {glueA, glueB, liner, medium};
    return mesh;
}

void validateFaceCorrespondence(const StarchPreparedInterfaceMesh& prepared) {
    require(prepared.pairedFaces.size() == 2u,
            "finite-glue fixture should emit liner/glue and medium/glue faces");
    bool foundLiner = false;
    bool foundMedium = false;
    for (const auto& face : prepared.pairedFaces) {
        require(face.paperTetId < prepared.tetrahedra.size() &&
                face.glueTetId < prepared.tetrahedra.size() &&
                face.paperFaceSlot < 4u && face.glueFaceSlot < 4u &&
                !face.constitutiveLawInstalled && face.referenceAreaM2 > 0.0,
                "paired face record has invalid neighbors, slots, area, or law status");
        require(prepared.tetrahedra[face.paperTetId].materialIndex ==
                    face.paperMaterialIndex &&
                prepared.tetrahedra[face.glueTetId].materialIndex ==
                    kStarchMaterialGlue,
                "paired face material identities do not match neighboring tets");
        foundLiner |= face.paperMaterialIndex == kStarchMaterialLiner;
        foundMedium |= face.paperMaterialIndex == kStarchMaterialMedium;
        require(near(fractureDot(face.paperOutwardUnitNormal,
                                 face.glueOutwardUnitNormal), -1.0, 1.0e-12),
                "paired tet outward normals are not opposite");
        for (std::size_t i = 0; i < 3u; ++i) {
            require(fractureNorm(prepared.referenceNodesM[face.paperNodeIds[i]] -
                                 face.matchingReferenceCoordinatesM[i]) == 0.0 &&
                    fractureNorm(prepared.referenceNodesM[face.glueNodeIds[i]] -
                                 face.matchingReferenceCoordinatesM[i]) == 0.0,
                    "paired nodes do not preserve matching reference coordinates");
        }
        const auto paperFaceNodes = [&] {
            std::array<std::size_t, 3> result{};
            std::size_t write = 0u;
            const auto& nodes = prepared.tetrahedra[face.paperTetId].nodeIds;
            for (std::size_t i = 0; i < 4u; ++i)
                if (i != face.paperFaceSlot) result[write++] = nodes[i];
            std::sort(result.begin(), result.end());
            return result;
        }();
        const auto glueFaceNodes = [&] {
            std::array<std::size_t, 3> result{};
            std::size_t write = 0u;
            const auto& nodes = prepared.tetrahedra[face.glueTetId].nodeIds;
            for (std::size_t i = 0; i < 4u; ++i)
                if (i != face.glueFaceSlot) result[write++] = nodes[i];
            std::sort(result.begin(), result.end());
            return result;
        }();
        auto recordedPaper = face.paperNodeIds;
        auto recordedGlue = face.glueNodeIds;
        std::sort(recordedPaper.begin(), recordedPaper.end());
        std::sort(recordedGlue.begin(), recordedGlue.end());
        require(recordedPaper == paperFaceNodes && recordedGlue == glueFaceNodes,
                "recorded face slots do not resolve to their paired node triples");
    }
    require(foundLiner && foundMedium,
            "fixture did not retain both distinct paper-region interface types");
}

void testSplitMassFramesAndBoundarySlots() {
    const StarchInterfaceMesh source = sampleMesh();
    const auto prepared = prepareStarchInterfaceMesh(source);
    validateFaceCorrespondence(prepared);
    require(prepared.sourceNodeCount == 7u &&
            prepared.duplicatedGlueNodeCount == 4u &&
            prepared.referenceNodesM.size() == 11u &&
            prepared.tetrahedra.size() == source.tetrahedra.size(),
            "glue-side split count or volume topology changed unexpectedly");
    require(prepared.sourceMaterialVolumeM3 == prepared.preparedMaterialVolumeM3,
            "region volume changed across topology split");
    require(near(prepared.sourceMassAudit.totalMassKg,
                 prepared.preparedMassAudit.totalMassKg) &&
            near(prepared.sourceMassAudit.consistentMassMatrixTotalKg,
                 prepared.preparedMassAudit.consistentMassMatrixTotalKg),
            "consistent mass total changed across topology split");
    const double lumpedSum = std::accumulate(
        prepared.preparedMassAudit.lumpedNodalMassKg.begin(),
        prepared.preparedMassAudit.lumpedNodalMassKg.end(), 0.0);
    require(near(lumpedSum, prepared.preparedMassAudit.totalMassKg),
            "recomputed split-node lumped mass does not sum to total tet mass");

    for (std::size_t i = 0; i < source.tetrahedra.size(); ++i) {
        require(prepared.tetrahedra[i].materialIndex == source.tetrahedra[i].materialIndex &&
                prepared.tetrahedra[i].densityKgPerM3 == source.tetrahedra[i].densityKgPerM3 &&
                prepared.tetrahedra[i].materialFrameToReference ==
                    source.tetrahedra[i].materialFrameToReference &&
                prepared.tetrahedra[i].boundarySlotByFace ==
                    source.tetrahedra[i].boundarySlotByFace,
                "source material label, density, frame, or boundary slots were changed");
        if (source.tetrahedra[i].materialIndex != kStarchMaterialGlue)
            require(prepared.tetrahedra[i].nodeIds == source.tetrahedra[i].nodeIds,
                    "paper tetrahedron connectivity changed during glue-side split");
    }
    // The shared internal glue/glue face BCD remains shared after the split.
    const auto glueAInternal = std::array<std::size_t, 3>{
        prepared.tetrahedra[0].nodeIds[1], prepared.tetrahedra[0].nodeIds[2],
        prepared.tetrahedra[0].nodeIds[3]};
    const auto glueBInternal = std::array<std::size_t, 3>{
        prepared.tetrahedra[1].nodeIds[0], prepared.tetrahedra[1].nodeIds[1],
        prepared.tetrahedra[1].nodeIds[2]};
    auto glueASorted = glueAInternal;
    auto glueBSorted = glueBInternal;
    std::sort(glueASorted.begin(), glueASorted.end());
    std::sort(glueBSorted.begin(), glueBSorted.end());
    require(glueASorted == glueBSorted,
            "glue-side duplication broke internal glue connectivity");

    // A split node receives freshly recomputed mass contributions from each
    // side. It is not assigned a copy of the original combined nodal mass.
    for (std::size_t node = source.referenceNodesM.size();
         node < prepared.referenceNodesM.size(); ++node) {
        const std::size_t original = prepared.sourceNodeForPreparedNode[node];
        require(original < source.referenceNodesM.size() &&
                near(fractureNorm(prepared.referenceNodesM[node] -
                                  source.referenceNodesM[original]), 0.0) &&
                near(prepared.preparedMassAudit.lumpedNodalMassKg[original] +
                     prepared.preparedMassAudit.lumpedNodalMassKg[node],
                     prepared.sourceMassAudit.lumpedNodalMassKg[original]) &&
                !near(prepared.preparedMassAudit.lumpedNodalMassKg[node],
                      prepared.sourceMassAudit.lumpedNodalMassKg[original], 1.0e-16),
                "split node mass was copied instead of recomputed from split tets");
    }

    std::size_t taggedBoundaries = 0u;
    for (const auto& boundary : prepared.boundaryFaces) {
        if (boundary.boundarySlot == 23) {
            ++taggedBoundaries;
            require(boundary.tetId == 0u && boundary.faceSlot == 2u &&
                    boundary.materialIndex == kStarchMaterialGlue &&
                    boundary.referenceAreaM2 > 0.0,
                    "cut-slot boundary tag or local face slot did not survive split");
            for (std::size_t node : boundary.outwardNodeIds)
                require(node >= source.referenceNodesM.size(),
                        "glue-side boundary face did not follow the duplicated glue nodes");
            const auto& face = boundary.outwardNodeIds;
            const FractureVec3 normal = fractureCross(
                prepared.referenceNodesM[face[1]] - prepared.referenceNodesM[face[0]],
                prepared.referenceNodesM[face[2]] - prepared.referenceNodesM[face[0]]);
            const FractureVec3 center = (
                prepared.referenceNodesM[face[0]] +
                prepared.referenceNodesM[face[1]] +
                prepared.referenceNodesM[face[2]]) / 3.0;
            const FractureVec3 opposite = prepared.referenceNodesM[
                prepared.tetrahedra[boundary.tetId].nodeIds[boundary.faceSlot]];
            require(fractureDot(normal, opposite - center) < 0.0,
                    "boundary face node order is not outward from its tetrahedron");
        }
    }
    require(taggedBoundaries == 1u,
            "boundary/cut slot tag was lost or duplicated");
}

void testTiedAndUnbondedControls() {
    const StarchInterfaceMesh source = sampleMesh();
    const auto tied = prepareStarchInterfaceMesh(
        source, StarchInterfacePreparationMode::TiedControl);
    validateFaceCorrespondence(tied);
    require(tied.referenceNodesM.size() == source.referenceNodesM.size() &&
            tied.duplicatedGlueNodeCount == 0u,
            "tied reference control unexpectedly duplicated nodes");
    for (const auto& face : tied.pairedFaces) {
        require(!face.hasIndependentNodePairs &&
                face.paperNodeIds == face.glueNodeIds,
                "tied reference control does not retain shared interface node IDs");
    }

    const auto unbonded = prepareStarchInterfaceMesh(
        source, StarchInterfacePreparationMode::UnbondedControl);
    validateFaceCorrespondence(unbonded);
    require(unbonded.duplicatedGlueNodeCount == 4u,
            "unbonded topology control did not split interface nodes");
    for (const auto& face : unbonded.pairedFaces)
        require(face.hasIndependentNodePairs && !face.constitutiveLawInstalled &&
                face.topologyMode == StarchInterfacePreparationMode::UnbondedControl,
                "unbonded control accidentally claims a native interface law");
}

void testInvalidAndNonmanifoldInput() {
    auto nonmanifold = sampleMesh();
    nonmanifold.referenceNodesM.push_back({0.0, 0.0, -2.0});
    nonmanifold.tetrahedra.push_back(
        tet({0u, 1u, 2u, 7u}, kStarchMaterialLiner, 840.0));
    expectInvalid([&] { (void)prepareStarchInterfaceMesh(nonmanifold); },
                  "three tets incident on one face were admitted");

    StarchInterfaceMesh directPaper;
    directPaper.referenceNodesM = {
        {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}, {0.0, 0.0, -1.0}};
    directPaper.tetrahedra = {
        tet({0u, 1u, 2u, 3u}, kStarchMaterialLiner, 840.0),
        tet({0u, 1u, 2u, 4u}, kStarchMaterialMedium, 720.0)};
    expectInvalid([&] { (void)prepareStarchInterfaceMesh(directPaper); },
                  "direct liner/medium contact without a finite glue tet was admitted");

    auto taggedInterior = sampleMesh();
    taggedInterior.tetrahedra[0].boundarySlotByFace[3] = 44;
    expectInvalid([&] { (void)prepareStarchInterfaceMesh(taggedInterior); },
                  "boundary/cut slot tag on an interior interface was admitted");

    auto zeroDensity = sampleMesh();
    zeroDensity.tetrahedra[1].densityKgPerM3 = 0.0;
    expectInvalid([&] { (void)prepareStarchInterfaceMesh(zeroDensity); },
                  "missing glue density was silently defaulted");

    auto reflectedFrame = sampleMesh();
    reflectedFrame.tetrahedra[0].materialFrameToReference[0] = -1.0;
    expectInvalid([&] { (void)prepareStarchInterfaceMesh(reflectedFrame); },
                  "improper source material frame was admitted");

    auto invalidLabel = sampleMesh();
    invalidLabel.tetrahedra[0].materialIndex = 3u;
    expectInvalid([&] { (void)prepareStarchInterfaceMesh(invalidLabel); },
                  "unsupported source material label was admitted");
}

} // namespace

int main() {
    try {
        testSplitMassFramesAndBoundarySlots();
        testTiedAndUnbondedControls();
        testInvalidAndNonmanifoldInput();
        const auto prepared = prepareStarchInterfaceMesh(sampleMesh());
        std::cout << "cardboard_starch_interface_mesh_cpu_passed "
                  << "nodes=" << prepared.referenceNodesM.size()
                  << " tets=" << prepared.tetrahedra.size()
                  << " duplicated_glue_nodes=" << prepared.duplicatedGlueNodeCount
                  << " paired_faces=" << prepared.pairedFaces.size()
                  << " boundary_faces=" << prepared.boundaryFaces.size()
                  << " total_mass_kg=" << prepared.preparedMassAudit.totalMassKg
                  << " constitutive_law_installed=false\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
