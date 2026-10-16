#pragma once

// Topology preparation for finite starch-layer/paper cohesive interfaces.
// This utility creates paired faces; it does not implement a traction law.

#include "cardboard_fracture_regularization.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace numi::cardboard {

inline constexpr std::uint8_t kStarchMaterialLiner = 0u;
inline constexpr std::uint8_t kStarchMaterialMedium = 1u;
inline constexpr std::uint8_t kStarchMaterialGlue = 2u;

enum class StarchInterfacePreparationMode {
    // Duplicate glue-side nodes and emit matching paper/glue triangle records
    // for later consumption by a native cohesive constitutive law.
    PreparedForNativeLaw,
    // Keep shared nodes as a topology-only tied reference control.
    TiedControl,
    // Duplicate glue-side nodes, but apply no traction law in the control.
    UnbondedControl,
};

enum class StarchPaperRegion : std::uint8_t {
    Liner = kStarchMaterialLiner,
    Medium = kStarchMaterialMedium,
};

struct StarchInterfaceTet {
    std::array<std::size_t, 4> nodeIds{};
    std::uint8_t materialIndex = kStarchMaterialLiner;
    // Density is supplied per tet. The helper has no default glue density.
    double densityKgPerM3 = 0.0;
    // Row-major proper rotation from the tet material frame to reference axes.
    std::array<double, 9> materialFrameToReference{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0};
    // Slot is the local opposite-vertex index (0..3). -1 means untagged.
    // A nonnegative tag is valid only on an exterior/cut face.
    std::array<std::int32_t, 4> boundarySlotByFace{-1, -1, -1, -1};
};

struct StarchInterfaceMesh {
    std::vector<FractureVec3> referenceNodesM;
    std::vector<StarchInterfaceTet> tetrahedra;
};

struct StarchMeshMassAudit {
    std::array<double, 3> referenceVolumeByMaterialM3{};
    std::array<double, 3> massByMaterialKg{};
    double totalMassKg = 0.0;
    // Sum of all entries of the standard P1 tetrahedron consistent mass matrix.
    // For constant density this equals rho*V per tet and totalMassKg globally.
    double consistentMassMatrixTotalKg = 0.0;
    // Standard row-sum lumping, recomputed from this mesh's tetrahedra.
    std::vector<double> lumpedNodalMassKg;
};

struct StarchPairedFaceRecord {
    std::size_t paperTetId = 0u;
    std::size_t glueTetId = 0u;
    std::uint8_t paperMaterialIndex = kStarchMaterialLiner;
    StarchPaperRegion paperRegion = StarchPaperRegion::Liner;
    // Local face slots are opposite-vertex indices in the source/output tets.
    std::uint8_t paperFaceSlot = 0u;
    std::uint8_t glueFaceSlot = 0u;
    // Entries share the same matching index and reference coordinate. In a
    // split mode, glueNodeIds are duplicated nodes; paperNodeIds stay original.
    std::array<std::size_t, 3> paperNodeIds{};
    std::array<std::size_t, 3> glueNodeIds{};
    std::array<FractureVec3, 3> matchingReferenceCoordinatesM{};
    FractureVec3 paperOutwardUnitNormal{};
    FractureVec3 glueOutwardUnitNormal{};
    double referenceAreaM2 = 0.0;
    StarchInterfacePreparationMode topologyMode =
        StarchInterfacePreparationMode::PreparedForNativeLaw;
    bool hasIndependentNodePairs = true;
    bool constitutiveLawInstalled = false;
};

struct StarchBoundaryFaceRecord {
    std::size_t tetId = 0u;
    std::uint8_t faceSlot = 0u;
    std::uint8_t materialIndex = kStarchMaterialLiner;
    std::int32_t boundarySlot = -1;
    std::array<std::size_t, 3> outwardNodeIds{};
    double referenceAreaM2 = 0.0;
};

struct StarchPreparedInterfaceMesh {
    std::vector<FractureVec3> referenceNodesM;
    std::vector<StarchInterfaceTet> tetrahedra;
    std::vector<std::size_t> sourceNodeForPreparedNode;
    std::vector<StarchPairedFaceRecord> pairedFaces;
    std::vector<StarchBoundaryFaceRecord> boundaryFaces;
    StarchMeshMassAudit sourceMassAudit;
    StarchMeshMassAudit preparedMassAudit;
    StarchInterfacePreparationMode topologyMode =
        StarchInterfacePreparationMode::PreparedForNativeLaw;
    std::size_t sourceNodeCount = 0u;
    std::size_t duplicatedGlueNodeCount = 0u;
    std::array<double, 3> sourceMaterialVolumeM3{};
    std::array<double, 3> preparedMaterialVolumeM3{};
};

namespace starch_mesh_detail {

using FaceKey = std::array<std::size_t, 3>;

struct FaceRef {
    FaceKey key{};
    std::size_t tetId = 0u;
    std::uint8_t slot = 0u;
};

struct InterfaceSeed {
    FaceKey originalNodeIds{};
    std::size_t paperTetId = 0u;
    std::size_t glueTetId = 0u;
    std::uint8_t paperSlot = 0u;
    std::uint8_t glueSlot = 0u;
};

struct BoundarySeed {
    std::size_t tetId = 0u;
    std::uint8_t slot = 0u;
    std::int32_t boundarySlot = -1;
};

inline double tetVolumeM3(const std::vector<FractureVec3>& nodes,
                          const StarchInterfaceTet& tet) {
    const auto& ids = tet.nodeIds;
    const FractureVec3 a = nodes[ids[0]];
    const FractureVec3 b = nodes[ids[1]];
    const FractureVec3 c = nodes[ids[2]];
    const FractureVec3 d = nodes[ids[3]];
    return std::abs(fractureDot(b - a, fractureCross(c - a, d - a))) / 6.0;
}

inline std::array<std::size_t, 3> faceNodeOrder(
    const StarchInterfaceTet& tet, std::uint8_t oppositeSlot) {
    std::array<std::size_t, 3> result{};
    std::size_t write = 0u;
    for (std::size_t i = 0; i < 4; ++i) {
        if (i != oppositeSlot) result[write++] = tet.nodeIds[i];
    }
    return result;
}

inline std::array<std::size_t, 3> outwardFaceNodeOrder(
    const std::vector<FractureVec3>& nodes,
    const StarchInterfaceTet& tet,
    std::uint8_t oppositeSlot) {
    auto result = faceNodeOrder(tet, oppositeSlot);
    const FractureVec3 p0 = nodes[result[0]];
    const FractureVec3 p1 = nodes[result[1]];
    const FractureVec3 p2 = nodes[result[2]];
    const FractureVec3 faceNormal = fractureCross(p1 - p0, p2 - p0);
    const FractureVec3 faceCenter = (p0 + p1 + p2) / 3.0;
    if (fractureDot(faceNormal, nodes[tet.nodeIds[oppositeSlot]] - faceCenter) > 0.0)
        std::swap(result[1], result[2]);
    return result;
}

inline FaceKey faceKey(const StarchInterfaceTet& tet, std::uint8_t oppositeSlot) {
    auto result = faceNodeOrder(tet, oppositeSlot);
    std::sort(result.begin(), result.end());
    return result;
}

inline FractureVec3 outwardUnitNormal(
    const std::vector<FractureVec3>& nodes,
    const StarchInterfaceTet& tet,
    std::uint8_t slot) {
    const auto face = faceNodeOrder(tet, slot);
    const FractureVec3 p0 = nodes[face[0]];
    const FractureVec3 p1 = nodes[face[1]];
    const FractureVec3 p2 = nodes[face[2]];
    FractureVec3 normal = fractureCross(p1 - p0, p2 - p0);
    const double normalLength = fractureNorm(normal);
    if (!(std::isfinite(normalLength) && normalLength > 0.0))
        throw std::invalid_argument("tetrahedron has a degenerate face");
    normal = normal / normalLength;
    const FractureVec3 faceCenter = (p0 + p1 + p2) / 3.0;
    const FractureVec3 opposite = nodes[tet.nodeIds[slot]];
    if (fractureDot(normal, opposite - faceCenter) > 0.0)
        normal = normal * -1.0;
    return normal;
}

inline double triangleAreaM2(const std::vector<FractureVec3>& nodes,
                             const std::array<std::size_t, 3>& ids) {
    return 0.5 * fractureNorm(fractureCross(nodes[ids[1]] - nodes[ids[0]],
                                             nodes[ids[2]] - nodes[ids[0]]));
}

inline void validateInput(const StarchInterfaceMesh& mesh) {
    if (mesh.referenceNodesM.empty() || mesh.tetrahedra.empty())
        throw std::invalid_argument("starch interface preparation requires nodes and tetrahedra");
    for (FractureVec3 node : mesh.referenceNodesM) {
        if (!fractureFinite(node))
            throw std::invalid_argument("starch interface mesh contains a nonfinite node");
    }
    for (const auto& tet : mesh.tetrahedra) {
        if (tet.materialIndex > kStarchMaterialGlue)
            throw std::invalid_argument("starch interface mesh has unsupported material index");
        if (!(std::isfinite(tet.densityKgPerM3) && tet.densityKgPerM3 > 0.0))
            throw std::invalid_argument("each region tet requires a finite positive source density");
        fracture_detail::validateProperRotation(
            FractureRotation3{tet.materialFrameToReference});
        auto ids = tet.nodeIds;
        std::sort(ids.begin(), ids.end());
        if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
            throw std::invalid_argument("tetrahedron repeats a node index");
        for (std::size_t node : tet.nodeIds) {
            if (node >= mesh.referenceNodesM.size())
                throw std::invalid_argument("tetrahedron node index is out of range");
        }
        for (std::int32_t tag : tet.boundarySlotByFace) {
            if (tag < -1)
                throw std::invalid_argument("boundary slot tags must be -1 or nonnegative");
        }
        const double volume = tetVolumeM3(mesh.referenceNodesM, tet);
        double edgeScale = 0.0;
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = i + 1; j < 4; ++j) {
                edgeScale = std::max(edgeScale, fractureNorm(
                    mesh.referenceNodesM[tet.nodeIds[j]] -
                    mesh.referenceNodesM[tet.nodeIds[i]]));
            }
        }
        const double volumeFloor = 128.0 * std::numeric_limits<double>::epsilon() *
                                  edgeScale * edgeScale * edgeScale;
        if (!(std::isfinite(volume) && edgeScale > 0.0 && volume > volumeFloor))
            throw std::invalid_argument("starch interface mesh has a degenerate tetrahedron");
    }
}

inline StarchMeshMassAudit computeMassAudit(
    const std::vector<FractureVec3>& nodes,
    const std::vector<StarchInterfaceTet>& tetrahedra) {
    StarchMeshMassAudit audit;
    audit.lumpedNodalMassKg.assign(nodes.size(), 0.0);
    for (const auto& tet : tetrahedra) {
        const double volume = tetVolumeM3(nodes, tet);
        const double mass = volume * tet.densityKgPerM3;
        audit.referenceVolumeByMaterialM3[tet.materialIndex] += volume;
        audit.massByMaterialKg[tet.materialIndex] += mass;
        audit.totalMassKg += mass;
        // The sum of all sixteen entries of the scalar P1 consistent element
        // mass matrix is rho*V; lumping assigns its row sums rho*V/4.
        audit.consistentMassMatrixTotalKg += mass;
        for (std::size_t node : tet.nodeIds)
            audit.lumpedNodalMassKg[node] += 0.25 * mass;
    }
    return audit;
}

inline bool sameNumber(double a, double b) {
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    return std::isfinite(a) && std::isfinite(b) &&
           std::abs(a - b) <= 2.0e-12 * scale;
}

} // namespace starch_mesh_detail

// Splits glue-side nodes only on actual topological paper/glue face adjacency.
// It does not infer contact from coincident coordinates with different node IDs.
inline StarchPreparedInterfaceMesh prepareStarchInterfaceMesh(
    const StarchInterfaceMesh& source,
    StarchInterfacePreparationMode mode =
        StarchInterfacePreparationMode::PreparedForNativeLaw) {
    using namespace starch_mesh_detail;
    if (mode != StarchInterfacePreparationMode::PreparedForNativeLaw &&
        mode != StarchInterfacePreparationMode::TiedControl &&
        mode != StarchInterfacePreparationMode::UnbondedControl)
        throw std::invalid_argument("unknown starch interface preparation mode");
    validateInput(source);

    std::vector<FaceRef> faces;
    faces.reserve(source.tetrahedra.size() * 4u);
    for (std::size_t tetId = 0; tetId < source.tetrahedra.size(); ++tetId) {
        for (std::uint8_t slot = 0; slot < 4u; ++slot)
            faces.push_back({faceKey(source.tetrahedra[tetId], slot), tetId, slot});
    }
    std::sort(faces.begin(), faces.end(), [](const FaceRef& a, const FaceRef& b) {
        if (a.key != b.key) return a.key < b.key;
        if (a.tetId != b.tetId) return a.tetId < b.tetId;
        return a.slot < b.slot;
    });

    std::vector<InterfaceSeed> interfaces;
    std::vector<BoundarySeed> boundaries;
    for (std::size_t begin = 0; begin < faces.size();) {
        std::size_t end = begin + 1u;
        while (end < faces.size() && faces[end].key == faces[begin].key) ++end;
        const std::size_t incidence = end - begin;
        if (incidence > 2u)
            throw std::invalid_argument("nonmanifold tetrahedral face has more than two incident tets");
        if (incidence == 1u) {
            const FaceRef& face = faces[begin];
            boundaries.push_back({face.tetId, face.slot,
                source.tetrahedra[face.tetId].boundarySlotByFace[face.slot]});
        } else {
            const FaceRef& left = faces[begin];
            const FaceRef& right = faces[begin + 1u];
            const auto& leftTet = source.tetrahedra[left.tetId];
            const auto& rightTet = source.tetrahedra[right.tetId];
            if (leftTet.boundarySlotByFace[left.slot] >= 0 ||
                rightTet.boundarySlotByFace[right.slot] >= 0)
                throw std::invalid_argument("boundary/cut slot tag appears on an interior face");
            if (leftTet.materialIndex != rightTet.materialIndex) {
                const bool leftPaper = leftTet.materialIndex == kStarchMaterialLiner ||
                                       leftTet.materialIndex == kStarchMaterialMedium;
                const bool rightPaper = rightTet.materialIndex == kStarchMaterialLiner ||
                                        rightTet.materialIndex == kStarchMaterialMedium;
                if (leftPaper == rightPaper)
                    throw std::invalid_argument(
                        "direct liner/medium adjacency is unsupported; finite glue tet is required");
                const FaceRef& paper = leftPaper ? left : right;
                const FaceRef& glue = leftPaper ? right : left;
                interfaces.push_back({paper.key, paper.tetId, glue.tetId,
                                      paper.slot, glue.slot});
            }
        }
        begin = end;
    }

    StarchPreparedInterfaceMesh result;
    result.topologyMode = mode;
    result.sourceNodeCount = source.referenceNodesM.size();
    result.referenceNodesM = source.referenceNodesM;
    result.tetrahedra = source.tetrahedra;
    result.sourceNodeForPreparedNode.resize(source.referenceNodesM.size());
    for (std::size_t i = 0; i < source.referenceNodesM.size(); ++i)
        result.sourceNodeForPreparedNode[i] = i;

    std::map<std::size_t, std::size_t> glueSideClone;
    const bool splitGlueSide = mode != StarchInterfacePreparationMode::TiedControl;
    if (splitGlueSide) {
        std::vector<std::size_t> nodesToDuplicate;
        nodesToDuplicate.reserve(interfaces.size() * 3u);
        for (const auto& interface : interfaces)
            nodesToDuplicate.insert(nodesToDuplicate.end(),
                                    interface.originalNodeIds.begin(),
                                    interface.originalNodeIds.end());
        std::sort(nodesToDuplicate.begin(), nodesToDuplicate.end());
        nodesToDuplicate.erase(std::unique(nodesToDuplicate.begin(),
                                           nodesToDuplicate.end()),
                               nodesToDuplicate.end());
        for (std::size_t sourceNode : nodesToDuplicate) {
            const std::size_t duplicate = result.referenceNodesM.size();
            result.referenceNodesM.push_back(source.referenceNodesM[sourceNode]);
            result.sourceNodeForPreparedNode.push_back(sourceNode);
            glueSideClone.emplace(sourceNode, duplicate);
        }
        for (auto& tet : result.tetrahedra) {
            if (tet.materialIndex != kStarchMaterialGlue) continue;
            for (std::size_t& node : tet.nodeIds) {
                const auto found = glueSideClone.find(node);
                if (found != glueSideClone.end()) node = found->second;
            }
        }
    }
    result.duplicatedGlueNodeCount = result.referenceNodesM.size() -
                                     result.sourceNodeCount;

    result.pairedFaces.reserve(interfaces.size());
    for (const auto& interface : interfaces) {
        const auto& paperTet = result.tetrahedra[interface.paperTetId];
        const auto& glueTet = result.tetrahedra[interface.glueTetId];
        StarchPairedFaceRecord record;
        record.paperTetId = interface.paperTetId;
        record.glueTetId = interface.glueTetId;
        record.paperMaterialIndex = paperTet.materialIndex;
        record.paperRegion = paperTet.materialIndex == kStarchMaterialLiner ?
            StarchPaperRegion::Liner : StarchPaperRegion::Medium;
        record.paperFaceSlot = interface.paperSlot;
        record.glueFaceSlot = interface.glueSlot;
        record.topologyMode = mode;
        record.hasIndependentNodePairs = splitGlueSide;
        record.constitutiveLawInstalled = false;
        for (std::size_t i = 0; i < 3u; ++i) {
            const std::size_t sourceNode = interface.originalNodeIds[i];
            record.paperNodeIds[i] = sourceNode;
            record.glueNodeIds[i] = splitGlueSide ? glueSideClone.at(sourceNode) : sourceNode;
            record.matchingReferenceCoordinatesM[i] = source.referenceNodesM[sourceNode];
            if (fractureNorm(result.referenceNodesM[record.paperNodeIds[i]] -
                             result.referenceNodesM[record.glueNodeIds[i]]) != 0.0)
                throw std::logic_error("prepared paper/glue node pair lost reference coincidence");
        }
        record.paperOutwardUnitNormal = outwardUnitNormal(
            result.referenceNodesM, paperTet, interface.paperSlot);
        record.glueOutwardUnitNormal = outwardUnitNormal(
            result.referenceNodesM, glueTet, interface.glueSlot);
        if (fractureDot(record.paperOutwardUnitNormal,
                        record.glueOutwardUnitNormal) > -1.0 + 1.0e-9)
            throw std::invalid_argument(
                "shared paper/glue tets are not on opposite sides of their common face");
        record.referenceAreaM2 = triangleAreaM2(
            result.referenceNodesM, record.paperNodeIds);
        if (!(std::isfinite(record.referenceAreaM2) && record.referenceAreaM2 > 0.0))
            throw std::invalid_argument("paper/glue interface has invalid reference area");
        result.pairedFaces.push_back(record);
    }

    result.boundaryFaces.reserve(boundaries.size());
    for (const auto& boundary : boundaries) {
        const auto& tet = result.tetrahedra[boundary.tetId];
        StarchBoundaryFaceRecord record;
        record.tetId = boundary.tetId;
        record.faceSlot = boundary.slot;
        record.materialIndex = tet.materialIndex;
        record.boundarySlot = boundary.boundarySlot;
        record.outwardNodeIds = outwardFaceNodeOrder(
            result.referenceNodesM, tet, boundary.slot);
        record.referenceAreaM2 = triangleAreaM2(result.referenceNodesM,
                                                 record.outwardNodeIds);
        if (!(std::isfinite(record.referenceAreaM2) && record.referenceAreaM2 > 0.0))
            throw std::invalid_argument("boundary/cut face has invalid reference area");
        result.boundaryFaces.push_back(record);
    }

    result.sourceMassAudit = computeMassAudit(source.referenceNodesM, source.tetrahedra);
    result.preparedMassAudit = computeMassAudit(result.referenceNodesM, result.tetrahedra);
    result.sourceMaterialVolumeM3 = result.sourceMassAudit.referenceVolumeByMaterialM3;
    result.preparedMaterialVolumeM3 = result.preparedMassAudit.referenceVolumeByMaterialM3;
    if (!sameNumber(result.sourceMassAudit.totalMassKg,
                    result.preparedMassAudit.totalMassKg) ||
        !sameNumber(result.sourceMassAudit.consistentMassMatrixTotalKg,
                    result.preparedMassAudit.consistentMassMatrixTotalKg))
        throw std::logic_error("node splitting changed total consistent/lumped mass");
    for (std::size_t material = 0; material < 3u; ++material) {
        if (!sameNumber(result.sourceMaterialVolumeM3[material],
                        result.preparedMaterialVolumeM3[material]) ||
            !sameNumber(result.sourceMassAudit.massByMaterialKg[material],
                        result.preparedMassAudit.massByMaterialKg[material]))
            throw std::logic_error("node splitting changed a material-region volume or mass");
    }
    const double sourceLumpedSum = std::accumulate(
        result.sourceMassAudit.lumpedNodalMassKg.begin(),
        result.sourceMassAudit.lumpedNodalMassKg.end(), 0.0);
    const double preparedLumpedSum = std::accumulate(
        result.preparedMassAudit.lumpedNodalMassKg.begin(),
        result.preparedMassAudit.lumpedNodalMassKg.end(), 0.0);
    if (!sameNumber(sourceLumpedSum, result.sourceMassAudit.totalMassKg) ||
        !sameNumber(preparedLumpedSum, result.preparedMassAudit.totalMassKg))
        throw std::logic_error("recomputed lumped nodal masses do not sum to tet mass");
    return result;
}

} // namespace numi::cardboard
