#pragma once

// Measurement-only support for deriving an impactor footprint from the
// accepted exterior roof.  It never edits the FEM mesh or accepted history.

#include "numi/matter/matter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace numi::cardboard {

// "20% larger footprint" denotes area, preserving the measured roof aspect ratio.
inline constexpr double kShippingImpactorFootprintAreaRatio = 1.2;
inline double shippingImpactorPlanScale() {
    return std::sqrt(kShippingImpactorFootprintAreaRatio);
}

struct ShippingOuterRoofBounds {
    double minimumX = 0.0;
    double maximumX = 0.0;
    double minimumZ = 0.0;
    double maximumZ = 0.0;
    double topY = 0.0;
    std::uint32_t exteriorFaceCount = 0u;
    bool valid = false;
};

struct ShippingOuterRoofMeasurement {
    ShippingOuterRoofBounds bounds{};
    bool measuredFromAcceptedFixtureFreeSettledCheckpoint = false;
    std::string checkpointSha256;
    std::uint64_t sourcePhysicsFingerprint = 0u;
    std::uint64_t compiledWorldFingerprint = 0u;
    std::string sourceBlankGeometrySha256;
    std::string sourceShippingBindingSha256;
    std::string acceptedCheckpointFile;
    std::string sourceMeasurementFileSha256;
    std::array<double, 3> compiledImpactorHalfExtentsM{};
    bool checkpointFileHashVerified = false;
    bool acceptedRecordSchemaVerified = false;
    std::string coordinateUnits = "m";
    std::string method =
        "accepted FEM nodes on compiled exterior triangles with |normal_y|>=0.75 and centroid within 2.5 board calipers of the accepted maximum-y roof; x/z projected bounds";
};

struct ShippingImpactorFootprintPlan {
    double fullLengthX = 0.0;
    double fullWidthZ = 0.0;
    double centerX = 0.0;
    double centerZ = 0.0;
    double quarterWidthOffsetZ = 0.0;
    double roofY = 0.0;
    std::string sourceCheckpointSha256;
    std::uint64_t sourcePhysicsFingerprint = 0u;
    std::uint64_t compiledWorldFingerprint = 0u;
    std::string basis = "1.2 times accepted outer-roof area; aspect ratio preserved";
};

inline std::array<double, 3> shippingAuthoredNominalImpactorHalfExtents(
    const double nominalOuterLengthXM,
    const double nominalOuterWidthZM,
    const double halfHeightM) {
    if (!(std::isfinite(nominalOuterLengthXM) && nominalOuterLengthXM > 0.0 &&
          std::isfinite(nominalOuterWidthZM) && nominalOuterWidthZM > 0.0 &&
          std::isfinite(halfHeightM) && halfHeightM > 0.0))
        throw std::invalid_argument(
            "authored shipping impactor dimensions must be finite and positive");
    // Full plan area is 1.2 times the nominal outer footprint. Both planar
    // dimensions scale by sqrt(1.2). These are an
    // authored geometry default, not evidence that a folded roof has this size.
    return {0.5 * shippingImpactorPlanScale() * nominalOuterLengthXM, halfHeightM,
            0.5 * shippingImpactorPlanScale() * nominalOuterWidthZM};
}

inline bool shippingImpactorHalfExtentsMatchMeasuredRoof(
    const ShippingOuterRoofMeasurement& measurement,
    const std::array<double, 3>& halfExtentsM) {
    const auto& bounds = measurement.bounds;
    if (!bounds.valid || !(std::isfinite(bounds.maximumX) &&
            std::isfinite(bounds.minimumX) && std::isfinite(bounds.maximumZ) &&
            std::isfinite(bounds.minimumZ) && bounds.maximumX > bounds.minimumX &&
            bounds.maximumZ > bounds.minimumZ))
        return false;
    for (const double extent : halfExtentsM)
        if (!(std::isfinite(extent) && extent > 0.0)) return false;
    // Match the exact FP32 extents that Matter cooks, rather than accepting a
    // loose geometric tolerance that could hide a different collision box.
    return static_cast<float>(halfExtentsM[0]) ==
               static_cast<float>(0.5 * shippingImpactorPlanScale() * (bounds.maximumX - bounds.minimumX)) &&
           static_cast<float>(halfExtentsM[2]) ==
               static_cast<float>(0.5 * shippingImpactorPlanScale() * (bounds.maximumZ - bounds.minimumZ));
}

inline std::string shippingImpactorProxyMismatchReason(
    const NMRigidProxyGPU& proxy,
    const std::uint32_t expectedBodyIndex,
    const std::array<double, 3>& halfExtentsM) {
    if (expectedBodyIndex == NM_INVALID_INDEX ||
        proxy.shapeKind != NM_RIGID_BOX ||
        proxy.bodyIndex != expectedBodyIndex ||
        proxy.sceneBodyIndex != expectedBodyIndex)
        return "compiled impactor proxy is not the expected body-backed BOX";
    if ((proxy.flags & NM_RIGID_DYNAMIC) == 0u ||
        (proxy.flags & (NM_RIGID_ARTICULATED |
                        NM_RIGID_PRESCRIBED_TRANSLATION)) != 0u)
        return "compiled impactor proxy has unexpected motion flags";
    const std::array<float, 3> expectedExtent{
        static_cast<float>(halfExtentsM[0]),
        static_cast<float>(halfExtentsM[1]),
        static_cast<float>(halfExtentsM[2])};
    if (proxy.localExtent.x != expectedExtent[0] ||
        proxy.localExtent.y != expectedExtent[1] ||
        proxy.localExtent.z != expectedExtent[2] || proxy.localExtent.w != 0.0f)
        return "compiled impactor BOX half-extents differ from the explicit branch geometry";
    if (proxy.localCenterAndRadius.x != 0.0f ||
        proxy.localCenterAndRadius.y != 0.0f ||
        proxy.localCenterAndRadius.z != 0.0f ||
        proxy.localCenterAndRadius.w != 0.0f)
        return "compiled impactor BOX has an unexpected local-center offset";
    if (proxy.localOrientation.x != 0.0f ||
        proxy.localOrientation.y != 0.0f ||
        proxy.localOrientation.z != 0.0f ||
        proxy.localOrientation.w != 1.0f)
        return "compiled impactor BOX local orientation is not identity";
    return {};
}

inline bool shippingRoofSha256Valid(const std::string& digest);

inline bool shippingRoofMeasurementMatchesCurrentSource(
    const ShippingOuterRoofMeasurement& measurement,
    const std::string& expectedBlankGeometrySha256,
    const std::string& expectedShippingBindingSha256) {
    return measurement.acceptedRecordSchemaVerified &&
        measurement.checkpointFileHashVerified &&
        measurement.measuredFromAcceptedFixtureFreeSettledCheckpoint &&
        measurement.coordinateUnits == "m" && measurement.bounds.valid &&
        shippingRoofSha256Valid(measurement.checkpointSha256) &&
        shippingRoofSha256Valid(measurement.sourceBlankGeometrySha256) &&
        shippingRoofSha256Valid(measurement.sourceShippingBindingSha256) &&
        shippingRoofSha256Valid(measurement.sourceMeasurementFileSha256) &&
        measurement.sourceBlankGeometrySha256 == expectedBlankGeometrySha256 &&
        measurement.sourceShippingBindingSha256 == expectedShippingBindingSha256 &&
        measurement.sourcePhysicsFingerprint != 0u &&
        measurement.compiledWorldFingerprint != 0u;
}

inline bool shippingRoofSha256Valid(const std::string& digest) {
    return digest.size() == 64u && std::all_of(digest.begin(), digest.end(),
        [](const unsigned char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        });
}

inline ShippingOuterRoofBounds measureShippingOuterRoofBounds(
    const std::span<const NMFEMNodeStateGPU> acceptedNodes,
    const std::uint32_t nodeCount,
    const std::span<const NMTetrahedronGPU> tetrahedra,
    const std::span<const NMFEMSurfaceFaceGPU> surfaceFaces,
    const double boardCaliperM,
    const std::uint32_t objectIndex = 0u,
    const std::uint32_t environment = 0u) {
    const std::uint64_t environmentNodeCount =
        (static_cast<std::uint64_t>(environment) + 1u) * nodeCount;
    if (nodeCount == 0u || !(std::isfinite(boardCaliperM) && boardCaliperM > 0.0) ||
        environmentNodeCount > acceptedNodes.size())
        throw std::invalid_argument("outer-roof measurement inputs are invalid or truncated");

    const std::size_t nodeBase = static_cast<std::size_t>(environment) * nodeCount;
    std::vector<std::uint8_t> objectNode(nodeCount, 0u);
    for (const auto& tet : tetrahedra) {
        if (tet.identity.y != objectIndex) continue;
        const std::array<std::uint32_t, 4> ids{
            tet.nodes.x, tet.nodes.y, tet.nodes.z, tet.nodes.w};
        for (const std::uint32_t id : ids) {
            if (id >= nodeCount)
                throw std::invalid_argument("outer-roof tetrahedron references an invalid node");
            objectNode[id] = 1u;
        }
    }
    double maximumObjectY = -std::numeric_limits<double>::infinity();
    for (std::uint32_t id = 0u; id < nodeCount; ++id) {
        if (!objectNode[id]) continue;
        const double y = acceptedNodes[nodeBase + id].positionAndMass.y;
        if (!std::isfinite(y))
            throw std::invalid_argument("outer-roof accepted node is nonfinite");
        maximumObjectY = std::max(maximumObjectY, y);
    }
    if (!std::isfinite(maximumObjectY))
        throw std::invalid_argument("outer-roof object has no accepted tetrahedral nodes");

    const double topBandM = std::max(2.5 * boardCaliperM, 1.0e-5);
    ShippingOuterRoofBounds result;
    result.minimumX = std::numeric_limits<double>::infinity();
    result.maximumX = -std::numeric_limits<double>::infinity();
    result.minimumZ = std::numeric_limits<double>::infinity();
    result.maximumZ = -std::numeric_limits<double>::infinity();
    result.topY = -std::numeric_limits<double>::infinity();

    for (const auto& face : surfaceFaces) {
        if (face.adjacency.z != objectIndex || face.adjacency.y != NM_INVALID_INDEX ||
            face.adjacency.x >= tetrahedra.size() || face.sides.x >= 4u)
            continue;
        const auto& tet = tetrahedra[face.adjacency.x];
        if (tet.identity.y != objectIndex) continue;
        const std::array<std::uint32_t, 4> tetNodes{
            tet.nodes.x, tet.nodes.y, tet.nodes.z, tet.nodes.w};
        std::array<std::uint32_t, 3> faceNodes{};
        std::size_t cursor = 0u;
        for (std::uint32_t corner = 0u; corner < 4u; ++corner)
            if (corner != face.sides.x) faceNodes[cursor++] = tetNodes[corner];
        if (cursor != faceNodes.size())
            throw std::invalid_argument("outer-roof face corner mapping is invalid");
        std::array<std::array<double, 3>, 3> p{};
        for (std::size_t vertex = 0u; vertex < 3u; ++vertex) {
            if (faceNodes[vertex] >= nodeCount)
                throw std::invalid_argument("outer-roof face references an invalid node");
            const auto& position = acceptedNodes[nodeBase + faceNodes[vertex]].positionAndMass;
            p[vertex] = {position.x, position.y, position.z};
            for (const double component : p[vertex])
                if (!std::isfinite(component))
                    throw std::invalid_argument("outer-roof face vertex is nonfinite");
        }
        const std::array<double, 3> a{
            p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]};
        const std::array<double, 3> b{
            p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]};
        const std::array<double, 3> normal{
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
        const double normalLength = std::sqrt(
            normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
        if (!(std::isfinite(normalLength) && normalLength > 1.0e-15)) continue;
        const double centroidY = (p[0][1] + p[1][1] + p[2][1]) / 3.0;
        if (std::abs(normal[1]) / normalLength < 0.75 ||
            centroidY < maximumObjectY - topBandM)
            continue;
        for (const auto& vertex : p) {
            result.minimumX = std::min(result.minimumX, vertex[0]);
            result.maximumX = std::max(result.maximumX, vertex[0]);
            result.minimumZ = std::min(result.minimumZ, vertex[2]);
            result.maximumZ = std::max(result.maximumZ, vertex[2]);
            result.topY = std::max(result.topY, vertex[1]);
        }
        ++result.exteriorFaceCount;
    }
    result.valid = result.exteriorFaceCount > 0u &&
        std::isfinite(result.minimumX) && std::isfinite(result.maximumX) &&
        std::isfinite(result.minimumZ) && std::isfinite(result.maximumZ) &&
        std::isfinite(result.topY) && result.maximumX > result.minimumX &&
        result.maximumZ > result.minimumZ;
    if (!result.valid)
        throw std::invalid_argument(
            "accepted geometry has no measurable upward-facing exterior roof triangles");
    return result;
}

inline ShippingImpactorFootprintPlan makeShippingImpactorFootprintPlan(
    const ShippingOuterRoofMeasurement& measurement,
    const std::string& expectedCheckpointSha256,
    const std::uint64_t expectedPhysicsFingerprint,
    const std::uint64_t expectedCompiledWorldFingerprint) {
    const auto& bounds = measurement.bounds;
    if (!measurement.measuredFromAcceptedFixtureFreeSettledCheckpoint ||
        measurement.coordinateUnits != "m" || !bounds.valid ||
        !shippingRoofSha256Valid(measurement.checkpointSha256) ||
        measurement.checkpointSha256 != expectedCheckpointSha256 ||
        measurement.sourcePhysicsFingerprint == 0u ||
        measurement.sourcePhysicsFingerprint != expectedPhysicsFingerprint ||
        measurement.compiledWorldFingerprint == 0u ||
        measurement.compiledWorldFingerprint != expectedCompiledWorldFingerprint ||
        !(std::isfinite(bounds.minimumX) && std::isfinite(bounds.maximumX) &&
          std::isfinite(bounds.minimumZ) && std::isfinite(bounds.maximumZ) &&
          std::isfinite(bounds.topY) && bounds.maximumX > bounds.minimumX &&
          bounds.maximumZ > bounds.minimumZ && bounds.exteriorFaceCount > 0u))
        throw std::invalid_argument(
            "shipping impact footprint requires measured outer roof bounds bound to the accepted settled checkpoint and compiled geometry");
    ShippingImpactorFootprintPlan result;
    const double spanX = bounds.maximumX - bounds.minimumX;
    const double spanZ = bounds.maximumZ - bounds.minimumZ;
    result.fullLengthX = shippingImpactorPlanScale() * spanX;
    result.fullWidthZ = shippingImpactorPlanScale() * spanZ;
    result.centerX = 0.5 * (bounds.minimumX + bounds.maximumX);
    result.centerZ = 0.5 * (bounds.minimumZ + bounds.maximumZ);
    result.quarterWidthOffsetZ = 0.25 * spanZ;
    result.roofY = bounds.topY;
    result.sourceCheckpointSha256 = measurement.checkpointSha256;
    result.sourcePhysicsFingerprint = measurement.sourcePhysicsFingerprint;
    result.compiledWorldFingerprint = measurement.compiledWorldFingerprint;
    if (!(std::isfinite(result.fullLengthX) && result.fullLengthX > spanX &&
          std::isfinite(result.fullWidthZ) && result.fullWidthZ > spanZ &&
          std::isfinite(result.centerX) && std::isfinite(result.centerZ) &&
          std::isfinite(result.roofY)))
        throw std::invalid_argument("shipping impact footprint plan is nonfinite");
    return result;
}

inline std::array<double, 3> shippingImpactStartCenter(
    const ShippingImpactorFootprintPlan& footprint,
    const double halfHeightM,
    const double dropHeightM,
    const double offsetFractionOfRoofWidth) {
    if (!(std::isfinite(halfHeightM) && halfHeightM > 0.0 &&
          std::isfinite(dropHeightM) && dropHeightM > 0.0 &&
          std::isfinite(offsetFractionOfRoofWidth) &&
          std::abs(offsetFractionOfRoofWidth) <= 0.25))
        throw std::invalid_argument("shipping impact pose inputs are invalid");
    return {footprint.centerX,
            footprint.roofY + dropHeightM + halfHeightM,
            footprint.centerZ + offsetFractionOfRoofWidth *
                (4.0 * footprint.quarterWidthOffsetZ)};
}

} // namespace numi::cardboard
