#include "cardboard_glue_mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using numi::cardboard::CellMaterial;
using numi::cardboard::ConformingGlueMesh;
using numi::cardboard::GlueMeshConfig;
using numi::cardboard::GlueMeshPoint;
using numi::cardboard::buildConformingGlueCrossSection;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(const bool condition, const std::string& message) {
    if (!condition) fail(message);
}

double twiceArea(const GlueMeshPoint& a,
                 const GlueMeshPoint& b,
                 const GlueMeshPoint& c) {
    return (b.x - a.x) * (c.z - a.z) -
        (b.z - a.z) * (c.x - a.x);
}

struct EdgeUse {
    std::array<std::uint32_t, 3> counts{};
    std::uint32_t total = 0u;
};

struct MeshSummary {
    std::array<std::uint64_t, 3> triangleCounts{};
    std::array<double, 3> areas{};
    std::array<std::uint64_t, 3> sharedEdgeCounts{};
};

MeshSummary audit(const ConformingGlueMesh& mesh) {
    MeshSummary result;
    std::map<std::pair<std::uint32_t, std::uint32_t>, EdgeUse> edges;
    std::vector<std::uint32_t> glueTriangleCoverage(mesh.triangles.size(), 0u);
    std::vector<std::array<bool, 3>> pointMaterials(mesh.points.size());

    for (std::size_t footprintIndex = 0u;
         footprintIndex < mesh.glueFootprints.size(); ++footprintIndex) {
        const auto& footprint = mesh.glueFootprints[footprintIndex];
        require(footprint.triangleCount > 0u,
                "glue footprint has no triangles");
        const std::uint64_t end = footprint.firstTriangle +
            footprint.triangleCount;
        require(end <= mesh.triangles.size(),
                "glue footprint triangle range exceeds mesh");
        for (std::uint64_t index = footprint.firstTriangle; index < end; ++index) {
            require(mesh.triangles[static_cast<std::size_t>(index)].material ==
                        CellMaterial::glue,
                    "footprint range includes a non-glue triangle");
            ++glueTriangleCoverage[static_cast<std::size_t>(index)];
        }
        require(footprint.actualHorizontalWidth > 0.0 &&
                    footprint.actualHorizontalWidth <=
                        footprint.requestedHorizontalWidth + 1.0e-14,
                "reported horizontal glue width is invalid");
        require(footprint.xRight > footprint.xLeft,
                "glue footprint has nonpositive horizontal extent");
        require(footprint.crossSectionArea > 0.0 &&
                    footprint.volumePerUnitBoardWidth > 0.0,
                "glue footprint has nonpositive area or volume metadata");
        require(std::abs(footprint.crossSectionArea -
                         footprint.volumePerUnitBoardWidth) <= 1.0e-12,
                "per-unit-width volume must equal the section area");
    }

    for (std::size_t index = 0u; index < mesh.triangles.size(); ++index) {
        const auto& triangle = mesh.triangles[index];
        const auto materialIndex = static_cast<std::size_t>(triangle.material);
        require(materialIndex < result.areas.size(),
                "triangle has an unknown material index");
        if (triangle.material == CellMaterial::glue)
            require(glueTriangleCoverage[index] == 1u,
                    "glue triangle is missing or multiply assigned to a footprint");

        const auto a = triangle.points[0];
        const auto b = triangle.points[1];
        const auto c = triangle.points[2];
        require(a < mesh.points.size() && b < mesh.points.size() &&
                    c < mesh.points.size() && a != b && b != c && c != a,
                "triangle has an invalid or repeated point index");
        pointMaterials[a][materialIndex] = true;
        pointMaterials[b][materialIndex] = true;
        pointMaterials[c][materialIndex] = true;
        const double area2 = twiceArea(mesh.points[a], mesh.points[b], mesh.points[c]);
        require(std::isfinite(area2) && area2 > 0.0,
                "triangle must be finite, positive-area, and CCW");
        require(std::isfinite(triangle.frameAngle),
                "triangle has a nonfinite material-frame angle");
        result.areas[materialIndex] += 0.5 * area2;
        ++result.triangleCounts[materialIndex];

        const std::array<std::uint32_t, 3> nodes{a, b, c};
        for (std::size_t edge = 0u; edge < 3u; ++edge) {
            const auto p = nodes[edge];
            const auto q = nodes[(edge + 1u) % 3u];
            const auto key = std::minmax(p, q);
            auto& use = edges[{key.first, key.second}];
            ++use.counts[materialIndex];
            ++use.total;
        }
    }

    for (std::size_t material = 0u; material < result.areas.size(); ++material) {
        require(result.triangleCounts[material] > 0u,
                "one of the liner, medium, or glue regions is empty");
        const double reported = mesh.crossSectionAreaByMaterial[material];
        require(std::abs(result.areas[material] - reported) <=
                    1.0e-11 * std::max(1.0, std::abs(reported)),
                "reported cross-section material area disagrees with triangle sum");
    }

    for (const auto& [edge, use] : edges) {
        const std::uint32_t materialKinds =
            static_cast<std::uint32_t>(use.counts[0] > 0u) +
            static_cast<std::uint32_t>(use.counts[1] > 0u) +
            static_cast<std::uint32_t>(use.counts[2] > 0u);
        if (materialKinds < 2u) continue;
        require(use.total == 2u,
                "a material interface edge is nonmanifold or split inconsistently");
        require(!(use.counts[0] > 0u && use.counts[1] > 0u),
                "liner and medium directly share an edge outside adhesive");
        if (use.counts[0] > 0u && use.counts[2] > 0u)
            ++result.sharedEdgeCounts[0];
        if (use.counts[1] > 0u && use.counts[2] > 0u)
            ++result.sharedEdgeCounts[1];
    }
    require(result.sharedEdgeCounts[0] > 0u && result.sharedEdgeCounts[1] > 0u,
            "glue must share complete edges with both liner and medium");

    // Detect hanging interface nodes geometrically, including a point placed
    // in the interior of an edge whose matching-material edge was not split.
    for (const auto& [edge, use] : edges) {
        (void)use;
        const auto [first, second] = edge;
        const auto& a = mesh.points[first];
        const auto& b = mesh.points[second];
        const double dx = b.x - a.x;
        const double dz = b.z - a.z;
        const double length = std::hypot(dx, dz);
        if (!(length > 0.0)) continue;
        for (std::size_t pointIndex = 0u;
             pointIndex < mesh.points.size(); ++pointIndex) {
            if (pointIndex == first || pointIndex == second) continue;
            bool usedByOtherMaterial = false;
            for (std::size_t material = 0u; material < 3u; ++material) {
                if (pointMaterials[pointIndex][material]) {
                    usedByOtherMaterial = true;
                    break;
                }
            }
            if (!usedByOtherMaterial) continue;
            const auto& point = mesh.points[pointIndex];
            const double t = ((point.x - a.x) * dx +
                              (point.z - a.z) * dz) / (length * length);
            if (t <= 1.0e-9 || t >= 1.0 - 1.0e-9) continue;
            const double distanceNumerator =
                std::abs(dx * (point.z - a.z) - dz * (point.x - a.x));
            require(distanceNumerator > 1.0e-13 * length,
                    "mesh has a hanging point inside an unsplit edge");
        }
    }

    for (const double area : result.areas)
        require(area > 0.0 && std::isfinite(area),
                "material section area is not finite and positive");
    const double occupiedArea = result.areas[0] + result.areas[1] + result.areas[2];
    require(occupiedArea < mesh.config.length * mesh.config.caliper,
            "geometry leaves no unbonded void space in the design envelope");
    const double exactLinerArea =
        2.0 * mesh.config.length * mesh.config.linerThickness;
    require(std::abs(result.areas[0] - exactLinerArea) <=
                1.0e-10 * exactLinerArea,
            "two flat liners do not preserve their authored area");
    return result;
}

void expectInvalid(const GlueMeshConfig& config, const char* caseName) {
    bool rejected = false;
    try {
        (void)buildConformingGlueCrossSection(config);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, std::string("invalid config was accepted: ") + caseName);
}

} // namespace

int main() {
    try {
        GlueMeshConfig config;
        config.length = 18.0e-3;
        config.pitch = 9.0e-3;
        config.caliper = 5.0e-3;
        config.linerThickness = 0.25e-3;
        config.mediumThickness = 0.25e-3;
        config.glueMinimumGap = 0.05e-3;
        config.bondWidth = 1.2e-3;
        config.nxPerPitch = 24u;
        config.thicknessSlices = 2u;

        const ConformingGlueMesh mesh =
            buildConformingGlueCrossSection(config);
        const MeshSummary summary = audit(mesh);
        const std::uint32_t expectedFootprints = 2u * 2u + 1u;
        require(mesh.glueFootprints.size() == expectedFootprints,
                "two-pitch section must have two crests and three valleys");
        require(mesh.mediumCenterlineRise > 0.0 &&
                    mesh.mediumAmplitude > 0.0 &&
                    mesh.centerlineTakeUpRatio > 1.0 &&
                    mesh.maximumCurvature > 0.0,
                "medium profile geometry metadata is invalid");
        require(mesh.glueFootprints.front().actualHorizontalWidth <
                    config.bondWidth,
                "end-clipped valley bond should report its smaller actual width");
        for (const auto& footprint : mesh.glueFootprints) {
            const bool edgeClipped = footprint.xLeft == 0.0 ||
                footprint.xRight == config.length;
            if (!edgeClipped)
                require(std::abs(footprint.actualHorizontalWidth - config.bondWidth) <
                            1.0e-13,
                        "interior bond must realize its requested horizontal width");
        }

        GlueMeshConfig asymmetric = config;
        asymmetric.glueMinimumGap = 0.10e-3;
        asymmetric.bondWidth = 0.80e-3;
        asymmetric.upperGlueMinimumGap = 0.03e-3;
        asymmetric.upperBondWidth = 0.60e-3;
        const ConformingGlueMesh doubleBack =
            buildConformingGlueCrossSection(asymmetric);
        const MeshSummary asymmetricSummary = audit(doubleBack);
        (void)asymmetricSummary;
        for (const auto& footprint : doubleBack.glueFootprints) {
            const double expectedWidth = footprint.crest
                ? asymmetric.upperBondWidth : asymmetric.bondWidth;
            if (footprint.xLeft > 0.0 && footprint.xRight < asymmetric.length)
                require(std::abs(footprint.actualHorizontalWidth - expectedWidth) <
                            1.0e-13,
                        "asymmetric double-back glue width was not preserved");
        }
        require(std::abs(doubleBack.mediumCenterlineRise -
                         (asymmetric.caliper - 2.0 * asymmetric.linerThickness -
                          asymmetric.mediumThickness -
                          asymmetric.glueMinimumGap -
                          asymmetric.upperGlueMinimumGap)) < 1.0e-14,
                "asymmetric upper/lower glue gaps do not set medium rise correctly");

        GlueMeshConfig bad = config;
        bad.glueMinimumGap = 0.0;
        expectInvalid(bad, "zero minimum gap");
        bad = config;
        bad.bondWidth = 0.5 * bad.pitch;
        expectInvalid(bad, "overlapping footprints");
        bad = config;
        bad.caliper = 2.0 * bad.linerThickness + bad.mediumThickness +
            2.0 * bad.glueMinimumGap;
        expectInvalid(bad, "zero medium rise");

        std::cout << std::setprecision(9)
                  << "PASS cardboard conforming finite-glue section\n"
                  << "profile=" << mesh.profileDescription << '\n'
                  << "points=" << mesh.points.size()
                  << " triangles=" << mesh.triangles.size()
                  << " liner_triangles=" << summary.triangleCounts[0]
                  << " medium_triangles=" << summary.triangleCounts[1]
                  << " glue_triangles=" << summary.triangleCounts[2] << '\n'
                  << "liner_area_m2=" << summary.areas[0]
                  << " medium_area_m2=" << summary.areas[1]
                  << " glue_area_m2=" << summary.areas[2] << '\n'
                  << "glue_liner_shared_edges=" << summary.sharedEdgeCounts[0]
                  << " glue_medium_shared_edges=" << summary.sharedEdgeCounts[1]
                  << "footprints=" << mesh.glueFootprints.size()
                  << " takeup_ratio=" << mesh.centerlineTakeUpRatio << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "FAIL cardboard conforming finite-glue section: "
                  << exception.what() << '\n';
        return 1;
    }
}
