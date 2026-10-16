#include "cardboard_box_blank.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <iomanip>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace numi::cardboard;
using Face = std::array<std::uint32_t, 3>;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

Face canonicalFace(Face face) {
    std::sort(face.begin(), face.end());
    return face;
}

double sixVolume(const BoxBlankPoint& a,
                 const BoxBlankPoint& b,
                 const BoxBlankPoint& c,
                 const BoxBlankPoint& d) {
    const std::array<double, 3> ab{b.x-a.x, b.y-a.y, b.z-a.z};
    const std::array<double, 3> ac{c.x-a.x, c.y-a.y, c.z-a.z};
    const std::array<double, 3> ad{d.x-a.x, d.y-a.y, d.z-a.z};
    return ab[0]*(ac[1]*ad[2]-ac[2]*ad[1]) -
        ab[1]*(ac[0]*ad[2]-ac[2]*ad[0]) +
        ab[2]*(ac[0]*ad[1]-ac[1]*ad[0]);
}

struct FaceUse {
    std::vector<std::uint32_t> tetrahedra;
};

void auditBlank(const FEFCO0201Blank& blank,
                const ConformingGlueMesh& section) {
    require(!blank.points.empty() && !blank.tetrahedra.empty(),
            "blank geometry is empty");
    require(blank.panels.size() == 4u,
            "FEFCO 0201 must contain four body panels");
    require(blank.scoreLines.size() == 12u,
            "blank must expose four wall and eight flap score centerlines");
    require(blank.slots.size() == 6u,
            "blank must contain three internal slots at each end");
    require(blank.assembly.executionState ==
                std::string("targets-authored-not-executed"),
            "assembly schedule must not claim physical execution");
    for (const auto& score : blank.scoreLines)
        require(!score.physicallyScored,
                "metadata-only score line was marked as physical damage");
    for (const auto& action : blank.assembly.actions)
        require(!action.executed,
                "fold/seal target was incorrectly marked as executed");

    const double expectedPerimeter = blank.config.manufacturerJointWidth +
        2.0 * (blank.config.panelLength + blank.config.panelWidth);
    const double expectedHeight = blank.config.wallHeight + blank.config.panelWidth;
    require(std::abs(blank.perimeterSpan - expectedPerimeter) < 1.0e-13 &&
                std::abs(blank.blankHeight - expectedHeight) < 1.0e-13,
            "FEFCO blank envelope formula is incorrect");
    require(std::abs(blank.topFlapDepth - 0.5 * blank.config.panelWidth) < 1.0e-13,
            "RSC flap depth is not half the panel width");
    require(blank.panels[0].longWall && !blank.panels[1].longWall &&
                blank.panels[2].longWall && !blank.panels[3].longWall,
            "panel sequence is not L/W/L/W");
    const std::array<double, 4> expectedPanelWidths{
        blank.config.panelLength, blank.config.panelWidth,
        blank.config.panelLength, blank.config.panelWidth,
    };
    for (std::size_t panel = 0u; panel < blank.panels.size(); ++panel) {
        require(std::abs((blank.panels[panel].xEnd - blank.panels[panel].xStart) -
                         expectedPanelWidths[panel]) < 1.0e-13 &&
                    std::abs((blank.panels[panel].yEnd - blank.panels[panel].yStart) -
                             blank.config.wallHeight) < 1.0e-13,
                "score-to-score panel dimensions are incorrect");
    }
    require(std::abs(blank.boardCaliper - 4.210e-3) < 1.0e-13,
            "example did not preserve the explicit SI caliper input");

    double minX = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();
    double minZ = std::numeric_limits<double>::infinity();
    double maxZ = -std::numeric_limits<double>::infinity();
    for (const auto& point : blank.points) {
        minX = std::min(minX, point.x);
        maxX = std::max(maxX, point.x);
        minY = std::min(minY, point.y);
        maxY = std::max(maxY, point.y);
        minZ = std::min(minZ, point.z);
        maxZ = std::max(maxZ, point.z);
    }
    require(std::abs(minX) < 1.0e-13 &&
                std::abs(maxX - blank.perimeterSpan) < 1.0e-13 &&
                std::abs(minY) < 1.0e-13 &&
                std::abs(maxY - blank.blankHeight) < 1.0e-13 &&
                std::abs(minZ) < 1.0e-13 &&
                std::abs(maxZ - blank.boardCaliper) < 1.0e-13,
            "blank node envelope does not match the SI geometry inputs");

    std::map<Face, FaceUse> faces;
    std::vector<std::vector<std::uint32_t>> adjacency(blank.tetrahedra.size());
    std::array<double, 3> volumeByMaterial{};
    for (std::uint32_t index = 0u; index < blank.tetrahedra.size(); ++index) {
        const auto& tetrahedron = blank.tetrahedra[index];
        std::set<std::uint32_t> uniqueNodes(tetrahedron.nodes.begin(),
                                            tetrahedron.nodes.end());
        require(uniqueNodes.size() == 4u,
                "tetrahedron has repeated nodes");
        for (const auto node : tetrahedron.nodes)
            require(node < blank.points.size(),
                    "tetrahedron node is outside point arena");
        const double volume = sixVolume(
            blank.points[tetrahedron.nodes[0]],
            blank.points[tetrahedron.nodes[1]],
            blank.points[tetrahedron.nodes[2]],
            blank.points[tetrahedron.nodes[3]]) / 6.0;
        require(std::isfinite(volume) && volume > 0.0,
                "tetrahedron is not positively oriented with finite volume");
        const auto material = static_cast<std::size_t>(tetrahedron.material);
        require(material < volumeByMaterial.size(),
                "unknown laminate material role");
        volumeByMaterial[material] += volume;

        const auto& n = tetrahedron.nodes;
        for (const Face face : {
                 canonicalFace({n[1], n[2], n[3]}),
                 canonicalFace({n[0], n[2], n[3]}),
                 canonicalFace({n[0], n[1], n[3]}),
                 canonicalFace({n[0], n[1], n[2]}),
             })
            faces[face].tetrahedra.push_back(index);
    }
    require(volumeByMaterial == blank.volumeByMaterial,
            "reported material volumes differ from the tetrahedron sum");

    std::uint64_t boundaryFaceCount = 0u;
    for (const auto& [face, use] : faces) {
        (void)face;
        require(use.tetrahedra.size() <= 2u,
                "mesh has a nonmanifold tetrahedral face");
        if (use.tetrahedra.size() == 1u) {
            ++boundaryFaceCount;
            continue;
        }
        require(use.tetrahedra.size() == 2u,
                "mesh contains an unused internal face");
        const auto a = static_cast<std::size_t>(
            blank.tetrahedra[use.tetrahedra[0]].material);
        const auto b = static_cast<std::size_t>(
            blank.tetrahedra[use.tetrahedra[1]].material);
        require(!(a == 0u && b == 1u) && !(a == 1u && b == 0u),
                "liner and medium directly share a face outside adhesive");
    }
    require(boundaryFaceCount > 0u,
            "blank has no exposed physical surfaces");

    // Prove full material-layer connectivity through shared tetra faces.
    std::vector<bool> reached(blank.tetrahedra.size(), false);
    std::queue<std::uint32_t> pending;
    pending.push(0u);
    reached[0] = true;
    for (const auto& [face, use] : faces) {
        (void)face;
        if (use.tetrahedra.size() != 2u) continue;
        const auto a = use.tetrahedra[0];
        const auto b = use.tetrahedra[1];
        adjacency[a].push_back(b);
        adjacency[b].push_back(a);
    }
    while (!pending.empty()) {
        const auto current = pending.front();
        pending.pop();
        for (const auto next : adjacency[current]) {
            if (reached[next]) continue;
            reached[next] = true;
            pending.push(next);
        }
    }
    require(std::ranges::all_of(reached, [](const bool value) { return value; }),
            "liner/core/glue blank is disconnected across its laminate or slots");

    for (std::size_t material = 0u; material < volumeByMaterial.size(); ++material) {
        const double expected = blank.plannedVolumeByMaterial[material];
        const double tolerance = 2.0e-9 * std::max(expected, 1.0e-18);
        if (std::abs(volumeByMaterial[material] - expected) > tolerance) {
            std::ostringstream detail;
            detail << std::setprecision(17)
                   << "tetrahedral material volume disagrees with per-slab blank integration: material="
                   << material << " actual=" << volumeByMaterial[material]
                   << " expected=" << expected
                   << " ratio=" << volumeByMaterial[material] / expected
                   << " section_area=" << section.crossSectionAreaByMaterial[material];
            throw std::runtime_error(detail.str());
        }
    }

    for (const auto& tetrahedron : blank.tetrahedra) {
        const auto& a = blank.points[tetrahedron.nodes[0]];
        const auto& b = blank.points[tetrahedron.nodes[1]];
        const auto& c = blank.points[tetrahedron.nodes[2]];
        const auto& d = blank.points[tetrahedron.nodes[3]];
        const double x = 0.25 * (a.x + b.x + c.x + d.x);
        const double y = 0.25 * (a.y + b.y + c.y + d.y);
        const bool inFlapBand = y < blank.bottomScoreY - 1.0e-12 ||
            y > blank.topScoreY + 1.0e-12;
        if (!inFlapBand) continue;
        for (const auto& slot : blank.slots) {
            if (x > slot.xLeft + 1.0e-12 && x < slot.xRight - 1.0e-12 &&
                y > slot.yStart && y < slot.yEnd)
                throw std::runtime_error("tetrahedron occupies a removed slot strip");
        }
    }

    for (const auto& slot : blank.slots) {
        require(std::abs((slot.xRight - slot.xLeft) - blank.config.slotKerf) < 1.0e-13,
                "slot kerf does not match explicit geometry input");
        const double expectedDepth = blank.topFlapDepth;
        require(std::abs((slot.yEnd - slot.yStart) - expectedDepth) < 1.0e-13,
                "slot depth differs from half-width flap geometry");
    }
    require(blank.scoreLines[0].kind == BoxScoreKind::manufacturerJointRoot &&
                blank.assembly.actions.size() == 15u,
            "box assembly protocol omitted a wall, flap, joint, or seal target");
    for (std::uint32_t index = 0u; index < blank.assembly.actions.size(); ++index) {
        const auto& action = blank.assembly.actions[index];
        if (index > 0u && action.kind == BoxAssemblyAction::Kind::bondManufacturerJoint)
            require(blank.assembly.actions[index - 1u].kind ==
                        BoxAssemblyAction::Kind::erectWallPanel,
                    "manufacturer joint must follow wall erection targets");

        switch (action.kind) {
            case BoxAssemblyAction::Kind::erectWallPanel: {
                require(action.feature.kind == BoxAssemblyFeatureKind::panel &&
                            action.feature.index < blank.panels.size(),
                        "wall action must reference a valid blank panel");
                require(action.hingeScoreLineIndex < blank.scoreLines.size(),
                        "wall action must reference its score-line hinge");
                const auto& hinge = blank.scoreLines[action.hingeScoreLineIndex];
                const std::uint32_t expectedPanel =
                    hinge.kind == BoxScoreKind::manufacturerJointRoot
                        ? 0u : hinge.panelIndex + 1u;
                require((hinge.kind == BoxScoreKind::manufacturerJointRoot ||
                         hinge.kind == BoxScoreKind::bodyCorner) &&
                            action.feature.index == expectedPanel,
                        "wall panel and hinge score-line identities disagree");
                break;
            }
            case BoxAssemblyAction::Kind::bondManufacturerJoint:
                require(action.feature.kind == BoxAssemblyFeatureKind::manufacturerJoint &&
                            action.feature.index == 0u &&
                            action.hingeScoreLineIndex == kNoBoxScoreLineIndex,
                        "joint-bond action must name the single tab region");
                break;
            case BoxAssemblyAction::Kind::foldFlap: {
                require(action.feature.kind == BoxAssemblyFeatureKind::scoreLine &&
                            action.feature.index < blank.scoreLines.size() &&
                            action.hingeScoreLineIndex == kNoBoxScoreLineIndex,
                        "flap action must reference a score-line feature");
                const auto& hinge = blank.scoreLines[action.feature.index];
                require(hinge.kind == BoxScoreKind::bottomFlap ||
                            hinge.kind == BoxScoreKind::topFlap,
                        "flap action references a non-flap score line");
                require((action.targetAngleDegrees > 0.0 &&
                         hinge.kind == BoxScoreKind::bottomFlap) ||
                            (action.targetAngleDegrees < 0.0 &&
                             hinge.kind == BoxScoreKind::topFlap),
                        "flap fold target sign disagrees with its end score");
                break;
            }
            case BoxAssemblyAction::Kind::closeEndWithTapeOrAdhesive:
                require(action.feature.kind == BoxAssemblyFeatureKind::boxEnd &&
                            action.feature.index <= 1u &&
                            action.hingeScoreLineIndex == kNoBoxScoreLineIndex,
                        "end-seal action must use 0=bottom or 1=top");
                break;
        }
    }

    const std::array<std::uint32_t, 4> closureFlapOrder{1u, 3u, 0u, 2u};
    for (std::uint32_t panel = 0u; panel < 4u; ++panel) {
        const auto& wallAction = blank.assembly.actions[panel];
        require(wallAction.kind == BoxAssemblyAction::Kind::erectWallPanel &&
                    wallAction.feature.index == panel &&
                    wallAction.hingeScoreLineIndex == panel,
                "wall assembly references must be panel IDs paired with their hinge IDs");
    }
    require(blank.assembly.actions[4].kind ==
                BoxAssemblyAction::Kind::bondManufacturerJoint &&
                blank.assembly.actions[4].feature.kind ==
                    BoxAssemblyFeatureKind::manufacturerJoint &&
                blank.assembly.actions[4].feature.index == 0u,
            "joint action must use its typed singleton tab-region reference");
    for (std::uint32_t end = 0u; end < 2u; ++end) {
        const bool top = end == 1u;
        const std::uint32_t blockStart = top ? 10u : 5u;
        const BoxScoreKind expectedScoreKind = top
            ? BoxScoreKind::topFlap : BoxScoreKind::bottomFlap;
        for (std::uint32_t order = 0u; order < closureFlapOrder.size(); ++order) {
            const std::uint32_t panel = closureFlapOrder[order];
            const auto& action = blank.assembly.actions[blockStart + order];
            require(action.kind == BoxAssemblyAction::Kind::foldFlap &&
                        action.feature.kind == BoxAssemblyFeatureKind::scoreLine &&
                        action.feature.index < blank.scoreLines.size(),
                    "flap target must use a typed score-line reference");
            const auto& score = blank.scoreLines[action.feature.index];
            require(score.kind == expectedScoreKind && score.panelIndex == panel,
                    "flap target score-line index must resolve to the intended panel and end");
        }
        const auto& seal = blank.assembly.actions[top ? 14u : 9u];
        require(seal.kind == BoxAssemblyAction::Kind::closeEndWithTapeOrAdhesive &&
                    seal.feature.kind == BoxAssemblyFeatureKind::boxEnd &&
                    seal.feature.index == end,
                "end target index must be 0=bottom or 1=top");
    }
}

void expectInvalid(const FEFCO0201BlankConfig& config,
                   const ConformingGlueMesh& section,
                   const std::string& label) {
    bool rejected = false;
    try {
        (void)buildFEFCO0201Blank(config, section);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "invalid blank config was accepted: " + label);
}

void run() {
    constexpr double pitch = 7.9e-3;
    GlueMeshConfig sectionConfig;
    sectionConfig.length = 15.0 * pitch;
    sectionConfig.pitch = pitch;
    sectionConfig.caliper = 4.210e-3;
    sectionConfig.linerThickness = 0.277e-3;
    sectionConfig.mediumThickness = 0.191e-3;
    sectionConfig.glueMinimumGap = 0.100e-3;
    sectionConfig.bondWidth = 0.800e-3;
    sectionConfig.upperGlueMinimumGap = 0.030e-3;
    sectionConfig.upperBondWidth = 0.600e-3;
    sectionConfig.nxPerPitch = 4u;
    sectionConfig.thicknessSlices = 1u;
    const auto section = buildConformingGlueCrossSection(sectionConfig);

    // Small reusable geometry example: panel score dimensions 4p x 3p,
    // wall score height 4p, and a one-pitch glue tab. These box dimensions and
    // the half-millimetre slot kerf are explicit test inputs, not published
    // test specimens or source-calibrated dimensions.
    FEFCO0201BlankConfig config;
    config.panelLength = 4.0 * pitch;
    config.panelWidth = 3.0 * pitch;
    config.wallHeight = 4.0 * pitch;
    config.manufacturerJointWidth = 1.0 * pitch;
    config.slotKerf = 0.5e-3;
    config.subdivisionsPerBand = 2u;
    const auto blank = buildFEFCO0201Blank(config, section);
    auditBlank(blank, section);

    const double expectedPerimeter = 15.0 * pitch;
    require(std::abs(blank.perimeterSpan - expectedPerimeter) < 1.0e-13,
            "example perimeter does not span fifteen source pitches");
    expectInvalid({0.0, config.panelWidth, config.wallHeight,
                   config.manufacturerJointWidth, config.slotKerf, 1u},
                  section, "zero panel length");
    expectInvalid({config.panelLength, config.panelWidth, config.wallHeight,
                   config.manufacturerJointWidth, 0.0, 1u},
                  section, "zero slot width");
    expectInvalid({config.panelLength, config.panelWidth, config.wallHeight,
                   config.manufacturerJointWidth, 0.5 * config.panelWidth, 1u},
                  section, "slot wider than panel allowance");
    auto mismatched = section;
    mismatched.config.length *= 0.5;
    expectInvalid(config, mismatched, "section does not span the box perimeter");

    std::cout << "fefco0201_blank_geometry_passed"
              << " nodes=" << blank.points.size()
              << " tetrahedra=" << blank.tetrahedra.size()
              << " score_lines=" << blank.scoreLines.size()
              << " slots=" << blank.slots.size()
              << " assembly_actions=" << blank.assembly.actions.size()
              << " folds_executed=false"
              << " score_damage_authored=false\n";
}

} // namespace

int main() {
    try {
        run();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
