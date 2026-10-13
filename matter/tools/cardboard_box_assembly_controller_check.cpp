#include "cardboard_box_assembly_controller.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace numi::cardboard;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

ConformingGlueMesh syntheticSection() {
    GlueMeshConfig config;
    config.length = 15.0 * 7.9e-3;
    config.pitch = 7.9e-3;
    config.caliper = 4.210e-3;
    config.linerThickness = 0.277e-3;
    config.mediumThickness = 0.191e-3;
    config.glueMinimumGap = 0.100e-3;
    config.bondWidth = 0.800e-3;
    config.upperGlueMinimumGap = 0.030e-3;
    config.upperBondWidth = 0.600e-3;
    config.nxPerPitch = 4u;
    config.thicknessSlices = 1u;
    return buildConformingGlueCrossSection(config);
}

FEFCO0201Blank buildBlank() {
    FEFCO0201BlankConfig config;
    config.panelLength = 4.0 * 7.9e-3;
    config.panelWidth = 3.0 * 7.9e-3;
    config.wallHeight = 4.0 * 7.9e-3;
    // Match the native probe aggregate {4p, 3p, 4p, p, 0.2p, ny}:
    // the tab is one pitch wide and 0.2p is the idealized slot kerf.
    config.manufacturerJointWidth = 7.9e-3;
    config.slotKerf = 0.2 * 7.9e-3;
    config.subdivisionsPerBand = 2u;
    return buildFEFCO0201Blank(config, syntheticSection());
}

NMFEMNodeStateGPU nodeAt(const BoxBlankPoint& point) {
    NMFEMNodeStateGPU node{};
    node.positionAndMass = {static_cast<float>(point.x),
        static_cast<float>(point.y), static_cast<float>(point.z), 1.0f};
    node.restAndFixed = {static_cast<float>(point.x),
        static_cast<float>(point.y), static_cast<float>(point.z), 0.0f};
    node.velocityAndInverseMass.w = 1.0f;
    return node;
}

void rotateBottomFlapAboutX(const FEFCO0201Blank& blank,
                            const BoxBlankPoint& rest,
                            NMFEMNodeStateGPU& node,
                            const double radians) {
    if (rest.x < blank.panels[0].xStart ||
        rest.x > blank.panels[0].xEnd || rest.y > blank.bottomScoreY)
        return;
    const double y = rest.y - blank.bottomScoreY;
    const double zMid = 0.5 * blank.boardCaliper;
    const double z = rest.z - zMid;
    const double c = std::cos(radians), s = std::sin(radians);
    node.positionAndMass.y = static_cast<float>(blank.bottomScoreY + c * y - s * z);
    node.positionAndMass.z = static_cast<float>(zMid + s * y + c * z);
}

double xAxisOnlyAngleDegrees(
    const std::vector<BoxBlankPoint>& restNodes,
    const std::vector<NMFEMNodeStateGPU>& acceptedNodes,
    const double xStart,
    const double xEnd,
    const double yStart,
    const double yEnd) {
    double meanRestX = 0.0, meanX = 0.0, meanZ = 0.0;
    std::uint32_t count = 0u;
    for (std::uint32_t node = 0u; node < restNodes.size(); ++node) {
        const auto& rest = restNodes[node];
        if (rest.x <= xStart || rest.x >= xEnd ||
            rest.y < yStart || rest.y > yEnd)
            continue;
        const BoxVec3 current = boxAcceptedPosition(acceptedNodes[node]);
        meanRestX += rest.x;
        meanX += current.x;
        meanZ += current.z;
        ++count;
    }
    if (count < 3u) return std::numeric_limits<double>::quiet_NaN();
    meanRestX /= count;
    meanX /= count;
    meanZ /= count;
    double covarianceX = 0.0, covarianceZ = 0.0;
    for (std::uint32_t node = 0u; node < restNodes.size(); ++node) {
        const auto& rest = restNodes[node];
        if (rest.x <= xStart || rest.x >= xEnd ||
            rest.y < yStart || rest.y > yEnd)
            continue;
        const BoxVec3 current = boxAcceptedPosition(acceptedNodes[node]);
        const double dx = rest.x - meanRestX;
        covarianceX += dx * (current.x - meanX);
        covarianceZ += dx * (current.z - meanZ);
    }
    return std::atan2(-covarianceZ, covarianceX) *
        180.0 / 3.14159265358979323846;
}

std::vector<std::uint32_t> exposedNodes(const FEFCO0201Blank& blank) {
    std::map<std::array<std::uint32_t, 3>, std::uint32_t> faceCounts;
    for (const auto& tetrahedron : blank.tetrahedra) {
        const auto& n = tetrahedron.nodes;
        for (auto face : std::array<std::array<std::uint32_t, 3>, 4>{{
                 {n[0], n[1], n[2]}, {n[0], n[1], n[3]},
                 {n[0], n[2], n[3]}, {n[1], n[2], n[3]}}}) {
            std::sort(face.begin(), face.end());
            ++faceCounts[face];
        }
    }
    std::set<std::uint32_t> nodes;
    for (const auto& [face, count] : faceCounts)
        if (count == 1u) nodes.insert(face.begin(), face.end());
    return {nodes.begin(), nodes.end()};
}

void run() {
    const FEFCO0201Blank blank = buildBlank();
    const auto tooling = makeBoxScoreTooling(
        blank, 0.75e-3, 2.0e-3, 0.010e-3, 0.010e-3, 3.0e-3, 2.0e-3);
    require(tooling.proxies.size() == 13u &&
                tooling.scoreBodyIndex == 0u && tooling.supportBodyIndex == 1u,
            "scoring die must expose 12 capsules on one body plus one support");
    for (std::uint32_t score = 0u; score < 12u; ++score) {
        const auto& proxy = tooling.proxies[score];
        const auto& line = blank.scoreLines[score];
        require(proxy.bodyIndex == tooling.scoreBodyIndex &&
                    proxy.shape == NM_RIGID_CAPSULE &&
                    proxy.prescribedEndPoseTranslation,
                "score capsule source identity or prescribed-motion flag is wrong");
        const BoxVec3 localFirst{proxy.localCenter[0], proxy.localCenter[1], 0.0};
        const BoxVec3 localSecond{proxy.localExtent[0], proxy.localExtent[1], 0.0};
        const BoxVec3 worldFirst = tooling.scoreBodyPosition + localFirst;
        const BoxVec3 worldSecond = tooling.scoreBodyPosition + localSecond;
        const double expectedCenterlineExtension = 2.0e-3 - 0.75e-3;
        const double firstExtension = std::hypot(
            worldFirst.x - line.first.x, worldFirst.y - line.first.y);
        const double secondExtension = std::hypot(
            worldSecond.x - line.second.x, worldSecond.y - line.second.y);
        require(std::abs(firstExtension - expectedCenterlineExtension) < 1.0e-12 &&
                    std::abs(secondExtension - expectedCenterlineExtension) < 1.0e-12,
                "scoring capsule physical end overhang differs from the authored input");
    }
    require(tooling.proxies.back().shape == NM_RIGID_BOX &&
                tooling.proxies.back().bodyIndex == tooling.supportBodyIndex,
            "support anvil body identity is invalid");
    const auto bodyStates = makeBoxToolBodyStates(
        tooling, 2u, blank.boardCaliper + 0.75e-3 + 5.0e-3);
    const BoxVec3 firstContact = boxScoreDieFirstContactPosition(tooling);
    const BoxVec3 parked = boxScoreDieParkPosition(
        tooling, blank.boardCaliper + 0.75e-3 + 5.0e-3);
    require(bodyStates.size() == 4u &&
                bodyStates[tooling.supportBodyIndex].flagsAndIndices[0] ==
                    MR_MOTION_KINEMATIC &&
                std::abs(firstContact.z -
                    (blank.boardCaliper + tooling.punchRadius)) < 1.0e-15 &&
                parked.z > firstContact.z,
            "die/anvil body-state arena has the wrong count or motion mode");

    const auto grips = makeBoxFeatureGrips(blank, blank.points);
    require(grips.size() == 12u,
            "four wall and eight flap actions must each have a grip boundary");
    for (const auto& grip : grips)
        require(!grip.nodeIndices.empty(),
                "assembly action grip selection is empty");
    const auto boundaryNodes = exposedNodes(blank);
    const auto tabAnchors = makeBoxTabBodyAnchorNodes(
        blank, blank.points, boundaryNodes);
    require(tabAnchors.size() >= 3u &&
                std::includes(boundaryNodes.begin(), boundaryNodes.end(),
                              tabAnchors.begin(), tabAnchors.end()),
            "tab body fixture anchors are not a subset of exposed mesh nodes");
    bool anchorsIncludeTheTabFreeEdge = false;
    const double anchorTolerance = 2.0e-10 *
        std::max({blank.perimeterSpan, blank.blankHeight, blank.boardCaliper});
    for (const std::uint32_t node : tabAnchors) {
        const auto& point = blank.points.at(node);
        require(point.x <= 0.85 * blank.config.manufacturerJointWidth + anchorTolerance &&
                    point.y >= blank.bottomScoreY - anchorTolerance &&
                    point.y <= blank.topScoreY + anchorTolerance,
                "tab body anchor selection crossed the score root or left the wall-height region");
        anchorsIncludeTheTabFreeEdge = anchorsIncludeTheTabFreeEdge ||
            std::abs(point.x) <= anchorTolerance;
    }
    require(anchorsIncludeTheTabFreeEdge,
            "tab anchor does not include the seam-side free edge that must later be released");
    const auto program = makeBoxExecutionProgram(blank);
    require(program.size() == 16u &&
                program[0].kind == BoxExecutionStage::Kind::score &&
                program[0].scoreLineIds.size() == 12u,
            "physical scoring stage does not cover all 12 IDs in one die pass");
    for (std::uint32_t score = 0u; score < 12u; ++score)
        require(program[0].scoreLineIds[score] == score,
                "scoring stage changed source score-line indexing");
    for (std::uint32_t action = 0u; action < 15u; ++action)
        require(program[action + 1u].assemblyActionIndex == action,
                "box execution program dropped or reordered an assembly target");

    std::vector<NMFEMNodeStateGPU> actionStart;
    actionStart.reserve(blank.points.size());
    for (const auto& point : blank.points) actionStart.push_back(nodeAt(point));
    std::vector<NMFEMNodeStateGPU> accepted = actionStart;
    constexpr double pi = 3.14159265358979323846;
    const BoxWallFit flatTabFit = measureBoxTabInteriorFit(
        blank, blank.points, accepted, 0u,
        static_cast<std::uint32_t>(blank.points.size()));
    require(flatTabFit.valid &&
                std::abs(flatTabFit.angleDegrees) < 1.0e-3,
            "flat default tab has a flute-profile angle bias");
    double maximumFlatWallAngle = 0.0;
    double maximumFlatRelativeTurn = 0.0;
    for (std::uint32_t panel = 0u; panel < blank.panels.size(); ++panel) {
        const BoxWallFit flatWallFit = measureBoxWallInteriorFit(
            blank, blank.points, accepted, 0u,
            static_cast<std::uint32_t>(blank.points.size()), panel);
        const auto flatRelativeFit = measureBoxRelativeWallTurn(
            blank, blank.points, accepted, 0u,
            static_cast<std::uint32_t>(blank.points.size()), panel);
        require(flatWallFit.valid && flatRelativeFit.valid,
                "flat default wall geometry has no accepted-state angle fit");
        require(flatWallFit.fitResidualRms <=
                    kDefaultBoxShapeResidualBoundCalipers * blank.boardCaliper &&
                    flatWallFit.fitCoverage >= kDefaultBoxShapeFitCoverage &&
                    flatRelativeFit.fitResidualRms <=
                        flatRelativeFit.residualBound &&
                    flatRelativeFit.fitCoverage >= kDefaultBoxShapeFitCoverage,
                "flat wall or adjacent-reference shape fit fails the engineering gate");
        maximumFlatWallAngle = std::max(maximumFlatWallAngle,
            std::abs(flatWallFit.angleDegrees));
        maximumFlatRelativeTurn = std::max(maximumFlatRelativeTurn,
            std::abs(flatRelativeFit.signedTurnDegrees));
    }
    const double oldFlatTabBias = xAxisOnlyAngleDegrees(
        blank.points, accepted,
        0.15 * blank.config.manufacturerJointWidth,
        0.85 * blank.config.manufacturerJointWidth,
        blank.bottomScoreY, blank.topScoreY);
    require(maximumFlatWallAngle < 1.0e-3 &&
                maximumFlatRelativeTurn < 1.0e-3 &&
                std::isfinite(oldFlatTabBias) &&
                std::abs(oldFlatTabBias) < 1.0e-10,
            "flat authored geometry acquired a spurious angle before folding");

    for (std::size_t node = 0; node < accepted.size(); ++node) {
        const double cx = 0.5 * blank.perimeterSpan;
        const double cz = 0.5 * blank.boardCaliper;
        const double x = blank.points[node].x - cx;
        const double z = blank.points[node].z - cz;
        accepted[node].positionAndMass.x = static_cast<float>(
            cx + std::cos(-0.5 * pi) * x + std::sin(-0.5 * pi) * z);
        accepted[node].positionAndMass.z = static_cast<float>(
            cz - std::sin(-0.5 * pi) * x + std::cos(-0.5 * pi) * z);
    }
    const BoxWallFit wallFit = measureBoxWallInteriorFit(
        blank, blank.points, accepted, 0u,
        static_cast<std::uint32_t>(blank.points.size()), 0u);
    require(wallFit.valid && std::abs(wallFit.angleDegrees + 90.0) < 1.0e-3 &&
                wallFit.sampleCount > 0u,
            "accepted interior panel fit did not recover a known quarter-turn");
    const BoxWallFit tabFit = measureBoxTabInteriorFit(
        blank, blank.points, accepted, 0u,
        static_cast<std::uint32_t>(blank.points.size()));
    const auto relativeFit = measureBoxRelativeWallTurn(
        blank, blank.points, accepted, 0u,
        static_cast<std::uint32_t>(blank.points.size()), 0u);
    require(tabFit.valid && std::abs(tabFit.angleDegrees + 90.0) < 1.0e-3 &&
                relativeFit.valid &&
                relativeFit.fitCoverage >= kDefaultBoxShapeFitCoverage &&
                relativeFit.fitResidualRms <= relativeFit.residualBound &&
                std::abs(relativeFit.signedTurnDegrees) < 1.0e-3 &&
                std::abs(relativeFit.targetTurnDegrees + 90.0) < 1.0e-12,
            "whole-blank rigid rotation was mistaken for the first relative wall fold");
    const auto rigidPlan = planBoxAssemblyActionStep(
        blank, blank.points, accepted, accepted, 0u,
        static_cast<std::uint32_t>(blank.points.size()), 0u, 15.0, 0.5);
    require(rigidPlan.valid &&
                rigidPlan.shapeFitWithinBound &&
                rigidPlan.phase == BoxAssemblyActionStep::Phase::moveGrip &&
                std::abs(rigidPlan.observedAngleDegrees) < 1.0e-3 &&
                std::abs(rigidPlan.commandedIncrementDegrees + 15.0) < 1.0e-12,
            "absolute wall alignment falsely completed under whole-blank rigid rotation");

    accepted = actionStart;
    for (std::size_t node = 0; node < accepted.size(); ++node)
        rotateBottomFlapAboutX(blank, blank.points[node], accepted[node], 0.5 * pi);
    const BoxFlapTurnFit flapFit = measureBoxFlapInteriorTurn(
        blank, blank.points, actionStart, accepted, 0u,
        static_cast<std::uint32_t>(blank.points.size()), 4u);
    require(flapFit.valid && std::abs(flapFit.signedTurnDegrees - 90.0) < 1.0e-3 &&
                flapFit.fitResidualRms <= flapFit.residualBound &&
                flapFit.fitCoverage >= kDefaultBoxShapeFitCoverage,
            "accepted interior flap fit did not recover its prescribed quarter-turn");
    std::uint32_t panelZeroFlapAction = kNoBoxScoreLineIndex;
    for (std::uint32_t action = 0u;
         action < blank.assembly.actions.size(); ++action) {
        const auto& candidate = blank.assembly.actions[action];
        if (candidate.kind == BoxAssemblyAction::Kind::foldFlap &&
            candidate.feature.index == 4u) {
            panelZeroFlapAction = action;
            break;
        }
    }
    require(panelZeroFlapAction != kNoBoxScoreLineIndex,
            "source assembly schedule has no action for the tested panel-zero flap");
    auto warpedFlap = accepted;
    std::uint32_t warpedFlapNodes = 0u;
    const BoxScoreLine& flapScore = blank.scoreLines[4u];
    for (std::uint32_t node = 0u; node < warpedFlap.size(); ++node) {
        const BoxBlankPoint& rest = blank.points[node];
        if (rest.x <= blank.panels[0].xStart + 0.20 * blank.config.panelLength ||
            rest.x >= blank.panels[0].xEnd - 0.20 * blank.config.panelLength ||
            rest.y >= flapScore.first.y || (node % 2u) != 0u)
            continue;
        warpedFlap[node].positionAndMass.x += static_cast<float>(
            2.0 * blank.boardCaliper);
        ++warpedFlapNodes;
    }
    const auto warpedFlapPlan = planBoxAssemblyActionStep(
        blank, blank.points, warpedFlap, actionStart, 0u,
        static_cast<std::uint32_t>(blank.points.size()), panelZeroFlapAction,
        30.0, 0.5);
    require(warpedFlapNodes > 0u && warpedFlapPlan.angleObservable &&
                std::abs(warpedFlapPlan.observedAngleDegrees - 90.0) < 1.0e-3 &&
                !warpedFlapPlan.valid && warpedFlapPlan.shapeFitRejected &&
                !warpedFlapPlan.shapeFitWithinBound && warpedFlapPlan.fitCoverage <
                    kDefaultBoxShapeFitCoverage &&
                warpedFlapPlan.fitResidualRms > warpedFlapPlan.shapeFitResidualBound &&
                warpedFlapPlan.nodeTargets.empty(),
            "target-angle but buckled flap was accepted or given interior targets");

    const auto& wallGrip = grips.front();
    std::vector<nm_float4> targetArena(blank.points.size(),
        nm_float4{0.0f, 0.0f, 0.0f, 0.0f});
    const auto selectedNode = wallGrip.nodeIndices.front();
    setBoxGripTarget(targetArena,
        static_cast<std::uint32_t>(blank.points.size()), 0u,
        selectedNode, {1.0e-3, 2.0e-3, 3.0e-3});
    require(targetArena[selectedNode].w == 3.0f &&
                std::abs(targetArena[selectedNode].x - 1.0e-3f) < 1.0e-9f,
            "temporary grip command did not retain its target/command tag");
    setBoxGripTarget(targetArena,
        static_cast<std::uint32_t>(blank.points.size()), 0u,
        selectedNode, {}, 2.0f);
    require(targetArena[selectedNode].w == 2.0f,
            "temporary grip release command was not represented");
    auto anchorTargetArena = std::vector<nm_float4>(blank.points.size(),
        nm_float4{0.0f, 0.0f, 0.0f, 0.0f});
    for (const std::uint32_t node : tabAnchors) {
        const BoxVec3 acceptedPosition = boxAcceptedPosition(nodeAt(blank.points[node]));
        setBoxGripTarget(anchorTargetArena,
            static_cast<std::uint32_t>(blank.points.size()), 0u,
            node, acceptedPosition, 3.0f);
        require(anchorTargetArena[node].w == 3.0f &&
                    anchorTargetArena[node].x == acceptedPosition.x &&
                    anchorTargetArena[node].y == acceptedPosition.y &&
                    anchorTargetArena[node].z == acceptedPosition.z,
                "tab body anchor command is not at the accepted authored pose");
    }

    const auto firstWallPlan = planBoxAssemblyActionStep(
        blank, blank.points, actionStart, actionStart, 0u,
        static_cast<std::uint32_t>(blank.points.size()), 0u, 15.0, 0.5);
    require(firstWallPlan.valid &&
                firstWallPlan.phase == BoxAssemblyActionStep::Phase::moveGrip &&
                firstWallPlan.hinge.valid &&
                std::abs(firstWallPlan.commandedIncrementDegrees + 15.0) < 1.0e-12 &&
                firstWallPlan.nodeTargets.size() == wallGrip.nodeIndices.size(),
            "first wall fold did not plan an accepted-state grip increment");
    for (const auto& target : firstWallPlan.nodeTargets)
        require(target.command == 3.0f && target.node < blank.points.size(),
                "wall fold target omitted the temporary-grip tag or node identity");

    std::uint32_t firstFlapAction = kNoBoxScoreLineIndex;
    for (std::uint32_t action = 0u;
         action < blank.assembly.actions.size(); ++action) {
        if (blank.assembly.actions[action].kind ==
                BoxAssemblyAction::Kind::foldFlap) {
            firstFlapAction = action;
            break;
        }
    }
    require(firstFlapAction != kNoBoxScoreLineIndex,
            "assembly schedule has no flap action to exercise");
    const auto firstFlapPlan = planBoxAssemblyActionStep(
        blank, blank.points, actionStart, actionStart, 0u,
        static_cast<std::uint32_t>(blank.points.size()), firstFlapAction,
        30.0, 0.5);
    require(firstFlapPlan.valid &&
                firstFlapPlan.phase == BoxAssemblyActionStep::Phase::moveGrip &&
                firstFlapPlan.hinge.valid &&
                std::abs(firstFlapPlan.commandedIncrementDegrees - 30.0) < 1.0e-12,
            "first flap fold did not plan a solver-driven grip increment");

    accepted = actionStart;
    const BoxHingeFrame firstWallHinge = measureBoxAcceptedHingeFrame(
        blank, blank.points, accepted, 0u,
        static_cast<std::uint32_t>(blank.points.size()), 0u);
    require(firstWallHinge.valid,
            "first wall hinge could not be recovered from accepted blank nodes");
    const BoxBlankPanel& firstPanel = blank.panels[0];
    for (std::size_t node = 0; node < accepted.size(); ++node) {
        const BoxBlankPoint& rest = blank.points[node];
        if (rest.x >= firstPanel.xStart && rest.x <= firstPanel.xEnd &&
            rest.y >= firstPanel.yStart && rest.y <= firstPanel.yEnd) {
            const BoxVec3 turned = boxRotateAboutAxis(
                boxAcceptedPosition(accepted[node]), firstWallHinge.origin,
                firstWallHinge.axis, -0.5 * pi);
            accepted[node].positionAndMass.x = static_cast<float>(turned.x);
            accepted[node].positionAndMass.y = static_cast<float>(turned.y);
            accepted[node].positionAndMass.z = static_cast<float>(turned.z);
        }
    }
    for (const std::uint32_t node : wallGrip.nodeIndices) {
        accepted[node].restAndFixed.w = 3.0f;
        accepted[node].velocityAndInverseMass.w = 0.0f;
    }
    const auto completedWallPlan = planBoxAssemblyActionStep(
        blank, blank.points, accepted, actionStart, 0u,
        static_cast<std::uint32_t>(blank.points.size()), 0u, 15.0, 0.5);
    require(completedWallPlan.valid &&
                completedWallPlan.phase == BoxAssemblyActionStep::Phase::acceptedContactOnly &&
                std::abs(completedWallPlan.observedAngleDegrees + 90.0) < 1.0e-3 &&
                completedWallPlan.shapeFitWithinBound &&
                completedWallPlan.fitCoverage >= kDefaultBoxShapeFitCoverage &&
                completedWallPlan.fitResidualRms <=
                    kDefaultBoxShapeResidualBoundCalipers * blank.boardCaliper &&
                completedWallPlan.nodeTargets.empty(),
            "completed wall turn did not retain its accepted external boundary clamp");
    for (const std::uint32_t node : wallGrip.nodeIndices)
        require(accepted[node].restAndFixed.w == 3.0f,
                "completed wall turn lost a retained boundary-grip tag");

    // The active wall is well-shaped, but 10% of the smaller parent-tab fit
    // is warped. A weighted aggregate can hide this required-reference
    // failure because the wall has many more interior samples.
    auto warpedParentAccepted = accepted;
    std::uint32_t parentCandidate = 0u;
    std::uint32_t warpedParentNodes = 0u;
    const double tabInset = 0.15 * blank.config.manufacturerJointWidth;
    for (std::uint32_t node = 0u; node < warpedParentAccepted.size(); ++node) {
        const BoxBlankPoint& rest = blank.points[node];
        if (rest.x <= tabInset ||
            rest.x >= blank.config.manufacturerJointWidth - tabInset ||
            rest.y < blank.bottomScoreY || rest.y > blank.topScoreY)
            continue;
        if ((parentCandidate++ % 10u) != 0u) continue;
        warpedParentAccepted[node].positionAndMass.y += static_cast<float>(
            2.0 * blank.boardCaliper);
        ++warpedParentNodes;
    }
    const BoxWallFit goodRequiredWall = measureBoxWallInteriorFit(
        blank, blank.points, warpedParentAccepted, 0u,
        static_cast<std::uint32_t>(blank.points.size()), 0u);
    const BoxWallFit warpedRequiredParent = measureBoxTabInteriorFit(
        blank, blank.points, warpedParentAccepted, 0u,
        static_cast<std::uint32_t>(blank.points.size()));
    const double legacyWeightedRms = std::sqrt(
        (goodRequiredWall.fitResidualRms * goodRequiredWall.fitResidualRms *
             goodRequiredWall.sampleCount +
         warpedRequiredParent.fitResidualRms * warpedRequiredParent.fitResidualRms *
             warpedRequiredParent.sampleCount) /
        (goodRequiredWall.sampleCount + warpedRequiredParent.sampleCount));
    const double legacyWeightedCoverage =
        (goodRequiredWall.fitCoverage * goodRequiredWall.sampleCount +
         warpedRequiredParent.fitCoverage * warpedRequiredParent.sampleCount) /
        (goodRequiredWall.sampleCount + warpedRequiredParent.sampleCount);
    const auto warpedParentPlan = planBoxAssemblyActionStep(
        blank, blank.points, warpedParentAccepted, actionStart, 0u,
        static_cast<std::uint32_t>(blank.points.size()), 0u, 15.0, 0.5);
    require(warpedParentNodes > 0u && goodRequiredWall.valid &&
                goodRequiredWall.fitResidualRms <=
                    kDefaultBoxShapeResidualBoundCalipers * blank.boardCaliper &&
                goodRequiredWall.fitCoverage >= kDefaultBoxShapeFitCoverage &&
                warpedRequiredParent.valid &&
                (warpedRequiredParent.fitResidualRms >
                     kDefaultBoxShapeResidualBoundCalipers * blank.boardCaliper ||
                 warpedRequiredParent.fitCoverage < kDefaultBoxShapeFitCoverage) &&
                legacyWeightedRms <=
                    kDefaultBoxShapeResidualBoundCalipers * blank.boardCaliper &&
                legacyWeightedCoverage >= kDefaultBoxShapeFitCoverage &&
                warpedParentPlan.angleObservable &&
                std::abs(warpedParentPlan.observedAngleDegrees + 90.0) < 1.0e-3 &&
                !warpedParentPlan.valid && warpedParentPlan.shapeFitRejected &&
                !warpedParentPlan.shapeFitWithinBound &&
                warpedParentPlan.fitResidualRms == warpedRequiredParent.fitResidualRms &&
                warpedParentPlan.fitCoverage == warpedRequiredParent.fitCoverage &&
                warpedParentPlan.nodeTargets.empty(),
            "bad parent reference was hidden by the larger good wall fit");

    // Keep the fitted x-z angle exactly at target while buckling half of the
    // interior points out of the authored x-z fold plane. The 3-D residual
    // and coverage gate must reject this as completion without prescribing
    // any interior node target.
    auto warpedAccepted = accepted;
    const BoxBlankPanel& warpedPanel = blank.panels[0];
    std::uint32_t warpedNodes = 0u;
    for (std::uint32_t node = 0u; node < warpedAccepted.size(); ++node) {
        const BoxBlankPoint& rest = blank.points[node];
        if (rest.x <= warpedPanel.xStart + 0.20 * (warpedPanel.xEnd - warpedPanel.xStart) ||
            rest.x >= warpedPanel.xEnd - 0.20 * (warpedPanel.xEnd - warpedPanel.xStart) ||
            rest.y < warpedPanel.yStart || rest.y > warpedPanel.yEnd ||
            (node % 2u) != 0u)
            continue;
        warpedAccepted[node].positionAndMass.y += static_cast<float>(
            2.0 * blank.boardCaliper);
        ++warpedNodes;
    }
    const auto warpedPlan = planBoxAssemblyActionStep(
        blank, blank.points, warpedAccepted, actionStart, 0u,
        static_cast<std::uint32_t>(blank.points.size()), 0u, 15.0, 0.5);
    require(warpedNodes > 0u && warpedPlan.angleObservable &&
                std::abs(warpedPlan.observedAngleDegrees + 90.0) < 1.0e-3 &&
                !warpedPlan.valid && warpedPlan.shapeFitRejected &&
                !warpedPlan.shapeFitWithinBound && warpedPlan.fitCoverage <
                    kDefaultBoxShapeFitCoverage &&
                warpedPlan.fitResidualRms > warpedPlan.shapeFitResidualBound &&
                warpedPlan.nodeTargets.empty(),
            "target-angle but buckled panel was accepted or given interior targets");

    const auto jointContactPlan = planBoxAssemblyActionStep(
        blank, blank.points, actionStart, actionStart, 0u,
        static_cast<std::uint32_t>(blank.points.size()), 4u, 15.0, 0.5);
    require(jointContactPlan.valid &&
                jointContactPlan.phase == BoxAssemblyActionStep::Phase::acceptedContactOnly &&
                jointContactPlan.physicalBondLawRequired &&
                jointContactPlan.nodeTargets.empty(),
            "unmodeled manufacturer's joint was treated as a bonded/gripped action");

    accepted = actionStart;
    const auto closureGaps = measureBoxClosureGapDiagnostics(
        blank, blank.points, accepted, 0u,
        static_cast<std::uint32_t>(blank.points.size()), false);
    require(closureGaps.manufacturerJoint.valid &&
                closureGaps.shortFlapOverlap.valid &&
                closureGaps.longFlapOverlap.valid &&
                std::isfinite(closureGaps.manufacturerJoint.distance) &&
                closureGaps.manufacturerJoint.distance >= 0.0,
            "box seam/flap closure diagnostics omitted authored regions");

    std::cout << "box_assembly_controller_cpu_passed"
              << " score_ids=12"
              << " assembly_actions=15"
              << " boundary_grips=12"
              << " wall_fit=interior_accepted_nodes"
              << " relative_turn=adjacent_panel_axis_difference"
              << " rigid_rotation_counterexample=passed"
              << " warped_wall_flap_shape_gate=passed"
              << " worst_required_parent_fit_gate=passed"
              << " legacy_weighted_rms_calipers="
              << legacyWeightedRms / blank.boardCaliper
              << " legacy_weighted_coverage=" << legacyWeightedCoverage
              << " warped_parent_coverage=" << warpedRequiredParent.fitCoverage
              << " shape_residual_bound_calipers="
              << kDefaultBoxShapeResidualBoundCalipers
              << " shape_coverage_min=" << kDefaultBoxShapeFitCoverage
              << " default_tab_axis_only_bias_deg=" << oldFlatTabBias
              << " flat_tab_relative_fit_deg=" << flatTabFit.angleDegrees
              << " flat_wall_angle_max_deg=" << maximumFlatWallAngle
              << " flat_relative_turn_max_deg=" << maximumFlatRelativeTurn
              << " tab_body_anchor_nodes=" << tabAnchors.size()
              << " flap_fit=interior_accepted_nodes"
              << " accepted_grip_steps=wall_clamp_retained_flap_release"
              << " bond_law_gap_fail_closed=true"
              << " seam_flap_gap=nearest_node_diagnostic_only"
              << " physical_validation=false\n";
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
