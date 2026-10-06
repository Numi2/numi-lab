#include "cardboard_tooling.hpp"

#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using numi::cardboard::CreaseToolingSpec;
using numi::cardboard::ToolPose;
using numi::cardboard::ToolQuaternion;
using numi::cardboard::ToolVec3;
using numi::cardboard::boxContactWitness;
using numi::cardboard::capsuleContactWitness;
using numi::cardboard::makeCreaseTooling;
using numi::cardboard::punchPoseForIndentation;

unsigned checks = 0u;

void require(const bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

bool near(const double a, const double b, const double tolerance) {
    return std::abs(a - b) <= tolerance;
}

bool near(const ToolVec3 a, const ToolVec3 b, const double tolerance) {
    return near(a.x, b.x, tolerance) && near(a.y, b.y, tolerance) &&
        near(a.z, b.z, tolerance);
}

void sourceProxyContract() {
    constexpr double mm = 1.0e-3;
    const CreaseToolingSpec spec{
        .boardLength = 31.6 * mm,
        .boardWidth = 20.0 * mm,
        .boardCaliper = 4.210 * mm,
        .creaseX = 0.0,
        .supportSurfaceZ = 0.0,
        .punchRadius = 0.75 * mm,
        .punchEndOverhang = 2.0 * mm,
        .supportEndOverhang = 2.0 * mm,
        .supportThickness = 3.0 * mm,
        .initialClearance = 0.10 * mm,
        .punchBodyIndex = 0u,
        .contactMaterialIndex = 0u,
    };
    const auto tooling = makeCreaseTooling(spec);
    require(tooling.punch.shape == NM_RIGID_CAPSULE,
            "crease punch must map to the native capsule proxy");
    require(tooling.support.shape == NM_RIGID_BOX,
            "backing anvil must map to the native box proxy");
    require(tooling.punch.bodyIndex == 0u &&
                !tooling.punch.articulated && !tooling.punch.dynamic &&
                tooling.punch.sceneBodyIndex == NM_INVALID_INDEX,
            "moving punch must use the body-backed kinematic proxy path");
    require(near(tooling.punch.radiusOrOffset, spec.punchRadius, 1.0e-15),
            "punch radius must remain the authored rounded nose radius");
    require(near(tooling.punchOverallSpan,
                 spec.boardWidth + 2.0 * spec.punchEndOverhang, 1.0e-15),
            "rounded punch must span the board width plus physical end overhang");
    require(near(tooling.punchCenterlineSpan,
                 tooling.punchOverallSpan - 2.0 * spec.punchRadius, 1.0e-15),
            "capsule centreline span must account for rounded end-cap extent");
    require(near(tooling.supportHalfLength,
                 0.5 * spec.boardLength + spec.supportEndOverhang, 1.0e-15) &&
                near(tooling.supportHalfWidth,
                     0.5 * spec.boardWidth + spec.supportEndOverhang, 1.0e-15),
            "finite support footprint must cover the coupon and overhang");
    require(near(tooling.support.localCenter[2] +
                     tooling.support.localExtent[2],
                 spec.supportSurfaceZ, 1.0e-15),
            "support box top must coincide with the declared support surface");
    require(near(tooling.punchInitialPose.translation.z,
                 spec.supportSurfaceZ + spec.boardCaliper +
                     spec.punchRadius + spec.initialClearance,
                 1.0e-15),
            "reference pose must place the rounded nose above the board by the requested gap");

    const ToolVec3 boardTop{
        spec.creaseX, 0.0,
        spec.supportSurfaceZ + spec.boardCaliper,
    };
    const auto openGap = capsuleContactWitness(
        tooling.punch, tooling.punchInitialPose, boardTop);
    require(near(openGap.separation, spec.initialClearance, 1.0e-12),
            "native capsule signed distance must equal the authored initial clearance");
    require(near(openGap.normal, {0.0, 0.0, -1.0}, 1.0e-12),
            "surface normal below the punch must point away from its centreline");
    require(near(openGap.point.z,
                 boardTop.z + spec.initialClearance, 1.0e-12),
            "capsule witness must lie on the rounded punch surface");

    const double stroke = 0.20 * mm;
    const ToolPose indented = punchPoseForIndentation(spec, stroke);
    const auto loaded = capsuleContactWitness(tooling.punch, indented, boardTop);
    require(near(loaded.separation, spec.initialClearance - stroke, 1.0e-12) &&
                loaded.separation < 0.0,
            "the reference tooling trajectory must create geometric punch penetration");
    require(near(indented.translation.z,
                 tooling.punchInitialPose.translation.z - stroke, 1.0e-15),
            "indentation target must preserve all pose coordinates except punch height");

    const ToolVec3 supportPoint{
        0.0, 0.0, spec.supportSurfaceZ + 0.2 * mm};
    const ToolPose supportPose{};
    const auto supportWitness = boxContactWitness(
        tooling.support, supportPose, supportPoint);
    require(near(supportWitness.separation, 0.2 * mm, 1.0e-12) &&
                near(supportWitness.normal, {0.0, 0.0, 1.0}, 1.0e-12),
            "finite anvil must return a positive outside distance and upward normal");
    require(near(supportWitness.point.z, spec.supportSurfaceZ, 1.0e-12),
            "anvil witness must lie on its finite top face");
    const auto embedded = boxContactWitness(
        tooling.support, supportPose,
        {0.0, 0.0, spec.supportSurfaceZ - 0.3 * mm});
    require(near(embedded.separation, -0.3 * mm, 1.0e-12) &&
                near(embedded.normal, {0.0, 0.0, 1.0}, 1.0e-12),
            "anvil signed distance must be negative inside the support solid");
}

void rotatedProxyContract() {
    constexpr double mm = 1.0e-3;
    const CreaseToolingSpec spec{
        .boardLength = 20.0 * mm,
        .boardWidth = 10.0 * mm,
        .boardCaliper = 2.0 * mm,
        .supportSurfaceZ = 0.0,
        .punchRadius = 0.5 * mm,
        .supportThickness = 1.0 * mm,
        .punchBodyIndex = 0u,
    };
    const auto tooling = makeCreaseTooling(spec);
    const ToolPose quarterTurn{
        tooling.punchInitialPose.translation,
        {0.0, 0.0, std::sqrt(0.5), std::sqrt(0.5)},
    };
    const double halfSpan = 0.5 * spec.boardWidth;
    const ToolVec3 endpoint = numi::cardboard::transformPoint(
        quarterTurn, {0.0, -halfSpan, 0.0});
    require(near(endpoint.x - quarterTurn.translation.x,
                 halfSpan, 1.0e-12) &&
                near(endpoint.y - quarterTurn.translation.y,
                     0.0, 1.0e-12),
            "body rotation must carry the capsule axis with the rigid tool");
    const ToolVec3 point{
        quarterTurn.translation.x,
        quarterTurn.translation.y,
        quarterTurn.translation.z - spec.punchRadius - 0.1 * mm,
    };
    const auto witness = capsuleContactWitness(
        tooling.punch, quarterTurn, point);
    require(near(witness.separation, 0.1 * mm, 1.0e-12) &&
                near(witness.normal, {0.0, 0.0, -1.0}, 1.0e-12),
            "rotating the punch about thickness axis must preserve normal clearance");
}

void rejectsInvalidGeometry() {
    CreaseToolingSpec spec{
        .boardLength = 10.0e-3,
        .boardWidth = 10.0e-3,
        .boardCaliper = 2.0e-3,
        .punchRadius = 0.5e-3,
        .supportThickness = 1.0e-3,
        .punchBodyIndex = 0u,
    };
    spec.punchBodyIndex = NM_INVALID_INDEX;
    bool rejected = false;
    try {
        (void)makeCreaseTooling(spec);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected,
            "factory must reject a punch with no external body-state binding");
    spec.punchBodyIndex = 0u;
    spec.punchRadius = 0.0;
    rejected = false;
    try {
        (void)makeCreaseTooling(spec);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "factory must reject a zero-radius punch");
}

} // namespace

int main() {
    try {
        sourceProxyContract();
        rotatedProxyContract();
        rejectsInvalidGeometry();
        std::cout << "cardboard tooling geometry checks passed: "
                  << checks << " assertions\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cardboard tooling geometry check failed after "
                  << checks << " assertions: " << error.what() << '\n';
        return 1;
    }
}
