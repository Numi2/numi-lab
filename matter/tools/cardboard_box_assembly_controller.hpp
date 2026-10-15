#pragma once

// Native boundary-action helpers for the source-authored FEFCO 0201 blank.
// This file builds tool/grip identities and accepted-state observables only:
// it does not author folded coordinates, weaken a score line, bond a seam, or
// claim calibration to the literature recipe.

#include "cardboard_box_blank.hpp"
#include "cardboard_shipping_protocol.hpp"
#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace numi::cardboard {

// The shipping scoring assembly has three fixed-orientation score families.
// Each family owns one capsule punch and two box rails; the shipping world
// adds a platen, floor, and finite-mass impactor proxy/body slot.
inline constexpr std::uint32_t kShippingScoreToolFamilyCount = 3u;
inline constexpr std::uint32_t kShippingRigidProxiesPerScoreFamily = 3u;
inline constexpr std::uint32_t kShippingAdditionalRigidProxyCount = 3u;
inline constexpr std::uint32_t kShippingSceneBodiesPerScoreFamily = 2u;
inline constexpr std::uint32_t kShippingAdditionalBodySlotCount = 3u;
inline constexpr std::uint32_t kShippingScoreToolProxyCount =
    kShippingScoreToolFamilyCount * kShippingRigidProxiesPerScoreFamily;
inline constexpr std::uint32_t kShippingRigidProxyCount =
    kShippingScoreToolProxyCount + kShippingAdditionalRigidProxyCount;
inline constexpr std::uint32_t kShippingScoreToolSceneBodyCount =
    kShippingScoreToolFamilyCount * kShippingSceneBodiesPerScoreFamily;
inline constexpr std::uint32_t kShippingRigidBodyCount =
    kShippingScoreToolSceneBodyCount + kShippingAdditionalBodySlotCount;

// Engineering diagnostic defaults, not published manufacturing tolerances.
// The residual bound scales with authored board caliper so the same controller
// remains meaningful when the coupon/blank dimensions change.
inline constexpr double kDefaultBoxShapeResidualBoundCalipers = 0.5;
inline constexpr double kDefaultBoxShapeFitCoverage = 0.95;

struct BoxVec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

inline BoxVec3 operator+(const BoxVec3 a, const BoxVec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
inline BoxVec3 operator-(const BoxVec3 a, const BoxVec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
inline BoxVec3 operator*(const double scale, const BoxVec3 value) {
    return {scale * value.x, scale * value.y, scale * value.z};
}
inline double boxDot(const BoxVec3 a, const BoxVec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline BoxVec3 boxCross(const BoxVec3 a, const BoxVec3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}
inline double boxLength(const BoxVec3 value) {
    return std::sqrt(boxDot(value, value));
}
inline BoxVec3 boxNormalized(const BoxVec3 value) {
    const double length = boxLength(value);
    if (!(length > 1.0e-15) || !std::isfinite(length))
        throw std::invalid_argument("box assembly vector is degenerate");
    return (1.0 / length) * value;
}
inline BoxVec3 boxRotateAboutAxis(const BoxVec3 position,
                                  const BoxVec3 origin,
                                  const BoxVec3 axis,
                                  const double radians) {
    const BoxVec3 relative = position - origin;
    const double cosine = std::cos(radians);
    const double sine = std::sin(radians);
    return origin + cosine * relative +
        sine * boxCross(axis, relative) +
        ((1.0 - cosine) * boxDot(axis, relative)) * axis;
}
inline BoxVec3 boxAcceptedPosition(const NMFEMNodeStateGPU& node) {
    return {node.positionAndMass.x, node.positionAndMass.y,
            node.positionAndMass.z};
}

struct BoxScoreToolingSet {
    // The legacy factory emits twelve per-line capsule proxies and optional
    // channel rails. Shipping mode instead uses three fixed-orientation
    // families through makeSequentialBoxScoreTooling below.
    std::vector<numi::matter::RigidProxySource> proxies;
    BoxVec3 scoreBodyPosition{};
    BoxVec3 supportBodyPosition{};
    std::uint32_t scoreBodyIndex = 0u;
    std::uint32_t supportBodyIndex = 1u;
    double punchRadius = 0.0;
    double boardCaliper = 0.0;
    double toolClearance = 0.0;
    double anvilClearance = 0.0;
    double supportThickness = 0.0;
    double supportHalfX = 0.0;
    double supportHalfY = 0.0;
    double channelWidth = 0.0;
    double supportRailWidth = 0.0;
    // The shipping blank can reuse a fixed-orientation punch/rail triplet for
    // each distinct (axis,length) score-line family.  Body poses stay
    // identity-oriented and relocate by translation only.  The original
    // all-lines tooling factory above remains available for legacy controls.
    struct SequentialFamily {
        std::uint32_t scoreBodyIndex = 0u;
        std::uint32_t supportBodyIndex = 0u;
        BoxVec3 referenceCenter{};
        BoxVec3 tangent{};
        BoxVec3 normal{};
        double lineLength = 0.0;
        std::vector<std::uint32_t> scoreLineIndices;
    };
    bool sequentialFamilies = false;
    std::vector<SequentialFamily> families;
    std::vector<std::uint32_t> scoreLineFamilyIndices;
};

inline std::array<double, 4> boxQuaternionFromLocalY(
    const double dx, const double dy) {
    const double length = std::hypot(dx, dy);
    if (!(length > 1.0e-15) || !std::isfinite(length))
        throw std::invalid_argument("box score line has zero length");
    // Rotate local +y into the authored score-line direction in the x/y plane.
    const double angle = std::atan2(-dx / length, dy / length);
    return {0.0, 0.0, std::sin(0.5 * angle), std::cos(0.5 * angle)};
}

inline BoxScoreToolingSet makeBoxScoreTooling(
    const FEFCO0201Blank& blank,
    const double punchRadius,
    const double punchOverhang,
    const double toolClearance,
    const double anvilClearance,
    const double supportThickness,
    const double anvilOverhang,
    const double channelWidth = 0.0,
    const double supportRailWidth = 0.010) {
    const auto positive = [](const double value) {
        return std::isfinite(value) && value > 0.0;
    };
    if (blank.scoreLines.size() != 12u || !positive(punchRadius) ||
        !std::isfinite(punchOverhang) || punchOverhang < 0.0 ||
        !positive(toolClearance) || !positive(anvilClearance) ||
        !positive(supportThickness) || !std::isfinite(anvilOverhang) ||
        anvilOverhang < 0.0 || !(toolClearance > 0.0) ||
        !std::isfinite(channelWidth) || channelWidth < 0.0 ||
        (channelWidth > 0.0 && !positive(supportRailWidth)))
        throw std::invalid_argument("box scoring tool inputs are invalid");

    BoxScoreToolingSet result;
    result.punchRadius = punchRadius;
    result.boardCaliper = blank.boardCaliper;
    result.toolClearance = toolClearance;
    result.anvilClearance = anvilClearance;
    result.supportThickness = supportThickness;
    result.supportHalfX = 0.5 * blank.perimeterSpan + anvilOverhang;
    result.supportHalfY = 0.5 * blank.blankHeight + anvilOverhang;
    result.channelWidth = channelWidth;
    result.supportRailWidth = channelWidth > 0.0 ? supportRailWidth : 0.0;
    result.scoreBodyIndex = 0u;
    result.supportBodyIndex = 1u;
    result.scoreBodyPosition = {
        0.5 * blank.perimeterSpan, 0.5 * blank.blankHeight,
        blank.boardCaliper + punchRadius + toolClearance};
    result.supportBodyPosition = {
        0.5 * blank.perimeterSpan, 0.5 * blank.blankHeight, 0.0};
    result.proxies.reserve(blank.scoreLines.size() *
        (channelWidth > 0.0 ? 3u : 1u) +
        (channelWidth > 0.0 ? 0u : 1u));

    for (std::uint32_t scoreIndex = 0u;
         scoreIndex < blank.scoreLines.size(); ++scoreIndex) {
        const BoxScoreLine& line = blank.scoreLines[scoreIndex];
        const double dx = line.second.x - line.first.x;
        const double dy = line.second.y - line.first.y;
        const double lineLength = std::hypot(dx, dy);
        const double halfCenterline =
            0.5 * lineLength + punchOverhang - punchRadius;
        if (!(lineLength > 0.0) || !(halfCenterline > 0.0) ||
            !std::isfinite(halfCenterline))
            throw std::invalid_argument("box score punch cannot span its line");

        numi::matter::RigidProxySource punch;
        punch.shape = NM_RIGID_CAPSULE;
        punch.bodyIndex = result.scoreBodyIndex;
        punch.materialIndex = 0u;
        // The one die body stays in the blank frame with identity orientation;
        // each capsule's endpoints are authored in that common local frame.
        // The source helper's overhang denotes total physical extension, so
        // centerline endpoints stop radius metres short of that allowance.
        const double extension = punchOverhang - punchRadius;
        const double ux = dx / lineLength;
        const double uy = dy / lineLength;
        punch.localCenter = {
            line.first.x - extension * ux - result.scoreBodyPosition.x,
            line.first.y - extension * uy - result.scoreBodyPosition.y, 0.0};
        punch.localExtent = {
            line.second.x + extension * ux - result.scoreBodyPosition.x,
            line.second.y + extension * uy - result.scoreBodyPosition.y, 0.0};
        punch.localOrientation = {0.0, 0.0, 0.0, 1.0};
        punch.radiusOrOffset = punchRadius;
        punch.articulated = false;
        punch.dynamic = false;
        punch.prescribedEndPoseTranslation = true;
        result.proxies.push_back(punch);
    }

    if (channelWidth > 0.0) {
        const double railHalfWidth = 0.5 * supportRailWidth;
        const double railOffset = 0.5 * channelWidth + railHalfWidth;
        for (const BoxScoreLine& line : blank.scoreLines) {
            const double dx = line.second.x - line.first.x;
            const double dy = line.second.y - line.first.y;
            const double lineLength = std::hypot(dx, dy);
            if (!(lineLength > 0.0) || !std::isfinite(lineLength))
                throw std::invalid_argument("score channel has a zero-length centerline");
            const double tangentX = dx / lineLength;
            const double tangentY = dy / lineLength;
            const double normalX = -tangentY;
            const double normalY = tangentX;
            const double midpointX = 0.5 * (line.first.x + line.second.x);
            const double midpointY = 0.5 * (line.first.y + line.second.y);
            const double halfSpan = 0.5 * lineLength + anvilOverhang;
            const double angle = std::atan2(tangentY, tangentX);
            const std::array<double, 4> orientation{
                0.0, 0.0, std::sin(0.5 * angle), std::cos(0.5 * angle)};
            for (const double side : {-1.0, 1.0}) {
                numi::matter::RigidProxySource rail;
                rail.shape = NM_RIGID_BOX;
                rail.bodyIndex = result.supportBodyIndex;
                rail.materialIndex = 0u;
                rail.localCenter = {
                    midpointX + side * railOffset * normalX -
                        result.supportBodyPosition.x,
                    midpointY + side * railOffset * normalY -
                        result.supportBodyPosition.y,
                    -anvilClearance - 0.5 * supportThickness};
                rail.localExtent = {
                    halfSpan, railHalfWidth, 0.5 * supportThickness};
                rail.localOrientation = orientation;
                rail.articulated = false;
                rail.dynamic = false;
                result.proxies.push_back(rail);
            }
        }
        return result;
    }

    numi::matter::RigidProxySource anvil;
    anvil.shape = NM_RIGID_BOX;
    anvil.bodyIndex = result.supportBodyIndex;
    anvil.materialIndex = 0u;
    anvil.localCenter = {
        0.0, 0.0, -anvilClearance - 0.5 * supportThickness};
    anvil.localExtent = {
        result.supportHalfX, result.supportHalfY, 0.5 * supportThickness};
    anvil.localOrientation = {0.0, 0.0, 0.0, 1.0};
    anvil.articulated = false;
    anvil.dynamic = false;
    result.proxies.push_back(anvil);
    return result;
}

// Build one physically movable punch and one pair of support rails per
// distinct score-line axis/length.  A score line selects one family and the
// driver moves only that family's two kinematic bodies to the line midpoint.
// Shapes have fixed authored local orientations; body orientations remain
// identity throughout, so the native prescribed-capsule path never rotates a
// capsule and BOX rails are not silently rebound between proxy indices.
inline BoxScoreToolingSet makeSequentialBoxScoreTooling(
    const FEFCO0201Blank& blank,
    const double punchRadius,
    const double punchOverhang,
    const double toolClearance,
    const double anvilClearance,
    const double supportThickness,
    const double anvilOverhang,
    const double channelWidth,
    const double supportRailWidth = 0.010) {
    const auto positive = [](const double value) {
        return std::isfinite(value) && value > 0.0;
    };
    if (blank.scoreLines.size() != 12u || !positive(punchRadius) ||
        !std::isfinite(punchOverhang) || punchOverhang < 0.0 ||
        !positive(toolClearance) || !positive(anvilClearance) ||
        !positive(supportThickness) || !std::isfinite(anvilOverhang) ||
        anvilOverhang < 0.0 || !positive(channelWidth) ||
        !positive(supportRailWidth))
        throw std::invalid_argument("sequential box scoring tool inputs are invalid");

    BoxScoreToolingSet result;
    result.sequentialFamilies = true;
    result.punchRadius = punchRadius;
    result.boardCaliper = blank.boardCaliper;
    result.toolClearance = toolClearance;
    result.anvilClearance = anvilClearance;
    result.supportThickness = supportThickness;
    result.supportHalfX = 0.5 * blank.perimeterSpan + anvilOverhang;
    result.supportHalfY = 0.5 * blank.blankHeight + anvilOverhang;
    result.channelWidth = channelWidth;
    result.supportRailWidth = supportRailWidth;
    result.scoreLineFamilyIndices.resize(blank.scoreLines.size());

    constexpr double directionTolerance = 1.0e-10;
    constexpr double lengthTolerance = 1.0e-10;
    for (std::uint32_t scoreIndex = 0u;
         scoreIndex < blank.scoreLines.size(); ++scoreIndex) {
        const BoxScoreLine& line = blank.scoreLines[scoreIndex];
        const double dx = line.second.x - line.first.x;
        const double dy = line.second.y - line.first.y;
        const double length = std::hypot(dx, dy);
        if (!(length > 0.0) || !std::isfinite(length) ||
            !(0.5 * length + anvilOverhang > 0.0))
            throw std::invalid_argument("sequential score line has invalid span");
        double tx = dx / length;
        double ty = dy / length;
        // Give each unoriented line a canonical tangent so opposite endpoint
        // order does not create duplicate families.
        if (tx < -directionTolerance ||
            (std::abs(tx) <= directionTolerance && ty < 0.0)) {
            tx = -tx;
            ty = -ty;
        }
        const auto found = std::find_if(
            result.families.begin(), result.families.end(),
            [&](const BoxScoreToolingSet::SequentialFamily& family) {
                return std::abs(family.tangent.x - tx) <= directionTolerance &&
                    std::abs(family.tangent.y - ty) <= directionTolerance &&
                    std::abs(family.lineLength - length) <= lengthTolerance;
            });
        std::uint32_t familyIndex = 0u;
        if (found == result.families.end()) {
            familyIndex = static_cast<std::uint32_t>(result.families.size());
            BoxScoreToolingSet::SequentialFamily family;
            family.scoreBodyIndex =
                kShippingSceneBodiesPerScoreFamily * familyIndex;
            family.supportBodyIndex = family.scoreBodyIndex + 1u;
            family.referenceCenter = {
                0.5 * (line.first.x + line.second.x),
                0.5 * (line.first.y + line.second.y), 0.0};
            family.tangent = {tx, ty, 0.0};
            family.normal = {-ty, tx, 0.0};
            family.lineLength = length;
            result.families.push_back(std::move(family));
        } else {
            familyIndex = static_cast<std::uint32_t>(
                std::distance(result.families.begin(), found));
        }
        result.scoreLineFamilyIndices[scoreIndex] = familyIndex;
        result.families[familyIndex].scoreLineIndices.push_back(scoreIndex);
    }
    if (result.families.size() != kShippingScoreToolFamilyCount)
        throw std::invalid_argument(
            "shipping score-line layout must reduce to three fixed-orientation families");
    result.scoreBodyIndex = result.families.front().scoreBodyIndex;
    result.supportBodyIndex = result.families.front().supportBodyIndex;
    result.scoreBodyPosition = result.families.front().referenceCenter;
    result.supportBodyPosition = result.families.front().referenceCenter;

    const double extension = punchOverhang - punchRadius;
    const double railHalfWidth = 0.5 * supportRailWidth;
    const double railOffset = 0.5 * channelWidth + railHalfWidth;
    result.proxies.reserve(kShippingScoreToolProxyCount);
    for (const auto& family : result.families) {
        const double halfCenterline =
            0.5 * family.lineLength + extension;
        if (!(halfCenterline > 0.0) || !std::isfinite(halfCenterline))
            throw std::invalid_argument(
                "sequential capsule cannot span its score-line family");
        numi::matter::RigidProxySource punch;
        punch.shape = NM_RIGID_CAPSULE;
        punch.bodyIndex = family.scoreBodyIndex;
        punch.materialIndex = 0u;
        punch.localCenter = {
            -halfCenterline * family.tangent.x,
            -halfCenterline * family.tangent.y, 0.0};
        punch.localExtent = {
            halfCenterline * family.tangent.x,
            halfCenterline * family.tangent.y, 0.0};
        punch.localOrientation = {0.0, 0.0, 0.0, 1.0};
        punch.radiusOrOffset = punchRadius;
        punch.articulated = false;
        punch.dynamic = false;
        punch.prescribedEndPoseTranslation = true;
        result.proxies.push_back(punch);

        const std::array<double, 4> railOrientation{
            0.0, 0.0,
            std::sin(0.5 * std::atan2(family.tangent.y,
                                      family.tangent.x)),
            std::cos(0.5 * std::atan2(family.tangent.y,
                                      family.tangent.x))};
        for (const double side : {-1.0, 1.0}) {
            numi::matter::RigidProxySource rail;
            rail.shape = NM_RIGID_BOX;
            rail.bodyIndex = family.supportBodyIndex;
            rail.materialIndex = 0u;
            rail.localCenter = {
                side * railOffset * family.normal.x,
                side * railOffset * family.normal.y,
                -anvilClearance - 0.5 * supportThickness};
            rail.localExtent = {
                0.5 * family.lineLength + anvilOverhang,
                railHalfWidth, 0.5 * supportThickness};
            rail.localOrientation = railOrientation;
            rail.articulated = false;
            rail.dynamic = false;
            // Shipping scoring relocates this BOX between score-line families.
            // Native admission must certify its prescribed translation.
            rail.prescribedEndPoseTranslation = true;
            result.proxies.push_back(rail);
        }
    }
    if (result.proxies.size() != kShippingScoreToolProxyCount)
        throw std::logic_error(
            "shipping score tooling proxy count differs from its shared layout");
    return result;
}

inline std::vector<MRBodyStateGPU> makeBoxToolBodyStates(
    const BoxScoreToolingSet& tooling,
    const std::uint32_t environmentCount,
    const double parkedHeight) {
    if (environmentCount == 0u || !std::isfinite(parkedHeight) ||
        parkedHeight <= tooling.boardCaliper + 2.0 * tooling.punchRadius)
        throw std::invalid_argument("box tool body-state inputs are invalid");
    const std::uint32_t bodyCount = tooling.sequentialFamilies
        ? static_cast<std::uint32_t>(tooling.families.size() *
              kShippingSceneBodiesPerScoreFamily) : 2u;
    std::vector<MRBodyStateGPU> result(
        static_cast<std::size_t>(environmentCount) * bodyCount);
    for (std::uint32_t environment = 0u;
         environment < environmentCount; ++environment) {
        const std::size_t base = static_cast<std::size_t>(environment) * bodyCount;
        const auto initializeBody = [](MRBodyStateGPU& body,
                                       const BoxVec3 position) {
            body.position = {static_cast<float>(position.x),
                static_cast<float>(position.y), static_cast<float>(position.z), 0.0f};
            body.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
            body.flagsAndIndices[0] = MR_MOTION_KINEMATIC;
            body.flagsAndIndices[1] = MR_INVALID_INDEX;
            body.flagsAndIndices[2] = MR_INVALID_INDEX;
            body.linearVelocityAndInverseMass = {0.0f, 0.0f, 0.0f, 0.0f};
        };
        if (tooling.sequentialFamilies) {
            for (const auto& family : tooling.families) {
                initializeBody(result[base + family.scoreBodyIndex],
                    {family.referenceCenter.x, family.referenceCenter.y,
                     parkedHeight});
                initializeBody(result[base + family.supportBodyIndex],
                    {family.referenceCenter.x, family.referenceCenter.y, -2.0});
            }
        } else {
            initializeBody(result[base + tooling.scoreBodyIndex],
                {tooling.scoreBodyPosition.x, tooling.scoreBodyPosition.y,
                 parkedHeight});
            initializeBody(result[base + tooling.supportBodyIndex],
                {tooling.supportBodyPosition.x, tooling.supportBodyPosition.y, 0.0});
        }
    }
    return result;
}

inline BoxVec3 boxScoreDieFirstContactPosition(
    const BoxScoreToolingSet& tooling) {
    return {tooling.scoreBodyPosition.x, tooling.scoreBodyPosition.y,
            tooling.boardCaliper + tooling.punchRadius};
}

inline BoxVec3 boxScoreDieParkPosition(
    const BoxScoreToolingSet& tooling,
    const double parkedHeight) {
    if (!std::isfinite(parkedHeight) ||
        parkedHeight <= tooling.boardCaliper + 2.0 * tooling.punchRadius)
        throw std::invalid_argument("box score-die park height is invalid");
    return {tooling.scoreBodyPosition.x, tooling.scoreBodyPosition.y,
            parkedHeight};
}

inline void setBoxToolBodyCommand(
    std::vector<MRBodyStateGPU>& bodies,
    const std::uint32_t bodyCountPerEnvironment,
    const std::uint32_t environment,
    const std::uint32_t bodyIndex,
    const BoxVec3 position,
    const BoxVec3 velocity) {
    if (bodyCountPerEnvironment == 0u || bodyIndex >= bodyCountPerEnvironment ||
        bodies.size() % bodyCountPerEnvironment != 0u ||
        environment >= bodies.size() / bodyCountPerEnvironment ||
        !std::isfinite(position.x) || !std::isfinite(position.y) ||
        !std::isfinite(position.z) || !std::isfinite(velocity.x) ||
        !std::isfinite(velocity.y) || !std::isfinite(velocity.z))
        throw std::invalid_argument("box rigid-tool command is invalid");
    MRBodyStateGPU& body = bodies[
        static_cast<std::size_t>(environment) * bodyCountPerEnvironment + bodyIndex];
    const double orientationNorm2 =
        static_cast<double>(body.orientation.x) * body.orientation.x +
        static_cast<double>(body.orientation.y) * body.orientation.y +
        static_cast<double>(body.orientation.z) * body.orientation.z +
        static_cast<double>(body.orientation.w) * body.orientation.w;
    if (!(std::isfinite(orientationNorm2) && orientationNorm2 > 1.0e-12))
        throw std::invalid_argument(
            "box rigid-tool command requires an existing valid fixed orientation");
    body.position = {static_cast<float>(position.x),
        static_cast<float>(position.y), static_cast<float>(position.z), 0.0f};
    body.flagsAndIndices[0] = MR_MOTION_KINEMATIC;
    body.linearVelocityAndInverseMass = {static_cast<float>(velocity.x),
        static_cast<float>(velocity.y), static_cast<float>(velocity.z), 0.0f};
}

inline std::uint32_t boxSequentialFamilyForScoreLine(
    const BoxScoreToolingSet& tooling, const std::uint32_t scoreLineIndex) {
    if (!tooling.sequentialFamilies ||
        scoreLineIndex >= tooling.scoreLineFamilyIndices.size())
        throw std::out_of_range("sequential score-line family lookup is invalid");
    return tooling.scoreLineFamilyIndices[scoreLineIndex];
}

inline BoxVec3 boxScoreLineCenter(
    const FEFCO0201Blank& blank, const std::uint32_t scoreLineIndex) {
    if (scoreLineIndex >= blank.scoreLines.size())
        throw std::out_of_range("score-line center lookup is invalid");
    const auto& line = blank.scoreLines[scoreLineIndex];
    return {0.5 * (line.first.x + line.second.x),
            0.5 * (line.first.y + line.second.y), 0.0};
}

struct BoxFeatureGrip {
    std::uint32_t assemblyActionIndex = 0u;
    std::uint32_t panelIndex = 0u;
    std::uint32_t scoreLineIndex = kNoBoxScoreLineIndex;
    std::vector<std::uint32_t> nodeIndices;
    struct ReferencePatch {
        // The patch is a selected exterior portion of the authored flat blank.
        // It records the kinematic clamp surface only; it does not represent a
        // rigid tool/contact model or a material crease.
        char planeAxis = 'z';
        double planeCoordinate = 0.0;
        double minimumX = 0.0;
        double maximumX = 0.0;
        double minimumY = 0.0;
        double maximumY = 0.0;
        std::uint32_t nodeCount = 0u;
        std::vector<std::uint32_t> nodeIndices;
    };
    std::vector<ReferencePatch> referencePatches;
};

inline std::vector<BoxFeatureGrip> makeBoxFeatureGrips(
    const FEFCO0201Blank& blank,
    const std::vector<BoxBlankPoint>& restNodes,
    const std::vector<std::uint32_t>& exposedBoundaryNodeIndices) {
    if (restNodes.size() != blank.points.size() || restNodes.empty() ||
        exposedBoundaryNodeIndices.empty())
        throw std::invalid_argument("box grip requires the compiled authored mesh and its exterior nodes");
    std::vector<std::uint8_t> isExterior(restNodes.size(), 0u);
    for (const std::uint32_t node : exposedBoundaryNodeIndices) {
        if (node >= restNodes.size() || isExterior[node] != 0u)
            throw std::invalid_argument("box grip exterior-node map is out of range or duplicated");
        isExterior[node] = 1u;
    }
    const double tolerance = 2.0e-10 *
        std::max({blank.perimeterSpan, blank.blankHeight, blank.boardCaliper});
    std::vector<BoxFeatureGrip> result;
    for (std::uint32_t actionIndex = 0u;
         actionIndex < blank.assembly.actions.size(); ++actionIndex) {
        const BoxAssemblyAction& action = blank.assembly.actions[actionIndex];
        BoxFeatureGrip grip;
        grip.assemblyActionIndex = actionIndex;
        bool hasGrip = false;
        if (action.kind == BoxAssemblyAction::Kind::erectWallPanel) {
            grip.panelIndex = action.feature.index;
            const BoxBlankPanel& panel = blank.panels.at(grip.panelIndex);
            const double panelWidth = panel.xEnd - panel.xStart;
            // A finite jaw patch sits in the distal upper half of the wall,
            // inboard from its score edges and reaching the upper cut edge.
            // Only the two exterior liner faces are clamped; interior
            // flute/glue nodes remain unknowns.
            const double patchX0 = panel.xEnd - 0.40 * panelWidth;
            const double patchX1 = panel.xEnd - 0.08 * panelWidth;
            const double patchY0 = panel.yEnd - 0.50 * (panel.yEnd - panel.yStart);
            const double patchY1 = panel.yEnd;
            std::array<BoxFeatureGrip::ReferencePatch, 2> surfacePatches{};
            surfacePatches[0].planeCoordinate = 0.0;
            surfacePatches[1].planeCoordinate = blank.boardCaliper;
            for (std::uint32_t node = 0u; node < restNodes.size(); ++node) {
                if (isExterior[node] == 0u) continue;
                const auto& p = restNodes[node];
                const bool inPatch = p.x >= patchX0 - tolerance &&
                    p.x <= patchX1 + tolerance &&
                    p.y >= patchY0 - tolerance && p.y <= patchY1 + tolerance;
                if (!inPatch) continue;
                const std::size_t surface =
                    std::abs(p.z) <= tolerance ? 0u :
                    std::abs(p.z - blank.boardCaliper) <= tolerance ? 1u : 2u;
                if (surface < surfacePatches.size()) {
                    grip.nodeIndices.push_back(node);
                    auto& reference = surfacePatches[surface];
                    if (reference.nodeCount == 0u) {
                        reference.minimumX = reference.maximumX = p.x;
                        reference.minimumY = reference.maximumY = p.y;
                    } else {
                        reference.minimumX = std::min(reference.minimumX, p.x);
                        reference.maximumX = std::max(reference.maximumX, p.x);
                        reference.minimumY = std::min(reference.minimumY, p.y);
                        reference.maximumY = std::max(reference.maximumY, p.y);
                    }
                    reference.nodeIndices.push_back(node);
                    ++reference.nodeCount;
                }
            }
            for (const auto& patch : surfacePatches)
                if (patch.nodeCount > 0u) grip.referencePatches.push_back(patch);
            grip.scoreLineIndex = action.hingeScoreLineIndex;
            hasGrip = true;
        } else if (action.kind == BoxAssemblyAction::Kind::foldFlap) {
            grip.scoreLineIndex = action.feature.index;
            const BoxScoreLine& score = blank.scoreLines.at(grip.scoreLineIndex);
            grip.panelIndex = score.panelIndex;
            const BoxBlankPanel& panel = blank.panels.at(grip.panelIndex);
            const bool top = score.kind == BoxScoreKind::topFlap;
            const double freeY = top ? blank.blankHeight : 0.0;
            BoxFeatureGrip::ReferencePatch patch;
            patch.planeAxis = 'y';
            patch.planeCoordinate = freeY;
            patch.minimumX = panel.xStart;
            patch.maximumX = panel.xEnd;
            patch.minimumY = patch.maximumY = freeY;
            for (std::uint32_t node = 0u; node < restNodes.size(); ++node) {
                if (isExterior[node] == 0u) continue;
                const auto& p = restNodes[node];
                if (std::abs(p.y - freeY) <= tolerance &&
                    p.x >= panel.xStart - tolerance &&
                    p.x <= panel.xEnd + tolerance)
                    grip.nodeIndices.push_back(node);
            }
            patch.nodeCount = static_cast<std::uint32_t>(grip.nodeIndices.size());
            patch.nodeIndices = grip.nodeIndices;
            if (patch.nodeCount > 0u) grip.referencePatches.push_back(patch);
            hasGrip = true;
        }
        if (!hasGrip) continue; // seam/end actions are observed, not bonded.
        if (grip.nodeIndices.empty())
            throw std::invalid_argument("box boundary action has no authored grip nodes");
        result.push_back(std::move(grip));
    }
    return result;
}

// restAndFixed.w == 3 is the temporary boundary-grip tag. Tag 1 is an
// authored immutable Dirichlet condition and is deliberately not counted as
// an acquired clamp. Validate the complete accepted FEM state before using
// this as a release/checkpoint postcondition.
inline std::uint32_t countActiveBoxTemporaryGripNodes(
    const std::vector<NMFEMNodeStateGPU>& acceptedNodes,
    const std::uint32_t environmentCount,
    const std::uint32_t nodeCount) {
    if (environmentCount == 0u || nodeCount == 0u ||
        acceptedNodes.size() !=
            static_cast<std::size_t>(environmentCount) * nodeCount)
        throw std::invalid_argument("accepted FEM state has the wrong shape for grip-release verification");
    std::uint32_t active = 0u;
    for (const auto& node : acceptedNodes) {
        const float tag = node.restAndFixed.w;
        if (!std::isfinite(tag) ||
            (tag != 0.0f && tag != 1.0f && tag != 2.0f && tag != 3.0f))
            throw std::invalid_argument("accepted FEM state contains an unknown constraint tag");
        if (tag == 3.0f) ++active;
    }
    return active;
}

// A completed released shipping blank has no permanent prescribed FEM nodes:
// every score, fold, and clamp actuator must have relinquished its node tags.
// This is intentionally stronger than the temporary-grip counter above,
// which remains useful while a fixture is still being assembled.
inline std::uint32_t countActiveBoxPrescribedNodes(
    const std::vector<NMFEMNodeStateGPU>& acceptedNodes,
    const std::uint32_t environmentCount,
    const std::uint32_t nodeCount) {
    if (environmentCount == 0u || nodeCount == 0u ||
        acceptedNodes.size() !=
            static_cast<std::size_t>(environmentCount) * nodeCount)
        throw std::invalid_argument(
            "accepted FEM state has the wrong shape for released-node verification");
    std::uint32_t active = 0u;
    for (const auto& node : acceptedNodes) {
        const float tag = node.restAndFixed.w;
        if (!std::isfinite(tag) ||
            (tag != 0.0f && tag != 1.0f && tag != 2.0f && tag != 3.0f))
            throw std::invalid_argument(
                "accepted FEM state contains an unknown constraint tag");
        if (tag > 0.5f) ++active;
    }
    return active;
}

// Select exposed boundary nodes on the manufacturer tab body, leaving a
// 15-percent margin next to its score root. This is an ideal external frame
// clamp at the accepted pose; it is not a material hinge or a prescribed
// folded mesh. The seam-side free edge remains in the clamp until wall
// boundary grips are established, then the driver releases this set before
// moving that edge with the external joint fixture.
inline std::vector<std::uint32_t> makeBoxTabBodyAnchorNodes(
    const FEFCO0201Blank& blank,
    const std::vector<BoxBlankPoint>& restNodes,
    const std::vector<std::uint32_t>& exposedBoundaryNodeIndices) {
    if (restNodes.empty() || !(blank.config.manufacturerJointWidth > 0.0))
        throw std::invalid_argument("box tab anchor input is invalid");
    const double tolerance = 2.0e-10 *
        std::max({blank.perimeterSpan, blank.blankHeight, blank.boardCaliper});
    const double tabEnd = 0.85 * blank.config.manufacturerJointWidth;
    std::vector<std::uint32_t> result;
    result.reserve(exposedBoundaryNodeIndices.size());
    std::vector<std::uint32_t> candidates = exposedBoundaryNodeIndices;
    std::sort(candidates.begin(), candidates.end());
    if (std::adjacent_find(candidates.begin(), candidates.end()) != candidates.end())
        throw std::invalid_argument("box tab anchor boundary node list has duplicates");
    for (const std::uint32_t node : candidates) {
        if (node >= restNodes.size())
            throw std::invalid_argument("box tab anchor boundary node is out of range");
        const BoxBlankPoint& point = restNodes[node];
        if (point.x < -tolerance || point.x > tabEnd + tolerance ||
            point.y < blank.bottomScoreY - tolerance ||
            point.y > blank.topScoreY + tolerance)
            continue;
        result.push_back(node);
    }
    if (result.size() < 3u)
        throw std::invalid_argument("box tab body has too few exposed anchor nodes");
    return result;
}

inline void setBoxGripTarget(
    std::vector<nm_float4>& targets,
    const std::uint32_t nodeCount,
    const std::uint32_t environment,
    const std::uint32_t node,
    const BoxVec3 target,
    const float command = 3.0f) {
    if (nodeCount == 0u || node >= nodeCount || targets.size() % nodeCount != 0u ||
        environment >= targets.size() / nodeCount ||
        (command != 2.0f && command != 3.0f) ||
        !std::isfinite(target.x) || !std::isfinite(target.y) ||
        !std::isfinite(target.z))
        throw std::invalid_argument("box grip target is invalid");
    const std::size_t index =
        static_cast<std::size_t>(environment) * nodeCount + node;
    targets[index] = {static_cast<float>(target.x),
        static_cast<float>(target.y), static_cast<float>(target.z), command};
}

struct BoxWallFit {
    bool valid = false;
    double angleDegrees = 0.0;
    double fitResidualRms = 0.0;
    double fitCoverage = 0.0;
    std::uint32_t sampleCount = 0u;
};

inline BoxWallFit measureBoxMaterialAxisFit(
    const std::vector<BoxBlankPoint>& restNodes,
    const std::vector<NMFEMNodeStateGPU>& acceptedNodes,
    const std::uint32_t environment,
    const std::uint32_t nodeCount,
    const double xStart,
    const double xEnd,
    const double yStart,
    const double yEnd,
    const double residualBound) {
    // The Procrustes angle is restricted to the blank's x-z section, matching
    // the current fixed-y-axis wall tool path. The residual additionally
    // checks current-y against its translated authored coordinate, but this
    // is not a general 3D panel-orientation fit.
    if (acceptedNodes.size() <
        static_cast<std::size_t>(environment + 1u) * nodeCount ||
        restNodes.size() != nodeCount || nodeCount == 0u ||
        !std::isfinite(xStart) || !std::isfinite(xEnd) ||
        !std::isfinite(yStart) || !std::isfinite(yEnd) ||
        !(xEnd > xStart) || !(yEnd >= yStart) ||
        !std::isfinite(residualBound) || !(residualBound > 0.0))
        throw std::invalid_argument("box accepted-state panel fit input is invalid");
    double meanRestX = 0.0, meanRestY = 0.0, meanRestZ = 0.0;
    double meanX = 0.0, meanY = 0.0, meanZ = 0.0;
    std::uint32_t count = 0u;
    for (std::uint32_t node = 0u; node < nodeCount; ++node) {
        const auto& rest = restNodes[node];
        if (rest.x <= xStart || rest.x >= xEnd ||
            rest.y < yStart || rest.y > yEnd)
            continue;
        const BoxVec3 current = boxAcceptedPosition(
            acceptedNodes[static_cast<std::size_t>(environment) * nodeCount + node]);
        meanRestX += rest.x;
        meanRestY += rest.y;
        meanRestZ += rest.z;
        meanX += current.x;
        meanY += current.y;
        meanZ += current.z;
        ++count;
    }
    if (count < 3u) return {};
    meanRestX /= count;
    meanRestY /= count;
    meanRestZ /= count;
    meanX /= count;
    meanY /= count;
    meanZ /= count;
    double cosineMoment = 0.0, sineMoment = 0.0;
    for (std::uint32_t node = 0u; node < nodeCount; ++node) {
        const auto& rest = restNodes[node];
        if (rest.x <= xStart || rest.x >= xEnd ||
            rest.y < yStart || rest.y > yEnd)
            continue;
        const BoxVec3 current = boxAcceptedPosition(
            acceptedNodes[static_cast<std::size_t>(environment) * nodeCount + node]);
        const double restX = rest.x - meanRestX;
        const double restZ = rest.z - meanRestZ;
        const double currentX = current.x - meanX;
        const double currentZ = current.z - meanZ;
        cosineMoment += restX * currentX + restZ * currentZ;
        sineMoment += restX * currentZ - restZ * currentX;
    }
    if (!(std::hypot(cosineMoment, sineMoment) > 1.0e-24)) return {};
    const double angle = std::atan2(-sineMoment, cosineMoment);
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    double residual2 = 0.0;
    std::uint32_t covered = 0u;
    for (std::uint32_t node = 0u; node < nodeCount; ++node) {
        const auto& rest = restNodes[node];
        if (rest.x <= xStart || rest.x >= xEnd ||
            rest.y < yStart || rest.y > yEnd)
            continue;
        const BoxVec3 current = boxAcceptedPosition(
            acceptedNodes[static_cast<std::size_t>(environment) * nodeCount + node]);
        const double restX = rest.x - meanRestX;
        const double restZ = rest.z - meanRestZ;
        const double predictionX = meanX + cosine * restX + sine * restZ;
        const double predictionY = meanY + (rest.y - meanRestY);
        const double predictionZ = meanZ - sine * restX + cosine * restZ;
        const double rx = current.x - predictionX;
        const double ry = current.y - predictionY;
        const double rz = current.z - predictionZ;
        const double residual = std::sqrt(rx * rx + ry * ry + rz * rz);
        if (!std::isfinite(residual)) return {};
        residual2 += residual * residual;
        if (residual <= residualBound) ++covered;
    }
    return {std::isfinite(angle),
        angle * 180.0 / 3.14159265358979323846,
        std::sqrt(residual2 / count),
        static_cast<double>(covered) / count, count};
}

inline BoxWallFit measureBoxWallInteriorFit(
    const FEFCO0201Blank& blank,
    const std::vector<BoxBlankPoint>& restNodes,
    const std::vector<NMFEMNodeStateGPU>& acceptedNodes,
    const std::uint32_t environment,
    const std::uint32_t nodeCount,
    const std::uint32_t panelIndex,
    const double residualBoundCalipers = kDefaultBoxShapeResidualBoundCalipers) {
    if (panelIndex >= blank.panels.size())
        throw std::invalid_argument("box accepted-state panel fit index is invalid");
    const BoxBlankPanel& panel = blank.panels[panelIndex];
    const double exclusion = 0.15 * (panel.xEnd - panel.xStart);
    if (!std::isfinite(residualBoundCalipers) || !(residualBoundCalipers > 0.0))
        return {};
    return measureBoxMaterialAxisFit(restNodes, acceptedNodes,
        environment, nodeCount, panel.xStart + exclusion,
        panel.xEnd - exclusion, panel.yStart, panel.yEnd,
        residualBoundCalipers * blank.boardCaliper);
}

inline BoxWallFit measureBoxTabInteriorFit(
    const FEFCO0201Blank& blank,
    const std::vector<BoxBlankPoint>& restNodes,
    const std::vector<NMFEMNodeStateGPU>& acceptedNodes,
    const std::uint32_t environment,
    const std::uint32_t nodeCount,
    const double residualBoundCalipers = kDefaultBoxShapeResidualBoundCalipers) {
    const double exclusion = 0.15 * blank.config.manufacturerJointWidth;
    if (!std::isfinite(residualBoundCalipers) || !(residualBoundCalipers > 0.0))
        return {};
    return measureBoxMaterialAxisFit(restNodes, acceptedNodes,
        environment, nodeCount, exclusion,
        blank.config.manufacturerJointWidth - exclusion,
        blank.bottomScoreY, blank.topScoreY,
        residualBoundCalipers * blank.boardCaliper);
}

struct BoxFlapTurnFit {
    bool valid = false;
    double signedTurnDegrees = 0.0;
    double fitResidualRms = 0.0;
    double fitCoverage = 0.0;
    double residualBound = 0.0;
    std::uint32_t sampleCount = 0u;
};

inline BoxVec3 boxRotateVectorBetweenAxes(const BoxVec3 value,
                                          const BoxVec3 from,
                                          const BoxVec3 to) {
    const double cosine = std::clamp(boxDot(from, to), -1.0, 1.0);
    const BoxVec3 cross = boxCross(from, to);
    const double sine = boxLength(cross);
    if (sine <= 1.0e-12) {
        if (cosine > 0.0) return value;
        const BoxVec3 basis = std::abs(from.x) < 0.8
            ? BoxVec3{1.0, 0.0, 0.0} : BoxVec3{0.0, 1.0, 0.0};
        const BoxVec3 axis = boxNormalized(boxCross(from, basis));
        return -1.0 * value + 2.0 * boxDot(axis, value) * axis;
    }
    const BoxVec3 axis = (1.0 / sine) * cross;
    const double angle = std::atan2(sine, cosine);
    return std::cos(angle) * value + std::sin(angle) * boxCross(axis, value) +
        (1.0 - std::cos(angle)) * boxDot(axis, value) * axis;
}

inline BoxFlapTurnFit measureBoxFlapInteriorTurn(
    const FEFCO0201Blank& blank,
    const std::vector<BoxBlankPoint>& restNodes,
    const std::vector<NMFEMNodeStateGPU>& actionStartNodes,
    const std::vector<NMFEMNodeStateGPU>& acceptedNodes,
    std::uint32_t environment,
    std::uint32_t nodeCount,
    std::uint32_t scoreLineIndex,
    double residualBoundCalipers = kDefaultBoxShapeResidualBoundCalipers);

inline double boxWallTargetOrientationDegrees(std::uint32_t panelIndex);

inline double wrapBoxAngleDegrees(double degrees) {
    degrees = std::fmod(degrees + 180.0, 360.0);
    if (degrees < 0.0) degrees += 360.0;
    return degrees - 180.0;
}

inline double boxWallRelativeTargetTurnDegrees(
    const std::uint32_t panelIndex) {
    const double current = boxWallTargetOrientationDegrees(panelIndex);
    const double previous = panelIndex == 0u ? 0.0 :
        boxWallTargetOrientationDegrees(panelIndex - 1u);
    return wrapBoxAngleDegrees(current - previous);
}

struct BoxRelativeWallTurnFit {
    bool valid = false;
    double signedTurnDegrees = 0.0;
    double targetTurnDegrees = 0.0;
    double fitResidualRms = 0.0;
    double fitCoverage = 0.0;
    double residualBound = 0.0;
    std::uint32_t sampleCount = 0u;
};

inline BoxRelativeWallTurnFit measureBoxRelativeWallTurn(
    const FEFCO0201Blank& blank,
    const std::vector<BoxBlankPoint>& restNodes,
    const std::vector<NMFEMNodeStateGPU>& acceptedNodes,
    const std::uint32_t environment,
    const std::uint32_t nodeCount,
    const std::uint32_t panelIndex,
    const double residualBoundCalipers = kDefaultBoxShapeResidualBoundCalipers) {
    if (panelIndex >= blank.panels.size()) return {};
    const BoxWallFit current = measureBoxWallInteriorFit(
        blank, restNodes, acceptedNodes, environment, nodeCount, panelIndex,
        residualBoundCalipers);
    const BoxWallFit reference = panelIndex == 0u
        ? measureBoxTabInteriorFit(blank, restNodes, acceptedNodes,
                                   environment, nodeCount,
                                   residualBoundCalipers)
        : measureBoxWallInteriorFit(blank, restNodes, acceptedNodes,
                                    environment, nodeCount, panelIndex - 1u,
                                    residualBoundCalipers);
    if (!current.valid || !reference.valid) return {};
    const std::uint32_t combinedCount = current.sampleCount + reference.sampleCount;
    if (combinedCount == 0u) return {};
    // Both adjacent panels are required references for this relative angle.
    // Gate on the worse fit rather than allowing a large, good panel to mask
    // a small, warped parent tab in a sample-weighted aggregate.
    return {true,
        wrapBoxAngleDegrees(current.angleDegrees - reference.angleDegrees),
        boxWallRelativeTargetTurnDegrees(panelIndex),
        std::max(current.fitResidualRms, reference.fitResidualRms),
        std::min(current.fitCoverage, reference.fitCoverage),
        residualBoundCalipers * blank.boardCaliper,
        combinedCount};
}

struct BoxHingeFrame {
    bool valid = false;
    BoxVec3 origin{};
    BoxVec3 axis{};
    std::uint32_t sampleCount = 0u;
};

inline BoxHingeFrame measureBoxAcceptedHingeFrame(
    const FEFCO0201Blank& blank,
    const std::vector<BoxBlankPoint>& restNodes,
    const std::vector<NMFEMNodeStateGPU>& acceptedNodes,
    const std::uint32_t environment,
    const std::uint32_t nodeCount,
    const std::uint32_t scoreLineIndex) {
    if (scoreLineIndex >= blank.scoreLines.size() ||
        restNodes.size() != nodeCount || nodeCount == 0u ||
        acceptedNodes.size() <
            static_cast<std::size_t>(environment + 1u) * nodeCount)
        return {};
    const BoxScoreLine& line = blank.scoreLines[scoreLineIndex];
    const double dx = line.second.x - line.first.x;
    const double dy = line.second.y - line.first.y;
    const double length2 = dx * dx + dy * dy;
    if (!(length2 > 1.0e-24)) return {};
    const double tolerance = 2.0e-10 *
        std::max({blank.perimeterSpan, blank.blankHeight, blank.boardCaliper});
    BoxVec3 origin{};
    BoxVec3 firstEnd{}, secondEnd{};
    std::uint32_t lineCount = 0u, firstCount = 0u, secondCount = 0u;
    for (std::uint32_t node = 0u; node < nodeCount; ++node) {
        const BoxBlankPoint& rest = restNodes[node];
        const double projection = std::clamp(
            ((rest.x - line.first.x) * dx +
             (rest.y - line.first.y) * dy) / length2, 0.0, 1.0);
        const double nearestX = line.first.x + projection * dx;
        const double nearestY = line.first.y + projection * dy;
        if (std::hypot(rest.x - nearestX, rest.y - nearestY) > tolerance)
            continue;
        const BoxVec3 current = boxAcceptedPosition(acceptedNodes[
            static_cast<std::size_t>(environment) * nodeCount + node]);
        origin = origin + current;
        ++lineCount;
        if (std::hypot(rest.x - line.first.x, rest.y - line.first.y) <= tolerance) {
            firstEnd = firstEnd + current;
            ++firstCount;
        }
        if (std::hypot(rest.x - line.second.x, rest.y - line.second.y) <= tolerance) {
            secondEnd = secondEnd + current;
            ++secondCount;
        }
    }
    if (lineCount == 0u || firstCount == 0u || secondCount == 0u) return {};
    origin = (1.0 / lineCount) * origin;
    firstEnd = (1.0 / firstCount) * firstEnd;
    secondEnd = (1.0 / secondCount) * secondEnd;
    const BoxVec3 axisVector = secondEnd - firstEnd;
    if (!(boxLength(axisVector) > tolerance)) return {};
    return {true, origin, boxNormalized(axisVector), lineCount};
}

struct BoxGripNodeTarget {
    std::uint32_t node = 0u;
    BoxVec3 endpoint{};
    float command = 0.0f;
};

struct BoxAssemblyActionStep {
    enum class Phase : std::uint32_t {
        invalid = 0u,
        moveGrip = 1u,
        releaseGrip = 2u,
        acceptedContactOnly = 3u,
    };
    bool valid = false;
    bool angleObservable = false;
    bool physicalBondLawRequired = false;
    bool shapeFitRejected = false;
    bool shapeFitWithinBound = false;
    Phase phase = Phase::invalid;
    std::uint32_t assemblyActionIndex = 0u;
    double observedAngleDegrees = 0.0;
    double targetAngleDegrees = 0.0;
    double remainingAngleDegrees = 0.0;
    double commandedIncrementDegrees = 0.0;
    double fitResidualRms = 0.0;
    double fitCoverage = 0.0;
    double shapeFitResidualBound = 0.0;
    double shapeFitCoverageMinimum = kDefaultBoxShapeFitCoverage;
    std::uint32_t fitSampleCount = 0u;
    BoxHingeFrame hinge{};
    std::vector<BoxGripNodeTarget> nodeTargets;
};

// Recompute this plan only from an accepted FEM snapshot. The returned w=3
// targets prescribe a boundary grip trajectory while interior tetrahedra
// remain solver unknowns. `actionStartNodes` is the accepted state captured
// immediately before one flap-fold action; wall fits use the authored blank
// reference and are independent of a host-folded target pose.
inline BoxAssemblyActionStep planBoxAssemblyActionStep(
    const FEFCO0201Blank& blank,
    const std::vector<BoxBlankPoint>& restNodes,
    const std::vector<std::uint32_t>& exposedBoundaryNodeIndices,
    const std::vector<NMFEMNodeStateGPU>& acceptedNodes,
    const std::vector<NMFEMNodeStateGPU>& actionStartNodes,
    const std::uint32_t environment,
    const std::uint32_t nodeCount,
    const std::uint32_t assemblyActionIndex,
    const double maximumIncrementDegrees,
    const double completionToleranceDegrees = 0.5,
    const double shapeResidualBoundCalipers =
        kDefaultBoxShapeResidualBoundCalipers,
    const double minimumShapeFitCoverage = kDefaultBoxShapeFitCoverage) {
    if (assemblyActionIndex >= blank.assembly.actions.size() ||
        restNodes.size() != nodeCount || nodeCount == 0u ||
        acceptedNodes.size() <
            static_cast<std::size_t>(environment + 1u) * nodeCount ||
        !std::isfinite(maximumIncrementDegrees) ||
        !(maximumIncrementDegrees > 0.0) ||
        !std::isfinite(completionToleranceDegrees) ||
        !(completionToleranceDegrees > 0.0) ||
        !std::isfinite(shapeResidualBoundCalipers) ||
        !(shapeResidualBoundCalipers > 0.0) ||
        !std::isfinite(minimumShapeFitCoverage) ||
        minimumShapeFitCoverage <= 0.0 || minimumShapeFitCoverage > 1.0)
        throw std::invalid_argument("box assembly action planning inputs are invalid");

    BoxAssemblyActionStep result;
    result.assemblyActionIndex = assemblyActionIndex;
    const BoxAssemblyAction& action = blank.assembly.actions[assemblyActionIndex];
    result.physicalBondLawRequired =
        action.kind == BoxAssemblyAction::Kind::bondManufacturerJoint ||
        action.kind == BoxAssemblyAction::Kind::closeEndWithTapeOrAdhesive;
    result.shapeFitResidualBound =
        shapeResidualBoundCalipers * blank.boardCaliper;
    result.shapeFitCoverageMinimum = minimumShapeFitCoverage;

    const auto grips = makeBoxFeatureGrips(
        blank, restNodes, exposedBoundaryNodeIndices);
    const auto stableGrip = std::find_if(
        grips.begin(), grips.end(), [assemblyActionIndex](const BoxFeatureGrip& grip) {
            return grip.assemblyActionIndex == assemblyActionIndex;
        });
    if (stableGrip == grips.end()) {
        result.valid = true;
        result.phase = BoxAssemblyActionStep::Phase::acceptedContactOnly;
        return result;
    }

    double observedAngle = 0.0;
    double targetAngle = 0.0;
    if (action.kind == BoxAssemblyAction::Kind::erectWallPanel) {
        const BoxRelativeWallTurnFit fit = measureBoxRelativeWallTurn(
            blank, restNodes, acceptedNodes, environment, nodeCount,
            action.feature.index, shapeResidualBoundCalipers);
        if (!fit.valid) return result;
        observedAngle = fit.signedTurnDegrees;
        targetAngle = fit.targetTurnDegrees;
        result.fitResidualRms = fit.fitResidualRms;
        result.fitCoverage = fit.fitCoverage;
        result.fitSampleCount = fit.sampleCount;
    } else if (action.kind == BoxAssemblyAction::Kind::foldFlap) {
        if (actionStartNodes.size() <
            static_cast<std::size_t>(environment + 1u) * nodeCount)
            return result;
        const BoxFlapTurnFit fit = measureBoxFlapInteriorTurn(
            blank, restNodes, actionStartNodes, acceptedNodes, environment,
            nodeCount, action.feature.index, shapeResidualBoundCalipers);
        if (!fit.valid) return result;
        observedAngle = fit.signedTurnDegrees;
        targetAngle = action.targetAngleDegrees;
        result.fitResidualRms = fit.fitResidualRms;
        result.fitCoverage = fit.fitCoverage;
        result.shapeFitResidualBound = fit.residualBound;
        result.fitSampleCount = fit.sampleCount;
    } else {
        result.valid = true;
        result.phase = BoxAssemblyActionStep::Phase::acceptedContactOnly;
        return result;
    }

    result.angleObservable = true;
    result.observedAngleDegrees = observedAngle;
    result.targetAngleDegrees = targetAngle;
    const double error = wrapBoxAngleDegrees(targetAngle - observedAngle);
    result.remainingAngleDegrees = error;
    result.shapeFitWithinBound =
        std::isfinite(result.fitResidualRms) &&
        result.fitResidualRms <= result.shapeFitResidualBound &&
        std::isfinite(result.fitCoverage) &&
        result.fitCoverage >= result.shapeFitCoverageMinimum;
    const bool gripIsFullyActive = std::all_of(
        stableGrip->nodeIndices.begin(), stableGrip->nodeIndices.end(),
        [&](const std::uint32_t node) {
            return acceptedNodes[static_cast<std::size_t>(environment) * nodeCount + node]
                .restAndFixed.w == 3.0f;
        });
    const bool gripHasActiveNodes = std::any_of(
        stableGrip->nodeIndices.begin(), stableGrip->nodeIndices.end(),
        [&](const std::uint32_t node) {
            return acceptedNodes[static_cast<std::size_t>(environment) * nodeCount + node]
                .restAndFixed.w == 3.0f;
        });
    for (const std::uint32_t node : stableGrip->nodeIndices) {
        const float tag = acceptedNodes[
            static_cast<std::size_t>(environment) * nodeCount + node]
            .restAndFixed.w;
        if (tag != 0.0f && tag != 3.0f) return result;
    }
    if (std::abs(error) <= completionToleranceDegrees) {
        if (!result.shapeFitWithinBound) {
            result.shapeFitRejected = true;
            return result;
        }
        result.valid = true;
        const bool retainWallGrip =
            action.kind == BoxAssemblyAction::Kind::erectWallPanel;
        result.phase = retainWallGrip && gripIsFullyActive
            ? BoxAssemblyActionStep::Phase::acceptedContactOnly
            : (retainWallGrip || gripHasActiveNodes
                ? BoxAssemblyActionStep::Phase::moveGrip
                : BoxAssemblyActionStep::Phase::acceptedContactOnly);
        if (retainWallGrip && !gripIsFullyActive) {
            for (const std::uint32_t node : stableGrip->nodeIndices) {
                const auto& state = acceptedNodes[
                    static_cast<std::size_t>(environment) * nodeCount + node];
                result.nodeTargets.push_back({node,
                    boxAcceptedPosition(state), 3.0f});
            }
        } else if (!retainWallGrip && gripHasActiveNodes) {
            result.phase = BoxAssemblyActionStep::Phase::releaseGrip;
            for (const std::uint32_t node : stableGrip->nodeIndices) {
                const auto& state = acceptedNodes[
                    static_cast<std::size_t>(environment) * nodeCount + node];
                if (state.restAndFixed.w == 3.0f)
                    result.nodeTargets.push_back({node,
                        boxAcceptedPosition(state), 2.0f});
            }
        }
        return result;
    }

    result.hinge = measureBoxAcceptedHingeFrame(
        blank, restNodes, acceptedNodes, environment, nodeCount,
        stableGrip->scoreLineIndex);
    if (!result.hinge.valid) return result;
    const double increment = std::copysign(
        std::min(std::abs(error), maximumIncrementDegrees), error);
    result.commandedIncrementDegrees = increment;
    const double radians = increment * 3.14159265358979323846 / 180.0;
    result.nodeTargets.reserve(stableGrip->nodeIndices.size());
    for (const std::uint32_t node : stableGrip->nodeIndices) {
        const BoxVec3 current = boxAcceptedPosition(acceptedNodes[
            static_cast<std::size_t>(environment) * nodeCount + node]);
        result.nodeTargets.push_back({node,
            boxRotateAboutAxis(current, result.hinge.origin,
                               result.hinge.axis, radians), 3.0f});
    }
    result.valid = !result.nodeTargets.empty();
    result.phase = BoxAssemblyActionStep::Phase::moveGrip;
    return result;
}

inline BoxFlapTurnFit measureBoxFlapInteriorTurn(
    const FEFCO0201Blank& blank,
    const std::vector<BoxBlankPoint>& restNodes,
    const std::vector<NMFEMNodeStateGPU>& actionStartNodes,
    const std::vector<NMFEMNodeStateGPU>& acceptedNodes,
    const std::uint32_t environment,
    const std::uint32_t nodeCount,
    const std::uint32_t scoreLineIndex,
    const double residualBoundCalipers) {
    if (scoreLineIndex >= blank.scoreLines.size() ||
        restNodes.size() != nodeCount ||
        actionStartNodes.size() <
            static_cast<std::size_t>(environment + 1u) * nodeCount ||
        acceptedNodes.size() <
            static_cast<std::size_t>(environment + 1u) * nodeCount)
        return {};
    const BoxScoreLine& score = blank.scoreLines[scoreLineIndex];
    if (score.kind != BoxScoreKind::bottomFlap &&
        score.kind != BoxScoreKind::topFlap)
        return {};
    if (!std::isfinite(residualBoundCalipers) ||
        !(residualBoundCalipers > 0.0) ||
        !std::isfinite(blank.boardCaliper) || !(blank.boardCaliper > 0.0))
        return {};
    const BoxBlankPanel& panel = blank.panels.at(score.panelIndex);
    const double xMargin = 0.15 * (panel.xEnd - panel.xStart);
    const double yMargin = 0.15 * blank.topFlapDepth;
    const double tolerance = 2.0e-10 *
        std::max({blank.perimeterSpan, blank.blankHeight, blank.boardCaliper});
    const auto lineDistance = [&](const BoxBlankPoint& p) {
        const double dx = score.second.x - score.first.x;
        const double dy = score.second.y - score.first.y;
        const double length2 = dx * dx + dy * dy;
        const double t = std::clamp(
            ((p.x - score.first.x) * dx + (p.y - score.first.y) * dy) /
                length2, 0.0, 1.0);
        return std::hypot(p.x - (score.first.x + t * dx),
                          p.y - (score.first.y + t * dy));
    };
    BoxVec3 startHinge{}, endHinge{};
    BoxVec3 startFirst{}, startSecond{}, endFirst{}, endSecond{};
    std::uint32_t hingeCount = 0u, firstCount = 0u, secondCount = 0u;
    BoxVec3 startFlap{}, endFlap{};
    std::vector<std::pair<BoxVec3, BoxVec3>> interiorPairs;
    std::uint32_t flapCount = 0u;
    const bool top = score.kind == BoxScoreKind::topFlap;
    for (std::uint32_t node = 0u; node < nodeCount; ++node) {
        const auto& rest = restNodes[node];
        const auto& startState = actionStartNodes[
            static_cast<std::size_t>(environment) * nodeCount + node];
        const auto& endState = acceptedNodes[
            static_cast<std::size_t>(environment) * nodeCount + node];
        const BoxVec3 startPosition = boxAcceptedPosition(startState);
        const BoxVec3 endPosition = boxAcceptedPosition(endState);
        if (lineDistance(rest) <= tolerance) {
            startHinge = startHinge + startPosition;
            endHinge = endHinge + endPosition;
            ++hingeCount;
            if (std::hypot(rest.x - score.first.x, rest.y - score.first.y) <= tolerance) {
                startFirst = startFirst + startPosition;
                endFirst = endFirst + endPosition;
                ++firstCount;
            }
            if (std::hypot(rest.x - score.second.x, rest.y - score.second.y) <= tolerance) {
                startSecond = startSecond + startPosition;
                endSecond = endSecond + endPosition;
                ++secondCount;
            }
        }
        const double freeY = top ? blank.blankHeight : 0.0;
        const bool interiorFlap =
            rest.x > panel.xStart + xMargin &&
            rest.x < panel.xEnd - xMargin &&
            (top ? (rest.y > blank.topScoreY + yMargin &&
                    rest.y < freeY - yMargin)
                 : (rest.y > freeY + yMargin &&
                    rest.y < blank.bottomScoreY - yMargin));
        if (interiorFlap) {
            startFlap = startFlap + startPosition;
            endFlap = endFlap + endPosition;
            interiorPairs.emplace_back(startPosition, endPosition);
            ++flapCount;
        }
    }
    if (hingeCount == 0u || firstCount == 0u || secondCount == 0u ||
        flapCount == 0u) return {};
    startHinge = (1.0 / hingeCount) * startHinge;
    endHinge = (1.0 / hingeCount) * endHinge;
    const BoxVec3 axisStart = boxNormalized(
        (1.0 / secondCount) * startSecond -
        (1.0 / firstCount) * startFirst);
    const BoxVec3 axisEnd = boxNormalized(
        (1.0 / secondCount) * endSecond -
        (1.0 / firstCount) * endFirst);
    const BoxVec3 axis = boxNormalized(axisStart + axisEnd);
    startFlap = (1.0 / flapCount) * startFlap - startHinge;
    endFlap = (1.0 / flapCount) * endFlap - endHinge;
    startFlap = startFlap - boxDot(startFlap, axis) * axis;
    endFlap = endFlap - boxDot(endFlap, axis) * axis;
    startFlap = boxNormalized(startFlap);
    endFlap = boxNormalized(endFlap);
    const double angle = std::atan2(
        boxDot(axis, boxCross(startFlap, endFlap)),
        std::clamp(boxDot(startFlap, endFlap), -1.0, 1.0));
    if (!std::isfinite(angle) || interiorPairs.size() != flapCount) return {};
    const double residualBound = residualBoundCalipers * blank.boardCaliper;
    double residual2 = 0.0;
    std::uint32_t covered = 0u;
    for (const auto& [startPosition, endPosition] : interiorPairs) {
        const BoxVec3 startRelative = startPosition - startHinge;
        const double axial = boxDot(startRelative, axisStart);
        const BoxVec3 startRadial = startRelative - axial * axisStart;
        const BoxVec3 alignedRadial = boxRotateVectorBetweenAxes(
            startRadial, axisStart, axisEnd);
        const BoxVec3 expectedRadial =
            std::cos(angle) * alignedRadial +
            std::sin(angle) * boxCross(axisEnd, alignedRadial) +
            (1.0 - std::cos(angle)) * boxDot(axisEnd, alignedRadial) * axisEnd;
        const BoxVec3 predicted = endHinge + axial * axisEnd + expectedRadial;
        const double residual = boxLength(endPosition - predicted);
        if (!std::isfinite(residual)) return {};
        residual2 += residual * residual;
        if (residual <= residualBound) ++covered;
    }
    return {true, angle * 180.0 / 3.14159265358979323846,
            std::sqrt(residual2 / flapCount),
            static_cast<double>(covered) / flapCount,
            residualBound, flapCount};
}

struct BoxPointPairGap {
    bool valid = false;
    double distance = 0.0;
    std::uint32_t firstNode = 0u;
    std::uint32_t secondNode = 0u;
};

// This is a nearest accepted-node diagnostic, not a continuous face-gap or
// contact-force certificate. It is useful for locating candidate seam/flap
// closure but is resolution-dependent and must be labelled accordingly.
inline BoxPointPairGap measureBoxPointPairGap(
    const std::vector<NMFEMNodeStateGPU>& acceptedNodes,
    const std::uint32_t nodeCount,
    const std::uint32_t environment,
    const std::vector<std::uint32_t>& firstNodes,
    const std::vector<std::uint32_t>& secondNodes) {
    if (nodeCount == 0u || acceptedNodes.size() <
            static_cast<std::size_t>(environment + 1u) * nodeCount ||
        firstNodes.empty() || secondNodes.empty())
        return {};
    BoxPointPairGap result{true, std::numeric_limits<double>::infinity(), 0u, 0u};
    for (const std::uint32_t first : firstNodes) {
        if (first >= nodeCount) throw std::invalid_argument("first box feature node is invalid");
        const BoxVec3 a = boxAcceptedPosition(
            acceptedNodes[static_cast<std::size_t>(environment) * nodeCount + first]);
        for (const std::uint32_t second : secondNodes) {
            if (second >= nodeCount) throw std::invalid_argument("second box feature node is invalid");
            const BoxVec3 b = boxAcceptedPosition(
                acceptedNodes[static_cast<std::size_t>(environment) * nodeCount + second]);
            const double distance = boxLength(b - a);
            if (distance < result.distance)
                result = {true, distance, first, second};
        }
    }
    return result;
}

struct BoxClosureGapDiagnostics {
    // These are discrete nearest-node estimates only. They are not continuous
    // face-gap, pressure, friction, or adhesive-bond certificates.
    BoxPointPairGap manufacturerJoint;
    BoxPointPairGap shortFlapOverlap;
    BoxPointPairGap longFlapOverlap;
};

inline std::vector<std::uint32_t> makeBoxManufacturerJointEdgeNodes(
    const FEFCO0201Blank& blank,
    const std::vector<BoxBlankPoint>& restNodes,
    const bool tabFreeEdge) {
    const double targetX = tabFreeEdge
        ? 0.0 : blank.perimeterSpan;
    const double tolerance = 2.0e-10 *
        std::max({blank.perimeterSpan, blank.blankHeight, blank.boardCaliper});
    std::vector<std::uint32_t> result;
    for (std::uint32_t node = 0u; node < restNodes.size(); ++node) {
        const BoxBlankPoint& p = restNodes[node];
        if (std::abs(p.x - targetX) <= tolerance &&
            p.y >= blank.bottomScoreY - tolerance &&
            p.y <= blank.topScoreY + tolerance)
            result.push_back(node);
    }
    return result;
}

inline std::vector<std::uint32_t> makeBoxFlapRegionNodes(
    const FEFCO0201Blank& blank,
    const std::vector<BoxBlankPoint>& restNodes,
    const std::uint32_t panelIndex,
    const bool top) {
    if (panelIndex >= blank.panels.size())
        throw std::out_of_range("box flap panel index is invalid");
    const BoxBlankPanel& panel = blank.panels[panelIndex];
    const double tolerance = 2.0e-10 *
        std::max({blank.perimeterSpan, blank.blankHeight, blank.boardCaliper});
    std::vector<std::uint32_t> result;
    for (std::uint32_t node = 0u; node < restNodes.size(); ++node) {
        const BoxBlankPoint& p = restNodes[node];
        const bool inPanel = p.x >= panel.xStart - tolerance &&
            p.x <= panel.xEnd + tolerance;
        const bool inFlap = top
            ? p.y >= blank.topScoreY - tolerance &&
                p.y <= blank.blankHeight + tolerance
            : p.y >= -tolerance && p.y <= blank.bottomScoreY + tolerance;
        if (inPanel && inFlap) result.push_back(node);
    }
    return result;
}

inline BoxClosureGapDiagnostics measureBoxClosureGapDiagnostics(
    const FEFCO0201Blank& blank,
    const std::vector<BoxBlankPoint>& restNodes,
    const std::vector<NMFEMNodeStateGPU>& acceptedNodes,
    const std::uint32_t environment,
    const std::uint32_t nodeCount,
    const bool topEnd) {
    if (restNodes.size() != nodeCount || blank.panels.size() != 4u) return {};
    const auto tab = makeBoxManufacturerJointEdgeNodes(blank, restNodes, true);
    const auto seam = makeBoxManufacturerJointEdgeNodes(blank, restNodes, false);
    const auto flap1 = makeBoxFlapRegionNodes(blank, restNodes, 1u, topEnd);
    const auto flap3 = makeBoxFlapRegionNodes(blank, restNodes, 3u, topEnd);
    const auto flap0 = makeBoxFlapRegionNodes(blank, restNodes, 0u, topEnd);
    const auto flap2 = makeBoxFlapRegionNodes(blank, restNodes, 2u, topEnd);
    return {
        measureBoxPointPairGap(acceptedNodes, nodeCount, environment, tab, seam),
        measureBoxPointPairGap(acceptedNodes, nodeCount, environment, flap1, flap3),
        measureBoxPointPairGap(acceptedNodes, nodeCount, environment, flap0, flap2),
    };
}

struct BoxExecutionStage {
    enum class Kind : std::uint32_t {
        score = 0u,
        wallFold = 1u,
        manufacturerJointContact = 2u,
        flapFold = 3u,
        endClosureContact = 4u,
    };
    Kind kind = Kind::score;
    std::uint32_t sourceIndex = 0u;
    std::uint32_t assemblyActionIndex = kNoBoxScoreLineIndex;
    double sourceTargetAngleDegrees = 0.0;
    std::vector<std::uint32_t> scoreLineIds;
};

inline std::vector<BoxExecutionStage> makeBoxExecutionProgram(
    const FEFCO0201Blank& blank) {
    if (blank.scoreLines.size() != 12u || blank.assembly.actions.size() != 15u)
        throw std::invalid_argument("box program requires all 12 score IDs and 15 assembly actions");
    std::vector<BoxExecutionStage> program;
    program.reserve(1u + blank.assembly.actions.size());
    BoxExecutionStage scoringDie;
    scoringDie.kind = BoxExecutionStage::Kind::score;
    scoringDie.sourceIndex = 0u;
    scoringDie.assemblyActionIndex = kNoBoxScoreLineIndex;
    for (std::uint32_t score = 0u; score < blank.scoreLines.size(); ++score)
        scoringDie.scoreLineIds.push_back(score);
    program.push_back(std::move(scoringDie));
    for (std::uint32_t actionIndex = 0u;
         actionIndex < blank.assembly.actions.size(); ++actionIndex) {
        const auto& action = blank.assembly.actions[actionIndex];
        BoxExecutionStage::Kind kind = BoxExecutionStage::Kind::score;
        switch (action.kind) {
            case BoxAssemblyAction::Kind::erectWallPanel:
                kind = BoxExecutionStage::Kind::wallFold;
                break;
            case BoxAssemblyAction::Kind::bondManufacturerJoint:
                kind = BoxExecutionStage::Kind::manufacturerJointContact;
                break;
            case BoxAssemblyAction::Kind::foldFlap:
                kind = BoxExecutionStage::Kind::flapFold;
                break;
            case BoxAssemblyAction::Kind::closeEndWithTapeOrAdhesive:
                kind = BoxExecutionStage::Kind::endClosureContact;
                break;
        }
        program.push_back({kind, action.feature.index, actionIndex,
            action.targetAngleDegrees, {}});
    }
    return program;
}

// Shipping-scale action declarations are separated from the retained mini
// controller. These records preserve accepted-state sequencing and name the
// glue overlaps that must be activated only after native contact is accepted.
// They do not fabricate node targets or mark the actions as executed.
struct ShippingAssemblyStage {
    enum class Kind : std::uint32_t {
        scorePass = 0u,
        foldWall = 1u,
        activateManufacturerLap = 2u,
        foldMinorFlaps = 3u,
        foldMajorFlaps = 4u,
        activateAcceptedFlapOverlaps = 5u,
        waitForHmaSet = 6u,
        releaseTemporaryFixtures = 7u,
        settleAndCheckpoint = 8u,
    };
    Kind kind = Kind::scorePass;
    std::uint32_t panelIndex = kNoBoxScoreLineIndex;
    ShippingEnd end = ShippingEnd::bottom;
    std::vector<std::string> acceptedInterfacePatchIds;
    bool requireAcceptedSnapshotBeforeAdvance = true;
    double measuredAdhesiveSetHoldSeconds =
        std::numeric_limits<double>::quiet_NaN();
    bool setHoldProtocolCalibrated = false;
    bool executed = false;
};

inline std::vector<ShippingAssemblyStage> makeShippingAssemblyProgram(
    const ShippingBoxDesign& design) {
    if (!(std::isfinite(design.perimeterSpanM) && design.perimeterSpanM > 0.0) ||
        design.blankConfig.panelLength <= 0.0 ||
        design.blankConfig.panelWidth <= 0.0 ||
        design.blankConfig.wallHeight <= 0.0 ||
        design.allowanceCoupon.couponId.empty())
        throw std::invalid_argument(
            "shipping assembly program requires coupon-derived dimensions");
    const ShippingClosurePlan closure = makeShippingClosurePlan(design);
    std::vector<ShippingAssemblyStage> stages;
    stages.reserve(15u);
    const auto append = [&](const ShippingAssemblyStage::Kind kind,
                            const std::uint32_t panel = kNoBoxScoreLineIndex,
                            const ShippingEnd end = ShippingEnd::bottom,
                            std::vector<std::string> patchIds = {}) {
        stages.push_back({kind, panel, end, std::move(patchIds), true,
            std::numeric_limits<double>::quiet_NaN(), false, false});
    };
    append(ShippingAssemblyStage::Kind::scorePass);
    for (std::uint32_t panel = 0u; panel < 4u; ++panel)
        append(ShippingAssemblyStage::Kind::foldWall, panel);
    append(ShippingAssemblyStage::Kind::activateManufacturerLap,
        kNoBoxScoreLineIndex, ShippingEnd::bottom,
        {closure.patches.front().patchId});
    for (const ShippingEnd end : {ShippingEnd::bottom, ShippingEnd::top}) {
        append(ShippingAssemblyStage::Kind::foldMinorFlaps,
            kNoBoxScoreLineIndex, end);
        append(ShippingAssemblyStage::Kind::foldMajorFlaps,
            kNoBoxScoreLineIndex, end);
        ShippingAssemblyStage activate{
            ShippingAssemblyStage::Kind::activateAcceptedFlapOverlaps,
            kNoBoxScoreLineIndex, end, {}, true,
            std::numeric_limits<double>::quiet_NaN(), false, false};
        for (std::size_t index = 1u; index < closure.patches.size(); ++index)
            if (closure.patches[index].end == end)
                activate.acceptedInterfacePatchIds.push_back(
                    closure.patches[index].patchId);
        stages.push_back(std::move(activate));
    }
    append(ShippingAssemblyStage::Kind::waitForHmaSet);
    append(ShippingAssemblyStage::Kind::releaseTemporaryFixtures);
    append(ShippingAssemblyStage::Kind::settleAndCheckpoint);
    return stages;
}

// The perimeter order is L/W/L/W. This is a prescribed tool-path target
// orientation for the wall panel's material +x direction, computed from the
// authored fold protocol and axis convention; it is not a folded-state edit.
inline double boxWallTargetOrientationDegrees(const std::uint32_t panelIndex) {
    if (panelIndex >= 4u) throw std::out_of_range("FEFCO wall panel index is invalid");
    return -90.0 + 90.0 * panelIndex;
}

} // namespace numi::cardboard
