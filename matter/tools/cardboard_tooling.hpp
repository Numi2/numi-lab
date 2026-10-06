#pragma once

// Analytic contact geometry for a rounded crease punch and a finite support
// anvil. These are real Matter rigid-proxy shapes (capsule + box), not board
// node constraints. The helper only authors geometry and a reference pose; a
// caller drives the body through EncodeRequest::rigid.currentBodies. Matter
// supports this body-backed, non-articulated/non-dynamic kinematic path without
// an ABA or scene-body solve. It has no standalone tool trajectory setter.

#include "numi/matter/matter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace numi::cardboard {

struct ToolVec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct ToolQuaternion {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double w = 1.0;
};

struct ToolPose {
    ToolVec3 translation{};
    ToolQuaternion orientation{};
};

struct CreaseToolingSpec {
    // Board frame: machine direction x, cross-machine direction y, thickness z.
    double boardLength = 0.0;
    double boardWidth = 0.0;
    double boardCaliper = 0.0;
    double creaseX = 0.0;
    double supportSurfaceZ = 0.0;
    double punchRadius = 0.0;
    double punchEndOverhang = 0.0;
    double supportEndOverhang = 0.0;
    double supportThickness = 0.0;
    double initialClearance = 0.0;
    std::uint32_t punchBodyIndex = NM_INVALID_INDEX;
    std::uint32_t contactMaterialIndex = 0u;
};

struct CreaseToolingGeometry {
    numi::matter::RigidProxySource punch;
    numi::matter::RigidProxySource support;
    ToolPose punchInitialPose;
    double punchCenterlineSpan = 0.0;
    double punchOverallSpan = 0.0;
    double supportHalfLength = 0.0;
    double supportHalfWidth = 0.0;
};

struct ToolContactWitness {
    double separation = 0.0;
    ToolVec3 normal{};
    ToolVec3 point{};
};

inline ToolVec3 operator+(const ToolVec3 a, const ToolVec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline ToolVec3 operator-(const ToolVec3 a, const ToolVec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline ToolVec3 operator*(const double scale, const ToolVec3 value) {
    return {scale * value.x, scale * value.y, scale * value.z};
}

inline double dot(const ToolVec3 a, const ToolVec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline double length(const ToolVec3 value) {
    return std::sqrt(dot(value, value));
}

inline ToolVec3 normalized(const ToolVec3 value) {
    const double magnitude = length(value);
    if (!(magnitude > 1.0e-15) || !std::isfinite(magnitude))
        throw std::invalid_argument("cannot normalize a zero tooling vector");
    return (1.0 / magnitude) * value;
}

inline ToolQuaternion normalized(const ToolQuaternion value) {
    const double magnitude = std::sqrt(
        value.x * value.x + value.y * value.y +
        value.z * value.z + value.w * value.w);
    if (!(magnitude > 1.0e-15) || !std::isfinite(magnitude))
        throw std::invalid_argument("tool pose quaternion must be nonzero");
    return {value.x / magnitude, value.y / magnitude,
            value.z / magnitude, value.w / magnitude};
}

inline ToolQuaternion multiply(const ToolQuaternion aRaw,
                               const ToolQuaternion bRaw) {
    const ToolQuaternion a = normalized(aRaw);
    const ToolQuaternion b = normalized(bRaw);
    return normalized({
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    });
}

inline ToolVec3 rotate(const ToolQuaternion raw, const ToolVec3 value) {
    const ToolQuaternion q = normalized(raw);
    const ToolVec3 u{q.x, q.y, q.z};
    const ToolVec3 uv{
        u.y * value.z - u.z * value.y,
        u.z * value.x - u.x * value.z,
        u.x * value.y - u.y * value.x,
    };
    const ToolVec3 uuv{
        u.y * uv.z - u.z * uv.y,
        u.z * uv.x - u.x * uv.z,
        u.x * uv.y - u.y * uv.x,
    };
    return value + (2.0 * q.w) * uv + 2.0 * uuv;
}

inline ToolVec3 transformPoint(const ToolPose pose, const ToolVec3 local) {
    return pose.translation + rotate(pose.orientation, local);
}

inline bool isFinitePositive(const double value) {
    return std::isfinite(value) && value > 0.0;
}

inline CreaseToolingGeometry makeCreaseTooling(
    const CreaseToolingSpec& spec) {
    if (!isFinitePositive(spec.boardLength) ||
        !isFinitePositive(spec.boardWidth) ||
        !isFinitePositive(spec.boardCaliper) ||
        !isFinitePositive(spec.punchRadius) ||
        !(0.5 * spec.boardWidth + spec.punchEndOverhang >
          spec.punchRadius) ||
        !isFinitePositive(spec.supportThickness) ||
        !std::isfinite(spec.creaseX) ||
        !std::isfinite(spec.supportSurfaceZ) ||
        !std::isfinite(spec.punchEndOverhang) ||
        spec.punchEndOverhang < 0.0 ||
        !std::isfinite(spec.supportEndOverhang) ||
        spec.supportEndOverhang < 0.0 ||
        !std::isfinite(spec.initialClearance) ||
        spec.initialClearance < 0.0 ||
        spec.punchBodyIndex == NM_INVALID_INDEX) {
        throw std::invalid_argument(
            "crease tooling dimensions, clearance, and body-backed punch are invalid");
    }
    if (std::abs(spec.creaseX) > 0.5 * spec.boardLength) {
        throw std::invalid_argument("crease line must lie within the board length");
    }

    CreaseToolingGeometry result;
    // Capsule end caps extend by the nose radius along the punch axis too.
    // Subtract that radius so punchEndOverhang describes the physical extent
    // beyond each board edge, rather than the centerline endpoint offset.
    const double punchHalfSpan = 0.5 * spec.boardWidth +
        spec.punchEndOverhang - spec.punchRadius;
    result.punch.shape = NM_RIGID_CAPSULE;
    result.punch.bodyIndex = spec.punchBodyIndex;
    result.punch.materialIndex = spec.contactMaterialIndex;
    result.punch.localCenter = {0.0, -punchHalfSpan, 0.0};
    result.punch.localExtent = {0.0, punchHalfSpan, 0.0};
    result.punch.localOrientation = {0.0, 0.0, 0.0, 1.0};
    result.punch.radiusOrOffset = spec.punchRadius;
    // Keep articulated and dynamic flags clear. Matter reads this body's
    // externally published MRBodyStateGPU pose/velocity; a zero rigid inverse
    // mass makes the punch kinematic so contact cannot move the tool.
    result.punch.articulated = false;
    result.punch.dynamic = false;
    result.punchCenterlineSpan = 2.0 * punchHalfSpan;
    result.punchOverallSpan = result.punchCenterlineSpan +
        2.0 * spec.punchRadius;

    result.support.shape = NM_RIGID_BOX;
    result.support.materialIndex = spec.contactMaterialIndex;
    result.support.localCenter = {
        0.0,
        0.0,
        spec.supportSurfaceZ - 0.5 * spec.supportThickness,
    };
    result.support.localExtent = {
        0.5 * spec.boardLength + spec.supportEndOverhang,
        0.5 * spec.boardWidth + spec.supportEndOverhang,
        0.5 * spec.supportThickness,
    };
    result.support.localOrientation = {0.0, 0.0, 0.0, 1.0};
    result.supportHalfLength = result.support.localExtent[0];
    result.supportHalfWidth = result.support.localExtent[1];

    // The body's origin is its capsule centreline midpoint. This is a reference
    // pose; the caller must publish it and subsequent kinematic targets to
    // Matter for a physical indentation trajectory.
    result.punchInitialPose.translation = {
        spec.creaseX,
        0.0,
        spec.supportSurfaceZ + spec.boardCaliper +
            spec.punchRadius + spec.initialClearance,
    };
    return result;
}

inline ToolPose punchPoseForIndentation(
    const CreaseToolingSpec& spec,
    const double indentation) {
    if (!std::isfinite(indentation) || indentation < 0.0 ||
        indentation > spec.boardCaliper) {
        throw std::invalid_argument(
            "tool indentation must lie between zero and the nominal board caliper");
    }
    const auto geometry = makeCreaseTooling(spec);
    ToolPose pose = geometry.punchInitialPose;
    pose.translation.z -= indentation;
    return pose;
}

inline ToolContactWitness capsuleContactWitness(
    const numi::matter::RigidProxySource& proxy,
    const ToolPose& bodyPose,
    const ToolVec3 point) {
    if (proxy.shape != NM_RIGID_CAPSULE ||
        !isFinitePositive(proxy.radiusOrOffset)) {
        throw std::invalid_argument("capsule witness requires a valid capsule proxy");
    }
    const ToolVec3 first = transformPoint(bodyPose, {
        proxy.localCenter[0], proxy.localCenter[1], proxy.localCenter[2]});
    const ToolVec3 second = transformPoint(bodyPose, {
        proxy.localExtent[0], proxy.localExtent[1], proxy.localExtent[2]});
    const ToolVec3 edge = second - first;
    const double edgeSquared = dot(edge, edge);
    const double parameter = edgeSquared > 1.0e-24
        ? std::clamp(dot(point - first, edge) / edgeSquared, 0.0, 1.0)
        : 0.0;
    const ToolVec3 center = first + parameter * edge;
    const ToolVec3 delta = point - center;
    const double distance = length(delta);
    const ToolVec3 normal = distance > 1.0e-12
        ? (1.0 / distance) * delta
        : ToolVec3{0.0, 0.0, 1.0};
    return {
        distance - proxy.radiusOrOffset,
        normal,
        center + proxy.radiusOrOffset * normal,
    };
}

inline ToolContactWitness boxContactWitness(
    const numi::matter::RigidProxySource& proxy,
    const ToolPose& bodyPose,
    const ToolVec3 point) {
    if (proxy.shape != NM_RIGID_BOX ||
        !isFinitePositive(proxy.localExtent[0]) ||
        !isFinitePositive(proxy.localExtent[1]) ||
        !isFinitePositive(proxy.localExtent[2])) {
        throw std::invalid_argument("box witness requires positive box half extents");
    }
    const ToolPose pose{
        transformPoint(bodyPose, {
            proxy.localCenter[0], proxy.localCenter[1], proxy.localCenter[2]}),
        multiply(bodyPose.orientation, {
            proxy.localOrientation[0], proxy.localOrientation[1],
            proxy.localOrientation[2], proxy.localOrientation[3]}),
    };
    const ToolQuaternion inverse{
        -pose.orientation.x, -pose.orientation.y,
        -pose.orientation.z, pose.orientation.w,
    };
    const ToolVec3 local = rotate(inverse, point - pose.translation);
    const ToolVec3 extent{
        proxy.localExtent[0], proxy.localExtent[1], proxy.localExtent[2]};
    const ToolVec3 signedOutside{
        std::abs(local.x) - extent.x,
        std::abs(local.y) - extent.y,
        std::abs(local.z) - extent.z,
    };
    const ToolVec3 outside{
        std::max(signedOutside.x, 0.0),
        std::max(signedOutside.y, 0.0),
        std::max(signedOutside.z, 0.0),
    };
    const double outsideDistance = length(outside);
    ToolVec3 localWitness{
        std::clamp(local.x, -extent.x, extent.x),
        std::clamp(local.y, -extent.y, extent.y),
        std::clamp(local.z, -extent.z, extent.z),
    };
    ToolVec3 localNormal{};
    double separation = outsideDistance;
    if (outsideDistance > 1.0e-12) {
        localNormal = (1.0 / outsideDistance) * (local - localWitness);
    } else {
        const ToolVec3 remaining{
            extent.x - std::abs(local.x),
            extent.y - std::abs(local.y),
            extent.z - std::abs(local.z),
        };
        unsigned axis = 0u;
        if (remaining.y < remaining.x) axis = 1u;
        if (remaining.z < (axis == 0u ? remaining.x : remaining.y)) axis = 2u;
        const std::array<double, 3> coordinates{local.x, local.y, local.z};
        const std::array<double, 3> halfExtents{extent.x, extent.y, extent.z};
        std::array<double, 3> normal{0.0, 0.0, 0.0};
        normal[axis] = coordinates[axis] >= 0.0 ? 1.0 : -1.0;
        localNormal = {normal[0], normal[1], normal[2]};
        double* witnessCoordinates[] = {
            &localWitness.x, &localWitness.y, &localWitness.z};
        *witnessCoordinates[axis] = normal[axis] * halfExtents[axis];
        separation = -std::array<double, 3>{
            remaining.x, remaining.y, remaining.z}[axis];
    }
    return {
        separation,
        rotate(pose.orientation, localNormal),
        transformPoint(pose, localWitness),
    };
}

} // namespace numi::cardboard
