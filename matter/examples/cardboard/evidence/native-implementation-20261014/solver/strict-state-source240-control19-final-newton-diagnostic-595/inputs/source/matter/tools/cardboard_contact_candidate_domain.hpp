#pragma once

// Conservative CPU broadphase for a swept FEM-node / rigid-tool candidate set.
// This is intentionally not wired into Matter's runtime pair table yet. It
// emits stable (object,node,proxy) identities and fails closed when the caller
// cannot bound a complete path or when the requested output would overflow.

#include "numi/matter/shared.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace numi::matter::cardboard::contact_candidates {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

inline Vec3 operator+(const Vec3 a, const Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 operator-(const Vec3 a, const Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 operator*(const Vec3 value, const double scalar) {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

inline double dot(const Vec3 a, const Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline double length(const Vec3 value) {
    return std::sqrt(dot(value, value));
}

inline bool finite(const Vec3 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
        std::isfinite(value.z);
}

inline double maxAbs(const Vec3 value) {
    return std::max({std::abs(value.x), std::abs(value.y),
        std::abs(value.z)});
}

struct NodeMotionSphere {
    std::uint32_t objectIndex = NM_INVALID_INDEX;
    std::uint32_t nodeIndex = NM_INVALID_INDEX;
    Vec3 acceptedPositionM{};
    Vec3 endpointPositionM{};
    // Upper bound on distance from acceptedPositionM to every position this
    // node may reach during the trial. It must enclose the entire path, not
    // just the endpoint displacement or an endpoint AABB.
    std::optional<double> wholePathRadiusM;
};

struct ToolSweep {
    std::uint32_t proxyIndex = NM_INVALID_INDEX;
    NMRigidProxyGPU proxy{};
    NMRigidStateGPU accepted{};
    NMRigidStateGPU endpoint{};
    // The current native prescribed BOX/capsule contract is linear
    // fixed-orientation translation between accepted and endpoint poses.
    // A moving path without that explicit contract is unbounded here.
    bool centerPathIsLinearTranslation = false;
};

struct StablePairKey {
    std::uint32_t objectIndex = NM_INVALID_INDEX;
    std::uint32_t nodeIndex = NM_INVALID_INDEX;
    std::uint32_t proxyIndex = NM_INVALID_INDEX;

    friend bool operator==(const StablePairKey&, const StablePairKey&) =
        default;
    friend bool operator<(const StablePairKey& a, const StablePairKey& b) {
        return std::tie(a.objectIndex, a.nodeIndex, a.proxyIndex) <
            std::tie(b.objectIndex, b.nodeIndex, b.proxyIndex);
    }
};

struct Candidate {
    StablePairKey key{};
    // A conservative lower bound on clearance from the entire FEM-node path
    // sphere to the swept tool AABB. It can be zero for overlapping bounds.
    double lowerBoundGapM = 0.0;
};

enum class Status {
    complete,
    invalidInput,
    unboundedMotion,
    unsupportedShape,
    unsupportedRotation,
    capacityOverflow,
};

struct Result {
    Status status = Status::invalidInput;
    std::size_t requiredCandidateCount = 0;
    std::vector<Candidate> candidates;
    const char* reason = "not evaluated";

    bool complete() const { return status == Status::complete; }
};

struct Options {
    double supportDistanceM = 0.0;
    double minimumGapFloorM = 0.0;
    double explicitCandidateMarginM = 0.0;
    std::size_t candidateCapacity = 0;
};

namespace detail {

struct Aabb {
    Vec3 minimum{};
    Vec3 maximum{};
};

inline Vec3 xyz(const nm_float4 value) {
    return {static_cast<double>(value.x), static_cast<double>(value.y),
        static_cast<double>(value.z)};
}

inline Vec3 quaternionRotate(const nm_float4 orientation, const Vec3 value) {
    const double qx = orientation.x;
    const double qy = orientation.y;
    const double qz = orientation.z;
    const double qw = orientation.w;
    const double norm = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    if (!(norm > 1.0e-20) || !std::isfinite(norm))
        return {std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0};
    const double x = qx / norm;
    const double y = qy / norm;
    const double z = qz / norm;
    const double w = qw / norm;
    const Vec3 cross1{y * value.z - z * value.y,
        z * value.x - x * value.z, x * value.y - y * value.x};
    const Vec3 cross2{y * cross1.z - z * cross1.y,
        z * cross1.x - x * cross1.z, x * cross1.y - y * cross1.x};
    return value + cross1 * (2.0 * w) + cross2 * 2.0;
}

inline bool equal3(const nm_float4 a, const nm_float4 b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline bool equal4(const nm_float4 a, const nm_float4 b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

inline Vec3 absVector(const Vec3 value) {
    return {std::abs(value.x), std::abs(value.y), std::abs(value.z)};
}

inline Aabb capsuleAabb(const NMRigidStateGPU& start,
                        const NMRigidStateGPU& end) {
    const Vec3 a0 = xyz(start.centerAndRadius);
    const Vec3 b0 = xyz(start.extent);
    const Vec3 a1 = xyz(end.centerAndRadius);
    const Vec3 b1 = xyz(end.extent);
    const double radius = static_cast<double>(start.centerAndRadius.w);
    const Vec3 minimum{
        std::min({a0.x, b0.x, a1.x, b1.x}) - radius,
        std::min({a0.y, b0.y, a1.y, b1.y}) - radius,
        std::min({a0.z, b0.z, a1.z, b1.z}) - radius,
    };
    const Vec3 maximum{
        std::max({a0.x, b0.x, a1.x, b1.x}) + radius,
        std::max({a0.y, b0.y, a1.y, b1.y}) + radius,
        std::max({a0.z, b0.z, a1.z, b1.z}) + radius,
    };
    return {minimum, maximum};
}

inline Aabb boxAabb(const NMRigidStateGPU& start,
                    const NMRigidStateGPU& end) {
    const Vec3 half = absVector(xyz(start.extent));
    const Vec3 axisX = quaternionRotate(start.orientation, {1.0, 0.0, 0.0});
    const Vec3 axisY = quaternionRotate(start.orientation, {0.0, 1.0, 0.0});
    const Vec3 axisZ = quaternionRotate(start.orientation, {0.0, 0.0, 1.0});
    const Vec3 worldHalf{
        std::abs(axisX.x) * half.x + std::abs(axisY.x) * half.y +
            std::abs(axisZ.x) * half.z,
        std::abs(axisX.y) * half.x + std::abs(axisY.y) * half.y +
            std::abs(axisZ.y) * half.z,
        std::abs(axisX.z) * half.x + std::abs(axisY.z) * half.y +
            std::abs(axisZ.z) * half.z,
    };
    const Vec3 c0 = xyz(start.centerAndRadius);
    const Vec3 c1 = xyz(end.centerAndRadius);
    return {
        {std::min(c0.x, c1.x) - worldHalf.x,
         std::min(c0.y, c1.y) - worldHalf.y,
         std::min(c0.z, c1.z) - worldHalf.z},
        {std::max(c0.x, c1.x) + worldHalf.x,
         std::max(c0.y, c1.y) + worldHalf.y,
         std::max(c0.z, c1.z) + worldHalf.z},
    };
}

inline double distanceToAabb(const Vec3 point, const Aabb& box) {
    const Vec3 outside{
        std::max({box.minimum.x - point.x, 0.0, point.x - box.maximum.x}),
        std::max({box.minimum.y - point.y, 0.0, point.y - box.maximum.y}),
        std::max({box.minimum.z - point.z, 0.0, point.z - box.maximum.z}),
    };
    return length(outside);
}

inline double coordinateScale(const Vec3 node, const Aabb& bounds,
                              const NMRigidStateGPU& start,
                              const NMRigidStateGPU& end) {
    return std::max({maxAbs(node), maxAbs(bounds.minimum),
        maxAbs(bounds.maximum), maxAbs(xyz(start.centerAndRadius)),
        maxAbs(xyz(end.centerAndRadius)), 1.0e-12});
}

inline bool validPositiveFinite(const double value) {
    return std::isfinite(value) && value >= 0.0;
}

inline Result failure(const Status status, const char* reason) {
    Result result;
    result.status = status;
    result.reason = reason;
    return result;
}

inline bool validPathShape(const ToolSweep& path, Status& failureStatus,
                           const char*& failureReason) {
    const NMRigidProxyGPU& proxy = path.proxy;
    if (proxy.shapeKind != NM_RIGID_CAPSULE && proxy.shapeKind != NM_RIGID_BOX) {
        failureStatus = Status::unsupportedShape;
        failureReason = "only capsules and boxes have this swept broadphase";
        return false;
    }
    const bool prescribed =
        (proxy.flags & NM_RIGID_PRESCRIBED_TRANSLATION) != 0u;
    const Vec3 startCenter = xyz(path.accepted.centerAndRadius);
    const Vec3 endCenter = xyz(path.endpoint.centerAndRadius);
    const bool isStationary = equal3(path.accepted.centerAndRadius,
        path.endpoint.centerAndRadius);
    if ((!prescribed && !isStationary) ||
        (proxy.flags & (NM_RIGID_ARTICULATED | NM_RIGID_DYNAMIC)) != 0u ||
        (proxy.flags & ~NM_RIGID_PRESCRIBED_TRANSLATION) != 0u) {
        failureStatus = Status::unsupportedRotation;
        failureReason = "moving tools must use prescribed fixed-orientation translation";
        return false;
    }
    if (!equal4(path.accepted.orientation, path.endpoint.orientation)) {
        failureStatus = Status::unsupportedRotation;
        failureReason = "tool rotation between accepted and endpoint poses is unsupported";
        return false;
    }
    if (proxy.shapeKind == NM_RIGID_CAPSULE) {
        if (path.accepted.centerAndRadius.w < 0.0f ||
            path.endpoint.centerAndRadius.w != path.accepted.centerAndRadius.w) {
            failureStatus = Status::invalidInput;
            failureReason = "capsule radius or rigid translation endpoint is inconsistent";
            return false;
        }
        const double pathScale = std::max({maxAbs(startCenter), maxAbs(endCenter),
            length(xyz(path.accepted.extent) - startCenter), 1.0e-12});
        const double tolerance = 16.0 * std::numeric_limits<float>::epsilon() *
            pathScale;
        if (length(xyz(path.endpoint.extent) - xyz(path.accepted.extent) -
                (endCenter - startCenter)) > tolerance) {
            failureStatus = Status::invalidInput;
            failureReason = "capsule endpoints do not share one translating rigid path";
            return false;
        }
    } else {
        if (!equal3(path.accepted.extent, path.endpoint.extent) ||
            !equal3(path.accepted.extent, proxy.localExtent)) {
            failureStatus = Status::invalidInput;
            failureReason = "box half-extents must remain fixed over the sweep";
            return false;
        }
    }
    if (!finite(startCenter) || !finite(endCenter) ||
        !finite(xyz(path.accepted.extent)) ||
        !finite(xyz(path.endpoint.extent)) ||
        !finite(xyz(path.accepted.centerAndRadius)) ||
        !finite(xyz(path.endpoint.centerAndRadius)) ||
        !std::isfinite(path.accepted.centerAndRadius.w) ||
        !std::isfinite(path.endpoint.centerAndRadius.w) ||
        !std::isfinite(path.accepted.orientation.x) ||
        !std::isfinite(path.accepted.orientation.y) ||
        !std::isfinite(path.accepted.orientation.z) ||
        !std::isfinite(path.accepted.orientation.w)) {
        failureStatus = Status::invalidInput;
        failureReason = "tool pose contains a nonfinite or degenerate value";
        return false;
    }
    if (!isStationary && !path.centerPathIsLinearTranslation) {
        failureStatus = Status::unboundedMotion;
        failureReason = "moving tool lacks the native linear-translation path contract";
        return false;
    }
    return true;
}

} // namespace detail

// Aabb distance is a broadphase lower bound for distance to the swept shape.
// A box's oriented AABB and a capsule's endpoint hull are supersets at every
// intermediate linearly translated pose. The FEM path, including nonlinear
// trial motion, is enclosed by a sphere about the accepted position; the
// threshold includes the solver's minimum gap floor plus support and margins.
inline Result enumerate(std::span<const NodeMotionSphere> nodes,
                        std::span<const ToolSweep> tools,
                        const Options& options) {
    using namespace detail;
    if (!std::isfinite(options.supportDistanceM) ||
        options.supportDistanceM < 0.0 ||
        !std::isfinite(options.minimumGapFloorM) ||
        options.minimumGapFloorM < 0.0 ||
        !std::isfinite(options.explicitCandidateMarginM) ||
        options.explicitCandidateMarginM < 0.0)
        return failure(Status::invalidInput, "candidate distances must be finite and nonnegative");

    std::vector<const NodeMotionSphere*> orderedNodes;
    orderedNodes.reserve(nodes.size());
    for (const NodeMotionSphere& node : nodes) {
        if (node.objectIndex == NM_INVALID_INDEX ||
            node.nodeIndex == NM_INVALID_INDEX || !finite(node.acceptedPositionM) ||
            !finite(node.endpointPositionM))
            return failure(Status::invalidInput, "node identity or endpoint position is invalid");
        if (!node.wholePathRadiusM ||
            !validPositiveFinite(*node.wholePathRadiusM))
            return failure(Status::unboundedMotion, "FEM node lacks a finite whole-path sphere");
        const double endpointDistance =
            length(node.endpointPositionM - node.acceptedPositionM);
        const double endpointAllowance = 16.0 *
            std::numeric_limits<float>::epsilon() *
            std::max({maxAbs(node.acceptedPositionM),
                maxAbs(node.endpointPositionM), 1.0e-12});
        if (endpointDistance > *node.wholePathRadiusM + endpointAllowance)
            return failure(Status::invalidInput,
                "whole-path sphere does not enclose the declared FEM endpoint");
        orderedNodes.push_back(&node);
    }
    std::sort(orderedNodes.begin(), orderedNodes.end(), [](const auto* a, const auto* b) {
        return std::tie(a->objectIndex, a->nodeIndex) <
            std::tie(b->objectIndex, b->nodeIndex);
    });
    for (std::size_t i = 1; i < orderedNodes.size(); ++i) {
        if (orderedNodes[i - 1u]->objectIndex == orderedNodes[i]->objectIndex &&
            orderedNodes[i - 1u]->nodeIndex == orderedNodes[i]->nodeIndex)
            return failure(Status::invalidInput, "duplicate FEM node identity");
    }

    std::vector<const ToolSweep*> orderedTools;
    orderedTools.reserve(tools.size());
    for (const ToolSweep& tool : tools) {
        if (tool.proxyIndex == NM_INVALID_INDEX)
            return failure(Status::invalidInput, "rigid proxy identity is invalid");
        Status toolFailureStatus = Status::invalidInput;
        const char* toolFailureReason = "invalid rigid path";
        if (!validPathShape(tool, toolFailureStatus, toolFailureReason))
            return failure(toolFailureStatus, toolFailureReason);
        orderedTools.push_back(&tool);
    }
    std::sort(orderedTools.begin(), orderedTools.end(), [](const auto* a, const auto* b) {
        return a->proxyIndex < b->proxyIndex;
    });
    for (std::size_t i = 1; i < orderedTools.size(); ++i) {
        if (orderedTools[i - 1u]->proxyIndex == orderedTools[i]->proxyIndex)
            return failure(Status::invalidInput, "duplicate rigid proxy identity");
    }

    const double clearanceThreshold = std::max(options.supportDistanceM,
        options.minimumGapFloorM) + options.explicitCandidateMarginM;
    if (!std::isfinite(clearanceThreshold))
        return failure(Status::invalidInput, "combined clearance threshold overflows");
    auto candidateGap = [&](const NodeMotionSphere& node,
                            const ToolSweep& tool) -> std::optional<double> {
        if (tool.proxy.adaptiveObjectIndex == node.objectIndex)
            return std::nullopt;
        const Aabb bounds = tool.proxy.shapeKind == NM_RIGID_CAPSULE
            ? capsuleAabb(tool.accepted, tool.endpoint)
            : boxAabb(tool.accepted, tool.endpoint);
        if (!finite(bounds.minimum) || !finite(bounds.maximum))
            return std::numeric_limits<double>::quiet_NaN();
        const double roundoff = 16.0 *
            std::numeric_limits<float>::epsilon() *
            coordinateScale(node.acceptedPositionM, bounds,
                tool.accepted, tool.endpoint);
        const double lowerBoundGap = std::max(0.0,
            distanceToAabb(node.acceptedPositionM, bounds) -
                *node.wholePathRadiusM);
        if (lowerBoundGap <= clearanceThreshold + roundoff)
            return lowerBoundGap;
        return std::nullopt;
    };

    // Count first. Overflow returns the exact required cardinality and no
    // partial pair list, even when the caller's cap is much smaller.
    std::size_t requiredCount = 0u;
    for (const NodeMotionSphere* node : orderedNodes) {
        for (const ToolSweep* tool : orderedTools) {
            const std::optional<double> gap = candidateGap(*node, *tool);
            if (gap && !std::isfinite(*gap))
                return failure(Status::invalidInput, "swept rigid AABB is nonfinite");
            if (gap) {
                if (requiredCount == std::numeric_limits<std::size_t>::max())
                    return failure(Status::capacityOverflow, "candidate count overflows size_t");
                ++requiredCount;
            }
        }
    }
    if (requiredCount > options.candidateCapacity) {
        Result result;
        result.requiredCandidateCount = requiredCount;
        result.status = Status::capacityOverflow;
        result.reason = "conservative candidate set exceeds output capacity";
        return result;
    }

    std::vector<Candidate> candidates;
    candidates.reserve(requiredCount);
    for (const NodeMotionSphere* node : orderedNodes) {
        for (const ToolSweep* tool : orderedTools) {
            const std::optional<double> gap = candidateGap(*node, *tool);
            if (gap)
                candidates.push_back({
                    {node->objectIndex, node->nodeIndex, tool->proxyIndex}, *gap});
        }
    }
    Result result;
    result.requiredCandidateCount = requiredCount;
    result.status = Status::complete;
    result.reason = "complete conservative candidate superset";
    result.candidates = std::move(candidates);
    return result;
}

} // namespace numi::matter::cardboard::contact_candidates
