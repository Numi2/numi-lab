#include "cardboard_contact_candidate_domain.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace {

using numi::matter::cardboard::contact_candidates::Candidate;
using numi::matter::cardboard::contact_candidates::NodeMotionSphere;
using numi::matter::cardboard::contact_candidates::Options;
using numi::matter::cardboard::contact_candidates::StablePairKey;
using numi::matter::cardboard::contact_candidates::Status;
using numi::matter::cardboard::contact_candidates::ToolSweep;
using numi::matter::cardboard::contact_candidates::Vec3;

struct Check {
    int assertions = 0;
    std::vector<std::string> failures;

    void require(const bool condition, const std::string& message) {
        ++assertions;
        if (!condition) failures.push_back(message);
    }
};

Vec3 subtract(const Vec3 a, const Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 add(const Vec3 a, const Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 scale(const Vec3 a, const double value) {
    return {a.x * value, a.y * value, a.z * value};
}

double squaredLength(const Vec3 value) {
    return value.x * value.x + value.y * value.y + value.z * value.z;
}

double segmentSegmentDistance(const Vec3 p0, const Vec3 p1,
                              const Vec3 q0, const Vec3 q1) {
    // Independent double-precision closest-segment oracle for capsule sweeps.
    const Vec3 u = subtract(p1, p0);
    const Vec3 v = subtract(q1, q0);
    const Vec3 w = subtract(p0, q0);
    const double a = squaredLength(u);
    const double b = u.x * v.x + u.y * v.y + u.z * v.z;
    const double c = squaredLength(v);
    const double d = u.x * w.x + u.y * w.y + u.z * w.z;
    const double e = v.x * w.x + v.y * w.y + v.z * w.z;
    const double denominator = a * c - b * b;
    double sNumerator = 0.0;
    double sDenominator = denominator;
    double tNumerator = 0.0;
    double tDenominator = denominator;
    constexpr double epsilon = 1.0e-24;
    if (denominator < epsilon) {
        sNumerator = 0.0;
        sDenominator = 1.0;
        tNumerator = e;
        tDenominator = c;
    } else {
        sNumerator = b * e - c * d;
        tNumerator = a * e - b * d;
        if (sNumerator < 0.0) {
            sNumerator = 0.0;
            tNumerator = e;
            tDenominator = c;
        } else if (sNumerator > sDenominator) {
            sNumerator = sDenominator;
            tNumerator = e + b;
            tDenominator = c;
        }
    }
    if (tNumerator < 0.0) {
        tNumerator = 0.0;
        if (-d < 0.0) sNumerator = 0.0;
        else if (-d > a) sNumerator = sDenominator;
        else {
            sNumerator = -d;
            sDenominator = a;
        }
    } else if (tNumerator > tDenominator) {
        tNumerator = tDenominator;
        if (-d + b < 0.0) sNumerator = 0.0;
        else if (-d + b > a) sNumerator = sDenominator;
        else {
            sNumerator = -d + b;
            sDenominator = a;
        }
    }
    const double s = std::abs(sNumerator) < epsilon
        ? 0.0 : sNumerator / sDenominator;
    const double t = std::abs(tNumerator) < epsilon
        ? 0.0 : tNumerator / tDenominator;
    return std::sqrt(squaredLength(add(w, subtract(scale(u, s), scale(v, t)))));
}

Vec3 rotateByQuaternion(const nm_float4 qRaw, const Vec3 value) {
    // Deliberately separate implementation from the candidate header.
    const double norm = std::sqrt(static_cast<double>(qRaw.x) * qRaw.x +
        static_cast<double>(qRaw.y) * qRaw.y +
        static_cast<double>(qRaw.z) * qRaw.z +
        static_cast<double>(qRaw.w) * qRaw.w);
    const double x = qRaw.x / norm;
    const double y = qRaw.y / norm;
    const double z = qRaw.z / norm;
    const double w = qRaw.w / norm;
    const Vec3 qv{x, y, z};
    const Vec3 first{
        qv.y * value.z - qv.z * value.y,
        qv.z * value.x - qv.x * value.z,
        qv.x * value.y - qv.y * value.x,
    };
    const Vec3 second{
        qv.y * first.z - qv.z * first.y,
        qv.z * first.x - qv.x * first.z,
        qv.x * first.y - qv.y * first.x,
    };
    return add(add(value, scale(first, 2.0 * w)), scale(second, 2.0));
}

double segmentAabbDistanceOracle(const Vec3 first, const Vec3 last,
                                 const Vec3 half) {
    // Exact segment-to-AABB minimum: split at coordinate/face crossings, then
    // minimize the quadratic distance on each piece. This is test-only and is
    // independent of the broadphase AABB-distance function.
    const Vec3 delta = subtract(last, first);
    std::vector<double> breakpoints{0.0, 1.0};
    const double starts[3]{first.x, first.y, first.z};
    const double slopes[3]{delta.x, delta.y, delta.z};
    const double halves[3]{half.x, half.y, half.z};
    for (int axis = 0; axis < 3; ++axis) {
        if (std::abs(slopes[axis]) < 1.0e-30) continue;
        for (const double face : {-halves[axis], halves[axis]}) {
            const double t = (face - starts[axis]) / slopes[axis];
            if (t > 0.0 && t < 1.0) breakpoints.push_back(t);
        }
    }
    std::sort(breakpoints.begin(), breakpoints.end());
    double minimumSquared = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i + 1 < breakpoints.size(); ++i) {
        const double lo = breakpoints[i];
        const double hi = breakpoints[i + 1];
        const double middle = 0.5 * (lo + hi);
        Vec3 offset{};
        Vec3 activeSlope{};
        const double p[3]{first.x, first.y, first.z};
        const double d[3]{delta.x, delta.y, delta.z};
        const double h[3]{half.x, half.y, half.z};
        double o[3]{};
        double s[3]{};
        for (int axis = 0; axis < 3; ++axis) {
            const double x = p[axis] + middle * d[axis];
            if (x > h[axis]) {
                o[axis] = p[axis] - h[axis];
                s[axis] = d[axis];
            } else if (x < -h[axis]) {
                o[axis] = p[axis] + h[axis];
                s[axis] = d[axis];
            }
        }
        offset = {o[0], o[1], o[2]};
        activeSlope = {s[0], s[1], s[2]};
        const double denom = squaredLength(activeSlope);
        const double stationary = denom > 0.0
            ? -((offset.x * activeSlope.x) + (offset.y * activeSlope.y) +
                (offset.z * activeSlope.z)) / denom
            : lo;
        const double t = std::clamp(stationary, lo, hi);
        minimumSquared = std::min(minimumSquared,
            squaredLength(add(offset, scale(activeSlope, t))));
    }
    return std::sqrt(std::max(0.0, minimumSquared));
}

std::set<StablePairKey> candidateKeys(const std::vector<Candidate>& candidates) {
    std::set<StablePairKey> keys;
    for (const Candidate& candidate : candidates) keys.insert(candidate.key);
    return keys;
}

std::set<StablePairKey> exactLinearPathHits(
    const std::vector<NodeMotionSphere>& nodes,
    const std::vector<ToolSweep>& tools,
    const double threshold) {
    std::set<StablePairKey> hits;
    for (const NodeMotionSphere& node : nodes) {
        for (const ToolSweep& tool : tools) {
            if (tool.proxy.adaptiveObjectIndex == node.objectIndex) continue;
            const Vec3 nodeStart = node.acceptedPositionM;
            // The fixture's exact FEM path is a straight segment between its
            // authored endpoints; the bound may be larger for curved cases.
            const Vec3 nodeEnd = node.endpointPositionM;
            double gap = 0.0;
            if (tool.proxy.shapeKind == NM_RIGID_CAPSULE) {
                const Vec3 rigidStart = numi::matter::cardboard::contact_candidates::detail::xyz(
                    tool.accepted.centerAndRadius);
                const Vec3 rigidEnd = numi::matter::cardboard::contact_candidates::detail::xyz(
                    tool.endpoint.centerAndRadius);
                const Vec3 relativeFirst = subtract(nodeStart, rigidStart);
                const Vec3 relativeLast = subtract(nodeEnd, rigidEnd);
                const Vec3 axis = subtract(
                    numi::matter::cardboard::contact_candidates::detail::xyz(tool.accepted.extent),
                    rigidStart);
                const double centerlineDistance = segmentSegmentDistance(
                    relativeFirst, relativeLast, {0.0, 0.0, 0.0}, axis);
                gap = centerlineDistance - tool.accepted.centerAndRadius.w;
            } else {
                const Vec3 rigidStart = numi::matter::cardboard::contact_candidates::detail::xyz(
                    tool.accepted.centerAndRadius);
                const Vec3 rigidEnd = numi::matter::cardboard::contact_candidates::detail::xyz(
                    tool.endpoint.centerAndRadius);
                const Vec3 relativeFirst = rotateByQuaternion(tool.accepted.orientation,
                    subtract(nodeStart, rigidStart));
                const Vec3 relativeLast = rotateByQuaternion(tool.accepted.orientation,
                    subtract(nodeEnd, rigidEnd));
                const Vec3 half = numi::matter::cardboard::contact_candidates::detail::xyz(
                    tool.accepted.extent);
                gap = segmentAabbDistanceOracle(relativeFirst, relativeLast, half);
            }
            if (gap <= threshold + 1.0e-12)
                hits.insert({node.objectIndex, node.nodeIndex, tool.proxyIndex});
        }
    }
    return hits;
}

nm_float4 f4(const float x, const float y, const float z, const float w) {
    return {x, y, z, w};
}

ToolSweep staticCapsule(const std::uint32_t proxyIndex, const Vec3 a,
                        const Vec3 b, const float radius) {
    ToolSweep tool;
    tool.proxyIndex = proxyIndex;
    tool.proxy.shapeKind = NM_RIGID_CAPSULE;
    tool.proxy.bodyIndex = NM_INVALID_INDEX;
    tool.proxy.adaptiveObjectIndex = NM_INVALID_INDEX;
    tool.proxy.localCenterAndRadius = f4(static_cast<float>(a.x),
        static_cast<float>(a.y), static_cast<float>(a.z), radius);
    tool.proxy.localExtent = f4(static_cast<float>(b.x),
        static_cast<float>(b.y), static_cast<float>(b.z), 0.0f);
    tool.proxy.localOrientation = f4(0.0f, 0.0f, 0.0f, 1.0f);
    tool.accepted.centerAndRadius = tool.endpoint.centerAndRadius =
        tool.proxy.localCenterAndRadius;
    tool.accepted.extent = tool.endpoint.extent = tool.proxy.localExtent;
    tool.accepted.orientation = tool.endpoint.orientation =
        tool.proxy.localOrientation;
    return tool;
}

ToolSweep movingCapsule(const std::uint32_t proxyIndex, const Vec3 a0,
                        const Vec3 b0, const Vec3 a1, const Vec3 b1,
                        const float radius) {
    ToolSweep tool = staticCapsule(proxyIndex, a0, b0, radius);
    tool.proxy.flags = NM_RIGID_PRESCRIBED_TRANSLATION;
    tool.endpoint.centerAndRadius = f4(static_cast<float>(a1.x),
        static_cast<float>(a1.y), static_cast<float>(a1.z), radius);
    tool.endpoint.extent = f4(static_cast<float>(b1.x),
        static_cast<float>(b1.y), static_cast<float>(b1.z), 0.0f);
    tool.centerPathIsLinearTranslation = true;
    return tool;
}

ToolSweep movingBox(const std::uint32_t proxyIndex, const Vec3 c0,
                    const Vec3 c1, const Vec3 halfExtent,
                    const nm_float4 orientation) {
    ToolSweep tool;
    tool.proxyIndex = proxyIndex;
    tool.proxy.shapeKind = NM_RIGID_BOX;
    tool.proxy.flags = NM_RIGID_PRESCRIBED_TRANSLATION;
    tool.proxy.adaptiveObjectIndex = NM_INVALID_INDEX;
    tool.proxy.localExtent = f4(static_cast<float>(halfExtent.x),
        static_cast<float>(halfExtent.y), static_cast<float>(halfExtent.z), 0.0f);
    tool.proxy.localOrientation = orientation;
    tool.accepted.centerAndRadius = f4(static_cast<float>(c0.x),
        static_cast<float>(c0.y), static_cast<float>(c0.z), 0.0f);
    tool.endpoint.centerAndRadius = f4(static_cast<float>(c1.x),
        static_cast<float>(c1.y), static_cast<float>(c1.z), 0.0f);
    tool.accepted.extent = tool.endpoint.extent = tool.proxy.localExtent;
    tool.accepted.orientation = tool.endpoint.orientation = orientation;
    tool.centerPathIsLinearTranslation = true;
    return tool;
}

void testExhaustiveEquivalenceAndStability(Check& check) {
    std::vector<NodeMotionSphere> nodes{
        {7u, 4u, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 0.0},
        {7u, 3u, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 0.0},
        {7u, 2u, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 0.0},
        {7u, 1u, {-0.010, 0.0, 0.0}, {0.010, 0.0, 0.0}, 0.020},
        {7u, 99u, {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0}, 0.0},
    };
    std::vector<ToolSweep> tools{
        movingBox(12u, {-0.020, 0.0, 0.0}, {0.020, 0.0, 0.0},
            {0.001, 0.002, 0.003}, f4(0.0f, 0.0f, 0.25881904f, 0.9659258f)),
        movingCapsule(11u, {-0.020, 0.0, 0.0}, {-0.020, 0.0, 0.010},
            {0.020, 0.0, 0.0}, {0.020, 0.0, 0.010}, 0.001f),
        staticCapsule(10u, {0.0, 0.0, -0.010}, {0.0, 0.0, 0.010}, 0.001f),
    };
    Options options{1.0e-5, 2.0e-5, 1.0e-5, 128u};
    const double threshold = std::max(options.supportDistanceM,
        options.minimumGapFloorM) + options.explicitCandidateMarginM;
    const auto first = numi::matter::cardboard::contact_candidates::enumerate(
        nodes, tools, options);
    check.require(first.status == Status::complete,
        "bounded capsule/BOX swept fixture enumerates successfully");
    check.require(first.requiredCandidateCount == first.candidates.size(),
        "complete result reports exact emitted pair cardinality");
    check.require(std::is_sorted(first.candidates.begin(), first.candidates.end(),
        [](const Candidate& a, const Candidate& b) { return a.key < b.key; }),
        "candidate identities are sorted in stable object/node/proxy order");
    check.require(std::adjacent_find(first.candidates.begin(), first.candidates.end(),
        [](const Candidate& a, const Candidate& b) { return a.key == b.key; }) ==
            first.candidates.end(),
        "no stable node/proxy identity is emitted more than once");

    const std::set<StablePairKey> exactHits = exactLinearPathHits(nodes, tools, threshold);
    const std::set<StablePairKey> broadphase = candidateKeys(first.candidates);
    for (const StablePairKey& exact : exactHits)
        check.require(broadphase.contains(exact),
            "conservative broadphase contains every exact capsule/BOX swept hit");
    check.require(exactHits.contains({7u, 1u, 10u}),
        "node trajectory crosses stationary capsule although both endpoints are clear");
    check.require(exactHits.contains({7u, 2u, 11u}),
        "moving capsule crosses a stationary node although tool endpoints are clear");
    check.require(exactHits.contains({7u, 2u, 12u}),
        "moving rotated BOX crosses a stationary node although tool endpoints are clear");
    check.require(0.010 - 0.001 > threshold,
        "moving-node/capsule initial and final clearances exceed candidate range");
    check.require(0.020 - 0.001 > threshold,
        "moving-capsule endpoint clearances exceed candidate range");
    check.require(0.020 - std::sqrt(0.001 * 0.001 + 0.002 * 0.002 +
                0.003 * 0.003) > threshold,
        "moving-BOX endpoint clearances exceed candidate range");
    check.require(std::none_of(first.candidates.begin(), first.candidates.end(),
        [](const Candidate& item) { return item.key.nodeIndex == 99u; }),
        "far no-hit node is excluded from every proxy candidate set");

    std::reverse(nodes.begin(), nodes.end());
    std::reverse(tools.begin(), tools.end());
    const auto permuted = numi::matter::cardboard::contact_candidates::enumerate(
        nodes, tools, options);
    check.require(permuted.status == Status::complete &&
        candidateKeys(permuted.candidates) == broadphase,
        "candidate keys do not depend on input enumeration order");

    ToolSweep adaptiveTool = staticCapsule(30u, {0.0, 0.0, -0.01},
        {0.0, 0.0, 0.01}, 0.001f);
    adaptiveTool.proxy.adaptiveObjectIndex = 7u;
    const auto selfExcluded = numi::matter::cardboard::contact_candidates::enumerate(
        std::span<const NodeMotionSphere>(&nodes.front(), 1u),
        std::span<const ToolSweep>(&adaptiveTool, 1u), options);
    check.require(selfExcluded.status == Status::complete &&
        selfExcluded.candidates.empty(),
        "compiler adaptive-object self-contact exclusion is preserved");

    std::vector<NodeMotionSphere> duplicateNodes{
        {4u, 2u, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 0.0},
        {4u, 2u, {0.1, 0.0, 0.0}, {0.1, 0.0, 0.0}, 0.0},
    };
    const auto duplicateNodeResult =
        numi::matter::cardboard::contact_candidates::enumerate(
            duplicateNodes, tools, options);
    check.require(duplicateNodeResult.status == Status::invalidInput,
        "duplicate contact-domain node key is rejected before enumeration");
}

void testNonlinearEnvelopesAndGapFloor(Check& check) {
    // The FEM node follows a curved path whose endpoints are both clear and
    // coincide, but whose midpoint crosses a stationary capsule. The one
    // global sphere around its accepted position encloses the full path.
    ToolSweep stationary = staticCapsule(4u, {0.0, 0.0, -0.005},
        {0.0, 0.0, 0.005}, 0.001f);
    NodeMotionSphere curvedPath{2u, 5u, {0.010, 0.0, 0.0},
        {0.010, 0.0, 0.0}, 0.010};
    const Options options{1.0e-5, 2.0e-5, 0.0, 8u};
    const auto result = numi::matter::cardboard::contact_candidates::enumerate(
        std::span<const NodeMotionSphere>(&curvedPath, 1u),
        std::span<const ToolSweep>(&stationary, 1u), options);
    check.require(result.status == Status::complete && result.candidates.size() == 1u,
        "whole-node motion sphere includes a nonlinear FEM path with clear endpoints");
    check.require(0.010 - 0.001 > options.minimumGapFloorM &&
        0.0 <= options.minimumGapFloorM,
        "nonlinear FEM path endpoints are clear while its interior reaches the tool");

    ToolSweep nearTool = staticCapsule(8u, {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.010}, 0.001f);
    NodeMotionSphere nearNode{1u, 3u, {0.0016, 0.0, 0.0},
        {0.0016, 0.0, 0.0}, 0.0};
    Options floorOptions{1.0e-4, 7.0e-4, 0.0, 4u};
    const auto floorCandidate = numi::matter::cardboard::contact_candidates::enumerate(
        std::span<const NodeMotionSphere>(&nearNode, 1u),
        std::span<const ToolSweep>(&nearTool, 1u), floorOptions);
    check.require(floorCandidate.status == Status::complete &&
        floorCandidate.candidates.size() == 1u,
        "minimum gap floor expands the candidate domain beyond physical support");
    check.require(0.0016 - 0.001 > floorOptions.supportDistanceM &&
        0.0016 - 0.001 <= floorOptions.minimumGapFloorM,
        "floor-only candidate is inside the declared floor and outside support");
    floorOptions.minimumGapFloorM = 0.0;
    const auto supportOnly = numi::matter::cardboard::contact_candidates::enumerate(
        std::span<const NodeMotionSphere>(&nearNode, 1u),
        std::span<const ToolSweep>(&nearTool, 1u), floorOptions);
    check.require(supportOnly.status == Status::complete &&
        supportOnly.candidates.empty(),
        "candidate outside support is omitted when the explicit floor is disabled");
}

void testFailClosedAndOverflow(Check& check) {
    std::vector<NodeMotionSphere> nodes{
        {0u, 0u, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 0.0},
        {0u, 1u, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 0.0},
    };
    std::vector<ToolSweep> tools{
        staticCapsule(0u, {0.0, 0.0, -0.001}, {0.0, 0.0, 0.001}, 0.001f),
        movingBox(1u, {-0.001, 0.0, 0.0}, {0.001, 0.0, 0.0},
            {0.001, 0.001, 0.001}, f4(0.0f, 0.0f, 0.0f, 1.0f)),
    };
    Options options{0.0, 0.0, 0.0, 16u};
    const auto complete = numi::matter::cardboard::contact_candidates::enumerate(
        nodes, tools, options);
    check.require(complete.status == Status::complete &&
        complete.requiredCandidateCount == 4u,
        "small exhaustive overlap fixture produces four exact stable pairs");
    options.candidateCapacity = complete.requiredCandidateCount - 1u;
    const auto overflow = numi::matter::cardboard::contact_candidates::enumerate(
        nodes, tools, options);
    check.require(overflow.status == Status::capacityOverflow,
        "capacity shortfall rejects the whole candidate result");
    check.require(overflow.requiredCandidateCount == complete.requiredCandidateCount &&
        overflow.candidates.empty(),
        "overflow reports exact required count and never returns a truncated list");

    NodeMotionSphere unboundedNode{0u, 3u, {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0}, std::nullopt};
    options.candidateCapacity = 16u;
    auto rejected = numi::matter::cardboard::contact_candidates::enumerate(
        std::span<const NodeMotionSphere>(&unboundedNode, 1u),
        std::span<const ToolSweep>(tools.data(), 1u), options);
    check.require(rejected.status == Status::unboundedMotion && rejected.candidates.empty(),
        "missing finite whole-node sphere fails closed before emitting pairs");

    ToolSweep unboundedTool = movingBox(7u, {-0.01, 0.0, 0.0},
        {0.01, 0.0, 0.0}, {0.001, 0.001, 0.001}, f4(0.0f, 0.0f, 0.0f, 1.0f));
    unboundedTool.centerPathIsLinearTranslation = false;
    const NodeMotionSphere boundedNode{0u, 4u, {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0}, 0.0};
    rejected = numi::matter::cardboard::contact_candidates::enumerate(
        std::span<const NodeMotionSphere>(&boundedNode, 1u),
        std::span<const ToolSweep>(&unboundedTool, 1u), options);
    check.require(rejected.status == Status::unboundedMotion && rejected.candidates.empty(),
        "moving rigid tool without the linear path contract fails closed");

    const NodeMotionSphere undersizedSphere{0u, 5u,
        {-0.010, 0.0, 0.0}, {0.010, 0.0, 0.0}, 0.019};
    rejected = numi::matter::cardboard::contact_candidates::enumerate(
        std::span<const NodeMotionSphere>(&undersizedSphere, 1u),
        std::span<const ToolSweep>(tools.data(), 1u), options);
    check.require(rejected.status == Status::invalidInput && rejected.candidates.empty(),
        "motion sphere that fails to include its declared endpoint is rejected");

    ToolSweep rotating = tools[1];
    rotating.endpoint.orientation.z = 0.1f;
    rotating.endpoint.orientation.w = 0.9949874f;
    rejected = numi::matter::cardboard::contact_candidates::enumerate(
        std::span<const NodeMotionSphere>(&boundedNode, 1u),
        std::span<const ToolSweep>(&rotating, 1u), options);
    check.require(rejected.status == Status::unsupportedRotation && rejected.candidates.empty(),
        "changed rigid orientation is rejected rather than swept approximately");

    ToolSweep nonRigidCapsule = movingCapsule(3u, {-0.001, 0.0, 0.0},
        {-0.001, 0.0, 0.001}, {0.001, 0.0, 0.0}, {0.0015, 0.0, 0.001}, 0.001f);
    rejected = numi::matter::cardboard::contact_candidates::enumerate(
        std::span<const NodeMotionSphere>(&boundedNode, 1u),
        std::span<const ToolSweep>(&nonRigidCapsule, 1u), options);
    check.require(rejected.status == Status::invalidInput && rejected.candidates.empty(),
        "capsule whose second endpoint does not share the rigid translation is rejected");

    ToolSweep unsupported = tools.front();
    unsupported.proxy.shapeKind = NM_RIGID_SPHERE;
    rejected = numi::matter::cardboard::contact_candidates::enumerate(
        std::span<const NodeMotionSphere>(&boundedNode, 1u),
        std::span<const ToolSweep>(&unsupported, 1u), options);
    check.require(rejected.status == Status::unsupportedShape,
        "unsupported sphere does not silently use capsule or box bounds");

    options.minimumGapFloorM = -1.0;
    rejected = numi::matter::cardboard::contact_candidates::enumerate(
        std::span<const NodeMotionSphere>(&boundedNode, 1u),
        std::span<const ToolSweep>(tools.data(), 1u), options);
    check.require(rejected.status == Status::invalidInput,
        "invalid negative gap floor is rejected");

    ToolSweep duplicateProxy = tools.front();
    std::array<ToolSweep, 2> duplicateProxies{tools.front(), duplicateProxy};
    options.minimumGapFloorM = 0.0;
    rejected = numi::matter::cardboard::contact_candidates::enumerate(
        std::span<const NodeMotionSphere>(&boundedNode, 1u), duplicateProxies, options);
    check.require(rejected.status == Status::invalidInput,
        "duplicate rigid proxy key is rejected before enumeration");
}

} // namespace

int main() {
    Check check;
    testExhaustiveEquivalenceAndStability(check);
    testNonlinearEnvelopesAndGapFloor(check);
    testFailClosedAndOverflow(check);
    if (!check.failures.empty()) {
        for (const std::string& failure : check.failures)
            std::cerr << "FAIL: " << failure << '\n';
        std::cerr << check.failures.size() << " of " << check.assertions
                  << " contact-candidate checks failed\n";
        return 1;
    }
    std::cout << "PASS cardboard swept contact candidate domain: "
              << check.assertions << " checks; CPU broadphase contract only\n";
    return 0;
}
