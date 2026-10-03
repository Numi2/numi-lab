#pragma once

// Decoder for Human's pinned FEBio rigid-graph program. This preserves the
// complete authored graph, boundaries, and load curves for one Matter case;
// decoding alone is not evidence of an accepted source-equivalent solve.
#include "numi/matter/open_knee_fiber_field.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace numi_matter_open_knee {

struct SourceRigidBodyRecord {
    std::uint32_t materialId = 0u;
    std::array<double, 3> centerOfMass{};
};
struct SourceCylindricalJointRecord {
    std::uint32_t bodyA = 0u, bodyB = 0u;
    std::array<double, 3> origin{}, axis{};
    double forcePenalty = 0.0, momentPenalty = 0.0;
    double translation = 0.0, rotation = 0.0;
    bool prescribedTranslation = false, prescribedRotation = false;
    std::int32_t translationCurve = -1, rotationCurve = -1;
};
struct SourceRigidSpringRecord {
    std::uint32_t bodyA = 0u, bodyB = 0u;
    std::array<double, 3> insertionA{}, insertionB{};
    double stiffness = 0.0, freeLength = 0.0;
};
struct SourceRigidBoundaryRecord {
    std::uint32_t bodyId = 0u;
    std::uint8_t coordinateMask = 0u;
    std::array<double, 6> values{};
    std::array<std::int32_t, 6> curveIds{};
};
struct SourceLoadCurveRecord {
    std::uint32_t id = 0u;
    std::vector<std::array<double, 2>> points;
};
struct SourceRigidGraphProgram {
    std::array<std::uint8_t, 32> deckSHA256{}, geometrySHA256{};
    std::vector<SourceRigidBodyRecord> bodies;
    std::vector<SourceCylindricalJointRecord> joints;
    std::vector<SourceRigidSpringRecord> springs;
    std::vector<SourceRigidBoundaryRecord> boundaries;
    std::vector<SourceLoadCurveRecord> curves;
};

inline bool sampleSourceLoadCurve(
    const SourceRigidGraphProgram& program, const std::int32_t curveId,
    const double sourceTime, double& value
) {
    if (curveId == -1) { value = 1.0; return true; }
    const auto curve = std::ranges::find_if(program.curves,
        [curveId](const auto& row) { return row.id ==
            static_cast<std::uint32_t>(curveId); });
    if (curve == program.curves.end() || !std::isfinite(sourceTime) ||
        curve->points.empty() || sourceTime < curve->points.front()[0] ||
        sourceTime > curve->points.back()[0]) return false;
    for (std::size_t right = 1u; right < curve->points.size(); ++right) {
        if (sourceTime > curve->points[right][0]) continue;
        const auto& a = curve->points[right - 1u];
        const auto& b = curve->points[right];
        const double alpha = (sourceTime - a[0]) / (b[0] - a[0]);
        value = a[1] + alpha * (b[1] - a[1]);
        return std::isfinite(value);
    }
    value = curve->points.back()[1];
    return std::isfinite(value);
}

inline bool decodeSourceRigidGraphProgram(
    const std::span<const std::uint8_t> bytes,
    SourceRigidGraphProgram& result, std::string& error
) {
    constexpr std::array<std::uint8_t, 8> magic{
        'N', 'H', 'R', 'G', 'P', 'H', '3', 0u};
    if (bytes.size() < 96u ||
        !std::equal(magic.begin(), magic.end(), bytes.begin()) ||
        readU32LE(bytes, 8u) != 3u || readU32LE(bytes, 12u) != 9u ||
        readU32LE(bytes, 16u) != 6u || readU32LE(bytes, 20u) != 1u ||
        readU32LE(bytes, 24u) != 2u || readU32LE(bytes, 28u) != 1u) {
        error = "source rigid-graph program header is invalid";
        return false;
    }
    SourceRigidGraphProgram candidate;
    std::copy_n(bytes.begin() + 32u, 32u, candidate.deckSHA256.begin());
    std::copy_n(bytes.begin() + 64u, 32u, candidate.geometrySHA256.begin());
    std::size_t offset = 96u;
    const auto room = [&](const std::size_t count) {
        return offset <= bytes.size() && count <= bytes.size() - offset;
    };
    const auto number = [&](const std::size_t address) {
        return readF64LE(bytes, address);
    };
    constexpr std::array<std::uint32_t, 9> expectedBodies{
        1u, 2u, 3u, 4u, 17u, 18u, 19u, 20u, 21u};
    for (const auto expectedId : expectedBodies) {
        if (!room(60u)) { error = "source rigid body table is truncated"; return false; }
        SourceRigidBodyRecord body;
        body.materialId = readU32LE(bytes, offset);
        for (std::size_t axis = 0u; axis < 3u; ++axis)
            body.centerOfMass[axis] = number(offset + 4u + 8u * axis);
        if (body.materialId != expectedId ||
            std::ranges::any_of(body.centerOfMass, [](double x) { return !std::isfinite(x); })) {
            error = "source rigid body identity or center changed";
            return false;
        }
        candidate.bodies.push_back(body);
        offset += 60u;
    }
    for (std::uint32_t index = 0u; index < 6u; ++index) {
        if (!room(136u)) { error = "source rigid joint table is truncated"; return false; }
        SourceCylindricalJointRecord joint;
        joint.bodyA = readU32LE(bytes, offset);
        joint.bodyB = readU32LE(bytes, offset + 4u);
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            joint.origin[axis] = number(offset + 8u + 8u * axis);
            joint.axis[axis] = number(offset + 32u + 8u * axis);
        }
        joint.forcePenalty = number(offset + 56u);
        joint.momentPenalty = number(offset + 64u);
        joint.translation = number(offset + 72u);
        joint.rotation = number(offset + 80u);
        const auto translationFlag = readU32LE(bytes, offset + 88u);
        const auto rotationFlag = readU32LE(bytes, offset + 92u);
        joint.prescribedTranslation = translationFlag != 0u;
        joint.prescribedRotation = rotationFlag != 0u;
        joint.translationCurve = static_cast<std::int32_t>(readU32LE(bytes, offset + 96u));
        joint.rotationCurve = static_cast<std::int32_t>(readU32LE(bytes, offset + 100u));
        const double axis2 = joint.axis[0] * joint.axis[0] +
            joint.axis[1] * joint.axis[1] + joint.axis[2] * joint.axis[2];
        if (joint.bodyA == joint.bodyB ||
            std::ranges::find(expectedBodies, joint.bodyA) == expectedBodies.end() ||
            std::ranges::find(expectedBodies, joint.bodyB) == expectedBodies.end() ||
            translationFlag > 1u || rotationFlag > 1u ||
            !std::isfinite(axis2) || std::abs(axis2 - 1.0) > 2.0e-5 ||
            !std::isfinite(joint.forcePenalty) || !(joint.forcePenalty > 0.0) ||
            !std::isfinite(joint.momentPenalty) || !(joint.momentPenalty > 0.0) ||
            !std::isfinite(joint.translation) || !std::isfinite(joint.rotation) ||
            std::ranges::any_of(joint.origin, [](double x) { return !std::isfinite(x); })) {
            error = "source cylindrical joint has invalid mechanics";
            return false;
        }
        candidate.joints.push_back(joint);
        offset += 136u;
    }
    if (!room(104u)) { error = "source rigid spring is truncated"; return false; }
    SourceRigidSpringRecord spring;
    spring.bodyA = readU32LE(bytes, offset);
    spring.bodyB = readU32LE(bytes, offset + 4u);
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
        spring.insertionA[axis] = number(offset + 8u + 8u * axis);
        spring.insertionB[axis] = number(offset + 32u + 8u * axis);
    }
    spring.stiffness = number(offset + 56u);
    spring.freeLength = number(offset + 64u);
    if (spring.bodyA == spring.bodyB ||
        std::ranges::find(expectedBodies, spring.bodyA) == expectedBodies.end() ||
        std::ranges::find(expectedBodies, spring.bodyB) == expectedBodies.end() ||
        !std::isfinite(spring.stiffness) || !(spring.stiffness > 0.0) ||
        !std::isfinite(spring.freeLength) || spring.freeLength < 0.0 ||
        std::ranges::any_of(spring.insertionA, [](double x) { return !std::isfinite(x); }) ||
        std::ranges::any_of(spring.insertionB, [](double x) { return !std::isfinite(x); })) {
        error = "source rigid spring has invalid mechanics";
        return false;
    }
    candidate.springs.push_back(spring);
    offset += 104u;
    for (std::uint32_t index = 0u; index < 2u; ++index) {
        if (!room(112u)) { error = "source rigid boundary is truncated"; return false; }
        SourceRigidBoundaryRecord boundary;
        boundary.bodyId = readU32LE(bytes, offset);
        boundary.coordinateMask = bytes[offset + 4u];
        if (bytes[offset + 5u] != 0u || bytes[offset + 6u] != 0u ||
            bytes[offset + 7u] != 0u) {
            error = "source rigid boundary padding is nonzero";
            return false;
        }
        for (std::size_t coordinate = 0u; coordinate < 6u; ++coordinate) {
            boundary.values[coordinate] = number(offset + 8u + 8u * coordinate);
            boundary.curveIds[coordinate] = static_cast<std::int32_t>(
                readU32LE(bytes, offset + 56u + 4u * coordinate));
        }
        if (boundary.bodyId != (index == 0u ? 3u : 2u) ||
            boundary.coordinateMask != 0x3fu ||
            std::ranges::any_of(boundary.values, [](double x) { return !std::isfinite(x); })) {
            error = "source prescribed rigid boundary changed";
            return false;
        }
        candidate.boundaries.push_back(boundary);
        offset += 112u;
    }
    if (!room(12u)) { error = "source load curve is truncated"; return false; }
    SourceLoadCurveRecord curve;
    curve.id = readU32LE(bytes, offset);
    const std::uint32_t interpolation = readU32LE(bytes, offset + 4u);
    const std::uint32_t count = readU32LE(bytes, offset + 8u);
    offset += 12u;
    if (curve.id != 9u || interpolation != 1u || count < 2u || count > 1024u ||
        !room(std::size_t(count) * 16u + 32u)) {
        error = "source load curve has unsupported layout";
        return false;
    }
    for (std::uint32_t point = 0u; point < count; ++point) {
        const std::array<double, 2> pair{
            number(offset + 16u * point), number(offset + 16u * point + 8u)};
        if (!std::isfinite(pair[0]) || !std::isfinite(pair[1]) ||
            (point != 0u && pair[0] <= curve.points.back()[0])) {
            error = "source load curve contains invalid points";
            return false;
        }
        curve.points.push_back(pair);
    }
    candidate.curves.push_back(std::move(curve));
    offset += std::size_t(count) * 16u + 32u;
    if (offset != bytes.size()) {
        error = "source rigid-graph program has trailing bytes";
        return false;
    }
    result = std::move(candidate);
    error.clear();
    return true;
}

} // namespace numi_matter_open_knee
