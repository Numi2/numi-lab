#pragma once

#include "numi/matter/shared.h"

#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <cstdint>

namespace numi::cardboard {

struct ReactionVec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct ConstraintReactionWrench {
    std::uint32_t nodeCount = 0u;
    ReactionVec3 pivot{};
    ReactionVec3 mechanicalConstraintForce{};
    ReactionVec3 contactForce{};
    ReactionVec3 mechanicalConstraintMoment{};
    ReactionVec3 contactMoment{};
    ReactionVec3 totalForce{};
    ReactionVec3 totalMoment{};
};

inline bool reactionConstraintTag(const float tag, const float mass) noexcept {
    return tag != 2.0f &&
        (tag == 1.0f || tag == 3.0f || !(mass > 0.0f));
}

inline ReactionVec3 forceOnFemFromRigidImpulse(
    const ReactionVec3 impulseOnRigid, const double timestepSeconds) {
    if (!std::isfinite(timestepSeconds) || !(timestepSeconds > 0.0) ||
        !std::isfinite(impulseOnRigid.x) ||
        !std::isfinite(impulseOnRigid.y) ||
        !std::isfinite(impulseOnRigid.z))
        throw std::invalid_argument("rigid contact impulse/timestep is invalid");
    return {-impulseOnRigid.x / timestepSeconds,
            -impulseOnRigid.y / timestepSeconds,
            -impulseOnRigid.z / timestepSeconds};
}

inline ReactionVec3 forceOnFemFromDeformableResidualImpulse(
    const ReactionVec3 localBaseImpulse,
    const double sourceTimeScale,
    const double sourceTimestepSeconds) {
    if (!std::isfinite(sourceTimeScale) || !(sourceTimeScale > 0.0) ||
        !std::isfinite(sourceTimestepSeconds) ||
        !(sourceTimestepSeconds > 0.0) ||
        !std::isfinite(localBaseImpulse.x) ||
        !std::isfinite(localBaseImpulse.y) ||
        !std::isfinite(localBaseImpulse.z))
        throw std::invalid_argument("deformable contact impulse/timestep is invalid");
    return {sourceTimeScale * localBaseImpulse.x / sourceTimestepSeconds,
            sourceTimeScale * localBaseImpulse.y / sourceTimestepSeconds,
            sourceTimeScale * localBaseImpulse.z / sourceTimestepSeconds};
}

inline ReactionVec3 momentAbout(const ReactionVec3 point,
                                const ReactionVec3 pivot,
                                const ReactionVec3 force) noexcept {
    const double rx = point.x - pivot.x;
    const double ry = point.y - pivot.y;
    const double rz = point.z - pivot.z;
    return {ry * force.z - rz * force.y,
            rz * force.x - rx * force.z,
            rx * force.y - ry * force.x};
}

inline ConstraintReactionWrench summarizeConstraintReactions(
    const std::span<const NMFEMNodeStateGPU> acceptedNodes,
    const std::span<const nm_float4> mechanicalConstraintForces,
    const std::span<const nm_float4> contactForces,
    const std::size_t environmentBase,
    const std::size_t environmentNodeCount,
    const std::span<const std::uint32_t> localNodeIndices,
    const std::optional<ReactionVec3> requestedPivot = std::nullopt) {
    ConstraintReactionWrench result;
    ReactionVec3 centroid{};
    std::uint32_t centroidCount = 0u;
    if (environmentBase > acceptedNodes.size() ||
        environmentNodeCount > acceptedNodes.size() - environmentBase ||
        mechanicalConstraintForces.size() < acceptedNodes.size() ||
        contactForces.size() < acceptedNodes.size())
        throw std::invalid_argument("reaction arrays do not cover the selected FEM environment");
    for (const std::uint32_t local : localNodeIndices) {
        if (local >= environmentNodeCount)
            throw std::out_of_range("reaction node index is outside the FEM environment");
        const std::size_t index = environmentBase + local;
        const auto& node = acceptedNodes[index];
        if (!reactionConstraintTag(node.restAndFixed.w, node.positionAndMass.w))
            continue;
        centroid.x += node.positionAndMass.x;
        centroid.y += node.positionAndMass.y;
        centroid.z += node.positionAndMass.z;
        ++centroidCount;
    }
    if (requestedPivot) result.pivot = *requestedPivot;
    else if (centroidCount > 0u) {
        result.pivot = {centroid.x / centroidCount,
                        centroid.y / centroidCount,
                        centroid.z / centroidCount};
    }
    for (const std::uint32_t local : localNodeIndices) {
        const std::size_t index = environmentBase + local;
        const auto& node = acceptedNodes[index];
        if (!reactionConstraintTag(node.restAndFixed.w, node.positionAndMass.w))
            continue;
        const auto& bulk = mechanicalConstraintForces[index];
        const auto& contact = contactForces[index];
        const ReactionVec3 bulkValue{bulk.x, bulk.y, bulk.z};
        const ReactionVec3 contactValue{contact.x, contact.y, contact.z};
        const ReactionVec3 point{node.positionAndMass.x,
                                 node.positionAndMass.y,
                                 node.positionAndMass.z};
        const ReactionVec3 bulkMoment = momentAbout(
            point, result.pivot, bulkValue);
        const ReactionVec3 contactMoment = momentAbout(
            point, result.pivot, contactValue);
        result.mechanicalConstraintForce.x += bulkValue.x;
        result.mechanicalConstraintForce.y += bulkValue.y;
        result.mechanicalConstraintForce.z += bulkValue.z;
        result.contactForce.x += contactValue.x;
        result.contactForce.y += contactValue.y;
        result.contactForce.z += contactValue.z;
        result.mechanicalConstraintMoment.x += bulkMoment.x;
        result.mechanicalConstraintMoment.y += bulkMoment.y;
        result.mechanicalConstraintMoment.z += bulkMoment.z;
        result.contactMoment.x += contactMoment.x;
        result.contactMoment.y += contactMoment.y;
        result.contactMoment.z += contactMoment.z;
        ++result.nodeCount;
    }
    result.totalForce = {
        result.mechanicalConstraintForce.x + result.contactForce.x,
        result.mechanicalConstraintForce.y + result.contactForce.y,
        result.mechanicalConstraintForce.z + result.contactForce.z};
    result.totalMoment = {
        result.mechanicalConstraintMoment.x + result.contactMoment.x,
        result.mechanicalConstraintMoment.y + result.contactMoment.y,
        result.mechanicalConstraintMoment.z + result.contactMoment.z};
    return result;
}

} // namespace numi::cardboard
