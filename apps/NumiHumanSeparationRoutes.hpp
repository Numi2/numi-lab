#pragma once

#include "metalrobo/EngineModel.hpp"
#include "metalrobo/mujoco_muscle_gpu.h"

#include <cstdint>
#include <span>
#include <vector>

namespace metalrobo::separation {

enum class RouteClassificationStatus : std::uint32_t { supported = 0u, unsupported };
enum class RouteClassificationIssueKind : std::uint32_t {
    invalidModelTopology = 0u, invalidSeam, invalidBodyOwner, invalidSiteRecord,
    invalidWrapRecord, invalidRouteRecord, unknownRouteKind, invalidMuscleRecord,
    emptyMuscleRoute,
};

struct RouteClassificationIssue {
    RouteClassificationIssueKind kind = RouteClassificationIssueKind::invalidModelTopology;
    std::uint32_t sourceIndex = MR_INVALID_INDEX;
    std::uint32_t detail = MR_INVALID_INDEX;
};

struct MuscleRoutePartition {
    RouteClassificationStatus status = RouteClassificationStatus::unsupported;
    std::uint32_t seamJointIndex = MR_INVALID_INDEX;
    std::uint32_t parentBodyIndex = MR_INVALID_INDEX;
    std::uint32_t childBodyIndex = MR_INVALID_INDEX;
    std::vector<std::uint8_t> childSideBodyMask;
    // Bit 0: parent side. Bit 1: child side. A crossing muscle has mask 3.
    std::vector<std::uint8_t> siteSideMasks;
    std::vector<std::uint8_t> wrapSideMasks;
    std::vector<std::uint8_t> routeNodeSideMasks;
    std::vector<std::uint8_t> muscleOwnerSideMasks;
    // Source MRMujocoMuscleGPU order; these indices remain stable on failure.
    std::vector<std::uint32_t> crossingMuscleIndices;
    std::vector<RouteClassificationIssue> issues;

    [[nodiscard]] bool supported() const noexcept {
        return status == RouteClassificationStatus::supported;
    }
};

// Classify force routes against the child subtree of an authored rigid-joint
// tree edge. Site, wrap, and conditional side-site owners are all included.
// Unknown node kinds or malformed references make the whole result
// unsupported; known crossings remain diagnostic only and are not a complete
// removal list in that state.
[[nodiscard]] inline MuscleRoutePartition classifyMuscleRoutesAcrossJoint(
    const EngineModel& model,
    const std::span<const MRMujocoMuscleGPU> muscles,
    const std::span<const MRMujocoMuscleSiteGPU> sites,
    const std::span<const MRMujocoMuscleWrapGPU> wraps,
    const std::span<const MRMujocoMuscleRouteNodeGPU> routeNodes,
    const std::uint32_t seamJointIndex = 16u
) {
    MuscleRoutePartition result{};
    result.seamJointIndex = seamJointIndex;
    const auto addIssue = [&result](const RouteClassificationIssueKind kind,
                                    const std::uint32_t sourceIndex,
                                    const std::uint32_t detail) {
        result.issues.push_back({kind, sourceIndex, detail});
    };

    if (model.world.bodyCount != model.bodies.size() ||
        model.world.jointCount != model.joints.size() ||
        model.world.articulationCount != model.articulations.size() ||
        seamJointIndex >= model.joints.size()) {
        addIssue(RouteClassificationIssueKind::invalidModelTopology,
                 seamJointIndex, MR_INVALID_INDEX);
        return result;
    }
    const MRJointDescriptorGPU& seam = model.joints[seamJointIndex];
    if (seam.parentBody >= model.bodies.size() ||
        seam.childBody >= model.bodies.size() || seam.parentBody == seam.childBody) {
        addIssue(RouteClassificationIssueKind::invalidSeam,
                 seamJointIndex, seam.childBody);
        return result;
    }
    result.parentBodyIndex = seam.parentBody;
    result.childBodyIndex = seam.childBody;
    const std::uint32_t articulationIndex = model.bodies[seam.parentBody].articulationIndex;
    if (articulationIndex >= model.articulations.size() ||
        model.bodies[seam.childBody].articulationIndex != articulationIndex) {
        addIssue(RouteClassificationIssueKind::invalidSeam,
                 seamJointIndex, articulationIndex);
        return result;
    }
    const MRArticulationGPU& articulation = model.articulations[articulationIndex];
    const std::uint64_t bodyEnd = std::uint64_t(articulation.firstBody) + articulation.bodyCount;
    const std::uint64_t jointEnd = std::uint64_t(articulation.firstJoint) + articulation.jointCount;
    if (articulation.rootBody < articulation.firstBody || articulation.rootBody >= bodyEnd ||
        bodyEnd > model.bodies.size() || jointEnd > model.joints.size() ||
        seamJointIndex < articulation.firstJoint || seamJointIndex >= jointEnd ||
        seam.parentBody < articulation.firstBody || seam.parentBody >= bodyEnd ||
        seam.childBody < articulation.firstBody || seam.childBody >= bodyEnd ||
        model.bodies[seam.childBody].parentBody != seam.parentBody ||
        model.bodies[seam.childBody].inboundJoint != seamJointIndex) {
        addIssue(RouteClassificationIssueKind::invalidSeam,
                 seamJointIndex, articulationIndex);
        return result;
    }

    std::vector<std::vector<std::uint32_t>> children(model.bodies.size());
    std::uint32_t rootCount = 0u;
    bool topologyValid = true;
    for (std::uint32_t body = articulation.firstBody; body < bodyEnd; ++body) {
        const MRBodyPropertiesGPU& properties = model.bodies[body];
        if (properties.articulationIndex != articulationIndex) {
            topologyValid = false;
            addIssue(RouteClassificationIssueKind::invalidModelTopology,
                     body, properties.articulationIndex);
            continue;
        }
        if (body == articulation.rootBody) {
            ++rootCount;
            if (properties.parentBody != MR_INVALID_INDEX ||
                properties.inboundJoint != MR_INVALID_INDEX) {
                topologyValid = false;
                addIssue(RouteClassificationIssueKind::invalidModelTopology,
                         body, properties.inboundJoint);
            }
            continue;
        }
        if (properties.parentBody < articulation.firstBody || properties.parentBody >= bodyEnd ||
            properties.inboundJoint < articulation.firstJoint || properties.inboundJoint >= jointEnd) {
            topologyValid = false;
            addIssue(RouteClassificationIssueKind::invalidModelTopology,
                     body, properties.inboundJoint);
            continue;
        }
        const MRJointDescriptorGPU& inbound = model.joints[properties.inboundJoint];
        if (inbound.parentBody != properties.parentBody || inbound.childBody != body) {
            topologyValid = false;
            addIssue(RouteClassificationIssueKind::invalidModelTopology,
                     properties.inboundJoint, body);
            continue;
        }
        children[properties.parentBody].push_back(body);
    }
    if (rootCount != 1u) {
        topologyValid = false;
        addIssue(RouteClassificationIssueKind::invalidModelTopology,
                 articulationIndex, rootCount);
    }
    for (std::uint32_t jointIndex = articulation.firstJoint; jointIndex < jointEnd; ++jointIndex) {
        const MRJointDescriptorGPU& joint = model.joints[jointIndex];
        if (joint.parentBody < articulation.firstBody || joint.parentBody >= bodyEnd ||
            joint.childBody < articulation.firstBody || joint.childBody >= bodyEnd ||
            joint.childBody == articulation.rootBody ||
            model.bodies[joint.childBody].inboundJoint != jointIndex ||
            model.bodies[joint.childBody].parentBody != joint.parentBody) {
            topologyValid = false;
            addIssue(RouteClassificationIssueKind::invalidModelTopology,
                     jointIndex, joint.childBody);
        }
    }
    if (!topologyValid) return result;

    result.childSideBodyMask.assign(model.bodies.size(), 0u);
    std::vector<std::uint32_t> pending{seam.childBody};
    while (!pending.empty()) {
        const std::uint32_t body = pending.back();
        pending.pop_back();
        if (result.childSideBodyMask[body] != 0u) continue;
        result.childSideBodyMask[body] = 1u;
        for (const std::uint32_t child : children[body]) pending.push_back(child);
    }
    std::vector<std::uint8_t> reachable(model.bodies.size(), 0u);
    pending.assign(1u, articulation.rootBody);
    while (!pending.empty()) {
        const std::uint32_t body = pending.back();
        pending.pop_back();
        if (reachable[body] != 0u) continue;
        reachable[body] = 1u;
        for (const std::uint32_t child : children[body]) pending.push_back(child);
    }
    std::uint32_t reachableCount = 0u;
    for (std::uint32_t body = articulation.firstBody; body < bodyEnd; ++body)
        reachableCount += reachable[body] != 0u;
    if (reachableCount != articulation.bodyCount || result.childSideBodyMask[seam.childBody] == 0u) {
        addIssue(RouteClassificationIssueKind::invalidModelTopology,
                 articulationIndex, reachableCount);
        return result;
    }

    const auto bodySideMask = [&result, &model, &articulation, articulationIndex, &addIssue](
        const std::uint32_t bodyIndex, const RouteClassificationIssueKind kind,
        const std::uint32_t sourceIndex) -> std::uint8_t {
        if (bodyIndex >= model.bodies.size() ||
            model.bodies[bodyIndex].articulationIndex != articulationIndex ||
            bodyIndex < articulation.firstBody ||
            std::uint64_t(bodyIndex) >= std::uint64_t(articulation.firstBody) + articulation.bodyCount) {
            addIssue(kind, sourceIndex, bodyIndex);
            return 0u;
        }
        return result.childSideBodyMask[bodyIndex] != 0u ? 2u : 1u;
    };

    result.siteSideMasks.resize(sites.size(), 0u);
    for (std::uint32_t i = 0u; i < sites.size(); ++i) {
        const MRMujocoMuscleSiteGPU& site = sites[i];
        if (site.reserved0 != 0u || site.reserved1 != 0u || site.reserved2 != 0u)
            addIssue(RouteClassificationIssueKind::invalidSiteRecord, i, site.bodyIndex);
        result.siteSideMasks[i] = bodySideMask(
            site.bodyIndex, RouteClassificationIssueKind::invalidBodyOwner, i);
    }
    result.wrapSideMasks.resize(wraps.size(), 0u);
    for (std::uint32_t i = 0u; i < wraps.size(); ++i) {
        const MRMujocoMuscleWrapGPU& wrap = wraps[i];
        if (wrap.reserved0 != 0u || wrap.reserved1 != 0u)
            addIssue(RouteClassificationIssueKind::invalidWrapRecord, i, wrap.bodyIndex);
        result.wrapSideMasks[i] = bodySideMask(
            wrap.bodyIndex, RouteClassificationIssueKind::invalidBodyOwner, i);
    }

    result.routeNodeSideMasks.resize(routeNodes.size(), 0u);
    for (std::uint32_t i = 0u; i < routeNodes.size(); ++i) {
        const MRMujocoMuscleRouteNodeGPU& node = routeNodes[i];
        if (node.reserved0 != 0u)
            addIssue(RouteClassificationIssueKind::invalidRouteRecord, i, node.reserved0);
        std::uint8_t mask = 0u;
        if (node.type == MR_MUJOCO_MUSCLE_ROUTE_SITE) {
            if (node.targetIndex >= sites.size())
                addIssue(RouteClassificationIssueKind::invalidRouteRecord, i, node.targetIndex);
            else mask |= result.siteSideMasks[node.targetIndex];
        } else if (node.type == MR_MUJOCO_MUSCLE_ROUTE_SPHERE ||
                   node.type == MR_MUJOCO_MUSCLE_ROUTE_CYLINDER) {
            if (node.targetIndex >= wraps.size()) {
                addIssue(RouteClassificationIssueKind::invalidRouteRecord, i, node.targetIndex);
            } else {
                const MRMujocoMuscleWrapGPU& wrap = wraps[node.targetIndex];
                if (wrap.type != node.type)
                    addIssue(RouteClassificationIssueKind::invalidRouteRecord, i, node.targetIndex);
                mask |= result.wrapSideMasks[node.targetIndex];
            }
        } else {
            addIssue(RouteClassificationIssueKind::unknownRouteKind, i, node.type);
        }
        if (node.sideSiteIndex != MR_INVALID_INDEX) {
            if (node.sideSiteIndex >= sites.size())
                addIssue(RouteClassificationIssueKind::invalidRouteRecord, i, node.sideSiteIndex);
            else mask |= result.siteSideMasks[node.sideSiteIndex];
        }
        result.routeNodeSideMasks[i] = mask;
    }

    result.muscleOwnerSideMasks.resize(muscles.size(), 0u);
    for (std::uint32_t i = 0u; i < muscles.size(); ++i) {
        const MRMujocoMuscleGPU& muscle = muscles[i];
        const std::uint32_t offset = muscle.route.x;
        const std::uint32_t count = muscle.route.y;
        if (muscle.route.z != 0u || muscle.route.w != 0u ||
            std::uint64_t(offset) + count > routeNodes.size()) {
            addIssue(RouteClassificationIssueKind::invalidMuscleRecord, i, offset);
            continue;
        }
        if (count == 0u) {
            addIssue(RouteClassificationIssueKind::emptyMuscleRoute, i, offset);
            continue;
        }
        std::uint8_t mask = 0u;
        for (std::uint32_t node = offset; node < offset + count; ++node)
            mask |= result.routeNodeSideMasks[node];
        result.muscleOwnerSideMasks[i] = mask;
        if (mask == 3u) result.crossingMuscleIndices.push_back(i);
    }

    result.status = result.issues.empty()
        ? RouteClassificationStatus::supported
        : RouteClassificationStatus::unsupported;
    return result;
}

} // namespace metalrobo::separation
