#pragma once

#include "metalrobo/EngineModel.hpp"
#include "metalrobo/mujoco_muscle_gpu.h"
#include "numi/matter/human_limits_gpu.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace numi_human_brain {

struct SupportEndpoint {
    std::uint32_t bodyIdentifier = 0u;
    std::uint32_t sourceGeometryIndex = 0u;
    std::uint32_t touchReceptorIndex = 0u;
};

// Source geometry evaluated once at the exact FP32 pose submitted as the
// first native Human q. Jacobians are d(path length)/d(v) in muscle-major,
// articulated v6..v127 order. This is preparation data, not a controller.
struct JointPathCalibration {
    std::span<const float> referenceQ;
    std::span<const std::uint32_t> optimalFiberLengthBitsByMuscle;
    std::span<const std::uint32_t> lengthJacobianBitsByMuscleDof;
};

namespace source_detail {

inline void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

inline bool finite(mr_float4 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
        std::isfinite(value.z) && std::isfinite(value.w);
}

// Same component order and arithmetic as NumanXRuntimeV1's anatomy exporter.
inline mr_float4 multiply(mr_float4 left, mr_float4 right) {
    return {
        left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
        left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
        left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
        left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z};
}

inline mr_float4 conjugate(mr_float4 value) {
    return {-value.x, -value.y, -value.z, value.w};
}

inline mr_float4 rotate(mr_float4 rotation, mr_float4 axis) {
    const auto rotated = multiply(multiply(rotation, {axis.x, axis.y, axis.z, 0.0f}),
                                  conjugate(rotation));
    return {rotated.x, rotated.y, rotated.z, 0.0f};
}

inline void point(std::ostream& output, mr_float4 value) {
    require(finite(value), "Human Brain source point is nonfinite");
    output << "{\"x\":" << value.x << ",\"y\":" << value.y
           << ",\"z\":" << value.z << '}';
}

inline void quaternion(std::ostream& output, mr_float4 value) {
    require(finite(value) &&
                value.x * value.x + value.y * value.y + value.z * value.z +
                    value.w * value.w > 1.0e-12f,
            "Human Brain source orientation is nonfinite or degenerate");
    output << "{\"x\":" << value.x << ",\"y\":" << value.y
           << ",\"z\":" << value.z << ",\"w\":" << value.w << '}';
}

} // namespace source_detail

// Serialize admitted source anatomy and the SAME native preparation's muscle
// calibration. This neither evaluates mechanics nor chooses a controller.
// modelSourceFingerprint is the caller's composed actual source/preparation
// identity, not a substitute hash of this JSON. Head identity is explicit;
// imported EngineModel::bodyNames can legitimately be empty.
//
// NumanXFullBodyTransportTemplate binds locomotor length/velocity to native
// proprioception features 4/5: PATH metres and metres/second. State.z is FIBRE
// metres (feature 2), exported separately and never substituted as the path
// reference. No lengths here are normalized.
inline std::string makeSourceJSON(
    const metalrobo::EngineModel& model,
    std::span<const MRMujocoMuscleGPU> muscles,
    std::span<const MRMujocoMuscleSiteGPU> sites,
    std::span<const MRMujocoMuscleRouteNodeGPU> routes,
    std::span<const MRMujocoMuscleStateGPU> preparedStates,
    std::span<const MRMujocoMuscleResultGPU> preparedResults,
    std::uint64_t modelSourceFingerprint,
    std::uint32_t headBodyIdentifier,
    std::span<const NMHumanJointLimitGPU> sourceJointLimits = {},
    std::span<const SupportEndpoint> supportEndpoints = {},
    const JointPathCalibration* jointPathCalibration = nullptr
) {
    using source_detail::require;
    require(modelSourceFingerprint != 0u && !model.bodies.empty() &&
                model.bodies.size() == model.world.bodyCount &&
                model.dofs.size() == model.world.nv &&
                model.defaultQ.size() == model.world.nq &&
                headBodyIdentifier < model.bodies.size() &&
                model.joints.size() <= std::numeric_limits<std::uint32_t>::max() &&
                !muscles.empty() && muscles.size() <= std::numeric_limits<std::uint32_t>::max() &&
                preparedStates.size() == muscles.size() &&
                preparedResults.size() == muscles.size(),
            "Human Brain source identity, dimensions or prepared calibration is incomplete");

    std::vector<const NMHumanJointLimitGPU*> limits(model.dofs.size(), nullptr);
    for (const auto& limit : sourceJointLimits) {
        require(limit.indices.y < limits.size() && limits[limit.indices.y] == nullptr &&
                    limit.indices.w == 0u &&
                    model.dofs[limit.indices.y].qIndex == limit.indices.x &&
                    (model.dofs[limit.indices.y].flags & MR_DOF_FLAG_ROOT) == 0u &&
                    std::isfinite(limit.rangeMarginInverseWeight.x) &&
                    std::isfinite(limit.rangeMarginInverseWeight.y) &&
                    limit.rangeMarginInverseWeight.x < limit.rangeMarginInverseWeight.y,
                "Human Brain source-compliant joint limit is malformed or repeated");
        limits[limit.indices.y] = &limit;
    }

    // HumanIO numbers touch receptors by first distinct source geometry in
    // the admitted NHCNT support payload. Preserve that same native order.
    if (!supportEndpoints.empty()) {
        require(supportEndpoints.size() == 10u,
                "Human Brain support source requires ten touch endpoints");
        std::set<std::pair<std::uint32_t, std::uint32_t>> sourceIdentities;
        for (std::size_t index = 0u; index < supportEndpoints.size(); ++index) {
            const auto& endpoint = supportEndpoints[index];
            require(endpoint.bodyIdentifier < model.bodies.size() &&
                        endpoint.sourceGeometryIndex != MR_INVALID_INDEX &&
                        endpoint.touchReceptorIndex == index &&
                        sourceIdentities.insert({endpoint.bodyIdentifier,
                                                 endpoint.sourceGeometryIndex}).second &&
                        (endpoint.bodyIdentifier != 0u || endpoint.sourceGeometryIndex != 0u),
                    "Human Brain support source identity or touch order is invalid");
        }
    }

    if (jointPathCalibration != nullptr) {
        require(!supportEndpoints.empty() && model.world.nq == 129u &&
                    model.world.nv == 128u && model.dofs.size() == 128u &&
                    muscles.size() == 416u &&
                    jointPathCalibration->referenceQ.size() == 129u &&
                    jointPathCalibration->optimalFiberLengthBitsByMuscle.size() == 416u &&
                    jointPathCalibration->lengthJacobianBitsByMuscleDof.size() == 416u * 122u,
                "Human Brain joint path calibration dimensions or support source are invalid");
        std::array<bool, 129u> ownedQ{};
        for (std::uint32_t vIndex = 6u; vIndex < 128u; ++vIndex) {
            const auto& dof = model.dofs[vIndex];
            require(dof.vIndex == vIndex && dof.qIndex >= 7u && dof.qIndex < 129u &&
                        (dof.flags & MR_DOF_FLAG_ROOT) == 0u &&
                        !ownedQ[dof.qIndex] &&
                        std::isfinite(jointPathCalibration->referenceQ[dof.qIndex]),
                    "Human Brain joint path reference coordinate is absent or duplicated");
            ownedQ[dof.qIndex] = true;
        }
        for (std::uint32_t qIndex = 7u; qIndex < 129u; ++qIndex)
            require(ownedQ[qIndex], "Human Brain joint path reference coverage is incomplete");
        for (std::size_t muscle = 0u; muscle < 416u; ++muscle) {
            const float optimum = std::bit_cast<float>(
                jointPathCalibration->optimalFiberLengthBitsByMuscle[muscle]);
            require(std::isfinite(optimum) && optimum > 0.0f &&
                        std::bit_cast<std::uint32_t>(optimum) ==
                            std::bit_cast<std::uint32_t>(muscles[muscle].compliantArchitecture0.x),
                    "Human Brain joint path optimum is not the source muscle value");
        }
        for (const std::uint32_t bits : jointPathCalibration->lengthJacobianBitsByMuscleDof)
            require(std::isfinite(std::bit_cast<float>(bits)),
                    "Human Brain joint path Jacobian is nonfinite");
    }

    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<float>::max_digits10)
           << "{\"version\":" << (jointPathCalibration != nullptr ? 3u :
                (supportEndpoints.empty() ? 1u : 2u))
           << ",\"modelSourceFingerprint\":" << modelSourceFingerprint
           << ",\"bodyCount\":" << model.bodies.size()
           << ",\"headBodyIdentifier\":" << headBodyIdentifier << ",\"joints\":[";
    std::vector<bool> ownedCoordinates(model.dofs.size(), false);
    for (std::size_t index = 0u; index < model.joints.size(); ++index) {
        const auto& joint = model.joints[index];
        require(joint.parentBody < model.bodies.size() &&
                    joint.childBody < model.bodies.size() && joint.parentBody != joint.childBody &&
                    joint.vOffset <= model.dofs.size() &&
                    joint.nv <= model.dofs.size() - joint.vOffset &&
                    source_detail::finite(joint.parentRotation) &&
                    source_detail::finite(joint.childRotation),
                "Human Brain joint anatomy is malformed");
        for (std::uint32_t local = 0u; local < joint.nv; ++local) {
            const auto global = joint.vOffset + local;
            const auto& dof = model.dofs[global];
            require(!ownedCoordinates[global] && dof.jointIndex == index &&
                        dof.localDof == local && dof.vIndex == global &&
                        (dof.flags & MR_DOF_FLAG_ROOT) == 0u,
                    "Human Brain joint coordinate ownership is inconsistent");
            ownedCoordinates[global] = true;
        }
        if (index != 0u) output << ',';
        output << "{\"jointIdentifier\":" << index
               << ",\"parentBodyIdentifier\":" << joint.parentBody
               << ",\"childBodyIdentifier\":" << joint.childBody
               << ",\"coordinateOffset\":" << joint.vOffset
               << ",\"coordinateCount\":" << joint.nv << ",\"parentLocalAnchor\":";
        source_detail::point(output, joint.parentAnchor);
        output << ",\"childLocalAnchor\":";
        source_detail::point(output, joint.childAnchor);
        output << ",\"restRelativeOrientation\":";
        source_detail::quaternion(output, source_detail::multiply(
            joint.parentRotation, source_detail::conjugate(joint.childRotation)));
        output << '}';
    }
    output << "],\"coordinates\":[";
    bool firstCoordinate = true;
    for (std::size_t index = 0u; index < model.dofs.size(); ++index) {
        const auto& dof = model.dofs[index];
        if ((dof.flags & MR_DOF_FLAG_ROOT) != 0u) continue;
        require(ownedCoordinates[index] && dof.jointIndex < model.joints.size() &&
                    dof.vIndex == index && dof.reserved0 == 0u && dof.reserved1 == 0u,
                "Human Brain coordinate anatomy is malformed or unowned");
        const auto& joint = model.joints[dof.jointIndex];
        require(dof.localDof < joint.nv && dof.localDof < 3u &&
                    joint.jointType != MR_JOINT_FUNCTION_BASED &&
                    joint.jointType != MR_JOINT_FREE,
                "Human Brain coordinate kind is unsupported by native anatomy transport");
        const auto axis = dof.localDof == 0u ? joint.axis0
            : (dof.localDof == 1u ? joint.axis1 : joint.axis2);
        const auto parentAxis = source_detail::rotate(joint.parentRotation, axis);
        require(source_detail::finite(parentAxis) &&
                    parentAxis.x * parentAxis.x + parentAxis.y * parentAxis.y +
                        parentAxis.z * parentAxis.z > 1.0e-12f,
                "Human Brain coordinate axis is nonfinite or degenerate");
        const bool linear = joint.jointType == MR_JOINT_PRISMATIC ||
            (joint.jointType == MR_JOINT_PLANAR && dof.localDof < 2u);
        const auto* sourceLimit = limits[index];
        const bool hasLimit = sourceLimit != nullptr || (dof.flags & MR_DOF_FLAG_POSITION_LIMIT) != 0u;
        const float extent = std::numeric_limits<float>::max();
        const float minimum = sourceLimit != nullptr ? sourceLimit->rangeMarginInverseWeight.x
            : (hasLimit ? dof.limits.x : -extent);
        const float maximum = sourceLimit != nullptr ? sourceLimit->rangeMarginInverseWeight.y
            : (hasLimit ? dof.limits.y : extent);
        const float rest = dof.qIndex != MR_INVALID_INDEX && dof.qIndex < model.defaultQ.size()
            ? model.defaultQ[dof.qIndex] : 0.0f;
        require(std::isfinite(minimum) && std::isfinite(maximum) && std::isfinite(rest) &&
                    minimum < maximum && (sourceLimit != nullptr || (rest >= minimum && rest <= maximum)),
                "Human Brain coordinate limits or source rest position are invalid");
        if (!firstCoordinate) output << ',';
        firstCoordinate = false;
        output << "{\"jointIdentifier\":" << dof.jointIndex
               << ",\"identifier\":" << dof.localDof << ",\"kind\":" << (linear ? 2u : 1u)
               << ",\"qIndex\":" << dof.qIndex << ",\"vIndex\":" << dof.vIndex
               << ",\"parentLocalAxis\":";
        source_detail::point(output, parentAxis);
        output << ",\"minimumPosition\":" << minimum << ",\"maximumPosition\":" << maximum
               << ",\"restPosition\":" << rest << ",\"sourceCompliantLimit\":"
               << (sourceLimit != nullptr ? "true" : "false") << '}';
    }
    output << "],\"attachments\":[";
    for (std::size_t index = 0u; index < muscles.size(); ++index) {
        const auto& muscle = muscles[index];
        const auto offset = muscle.route.x, count = muscle.route.y;
        require(count >= 2u && offset <= routes.size() && count <= routes.size() - offset,
                "Human Brain muscle route range is invalid");
        const auto& first = routes[offset];
        const auto& last = routes[offset + count - 1u];
        require(first.type == MR_MUJOCO_MUSCLE_ROUTE_SITE &&
                    last.type == MR_MUJOCO_MUSCLE_ROUTE_SITE &&
                    first.targetIndex < sites.size() && last.targetIndex < sites.size(),
                "Human Brain muscle route endpoints are not source sites");
        const auto& origin = sites[first.targetIndex];
        const auto& insertion = sites[last.targetIndex];
        require(origin.bodyIndex < model.bodies.size() && insertion.bodyIndex < model.bodies.size(),
                "Human Brain muscle attachment body is invalid");
        if (index != 0u) output << ',';
        output << "{\"muscleIdentifier\":" << index << ",\"routeNodeCount\":" << count
               << ",\"firstBodyIdentifier\":" << origin.bodyIndex
               << ",\"terminalBodyIdentifier\":" << insertion.bodyIndex
               << ",\"firstLocalPoint\":";
        source_detail::point(output, origin.localPoint);
        output << ",\"terminalLocalPoint\":";
        source_detail::point(output, insertion.localPoint);
        output << '}';
    }
    output << "],\"channels\":[";
    for (std::size_t index = 0u; index < muscles.size(); ++index) {
        const auto& state = preparedStates[index].excitationAndActivation;
        const auto& result = preparedResults[index];
        require(source_detail::finite(state) && state.x >= 0.0f && state.x <= 1.0f &&
                    state.y >= 0.0f && state.y <= 1.0f && state.z >= 0.0f &&
                    result.status == MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS &&
                    result.environment == 0u && result.muscleIndex == index &&
                    source_detail::finite(result.pathForceAndActivationDerivative) &&
                    result.pathForceAndActivationDerivative.x > 0.0f,
                "Human Brain prepared muscle calibration is stale, failed or nonfinite");
        if (index != 0u) output << ',';
        output << "{\"muscleIdentifier\":" << index
               << ",\"referenceLengthMeters\":" << result.pathForceAndActivationDerivative.x
               << ",\"referenceFiberLengthMeters\":" << state.z
               << ",\"tonicExcitation\":" << state.x
               << ",\"preparedActivation\":" << state.y << '}';
    }
    output << ']';
    if (!supportEndpoints.empty()) {
        output << ",\"supportEndpoints\":[";
        for (std::size_t index = 0u; index < supportEndpoints.size(); ++index) {
            const auto& endpoint = supportEndpoints[index];
            const std::uint64_t sourceEndpointIdentifier =
                (std::uint64_t{endpoint.bodyIdentifier} << 32u) |
                endpoint.sourceGeometryIndex;
            if (index != 0u) output << ',';
            output << "{\"sourceEndpointIdentifier\":" << sourceEndpointIdentifier
                   << ",\"bodyIdentifier\":" << endpoint.bodyIdentifier
                   << ",\"sourceGeometryIndex\":" << endpoint.sourceGeometryIndex
                   << ",\"touchReceptorIndex\":" << endpoint.touchReceptorIndex << '}';
        }
        output << ']';
    }
    if (jointPathCalibration != nullptr) {
        output << ",\"jointPathCalibration\":{\"version\":1,\"referencePositionBitsByDof\":[";
        for (std::uint32_t vIndex = 6u; vIndex < 128u; ++vIndex) {
            if (vIndex != 6u) output << ',';
            output << std::bit_cast<std::uint32_t>(
                jointPathCalibration->referenceQ[model.dofs[vIndex].qIndex]);
        }
        output << "],\"optimalFiberLengthBitsByMuscle\":[";
        for (std::size_t muscle = 0u; muscle < 416u; ++muscle) {
            if (muscle != 0u) output << ',';
            output << jointPathCalibration->optimalFiberLengthBitsByMuscle[muscle];
        }
        output << "],\"lengthJacobianBitsByMuscleDof\":[";
        for (std::size_t index = 0u;
             index < jointPathCalibration->lengthJacobianBitsByMuscleDof.size(); ++index) {
            if (index != 0u) output << ',';
            output << jointPathCalibration->lengthJacobianBitsByMuscleDof[index];
        }
        output << "]}";
    }
    output << '}';
    return output.str();
}

} // namespace numi_human_brain
