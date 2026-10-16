#pragma once

#include "metalrobo/ArticulatedDynamics.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace metalrobo::human::observer {

inline constexpr std::string_view kBodyMotionPartitionCsvHeader =
    "accepted_step,time_s,body_index,body_name,mass_kg,root_x_m,root_y_m,root_z_m,"
    "com_relative_root_x_m,com_relative_root_y_m,com_relative_root_z_m,"
    "mass_weighted_relative_com_x_kg_m,mass_weighted_relative_com_y_kg_m,"
    "mass_weighted_relative_com_z_kg_m,orientation_x,orientation_y,orientation_z,"
    "orientation_w,linear_velocity_x_m_s,linear_velocity_y_m_s,linear_velocity_z_m_s,"
    "angular_velocity_x_rad_s,angular_velocity_y_rad_s,angular_velocity_z_rad_s,"
    "linear_momentum_x_kg_m_s,linear_momentum_y_kg_m_s,linear_momentum_z_kg_m_s";

inline constexpr std::string_view kSupportPointMotionCsvHeader =
    "accepted_step,accepted_time_s,pre_step_index,pre_step_time_s,contact_index,"
    "source_geometry_index,body_index,point_query_index,point_world_x_m,point_world_y_m,point_world_z_m,"
    "pre_step_velocity_fingerprint_fnv64,point_velocity_world_x_m_s,"
    "point_velocity_world_y_m_s,point_velocity_world_z_m_s,"
    "tangent0_velocity_m_s,tangent1_velocity_m_s,tangential_speed_m_s,"
    "velocity_basis";

inline constexpr std::string_view kSupportSelectorCsvHeader =
    "accepted_step,accepted_time_s,pre_dynamics_step_index,pre_dynamics_time_s,"
    "region,source_geometry_index,point_query_index,selected_skin_vertex_local,selected_vertex_map_index,"
    "region_error,global_error,nonfinite_jacobian_count,selected_point_world_x_m,"
    "selected_point_world_y_m,selected_point_world_z_m,max_abs_point_jacobian";

// Opt-in switches use a strict 0/1 grammar so an unexpected environment value
// cannot silently enable or disable an evidence-only observer.
inline bool enabledFromEnvironment(const char* value) {
    if (value == nullptr || value[0] == '\0' ||
        (value[0] == '0' && value[1] == '\0'))
        return false;
    if (value[0] == '1' && value[1] == '\0') return true;
    throw std::invalid_argument("motion observer setting must be 0 or 1");
}

struct BodyMotionPartitionRow {
    std::uint32_t bodyIndex = 0u;
    double massKg = 0.0;
    std::array<double, 3u> centerOfMassWorldMeters{};
    std::array<double, 3u> centerOfMassRelativeToRootMeters{};
    std::array<double, 3u> massWeightedRelativeComKgMeters{};
    std::array<double, 4u> orientationBodyToWorldXyzw{};
    std::array<double, 3u> linearVelocityWorldMetersPerSecond{};
    std::array<double, 3u> angularVelocityWorldRadiansPerSecond{};
    std::array<double, 3u> linearMomentumKgMetersPerSecond{};
};

// Produces a read-only, body-resolved contribution to the current total COM.
// Root-relative coordinates remove root translation only; root rotation and
// articulated motion remain in the reported relative coordinates.
inline std::vector<BodyMotionPartitionRow> makeBodyMotionPartition(
    const std::span<const ArticulatedBodyKinematics> bodyKinematics,
    const std::span<const double> massesKg,
    const std::uint32_t firstBodyIndex,
    const std::array<double, 3u>& rootTranslationWorldMeters
) {
    if (bodyKinematics.empty() || bodyKinematics.size() != massesKg.size())
        throw std::invalid_argument("motion observer body/mass extents differ");
    for (const double value : rootTranslationWorldMeters)
        if (!std::isfinite(value))
            throw std::invalid_argument("motion observer root translation is non-finite");

    std::vector<BodyMotionPartitionRow> rows;
    rows.reserve(bodyKinematics.size());
    for (std::size_t body = 0u; body < bodyKinematics.size(); ++body) {
        const auto& source = bodyKinematics[body];
        const double mass = massesKg[body];
        if (!std::isfinite(mass) || mass < 0.0 ||
            body > std::numeric_limits<std::uint32_t>::max() - firstBodyIndex)
            throw std::invalid_argument("motion observer body mass or index is invalid");
        BodyMotionPartitionRow row{};
        row.bodyIndex = firstBodyIndex + static_cast<std::uint32_t>(body);
        row.massKg = mass;
        row.orientationBodyToWorldXyzw = source.orientation;
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            const double position = source.centerOfMassPosition[axis];
            const double linearVelocity = source.linearVelocity[axis];
            const double angularVelocity = source.angularVelocity[axis];
            if (!std::isfinite(position) || !std::isfinite(linearVelocity) ||
                !std::isfinite(angularVelocity))
                throw std::invalid_argument("motion observer body kinematics are non-finite");
            row.centerOfMassWorldMeters[axis] = position;
            row.centerOfMassRelativeToRootMeters[axis] =
                position - rootTranslationWorldMeters[axis];
            row.massWeightedRelativeComKgMeters[axis] =
                mass * row.centerOfMassRelativeToRootMeters[axis];
            row.linearVelocityWorldMetersPerSecond[axis] = linearVelocity;
            row.angularVelocityWorldRadiansPerSecond[axis] = angularVelocity;
            row.linearMomentumKgMetersPerSecond[axis] = mass * linearVelocity;
        }
        for (const double value : row.orientationBodyToWorldXyzw)
            if (!std::isfinite(value))
                throw std::invalid_argument("motion observer body orientation is non-finite");
        rows.push_back(row);
    }
    return rows;
}

struct SupportPointVelocity {
    std::array<double, 3u> worldMetersPerSecond{};
    double tangent0MetersPerSecond = 0.0;
    double tangent1MetersPerSecond = 0.0;
    double tangentialSpeedMetersPerSecond = 0.0;
};

// Applies the already-published pre-step point Jacobian to the matching
// pre-step velocity. This helper never evaluates physics or changes either
// input; callers must bind the Jacobian and velocity to the same accepted step.
inline SupportPointVelocity makeSupportPointVelocity(
    const std::span<const float> pointJacobians,
    const std::uint32_t pointQueryIndex,
    const std::uint32_t dofCount,
    const std::span<const float> preStepVelocity,
    const std::array<double, 3u>& groundNormal
) {
    if (dofCount == 0u || preStepVelocity.size() != dofCount)
        throw std::invalid_argument("motion observer point velocity extent is invalid");
    const std::size_t rowElements = 3u * static_cast<std::size_t>(dofCount);
    const std::size_t queryEnd = static_cast<std::size_t>(pointQueryIndex) + 1u;
    if (queryEnd > std::numeric_limits<std::size_t>::max() / rowElements ||
        queryEnd * rowElements > pointJacobians.size())
        throw std::invalid_argument("motion observer point Jacobian extent is invalid");

    std::array<double, 3u> normal = groundNormal;
    double normalLengthSquared = 0.0;
    for (const double value : normal) {
        if (!std::isfinite(value))
            throw std::invalid_argument("motion observer ground normal is non-finite");
        normalLengthSquared += value * value;
    }
    if (!(normalLengthSquared > 0.0) || !std::isfinite(normalLengthSquared))
        throw std::invalid_argument("motion observer ground normal has zero length");
    const double inverseNormalLength = 1.0 / std::sqrt(normalLengthSquared);
    for (double& value : normal) value *= inverseNormalLength;

    SupportPointVelocity result{};
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
        double velocity = 0.0;
        for (std::size_t dof = 0u; dof < dofCount; ++dof) {
            const float jacobian = pointJacobians[
                (static_cast<std::size_t>(pointQueryIndex) * 3u + axis) * dofCount + dof];
            const float stageVelocity = preStepVelocity[dof];
            if (!std::isfinite(jacobian) || !std::isfinite(stageVelocity))
                throw std::invalid_argument("motion observer point velocity input is non-finite");
            velocity += static_cast<double>(jacobian) * stageVelocity;
        }
        result.worldMetersPerSecond[axis] = velocity;
    }
    const std::array<double, 3u> reference = std::abs(normal[0]) < 0.8
        ? std::array<double, 3u>{1.0, 0.0, 0.0}
        : std::array<double, 3u>{0.0, 1.0, 0.0};
    const double projection = normal[0] * reference[0] +
        normal[1] * reference[1] + normal[2] * reference[2];
    std::array<double, 3u> tangent0{};
    for (std::size_t axis = 0u; axis < 3u; ++axis)
        tangent0[axis] = reference[axis] - projection * normal[axis];
    const double tangentLength = std::sqrt(tangent0[0] * tangent0[0] +
        tangent0[1] * tangent0[1] + tangent0[2] * tangent0[2]);
    if (!(tangentLength > 0.0) || !std::isfinite(tangentLength))
        throw std::invalid_argument("motion observer tangent basis is invalid");
    for (double& value : tangent0) value /= tangentLength;
    const std::array<double, 3u> tangent1{
        normal[1] * tangent0[2] - normal[2] * tangent0[1],
        normal[2] * tangent0[0] - normal[0] * tangent0[2],
        normal[0] * tangent0[1] - normal[1] * tangent0[0]};
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
        result.tangent0MetersPerSecond +=
            result.worldMetersPerSecond[axis] * tangent0[axis];
        result.tangent1MetersPerSecond +=
            result.worldMetersPerSecond[axis] * tangent1[axis];
    }
    result.tangentialSpeedMetersPerSecond = std::hypot(
        result.tangent0MetersPerSecond, result.tangent1MetersPerSecond);
    return result;
}

}  // namespace metalrobo::human::observer
