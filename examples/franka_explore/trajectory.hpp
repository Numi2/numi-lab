#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <string_view>

namespace metalrobo::franka_explore {

inline constexpr std::size_t kArmDofCount = 7u;
inline constexpr double kTransitionSeconds = 3.0;
inline constexpr double kDwellSeconds = 0.5;

// Franka Robotics' own joint-impedance example supplies both this sequence
// and the minimum-jerk timing contract. Source pinned for provenance:
// libfranka 85912fe02258d8cb811d3eff1f11e52ce89e3217,
// pylibfranka/examples/joint_impedance_example.py.
struct Waypoint {
    std::string_view label;
    std::array<double, kArmDofCount> q;
};

inline constexpr std::array<Waypoint, 5u> kOfficialWaypoints{{
    {
        "home",
        {0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0},
    },
    {
        "forward",
        {0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0},
    },
    {
        "right",
        {0.5, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0},
    },
    {
        "left",
        {-0.5, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0},
    },
    {
        "home_return",
        {0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0},
    },
}};

inline double minimumJerk(const double progress) {
    const double t = std::clamp(progress, 0.0, 1.0);
    return 10.0 * t * t * t -
        15.0 * t * t * t * t +
        6.0 * t * t * t * t * t;
}

inline std::array<double, kArmDofCount> interpolate(
    const std::array<double, kArmDofCount>& start,
    const std::array<double, kArmDofCount>& target,
    const double progress
) {
    const double blend = minimumJerk(progress);
    std::array<double, kArmDofCount> result{};
    for (std::size_t joint = 0u; joint < result.size(); ++joint) {
        result[joint] = start[joint] +
            blend * (target[joint] - start[joint]);
    }
    return result;
}

} // namespace metalrobo::franka_explore
