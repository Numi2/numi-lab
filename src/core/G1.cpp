#include "metalrobo/G1.hpp"

#include "metalrobo/GeometryCooker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

// Unitree model-data attribution
// --------------------------------
// The factual topology, transforms, masses, COMs, inertia tensors, limits,
// official primitive colliders, and IMU frames below are adapted from:
//
//   unitreerobotics/unitree_ros
//   robots/g1_description/g1_29dof_rev_1_0.{urdf,xml}
//   commit aa0f5c68b5aba347bad409e71b6430407da758d7
//   Copyright (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD.
//   SPDX-License-Identifier: BSD-3-Clause
//   https://github.com/unitreerobotics/unitree_ros/blob/
//     aa0f5c68b5aba347bad409e71b6430407da758d7/LICENSE
//
// The named PD/armature/reset preset is adapted from:
//
//   unitreerobotics/unitree_rl_mjlab
//   deploy/robots/g1/config/policy/velocity/v0/params/deploy.yaml
//   commit 1425b15f73bd4095f0df53709d7c389c3eb9e790
//   SPDX-License-Identifier: BSD-3-Clause
//
// Fixed-link inertials are folded from the pinned upstream values with the
// parallel-axis theorem. MetalRobo's storage layout and inverse calculation
// are original implementation work.

namespace metalrobo {
namespace {

struct BodySource {
    const char* name;
    std::uint32_t parentBody;
    std::uint32_t inboundJoint;
    double mass;
    std::array<double, 3> centerOfMass;
    // Ixx, Ixy, Ixz, Iyy, Iyz, Izz about COM in the body frame.
    std::array<double, 6> inertia;
};

struct JointSource {
    const char* name;
    std::uint32_t parentBody;
    std::uint32_t childBody;
    std::array<double, 3> origin;
    std::array<double, 3> rpy;
    std::array<double, 3> axis;
    double lower;
    double upper;
    double effort;
    double velocity;
};

// Exact local envelope of the pinned left/right ankle-roll visual meshes,
// reduced to one non-overlapping sole box. The upper face stays at the former
// MuJoCo-Lab sole height (-0.015 m); the remaining faces cover the physical
// shell instead of allowing the visible heel and toe to pass through support.
constexpr double kSoleMinimumX = -0.06584137;
constexpr double kSoleMaximumX = 0.14236535;
constexpr double kSoleHalfY = 0.03784090;
constexpr double kSoleMinimumZ = -0.03540915;
constexpr double kSoleMaximumZ = -0.015;
constexpr double kSoleCenterX =
    0.5 * (kSoleMinimumX + kSoleMaximumX);
constexpr double kSoleCenterZ =
    0.5 * (kSoleMinimumZ + kSoleMaximumZ);
constexpr double kSoleHalfX =
    0.5 * (kSoleMaximumX - kSoleMinimumX);
constexpr double kSoleHalfZ =
    0.5 * (kSoleMaximumZ - kSoleMinimumZ);

constexpr std::array<BodySource, kUnitreeG1BodyCount> kBodies{{
    {
        "pelvis", MR_INVALID_INDEX, MR_INVALID_INDEX, 3.814,
        {0.0, 0.0, -0.076030060304},
        {0.010554882086, 0.0, 2.1e-06, 0.0093147820861, 0.0, 0.0079185},
    },
    {
        "left_hip_pitch_link", 0u, 0u, 1.35,
        {0.002741, 0.047791, -0.02606},
        {0.001811, 3.68e-05, -3.44e-05, 0.0014193, 0.000171, 0.0012812},
    },
    {
        "left_hip_roll_link", 1u, 1u, 1.52,
        {0.029812, -0.001045, -0.087934},
        {0.0023773, -3.8e-06, -0.0003908, 0.0024123, 1.84e-05, 0.0016595},
    },
    {
        "left_hip_yaw_link", 2u, 2u, 1.702,
        {-0.057709, -0.010981, -0.15078},
        {0.0057774, -0.0005411, -0.0023948, 0.0076124, -0.0007072, 0.003149},
    },
    {
        "left_knee_link", 3u, 3u, 1.932,
        {0.005457, 0.003964, -0.12074},
        {0.011329, 4.82e-05, -4.49e-05, 0.011277, -0.0007146, 0.0015168},
    },
    {
        "left_ankle_pitch_link", 4u, 4u, 0.074,
        {-0.007269, 0.0, 0.011137},
        {8.4e-06, 0.0, -2.9e-06, 1.89e-05, 0.0, 1.26e-05},
    },
    {
        "left_ankle_roll_link", 5u, 5u, 0.608,
        {0.026505, 0.0, -0.016425},
        {0.0002231, 2.0e-07, 8.91e-05, 0.0016161, -1.0e-07, 0.0016667},
    },
    {
        "right_hip_pitch_link", 0u, 6u, 1.35,
        {0.002741, -0.047791, -0.02606},
        {0.001811, -3.68e-05, -3.44e-05, 0.0014193, -0.000171, 0.0012812},
    },
    {
        "right_hip_roll_link", 7u, 7u, 1.52,
        {0.029812, 0.001045, -0.087934},
        {0.0023773, 3.8e-06, -0.0003908, 0.0024123, -1.84e-05, 0.0016595},
    },
    {
        "right_hip_yaw_link", 8u, 8u, 1.702,
        {-0.057709, 0.010981, -0.15078},
        {0.0057774, 0.0005411, -0.0023948, 0.0076124, 0.0007072, 0.003149},
    },
    {
        "right_knee_link", 9u, 9u, 1.932,
        {0.005457, -0.003964, -0.12074},
        {0.011329, -4.82e-05, 4.49e-05, 0.011277, 0.0007146, 0.0015168},
    },
    {
        "right_ankle_pitch_link", 10u, 10u, 0.074,
        {-0.007269, 0.0, 0.011137},
        {8.4e-06, 0.0, -2.9e-06, 1.89e-05, 0.0, 1.26e-05},
    },
    {
        "right_ankle_roll_link", 11u, 11u, 0.608,
        {0.026505, 0.0, -0.016425},
        {0.0002231, -2.0e-07, 8.91e-05, 0.0016161, 1.0e-07, 0.0016667},
    },
    {
        "waist_yaw_link", 0u, 12u, 0.214,
        {0.003494, 0.000233, 0.018034},
        {0.00010673, 2.703e-06, -7.631e-06, 0.00010422, -2.01e-07, 0.0001625},
    },
    {
        "waist_roll_link", 13u, 13u, 0.086,
        {0.0, 2.3e-05, 0.0},
        {7.079e-06, 0.0, 0.0, 6.339e-06, 0.0, 8.245e-06},
    },
    {
        "torso_link", 14u, 14u, 7.817,
        {0.0020313344633, 0.00033972674939, 0.1845971452},
        {
            0.12164651964, 3.1110210298e-05, -0.0037428195854,
            0.10977258486, -1.5429925458e-05, 0.027521919415,
        },
    },
    {
        "left_shoulder_pitch_link", 15u, 15u, 0.718,
        {0.0, 0.035892, -0.011628},
        {0.0004291, -9.2e-06, 6.4e-06, 0.000453, 2.26e-05, 0.000423},
    },
    {
        "left_shoulder_roll_link", 16u, 16u, 0.643,
        {-0.000227, 0.00727, -0.063243},
        {0.0006177, -1.0e-06, 8.7e-06, 0.0006912, -5.3e-06, 0.0003894},
    },
    {
        "left_shoulder_yaw_link", 17u, 17u, 0.734,
        {0.010773, -0.002949, -0.072009},
        {0.0009988, 7.9e-06, 0.0001412, 0.0010605, -2.86e-05, 0.0004354},
    },
    {
        "left_elbow_link", 18u, 18u, 0.6,
        {0.064956, 0.004454, -0.010062},
        {0.0002891, 6.53e-05, 1.72e-05, 0.0004152, -5.6e-06, 0.0004197},
    },
    {
        "left_wrist_roll_link", 19u, 19u, 0.08544498,
        {0.01713944778, 0.00053759094, 0.00000048864},
        {
            0.00004821544023, -0.00000424511021, 0.00000000510599,
            0.00003722899093, -0.00000000123525, 0.00005482106541,
        },
    },
    {
        "left_wrist_pitch_link", 20u, 20u, 0.48404956,
        {0.02299989837, -0.00111685314, -0.00111658096},
        {
            0.00016579646273, -0.00001231206746, 0.00001231699194,
            0.00042954057410, 0.00000081417712, 0.00042953697654,
        },
    },
    {
        "left_wrist_yaw_link", 21u, 21u, 0.25457647,
        {0.070824430201, 0.0001917452912, 0.0016174161392},
        {
            0.00015044517917, 0.00003760275409, -0.0000029549415859,
            0.00064311326853, 0.0000037754842413, 0.0005601139475,
        },
    },
    {
        "right_shoulder_pitch_link", 15u, 22u, 0.718,
        {0.0, -0.035892, -0.011628},
        {0.0004291, 9.2e-06, 6.4e-06, 0.000453, -2.26e-05, 0.000423},
    },
    {
        "right_shoulder_roll_link", 23u, 23u, 0.643,
        {-0.000227, -0.00727, -0.063243},
        {0.0006177, 1.0e-06, 8.7e-06, 0.0006912, 5.3e-06, 0.0003894},
    },
    {
        "right_shoulder_yaw_link", 24u, 24u, 0.734,
        {0.010773, 0.002949, -0.072009},
        {0.0009988, -7.9e-06, 0.0001412, 0.0010605, 2.86e-05, 0.0004354},
    },
    {
        "right_elbow_link", 25u, 25u, 0.6,
        {0.064956, -0.004454, -0.010062},
        {0.0002891, -6.53e-05, 1.72e-05, 0.0004152, 5.6e-06, 0.0004197},
    },
    {
        "right_wrist_roll_link", 26u, 26u, 0.08544498,
        {0.01713944778, -0.00053759094, 0.00000048864},
        {
            0.00004821544023, 0.00000424511021, 0.00000000510599,
            0.00003722899093, 0.00000000123525, 0.00005482106541,
        },
    },
    {
        "right_wrist_pitch_link", 27u, 27u, 0.48404956,
        {0.02299989837, 0.00111685314, -0.00111658096},
        {
            0.00016579646273, 0.00001231206746, 0.00001231699194,
            0.00042954057410, -0.00000081417712, 0.00042953697654,
        },
    },
    {
        "right_wrist_yaw_link", 28u, 28u, 0.25457647,
        {0.070824430201, -0.0001917452912, 0.0016174161392},
        {
            0.00015044517917, -0.00003760275409, -0.0000029549415859,
            0.00064311326853, -0.0000037754842413, 0.0005601139475,
        },
    },
}};

constexpr std::array<JointSource, kUnitreeG1JointCount> kJoints{{
    {
        "left_hip_pitch_joint", 0u, 1u,
        {0.0, 0.064452, -0.1027}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
        -2.5307, 2.8798, 88.0, 32.0,
    },
    {
        "left_hip_roll_joint", 1u, 2u,
        {0.0, 0.052, -0.030465}, {0.0, -0.1749, 0.0}, {1.0, 0.0, 0.0},
        -0.5236, 2.9671, 139.0, 20.0,
    },
    {
        "left_hip_yaw_joint", 2u, 3u,
        {0.025001, 0.0, -0.12412}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
        -2.7576, 2.7576, 88.0, 32.0,
    },
    {
        "left_knee_joint", 3u, 4u,
        {-0.078273, 0.0021489, -0.17734}, {0.0, 0.1749, 0.0}, {0.0, 1.0, 0.0},
        -0.087267, 2.8798, 139.0, 20.0,
    },
    {
        "left_ankle_pitch_joint", 4u, 5u,
        {0.0, -9.4445e-05, -0.30001}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
        -0.87267, 0.5236, 35.0, 30.0,
    },
    {
        "left_ankle_roll_joint", 5u, 6u,
        {0.0, 0.0, -0.017558}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
        -0.2618, 0.2618, 35.0, 30.0,
    },
    {
        "right_hip_pitch_joint", 0u, 7u,
        {0.0, -0.064452, -0.1027}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
        -2.5307, 2.8798, 88.0, 32.0,
    },
    {
        "right_hip_roll_joint", 7u, 8u,
        {0.0, -0.052, -0.030465}, {0.0, -0.1749, 0.0}, {1.0, 0.0, 0.0},
        -2.9671, 0.5236, 139.0, 20.0,
    },
    {
        "right_hip_yaw_joint", 8u, 9u,
        {0.025001, 0.0, -0.12412}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
        -2.7576, 2.7576, 88.0, 32.0,
    },
    {
        "right_knee_joint", 9u, 10u,
        {-0.078273, -0.0021489, -0.17734}, {0.0, 0.1749, 0.0}, {0.0, 1.0, 0.0},
        -0.087267, 2.8798, 139.0, 20.0,
    },
    {
        "right_ankle_pitch_joint", 10u, 11u,
        {0.0, 9.4445e-05, -0.30001}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
        -0.87267, 0.5236, 35.0, 30.0,
    },
    {
        "right_ankle_roll_joint", 11u, 12u,
        {0.0, 0.0, -0.017558}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
        -0.2618, 0.2618, 35.0, 30.0,
    },
    {
        "waist_yaw_joint", 0u, 13u,
        {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
        -2.618, 2.618, 88.0, 32.0,
    },
    {
        "waist_roll_joint", 13u, 14u,
        {-0.0039635, 0.0, 0.044}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
        -0.52, 0.52, 35.0, 30.0,
    },
    {
        "waist_pitch_joint", 14u, 15u,
        {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
        -0.52, 0.52, 35.0, 30.0,
    },
    {
        "left_shoulder_pitch_joint", 15u, 16u,
        {0.0039563, 0.10022, 0.24778},
        {0.27931, 5.4949e-05, -0.00019159}, {0.0, 1.0, 0.0},
        -3.0892, 2.6704, 25.0, 37.0,
    },
    {
        "left_shoulder_roll_joint", 16u, 17u,
        {0.0, 0.038, -0.013831}, {-0.27925, 0.0, 0.0}, {1.0, 0.0, 0.0},
        -1.5882, 2.2515, 25.0, 37.0,
    },
    {
        "left_shoulder_yaw_joint", 17u, 18u,
        {0.0, 0.00624, -0.1032}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
        -2.618, 2.618, 25.0, 37.0,
    },
    {
        "left_elbow_joint", 18u, 19u,
        {0.015783, 0.0, -0.080518}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
        -1.0472, 2.0944, 25.0, 37.0,
    },
    {
        "left_wrist_roll_joint", 19u, 20u,
        {0.1, 0.00188791, -0.01}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
        -1.972222054, 1.972222054, 25.0, 37.0,
    },
    {
        "left_wrist_pitch_joint", 20u, 21u,
        {0.038, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
        -1.614429558, 1.614429558, 5.0, 22.0,
    },
    {
        "left_wrist_yaw_joint", 21u, 22u,
        {0.046, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
        -1.614429558, 1.614429558, 5.0, 22.0,
    },
    {
        "right_shoulder_pitch_joint", 15u, 23u,
        {0.0039563, -0.10021, 0.24778},
        {-0.27931, 5.4949e-05, 0.00019159}, {0.0, 1.0, 0.0},
        -3.0892, 2.6704, 25.0, 37.0,
    },
    {
        "right_shoulder_roll_joint", 23u, 24u,
        {0.0, -0.038, -0.013831}, {0.27925, 0.0, 0.0}, {1.0, 0.0, 0.0},
        -2.2515, 1.5882, 25.0, 37.0,
    },
    {
        "right_shoulder_yaw_joint", 24u, 25u,
        {0.0, -0.00624, -0.1032}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
        -2.618, 2.618, 25.0, 37.0,
    },
    {
        "right_elbow_joint", 25u, 26u,
        {0.015783, 0.0, -0.080518}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
        -1.0472, 2.0944, 25.0, 37.0,
    },
    {
        "right_wrist_roll_joint", 26u, 27u,
        {0.1, -0.00188791, -0.01}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
        -1.972222054, 1.972222054, 25.0, 37.0,
    },
    {
        "right_wrist_pitch_joint", 27u, 28u,
        {0.038, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
        -1.614429558, 1.614429558, 5.0, 22.0,
    },
    {
        "right_wrist_yaw_joint", 28u, 29u,
        {0.046, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
        -1.614429558, 1.614429558, 5.0, 22.0,
    },
}};

struct G1CollisionHullRecord {
    std::string_view sourceMesh;
    std::uint32_t sourceElement;
    std::uint32_t pieceIndex;
    std::uint32_t pieceCount;
    std::uint32_t bodyIndex;
    std::array<float, 3> center;
    std::array<float, 3> lower;
    std::array<float, 3> extent;
    std::uint32_t vertexOffset;
    std::uint32_t vertexCount;
    std::uint32_t indexOffset;
    std::uint32_t indexCount;
    float volumeRatio;
    std::string_view sourceSha256;
};

#include "G1CollisionHulls.inc"

constexpr std::array<float, kUnitreeG1JointCount> kRLLabStiffness{{
    40.2f, 99.1f, 40.2f, 99.1f, 28.5f, 28.5f,
    40.2f, 99.1f, 40.2f, 99.1f, 28.5f, 28.5f,
    40.2f, 28.5f, 28.5f,
    14.3f, 14.3f, 14.3f, 14.3f, 14.3f, 16.8f, 16.8f,
    14.3f, 14.3f, 14.3f, 14.3f, 14.3f, 16.8f, 16.8f,
}};

constexpr std::array<float, kUnitreeG1JointCount> kRLLabDamping{{
    2.6f, 6.3f, 2.6f, 6.3f, 1.8f, 1.8f,
    2.6f, 6.3f, 2.6f, 6.3f, 1.8f, 1.8f,
    2.6f, 1.8f, 1.8f,
    0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 1.1f, 1.1f,
    0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 1.1f, 1.1f,
}};

constexpr std::array<float, kUnitreeG1JointCount> kRLLabResetQ{{
    -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,
    -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,
    0.0f, 0.0f, 0.0f,
    0.35f, 0.18f, 0.0f, 0.87f, 0.0f, 0.0f, 0.0f,
    0.35f, -0.18f, 0.0f, 0.87f, 0.0f, 0.0f, 0.0f,
}};

constexpr std::array<float, kUnitreeG1JointCount> kMjLabEffortLimit{{
    88.0f, 139.0f, 88.0f, 139.0f, 50.0f, 50.0f,
    88.0f, 139.0f, 88.0f, 139.0f, 50.0f, 50.0f,
    88.0f, 50.0f, 50.0f,
    25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 5.0f, 5.0f,
    25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 5.0f, 5.0f,
}};

mr_float4 f4(
    const double x,
    const double y,
    const double z,
    const double w = 0.0
) {
    return {
        static_cast<float>(x),
        static_cast<float>(y),
        static_cast<float>(z),
        static_cast<float>(w),
    };
}

mr_float4 quaternionFromRpy(const std::array<double, 3> rpy) {
    const double cr = std::cos(0.5 * rpy[0]);
    const double sr = std::sin(0.5 * rpy[0]);
    const double cp = std::cos(0.5 * rpy[1]);
    const double sp = std::sin(0.5 * rpy[1]);
    const double cy = std::cos(0.5 * rpy[2]);
    const double sy = std::sin(0.5 * rpy[2]);
    return f4(
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy
    );
}

std::array<mr_float4, 3> inverseInertia(
    const std::array<double, 6>& inertia
) {
    const double a = inertia[0];
    const double b = inertia[1];
    const double c = inertia[2];
    const double d = inertia[3];
    const double e = inertia[4];
    const double f = inertia[5];
    const double determinant =
        a * (d * f - e * e) -
        b * (b * f - c * e) +
        c * (b * e - c * d);
    if (!(determinant > 0.0) || !std::isfinite(determinant)) {
        throw std::logic_error("pinned G1 inertia is not positive definite");
    }
    const double inverse00 = (d * f - e * e) / determinant;
    const double inverse01 = (c * e - b * f) / determinant;
    const double inverse02 = (b * e - c * d) / determinant;
    const double inverse11 = (a * f - c * c) / determinant;
    const double inverse12 = (b * c - a * e) / determinant;
    const double inverse22 = (a * d - b * b) / determinant;
    return {
        f4(inverse00, inverse01, inverse02),
        f4(inverse01, inverse11, inverse12),
        f4(inverse02, inverse12, inverse22),
    };
}

MRBodyPropertiesGPU makeBody(const BodySource& source) {
    MRBodyPropertiesGPU body{};
    body.articulationIndex = 0u;
    body.parentBody = source.parentBody;
    body.inboundJoint = source.inboundJoint;
    body.motionType = MR_MOTION_DYNAMIC;
    body.massAndInverseMass =
        f4(source.mass, 1.0 / source.mass, 0.0, 0.0);
    body.centerOfMass = f4(
        source.centerOfMass[0],
        source.centerOfMass[1],
        source.centerOfMass[2]
    );
    body.inertiaRow0 = f4(
        source.inertia[0],
        source.inertia[1],
        source.inertia[2]
    );
    body.inertiaRow1 = f4(
        source.inertia[1],
        source.inertia[3],
        source.inertia[4]
    );
    body.inertiaRow2 = f4(
        source.inertia[2],
        source.inertia[4],
        source.inertia[5]
    );
    const std::array<mr_float4, 3> inverse =
        inverseInertia(source.inertia);
    body.inverseInertiaRow0 = inverse[0];
    body.inverseInertiaRow1 = inverse[1];
    body.inverseInertiaRow2 = inverse[2];
    body.dampingAndSpeedLimits = f4(0.0, 0.0, 1.0e6, 1.0e6);
    return body;
}

MRJointDescriptorGPU makeJoint(
    const JointSource& source,
    const std::uint32_t jointIndex
) {
    MRJointDescriptorGPU joint{};
    joint.parentBody = source.parentBody;
    joint.childBody = source.childBody;
    joint.jointType = MR_JOINT_REVOLUTE;
    joint.flags = 0u;
    joint.qOffset = 7u + jointIndex;
    joint.nq = 1u;
    joint.vOffset = 6u + jointIndex;
    joint.nv = 1u;
    joint.axis0 = f4(source.axis[0], source.axis[1], source.axis[2]);
    // MRBodyStateGPU stores each body's COM pose. URDF joint origins are
    // expressed in link frames, so both anchors must be shifted into the
    // corresponding COM-centred coordinates.
    const BodySource& parent = kBodies[source.parentBody];
    const BodySource& child = kBodies[source.childBody];
    joint.parentAnchor = f4(
        source.origin[0] - parent.centerOfMass[0],
        source.origin[1] - parent.centerOfMass[1],
        source.origin[2] - parent.centerOfMass[2]
    );
    joint.childAnchor = f4(
        -child.centerOfMass[0],
        -child.centerOfMass[1],
        -child.centerOfMass[2]
    );
    joint.parentRotation = quaternionFromRpy(source.rpy);
    joint.childRotation = f4(0.0, 0.0, 0.0, 1.0);
    return joint;
}

MRShapeGPU makePrimitive(
    const std::uint32_t bodyIndex,
    const std::uint32_t shapeType,
    const mr_float4 localPosition,
    const mr_float4 localRotation,
    const mr_float4 dimensions,
    const double boundingRadius,
    const bool simulationEnabled = true
) {
    MRShapeGPU shape{};
    shape.bodyIndex = bodyIndex;
    shape.shapeType = shapeType;
    shape.materialIndex = 0u;
    shape.flags = simulationEnabled
        ? 0u
        : MR_SHAPE_FLAG_SIMULATION_DISABLED;
    shape.collisionGroup = 1u;
    shape.collisionMask = ~0u;
    shape.slotGeneration = 1u;
    // Collision points share MRBodyStateGPU's COM-centred pose convention.
    // Official primitive records are link-local and need the same shift as
    // joint anchors.
    const BodySource& body = kBodies[bodyIndex];
    shape.localPosition = f4(
        localPosition.x - body.centerOfMass[0],
        localPosition.y - body.centerOfMass[1],
        localPosition.z - body.centerOfMass[2],
        1.0
    );
    shape.localRotation = localRotation;
    shape.dimensions = dimensions;
    // Unitree does not specify contact/rest offsets for this URDF.
    shape.contactRestAndBoundingRadius =
        f4(0.0, 0.0, boundingRadius, 0.0);
    return shape;
}

std::uint32_t appendAuthoredCollisionHull(
    EngineModel& model,
    const G1CollisionHullRecord& source
) {
    const std::uint64_t vertexEnd =
        static_cast<std::uint64_t>(source.vertexOffset) +
        source.vertexCount;
    const std::uint64_t indexEnd =
        static_cast<std::uint64_t>(source.indexOffset) +
        source.indexCount;
    if (source.bodyIndex >= kBodies.size() ||
        source.vertexCount < 4u ||
        source.vertexCount > 255u ||
        source.indexCount < 12u ||
        source.indexCount % 3u != 0u ||
        vertexEnd * 3u > kG1CollisionHullVertices.size() ||
        indexEnd > kG1CollisionHullIndices.size()) {
        throw std::logic_error(
            "pinned G1 collision hull record is invalid: " +
            std::string{source.sourceMesh}
        );
    }

    std::vector<mr_float4> vertices;
    vertices.reserve(source.vertexCount);
    double boundingRadiusSquared = 0.0;
    for (std::uint32_t index = 0u;
         index < source.vertexCount;
         ++index) {
        const std::size_t packedIndex =
            3u * (source.vertexOffset + index);
        const double x =
            source.lower[0] +
            source.extent[0] *
                static_cast<double>(
                    kG1CollisionHullVertices[packedIndex]
                ) /
                65535.0;
        const double y =
            source.lower[1] +
            source.extent[1] *
                static_cast<double>(
                    kG1CollisionHullVertices[packedIndex + 1u]
                ) /
                65535.0;
        const double z =
            source.lower[2] +
            source.extent[2] *
                static_cast<double>(
                    kG1CollisionHullVertices[packedIndex + 2u]
                ) /
                65535.0;
        vertices.push_back(f4(x, y, z, 1.0));
        boundingRadiusSquared = std::max(
            boundingRadiusSquared,
            x * x + y * y + z * z
        );
    }

    std::vector<std::uint32_t> triangleIndices;
    triangleIndices.reserve(source.indexCount);
    for (std::uint32_t index = 0u;
         index < source.indexCount;
         ++index) {
        const std::uint32_t vertex =
            kG1CollisionHullIndices[source.indexOffset + index];
        if (vertex >= source.vertexCount) {
            throw std::logic_error(
                "pinned G1 collision hull index is invalid: " +
                std::string{source.sourceMesh}
            );
        }
        triangleIndices.push_back(vertex);
    }

    const GeometryCookResult cooked = cookConvexGeometry(
        model,
        vertices,
        triangleIndices
    );
    if (!cooked.succeeded()) {
        throw std::logic_error(
            "pinned G1 collision hull cook failed for " +
            std::string{source.sourceMesh} + ": " + cooked.message
        );
    }

    MRShapeGPU shape = makePrimitive(
        source.bodyIndex,
        MR_SHAPE_CONVEX,
        f4(source.center[0], source.center[1], source.center[2]),
        f4(0.0, 0.0, 0.0, 1.0),
        f4(1.0, 1.0, 1.0),
        std::sqrt(boundingRadiusSquared)
    );
    shape.geometryOffset = cooked.geometryIndex;
    shape.geometryCount = 1u;
    const std::uint32_t shapeIndex =
        static_cast<std::uint32_t>(model.shapes.size());
    model.shapes.push_back(shape);
    return shapeIndex;
}

} // namespace

const G1ModelMetadata& unitreeG1Metadata() noexcept {
    static const G1ModelMetadata metadata = [] {
        G1ModelMetadata value{};
        value.modelName = "unitree_g1_29dof_rev_1_0";
        value.sourceRepository =
            "https://github.com/unitreerobotics/unitree_ros";
        value.sourceCommit =
            "aa0f5c68b5aba347bad409e71b6430407da758d7";
        value.sourceModelPath =
            "robots/g1_description/g1_29dof_rev_1_0.urdf";
        value.sourceLicense = "BSD-3-Clause";
        value.simulatorRepository =
            "https://github.com/unitreerobotics/unitree_mujoco";
        value.simulatorCommit =
            "ae6a8403e272733e9996ef59990880330496177f";
        value.simulatorModelPath =
            "unitree_robots/g1/scene_29dof.xml";
        value.rlPresetRepository =
            "https://github.com/unitreerobotics/unitree_rl_mjlab";
        value.rlPresetCommit =
            "1425b15f73bd4095f0df53709d7c389c3eb9e790";
        value.rlPresetLicense =
            "BSD-3-Clause per-file (repository LICENCE is Apache-2.0)";
        value.collisionMaterialPreset =
            "Unitree MuJoCo-Lab locomotion material: friction=0.6";
        value.collisionCookMethod =
            "official STL deterministic V-HACD two-piece compound, "
            "normalized farthest-point compact support sets, uint16 "
            "local quantization";
        value.collisionCookHash = kG1CollisionHullCookHash;
        value.modeMachine = 5u;
        value.modePr = 0u;
        value.canonicalMassKg = 33.34114202;

        for (std::size_t index = 0; index < kBodies.size(); ++index) {
            value.bodyNames[index] = kBodies[index].name;
        }
        for (std::size_t index = 0; index < kJoints.size(); ++index) {
            const JointSource& joint = kJoints[index];
            value.jointLimits[index] = {
                joint.name,
                joint.parentBody,
                joint.childBody,
                static_cast<float>(joint.lower),
                static_cast<float>(joint.upper),
                kMjLabEffortLimit[index],
                static_cast<float>(joint.velocity),
            };
            value.rlLabDrives[index] = {
                joint.name,
                kRLLabStiffness[index],
                kRLLabDamping[index],
                kRLLabStiffness[index] /
                    std::pow(
                        20.0f * std::numbers::pi_v<float>,
                        2.0f
                    ),
            };
        }

        value.feet[0] = {
            "left_sole",
            "left_ankle_roll_link",
            6u,
            f4(
                kSoleCenterX - kBodies[6].centerOfMass[0],
                0.0 - kBodies[6].centerOfMass[1],
                kSoleMinimumZ - kBodies[6].centerOfMass[2]
            ),
            f4(0.0, 0.0, 0.0, 1.0),
            0u,
            f4(-kSoleHalfX, -kSoleHalfY, kSoleHalfX, kSoleHalfY),
        };
        value.feet[1] = {
            "right_sole",
            "right_ankle_roll_link",
            12u,
            f4(
                kSoleCenterX - kBodies[12].centerOfMass[0],
                0.0 - kBodies[12].centerOfMass[1],
                kSoleMinimumZ - kBodies[12].centerOfMass[2]
            ),
            f4(0.0, 0.0, 0.0, 1.0),
            1u,
            f4(-kSoleHalfX, -kSoleHalfY, kSoleHalfX, kSoleHalfY),
        };

        value.imus[0] = {
            "imu_in_pelvis",
            "pelvis",
            "LowState.imu_state",
            0u,
            f4(
                0.04525 - kBodies[0].centerOfMass[0],
                0.0 - kBodies[0].centerOfMass[1],
                -0.08339 - kBodies[0].centerOfMass[2]
            ),
            f4(0.0, 0.0, 0.0, 1.0),
            5.0e-4f,
            34.9f,
            1.0e-2f,
            157.0f,
        };
        value.imus[1] = {
            "imu_in_torso",
            "torso_link",
            "rt/secondary_imu",
            15u,
            f4(
                -0.03959 - kBodies[15].centerOfMass[0],
                -0.00224 - kBodies[15].centerOfMass[1],
                0.14792 - kBodies[15].centerOfMass[2]
            ),
            f4(0.0, 0.0, 0.0, 1.0),
            5.0e-4f,
            34.9f,
            1.0e-2f,
            157.0f,
        };
        return value;
    }();
    return metadata;
}

EngineModel makeUnitreeG1EngineModel() {
    EngineModel model;
    const G1ModelMetadata& metadata = unitreeG1Metadata();
    model.name = std::string{metadata.modelName};
    model.bodyNames.reserve(metadata.bodyNames.size());
    for (const std::string_view name : metadata.bodyNames) {
        model.bodyNames.emplace_back(name);
    }
    model.jointNames.reserve(metadata.jointLimits.size());
    for (const G1JointLimit& joint : metadata.jointLimits) {
        model.jointNames.emplace_back(joint.name);
    }
    model.dofNames = {
        "root_linear_x",
        "root_linear_y",
        "root_linear_z",
        "root_angular_x",
        "root_angular_y",
        "root_angular_z",
    };
    model.dofNames.insert(
        model.dofNames.end(),
        model.jointNames.begin(),
        model.jointNames.end()
    );
    model.world.abiVersion = MR_ENGINE_ABI_VERSION;
    model.world.bodyCount = static_cast<mr_u32>(kUnitreeG1BodyCount);
    model.world.articulationCount = 1u;
    model.world.jointCount = static_cast<mr_u32>(kUnitreeG1JointCount);
    model.world.shapeCount = 0u;
    model.world.materialCount = 1u;
    model.world.nq = 36u;
    model.world.nv = 35u;
    model.world.pairCapacity = 2048u;
    model.world.contactCapacity = 1024u;
    model.world.constraintCapacity = 2048u;
    model.world.islandCapacity = 32u;
    model.world.solverType = MR_SOLVER_TEMPORAL_CONE;
    model.world.frictionConeType = MR_FRICTION_CONE_ELLIPTIC;
    model.world.gravityAndTimestep = f4(0.0, 0.0, -9.81, 1.0 / 240.0);
    model.world.solverScales = f4(1.0e-7, 1.0e-9, 2.0, 1.0e-4);

    MRArticulationGPU articulation{};
    articulation.rootBody = 0u;
    articulation.rootType = MR_ROOT_FLOATING;
    articulation.firstBody = 0u;
    articulation.bodyCount = static_cast<mr_u32>(kUnitreeG1BodyCount);
    articulation.firstJoint = 0u;
    articulation.jointCount = static_cast<mr_u32>(kUnitreeG1JointCount);
    articulation.qOffset = 0u;
    articulation.nq = 36u;
    articulation.vOffset = 0u;
    articulation.nv = 35u;
    articulation.solverGroup = 0u;
    model.articulations.push_back(articulation);

    model.bodies.reserve(kBodies.size());
    for (const BodySource& body : kBodies) {
        model.bodies.push_back(makeBody(body));
    }

    model.joints.reserve(kJoints.size());
    for (std::size_t index = 0; index < kJoints.size(); ++index) {
        model.joints.push_back(
            makeJoint(kJoints[index], static_cast<std::uint32_t>(index))
        );
    }

    model.dofs.reserve(35u);
    for (mr_u32 localDof = 0u; localDof < 6u; ++localDof) {
        MRDofPropertiesGPU dof{};
        dof.articulationIndex = 0u;
        dof.jointIndex = MR_INVALID_INDEX;
        dof.qIndex = localDof < 3u
            ? localDof
            : MR_INVALID_INDEX;
        dof.vIndex = localDof;
        dof.localDof = localDof;
        dof.flags = MR_DOF_FLAG_ROOT;
        model.dofs.push_back(dof);
    }
    for (std::size_t index = 0u;
         index < kUnitreeG1JointCount;
         ++index) {
        const G1JointLimit& limit = metadata.jointLimits[index];
        const G1RLLabJointDrive& drive =
            metadata.rlLabDrives[index];
        MRDofPropertiesGPU dof{};
        dof.articulationIndex = 0u;
        dof.jointIndex = static_cast<mr_u32>(index);
        dof.qIndex = 7u + static_cast<mr_u32>(index);
        dof.vIndex = 6u + static_cast<mr_u32>(index);
        dof.localDof = 0u;
        dof.flags =
            MR_DOF_FLAG_ACTUATED |
            MR_DOF_FLAG_POSITION_LIMIT |
            MR_DOF_FLAG_VELOCITY_LIMIT |
            MR_DOF_FLAG_EFFORT_LIMIT |
            MR_DOF_FLAG_DRIVE;
        dof.limits = f4(
            limit.lowerPosition,
            limit.upperPosition,
            limit.maximumVelocity,
            limit.maximumEffort
        );
        dof.drive = f4(
            drive.stiffness,
            drive.damping,
            drive.armature,
            0.0f
        );
        model.dofs.push_back(dof);
    }

    MRMaterialGPU material{};
    // Named Unitree MuJoCo-Lab locomotion preset, not hardware identification.
    material.friction = f4(0.6, 0.6, 0.0, 0.0);
    material.response = f4(0.0, 0.5, 0.0, 0.0);
    material.geometry = f4(0.0, 0.0, 0.0, 0.0);
    model.materials.push_back(material);

    // One convex sole per foot covers the pinned visible shell without
    // creating seven mutually overlapping contact manifolds.
    const mr_float4 identity = f4(0.0, 0.0, 0.0, 1.0);
    for (const std::uint32_t bodyIndex : {6u, 12u}) {
        model.shapes.push_back(makePrimitive(
            bodyIndex,
            MR_SHAPE_BOX,
            f4(kSoleCenterX, 0.0, kSoleCenterZ),
            identity,
            f4(kSoleHalfX, kSoleHalfY, kSoleHalfZ, 0.0),
            std::sqrt(
                kSoleHalfX * kSoleHalfX +
                kSoleHalfY * kSoleHalfY +
                kSoleHalfZ * kSoleHalfZ
            )
        ));
    }

    const mr_float4 shoulderPitchRotation =
        quaternionFromRpy({0.0, std::numbers::pi_v<double> / 2.0, 0.0});
    model.shapes.push_back(makePrimitive(
        16u,
        MR_SHAPE_CYLINDER,
        f4(0.0, 0.04, -0.01),
        shoulderPitchRotation,
        f4(0.03, 0.025, 0.0, 0.0),
        std::hypot(0.03, 0.025)
    ));
    model.shapes.push_back(makePrimitive(
        17u,
        MR_SHAPE_CYLINDER,
        f4(-0.004, 0.006, -0.053),
        identity,
        f4(0.03, 0.015, 0.0, 0.0),
        std::hypot(0.03, 0.015)
    ));
    model.shapes.push_back(makePrimitive(
        23u,
        MR_SHAPE_CYLINDER,
        f4(0.0, -0.04, -0.01),
        shoulderPitchRotation,
        f4(0.03, 0.025, 0.0, 0.0),
        std::hypot(0.03, 0.025)
    ));
    model.shapes.push_back(makePrimitive(
        24u,
        MR_SHAPE_CYLINDER,
        f4(-0.004, -0.006, -0.053),
        identity,
        f4(0.03, 0.015, 0.0, 0.0),
        std::hypot(0.03, 0.015)
    ));

    std::array<
        std::vector<std::uint32_t>,
        kUnitreeG1MeshCollisionCount
    > elementShapes;
    for (const G1CollisionHullRecord& source : kG1CollisionHulls) {
        if (source.sourceElement >= elementShapes.size() ||
            source.pieceCount != 2u ||
            source.pieceIndex >= source.pieceCount) {
            throw std::logic_error(
                "pinned G1 compound collision record is invalid: " +
                std::string{source.sourceMesh}
            );
        }
        elementShapes[source.sourceElement].push_back(
            appendAuthoredCollisionHull(model, source)
        );
    }
    // The official wrist and elbow shells intentionally overlap across the
    // intervening compact wrist joint at the authored default pose. Preserve
    // self-collision everywhere else, but remove these four permanent
    // internal interfaces from broadphase just as the connected-link filter
    // removes direct joint neighbours.
    constexpr std::array<std::array<std::uint32_t, 2>, 4>
        overlappingElements{{
            {15u, 17u},
            {16u, 18u},
            {20u, 22u},
            {21u, 23u},
        }};
    for (const auto& elements : overlappingElements) {
        for (const std::uint32_t left :
             elementShapes[elements[0]]) {
            for (const std::uint32_t right :
                 elementShapes[elements[1]]) {
                model.collisionExclusions.push_back({left, right});
            }
        }
    }
    model.world.shapeCount =
        static_cast<mr_u32>(model.shapes.size());
    if (model.world.shapeCount !=
        kUnitreeG1ExecutableShapeCount) {
        throw std::logic_error(
            "pinned G1 compound collision topology changed"
        );
    }
    model.shapeNames.reserve(model.shapes.size());
    for (std::size_t shapeIndex = 0u;
         shapeIndex < model.shapes.size();
         ++shapeIndex) {
        const std::uint32_t bodyIndex =
            model.shapes[shapeIndex].bodyIndex;
        model.shapeNames.push_back(
            model.bodyNames[bodyIndex] +
            "/collision_" + std::to_string(shapeIndex)
        );
    }

    model.defaultQ.reserve(36u);
    model.defaultQ.insert(
        model.defaultQ.end(),
        {
            // RL Lab's 0.8 m reset is the pelvis link-frame origin. The
            // generic ABI stores root translation at the COM.
            0.0f,
            0.0f,
            static_cast<float>(
                0.8 + kBodies[0].centerOfMass[2]
            ),
            0.0f, 0.0f, 0.0f, 1.0f,
        }
    );
    model.defaultQ.insert(
        model.defaultQ.end(),
        kRLLabResetQ.begin(),
        kRLLabResetQ.end()
    );
    model.name += "_mjlab_control";
    model.defaultV.assign(35u, 0.0f);
    return model;
}

} // namespace metalrobo
