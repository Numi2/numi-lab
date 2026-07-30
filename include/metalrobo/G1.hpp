#pragma once

#include "metalrobo/EngineModel.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace metalrobo {

inline constexpr std::size_t kUnitreeG1BodyCount = 30u;
inline constexpr std::size_t kUnitreeG1JointCount = 29u;
inline constexpr std::size_t kUnitreeG1FootCount = 2u;
inline constexpr std::size_t kUnitreeG1ImuCount = 2u;
inline constexpr std::size_t kUnitreeG1FootSphereCount = 8u;
inline constexpr std::size_t kUnitreeG1MeshCollisionCount = 24u;
inline constexpr std::size_t kUnitreeG1PrimitiveShapeCount = 12u;
inline constexpr std::size_t kUnitreeG1OfficialCollisionElementCount = 36u;
inline constexpr std::size_t kUnitreeG1ExecutableShapeCount = 60u;

struct G1JointLimit {
    std::string_view name;
    std::uint32_t parentBody = 0u;
    std::uint32_t childBody = 0u;
    float lowerPosition = 0.0f;
    float upperPosition = 0.0f;
    float maximumEffort = 0.0f;
    float maximumVelocity = 0.0f;
};

// This is a named training preset, not an intrinsic motor specification.
struct G1RLLabJointDrive {
    std::string_view name;
    float stiffness = 0.0f;
    float damping = 0.0f;
    float armature = 0.0f;
};

struct G1FootFrame {
    std::string_view name;
    std::string_view bodyName;
    std::uint32_t bodyIndex = 0u;
    // Derived sole frame relative to the ankle-roll body's COM-centred
    // runtime state, quaternion xyzw.
    mr_float4 solePosition{};
    mr_float4 soleRotation{};
    std::array<std::uint32_t, 4> sphereShapeIndices{};
};

struct G1ImuFrame {
    std::string_view name;
    std::string_view bodyName;
    std::string_view transport;
    std::uint32_t bodyIndex = 0u;
    // Position relative to the body's COM-centred runtime state.
    mr_float4 localPosition{};
    mr_float4 localRotation{};
    float gyroscopeNoise = 0.0f;
    float gyroscopeCutoff = 0.0f;
    float accelerometerNoise = 0.0f;
    float accelerometerCutoff = 0.0f;
};

struct G1ModelMetadata {
    std::string_view modelName;
    std::string_view sourceRepository;
    std::string_view sourceCommit;
    std::string_view sourceModelPath;
    std::string_view sourceLicense;
    std::string_view rlPresetRepository;
    std::string_view rlPresetCommit;
    std::string_view rlPresetLicense;
    std::string_view collisionMaterialPreset;
    std::string_view collisionCookMethod;
    std::string_view collisionCookHash;

    std::uint32_t modeMachine = 0u;
    std::uint32_t modePr = 0u;
    double canonicalMassKg = 0.0;

    std::array<std::string_view, kUnitreeG1BodyCount> bodyNames{};
    std::array<G1JointLimit, kUnitreeG1JointCount> jointLimits{};
    std::array<G1RLLabJointDrive, kUnitreeG1JointCount> rlLabDrives{};
    std::array<G1FootFrame, kUnitreeG1FootCount> feet{};
    std::array<G1ImuFrame, kUnitreeG1ImuCount> imus{};
};

// Factual model constants and primitive collision records are adapted from:
//   Copyright (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD.
//   unitreerobotics/unitree_ros, BSD-3-Clause
//   commit aa0f5c68b5aba347bad409e71b6430407da758d7
// The named RL preset is adapted from Unitree RL Lab's pinned G1 configuration;
// its source file carries Copyright (c) 2022-2025, The Isaac Lab Project
// Developers, SPDX-License-Identifier: BSD-3-Clause.
[[nodiscard]] const G1ModelMetadata& unitreeG1Metadata() noexcept;

// Compiles the pinned mode_machine=5, mode_pr=0 G1 revision into the generic
// engine ABI. Root q = COM xyz + body/link orientation quaternion xyzw;
// root v = COM linear velocity + world angular velocity.
[[nodiscard]] EngineModel makeUnitreeG1EngineModel();

} // namespace metalrobo
