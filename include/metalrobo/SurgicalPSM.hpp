#pragma once

#include "metalrobo/EngineModel.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace metalrobo {

inline constexpr std::size_t kSurgicalPSMBodyCount = 9u;
inline constexpr std::size_t kSurgicalPSMJointCount = 8u;
inline constexpr std::size_t kSurgicalPSMShapeCount = 20u;
inline constexpr std::size_t kSurgicalPSMJawCount = 2u;
inline constexpr std::size_t kSurgicalPSMArmDofCount = 6u;
inline constexpr std::size_t
    kSurgicalPSMLogicalPositionTargetCount = 7u;
inline constexpr std::size_t
    kSurgicalPSMLogicalJawApertureIndex = 6u;

struct SurgicalPSMJointMetadata {
    std::string_view name;
    std::uint32_t jointType = MR_JOINT_FIXED;
    std::uint32_t parentBody = 0u;
    std::uint32_t childBody = 0u;
    float lowerPosition = 0.0f;
    float upperPosition = 0.0f;
    float maximumEffort = 0.0f;
    float maximumVelocity = 0.0f;
    float stiffness = 0.0f;
    float damping = 0.0f;
    float armature = 0.0f;
};

struct SurgicalPSMModelMetadata {
    std::string_view modelName;
    std::string_view fidelityContract;

    std::string_view orbitRepository;
    std::string_view orbitCommit;
    std::string_view orbitLicense;
    std::string_view orbitModelPath;
    std::string_view orbitPresetPath;

    std::string_view dvrkRepository;
    std::string_view dvrkCommit;
    std::string_view dvrkLicenseRepository;
    std::string_view dvrkLicenseCommit;
    std::string_view dvrkLicense;
    std::string_view dvrkArmKinematicsPath;
    std::string_view dvrkToolKinematicsPath;
    std::string_view toolModelNumber;

    std::array<std::string_view, kSurgicalPSMBodyCount> bodyNames{};
    std::array<SurgicalPSMJointMetadata, kSurgicalPSMJointCount> joints{};

    // The fixed root state is COM-centred. These two points are expressed
    // relative to their owning body's COM-centred runtime pose. The tool
    // control point is an authored RL task frame 32 mm past the wrist-yaw
    // origin, not a calibrated physical tooltip.
    std::uint32_t remoteCenterBodyIndex = 0u;
    mr_float4 remoteCenterLocalPosition{};
    std::uint32_t researchToolControlPointBodyIndex = 0u;
    mr_float4 researchToolControlPointLocalPosition{};

    float classicShaftLength = 0.0f;
    float wristLinkOffset = 0.0f;
    float orbitToolYawLinkMass = 0.0f;
    float orbitFixedToolTipMass = 0.0f;
    bool independentJawCoordinates = false;
    bool fixedToolTipMassFoldedIntoYaw = false;
    bool calibratedInertias = false;
    bool clinicallyValidated = false;
};

// Open-source attribution and fidelity boundary
// ------------------------------------------------
// Topology, names, masses, limits, reset state, and the named actuator preset
// are adapted from ORBIT-Surgical (BSD-3-Clause). The upstream fixed 0.1 kg
// tooltip is explicitly folded into its moving yaw parent:
//   Copyright (c) 2024, The ORBIT-Surgical Project Developers.
//   https://github.com/orbit-surgical/orbit-surgical
//   commit 6e47534f7d412e4be523116f250c992a63146883
//
// The JHU dVRK Classic PSM and Large Needle Driver definitions supply the
// Classic shaft/wrist controller records and primary-source kinematic
// cross-checks. Their license explicitly advises against clinical use:
//   https://github.com/jhu-dvrk/sawIntuitiveResearchKit
//   commit 53a401d014e5ef8a7d5e3ad05f0680084507662c
//
// MetalRobo's serial RCM construction, primitive collision decomposition,
// inertial approximation, ABI compilation, and inverse calculation are
// original implementation work. This model is not a digital twin, dynamics
// calibration, safety model, or clinical device representation.
[[nodiscard]] const SurgicalPSMModelMetadata&
surgicalPSMMetadata() noexcept;

// Fixed-root, eight-coordinate dVRK-style PSM with a Classic Large Needle
// Driver. The first six coordinates reproduce the research control topology:
// yaw, pitch, prismatic insertion, roll, wrist pitch, wrist yaw. The two jaws
// remain independent generalized coordinates because tendon/transmission
// constraints are not yet executable in the articulated ABA path.
[[nodiscard]] EngineModel
makeDvrkPsmLargeNeedleDriverEngineModel();

enum class SurgicalPSMCommandMapStatus : std::uint32_t {
    success = 0u,
    invalidModel,
    invalidDimensions,
    nonfiniteTarget,
    negativeJawAperture,
    physicalLimitViolation,
};

struct SurgicalPSMCommandMapDiagnostics {
    SurgicalPSMCommandMapStatus status =
        SurgicalPSMCommandMapStatus::success;
    std::uint32_t rejectedLogicalIndex = MR_INVALID_INDEX;
    std::uint32_t rejectedPhysicalIndex = MR_INVALID_INDEX;
    double requestedJawAperture = 0.0;
    double maximumJawAperture = 0.0;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == SurgicalPSMCommandMapStatus::success;
    }
};

// Policy-facing position-command map:
//   logical[0..5] -> yaw, pitch, insertion, roll, wrist pitch, wrist yaw
//   logical[6]    -> non-negative total angular jaw aperture
//   physical[6]  = -logical[6] / 2
//   physical[7]  = +logical[6] / 2
//
// This is only a symmetric command-coordinate map. It does not add tendon,
// cable, gear, transmission, or constraint dynamics to the independent-jaw
// articulated model. Every logical input and resulting physical target is
// checked against the supplied canonical model. There is no clamping. On any
// failure, physicalTargets is unchanged.
[[nodiscard]] SurgicalPSMCommandMapDiagnostics
expandSurgicalPSMLogicalPositionTargets(
    const EngineModel& model,
    std::span<const double> logicalTargets,
    std::vector<double>& physicalTargets
);

// Canonical ORBIT reset expressed in the seven-coordinate command space.
// The jaw entry is the total aperture, defaultQ[7] - defaultQ[6].
[[nodiscard]] std::array<
    float,
    kSurgicalPSMLogicalPositionTargetCount
> surgicalPSMDefaultLogicalPositionTargets() noexcept;

[[nodiscard]] const char* surgicalPSMCommandMapStatusName(
    SurgicalPSMCommandMapStatus status
) noexcept;

} // namespace metalrobo
