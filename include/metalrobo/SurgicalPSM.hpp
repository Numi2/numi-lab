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
inline constexpr std::size_t kSurgicalPSMShapeCount = 24u;
inline constexpr std::size_t kSurgicalPSMJawCount = 2u;
inline constexpr std::size_t kSurgicalPSMArmDofCount = 6u;
inline constexpr std::size_t
    kSurgicalPSMLogicalPositionTargetCount = 7u;
inline constexpr std::size_t
    kSurgicalPSMLogicalJawApertureIndex = 6u;
inline constexpr std::size_t kSurgicalPSMToolActuatorCount = 4u;

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
    // Official 8 mm X/Xi Large Needle Driver envelope. The Classic research
    // transmission remains JHU-sourced; this dimensional reference does not
    // imply interchangeability or clinical validation.
    float instrumentDiameter = 0.0f;
    float largeNeedleDriverJawLength = 0.0f;
    // Research system calibration for the unresolved replaceable insert,
    // clevis, transmission, and contact patch. These are not manufacturer
    // material data or a clinical force prescription. Surgical composition
    // rescales each insert material so geometric pair mixing with the authored
    // needle material yields the target effective friction coefficients.
    float insertSystemNormalComplianceMPerN = 0.0f;
    float targetNeedleInsertStaticFriction = 0.0f;
    float targetNeedleInsertDynamicFriction = 0.0f;
    std::string_view intuitiveInstrumentCatalog;
    std::string_view intuitiveInstrumentPartNumber;
    float orbitToolYawLinkMass = 0.0f;
    float orbitFixedToolTipMass = 0.0f;
    bool independentJawCoordinates = false;
    bool fixedToolTipMassFoldedIntoYaw = false;
    bool calibratedInertias = false;
    bool clinicallyValidated = false;

    // JHU Large Needle Driver 400006 transmission. Rows are the physical
    // roll, wrist-pitch, wrist-yaw, and jaw coordinates; columns are the four
    // tool actuator coordinates. This is q_tool = C * q_actuator.
    std::array<float,
        kSurgicalPSMToolActuatorCount *
        kSurgicalPSMToolActuatorCount> actuatorToJointPosition{};
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

// Fixed-root, eight-coordinate dVRK PSM with a Classic Large Needle Driver.
// The first six coordinates reproduce the research control topology:
// yaw, pitch, prismatic insertion, roll, wrist pitch, wrist yaw. The two jaws
// are retained as collision-bearing generalized coordinates and coupled by an
// executable ConstraintIR gear row. Source-provenanced JHU actuator coupling
// is exposed separately because actuator coordinates are not generalized
// coordinates and must not be baked into the rigid-body tree.
[[nodiscard]] EngineModel
makeDvrkPsmLargeNeedleDriverEngineModel();

enum class SurgicalPSMCommandMapStatus : std::uint32_t {
    success = 0u,
    invalidModel,
    invalidDimensions,
    nonfiniteTarget,
    negativeJawAperture,
    physicalLimitViolation,
    physicalEffortLimitViolation,
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

// Hardware-facing position-command map:
//   actuator[0..2] -> arm yaw, pitch, insertion
//   actuator[3..6] -> LND roll, pitch, yaw, jaw actuator coordinates
// The pinned JHU 4x4 ActuatorToJointPosition matrix is applied to the tool
// coordinates, then the resulting jaw coordinate is represented by the two
// constrained collision jaws. No clamping is performed and output remains
// unchanged on rejection.
[[nodiscard]] SurgicalPSMCommandMapDiagnostics
expandSurgicalPSMActuatorPositionTargets(
    const EngineModel& model,
    std::span<const double> actuatorTargets,
    std::vector<double>& physicalTargets
);

// Hardware-facing effort map, conjugate to the position map by virtual work:
//   tau_tool_joint = inverse(transpose(C)) * tau_tool_actuator
// where C is the pinned JHU ActuatorToJointPosition matrix. The jaw effort is
// expanded into equal-and-opposite generalized efforts for the constrained
// collision jaws. Inputs and source-authored joint effort limits are checked;
// output remains unchanged on rejection.
[[nodiscard]] SurgicalPSMCommandMapDiagnostics
expandSurgicalPSMActuatorEfforts(
    const EngineModel& model,
    std::span<const double> actuatorEfforts,
    std::vector<double>& generalizedEfforts
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
