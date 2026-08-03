#include "metalrobo/SurgicalPSM.hpp"

#include "metalrobo/ConstraintIR.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

// Surgical PSM source attribution and fidelity boundary
// ------------------------------------------------------
// The open numerical records used here are adapted from:
//
//   ORBIT-Surgical, BSD-3-Clause
//   Copyright (c) 2024, The ORBIT-Surgical Project Developers.
//   commit 6e47534f7d412e4be523116f250c992a63146883
//   https://github.com/orbit-surgical/orbit-surgical/blob/
//     6e47534f7d412e4be523116f250c992a63146883/LICENCE
//   source/extensions/orbit.surgical.assets/data/Robots/dVRK/PSM/psm_col.usd
//   source/extensions/orbit.surgical.assets/orbit/surgical/assets/psm.py
//
// The kinematic topology and Classic Large Needle Driver identity were
// independently cross-checked against JHU's primary dVRK definitions:
//
//   https://github.com/jhu-dvrk/sawIntuitiveResearchKit/blob/
//     53a401d014e5ef8a7d5e3ad05f0680084507662c/
//     share/kinematic/PSM.json
//   https://github.com/jhu-dvrk/sawIntuitiveResearchKit/blob/
//     53a401d014e5ef8a7d5e3ad05f0680084507662c/
//     share/tool/LARGE_NEEDLE_DRIVER_400006.json
//
// sawIntuitiveResearchKit points to the CISST Software License Agreement:
//   https://github.com/jhu-cisst/cisst/blob/
//     7e95680b9461009b745567f382d1b498eabc046b/license.txt
// That agreement grants reuse and sublicensing rights, describes the JHU
// software as designed for research purposes, and separately says clinical
// use is neither recommended nor advised. The complete agreement is retained
// in licenses/CISST_LICENSE.txt.
//
// The full dVRK mechanism contains cable elasticity, backlash, sterile-adapter
// compliance, and hardware-specific calibration that are not published by
// the primary source and therefore cannot be truthfully invented here. This
// is an original serial rigid-body
// equivalent: its first three axes share an exact remote center and its true
// prismatic coordinate moves a 0.4162 m Classic shaft through that center.
// ORBIT's fixed 0.1 kg psm_tool_tip_link is folded into its 0.1 kg moving yaw
// parent instead of being dropped from the reduced tree. ORBIT exposes two jaw
// coordinates; they intentionally remain independent until MetalRobo has
// executable tendon/transmission constraints. Primitive inertias below are
// conservative geometry-based research approximations, not copied hardware
// calibration.

namespace metalrobo {
namespace {

struct BodySource {
    const char* name;
    std::uint32_t parentBody;
    std::uint32_t inboundJoint;
    double mass;
    std::array<double, 3> centerOfMass;
    // Ixx, Ixy, Ixz, Iyy, Iyz, Izz about COM in the link frame.
    std::array<double, 6> inertia;
};

struct JointSource {
    const char* name;
    std::uint32_t parentBody;
    std::uint32_t childBody;
    std::uint32_t jointType;
    std::array<double, 3> origin;
    std::array<double, 3> axis;
    double lower;
    double upper;
    double effort;
    double velocity;
    double stiffness;
    double damping;
    double armature;
};

struct ShapeSource {
    std::uint32_t bodyIndex;
    std::uint32_t shapeType;
    std::array<double, 3> position;
    std::array<double, 4> rotation;
    std::array<double, 3> dimensions;
    double boundingRadius;
};

constexpr double kClassicShaftLength = 0.4162;
constexpr double kWristLinkOffset = 0.0091;
constexpr double kOrbitToolYawLinkMass = 0.1;
constexpr double kOrbitFixedToolTipMass = 0.1;
// ORBIT's fixed joint places the tooltip at +Y=9.3 mm in its yaw-link frame.
// MetalRobo's serial frame maps that tool-longitudinal direction onto +Z.
constexpr double kCanonicalFixedToolTipZ = 0.0093;
constexpr double kFoldedToolYawMass =
    kOrbitToolYawLinkMass + kOrbitFixedToolTipMass;
constexpr double kFoldedToolYawCenterZ =
    kOrbitFixedToolTipMass * kCanonicalFixedToolTipZ /
    kFoldedToolYawMass;
constexpr std::uint32_t kRobotCollisionGroup = 1u << 8u;
constexpr std::uint32_t kJawConstraintKey = 0x50534d4au;
constexpr std::array<double, 16u> kActuatorToJointPosition{{
    -1.5632,  0.0000,  0.0000,  0.0000,
     0.0000,  1.0186,  0.0000,  0.0000,
     0.0000, -0.8306,  0.6089,  0.6089,
     0.0000,  0.0000, -1.2177,  1.2177,
}};
// inverse(transpose(kActuatorToJointPosition)); retained explicitly so the
// real-time command boundary performs a fixed multiply, not a matrix solve.
constexpr std::array<double, 16u> kActuatorToJointEffort{{
    -0.639713408393, 0.0,            0.0,            0.0,
     0.0,            0.981739642647, 0.669595128250, 0.669595128250,
     0.0,            0.0,            0.821152898670, 0.821152898670,
     0.0,            0.0,           -0.410610166708, 0.410610166708,
}};

constexpr std::array<double, 6> boxInertia(
    const double mass,
    const double x,
    const double y,
    const double z
) {
    return {
        mass * (y * y + z * z) / 12.0,
        0.0,
        0.0,
        mass * (x * x + z * z) / 12.0,
        0.0,
        mass * (x * x + y * y) / 12.0,
    };
}

constexpr std::array<double, 6> axialCylinderInertia(
    const double mass,
    const double radius,
    const double length
) {
    const double transverse =
        mass * (3.0 * radius * radius + length * length) / 12.0;
    return {
        transverse,
        0.0,
        0.0,
        transverse,
        0.0,
        0.5 * mass * radius * radius,
    };
}

constexpr std::array<double, 6> foldedToolYawInertia() {
    // The upstream USD supplies masses and the fixed-joint transform but no
    // inertia tensors for either rigid prim. Keep the yaw housing's authored
    // box inertia and treat the folded tooltip as a point mass, then apply the
    // parallel-axis theorem about the combined canonical COM.
    std::array<double, 6> result = boxInertia(
        kOrbitToolYawLinkMass,
        0.012,
        0.018,
        0.018
    );
    const double yawOffset = -kFoldedToolYawCenterZ;
    const double tipOffset =
        kCanonicalFixedToolTipZ - kFoldedToolYawCenterZ;
    const double transverseShift =
        kOrbitToolYawLinkMass * yawOffset * yawOffset +
        kOrbitFixedToolTipMass * tipOffset * tipOffset;
    result[0] += transverseShift;
    result[3] += transverseShift;
    return result;
}

constexpr std::array<BodySource, kSurgicalPSMBodyCount> kBodies{{
    {
        "psm_base_link",
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
        2.0161,
        {0.0, 0.0, -0.15},
        boxInertia(2.0161, 0.22, 0.18, 0.24),
    },
    {
        "psm_yaw_link",
        0u,
        0u,
        1.4705,
        {0.0, 0.0, -0.14},
        boxInertia(1.4705, 0.11, 0.11, 0.30),
    },
    {
        "psm_pitch_link",
        1u,
        1u,
        2.091,
        {0.0, 0.0, -0.08},
        boxInertia(2.091, 0.14, 0.10, 0.20),
    },
    {
        "psm_main_insertion_link",
        2u,
        2u,
        0.22491,
        {0.0, 0.0, -0.5 * kClassicShaftLength},
        axialCylinderInertia(
            0.22491,
            0.004,
            kClassicShaftLength
        ),
    },
    {
        "psm_tool_roll_link",
        3u,
        3u,
        0.00033225,
        {0.0, 0.0, 0.004},
        axialCylinderInertia(0.00033225, 0.006, 0.014),
    },
    {
        "psm_tool_pitch_link",
        4u,
        4u,
        0.00025784,
        {0.0, 0.0, 0.004},
        boxInertia(0.00025784, 0.012, 0.012, 0.014),
    },
    {
        "psm_tool_yaw_link",
        5u,
        5u,
        kFoldedToolYawMass,
        {0.0, 0.0, kFoldedToolYawCenterZ},
        foldedToolYawInertia(),
    },
    {
        "psm_tool_gripper1_link",
        6u,
        6u,
        0.02,
        {0.0013, 0.0, 0.011},
        boxInertia(0.02, 0.004, 0.003, 0.022),
    },
    {
        "psm_tool_gripper2_link",
        6u,
        7u,
        0.02,
        {-0.0013, 0.0, 0.011},
        boxInertia(0.02, 0.004, 0.003, 0.022),
    },
}};

// Limits, reset values, and the named PD preset match ORBIT-Surgical's open
// PSM asset/config. Armature is an explicitly MetalRobo-authored numerical
// conditioning prior, not an identified reflected rotor inertia.
constexpr std::array<JointSource, kSurgicalPSMJointCount> kJoints{{
    {
        "psm_yaw_joint",
        0u, 1u, MR_JOINT_REVOLUTE,
        {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
        -1.588, 1.588,
        3.316101, 1.0, 800.0, 40.0, 0.01,
    },
    {
        "psm_pitch_end_joint",
        1u, 2u, MR_JOINT_REVOLUTE,
        {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
        -0.925025, 0.925025,
        3.316101, 1.0, 800.0, 40.0, 0.01,
    },
    {
        "psm_main_insertion_joint",
        2u, 3u, MR_JOINT_PRISMATIC,
        {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
        0.0, 0.24, 9.877926, 1.0, 800.0, 40.0, 0.02,
    },
    {
        "psm_tool_roll_joint",
        3u, 4u, MR_JOINT_REVOLUTE,
        {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
        -4.53786, 4.53786,
        0.33, 1.0, 800.0, 40.0, 0.001,
    },
    {
        "psm_tool_pitch_joint",
        4u, 5u, MR_JOINT_REVOLUTE,
        {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
        -1.39626, 1.39626,
        0.25, 1.0, 800.0, 40.0, 0.0002,
    },
    {
        "psm_tool_yaw_joint",
        5u, 6u, MR_JOINT_REVOLUTE,
        {0.0, 0.0, kWristLinkOffset}, {1.0, 0.0, 0.0},
        -1.39626, 1.39626,
        0.20, 1.0, 800.0, 40.0, 0.0002,
    },
    {
        "psm_tool_gripper1_joint",
        6u, 7u, MR_JOINT_REVOLUTE,
        {0.0, 0.0, 0.012}, {0.0, -1.0, 0.0},
        -0.698132, 0.0,
        0.16, 0.2, 500.0, 0.1, 0.00002,
    },
    {
        "psm_tool_gripper2_joint",
        6u, 8u, MR_JOINT_REVOLUTE,
        {0.0, 0.0, 0.012}, {0.0, -1.0, 0.0},
        0.0, 0.698132,
        0.16, 0.2, 500.0, 0.1, 0.00002,
    },
}};

constexpr double kSqrtHalf = 0.7071067811865475244;
constexpr std::array<double, 4> kIdentity{0.0, 0.0, 0.0, 1.0};
// Rotate the primitive's canonical +Y axis onto +Z.
constexpr std::array<double, 4> kYToZ{
    kSqrtHalf, 0.0, 0.0, kSqrtHalf,
};

// Compound collision is deliberately low-poly and executable by both the CPU
// and Metal narrow phases. All robot shapes use an internal group bit excluded
// from their own masks, matching ORBIT's disabled PSM self-collision preset
// while retaining collision with differently grouped scene assets.
constexpr std::array<ShapeSource, kSurgicalPSMShapeCount> kShapes{{
    {0u, MR_SHAPE_BOX, {0.0, 0.0, -0.16}, kIdentity,
     {0.11, 0.09, 0.12}, 0.1854723699},
    {0u, MR_SHAPE_CAPSULE, {0.0, 0.0, -0.035}, kYToZ,
     {0.025, 0.04, 0.0}, 0.065},

    {1u, MR_SHAPE_BOX, {0.0, 0.0, -0.15}, kIdentity,
     {0.055, 0.055, 0.15}, 0.1691153453},
    {1u, MR_SHAPE_SPHERE, {0.0, 0.0, 0.0}, kIdentity,
     {0.04, 0.0, 0.0}, 0.04},

    {2u, MR_SHAPE_BOX, {0.0, 0.0, -0.10}, kIdentity,
     {0.07, 0.05, 0.10}, 0.1319090596},
    {2u, MR_SHAPE_CYLINDER, {0.0, 0.0, 0.0}, kIdentity,
     {0.045, 0.05, 0.0}, 0.0672681202},

    {3u, MR_SHAPE_CAPSULE,
     {0.0, 0.0, -0.5 * kClassicShaftLength}, kYToZ,
     {0.004, 0.2041, 0.0}, 0.2081},
    {3u, MR_SHAPE_CYLINDER, {0.0, 0.0, -0.02}, kYToZ,
     {0.011, 0.018, 0.0}, 0.0210950231},

    {4u, MR_SHAPE_CYLINDER, {0.0, 0.0, 0.004}, kYToZ,
     {0.006, 0.007, 0.0}, 0.0092195445},
    {4u, MR_SHAPE_SPHERE, {0.0, 0.0, 0.0}, kIdentity,
     {0.007, 0.0, 0.0}, 0.007},

    {5u, MR_SHAPE_BOX, {0.0, 0.0, 0.005}, kIdentity,
     {0.006, 0.006, 0.007}, 0.011},
    {5u, MR_SHAPE_SPHERE, {0.0, 0.0, kWristLinkOffset}, kIdentity,
     {0.006, 0.0, 0.0}, 0.006},

    {6u, MR_SHAPE_BOX, {0.0, 0.0, 0.008}, kIdentity,
     {0.006, 0.009, 0.008}, 0.0134536240},
    {6u, MR_SHAPE_CAPSULE, {0.0, 0.0, 0.018}, kYToZ,
     {0.004, 0.008, 0.0}, 0.012},

    // At q=0 the two authored jaw surfaces are tangent on the center plane.
    // With the -Y joint axes above, ORBIT's [-q,+q] convention then opens the
    // jaws monotonically instead of driving the primitive tips through one
    // another. The four 0.35 mm spheres form a deliberately research-default
    // finite contact patch: two proud teeth per jaw, separated along the jaw's
    // longitudinal Z axis. That separation supplies a real contact moment arm
    // around a transversely held needle instead of duplicating one point
    // contact. This is executable collision geometry, not a calibrated LND
    // tooth profile.
    {7u, MR_SHAPE_CAPSULE, {0.0013, 0.0, 0.011}, kYToZ,
     {0.0013, 0.010, 0.0}, 0.0113},
    {7u, MR_SHAPE_SPHERE, {0.00035, 0.0, 0.02160}, kIdentity,
     {0.00035, 0.0, 0.0}, 0.00035},
    {8u, MR_SHAPE_CAPSULE, {-0.0013, 0.0, 0.011}, kYToZ,
     {0.0013, 0.010, 0.0}, 0.0113},
    {8u, MR_SHAPE_SPHERE, {-0.00035, 0.0, 0.02160}, kIdentity,
     {0.00035, 0.0, 0.0}, 0.00035},
    {7u, MR_SHAPE_SPHERE, {0.00035, 0.0, 0.02240}, kIdentity,
     {0.00035, 0.0, 0.0}, 0.00035},
    {8u, MR_SHAPE_SPHERE, {-0.00035, 0.0, 0.02240}, kIdentity,
     {0.00035, 0.0, 0.0}, 0.00035},
}};

constexpr std::array<float, kSurgicalPSMJointCount> kDefaultQ{{
    0.01f,
    0.01f,
    0.07f,
    0.01f,
    0.01f,
    0.01f,
    -0.09f,
    0.09f,
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
        throw std::logic_error(
            "surgical PSM research inertia is not positive definite"
        );
    }
    return {
        f4(
            (d * f - e * e) / determinant,
            (c * e - b * f) / determinant,
            (b * e - c * d) / determinant
        ),
        f4(
            (c * e - b * f) / determinant,
            (a * f - c * c) / determinant,
            (b * c - a * e) / determinant
        ),
        f4(
            (b * e - c * d) / determinant,
            (b * c - a * e) / determinant,
            (a * d - b * b) / determinant
        ),
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
    body.dampingAndSpeedLimits = f4(0.02, 0.02, 20.0, 20.0);
    return body;
}

MRJointDescriptorGPU makeJoint(
    const JointSource& source,
    const std::uint32_t jointIndex
) {
    MRJointDescriptorGPU joint{};
    joint.parentBody = source.parentBody;
    joint.childBody = source.childBody;
    joint.jointType = source.jointType;
    joint.qOffset = jointIndex;
    joint.nq = 1u;
    joint.vOffset = jointIndex;
    joint.nv = 1u;
    joint.axis0 = f4(source.axis[0], source.axis[1], source.axis[2]);

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
    joint.parentRotation = f4(0.0, 0.0, 0.0, 1.0);
    joint.childRotation = f4(0.0, 0.0, 0.0, 1.0);
    return joint;
}

MRDofPropertiesGPU makeDof(
    const JointSource& source,
    const std::uint32_t jointIndex
) {
    MRDofPropertiesGPU dof{};
    dof.articulationIndex = 0u;
    dof.jointIndex = jointIndex;
    dof.qIndex = jointIndex;
    dof.vIndex = jointIndex;
    dof.localDof = 0u;
    dof.flags =
        MR_DOF_FLAG_ACTUATED |
        MR_DOF_FLAG_POSITION_LIMIT |
        MR_DOF_FLAG_VELOCITY_LIMIT |
        MR_DOF_FLAG_EFFORT_LIMIT |
        MR_DOF_FLAG_DRIVE;
    dof.limits = f4(
        source.lower,
        source.upper,
        source.velocity,
        source.effort
    );
    dof.drive = f4(
        source.stiffness,
        source.damping,
        source.armature,
        0.0
    );
    return dof;
}

MRShapeGPU makeShape(const ShapeSource& source) {
    MRShapeGPU shape{};
    shape.bodyIndex = source.bodyIndex;
    shape.shapeType = source.shapeType;
    shape.materialIndex = 0u;
    shape.collisionGroup = kRobotCollisionGroup;
    shape.collisionMask = ~kRobotCollisionGroup;
    shape.slotGeneration = 1u;
    const BodySource& body = kBodies[source.bodyIndex];
    shape.localPosition = f4(
        source.position[0] - body.centerOfMass[0],
        source.position[1] - body.centerOfMass[1],
        source.position[2] - body.centerOfMass[2],
        1.0
    );
    shape.localRotation = f4(
        source.rotation[0],
        source.rotation[1],
        source.rotation[2],
        source.rotation[3]
    );
    shape.dimensions = f4(
        source.dimensions[0],
        source.dimensions[1],
        source.dimensions[2]
    );
    shape.contactRestAndBoundingRadius =
        f4(0.0, 0.0, source.boundingRadius, 0.0);
    return shape;
}

} // namespace

const SurgicalPSMModelMetadata& surgicalPSMMetadata() noexcept {
    static const SurgicalPSMModelMetadata metadata = [] {
        SurgicalPSMModelMetadata value{};
        value.modelName =
            "dvrk_psm_classic_lnd_source_coupled_abi_v3";
        value.fidelityContract =
            "source-grounded serial RCM mechanism; JHU limits/efforts/LND "
            "actuator coupling and executable symmetric jaw gear; ORBIT "
            "masses/reset/drive with fixed tooltip folded into yaw; authored "
            "primitive inertias/collision; no hardware-specific calibration "
            "or clinical validation";

        value.orbitRepository =
            "https://github.com/orbit-surgical/orbit-surgical";
        value.orbitCommit =
            "6e47534f7d412e4be523116f250c992a63146883";
        value.orbitLicense = "BSD-3-Clause";
        value.orbitModelPath =
            "source/extensions/orbit.surgical.assets/data/Robots/"
            "dVRK/PSM/psm_col.usd";
        value.orbitPresetPath =
            "source/extensions/orbit.surgical.assets/orbit/surgical/"
            "assets/psm.py";

        value.dvrkRepository =
            "https://github.com/jhu-dvrk/sawIntuitiveResearchKit";
        value.dvrkCommit =
            "53a401d014e5ef8a7d5e3ad05f0680084507662c";
        value.dvrkLicenseRepository =
            "https://github.com/jhu-cisst/cisst";
        value.dvrkLicenseCommit =
            "7e95680b9461009b745567f382d1b498eabc046b";
        value.dvrkLicense =
            "CISST Software License Agreement 1.0; complete text in "
            "licenses/CISST_LICENSE.txt; clinical-use advisory applies";
        value.dvrkArmKinematicsPath = "share/kinematic/PSM.json";
        value.dvrkToolKinematicsPath =
            "share/tool/LARGE_NEEDLE_DRIVER_400006.json";
        value.toolModelNumber = "400006";

        for (std::size_t index = 0u; index < kBodies.size(); ++index) {
            value.bodyNames[index] = kBodies[index].name;
        }
        for (std::size_t index = 0u; index < kJoints.size(); ++index) {
            const JointSource& source = kJoints[index];
            value.joints[index] = {
                source.name,
                source.jointType,
                source.parentBody,
                source.childBody,
                static_cast<float>(source.lower),
                static_cast<float>(source.upper),
                static_cast<float>(source.effort),
                static_cast<float>(source.velocity),
                static_cast<float>(source.stiffness),
                static_cast<float>(source.damping),
                static_cast<float>(source.armature),
            };
        }

        value.remoteCenterBodyIndex = 0u;
        value.remoteCenterLocalPosition = f4(
            -kBodies[0].centerOfMass[0],
            -kBodies[0].centerOfMass[1],
            -kBodies[0].centerOfMass[2],
            1.0
        );
        value.researchToolControlPointBodyIndex = 6u;
        value.researchToolControlPointLocalPosition = f4(
            -kBodies[6].centerOfMass[0],
            -kBodies[6].centerOfMass[1],
            0.032 - kBodies[6].centerOfMass[2],
            1.0
        );
        value.classicShaftLength =
            static_cast<float>(kClassicShaftLength);
        value.wristLinkOffset =
            static_cast<float>(kWristLinkOffset);
        value.orbitToolYawLinkMass =
            static_cast<float>(kOrbitToolYawLinkMass);
        value.orbitFixedToolTipMass =
            static_cast<float>(kOrbitFixedToolTipMass);
        value.independentJawCoordinates = false;
        value.fixedToolTipMassFoldedIntoYaw = true;
        value.calibratedInertias = false;
        value.clinicallyValidated = false;
        std::transform(
            kActuatorToJointPosition.begin(),
            kActuatorToJointPosition.end(),
            value.actuatorToJointPosition.begin(),
            [](const double entry) {
                return static_cast<float>(entry);
            }
        );
        return value;
    }();
    return metadata;
}

EngineModel makeDvrkPsmLargeNeedleDriverEngineModel() {
    EngineModel model;
    const SurgicalPSMModelMetadata& metadata =
        surgicalPSMMetadata();
    model.name = std::string{metadata.modelName};

    model.world.abiVersion = MR_ENGINE_ABI_VERSION;
    model.world.bodyCount =
        static_cast<mr_u32>(kSurgicalPSMBodyCount);
    model.world.articulationCount = 1u;
    model.world.jointCount =
        static_cast<mr_u32>(kSurgicalPSMJointCount);
    model.world.shapeCount =
        static_cast<mr_u32>(kSurgicalPSMShapeCount);
    model.world.materialCount = 1u;
    model.world.nq =
        static_cast<mr_u32>(kSurgicalPSMJointCount);
    model.world.nv =
        static_cast<mr_u32>(kSurgicalPSMJointCount);
    model.world.pairCapacity = 256u;
    model.world.contactCapacity = 256u;
    model.world.constraintCapacity = 512u;
    model.world.islandCapacity = 8u;
    model.world.solverType = MR_SOLVER_THROUGHPUT_PGS;
    model.world.frictionConeType = MR_FRICTION_CONE_ELLIPTIC;
    model.world.gravityAndTimestep = f4(0.0, 0.0, -9.81, 1.0 / 1000.0);
    model.world.solverScales = f4(1.0e-7, 1.0e-9, 2.0, 1.0e-5);

    MRArticulationGPU articulation{};
    articulation.rootBody = 0u;
    articulation.rootType = MR_ROOT_FIXED;
    articulation.firstBody = 0u;
    articulation.bodyCount =
        static_cast<mr_u32>(kSurgicalPSMBodyCount);
    articulation.firstJoint = 0u;
    articulation.jointCount =
        static_cast<mr_u32>(kSurgicalPSMJointCount);
    articulation.qOffset = 0u;
    articulation.nq =
        static_cast<mr_u32>(kSurgicalPSMJointCount);
    articulation.vOffset = 0u;
    articulation.nv =
        static_cast<mr_u32>(kSurgicalPSMJointCount);
    articulation.solverGroup = 0u;
    model.articulations.push_back(articulation);

    model.bodies.reserve(kBodies.size());
    for (const BodySource& source : kBodies) {
        model.bodies.push_back(makeBody(source));
    }

    model.joints.reserve(kJoints.size());
    model.dofs.reserve(kJoints.size());
    for (std::size_t index = 0u; index < kJoints.size(); ++index) {
        const std::uint32_t jointIndex =
            static_cast<std::uint32_t>(index);
        model.joints.push_back(
            makeJoint(kJoints[index], jointIndex)
        );
        model.dofs.push_back(
            makeDof(kJoints[index], jointIndex)
        );
    }

    MRMaterialGPU material{};
    // Generic steel-on-training-fixture contact preset. It is intentionally
    // not labelled as a measured surgical material pair.
    material.friction = f4(0.6, 0.5, 0.0, 0.0);
    material.response = f4(0.0, 0.2, 0.0, 0.0);
    material.geometry = f4(0.0, 0.0, 0.0, 0.0);
    model.materials.push_back(material);

    model.shapes.reserve(kShapes.size());
    for (const ShapeSource& source : kShapes) {
        model.shapes.push_back(makeShape(source));
    }

    model.defaultQ.assign(kDefaultQ.begin(), kDefaultQ.end());
    model.defaultV.assign(kSurgicalPSMJointCount, 0.0f);

    std::array<ConstraintIREndpoint, 2u> jawEndpoints{{
        makeConstraintIRGeneralizedEndpoint(0u, 6u, 6u, 0u, 1.0f),
        makeConstraintIRGeneralizedEndpoint(0u, 7u, 7u, 0u, 1.0f),
    }};
    ConstraintIRRow jawRow{};
    jawRow.direction = f4(1.0, 0.0, 0.0);
    jawRow.timeConstant = 0.01f;
    jawRow.dampingRatio = 1.0f;
    jawRow.impulseLower = -kConstraintIRUnbounded;
    jawRow.impulseUpper = kConstraintIRUnbounded;
    std::array<ConstraintIRRow, 1u> jawRows{{jawRow}};
    std::array<float, 1u> jawWarmStart{{0.0f}};
    ConstraintIRStableKey jawKey{};
    jawKey.words[0] = kJawConstraintKey;
    std::array<ConstraintIRSourceBlock, 1u> jawSources{{{
        .key = jawKey,
        .type = MR_CONSTRAINT_GEAR,
        .flags = 0u,
        .islandIndex = 0u,
        .eventSlot = kConstraintIRInvalidIndex,
        .endpoints = jawEndpoints,
        .rows = jawRows,
        .cone = std::nullopt,
        .warmImpulses = jawWarmStart,
    }}};
    ConstraintIRCompilationResult jawProgram =
        compileConstraintIR(jawSources);
    if (!jawProgram.succeeded()) {
        throw std::logic_error(
            "internal surgical PSM jaw transmission is invalid: " +
            jawProgram.diagnostics.message
        );
    }
    model.constraintProgram = std::move(jawProgram.ir);

    std::string reason;
    if (!model.valid(&reason)) {
        throw std::logic_error(
            "internal surgical PSM model is invalid: " + reason
        );
    }
    return model;
}

SurgicalPSMCommandMapDiagnostics
expandSurgicalPSMLogicalPositionTargets(
    const EngineModel& model,
    const std::span<const double> logicalTargets,
    std::vector<double>& physicalTargets
) {
    SurgicalPSMCommandMapDiagnostics diagnostics;

    if (model.world.nq != kSurgicalPSMJointCount ||
        model.world.nv != kSurgicalPSMJointCount ||
        model.joints.size() != kSurgicalPSMJointCount ||
        model.dofs.size() != kSurgicalPSMJointCount ||
        model.defaultQ.size() != kSurgicalPSMJointCount) {
        diagnostics.status =
            SurgicalPSMCommandMapStatus::invalidModel;
        return diagnostics;
    }
    for (std::size_t index = 0u;
         index < kSurgicalPSMJointCount;
         ++index) {
        const MRJointDescriptorGPU& joint = model.joints[index];
        const MRDofPropertiesGPU& dof = model.dofs[index];
        if (joint.jointType != kJoints[index].jointType ||
            joint.qOffset != index ||
            joint.nq != 1u ||
            joint.vOffset != index ||
            joint.nv != 1u ||
            dof.jointIndex != index ||
            dof.qIndex != index ||
            dof.vIndex != index ||
            (dof.flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u ||
            !std::isfinite(dof.limits.x) ||
            !std::isfinite(dof.limits.y) ||
            dof.limits.x > dof.limits.y) {
            diagnostics.status =
                SurgicalPSMCommandMapStatus::invalidModel;
            diagnostics.rejectedPhysicalIndex =
                static_cast<std::uint32_t>(index);
            return diagnostics;
        }
    }

    if (logicalTargets.size() !=
        kSurgicalPSMLogicalPositionTargetCount) {
        diagnostics.status =
            SurgicalPSMCommandMapStatus::invalidDimensions;
        return diagnostics;
    }
    for (std::size_t index = 0u;
         index < logicalTargets.size();
         ++index) {
        if (!std::isfinite(logicalTargets[index])) {
            diagnostics.status =
                SurgicalPSMCommandMapStatus::nonfiniteTarget;
            diagnostics.rejectedLogicalIndex =
                static_cast<std::uint32_t>(index);
            return diagnostics;
        }
    }

    const double jawAperture =
        logicalTargets[kSurgicalPSMLogicalJawApertureIndex];
    diagnostics.requestedJawAperture = jawAperture;
    diagnostics.maximumJawAperture = std::min(
        -2.0 * static_cast<double>(model.dofs[6].limits.x),
        2.0 * static_cast<double>(model.dofs[7].limits.y)
    );
    if (jawAperture < 0.0) {
        diagnostics.status =
            SurgicalPSMCommandMapStatus::negativeJawAperture;
        diagnostics.rejectedLogicalIndex =
            static_cast<std::uint32_t>(
                kSurgicalPSMLogicalJawApertureIndex
            );
        return diagnostics;
    }

    std::vector<double> candidate(kSurgicalPSMJointCount, 0.0);
    std::copy_n(
        logicalTargets.begin(),
        kSurgicalPSMArmDofCount,
        candidate.begin()
    );
    candidate[6] = -0.5 * jawAperture;
    candidate[7] = 0.5 * jawAperture;

    for (std::size_t index = 0u;
         index < candidate.size();
         ++index) {
        const MRDofPropertiesGPU& dof = model.dofs[index];
        if (candidate[index] <
                static_cast<double>(dof.limits.x) ||
            candidate[index] >
                static_cast<double>(dof.limits.y)) {
            diagnostics.status =
                SurgicalPSMCommandMapStatus::physicalLimitViolation;
            diagnostics.rejectedLogicalIndex =
                static_cast<std::uint32_t>(
                    index < kSurgicalPSMArmDofCount
                        ? index
                        : kSurgicalPSMLogicalJawApertureIndex
                );
            diagnostics.rejectedPhysicalIndex =
                static_cast<std::uint32_t>(index);
            return diagnostics;
        }
    }

    physicalTargets = std::move(candidate);
    return diagnostics;
}

SurgicalPSMCommandMapDiagnostics
expandSurgicalPSMActuatorPositionTargets(
    const EngineModel& model,
    const std::span<const double> actuatorTargets,
    std::vector<double>& physicalTargets
) {
    SurgicalPSMCommandMapDiagnostics diagnostics;
    if (actuatorTargets.size() !=
        kSurgicalPSMLogicalPositionTargetCount) {
        diagnostics.status =
            SurgicalPSMCommandMapStatus::invalidDimensions;
        return diagnostics;
    }
    for (std::size_t index = 0u;
         index < actuatorTargets.size();
         ++index) {
        if (!std::isfinite(actuatorTargets[index])) {
            diagnostics.status =
                SurgicalPSMCommandMapStatus::nonfiniteTarget;
            diagnostics.rejectedLogicalIndex =
                static_cast<std::uint32_t>(index);
            return diagnostics;
        }
    }

    std::array<double,
        kSurgicalPSMLogicalPositionTargetCount> logical{};
    std::copy_n(actuatorTargets.begin(), 3u, logical.begin());
    std::array<double, kSurgicalPSMToolActuatorCount> tool{};
    for (std::size_t row = 0u;
         row < kSurgicalPSMToolActuatorCount;
         ++row) {
        for (std::size_t column = 0u;
             column < kSurgicalPSMToolActuatorCount;
             ++column) {
            tool[row] +=
                kActuatorToJointPosition[
                    row * kSurgicalPSMToolActuatorCount + column
                ] * actuatorTargets[3u + column];
        }
    }
    logical[3] = tool[0];
    logical[4] = tool[1];
    logical[5] = tool[2];
    logical[kSurgicalPSMLogicalJawApertureIndex] = tool[3];
    return expandSurgicalPSMLogicalPositionTargets(
        model,
        logical,
        physicalTargets
    );
}

SurgicalPSMCommandMapDiagnostics
expandSurgicalPSMActuatorEfforts(
    const EngineModel& model,
    const std::span<const double> actuatorEfforts,
    std::vector<double>& generalizedEfforts
) {
    SurgicalPSMCommandMapDiagnostics diagnostics;
    if (model.dofs.size() != kSurgicalPSMJointCount) {
        diagnostics.status = SurgicalPSMCommandMapStatus::invalidModel;
        return diagnostics;
    }
    if (actuatorEfforts.size() !=
        kSurgicalPSMLogicalPositionTargetCount) {
        diagnostics.status =
            SurgicalPSMCommandMapStatus::invalidDimensions;
        return diagnostics;
    }
    for (std::size_t index = 0u;
         index < actuatorEfforts.size();
         ++index) {
        if (!std::isfinite(actuatorEfforts[index])) {
            diagnostics.status =
                SurgicalPSMCommandMapStatus::nonfiniteTarget;
            diagnostics.rejectedLogicalIndex =
                static_cast<std::uint32_t>(index);
            return diagnostics;
        }
    }

    std::vector<double> candidate(kSurgicalPSMJointCount, 0.0);
    std::copy_n(actuatorEfforts.begin(), 3u, candidate.begin());
    std::array<double, kSurgicalPSMToolActuatorCount> tool{};
    for (std::size_t row = 0u;
         row < kSurgicalPSMToolActuatorCount;
         ++row) {
        for (std::size_t column = 0u;
             column < kSurgicalPSMToolActuatorCount;
             ++column) {
            tool[row] +=
                kActuatorToJointEffort[
                    row * kSurgicalPSMToolActuatorCount + column
                ] * actuatorEfforts[3u + column];
        }
    }
    candidate[3] = tool[0];
    candidate[4] = tool[1];
    candidate[5] = tool[2];
    candidate[6] = -tool[3];
    candidate[7] = tool[3];
    for (std::size_t index = 0u;
         index < candidate.size();
         ++index) {
        const double limit = model.dofs[index].limits.w;
        if (!std::isfinite(limit) || limit < 0.0 ||
            std::abs(candidate[index]) > limit) {
            diagnostics.status =
                SurgicalPSMCommandMapStatus::physicalEffortLimitViolation;
            diagnostics.rejectedLogicalIndex =
                index < 3u
                ? static_cast<std::uint32_t>(index)
                : MR_INVALID_INDEX;
            diagnostics.rejectedPhysicalIndex =
                static_cast<std::uint32_t>(index);
            return diagnostics;
        }
    }
    generalizedEfforts = std::move(candidate);
    return diagnostics;
}

std::array<float, kSurgicalPSMLogicalPositionTargetCount>
surgicalPSMDefaultLogicalPositionTargets() noexcept {
    return {
        kDefaultQ[0],
        kDefaultQ[1],
        kDefaultQ[2],
        kDefaultQ[3],
        kDefaultQ[4],
        kDefaultQ[5],
        kDefaultQ[7] - kDefaultQ[6],
    };
}

const char* surgicalPSMCommandMapStatusName(
    const SurgicalPSMCommandMapStatus status
) noexcept {
    switch (status) {
    case SurgicalPSMCommandMapStatus::success:
        return "success";
    case SurgicalPSMCommandMapStatus::invalidModel:
        return "invalid_model";
    case SurgicalPSMCommandMapStatus::invalidDimensions:
        return "invalid_dimensions";
    case SurgicalPSMCommandMapStatus::nonfiniteTarget:
        return "nonfinite_target";
    case SurgicalPSMCommandMapStatus::negativeJawAperture:
        return "negative_jaw_aperture";
    case SurgicalPSMCommandMapStatus::physicalLimitViolation:
        return "physical_limit_violation";
    case SurgicalPSMCommandMapStatus::physicalEffortLimitViolation:
        return "physical_effort_limit_violation";
    }
    return "unknown";
}

} // namespace metalrobo
