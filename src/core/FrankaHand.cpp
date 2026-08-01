#include "metalrobo/Franka.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace metalrobo {
namespace {

constexpr float kPi = 3.14159265358979323846f;

mr_float4 f4(
    const float x,
    const float y,
    const float z,
    const float w = 0.0f
) {
    return {x, y, z, w};
}

mr_float4 quaternionFromRpy(
    const float roll,
    const float pitch,
    const float yaw
) {
    const float halfRoll = 0.5f * roll;
    const float halfPitch = 0.5f * pitch;
    const float halfYaw = 0.5f * yaw;
    const float cr = std::cos(halfRoll);
    const float sr = std::sin(halfRoll);
    const float cp = std::cos(halfPitch);
    const float sp = std::sin(halfPitch);
    const float cy = std::cos(halfYaw);
    const float sy = std::sin(halfYaw);
    return {
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    };
}

std::array<mr_float4, 3> invertSymmetric(
    const mr_float4 row0,
    const mr_float4 row1,
    const mr_float4 row2
) {
    const double a = row0.x;
    const double b = row0.y;
    const double c = row0.z;
    const double d = row1.y;
    const double e = row1.z;
    const double f = row2.z;
    const double determinant =
        a * (d * f - e * e) -
        b * (b * f - c * e) +
        c * (b * e - c * d);
    if (!(determinant > 0.0) || !std::isfinite(determinant)) {
        throw std::logic_error(
            "pinned Franka Hand inertia is not positive definite"
        );
    }
    return {{
        f4(
            static_cast<float>((d * f - e * e) / determinant),
            static_cast<float>((c * e - b * f) / determinant),
            static_cast<float>((b * e - c * d) / determinant)
        ),
        f4(
            static_cast<float>((c * e - b * f) / determinant),
            static_cast<float>((a * f - c * c) / determinant),
            static_cast<float>((b * c - a * e) / determinant)
        ),
        f4(
            static_cast<float>((b * e - c * d) / determinant),
            static_cast<float>((b * c - a * e) / determinant),
            static_cast<float>((a * d - b * b) / determinant)
        ),
    }};
}

MRBodyPropertiesGPU handBody(
    const std::uint32_t parentBody,
    const std::uint32_t inboundJoint,
    const float mass,
    const mr_float4 centerOfMass,
    const mr_float4 inertiaRow0,
    const mr_float4 inertiaRow1,
    const mr_float4 inertiaRow2
) {
    const auto inverse = invertSymmetric(
        inertiaRow0,
        inertiaRow1,
        inertiaRow2
    );
    MRBodyPropertiesGPU body{};
    body.articulationIndex = 0u;
    body.parentBody = parentBody;
    body.inboundJoint = inboundJoint;
    body.motionType = MR_MOTION_DYNAMIC;
    body.massAndInverseMass = {mass, 1.0f / mass, 0.0f, 0.0f};
    body.centerOfMass = centerOfMass;
    body.inertiaRow0 = inertiaRow0;
    body.inertiaRow1 = inertiaRow1;
    body.inertiaRow2 = inertiaRow2;
    body.inverseInertiaRow0 = inverse[0];
    body.inverseInertiaRow1 = inverse[1];
    body.inverseInertiaRow2 = inverse[2];
    body.dampingAndSpeedLimits =
        {0.0f, 0.0f, 1.0e6f, 1.0e6f};
    return body;
}

MRJointDescriptorGPU fixedHandJoint(
    const std::uint32_t parentBody,
    const std::uint32_t childBody,
    const mr_float4 parentCenterOfMass,
    const mr_float4 childCenterOfMass
) {
    MRJointDescriptorGPU joint{};
    joint.parentBody = parentBody;
    joint.childBody = childBody;
    joint.jointType = MR_JOINT_FIXED;
    joint.qOffset = 7u;
    joint.nq = 0u;
    joint.vOffset = 7u;
    joint.nv = 0u;
    joint.parentAnchor = {
        -parentCenterOfMass.x,
        -parentCenterOfMass.y,
        0.107f - parentCenterOfMass.z,
        0.0f,
    };
    joint.childAnchor = {
        -childCenterOfMass.x,
        -childCenterOfMass.y,
        -childCenterOfMass.z,
        0.0f,
    };
    joint.parentRotation = quaternionFromRpy(
        0.0f,
        0.0f,
        -0.25f * kPi
    );
    joint.childRotation.w = 1.0f;
    return joint;
}

MRJointDescriptorGPU fingerJoint(
    const std::uint32_t childBody,
    const std::uint32_t qOffset,
    const std::uint32_t vOffset,
    const mr_float4 handCenterOfMass,
    const mr_float4 fingerCenterOfMass,
    const bool mirrored
) {
    MRJointDescriptorGPU joint{};
    joint.parentBody = 8u;
    joint.childBody = childBody;
    joint.jointType = MR_JOINT_PRISMATIC;
    joint.qOffset = qOffset;
    joint.nq = 1u;
    joint.vOffset = vOffset;
    joint.nv = 1u;
    joint.axis0 = {0.0f, 1.0f, 0.0f, 0.0f};
    joint.parentAnchor = {
        -handCenterOfMass.x,
        -handCenterOfMass.y,
        0.0584f - handCenterOfMass.z,
        0.0f,
    };
    joint.childAnchor = {
        -fingerCenterOfMass.x,
        -fingerCenterOfMass.y,
        -fingerCenterOfMass.z,
        0.0f,
    };
    joint.parentRotation = mirrored
        ? quaternionFromRpy(0.0f, 0.0f, kPi)
        : f4(0.0f, 0.0f, 0.0f, 1.0f);
    joint.childRotation.w = 1.0f;
    return joint;
}

MRDofPropertiesGPU fingerDof(
    const std::uint32_t jointIndex,
    const std::uint32_t coordinate
) {
    MRDofPropertiesGPU dof{};
    dof.articulationIndex = 0u;
    dof.jointIndex = jointIndex;
    dof.qIndex = coordinate;
    dof.vIndex = coordinate;
    dof.localDof = 0u;
    dof.flags =
        MR_DOF_FLAG_ACTUATED |
        MR_DOF_FLAG_POSITION_LIMIT |
        MR_DOF_FLAG_VELOCITY_LIMIT |
        MR_DOF_FLAG_EFFORT_LIMIT |
        MR_DOF_FLAG_DRIVE;
    dof.limits = {0.0f, 0.04f, 0.2f, 100.0f};
    dof.drive = {1200.0f, 30.0f, 0.0f, 0.0f};
    return dof;
}

void appendCapsule(
    EngineModel& model,
    const std::uint32_t body,
    const mr_float4 centerOfMass,
    const float z,
    const float radius,
    const float halfLength
) {
    MRShapeGPU shape{};
    shape.bodyIndex = body;
    shape.shapeType = MR_SHAPE_CAPSULE;
    shape.materialIndex = 0u;
    // Robot shapes occupy a dedicated group and only collide with the scene
    // group. This first executable hand profile intentionally omits robot
    // self-collision; link-pair filtering can be authored separately later.
    shape.collisionGroup = 2u;
    shape.collisionMask = 1u;
    shape.slotGeneration =
        static_cast<std::uint32_t>(model.shapes.size() + 1u);
    shape.localPosition = {
        -centerOfMass.x,
        -centerOfMass.y,
        z - centerOfMass.z,
        1.0f,
    };
    shape.localRotation.w = 1.0f;
    shape.dimensions = {radius, halfLength, 0.0f, 0.0f};
    shape.contactRestAndBoundingRadius =
        {0.001f, 0.0f, radius + halfLength, 0.0f};
    model.shapes.push_back(shape);
}

void appendFingerBox(
    EngineModel& model,
    const std::uint32_t body,
    const mr_float4 centerOfMass,
    const mr_float4 center,
    const mr_float4 halfExtents,
    const mr_float4 rotation,
    const std::uint32_t material
) {
    MRShapeGPU shape{};
    shape.bodyIndex = body;
    shape.shapeType = MR_SHAPE_BOX;
    shape.materialIndex = material;
    shape.collisionGroup = 2u;
    shape.collisionMask = 1u;
    shape.slotGeneration =
        static_cast<std::uint32_t>(model.shapes.size() + 1u);
    shape.localPosition = {
        center.x - centerOfMass.x,
        center.y - centerOfMass.y,
        center.z - centerOfMass.z,
        1.0f,
    };
    shape.localRotation = rotation;
    shape.dimensions = halfExtents;
    shape.contactRestAndBoundingRadius = {
        0.001f,
        0.0f,
        std::sqrt(
            halfExtents.x * halfExtents.x +
            halfExtents.y * halfExtents.y +
            halfExtents.z * halfExtents.z
        ),
        0.0f,
    };
    model.shapes.push_back(shape);
}

} // namespace

EngineModel makeFrankaPandaHandEngineModel() {
    EngineModel model = makeFrankaPandaEngineModel();
    model.name = "franka_emika_panda_fer_hand_abi_v2";
    for (MRShapeGPU& shape : model.shapes) {
        shape.collisionGroup = 2u;
        shape.collisionMask = 1u;
    }

    constexpr mr_float4 handCom{
        -0.0000376f,
        0.0119128f,
        0.0207260f,
        0.0f,
    };
    constexpr mr_float4 fingerCom{
        0.0f,
        0.0152850f,
        0.0219675f,
        0.0f,
    };
    const mr_float4 link7Com = model.bodies[7u].centerOfMass;
    model.bodies.push_back(handBody(
        7u,
        7u,
        0.6544f,
        handCom,
        {0.00186f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.00030f, -0.00002f, 0.0f},
        {0.0f, -0.00002f, 0.00174f, 0.0f}
    ));
    for (const std::uint32_t child : {9u, 10u}) {
        model.bodies.push_back(handBody(
            8u,
            child == 9u ? 8u : 9u,
            0.0291f,
            fingerCom,
            {0.00000849f, 0.0f, 0.0f, 0.0f},
            {0.0f, 0.00000853f, -0.00000106f, 0.0f},
            {0.0f, -0.00000106f, 0.00000177f, 0.0f}
        ));
    }

    model.joints.push_back(fixedHandJoint(
        7u,
        8u,
        link7Com,
        handCom
    ));
    model.joints.push_back(fingerJoint(
        9u,
        7u,
        7u,
        handCom,
        fingerCom,
        false
    ));
    model.joints.push_back(fingerJoint(
        10u,
        8u,
        8u,
        handCom,
        fingerCom,
        true
    ));
    model.dofs.push_back(fingerDof(8u, 7u));
    model.dofs.push_back(fingerDof(9u, 8u));

    MRMaterialGPU rubber = model.materials.front();
    rubber.friction = {1.13f, 1.13f, 0.0f, 0.0f};
    model.materials.push_back(rubber);

    appendCapsule(model, 8u, handCom, 0.04f, 0.04f, 0.05f);
    appendCapsule(model, 8u, handCom, 0.10f, 0.02f, 0.05f);
    constexpr std::array<mr_float4, 4> centers{{
        {0.0f, 0.0185f, 0.011f, 0.0f},
        {0.0f, 0.0068f, 0.0022f, 0.0f},
        {0.0f, 0.0159f, 0.02835f, 0.0f},
        {0.0f, 0.00758f, 0.04525f, 0.0f},
    }};
    constexpr std::array<mr_float4, 4> halfExtents{{
        {0.011f, 0.0075f, 0.010f, 0.0f},
        {0.011f, 0.0044f, 0.0019f, 0.0f},
        {0.00875f, 0.0035f, 0.01175f, 0.0f},
        {0.00875f, 0.0076f, 0.00925f, 0.0f},
    }};
    for (std::uint32_t finger = 0u; finger < 2u; ++finger) {
        const std::uint32_t body = 9u + finger;
        for (std::uint32_t box = 0u; box < centers.size(); ++box) {
            mr_float4 rotation{0.0f, 0.0f, 0.0f, 1.0f};
            if (box == 2u) {
                rotation = finger == 0u
                    ? quaternionFromRpy(kPi / 6.0f, 0.0f, 0.0f)
                    : quaternionFromRpy(-kPi / 6.0f, 0.0f, kPi);
            }
            appendFingerBox(
                model,
                body,
                fingerCom,
                centers[box],
                halfExtents[box],
                rotation,
                box == 3u ? 1u : 0u
            );
        }
    }

    MRArticulationGPU& articulation = model.articulations[0u];
    articulation.bodyCount = 11u;
    articulation.jointCount = 10u;
    articulation.nq = 9u;
    articulation.nv = 9u;
    model.world.bodyCount =
        static_cast<std::uint32_t>(model.bodies.size());
    model.world.jointCount =
        static_cast<std::uint32_t>(model.joints.size());
    model.world.shapeCount =
        static_cast<std::uint32_t>(model.shapes.size());
    model.world.materialCount =
        static_cast<std::uint32_t>(model.materials.size());
    model.world.nq = 9u;
    model.world.nv = 9u;
    model.bodyNames = {
        "panda_link0",
        "panda_link1",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
        "panda_link7",
        "panda_hand",
        "panda_leftfinger",
        "panda_rightfinger",
    };
    model.jointNames.insert(
        model.jointNames.end(),
        {
            "panda_joint8",
            "panda_finger_joint1",
            "panda_finger_joint2",
        }
    );
    model.dofNames.insert(
        model.dofNames.end(),
        {
            "panda_finger_joint1",
            "panda_finger_joint2",
        }
    );
    model.defaultQ.push_back(0.035f);
    model.defaultQ.push_back(0.035f);
    model.defaultV.push_back(0.0f);
    model.defaultV.push_back(0.0f);

    std::string reason;
    if (!model.valid(&reason)) {
        throw std::logic_error(
            "Franka Hand model compilation failed: " + reason
        );
    }
    return model;
}

} // namespace metalrobo
