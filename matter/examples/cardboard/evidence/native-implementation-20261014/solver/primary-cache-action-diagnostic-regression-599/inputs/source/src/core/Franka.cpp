#include "metalrobo/Model.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

namespace metalrobo {
namespace {

constexpr float kPi = 3.14159265358979323846f;

constexpr mr_float4 f4(
    const float x,
    const float y,
    const float z,
    const float w = 0.0f
) {
    return {x, y, z, w};
}

mr_float4 quaternionFromRPY(
    const float roll,
    const float pitch,
    const float yaw
) {
    const float cr = std::cos(0.5f * roll);
    const float sr = std::sin(0.5f * roll);
    const float cp = std::cos(0.5f * pitch);
    const float sp = std::sin(0.5f * pitch);
    const float cy = std::cos(0.5f * yaw);
    const float sy = std::sin(0.5f * yaw);
    return f4(
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy
    );
}

MRLinkGPU link(
    const float mass,
    const std::array<float, 3>& com,
    const std::array<float, 6>& inertia
) {
    MRLinkGPU result{};
    result.massAndCOMX = f4(mass, com[0], com[1], com[2]);
    result.inertiaRow0 = f4(inertia[0], inertia[1], inertia[2]);
    result.inertiaRow1 = f4(inertia[1], inertia[3], inertia[4]);
    result.inertiaRow2 = f4(inertia[2], inertia[4], inertia[5]);
    return result;
}

void addSphere(
    Model& model,
    const std::int32_t linkIndex,
    const float x,
    const float y,
    const float z,
    const float radius
) {
    MRColliderGPU collider{};
    collider.linkIndex = linkIndex;
    collider.shapeType = MR_SHAPE_SPHERE;
    collider.collisionGroup = 1u;
    collider.collisionMask = 1u;
    collider.centerAndRadius = f4(x, y, z, radius);
    collider.extent = f4(0.0f, 0.0f, 0.0f, 0.0f);
    // Polycarbonate/metal-like contact against a rigid floor. The compliance
    // is intentional: penalty contact stays stable at the 240 Hz substep.
    collider.material = f4(0.7f, 0.02f, 18000.0f, 180.0f);
    model.colliders.push_back(collider);
}

} // namespace

Model makeFrankaPandaModel() {
    Model model;
    model.name = "Franka Emika Panda";
    model.gpu.dofCount = 7u;
    model.gpu.linkCount = 8u;
    model.gpu.observationCount = 20u;
    model.gpu.actionCount = 7u;
    model.gpu.episodeHorizon = 600u;
    model.gpu.substeps = 4u;
    model.gpu.flags = 0u;
    model.gpu.gravityAndTimestep = f4(0.0f, 0.0f, -9.81f, 1.0f / 60.0f);
    // FER link geometry reaches slightly below its mounting frame.
    model.gpu.groundPlane = f4(0.0f, 0.0f, 1.0f, -0.10f);
    model.gpu.targetLowerAndRadius = f4(0.20f, -0.45f, 0.15f, 0.05f);
    model.gpu.targetUpperAndBonus = f4(0.75f, 0.45f, 0.80f, 5.0f);
    model.gpu.rewardScales = f4(4.0f, 0.01f, 0.001f, 0.002f);

    // Source: Franka Robotics' franka_description 2.8.1 FER model,
    // commit 02afaae282d4a8e10d7d2f781b23b3515c303ce5,
    // robots/fer/{inertials,kinematics,joint_limits,dynamics}.yaml.
    // Inertia order is ixx, ixy, ixz, iyy, iyz, izz, about each COM.
    model.links = {
        link(
            2.3966f,
            {-0.0172f, 0.0004f, 0.0745f},
            {0.0090f, 0.0f, 0.0020f, 0.0115f, 0.0f, 0.0085f}
        ),
        link(
            2.7907f,
            {0.00033f, -0.02204f, -0.04762f},
            {0.01564782655f, 0.00000531236f, -0.00003676721f,
             0.01439883526f, -0.00248480843f, 0.00500443991f}
        ),
        link(
            2.5420f,
            {0.00038f, -0.09211f, 0.01908f},
            {0.01662427728f, 0.00003086077f, 0.00000835940f,
             0.00462501261f, 0.00355899830f, 0.01545124526f}
        ),
        link(
            2.2513f,
            {0.05152f, 0.01696f, -0.02971f},
            {0.00631741992f, 0.00097309955f, 0.00292525834f,
             0.00866285493f, 0.00093469224f, 0.00657804051f}
        ),
        link(
            2.2037f,
            {-0.05113f, 0.05825f, 0.01698f},
            {0.00784585566f, -0.00339591957f, 0.00158704074f,
             0.00649543136f, -0.00181477884f, 0.01003079917f}
        ),
        link(
            2.2855f,
            {-0.00005f, 0.03730f, -0.09280f},
            {0.02297014781f, -0.00000949345f, -0.00002063156f,
             0.02095060919f, 0.00382345782f, 0.00430606551f}
        ),
        link(
            1.3530f,
            {0.06572f, -0.00371f, 0.00153f},
            {0.00087964522f, -0.00021487814f, -0.00011911662f,
             0.00277796968f, 0.00001274322f, 0.00286701969f}
        ),
        link(
            0.35973f,
            {0.00089f, -0.00044f, 0.05491f},
            {0.00019541063f, 0.00000165231f, 0.00000148826f,
             0.00019210361f, -0.00000131132f, 0.00017936256f}
        ),
    };

    constexpr std::array<std::array<float, 4>, 7> limits{{
        {-2.8973f, 2.8973f, 2.1750f, 87.0f},
        {-1.7628f, 1.7628f, 2.1750f, 87.0f},
        {-2.8973f, 2.8973f, 2.1750f, 87.0f},
        {-3.0718f, -0.0698f, 2.1750f, 87.0f},
        {-2.8973f, 2.8973f, 2.6100f, 12.0f},
        {-0.0175f, 3.7525f, 2.6100f, 12.0f},
        {-2.8973f, 2.8973f, 2.6100f, 12.0f},
    }};
    constexpr std::array<std::array<float, 3>, 7> offsets{{
        {0.0f, 0.0f, 0.333f},
        {0.0f, 0.0f, 0.0f},
        {0.0f, -0.316f, 0.0f},
        {0.0825f, 0.0f, 0.0f},
        {-0.0825f, 0.384f, 0.0f},
        {0.0f, 0.0f, 0.0f},
        {0.088f, 0.0f, 0.0f},
    }};
    constexpr std::array<float, 7> rolls{{
        0.0f,
        -0.5f * kPi,
        0.5f * kPi,
        0.5f * kPi,
        -0.5f * kPi,
        0.5f * kPi,
        0.5f * kPi,
    }};
    constexpr std::array<float, 7> kp{{
        600.0f, 600.0f, 600.0f, 450.0f, 180.0f, 120.0f, 70.0f,
    }};
    constexpr std::array<float, 7> kd{{
        50.0f, 50.0f, 45.0f, 35.0f, 18.0f, 12.0f, 7.0f,
    }};
    constexpr std::array<float, 7> actionScale{{
        0.35f, 0.35f, 0.35f, 0.35f, 0.30f, 0.30f, 0.35f,
    }};
    // Reflected rotor inertia: motor_inertia * gear_ratio^2.
    constexpr std::array<float, 7> armature{{
        0.6057215f, 0.6057215f, 0.4624741f, 0.4624741f,
        0.2055441f, 0.2055441f, 0.2055441f,
    }};

    model.joints.resize(7);
    for (std::size_t index = 0; index < model.joints.size(); ++index) {
        MRJointGPU& joint = model.joints[index];
        joint.parentLink = static_cast<mr_i32>(index);
        joint.childLink = static_cast<mr_u32>(index + 1);
        joint.jointType = 0u;
        joint.reserved = 0u;
        joint.axis = f4(0.0f, 0.0f, 1.0f);
        joint.parentOffset =
            f4(offsets[index][0], offsets[index][1], offsets[index][2]);
        joint.parentRotation = quaternionFromRPY(rolls[index], 0.0f, 0.0f);
        joint.limits = f4(
            limits[index][0],
            limits[index][1],
            limits[index][2],
            limits[index][3]
        );
        joint.drive =
            f4(kp[index], kd[index], actionScale[index], armature[index]);
    }

    model.homePosition = {
        0.0f,
        -0.7853981634f,
        0.0f,
        -2.3561944902f,
        0.0f,
        1.5707963268f,
        0.7853981634f,
    };

    // Sphere-only broad/contact geometry derived from the official coarse
    // capsules. Multiple spheres retain the bent-link silhouette while
    // keeping the first Metal contact kernel branch-free.
    addSphere(model, 0, -0.075f, 0.0f, 0.060f, 0.090f);
    addSphere(model, 0, -0.010f, 0.0f, 0.060f, 0.090f);
    addSphere(model, 1, 0.0f, 0.0f, -0.070f, 0.065f);
    addSphere(model, 1, 0.0f, 0.0f, -0.190f, 0.065f);
    addSphere(model, 1, 0.0f, 0.0f, -0.310f, 0.065f);
    addSphere(model, 2, 0.0f, 0.0f, -0.050f, 0.063f);
    addSphere(model, 2, 0.0f, 0.0f, 0.050f, 0.063f);
    addSphere(model, 3, 0.0f, 0.0f, -0.070f, 0.063f);
    addSphere(model, 3, 0.0f, 0.0f, -0.145f, 0.063f);
    addSphere(model, 3, 0.0f, 0.0f, -0.220f, 0.063f);
    addSphere(model, 4, 0.0f, 0.0f, -0.050f, 0.063f);
    addSphere(model, 4, 0.0f, 0.0f, 0.050f, 0.063f);
    addSphere(model, 5, 0.0f, 0.0f, -0.210f, 0.062f);
    addSphere(model, 5, 0.0f, 0.0f, -0.310f, 0.062f);
    addSphere(model, 5, 0.0f, 0.080f, -0.080f, 0.035f);
    addSphere(model, 5, 0.0f, 0.080f, -0.150f, 0.035f);
    addSphere(model, 5, 0.0f, 0.080f, -0.210f, 0.035f);
    addSphere(model, 6, 0.0f, 0.0f, -0.070f, 0.052f);
    addSphere(model, 6, 0.0f, 0.0f, 0.010f, 0.052f);
    addSphere(model, 7, 0.0f, 0.0f, 0.010f, 0.045f);
    addSphere(model, 7, 0.0f, 0.0f, 0.080f, 0.045f);
    // Fixed joint8/flange location, used as the task end-effector.
    addSphere(model, 7, 0.0f, 0.0f, 0.107f, 0.040f);
    model.gpu.colliderCount = static_cast<mr_u32>(model.colliders.size());

    std::string reason;
    if (!model.valid(&reason)) {
        throw std::logic_error("internal Franka model is invalid: " + reason);
    }
    return model;
}

} // namespace metalrobo
