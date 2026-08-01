#include "metalrobo/Franka.hpp"

#include "metalrobo/Model.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace metalrobo {
namespace {

// Exact joint1..joint7 `friction` records from robots/fer/dynamics.yaml at
// franka_description commit 02afaae282d4a8e10d7d2f781b23b3515c303ce5.
constexpr std::array<float, kFrankaPandaJointCount>
    kFrankaJointDryFriction{{
        0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f,
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

std::array<mr_float4, 3> inverseInertia(const MRLinkGPU& link) {
    const double a = link.inertiaRow0.x;
    const double b = link.inertiaRow0.y;
    const double c = link.inertiaRow0.z;
    const double d = link.inertiaRow1.y;
    const double e = link.inertiaRow1.z;
    const double f = link.inertiaRow2.z;
    const double determinant =
        a * (d * f - e * e) -
        b * (b * f - c * e) +
        c * (b * e - c * d);
    if (!(determinant > 0.0) || !std::isfinite(determinant)) {
        throw std::logic_error(
            "pinned Franka inertia is not positive definite"
        );
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

MRBodyPropertiesGPU compileBody(
    const Model& source,
    const std::size_t bodyIndex
) {
    const MRLinkGPU& link = source.links[bodyIndex];
    MRBodyPropertiesGPU body{};
    body.articulationIndex = 0u;
    body.parentBody = bodyIndex == 0u
        ? MR_INVALID_INDEX
        : static_cast<mr_u32>(bodyIndex - 1u);
    body.inboundJoint = bodyIndex == 0u
        ? MR_INVALID_INDEX
        : static_cast<mr_u32>(bodyIndex - 1u);
    body.motionType = MR_MOTION_DYNAMIC;
    const double mass = link.massAndCOMX.x;
    body.massAndInverseMass = f4(mass, 1.0 / mass, 0.0, 0.0);
    body.centerOfMass = f4(
        link.massAndCOMX.y,
        link.massAndCOMX.z,
        link.massAndCOMX.w
    );
    body.inertiaRow0 = link.inertiaRow0;
    body.inertiaRow1 = link.inertiaRow1;
    body.inertiaRow2 = link.inertiaRow2;
    const std::array<mr_float4, 3> inverse =
        inverseInertia(link);
    body.inverseInertiaRow0 = inverse[0];
    body.inverseInertiaRow1 = inverse[1];
    body.inverseInertiaRow2 = inverse[2];
    body.dampingAndSpeedLimits = f4(0.0, 0.0, 1.0e6, 1.0e6);
    return body;
}

MRJointDescriptorGPU compileJoint(
    const Model& source,
    const std::size_t jointIndex
) {
    const MRJointGPU& legacy = source.joints[jointIndex];
    const MRLinkGPU& parent = source.links[jointIndex];
    const MRLinkGPU& child = source.links[jointIndex + 1u];
    MRJointDescriptorGPU joint{};
    joint.parentBody = static_cast<mr_u32>(jointIndex);
    joint.childBody = static_cast<mr_u32>(jointIndex + 1u);
    joint.jointType = MR_JOINT_REVOLUTE;
    joint.qOffset = static_cast<mr_u32>(jointIndex);
    joint.nq = 1u;
    joint.vOffset = static_cast<mr_u32>(jointIndex);
    joint.nv = 1u;
    joint.axis0 = legacy.axis;
    // The pinned records use URDF link-frame origins. Generic runtime state
    // is centred at each body's COM, so both sides of the joint are shifted.
    joint.parentAnchor = f4(
        legacy.parentOffset.x - parent.massAndCOMX.y,
        legacy.parentOffset.y - parent.massAndCOMX.z,
        legacy.parentOffset.z - parent.massAndCOMX.w
    );
    joint.childAnchor = f4(
        -child.massAndCOMX.y,
        -child.massAndCOMX.z,
        -child.massAndCOMX.w
    );
    joint.parentRotation = legacy.parentRotation;
    joint.childRotation = f4(0.0, 0.0, 0.0, 1.0);
    return joint;
}

MRDofPropertiesGPU compileDof(
    const Model& source,
    const std::size_t dofIndex
) {
    const MRJointGPU& legacy = source.joints[dofIndex];
    MRDofPropertiesGPU dof{};
    dof.articulationIndex = 0u;
    dof.jointIndex = static_cast<mr_u32>(dofIndex);
    dof.qIndex = static_cast<mr_u32>(dofIndex);
    dof.vIndex = static_cast<mr_u32>(dofIndex);
    dof.localDof = 0u;
    dof.flags =
        MR_DOF_FLAG_ACTUATED |
        MR_DOF_FLAG_POSITION_LIMIT |
        MR_DOF_FLAG_VELOCITY_LIMIT |
        MR_DOF_FLAG_EFFORT_LIMIT |
        MR_DOF_FLAG_DRIVE;
    dof.limits = legacy.limits;
    // x/y retain MetalRobo's established Franka model-PD preset. z is the
    // official reflected rotor inertia already compiled by the legacy model.
    // w is the pinned FER Coulomb-friction value from dynamics.yaml.
    dof.drive = f4(
        legacy.drive.x,
        legacy.drive.y,
        legacy.drive.w,
        kFrankaJointDryFriction[dofIndex]
    );
    return dof;
}

MRShapeGPU compileShape(
    const Model& source,
    const MRColliderGPU& collider
) {
    const std::size_t bodyIndex =
        static_cast<std::size_t>(collider.linkIndex);
    const MRLinkGPU& body = source.links[bodyIndex];
    MRShapeGPU shape{};
    shape.bodyIndex = static_cast<mr_u32>(bodyIndex);
    shape.shapeType = collider.shapeType;
    shape.materialIndex = 0u;
    shape.collisionGroup = collider.collisionGroup;
    shape.collisionMask = collider.collisionMask;
    shape.slotGeneration = 1u;
    shape.localPosition = f4(
        collider.centerAndRadius.x - body.massAndCOMX.y,
        collider.centerAndRadius.y - body.massAndCOMX.z,
        collider.centerAndRadius.z - body.massAndCOMX.w,
        1.0
    );
    shape.localRotation = f4(0.0, 0.0, 0.0, 1.0);
    shape.dimensions = f4(collider.centerAndRadius.w, 0.0, 0.0, 0.0);
    shape.contactRestAndBoundingRadius =
        f4(0.0, 0.0, collider.centerAndRadius.w, 0.0);
    return shape;
}

} // namespace

EngineModel makeFrankaPandaEngineModel() {
    const Model source = makeFrankaPandaModel();
    if (source.links.size() != kFrankaPandaBodyCount ||
        source.joints.size() != kFrankaPandaJointCount ||
        source.colliders.size() != kFrankaPandaShapeCount ||
        source.homePosition.size() != kFrankaPandaJointCount) {
        throw std::logic_error(
            "legacy Franka records disagree with canonical ABI counts"
        );
    }

    EngineModel model;
    model.name = "franka_emika_panda_fer_abi_v2";
    model.world.abiVersion = MR_ENGINE_ABI_VERSION;
    model.world.bodyCount =
        static_cast<mr_u32>(kFrankaPandaBodyCount);
    model.world.articulationCount = 1u;
    model.world.jointCount =
        static_cast<mr_u32>(kFrankaPandaJointCount);
    model.world.shapeCount =
        static_cast<mr_u32>(kFrankaPandaShapeCount);
    model.world.materialCount = 1u;
    model.world.nq =
        static_cast<mr_u32>(kFrankaPandaJointCount);
    model.world.nv =
        static_cast<mr_u32>(kFrankaPandaJointCount);
    model.world.pairCapacity = 512u;
    model.world.contactCapacity = 512u;
    model.world.constraintCapacity = 1024u;
    model.world.islandCapacity = 8u;
    model.world.solverType = MR_SOLVER_LEGACY_PROJECTED;
    model.world.frictionConeType = MR_FRICTION_CONE_ELLIPTIC;
    const double substeps = source.gpu.substeps > 0u
        ? static_cast<double>(source.gpu.substeps)
        : 1.0;
    model.world.gravityAndTimestep = f4(
        source.gpu.gravityAndTimestep.x,
        source.gpu.gravityAndTimestep.y,
        source.gpu.gravityAndTimestep.z,
        static_cast<double>(source.gpu.gravityAndTimestep.w) /
            substeps
    );
    model.world.solverScales =
        f4(1.0e-7, 1.0e-9, 2.0, 1.0e-4);

    MRArticulationGPU articulation{};
    articulation.rootBody = 0u;
    articulation.rootType = MR_ROOT_FIXED;
    articulation.firstBody = 0u;
    articulation.bodyCount =
        static_cast<mr_u32>(kFrankaPandaBodyCount);
    articulation.firstJoint = 0u;
    articulation.jointCount =
        static_cast<mr_u32>(kFrankaPandaJointCount);
    articulation.qOffset = 0u;
    articulation.nq =
        static_cast<mr_u32>(kFrankaPandaJointCount);
    articulation.vOffset = 0u;
    articulation.nv =
        static_cast<mr_u32>(kFrankaPandaJointCount);
    articulation.solverGroup = 0u;
    model.articulations.push_back(articulation);

    model.bodies.reserve(kFrankaPandaBodyCount);
    for (std::size_t body = 0u;
         body < kFrankaPandaBodyCount;
         ++body) {
        model.bodies.push_back(compileBody(source, body));
    }

    model.joints.reserve(kFrankaPandaJointCount);
    model.dofs.reserve(kFrankaPandaJointCount);
    for (std::size_t joint = 0u;
         joint < kFrankaPandaJointCount;
         ++joint) {
        model.joints.push_back(compileJoint(source, joint));
        model.dofs.push_back(compileDof(source, joint));
    }

    MRMaterialGPU material{};
    const mr_float4 legacyMaterial =
        source.colliders.front().material;
    material.friction = f4(
        legacyMaterial.x,
        legacyMaterial.x,
        0.0,
        0.0
    );
    material.response = f4(
        legacyMaterial.y,
        0.2,
        legacyMaterial.z > 0.0f
            ? 1.0 / static_cast<double>(legacyMaterial.z)
            : 0.0,
        // The legacy penalty damping coefficient is not the ABI-v2
        // dissipation coefficient. Keep the executable material explicit
        // instead of silently assigning incompatible units.
        0.0
    );
    material.geometry = f4(0.0, 0.0, 0.0, 0.0);
    model.materials.push_back(material);

    model.shapes.reserve(kFrankaPandaShapeCount);
    for (const MRColliderGPU& collider : source.colliders) {
        model.shapes.push_back(compileShape(source, collider));
    }

    model.defaultQ = source.homePosition;
    model.defaultV.assign(kFrankaPandaJointCount, 0.0f);
    model.bodyNames = {
        "panda_link0",
        "panda_link1",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
        "panda_link7",
    };
    model.jointNames = {
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    };
    model.dofNames = model.jointNames;

    std::string reason;
    if (!model.valid(&reason)) {
        throw std::logic_error(
            "internal canonical Franka model is invalid: " + reason
        );
    }
    return model;
}

} // namespace metalrobo
