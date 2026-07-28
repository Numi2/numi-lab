#include "metalrobo/Model.hpp"

#include <cmath>
#include <sstream>

namespace metalrobo {
namespace {

bool finite(const float value) {
    return std::isfinite(value);
}

bool finite(const mr_float4 value) {
    return finite(value.x) && finite(value.y) && finite(value.z) &&
        finite(value.w);
}

bool fail(std::string* reason, std::string message) {
    if (reason != nullptr) {
        *reason = std::move(message);
    }
    return false;
}

} // namespace

bool Model::valid(std::string* reason) const {
    if (reason != nullptr) {
        reason->clear();
    }
    if (gpu.dofCount == 0 || gpu.dofCount > MR_MAX_DOF) {
        return fail(reason, "dofCount must be in [1, MR_MAX_DOF].");
    }
    if (gpu.linkCount != gpu.dofCount + 1 ||
        gpu.linkCount > MR_MAX_LINKS) {
        return fail(
            reason,
            "linkCount must be dofCount + 1 and fit the GPU ABI."
        );
    }
    if (gpu.colliderCount > MR_MAX_COLLIDERS) {
        return fail(reason, "colliderCount exceeds MR_MAX_COLLIDERS.");
    }
    if (gpu.actionCount != gpu.dofCount) {
        return fail(reason, "actionCount must equal dofCount.");
    }
    if (gpu.observationCount == 0 ||
        gpu.observationCount > MR_MAX_OBSERVATIONS) {
        return fail(reason, "observationCount is outside the GPU ABI.");
    }
    if (gpu.substeps == 0 || gpu.episodeHorizon == 0) {
        return fail(reason, "substeps and episodeHorizon must be positive.");
    }
    if (joints.size() != gpu.dofCount || links.size() != gpu.linkCount ||
        colliders.size() != gpu.colliderCount ||
        homePosition.size() != gpu.dofCount) {
        return fail(reason, "CPU model arrays disagree with GPU counts.");
    }
    if (!finite(gpu.gravityAndTimestep) ||
        gpu.gravityAndTimestep.w <= 0.0f) {
        return fail(reason, "gravity and control timestep must be finite.");
    }
    if (!finite(gpu.groundPlane) ||
        gpu.groundPlane.x * gpu.groundPlane.x +
                gpu.groundPlane.y * gpu.groundPlane.y +
                gpu.groundPlane.z * gpu.groundPlane.z <=
            1.0e-12f) {
        return fail(reason, "ground plane must have a finite normal.");
    }
    if (!finite(gpu.targetLowerAndRadius) ||
        !finite(gpu.targetUpperAndBonus) ||
        !finite(gpu.rewardScales)) {
        return fail(reason, "task metadata contains a non-finite value.");
    }
    if (gpu.targetLowerAndRadius.x > gpu.targetUpperAndBonus.x ||
        gpu.targetLowerAndRadius.y > gpu.targetUpperAndBonus.y ||
        gpu.targetLowerAndRadius.z > gpu.targetUpperAndBonus.z ||
        gpu.targetLowerAndRadius.w <= 0.0f) {
        return fail(reason, "target bounds or success radius are invalid.");
    }

    for (std::size_t index = 0; index < joints.size(); ++index) {
        const MRJointGPU& joint = joints[index];
        const auto expectedChild = static_cast<mr_u32>(index + 1);
        if (joint.parentLink < 0 ||
            static_cast<mr_u32>(joint.parentLink) >= expectedChild ||
            joint.childLink != expectedChild) {
            return fail(
                reason,
                "joints must be a topologically ordered fixed-base tree."
            );
        }
        if (joint.jointType > 1u) {
            return fail(reason, "jointType must be revolute(0) or prismatic(1).");
        }
        if (!finite(joint.axis) || !finite(joint.parentOffset) ||
            !finite(joint.parentRotation) || !finite(joint.limits) ||
            !finite(joint.drive)) {
            return fail(reason, "joint contains a non-finite parameter.");
        }
        const float axisNormSquared =
            joint.axis.x * joint.axis.x + joint.axis.y * joint.axis.y +
            joint.axis.z * joint.axis.z;
        const float quaternionNormSquared =
            joint.parentRotation.x * joint.parentRotation.x +
            joint.parentRotation.y * joint.parentRotation.y +
            joint.parentRotation.z * joint.parentRotation.z +
            joint.parentRotation.w * joint.parentRotation.w;
        if (axisNormSquared < 0.999f || axisNormSquared > 1.001f) {
            return fail(reason, "joint axes must be normalized.");
        }
        if (quaternionNormSquared < 0.999f ||
            quaternionNormSquared > 1.001f) {
            return fail(reason, "joint origin quaternions must be normalized.");
        }
        if (joint.limits.x >= joint.limits.y || joint.limits.z <= 0.0f ||
            joint.limits.w <= 0.0f) {
            return fail(reason, "joint position/velocity/effort limits invalid.");
        }
        if (joint.drive.x < 0.0f || joint.drive.y < 0.0f ||
            joint.drive.z < 0.0f || joint.drive.w < 0.0f) {
            return fail(reason, "joint drive parameters must be non-negative.");
        }
        if (!finite(homePosition[index]) ||
            homePosition[index] < joint.limits.x ||
            homePosition[index] > joint.limits.y) {
            return fail(reason, "homePosition lies outside a joint limit.");
        }
    }

    for (const MRLinkGPU& link : links) {
        if (!finite(link.massAndCOMX) || !finite(link.inertiaRow0) ||
            !finite(link.inertiaRow1) || !finite(link.inertiaRow2) ||
            link.massAndCOMX.x <= 0.0f) {
            return fail(reason, "link mass/inertia contains an invalid value.");
        }
        const double ixx = link.inertiaRow0.x;
        const double ixy = link.inertiaRow0.y;
        const double ixz = link.inertiaRow0.z;
        const double iyy = link.inertiaRow1.y;
        const double iyz = link.inertiaRow1.z;
        const double izz = link.inertiaRow2.z;
        const double minor2 = ixx * iyy - ixy * ixy;
        const double determinant =
            ixx * (iyy * izz - iyz * iyz) -
            ixy * (ixy * izz - iyz * ixz) +
            ixz * (ixy * iyz - iyy * ixz);
        if (ixx <= 0.0 || minor2 <= 0.0 || determinant <= 0.0) {
            return fail(reason, "link inertia must be positive definite.");
        }
    }

    for (const MRColliderGPU& collider : colliders) {
        if (collider.linkIndex < 0 ||
            static_cast<mr_u32>(collider.linkIndex) >= gpu.linkCount) {
            return fail(reason, "collider refers to an invalid link.");
        }
        if (collider.shapeType > MR_SHAPE_BOX ||
            !finite(collider.centerAndRadius) || !finite(collider.extent) ||
            !finite(collider.material)) {
            return fail(reason, "collider contains an invalid parameter.");
        }
        if (collider.shapeType != MR_SHAPE_BOX &&
            collider.centerAndRadius.w <= 0.0f) {
            return fail(reason, "sphere/capsule radius must be positive.");
        }
        if (collider.shapeType == MR_SHAPE_BOX &&
            (collider.extent.x <= 0.0f || collider.extent.y <= 0.0f ||
             collider.extent.z <= 0.0f)) {
            return fail(reason, "box half extents must be positive.");
        }
        if (collider.material.x < 0.0f || collider.material.y < 0.0f ||
            collider.material.z < 0.0f || collider.material.w < 0.0f) {
            return fail(reason, "contact material values must be non-negative.");
        }
    }
    return true;
}

} // namespace metalrobo
