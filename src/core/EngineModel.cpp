#include "metalrobo/EngineModel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

bool fail(std::string* reason, std::string message) {
    if (reason != nullptr) {
        *reason = std::move(message);
    }
    return false;
}

bool finite(const float value) {
    return std::isfinite(value);
}

bool finite(const mr_float4 value) {
    return finite(value.x) && finite(value.y) && finite(value.z) &&
        finite(value.w);
}

double vectorNormSquared(const mr_float4 value) {
    return
        static_cast<double>(value.x) * value.x +
        static_cast<double>(value.y) * value.y +
        static_cast<double>(value.z) * value.z;
}

bool unitVector(const mr_float4 value) {
    return finite(value) &&
        std::abs(vectorNormSquared(value) - 1.0) <= 1.0e-5;
}

bool unitQuaternion(const mr_float4 value) {
    return finite(value) &&
        std::abs(
            vectorNormSquared(value) +
                static_cast<double>(value.w) * value.w -
                1.0
        ) <= 1.0e-5;
}

double determinant(const mr_float4 r0, const mr_float4 r1, const mr_float4 r2) {
    return
        static_cast<double>(r0.x) *
            (static_cast<double>(r1.y) * r2.z -
             static_cast<double>(r1.z) * r2.y) -
        static_cast<double>(r0.y) *
            (static_cast<double>(r1.x) * r2.z -
             static_cast<double>(r1.z) * r2.x) +
        static_cast<double>(r0.z) *
            (static_cast<double>(r1.x) * r2.y -
             static_cast<double>(r1.y) * r2.x);
}

bool positiveDefinite(
    const mr_float4 r0,
    const mr_float4 r1,
    const mr_float4 r2
) {
    const double firstMinor = r0.x;
    const double secondMinor =
        static_cast<double>(r0.x) * r1.y -
        static_cast<double>(r0.y) * r1.x;
    return firstMinor > 0.0 && secondMinor > 0.0 &&
        determinant(r0, r1, r2) > 0.0;
}

bool symmetric(
    const mr_float4 r0,
    const mr_float4 r1,
    const mr_float4 r2
) {
    constexpr double tolerance = 1.0e-6;
    return std::abs(static_cast<double>(r0.y) - r1.x) <= tolerance &&
        std::abs(static_cast<double>(r0.z) - r2.x) <= tolerance &&
        std::abs(static_cast<double>(r1.z) - r2.y) <= tolerance;
}

bool inverseConsistent(
    const std::array<mr_float4, 3>& matrix,
    const std::array<mr_float4, 3>& inverse
) {
    constexpr double tolerance = 2.0e-4;
    for (std::size_t row = 0; row < 3; ++row) {
        const std::array<double, 3> left{
            matrix[row].x,
            matrix[row].y,
            matrix[row].z,
        };
        for (std::size_t column = 0; column < 3; ++column) {
            const double value =
                left[0] * (&inverse[0].x)[column] +
                left[1] * (&inverse[1].x)[column] +
                left[2] * (&inverse[2].x)[column];
            const double expected = row == column ? 1.0 : 0.0;
            if (std::abs(value - expected) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

std::pair<mr_u32, mr_u32> expectedJointCoordinates(const mr_u32 type) {
    switch (type) {
    case MR_JOINT_REVOLUTE:
    case MR_JOINT_PRISMATIC:
    case MR_JOINT_CONTINUOUS:
        return {1u, 1u};
    case MR_JOINT_SPHERICAL:
        return {4u, 3u};
    case MR_JOINT_PLANAR:
        return {3u, 3u};
    case MR_JOINT_FIXED:
        return {0u, 0u};
    case MR_JOINT_FREE:
        return {7u, 6u};
    default:
        return {MR_INVALID_INDEX, MR_INVALID_INDEX};
    }
}

constexpr mr_u32 kKnownDofFlags =
    MR_DOF_FLAG_ROOT |
    MR_DOF_FLAG_ACTUATED |
    MR_DOF_FLAG_POSITION_LIMIT |
    MR_DOF_FLAG_VELOCITY_LIMIT |
    MR_DOF_FLAG_EFFORT_LIMIT |
    MR_DOF_FLAG_DRIVE;

bool zero(const mr_float4 value) {
    return value.x == 0.0f && value.y == 0.0f &&
        value.z == 0.0f && value.w == 0.0f;
}

mr_u32 expectedJointQIndex(
    const MRJointDescriptorGPU& joint,
    const mr_u32 localDof
) {
    switch (joint.jointType) {
    case MR_JOINT_REVOLUTE:
    case MR_JOINT_PRISMATIC:
    case MR_JOINT_CONTINUOUS:
    case MR_JOINT_PLANAR:
        return joint.qOffset + localDof;
    case MR_JOINT_FREE:
        return localDof < 3u
            ? joint.qOffset + localDof
            : MR_INVALID_INDEX;
    case MR_JOINT_SPHERICAL:
    case MR_JOINT_FIXED:
    default:
        return MR_INVALID_INDEX;
    }
}

bool validDofParameters(
    const MRDofPropertiesGPU& dof,
    const bool root,
    const mr_u32 jointType
) {
    if (!finite(dof.limits) || !finite(dof.drive) ||
        dof.reserved0 != 0u || dof.reserved1 != 0u ||
        (dof.flags & ~kKnownDofFlags) != 0u ||
        dof.drive.x < 0.0f || dof.drive.y < 0.0f ||
        dof.drive.z < 0.0f || dof.drive.w < 0.0f) {
        return false;
    }
    if (root) {
        return dof.flags == MR_DOF_FLAG_ROOT &&
            zero(dof.limits) && zero(dof.drive);
    }
    if ((dof.flags & MR_DOF_FLAG_ROOT) != 0u) {
        return false;
    }
    const bool actuated =
        (dof.flags & MR_DOF_FLAG_ACTUATED) != 0u;
    const bool positionLimited =
        (dof.flags & MR_DOF_FLAG_POSITION_LIMIT) != 0u;
    const bool velocityLimited =
        (dof.flags & MR_DOF_FLAG_VELOCITY_LIMIT) != 0u;
    const bool effortLimited =
        (dof.flags & MR_DOF_FLAG_EFFORT_LIMIT) != 0u;
    const bool driven =
        (dof.flags & MR_DOF_FLAG_DRIVE) != 0u;
    if ((!actuated && (effortLimited || driven)) ||
        (driven && !actuated) ||
        (!driven && (dof.drive.x != 0.0f ||
                     dof.drive.y != 0.0f)) ||
        (positionLimited &&
         (dof.qIndex == MR_INVALID_INDEX ||
          jointType == MR_JOINT_CONTINUOUS ||
          dof.limits.x > dof.limits.y)) ||
        (!positionLimited &&
         (dof.limits.x != 0.0f || dof.limits.y != 0.0f)) ||
        (velocityLimited
             ? !(dof.limits.z > 0.0f)
             : dof.limits.z != 0.0f) ||
        (effortLimited
             ? !(dof.limits.w > 0.0f)
             : dof.limits.w != 0.0f)) {
        return false;
    }
    return true;
}

bool normalizedQuaternion(
    std::span<const float> q,
    const std::size_t offset
) {
    if (offset > q.size() || q.size() - offset < 4u) {
        return false;
    }
    double normSquared = 0.0;
    for (std::size_t index = 0; index < 4; ++index) {
        if (!finite(q[offset + index])) {
            return false;
        }
        normSquared += static_cast<double>(q[offset + index]) *
            q[offset + index];
    }
    return std::abs(normSquared - 1.0) <= 1.0e-5;
}

bool rangeWithin(
    const mr_u32 offset,
    const mr_u32 count,
    const std::size_t size
) {
    const std::size_t begin = offset;
    const std::size_t length = count;
    return begin <= size && length <= size - begin;
}

mr_float4 f4(
    const float x,
    const float y,
    const float z,
    const float w
) {
    return {x, y, z, w};
}

} // namespace

bool EngineModel::valid(std::string* reason) const {
    if (reason != nullptr) {
        reason->clear();
    }
    if (world.abiVersion != MR_ENGINE_ABI_VERSION) {
        return fail(reason, "engine ABI version mismatch");
    }
    if (world.bodyCount != bodies.size() ||
        world.articulationCount != articulations.size() ||
        world.jointCount != joints.size() ||
        world.nv != dofs.size() ||
        world.shapeCount != shapes.size() ||
        world.materialCount != materials.size() ||
        world.nq != defaultQ.size() || world.nv != defaultV.size()) {
        return fail(reason, "world counts disagree with compiled arrays");
    }
    if (world.solverType > MR_SOLVER_THROUGHPUT_PGS ||
        world.frictionConeType > MR_FRICTION_CONE_PYRAMID_8) {
        return fail(reason, "unknown solver or friction-cone type");
    }
    if (!finite(world.gravityAndTimestep) ||
        !finite(world.solverScales) ||
        world.gravityAndTimestep.w <= 0.0f ||
        world.solverScales.x < 0.0f ||
        world.solverScales.y < 0.0f ||
        world.solverScales.z < 0.0f ||
        world.solverScales.w < 0.0f) {
        return fail(reason, "invalid gravity, timestep, or solver scales");
    }
    if (world.contactCapacity == 0u || world.constraintCapacity == 0u ||
        world.islandCapacity == 0u || world.pairCapacity == 0u) {
        return fail(reason, "all production capacities must be explicit");
    }
    if (!std::ranges::all_of(defaultQ, [](const float value) {
            return finite(value);
        }) ||
        !std::ranges::all_of(defaultV, [](const float value) {
            return finite(value);
        })) {
        return fail(reason, "default generalized state is non-finite");
    }

    std::vector<std::uint8_t> bodyOwner(bodies.size(), 0u);
    std::vector<std::uint8_t> qOwner(defaultQ.size(), 0u);
    std::vector<std::uint8_t> vOwner(defaultV.size(), 0u);
    std::vector<std::uint8_t> jointOwner(joints.size(), 0u);

    for (std::size_t articulationIndex = 0;
         articulationIndex < articulations.size();
         ++articulationIndex) {
        const MRArticulationGPU& articulation =
            articulations[articulationIndex];
        const std::size_t firstBody = articulation.firstBody;
        const std::size_t bodyEnd =
            firstBody + static_cast<std::size_t>(articulation.bodyCount);
        const std::size_t firstJoint = articulation.firstJoint;
        const std::size_t jointEnd =
            firstJoint + static_cast<std::size_t>(articulation.jointCount);
        const std::size_t qBegin = articulation.qOffset;
        const std::size_t qEnd =
            qBegin + static_cast<std::size_t>(articulation.nq);
        const std::size_t vBegin = articulation.vOffset;
        const std::size_t vEnd =
            vBegin + static_cast<std::size_t>(articulation.nv);
        if (articulation.rootType > MR_ROOT_FLOATING ||
            articulation.bodyCount == 0u ||
            !rangeWithin(
                articulation.firstBody,
                articulation.bodyCount,
                bodies.size()
            ) ||
            !rangeWithin(
                articulation.firstJoint,
                articulation.jointCount,
                joints.size()
            ) ||
            !rangeWithin(
                articulation.qOffset,
                articulation.nq,
                defaultQ.size()
            ) ||
            !rangeWithin(
                articulation.vOffset,
                articulation.nv,
                defaultV.size()
            ) ||
            static_cast<std::size_t>(articulation.rootBody) < firstBody ||
            static_cast<std::size_t>(articulation.rootBody) >= bodyEnd) {
            return fail(reason, "articulation range or root is invalid");
        }

        const std::size_t rootNq =
            articulation.rootType == MR_ROOT_FLOATING ? 7u : 0u;
        const std::size_t rootNv =
            articulation.rootType == MR_ROOT_FLOATING ? 6u : 0u;
        std::size_t expectedNq = rootNq;
        std::size_t expectedNv = rootNv;
        if (static_cast<std::size_t>(articulation.jointCount) + 1u !=
            static_cast<std::size_t>(articulation.bodyCount)) {
            return fail(
                reason,
                "tree articulation must have one inbound joint per non-root"
            );
        }
        if (expectedNq > qEnd - qBegin ||
            expectedNv > vEnd - vBegin) {
            return fail(reason, "articulation root coordinates exceed owner");
        }
        for (std::size_t localDof = 0u;
             localDof < rootNv;
             ++localDof) {
            const std::size_t globalV = vBegin + localDof;
            const MRDofPropertiesGPU& dof = dofs[globalV];
            const mr_u32 expectedQ =
                localDof < 3u
                    ? static_cast<mr_u32>(qBegin + localDof)
                    : MR_INVALID_INDEX;
            if (dof.articulationIndex != articulationIndex ||
                dof.jointIndex != MR_INVALID_INDEX ||
                dof.qIndex != expectedQ ||
                dof.vIndex != globalV ||
                dof.localDof != localDof ||
                !validDofParameters(
                    dof,
                    true,
                    MR_JOINT_FREE
                )) {
                return fail(
                    reason,
                    "floating-root DoF ownership or properties are invalid"
                );
            }
        }
        std::vector<mr_u32> childOwner(
            articulation.bodyCount,
            MR_INVALID_INDEX
        );

        for (std::size_t body = firstBody;
             body < bodyEnd;
             ++body) {
            if (bodyOwner[body] != 0u) {
                return fail(reason, "body belongs to multiple articulations");
            }
            bodyOwner[body] = 1u;
            if (bodies[body].articulationIndex != articulationIndex) {
                return fail(reason, "body articulation index is inconsistent");
            }
        }
        for (std::size_t jointIndex = firstJoint;
             jointIndex < jointEnd;
             ++jointIndex) {
            if (jointOwner[jointIndex] != 0u) {
                return fail(reason, "joint belongs to multiple articulations");
            }
            jointOwner[jointIndex] = 1u;
            const MRJointDescriptorGPU& joint = joints[jointIndex];
            const auto [jointNq, jointNv] =
                expectedJointCoordinates(joint.jointType);
            if (jointNq == MR_INVALID_INDEX ||
                joint.nq != jointNq || joint.nv != jointNv) {
                return fail(reason, "joint nq/nv does not match its type");
            }
            if (static_cast<std::size_t>(joint.parentBody) < firstBody ||
                static_cast<std::size_t>(joint.parentBody) >= bodyEnd ||
                joint.childBody <= joint.parentBody ||
                static_cast<std::size_t>(joint.childBody) >= bodyEnd) {
                return fail(
                    reason,
                    "articulation joints must be topologically ordered"
                );
            }
            if (joint.childBody == articulation.rootBody) {
                return fail(reason, "articulation root has an inbound joint");
            }
            const std::size_t localChild =
                static_cast<std::size_t>(joint.childBody) - firstBody;
            if (childOwner[localChild] != MR_INVALID_INDEX) {
                return fail(reason, "body has multiple inbound joints");
            }
            childOwner[localChild] =
                static_cast<mr_u32>(jointIndex);
            if (static_cast<std::size_t>(joint.qOffset) !=
                    qBegin + expectedNq ||
                static_cast<std::size_t>(joint.vOffset) !=
                    vBegin + expectedNv) {
                return fail(reason, "joint generalized offsets are not packed");
            }
            for (mr_u32 localDof = 0u;
                 localDof < joint.nv;
                 ++localDof) {
                const mr_u32 globalV =
                    joint.vOffset + localDof;
                const MRDofPropertiesGPU& dof = dofs[globalV];
                if (dof.articulationIndex != articulationIndex ||
                    dof.jointIndex != jointIndex ||
                    dof.qIndex !=
                        expectedJointQIndex(joint, localDof) ||
                    dof.vIndex != globalV ||
                    dof.localDof != localDof ||
                    !validDofParameters(
                        dof,
                        false,
                        joint.jointType
                    )) {
                    return fail(
                        reason,
                        "joint DoF ownership or properties are invalid"
                    );
                }
                if (((dof.flags & MR_DOF_FLAG_POSITION_LIMIT) != 0u &&
                     (defaultQ[dof.qIndex] < dof.limits.x ||
                      defaultQ[dof.qIndex] > dof.limits.y)) ||
                    ((dof.flags & MR_DOF_FLAG_VELOCITY_LIMIT) != 0u &&
                     std::abs(defaultV[dof.vIndex]) >
                         dof.limits.z)) {
                    return fail(
                        reason,
                        "default generalized state violates DoF limits"
                    );
                }
            }
            if (!finite(joint.axis0) ||
                !finite(joint.axis1) ||
                !finite(joint.axis2) ||
                !finite(joint.parentAnchor) ||
                !finite(joint.childAnchor) ||
                !unitQuaternion(joint.parentRotation) ||
                !unitQuaternion(joint.childRotation)) {
                return fail(reason, "joint frame contains an invalid value");
            }
            if ((joint.jointType == MR_JOINT_REVOLUTE ||
                 joint.jointType == MR_JOINT_PRISMATIC ||
                 joint.jointType == MR_JOINT_CONTINUOUS) &&
                !unitVector(joint.axis0)) {
                return fail(reason, "one-DOF joint axis is not unit length");
            }
            if (joint.jointType == MR_JOINT_PLANAR) {
                const double dot01 =
                    static_cast<double>(joint.axis0.x) * joint.axis1.x +
                    static_cast<double>(joint.axis0.y) * joint.axis1.y +
                    static_cast<double>(joint.axis0.z) * joint.axis1.z;
                const double dot02 =
                    static_cast<double>(joint.axis0.x) * joint.axis2.x +
                    static_cast<double>(joint.axis0.y) * joint.axis2.y +
                    static_cast<double>(joint.axis0.z) * joint.axis2.z;
                const double dot12 =
                    static_cast<double>(joint.axis1.x) * joint.axis2.x +
                    static_cast<double>(joint.axis1.y) * joint.axis2.y +
                    static_cast<double>(joint.axis1.z) * joint.axis2.z;
                if (!unitVector(joint.axis0) ||
                    !unitVector(joint.axis1) ||
                    !unitVector(joint.axis2) ||
                    std::abs(dot01) > 1.0e-5 ||
                    std::abs(dot02) > 1.0e-5 ||
                    std::abs(dot12) > 1.0e-5) {
                    return fail(
                        reason,
                        "planar joint axes are not orthonormal"
                    );
                }
            }
            if (expectedNq > qEnd - qBegin ||
                expectedNv > vEnd - vBegin ||
                static_cast<std::size_t>(joint.nq) >
                    qEnd - qBegin - expectedNq ||
                static_cast<std::size_t>(joint.nv) >
                    vEnd - vBegin - expectedNv) {
                return fail(reason, "joint generalized range exceeds owner");
            }
            expectedNq += static_cast<std::size_t>(joint.nq);
            expectedNv += static_cast<std::size_t>(joint.nv);
        }
        if (expectedNq != static_cast<std::size_t>(articulation.nq) ||
            expectedNv != static_cast<std::size_t>(articulation.nv)) {
            return fail(reason, "articulation nq/nv does not match its joints");
        }
        const std::size_t localRoot =
            static_cast<std::size_t>(articulation.rootBody) - firstBody;
        for (std::size_t localBody = 0u;
             localBody < static_cast<std::size_t>(articulation.bodyCount);
             ++localBody) {
            const std::size_t bodyIndex = firstBody + localBody;
            const MRBodyPropertiesGPU& body = bodies[bodyIndex];
            if (localBody == localRoot) {
                if (childOwner[localBody] != MR_INVALID_INDEX ||
                    body.parentBody != MR_INVALID_INDEX ||
                    body.inboundJoint != MR_INVALID_INDEX) {
                    return fail(
                        reason,
                        "articulation root ownership is inconsistent"
                    );
                }
                continue;
            }
            const mr_u32 inbound = childOwner[localBody];
            if (inbound == MR_INVALID_INDEX ||
                body.inboundJoint != inbound ||
                body.parentBody != joints[inbound].parentBody) {
                return fail(
                    reason,
                    "body parent/inbound-joint metadata is inconsistent"
                );
            }
        }
        std::vector<std::uint8_t> connected(
            articulation.bodyCount,
            0u
        );
        connected[localRoot] = 1u;
        for (std::size_t pass = 0u;
             pass < static_cast<std::size_t>(articulation.bodyCount);
             ++pass) {
            bool progressed = false;
            for (std::size_t jointIndex = firstJoint;
                 jointIndex < jointEnd;
                 ++jointIndex) {
                const MRJointDescriptorGPU& joint = joints[jointIndex];
                const std::size_t parent =
                    static_cast<std::size_t>(joint.parentBody) - firstBody;
                const std::size_t child =
                    static_cast<std::size_t>(joint.childBody) - firstBody;
                if (connected[parent] != 0u &&
                    connected[child] == 0u) {
                    connected[child] = 1u;
                    progressed = true;
                }
            }
            if (!progressed) {
                break;
            }
        }
        if (!std::ranges::all_of(
                connected,
                [](const std::uint8_t value) {
                    return value != 0u;
                }
            )) {
            return fail(reason, "articulation tree is disconnected");
        }

        for (std::size_t offset = qBegin; offset < qEnd; ++offset) {
            if (qOwner[offset] != 0u) {
                return fail(reason, "overlapping articulation q ranges");
            }
            qOwner[offset] = 1u;
        }
        for (std::size_t offset = vBegin; offset < vEnd; ++offset) {
            if (vOwner[offset] != 0u) {
                return fail(reason, "overlapping articulation v ranges");
            }
            vOwner[offset] = 1u;
        }
        if (articulation.rootType == MR_ROOT_FLOATING &&
            !normalizedQuaternion(defaultQ, qBegin + 3u)) {
            return fail(reason, "floating-root quaternion is not normalized");
        }
        for (std::size_t jointIndex = firstJoint;
             jointIndex < jointEnd;
             ++jointIndex) {
            const MRJointDescriptorGPU& joint = joints[jointIndex];
            if (joint.jointType == MR_JOINT_SPHERICAL &&
                !normalizedQuaternion(defaultQ, joint.qOffset)) {
                return fail(reason, "spherical-joint quaternion is invalid");
            }
        }
    }

    if (!std::ranges::all_of(qOwner, [](const std::uint8_t owner) {
            return owner == 1u;
        }) ||
        !std::ranges::all_of(vOwner, [](const std::uint8_t owner) {
            return owner == 1u;
        }) ||
        !std::ranges::all_of(jointOwner, [](const std::uint8_t owner) {
            return owner == 1u;
        })) {
        return fail(reason, "generalized or joint ranges contain holes");
    }

    for (std::size_t index = 0; index < bodies.size(); ++index) {
        const MRBodyPropertiesGPU& body = bodies[index];
        if (body.motionType > MR_MOTION_DYNAMIC ||
            !finite(body.massAndInverseMass) ||
            !finite(body.centerOfMass) ||
            !finite(body.inertiaRow0) ||
            !finite(body.inertiaRow1) ||
            !finite(body.inertiaRow2) ||
            !finite(body.inverseInertiaRow0) ||
            !finite(body.inverseInertiaRow1) ||
            !finite(body.inverseInertiaRow2) ||
            !finite(body.dampingAndSpeedLimits) ||
            body.dampingAndSpeedLimits.x < 0.0f ||
            body.dampingAndSpeedLimits.y < 0.0f ||
            body.dampingAndSpeedLimits.z < 0.0f ||
            body.dampingAndSpeedLimits.w < 0.0f) {
            return fail(reason, "body contains an invalid numeric field");
        }
        if (body.motionType == MR_MOTION_DYNAMIC) {
            if (body.massAndInverseMass.x <= 0.0f ||
                body.massAndInverseMass.y <= 0.0f ||
                std::abs(
                    static_cast<double>(body.massAndInverseMass.x) *
                        body.massAndInverseMass.y -
                    1.0
                ) > 2.0e-5 ||
                !symmetric(
                    body.inertiaRow0,
                    body.inertiaRow1,
                    body.inertiaRow2
                ) ||
                !symmetric(
                    body.inverseInertiaRow0,
                    body.inverseInertiaRow1,
                    body.inverseInertiaRow2
                ) ||
                !positiveDefinite(
                    body.inertiaRow0,
                    body.inertiaRow1,
                    body.inertiaRow2
                ) ||
                !positiveDefinite(
                    body.inverseInertiaRow0,
                    body.inverseInertiaRow1,
                    body.inverseInertiaRow2
                ) ||
                !inverseConsistent(
                    {
                        body.inertiaRow0,
                        body.inertiaRow1,
                        body.inertiaRow2,
                    },
                    {
                        body.inverseInertiaRow0,
                        body.inverseInertiaRow1,
                        body.inverseInertiaRow2,
                    }
                )) {
                return fail(reason, "dynamic body mass/inertia must be positive");
            }
        } else if (body.massAndInverseMass.y != 0.0f) {
            return fail(reason, "static/kinematic body has nonzero inverse mass");
        }
        if (body.articulationIndex == MR_INVALID_INDEX) {
            if (bodyOwner[index] != 0u) {
                return fail(reason, "unowned body is marked as articulated");
            }
        } else if (body.articulationIndex >= articulations.size() ||
                   bodyOwner[index] == 0u) {
            return fail(reason, "body has an invalid articulation owner");
        }
    }

    for (const MRMaterialGPU& material : materials) {
        if (!finite(material.friction) || !finite(material.response) ||
            !finite(material.geometry) ||
            material.friction.x < 0.0f || material.friction.y < 0.0f ||
            material.friction.x < material.friction.y ||
            material.friction.z < 0.0f || material.friction.w < 0.0f ||
            material.response.x < 0.0f || material.response.x > 1.0f ||
            material.response.y < 0.0f || material.response.z < 0.0f ||
            material.response.w < 0.0f ||
            material.geometry.x < 0.0f ||
            material.geometry.y < 0.0f) {
            return fail(reason, "material parameters are invalid");
        }
    }

    const auto validRange = [](
        const std::uint32_t offset,
        const std::uint32_t count,
        const std::size_t size
    ) {
        return static_cast<std::uint64_t>(offset) + count <= size;
    };
    for (const MRGeometryHeaderGPU& geometry : geometryHeaders) {
        const std::uint32_t knownFlags =
            MR_GEOMETRY_FLAG_CLOSED |
            MR_GEOMETRY_FLAG_CONVEX |
            MR_GEOMETRY_FLAG_TWO_SIDED |
            MR_GEOMETRY_FLAG_QUANTIZED_BVH;
        if ((geometry.kind != MR_GEOMETRY_CONVEX &&
             geometry.kind != MR_GEOMETRY_TRIANGLE_MESH) ||
            (geometry.flags & ~knownFlags) != 0u ||
            !finite(geometry.localLower) ||
            !finite(geometry.localUpper) ||
            geometry.localLower.x > geometry.localUpper.x ||
            geometry.localLower.y > geometry.localUpper.y ||
            geometry.localLower.z > geometry.localUpper.z ||
            !validRange(
                geometry.vertexOffset,
                geometry.vertexCount,
                geometryVertices.size()
            ) ||
            !validRange(
                geometry.indexOffset,
                geometry.indexCount,
                geometryIndices.size()
            ) ||
            !validRange(
                geometry.faceOffset,
                geometry.faceCount,
                convexFaces.size()
            ) ||
            !validRange(
                geometry.halfEdgeOffset,
                geometry.halfEdgeCount,
                convexHalfEdges.size()
            ) ||
            !validRange(
                geometry.bvhOffset,
                geometry.bvhCount,
                meshBvhNodes.size()
            ) ||
            !validRange(
                geometry.triangleOffset,
                geometry.triangleCount,
                meshTriangles.size()
            )) {
            return fail(reason, "geometry arena range or bounds are invalid");
        }
        if (geometry.kind == MR_GEOMETRY_CONVEX &&
            (geometry.vertexCount < 4u ||
             geometry.faceCount < 4u ||
             geometry.halfEdgeCount < 12u)) {
            return fail(reason, "convex geometry is incomplete");
        }
        if (geometry.kind == MR_GEOMETRY_TRIANGLE_MESH &&
            (geometry.triangleCount == 0u ||
             geometry.bvhCount == 0u)) {
            return fail(reason, "triangle mesh has no cooked BVH");
        }
    }
    for (const mr_float4& vertex : geometryVertices) {
        if (!finite(vertex)) {
            return fail(reason, "geometry vertex is nonfinite");
        }
    }

    for (const MRShapeGPU& shape : shapes) {
        const std::uint32_t knownShapeFlags =
            MR_SHAPE_FLAG_SIMULATION_DISABLED |
            MR_SHAPE_FLAG_ENABLE_CCD |
            MR_SHAPE_FLAG_MESH_TWO_SIDED;
        if (shape.bodyIndex >= bodies.size() ||
            shape.shapeType > MR_SHAPE_SDF ||
            shape.materialIndex >= materials.size() ||
            (shape.flags & ~knownShapeFlags) != 0u ||
            !finite(shape.localPosition) || !finite(shape.localRotation) ||
            !finite(shape.dimensions) ||
            !finite(shape.contactRestAndBoundingRadius) ||
            shape.contactRestAndBoundingRadius.x < 0.0f ||
            shape.contactRestAndBoundingRadius.x <
                shape.contactRestAndBoundingRadius.y ||
            shape.contactRestAndBoundingRadius.z < 0.0f) {
            return fail(reason, "shape reference, transform, or offsets invalid");
        }
        const double quaternionNorm =
            static_cast<double>(shape.localRotation.x) *
                shape.localRotation.x +
            static_cast<double>(shape.localRotation.y) *
                shape.localRotation.y +
            static_cast<double>(shape.localRotation.z) *
                shape.localRotation.z +
            static_cast<double>(shape.localRotation.w) *
                shape.localRotation.w;
        if (std::abs(quaternionNorm - 1.0) > 1.0e-5) {
            return fail(reason, "shape quaternion is not normalized");
        }
        if ((shape.shapeType == MR_SHAPE_SPHERE &&
             shape.dimensions.x <= 0.0f) ||
            ((shape.shapeType == MR_SHAPE_CAPSULE ||
              shape.shapeType == MR_SHAPE_CYLINDER) &&
             (shape.dimensions.x <= 0.0f || shape.dimensions.y <= 0.0f)) ||
            (shape.shapeType == MR_SHAPE_BOX &&
             (shape.dimensions.x <= 0.0f || shape.dimensions.y <= 0.0f ||
              shape.dimensions.z <= 0.0f))) {
            return fail(reason, "primitive shape dimensions are invalid");
        }
        if (shape.shapeType == MR_SHAPE_CONVEX ||
            shape.shapeType == MR_SHAPE_TRIANGLE_MESH) {
            if (shape.geometryCount != 1u ||
                shape.dimensions.x <= 0.0f ||
                shape.dimensions.y <= 0.0f ||
                shape.dimensions.z <= 0.0f ||
                shape.geometryOffset >= geometryHeaders.size()) {
                return fail(reason, "shape has no cooked geometry");
            }
            const MRGeometryHeaderGPU& geometry =
                geometryHeaders[shape.geometryOffset];
            if ((shape.shapeType == MR_SHAPE_CONVEX &&
                 geometry.kind != MR_GEOMETRY_CONVEX) ||
                (shape.shapeType == MR_SHAPE_TRIANGLE_MESH &&
                 geometry.kind != MR_GEOMETRY_TRIANGLE_MESH)) {
                return fail(reason, "shape and geometry kinds disagree");
            }
        } else if (shape.geometryCount != 0u) {
            return fail(reason, "primitive shape references cooked geometry");
        }
    }

    return true;
}

EngineModel makeFreeSphereEngineModel() {
    EngineModel model;
    model.name = "free_sphere_reference";
    model.world.abiVersion = MR_ENGINE_ABI_VERSION;
    model.world.bodyCount = 2u;
    model.world.articulationCount = 1u;
    model.world.jointCount = 0u;
    model.world.shapeCount = 2u;
    model.world.materialCount = 1u;
    model.world.nq = 7u;
    model.world.nv = 6u;
    model.world.pairCapacity = 4u;
    model.world.contactCapacity = 8u;
    model.world.constraintCapacity = 8u;
    model.world.islandCapacity = 2u;
    model.world.solverType = MR_SOLVER_REFERENCE_FP64;
    model.world.frictionConeType = MR_FRICTION_CONE_ELLIPTIC;
    model.world.gravityAndTimestep = f4(0.0f, -9.81f, 0.0f, 1.0f / 240.0f);
    model.world.solverScales = f4(1.0e-8f, 1.0e-9f, 2.0f, 1.0e-4f);

    MRArticulationGPU freeBody{};
    freeBody.rootBody = 1u;
    freeBody.rootType = MR_ROOT_FLOATING;
    freeBody.firstBody = 1u;
    freeBody.bodyCount = 1u;
    freeBody.firstJoint = 0u;
    freeBody.jointCount = 0u;
    freeBody.qOffset = 0u;
    freeBody.nq = 7u;
    freeBody.vOffset = 0u;
    freeBody.nv = 6u;
    model.articulations.push_back(freeBody);

    model.dofs.reserve(6u);
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

    MRBodyPropertiesGPU floor{};
    floor.articulationIndex = MR_INVALID_INDEX;
    floor.parentBody = MR_INVALID_INDEX;
    floor.inboundJoint = MR_INVALID_INDEX;
    floor.motionType = MR_MOTION_STATIC;
    floor.centerOfMass = f4(0.0f, 0.0f, 0.0f, 0.0f);
    floor.dampingAndSpeedLimits =
        f4(0.0f, 0.0f, 1.0e6f, 1.0e6f);
    model.bodies.push_back(floor);

    MRBodyPropertiesGPU sphere{};
    sphere.articulationIndex = 0u;
    sphere.parentBody = MR_INVALID_INDEX;
    sphere.inboundJoint = MR_INVALID_INDEX;
    sphere.motionType = MR_MOTION_DYNAMIC;
    sphere.massAndInverseMass = f4(1.0f, 1.0f, 0.0f, 0.0f);
    sphere.centerOfMass = f4(0.0f, 0.0f, 0.0f, 0.0f);
    sphere.inertiaRow0 = f4(0.1f, 0.0f, 0.0f, 0.0f);
    sphere.inertiaRow1 = f4(0.0f, 0.1f, 0.0f, 0.0f);
    sphere.inertiaRow2 = f4(0.0f, 0.0f, 0.1f, 0.0f);
    sphere.inverseInertiaRow0 = f4(10.0f, 0.0f, 0.0f, 0.0f);
    sphere.inverseInertiaRow1 = f4(0.0f, 10.0f, 0.0f, 0.0f);
    sphere.inverseInertiaRow2 = f4(0.0f, 0.0f, 10.0f, 0.0f);
    sphere.dampingAndSpeedLimits =
        f4(0.0f, 0.0f, 1.0e6f, 1.0e6f);
    model.bodies.push_back(sphere);

    MRMaterialGPU material{};
    material.friction = f4(0.8f, 0.6f, 0.0f, 0.0f);
    material.response = f4(0.1f, 0.5f, 0.0f, 0.0f);
    material.geometry = f4(0.01f, 0.0f, 0.0f, 0.0f);
    model.materials.push_back(material);

    MRShapeGPU plane{};
    plane.bodyIndex = 0u;
    plane.shapeType = MR_SHAPE_PLANE;
    plane.materialIndex = 0u;
    plane.collisionGroup = 1u;
    plane.collisionMask = ~0u;
    plane.slotGeneration = 1u;
    plane.localRotation = f4(0.0f, 0.0f, 0.0f, 1.0f);
    model.shapes.push_back(plane);

    MRShapeGPU ball{};
    ball.bodyIndex = 1u;
    ball.shapeType = MR_SHAPE_SPHERE;
    ball.materialIndex = 0u;
    ball.collisionGroup = 1u;
    ball.collisionMask = ~0u;
    ball.slotGeneration = 1u;
    ball.localRotation = f4(0.0f, 0.0f, 0.0f, 1.0f);
    ball.dimensions = f4(0.25f, 0.0f, 0.0f, 0.0f);
    ball.contactRestAndBoundingRadius = f4(0.01f, 0.0f, 0.25f, 0.0f);
    model.shapes.push_back(ball);

    model.defaultQ = {
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    model.defaultV.assign(6u, 0.0f);
    return model;
}

} // namespace metalrobo
