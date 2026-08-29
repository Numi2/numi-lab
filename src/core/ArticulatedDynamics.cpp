#include "metalrobo/ArticulatedDynamics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Mat3 {
    double m[3][3]{};
};

struct Quaternion {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double w = 1.0;
};

// Spatial motion is ordered angular, linear. The linear component is always
// evaluated at the body's center of mass, so its spatial inertia is block
// diagonal and remains especially well conditioned for the dense reference.
struct SpatialMotion {
    Vec3 angular;
    Vec3 linear;
};

struct SpatialForce {
    Vec3 torque;
    Vec3 force;
};

struct SpatialMatrix {
    double m[6][6]{};
};

struct BodyKinematics {
    Mat3 rotation{};
    // Despite the legacy member name, this is the COM position/velocity. The
    // engine ABI stores rigid-body translation at COM while orientation stays
    // in the body/link frame.
    Vec3 originPosition{};
    Vec3 centerOfMassPosition{};
    Vec3 originLinearVelocity{};
    Vec3 centerOfMassLinearVelocity{};
    Vec3 angularVelocity{};
    Vec3 originLinearAcceleration{};
    Vec3 centerOfMassLinearAcceleration{};
    Vec3 angularAcceleration{};
    Vec3 inboundJointPosition{};
    // Spatial columns of the inbound joint evaluated at the current state.
    // They are world-frame twists at inboundJointPosition and remain one
    // entry per joint velocity coordinate. Scalar legacy joints simply have
    // one column; FunctionBased CustomJoints retain all source coordinates.
    std::vector<SpatialMotion> inboundMotionSubspace;
    std::vector<SpatialMotion> inboundMotionSubspaceDot;
    Mat3 inertiaWorld{};
};

struct Topology {
    const MRArticulationGPU* articulation = nullptr;
    std::vector<std::uint32_t> traversal;
    std::vector<std::uint32_t> inboundJoint;
    std::vector<std::uint32_t> childForJoint;
    std::uint32_t rootQCount = 0u;
    std::uint32_t rootVCount = 0u;
};

const CompiledOpenSimSpatialTransform* functionBasedTransform(
    const EngineModel& model,
    const std::uint32_t jointIndex
) {
    for (const FunctionBasedJointProgram& program :
         model.functionBasedJointPrograms) {
        if (program.jointIndex == jointIndex) {
            return &program.transform;
        }
    }
    return nullptr;
}

Vec3 operator+(const Vec3 left, const Vec3 right) {
    return {left.x + right.x, left.y + right.y, left.z + right.z};
}

Vec3 operator-(const Vec3 left, const Vec3 right) {
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

Vec3 operator*(const Vec3 value, const double scale) {
    return {value.x * scale, value.y * scale, value.z * scale};
}

Vec3 operator/(const Vec3 value, const double scale) {
    return value * (1.0 / scale);
}

Vec3& operator+=(Vec3& left, const Vec3 right) {
    left = left + right;
    return left;
}

Vec3& operator-=(Vec3& left, const Vec3 right) {
    left = left - right;
    return left;
}

double dot(const Vec3 left, const Vec3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

Vec3 cross(const Vec3 left, const Vec3 right) {
    return {
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
    };
}

double normSquared(const Vec3 value) {
    return dot(value, value);
}

double norm(const Vec3 value) {
    return std::sqrt(normSquared(value));
}

bool finite(const double value) {
    return std::isfinite(value);
}

bool finite(const Vec3 value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

bool finite(const Mat3& value) {
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            if (!finite(value.m[row][column])) {
                return false;
            }
        }
    }
    return true;
}

bool zero(const Mat3& value) {
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            if (value.m[row][column] != 0.0) {
                return false;
            }
        }
    }
    return true;
}

bool finite(const Quaternion value) {
    return finite(value.x) && finite(value.y) && finite(value.z) &&
        finite(value.w);
}

bool finite(const mr_float4 value) {
    return finite(static_cast<double>(value.x)) &&
        finite(static_cast<double>(value.y)) &&
        finite(static_cast<double>(value.z)) &&
        finite(static_cast<double>(value.w));
}

Vec3 xyz(const mr_float4 value) {
    return {value.x, value.y, value.z};
}

Vec3 basis(const std::size_t index) {
    Vec3 result{};
    (&result.x)[index] = 1.0;
    return result;
}

Mat3 identityMatrix() {
    Mat3 result{};
    result.m[0][0] = 1.0;
    result.m[1][1] = 1.0;
    result.m[2][2] = 1.0;
    return result;
}

Mat3 matrix(
    const mr_float4 row0,
    const mr_float4 row1,
    const mr_float4 row2
) {
    return {{
        {row0.x, row0.y, row0.z},
        {row1.x, row1.y, row1.z},
        {row2.x, row2.y, row2.z},
    }};
}

Mat3 transpose(const Mat3& value) {
    Mat3 result{};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            result.m[row][column] = value.m[column][row];
        }
    }
    return result;
}

Mat3 operator*(const Mat3& left, const Mat3& right) {
    Mat3 result{};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            for (std::size_t inner = 0; inner < 3; ++inner) {
                result.m[row][column] +=
                    left.m[row][inner] * right.m[inner][column];
            }
        }
    }
    return result;
}

Vec3 operator*(const Mat3& value, const Vec3 vector) {
    return {
        value.m[0][0] * vector.x + value.m[0][1] * vector.y +
            value.m[0][2] * vector.z,
        value.m[1][0] * vector.x + value.m[1][1] * vector.y +
            value.m[1][2] * vector.z,
        value.m[2][0] * vector.x + value.m[2][1] * vector.y +
            value.m[2][2] * vector.z,
    };
}

double determinant(const Mat3& value) {
    return
        value.m[0][0] *
            (value.m[1][1] * value.m[2][2] -
             value.m[1][2] * value.m[2][1]) -
        value.m[0][1] *
            (value.m[1][0] * value.m[2][2] -
             value.m[1][2] * value.m[2][0]) +
        value.m[0][2] *
            (value.m[1][0] * value.m[2][1] -
             value.m[1][1] * value.m[2][0]);
}

double quaternionNorm(const Quaternion value) {
    return std::sqrt(
        value.x * value.x + value.y * value.y + value.z * value.z +
        value.w * value.w
    );
}

Quaternion normalized(const Quaternion value) {
    const double magnitude = quaternionNorm(value);
    if (!(magnitude > 1.0e-15) || !finite(magnitude)) {
        return {};
    }
    return {
        value.x / magnitude,
        value.y / magnitude,
        value.z / magnitude,
        value.w / magnitude,
    };
}

Quaternion operator*(const Quaternion left, const Quaternion right) {
    return {
        left.w * right.x + left.x * right.w + left.y * right.z -
            left.z * right.y,
        left.w * right.y - left.x * right.z + left.y * right.w +
            left.z * right.x,
        left.w * right.z + left.x * right.y - left.y * right.x +
            left.z * right.w,
        left.w * right.w - left.x * right.x - left.y * right.y -
            left.z * right.z,
    };
}

Quaternion quaternion(const mr_float4 value) {
    return {value.x, value.y, value.z, value.w};
}

Quaternion exponentialQuaternion(const Vec3 rotationVector) {
    const double angle = norm(rotationVector);
    if (angle < 1.0e-10) {
        const double angleSquared = angle * angle;
        const double scale = 0.5 - angleSquared / 48.0;
        return normalized({
            rotationVector.x * scale,
            rotationVector.y * scale,
            rotationVector.z * scale,
            1.0 - angleSquared / 8.0,
        });
    }
    const double halfAngle = 0.5 * angle;
    const double scale = std::sin(halfAngle) / angle;
    return {
        rotationVector.x * scale,
        rotationVector.y * scale,
        rotationVector.z * scale,
        std::cos(halfAngle),
    };
}

Mat3 rotationMatrix(const Quaternion input) {
    const Quaternion q = normalized(input);
    const double xx = q.x * q.x;
    const double yy = q.y * q.y;
    const double zz = q.z * q.z;
    const double xy = q.x * q.y;
    const double xz = q.x * q.z;
    const double yz = q.y * q.z;
    const double xw = q.x * q.w;
    const double yw = q.y * q.w;
    const double zw = q.z * q.w;
    return {{
        {1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw), 2.0 * (xz + yw)},
        {2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw)},
        {2.0 * (xz - yw), 2.0 * (yz + xw), 1.0 - 2.0 * (xx + yy)},
    }};
}

Quaternion quaternionFromRotationMatrix(const Mat3& rotation) {
    Quaternion result{};
    const double trace =
        rotation.m[0][0] + rotation.m[1][1] + rotation.m[2][2];
    if (trace > 0.0) {
        const double scale = 2.0 * std::sqrt(trace + 1.0);
        result.w = 0.25 * scale;
        result.x = (rotation.m[2][1] - rotation.m[1][2]) / scale;
        result.y = (rotation.m[0][2] - rotation.m[2][0]) / scale;
        result.z = (rotation.m[1][0] - rotation.m[0][1]) / scale;
    } else if (rotation.m[0][0] > rotation.m[1][1] &&
               rotation.m[0][0] > rotation.m[2][2]) {
        const double scale = 2.0 * std::sqrt(
            1.0 + rotation.m[0][0] -
            rotation.m[1][1] - rotation.m[2][2]
        );
        result.w = (rotation.m[2][1] - rotation.m[1][2]) / scale;
        result.x = 0.25 * scale;
        result.y = (rotation.m[0][1] + rotation.m[1][0]) / scale;
        result.z = (rotation.m[0][2] + rotation.m[2][0]) / scale;
    } else if (rotation.m[1][1] > rotation.m[2][2]) {
        const double scale = 2.0 * std::sqrt(
            1.0 + rotation.m[1][1] -
            rotation.m[0][0] - rotation.m[2][2]
        );
        result.w = (rotation.m[0][2] - rotation.m[2][0]) / scale;
        result.x = (rotation.m[0][1] + rotation.m[1][0]) / scale;
        result.y = 0.25 * scale;
        result.z = (rotation.m[1][2] + rotation.m[2][1]) / scale;
    } else {
        const double scale = 2.0 * std::sqrt(
            1.0 + rotation.m[2][2] -
            rotation.m[0][0] - rotation.m[1][1]
        );
        result.w = (rotation.m[1][0] - rotation.m[0][1]) / scale;
        result.x = (rotation.m[0][2] + rotation.m[2][0]) / scale;
        result.y = (rotation.m[1][2] + rotation.m[2][1]) / scale;
        result.z = 0.25 * scale;
    }
    result = normalized(result);
    // q and -q represent the same pose. Canonicalizing the sign keeps public
    // kinematics deterministic across algebraically equivalent branches.
    if (result.w < 0.0) {
        result = {
            -result.x,
            -result.y,
            -result.z,
            -result.w,
        };
    }
    return result;
}

Mat3 rotationAroundAxis(const Vec3 inputAxis, const double angle) {
    const Vec3 axis = inputAxis / norm(inputAxis);
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double oneMinusC = 1.0 - c;
    return {{
        {
            c + axis.x * axis.x * oneMinusC,
            axis.x * axis.y * oneMinusC - axis.z * s,
            axis.x * axis.z * oneMinusC + axis.y * s,
        },
        {
            axis.y * axis.x * oneMinusC + axis.z * s,
            c + axis.y * axis.y * oneMinusC,
            axis.y * axis.z * oneMinusC - axis.x * s,
        },
        {
            axis.z * axis.x * oneMinusC - axis.y * s,
            axis.z * axis.y * oneMinusC + axis.x * s,
            c + axis.z * axis.z * oneMinusC,
        },
    }};
}

bool symmetricPositiveDefinite(const Mat3& value) {
    constexpr double symmetryTolerance = 1.0e-10;
    const double scale = std::max({
        1.0,
        std::abs(value.m[0][0]),
        std::abs(value.m[1][1]),
        std::abs(value.m[2][2]),
    });
    if (std::abs(value.m[0][1] - value.m[1][0]) >
            symmetryTolerance * scale ||
        std::abs(value.m[0][2] - value.m[2][0]) >
            symmetryTolerance * scale ||
        std::abs(value.m[1][2] - value.m[2][1]) >
            symmetryTolerance * scale) {
        return false;
    }
    const double minor1 = value.m[0][0];
    const double minor2 =
        value.m[0][0] * value.m[1][1] -
        value.m[0][1] * value.m[1][0];
    return minor1 > 0.0 && minor2 > 0.0 && determinant(value) > 0.0;
}

bool inverseConsistent(
    const Mat3& value,
    const Mat3& inverse
) {
    constexpr double tolerance = 3.0e-4;
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            double product = 0.0;
            for (std::size_t inner = 0u; inner < 3u; ++inner) {
                product +=
                    value.m[row][inner] *
                    inverse.m[inner][column];
            }
            const double expected = row == column ? 1.0 : 0.0;
            if (std::abs(product - expected) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

ArticulatedDynamicsDiagnostics diagnosticsFor(
    const std::uint32_t articulationIndex
) {
    ArticulatedDynamicsDiagnostics result;
    result.articulationIndex = articulationIndex;
    return result;
}

ArticulatedDynamicsDiagnostics failure(
    const std::uint32_t articulationIndex,
    const ArticulatedDynamicsStatus status
) {
    ArticulatedDynamicsDiagnostics result =
        diagnosticsFor(articulationIndex);
    result.status = status;
    return result;
}

bool finiteSpan(const std::span<const double> values) {
    return std::ranges::all_of(values, [](const double value) {
        return finite(value);
    });
}

bool finiteWrench(const ArticulatedBodyWrench& wrench) {
    return std::ranges::all_of(wrench.force, [](const double value) {
               return finite(value);
           }) &&
        std::ranges::all_of(wrench.torque, [](const double value) {
            return finite(value);
        });
}

ArticulatedDynamicsStatus validateConfig(
    const ArticulatedDynamicsConfig& config
) {
    if (!std::ranges::all_of(config.gravity, [](const double value) {
            return finite(value);
        }) ||
        !(config.timestep > 0.0) || !finite(config.timestep) ||
        config.nonlinearIterations == 0u ||
        !(config.nonlinearTolerance > 0.0) ||
        !finite(config.nonlinearTolerance) ||
        !(config.limits.tolerance >= 0.0) ||
        !finite(config.limits.tolerance) ||
        static_cast<std::uint32_t>(config.integrator) >
            static_cast<std::uint32_t>(
                ArticulatedIntegrator::implicitMidpoint
            )) {
        return ArticulatedDynamicsStatus::nonfiniteInput;
    }
    return ArticulatedDynamicsStatus::success;
}

constexpr mr_u32 kKnownArticulatedDofFlags =
    MR_DOF_FLAG_ROOT |
    MR_DOF_FLAG_ACTUATED |
    MR_DOF_FLAG_POSITION_LIMIT |
    MR_DOF_FLAG_VELOCITY_LIMIT |
    MR_DOF_FLAG_EFFORT_LIMIT |
    MR_DOF_FLAG_DRIVE;

bool zeroDofVector(const mr_float4 value) {
    return value.x == 0.0f && value.y == 0.0f &&
        value.z == 0.0f && value.w == 0.0f;
}

bool validDofDynamicsParameters(
    const MRDofPropertiesGPU& dof,
    const bool root,
    const mr_u32 jointType
) {
    if (!finite(dof.limits) || !finite(dof.drive) ||
        dof.reserved0 != 0u || dof.reserved1 != 0u ||
        (dof.flags & ~kKnownArticulatedDofFlags) != 0u ||
        dof.drive.x < 0.0f || dof.drive.y < 0.0f ||
        dof.drive.z < 0.0f || dof.drive.w < 0.0f) {
        return false;
    }
    if (root) {
        return dof.flags == MR_DOF_FLAG_ROOT &&
            zeroDofVector(dof.limits) &&
            zeroDofVector(dof.drive);
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
    return
        (actuated || (!effortLimited && !driven)) &&
        (!driven || actuated) &&
        (driven || dof.drive.x == 0.0f) &&
        (!positionLimited ||
         (dof.qIndex != MR_INVALID_INDEX &&
          jointType != MR_JOINT_CONTINUOUS &&
          dof.limits.x <= dof.limits.y)) &&
        (positionLimited ||
         (dof.limits.x == 0.0f && dof.limits.y == 0.0f)) &&
        (velocityLimited
             ? dof.limits.z > 0.0f
             : dof.limits.z == 0.0f) &&
        (effortLimited
             ? dof.limits.w > 0.0f
             : dof.limits.w == 0.0f);
}

ArticulatedDynamicsStatus buildTopology(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    Topology& topology
) {
    if (articulationIndex >= model.articulations.size()) {
        return ArticulatedDynamicsStatus::invalidModel;
    }
    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    topology.articulation = &articulation;
    topology.rootQCount =
        articulation.rootType == MR_ROOT_FLOATING ? 7u : 0u;
    topology.rootVCount =
        articulation.rootType == MR_ROOT_FLOATING ? 6u : 0u;

    if ((articulation.rootType != MR_ROOT_FIXED &&
         articulation.rootType != MR_ROOT_FLOATING) ||
        articulation.bodyCount == 0u ||
        articulation.firstBody > model.bodies.size() ||
        articulation.bodyCount >
            model.bodies.size() - articulation.firstBody ||
        articulation.firstJoint > model.joints.size() ||
        articulation.jointCount >
            model.joints.size() - articulation.firstJoint ||
        articulation.vOffset > model.dofs.size() ||
        articulation.nv >
            model.dofs.size() - articulation.vOffset ||
        articulation.rootBody < articulation.firstBody ||
        articulation.rootBody >=
            articulation.firstBody + articulation.bodyCount ||
        articulation.jointCount + 1u != articulation.bodyCount) {
        return ArticulatedDynamicsStatus::invalidModel;
    }

    topology.inboundJoint.assign(
        articulation.bodyCount,
        MR_INVALID_INDEX
    );
    topology.childForJoint.assign(
        articulation.jointCount,
        MR_INVALID_INDEX
    );
    std::vector<std::uint8_t> childOwned(articulation.bodyCount, 0u);
    std::uint32_t expectedNq = topology.rootQCount;
    std::uint32_t expectedNv = topology.rootVCount;

    for (std::uint32_t localDof = 0u;
         localDof < topology.rootVCount;
         ++localDof) {
        const mr_u32 globalV =
            articulation.vOffset + localDof;
        const mr_u32 expectedQ =
            localDof < 3u
                ? articulation.qOffset + localDof
                : MR_INVALID_INDEX;
        const MRDofPropertiesGPU& dof = model.dofs[globalV];
        if (dof.articulationIndex != articulationIndex ||
            dof.jointIndex != MR_INVALID_INDEX ||
            dof.qIndex != expectedQ ||
            dof.vIndex != globalV ||
            dof.localDof != localDof ||
            !validDofDynamicsParameters(
                dof,
                true,
                MR_JOINT_FREE
            )) {
            return ArticulatedDynamicsStatus::invalidModel;
        }
    }

    for (std::uint32_t localJoint = 0u;
         localJoint < articulation.jointCount;
         ++localJoint) {
        const std::uint32_t globalJoint =
            articulation.firstJoint + localJoint;
        const MRJointDescriptorGPU& joint = model.joints[globalJoint];
        if (joint.parentBody < articulation.firstBody ||
            joint.parentBody >=
                articulation.firstBody + articulation.bodyCount ||
            joint.childBody < articulation.firstBody ||
            joint.childBody >=
                articulation.firstBody + articulation.bodyCount ||
            joint.childBody == articulation.rootBody ||
            joint.parentBody == joint.childBody) {
            return ArticulatedDynamicsStatus::invalidModel;
        }

        std::uint32_t expectedJointNq = 0u;
        std::uint32_t expectedJointNv = 0u;
        if (joint.jointType == MR_JOINT_REVOLUTE ||
            joint.jointType == MR_JOINT_CONTINUOUS ||
            joint.jointType == MR_JOINT_PRISMATIC) {
            expectedJointNq = 1u;
            expectedJointNv = 1u;
        } else if (joint.jointType == MR_JOINT_FUNCTION_BASED) {
            const CompiledOpenSimSpatialTransform* transform =
                functionBasedTransform(model, globalJoint);
            if (transform == nullptr || joint.nq == 0u || joint.nq > 6u ||
                joint.nq != joint.nv ||
                transform->coordinateCount != joint.nq) {
                return ArticulatedDynamicsStatus::invalidModel;
            }
            expectedJointNq = joint.nq;
            expectedJointNv = joint.nv;
        } else if (joint.jointType != MR_JOINT_FIXED) {
            return ArticulatedDynamicsStatus::unsupportedTopology;
        }
        if (joint.nq != expectedJointNq ||
            joint.nv != expectedJointNv ||
            expectedNq > articulation.nq ||
            expectedJointNq > articulation.nq - expectedNq ||
            expectedNv > articulation.nv ||
            expectedJointNv > articulation.nv - expectedNv ||
            joint.qOffset != articulation.qOffset + expectedNq ||
            joint.vOffset != articulation.vOffset + expectedNv ||
            !finite(joint.parentAnchor) ||
            !finite(joint.childAnchor) ||
            !finite(joint.parentRotation) ||
            !finite(joint.childRotation)) {
            return ArticulatedDynamicsStatus::invalidModel;
        }
        if (expectedJointNv == 1u &&
            joint.jointType != MR_JOINT_FUNCTION_BASED) {
            const Vec3 axis = xyz(joint.axis0);
            if (!finite(axis) || !(norm(axis) > 1.0e-12)) {
                return ArticulatedDynamicsStatus::invalidModel;
            }
        }
        for (std::uint32_t localDof = 0u;
             localDof < expectedJointNv;
             ++localDof) {
            const std::uint32_t expectedQ =
                joint.qOffset + localDof;
            const MRDofPropertiesGPU& dof =
                model.dofs[joint.vOffset + localDof];
            if (dof.articulationIndex != articulationIndex ||
                dof.jointIndex != globalJoint ||
                dof.qIndex != expectedQ ||
                dof.vIndex != joint.vOffset + localDof ||
                dof.localDof != localDof ||
                !validDofDynamicsParameters(
                    dof,
                    false,
                    joint.jointType
                )) {
                return ArticulatedDynamicsStatus::invalidModel;
            }
        }
        const double parentQuaternionNorm =
            quaternionNorm(quaternion(joint.parentRotation));
        const double childQuaternionNorm =
            quaternionNorm(quaternion(joint.childRotation));
        if (std::abs(parentQuaternionNorm - 1.0) > 2.0e-5 ||
            std::abs(childQuaternionNorm - 1.0) > 2.0e-5) {
            return ArticulatedDynamicsStatus::invalidModel;
        }

        const std::uint32_t localChild =
            joint.childBody - articulation.firstBody;
        if (childOwned[localChild] != 0u) {
            return ArticulatedDynamicsStatus::unsupportedTopology;
        }
        childOwned[localChild] = 1u;
        topology.inboundJoint[localChild] = globalJoint;
        topology.childForJoint[localJoint] = joint.childBody;
        expectedNq += expectedJointNq;
        expectedNv += expectedJointNv;
    }
    if (expectedNq != articulation.nq ||
        expectedNv != articulation.nv) {
        return ArticulatedDynamicsStatus::invalidModel;
    }

    const std::uint32_t localRoot =
        articulation.rootBody - articulation.firstBody;
    if (childOwned[localRoot] != 0u) {
        return ArticulatedDynamicsStatus::unsupportedTopology;
    }
    for (std::uint32_t localBody = 0u;
         localBody < articulation.bodyCount;
         ++localBody) {
        const std::uint32_t globalBody =
            articulation.firstBody + localBody;
        const MRBodyPropertiesGPU& body = model.bodies[globalBody];
        const Mat3 inertia = matrix(
            body.inertiaRow0,
            body.inertiaRow1,
            body.inertiaRow2
        );
        const Mat3 inverseInertia = matrix(
            body.inverseInertiaRow0,
            body.inverseInertiaRow1,
            body.inverseInertiaRow2
        );
        if (body.articulationIndex != articulationIndex ||
            !finite(body.massAndInverseMass) ||
            !finite(body.centerOfMass) ||
            !finite(inertia) ||
            !finite(inverseInertia) ||
            !finite(body.dampingAndSpeedLimits) ||
            body.dampingAndSpeedLimits.x < 0.0f ||
            body.dampingAndSpeedLimits.y < 0.0f ||
            body.dampingAndSpeedLimits.z < 0.0f ||
            body.dampingAndSpeedLimits.w < 0.0f) {
            return ArticulatedDynamicsStatus::invalidModel;
        }
        if (body.motionType == MR_MOTION_DYNAMIC) {
            if (!(body.massAndInverseMass.x > 0.0f) ||
                !(body.massAndInverseMass.y > 0.0f) ||
                std::abs(
                    static_cast<double>(body.massAndInverseMass.x) *
                        body.massAndInverseMass.y -
                    1.0
                ) > 3.0e-5 ||
                !symmetricPositiveDefinite(inertia) ||
                !symmetricPositiveDefinite(inverseInertia) ||
                !inverseConsistent(inertia, inverseInertia)) {
                return ArticulatedDynamicsStatus::invalidModel;
            }
        } else if (body.motionType != MR_MOTION_STATIC ||
                   body.massAndInverseMass.x != 0.0f ||
                   body.massAndInverseMass.y != 0.0f ||
                   !zero(inertia) || !zero(inverseInertia)) {
            // A source body can own several serial joints.  Core represents
            // the intermediate frames with exact zero-inertia transform
            // carriers; they contribute kinematics but no invented mass.
            return ArticulatedDynamicsStatus::invalidModel;
        }
        if (localBody == localRoot) {
            if (body.parentBody != MR_INVALID_INDEX ||
                body.inboundJoint != MR_INVALID_INDEX) {
                return ArticulatedDynamicsStatus::invalidModel;
            }
        } else {
            const std::uint32_t inbound =
                topology.inboundJoint[localBody];
            if (inbound == MR_INVALID_INDEX ||
                body.parentBody != model.joints[inbound].parentBody ||
                body.inboundJoint != inbound) {
                return ArticulatedDynamicsStatus::invalidModel;
            }
        }
    }

    // Discover from the root rather than relying on body or joint array order.
    std::vector<std::uint8_t> known(articulation.bodyCount, 0u);
    known[localRoot] = 1u;
    topology.traversal.clear();
    topology.traversal.reserve(articulation.bodyCount);
    topology.traversal.push_back(articulation.rootBody);
    for (std::uint32_t pass = 0u;
         pass < articulation.bodyCount;
         ++pass) {
        bool progressed = false;
        for (std::uint32_t localJoint = 0u;
             localJoint < articulation.jointCount;
             ++localJoint) {
            const MRJointDescriptorGPU& joint =
                model.joints[articulation.firstJoint + localJoint];
            const std::uint32_t localParent =
                joint.parentBody - articulation.firstBody;
            const std::uint32_t localChild =
                joint.childBody - articulation.firstBody;
            if (known[localParent] != 0u && known[localChild] == 0u) {
                known[localChild] = 1u;
                topology.traversal.push_back(joint.childBody);
                progressed = true;
            }
        }
        if (!progressed) {
            break;
        }
    }
    if (topology.traversal.size() != articulation.bodyCount) {
        return ArticulatedDynamicsStatus::unsupportedTopology;
    }
    return ArticulatedDynamicsStatus::success;
}

ArticulatedDynamicsStatus validateState(
    const EngineModel& model,
    const Topology& topology,
    const std::span<const double> q,
    const std::span<const double> v,
    const std::span<const double> acceleration,
    const std::span<const ArticulatedBodyWrench> externalWrenches,
    const ArticulatedDynamicsConfig& config,
    double& quaternionNormError
) {
    const MRArticulationGPU& articulation = *topology.articulation;
    if (q.size() != articulation.nq ||
        (!v.empty() && v.size() != articulation.nv) ||
        (!acceleration.empty() &&
         acceleration.size() != articulation.nv) ||
        (!externalWrenches.empty() &&
         externalWrenches.size() != model.bodies.size())) {
        return ArticulatedDynamicsStatus::invalidDimensions;
    }
    if (!finiteSpan(q) || (!v.empty() && !finiteSpan(v)) ||
        (!acceleration.empty() && !finiteSpan(acceleration))) {
        return ArticulatedDynamicsStatus::nonfiniteInput;
    }
    for (const std::uint32_t body : topology.traversal) {
        if (!externalWrenches.empty() &&
            !finiteWrench(externalWrenches[body])) {
            return ArticulatedDynamicsStatus::nonfiniteInput;
        }
    }

    if (articulation.rootType == MR_ROOT_FLOATING) {
        const Quaternion root{
            q[3], q[4], q[5], q[6],
        };
        const double magnitude = quaternionNorm(root);
        if (!(magnitude > 1.0e-12) || !finite(magnitude)) {
            return ArticulatedDynamicsStatus::invalidQuaternion;
        }
        quaternionNormError = std::abs(magnitude - 1.0);
        if (quaternionNormError > 2.0e-5) {
            return ArticulatedDynamicsStatus::invalidQuaternion;
        }
    } else {
        quaternionNormError = 0.0;
    }

    const bool anyLimits =
        !config.limits.lower.empty() || !config.limits.upper.empty();
    if (anyLimits) {
        if (config.limits.lower.size() != q.size() ||
            config.limits.upper.size() != q.size()) {
            return ArticulatedDynamicsStatus::invalidDimensions;
        }
        for (std::size_t index = 0; index < q.size(); ++index) {
            const double lower = config.limits.lower[index];
            const double upper = config.limits.upper[index];
            if (std::isnan(lower) || std::isnan(upper) ||
                lower > upper) {
                return ArticulatedDynamicsStatus::nonfiniteInput;
            }
            if (q[index] < lower - config.limits.tolerance ||
                q[index] > upper + config.limits.tolerance) {
                return ArticulatedDynamicsStatus::jointLimitViolation;
            }
        }
    }
    return ArticulatedDynamicsStatus::success;
}

ArticulatedDynamicsStatus buildKinematics(
    const EngineModel& model,
    const Topology& topology,
    const std::span<const double> q,
    const std::span<const double> v,
    const std::span<const double> acceleration,
    const bool enforceSpeedLimits,
    std::vector<BodyKinematics>& kinematics
) {
    const MRArticulationGPU& articulation = *topology.articulation;
    kinematics.assign(articulation.bodyCount, {});
    const std::uint32_t localRoot =
        articulation.rootBody - articulation.firstBody;
    BodyKinematics& root = kinematics[localRoot];

    if (articulation.rootType == MR_ROOT_FLOATING) {
        root.originPosition = {q[0], q[1], q[2]};
        root.rotation = rotationMatrix(
            normalized({q[3], q[4], q[5], q[6]})
        );
        if (!v.empty()) {
            root.originLinearVelocity = {v[0], v[1], v[2]};
            root.angularVelocity = {v[3], v[4], v[5]};
        }
        if (!acceleration.empty()) {
            root.originLinearAcceleration = {
                acceleration[0],
                acceleration[1],
                acceleration[2],
            };
            root.angularAcceleration = {
                acceleration[3],
                acceleration[4],
                acceleration[5],
            };
        }
    } else {
        root.rotation = identityMatrix();
    }

    for (std::size_t traversalIndex = 1u;
         traversalIndex < topology.traversal.size();
         ++traversalIndex) {
        const std::uint32_t globalChild =
            topology.traversal[traversalIndex];
        const std::uint32_t localChild =
            globalChild - articulation.firstBody;
        const std::uint32_t globalJoint =
            topology.inboundJoint[localChild];
        const MRJointDescriptorGPU& joint = model.joints[globalJoint];
        const std::uint32_t localParent =
            joint.parentBody - articulation.firstBody;
        const BodyKinematics& parent = kinematics[localParent];
        BodyKinematics& child = kinematics[localChild];

        const Mat3 parentJointRotation =
            rotationMatrix(normalized(quaternion(joint.parentRotation)));
        const Mat3 childJointRotation =
            rotationMatrix(normalized(quaternion(joint.childRotation)));
        const Mat3 parentToJoint =
            parent.rotation * parentJointRotation;
        Mat3 motionRotation = identityMatrix();
        Vec3 translationJoint{};
        std::vector<SpatialMotion> localMotionSubspace(joint.nv);
        std::vector<SpatialMotion> localMotionSubspaceDot(joint.nv);
        if (joint.jointType == MR_JOINT_FUNCTION_BASED) {
            const CompiledOpenSimSpatialTransform* transform =
                functionBasedTransform(model, globalJoint);
            if (transform == nullptr) {
                return ArticulatedDynamicsStatus::invalidModel;
            }
            std::vector<double> coordinates(joint.nq);
            std::vector<double> velocities(joint.nv, 0.0);
            for (std::size_t localDof = 0u;
                 localDof < coordinates.size();
                 ++localDof) {
                coordinates[localDof] = q[
                    joint.qOffset - articulation.qOffset + localDof
                ];
                if (!v.empty()) {
                    velocities[localDof] = v[
                        joint.vOffset - articulation.vOffset + localDof
                    ];
                }
            }
            const OpenSimSpatialTransformEvaluation evaluation =
                evaluateOpenSimSpatialTransform(
                    *transform,
                    coordinates,
                    velocities
                );
            if (!evaluation.succeeded()) {
                return evaluation.status ==
                        OpenSimSpatialTransformStatus::nonfiniteInput
                    ? ArticulatedDynamicsStatus::nonfiniteInput
                    : ArticulatedDynamicsStatus::invalidModel;
            }
            motionRotation = {{
                {evaluation.rotation[0], evaluation.rotation[1], evaluation.rotation[2]},
                {evaluation.rotation[3], evaluation.rotation[4], evaluation.rotation[5]},
                {evaluation.rotation[6], evaluation.rotation[7], evaluation.rotation[8]},
            }};
            translationJoint = {
                evaluation.translation[0],
                evaluation.translation[1],
                evaluation.translation[2],
            };
            for (std::size_t localDof = 0u;
                 localDof < localMotionSubspace.size();
                 ++localDof) {
                localMotionSubspace[localDof].angular = {
                    evaluation.motionSubspace[localDof].angular[0],
                    evaluation.motionSubspace[localDof].angular[1],
                    evaluation.motionSubspace[localDof].angular[2],
                };
                localMotionSubspace[localDof].linear = {
                    evaluation.motionSubspace[localDof].linear[0],
                    evaluation.motionSubspace[localDof].linear[1],
                    evaluation.motionSubspace[localDof].linear[2],
                };
                localMotionSubspaceDot[localDof].angular = {
                    evaluation.motionSubspaceDot[localDof].angular[0],
                    evaluation.motionSubspaceDot[localDof].angular[1],
                    evaluation.motionSubspaceDot[localDof].angular[2],
                };
                localMotionSubspaceDot[localDof].linear = {
                    evaluation.motionSubspaceDot[localDof].linear[0],
                    evaluation.motionSubspaceDot[localDof].linear[1],
                    evaluation.motionSubspaceDot[localDof].linear[2],
                };
            }
        } else if (joint.nv == 1u) {
            Vec3 axisJoint = xyz(joint.axis0);
            axisJoint = axisJoint / norm(axisJoint);
            const std::size_t qIndex =
                joint.qOffset - articulation.qOffset;
            const double jointPosition = q[qIndex];
            if (joint.jointType == MR_JOINT_REVOLUTE ||
                joint.jointType == MR_JOINT_CONTINUOUS) {
                motionRotation =
                    rotationAroundAxis(axisJoint, jointPosition);
                localMotionSubspace[0u].angular = axisJoint;
            } else if (joint.jointType == MR_JOINT_PRISMATIC) {
                translationJoint = axisJoint * jointPosition;
                localMotionSubspace[0u].linear = axisJoint;
            }
        }

        child.rotation =
            parentToJoint * motionRotation *
            transpose(childJointRotation);
        child.inboundJointPosition =
            parent.originPosition +
            parent.rotation * xyz(joint.parentAnchor) +
            parentToJoint * translationJoint;
        child.inboundMotionSubspace.resize(joint.nv);
        child.inboundMotionSubspaceDot.resize(joint.nv);
        for (std::size_t localDof = 0u;
             localDof < child.inboundMotionSubspace.size();
             ++localDof) {
            child.inboundMotionSubspace[localDof].angular =
                parentToJoint * localMotionSubspace[localDof].angular;
            child.inboundMotionSubspace[localDof].linear =
                parentToJoint * localMotionSubspace[localDof].linear;
            child.inboundMotionSubspaceDot[localDof].angular =
                parentToJoint * localMotionSubspaceDot[localDof].angular;
            child.inboundMotionSubspaceDot[localDof].linear =
                parentToJoint * localMotionSubspaceDot[localDof].linear;
        }
        const Vec3 childAnchorWorld =
            child.rotation * xyz(joint.childAnchor);
        child.originPosition =
            child.inboundJointPosition - childAnchorWorld;

        const Vec3 parentToChildJointVector =
            child.inboundJointPosition - parent.originPosition;
        Vec3 relativeAngularVelocity{};
        Vec3 relativeLinearVelocity{};
        Vec3 relativeAngularAcceleration{};
        Vec3 relativeLinearAcceleration{};
        for (std::size_t localDof = 0u;
             localDof < child.inboundMotionSubspace.size();
             ++localDof) {
            const std::size_t velocityIndex =
                joint.vOffset - articulation.vOffset + localDof;
            const double rate = v.empty() ? 0.0 : v[velocityIndex];
            const double jointAcceleration = acceleration.empty()
                ? 0.0
                : acceleration[velocityIndex];
            relativeAngularVelocity +=
                child.inboundMotionSubspace[localDof].angular * rate;
            relativeLinearVelocity +=
                child.inboundMotionSubspace[localDof].linear * rate;
            relativeAngularAcceleration +=
                child.inboundMotionSubspace[localDof].angular *
                    jointAcceleration +
                child.inboundMotionSubspaceDot[localDof].angular * rate;
            relativeLinearAcceleration +=
                child.inboundMotionSubspace[localDof].linear *
                    jointAcceleration +
                child.inboundMotionSubspaceDot[localDof].linear * rate;
        }
        const Vec3 childJointLinearVelocity =
            parent.originLinearVelocity +
            cross(parent.angularVelocity, parentToChildJointVector) +
            relativeLinearVelocity;
        child.angularVelocity =
            parent.angularVelocity + relativeAngularVelocity;
        child.originLinearVelocity =
            childJointLinearVelocity -
            cross(child.angularVelocity, childAnchorWorld);

        const Vec3 childJointLinearAcceleration =
            parent.originLinearAcceleration +
            cross(parent.angularAcceleration, parentToChildJointVector) +
            cross(
                parent.angularVelocity,
                cross(parent.angularVelocity, parentToChildJointVector)
            ) +
            cross(parent.angularVelocity, relativeLinearVelocity) * 2.0 +
            relativeLinearAcceleration;
        child.angularAcceleration = parent.angularAcceleration +
            cross(parent.angularVelocity, relativeAngularVelocity) +
            relativeAngularAcceleration;
        child.originLinearAcceleration =
            childJointLinearAcceleration -
            cross(child.angularAcceleration, childAnchorWorld) -
            cross(
                child.angularVelocity,
                cross(child.angularVelocity, childAnchorWorld)
            );
    }

    for (const std::uint32_t globalBody : topology.traversal) {
        const std::uint32_t localBody =
            globalBody - articulation.firstBody;
        const MRBodyPropertiesGPU& body = model.bodies[globalBody];
        BodyKinematics& bodyKinematics = kinematics[localBody];
        bodyKinematics.centerOfMassPosition =
            bodyKinematics.originPosition;
        bodyKinematics.centerOfMassLinearVelocity =
            bodyKinematics.originLinearVelocity;
        bodyKinematics.centerOfMassLinearAcceleration =
            bodyKinematics.originLinearAcceleration;
        bodyKinematics.inertiaWorld =
            bodyKinematics.rotation *
            matrix(
                body.inertiaRow0,
                body.inertiaRow1,
                body.inertiaRow2
            ) *
            transpose(bodyKinematics.rotation);

        if (!finite(bodyKinematics.rotation) ||
            !finite(bodyKinematics.centerOfMassPosition) ||
            !finite(bodyKinematics.centerOfMassLinearVelocity) ||
            !finite(bodyKinematics.angularVelocity) ||
            !finite(bodyKinematics.centerOfMassLinearAcceleration) ||
            !finite(bodyKinematics.angularAcceleration) ||
            !finite(bodyKinematics.inertiaWorld)) {
            return ArticulatedDynamicsStatus::nonfiniteResult;
        }
        if (enforceSpeedLimits) {
            const double maximumLinearSpeed =
                body.dampingAndSpeedLimits.z;
            const double maximumAngularSpeed =
                body.dampingAndSpeedLimits.w;
            constexpr double speedTolerance = 1.0e-10;
            if ((maximumLinearSpeed > 0.0 &&
                 norm(bodyKinematics.centerOfMassLinearVelocity) >
                     maximumLinearSpeed + speedTolerance) ||
                (maximumAngularSpeed > 0.0 &&
                 norm(bodyKinematics.angularVelocity) >
                     maximumAngularSpeed + speedTolerance)) {
                return ArticulatedDynamicsStatus::bodySpeedLimitViolation;
            }
        }
    }
    return ArticulatedDynamicsStatus::success;
}

std::vector<SpatialMotion> bodyJacobian(
    const EngineModel& model,
    const Topology& topology,
    const std::vector<BodyKinematics>& kinematics,
    const std::uint32_t globalBody
) {
    const MRArticulationGPU& articulation = *topology.articulation;
    const std::uint32_t dofCount = articulation.nv;
    const std::uint32_t localBody =
        globalBody - articulation.firstBody;
    const BodyKinematics& body = kinematics[localBody];
    std::vector<SpatialMotion> jacobian(dofCount);

    if (articulation.rootType == MR_ROOT_FLOATING) {
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            jacobian[axis].linear = basis(axis);
            jacobian[3u + axis].angular = basis(axis);
            jacobian[3u + axis].linear =
                cross(
                    basis(axis),
                    body.centerOfMassPosition -
                        kinematics[
                            articulation.rootBody -
                            articulation.firstBody
                        ].originPosition
                );
        }
    }

    std::uint32_t cursor = globalBody;
    while (cursor != articulation.rootBody) {
        const std::uint32_t localCursor =
            cursor - articulation.firstBody;
        const std::uint32_t globalJoint =
            topology.inboundJoint[localCursor];
        const MRJointDescriptorGPU& joint = model.joints[globalJoint];
        const BodyKinematics& cursorKinematics =
            kinematics[localCursor];
        for (std::size_t localDof = 0u;
             localDof < cursorKinematics.inboundMotionSubspace.size();
             ++localDof) {
            const std::size_t velocityIndex =
                joint.vOffset - articulation.vOffset + localDof;
            SpatialMotion motion =
                cursorKinematics.inboundMotionSubspace[localDof];
            motion.linear += cross(
                motion.angular,
                body.centerOfMassPosition -
                    cursorKinematics.inboundJointPosition
            );
            jacobian[velocityIndex] = motion;
        }
        cursor = joint.parentBody;
    }
    return jacobian;
}

double spatialDot(
    const SpatialMotion motion,
    const SpatialForce force
) {
    return dot(motion.angular, force.torque) +
        dot(motion.linear, force.force);
}

double motionComponent(
    const SpatialMotion motion,
    const std::size_t index
) {
    return index < 3u
        ? (&motion.angular.x)[index]
        : (&motion.linear.x)[index - 3u];
}

void setForceComponent(
    SpatialForce& force,
    const std::size_t index,
    const double value
) {
    if (index < 3u) {
        (&force.torque.x)[index] = value;
    } else {
        (&force.force.x)[index - 3u] = value;
    }
}

SpatialMatrix spatialInertia(
    const double mass,
    const Mat3& inertiaWorld
) {
    SpatialMatrix result{};
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            result.m[row][column] = inertiaWorld.m[row][column];
        }
        result.m[3u + row][3u + row] = mass;
    }
    return result;
}

SpatialForce operator*(
    const SpatialMatrix& matrixValue,
    const SpatialMotion motion
) {
    SpatialForce result{};
    for (std::size_t row = 0u; row < 6u; ++row) {
        double value = 0.0;
        for (std::size_t column = 0u; column < 6u; ++column) {
            value +=
                matrixValue.m[row][column] *
                motionComponent(motion, column);
        }
        setForceComponent(result, row, value);
    }
    return result;
}

struct Factorization {
    bool succeeded = false;
    std::vector<double> lower;
    double minimumPivot = 0.0;
    double maximumPivot = 0.0;
};

Factorization choleskyFactor(
    const std::span<const double> matrixValues,
    const std::size_t dimension
) {
    Factorization result;
    result.lower.assign(dimension * dimension, 0.0);
    result.minimumPivot = std::numeric_limits<double>::infinity();
    result.maximumPivot = 0.0;
    if (dimension == 0u) {
        result.succeeded = true;
        result.minimumPivot = 0.0;
        return result;
    }
    double maximumDiagonal = 0.0;
    for (std::size_t index = 0u; index < dimension; ++index) {
        maximumDiagonal = std::max(
            maximumDiagonal,
            std::abs(matrixValues[index * dimension + index])
        );
    }
    const double pivotFloor =
        std::max(1.0, maximumDiagonal) * 1.0e-13;
    for (std::size_t row = 0u; row < dimension; ++row) {
        for (std::size_t column = 0u; column <= row; ++column) {
            double value = matrixValues[row * dimension + column];
            for (std::size_t inner = 0u; inner < column; ++inner) {
                value -=
                    result.lower[row * dimension + inner] *
                    result.lower[column * dimension + inner];
            }
            if (row == column) {
                if (!(value > pivotFloor) || !finite(value)) {
                    return result;
                }
                const double pivot = std::sqrt(value);
                result.lower[row * dimension + column] = pivot;
                result.minimumPivot =
                    std::min(result.minimumPivot, pivot);
                result.maximumPivot =
                    std::max(result.maximumPivot, pivot);
            } else {
                result.lower[row * dimension + column] =
                    value /
                    result.lower[column * dimension + column];
            }
        }
    }
    result.succeeded = true;
    return result;
}

bool choleskySolve(
    const Factorization& factorization,
    const std::span<const double> right,
    std::vector<double>& solution
) {
    const std::size_t dimension = right.size();
    if (!factorization.succeeded ||
        factorization.lower.size() != dimension * dimension) {
        return false;
    }
    std::vector<double> intermediate(dimension, 0.0);
    for (std::size_t row = 0u; row < dimension; ++row) {
        double value = right[row];
        for (std::size_t column = 0u; column < row; ++column) {
            value -=
                factorization.lower[row * dimension + column] *
                intermediate[column];
        }
        intermediate[row] =
            value /
            factorization.lower[row * dimension + row];
    }
    solution.assign(dimension, 0.0);
    for (std::size_t reverse = 0u; reverse < dimension; ++reverse) {
        const std::size_t row = dimension - 1u - reverse;
        double value = intermediate[row];
        for (std::size_t column = row + 1u;
             column < dimension;
             ++column) {
            value -=
                factorization.lower[column * dimension + row] *
                solution[column];
        }
        solution[row] =
            value /
            factorization.lower[row * dimension + row];
    }
    return finiteSpan(solution);
}

bool denseSolve(
    std::vector<double> matrixValues,
    std::vector<double> right,
    std::vector<double>& solution
) {
    const std::size_t dimension = right.size();
    if (matrixValues.size() != dimension * dimension) {
        return false;
    }
    for (std::size_t column = 0u; column < dimension; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1u;
             row < dimension;
             ++row) {
            if (std::abs(matrixValues[row * dimension + column]) >
                std::abs(matrixValues[pivot * dimension + column])) {
                pivot = row;
            }
        }
        if (!(std::abs(matrixValues[pivot * dimension + column]) >
              1.0e-14)) {
            return false;
        }
        if (pivot != column) {
            for (std::size_t entry = column;
                 entry < dimension;
                 ++entry) {
                std::swap(
                    matrixValues[column * dimension + entry],
                    matrixValues[pivot * dimension + entry]
                );
            }
            std::swap(right[column], right[pivot]);
        }
        const double inversePivot =
            1.0 / matrixValues[column * dimension + column];
        for (std::size_t row = column + 1u;
             row < dimension;
             ++row) {
            const double scale =
                matrixValues[row * dimension + column] * inversePivot;
            matrixValues[row * dimension + column] = 0.0;
            for (std::size_t entry = column + 1u;
                 entry < dimension;
                 ++entry) {
                matrixValues[row * dimension + entry] -=
                    scale *
                    matrixValues[column * dimension + entry];
            }
            right[row] -= scale * right[column];
        }
    }
    solution.assign(dimension, 0.0);
    for (std::size_t reverse = 0u; reverse < dimension; ++reverse) {
        const std::size_t row = dimension - 1u - reverse;
        double value = right[row];
        for (std::size_t column = row + 1u;
             column < dimension;
             ++column) {
            value -=
                matrixValues[row * dimension + column] *
                solution[column];
        }
        solution[row] =
            value / matrixValues[row * dimension + row];
    }
    return finiteSpan(solution);
}

ArticulatedDynamicsDiagnostics preflight(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> q,
    const std::span<const double> v,
    const std::span<const double> acceleration,
    const std::span<const ArticulatedBodyWrench> externalWrenches,
    const ArticulatedDynamicsConfig& config,
    Topology& topology
) {
    ArticulatedDynamicsDiagnostics diagnostics =
        diagnosticsFor(articulationIndex);
    const ArticulatedDynamicsStatus configStatus =
        validateConfig(config);
    if (configStatus != ArticulatedDynamicsStatus::success) {
        diagnostics.status = configStatus;
        return diagnostics;
    }
    const ArticulatedDynamicsStatus topologyStatus =
        buildTopology(model, articulationIndex, topology);
    if (topologyStatus != ArticulatedDynamicsStatus::success) {
        diagnostics.status = topologyStatus;
        return diagnostics;
    }
    diagnostics.bodyCount = topology.articulation->bodyCount;
    diagnostics.dofCount = topology.articulation->nv;
    const ArticulatedDynamicsStatus stateStatus = validateState(
        model,
        topology,
        q,
        v,
        acceleration,
        externalWrenches,
        config,
        diagnostics.quaternionNormError
    );
    diagnostics.status = stateStatus;
    return diagnostics;
}

ArticulatedDynamicsDiagnostics assembleMassMatrix(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const Topology& topology,
    const std::span<const double> q,
    const ArticulatedDynamicsConfig& config,
    std::vector<double>& massMatrix
) {
    ArticulatedDynamicsDiagnostics diagnostics =
        diagnosticsFor(articulationIndex);
    diagnostics.bodyCount = topology.articulation->bodyCount;
    diagnostics.dofCount = topology.articulation->nv;
    const std::size_t dofCount = topology.articulation->nv;
    std::vector<double> zeroVelocity(dofCount, 0.0);
    std::vector<double> zeroAcceleration(dofCount, 0.0);
    std::vector<BodyKinematics> kinematics;
    const ArticulatedDynamicsStatus kinematicsStatus =
        buildKinematics(
            model,
            topology,
            q,
            zeroVelocity,
            zeroAcceleration,
            config.enforceBodySpeedLimits,
            kinematics
        );
    if (kinematicsStatus != ArticulatedDynamicsStatus::success) {
        diagnostics.status = kinematicsStatus;
        return diagnostics;
    }

    const MRArticulationGPU& articulation = *topology.articulation;
    // This dense FP64 reference is deliberately assembled from the analytic
    // tree Jacobian rather than finite differences. It therefore accepts any
    // number of FunctionBased H columns while retaining exactly the same
    // rigid-body inertia operator as the scalar-joint path. The throughput
    // Metal ABA remains a separate bounded implementation.
    massMatrix.assign(dofCount * dofCount, 0.0);
    for (const std::uint32_t globalBody : topology.traversal) {
        const std::uint32_t localBody =
            globalBody - articulation.firstBody;
        const MRBodyPropertiesGPU& body = model.bodies[globalBody];
        const SpatialMatrix inertia = spatialInertia(
            body.massAndInverseMass.x,
            kinematics[localBody].inertiaWorld
        );
        const std::vector<SpatialMotion> jacobian = bodyJacobian(
            model,
            topology,
            kinematics,
            globalBody
        );
        for (std::size_t row = 0u; row < dofCount; ++row) {
            for (std::size_t column = row; column < dofCount; ++column) {
                const double value = spatialDot(
                    jacobian[row],
                    inertia * jacobian[column]
                );
                massMatrix[row * dofCount + column] += value;
                if (row != column) {
                    massMatrix[column * dofCount + row] += value;
                }
            }
        }
    }
    // Rotor/armature inertia is generalized-coordinate inertia. It belongs
    // directly on M's diagonal, so every downstream factor solve (forward
    // dynamics, contact effective mass, and impulse response) observes the
    // same physical operator.
    for (std::size_t localDof = 0u;
         localDof < dofCount;
         ++localDof) {
        const MRDofPropertiesGPU& dof =
            model.dofs[articulation.vOffset + localDof];
        massMatrix[localDof * dofCount + localDof] +=
            static_cast<double>(dof.drive.z);
    }
    if (!finiteSpan(massMatrix)) {
        diagnostics.status =
            ArticulatedDynamicsStatus::nonfiniteResult;
        return diagnostics;
    }
    const Factorization factorization =
        choleskyFactor(massMatrix, dofCount);
    if (!factorization.succeeded) {
        diagnostics.status =
            ArticulatedDynamicsStatus::massMatrixNotPositiveDefinite;
        return diagnostics;
    }
    diagnostics.minimumCholeskyPivot =
        factorization.minimumPivot;
    diagnostics.maximumCholeskyPivot =
        factorization.maximumPivot;
    if (factorization.minimumPivot > 0.0) {
        const double ratio =
            factorization.maximumPivot /
            factorization.minimumPivot;
        diagnostics.estimatedMassMatrixCondition = ratio * ratio;
    }
    return diagnostics;
}

ArticulatedDynamicsDiagnostics inverseDynamicsInternal(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const Topology& topology,
    const std::span<const double> q,
    const std::span<const double> v,
    const std::span<const double> generalizedAcceleration,
    const std::span<const ArticulatedBodyWrench> externalWrenches,
    const ArticulatedDynamicsConfig& config,
    std::vector<double>& generalizedForce
) {
    ArticulatedDynamicsDiagnostics diagnostics =
        diagnosticsFor(articulationIndex);
    diagnostics.bodyCount = topology.articulation->bodyCount;
    diagnostics.dofCount = topology.articulation->nv;
    std::vector<BodyKinematics> kinematics;
    const ArticulatedDynamicsStatus kinematicsStatus =
        buildKinematics(
            model,
            topology,
            q,
            v,
            generalizedAcceleration,
            config.enforceBodySpeedLimits,
            kinematics
        );
    if (kinematicsStatus != ArticulatedDynamicsStatus::success) {
        diagnostics.status = kinematicsStatus;
        return diagnostics;
    }

    generalizedForce.assign(topology.articulation->nv, 0.0);
    const Vec3 gravity{
        config.gravity[0],
        config.gravity[1],
        config.gravity[2],
    };
    for (const std::uint32_t globalBody : topology.traversal) {
        const std::uint32_t localBody =
            globalBody - topology.articulation->firstBody;
        const MRBodyPropertiesGPU& body = model.bodies[globalBody];
        const BodyKinematics& bodyKinematics =
            kinematics[localBody];
        const double mass = body.massAndInverseMass.x;
        Vec3 requiredForce =
            (bodyKinematics.centerOfMassLinearAcceleration - gravity) *
            mass;
        Vec3 angularMomentum =
            bodyKinematics.inertiaWorld *
            bodyKinematics.angularVelocity;
        Vec3 requiredTorque =
            bodyKinematics.inertiaWorld *
                bodyKinematics.angularAcceleration +
            cross(bodyKinematics.angularVelocity, angularMomentum);

        if (!externalWrenches.empty()) {
            const ArticulatedBodyWrench& external =
                externalWrenches[globalBody];
            requiredForce -= {
                external.force[0],
                external.force[1],
                external.force[2],
            };
            requiredTorque -= {
                external.torque[0],
                external.torque[1],
                external.torque[2],
            };
        }
        if (config.applyBodyDamping) {
            requiredForce +=
                bodyKinematics.centerOfMassLinearVelocity *
                body.dampingAndSpeedLimits.x;
            requiredTorque +=
                bodyKinematics.angularVelocity *
                body.dampingAndSpeedLimits.y;
        }

        const std::vector<SpatialMotion> jacobian =
            bodyJacobian(
                model,
                topology,
                kinematics,
                globalBody
            );
        const SpatialForce requiredWrench{
            requiredTorque,
            requiredForce,
        };
        for (std::size_t dof = 0u;
             dof < generalizedForce.size();
             ++dof) {
            generalizedForce[dof] +=
                spatialDot(jacobian[dof], requiredWrench);
        }
    }
    const MRArticulationGPU& articulation =
        *topology.articulation;
    for (std::size_t localDof = 0u;
         localDof < generalizedForce.size();
         ++localDof) {
        const MRDofPropertiesGPU& dof =
            model.dofs[articulation.vOffset + localDof];
        generalizedForce[localDof] +=
            static_cast<double>(dof.drive.z) *
            generalizedAcceleration[localDof];
        // A non-driven damping coefficient is passive source physics. Driven
        // damping remains an actuator gain and is evaluated by the explicit
        // actuation layer, so it must not be counted twice here.
        if ((dof.flags & MR_DOF_FLAG_DRIVE) == 0u) {
            generalizedForce[localDof] +=
                static_cast<double>(dof.drive.y) * v[localDof];
        }
    }
    if (!finiteSpan(generalizedForce)) {
        diagnostics.status =
            ArticulatedDynamicsStatus::nonfiniteResult;
    }
    return diagnostics;
}

ArticulatedDynamicsStatus integrateConfiguration(
    const EngineModel& model,
    const Topology& topology,
    const std::span<const double> q,
    const std::span<const double> velocity,
    const double timestep,
    std::vector<double>& integrated
) {
    const MRArticulationGPU& articulation = *topology.articulation;
    integrated.assign(q.begin(), q.end());
    if (articulation.rootType == MR_ROOT_FLOATING) {
        integrated[0] += timestep * velocity[0];
        integrated[1] += timestep * velocity[1];
        integrated[2] += timestep * velocity[2];
        const Quaternion orientation =
            normalized({q[3], q[4], q[5], q[6]});
        const Quaternion increment = exponentialQuaternion({
            timestep * velocity[3],
            timestep * velocity[4],
            timestep * velocity[5],
        });
        const Quaternion updated =
            normalized(increment * orientation);
        if (!finite(updated) ||
            !(quaternionNorm(updated) > 1.0e-12)) {
            return ArticulatedDynamicsStatus::nonfiniteResult;
        }
        integrated[3] = updated.x;
        integrated[4] = updated.y;
        integrated[5] = updated.z;
        integrated[6] = updated.w;
    }
    for (std::uint32_t localJoint = 0u;
         localJoint < articulation.jointCount;
         ++localJoint) {
        const MRJointDescriptorGPU& joint =
            model.joints[articulation.firstJoint + localJoint];
        // FunctionBased CustomJoints have source scalar coordinates (nq ==
        // nv), as do the legacy one-DoF joints. Quaternion joints remain
        // outside this reference integrator's admitted topology.
        if (joint.nq == joint.nv) {
            for (std::size_t localDof = 0u;
                 localDof < joint.nv;
                 ++localDof) {
                const std::size_t qIndex =
                    joint.qOffset - articulation.qOffset + localDof;
                const std::size_t velocityIndex =
                    joint.vOffset - articulation.vOffset + localDof;
                integrated[qIndex] +=
                    timestep * velocity[velocityIndex];
            }
        }
    }
    return finiteSpan(integrated)
        ? ArticulatedDynamicsStatus::success
        : ArticulatedDynamicsStatus::nonfiniteResult;
}

double maximumAbsolute(const std::span<const double> values) {
    double result = 0.0;
    for (const double value : values) {
        result = std::max(result, std::abs(value));
    }
    return result;
}

void mergeFactorizationDiagnostics(
    ArticulatedDynamicsDiagnostics& target,
    const ArticulatedDynamicsDiagnostics& source
) {
    if (source.minimumCholeskyPivot > 0.0 &&
        (target.minimumCholeskyPivot == 0.0 ||
         source.minimumCholeskyPivot <
             target.minimumCholeskyPivot)) {
        target.minimumCholeskyPivot =
            source.minimumCholeskyPivot;
    }
    target.maximumCholeskyPivot = std::max(
        target.maximumCholeskyPivot,
        source.maximumCholeskyPivot
    );
    target.estimatedMassMatrixCondition = std::max(
        target.estimatedMassMatrixCondition,
        source.estimatedMassMatrixCondition
    );
}

} // namespace

ArticulatedDynamicsDiagnostics computeArticulatedBodyKinematics(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> q,
    const std::span<const double> v,
    const std::span<ArticulatedBodyKinematics> bodyKinematics,
    const ArticulatedDynamicsConfig& config
) {
    Topology topology;
    ArticulatedDynamicsDiagnostics diagnostics = preflight(
        model,
        articulationIndex,
        q,
        v,
        {},
        {},
        config,
        topology
    );
    if (!diagnostics.succeeded()) {
        return diagnostics;
    }
    if (bodyKinematics.size() != topology.articulation->bodyCount) {
        diagnostics.status =
            ArticulatedDynamicsStatus::invalidDimensions;
        return diagnostics;
    }

    std::vector<BodyKinematics> internal;
    const ArticulatedDynamicsStatus kinematicsStatus =
        buildKinematics(
            model,
            topology,
            q,
            v,
            {},
            config.enforceBodySpeedLimits,
            internal
        );
    if (kinematicsStatus != ArticulatedDynamicsStatus::success) {
        diagnostics.status = kinematicsStatus;
        return diagnostics;
    }

    for (const BodyKinematics& source : internal) {
        const Quaternion orientation =
            quaternionFromRotationMatrix(source.rotation);
        if (!finite(source.centerOfMassPosition) ||
            !finite(source.centerOfMassLinearVelocity) ||
            !finite(source.angularVelocity) ||
            !finite(orientation)) {
            diagnostics.status =
                ArticulatedDynamicsStatus::nonfiniteResult;
            return diagnostics;
        }
    }
    for (std::size_t localBody = 0u;
         localBody < bodyKinematics.size();
         ++localBody) {
        const BodyKinematics& source = internal[localBody];
        const Quaternion orientation =
            quaternionFromRotationMatrix(source.rotation);
        ArticulatedBodyKinematics& destination =
            bodyKinematics[localBody];
        destination.bodyIndex =
            topology.articulation->firstBody +
            static_cast<std::uint32_t>(localBody);
        destination.centerOfMassPosition = {
            source.centerOfMassPosition.x,
            source.centerOfMassPosition.y,
            source.centerOfMassPosition.z,
        };
        destination.orientation = {
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        };
        destination.linearVelocity = {
            source.centerOfMassLinearVelocity.x,
            source.centerOfMassLinearVelocity.y,
            source.centerOfMassLinearVelocity.z,
        };
        destination.angularVelocity = {
            source.angularVelocity.x,
            source.angularVelocity.y,
            source.angularVelocity.z,
        };
    }
    return diagnostics;
}

ArticulatedDynamicsDiagnostics computeArticulatedPointJacobians(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> q,
    const std::span<const double> v,
    const std::span<const ArticulatedPointQuery> points,
    const std::span<ArticulatedPointKinematics> pointKinematics,
    const std::span<double> pointJacobiansRowMajor,
    const ArticulatedDynamicsConfig& config
) {
    Topology topology;
    ArticulatedDynamicsDiagnostics diagnostics = preflight(
        model,
        articulationIndex,
        q,
        v,
        {},
        {},
        config,
        topology
    );
    if (!diagnostics.succeeded()) {
        return diagnostics;
    }
    const std::size_t dofCount = topology.articulation->nv;
    if (points.size() >
        std::numeric_limits<std::size_t>::max() /
            std::max<std::size_t>(3u * dofCount, 1u) ||
        pointKinematics.size() != points.size() ||
        pointJacobiansRowMajor.size() !=
            points.size() * 3u * dofCount) {
        diagnostics.status =
            ArticulatedDynamicsStatus::invalidDimensions;
        return diagnostics;
    }

    for (const ArticulatedPointQuery& point : points) {
        if (point.bodyIndex < topology.articulation->firstBody ||
            point.bodyIndex >=
                topology.articulation->firstBody +
                    topology.articulation->bodyCount ||
            !std::ranges::all_of(
                point.localPoint,
                [](const double value) {
                    return finite(value);
                }
            )) {
            diagnostics.status =
                point.bodyIndex < topology.articulation->firstBody ||
                    point.bodyIndex >=
                        topology.articulation->firstBody +
                            topology.articulation->bodyCount
                ? ArticulatedDynamicsStatus::invalidModel
                : ArticulatedDynamicsStatus::nonfiniteInput;
            return diagnostics;
        }
    }

    std::vector<BodyKinematics> internal;
    const ArticulatedDynamicsStatus kinematicsStatus =
        buildKinematics(
            model,
            topology,
            q,
            v,
            {},
            config.enforceBodySpeedLimits,
            internal
        );
    if (kinematicsStatus != ArticulatedDynamicsStatus::success) {
        diagnostics.status = kinematicsStatus;
        return diagnostics;
    }

    std::vector<ArticulatedPointKinematics> result(points.size());
    std::vector<double> jacobians(
        points.size() * 3u * dofCount,
        0.0
    );
    for (std::size_t pointIndex = 0u;
         pointIndex < points.size();
         ++pointIndex) {
        const ArticulatedPointQuery& query = points[pointIndex];
        const std::size_t localBody =
            query.bodyIndex - topology.articulation->firstBody;
        const BodyKinematics& body = internal[localBody];
        const Vec3 localPoint{
            query.localPoint[0],
            query.localPoint[1],
            query.localPoint[2],
        };
        const Vec3 centerToPoint = body.rotation * localPoint;
        const Vec3 position =
            body.centerOfMassPosition + centerToPoint;
        const Vec3 velocity =
            body.centerOfMassLinearVelocity +
            cross(body.angularVelocity, centerToPoint);
        result[pointIndex].position = {
            position.x,
            position.y,
            position.z,
        };
        result[pointIndex].linearVelocity = {
            velocity.x,
            velocity.y,
            velocity.z,
        };

        const std::vector<SpatialMotion> bodyJ =
            bodyJacobian(model, topology, internal, query.bodyIndex);
        for (std::size_t dof = 0u; dof < dofCount; ++dof) {
            const Vec3 pointColumn =
                bodyJ[dof].linear +
                cross(bodyJ[dof].angular, centerToPoint);
            jacobians[
                (pointIndex * 3u + 0u) * dofCount + dof
            ] = pointColumn.x;
            jacobians[
                (pointIndex * 3u + 1u) * dofCount + dof
            ] = pointColumn.y;
            jacobians[
                (pointIndex * 3u + 2u) * dofCount + dof
            ] = pointColumn.z;
        }
        if (!finite(position) || !finite(velocity)) {
            diagnostics.status =
                ArticulatedDynamicsStatus::nonfiniteResult;
            return diagnostics;
        }
    }
    if (!finiteSpan(jacobians)) {
        diagnostics.status =
            ArticulatedDynamicsStatus::nonfiniteResult;
        return diagnostics;
    }

    std::ranges::copy(result, pointKinematics.begin());
    std::ranges::copy(jacobians, pointJacobiansRowMajor.begin());
    return diagnostics;
}

ArticulatedDynamicsDiagnostics computeArticulatedMassMatrix(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> q,
    const std::span<double> massMatrixRowMajor,
    const ArticulatedDynamicsConfig& config
) {
    Topology topology;
    ArticulatedDynamicsDiagnostics diagnostics = preflight(
        model,
        articulationIndex,
        q,
        {},
        {},
        {},
        config,
        topology
    );
    if (!diagnostics.succeeded()) {
        return diagnostics;
    }
    const std::size_t dofCount = topology.articulation->nv;
    if (massMatrixRowMajor.size() != dofCount * dofCount) {
        diagnostics.status =
            ArticulatedDynamicsStatus::invalidDimensions;
        return diagnostics;
    }

    std::vector<double> matrixValues;
    diagnostics = assembleMassMatrix(
        model,
        articulationIndex,
        topology,
        q,
        config,
        matrixValues
    );
    if (diagnostics.succeeded()) {
        std::ranges::copy(matrixValues, massMatrixRowMajor.begin());
    }
    return diagnostics;
}

ArticulatedDynamicsDiagnostics computeArticulatedInverseMassResponses(
    const EngineModel& model, const std::uint32_t articulationIndex,
    const std::span<const double> q, const std::span<const double> rhsRowMajor,
    const std::span<double> responseRowMajor, const ArticulatedDynamicsConfig& config
) {
    Topology topology;
    ArticulatedDynamicsDiagnostics diagnostics = preflight(model, articulationIndex, q, {}, {}, {}, config, topology);
    if (!diagnostics.succeeded()) return diagnostics;
    const std::size_t nv = topology.articulation->nv;
    if (rhsRowMajor.empty() || rhsRowMajor.size() % nv != 0u || responseRowMajor.size() != rhsRowMajor.size() || !finiteSpan(rhsRowMajor)) {
        diagnostics.status = ArticulatedDynamicsStatus::invalidDimensions;
        return diagnostics;
    }
    std::vector<double> mass;
    diagnostics = assembleMassMatrix(model, articulationIndex, topology, q, config, mass);
    if (!diagnostics.succeeded()) return diagnostics;
    const Factorization factor = choleskyFactor(mass, nv);
    if (!factor.succeeded) { diagnostics.status = ArticulatedDynamicsStatus::massMatrixNotPositiveDefinite; return diagnostics; }
    for (std::size_t offset = 0u; offset < rhsRowMajor.size(); offset += nv) {
        std::vector<double> solved;
        if (!choleskySolve(factor, rhsRowMajor.subspan(offset, nv), solved)) { diagnostics.status = ArticulatedDynamicsStatus::massMatrixNotPositiveDefinite; return diagnostics; }
        std::ranges::copy(solved, responseRowMajor.begin() + offset);
    }
    diagnostics.minimumCholeskyPivot = factor.minimumPivot;
    return diagnostics;
}

ArticulatedDynamicsDiagnostics computeArticulatedInverseDynamics(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> q,
    const std::span<const double> v,
    const std::span<const double> generalizedAcceleration,
    const std::span<const ArticulatedBodyWrench> externalWrenches,
    const std::span<double> generalizedForce,
    const ArticulatedDynamicsConfig& config
) {
    Topology topology;
    ArticulatedDynamicsDiagnostics diagnostics = preflight(
        model,
        articulationIndex,
        q,
        v,
        generalizedAcceleration,
        externalWrenches,
        config,
        topology
    );
    if (!diagnostics.succeeded()) {
        return diagnostics;
    }
    if (generalizedForce.size() != topology.articulation->nv) {
        diagnostics.status =
            ArticulatedDynamicsStatus::invalidDimensions;
        return diagnostics;
    }
    std::vector<double> forceValues;
    diagnostics = inverseDynamicsInternal(
        model,
        articulationIndex,
        topology,
        q,
        v,
        generalizedAcceleration,
        externalWrenches,
        config,
        forceValues
    );
    if (diagnostics.succeeded()) {
        std::ranges::copy(forceValues, generalizedForce.begin());
    }
    return diagnostics;
}

ArticulatedDynamicsDiagnostics computeArticulatedForwardDynamics(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> q,
    const std::span<const double> v,
    const std::span<const double> generalizedForce,
    const std::span<const ArticulatedBodyWrench> externalWrenches,
    const std::span<double> generalizedAcceleration,
    const ArticulatedDynamicsConfig& config
) {
    Topology topology;
    ArticulatedDynamicsDiagnostics diagnostics = preflight(
        model,
        articulationIndex,
        q,
        v,
        {},
        externalWrenches,
        config,
        topology
    );
    if (!diagnostics.succeeded()) {
        return diagnostics;
    }
    const std::size_t dofCount = topology.articulation->nv;
    if (generalizedForce.size() != dofCount ||
        generalizedAcceleration.size() != dofCount ||
        !finiteSpan(generalizedForce)) {
        diagnostics.status =
            generalizedForce.size() != dofCount ||
                generalizedAcceleration.size() != dofCount
            ? ArticulatedDynamicsStatus::invalidDimensions
            : ArticulatedDynamicsStatus::nonfiniteInput;
        return diagnostics;
    }

    std::vector<double> massMatrix;
    const ArticulatedDynamicsDiagnostics massDiagnostics =
        assembleMassMatrix(
            model,
            articulationIndex,
            topology,
            q,
            config,
            massMatrix
        );
    if (!massDiagnostics.succeeded()) {
        return massDiagnostics;
    }
    mergeFactorizationDiagnostics(diagnostics, massDiagnostics);

    std::vector<double> zeroAcceleration(dofCount, 0.0);
    std::vector<double> bias;
    const ArticulatedDynamicsDiagnostics biasDiagnostics =
        inverseDynamicsInternal(
            model,
            articulationIndex,
            topology,
            q,
            v,
            zeroAcceleration,
            externalWrenches,
            config,
            bias
        );
    if (!biasDiagnostics.succeeded()) {
        return biasDiagnostics;
    }
    std::vector<double> right(dofCount, 0.0);
    for (std::size_t dof = 0u; dof < dofCount; ++dof) {
        right[dof] = generalizedForce[dof] - bias[dof];
    }
    const Factorization factorization =
        choleskyFactor(massMatrix, dofCount);
    std::vector<double> acceleration;
    if (!choleskySolve(factorization, right, acceleration)) {
        diagnostics.status =
            ArticulatedDynamicsStatus::massMatrixNotPositiveDefinite;
        return diagnostics;
    }
    if (!finiteSpan(acceleration)) {
        diagnostics.status =
            ArticulatedDynamicsStatus::nonfiniteResult;
        return diagnostics;
    }
    std::ranges::copy(
        acceleration,
        generalizedAcceleration.begin()
    );
    return diagnostics;
}

ArticulatedDynamicsDiagnostics integrateArticulatedState(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<double> q,
    const std::span<double> v,
    const std::span<const double> generalizedForce,
    const std::span<const ArticulatedBodyWrench> externalWrenches,
    const ArticulatedDynamicsConfig& config
) {
    Topology topology;
    ArticulatedDynamicsDiagnostics diagnostics = preflight(
        model,
        articulationIndex,
        q,
        v,
        {},
        externalWrenches,
        config,
        topology
    );
    if (!diagnostics.succeeded()) {
        return diagnostics;
    }
    const std::size_t dofCount = topology.articulation->nv;
    if (generalizedForce.size() != dofCount ||
        !finiteSpan(generalizedForce)) {
        diagnostics.status =
            generalizedForce.size() != dofCount
            ? ArticulatedDynamicsStatus::invalidDimensions
            : ArticulatedDynamicsStatus::nonfiniteInput;
        return diagnostics;
    }

    const std::vector<double> initialQ(q.begin(), q.end());
    const std::vector<double> initialV(v.begin(), v.end());
    std::vector<double> finalQ;
    std::vector<double> finalV;

    if (config.integrator ==
        ArticulatedIntegrator::symplecticEuler) {
        std::vector<double> acceleration(dofCount, 0.0);
        const ArticulatedDynamicsDiagnostics forwardDiagnostics =
            computeArticulatedForwardDynamics(
                model,
                articulationIndex,
                initialQ,
                initialV,
                generalizedForce,
                externalWrenches,
                acceleration,
                config
            );
        if (!forwardDiagnostics.succeeded()) {
            return forwardDiagnostics;
        }
        mergeFactorizationDiagnostics(
            diagnostics,
            forwardDiagnostics
        );
        finalV.resize(dofCount);
        for (std::size_t dof = 0u; dof < dofCount; ++dof) {
            finalV[dof] =
                initialV[dof] + config.timestep * acceleration[dof];
        }
        const ArticulatedDynamicsStatus integrationStatus =
            integrateConfiguration(
                model,
                topology,
                initialQ,
                finalV,
                config.timestep,
                finalQ
            );
        if (integrationStatus !=
            ArticulatedDynamicsStatus::success) {
            diagnostics.status = integrationStatus;
            return diagnostics;
        }
    } else {
        std::vector<double> initialAcceleration(dofCount, 0.0);
        const ArticulatedDynamicsDiagnostics forwardDiagnostics =
            computeArticulatedForwardDynamics(
                model,
                articulationIndex,
                initialQ,
                initialV,
                generalizedForce,
                externalWrenches,
                initialAcceleration,
                config
            );
        if (!forwardDiagnostics.succeeded()) {
            return forwardDiagnostics;
        }
        mergeFactorizationDiagnostics(
            diagnostics,
            forwardDiagnostics
        );
        std::vector<double> candidate(dofCount, 0.0);
        for (std::size_t dof = 0u; dof < dofCount; ++dof) {
            candidate[dof] =
                initialV[dof] +
                config.timestep * initialAcceleration[dof];
        }

        auto residualAt =
            [&](const std::span<const double> candidateVelocity,
                std::vector<double>& residual)
            -> ArticulatedDynamicsDiagnostics {
                std::vector<double> midpointVelocity(dofCount, 0.0);
                for (std::size_t dof = 0u;
                     dof < dofCount;
                     ++dof) {
                    midpointVelocity[dof] =
                        0.5 *
                        (initialV[dof] + candidateVelocity[dof]);
                }
                std::vector<double> midpointQ;
                const ArticulatedDynamicsStatus midpointStatus =
                    integrateConfiguration(
                        model,
                        topology,
                        initialQ,
                        midpointVelocity,
                        0.5 * config.timestep,
                        midpointQ
                    );
                if (midpointStatus !=
                    ArticulatedDynamicsStatus::success) {
                    return failure(
                        articulationIndex,
                        midpointStatus
                    );
                }
                std::vector<double> midpointAcceleration(
                    dofCount,
                    0.0
                );
                ArticulatedDynamicsDiagnostics result =
                    computeArticulatedForwardDynamics(
                        model,
                        articulationIndex,
                        midpointQ,
                        midpointVelocity,
                        generalizedForce,
                        externalWrenches,
                        midpointAcceleration,
                        config
                    );
                if (!result.succeeded()) {
                    return result;
                }
                residual.resize(dofCount);
                for (std::size_t dof = 0u;
                     dof < dofCount;
                     ++dof) {
                    residual[dof] =
                        candidateVelocity[dof] - initialV[dof] -
                        config.timestep * midpointAcceleration[dof];
                }
                if (!finiteSpan(residual)) {
                    result.status =
                        ArticulatedDynamicsStatus::nonfiniteResult;
                }
                return result;
            };

        std::vector<double> residual;
        bool converged = false;
        for (std::uint32_t iteration = 0u;
             iteration < config.nonlinearIterations;
             ++iteration) {
            const ArticulatedDynamicsDiagnostics residualDiagnostics =
                residualAt(candidate, residual);
            if (!residualDiagnostics.succeeded()) {
                return residualDiagnostics;
            }
            mergeFactorizationDiagnostics(
                diagnostics,
                residualDiagnostics
            );
            diagnostics.nonlinearIterations = iteration + 1u;
            diagnostics.nonlinearResidual =
                maximumAbsolute(residual);
            if (diagnostics.nonlinearResidual <=
                config.nonlinearTolerance) {
                converged = true;
                break;
            }

            std::vector<double> jacobian(
                dofCount * dofCount,
                0.0
            );
            for (std::size_t column = 0u;
                 column < dofCount;
                 ++column) {
                std::vector<double> perturbed = candidate;
                const double step =
                    std::sqrt(std::numeric_limits<double>::epsilon()) *
                    std::max(1.0, std::abs(candidate[column]));
                perturbed[column] += step;
                std::vector<double> perturbedResidual;
                const ArticulatedDynamicsDiagnostics
                    perturbedDiagnostics =
                        residualAt(
                            perturbed,
                            perturbedResidual
                        );
                if (!perturbedDiagnostics.succeeded()) {
                    return perturbedDiagnostics;
                }
                for (std::size_t row = 0u;
                     row < dofCount;
                     ++row) {
                    jacobian[row * dofCount + column] =
                        (
                            perturbedResidual[row] -
                            residual[row]
                        ) / step;
                }
            }
            std::vector<double> right(dofCount, 0.0);
            for (std::size_t dof = 0u; dof < dofCount; ++dof) {
                right[dof] = -residual[dof];
            }
            std::vector<double> step;
            if (!denseSolve(jacobian, right, step)) {
                diagnostics.status =
                    ArticulatedDynamicsStatus::nonlinearSolveFailed;
                return diagnostics;
            }

            const double oldResidual =
                diagnostics.nonlinearResidual;
            bool accepted = false;
            double scale = 1.0;
            for (std::uint32_t lineSearch = 0u;
                 lineSearch < 8u;
                 ++lineSearch) {
                std::vector<double> trial = candidate;
                for (std::size_t dof = 0u;
                     dof < dofCount;
                     ++dof) {
                    trial[dof] += scale * step[dof];
                }
                std::vector<double> trialResidual;
                const ArticulatedDynamicsDiagnostics trialDiagnostics =
                    residualAt(trial, trialResidual);
                if (trialDiagnostics.succeeded() &&
                    maximumAbsolute(trialResidual) < oldResidual) {
                    candidate = std::move(trial);
                    accepted = true;
                    break;
                }
                scale *= 0.5;
            }
            if (!accepted) {
                diagnostics.status =
                    ArticulatedDynamicsStatus::nonlinearSolveFailed;
                return diagnostics;
            }
        }
        if (!converged) {
            diagnostics.status =
                ArticulatedDynamicsStatus::nonlinearSolveFailed;
            return diagnostics;
        }
        finalV = std::move(candidate);
        std::vector<double> midpointVelocity(dofCount, 0.0);
        for (std::size_t dof = 0u; dof < dofCount; ++dof) {
            midpointVelocity[dof] =
                0.5 * (initialV[dof] + finalV[dof]);
        }
        const ArticulatedDynamicsStatus integrationStatus =
            integrateConfiguration(
                model,
                topology,
                initialQ,
                midpointVelocity,
                config.timestep,
                finalQ
            );
        if (integrationStatus !=
            ArticulatedDynamicsStatus::success) {
            diagnostics.status = integrationStatus;
            return diagnostics;
        }
    }

    double finalQuaternionError = 0.0;
    const ArticulatedDynamicsStatus finalStateStatus =
        validateState(
            model,
            topology,
            finalQ,
            finalV,
            {},
            externalWrenches,
            config,
            finalQuaternionError
        );
    if (finalStateStatus != ArticulatedDynamicsStatus::success) {
        diagnostics.status = finalStateStatus;
        return diagnostics;
    }
    std::vector<BodyKinematics> finalKinematics;
    const ArticulatedDynamicsStatus finalKinematicsStatus =
        buildKinematics(
            model,
            topology,
            finalQ,
            finalV,
            {},
            config.enforceBodySpeedLimits,
            finalKinematics
        );
    if (finalKinematicsStatus !=
        ArticulatedDynamicsStatus::success) {
        diagnostics.status = finalKinematicsStatus;
        return diagnostics;
    }
    diagnostics.quaternionNormError = finalQuaternionError;
    std::ranges::copy(finalQ, q.begin());
    std::ranges::copy(finalV, v.begin());
    return diagnostics;
}

ArticulatedDynamicsDiagnostics integrateArticulatedConfiguration(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<double> q,
    const std::span<const double> velocity,
    const ArticulatedDynamicsConfig& config
) {
    Topology topology;
    ArticulatedDynamicsDiagnostics diagnostics = preflight(
        model,
        articulationIndex,
        q,
        velocity,
        {},
        {},
        config,
        topology
    );
    if (!diagnostics.succeeded()) {
        return diagnostics;
    }

    std::vector<double> candidate;
    const ArticulatedDynamicsStatus integrationStatus =
        integrateConfiguration(
            model,
            topology,
            q,
            velocity,
            config.timestep,
            candidate
        );
    if (integrationStatus != ArticulatedDynamicsStatus::success) {
        diagnostics.status = integrationStatus;
        return diagnostics;
    }

    double quaternionNormError = 0.0;
    const ArticulatedDynamicsStatus stateStatus = validateState(
        model,
        topology,
        candidate,
        velocity,
        {},
        {},
        config,
        quaternionNormError
    );
    if (stateStatus != ArticulatedDynamicsStatus::success) {
        diagnostics.status = stateStatus;
        return diagnostics;
    }
    std::vector<BodyKinematics> kinematics;
    const ArticulatedDynamicsStatus kinematicsStatus =
        buildKinematics(
            model,
            topology,
            candidate,
            velocity,
            {},
            config.enforceBodySpeedLimits,
            kinematics
        );
    if (kinematicsStatus != ArticulatedDynamicsStatus::success) {
        diagnostics.status = kinematicsStatus;
        return diagnostics;
    }

    diagnostics.quaternionNormError = quaternionNormError;
    std::ranges::copy(candidate, q.begin());
    return diagnostics;
}

ArticulatedDynamicsDiagnostics computeArticulatedInvariants(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> q,
    const std::span<const double> v,
    ArticulatedInvariants& invariants,
    const ArticulatedDynamicsConfig& config
) {
    Topology topology;
    ArticulatedDynamicsDiagnostics diagnostics = preflight(
        model,
        articulationIndex,
        q,
        v,
        {},
        {},
        config,
        topology
    );
    if (!diagnostics.succeeded()) {
        return diagnostics;
    }
    std::vector<BodyKinematics> kinematics;
    const ArticulatedDynamicsStatus kinematicsStatus =
        buildKinematics(
            model,
            topology,
            q,
            v,
            {},
            config.enforceBodySpeedLimits,
            kinematics
        );
    if (kinematicsStatus != ArticulatedDynamicsStatus::success) {
        diagnostics.status = kinematicsStatus;
        return diagnostics;
    }

    ArticulatedInvariants result;
    const Vec3 gravity{
        config.gravity[0],
        config.gravity[1],
        config.gravity[2],
    };
    Vec3 linearMomentum{};
    Vec3 angularMomentum{};
    for (const std::uint32_t globalBody : topology.traversal) {
        const std::uint32_t localBody =
            globalBody - topology.articulation->firstBody;
        const MRBodyPropertiesGPU& body = model.bodies[globalBody];
        const BodyKinematics& bodyKinematics =
            kinematics[localBody];
        const double mass = body.massAndInverseMass.x;
        const Vec3 bodyLinearMomentum =
            bodyKinematics.centerOfMassLinearVelocity * mass;
        const Vec3 bodyAngularMomentum =
            bodyKinematics.inertiaWorld *
                bodyKinematics.angularVelocity +
            cross(
                bodyKinematics.centerOfMassPosition,
                bodyLinearMomentum
            );
        result.kineticEnergy +=
            0.5 * mass *
                normSquared(
                    bodyKinematics.centerOfMassLinearVelocity
                ) +
            0.5 *
                dot(
                    bodyKinematics.angularVelocity,
                    bodyKinematics.inertiaWorld *
                        bodyKinematics.angularVelocity
                );
        result.potentialEnergy -=
            mass *
            dot(gravity, bodyKinematics.centerOfMassPosition);
        linearMomentum += bodyLinearMomentum;
        angularMomentum += bodyAngularMomentum;
    }
    const MRArticulationGPU& articulation =
        *topology.articulation;
    for (std::size_t localDof = 0u;
         localDof < v.size();
         ++localDof) {
        const double armature = model.dofs[
            articulation.vOffset + localDof
        ].drive.z;
        result.kineticEnergy +=
            0.5 * armature * v[localDof] * v[localDof];
    }
    result.totalEnergy =
        result.kineticEnergy + result.potentialEnergy;
    result.linearMomentum = {
        linearMomentum.x,
        linearMomentum.y,
        linearMomentum.z,
    };
    result.angularMomentum = {
        angularMomentum.x,
        angularMomentum.y,
        angularMomentum.z,
    };
    if (!finite(result.kineticEnergy) ||
        !finite(result.potentialEnergy) ||
        !finite(result.totalEnergy) ||
        !finite(linearMomentum) ||
        !finite(angularMomentum)) {
        diagnostics.status =
            ArticulatedDynamicsStatus::nonfiniteResult;
        return diagnostics;
    }
    invariants = result;
    return diagnostics;
}

} // namespace metalrobo
