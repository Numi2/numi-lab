#include "metalrobo/Model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace metalrobo {
namespace {

using Scalar = double;

struct Vec3 {
    Scalar x = 0.0;
    Scalar y = 0.0;
    Scalar z = 0.0;
};

Vec3 operator+(const Vec3 a, const Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(const Vec3 a, const Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator*(const Vec3 value, const Scalar scale) {
    return {value.x * scale, value.y * scale, value.z * scale};
}

Vec3 operator/(const Vec3 value, const Scalar scale) {
    return {value.x / scale, value.y / scale, value.z / scale};
}

Vec3& operator+=(Vec3& target, const Vec3 value) {
    target = target + value;
    return target;
}

Scalar dot(const Vec3 a, const Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(const Vec3 a, const Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

Scalar length(const Vec3 value) {
    return std::sqrt(dot(value, value));
}

Vec3 normalized(const Vec3 value) {
    const Scalar magnitude = length(value);
    if (magnitude <= 1.0e-12) {
        throw std::invalid_argument("cannot normalize a zero vector");
    }
    return value / magnitude;
}

Vec3 xyz(const mr_float4 value) {
    return {value.x, value.y, value.z};
}

struct Mat3 {
    Scalar m[3][3]{};
};

Mat3 identity() {
    Mat3 result{};
    result.m[0][0] = 1.0;
    result.m[1][1] = 1.0;
    result.m[2][2] = 1.0;
    return result;
}

Mat3 transpose(const Mat3& matrix) {
    Mat3 result{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result.m[row][column] = matrix.m[column][row];
        }
    }
    return result;
}

Mat3 operator*(const Mat3& left, const Mat3& right) {
    Mat3 result{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            for (int inner = 0; inner < 3; ++inner) {
                result.m[row][column] +=
                    left.m[row][inner] * right.m[inner][column];
            }
        }
    }
    return result;
}

Vec3 operator*(const Mat3& matrix, const Vec3 vector) {
    return {
        matrix.m[0][0] * vector.x + matrix.m[0][1] * vector.y +
            matrix.m[0][2] * vector.z,
        matrix.m[1][0] * vector.x + matrix.m[1][1] * vector.y +
            matrix.m[1][2] * vector.z,
        matrix.m[2][0] * vector.x + matrix.m[2][1] * vector.y +
            matrix.m[2][2] * vector.z,
    };
}

Mat3 quaternionMatrix(const mr_float4 quaternion) {
    Scalar x = quaternion.x;
    Scalar y = quaternion.y;
    Scalar z = quaternion.z;
    Scalar w = quaternion.w;
    const Scalar inverseNorm =
        1.0 / std::sqrt(x * x + y * y + z * z + w * w);
    x *= inverseNorm;
    y *= inverseNorm;
    z *= inverseNorm;
    w *= inverseNorm;

    Mat3 result{};
    result.m[0][0] = 1.0 - 2.0 * (y * y + z * z);
    result.m[0][1] = 2.0 * (x * y - z * w);
    result.m[0][2] = 2.0 * (x * z + y * w);
    result.m[1][0] = 2.0 * (x * y + z * w);
    result.m[1][1] = 1.0 - 2.0 * (x * x + z * z);
    result.m[1][2] = 2.0 * (y * z - x * w);
    result.m[2][0] = 2.0 * (x * z - y * w);
    result.m[2][1] = 2.0 * (y * z + x * w);
    result.m[2][2] = 1.0 - 2.0 * (x * x + y * y);
    return result;
}

Mat3 axisAngle(const Vec3 unitAxis, const Scalar angle) {
    const Scalar cosine = std::cos(angle);
    const Scalar sine = std::sin(angle);
    const Scalar oneMinusCosine = 1.0 - cosine;
    const Scalar x = unitAxis.x;
    const Scalar y = unitAxis.y;
    const Scalar z = unitAxis.z;

    Mat3 result{};
    result.m[0][0] = cosine + x * x * oneMinusCosine;
    result.m[0][1] = x * y * oneMinusCosine - z * sine;
    result.m[0][2] = x * z * oneMinusCosine + y * sine;
    result.m[1][0] = y * x * oneMinusCosine + z * sine;
    result.m[1][1] = cosine + y * y * oneMinusCosine;
    result.m[1][2] = y * z * oneMinusCosine - x * sine;
    result.m[2][0] = z * x * oneMinusCosine - y * sine;
    result.m[2][1] = z * y * oneMinusCosine + x * sine;
    result.m[2][2] = cosine + z * z * oneMinusCosine;
    return result;
}

Mat3 bodyInertia(const MRLinkGPU& link) {
    Mat3 result{};
    result.m[0][0] = link.inertiaRow0.x;
    result.m[0][1] = link.inertiaRow0.y;
    result.m[0][2] = link.inertiaRow0.z;
    result.m[1][0] = link.inertiaRow1.x;
    result.m[1][1] = link.inertiaRow1.y;
    result.m[1][2] = link.inertiaRow1.z;
    result.m[2][0] = link.inertiaRow2.x;
    result.m[2][1] = link.inertiaRow2.y;
    result.m[2][2] = link.inertiaRow2.z;
    return result;
}

mr_float4 matrixQuaternion(const Mat3& matrix) {
    Scalar x = 0.0;
    Scalar y = 0.0;
    Scalar z = 0.0;
    Scalar w = 1.0;
    const Scalar trace =
        matrix.m[0][0] + matrix.m[1][1] + matrix.m[2][2];
    if (trace > 0.0) {
        const Scalar scale = 2.0 * std::sqrt(trace + 1.0);
        w = 0.25 * scale;
        x = (matrix.m[2][1] - matrix.m[1][2]) / scale;
        y = (matrix.m[0][2] - matrix.m[2][0]) / scale;
        z = (matrix.m[1][0] - matrix.m[0][1]) / scale;
    } else if (matrix.m[0][0] > matrix.m[1][1] &&
               matrix.m[0][0] > matrix.m[2][2]) {
        const Scalar scale =
            2.0 * std::sqrt(1.0 + matrix.m[0][0] - matrix.m[1][1] -
                            matrix.m[2][2]);
        w = (matrix.m[2][1] - matrix.m[1][2]) / scale;
        x = 0.25 * scale;
        y = (matrix.m[0][1] + matrix.m[1][0]) / scale;
        z = (matrix.m[0][2] + matrix.m[2][0]) / scale;
    } else if (matrix.m[1][1] > matrix.m[2][2]) {
        const Scalar scale =
            2.0 * std::sqrt(1.0 + matrix.m[1][1] - matrix.m[0][0] -
                            matrix.m[2][2]);
        w = (matrix.m[0][2] - matrix.m[2][0]) / scale;
        x = (matrix.m[0][1] + matrix.m[1][0]) / scale;
        y = 0.25 * scale;
        z = (matrix.m[1][2] + matrix.m[2][1]) / scale;
    } else {
        const Scalar scale =
            2.0 * std::sqrt(1.0 + matrix.m[2][2] - matrix.m[0][0] -
                            matrix.m[1][1]);
        w = (matrix.m[1][0] - matrix.m[0][1]) / scale;
        x = (matrix.m[0][2] + matrix.m[2][0]) / scale;
        y = (matrix.m[1][2] + matrix.m[2][1]) / scale;
        z = 0.25 * scale;
    }
    const Scalar inverseNorm =
        1.0 / std::sqrt(x * x + y * y + z * z + w * w);
    return {
        static_cast<float>(x * inverseNorm),
        static_cast<float>(y * inverseNorm),
        static_cast<float>(z * inverseNorm),
        static_cast<float>(w * inverseNorm),
    };
}

struct Kinematics {
    std::vector<Vec3> position;
    std::vector<Mat3> rotation;
    std::vector<Vec3> linearVelocity;
    std::vector<Vec3> angularVelocity;
    std::vector<Vec3> linearAcceleration;
    std::vector<Vec3> angularAcceleration;
    std::vector<Vec3> axisWorld;
};

Kinematics computeKinematics(
    const Model& model,
    const std::vector<float>& q,
    const std::vector<float>& qd,
    const std::vector<float>& qdd
) {
    const std::size_t linkCount = model.links.size();
    Kinematics result;
    result.position.resize(linkCount);
    result.rotation.assign(linkCount, identity());
    result.linearVelocity.resize(linkCount);
    result.angularVelocity.resize(linkCount);
    result.linearAcceleration.resize(linkCount);
    result.angularAcceleration.resize(linkCount);
    result.axisWorld.resize(model.joints.size());

    for (std::size_t index = 0; index < model.joints.size(); ++index) {
        const MRJointGPU& joint = model.joints[index];
        const std::size_t parent = static_cast<std::size_t>(joint.parentLink);
        const std::size_t child = joint.childLink;
        const Mat3 originRotation = quaternionMatrix(joint.parentRotation);
        const Mat3 jointFrameWorld =
            result.rotation[parent] * originRotation;
        const Vec3 axis = normalized(xyz(joint.axis));
        const Vec3 axisWorld = jointFrameWorld * axis;
        result.axisWorld[index] = axisWorld;

        const Vec3 fixedOffsetWorld =
            result.rotation[parent] * xyz(joint.parentOffset);
        if (joint.jointType == 0u) {
            result.position[child] =
                result.position[parent] + fixedOffsetWorld;
            result.rotation[child] =
                jointFrameWorld * axisAngle(axis, q[index]);
            result.linearVelocity[child] =
                result.linearVelocity[parent] +
                cross(result.angularVelocity[parent], fixedOffsetWorld);
            result.angularVelocity[child] =
                result.angularVelocity[parent] + axisWorld * qd[index];
            result.linearAcceleration[child] =
                result.linearAcceleration[parent] +
                cross(
                    result.angularAcceleration[parent],
                    fixedOffsetWorld
                ) +
                cross(
                    result.angularVelocity[parent],
                    cross(
                        result.angularVelocity[parent],
                        fixedOffsetWorld
                    )
                );
            result.angularAcceleration[child] =
                result.angularAcceleration[parent] +
                axisWorld * qdd[index] +
                cross(
                    result.angularVelocity[parent],
                    axisWorld * qd[index]
                );
        } else {
            const Vec3 slidingOffset = axisWorld * q[index];
            const Vec3 completeOffset = fixedOffsetWorld + slidingOffset;
            result.position[child] =
                result.position[parent] + completeOffset;
            result.rotation[child] = jointFrameWorld;
            result.angularVelocity[child] =
                result.angularVelocity[parent];
            result.linearVelocity[child] =
                result.linearVelocity[parent] +
                cross(
                    result.angularVelocity[parent],
                    completeOffset
                ) +
                axisWorld * qd[index];
            result.angularAcceleration[child] =
                result.angularAcceleration[parent];
            result.linearAcceleration[child] =
                result.linearAcceleration[parent] +
                cross(
                    result.angularAcceleration[parent],
                    completeOffset
                ) +
                cross(
                    result.angularVelocity[parent],
                    cross(
                        result.angularVelocity[parent],
                        completeOffset
                    )
                ) +
                cross(
                    result.angularVelocity[parent],
                    axisWorld * (2.0 * qd[index])
                ) +
                axisWorld * qdd[index];
        }
    }
    return result;
}

std::vector<Scalar> inverseDynamics(
    const Model& model,
    const std::vector<float>& q,
    const std::vector<float>& qd,
    const std::vector<float>& qdd,
    const Vec3 gravity
) {
    const Kinematics kinematics = computeKinematics(model, q, qd, qdd);
    std::vector<Vec3> force(model.links.size());
    std::vector<Vec3> moment(model.links.size());
    std::vector<Scalar> generalized(model.joints.size(), 0.0);

    // Link 0 is fixed to the world. Dynamic links are each joint's child.
    for (const MRJointGPU& joint : model.joints) {
        const std::size_t child = joint.childLink;
        const MRLinkGPU& link = model.links[child];
        const Scalar mass = link.massAndCOMX.x;
        const Vec3 comWorld =
            kinematics.rotation[child] *
            Vec3{
                link.massAndCOMX.y,
                link.massAndCOMX.z,
                link.massAndCOMX.w,
            };
        const Vec3 comAcceleration =
            kinematics.linearAcceleration[child] +
            cross(kinematics.angularAcceleration[child], comWorld) +
            cross(
                kinematics.angularVelocity[child],
                cross(kinematics.angularVelocity[child], comWorld)
            );
        force[child] = (comAcceleration - gravity) * mass;

        const Mat3 worldInertia =
            kinematics.rotation[child] * bodyInertia(link) *
            transpose(kinematics.rotation[child]);
        const Vec3 angularMomentum =
            worldInertia * kinematics.angularVelocity[child];
        moment[child] =
            worldInertia * kinematics.angularAcceleration[child] +
            cross(kinematics.angularVelocity[child], angularMomentum) +
            cross(comWorld, force[child]);
    }

    for (std::size_t reverse = model.joints.size(); reverse-- > 0;) {
        const MRJointGPU& joint = model.joints[reverse];
        const std::size_t parent = static_cast<std::size_t>(joint.parentLink);
        const std::size_t child = joint.childLink;
        generalized[reverse] =
            joint.jointType == 0u
            ? dot(kinematics.axisWorld[reverse], moment[child])
            : dot(kinematics.axisWorld[reverse], force[child]);
        const Vec3 parentToChild =
            kinematics.position[child] - kinematics.position[parent];
        force[parent] += force[child];
        moment[parent] +=
            moment[child] + cross(parentToChild, force[child]);
    }
    return generalized;
}

std::vector<Scalar> contactForces(
    const Model& model,
    const std::vector<float>& q,
    const std::vector<float>& qd
) {
    std::vector<float> zero(model.joints.size(), 0.0f);
    const Kinematics kinematics = computeKinematics(model, q, qd, zero);
    std::vector<Scalar> generalized(model.joints.size(), 0.0);

    Vec3 planeNormal = xyz(model.gpu.groundPlane);
    const Scalar normalMagnitude = length(planeNormal);
    planeNormal = planeNormal / normalMagnitude;
    const Scalar planeOffset = model.gpu.groundPlane.w / normalMagnitude;

    for (const MRColliderGPU& collider : model.colliders) {
        if (collider.linkIndex <= 0) {
            continue;
        }
        const std::size_t linkIndex =
            static_cast<std::size_t>(collider.linkIndex);
        const Mat3& rotation = kinematics.rotation[linkIndex];
        Vec3 center =
            kinematics.position[linkIndex] +
            rotation * xyz(collider.centerAndRadius);
        Vec3 contactPoint = center;

        if (collider.shapeType == MR_SHAPE_SPHERE) {
            contactPoint =
                center - planeNormal * collider.centerAndRadius.w;
        } else if (collider.shapeType == MR_SHAPE_CAPSULE) {
            const Vec3 second =
                kinematics.position[linkIndex] +
                rotation * xyz(collider.extent);
            if (dot(planeNormal, second) < dot(planeNormal, center)) {
                center = second;
            }
            contactPoint =
                center - planeNormal * collider.centerAndRadius.w;
        } else {
            const Vec3 localNormal = transpose(rotation) * planeNormal;
            const Vec3 localSupport{
                localNormal.x >= 0.0 ? -collider.extent.x
                                     : collider.extent.x,
                localNormal.y >= 0.0 ? -collider.extent.y
                                     : collider.extent.y,
                localNormal.z >= 0.0 ? -collider.extent.z
                                     : collider.extent.z,
            };
            contactPoint = center + rotation * localSupport;
        }

        const Scalar penetration =
            planeOffset - dot(planeNormal, contactPoint);
        if (penetration <= 0.0) {
            continue;
        }
        const Vec3 contactOffset =
            contactPoint - kinematics.position[linkIndex];
        const Vec3 pointVelocity =
            kinematics.linearVelocity[linkIndex] +
            cross(kinematics.angularVelocity[linkIndex], contactOffset);
        const Scalar normalVelocity = dot(planeNormal, pointVelocity);
        Scalar normalForce =
            collider.material.z * penetration -
            collider.material.w * normalVelocity;
        normalForce = std::clamp(normalForce, 0.0, 5000.0);
        if (normalForce <= 0.0) {
            continue;
        }

        const Vec3 tangentialVelocity =
            pointVelocity - planeNormal * normalVelocity;
        const Scalar tangentialSpeed = length(tangentialVelocity);
        Vec3 force = planeNormal * normalForce;
        if (tangentialSpeed > 1.0e-9 && collider.material.x > 0.0f) {
            const Scalar regularizedSpeed =
                std::sqrt(tangentialSpeed * tangentialSpeed + 0.0004);
            force += tangentialVelocity *
                (-collider.material.x * normalForce / regularizedSpeed);
        }

        std::size_t currentLink = linkIndex;
        while (currentLink > 0) {
            const std::size_t jointIndex = currentLink - 1;
            const MRJointGPU& joint = model.joints[jointIndex];
            if (joint.jointType == 0u) {
                generalized[jointIndex] += dot(
                    kinematics.axisWorld[jointIndex],
                    cross(
                        contactPoint -
                            kinematics.position[joint.childLink],
                        force
                    )
                );
            } else {
                generalized[jointIndex] +=
                    dot(kinematics.axisWorld[jointIndex], force);
            }
            currentLink = static_cast<std::size_t>(joint.parentLink);
        }
    }
    return generalized;
}

std::vector<Scalar> solveForwardDynamics(
    const Model& model,
    const std::vector<float>& q,
    const std::vector<float>& qd,
    const std::vector<Scalar>& applied
) {
    const std::size_t dof = model.joints.size();
    const Vec3 gravity = xyz(model.gpu.gravityAndTimestep);
    std::vector<float> zero(dof, 0.0f);
    const std::vector<Scalar> bias =
        inverseDynamics(model, q, qd, zero, gravity);

    // Exact reduced-coordinate dynamics. RNEA columns construct M(q), then a
    // dense Cholesky solve computes qdd = M^-1(tau - C - g). This is the
    // small-DOF CPU oracle equivalent of the O(n) ABA used by Metal.
    std::vector<Scalar> massMatrix(dof * dof, 0.0);
    for (std::size_t column = 0; column < dof; ++column) {
        std::vector<float> unitAcceleration(dof, 0.0f);
        unitAcceleration[column] = 1.0f;
        const std::vector<Scalar> response =
            inverseDynamics(model, q, zero, unitAcceleration, {});
        for (std::size_t row = 0; row < dof; ++row) {
            massMatrix[row * dof + column] = response[row];
        }
    }
    for (std::size_t row = 0; row < dof; ++row) {
        for (std::size_t column = row + 1; column < dof; ++column) {
            const Scalar symmetric =
                0.5 * (massMatrix[row * dof + column] +
                       massMatrix[column * dof + row]);
            massMatrix[row * dof + column] = symmetric;
            massMatrix[column * dof + row] = symmetric;
        }
        massMatrix[row * dof + row] +=
            model.joints[row].drive.w + 1.0e-8;
    }

    std::vector<Scalar> lower(dof * dof, 0.0);
    for (std::size_t row = 0; row < dof; ++row) {
        for (std::size_t column = 0; column <= row; ++column) {
            Scalar value = massMatrix[row * dof + column];
            for (std::size_t inner = 0; inner < column; ++inner) {
                value -= lower[row * dof + inner] *
                    lower[column * dof + inner];
            }
            if (row == column) {
                if (value <= 1.0e-12 || !std::isfinite(value)) {
                    throw std::runtime_error(
                        "articulated mass matrix is not positive definite"
                    );
                }
                lower[row * dof + column] = std::sqrt(value);
            } else {
                lower[row * dof + column] =
                    value / lower[column * dof + column];
            }
        }
    }

    std::vector<Scalar> intermediate(dof, 0.0);
    for (std::size_t row = 0; row < dof; ++row) {
        Scalar value = applied[row] - bias[row];
        for (std::size_t column = 0; column < row; ++column) {
            value -= lower[row * dof + column] * intermediate[column];
        }
        intermediate[row] = value / lower[row * dof + row];
    }
    std::vector<Scalar> acceleration(dof, 0.0);
    for (std::size_t reverse = dof; reverse-- > 0;) {
        Scalar value = intermediate[reverse];
        for (std::size_t row = reverse + 1; row < dof; ++row) {
            value -= lower[row * dof + reverse] * acceleration[row];
        }
        acceleration[reverse] =
            value / lower[reverse * dof + reverse];
        if (!std::isfinite(acceleration[reverse])) {
            throw std::runtime_error(
                "forward dynamics produced a non-finite acceleration"
            );
        }
    }
    return acceleration;
}

void updateBodyPoses(const Model& model, CpuState& state) {
    std::vector<float> zero(model.joints.size(), 0.0f);
    const Kinematics kinematics =
        computeKinematics(model, state.q, zero, zero);
    state.bodyPositions.resize(model.links.size());
    state.bodyRotations.resize(model.links.size());
    for (std::size_t index = 0; index < model.links.size(); ++index) {
        const Vec3 position = kinematics.position[index];
        state.bodyPositions[index] = {
            static_cast<float>(position.x),
            static_cast<float>(position.y),
            static_cast<float>(position.z),
            1.0f,
        };
        state.bodyRotations[index] =
            matrixQuaternion(kinematics.rotation[index]);
    }
}

class SplitMix64 {
public:
    explicit SplitMix64(const std::uint64_t seed) : state_(seed) {}

    Scalar unit() {
        std::uint64_t value = (state_ += 0x9e3779b97f4a7c15ULL);
        value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27u)) * 0x94d049bb133111ebULL;
        value ^= value >> 31u;
        return static_cast<Scalar>(value >> 11u) *
            (1.0 / 9007199254740992.0);
    }

private:
    std::uint64_t state_;
};

} // namespace

void resetCpuState(
    const Model& model,
    CpuState& state,
    const std::uint64_t seed
) {
    std::string reason;
    if (!model.valid(&reason)) {
        throw std::invalid_argument("invalid model: " + reason);
    }

    SplitMix64 random(seed);
    const std::size_t dof = model.joints.size();
    state.q.resize(dof);
    state.qd.assign(dof, 0.0f);
    state.qdd.assign(dof, 0.0f);
    state.torque.assign(dof, 0.0f);
    for (std::size_t index = 0; index < dof; ++index) {
        const MRJointGPU& joint = model.joints[index];
        const Scalar noise = (2.0 * random.unit() - 1.0) * 0.025;
        state.q[index] = static_cast<float>(std::clamp(
            static_cast<Scalar>(model.homePosition[index]) + noise,
            static_cast<Scalar>(joint.limits.x),
            static_cast<Scalar>(joint.limits.y)
        ));
    }
    const mr_float4 lower = model.gpu.targetLowerAndRadius;
    const mr_float4 upper = model.gpu.targetUpperAndBonus;
    state.target = {
        static_cast<float>(lower.x + random.unit() * (upper.x - lower.x)),
        static_cast<float>(lower.y + random.unit() * (upper.y - lower.y)),
        static_cast<float>(lower.z + random.unit() * (upper.z - lower.z)),
        1.0f,
    };
    state.step = 0u;
    updateBodyPoses(model, state);
}

void stepCpu(
    const Model& model,
    CpuState& state,
    const std::span<const float> normalizedActions
) {
    const std::size_t dof = model.joints.size();
    if (normalizedActions.size() != dof) {
        throw std::invalid_argument(
            "normalized action count must equal model actionCount"
        );
    }
    if (state.q.size() != dof || state.qd.size() != dof ||
        state.qdd.size() != dof || state.torque.size() != dof) {
        throw std::invalid_argument(
            "CpuState must be initialized by resetCpuState"
        );
    }
    for (const float action : normalizedActions) {
        if (!std::isfinite(action)) {
            throw std::invalid_argument("normalized action is not finite");
        }
    }

    const Scalar controlTimestep = model.gpu.gravityAndTimestep.w;
    const Scalar timestep =
        controlTimestep / static_cast<Scalar>(model.gpu.substeps);

    for (mr_u32 substep = 0; substep < model.gpu.substeps; ++substep) {
        std::vector<Scalar> applied =
            contactForces(model, state.q, state.qd);
        for (std::size_t index = 0; index < dof; ++index) {
            const MRJointGPU& joint = model.joints[index];
            const Scalar action =
                std::clamp(static_cast<Scalar>(normalizedActions[index]),
                           -1.0,
                           1.0);
            const Scalar target = std::clamp(
                static_cast<Scalar>(model.homePosition[index]) +
                    action * joint.drive.z,
                static_cast<Scalar>(joint.limits.x),
                static_cast<Scalar>(joint.limits.y)
            );
            Scalar actuator =
                joint.drive.x * (target - state.q[index]) -
                joint.drive.y * state.qd[index];
            actuator = std::clamp(
                actuator,
                -static_cast<Scalar>(joint.limits.w),
                static_cast<Scalar>(joint.limits.w)
            );
            state.torque[index] = static_cast<float>(actuator);
            applied[index] += actuator;

            const Scalar limitStiffness =
                std::max<Scalar>(500.0, 4.0 * joint.drive.x);
            const Scalar limitDamping = 0.8 * 2.0 *
                std::sqrt(
                    limitStiffness *
                    (static_cast<Scalar>(joint.drive.w) + 0.05)
                );
            if (state.q[index] < joint.limits.x) {
                applied[index] +=
                    limitStiffness * (joint.limits.x - state.q[index]);
                if (state.qd[index] < 0.0f) {
                    applied[index] -= limitDamping * state.qd[index];
                }
            } else if (state.q[index] > joint.limits.y) {
                applied[index] -=
                    limitStiffness * (state.q[index] - joint.limits.y);
                if (state.qd[index] > 0.0f) {
                    applied[index] -= limitDamping * state.qd[index];
                }
            }
        }

        const std::vector<Scalar> acceleration =
            solveForwardDynamics(model, state.q, state.qd, applied);
        for (std::size_t index = 0; index < dof; ++index) {
            const MRJointGPU& joint = model.joints[index];
            state.qdd[index] = static_cast<float>(acceleration[index]);
            state.qd[index] = static_cast<float>(std::clamp(
                static_cast<Scalar>(state.qd[index]) +
                    acceleration[index] * timestep,
                -static_cast<Scalar>(joint.limits.z),
                static_cast<Scalar>(joint.limits.z)
            ));
            state.q[index] +=
                static_cast<float>(state.qd[index] * timestep);

            if (state.q[index] < joint.limits.x) {
                state.q[index] = joint.limits.x;
                if (state.qd[index] < 0.0f) {
                    state.qd[index] = 0.0f;
                }
            } else if (state.q[index] > joint.limits.y) {
                state.q[index] = joint.limits.y;
                if (state.qd[index] > 0.0f) {
                    state.qd[index] = 0.0f;
                }
            }
            if (!std::isfinite(state.q[index]) ||
                !std::isfinite(state.qd[index])) {
                throw std::runtime_error(
                    "semi-implicit integration produced a non-finite state"
                );
            }
        }
    }

    ++state.step;
    updateBodyPoses(model, state);
}

} // namespace metalrobo
