#include "metalrobo/FreeBodyDynamics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
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

Vec3 operator+(const Vec3 a, const Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(const Vec3 a, const Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator*(const Vec3 value, const double scale) {
    return {value.x * scale, value.y * scale, value.z * scale};
}

Vec3 operator/(const Vec3 value, const double scale) {
    return {value.x / scale, value.y / scale, value.z / scale};
}

Vec3& operator+=(Vec3& left, const Vec3 right) {
    left = left + right;
    return left;
}

double dot(const Vec3 a, const Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(const Vec3 a, const Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

double norm(const Vec3 value) {
    return std::sqrt(dot(value, value));
}

bool finite(const Vec3 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
        std::isfinite(value.z);
}

bool finite(const mr_float4 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
        std::isfinite(value.z) && std::isfinite(value.w);
}

bool representableAsFloat(const double value) {
    return std::isfinite(value) &&
        std::abs(value) <= std::numeric_limits<float>::max();
}

bool representableAsFloat(const Vec3 value) {
    return representableAsFloat(value.x) &&
        representableAsFloat(value.y) &&
        representableAsFloat(value.z);
}

Vec3 xyz(const mr_float4 value) {
    return {value.x, value.y, value.z};
}

mr_float4 f4(const Vec3 value, const float w = 0.0f) {
    return {
        static_cast<float>(value.x),
        static_cast<float>(value.y),
        static_cast<float>(value.z),
        w,
    };
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

Quaternion normalized(const Quaternion value) {
    const double magnitude = std::sqrt(
        value.x * value.x + value.y * value.y + value.z * value.z +
        value.w * value.w
    );
    if (!(magnitude > 1.0e-15) || !std::isfinite(magnitude)) {
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

Quaternion exponentialQuaternion(const Vec3 rotationVector) {
    const double angle = norm(rotationVector);
    if (angle < 1.0e-10) {
        const double angleSquared = angle * angle;
        const double vectorScale = 0.5 - angleSquared / 48.0;
        return normalized({
            rotationVector.x * vectorScale,
            rotationVector.y * vectorScale,
            rotationVector.z * vectorScale,
            1.0 - angleSquared / 8.0,
        });
    }
    const double halfAngle = 0.5 * angle;
    const double vectorScale = std::sin(halfAngle) / angle;
    return {
        rotationVector.x * vectorScale,
        rotationVector.y * vectorScale,
        rotationVector.z * vectorScale,
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

Vec3 solve3x3(Mat3 matrixValue, Vec3 right, bool& succeeded) {
    double augmented[3][4]{
        {
            matrixValue.m[0][0],
            matrixValue.m[0][1],
            matrixValue.m[0][2],
            right.x,
        },
        {
            matrixValue.m[1][0],
            matrixValue.m[1][1],
            matrixValue.m[1][2],
            right.y,
        },
        {
            matrixValue.m[2][0],
            matrixValue.m[2][1],
            matrixValue.m[2][2],
            right.z,
        },
    };
    for (std::size_t column = 0; column < 3; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < 3; ++row) {
            if (std::abs(augmented[row][column]) >
                std::abs(augmented[pivot][column])) {
                pivot = row;
            }
        }
        if (std::abs(augmented[pivot][column]) < 1.0e-15) {
            succeeded = false;
            return {};
        }
        if (pivot != column) {
            for (std::size_t entry = column; entry < 4; ++entry) {
                std::swap(
                    augmented[column][entry],
                    augmented[pivot][entry]
                );
            }
        }
        const double inversePivot = 1.0 / augmented[column][column];
        for (std::size_t entry = column; entry < 4; ++entry) {
            augmented[column][entry] *= inversePivot;
        }
        for (std::size_t row = 0; row < 3; ++row) {
            if (row == column) {
                continue;
            }
            const double scale = augmented[row][column];
            for (std::size_t entry = column; entry < 4; ++entry) {
                augmented[row][entry] -= scale * augmented[column][entry];
            }
        }
    }
    succeeded = true;
    return {augmented[0][3], augmented[1][3], augmented[2][3]};
}

Vec3 angularDerivative(
    const Vec3 omega,
    const Vec3 torque,
    const Mat3& inertia,
    const Mat3& inverseInertia
) {
    return inverseInertia * (torque - cross(omega, inertia * omega));
}

struct MidpointResult {
    Vec3 omega;
    std::uint32_t iterations = 0;
    double residual = 0.0;
    bool converged = false;
};

MidpointResult implicitMidpointAngularVelocity(
    const Vec3 omega0,
    const Vec3 torqueBody,
    const Mat3& inertia,
    const Mat3& inverseInertia,
    const FreeBodyIntegratorConfig& config
) {
    MidpointResult result;
    result.omega =
        omega0 +
        angularDerivative(omega0, torqueBody, inertia, inverseInertia) *
            config.timestep;

    for (std::uint32_t iteration = 0;
         iteration < config.nonlinearIterations;
         ++iteration) {
        const Vec3 midpoint = (omega0 + result.omega) * 0.5;
        const Vec3 residualVector =
            result.omega - omega0 -
            angularDerivative(
                midpoint,
                torqueBody,
                inertia,
                inverseInertia
            ) * config.timestep;
        result.residual = norm(residualVector);
        result.iterations = iteration + 1u;
        if (result.residual <= config.nonlinearTolerance) {
            result.converged = true;
            return result;
        }

        Mat3 jacobian{};
        for (std::size_t column = 0; column < 3; ++column) {
            Vec3 basis{};
            (&basis.x)[column] = 1.0;
            const Vec3 derivative =
                inverseInertia *
                (
                    cross(basis, inertia * midpoint) +
                    cross(midpoint, inertia * basis)
                ) *
                (-0.5 * config.timestep);
            for (std::size_t row = 0; row < 3; ++row) {
                (&jacobian.m[row][0])[column] =
                    (row == column ? 1.0 : 0.0) -
                    (&derivative.x)[row];
            }
        }
        bool solved = false;
        const Vec3 step = solve3x3(
            jacobian,
            residualVector * -1.0,
            solved
        );
        if (!solved || !finite(step)) {
            return result;
        }
        result.omega += step;
    }
    return result;
}

bool writeInverseInertiaWorld(
    MRBodyStateGPU& state,
    const Mat3& inverseInertiaBody,
    const Quaternion orientation
) {
    const Mat3 rotation = rotationMatrix(orientation);
    const Mat3 world =
        rotation * inverseInertiaBody * transpose(rotation);
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            if (!representableAsFloat(world.m[row][column])) {
                return false;
            }
        }
    }
    state.inverseInertiaWorldRow0 = {
        static_cast<float>(world.m[0][0]),
        static_cast<float>(world.m[0][1]),
        static_cast<float>(world.m[0][2]),
        0.0f,
    };
    state.inverseInertiaWorldRow1 = {
        static_cast<float>(world.m[1][0]),
        static_cast<float>(world.m[1][1]),
        static_cast<float>(world.m[1][2]),
        0.0f,
    };
    state.inverseInertiaWorldRow2 = {
        static_cast<float>(world.m[2][0]),
        static_cast<float>(world.m[2][1]),
        static_cast<float>(world.m[2][2]),
        0.0f,
    };
    return true;
}

Vec3 clampMagnitude(const Vec3 value, const double maximum) {
    const double magnitude = norm(value);
    if (!(maximum > 0.0) || magnitude <= maximum) {
        return value;
    }
    return value * (maximum / magnitude);
}

} // namespace

FreeBodyIntegratorDiagnostics integrateFreeBodies(
    const std::span<const MRBodyPropertiesGPU> properties,
    const std::span<MRBodyStateGPU> states,
    const std::span<const BodyWrench> wrenches,
    const FreeBodyIntegratorConfig& config
) {
    FreeBodyIntegratorDiagnostics diagnostics;
    if (properties.size() != states.size() ||
        (!wrenches.empty() && wrenches.size() != states.size()) ||
        !(config.timestep > 0.0) ||
        !std::isfinite(config.timestep) ||
        config.nonlinearIterations == 0u ||
        !(config.nonlinearTolerance > 0.0) ||
        !std::isfinite(config.nonlinearTolerance) ||
        static_cast<std::uint32_t>(config.integrator) >
            static_cast<std::uint32_t>(
                FreeBodyIntegrator::implicitMidpoint
            ) ||
        !finite(config.gravity)) {
        diagnostics.code = MR_STEP_NONFINITE_INPUT;
        return diagnostics;
    }

    const Vec3 gravity = xyz(config.gravity);
    if (!finite(gravity)) {
        diagnostics.code = MR_STEP_NONFINITE_INPUT;
        return diagnostics;
    }

    // Preflight the complete batch so the direct CPU API is transactional.
    for (std::size_t index = 0u; index < states.size(); ++index) {
        const MRBodyPropertiesGPU& body = properties[index];
        const MRBodyStateGPU& state = states[index];
        const std::uint32_t motionType = state.flagsAndIndices[0];
        if (motionType > MR_MOTION_DYNAMIC ||
            body.motionType != motionType ||
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
            diagnostics.code = MR_STEP_NONFINITE_INPUT;
            return diagnostics;
        }
        if (motionType == MR_MOTION_STATIC) {
            continue;
        }
        const double orientationNormSquared =
            static_cast<double>(state.orientation.x) *
                state.orientation.x +
            static_cast<double>(state.orientation.y) *
                state.orientation.y +
            static_cast<double>(state.orientation.z) *
                state.orientation.z +
            static_cast<double>(state.orientation.w) *
                state.orientation.w;
        if (!finite(state.position) ||
            !finite(state.orientation) ||
            !finite(state.linearVelocityAndInverseMass) ||
            !finite(state.angularVelocity) ||
            !(orientationNormSquared > 1.0e-15) ||
            !std::isfinite(orientationNormSquared)) {
            diagnostics.code = MR_STEP_NONFINITE_INPUT;
            return diagnostics;
        }
        if (motionType == MR_MOTION_DYNAMIC) {
            const BodyWrench wrench =
                wrenches.empty() ? BodyWrench{} : wrenches[index];
            if (!(state.linearVelocityAndInverseMass.w > 0.0f) ||
                !(body.massAndInverseMass.x > 0.0f) ||
                !(body.massAndInverseMass.y > 0.0f) ||
                !finite(wrench.force) ||
                !finite(wrench.torque)) {
                diagnostics.code = MR_STEP_NONFINITE_INPUT;
                return diagnostics;
            }
        }
    }

    std::vector<MRBodyStateGPU> working(states.begin(), states.end());
    for (std::size_t index = 0; index < working.size(); ++index) {
        MRBodyStateGPU& state = working[index];
        const MRBodyPropertiesGPU& body = properties[index];
        const mr_u32 motionType = state.flagsAndIndices[0];
        if (motionType == MR_MOTION_STATIC) {
            continue;
        }

        Quaternion orientation{
            state.orientation.x,
            state.orientation.y,
            state.orientation.z,
            state.orientation.w,
        };
        orientation = normalized(orientation);
        Vec3 linearVelocity = xyz(state.linearVelocityAndInverseMass);
        Vec3 angularVelocityWorld = xyz(state.angularVelocity);
        Vec3 position = xyz(state.position);
        if (!finite(linearVelocity) || !finite(angularVelocityWorld) ||
            !finite(position)) {
            diagnostics.code = MR_STEP_NONFINITE_INPUT;
            return diagnostics;
        }

        const Mat3 rotation0 = rotationMatrix(orientation);
        if (motionType == MR_MOTION_KINEMATIC) {
            position += linearVelocity * config.timestep;
            const Vec3 angularVelocityBody =
                transpose(rotation0) * angularVelocityWorld;
            orientation = normalized(
                orientation *
                exponentialQuaternion(
                    angularVelocityBody * config.timestep
                )
            );
            if (!representableAsFloat(position) ||
                !representableAsFloat(orientation.x) ||
                !representableAsFloat(orientation.y) ||
                !representableAsFloat(orientation.z) ||
                !representableAsFloat(orientation.w)) {
                diagnostics.code = MR_STEP_NONFINITE_RESULT;
                return diagnostics;
            }
            state.position = f4(position, 1.0f);
            state.orientation = {
                static_cast<float>(orientation.x),
                static_cast<float>(orientation.y),
                static_cast<float>(orientation.z),
                static_cast<float>(orientation.w),
            };
            if (!writeInverseInertiaWorld(
                    state,
                    matrix(
                        body.inverseInertiaRow0,
                        body.inverseInertiaRow1,
                        body.inverseInertiaRow2
                    ),
                    orientation
                )) {
                diagnostics.code = MR_STEP_NONFINITE_RESULT;
                return diagnostics;
            }
            ++diagnostics.bodiesIntegrated;
            continue;
        }
        if (motionType != MR_MOTION_DYNAMIC ||
            !(state.linearVelocityAndInverseMass.w > 0.0f)) {
            diagnostics.code = MR_STEP_NONFINITE_INPUT;
            return diagnostics;
        }

        const BodyWrench wrench =
            wrenches.empty() ? BodyWrench{} : wrenches[index];
        const Vec3 force = xyz(wrench.force);
        const Vec3 torqueWorld = xyz(wrench.torque);
        if (!finite(force) || !finite(torqueWorld)) {
            diagnostics.code = MR_STEP_NONFINITE_INPUT;
            return diagnostics;
        }

        const double inverseMass = state.linearVelocityAndInverseMass.w;
        const Vec3 linearAcceleration = gravity + force * inverseMass;
        const Vec3 oldLinearVelocity = linearVelocity;
        linearVelocity += linearAcceleration * config.timestep;
        if (body.dampingAndSpeedLimits.x > 0.0f) {
            linearVelocity =
                linearVelocity /
                (1.0 + body.dampingAndSpeedLimits.x * config.timestep);
        }
        linearVelocity = clampMagnitude(
            linearVelocity,
            body.dampingAndSpeedLimits.z
        );

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
        const Vec3 omega0 = transpose(rotation0) * angularVelocityWorld;
        const Vec3 torqueBody = transpose(rotation0) * torqueWorld;

        Vec3 omega1{};
        Vec3 omegaMidpoint{};
        if (config.integrator == FreeBodyIntegrator::implicitMidpoint) {
            const MidpointResult solve = implicitMidpointAngularVelocity(
                omega0,
                torqueBody,
                inertia,
                inverseInertia,
                config
            );
            diagnostics.maximumIterations =
                std::max(diagnostics.maximumIterations, solve.iterations);
            diagnostics.maximumResidual =
                std::max(diagnostics.maximumResidual, solve.residual);
            if (!solve.converged) {
                diagnostics.code = MR_STEP_DID_NOT_CONVERGE;
                return diagnostics;
            }
            omega1 = solve.omega;
            omegaMidpoint = (omega0 + omega1) * 0.5;
            position +=
                (oldLinearVelocity + linearVelocity) *
                (0.5 * config.timestep);
        } else {
            omega1 =
                omega0 +
                angularDerivative(
                    omega0,
                    torqueBody,
                    inertia,
                    inverseInertia
                ) * config.timestep;
            omegaMidpoint = omega1;
            position += linearVelocity * config.timestep;
            diagnostics.maximumIterations =
                std::max(diagnostics.maximumIterations, 1u);
        }

        if (body.dampingAndSpeedLimits.y > 0.0f) {
            omega1 =
                omega1 /
                (1.0 + body.dampingAndSpeedLimits.y * config.timestep);
            omegaMidpoint = (omega0 + omega1) * 0.5;
        }
        omega1 = clampMagnitude(
            omega1,
            body.dampingAndSpeedLimits.w
        );
        omegaMidpoint =
            config.integrator == FreeBodyIntegrator::implicitMidpoint
            ? (omega0 + omega1) * 0.5
            : omega1;
        orientation = normalized(
            orientation *
            exponentialQuaternion(omegaMidpoint * config.timestep)
        );
        const Mat3 rotation1 = rotationMatrix(orientation);
        angularVelocityWorld = rotation1 * omega1;

        if (!finite(position) || !finite(linearVelocity) ||
            !finite(angularVelocityWorld) ||
            !std::isfinite(orientation.x) ||
            !std::isfinite(orientation.y) ||
            !std::isfinite(orientation.z) ||
            !std::isfinite(orientation.w) ||
            !representableAsFloat(position) ||
            !representableAsFloat(linearVelocity) ||
            !representableAsFloat(angularVelocityWorld) ||
            !representableAsFloat(orientation.x) ||
            !representableAsFloat(orientation.y) ||
            !representableAsFloat(orientation.z) ||
            !representableAsFloat(orientation.w)) {
            diagnostics.code = MR_STEP_NONFINITE_RESULT;
            return diagnostics;
        }

        state.position = f4(position, 1.0f);
        state.orientation = {
            static_cast<float>(orientation.x),
            static_cast<float>(orientation.y),
            static_cast<float>(orientation.z),
            static_cast<float>(orientation.w),
        };
        state.linearVelocityAndInverseMass =
            f4(linearVelocity, static_cast<float>(inverseMass));
        state.angularVelocity = f4(angularVelocityWorld);
        if (!writeInverseInertiaWorld(
                state,
                inverseInertia,
                orientation
            )) {
            diagnostics.code = MR_STEP_NONFINITE_RESULT;
            return diagnostics;
        }
        ++diagnostics.bodiesIntegrated;
    }
    std::copy(working.begin(), working.end(), states.begin());
    return diagnostics;
}

} // namespace metalrobo
