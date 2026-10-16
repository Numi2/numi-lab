#include <metal_stdlib>
#include "metalrobo/engine_types.h"

using namespace metal;

namespace {

constant uint kImplicitMidpoint = 1u;
constant float kQuaternionMinimum = 1.0e-12f;
constant float kPivotMinimum = 1.0e-12f;

struct Mat3 {
    float3 row[3];
};

struct MidpointResult {
    float3 omega;
    uint iterations;
    float residual;
    bool converged;
    bool factorizationFailed;
};

inline bool finite3(const float3 value) {
    return all(isfinite(value));
}

inline bool finite4(const float4 value) {
    return all(isfinite(value));
}

inline bool finiteMatrix(thread const Mat3& value) {
    return finite3(value.row[0]) &&
        finite3(value.row[1]) &&
        finite3(value.row[2]);
}

inline Mat3 makeMat3(
    const float4 row0,
    const float4 row1,
    const float4 row2
) {
    Mat3 result;
    result.row[0] = row0.xyz;
    result.row[1] = row1.xyz;
    result.row[2] = row2.xyz;
    return result;
}

inline Mat3 transpose(thread const Mat3& value) {
    Mat3 result;
    result.row[0] =
        float3(value.row[0].x, value.row[1].x, value.row[2].x);
    result.row[1] =
        float3(value.row[0].y, value.row[1].y, value.row[2].y);
    result.row[2] =
        float3(value.row[0].z, value.row[1].z, value.row[2].z);
    return result;
}

inline float3 multiply(thread const Mat3& value, const float3 vector) {
    return float3(
        dot(value.row[0], vector),
        dot(value.row[1], vector),
        dot(value.row[2], vector)
    );
}

inline Mat3 multiply(
    thread const Mat3& left,
    thread const Mat3& right
) {
    const Mat3 rightTranspose = transpose(right);
    Mat3 result;
    for (uint row = 0u; row < 3u; ++row) {
        result.row[row] = float3(
            dot(left.row[row], rightTranspose.row[0]),
            dot(left.row[row], rightTranspose.row[1]),
            dot(left.row[row], rightTranspose.row[2])
        );
    }
    return result;
}

inline bool normalizeQuaternion(
    const float4 input,
    thread float4& output
) {
    const float magnitudeSquared = dot(input, input);
    if (!(magnitudeSquared > kQuaternionMinimum) ||
        !isfinite(magnitudeSquared)) {
        output = float4(0.0f, 0.0f, 0.0f, 1.0f);
        return false;
    }
    output = input * rsqrt(magnitudeSquared);
    return finite4(output);
}

inline float4 quaternionMultiply(
    const float4 left,
    const float4 right
) {
    return float4(
        left.w * right.x + left.x * right.w +
            left.y * right.z - left.z * right.y,
        left.w * right.y - left.x * right.z +
            left.y * right.w + left.z * right.x,
        left.w * right.z + left.x * right.y -
            left.y * right.x + left.z * right.w,
        left.w * right.w - left.x * right.x -
            left.y * right.y - left.z * right.z
    );
}

inline bool exponentialQuaternion(
    const float3 rotationVector,
    thread float4& result
) {
    const float angleSquared = dot(rotationVector, rotationVector);
    if (!isfinite(angleSquared)) {
        return false;
    }
    if (angleSquared < 1.0e-20f) {
        const float vectorScale = 0.5f - angleSquared / 48.0f;
        return normalizeQuaternion(
            float4(
                rotationVector * vectorScale,
                1.0f - angleSquared / 8.0f
            ),
            result
        );
    }
    const float angle = sqrt(angleSquared);
    const float halfAngle = 0.5f * angle;
    result = float4(
        rotationVector * (sin(halfAngle) / angle),
        cos(halfAngle)
    );
    return finite4(result);
}

inline Mat3 rotationMatrix(const float4 q) {
    const float xx = q.x * q.x;
    const float yy = q.y * q.y;
    const float zz = q.z * q.z;
    const float xy = q.x * q.y;
    const float xz = q.x * q.z;
    const float yz = q.y * q.z;
    const float xw = q.x * q.w;
    const float yw = q.y * q.w;
    const float zw = q.z * q.w;
    Mat3 result;
    result.row[0] = float3(
        1.0f - 2.0f * (yy + zz),
        2.0f * (xy - zw),
        2.0f * (xz + yw)
    );
    result.row[1] = float3(
        2.0f * (xy + zw),
        1.0f - 2.0f * (xx + zz),
        2.0f * (yz - xw)
    );
    result.row[2] = float3(
        2.0f * (xz - yw),
        2.0f * (yz + xw),
        1.0f - 2.0f * (xx + yy)
    );
    return result;
}

inline bool solve3x3(
    thread const Mat3& inputMatrix,
    const float3 inputRight,
    thread float3& solution
) {
    float augmented[3][4] = {
        {
            inputMatrix.row[0].x,
            inputMatrix.row[0].y,
            inputMatrix.row[0].z,
            inputRight.x,
        },
        {
            inputMatrix.row[1].x,
            inputMatrix.row[1].y,
            inputMatrix.row[1].z,
            inputRight.y,
        },
        {
            inputMatrix.row[2].x,
            inputMatrix.row[2].y,
            inputMatrix.row[2].z,
            inputRight.z,
        },
    };

    for (uint column = 0u; column < 3u; ++column) {
        uint pivot = column;
        for (uint row = column + 1u; row < 3u; ++row) {
            if (abs(augmented[row][column]) >
                abs(augmented[pivot][column])) {
                pivot = row;
            }
        }
        if (!(abs(augmented[pivot][column]) > kPivotMinimum) ||
            !isfinite(augmented[pivot][column])) {
            return false;
        }
        if (pivot != column) {
            for (uint entry = column; entry < 4u; ++entry) {
                const float temporary = augmented[column][entry];
                augmented[column][entry] = augmented[pivot][entry];
                augmented[pivot][entry] = temporary;
            }
        }
        const float inversePivot = 1.0f / augmented[column][column];
        for (uint entry = column; entry < 4u; ++entry) {
            augmented[column][entry] *= inversePivot;
        }
        for (uint row = 0u; row < 3u; ++row) {
            if (row == column) {
                continue;
            }
            const float scale = augmented[row][column];
            for (uint entry = column; entry < 4u; ++entry) {
                augmented[row][entry] -=
                    scale * augmented[column][entry];
            }
        }
    }
    solution =
        float3(augmented[0][3], augmented[1][3], augmented[2][3]);
    return finite3(solution);
}

inline float3 angularDerivative(
    const float3 omega,
    const float3 torque,
    thread const Mat3& inertia,
    thread const Mat3& inverseInertia
) {
    return multiply(
        inverseInertia,
        torque - cross(omega, multiply(inertia, omega))
    );
}

inline MidpointResult implicitMidpointAngularVelocity(
    const float3 omega0,
    const float3 torqueBody,
    thread const Mat3& inertia,
    thread const Mat3& inverseInertia,
    const float timestep,
    const uint maximumIterations,
    const float tolerance
) {
    MidpointResult result;
    result.omega =
        omega0 +
        angularDerivative(
            omega0,
            torqueBody,
            inertia,
            inverseInertia
        ) * timestep;
    result.iterations = 0u;
    result.residual = INFINITY;
    result.converged = false;
    result.factorizationFailed = false;

    for (uint iteration = 0u;
         iteration < maximumIterations;
         ++iteration) {
        const float3 midpoint = 0.5f * (omega0 + result.omega);
        const float3 residualVector =
            result.omega - omega0 -
            angularDerivative(
                midpoint,
                torqueBody,
                inertia,
                inverseInertia
            ) * timestep;
        result.residual = length(residualVector);
        result.iterations = iteration + 1u;
        if (!isfinite(result.residual)) {
            return result;
        }
        if (result.residual <= tolerance) {
            result.converged = true;
            return result;
        }

        Mat3 jacobian;
        for (uint column = 0u; column < 3u; ++column) {
            float3 basis = float3(0.0f);
            basis[column] = 1.0f;
            const float3 derivative =
                multiply(
                    inverseInertia,
                    cross(basis, multiply(inertia, midpoint)) +
                    cross(midpoint, multiply(inertia, basis))
                ) * (-0.5f * timestep);
            for (uint row = 0u; row < 3u; ++row) {
                jacobian.row[row][column] =
                    (row == column ? 1.0f : 0.0f) -
                    derivative[row];
            }
        }
        float3 step;
        if (!finiteMatrix(jacobian) ||
            !solve3x3(jacobian, -residualVector, step)) {
            result.factorizationFailed = true;
            return result;
        }
        result.omega += step;
        if (!finite3(result.omega)) {
            return result;
        }
    }
    return result;
}

inline float3 clampMagnitude(
    const float3 value,
    const float maximum
) {
    const float magnitudeSquared = dot(value, value);
    if (!(maximum > 0.0f) ||
        magnitudeSquared <= maximum * maximum) {
        return value;
    }
    return value * (maximum * rsqrt(magnitudeSquared));
}

inline bool writeInverseInertiaWorld(
    thread MRBodyStateGPU& state,
    thread const Mat3& inverseInertiaBody,
    const float4 orientation
) {
    const Mat3 rotation = rotationMatrix(orientation);
    const Mat3 rotationTranspose = transpose(rotation);
    const Mat3 intermediate = multiply(rotation, inverseInertiaBody);
    const Mat3 world = multiply(intermediate, rotationTranspose);
    if (!finiteMatrix(world)) {
        return false;
    }
    state.inverseInertiaWorldRow0 = float4(world.row[0], 0.0f);
    state.inverseInertiaWorldRow1 = float4(world.row[1], 0.0f);
    state.inverseInertiaWorldRow2 = float4(world.row[2], 0.0f);
    return true;
}

inline bool validBatch(constant const MRFreeBodyBatchGPU& batch) {
    return
        batch.bodyCount > 0u &&
        batch.nonlinearIterations > 0u &&
        finite4(batch.gravityAndTimestep) &&
        batch.gravityAndTimestep.w > 0.0f &&
        finite4(batch.convergence) &&
        batch.convergence.x > 0.0f;
}

inline bool validProperties(
    device const MRBodyPropertiesGPU& body
) {
    return
        finite4(body.massAndInverseMass) &&
        finite4(body.centerOfMass) &&
        finite4(body.inertiaRow0) &&
        finite4(body.inertiaRow1) &&
        finite4(body.inertiaRow2) &&
        finite4(body.inverseInertiaRow0) &&
        finite4(body.inverseInertiaRow1) &&
        finite4(body.inverseInertiaRow2) &&
        finite4(body.dampingAndSpeedLimits) &&
        all(body.dampingAndSpeedLimits >= 0.0f);
}

inline bool validMovingState(device const MRBodyStateGPU& state) {
    return
        finite4(state.position) &&
        finite4(state.orientation) &&
        finite4(state.linearVelocityAndInverseMass) &&
        finite4(state.angularVelocity);
}

inline MRFreeBodyStatusGPU makeStatus(
    const uint bodyIndex,
    const uint code
) {
    MRFreeBodyStatusGPU status = {};
    status.code = code;
    status.bodyIndex = bodyIndex;
    return status;
}

} // namespace

kernel void mr_integrate_free_bodies(
    device const MRBodyPropertiesGPU* properties [[buffer(0)]],
    device MRBodyStateGPU* states [[buffer(1)]],
    device const MRBodyWrenchGPU* wrenches [[buffer(2)]],
    constant const MRFreeBodyBatchGPU& batch [[buffer(3)]],
    device MRFreeBodyStatusGPU* statuses [[buffer(4)]],
    const uint localBodyIndex [[thread_position_in_grid]]
) {
    if (localBodyIndex >= batch.bodyCount) {
        return;
    }
    if (batch.bodyCount > MR_INVALID_INDEX - batch.bodyOffset) {
        MRFreeBodyStatusGPU status =
            makeStatus(batch.bodyOffset, MR_STEP_NONFINITE_INPUT);
        statuses[localBodyIndex] = status;
        return;
    }
    const uint bodyIndex = batch.bodyOffset + localBodyIndex;
    MRFreeBodyStatusGPU status =
        makeStatus(bodyIndex, MR_STEP_SUCCESS);
    if (!validBatch(batch)) {
        status.code = MR_STEP_NONFINITE_INPUT;
        statuses[localBodyIndex] = status;
        return;
    }
    if (batch.integratorType > kImplicitMidpoint) {
        status.code = MR_STEP_UNSUPPORTED;
        statuses[localBodyIndex] = status;
        return;
    }

    device const MRBodyPropertiesGPU& body = properties[bodyIndex];
    device const MRBodyStateGPU& input = states[bodyIndex];
    const uint motionType = input.flagsAndIndices[0];
    if (!validProperties(body) ||
        motionType > MR_MOTION_DYNAMIC ||
        body.motionType != motionType) {
        status.code = MR_STEP_NONFINITE_INPUT;
        statuses[localBodyIndex] = status;
        return;
    }

    if (motionType == MR_MOTION_STATIC) {
        statuses[localBodyIndex] = status;
        return;
    }
    if (!validMovingState(input)) {
        status.code = MR_STEP_NONFINITE_INPUT;
        statuses[localBodyIndex] = status;
        return;
    }

    float4 orientation;
    if (!normalizeQuaternion(input.orientation, orientation)) {
        status.code = MR_STEP_NONFINITE_INPUT;
        statuses[localBodyIndex] = status;
        return;
    }
    const float timestep = batch.gravityAndTimestep.w;
    float3 position = input.position.xyz;
    float3 linearVelocity = input.linearVelocityAndInverseMass.xyz;
    float3 angularVelocityWorld = input.angularVelocity.xyz;
    const Mat3 rotation0 = rotationMatrix(orientation);
    const Mat3 rotation0Transpose = transpose(rotation0);
    const Mat3 inverseInertia = makeMat3(
        body.inverseInertiaRow0,
        body.inverseInertiaRow1,
        body.inverseInertiaRow2
    );

    MRBodyStateGPU output = input;
    if (motionType == MR_MOTION_KINEMATIC) {
        position += linearVelocity * timestep;
        const float3 angularVelocityBody =
            multiply(rotation0Transpose, angularVelocityWorld);
        float4 rotationIncrement;
        if (!exponentialQuaternion(
                angularVelocityBody * timestep,
                rotationIncrement
            ) ||
            !normalizeQuaternion(
                quaternionMultiply(orientation, rotationIncrement),
                orientation
            )) {
            status.code = MR_STEP_NONFINITE_RESULT;
            statuses[localBodyIndex] = status;
            return;
        }
        output.position = float4(position, 1.0f);
        output.orientation = orientation;
        if (!writeInverseInertiaWorld(
                output,
                inverseInertia,
                orientation
            )) {
            status.code = MR_STEP_NONFINITE_RESULT;
            statuses[localBodyIndex] = status;
            return;
        }
        status.diagnostics.y =
            abs(length(orientation) - 1.0f);
        status.diagnostics.z = length(angularVelocityWorld);
        states[bodyIndex] = output;
        statuses[localBodyIndex] = status;
        return;
    }

    if (!(input.linearVelocityAndInverseMass.w > 0.0f) ||
        !(body.massAndInverseMass.x > 0.0f) ||
        !(body.massAndInverseMass.y > 0.0f) ||
        !finite4(wrenches[bodyIndex].force) ||
        !finite4(wrenches[bodyIndex].torque)) {
        status.code = MR_STEP_NONFINITE_INPUT;
        statuses[localBodyIndex] = status;
        return;
    }
    const Mat3 inertia = makeMat3(
        body.inertiaRow0,
        body.inertiaRow1,
        body.inertiaRow2
    );
    if (!finiteMatrix(inertia) || !finiteMatrix(inverseInertia)) {
        status.code = MR_STEP_NONFINITE_INPUT;
        statuses[localBodyIndex] = status;
        return;
    }

    const float inverseMass =
        input.linearVelocityAndInverseMass.w;
    const float3 linearAcceleration =
        batch.gravityAndTimestep.xyz +
        wrenches[bodyIndex].force.xyz * inverseMass;
    const float3 oldLinearVelocity = linearVelocity;
    linearVelocity += linearAcceleration * timestep;
    if (body.dampingAndSpeedLimits.x > 0.0f) {
        linearVelocity /=
            1.0f + body.dampingAndSpeedLimits.x * timestep;
    }
    linearVelocity = clampMagnitude(
        linearVelocity,
        body.dampingAndSpeedLimits.z
    );

    const float3 omega0 =
        multiply(rotation0Transpose, angularVelocityWorld);
    const float3 torqueBody =
        multiply(
            rotation0Transpose,
            wrenches[bodyIndex].torque.xyz
        );
    float3 omega1;
    float3 omegaMidpoint;
    float nonlinearResidual = 0.0f;
    if (batch.integratorType == kImplicitMidpoint) {
        const MidpointResult solve =
            implicitMidpointAngularVelocity(
                omega0,
                torqueBody,
                inertia,
                inverseInertia,
                timestep,
                batch.nonlinearIterations,
                batch.convergence.x
            );
        status.iterations = solve.iterations;
        status.diagnostics.x = solve.residual;
        if (!solve.converged) {
            status.code =
                solve.factorizationFailed
                ? MR_STEP_FACTORIZATION_FAILED
                : MR_STEP_DID_NOT_CONVERGE;
            statuses[localBodyIndex] = status;
            return;
        }
        omega1 = solve.omega;
        omegaMidpoint = 0.5f * (omega0 + omega1);
        position +=
            0.5f * (oldLinearVelocity + linearVelocity) *
            timestep;
        nonlinearResidual = solve.residual;
    } else {
        omega1 =
            omega0 +
            angularDerivative(
                omega0,
                torqueBody,
                inertia,
                inverseInertia
            ) * timestep;
        omegaMidpoint = omega1;
        position += linearVelocity * timestep;
        status.iterations = 1u;
    }

    if (body.dampingAndSpeedLimits.y > 0.0f) {
        omega1 /=
            1.0f + body.dampingAndSpeedLimits.y * timestep;
        omegaMidpoint = 0.5f * (omega0 + omega1);
    }
    omega1 = clampMagnitude(
        omega1,
        body.dampingAndSpeedLimits.w
    );
    omegaMidpoint =
        batch.integratorType == kImplicitMidpoint
        ? 0.5f * (omega0 + omega1)
        : omega1;

    float4 rotationIncrement;
    if (!finite3(position) ||
        !finite3(linearVelocity) ||
        !finite3(omega1) ||
        !finite3(omegaMidpoint) ||
        !exponentialQuaternion(
            omegaMidpoint * timestep,
            rotationIncrement
        ) ||
        !normalizeQuaternion(
            quaternionMultiply(orientation, rotationIncrement),
            orientation
        )) {
        status.code = MR_STEP_NONFINITE_RESULT;
        statuses[localBodyIndex] = status;
        return;
    }

    const Mat3 rotation1 = rotationMatrix(orientation);
    angularVelocityWorld = multiply(rotation1, omega1);
    if (!finite3(angularVelocityWorld)) {
        status.code = MR_STEP_NONFINITE_RESULT;
        statuses[localBodyIndex] = status;
        return;
    }

    output.position = float4(position, 1.0f);
    output.orientation = orientation;
    output.linearVelocityAndInverseMass =
        float4(linearVelocity, inverseMass);
    output.angularVelocity = float4(angularVelocityWorld, 0.0f);
    if (!writeInverseInertiaWorld(
            output,
            inverseInertia,
            orientation
        )) {
        status.code = MR_STEP_NONFINITE_RESULT;
        statuses[localBodyIndex] = status;
        return;
    }

    status.diagnostics.x = nonlinearResidual;
    status.diagnostics.y =
        abs(length(orientation) - 1.0f);
    status.diagnostics.z = length(angularVelocityWorld);
    if (!finite4(status.diagnostics)) {
        status.code = MR_STEP_NONFINITE_RESULT;
        statuses[localBodyIndex] = status;
        return;
    }
    states[bodyIndex] = output;
    statuses[localBodyIndex] = status;
}
