#include <metal_stdlib>

#include "metalrobo/opensim_spatial_transform_gpu.h"

using namespace metal;

namespace {

inline float packedScalar(
    thread const mr_float4* blocks,
    const uint index
) {
    return blocks[index >> 2u][index & 3u];
}

inline bool finite3(const float3 value) {
    return all(isfinite(value));
}

inline float3x3 axisAngle(const float3 axis, const float angle) {
    const float cosine = cos(angle);
    const float sine = sin(angle);
    const float remainder = 1.0f - cosine;
    const float x = axis.x;
    const float y = axis.y;
    const float z = axis.z;
    // float3x3 is column major; these are the columns of Rodrigues' matrix.
    return float3x3(
        float3(
            cosine + x * x * remainder,
            y * x * remainder + z * sine,
            z * x * remainder - y * sine
        ),
        float3(
            x * y * remainder - z * sine,
            cosine + y * y * remainder,
            z * y * remainder + x * sine
        ),
        float3(
            x * z * remainder + y * sine,
            y * z * remainder - x * sine,
            cosine + z * z * remainder
        )
    );
}

inline bool evaluateFunction(
    thread const MROpenSimFunctionGPU& function,
    const float argument,
    thread float3& result
) {
    if (!isfinite(argument)) {
        return false;
    }
    const uint coefficientCount = function.coefficientCount;
    const uint knotCount = function.knotCount;
    if (function.kind == MR_OPENSIM_FUNCTION_CONSTANT) {
        if (coefficientCount != 1u || knotCount != 0u) {
            return false;
        }
        result = float3(packedScalar(function.coefficients, 0u), 0.0f, 0.0f);
        return isfinite(result.x);
    }
    if (function.kind == MR_OPENSIM_FUNCTION_LINEAR) {
        if (coefficientCount != 2u || knotCount != 0u) {
            return false;
        }
        const float slope = packedScalar(function.coefficients, 0u);
        result = float3(
            slope * argument + packedScalar(function.coefficients, 1u),
            slope,
            0.0f
        );
        return finite3(result);
    }
    if (function.kind == MR_OPENSIM_FUNCTION_POLYNOMIAL) {
        if (coefficientCount == 0u ||
            coefficientCount > MR_OPENSIM_SPATIAL_MAX_COEFFICIENTS ||
            knotCount != 0u) {
            return false;
        }
        float value = 0.0f;
        float derivative = 0.0f;
        float secondDerivative = 0.0f;
        for (uint index = 0u; index < coefficientCount; ++index) {
            secondDerivative = secondDerivative * argument + 2.0f * derivative;
            derivative = derivative * argument + value;
            value = value * argument + packedScalar(function.coefficients, index);
        }
        result = float3(value, derivative, secondDerivative);
        return finite3(result);
    }
    if (function.kind != MR_OPENSIM_FUNCTION_SIMM_SPLINE ||
        coefficientCount != 0u || knotCount < 2u ||
        knotCount > MR_OPENSIM_SPATIAL_MAX_KNOTS) {
        return false;
    }
    for (uint index = 0u; index < knotCount; ++index) {
        const float x = packedScalar(function.abscissae, index);
        const float y = packedScalar(function.ordinates, index);
        const float slope = packedScalar(function.splineSlope, index);
        const float quadratic = packedScalar(function.splineQuadratic, index);
        const float cubic = packedScalar(function.splineCubic, index);
        if (!isfinite(x) || !isfinite(y) || !isfinite(slope) ||
            !isfinite(quadratic) || !isfinite(cubic) ||
            (index > 0u && !(x > packedScalar(function.abscissae, index - 1u)))) {
            return false;
        }
    }
    const uint final = knotCount - 1u;
    const float firstX = packedScalar(function.abscissae, 0u);
    const float finalX = packedScalar(function.abscissae, final);
    if (argument < firstX) {
        const float slope = packedScalar(function.splineSlope, 0u);
        result = float3(
            packedScalar(function.ordinates, 0u) + (argument - firstX) * slope,
            slope,
            0.0f
        );
        return finite3(result);
    }
    if (argument > finalX) {
        const float slope = packedScalar(function.splineSlope, final);
        result = float3(
            packedScalar(function.ordinates, final) + (argument - finalX) * slope,
            slope,
            0.0f
        );
        return finite3(result);
    }
    uint low = 0u;
    uint high = final;
    uint interval = 0u;
    for (uint iteration = 0u; iteration < 5u; ++iteration) {
        interval = (low + high) >> 1u;
        if (argument < packedScalar(function.abscissae, interval)) {
            high = interval;
        } else if (argument > packedScalar(function.abscissae, interval + 1u)) {
            low = interval;
        } else {
            break;
        }
    }
    const float delta = argument - packedScalar(function.abscissae, interval);
    const float slope = packedScalar(function.splineSlope, interval);
    const float quadratic = packedScalar(function.splineQuadratic, interval);
    const float cubic = packedScalar(function.splineCubic, interval);
    result = float3(
        packedScalar(function.ordinates, interval) + delta *
            (slope + delta * (quadratic + delta * cubic)),
        slope + delta * (2.0f * quadratic + 3.0f * delta * cubic),
        2.0f * quadratic + 6.0f * delta * cubic
    );
    return finite3(result);
}

inline void writeFailure(
    thread MROpenSimSpatialTransformResultGPU& result,
    const uint status,
    const uint coordinateCount
) {
    result.status = status;
    result.coordinateCount = coordinateCount;
}

} // namespace

// One thread evaluates one immutable source transform. Coordinates and rates
// have a fixed six-float stride; entries beyond each program's coordinateCount
// are ignored and result entries are zero.
kernel void mr_opensim_spatial_transform_evaluate(
    device const MROpenSimSpatialTransformGPU* programs [[buffer(0)]],
    device const float* coordinates [[buffer(1)]],
    device const float* velocities [[buffer(2)]],
    device MROpenSimSpatialTransformResultGPU* results [[buffer(3)]],
    const uint programIndex [[thread_position_in_grid]]
) {
    const MROpenSimSpatialTransformGPU program = programs[programIndex];
    MROpenSimSpatialTransformResultGPU result{};
    if (program.abiVersion != MR_OPENSIM_SPATIAL_TRANSFORM_GPU_ABI_VERSION ||
        program.coordinateCount == 0u ||
        program.coordinateCount > MR_OPENSIM_SPATIAL_MAX_COORDINATES) {
        writeFailure(result, MR_OPENSIM_SPATIAL_INVALID_PROGRAM, program.coordinateCount);
        results[programIndex] = result;
        return;
    }
    device const float* q = coordinates +
        programIndex * MR_OPENSIM_SPATIAL_MAX_COORDINATES;
    device const float* qdot = velocities +
        programIndex * MR_OPENSIM_SPATIAL_MAX_COORDINATES;
    for (uint coordinate = 0u; coordinate < program.coordinateCount; ++coordinate) {
        if (!isfinite(q[coordinate]) || !isfinite(qdot[coordinate])) {
            writeFailure(result, MR_OPENSIM_SPATIAL_NONFINITE_INPUT, program.coordinateCount);
            results[programIndex] = result;
            return;
        }
    }

    float3 values[6];
    float3 axes[6];
    uint coordinateIndex[6];
    for (uint index = 0u; index < 6u; ++index) {
        const MROpenSimFunctionGPU function = program.axes[index];
        const bool isConstant = function.kind == MR_OPENSIM_FUNCTION_CONSTANT;
        if ((isConstant && function.coordinateIndex != MR_OPENSIM_SPATIAL_NO_COORDINATE) ||
            (!isConstant && (function.coordinateIndex == MR_OPENSIM_SPATIAL_NO_COORDINATE ||
                           function.coordinateIndex >= program.coordinateCount)) ||
            !finite3(function.axis.xyz) ||
            !(dot(function.axis.xyz, function.axis.xyz) > 1.0e-10f)) {
            writeFailure(result, MR_OPENSIM_SPATIAL_INVALID_PROGRAM, program.coordinateCount);
            results[programIndex] = result;
            return;
        }
        coordinateIndex[index] = function.coordinateIndex;
        axes[index] = normalize(function.axis.xyz);
        const float argument = isConstant ? 0.0f : q[function.coordinateIndex];
        if (!evaluateFunction(function, argument, values[index])) {
            writeFailure(result, MR_OPENSIM_SPATIAL_INVALID_PROGRAM, program.coordinateCount);
            results[programIndex] = result;
            return;
        }
    }

    const float3x3 rotation0 = axisAngle(axes[0], values[0].x);
    const float3x3 rotation01 = rotation0 * axisAngle(axes[1], values[1].x);
    const float3x3 rotation = rotation01 * axisAngle(axes[2], values[2].x);
    result.rotationRow0 = float4(rotation[0][0], rotation[1][0], rotation[2][0], 0.0f);
    result.rotationRow1 = float4(rotation[0][1], rotation[1][1], rotation[2][1], 0.0f);
    result.rotationRow2 = float4(rotation[0][2], rotation[1][2], rotation[2][2], 0.0f);
    result.translation = float4(
        axes[3] * values[3].x + axes[4] * values[4].x + axes[5] * values[5].x,
        0.0f
    );

    const float3 angularAxes[3] = {
        axes[0],
        rotation0 * axes[1],
        rotation01 * axes[2],
    };
    const float thetaDot0 = coordinateIndex[0] == MR_OPENSIM_SPATIAL_NO_COORDINATE
        ? 0.0f
        : values[0].y * qdot[coordinateIndex[0]];
    const float thetaDot1 = coordinateIndex[1] == MR_OPENSIM_SPATIAL_NO_COORDINATE
        ? 0.0f
        : values[1].y * qdot[coordinateIndex[1]];
    const float3 angularAxisDots[3] = {
        float3(0.0f),
        cross(angularAxes[0] * thetaDot0, angularAxes[1]),
        cross(
            angularAxes[0] * thetaDot0 + angularAxes[1] * thetaDot1,
            angularAxes[2]
        ),
    };
    for (uint index = 0u; index < 6u; ++index) {
        const uint coordinate = coordinateIndex[index];
        if (coordinate == MR_OPENSIM_SPATIAL_NO_COORDINATE) {
            continue;
        }
        const float derivative = values[index].y;
        const float derivativeDot = values[index].z * qdot[coordinate];
        if (index < 3u) {
            result.motionAngular[coordinate].xyz += angularAxes[index] * derivative;
            result.motionAngularDot[coordinate].xyz +=
                angularAxisDots[index] * derivative +
                angularAxes[index] * derivativeDot;
        } else {
            result.motionLinear[coordinate].xyz += axes[index] * derivative;
            result.motionLinearDot[coordinate].xyz += axes[index] * derivativeDot;
        }
    }
    if (!finite3(result.rotationRow0.xyz) || !finite3(result.rotationRow1.xyz) ||
        !finite3(result.rotationRow2.xyz) || !finite3(result.translation.xyz)) {
        writeFailure(result, MR_OPENSIM_SPATIAL_NONFINITE_RESULT, program.coordinateCount);
        results[programIndex] = result;
        return;
    }
    result.status = MR_OPENSIM_SPATIAL_SUCCESS;
    result.coordinateCount = program.coordinateCount;
    results[programIndex] = result;
}
