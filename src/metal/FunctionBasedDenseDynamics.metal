#include <metal_stdlib>

#include "metalrobo/engine_types.h"
#include "metalrobo/opensim_spatial_transform_gpu.h"

using namespace metal;

// Bounded dense source dynamics for immutable OpenSim FunctionBased trees.
//
// MetalWorld normally advances scalar trees with its O(n) ABA kernels.  An
// OpenSim CustomJoint has up to six coupled coordinates, so forcing it into
// that scalar representation would silently discard source kinematics.  This
// kernel instead follows the FP64 reference formulation in
// ArticulatedDynamics.cpp: exact source SpatialTransform kinematics and
// Sdot*qdot bias, dense M(q), recursive Newton-Euler bias, then a Cholesky
// solve.  It is deliberately bounded to the existing full ABA bucket and is
// selected only for fixed-root FunctionBased articulations.

namespace {

constant uint kMaxBodies = MR_ARTICULATED_ABA_MAX_BODIES;
constant uint kMaxDofs = MR_ARTICULATED_ABA_MAX_DOFS;
constant uint kMaxQ = MR_ARTICULATED_ABA_MAX_Q;
constant float kEpsilon = 1.1920928955078125e-7f;
constant float kQuaternionMinimum = 1.0e-12f;
constant float kPivotFloor = 1.0e-12f;

struct Motion {
    float3 angular;
    float3 linear;
};

struct FunctionKinematics {
    float4 rotation;
    float3 translation;
    float3 angular[MR_OPENSIM_SPATIAL_MAX_COORDINATES];
    float3 linear[MR_OPENSIM_SPATIAL_MAX_COORDINATES];
    float3 angularDot[MR_OPENSIM_SPATIAL_MAX_COORDINATES];
    float3 linearDot[MR_OPENSIM_SPATIAL_MAX_COORDINATES];
    uint coordinateCount;
};

struct DenseScratch {
    float3 bodyPosition[MR_ARTICULATED_ABA_MAX_BODIES];
    float4 bodyRotation[MR_ARTICULATED_ABA_MAX_BODIES];
    float3 angularVelocity[MR_ARTICULATED_ABA_MAX_BODIES];
    float3 linearVelocity[MR_ARTICULATED_ABA_MAX_BODIES];
    float3 angularAcceleration[MR_ARTICULATED_ABA_MAX_BODIES];
    float3 linearAcceleration[MR_ARTICULATED_ABA_MAX_BODIES];
    float3 jointPosition[MR_ARTICULATED_ABA_MAX_BODIES];
    float3 jointAxis[MR_ARTICULATED_ABA_MAX_BODIES];
    uint inboundJoint[MR_ARTICULATED_ABA_MAX_BODIES];
    uint parentLocal[MR_ARTICULATED_ABA_MAX_BODIES];
    uint traversal[MR_ARTICULATED_ABA_MAX_BODIES];
    uchar known[MR_ARTICULATED_ABA_MAX_BODIES];
    float mass[MR_ARTICULATED_ABA_MAX_DOFS * MR_ARTICULATED_ABA_MAX_DOFS];
    float factor[MR_ARTICULATED_ABA_MAX_DOFS * MR_ARTICULATED_ABA_MAX_DOFS];
    float bias[MR_ARTICULATED_ABA_MAX_DOFS];
    float acceleration[MR_ARTICULATED_ABA_MAX_DOFS];
    float nextV[MR_ARTICULATED_ABA_MAX_DOFS];
    float nextQ[MR_ARTICULATED_ABA_MAX_Q];
};

inline bool finite3(const float3 value) { return all(isfinite(value)); }
inline bool finite4(const float4 value) { return all(isfinite(value)); }

inline float4 quaternionConjugate(const float4 value) {
    return float4(-value.xyz, value.w);
}

inline float4 quaternionMultiply(const float4 left, const float4 right) {
    return float4(
        left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
        left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
        left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
        left.w * right.w - dot(left.xyz, right.xyz)
    );
}

inline float3 quaternionRotate(const float4 quaternion, const float3 value) {
    const float3 doubledCross = 2.0f * cross(quaternion.xyz, value);
    return value + quaternion.w * doubledCross + cross(quaternion.xyz, doubledCross);
}

inline bool normalizedQuaternion(const float4 input, thread float4& output) {
    if (!finite4(input)) return false;
    const float normSquared = dot(input, input);
    if (!(normSquared > kQuaternionMinimum) || !isfinite(normSquared)) return false;
    output = input * rsqrt(normSquared);
    return finite4(output);
}

inline float4 axisAngleQuaternion(const float3 axis, const float angle) {
    const float halfAngle = 0.5f * angle;
    return float4(axis * sin(halfAngle), cos(halfAngle));
}

inline float packedScalar(thread const mr_float4* blocks, const uint index) {
    return blocks[index >> 2u][index & 3u];
}

// Returns value, first derivative, second derivative.  This is kept bitwise
// structurally aligned with OpenSimSpatialTransform.metal and the CPU source
// transform evaluator; it is not a fitted joint approximation.
inline bool evaluateFunction(
    thread const MROpenSimFunctionGPU& function,
    const float argument,
    thread float3& result
) {
    if (!isfinite(argument)) return false;
    if (function.kind == MR_OPENSIM_FUNCTION_CONSTANT) {
        if (function.coefficientCount != 1u || function.knotCount != 0u) return false;
        result = float3(packedScalar(function.coefficients, 0u), 0.0f, 0.0f);
        return finite3(result);
    }
    if (function.kind == MR_OPENSIM_FUNCTION_LINEAR) {
        if (function.coefficientCount != 2u || function.knotCount != 0u) return false;
        const float slope = packedScalar(function.coefficients, 0u);
        result = float3(slope * argument + packedScalar(function.coefficients, 1u), slope, 0.0f);
        return finite3(result);
    }
    if (function.kind == MR_OPENSIM_FUNCTION_POLYNOMIAL) {
        if (function.coefficientCount == 0u ||
            function.coefficientCount > MR_OPENSIM_SPATIAL_MAX_COEFFICIENTS ||
            function.knotCount != 0u) return false;
        float value = 0.0f;
        float derivative = 0.0f;
        float secondDerivative = 0.0f;
        for (uint index = 0u; index < function.coefficientCount; ++index) {
            secondDerivative = secondDerivative * argument + 2.0f * derivative;
            derivative = derivative * argument + value;
            value = value * argument + packedScalar(function.coefficients, index);
        }
        result = float3(value, derivative, secondDerivative);
        return finite3(result);
    }
    if (function.kind != MR_OPENSIM_FUNCTION_SIMM_SPLINE ||
        function.coefficientCount != 0u || function.knotCount < 2u ||
        function.knotCount > MR_OPENSIM_SPATIAL_MAX_KNOTS) return false;
    for (uint index = 0u; index < function.knotCount; ++index) {
        const float x = packedScalar(function.abscissae, index);
        const float y = packedScalar(function.ordinates, index);
        const float slope = packedScalar(function.splineSlope, index);
        const float quadratic = packedScalar(function.splineQuadratic, index);
        const float cubic = packedScalar(function.splineCubic, index);
        if (!isfinite(x) || !isfinite(y) || !isfinite(slope) ||
            !isfinite(quadratic) || !isfinite(cubic) ||
            (index > 0u && !(x > packedScalar(function.abscissae, index - 1u)))) return false;
    }
    const uint final = function.knotCount - 1u;
    const float firstX = packedScalar(function.abscissae, 0u);
    const float finalX = packedScalar(function.abscissae, final);
    if (argument < firstX || argument > finalX) {
        const uint endpoint = argument < firstX ? 0u : final;
        const float x = packedScalar(function.abscissae, endpoint);
        const float y = packedScalar(function.ordinates, endpoint);
        const float slope = packedScalar(function.splineSlope, endpoint);
        result = float3(y + (argument - x) * slope, slope, 0.0f);
        return finite3(result);
    }
    uint low = 0u;
    uint high = final;
    uint interval = 0u;
    for (uint iteration = 0u; iteration < 5u; ++iteration) {
        interval = (low + high) >> 1u;
        if (argument < packedScalar(function.abscissae, interval)) high = interval;
        else if (argument > packedScalar(function.abscissae, interval + 1u)) low = interval;
        else break;
    }
    const float delta = argument - packedScalar(function.abscissae, interval);
    const float slope = packedScalar(function.splineSlope, interval);
    const float quadratic = packedScalar(function.splineQuadratic, interval);
    const float cubic = packedScalar(function.splineCubic, interval);
    result = float3(
        packedScalar(function.ordinates, interval) + delta * (slope + delta * (quadratic + delta * cubic)),
        slope + delta * (2.0f * quadratic + 3.0f * delta * cubic),
        2.0f * quadratic + 6.0f * delta * cubic
    );
    return finite3(result);
}

inline bool evaluateFunctionBasedJoint(
    device const MROpenSimSpatialTransformGPU& program,
    device const float* coordinates,
    device const float* velocities,
    thread FunctionKinematics& result
) {
    result = {};
    if (program.abiVersion != MR_OPENSIM_SPATIAL_TRANSFORM_GPU_ABI_VERSION ||
        program.coordinateCount == 0u ||
        program.coordinateCount > MR_OPENSIM_SPATIAL_MAX_COORDINATES ||
        program.reserved0 != 0u || program.reserved1 != 0u) return false;
    float3 values[6];
    float3 axes[6];
    uint coordinateIndex[6];
    for (uint index = 0u; index < 6u; ++index) {
        const MROpenSimFunctionGPU function = program.axes[index];
        const bool functionIsConstant =
            function.kind == MR_OPENSIM_FUNCTION_CONSTANT;
        if ((functionIsConstant &&
             function.coordinateIndex != MR_OPENSIM_SPATIAL_NO_COORDINATE) ||
            (!functionIsConstant &&
             (function.coordinateIndex == MR_OPENSIM_SPATIAL_NO_COORDINATE ||
                           function.coordinateIndex >= program.coordinateCount)) ||
            !finite4(function.axis) || function.axis.w != 0.0f ||
            !(dot(function.axis.xyz, function.axis.xyz) > 1.0e-10f)) return false;
        axes[index] = normalize(function.axis.xyz);
        coordinateIndex[index] = function.coordinateIndex;
        const float argument = functionIsConstant
            ? 0.0f
            : coordinates[function.coordinateIndex];
        if (!evaluateFunction(function, argument, values[index])) return false;
    }
    const float4 rotation0 = axisAngleQuaternion(axes[0], values[0].x);
    const float4 rotation01 = quaternionMultiply(rotation0, axisAngleQuaternion(axes[1], values[1].x));
    result.rotation = quaternionMultiply(rotation01, axisAngleQuaternion(axes[2], values[2].x));
    result.translation = axes[3] * values[3].x + axes[4] * values[4].x + axes[5] * values[5].x;
    const float3 angularAxes[3] = { axes[0], quaternionRotate(rotation0, axes[1]), quaternionRotate(rotation01, axes[2]) };
    const float thetaDot0 = coordinateIndex[0] == MR_OPENSIM_SPATIAL_NO_COORDINATE ? 0.0f : values[0].y * velocities[coordinateIndex[0]];
    const float thetaDot1 = coordinateIndex[1] == MR_OPENSIM_SPATIAL_NO_COORDINATE ? 0.0f : values[1].y * velocities[coordinateIndex[1]];
    const float3 angularAxisDots[3] = {
        float3(0.0f),
        cross(angularAxes[0] * thetaDot0, angularAxes[1]),
        cross(angularAxes[0] * thetaDot0 + angularAxes[1] * thetaDot1, angularAxes[2]),
    };
    for (uint index = 0u; index < 6u; ++index) {
        const uint coordinate = coordinateIndex[index];
        if (coordinate == MR_OPENSIM_SPATIAL_NO_COORDINATE) continue;
        const float derivativeDot = values[index].z * velocities[coordinate];
        if (index < 3u) {
            result.angular[coordinate] += angularAxes[index] * values[index].y;
            result.angularDot[coordinate] += angularAxisDots[index] * values[index].y + angularAxes[index] * derivativeDot;
        } else {
            result.linear[coordinate] += axes[index] * values[index].y;
            result.linearDot[coordinate] += axes[index] * derivativeDot;
        }
    }
    result.coordinateCount = program.coordinateCount;
    return finite4(result.rotation) && finite3(result.translation);
}

inline float3 inertiaMultiply(device const MRBodyPropertiesGPU& body, const float3 value) {
    return float3(dot(body.inertiaRow0.xyz, value), dot(body.inertiaRow1.xyz, value), dot(body.inertiaRow2.xyz, value));
}

inline float3 worldInertiaMultiply(
    device const MRBodyPropertiesGPU& body,
    const float4 rotation,
    const float3 value
) {
    const float4 inverse = quaternionConjugate(rotation);
    return quaternionRotate(rotation, inertiaMultiply(body, quaternionRotate(inverse, value)));
}

inline void fail(thread MRABAStatusGPU& status, const uint code, const uint index) {
    status.code = code;
    status.failingIndex = index;
}

inline Motion bodyMotionForDof(
    const uint localBody,
    const uint dof,
    device const MRArticulationGPU& articulation,
    device const MRJointDescriptorGPU* joints,
    device const MROpenSimSpatialTransformGPU* programs,
    device const float* q,
    device const float* v,
    threadgroup const float3* bodyPosition,
    threadgroup const float4* bodyRotation,
    threadgroup const float3* jointPosition,
    threadgroup const float3* jointAxis,
    threadgroup const uint* inboundJoint,
    threadgroup const uint* parentLocal
) {
    Motion result{};
    const uint rootLocal = articulation.rootBody - articulation.firstBody;
    uint cursor = localBody;
    for (uint depth = 0u; depth < articulation.bodyCount && cursor != rootLocal; ++depth) {
        const uint globalJoint = inboundJoint[cursor];
        if (globalJoint == MR_INVALID_INDEX) return result;
        device const MRJointDescriptorGPU& joint = joints[globalJoint];
        const uint localV = joint.vOffset - articulation.vOffset;
        if (joint.jointType == MR_JOINT_FUNCTION_BASED && dof >= localV && dof - localV < joint.nv) {
            const uint localQ = joint.qOffset - articulation.qOffset;
            FunctionKinematics state;
            if (!evaluateFunctionBasedJoint(programs[globalJoint], q + localQ, v + localV, state) || state.coordinateCount != joint.nv) return Motion{};
            const float4 parentToJoint = quaternionMultiply(bodyRotation[parentLocal[cursor]], joint.parentRotation);
            const uint localDof = dof - localV;
            result.angular = quaternionRotate(parentToJoint, state.angular[localDof]);
            result.linear = quaternionRotate(parentToJoint, state.linear[localDof]) + cross(result.angular, bodyPosition[localBody] - jointPosition[cursor]);
            return result;
        }
        if (joint.nv == 1u && localV == dof) {
            if (joint.jointType == MR_JOINT_PRISMATIC) result.linear = jointAxis[cursor];
            else {
                result.angular = jointAxis[cursor];
                result.linear = cross(result.angular, bodyPosition[localBody] - jointPosition[cursor]);
            }
            return result;
        }
        cursor = parentLocal[cursor];
    }
    return result;
}

} // namespace

kernel void mr_function_based_dense_dynamics_step(
    device const MRWorldGPU* worlds [[buffer(0)]],
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const MRJointDescriptorGPU* joints [[buffer(2)]],
    device const MRDofPropertiesGPU* dofs [[buffer(3)]],
    device const MRBodyPropertiesGPU* bodies [[buffer(4)]],
    device const MRABADispatchGPU& dispatch [[buffer(5)]],
    device const float* qInput [[buffer(6)]],
    device const float* vInput [[buffer(7)]],
    device const float* effortInput [[buffer(8)]],
    device const MRABABodyWrenchGPU* bodyWrenches [[buffer(9)]],
    device float* accelerationOutput [[buffer(10)]],
    device float* nextVOutput [[buffer(11)]],
    device float* nextQOutput [[buffer(12)]],
    device MRABAStatusGPU* statuses [[buffer(13)]],
    device const MROpenSimSpatialTransformGPU* programs [[buffer(14)]],
    uint environment [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (lane != 0u || environment >= dispatch.environmentCount) return;
    threadgroup DenseScratch storage;
    threadgroup DenseScratch& scratch = storage;
    MRABAStatusGPU status{};
    status.code = MR_ABA_SUCCESS;
    status.environment = environment;
    status.articulationIndex = dispatch.articulationIndex;
    status.failingIndex = MR_INVALID_INDEX;
    status.flags = dispatch.flags;
    device const MRWorldGPU& world = worlds[0];
    if (dispatch.articulationIndex >= world.articulationCount ||
        dispatch.qStride < world.nq || dispatch.vStride < world.nv ||
        dispatch.effortStride < world.nv || dispatch.accelerationStride < world.nv ||
        dispatch.nextVStride < world.nv || dispatch.nextQStride < world.nq) {
        fail(status, MR_ABA_INVALID_DISPATCH, MR_INVALID_INDEX);
        statuses[environment] = status;
        return;
    }
    device const MRArticulationGPU& articulation = articulations[dispatch.articulationIndex];
    status.bodyCount = articulation.bodyCount;
    status.nq = articulation.nq;
    status.nv = articulation.nv;
    if (articulation.rootType != MR_ROOT_FIXED || articulation.bodyCount == 0u ||
        articulation.bodyCount > kMaxBodies || articulation.nv == 0u ||
        articulation.nv > kMaxDofs || articulation.nq > kMaxQ ||
        articulation.jointCount + 1u != articulation.bodyCount ||
        articulation.firstBody + articulation.bodyCount > world.bodyCount ||
        articulation.firstJoint + articulation.jointCount > world.jointCount ||
        articulation.qOffset + articulation.nq > world.nq ||
        articulation.vOffset + articulation.nv > world.nv) {
        fail(status, MR_ABA_UNSUPPORTED_TOPOLOGY, MR_INVALID_INDEX);
        statuses[environment] = status;
        return;
    }
    const uint rootLocal = articulation.rootBody - articulation.firstBody;
    if (rootLocal >= articulation.bodyCount) {
        fail(status, MR_ABA_INVALID_MODEL, articulation.rootBody);
        statuses[environment] = status;
        return;
    }
    for (uint body = 0u; body < articulation.bodyCount; ++body) {
        scratch.inboundJoint[body] = MR_INVALID_INDEX;
        scratch.parentLocal[body] = MR_INVALID_INDEX;
        scratch.known[body] = 0u;
    }
    uint expectedQ = 0u;
    uint expectedV = 0u;
    for (uint localJoint = 0u; localJoint < articulation.jointCount; ++localJoint) {
        const uint globalJoint = articulation.firstJoint + localJoint;
        device const MRJointDescriptorGPU& joint = joints[globalJoint];
        const bool scalar = joint.jointType == MR_JOINT_REVOLUTE || joint.jointType == MR_JOINT_CONTINUOUS || joint.jointType == MR_JOINT_PRISMATIC;
        const bool fixed = joint.jointType == MR_JOINT_FIXED;
        const bool functionBased = joint.jointType == MR_JOINT_FUNCTION_BASED;
        if (joint.flags != 0u || joint.parentBody < articulation.firstBody ||
            joint.parentBody >= articulation.firstBody + articulation.bodyCount ||
            joint.childBody < articulation.firstBody ||
            joint.childBody >= articulation.firstBody + articulation.bodyCount ||
            joint.childBody == articulation.rootBody || joint.parentBody == joint.childBody ||
            (!scalar && !fixed && !functionBased) ||
            (scalar && (joint.nq != 1u || joint.nv != 1u)) ||
            (fixed && (joint.nq != 0u || joint.nv != 0u)) ||
            (functionBased && (joint.nq == 0u || joint.nq > MR_OPENSIM_SPATIAL_MAX_COORDINATES || joint.nq != joint.nv)) ||
            joint.qOffset != articulation.qOffset + expectedQ ||
            joint.vOffset != articulation.vOffset + expectedV ||
            !finite4(joint.parentAnchor) || !finite4(joint.childAnchor) ||
            !finite4(joint.parentRotation) || !finite4(joint.childRotation)) {
            fail(status, MR_ABA_INVALID_MODEL, globalJoint);
            statuses[environment] = status;
            return;
        }
        if (scalar && (!finite4(joint.axis0) || !(dot(joint.axis0.xyz, joint.axis0.xyz) > 1.0e-12f))) {
            fail(status, MR_ABA_INVALID_MODEL, globalJoint);
            statuses[environment] = status;
            return;
        }
        const uint child = joint.childBody - articulation.firstBody;
        if (scratch.inboundJoint[child] != MR_INVALID_INDEX) {
            fail(status, MR_ABA_UNSUPPORTED_TOPOLOGY, joint.childBody);
            statuses[environment] = status;
            return;
        }
        scratch.inboundJoint[child] = globalJoint;
        scratch.parentLocal[child] = joint.parentBody - articulation.firstBody;
        expectedQ += joint.nq;
        expectedV += joint.nv;
    }
    if (expectedQ != articulation.nq || expectedV != articulation.nv || scratch.inboundJoint[rootLocal] != MR_INVALID_INDEX) {
        fail(status, MR_ABA_INVALID_MODEL, MR_INVALID_INDEX);
        statuses[environment] = status;
        return;
    }
    scratch.known[rootLocal] = 1u;
    scratch.traversal[0] = rootLocal;
    uint discovered = 1u;
    for (uint pass = 0u; pass < articulation.bodyCount && discovered < articulation.bodyCount; ++pass) {
        bool progressed = false;
        for (uint localJoint = 0u; localJoint < articulation.jointCount; ++localJoint) {
            const uint globalJoint = articulation.firstJoint + localJoint;
            device const MRJointDescriptorGPU& joint = joints[globalJoint];
            const uint parent = joint.parentBody - articulation.firstBody;
            const uint child = joint.childBody - articulation.firstBody;
            if (scratch.known[parent] != 0u && scratch.known[child] == 0u) {
                scratch.known[child] = 1u;
                scratch.traversal[discovered++] = child;
                progressed = true;
            }
        }
        if (!progressed) break;
    }
    if (discovered != articulation.bodyCount) {
        fail(status, MR_ABA_UNSUPPORTED_TOPOLOGY, MR_INVALID_INDEX);
        statuses[environment] = status;
        return;
    }
    const uint qBase = environment * dispatch.qStride;
    const uint vBase = environment * dispatch.vStride;
    const uint effortBase = environment * dispatch.effortStride;
    device const float* q = qInput + qBase + articulation.qOffset;
    device const float* v = vInput + vBase + articulation.vOffset;
    device const float* effort = effortInput + effortBase + articulation.vOffset;
    for (uint index = 0u; index < articulation.nq; ++index) {
        if (!isfinite(q[index])) {
            fail(status, MR_ABA_NONFINITE_INPUT, index);
            statuses[environment] = status;
            return;
        }
        scratch.nextQ[index] = q[index];
    }
    for (uint index = 0u; index < articulation.nv; ++index) {
        if (!isfinite(v[index]) || !isfinite(effort[index]) || !finite4(dofs[articulation.vOffset + index].drive)) {
            fail(status, MR_ABA_NONFINITE_INPUT, index);
            statuses[environment] = status;
            return;
        }
    }
    scratch.bodyPosition[rootLocal] = float3(0.0f);
    scratch.bodyRotation[rootLocal] = float4(0.0f, 0.0f, 0.0f, 1.0f);
    scratch.angularVelocity[rootLocal] = float3(0.0f);
    scratch.linearVelocity[rootLocal] = float3(0.0f);
    scratch.angularAcceleration[rootLocal] = float3(0.0f);
    scratch.linearAcceleration[rootLocal] = float3(0.0f);
    scratch.jointPosition[rootLocal] = float3(0.0f);
    scratch.jointAxis[rootLocal] = float3(0.0f);
    for (uint order = 1u; order < articulation.bodyCount; ++order) {
        const uint child = scratch.traversal[order];
        const uint globalJoint = scratch.inboundJoint[child];
        device const MRJointDescriptorGPU& joint = joints[globalJoint];
        const uint parent = scratch.parentLocal[child];
        float4 parentRotation;
        float4 childRotation;
        if (!normalizedQuaternion(joint.parentRotation, parentRotation) || !normalizedQuaternion(joint.childRotation, childRotation)) {
            fail(status, MR_ABA_INVALID_MODEL, globalJoint);
            statuses[environment] = status;
            return;
        }
        const float4 parentToJoint = quaternionMultiply(scratch.bodyRotation[parent], parentRotation);
        float4 motionRotation = float4(0.0f, 0.0f, 0.0f, 1.0f);
        float3 translation = float3(0.0f);
        float3 relativeAngular = float3(0.0f);
        float3 relativeLinear = float3(0.0f);
        float3 relativeAngularDot = float3(0.0f);
        float3 relativeLinearDot = float3(0.0f);
        float3 axis = float3(1.0f, 0.0f, 0.0f);
        if (joint.jointType == MR_JOINT_FUNCTION_BASED) {
            const uint localQ = joint.qOffset - articulation.qOffset;
            const uint localV = joint.vOffset - articulation.vOffset;
            FunctionKinematics state;
            if (!evaluateFunctionBasedJoint(programs[globalJoint], q + localQ, v + localV, state) || state.coordinateCount != joint.nv) {
                fail(status, MR_ABA_INVALID_MODEL, globalJoint);
                statuses[environment] = status;
                return;
            }
            motionRotation = state.rotation;
            translation = state.translation;
            for (uint local = 0u; local < joint.nv; ++local) {
                relativeAngular += state.angular[local] * v[localV + local];
                relativeLinear += state.linear[local] * v[localV + local];
                relativeAngularDot += state.angularDot[local] * v[localV + local];
                relativeLinearDot += state.linearDot[local] * v[localV + local];
            }
            relativeAngular = quaternionRotate(parentToJoint, relativeAngular);
            relativeLinear = quaternionRotate(parentToJoint, relativeLinear);
            relativeAngularDot = quaternionRotate(parentToJoint, relativeAngularDot);
            relativeLinearDot = quaternionRotate(parentToJoint, relativeLinearDot);
        } else if (joint.nv == 1u) {
            axis = normalize(joint.axis0.xyz);
            const uint localQ = joint.qOffset - articulation.qOffset;
            const uint localV = joint.vOffset - articulation.vOffset;
            if (joint.jointType == MR_JOINT_REVOLUTE || joint.jointType == MR_JOINT_CONTINUOUS) {
                motionRotation = axisAngleQuaternion(axis, q[localQ]);
                relativeAngular = quaternionRotate(parentToJoint, axis) * v[localV];
            } else if (joint.jointType == MR_JOINT_PRISMATIC) {
                translation = axis * q[localQ];
                relativeLinear = quaternionRotate(parentToJoint, axis) * v[localV];
            }
        }
        float4 candidateRotation;
        if (!normalizedQuaternion(quaternionMultiply(quaternionMultiply(parentToJoint, motionRotation), quaternionConjugate(childRotation)), candidateRotation)) {
            fail(status, MR_ABA_NONFINITE_RESULT, joint.childBody);
            statuses[environment] = status;
            return;
        }
        scratch.bodyRotation[child] = candidateRotation;
        scratch.jointAxis[child] = quaternionRotate(parentToJoint, axis);
        scratch.jointPosition[child] = scratch.bodyPosition[parent] +
            quaternionRotate(scratch.bodyRotation[parent], joint.parentAnchor.xyz) +
            quaternionRotate(parentToJoint, translation);
        const float3 childAnchor = quaternionRotate(candidateRotation, joint.childAnchor.xyz);
        scratch.bodyPosition[child] = scratch.jointPosition[child] - childAnchor;
        const float3 parentToJointPosition = scratch.jointPosition[child] - scratch.bodyPosition[parent];
        const float3 jointLinearVelocity = scratch.linearVelocity[parent] +
            cross(scratch.angularVelocity[parent], parentToJointPosition) + relativeLinear;
        scratch.angularVelocity[child] = scratch.angularVelocity[parent] + relativeAngular;
        scratch.linearVelocity[child] = jointLinearVelocity - cross(scratch.angularVelocity[child], childAnchor);
        const float3 jointLinearAcceleration = scratch.linearAcceleration[parent] +
            cross(scratch.angularAcceleration[parent], parentToJointPosition) +
            cross(scratch.angularVelocity[parent], cross(scratch.angularVelocity[parent], parentToJointPosition)) +
            2.0f * cross(scratch.angularVelocity[parent], relativeLinear) + relativeLinearDot;
        scratch.angularAcceleration[child] = scratch.angularAcceleration[parent] +
            cross(scratch.angularVelocity[parent], relativeAngular) + relativeAngularDot;
        scratch.linearAcceleration[child] = jointLinearAcceleration -
            cross(scratch.angularAcceleration[child], childAnchor) -
            cross(scratch.angularVelocity[child], cross(scratch.angularVelocity[child], childAnchor));
        if (!finite3(scratch.bodyPosition[child]) || !finite4(scratch.bodyRotation[child]) ||
            !finite3(scratch.angularVelocity[child]) || !finite3(scratch.linearVelocity[child]) ||
            !finite3(scratch.angularAcceleration[child]) || !finite3(scratch.linearAcceleration[child])) {
            fail(status, MR_ABA_NONFINITE_RESULT, joint.childBody);
            statuses[environment] = status;
            return;
        }
    }
    for (uint row = 0u; row < articulation.nv; ++row) {
        scratch.bias[row] = 0.0f;
        for (uint column = 0u; column < articulation.nv; ++column) scratch.mass[row * kMaxDofs + column] = 0.0f;
    }
    const uint wrenchBase = environment * dispatch.wrenchStride;
    for (uint localBody = 0u; localBody < articulation.bodyCount; ++localBody) {
        const uint globalBody = articulation.firstBody + localBody;
        device const MRBodyPropertiesGPU& body = bodies[globalBody];
        if (body.articulationIndex != dispatch.articulationIndex || body.motionType != MR_MOTION_DYNAMIC ||
            !(body.massAndInverseMass.x > 0.0f) || !finite4(body.massAndInverseMass) ||
            !finite4(body.inertiaRow0) || !finite4(body.inertiaRow1) || !finite4(body.inertiaRow2)) {
            fail(status, MR_ABA_INVALID_MODEL, globalBody);
            statuses[environment] = status;
            return;
        }
        float3 requiredForce = (scratch.linearAcceleration[localBody] - world.gravityAndTimestep.xyz) * body.massAndInverseMass.x;
        float3 requiredTorque = worldInertiaMultiply(body, scratch.bodyRotation[localBody], scratch.angularAcceleration[localBody]) +
            cross(scratch.angularVelocity[localBody], worldInertiaMultiply(body, scratch.bodyRotation[localBody], scratch.angularVelocity[localBody]));
        if ((dispatch.flags & MR_ABA_HAS_BODY_WRENCHES) != 0u) {
            device const MRABABodyWrenchGPU& wrench = bodyWrenches[wrenchBase + localBody];
            if (!finite4(wrench.force) || !finite4(wrench.torque) || wrench.force.w != 0.0f || wrench.torque.w != 0.0f) {
                fail(status, MR_ABA_NONFINITE_INPUT, globalBody);
                statuses[environment] = status;
                return;
            }
            requiredForce -= wrench.force.xyz;
            requiredTorque -= wrench.torque.xyz;
        }
        if ((dispatch.flags & MR_ABA_APPLY_BODY_DAMPING) != 0u) {
            requiredForce += scratch.linearVelocity[localBody] * body.dampingAndSpeedLimits.x;
            requiredTorque += scratch.angularVelocity[localBody] * body.dampingAndSpeedLimits.y;
        }
        for (uint row = 0u; row < articulation.nv; ++row) {
            const Motion left = bodyMotionForDof(localBody, row, articulation, joints, programs, q, v,
                scratch.bodyPosition, scratch.bodyRotation, scratch.jointPosition, scratch.jointAxis,
                scratch.inboundJoint, scratch.parentLocal);
            scratch.bias[row] += dot(left.angular, requiredTorque) + dot(left.linear, requiredForce);
            for (uint column = row; column < articulation.nv; ++column) {
                const Motion right = bodyMotionForDof(localBody, column, articulation, joints, programs, q, v,
                    scratch.bodyPosition, scratch.bodyRotation, scratch.jointPosition, scratch.jointAxis,
                    scratch.inboundJoint, scratch.parentLocal);
                const float value = dot(left.angular, worldInertiaMultiply(body, scratch.bodyRotation[localBody], right.angular)) +
                    body.massAndInverseMass.x * dot(left.linear, right.linear);
                scratch.mass[row * kMaxDofs + column] += value;
                if (row != column) scratch.mass[column * kMaxDofs + row] += value;
            }
        }
    }
    float minimumPivot = INFINITY;
    float maximumPivot = 0.0f;
    for (uint row = 0u; row < articulation.nv; ++row) {
        for (uint column = 0u; column < articulation.nv; ++column) scratch.factor[row * kMaxDofs + column] = scratch.mass[row * kMaxDofs + column];
        scratch.factor[row * kMaxDofs + row] += dofs[articulation.vOffset + row].drive.z;
    }
    for (uint row = 0u; row < articulation.nv; ++row) {
        float diagonalScale = 0.0f;
        for (uint column = 0u; column < articulation.nv; ++column) diagonalScale = max(diagonalScale, abs(scratch.factor[row * kMaxDofs + column]));
        for (uint column = 0u; column <= row; ++column) {
            float value = scratch.factor[row * kMaxDofs + column];
            for (uint inner = 0u; inner < column; ++inner) value -= scratch.factor[row * kMaxDofs + inner] * scratch.factor[column * kMaxDofs + inner];
            if (row == column) {
                if (!(value > max(kPivotFloor, diagonalScale * 6.0f * kEpsilon)) || !isfinite(value)) {
                    fail(status, MR_ABA_FACTORIZATION_FAILED, row);
                    status.diagnostics = float4(minimumPivot, maximumPivot, value, diagonalScale);
                    statuses[environment] = status;
                    return;
                }
                const float pivot = sqrt(value);
                scratch.factor[row * kMaxDofs + row] = pivot;
                minimumPivot = min(minimumPivot, pivot);
                maximumPivot = max(maximumPivot, pivot);
            } else {
                scratch.factor[row * kMaxDofs + column] = value / scratch.factor[column * kMaxDofs + column];
            }
        }
    }
    for (uint row = 0u; row < articulation.nv; ++row) {
        float value = effort[row] - scratch.bias[row];
        for (uint column = 0u; column < row; ++column) value -= scratch.factor[row * kMaxDofs + column] * scratch.acceleration[column];
        scratch.acceleration[row] = value / scratch.factor[row * kMaxDofs + row];
    }
    for (uint reverse = 0u; reverse < articulation.nv; ++reverse) {
        const uint row = articulation.nv - 1u - reverse;
        float value = scratch.acceleration[row];
        for (uint column = row + 1u; column < articulation.nv; ++column) value -= scratch.factor[column * kMaxDofs + row] * scratch.acceleration[column];
        scratch.acceleration[row] = value / scratch.factor[row * kMaxDofs + row];
    }
    const float timestep = world.gravityAndTimestep.w;
    float maximumAcceleration = 0.0f;
    for (uint index = 0u; index < articulation.nv; ++index) {
        if (!isfinite(scratch.acceleration[index])) {
            fail(status, MR_ABA_NONFINITE_RESULT, index);
            statuses[environment] = status;
            return;
        }
        scratch.nextV[index] = v[index] + timestep * scratch.acceleration[index];
        scratch.nextQ[index] = q[index] + timestep * scratch.nextV[index];
        if (!isfinite(scratch.nextV[index]) || !isfinite(scratch.nextQ[index])) {
            fail(status, MR_ABA_NONFINITE_RESULT, index);
            statuses[environment] = status;
            return;
        }
        maximumAcceleration = max(maximumAcceleration, abs(scratch.acceleration[index]));
    }
    const uint accelerationBase = environment * dispatch.accelerationStride + articulation.vOffset;
    const uint nextVBase = environment * dispatch.nextVStride + articulation.vOffset;
    const uint nextQBase = environment * dispatch.nextQStride + articulation.qOffset;
    for (uint index = 0u; index < articulation.nv; ++index) {
        accelerationOutput[accelerationBase + index] = scratch.acceleration[index];
        nextVOutput[nextVBase + index] = scratch.nextV[index];
        nextQOutput[nextQBase + index] = scratch.nextQ[index];
    }
    status.diagnostics = float4(isfinite(minimumPivot) ? minimumPivot : 0.0f, maximumPivot, maximumAcceleration, 0.0f);
    statuses[environment] = status;
}
