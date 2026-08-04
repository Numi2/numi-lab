#include <metal_stdlib>
using namespace metal;

#include "metalrobo/multicopter_types.h"

namespace {
bool finite4(const float4 value) { return all(isfinite(value)); }
float4 normalizedQuaternion(const float4 value) { return value * rsqrt(dot(value, value)); }
float3 rotate(const float4 raw, const float3 value) {
    const float4 q = normalizedQuaternion(raw);
    return value + 2.0f * cross(q.xyz, cross(q.xyz, value) + q.w * value);
}
float rotorSpeed(const MRMulticopterStateGPU state, const uint index) { return index < 4u ? state.rotorSpeed01[index] : state.rotorSpeed45[index - 4u]; }
void setRotorSpeed(thread MRMulticopterStateGPU& state, const uint index, const float value) { if (index < 4u) state.rotorSpeed01[index] = value; else state.rotorSpeed45[index - 4u] = value; }
}

kernel void mr_step_multicopters(
    device const MRMulticopterRotorGPU* rotors [[buffer(0)]],
    constant const MRMulticopterModelGPU& model [[buffer(1)]],
    device MRMulticopterStateGPU* states [[buffer(2)]],
    device const float* commands [[buffer(3)]],
    device const MRBodyStateGPU* bodies [[buffer(4)]],
    device MRBodyWrenchGPU* wrenches [[buffer(5)]],
    constant const MRMulticopterDispatchGPU& dispatch [[buffer(6)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) return;
    const uint bodyIndex = dispatch.bodyOffset + environment * dispatch.bodyStride;
    if (model.rotorCount == 0u || model.rotorCount > MR_MULTICOPTER_MAX_ROTORS ||
        !finite4(model.coefficients) || !finite4(model.motorAndTimestep) ||
        !(model.coefficients.x > 0.0f) || !(model.coefficients.y >= 0.0f) ||
        !(model.motorAndTimestep.x > 0.0f) || !(model.motorAndTimestep.y > 0.0f) ||
        !(model.motorAndTimestep.z > 0.0f) || !(model.motorAndTimestep.w > 0.0f) ||
        !finite4(dispatch.windVelocity) || !finite4(bodies[bodyIndex].orientation) ||
        !finite4(bodies[bodyIndex].linearVelocityAndInverseMass) ||
        !(dot(bodies[bodyIndex].orientation, bodies[bodyIndex].orientation) > 1.0e-12f)) {
        wrenches[bodyIndex] = {};
        return;
    }
    MRMulticopterStateGPU state = states[environment];
    float3 forceBody = float3(0.0f);
    float3 torqueBody = float3(0.0f);
    for (uint rotor = 0u; rotor < model.rotorCount; ++rotor) {
        const float target = clamp(commands[environment * model.rotorCount + rotor], 0.0f, model.motorAndTimestep.z);
        const float previous = rotorSpeed(state, rotor);
        if (!isfinite(target) || !isfinite(previous) || previous < 0.0f || !finite4(rotors[rotor].positionAndReactionSign)) { wrenches[bodyIndex] = {}; return; }
        const float tau = target > previous ? model.motorAndTimestep.x : model.motorAndTimestep.y;
        const float current = previous + (1.0f - exp(-model.motorAndTimestep.w / tau)) * (target - previous);
        setRotorSpeed(state, rotor, current);
        const float thrust = model.coefficients.x * current * current;
        const float3 rotorForce = float3(0.0f, 0.0f, thrust);
        forceBody += rotorForce;
        torqueBody += cross(rotors[rotor].positionAndReactionSign.xyz, rotorForce);
        torqueBody.z += rotors[rotor].positionAndReactionSign.w * model.coefficients.x * model.coefficients.y * current * current;
    }
    states[environment] = state;
    const float3 axisWorld = rotate(bodies[bodyIndex].orientation, float3(0.0f, 0.0f, 1.0f));
    const float3 relativeVelocity = bodies[bodyIndex].linearVelocityAndInverseMass.xyz - dispatch.windVelocity.xyz;
    const float3 perpendicularVelocity = relativeVelocity - dot(relativeVelocity, axisWorld) * axisWorld;
    float rotorSpeedSum = 0.0f;
    for (uint rotor = 0u; rotor < model.rotorCount; ++rotor) rotorSpeedSum += abs(rotorSpeed(state, rotor));
    const float3 rotorDrag = -rotorSpeedSum * model.coefficients.z * perpendicularVelocity;
    const float3 rollingMoment = -rotorSpeedSum * model.coefficients.w * perpendicularVelocity;
    wrenches[bodyIndex].force = float4(rotate(bodies[bodyIndex].orientation, forceBody) + rotorDrag, 0.0f);
    wrenches[bodyIndex].torque = float4(rotate(bodies[bodyIndex].orientation, torqueBody) + rollingMoment, 0.0f);
}
