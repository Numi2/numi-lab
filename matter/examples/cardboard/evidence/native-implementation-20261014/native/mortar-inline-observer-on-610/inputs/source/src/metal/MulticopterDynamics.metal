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

kernel void mr_mix_multicopter_actions(
    device const MRMulticopterRotorGPU* rotors [[buffer(0)]],
    constant const MRMulticopterModelGPU& model [[buffer(1)]],
    device const MRMulticopterActionGPU* actions [[buffer(2)]],
    device float* commands [[buffer(3)]],
    constant const MRMulticopterMixerGPU& mixer [[buffer(4)]],
    constant const MRMulticopterDispatchGPU& dispatch [[buffer(5)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount || model.rotorCount == 0u ||
        model.rotorCount > MR_MULTICOPTER_MAX_ROTORS) return;
    const float4 action = clamp(actions[environment].collectiveRollPitchYaw, -1.0f, 1.0f);
    for (uint rotor = 0u; rotor < model.rotorCount; ++rotor) {
        const float4 attachment = rotors[rotor].positionAndReactionSign;
        const float roll = sign(attachment.y) * action.y;
        const float pitch = -sign(attachment.x) * action.z;
        const float yaw = sign(attachment.w) * action.w;
        commands[environment * model.rotorCount + rotor] = clamp(
            mixer.hoverAndScales.x + mixer.hoverAndScales.y * action.x +
                mixer.hoverAndScales.z * (roll + pitch) + mixer.hoverAndScales.w * yaw,
            0.0f,
            model.motorAndTimestep.z
        );
    }
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

kernel void mr_step_compiled_multicopters(
    device const MRMulticopterRotorGPU* rotors [[buffer(0)]],
    constant const MRMulticopterModelGPU& model [[buffer(1)]],
    constant const MRMulticopterMixerGPU& mixer [[buffer(2)]],
    constant const MRCompiledMulticopterDispatchGPU& dispatch [[buffer(3)]],
    device const float* actionHistory [[buffer(4)]],
    device const uint* resetMasks [[buffer(5)]],
    device const float* qState [[buffer(6)]],
    device const float* vState [[buffer(7)]],
    device const MRMulticopterStateGPU* sourceStates [[buffer(8)]],
    device MRMulticopterStateGPU* candidateStates [[buffer(9)]],
    device MRABABodyWrenchGPU* wrenches [[buffer(10)]],
    constant const MRMetalWorldPassGPU& pass [[buffer(11)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount ||
        model.rotorCount == 0u ||
        model.rotorCount > MR_MULTICOPTER_MAX_ROTORS ||
        dispatch.actionCount < dispatch.firstAction + 4u) {
        return;
    }
    const uint wrenchIndex =
        environment * dispatch.bodyStride + dispatch.bodyIndex;
    const uint resetIndex =
        pass.controlStep * dispatch.environmentCount + environment;
    MRMulticopterStateGPU state = sourceStates[environment];
    if (resetMasks[resetIndex] != 0u) {
        state = {};
        for (uint rotor = 0u; rotor < model.rotorCount; ++rotor) {
            setRotorSpeed(state, rotor, mixer.hoverAndScales.x);
        }
    }
    const uint historyBase =
        environment * dispatch.actionHistoryStride +
        dispatch.filterSlot * dispatch.actionCount +
        dispatch.firstAction;
    const float4 action = clamp(float4(
        actionHistory[historyBase + 0u],
        actionHistory[historyBase + 1u],
        actionHistory[historyBase + 2u],
        actionHistory[historyBase + 3u]
    ), -1.0f, 1.0f);
    const uint qBase = environment * dispatch.qStride + dispatch.qOffset;
    const uint vBase = environment * dispatch.vStride + dispatch.vOffset;
    const float4 orientation = float4(
        qState[qBase + 3u], qState[qBase + 4u],
        qState[qBase + 5u], qState[qBase + 6u]
    );
    if (!finite4(orientation) ||
        !(dot(orientation, orientation) > 1.0e-12f)) {
        return;
    }
    float3 forceBody = float3(0.0f);
    float3 torqueBody = float3(0.0f);
    float rotorSpeedSum = 0.0f;
    for (uint rotor = 0u; rotor < model.rotorCount; ++rotor) {
        const float4 attachment = rotors[rotor].positionAndReactionSign;
        const float roll = sign(attachment.y) * action.y;
        const float pitch = -sign(attachment.x) * action.z;
        const float yaw = sign(attachment.w) * action.w;
        const float target = clamp(
            mixer.hoverAndScales.x +
                mixer.hoverAndScales.y * action.x +
                mixer.hoverAndScales.z * (roll + pitch) +
                mixer.hoverAndScales.w * yaw,
            0.0f,
            model.motorAndTimestep.z
        );
        const float previous = rotorSpeed(state, rotor);
        const float tau = target > previous
            ? model.motorAndTimestep.x
            : model.motorAndTimestep.y;
        const float current = previous +
            (1.0f - exp(-model.motorAndTimestep.w / tau)) *
                (target - previous);
        setRotorSpeed(state, rotor, current);
        rotorSpeedSum += abs(current);
        const float thrust = model.coefficients.x * current * current;
        const float3 rotorForce = float3(0.0f, 0.0f, thrust);
        forceBody += rotorForce;
        torqueBody += cross(attachment.xyz, rotorForce);
        torqueBody.z += attachment.w * model.coefficients.x *
            model.coefficients.y * current * current;
    }
    candidateStates[environment] = state;
    const float3 axisWorld = rotate(
        orientation,
        float3(0.0f, 0.0f, 1.0f)
    );
    const float3 linearVelocity = float3(
        vState[vBase + 0u], vState[vBase + 1u], vState[vBase + 2u]
    );
    const float3 relativeVelocity =
        linearVelocity - dispatch.windVelocity.xyz;
    const float3 perpendicularVelocity =
        relativeVelocity - dot(relativeVelocity, axisWorld) * axisWorld;
    const float3 rotorDrag =
        -rotorSpeedSum * model.coefficients.z * perpendicularVelocity;
    const float3 rollingMoment =
        -rotorSpeedSum * model.coefficients.w * perpendicularVelocity;
    MRABABodyWrenchGPU wrench = wrenches[wrenchIndex];
    wrench.force.xyz += rotate(orientation, forceBody) + rotorDrag;
    wrench.torque.xyz += rotate(orientation, torqueBody) + rollingMoment;
    wrenches[wrenchIndex] = wrench;
}

kernel void mr_commit_compiled_multicopters(
    device const MRMulticopterStateGPU* sourceStates [[buffer(0)]],
    device const MRMulticopterStateGPU* candidateStates [[buffer(1)]],
    device MRMulticopterStateGPU* destinationStates [[buffer(2)]],
    device const MRMetalWorldStatusGPU* statuses [[buffer(3)]],
    constant const MRCompiledMulticopterDispatchGPU& dispatch [[buffer(4)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    destinationStates[environment] =
        statuses[environment].code == MR_STEP_SUCCESS
        ? candidateStates[environment]
        : sourceStates[environment];
}

kernel void mr_evaluate_multicopter_flight_task(
    constant const MRMulticopterModelGPU& model [[buffer(0)]],
    device const MRMulticopterStateGPU* motors [[buffer(1)]],
    device const MRBodyStateGPU* bodies [[buffer(2)]],
    device MRMulticopterFlightTransitionGPU* transitions [[buffer(3)]],
    constant const MRMulticopterFlightTaskGPU& task [[buffer(4)]],
    constant const MRMulticopterDispatchGPU& dispatch [[buffer(5)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) return;
    const uint bodyIndex = dispatch.bodyOffset + environment * dispatch.bodyStride;
    const MRBodyStateGPU body = bodies[bodyIndex];
    MRMulticopterFlightTransitionGPU output{};
    if (!finite4(body.position) || !finite4(body.orientation) ||
        !finite4(body.linearVelocityAndInverseMass) || !finite4(body.angularVelocity)) {
        output.rewardAndDone = float4(-10.0f, 1.0f, INFINITY, INFINITY);
        transitions[environment] = output;
        return;
    }
    const float3 positionError = body.position.xyz - task.targetPositionAndMinimumHeight.xyz;
    const float3 up = rotate(body.orientation, float3(0.0f, 0.0f, 1.0f));
    const float tilt = acos(clamp(up.z, -1.0f, 1.0f));
    const float distance = length(positionError);
    output.positionErrorAndHeight = float4(positionError, body.position.z);
    output.linearVelocity = float4(body.linearVelocityAndInverseMass.xyz, 0.0f);
    output.bodyUpAndTilt = float4(up, tilt);
    output.angularVelocity = float4(body.angularVelocity.xyz, 0.0f);
    output.normalizedRotorSpeed = float4(
        rotorSpeed(motors[environment], 0u), rotorSpeed(motors[environment], 1u),
        rotorSpeed(motors[environment], 2u), rotorSpeed(motors[environment], 3u)
    ) / max(model.motorAndTimestep.z, 1.0f);
    const float velocity = length(body.linearVelocityAndInverseMass.xyz);
    const float reward = exp(-task.maximumHeightTiltAndScales.z * distance * distance) +
        0.5f * exp(-task.maximumHeightTiltAndScales.w * tilt * tilt) +
        0.25f * exp(-0.25f * velocity * velocity);
    const bool done = body.position.z < task.targetPositionAndMinimumHeight.w ||
        body.position.z > task.maximumHeightTiltAndScales.x || tilt > task.maximumHeightTiltAndScales.y;
    output.rewardAndDone = float4(reward, done ? 1.0f : 0.0f, tilt, distance);
    transitions[environment] = output;
}

kernel void mr_reset_multicopter_flights(
    device MRBodyStateGPU* bodies [[buffer(0)]],
    device MRMulticopterStateGPU* motors [[buffer(1)]],
    device const MRMulticopterFlightTransitionGPU* transitions [[buffer(2)]],
    device const MRBodyStateGPU* resetBodies [[buffer(3)]],
    device const MRMulticopterStateGPU* resetMotors [[buffer(4)]],
    constant const MRMulticopterDispatchGPU& dispatch [[buffer(5)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount || transitions[environment].rewardAndDone.y == 0.0f) return;
    const uint bodyIndex = dispatch.bodyOffset + environment * dispatch.bodyStride;
    bodies[bodyIndex] = resetBodies[bodyIndex];
    motors[environment] = resetMotors[environment];
}
