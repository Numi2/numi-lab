#include <metal_stdlib>
using namespace metal;

#include "metalrobo/measured_surface_types.h"
#include "metalrobo/task_program_types.h"

namespace {
#define MR_MEASURED_SURFACE_THREADS 256u

bool finite4(const float4 value) { return all(isfinite(value)); }
float stateLane(const thread MRMeasuredSurfaceStateGPU& state, uint index) {
    return state.position[index >> 2u][index & 3u];
}
float velocityLane(const thread MRMeasuredSurfaceStateGPU& state, uint index) {
    return state.velocity[index >> 2u][index & 3u];
}
float groupStateLane(
    const threadgroup MRMeasuredSurfaceStateGPU& state, uint index
) {
    return state.position[index >> 2u][index & 3u];
}
float groupVelocityLane(
    const threadgroup MRMeasuredSurfaceStateGPU& state, uint index
) {
    return state.velocity[index >> 2u][index & 3u];
}
void setStateLane(thread MRMeasuredSurfaceStateGPU& state, uint index, float value) {
    state.position[index >> 2u][index & 3u] = value;
}
void setVelocityLane(thread MRMeasuredSurfaceStateGPU& state, uint index, float value) {
    state.velocity[index >> 2u][index & 3u] = value;
}
float4 normalizedQuaternion(float4 value) {
    return value * rsqrt(dot(value, value));
}
float3 rotate(float4 raw, float3 value) {
    const float4 q = normalizedQuaternion(raw);
    return value + 2.0f * cross(q.xyz, cross(q.xyz, value) + q.w * value);
}
float3 rotateAxis(float3 point, float3 axis, float angle) {
    const float sine = sin(angle), cosine = cos(angle);
    return point * cosine + cross(axis, point) * sine +
        axis * dot(axis, point) * (1.0f - cosine);
}
float3 measuredPoint(
    constant const MRMeasuredSurfaceModelGPU& model,
    device const float* positions,
    float phase,
    uint vertexIndex
) {
    phase = clamp(phase, 0.0f, float(model.frameCount - 1u));
    const uint first = min(uint(floor(phase)), model.frameCount - 1u);
    const uint second = min(first + 1u, model.frameCount - 1u);
    const float blend = phase - float(first);
    const uint a = (first * model.vertexCount + vertexIndex) * 3u;
    const uint b = (second * model.vertexCount + vertexIndex) * 3u;
    return mix(
        float3(positions[a], positions[a + 1u], positions[a + 2u]),
        float3(positions[b], positions[b + 1u], positions[b + 2u]),
        blend
    );
}
float3 surfacePoint(
    constant const MRMeasuredSurfaceModelGPU& model,
    device const float* positions,
    device const uchar* parts,
    const threadgroup MRMeasuredSurfaceStateGPU& state,
    uint vertexIndex
) {
    const uint part = uint(parts[vertexIndex]);
    const uint actionBase = part == 2u ? 4u : (part == 3u ? 12u : 4u);
    const float phaseShift = 0.10f * groupStateLane(state, 0u) +
        ((part == 2u || part == 3u)
            ? 0.08f * groupStateLane(state, actionBase)
            : 0.0f);
    float3 point = measuredPoint(
        model, positions, state.phaseRateImpulseStep.x + phaseShift, vertexIndex);
    if (part == 2u || part == 3u || part == 4u) {
        const float3 reference = measuredPoint(
            model, positions, 0.0f, vertexIndex);
        const float measuredAmplitude = clamp(
            1.0f + groupStateLane(state, 2u), 0.5f, 2.25f);
        point = reference + measuredAmplitude * (point - reference);
    }
    const float3 span = model.boundsMaximum.xyz - model.boundsMinimum.xyz;
    const float normalizedX =
        (point.x - model.boundsMinimum.x) / max(span.x, 1.0e-6f);
    point.z += 0.004f * groupStateLane(state, 1u) *
        (part == 1u ? 0.25f : 1.0f);
    if (part == 2u || part == 3u) {
        const uint action = part == 2u ? 4u : 12u;
        const float side = part == 2u ? 1.0f : -1.0f;
        const float hingeY = side * 0.020f;
        const float normalizedSpan = clamp(
            abs(point.y - hingeY) / max(0.001f, 0.5f * span.y),
            0.0f, 1.0f);
        float3 relative = point - float3(
            model.centerAndRadius.x, hingeY, model.centerAndRadius.z);
        relative.y *= 1.0f + 0.18f *
            groupStateLane(state, action + 6u) * normalizedSpan;
        relative = rotateAxis(relative, float3(1.0f, 0.0f, 0.0f),
            side * (0.28f * groupStateLane(state, action + 1u) +
                    0.18f * groupStateLane(state, action + 2u) +
                    0.10f * groupStateLane(state, 2u) +
                    0.08f * groupStateLane(state, 3u)) * normalizedSpan);
        relative = rotateAxis(relative, float3(0.0f, 0.0f, 1.0f),
            -side * 0.22f * groupStateLane(state, action + 5u) * normalizedSpan);
        relative = rotateAxis(relative,
            normalize(float3(0.0f, side, 0.08f)),
            side * (0.34f * groupStateLane(state, action + 3u) +
                    0.25f * groupStateLane(state, action + 4u) * normalizedSpan) *
                normalizedSpan);
        point = float3(model.centerAndRadius.x, hingeY,
                       model.centerAndRadius.z) + relative;
        point.x += 0.030f * groupStateLane(state, action + 5u) * normalizedSpan;
        point.z += 0.012f * groupStateLane(state, action + 7u) *
            sin(M_PI_F * normalizedX) * normalizedSpan;
    } else if (part == 4u) {
        const float3 pivot = float3(
            model.boundsMinimum.x,
            model.centerAndRadius.y,
            model.centerAndRadius.z);
        float3 relative = point - pivot;
        relative = rotateAxis(relative, float3(0.0f, 1.0f, 0.0f),
            -0.30f * groupStateLane(state, 20u));
        relative = rotateAxis(relative, float3(0.0f, 0.0f, 1.0f),
            0.25f * groupStateLane(state, 21u));
        relative = rotateAxis(relative, float3(1.0f, 0.0f, 0.0f),
            0.28f * groupStateLane(state, 22u));
        relative.y *= 1.0f + 0.22f * groupStateLane(state, 23u);
        point = pivot + relative;
    }
    return point;
}

void advancePhase(
    constant const MRMeasuredSurfaceModelGPU& model,
    thread MRMeasuredSurfaceStateGPU& state,
    float timestep
) {
    float direction = state.phaseRateImpulseStep.y;
    if (direction == 0.0f) direction = 1.0f;
    const float rateScale = clamp(1.0f + stateLane(state, 0u), 0.25f, 2.0f);
    float phase = state.phaseRateImpulseStep.x +
        direction * model.samplingAndAerodynamics.x * timestep * rateScale;
    const float maximum = float(model.frameCount - 1u);
    if (model.phaseBoundaryMode == MR_MEASURED_SURFACE_PHASE_REFLECT) {
        while (phase > maximum || phase < 0.0f) {
            if (phase > maximum) { phase = 2.0f * maximum - phase; direction = -1.0f; }
            if (phase < 0.0f) { phase = -phase; direction = 1.0f; }
        }
    } else if (model.phaseBoundaryMode == MR_MEASURED_SURFACE_PHASE_WRAP) {
        phase = fmod(maximum + fmod(phase, maximum), maximum);
    } else {
        phase = clamp(phase, 0.0f, maximum);
        if (phase == 0.0f || phase == maximum) direction = 0.0f;
    }
    state.phaseRateImpulseStep.x = phase;
    state.phaseRateImpulseStep.y = direction;
}
}

kernel void mr_step_measured_surface_mechanics(
    constant const MRMeasuredSurfaceModelGPU& model [[buffer(0)]],
    device const MRMeasuredSurfaceActionGPU* actions [[buffer(1)]],
    device const float* positions [[buffer(2)]],
    device const ushort* triangles [[buffer(3)]],
    device const uchar* vertexParts [[buffer(4)]],
    constant const MRCompiledMeasuredSurfaceDispatchGPU& dispatch [[buffer(5)]],
    device const float* actionHistory [[buffer(6)]],
    device const uint* resetMasks [[buffer(7)]],
    device const float* qState [[buffer(8)]],
    device const float* vState [[buffer(9)]],
    device MRMeasuredSurfaceStateGPU* acceptedStates [[buffer(10)]],
    device MRMeasuredSurfaceStateGPU* candidateStates [[buffer(11)]],
    device MRMeasuredSurfaceStateGPU* checkpointStates [[buffer(12)]],
    device MRMeasuredSurfaceEvidenceGPU* acceptedEvidence [[buffer(13)]],
    device MRMeasuredSurfaceEvidenceGPU* candidateEvidence [[buffer(14)]],
    device MRMeasuredSurfaceEvidenceGPU* checkpointEvidence [[buffer(15)]],
    device MRABABodyWrenchGPU* bodyWrenches [[buffer(16)]],
    device MRMetalWorldStatusGPU* statuses [[buffer(17)]],
    constant const MRMetalWorldPassGPU& pass [[buffer(18)]],
    threadgroup float4* forceScratch [[threadgroup(0)]],
    threadgroup float4* torqueScratch [[threadgroup(1)]],
    threadgroup float2* scalarScratch [[threadgroup(2)]],
    threadgroup uint* invalidScratch [[threadgroup(3)]],
    threadgroup MRMeasuredSurfaceStateGPU* stateScratch [[threadgroup(4)]],
    uint lane [[thread_index_in_threadgroup]],
    uint simdLane [[thread_index_in_simdgroup]],
    uint simdGroup [[simdgroup_index_in_threadgroup]],
    uint environment [[threadgroup_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount || lane >= MR_MEASURED_SURFACE_THREADS) return;
    const uint resetIndex = pass.controlStep * dispatch.environmentCount + environment;
    if (lane == 0u && pass.physicsSubstep == 0u) {
        if (resetMasks[resetIndex] != 0u) {
            acceptedStates[environment] = {};
            for (uint action = 0u; action < model.actionCount; ++action) {
                const float normalized =
                    actions[action].normalizedBiasReserved.x;
                const float4 contract =
                    actions[action].boundsFrequencyDamping;
                const float target = normalized >= 0.0f
                    ? normalized * contract.y
                    : -normalized * contract.x;
                acceptedStates[environment].position[action >> 2u]
                    [action & 3u] = target;
            }
            acceptedStates[environment].phaseRateImpulseStep.y = 1.0f;
            acceptedEvidence[environment] = {};
        }
        checkpointStates[environment] = acceptedStates[environment];
        checkpointEvidence[environment] = acceptedEvidence[environment];
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (lane == 0u) {
        MRMeasuredSurfaceStateGPU candidate = acceptedStates[environment];
        const uint historyBase = environment * dispatch.actionHistoryStride +
            dispatch.filterSlot * dispatch.actionCount + dispatch.firstAction;
        for (uint action = 0u; action < model.actionCount; ++action) {
            const float normalized = clamp(
                actions[action].normalizedBiasReserved.x +
                    actionHistory[historyBase + action],
                -1.0f, 1.0f);
            const float4 contract = actions[action].boundsFrequencyDamping;
            const float target = normalized >= 0.0f
                ? normalized * contract.y
                : -normalized * contract.x;
            float position = stateLane(candidate, action);
            float velocity = velocityLane(candidate, action);
            const float omega = 2.0f * M_PI_F * contract.z;
            const float acceleration = omega * omega * (target - position) -
                2.0f * contract.w * omega * velocity;
            velocity += dispatch.timestepAndWindX.x * acceleration;
            position += dispatch.timestepAndWindX.x * velocity;
            setVelocityLane(candidate, action, velocity);
            setStateLane(candidate, action, position);
        }
        advancePhase(model, candidate, dispatch.timestepAndWindX.x);
        candidate.phaseRateImpulseStep.w += 1.0f;
        candidateStates[environment] = candidate;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (lane < MR_MEASURED_SURFACE_ACTION_CAPACITY / 4u) {
        stateScratch[0].position[lane] =
            acceptedStates[environment].position[lane];
        stateScratch[0].velocity[lane] =
            acceptedStates[environment].velocity[lane];
        stateScratch[1].position[lane] =
            candidateStates[environment].position[lane];
        stateScratch[1].velocity[lane] =
            candidateStates[environment].velocity[lane];
    }
    if (lane == 0u) {
        stateScratch[0].phaseRateImpulseStep =
            acceptedStates[environment].phaseRateImpulseStep;
        stateScratch[1].phaseRateImpulseStep =
            candidateStates[environment].phaseRateImpulseStep;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const threadgroup MRMeasuredSurfaceStateGPU& source = stateScratch[0];
    const threadgroup MRMeasuredSurfaceStateGPU& candidate = stateScratch[1];
    const uint qBase = environment * dispatch.qStride + dispatch.qOffset;
    const uint vBase = environment * dispatch.vStride + dispatch.vOffset;
    const float4 orientation = float4(
        qState[qBase + 3u], qState[qBase + 4u],
        qState[qBase + 5u], qState[qBase + 6u]);
    const float3 bodyVelocity = float3(
        vState[vBase], vState[vBase + 1u], vState[vBase + 2u]);
    const float3 bodyOmega = float3(
        vState[vBase + 3u], vState[vBase + 4u], vState[vBase + 5u]);
    bool baseValid = model.abiVersion == MR_MEASURED_SURFACE_ABI_VERSION &&
        model.actionCount <= MR_MEASURED_SURFACE_ACTION_CAPACITY &&
        dispatch.actionCount >= dispatch.firstAction + model.actionCount &&
        dispatch.timestepAndWindX.x > 0.0f && finite4(orientation) &&
        dot(orientation, orientation) > 1.0e-12f &&
        finite4(float4(bodyVelocity, 0.0f)) && finite4(float4(bodyOmega, 0.0f));
    for (uint action = lane; action < model.actionCount;
         action += MR_MEASURED_SURFACE_THREADS) {
        baseValid = baseValid &&
            isfinite(groupStateLane(source, action)) &&
            isfinite(groupVelocityLane(source, action)) &&
            isfinite(groupStateLane(candidate, action)) &&
            isfinite(groupVelocityLane(candidate, action));
    }
    if (lane == 0u) {
        baseValid = baseValid && finite4(source.phaseRateImpulseStep) &&
            finite4(candidate.phaseRateImpulseStep);
    }
    float3 force = float3(0.0f), torque = float3(0.0f);
    float areaSum = 0.0f, maximumDeformation = 0.0f;
    uint invalid = baseValid ? 0u : 1u;
    for (uint triangle = lane; triangle < model.triangleCount; triangle += MR_MEASURED_SURFACE_THREADS) {
        const ushort3 indices = ushort3(
            triangles[3u * triangle], triangles[3u * triangle + 1u],
            triangles[3u * triangle + 2u]);
        const float3 a0 = surfacePoint(model, positions, vertexParts, source, indices.x);
        const float3 b0 = surfacePoint(model, positions, vertexParts, source, indices.y);
        const float3 c0 = surfacePoint(model, positions, vertexParts, source, indices.z);
        const float3 a = surfacePoint(model, positions, vertexParts, candidate, indices.x);
        const float3 b = surfacePoint(model, positions, vertexParts, candidate, indices.y);
        const float3 c = surfacePoint(model, positions, vertexParts, candidate, indices.z);
        const float3 normalRaw = cross(b - a, c - a);
        const float twiceArea = length(normalRaw);
        if (!(twiceArea > 1.0e-12f) || !isfinite(twiceArea)) continue;
        const float3 normalLocal = normalRaw / twiceArea;
        const float area = 0.5f * twiceArea;
        const float3 centroid = (a + b + c) / 3.0f;
        const float3 surfaceVelocityLocal =
            ((a - a0) + (b - b0) + (c - c0)) /
            (3.0f * dispatch.timestepAndWindX.x);
        const float3 armWorld = rotate(
            orientation, centroid - model.centerAndRadius.xyz);
        const float3 velocityWorld = bodyVelocity +
            cross(bodyOmega, armWorld) + rotate(orientation, surfaceVelocityLocal);
        const float3 wind = float3(
            dispatch.timestepAndWindX.y,
            dispatch.timestepAndWindX.z,
            dispatch.timestepAndWindX.w);
        const float3 relative = velocityWorld - wind;
        const float3 normalWorld = rotate(orientation, normalLocal);
        const float normalSpeed = dot(relative, normalWorld);
        const float3 tangential = relative - normalSpeed * normalWorld;
        const float density = model.samplingAndAerodynamics.y;
        const float relativeSpeed = length(relative);
        const float normalRatio = abs(normalSpeed) /
            max(relativeSpeed, 1.0e-6f);
        const float separated = smoothstep(
            model.aerodynamicCorrections.x, 1.0f, normalRatio);
        const float normalRetention = mix(
            1.0f, model.aerodynamicCorrections.y, separated);
        const float dynamicScale = -0.5f * density * area;
        float3 normalForce = dynamicScale *
            model.samplingAndAerodynamics.z * normalRetention *
            abs(normalSpeed) * normalSpeed * normalWorld;
        const float wingSpan = max(
            model.boundsMaximum.y - model.boundsMinimum.y, 1.0e-3f);
        const float groundScale = max(
            model.aerodynamicCorrections.w * wingSpan, 1.0e-3f);
        const float rootHeight = max(qState[qBase + 2u], 0.0f);
        const float groundRatio = rootHeight / groundScale;
        const float groundLiftMultiplier = 1.0f +
            model.aerodynamicCorrections.z /
                (1.0f + groundRatio * groundRatio);
        if (normalForce.z > 0.0f) normalForce *= groundLiftMultiplier;
        const float3 tangentialForce = dynamicScale *
            model.samplingAndAerodynamics.w * length(tangential) * tangential;
        const float3 triangleForce = normalForce + tangentialForce;
        if (!finite4(float4(triangleForce, 0.0f))) { invalid = 1u; continue; }
        force += triangleForce;
        torque += cross(armWorld, triangleForce);
        areaSum += area;
        maximumDeformation = max(maximumDeformation,
            max(length(a - measuredPoint(model, positions, candidate.phaseRateImpulseStep.x, indices.x)),
                max(length(b - measuredPoint(model, positions, candidate.phaseRateImpulseStep.x, indices.y)),
                    length(c - measuredPoint(model, positions, candidate.phaseRateImpulseStep.x, indices.z)))));
    }
    const float4 simdForce = float4(
        simd_sum(force.x), simd_sum(force.y), simd_sum(force.z), 0.0f);
    const float4 simdTorque = float4(
        simd_sum(torque.x), simd_sum(torque.y), simd_sum(torque.z), 0.0f);
    const float2 simdScalars = float2(
        simd_sum(areaSum), simd_max(maximumDeformation));
    const uint simdInvalid = simd_sum(invalid);
    if (simdLane == 0u) {
        forceScratch[simdGroup] = simdForce;
        torqueScratch[simdGroup] = simdTorque;
        scalarScratch[simdGroup] = simdScalars;
        invalidScratch[simdGroup] = simdInvalid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdGroup == 0u) {
        const bool livePartial = simdLane <
            MR_MEASURED_SURFACE_THREADS / 32u;
        const float4 partialForce = livePartial
            ? forceScratch[simdLane] : float4(0.0f);
        const float4 partialTorque = livePartial
            ? torqueScratch[simdLane] : float4(0.0f);
        const float2 partialScalars = livePartial
            ? scalarScratch[simdLane] : float2(0.0f);
        const uint partialInvalid = livePartial
            ? invalidScratch[simdLane] : 0u;
        const float4 finalForce = float4(
            simd_sum(partialForce.x),
            simd_sum(partialForce.y),
            simd_sum(partialForce.z), 0.0f);
        const float4 finalTorque = float4(
            simd_sum(partialTorque.x),
            simd_sum(partialTorque.y),
            simd_sum(partialTorque.z), 0.0f);
        const float2 finalScalars = float2(
            simd_sum(partialScalars.x),
            simd_max(partialScalars.y));
        const uint finalInvalid = simd_sum(partialInvalid);
        if (simdLane == 0u) {
            forceScratch[0] = finalForce;
            torqueScratch[0] = finalTorque;
            scalarScratch[0] = finalScalars;
            invalidScratch[0] = finalInvalid;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        float actuatorNormSquared = 0.0f;
        for (uint action = 0u; action < model.actionCount; ++action) {
            const float value = groupStateLane(candidate, action);
            actuatorNormSquared += value * value;
        }
        const bool valid = invalidScratch[0] == 0u &&
            finite4(forceScratch[0]) && finite4(torqueScratch[0]) &&
            isfinite(scalarScratch[0].x) && isfinite(scalarScratch[0].y) &&
            isfinite(actuatorNormSquared);
        candidateEvidence[environment].loadsAreaPhase = float4(
            length(forceScratch[0].xyz), length(torqueScratch[0].xyz),
            scalarScratch[0].x, candidate.phaseRateImpulseStep.x);
        candidateEvidence[environment].deformationActuationStatus = float4(
            scalarScratch[0].y, sqrt(actuatorNormSquared), valid ? 0.0f : 1.0f, 0.0f);
        candidateEvidence[environment].worldForceAndMagnitude = float4(
            forceScratch[0].xyz, length(forceScratch[0].xyz));
        candidateEvidence[environment].worldTorqueAndMagnitude = float4(
            torqueScratch[0].xyz, length(torqueScratch[0].xyz));
        candidateEvidence[environment].worldForceImpulseAndTime =
            acceptedEvidence[environment].worldForceImpulseAndTime;
        candidateEvidence[environment].worldTorqueImpulse =
            acceptedEvidence[environment].worldTorqueImpulse;
        if (valid) {
            candidateEvidence[environment].worldForceImpulseAndTime.xyz +=
                dispatch.timestepAndWindX.x * forceScratch[0].xyz;
            candidateEvidence[environment].worldForceImpulseAndTime.w +=
                dispatch.timestepAndWindX.x;
            candidateEvidence[environment].worldTorqueImpulse.xyz +=
                dispatch.timestepAndWindX.x * torqueScratch[0].xyz;
            MRMeasuredSurfaceStateGPU accumulated =
                candidateStates[environment];
            accumulated.phaseRateImpulseStep.z +=
                dispatch.timestepAndWindX.x * length(forceScratch[0].xyz);
            candidateStates[environment] = accumulated;
            const uint wrenchIndex = environment * dispatch.bodyStride + dispatch.bodyIndex;
            MRABABodyWrenchGPU wrench = bodyWrenches[wrenchIndex];
            wrench.force.xyz += forceScratch[0].xyz;
            wrench.torque.xyz += torqueScratch[0].xyz;
            bodyWrenches[wrenchIndex] = wrench;
        } else if (statuses[environment].code == MR_STEP_SUCCESS) {
            statuses[environment].code = MR_STEP_NONFINITE_INPUT;
            statuses[environment].failingSubstep = pass.physicsSubstep;
            statuses[environment].failingIndex = MR_INVALID_INDEX;
        }
    }
}

kernel void mr_commit_measured_surface_mechanics(
    device MRMeasuredSurfaceStateGPU* acceptedStates [[buffer(0)]],
    device const MRMeasuredSurfaceStateGPU* candidateStates [[buffer(1)]],
    device const MRMeasuredSurfaceStateGPU* checkpointStates [[buffer(2)]],
    device MRMeasuredSurfaceEvidenceGPU* acceptedEvidence [[buffer(3)]],
    device const MRMeasuredSurfaceEvidenceGPU* candidateEvidence [[buffer(4)]],
    device const MRMeasuredSurfaceEvidenceGPU* checkpointEvidence [[buffer(5)]],
    device const MRMetalWorldStatusGPU* statuses [[buffer(6)]],
    constant const MRCompiledMeasuredSurfaceDispatchGPU& dispatch [[buffer(7)]],
    device MRTaskStateGPU* taskStates [[buffer(8)]],
    uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) return;
    if (statuses[environment].code == MR_STEP_SUCCESS &&
        candidateEvidence[environment].deformationActuationStatus.z == 0.0f) {
        acceptedStates[environment] = candidateStates[environment];
        acceptedEvidence[environment] = candidateEvidence[environment];
    } else {
        // MetalWorld rolls q/v and contact back to the control-step checkpoint,
        // not merely the preceding physics substep. Surface phase, filters,
        // and evidence must restore to the same transaction boundary.
        acceptedStates[environment] = checkpointStates[environment];
        acceptedEvidence[environment] = checkpointEvidence[environment];
    }
    const MRMeasuredSurfaceStateGPU accepted = acceptedStates[environment];
    const MRMeasuredSurfaceEvidenceGPU evidence = acceptedEvidence[environment];
    const float maximumPhase = max(float(dispatch.reserved0 - 1u), 1.0f);
    taskStates[environment].deviceMechanics = float4(
        accepted.phaseRateImpulseStep.x / maximumPhase,
        accepted.phaseRateImpulseStep.y,
        evidence.loadsAreaPhase.x,
        evidence.deformationActuationStatus.y);
}
