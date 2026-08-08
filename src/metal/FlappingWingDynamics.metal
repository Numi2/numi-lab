#include <metal_stdlib>
using namespace metal;

#include "metalrobo/flapping_wing_types.h"

namespace {

bool finite4(const float4 value) { return all(isfinite(value)); }

float4 normalizeQuaternion(const float4 raw) {
    const float normSquared = dot(raw, raw);
    return normSquared > 1.0e-12f
        ? raw * rsqrt(normSquared)
        : float4(0.0f, 0.0f, 0.0f, 1.0f);
}

float4 multiplyQuaternion(const float4 left, const float4 right) {
    return float4(
        left.w * right.xyz + right.w * left.xyz + cross(left.xyz, right.xyz),
        left.w * right.w - dot(left.xyz, right.xyz)
    );
}

float4 axisAngle(const float3 axis, const float angle) {
    const float halfAngle = 0.5f * angle;
    const float sine = sin(halfAngle);
    return float4(axis * sine, cos(halfAngle));
}

float3 rotate(const float4 raw, const float3 value) {
    const float4 q = normalizeQuaternion(raw);
    return value + 2.0f * cross(q.xyz, cross(q.xyz, value) + q.w * value);
}

float3 safeNormal(const float3 value) {
    const float magnitudeSquared = dot(value, value);
    return magnitudeSquared > 1.0e-12f
        ? value * rsqrt(magnitudeSquared)
        : float3(0.0f);
}

} // namespace

// Bilateral blade-element loads from the accepted source state.  Each thread
// owns two distinct wing-body wrench records, so no atomics or host
// aggregation enter the hot loop.  The generic articulated ABA consumes the
// resulting world-frame wrenches in this same microstep.
kernel void mr_step_compiled_flapping_wings(
    constant const MRFlappingWingGPU* wings [[buffer(0)]],
    constant const MRCompiledFlappingWingDispatchGPU& dispatch [[buffer(1)]],
    device const float* qState [[buffer(2)]],
    device const float* vState [[buffer(3)]],
    device MRABABodyWrenchGPU* wrenches [[buffer(4)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount ||
        !finite4(dispatch.windVelocityAndDensity) ||
        !(dispatch.windVelocityAndDensity.w > 0.0f)) {
        return;
    }
    const uint qBase = environment * dispatch.qStride + dispatch.qOffset;
    const uint vBase = environment * dispatch.vStride + dispatch.vOffset;
    const float4 rootOrientation = float4(
        qState[qBase + 3u], qState[qBase + 4u],
        qState[qBase + 5u], qState[qBase + 6u]
    );
    if (!finite4(rootOrientation) || !(dot(rootOrientation, rootOrientation) > 1.0e-12f)) {
        return;
    }
    const float3 rootVelocity = float3(
        vState[vBase + 0u], vState[vBase + 1u], vState[vBase + 2u]
    );
    const float3 rootAngularVelocity = float3(
        vState[vBase + 3u], vState[vBase + 4u], vState[vBase + 5u]
    );
    for (uint side = 0u; side < 2u; ++side) {
        const MRFlappingWingGPU wing = wings[side];
        if (wing.bodyIndex >= dispatch.bodyStride ||
            wing.qIndex < dispatch.qOffset || wing.vIndex < dispatch.vOffset ||
            !finite4(wing.rootToCenterAndArea) ||
            !finite4(wing.hingeAxisAndChord) || !finite4(wing.coefficients) ||
            !(wing.rootToCenterAndArea.w > 0.0f) ||
            !(wing.hingeAxisAndChord.w > 0.0f)) {
            continue;
        }
        const float3 hingeAxis = safeNormal(wing.hingeAxisAndChord.xyz);
        if (dot(hingeAxis, hingeAxis) == 0.0f) {
            continue;
        }
        const float angle = qState[environment * dispatch.qStride + wing.qIndex];
        const float angularRate = vState[environment * dispatch.vStride + wing.vIndex];
        if (!isfinite(angle) || !isfinite(angularRate)) {
            continue;
        }
        const float4 hingeRotation = axisAngle(hingeAxis, angle);
        const float4 wingOrientation = multiplyQuaternion(
            normalizeQuaternion(rootOrientation), hingeRotation
        );
        const float3 rootToCenter = rotate(
            rootOrientation,
            rotate(hingeRotation, wing.rootToCenterAndArea.xyz)
        );
        const float3 hingeAxisWorld = rotate(rootOrientation, hingeAxis);
        const float3 wingVelocity = rootVelocity +
            cross(rootAngularVelocity, rootToCenter) +
            cross(hingeAxisWorld * angularRate, rootToCenter);
        const float3 incomingAir = dispatch.windVelocityAndDensity.xyz - wingVelocity;
        const float speedSquared = dot(incomingAir, incomingAir);
        if (!(speedSquared > 1.0e-8f) || !isfinite(speedSquared)) {
            continue;
        }
        const float3 flow = incomingAir * rsqrt(speedSquared);
        const float3 chord = rotate(wingOrientation, float3(1.0f, 0.0f, 0.0f));
        // The authored root-to-center vector carries the bilateral span sign;
        // a mirrored wing cannot share the left wing's positive-y span basis.
        const float spanSign = wing.rootToCenterAndArea.y >= 0.0f ? 1.0f : -1.0f;
        const float3 span = rotate(
            wingOrientation, float3(0.0f, spanSign, 0.0f)
        );
        const float3 normal = safeNormal(cross(chord, span));
        const float angleOfAttack = atan2(dot(flow, normal), dot(flow, chord));
        const float liftCoefficient = clamp(
            wing.coefficients.x * angleOfAttack,
            -wing.coefficients.w,
            wing.coefficients.w
        );
        const float dragCoefficient = wing.coefficients.y +
            wing.coefficients.z * liftCoefficient * liftCoefficient;
        const float dynamicPressure = 0.5f *
            dispatch.windVelocityAndDensity.w * speedSquared;
        const float liftMagnitude = dynamicPressure *
            wing.rootToCenterAndArea.w * liftCoefficient;
        const float dragMagnitude = dynamicPressure *
            wing.rootToCenterAndArea.w * dragCoefficient;
        const float3 liftDirection = safeNormal(cross(span, flow));
        // A single hinge cannot explicitly represent the rapid passive
        // feathering of a bird wing.  This quasi-steady closure directs the
        // stroke-dependent component through the airframe's local up axis;
        // it is deliberately an authored hybrid assumption, not a measured
        // Deetjen coefficient.  It vanishes without angular wing motion and
        // remains responsive to the resolved body/wind relative flow.
        const float3 strokeVelocity = cross(
            hingeAxisWorld * angularRate, rootToCenter
        );
        const float strokeSpeedSquared = dot(strokeVelocity, strokeVelocity);
        const float strokeFraction = clamp(
            strokeSpeedSquared / max(speedSquared, 1.0e-8f), 0.0f, 1.0f
        );
        const float strokeLiftMagnitude = dynamicPressure *
            wing.rootToCenterAndArea.w * wing.coefficients.w * strokeFraction;
        const float3 airframeUp = rotate(rootOrientation, float3(0.0f, 0.0f, 1.0f));
        const float3 force = liftMagnitude * liftDirection +
            dragMagnitude * flow + strokeLiftMagnitude * airframeUp;
        if (!all(isfinite(force))) {
            continue;
        }
        const uint wrenchIndex = environment * dispatch.bodyStride + wing.bodyIndex;
        MRABABodyWrenchGPU wrench = wrenches[wrenchIndex];
        wrench.force.xyz += force;
        // A quarter-chord aerodynamic center supplies a moment about the wing
        // COM while the force itself remains a physical external wrench.
        const float3 aerodynamicCenter = -0.25f * wing.hingeAxisAndChord.w * chord;
        wrench.torque.xyz += cross(aerodynamicCenter, force);
        wrenches[wrenchIndex] = wrench;
    }
}
