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
    constant const MRAeroTailGPU& tail [[buffer(5)]],
    constant const MRAeroFuselageGPU& fuselage [[buffer(6)]],
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
    // A free airframe cannot have mass/collision only: root-relative flow
    // produces a body drag wrench even when the wing load is momentarily
    // symmetric.  Axis-specific reference areas keep this generic primitive
    // tied to the resolved airframe orientation rather than world axes.
    if (fuselage.bodyIndex == dispatch.rootBodyIndex &&
        fuselage.rootBodyIndex == dispatch.rootBodyIndex &&
        fuselage.bodyIndex < dispatch.bodyStride &&
        finite4(fuselage.referenceAreasAndDrag) &&
        finite4(fuselage.angularDamping) &&
        fuselage.referenceAreasAndDrag.x > 0.0f &&
        fuselage.referenceAreasAndDrag.y > 0.0f &&
        fuselage.referenceAreasAndDrag.z > 0.0f &&
        fuselage.referenceAreasAndDrag.w >= 0.0f &&
        all(fuselage.angularDamping.xyz >= 0.0f)) {
        const float3 forward = safeNormal(
            rotate(rootOrientation, float3(1.0f, 0.0f, 0.0f))
        );
        const float3 span = safeNormal(
            rotate(rootOrientation, float3(0.0f, 1.0f, 0.0f))
        );
        const float3 up = safeNormal(cross(forward, span));
        const float3 incomingAir =
            dispatch.windVelocityAndDensity.xyz - rootVelocity;
        const float3 localAir = float3(
            dot(incomingAir, forward), dot(incomingAir, span),
            dot(incomingAir, up)
        );
        const float dragScale = 0.5f * dispatch.windVelocityAndDensity.w *
            fuselage.referenceAreasAndDrag.w;
        const float3 localForce = dragScale *
            fuselage.referenceAreasAndDrag.xyz * localAir * abs(localAir);
        const float3 localAngularVelocity = float3(
            dot(rootAngularVelocity, forward),
            dot(rootAngularVelocity, span), dot(rootAngularVelocity, up)
        );
        const float3 localTorque = -fuselage.angularDamping.xyz *
            localAngularVelocity;
        const float3 force = forward * localForce.x + span * localForce.y +
            up * localForce.z;
        const float3 torque = forward * localTorque.x + span * localTorque.y +
            up * localTorque.z;
        if (all(isfinite(force)) && all(isfinite(torque))) {
            const uint wrenchIndex =
                environment * dispatch.bodyStride + fuselage.bodyIndex;
            MRABABodyWrenchGPU wrench = wrenches[wrenchIndex];
            wrench.force.xyz += force;
            wrench.torque.xyz += torque;
            wrenches[wrenchIndex] = wrench;
        }
    }
    float wingStrokeSpeedSquared = 0.0f;
    float2 sideStrokeSpeedSquared = float2(0.0f);
    constexpr uint bladeElementCount = 8u;
    constexpr float inverseEllipticAreaIntegral = 4.0f / M_PI_F;
    for (uint side = 0u; side < 2u; ++side) {
        const MRFlappingWingGPU wing = wings[side];
        if (wing.bodyIndex >= dispatch.bodyStride ||
            wing.flapQIndex < dispatch.qOffset ||
            wing.flapVIndex < dispatch.vOffset ||
            ((wing.sweepQIndex == MR_INVALID_INDEX) !=
             (wing.sweepVIndex == MR_INVALID_INDEX)) ||
            ((wing.pronationQIndex == MR_INVALID_INDEX) !=
             (wing.pronationVIndex == MR_INVALID_INDEX)) ||
            !finite4(wing.rootToCenterAndArea) ||
            !finite4(wing.hingeAxisAndChord) || !finite4(wing.coefficients) ||
            !finite4(wing.unsteadyCoefficients) ||
            !(wing.rootToCenterAndArea.w > 0.0f) ||
            !(wing.hingeAxisAndChord.w > 0.0f)) {
            continue;
        }
        const float3 hingeAxis = safeNormal(wing.hingeAxisAndChord.xyz);
        if (dot(hingeAxis, hingeAxis) == 0.0f) {
            continue;
        }
        const bool hasPronation = wing.pronationQIndex != MR_INVALID_INDEX;
        const bool hasSweep = wing.sweepQIndex != MR_INVALID_INDEX;
        const float3 sweepAxis = hasSweep
            ? safeNormal(wing.sweepAxisAndReserved.xyz)
            : float3(0.0f);
        const float3 pronationAxis = hasPronation
            ? safeNormal(wing.pronationAxisAndReserved.xyz)
            : float3(0.0f);
        if (hasSweep &&
            (wing.sweepQIndex < dispatch.qOffset ||
             wing.sweepVIndex < dispatch.vOffset ||
             !finite4(wing.sweepAxisAndReserved) ||
             dot(sweepAxis, sweepAxis) == 0.0f)) {
            continue;
        }
        if (hasPronation &&
            (wing.pronationQIndex < dispatch.qOffset ||
             wing.pronationVIndex < dispatch.vOffset ||
             !finite4(wing.pronationAxisAndReserved) ||
             dot(pronationAxis, pronationAxis) == 0.0f)) {
            continue;
        }
        const float flapAngle = qState[
            environment * dispatch.qStride + wing.flapQIndex
        ];
        const float flapAngularRate = vState[
            environment * dispatch.vStride + wing.flapVIndex
        ];
        const float sweepAngle = hasSweep
            ? qState[environment * dispatch.qStride + wing.sweepQIndex]
            : 0.0f;
        const float sweepAngularRate = hasSweep
            ? vState[environment * dispatch.vStride + wing.sweepVIndex]
            : 0.0f;
        const float pronationAngle = hasPronation
            ? qState[environment * dispatch.qStride + wing.pronationQIndex]
            : 0.0f;
        const float pronationAngularRate = hasPronation
            ? vState[environment * dispatch.vStride + wing.pronationVIndex]
            : 0.0f;
        if (!isfinite(flapAngle) || !isfinite(flapAngularRate) ||
            !isfinite(sweepAngle) || !isfinite(sweepAngularRate) ||
            !isfinite(pronationAngle) || !isfinite(pronationAngularRate)) {
            continue;
        }
        const float4 sweepRotation = axisAngle(sweepAxis, sweepAngle);
        const float4 hingeRotation = axisAngle(hingeAxis, flapAngle);
        const float4 pronationRotation = axisAngle(
            pronationAxis, pronationAngle
        );
        const float4 wingOrientation = multiplyQuaternion(
            multiplyQuaternion(
                multiplyQuaternion(normalizeQuaternion(rootOrientation),
                                   sweepRotation),
                hingeRotation
            ),
            pronationRotation
        );
        const float3 sweepAxisWorld = hasSweep
            ? rotate(rootOrientation, sweepAxis)
            : float3(0.0f);
        const float3 hingeAxisWorld = rotate(
            multiplyQuaternion(normalizeQuaternion(rootOrientation),
                               sweepRotation),
            hingeAxis
        );
        const float3 pronationAxisWorld = hasPronation
            ? rotate(
                multiplyQuaternion(
                    multiplyQuaternion(normalizeQuaternion(rootOrientation),
                                       sweepRotation),
                    hingeRotation
                ),
                pronationAxis
            )
            : float3(0.0f);
        const float3 chord = rotate(wingOrientation, float3(1.0f, 0.0f, 0.0f));
        // The authored root-to-center vector carries the bilateral span sign;
        // a mirrored wing cannot share the left wing's positive-y span basis.
        const float spanSign = wing.rootToCenterAndArea.y >= 0.0f ? 1.0f : -1.0f;
        const float3 span = rotate(
            wingOrientation, float3(0.0f, spanSign, 0.0f)
        );
        const float3 normal = safeNormal(cross(chord, span));
        const float3 airframeUp = rotate(
            rootOrientation, float3(0.0f, 0.0f, 1.0f)
        );
        const float3 airframeForward = rotate(
            rootOrientation, float3(1.0f, 0.0f, 0.0f)
        );
        const float3 unsteadyDirection = safeNormal(
            airframeUp + wing.unsteadyCoefficients.y * airframeForward
        );
        float3 force = float3(0.0f);
        float3 torque = float3(0.0f);
        // Resolve the span instead of applying one point load at the wing
        // COM.  The authored root-to-center vector is the half-span station;
        // an elliptic midpoint quadrature distributes the same total area
        // from shoulder to tip.  Every station sees its own rotational
        // velocity, angle of attack, lift, drag, and moment arm.
        for (uint element = 0u; element < bladeElementCount; ++element) {
            const float eta = (float(element) + 0.5f) /
                float(bladeElementCount);
            const float ellipticWeight = sqrt(max(1.0f - eta * eta, 0.0f)) *
                inverseEllipticAreaIntegral;
            const float elementArea = wing.rootToCenterAndArea.w *
                ellipticWeight / float(bladeElementCount);
            const float3 neutralPoint = float3(
                wing.rootToCenterAndArea.x,
                2.0f * eta * wing.rootToCenterAndArea.y,
                wing.rootToCenterAndArea.z
            );
            const float3 rootToPoint = rotate(
                rootOrientation,
                rotate(
                    sweepRotation,
                    rotate(hingeRotation,
                           rotate(pronationRotation, neutralPoint))
                )
            );
            const float3 sweepStrokeVelocity = hasSweep
                ? cross(sweepAxisWorld * sweepAngularRate, rootToPoint)
                : float3(0.0f);
            const float3 flapStrokeVelocity = cross(
                hingeAxisWorld * flapAngularRate, rootToPoint
            );
            const float3 strokeVelocity = flapStrokeVelocity +
                sweepStrokeVelocity +
                (hasPronation
                    ? cross(pronationAxisWorld * pronationAngularRate,
                            rootToPoint)
                    : float3(0.0f));
            const float3 pointVelocity = rootVelocity +
                cross(rootAngularVelocity, rootToPoint) + strokeVelocity;
            const float3 incomingAir =
                dispatch.windVelocityAndDensity.xyz - pointVelocity;
            const float speedSquared = dot(incomingAir, incomingAir);
            if (!(speedSquared > 1.0e-8f) || !isfinite(speedSquared)) {
                continue;
            }
            const float3 flow = incomingAir * rsqrt(speedSquared);
            // Relative wind points toward the trailing direction while the
            // section chord points in vehicle travel direction. Incidence is
            // therefore measured against the opposite vector.
            const float3 travelDirection = -flow;
            const float angleOfAttack = atan2(
                dot(travelDirection, normal),
                dot(travelDirection, chord)
            );
            const float liftCoefficient = clamp(
                wing.coefficients.x * angleOfAttack,
                -wing.coefficients.w,
                wing.coefficients.w
            );
            const float dragCoefficient = wing.coefficients.y +
                wing.coefficients.z * liftCoefficient * liftCoefficient;
            const float dynamicPressure = 0.5f *
                dispatch.windVelocityAndDensity.w * speedSquared;
            const float3 liftDirection = safeNormal(cross(span, flow));
            const float strokeSpeedSquared = dot(
                flapStrokeVelocity, flapStrokeVelocity
            );
            wingStrokeSpeedSquared +=
                ellipticWeight * strokeSpeedSquared /
                float(bladeElementCount);
            sideStrokeSpeedSquared[side] +=
                ellipticWeight * strokeSpeedSquared /
                float(bladeElementCount);
            const float strokeFraction = clamp(
                strokeSpeedSquared / speedSquared, 0.0f, 1.0f
            );
            const float3 elementForce = dynamicPressure * elementArea * (
                liftCoefficient * liftDirection +
                dragCoefficient * flow +
                wing.unsteadyCoefficients.x * strokeFraction *
                    unsteadyDirection
            );
            force += elementForce;
            torque += cross(rootToPoint, elementForce);
        }
        if (!all(isfinite(force)) || !all(isfinite(torque))) {
            continue;
        }
        // The load integral is formed about the airframe root because that is
        // the common aerodynamic reference. ABA body wrenches are instead
        // defined about each body's COM. Resolve the live distal-wing COM
        // from the compiled root joint and re-express the same resultant
        // wrench before it enters the articulated chain. This preserves the
        // net force and root moment for a locked wing while exposing the
        // physical reaction to finite sweep, flap, and pronation drives.
        const float4 rootJointRotation = hasSweep
            ? sweepRotation : hingeRotation;
        const float3 rootToWingOrigin = rotate(
            rootOrientation,
            wing.rootJointParentAnchor.xyz - rotate(
                rootJointRotation, wing.rootJointChildAnchor.xyz
            )
        );
        const float3 rootToWingCOM = rootToWingOrigin + rotate(
            wingOrientation, wing.bodyCenterOfMass.xyz
        );
        const float3 wingTorque = torque - cross(rootToWingCOM, force);
        if (!all(isfinite(rootToWingCOM)) || !all(isfinite(wingTorque))) {
            continue;
        }
        const uint wrenchIndex =
            environment * dispatch.bodyStride + wing.bodyIndex;
        MRABABodyWrenchGPU wrench = wrenches[wrenchIndex];
        wrench.force.xyz += force;
        wrench.torque.xyz += wingTorque;
        wrenches[wrenchIndex] = wrench;
    }
    // Differential live stroke energy produces a bounded yaw moment. It
    // vanishes exactly for a bilateral stroke and stays on the accepted-state
    // aerodynamic timeline rather than replaying an external force trace.
    const float yawMomentCoefficient = 0.5f * (
        wings[0].unsteadyCoefficients.z +
        wings[1].unsteadyCoefficients.z
    );
    if (fuselage.bodyIndex == dispatch.rootBodyIndex &&
        yawMomentCoefficient != 0.0f &&
        all(isfinite(sideStrokeSpeedSquared))) {
        const float meanWingArea = 0.5f * (
            wings[0].rootToCenterAndArea.w +
            wings[1].rootToCenterAndArea.w
        );
        const float meanHalfSpan = 0.5f * (
            abs(wings[0].rootToCenterAndArea.y) +
            abs(wings[1].rootToCenterAndArea.y)
        );
        const float yawMoment = clamp(
            0.5f * dispatch.windVelocityAndDensity.w * meanWingArea *
                meanHalfSpan * yawMomentCoefficient *
                (sideStrokeSpeedSquared.y - sideStrokeSpeedSquared.x),
            -0.02f,
            0.02f
        );
        const float3 rootUp = safeNormal(rotate(
            rootOrientation, float3(0.0f, 0.0f, 1.0f)
        ));
        if (all(isfinite(rootUp)) && isfinite(yawMoment)) {
            const uint rootWrenchIndex =
                environment * dispatch.bodyStride + dispatch.rootBodyIndex;
            MRABABodyWrenchGPU rootWrench = wrenches[rootWrenchIndex];
            rootWrench.torque.xyz += rootUp * yawMoment;
            wrenches[rootWrenchIndex] = rootWrench;
        }
    }
    // A direct tail-pitch coordinate rotates the local aerodynamic frame;
    // a fixed tail uses zero deflection. This is not a coupled D3Q19 wake.
    if (tail.bodyIndex >= dispatch.bodyStride ||
        tail.rootBodyIndex != dispatch.rootBodyIndex ||
        !finite4(tail.rootToCenterAndArea) ||
        !finite4(tail.chordAndCoefficients) ||
        !(tail.rootToCenterAndArea.w > 0.0f) ||
        !(tail.chordAndCoefficients.x > 0.0f)) {
        return;
    }
    const bool articulatedTail = tail.qIndex != MR_INVALID_INDEX &&
        tail.vIndex != MR_INVALID_INDEX &&
        tail.qIndex < dispatch.qStride && tail.vIndex < dispatch.vStride;
    const float tailPitch = articulatedTail
        ? qState[qBase + tail.qIndex] : 0.0f;
    const float tailPitchRate = articulatedTail
        ? vState[vBase + tail.vIndex] : 0.0f;
    const float4 tailOrientation = multiplyQuaternion(
        rootOrientation, axisAngle(float3(0.0f, 1.0f, 0.0f), tailPitch)
    );
    const float3 rootToTail = rotate(
        rootOrientation, tail.rootToCenterAndArea.xyz
    );
    const float3 tailVelocity = rootVelocity +
        cross(rootAngularVelocity, rootToTail) +
        cross(rotate(rootOrientation, float3(0.0f, tailPitchRate, 0.0f)),
              rootToTail);
    const float3 tailAir = dispatch.windVelocityAndDensity.xyz - tailVelocity;
    const float tailAirSpeedSquared = dot(tailAir, tailAir);
    float3 tailForce = float3(0.0f);
    if (tailAirSpeedSquared > 1.0e-8f && isfinite(tailAirSpeedSquared)) {
        const float3 tailFlow = tailAir * rsqrt(tailAirSpeedSquared);
        const float3 tailTravelDirection = -tailFlow;
        const float3 rootForward = safeNormal(
            rotate(tailOrientation, float3(1.0f, 0.0f, 0.0f))
        );
        const float3 rootSpan = safeNormal(
            rotate(tailOrientation, float3(0.0f, 1.0f, 0.0f))
        );
        const float3 rootUp = safeNormal(cross(rootForward, rootSpan));
        const float angleOfAttack = atan2(
            dot(tailTravelDirection, rootUp),
            dot(tailTravelDirection, rootForward)
        );
        const float liftCoefficient = clamp(
            tail.chordAndCoefficients.y * angleOfAttack, -1.5f, 1.5f
        );
        const float dragCoefficient = tail.chordAndCoefficients.z +
            0.16f * liftCoefficient * liftCoefficient;
        const float dynamicPressure = 0.5f *
            dispatch.windVelocityAndDensity.w * tailAirSpeedSquared;
        tailForce += dynamicPressure * tail.rootToCenterAndArea.w * (
            liftCoefficient * safeNormal(cross(rootSpan, tailFlow)) +
            dragCoefficient * tailFlow
        );
    }
    const float washDynamicPressure = 0.5f *
        dispatch.windVelocityAndDensity.w *
        (tailAirSpeedSquared + 0.35f * wingStrokeSpeedSquared);
    const float3 rootSpan = safeNormal(
        rotate(rootOrientation, float3(0.0f, 1.0f, 0.0f))
    );
    const float3 rootUp = safeNormal(cross(
        safeNormal(rotate(rootOrientation, float3(1.0f, 0.0f, 0.0f))), rootSpan
    ));
    const float pitchRate = clamp(dot(rootAngularVelocity, rootSpan), -20.0f, 20.0f);
    tailForce += rootUp * (-washDynamicPressure * tail.rootToCenterAndArea.w *
        tail.chordAndCoefficients.w * pitchRate);
    if (all(isfinite(tailForce))) {
        const uint wrenchIndex = environment * dispatch.bodyStride + tail.bodyIndex;
        MRABABodyWrenchGPU wrench = wrenches[wrenchIndex];
        wrench.force.xyz += tailForce;
        wrenches[wrenchIndex] = wrench;
    }
}
