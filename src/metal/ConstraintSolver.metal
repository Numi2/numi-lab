#include <metal_stdlib>
#include "metalrobo/engine_types.h"

using namespace metal;

namespace {

constant float kMinimumScalar = 1.0e-12f;
constant float kMaximumFiniteImpulse = 1.0e30f;

inline bool finiteFloat4(const float4 value) {
    return all(isfinite(value));
}

inline bool finiteFloat3(const float3 value) {
    return all(isfinite(value));
}

inline bool bodyIsDynamic(device const MRBodyStateGPU& body) {
    return body.flagsAndIndices[0] == MR_MOTION_DYNAMIC &&
        body.linearVelocityAndInverseMass.w > 0.0f;
}

inline float3 inverseInertiaMultiply(
    device const MRBodyStateGPU& body,
    const float3 value
) {
    return float3(
        dot(body.inverseInertiaWorldRow0.xyz, value),
        dot(body.inverseInertiaWorldRow1.xyz, value),
        dot(body.inverseInertiaWorldRow2.xyz, value)
    );
}

inline float3 pointVelocity(
    device const MRBodyStateGPU& body,
    const float3 offset
) {
    if (body.flagsAndIndices[0] == MR_MOTION_STATIC) {
        return float3(0.0f);
    }
    return body.linearVelocityAndInverseMass.xyz +
        cross(body.angularVelocity.xyz, offset);
}

inline float3 relativePointVelocity(
    device const MRBodyStateGPU& bodyA,
    device const MRBodyStateGPU& bodyB,
    const float3 offsetA,
    const float3 offsetB
) {
    return pointVelocity(bodyB, offsetB) -
        pointVelocity(bodyA, offsetA);
}

inline float directionalCoupling(
    device const MRBodyStateGPU& bodyA,
    device const MRBodyStateGPU& bodyB,
    const float3 offsetA,
    const float3 offsetB,
    const float3 direction0,
    const float3 direction1
) {
    float result = 0.0f;
    if (bodyIsDynamic(bodyA)) {
        result += bodyA.linearVelocityAndInverseMass.w *
            dot(direction0, direction1);
        const float3 angular0 = cross(offsetA, direction0);
        const float3 angular1 = cross(offsetA, direction1);
        result += dot(
            angular0,
            inverseInertiaMultiply(bodyA, angular1)
        );
    }
    if (bodyIsDynamic(bodyB)) {
        result += bodyB.linearVelocityAndInverseMass.w *
            dot(direction0, direction1);
        const float3 angular0 = cross(offsetB, direction0);
        const float3 angular1 = cross(offsetB, direction1);
        result += dot(
            angular0,
            inverseInertiaMultiply(bodyB, angular1)
        );
    }
    return result;
}

inline float angularCoupling(
    device const MRBodyStateGPU& bodyA,
    device const MRBodyStateGPU& bodyB,
    const float3 direction
) {
    float result = 0.0f;
    if (bodyIsDynamic(bodyA)) {
        result += dot(
            direction,
            inverseInertiaMultiply(bodyA, direction)
        );
    }
    if (bodyIsDynamic(bodyB)) {
        result += dot(
            direction,
            inverseInertiaMultiply(bodyB, direction)
        );
    }
    return result;
}

inline void contactBasis(
    const float3 normal,
    thread float3& tangentU,
    thread float3& tangentV
) {
    const float3 absoluteNormal = abs(normal);
    float3 reference;
    if (absoluteNormal.x <= absoluteNormal.y &&
        absoluteNormal.x <= absoluteNormal.z) {
        reference = float3(1.0f, 0.0f, 0.0f);
    } else if (absoluteNormal.y <= absoluteNormal.z) {
        reference = float3(0.0f, 1.0f, 0.0f);
    } else {
        reference = float3(0.0f, 0.0f, 1.0f);
    }
    tangentU = normalize(cross(reference, normal));
    tangentV = cross(normal, tangentU);
}

inline bool validBodyState(device const MRBodyStateGPU& body) {
    return
        finiteFloat4(body.position) &&
        finiteFloat4(body.orientation) &&
        finiteFloat4(body.linearVelocityAndInverseMass) &&
        finiteFloat4(body.angularVelocity) &&
        finiteFloat4(body.inverseInertiaWorldRow0) &&
        finiteFloat4(body.inverseInertiaWorldRow1) &&
        finiteFloat4(body.inverseInertiaWorldRow2) &&
        body.linearVelocityAndInverseMass.w >= 0.0f &&
        body.flagsAndIndices[0] <= MR_MOTION_DYNAMIC;
}

inline bool validContactInput(
    device const MRContactConstraintGPU& contact
) {
    if (!finiteFloat4(contact.pointAndSeparation) ||
        !finiteFloat4(contact.normal) ||
        !finiteFloat4(contact.friction) ||
        !finiteFloat4(contact.response) ||
        !finiteFloat4(contact.targetVelocityAndPreSolveNormal) ||
        !finiteFloat4(contact.impulses)) {
        return false;
    }

    const float normalLengthSquared =
        dot(contact.normal.xyz, contact.normal.xyz);
    return
        abs(normalLengthSquared - 1.0f) <= 2.0e-4f &&
        all(contact.friction >= 0.0f) &&
        contact.friction.x >= contact.friction.y &&
        all(contact.response >= 0.0f) &&
        contact.response.x <= 1.0f;
}

inline bool applyPairImpulse(
    device MRBodyStateGPU& bodyA,
    device MRBodyStateGPU& bodyB,
    const float3 offsetA,
    const float3 offsetB,
    const float3 impulseOnB
) {
    float3 linearA = bodyA.linearVelocityAndInverseMass.xyz;
    float3 angularA = bodyA.angularVelocity.xyz;
    float3 linearB = bodyB.linearVelocityAndInverseMass.xyz;
    float3 angularB = bodyB.angularVelocity.xyz;

    if (bodyIsDynamic(bodyA)) {
        linearA -=
            bodyA.linearVelocityAndInverseMass.w * impulseOnB;
        angularA -= inverseInertiaMultiply(
            bodyA,
            cross(offsetA, impulseOnB)
        );
    }
    if (bodyIsDynamic(bodyB)) {
        linearB +=
            bodyB.linearVelocityAndInverseMass.w * impulseOnB;
        angularB += inverseInertiaMultiply(
            bodyB,
            cross(offsetB, impulseOnB)
        );
    }

    if (!finiteFloat3(linearA) ||
        !finiteFloat3(angularA) ||
        !finiteFloat3(linearB) ||
        !finiteFloat3(angularB)) {
        return false;
    }

    if (bodyIsDynamic(bodyA)) {
        bodyA.linearVelocityAndInverseMass.xyz = linearA;
        bodyA.angularVelocity.xyz = angularA;
    }
    if (bodyIsDynamic(bodyB)) {
        bodyB.linearVelocityAndInverseMass.xyz = linearB;
        bodyB.angularVelocity.xyz = angularB;
    }
    return true;
}

inline bool applyPairAngularImpulse(
    device MRBodyStateGPU& bodyA,
    device MRBodyStateGPU& bodyB,
    const float3 angularImpulseOnB
) {
    float3 angularA = bodyA.angularVelocity.xyz;
    float3 angularB = bodyB.angularVelocity.xyz;

    if (bodyIsDynamic(bodyA)) {
        angularA -=
            inverseInertiaMultiply(bodyA, angularImpulseOnB);
    }
    if (bodyIsDynamic(bodyB)) {
        angularB +=
            inverseInertiaMultiply(bodyB, angularImpulseOnB);
    }

    if (!finiteFloat3(angularA) || !finiteFloat3(angularB)) {
        return false;
    }

    if (bodyIsDynamic(bodyA)) {
        bodyA.angularVelocity.xyz = angularA;
    }
    if (bodyIsDynamic(bodyB)) {
        bodyB.angularVelocity.xyz = angularB;
    }
    return true;
}

inline float normalTargetVelocity(
    device const MRContactConstraintGPU& contact,
    const float3 normal,
    device const MRSolverBatchGPU& batch
) {
    const float timestep = batch.timestepAndBias.x;
    const float errorReduction = batch.timestepAndBias.y;
    const float penetrationSlop = batch.timestepAndBias.z;
    const float maximumDepenetrationVelocity =
        batch.timestepAndBias.w;

    const float penetration =
        min(contact.pointAndSeparation.w + penetrationSlop, 0.0f);
    const float positionalTarget = min(
        maximumDepenetrationVelocity,
        -errorReduction * penetration / timestep
    );

    float restitutionTarget = 0.0f;
    if ((contact.flags & MR_CONSTRAINT_FLAG_NEW_IMPACT) != 0u &&
        contact.targetVelocityAndPreSolveNormal.w <
            -contact.response.y) {
        restitutionTarget =
            -contact.response.x *
            contact.targetVelocityAndPreSolveNormal.w;
    }

    return
        dot(contact.targetVelocityAndPreSolveNormal.xyz, normal) +
        max(positionalTarget, restitutionTarget);
}

inline float normalSoftness(
    device const MRContactConstraintGPU& contact,
    device const MRSolverBatchGPU& batch
) {
    const float timestep = batch.timestepAndBias.x;
    return contact.response.z / (timestep * timestep);
}

inline float2 projectFrictionImpulse(
    const float2 candidate,
    const float normalImpulse,
    const float staticFriction,
    const float dynamicFriction
) {
    const float candidateLengthSquared = dot(candidate, candidate);
    if (!(candidateLengthSquared > 0.0f) ||
        !(normalImpulse > 0.0f)) {
        return float2(0.0f);
    }

    const float candidateLength = sqrt(candidateLengthSquared);
    const float staticLimit =
        max(staticFriction, dynamicFriction) * normalImpulse;
    if (candidateLength <= staticLimit) {
        return candidate;
    }

    const float dynamicLimit = dynamicFriction * normalImpulse;
    return candidate * (dynamicLimit / candidateLength);
}

inline bool validateEffectiveMasses(
    device const MRSolverBatchGPU& batch,
    device const MRBodyStateGPU* bodies,
    device const MRContactConstraintGPU* contacts,
    thread float& minimumEigenvalue,
    thread float& maximumEigenvalue
) {
    const float minimumLinearDenominator =
        max(batch.convergence.z, kMinimumScalar);
    const float minimumAngularDenominator =
        max(batch.convergence.w, kMinimumScalar);
    minimumEigenvalue = kMaximumFiniteImpulse;
    maximumEigenvalue = 0.0f;

    for (uint localIndex = 0u;
         localIndex < batch.contactCount;
         ++localIndex) {
        device const MRContactConstraintGPU& contact =
            contacts[batch.contactOffset + localIndex];
        if ((contact.flags & MR_CONSTRAINT_FLAG_DISABLED) != 0u) {
            continue;
        }

        device const MRBodyStateGPU& bodyA = bodies[contact.bodyA];
        device const MRBodyStateGPU& bodyB = bodies[contact.bodyB];
        const float3 point = contact.pointAndSeparation.xyz;
        const float3 offsetA = point - bodyA.position.xyz;
        const float3 offsetB = point - bodyB.position.xyz;
        const float3 normal = normalize(contact.normal.xyz);

        const float normalDenominator =
            directionalCoupling(
                bodyA,
                bodyB,
                offsetA,
                offsetB,
                normal,
                normal
            ) +
            normalSoftness(contact, batch);
        if (!isfinite(normalDenominator) ||
            normalDenominator < minimumLinearDenominator) {
            return false;
        }
        minimumEigenvalue =
            min(minimumEigenvalue, normalDenominator);
        maximumEigenvalue =
            max(maximumEigenvalue, normalDenominator);

        if (max(contact.friction.x, contact.friction.y) > 0.0f) {
            float3 tangentU;
            float3 tangentV;
            contactBasis(normal, tangentU, tangentV);

            const float kuu = directionalCoupling(
                bodyA,
                bodyB,
                offsetA,
                offsetB,
                tangentU,
                tangentU
            );
            const float kvv = directionalCoupling(
                bodyA,
                bodyB,
                offsetA,
                offsetB,
                tangentV,
                tangentV
            );
            const float kuv = 0.5f * (
                directionalCoupling(
                    bodyA,
                    bodyB,
                    offsetA,
                    offsetB,
                    tangentU,
                    tangentV
                ) +
                directionalCoupling(
                    bodyA,
                    bodyB,
                    offsetA,
                    offsetB,
                    tangentV,
                    tangentU
                )
            );
            const float trace = kuu + kvv;
            const float discriminant = sqrt(max(
                (kuu - kvv) * (kuu - kvv) + 4.0f * kuv * kuv,
                0.0f
            ));
            const float tangentMinimum =
                0.5f * (trace - discriminant);
            const float tangentMaximum =
                0.5f * (trace + discriminant);
            const float determinant = kuu * kvv - kuv * kuv;
            if (!isfinite(kuu) ||
                !isfinite(kvv) ||
                !isfinite(kuv) ||
                !isfinite(determinant) ||
                determinant <=
                    minimumLinearDenominator *
                        minimumLinearDenominator ||
                tangentMinimum < minimumLinearDenominator) {
                return false;
            }
            minimumEigenvalue =
                min(minimumEigenvalue, tangentMinimum);
            maximumEigenvalue =
                max(maximumEigenvalue, tangentMaximum);
        }

        if (contact.friction.w > 0.0f) {
            const float torsionalDenominator =
                angularCoupling(bodyA, bodyB, normal);
            if (!isfinite(torsionalDenominator) ||
                torsionalDenominator < minimumAngularDenominator) {
                return false;
            }
        }
    }

    return true;
}

inline bool warmStartContacts(
    device const MRSolverBatchGPU& batch,
    device MRBodyStateGPU* bodies,
    device MRContactConstraintGPU* contacts
) {
    const float warmStartScale =
        max(batch.convergence.y, 0.0f);

    for (uint localIndex = 0u;
         localIndex < batch.contactCount;
         ++localIndex) {
        device MRContactConstraintGPU& contact =
            contacts[batch.contactOffset + localIndex];
        if ((contact.flags & MR_CONSTRAINT_FLAG_DISABLED) != 0u) {
            contact.impulses = float4(0.0f);
            contact.flags &= ~MR_CONSTRAINT_FLAG_WARM_STARTED;
            continue;
        }

        if (batch.enableWarmStart == 0u) {
            contact.impulses = float4(0.0f);
            contact.flags &= ~MR_CONSTRAINT_FLAG_WARM_STARTED;
            continue;
        }

        device MRBodyStateGPU& bodyA = bodies[contact.bodyA];
        device MRBodyStateGPU& bodyB = bodies[contact.bodyB];
        const float3 point = contact.pointAndSeparation.xyz;
        const float3 offsetA = point - bodyA.position.xyz;
        const float3 offsetB = point - bodyB.position.xyz;
        const float3 normal = normalize(contact.normal.xyz);
        float3 tangentU;
        float3 tangentV;
        contactBasis(normal, tangentU, tangentV);

        float normalImpulse =
            max(contact.impulses.x * warmStartScale, 0.0f);
        if (contact.response.w > 0.0f) {
            normalImpulse =
                min(normalImpulse, contact.response.w);
        }
        normalImpulse =
            min(normalImpulse, kMaximumFiniteImpulse);

        const float2 tangentImpulse = projectFrictionImpulse(
            contact.impulses.yz * warmStartScale,
            normalImpulse,
            contact.friction.x,
            contact.friction.y
        );
        const float torsionalLimit =
            contact.friction.w * normalImpulse;
        const float torsionalImpulse = clamp(
            contact.impulses.w * warmStartScale,
            -torsionalLimit,
            torsionalLimit
        );

        contact.impulses = float4(
            normalImpulse,
            tangentImpulse,
            torsionalImpulse
        );

        const float3 linearImpulse =
            normal * normalImpulse +
            tangentU * tangentImpulse.x +
            tangentV * tangentImpulse.y;
        if (!applyPairImpulse(
                bodyA,
                bodyB,
                offsetA,
                offsetB,
                linearImpulse
            ) ||
            !applyPairAngularImpulse(
                bodyA,
                bodyB,
                normal * torsionalImpulse
            )) {
            return false;
        }

        if (normalImpulse > 0.0f ||
            dot(tangentImpulse, tangentImpulse) > 0.0f ||
            torsionalImpulse != 0.0f) {
            contact.flags |= MR_CONSTRAINT_FLAG_WARM_STARTED;
        } else {
            contact.flags &= ~MR_CONSTRAINT_FLAG_WARM_STARTED;
        }
    }
    return true;
}

inline bool solveOneContact(
    device const MRSolverBatchGPU& batch,
    device MRBodyStateGPU* bodies,
    device MRContactConstraintGPU& contact,
    thread float& maximumImpulseDelta
) {
    device MRBodyStateGPU& bodyA = bodies[contact.bodyA];
    device MRBodyStateGPU& bodyB = bodies[contact.bodyB];

    const float3 point = contact.pointAndSeparation.xyz;
    const float3 offsetA = point - bodyA.position.xyz;
    const float3 offsetB = point - bodyB.position.xyz;
    const float3 normal = normalize(contact.normal.xyz);
    float3 tangentU;
    float3 tangentV;
    contactBasis(normal, tangentU, tangentV);

    const float softness = normalSoftness(contact, batch);
    const float normalDenominator =
        directionalCoupling(
            bodyA,
            bodyB,
            offsetA,
            offsetB,
            normal,
            normal
        ) +
        softness;
    const float normalVelocity = dot(
        relativePointVelocity(bodyA, bodyB, offsetA, offsetB),
        normal
    );
    const float targetNormal =
        normalTargetVelocity(contact, normal, batch);
    const float oldNormalImpulse = contact.impulses.x;
    const float normalDeltaUnclamped =
        (targetNormal - normalVelocity -
         softness * oldNormalImpulse) /
        normalDenominator;

    float newNormalImpulse =
        max(oldNormalImpulse + normalDeltaUnclamped, 0.0f);
    if (contact.response.w > 0.0f) {
        newNormalImpulse =
            min(newNormalImpulse, contact.response.w);
    }
    newNormalImpulse =
        min(newNormalImpulse, kMaximumFiniteImpulse);
    const float normalDelta =
        newNormalImpulse - oldNormalImpulse;

    if (!isfinite(normalDelta) ||
        !applyPairImpulse(
            bodyA,
            bodyB,
            offsetA,
            offsetB,
            normal * normalDelta
        )) {
        return false;
    }
    contact.impulses.x = newNormalImpulse;
    maximumImpulseDelta =
        max(maximumImpulseDelta, abs(normalDelta));

    const float staticFriction =
        max(contact.friction.x, contact.friction.y);
    const float dynamicFriction = contact.friction.y;
    if (staticFriction > 0.0f) {
        const float kuu = directionalCoupling(
            bodyA,
            bodyB,
            offsetA,
            offsetB,
            tangentU,
            tangentU
        );
        const float kvv = directionalCoupling(
            bodyA,
            bodyB,
            offsetA,
            offsetB,
            tangentV,
            tangentV
        );
        const float kuv = 0.5f * (
            directionalCoupling(
                bodyA,
                bodyB,
                offsetA,
                offsetB,
                tangentU,
                tangentV
            ) +
            directionalCoupling(
                bodyA,
                bodyB,
                offsetA,
                offsetB,
                tangentV,
                tangentU
            )
        );
        const float determinant = kuu * kvv - kuv * kuv;
        const float inverseDeterminant = 1.0f / determinant;

        const float3 tangentVelocity =
            relativePointVelocity(
                bodyA,
                bodyB,
                offsetA,
                offsetB
            ) -
            contact.targetVelocityAndPreSolveNormal.xyz;
        const float2 tangentRightHandSide = -float2(
            dot(tangentVelocity, tangentU),
            dot(tangentVelocity, tangentV)
        );
        const float2 tangentDeltaUnclamped = float2(
            (
                kvv * tangentRightHandSide.x -
                kuv * tangentRightHandSide.y
            ) * inverseDeterminant,
            (
                kuu * tangentRightHandSide.y -
                kuv * tangentRightHandSide.x
            ) * inverseDeterminant
        );
        const float2 oldTangentImpulse = contact.impulses.yz;
        const float2 newTangentImpulse = projectFrictionImpulse(
            oldTangentImpulse + tangentDeltaUnclamped,
            newNormalImpulse,
            staticFriction,
            dynamicFriction
        );
        const float2 tangentDelta =
            newTangentImpulse - oldTangentImpulse;
        if (!all(isfinite(tangentDelta)) ||
            !applyPairImpulse(
                bodyA,
                bodyB,
                offsetA,
                offsetB,
                tangentU * tangentDelta.x +
                    tangentV * tangentDelta.y
            )) {
            return false;
        }
        contact.impulses.yz = newTangentImpulse;
        maximumImpulseDelta = max(
            maximumImpulseDelta,
            length(tangentDelta)
        );
    } else if (any(contact.impulses.yz != 0.0f)) {
        const float2 tangentDelta = -contact.impulses.yz;
        if (!applyPairImpulse(
                bodyA,
                bodyB,
                offsetA,
                offsetB,
                tangentU * tangentDelta.x +
                    tangentV * tangentDelta.y
            )) {
            return false;
        }
        contact.impulses.yz = float2(0.0f);
        maximumImpulseDelta = max(
            maximumImpulseDelta,
            length(tangentDelta)
        );
    }

    if (contact.friction.w > 0.0f) {
        const float torsionalDenominator =
            angularCoupling(bodyA, bodyB, normal);
        const float torsionalVelocity =
            dot(
                bodyB.angularVelocity.xyz -
                    bodyA.angularVelocity.xyz,
                normal
            );
        const float oldTorsionalImpulse = contact.impulses.w;
        const float torsionalLimit =
            contact.friction.w * newNormalImpulse;
        const float newTorsionalImpulse = clamp(
            oldTorsionalImpulse -
                torsionalVelocity / torsionalDenominator,
            -torsionalLimit,
            torsionalLimit
        );
        const float torsionalDelta =
            newTorsionalImpulse - oldTorsionalImpulse;
        if (!isfinite(torsionalDelta) ||
            !applyPairAngularImpulse(
                bodyA,
                bodyB,
                normal * torsionalDelta
            )) {
            return false;
        }
        contact.impulses.w = newTorsionalImpulse;
        maximumImpulseDelta =
            max(maximumImpulseDelta, abs(torsionalDelta));
    } else if (contact.impulses.w != 0.0f) {
        const float torsionalDelta = -contact.impulses.w;
        if (!applyPairAngularImpulse(
                bodyA,
                bodyB,
                normal * torsionalDelta
            )) {
            return false;
        }
        contact.impulses.w = 0.0f;
        maximumImpulseDelta =
            max(maximumImpulseDelta, abs(torsionalDelta));
    }

    return finiteFloat4(contact.impulses);
}

inline bool computeFinalResiduals(
    device const MRSolverBatchGPU& batch,
    device const MRBodyStateGPU* bodies,
    device const MRContactConstraintGPU* contacts,
    thread float& maximumNormalResidual,
    thread float& maximumConeViolation
) {
    maximumNormalResidual = 0.0f;
    maximumConeViolation = 0.0f;
    const float impulseTolerance =
        max(batch.convergence.x, 0.0f);

    for (uint localIndex = 0u;
         localIndex < batch.contactCount;
         ++localIndex) {
        device const MRContactConstraintGPU& contact =
            contacts[batch.contactOffset + localIndex];
        if ((contact.flags & MR_CONSTRAINT_FLAG_DISABLED) != 0u) {
            continue;
        }

        device const MRBodyStateGPU& bodyA = bodies[contact.bodyA];
        device const MRBodyStateGPU& bodyB = bodies[contact.bodyB];
        const float3 point = contact.pointAndSeparation.xyz;
        const float3 offsetA = point - bodyA.position.xyz;
        const float3 offsetB = point - bodyB.position.xyz;
        const float3 normal = normalize(contact.normal.xyz);

        const float normalVelocity = dot(
            relativePointVelocity(
                bodyA,
                bodyB,
                offsetA,
                offsetB
            ),
            normal
        );
        const float residual =
            normalVelocity -
            normalTargetVelocity(contact, normal, batch) +
            normalSoftness(contact, batch) *
                contact.impulses.x;
        float complementarityResidual;
        if (contact.impulses.x <= impulseTolerance) {
            complementarityResidual = max(-residual, 0.0f);
        } else if (
            contact.response.w > 0.0f &&
            contact.impulses.x >=
                contact.response.w - impulseTolerance
        ) {
            complementarityResidual = max(residual, 0.0f);
        } else {
            complementarityResidual = abs(residual);
        }
        maximumNormalResidual =
            max(maximumNormalResidual, complementarityResidual);

        const float tangentMagnitude =
            length(contact.impulses.yz);
        const float frictionLimit =
            max(contact.friction.x, contact.friction.y) *
            contact.impulses.x;
        const float torsionalViolation = max(
            abs(contact.impulses.w) -
                contact.friction.w * contact.impulses.x,
            0.0f
        );
        maximumConeViolation = max(
            maximumConeViolation,
            max(
                tangentMagnitude - frictionLimit,
                torsionalViolation
            )
        );
    }

    return isfinite(maximumNormalResidual) &&
        isfinite(maximumConeViolation);
}

inline void restoreSolverState(
    device MRBodyStateGPU* bodies,
    device MRContactConstraintGPU* contacts,
    const uint contactOffset,
    const uint contactCount,
    const uint bodyBackupCount,
    thread const uint* bodyIndices,
    thread const float4* linearVelocityBackups,
    thread const float4* angularVelocityBackups,
    thread const float4* impulseBackups,
    thread const uint* flagBackups
) {
    for (uint index = 0u; index < bodyBackupCount; ++index) {
        device MRBodyStateGPU& body = bodies[bodyIndices[index]];
        body.linearVelocityAndInverseMass =
            linearVelocityBackups[index];
        body.angularVelocity = angularVelocityBackups[index];
    }
    for (uint localIndex = 0u;
         localIndex < contactCount;
         ++localIndex) {
        device MRContactConstraintGPU& contact =
            contacts[contactOffset + localIndex];
        contact.impulses = impulseBackups[localIndex];
        contact.flags = flagBackups[localIndex];
    }
}

} // namespace

kernel void mr_solve_contact_constraints(
    device const MRSolverBatchGPU* batches [[buffer(0)]],
    device MRBodyStateGPU* bodies [[buffer(1)]],
    device MRContactConstraintGPU* contacts [[buffer(2)]],
    device MRSolverStatusGPU* statuses [[buffer(3)]],
    const uint batchIndex [[thread_position_in_grid]]
) {
    device const MRSolverBatchGPU& batch = batches[batchIndex];
    MRSolverStatusGPU status = {};
    status.code = MR_STEP_SUCCESS;

    if (batch.contactCount > MR_MAX_CONTACTS_PER_SOLVER_BATCH) {
        status.code = MR_STEP_CONTACT_CAPACITY_OVERFLOW;
        status.requiredContacts = batch.contactCount;
        statuses[batchIndex] = status;
        return;
    }

    const uint bodyEnd = batch.bodyOffset + batch.bodyCount;
    const uint contactEnd = batch.contactOffset + batch.contactCount;
    if (bodyEnd < batch.bodyOffset ||
        contactEnd < batch.contactOffset ||
        !(batch.timestepAndBias.x > 0.0f) ||
        !finiteFloat4(batch.timestepAndBias) ||
        !finiteFloat4(batch.convergence) ||
        batch.timestepAndBias.y < 0.0f ||
        batch.timestepAndBias.z < 0.0f ||
        batch.timestepAndBias.w < 0.0f ||
        batch.convergence.x < 0.0f ||
        batch.convergence.y < 0.0f ||
        batch.convergence.z < 0.0f ||
        batch.convergence.w < 0.0f) {
        status.code = MR_STEP_NONFINITE_INPUT;
        statuses[batchIndex] = status;
        return;
    }

    for (uint bodyIndex = batch.bodyOffset;
         bodyIndex < bodyEnd;
         ++bodyIndex) {
        if (!validBodyState(bodies[bodyIndex])) {
            status.code = MR_STEP_NONFINITE_INPUT;
            statuses[batchIndex] = status;
            return;
        }
    }

    ulong previousPairKey = 0ul;
    ulong previousFeatureKey = 0ul;
    bool havePreviousKey = false;
    for (uint contactIndex = batch.contactOffset;
         contactIndex < contactEnd;
         ++contactIndex) {
        device const MRContactConstraintGPU& contact =
            contacts[contactIndex];
        if ((contact.flags & MR_CONSTRAINT_FLAG_DISABLED) != 0u) {
            continue;
        }
        ++status.activeContacts;
        if (contact.bodyA == contact.bodyB ||
            contact.bodyA < batch.bodyOffset ||
            contact.bodyA >= bodyEnd ||
            contact.bodyB < batch.bodyOffset ||
            contact.bodyB >= bodyEnd) {
            status.code = MR_STEP_UNSUPPORTED;
            statuses[batchIndex] = status;
            return;
        }
        if (!validContactInput(contact)) {
            status.code = MR_STEP_NONFINITE_INPUT;
            statuses[batchIndex] = status;
            return;
        }
        if (!bodyIsDynamic(bodies[contact.bodyA]) &&
            !bodyIsDynamic(bodies[contact.bodyB])) {
            status.code = MR_STEP_UNSUPPORTED;
            statuses[batchIndex] = status;
            return;
        }
        // Rolling friction is part of the ABI but not this first throughput
        // kernel. Reject it explicitly instead of silently ignoring it.
        if (contact.friction.z > 0.0f) {
            status.code = MR_STEP_UNSUPPORTED;
            statuses[batchIndex] = status;
            return;
        }
        if (batch.deterministic != 0u) {
            const bool outOfOrder =
                havePreviousKey &&
                (
                    contact.pairKey < previousPairKey ||
                    (
                        contact.pairKey == previousPairKey &&
                        contact.featureKey < previousFeatureKey
                    )
                );
            if (outOfOrder) {
                status.code = MR_STEP_UNSUPPORTED;
                statuses[batchIndex] = status;
                return;
            }
            previousPairKey = contact.pairKey;
            previousFeatureKey = contact.featureKey;
            havePreviousKey = true;
        }
    }
    if (status.activeContacts == 0u) {
        for (uint contactIndex = batch.contactOffset;
             contactIndex < contactEnd;
             ++contactIndex) {
            contacts[contactIndex].impulses = float4(0.0f);
            contacts[contactIndex].flags &=
                ~MR_CONSTRAINT_FLAG_WARM_STARTED;
        }
        statuses[batchIndex] = status;
        return;
    }

    // Count dynamic-body connectivity without treating a shared static floor
    // as an island edge. The bounded contact batch keeps this deterministic
    // O(C^2) reference path small; the production graph pipeline supplies
    // precomputed islands to parallel kernels.
    uint contactParent[MR_MAX_CONTACTS_PER_SOLVER_BATCH];
    for (uint localIndex = 0u;
         localIndex < batch.contactCount;
         ++localIndex) {
        contactParent[localIndex] = localIndex;
    }
    for (uint left = 0u; left < batch.contactCount; ++left) {
        device const MRContactConstraintGPU& contactLeft =
            contacts[batch.contactOffset + left];
        if ((contactLeft.flags &
             MR_CONSTRAINT_FLAG_DISABLED) != 0u) {
            continue;
        }
        for (uint right = 0u; right < left; ++right) {
            device const MRContactConstraintGPU& contactRight =
                contacts[batch.contactOffset + right];
            if ((contactRight.flags &
                 MR_CONSTRAINT_FLAG_DISABLED) != 0u) {
                continue;
            }
            const bool connected =
                (
                    contactLeft.bodyA == contactRight.bodyA &&
                    bodyIsDynamic(bodies[contactLeft.bodyA])
                ) ||
                (
                    contactLeft.bodyA == contactRight.bodyB &&
                    bodyIsDynamic(bodies[contactLeft.bodyA])
                ) ||
                (
                    contactLeft.bodyB == contactRight.bodyA &&
                    bodyIsDynamic(bodies[contactLeft.bodyB])
                ) ||
                (
                    contactLeft.bodyB == contactRight.bodyB &&
                    bodyIsDynamic(bodies[contactLeft.bodyB])
                );
            if (!connected) {
                continue;
            }
            uint rootLeft = left;
            while (contactParent[rootLeft] != rootLeft) {
                rootLeft = contactParent[rootLeft];
            }
            uint rootRight = right;
            while (contactParent[rootRight] != rootRight) {
                rootRight = contactParent[rootRight];
            }
            if (rootLeft != rootRight) {
                const uint lower = min(rootLeft, rootRight);
                const uint upper = max(rootLeft, rootRight);
                contactParent[upper] = lower;
            }
        }
    }
    for (uint localIndex = 0u;
         localIndex < batch.contactCount;
         ++localIndex) {
        device const MRContactConstraintGPU& contact =
            contacts[batch.contactOffset + localIndex];
        if ((contact.flags & MR_CONSTRAINT_FLAG_DISABLED) != 0u) {
            continue;
        }
        uint root = localIndex;
        while (contactParent[root] != root) {
            root = contactParent[root];
        }
        if (root == localIndex) {
            ++status.islandCount;
        }
    }

    float minimumEigenvalue;
    float maximumEigenvalue;
    if (!validateEffectiveMasses(
            batch,
            bodies,
            contacts,
            minimumEigenvalue,
            maximumEigenvalue
        )) {
        status.code = MR_STEP_FACTORIZATION_FAILED;
        statuses[batchIndex] = status;
        return;
    }

    status.residuals.w =
        maximumEigenvalue /
        max(minimumEigenvalue, kMinimumScalar);
    if (!isfinite(status.residuals.w)) {
        status.code = MR_STEP_NONFINITE_INPUT;
        statuses[batchIndex] = status;
        return;
    }

    // The solver mutates velocities and cached impulses in place. Preserve
    // exactly the records it can touch so arithmetic failure has the same
    // transactional publication contract as the CPU path.
    uint bodyBackupCount = 0u;
    uint bodyIndices[MR_MAX_BODIES_PER_SOLVER_BATCH];
    float4 linearVelocityBackups[MR_MAX_BODIES_PER_SOLVER_BATCH];
    float4 angularVelocityBackups[MR_MAX_BODIES_PER_SOLVER_BATCH];
    float4 impulseBackups[MR_MAX_CONTACTS_PER_SOLVER_BATCH];
    uint flagBackups[MR_MAX_CONTACTS_PER_SOLVER_BATCH];
    for (uint localIndex = 0u;
         localIndex < batch.contactCount;
         ++localIndex) {
        device const MRContactConstraintGPU& contact =
            contacts[batch.contactOffset + localIndex];
        impulseBackups[localIndex] = contact.impulses;
        flagBackups[localIndex] = contact.flags;
        if ((contact.flags & MR_CONSTRAINT_FLAG_DISABLED) != 0u) {
            continue;
        }
        for (uint endpointIndex = 0u;
             endpointIndex < 2u;
             ++endpointIndex) {
            const uint bodyIndex =
                endpointIndex == 0u ? contact.bodyA : contact.bodyB;
            if (!bodyIsDynamic(bodies[bodyIndex])) {
                continue;
            }
            bool alreadyBackedUp = false;
            for (uint backupIndex = 0u;
                 backupIndex < bodyBackupCount;
                 ++backupIndex) {
                alreadyBackedUp =
                    alreadyBackedUp ||
                    bodyIndices[backupIndex] == bodyIndex;
            }
            if (alreadyBackedUp) {
                continue;
            }
            bodyIndices[bodyBackupCount] = bodyIndex;
            linearVelocityBackups[bodyBackupCount] =
                bodies[bodyIndex].linearVelocityAndInverseMass;
            angularVelocityBackups[bodyBackupCount] =
                bodies[bodyIndex].angularVelocity;
            ++bodyBackupCount;
        }
    }

    if (!warmStartContacts(batch, bodies, contacts)) {
        restoreSolverState(
            bodies,
            contacts,
            batch.contactOffset,
            batch.contactCount,
            bodyBackupCount,
            bodyIndices,
            linearVelocityBackups,
            angularVelocityBackups,
            impulseBackups,
            flagBackups
        );
        status.code = MR_STEP_NONFINITE_RESULT;
        statuses[batchIndex] = status;
        return;
    }

    bool converged = false;
    float finalImpulseDelta = 0.0f;
    for (uint iteration = 0u;
         iteration < batch.velocityIterations;
         ++iteration) {
        float iterationImpulseDelta = 0.0f;
        for (uint localIndex = 0u;
             localIndex < batch.contactCount;
             ++localIndex) {
            device MRContactConstraintGPU& contact =
                contacts[batch.contactOffset + localIndex];
            if ((contact.flags &
                 MR_CONSTRAINT_FLAG_DISABLED) != 0u) {
                continue;
            }
            if (!solveOneContact(
                    batch,
                    bodies,
                    contact,
                    iterationImpulseDelta
                )) {
                restoreSolverState(
                    bodies,
                    contacts,
                    batch.contactOffset,
                    batch.contactCount,
                    bodyBackupCount,
                    bodyIndices,
                    linearVelocityBackups,
                    angularVelocityBackups,
                    impulseBackups,
                    flagBackups
                );
                status.code = MR_STEP_NONFINITE_RESULT;
                status.iterations = iteration + 1u;
                status.residuals.x = iterationImpulseDelta;
                statuses[batchIndex] = status;
                return;
            }
        }

        finalImpulseDelta = iterationImpulseDelta;
        status.iterations = iteration + 1u;
        if (batch.deterministic == 0u &&
            batch.enableEarlyExit != 0u &&
            iterationImpulseDelta <= batch.convergence.x) {
            converged = true;
            break;
        }
    }

    float maximumNormalResidual;
    float maximumConeViolation;
    if (!computeFinalResiduals(
            batch,
            bodies,
            contacts,
            maximumNormalResidual,
            maximumConeViolation
        ) ||
        !isfinite(finalImpulseDelta)) {
        restoreSolverState(
            bodies,
            contacts,
            batch.contactOffset,
            batch.contactCount,
            bodyBackupCount,
            bodyIndices,
            linearVelocityBackups,
            angularVelocityBackups,
            impulseBackups,
            flagBackups
        );
        status.code = MR_STEP_NONFINITE_RESULT;
        statuses[batchIndex] = status;
        return;
    }

    status.residuals.x = finalImpulseDelta;
    status.residuals.y = maximumNormalResidual;
    status.residuals.z = maximumConeViolation;
    status.code =
        converged
        ? MR_STEP_SUCCESS
        : MR_STEP_FIXED_BUDGET_COMPLETE;
    statuses[batchIndex] = status;
}
