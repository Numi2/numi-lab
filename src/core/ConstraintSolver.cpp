#include "metalrobo/ConstraintSolver.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

constexpr double kTiny = 1.0e-12;
constexpr double kMaximumImpulse = 1.0e30;
constexpr double kMaximumFloat = std::numeric_limits<float>::max();

struct Vec2 {
    double x = 0.0;
    double y = 0.0;
};

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

Vec2 operator+(const Vec2 a, const Vec2 b) {
    return {a.x + b.x, a.y + b.y};
}

Vec2 operator-(const Vec2 a, const Vec2 b) {
    return {a.x - b.x, a.y - b.y};
}

Vec2 operator*(const Vec2 value, const double scale) {
    return {value.x * scale, value.y * scale};
}

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

double dot(const Vec2 a, const Vec2 b) {
    return a.x * b.x + a.y * b.y;
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

double length(const Vec2 value) {
    return std::sqrt(dot(value, value));
}

double length(const Vec3 value) {
    return std::sqrt(dot(value, value));
}

bool finite(const double value) {
    return std::isfinite(value);
}

bool finite(const Vec2 value) {
    return finite(value.x) && finite(value.y);
}

bool finite(const Vec3 value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

bool finite(const mr_float4 value) {
    return finite(value.x) && finite(value.y) && finite(value.z) &&
        finite(value.w);
}

bool representableAsFloat(const double value) {
    return finite(value) && std::abs(value) <= kMaximumFloat;
}

bool preservedByFloat(const double value) {
    return representableAsFloat(value) &&
        (value == 0.0 || static_cast<float>(value) != 0.0f);
}

bool representableAsFloat(const Vec2 value) {
    return representableAsFloat(value.x) &&
        representableAsFloat(value.y);
}

bool representableAsFloat(const Vec3 value) {
    return representableAsFloat(value.x) &&
        representableAsFloat(value.y) &&
        representableAsFloat(value.z);
}

float checkedFloat(const double value) {
    if (!representableAsFloat(value)) {
        throw std::invalid_argument(
            "solver configuration is not representable in FP32"
        );
    }
    const float narrowed = static_cast<float>(value);
    if (value != 0.0 && narrowed == 0.0f) {
        throw std::invalid_argument(
            "solver configuration underflows the FP32 ABI"
        );
    }
    return narrowed;
}

Vec3 xyz(const mr_float4 value) {
    return {value.x, value.y, value.z};
}

mr_float4 f4(const Vec3 value, const float w) {
    return {
        static_cast<float>(value.x),
        static_cast<float>(value.y),
        static_cast<float>(value.z),
        w,
    };
}

bool bodyIsDynamic(const MRBodyStateGPU& body) {
    return body.flagsAndIndices[0] == MR_MOTION_DYNAMIC &&
        body.linearVelocityAndInverseMass.w > 0.0f;
}

Vec3 inverseInertiaMultiply(
    const MRBodyStateGPU& body,
    const Vec3 value
) {
    return {
        dot(xyz(body.inverseInertiaWorldRow0), value),
        dot(xyz(body.inverseInertiaWorldRow1), value),
        dot(xyz(body.inverseInertiaWorldRow2), value),
    };
}

Vec3 pointVelocity(const MRBodyStateGPU& body, const Vec3 offset) {
    if (body.flagsAndIndices[0] == MR_MOTION_STATIC) {
        return {};
    }
    return xyz(body.linearVelocityAndInverseMass) +
        cross(xyz(body.angularVelocity), offset);
}

Vec3 relativePointVelocity(
    const MRBodyStateGPU& bodyA,
    const MRBodyStateGPU& bodyB,
    const Vec3 offsetA,
    const Vec3 offsetB
) {
    return
        pointVelocity(bodyB, offsetB) -
        pointVelocity(bodyA, offsetA);
}

double directionalCoupling(
    const MRBodyStateGPU& bodyA,
    const MRBodyStateGPU& bodyB,
    const Vec3 offsetA,
    const Vec3 offsetB,
    const Vec3 direction0,
    const Vec3 direction1
) {
    double result = 0.0;
    if (bodyIsDynamic(bodyA)) {
        result += bodyA.linearVelocityAndInverseMass.w *
            dot(direction0, direction1);
        result += dot(
            cross(offsetA, direction0),
            inverseInertiaMultiply(
                bodyA,
                cross(offsetA, direction1)
            )
        );
    }
    if (bodyIsDynamic(bodyB)) {
        result += bodyB.linearVelocityAndInverseMass.w *
            dot(direction0, direction1);
        result += dot(
            cross(offsetB, direction0),
            inverseInertiaMultiply(
                bodyB,
                cross(offsetB, direction1)
            )
        );
    }
    return result;
}

double angularCoupling(
    const MRBodyStateGPU& bodyA,
    const MRBodyStateGPU& bodyB,
    const Vec3 direction
) {
    double result = 0.0;
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

std::pair<Vec3, Vec3> contactBasis(
    const Vec3 unitNormal,
    const Vec3 authoredTangent
) {
    const Vec3 projected =
        authoredTangent -
        unitNormal * dot(unitNormal, authoredTangent);
    const Vec3 normalizedU = projected / length(projected);
    return {normalizedU, cross(unitNormal, normalizedU)};
}

bool validBody(const MRBodyStateGPU& body) {
    return finite(body.position) && finite(body.orientation) &&
        finite(body.linearVelocityAndInverseMass) &&
        finite(body.angularVelocity) &&
        finite(body.inverseInertiaWorldRow0) &&
        finite(body.inverseInertiaWorldRow1) &&
        finite(body.inverseInertiaWorldRow2) &&
        body.linearVelocityAndInverseMass.w >= 0.0f &&
        body.flagsAndIndices[0] <= MR_MOTION_DYNAMIC;
}

bool validContact(const MRContactConstraintGPU& contact) {
    const Vec3 normal = xyz(contact.normal);
    const Vec3 tangent = xyz(contact.tangent);
    const double normalLengthSquared = dot(
        normal,
        normal
    );
    const double tangentLengthSquared = dot(tangent, tangent);
    return finite(contact.pointAndSeparation) &&
        finite(contact.normal) &&
        finite(contact.tangent) &&
        finite(contact.friction) &&
        finite(contact.response) &&
        finite(contact.targetVelocityAndPreSolveNormal) &&
        finite(contact.impulses) &&
        std::abs(normalLengthSquared - 1.0) <= 2.0e-4 &&
        std::abs(tangentLengthSquared - 1.0) <= 2.0e-4 &&
        std::abs(dot(normal, tangent)) <= 2.0e-4 &&
        contact.friction.x >= 0.0f &&
        contact.friction.y >= 0.0f &&
        contact.friction.x >= contact.friction.y &&
        contact.friction.z >= 0.0f &&
        contact.friction.w >= 0.0f &&
        contact.response.x >= 0.0f &&
        contact.response.x <= 1.0f &&
        contact.response.y >= 0.0f &&
        contact.response.z >= 0.0f &&
        contact.response.w >= 0.0f;
}

bool applyPairImpulse(
    MRBodyStateGPU& bodyA,
    MRBodyStateGPU& bodyB,
    const Vec3 offsetA,
    const Vec3 offsetB,
    const Vec3 impulseOnB
) {
    Vec3 linearA = xyz(bodyA.linearVelocityAndInverseMass);
    Vec3 angularA = xyz(bodyA.angularVelocity);
    Vec3 linearB = xyz(bodyB.linearVelocityAndInverseMass);
    Vec3 angularB = xyz(bodyB.angularVelocity);
    if (bodyIsDynamic(bodyA)) {
        linearA = linearA -
            impulseOnB * bodyA.linearVelocityAndInverseMass.w;
        angularA = angularA -
            inverseInertiaMultiply(
                bodyA,
                cross(offsetA, impulseOnB)
            );
    }
    if (bodyIsDynamic(bodyB)) {
        linearB = linearB +
            impulseOnB * bodyB.linearVelocityAndInverseMass.w;
        angularB = angularB +
            inverseInertiaMultiply(
                bodyB,
                cross(offsetB, impulseOnB)
            );
    }
    if (!finite(linearA) || !finite(angularA) ||
        !finite(linearB) || !finite(angularB) ||
        !representableAsFloat(linearA) ||
        !representableAsFloat(angularA) ||
        !representableAsFloat(linearB) ||
        !representableAsFloat(angularB)) {
        return false;
    }
    if (bodyIsDynamic(bodyA)) {
        bodyA.linearVelocityAndInverseMass =
            f4(linearA, bodyA.linearVelocityAndInverseMass.w);
        bodyA.angularVelocity = f4(angularA, bodyA.angularVelocity.w);
    }
    if (bodyIsDynamic(bodyB)) {
        bodyB.linearVelocityAndInverseMass =
            f4(linearB, bodyB.linearVelocityAndInverseMass.w);
        bodyB.angularVelocity = f4(angularB, bodyB.angularVelocity.w);
    }
    return true;
}

bool applyPairAngularImpulse(
    MRBodyStateGPU& bodyA,
    MRBodyStateGPU& bodyB,
    const Vec3 angularImpulseOnB
) {
    Vec3 angularA = xyz(bodyA.angularVelocity);
    Vec3 angularB = xyz(bodyB.angularVelocity);
    if (bodyIsDynamic(bodyA)) {
        angularA = angularA -
            inverseInertiaMultiply(bodyA, angularImpulseOnB);
    }
    if (bodyIsDynamic(bodyB)) {
        angularB = angularB +
            inverseInertiaMultiply(bodyB, angularImpulseOnB);
    }
    if (!finite(angularA) || !finite(angularB) ||
        !representableAsFloat(angularA) ||
        !representableAsFloat(angularB)) {
        return false;
    }
    if (bodyIsDynamic(bodyA)) {
        bodyA.angularVelocity = f4(angularA, bodyA.angularVelocity.w);
    }
    if (bodyIsDynamic(bodyB)) {
        bodyB.angularVelocity = f4(angularB, bodyB.angularVelocity.w);
    }
    return true;
}

double normalSoftness(
    const MRContactConstraintGPU& contact,
    const ContactSolverConfig& config
) {
    return contact.response.z / (config.timestep * config.timestep);
}

double normalTargetVelocity(
    const MRContactConstraintGPU& contact,
    const Vec3 normal,
    const ContactSolverConfig& config
) {
    const double penetration = std::min(
        static_cast<double>(contact.pointAndSeparation.w) +
            config.penetrationSlop,
        0.0
    );
    const double positionalTarget = std::min(
        config.maxDepenetrationVelocity,
        -config.errorReduction * penetration / config.timestep
    );
    double restitutionTarget = 0.0;
    if ((contact.flags & MR_CONSTRAINT_FLAG_NEW_IMPACT) != 0u &&
        contact.targetVelocityAndPreSolveNormal.w <
            -contact.response.y) {
        restitutionTarget =
            -static_cast<double>(contact.response.x) *
            contact.targetVelocityAndPreSolveNormal.w;
    }
    return
        dot(xyz(contact.targetVelocityAndPreSolveNormal), normal) +
        std::max(positionalTarget, restitutionTarget);
}

Vec2 projectFriction(
    const Vec2 candidate,
    const double normalImpulse,
    const double staticFriction,
    const double dynamicFriction
) {
    const double candidateLength = length(candidate);
    if (!(candidateLength > 0.0) || !(normalImpulse > 0.0)) {
        return {};
    }
    const double staticLimit =
        std::max(staticFriction, dynamicFriction) * normalImpulse;
    if (candidateLength <= staticLimit) {
        return candidate;
    }
    return candidate *
        (dynamicFriction * normalImpulse / candidateLength);
}

struct EffectiveMassRange {
    double minimum = kMaximumImpulse;
    double maximum = 0.0;
};

bool validateEffectiveMasses(
    const std::span<const MRBodyStateGPU> bodies,
    const std::span<const MRContactConstraintGPU> contacts,
    const ContactSolverConfig& config,
    EffectiveMassRange& range
) {
    const double minimumLinear = std::max(
        config.minimumInverseLinearEffectiveMass,
        kTiny
    );
    const double minimumAngular = std::max(
        config.minimumInverseAngularEffectiveMass,
        kTiny
    );
    for (const MRContactConstraintGPU& contact : contacts) {
        if ((contact.flags & MR_CONSTRAINT_FLAG_DISABLED) != 0u) {
            continue;
        }
        const MRBodyStateGPU& bodyA = bodies[contact.bodyA];
        const MRBodyStateGPU& bodyB = bodies[contact.bodyB];
        const Vec3 point = xyz(contact.pointAndSeparation);
        const Vec3 offsetA = point - xyz(bodyA.position);
        const Vec3 offsetB = point - xyz(bodyB.position);
        const Vec3 normal =
            xyz(contact.normal) / length(xyz(contact.normal));
        const double normalMass = directionalCoupling(
            bodyA,
            bodyB,
            offsetA,
            offsetB,
            normal,
            normal
        ) + normalSoftness(contact, config);
        if (!finite(normalMass) || normalMass < minimumLinear) {
            return false;
        }
        range.minimum = std::min(range.minimum, normalMass);
        range.maximum = std::max(range.maximum, normalMass);

        if (std::max(contact.friction.x, contact.friction.y) > 0.0f) {
            const auto [tangentU, tangentV] =
                contactBasis(normal, xyz(contact.tangent));
            const double kuu = directionalCoupling(
                bodyA, bodyB, offsetA, offsetB, tangentU, tangentU
            );
            const double kvv = directionalCoupling(
                bodyA, bodyB, offsetA, offsetB, tangentV, tangentV
            );
            const double kuv = 0.5 * (
                directionalCoupling(
                    bodyA, bodyB, offsetA, offsetB, tangentU, tangentV
                ) +
                directionalCoupling(
                    bodyA, bodyB, offsetA, offsetB, tangentV, tangentU
                )
            );
            const double determinant = kuu * kvv - kuv * kuv;
            const double discriminant = std::sqrt(std::max(
                (kuu - kvv) * (kuu - kvv) + 4.0 * kuv * kuv,
                0.0
            ));
            const double tangentMinimum =
                0.5 * (kuu + kvv - discriminant);
            const double tangentMaximum =
                0.5 * (kuu + kvv + discriminant);
            if (!finite(determinant) ||
                determinant <= minimumLinear * minimumLinear ||
                tangentMinimum < minimumLinear) {
                return false;
            }
            range.minimum = std::min(range.minimum, tangentMinimum);
            range.maximum = std::max(range.maximum, tangentMaximum);
        }
        if (contact.friction.w > 0.0f) {
            const double torsional =
                angularCoupling(bodyA, bodyB, normal);
            if (!finite(torsional) || torsional < minimumAngular) {
                return false;
            }
        }
    }
    return true;
}

bool warmStart(
    const std::span<MRBodyStateGPU> bodies,
    const std::span<MRContactConstraintGPU> contacts,
    const ContactSolverConfig& config
) {
    const double scale = std::max(config.warmStartScale, 0.0);
    for (MRContactConstraintGPU& contact : contacts) {
        if ((contact.flags & MR_CONSTRAINT_FLAG_DISABLED) != 0u ||
            !config.enableWarmStart) {
            contact.impulses = {};
            contact.flags &= ~MR_CONSTRAINT_FLAG_WARM_STARTED;
            continue;
        }
        MRBodyStateGPU& bodyA = bodies[contact.bodyA];
        MRBodyStateGPU& bodyB = bodies[contact.bodyB];
        const Vec3 point = xyz(contact.pointAndSeparation);
        const Vec3 offsetA = point - xyz(bodyA.position);
        const Vec3 offsetB = point - xyz(bodyB.position);
        const Vec3 normal =
            xyz(contact.normal) / length(xyz(contact.normal));
        const auto [tangentU, tangentV] =
            contactBasis(normal, xyz(contact.tangent));

        double normalImpulse =
            std::max(static_cast<double>(contact.impulses.x) * scale, 0.0);
        if (contact.response.w > 0.0f) {
            normalImpulse = std::min(
                normalImpulse,
                static_cast<double>(contact.response.w)
            );
        }
        normalImpulse = std::min(normalImpulse, kMaximumImpulse);
        const Vec2 tangentImpulse = projectFriction(
            {
                static_cast<double>(contact.impulses.y) * scale,
                static_cast<double>(contact.impulses.z) * scale,
            },
            normalImpulse,
            contact.friction.x,
            contact.friction.y
        );
        const double torsionalLimit =
            contact.friction.w * normalImpulse;
        const double torsionalImpulse = std::clamp(
            static_cast<double>(contact.impulses.w) * scale,
            -torsionalLimit,
            torsionalLimit
        );
        if (!representableAsFloat(normalImpulse) ||
            !representableAsFloat(tangentImpulse) ||
            !representableAsFloat(torsionalImpulse)) {
            return false;
        }
        contact.impulses = {
            static_cast<float>(normalImpulse),
            static_cast<float>(tangentImpulse.x),
            static_cast<float>(tangentImpulse.y),
            static_cast<float>(torsionalImpulse),
        };
        if (!applyPairImpulse(
                bodyA,
                bodyB,
                offsetA,
                offsetB,
                normal * normalImpulse +
                    tangentU * tangentImpulse.x +
                    tangentV * tangentImpulse.y
            ) ||
            !applyPairAngularImpulse(
                bodyA,
                bodyB,
                normal * torsionalImpulse
            )) {
            return false;
        }
        if (normalImpulse > 0.0 || length(tangentImpulse) > 0.0 ||
            torsionalImpulse != 0.0) {
            contact.flags |= MR_CONSTRAINT_FLAG_WARM_STARTED;
        } else {
            contact.flags &= ~MR_CONSTRAINT_FLAG_WARM_STARTED;
        }
    }
    return true;
}

bool solveOne(
    const std::span<MRBodyStateGPU> bodies,
    MRContactConstraintGPU& contact,
    const ContactSolverConfig& config,
    double& maximumImpulseDelta
) {
    MRBodyStateGPU& bodyA = bodies[contact.bodyA];
    MRBodyStateGPU& bodyB = bodies[contact.bodyB];
    const Vec3 point = xyz(contact.pointAndSeparation);
    const Vec3 offsetA = point - xyz(bodyA.position);
    const Vec3 offsetB = point - xyz(bodyB.position);
    const Vec3 normal =
        xyz(contact.normal) / length(xyz(contact.normal));
    const auto [tangentU, tangentV] =
        contactBasis(normal, xyz(contact.tangent));

    const double softness = normalSoftness(contact, config);
    const double normalDenominator = directionalCoupling(
        bodyA,
        bodyB,
        offsetA,
        offsetB,
        normal,
        normal
    ) + softness;
    const double normalVelocity = dot(
        relativePointVelocity(bodyA, bodyB, offsetA, offsetB),
        normal
    );
    const double oldNormalImpulse = contact.impulses.x;
    double newNormalImpulse = std::max(
        oldNormalImpulse +
            (
                normalTargetVelocity(contact, normal, config) -
                normalVelocity -
                softness * oldNormalImpulse
            ) / normalDenominator,
        0.0
    );
    if (contact.response.w > 0.0f) {
        newNormalImpulse = std::min(
            newNormalImpulse,
            static_cast<double>(contact.response.w)
        );
    }
    newNormalImpulse = std::min(newNormalImpulse, kMaximumImpulse);
    const double normalDelta = newNormalImpulse - oldNormalImpulse;
    if (!finite(normalDelta) ||
        !representableAsFloat(newNormalImpulse) ||
        !applyPairImpulse(
            bodyA,
            bodyB,
            offsetA,
            offsetB,
            normal * normalDelta
        )) {
        return false;
    }
    contact.impulses.x = static_cast<float>(newNormalImpulse);
    maximumImpulseDelta =
        std::max(maximumImpulseDelta, std::abs(normalDelta));

    const double staticFriction =
        std::max(contact.friction.x, contact.friction.y);
    const double dynamicFriction = contact.friction.y;
    if (staticFriction > 0.0) {
        const double kuu = directionalCoupling(
            bodyA, bodyB, offsetA, offsetB, tangentU, tangentU
        );
        const double kvv = directionalCoupling(
            bodyA, bodyB, offsetA, offsetB, tangentV, tangentV
        );
        const double kuv = 0.5 * (
            directionalCoupling(
                bodyA, bodyB, offsetA, offsetB, tangentU, tangentV
            ) +
            directionalCoupling(
                bodyA, bodyB, offsetA, offsetB, tangentV, tangentU
            )
        );
        const double inverseDeterminant =
            1.0 / (kuu * kvv - kuv * kuv);
        const Vec3 tangentVelocity =
            relativePointVelocity(bodyA, bodyB, offsetA, offsetB) -
            xyz(contact.targetVelocityAndPreSolveNormal);
        const Vec2 right{
            -dot(tangentVelocity, tangentU),
            -dot(tangentVelocity, tangentV),
        };
        const Vec2 deltaUnclamped{
            (kvv * right.x - kuv * right.y) * inverseDeterminant,
            (kuu * right.y - kuv * right.x) * inverseDeterminant,
        };
        const Vec2 oldTangent{
            contact.impulses.y,
            contact.impulses.z,
        };
        const Vec2 newTangent = projectFriction(
            oldTangent + deltaUnclamped,
            newNormalImpulse,
            staticFriction,
            dynamicFriction
        );
        const Vec2 tangentDelta = newTangent - oldTangent;
        if (!finite(tangentDelta) ||
            !representableAsFloat(newTangent) ||
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
        contact.impulses.y = static_cast<float>(newTangent.x);
        contact.impulses.z = static_cast<float>(newTangent.y);
        maximumImpulseDelta =
            std::max(maximumImpulseDelta, length(tangentDelta));
    } else if (contact.impulses.y != 0.0f ||
               contact.impulses.z != 0.0f) {
        const Vec2 tangentDelta{
            -contact.impulses.y,
            -contact.impulses.z,
        };
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
        contact.impulses.y = 0.0f;
        contact.impulses.z = 0.0f;
        maximumImpulseDelta =
            std::max(maximumImpulseDelta, length(tangentDelta));
    }

    if (contact.friction.w > 0.0f) {
        const double torsionalDenominator =
            angularCoupling(bodyA, bodyB, normal);
        const double torsionalVelocity = dot(
            xyz(bodyB.angularVelocity) - xyz(bodyA.angularVelocity),
            normal
        );
        const double oldTorsional = contact.impulses.w;
        const double limit =
            contact.friction.w * newNormalImpulse;
        const double newTorsional = std::clamp(
            oldTorsional -
                torsionalVelocity / torsionalDenominator,
            -limit,
            limit
        );
        const double delta = newTorsional - oldTorsional;
        if (!finite(delta) ||
            !representableAsFloat(newTorsional) ||
            !applyPairAngularImpulse(
                bodyA,
                bodyB,
                normal * delta
            )) {
            return false;
        }
        contact.impulses.w = static_cast<float>(newTorsional);
        maximumImpulseDelta =
            std::max(maximumImpulseDelta, std::abs(delta));
    } else if (contact.impulses.w != 0.0f) {
        const double delta = -contact.impulses.w;
        if (!applyPairAngularImpulse(
                bodyA,
                bodyB,
                normal * delta
            )) {
            return false;
        }
        contact.impulses.w = 0.0f;
        maximumImpulseDelta =
            std::max(maximumImpulseDelta, std::abs(delta));
    }
    return finite(contact.impulses);
}

bool finalResiduals(
    const std::span<const MRBodyStateGPU> bodies,
    const std::span<const MRContactConstraintGPU> contacts,
    const ContactSolverConfig& config,
    double& maximumNormalResidual,
    double& maximumConeViolation
) {
    for (const MRContactConstraintGPU& contact : contacts) {
        if ((contact.flags & MR_CONSTRAINT_FLAG_DISABLED) != 0u) {
            continue;
        }
        const MRBodyStateGPU& bodyA = bodies[contact.bodyA];
        const MRBodyStateGPU& bodyB = bodies[contact.bodyB];
        const Vec3 point = xyz(contact.pointAndSeparation);
        const Vec3 offsetA = point - xyz(bodyA.position);
        const Vec3 offsetB = point - xyz(bodyB.position);
        const Vec3 normal =
            xyz(contact.normal) / length(xyz(contact.normal));
        const double residual =
            dot(
                relativePointVelocity(bodyA, bodyB, offsetA, offsetB),
                normal
            ) -
            normalTargetVelocity(contact, normal, config) +
            normalSoftness(contact, config) * contact.impulses.x;
        double complementarity = 0.0;
        if (contact.impulses.x <= config.impulseTolerance) {
            complementarity = std::max(-residual, 0.0);
        } else if (
            contact.response.w > 0.0f &&
            contact.impulses.x >=
                contact.response.w - config.impulseTolerance
        ) {
            complementarity = std::max(residual, 0.0);
        } else {
            complementarity = std::abs(residual);
        }
        maximumNormalResidual =
            std::max(maximumNormalResidual, complementarity);
        maximumConeViolation = std::max(
            maximumConeViolation,
            std::max(
                std::hypot(
                    static_cast<double>(contact.impulses.y),
                    static_cast<double>(contact.impulses.z)
                ) -
                    static_cast<double>(
                        std::max(contact.friction.x, contact.friction.y)
                    ) *
                        contact.impulses.x,
                std::abs(static_cast<double>(contact.impulses.w)) -
                    contact.friction.w * contact.impulses.x
            )
        );
    }
    maximumConeViolation = std::max(maximumConeViolation, 0.0);
    return finite(maximumNormalResidual) && finite(maximumConeViolation);
}

} // namespace

std::vector<ConstraintIsland> buildConstraintIslands(
    const std::span<const MRBodyStateGPU> bodies,
    const std::span<const MRContactConstraintGPU> contacts
) {
    std::vector<std::uint32_t> parent(bodies.size());
    std::iota(parent.begin(), parent.end(), 0u);
    const auto findRoot = [&parent](std::uint32_t body) {
        std::uint32_t root = body;
        while (parent[root] != root) {
            root = parent[root];
        }
        while (parent[body] != body) {
            const std::uint32_t next = parent[body];
            parent[body] = root;
            body = next;
        }
        return root;
    };
    const auto dynamic = [&bodies](const std::uint32_t body) {
        return body < bodies.size() && bodyIsDynamic(bodies[body]);
    };

    for (const MRContactConstraintGPU& contact : contacts) {
        if ((contact.flags & MR_CONSTRAINT_FLAG_DISABLED) != 0u ||
            contact.bodyA >= bodies.size() ||
            contact.bodyB >= bodies.size()) {
            continue;
        }
        if (dynamic(contact.bodyA) && dynamic(contact.bodyB)) {
            std::uint32_t rootA = findRoot(contact.bodyA);
            std::uint32_t rootB = findRoot(contact.bodyB);
            if (rootA != rootB) {
                const std::uint32_t low = std::min(rootA, rootB);
                const std::uint32_t high = std::max(rootA, rootB);
                parent[high] = low;
            }
        }
    }

    std::vector<std::uint8_t> rootHasContact(bodies.size(), 0u);
    std::vector<std::uint32_t> contactRoot(
        contacts.size(),
        MR_INVALID_INDEX
    );
    for (std::size_t index = 0; index < contacts.size(); ++index) {
        const MRContactConstraintGPU& contact = contacts[index];
        if ((contact.flags & MR_CONSTRAINT_FLAG_DISABLED) != 0u ||
            contact.bodyA >= bodies.size() ||
            contact.bodyB >= bodies.size()) {
            continue;
        }
        std::uint32_t body = MR_INVALID_INDEX;
        if (dynamic(contact.bodyA)) {
            body = contact.bodyA;
        } else if (dynamic(contact.bodyB)) {
            body = contact.bodyB;
        }
        if (body != MR_INVALID_INDEX) {
            const std::uint32_t root = findRoot(body);
            contactRoot[index] = root;
            rootHasContact[root] = 1u;
        }
    }

    std::vector<std::uint32_t> rootToIsland(
        bodies.size(),
        MR_INVALID_INDEX
    );
    std::vector<ConstraintIsland> islands;
    for (std::uint32_t body = 0u; body < bodies.size(); ++body) {
        if (findRoot(body) == body && rootHasContact[body] != 0u) {
            rootToIsland[body] =
                static_cast<std::uint32_t>(islands.size());
            islands.emplace_back();
        }
    }
    for (std::uint32_t body = 0u; body < bodies.size(); ++body) {
        if (dynamic(body)) {
            const std::uint32_t root = findRoot(body);
            if (rootToIsland[root] != MR_INVALID_INDEX) {
                islands[rootToIsland[root]].dynamicBodies.push_back(body);
            }
        }
    }
    for (std::uint32_t contact = 0u; contact < contacts.size(); ++contact) {
        const std::uint32_t root = contactRoot[contact];
        if (root != MR_INVALID_INDEX) {
            islands[rootToIsland[root]].contacts.push_back(contact);
        }
    }
    return islands;
}

std::size_t ContactImpulseCache::KeyHash::operator()(
    const Key& key
) const noexcept {
    std::uint64_t value =
        key.pair + 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    value ^= value >> 31u;
    value ^= key.feature +
        0x9e3779b97f4a7c15ull + (value << 6u) + (value >> 2u);
    return static_cast<std::size_t>(value);
}

void ContactImpulseCache::beginStep(const std::uint64_t step) {
    step_ = step;
}

void ContactImpulseCache::seed(
    const std::span<MRContactConstraintGPU> contacts
) const {
    for (MRContactConstraintGPU& contact : contacts) {
        const auto entry = entries_.find({contact.pairKey, contact.featureKey});
        if (entry == entries_.end() ||
            (contact.flags & MR_CONSTRAINT_FLAG_NEW_IMPACT) != 0u ||
            entry->second.lastSeenStep > step_ ||
            step_ - entry->second.lastSeenStep > 1u) {
            contact.impulses = {};
            contact.flags &= ~MR_CONSTRAINT_FLAG_WARM_STARTED;
            continue;
        }
        const double newNormalLength = length(xyz(contact.normal));
        const double oldNormalLength = length(xyz(entry->second.normal));
        if (!(newNormalLength > kTiny) ||
            !(oldNormalLength > kTiny)) {
            contact.impulses = {};
            contact.flags &= ~MR_CONSTRAINT_FLAG_WARM_STARTED;
            continue;
        }
        const Vec3 newNormal =
            xyz(contact.normal) / newNormalLength;
        const Vec3 oldNormal =
            xyz(entry->second.normal) / oldNormalLength;
        if (dot(newNormal, oldNormal) < 0.5) {
            contact.impulses = {};
            contact.flags &= ~MR_CONSTRAINT_FLAG_WARM_STARTED;
            continue;
        }
        const auto [newTangentU, newTangentV] =
            contactBasis(newNormal, xyz(contact.tangent));
        const Vec3 oldTangentU = xyz(entry->second.tangentU);
        const Vec3 oldTangentV = cross(oldNormal, oldTangentU);
        const Vec3 tangentWorld =
            oldTangentU * entry->second.impulses.y +
            oldTangentV * entry->second.impulses.z;
        const double normalAlignment =
            std::clamp(dot(newNormal, oldNormal), 0.0, 1.0);
        contact.impulses = {
            static_cast<float>(
                entry->second.impulses.x * normalAlignment
            ),
            static_cast<float>(dot(tangentWorld, newTangentU)),
            static_cast<float>(dot(tangentWorld, newTangentV)),
            static_cast<float>(
                entry->second.impulses.w * normalAlignment
            ),
        };
        contact.flags |= MR_CONSTRAINT_FLAG_WARM_STARTED;
    }
}

void ContactImpulseCache::commit(
    const std::span<const MRContactConstraintGPU> contacts
) {
    for (const MRContactConstraintGPU& contact : contacts) {
        if ((contact.flags & MR_CONSTRAINT_FLAG_DISABLED) != 0u) {
            continue;
        }
        const Vec3 normal =
            xyz(contact.normal) / length(xyz(contact.normal));
        const auto [tangentU, tangentV] =
            contactBasis(normal, xyz(contact.tangent));
        static_cast<void>(tangentV);
        entries_[{contact.pairKey, contact.featureKey}] = {
            contact.impulses,
            f4(normal, 0.0f),
            f4(tangentU, 0.0f),
            step_,
        };
    }
}

void ContactImpulseCache::prune(const std::uint64_t maximumAge) {
    std::erase_if(entries_, [this, maximumAge](const auto& pair) {
        return step_ > pair.second.lastSeenStep &&
            step_ - pair.second.lastSeenStep > maximumAge;
    });
}

void ContactImpulseCache::clear() {
    entries_.clear();
}

std::size_t ContactImpulseCache::size() const noexcept {
    return entries_.size();
}

ContactSolverDiagnostics solveContactConstraints(
    const std::span<MRBodyStateGPU> bodies,
    const std::span<MRContactConstraintGPU> contacts,
    const ContactSolverConfig& config
) {
    ContactSolverDiagnostics diagnostics;
    if (contacts.size() > MR_MAX_CONTACTS_PER_SOLVER_BATCH) {
        diagnostics.code = MR_STEP_CONTACT_CAPACITY_OVERFLOW;
        diagnostics.requiredContacts = static_cast<std::uint32_t>(
            std::min<std::size_t>(
                contacts.size(),
                std::numeric_limits<std::uint32_t>::max()
            )
        );
        return diagnostics;
    }
    if (!(config.timestep > 0.0) ||
        !finite(config.timestep) ||
        !preservedByFloat(config.timestep) ||
        config.errorReduction < 0.0 ||
        !finite(config.errorReduction) ||
        !preservedByFloat(config.errorReduction) ||
        config.penetrationSlop < 0.0 ||
        !finite(config.penetrationSlop) ||
        !preservedByFloat(config.penetrationSlop) ||
        config.maxDepenetrationVelocity < 0.0 ||
        !finite(config.maxDepenetrationVelocity) ||
        !preservedByFloat(config.maxDepenetrationVelocity) ||
        config.impulseTolerance < 0.0 ||
        !finite(config.impulseTolerance) ||
        !preservedByFloat(config.impulseTolerance) ||
        config.warmStartScale < 0.0 ||
        !finite(config.warmStartScale) ||
        !preservedByFloat(config.warmStartScale) ||
        config.minimumInverseLinearEffectiveMass < 0.0 ||
        !finite(config.minimumInverseLinearEffectiveMass) ||
        !preservedByFloat(
            config.minimumInverseLinearEffectiveMass
        ) ||
        config.minimumInverseAngularEffectiveMass < 0.0 ||
        !finite(config.minimumInverseAngularEffectiveMass) ||
        !preservedByFloat(
            config.minimumInverseAngularEffectiveMass
        )) {
        diagnostics.code = MR_STEP_NONFINITE_INPUT;
        return diagnostics;
    }
    if (!std::ranges::all_of(bodies, validBody)) {
        diagnostics.code = MR_STEP_NONFINITE_INPUT;
        return diagnostics;
    }
    std::uint64_t previousPair = 0u;
    std::uint64_t previousFeature = 0u;
    bool havePreviousKey = false;
    for (const MRContactConstraintGPU& contact : contacts) {
        if ((contact.flags & MR_CONSTRAINT_FLAG_DISABLED) != 0u) {
            continue;
        }
        ++diagnostics.activeContacts;
        if (contact.bodyA >= bodies.size() ||
            contact.bodyB >= bodies.size() ||
            contact.bodyA == contact.bodyB) {
            diagnostics.code = MR_STEP_UNSUPPORTED;
            return diagnostics;
        }
        if (!bodyIsDynamic(bodies[contact.bodyA]) &&
            !bodyIsDynamic(bodies[contact.bodyB])) {
            diagnostics.code = MR_STEP_UNSUPPORTED;
            return diagnostics;
        }
        if (!validContact(contact)) {
            diagnostics.code = MR_STEP_NONFINITE_INPUT;
            return diagnostics;
        }
        if (contact.friction.z > 0.0f) {
            diagnostics.code = MR_STEP_UNSUPPORTED;
            return diagnostics;
        }
        if (config.deterministic) {
            const bool outOfOrder =
                havePreviousKey &&
                (
                    contact.pairKey < previousPair ||
                    (
                        contact.pairKey == previousPair &&
                        contact.featureKey < previousFeature
                    )
                );
            if (outOfOrder) {
                diagnostics.code = MR_STEP_UNSUPPORTED;
                return diagnostics;
            }
            previousPair = contact.pairKey;
            previousFeature = contact.featureKey;
            havePreviousKey = true;
        }
    }
    diagnostics.islandCount = static_cast<std::uint32_t>(
        buildConstraintIslands(bodies, contacts).size()
    );
    if (diagnostics.activeContacts == 0u) {
        for (MRContactConstraintGPU& contact : contacts) {
            contact.impulses = {};
            contact.flags &= ~MR_CONSTRAINT_FLAG_WARM_STARTED;
        }
        return diagnostics;
    }

    EffectiveMassRange range;
    if (!validateEffectiveMasses(bodies, contacts, config, range)) {
        diagnostics.code = MR_STEP_FACTORIZATION_FAILED;
        return diagnostics;
    }
    diagnostics.inverseLinearEffectiveMassSpread =
        range.maximum / std::max(range.minimum, kTiny);
    if (!finite(diagnostics.inverseLinearEffectiveMassSpread)) {
        diagnostics.code = MR_STEP_NONFINITE_INPUT;
        return diagnostics;
    }
    // The FP64/CPU path is transactional. The bounded Metal throughput kernel
    // backs up its touched velocity/impulse records and restores them on an
    // arithmetic failure; a future production world can instead publish
    // versioned output buffers to avoid that per-dispatch copy.
    const std::vector<MRBodyStateGPU> originalBodies(
        bodies.begin(),
        bodies.end()
    );
    const std::vector<MRContactConstraintGPU> originalContacts(
        contacts.begin(),
        contacts.end()
    );
    const auto rollback = [&]() {
        std::copy(originalBodies.begin(), originalBodies.end(), bodies.begin());
        std::copy(
            originalContacts.begin(),
            originalContacts.end(),
            contacts.begin()
        );
    };
    if (!warmStart(bodies, contacts, config)) {
        rollback();
        diagnostics.code = MR_STEP_NONFINITE_RESULT;
        return diagnostics;
    }

    bool converged = false;
    for (std::uint32_t iteration = 0u;
         iteration < config.velocityIterations;
         ++iteration) {
        double maximumDelta = 0.0;
        for (MRContactConstraintGPU& contact : contacts) {
            if ((contact.flags & MR_CONSTRAINT_FLAG_DISABLED) != 0u) {
                continue;
            }
            if (!solveOne(bodies, contact, config, maximumDelta)) {
                rollback();
                diagnostics.code = MR_STEP_NONFINITE_RESULT;
                diagnostics.iterations = iteration + 1u;
                diagnostics.maximumImpulseDelta = maximumDelta;
                return diagnostics;
            }
        }
        diagnostics.maximumImpulseDelta = maximumDelta;
        diagnostics.iterations = iteration + 1u;
        if (!config.deterministic &&
            config.enableEarlyExit &&
            maximumDelta <= config.impulseTolerance) {
            converged = true;
            break;
        }
    }
    if (!finalResiduals(
            bodies,
            contacts,
            config,
            diagnostics.maximumNormalResidual,
            diagnostics.maximumConeViolation
        )) {
        rollback();
        diagnostics.code = MR_STEP_NONFINITE_RESULT;
        return diagnostics;
    }
    diagnostics.code = converged
        ? MR_STEP_SUCCESS
        : MR_STEP_FIXED_BUDGET_COMPLETE;
    return diagnostics;
}

MRSolverBatchGPU makeSolverBatch(
    const std::uint32_t bodyOffset,
    const std::uint32_t bodyCount,
    const std::uint32_t contactOffset,
    const std::uint32_t contactCount,
    const ContactSolverConfig& config
) {
    MRSolverBatchGPU batch{};
    batch.bodyOffset = bodyOffset;
    batch.bodyCount = bodyCount;
    batch.contactOffset = contactOffset;
    batch.contactCount = contactCount;
    batch.velocityIterations = config.velocityIterations;
    batch.enableWarmStart = config.enableWarmStart ? 1u : 0u;
    batch.enableEarlyExit =
        config.enableEarlyExit && !config.deterministic ? 1u : 0u;
    batch.deterministic = config.deterministic ? 1u : 0u;
    batch.timestepAndBias = {
        checkedFloat(config.timestep),
        checkedFloat(config.errorReduction),
        checkedFloat(config.penetrationSlop),
        checkedFloat(config.maxDepenetrationVelocity),
    };
    batch.convergence = {
        checkedFloat(config.impulseTolerance),
        checkedFloat(config.warmStartScale),
        checkedFloat(config.minimumInverseLinearEffectiveMass),
        checkedFloat(config.minimumInverseAngularEffectiveMass),
    };
    return batch;
}

} // namespace metalrobo
