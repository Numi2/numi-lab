#include <metal_stdlib>

#include "metalrobo/engine_types.h"

using namespace metal;

namespace {

constant float kQuaternionTolerance = 1.0e-5f;
constant float kTiny = 1.0e-14f;
constant uint kPairSphereSphere = 1u;
constant uint kPairSpherePlane = 2u;

struct WorldShape {
    uint index;
    uint type;
    uint body;
    uint disabled;
    float3 center;
    float4 rotation;
    float3 lower;
    float3 upper;
    float3 planeNormal;
    float radius;
    float contactOffset;
};

bool finiteFloat4(const float4 value) {
    return all(isfinite(value));
}

bool finiteFloat3(const float3 value) {
    return all(isfinite(value));
}

float4 quaternionMultiply(const float4 left, const float4 right) {
    return float4(
        left.w * right.x + right.w * left.x +
            left.y * right.z - left.z * right.y,
        left.w * right.y + right.w * left.y +
            left.z * right.x - left.x * right.z,
        left.w * right.z + right.w * left.z +
            left.x * right.y - left.y * right.x,
        left.w * right.w -
            left.x * right.x -
            left.y * right.y -
            left.z * right.z
    );
}

float3 quaternionRotate(const float4 quaternion, const float3 value) {
    const float3 vector = quaternion.xyz;
    const float3 twiceCross = 2.0f * cross(vector, value);
    return value + quaternion.w * twiceCross +
        cross(vector, twiceCross);
}

bool checkedQuaternion(const float4 input, thread float4& output) {
    if (!finiteFloat4(input)) {
        return false;
    }
    const float squared = dot(input, input);
    if (!(squared > kTiny) ||
        abs(squared - 1.0f) > kQuaternionTolerance) {
        return false;
    }
    output = input * rsqrt(squared);
    return finiteFloat4(output);
}

uint supportedPairClass(const uint typeA, const uint typeB) {
    if (typeA == MR_SHAPE_SPHERE &&
        typeB == MR_SHAPE_SPHERE) {
        return kPairSphereSphere;
    }
    if ((typeA == MR_SHAPE_SPHERE &&
         typeB == MR_SHAPE_PLANE) ||
        (typeA == MR_SHAPE_PLANE &&
         typeB == MR_SHAPE_SPHERE)) {
        return kPairSpherePlane;
    }
    return 0u;
}

bool makeWorldShape(
    const uint index,
    device const MRShapeGPU* shapes,
    device const MRBodyStateGPU* bodies,
    const uint bodyCount,
    thread WorldShape& output,
    thread uint& failureCode
) {
    const MRShapeGPU shape = shapes[index];
    if (shape.bodyIndex >= bodyCount ||
        !finiteFloat4(shape.localPosition) ||
        !finiteFloat4(shape.localRotation) ||
        !finiteFloat4(shape.dimensions) ||
        !finiteFloat4(shape.contactRestAndBoundingRadius) ||
        shape.contactRestAndBoundingRadius.x < 0.0f ||
        shape.contactRestAndBoundingRadius.x <
            shape.contactRestAndBoundingRadius.y ||
        shape.contactRestAndBoundingRadius.z < 0.0f) {
        failureCode = MR_STEP_NONFINITE_INPUT;
        return false;
    }

    const MRBodyStateGPU body = bodies[shape.bodyIndex];
    if (!finiteFloat4(body.position) ||
        !finiteFloat4(body.orientation) ||
        body.flagsAndIndices[0] > MR_MOTION_DYNAMIC) {
        failureCode = MR_STEP_NONFINITE_INPUT;
        return false;
    }

    float4 bodyRotation;
    float4 localRotation;
    if (!checkedQuaternion(body.orientation, bodyRotation) ||
        !checkedQuaternion(shape.localRotation, localRotation)) {
        failureCode = MR_STEP_NONFINITE_INPUT;
        return false;
    }

    output.index = index;
    output.type = shape.shapeType;
    output.body = shape.bodyIndex;
    output.disabled =
        (shape.flags & MR_SHAPE_FLAG_SIMULATION_DISABLED) != 0u;
    output.rotation =
        quaternionMultiply(bodyRotation, localRotation);
    output.center = body.position.xyz +
        quaternionRotate(bodyRotation, shape.localPosition.xyz);
    output.contactOffset =
        shape.contactRestAndBoundingRadius.x;
    output.radius = 0.0f;
    output.lower = float3(0.0f);
    output.upper = float3(0.0f);
    output.planeNormal = float3(0.0f);

    if (!finiteFloat3(output.center) ||
        !finiteFloat4(output.rotation)) {
        failureCode = MR_STEP_NONFINITE_INPUT;
        return false;
    }
    if (output.disabled != 0u) {
        return true;
    }
    if (shape.shapeType != MR_SHAPE_SPHERE &&
        shape.shapeType != MR_SHAPE_PLANE) {
        failureCode = MR_STEP_UNSUPPORTED;
        return false;
    }

    if (shape.shapeType == MR_SHAPE_SPHERE) {
        output.radius = shape.dimensions.x;
        if (!(output.radius > 0.0f)) {
            failureCode = MR_STEP_NONFINITE_INPUT;
            return false;
        }
        const float expansion =
            output.radius + output.contactOffset;
        output.lower = output.center - expansion;
        output.upper = output.center + expansion;
        if (!finiteFloat3(output.lower) ||
            !finiteFloat3(output.upper)) {
            failureCode = MR_STEP_NONFINITE_INPUT;
            return false;
        }
        return true;
    }

    output.planeNormal = quaternionRotate(
        output.rotation,
        float3(0.0f, 1.0f, 0.0f)
    );
    const float normalSquared =
        dot(output.planeNormal, output.planeNormal);
    if (!(normalSquared > kTiny) || !isfinite(normalSquared)) {
        failureCode = MR_STEP_NONFINITE_INPUT;
        return false;
    }
    output.planeNormal *= rsqrt(normalSquared);
    if (!finiteFloat3(output.planeNormal)) {
        failureCode = MR_STEP_NONFINITE_INPUT;
        return false;
    }
    return true;
}

bool excludedPair(
    const uint colliderA,
    const uint colliderB,
    device const MRCandidatePairGPU* exclusions,
    const uint exclusionCount
) {
    for (uint index = 0u; index < exclusionCount; ++index) {
        const uint left = exclusions[index].colliderA;
        const uint right = exclusions[index].colliderB;
        if (left == right) {
            continue;
        }
        const uint low = min(left, right);
        const uint high = max(left, right);
        if (low == colliderA && high == colliderB) {
            return true;
        }
    }
    return false;
}

bool aabbOverlap(
    const thread WorldShape& left,
    const thread WorldShape& right
) {
    return all(left.lower <= right.upper) &&
        all(left.upper >= right.lower);
}

bool planeMayOverlap(
    const thread WorldShape& plane,
    const thread WorldShape& sphere
) {
    const float3 center = 0.5f * (sphere.lower + sphere.upper);
    const float3 halfExtent =
        0.5f * (sphere.upper - sphere.lower);
    const float minimumSignedDistance =
        dot(plane.planeNormal, center - plane.center) -
        dot(abs(plane.planeNormal), halfExtent);
    return minimumSignedDistance <= plane.contactOffset;
}

bool pairPassesFilter(
    const uint colliderA,
    const uint colliderB,
    const thread WorldShape& shapeA,
    const thread WorldShape& shapeB,
    device const MRShapeGPU* shapes,
    device const MRBodyStateGPU* bodies,
    device const MRCandidatePairGPU* exclusions,
    const uint exclusionCount,
    thread uint& pairClass
) {
    const MRShapeGPU sourceA = shapes[colliderA];
    const MRShapeGPU sourceB = shapes[colliderB];
    pairClass = supportedPairClass(shapeA.type, shapeB.type);
    if ((sourceA.flags & MR_SHAPE_FLAG_SIMULATION_DISABLED) != 0u ||
        (sourceB.flags & MR_SHAPE_FLAG_SIMULATION_DISABLED) != 0u ||
        pairClass == 0u ||
        sourceA.bodyIndex == sourceB.bodyIndex ||
        (sourceA.collisionGroup & sourceB.collisionMask) == 0u ||
        (sourceB.collisionGroup & sourceA.collisionMask) == 0u) {
        return false;
    }
    const uint motionA =
        bodies[sourceA.bodyIndex].flagsAndIndices[0];
    const uint motionB =
        bodies[sourceB.bodyIndex].flagsAndIndices[0];
    if ((motionA != MR_MOTION_DYNAMIC &&
         motionB != MR_MOTION_DYNAMIC) ||
        excludedPair(
            colliderA,
            colliderB,
            exclusions,
            exclusionCount
        )) {
        return false;
    }
    if (pairClass == kPairSphereSphere) {
        return aabbOverlap(shapeA, shapeB);
    }
    const thread WorldShape& plane =
        shapeA.type == MR_SHAPE_PLANE ? shapeA : shapeB;
    const thread WorldShape& sphere =
        shapeA.type == MR_SHAPE_SPHERE ? shapeA : shapeB;
    return planeMayOverlap(plane, sphere);
}

float3 coincidentNormal(
    const uint colliderA,
    const uint colliderB
) {
    const uint hash =
        colliderA * 73856093u ^ colliderB * 19349663u;
    float3 result = float3(0.0f);
    const uint axis = hash % 3u;
    const float sign = (hash & 4u) == 0u ? 1.0f : -1.0f;
    result[axis] = sign;
    return result;
}

uint featureKey(const uint shapeType, const uint localFeature) {
    return ((shapeType & 0x0fu) << 28u) |
        (localFeature & 0x0fffffffu);
}

MRRawContactGPU makeContact(
    const float3 normal,
    const float separation,
    const float3 pointA,
    const float3 pointB,
    const uint featureA,
    const uint featureB
) {
    MRRawContactGPU contact{};
    contact.normalAndSeparation =
        float4(normal, separation);
    contact.pointAWorld = float4(pointA, 1.0f);
    contact.pointBWorld = float4(pointB, 1.0f);
    contact.featureAndFlags[0] = featureA;
    contact.featureAndFlags[1] = featureB;
    contact.featureAndFlags[2] = 0u;
    contact.featureAndFlags[3] = 0u;
    return contact;
}

MRRawContactGPU swappedContact(
    const thread MRRawContactGPU& input
) {
    MRRawContactGPU result = input;
    result.normalAndSeparation.xyz =
        -input.normalAndSeparation.xyz;
    result.pointAWorld = input.pointBWorld;
    result.pointBWorld = input.pointAWorld;
    result.featureAndFlags[0] = input.featureAndFlags[1];
    result.featureAndFlags[1] = input.featureAndFlags[0];
    return result;
}

bool generateContact(
    const uint colliderA,
    const uint colliderB,
    const uint pairClass,
    const thread WorldShape& shapeA,
    const thread WorldShape& shapeB,
    thread MRRawContactGPU& contact
) {
    const float contactDistance =
        shapeA.contactOffset + shapeB.contactOffset;
    if (pairClass == kPairSphereSphere) {
        const float3 delta = shapeB.center - shapeA.center;
        const float centerDistance = length(delta);
        const float3 normal = centerDistance > kTiny
            ? delta / centerDistance
            : coincidentNormal(colliderA, colliderB);
        const float separation = centerDistance -
            shapeA.radius - shapeB.radius;
        if (separation > contactDistance) {
            return false;
        }
        contact = makeContact(
            normal,
            separation,
            shapeA.center + normal * shapeA.radius,
            shapeB.center - normal * shapeB.radius,
            featureKey(MR_SHAPE_SPHERE, 0u),
            featureKey(MR_SHAPE_SPHERE, 0u)
        );
        return true;
    }

    const thread WorldShape& plane =
        shapeA.type == MR_SHAPE_PLANE ? shapeA : shapeB;
    const thread WorldShape& sphere =
        shapeA.type == MR_SHAPE_SPHERE ? shapeA : shapeB;
    const float3 surface =
        sphere.center - plane.planeNormal * sphere.radius;
    const float separation =
        dot(plane.planeNormal, surface - plane.center);
    if (separation > contactDistance) {
        return false;
    }
    contact = makeContact(
        -plane.planeNormal,
        separation,
        surface,
        surface - plane.planeNormal * separation,
        featureKey(MR_SHAPE_SPHERE, 0u),
        featureKey(MR_SHAPE_PLANE, 0u)
    );
    if (colliderA == plane.index) {
        contact = swappedContact(contact);
    }
    return true;
}

bool finiteContact(const thread MRRawContactGPU& contact) {
    const float normalLength =
        length(contact.normalAndSeparation.xyz);
    return finiteFloat4(contact.normalAndSeparation) &&
        finiteFloat4(contact.pointAWorld) &&
        finiteFloat4(contact.pointBWorld) &&
        abs(normalLength - 1.0f) <= 2.0e-5f;
}

uint saturatedCount(const ulong count) {
    return count > ulong(0xffffffffu)
        ? 0xffffffffu
        : uint(count);
}

} // namespace

// Correct deterministic baseline: one thread performs canonical O(n^2)
// enumeration and analytic sphere collision. Production broadphase will
// replace this with a parallel LBVH/SAP pipeline without changing the ABI.
kernel void mr_collide_baseline(
    device const MRShapeGPU* shapes [[buffer(0)]],
    device const MRBodyStateGPU* bodies [[buffer(1)]],
    device const MRCandidatePairGPU* exclusions [[buffer(2)]],
    device MRCandidatePairGPU* outputPairs [[buffer(3)]],
    device MRRawContactGPU* outputContacts [[buffer(4)]],
    device uint* outputContactPairIndices [[buffer(5)]],
    device MRSolverStatusGPU* outputStatus [[buffer(6)]],
    constant uint& bodyCount [[buffer(7)]],
    constant uint& shapeCount [[buffer(8)]],
    constant uint& environment [[buffer(9)]],
    constant uint& exclusionCount [[buffer(10)]],
    constant uint& pairCapacity [[buffer(11)]],
    constant uint& contactCapacity [[buffer(12)]],
    uint threadIndex [[thread_position_in_grid]]
) {
    if (threadIndex != 0u) {
        return;
    }

    MRSolverStatusGPU status{};
    status.code = MR_STEP_SUCCESS;

    uint failureCode = MR_STEP_SUCCESS;
    for (uint shape = 0u; shape < shapeCount; ++shape) {
        WorldShape world;
        if (!makeWorldShape(
                shape,
                shapes,
                bodies,
                bodyCount,
                world,
                failureCode
            )) {
            status.code = failureCode;
            outputStatus[0] = status;
            return;
        }
    }

    ulong requiredPairs = 0ul;
    ulong requiredContacts = 0ul;
    for (uint colliderA = 0u;
         colliderA < shapeCount;
         ++colliderA) {
        WorldShape shapeA;
        if (!makeWorldShape(
                colliderA,
                shapes,
                bodies,
                bodyCount,
                shapeA,
                failureCode
            )) {
            status.code = failureCode;
            outputStatus[0] = status;
            return;
        }
        for (uint colliderB = colliderA + 1u;
             colliderB < shapeCount;
             ++colliderB) {
            WorldShape shapeB;
            if (!makeWorldShape(
                    colliderB,
                    shapes,
                    bodies,
                    bodyCount,
                    shapeB,
                    failureCode
                )) {
                status.code = failureCode;
                outputStatus[0] = status;
                return;
            }
            uint pairClass = 0u;
            if (!pairPassesFilter(
                    colliderA,
                    colliderB,
                    shapeA,
                    shapeB,
                    shapes,
                    bodies,
                    exclusions,
                    exclusionCount,
                    pairClass
                )) {
                continue;
            }
            ++requiredPairs;
            MRRawContactGPU contact;
            if (generateContact(
                    colliderA,
                    colliderB,
                    pairClass,
                    shapeA,
                    shapeB,
                    contact
                )) {
                if (!finiteContact(contact)) {
                    status.code = MR_STEP_NONFINITE_RESULT;
                    outputStatus[0] = status;
                    return;
                }
                ++requiredContacts;
            }
        }
    }

    status.requiredPairs = saturatedCount(requiredPairs);
    status.requiredContacts = saturatedCount(requiredContacts);
    status.activeContacts = status.requiredContacts;
    if (requiredPairs > ulong(pairCapacity)) {
        status.code = MR_STEP_PAIR_CAPACITY_OVERFLOW;
        outputStatus[0] = status;
        return;
    }
    if (requiredContacts > ulong(contactCapacity)) {
        status.code = MR_STEP_CONTACT_CAPACITY_OVERFLOW;
        outputStatus[0] = status;
        return;
    }

    uint pairIndex = 0u;
    uint contactIndex = 0u;
    for (uint colliderA = 0u;
         colliderA < shapeCount;
         ++colliderA) {
        WorldShape shapeA;
        if (!makeWorldShape(
                colliderA,
                shapes,
                bodies,
                bodyCount,
                shapeA,
                failureCode
            )) {
            status.code = failureCode;
            outputStatus[0] = status;
            return;
        }
        for (uint colliderB = colliderA + 1u;
             colliderB < shapeCount;
             ++colliderB) {
            WorldShape shapeB;
            if (!makeWorldShape(
                    colliderB,
                    shapes,
                    bodies,
                    bodyCount,
                    shapeB,
                    failureCode
                )) {
                status.code = failureCode;
                outputStatus[0] = status;
                return;
            }
            uint pairClass = 0u;
            if (!pairPassesFilter(
                    colliderA,
                    colliderB,
                    shapeA,
                    shapeB,
                    shapes,
                    bodies,
                    exclusions,
                    exclusionCount,
                    pairClass
                )) {
                continue;
            }

            MRCandidatePairGPU pair{};
            pair.environment = environment;
            pair.colliderA = colliderA;
            pair.colliderB = colliderB;
            pair.flags = pairClass;
            outputPairs[pairIndex] = pair;

            MRRawContactGPU contact;
            if (generateContact(
                    colliderA,
                    colliderB,
                    pairClass,
                    shapeA,
                    shapeB,
                    contact
                )) {
                outputContacts[contactIndex] = contact;
                outputContactPairIndices[contactIndex] = pairIndex;
                ++contactIndex;
            }
            ++pairIndex;
        }
    }

    if (pairIndex != status.requiredPairs ||
        contactIndex != status.requiredContacts) {
        status.code = MR_STEP_NONFINITE_RESULT;
    }
    outputStatus[0] = status;
}
