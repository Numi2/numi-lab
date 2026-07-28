#include <metal_stdlib>

#include "metalrobo/engine_types.h"

using namespace metal;

namespace {

constant float kTiny = 1.0e-14f;
constant uint kPairSphereSphere = 1u;
constant uint kPairSpherePlane = 2u;
constant uint kPairCapsulePlane = 3u;
constant uint kPairBoxPlane = 4u;

struct WorldShape {
    uint index;
    uint type;
    uint body;
    uint disabled;
    float3 center;
    float4 rotation;
    float3 lower;
    float3 upper;
    float3 halfExtents;
    float3 capsuleEndpoint0;
    float3 capsuleEndpoint1;
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

bool canonicalFloat(const float value) {
    const uint bits = as_type<uint>(value);
    const uint exponent = bits & 0x7f800000u;
    const uint mantissa = bits & 0x007fffffu;
    return
        isfinite(value) &&
        (exponent != 0u || mantissa == 0u);
}

bool canonicalFloat4(const float4 value) {
    return
        canonicalFloat(value.x) &&
        canonicalFloat(value.y) &&
        canonicalFloat(value.z) &&
        canonicalFloat(value.w);
}

bool collisionDomain(const float3 value) {
    return
        canonicalFloat(value.x) &&
        canonicalFloat(value.y) &&
        canonicalFloat(value.z) &&
        all(abs(value) <=
            float3(MR_MAX_COLLISION_COORDINATE));
}

bool collisionInputDomain(const float3 value) {
    return
        canonicalFloat(value.x) &&
        canonicalFloat(value.y) &&
        canonicalFloat(value.z) &&
        all(abs(value) <=
            float3(MR_MAX_COLLISION_INPUT_COORDINATE));
}

bool collisionInputDomainXyz(const float4 value) {
    return
        canonicalFloat4(value) &&
        collisionInputDomain(value.xyz);
}

float maximumAbsoluteComponent(const float3 value) {
    return max(max(abs(value.x), abs(value.y)), abs(value.z));
}

void inflateFiniteBounds(thread WorldShape& shape) {
    const float3 scale =
        max(abs(shape.lower), abs(shape.upper)) + 1.0f;
    const float3 padding =
        scale * MR_COLLISION_AABB_RELATIVE_PAD;
    shape.lower -= padding;
    shape.upper += padding;
}

float worldShapeScale(const thread WorldShape& shape) {
    return max(
        max(
            maximumAbsoluteComponent(shape.center),
            maximumAbsoluteComponent(shape.halfExtents)
        ),
        max(
            max(
                maximumAbsoluteComponent(
                    shape.capsuleEndpoint0
                ),
                maximumAbsoluteComponent(
                    shape.capsuleEndpoint1
                )
            ),
            max(abs(shape.radius), abs(shape.contactOffset))
        )
    );
}

float pairQueryPadding(
    const thread WorldShape& left,
    const thread WorldShape& right
) {
    return
        (max(worldShapeScale(left), worldShapeScale(right)) + 1.0f) *
        MR_COLLISION_QUERY_RELATIVE_PAD;
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
    if (!canonicalFloat4(input)) {
        return false;
    }
    const float maximumComponent = max(
        max(abs(input.x), abs(input.y)),
        max(abs(input.z), abs(input.w))
    );
    if (maximumComponent <
            MR_MIN_QUATERNION_MAX_COMPONENT ||
        maximumComponent >
            MR_MAX_QUATERNION_MAX_COMPONENT) {
        return false;
    }
    const float squared = dot(input, input);
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
    if ((typeA == MR_SHAPE_CAPSULE &&
         typeB == MR_SHAPE_PLANE) ||
        (typeA == MR_SHAPE_PLANE &&
         typeB == MR_SHAPE_CAPSULE)) {
        return kPairCapsulePlane;
    }
    if ((typeA == MR_SHAPE_BOX &&
         typeB == MR_SHAPE_PLANE) ||
        (typeA == MR_SHAPE_PLANE &&
         typeB == MR_SHAPE_BOX)) {
        return kPairBoxPlane;
    }
    return 0u;
}

bool validWorldShapeRecord(
    const uint index,
    device const MRShapeGPU* shapes,
    device const MRBodyStateGPU* bodies,
    const uint bodyCount
) {
    const MRShapeGPU shape = shapes[index];
    if (shape.bodyIndex >= bodyCount ||
        (shape.flags & ~MR_SHAPE_FLAG_SIMULATION_DISABLED) != 0u ||
        !collisionInputDomainXyz(shape.localPosition) ||
        !finiteFloat4(shape.localRotation) ||
        !collisionInputDomainXyz(shape.dimensions) ||
        !collisionInputDomainXyz(
            shape.contactRestAndBoundingRadius
        ) ||
        shape.contactRestAndBoundingRadius.x < 0.0f ||
        shape.contactRestAndBoundingRadius.x <
            shape.contactRestAndBoundingRadius.y ||
        shape.contactRestAndBoundingRadius.z < 0.0f) {
        return false;
    }
    float4 bodyRotation;
    float4 localRotation;
    if (!checkedQuaternion(
            bodies[shape.bodyIndex].orientation,
            bodyRotation
        ) ||
        !checkedQuaternion(
            shape.localRotation,
            localRotation
        )) {
        return false;
    }
    const float4 worldRotation =
        quaternionMultiply(bodyRotation, localRotation);
    const float3 worldCenter =
        bodies[shape.bodyIndex].position.xyz +
        quaternionRotate(
            bodyRotation,
            shape.localPosition.xyz
        );
    return
        finiteFloat4(worldRotation) &&
        collisionDomain(worldCenter);
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
        (shape.flags & ~MR_SHAPE_FLAG_SIMULATION_DISABLED) != 0u ||
        !collisionInputDomainXyz(shape.localPosition) ||
        !finiteFloat4(shape.localRotation) ||
        !collisionInputDomainXyz(shape.dimensions) ||
        !collisionInputDomainXyz(
            shape.contactRestAndBoundingRadius
        ) ||
        shape.contactRestAndBoundingRadius.x < 0.0f ||
        shape.contactRestAndBoundingRadius.x <
            shape.contactRestAndBoundingRadius.y ||
        shape.contactRestAndBoundingRadius.z < 0.0f) {
        failureCode = MR_STEP_NONFINITE_INPUT;
        return false;
    }

    const MRBodyStateGPU body = bodies[shape.bodyIndex];
    if (!collisionInputDomainXyz(body.position) ||
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
    output.halfExtents = float3(0.0f);
    output.capsuleEndpoint0 = float3(0.0f);
    output.capsuleEndpoint1 = float3(0.0f);
    output.planeNormal = float3(0.0f);

    if (!collisionDomain(output.center) ||
        !finiteFloat4(output.rotation)) {
        failureCode = MR_STEP_NONFINITE_INPUT;
        return false;
    }
    if (output.disabled != 0u) {
        return true;
    }
    if (shape.shapeType != MR_SHAPE_SPHERE &&
        shape.shapeType != MR_SHAPE_CAPSULE &&
        shape.shapeType != MR_SHAPE_BOX &&
        shape.shapeType != MR_SHAPE_PLANE) {
        failureCode = MR_STEP_UNSUPPORTED;
        return false;
    }

    if (shape.shapeType == MR_SHAPE_SPHERE) {
        output.radius = shape.dimensions.x;
        if (output.radius < MR_MIN_COLLISION_EXTENT) {
            failureCode = MR_STEP_NONFINITE_INPUT;
            return false;
        }
        const float expansion =
            output.radius + output.contactOffset;
        output.lower = output.center - expansion;
        output.upper = output.center + expansion;
        inflateFiniteBounds(output);
        if (!collisionDomain(output.lower) ||
            !collisionDomain(output.upper)) {
            failureCode = MR_STEP_NONFINITE_INPUT;
            return false;
        }
        return true;
    }

    if (shape.shapeType == MR_SHAPE_CAPSULE) {
        output.radius = shape.dimensions.x;
        const float halfLength = shape.dimensions.y;
        if (output.radius < MR_MIN_COLLISION_EXTENT ||
            halfLength < MR_MIN_COLLISION_EXTENT) {
            failureCode = MR_STEP_NONFINITE_INPUT;
            return false;
        }
        const float3 axis = quaternionRotate(
            output.rotation,
            float3(0.0f, halfLength, 0.0f)
        );
        output.capsuleEndpoint0 = output.center - axis;
        output.capsuleEndpoint1 = output.center + axis;
        const float expansion =
            output.radius + output.contactOffset;
        output.lower = min(
            output.capsuleEndpoint0,
            output.capsuleEndpoint1
        ) - expansion;
        output.upper = max(
            output.capsuleEndpoint0,
            output.capsuleEndpoint1
        ) + expansion;
        inflateFiniteBounds(output);
        if (!collisionDomain(output.capsuleEndpoint0) ||
            !collisionDomain(output.capsuleEndpoint1) ||
            !collisionDomain(output.lower) ||
            !collisionDomain(output.upper)) {
            failureCode = MR_STEP_NONFINITE_INPUT;
            return false;
        }
        return true;
    }

    if (shape.shapeType == MR_SHAPE_BOX) {
        output.halfExtents = shape.dimensions.xyz;
        if (!all(
                output.halfExtents >=
                    float3(MR_MIN_COLLISION_EXTENT)
            )) {
            failureCode = MR_STEP_NONFINITE_INPUT;
            return false;
        }
        const float3 basisX = quaternionRotate(
            output.rotation,
            float3(1.0f, 0.0f, 0.0f)
        );
        const float3 basisY = quaternionRotate(
            output.rotation,
            float3(0.0f, 1.0f, 0.0f)
        );
        const float3 basisZ = quaternionRotate(
            output.rotation,
            float3(0.0f, 0.0f, 1.0f)
        );
        const float3 extent =
            abs(basisX) * output.halfExtents.x +
            abs(basisY) * output.halfExtents.y +
            abs(basisZ) * output.halfExtents.z +
            output.contactOffset;
        output.lower = output.center - extent;
        output.upper = output.center + extent;
        inflateFiniteBounds(output);
        if (!collisionDomain(output.lower) ||
            !collisionDomain(output.upper)) {
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
    const thread WorldShape& finiteShape
) {
    const float3 center =
        0.5f * (finiteShape.lower + finiteShape.upper);
    const float3 halfExtent =
        0.5f * (finiteShape.upper - finiteShape.lower);
    const float minimumSignedDistance =
        dot(plane.planeNormal, center - plane.center) -
        dot(abs(plane.planeNormal), halfExtent);
    return minimumSignedDistance <=
        plane.contactOffset +
        pairQueryPadding(plane, finiteShape);
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
    const thread WorldShape& finiteShape =
        shapeA.type == MR_SHAPE_PLANE ? shapeB : shapeA;
    return planeMayOverlap(plane, finiteShape);
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

struct ContactBatch {
    MRRawContactGPU contacts[8];
    uint count;
};

void appendFinitePlaneContact(
    thread ContactBatch& result,
    const uint colliderA,
    const thread WorldShape& plane,
    const float3 finiteSurfacePoint,
    const float separation,
    const uint finiteFeature
) {
    const float3 planePoint =
        finiteSurfacePoint - plane.planeNormal * separation;
    MRRawContactGPU contact = makeContact(
        -plane.planeNormal,
        separation,
        finiteSurfacePoint,
        planePoint,
        finiteFeature,
        featureKey(MR_SHAPE_PLANE, 0u)
    );
    if (colliderA == plane.index) {
        contact = swappedContact(contact);
    }
    result.contacts[result.count++] = contact;
}

ContactBatch generateContacts(
    const uint colliderA,
    const uint colliderB,
    const uint pairClass,
    const thread WorldShape& shapeA,
    const thread WorldShape& shapeB
) {
    ContactBatch result{};
    const float contactDistance =
        shapeA.contactOffset + shapeB.contactOffset;
    const float acceptedContactDistance =
        contactDistance + pairQueryPadding(shapeA, shapeB);
    if (pairClass == kPairSphereSphere) {
        const float3 delta = shapeB.center - shapeA.center;
        const float centerDistance = length(delta);
        const float3 normal = centerDistance > kTiny
            ? delta / centerDistance
            : coincidentNormal(colliderA, colliderB);
        const float separation = centerDistance -
            shapeA.radius - shapeB.radius;
        if (separation > acceptedContactDistance) {
            return result;
        }
        result.contacts[0] = makeContact(
            normal,
            separation,
            shapeA.center + normal * shapeA.radius,
            shapeB.center - normal * shapeB.radius,
            featureKey(MR_SHAPE_SPHERE, 0u),
            featureKey(MR_SHAPE_SPHERE, 0u)
        );
        result.count = 1u;
        return result;
    }

    const thread WorldShape& plane =
        shapeA.type == MR_SHAPE_PLANE ? shapeA : shapeB;
    const thread WorldShape& finiteShape =
        shapeA.type == MR_SHAPE_PLANE ? shapeB : shapeA;

    if (pairClass == kPairSpherePlane) {
        const float3 surface =
            finiteShape.center -
            plane.planeNormal * finiteShape.radius;
        const float separation =
            dot(plane.planeNormal, surface - plane.center);
        if (separation <= acceptedContactDistance) {
            appendFinitePlaneContact(
                result,
                colliderA,
                plane,
                surface,
                separation,
                featureKey(MR_SHAPE_SPHERE, 0u)
            );
        }
        return result;
    }

    if (pairClass == kPairCapsulePlane) {
        for (uint endpoint = 0u; endpoint < 2u; ++endpoint) {
            const float3 capsuleEndpoint = endpoint == 0u
                ? finiteShape.capsuleEndpoint0
                : finiteShape.capsuleEndpoint1;
            const float3 surface =
                capsuleEndpoint -
                plane.planeNormal * finiteShape.radius;
            const float separation =
                dot(plane.planeNormal, surface - plane.center);
            if (separation <= acceptedContactDistance) {
                appendFinitePlaneContact(
                    result,
                    colliderA,
                    plane,
                    surface,
                    separation,
                    featureKey(MR_SHAPE_CAPSULE, endpoint)
                );
            }
        }
        return result;
    }

    for (uint boxVertex = 0u;
         boxVertex < 8u;
         ++boxVertex) {
        const float3 local = float3(
            (boxVertex & 1u) != 0u
                ? finiteShape.halfExtents.x
                : -finiteShape.halfExtents.x,
            (boxVertex & 2u) != 0u
                ? finiteShape.halfExtents.y
                : -finiteShape.halfExtents.y,
            (boxVertex & 4u) != 0u
                ? finiteShape.halfExtents.z
                : -finiteShape.halfExtents.z
        );
        const float3 world =
            finiteShape.center +
            quaternionRotate(finiteShape.rotation, local);
        const float separation =
            dot(plane.planeNormal, world - plane.center);
        if (separation <= acceptedContactDistance) {
            appendFinitePlaneContact(
                result,
                colliderA,
                plane,
                world,
                separation,
                featureKey(MR_SHAPE_BOX, boxVertex)
            );
        }
    }
    return result;
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

    for (uint bodyIndex = 0u;
         bodyIndex < bodyCount;
         ++bodyIndex) {
        const MRBodyStateGPU body = bodies[bodyIndex];
        float4 bodyRotation;
        if (body.flagsAndIndices[0] > MR_MOTION_DYNAMIC ||
            !collisionInputDomainXyz(body.position) ||
            !checkedQuaternion(
                body.orientation,
                bodyRotation
            )) {
            status.code = MR_STEP_NONFINITE_INPUT;
            outputStatus[0] = status;
            return;
        }
    }

    for (uint exclusionIndex = 0u;
         exclusionIndex < exclusionCount;
         ++exclusionIndex) {
        const MRCandidatePairGPU exclusion =
            exclusions[exclusionIndex];
        if (exclusion.colliderA >= shapeCount ||
            exclusion.colliderB >= shapeCount) {
            status.code = MR_STEP_NONFINITE_INPUT;
            outputStatus[0] = status;
            return;
        }
    }
    // Canonical error precedence: validate every common record and derived
    // FP32 transform before any active shape is type-classified.
    for (uint shape = 0u; shape < shapeCount; ++shape) {
        if (!validWorldShapeRecord(
                shape,
                shapes,
                bodies,
                bodyCount
            )) {
            status.code = MR_STEP_NONFINITE_INPUT;
            outputStatus[0] = status;
            return;
        }
    }

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
            const ContactBatch batch = generateContacts(
                colliderA,
                colliderB,
                pairClass,
                shapeA,
                shapeB
            );
            for (uint batchIndex = 0u;
                 batchIndex < batch.count;
                 ++batchIndex) {
                if (!finiteContact(batch.contacts[batchIndex])) {
                    status.code = MR_STEP_NONFINITE_RESULT;
                    outputStatus[0] = status;
                    return;
                }
            }
            if (requiredContacts >
                ~ulong(0) - ulong(batch.count)) {
                requiredContacts = ~ulong(0);
            } else {
                requiredContacts += ulong(batch.count);
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

            const ContactBatch batch = generateContacts(
                colliderA,
                colliderB,
                pairClass,
                shapeA,
                shapeB
            );
            for (uint batchIndex = 0u;
                 batchIndex < batch.count;
                 ++batchIndex) {
                outputContacts[contactIndex] =
                    batch.contacts[batchIndex];
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
