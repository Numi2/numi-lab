#include <metal_stdlib>

#include "metalrobo/engine_types.h"

using namespace metal;

namespace {

constant float kTiny = 1.0e-14f;
constant uint kPairSphereSphere = MR_COLLISION_PAIR_SPHERE_SPHERE;
constant uint kPairSpherePlane = MR_COLLISION_PAIR_SPHERE_PLANE;
constant uint kPairCapsulePlane = MR_COLLISION_PAIR_CAPSULE_PLANE;
constant uint kPairBoxPlane = MR_COLLISION_PAIR_BOX_PLANE;
constant uint kPairCylinderPlane = MR_COLLISION_PAIR_CYLINDER_PLANE;
constant uint kPairSphereCapsule = MR_COLLISION_PAIR_SPHERE_CAPSULE;
constant uint kPairCapsuleCapsule = MR_COLLISION_PAIR_CAPSULE_CAPSULE;
constant uint kPairSphereBox = MR_COLLISION_PAIR_SPHERE_BOX;
constant uint kPairCapsuleBox = MR_COLLISION_PAIR_CAPSULE_BOX;
constant uint kPairBoxBox = MR_COLLISION_PAIR_BOX_BOX;
constant float kCylinderAlignmentTolerance = 1.0e-6f;
constant uint kCylinderNegativeCapRingBase = 0u;
constant uint kCylinderPositiveCapRingBase = 4u;
constant uint kCylinderNegativeSideRim = 8u;
constant uint kCylinderPositiveSideRim = 9u;
constant uint kCylinderNegativeGeneralRim = 10u;
constant uint kCylinderPositiveGeneralRim = 11u;

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
    float3 cylinderAxis;
    float3 cylinderBasisX;
    float3 cylinderBasisZ;
    float3 planeNormal;
    float radius;
    float halfLength;
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
            max(
                max(abs(shape.radius), abs(shape.halfLength)),
                abs(shape.contactOffset)
            )
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

float3 quaternionInverseRotate(
    const float4 quaternion,
    const float3 value
) {
    return quaternionRotate(
        float4(-quaternion.xyz, quaternion.w),
        value
    );
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
    if ((typeA == MR_SHAPE_CYLINDER &&
         typeB == MR_SHAPE_PLANE) ||
        (typeA == MR_SHAPE_PLANE &&
         typeB == MR_SHAPE_CYLINDER)) {
        return kPairCylinderPlane;
    }
    if ((typeA == MR_SHAPE_SPHERE &&
         typeB == MR_SHAPE_CAPSULE) ||
        (typeA == MR_SHAPE_CAPSULE &&
         typeB == MR_SHAPE_SPHERE)) {
        return kPairSphereCapsule;
    }
    if (typeA == MR_SHAPE_CAPSULE &&
        typeB == MR_SHAPE_CAPSULE) {
        return kPairCapsuleCapsule;
    }
    if ((typeA == MR_SHAPE_SPHERE &&
         typeB == MR_SHAPE_BOX) ||
        (typeA == MR_SHAPE_BOX &&
         typeB == MR_SHAPE_SPHERE)) {
        return kPairSphereBox;
    }
    if ((typeA == MR_SHAPE_CAPSULE &&
         typeB == MR_SHAPE_BOX) ||
        (typeA == MR_SHAPE_BOX &&
         typeB == MR_SHAPE_CAPSULE)) {
        return kPairCapsuleBox;
    }
    if (typeA == MR_SHAPE_BOX &&
        typeB == MR_SHAPE_BOX) {
        return kPairBoxBox;
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
    output.cylinderAxis = float3(0.0f);
    output.cylinderBasisX = float3(0.0f);
    output.cylinderBasisZ = float3(0.0f);
    output.planeNormal = float3(0.0f);
    output.halfLength = 0.0f;

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
        shape.shapeType != MR_SHAPE_CYLINDER &&
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

    if (shape.shapeType == MR_SHAPE_CYLINDER) {
        output.radius = shape.dimensions.x;
        output.halfLength = shape.dimensions.y;
        if (output.radius < MR_MIN_COLLISION_EXTENT ||
            output.halfLength < MR_MIN_COLLISION_EXTENT) {
            failureCode = MR_STEP_NONFINITE_INPUT;
            return false;
        }

        output.cylinderAxis = quaternionRotate(
            output.rotation,
            float3(0.0f, 1.0f, 0.0f)
        );
        const float axisSquared =
            dot(output.cylinderAxis, output.cylinderAxis);
        if (!(axisSquared > kTiny) || !isfinite(axisSquared)) {
            failureCode = MR_STEP_NONFINITE_INPUT;
            return false;
        }
        output.cylinderAxis *= rsqrt(axisSquared);

        output.cylinderBasisX = quaternionRotate(
            output.rotation,
            float3(1.0f, 0.0f, 0.0f)
        );
        output.cylinderBasisX -=
            output.cylinderAxis *
            dot(output.cylinderAxis, output.cylinderBasisX);
        const float basisXSquared = dot(
            output.cylinderBasisX,
            output.cylinderBasisX
        );
        if (!(basisXSquared > kTiny) ||
            !isfinite(basisXSquared)) {
            failureCode = MR_STEP_NONFINITE_INPUT;
            return false;
        }
        output.cylinderBasisX *= rsqrt(basisXSquared);

        output.cylinderBasisZ = cross(
            output.cylinderBasisX,
            output.cylinderAxis
        );
        const float basisZSquared = dot(
            output.cylinderBasisZ,
            output.cylinderBasisZ
        );
        if (!(basisZSquared > kTiny) ||
            !isfinite(basisZSquared)) {
            failureCode = MR_STEP_NONFINITE_INPUT;
            return false;
        }
        output.cylinderBasisZ *= rsqrt(basisZSquared);

        const float3 radialExtent = sqrt(max(
            float3(0.0f),
            float3(1.0f) -
                output.cylinderAxis * output.cylinderAxis
        )) * output.radius;
        const float3 extent =
            abs(output.cylinderAxis) * output.halfLength +
            radialExtent +
            output.contactOffset;
        output.lower = output.center - extent;
        output.upper = output.center + extent;
        inflateFiniteBounds(output);
        if (!finiteFloat3(output.cylinderAxis) ||
            !finiteFloat3(output.cylinderBasisX) ||
            !finiteFloat3(output.cylinderBasisZ) ||
            !collisionDomain(output.lower) ||
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

MRProjectedColliderGPU projectedCollider(
    const thread WorldShape& shape,
    const uint status
) {
    MRProjectedColliderGPU projected = {};
    projected.statusAndFlags = uint4(
        status,
        shape.disabled,
        0u,
        0u
    );
    projected.centerAndRadius =
        float4(shape.center, shape.radius);
    projected.rotation = shape.rotation;
    projected.lowerAndHalfLength =
        float4(shape.lower, shape.halfLength);
    projected.upperAndContactOffset =
        float4(shape.upper, shape.contactOffset);
    return projected;
}

bool loadProjectedCollider(
    const uint collider,
    device const MRShapeGPU& source,
    device const MRProjectedColliderGPU& projected,
    thread WorldShape& shape,
    thread uint& failureCode
) {
    failureCode = projected.statusAndFlags.x;
    if (failureCode != MR_STEP_SUCCESS) {
        return false;
    }
    shape.index = collider;
    shape.type = source.shapeType;
    shape.body = source.bodyIndex;
    shape.disabled = projected.statusAndFlags.y;
    shape.center = projected.centerAndRadius.xyz;
    shape.radius = projected.centerAndRadius.w;
    shape.rotation = projected.rotation;
    shape.lower = projected.lowerAndHalfLength.xyz;
    shape.halfLength = projected.lowerAndHalfLength.w;
    shape.upper = projected.upperAndContactOffset.xyz;
    shape.contactOffset = projected.upperAndContactOffset.w;
    shape.halfExtents = float3(0.0f);
    shape.capsuleEndpoint0 = float3(0.0f);
    shape.capsuleEndpoint1 = float3(0.0f);
    shape.cylinderAxis = float3(0.0f);
    shape.cylinderBasisX = float3(0.0f);
    shape.cylinderBasisZ = float3(0.0f);
    shape.planeNormal = float3(0.0f);
    if (shape.type == MR_SHAPE_CAPSULE) {
        const float3 axis = quaternionRotate(
            shape.rotation,
            float3(0.0f, shape.halfLength, 0.0f)
        );
        shape.capsuleEndpoint0 = shape.center - axis;
        shape.capsuleEndpoint1 = shape.center + axis;
    } else if (shape.type == MR_SHAPE_BOX) {
        shape.halfExtents = source.dimensions.xyz;
    } else if (shape.type == MR_SHAPE_CYLINDER) {
        shape.cylinderAxis = quaternionRotate(
            shape.rotation,
            float3(0.0f, 1.0f, 0.0f)
        );
        shape.cylinderAxis *= rsqrt(
            dot(shape.cylinderAxis, shape.cylinderAxis)
        );
        shape.cylinderBasisX = quaternionRotate(
            shape.rotation,
            float3(1.0f, 0.0f, 0.0f)
        );
        shape.cylinderBasisX -=
            shape.cylinderAxis *
            dot(shape.cylinderAxis, shape.cylinderBasisX);
        shape.cylinderBasisX *= rsqrt(
            dot(shape.cylinderBasisX, shape.cylinderBasisX)
        );
        shape.cylinderBasisZ = cross(
            shape.cylinderBasisX,
            shape.cylinderAxis
        );
        shape.cylinderBasisZ *= rsqrt(
            dot(shape.cylinderBasisZ, shape.cylinderBasisZ)
        );
    } else if (shape.type == MR_SHAPE_PLANE) {
        shape.planeNormal = quaternionRotate(
            shape.rotation,
            float3(0.0f, 1.0f, 0.0f)
        );
        shape.planeNormal *= rsqrt(
            dot(shape.planeNormal, shape.planeNormal)
        );
    }
    return true;
}

bool projectedPairMayOverlap(
    const uint typeA,
    const uint typeB,
    device const MRProjectedColliderGPU& projectedA,
    device const MRProjectedColliderGPU& projectedB
) {
    if (projectedA.statusAndFlags.y != 0u ||
        projectedB.statusAndFlags.y != 0u) {
        return false;
    }
    if (typeA != MR_SHAPE_PLANE &&
        typeB != MR_SHAPE_PLANE) {
        return
            all(
                projectedA.lowerAndHalfLength.xyz <=
                    projectedB.upperAndContactOffset.xyz
            ) &&
            all(
                projectedA.upperAndContactOffset.xyz >=
                    projectedB.lowerAndHalfLength.xyz
            );
    }
    device const MRProjectedColliderGPU& plane =
        typeA == MR_SHAPE_PLANE ? projectedA : projectedB;
    device const MRProjectedColliderGPU& finiteShape =
        typeA == MR_SHAPE_PLANE ? projectedB : projectedA;
    float3 normal = quaternionRotate(
        plane.rotation,
        float3(0.0f, 1.0f, 0.0f)
    );
    normal *= rsqrt(dot(normal, normal));
    const float3 lower =
        finiteShape.lowerAndHalfLength.xyz;
    const float3 upper =
        finiteShape.upperAndContactOffset.xyz;
    const float3 center = 0.5f * (lower + upper);
    const float3 halfExtent = 0.5f * (upper - lower);
    const float scale = max(
        max(
            maximumAbsoluteComponent(
                plane.centerAndRadius.xyz
            ),
            maximumAbsoluteComponent(lower)
        ),
        max(
            maximumAbsoluteComponent(upper),
            max(
                abs(plane.upperAndContactOffset.w),
                abs(finiteShape.upperAndContactOffset.w)
            )
        )
    );
    const float queryPadding =
        (scale + 1.0f) * MR_COLLISION_QUERY_RELATIVE_PAD;
    const float minimumSignedDistance =
        dot(
            normal,
            center - plane.centerAndRadius.xyz
        ) -
        dot(abs(normal), halfExtent);
    return minimumSignedDistance <=
        plane.upperAndContactOffset.w + queryPadding;
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
    if (shapeA.type != MR_SHAPE_PLANE &&
        shapeB.type != MR_SHAPE_PLANE) {
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

float3 stableSegmentNormal(
    const float3 directionA,
    const float3 directionB,
    const uint colliderA,
    const uint colliderB
) {
    const float3 crossed = cross(directionA, directionB);
    if (dot(crossed, crossed) > kTiny) {
        return normalize(crossed);
    }

    float3 direction = float3(0.0f);
    if (dot(directionA, directionA) > kTiny) {
        direction = normalize(directionA);
    } else if (dot(directionB, directionB) > kTiny) {
        direction = normalize(directionB);
    } else {
        return coincidentNormal(colliderA, colliderB);
    }

    const float3 absoluteDirection = abs(direction);
    float3 reference;
    if (absoluteDirection.x <= absoluteDirection.y &&
        absoluteDirection.x <= absoluteDirection.z) {
        reference = float3(1.0f, 0.0f, 0.0f);
    } else if (absoluteDirection.y <= absoluteDirection.z) {
        reference = float3(0.0f, 1.0f, 0.0f);
    } else {
        reference = float3(0.0f, 0.0f, 1.0f);
    }
    float3 result = normalize(cross(direction, reference));
    if (((colliderA * 73856093u) ^
         (colliderB * 19349663u)) & 4u) {
        result = -result;
    }
    return result;
}

struct SegmentClosestPoint {
    float3 point;
    float parameter;
};

SegmentClosestPoint closestPointOnSegment(
    const float3 point,
    const float3 endpoint0,
    const float3 endpoint1
) {
    const float3 segment = endpoint1 - endpoint0;
    const float squaredLength = dot(segment, segment);
    if (!(squaredLength > kTiny)) {
        return {endpoint0, 0.0f};
    }
    const float parameter = clamp(
        dot(point - endpoint0, segment) / squaredLength,
        0.0f,
        1.0f
    );
    return {
        endpoint0 + segment * parameter,
        parameter,
    };
}

struct SegmentPairClosestPoints {
    float3 pointA;
    float3 pointB;
    float parameterA;
    float parameterB;
};

SegmentPairClosestPoints closestPointsOnSegments(
    const float3 endpointA0,
    const float3 endpointA1,
    const float3 endpointB0,
    const float3 endpointB1
) {
    const float3 directionA = endpointA1 - endpointA0;
    const float3 directionB = endpointB1 - endpointB0;
    const float3 offset = endpointA0 - endpointB0;
    const float squaredA = dot(directionA, directionA);
    const float squaredB = dot(directionB, directionB);
    const float projectedA = dot(directionA, offset);
    const float projectedB = dot(directionB, offset);

    float parameterA = 0.0f;
    float parameterB = 0.0f;
    if (!(squaredA > kTiny) && !(squaredB > kTiny)) {
        return {endpointA0, endpointB0, 0.0f, 0.0f};
    }
    if (!(squaredA > kTiny)) {
        parameterB = clamp(
            projectedB / squaredB,
            0.0f,
            1.0f
        );
    } else {
        const float crossProjection =
            dot(directionA, directionB);
        if (!(squaredB > kTiny)) {
            parameterA = clamp(
                -projectedA / squaredA,
                0.0f,
                1.0f
            );
        } else {
            const float denominator =
                squaredA * squaredB -
                crossProjection * crossProjection;
            if (denominator > kTiny * squaredA * squaredB) {
                parameterA = clamp(
                    (
                        crossProjection * projectedB -
                        projectedA * squaredB
                    ) /
                        denominator,
                    0.0f,
                    1.0f
                );
            }
            parameterB =
                (
                    crossProjection * parameterA +
                    projectedB
                ) /
                squaredB;
            if (parameterB < 0.0f) {
                parameterB = 0.0f;
                parameterA = clamp(
                    -projectedA / squaredA,
                    0.0f,
                    1.0f
                );
            } else if (parameterB > 1.0f) {
                parameterB = 1.0f;
                parameterA = clamp(
                    (
                        crossProjection - projectedA
                    ) /
                        squaredA,
                    0.0f,
                    1.0f
                );
            }
        }
    }
    return {
        endpointA0 + directionA * parameterA,
        endpointB0 + directionB * parameterB,
        parameterA,
        parameterB,
    };
}

uint capsuleFeature(const float parameter) {
    if (parameter <= 0.0f) {
        return 0u;
    }
    return parameter >= 1.0f ? 1u : 2u;
}

struct SphereBoxWitness {
    float3 normal;
    float3 boxPoint;
    float separation;
    uint boxFeature;
};

struct CapsuleBoxWitness {
    float3 normal;
    float3 capsulePoint;
    float3 boxPoint;
    float separation;
    float capsuleParameter;
    uint boxFeature;
};

uint boxRegion(
    const float coordinate,
    const float halfExtent
) {
    if (coordinate < -halfExtent) {
        return 0u;
    }
    return coordinate > halfExtent ? 2u : 1u;
}

SphereBoxWitness sphereBoxWitness(
    const thread WorldShape& sphere,
    const thread WorldShape& box
) {
    const float3 localCenter = quaternionInverseRotate(
        box.rotation,
        sphere.center - box.center
    );
    float3 closest = clamp(
        localCenter,
        -box.halfExtents,
        box.halfExtents
    );
    const float3 localDelta = closest - localCenter;
    const float distance = length(localDelta);

    SphereBoxWitness result{};
    if (distance > kTiny) {
        const float3 localNormal = localDelta / distance;
        result.normal =
            quaternionRotate(box.rotation, localNormal);
        result.separation = distance - sphere.radius;
        result.boxFeature =
            boxRegion(localCenter.x, box.halfExtents.x) +
            3u * boxRegion(
                localCenter.y,
                box.halfExtents.y
            ) +
            9u * boxRegion(
                localCenter.z,
                box.halfExtents.z
            );
    } else {
        const float3 distances =
            box.halfExtents - abs(localCenter);
        uint axis = 0u;
        if (distances.y < distances[axis]) {
            axis = 1u;
        }
        if (distances.z < distances[axis]) {
            axis = 2u;
        }
        const float coordinate = localCenter[axis];
        const float sign = coordinate >= 0.0f ? 1.0f : -1.0f;
        float3 outward = float3(0.0f);
        outward[axis] = sign;
        closest[axis] = sign * box.halfExtents[axis];
        result.normal =
            quaternionRotate(box.rotation, -outward);
        result.separation =
            -distances[axis] - sphere.radius;
        result.boxFeature =
            27u + 2u * axis + (sign > 0.0f ? 1u : 0u);
    }
    result.boxPoint =
        box.center + quaternionRotate(box.rotation, closest);
    return result;
}

void considerCapsuleBoxCandidate(
    const float3 segmentPoint,
    const float3 boxPoint,
    const float parameter,
    const uint feature,
    thread float& bestSquared,
    thread float3& bestSegmentPoint,
    thread float3& bestBoxPoint,
    thread float& bestParameter,
    thread uint& bestFeature
) {
    const float squared = dot(
        boxPoint - segmentPoint,
        boxPoint - segmentPoint
    );
    const float tolerance =
        1.0e-12f *
        (1.0f + max(bestSquared, squared));
    if (!isfinite(bestSquared) ||
        squared < bestSquared - tolerance ||
        (abs(squared - bestSquared) <= tolerance &&
         feature < bestFeature)) {
        bestSquared = squared;
        bestSegmentPoint = segmentPoint;
        bestBoxPoint = boxPoint;
        bestParameter = parameter;
        bestFeature = feature;
    }
}

CapsuleBoxWitness capsuleBoxWitness(
    const thread WorldShape& capsule,
    const thread WorldShape& box
) {
    const float3 segment0 = quaternionInverseRotate(
        box.rotation,
        capsule.capsuleEndpoint0 - box.center
    );
    const float3 segment1 = quaternionInverseRotate(
        box.rotation,
        capsule.capsuleEndpoint1 - box.center
    );
    const float3 direction = segment1 - segment0;

    float enter = 0.0f;
    float exit = 1.0f;
    bool intersects = true;
    for (uint axis = 0u; axis < 3u; ++axis) {
        if (abs(direction[axis]) <= kTiny) {
            if (segment0[axis] < -box.halfExtents[axis] ||
                segment0[axis] > box.halfExtents[axis]) {
                intersects = false;
            }
            continue;
        }
        float first =
            (-box.halfExtents[axis] - segment0[axis]) /
            direction[axis];
        float second =
            (box.halfExtents[axis] - segment0[axis]) /
            direction[axis];
        if (first > second) {
            const float temporary = first;
            first = second;
            second = temporary;
        }
        enter = max(enter, first);
        exit = min(exit, second);
        if (enter > exit) {
            intersects = false;
        }
    }

    CapsuleBoxWitness result{};
    if (intersects) {
        result.capsuleParameter =
            clamp(0.5f * (enter + exit), 0.0f, 1.0f);
        const float3 core =
            segment0 + direction * result.capsuleParameter;
        const float3 distances =
            box.halfExtents - abs(core);
        uint axis = 0u;
        if (distances.y < distances.x) {
            axis = 1u;
        }
        if (distances.z < distances[axis]) {
            axis = 2u;
        }
        const float sign = core[axis] >= 0.0f ? 1.0f : -1.0f;
        float3 outward = float3(0.0f);
        outward[axis] = sign;
        float3 boxPoint = core;
        boxPoint[axis] = sign * box.halfExtents[axis];
        const float3 localNormal = -outward;
        result.normal =
            quaternionRotate(box.rotation, localNormal);
        result.capsulePoint =
            box.center +
            quaternionRotate(
                box.rotation,
                core + localNormal * capsule.radius
            );
        result.boxPoint =
            box.center +
            quaternionRotate(box.rotation, boxPoint);
        result.separation =
            -distances[axis] - capsule.radius;
        result.boxFeature =
            27u + 2u * axis + (sign > 0.0f ? 1u : 0u);
        return result;
    }

    float bestSquared = INFINITY;
    float3 bestSegmentPoint = float3(0.0f);
    float3 bestBoxPoint = float3(0.0f);
    float bestParameter = 0.0f;
    uint bestFeature = 0xffffffffu;
    for (uint endpoint = 0u; endpoint < 2u; ++endpoint) {
        const float3 segmentPoint =
            endpoint == 0u ? segment0 : segment1;
        const float3 boxPoint = clamp(
            segmentPoint,
            -box.halfExtents,
            box.halfExtents
        );
        considerCapsuleBoxCandidate(
            segmentPoint,
            boxPoint,
            float(endpoint),
            endpoint,
            bestSquared,
            bestSegmentPoint,
            bestBoxPoint,
            bestParameter,
            bestFeature
        );
    }
    uint edgeFeature = 2u;
    for (uint varyingAxis = 0u;
         varyingAxis < 3u;
         ++varyingAxis) {
        const uint fixedAxis0 = (varyingAxis + 1u) % 3u;
        const uint fixedAxis1 = (varyingAxis + 2u) % 3u;
        for (uint signs = 0u; signs < 4u; ++signs) {
            float3 edge0 = float3(0.0f);
            float3 edge1 = float3(0.0f);
            edge0[varyingAxis] =
                -box.halfExtents[varyingAxis];
            edge1[varyingAxis] =
                box.halfExtents[varyingAxis];
            const float sign0 =
                (signs & 1u) == 0u ? -1.0f : 1.0f;
            const float sign1 =
                (signs & 2u) == 0u ? -1.0f : 1.0f;
            edge0[fixedAxis0] =
                sign0 * box.halfExtents[fixedAxis0];
            edge1[fixedAxis0] = edge0[fixedAxis0];
            edge0[fixedAxis1] =
                sign1 * box.halfExtents[fixedAxis1];
            edge1[fixedAxis1] = edge0[fixedAxis1];
            const SegmentPairClosestPoints closest =
                closestPointsOnSegments(
                    segment0,
                    segment1,
                    edge0,
                    edge1
                );
            considerCapsuleBoxCandidate(
                closest.pointA,
                closest.pointB,
                closest.parameterA,
                edgeFeature,
                bestSquared,
                bestSegmentPoint,
                bestBoxPoint,
                bestParameter,
                bestFeature
            );
            ++edgeFeature;
        }
    }
    const float distance = sqrt(max(bestSquared, 0.0f));
    const float3 localNormal =
        distance > kTiny
        ? (bestBoxPoint - bestSegmentPoint) / distance
        : quaternionInverseRotate(
              box.rotation,
              coincidentNormal(capsule.index, box.index)
          );
    result.normal =
        quaternionRotate(box.rotation, localNormal);
    result.capsulePoint =
        box.center +
        quaternionRotate(
            box.rotation,
            bestSegmentPoint +
                localNormal * capsule.radius
        );
    result.boxPoint =
        box.center +
        quaternionRotate(box.rotation, bestBoxPoint);
    result.separation = distance - capsule.radius;
    result.capsuleParameter = bestParameter;
    result.boxFeature = 64u + bestFeature;
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

float3 boxAxis(
    const thread WorldShape& box,
    const uint axis
) {
    float3 local = float3(0.0f);
    local[axis] = 1.0f;
    return quaternionRotate(box.rotation, local);
}

float boxProjectionRadius(
    const thread WorldShape& box,
    const float3 axis
) {
    return
        abs(dot(axis, boxAxis(box, 0u))) *
            box.halfExtents.x +
        abs(dot(axis, boxAxis(box, 1u))) *
            box.halfExtents.y +
        abs(dot(axis, boxAxis(box, 2u))) *
            box.halfExtents.z;
}

float3 boxVertex(
    const thread WorldShape& box,
    const uint vertexIndex
) {
    const float3 local = float3(
        (vertexIndex & 1u) == 0u
            ? -box.halfExtents.x
            : box.halfExtents.x,
        (vertexIndex & 2u) == 0u
            ? -box.halfExtents.y
            : box.halfExtents.y,
        (vertexIndex & 4u) == 0u
            ? -box.halfExtents.z
            : box.halfExtents.z
    );
    return box.center + quaternionRotate(box.rotation, local);
}

bool pointInsideInflatedBox(
    const float3 point,
    const thread WorldShape& box,
    const float inflation
) {
    const float3 local = quaternionInverseRotate(
        box.rotation,
        point - box.center
    );
    return all(
        abs(local) <= box.halfExtents + inflation
    );
}

float3 boxSupport(
    const thread WorldShape& box,
    const float3 direction
) {
    float3 result = box.center;
    for (uint axis = 0u; axis < 3u; ++axis) {
        const float3 basis = boxAxis(box, axis);
        result +=
            (dot(direction, basis) >= 0.0f
                 ? box.halfExtents[axis]
                 : -box.halfExtents[axis]) *
            basis;
    }
    return result;
}

void appendBoxBoxContact(
    thread ContactBatch& result,
    const float3 normal,
    const float separation,
    const float3 pointA,
    const float3 pointB,
    const uint featureA,
    const uint featureB
) {
    if (result.count >= 8u) {
        return;
    }
    result.contacts[result.count++] = makeContact(
        normal,
        separation,
        pointA,
        pointB,
        featureKey(MR_SHAPE_BOX, featureA),
        featureKey(MR_SHAPE_BOX, featureB)
    );
}

ContactBatch boxBoxContacts(
    const uint colliderA,
    const uint colliderB,
    const thread WorldShape& boxA,
    const thread WorldShape& boxB,
    const float acceptedContactDistance
) {
    ContactBatch result{};
    float3 axes[15];
    uint axisCount = 0u;
    for (uint axis = 0u; axis < 3u; ++axis) {
        axes[axisCount++] = boxAxis(boxA, axis);
    }
    for (uint axis = 0u; axis < 3u; ++axis) {
        axes[axisCount++] = boxAxis(boxB, axis);
    }
    for (uint axisA = 0u; axisA < 3u; ++axisA) {
        for (uint axisB = 0u; axisB < 3u; ++axisB) {
            const float3 crossed = cross(
                boxAxis(boxA, axisA),
                boxAxis(boxB, axisB)
            );
            const float squared = dot(crossed, crossed);
            if (squared > 1.0e-12f) {
                axes[axisCount++] = crossed * rsqrt(squared);
            }
        }
    }

    const float3 centerDelta = boxB.center - boxA.center;
    float bestSeparation = -INFINITY;
    float3 bestNormal =
        coincidentNormal(colliderA, colliderB);
    uint bestAxis = 0u;
    for (uint axisIndex = 0u;
         axisIndex < axisCount;
         ++axisIndex) {
        float3 axis = axes[axisIndex];
        const float projection = dot(centerDelta, axis);
        if (projection < 0.0f ||
            (projection == 0.0f &&
             dot(axis, bestNormal) < 0.0f)) {
            axis = -axis;
        }
        const float separation =
            abs(projection) -
            boxProjectionRadius(boxA, axis) -
            boxProjectionRadius(boxB, axis);
        if (separation > acceptedContactDistance) {
            return result;
        }
        const float tieTolerance =
            1.0e-7f *
            (1.0f + max(abs(bestSeparation), abs(separation)));
        if (!isfinite(bestSeparation) ||
            separation > bestSeparation + tieTolerance ||
            (abs(separation - bestSeparation) <= tieTolerance &&
             axisIndex < bestAxis)) {
            bestSeparation = separation;
            bestNormal = axis;
            bestAxis = axisIndex;
        }
    }

    const float radiusA =
        boxProjectionRadius(boxA, bestNormal);
    const float radiusB =
        boxProjectionRadius(boxB, bestNormal);
    const float nearPlaneB =
        dot(boxB.center, bestNormal) - radiusB;
    const float farPlaneA =
        dot(boxA.center, bestNormal) + radiusA;
    for (uint vertexIndex = 0u;
         vertexIndex < 8u;
         ++vertexIndex) {
        const float3 pointA =
            boxVertex(boxA, vertexIndex);
        if (!pointInsideInflatedBox(
                pointA,
                boxB,
                acceptedContactDistance
            )) {
            continue;
        }
        const float separation =
            nearPlaneB - dot(pointA, bestNormal);
        if (separation <= acceptedContactDistance) {
            appendBoxBoxContact(
                result,
                bestNormal,
                separation,
                pointA,
                pointA + bestNormal * separation,
                vertexIndex,
                128u + bestAxis
            );
        }
    }
    for (uint vertexIndex = 0u;
         vertexIndex < 8u;
         ++vertexIndex) {
        const float3 pointB =
            boxVertex(boxB, vertexIndex);
        if (!pointInsideInflatedBox(
                pointB,
                boxA,
                acceptedContactDistance
            )) {
            continue;
        }
        const float separation =
            dot(pointB, bestNormal) - farPlaneA;
        if (separation <= acceptedContactDistance) {
            appendBoxBoxContact(
                result,
                bestNormal,
                separation,
                pointB - bestNormal * separation,
                pointB,
                128u + bestAxis,
                vertexIndex
            );
        }
    }
    if (result.count == 0u) {
        const float3 pointA =
            boxSupport(boxA, bestNormal);
        const float3 pointB =
            boxSupport(boxB, -bestNormal);
        appendBoxBoxContact(
            result,
            bestNormal,
            dot(pointB - pointA, bestNormal),
            pointA,
            pointB,
            256u + bestAxis,
            256u + bestAxis
        );
    }
    return result;
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

    if (pairClass == kPairSphereCapsule) {
        const bool sphereIsA = shapeA.type == MR_SHAPE_SPHERE;
        const thread WorldShape& sphere =
            sphereIsA ? shapeA : shapeB;
        const thread WorldShape& capsule =
            sphereIsA ? shapeB : shapeA;
        const SegmentClosestPoint closest =
            closestPointOnSegment(
                sphere.center,
                capsule.capsuleEndpoint0,
                capsule.capsuleEndpoint1
            );
        const float3 delta = closest.point - sphere.center;
        const float centerDistance = length(delta);
        const float3 normal =
            centerDistance > kTiny
            ? delta / centerDistance
            : stableSegmentNormal(
                  capsule.capsuleEndpoint1 -
                      capsule.capsuleEndpoint0,
                  float3(0.0f),
                  sphere.index,
                  capsule.index
              );
        const float separation =
            centerDistance - sphere.radius - capsule.radius;
        if (separation <= acceptedContactDistance) {
            MRRawContactGPU contact = makeContact(
                normal,
                separation,
                sphere.center + normal * sphere.radius,
                closest.point - normal * capsule.radius,
                featureKey(MR_SHAPE_SPHERE, 0u),
                featureKey(
                    MR_SHAPE_CAPSULE,
                    capsuleFeature(closest.parameter)
                )
            );
            if (!sphereIsA) {
                contact = swappedContact(contact);
            }
            result.contacts[0] = contact;
            result.count = 1u;
        }
        return result;
    }

    if (pairClass == kPairCapsuleCapsule) {
        const SegmentPairClosestPoints closest =
            closestPointsOnSegments(
                shapeA.capsuleEndpoint0,
                shapeA.capsuleEndpoint1,
                shapeB.capsuleEndpoint0,
                shapeB.capsuleEndpoint1
            );
        const float3 delta = closest.pointB - closest.pointA;
        const float centerDistance = length(delta);
        const float3 normal =
            centerDistance > kTiny
            ? delta / centerDistance
            : stableSegmentNormal(
                  shapeA.capsuleEndpoint1 -
                      shapeA.capsuleEndpoint0,
                  shapeB.capsuleEndpoint1 -
                      shapeB.capsuleEndpoint0,
                  colliderA,
                  colliderB
              );
        const float separation =
            centerDistance - shapeA.radius - shapeB.radius;
        if (separation <= acceptedContactDistance) {
            result.contacts[0] = makeContact(
                normal,
                separation,
                closest.pointA + normal * shapeA.radius,
                closest.pointB - normal * shapeB.radius,
                featureKey(
                    MR_SHAPE_CAPSULE,
                    capsuleFeature(closest.parameterA)
                ),
                featureKey(
                    MR_SHAPE_CAPSULE,
                    capsuleFeature(closest.parameterB)
                )
            );
            result.count = 1u;
        }
        return result;
    }

    if (pairClass == kPairSphereBox) {
        const bool sphereIsA = shapeA.type == MR_SHAPE_SPHERE;
        const thread WorldShape& sphere =
            sphereIsA ? shapeA : shapeB;
        const thread WorldShape& box =
            sphereIsA ? shapeB : shapeA;
        const SphereBoxWitness witness =
            sphereBoxWitness(sphere, box);
        if (witness.separation <= acceptedContactDistance) {
            MRRawContactGPU contact = makeContact(
                witness.normal,
                witness.separation,
                sphere.center +
                    witness.normal * sphere.radius,
                witness.boxPoint,
                featureKey(MR_SHAPE_SPHERE, 0u),
                featureKey(
                    MR_SHAPE_BOX,
                    witness.boxFeature
                )
            );
            if (!sphereIsA) {
                contact = swappedContact(contact);
            }
            result.contacts[0] = contact;
            result.count = 1u;
        }
        return result;
    }

    if (pairClass == kPairCapsuleBox) {
        const bool capsuleIsA =
            shapeA.type == MR_SHAPE_CAPSULE;
        const thread WorldShape& capsule =
            capsuleIsA ? shapeA : shapeB;
        const thread WorldShape& box =
            capsuleIsA ? shapeB : shapeA;
        const CapsuleBoxWitness witness =
            capsuleBoxWitness(capsule, box);
        if (witness.separation <= acceptedContactDistance) {
            MRRawContactGPU contact = makeContact(
                witness.normal,
                witness.separation,
                witness.capsulePoint,
                witness.boxPoint,
                featureKey(
                    MR_SHAPE_CAPSULE,
                    capsuleFeature(
                        witness.capsuleParameter
                    )
                ),
                featureKey(
                    MR_SHAPE_BOX,
                    witness.boxFeature
                )
            );
            if (!capsuleIsA) {
                contact = swappedContact(contact);
            }
            result.contacts[0] = contact;
            result.count = 1u;
        }
        return result;
    }

    if (pairClass == kPairBoxBox) {
        return boxBoxContacts(
            colliderA,
            colliderB,
            shapeA,
            shapeB,
            acceptedContactDistance
        );
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

    if (pairClass == kPairCylinderPlane) {
        const float3 axis = finiteShape.cylinderAxis;
        const float axialProjection =
            dot(plane.planeNormal, axis);
        // Project through the orthonormal disk basis. This preserves the
        // first-order tilt and keeps the witness in the represented disk plane
        // when `n - axis * dot(n, axis)` would cancel in FP32.
        const float radialX = dot(
            plane.planeNormal,
            finiteShape.cylinderBasisX
        );
        const float radialZ = dot(
            plane.planeNormal,
            finiteShape.cylinderBasisZ
        );
        const float3 radialProjection =
            finiteShape.cylinderBasisX * radialX +
            finiteShape.cylinderBasisZ * radialZ;
        const float radialSquared =
            max(0.0f, radialX * radialX + radialZ * radialZ);

        if (radialSquared <=
            kCylinderAlignmentTolerance *
                kCylinderAlignmentTolerance) {
            const bool positiveCap = axialProjection < 0.0f;
            const float3 capCenter =
                finiteShape.center +
                axis *
                    (positiveCap
                        ? finiteShape.halfLength
                        : -finiteShape.halfLength);
            const uint featureBase =
                positiveCap
                ? kCylinderPositiveCapRingBase
                : kCylinderNegativeCapRingBase;
            for (uint point = 0u; point < 4u; ++point) {
                float3 direction;
                if (point == 0u) {
                    direction = finiteShape.cylinderBasisX;
                } else if (point == 1u) {
                    direction = -finiteShape.cylinderBasisX;
                } else if (point == 2u) {
                    direction = finiteShape.cylinderBasisZ;
                } else {
                    direction = -finiteShape.cylinderBasisZ;
                }
                const float3 surface =
                    capCenter +
                    direction * finiteShape.radius;
                const float separation =
                    dot(
                        plane.planeNormal,
                        surface - plane.center
                    );
                if (separation <= acceptedContactDistance) {
                    appendFinitePlaneContact(
                        result,
                        colliderA,
                        plane,
                        surface,
                        separation,
                        featureKey(
                            MR_SHAPE_CYLINDER,
                            featureBase + point
                        )
                    );
                }
            }
            // Fixed ring axes alone are not conservative for a small
            // arbitrary-azimuth tilt. Add the exact resolved support
            // direction so a shallow rim impact cannot fall between samples.
            // Do not reuse the geometry-length epsilon for this dimensionless
            // direction. A tiny tilt times a large valid radius can still be
            // a material support displacement. Values that underflow to zero
            // are already below the FP32 collision-query resolution.
            if (radialSquared > 0.0f) {
                const float3 radialDirection =
                    -radialProjection * rsqrt(radialSquared);
                const float3 surface =
                    capCenter +
                    radialDirection * finiteShape.radius;
                const float separation =
                    dot(
                        plane.planeNormal,
                        surface - plane.center
                    );
                if (separation <= acceptedContactDistance) {
                    appendFinitePlaneContact(
                        result,
                        colliderA,
                        plane,
                        surface,
                        separation,
                        featureKey(
                            MR_SHAPE_CYLINDER,
                            positiveCap
                                ? kCylinderPositiveGeneralRim
                                : kCylinderNegativeGeneralRim
                        )
                    );
                }
            }
            return result;
        }

        const float3 radialDirection =
            -radialProjection * rsqrt(radialSquared);
        if (abs(axialProjection) <=
            kCylinderAlignmentTolerance) {
            for (uint point = 0u; point < 2u; ++point) {
                const bool positiveCap = point != 0u;
                const float3 capCenter =
                    finiteShape.center +
                    axis *
                        (positiveCap
                            ? finiteShape.halfLength
                            : -finiteShape.halfLength);
                const float3 surface =
                    capCenter +
                    radialDirection * finiteShape.radius;
                const float separation =
                    dot(
                        plane.planeNormal,
                        surface - plane.center
                    );
                if (separation <= acceptedContactDistance) {
                    appendFinitePlaneContact(
                        result,
                        colliderA,
                        plane,
                        surface,
                        separation,
                        featureKey(
                            MR_SHAPE_CYLINDER,
                            positiveCap
                                ? kCylinderPositiveSideRim
                                : kCylinderNegativeSideRim
                        )
                    );
                }
            }
            return result;
        }

        const bool positiveCap = axialProjection < 0.0f;
        const float3 surface =
            finiteShape.center +
            axis *
                (positiveCap
                    ? finiteShape.halfLength
                    : -finiteShape.halfLength) +
            radialDirection * finiteShape.radius;
        const float separation =
            dot(plane.planeNormal, surface - plane.center);
        if (separation <= acceptedContactDistance) {
            appendFinitePlaneContact(
                result,
                colliderA,
                plane,
                surface,
                separation,
                featureKey(
                    MR_SHAPE_CYLINDER,
                    positiveCap
                        ? kCylinderPositiveGeneralRim
                        : kCylinderNegativeGeneralRim
                )
            );
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

namespace {

struct WorldManifoldCandidate {
    MRManifoldPointGPU point;
    float3 worldPoint;
    float separation;
    float tangentialDrift;
};

inline ulong worldFeatureKey(
    thread const WorldManifoldCandidate& candidate
) {
    return
        (ulong(candidate.point.featureAndLife[0]) << 32u) |
        ulong(candidate.point.featureAndLife[1]);
}

inline bool candidateLess(
    thread const WorldManifoldCandidate& left,
    thread const WorldManifoldCandidate& right
) {
    if (left.separation != right.separation) {
        return left.separation < right.separation;
    }
    return worldFeatureKey(left) < worldFeatureKey(right);
}

inline float3 stableTangent(const float3 normal) {
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
    return normalize(cross(reference, normal));
}

inline float3 worldPointFromAnchor(
    device const MRBodyStateGPU& body,
    const float4 rotation,
    const float4 localAnchor
) {
    return body.position.xyz +
        quaternionRotate(rotation, localAnchor.xyz);
}

inline float4 localAnchorFromWorld(
    device const MRBodyStateGPU& body,
    const float4 rotation,
    const float3 worldPoint
) {
    return float4(
        quaternionInverseRotate(
            rotation,
            worldPoint - body.position.xyz
        ),
        0.0f
    );
}

inline void sortCandidates(
    thread WorldManifoldCandidate* candidates,
    const uint count
) {
    for (uint index = 1u; index < count; ++index) {
        const WorldManifoldCandidate value = candidates[index];
        uint destination = index;
        while (destination > 0u &&
               candidateLess(value, candidates[destination - 1u])) {
            candidates[destination] =
                candidates[destination - 1u];
            --destination;
        }
        candidates[destination] = value;
    }
}

inline void reduceCandidates(
    thread WorldManifoldCandidate* candidates,
    thread uint& count,
    const float3 normal
) {
    sortCandidates(candidates, count);
    if (count > MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY) {
        bool selected[12];
        for (uint index = 0u; index < 12u; ++index) {
            selected[index] = false;
        }
        uint chosen[MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY];
        chosen[0] = 0u;
        selected[0] = true;
        uint chosenCount = 1u;

        for (uint stage = 1u;
             stage < MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
             ++stage) {
            uint best = MR_INVALID_INDEX;
            float bestScore = -1.0f;
            ulong bestFeature = ~ulong(0);
            for (uint index = 0u; index < count; ++index) {
                if (selected[index]) {
                    continue;
                }
                float score = 0.0f;
                if (stage == 1u) {
                    const float3 delta =
                        candidates[index].worldPoint -
                        candidates[chosen[0]].worldPoint;
                    const float3 tangent =
                        delta - normal * dot(delta, normal);
                    score = dot(tangent, tangent);
                } else if (stage == 2u) {
                    const float3 a =
                        candidates[chosen[1]].worldPoint -
                        candidates[chosen[0]].worldPoint;
                    const float3 b =
                        candidates[index].worldPoint -
                        candidates[chosen[0]].worldPoint;
                    score = abs(dot(cross(a, b), normal));
                } else {
                    score = INFINITY;
                    for (uint selectedIndex = 0u;
                         selectedIndex < chosenCount;
                         ++selectedIndex) {
                        const float3 delta =
                            candidates[index].worldPoint -
                            candidates[
                                chosen[selectedIndex]
                            ].worldPoint;
                        const float3 tangent =
                            delta - normal * dot(delta, normal);
                        score = min(score, dot(tangent, tangent));
                    }
                }
                const ulong feature = worldFeatureKey(
                    candidates[index]
                );
                if (score > bestScore + 1.0e-12f ||
                    (abs(score - bestScore) <= 1.0e-12f &&
                     feature < bestFeature)) {
                    best = index;
                    bestScore = score;
                    bestFeature = feature;
                }
            }
            if (best != MR_INVALID_INDEX) {
                chosen[chosenCount++] = best;
                selected[best] = true;
            }
        }

        WorldManifoldCandidate reduced[
            MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY
        ];
        for (uint index = 0u; index < chosenCount; ++index) {
            reduced[index] = candidates[chosen[index]];
        }
        count = chosenCount;
        for (uint index = 0u; index < count; ++index) {
            candidates[index] = reduced[index];
        }
    }

    // Canonical feature order makes warm-start and IR order independent of
    // support-point enumeration.
    for (uint index = 1u; index < count; ++index) {
        const WorldManifoldCandidate value = candidates[index];
        const ulong feature = worldFeatureKey(value);
        uint destination = index;
        while (destination > 0u &&
               feature <
                   worldFeatureKey(candidates[destination - 1u])) {
            candidates[destination] =
                candidates[destination - 1u];
            --destination;
        }
        candidates[destination] = value;
    }
}

inline bool buildPersistentWorldManifold(
    const uint environment,
    const MRCompiledCollisionPairGPU pair,
    thread const ContactBatch& raw,
    device const MRShapeGPU* shapes,
    device const MRBodyStateGPU* environmentBodies,
    const bool hasOld,
    device const MRManifoldHeaderGPU* oldHeaderPointer,
    device const MRManifoldPointGPU* oldPointPointer,
    const float4 thresholds,
    thread MRManifoldHeaderGPU& outputHeader,
    thread MRManifoldPointGPU* outputPoints,
    thread uint& retainedPoints,
    thread uint& newPoints
) {
    if (raw.count == 0u) {
        return false;
    }
    const MRShapeGPU shapeA = shapes[pair.colliderA];
    const MRShapeGPU shapeB = shapes[pair.colliderB];
    device const MRBodyStateGPU& bodyA =
        environmentBodies[shapeA.bodyIndex];
    device const MRBodyStateGPU& bodyB =
        environmentBodies[shapeB.bodyIndex];
    float4 rotationA;
    float4 rotationB;
    if (!checkedQuaternion(bodyA.orientation, rotationA) ||
        !checkedQuaternion(bodyB.orientation, rotationB)) {
        return false;
    }

    uint deepest = 0u;
    for (uint index = 1u; index < raw.count; ++index) {
        const MRRawContactGPU left = raw.contacts[index];
        const MRRawContactGPU right = raw.contacts[deepest];
        if (left.normalAndSeparation.w <
                right.normalAndSeparation.w ||
            (left.normalAndSeparation.w ==
                 right.normalAndSeparation.w &&
             (
                 left.featureAndFlags[0] <
                     right.featureAndFlags[0] ||
                 (
                     left.featureAndFlags[0] ==
                         right.featureAndFlags[0] &&
                     left.featureAndFlags[1] <
                         right.featureAndFlags[1]
                 )
             ))) {
            deepest = index;
        }
    }
    const float3 normalWorld =
        normalize(raw.contacts[deepest].normalAndSeparation.xyz);
    if (!finiteFloat3(normalWorld) ||
        dot(normalWorld, normalWorld) < 0.9999f) {
        return false;
    }

    WorldManifoldCandidate candidates[12];
    uint candidateCount = 0u;
    bool oldNormalCompatible = false;
    float3 transportedTangent = float3(0.0f);
    float oldAge = 0.0f;
    if (hasOld) {
        const MRManifoldHeaderGPU oldHeader =
            oldHeaderPointer[0];
        const bool generationsMatch =
            oldHeader.generationsAndFlags[0] ==
                shapeA.slotGeneration &&
            oldHeader.generationsAndFlags[1] ==
                shapeB.slotGeneration &&
            oldHeader.pairAndCount[3] <=
                MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
        if (generationsMatch) {
            const float3 oldNormal = normalize(
                quaternionRotate(
                    rotationA,
                    oldHeader.normalAndAge.xyz
                )
            );
            oldNormalCompatible =
                finiteFloat3(oldNormal) &&
                dot(oldNormal, normalWorld) >= thresholds.w;
            if (oldNormalCompatible) {
                const float3 oldTangent = quaternionRotate(
                    rotationA,
                    oldHeader.tangentAndMetric.xyz
                );
                transportedTangent =
                    oldTangent -
                    normalWorld * dot(oldTangent, normalWorld);
                oldAge = oldHeader.normalAndAge.w;
                for (uint pointIndex = 0u;
                     pointIndex < oldHeader.pairAndCount[3];
                     ++pointIndex) {
                    const MRManifoldPointGPU oldPoint =
                        oldPointPointer[pointIndex];
                    const float3 pointAWorld =
                        worldPointFromAnchor(
                            bodyA,
                            rotationA,
                            oldPoint.localAnchorA
                        );
                    const float3 pointBWorld =
                        worldPointFromAnchor(
                            bodyB,
                            rotationB,
                            oldPoint.localAnchorB
                        );
                    const float3 delta =
                        pointBWorld - pointAWorld;
                    const float separation =
                        dot(delta, normalWorld);
                    const float3 tangent =
                        delta - normalWorld * separation;
                    const float drift = length(tangent);
                    if (separation > thresholds.x ||
                        drift > thresholds.y) {
                        continue;
                    }
                    WorldManifoldCandidate candidate;
                    candidate.point = oldPoint;
                    candidate.point.featureAndLife[2] =
                        oldPoint.featureAndLife[2] ==
                            0xffffffffu
                        ? 0xffffffffu
                        : oldPoint.featureAndLife[2] + 1u;
                    candidate.worldPoint =
                        0.5f * (pointAWorld + pointBWorld);
                    candidate.separation = separation;
                    candidate.tangentialDrift = drift;
                    candidates[candidateCount++] = candidate;
                    ++retainedPoints;
                }
            }
        }
    }

    const float mergeDistanceSquared = thresholds.z * thresholds.z;
    for (uint rawIndex = 0u; rawIndex < raw.count; ++rawIndex) {
        const MRRawContactGPU contact = raw.contacts[rawIndex];
        const float4 localA = localAnchorFromWorld(
            bodyA,
            rotationA,
            contact.pointAWorld.xyz
        );
        const float4 localB = localAnchorFromWorld(
            bodyB,
            rotationB,
            contact.pointBWorld.xyz
        );
        uint match = MR_INVALID_INDEX;
        for (uint candidateIndex = 0u;
             candidateIndex < candidateCount;
             ++candidateIndex) {
            if (candidates[candidateIndex]
                    .point.featureAndLife[0] ==
                    contact.featureAndFlags[0] &&
                candidates[candidateIndex]
                    .point.featureAndLife[1] ==
                    contact.featureAndFlags[1]) {
                match = candidateIndex;
                break;
            }
        }
        if (match == MR_INVALID_INDEX) {
            for (uint candidateIndex = 0u;
                 candidateIndex < candidateCount;
                 ++candidateIndex) {
                const float3 deltaA =
                    candidates[candidateIndex]
                        .point.localAnchorA.xyz -
                    localA.xyz;
                const float3 deltaB =
                    candidates[candidateIndex]
                        .point.localAnchorB.xyz -
                    localB.xyz;
                if (dot(deltaA, deltaA) + dot(deltaB, deltaB) <=
                    2.0f * mergeDistanceSquared) {
                    match = candidateIndex;
                    break;
                }
            }
        }

        if (match == MR_INVALID_INDEX) {
            if (candidateCount >= 12u) {
                return false;
            }
            match = candidateCount++;
            candidates[match] = {};
            candidates[match].point.featureAndLife[2] = 0u;
            ++newPoints;
        }
        candidates[match].point.localAnchorA = localA;
        candidates[match].point.localAnchorB = localB;
        candidates[match].point.featureAndLife[0] =
            contact.featureAndFlags[0];
        candidates[match].point.featureAndLife[1] =
            contact.featureAndFlags[1];
        candidates[match].point.featureAndLife[3] =
            contact.featureAndFlags[3];
        candidates[match].worldPoint =
            0.5f *
            (contact.pointAWorld.xyz + contact.pointBWorld.xyz);
        candidates[match].separation =
            contact.normalAndSeparation.w;
        candidates[match].tangentialDrift = 0.0f;
    }

    if (candidateCount == 0u) {
        return false;
    }
    reduceCandidates(candidates, candidateCount, normalWorld);

    float3 tangentWorld = transportedTangent;
    if (!oldNormalCompatible ||
        !finiteFloat3(tangentWorld) ||
        dot(tangentWorld, tangentWorld) <= 1.0e-12f) {
        tangentWorld = stableTangent(normalWorld);
    } else {
        tangentWorld = normalize(tangentWorld);
    }
    outputHeader = {};
    outputHeader.pairAndCount[0] = environment;
    outputHeader.pairAndCount[1] = pair.colliderA;
    outputHeader.pairAndCount[2] = pair.colliderB;
    outputHeader.pairAndCount[3] = candidateCount;
    outputHeader.generationsAndFlags[0] =
        shapeA.slotGeneration;
    outputHeader.generationsAndFlags[1] =
        shapeB.slotGeneration;
    outputHeader.generationsAndFlags[2] = 0u;
    outputHeader.generationsAndFlags[3] = 0u;
    outputHeader.normalAndAge = float4(
        quaternionInverseRotate(rotationA, normalWorld),
        oldNormalCompatible
            ? min(oldAge + 1.0f, 65535.0f)
            : 0.0f
    );

    float breakingMetric = 0.0f;
    for (uint pointIndex = 0u;
         pointIndex < candidateCount;
         ++pointIndex) {
        outputPoints[pointIndex] = candidates[pointIndex].point;
        breakingMetric = max(
            breakingMetric,
            max(
                max(candidates[pointIndex].separation, 0.0f),
                candidates[pointIndex].tangentialDrift
            )
        );
    }
    for (uint pointIndex = candidateCount;
         pointIndex < MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
         ++pointIndex) {
        outputPoints[pointIndex] = {};
    }
    outputHeader.tangentAndMetric = float4(
        quaternionInverseRotate(rotationA, tangentWorld),
        breakingMetric
    );
    return true;
}

inline float geometricMean(const float left, const float right) {
    return sqrt(max(0.0f, left * right));
}

inline ulong collisionPairKey(
    const uint colliderA,
    const uint colliderB
) {
    return (ulong(colliderA) << 32u) | ulong(colliderB);
}

inline ulong collisionFeatureKey(
    const MRManifoldPointGPU point
) {
    return
        (ulong(point.featureAndLife[0]) << 32u) |
        ulong(point.featureAndLife[1]);
}

inline bool finiteContactDispatch(
    device const MRMetalWorldContactDispatchGPU& dispatch
) {
    const uint knownFlags =
        MR_METAL_WORLD_CONTACT_DETERMINISTIC |
        MR_METAL_WORLD_CONTACT_WARM_START |
        MR_METAL_WORLD_CONTACT_CAPTURE_EVIDENCE |
        MR_METAL_WORLD_CONTACT_HAS_KINEMATIC_TARGETS;
    return
        dispatch.abiVersion ==
            MR_METAL_WORLD_CONTACT_ABI_VERSION &&
        dispatch.environmentCount > 0u &&
        dispatch.bodyCount > 0u &&
        dispatch.shapeCount > 0u &&
        dispatch.pairStride >= dispatch.pairCapacity &&
        dispatch.rawContactStride >=
            dispatch.rawContactCapacity &&
        dispatch.manifoldStride >= dispatch.manifoldCapacity &&
        dispatch.constraintStride >=
            dispatch.constraintCapacity &&
        dispatch.rowStride >= dispatch.rowCapacity &&
        dispatch.islandStride >= dispatch.islandCapacity &&
        dispatch.pointQueryStride >=
            2u * dispatch.constraintCapacity &&
        dispatch.factorStride >= dispatch.nv * dispatch.nv &&
        dispatch.rowCapacity >=
            3u * dispatch.constraintCapacity &&
        dispatch.nv > 0u &&
        dispatch.velocityIterations > 0u &&
        dispatch.solverType >= MR_SOLVER_THROUGHPUT_TGS &&
        dispatch.solverType <= MR_SOLVER_THROUGHPUT_PGS &&
        (dispatch.flags & ~knownFlags) == 0u &&
        finiteFloat4(dispatch.timestepAndBias) &&
        dispatch.timestepAndBias.x > 0.0f &&
        all(dispatch.timestepAndBias.yzw >= 0.0f) &&
        finiteFloat4(dispatch.manifoldThresholds) &&
        all(dispatch.manifoldThresholds.xyz >= 0.0f) &&
        dispatch.manifoldThresholds.w >= -1.0f &&
        dispatch.manifoldThresholds.w <= 1.0f;
}

} // namespace

// Projects each collider exactly once per environment and microstep. The
// flattened environment-major grid keeps adjacent SIMD lanes on adjacent
// colliders while preserving deterministic pair order in the later compiler.
kernel void mr_world_project_colliders(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRShapeGPU* shapes [[buffer(1)]],
    device const MRBodyStateGPU* bodies [[buffer(2)]],
    device MRProjectedColliderGPU* projectedColliders [[buffer(3)]],
    const uint threadIndex [[thread_position_in_grid]]
) {
    if (dispatch.shapeCount == 0u) {
        return;
    }
    const uint environment = threadIndex / dispatch.shapeCount;
    if (environment >= dispatch.environmentCount) {
        return;
    }
    const uint collider =
        threadIndex - environment * dispatch.shapeCount;
    const uint outputIndex =
        environment * dispatch.shapeCount + collider;
    WorldShape worldShape = {};
    uint failureCode = MR_STEP_SUCCESS;
    const bool projected = makeWorldShape(
        collider,
        shapes,
        bodies + environment * dispatch.bodyStateStride,
        dispatch.bodyCount,
        worldShape,
        failureCode
    );
    if (!projected) {
        worldShape.index = collider;
        worldShape.type = shapes[collider].shapeType;
        worldShape.body = shapes[collider].bodyIndex;
        projectedColliders[outputIndex] = projectedCollider(
            worldShape,
            failureCode
        );
        return;
    }
    projectedColliders[outputIndex] = projectedCollider(
        worldShape,
        MR_STEP_SUCCESS
    );
}

// Executes the compiled-pair broadphase as a flat environment-major queue.
// Every eligible pair owns one deterministic flag slot; the serial manifold
// compiler consumes those slots in compiled order without a host-visible
// count or global append atomic.
kernel void mr_world_flag_eligible_pairs(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRShapeGPU* shapes [[buffer(1)]],
    device const MRCompiledCollisionPairGPU* eligiblePairs [[buffer(2)]],
    device const MRProjectedColliderGPU* projectedColliders [[buffer(3)]],
    device uint* overlapFlags [[buffer(4)]],
    const uint threadIndex [[thread_position_in_grid]]
) {
    if (dispatch.eligiblePairCount == 0u) {
        return;
    }
    const uint environment =
        threadIndex / dispatch.eligiblePairCount;
    if (environment >= dispatch.environmentCount) {
        return;
    }
    const uint eligibleIndex =
        threadIndex -
        environment * dispatch.eligiblePairCount;
    const MRCompiledCollisionPairGPU compiled =
        eligiblePairs[eligibleIndex];
    uint flag = 0u;
    if (compiled.colliderA >= dispatch.shapeCount ||
        compiled.colliderB >= dispatch.shapeCount ||
        compiled.colliderA >= compiled.colliderB ||
        compiled.pairClass == MR_COLLISION_PAIR_UNSUPPORTED) {
        flag = 0x80000000u | MR_STEP_UNSUPPORTED;
    } else {
        device const MRProjectedColliderGPU& projectedA =
            projectedColliders[
                environment * dispatch.shapeCount +
                compiled.colliderA
            ];
        device const MRProjectedColliderGPU& projectedB =
            projectedColliders[
                environment * dispatch.shapeCount +
                compiled.colliderB
            ];
        if (projectedA.statusAndFlags.x != MR_STEP_SUCCESS ||
            projectedB.statusAndFlags.x != MR_STEP_SUCCESS) {
            const uint failureCode =
                projectedA.statusAndFlags.x != MR_STEP_SUCCESS
                ? projectedA.statusAndFlags.x
                : projectedB.statusAndFlags.x;
            flag = 0x80000000u | failureCode;
        } else if (projectedPairMayOverlap(
                       shapes[compiled.colliderA].shapeType,
                       shapes[compiled.colliderB].shapeType,
                       projectedA,
                       projectedB
                   )) {
            flag = 1u;
        }
    }
    overlapFlags[
        environment * dispatch.eligiblePairCount +
        eligibleIndex
    ] = flag;
}

// Deterministic environment-major collision/manifold/ConstraintIR compiler.
// One thread owns one cloned environment. This deliberately favors the small,
// stable eligible-pair streams common in batched robot RL; dynamic counts and
// every capacity decision remain device-resident.
kernel void mr_world_collide_compile(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRShapeGPU* shapes [[buffer(1)]],
    device const MRMaterialGPU* materials [[buffer(2)]],
    device const MRBodyStateGPU* bodies [[buffer(3)]],
    device const MRArticulationGPU* articulations [[buffer(4)]],
    device const MRCompiledCollisionPairGPU* eligiblePairs [[buffer(5)]],
    device const uint* oldManifoldCounts [[buffer(6)]],
    device const MRManifoldHeaderGPU* oldManifoldHeaders [[buffer(7)]],
    device const MRManifoldPointGPU* oldManifoldPoints [[buffer(8)]],
    device MRCandidatePairGPU* outputPairs [[buffer(9)]],
    device MRRawContactGPU* outputRawContacts [[buffer(10)]],
    device uint* outputRawPairIndices [[buffer(11)]],
    device MRManifoldHeaderGPU* candidateManifoldHeaders [[buffer(12)]],
    device MRManifoldPointGPU* candidateManifoldPoints [[buffer(13)]],
    device uint* candidateManifoldCounts [[buffer(14)]],
    device MRContactConstraintGPU* contacts [[buffer(15)]],
    device MRContactPointMetaGPU* contactMetadata [[buffer(16)]],
    device MRConstraintIRBlockGPU* blocks [[buffer(17)]],
    device MRConstraintIREndpointGPU* endpoints [[buffer(18)]],
    device MRConstraintIRRowGPU* rows [[buffer(19)]],
    device MRConstraintIRConeGPU* cones [[buffer(20)]],
    device MRArticulatedPointImpulseGPU* pointQueries [[buffer(21)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(22)]],
    constant MRMetalWorldPassGPU& pass [[buffer(23)]],
    device const MRProjectedColliderGPU* projectedColliders
        [[buffer(24)]],
    device const uint* pairOverlapFlags [[buffer(25)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }

    MRMetalWorldContactStatusGPU status = statuses[environment];
    if (status.code != MR_STEP_SUCCESS) {
        return;
    }
    status.environment = environment;
    status.controlStep = pass.controlStep;
    status.physicsSubstep = pass.physicsSubstep;
    status.firstFailingPair = MR_INVALID_INDEX;
    status.firstFailingConstraint = MR_INVALID_INDEX;
    if (!finiteContactDispatch(dispatch) ||
        pass.physicsSubstep >=
            MR_METAL_WORLD_MAX_PHYSICS_SUBSTEPS) {
        status.code = MR_STEP_UNSUPPORTED;
        statuses[environment] = status;
        return;
    }

    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    const uint pairBase =
        environment * dispatch.pairStride;
    const uint rawBase =
        environment * dispatch.rawContactStride;
    const uint manifoldBase =
        environment * dispatch.manifoldStride;
    const uint manifoldPointBase =
        manifoldBase *
        MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
    const uint constraintBase =
        environment * dispatch.constraintStride;
    const uint rowBase =
        environment * dispatch.rowStride;
    const uint endpointBase = 2u * constraintBase;
    const uint queryBase =
        environment * dispatch.pointQueryStride;
    const uint rootBody =
        articulations[dispatch.articulationIndex].rootBody;
    device const MRProjectedColliderGPU*
        environmentProjectedColliders =
            projectedColliders +
            environment * dispatch.shapeCount;
    device const uint* environmentPairOverlapFlags =
        pairOverlapFlags +
        environment * dispatch.eligiblePairCount;

    MRArticulatedPointImpulseGPU dummyQuery = {};
    dummyQuery.bodyIndex = rootBody;

    const uint oldCount = min(
        oldManifoldCounts[environment],
        dispatch.manifoldCapacity
    );
    uint oldCursor = 0u;
    uint pairCount = 0u;
    uint rawCount = 0u;
    uint manifoldCount = 0u;
    uint constraintCount = 0u;
    uint retainedPoints = 0u;
    uint newPoints = 0u;
    float maximumPenetration = 0.0f;
    uint failureCode = MR_STEP_SUCCESS;
    uint firstPairOverflow = MR_INVALID_INDEX;
    uint firstRawOverflowPair = MR_INVALID_INDEX;
    uint firstManifoldOverflowPair = MR_INVALID_INDEX;
    uint firstConstraintOverflow = MR_INVALID_INDEX;

    device const MRBodyStateGPU* environmentBodies =
        bodies + bodyBase;
    for (uint eligibleIndex = 0u;
         eligibleIndex < dispatch.eligiblePairCount;
         ++eligibleIndex) {
        const MRCompiledCollisionPairGPU compiled =
            eligiblePairs[eligibleIndex];
        if (compiled.colliderA >= dispatch.shapeCount ||
            compiled.colliderB >= dispatch.shapeCount ||
            compiled.colliderA >= compiled.colliderB ||
            compiled.pairClass == MR_COLLISION_PAIR_UNSUPPORTED) {
            status.code = MR_STEP_UNSUPPORTED;
            status.firstFailingPair = eligibleIndex;
            break;
        }

        const uint overlapFlag =
            environmentPairOverlapFlags[eligibleIndex];
        if ((overlapFlag & 0x80000000u) != 0u) {
            status.code = overlapFlag & 0x7fffffffu;
            status.firstFailingPair = eligibleIndex;
            break;
        }
        if (overlapFlag == 0u) {
            continue;
        }

        device const MRProjectedColliderGPU& projectedA =
            environmentProjectedColliders[compiled.colliderA];
        device const MRProjectedColliderGPU& projectedB =
            environmentProjectedColliders[compiled.colliderB];
        WorldShape shapeA;
        WorldShape shapeB;
        if (!loadProjectedCollider(
                compiled.colliderA,
                shapes[compiled.colliderA],
                projectedA,
                shapeA,
                failureCode
            ) ||
            !loadProjectedCollider(
                compiled.colliderB,
                shapes[compiled.colliderB],
                projectedB,
                shapeB,
                failureCode
            )) {
            status.code = failureCode;
            status.firstFailingPair = eligibleIndex;
            break;
        }

        const uint currentPairIndex = pairCount++;
        if (currentPairIndex < dispatch.pairCapacity) {
            MRCandidatePairGPU outputPair = {};
            outputPair.environment = environment;
            outputPair.colliderA = compiled.colliderA;
            outputPair.colliderB = compiled.colliderB;
            outputPair.flags = compiled.pairClass;
            outputPairs[pairBase + currentPairIndex] =
                outputPair;
        } else if (firstPairOverflow == MR_INVALID_INDEX) {
            firstPairOverflow = eligibleIndex;
        }

        const ContactBatch raw = generateContacts(
            compiled.colliderA,
            compiled.colliderB,
            compiled.pairClass,
            shapeA,
            shapeB
        );
        for (uint rawIndex = 0u;
             rawIndex < raw.count;
             ++rawIndex) {
            if (!finiteContact(raw.contacts[rawIndex])) {
                status.code = MR_STEP_NONFINITE_RESULT;
                status.firstFailingPair = eligibleIndex;
                break;
            }
            if (rawCount < dispatch.rawContactCapacity) {
                outputRawContacts[rawBase + rawCount] =
                    raw.contacts[rawIndex];
                outputRawPairIndices[rawBase + rawCount] =
                    currentPairIndex;
            } else if (
                firstRawOverflowPair == MR_INVALID_INDEX
            ) {
                firstRawOverflowPair = eligibleIndex;
            }
            ++rawCount;
        }
        if (status.code != MR_STEP_SUCCESS || raw.count == 0u) {
            if (status.code != MR_STEP_SUCCESS) {
                break;
            }
            continue;
        }

        while (oldCursor < oldCount) {
            const MRManifoldHeaderGPU oldHeader =
                oldManifoldHeaders[manifoldBase + oldCursor];
            if (oldHeader.pairAndCount[1] > compiled.colliderA ||
                (oldHeader.pairAndCount[1] == compiled.colliderA &&
                 oldHeader.pairAndCount[2] >=
                     compiled.colliderB)) {
                break;
            }
            ++oldCursor;
        }
        bool hasOld = false;
        if (oldCursor < oldCount) {
            const MRManifoldHeaderGPU oldHeader =
                oldManifoldHeaders[manifoldBase + oldCursor];
            hasOld =
                oldHeader.pairAndCount[0] == environment &&
                oldHeader.pairAndCount[1] == compiled.colliderA &&
                oldHeader.pairAndCount[2] == compiled.colliderB;
        }

        MRManifoldHeaderGPU manifoldHeader = {};
        MRManifoldPointGPU manifoldPoints[
            MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY
        ];
        device const MRManifoldHeaderGPU* oldHeaderPointer =
            oldManifoldHeaders + manifoldBase + oldCursor;
        device const MRManifoldPointGPU* oldPointPointer =
            oldManifoldPoints +
            (
                manifoldBase + oldCursor
            ) * MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
        if (!buildPersistentWorldManifold(
                environment,
                compiled,
                raw,
                shapes,
                environmentBodies,
                hasOld,
                oldHeaderPointer,
                oldPointPointer,
                dispatch.manifoldThresholds,
                manifoldHeader,
                manifoldPoints,
                retainedPoints,
                newPoints
            )) {
            status.code = MR_STEP_NONFINITE_RESULT;
            status.firstFailingPair = eligibleIndex;
            break;
        }

        const uint currentManifold = manifoldCount++;
        if (currentManifold < dispatch.manifoldCapacity) {
            candidateManifoldHeaders[
                manifoldBase + currentManifold
            ] = manifoldHeader;
            for (uint point = 0u;
                 point <
                     MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
                 ++point) {
                candidateManifoldPoints[
                    manifoldPointBase +
                    currentManifold *
                        MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY +
                    point
                ] = manifoldPoints[point];
            }
        } else if (
            firstManifoldOverflowPair == MR_INVALID_INDEX
        ) {
            firstManifoldOverflowPair = eligibleIndex;
        }

        const MRShapeGPU sourceA = shapes[compiled.colliderA];
        const MRShapeGPU sourceB = shapes[compiled.colliderB];
        const MRMaterialGPU materialA =
            materials[sourceA.materialIndex];
        const MRMaterialGPU materialB =
            materials[sourceB.materialIndex];
        device const MRBodyStateGPU& bodyA =
            environmentBodies[sourceA.bodyIndex];
        device const MRBodyStateGPU& bodyB =
            environmentBodies[sourceB.bodyIndex];
        float4 rotationA;
        float4 rotationB;
        if (!checkedQuaternion(bodyA.orientation, rotationA) ||
            !checkedQuaternion(bodyB.orientation, rotationB)) {
            status.code = MR_STEP_NONFINITE_INPUT;
            status.firstFailingPair = eligibleIndex;
            break;
        }
        const float3 normalWorld = normalize(
            quaternionRotate(
                rotationA,
                manifoldHeader.normalAndAge.xyz
            )
        );
        float3 tangentWorld = quaternionRotate(
            rotationA,
            manifoldHeader.tangentAndMetric.xyz
        );
        tangentWorld -=
            normalWorld * dot(tangentWorld, normalWorld);
        tangentWorld =
            dot(tangentWorld, tangentWorld) > 1.0e-12f
            ? normalize(tangentWorld)
            : stableTangent(normalWorld);
        const float3 bitangentWorld =
            cross(normalWorld, tangentWorld);

        for (uint pointIndex = 0u;
             pointIndex < manifoldHeader.pairAndCount[3];
             ++pointIndex) {
            const uint currentConstraint = constraintCount++;
            if (currentConstraint >= dispatch.constraintCapacity) {
                if (firstConstraintOverflow ==
                    MR_INVALID_INDEX) {
                    firstConstraintOverflow =
                        currentConstraint;
                }
                continue;
            }
            const MRManifoldPointGPU manifoldPoint =
                manifoldPoints[pointIndex];
            const float3 pointAWorld = worldPointFromAnchor(
                bodyA,
                rotationA,
                manifoldPoint.localAnchorA
            );
            const float3 pointBWorld = worldPointFromAnchor(
                bodyB,
                rotationB,
                manifoldPoint.localAnchorB
            );
            const float3 pointWorld =
                0.5f * (pointAWorld + pointBWorld);
            const float geometricSeparation =
                dot(pointBWorld - pointAWorld, normalWorld);
            const float restSeparation =
                sourceA.contactRestAndBoundingRadius.y +
                sourceB.contactRestAndBoundingRadius.y;
            const float effectiveSeparation =
                geometricSeparation - restSeparation;
            maximumPenetration = max(
                maximumPenetration,
                max(-effectiveSeparation, 0.0f)
            );

            const uint outputConstraint =
                constraintBase + currentConstraint;
            MRContactConstraintGPU contact = {};
            contact.bodyA = sourceA.bodyIndex;
            contact.bodyB = sourceB.bodyIndex;
            contact.flags =
                manifoldPoint.featureAndLife[2] == 0u
                ? MR_CONSTRAINT_FLAG_NEW_IMPACT
                : MR_CONSTRAINT_FLAG_WARM_STARTED;
            contact.islandIndex = MR_INVALID_INDEX;
            contact.pairKey = collisionPairKey(
                compiled.colliderA,
                compiled.colliderB
            );
            contact.featureKey =
                collisionFeatureKey(manifoldPoint);
            contact.pointAndSeparation =
                float4(pointWorld, effectiveSeparation);
            contact.normal = float4(normalWorld, 0.0f);
            contact.friction = float4(
                geometricMean(
                    materialA.friction.x,
                    materialB.friction.x
                ),
                geometricMean(
                    materialA.friction.y,
                    materialB.friction.y
                ),
                geometricMean(
                    materialA.friction.z,
                    materialB.friction.z
                ),
                geometricMean(
                    materialA.friction.w,
                    materialB.friction.w
                )
            );
            contact.response = float4(
                max(materialA.response.x, materialB.response.x),
                max(materialA.response.y, materialB.response.y),
                materialA.response.z + materialB.response.z,
                0.0f
            );
            contact.targetVelocityAndPreSolveNormal =
                float4(0.0f);
            contact.impulses =
                (dispatch.flags &
                 MR_METAL_WORLD_CONTACT_WARM_START) != 0u
                ? manifoldPoint.impulses
                : float4(0.0f);
            contacts[outputConstraint] = contact;

            MRContactPointMetaGPU metadata = {};
            metadata.colliderA = compiled.colliderA;
            metadata.colliderB = compiled.colliderB;
            metadata.manifoldIndex = currentManifold;
            metadata.pointIndex = pointIndex;
            metadata.localAnchorA =
                manifoldPoint.localAnchorA;
            metadata.localAnchorB =
                manifoldPoint.localAnchorB;
            contactMetadata[outputConstraint] = metadata;

            MRConstraintIRBlockGPU block = {};
            block.key.words[0] = compiled.colliderA;
            block.key.words[1] = compiled.colliderB;
            block.key.words[2] =
                manifoldPoint.featureAndLife[0];
            block.key.words[3] =
                manifoldPoint.featureAndLife[1];
            block.type = MR_CONSTRAINT_CONTACT;
            block.dimension = 3u;
            block.flags = contact.flags;
            block.islandIndex = MR_INVALID_INDEX;
            block.endpointOffset = 2u * currentConstraint;
            block.endpointCount = 2u;
            block.rowOffset = 3u * currentConstraint;
            block.impulseOffset = 3u * currentConstraint;
            block.coneIndex = currentConstraint;
            block.eventSlot = MR_CONSTRAINT_IR_INVALID_INDEX;
            blocks[outputConstraint] = block;

            MRConstraintIREndpointGPU endpointA = {};
            endpointA.objectIndex = sourceA.bodyIndex;
            endpointA.articulationIndex =
                bodyA.flagsAndIndices[1];
            endpointA.linkIndex = bodyA.flagsAndIndices[2];
            endpointA.role = MR_CONSTRAINT_IR_ENDPOINT_A;
            endpointA.jacobianKind =
                MR_CONSTRAINT_IR_JACOBIAN_BODY_LOCAL_POINT;
            endpointA.anchor = manifoldPoint.localAnchorA;
            MRConstraintIREndpointGPU endpointB = endpointA;
            endpointB.objectIndex = sourceB.bodyIndex;
            endpointB.articulationIndex =
                bodyB.flagsAndIndices[1];
            endpointB.linkIndex = bodyB.flagsAndIndices[2];
            endpointB.role = MR_CONSTRAINT_IR_ENDPOINT_B;
            endpointB.anchor = manifoldPoint.localAnchorB;
            endpoints[endpointBase + 2u * currentConstraint] =
                endpointA;
            endpoints[
                endpointBase + 2u * currentConstraint + 1u
            ] = endpointB;

            const float3 directions[3] = {
                normalWorld,
                tangentWorld,
                bitangentWorld,
            };
            for (uint localRow = 0u;
                 localRow < 3u;
                 ++localRow) {
                MRConstraintIRRowGPU row = {};
                row.direction =
                    float4(directions[localRow], 0.0f);
                row.positionError =
                    localRow == 0u ? effectiveSeparation : 0.0f;
                row.targetVelocity = 0.0f;
                row.compliance =
                    localRow == 0u
                    ? materialA.response.z +
                        materialB.response.z
                    : 0.0f;
                row.dissipation =
                    localRow == 0u
                    ? materialA.response.w +
                        materialB.response.w
                    : 0.0f;
                row.timeConstant =
                    2.0f * dispatch.timestepAndBias.x;
                row.dampingRatio = 1.0f;
                row.impulseLower =
                    localRow == 0u
                    ? 0.0f
                    : -MR_CONSTRAINT_IR_UNBOUNDED;
                row.impulseUpper =
                    MR_CONSTRAINT_IR_UNBOUNDED;
                row.flags =
                    localRow == 0u
                    ? (
                        MR_CONSTRAINT_IR_ROW_POSITION_STABILIZED |
                        MR_CONSTRAINT_IR_ROW_UNILATERAL |
                        MR_CONSTRAINT_IR_ROW_CONTACT_NORMAL
                    )
                    : MR_CONSTRAINT_IR_ROW_CONTACT_TANGENT;
                rows[
                    rowBase + 3u * currentConstraint + localRow
                ] = row;
            }

            MRConstraintIRConeGPU cone = {};
            cone.staticFrictionU = contact.friction.x;
            cone.staticFrictionV = contact.friction.x;
            cone.dynamicFrictionU = contact.friction.y;
            cone.dynamicFrictionV = contact.friction.y;
            cone.rollingLength = contact.friction.z;
            cone.torsionalLength = contact.friction.w;
            cone.restitution = contact.response.x;
            cone.restitutionThreshold = contact.response.y;
            cone.stictionTransitionVelocity = 1.0e-3f;
            cones[outputConstraint] = cone;

            MRArticulatedPointImpulseGPU queryA = dummyQuery;
            MRArticulatedPointImpulseGPU queryB = dummyQuery;
            if (bodyA.flagsAndIndices[1] ==
                dispatch.articulationIndex) {
                queryA.bodyIndex = sourceA.bodyIndex;
                queryA.localPoint = manifoldPoint.localAnchorA;
            }
            if (bodyB.flagsAndIndices[1] ==
                dispatch.articulationIndex) {
                queryB.bodyIndex = sourceB.bodyIndex;
                queryB.localPoint = manifoldPoint.localAnchorB;
            }
            pointQueries[
                queryBase + 2u * currentConstraint
            ] = queryA;
            pointQueries[
                queryBase + 2u * currentConstraint + 1u
            ] = queryB;
        }
    }

    status.requiredPairs = pairCount;
    status.requiredRawContacts = rawCount;
    status.requiredManifolds = manifoldCount;
    status.requiredConstraints = constraintCount;
    status.requiredRows =
        constraintCount > 0x55555555u
        ? 0xffffffffu
        : 3u * constraintCount;
    status.activePairs = pairCount;
    status.activeContacts = constraintCount;
    status.retainedPoints = retainedPoints;
    status.newPoints = newPoints;
    status.diagnostics.x =
        retainedPoints + newPoints == 0u
        ? 1.0f
        : float(retainedPoints) /
            float(retainedPoints + newPoints);
    status.diagnostics.y = maximumPenetration;
    if (status.code == MR_STEP_SUCCESS) {
        if (pairCount > dispatch.pairCapacity) {
            status.code = MR_STEP_PAIR_CAPACITY_OVERFLOW;
            status.firstFailingPair = firstPairOverflow;
        } else if (rawCount > dispatch.rawContactCapacity) {
            status.code = MR_STEP_CONTACT_CAPACITY_OVERFLOW;
            status.firstFailingPair = firstRawOverflowPair;
        } else if (manifoldCount > dispatch.manifoldCapacity) {
            status.code = MR_STEP_MANIFOLD_CAPACITY_OVERFLOW;
            status.firstFailingPair =
                firstManifoldOverflowPair;
        } else if (constraintCount > dispatch.constraintCapacity ||
                   status.requiredRows > dispatch.rowCapacity) {
            status.code = MR_STEP_CONSTRAINT_CAPACITY_OVERFLOW;
            status.firstFailingConstraint =
                firstConstraintOverflow != MR_INVALID_INDEX
                ? firstConstraintOverflow
                : dispatch.constraintCapacity;
        }
    }
    candidateManifoldCounts[environment] =
        status.code == MR_STEP_SUCCESS ? manifoldCount : 0u;
    statuses[environment] = status;
}
