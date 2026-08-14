#include <metal_stdlib>

#include "metalrobo/engine_types.h"
#include "metalrobo/rod_gpu_shared.h"

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
// Internal support-mapped query shape. It never crosses the public ABI:
// center/halfExtents/capsuleEndpoint{0,1} carry the triangle centroid/C/A/B.
constant uint kQueryTriangleShape = 0xfffffffdu;

bool isSurfaceShapeType(const uint type) {
    return
        type == MR_SHAPE_TRIANGLE_MESH ||
        type == MR_SHAPE_HEIGHTFIELD;
}

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
    float3 scale;
    uint geometryIndex;
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
                max(
                    abs(shape.contactOffset),
                    max(
                        maximumAbsoluteComponent(shape.lower),
                        maximumAbsoluteComponent(shape.upper)
                    )
                )
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
        (shape.flags &
         ~(
             MR_SHAPE_FLAG_SIMULATION_DISABLED |
             MR_SHAPE_FLAG_ENABLE_CCD |
             MR_SHAPE_FLAG_MESH_TWO_SIDED
         )) != 0u ||
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
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const MRBodyStateGPU* bodies,
    const uint bodyCount,
    thread WorldShape& output,
    thread uint& failureCode
) {
    const MRShapeGPU shape = shapes[index];
    if (shape.bodyIndex >= bodyCount ||
        (shape.flags &
         ~(
             MR_SHAPE_FLAG_SIMULATION_DISABLED |
             MR_SHAPE_FLAG_ENABLE_CCD |
             MR_SHAPE_FLAG_MESH_TWO_SIDED
         )) != 0u ||
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
        (shape.flags & MR_SHAPE_FLAG_SIMULATION_DISABLED) != 0u ||
        (body.flagsAndIndices[3] &
         MR_BODY_STATE_COLLISION_DISABLED) != 0u;
    output.rotation =
        quaternionMultiply(bodyRotation, localRotation);
    output.center = body.position.xyz +
        quaternionRotate(bodyRotation, shape.localPosition.xyz);
    output.contactOffset =
        shape.contactRestAndBoundingRadius.x;
    output.scale = shape.dimensions.xyz;
    output.geometryIndex = shape.geometryOffset;
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
        shape.shapeType != MR_SHAPE_PLANE &&
        shape.shapeType != MR_SHAPE_CONVEX &&
        shape.shapeType != MR_SHAPE_TRIANGLE_MESH &&
        shape.shapeType != MR_SHAPE_HEIGHTFIELD) {
        failureCode = MR_STEP_UNSUPPORTED;
        return false;
    }

    if (shape.shapeType == MR_SHAPE_CONVEX ||
        isSurfaceShapeType(shape.shapeType)) {
        if (shape.geometryCount != 1u ||
            !all(shape.dimensions.xyz >=
                 float3(MR_MIN_COLLISION_EXTENT))) {
            failureCode = MR_STEP_NONFINITE_INPUT;
            return false;
        }
        const MRGeometryHeaderGPU geometry =
            geometryHeaders[shape.geometryOffset];
        const uint expectedKind =
            shape.shapeType == MR_SHAPE_CONVEX
            ? MR_GEOMETRY_CONVEX
            : shape.shapeType == MR_SHAPE_TRIANGLE_MESH
            ? MR_GEOMETRY_TRIANGLE_MESH
            : MR_GEOMETRY_HEIGHTFIELD;
        if (geometry.kind != expectedKind ||
            !collisionInputDomainXyz(geometry.localLower) ||
            !collisionInputDomainXyz(geometry.localUpper) ||
            any(geometry.localUpper.xyz <
                geometry.localLower.xyz)) {
            failureCode = MR_STEP_NONFINITE_INPUT;
            return false;
        }
        const float3 localCenter =
            0.5f *
            (geometry.localLower.xyz +
             geometry.localUpper.xyz) *
            shape.dimensions.xyz;
        const float3 localHalf =
            0.5f *
            (geometry.localUpper.xyz -
             geometry.localLower.xyz) *
            shape.dimensions.xyz;
        const float3 boundsCenter =
            output.center +
            quaternionRotate(output.rotation, localCenter);
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
            abs(basisX) * localHalf.x +
            abs(basisY) * localHalf.y +
            abs(basisZ) * localHalf.z +
            output.contactOffset;
        output.lower = boundsCenter - extent;
        output.upper = boundsCenter + extent;
        output.radius =
            shape.contactRestAndBoundingRadius.z;
        inflateFiniteBounds(output);
        if (!collisionDomain(output.lower) ||
            !collisionDomain(output.upper)) {
            failureCode = MR_STEP_NONFINITE_INPUT;
            return false;
        }
        return true;
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
        output.halfLength = halfLength;
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

// Compatibility path for the standalone primitive collision oracle. Cooked
// geometry is executed by the world graph, whose projection kernel binds the
// immutable geometry arena explicitly.
bool makeWorldShape(
    const uint index,
    device const MRShapeGPU* shapes,
    device const MRBodyStateGPU* bodies,
    const uint bodyCount,
    thread WorldShape& output,
    thread uint& failureCode
) {
    if ((shapes[index].flags &
         MR_SHAPE_FLAG_SIMULATION_DISABLED) == 0u &&
        (shapes[index].shapeType == MR_SHAPE_CONVEX ||
         isSurfaceShapeType(shapes[index].shapeType))) {
        failureCode = MR_STEP_UNSUPPORTED;
        return false;
    }
    device const MRGeometryHeaderGPU* emptyGeometry =
        nullptr;
    return makeWorldShape(
        index,
        shapes,
        emptyGeometry,
        bodies,
        bodyCount,
        output,
        failureCode
    );
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
    shape.scale = source.dimensions.xyz;
    shape.geometryIndex = source.geometryOffset;
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

struct ConvexSupportPoint {
    float3 minkowski;
    float3 pointA;
    float3 pointB;
    uint featureA;
    uint featureB;
};

struct ConvexQueryResult {
    uint status;
    uint iterations;
    uint fallback;
    uint reserved;
    float3 normal;
    float separation;
    float3 pointA;
    float3 pointB;
    uint featureA;
    uint featureB;
};

struct EPAFace {
    uint a;
    uint b;
    uint c;
    uint active;
    float3 normal;
    float distance;
};

float3 deterministicDirection(const uint stableKey) {
    const uint axis = stableKey % 3u;
    float3 result = float3(0.0f);
    result[axis] = (stableKey & 4u) == 0u ? 1.0f : -1.0f;
    return result;
}

bool supportWorldShape(
    const thread WorldShape& shape,
    device const MRShapeGPU& source,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    const float3 requestedDirection,
    thread float3& point,
    thread uint& feature
) {
    float3 direction = requestedDirection;
    const float directionSquared = dot(direction, direction);
    if (!(directionSquared > kTiny) ||
        !isfinite(directionSquared)) {
        direction = deterministicDirection(shape.index);
    } else {
        direction *= rsqrt(directionSquared);
    }
    if (shape.type == kQueryTriangleShape) {
        const float3 triangleVertices[3] = {
            shape.capsuleEndpoint0,
            shape.capsuleEndpoint1,
            shape.halfExtents,
        };
        uint bestVertex = 0u;
        float bestProjection = dot(
            triangleVertices[0],
            direction
        );
        for (uint vertexIndex = 1u;
             vertexIndex < 3u;
             ++vertexIndex) {
            const float projection = dot(
                triangleVertices[vertexIndex],
                direction
            );
            if (projection > bestProjection ||
                (projection == bestProjection &&
                 vertexIndex < bestVertex)) {
                bestProjection = projection;
                bestVertex = vertexIndex;
            }
        }
        point = triangleVertices[bestVertex];
        feature = featureKey(
            MR_SHAPE_TRIANGLE_MESH,
            3u * shape.index + bestVertex
        );
        return true;
    }
    if (shape.type == MR_SHAPE_SPHERE) {
        point = shape.center + direction * shape.radius;
        feature = featureKey(MR_SHAPE_SPHERE, 0u);
        return true;
    }
    if (shape.type == MR_SHAPE_CAPSULE) {
        const float projection = dot(
            direction,
            shape.capsuleEndpoint1 -
                shape.capsuleEndpoint0
        );
        const bool positive = projection >= 0.0f;
        point =
            (positive
                 ? shape.capsuleEndpoint1
                 : shape.capsuleEndpoint0) +
            direction * shape.radius;
        feature = featureKey(
            MR_SHAPE_CAPSULE,
            positive ? 1u : 0u
        );
        return true;
    }
    if (shape.type == MR_SHAPE_BOX) {
        const float3 localDirection =
            quaternionInverseRotate(
                shape.rotation,
                direction
            );
        uint vertexIndex = 0u;
        float3 localPoint = -shape.halfExtents;
        if (localDirection.x >= 0.0f) {
            vertexIndex |= 1u;
            localPoint.x = shape.halfExtents.x;
        }
        if (localDirection.y >= 0.0f) {
            vertexIndex |= 2u;
            localPoint.y = shape.halfExtents.y;
        }
        if (localDirection.z >= 0.0f) {
            vertexIndex |= 4u;
            localPoint.z = shape.halfExtents.z;
        }
        point =
            shape.center +
            quaternionRotate(shape.rotation, localPoint);
        feature = featureKey(MR_SHAPE_BOX, vertexIndex);
        return true;
    }
    if (shape.type == MR_SHAPE_CYLINDER) {
        const float axial =
            dot(direction, shape.cylinderAxis);
        const float3 radial =
            direction - shape.cylinderAxis * axial;
        const float radialSquared = dot(radial, radial);
        const float3 radialDirection =
            radialSquared > kTiny
            ? radial * rsqrt(radialSquared)
            : shape.cylinderBasisX;
        const bool positiveCap = axial >= 0.0f;
        point =
            shape.center +
            shape.cylinderAxis *
                (positiveCap
                     ? shape.halfLength
                     : -shape.halfLength) +
            radialDirection * shape.radius;
        uint localFeature =
            positiveCap
            ? kCylinderPositiveGeneralRim
            : kCylinderNegativeGeneralRim;
        if (abs(axial) <= kCylinderAlignmentTolerance) {
            const float basisX = dot(
                radialDirection,
                shape.cylinderBasisX
            );
            const float basisZ = dot(
                radialDirection,
                shape.cylinderBasisZ
            );
            const uint sector =
                abs(basisX) >= abs(basisZ)
                ? (basisX >= 0.0f ? 0u : 1u)
                : (basisZ >= 0.0f ? 2u : 3u);
            localFeature = 32u + sector;
        } else if (radialSquared <= kTiny) {
            localFeature = positiveCap ? 17u : 16u;
        }
        feature = featureKey(
            MR_SHAPE_CYLINDER,
            localFeature
        );
        return finiteFloat3(point);
    }
    if (shape.type == MR_SHAPE_CONVEX) {
        const MRGeometryHeaderGPU geometry =
            geometryHeaders[source.geometryOffset];
        if (geometry.kind != MR_GEOMETRY_CONVEX ||
            geometry.vertexCount == 0u) {
            return false;
        }
        const float3 localDirection =
            quaternionInverseRotate(
                shape.rotation,
                direction
            );
        float bestProjection = -INFINITY;
        uint bestVertex = 0u;
        float3 bestPoint = float3(0.0f);
        for (uint vertexIndex = 0u;
             vertexIndex < geometry.vertexCount;
             ++vertexIndex) {
            const float3 localPoint =
                geometryVertices[
                    geometry.vertexOffset + vertexIndex
                ].xyz * source.dimensions.xyz;
            const float projection =
                dot(localPoint, localDirection);
            if (projection > bestProjection ||
                (projection == bestProjection &&
                 vertexIndex < bestVertex)) {
                bestProjection = projection;
                bestVertex = vertexIndex;
                bestPoint = localPoint;
            }
        }
        point =
            shape.center +
            quaternionRotate(shape.rotation, bestPoint);
        feature = featureKey(MR_SHAPE_CONVEX, bestVertex);
        return finiteFloat3(point);
    }
    return false;
}

bool supportMinkowski(
    const thread WorldShape& shapeA,
    const thread WorldShape& shapeB,
    device const MRShapeGPU& sourceA,
    device const MRShapeGPU& sourceB,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    const float3 direction,
    thread ConvexSupportPoint& result
) {
    if (!supportWorldShape(
            shapeA,
            sourceA,
            geometryHeaders,
            geometryVertices,
            direction,
            result.pointA,
            result.featureA
        ) ||
        !supportWorldShape(
            shapeB,
            sourceB,
            geometryHeaders,
            geometryVertices,
            -direction,
            result.pointB,
            result.featureB
        )) {
        return false;
    }
    result.minkowski = result.pointA - result.pointB;
    return finiteFloat3(result.minkowski);
}

float3 closestTriangleWeights(
    const float3 a,
    const float3 b,
    const float3 c,
    thread float3& weights
) {
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 ap = -a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        weights = float3(1.0f, 0.0f, 0.0f);
        return a;
    }
    const float3 bp = -b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        weights = float3(0.0f, 1.0f, 0.0f);
        return b;
    }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float denominator = d1 - d3;
        const float v = denominator > kTiny
            ? d1 / denominator
            : 0.0f;
        weights = float3(1.0f - v, v, 0.0f);
        return a + ab * v;
    }
    const float3 cp = -c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        weights = float3(0.0f, 0.0f, 1.0f);
        return c;
    }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float denominator = d2 - d6;
        const float w = denominator > kTiny
            ? d2 / denominator
            : 0.0f;
        weights = float3(1.0f - w, 0.0f, w);
        return a + ac * w;
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f &&
        (d4 - d3) >= 0.0f &&
        (d5 - d6) >= 0.0f) {
        const float denominator =
            (d4 - d3) + (d5 - d6);
        const float w = denominator > kTiny
            ? (d4 - d3) / denominator
            : 0.0f;
        weights = float3(0.0f, 1.0f - w, w);
        return b + (c - b) * w;
    }
    const float denominator = va + vb + vc;
    if (!(abs(denominator) > kTiny) ||
        !isfinite(denominator)) {
        weights = float3(1.0f, 0.0f, 0.0f);
        return a;
    }
    const float reciprocal = 1.0f / denominator;
    const float v = vb * reciprocal;
    const float w = vc * reciprocal;
    weights = float3(1.0f - v - w, v, w);
    return a + ab * v + ac * w;
}

float2 exactTwoSum(const float a, const float b) {
    const float sum = a + b;
    const float recoveredB = sum - a;
    const float error =
        (a - (sum - recoveredB)) +
        (b - recoveredB);
    return float2(sum, error);
}

float2 exactTwoProduct(const float a, const float b) {
    const float product = a * b;
    return float2(product, fma(a, b, -product));
}

float2 compensatedDifference(
    const float2 lhs,
    const float2 rhs
) {
    const float2 leading = exactTwoSum(lhs.x, -rhs.x);
    return float2(
        leading.x,
        leading.y + lhs.y - rhs.y
    );
}

float2 compensatedCrossComponent(
    const float a,
    const float b,
    const float c,
    const float d
) {
    return compensatedDifference(
        exactTwoProduct(a, b),
        exactTwoProduct(c, d)
    );
}

float compensatedTripleProduct(
    const float3 a,
    const float3 b,
    const float3 c
) {
    // A twofold FP32 scalar triple product is reserved for ambiguous
    // simplex predicates. FMA captures each product residual, then exact
    // two-sums retain the low components through the final reduction.
    const float2 crossX = compensatedCrossComponent(
        b.y,
        c.z,
        b.z,
        c.y
    );
    const float2 crossY = compensatedCrossComponent(
        b.z,
        c.x,
        b.x,
        c.z
    );
    const float2 crossZ = compensatedCrossComponent(
        b.x,
        c.y,
        b.y,
        c.x
    );
    float2 termX = exactTwoProduct(a.x, crossX.x);
    float2 termY = exactTwoProduct(a.y, crossY.x);
    float2 termZ = exactTwoProduct(a.z, crossZ.x);
    termX.y += a.x * crossX.y;
    termY.y += a.y * crossY.y;
    termZ.y += a.z * crossZ.y;
    const float2 sumXY = exactTwoSum(termX.x, termY.x);
    const float2 sumXYZ = exactTwoSum(sumXY.x, termZ.x);
    return sumXYZ.x +
        (
            sumXY.y +
            sumXYZ.y +
            termX.y +
            termY.y +
            termZ.y
        );
}

void compressSimplex(
    thread ConvexSupportPoint* simplex,
    thread uint& count,
    thread float* weights
) {
    ConvexSupportPoint compact[4];
    float compactWeights[4];
    uint compactCount = 0u;
    uint best = 0u;
    for (uint index = 1u; index < count; ++index) {
        if (weights[index] > weights[best]) {
            best = index;
        }
    }
    for (uint index = 0u; index < count; ++index) {
        if (weights[index] > 1.0e-7f) {
            compact[compactCount] = simplex[index];
            compactWeights[compactCount] = weights[index];
            ++compactCount;
        }
    }
    if (compactCount == 0u) {
        compact[0] = simplex[best];
        compactWeights[0] = 1.0f;
        compactCount = 1u;
    }
    float total = 0.0f;
    for (uint index = 0u; index < compactCount; ++index) {
        total += compactWeights[index];
    }
    const float reciprocal =
        total > kTiny ? 1.0f / total : 1.0f;
    count = compactCount;
    for (uint index = 0u; index < compactCount; ++index) {
        simplex[index] = compact[index];
        weights[index] =
            compactWeights[index] * reciprocal;
    }
}

bool closestSimplex(
    thread ConvexSupportPoint* simplex,
    thread uint& count,
    thread float* weights,
    thread float3& closest
) {
    if (count == 1u) {
        weights[0] = 1.0f;
        closest = simplex[0].minkowski;
        return false;
    }
    if (count == 2u) {
        const float3 a = simplex[0].minkowski;
        const float3 edge =
            simplex[1].minkowski - a;
        const float denominator = dot(edge, edge);
        const float t = denominator > kTiny
            ? clamp(-dot(a, edge) / denominator, 0.0f, 1.0f)
            : 0.0f;
        weights[0] = 1.0f - t;
        weights[1] = t;
        closest = a + edge * t;
        compressSimplex(simplex, count, weights);
        return false;
    }
    if (count == 3u) {
        float3 triangleWeights;
        closest = closestTriangleWeights(
            simplex[0].minkowski,
            simplex[1].minkowski,
            simplex[2].minkowski,
            triangleWeights
        );
        weights[0] = triangleWeights.x;
        weights[1] = triangleWeights.y;
        weights[2] = triangleWeights.z;
        compressSimplex(simplex, count, weights);
        return false;
    }

    const float3 a = simplex[0].minkowski;
    const float3 ab = simplex[1].minkowski - a;
    const float3 ac = simplex[2].minkowski - a;
    const float3 ad = simplex[3].minkowski - a;
    const float determinant =
        compensatedTripleProduct(ab, ac, ad);
    if (abs(determinant) > kTiny &&
        isfinite(determinant)) {
        const float reciprocal = 1.0f / determinant;
        const float u =
            compensatedTripleProduct(-a, ac, ad) *
            reciprocal;
        const float v =
            compensatedTripleProduct(ab, -a, ad) *
            reciprocal;
        const float w =
            compensatedTripleProduct(ab, ac, -a) *
            reciprocal;
        const float base = 1.0f - u - v - w;
        const float tolerance = -2.0e-6f;
        if (base >= tolerance &&
            u >= tolerance &&
            v >= tolerance &&
            w >= tolerance) {
            weights[0] = base;
            weights[1] = u;
            weights[2] = v;
            weights[3] = w;
            closest = float3(0.0f);
            return true;
        }
    }

    const uint3 faces[4] = {
        uint3(0u, 1u, 2u),
        uint3(0u, 3u, 1u),
        uint3(0u, 2u, 3u),
        uint3(1u, 3u, 2u),
    };
    float bestDistance = INFINITY;
    float4 bestWeights = float4(0.0f);
    float3 bestPoint = float3(0.0f);
    for (uint face = 0u; face < 4u; ++face) {
        float3 triangleWeights;
        const float3 point = closestTriangleWeights(
            simplex[faces[face].x].minkowski,
            simplex[faces[face].y].minkowski,
            simplex[faces[face].z].minkowski,
            triangleWeights
        );
        const float distance = dot(point, point);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestPoint = point;
            bestWeights = float4(0.0f);
            bestWeights[faces[face].x] = triangleWeights.x;
            bestWeights[faces[face].y] = triangleWeights.y;
            bestWeights[faces[face].z] = triangleWeights.z;
        }
    }
    closest = bestPoint;
    for (uint index = 0u; index < 4u; ++index) {
        weights[index] = bestWeights[index];
    }
    compressSimplex(simplex, count, weights);
    return false;
}

void simplexWitness(
    thread const ConvexSupportPoint* simplex,
    const uint count,
    thread const float* weights,
    thread float3& pointA,
    thread float3& pointB,
    thread uint& featureA,
    thread uint& featureB
) {
    pointA = float3(0.0f);
    pointB = float3(0.0f);
    uint best = 0u;
    for (uint index = 0u; index < count; ++index) {
        pointA += simplex[index].pointA * weights[index];
        pointB += simplex[index].pointB * weights[index];
        if (weights[index] > weights[best]) {
            best = index;
        }
    }
    featureA = simplex[best].featureA;
    featureB = simplex[best].featureB;
}

bool gjkDistance(
    const thread WorldShape& shapeA,
    const thread WorldShape& shapeB,
    device const MRShapeGPU& sourceA,
    device const MRShapeGPU& sourceB,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    thread ConvexSupportPoint* simplex,
    thread uint& simplexCount,
    thread float* weights,
    const float3 cachedDirection,
    thread ConvexQueryResult& result
) {
    const float queryScale =
        max(worldShapeScale(shapeA), worldShapeScale(shapeB)) +
        1.0f;
    const float tolerance =
        32.0f * 1.1920929e-7f * queryScale;
    float3 direction = cachedDirection;
    if (!finiteFloat3(direction) ||
        dot(direction, direction) <= kTiny) {
        direction = shapeB.center - shapeA.center;
    }
    if (dot(direction, direction) <= kTiny) {
        direction = deterministicDirection(
            shapeA.index * 65537u + shapeB.index
        );
    }
    simplexCount = 1u;
    if (!supportMinkowski(
            shapeA,
            shapeB,
            sourceA,
            sourceB,
            geometryHeaders,
            geometryVertices,
            direction,
            simplex[0]
        )) {
        result.status = MR_STEP_UNSUPPORTED;
        return false;
    }
    weights[0] = 1.0f;
    float3 closest = simplex[0].minkowski;
    float previousDistanceSquared = INFINITY;
    for (uint iteration = 0u;
         iteration < MR_GJK_MAX_ITERATIONS;
         ++iteration) {
        result.iterations = iteration + 1u;
        const float distanceSquared = dot(closest, closest);
        if (distanceSquared <= tolerance * tolerance) {
            result.status = MR_STEP_SUCCESS;
            result.separation = -0.0f;
            return true;
        }
        direction = -closest;
        ConvexSupportPoint candidate;
        if (!supportMinkowski(
                shapeA,
                shapeB,
                sourceA,
                sourceB,
                geometryHeaders,
                geometryVertices,
                direction,
                candidate
            )) {
            result.status = MR_STEP_UNSUPPORTED;
            return false;
        }
        bool duplicate = false;
        for (uint index = 0u;
             index < simplexCount;
             ++index) {
            duplicate =
                duplicate ||
                dot(
                    candidate.minkowski -
                        simplex[index].minkowski,
                    candidate.minkowski -
                        simplex[index].minkowski
                ) <= tolerance * tolerance;
        }
        const float supportAdvance =
            dot(candidate.minkowski, direction) -
            dot(closest, direction);
        if (duplicate ||
            supportAdvance <=
                tolerance * max(length(direction), 1.0f)) {
            simplexWitness(
                simplex,
                simplexCount,
                weights,
                result.pointA,
                result.pointB,
                result.featureA,
                result.featureB
            );
            const float3 witnessDelta =
                result.pointB - result.pointA;
            const float witnessDistance =
                length(witnessDelta);
            result.normal =
                witnessDistance > tolerance
                ? witnessDelta / witnessDistance
                : normalize(direction);
            result.separation = witnessDistance;
            result.status = MR_STEP_SUCCESS;
            return false;
        }
        simplex[simplexCount++] = candidate;
        if (closestSimplex(
                simplex,
                simplexCount,
                weights,
                closest
            )) {
            result.status = MR_STEP_SUCCESS;
            result.separation = -0.0f;
            return true;
        }
        const float updatedDistanceSquared =
            dot(closest, closest);
        if (previousDistanceSquared < INFINITY &&
            previousDistanceSquared - updatedDistanceSquared <=
                tolerance * tolerance) {
            simplexWitness(
                simplex,
                simplexCount,
                weights,
                result.pointA,
                result.pointB,
                result.featureA,
                result.featureB
            );
            const float3 witnessDelta =
                result.pointB - result.pointA;
            const float witnessDistance =
                length(witnessDelta);
            result.normal =
                witnessDistance > tolerance
                ? witnessDelta / witnessDistance
                : normalize(direction);
            result.separation = witnessDistance;
            result.status = MR_STEP_SUCCESS;
            return false;
        }
        previousDistanceSquared = updatedDistanceSquared;
    }
    result.status = MR_STEP_DID_NOT_CONVERGE;
    return false;
}

bool makeEPAFace(
    thread const ConvexSupportPoint* vertices,
    const uint a,
    const uint b,
    const uint c,
    thread EPAFace& face
) {
    float3 normal = cross(
        vertices[b].minkowski -
            vertices[a].minkowski,
        vertices[c].minkowski -
            vertices[a].minkowski
    );
    const float normalSquared = dot(normal, normal);
    if (!(normalSquared > kTiny) ||
        !isfinite(normalSquared)) {
        return false;
    }
    normal *= rsqrt(normalSquared);
    uint second = b;
    uint third = c;
    float distance =
        dot(normal, vertices[a].minkowski);
    if (distance < 0.0f) {
        normal = -normal;
        distance = -distance;
        second = c;
        third = b;
    }
    face.a = a;
    face.b = second;
    face.c = third;
    face.active = 1u;
    face.normal = normal;
    face.distance = distance;
    return isfinite(distance);
}

void addHorizonEdge(
    thread uint* edgeA,
    thread uint* edgeB,
    thread uint& edgeCount,
    const uint a,
    const uint b
) {
    for (uint edge = 0u; edge < edgeCount; ++edge) {
        if (edgeA[edge] == b && edgeB[edge] == a) {
            --edgeCount;
            edgeA[edge] = edgeA[edgeCount];
            edgeB[edge] = edgeB[edgeCount];
            return;
        }
    }
    if (edgeCount < 3u * MR_EPA_FACE_CAPACITY) {
        edgeA[edgeCount] = a;
        edgeB[edgeCount] = b;
        ++edgeCount;
    }
}

bool epaPenetration(
    const thread WorldShape& shapeA,
    const thread WorldShape& shapeB,
    device const MRShapeGPU& sourceA,
    device const MRShapeGPU& sourceB,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    thread ConvexSupportPoint* seed,
    thread uint& seedCount,
    thread ConvexQueryResult& result
) {
    ConvexSupportPoint vertices[MR_EPA_VERTEX_CAPACITY];
    uint vertexCount = seedCount;
    for (uint index = 0u; index < seedCount; ++index) {
        vertices[index] = seed[index];
    }
    const float3 seedDirections[6] = {
        float3(1.0f, 0.0f, 0.0f),
        float3(0.0f, 1.0f, 0.0f),
        float3(0.0f, 0.0f, 1.0f),
        float3(-1.0f, 0.0f, 0.0f),
        float3(0.0f, -1.0f, 0.0f),
        float3(0.0f, 0.0f, -1.0f),
    };
    for (uint directionIndex = 0u;
         vertexCount < 4u && directionIndex < 6u;
         ++directionIndex) {
        ConvexSupportPoint candidate;
        if (!supportMinkowski(
                shapeA,
                shapeB,
                sourceA,
                sourceB,
                geometryHeaders,
                geometryVertices,
                seedDirections[directionIndex],
                candidate
            )) {
            return false;
        }
        bool duplicate = false;
        for (uint existing = 0u;
             existing < vertexCount;
             ++existing) {
            duplicate =
                duplicate ||
                dot(
                    candidate.minkowski -
                        vertices[existing].minkowski,
                    candidate.minkowski -
                        vertices[existing].minkowski
                ) <= 1.0e-12f;
        }
        if (!duplicate) {
            vertices[vertexCount++] = candidate;
        }
    }
    if (vertexCount < 4u) {
        return false;
    }

    EPAFace faces[MR_EPA_FACE_CAPACITY];
    for (uint face = 0u;
         face < MR_EPA_FACE_CAPACITY;
         ++face) {
        faces[face].active = 0u;
    }
    const uint3 initialFaces[4] = {
        uint3(0u, 1u, 2u),
        uint3(0u, 3u, 1u),
        uint3(0u, 2u, 3u),
        uint3(1u, 3u, 2u),
    };
    uint faceCount = 0u;
    for (uint face = 0u; face < 4u; ++face) {
        if (makeEPAFace(
                vertices,
                initialFaces[face].x,
                initialFaces[face].y,
                initialFaces[face].z,
                faces[faceCount]
            )) {
            ++faceCount;
        }
    }
    if (faceCount < 4u) {
        return false;
    }

    const float queryScale =
        max(worldShapeScale(shapeA), worldShapeScale(shapeB)) +
        1.0f;
    const float tolerance =
        64.0f * 1.1920929e-7f * queryScale;
    uint bestFace = MR_INVALID_INDEX;
    for (uint iteration = 0u;
         iteration < MR_EPA_MAX_ITERATIONS;
         ++iteration) {
        float bestDistance = INFINITY;
        bestFace = MR_INVALID_INDEX;
        for (uint face = 0u; face < faceCount; ++face) {
            if (faces[face].active != 0u &&
                (faces[face].distance < bestDistance ||
                 (faces[face].distance == bestDistance &&
                  face < bestFace))) {
                bestDistance = faces[face].distance;
                bestFace = face;
            }
        }
        if (bestFace == MR_INVALID_INDEX) {
            return false;
        }
        ConvexSupportPoint candidate;
        if (!supportMinkowski(
                shapeA,
                shapeB,
                sourceA,
                sourceB,
                geometryHeaders,
                geometryVertices,
                faces[bestFace].normal,
                candidate
            )) {
            return false;
        }
        const float supportDistance = dot(
            faces[bestFace].normal,
            candidate.minkowski
        );
        bool duplicate = false;
        for (uint vertexIndex = 0u;
             vertexIndex < vertexCount;
             ++vertexIndex) {
            duplicate =
                duplicate ||
                dot(
                    candidate.minkowski -
                        vertices[vertexIndex].minkowski,
                    candidate.minkowski -
                        vertices[vertexIndex].minkowski
                ) <= tolerance * tolerance;
        }
        if (duplicate ||
            supportDistance - bestDistance <= tolerance) {
            const EPAFace face = faces[bestFace];
            float3 barycentric;
            closestTriangleWeights(
                vertices[face.a].minkowski,
                vertices[face.b].minkowski,
                vertices[face.c].minkowski,
                barycentric
            );
            result.pointA =
                vertices[face.a].pointA * barycentric.x +
                vertices[face.b].pointA * barycentric.y +
                vertices[face.c].pointA * barycentric.z;
            result.pointB =
                vertices[face.a].pointB * barycentric.x +
                vertices[face.b].pointB * barycentric.y +
                vertices[face.c].pointB * barycentric.z;
            uint featureVertex = face.a;
            if (barycentric.y > barycentric.x) {
                featureVertex = face.b;
            }
            if (barycentric.z >
                (featureVertex == face.a
                     ? barycentric.x
                     : barycentric.y)) {
                featureVertex = face.c;
            }
            result.featureA =
                vertices[featureVertex].featureA;
            result.featureB =
                vertices[featureVertex].featureB;
            result.normal = -face.normal;
            result.separation = -bestDistance;
            result.status = MR_STEP_SUCCESS;
            // Distinguish the expensive certified penetration path from
            // ordinary cached GJK and the conservative MPR witness.
            result.fallback = 3u;
            result.iterations += iteration + 1u;
            return
                finiteFloat3(result.pointA) &&
                finiteFloat3(result.pointB) &&
                finiteFloat3(result.normal) &&
                isfinite(result.separation);
        }
        if (vertexCount >= MR_EPA_VERTEX_CAPACITY) {
            return false;
        }
        const uint newVertex = vertexCount++;
        vertices[newVertex] = candidate;
        uint edgeA[3u * MR_EPA_FACE_CAPACITY];
        uint edgeB[3u * MR_EPA_FACE_CAPACITY];
        uint edgeCount = 0u;
        for (uint face = 0u; face < faceCount; ++face) {
            if (faces[face].active == 0u ||
                dot(
                    faces[face].normal,
                    candidate.minkowski -
                        vertices[faces[face].a].minkowski
                ) <= tolerance) {
                continue;
            }
            addHorizonEdge(
                edgeA,
                edgeB,
                edgeCount,
                faces[face].a,
                faces[face].b
            );
            addHorizonEdge(
                edgeA,
                edgeB,
                edgeCount,
                faces[face].b,
                faces[face].c
            );
            addHorizonEdge(
                edgeA,
                edgeB,
                edgeCount,
                faces[face].c,
                faces[face].a
            );
            faces[face].active = 0u;
        }
        if (edgeCount == 0u) {
            return false;
        }
        for (uint edge = 0u; edge < edgeCount; ++edge) {
            uint destination = MR_INVALID_INDEX;
            for (uint face = 0u;
                 face < faceCount;
                 ++face) {
                if (faces[face].active == 0u) {
                    destination = face;
                    break;
                }
            }
            if (destination == MR_INVALID_INDEX) {
                if (faceCount >= MR_EPA_FACE_CAPACITY) {
                    return false;
                }
                destination = faceCount++;
            }
            if (!makeEPAFace(
                    vertices,
                    edgeA[edge],
                    edgeB[edge],
                    newVertex,
                    faces[destination]
                )) {
                faces[destination].active = 0u;
            }
        }
    }
    return false;
}

float3 shapeInteriorPoint(
    const thread WorldShape& shape,
    device const MRShapeGPU& source,
    device const MRGeometryHeaderGPU* geometryHeaders
) {
    if (shape.type != MR_SHAPE_CONVEX) {
        return shape.center;
    }
    const MRGeometryHeaderGPU geometry =
        geometryHeaders[source.geometryOffset];
    const float3 localInterior =
        0.5f *
        (geometry.localLower.xyz + geometry.localUpper.xyz) *
        source.dimensions.xyz;
    return shape.center +
        quaternionRotate(shape.rotation, localInterior);
}

bool normalizedPortalDirection(
    thread const ConvexSupportPoint* portal,
    thread float3& direction
) {
    direction = cross(
        portal[2].minkowski - portal[1].minkowski,
        portal[3].minkowski - portal[1].minkowski
    );
    const float squared = dot(direction, direction);
    if (!(squared > kTiny) || !isfinite(squared)) {
        return false;
    }
    direction *= rsqrt(squared);
    return true;
}

void expandMPRPortal(
    thread ConvexSupportPoint* portal,
    const thread ConvexSupportPoint& candidate
) {
    const float3 candidateCrossCenter = cross(
        candidate.minkowski,
        portal[0].minkowski
    );
    if (dot(
            portal[1].minkowski,
            candidateCrossCenter
        ) > 0.0f) {
        if (dot(
                portal[2].minkowski,
                candidateCrossCenter
            ) > 0.0f) {
            portal[1] = candidate;
        } else {
            portal[3] = candidate;
        }
    } else if (dot(
                   portal[3].minkowski,
                   candidateCrossCenter
               ) > 0.0f) {
        portal[2] = candidate;
    } else {
        portal[1] = candidate;
    }
}

bool mprPortalWitness(
    const thread WorldShape& shapeA,
    const thread WorldShape& shapeB,
    thread const ConvexSupportPoint* portal,
    thread ConvexQueryResult& result
) {
    float3 weights;
    const float3 closest = closestTriangleWeights(
        portal[1].minkowski,
        portal[2].minkowski,
        portal[3].minkowski,
        weights
    );
    result.pointA =
        portal[1].pointA * weights.x +
        portal[2].pointA * weights.y +
        portal[3].pointA * weights.z;
    result.pointB =
        portal[1].pointB * weights.x +
        portal[2].pointB * weights.y +
        portal[3].pointB * weights.z;
    uint featureVertex = 1u;
    float featureWeight = weights.x;
    if (weights.y > featureWeight) {
        featureVertex = 2u;
        featureWeight = weights.y;
    }
    if (weights.z > featureWeight) {
        featureVertex = 3u;
    }
    result.featureA = portal[featureVertex].featureA;
    result.featureB = portal[featureVertex].featureB;

    float3 direction = closest;
    float distance = length(direction);
    if (!(distance > kTiny)) {
        direction = shapeB.center - shapeA.center;
        distance = length(direction);
    }
    if (!(distance > kTiny)) {
        direction = deterministicDirection(
            shapeA.index * 65537u + shapeB.index
        );
    } else {
        direction /= distance;
    }
    const float centerOrientation =
        dot(direction, shapeB.center - shapeA.center);
    if (centerOrientation < 0.0f) {
        direction = -direction;
    }
    float separation = dot(
        result.pointB - result.pointA,
        direction
    );
    if (separation > 0.0f) {
        direction = -direction;
        separation = -separation;
    }
    result.status = MR_STEP_SUCCESS;
    result.fallback = 1u;
    result.normal = direction;
    // This path is entered only after an ambiguous/overlap GJK result.
    // Never turn a capacity/degeneracy fallback into silent separation.
    result.separation = min(separation, -0.0f);
    return
        finiteFloat3(result.pointA) &&
        finiteFloat3(result.pointB) &&
        finiteFloat3(result.normal) &&
        isfinite(result.separation);
}

bool mprConservativeWitness(
    const thread WorldShape& shapeA,
    const thread WorldShape& shapeB,
    device const MRShapeGPU& sourceA,
    device const MRShapeGPU& sourceB,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    thread ConvexQueryResult& result
) {
    ConvexSupportPoint portal[4];
    portal[0] = {};
    portal[0].pointA = shapeInteriorPoint(
        shapeA,
        sourceA,
        geometryHeaders
    );
    portal[0].pointB = shapeInteriorPoint(
        shapeB,
        sourceB,
        geometryHeaders
    );
    portal[0].minkowski =
        portal[0].pointA - portal[0].pointB;
    const float queryScale =
        max(worldShapeScale(shapeA), worldShapeScale(shapeB)) +
        1.0f;
    const float tolerance =
        64.0f * 1.1920929e-7f * queryScale;
    if (dot(
            portal[0].minkowski,
            portal[0].minkowski
        ) <= tolerance * tolerance) {
        portal[0].minkowski +=
            deterministicDirection(
                shapeA.index * 65537u + shapeB.index
            ) * tolerance;
    }

    float3 direction = -portal[0].minkowski;
    direction *= rsqrt(dot(direction, direction));
    if (!supportMinkowski(
            shapeA,
            shapeB,
            sourceA,
            sourceB,
            geometryHeaders,
            geometryVertices,
            direction,
            portal[1]
        ) ||
        dot(portal[1].minkowski, direction) < -tolerance) {
        return false;
    }
    direction = cross(
        portal[0].minkowski,
        portal[1].minkowski
    );
    float directionSquared = dot(direction, direction);
    if (!(directionSquared > tolerance * tolerance)) {
        portal[2] = portal[1];
        portal[3] = portal[1];
        return mprPortalWitness(
            shapeA,
            shapeB,
            portal,
            result
        );
    }
    direction *= rsqrt(directionSquared);
    if (!supportMinkowski(
            shapeA,
            shapeB,
            sourceA,
            sourceB,
            geometryHeaders,
            geometryVertices,
            direction,
            portal[2]
        ) ||
        dot(portal[2].minkowski, direction) < -tolerance) {
        return false;
    }

    direction = cross(
        portal[1].minkowski - portal[0].minkowski,
        portal[2].minkowski - portal[0].minkowski
    );
    directionSquared = dot(direction, direction);
    if (!(directionSquared > tolerance * tolerance)) {
        portal[3] = portal[2];
        return mprPortalWitness(
            shapeA,
            shapeB,
            portal,
            result
        );
    }
    direction *= rsqrt(directionSquared);
    if (dot(direction, portal[0].minkowski) > 0.0f) {
        const ConvexSupportPoint swap = portal[1];
        portal[1] = portal[2];
        portal[2] = swap;
        direction = -direction;
    }

    bool discovered = false;
    for (uint iteration = 0u;
         iteration < MR_GJK_MAX_ITERATIONS;
         ++iteration) {
        ++result.iterations;
        if (!supportMinkowski(
                shapeA,
                shapeB,
                sourceA,
                sourceB,
                geometryHeaders,
                geometryVertices,
                direction,
                portal[3]
            ) ||
            dot(portal[3].minkowski, direction) <
                -tolerance) {
            return false;
        }
        const float side13 = dot(
            cross(
                portal[1].minkowski,
                portal[3].minkowski
            ),
            portal[0].minkowski
        );
        if (side13 < -tolerance) {
            portal[2] = portal[3];
        } else {
            const float side32 = dot(
                cross(
                    portal[3].minkowski,
                    portal[2].minkowski
                ),
                portal[0].minkowski
            );
            if (side32 < -tolerance) {
                portal[1] = portal[3];
            } else {
                discovered = true;
                break;
            }
        }
        direction = cross(
            portal[1].minkowski -
                portal[0].minkowski,
            portal[2].minkowski -
                portal[0].minkowski
        );
        directionSquared = dot(direction, direction);
        if (!(directionSquared > tolerance * tolerance)) {
            portal[3] = portal[2];
            return mprPortalWitness(
                shapeA,
                shapeB,
                portal,
                result
            );
        }
        direction *= rsqrt(directionSquared);
    }
    if (!discovered) {
        return false;
    }

    for (uint iteration = 0u;
         iteration < MR_EPA_MAX_ITERATIONS;
         ++iteration) {
        ++result.iterations;
        if (!normalizedPortalDirection(
                portal,
                direction
            )) {
            return mprPortalWitness(
                shapeA,
                shapeB,
                portal,
                result
            );
        }
        ConvexSupportPoint candidate;
        if (!supportMinkowski(
                shapeA,
                shapeB,
                sourceA,
                sourceB,
                geometryHeaders,
                geometryVertices,
                direction,
                candidate
            )) {
            return false;
        }
        const float candidateProjection =
            dot(candidate.minkowski, direction);
        const float portalProjection = min(
            dot(portal[1].minkowski, direction),
            min(
                dot(portal[2].minkowski, direction),
                dot(portal[3].minkowski, direction)
            )
        );
        if (candidateProjection - portalProjection <=
                tolerance ||
            iteration + 1u == MR_EPA_MAX_ITERATIONS) {
            return mprPortalWitness(
                shapeA,
                shapeB,
                portal,
                result
            );
        }
        expandMPRPortal(portal, candidate);
    }
    return false;
}

bool updateSupportAxisPenetration(
    const thread WorldShape& shapeA,
    const thread WorldShape& shapeB,
    device const MRShapeGPU& sourceA,
    device const MRShapeGPU& sourceB,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    const float3 centerDelta,
    const float3 requestedAxis,
    thread bool& found,
    thread ConvexQueryResult& result
) {
    const float axisSquared = dot(requestedAxis, requestedAxis);
    if (!(axisSquared > kTiny) || !isfinite(axisSquared)) {
        return true;
    }
    float3 axis = requestedAxis * rsqrt(axisSquared);
    if (dot(centerDelta, centerDelta) > kTiny &&
        dot(axis, centerDelta) < 0.0f) {
        axis = -axis;
    }
    float3 pointA;
    float3 pointB;
    uint featureA = 0u;
    uint featureB = 0u;
    if (!supportWorldShape(
            shapeA,
            sourceA,
            geometryHeaders,
            geometryVertices,
            axis,
            pointA,
            featureA
        ) ||
        !supportWorldShape(
            shapeB,
            sourceB,
            geometryHeaders,
            geometryVertices,
            -axis,
            pointB,
            featureB
        )) {
        return false;
    }
    const float separation = dot(pointB - pointA, axis);
    if (!isfinite(separation)) {
        return false;
    }
    if (!found || separation > result.separation) {
        found = true;
        result.normal = axis;
        result.separation = separation;
        result.pointA = pointA;
        result.pointB = pointB;
        result.featureA = featureA;
        result.featureB = featureB;
    }
    return true;
}

// Bounded online penetration witness for authored locomotion hulls. GJK has
// already established overlap before this function is entered. Sampling the
// center/cached directions and both local frames yields stable support points
// and a shallow separating-axis estimate without MPR/EPA's private portal
// state in the high-throughput Apple-GPU kernel.
bool supportAxisPenetrationWitness(
    const thread WorldShape& shapeA,
    const thread WorldShape& shapeB,
    device const MRShapeGPU& sourceA,
    device const MRShapeGPU& sourceB,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    const float3 cachedDirection,
    thread ConvexQueryResult& result
) {
    const float3 centerDelta = shapeB.center - shapeA.center;
    bool found = false;
    if (!updateSupportAxisPenetration(
            shapeA,
            shapeB,
            sourceA,
            sourceB,
            geometryHeaders,
            geometryVertices,
            centerDelta,
            centerDelta,
            found,
            result
        ) ||
        !updateSupportAxisPenetration(
            shapeA,
            shapeB,
            sourceA,
            sourceB,
            geometryHeaders,
            geometryVertices,
            centerDelta,
            cachedDirection,
            found,
            result
        )) {
        return false;
    }
    const float3 basis[3] = {
        float3(1.0f, 0.0f, 0.0f),
        float3(0.0f, 1.0f, 0.0f),
        float3(0.0f, 0.0f, 1.0f),
    };
    for (uint axis = 0u; axis < 3u; ++axis) {
        if (!updateSupportAxisPenetration(
                shapeA,
                shapeB,
                sourceA,
                sourceB,
                geometryHeaders,
                geometryVertices,
                centerDelta,
                quaternionRotate(shapeA.rotation, basis[axis]),
                found,
                result
            ) ||
            !updateSupportAxisPenetration(
                shapeA,
                shapeB,
                sourceA,
                sourceB,
                geometryHeaders,
                geometryVertices,
                centerDelta,
                quaternionRotate(shapeB.rotation, basis[axis]),
                found,
                result
            )) {
            return false;
        }
    }
    if (!found) {
        return false;
    }
    result.status = MR_STEP_SUCCESS;
    result.fallback = 2u;
    result.separation = min(result.separation, -0.0f);
    return
        finiteFloat3(result.pointA) &&
        finiteFloat3(result.pointB) &&
        finiteFloat3(result.normal) &&
        isfinite(result.separation);
}

float3 contactPatchTangent(const float3 normal) {
    const float3 absoluteNormal = abs(normal);
    const float3 reference =
        absoluteNormal.x <= absoluteNormal.y &&
            absoluteNormal.x <= absoluteNormal.z
        ? float3(1.0f, 0.0f, 0.0f)
        : absoluteNormal.y <= absoluteNormal.z
            ? float3(0.0f, 1.0f, 0.0f)
            : float3(0.0f, 0.0f, 1.0f);
    return normalize(cross(normal, reference));
}

void appendSupportPatchPoint(
    thread ContactBatch& result,
    const float3 normal,
    const float separation,
    const float3 pointA,
    const float3 pointB,
    const uint featureA,
    const uint featureB,
    const float duplicateToleranceSquared
) {
    for (uint index = 0u; index < result.count; ++index) {
        const float3 deltaA =
            result.contacts[index].pointAWorld.xyz - pointA;
        const float3 deltaB =
            result.contacts[index].pointBWorld.xyz - pointB;
        if (dot(deltaA, deltaA) <=
                duplicateToleranceSquared &&
            dot(deltaB, deltaB) <=
                duplicateToleranceSquared) {
            return;
        }
    }
    if (result.count >=
        MR_METAL_WORLD_RAW_CONTACTS_PER_PAIR) {
        return;
    }
    result.contacts[result.count++] = makeContact(
        normal,
        separation,
        pointA,
        pointB,
        featureA,
        featureB
    );
}

void appendSupportMappedPatch(
    const thread WorldShape& shapeA,
    const thread WorldShape& shapeB,
    device const MRShapeGPU& sourceA,
    device const MRShapeGPU& sourceB,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    const thread ConvexQueryResult& query,
    thread ContactBatch& result
) {
    const bool cylinderA = shapeA.type == MR_SHAPE_CYLINDER;
    const bool cylinderB = shapeB.type == MR_SHAPE_CYLINDER;
    const bool convexA = shapeA.type == MR_SHAPE_CONVEX;
    const bool convexB = shapeB.type == MR_SHAPE_CONVEX;
    if (!cylinderA && !cylinderB && !convexA && !convexB) {
        return;
    }
    const float queryScale =
        max(worldShapeScale(shapeA), worldShapeScale(shapeB)) +
        1.0f;
    const float duplicateTolerance =
        32.0f * 1.1920929e-7f * queryScale;
    const float duplicateToleranceSquared =
        duplicateTolerance * duplicateTolerance;
    const float perturbation = 2.0e-4f;
    const bool sampleA =
        cylinderA || (!cylinderB && convexA);
    const thread WorldShape& sampledShape =
        sampleA ? shapeA : shapeB;
    device const MRShapeGPU& sampledSource =
        sampleA ? sourceA : sourceB;
    const float3 supportNormal =
        sampleA ? query.normal : -query.normal;

    float3 sampleDirections[4];
    uint sampleCount = 0u;
    if (sampledShape.type == MR_SHAPE_CYLINDER) {
        const float axialAlignment = abs(
            dot(supportNormal, sampledShape.cylinderAxis)
        );
        if (axialAlignment >= 0.95f) {
            sampleDirections[0] =
                sampledShape.cylinderBasisX;
            sampleDirections[1] =
                -sampledShape.cylinderBasisX;
            sampleDirections[2] =
                sampledShape.cylinderBasisZ;
            sampleDirections[3] =
                -sampledShape.cylinderBasisZ;
            sampleCount = 4u;
        } else if (axialAlignment <= 0.20f) {
            sampleDirections[0] =
                sampledShape.cylinderAxis;
            sampleDirections[1] =
                -sampledShape.cylinderAxis;
            sampleCount = 2u;
        }
    } else {
        const float3 tangent =
            contactPatchTangent(query.normal);
        const float3 bitangent =
            cross(query.normal, tangent);
        sampleDirections[0] = tangent;
        sampleDirections[1] = -tangent;
        sampleDirections[2] = bitangent;
        sampleDirections[3] = -bitangent;
        sampleCount = 4u;
    }

    for (uint sample = 0u;
         sample < sampleCount;
         ++sample) {
        float3 point;
        uint feature = 0u;
        if (!supportWorldShape(
                sampledShape,
                sampledSource,
                geometryHeaders,
                geometryVertices,
                supportNormal +
                    perturbation * sampleDirections[sample],
                point,
                feature
            )) {
            continue;
        }
        const float3 pointA =
            sampleA
            ? point
            : point - query.normal * query.separation;
        const float3 pointB =
            sampleA
            ? point + query.normal * query.separation
            : point;
        appendSupportPatchPoint(
            result,
            query.normal,
            query.separation,
            pointA,
            pointB,
            sampleA ? feature : query.featureA,
            sampleA ? query.featureB : feature,
            duplicateToleranceSquared
        );
    }
}

ContactBatch supportMappedContacts(
    const uint colliderA,
    const uint colliderB,
    const thread WorldShape& shapeA,
    const thread WorldShape& shapeB,
    device const MRShapeGPU& sourceA,
    device const MRShapeGPU& sourceB,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    const float3 cachedDirection,
    thread ConvexQueryResult& query
) {
    ContactBatch result = {};
    const float acceptedContactDistance =
        shapeA.contactOffset +
        shapeB.contactOffset +
        pairQueryPadding(shapeA, shapeB);
    if (shapeA.type == MR_SHAPE_PLANE ||
        shapeB.type == MR_SHAPE_PLANE) {
        const thread WorldShape& plane =
            shapeA.type == MR_SHAPE_PLANE ? shapeA : shapeB;
        const thread WorldShape& finiteShape =
            shapeA.type == MR_SHAPE_PLANE ? shapeB : shapeA;
        device const MRShapeGPU& finiteSource =
            shapeA.type == MR_SHAPE_PLANE ? sourceB : sourceA;
        float3 surface;
        uint feature = 0u;
        if (!supportWorldShape(
                finiteShape,
                finiteSource,
                geometryHeaders,
                geometryVertices,
                -plane.planeNormal,
                surface,
                feature
            )) {
            query.status = MR_STEP_UNSUPPORTED;
            return result;
        }
        const float separation = dot(
            plane.planeNormal,
            surface - plane.center
        );
        query.status = MR_STEP_SUCCESS;
        query.normal =
            colliderA == plane.index
            ? plane.planeNormal
            : -plane.planeNormal;
        query.separation = separation;
        if (separation <= acceptedContactDistance) {
            appendFinitePlaneContact(
                result,
                colliderA,
                plane,
                surface,
                separation,
                feature
            );
            query.pointA = result.contacts[0].pointAWorld.xyz;
            query.pointB = result.contacts[0].pointBWorld.xyz;
            query.featureA =
                result.contacts[0].featureAndFlags[0];
            query.featureB =
                result.contacts[0].featureAndFlags[1];
        }
        return result;
    }

    ConvexSupportPoint simplex[4];
    float weights[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint simplexCount = 0u;
    const bool overlap = gjkDistance(
        shapeA,
        shapeB,
        sourceA,
        sourceB,
        geometryHeaders,
        geometryVertices,
        simplex,
        simplexCount,
        weights,
        cachedDirection,
        query
    );
    if (query.status != MR_STEP_SUCCESS &&
        query.status != MR_STEP_DID_NOT_CONVERGE) {
        return result;
    }
    if (overlap) {
        const bool hullPair =
            shapeA.type == MR_SHAPE_CONVEX ||
            shapeB.type == MR_SHAPE_CONVEX;
        // MPR carries a four-point portal and is the compact online path for
        // authored hull overlap. EPA's 64-vertex/128-face workspace is a
        // useful precision fallback, but entering it first for every limb
        // contact creates large per-lane spills and poor occupancy in
        // humanoid batches. Primitive pairs retain EPA-first resolution.
        const bool resolved =
            hullPair
            ? (
                mprConservativeWitness(
                    shapeA,
                    shapeB,
                    sourceA,
                    sourceB,
                    geometryHeaders,
                    geometryVertices,
                    query
                ) ||
                epaPenetration(
                    shapeA,
                    shapeB,
                    sourceA,
                    sourceB,
                    geometryHeaders,
                    geometryVertices,
                    simplex,
                    simplexCount,
                    query
                )
            )
            : (
                epaPenetration(
                    shapeA,
                    shapeB,
                    sourceA,
                    sourceB,
                    geometryHeaders,
                    geometryVertices,
                    simplex,
                    simplexCount,
                    query
                ) ||
                mprConservativeWitness(
                    shapeA,
                    shapeB,
                    sourceA,
                    sourceB,
                    geometryHeaders,
                    geometryVertices,
                    query
                )
            );
        if (!resolved) {
            query.status = MR_STEP_DID_NOT_CONVERGE;
            return result;
        }
    } else if (query.status == MR_STEP_DID_NOT_CONVERGE) {
        if (!mprConservativeWitness(
                shapeA,
                shapeB,
                sourceA,
                sourceB,
                geometryHeaders,
                geometryVertices,
                query
            )) {
            return result;
        }
    }
    if (query.separation <= acceptedContactDistance) {
        result.contacts[0] = makeContact(
            query.normal,
            query.separation,
            query.pointA,
            query.pointB,
            query.featureA,
            query.featureB
        );
        result.count = 1u;
        appendSupportMappedPatch(
            shapeA,
            shapeB,
            sourceA,
            sourceB,
            geometryHeaders,
            geometryVertices,
            query,
            result
        );
    }
    return result;
}

// Online articulated meshes use compact authored convex hulls. Keep their
// penetration path in a separate call graph so Metal does not reserve EPA's
// large per-lane vertex, face, and horizon workspaces for every humanoid
// self-collision query. The bounded support-axis witness retains only a few
// directions per lane, which keeps this high-throughput kernel resident.
ContactBatch supportMappedHullContacts(
    const uint colliderA,
    const uint colliderB,
    const thread WorldShape& shapeA,
    const thread WorldShape& shapeB,
    device const MRShapeGPU& sourceA,
    device const MRShapeGPU& sourceB,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    const float3 cachedDirection,
    thread ConvexQueryResult& query
) {
    ContactBatch result = {};
    const float acceptedContactDistance =
        shapeA.contactOffset +
        shapeB.contactOffset +
        pairQueryPadding(shapeA, shapeB);
    if (shapeA.type == MR_SHAPE_PLANE ||
        shapeB.type == MR_SHAPE_PLANE) {
        const thread WorldShape& plane =
            shapeA.type == MR_SHAPE_PLANE ? shapeA : shapeB;
        const thread WorldShape& finiteShape =
            shapeA.type == MR_SHAPE_PLANE ? shapeB : shapeA;
        device const MRShapeGPU& finiteSource =
            shapeA.type == MR_SHAPE_PLANE ? sourceB : sourceA;
        float3 surface;
        uint feature = 0u;
        if (!supportWorldShape(
                finiteShape,
                finiteSource,
                geometryHeaders,
                geometryVertices,
                -plane.planeNormal,
                surface,
                feature
            )) {
            query.status = MR_STEP_UNSUPPORTED;
            return result;
        }
        const float separation = dot(
            plane.planeNormal,
            surface - plane.center
        );
        query.status = MR_STEP_SUCCESS;
        query.normal =
            colliderA == plane.index
            ? plane.planeNormal
            : -plane.planeNormal;
        query.separation = separation;
        if (separation <= acceptedContactDistance) {
            appendFinitePlaneContact(
                result,
                colliderA,
                plane,
                surface,
                separation,
                feature
            );
            query.pointA = result.contacts[0].pointAWorld.xyz;
            query.pointB = result.contacts[0].pointBWorld.xyz;
            query.featureA =
                result.contacts[0].featureAndFlags[0];
            query.featureB =
                result.contacts[0].featureAndFlags[1];
        }
        return result;
    }

    ConvexSupportPoint simplex[4];
    float weights[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint simplexCount = 0u;
    const bool overlap = gjkDistance(
        shapeA,
        shapeB,
        sourceA,
        sourceB,
        geometryHeaders,
        geometryVertices,
        simplex,
        simplexCount,
        weights,
        cachedDirection,
        query
    );
    if (query.status != MR_STEP_SUCCESS &&
        query.status != MR_STEP_DID_NOT_CONVERGE) {
        return result;
    }
    if ((overlap || query.status == MR_STEP_DID_NOT_CONVERGE) &&
        !supportAxisPenetrationWitness(
            shapeA,
            shapeB,
            sourceA,
            sourceB,
            geometryHeaders,
            geometryVertices,
            cachedDirection,
            query
        )) {
        query.status = MR_STEP_DID_NOT_CONVERGE;
        return result;
    }
    if (query.separation <= acceptedContactDistance) {
        result.contacts[0] = makeContact(
            query.normal,
            query.separation,
            query.pointA,
            query.pointB,
            query.featureA,
            query.featureB
        );
        result.count = 1u;
        appendSupportMappedPatch(
            shapeA,
            shapeB,
            sourceA,
            sourceB,
            geometryHeaders,
            geometryVertices,
            query,
            result
        );
    }
    return result;
}

uint uint4Component(const uint4 value, const uint index) {
    return index == 0u ? value.x :
        index == 1u ? value.y :
        index == 2u ? value.z : value.w;
}

void finiteBoundsInMeshLocal(
    const thread WorldShape& finiteShape,
    const thread WorldShape& mesh,
    thread float3& lower,
    thread float3& upper
) {
    lower = float3(INFINITY);
    upper = float3(-INFINITY);
    for (uint corner = 0u; corner < 8u; ++corner) {
        const float3 world = float3(
            (corner & 1u) != 0u
                ? finiteShape.upper.x
                : finiteShape.lower.x,
            (corner & 2u) != 0u
                ? finiteShape.upper.y
                : finiteShape.lower.y,
            (corner & 4u) != 0u
                ? finiteShape.upper.z
                : finiteShape.lower.z
        );
        const float3 local =
            quaternionInverseRotate(
                mesh.rotation,
                world - mesh.center
            ) / mesh.scale;
        lower = min(lower, local);
        upper = max(upper, local);
    }
}

float3 dequantizedMeshBound(
    const uint4 quantized,
    const MRGeometryHeaderGPU geometry
) {
    const float3 normalized =
        float3(quantized.xyz) * (1.0f / 65535.0f);
    return geometry.localLower.xyz +
        normalized *
        (geometry.localUpper.xyz -
         geometry.localLower.xyz);
}

bool meshAabbOverlap(
    const float3 queryLower,
    const float3 queryUpper,
    const float3 childLower,
    const float3 childUpper
) {
    return
        all(queryLower <= childUpper) &&
        all(queryUpper >= childLower);
}

void insertMeshContact(
    thread ContactBatch& contacts,
    const thread MRRawContactGPU& candidate
) {
    if (contacts.count <
        MR_METAL_WORLD_RAW_CONTACTS_PER_PAIR) {
        contacts.contacts[contacts.count++] = candidate;
        return;
    }
    uint worst = 0u;
    for (uint index = 1u;
         index < contacts.count;
         ++index) {
        const MRRawContactGPU current =
            contacts.contacts[index];
        const MRRawContactGPU selected =
            contacts.contacts[worst];
        if (current.normalAndSeparation.w >
                selected.normalAndSeparation.w ||
            (current.normalAndSeparation.w ==
                 selected.normalAndSeparation.w &&
             current.featureAndFlags[1] >
                 selected.featureAndFlags[1])) {
            worst = index;
        }
    }
    if (candidate.normalAndSeparation.w <
            contacts.contacts[worst]
                .normalAndSeparation.w ||
        (candidate.normalAndSeparation.w ==
             contacts.contacts[worst]
                 .normalAndSeparation.w &&
         candidate.featureAndFlags[1] <
             contacts.contacts[worst]
                 .featureAndFlags[1])) {
        contacts.contacts[worst] = candidate;
    }
}

void convexTriangleContact(
    const bool meshIsA,
    const thread WorldShape& finiteShape,
    const thread WorldShape& mesh,
    device const MRShapeGPU& finiteSource,
    device const MRShapeGPU& meshSource,
    const MRMeshTriangleGPU triangle,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    thread ContactBatch& contacts
) {
    const float3 localA =
        geometryVertices[
            triangle.verticesAndFeature.x
        ].xyz * mesh.scale;
    const float3 localB =
        geometryVertices[
            triangle.verticesAndFeature.y
        ].xyz * mesh.scale;
    const float3 localC =
        geometryVertices[
            triangle.verticesAndFeature.z
        ].xyz * mesh.scale;
    const float3 a =
        mesh.center +
        quaternionRotate(mesh.rotation, localA);
    const float3 b =
        mesh.center +
        quaternionRotate(mesh.rotation, localB);
    const float3 c =
        mesh.center +
        quaternionRotate(mesh.rotation, localC);
    float3 normal = cross(b - a, c - a);
    const float normalSquared = dot(normal, normal);
    if (!(normalSquared > kTiny)) {
        return;
    }
    normal *= rsqrt(normalSquared);
    const float centerSide =
        dot(normal, finiteShape.center - a);
    const bool twoSided =
        (meshSource.flags &
         MR_SHAPE_FLAG_MESH_TWO_SIDED) != 0u;
    if (!twoSided && centerSide < 0.0f) {
        return;
    }
    if (twoSided && centerSide < 0.0f) {
        normal = -normal;
    }

    if (finiteShape.type == MR_SHAPE_SPHERE) {
        float3 sphereBarycentric;
        const float3 relativeClosest =
            closestTriangleWeights(
                a - finiteShape.center,
                b - finiteShape.center,
                c - finiteShape.center,
                sphereBarycentric
            );
        const float centerDistance = length(relativeClosest);
        const float3 finiteToMeshNormal =
            centerDistance > kTiny
            ? relativeClosest / centerDistance
            : -normal;
        const float separation =
            centerDistance - finiteShape.radius;
        const float acceptedDistance =
            finiteShape.contactOffset +
            mesh.contactOffset +
            pairQueryPadding(finiteShape, mesh);
        if (separation > acceptedDistance) {
            return;
        }
        uint edge = MR_INVALID_INDEX;
        if (sphereBarycentric.z <= 1.0e-5f) {
            edge = 0u;
        } else if (sphereBarycentric.x <= 1.0e-5f) {
            edge = 1u;
        } else if (sphereBarycentric.y <= 1.0e-5f) {
            edge = 2u;
        }
        if (edge != MR_INVALID_INDEX &&
            (triangle.adjacencyAndEdges.w &
             (1u << edge)) == 0u) {
            return;
        }
        MRRawContactGPU contact = makeContact(
            finiteToMeshNormal,
            separation,
            finiteShape.center +
                finiteToMeshNormal * finiteShape.radius,
            finiteShape.center + relativeClosest,
            featureKey(MR_SHAPE_SPHERE, 0u),
            featureKey(
                mesh.type,
                triangle.verticesAndFeature.w
            )
        );
        contact.featureAndFlags[3] =
            MR_RAW_CONTACT_MATERIAL_OVERRIDE |
            (
                triangle.materialAndFlags.x &
                MR_RAW_CONTACT_MATERIAL_INDEX_MASK
            );
        if (meshIsA) {
            contact = swappedContact(contact);
        }
        insertMeshContact(contacts, contact);
        return;
    }

    float3 planeFinitePoint;
    uint planeFiniteFeature = 0u;
    if (!supportWorldShape(
            finiteShape,
            finiteSource,
            geometryHeaders,
            geometryVertices,
            -normal,
            planeFinitePoint,
            planeFiniteFeature
        )) {
        return;
    }
    float3 planeBarycentric;
    const float3 planeRelativeClosest =
        closestTriangleWeights(
            a - planeFinitePoint,
            b - planeFinitePoint,
            c - planeFinitePoint,
            planeBarycentric
        );
    const float acceptedDistance =
        finiteShape.contactOffset +
        mesh.contactOffset +
        pairQueryPadding(finiteShape, mesh);

    float3 finitePoint = planeFinitePoint;
    float3 trianglePoint =
        planeFinitePoint + planeRelativeClosest;
    float3 finiteToMeshNormal = -normal;
    float separation = dot(
        normal,
        planeFinitePoint - a
    );
    uint finiteFeature = planeFiniteFeature;
    float3 witnessBarycentric = planeBarycentric;
    const bool insideFace =
        all(planeBarycentric > float3(1.0e-5f));
    if (!insideFace) {
        // Edge/vertex cases require the full finite-shape support function.
        // Query the zero-thickness triangle with cached-free certified GJK
        // instead of approximating the finite body by one plane support.
        WorldShape triangleShape = {};
        triangleShape.index =
            triangle.verticesAndFeature.w;
        triangleShape.type = kQueryTriangleShape;
        triangleShape.center = (a + b + c) / 3.0f;
        triangleShape.capsuleEndpoint0 = a;
        triangleShape.capsuleEndpoint1 = b;
        triangleShape.halfExtents = c;
        triangleShape.lower = min(a, min(b, c));
        triangleShape.upper = max(a, max(b, c));
        triangleShape.contactOffset = mesh.contactOffset;
        triangleShape.scale = float3(1.0f);

        ConvexSupportPoint simplex[4];
        float weights[4] = {
            0.0f,
            0.0f,
            0.0f,
            0.0f,
        };
        uint simplexCount = 0u;
        ConvexQueryResult query = {};
        query.status = MR_STEP_SUCCESS;
        const bool overlap = gjkDistance(
            finiteShape,
            triangleShape,
            finiteSource,
            meshSource,
            geometryHeaders,
            geometryVertices,
            simplex,
            simplexCount,
            weights,
            -normal,
            query
        );
        if (query.status == MR_STEP_SUCCESS && !overlap) {
            finitePoint = query.pointA;
            trianglePoint = query.pointB;
            finiteToMeshNormal = query.normal;
            separation = query.separation;
            finiteFeature = query.featureA;
            closestTriangleWeights(
                a - trianglePoint,
                b - trianglePoint,
                c - trianglePoint,
                witnessBarycentric
            );
        } else {
            const float distance =
                length(planeRelativeClosest);
            separation = distance;
            if (distance > kTiny) {
                finiteToMeshNormal =
                    planeRelativeClosest / distance;
            }
        }
    }
    if (separation > acceptedDistance) {
        return;
    }
    uint edge = MR_INVALID_INDEX;
    if (witnessBarycentric.z <= 1.0e-5f) {
        edge = 0u;
    } else if (witnessBarycentric.x <= 1.0e-5f) {
        edge = 1u;
    } else if (witnessBarycentric.y <= 1.0e-5f) {
        edge = 2u;
    }
    if (edge != MR_INVALID_INDEX &&
        (triangle.adjacencyAndEdges.w &
         (1u << edge)) == 0u) {
        return;
    }
    MRRawContactGPU contact = makeContact(
        finiteToMeshNormal,
        separation,
        finitePoint,
        trianglePoint,
        finiteFeature,
        featureKey(
            mesh.type,
            triangle.verticesAndFeature.w
        )
    );
    contact.featureAndFlags[3] =
        MR_RAW_CONTACT_MATERIAL_OVERRIDE |
        (
            triangle.materialAndFlags.x &
            MR_RAW_CONTACT_MATERIAL_INDEX_MASK
        );
    if (meshIsA) {
        contact = swappedContact(contact);
    }
    insertMeshContact(contacts, contact);
}

ContactBatch meshContacts(
    const thread WorldShape& shapeA,
    const thread WorldShape& shapeB,
    device const MRShapeGPU& sourceA,
    device const MRShapeGPU& sourceB,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    device const MRMeshBVHNodeGPU* meshNodes,
    device const MRMeshTriangleGPU* meshTriangles,
    thread uint& triangleCandidates
) {
    ContactBatch result = {};
    const bool meshIsA = isSurfaceShapeType(shapeA.type);
    const thread WorldShape& mesh =
        meshIsA ? shapeA : shapeB;
    const thread WorldShape& finiteShape =
        meshIsA ? shapeB : shapeA;
    device const MRShapeGPU& meshSource =
        meshIsA ? sourceA : sourceB;
    device const MRShapeGPU& finiteSource =
        meshIsA ? sourceB : sourceA;
    if (isSurfaceShapeType(finiteShape.type)) {
        return result;
    }
    const MRGeometryHeaderGPU geometry =
        geometryHeaders[meshSource.geometryOffset];
    const uint expectedKind =
        mesh.type == MR_SHAPE_TRIANGLE_MESH
        ? MR_GEOMETRY_TRIANGLE_MESH
        : MR_GEOMETRY_HEIGHTFIELD;
    if (geometry.kind != expectedKind) {
        return result;
    }
    float3 queryLower;
    float3 queryUpper;
    finiteBoundsInMeshLocal(
        finiteShape,
        mesh,
        queryLower,
        queryUpper
    );
    if (mesh.type == MR_SHAPE_HEIGHTFIELD) {
        if (geometry.vertexCount < 4u ||
            !(geometry.localLower.w > 0.0f) ||
            !(geometry.localUpper.w > 0.0f) ||
            queryUpper.x < geometry.localLower.x ||
            queryLower.x > geometry.localUpper.x ||
            queryUpper.y < geometry.localLower.y ||
            queryLower.y > geometry.localUpper.y) {
            return result;
        }
        const float inverseSpacing = geometry.localUpper.w;
        const uint cellCountX = uint(round(
            (geometry.localUpper.x - geometry.localLower.x) *
            inverseSpacing
        ));
        const uint cellCountY = uint(round(
            (geometry.localUpper.y - geometry.localLower.y) *
            inverseSpacing
        ));
        if (cellCountX == 0u || cellCountY == 0u ||
            (cellCountX + 1u) * (cellCountY + 1u) !=
                geometry.vertexCount) {
            return result;
        }
        const int firstX = clamp(
            int(floor(
                (queryLower.x - geometry.localLower.x) *
                inverseSpacing
            )) - 1,
            0,
            int(cellCountX - 1u)
        );
        const int firstY = clamp(
            int(floor(
                (queryLower.y - geometry.localLower.y) *
                inverseSpacing
            )) - 1,
            0,
            int(cellCountY - 1u)
        );
        const int lastX = clamp(
            int(floor(
                (queryUpper.x - geometry.localLower.x) *
                inverseSpacing
            )),
            0,
            int(cellCountX - 1u)
        );
        const int lastY = clamp(
            int(floor(
                (queryUpper.y - geometry.localLower.y) *
                inverseSpacing
            )),
            0,
            int(cellCountY - 1u)
        );
        const uint width = cellCountX + 1u;
        const float localContactDistance =
            (
                finiteShape.contactOffset +
                mesh.contactOffset +
                pairQueryPadding(finiteShape, mesh)
            ) /
            max(abs(mesh.scale.z), MR_MIN_COLLISION_EXTENT);
        for (int y = firstY; y <= lastY; ++y) {
            for (int x = firstX; x <= lastX; ++x) {
                const uint local00 =
                    uint(y) * width + uint(x);
                const uint vertex00 =
                    geometry.vertexOffset + local00;
                const uint vertex10 = vertex00 + 1u;
                const uint vertex01 = vertex00 + width;
                const uint vertex11 = vertex01 + 1u;
                const float height00 =
                    geometryVertices[vertex00].z;
                const float height10 =
                    geometryVertices[vertex10].z;
                const float height01 =
                    geometryVertices[vertex01].z;
                const float height11 =
                    geometryVertices[vertex11].z;
                const float cellMinimum = min(
                    min(height00, height10),
                    min(height01, height11)
                );
                const float cellMaximum = max(
                    max(height00, height10),
                    max(height01, height11)
                );
                if (queryLower.z >
                        cellMaximum + localContactDistance ||
                    queryUpper.z <
                        cellMinimum - localContactDistance) {
                    continue;
                }
                const uint cell =
                    uint(y) * cellCountX + uint(x);
                MRMeshTriangleGPU triangle0 = {};
                triangle0.verticesAndFeature = uint4(
                    vertex00,
                    vertex10,
                    vertex11,
                    2u * cell
                );
                triangle0.adjacencyAndEdges = uint4(
                    MR_INVALID_INDEX,
                    MR_INVALID_INDEX,
                    MR_INVALID_INDEX,
                    0x7u
                );
                triangle0.materialAndFlags.x =
                    meshSource.materialIndex;
                if (triangleCandidates != 0xffffffffu) {
                    ++triangleCandidates;
                }
                convexTriangleContact(
                    meshIsA,
                    finiteShape,
                    mesh,
                    finiteSource,
                    meshSource,
                    triangle0,
                    geometryHeaders,
                    geometryVertices,
                    result
                );

                MRMeshTriangleGPU triangle1 = {};
                triangle1.verticesAndFeature = uint4(
                    vertex00,
                    vertex11,
                    vertex01,
                    2u * cell + 1u
                );
                triangle1.adjacencyAndEdges = uint4(
                    MR_INVALID_INDEX,
                    MR_INVALID_INDEX,
                    MR_INVALID_INDEX,
                    0x7u
                );
                triangle1.materialAndFlags.x =
                    meshSource.materialIndex;
                if (triangleCandidates != 0xffffffffu) {
                    ++triangleCandidates;
                }
                convexTriangleContact(
                    meshIsA,
                    finiteShape,
                    mesh,
                    finiteSource,
                    meshSource,
                    triangle1,
                    geometryHeaders,
                    geometryVertices,
                    result
                );
            }
        }
        return result;
    }
    if (geometry.bvhCount == 0u) {
        return result;
    }
    uint cursor = 0u;
    uint visits = 0u;
    const uint maximumVisits =
        geometry.bvhCount * MR_MESH_BVH_BRANCHING;
    while (cursor != MR_MESH_BVH_INVALID_ESCAPE &&
           visits++ < maximumVisits) {
        const uint nodeIndex =
            cursor / MR_MESH_BVH_BRANCHING;
        const uint slot =
            cursor -
            nodeIndex * MR_MESH_BVH_BRANCHING;
        if (nodeIndex >= geometry.bvhCount) {
            break;
        }
        const MRMeshBVHNodeGPU node =
            meshNodes[geometry.bvhOffset + nodeIndex];
        const uint metadata =
            uint4Component(node.childMeta, slot);
        const uint escape =
            (metadata & MR_MESH_BVH_ESCAPE_MASK) >>
            MR_MESH_BVH_ESCAPE_SHIFT;
        const uint child =
            uint4Component(node.childIndices, slot);
        if (child == MR_INVALID_INDEX) {
            cursor = escape;
            continue;
        }
        const float3 childLower =
            dequantizedMeshBound(
                node.quantizedLower[slot],
                geometry
            );
        const float3 childUpper =
            dequantizedMeshBound(
                node.quantizedUpper[slot],
                geometry
            );
        if (!meshAabbOverlap(
                queryLower,
                queryUpper,
                childLower,
                childUpper
            )) {
            cursor = escape;
            continue;
        }
        if ((metadata & MR_MESH_BVH_LEAF_BIT) == 0u) {
            cursor = child * MR_MESH_BVH_BRANCHING;
            continue;
        }
        const uint triangleCount =
            metadata & MR_MESH_BVH_LEAF_COUNT_MASK;
        for (uint triangle = 0u;
             triangle < triangleCount;
             ++triangle) {
            if (triangleCandidates != 0xffffffffu) {
                ++triangleCandidates;
            }
            convexTriangleContact(
                meshIsA,
                finiteShape,
                mesh,
                finiteSource,
                meshSource,
                meshTriangles[
                    geometry.triangleOffset +
                    child + triangle
                ],
                geometryHeaders,
                geometryVertices,
                result
            );
        }
        cursor = escape;
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
        MR_METAL_WORLD_CONTACT_HAS_KINEMATIC_TARGETS |
        MR_METAL_WORLD_CONTACT_WAVE32 |
        MR_METAL_WORLD_CONTACT_CCD |
        MR_METAL_WORLD_CONTACT_HAS_FUTURE_KINEMATICS |
        MR_METAL_WORLD_CONTACT_QUALITY |
        MR_METAL_WORLD_CONTACT_BODY_PARAMETERS |
        MR_METAL_WORLD_CONTACT_STREAMED_RESPONSES |
        MR_METAL_WORLD_CONTACT_BODY_WRENCHES;
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
        dispatch.workQueueClassCount ==
            MR_WORLD_WORK_CLASS_COUNT &&
        dispatch.queueStride >= dispatch.islandCapacity &&
        dispatch.solverTileCapacity > 0u &&
        dispatch.ccdMode <= MR_WORLD_CCD_HYBRID &&
        dispatch.maxCCDEvents > 0u &&
        dispatch.maxCCDEvents <= MR_CCD_MAX_EVENTS &&
        dispatch.maxCCDAdvanceSolvePasses > 0u &&
        dispatch.maxCCDAdvanceSolvePasses <=
            MR_CCD_MAX_ADVANCE_SOLVE_PASSES &&
        dispatch.maxCCDZeroTimeReplays <=
            MR_CCD_MAX_ZERO_TIME_REPLAYS &&
        dispatch.waveWorkerGroupCount > 0u &&
        dispatch.articulationCount > 0u &&
        dispatch.dynamicNodeCount > 0u &&
        dispatch.dynamicNodeCount <=
            MR_WORLD_MAX_DYNAMIC_NODES &&
        dispatch.islandNodeReferenceCapacity >=
            dispatch.dynamicNodeCount &&
        dispatch.islandConstraintReferenceCapacity >=
            dispatch.constraintCapacity &&
        dispatch.operatorVelocityCapacity >= dispatch.nv &&
        dispatch.maxConservativeAdvancementIterations > 0u &&
        dispatch.rowCapacity > 0u &&
        dispatch.nv > 0u &&
        dispatch.velocityIterations > 0u &&
        (
            dispatch.solverType ==
                MR_SOLVER_QUALITY_NEWTON ||
            (
                dispatch.solverType >=
                    MR_SOLVER_TEMPORAL_CONE &&
                dispatch.solverType <=
                    MR_SOLVER_THROUGHPUT_PGS
            )
        ) &&
        (dispatch.flags & ~knownFlags) == 0u &&
        finiteFloat4(dispatch.timestepAndBias) &&
        dispatch.timestepAndBias.x > 0.0f &&
        all(dispatch.timestepAndBias.yzw >= 0.0f) &&
        finiteFloat4(dispatch.manifoldThresholds) &&
        all(dispatch.manifoldThresholds.xyz >= 0.0f) &&
        dispatch.manifoldThresholds.w >= -1.0f &&
        dispatch.manifoldThresholds.w <= 1.0f &&
        finiteFloat4(dispatch.ccdParameters) &&
        all(dispatch.ccdParameters.xyz >= 0.0f) &&
        dispatch.ccdParameters.x > 0.0f &&
        dispatch.ccdParameters.y > 0.0f &&
        dispatch.ccdParameters.w > 0.0f &&
        finiteFloat4(dispatch.ccdEventParameters) &&
        dispatch.ccdEventParameters.x > 0.0f &&
        dispatch.ccdEventParameters.y > 0.0f;
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
    device const MRGeometryHeaderGPU* geometryHeaders [[buffer(4)]],
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
        geometryHeaders,
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

float shapeSweepRadius(
    const MRShapeGPU source,
    const MRProjectedColliderGPU projected
) {
    float radius =
        source.contactRestAndBoundingRadius.z;
    if (radius > 0.0f && isfinite(radius)) {
        return radius;
    }
    if (source.shapeType == MR_SHAPE_SPHERE) {
        return source.dimensions.x;
    }
    if (source.shapeType == MR_SHAPE_CAPSULE) {
        return source.dimensions.x + source.dimensions.y;
    }
    if (source.shapeType == MR_SHAPE_BOX) {
        return length(source.dimensions.xyz);
    }
    if (source.shapeType == MR_SHAPE_CYLINDER) {
        return length(
            float2(source.dimensions.x, source.dimensions.y)
        );
    }
    return length(
        0.5f *
        (
            projected.upperAndContactOffset.xyz -
            projected.lowerAndHalfLength.xyz
        )
    );
}

float4 integrateWorldQuaternion(
    const float4 orientation,
    const float3 angularVelocity,
    const float timestep
) {
    const float angularSpeed = length(angularVelocity);
    if (!(angularSpeed > 1.0e-8f) || !(timestep > 0.0f)) {
        return orientation;
    }
    const float halfAngle = 0.5f * angularSpeed * timestep;
    const float4 delta = float4(
        angularVelocity / angularSpeed * sin(halfAngle),
        cos(halfAngle)
    );
    const float4 candidate =
        quaternionMultiply(delta, orientation);
    return candidate * rsqrt(dot(candidate, candidate));
}

MRProjectedColliderGPU looseProjectedCollider(
    const MRShapeGPU source,
    const MRProjectedColliderGPU current,
    const float3 center,
    const float4 rotation,
    const float sweepRadius
) {
    MRProjectedColliderGPU result = current;
    result.statusAndFlags.z = 0u;
    result.statusAndFlags.w = source.flags;
    result.centerAndRadius.xyz = center;
    result.rotation = rotation;
    result.upperAndContactOffset.w =
        source.contactRestAndBoundingRadius.x;
    if (source.shapeType != MR_SHAPE_PLANE) {
        const float extent =
            max(
                sweepRadius,
                source.contactRestAndBoundingRadius.x
            ) +
            source.contactRestAndBoundingRadius.x;
        result.lowerAndHalfLength.xyz = center - extent;
        result.upperAndContactOffset.xyz = center + extent;
    }
    return result;
}

// Produces an unconstrained end transform and turns the current collider
// record into a conservative swept/speculative record. Articulation motion is
// taken from ABA's candidate q through a second kinematics projection rather
// than guessed from a dense or host-visible velocity representation.
kernel void mr_world_project_swept_colliders(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRShapeGPU* shapes [[buffer(1)]],
    device const MRArticulationGPU* articulations [[buffer(2)]],
    device const MRBodyStateGPU* currentBodies [[buffer(3)]],
    device const MRBodyStateGPU* candidateBodies [[buffer(4)]],
    device const MRArticulatedBodyPoseGPU* futureBodyPoses [[buffer(5)]],
    device const float* candidateV [[buffer(6)]],
    device const MRGeometryHeaderGPU* geometryHeaders [[buffer(7)]],
    device MRProjectedColliderGPU* projectedColliders [[buffer(8)]],
    device MRProjectedColliderGPU* futureProjectedColliders [[buffer(9)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(10)]],
    device const MRCCDEventStateGPU* eventStates [[buffer(11)]],
    const uint threadIndex [[thread_position_in_grid]]
) {
    if (dispatch.shapeCount == 0u) {
        return;
    }
    const uint environment =
        threadIndex / dispatch.shapeCount;
    if (environment >= dispatch.environmentCount) {
        return;
    }
    const uint collider =
        threadIndex - environment * dispatch.shapeCount;
    const uint projectedIndex =
        environment * dispatch.shapeCount + collider;
    const MRShapeGPU source = shapes[collider];
    WorldShape currentShape = {};
    uint projectionFailure = MR_STEP_SUCCESS;
    const bool projectedSuccessfully = makeWorldShape(
        collider,
        shapes,
        geometryHeaders,
        currentBodies +
            environment * dispatch.bodyStateStride,
        dispatch.bodyCount,
        currentShape,
        projectionFailure
    );
    MRProjectedColliderGPU initialProjection =
        projectedCollider(currentShape, projectionFailure);
    projectedColliders[projectedIndex] = initialProjection;
    device MRProjectedColliderGPU& projected =
        projectedColliders[projectedIndex];
    if (statuses[environment].code != MR_STEP_SUCCESS ||
        !projectedSuccessfully) {
        futureProjectedColliders[projectedIndex] = projected;
        return;
    }
    projected.statusAndFlags.w = source.flags;
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    const MRBodyStateGPU currentBody =
        currentBodies[bodyBase + source.bodyIndex];
    const uint articulationOwner =
        currentBody.flagsAndIndices[1];
    const bool articulated =
        articulationOwner != MR_INVALID_INDEX &&
        articulationOwner < dispatch.articulationCount;
    const MRArticulationGPU articulation =
        articulations[
            articulated
            ? articulationOwner
            : dispatch.articulationIndex
        ];
    const bool hasFutureKinematics =
        (dispatch.flags &
         MR_METAL_WORLD_CONTACT_HAS_FUTURE_KINEMATICS) != 0u;
    const float segmentDuration =
        dispatch.ccdMode == MR_WORLD_CCD_HYBRID
        ? max(eventStates[environment].time.y, 0.0f)
        : dispatch.timestepAndBias.x;
    if (dispatch.ccdMode == MR_WORLD_CCD_DISABLED ||
        source.shapeType == MR_SHAPE_PLANE ||
        currentBody.flagsAndIndices[0] == MR_MOTION_STATIC) {
        projected.statusAndFlags.z = 0u;
        futureProjectedColliders[projectedIndex] = projected;
        return;
    }

    // Most RL contacts use speculative CCD without opting into exact event
    // casts. For articulated links, rebuilding an approximate future pose
    // from a zero link velocity adds work but no information. Inflate the
    // current bound directly from the generalized-speed envelope instead.
    // Exact-CCD colliders continue through the future-FK path below.
    const bool exactArticulatedCCD =
        articulated &&
        hasFutureKinematics &&
        (source.flags & MR_SHAPE_FLAG_ENABLE_CCD) != 0u;
    if (articulated && !exactArticulatedCCD) {
        const float sweepRadius =
            shapeSweepRadius(source, projected);
        const uint velocityBase =
            environment * dispatch.vStride;
        float generalizedSpeed = 0.0f;
        for (uint localDof = 0u;
             localDof < articulation.nv;
             ++localDof) {
            generalizedSpeed +=
                abs(
                    candidateV[
                        velocityBase +
                        articulation.vOffset +
                        localDof
                    ]
                );
        }
        const float envelope = min(
            segmentDuration *
                generalizedSpeed *
                max(sweepRadius, 0.25f),
            segmentDuration *
                dispatch.ccdParameters.w
        );
        const float speculativeMargin =
            dispatch.ccdParameters.z * envelope;
        projected.lowerAndHalfLength.xyz -=
            speculativeMargin;
        projected.upperAndContactOffset.xyz +=
            speculativeMargin;
        projected.upperAndContactOffset.w =
            source.contactRestAndBoundingRadius.x +
            speculativeMargin;
        projected.statusAndFlags.z =
            speculativeMargin > 0.0f ? 1u : 0u;
        futureProjectedColliders[projectedIndex] = projected;
        return;
    }

    float3 futureBodyPosition = currentBody.position.xyz;
    float4 futureBodyRotation = currentBody.orientation;
    if (articulated && hasFutureKinematics) {
        if (source.bodyIndex < articulation.firstBody ||
            source.bodyIndex >=
                articulation.firstBody + articulation.bodyCount) {
            MRMetalWorldContactStatusGPU status =
                statuses[environment];
            status.code = MR_STEP_UNSUPPORTED;
            status.firstFailingPair = collider;
            statuses[environment] = status;
            futureProjectedColliders[projectedIndex] = projected;
            return;
        }
        const MRArticulatedBodyPoseGPU futurePose =
            futureBodyPoses[
                environment * dispatch.bodyCount +
                source.bodyIndex
            ];
        futureBodyPosition = futurePose.position.xyz;
        futureBodyRotation = futurePose.orientation;
    } else {
        const MRBodyStateGPU candidate =
            candidateBodies[bodyBase + source.bodyIndex];
        if (candidate.flagsAndIndices[0] != MR_MOTION_STATIC) {
            futureBodyPosition +=
                segmentDuration *
                candidate.linearVelocityAndInverseMass.xyz;
            futureBodyRotation = integrateWorldQuaternion(
                currentBody.orientation,
                candidate.angularVelocity.xyz,
                segmentDuration
            );
        }
    }
    float4 localRotation;
    float4 normalizedFutureBodyRotation;
    if (!checkedQuaternion(
            source.localRotation,
            localRotation
        ) ||
        !checkedQuaternion(
            futureBodyRotation,
            normalizedFutureBodyRotation
        )) {
        MRMetalWorldContactStatusGPU status =
            statuses[environment];
        status.code = MR_STEP_NONFINITE_RESULT;
        status.firstFailingPair = collider;
        statuses[environment] = status;
        futureProjectedColliders[projectedIndex] = projected;
        return;
    }
    const float3 futureCenter =
        futureBodyPosition +
        quaternionRotate(
            normalizedFutureBodyRotation,
            source.localPosition.xyz
        );
    float4 futureRotation = quaternionMultiply(
        normalizedFutureBodyRotation,
        localRotation
    );
    futureRotation *= rsqrt(dot(futureRotation, futureRotation));
    const float sweepRadius =
        shapeSweepRadius(source, projected);
    MRProjectedColliderGPU future = looseProjectedCollider(
        source,
        projected,
        futureCenter,
        futureRotation,
        sweepRadius
    );
    futureProjectedColliders[projectedIndex] = future;

    const float translation =
        length(futureCenter - projected.centerAndRadius.xyz);
    const float orientationCosine = clamp(
        abs(dot(projected.rotation, futureRotation)),
        0.0f,
        1.0f
    );
    const float rotationalChord =
        2.0f * sweepRadius *
        sqrt(max(0.0f, 1.0f - orientationCosine *
                                  orientationCosine));
    const float envelope = min(
        translation + rotationalChord,
        segmentDuration *
            dispatch.ccdParameters.w
    );
    const float speculativeMargin =
        dispatch.ccdParameters.z * envelope;
    projected.lowerAndHalfLength.xyz = min(
        projected.lowerAndHalfLength.xyz,
        future.lowerAndHalfLength.xyz
    ) - speculativeMargin;
    projected.upperAndContactOffset.xyz = max(
        projected.upperAndContactOffset.xyz,
        future.upperAndContactOffset.xyz
    ) + speculativeMargin;
    projected.upperAndContactOffset.w =
        source.contactRestAndBoundingRadius.x +
        speculativeMargin;
    projected.statusAndFlags.z =
        as_type<uint>(speculativeMargin);
}

float4 rodCapsuleOrientation(const float3 axis) {
    const float3 reference = float3(0.0f, 1.0f, 0.0f);
    const float cosine = clamp(dot(reference, axis), -1.0f, 1.0f);
    if (cosine < -1.0f + 1.0e-6f) {
        return float4(1.0f, 0.0f, 0.0f, 0.0f);
    }
    float4 rotation = float4(
        cross(reference, axis),
        1.0f + cosine
    );
    const float normSquared = dot(rotation, rotation);
    return normSquared > kTiny
        ? rotation * rsqrt(normSquared)
        : float4(0.0f, 0.0f, 0.0f, 1.0f);
}

MRProjectedColliderGPU projectedRodCapsule(
    const MRRodColliderGPU collider,
    const float3 endpointA,
    const float3 endpointB
) {
    MRProjectedColliderGPU projected = {};
    const float3 edge = endpointB - endpointA;
    const float lengthSquared = dot(edge, edge);
    if (!(lengthSquared > kTiny) ||
        !finiteFloat3(endpointA) ||
        !finiteFloat3(endpointB)) {
        projected.statusAndFlags.x =
            MR_STEP_NONFINITE_INPUT;
        return projected;
    }
    const float edgeLength = sqrt(lengthSquared);
    const float radius = collider.radiusAndOffsets.x;
    const float contactOffset = collider.radiusAndOffsets.y;
    const float extent = radius + contactOffset;
    projected.statusAndFlags = uint4(
        MR_STEP_SUCCESS,
        0u,
        0u,
        MR_SHAPE_FLAG_ENABLE_CCD
    );
    projected.centerAndRadius = float4(
        0.5f * (endpointA + endpointB),
        radius
    );
    projected.rotation =
        rodCapsuleOrientation(edge / edgeLength);
    projected.lowerAndHalfLength = float4(
        min(endpointA, endpointB) - extent,
        0.5f * edgeLength
    );
    projected.upperAndContactOffset = float4(
        max(endpointA, endpointB) + extent,
        contactOffset
    );
    return projected;
}

// Projects the accepted and remaining-time DER states into deforming capsule
// records. The accepted record carries the swept union for broadphase/mesh
// traversal while preserving its exact center, rotation, and half length for
// conservative advancement.
kernel void mr_world_project_swept_rod_colliders(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRRodColliderGPU* rodColliders [[buffer(1)]],
    device const MRRodNodeStateGPU* currentNodes [[buffer(2)]],
    device const MRRodNodeStateGPU* futureNodes [[buffer(3)]],
    device MRProjectedColliderGPU* projectedRodColliders [[buffer(4)]],
    device MRProjectedColliderGPU* futureProjectedRodColliders [[buffer(5)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(6)]],
    const uint threadIndex [[thread_position_in_grid]]
) {
    if (dispatch.rodEdgeCount == 0u) {
        return;
    }
    const uint environment = threadIndex / dispatch.rodEdgeCount;
    if (environment >= dispatch.environmentCount) {
        return;
    }
    const uint rodCollider =
        threadIndex - environment * dispatch.rodEdgeCount;
    const MRRodColliderGPU collider =
        rodColliders[rodCollider];
    const uint nodeBase = environment * dispatch.rodNodeCount;
    const uint output =
        environment * dispatch.rodEdgeCount + rodCollider;
    if (collider.nodeA >= dispatch.rodNodeCount ||
        collider.nodeB >= dispatch.rodNodeCount ||
        collider.nodeA == collider.nodeB) {
        MRProjectedColliderGPU failed = {};
        failed.statusAndFlags.x = MR_STEP_UNSUPPORTED;
        projectedRodColliders[output] = failed;
        futureProjectedRodColliders[output] = failed;
        if (statuses[environment].code == MR_STEP_SUCCESS) {
            MRMetalWorldContactStatusGPU status =
                statuses[environment];
            status.code = MR_STEP_UNSUPPORTED;
            status.firstFailingPair = rodCollider;
            statuses[environment] = status;
        }
        return;
    }
    const float3 currentA =
        currentNodes[nodeBase + collider.nodeA].position.xyz;
    const float3 currentB =
        currentNodes[nodeBase + collider.nodeB].position.xyz;
    const float3 futureA =
        futureNodes[nodeBase + collider.nodeA].position.xyz;
    const float3 futureB =
        futureNodes[nodeBase + collider.nodeB].position.xyz;
    MRProjectedColliderGPU current = projectedRodCapsule(
        collider,
        currentA,
        currentB
    );
    MRProjectedColliderGPU future = projectedRodCapsule(
        collider,
        futureA,
        futureB
    );
    if (current.statusAndFlags.x != MR_STEP_SUCCESS ||
        future.statusAndFlags.x != MR_STEP_SUCCESS) {
        if (statuses[environment].code == MR_STEP_SUCCESS) {
            MRMetalWorldContactStatusGPU status =
                statuses[environment];
            status.code = MR_STEP_NONFINITE_INPUT;
            status.firstFailingPair = rodCollider;
            statuses[environment] = status;
        }
        projectedRodColliders[output] = current;
        futureProjectedRodColliders[output] = future;
        return;
    }
    const float endpointMotion = max(
        length(futureA - currentA),
        length(futureB - currentB)
    );
    const float speculativeMargin =
        dispatch.ccdParameters.z * endpointMotion;
    current.lowerAndHalfLength.xyz =
        min(
            current.lowerAndHalfLength.xyz,
            future.lowerAndHalfLength.xyz
        ) - speculativeMargin;
    current.upperAndContactOffset.xyz =
        max(
            current.upperAndContactOffset.xyz,
            future.upperAndContactOffset.xyz
        ) + speculativeMargin;
    current.upperAndContactOffset.w += speculativeMargin;
    current.statusAndFlags.z =
        as_type<uint>(speculativeMargin);
    projectedRodColliders[output] = current;
    futureProjectedRodColliders[output] = future;
}

// Projects the already-advanced event configuration without adding a second
// speculative sweep. Literal CCD uses this after advancing to the selected
// TOI so narrowphase witnesses and manifold anchors belong to the impact
// configuration rather than the beginning of the remaining interval.
kernel void mr_world_project_event_colliders(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRShapeGPU* shapes [[buffer(1)]],
    device const MRGeometryHeaderGPU* geometryHeaders [[buffer(2)]],
    device const MRBodyStateGPU* eventBodies [[buffer(3)]],
    device MRProjectedColliderGPU* projectedColliders [[buffer(4)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(5)]],
    const uint threadIndex [[thread_position_in_grid]]
) {
    if (dispatch.shapeCount == 0u) {
        return;
    }
    const uint environment =
        threadIndex / dispatch.shapeCount;
    if (environment >= dispatch.environmentCount) {
        return;
    }
    const uint collider =
        threadIndex - environment * dispatch.shapeCount;
    const uint projectedIndex =
        environment * dispatch.shapeCount + collider;
    if (statuses[environment].code != MR_STEP_SUCCESS) {
        return;
    }
    WorldShape shape = {};
    uint projectionFailure = MR_STEP_SUCCESS;
    const bool projectedSuccessfully = makeWorldShape(
        collider,
        shapes,
        geometryHeaders,
        eventBodies +
            environment * dispatch.bodyStateStride,
        dispatch.bodyCount,
        shape,
        projectionFailure
    );
    MRProjectedColliderGPU projected =
        projectedCollider(shape, projectionFailure);
    projected.statusAndFlags.w = shapes[collider].flags;
    projected.statusAndFlags.z = 0u;
    projectedColliders[projectedIndex] = projected;
    if (!projectedSuccessfully) {
        MRMetalWorldContactStatusGPU status =
            statuses[environment];
        status.code = projectionFailure;
        status.firstFailingPair = collider;
        status.firstFailingStableKeyLow = collider;
        status.firstFailingStableKeyHigh = environment;
        statuses[environment] = status;
    }
}

float4 shortestNlerp(
    const float4 start,
    float4 end,
    const float alpha
) {
    if (dot(start, end) < 0.0f) {
        end = -end;
    }
    const float4 value = mix(start, end, alpha);
    return value * rsqrt(dot(value, value));
}

WorldShape interpolatedCCDShape(
    const uint collider,
    device const MRShapeGPU& source,
    device const MRProjectedColliderGPU& startProjected,
    device const MRProjectedColliderGPU& endProjected,
    const float alpha
) {
    WorldShape shape = {};
    uint failure = MR_STEP_SUCCESS;
    loadProjectedCollider(
        collider,
        source,
        startProjected,
        shape,
        failure
    );
    shape.center = mix(
        startProjected.centerAndRadius.xyz,
        endProjected.centerAndRadius.xyz,
        alpha
    );
    shape.rotation = shortestNlerp(
        startProjected.rotation,
        endProjected.rotation,
        alpha
    );
    shape.contactOffset =
        source.contactRestAndBoundingRadius.x;
    const float bound = shapeSweepRadius(source, startProjected) +
        shape.contactOffset;
    shape.lower = shape.center - bound;
    shape.upper = shape.center + bound;
    if (shape.type == MR_SHAPE_CAPSULE) {
        // Procedural rod capsules deform, so their half length lives in the
        // projected records. Rigid capsules store the same value there.
        shape.halfLength = mix(
            startProjected.lowerAndHalfLength.w,
            endProjected.lowerAndHalfLength.w,
            alpha
        );
        const float3 axis = quaternionRotate(
            shape.rotation,
            float3(0.0f, shape.halfLength, 0.0f)
        );
        shape.capsuleEndpoint0 = shape.center - axis;
        shape.capsuleEndpoint1 = shape.center + axis;
    } else if (shape.type == MR_SHAPE_BOX) {
        shape.halfExtents = source.dimensions.xyz;
    } else if (shape.type == MR_SHAPE_CYLINDER) {
        shape.halfLength = source.dimensions.y;
        shape.cylinderAxis = normalize(quaternionRotate(
            shape.rotation,
            float3(0.0f, 1.0f, 0.0f)
        ));
        shape.cylinderBasisX = normalize(quaternionRotate(
            shape.rotation,
            float3(1.0f, 0.0f, 0.0f)
        ));
        shape.cylinderBasisZ = normalize(cross(
            shape.cylinderBasisX,
            shape.cylinderAxis
        ));
    } else if (shape.type == MR_SHAPE_PLANE) {
        shape.planeNormal = normalize(quaternionRotate(
            shape.rotation,
            float3(0.0f, 1.0f, 0.0f)
        ));
    }
    return shape;
}

__attribute__((noinline))
void updateMeshConvexDistance(
    const bool meshIsA,
    const thread WorldShape& finiteShape,
    const thread WorldShape& mesh,
    device const MRShapeGPU& finiteSource,
    device const MRShapeGPU& meshSource,
    const bool twoSided,
    const MRMeshTriangleGPU triangle,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    thread bool& found,
    thread float& bestSeparation,
    thread float3& bestNormal,
    thread float3& bestFinitePoint,
    thread float3& bestMeshPoint,
    thread uint& bestFeature
) {
    const float3 a = mesh.center + quaternionRotate(
        mesh.rotation,
        geometryVertices[
            triangle.verticesAndFeature.x
        ].xyz * mesh.scale
    );
    const float3 b = mesh.center + quaternionRotate(
        mesh.rotation,
        geometryVertices[
            triangle.verticesAndFeature.y
        ].xyz * mesh.scale
    );
    const float3 c = mesh.center + quaternionRotate(
        mesh.rotation,
        geometryVertices[
            triangle.verticesAndFeature.z
        ].xyz * mesh.scale
    );
    float3 faceNormal = cross(b - a, c - a);
    const float faceLengthSquared =
        dot(faceNormal, faceNormal);
    if (!(faceLengthSquared > kTiny)) {
        return;
    }
    faceNormal *= rsqrt(faceLengthSquared);
    const float side =
        dot(faceNormal, finiteShape.center - a);
    if (!twoSided && side < 0.0f) {
        return;
    }

    WorldShape triangleShape = {};
    triangleShape.index = triangle.verticesAndFeature.w;
    triangleShape.type = kQueryTriangleShape;
    triangleShape.center = (a + b + c) / 3.0f;
    triangleShape.capsuleEndpoint0 = a;
    triangleShape.capsuleEndpoint1 = b;
    triangleShape.halfExtents = c;
    triangleShape.lower = min(a, min(b, c));
    triangleShape.upper = max(a, max(b, c));
    triangleShape.contactOffset = 0.0f;
    triangleShape.scale = float3(1.0f);
    ConvexQueryResult exact = {};
    supportMappedContacts(
        finiteShape.index,
        triangleShape.index,
        finiteShape,
        triangleShape,
        finiteSource,
        meshSource,
        geometryHeaders,
        geometryVertices,
        -faceNormal,
        exact
    );
    if (exact.status != MR_STEP_SUCCESS ||
        !isfinite(exact.separation) ||
        !finiteFloat3(exact.normal) ||
        !finiteFloat3(exact.pointA) ||
        !finiteFloat3(exact.pointB)) {
        return;
    }
    const float separation = exact.separation;
    if (!found ||
        separation < bestSeparation ||
        (separation == bestSeparation &&
         triangle.verticesAndFeature.w < bestFeature)) {
        found = true;
        bestSeparation = separation;
        bestNormal = exact.normal;
        bestFinitePoint = exact.pointA;
        bestMeshPoint = exact.pointB;
        bestFeature = triangle.verticesAndFeature.w;
    }
    static_cast<void>(meshIsA);
}

__attribute__((noinline))
bool meshConvexDistance(
    const bool meshIsA,
    const thread WorldShape& finiteShape,
    const thread WorldShape& mesh,
    device const MRProjectedColliderGPU& finiteSweptProjected,
    device const MRShapeGPU& finiteSource,
    device const MRShapeGPU& meshSource,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    device const MRMeshBVHNodeGPU* meshNodes,
    device const MRMeshTriangleGPU* meshTriangles,
    thread ConvexQueryResult& query
) {
    const MRGeometryHeaderGPU geometry =
        geometryHeaders[mesh.geometryIndex];
    bool found = false;
    float bestSeparation = INFINITY;
    float3 bestNormal = float3(0.0f, 1.0f, 0.0f);
    float3 bestFinitePoint = finiteShape.center;
    float3 bestMeshPoint = finiteShape.center;
    uint bestFeature = 0u;
    const bool twoSided =
        (meshSource.flags &
         MR_SHAPE_FLAG_MESH_TWO_SIDED) != 0u;

    WorldShape sweptFinite = finiteShape;
    sweptFinite.lower =
        finiteSweptProjected.lowerAndHalfLength.xyz;
    sweptFinite.upper =
        finiteSweptProjected.upperAndContactOffset.xyz;
    float3 queryLower;
    float3 queryUpper;
    finiteBoundsInMeshLocal(
        sweptFinite,
        mesh,
        queryLower,
        queryUpper
    );
    uint cursor = 0u;
    uint visits = 0u;
    const uint maximumVisits =
        geometry.bvhCount * MR_MESH_BVH_BRANCHING;
    while (cursor != MR_MESH_BVH_INVALID_ESCAPE &&
           visits++ < maximumVisits) {
        const uint nodeIndex =
            cursor / MR_MESH_BVH_BRANCHING;
        const uint slot =
            cursor - nodeIndex * MR_MESH_BVH_BRANCHING;
        if (nodeIndex >= geometry.bvhCount) {
            break;
        }
        const MRMeshBVHNodeGPU node =
            meshNodes[geometry.bvhOffset + nodeIndex];
        const uint metadata =
            uint4Component(node.childMeta, slot);
        const uint escape =
            (metadata & MR_MESH_BVH_ESCAPE_MASK) >>
            MR_MESH_BVH_ESCAPE_SHIFT;
        const uint child =
            uint4Component(node.childIndices, slot);
        if (child == MR_INVALID_INDEX ||
            !meshAabbOverlap(
                queryLower,
                queryUpper,
                dequantizedMeshBound(
                    node.quantizedLower[slot],
                    geometry
                ),
                dequantizedMeshBound(
                    node.quantizedUpper[slot],
                    geometry
                )
            )) {
            cursor = escape;
            continue;
        }
        if ((metadata & MR_MESH_BVH_LEAF_BIT) == 0u) {
            cursor = child * MR_MESH_BVH_BRANCHING;
            continue;
        }
        const uint triangleCount =
            metadata & MR_MESH_BVH_LEAF_COUNT_MASK;
        for (uint triangle = 0u;
             triangle < triangleCount;
             ++triangle) {
            updateMeshConvexDistance(
                meshIsA,
                finiteShape,
                mesh,
                finiteSource,
                meshSource,
                twoSided,
                meshTriangles[
                    geometry.triangleOffset +
                    child + triangle
                ],
                geometryHeaders,
                geometryVertices,
                found,
                bestSeparation,
                bestNormal,
                bestFinitePoint,
                bestMeshPoint,
                bestFeature
            );
        }
        cursor = escape;
    }
    // Quantization is outward conservative, but a malformed/zero-extent
    // authored axis must still degrade to deterministic GPU traversal rather
    // than an unresolved host fallback.
    if (!found) {
        for (uint localTriangle = 0u;
             localTriangle < geometry.triangleCount;
             ++localTriangle) {
            updateMeshConvexDistance(
                meshIsA,
                finiteShape,
                mesh,
                finiteSource,
                meshSource,
                twoSided,
                meshTriangles[
                    geometry.triangleOffset + localTriangle
                ],
                geometryHeaders,
                geometryVertices,
                found,
                bestSeparation,
                bestNormal,
                bestFinitePoint,
                bestMeshPoint,
                bestFeature
            );
        }
    }
    if (!found) {
        return false;
    }
    query = {};
    query.status = MR_STEP_SUCCESS;
    query.fallback = 2u;
    query.separation = bestSeparation;
    query.normal = meshIsA ? -bestNormal : bestNormal;
    query.pointA =
        meshIsA
        ? bestMeshPoint
        : bestFinitePoint;
    query.pointB =
        meshIsA
        ? bestFinitePoint
        : bestMeshPoint;
    query.featureA =
        meshIsA
        ? featureKey(MR_SHAPE_TRIANGLE_MESH, bestFeature)
        : featureKey(finiteShape.type, 0u);
    query.featureB =
        meshIsA
        ? featureKey(finiteShape.type, 0u)
        : featureKey(MR_SHAPE_TRIANGLE_MESH, bestFeature);
    return true;
}

bool ccdDistanceAtAlpha(
    const uint colliderA,
    const uint colliderB,
    device const MRShapeGPU& sourceA,
    device const MRShapeGPU& sourceB,
    device const MRProjectedColliderGPU& startA,
    device const MRProjectedColliderGPU& startB,
    device const MRProjectedColliderGPU& endA,
    device const MRProjectedColliderGPU& endB,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    device const MRMeshBVHNodeGPU* meshNodes,
    device const MRMeshTriangleGPU* meshTriangles,
    const float alpha,
    thread ConvexQueryResult& query
) {
    const WorldShape shapeA = interpolatedCCDShape(
        colliderA,
        sourceA,
        startA,
        endA,
        alpha
    );
    const WorldShape shapeB = interpolatedCCDShape(
        colliderB,
        sourceB,
        startB,
        endB,
        alpha
    );
    if (shapeA.type == MR_SHAPE_HEIGHTFIELD ||
        shapeB.type == MR_SHAPE_HEIGHTFIELD) {
        query = {};
        query.status = MR_STEP_UNSUPPORTED;
        return false;
    }
    if (shapeA.type == MR_SHAPE_TRIANGLE_MESH ||
        shapeB.type == MR_SHAPE_TRIANGLE_MESH) {
        const bool meshIsA =
            shapeA.type == MR_SHAPE_TRIANGLE_MESH;
        return meshConvexDistance(
            meshIsA,
            meshIsA ? shapeB : shapeA,
            meshIsA ? shapeA : shapeB,
            meshIsA ? startB : startA,
            meshIsA ? sourceB : sourceA,
            meshIsA ? sourceA : sourceB,
            geometryHeaders,
            geometryVertices,
            meshNodes,
            meshTriangles,
            query
        );
    }
    supportMappedContacts(
        colliderA,
        colliderB,
        shapeA,
        shapeB,
        sourceA,
        sourceB,
        geometryHeaders,
        geometryVertices,
        float3(0.0f),
        query
    );
    return query.status == MR_STEP_SUCCESS;
}

bool speculativeEnvelopeCoversCCDPair(
    device const MRShapeGPU& sourceA,
    device const MRShapeGPU& sourceB,
    device const MRProjectedColliderGPU& startA,
    device const MRProjectedColliderGPU& startB,
    device const MRProjectedColliderGPU& endA,
    device const MRProjectedColliderGPU& endB,
    const float tolerance
) {
    const float orientationA = clamp(
        abs(dot(startA.rotation, endA.rotation)),
        0.0f,
        1.0f
    );
    const float orientationB = clamp(
        abs(dot(startB.rotation, endB.rotation)),
        0.0f,
        1.0f
    );
    const float motionA =
        length(
            endA.centerAndRadius.xyz -
            startA.centerAndRadius.xyz
        ) +
        2.0f * shapeSweepRadius(sourceA, startA) *
            sqrt(max(
                0.0f,
                1.0f - orientationA * orientationA
            ));
    const float motionB =
        length(
            endB.centerAndRadius.xyz -
            startB.centerAndRadius.xyz
        ) +
        2.0f * shapeSweepRadius(sourceB, startB) *
            sqrt(max(
                0.0f,
                1.0f - orientationB * orientationB
            ));
    const float marginA =
        as_type<float>(startA.statusAndFlags.z);
    const float marginB =
        as_type<float>(startB.statusAndFlags.z);
    return
        isfinite(motionA) &&
        isfinite(motionB) &&
        isfinite(marginA) &&
        isfinite(marginB) &&
        marginA + tolerance >= motionA &&
        marginB + tolerance >= motionB;
}

MRCCDPairGPU resolveCCDPair(
    const uint environment,
    const uint compiledPair,
    const MRCompiledCollisionPairGPU pair,
    device const MRShapeGPU* shapes,
    device const MRProjectedColliderGPU* currentProjected,
    device const MRProjectedColliderGPU* futureProjected,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    device const MRMeshBVHNodeGPU* meshNodes,
    device const MRMeshTriangleGPU* meshTriangles,
    device const MRMetalWorldContactDispatchGPU& dispatch,
    const float timestep
) {
    MRCCDPairGPU record = {};
    record.environment = environment;
    record.compiledPair = compiledPair;
    record.colliderA = pair.colliderA;
    record.colliderB = pair.colliderB;
    record.stableKeyLow = compiledPair;
    record.stableKeyHigh = environment;
    record.flags = MR_CCD_PAIR_VALID;
    record.status = MR_STEP_SUCCESS;
    device const MRShapeGPU& sourceA = shapes[pair.colliderA];
    device const MRShapeGPU& sourceB = shapes[pair.colliderB];
    device const MRProjectedColliderGPU& startA =
        currentProjected[pair.colliderA];
    device const MRProjectedColliderGPU& startB =
        currentProjected[pair.colliderB];
    device const MRProjectedColliderGPU& endA =
        futureProjected[pair.colliderA];
    device const MRProjectedColliderGPU& endB =
        futureProjected[pair.colliderB];
    const float tolerance = dispatch.ccdParameters.y;
    const float motionA =
        length(
            endA.centerAndRadius.xyz -
            startA.centerAndRadius.xyz
        ) +
        as_type<float>(startA.statusAndFlags.z);
    const float motionB =
        length(
            endB.centerAndRadius.xyz -
            startB.centerAndRadius.xyz
        ) +
        as_type<float>(startB.statusAndFlags.z);
    const float totalMotion = max(
        motionA + motionB,
        tolerance
    );
    record.intervalAndDistance.w =
        min(
            totalMotion / max(timestep, kTiny),
            dispatch.ccdParameters.w
        );

    ConvexQueryResult query = {};
    if (!ccdDistanceAtAlpha(
            pair.colliderA,
            pair.colliderB,
            sourceA,
            sourceB,
            startA,
            startB,
            endA,
            endB,
            geometryHeaders,
            geometryVertices,
            meshNodes,
            meshTriangles,
            0.0f,
            query
        )) {
        record.flags |= MR_CCD_PAIR_UNRESOLVED;
        record.status =
            query.status == MR_STEP_SUCCESS
            ? MR_STEP_DID_NOT_CONVERGE
            : query.status;
        return record;
    }
    record.intervalAndDistance.z = query.separation;
    record.normalAndIteration.xyz = query.normal;
    if (query.separation <= tolerance) {
        // Existing/touching contacts are resolved by the discrete manifold
        // solve at the current event configuration. Treating every resting
        // pair as a zero-time event would consume the split budget forever
        // even after the impact velocity has been removed.
        record.flags |= MR_CCD_PAIR_START_OVERLAP;
        record.intervalAndDistance.x = 0.0f;
        record.intervalAndDistance.y = 0.0f;
        return record;
    }

    if (sourceA.shapeType == MR_SHAPE_SPHERE &&
        sourceB.shapeType == MR_SHAPE_SPHERE) {
        record.flags |= MR_CCD_PAIR_ANALYTIC;
        const float3 startRelative =
            startB.centerAndRadius.xyz -
            startA.centerAndRadius.xyz;
        const float3 deltaRelative =
            (
                endB.centerAndRadius.xyz -
                startB.centerAndRadius.xyz
            ) -
            (
                endA.centerAndRadius.xyz -
                startA.centerAndRadius.xyz
            );
        const float radius =
            sourceA.dimensions.x + sourceB.dimensions.x;
        const float aa = dot(deltaRelative, deltaRelative);
        const float bb = 2.0f *
            dot(startRelative, deltaRelative);
        const float cc =
            dot(startRelative, startRelative) -
            radius * radius;
        const float discriminant = bb * bb - 4.0f * aa * cc;
        if (aa > kTiny && discriminant >= 0.0f) {
            const float alpha =
                (-bb - sqrt(discriminant)) / (2.0f * aa);
            if (alpha >= 0.0f && alpha <= 1.0f) {
                record.flags |= MR_CCD_PAIR_HAS_IMPACT;
                record.intervalAndDistance.x =
                    alpha * timestep;
                record.intervalAndDistance.y =
                    alpha * timestep;
                const float3 relativeAtImpact =
                    startRelative + alpha * deltaRelative;
                record.normalAndIteration.xyz =
                    length(relativeAtImpact) > kTiny
                    ? normalize(relativeAtImpact)
                    : coincidentNormal(
                          pair.colliderA,
                          pair.colliderB
                      );
            }
        }
        return record;
    }

    if (isSurfaceShapeType(sourceA.shapeType) ||
        isSurfaceShapeType(sourceB.shapeType)) {
        record.flags |= MR_CCD_PAIR_MESH;
    } else {
        record.flags |=
            MR_CCD_PAIR_CONSERVATIVE_ADVANCEMENT;
    }
    float alpha = 0.0f;
    const float minimumAlpha = min(
        1.0f,
        dispatch.ccdParameters.x /
            max(timestep, kTiny)
    );
    float previousAlpha = 0.0f;
    bool resolved = false;
    bool queryFailed = false;
    for (uint iteration = 0u;
         iteration <
             dispatch.maxConservativeAdvancementIterations;
         ++iteration) {
        record.normalAndIteration.w = float(iteration + 1u);
        if (query.separation <= tolerance) {
            resolved = true;
            record.flags |= MR_CCD_PAIR_HAS_IMPACT;
            record.intervalAndDistance.x =
                previousAlpha * timestep;
            record.intervalAndDistance.y =
                alpha * timestep;
            record.normalAndIteration.xyz = query.normal;
            break;
        }
        const float advancement = max(
            (query.separation - tolerance) / totalMotion,
            minimumAlpha
        );
        previousAlpha = alpha;
        alpha += advancement;
        if (alpha > 1.0f) {
            resolved = true;
            break;
        }
        if (!ccdDistanceAtAlpha(
                pair.colliderA,
                pair.colliderB,
                sourceA,
                sourceB,
                startA,
                startB,
                endA,
                endB,
                geometryHeaders,
                geometryVertices,
                meshNodes,
                meshTriangles,
                alpha,
                query
            )) {
            queryFailed = true;
            break;
        }
    }
    if (!resolved) {
        record.flags |= MR_CCD_PAIR_UNRESOLVED;
        if (!queryFailed &&
            speculativeEnvelopeCoversCCDPair(
                sourceA,
                sourceB,
                startA,
                startB,
                endA,
                endB,
                tolerance
            )) {
            record.flags |=
                MR_CCD_PAIR_SPECULATIVE_FALLBACK;
        }
        record.status = MR_STEP_DID_NOT_CONVERGE;
    }
    return record;
}

bool ccdEventPrecedes(
    const MRCCDPairGPU lhs,
    const MRCCDPairGPU rhs
) {
    // TOIs are finite and non-negative by construction, so their IEEE bit
    // ordering is the ordered-float ordering. Stable keys are the final
    // deterministic tie break for simultaneous impacts.
    const uint lhsTOI = as_type<uint>(
        max(lhs.intervalAndDistance.x, 0.0f)
    );
    const uint rhsTOI = as_type<uint>(
        max(rhs.intervalAndDistance.x, 0.0f)
    );
    if (lhsTOI != rhsTOI) {
        return lhsTOI < rhsTOI;
    }
    if (lhs.stableKeyHigh != rhs.stableKeyHigh) {
        return lhs.stableKeyHigh < rhs.stableKeyHigh;
    }
    return lhs.stableKeyLow < rhs.stableKeyLow;
}

MRCCDPairGPU resolveRodCCDPair(
    const uint environment,
    const uint globalPair,
    const MRRodToolPairGPU pair,
    device const MRRodColliderGPU* rodColliders,
    device const MRShapeGPU* rodShapeSources,
    device const MRShapeGPU* toolShapes,
    device const MRProjectedColliderGPU* currentRodProjected,
    device const MRProjectedColliderGPU* futureRodProjected,
    device const MRProjectedColliderGPU* currentToolProjected,
    device const MRProjectedColliderGPU* futureToolProjected,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    device const MRMeshBVHNodeGPU* meshNodes,
    device const MRMeshTriangleGPU* meshTriangles,
    device const MRMetalWorldContactDispatchGPU& dispatch,
    const float timestep
) {
    MRCCDPairGPU record = {};
    const uint encodedPair = 0x80000000u | globalPair;
    record.environment = environment;
    record.compiledPair = encodedPair;
    record.colliderA = dispatch.shapeCount + pair.rodCollider;
    record.colliderB = pair.rigidCollider;
    record.stableKeyLow = globalPair;
    record.stableKeyHigh = 0x80000000u | environment;
    record.flags =
        MR_CCD_PAIR_VALID |
        MR_CCD_PAIR_CONSERVATIVE_ADVANCEMENT;
    record.status = MR_STEP_SUCCESS;
    device const MRRodColliderGPU& rod =
        rodColliders[pair.rodCollider];
    device const MRShapeGPU& sourceA =
        rodShapeSources[pair.rodCollider];
    device const MRShapeGPU& sourceB =
        toolShapes[pair.rigidCollider];
    device const MRProjectedColliderGPU& startA =
        currentRodProjected[pair.rodCollider];
    device const MRProjectedColliderGPU& endA =
        futureRodProjected[pair.rodCollider];
    device const MRProjectedColliderGPU& startB =
        currentToolProjected[pair.rigidCollider];
    device const MRProjectedColliderGPU& endB =
        futureToolProjected[pair.rigidCollider];
    const float tolerance =
        dispatch.ccdParameters.y +
        max(rod.radiusAndOffsets.z, 0.0f);
    const float orientationA = clamp(
        abs(dot(startA.rotation, endA.rotation)),
        0.0f,
        1.0f
    );
    const float orientationB = clamp(
        abs(dot(startB.rotation, endB.rotation)),
        0.0f,
        1.0f
    );
    const float motionA =
        length(
            endA.centerAndRadius.xyz -
            startA.centerAndRadius.xyz
        ) +
        2.0f * max(
            startA.lowerAndHalfLength.w,
            endA.lowerAndHalfLength.w
        ) * sqrt(max(
            0.0f,
            1.0f - orientationA * orientationA
        )) +
        abs(
            endA.lowerAndHalfLength.w -
            startA.lowerAndHalfLength.w
        );
    const float motionB =
        length(
            endB.centerAndRadius.xyz -
            startB.centerAndRadius.xyz
        ) +
        2.0f * shapeSweepRadius(sourceB, startB) *
            sqrt(max(
                0.0f,
                1.0f - orientationB * orientationB
            ));
    const float totalMotion = max(
        motionA + motionB,
        tolerance
    );
    record.intervalAndDistance.w = min(
        totalMotion / max(timestep, kTiny),
        dispatch.ccdParameters.w
    );
    if (isSurfaceShapeType(sourceB.shapeType)) {
        record.flags |= MR_CCD_PAIR_MESH;
    }

    ConvexQueryResult query = {};
    if (!ccdDistanceAtAlpha(
            record.colliderA,
            record.colliderB,
            sourceA,
            sourceB,
            startA,
            startB,
            endA,
            endB,
            geometryHeaders,
            geometryVertices,
            meshNodes,
            meshTriangles,
            0.0f,
            query
        )) {
        record.flags |= MR_CCD_PAIR_UNRESOLVED;
        record.status =
            query.status == MR_STEP_SUCCESS
            ? MR_STEP_DID_NOT_CONVERGE
            : query.status;
        return record;
    }
    record.intervalAndDistance.z = query.separation;
    record.normalAndIteration.xyz = query.normal;
    if (query.separation <= tolerance) {
        record.flags |= MR_CCD_PAIR_START_OVERLAP;
        record.intervalAndDistance.x = 0.0f;
        record.intervalAndDistance.y = 0.0f;
        return record;
    }

    float alpha = 0.0f;
    float previousAlpha = 0.0f;
    const float minimumAlpha = min(
        1.0f,
        dispatch.ccdParameters.x /
            max(timestep, kTiny)
    );
    bool resolved = false;
    bool queryFailed = false;
    for (uint iteration = 0u;
         iteration <
             dispatch.maxConservativeAdvancementIterations;
         ++iteration) {
        record.normalAndIteration.w = float(iteration + 1u);
        if (query.separation <= tolerance) {
            resolved = true;
            record.flags |= MR_CCD_PAIR_HAS_IMPACT;
            record.intervalAndDistance.x =
                previousAlpha * timestep;
            record.intervalAndDistance.y =
                alpha * timestep;
            record.normalAndIteration.xyz = query.normal;
            break;
        }
        const float advancement = max(
            (query.separation - tolerance) / totalMotion,
            minimumAlpha
        );
        previousAlpha = alpha;
        alpha += advancement;
        if (alpha > 1.0f) {
            resolved = true;
            break;
        }
        if (!ccdDistanceAtAlpha(
                record.colliderA,
                record.colliderB,
                sourceA,
                sourceB,
                startA,
                startB,
                endA,
                endB,
                geometryHeaders,
                geometryVertices,
                meshNodes,
                meshTriangles,
                alpha,
                query
            )) {
            queryFailed = true;
            break;
        }
    }
    if (!resolved) {
        record.flags |= MR_CCD_PAIR_UNRESOLVED;
        if (!queryFailed &&
            speculativeEnvelopeCoversCCDPair(
                sourceA,
                sourceB,
                startA,
                startB,
                endA,
                endB,
                tolerance
            )) {
            record.flags |=
                MR_CCD_PAIR_SPECULATIVE_FALLBACK;
        }
        record.status = MR_STEP_DID_NOT_CONVERGE;
    }
    return record;
}

// Exact CCD is intentionally a rare per-environment pass. It consumes the
// stable compiled-pair stream, writes fixed per-environment segments with
// impact events first in deterministic TOI/key order, and leaves the broad
// speculative constraints authoritative if conservative advancement cannot
// certify an interval.
kernel void mr_world_resolve_ccd(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRShapeGPU* shapes [[buffer(1)]],
    device const MRCompiledCollisionPairGPU* eligiblePairs [[buffer(2)]],
    device const uint* overlapFlags [[buffer(3)]],
    device const MRProjectedColliderGPU* projectedColliders [[buffer(4)]],
    device const MRProjectedColliderGPU* futureProjectedColliders [[buffer(5)]],
    device const MRGeometryHeaderGPU* geometryHeaders [[buffer(6)]],
    device const float4* geometryVertices [[buffer(7)]],
    device const MRMeshBVHNodeGPU* meshNodes [[buffer(8)]],
    device const MRMeshTriangleGPU* meshTriangles [[buffer(9)]],
    device MRCCDPairGPU* candidates [[buffer(10)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(11)]],
    device const MRCCDEventStateGPU* eventStates [[buffer(12)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount ||
        dispatch.ccdMode != MR_WORLD_CCD_HYBRID) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    if (status.code != MR_STEP_SUCCESS) {
        return;
    }
    const uint projectionBase =
        environment * dispatch.shapeCount;
    const uint overlapBase =
        environment * dispatch.eligiblePairCount;
    const uint candidateBase =
        environment * dispatch.ccdCandidateCapacity;
    const float timestep = max(
        eventStates[environment].time.y,
        0.0f
    );
    uint candidateCount = 0u;
    uint eventCount = 0u;
    uint storedEventCount = 0u;
    uint unresolvedCount = 0u;
    uint unsafeUnresolvedCount = 0u;
    uint firstOverflow = MR_INVALID_INDEX;
    uint firstUnsafe = MR_INVALID_INDEX;
    for (uint compiledPair = 0u;
         compiledPair < dispatch.eligiblePairCount;
         ++compiledPair) {
        const MRCompiledCollisionPairGPU pair =
            eligiblePairs[compiledPair];
        const MRShapeGPU sourceA = shapes[pair.colliderA];
        const MRShapeGPU sourceB = shapes[pair.colliderB];
        if (overlapFlags[overlapBase + compiledPair] != 1u ||
            ((sourceA.flags | sourceB.flags) &
             MR_SHAPE_FLAG_ENABLE_CCD) == 0u) {
            continue;
        }
        const uint destination = candidateCount++;
        MRCCDPairGPU record = resolveCCDPair(
            environment,
            compiledPair,
            pair,
            shapes,
            projectedColliders + projectionBase,
            futureProjectedColliders + projectionBase,
            geometryHeaders,
            geometryVertices,
            meshNodes,
            meshTriangles,
            dispatch,
            timestep
        );
        const bool hasImpact =
            (record.flags & MR_CCD_PAIR_HAS_IMPACT) != 0u;
        if (hasImpact) {
            ++eventCount;
        }
        if ((record.flags & MR_CCD_PAIR_UNRESOLVED) != 0u) {
            ++unresolvedCount;
            if ((record.flags &
                 MR_CCD_PAIR_SPECULATIVE_FALLBACK) == 0u) {
                ++unsafeUnresolvedCount;
                firstUnsafe = min(
                    firstUnsafe,
                    compiledPair
                );
            }
        }
        if (destination >= dispatch.ccdCandidateCapacity) {
            if (firstOverflow == MR_INVALID_INDEX) {
                firstOverflow = compiledPair;
            }
            continue;
        }
        if (!hasImpact) {
            candidates[candidateBase + destination] = record;
            continue;
        }

        // Exact CCD is rare and capped. A deterministic in-place insertion
        // keeps the event prefix ordered without a second global sort or an
        // atomic append. Non-events retain compiled-pair order after it.
        uint insertion = storedEventCount;
        while (insertion > 0u &&
               ccdEventPrecedes(
                   record,
                   candidates[
                       candidateBase + insertion - 1u
                   ]
               )) {
            --insertion;
        }
        for (uint move = destination;
             move > insertion;
             --move) {
            candidates[candidateBase + move] =
                candidates[candidateBase + move - 1u];
        }
        candidates[candidateBase + insertion] = record;
        ++storedEventCount;
    }
    status.requiredCCDCandidates = max(
        status.requiredCCDCandidates,
        candidateCount
    );
    status.requiredCCDEvents += eventCount;
    status.ccdCandidates = min(
        candidateCount,
        dispatch.ccdCandidateCapacity
    );
    status.ccdEvents = min(
        eventCount,
        min(
            dispatch.ccdEventCapacity,
            dispatch.maxCCDEvents
        )
    );
    status.unresolvedCCDCount = unresolvedCount;
    if (eventCount > dispatch.maxCCDEvents) {
        status.queueFlags |=
            1u << MR_WORLD_WORK_CCD;
    }
    if (candidateCount > dispatch.ccdCandidateCapacity) {
        status.code = MR_STEP_CCD_CAPACITY_OVERFLOW;
        status.firstFailingPair = firstOverflow;
        status.firstFailingStableKeyLow = firstOverflow;
        status.firstFailingStableKeyHigh = environment;
    } else if (eventCount > dispatch.maxCCDEvents) {
        status.code = MR_STEP_CCD_EVENT_BUDGET_EXHAUSTED;
        const uint budgetEvent = dispatch.maxCCDEvents;
        status.firstFailingPair =
            budgetEvent < storedEventCount
            ? candidates[
                  candidateBase + budgetEvent
              ].compiledPair
            : firstOverflow;
    } else if (eventCount > dispatch.ccdEventCapacity) {
        status.code = MR_STEP_CCD_CAPACITY_OVERFLOW;
        const uint overflowEvent =
            dispatch.ccdEventCapacity;
        status.firstFailingPair =
            overflowEvent < storedEventCount
            ? candidates[
                  candidateBase + overflowEvent
              ].compiledPair
            : firstOverflow;
    } else if (unsafeUnresolvedCount != 0u) {
        status.code = MR_STEP_DID_NOT_CONVERGE;
        status.firstFailingPair = firstUnsafe;
    }
    if (status.code != MR_STEP_SUCCESS &&
        status.firstFailingPair != MR_INVALID_INDEX) {
        status.firstFailingStableKeyLow =
            status.firstFailingPair;
        status.firstFailingStableKeyHigh = environment;
        status.firstFailingEventKeyLow =
            status.firstFailingPair;
        status.firstFailingEventKeyHigh = environment;
    }
    statuses[environment] = status;
}

// Merges procedural rod/tool TOIs into the rigid candidate segment in the same
// deterministic (ordered TOI, stable key) order. The high bit of compiledPair
// identifies the rod namespace without changing MRCCDPairGPU's ABI.
kernel void mr_world_resolve_rod_ccd(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRRodColliderGPU* rodColliders [[buffer(1)]],
    device const MRShapeGPU* rodShapeSources [[buffer(2)]],
    device const MRRodToolPairGPU* toolPairs [[buffer(3)]],
    device const MRShapeGPU* toolShapes [[buffer(4)]],
    device const MRProjectedColliderGPU* projectedRodColliders [[buffer(5)]],
    device const MRProjectedColliderGPU* futureProjectedRodColliders [[buffer(6)]],
    device const MRProjectedColliderGPU* projectedToolColliders [[buffer(7)]],
    device const MRProjectedColliderGPU* futureProjectedToolColliders [[buffer(8)]],
    device const MRGeometryHeaderGPU* geometryHeaders [[buffer(9)]],
    device const float4* geometryVertices [[buffer(10)]],
    device const MRMeshBVHNodeGPU* meshNodes [[buffer(11)]],
    device const MRMeshTriangleGPU* meshTriangles [[buffer(12)]],
    device MRCCDPairGPU* candidates [[buffer(13)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(14)]],
    device const MRCCDEventStateGPU* eventStates [[buffer(15)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount ||
        dispatch.ccdMode != MR_WORLD_CCD_HYBRID ||
        dispatch.rodToolPairCount == 0u) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    if (status.code != MR_STEP_SUCCESS) {
        return;
    }
    const uint candidateBase =
        environment * dispatch.ccdCandidateCapacity;
    const uint rodProjectionBase =
        environment * dispatch.rodEdgeCount;
    const uint toolProjectionBase =
        environment * dispatch.shapeCount;
    const float timestep = max(
        eventStates[environment].time.y,
        0.0f
    );
    uint candidateCount = status.ccdCandidates;
    uint eventCount = status.ccdEvents;
    uint addedEvents = 0u;
    uint unresolvedCount = status.unresolvedCCDCount;
    uint unsafeUnresolvedCount = 0u;
    uint firstOverflow = MR_INVALID_INDEX;
    uint firstUnsafe = MR_INVALID_INDEX;
    for (uint globalPair = 0u;
         globalPair < dispatch.rodToolPairCount;
         ++globalPair) {
        const MRRodToolPairGPU pair = toolPairs[globalPair];
        if ((pair.flags &
             (
                 MR_ROD_TOOL_PAIR_VALID |
                 MR_ROD_TOOL_PAIR_ENABLE_CCD
             )) !=
                (
                    MR_ROD_TOOL_PAIR_VALID |
                    MR_ROD_TOOL_PAIR_ENABLE_CCD
                ) ||
            pair.rodCollider >= dispatch.rodEdgeCount ||
            pair.rigidCollider >= dispatch.shapeCount) {
            continue;
        }
        device const MRProjectedColliderGPU& rodProjected =
            projectedRodColliders[
                rodProjectionBase + pair.rodCollider
            ];
        device const MRProjectedColliderGPU& toolProjected =
            projectedToolColliders[
                toolProjectionBase + pair.rigidCollider
            ];
        const MRShapeGPU toolSource =
            toolShapes[pair.rigidCollider];
        if (rodProjected.statusAndFlags.x != MR_STEP_SUCCESS ||
            toolProjected.statusAndFlags.x != MR_STEP_SUCCESS ||
            (
                toolSource.shapeType != MR_SHAPE_PLANE &&
                (
                    any(
                        rodProjected.lowerAndHalfLength.xyz >
                        toolProjected
                            .upperAndContactOffset.xyz
                    ) ||
                    any(
                        toolProjected.lowerAndHalfLength.xyz >
                        rodProjected
                            .upperAndContactOffset.xyz
                    )
                )
            )) {
            continue;
        }
        MRCCDPairGPU record = resolveRodCCDPair(
            environment,
            globalPair,
            pair,
            rodColliders,
            rodShapeSources,
            toolShapes,
            projectedRodColliders + rodProjectionBase,
            futureProjectedRodColliders + rodProjectionBase,
            projectedToolColliders + toolProjectionBase,
            futureProjectedToolColliders + toolProjectionBase,
            geometryHeaders,
            geometryVertices,
            meshNodes,
            meshTriangles,
            dispatch,
            timestep
        );
        const bool hasImpact =
            (record.flags & MR_CCD_PAIR_HAS_IMPACT) != 0u;
        const uint destination = candidateCount;
        ++candidateCount;
        if ((record.flags & MR_CCD_PAIR_UNRESOLVED) != 0u) {
            ++unresolvedCount;
            if ((record.flags &
                 MR_CCD_PAIR_SPECULATIVE_FALLBACK) == 0u) {
                ++unsafeUnresolvedCount;
                firstUnsafe = min(
                    firstUnsafe,
                    record.compiledPair
                );
            }
        }
        if (destination >= dispatch.ccdCandidateCapacity) {
            firstOverflow = min(
                firstOverflow,
                record.compiledPair
            );
            if (hasImpact) {
                ++eventCount;
                ++addedEvents;
            }
            continue;
        }
        if (!hasImpact) {
            candidates[candidateBase + destination] = record;
            continue;
        }
        uint insertion = eventCount;
        while (insertion > 0u &&
               ccdEventPrecedes(
                   record,
                   candidates[
                       candidateBase + insertion - 1u
                   ]
               )) {
            --insertion;
        }
        for (uint move = destination;
             move > insertion;
             --move) {
            candidates[candidateBase + move] =
                candidates[candidateBase + move - 1u];
        }
        candidates[candidateBase + insertion] = record;
        ++eventCount;
        ++addedEvents;
    }
    status.requiredCCDCandidates = max(
        status.requiredCCDCandidates,
        candidateCount
    );
    status.requiredCCDEvents += addedEvents;
    status.ccdCandidates = min(
        candidateCount,
        dispatch.ccdCandidateCapacity
    );
    status.ccdEvents = min(
        eventCount,
        min(
            dispatch.ccdEventCapacity,
            dispatch.maxCCDEvents
        )
    );
    status.unresolvedCCDCount = unresolvedCount;
    if (candidateCount > dispatch.ccdCandidateCapacity) {
        status.code = MR_STEP_CCD_CAPACITY_OVERFLOW;
        status.firstFailingPair = firstOverflow;
    } else if (eventCount > dispatch.maxCCDEvents) {
        status.code = MR_STEP_CCD_EVENT_BUDGET_EXHAUSTED;
        status.firstFailingPair =
            candidates[
                candidateBase +
                min(
                    dispatch.maxCCDEvents,
                    dispatch.ccdCandidateCapacity - 1u
                )
            ].compiledPair;
    } else if (eventCount > dispatch.ccdEventCapacity) {
        status.code = MR_STEP_CCD_CAPACITY_OVERFLOW;
        status.firstFailingPair =
            candidates[
                candidateBase +
                min(
                    dispatch.ccdEventCapacity,
                    dispatch.ccdCandidateCapacity - 1u
                )
            ].compiledPair;
    } else if (unsafeUnresolvedCount != 0u) {
        status.code = MR_STEP_DID_NOT_CONVERGE;
        status.firstFailingPair = firstUnsafe;
    }
    if (status.code != MR_STEP_SUCCESS &&
        status.firstFailingPair != MR_INVALID_INDEX) {
        status.firstFailingStableKeyLow =
            status.firstFailingPair;
        status.firstFailingStableKeyHigh =
            0x80000000u | environment;
        status.firstFailingEventKeyLow =
            status.firstFailingPair;
        status.firstFailingEventKeyHigh =
            0x80000000u | environment;
    }
    statuses[environment] = status;
}

kernel void mr_world_initialize_ccd_event_state(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device MRCCDEventStateGPU* eventStatesA [[buffer(1)]],
    device MRCCDEventStateGPU* eventStatesB [[buffer(2)]],
    device MRCCDImpactClusterGPU* clusters [[buffer(3)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(4)]],
    constant MRMetalWorldPassGPU& pass [[buffer(5)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRCCDEventStateGPU state = {};
    state.environment = environment;
    state.flags =
        dispatch.ccdMode == MR_WORLD_CCD_HYBRID
        ? MR_CCD_EVENT_ACTIVE
        : MR_CCD_EVENT_FINISHED;
    state.generation =
        pass.physicsSubstep *
        dispatch.maxCCDAdvanceSolvePasses;
    state.lastStableKeyLow = MR_INVALID_INDEX;
    state.lastStableKeyHigh = MR_INVALID_INDEX;
    state.time = float4(
        0.0f,
        dispatch.timestepAndBias.x,
        0.0f,
        dispatch.timestepAndBias.x
    );
    state.cluster = uint4(
        MR_INVALID_INDEX,
        0u,
        0u,
        0u
    );
    MRCCDImpactClusterGPU cluster = {};
    cluster.environment = environment;
    cluster.generation = state.generation;
    cluster.firstEventSlot = MR_INVALID_INDEX;
    cluster.stableKeyLow = MR_INVALID_INDEX;
    cluster.stableKeyHigh = MR_INVALID_INDEX;
    cluster.interval = float4(
        dispatch.timestepAndBias.x,
        dispatch.timestepAndBias.x,
        dispatch.timestepAndBias.x,
        dispatch.ccdEventParameters.x
    );
    eventStatesA[environment] = state;
    eventStatesB[environment] = state;
    clusters[environment] = cluster;

    MRMetalWorldContactStatusGPU status = statuses[environment];
    status.ccdAdvanceCount = 0u;
    status.clusteredCCDImpacts = 0u;
    status.zeroTimeCCDReplays = 0u;
    status.speculativeRemainderUses = 0u;
    status.eventGeneration = state.generation;
    status.firstFailingEventKeyLow = MR_INVALID_INDEX;
    status.firstFailingEventKeyHigh = MR_INVALID_INDEX;
    status.requiredCCDCandidates = 0u;
    status.requiredCCDEvents = 0u;
    status.reservedEvent0 = 0u;
    status.reservedEvent1 = 0u;
    status.eventTimes = float4(
        0.0f,
        dispatch.timestepAndBias.x,
        dispatch.timestepAndBias.x,
        dispatch.timestepAndBias.x
    );
    statuses[environment] = status;
}

// Marks already-finished environments with a private in-graph sentinel. All
// existing contact kernels already ignore non-success records, so a fixed
// MLX grid can encode every event pass without mutating finished worlds.
kernel void mr_world_prepare_ccd_event_pass(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRCCDEventStateGPU* eventStates [[buffer(1)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(2)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    const MRCCDEventStateGPU state = eventStates[environment];
    MRMetalWorldContactStatusGPU status = statuses[environment];
    if ((state.flags & MR_CCD_EVENT_FAILED) != 0u) {
        return;
    }
    if ((state.flags & MR_CCD_EVENT_FINISHED) != 0u) {
        if (status.code == MR_STEP_SUCCESS) {
            status.code = MR_STEP_FIXED_BUDGET_COMPLETE;
        }
    } else if (status.code == MR_STEP_FIXED_BUDGET_COMPLETE) {
        status.code = MR_STEP_SUCCESS;
    }
    statuses[environment] = status;
}

// Materializes the sorted CCD prefix into the next transient event cursor.
// Candidate intervals and selected duration are physical seconds relative to
// the current cursor's remaining interval.
kernel void mr_world_select_ccd_event_state(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRCCDPairGPU* candidates [[buffer(1)]],
    device const MRCCDEventStateGPU* eventStatesA [[buffer(2)]],
    device MRCCDEventStateGPU* eventStatesB [[buffer(3)]],
    device MRCCDImpactClusterGPU* clusters [[buffer(4)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(5)]],
    constant MRMetalWorldPassGPU& pass [[buffer(6)]],
    constant uint& eventPass [[buffer(7)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    const MRCCDEventStateGPU previous =
        eventStatesA[environment];
    MRCCDEventStateGPU state = previous;
    MRCCDImpactClusterGPU cluster = {};
    cluster.environment = environment;
    cluster.generation = previous.generation;
    cluster.firstEventSlot = MR_INVALID_INDEX;
    cluster.stableKeyLow = MR_INVALID_INDEX;
    cluster.stableKeyHigh = MR_INVALID_INDEX;
    cluster.interval = float4(
        previous.time.y,
        previous.time.y,
        previous.time.y,
        dispatch.ccdEventParameters.x
    );
    if (status.code != MR_STEP_SUCCESS ||
        dispatch.ccdMode != MR_WORLD_CCD_HYBRID ||
        (previous.flags & MR_CCD_EVENT_ACTIVE) == 0u) {
        if (status.code != MR_STEP_SUCCESS &&
            status.code != MR_STEP_FIXED_BUDGET_COMPLETE) {
            state.flags |= MR_CCD_EVENT_FAILED;
        }
        eventStatesB[environment] = state;
        clusters[environment] = cluster;
        status.eventGeneration = state.generation;
        statuses[environment] = status;
        return;
    }
    state.flags &=
        ~static_cast<uint>(
            MR_CCD_EVENT_FINISHED |
            MR_CCD_EVENT_HAS_IMPACT |
            MR_CCD_EVENT_SPECULATIVE_REMAINDER |
            MR_CCD_EVENT_ZERO_TIME_REPLAY
        );
    state.flags |= MR_CCD_EVENT_ACTIVE;
    state.generation =
        pass.physicsSubstep *
            dispatch.maxCCDAdvanceSolvePasses +
        eventPass;
    state.time.w = max(previous.time.y, 0.0f);
    state.cluster = uint4(
        MR_INVALID_INDEX,
        0u,
        0u,
        0u
    );
    status.reservedEvent0 = 0u;
    cluster.generation = state.generation;
    cluster.interval = float4(
        state.time.y,
        state.time.y,
        state.time.y,
        dispatch.ccdEventParameters.x
    );

    const uint storedEvents = min(
        status.ccdEvents,
        min(
            dispatch.ccdCandidateCapacity,
            dispatch.ccdEventCapacity
        )
    );
    if (storedEvents != 0u) {
        const uint candidateBase =
            environment * dispatch.ccdCandidateCapacity;
        const MRCCDPairGPU first = candidates[candidateBase];
        const float earliest = clamp(
            first.intervalAndDistance.x,
            0.0f,
            state.time.y
        );
        const float tolerance = max(
            dispatch.ccdEventParameters.x,
            max(
                dispatch.ccdParameters.y,
                8.0f * abs(
                    nextafter(earliest, 2.0f) - earliest
                )
            )
        );
        uint clustered = 0u;
        for (uint eventSlot = 0u;
             eventSlot < storedEvents;
             ++eventSlot) {
            const MRCCDPairGPU event =
                candidates[candidateBase + eventSlot];
            if (event.intervalAndDistance.x >
                earliest + tolerance) {
                break;
            }
            ++clustered;
        }
        state.flags |= MR_CCD_EVENT_HAS_IMPACT;
        state.simultaneousEventCount = clustered;
        state.lastStableKeyLow = first.stableKeyLow;
        state.lastStableKeyHigh = first.stableKeyHigh;
        state.time.w = earliest;
        state.cluster = uint4(
            0u,
            clustered,
            status.unresolvedCCDCount == 0u ? 1u : 0u,
            0u
        );
        cluster.firstEventSlot = 0u;
        cluster.eventCount = clustered;
        cluster.stableKeyLow = first.stableKeyLow;
        cluster.stableKeyHigh = first.stableKeyHigh;
        cluster.flags = MR_CCD_EVENT_HAS_IMPACT;
        cluster.interval = float4(
            earliest,
            first.intervalAndDistance.x,
            first.intervalAndDistance.y,
            tolerance
        );
        status.reservedEvent0 = clustered;
        status.clusteredCCDImpacts += clustered;
        status.eventTimes.z = min(
            status.eventTimes.z,
            state.time.w
        );
        status.eventTimes.w = state.time.w;
    } else {
        status.eventTimes.w = state.time.y;
    }
    eventStatesB[environment] = state;
    clusters[environment] = cluster;
    status.eventGeneration = state.generation;
    statuses[environment] = status;
}

// Accepts the duration materialized and solved by one literal event pass.
// Remaining time stays active for the next statically encoded pass. A pass
// budget can never silently discard time.
kernel void mr_world_finalize_ccd_event_state(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device MRCCDEventStateGPU* eventStates [[buffer(1)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(2)]],
    constant uint& eventPass [[buffer(3)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    MRCCDEventStateGPU state = eventStates[environment];
    if (status.code == MR_STEP_FIXED_BUDGET_COMPLETE &&
        (state.flags & MR_CCD_EVENT_FINISHED) != 0u) {
        status.code = MR_STEP_SUCCESS;
        status.eventTimes.x = state.time.z;
        status.eventTimes.y = state.time.y;
        statuses[environment] = status;
        return;
    }
    if (status.code != MR_STEP_SUCCESS) {
        state.flags |= MR_CCD_EVENT_FAILED;
        eventStates[environment] = state;
        return;
    }
    const float fullTolerance = max(
        dispatch.ccdEventParameters.y,
        dispatch.ccdParameters.y
    );
    const float selected = clamp(
        state.time.w,
        0.0f,
        state.time.y
    );
    const bool hasImpact =
        (state.flags & MR_CCD_EVENT_HAS_IMPACT) != 0u;
    if (hasImpact) {
        ++state.splitCount;
        ++status.ccdAdvanceCount;
        if (selected <= fullTolerance) {
            state.flags |= MR_CCD_EVENT_ZERO_TIME_REPLAY;
            ++state.zeroTimeReplayCount;
            ++status.zeroTimeCCDReplays;
            if (state.zeroTimeReplayCount >
                dispatch.maxCCDZeroTimeReplays) {
                status.code = MR_STEP_DID_NOT_CONVERGE;
                status.firstFailingEventKeyLow =
                    state.lastStableKeyLow;
                status.firstFailingEventKeyHigh =
                    state.lastStableKeyHigh;
                state.flags |= MR_CCD_EVENT_FAILED;
            }
        } else {
            state.zeroTimeReplayCount = 0u;
        }
        state.time.x += selected;
        state.time.z += selected;
        state.time.y = max(state.time.y - selected, 0.0f);
        if (status.code == MR_STEP_SUCCESS &&
            state.time.y <= fullTolerance) {
            state.time.y = 0.0f;
            state.flags &=
                ~static_cast<uint>(MR_CCD_EVENT_ACTIVE);
            state.flags |= MR_CCD_EVENT_FINISHED;
        }
    } else {
        state.time.x += state.time.y;
        state.time.z += state.time.y;
        state.time.y = 0.0f;
        state.flags &=
            ~static_cast<uint>(MR_CCD_EVENT_ACTIVE);
        state.flags |= MR_CCD_EVENT_FINISHED;
    }
    if (status.code == MR_STEP_SUCCESS &&
        (state.flags & MR_CCD_EVENT_ACTIVE) != 0u &&
        eventPass + 1u >=
            dispatch.maxCCDAdvanceSolvePasses) {
        status.code =
            MR_STEP_CCD_EVENT_BUDGET_EXHAUSTED;
        status.firstFailingEventKeyLow =
            state.lastStableKeyLow;
        status.firstFailingEventKeyHigh =
            state.lastStableKeyHigh;
        status.firstFailingStableKeyLow =
            state.lastStableKeyLow;
        status.firstFailingStableKeyHigh =
            state.lastStableKeyHigh;
        state.flags |= MR_CCD_EVENT_FAILED;
    }
    status.eventTimes.x = state.time.z;
    status.eventTimes.y = state.time.y;
    eventStates[environment] = state;
    statuses[environment] = status;
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

// Generic arbitrary-length exclusive scan building block. The host encodes
// as many levels as the configured element count requires; no intermediate
// count becomes CPU-visible.
kernel void mr_world_scan_blocks(
    device const uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    device uint* blockSums [[buffer(2)]],
    constant MRScanLevelGPU& level [[buffer(3)]],
    const uint globalIndex [[thread_position_in_grid]],
    const uint localIndex [[thread_index_in_threadgroup]],
    const uint blockIndex [[threadgroup_position_in_grid]]
) {
    threadgroup uint values[MR_BROADPHASE_SCAN_BLOCK_SIZE];
    uint value = 0u;
    if (globalIndex < level.elementCount) {
        value = input[level.inputOffset + globalIndex];
        if ((level.flags & MR_SCAN_BOOLEAN_INPUT) != 0u) {
            value = value == 1u ? 1u : 0u;
        }
    }
    values[localIndex] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u;
         stride < MR_BROADPHASE_SCAN_BLOCK_SIZE;
         stride <<= 1u) {
        const uint addend =
            localIndex >= stride
            ? values[localIndex - stride]
            : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        values[localIndex] += addend;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (globalIndex < level.elementCount) {
        output[level.outputOffset + globalIndex] =
            values[localIndex] - value;
    }
    if (localIndex + 1u == MR_BROADPHASE_SCAN_BLOCK_SIZE &&
        blockIndex < level.blockCount) {
        blockSums[level.blockSumOffset + blockIndex] =
            values[localIndex];
    }
}

// Adds recursively-scanned parent block offsets while descending the scan
// hierarchy.
kernel void mr_world_scan_add_block_offsets(
    device uint* output [[buffer(0)]],
    device const uint* parentOffsets [[buffer(1)]],
    constant MRScanLevelGPU& level [[buffer(2)]],
    const uint globalIndex [[thread_position_in_grid]]
) {
    if (globalIndex >= level.elementCount) {
        return;
    }
    const uint blockIndex =
        globalIndex / MR_BROADPHASE_SCAN_BLOCK_SIZE;
    output[level.outputOffset + globalIndex] +=
        parentOffsets[level.parentOffset + blockIndex];
}

inline uint worldPairWorkClass(
    const MRCompiledCollisionPairGPU pair,
    device const MRShapeGPU* shapes
) {
    if (pair.pairClass == MR_COLLISION_PAIR_BOX_BOX) {
        return MR_WORLD_WORK_SAT_CLIP;
    }
    if (pair.pairClass == MR_COLLISION_PAIR_MESH) {
        return MR_WORLD_WORK_MESH;
    }
    if (pair.pairClass == MR_COLLISION_PAIR_CONVEX) {
        return
            shapes[pair.colliderA].shapeType == MR_SHAPE_CONVEX ||
            shapes[pair.colliderB].shapeType == MR_SHAPE_CONVEX
            ? MR_WORLD_WORK_HULL_GJK
            : MR_WORLD_WORK_PRIMITIVE_GJK;
    }
    return MR_WORLD_WORK_ANALYTIC;
}

// Builds one class-specific boolean stream from the broadphase result.
// Running this only for classes present in immutable compiled topology keeps
// homogeneous RL scenes cheap while preventing algorithm-divergent lanes.
kernel void mr_world_flag_pair_work_class(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRShapeGPU* shapes [[buffer(1)]],
    device const MRCompiledCollisionPairGPU* eligiblePairs [[buffer(2)]],
    device const uint* overlapFlags [[buffer(3)]],
    device uint* classFlags [[buffer(4)]],
    constant uint4& classConfig [[buffer(5)]],
    const uint globalIndex [[thread_position_in_grid]]
) {
    const uint total =
        dispatch.environmentCount * dispatch.eligiblePairCount;
    if (globalIndex >= total) {
        return;
    }
    const uint compiledPair =
        globalIndex % dispatch.eligiblePairCount;
    classFlags[globalIndex] =
        overlapFlags[globalIndex] == 1u &&
        worldPairWorkClass(
            eligiblePairs[compiledPair],
            shapes
        ) == classConfig.x
        ? 1u
        : 0u;
}

// Stable scatter from the environment-major overlap stream into one compact
// class-partitioned pair queue. Atomics never determine placement: class
// order is fixed, and each class retains environment/stable-pair order.
kernel void mr_world_scatter_pair_queue(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRShapeGPU* shapes [[buffer(1)]],
    device const MRCompiledCollisionPairGPU* eligiblePairs [[buffer(2)]],
    device const uint* classFlags [[buffer(3)]],
    device const uint* offsets [[buffer(4)]],
    device MRPairWorkGPU* queue [[buffer(5)]],
    device MRWorkQueueHeaderGPU* headers [[buffer(6)]],
    constant uint4& classConfig [[buffer(7)]],
    const uint globalIndex [[thread_position_in_grid]]
) {
    const uint total =
        dispatch.environmentCount * dispatch.eligiblePairCount;
    const uint workClass = classConfig.x;
    const uint activeClassMask = classConfig.y;
    ulong classBaseWide = 0u;
    for (uint previousClass = 0u;
         previousClass < workClass;
         ++previousClass) {
        if ((activeClassMask &
             (1u << previousClass)) != 0u) {
            classBaseWide +=
                static_cast<ulong>(
                    headers[previousClass].required
                );
        }
    }
    const uint capacity =
        dispatch.environmentCount * dispatch.pairCapacity;
    const uint classBase = static_cast<uint>(
        min(classBaseWide, static_cast<ulong>(capacity))
    );
    const uint classCount =
        total == 0u
        ? 0u
        : offsets[total - 1u] +
            (classFlags[total - 1u] == 1u ? 1u : 0u);
    const uint available = capacity - classBase;
    const uint residentCount = min(classCount, available);
    if (globalIndex == 0u) {
        device MRWorkQueueHeaderGPU& header =
            headers[workClass];
        header = {};
        header.count = residentCount;
        header.capacity = available;
        header.required = classCount;
        header.workClass = workClass;
        header.overflow =
            classCount > available ? 1u : 0u;
        header.indirect.threadgroupsX =
            (
                header.count +
                MR_WORLD_QUEUE_THREADS_PER_THREADGROUP - 1u
            ) / MR_WORLD_QUEUE_THREADS_PER_THREADGROUP;
        header.indirect.threadgroupsY = 1u;
        header.indirect.threadgroupsZ = 1u;
        header.indirect.activeCount = header.count;
        header.reserved0 = classBase;
    }
    if (globalIndex >= total ||
        classFlags[globalIndex] != 1u) {
        return;
    }
    const uint classOffset = offsets[globalIndex];
    if (classOffset >= residentCount) {
        if (classOffset == residentCount) {
            const uint environment =
                globalIndex / dispatch.eligiblePairCount;
            const uint compiledPair =
                globalIndex -
                environment * dispatch.eligiblePairCount;
            headers[workClass].firstStableKeyLow =
                compiledPair;
            headers[workClass].firstStableKeyHigh =
                environment;
        }
        return;
    }
    const uint destination = classBase + classOffset;
    const uint environment =
        globalIndex / dispatch.eligiblePairCount;
    const uint compiledPair =
        globalIndex - environment * dispatch.eligiblePairCount;
    MRPairWorkGPU work = {};
    work.environment = environment;
    work.compiledPair = compiledPair;
    const uint localCacheSlot =
        eligiblePairs[compiledPair].convexCacheSlot;
    work.cacheSlot = localCacheSlot == MR_INVALID_INDEX
        ? MR_INVALID_INDEX
        :
        environment * dispatch.convexCacheStride +
            localCacheSlot;
    work.workClass = workClass;
    work.stableKeyLow = compiledPair;
    work.stableKeyHigh = environment;
    work.reserved = destination;
    queue[destination] = work;
}

// SIMD-dense narrowphase. One lane consumes one compact work item and writes
// a deterministic compact-queue staging span. The eligible-pair count record
// retains that slot, while the later segmented scan continues to own canonical
// compiled-pair ordering for manifolds and ConstraintIR.
kernel void mr_world_narrowphase_pair_queue(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRShapeGPU* shapes [[buffer(1)]],
    device const MRCompiledCollisionPairGPU* eligiblePairs [[buffer(2)]],
    device const MRProjectedColliderGPU* projectedColliders [[buffer(3)]],
    device const MRPairWorkGPU* queue [[buffer(4)]],
    device MRWorkQueueHeaderGPU* headers [[buffer(5)]],
    device MRRawContactGPU* pairRawContacts [[buffer(6)]],
    device uint2* pairRawCounts [[buffer(7)]],
    constant uint4& classConfig [[buffer(8)]],
    const uint workerIndex [[thread_position_in_grid]],
    const uint workerCount [[threads_per_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    const uint workClass = classConfig.x;
    const MRWorkQueueHeaderGPU header = headers[workClass];
    const uint stride = max(workerCount, 1u);
    device atomic_uint* workerCursor =
        reinterpret_cast<device atomic_uint*>(
            &headers[workClass].workerCursor
        );
    for (uint iteration = 0u;; ++iteration) {
        uint queueIndex = 0u;
        if (classConfig.z != 0u) {
            uint claimedBase = 0u;
            if (lane == 0u) {
                claimedBase = atomic_fetch_add_explicit(
                    workerCursor,
                    MR_WAVE32_CONTACTS_PER_TILE,
                    memory_order_relaxed
                );
            }
            claimedBase =
                simd_broadcast_first(claimedBase);
            if (claimedBase >= header.count) {
                break;
            }
            queueIndex = claimedBase + lane;
            if (queueIndex >= header.count) {
                continue;
            }
        } else {
            const ulong linearIndex =
                static_cast<ulong>(workerIndex) +
                static_cast<ulong>(iteration) * stride;
            if (linearIndex >= header.count) {
                break;
            }
            queueIndex = static_cast<uint>(linearIndex);
        }
        const MRPairWorkGPU work =
            queue[header.reserved0 + queueIndex];
        if (work.environment >= dispatch.environmentCount ||
            work.compiledPair >= dispatch.eligiblePairCount ||
            work.workClass != workClass ||
            (workClass != MR_WORLD_WORK_ANALYTIC &&
             workClass != MR_WORLD_WORK_SAT_CLIP)) {
            continue;
        }
        const MRCompiledCollisionPairGPU pair =
            eligiblePairs[work.compiledPair];
        const uint projectionBase =
            work.environment * dispatch.shapeCount;
        WorldShape shapeA;
        WorldShape shapeB;
        uint failureCode = MR_STEP_SUCCESS;
        if (!loadProjectedCollider(
                pair.colliderA,
                shapes[pair.colliderA],
                projectedColliders[
                    projectionBase + pair.colliderA
                ],
                shapeA,
                failureCode
            ) ||
            !loadProjectedCollider(
                pair.colliderB,
                shapes[pair.colliderB],
                projectedColliders[
                    projectionBase + pair.colliderB
                ],
                shapeB,
                failureCode
            )) {
            pairRawCounts[
                work.environment *
                    dispatch.eligiblePairCount +
                work.compiledPair
            ] = uint2(
                0x80000000u | failureCode,
                work.reserved
            );
            continue;
        }
        const ContactBatch contacts = generateContacts(
            pair.colliderA,
            pair.colliderB,
            pair.pairClass,
            shapeA,
            shapeB
        );
        const uint stagingBase =
            work.reserved * MR_METAL_WORLD_RAW_CONTACTS_PER_PAIR;
        uint count = contacts.count;
        for (uint index = 0u;
             index < contacts.count;
             ++index) {
            if (!finiteContact(contacts.contacts[index])) {
                count =
                    0x80000000u | MR_STEP_NONFINITE_RESULT;
                break;
            }
            pairRawContacts[stagingBase + index] =
                contacts.contacts[index];
        }
        pairRawCounts[
            work.environment * dispatch.eligiblePairCount +
            work.compiledPair
        ] = uint2(count, work.reserved);
    }
}

kernel void mr_world_narrowphase_convex_queue(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRShapeGPU* shapes [[buffer(1)]],
    device const MRCompiledCollisionPairGPU* eligiblePairs [[buffer(2)]],
    device const MRProjectedColliderGPU* projectedColliders [[buffer(3)]],
    device const MRPairWorkGPU* queue [[buffer(4)]],
    device MRWorkQueueHeaderGPU* headers [[buffer(5)]],
    device const MRGeometryHeaderGPU* geometryHeaders [[buffer(6)]],
    device const float4* geometryVertices [[buffer(7)]],
    device MRRawContactGPU* pairRawContacts [[buffer(8)]],
    device uint2* pairRawCounts [[buffer(9)]],
    device const MRConvexQueryCacheGPU* previousCaches
        [[buffer(10)]],
    device MRConvexQueryCacheGPU* candidateCaches
        [[buffer(11)]],
    constant uint4& classConfig [[buffer(12)]],
    const uint workerIndex [[thread_position_in_grid]],
    const uint workerCount [[threads_per_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    const uint workClass = classConfig.x;
    const MRWorkQueueHeaderGPU header = headers[workClass];
    const uint stride = max(workerCount, 1u);
    device atomic_uint* workerCursor =
        reinterpret_cast<device atomic_uint*>(
            &headers[workClass].workerCursor
        );
    for (uint iteration = 0u;; ++iteration) {
        uint queueIndex = 0u;
        if (classConfig.z != 0u) {
            uint claimedBase = 0u;
            if (lane == 0u) {
                claimedBase = atomic_fetch_add_explicit(
                    workerCursor,
                    MR_WAVE32_CONTACTS_PER_TILE,
                    memory_order_relaxed
                );
            }
            claimedBase =
                simd_broadcast_first(claimedBase);
            if (claimedBase >= header.count) {
                break;
            }
            queueIndex = claimedBase + lane;
            if (queueIndex >= header.count) {
                continue;
            }
        } else {
            const ulong linearIndex =
                static_cast<ulong>(workerIndex) +
                static_cast<ulong>(iteration) * stride;
            if (linearIndex >= header.count) {
                break;
            }
            queueIndex = static_cast<uint>(linearIndex);
        }
        const MRPairWorkGPU work =
            queue[header.reserved0 + queueIndex];
        if (work.environment >= dispatch.environmentCount ||
            work.compiledPair >= dispatch.eligiblePairCount ||
            work.workClass != workClass ||
            (workClass != MR_WORLD_WORK_PRIMITIVE_GJK &&
             workClass != MR_WORLD_WORK_HARD_CONVEX)) {
            continue;
        }
        const MRCompiledCollisionPairGPU pair =
            eligiblePairs[work.compiledPair];
        const uint projectionBase =
            work.environment * dispatch.shapeCount;
        WorldShape shapeA;
        WorldShape shapeB;
        uint failureCode = MR_STEP_SUCCESS;
        if (!loadProjectedCollider(
                pair.colliderA,
                shapes[pair.colliderA],
                projectedColliders[
                    projectionBase + pair.colliderA
                ],
                shapeA,
                failureCode
            ) ||
            !loadProjectedCollider(
                pair.colliderB,
                shapes[pair.colliderB],
                projectedColliders[
                    projectionBase + pair.colliderB
                ],
                shapeB,
                failureCode
            )) {
            pairRawCounts[
                work.environment *
                    dispatch.eligiblePairCount +
                work.compiledPair
            ] = uint2(
                0x80000000u | failureCode,
                work.reserved
            );
            continue;
        }

        ConvexQueryResult query = {};
        query.status = MR_STEP_SUCCESS;
        const MRConvexQueryCacheGPU previousCache =
            previousCaches[work.cacheSlot];
        const float3 cachedDirection =
            previousCache.featureAndStatus.x ==
                    MR_STEP_SUCCESS
            ? previousCache.separatingAxisAndDistance.xyz
            : float3(0.0f);
        const ContactBatch contacts = supportMappedContacts(
            pair.colliderA,
            pair.colliderB,
            shapeA,
            shapeB,
            shapes[pair.colliderA],
            shapes[pair.colliderB],
            geometryHeaders,
            geometryVertices,
            cachedDirection,
            query
        );
        MRConvexQueryCacheGPU cache = {};
        cache.separatingAxisAndDistance =
            float4(query.normal, query.separation);
        cache.supportA = uint4(
            query.featureA,
            query.featureB,
            query.iterations,
            query.fallback
        );
        cache.featureAndStatus = uint4(
            query.status,
            work.workClass,
            work.stableKeyLow,
            work.stableKeyHigh
        );
        candidateCaches[work.cacheSlot] = cache;

        const uint countIndex =
            work.environment * dispatch.eligiblePairCount +
            work.compiledPair;
        if (query.status != MR_STEP_SUCCESS) {
            pairRawCounts[countIndex] =
                uint2(
                    0x80000000u | query.status,
                    work.reserved
                );
            continue;
        }
        const uint stagingBase =
            work.reserved *
            MR_METAL_WORLD_RAW_CONTACTS_PER_PAIR;
        uint count = contacts.count;
        for (uint index = 0u;
             index < contacts.count;
             ++index) {
            if (!finiteContact(contacts.contacts[index])) {
                count =
                    0x80000000u |
                    MR_STEP_NONFINITE_RESULT;
                break;
            }
            pairRawContacts[stagingBase + index] =
                contacts.contacts[index];
        }
        pairRawCounts[countIndex] = uint2(count, work.reserved);
    }
}

// Authored-hull queue with a bounded penetration call graph. Keeping this
// distinct from the general convex kernel materially reduces per-lane private
// storage and improves occupancy for batched humanoid self-collision.
kernel void mr_world_narrowphase_hull_queue(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRShapeGPU* shapes [[buffer(1)]],
    device const MRCompiledCollisionPairGPU* eligiblePairs [[buffer(2)]],
    device const MRProjectedColliderGPU* projectedColliders [[buffer(3)]],
    device const MRPairWorkGPU* queue [[buffer(4)]],
    device MRWorkQueueHeaderGPU* headers [[buffer(5)]],
    device const MRGeometryHeaderGPU* geometryHeaders [[buffer(6)]],
    device const float4* geometryVertices [[buffer(7)]],
    device MRRawContactGPU* pairRawContacts [[buffer(8)]],
    device uint2* pairRawCounts [[buffer(9)]],
    device const MRConvexQueryCacheGPU* previousCaches
        [[buffer(10)]],
    device MRConvexQueryCacheGPU* candidateCaches
        [[buffer(11)]],
    constant uint4& classConfig [[buffer(12)]],
    const uint workerIndex [[thread_position_in_grid]],
    const uint workerCount [[threads_per_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    const uint workClass = classConfig.x;
    const MRWorkQueueHeaderGPU header = headers[workClass];
    const uint stride = max(workerCount, 1u);
    device atomic_uint* workerCursor =
        reinterpret_cast<device atomic_uint*>(
            &headers[workClass].workerCursor
        );
    for (uint iteration = 0u;; ++iteration) {
        uint queueIndex = 0u;
        if (classConfig.z != 0u) {
            uint claimedBase = 0u;
            if (lane == 0u) {
                claimedBase = atomic_fetch_add_explicit(
                    workerCursor,
                    MR_WAVE32_CONTACTS_PER_TILE,
                    memory_order_relaxed
                );
            }
            claimedBase =
                simd_broadcast_first(claimedBase);
            if (claimedBase >= header.count) {
                break;
            }
            queueIndex = claimedBase + lane;
            if (queueIndex >= header.count) {
                continue;
            }
        } else {
            const ulong linearIndex =
                static_cast<ulong>(workerIndex) +
                static_cast<ulong>(iteration) * stride;
            if (linearIndex >= header.count) {
                break;
            }
            queueIndex = static_cast<uint>(linearIndex);
        }
        const MRPairWorkGPU work =
            queue[header.reserved0 + queueIndex];
        if (work.environment >= dispatch.environmentCount ||
            work.compiledPair >= dispatch.eligiblePairCount ||
            work.workClass != workClass ||
            workClass != MR_WORLD_WORK_HULL_GJK) {
            continue;
        }
        const MRCompiledCollisionPairGPU pair =
            eligiblePairs[work.compiledPair];
        const uint projectionBase =
            work.environment * dispatch.shapeCount;
        WorldShape shapeA;
        WorldShape shapeB;
        uint failureCode = MR_STEP_SUCCESS;
        if (!loadProjectedCollider(
                pair.colliderA,
                shapes[pair.colliderA],
                projectedColliders[
                    projectionBase + pair.colliderA
                ],
                shapeA,
                failureCode
            ) ||
            !loadProjectedCollider(
                pair.colliderB,
                shapes[pair.colliderB],
                projectedColliders[
                    projectionBase + pair.colliderB
                ],
                shapeB,
                failureCode
            )) {
            pairRawCounts[
                work.environment *
                    dispatch.eligiblePairCount +
                work.compiledPair
            ] = uint2(
                0x80000000u | failureCode,
                work.reserved
            );
            continue;
        }

        ConvexQueryResult query = {};
        query.status = MR_STEP_SUCCESS;
        const MRConvexQueryCacheGPU previousCache =
            previousCaches[work.cacheSlot];
        const float3 cachedDirection =
            previousCache.featureAndStatus.x ==
                    MR_STEP_SUCCESS
            ? previousCache.separatingAxisAndDistance.xyz
            : float3(0.0f);
        const ContactBatch contacts = supportMappedHullContacts(
            pair.colliderA,
            pair.colliderB,
            shapeA,
            shapeB,
            shapes[pair.colliderA],
            shapes[pair.colliderB],
            geometryHeaders,
            geometryVertices,
            cachedDirection,
            query
        );
        MRConvexQueryCacheGPU cache = {};
        cache.separatingAxisAndDistance =
            float4(query.normal, query.separation);
        cache.supportA = uint4(
            query.featureA,
            query.featureB,
            query.iterations,
            query.fallback
        );
        cache.featureAndStatus = uint4(
            query.status,
            work.workClass,
            work.stableKeyLow,
            work.stableKeyHigh
        );
        candidateCaches[work.cacheSlot] = cache;

        const uint countIndex =
            work.environment * dispatch.eligiblePairCount +
            work.compiledPair;
        if (query.status != MR_STEP_SUCCESS) {
            pairRawCounts[countIndex] =
                uint2(
                    0x80000000u | query.status,
                    work.reserved
                );
            continue;
        }
        const uint stagingBase =
            work.reserved *
            MR_METAL_WORLD_RAW_CONTACTS_PER_PAIR;
        uint count = contacts.count;
        for (uint index = 0u;
             index < contacts.count;
             ++index) {
            if (!finiteContact(contacts.contacts[index])) {
                count =
                    0x80000000u |
                    MR_STEP_NONFINITE_RESULT;
                break;
            }
            pairRawContacts[stagingBase + index] =
                contacts.contacts[index];
        }
        pairRawCounts[countIndex] = uint2(count, work.reserved);
    }
}

kernel void mr_world_narrowphase_mesh_queue(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRShapeGPU* shapes [[buffer(1)]],
    device const MRCompiledCollisionPairGPU* eligiblePairs [[buffer(2)]],
    device const MRProjectedColliderGPU* projectedColliders [[buffer(3)]],
    device const MRPairWorkGPU* queue [[buffer(4)]],
    device MRWorkQueueHeaderGPU* headers [[buffer(5)]],
    device const MRGeometryHeaderGPU* geometryHeaders [[buffer(6)]],
    device const float4* geometryVertices [[buffer(7)]],
    device const MRMeshBVHNodeGPU* meshNodes [[buffer(8)]],
    device const MRMeshTriangleGPU* meshTriangles [[buffer(9)]],
    device MRRawContactGPU* pairRawContacts [[buffer(10)]],
    device uint2* pairRawCounts [[buffer(11)]],
    device MRConvexQueryCacheGPU* caches [[buffer(12)]],
    constant uint4& classConfig [[buffer(13)]],
    const uint workerIndex [[thread_position_in_grid]],
    const uint workerCount [[threads_per_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    const uint workClass = classConfig.x;
    const MRWorkQueueHeaderGPU header = headers[workClass];
    const uint stride = max(workerCount, 1u);
    device atomic_uint* workerCursor =
        reinterpret_cast<device atomic_uint*>(
            &headers[workClass].workerCursor
        );
    for (uint iteration = 0u;; ++iteration) {
        uint queueIndex = 0u;
        if (classConfig.z != 0u) {
            uint claimedBase = 0u;
            if (lane == 0u) {
                claimedBase = atomic_fetch_add_explicit(
                    workerCursor,
                    MR_WAVE32_CONTACTS_PER_TILE,
                    memory_order_relaxed
                );
            }
            claimedBase =
                simd_broadcast_first(claimedBase);
            if (claimedBase >= header.count) {
                break;
            }
            queueIndex = claimedBase + lane;
            if (queueIndex >= header.count) {
                continue;
            }
        } else {
            const ulong linearIndex =
                static_cast<ulong>(workerIndex) +
                static_cast<ulong>(iteration) * stride;
            if (linearIndex >= header.count) {
                break;
            }
            queueIndex = static_cast<uint>(linearIndex);
        }
        const MRPairWorkGPU work =
            queue[header.reserved0 + queueIndex];
        if (work.environment >= dispatch.environmentCount ||
            work.compiledPair >= dispatch.eligiblePairCount ||
            work.workClass != workClass ||
            workClass != MR_WORLD_WORK_MESH) {
            continue;
        }
        const MRCompiledCollisionPairGPU pair =
            eligiblePairs[work.compiledPair];
        const uint projectionBase =
            work.environment * dispatch.shapeCount;
        WorldShape shapeA;
        WorldShape shapeB;
        uint failureCode = MR_STEP_SUCCESS;
        if (!loadProjectedCollider(
                pair.colliderA,
                shapes[pair.colliderA],
                projectedColliders[
                    projectionBase + pair.colliderA
                ],
                shapeA,
                failureCode
            ) ||
            !loadProjectedCollider(
                pair.colliderB,
                shapes[pair.colliderB],
                projectedColliders[
                    projectionBase + pair.colliderB
                ],
                shapeB,
                failureCode
            )) {
            pairRawCounts[
                work.environment *
                    dispatch.eligiblePairCount +
                work.compiledPair
            ] = uint2(
                0x80000000u | failureCode,
                work.reserved
            );
            continue;
        }
        uint triangleCandidates = 0u;
        const ContactBatch contacts = meshContacts(
            shapeA,
            shapeB,
            shapes[pair.colliderA],
            shapes[pair.colliderB],
            geometryHeaders,
            geometryVertices,
            meshNodes,
            meshTriangles,
            triangleCandidates
        );
        MRConvexQueryCacheGPU cache = {};
        cache.supportA = uint4(
            0u,
            0u,
            triangleCandidates,
            2u
        );
        cache.featureAndStatus = uint4(
            MR_STEP_SUCCESS,
            work.workClass,
            work.stableKeyLow,
            work.stableKeyHigh
        );
        caches[work.cacheSlot] = cache;
        const uint countIndex =
            work.environment * dispatch.eligiblePairCount +
            work.compiledPair;
        const uint stagingBase =
            work.reserved *
            MR_METAL_WORLD_RAW_CONTACTS_PER_PAIR;
        uint count = contacts.count;
        for (uint index = 0u;
             index < contacts.count;
             ++index) {
            if (!finiteContact(contacts.contacts[index])) {
                count =
                    0x80000000u |
                    MR_STEP_NONFINITE_RESULT;
                break;
            }
            pairRawContacts[stagingBase + index] =
                contacts.contacts[index];
        }
        pairRawCounts[countIndex] = uint2(count, work.reserved);
    }
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
    device const uint2* pairRawCounts [[buffer(26)]],
    device const MRRawContactGPU* pairRawContactStaging
        [[buffer(27)]],
    device const MRConvexQueryCacheGPU* convexCaches
        [[buffer(28)]],
    device const MRCCDPairGPU* ccdPairs [[buffer(29)]],
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
    status.requiredHardConvexPairs = 0u;
    status.requiredMeshCandidates = 0u;
    status.hardConvexPairs = 0u;
    status.meshCandidates = 0u;
    status.hardFallbacks = 0u;
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
    const uint ccdBase =
        environment * dispatch.ccdCandidateCapacity;
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
    dummyQuery.flags = MR_ARTICULATED_POINT_INACTIVE;

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
    uint requiredHardConvexPairs = 0u;
    uint requiredMeshCandidates = 0u;
    uint hardFallbacks = 0u;
    uint firstHardConvexOverflowPair = MR_INVALID_INDEX;
    uint firstMeshOverflowPair = MR_INVALID_INDEX;

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
        uint pairEventSlot = MR_CONSTRAINT_IR_INVALID_INDEX;
        if (dispatch.ccdMode == MR_WORLD_CCD_HYBRID) {
            const uint storedEvents = min(
                status.reservedEvent0,
                min(
                    dispatch.ccdCandidateCapacity,
                    dispatch.ccdEventCapacity
                )
            );
            for (uint eventSlot = 0u;
                 eventSlot < storedEvents;
                 ++eventSlot) {
                const MRCCDPairGPU event =
                    ccdPairs[ccdBase + eventSlot];
                if (event.compiledPair == eligibleIndex &&
                    (event.flags &
                     MR_CCD_PAIR_HAS_IMPACT) != 0u) {
                    pairEventSlot = eventSlot;
                    break;
                }
            }
        }

        const uint workClass =
            worldPairWorkClass(compiled, shapes);
        if (workClass == MR_WORLD_WORK_PRIMITIVE_GJK ||
            workClass == MR_WORLD_WORK_HULL_GJK ||
            workClass == MR_WORLD_WORK_HARD_CONVEX ||
            workClass == MR_WORLD_WORK_MESH) {
            const MRConvexQueryCacheGPU queryCache =
                convexCaches[
                    environment * dispatch.convexCacheStride +
                    compiled.convexCacheSlot
                ];
            const uint fallback = queryCache.supportA.w;
            if (workClass == MR_WORLD_WORK_MESH) {
                const uint previousMeshRequirement =
                    requiredMeshCandidates;
                const uint triangleCandidates =
                    queryCache.supportA.z;
                requiredMeshCandidates =
                    triangleCandidates >
                        0xffffffffu - requiredMeshCandidates
                    ? 0xffffffffu
                    : requiredMeshCandidates +
                        triangleCandidates;
                if (firstMeshOverflowPair ==
                        MR_INVALID_INDEX &&
                    (
                        previousMeshRequirement >
                            dispatch.meshCandidateCapacity ||
                        triangleCandidates >
                            dispatch.meshCandidateCapacity -
                                min(
                                    previousMeshRequirement,
                                    dispatch.meshCandidateCapacity
                                )
                    )) {
                    firstMeshOverflowPair = eligibleIndex;
                }
            } else if (fallback == 1u || fallback == 3u) {
                if (requiredHardConvexPairs != 0xffffffffu) {
                    ++requiredHardConvexPairs;
                }
                if (fallback == 1u &&
                    hardFallbacks != 0xffffffffu) {
                    ++hardFallbacks;
                }
                if (firstHardConvexOverflowPair ==
                        MR_INVALID_INDEX &&
                    requiredHardConvexPairs >
                        dispatch.hardConvexCapacity) {
                    firstHardConvexOverflowPair =
                        eligibleIndex;
                }
            }
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

        ContactBatch raw = {};
        const uint2 stagedRecord =
            pairRawCounts[
                environment * dispatch.eligiblePairCount +
                eligibleIndex
            ];
        const uint stagedCount = stagedRecord.x;
        if ((stagedCount & 0x80000000u) != 0u) {
            status.code = stagedCount & 0x7fffffffu;
            status.firstFailingPair = eligibleIndex;
            break;
        }
        if (stagedCount >
            MR_METAL_WORLD_RAW_CONTACTS_PER_PAIR) {
            status.code = MR_STEP_NONFINITE_RESULT;
            status.firstFailingPair = eligibleIndex;
            break;
        }
        raw.count = stagedCount;
        const uint stagedBase =
            stagedRecord.y * MR_METAL_WORLD_RAW_CONTACTS_PER_PAIR;
        for (uint stagedIndex = 0u;
             stagedIndex < stagedCount;
             ++stagedIndex) {
            raw.contacts[stagedIndex] =
                pairRawContactStaging[
                    stagedBase + stagedIndex
                ];
        }
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
            uint materialIndexA = sourceA.materialIndex;
            uint materialIndexB = sourceB.materialIndex;
            const uint materialOverride =
                manifoldPoint.featureAndLife[3];
            if ((materialOverride &
                 MR_RAW_CONTACT_MATERIAL_OVERRIDE) != 0u) {
                const uint cookedMaterial =
                    materialOverride &
                    MR_RAW_CONTACT_MATERIAL_INDEX_MASK;
                if (isSurfaceShapeType(sourceA.shapeType)) {
                    materialIndexA = cookedMaterial;
                } else if (isSurfaceShapeType(
                               sourceB.shapeType
                           )) {
                    materialIndexB = cookedMaterial;
                }
            }
            const MRMaterialGPU materialA =
                materials[materialIndexA];
            const MRMaterialGPU materialB =
                materials[materialIndexB];
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
                pairEventSlot !=
                        MR_CONSTRAINT_IR_INVALID_INDEX ||
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
            contact.tangent = float4(tangentWorld, 0.0f);
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
            block.eventSlot = pairEventSlot;
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
                // A positive-separation speculative contact only removes the
                // portion of closing velocity that would cross the surface
                // during this microstep. Resting/penetrating contacts retain
                // the ordinary zero normal target.
                row.targetVelocity =
                    localRow == 0u &&
                    effectiveSeparation > 0.0f
                    ? -effectiveSeparation /
                        dispatch.timestepAndBias.x
                    : 0.0f;
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
                queryA.flags = 0u;
                queryA.localPoint = manifoldPoint.localAnchorA;
            }
            if (bodyB.flagsAndIndices[1] ==
                dispatch.articulationIndex) {
                queryB.bodyIndex = sourceB.bodyIndex;
                queryB.flags = 0u;
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
    status.requiredHardConvexPairs =
        requiredHardConvexPairs;
    status.requiredMeshCandidates =
        requiredMeshCandidates;
    status.hardConvexPairs = min(
        requiredHardConvexPairs,
        dispatch.hardConvexCapacity
    );
    status.meshCandidates = min(
        requiredMeshCandidates,
        dispatch.meshCandidateCapacity
    );
    status.hardFallbacks = hardFallbacks;
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
        } else if (
            requiredHardConvexPairs >
                dispatch.hardConvexCapacity
        ) {
            status.code = MR_STEP_CONTACT_CAPACITY_OVERFLOW;
            status.firstFailingPair =
                firstHardConvexOverflowPair;
        } else if (
            requiredMeshCandidates >
                dispatch.meshCandidateCapacity
        ) {
            status.code = MR_STEP_CONTACT_CAPACITY_OVERFLOW;
            status.firstFailingPair =
                firstMeshOverflowPair;
        }
    }
    if (status.firstFailingPair != MR_INVALID_INDEX &&
        status.firstFailingStableKeyLow == 0u &&
        status.firstFailingStableKeyHigh == 0u) {
        status.firstFailingStableKeyLow =
            status.firstFailingPair;
        status.firstFailingStableKeyHigh = environment;
    }
    candidateManifoldCounts[environment] =
        status.code == MR_STEP_SUCCESS ? manifoldCount : 0u;
    statuses[environment] = status;
}

// Pair-parallel replacement for the manifold portion of
// mr_world_collide_compile. Every invocation owns one immutable compiled-pair
// slot, so scheduling order cannot affect cache matching or output order.
kernel void mr_world_finalize_pair_manifold(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRShapeGPU* shapes [[buffer(1)]],
    device const MRBodyStateGPU* bodies [[buffer(2)]],
    device const MRCompiledCollisionPairGPU* eligiblePairs [[buffer(3)]],
    device const uint* oldManifoldCounts [[buffer(4)]],
    device const MRManifoldHeaderGPU* oldManifoldHeaders [[buffer(5)]],
    device const MRManifoldPointGPU* oldManifoldPoints [[buffer(6)]],
    device const uint* pairOverlapFlags [[buffer(7)]],
    device const uint2* pairRawCounts [[buffer(8)]],
    device const MRRawContactGPU* pairRawContactStaging [[buffer(9)]],
    device const MRConvexQueryCacheGPU* convexCaches [[buffer(10)]],
    device const MRCCDPairGPU* ccdPairs [[buffer(11)]],
    device MRManifoldHeaderGPU* pairManifoldHeaders [[buffer(12)]],
    device MRManifoldPointGPU* pairManifoldPoints [[buffer(13)]],
    device MRManifoldIRScatterGPU* scatterRecords [[buffer(14)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(15)]],
    constant MRMetalWorldPassGPU& pass [[buffer(16)]],
    const uint flatPair [[thread_position_in_grid]]
) {
    const uint pairDomain =
        dispatch.environmentCount * dispatch.eligiblePairCount;
    if (flatPair >= pairDomain ||
        dispatch.eligiblePairCount == 0u) {
        return;
    }
    const uint environment =
        flatPair / dispatch.eligiblePairCount;
    const uint eligibleIndex =
        flatPair - environment * dispatch.eligiblePairCount;
    MRManifoldIRScatterGPU record = {};
    record.diagnostics0.y = MR_CONSTRAINT_IR_INVALID_INDEX;
    record.diagnostics1.w = MR_INVALID_INDEX;

    const MRMetalWorldContactStatusGPU incoming =
        statuses[environment];
    if (incoming.code != MR_STEP_SUCCESS) {
        record.diagnostics0.x = incoming.code;
        record.diagnostics1.w = eligibleIndex;
        scatterRecords[flatPair] = record;
        return;
    }
    if (!finiteContactDispatch(dispatch) ||
        pass.physicsSubstep >=
            MR_METAL_WORLD_MAX_PHYSICS_SUBSTEPS) {
        record.diagnostics0.x = MR_STEP_UNSUPPORTED;
        record.diagnostics1.w = eligibleIndex;
        scatterRecords[flatPair] = record;
        return;
    }

    const MRCompiledCollisionPairGPU compiled =
        eligiblePairs[eligibleIndex];
    if (compiled.colliderA >= dispatch.shapeCount ||
        compiled.colliderB >= dispatch.shapeCount ||
        compiled.colliderA >= compiled.colliderB ||
        compiled.pairClass == MR_COLLISION_PAIR_UNSUPPORTED) {
        record.diagnostics0.x = MR_STEP_UNSUPPORTED;
        record.diagnostics1.w = eligibleIndex;
        scatterRecords[flatPair] = record;
        return;
    }
    const uint overlapFlag = pairOverlapFlags[flatPair];
    if ((overlapFlag & 0x80000000u) != 0u) {
        record.diagnostics0.x =
            overlapFlag & 0x7fffffffu;
        record.diagnostics1.w = eligibleIndex;
        scatterRecords[flatPair] = record;
        return;
    }
    if (overlapFlag == 0u) {
        scatterRecords[flatPair] = record;
        return;
    }
    record.counts0.x = 1u;

    if (dispatch.ccdMode == MR_WORLD_CCD_HYBRID) {
        const uint storedEvents = min(
            incoming.reservedEvent0,
            min(
                dispatch.ccdCandidateCapacity,
                dispatch.ccdEventCapacity
            )
        );
        const uint ccdBase =
            environment * dispatch.ccdCandidateCapacity;
        for (uint eventSlot = 0u;
             eventSlot < storedEvents;
             ++eventSlot) {
            const MRCCDPairGPU event =
                ccdPairs[ccdBase + eventSlot];
            if (event.compiledPair == eligibleIndex &&
                (event.flags & MR_CCD_PAIR_HAS_IMPACT) != 0u) {
                record.diagnostics0.y = eventSlot;
                break;
            }
        }
    }

    const uint workClass =
        worldPairWorkClass(compiled, shapes);
    if (workClass == MR_WORLD_WORK_PRIMITIVE_GJK ||
        workClass == MR_WORLD_WORK_HULL_GJK ||
        workClass == MR_WORLD_WORK_HARD_CONVEX ||
        workClass == MR_WORLD_WORK_MESH) {
        const MRConvexQueryCacheGPU cache =
            convexCaches[
                environment * dispatch.convexCacheStride +
                compiled.convexCacheSlot
            ];
        if (workClass == MR_WORLD_WORK_MESH) {
            record.diagnostics1.y = cache.supportA.z;
        } else if (
            cache.supportA.w == 1u ||
            cache.supportA.w == 3u
        ) {
            record.diagnostics1.x = 1u;
            record.diagnostics1.z =
                cache.supportA.w == 1u ? 1u : 0u;
        }
    }

    const uint2 stagedRecord = pairRawCounts[flatPair];
    const uint stagedCount = stagedRecord.x;
    if ((stagedCount & 0x80000000u) != 0u) {
        record.diagnostics0.x =
            stagedCount & 0x7fffffffu;
        record.diagnostics1.w = eligibleIndex;
        scatterRecords[flatPair] = record;
        return;
    }
    if (stagedCount >
        MR_METAL_WORLD_RAW_CONTACTS_PER_PAIR) {
        record.diagnostics0.x = MR_STEP_NONFINITE_RESULT;
        record.diagnostics1.w = eligibleIndex;
        scatterRecords[flatPair] = record;
        return;
    }
    record.counts0.y = stagedCount;
    record.reserved.x = stagedRecord.y;
    if (stagedCount == 0u) {
        scatterRecords[flatPair] = record;
        return;
    }

    ContactBatch raw = {};
    raw.count = stagedCount;
    const uint stagedBase =
        stagedRecord.y * MR_METAL_WORLD_RAW_CONTACTS_PER_PAIR;
    const MRShapeGPU sourceA = shapes[compiled.colliderA];
    const MRShapeGPU sourceB = shapes[compiled.colliderB];
    const float restSeparation =
        sourceA.contactRestAndBoundingRadius.y +
        sourceB.contactRestAndBoundingRadius.y;
    float maximumPenetration = 0.0f;
    for (uint rawIndex = 0u;
         rawIndex < stagedCount;
         ++rawIndex) {
        const MRRawContactGPU witness =
            pairRawContactStaging[stagedBase + rawIndex];
        if (!finiteContact(witness)) {
            record.diagnostics0.x =
                MR_STEP_NONFINITE_RESULT;
            record.diagnostics1.w = eligibleIndex;
            scatterRecords[flatPair] = record;
            return;
        }
        raw.contacts[rawIndex] = witness;
        maximumPenetration = max(
            maximumPenetration,
            max(
                restSeparation -
                    witness.normalAndSeparation.w,
                0.0f
            )
        );
    }

    const uint manifoldBase =
        environment * dispatch.manifoldStride;
    const uint oldCount = min(
        oldManifoldCounts[environment],
        dispatch.manifoldCapacity
    );
    uint oldIndex = 0u;
    bool hasOld = false;
    for (; oldIndex < oldCount; ++oldIndex) {
        const MRManifoldHeaderGPU oldHeader =
            oldManifoldHeaders[manifoldBase + oldIndex];
        if (oldHeader.pairAndCount[1] ==
                compiled.colliderA &&
            oldHeader.pairAndCount[2] ==
                compiled.colliderB) {
            hasOld = true;
            break;
        }
        if (oldHeader.pairAndCount[1] >
                compiled.colliderA ||
            (
                oldHeader.pairAndCount[1] ==
                    compiled.colliderA &&
                oldHeader.pairAndCount[2] >
                    compiled.colliderB
            )) {
            break;
        }
    }

    MRManifoldHeaderGPU manifoldHeader = {};
    MRManifoldPointGPU manifoldPoints[
        MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY
    ];
    uint retainedPoints = 0u;
    uint newPoints = 0u;
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    if (!buildPersistentWorldManifold(
            environment,
            compiled,
            raw,
            shapes,
            bodies + bodyBase,
            hasOld,
            oldManifoldHeaders + manifoldBase + oldIndex,
            oldManifoldPoints +
                (
                    manifoldBase + oldIndex
                ) * MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY,
            dispatch.manifoldThresholds,
            manifoldHeader,
            manifoldPoints,
            retainedPoints,
            newPoints
        )) {
        record.diagnostics0.x = MR_STEP_NONFINITE_RESULT;
        record.diagnostics1.w = eligibleIndex;
        scatterRecords[flatPair] = record;
        return;
    }

    const uint stagingSlot = stagedRecord.y;
    pairManifoldHeaders[stagingSlot] = manifoldHeader;
    const uint pairPointBase =
        stagingSlot * MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
    for (uint point = 0u;
         point < MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
         ++point) {
        pairManifoldPoints[pairPointBase + point] =
            manifoldPoints[point];
    }
    const uint pointCount = manifoldHeader.pairAndCount[3];
    record.counts0.z = 1u;
    record.counts0.w = pointCount;
    record.counts1.x =
        pointCount > 0x55555555u
        ? 0xffffffffu
        : 3u * pointCount;
    record.counts1.y =
        pointCount > 0x7fffffffu
        ? 0xffffffffu
        : 2u * pointCount;
    record.counts1.z = record.counts1.y;
    record.counts1.w = pointCount;
    record.diagnostics0.z = retainedPoints;
    record.diagnostics0.w = newPoints;
    record.metrics.x = maximumPenetration;
    scatterRecords[flatPair] = record;
}

// Exactly one SIMD32 group owns one environment. It scans arbitrary configured
// pair counts in 32-pair tiles and writes stable exclusive prefixes. The only
// serial state is one lane's tile accumulator; all pair work remains lane
// saturated and no count becomes CPU-visible.
kernel void mr_world_scan_manifold_ir(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device MRManifoldIRScatterGPU* scatterRecords [[buffer(1)]],
    device uint* candidateManifoldCounts [[buffer(2)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(3)]],
    constant MRMetalWorldPassGPU& pass [[buffer(4)]],
    const uint environment [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    if (environment >= dispatch.environmentCount ||
        lane >= MR_SIMD_WIDTH) {
        return;
    }
    // Event-time graph passes use non-success transient status codes to mark
    // inactive environments. The legacy compiler returned before touching
    // counts in this case; preserving that contract prevents a later
    // successful closeout from erasing the last solved impact evidence.
    if (statuses[environment].code != MR_STEP_SUCCESS) {
        return;
    }
    // The immutable mechanism program owns the canonical prefix. Dynamic
    // rigid manifolds and rod witnesses append after it, so every later stage
    // sees one ConstraintIR stream and stable offsets irrespective of how
    // collision work was scheduled.
    uint4 totals0 = uint4(
        0u,
        0u,
        0u,
        dispatch.authoredConstraintCount
    );
    uint4 totals1 = uint4(
        3u * dispatch.authoredConstraintCount,
        2u * dispatch.authoredConstraintCount,
        2u * dispatch.authoredConstraintCount,
        0u
    );
    uint retained = 0u;
    uint fresh = 0u;
    uint hard = 0u;
    uint mesh = 0u;
    uint fallbacks = 0u;
    float penetration = 0.0f;
    uint firstError = MR_INVALID_INDEX;
    uint firstErrorCode = MR_STEP_SUCCESS;
    uint firstPairOverflow = MR_INVALID_INDEX;
    uint firstRawOverflow = MR_INVALID_INDEX;
    uint firstManifoldOverflow = MR_INVALID_INDEX;
    uint firstConstraintOverflow = MR_INVALID_INDEX;
    uint firstHardOverflow = MR_INVALID_INDEX;
    uint firstMeshOverflow = MR_INVALID_INDEX;

    const uint pairBase =
        environment * dispatch.eligiblePairCount;
    for (uint tile = 0u;
         tile < dispatch.eligiblePairCount;
         tile += MR_SIMD_WIDTH) {
        const uint pair = tile + lane;
        const bool valid = pair < dispatch.eligiblePairCount;
        MRManifoldIRScatterGPU record = {};
        if (valid) {
            record = scatterRecords[pairBase + pair];
        }
        const uint4 count0 = valid ? record.counts0 : uint4(0u);
        const uint4 count1 = valid ? record.counts1 : uint4(0u);
        const uint4 prefix0 = uint4(
            simd_prefix_exclusive_sum(count0.x),
            simd_prefix_exclusive_sum(count0.y),
            simd_prefix_exclusive_sum(count0.z),
            simd_prefix_exclusive_sum(count0.w)
        );
        const uint4 prefix1 = uint4(
            simd_prefix_exclusive_sum(count1.x),
            simd_prefix_exclusive_sum(count1.y),
            simd_prefix_exclusive_sum(count1.z),
            simd_prefix_exclusive_sum(count1.w)
        );
        if (valid) {
            record.offsets0 = totals0 + prefix0;
            record.offsets1 = totals1 + prefix1;
            scatterRecords[pairBase + pair] = record;
        }
        const uint tilePairs = simd_sum(count0.x);
        const uint tileRaw = simd_sum(count0.y);
        const uint tileManifolds = simd_sum(count0.z);
        const uint tileConstraints = simd_sum(count0.w);
        const uint tileRows = simd_sum(count1.x);
        const uint tileEndpoints = simd_sum(count1.y);
        const uint tileQueries = simd_sum(count1.z);
        const uint tileEvidence = simd_sum(count1.w);

        const uint errorPair =
            valid &&
            record.diagnostics0.x != MR_STEP_SUCCESS
            ? pair
            : MR_INVALID_INDEX;
        const uint tileFirstError = simd_min(errorPair);
        const uint pairOverflow =
            valid &&
            record.offsets0.x + count0.x >
                dispatch.pairCapacity
            ? pair
            : MR_INVALID_INDEX;
        const uint rawOverflow =
            valid &&
            record.offsets0.y + count0.y >
                dispatch.rawContactCapacity
            ? pair
            : MR_INVALID_INDEX;
        const uint manifoldOverflow =
            valid &&
            record.offsets0.z + count0.z >
                dispatch.manifoldCapacity
            ? pair
            : MR_INVALID_INDEX;
        const uint constraintOverflow =
            valid &&
            (
                record.offsets0.w + count0.w >
                    dispatch.constraintCapacity ||
                record.offsets1.x + count1.x >
                    dispatch.rowCapacity
            )
            ? pair
            : MR_INVALID_INDEX;
        const uint hardOverflow =
            valid &&
            record.offsets0.x <= dispatch.pairCapacity &&
            record.diagnostics1.x != 0u
            ? pair
            : MR_INVALID_INDEX;
        const uint meshOverflow =
            valid &&
            record.diagnostics1.y != 0u
            ? pair
            : MR_INVALID_INDEX;
        const uint tilePairOverflow = simd_min(pairOverflow);
        const uint tileRawOverflow = simd_min(rawOverflow);
        const uint tileManifoldOverflow =
            simd_min(manifoldOverflow);
        const uint tileConstraintOverflow =
            simd_min(constraintOverflow);
        const uint tileHardOverflow = simd_min(hardOverflow);
        const uint tileMeshOverflow = simd_min(meshOverflow);
        const uint tileRetained = simd_sum(
            valid ? record.diagnostics0.z : 0u
        );
        const uint tileFresh = simd_sum(
            valid ? record.diagnostics0.w : 0u
        );
        const uint tileHard = simd_sum(
            valid ? record.diagnostics1.x : 0u
        );
        const uint tileMesh = simd_sum(
            valid ? record.diagnostics1.y : 0u
        );
        const uint tileFallbacks = simd_sum(
            valid ? record.diagnostics1.z : 0u
        );
        const float tilePenetration = simd_max(
            valid ? record.metrics.x : 0.0f
        );
        if (lane == 0u) {
            if (firstError == MR_INVALID_INDEX &&
                tileFirstError != MR_INVALID_INDEX) {
                firstError = tileFirstError;
                firstErrorCode =
                    scatterRecords[
                        pairBase + tileFirstError
                    ].diagnostics0.x;
            }
            firstPairOverflow =
                min(firstPairOverflow, tilePairOverflow);
            firstRawOverflow =
                min(firstRawOverflow, tileRawOverflow);
            firstManifoldOverflow = min(
                firstManifoldOverflow,
                tileManifoldOverflow
            );
            firstConstraintOverflow = min(
                firstConstraintOverflow,
                tileConstraintOverflow
            );
            firstHardOverflow =
                min(firstHardOverflow, tileHardOverflow);
            firstMeshOverflow =
                min(firstMeshOverflow, tileMeshOverflow);
            totals0 += uint4(
                tilePairs,
                tileRaw,
                tileManifolds,
                tileConstraints
            );
            totals1 += uint4(
                tileRows,
                tileEndpoints,
                tileQueries,
                tileEvidence
            );
            retained += tileRetained;
            fresh += tileFresh;
            hard += tileHard;
            mesh += tileMesh;
            fallbacks += tileFallbacks;
            penetration = max(
                penetration,
                tilePenetration
            );
        }
        totals0 = simd_broadcast(totals0, 0u);
        totals1 = simd_broadcast(totals1, 0u);
    }

    if (lane != 0u) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    status.environment = environment;
    status.controlStep = pass.controlStep;
    status.physicsSubstep = pass.physicsSubstep;
    status.requiredPairs = totals0.x;
    status.requiredRawContacts = totals0.y;
    status.requiredManifolds = totals0.z;
    status.requiredConstraints = totals0.w;
    status.requiredRows = totals1.x;
    status.requiredHardConvexPairs = hard;
    status.requiredMeshCandidates = mesh;
    status.hardConvexPairs = min(
        hard,
        dispatch.hardConvexCapacity
    );
    status.meshCandidates = min(
        mesh,
        dispatch.meshCandidateCapacity
    );
    status.hardFallbacks = fallbacks;
    status.activePairs = totals0.x;
    status.activeContacts = totals0.w;
    status.retainedPoints = retained;
    status.newPoints = fresh;
    status.diagnostics.x =
        retained + fresh == 0u
        ? 1.0f
        : float(retained) / float(retained + fresh);
    status.diagnostics.y = penetration;
    status.firstFailingPair = MR_INVALID_INDEX;
    status.firstFailingConstraint = MR_INVALID_INDEX;
    if (status.code == MR_STEP_SUCCESS) {
        if (firstError != MR_INVALID_INDEX) {
            status.code = firstErrorCode;
            status.firstFailingPair = firstError;
        } else if (totals0.x > dispatch.pairCapacity) {
            status.code = MR_STEP_PAIR_CAPACITY_OVERFLOW;
            status.firstFailingPair = firstPairOverflow;
        } else if (
            totals0.y > dispatch.rawContactCapacity
        ) {
            status.code = MR_STEP_CONTACT_CAPACITY_OVERFLOW;
            status.firstFailingPair = firstRawOverflow;
        } else if (
            totals0.z > dispatch.manifoldCapacity
        ) {
            status.code = MR_STEP_MANIFOLD_CAPACITY_OVERFLOW;
            status.firstFailingPair =
                firstManifoldOverflow;
        } else if (
            totals0.w > dispatch.constraintCapacity ||
            totals1.x > dispatch.rowCapacity
        ) {
            status.code =
                MR_STEP_CONSTRAINT_CAPACITY_OVERFLOW;
            status.firstFailingConstraint =
                firstConstraintOverflow;
        } else if (hard > dispatch.hardConvexCapacity) {
            status.code = MR_STEP_CONTACT_CAPACITY_OVERFLOW;
            status.firstFailingPair = firstHardOverflow;
        } else if (
            mesh > dispatch.meshCandidateCapacity
        ) {
            status.code = MR_STEP_CONTACT_CAPACITY_OVERFLOW;
            status.firstFailingPair = firstMeshOverflow;
        }
    }
    if (status.firstFailingPair != MR_INVALID_INDEX &&
        status.firstFailingStableKeyLow == 0u &&
        status.firstFailingStableKeyHigh == 0u) {
        status.firstFailingStableKeyLow =
            status.firstFailingPair;
        status.firstFailingStableKeyHigh = environment;
    }
    candidateManifoldCounts[environment] =
        status.code == MR_STEP_SUCCESS ? totals0.z : 0u;
    statuses[environment] = status;
}

kernel void mr_world_scatter_manifold_records(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRCompiledCollisionPairGPU* eligiblePairs [[buffer(1)]],
    device const MRRawContactGPU* pairRawContactStaging [[buffer(2)]],
    device const MRManifoldHeaderGPU* pairManifoldHeaders [[buffer(3)]],
    device const MRManifoldPointGPU* pairManifoldPoints [[buffer(4)]],
    device const MRManifoldIRScatterGPU* scatterRecords [[buffer(5)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(6)]],
    device MRCandidatePairGPU* outputPairs [[buffer(7)]],
    device MRRawContactGPU* outputRawContacts [[buffer(8)]],
    device uint* outputRawPairIndices [[buffer(9)]],
    device MRManifoldHeaderGPU* candidateManifoldHeaders [[buffer(10)]],
    device MRManifoldPointGPU* candidateManifoldPoints [[buffer(11)]],
    const uint flatPair [[thread_position_in_grid]]
) {
    const uint pairDomain =
        dispatch.environmentCount * dispatch.eligiblePairCount;
    if (flatPair >= pairDomain ||
        dispatch.eligiblePairCount == 0u) {
        return;
    }
    const uint environment =
        flatPair / dispatch.eligiblePairCount;
    const uint eligibleIndex =
        flatPair - environment * dispatch.eligiblePairCount;
    if (statuses[environment].code != MR_STEP_SUCCESS) {
        return;
    }
    const MRManifoldIRScatterGPU record =
        scatterRecords[flatPair];
    const MRCompiledCollisionPairGPU compiled =
        eligiblePairs[eligibleIndex];
    if (record.counts0.x != 0u) {
        MRCandidatePairGPU pair = {};
        pair.environment = environment;
        pair.colliderA = compiled.colliderA;
        pair.colliderB = compiled.colliderB;
        pair.flags = compiled.pairClass;
        outputPairs[
            environment * dispatch.pairStride +
            record.offsets0.x
        ] = pair;
    }
    const uint stagedBase =
        record.reserved.x * MR_METAL_WORLD_RAW_CONTACTS_PER_PAIR;
    const uint rawBase =
        environment * dispatch.rawContactStride;
    // Finalization bounds this count and the segmented scan certifies its
    // output span. The narrowphase source count is not canonical here.
    for (uint rawIndex = 0u;
         rawIndex < record.counts0.y;
         ++rawIndex) {
        outputRawContacts[
            rawBase + record.offsets0.y + rawIndex
        ] = pairRawContactStaging[stagedBase + rawIndex];
        outputRawPairIndices[
            rawBase + record.offsets0.y + rawIndex
        ] = record.offsets0.x;
    }
    if (record.counts0.z == 0u) {
        return;
    }
    const uint manifoldOutput =
        environment * dispatch.manifoldStride +
        record.offsets0.z;
    const uint stagingSlot = record.reserved.x;
    candidateManifoldHeaders[manifoldOutput] =
        pairManifoldHeaders[stagingSlot];
    const uint sourcePointBase =
        stagingSlot * MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
    const uint outputPointBase =
        manifoldOutput * MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
    for (uint point = 0u;
         point < MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
         ++point) {
        candidateManifoldPoints[outputPointBase + point] =
            pairManifoldPoints[sourcePointBase + point];
    }
}

inline MRConstraintEndpointRuntimeGPU
worldEndpointRuntime(
    const MRBodyStateGPU body,
    const uint bodyIndex,
    const uint dynamicNode,
    const uint queryIndex,
    const float4 localAnchor
) {
    MRConstraintEndpointRuntimeGPU runtime = {};
    runtime.dynamicNode = MR_CONSTRAINT_IR_INVALID_INDEX;
    runtime.ownerIndex = MR_CONSTRAINT_IR_INVALID_INDEX;
    runtime.elementIndex = bodyIndex;
    runtime.queryIndex = MR_CONSTRAINT_IR_INVALID_INDEX;
    runtime.secondaryIndex = MR_CONSTRAINT_IR_INVALID_INDEX;
    runtime.twistIndex = MR_CONSTRAINT_IR_INVALID_INDEX;
    runtime.localAnchorOrRadial = localAnchor;
    const uint motion = body.flagsAndIndices[0];
    const uint articulation = body.flagsAndIndices[1];
    if (articulation != MR_INVALID_INDEX) {
        runtime.dynamicNode = dynamicNode;
        runtime.ownerKind =
            MR_CONSTRAINT_IR_OWNER_ARTICULATION;
        runtime.ownerIndex = articulation;
        runtime.queryIndex = queryIndex;
        runtime.flags =
            MR_CONSTRAINT_IR_RUNTIME_DYNAMIC |
            MR_CONSTRAINT_IR_RUNTIME_HAS_POINT_QUERY;
    } else {
        runtime.ownerKind =
            motion == MR_MOTION_STATIC
            ? MR_CONSTRAINT_IR_OWNER_WORLD
            : MR_CONSTRAINT_IR_OWNER_FREE_BODY;
        runtime.ownerIndex =
            motion == MR_MOTION_STATIC
            ? MR_CONSTRAINT_IR_INVALID_INDEX
            : bodyIndex;
        runtime.dynamicNode = dynamicNode;
        runtime.flags =
            motion == MR_MOTION_DYNAMIC
            ? MR_CONSTRAINT_IR_RUNTIME_DYNAMIC
            : (
                motion == MR_MOTION_KINEMATIC
                ? MR_CONSTRAINT_IR_RUNTIME_KINEMATIC
                : 0u
            );
    }
    return runtime;
}

kernel void mr_world_scatter_manifold_ir(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRShapeGPU* shapes [[buffer(1)]],
    device const MRMaterialGPU* materials [[buffer(2)]],
    device const MRBodyStateGPU* bodies [[buffer(3)]],
    device const MRArticulationGPU* articulations [[buffer(4)]],
    device const MRCompiledCollisionPairGPU* eligiblePairs [[buffer(5)]],
    device const MRManifoldHeaderGPU* pairManifoldHeaders [[buffer(6)]],
    device const MRManifoldPointGPU* pairManifoldPoints [[buffer(7)]],
    device const MRManifoldIRScatterGPU* scatterRecords [[buffer(8)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(9)]],
    device MRContactConstraintGPU* contacts [[buffer(10)]],
    device MRContactPointMetaGPU* contactMetadata [[buffer(11)]],
    device MRConstraintIRBlockGPU* blocks [[buffer(12)]],
    device MRConstraintIREndpointGPU* endpoints [[buffer(13)]],
    device MRConstraintEndpointRuntimeGPU* endpointRuntime [[buffer(14)]],
    device MRConstraintIRRowGPU* rows [[buffer(15)]],
    device MRConstraintIRConeGPU* cones [[buffer(16)]],
    device MRArticulatedPointImpulseGPU* pointQueries [[buffer(17)]],
    device const uint* bodyDynamicNodes [[buffer(18)]],
    device const float4* bodyParameters [[buffer(19)]],
    const uint flatPoint [[thread_position_in_grid]]
) {
    const uint pointsPerEnvironment =
        dispatch.eligiblePairCount *
        MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
    const uint pointDomain =
        dispatch.environmentCount * pointsPerEnvironment;
    if (flatPoint >= pointDomain ||
        pointsPerEnvironment == 0u) {
        return;
    }
    const uint environment =
        flatPoint / pointsPerEnvironment;
    if (statuses[environment].code != MR_STEP_SUCCESS) {
        return;
    }
    const uint localFlat =
        flatPoint - environment * pointsPerEnvironment;
    const uint eligibleIndex =
        localFlat / MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
    const uint pointIndex =
        localFlat %
        MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
    const uint flatPair =
        environment * dispatch.eligiblePairCount +
        eligibleIndex;
    const MRManifoldIRScatterGPU record =
        scatterRecords[flatPair];
    if (record.counts0.z == 0u) {
        return;
    }
    const MRManifoldHeaderGPU manifoldHeader =
        pairManifoldHeaders[record.reserved.x];
    if (pointIndex >= manifoldHeader.pairAndCount[3]) {
        return;
    }
    const MRManifoldPointGPU manifoldPoint =
        pairManifoldPoints[
            record.reserved.x *
                MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY +
            pointIndex
        ];
    const MRCompiledCollisionPairGPU compiled =
        eligiblePairs[eligibleIndex];
    const MRShapeGPU sourceA = shapes[compiled.colliderA];
    const MRShapeGPU sourceB = shapes[compiled.colliderB];
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    device const MRBodyStateGPU& bodyA =
        bodies[bodyBase + sourceA.bodyIndex];
    device const MRBodyStateGPU& bodyB =
        bodies[bodyBase + sourceB.bodyIndex];
    float4 rotationA;
    float4 rotationB;
    if (!checkedQuaternion(bodyA.orientation, rotationA) ||
        !checkedQuaternion(bodyB.orientation, rotationB)) {
        return;
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

    uint materialIndexA = sourceA.materialIndex;
    uint materialIndexB = sourceB.materialIndex;
    const uint materialOverride =
        manifoldPoint.featureAndLife[3];
    if ((materialOverride &
         MR_RAW_CONTACT_MATERIAL_OVERRIDE) != 0u) {
        const uint cookedMaterial =
            materialOverride &
            MR_RAW_CONTACT_MATERIAL_INDEX_MASK;
        if (isSurfaceShapeType(sourceA.shapeType)) {
            materialIndexA = cookedMaterial;
        } else if (isSurfaceShapeType(sourceB.shapeType)) {
            materialIndexB = cookedMaterial;
        }
    }
    const MRMaterialGPU materialA =
        materials[materialIndexA];
    const MRMaterialGPU materialB =
        materials[materialIndexB];
    float frictionScaleA = 1.0f;
    float frictionScaleB = 1.0f;
    float restitutionScaleA = 1.0f;
    float restitutionScaleB = 1.0f;
    if ((dispatch.flags &
         MR_METAL_WORLD_CONTACT_BODY_PARAMETERS) != 0u) {
        const uint parameterBase =
            environment * dispatch.bodyCount;
        const float4 parametersA =
            bodyParameters[parameterBase + sourceA.bodyIndex];
        const float4 parametersB =
            bodyParameters[parameterBase + sourceB.bodyIndex];
        frictionScaleA = max(parametersA.y, 0.0f);
        frictionScaleB = max(parametersB.y, 0.0f);
        restitutionScaleA = max(parametersA.z, 0.0f);
        restitutionScaleB = max(parametersB.z, 0.0f);
    }
    const float frictionScale =
        sqrt(frictionScaleA * frictionScaleB);
    const float restitutionScale =
        max(restitutionScaleA, restitutionScaleB);
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
    const float effectiveSeparation =
        geometricSeparation -
        sourceA.contactRestAndBoundingRadius.y -
        sourceB.contactRestAndBoundingRadius.y;

    const uint currentConstraint =
        record.offsets0.w + pointIndex;
    const uint outputConstraint =
        environment * dispatch.constraintStride +
        currentConstraint;
    MRContactConstraintGPU contact = {};
    contact.bodyA = sourceA.bodyIndex;
    contact.bodyB = sourceB.bodyIndex;
    contact.flags =
        record.diagnostics0.y !=
                MR_CONSTRAINT_IR_INVALID_INDEX ||
            manifoldPoint.featureAndLife[2] == 0u
        ? MR_CONSTRAINT_FLAG_NEW_IMPACT
        : MR_CONSTRAINT_FLAG_WARM_STARTED;
    contact.islandIndex = MR_INVALID_INDEX;
    contact.pairKey = collisionPairKey(
        compiled.colliderA,
        compiled.colliderB
    );
    contact.featureKey = collisionFeatureKey(manifoldPoint);
    contact.pointAndSeparation =
        float4(pointWorld, effectiveSeparation);
    contact.normal = float4(normalWorld, 0.0f);
    contact.tangent = float4(tangentWorld, 0.0f);
    contact.friction = float4(
        geometricMean(
            materialA.friction.x,
            materialB.friction.x
        ) * frictionScale,
        geometricMean(
            materialA.friction.y,
            materialB.friction.y
        ) * frictionScale,
        geometricMean(
            materialA.friction.z,
            materialB.friction.z
        ) * frictionScale,
        geometricMean(
            materialA.friction.w,
            materialB.friction.w
        ) * frictionScale
    );
    contact.response = float4(
        clamp(
            max(materialA.response.x, materialB.response.x) *
                restitutionScale,
            0.0f,
            1.0f
        ),
        max(materialA.response.y, materialB.response.y),
        materialA.response.z + materialB.response.z,
        0.0f
    );
    contact.impulses =
        (dispatch.flags &
         MR_METAL_WORLD_CONTACT_WARM_START) != 0u
        ? manifoldPoint.impulses
        : float4(0.0f);
    contacts[outputConstraint] = contact;

    MRContactPointMetaGPU metadata = {};
    metadata.colliderA = compiled.colliderA;
    metadata.colliderB = compiled.colliderB;
    metadata.manifoldIndex = record.offsets0.z;
    metadata.pointIndex = pointIndex;
    metadata.localAnchorA = manifoldPoint.localAnchorA;
    metadata.localAnchorB = manifoldPoint.localAnchorB;
    contactMetadata[outputConstraint] = metadata;

    MRConstraintIRBlockGPU block = {};
    block.key.words[0] = compiled.colliderA;
    block.key.words[1] = compiled.colliderB;
    block.key.words[2] = manifoldPoint.featureAndLife[0];
    block.key.words[3] = manifoldPoint.featureAndLife[1];
    block.type = MR_CONSTRAINT_CONTACT;
    block.dimension = 3u;
    block.flags = contact.flags;
    block.islandIndex = MR_INVALID_INDEX;
    block.endpointOffset = 2u * currentConstraint;
    block.endpointCount = 2u;
    block.rowOffset = 3u * currentConstraint;
    block.impulseOffset = 3u * currentConstraint;
    block.coneIndex = currentConstraint;
    block.eventSlot = record.diagnostics0.y;
    blocks[outputConstraint] = block;

    const uint endpointBase =
        2u * environment * dispatch.constraintStride;
    const uint endpointAIndex =
        endpointBase + 2u * currentConstraint;
    const uint endpointBIndex = endpointAIndex + 1u;
    MRConstraintIREndpointGPU endpointA = {};
    endpointA.objectIndex = sourceA.bodyIndex;
    endpointA.articulationIndex = bodyA.flagsAndIndices[1];
    endpointA.linkIndex = bodyA.flagsAndIndices[2];
    endpointA.role = MR_CONSTRAINT_IR_ENDPOINT_A;
    endpointA.jacobianKind =
        MR_CONSTRAINT_IR_JACOBIAN_BODY_LOCAL_POINT;
    endpointA.anchor = manifoldPoint.localAnchorA;
    MRConstraintIREndpointGPU endpointB = endpointA;
    endpointB.objectIndex = sourceB.bodyIndex;
    endpointB.articulationIndex = bodyB.flagsAndIndices[1];
    endpointB.linkIndex = bodyB.flagsAndIndices[2];
    endpointB.role = MR_CONSTRAINT_IR_ENDPOINT_B;
    endpointB.anchor = manifoldPoint.localAnchorB;
    endpoints[endpointAIndex] = endpointA;
    endpoints[endpointBIndex] = endpointB;

    const uint localQueryA = 2u * currentConstraint;
    const uint localQueryB = localQueryA + 1u;
    const uint queryBase =
        environment * dispatch.pointQueryStride;
    const uint queryAIndex = queryBase + localQueryA;
    const uint queryBIndex = queryBase + localQueryB;
    endpointRuntime[endpointAIndex] = worldEndpointRuntime(
        bodyA,
        sourceA.bodyIndex,
        bodyDynamicNodes[sourceA.bodyIndex],
        queryAIndex,
        manifoldPoint.localAnchorA
    );
    endpointRuntime[endpointBIndex] = worldEndpointRuntime(
        bodyB,
        sourceB.bodyIndex,
        bodyDynamicNodes[sourceB.bodyIndex],
        queryBIndex,
        manifoldPoint.localAnchorB
    );

    const uint rowBase =
        environment * dispatch.rowStride +
        3u * currentConstraint;
    const float3 directions[3] = {
        normalWorld,
        tangentWorld,
        bitangentWorld,
    };
    for (uint localRow = 0u;
         localRow < 3u;
         ++localRow) {
        MRConstraintIRRowGPU row = {};
        row.direction = float4(directions[localRow], 0.0f);
        row.positionError =
            localRow == 0u ? effectiveSeparation : 0.0f;
        row.targetVelocity =
            localRow == 0u &&
            effectiveSeparation > 0.0f
            ? -effectiveSeparation /
                dispatch.timestepAndBias.x
            : 0.0f;
        row.compliance =
            localRow == 0u
            ? materialA.response.z + materialB.response.z
            : 0.0f;
        row.dissipation =
            localRow == 0u
            ? materialA.response.w + materialB.response.w
            : 0.0f;
        row.timeConstant =
            2.0f * dispatch.timestepAndBias.x;
        row.dampingRatio = 1.0f;
        row.impulseLower =
            localRow == 0u
            ? 0.0f
            : -MR_CONSTRAINT_IR_UNBOUNDED;
        row.impulseUpper = MR_CONSTRAINT_IR_UNBOUNDED;
        row.flags =
            localRow == 0u
            ? (
                MR_CONSTRAINT_IR_ROW_POSITION_STABILIZED |
                MR_CONSTRAINT_IR_ROW_UNILATERAL |
                MR_CONSTRAINT_IR_ROW_CONTACT_NORMAL
            )
            : MR_CONSTRAINT_IR_ROW_CONTACT_TANGENT;
        rows[rowBase + localRow] = row;
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

    const uint ownerA = bodyA.flagsAndIndices[1];
    const uint ownerB = bodyB.flagsAndIndices[1];
    if (ownerA != MR_INVALID_INDEX &&
        ownerA < dispatch.articulationCount) {
        MRArticulatedPointImpulseGPU queryA = {};
        queryA.bodyIndex = sourceA.bodyIndex;
        queryA.localPoint = manifoldPoint.localAnchorA;
        const uint ownerQueryA =
            (ownerA * dispatch.environmentCount + environment) *
                dispatch.pointQueryStride +
            localQueryA;
        pointQueries[ownerQueryA] = queryA;
    }
    if (ownerB != MR_INVALID_INDEX &&
        ownerB < dispatch.articulationCount) {
        MRArticulatedPointImpulseGPU queryB = {};
        queryB.bodyIndex = sourceB.bodyIndex;
        queryB.localPoint = manifoldPoint.localAnchorB;
        const uint ownerQueryB =
            (ownerB * dispatch.environmentCount + environment) *
                dispatch.pointQueryStride +
            localQueryB;
        pointQueries[ownerQueryB] = queryB;
    }
}

// Broadphase-compacts one rod's full eligible edge/tool graph in stable pair
// order. One SIMD32 group owns one environment; its prefix scan publishes a
// dense active-pair list while clearing every pair-owned output slot. The
// heavyweight primitive/convex/mesh kernel therefore sees only overlapping
// pairs without changing eligibility, feature order, or warm-start identity.
kernel void mr_compact_rod_tool_pairs(
    device const MRRodGPUDispatch& dispatch [[buffer(0)]],
    device const MRRodColliderGPU* rodColliders [[buffer(1)]],
    device const MRRodToolPairGPU* toolPairs [[buffer(2)]],
    device const MRShapeGPU* toolShapes [[buffer(3)]],
    device const float4* rodPositions [[buffer(4)]],
    device const MRProjectedColliderGPU* projectedTools [[buffer(5)]],
    device uint* pairContactCounts [[buffer(6)]],
    device MRRodToolWitnessGPU* outputWitnesses [[buffer(7)]],
    device uint* activePairs [[buffer(8)]],
    device MRRodGPUStatus* statuses [[buffer(9)]],
    constant uint& rodIndex [[buffer(10)]],
    constant uint& rodCount [[buffer(11)]],
    const uint environment [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    if (environment >= dispatch.environmentCount ||
        lane >= MR_SIMD_WIDTH ||
        dispatch.toolPairCount == 0u ||
        rodIndex >= rodCount) {
        return;
    }
    const uint witnessDomain =
        dispatch.environmentCount * dispatch.toolContactStride;
    const uint pairDomain =
        dispatch.environmentCount * dispatch.toolPairWorldStride;
    const uint activeBase = witnessDomain +
        environment * dispatch.toolPairWorldStride;
    const uint metadataStride =
        MR_ROD_ACTIVE_GLOBAL_METADATA_WORDS +
        MR_ROD_ACTIVE_PER_ROD_METADATA_WORDS * rodCount;
    const uint metadataBase = witnessDomain + pairDomain +
        environment * metadataStride;
    const uint rodMetadataBase = metadataBase +
        MR_ROD_ACTIVE_GLOBAL_METADATA_WORDS +
        rodIndex * MR_ROD_ACTIVE_PER_ROD_METADATA_WORDS;
    const uint runningBase = rodIndex == 0u
        ? 0u
        : activePairs[metadataBase];
    const uint pairWorldBase =
        environment * dispatch.toolPairWorldStride;
    const uint nodeBase =
        environment * dispatch.stateNodeStride;
    const uint projectionBase =
        environment * dispatch.toolShapeCount;
    uint running = 0u;
    for (uint tile = 0u;
         tile < dispatch.toolPairCount;
         tile += MR_SIMD_WIDTH) {
        const uint pairIndex = tile + lane;
        const bool inRange = pairIndex < dispatch.toolPairCount;
        const uint globalPair = inRange
            ? dispatch.toolPairBase + pairIndex
            : 0u;
        const uint flatWorldPair = pairWorldBase + globalPair;
        bool active = false;
        uint failure = MR_ROD_GPU_SUCCESS;
        if (inRange) {
            pairContactCounts[flatWorldPair] = 0u;
            const uint witnessBase =
                flatWorldPair * MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
            for (uint slot = 0u;
                 slot < MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
                 ++slot) {
                outputWitnesses[
                    witnessBase + slot
                ].featuresAndFlags.w = 0u;
            }
            if (statuses[environment].code == MR_ROD_GPU_SUCCESS) {
                const MRRodToolPairGPU pair = toolPairs[globalPair];
                if ((pair.flags & MR_ROD_TOOL_PAIR_VALID) == 0u ||
                    pair.rodCollider < dispatch.rodEdgeBase ||
                    pair.rodCollider >=
                        dispatch.rodEdgeBase + dispatch.edgeCount ||
                    pair.rigidCollider >= dispatch.toolShapeCount) {
                    failure = MR_ROD_GPU_INVALID_DISPATCH;
                } else {
                    const MRRodColliderGPU rod =
                        rodColliders[pair.rodCollider];
                    if (rod.nodeA < dispatch.rodNodeBase ||
                        rod.nodeB < dispatch.rodNodeBase ||
                        rod.nodeA >=
                            dispatch.rodNodeBase + dispatch.nodeCount ||
                        rod.nodeB >=
                            dispatch.rodNodeBase + dispatch.nodeCount ||
                        rod.nodeA == rod.nodeB ||
                        !(rod.radiusAndOffsets.x > 0.0f) ||
                        !(rod.radiusAndOffsets.y >= 0.0f)) {
                        failure = MR_ROD_GPU_INVALID_DISPATCH;
                    } else {
                        const float3 endpointA =
                            rodPositions[nodeBase + rod.nodeA].xyz;
                        const float3 endpointB =
                            rodPositions[nodeBase + rod.nodeB].xyz;
                        if (!finiteFloat3(endpointA) ||
                            !finiteFloat3(endpointB) ||
                            !(dot(
                                endpointB - endpointA,
                                endpointB - endpointA
                            ) > 1.0e-20f)) {
                            failure = MR_ROD_GPU_DEGENERATE_GEOMETRY;
                        } else {
                            const MRProjectedColliderGPU projected =
                                projectedTools[
                                    projectionBase + pair.rigidCollider
                                ];
                            if (projected.statusAndFlags.x !=
                                MR_STEP_SUCCESS) {
                                failure =
                                    projected.statusAndFlags.x ==
                                            MR_STEP_NONFINITE_INPUT
                                    ? MR_ROD_GPU_NONFINITE_RESULT
                                    : MR_ROD_GPU_INVALID_DISPATCH;
                            } else if (
                                projected.statusAndFlags.y == 0u
                            ) {
                                const float expansion =
                                    rod.radiusAndOffsets.x +
                                    rod.radiusAndOffsets.y;
                                const float3 lower =
                                    min(endpointA, endpointB) - expansion;
                                const float3 upper =
                                    max(endpointA, endpointB) + expansion;
                                active =
                                    toolShapes[pair.rigidCollider].shapeType ==
                                        MR_SHAPE_PLANE ||
                                    !(
                                        any(
                                            lower > projected
                                                .upperAndContactOffset.xyz
                                        ) ||
                                        any(
                                            projected
                                                .lowerAndHalfLength.xyz >
                                            upper
                                        )
                                    );
                            }
                        }
                    }
                }
            }
            if (failure != MR_ROD_GPU_SUCCESS) {
                pairContactCounts[flatWorldPair] =
                    0x80000000u | failure;
                active = true;
            }
        }
        const uint prefix =
            simd_prefix_exclusive_sum(active ? 1u : 0u);
        if (active) {
            activePairs[
                activeBase + runningBase + running + prefix
            ] = globalPair;
        }
        running += simd_sum(active ? 1u : 0u);
    }
    if (lane == 0u) {
        const uint total = runningBase + running;
        activePairs[metadataBase + 0u] = total;
        activePairs[metadataBase + 1u] =
            (4u * total + MR_SIMD_WIDTH - 1u) / MR_SIMD_WIDTH;
        activePairs[metadataBase + 2u] = 1u;
        activePairs[metadataBase + 3u] = 1u;
        activePairs[rodMetadataBase + 0u] = runningBase;
        activePairs[rodMetadataBase + 1u] = running;
        activePairs[rodMetadataBase + 2u] =
            (running + MR_SIMD_WIDTH - 1u) / MR_SIMD_WIDTH;
        activePairs[rodMetadataBase + 3u] = 1u;
        activePairs[rodMetadataBase + 4u] = 1u;
    }
}

// Procedural rod edges enter the same certified primitive/convex/mesh query
// implementation as rigid colliders. Every pair owns four witness slots, so
// neither worker scheduling nor atomics can alter contact ordering.
kernel void mr_rod_tool_narrowphase(
    device const MRRodGPUDispatch& dispatch [[buffer(0)]],
    device const MRRodColliderGPU* rodColliders [[buffer(1)]],
    device const MRShapeGPU* rodShapeSources [[buffer(2)]],
    device const MRRodToolPairGPU* toolPairs [[buffer(3)]],
    device const MRShapeGPU* toolShapes [[buffer(4)]],
    device const MRBodyStateGPU* toolBodies [[buffer(5)]],
    device const MRGeometryHeaderGPU* geometryHeaders [[buffer(6)]],
    device const float4* geometryVertices [[buffer(7)]],
    device const MRMeshBVHNodeGPU* meshNodes [[buffer(8)]],
    device const MRMeshTriangleGPU* meshTriangles [[buffer(9)]],
    device const float4* rodPositions [[buffer(10)]],
    device const float4* rodVelocities [[buffer(11)]],
    device const float* rodTwistRates [[buffer(12)]],
    device const MRRodToolWitnessGPU* previousWitnesses [[buffer(13)]],
    device uint* pairContactCounts [[buffer(14)]],
    device MRRodToolWitnessGPU* outputWitnesses [[buffer(15)]],
    device const MRRodGPUStatus* statuses [[buffer(16)]],
    device const uint* activePairs [[buffer(17)]],
    constant uint& rodIndex [[buffer(18)]],
    constant uint& rodCount [[buffer(19)]],
    constant uint& environment [[buffer(20)]],
    const uint activeSlot [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount ||
        dispatch.toolPairCount == 0u ||
        rodIndex >= rodCount) {
        return;
    }
    const uint witnessDomain =
        dispatch.environmentCount * dispatch.toolContactStride;
    const uint pairDomain =
        dispatch.environmentCount * dispatch.toolPairWorldStride;
    const uint activeBase = witnessDomain +
        environment * dispatch.toolPairWorldStride;
    const uint metadataStride =
        MR_ROD_ACTIVE_GLOBAL_METADATA_WORDS +
        MR_ROD_ACTIVE_PER_ROD_METADATA_WORDS * rodCount;
    const uint rodMetadataBase = witnessDomain + pairDomain +
        environment * metadataStride +
        MR_ROD_ACTIVE_GLOBAL_METADATA_WORDS +
        rodIndex * MR_ROD_ACTIVE_PER_ROD_METADATA_WORDS;
    const uint activeStart = activePairs[rodMetadataBase + 0u];
    const uint activeCount = activePairs[rodMetadataBase + 1u];
    if (activeSlot >= activeCount) {
        return;
    }
    const uint globalPair = activePairs[
        activeBase + activeStart + activeSlot
    ];
    if (globalPair < dispatch.toolPairBase ||
        globalPair >=
            dispatch.toolPairBase + dispatch.toolPairCount) {
        return;
    }
    const uint flatWorldPair =
        environment * dispatch.toolPairWorldStride +
        globalPair;
    const uint witnessBase =
        flatWorldPair * MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
    if ((pairContactCounts[flatWorldPair] & 0x80000000u) != 0u) {
        return;
    }
    if (statuses[environment].code != MR_ROD_GPU_SUCCESS) {
        return;
    }

    const MRRodToolPairGPU pair = toolPairs[globalPair];
    if ((pair.flags & MR_ROD_TOOL_PAIR_VALID) == 0u ||
        pair.rodCollider < dispatch.rodEdgeBase ||
        pair.rodCollider >=
            dispatch.rodEdgeBase + dispatch.edgeCount ||
        pair.rigidCollider >= dispatch.toolShapeCount) {
        pairContactCounts[flatWorldPair] =
            0x80000000u | MR_ROD_GPU_INVALID_DISPATCH;
        return;
    }
    const MRRodColliderGPU rod =
        rodColliders[pair.rodCollider];
    if (rod.nodeA < dispatch.rodNodeBase ||
        rod.nodeB < dispatch.rodNodeBase ||
        rod.nodeA >=
            dispatch.rodNodeBase + dispatch.nodeCount ||
        rod.nodeB >=
            dispatch.rodNodeBase + dispatch.nodeCount ||
        rod.nodeA == rod.nodeB) {
        pairContactCounts[flatWorldPair] =
            0x80000000u | MR_ROD_GPU_INVALID_DISPATCH;
        return;
    }
    const uint nodeBase =
        environment * dispatch.stateNodeStride;
    const uint bodyBase =
        environment * dispatch.stateBodyStride;
    const float3 endpointA =
        rodPositions[nodeBase + rod.nodeA].xyz;
    const float3 endpointB =
        rodPositions[nodeBase + rod.nodeB].xyz;
    const float3 edge = endpointB - endpointA;
    const float edgeLengthSquared = dot(edge, edge);
    if (!(edgeLengthSquared > 1.0e-20f) ||
        !finiteFloat3(endpointA) ||
        !finiteFloat3(endpointB)) {
        pairContactCounts[flatWorldPair] =
            0x80000000u | MR_ROD_GPU_DEGENERATE_GEOMETRY;
        return;
    }
    const float edgeLength = sqrt(edgeLengthSquared);
    const float3 edgeAxis = edge / edgeLength;
    const float radius = rod.radiusAndOffsets.x;
    const float contactOffset = rod.radiusAndOffsets.y;
    if (!(radius > 0.0f) ||
        !(contactOffset >= 0.0f) ||
        !isfinite(radius) ||
        !isfinite(contactOffset)) {
        pairContactCounts[flatWorldPair] =
            0x80000000u | MR_ROD_GPU_INVALID_DISPATCH;
        return;
    }

    WorldShape rodShape = {};
    rodShape.index = dispatch.toolShapeCount + pair.rodCollider;
    rodShape.type = MR_SHAPE_CAPSULE;
    rodShape.body = MR_INVALID_INDEX;
    rodShape.rotation = float4(0.0f, 0.0f, 0.0f, 1.0f);
    rodShape.center = 0.5f * (endpointA + endpointB);
    rodShape.contactOffset = contactOffset;
    rodShape.radius = radius;
    rodShape.halfLength = 0.5f * edgeLength;
    rodShape.capsuleEndpoint0 = endpointA;
    rodShape.capsuleEndpoint1 = endpointB;
    rodShape.lower =
        min(endpointA, endpointB) - (radius + contactOffset);
    rodShape.upper =
        max(endpointA, endpointB) + (radius + contactOffset);
    rodShape.scale = float3(1.0f);

    WorldShape toolShape = {};
    uint failure = MR_STEP_SUCCESS;
    if (!makeWorldShape(
            pair.rigidCollider,
            toolShapes,
            geometryHeaders,
            toolBodies + bodyBase,
            dispatch.rigidBodyCount,
            toolShape,
            failure
        )) {
        pairContactCounts[flatWorldPair] =
            0x80000000u |
            (
                failure == MR_STEP_NONFINITE_INPUT
                ? MR_ROD_GPU_NONFINITE_RESULT
                : MR_ROD_GPU_INVALID_DISPATCH
            );
        return;
    }
    if (toolShape.disabled) {
        return;
    }
    if (toolShape.type != MR_SHAPE_PLANE &&
        (
            any(rodShape.lower > toolShape.upper) ||
            any(toolShape.lower > rodShape.upper)
        )) {
        return;
    }

    device const MRShapeGPU& rodSource =
        rodShapeSources[pair.rodCollider];
    device const MRShapeGPU& toolSource =
        toolShapes[pair.rigidCollider];
    ContactBatch batch = {};
    if (pair.pairClass == MR_COLLISION_PAIR_MESH) {
        uint triangleCandidates = 0u;
        batch = meshContacts(
            rodShape,
            toolShape,
            rodSource,
            toolSource,
            geometryHeaders,
            geometryVertices,
            meshNodes,
            meshTriangles,
            triangleCandidates
        );
    } else if (
        pair.pairClass == MR_COLLISION_PAIR_CONVEX
    ) {
        ConvexQueryResult query = {};
        batch = supportMappedContacts(
            rodShape.index,
            toolShape.index,
            rodShape,
            toolShape,
            rodSource,
            toolSource,
            geometryHeaders,
            geometryVertices,
            float3(0.0f),
            query
        );
        if (query.status != MR_STEP_SUCCESS &&
            query.status != MR_STEP_DID_NOT_CONVERGE) {
            pairContactCounts[flatWorldPair] =
                0x80000000u |
                MR_ROD_GPU_NONFINITE_RESULT;
            return;
        }
    } else {
        batch = generateContacts(
            rodShape.index,
            toolShape.index,
            pair.pairClass,
            rodShape,
            toolShape
        );
    }
    if (batch.count == 0u) {
        return;
    }

    WorldManifoldCandidate candidates[12];
    uint candidateCount = min(batch.count, 12u);
    for (uint contactIndex = 0u;
         contactIndex < candidateCount;
         ++contactIndex) {
        const MRRawContactGPU raw =
            batch.contacts[contactIndex];
        if (!finiteContact(raw)) {
            pairContactCounts[flatWorldPair] =
                0x80000000u |
                MR_ROD_GPU_NONFINITE_RESULT;
            return;
        }
        candidates[contactIndex] = {};
        candidates[contactIndex].point.localAnchorA =
            raw.pointAWorld;
        candidates[contactIndex].point.localAnchorB =
            raw.pointBWorld;
        candidates[contactIndex].point.featureAndLife[0] =
            raw.featureAndFlags[0];
        candidates[contactIndex].point.featureAndLife[1] =
            raw.featureAndFlags[1];
        candidates[contactIndex].point.featureAndLife[3] =
            raw.featureAndFlags[3];
        candidates[contactIndex].worldPoint =
            0.5f * (
                raw.pointAWorld.xyz + raw.pointBWorld.xyz
            );
        candidates[contactIndex].separation =
            raw.normalAndSeparation.w;
    }
    const float3 patchNormal = normalize(
        batch.contacts[0].normalAndSeparation.xyz
    );
    reduceCandidates(
        candidates,
        candidateCount,
        patchNormal
    );
    const uint count = min(
        candidateCount,
        uint(MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR)
    );
    for (uint slot = 0u; slot < count; ++slot) {
        const WorldManifoldCandidate selected =
            candidates[slot];
        const float3 rodPoint =
            selected.point.localAnchorA.xyz;
        const float3 toolPoint =
            selected.point.localAnchorB.xyz;
        const float weight = clamp(
            dot(rodPoint - endpointA, edge) /
                edgeLengthSquared,
            0.0f,
            1.0f
        );
        const float3 centerline =
            mix(endpointA, endpointB, weight);
        float3 radial = rodPoint - centerline;
        const float radialSquared = dot(radial, radial);
        radial =
            radialSquared > 1.0e-20f
            ? radial * rsqrt(radialSquared)
            : stableTangent(patchNormal);
        float3 tangentU =
            edgeAxis - patchNormal *
                dot(edgeAxis, patchNormal);
        const float tangentSquared = dot(tangentU, tangentU);
        tangentU =
            tangentSquared > 1.0e-20f
            ? tangentU * rsqrt(tangentSquared)
            : stableTangent(patchNormal);

        float4 warmImpulse = float4(0.0f);
        bool warmStarted = false;
        if ((dispatch.flags &
             MR_ROD_GPU_FLAG_TOOL_WARM_START) != 0u) {
            for (uint oldSlot = 0u;
                 oldSlot <
                     MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
                 ++oldSlot) {
                const MRRodToolWitnessGPU previous =
                    previousWitnesses[
                        witnessBase + oldSlot
                    ];
                if ((previous.featuresAndFlags.w &
                     MR_ROD_TOOL_WITNESS_VALID) != 0u &&
                    previous.identity.y == globalPair &&
                    previous.featuresAndFlags.x ==
                        selected.point.featureAndLife[0] &&
                    previous.featuresAndFlags.y ==
                        selected.point.featureAndLife[1]) {
                    warmImpulse = previous.impulses;
                    if (dot(
                            tangentU,
                            previous
                                .tangentUAndTwistJacobian.xyz
                        ) < 0.0f) {
                        tangentU = -tangentU;
                        warmImpulse.y = -warmImpulse.y;
                    }
                    warmStarted = true;
                    break;
                }
            }
        }
        const float3 tangentV =
            cross(patchNormal, tangentU);
        const float3 surfaceJacobian =
            cross(edgeAxis, radius * radial);
        const float twistJacobianU =
            dot(tangentU, surfaceJacobian);
        const float twistJacobianV =
            dot(tangentV, surfaceJacobian);
        const float3 rodVelocity =
            mix(
                rodVelocities[nodeBase + rod.nodeA].xyz,
                rodVelocities[nodeBase + rod.nodeB].xyz,
                weight
            ) +
            rodTwistRates[
                environment * dispatch.stateEdgeStride +
                pair.rodCollider
            ] * surfaceJacobian;
        device const MRBodyStateGPU& body =
            toolBodies[bodyBase + toolSource.bodyIndex];
        const float3 toolVelocity =
            body.flagsAndIndices[0] == MR_MOTION_STATIC
            ? float3(0.0f)
            : body.linearVelocityAndInverseMass.xyz +
                cross(
                    body.angularVelocity.xyz,
                    toolPoint - body.position.xyz
                );
        MRRodToolWitnessGPU witness = {};
        witness.identity = uint4(
            environment,
            globalPair,
            pair.rodCollider,
            pair.rigidCollider
        );
        witness.featuresAndFlags = uint4(
            selected.point.featureAndLife[0],
            selected.point.featureAndLife[1],
            toolSource.bodyIndex,
            MR_ROD_TOOL_WITNESS_VALID |
                (
                    warmStarted
                    ? MR_ROD_TOOL_WITNESS_WARM_STARTED
                    : MR_ROD_TOOL_WITNESS_NEW_IMPACT
                ) |
                (
                    isSurfaceShapeType(toolShape.type)
                    ? MR_ROD_TOOL_WITNESS_MESH
                    : 0u
                ) |
                (
                    pair.pairClass ==
                            MR_COLLISION_PAIR_CONVEX
                    ? MR_ROD_TOOL_WITNESS_HARD_CONVEX
                    : 0u
                )
        );
        uint toolMaterialIndex = toolSource.materialIndex;
        if ((selected.point.featureAndLife[3] &
             MR_RAW_CONTACT_MATERIAL_OVERRIDE) != 0u) {
            toolMaterialIndex =
                selected.point.featureAndLife[3] &
                MR_RAW_CONTACT_MATERIAL_INDEX_MASK;
        }
        witness.materialAndGeneration = uint4(
            rod.materialIndex,
            toolMaterialIndex,
            rod.topologyGeneration,
            MR_INVALID_INDEX
        );
        witness.rodPointAndWeight =
            float4(rodPoint, weight);
        witness.toolPointAndSeparation = float4(
            toolPoint,
            selected.separation -
                rod.radiusAndOffsets.z -
                toolSource.contactRestAndBoundingRadius.y
        );
        witness.normalAndPreSolveVelocity = float4(
            patchNormal,
            dot(toolVelocity - rodVelocity, patchNormal)
        );
        witness.tangentUAndTwistJacobian =
            float4(tangentU, twistJacobianU);
        witness.radialAndTwistJacobianV =
            float4(radial, twistJacobianV);
        witness.impulses = warmImpulse;
        outputWitnesses[witnessBase + slot] = witness;
    }
    pairContactCounts[flatWorldPair] = count;
}

// Associates pair-owned rod witnesses with the selected exact-impact prefix.
// This preserves one restitution semantic for rigid and rod contacts without
// coupling the generic rod narrowphase to the event scheduler.
kernel void mr_world_tag_rod_ccd_witnesses(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRCCDPairGPU* ccdPairs [[buffer(1)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(2)]],
    device MRRodToolWitnessGPU* witnesses [[buffer(3)]],
    const uint flatWitness [[thread_position_in_grid]]
) {
    const uint witnessStride =
        dispatch.rodToolPairCount *
        MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
    const uint witnessCount =
        dispatch.environmentCount * witnessStride;
    if (flatWitness >= witnessCount || witnessStride == 0u) {
        return;
    }
    const uint environment = flatWitness / witnessStride;
    device MRRodToolWitnessGPU& witness =
        witnesses[flatWitness];
    if ((witness.featuresAndFlags.w &
         MR_ROD_TOOL_WITNESS_VALID) == 0u) {
        return;
    }
    witness.materialAndGeneration.w = MR_INVALID_INDEX;
    const MRMetalWorldContactStatusGPU status =
        statuses[environment];
    const uint eventCount = min(
        status.reservedEvent0,
        min(
            status.ccdEvents,
            dispatch.ccdCandidateCapacity
        )
    );
    const uint encodedPair =
        0x80000000u | witness.identity.y;
    const uint candidateBase =
        environment * dispatch.ccdCandidateCapacity;
    for (uint eventSlot = 0u;
         eventSlot < eventCount;
         ++eventSlot) {
        const MRCCDPairGPU event =
            ccdPairs[candidateBase + eventSlot];
        if (event.compiledPair != encodedPair) {
            continue;
        }
        witness.materialAndGeneration.w = eventSlot;
        witness.featuresAndFlags.w &=
            ~static_cast<uint>(
                MR_ROD_TOOL_WITNESS_WARM_STARTED
            );
        witness.featuresAndFlags.w |=
            MR_ROD_TOOL_WITNESS_NEW_IMPACT;
        return;
    }
}

// Appends the already feature-reduced rod witnesses after the rigid manifold
// prefix. One SIMD32 group owns one environment and scans pairs in stable
// order. Each lane consumes its pair's at-most-four feature slots locally,
// then a SIMD prefix over the per-pair valid count preserves exact
// (pair, feature-slot) ordering. This avoids four full passes over the sparse
// all-pairs arena without introducing atomics or changing semantic order.
kernel void mr_world_scan_rod_contact_ir(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const uint* pairContactCounts [[buffer(1)]],
    device const MRRodToolWitnessGPU* witnesses [[buffer(2)]],
    device uint* witnessConstraintOffsets [[buffer(3)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(4)]],
    device const uint* activePairs [[buffer(5)]],
    constant uint& rodCount [[buffer(6)]],
    const uint environment [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    if (environment >= dispatch.environmentCount ||
        lane >= MR_SIMD_WIDTH ||
        dispatch.rodToolPairCount == 0u ||
        statuses[environment].code != MR_STEP_SUCCESS) {
        return;
    }
    const uint witnessStride =
        dispatch.rodToolPairCount *
        MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
    const uint witnessBase = environment * witnessStride;
    const uint pairBase =
        environment * dispatch.rodToolPairCount;
    const uint witnessDomain =
        dispatch.environmentCount * witnessStride;
    const uint pairDomain =
        dispatch.environmentCount * dispatch.rodToolPairCount;
    const uint activeBase = witnessDomain +
        environment * dispatch.rodToolPairCount;
    const uint metadataStride =
        MR_ROD_ACTIVE_GLOBAL_METADATA_WORDS +
        MR_ROD_ACTIVE_PER_ROD_METADATA_WORDS * rodCount;
    const uint activeCount = activePairs[
        witnessDomain + pairDomain +
        environment * metadataStride
    ];
    uint running = 0u;
    uint firstErrorPair = MR_INVALID_INDEX;
    uint firstErrorCode = MR_STEP_SUCCESS;
    float maximumPenetration = 0.0f;
    const uint rigidConstraintBase =
        statuses[environment].requiredConstraints;

    for (uint tile = 0u;
         tile < activeCount;
         tile += MR_SIMD_WIDTH) {
        const uint activePair = tile + lane;
        const bool inRange = activePair < activeCount;
        const uint pair = inRange
            ? activePairs[activeBase + activePair]
            : 0u;
        const uint encodedCount =
            inRange
            ? pairContactCounts[pairBase + pair]
            : 0u;
        const bool failed =
            inRange &&
            (encodedCount & 0x80000000u) != 0u;
        const uint count =
            encodedCount & 0x7fffffffu;
        uint pairValidCount = 0u;
        float pairMaximumPenetration = 0.0f;
        bool slotValid[MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR];
        for (uint slot = 0u;
             slot < MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
             ++slot) {
            slotValid[slot] = false;
            if (!inRange) {
                continue;
            }
            const uint localWitness =
                pair * MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR + slot;
            witnessConstraintOffsets[
                witnessBase + localWitness
            ] = MR_INVALID_INDEX;
            if (failed ||
                count > MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR ||
                slot >= count) {
                continue;
            }
            const MRRodToolWitnessGPU witness = witnesses[
                witnessBase + localWitness
            ];
            const bool valid =
                (witness.featuresAndFlags.w &
                 MR_ROD_TOOL_WITNESS_VALID) != 0u;
            slotValid[slot] = valid;
            pairValidCount += valid ? 1u : 0u;
            pairMaximumPenetration = max(
                pairMaximumPenetration,
                valid
                ? max(-witness.toolPointAndSeparation.w, 0.0f)
                : 0.0f
            );
        }
        const uint pairPrefix =
            simd_prefix_exclusive_sum(pairValidCount);
        uint localPrefix = 0u;
        if (inRange) {
            for (uint slot = 0u;
                 slot < MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
                 ++slot) {
                if (!slotValid[slot]) {
                    continue;
                }
                const uint localWitness =
                    pair * MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR + slot;
                witnessConstraintOffsets[
                    witnessBase + localWitness
                ] = rigidConstraintBase + running + pairPrefix +
                    localPrefix;
                ++localPrefix;
            }
        }
        const uint tileCount = simd_sum(pairValidCount);
        const uint errorPair = failed
            ? pair
            : MR_INVALID_INDEX;
        const uint tileFirstError = simd_min(errorPair);
        const float tilePenetration = simd_max(
            pairMaximumPenetration
        );
        if (lane == 0u) {
            if (firstErrorPair == MR_INVALID_INDEX &&
                tileFirstError != MR_INVALID_INDEX) {
                firstErrorPair = tileFirstError;
                const uint rodCode =
                    pairContactCounts[
                        pairBase + tileFirstError
                    ] & 0x7fffffffu;
                firstErrorCode =
                    rodCode == MR_ROD_GPU_NONFINITE_RESULT
                    ? MR_STEP_NONFINITE_RESULT
                    : (
                        rodCode ==
                                MR_ROD_GPU_DEGENERATE_GEOMETRY
                        ? MR_STEP_NONFINITE_INPUT
                        : MR_STEP_UNSUPPORTED
                    );
            }
            running += tileCount;
            maximumPenetration = max(
                maximumPenetration,
                tilePenetration
            );
        }
        running = simd_broadcast(running, 0u);
    }
    if (lane != 0u) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    const ulong requiredConstraints =
        static_cast<ulong>(rigidConstraintBase) +
        static_cast<ulong>(running);
    const ulong requiredRows =
        static_cast<ulong>(status.requiredRows) +
        3ul * static_cast<ulong>(running);
    status.requiredConstraints = static_cast<uint>(
        min(requiredConstraints, 0xfffffffful)
    );
    status.requiredRows = static_cast<uint>(
        min(requiredRows, 0xfffffffful)
    );
    status.activeContacts = status.requiredConstraints;
    status.newPoints =
        running > 0xffffffffu - status.newPoints
        ? 0xffffffffu
        : status.newPoints + running;
    status.diagnostics.y = max(
        status.diagnostics.y,
        maximumPenetration
    );
    if (firstErrorPair != MR_INVALID_INDEX) {
        status.code = firstErrorCode;
        status.firstFailingPair =
            0x80000000u | firstErrorPair;
        status.firstFailingStableKeyLow = firstErrorPair;
        status.firstFailingStableKeyHigh =
            0x524f4400u | environment;
    } else if (
        requiredConstraints > dispatch.constraintCapacity ||
        requiredRows > dispatch.rowCapacity ||
        2ul * requiredConstraints >
            dispatch.pointQueryStride) {
        status.code = MR_STEP_CONSTRAINT_CAPACITY_OVERFLOW;
        status.firstFailingConstraint =
            rigidConstraintBase +
            min(
                running,
                dispatch.constraintCapacity >
                        rigidConstraintBase
                    ? dispatch.constraintCapacity -
                        rigidConstraintBase
                    : 0u
            );
    }
    statuses[environment] = status;
}

kernel void mr_world_scatter_rod_contact_ir(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRRodColliderGPU* rodColliders [[buffer(1)]],
    device const MRRodToolPairGPU* toolPairs [[buffer(2)]],
    device const MRShapeGPU* shapes [[buffer(3)]],
    device const MRMaterialGPU* materials [[buffer(4)]],
    device const MRBodyStateGPU* bodies [[buffer(5)]],
    device const uint* pairContactCounts [[buffer(6)]],
    device const MRRodToolWitnessGPU* witnesses [[buffer(7)]],
    device const uint* witnessConstraintOffsets [[buffer(8)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(9)]],
    device MRContactConstraintGPU* contacts [[buffer(10)]],
    device MRContactPointMetaGPU* contactMetadata [[buffer(11)]],
    device MRConstraintIRBlockGPU* blocks [[buffer(12)]],
    device MRConstraintIREndpointGPU* endpoints [[buffer(13)]],
    device MRConstraintEndpointRuntimeGPU* endpointRuntime [[buffer(14)]],
    device MRConstraintIRRowGPU* rows [[buffer(15)]],
    device MRConstraintIRConeGPU* cones [[buffer(16)]],
    device MRArticulatedPointImpulseGPU* pointQueries [[buffer(17)]],
    device const uint* bodyDynamicNodes [[buffer(18)]],
    device uint* constraintWitnessIndices [[buffer(19)]],
    device const uint* activePairs [[buffer(20)]],
    constant uint& rodCount [[buffer(21)]],
    constant uint& environment [[buffer(22)]],
    const uint activeWitness [[thread_position_in_grid]]
) {
    const uint witnessStride =
        dispatch.rodToolPairCount *
        MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
    if (environment >= dispatch.environmentCount ||
        witnessStride == 0u) {
        return;
    }
    const uint witnessDomain =
        dispatch.environmentCount * witnessStride;
    const uint pairDomain =
        dispatch.environmentCount * dispatch.rodToolPairCount;
    const uint metadataStride =
        MR_ROD_ACTIVE_GLOBAL_METADATA_WORDS +
        MR_ROD_ACTIVE_PER_ROD_METADATA_WORDS * rodCount;
    const uint metadataBase = witnessDomain + pairDomain +
        environment * metadataStride;
    const uint activePairCount = activePairs[metadataBase];
    const uint activePair =
        activeWitness / MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
    const uint slot =
        activeWitness % MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
    if (activePair >= activePairCount) {
        return;
    }
    if (statuses[environment].code != MR_STEP_SUCCESS) {
        return;
    }
    const uint activeBase = witnessDomain +
        environment * dispatch.rodToolPairCount;
    const uint pairIndex = activePairs[
        activeBase + activePair
    ];
    if (pairIndex >= dispatch.rodToolPairCount) {
        return;
    }
    const uint localWitness =
        pairIndex * MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR + slot;
    const uint flatWitness =
        environment * witnessStride + localWitness;
    const uint pairCount =
        pairContactCounts[
            environment * dispatch.rodToolPairCount +
            pairIndex
        ];
    const uint currentConstraint =
        witnessConstraintOffsets[flatWitness];
    if (slot >= pairCount ||
        currentConstraint == MR_INVALID_INDEX ||
        currentConstraint >= dispatch.constraintCapacity) {
        return;
    }
    const MRRodToolWitnessGPU witness =
        witnesses[flatWitness];
    if ((witness.featuresAndFlags.w &
         MR_ROD_TOOL_WITNESS_VALID) == 0u) {
        return;
    }
    const MRRodToolPairGPU pair = toolPairs[pairIndex];
    const MRRodColliderGPU rod =
        rodColliders[pair.rodCollider];
    const MRShapeGPU toolShape = shapes[pair.rigidCollider];
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    device const MRBodyStateGPU& toolBody =
        bodies[bodyBase + toolShape.bodyIndex];
    float4 toolRotation;
    if (!checkedQuaternion(
            toolBody.orientation,
            toolRotation
        )) {
        return;
    }
    const float4 toolLocalAnchor = localAnchorFromWorld(
        toolBody,
        toolRotation,
        witness.toolPointAndSeparation.xyz
    );
    const float3 normal = normalize(
        witness.normalAndPreSolveVelocity.xyz
    );
    float3 tangentU =
        witness.tangentUAndTwistJacobian.xyz;
    tangentU -= normal * dot(tangentU, normal);
    tangentU =
        dot(tangentU, tangentU) > 1.0e-12f
        ? normalize(tangentU)
        : stableTangent(normal);
    const float3 tangentV = cross(normal, tangentU);
    const MRMaterialGPU rodMaterial =
        materials[witness.materialAndGeneration.x];
    const MRMaterialGPU toolMaterial =
        materials[witness.materialAndGeneration.y];
    const uint outputConstraint =
        environment * dispatch.constraintStride +
        currentConstraint;
    const uint toolBodyIndex = toolShape.bodyIndex;

    MRContactConstraintGPU contact = {};
    // Legacy rigid-only evaluators see a finite boundary record, while the
    // rod flag and endpoint sidecar select the exact typed operator.
    contact.bodyA = toolBodyIndex;
    contact.bodyB = toolBodyIndex;
    contact.flags =
        MR_CONSTRAINT_FLAG_ROD_ENDPOINT |
        (
            (witness.featuresAndFlags.w &
             MR_ROD_TOOL_WITNESS_NEW_IMPACT) != 0u
            ? MR_CONSTRAINT_FLAG_NEW_IMPACT
            : MR_CONSTRAINT_FLAG_WARM_STARTED
        );
    contact.islandIndex = MR_INVALID_INDEX;
    contact.pairKey =
        (static_cast<ulong>(
             dispatch.shapeCount + pair.rodCollider
         ) << 32u) |
        static_cast<ulong>(pair.rigidCollider);
    contact.featureKey =
        (static_cast<ulong>(witness.featuresAndFlags.x)
         << 32u) |
        static_cast<ulong>(witness.featuresAndFlags.y);
    contact.pointAndSeparation = float4(
        0.5f * (
            witness.rodPointAndWeight.xyz +
            witness.toolPointAndSeparation.xyz
        ),
        witness.toolPointAndSeparation.w
    );
    contact.normal = float4(normal, 0.0f);
    contact.tangent = float4(tangentU, 0.0f);
    contact.friction = float4(
        geometricMean(
            rodMaterial.friction.x,
            toolMaterial.friction.x
        ),
        geometricMean(
            rodMaterial.friction.y,
            toolMaterial.friction.y
        ),
        geometricMean(
            rodMaterial.friction.z,
            toolMaterial.friction.z
        ),
        geometricMean(
            rodMaterial.friction.w,
            toolMaterial.friction.w
        )
    );
    contact.response = float4(
        max(rodMaterial.response.x, toolMaterial.response.x),
        max(rodMaterial.response.y, toolMaterial.response.y),
        rodMaterial.response.z + toolMaterial.response.z,
        0.0f
    );
    contact.targetVelocityAndPreSolveNormal =
        float4(
            0.0f,
            0.0f,
            0.0f,
            witness.normalAndPreSolveVelocity.w
        );
    contact.impulses =
        (dispatch.flags &
         MR_METAL_WORLD_CONTACT_WARM_START) != 0u
        ? witness.impulses
        : float4(0.0f);
    contacts[outputConstraint] = contact;

    MRContactPointMetaGPU metadata = {};
    metadata.colliderA =
        dispatch.shapeCount + pair.rodCollider;
    metadata.colliderB = pair.rigidCollider;
    metadata.manifoldIndex = MR_INVALID_INDEX;
    metadata.pointIndex = slot;
    metadata.localAnchorA = float4(
        witness.radialAndTwistJacobianV.xyz,
        witness.rodPointAndWeight.w
    );
    metadata.localAnchorB = toolLocalAnchor;
    contactMetadata[outputConstraint] = metadata;

    MRConstraintIRBlockGPU block = {};
    block.key.words[0] =
        dispatch.shapeCount + pair.rodCollider;
    block.key.words[1] = pair.rigidCollider;
    block.key.words[2] = witness.featuresAndFlags.x;
    block.key.words[3] = witness.featuresAndFlags.y;
    block.type = MR_CONSTRAINT_CONTACT;
    block.dimension = 3u;
    block.flags = contact.flags;
    block.islandIndex = MR_INVALID_INDEX;
    block.endpointOffset = 2u * currentConstraint;
    block.endpointCount = 2u;
    block.rowOffset = 3u * currentConstraint;
    block.impulseOffset = 3u * currentConstraint;
    block.coneIndex = currentConstraint;
    block.eventSlot = witness.materialAndGeneration.w;
    block.reserved0 = flatWitness;
    blocks[outputConstraint] = block;
    constraintWitnessIndices[outputConstraint] = flatWitness;

    const uint endpointBase =
        2u * environment * dispatch.constraintStride;
    const uint endpointAIndex =
        endpointBase + 2u * currentConstraint;
    const uint endpointBIndex = endpointAIndex + 1u;
    MRConstraintIREndpointGPU endpointA = {};
    endpointA.objectIndex = rod.rodIndex;
    endpointA.articulationIndex =
        MR_CONSTRAINT_IR_INVALID_INDEX;
    endpointA.linkIndex = rod.edgeIndex;
    endpointA.role = MR_CONSTRAINT_IR_ENDPOINT_A;
    endpointA.jacobianKind =
        MR_CONSTRAINT_IR_JACOBIAN_ROD_EDGE;
    endpointA.anchor = float4(
        witness.radialAndTwistJacobianV.xyz,
        witness.rodPointAndWeight.w
    );
    endpointA.axis = float4(tangentU, 0.0f);
    MRConstraintIREndpointGPU endpointB = {};
    endpointB.objectIndex = toolBodyIndex;
    endpointB.articulationIndex =
        toolBody.flagsAndIndices[1];
    endpointB.linkIndex = toolBody.flagsAndIndices[2];
    endpointB.role = MR_CONSTRAINT_IR_ENDPOINT_B;
    endpointB.jacobianKind =
        MR_CONSTRAINT_IR_JACOBIAN_BODY_LOCAL_POINT;
    endpointB.anchor = toolLocalAnchor;
    endpoints[endpointAIndex] = endpointA;
    endpoints[endpointBIndex] = endpointB;

    MRConstraintEndpointRuntimeGPU rodRuntime = {};
    rodRuntime.dynamicNode = rod.flagsAndExclusions.y;
    rodRuntime.ownerKind =
        MR_CONSTRAINT_IR_OWNER_ROD_EDGE;
    rodRuntime.ownerIndex = rod.rodIndex;
    rodRuntime.elementIndex = rod.edgeIndex;
    rodRuntime.queryIndex =
        MR_CONSTRAINT_IR_INVALID_INDEX;
    rodRuntime.secondaryIndex = rod.nodeB;
    rodRuntime.twistIndex = rod.edgeIndex;
    rodRuntime.flags =
        MR_CONSTRAINT_IR_RUNTIME_DYNAMIC |
        MR_CONSTRAINT_IR_RUNTIME_HAS_TWIST;
    rodRuntime.weights = float4(
        1.0f - witness.rodPointAndWeight.w,
        witness.rodPointAndWeight.w,
        0.0f,
        0.0f
    );
    rodRuntime.localAnchorOrRadial =
        float4(
            witness.radialAndTwistJacobianV.xyz,
            0.0f
        );
    endpointRuntime[endpointAIndex] = rodRuntime;

    const uint localQueryB =
        2u * currentConstraint + 1u;
    const uint queryBase =
        environment * dispatch.pointQueryStride;
    endpointRuntime[endpointBIndex] = worldEndpointRuntime(
        toolBody,
        toolBodyIndex,
        bodyDynamicNodes[toolBodyIndex],
        queryBase + localQueryB,
        toolLocalAnchor
    );

    const uint rowBase =
        environment * dispatch.rowStride +
        3u * currentConstraint;
    const float3 directions[3] = {
        normal,
        tangentU,
        tangentV,
    };
    for (uint localRow = 0u;
         localRow < 3u;
         ++localRow) {
        MRConstraintIRRowGPU row = {};
        row.direction = float4(directions[localRow], 0.0f);
        row.positionError =
            localRow == 0u
            ? witness.toolPointAndSeparation.w
            : 0.0f;
        row.targetVelocity =
            localRow == 0u &&
            witness.toolPointAndSeparation.w > 0.0f
            ? -witness.toolPointAndSeparation.w /
                dispatch.timestepAndBias.x
            : 0.0f;
        row.compliance =
            localRow == 0u
            ? rodMaterial.response.z +
                toolMaterial.response.z
            : 0.0f;
        row.dissipation =
            localRow == 0u
            ? rodMaterial.response.w +
                toolMaterial.response.w
            : 0.0f;
        row.timeConstant =
            2.0f * dispatch.timestepAndBias.x;
        row.dampingRatio = 1.0f;
        row.impulseLower =
            localRow == 0u
            ? 0.0f
            : -MR_CONSTRAINT_IR_UNBOUNDED;
        row.impulseUpper = MR_CONSTRAINT_IR_UNBOUNDED;
        row.flags =
            localRow == 0u
            ? (
                MR_CONSTRAINT_IR_ROW_POSITION_STABILIZED |
                MR_CONSTRAINT_IR_ROW_UNILATERAL |
                MR_CONSTRAINT_IR_ROW_CONTACT_NORMAL
            )
            : MR_CONSTRAINT_IR_ROW_CONTACT_TANGENT;
        rows[rowBase + localRow] = row;
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

    const uint owner = toolBody.flagsAndIndices[1];
    if (owner != MR_INVALID_INDEX &&
        owner < dispatch.articulationCount) {
        MRArticulatedPointImpulseGPU query = {};
        query.bodyIndex = toolBodyIndex;
        query.localPoint = toolLocalAnchor;
        const uint ownerQuery =
            (owner * dispatch.environmentCount + environment) *
                dispatch.pointQueryStride +
            localQueryB;
        pointQueries[ownerQuery] = query;
    }
}
