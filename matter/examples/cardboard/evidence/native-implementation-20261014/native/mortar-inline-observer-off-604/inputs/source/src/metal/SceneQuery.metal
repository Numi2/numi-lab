#include <metal_stdlib>

#include "metalrobo/engine_types.h"
#include "metalrobo/scene_query_types.h"

using namespace metal;

namespace {

constant float kRayMinimum = 1.0e-6f;
constant float kDirectionMinimum = 1.0e-16f;
constant float kQuaternionMinimum = 1.0e-12f;
constant bool kSceneQueryUsesProjectedShapes
    [[function_constant(0)]];

struct Mat3 {
    float3 row0;
    float3 row1;
    float3 row2;
};

struct RayHit {
    float distance;
    float3 normal;
    uint shape;
    uint body;
    uint material;
    uint feature;
    bool valid;
};

inline bool finite3(const float3 value) {
    return all(isfinite(value));
}

inline bool finite4(const float4 value) {
    return all(isfinite(value));
}

inline bool normalizedQuaternion(
    const float4 input,
    thread float4& output
) {
    if (!finite4(input)) {
        return false;
    }
    const float squared = dot(input, input);
    if (!(squared > kQuaternionMinimum) ||
        !isfinite(squared)) {
        return false;
    }
    output = input * rsqrt(squared);
    return finite4(output);
}

inline float4 quaternionMultiply(
    const float4 left,
    const float4 right
) {
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

inline float3 quaternionRotate(
    const float4 quaternion,
    const float3 value
) {
    const float3 twiceCross =
        2.0f * cross(quaternion.xyz, value);
    return value +
        quaternion.w * twiceCross +
        cross(quaternion.xyz, twiceCross);
}

inline float3 quaternionInverseRotate(
    const float4 quaternion,
    const float3 value
) {
    return quaternionRotate(
        float4(-quaternion.xyz, quaternion.w),
        value
    );
}

inline Mat3 rotationMatrix(const float4 quaternion) {
    const float x = quaternion.x;
    const float y = quaternion.y;
    const float z = quaternion.z;
    const float w = quaternion.w;
    Mat3 result;
    result.row0 = float3(
        1.0f - 2.0f * (y * y + z * z),
        2.0f * (x * y - z * w),
        2.0f * (x * z + y * w)
    );
    result.row1 = float3(
        2.0f * (x * y + z * w),
        1.0f - 2.0f * (x * x + z * z),
        2.0f * (y * z - x * w)
    );
    result.row2 = float3(
        2.0f * (x * z - y * w),
        2.0f * (y * z + x * w),
        1.0f - 2.0f * (x * x + y * y)
    );
    return result;
}

inline Mat3 transpose(const thread Mat3& matrix) {
    Mat3 result;
    result.row0 = float3(
        matrix.row0.x,
        matrix.row1.x,
        matrix.row2.x
    );
    result.row1 = float3(
        matrix.row0.y,
        matrix.row1.y,
        matrix.row2.y
    );
    result.row2 = float3(
        matrix.row0.z,
        matrix.row1.z,
        matrix.row2.z
    );
    return result;
}

inline Mat3 multiply(
    const thread Mat3& left,
    const thread Mat3& right
) {
    const Mat3 transposed = transpose(right);
    Mat3 result;
    result.row0 = float3(
        dot(left.row0, transposed.row0),
        dot(left.row0, transposed.row1),
        dot(left.row0, transposed.row2)
    );
    result.row1 = float3(
        dot(left.row1, transposed.row0),
        dot(left.row1, transposed.row1),
        dot(left.row1, transposed.row2)
    );
    result.row2 = float3(
        dot(left.row2, transposed.row0),
        dot(left.row2, transposed.row1),
        dot(left.row2, transposed.row2)
    );
    return result;
}

inline bool writeWorldInverseInertia(
    thread MRBodyStateGPU& state,
    device const MRBodyPropertiesGPU& body,
    const float4 orientation
) {
    Mat3 bodyInverse;
    bodyInverse.row0 = body.inverseInertiaRow0.xyz;
    bodyInverse.row1 = body.inverseInertiaRow1.xyz;
    bodyInverse.row2 = body.inverseInertiaRow2.xyz;
    const Mat3 rotation = rotationMatrix(orientation);
    const Mat3 worldInverse = multiply(
        multiply(rotation, bodyInverse),
        transpose(rotation)
    );
    if (!finite3(worldInverse.row0) ||
        !finite3(worldInverse.row1) ||
        !finite3(worldInverse.row2)) {
        return false;
    }
    state.inverseInertiaWorldRow0 =
        float4(worldInverse.row0, 0.0f);
    state.inverseInertiaWorldRow1 =
        float4(worldInverse.row1, 0.0f);
    state.inverseInertiaWorldRow2 =
        float4(worldInverse.row2, 0.0f);
    return true;
}

inline uint component(const uint4 value, const uint index) {
    return value[index];
}

inline float3 dequantizedMeshBound(
    const uint4 encoded,
    const MRGeometryHeaderGPU geometry
) {
    const float3 fraction =
        float3(encoded.xyz) * (1.0f / 65535.0f);
    return mix(
        geometry.localLower.xyz,
        geometry.localUpper.xyz,
        fraction
    );
}

inline bool rayAabb(
    const float3 origin,
    const float3 direction,
    const float3 lower,
    const float3 upper,
    const float maximumDistance
) {
    float nearValue = 0.0f;
    float farValue = maximumDistance;
    for (uint axis = 0u; axis < 3u; ++axis) {
        if (abs(direction[axis]) <= kDirectionMinimum) {
            if (origin[axis] < lower[axis] ||
                origin[axis] > upper[axis]) {
                return false;
            }
            continue;
        }
        const float inverse = 1.0f / direction[axis];
        float first = (lower[axis] - origin[axis]) * inverse;
        float second = (upper[axis] - origin[axis]) * inverse;
        if (first > second) {
            const float temporary = first;
            first = second;
            second = temporary;
        }
        nearValue = max(nearValue, first);
        farValue = min(farValue, second);
        if (nearValue > farValue) {
            return false;
        }
    }
    return farValue >= kRayMinimum;
}

inline bool rayBoundingSphere(
    const float3 origin,
    const float3 direction,
    const float3 center,
    const float radius,
    const float maximumDistance
) {
    const float3 delta = center - origin;
    const float projection = dot(delta, direction);
    const float radiusSquared = radius * radius;
    const float closestSquared =
        max(dot(delta, delta) - projection * projection, 0.0f);
    if (closestSquared > radiusSquared) {
        return false;
    }
    const float halfChord =
        sqrt(max(radiusSquared - closestSquared, 0.0f));
    return
        projection + halfChord >= kRayMinimum &&
        projection - halfChord <= maximumDistance;
}

inline bool rayTriangle(
    const float3 origin,
    const float3 direction,
    const float3 a,
    const float3 b,
    const float3 c,
    const bool twoSided,
    thread float& parameter,
    thread float3& normal
) {
    const float3 edge1 = b - a;
    const float3 edge2 = c - a;
    const float3 rawNormal = cross(edge1, edge2);
    const float normalSquared = dot(rawNormal, rawNormal);
    if (!(normalSquared > kDirectionMinimum)) {
        return false;
    }
    const float3 p = cross(direction, edge2);
    const float determinant = dot(edge1, p);
    const float scale =
        max(max(length(edge1), length(edge2)), 1.0f);
    const float epsilon =
        32.0f * 1.1920929e-7f * scale;
    if ((twoSided && abs(determinant) <= epsilon) ||
        (!twoSided && determinant <= epsilon)) {
        return false;
    }
    const float inverse = 1.0f / determinant;
    const float3 translated = origin - a;
    const float u = dot(translated, p) * inverse;
    if (u < -epsilon || u > 1.0f + epsilon) {
        return false;
    }
    const float3 q = cross(translated, edge1);
    const float v = dot(direction, q) * inverse;
    if (v < -epsilon || u + v > 1.0f + epsilon) {
        return false;
    }
    const float candidate = dot(edge2, q) * inverse;
    if (!(candidate >= kRayMinimum) || !isfinite(candidate)) {
        return false;
    }
    parameter = candidate;
    normal = rawNormal * rsqrt(normalSquared);
    return true;
}

inline bool betterHit(
    const float candidate,
    const uint shape,
    const uint feature,
    const thread RayHit& current
) {
    if (!current.valid) {
        return true;
    }
    const float tolerance =
        8.0f * 1.1920929e-7f *
        max(max(candidate, current.distance), 1.0f);
    if (candidate < current.distance - tolerance) {
        return true;
    }
    if (abs(candidate - current.distance) > tolerance) {
        return false;
    }
    return shape < current.shape ||
        (shape == current.shape && feature < current.feature);
}

inline void commitHit(
    const float distance,
    const float3 normal,
    const uint shape,
    const uint body,
    const uint material,
    const uint feature,
    thread RayHit& hit
) {
    if (!isfinite(distance) || !finite3(normal) ||
        !betterHit(distance, shape, feature, hit)) {
        return;
    }
    const float squared = dot(normal, normal);
    if (!(squared > kDirectionMinimum)) {
        return;
    }
    hit.distance = distance;
    hit.normal = normal * rsqrt(squared);
    hit.shape = shape;
    hit.body = body;
    hit.material = material;
    hit.feature = feature;
    hit.valid = true;
}

inline void intersectSphere(
    const float3 origin,
    const float3 direction,
    const float radius,
    const float maximumDistance,
    const uint shape,
    const uint body,
    const uint material,
    thread RayHit& hit
) {
    const float b = dot(origin, direction);
    const float c = dot(origin, origin) - radius * radius;
    const float discriminant = b * b - c;
    if (discriminant < 0.0f) {
        return;
    }
    const float root = sqrt(max(discriminant, 0.0f));
    const float first = -b - root;
    const float second = -b + root;
    const float candidate =
        first >= kRayMinimum ? first : second;
    if (candidate < kRayMinimum ||
        candidate > maximumDistance) {
        return;
    }
    commitHit(
        candidate,
        origin + candidate * direction,
        shape,
        body,
        material,
        0u,
        hit
    );
}

inline void intersectPlane(
    const float3 origin,
    const float3 direction,
    const float maximumDistance,
    const uint shape,
    const uint body,
    const uint material,
    thread RayHit& hit
) {
    if (abs(direction.y) <= kDirectionMinimum) {
        return;
    }
    const float candidate = -origin.y / direction.y;
    if (candidate < kRayMinimum ||
        candidate > maximumDistance) {
        return;
    }
    commitHit(
        candidate,
        float3(0.0f, 1.0f, 0.0f),
        shape,
        body,
        material,
        0u,
        hit
    );
}

inline void intersectBox(
    const float3 origin,
    const float3 direction,
    const float3 halfExtent,
    const float maximumDistance,
    const uint shape,
    const uint body,
    const uint material,
    thread RayHit& hit
) {
    float nearValue = -INFINITY;
    float farValue = INFINITY;
    float3 nearNormal = float3(0.0f);
    float3 farNormal = float3(0.0f);
    uint nearFeature = 0u;
    uint farFeature = 0u;
    for (uint axis = 0u; axis < 3u; ++axis) {
        if (abs(direction[axis]) <= kDirectionMinimum) {
            if (origin[axis] < -halfExtent[axis] ||
                origin[axis] > halfExtent[axis]) {
                return;
            }
            continue;
        }
        float first =
            (-halfExtent[axis] - origin[axis]) /
            direction[axis];
        float second =
            (halfExtent[axis] - origin[axis]) /
            direction[axis];
        float3 firstNormal = float3(0.0f);
        float3 secondNormal = float3(0.0f);
        firstNormal[axis] = -1.0f;
        secondNormal[axis] = 1.0f;
        uint firstFeature = 2u * axis;
        uint secondFeature = 2u * axis + 1u;
        if (first > second) {
            const float temporary = first;
            first = second;
            second = temporary;
            const float3 normalTemporary = firstNormal;
            firstNormal = secondNormal;
            secondNormal = normalTemporary;
            const uint featureTemporary = firstFeature;
            firstFeature = secondFeature;
            secondFeature = featureTemporary;
        }
        if (first > nearValue) {
            nearValue = first;
            nearNormal = firstNormal;
            nearFeature = firstFeature;
        }
        if (second < farValue) {
            farValue = second;
            farNormal = secondNormal;
            farFeature = secondFeature;
        }
        if (nearValue > farValue) {
            return;
        }
    }
    const bool outside = nearValue >= kRayMinimum;
    const float candidate = outside ? nearValue : farValue;
    if (candidate < kRayMinimum ||
        candidate > maximumDistance) {
        return;
    }
    commitHit(
        candidate,
        outside ? nearNormal : farNormal,
        shape,
        body,
        material,
        outside ? nearFeature : farFeature,
        hit
    );
}

inline void intersectCylinder(
    const float3 origin,
    const float3 direction,
    const float radius,
    const float halfLength,
    const float maximumDistance,
    const uint shape,
    const uint body,
    const uint material,
    thread RayHit& hit
) {
    const float a =
        direction.x * direction.x +
        direction.z * direction.z;
    if (a > kDirectionMinimum) {
        const float b =
            2.0f *
            (origin.x * direction.x +
             origin.z * direction.z);
        const float c =
            origin.x * origin.x +
            origin.z * origin.z -
            radius * radius;
        const float discriminant = b * b - 4.0f * a * c;
        if (discriminant >= 0.0f) {
            const float root = sqrt(max(discriminant, 0.0f));
            const float inverse = 0.5f / a;
            const float candidates[2] = {
                (-b - root) * inverse,
                (-b + root) * inverse,
            };
            for (uint index = 0u; index < 2u; ++index) {
                const float candidate = candidates[index];
                const float y =
                    origin.y + candidate * direction.y;
                if (candidate >= kRayMinimum &&
                    candidate <= maximumDistance &&
                    abs(y) <= halfLength) {
                    const float3 point =
                        origin + candidate * direction;
                    commitHit(
                        candidate,
                        float3(point.x, 0.0f, point.z),
                        shape,
                        body,
                        material,
                        0u,
                        hit
                    );
                }
            }
        }
    }
    if (abs(direction.y) <= kDirectionMinimum) {
        return;
    }
    for (uint cap = 0u; cap < 2u; ++cap) {
        const float y = cap == 0u ? -halfLength : halfLength;
        const float candidate = (y - origin.y) / direction.y;
        const float3 point = origin + candidate * direction;
        if (candidate >= kRayMinimum &&
            candidate <= maximumDistance &&
            point.x * point.x + point.z * point.z <=
                radius * radius) {
            commitHit(
                candidate,
                cap == 0u
                    ? float3(0.0f, -1.0f, 0.0f)
                    : float3(0.0f, 1.0f, 0.0f),
                shape,
                body,
                material,
                cap + 1u,
                hit
            );
        }
    }
}

inline void intersectCapsule(
    const float3 origin,
    const float3 direction,
    const float radius,
    const float halfLength,
    const float maximumDistance,
    const uint shape,
    const uint body,
    const uint material,
    thread RayHit& hit
) {
    const float a =
        direction.x * direction.x +
        direction.z * direction.z;
    if (a > kDirectionMinimum) {
        const float b =
            2.0f *
            (origin.x * direction.x +
             origin.z * direction.z);
        const float c =
            origin.x * origin.x +
            origin.z * origin.z -
            radius * radius;
        const float discriminant = b * b - 4.0f * a * c;
        if (discriminant >= 0.0f) {
            const float root = sqrt(max(discriminant, 0.0f));
            const float inverse = 0.5f / a;
            const float candidates[2] = {
                (-b - root) * inverse,
                (-b + root) * inverse,
            };
            for (uint index = 0u; index < 2u; ++index) {
                const float candidate = candidates[index];
                const float y =
                    origin.y + candidate * direction.y;
                if (candidate >= kRayMinimum &&
                    candidate <= maximumDistance &&
                    abs(y) <= halfLength) {
                    const float3 point =
                        origin + candidate * direction;
                    commitHit(
                        candidate,
                        float3(point.x, 0.0f, point.z),
                        shape,
                        body,
                        material,
                        0u,
                        hit
                    );
                }
            }
        }
    }
    for (uint cap = 0u; cap < 2u; ++cap) {
        const float3 center = float3(
            0.0f,
            cap == 0u ? -halfLength : halfLength,
            0.0f
        );
        const float3 relative = origin - center;
        const float b = dot(relative, direction);
        const float c = dot(relative, relative) -
            radius * radius;
        const float discriminant = b * b - c;
        if (discriminant < 0.0f) {
            continue;
        }
        const float root = sqrt(max(discriminant, 0.0f));
        const float candidates[2] = {
            -b - root,
            -b + root,
        };
        for (uint index = 0u; index < 2u; ++index) {
            const float candidate = candidates[index];
            const float3 point =
                origin + candidate * direction;
            const bool onCap =
                cap == 0u
                ? point.y <= -halfLength
                : point.y >= halfLength;
            if (candidate >= kRayMinimum &&
                candidate <= maximumDistance &&
                onCap) {
                commitHit(
                    candidate,
                    point - center,
                    shape,
                    body,
                    material,
                    cap + 1u,
                    hit
                );
            }
        }
    }
}

inline void intersectConvex(
    const float3 origin,
    const float3 direction,
    const float3 scale,
    const MRGeometryHeaderGPU geometry,
    const uint convexFaceCount,
    device const MRConvexFaceGPU* faces,
    const float maximumDistance,
    const uint shape,
    const uint body,
    const uint material,
    thread RayHit& hit
) {
    if (geometry.faceCount == 0u ||
        geometry.faceOffset + geometry.faceCount >
            convexFaceCount ||
        any(scale <= float3(kRayMinimum))) {
        return;
    }
    const float3 localOrigin = origin / scale;
    const float3 localDirection = direction / scale;
    float enter = 0.0f;
    float exit = maximumDistance;
    float3 enterNormal = float3(0.0f);
    float3 exitNormal = float3(0.0f);
    uint enterFeature = 0u;
    uint exitFeature = 0u;
    bool hasEnter = false;
    bool hasExit = false;
    for (uint localFace = 0u;
         localFace < geometry.faceCount;
         ++localFace) {
        const MRConvexFaceGPU face =
            faces[geometry.faceOffset + localFace];
        const float signedDistance =
            dot(face.plane.xyz, localOrigin) -
            face.plane.w;
        const float denominator =
            dot(face.plane.xyz, localDirection);
        if (abs(denominator) <= kDirectionMinimum) {
            if (signedDistance > kRayMinimum) {
                return;
            }
            continue;
        }
        const float candidate =
            -signedDistance / denominator;
        if (denominator < 0.0f) {
            if (candidate > enter) {
                enter = candidate;
                enterNormal = face.plane.xyz;
                enterFeature = face.featureKey;
                hasEnter = true;
            }
        } else if (candidate < exit) {
            exit = candidate;
            exitNormal = face.plane.xyz;
            exitFeature = face.featureKey;
            hasExit = true;
        }
        if (enter > exit) {
            return;
        }
    }
    const bool outside = hasEnter && enter >= kRayMinimum;
    const float candidate = outside ? enter : exit;
    if (candidate < kRayMinimum ||
        candidate > maximumDistance ||
        (!outside && !hasExit)) {
        return;
    }
    const float3 localNormal =
        outside ? enterNormal : exitNormal;
    commitHit(
        candidate,
        localNormal / scale,
        shape,
        body,
        material,
        outside ? enterFeature : exitFeature,
        hit
    );
}

inline void intersectMesh(
    const float3 origin,
    const float3 direction,
    const float3 scale,
    const MRGeometryHeaderGPU geometry,
    const MRSceneQueryDispatchGPU dispatch,
    device const float4* vertices,
    device const MRMeshBVHNodeGPU* nodes,
    device const MRMeshTriangleGPU* triangles,
    const bool twoSided,
    const float maximumDistance,
    const uint shape,
    const uint body,
    const uint defaultMaterial,
    thread RayHit& hit
) {
    if (geometry.bvhCount == 0u ||
        geometry.bvhOffset + geometry.bvhCount >
            dispatch.meshNodeCount ||
        geometry.triangleOffset + geometry.triangleCount >
            dispatch.meshTriangleCount ||
        any(scale <= float3(kRayMinimum))) {
        return;
    }
    const float3 localOrigin = origin / scale;
    const float3 localDirection = direction / scale;
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
            nodes[geometry.bvhOffset + nodeIndex];
        const uint metadata = component(node.childMeta, slot);
        const uint escape =
            (metadata & MR_MESH_BVH_ESCAPE_MASK) >>
            MR_MESH_BVH_ESCAPE_SHIFT;
        const uint child = component(node.childIndices, slot);
        const float distanceLimit =
            hit.valid
            ? min(hit.distance, maximumDistance)
            : maximumDistance;
        if (child == MR_INVALID_INDEX ||
            !rayAabb(
                localOrigin,
                localDirection,
                dequantizedMeshBound(
                    node.quantizedLower[slot],
                    geometry
                ),
                dequantizedMeshBound(
                    node.quantizedUpper[slot],
                    geometry
                ),
                distanceLimit
            )) {
            cursor = escape;
            continue;
        }
        if ((metadata & MR_MESH_BVH_LEAF_BIT) == 0u) {
            cursor = child * MR_MESH_BVH_BRANCHING;
            continue;
        }
        const uint count =
            metadata & MR_MESH_BVH_LEAF_COUNT_MASK;
        for (uint local = 0u; local < count; ++local) {
            const uint triangleIndex =
                geometry.triangleOffset + child + local;
            if (triangleIndex >=
                    geometry.triangleOffset +
                        geometry.triangleCount ||
                triangleIndex >= dispatch.meshTriangleCount) {
                continue;
            }
            const MRMeshTriangleGPU triangle =
                triangles[triangleIndex];
            if (triangle.verticesAndFeature.x >=
                    dispatch.vertexCount ||
                triangle.verticesAndFeature.y >=
                    dispatch.vertexCount ||
                triangle.verticesAndFeature.z >=
                    dispatch.vertexCount) {
                continue;
            }
            float parameter = 0.0f;
            float3 localNormal = float3(0.0f);
            if (!rayTriangle(
                    localOrigin,
                    localDirection,
                    vertices[
                        triangle.verticesAndFeature.x
                    ].xyz,
                    vertices[
                        triangle.verticesAndFeature.y
                    ].xyz,
                    vertices[
                        triangle.verticesAndFeature.z
                    ].xyz,
                    twoSided,
                    parameter,
                    localNormal
                ) ||
                parameter > maximumDistance) {
                continue;
            }
            commitHit(
                parameter,
                localNormal / scale,
                shape,
                body,
                triangle.materialAndFlags.x ==
                        MR_INVALID_INDEX
                    ? defaultMaterial
                    : triangle.materialAndFlags.x,
                triangle.verticesAndFeature.w,
                hit
            );
        }
        cursor = escape;
    }
}

inline void clearRayResult(
    device float* distances,
    device float4* points,
    device float4* normals,
    device uint4* identities,
    device uint* validity,
    const uint index
) {
    distances[index] = -1.0f;
    points[index] = float4(0.0f);
    normals[index] = float4(0.0f);
    identities[index] = uint4(MR_INVALID_INDEX);
    validity[index] = 0u;
}

inline bool traceSceneRay(
    const MRSceneQueryDispatchGPU dispatch,
    device const MRShapeGPU* shapes,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    device const MRMeshBVHNodeGPU* meshNodes,
    device const MRMeshTriangleGPU* meshTriangles,
    device const MRConvexFaceGPU* convexFaces,
    device const MRBodyStateGPU* bodyStates,
    device const MRSceneQueryShapeStateGPU* projectedShapes,
    const float3 origin,
    const float3 rawDirection,
    const float maximumDistance,
    const uint4 queryOptions,
    const uint environment,
    thread float3& direction,
    thread RayHit& hit
) {
    direction = float3(0.0f);
    hit.distance =
        isfinite(maximumDistance) ? maximumDistance : 0.0f;
    hit.normal = float3(0.0f);
    hit.shape = MR_INVALID_INDEX;
    hit.body = MR_INVALID_INDEX;
    hit.material = MR_INVALID_INDEX;
    hit.feature = MR_INVALID_INDEX;
    hit.valid = false;

    const float directionSquared =
        dot(rawDirection, rawDirection);
    if (environment >= dispatch.environmentCount ||
        !finite3(origin) ||
        !finite3(rawDirection) ||
        !(directionSquared > kDirectionMinimum) ||
        !isfinite(maximumDistance) ||
        !(maximumDistance > 0.0f)) {
        return false;
    }
    direction = rawDirection * rsqrt(directionSquared);
    const uint queryGroup = queryOptions.x;
    const uint queryMask = queryOptions.y;
    const uint excludedBody = queryOptions.z;
    const uint flags = queryOptions.w;
    const uint bodyBase = environment * dispatch.bodyStride;

    for (uint shapeIndex = 0u;
         shapeIndex < dispatch.shapeCount;
         ++shapeIndex) {
        const MRShapeGPU shape = shapes[shapeIndex];
        if (shape.bodyIndex >= dispatch.bodyCount ||
            shape.bodyIndex == excludedBody ||
            (
                (shape.flags &
                 MR_SHAPE_FLAG_SIMULATION_DISABLED) != 0u &&
                (flags &
                 MR_SCENE_QUERY_INCLUDE_DISABLED) == 0u
            ) ||
            (shape.collisionGroup & queryMask) == 0u ||
            (queryGroup & shape.collisionMask) == 0u) {
            continue;
        }
        float4 rotation;
        float3 center;
        if (kSceneQueryUsesProjectedShapes) {
            const MRSceneQueryShapeStateGPU projected =
                projectedShapes[
                    environment * dispatch.shapeCount +
                    shapeIndex
                ];
            if (projected.centerAndRadius.w < -1.5f ||
                !finite4(projected.centerAndRadius) ||
                !finite4(projected.rotation)) {
                continue;
            }
            center = projected.centerAndRadius.xyz;
            rotation = projected.rotation;
            if (projected.centerAndRadius.w >= 0.0f) {
                const float distanceLimit =
                    hit.valid
                    ? min(hit.distance, maximumDistance)
                    : maximumDistance;
                if (!rayBoundingSphere(
                        origin,
                        direction,
                        center,
                        projected.centerAndRadius.w,
                        distanceLimit
                    )) {
                    continue;
                }
            }
        } else {
            const MRBodyStateGPU body =
                bodyStates[bodyBase + shape.bodyIndex];
            float4 bodyRotation;
            float4 localRotation;
            if (!finite4(body.position) ||
                !normalizedQuaternion(
                    body.orientation,
                    bodyRotation
                ) ||
                !normalizedQuaternion(
                    shape.localRotation,
                    localRotation
                )) {
                continue;
            }
            rotation =
                quaternionMultiply(bodyRotation, localRotation);
            center =
                body.position.xyz +
                quaternionRotate(
                    bodyRotation,
                    shape.localPosition.xyz
                );
        }
        const float3 localOrigin =
            quaternionInverseRotate(
                rotation,
                origin - center
            );
        const float3 localDirection =
            quaternionInverseRotate(rotation, direction);
        RayHit localHit = hit;
        localHit.normal = float3(0.0f);

        switch (shape.shapeType) {
        case MR_SHAPE_SPHERE:
            intersectSphere(
                localOrigin,
                localDirection,
                shape.dimensions.x,
                maximumDistance,
                shapeIndex,
                shape.bodyIndex,
                shape.materialIndex,
                localHit
            );
            break;
        case MR_SHAPE_CAPSULE:
            intersectCapsule(
                localOrigin,
                localDirection,
                shape.dimensions.x,
                shape.dimensions.y,
                maximumDistance,
                shapeIndex,
                shape.bodyIndex,
                shape.materialIndex,
                localHit
            );
            break;
        case MR_SHAPE_BOX:
            intersectBox(
                localOrigin,
                localDirection,
                shape.dimensions.xyz,
                maximumDistance,
                shapeIndex,
                shape.bodyIndex,
                shape.materialIndex,
                localHit
            );
            break;
        case MR_SHAPE_PLANE:
            intersectPlane(
                localOrigin,
                localDirection,
                maximumDistance,
                shapeIndex,
                shape.bodyIndex,
                shape.materialIndex,
                localHit
            );
            break;
        case MR_SHAPE_CYLINDER:
            intersectCylinder(
                localOrigin,
                localDirection,
                shape.dimensions.x,
                shape.dimensions.y,
                maximumDistance,
                shapeIndex,
                shape.bodyIndex,
                shape.materialIndex,
                localHit
            );
            break;
        case MR_SHAPE_CONVEX:
            if (shape.geometryCount == 1u &&
                shape.geometryOffset < dispatch.geometryCount) {
                intersectConvex(
                    localOrigin,
                    localDirection,
                    shape.dimensions.xyz,
                    geometryHeaders[shape.geometryOffset],
                    dispatch.convexFaceCount,
                    convexFaces,
                    maximumDistance,
                    shapeIndex,
                    shape.bodyIndex,
                    shape.materialIndex,
                    localHit
                );
            }
            break;
        case MR_SHAPE_TRIANGLE_MESH:
            if (shape.geometryCount == 1u &&
                shape.geometryOffset < dispatch.geometryCount) {
                const bool twoSided =
                    (shape.flags &
                     MR_SHAPE_FLAG_MESH_TWO_SIDED) != 0u ||
                    (flags &
                     MR_SCENE_QUERY_FORCE_TWO_SIDED) != 0u;
                intersectMesh(
                    localOrigin,
                    localDirection,
                    shape.dimensions.xyz,
                    geometryHeaders[shape.geometryOffset],
                    dispatch,
                    geometryVertices,
                    meshNodes,
                    meshTriangles,
                    twoSided,
                    maximumDistance,
                    shapeIndex,
                    shape.bodyIndex,
                    shape.materialIndex,
                    localHit
                );
            }
            break;
        default:
            break;
        }

        if (localHit.valid &&
            betterHit(
                localHit.distance,
                localHit.shape,
                localHit.feature,
                hit
            )) {
            float3 worldNormal = quaternionRotate(
                rotation,
                localHit.normal
            );
            const float normalSquared =
                dot(worldNormal, worldNormal);
            if (!(normalSquared > kDirectionMinimum) ||
                !finite3(worldNormal)) {
                continue;
            }
            worldNormal *= rsqrt(normalSquared);
            hit = localHit;
            hit.normal = worldNormal;
        }
    }

    if (!hit.valid) {
        return false;
    }
    if ((flags & MR_SCENE_QUERY_FACE_FORWARD_NORMAL) != 0u &&
        dot(hit.normal, direction) > 0.0f) {
        hit.normal = -hit.normal;
    }
    return true;
}

inline void writeRayHit(
    device float* distances,
    device float4* points,
    device float4* normals,
    device uint4* identities,
    device uint* validity,
    const uint index,
    const float3 origin,
    const float3 direction,
    const thread RayHit& hit
) {
    distances[index] = hit.distance;
    points[index] = float4(
        origin + hit.distance * direction,
        1.0f
    );
    normals[index] = float4(hit.normal, 0.0f);
    identities[index] = uint4(
        hit.shape,
        hit.body,
        hit.material,
        hit.feature
    );
    validity[index] = 1u;
}

} // namespace

kernel void mr_scene_query_project_shapes(
    constant MRSceneQueryDispatchGPU& dispatch [[buffer(0)]],
    device const MRShapeGPU* shapes [[buffer(1)]],
    device const MRGeometryHeaderGPU* geometryHeaders
        [[buffer(2)]],
    device const MRBodyStateGPU* bodyStates [[buffer(3)]],
    device MRSceneQueryShapeStateGPU* projectedShapes
        [[buffer(4)]],
    const uint index [[thread_position_in_grid]]
) {
    const uint count =
        dispatch.environmentCount * dispatch.shapeCount;
    if (dispatch.abiVersion != MR_SCENE_QUERY_ABI_VERSION ||
        dispatch.bodyStride < dispatch.bodyCount ||
        index >= count) {
        return;
    }
    MRSceneQueryShapeStateGPU output{};
    output.centerAndRadius.w = -2.0f;
    const uint environment = index / dispatch.shapeCount;
    const uint shapeIndex =
        index - environment * dispatch.shapeCount;
    const MRShapeGPU shape = shapes[shapeIndex];
    if (shape.bodyIndex >= dispatch.bodyCount ||
        !finite4(shape.localPosition) ||
        !finite4(shape.localRotation) ||
        !finite4(shape.dimensions) ||
        !finite4(shape.contactRestAndBoundingRadius)) {
        projectedShapes[index] = output;
        return;
    }
    const MRBodyStateGPU body =
        bodyStates[
            environment * dispatch.bodyStride +
            shape.bodyIndex
        ];
    float4 bodyRotation;
    float4 localRotation;
    if (!finite4(body.position) ||
        !normalizedQuaternion(
            body.orientation,
            bodyRotation
        ) ||
        !normalizedQuaternion(
            shape.localRotation,
            localRotation
        )) {
        projectedShapes[index] = output;
        return;
    }
    const float3 center =
        body.position.xyz +
        quaternionRotate(
            bodyRotation,
            shape.localPosition.xyz
        );
    const float4 rotation =
        quaternionMultiply(bodyRotation, localRotation);
    float radius = -2.0f;
    switch (shape.shapeType) {
    case MR_SHAPE_PLANE:
        radius = -1.0f;
        break;
    case MR_SHAPE_SPHERE:
        if (shape.dimensions.x > 0.0f) {
            radius = max(
                shape.contactRestAndBoundingRadius.z,
                shape.dimensions.x
            );
        }
        break;
    case MR_SHAPE_CAPSULE:
        if (shape.dimensions.x > 0.0f &&
            shape.dimensions.y >= 0.0f) {
            radius = max(
                shape.contactRestAndBoundingRadius.z,
                shape.dimensions.x + shape.dimensions.y
            );
        }
        break;
    case MR_SHAPE_BOX:
        if (all(shape.dimensions.xyz > float3(0.0f))) {
            radius = max(
                shape.contactRestAndBoundingRadius.z,
                length(shape.dimensions.xyz)
            );
        }
        break;
    case MR_SHAPE_CYLINDER:
        if (shape.dimensions.x > 0.0f &&
            shape.dimensions.y > 0.0f) {
            radius = max(
                shape.contactRestAndBoundingRadius.z,
                length(
                    float2(
                        shape.dimensions.x,
                        shape.dimensions.y
                    )
                )
            );
        }
        break;
    case MR_SHAPE_CONVEX:
    case MR_SHAPE_TRIANGLE_MESH:
        if (shape.geometryCount == 1u &&
            shape.geometryOffset < dispatch.geometryCount &&
            all(shape.dimensions.xyz > float3(0.0f))) {
            const MRGeometryHeaderGPU geometry =
                geometryHeaders[shape.geometryOffset];
            if (finite4(geometry.localLower) &&
                finite4(geometry.localUpper) &&
                all(geometry.localLower.xyz <=
                    geometry.localUpper.xyz)) {
                const float3 corner = max(
                    abs(
                        geometry.localLower.xyz *
                        shape.dimensions.xyz
                    ),
                    abs(
                        geometry.localUpper.xyz *
                        shape.dimensions.xyz
                    )
                );
                radius = max(
                    shape.contactRestAndBoundingRadius.z,
                    length(corner)
                );
            }
        }
        break;
    default:
        break;
    }
    if (!finite3(center) ||
        !finite4(rotation) ||
        !isfinite(radius) ||
        radius < -1.0f) {
        projectedShapes[index] = output;
        return;
    }
    if (radius >= 0.0f) {
        const float scale = max(
            max(
                max(abs(center.x), abs(center.y)),
                abs(center.z)
            ),
            radius
        );
        radius +=
            (scale + 1.0f) *
            MR_COLLISION_AABB_RELATIVE_PAD;
    }
    output.centerAndRadius = float4(center, radius);
    output.rotation = rotation;
    projectedShapes[index] = output;
}

kernel void mr_scene_query_pack_body_states(
    constant MRBodyStateMaterializeDispatchGPU& dispatch
        [[buffer(0)]],
    device const MRBodyPropertiesGPU* bodyProperties
        [[buffer(1)]],
    device const uint* bodyToScene [[buffer(2)]],
    device const MRArticulatedBodyPoseGPU* bodyPoses
        [[buffer(3)]],
    device const MRArticulatedOperatorStatusGPU* operatorStatuses
        [[buffer(4)]],
    device const float4* scenePosition [[buffer(5)]],
    device const float4* sceneOrientation [[buffer(6)]],
    device const float4* sceneLinearVelocity [[buffer(7)]],
    device const float4* sceneAngularVelocity [[buffer(8)]],
    device MRBodyStateGPU* bodyStates [[buffer(9)]],
    const uint index [[thread_position_in_grid]]
) {
    const uint count =
        dispatch.environmentCount * dispatch.bodyCount;
    if (dispatch.abiVersion != MR_SCENE_QUERY_ABI_VERSION ||
        dispatch.bodyStride < dispatch.bodyCount ||
        index >= count) {
        return;
    }
    const uint environment = index / dispatch.bodyCount;
    const uint body =
        index - environment * dispatch.bodyCount;
    device const MRBodyPropertiesGPU& properties =
        bodyProperties[body];
    MRBodyStateGPU state = {};
    state.flagsAndIndices[0] = properties.motionType;
    state.flagsAndIndices[1] = properties.articulationIndex;
    state.flagsAndIndices[2] = body;
    state.flagsAndIndices[3] = 0u;

    if (properties.articulationIndex != MR_INVALID_INDEX) {
        const uint owner = properties.articulationIndex;
        if (owner >= dispatch.articulationCount ||
            operatorStatuses[
                owner * dispatch.environmentCount +
                environment
            ].code != MR_ARTICULATED_OPERATOR_SUCCESS) {
            bodyStates[
                environment * dispatch.bodyStride + body
            ] = state;
            return;
        }
        const MRArticulatedBodyPoseGPU pose =
            bodyPoses[
                environment * dispatch.bodyCount + body
            ];
        if (!finite4(pose.position) ||
            !finite4(pose.orientation) ||
            !(dot(pose.orientation, pose.orientation) >
              kQuaternionMinimum)) {
            bodyStates[
                environment * dispatch.bodyStride + body
            ] = state;
            return;
        }
        state.position = float4(pose.position.xyz, 1.0f);
        // The articulated operator already owns quaternion normalization.
        // Preserve its exact published components so this materializer and
        // the solver's body arena cannot introduce different visual motion.
        state.orientation = pose.orientation;
        bodyStates[
            environment * dispatch.bodyStride + body
        ] = state;
        return;
    }

    const uint localScene = bodyToScene[body];
    if (localScene >= dispatch.sceneBodyCount) {
        bodyStates[
            environment * dispatch.bodyStride + body
        ] = state;
        return;
    }
    const uint sceneIndex =
        environment * dispatch.sceneBodyCount + localScene;
    float4 orientation;
    if (!finite4(scenePosition[sceneIndex]) ||
        !normalizedQuaternion(
            sceneOrientation[sceneIndex],
            orientation
        ) ||
        !finite4(sceneLinearVelocity[sceneIndex]) ||
        !finite4(sceneAngularVelocity[sceneIndex])) {
        bodyStates[
            environment * dispatch.bodyStride + body
        ] = state;
        return;
    }
    state.position =
        float4(scenePosition[sceneIndex].xyz, 1.0f);
    state.orientation = orientation;
    state.linearVelocityAndInverseMass = float4(
        sceneLinearVelocity[sceneIndex].xyz,
        properties.motionType == MR_MOTION_DYNAMIC
            ? properties.massAndInverseMass.y
            : 0.0f
    );
    state.angularVelocity =
        float4(sceneAngularVelocity[sceneIndex].xyz, 0.0f);
    state.flagsAndIndices[1] = MR_INVALID_INDEX;
    if (!writeWorldInverseInertia(
            state,
            properties,
            orientation
        )) {
        state.orientation = float4(0.0f);
    }
    bodyStates[
        environment * dispatch.bodyStride + body
    ] = state;
}

kernel void mr_scene_raycast(
    constant MRSceneQueryDispatchGPU& dispatch [[buffer(0)]],
    device const MRShapeGPU* shapes [[buffer(1)]],
    device const MRGeometryHeaderGPU* geometryHeaders
        [[buffer(2)]],
    device const float4* geometryVertices [[buffer(3)]],
    device const MRMeshBVHNodeGPU* meshNodes [[buffer(4)]],
    device const MRMeshTriangleGPU* meshTriangles [[buffer(5)]],
    device const MRConvexFaceGPU* convexFaces [[buffer(6)]],
    device const MRBodyStateGPU* bodyStates [[buffer(7)]],
    device const MRSceneQueryShapeStateGPU* projectedShapes
        [[buffer(8)]],
    device const float4* origins [[buffer(9)]],
    device const float4* directions [[buffer(10)]],
    device const float* maximumDistances [[buffer(11)]],
    device const uint4* options [[buffer(12)]],
    device float* distances [[buffer(13)]],
    device float4* points [[buffer(14)]],
    device float4* normals [[buffer(15)]],
    device uint4* identities [[buffer(16)]],
    device uint* validity [[buffer(17)]],
    const uint index [[thread_position_in_grid]]
) {
    const uint queryCount =
        dispatch.environmentCount * dispatch.rayCount;
    if (dispatch.abiVersion != MR_SCENE_QUERY_ABI_VERSION ||
        dispatch.rayCount == 0u ||
        dispatch.rayStride < dispatch.rayCount ||
        dispatch.bodyStride < dispatch.bodyCount ||
        index >= queryCount) {
        return;
    }
    clearRayResult(
        distances,
        points,
        normals,
        identities,
        validity,
        index
    );

    const float3 origin = origins[index].xyz;
    const float3 rawDirection = directions[index].xyz;
    const float maximumDistance = maximumDistances[index];
    const uint4 queryOptions = options[index];
    const uint environment = index / dispatch.rayCount;
    float3 direction;
    RayHit hit;
    if (!traceSceneRay(
            dispatch,
            shapes,
            geometryHeaders,
            geometryVertices,
            meshNodes,
            meshTriangles,
            convexFaces,
            bodyStates,
            projectedShapes,
            origin,
            rawDirection,
            maximumDistance,
            queryOptions,
            environment,
            direction,
            hit
        )) {
        return;
    }
    writeRayHit(
        distances,
        points,
        normals,
        identities,
        validity,
        index,
        origin,
        direction,
        hit
    );
}

kernel void mr_scene_raycast_pattern(
    constant MRSceneQueryDispatchGPU& dispatch [[buffer(0)]],
    device const MRShapeGPU* shapes [[buffer(1)]],
    device const MRGeometryHeaderGPU* geometryHeaders
        [[buffer(2)]],
    device const float4* geometryVertices [[buffer(3)]],
    device const MRMeshBVHNodeGPU* meshNodes [[buffer(4)]],
    device const MRMeshTriangleGPU* meshTriangles [[buffer(5)]],
    device const MRConvexFaceGPU* convexFaces [[buffer(6)]],
    device const MRBodyStateGPU* bodyStates [[buffer(7)]],
    device const MRSceneQueryShapeStateGPU* projectedShapes
        [[buffer(8)]],
    device const uint* parentBodies [[buffer(9)]],
    device const float4* localOrigins [[buffer(10)]],
    device const float4* localDirections [[buffer(11)]],
    device const float* maximumDistances [[buffer(12)]],
    device const uint4* options [[buffer(13)]],
    device float* distances [[buffer(14)]],
    device float4* points [[buffer(15)]],
    device float4* normals [[buffer(16)]],
    device uint4* identities [[buffer(17)]],
    device uint* validity [[buffer(18)]],
    const uint index [[thread_position_in_grid]]
) {
    const uint queryCount =
        dispatch.environmentCount * dispatch.rayCount;
    if (dispatch.abiVersion != MR_SCENE_QUERY_ABI_VERSION ||
        dispatch.rayCount == 0u ||
        dispatch.rayStride < dispatch.rayCount ||
        dispatch.bodyStride < dispatch.bodyCount ||
        index >= queryCount) {
        return;
    }
    clearRayResult(
        distances,
        points,
        normals,
        identities,
        validity,
        index
    );

    const uint environment = index / dispatch.rayCount;
    const uint localRay =
        index - environment * dispatch.rayCount;
    const uint parentBody = parentBodies[localRay];
    float3 origin = localOrigins[localRay].xyz;
    float3 rawDirection = localDirections[localRay].xyz;
    if (parentBody != MR_INVALID_INDEX) {
        if (parentBody >= dispatch.bodyCount) {
            return;
        }
        const MRBodyStateGPU parent =
            bodyStates[
                environment * dispatch.bodyStride + parentBody
            ];
        float4 parentRotation;
        if (!finite4(parent.position) ||
            !normalizedQuaternion(
                parent.orientation,
                parentRotation
            )) {
            return;
        }
        origin =
            parent.position.xyz +
            quaternionRotate(parentRotation, origin);
        rawDirection = quaternionRotate(
            parentRotation,
            rawDirection
        );
    }

    float3 direction;
    RayHit hit;
    if (!traceSceneRay(
            dispatch,
            shapes,
            geometryHeaders,
            geometryVertices,
            meshNodes,
            meshTriangles,
            convexFaces,
            bodyStates,
            projectedShapes,
            origin,
            rawDirection,
            maximumDistances[index],
            options[index],
            environment,
            direction,
            hit
        )) {
        return;
    }
    writeRayHit(
        distances,
        points,
        normals,
        identities,
        validity,
        index,
        origin,
        direction,
        hit
    );
}
