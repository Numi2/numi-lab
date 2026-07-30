#include <metal_stdlib>

#include "metalrobo/tactile_types.h"

using namespace metal;

namespace {

constant float kTiny = 1.0e-12f;
constant float kFloatMaximum = 3.402823466e38f;
constant bool kDebugHitsEnabled [[function_constant(0)]];

struct TactilePose {
    float3 position;
    float4 orientation;
};

inline float4 quaternionMultiply(
    const float4 a,
    const float4 b
) {
    return float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - dot(a.xyz, b.xyz)
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

inline float4 quaternionConjugate(const float4 value) {
    return float4(-value.xyz, value.w);
}

inline TactilePose composePose(
    const TactilePose parent,
    const TactilePose local
) {
    TactilePose result;
    result.position =
        parent.position +
        quaternionRotate(parent.orientation, local.position);
    result.orientation =
        quaternionMultiply(parent.orientation, local.orientation);
    return result;
}

inline TactilePose bodyPose(const MRBodyStateGPU body) {
    TactilePose result;
    result.position = body.position.xyz;
    result.orientation = body.orientation;
    return result;
}

inline TactilePose sensorPose(
    const MRTactileSensorGPU sensor,
    const MRBodyStateGPU body
) {
    TactilePose local;
    local.position = sensor.localPositionAndQueryEpsilon.xyz;
    local.orientation = sensor.localOrientation;
    return composePose(bodyPose(body), local);
}

inline TactilePose shapePose(
    const MRShapeGPU shape,
    const MRBodyStateGPU body
) {
    TactilePose local;
    local.position = shape.localPosition.xyz;
    local.orientation = shape.localRotation;
    return composePose(bodyPose(body), local);
}

inline float3 inverseTransformPoint(
    const TactilePose pose,
    const float3 point
) {
    return quaternionRotate(
        quaternionConjugate(pose.orientation),
        point - pose.position
    );
}

inline float3 inverseTransformVector(
    const TactilePose pose,
    const float3 direction
) {
    return quaternionRotate(
        quaternionConjugate(pose.orientation),
        direction
    );
}

inline float3 bodyPointVelocity(
    const MRBodyStateGPU body,
    const float3 worldPoint
) {
    return
        body.linearVelocityAndInverseMass.xyz +
        cross(
            body.angularVelocity.xyz,
            worldPoint - body.position.xyz
        );
}

inline bool validDispatch(
    constant const MRTactileDispatchGPU& dispatch
) {
    return
        dispatch.frameAndAbi.z == MR_TACTILE_ABI_VERSION &&
        dispatch.counts.x > 0u &&
        dispatch.counts.y > 0u &&
        dispatch.counts.z > 0u &&
        dispatch.counts.w > 0u &&
        isfinite(dispatch.timing.x) &&
        dispatch.timing.x > 0.0f &&
        isfinite(dispatch.timing.y) &&
        dispatch.timing.y > 0.0f &&
        isfinite(dispatch.timing.z) &&
        isfinite(dispatch.timing.w) &&
        dispatch.timing.w > 0.0f;
}

inline bool debugHitsEnabled() {
    return kDebugHitsEnabled;
}

inline bool sphereExit(
    const float3 origin,
    const float3 direction,
    const float radius,
    const float tolerance,
    thread float& result
) {
    const float squared = dot(origin, origin);
    if (squared > radius * radius + tolerance) {
        return false;
    }
    const float b = dot(origin, direction);
    const float discriminant =
        b * b - (squared - radius * radius);
    if (discriminant < -tolerance) {
        return false;
    }
    result = max(
        0.0f,
        -b + sqrt(max(0.0f, discriminant))
    );
    return isfinite(result);
}

inline bool boxExit(
    const float3 origin,
    const float3 direction,
    const float3 halfExtent,
    const float tolerance,
    thread float& result
) {
    if (any(abs(origin) > halfExtent + tolerance)) {
        return false;
    }
    float exit = kFloatMaximum;
    for (uint axis = 0u; axis < 3u; ++axis) {
        if (abs(direction[axis]) <= kTiny) {
            continue;
        }
        const float boundary =
            direction[axis] > 0.0f
            ? halfExtent[axis]
            : -halfExtent[axis];
        const float candidate =
            (boundary - origin[axis]) / direction[axis];
        if (candidate >= -tolerance) {
            exit = min(exit, max(0.0f, candidate));
        }
    }
    result = exit;
    return exit < kFloatMaximum && isfinite(exit);
}

inline bool cylinderExit(
    const float3 origin,
    const float3 direction,
    const float radius,
    const float halfLength,
    const float tolerance,
    thread float& result
) {
    const float radialSquared =
        origin.x * origin.x + origin.z * origin.z;
    if (radialSquared > radius * radius + tolerance ||
        abs(origin.y) > halfLength + tolerance) {
        return false;
    }
    float exit = kFloatMaximum;
    if (abs(direction.y) > kTiny) {
        const float cap =
            direction.y > 0.0f ? halfLength : -halfLength;
        const float candidate =
            (cap - origin.y) / direction.y;
        if (candidate >= -tolerance) {
            exit = min(exit, max(0.0f, candidate));
        }
    }
    const float a =
        direction.x * direction.x +
        direction.z * direction.z;
    if (a > kTiny) {
        const float b =
            origin.x * direction.x +
            origin.z * direction.z;
        const float c = radialSquared - radius * radius;
        const float discriminant = b * b - a * c;
        if (discriminant >= -tolerance) {
            const float candidate =
                (-b + sqrt(max(0.0f, discriminant))) / a;
            if (candidate >= -tolerance) {
                exit = min(exit, max(0.0f, candidate));
            }
        }
    }
    result = exit;
    return exit < kFloatMaximum && isfinite(exit);
}

inline float capsuleSignedDistance(
    const float3 point,
    const float radius,
    const float halfLength
) {
    const float y = clamp(point.y, -halfLength, halfLength);
    return length(point - float3(0.0f, y, 0.0f)) - radius;
}

inline bool capsuleExit(
    const float3 origin,
    const float3 direction,
    const float radius,
    const float halfLength,
    const float maximumDepth,
    const float tolerance,
    thread float& result
) {
    if (capsuleSignedDistance(origin, radius, halfLength) >
        tolerance) {
        return false;
    }
    if (capsuleSignedDistance(
            origin + direction * maximumDepth,
            radius,
            halfLength
        ) <= 0.0f) {
        result = maximumDepth;
        return true;
    }
    float lower = 0.0f;
    float upper = maximumDepth;
    for (uint iteration = 0u; iteration < 28u; ++iteration) {
        const float middle = 0.5f * (lower + upper);
        if (capsuleSignedDistance(
                origin + direction * middle,
                radius,
                halfLength
            ) <= 0.0f) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    result = 0.5f * (lower + upper);
    return isfinite(result);
}

inline bool convexExit(
    const float3 origin,
    const float3 direction,
    const MRShapeGPU shape,
    constant const MRTactileDispatchGPU& dispatch,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const MRConvexFaceGPU* convexFaces,
    const float tolerance,
    thread float& result
) {
    if (shape.geometryOffset >= dispatch.geometryCounts.y ||
        any(shape.dimensions.xyz <= 0.0f)) {
        return false;
    }
    const MRGeometryHeaderGPU geometry =
        geometryHeaders[shape.geometryOffset];
    if (geometry.kind != MR_GEOMETRY_CONVEX ||
        geometry.faceOffset > dispatch.geometryCounts.w ||
        geometry.faceCount >
            dispatch.geometryCounts.w - geometry.faceOffset) {
        return false;
    }
    const float3 scaledOrigin = origin / shape.dimensions.xyz;
    const float3 scaledDirection =
        direction / shape.dimensions.xyz;
    float exit = kFloatMaximum;
    for (uint faceIndex = 0u;
         faceIndex < geometry.faceCount;
         ++faceIndex) {
        const MRConvexFaceGPU face =
            convexFaces[geometry.faceOffset + faceIndex];
        const float signedDistance =
            dot(face.plane.xyz, scaledOrigin) - face.plane.w;
        if (signedDistance > tolerance) {
            return false;
        }
        const float denominator =
            dot(face.plane.xyz, scaledDirection);
        if (denominator > kTiny) {
            const float candidate =
                -signedDistance / denominator;
            if (candidate >= -tolerance) {
                exit = min(exit, max(0.0f, candidate));
            }
        }
    }
    result = exit;
    return exit < kFloatMaximum && isfinite(exit);
}

inline bool rayTriangle(
    const float3 origin,
    const float3 direction,
    const float3 a,
    const float3 b,
    const float3 c,
    thread float& parameter
) {
    const float3 edge1 = b - a;
    const float3 edge2 = c - a;
    const float3 p = cross(direction, edge2);
    const float determinant = dot(edge1, p);
    const float scale =
        max(max(length(edge1), length(edge2)), 1.0f);
    const float epsilon =
        32.0f * 1.1920929e-7f * scale;
    if (abs(determinant) <= epsilon) {
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
    if (candidate <= epsilon) {
        return false;
    }
    parameter = candidate;
    return true;
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
    const float3 upper
) {
    float nearValue = 0.0f;
    float farValue = kFloatMaximum;
    for (uint axis = 0u; axis < 3u; ++axis) {
        if (abs(direction[axis]) <= kTiny) {
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
    return farValue >= 0.0f;
}

inline uint uint4Component(const uint4 value, const uint index) {
    return value[index];
}

inline uint meshIntersectionCount(
    const float3 origin,
    const float3 direction,
    const MRGeometryHeaderGPU geometry,
    constant const MRTactileDispatchGPU& dispatch,
    device const float4* geometryVertices,
    device const MRMeshBVHNodeGPU* meshNodes,
    device const MRMeshTriangleGPU* meshTriangles,
    thread float& firstHit
) {
    uint intersections = 0u;
    uint cursor = 0u;
    uint visits = 0u;
    firstHit = kFloatMaximum;
    const uint maximumVisits =
        geometry.bvhCount * MR_MESH_BVH_BRANCHING;
    while (cursor != MR_MESH_BVH_INVALID_ESCAPE &&
           visits++ < maximumVisits) {
        const uint nodeIndex =
            cursor / MR_MESH_BVH_BRANCHING;
        const uint slot =
            cursor - nodeIndex * MR_MESH_BVH_BRANCHING;
        if (nodeIndex >= geometry.bvhCount ||
            geometry.bvhOffset + nodeIndex >=
                dispatch.queryCounts.x) {
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
            !rayAabb(
                origin,
                direction,
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
        for (uint local = 0u; local < triangleCount; ++local) {
            const uint triangleIndex =
                geometry.triangleOffset + child + local;
            if (triangleIndex >= dispatch.queryCounts.y) {
                continue;
            }
            const MRMeshTriangleGPU triangle =
                meshTriangles[triangleIndex];
            if (triangle.verticesAndFeature.x >=
                    dispatch.geometryCounts.z ||
                triangle.verticesAndFeature.y >=
                    dispatch.geometryCounts.z ||
                triangle.verticesAndFeature.z >=
                    dispatch.geometryCounts.z) {
                continue;
            }
            float parameter = 0.0f;
            if (rayTriangle(
                    origin,
                    direction,
                    geometryVertices[
                        triangle.verticesAndFeature.x
                    ].xyz,
                    geometryVertices[
                        triangle.verticesAndFeature.y
                    ].xyz,
                    geometryVertices[
                        triangle.verticesAndFeature.z
                    ].xyz,
                    parameter
                )) {
                ++intersections;
                firstHit = min(firstHit, parameter);
            }
        }
        cursor = escape;
    }
    return intersections;
}

inline bool meshExit(
    const float3 origin,
    const float3 direction,
    const MRShapeGPU shape,
    constant const MRTactileDispatchGPU& dispatch,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    device const MRMeshBVHNodeGPU* meshNodes,
    device const MRMeshTriangleGPU* meshTriangles,
    thread float& result
) {
    if (shape.geometryOffset >= dispatch.geometryCounts.y ||
        any(shape.dimensions.xyz <= 0.0f)) {
        return false;
    }
    const MRGeometryHeaderGPU geometry =
        geometryHeaders[shape.geometryOffset];
    if (geometry.kind != MR_GEOMETRY_TRIANGLE_MESH ||
        (geometry.flags & MR_GEOMETRY_FLAG_CLOSED) == 0u ||
        geometry.bvhCount == 0u) {
        return false;
    }
    const float3 localOrigin = origin / shape.dimensions.xyz;
    const float3 localDirection =
        direction / shape.dimensions.xyz;
    const float3 parityDirection =
        normalize(float3(1.0f, 0.17320508f, 0.071f));
    float ignored = kFloatMaximum;
    const uint parity = meshIntersectionCount(
        localOrigin,
        parityDirection,
        geometry,
        dispatch,
        geometryVertices,
        meshNodes,
        meshTriangles,
        ignored
    );
    if ((parity & 1u) == 0u) {
        return false;
    }
    float exit = kFloatMaximum;
    (void)meshIntersectionCount(
        localOrigin,
        localDirection,
        geometry,
        dispatch,
        geometryVertices,
        meshNodes,
        meshTriangles,
        exit
    );
    result = exit;
    return exit < kFloatMaximum && isfinite(exit);
}

inline bool shapeExit(
    const float3 worldOrigin,
    const float3 worldDirection,
    const float maximumDepth,
    const float queryEpsilon,
    const MRShapeGPU shape,
    const MRBodyStateGPU body,
    constant const MRTactileDispatchGPU& dispatch,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    device const MRConvexFaceGPU* convexFaces,
    device const MRMeshBVHNodeGPU* meshNodes,
    device const MRMeshTriangleGPU* meshTriangles,
    thread float& result
) {
    const TactilePose pose = shapePose(shape, body);
    const float3 origin = inverseTransformPoint(pose, worldOrigin);
    const float3 direction =
        inverseTransformVector(pose, worldDirection);
    const float tolerance = max(
        queryEpsilon,
        32.0f * 1.1920929e-7f * (maximumDepth + 1.0f)
    );
    switch (shape.shapeType) {
    case MR_SHAPE_SPHERE:
        return sphereExit(
            origin,
            direction,
            shape.dimensions.x,
            tolerance,
            result
        );
    case MR_SHAPE_BOX:
        return boxExit(
            origin,
            direction,
            shape.dimensions.xyz,
            tolerance,
            result
        );
    case MR_SHAPE_CYLINDER:
        return cylinderExit(
            origin,
            direction,
            shape.dimensions.x,
            shape.dimensions.y,
            tolerance,
            result
        );
    case MR_SHAPE_CAPSULE:
        return capsuleExit(
            origin,
            direction,
            shape.dimensions.x,
            shape.dimensions.y,
            maximumDepth,
            tolerance,
            result
        );
    case MR_SHAPE_CONVEX:
        return convexExit(
            origin,
            direction,
            shape,
            dispatch,
            geometryHeaders,
            convexFaces,
            tolerance,
            result
        );
    case MR_SHAPE_TRIANGLE_MESH:
        return meshExit(
            origin,
            direction,
            shape,
            dispatch,
            geometryHeaders,
            geometryVertices,
            meshNodes,
            meshTriangles,
            result
        );
    default:
        return false;
    }
}

} // namespace

kernel void mr_tactile_sample(
    constant const MRTactileDispatchGPU& dispatch [[buffer(0)]],
    device const MRTactileSensorGPU* sensors [[buffer(1)]],
    device const MRTactileSampleGPU* samples [[buffer(2)]],
    device const uint* targetShapeIndices [[buffer(3)]],
    device const MRShapeGPU* shapes [[buffer(4)]],
    device const MRGeometryHeaderGPU* geometryHeaders [[buffer(5)]],
    device const float4* geometryVertices [[buffer(6)]],
    device const MRConvexFaceGPU* convexFaces [[buffer(7)]],
    device const MRMeshBVHNodeGPU* meshNodes [[buffer(8)]],
    device const MRMeshTriangleGPU* meshTriangles [[buffer(9)]],
    device const MRBodyStateGPU* bodies [[buffer(10)]],
    device const uint* resetMask [[buffer(11)]],
    device const float* previousDepth [[buffer(12)]],
    device const uint* previousValidity [[buffer(13)]],
    device const uint* previousObject [[buffer(14)]],
    device const MRTactileHitGPU* previousHits [[buffer(15)]],
    device const MRTactileTangentialMotionGPU*
        previousTangentialMotion [[buffer(16)]],
    device const float4* previousTargetLocalAnchor [[buffer(17)]],
    device float* depth [[buffer(18)]],
    device float* depthVelocity [[buffer(19)]],
    device MRTactileTangentialMotionGPU*
        tangentialMotion [[buffer(20)]],
    device float4* targetLocalAnchor [[buffer(21)]],
    device uint* validity [[buffer(22)]],
    device uint* objectShape [[buffer(23)]],
    device MRTactileHitGPU* hits [[buffer(24)]],
    device const ulong* frameIndices [[buffer(25)]],
    const uint threadIndex [[thread_position_in_grid]]
) {
    if (!validDispatch(dispatch)) {
        return;
    }
    const uint total =
        dispatch.counts.x * dispatch.counts.w;
    if (threadIndex >= total) {
        return;
    }
    const uint environment =
        threadIndex / dispatch.counts.w;
    const uint sampleIndex =
        threadIndex - environment * dispatch.counts.w;
    const MRTactileSampleGPU sample = samples[sampleIndex];
    const uint sensorIndex = sample.atlasAndIdentity.z;
    if (sensorIndex >= dispatch.counts.z) {
        depth[threadIndex] = 0.0f;
        depthVelocity[threadIndex] = 0.0f;
        tangentialMotion[threadIndex] = {};
        targetLocalAnchor[threadIndex] = float4(0.0f);
        validity[threadIndex] = 0u;
        objectShape[threadIndex] = MR_INVALID_INDEX;
        if (debugHitsEnabled()) {
            hits[threadIndex] = {};
        }
        return;
    }
    const MRTactileSensorGPU sensor = sensors[sensorIndex];
    if ((sample.atlasAndIdentity.w &
         MR_TACTILE_SAMPLE_VALID) == 0u) {
        depth[threadIndex] = 0.0f;
        depthVelocity[threadIndex] = 0.0f;
        tangentialMotion[threadIndex] = {};
        targetLocalAnchor[threadIndex] = float4(0.0f);
        validity[threadIndex] = 0u;
        objectShape[threadIndex] = MR_INVALID_INDEX;
        if (debugHitsEnabled()) {
            hits[threadIndex] = {};
        }
        return;
    }
    const ulong frameIndex = frameIndices[environment];
    const bool update =
        frameIndex %
            static_cast<ulong>(sensor.scheduleAndIdentity.x) == 0u;
    if (!update) {
        depth[threadIndex] = previousDepth[threadIndex];
        depthVelocity[threadIndex] = 0.0f;
        tangentialMotion[threadIndex].displacementAndVelocity =
            float4(
                previousTangentialMotion[
                    threadIndex
                ].displacementAndVelocity.xy,
                0.0f,
                0.0f
            );
        targetLocalAnchor[threadIndex] =
            previousTargetLocalAnchor[threadIndex];
        validity[threadIndex] = previousValidity[threadIndex];
        objectShape[threadIndex] = previousObject[threadIndex];
        if (debugHitsEnabled()) {
            hits[threadIndex] = previousHits[threadIndex];
        }
        return;
    }

    const uint bodyBase = environment * dispatch.counts.y;
    const TactilePose worldSensor = sensorPose(
        sensor,
        bodies[bodyBase + sensor.topology.x]
    );
    const float3 worldOrigin =
        worldSensor.position +
        quaternionRotate(
            worldSensor.orientation,
            sample.localPositionAndArea.xyz
        );
    const float3 worldNormal = normalize(
        quaternionRotate(
            worldSensor.orientation,
            sample.localNormalAndMaximumDepth.xyz
        )
    );
    const float3 worldDirection = -worldNormal;
    const float maximumDepth = min(
        sensor.depth.x,
        sample.localNormalAndMaximumDepth.w
    );
    float bestDepth = 0.0f;
    uint bestShape = MR_INVALID_INDEX;
    for (uint localTarget = 0u;
         localTarget < sensor.atlasAndTargets.w;
         ++localTarget) {
        const uint targetOffset =
            sensor.atlasAndTargets.z + localTarget;
        if (targetOffset >= dispatch.queryCounts.z) {
            continue;
        }
        const uint shapeIndex =
            targetShapeIndices[targetOffset];
        if (shapeIndex >= dispatch.geometryCounts.x) {
            continue;
        }
        const MRShapeGPU shape = shapes[shapeIndex];
        if (shape.bodyIndex >= dispatch.counts.y) {
            continue;
        }
        float exit = 0.0f;
        if (!shapeExit(
                worldOrigin,
                worldDirection,
                maximumDepth,
                sensor.localPositionAndQueryEpsilon.w,
                shape,
                bodies[bodyBase + shape.bodyIndex],
                dispatch,
                geometryHeaders,
                geometryVertices,
                convexFaces,
                meshNodes,
                meshTriangles,
                exit
            ) ||
            exit <= sensor.depth.y) {
            continue;
        }
        const float candidate =
            clamp(exit, 0.0f, maximumDepth);
        if (candidate > bestDepth ||
            (candidate == bestDepth &&
             shapeIndex < bestShape)) {
            bestDepth = candidate;
            bestShape = shapeIndex;
        }
    }

    uint outputValidity = MR_TACTILE_VALIDITY_SAMPLE;
    MRTactileHitGPU hit = {};
    MRTactileTangentialMotionGPU motion = {};
    float4 anchor = float4(0.0f);
    if (bestShape != MR_INVALID_INDEX) {
        outputValidity |=
            MR_TACTILE_VALIDITY_CONTACT |
            MR_TACTILE_VALIDITY_FILTERED_TARGET;
        if (bestDepth >= maximumDepth -
                max(
                    sensor.localPositionAndQueryEpsilon.w,
                    1.0e-8f
                )) {
            outputValidity |= MR_TACTILE_VALIDITY_SATURATED;
        }
        hit.worldPointAndDepth = float4(
            worldOrigin + worldDirection * bestDepth,
            bestDepth
        );
        hit.worldNormalAndRayParameter =
            float4(worldNormal, bestDepth);
        hit.identityAndFlags = uint4(
            bestShape,
            sampleIndex - sensor.topology.z,
            outputValidity,
            0u
        );

        const MRShapeGPU targetShape = shapes[bestShape];
        const MRBodyStateGPU targetBody =
            bodies[bodyBase + targetShape.bodyIndex];
        const float3 worldHit = hit.worldPointAndDepth.xyz;
        const TactilePose targetBodyPose = bodyPose(targetBody);
        const bool reset = resetMask[environment] != 0u;
        const bool continuingContact =
            !reset &&
            (previousValidity[threadIndex] &
             MR_TACTILE_VALIDITY_CONTACT) != 0u &&
            previousObject[threadIndex] == bestShape;
        const float3 anchorLocal =
            continuingContact
            ? previousTargetLocalAnchor[threadIndex].xyz
            : inverseTransformPoint(targetBodyPose, worldHit);
        anchor = float4(anchorLocal, 0.0f);

        float2 displacement = float2(0.0f);
        if (continuingContact) {
            const float3 anchorWorld =
                targetBodyPose.position +
                quaternionRotate(
                    targetBodyPose.orientation,
                    anchorLocal
                );
            const float3 relativeLocal =
                inverseTransformPoint(worldSensor, anchorWorld) -
                sample.localPositionAndArea.xyz;
            displacement = float2(
                dot(relativeLocal, sample.localTangentU.xyz),
                dot(relativeLocal, sample.localTangentV.xyz)
            );
            const float magnitude = length(displacement);
            if (magnitude > sensor.depth.w) {
                displacement *= sensor.depth.w / magnitude;
            }
        }

        const MRBodyStateGPU sensorBody =
            bodies[bodyBase + sensor.topology.x];
        const float3 relativeVelocity =
            bodyPointVelocity(targetBody, worldHit) -
            bodyPointVelocity(sensorBody, worldHit);
        const float3 worldTangentU = quaternionRotate(
            worldSensor.orientation,
            sample.localTangentU.xyz
        );
        const float3 worldTangentV = quaternionRotate(
            worldSensor.orientation,
            sample.localTangentV.xyz
        );
        motion.displacementAndVelocity = float4(
            displacement,
            dot(relativeVelocity, worldTangentU),
            dot(relativeVelocity, worldTangentV)
        );
    }
    depth[threadIndex] = bestDepth;
    tangentialMotion[threadIndex] = motion;
    targetLocalAnchor[threadIndex] = anchor;
    validity[threadIndex] = outputValidity;
    objectShape[threadIndex] = bestShape;
    if (debugHitsEnabled()) {
        hits[threadIndex] = hit;
    }
    const bool reset = resetMask[environment] != 0u;
    depthVelocity[threadIndex] =
        reset
        ? 0.0f
        : (bestDepth - previousDepth[threadIndex]) *
            dispatch.timing.y /
            static_cast<float>(sensor.scheduleAndIdentity.x);
}

kernel void mr_tactile_reduce(
    constant const MRTactileDispatchGPU& dispatch [[buffer(0)]],
    device const MRTactileSensorGPU* sensors [[buffer(1)]],
    device const MRTactileSampleGPU* samples [[buffer(2)]],
    device const MRBodyStateGPU* bodies [[buffer(3)]],
    device const MRTactileContactGPU* contacts [[buffer(4)]],
    device const uint* contactCounts [[buffer(5)]],
    device const uint* resetMask [[buffer(6)]],
    device const float* depth [[buffer(7)]],
    device const uint* validity [[buffer(8)]],
    device const uint* objectShape [[buffer(9)]],
    device const MRTactileTangentialMotionGPU*
        tangentialMotion [[buffer(10)]],
    device MRTactileSummaryGPU* summaries [[buffer(11)]],
    device MRTactileStatusGPU* statuses [[buffer(12)]],
    device const ulong* frameIndices [[buffer(13)]],
    const uint threadIndex [[thread_position_in_grid]]
) {
    const uint total =
        dispatch.counts.x * dispatch.counts.z;
    if (threadIndex >= total || !validDispatch(dispatch)) {
        return;
    }
    const uint environment =
        threadIndex / dispatch.counts.z;
    const uint sensorIndex =
        threadIndex - environment * dispatch.counts.z;
    const MRTactileSensorGPU sensor = sensors[sensorIndex];
    const uint bodyBase = environment * dispatch.counts.y;
    const TactilePose worldSensor = sensorPose(
        sensor,
        bodies[bodyBase + sensor.topology.x]
    );

    float3 centroidLocal = float3(0.0f);
    float3 centroidWorld = float3(0.0f);
    float activeArea = 0.0f;
    float areaDepth = 0.0f;
    float maximumDepth = 0.0f;
    float areaTangentialSpeedSquared = 0.0f;
    float maximumTangentialDisplacement = 0.0f;
    uint activeCount = 0u;
    uint saturatedCount = 0u;
    uint unanimousObject = MR_INVALID_INDEX;
    bool objectDisagrees = false;
    for (uint localSample = 0u;
         localSample < sensor.topology.w;
         ++localSample) {
        const uint sampleIndex =
            sensor.topology.z + localSample;
        const uint denseIndex =
            environment * dispatch.counts.w + sampleIndex;
        const float sampleDepth = depth[denseIndex];
        if (sampleDepth <= sensor.depth.y ||
            (validity[denseIndex] &
             MR_TACTILE_VALIDITY_CONTACT) == 0u) {
            continue;
        }
        const MRTactileSampleGPU sample = samples[sampleIndex];
        const float area = sample.localPositionAndArea.w;
        const float3 localPoint =
            sample.localPositionAndArea.xyz -
            sample.localNormalAndMaximumDepth.xyz *
                sampleDepth;
        const float3 worldPoint =
            worldSensor.position +
            quaternionRotate(
                worldSensor.orientation,
                localPoint
            );
        centroidLocal += localPoint * area;
        centroidWorld += worldPoint * area;
        activeArea += area;
        areaDepth += area * sampleDepth;
        maximumDepth = max(maximumDepth, sampleDepth);
        const float4 motion =
            tangentialMotion[denseIndex]
                .displacementAndVelocity;
        areaTangentialSpeedSquared +=
            area * dot(motion.zw, motion.zw);
        maximumTangentialDisplacement = max(
            maximumTangentialDisplacement,
            length(motion.xy)
        );
        ++activeCount;
        if ((validity[denseIndex] &
             MR_TACTILE_VALIDITY_SATURATED) != 0u) {
            ++saturatedCount;
        }
        const uint object = objectShape[denseIndex];
        if (unanimousObject == MR_INVALID_INDEX) {
            unanimousObject = object;
        } else if (unanimousObject != object) {
            objectDisagrees = true;
        }
    }
    if (activeArea > 0.0f) {
        centroidLocal /= activeArea;
        centroidWorld /= activeArea;
    }

    float3 force = float3(0.0f);
    float3 torque = float3(0.0f);
    float3 centerOfPressureWorld = float3(0.0f);
    float centerOfPressureForceWeight = 0.0f;
    uint contactContributors = 0u;
    uint centerOfPressureContributors = 0u;
    float weightedFrictionUtilization = 0.0f;
    float frictionUtilizationWeight = 0.0f;
    float maximumFrictionUtilization = 0.0f;
    const uint contactCount = min(
        contactCounts[environment],
        dispatch.queryCounts.w
    );
    const uint contactBase =
        environment * dispatch.queryCounts.w;
    for (uint contactIndex = 0u;
         contactIndex < contactCount;
         ++contactIndex) {
        const MRTactileContactGPU contact =
            contacts[contactBase + contactIndex];
        float3 impulse = float3(0.0f);
        if (contact.shapesAndFlags.x == sensor.topology.y) {
            impulse = contact.worldImpulseOnA.xyz;
        } else if (
            contact.shapesAndFlags.y == sensor.topology.y
        ) {
            impulse = -contact.worldImpulseOnA.xyz;
        } else {
            continue;
        }
        const float3 contactForce =
            impulse * dispatch.timing.w;
        force += contactForce;
        torque += cross(
            contact.worldPoint.xyz - worldSensor.position,
            contactForce
        );
        const float forceWeight = length(contactForce);
        if (forceWeight > kTiny) {
            centerOfPressureWorld +=
                contact.worldPoint.xyz * forceWeight;
            centerOfPressureForceWeight += forceWeight;
            ++centerOfPressureContributors;
        }
        const float normalImpulse =
            contact.solverImpulseAndFriction.x;
        const float tangentialImpulse =
            contact.solverImpulseAndFriction.y;
        const float staticFriction =
            contact.solverImpulseAndFriction.z;
        float utilization = 0.0f;
        if (normalImpulse > kTiny) {
            const float capacity =
                staticFriction * normalImpulse;
            utilization = capacity > kTiny
                ? clamp(
                    tangentialImpulse / capacity,
                    0.0f,
                    1.0f
                )
                : (tangentialImpulse > kTiny ? 1.0f : 0.0f);
        }
        const float normalForceWeight =
            normalImpulse * dispatch.timing.w;
        if (normalForceWeight > kTiny) {
            weightedFrictionUtilization +=
                normalForceWeight * utilization;
            frictionUtilizationWeight += normalForceWeight;
        }
        maximumFrictionUtilization = max(
            maximumFrictionUtilization,
            utilization
        );
        ++contactContributors;
    }
    if (centerOfPressureForceWeight > 0.0f) {
        centerOfPressureWorld /=
            centerOfPressureForceWeight;
    }
    const float3 centerOfPressureLocal =
        centerOfPressureForceWeight > 0.0f
        ? inverseTransformPoint(
            worldSensor,
            centerOfPressureWorld
        )
        : float3(0.0f);

    const ulong frameIndex = frameIndices[environment];
    const bool update =
        frameIndex %
            static_cast<ulong>(sensor.scheduleAndIdentity.x) == 0u;
    const bool reset = resetMask[environment] != 0u;
    const uint summaryFlags =
        MR_TACTILE_SUMMARY_DEPTH_VALID |
        (dispatch.queryCounts.w > 0u
            ? MR_TACTILE_SUMMARY_WRENCH_VALID
            : 0u) |
        (update ? MR_TACTILE_SUMMARY_UPDATED : 0u) |
        (reset ? MR_TACTILE_SUMMARY_RESET : 0u);
    MRTactileSummaryGPU summary = {};
    summary.posePositionAndTimestamp =
        float4(worldSensor.position, dispatch.timing.z);
    summary.poseOrientation = worldSensor.orientation;
    summary.netForceAndContactArea =
        float4(force, activeArea);
    summary.netTorqueAndMaximumDepth =
        float4(torque, maximumDepth);
    summary.centroidLocalAndMeanDepth = float4(
        centroidLocal,
        activeArea > 0.0f ? areaDepth / activeArea : 0.0f
    );
    summary.centroidWorldAndActiveCount =
        float4(centroidWorld, static_cast<float>(activeCount));
    summary.centerOfPressureLocalAndForceWeight = float4(
        centerOfPressureLocal,
        centerOfPressureForceWeight
    );
    summary.centerOfPressureWorldAndContactCount = float4(
        centerOfPressureWorld,
        static_cast<float>(centerOfPressureContributors)
    );
    summary.tangentialMotionAndFriction = float4(
        activeArea > 0.0f
            ? sqrt(areaTangentialSpeedSquared / activeArea)
            : 0.0f,
        maximumTangentialDisplacement,
        frictionUtilizationWeight > 0.0f
            ? weightedFrictionUtilization /
                frictionUtilizationWeight
            : 0.0f,
        maximumFrictionUtilization
    );
    summary.statisticsAndIdentity = uint4(
        saturatedCount,
        contactContributors,
        summaryFlags,
        objectDisagrees ? MR_INVALID_INDEX : unanimousObject
    );
    summaries[threadIndex] = summary;
    statuses[threadIndex] = {
        MR_TACTILE_SUCCESS,
        environment,
        sensorIndex,
        MR_INVALID_INDEX,
        {
            MR_TACTILE_QUERY_METAL_ANALYTIC_BVH4,
            sensor.atlasAndTargets.w * sensor.topology.w,
            0u,
            0u,
        },
    };
}

kernel void mr_tactile_commit_history(
    constant const MRTactileDispatchGPU& dispatch [[buffer(0)]],
    device const float* depth [[buffer(1)]],
    device const uint* validity [[buffer(2)]],
    device const uint* objectShape [[buffer(3)]],
    device const MRTactileHitGPU* hits [[buffer(4)]],
    device const MRTactileTangentialMotionGPU*
        tangentialMotion [[buffer(5)]],
    device const float4* targetLocalAnchor [[buffer(6)]],
    device float* previousDepth [[buffer(7)]],
    device uint* previousValidity [[buffer(8)]],
    device uint* previousObject [[buffer(9)]],
    device MRTactileHitGPU* previousHits [[buffer(10)]],
    device MRTactileTangentialMotionGPU*
        previousTangentialMotion [[buffer(11)]],
    device float4* previousTargetLocalAnchor [[buffer(12)]],
    const uint threadIndex [[thread_position_in_grid]]
) {
    const uint total =
        dispatch.counts.x * dispatch.counts.w;
    if (threadIndex >= total || !validDispatch(dispatch)) {
        return;
    }
    previousDepth[threadIndex] = depth[threadIndex];
    previousValidity[threadIndex] = validity[threadIndex];
    previousObject[threadIndex] = objectShape[threadIndex];
    previousTangentialMotion[threadIndex] =
        tangentialMotion[threadIndex];
    previousTargetLocalAnchor[threadIndex] =
        targetLocalAnchor[threadIndex];
    if (debugHitsEnabled()) {
        previousHits[threadIndex] = hits[threadIndex];
    }
}
