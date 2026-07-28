#include <metal_stdlib>

#include "metalrobo/engine_types.h"

using namespace metal;

namespace {

constant float kTiny = 1.0e-14f;
constant uint kPairSphereSphere = 1u;
constant uint kPairSpherePlane = 2u;
constant uint kPairCapsulePlane = 3u;
constant uint kPairBoxPlane = 4u;
constant uint kPairCylinderPlane = 5u;

bool finite4(const float4 value) {
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

bool checkedQuaternion(
    const float4 input,
    thread float4& output
) {
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
    return finite4(output);
}

float4 quaternionMultiply(
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

float3 quaternionRotate(
    const float4 quaternion,
    const float3 value
) {
    const float3 vector = quaternion.xyz;
    const float3 twiceCross = 2.0f * cross(vector, value);
    return value + quaternion.w * twiceCross +
        cross(vector, twiceCross);
}

uint pairClass(const uint typeA, const uint typeB) {
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
    return 0u;
}

bool supportedShapeType(const uint type) {
    return
        type == MR_SHAPE_SPHERE ||
        type == MR_SHAPE_CAPSULE ||
        type == MR_SHAPE_BOX ||
        type == MR_SHAPE_CYLINDER ||
        type == MR_SHAPE_PLANE;
}

bool validActiveDimensions(const MRShapeGPU shape) {
    if (shape.shapeType == MR_SHAPE_SPHERE) {
        return
            shape.dimensions.x >=
                MR_MIN_COLLISION_EXTENT;
    }
    if (shape.shapeType == MR_SHAPE_CAPSULE ||
        shape.shapeType == MR_SHAPE_CYLINDER) {
        return
            shape.dimensions.x >=
                MR_MIN_COLLISION_EXTENT &&
            shape.dimensions.y >=
                MR_MIN_COLLISION_EXTENT;
    }
    if (shape.shapeType == MR_SHAPE_BOX) {
        return all(
            shape.dimensions.xyz >=
                float3(MR_MIN_COLLISION_EXTENT)
        );
    }
    return shape.shapeType == MR_SHAPE_PLANE;
}

ulong rowStart(const uint row, const uint shapeCount) {
    return
        static_cast<ulong>(row) *
        static_cast<ulong>(2u * shapeCount - row - 1u) /
        2ul;
}

void decodeLogicalPair(
    const uint logicalIndex,
    const uint shapeCount,
    thread uint& colliderA,
    thread uint& colliderB
) {
    uint low = 0u;
    uint high = shapeCount - 1u;
    while (low + 1u < high) {
        const uint middle = low + (high - low) / 2u;
        if (rowStart(middle, shapeCount) <= logicalIndex) {
            low = middle;
        } else {
            high = middle;
        }
    }
    const ulong start = rowStart(low, shapeCount);
    colliderA = low;
    colliderB =
        low + 1u +
        static_cast<uint>(
            static_cast<ulong>(logicalIndex) - start
        );
}

bool aabbOverlap(
    device const MRAabbGPU& left,
    device const MRAabbGPU& right
) {
    return
        all(left.lower.xyz <= right.upper.xyz) &&
        all(left.upper.xyz >= right.lower.xyz);
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
        if (min(left, right) == colliderA &&
            max(left, right) == colliderB) {
            return true;
        }
    }
    return false;
}

bool planeMayOverlap(
    const uint planeIndex,
    const uint finiteIndex,
    device const MRShapeGPU* shapes,
    device const MRBodyStateGPU* bodies,
    device const MRAabbGPU* aabbs
) {
    const MRShapeGPU plane = shapes[planeIndex];
    const MRBodyStateGPU body = bodies[plane.bodyIndex];
    float4 bodyRotation;
    float4 localRotation;
    if (!checkedQuaternion(body.orientation, bodyRotation) ||
        !checkedQuaternion(plane.localRotation, localRotation)) {
        return false;
    }
    const float4 rotation =
        quaternionMultiply(bodyRotation, localRotation);
    float3 normal = quaternionRotate(
        rotation,
        float3(0.0f, 1.0f, 0.0f)
    );
    const float normalSquared = dot(normal, normal);
    if (!(normalSquared > kTiny) || !isfinite(normalSquared)) {
        return false;
    }
    normal *= rsqrt(normalSquared);
    const float3 planeCenter =
        body.position.xyz +
        quaternionRotate(bodyRotation, plane.localPosition.xyz);
    device const MRAabbGPU& finiteAabb = aabbs[finiteIndex];
    const float3 center =
        0.5f * (finiteAabb.lower.xyz + finiteAabb.upper.xyz);
    const float3 halfExtent =
        0.5f * (finiteAabb.upper.xyz - finiteAabb.lower.xyz);
    const float minimumSignedDistance =
        dot(normal, center - planeCenter) -
        dot(abs(normal), halfExtent);
    const float queryScale = max(
        maximumAbsoluteComponent(planeCenter),
        max(
            maximumAbsoluteComponent(finiteAabb.lower.xyz),
            maximumAbsoluteComponent(finiteAabb.upper.xyz)
        )
    );
    const float queryPadding =
        (queryScale + 1.0f) *
        MR_COLLISION_QUERY_RELATIVE_PAD;
    return
        isfinite(minimumSignedDistance) &&
        minimumSignedDistance <=
            plane.contactRestAndBoundingRadius.x +
            queryPadding;
}

bool pairPasses(
    const uint colliderA,
    const uint colliderB,
    device const MRShapeGPU* shapes,
    device const MRBodyStateGPU* bodies,
    device const MRAabbGPU* aabbs,
    device const MRCandidatePairGPU* exclusions,
    const uint exclusionCount,
    thread uint& outputClass
) {
    const MRShapeGPU shapeA = shapes[colliderA];
    const MRShapeGPU shapeB = shapes[colliderB];
    outputClass = pairClass(shapeA.shapeType, shapeB.shapeType);
    if (outputClass == 0u ||
        (shapeA.flags & MR_SHAPE_FLAG_SIMULATION_DISABLED) != 0u ||
        (shapeB.flags & MR_SHAPE_FLAG_SIMULATION_DISABLED) != 0u ||
        shapeA.bodyIndex == shapeB.bodyIndex ||
        (shapeA.collisionGroup & shapeB.collisionMask) == 0u ||
        (shapeB.collisionGroup & shapeA.collisionMask) == 0u) {
        return false;
    }
    const uint motionA =
        bodies[shapeA.bodyIndex].flagsAndIndices[0];
    const uint motionB =
        bodies[shapeB.bodyIndex].flagsAndIndices[0];
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

    if (shapeA.shapeType == MR_SHAPE_PLANE) {
        return planeMayOverlap(
            colliderA,
            colliderB,
            shapes,
            bodies,
            aabbs
        );
    }
    if (shapeB.shapeType == MR_SHAPE_PLANE) {
        return planeMayOverlap(
            colliderB,
            colliderA,
            shapes,
            bodies,
            aabbs
        );
    }
    return aabbOverlap(aabbs[colliderA], aabbs[colliderB]);
}

} // namespace

kernel void mr_broadphase_preflight(
    constant const MRBroadphaseDispatchGPU& dispatch [[buffer(0)]],
    device const MRShapeGPU* shapes [[buffer(1)]],
    device const MRBodyStateGPU* bodies [[buffer(2)]],
    device const MRAabbGPU* aabbs [[buffer(3)]],
    device const MRCandidatePairGPU* exclusions [[buffer(4)]],
    device MRBroadphaseStatusGPU& status [[buffer(5)]],
    const uint index [[thread_position_in_grid]]
) {
    if (index != 0u) {
        return;
    }
    status = {};
    status.code = MR_STEP_SUCCESS;
    const ulong logicalPairs =
        dispatch.shapeCount < 2u
        ? 0ul
        : static_cast<ulong>(dispatch.shapeCount) *
            static_cast<ulong>(dispatch.shapeCount - 1u) / 2ul;
    status.logicalPairs =
        logicalPairs <= 0xfffffffful
        ? static_cast<uint>(logicalPairs)
        : 0xffffffffu;
    const ulong expectedBlocks =
        (
            logicalPairs +
            static_cast<ulong>(MR_BROADPHASE_SCAN_BLOCK_SIZE) -
            1ul
        ) /
        static_cast<ulong>(MR_BROADPHASE_SCAN_BLOCK_SIZE);
    if (logicalPairs > 0xfffffffful ||
        dispatch.logicalPairCount != logicalPairs ||
        dispatch.scanBlockCount != expectedBlocks ||
        dispatch.scanBlockCount >
            MR_MAX_BROADPHASE_SCAN_BLOCKS) {
        status.code = MR_STEP_PAIR_CAPACITY_OVERFLOW;
        status.requiredPairs = status.logicalPairs;
        return;
    }

    for (uint bodyIndex = 0u;
         bodyIndex < dispatch.bodyCount;
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
            return;
        }
    }
    for (uint exclusionIndex = 0u;
         exclusionIndex < dispatch.exclusionCount;
         ++exclusionIndex) {
        const MRCandidatePairGPU exclusion =
            exclusions[exclusionIndex];
        if (exclusion.colliderA >= dispatch.shapeCount ||
            exclusion.colliderB >= dispatch.shapeCount) {
            status.code = MR_STEP_NONFINITE_INPUT;
            return;
        }
    }
    for (uint shapeIndex = 0u;
         shapeIndex < dispatch.shapeCount;
         ++shapeIndex) {
        const MRShapeGPU shape = shapes[shapeIndex];
        const MRAabbGPU aabb = aabbs[shapeIndex];
        float4 localRotation;
        if (shape.bodyIndex >= dispatch.bodyCount ||
            (shape.flags & ~MR_SHAPE_FLAG_SIMULATION_DISABLED) != 0u ||
            !collisionInputDomainXyz(shape.localPosition) ||
            !checkedQuaternion(
                shape.localRotation,
                localRotation
            ) ||
            !collisionInputDomainXyz(shape.dimensions) ||
            !collisionInputDomainXyz(
                shape.contactRestAndBoundingRadius
            ) ||
            shape.contactRestAndBoundingRadius.x < 0.0f ||
            shape.contactRestAndBoundingRadius.x <
                shape.contactRestAndBoundingRadius.y ||
            shape.contactRestAndBoundingRadius.z < 0.0f ||
            !canonicalFloat4(aabb.lower) ||
            !canonicalFloat4(aabb.upper) ||
            any(aabb.lower.xyz > aabb.upper.xyz)) {
            status.code = MR_STEP_NONFINITE_INPUT;
            return;
        }
        float4 bodyRotation;
        if (!checkedQuaternion(
                bodies[shape.bodyIndex].orientation,
                bodyRotation
            )) {
            status.code = MR_STEP_NONFINITE_INPUT;
            return;
        }
        const float4 worldRotation =
            quaternionMultiply(bodyRotation, localRotation);
        const float3 worldCenter =
            bodies[shape.bodyIndex].position.xyz +
            quaternionRotate(
                bodyRotation,
                shape.localPosition.xyz
            );
        if (!finite4(worldRotation) ||
            !collisionDomain(worldCenter)) {
            status.code = MR_STEP_NONFINITE_INPUT;
            return;
        }
    }
    for (uint shapeIndex = 0u;
         shapeIndex < dispatch.shapeCount;
         ++shapeIndex) {
        const MRShapeGPU shape = shapes[shapeIndex];
        if ((shape.flags & MR_SHAPE_FLAG_SIMULATION_DISABLED) !=
            0u) {
            continue;
        }
        if (!supportedShapeType(shape.shapeType)) {
            status.code = MR_STEP_UNSUPPORTED;
            return;
        }
        if (!validActiveDimensions(shape)) {
            status.code = MR_STEP_NONFINITE_INPUT;
            return;
        }
        if (shape.shapeType != MR_SHAPE_PLANE) {
            const MRAabbGPU aabb = aabbs[shapeIndex];
            if (!collisionDomain(aabb.lower.xyz) ||
                !collisionDomain(aabb.upper.xyz)) {
                status.code = MR_STEP_NONFINITE_INPUT;
                return;
            }
        }
    }
}

kernel void mr_broadphase_classify_pairs(
    constant const MRBroadphaseDispatchGPU& dispatch [[buffer(0)]],
    device const MRShapeGPU* shapes [[buffer(1)]],
    device const MRBodyStateGPU* bodies [[buffer(2)]],
    device const MRAabbGPU* aabbs [[buffer(3)]],
    device const MRCandidatePairGPU* exclusions [[buffer(4)]],
    device const MRBroadphaseStatusGPU& status [[buffer(5)]],
    device uint* pairFlags [[buffer(6)]],
    device MRCandidatePairGPU* logicalPairs [[buffer(7)]],
    const uint logicalIndex [[thread_position_in_grid]]
) {
    if (status.code != MR_STEP_SUCCESS ||
        logicalIndex >= dispatch.logicalPairCount) {
        return;
    }
    uint colliderA;
    uint colliderB;
    decodeLogicalPair(
        logicalIndex,
        dispatch.shapeCount,
        colliderA,
        colliderB
    );
    uint classification = 0u;
    const bool accepted = pairPasses(
        colliderA,
        colliderB,
        shapes,
        bodies,
        aabbs,
        exclusions,
        dispatch.exclusionCount,
        classification
    );
    pairFlags[logicalIndex] = accepted ? 1u : 0u;
    MRCandidatePairGPU pair = {};
    pair.environment = dispatch.environment;
    pair.colliderA = colliderA;
    pair.colliderB = colliderB;
    pair.flags = classification;
    logicalPairs[logicalIndex] = pair;
}

kernel void mr_broadphase_scan_pair_blocks(
    constant const MRBroadphaseDispatchGPU& dispatch [[buffer(0)]],
    device const MRBroadphaseStatusGPU& status [[buffer(1)]],
    device const uint* pairFlags [[buffer(2)]],
    device uint* pairOffsets [[buffer(3)]],
    device uint* blockSums [[buffer(4)]],
    const uint logicalIndex [[thread_position_in_grid]],
    const uint localIndex [[thread_index_in_threadgroup]],
    const uint blockIndex [[threadgroup_position_in_grid]]
) {
    threadgroup uint scan[MR_BROADPHASE_SCAN_BLOCK_SIZE];
    const uint flag =
        status.code == MR_STEP_SUCCESS &&
        logicalIndex < dispatch.logicalPairCount
        ? pairFlags[logicalIndex]
        : 0u;
    scan[localIndex] = flag;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u;
         stride < MR_BROADPHASE_SCAN_BLOCK_SIZE;
         stride <<= 1u) {
        const uint addend =
            localIndex >= stride ? scan[localIndex - stride] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scan[localIndex] += addend;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (status.code == MR_STEP_SUCCESS &&
        logicalIndex < dispatch.logicalPairCount) {
        pairOffsets[logicalIndex] = scan[localIndex] - flag;
    }
    if (localIndex == MR_BROADPHASE_SCAN_BLOCK_SIZE - 1u &&
        blockIndex < dispatch.scanBlockCount) {
        blockSums[blockIndex] = scan[localIndex];
    }
}

kernel void mr_broadphase_scan_block_sums(
    constant const MRBroadphaseDispatchGPU& dispatch [[buffer(0)]],
    device const MRBroadphaseStatusGPU& status [[buffer(1)]],
    device const uint* blockSums [[buffer(2)]],
    device uint* blockOffsets [[buffer(3)]],
    const uint localIndex [[thread_index_in_threadgroup]]
) {
    threadgroup uint scan[MR_MAX_BROADPHASE_SCAN_BLOCKS];
    const uint value =
        status.code == MR_STEP_SUCCESS &&
        localIndex < dispatch.scanBlockCount
        ? blockSums[localIndex]
        : 0u;
    scan[localIndex] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u;
         stride < MR_MAX_BROADPHASE_SCAN_BLOCKS;
         stride <<= 1u) {
        const uint addend =
            localIndex >= stride ? scan[localIndex - stride] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scan[localIndex] += addend;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (status.code == MR_STEP_SUCCESS &&
        localIndex < dispatch.scanBlockCount) {
        blockOffsets[localIndex] = scan[localIndex] - value;
    }
}

kernel void mr_broadphase_finalize_count(
    constant const MRBroadphaseDispatchGPU& dispatch [[buffer(0)]],
    device const uint* blockSums [[buffer(1)]],
    device const uint* blockOffsets [[buffer(2)]],
    device MRBroadphaseStatusGPU& status [[buffer(3)]],
    const uint index [[thread_position_in_grid]]
) {
    if (index != 0u || status.code != MR_STEP_SUCCESS) {
        return;
    }
    const uint required =
        dispatch.scanBlockCount == 0u
        ? 0u
        : blockOffsets[dispatch.scanBlockCount - 1u] +
            blockSums[dispatch.scanBlockCount - 1u];
    status.requiredPairs = required;
    if (required > dispatch.pairCapacity) {
        status.code = MR_STEP_PAIR_CAPACITY_OVERFLOW;
        status.emittedPairs = 0u;
        return;
    }
    status.emittedPairs = required;
}

kernel void mr_broadphase_scatter_pairs(
    constant const MRBroadphaseDispatchGPU& dispatch [[buffer(0)]],
    device const MRBroadphaseStatusGPU& status [[buffer(1)]],
    device const uint* pairFlags [[buffer(2)]],
    device const uint* pairOffsets [[buffer(3)]],
    device const uint* blockOffsets [[buffer(4)]],
    device const MRCandidatePairGPU* logicalPairs [[buffer(5)]],
    device MRCandidatePairGPU* outputPairs [[buffer(6)]],
    const uint logicalIndex [[thread_position_in_grid]]
) {
    if (status.code != MR_STEP_SUCCESS ||
        logicalIndex >= dispatch.logicalPairCount ||
        pairFlags[logicalIndex] == 0u) {
        return;
    }
    const uint blockIndex =
        logicalIndex / MR_BROADPHASE_SCAN_BLOCK_SIZE;
    const uint outputIndex =
        blockOffsets[blockIndex] + pairOffsets[logicalIndex];
    outputPairs[outputIndex] = logicalPairs[logicalIndex];
}
