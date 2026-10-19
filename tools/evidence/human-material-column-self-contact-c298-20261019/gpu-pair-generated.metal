#include <metal_stdlib>
using namespace metal;
#define NM_INVALID_INDEX 0xffffffffu
#define NM_FEM_SURFACE_DISABLE_DEFORMABLE_CONTACT (1u << 31u)
#define NM_OBJECT_DISABLE_SELF_CONTACT (1u << 7u)
#define NM_OBJECT_DISABLE_DEFORMABLE_CONTACT (1u << 8u)
#define NM_CONTINUUM_SURFACE_POINT 1u
#define NM_TOPOLOGY_PAIRED_COHESIVE (1u << 5u)
#define NM_TOPOLOGY_COHESIVE_MIXED_MODE (1u << 6u)
#define NM_TOPOLOGY_COHESIVE_MORTAR (1u << 8u)
#define NM_COHESIVE_LAW_MIXED_MODE_BILINEAR 1u
struct NMMatterDispatchGPU { uint objectCount; uint gridNodeCount; uint femNodeCount; uint tetrahedronCount; uint cohesiveFaceCount; };
struct NMContinuumSurfacePrimitiveGPU { uint4 nodesAndObject; uint4 identity; float4 boundsMinimum; float4 boundsMaximum; };
struct NMFEMTopologyNodeGPU { uint4 identity; };
struct NMContinuumObjectGPU { uint representation; uint materialIndex; uint flags; uint schedulerIndex; };
struct NMIncidenceRangeGPU { uint first; uint count; uint objectIndex; uint reserved; };
typedef uint4 nm_uint4;
struct NMCohesiveFaceGPU { uint4 nodesAndFirst; uint4 adjacency; uint4 lawIdentity; };

inline bool surfacePrimitivesShareNode(
    const NMMatterDispatchGPU dispatch,
    const NMContinuumSurfacePrimitiveGPU left,
    const NMContinuumSurfacePrimitiveGPU right,
    device const NMFEMTopologyNodeGPU* topologyNodes,
    const uint environment
) {
    for (uint a = 0u; a < 3u; ++a)
        for (uint b = 0u; b < 3u; ++b) {
            if (left.nodesAndObject[a] != NM_INVALID_INDEX &&
                left.nodesAndObject[a] == right.nodesAndObject[b])
                return true;
            if (left.nodesAndObject[a] < dispatch.gridNodeCount ||
                right.nodesAndObject[b] < dispatch.gridNodeCount ||
                left.nodesAndObject[a] == NM_INVALID_INDEX ||
                right.nodesAndObject[b] == NM_INVALID_INDEX) continue;
            const uint leftNode =
                left.nodesAndObject[a] - dispatch.gridNodeCount;
            const uint rightNode =
                right.nodesAndObject[b] - dispatch.gridNodeCount;
            if (leftNode >= dispatch.femNodeCount ||
                rightNode >= dispatch.femNodeCount) continue;
            const uint base = environment * dispatch.femNodeCount;
            const NMFEMTopologyNodeGPU leftTopology =
                topologyNodes[base + leftNode];
            const NMFEMTopologyNodeGPU rightTopology =
                topologyNodes[base + rightNode];
            if (leftTopology.identity.x == rightTopology.identity.x)
                return true;
            // The Human three-layer shell stores a source material-column ID
            // in the otherwise-unused upper 22 bits of topology flags. Only
            // same-owner nodes with the same nonzero tag are material-adjacent;
            // distinct columns remain eligible for nonlocal self-contact.
            constexpr uint materialColumnShift = 10u;
            const uint leftMaterialColumn =
                leftTopology.identity.w >> materialColumnShift;
            const uint rightMaterialColumn =
                rightTopology.identity.w >> materialColumnShift;
            if (left.nodesAndObject.w == right.nodesAndObject.w &&
                leftMaterialColumn != 0u &&
                leftMaterialColumn == rightMaterialColumn)
                return true;
        }
    return false;
}

inline bool exactMixedMortarParentPair(
    const NMMatterDispatchGPU dispatch,
    const NMContinuumSurfacePrimitiveGPU left,
    const NMContinuumSurfacePrimitiveGPU right,
    const uint environment,
    device const NMIncidenceRangeGPU* cohesiveSurfaceFaceRanges,
    device const nm_uint4* cohesiveSurfaceFaceRefs,
    device const NMCohesiveFaceGPU* cohesiveFaces
) {
    const uint primitiveLimit = 4u * dispatch.tetrahedronCount;
    if (left.identity.x >= primitiveLimit ||
        right.identity.x >= primitiveLimit ||
        dispatch.cohesiveFaceCount == 0u) return false;
    const NMIncidenceRangeGPU range =
        cohesiveSurfaceFaceRanges[left.identity.x];
    const uint referenceCount = 2u * dispatch.cohesiveFaceCount;
    if (range.first >= referenceCount ||
        range.count > referenceCount - range.first) return false;
    const uint faceBase = environment * dispatch.cohesiveFaceCount;
    const uint required = NM_TOPOLOGY_PAIRED_COHESIVE |
        NM_TOPOLOGY_COHESIVE_MIXED_MODE | NM_TOPOLOGY_COHESIVE_MORTAR;
    for (uint local = 0u; local < range.count; ++local) {
        const nm_uint4 reference =
            cohesiveSurfaceFaceRefs[range.first + local];
        if (reference.x != right.identity.x ||
            reference.y >= dispatch.cohesiveFaceCount || reference.z > 1u)
            continue;
        const NMCohesiveFaceGPU face =
            cohesiveFaces[faceBase + reference.y];
        if ((face.adjacency.w & required) != required ||
            face.lawIdentity.x != NM_COHESIVE_LAW_MIXED_MODE_BILINEAR)
            continue;
        const bool leftIsSideB = reference.z == 1u;
        const uint leftTet = leftIsSideB
            ? face.adjacency.x : face.nodesAndFirst.w;
        const uint rightTet = leftIsSideB
            ? face.nodesAndFirst.w : face.adjacency.x;
        if ((left.identity.x >> 2u) == leftTet &&
            (right.identity.x >> 2u) == rightTet) return true;
    }
    return false;
}

inline bool surfacePairEligible(
    const NMMatterDispatchGPU dispatch,
    const NMContinuumSurfacePrimitiveGPU left,
    const NMContinuumSurfacePrimitiveGPU right,
    device const NMContinuumObjectGPU* objects,
    device const NMFEMTopologyNodeGPU* topologyNodes,
    const uint environment,
    device const NMIncidenceRangeGPU* cohesiveSurfaceFaceRanges,
    device const nm_uint4* cohesiveSurfaceFaceRefs,
    device const NMCohesiveFaceGPU* cohesiveFaces
) {
    if (((left.identity.w | right.identity.w) &
         NM_FEM_SURFACE_DISABLE_DEFORMABLE_CONTACT) != 0u) return false;
    if ((left.nodesAndObject.w < dispatch.objectCount &&
         (objects[left.nodesAndObject.w].flags &
            NM_OBJECT_DISABLE_DEFORMABLE_CONTACT) != 0u) ||
        (right.nodesAndObject.w < dispatch.objectCount &&
         (objects[right.nodesAndObject.w].flags &
            NM_OBJECT_DISABLE_DEFORMABLE_CONTACT) != 0u)) return false;
    if (left.nodesAndObject.w != right.nodesAndObject.w) return true;
    if (left.nodesAndObject.w < dispatch.objectCount &&
        (objects[left.nodesAndObject.w].flags &
            NM_OBJECT_DISABLE_SELF_CONTACT) != 0u) return false;
    if (left.identity.y == NM_CONTINUUM_SURFACE_POINT &&
        right.identity.y == NM_CONTINUUM_SURFACE_POINT) return false;
    if (!surfacePrimitivesShareNode(
            dispatch, left, right, topologyNodes, environment)) return true;
    // Keep the legacy sibling-culling rule except for the exact two parent
    // primitives represented by a mixed-mode mortar patch. Narrowphase then
    // suppresses covered×covered points and retains every uncovered region.
    return exactMixedMortarParentPair(
        dispatch, left, right, environment,
        cohesiveSurfaceFaceRanges, cohesiveSurfaceFaceRefs, cohesiveFaces);
}

kernel void testMaterialColumnPairEligibility(
    const device NMMatterDispatchGPU* dispatch [[buffer(0)]],
    const device NMContinuumSurfacePrimitiveGPU* left [[buffer(1)]],
    const device NMContinuumSurfacePrimitiveGPU* right [[buffer(2)]],
    const device NMContinuumObjectGPU* objects [[buffer(3)]],
    const device NMFEMTopologyNodeGPU* nodes [[buffer(4)]],
    const device NMIncidenceRangeGPU* faceRanges [[buffer(5)]],
    const device nm_uint4* faceRefs [[buffer(6)]],
    const device NMCohesiveFaceGPU* cohesiveFaces [[buffer(7)]],
    device const uint* environmentIndices [[buffer(8)]],
    device uint* result [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= 8u) return;
    result[tid] = surfacePairEligible(
        dispatch[0], left[tid], right[tid], objects, nodes, environmentIndices[tid],
        faceRanges, faceRefs, cohesiveFaces) ? 1u : 0u;
}
