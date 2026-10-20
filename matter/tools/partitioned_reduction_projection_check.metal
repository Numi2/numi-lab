#include "../src/metal/common.metalinc"
#include "../src/metal/fem.metalinc"
#include "../src/metal/partitioned_reduction_projection.metalinc"
#include "../src/metal/partitioned_reduction_fem_tangent.metalinc"

namespace numi_matter_metal {

// Small production-helper witness. The tet node IDs deliberately have a
// nonzero global offset and an independently permuted slot-to-basis-row map.
kernel void nm_partitioned_reduction_fem_tangent_fd_check(
    constant NMTetrahedronGPU& tet [[buffer(0)]],
    constant NMContinuumObjectGPU& object [[buffer(1)]],
    device const float* tangentCache [[buffer(2)]],
    device const nm_float4* basisValues [[buffer(3)]],
    device const uint* tetrahedronBasisRows [[buffer(4)]],
    constant float& dt [[buffer(5)]],
    device float* projectedOffDiagonal [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    constexpr uint kModeCount = 2u;
    if (gid >= kModeCount * kModeCount) return;
    const uint leftMode = gid / kModeCount;
    const uint rightMode = gid % kModeCount;
    const uint row0 = tetrahedronBasisRows[0];
    const uint row1 = tetrahedronBasisRows[1];
    const uint row2 = tetrahedronBasisRows[2];
    const uint row3 = tetrahedronBasisRows[3];
    const ulong leftBase = ulong(leftMode) * 4u;
    const ulong rightBase = ulong(rightMode) * 4u;
    const float3 left[4] = {
        basisValues[leftBase + row0].xyz,
        basisValues[leftBase + row1].xyz,
        basisValues[leftBase + row2].xyz,
        basisValues[leftBase + row3].xyz};
    const float3 right[4] = {
        basisValues[rightBase + row0].xyz,
        basisValues[rightBase + row1].xyz,
        basisValues[rightBase + row2].xyz,
        basisValues[rightBase + row3].xyz};
    const float3x3 inverseRest = matrixFromRows(
        tet.inverseRestRow0, tet.inverseRestRow1, tet.inverseRestRow2);
    const ulong globalTet = 0u;
    const float3x3 fullDp = float3x3(
        right[1] - right[0], right[2] - right[0], right[3] - right[0]);
    NMFEMElementVectorGPU full;
    if (!nmHumanRegionalCachedFEMAction(
            object, tet, tangentCache, globalTet, fullDp * inverseRest, dt,
            full)) {
        projectedOffDiagonal[gid] = as_type<float>(0x7fc00000u);
        return;
    }
    const float3 zero = float3(0.0f);
    const float3x3 nodeOnlyDp[4] = {
        float3x3(-right[0], -right[0], -right[0]),
        float3x3(right[1], zero, zero),
        float3x3(zero, right[2], zero),
        float3x3(zero, zero, right[3])};
    float3 self[4];
    for (uint node = 0u; node < 4u; ++node) {
        NMFEMElementVectorGPU local;
        if (!nmHumanRegionalCachedFEMAction(
                object, tet, tangentCache, globalTet,
                nodeOnlyDp[node] * inverseRest, dt, local)) {
            projectedOffDiagonal[gid] = as_type<float>(0x7fc00000u);
            return;
        }
        self[node] = node == 0u ? local.node0.xyz :
            (node == 1u ? local.node1.xyz :
             (node == 2u ? local.node2.xyz : local.node3.xyz));
    }
    projectedOffDiagonal[gid] =
        dot(left[0], full.node0.xyz - self[0]) +
        dot(left[1], full.node1.xyz - self[1]) +
        dot(left[2], full.node2.xyz - self[2]) +
        dot(left[3], full.node3.xyz - self[3]);
}

} // namespace numi_matter_metal

