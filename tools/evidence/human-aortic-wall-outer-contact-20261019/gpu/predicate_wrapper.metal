
namespace numi_matter_metal {
kernel void nm_outer_contact_test_pair_eligibility(
    constant NMMatterDispatchGPU& dispatch [[buffer(0)]],
    device const NMContinuumSurfacePrimitiveGPU* primitives [[buffer(1)]],
    device const NMContinuumObjectGPU* objects [[buffer(2)]],
    device const NMFEMTopologyNodeGPU* topologyNodes [[buffer(3)]],
    device uint4* result [[buffer(4)]],
    device const NMIncidenceRangeGPU* cohesiveSurfaceFaceRanges [[buffer(5)]],
    device const nm_uint4* cohesiveSurfaceFaceRefs [[buffer(6)]],
    device const NMCohesiveFaceGPU* cohesiveFaces [[buffer(7)]],
    uint global [[thread_position_in_grid]])
{
    if (global >= 2u) return;
    const uint leftIndex = global == 0u ? 0u : 1u;
    const uint rightIndex = 2u;
    const NMContinuumSurfacePrimitiveGPU left = primitives[leftIndex];
    const NMContinuumSurfacePrimitiveGPU right = primitives[rightIndex];
    const bool forward = surfacePairEligible(
        dispatch, left, right, objects, topologyNodes, 0u,
        cohesiveSurfaceFaceRanges, cohesiveSurfaceFaceRefs, cohesiveFaces);
    const bool reverse = surfacePairEligible(
        dispatch, right, left, objects, topologyNodes, 0u,
        cohesiveSurfaceFaceRanges, cohesiveSurfaceFaceRefs, cohesiveFaces);
    result[global] = uint4(forward ? 1u : 0u, reverse ? 1u : 0u,
        left.identity.w, right.identity.w);
}
} // namespace numi_matter_metal
