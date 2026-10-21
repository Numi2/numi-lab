#include <metal_stdlib>
using namespace metal;

#include "../src/metal/NumiHumanBilateralKKT.metalinc"

kernel void numanx_human_bilateral_kkt_prepare_fixture(
    device const float* a0Factor [[buffer(0)]],
    device float* equalityRows [[buffer(1)]],
    device float* inverseResponses [[buffer(2)]],
    device float* schurLu [[buffer(3)]],
    device float* schurInverseScale [[buffer(4)]],
    device float* schurPivots [[buffer(5)]],
    device float* witness [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) return;
    constexpr uint dofCount = 2u;
    float row[dofCount];
    if (!nmHumanConstraintBuildEqualityRow(
            0u, 1u, -1.0f, dofCount, row)) {
        witness[0] = NAN;
        return;
    }
    for (uint dof = 0u; dof < dofCount; ++dof) {
        equalityRows[dof] = row[dof];
        inverseResponses[dof] = row[dof];
    }
    float workspace[dofCount];
    float minPivot = 0.0f;
    float maxPivot = 0.0f;
    if (!nmHumanConstraintSolveDenseFactor(
            a0Factor, 0u, workspace, inverseResponses, dofCount,
            1.0e-12f, true, minPivot, maxPivot)) {
        witness[0] = NAN;
        return;
    }
    schurLu[0] = nmHumanConstraintDelassusDot(
        equalityRows, inverseResponses, dofCount);
    witness[0] = schurLu[0];
    if (!mrNumiHumanBilateralFactor(
            schurLu, schurInverseScale, schurPivots, 1u))
        witness[0] = NAN;
}
