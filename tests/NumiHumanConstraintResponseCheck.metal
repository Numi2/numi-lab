#include <metal_stdlib>
using namespace metal;

#include "../src/metal/NumiHumanConstraintResponse.metalinc"

kernel void numi_human_constraint_response_check(
    device const float* factor [[buffer(0)]],
    device const float* pointJacobians [[buffer(1)]],
    device float* sourceRows [[buffer(2)]],
    device float* inverseResponses [[buffer(3)]],
    device float* delassus [[buffer(4)]],
    device uint* result [[buffer(5)]],
    device const float* singularFactor [[buffer(6)]],
    uint lane [[thread_position_in_grid]]
) {
    if (lane != 0u) return;
    constexpr uint nv = 3u;
    if (!nmHumanConstraintBuildEqualityRow(
            0u, 2u, 0.35f, nv, sourceRows)) {
        result[0] = 1u;
        return;
    }
    const float3 normal = float3(0.3f, 0.4f, sqrt(0.75f));
    nmHumanConstraintBuildPointRow(
        pointJacobians, 0u, 0u, nv, normal, sourceRows + nv);

    for (uint row = 0u; row < 2u; ++row) {
        device float* response = inverseResponses + row * nv;
        for (uint dof = 0u; dof < nv; ++dof)
            response[dof] = sourceRows[row * nv + dof];
        float workspace[3];
        float minimumPivot = 0.0f;
        float maximumPivot = 0.0f;
        if (!nmHumanConstraintSolveDenseFactor(
                factor, 0u, workspace, response, nv, 1.0e-12f, true,
                minimumPivot, maximumPivot)) {
            result[0] = 2u + row;
            return;
        }
    }
    for (uint row = 0u; row < 2u; ++row) {
        for (uint column = 0u; column < 2u; ++column) {
            delassus[row * 2u + column] = row == 0u
                ? nmHumanConstraintBilateralResponse(
                    inverseResponses + column * nv, 0u, 2u, 0.35f)
                : nmHumanConstraintDelassusDot(
                    sourceRows + row * nv,
                    inverseResponses + column * nv, nv);
        }
    }

    float invalidRow[3] = {0.0f, 0.0f, 0.0f};
    const bool invalidIndexRejected =
        !nmHumanConstraintBuildEqualityRow(
            nv, 0xffffffffu, 0.0f, nv, invalidRow);
    const bool repeatedIndexRejected =
        !nmHumanConstraintBuildEqualityRow(1u, 1u, 0.1f, nv, invalidRow);
    const bool nonfiniteDerivativeRejected =
        !nmHumanConstraintBuildEqualityRow(
            0u, 2u, NAN, nv, invalidRow);
    float oneDofRow[1] = {0.0f};
    const bool smallestRowAccepted =
        nmHumanConstraintBuildEqualityRow(
            0u, 0xffffffffu, 0.0f, 1u, oneDofRow) &&
        oneDofRow[0] == 1.0f;
    nmHumanConstraintBuildLimitRow(0u, 1u, oneDofRow);
    const bool smallestLimitAccepted = oneDofRow[0] == 1.0f;
    const bool normalValid = nmHumanConstraintUnitNormalValid(
        normal, 2.0e-4f);
    const bool zeroNormalRejected = !nmHumanConstraintUnitNormalValid(
        float3(0.0f), 2.0e-4f);
    float singularWorkspace[3];
    float singularOutput[3] = {1.0f, 0.0f, 0.0f};
    float singularMin = 0.0f;
    float singularMax = 0.0f;
    const bool singularFactorRejected =
        !nmHumanConstraintSolveDenseFactor(
            singularFactor, 0u, singularWorkspace, singularOutput, nv,
            1.0e-12f, true, singularMin, singularMax);
    result[1] = invalidIndexRejected && repeatedIndexRejected &&
            nonfiniteDerivativeRejected && smallestRowAccepted &&
            smallestLimitAccepted
        ? 0u : 1u;
    result[2] = normalValid && zeroNormalRejected ? 0u : 1u;
    result[3] = singularFactorRejected ? 0u : 1u;
    result[0] = 0u;
}
