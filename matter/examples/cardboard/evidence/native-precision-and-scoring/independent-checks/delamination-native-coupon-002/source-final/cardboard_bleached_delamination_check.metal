#include "../src/metal/common.metalinc"

// Direct constitutive instrument. This calls the production material validity,
// state projection, stress and algorithmic-tangent entry points; it is not an
// alternate constitutive implementation or a physical coupon simulation.
kernel void cardboard_bleached_delamination_check(
    constant NMMaterialGPU& material [[buffer(0)]],
    device const NMScalarProgramGPU* programs [[buffer(1)]],
    device const NMExpressionInstructionGPU* instructions [[buffer(2)]],
    device const float* parameters [[buffer(3)]],
    device const float* acceptedState [[buffer(4)]],
    device const float* inputs [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (index != 0u) return;
    using namespace numi_matter_metal;

    float f[9], direction[9];
    for (uint i = 0u; i < 9u; ++i) {
        f[i] = inputs[i];
        direction[i] = inputs[9u + i];
    }
    const float3x3 deformation = componentsMatrix(f);
    const float3x3 deformationDirection = componentsMatrix(direction);
    constexpr float dt = 1.0e-4f;
    constexpr float temperature = 293.15f;

    float validityMargin = 0.0f;
    const bool validityEvaluated = evaluateMaterialValidity(
        material, programs, instructions, parameters, acceptedState,
        deformation, float3x3(0.0f), dt, temperature, validityMargin
    );
    bool valid = validityEvaluated && validityMargin > 0.0f;
    const bool admitted = valid;

    float projected[NM_MAX_MATERIAL_STATE] = {};
    if (valid) {
        valid = projectMaterialState(
            material, programs, instructions, parameters, acceptedState,
            deformation, float3x3(0.0f), dt, temperature, projected
        );
    }

    bool directStressValid = valid;
    (void)evaluateStressWithCandidate(
        material, programs, instructions, parameters, acceptedState, projected,
        deformation, float3x3(0.0f), dt, temperature, directStressValid
    );
    bool directTangentValid = valid;
    (void)evaluateTangentVectorWithCandidate(
        material, programs, instructions, parameters, acceptedState, projected,
        deformation, deformationDirection, float3x3(0.0f), float3x3(0.0f),
        dt, temperature, directTangentValid
    );

    float3x3 stress(0.0f), tangent(0.0f);
    const bool projectionSucceeded = valid;
    if (valid) {
        evaluateAlgorithmicStressTangent(
            material, programs, instructions, parameters, acceptedState,
            deformation, deformationDirection, float3x3(0.0f),
            float3x3(0.0f), dt, temperature, stress, tangent, valid
        );
    }
    const bool stressSucceeded = valid;

    float p[9], dp[9];
    matrixComponents(stress, p);
    matrixComponents(tangent, dp);
    output[0] = valid ? 1.0f : 0.0f;
    output[1] = validityMargin;
    for (uint i = 0u; i < 9u; ++i) {
        output[2u + i] = p[i];
        output[11u + i] = dp[i];
    }
    for (uint i = 0u; i < NM_MAX_MATERIAL_STATE; ++i) {
        output[20u + i] = valid && i < material.stateCount ? projected[i] : 0.0f;
    }
    // Explicitly retain input acceptance history for host-side transaction
    // boundary comparison; the production evaluator receives it as const.
    for (uint i = 0u; i < NM_MAX_MATERIAL_STATE; ++i) {
        output[20u + NM_MAX_MATERIAL_STATE + i] =
            i < material.stateCount ? acceptedState[i] : 0.0f;
    }
    output[20u + 2u * NM_MAX_MATERIAL_STATE] = validityEvaluated ? 1.0f : 0.0f;
    output[21u + 2u * NM_MAX_MATERIAL_STATE] = admitted ? 1.0f : 0.0f;
    output[22u + 2u * NM_MAX_MATERIAL_STATE] = projectionSucceeded ? 1.0f : 0.0f;
    output[23u + 2u * NM_MAX_MATERIAL_STATE] = stressSucceeded ? 1.0f : 0.0f;
    output[24u + 2u * NM_MAX_MATERIAL_STATE] = directStressValid ? 1.0f : 0.0f;
    output[25u + 2u * NM_MAX_MATERIAL_STATE] = directTangentValid ? 1.0f : 0.0f;
}
