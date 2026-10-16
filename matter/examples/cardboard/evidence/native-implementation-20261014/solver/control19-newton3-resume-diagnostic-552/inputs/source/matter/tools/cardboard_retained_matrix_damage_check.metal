#include "../src/metal/common.metalinc"

// Instrument only: exercise the production material projection, stress, and
// consistent tangent on the retained Haj-Ali matrix plus an explicitly
// synthetic crack-band overlay. This is not a physical FEM step or calibration.
kernel void cardboard_retained_matrix_damage_check(
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

    float f[9], direction[9], fLow[9];
    for (uint i = 0u; i < 9u; ++i) {
        f[i] = inputs[i];
        direction[i] = inputs[9u + i];
        fLow[i] = inputs[18u + i];
    }
    const bool compensatedDeformation = inputs[27u] != 0.0f;

    const float3x3 deformation = componentsMatrix(f);
    const float3x3 deformationDirection = componentsMatrix(direction);
    constexpr float dt = 2.0e-3f;
    constexpr float temperature = 293.15f;
    float state[NM_MAX_MATERIAL_STATE] = {};
    float stateLow[NM_MAX_MATERIAL_STATE] = {};
    float residual = -1.0f;
    float stateNorm = 0.0f;
    uint iterations = 0u;
    uint flags = 0u;
    uint failure = 0u;
    bool valid = true;
    float3x3 stress(0.0f);
    float3x3 tangent(0.0f);
    evaluateAlgorithmicStressTangent(
        material, programs, instructions, parameters, acceptedState,
        deformation, deformationDirection, float3x3(0.0f), float3x3(0.0f),
        dt, temperature, stress, tangent, valid, state, &flags, &residual,
        &stateNorm, &iterations, &failure,
        compensatedDeformation ? fLow : nullptr, stateLow);

    float p[9], dp[9];
    matrixComponents(stress, p);
    matrixComponents(tangent, dp);
    output[0] = valid ? 1.0f : 0.0f;
    for (uint i = 0u; i < 9u; ++i) {
        output[1u + i] = p[i];
        output[10u + i] = dp[i];
    }
    for (uint i = 0u; i < NM_MAX_MATERIAL_STATE; ++i) {
        output[19u + i] = i < material.stateCount ? state[i] : 0.0f;
    }
    output[19u + NM_MAX_MATERIAL_STATE] =
        (flags & kMaterialDiagProjectionConverged) != 0u ? 1.0f : 0.0f;
    output[20u + NM_MAX_MATERIAL_STATE] = residual;
    output[21u + NM_MAX_MATERIAL_STATE] = stateNorm;
    output[22u + NM_MAX_MATERIAL_STATE] = float(iterations);
    output[23u + NM_MAX_MATERIAL_STATE] = float(flags);
    output[24u + NM_MAX_MATERIAL_STATE] = float(failure);
    constexpr uint stateLowOffset = 25u + NM_MAX_MATERIAL_STATE;
    for (uint i = 0u; i < NM_MAX_MATERIAL_STATE; ++i)
        output[stateLowOffset + i] = i < material.stateCount ? stateLow[i] : 0.0f;
}
