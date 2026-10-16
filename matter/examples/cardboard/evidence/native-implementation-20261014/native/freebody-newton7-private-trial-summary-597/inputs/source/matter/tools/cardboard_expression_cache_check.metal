#include "../src/metal/common.metalinc"

// Direct native instrument for generated cache bytecode and one-sided branch
// derivatives. This calls the production constitutive evaluator; it is not a
// second constitutive implementation or a physical FEM step.
kernel void cardboard_expression_cache_check(
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
    float f[9], h[9];
    for (uint i = 0u; i < 9u; ++i) {
        f[i] = inputs[i];
        h[i] = inputs[9u + i];
    }
    const float3x3 F = componentsMatrix(f);
    const float3x3 H = componentsMatrix(h);
    constexpr float dt = 1.0e-4f;
    constexpr float temperature = 293.15f;
    float margin = 0.0f;
    bool valid = evaluateMaterialValidity(material, programs, instructions,
        parameters, acceptedState, F, float3x3(0.0f), dt, temperature, margin);
    valid = valid && margin > 0.0f;
    float3x3 stress(0.0f), tangent(0.0f);
    if (valid) evaluateAlgorithmicStressTangent(material, programs, instructions,
        parameters, acceptedState, F, H, float3x3(0.0f), float3x3(0.0f),
        dt, temperature, stress, tangent, valid);
    float p[9], dp[9];
    matrixComponents(stress, p);
    matrixComponents(tangent, dp);
    output[0] = valid ? 1.0f : 0.0f;
    output[1] = margin;
    for (uint i = 0u; i < 9u; ++i) {
        output[2u + i] = p[i];
        output[11u + i] = dp[i];
    }
}
