#include "../src/metal/common.metalinc"

// Instrument only: calls the production material projection, stress and
// consistent tangent. It does not implement a second constitutive law.
kernel void cardboard_paper_check(
    constant NMMaterialGPU& material [[buffer(0)]],
    device const NMScalarProgramGPU* programs [[buffer(1)]],
    device const NMExpressionInstructionGPU* instructions [[buffer(2)]],
    device const float* parameters [[buffer(3)]],
    device const float* accepted [[buffer(4)]],
    device const float* inputs [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (index != 0u) return;
    using namespace numi_matter_metal;
    float f[9], h[9];
    for (uint i=0; i<9; ++i) { f[i]=inputs[i]; h[i]=inputs[9+i]; }
    const float3x3 F=componentsMatrix(f), H=componentsMatrix(h);
    float state[NM_MAX_MATERIAL_STATE];
    bool valid=projectMaterialState(material,programs,instructions,parameters,
        accepted,F,float3x3(0.0f),1e-4f,293.15f,state);
    const bool projectionValid = valid;
    float3x3 stress(0.0f), tangent(0.0f);
    if (valid) evaluateAlgorithmicStressTangent(material,programs,instructions,
        parameters,accepted,F,H,float3x3(0.0f),float3x3(0.0f),1e-4f,293.15f,
        stress,tangent,valid);
    float p[9], dp[9]; matrixComponents(stress,p); matrixComponents(tangent,dp);
    output[0]=valid?1.0f:0.0f;
    output[35]=projectionValid?1.0f:0.0f;
    for (uint i=0;i<9;++i) { output[1+i]=p[i]; output[10+i]=dp[i]; }
    for (uint i=0;i<NM_MAX_MATERIAL_STATE;++i)
        output[19+i]=projectionValid && i<material.stateCount ? state[i] : 0.0f;
}
