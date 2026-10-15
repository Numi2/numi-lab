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
    float f[9], h[9], fLow[9];
    for (uint i=0; i<9; ++i) {
        f[i]=inputs[i];
        h[i]=inputs[9+i];
        fLow[i]=inputs[18+i];
    }
    const bool accurateProjection = inputs[27] != 0.0f;
    const float3x3 F=componentsMatrix(f), H=componentsMatrix(h);
    float state[NM_MAX_MATERIAL_STATE];
    float stateLow[NM_MAX_MATERIAL_STATE] = {0.0f};
    float residual=-1.0f, stateNorm=0.0f;
    uint iterations=0u, flags=0u, failure=0u;
    bool valid=true;
    float3x3 stress(0.0f), tangent(0.0f);
    evaluateAlgorithmicStressTangent(material,programs,instructions,
        parameters,accepted,F,H,float3x3(0.0f),float3x3(0.0f),1e-4f,293.15f,
        stress,tangent,valid,state,&flags,&residual,&stateNorm,&iterations,&failure,
        accurateProjection ? fLow : nullptr,
        accurateProjection ? stateLow : nullptr);
    const bool projectionValid = (flags & kMaterialDiagProjectionConverged) != 0u;
    float p[9], dp[9]; matrixComponents(stress,p); matrixComponents(tangent,dp);
    output[0]=valid?1.0f:0.0f;
    for (uint i=0;i<9;++i) { output[1+i]=p[i]; output[10+i]=dp[i]; }
    for (uint i=0;i<NM_MAX_MATERIAL_STATE;++i)
        output[19+i]=i<material.stateCount ? state[i] : 0.0f;
    // Keep metadata outside the complete state arena. Slot 35 is itself a
    // state slot when NM_MAX_MATERIAL_STATE exceeds sixteen.
    output[19+NM_MAX_MATERIAL_STATE]=projectionValid?1.0f:0.0f;
    output[20+NM_MAX_MATERIAL_STATE]=residual;
    output[21+NM_MAX_MATERIAL_STATE]=stateNorm;
    output[22+NM_MAX_MATERIAL_STATE]=float(iterations);
    output[23+NM_MAX_MATERIAL_STATE]=float(flags);
    output[24+NM_MAX_MATERIAL_STATE]=float(failure);
    for (uint i=0;i<NM_MAX_MATERIAL_STATE;++i)
        output[25+NM_MAX_MATERIAL_STATE+i] =
            accurateProjection && i<material.stateCount ? stateLow[i] : 0.0f;
}
