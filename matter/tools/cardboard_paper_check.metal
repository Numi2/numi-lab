#include "../src/metal/common.metalinc"

// Replays exact captured source15 inputs through the same production material
// evaluator.  The second direction matrix is d(Fdot)/d(direction), supplied
// from the sealed FEM operator capture rather than inferred here.
kernel void cardboard_bilinear15_captured_point(
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
    float f[9], direction[9], fLow[9], fDot[9], rateDirection[9];
    for (uint i=0u; i<9u; ++i) {
        f[i]=inputs[i];
        direction[i]=inputs[9u+i];
        fLow[i]=inputs[18u+i];
        fDot[i]=inputs[27u+i];
        rateDirection[i]=inputs[36u+i];
    }
    const float timestep=inputs[45u];
    const float temperature=inputs[46u];
    const bool accurateProjection=inputs[47u]>=0.5f;
    float state[NM_MAX_MATERIAL_STATE]={0.0f};
    float stateLow[NM_MAX_MATERIAL_STATE]={0.0f};
    float residual=-1.0f, stateNorm=0.0f;
    uint iterations=0u, flags=0u, failure=0u;
    bool valid=true;
    float3x3 stress(0.0f), tangent(0.0f);
    evaluateAlgorithmicStressTangent(
        material, programs, instructions, parameters, accepted,
        componentsMatrix(f), componentsMatrix(direction), componentsMatrix(fDot),
        componentsMatrix(rateDirection), timestep, temperature,
        stress, tangent, valid, state, &flags, &residual, &stateNorm,
        &iterations, &failure,
        fLow,
        accurateProjection ? stateLow : nullptr);
    float p[9], dp[9];
    matrixComponents(stress,p);
    matrixComponents(tangent,dp);
    output[0]=valid?1.0f:0.0f;
    for (uint i=0u;i<9u;++i) { output[1u+i]=p[i]; output[10u+i]=dp[i]; }
    for (uint i=0u;i<NM_MAX_MATERIAL_STATE;++i) {
        output[19u+i]=i<material.stateCount?state[i]:0.0f;
        output[25u+NM_MAX_MATERIAL_STATE+i]=
            accurateProjection&&i<material.stateCount?stateLow[i]:0.0f;
    }
    output[19u+NM_MAX_MATERIAL_STATE]=
        (flags&kMaterialDiagProjectionConverged)!=0u?1.0f:0.0f;
    output[20u+NM_MAX_MATERIAL_STATE]=residual;
    output[21u+NM_MAX_MATERIAL_STATE]=stateNorm;
    output[22u+NM_MAX_MATERIAL_STATE]=float(iterations);
    output[23u+NM_MAX_MATERIAL_STATE]=float(flags);
    output[24u+NM_MAX_MATERIAL_STATE]=float(failure);
}

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
    float f[9], h[9], fLow[9], rate[9];
    for (uint i=0; i<9; ++i) {
        f[i]=inputs[i];
        h[i]=inputs[9+i];
        fLow[i]=inputs[18+i];
        rate[i]=inputs[27+i];
    }
    const float timestep = inputs[36];
    const bool accurateProjection = inputs[37] != 0.0f;
    const float3x3 F=componentsMatrix(f), H=componentsMatrix(h);
    float state[NM_MAX_MATERIAL_STATE];
    float stateLow[NM_MAX_MATERIAL_STATE] = {0.0f};
    float residual=-1.0f, stateNorm=0.0f;
    uint iterations=0u, flags=0u, failure=0u;
    bool valid=true;
    float3x3 stress(0.0f), tangent(0.0f);
    evaluateAlgorithmicStressTangent(material,programs,instructions,
        parameters,accepted,F,H,componentsMatrix(rate),float3x3(0.0f),
        timestep,293.15f,
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

// Bounded replay diagnostic: evaluate the compiled production implicit
// residual bytecode at four caller-supplied state snapshots. This is not a
// projection and cannot modify accepted history. Each thread handles one
// candidate so the source/reference states remain directly comparable.
kernel void cardboard_paper_compiled_residual_rows(
    constant NMMaterialGPU& material [[buffer(0)]],
    device const NMScalarProgramGPU* programs [[buffer(1)]],
    device const NMExpressionInstructionGPU* instructions [[buffer(2)]],
    device const float* parameters [[buffer(3)]],
    device const float* accepted [[buffer(4)]],
    device const float* inputs [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint candidateIndex [[thread_position_in_grid]]) {
    constexpr uint candidateCount=4u;
    constexpr uint candidateStride=1u+2u*NM_MAX_MATERIAL_STATE;
    constexpr uint outputStride=2u+2u*NM_MAX_MATERIAL_STATE;
    if(candidateIndex>=candidateCount) return;
    using namespace numi_matter_metal;
    float f[9],fLow[9],rate[9];
    for(uint i=0u;i<9u;++i){f[i]=inputs[i];fLow[i]=inputs[9u+i];rate[i]=inputs[18u+i];}
    float candidateHigh[NM_MAX_MATERIAL_STATE]={0.0f};
    float candidateLow[NM_MAX_MATERIAL_STATE]={0.0f};
    const uint candidateBase=40u+candidateIndex*candidateStride;
    const bool candidateAvailable=inputs[candidateBase]>=0.5f;
    if(!candidateAvailable){
        const uint base=candidateIndex*outputStride;
        output[base]=0.0f;output[base+1u]=0.0f;
        for(uint i=0u;i<2u*NM_MAX_MATERIAL_STATE;++i)output[base+2u+i]=0.0f;
        return;
    }
    for(uint i=0u;i<material.stateCount;++i){
        candidateHigh[i]=inputs[candidateBase+1u+i];
        candidateLow[i]=inputs[candidateBase+1u+NM_MAX_MATERIAL_STATE+i];
    }
    const NMPaperCandidateState candidate{candidateHigh,candidateLow};
    float residual[NM_MAX_MATERIAL_STATE]={0.0f};
    float unusedJacobian[1]={0.0f};
    MRCompensatedScalar residualWords[NM_MAX_MATERIAL_STATE];
    bool valid=material.stateCount>0u&&
        material.stateCount<=NM_MAX_MATERIAL_STATE&&
        material.implicitResidualProgramOffset!=NM_INVALID_INDEX;
    bool nonfinite=false;
    if(valid){
        valid=evaluateImplicitMaterialSystem(material,programs,instructions,
            parameters,accepted,candidate,componentsMatrix(f),float3x3(0.0f),
            componentsMatrix(rate),inputs[300u],293.15f,residual,unusedJacobian,false,
            &nonfinite,fLow,residualWords,nullptr);
    }
    const uint base=candidateIndex*outputStride;
    output[base]=valid?1.0f:0.0f;
    output[base+1u]=nonfinite?1.0f:0.0f;
    for(uint i=0u;i<NM_MAX_MATERIAL_STATE;++i){
        output[base+2u+i]=valid&&i<material.stateCount?
            residualWords[i].high:0.0f;
        output[base+2u+NM_MAX_MATERIAL_STATE+i]=
            valid&&i<material.stateCount?residualWords[i].low:0.0f;
    }
}

// Diagnostic sibling of the production-path checker above. It runs the same
// generic projection/stress/tangent entrypoint, then independently emits the
// local implicit system and its tangent contraction so large opposing terms
// can be audited without changing the constitutive law or acceptance gates.
kernel void cardboard_paper_tangent_diagnostic_check(
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
    for (uint i=0;i<9u;++i) { f[i]=inputs[i]; h[i]=inputs[9u+i]; fLow[i]=inputs[18u+i]; }
    const bool accurateProjection=inputs[27]!=0.0f;
    const bool captureProjectionTrace=inputs[28]!=0.0f;
    const float3x3 F=componentsMatrix(f), H=componentsMatrix(h);
    float state[NM_MAX_MATERIAL_STATE];
    float stateLow[NM_MAX_MATERIAL_STATE]={0.0f};
    float residualNorm=-1.0f,stateNorm=0.0f;
    uint iterations=0u,flags=0u,failure=0u;
    bool valid=true;
    float3x3 stress(0.0f),tangent(0.0f);
    float projectionTrace[kMaterialProjectionTraceSlots];
    evaluateAlgorithmicStressTangent(material,programs,instructions,parameters,
        accepted,F,H,float3x3(0.0f),float3x3(0.0f),1.0e-4f,293.15f,
        stress,tangent,valid,state,&flags,&residualNorm,&stateNorm,&iterations,&failure,
        accurateProjection?fLow:nullptr,
        accurateProjection?stateLow:nullptr,
        captureProjectionTrace?projectionTrace:nullptr);
    const bool projectionValid=(flags&kMaterialDiagProjectionConverged)!=0u;
    float p[9],dp[9]; matrixComponents(stress,p); matrixComponents(tangent,dp);
    output[0]=valid?1.0f:0.0f;
    for(uint i=0;i<9u;++i){output[1u+i]=p[i];output[10u+i]=dp[i];}
    for(uint i=0;i<NM_MAX_MATERIAL_STATE;++i)
        output[19u+i]=i<material.stateCount?state[i]:0.0f;
    output[19u+NM_MAX_MATERIAL_STATE]=projectionValid?1.0f:0.0f;
    output[20u+NM_MAX_MATERIAL_STATE]=residualNorm;
    output[21u+NM_MAX_MATERIAL_STATE]=stateNorm;
    output[22u+NM_MAX_MATERIAL_STATE]=float(iterations);
    output[23u+NM_MAX_MATERIAL_STATE]=float(flags);
    output[24u+NM_MAX_MATERIAL_STATE]=float(failure);
    for(uint i=0;i<NM_MAX_MATERIAL_STATE;++i)
        output[25u+NM_MAX_MATERIAL_STATE+i]=
            accurateProjection&&i<material.stateCount?stateLow[i]:0.0f;

    const uint base=25u+2u*NM_MAX_MATERIAL_STATE;
    const uint n=material.stateCount;
    const uint diagnosticCapacity=4u+NM_MAX_MATERIAL_STATE+
        NM_MAX_MATERIAL_STATE*NM_MAX_MATERIAL_STATE+NM_MAX_MATERIAL_STATE+
        9u*NM_MAX_MATERIAL_STATE+9u+NM_MAX_MATERIAL_STATE+9u;
    if(captureProjectionTrace) {
        const uint traceOffset=base+diagnosticCapacity;
        for(uint i=0u;i<kMaterialProjectionTraceSlots;++i)
            output[traceOffset+i]=projectionTrace[i];
    }
    for(uint i=0u;i<diagnosticCapacity;++i) output[base+i]=0.0f;
    output[base]=1.0f;
    output[base+1u]=0.0f;
    output[base+2u]=0.0f;
    output[base+3u]=float(n);
    if(n==0u||n>NM_MAX_MATERIAL_STATE||
       material.implicitResidualProgramOffset==NM_INVALID_INDEX||
       material.implicitJacobianProgramOffset==NM_INVALID_INDEX||
       material.implicitDeformationProgramOffset==NM_INVALID_INDEX||
       material.stressStateDerivativeProgramOffset==NM_INVALID_INDEX) return;

    float localResidual[NM_MAX_MATERIAL_STATE];
    float localJacobian[NM_MAX_MATERIAL_STATE*NM_MAX_MATERIAL_STATE];
    bool diagnosticValid=true,nonfinite=false;
    const bool evaluated=evaluateImplicitMaterialSystem(material,programs,instructions,
        parameters,accepted,state,F,H,float3x3(0.0f),1.0e-4f,293.15f,
        localResidual,localJacobian,true,&nonfinite,
        accurateProjection?fLow:nullptr);
    if(!evaluated||nonfinite) return;
    uint cursor=base+4u;
    const uint residualOffset=cursor;
    for(uint row=0u;row<n;++row) output[cursor++]=localResidual[row];
    const uint jacobianOffset=cursor;
    for(uint row=0u;row<n;++row)for(uint column=0u;column<n;++column)
        output[cursor++]=localJacobian[row*n+column];

    float deformation[9],direction[9],rate[9];
    matrixComponents(F,deformation); matrixComponents(H,direction);
    for(uint i=0u;i<9u;++i) rate[i]=0.0f;
    float deformationResidualDerivative[NM_MAX_MATERIAL_STATE];
    for(uint row=0u;row<n;++row) {
        bool derivativeNonfinite=false;
        deformationResidualDerivative[row]=
            evaluateScalarProgramWithCandidate(material.implicitDeformationProgramOffset+row,
                programs,instructions,parameters+material.parameterOffset,
                accepted,state,deformation,direction,rate,1.0e-4f,293.15f,
                diagnosticValid,&derivativeNonfinite,
                accurateProjection?fLow:nullptr);
        if(!diagnosticValid) return;
    }
    const uint deformationDerivativeOffset=cursor;
    for(uint row=0u;row<n;++row) output[cursor++]=deformationResidualDerivative[row];

    float stressStateDerivative[9u*NM_MAX_MATERIAL_STATE];
    for(uint component=0u;component<9u;++component) {
        for(uint stateIndex=0u;stateIndex<n;++stateIndex) {
            bool derivativeNonfinite=false;
            stressStateDerivative[component*n+stateIndex]=
                evaluateScalarProgramWithCandidate(
                    material.stressStateDerivativeProgramOffset+component*n+stateIndex,
                    programs,instructions,parameters+material.parameterOffset,
                    accepted,state,deformation,direction,rate,1.0e-4f,293.15f,
                    diagnosticValid,&derivativeNonfinite,
                    accurateProjection?fLow:nullptr);
            if(!diagnosticValid) return;
        }
    }
    const uint stressStateOffset=cursor;
    for(uint i=0u;i<9u*n;++i) output[cursor++]=stressStateDerivative[i];

    bool baseValid=true,baseNonfinite=false;
    const float3x3 baseTangent=evaluateTangentVectorWithCandidate(
        material,programs,instructions,parameters,accepted,state,F,H,
        float3x3(0.0f),float3x3(0.0f),1.0e-4f,293.15f,
        baseValid,&baseNonfinite,accurateProjection?fLow:nullptr);
    if(!baseValid) return;
    float baseValues[9]; matrixComponents(baseTangent,baseValues);
    const uint baseTangentOffset=cursor;
    for(uint i=0u;i<9u;++i) output[cursor++]=baseValues[i];

    float solveMatrix[NM_MAX_MATERIAL_STATE*NM_MAX_MATERIAL_STATE];
    float stateDirection[NM_MAX_MATERIAL_STATE];
    for(uint row=0u;row<n;++row) {
        stateDirection[row]=-deformationResidualDerivative[row];
        for(uint column=0u;column<n;++column)
            solveMatrix[row*n+column]=localJacobian[row*n+column];
    }
    const bool solveValid=solveLocalMaterialSystem(solveMatrix,stateDirection,n);
    output[base+2u]=solveValid?1.0f:0.0f;
    if(!solveValid) return;
    const uint stateDirectionOffset=cursor;
    for(uint row=0u;row<n;++row) output[cursor++]=stateDirection[row];

    float reconstructed[9];
    for(uint component=0u;component<9u;++component) {
        float value=baseValues[component];
        for(uint stateIndex=0u;stateIndex<n;++stateIndex)
            value+=stressStateDerivative[component*n+stateIndex]*stateDirection[stateIndex];
        reconstructed[component]=value;
    }
    const uint reconstructedOffset=cursor;
    for(uint i=0u;i<9u;++i) output[cursor++]=reconstructed[i];
    output[base+1u]=1.0f;
    // Keep computed offsets explicit for host schema review. The packed order
    // is residual, J, r_F[H], dP/dz, fixed-state tangent, dz/dF[H], recon.
    (void)residualOffset;
    (void)jacobianOffset;
    (void)deformationDerivativeOffset;
    (void)stressStateOffset;
    (void)baseTangentOffset;
    (void)stateDirectionOffset;
    (void)reconstructedOffset;
}
