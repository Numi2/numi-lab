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

// Bounded, read-only replay of sealed FEM linearization capture rows.  The
// direct stress and direct algorithmic tangent paths are evaluated separately
// so a failure in either can be localized without changing solver state.
// Input stride: Fhi[9], Flo[9], Fdot[9], dF[9], dFdot[9], dt, temperature.
// Output stride 65: direct-valid, tangent-valid, projection-converged,
// residual, state norm, iterations, flags, failure, P_direct[9],
// P_tangent[9], dP[9], state-hi[15], state-lo[15].
kernel void cardboard_bilinear15_capture_batch(
    constant NMMaterialGPU* materials [[buffer(0)]],
    device const NMScalarProgramGPU* programs [[buffer(1)]],
    device const NMExpressionInstructionGPU* instructions [[buffer(2)]],
    device const float* parameters [[buffer(3)]],
    device const float* acceptedStates [[buffer(4)]],
    device const float* inputs [[buffer(5)]],
    device const uint* materialIndices [[buffer(6)]],
    device float* output [[buffer(7)]],
    constant uint& recordCount [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
    if(index>=recordCount) return;
    using namespace numi_matter_metal;
    constexpr uint inputStride=47u;
    constexpr uint acceptedStride=15u;
    constexpr uint outputStride=65u;
    const uint materialIndex=materialIndices[index];
    const uint inputBase=index*inputStride;
    const uint stateBase=index*acceptedStride;
    const uint outputBase=index*outputStride;
    const NMMaterialGPU material=materials[materialIndex];
    float f[9],fLow[9],fDot[9],direction[9],rateDirection[9];
    for(uint i=0u;i<9u;++i){
        f[i]=inputs[inputBase+i];
        fLow[i]=inputs[inputBase+9u+i];
        fDot[i]=inputs[inputBase+18u+i];
        direction[i]=inputs[inputBase+27u+i];
        rateDirection[i]=inputs[inputBase+36u+i];
    }
    const float dt=inputs[inputBase+45u];
    const float temperature=inputs[inputBase+46u];
    device const float* accepted=acceptedStates+stateBase;

    bool stressValid=true;
    const float3x3 directStress=evaluateAlgorithmicStress(
        material,programs,instructions,parameters,accepted,
        componentsMatrix(f),componentsMatrix(fDot),dt,temperature,stressValid,fLow);
    float directP[9]; matrixComponents(directStress,directP);

    bool tangentValid=true;
    uint flags=0u,iterations=0u,failure=0u;
    float residual=-1.0f,stateNorm=0.0f;
    float projected[NM_MAX_MATERIAL_STATE]={0.0f};
    float projectedLow[NM_MAX_MATERIAL_STATE]={0.0f};
    float3x3 tangentStress(0.0f),tangent(0.0f);
    evaluateAlgorithmicStressTangent(
        material,programs,instructions,parameters,accepted,
        componentsMatrix(f),componentsMatrix(direction),componentsMatrix(fDot),
        componentsMatrix(rateDirection),dt,temperature,
        tangentStress,tangent,tangentValid,projected,&flags,&residual,&stateNorm,
        &iterations,&failure,fLow,projectedLow);
    float tangentP[9],tangentFlat[9];
    matrixComponents(tangentStress,tangentP);
    matrixComponents(tangent,tangentFlat);
    output[outputBase+0u]=stressValid?1.0f:0.0f;
    output[outputBase+1u]=tangentValid?1.0f:0.0f;
    output[outputBase+2u]=(flags&kMaterialDiagProjectionConverged)!=0u?1.0f:0.0f;
    output[outputBase+3u]=residual;
    output[outputBase+4u]=stateNorm;
    output[outputBase+5u]=float(iterations);
    output[outputBase+6u]=float(flags);
    output[outputBase+7u]=float(failure);
    for(uint i=0u;i<9u;++i){
        output[outputBase+8u+i]=directP[i];
        output[outputBase+17u+i]=tangentP[i];
        output[outputBase+26u+i]=tangentFlat[i];
    }
    for(uint i=0u;i<15u;++i){
        output[outputBase+35u+i]=projected[i];
        output[outputBase+50u+i]=projectedLow[i];
    }
}

// Single-row companion: directly calls projectMaterialState with its bounded
// iteration trace, then independently invokes stress and tangent evaluation.
// It is intended for the exact env0/tet1528 failure point; callers must dispatch
// only one thread because the trace occupies 5992 floats of output.
kernel void cardboard_bilinear15_capture_trace(
    constant NMMaterialGPU* materials [[buffer(0)]],
    device const NMScalarProgramGPU* programs [[buffer(1)]],
    device const NMExpressionInstructionGPU* instructions [[buffer(2)]],
    device const float* parameters [[buffer(3)]],
    device const float* accepted [[buffer(4)]],
    device const float* inputs [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& materialIndex [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    if(index!=0u) return;
    using namespace numi_matter_metal;
    float f[9],fLow[9],fDot[9],direction[9],rateDirection[9];
    for(uint i=0u;i<9u;++i){
        f[i]=inputs[i];fLow[i]=inputs[9u+i];fDot[i]=inputs[18u+i];
        direction[i]=inputs[27u+i];rateDirection[i]=inputs[36u+i];
    }
    const float dt=inputs[45u],temperature=inputs[46u];
    const NMMaterialGPU material=materials[materialIndex];
    device const float* acceptedState=accepted;
    float projectionState[NM_MAX_MATERIAL_STATE]={0.0f};
    float projectionLow[NM_MAX_MATERIAL_STATE]={0.0f};
    float projectionResidual=-1.0f,projectionStateNorm=0.0f;
    uint projectionIterations=0u,projectionFlags=0u,projectionFailure=0u;
    float trace[kMaterialProjectionTraceSlots];
    const bool projectionOK=projectMaterialState(
        material,programs,instructions,parameters,acceptedState,
        componentsMatrix(f),componentsMatrix(fDot),dt,temperature,
        projectionState,&projectionResidual,&projectionStateNorm,
        &projectionIterations,&projectionFlags,&projectionFailure,
        fLow,projectionLow,trace);

    bool stressValid=true;
    const float3x3 directStress=evaluateAlgorithmicStress(
        material,programs,instructions,parameters,acceptedState,
        componentsMatrix(f),componentsMatrix(fDot),dt,temperature,stressValid,fLow);
    float directP[9];matrixComponents(directStress,directP);
    bool tangentValid=true;
    uint tangentFlags=0u,tangentIterations=0u,tangentFailure=0u;
    float tangentResidual=-1.0f,tangentStateNorm=0.0f;
    float tangentState[NM_MAX_MATERIAL_STATE]={0.0f};
    float tangentLow[NM_MAX_MATERIAL_STATE]={0.0f};
    float3x3 tangentStress(0.0f),tangent(0.0f);
    evaluateAlgorithmicStressTangent(
        material,programs,instructions,parameters,acceptedState,
        componentsMatrix(f),componentsMatrix(direction),componentsMatrix(fDot),
        componentsMatrix(rateDirection),dt,temperature,
        tangentStress,tangent,tangentValid,tangentState,&tangentFlags,
        &tangentResidual,&tangentStateNorm,&tangentIterations,&tangentFailure,
        fLow,tangentLow);
    float tangentP[9],tangentFlat[9];
    matrixComponents(tangentStress,tangentP);matrixComponents(tangent,tangentFlat);

    constexpr uint summaryCount=73u;
    output[0u]=projectionOK?1.0f:0.0f;
    output[1u]=projectionResidual;
    output[2u]=projectionStateNorm;
    output[3u]=float(projectionIterations);
    output[4u]=float(projectionFlags);
    output[5u]=float(projectionFailure);
    output[6u]=stressValid?1.0f:0.0f;
    output[7u]=tangentValid?1.0f:0.0f;
    output[8u]=(tangentFlags&kMaterialDiagProjectionConverged)!=0u?1.0f:0.0f;
    output[9u]=tangentResidual;
    output[10u]=tangentStateNorm;
    output[11u]=float(tangentIterations);
    output[12u]=float(tangentFlags);
    output[13u]=float(tangentFailure);
    // Packing sentinels occupy the two slots after the 15-state high/low pair.
    // The host verifies their exact words before accepting the trace payload.
    output[71u]=0.15625f;
    output[72u]=0.3125f;
    for(uint i=0u;i<9u;++i){
        output[14u+i]=directP[i];output[23u+i]=tangentP[i];
        output[32u+i]=tangentFlat[i];
    }
    for(uint i=0u;i<15u;++i){
        output[41u+i]=projectionState[i];
        output[56u+i]=projectionLow[i];
    }
    for(uint i=0u;i<kMaterialProjectionTraceSlots;++i)
        output[summaryCount+i]=trace[i];
}
