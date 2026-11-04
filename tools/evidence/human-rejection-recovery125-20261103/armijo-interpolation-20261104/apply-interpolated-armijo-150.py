from pathlib import Path
import tarfile
R=Path('/Users/numi/Work/skin-fat-native-integration-209')
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125')
rels=['matter/src/metal/contact.metalinc','matter/tools/fgmres_true_residual_native_check.mm']
with tarfile.open(D/'interpolated-armijo-preimage-150.tar.gz','w:gz') as t:
 for rel in rels:t.add(R/rel,arcname=rel)
p=R/rels[0];s=p.read_text()
anchor='// Offset only the numerical geometry boundary,'
assert s.count(anchor)==1
helper='''// Safeguarded quadratic interpolation proposes a smaller step; the existing
// exact endpoint energy, CCD and final certificates still decide acceptance.
// Normalize by alpha*slope rather than squaring a possibly tiny step.
inline float contactInterpolatedArmijoAlpha(MRCompensatedScalar energyDelta,
    float slope,float alpha) {
    if(!(alpha>0.0f)||!isfinite(alpha))return 0.0f;
    const float fallback=0.5f*alpha,linear=alpha*slope;
    const float delta=energyDelta.high+energyDelta.low;
    if(!(slope<0.0f)||!isfinite(linear)||!(linear<0.0f)||
       !isfinite(delta))return fallback;
    const float denominator=1.0f-delta/linear;
    if(!(denominator>0.0f)||!isfinite(denominator))return fallback;
    const float fraction=0.5f/denominator;
    if(!isfinite(fraction))return fallback;
    return alpha*clamp(fraction,0.1f,0.5f);
}
kernel void nm_contact_armijo_interpolation_check(
    device const float4* input [[buffer(0)]],
    device float2* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=count)return;
    const auto v=input[i];
    const float next=contactInterpolatedArmijoAlpha({v.x,v.y},v.z,v.w);
    // An independent manufactured stiff quadratic exercises the production
    // proposal under the unchanged 8-evaluation Armijo acceptance rule.
    const float curvature=v.x,slope=-1.0f;float alpha=1.0f,accepted=0.0f;
    for(uint trial=0u;trial<8u;++trial) {
        const float delta=slope*alpha+0.5f*curvature*alpha*alpha;
        if(isfinite(delta)&&delta<=1.0e-4f*alpha*slope){accepted=alpha;break;}
        alpha=contactInterpolatedArmijoAlpha({delta,0.0f},slope,alpha);
    }
    output[i]=float2(next,accepted);
}

'''
s=s.replace(anchor,helper+anchor,1)
old='''                for (uint trial = 0u; trial < lineSearchSteps; ++trial) {
                    const float alpha = startAlpha * exp2(-float(trial));
                    float3 trialNodePosition[6];'''
new='''                float nextTrialAlpha=startAlpha;
                for (uint trial = 0u; trial < lineSearchSteps; ++trial) {
                    const float alpha=nextTrialAlpha;
                    nextTrialAlpha=0.5f*alpha;
                    float3 trialNodePosition[6];'''
assert s.count(old)==1;s=s.replace(old,new,1)
old='''                    if (captureThisTrial)
                        trialTraceRecords[trialTraceIndex].outcome.y = !finiteTrial ? 5u : 6u;'''
new='''                    if(referenceModeContact&&finiteTrial)
                        nextTrialAlpha=contactInterpolatedArmijoAlpha(
                            contactCompensatedSubtract(trialEnergyWords,currentEnergyWords),
                            directionalDerivative,alpha);
                    if (captureThisTrial)
                        trialTraceRecords[trialTraceIndex].outcome.y = !finiteTrial ? 5u : 6u;'''
assert s.count(old)==1;s=s.replace(old,new,1);p.write_text(s)
p=R/rels[1];s=p.read_text()
anchor='void checkFEMAlignedBarrier('
a=s.index(anchor)
fn='''void checkContactArmijoInterpolation(id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLLibrary> library) {
    const float nan=std::numeric_limits<float>::quiet_NaN();
    std::vector<nm_float4> input;
    for(float curvature:{1.0f,1.0e2f,1.0e4f,1.0e6f})
        for(float scale:{1.0f,1.0e-15f})
            input.push_back({curvature*scale,0.0f,-scale,1.0f});
    input.push_back({nan,0,-1,1});
    input.push_back({1,0,0,1});
    input.push_back({-1,0,-1,1});
    const unsigned count=unsigned(input.size());
    auto out=makeBuffer(device,std::vector<nm_float2>(count+1,{-7,-7}));
    dispatch(device,queue,pipeline(device,library,"numi_matter_metal::nm_contact_armijo_interpolation_check"),{
        {0u,makeBuffer(device,input)},{1u,out},
        {2u,makeBuffer(device,std::vector<unsigned>{count})}},1u,32u);
    const auto result=readBuffer<nm_float2>(out,count+1);
    for(unsigned i=0;i<count;++i) {
        const auto v=input[i];const double delta=double(v.x)+v.y;
        const double linear=double(v.w)*v.z;
        double fraction=0.5;
        if(v.z<0&&std::isfinite(delta)&&linear<0&&std::isfinite(linear)) {
            const double denominator=1-delta/linear;
            if(denominator>0&&std::isfinite(denominator))
                fraction=std::clamp(0.5/denominator,0.1,0.5);
        }
        require(std::isfinite(result[i].x)&&
            std::abs(result[i].x-v.w*fraction)<2.0e-7,"Armijo interpolation Float64 reference");
        require(result[i].x>=0.1f*v.w&&result[i].x<=0.5f*v.w,
            "Armijo proposal escaped safeguarded decreasing interval");
        if(std::isfinite(v.x)&&v.x>0) {
            require(result[i].y>0,"stiff quadratic did not find a certified step in eight evaluations");
            const double alpha=result[i].y;
            const double energyDelta=-alpha+0.5*double(v.x)*alpha*alpha;
            require(energyDelta<=-1.0e-4*alpha+1.0e-12,"quadratic Armijo was bypassed");
        }
    }
    bool dyadicAccepted=false;
    for(unsigned j=0;j<8;++j) {
        const double alpha=std::exp2(-double(j));
        dyadicAccepted=dyadicAccepted||(-alpha+0.5*1.0e6*alpha*alpha<=-1.0e-4*alpha);
    }
    require(!dyadicAccepted,"stiff regression no longer distinguishes fixed dyadic search");
    require(result[count].x==-7&&result[count].y==-7,"Armijo interpolation output canary");
    std::cout<<"contact_armijo_interpolation=true Float64_reference=true fixed_evaluation_budget=8 stiff_quadratic=true invalid_data_halving=true canary=true cases="<<count<<" PASS\\\\n";
}

'''
s=s[:a]+fn+s[a:]
anchor='''            if(argc==2&&std::string(argv[1])=="--fem-aligned-barrier-only") {'''
new='''            if(argc==2&&std::string(argv[1])=="--contact-armijo-interpolation-only") {
                checkContactArmijoInterpolation(device,queue,library);return 0;
            }
'''+anchor
assert s.count(anchor)==1;s=s.replace(anchor,new,1)
anchor=chr(10)+'            checkFEMAlignedBarrier(device,queue,library);'
assert s.count(anchor)==1
s=s.replace(anchor,chr(10)+'            checkContactArmijoInterpolation(device,queue,library);'+anchor,1)
s=s.replace('\\\\n','\\n')
p.write_text(s)
print('applied interpolated armijo150')

