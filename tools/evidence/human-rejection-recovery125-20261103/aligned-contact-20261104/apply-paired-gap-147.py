from pathlib import Path
import tarfile,json,hashlib
R=Path('/Users/numi/Work/skin-fat-native-integration-209')
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125')
paths=['matter/src/metal/contact.metalinc','matter/src/metal/mixed_fem.metalinc','matter/tools/fgmres_true_residual_native_check.mm']
with tarfile.open(D/'paired-gap-preimage-147.tar.gz','w:gz') as t:
 for rel in paths:t.add(R/rel,arcname=rel)
p=R/paths[0];s=p.read_text()
a=s.index('inline float4 nmFEMClearanceAlignedResponse(float separation')
b=s.index('// Direct arithmetic check',a)
old=s[a:b]
body=old[old.index('    const float width='):old.rfind('}')].replace('const float width=thickness-offset,gap=separation-offset;','const float width=thickness-offset;')
new='''inline float4 nmFEMClearanceAlignedResponseAtGap(float gap,float thickness,
    float inverseResponse,float dt,float closing,float offset) {
'''+body+'''}
inline float4 nmFEMClearanceAlignedResponse(float separation,float thickness,
    float inverseResponse,float dt,float closing,float offset) {
    if(offset==0.0f)
        return ipcBarrierResponse(separation,thickness,inverseResponse,dt,closing,false,0.0f,0.0f);
    if(!isfinite(offset)||offset<0.0f||!(thickness>offset)||!(separation>offset))
        return float4(NAN);
    if(!(inverseResponse>0.0f)||!(dt>0.0f))return float4(0.0f,0.0f,thickness,0.0f);
    return nmFEMClearanceAlignedResponseAtGap(separation-offset,thickness,
        inverseResponse,dt,closing,offset);
}
// Subtract the clearance before projecting to Float32. Rounding the world
// distance first loses the small gap that the paired Armijo energy retains.
inline float4 nmFEMClearanceAlignedResponse(MRCompensatedScalar separation,
    float thickness,float inverseResponse,float dt,float closing,float offset) {
    if(offset==0.0f)return nmFEMClearanceAlignedResponse(
        separation.high+separation.low,thickness,inverseResponse,dt,closing,offset);
    const auto gapWords=mrCompensatedAdd(separation,{-offset,0.0f});
    const float gap=gapWords.high+gapWords.low;
    if(!isfinite(offset)||offset<0.0f||!(thickness>offset)||
       !isfinite(gap)||!(gap>0.0f))return float4(NAN);
    if(!(inverseResponse>0.0f)||!(dt>0.0f))return float4(0.0f,0.0f,thickness,0.0f);
    return nmFEMClearanceAlignedResponseAtGap(gap,thickness,
        inverseResponse,dt,closing,offset);
}

'''
s=s[:a]+new+s[b:]
s=s.replace('nmFEMClearanceAlignedResponse(a.x,a.y,p.x,p.y,p.z,a.z);','nmFEMClearanceAlignedResponse(MRCompensatedScalar{a.x,a.w},a.y,p.x,p.y,p.z,a.z);',1)
s=s.replace('nmFEMClearanceAlignedEnergy(a.x,a.y,response.w,a.z),0.0f);','a.w==0.0f?nmFEMClearanceAlignedEnergy(a.x,a.y,response.w,a.z):energy.high+energy.low,0.0f);',1)
anchor='// Optional quasi-Newton metric.'
assert s.count(anchor)==1
helper='''// The same nearest-feature geometry and paired length as the Armijo energy.
// Used only for the tiny aligned-gap tail; no extra whole-body scan or storage.
inline bool contactCompensatedFeatureDistance(uint kind,
    thread const MRCompensatedPositionGPU* position,
    thread MRCompensatedScalar& distance) {
    MRCompensatedPositionGPU relative;
    if(kind==NM_DEFORMABLE_CONTACT_POINT_POINT) {
        relative=mrCompensatedVectorAdd(position[0],mrCompensatedVectorNegate(position[1]));
    } else if(kind==NM_DEFORMABLE_CONTACT_VERTEX_TRIANGLE) {
        float3 bary,closest;float projected;
        if(!contactCompensatedVertexTriangleWitness(position[0],position[1],
            position[2],position[3],bary,closest,relative,projected))return false;
    } else if(kind==NM_DEFORMABLE_CONTACT_EDGE_EDGE) {
        float first,second,projected;float3 p0,p1;
        if(!contactCompensatedSegmentWitness(position[0],position[1],
            position[2],position[3],first,second,p0,p1,relative,projected))return false;
    } else return false;
    const auto squared=mrCompensatedVectorDot(relative,relative);
    if(contactCompensatedSign(squared)<=0 ||
       !isfinite(squared.high)||!isfinite(squared.low))return false;
    distance=mrCompensatedSqrt(squared);
    return isfinite(distance.high)&&isfinite(distance.low);
}

'''
s=s.replace(anchor,helper+anchor,1)
old='''    const float4 baseBarrier = nmFEMClearanceAlignedResponse(
        bestSeparation, barrierThickness, inverseResponse, contactTimestep,
        max(-dot(bestNormal, relativeVelocity), 0.0f), barrierClearance);
    const float baseBarrierEnergy = nmFEMClearanceAlignedEnergy(
        bestSeparation, barrierThickness, baseBarrier.w, barrierClearance);
'''
new='''    MRCompensatedScalar barrierSeparation={bestSeparation,0.0f};
    float resolvedGap=0.0f;
    const bool resolveTinyGap=barrierClearance>0.0f &&
        bestSeparation-barrierClearance<1.0e-4f*(barrierThickness-barrierClearance);
    if(resolveTinyGap) {
        MRCompensatedPositionGPU feature[4];
        if(!nmMixedReferenceContactFeatures(dispatch,contact,femNodes,environment,feature)||
           !contactCompensatedFeatureDistance(contact.identity.z,feature,barrierSeparation)) {
            recordFailure(statuses[environment],NM_STATUS_CONTACT_FAILURE,
                contact.identity.x,candidateGlobal,float4(bestSeparation,barrierThickness,barrierClearance,43.0f));
            return;
        }
        const auto gap=mrCompensatedAdd(barrierSeparation,{-barrierClearance,0.0f});
        resolvedGap=gap.high+gap.low;
    }
    const float4 baseBarrier = resolveTinyGap
        ? nmFEMClearanceAlignedResponse(barrierSeparation,barrierThickness,
            inverseResponse,contactTimestep,max(-dot(bestNormal,relativeVelocity),0.0f),barrierClearance)
        : nmFEMClearanceAlignedResponse(bestSeparation,barrierThickness,
            inverseResponse,contactTimestep,max(-dot(bestNormal,relativeVelocity),0.0f),barrierClearance);
    const auto pairedEnergy=resolveTinyGap
        ? ipcFEMBarrierEnergyCompensated(barrierSeparation,barrierThickness,baseBarrier.w,barrierClearance)
        : MRCompensatedScalar{0.0f,0.0f};
    const float baseBarrierEnergy=resolveTinyGap ? pairedEnergy.high+pairedEnergy.low
        : nmFEMClearanceAlignedEnergy(bestSeparation,barrierThickness,baseBarrier.w,barrierClearance);
'''
assert s.count(old)==1;s=s.replace(old,new,1)
s=s.replace('const float clearance = cached.barrierHessianRow2.w;','const float clearance = cached.barrierHessianRow2.w;\n        const float resolvedGap = cached.barrierHessianRow0.w;',1)
s=s.replace('cached.barrierHessianRow2.w = clearance;','cached.barrierHessianRow2.w = clearance;\n        cached.barrierHessianRow0.w = resolvedGap;',1)
s=s.replace('contact.barrierHessianRow2.w=barrierClearance;','contact.barrierHessianRow2.w=barrierClearance;\n    // Internal metadata, xyz remains the physical PSD metric.\n    contact.barrierHessianRow0.w=resolvedGap;',1)
p.write_text(s)
p=R/paths[1];s=p.read_text()
old='''inline float nmMixedDeformableContactEnergy(const NMDeformableContactGPU c) {
    return'''
new='''inline float nmMixedDeformableContactEnergy(const NMDeformableContactGPU c) {
    if(nmFEMClearanceAlignedBarrierActive&&c.barrierHessianRow2.w>0.0f&&
       c.barrierHessianRow0.w>0.0f) {
        const auto energy=nmFEMPositiveGapIPCEnergy(
            {c.barrierHessianRow0.w,0.0f},c.barrier.z-c.barrierHessianRow2.w,c.barrier.w);
        return energy.high+energy.low;
    }
    return'''
assert s.count(old)==1;s=s.replace(old,new,1);p.write_text(s)
p=R/paths[2];s=p.read_text()
anchor='    const unsigned validCount=unsigned(input.size());'
pos=s.index(anchor,s.index('void checkFEMAlignedBarrier('))
extra='''    // Paired distances whose positive gap is lost or distorted by projecting
    // the distance before subtracting clearance, including the native141 scale.
    for(float low:{2.0e-17f,5.0e-17f,7.0e-16f,1.0e-19f}) {
        input.push_back({1.0e-9f,2.0e-9f,1.0e-9f,low});
        parameters.push_back({6.76168e12f,1.5625e-5f,0.0f,0.0f});
    }
'''
s=s[:pos]+extra+s[pos:]
a=s.index('void checkFEMAlignedBarrier(');b=s.index('\n\nint main',a)
body=s[a:b]
body=body.replace('const double gap=double(a.x)-a.z,','const double gap=(double(a.x)-a.z)+a.w,')
body=body.replace('contacts[i].barrierHessianRow2.w=input[i].z;','''contacts[i].barrierHessianRow2.w=input[i].z;
        if(input[i].w!=0.0f)contacts[i].barrierHessianRow0.w=
            float((double(input[i].x)-input[i].z)+input[i].w);''')
body=body.replace('positive_gap_no_flat_curvature=true ','positive_gap_no_flat_curvature=true paired_response_and_certificate=true ')
s=s[:a]+body+s[b:];p.write_text(s)
print('applied paired-gap-147')

