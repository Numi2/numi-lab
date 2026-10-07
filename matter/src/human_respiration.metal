// The respiratory muscles use the existing MyoSim activation and compliant
// fibre/tendon implementation. No independent muscle constitutive law.
#include "MujocoMuscleReference.metal"
#include "numi/matter/human_respiration.h"
#include "RespiratoryChemoreflexV1.metal"
#include "metalrobo/numi_human_stand_gpu.h"
#include "metalrobo/numi_human_resting_visual_gpu.h"
#include "NumiHumanRestingCommonField.metalinc"
#include "NumiHumanRestingSupportGeometry.metalinc"

inline float3 restingRotate(float4 q,float3 v) {
    return v+2.0f*cross(q.xyz,cross(q.xyz,v)+q.w*v);
}
inline float4 restingRibRotation(constant MRHumanRestingAnatomyGPU& anatomy,uint rib,float excursion) {
    const float angle=excursion*anatomy.ribPivotAndGain[rib].w;
    return float4(anatomy.ribAxis[rib].xyz*sin(.5f*angle),cos(.5f*angle));
}
inline float3 restingRibPoint(constant MRHumanRestingAnatomyGPU& anatomy,uint rib,float4 rotation,float3 p) {
    const float3 pivot=anatomy.ribPivotAndGain[rib].xyz;return pivot+restingRotate(rotation,p-pivot);
}
kernel void nm_human_resting_common_coordinates(
    device const NMHumanRespirationState* respiration [[buffer(0)]],
    constant MRHumanRestingCommonFieldGPU& parameters [[buffer(1)]],
    device const MRHumanRestingCommonCoordinateBoxGPU* boxes [[buffer(2)]],
    device MRHumanRestingCommonCoordinatesGPU* output [[buffer(3)]],uint lane [[thread_position_in_grid]]) {
    if(lane)return;
    const auto state=respiration[0];
    const float targets[7]={state.chamberVolumes.x,state.chamberVolumes.y,state.chamberVolumes.z,
        state.chamberVolumes.w,parameters.materialTargetVolumes.x,parameters.materialTargetVolumes.y,
        parameters.materialTargetVolumes.z};
    float coordinates[7]={0,0,0,0,0,0,0},residual=INFINITY;
    uint iterations=0,matched=MR_INVALID_INDEX;
    const uint status=nmHumanRestingCommonSolve(parameters,boxes,targets,coordinates,iterations,matched,residual);
    output[0].first=status?float4(NAN):float4(coordinates[0],coordinates[1],coordinates[2],coordinates[3]);
    output[0].second=status?float4(NAN):float4(coordinates[4],coordinates[5],coordinates[6],0);
    output[0].status=uint4(status,iterations,matched,0);
    output[0].diagnostics=float4(residual,0,0,0);
}
kernel void nm_human_resting_common_coordinate_status_gate(
    device const MRHumanRestingCommonCoordinatesGPU* coordinates [[buffer(0)]],
    device NMMatterStatusGPU* statuses [[buffer(1)]],uint lane [[thread_position_in_grid]]) {
    if(lane==0u&&coordinates[0].status.x!=0u&&statuses[0].code==NM_STATUS_SUCCESS)
        statuses[0].code=NM_STATUS_MULTIPHYSICS_FAILURE;
}

inline float nmHumanRestingCardiacVolume(constant MRHumanRestingAnatomyGPU& anatomy,uint chamber,float q) {
    const float4 c=anatomy.chamberVolumePolynomial[chamber];return ((c.w*q+c.z)*q+c.y)*q+c.x;
}
// Four scalar roots are derived from the accepted hydraulic chamber volumes;
// this presentation field never writes back into the physiological state.
kernel void nm_human_resting_cardiac_q(
    device const NMHumanRespirationState* respiration [[buffer(0)]],
    constant MRHumanRestingAnatomyGPU& anatomy [[buffer(1)]],device float* qOut [[buffer(2)]],
    uint chamber [[thread_position_in_grid]]) {
    if(chamber>=4)return;
    const float target=respiration[0].chamberVolumes[chamber];
    float low=-.999f,high=1.0f;
    if(!isfinite(target)||nmHumanRestingCardiacVolume(anatomy,chamber,low)>target||
       nmHumanRestingCardiacVolume(anatomy,chamber,high)<target){qOut[chamber]=NAN;return;}
    for(uint iteration=0;iteration<28;++iteration){
        const float mid=.5f*(low+high);
        if(nmHumanRestingCardiacVolume(anatomy,chamber,mid)<target)low=mid;else high=mid;
    }
    qOut[chamber]=.5f*(low+high);
}
kernel void nm_human_resting_cardiac_wall_q(
    constant MRHumanRestingCardiacWallGPU& wall [[buffer(0)]],
    device const float* cardiacQ [[buffer(1)]],device float4* result [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if(i)return;
    if(!wall.chambersAndFlags.z){result[0]=0;return;}
    const float r=cardiacQ[wall.chambersAndFlags.x]/wall.scalesAndVolume.x;
    const float l=cardiacQ[wall.chambersAndFlags.y]/wall.scalesAndVolume.y;
    const float4 p0=wall.volumePolynomial[0],p1=wall.volumePolynomial[1],p2=wall.volumePolynomial[2],
        p3=wall.volumePolynomial[3],p4=wall.volumePolynomial[4];
    // Cubic source-mesh volume in normalized (r,l,s). This correction is
    // derived from accepted chamber q; it is not a second pump or a history.
    const float f0=p0.x+p0.y*r+p0.z*l+p0.w*r*r+p1.x*r*l+p1.y*l*l+
        p1.z*r*r*r+p1.w*r*r*l+p2.x*r*l*l+p2.y*l*l*l;
    const float f1=p2.z+p2.w*r+p3.x*l+p3.y*r*r+p3.z*r*l+p3.w*l*l;
    const float f2=p4.x+p4.y*r+p4.z*l,f3=p4.w;
    float low=wall.closureBoundsAndTolerance.x/wall.scalesAndVolume.z;
    float high=wall.closureBoundsAndTolerance.y/wall.scalesAndVolume.z;
    const float target=wall.scalesAndVolume.w;
    const float vlo=((f3*low+f2)*low+f1)*low+f0;
    const float vhi=((f3*high+f2)*high+f1)*high+f0;
    float minimumDerivative=min((3*f3*low+2*f2)*low+f1,(3*f3*high+2*f2)*high+f1);
    if(f3!=0) {
        const float stationary=-f2/(3*f3);
        if(stationary>low&&stationary<high)
            minimumDerivative=min(minimumDerivative,(3*f3*stationary+2*f2)*stationary+f1);
    }
    if(!all(isfinite(float4(vlo,vhi,minimumDerivative,target)))||target<=0||low>=high||
       minimumDerivative<=1.e-12f||vlo>target||vhi<target) {
        result[0]=float4(NAN,NAN,target,1);return;
    }
    for(uint iteration=0;iteration<32;++iteration) {
        const float mid=.5f*(low+high);
        if(((f3*mid+f2)*mid+f1)*mid+f0<target)low=mid;else high=mid;
    }
    const float s=.5f*(low+high),volume=((f3*s+f2)*s+f1)*s+f0;
    const float error=abs(volume-target)/target;
    result[0]=float4(s*wall.scalesAndVolume.z,volume,target,
        !isfinite(error)||error>wall.closureBoundsAndTolerance.z?1.0f:0.0f);
}

kernel void nm_human_resting_skin(
    constant uint4& d [[buffer(0)]], device const MRHumanRestingVertexMap* map [[buffer(1)]],
    device const MRHumanRestingInfluence* influences [[buffer(2)]], device const MRBodyStateGPU* bodies [[buffer(3)]],
    device MRVisualVertexGPUV2* vertices [[buffer(4)]],
    device const NMHumanRespirationState* respiration [[buffer(5)]],
    constant MRHumanRestingAnatomyGPU& anatomy [[buffer(6)]],device const float* cardiacQ [[buffer(7)]],
    device const MRHumanRestingCardiacWallVertexGPU* wallMap [[buffer(8)]],
    constant MRHumanRestingCardiacWallGPU& wall [[buffer(9)]],device const float4* wallQ [[buffer(10)]],
    device const MRHumanRestingCommonFieldVertexGPU* commonMap [[buffer(11)]],
    device const MRHumanRestingCommonCoordinatesGPU* commonCoordinates [[buffer(12)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=d.x)return;
    auto m=map[i];if(!m.influenceCount)return;
    const auto soleInfluence=influences[m.firstInfluence];
    const bool sourceLocalDeformation=m.deformationKind==1u||m.deformationKind==3u||
        m.deformationKind==4u||m.deformationKind==9u||m.deformationKind==11u||m.deformationKind==12u;
    const bool exactBodyLocal=sourceLocalDeformation&&m.influenceCount==1u&&
        soleInfluence.body.x==anatomy.bodyAndFlags.x&&soleInfluence.positionAndWeight.w==1.0f;
    float3 p=0,n=0,strongest=0;float strongestWeight=-1;
    for(uint j=0;j<m.influenceCount;++j) {
        auto influence=influences[m.firstInfluence+j];float w=influence.positionAndWeight.w;
        float3 ni;
        if(influence.body.x==MR_INVALID_INDEX) {
            p+=w*influence.positionAndWeight.xyz;ni=influence.normal.xyz;
        }else {
            auto body=bodies[influence.body.x];
            p+=w*(body.position.xyz+restingRotate(body.orientation,influence.positionAndWeight.xyz));
            ni=restingRotate(body.orientation,influence.normal.xyz);
        }
        n+=w*ni;if(w>strongestWeight){strongestWeight=w;strongest=ni;}
    }
    if(m.deformationKind && anatomy.bodyAndFlags.y) {
        auto body=bodies[anatomy.bodyAndFlags.x];
        const float4 inverse=float4(-body.orientation.xyz,body.orientation.w);
        float3 local=exactBodyLocal?soleInfluence.positionAndWeight.xyz:
            restingRotate(inverse,p-body.position.xyz);
        float3 normal=exactBodyLocal?soleInfluence.normal.xyz:restingRotate(inverse,n);
        const auto state=respiration[0];
        if(m.deformationKind==2) {
            const float4 cavity=anatomy.chamberCenterAndVolume[m.chamberIndex];
            const float weight=m.deformationWeight.x;
            if(weight!=0.0f) {
                const float scale=1.0f+cardiacQ[m.chamberIndex]*weight;
                local=cavity.xyz+scale*(local-cavity.xyz);
                normal/=scale;
            }
        } else if(m.deformationKind==12) {
            const auto solved=commonCoordinates[0];
            if(solved.status.x!=0u) local=float3(NAN);
            else {
                const auto basis=commonMap[m.chamberIndex];
                const float coordinate[7]={solved.first.x,solved.first.y,solved.first.z,solved.first.w,
                    solved.second.x,solved.second.y,solved.second.z};
                for(uint c=0;c<7;++c) local+=basis.displacement[c].xyz*coordinate[c];
            }
        } else if(m.deformationKind==10) {
            const auto basis=wallMap[m.chamberIndex];
            local+=basis.first.xyz*cardiacQ[wall.chambersAndFlags.x]+
                basis.second.xyz*cardiacQ[wall.chambersAndFlags.y]+basis.closure.xyz*wallQ[0].x;
            if(wallQ[0].w!=0)local=float3(NAN);
            // Normals are rebuilt from the deformed source triangles below.
        } else if(m.deformationKind==8) {
            const float q=cardiacQ[m.chamberIndex],weight=m.deformationWeight.w;
            const float scale=1.0f+q*weight;
            local+=m.deformationWeight.xyz*q*weight;
            const float radialSquared=dot(m.deformationWeight.xyz,m.deformationWeight.xyz);
            if(radialSquared>1.e-12f) {
                const float3 radial=m.deformationWeight.xyz*rsqrt(radialSquared);
                normal+=radial*dot(normal,radial)*(1.0f/scale-1.0f);
            }
        } else if(m.deformationKind==5||m.deformationKind==7) {
            const float excursion=state.motion.y/anatomy.muscleAreas.y;
            const float4 rotation=restingRibRotation(anatomy,m.chamberIndex,excursion);
            float3 mapped=restingRibPoint(anatomy,m.chamberIndex,rotation,local);
            float3 mappedNormal=restingRotate(rotation,normal);
            if(m.deformationKind==7) {
                const uint other=uint(m.deformationWeight.y);
                const float4 otherRotation=restingRibRotation(anatomy,other,excursion);
                mapped=mix(mapped,restingRibPoint(anatomy,other,otherRotation,local),m.deformationWeight.z);
                mappedNormal=mix(mappedNormal,restingRotate(otherRotation,normal),m.deformationWeight.z);
            }
            local=mapped;normal=mappedNormal;
        } else if(m.deformationKind==6) {
            local+=anatomy.anteriorAxis.xyz*(state.motion.y/anatomy.muscleAreas.y);
        } else {
            const float3 axis=anatomy.superiorAxisAndHeight.xyz;
            const float basalWeight=m.respiratoryBasis.w;
            const float3 gradient=m.respiratoryBasis.xyz;
            const float displacement=state.motion.x/anatomy.lungBasalBlend.z;
            // This common superior-only field contributes exactly qD to the
            // five closed lobe volumes. Radial expansion contributes only qR.
            // Shared fissure vertices use the same map and remain shared.
            const float afterDiaphragm=anatomy.lungAnchorAndVolume.w+state.motion.x;
            const float radial=sqrt((afterDiaphragm+state.motion.y)/afterDiaphragm);
            const float3 offset=local-anatomy.lungAnchorAndVolume.xyz;
            const float3 along=dot(offset,axis)*axis,across=offset-along;
            // Passive viscera share the diaphragm field cranially, tapering
            // toward their common pelvic attachment. They add no forces,
            // physical mass or independent physiological state.
            const float weight=m.deformationKind==11?m.deformationWeight.x:
                ((m.deformationKind==3||m.deformationKind==9)?m.deformationWeight.x:1.0f);
            const float3 mapped=local-axis*(displacement*basalWeight)+(radial-1.0f)*across;
            local=mix(local,mapped,weight);
            const float3 normalAlong=dot(normal,axis)*axis;
            const float axialJacobian=1.0f-weight*displacement*dot(gradient,axis);
            normal=normalAlong/axialJacobian+
                (normal-normalAlong+weight*displacement*dot(normal,axis)/axialJacobian*
                 (gradient-dot(gradient,axis)*axis))/mix(1.0f,radial,weight);
        }
        p=body.position.xyz+restingRotate(body.orientation,local);
        n=restingRotate(body.orientation,normal);
    }
    vertices[i].position=float4(p,1);
    const float3 unitNormal=normalize(dot(n,n)>1e-12f?n:strongest);
    vertices[i].normalAndTangentSign=float4(unitNormal,1);
    const float3 tangentAxis=abs(unitNormal.z)<.9f?float3(0,0,1):float3(0,1,0);
    vertices[i].tangent=float4(normalize(cross(tangentAxis,unitNormal)),0);
}
kernel void nm_human_resting_cardiac_wall_normals(
    constant uint& count [[buffer(0)]],device const uint4* ranges [[buffer(1)]],
    device const uint* incidentTriangleStarts [[buffer(2)]],device const uint* indices [[buffer(3)]],
    device MRVisualVertexGPUV2* vertices [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    if(i>=count)return;
    const uint4 row=ranges[i];float3 normal=0;
    for(uint j=row.y;j<row.y+row.z;++j) {
        const uint at=incidentTriangleStarts[j];
        const float3 a=vertices[indices[at]].position.xyz,b=vertices[indices[at+1]].position.xyz,
            c=vertices[indices[at+2]].position.xyz;
        normal+=cross(b-a,c-a);
    }
    const float squared=dot(normal,normal);
    if(!isfinite(squared)||squared<=1.e-30f){vertices[row.x].normalAndTangentSign=float4(NAN);return;}
    normal*=rsqrt(squared);vertices[row.x].normalAndTangentSign=float4(normal,1);
    const float3 tangentAxis=abs(normal.z)<.9f?float3(0,0,1):float3(0,1,0);
    vertices[row.x].tangent=float4(normalize(cross(tangentAxis,normal)),0);
}
kernel void nm_human_resting_layers(
    constant uint4& d [[buffer(0)]], device MRVisualInstanceGPUV2* instances [[buffer(1)]],
    device const uint* layerMask [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=d.y)return;
    bool visible=d.z<7 && (layerMask[i]&(1u<<d.z));
    uint bits=MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR|MR_VISUAL_INSTANCE_CASTS_SHADOW|MR_VISUAL_INSTANCE_RECEIVES_SHADOW;
    instances[i].binding.w=visible?bits:0;
}

// Parallelize each functional surface volume without dropping any source
// triangle. The CPU-built descriptors assign each group a bounded contiguous
// range; 256 lanes each evaluate one triangle and reduce compact partials.
kernel void nm_human_resting_audit_volume_partials(
    constant uint4& d [[buffer(0)]],
    device const MRHumanRestingSurfaceAuditGPU* surfaces [[buffer(1)]],
    device const uint* indices [[buffer(2)]],
    device const MRVisualVertexGPUV2* vertices [[buffer(3)]],
    device const MRHumanRestingVolumeAuditGroupGPU* groups [[buffer(4)]],
    device MRHumanRestingVolumeAuditPartialGPU* partials [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    if(group.x>=d.x)return;
    const auto descriptor=groups[group.x].surfaceAndTriangleRange;
    const auto surface=surfaces[descriptor.x];
    const uint4 owner=surface.indicesAndOwner;
    const float3 origin=vertices[indices[owner.x]].position.xyz;
    threadgroup float volumes[MR_HUMAN_RESTING_VOLUME_AUDIT_GROUP_THREADS];
    threadgroup uint invalid[MR_HUMAN_RESTING_VOLUME_AUDIT_GROUP_THREADS];
    threadgroup uint first[MR_HUMAN_RESTING_VOLUME_AUDIT_GROUP_THREADS];
    float volume=0;uint invalidCount=0;
    uint firstInvalid=MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE;
    for(uint local=lane;local<descriptor.z;local+=MR_HUMAN_RESTING_VOLUME_AUDIT_GROUP_THREADS) {
        const uint triangle=descriptor.y+local;
        const uint j=owner.x+3u*triangle;
        const float3 pa=vertices[indices[j]].position.xyz;
        const float3 pb=vertices[indices[j+1u]].position.xyz;
        const float3 pc=vertices[indices[j+2u]].position.xyz;
        const float3 area=cross(pb-pa,pc-pa);
        const bool bad=!all(isfinite(area));
        const bool collapsed=!bad&&all(area==float3(0));
        if(bad||collapsed) {++invalidCount;firstInvalid=min(firstInvalid,triangle);}
        const float3 a=pa-origin,b=pb-origin,c=pc-origin;
        volume+=dot(a,cross(b,c))/6.0f;
    }
    volumes[lane]=volume;invalid[lane]=invalidCount;first[lane]=firstInvalid;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=MR_HUMAN_RESTING_VOLUME_AUDIT_GROUP_THREADS/2u;stride;stride>>=1u) {
        if(lane<stride) {
            volumes[lane]+=volumes[lane+stride];
            invalid[lane]+=invalid[lane+stride];
            first[lane]=min(first[lane],first[lane+stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(!lane)partials[group.x]={volumes[0],invalid[0],first[0],0u};
}

// Finish the per-surface compensated sum, preserve the original expected
// volume/status rules, and reconstruct the same first invalid-triangle witness.
kernel void nm_human_resting_reduce_volume_audits(
    constant uint4& d [[buffer(0)]],
    device const MRHumanRestingSurfaceAuditGPU* surfaces [[buffer(1)]],
    device const uint* indices [[buffer(2)]],
    device const MRVisualVertexGPUV2* vertices [[buffer(3)]],
    device const MRHumanRestingVolumeAuditRangeGPU* groupRanges [[buffer(4)]],
    device const MRHumanRestingVolumeAuditPartialGPU* partials [[buffer(5)]],
    device const NMHumanRespirationState* respiration [[buffer(6)]],
    constant MRHumanRestingAnatomyGPU& anatomy [[buffer(7)]],
    device float4* result [[buffer(8)]],
    constant MRHumanRestingCardiacWallGPU& wall [[buffer(9)]],
    device MRHumanRestingSurfaceFailureGPU* failureResults [[buffer(10)]],
    constant MRHumanRestingCommonFieldGPU& common [[buffer(11)]],
    device const MRHumanRestingCommonCoordinatesGPU* commonCoordinates [[buffer(12)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=d.y)return;
    const auto surface=surfaces[i];const uint4 owner=surface.indicesAndOwner;
    const uint4 groupRange=groupRanges[i].groupRange;
    float volume=0,compensation=0;uint invalidTriangles=0;
    uint firstInvalid=MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE;
    for(uint g=groupRange.x;g<groupRange.x+groupRange.y;++g) {
        const auto partial=partials[g];
        const float y=partial.signedVolume-compensation;
        const float next=volume+y;compensation=(next-volume)-y;volume=next;
        invalidTriangles+=partial.invalidTriangleCount;
        firstInvalid=min(firstInvalid,partial.firstInvalidTriangle);
    }
    MRHumanRestingSurfaceFailureGPU firstFailure;
    firstFailure.surfaceTriangleKind=uint4(MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE,0,0,0);
    firstFailure.vertexIndices=uint4(0);
    firstFailure.renderedPositions[0]=float4(0);
    firstFailure.renderedPositions[1]=float4(0);
    firstFailure.renderedPositions[2]=float4(0);
    if(firstInvalid<owner.y/3u) {
        const uint j=owner.x+3u*firstInvalid;
        const uint3 ids=uint3(indices[j],indices[j+1u],indices[j+2u]);
        const float3 pa=vertices[ids.x].position.xyz;
        const float3 pb=vertices[ids.y].position.xyz;
        const float3 pc=vertices[ids.z].position.xyz;
        const float3 area=cross(pb-pa,pc-pa);
        const uint areaFailure=!all(isfinite(area))?MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONFINITE_AREA:
            MR_HUMAN_RESTING_TRIANGLE_FAILURE_EXACT_ZERO_AREA;
        firstFailure.surfaceTriangleKind=uint4(i,firstInvalid,areaFailure,0);
        firstFailure.vertexIndices=uint4(ids,0);
        firstFailure.renderedPositions[0]=float4(pa,0);
        firstFailure.renderedPositions[1]=float4(pb,0);
        firstFailure.renderedPositions[2]=float4(pc,0);
    }
    const auto state=respiration[0];
    const float afterDiaphragm=anatomy.lungAnchorAndVolume.w+state.motion.x;
    const float commonExpected=owner.w<4u?state.chamberVolumes[owner.w]:common.materialTargetVolumes[owner.w-4u];
    const float expected=owner.z==10?wall.scalesAndVolume.w:owner.z==2?state.chamberVolumes[owner.w]:
        owner.z==12?commonExpected:(surface.reference.x+surface.reference.y*state.motion.x/anatomy.lungBasalBlend.z)*
        (1+state.motion.y/afterDiaphragm);
    const float relative=abs(abs(volume)-expected)/expected;
    const uint commonFailure=(owner.z==12u&&commonCoordinates[0].status.x!=0u)?4u:0u;
    const uint status=(!isfinite(relative)||relative>2.e-4f?1u:0u)|(invalidTriangles?2u:0u)|commonFailure;
    result[i]=float4(abs(volume),expected,relative,float(status));
    failureResults[i]=firstFailure;
}

// Whole native mesh validity is independent of which surfaces own a volume.
// Bounded parallel reductions leave only counts and the first witness on host.
kernel void nm_human_resting_audit_mesh_triangles(
    constant uint4& d [[buffer(0)]],device const uint* indices [[buffer(1)]],
    device const MRVisualVertexGPUV2* vertices [[buffer(2)]],
    device uint4* partials [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup uint zero[256],nonfinite[256],first[256];
    uint z=0,n=0,f=MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE;
    for(uint t=group.x*256u+lane;t<d.x;t+=d.y*256u) {
        const float3 a=vertices[indices[3u*t]].position.xyz;
        const float3 b=vertices[indices[3u*t+1u]].position.xyz;
        const float3 c=vertices[indices[3u*t+2u]].position.xyz;
        const float3 area=cross(b-a,c-a);
        const bool bad=!all(isfinite(area));
        const bool collapsed=!bad&&all(area==float3(0));
        z+=uint(collapsed);n+=uint(bad);
        if(bad||collapsed)f=min(f,t);
    }
    zero[lane]=z;nonfinite[lane]=n;first[lane]=f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=128;stride;stride>>=1) {
        if(lane<stride) {
            zero[lane]+=zero[lane+stride];nonfinite[lane]+=nonfinite[lane+stride];
            first[lane]=min(first[lane],first[lane+stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(!lane)partials[group.x]=uint4(zero[0],nonfinite[0],first[0],0);
}
kernel void nm_human_resting_reduce_mesh_audit(
    constant uint4& d [[buffer(0)]],device const uint4* partials [[buffer(1)]],
    device const uint* indices [[buffer(2)]],device const MRVisualVertexGPUV2* vertices [[buffer(3)]],
    device uint4* result [[buffer(4)]],device MRHumanRestingSurfaceFailureGPU* failure [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
    if(i)return;
    uint4 sum=uint4(0,0,MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE,d.x);
    for(uint j=0;j<d.y;++j) {sum.xy+=partials[j].xy;sum.z=min(sum.z,partials[j].z);}
    result[0]=sum;
    MRHumanRestingSurfaceFailureGPU witness;
    witness.surfaceTriangleKind=uint4(MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE,0,0,0);
    witness.vertexIndices=uint4(0);
    for(uint k=0;k<3;++k)witness.renderedPositions[k]=float4(0);
    if(sum.z<d.x) {
        const uint3 ids=uint3(indices[3u*sum.z],indices[3u*sum.z+1u],indices[3u*sum.z+2u]);
        const float3 a=vertices[ids.x].position.xyz,b=vertices[ids.y].position.xyz,c=vertices[ids.z].position.xyz;
        const float3 area=cross(b-a,c-a);
        const uint kind=!all(isfinite(area))?MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONFINITE_AREA:
            MR_HUMAN_RESTING_TRIANGLE_FAILURE_EXACT_ZERO_AREA;
        witness.surfaceTriangleKind=uint4(0,sum.z,kind,0);
        witness.vertexIndices=uint4(ids,0);
        witness.renderedPositions[0]=float4(a,0);witness.renderedPositions[1]=float4(b,0);
        witness.renderedPositions[2]=float4(c,0);
    }
    failure[0]=witness;
}
kernel void nm_human_resting_audit_skin(
    constant uint4& d [[buffer(0)]], device const MRHumanRestingVertexMap* map [[buffer(1)]],
    device const MRVisualVertexGPUV2* vertices [[buffer(2)]],device float4* result [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float minimum[256];threadgroup uint owner[256],below[256],invalid[256];
    float value=INFINITY;uint index=MR_INVALID_INDEX,count=0,nonfinite=0;
    for(uint i=lane;i<d.x;i+=256) {
        if(!all(isfinite(vertices[i].position))||!all(isfinite(vertices[i].normalAndTangentSign)))++nonfinite;
        if(map[i].deformationKind!=3)continue;
        const float z=vertices[i].position.z;
        if(z<value){value=z;index=i;}if(z<-.001f)++count;
    }
    minimum[lane]=value;owner[lane]=index;below[lane]=count;invalid[lane]=nonfinite;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=128;stride;stride>>=1) {
        if(lane<stride) {
            if(minimum[lane+stride]<minimum[lane]){minimum[lane]=minimum[lane+stride];owner[lane]=owner[lane+stride];}
            below[lane]+=below[lane+stride];invalid[lane]+=invalid[lane+stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(lane==0)result[d.w]=float4(minimum[0],float(owner[0]),float(below[0]),float(invalid[0]));
}

// Presentation-only reduction of the same committed body poses used by the
// anatomy renderer. No extra body readback, alternate integrator, or state
// history; this compact trace distinguishes postural settling from COM drift.
kernel void nm_human_resting_audit_body(
    constant uint4& d [[buffer(0)]],device const MRBodyStateGPU* bodies [[buffer(1)]],
    device float4* result [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i)return;
    float4 sum=0,compensation=0;
    for(uint b=0;b<d.x;++b) {
        const auto body=bodies[b];
        const float inverseMass=body.linearVelocityAndInverseMass.w;
        if(inverseMass==0.0f)continue; // Source massless articulated links.
        if(!isfinite(inverseMass)||inverseMass<0||!all(isfinite(body.position.xyz))) {
            result[d.y]=float4(NAN);return;
        }
        const float mass=1.0f/inverseMass;
        const float4 term=float4(body.position.xyz*mass,mass)-compensation;
        const float4 next=sum+term;compensation=(next-sum)-term;sum=next;
    }
    result[d.y]=float4(sum.xyz/sum.w,sum.w);
}

kernel void nm_human_resting_prepare_world(
    constant uint4& d [[buffer(0)]], device const MRNumiHumanStandStatusGPU* body [[buffer(1)]],
    device MRMetalWorldStatusGPU* world [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if(i)return;
    MRMetalWorldStatusGPU s{};
    s.code=body[0].code==MR_NUMI_HUMAN_STAND_SUCCESS?MR_STEP_SUCCESS:MR_STEP_DID_NOT_CONVERGE;
    world[0]=s;(void)d;
}
kernel void nm_human_resting_validate_body(
    constant uint4& d [[buffer(0)]], device const MRNumiHumanStandStatusGPU* body [[buffer(1)]],
    device MRMetalWorldStatusGPU* world [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if(i)return;
    if(body[0].code!=MR_NUMI_HUMAN_STAND_SUCCESS||body[0].completedSteps!=d.x+1)
        world[0].code=MR_STEP_DID_NOT_CONVERGE;
}
kernel void nm_human_resting_validate_matter(
    constant uint4& d [[buffer(0)]], device MRNumiHumanStandStatusGPU* body [[buffer(1)]],
    device const MRMetalWorldStatusGPU* world [[buffer(2)]],
    device const NMMatterStatusGPU* matter [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if(i)return;
    if(body[0].code==MR_NUMI_HUMAN_STAND_SUCCESS&&
       (world[0].code!=MR_STEP_SUCCESS||matter[0].code!=NM_STATUS_SUCCESS)) {
        body[0].code=MR_NUMI_HUMAN_STAND_EXTERNAL_PHYSICS_FAILED;body[0].failingIndex=d.x;
    }
}
kernel void nm_human_resting_latch_common_failure(
    constant uint4& d [[buffer(0)]],
    device const MRNumiHumanStandStatusGPU* stand [[buffer(1)]],
    device const NMMatterStatusGPU* matter [[buffer(2)]],
    device const NMHumanRespirationState* respirationCandidate [[buffer(3)]],
    device const NMHumanRespirationState* respirationAccepted [[buffer(4)]],
    device const MRHumanRestingCommonCoordinatesGPU* commonCoordinates [[buffer(5)]],
    device const NBNumiRespiratoryChemoreflexInputV1* brainInput [[buffer(6)]],
    device const NBNumiRespiratoryChemoreflexStateV1* brainAccepted [[buffer(7)]],
    device const NBNumiRespiratoryChemoreflexStateV1* brainCandidate [[buffer(8)]],
    device const NBNumiRespiratoryChemoreflexOutputV1* brainOutput [[buffer(9)]],
    device const float4* excitation [[buffer(10)]],
    device MRHumanRestingCommonFailureGPU* latched [[buffer(11)]],
    constant MRHumanRestingCommonFieldGPU& commonParameters [[buffer(12)]],
    device const MRHumanRestingCommonCoordinateBoxGPU* commonBoxes [[buffer(13)]],
    uint lane [[thread_position_in_grid]]) {
    if(lane!=0u||latched[0].identity.w!=0u)return;
    const auto s=stand[0];const auto m=matter[0];const auto r=respirationCandidate[0];
    const auto c=commonCoordinates[0];
    if(s.code==MR_NUMI_HUMAN_STAND_SUCCESS&&m.code==NM_STATUS_SUCCESS&&r.status.w==0u&&c.status.x==0u)return;
    const auto bi=brainInput[0];const auto ba=brainAccepted[0];
    const auto bc=brainCandidate[0];const auto bo=brainOutput[0];
    device MRHumanRestingCommonFailureGPU& out=latched[0];
    out.identity=uint4(d.x,m.code,s.code,1u);
    out.matterStatus=uint4(m.environment,m.objectIndex,m.failingIndex,m.completedMicrosteps);
    out.matterDiagnostics=m.diagnostics;
    out.standStatus=uint4(s.completedSteps,s.failingIndex,s.contactIterations,s.flags);
    out.respirationCandidateStatus=r.status;
    out.respirationAcceptedStatus=respirationAccepted[0].status;
    out.chamberVolumes=r.chamberVolumes;
    out.respirationControl=r.control;
    // Re-evaluate the exact accepted candidate targets only on failure. The
    // runtime output sanitizes every nonzero status to NaN, so this preserves
    // the converged trial point that status 5 rejected without making it a
    // renderable/published coordinate.
    float targetVolumes[7]={r.chamberVolumes.x,r.chamberVolumes.y,r.chamberVolumes.z,
        r.chamberVolumes.w,commonParameters.materialTargetVolumes.x,
        commonParameters.materialTargetVolumes.y,commonParameters.materialTargetVolumes.z};
    float attempted[7]={0,0,0,0,0,0,0},attemptResidual=INFINITY;
    uint attemptIterations=0u,attemptBox=MR_INVALID_INDEX;
    const uint attemptStatus=nmHumanRestingCommonSolve(commonParameters,commonBoxes,
        targetVolumes,attempted,attemptIterations,attemptBox,attemptResidual);
    out.commonFirst=float4(attempted[0],attempted[1],attempted[2],attempted[3]);
    out.commonSecond=float4(attempted[4],attempted[5],attempted[6],0.0f);
    out.commonStatus=uint4(attemptStatus,attemptIterations,attemptBox,0u);
    out.commonDiagnostics=float4(attemptResidual,0.0f,0.0f,0.0f);
    out.brainInputMetadata=uint4(bi.validityMask,bi.flags,uint(bi.sourceTimestampMicroseconds),uint(bi.targetTimestampMicroseconds));
    out.brainInputRoots=uint4(uint(bi.sourceAcceptedRootFingerprint),uint(bi.sourceAcceptedRootFingerprint>>32),
        uint(bi.targetTransactionFingerprint),uint(bi.targetTransactionFingerprint>>32));
    out.brainInputTimestampHighs=uint4(uint(bi.sourceTimestampMicroseconds>>32),uint(bi.targetTimestampMicroseconds>>32),0u,0u);
    out.brainAcceptedMetadata=uint4(ba.flags,uint(ba.timestampMicroseconds),uint(ba.timestampMicroseconds>>32),0u);
    out.brainAcceptedRoots=uint4(uint(ba.acceptedRootFingerprint),uint(ba.acceptedRootFingerprint>>32),0u,0u);
    out.brainCandidateMetadata=uint4(bc.flags,uint(bc.timestampMicroseconds),uint(bc.timestampMicroseconds>>32),0u);
    out.brainCandidateRoot=uint4(uint(bc.acceptedRootFingerprint),uint(bc.acceptedRootFingerprint>>32),0u,0u);
    out.brainOutputMetadata=uint4(bo.flags,uint(bo.targetTimestampMicroseconds),uint(bo.targetTimestampMicroseconds>>32),0u);
    out.brainOutputRoots=uint4(uint(bo.sourceAcceptedRootFingerprint),uint(bo.sourceAcceptedRootFingerprint>>32),
        uint(bo.targetTransactionFingerprint),uint(bo.targetTransactionFingerprint>>32));
    out.brainOutputSetpoints=float4(bo.targetMinuteVentilationLitresPerMinute,bo.targetFrequencyBreathsPerMinute,
        bo.targetTidalVolumeLitres,bo.diaphragmExcitation);
    out.brainOutputExcitations=float4(bo.intercostalExcitation,0.0f,0.0f,0.0f);
    out.excitationBuffer=excitation[0];
}
kernel void nm_human_resting_capture(
    constant uint4& d [[buffer(0)]], device const MRArticulatedBodyPoseGPU* poses [[buffer(1)]],
    device const MRBodyStateGPU* properties [[buffer(2)]], device MRBodyStateGPU* bodies [[buffer(3)]],
    device const NMHumanRespirationState* respiration [[buffer(4)]],
    device NMHumanRespirationState* frame [[buffer(5)]],
    device const MRNumiHumanStandStatusGPU* statuses [[buffer(6)]],uint i [[thread_position_in_grid]]) {
    if(i>=d.y||statuses[0].code!=MR_NUMI_HUMAN_STAND_SUCCESS||respiration[0].status.x!=d.x)return;
    auto b=properties[i];b.position=poses[i].position;
    // Match the existing visualBodyStates boundary: presentation rotations
    // are unit quaternions even when the physical pose stream has roundoff.
    b.orientation=normalize(poses[i].orientation);bodies[i]=b;
    if(i==0)frame[0]=respiration[0];
}

kernel void nm_human_resting_present_commit(
    constant uint4& d [[buffer(0)]], device const MRBodyStateGPU* candidateBodies [[buffer(1)]],
    device const NMHumanRespirationState* candidateRespiration [[buffer(2)]],
    device MRBodyStateGPU* bodies [[buffer(3)]], device NMHumanRespirationState* respiration [[buffer(4)]],
    device const MRNumiHumanStandStatusGPU* statuses [[buffer(5)]],
    device MRHumanRestingCommonCoordinatesGPU* frameCommonCoordinates [[buffer(6)]],
    device MRHumanRestingCommonCoordinatesGPU* commonCoordinates [[buffer(7)]],
    device const MRHumanRestingCommonCoordinatesGPU* candidateCommonCoordinates [[buffer(8)]],
    device MRHumanRestingCommonCoordinatesGPU* acceptedCommonCoordinates [[buffer(9)]],
    device const NMMatterStatusGPU* matterStatuses [[buffer(10)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=d.y||statuses[0].code!=MR_NUMI_HUMAN_STAND_SUCCESS||
       statuses[0].completedSteps!=d.x+1||candidateRespiration[0].status.x!=d.x||
       (d.z&&matterStatuses[0].code!=NM_STATUS_SUCCESS))return;
    bodies[i]=candidateBodies[i];
    if(i==0) {
        respiration[0]=candidateRespiration[0];
        if(d.z) {
            // Capture the exact common coordinates paired with the body and
            // respiratory frame candidates. This occurs only after the whole
            // physical transaction succeeds, so a rejected attempt cannot
            // mutate the published-frame snapshot.
            frameCommonCoordinates[0]=acceptedCommonCoordinates[0];
            commonCoordinates[0]=frameCommonCoordinates[0];
            // Candidate coordinates belong to this accepted physical step,
            // not to the renderer's sparse frame-export cadence.
            acceptedCommonCoordinates[0]=candidateCommonCoordinates[0];
        } else {
            commonCoordinates[0]=candidateCommonCoordinates[0];
        }
    }
}

kernel void nm_human_respiration_brain_observe(
    constant NMHumanRespirationBrainDispatch& d [[buffer(0)]],
    device const NMHumanRespirationState* accepted [[buffer(1)]],
    device NBNumiRespiratoryChemoreflexInputV1* input [[buffer(2)]],
    uint env [[thread_position_in_grid]]) {
    if(env>=d.environmentCount)return;
    const auto s=accepted[env];
    NBNumiRespiratoryChemoreflexInputV1 sample{};
    sample.sourceTimestampMicroseconds=d.sourceTimeMicroseconds;
    sample.sourceAcceptedRootFingerprint=d.sourceRootIdentity;
    sample.targetTimestampMicroseconds=d.targetTimeMicroseconds;
    sample.targetTransactionFingerprint=d.targetRootIdentity;
    sample.paO2MillimetersMercury=s.observation.x;
    sample.paCO2MillimetersMercury=s.observation.y;
    sample.validityMask=NB_NUMI_RESPIRATORY_VALID_PA_O2|NB_NUMI_RESPIRATORY_VALID_PA_CO2;
    sample.flags=s.status.x==d.sourceStep&&!s.status.w?NB_NUMI_RESPIRATORY_INPUT_ACCEPTED:0;
    input[env]=sample;
}

kernel void nm_human_respiration_brain_deliver(
    constant NMHumanRespirationBrainDispatch& d [[buffer(0)]],
    device const NBNumiRespiratoryChemoreflexOutputV1* output [[buffer(1)]],
    device float4* excitation [[buffer(2)]],
    uint env [[thread_position_in_grid]]) {
    if(env>=d.environmentCount)return;
    auto o=output[env];
    if(!(o.flags&NB_NUMI_RESPIRATORY_RECORD_VALID)||o.targetTransactionFingerprint!=d.targetRootIdentity)
        excitation[env]=float4(NAN);
    else excitation[env]=float4(clamp(d.driveScale*float2(o.diaphragmExcitation,o.intercostalExcitation),0.0f,1.0f),
        o.targetFrequencyBreathsPerMinute,o.targetMinuteVentilationLitresPerMinute);
}

kernel void nm_human_respiration_brain_resolve(
    constant NMHumanRespirationBrainDispatch& d [[buffer(0)]],
    device const NBNumiRespiratoryChemoreflexStateV1* candidate [[buffer(1)]],
    device NBNumiRespiratoryChemoreflexStateV1* accepted [[buffer(2)]],
    device const NMMatterStatusGPU* statuses [[buffer(3)]],
    device const NMHumanRespirationState* respiration [[buffer(4)]],
    uint env [[thread_position_in_grid]]) {
    if(env<d.environmentCount&&statuses[env].code==NM_STATUS_SUCCESS&&!respiration[env].status.w&&
       (candidate[env].flags&NB_NUMI_RESPIRATORY_RECORD_VALID)&&
       candidate[env].acceptedRootFingerprint==d.targetRootIdentity)
        accepted[env]=candidate[env];
}

namespace human_respiration {
inline void addGas(thread float4& amount, float2 delta) {
    float2 y = delta - amount.zw;
    float2 next = amount.xy + y;
    amount.zw = (next - amount.xy) - y;
    amount.xy = next;
}
inline float saturation(float pressure, constant NMHumanRespirationParameters& p) {
    float ratio = pow(max(pressure, 0.0f) / p.oxygen.z, p.oxygen.w);
    return ratio / (1.0f + ratio);
}
inline float oxygenContent(float pressure, constant NMHumanRespirationParameters& p) {
    return p.oxygen.x * saturation(pressure, p) + p.oxygen.y * pressure;
}
inline float oxygenPressure(float content, constant NMHumanRespirationParameters& p) {
    float low = 0.0f, high = 760.0f;
    for (uint i=0; i<24; ++i) {
        float mid = (low + high) * 0.5f;
        if (oxygenContent(mid,p) < content) low=mid; else high=mid;
    }
    return (low+high)*0.5f;
}
inline float physical(device const float4* values,
                      device const NMVascularUnknownGPU* unknowns, uint base, uint row) {
    return values[base+row].x * unknowns[row].initialAndScaling.y;
}
}

kernel void nm_human_respiration_predict(
    constant NMHumanRespirationParameters& p [[buffer(0)]],
    constant NMHumanRespirationDispatch& d [[buffer(1)]],
    device const NMHumanRespirationState* accepted [[buffer(2)]],
    device NMHumanRespirationState* candidate [[buffer(3)]],
    device const float4* excitations [[buffer(4)]],
    uint env [[thread_position_in_grid]]) {
    if (env>=d.environmentCount) return;
    NMHumanRespirationState n=accepted[env];
    n.control=excitations[env];
    n.status.w=0;
    const float dt=p.environment.w;
    float2 pressure=0;
    for(uint m=0; m<2; ++m) {
        auto muscle=p.muscles[m];
        auto state=n.muscles[m];
        float excitation=excitations[env][m];
        if (!isfinite(excitation) || excitation<0 || excitation>1) {n.status.w=1;break;}
        float a=state.excitationAndActivation.y;
        float derivative=activationDerivative(muscle,excitation,a);
        float tau=derivative!=0 ? (excitation-a)/derivative : 1;
        float nextA=excitation-(excitation-a)*exp(-dt/tau);
        const float area=p.geometry[m+2];
        const float path=muscle.compliantArchitecture0.x+muscle.compliantArchitecture0.y-n.motion[m]/area;
        float fibre,velocity,tension,residual;
        if (!solveCompliantFiber(path,-n.motion[m+2]/area,dt,nextA,state,muscle,
                                 fibre,velocity,tension,residual) || abs(residual)>1.e-3f) {
            n.status.w=2;break;
        }
        n.muscles[m].excitationAndActivation=float4(excitation,nextA,fibre,velocity);
        pressure[m]=tension*forceScale(muscle.gainParameters,muscle.lengthRangeAndAcceleration.z)/area;
    }
    // Backward Euler for two muscle-driven thoracic volume coordinates.
    // Lung recoil and mouth resistance act on their sum, giving a symmetric
    // coupled mechanical solve. No phase/clock function writes lung volume.
    const float lungE=1.0f/p.lung.z;
    const float common=dt*(p.lung.w+dt*lungE);
    const float a=p.geometry.x+dt*p.chest.z+dt*dt/p.chest.x+common;
    const float b=common;
    const float c=p.geometry.y+dt*p.chest.w+dt*dt/p.chest.y+common;
    const float recoil=lungE*(n.motion.x+n.motion.y);
    const float2 rhs=p.geometry.xy*n.motion.zw+dt*(pressure-float2(n.motion.x/p.chest.x,n.motion.y/p.chest.y)-recoil);
    const float determinant=a*c-b*b;
    const float2 speed=float2(c*rhs.x-b*rhs.y,a*rhs.y-b*rhs.x)/determinant;
    n.motion.xy+=dt*speed;
    n.motion.zw=speed;
    const float flow=speed.x+speed.y;
    const float volume=p.lung.x+n.motion.x+n.motion.y;
    const float alveolarPressure=-p.lung.w*flow;
    n.mechanics=float4(volume,alveolarPressure,
        p.environment.x+alveolarPressure-lungE*(volume-p.lung.x),flow);
    if (!all(isfinite(n.motion)) || !all(isfinite(n.mechanics)) ||
        !(volume>p.lung.y) || volume>0.008f || determinant<=0 || d.reject) n.status.w=3;
    candidate[env]=n;
}

kernel void nm_human_respiration_exchange(
    constant NMHumanRespirationParameters& p [[buffer(0)]],
    constant NMHumanRespirationDispatch& d [[buffer(1)]],
    device const NMHumanRespirationState* accepted [[buffer(2)]],
    device NMHumanRespirationState* candidate [[buffer(3)]],
    device const float4* vascularBefore [[buffer(4)]],
    device const float4* vascularAfter [[buffer(5)]],
    device const NMVascularUnknownGPU* unknowns [[buffer(6)]],
    device const NMVascularConnectionGPU* connections [[buffer(7)]],
    device NMMatterStatusGPU* statuses [[buffer(8)]],
    device const NMVascularCompartmentGPU* compartments [[buffer(9)]],
    device const float* elastance [[buffer(10)]],
    uint env [[thread_position_in_grid]]) {
    if(env>=d.environmentCount) return;
    NMHumanRespirationState n=candidate[env];
    if(statuses[env].code!=NM_STATUS_SUCCESS) return;
    const NMHumanRespirationState old=accepted[env];
    const float dt=p.environment.w;
    const uint base=env*d.vascularStride;
    float2 delta[21];
    for(uint row=0;row<21;++row) delta[row]=0;
    float2 alveolarDelta=0;
    const float alveolarVolume=old.mechanics.x-p.lung.y;
    const float2 alveolarFraction=old.alveolarGas.xy/(alveolarVolume*p.environment.z);
    const float2 deadFraction=old.deadSpaceGas.xy/(p.lung.y*p.environment.z);
    const float2 alveolarPressure=alveolarFraction*(p.environment.y/133.322387415f);
    const float2 capillaryContent=float2(human_respiration::oxygenContent(alveolarPressure.x,p),
        p.carbonDioxide.x+p.carbonDioxide.y*(alveolarPressure.y-40.0f));
    float pulmonaryO2=0;
    for(uint edge=0;edge<p.topology.w;++edge) {
        const auto con=connections[edge];
        const float flow=human_respiration::physical(vascularAfter,unknowns,base,21+edge);
        const uint from=flow>=0?con.identity.y:con.identity.z;
        const uint to=flow>=0?con.identity.z:con.identity.y;
        const float fromVolume=human_respiration::physical(vascularBefore,unknowns,base,from);
        const float2 content=old.bloodGas[from].xy/fromVolume;
        const float2 transported=dt*abs(flow)*content;
        delta[from]-=transported;
        delta[to]+=transported;
        if(edge==p.topology.x && flow>0) {
            // Perfusion-limited capillary equilibration; an explicit finite
            // effectiveness allows a matched diffusion sensitivity check.
            const float2 exchange=dt*flow*p.carbonDioxide.z*(capillaryContent-content);
            delta[to]+=exchange;
            alveolarDelta-=exchange;
            pulmonaryO2=exchange.x/dt;
        }
        if(dt*abs(flow)>0.1f*fromVolume || !all(isfinite(transported))) n.status.w=4;
    }
    for(uint bed=0;bed<4;++bed) {
        const float2 demand=dt*p.metabolism.zw*p.tissueFractions[bed];
        delta[p.tissueRows[bed]]+=float2(-demand.x,demand.y);
    }
    human_respiration::addGas(n.metabolicGas,dt*p.metabolism.zw);
    // Explicit upwind exchange with one anatomical dead-space reservoir.
    // The same flux leaves one reservoir and enters the next.
    const float transport=dt*n.mechanics.w*p.environment.z;
    const float2 outsideFlux=transport*(transport>=0?p.metabolism.xy:deadFraction);
    const float2 airwayFlux=transport*(transport>=0?deadFraction:alveolarFraction);
    human_respiration::addGas(n.environmentGas,outsideFlux);
    human_respiration::addGas(n.deadSpaceGas,outsideFlux-airwayFlux);
    human_respiration::addGas(n.alveolarGas,airwayFlux+alveolarDelta);
    for(uint row=0;row<21;++row) {
        human_respiration::addGas(n.bloodGas[row],delta[row]);
        if(!all(isfinite(n.bloodGas[row]))||any(n.bloodGas[row].xy<0)) n.status.w=5;
    }
    const uint sensed=p.topology.y;
    const float volume=human_respiration::physical(vascularAfter,unknowns,base,sensed);
    const float2 content=n.bloodGas[sensed].xy/volume;
    const float po2=human_respiration::oxygenPressure(content.x,p);
    const float pco2=40+(content.y-p.carbonDioxide.x)/p.carbonDioxide.y;
    n.observation=float4(po2,pco2,human_respiration::saturation(po2,p),pulmonaryO2);
    // These four source compartments use CVSim's linear/time-varying
    // elastance pressure law (atan venous reservoirs are not sampled here).
    const uint4 pressureRows=uint4(20,16,0,17);
    for(uint i=0;i<4;++i) {
        uint row=pressureRows[i];auto c=compartments[row];
        const float v=human_respiration::physical(vascularAfter,unknowns,base,row);
        n.cardiacPressure[i]=c.compliance.w+c.compliance.y+
            (v-c.compliance.x)*elastance[env*21+row];
    }
    float bloodVolume=0;
    for(uint row=0;row<21;++row) bloodVolume+=human_respiration::physical(vascularAfter,unknowns,base,row);
    n.circulation=float4(bloodVolume,
        human_respiration::physical(vascularAfter,unknowns,base,20),
        human_respiration::physical(vascularAfter,unknowns,base,16),
        max(n.circulation.w,abs(bloodVolume-0.00515f)));
    // In the closed 21-compartment network every edge is internal: its signed
    // dt*Q term appears once at each endpoint and cancels in the global volume
    // equation. Thus the global continuity residual is the sum of each node's
    // normalized volume term. Compare that with the rounded physical delta;
    // their difference isolates state-normalization arithmetic, while the
    // endpoint total also includes its simple FP32 reduction-order error.
    float stepResidual=0,stepResidualCorrection=0;
    float stepPhysicalDelta=0,stepPhysicalDeltaCorrection=0;
    for(uint node=0;node<p.topology.z;++node) {
        const float oldNormalized=vascularBefore[base+node].x;
        const float newNormalized=vascularAfter[base+node].x;
        const float scale=unknowns[node].initialAndScaling.y;
        const float volumeTerm=scale*(newNormalized-oldNormalized);
        const float oldPhysical=scale*oldNormalized;
        const float newPhysical=scale*newNormalized;
        const float physicalDelta=newPhysical-oldPhysical;
        const float residualAdjusted=volumeTerm-stepResidualCorrection;
        const float residualNext=stepResidual+residualAdjusted;
        stepResidualCorrection=(residualNext-stepResidual)-residualAdjusted;
        stepResidual=residualNext;
        const float physicalAdjusted=physicalDelta-stepPhysicalDeltaCorrection;
        const float physicalNext=stepPhysicalDelta+physicalAdjusted;
        stepPhysicalDeltaCorrection=(physicalNext-stepPhysicalDelta)-physicalAdjusted;
        stepPhysicalDelta=physicalNext;
    }
    const float residualValue=stepResidual-stepResidualCorrection;
    const float residualAdjusted=residualValue-n.bloodBalance.continuityResidualCompensationM3;
    const float residualNext=n.bloodBalance.continuityResidualSumM3+residualAdjusted;
    n.bloodBalance.continuityResidualCompensationM3=
        (residualNext-n.bloodBalance.continuityResidualSumM3)-residualAdjusted;
    n.bloodBalance.continuityResidualSumM3=residualNext;
    const float physicalValue=stepPhysicalDelta-stepPhysicalDeltaCorrection;
    const float physicalAdjusted=physicalValue-n.bloodBalance.physicalVolumeDeltaCompensationM3;
    const float physicalNext=n.bloodBalance.physicalVolumeDeltaSumM3+physicalAdjusted;
    n.bloodBalance.physicalVolumeDeltaCompensationM3=
        (physicalNext-n.bloodBalance.physicalVolumeDeltaSumM3)-physicalAdjusted;
    n.bloodBalance.physicalVolumeDeltaSumM3=physicalNext;
    n.chamberVolumes=float4(
        human_respiration::physical(vascularAfter,unknowns,base,15),
        human_respiration::physical(vascularAfter,unknowns,base,16),
        human_respiration::physical(vascularAfter,unknowns,base,19),
        human_respiration::physical(vascularAfter,unknowns,base,20));
    const float aortic=max(0.0f,human_respiration::physical(vascularAfter,unknowns,base,21));
    const float pulmonary=max(0.0f,human_respiration::physical(vascularAfter,unknowns,base,41));
    const float mitral=max(0.0f,human_respiration::physical(vascularAfter,unknowns,base,44));
    n.cardiacFlow.xy+=dt*float2(aortic,pulmonary);
    n.cardiacFlow.z+=dt*aortic;
    if(mitral>1.e-9f)n.cardiacStatus.y=1;
    if(aortic<=1.e-9f && n.cardiacStatus.z && n.cardiacStatus.y) {
        ++n.cardiacStatus.x;n.cardiacStatus.y=0;
        n.cardiacFlow.w=n.cardiacFlow.z;n.cardiacFlow.z=0;
    }
    n.cardiacStatus.z=aortic>1.e-9f;
    float4 total=0;
    human_respiration::addGas(total,n.alveolarGas.xy);
    human_respiration::addGas(total,n.deadSpaceGas.xy);
    for(uint row=0;row<21;++row) human_respiration::addGas(total,n.bloodGas[row].xy);
    const float2 balance=total.xy-n.environmentGas.xy+
        float2(n.metabolicGas.x,-n.metabolicGas.y)-n.gasBudget.xy;
    n.gasBudget.zw=max(n.gasBudget.zw,abs(balance));
    if(any(abs(balance)>5.e-5f*n.gasBudget.xy)) n.status.w=7;
    n.breath.x=max(n.breath.x,n.mechanics.x);
    n.breath.y=min(n.breath.y,n.mechanics.x);
    const float inspiredAdjusted=dt*max(0.0f,n.mechanics.w)-n.breathAccounting.y;
    const float inspiredNext=n.breath.w+inspiredAdjusted;
    n.breathAccounting.y=(inspiredNext-n.breath.w)-inspiredAdjusted;
    n.breath.w=inspiredNext;
    // Breath observation only: a 0.001 ml/s Schmitt threshold excludes
    // round-off sign chatter near zero flow (observed below 0.00001 ml/s).
    // This does not clip airflow or alter mechanics, gas transport or control.
    constexpr float breathFlowHysteresisM3PerS=1.e-9f;
    const uint inspiration=n.mechanics.w>breathFlowHysteresisM3PerS ? 1u :
        (n.mechanics.w < -breathFlowHysteresisM3PerS ? 0u : n.status.z);
    if(inspiration&&!n.status.z && n.status.x>0) {
        n.breath.z=n.breath.x-n.breath.y;
        n.breath.xy=float2(n.mechanics.x);
        n.breathTiming.y=n.breathTiming.x;
        n.breathTiming.x=n.status.x+1u;
        n.breathTiming.z=n.breathTiming.x-n.breathTiming.y;
        n.breathAccounting.z=(n.breath.w-n.breathAccounting.x)-
            (n.breathAccounting.y-n.breathAccounting.w);
        n.breathAccounting.x=n.breath.w;
        n.breathAccounting.w=n.breathAccounting.y;
        ++n.status.y;
    }
    n.status.z=inspiration;
    ++n.status.x;
    if(!all(isfinite(n.observation))||!all(isfinite(n.alveolarGas))||
       any(n.alveolarGas.xy<0)||any(n.deadSpaceGas.xy<0)||
       abs(transport)>0.1f*p.lung.y*p.environment.z) n.status.w=6;
    if(n.status.w) {statuses[env].code=NM_STATUS_MULTIPHYSICS_FAILURE;}
    candidate[env]=n;
}

kernel void nm_human_respiration_resolve(
    constant NMHumanRespirationDispatch& d [[buffer(0)]],
    device const NMHumanRespirationState* candidate [[buffer(1)]],
    device NMHumanRespirationState* accepted [[buffer(2)]],
    device const NMMatterStatusGPU* statuses [[buffer(3)]],
    uint env [[thread_position_in_grid]]) {
    if(env<d.environmentCount && statuses[env].code==NM_STATUS_SUCCESS && !candidate[env].status.w)
        accepted[env]=candidate[env];
}

// Selected accepted-render snapshots only. The caller allocates the shared
// destination only when explicit physical step IDs were requested, then
// dispatches this after deformation on the same command buffer.
kernel void nm_human_resting_capture_vertices(
    device const MRVisualVertexGPUV2* source [[buffer(0)]],
    device MRVisualVertexGPUV2* destination [[buffer(1)]],
    constant uint& vertexCount [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if(i<vertexCount)destination[i]=source[i];
}
