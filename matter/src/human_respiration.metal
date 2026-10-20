// The respiratory muscles use the existing MyoSim activation and compliant
// fibre/tendon implementation. No independent muscle constitutive law.
#include "MujocoMuscleReference.metal"
#include "numi/matter/human_respiration.h"
#include "RespiratoryChemoreflexV1.metal"
#include "metalrobo/engine_types.h"
#include "metalrobo/numi_human_stand_gpu.h"
#include "metalrobo/numi_human_resting_visual_gpu.h"
#include "NumiHumanRestingCommonField.metalinc"
#include "NumiHumanRestingSupportGeometry.metalinc"

constant bool kGasTransportSubcycling [[function_constant(40)]];
constant bool kUseGasTransportSubcycling =
    is_function_constant_defined(kGasTransportSubcycling) ? kGasTransportSubcycling : false;

constant bool kRespiratorySubcycling [[function_constant(41)]];
constant bool kUseRespiratorySubcycling =
    is_function_constant_defined(kRespiratorySubcycling) ? kRespiratorySubcycling : false;

// Two independent MyoSim respiratory actuators share one SIMD group. The
// remainder of the candidate is identical in every lane and lane zero alone
// publishes it, so accepted state and failure ownership remain unchanged.
constant bool kParallelRespiratoryMuscles [[function_constant(44)]];
constant bool kUseParallelRespiratoryMuscles =
    is_function_constant_defined(kParallelRespiratoryMuscles)
        ? kParallelRespiratoryMuscles : false;

// With one SIMD group per environment, a lane owns each blood compartment.
// Edge order is unchanged; shuffles gather donor content without a dynamically
// indexed private delta array. No parallel atomic gas accumulation is used.
constant bool kParallelRespiratoryGas [[function_constant(46)]];
constant bool kUseParallelRespiratoryGas =
    is_function_constant_defined(kParallelRespiratoryGas)
        ? kParallelRespiratoryGas : false;

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
// SIMD variant used by the integrated frame path. The scalar kernel above
// remains the reference path for fixture comparisons and failure receipts.
kernel void nm_human_resting_common_coordinates_simd(
    device const NMHumanRespirationState* respiration [[buffer(0)]],
    constant MRHumanRestingCommonFieldGPU& parameters [[buffer(1)]],
    device const MRHumanRestingCommonCoordinateBoxGPU* boxes [[buffer(2)]],
    device MRHumanRestingCommonCoordinatesGPU* output [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]]) {
    const auto state=respiration[0];
    const float targets[7]={state.chamberVolumes.x,state.chamberVolumes.y,state.chamberVolumes.z,
        state.chamberVolumes.w,parameters.materialTargetVolumes.x,parameters.materialTargetVolumes.y,
        parameters.materialTargetVolumes.z};
    float coordinates[7]={0,0,0,0,0,0,0},residual=INFINITY;
    uint iterations=0,matched=MR_INVALID_INDEX;
    const uint status=nmHumanRestingCommonSolveSimd(
        parameters,boxes,targets,coordinates,lane,iterations,matched,residual);
    if(lane!=0u) return;
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

// Preserve the correction under safe Metal math; a reassociated expression
// would erase the low-order contribution this operation is intended to retain.
inline void restingCompensatedPositionAdd(
    float3 term, thread float3& sum, thread float3& correction) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
    const float3 adjusted=term-correction;
    const float3 next=sum+adjusted;
    correction=(next-sum)-adjusted;
    sum=next;
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
    const uint influenceStride = m.deformationKind == 3u
        ? nm_human_resting_support::skinInfluenceStride() : 1u;
    const auto soleInfluence=influences[m.firstInfluence];
    const bool sourceLocalDeformation=m.deformationKind==1u||m.deformationKind==3u||
        m.deformationKind==4u||m.deformationKind==5u||m.deformationKind==6u||m.deformationKind==7u||m.deformationKind==9u||m.deformationKind==11u||m.deformationKind==12u;
    const bool exactBodyLocal=sourceLocalDeformation&&m.influenceCount==1u&&
        soleInfluence.body.x==anatomy.bodyAndFlags.x&&soleInfluence.positionAndWeight.w==1.0f;
    const bool compensatedPosition=
        soleInfluence.body.y==MR_HUMAN_RESTING_COMPENSATED_POSITION;
    const bool ownerLocal=m.deformationKind==0u&&m.deformationWeight.w==3.0f;
    float3 positionCorrection=0;
    float3 p=0,n=0,strongest=0;float strongestWeight=-1;
    for(uint j=0;j<m.influenceCount;++j) {
        auto influence=influences[m.firstInfluence+j*influenceStride];
        float w=influence.positionAndWeight.w;
        float3 ni;
        if(ownerLocal) {
            const auto frame=bodies[m.chamberIndex];
            const float4 inverse=float4(-frame.orientation.xyz,frame.orientation.w);
            float3 point=influence.positionAndWeight.xyz;ni=influence.normal.xyz;
            if(influence.body.x!=m.chamberIndex) {
                if(influence.body.x!=MR_INVALID_INDEX) {
                    const auto body=bodies[influence.body.x];
                    point=(body.position.xyz-frame.position.xyz)+restingRotate(body.orientation,point);
                    ni=restingRotate(body.orientation,ni);
                }else point-=frame.position.xyz;
                point=restingRotate(inverse,point);ni=restingRotate(inverse,ni);
            }
            if(compensatedPosition)restingCompensatedPositionAdd(w*point,p,positionCorrection);
            else p+=w*point;
        }else if(influence.body.x==MR_INVALID_INDEX) {
            if(compensatedPosition)
                restingCompensatedPositionAdd(w*influence.positionAndWeight.xyz,p,positionCorrection);
            else p+=w*influence.positionAndWeight.xyz;
            ni=influence.normal.xyz;
        }else {
            auto body=bodies[influence.body.x];
            const float3 world=body.position.xyz+restingRotate(body.orientation,influence.positionAndWeight.xyz);
            if(compensatedPosition)
                restingCompensatedPositionAdd(w*world,p,positionCorrection);
            else p+=w*world;
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
        if(m.deformationWeight.w==2.0f&&(m.deformationKind==12u||m.deformationKind==1u||m.deformationKind==4u||m.deformationKind==5u||m.deformationKind==6u||m.deformationKind==7u||m.deformationKind==9u||m.deformationKind==11u)) {
            // Actual vertex-buffer geometry stays precise and torso-local.
            // Existing articulated instances supply the same world pose.
            // Strict area/volume guards inspect these actual local vertices.
            p=local;n=normal;
        }else {
            p=body.position.xyz+restingRotate(body.orientation,local);
            n=restingRotate(body.orientation,normal);
        }
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
// Whole native mesh validity is independent of which surfaces own a volume.
// Bounded parallel reductions leave only counts and the first witness on host.
// The rendered vertices are binary32. A fused multiply-subtract can leave a
// nonzero residual for an exactly collinear cross product, so confirm an
// apparent zero with product residuals. The integer fallback handles exact
// products and subnormal residuals without losing low bits.
inline bool nm_human_exact_binary32_products_equal(float a,float b,float c,float d) {
    if(!isfinite(a)||!isfinite(b)||!isfinite(c)||!isfinite(d))return false;
    const float ab=a*b,cd=c*d;
    if(ab!=cd)return false;
    const float abError=fma(a,b,-ab),cdError=fma(c,d,-cd);
    const float minimumNormal=0x1p-126f;
    if(isfinite(abError)&&isfinite(cdError)&&abError!=0.0f&&cdError!=0.0f&&
       abs(abError)>=minimumNormal&&abs(cdError)>=minimumNormal)return abError==cdError;

    const uint aBits=as_type<uint>(a),bBits=as_type<uint>(b);
    const uint cBits=as_type<uint>(c),dBits=as_type<uint>(d);
    const uint aExponent=(aBits>>23u)&0xffu,bExponent=(bBits>>23u)&0xffu;
    const uint cExponent=(cBits>>23u)&0xffu,dExponent=(dBits>>23u)&0xffu;
    if(aExponent==0xffu||bExponent==0xffu||cExponent==0xffu||dExponent==0xffu)return false;
    ulong aSignificand=ulong(aBits&0x7fffffu),bSignificand=ulong(bBits&0x7fffffu);
    ulong cSignificand=ulong(cBits&0x7fffffu),dSignificand=ulong(dBits&0x7fffffu);
    int aPower=-149,bPower=-149,cPower=-149,dPower=-149;
    if(aExponent){aSignificand|=0x800000ul;aPower=int(aExponent)-150;}
    if(bExponent){bSignificand|=0x800000ul;bPower=int(bExponent)-150;}
    if(cExponent){cSignificand|=0x800000ul;cPower=int(cExponent)-150;}
    if(dExponent){dSignificand|=0x800000ul;dPower=int(dExponent)-150;}
    ulong abSignificand=aSignificand*bSignificand,cdSignificand=cSignificand*dSignificand;
    if(abSignificand==0ul||cdSignificand==0ul)return abSignificand==cdSignificand;
    const bool abNegative=((aBits^bBits)&0x80000000u)!=0u;
    const bool cdNegative=((cBits^dBits)&0x80000000u)!=0u;
    if(abNegative!=cdNegative)return false;
    int abPower=aPower+bPower,cdPower=cPower+dPower;
    while((abSignificand&1ul)==0ul){abSignificand>>=1u;++abPower;}
    while((cdSignificand&1ul)==0ul){cdSignificand>>=1u;++cdPower;}
    return abSignificand==cdSignificand&&abPower==cdPower;
}
inline bool nm_human_binary32_requires_point_fallback(float value) {
    const uint bits=as_type<uint>(value);
    const uint exponent=(bits>>23u)&0xffu;
    const uint fraction=bits&0x7fffffu;
    // Metal may flush subnormal intermediate residuals. Below biased exponent
    // 25, an exact subtraction tail can be subnormal even when both inputs are
    // normal, so use the original-coordinate integer determinant directly.
    return (exponent==0u&&fraction!=0u)||(exponent>0u&&exponent<25u);
}
inline float nm_human_binary32_difference_tail(float a,float b,float difference) {
    const float bVirtual=a-difference;
    const float aVirtual=difference+bVirtual;
    const float bRound=bVirtual-b;
    const float aRound=a-aVirtual;
    return aRound+bRound;
}
inline void nm_human_add_binary32_word(thread ulong* accumulator,uint limb,ulong word) {
    while(word!=0ul&&limb<9u) {
        const ulong previous=accumulator[limb];
        accumulator[limb]=previous+word;
        word=accumulator[limb]<previous?1ul:0ul;
        ++limb;
    }
}
inline void nm_human_accumulate_binary32_product(
    thread ulong* positive,thread ulong* negative,float a,float b,bool negate) {
    const uint aBits=as_type<uint>(a),bBits=as_type<uint>(b);
    const uint aExponent=(aBits>>23u)&0xffu,bExponent=(bBits>>23u)&0xffu;
    if(aExponent==0xffu||bExponent==0xffu)return;
    ulong aSignificand=ulong(aBits&0x7fffffu),bSignificand=ulong(bBits&0x7fffffu);
    int aPower=-149,bPower=-149;
    if(aExponent){aSignificand|=0x800000ul;aPower=int(aExponent)-150;}
    if(bExponent){bSignificand|=0x800000ul;bPower=int(bExponent)-150;}
    const ulong significand=aSignificand*bSignificand;
    if(significand==0ul)return;
    const int power=aPower+bPower;
    const uint shift=uint(power+298);
    const uint limb=shift>>6u,bit=shift&63u;
    const ulong low=significand<<bit;
    const ulong high=bit?significand>>(64u-bit):0ul;
    const bool negativeProduct=(((aBits^bBits)&0x80000000u)!=0u)!=negate;
    thread ulong* accumulator=negativeProduct?negative:positive;
    nm_human_add_binary32_word(accumulator,limb,low);
    if(high)nm_human_add_binary32_word(accumulator,limb+1u,high);
}
inline bool nm_human_exact_binary32_point_area_component_is_zero(
    float a0,float b0,float c0,float a1,float b1,float c1) {
    thread ulong positive[9],negative[9];
    for(uint i=0;i<9u;++i){positive[i]=0ul;negative[i]=0ul;}
    // Expand (b0-a0)*(c1-a1) - (b1-a1)*(c0-a0).
    // The equal a0*a1 terms cancel exactly and are omitted.
    nm_human_accumulate_binary32_product(positive,negative,b0,c1,false);
    nm_human_accumulate_binary32_product(positive,negative,b0,a1,true);
    nm_human_accumulate_binary32_product(positive,negative,a0,c1,true);
    nm_human_accumulate_binary32_product(positive,negative,b1,c0,true);
    nm_human_accumulate_binary32_product(positive,negative,b1,a0,false);
    nm_human_accumulate_binary32_product(positive,negative,a1,c0,false);
    for(uint i=0;i<9u;++i)if(positive[i]!=negative[i])return false;
    return true;
}
inline bool nm_human_exact_binary32_triangle_is_zero(float3 a,float3 b,float3 c) {
    const float3 u=b-a,v=c-a;
    const bool safeCompensatedRange=
        !nm_human_binary32_requires_point_fallback(a.x)&&!nm_human_binary32_requires_point_fallback(a.y)&&
        !nm_human_binary32_requires_point_fallback(a.z)&&!nm_human_binary32_requires_point_fallback(b.x)&&
        !nm_human_binary32_requires_point_fallback(b.y)&&!nm_human_binary32_requires_point_fallback(b.z)&&
        !nm_human_binary32_requires_point_fallback(c.x)&&!nm_human_binary32_requires_point_fallback(c.y)&&
        !nm_human_binary32_requires_point_fallback(c.z)&&all(isfinite(u))&&all(isfinite(v));
    const bool exactEdges=safeCompensatedRange&&
        nm_human_binary32_difference_tail(b.x,a.x,u.x)==0.0f&&
        nm_human_binary32_difference_tail(b.y,a.y,u.y)==0.0f&&
        nm_human_binary32_difference_tail(b.z,a.z,u.z)==0.0f&&
        nm_human_binary32_difference_tail(c.x,a.x,v.x)==0.0f&&
        nm_human_binary32_difference_tail(c.y,a.y,v.y)==0.0f&&
        nm_human_binary32_difference_tail(c.z,a.z,v.z)==0.0f;
    if(!exactEdges) {
        // Full dyadic point-coordinate determinant fallback: exact even when
        // subtracting the origin rounded away low bits or products underflow.
        return nm_human_exact_binary32_point_area_component_is_zero(a.y,b.y,c.y,a.z,b.z,c.z)&&
            nm_human_exact_binary32_point_area_component_is_zero(a.z,b.z,c.z,a.x,b.x,c.x)&&
            nm_human_exact_binary32_point_area_component_is_zero(a.x,b.x,c.x,a.y,b.y,c.y);
    }
    return nm_human_exact_binary32_products_equal(u.y,v.z,u.z,v.y)&&
        nm_human_exact_binary32_products_equal(u.z,v.x,u.x,v.z)&&
        nm_human_exact_binary32_products_equal(u.x,v.y,u.y,v.x);
}

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
        const float3 u=pb-pa,v=pc-pa;
        const float3 area=cross(u,v);
        const bool bad=!all(isfinite(area));
        const bool collapsed=!bad&&nm_human_exact_binary32_triangle_is_zero(pa,pb,pc);
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
        const float3 u=pb-pa,v=pc-pa;
        const float3 area=cross(u,v);
        const bool bad=!all(isfinite(area));
        const uint areaFailure=bad?MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONFINITE_AREA:
            (nm_human_exact_binary32_triangle_is_zero(pa,pb,pc)?MR_HUMAN_RESTING_TRIANGLE_FAILURE_EXACT_ZERO_AREA:
                MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONE);
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
        const float3 u=b-a,v=c-a;
        const float3 area=cross(u,v);
        const bool bad=!all(isfinite(area));
        const bool collapsed=!bad&&nm_human_exact_binary32_triangle_is_zero(a,b,c);
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
        const float3 u=b-a,v=c-a;
        const float3 area=cross(u,v);
        const bool bad=!all(isfinite(area));
        const uint kind=bad?MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONFINITE_AREA:
            (nm_human_exact_binary32_triangle_is_zero(a,b,c)?MR_HUMAN_RESTING_TRIANGLE_FAILURE_EXACT_ZERO_AREA:
                MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONE);
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

// Preserve the original single-group skin audit's left-biased reduction
// witness order: stride-halving orders the 8-bit lane indices by bit reversal;
// ties within a lane choose the earliest original loop iteration.
inline uint nmHumanRestingSkinAuditLanePriority(uint lane) {
    uint x=lane&255u;
    x=((x&0x55u)<<1)|((x>>1)&0x55u);
    x=((x&0x33u)<<2)|((x>>2)&0x33u);
    x=((x&0x0fu)<<4)|((x>>4)&0x0fu);
    return x;
}
inline bool nmHumanRestingSkinAuditOwnerBefore(uint candidate,uint current) {
    if(candidate==MR_INVALID_INDEX)return false;
    if(current==MR_INVALID_INDEX)return true;
    const uint candidatePriority=nmHumanRestingSkinAuditLanePriority(candidate);
    const uint currentPriority=nmHumanRestingSkinAuditLanePriority(current);
    return candidatePriority<currentPriority||
        (candidatePriority==currentPriority&&candidate/256u<current/256u);
}
kernel void nm_human_resting_audit_skin_partials(
    constant uint4& d [[buffer(0)]],device const MRHumanRestingVertexMap* map [[buffer(1)]],
    device const MRVisualVertexGPUV2* vertices [[buffer(2)]],device uint4* partials [[buffer(3)]],
    constant MRHumanRestingBedGPU& bed [[buffer(4), function_constant(kUseRestingContouredBed)]],
    device const float* bedHeights [[buffer(5), function_constant(kUseRestingContouredBed)]],
    uint lane [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float minimum[256];threadgroup uint owner[256],below[256],invalid[256];
    float value=INFINITY;uint index=MR_INVALID_INDEX,count=0,nonfinite=0;
    const uint i=group.x*256u+lane;
    if(i<d.x) {
        if(!all(isfinite(vertices[i].position))||!all(isfinite(vertices[i].normalAndTangentSign)))++nonfinite;
        if(map[i].deformationKind==3u) {
            float z=vertices[i].position.z;
            if(kUseRestingContouredBed) {
                const auto hit=nmHumanRestingBedQuery(bed,bedHeights,vertices[i].position.xyz);
                if(!hit.valid)++nonfinite;
                z=hit.signedGap;
            }
            if(z<value){value=z;index=i;}
            if(z<-.001f)++count;
        }
    }
    minimum[lane]=value;owner[lane]=index;below[lane]=count;invalid[lane]=nonfinite;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=128;stride;stride>>=1) {
        if(lane<stride) {
            const uint other=lane+stride;
            if(minimum[other]<minimum[lane]||
               (minimum[other]==minimum[lane]&&
                nmHumanRestingSkinAuditOwnerBefore(owner[other],owner[lane]))) {
                minimum[lane]=minimum[other];owner[lane]=owner[other];
            }
            below[lane]+=below[other];invalid[lane]+=invalid[other];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(lane==0)partials[group.x]=uint4(as_type<uint>(minimum[0]),owner[0],below[0],invalid[0]);
}
kernel void nm_human_resting_reduce_skin_audit(
    constant uint4& d [[buffer(0)]],device const uint4* partials [[buffer(1)]],
    device float4* result [[buffer(2)]],uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float minimum[256];threadgroup uint owner[256],below[256],invalid[256];
    float value=INFINITY;uint index=MR_INVALID_INDEX,count=0,nonfinite=0;
    for(uint group=lane;group<d.x;group+=256u) {
        const uint4 partial=partials[group];const float candidate=as_type<float>(partial.x);
        if(candidate<value||
           (candidate==value&&nmHumanRestingSkinAuditOwnerBefore(partial.y,index))) {
            value=candidate;index=partial.y;
        }
        count+=partial.z;nonfinite+=partial.w;
    }
    minimum[lane]=value;owner[lane]=index;below[lane]=count;invalid[lane]=nonfinite;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=128;stride;stride>>=1) {
        if(lane<stride) {
            const uint other=lane+stride;
            if(minimum[other]<minimum[lane]||
               (minimum[other]==minimum[lane]&&
                nmHumanRestingSkinAuditOwnerBefore(owner[other],owner[lane]))) {
                minimum[lane]=minimum[other];owner[lane]=owner[other];
            }
            below[lane]+=below[other];invalid[lane]+=invalid[other];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(lane==0)result[d.y]=float4(minimum[0],float(owner[0]),float(below[0]),float(invalid[0]));
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

kernel void nm_human_resting_terminal_present_commit(
    constant uint4& d [[buffer(0)]], device const MRBodyStateGPU* candidateBodies [[buffer(1)]],
    device const NMHumanRespirationState* candidateRespiration [[buffer(2)]],
    device MRBodyStateGPU* bodies [[buffer(3)]], device NMHumanRespirationState* respiration [[buffer(4)]],
    device const MRNumiHumanStandStatusGPU* statuses [[buffer(5)]],
    device MRHumanRestingCommonCoordinatesGPU* frameCommonCoordinates [[buffer(6)]],
    device MRHumanRestingCommonCoordinatesGPU* commonCoordinates [[buffer(7)]],
    device const MRHumanRestingCommonCoordinatesGPU* acceptedCommonCoordinates [[buffer(8)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=d.y||statuses[0].code!=MR_NUMI_HUMAN_STAND_SUCCESS||
       statuses[0].completedSteps!=d.x||candidateRespiration[0].status.x!=d.x||
       candidateRespiration[0].status.w!=0u)return;
    bodies[i]=candidateBodies[i];
    if(i==0u) {
        respiration[0]=candidateRespiration[0];
        if(d.z) {
            // The final accepted transaction already advanced this
            // registered GPU-owned buffer with the accepted respiration state.
            // Terminal publication only catches the presentation snapshot up.
            frameCommonCoordinates[0]=acceptedCommonCoordinates[0];
            commonCoordinates[0]=acceptedCommonCoordinates[0];
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

struct NMRespiratoryMuscleCandidate {
    float4 state;
    float pressure;
    uint failure;
};

inline NMRespiratoryMuscleCandidate evaluateHumanRespiratoryMuscle(
    thread const NMHumanRespirationState& n,
    constant NMHumanRespirationParameters& p, const float dt, const uint m) {
    auto muscle=p.muscles[m];
    auto state=n.muscles[m];
    float excitation=n.control[m];
    NMRespiratoryMuscleCandidate result{state.excitationAndActivation,0.0f,0u};
    if (!isfinite(excitation) || excitation<0 || excitation>1) {
        result.failure=1u;return result;
    }
    float a=state.excitationAndActivation.y;
    float derivative=activationDerivative(muscle,excitation,a);
    float tau=derivative!=0 ? (excitation-a)/derivative : 1;
    float nextA=excitation-(excitation-a)*exp(-dt/tau);
    const float area=p.geometry[m+2];
    const float path=muscle.compliantArchitecture0.x+muscle.compliantArchitecture0.y-n.motion[m]/area;
    float fibre,velocity,tension,residual;
    if (!solveCompliantFiber(path,-n.motion[m+2]/area,dt,nextA,state,muscle,
                             fibre,velocity,tension,residual) || abs(residual)>1.e-3f) {
        result.failure=2u;return result;
    }
    result.state=float4(excitation,nextA,fibre,velocity);
    result.pressure=tension*forceScale(muscle.gainParameters,muscle.lengthRangeAndAcceleration.z)/area;
    return result;
}

inline void advanceHumanRespiratoryMechanics(
    thread NMHumanRespirationState& n,
    constant NMHumanRespirationParameters& p, const float dt, const bool reject,
    const uint lane=0u, const bool cooperative=false) {
    float2 pressure=0;
    if (cooperative) {
        NMRespiratoryMuscleCandidate pair{float4(0.0f),0.0f,0u};
        if (lane<2u) pair=evaluateHumanRespiratoryMuscle(n,p,dt,lane);
        // Preserve first-failure precedence and each actuator's publication
        // order, including the case where only the second actuator fails.
        for(uint m=0u;m<2u;++m) {
            const uint failure=simd_broadcast(pair.failure,m);
            const float4 state=simd_broadcast(pair.state,m);
            const float force=simd_broadcast(pair.pressure,m);
            if(failure) {n.status.w=failure;break;}
            n.muscles[m].excitationAndActivation=state;
            pressure[m]=force;
        }
    } else {
        for(uint m=0u;m<2u;++m) {
            const auto result=evaluateHumanRespiratoryMuscle(n,p,dt,m);
            if(result.failure) {n.status.w=result.failure;break;}
            n.muscles[m].excitationAndActivation=result.state;
            pressure[m]=result.pressure;
        }
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
        !(volume>p.lung.y) || volume>0.008f || determinant<=0 || reject) n.status.w=3;
}

inline float3x3 nmHumanBloodMatrixFromRows(
    const float4 row0, const float4 row1, const float4 row2
) {
    return transpose(float3x3(row0.xyz, row1.xyz, row2.xyz));
}

inline void nmHumanBloodMatrixToRows(
    const float3x3 value,
    thread float4& row0,
    thread float4& row1,
    thread float4& row2
) {
    const float3x3 transposed = transpose(value);
    row0 = float4(transposed[0], 0.0f);
    row1 = float4(transposed[1], 0.0f);
    row2 = float4(transposed[2], 0.0f);
}

inline float nmHumanBloodDeterminantSafe(const float3x3 value) {
    return determinant(value);
}

inline float nmHumanBloodRowDifference(float4 a, float4 b) {
    const float4 difference = abs(a - b);
    return max(max(difference.x, difference.y), max(difference.z, difference.w));
}

inline float3 nmHumanBloodAngularVelocity(float4 row0, float4 row1, float4 row2) {
    // APIC rows encode the spatial velocity gradient. Its antisymmetric part is
    // the unresolved particle spin, using the same quadratic B-spline moment as
    // the handoff's omega = 2 L / (m h^2) conversion.
    return 0.5f * float3(row2.y - row1.z, row0.z - row2.x, row1.x - row0.y);
}

inline bool nmHumanBloodMergeDeformation(
    NMParticleStateGPU a,
    NMParticleStateGPU b,
    float h,
    float bulkModulus,
    thread float3x3& mergedDeformation,
    thread float& bulkEnergyLoss
) {
    const float volumeA=a.velocityAndReferenceVolume.w;
    const float volumeB=b.velocityAndReferenceVolume.w;
    const float mergedVolume=volumeA+volumeB;
    if(!isfinite(volumeA)||!(volumeA>0.0f)||
       !isfinite(volumeB)||!(volumeB>0.0f)||
       !isfinite(mergedVolume)||!(mergedVolume>0.0f)||
       !isfinite(h)||!(h>0.0f)||
       !isfinite(bulkModulus)||!(bulkModulus>0.0f)) return false;
    const float3x3 deformationA=nmHumanBloodMatrixFromRows(
        a.deformationRow0,a.deformationRow1,a.deformationRow2);
    const float3x3 deformationB=nmHumanBloodMatrixFromRows(
        b.deformationRow0,b.deformationRow1,b.deformationRow2);
    const float jacobianA=nmHumanBloodDeterminantSafe(deformationA);
    const float jacobianB=nmHumanBloodDeterminantSafe(deformationB);
    if(!all(isfinite(deformationA[0]))||!all(isfinite(deformationA[1]))||
       !all(isfinite(deformationA[2]))||!all(isfinite(deformationB[0]))||
       !all(isfinite(deformationB[1]))||!all(isfinite(deformationB[2]))||
       !isfinite(jacobianA)||!(jacobianA>0.0f)||
       !isfinite(jacobianB)||!(jacobianB>0.0f)) return false;
    const float targetJacobian=(volumeA*jacobianA+volumeB*jacobianB)/mergedVolume;
    const float3x3 averageDeformation=
        (volumeA/mergedVolume)*deformationA+(volumeB/mergedVolume)*deformationB;
    const float averageJacobian=nmHumanBloodDeterminantSafe(averageDeformation);
    if(!isfinite(targetJacobian)||!(targetJacobian>0.0f)||
       !isfinite(averageJacobian)||!(averageJacobian>0.0f)) return false;
    const float rescale=pow(targetJacobian/averageJacobian,1.0f/3.0f);
    mergedDeformation=rescale*averageDeformation;
    const float mergedJacobian=nmHumanBloodDeterminantSafe(mergedDeformation);
    if(!isfinite(rescale)||!(rescale>0.0f)||
       !all(isfinite(mergedDeformation[0]))||
       !all(isfinite(mergedDeformation[1]))||
       !all(isfinite(mergedDeformation[2]))||
       !isfinite(mergedJacobian)||!(mergedJacobian>0.0f)) return false;
    const float originalDeformedVolume=
        volumeA*jacobianA+volumeB*jacobianB;
    const float mergedDeformedVolume=mergedVolume*mergedJacobian;
    const float floatEpsilon=1.1920928955078125e-7f;
    const float volumeTolerance=
        8.0f*floatEpsilon*max(abs(originalDeformedVolume),abs(mergedDeformedVolume));
    const float jacobianTolerance=
        8.0f*floatEpsilon*max(1.0f,abs(targetJacobian));
    if(!isfinite(originalDeformedVolume)||!(originalDeformedVolume>0.0f)||
       !isfinite(mergedDeformedVolume)||
       abs(mergedDeformedVolume-originalDeformedVolume)>volumeTolerance||
       abs(mergedJacobian-targetJacobian)>jacobianTolerance||
       mergedDeformedVolume>0.5f*h*h*h) return false;
    const float bulkEnergyBefore=0.5f*bulkModulus*(
        volumeA*(jacobianA-1.0f)*(jacobianA-1.0f)+
        volumeB*(jacobianB-1.0f)*(jacobianB-1.0f));
    const float bulkEnergyAfter=0.5f*bulkModulus*mergedVolume*
        (mergedJacobian-1.0f)*(mergedJacobian-1.0f);
    const float energyTolerance=max(1.0e-30f,
        16.0f*floatEpsilon*max(bulkEnergyBefore,bulkEnergyAfter));
    const float signedBulkLoss=bulkEnergyBefore-bulkEnergyAfter;
    if(!isfinite(bulkEnergyBefore)||!isfinite(bulkEnergyAfter)||
       !isfinite(signedBulkLoss)||signedBulkLoss < -energyTolerance) return false;
    bulkEnergyLoss=max(0.0f,signedBulkLoss);
    return isfinite(bulkEnergyLoss);
}

inline float nmHumanBloodAffineNormSquared(float4 row0, float4 row1, float4 row2) {
    return dot(row0.xyz, row0.xyz) +
        dot(row1.xyz, row1.xyz) +
        dot(row2.xyz, row2.xyz);
}

inline bool nmHumanBloodMergeKineticAdmissible(
    NMParticleStateGPU a,
    NMParticleStateGPU b,
    float h
) {
    const float massA = a.positionAndMass.w;
    const float massB = b.positionAndMass.w;
    const float mergedMass = massA + massB;
    const float hSquared = h * h;
    if (!isfinite(massA) || !(massA > 0.0f) ||
        !isfinite(massB) || !(massB > 0.0f) ||
        !isfinite(mergedMass) || !(mergedMass > 0.0f) ||
        !isfinite(hSquared) || !(hSquared > 0.0f)) return false;

    const float3 momentumA = massA * a.velocityAndReferenceVolume.xyz;
    const float3 momentumB = massB * b.velocityAndReferenceVolume.xyz;
    const float3 totalMomentum = momentumA + momentumB;
    const float3 center = (massA * a.positionAndMass.xyz +
        massB * b.positionAndMass.xyz) / mergedMass;
    const float3 velocity = totalMomentum / mergedMass;
    const float3 omegaA = nmHumanBloodAngularVelocity(
        a.affineRow0, a.affineRow1, a.affineRow2);
    const float3 omegaB = nmHumanBloodAngularVelocity(
        b.affineRow0, b.affineRow1, b.affineRow2);
    const float3 angularMomentumAboutCenter =
        cross(a.positionAndMass.xyz - center, momentumA - massA * velocity) +
        cross(b.positionAndMass.xyz - center, momentumB - massB * velocity) +
        (0.5f * hSquared) * (massA * omegaA + massB * omegaB);
    const float3 omegaMerged =
        (2.0f / (mergedMass * hSquared)) * angularMomentumAboutCenter;
    float4 affineRow0 = (massA * a.affineRow0 + massB * b.affineRow0) / mergedMass;
    float4 affineRow1 = (massA * a.affineRow1 + massB * b.affineRow1) / mergedMass;
    float4 affineRow2 = (massA * a.affineRow2 + massB * b.affineRow2) / mergedMass;
    const float3 omegaAverage = nmHumanBloodAngularVelocity(
        affineRow0, affineRow1, affineRow2);
    const float3 omegaCorrection = omegaMerged - omegaAverage;
    affineRow0.y -= omegaCorrection.z;
    affineRow0.z += omegaCorrection.y;
    affineRow1.x += omegaCorrection.z;
    affineRow1.z -= omegaCorrection.x;
    affineRow2.x -= omegaCorrection.y;
    affineRow2.y += omegaCorrection.x;

    const float kineticBefore =
        0.5f * massA * dot(a.velocityAndReferenceVolume.xyz,
            a.velocityAndReferenceVolume.xyz) +
        0.5f * massB * dot(b.velocityAndReferenceVolume.xyz,
            b.velocityAndReferenceVolume.xyz) +
        (hSquared / 8.0f) * (
            massA * nmHumanBloodAffineNormSquared(
                a.affineRow0, a.affineRow1, a.affineRow2) +
            massB * nmHumanBloodAffineNormSquared(
                b.affineRow0, b.affineRow1, b.affineRow2));
    const float kineticAfter =
        0.5f * mergedMass * dot(velocity, velocity) +
        (mergedMass * hSquared / 8.0f) *
            nmHumanBloodAffineNormSquared(affineRow0, affineRow1, affineRow2);
    const float signedKineticLoss=kineticBefore-kineticAfter;
    const float kineticTolerance=max(1.0e-30f,
        16.0f*1.1920928955078125e-7f*max(kineticBefore,kineticAfter));
    return all(isfinite(totalMomentum)) && all(isfinite(center)) &&
        all(isfinite(velocity)) && all(isfinite(angularMomentumAboutCenter)) &&
        all(isfinite(omegaMerged)) && all(isfinite(affineRow0)) &&
        all(isfinite(affineRow1)) && all(isfinite(affineRow2)) &&
        isfinite(kineticBefore) && isfinite(kineticAfter) &&
        isfinite(signedKineticLoss) && signedKineticLoss >= -kineticTolerance;
}

inline bool nmHumanBloodMergeStatesCompatible(
    NMParticleStateGPU a,
    NMParticleStateGPU b,
    device const float* materialState,
    uint stateBaseA,
    uint stateBaseB,
    uint stateStride,
    uint stateCount,
    float h,
    float bulkModulus
) {
    // The caller admits only the SHA-pinned stateless high-shear Newtonian
    // proxy. Preserve its complete material history and temperature exactly;
    // the source energy depends on J alone, so F may be projected only by the
    // bounded volume-preserving helper, which ledgers its measured bulk loss.
    if(!all(isfinite(a.deformationRow0))||
       !all(isfinite(a.deformationRow1))||
       !all(isfinite(a.deformationRow2))||
       !all(isfinite(b.deformationRow0))||
       !all(isfinite(b.deformationRow1))||
       !all(isfinite(b.deformationRow2))||
       !all(isfinite(a.affineRow0))||!all(isfinite(a.affineRow1))||
       !all(isfinite(a.affineRow2))||!all(isfinite(b.affineRow0))||
       !all(isfinite(b.affineRow1))||!all(isfinite(b.affineRow2))||
       !isfinite(a.referenceAndTemperature.w)||
       !isfinite(b.referenceAndTemperature.w)||
       a.referenceAndTemperature.w!=b.referenceAndTemperature.w) return false;
    float3x3 mergedDeformation;
    float bulkEnergyLoss=0.0f;
    if(!nmHumanBloodMergeDeformation(
        a,b,h,bulkModulus,mergedDeformation,bulkEnergyLoss)) return false;
    for(uint state=0u;state<stateCount;++state) {
        const float valueA=materialState[stateBaseA+state];
        const float valueB=materialState[stateBaseB+state];
        if(!isfinite(valueA)||!isfinite(valueB)||valueA!=valueB) return false;
    }
    return true;
}

// Transfer one already-accepted outlet batch into preauthored dormant MPM
// capacity. Matter has copied particle and material checkpoints immediately
// before frameBegin; those checkpoints remain immutable so normal rejection
// restores the activation. externalBlood.total is unchanged: mpmRepresented
// is only a subset of that external inventory.
kernel void nm_human_blood_mpm_handoff(
    constant NMHumanRespirationDispatch& d [[buffer(0)]],
    device NMHumanRespirationState* respirationCandidate [[buffer(1)]],
    device NMMatterStatusGPU* statuses [[buffer(2)]],
    device NMParticleStateGPU* particleAccepted [[buffer(3)]],
    device NMParticleStateGPU* particleCandidate [[buffer(4)]],
    device const NMParticleStateGPU* particleCheckpoint [[buffer(5)]],
    device const float* materialStateDefaults [[buffer(6)]],
    device float* materialStateAccepted [[buffer(7)]],
    device float* materialStateCandidate [[buffer(8)]],
    uint env [[thread_position_in_grid]]) {
    if (env >= d.environmentCount || d.bloodMpmEnabled == 0u) return;
    if (statuses[env].code != NM_STATUS_SUCCESS) return;

    NMHumanRespirationState n = respirationCandidate[env];
    const float4 batch = n.externalBlood.pendingParcel;
    if (batch.x == 0.0f && batch.y == 0.0f &&
        batch.z == 0.0f && batch.w == 0.0f) return;

    bool valid = d.bloodMpmParticleCount > 0u &&
        d.bloodMpmFirstParticle <= d.particleCount &&
        d.bloodMpmParticleCount <= d.particleCount - d.bloodMpmFirstParticle &&
        d.bloodMpmObjectIndex != NM_INVALID_INDEX &&
        d.bloodMpmMaterialIndex != NM_INVALID_INDEX &&
        d.bloodMpmMaterialStateCount <= d.materialStateStride &&
        isfinite(d.bloodMpmDensity) && d.bloodMpmDensity > 0.0f &&
        isfinite(d.bloodMpmBulkModulus) && d.bloodMpmBulkModulus > 0.0f &&
        isfinite(d.bloodMpmGridCellWidth) && d.bloodMpmGridCellWidth > 0.0f &&
        all(d.bloodMpmGridNodeDimensions.xyz > uint3(2u)) &&
        all(isfinite(batch)) && batch.x > 0.0f && batch.y > 0.0f &&
        batch.z >= 0.0f && batch.w >= 0.0f;
    const float4 positionMass = n.externalBlood.pendingPositionMass;
    const float3 momentum = n.externalBlood.pendingMomentum.xyz;
    const float4 angular = n.externalBlood.pendingAngularMomentum;
    if (!all(isfinite(positionMass)) || !(positionMass.w > 0.0f) ||
        !all(isfinite(momentum)) || !all(isfinite(angular.xyz))) valid = false;

    const float densityMass = d.bloodMpmDensity * batch.x;
    const float densityTolerance = max(1.0e-12f, 4.0e-6f * max(batch.y, densityMass));
    if (!isfinite(densityMass) || abs(batch.y - densityMass) > densityTolerance ||
        abs(positionMass.w - batch.y) > max(1.0e-12f, 4.0e-6f * batch.y))
        valid = false;

    const float4 external = n.externalBlood.total - n.externalBlood.compensation;
    const float4 represented = n.externalBlood.mpmRepresented -
        n.externalBlood.mpmRepresentedCompensation;
    const float4 inventoryTolerance = max(float4(1.0e-11f), 2.0e-5f * max(abs(external), abs(batch)));
    if (!all(isfinite(external)) || !all(isfinite(represented)) ||
        any(represented < float4(0.0f)) ||
        any(represented + batch > external + inventoryTolerance))
        valid = false;

    const float3 spawn = positionMass.xyz / positionMass.w;
    const float3 intrinsicAngular = angular.xyz - cross(spawn, momentum);
    const float h = d.bloodMpmGridCellWidth;
    const int3 base = int3(floor(spawn / h - 0.5f));
    const int3 minimum = d.bloodMpmGridNodeMinimum.xyz;
    const int3 maximum = minimum + int3(d.bloodMpmGridNodeDimensions.xyz) - 1;
    if (!all(isfinite(spawn)) || !all(isfinite(intrinsicAngular)) ||
        any(base < minimum) || any(base + int3(2) > maximum))
        valid = false;

    const uint particleBase = env * d.particleCount;
    const uint stateBase = env * d.particleCount * d.materialStateStride;
    float capacityVolume = 0.0f;
    if (valid) {
        for (uint local = 0u; local < d.bloodMpmParticleCount; ++local) {
            const uint index = d.bloodMpmFirstParticle + local;
            const uint global = particleBase + index;
            const NMParticleStateGPU slot = particleCheckpoint[global];
            const NMParticleStateGPU accepted = particleAccepted[global];
            const uint stableFlags = ~NM_OBJECT_ACTIVE;
            if (slot.identity.x != d.bloodMpmObjectIndex ||
                slot.identity.y != d.bloodMpmMaterialIndex ||
                slot.identity.z != d.bloodMpmTopologyGeneration ||
                accepted.identity.x != slot.identity.x ||
                accepted.identity.y != slot.identity.y ||
                accepted.identity.z != slot.identity.z ||
                (accepted.identity.w & stableFlags) != (slot.identity.w & stableFlags) ||
                !all(isfinite(accepted.positionAndMass)) ||
                !all(isfinite(accepted.velocityAndReferenceVolume)) ||
                !(accepted.positionAndMass.w > 0.0f) ||
                !(accepted.velocityAndReferenceVolume.w > 0.0f) ||
                abs(accepted.positionAndMass.w -
                    d.bloodMpmDensity * accepted.velocityAndReferenceVolume.w) >
                    max(1.0e-12f, 4.0e-6f * accepted.positionAndMass.w)) {
                valid = false;
                break;
            }
            if ((accepted.identity.w & NM_OBJECT_ACTIVE) == 0u)
                capacityVolume += accepted.velocityAndReferenceVolume.w;
        }
    }
    const float capacityTolerance = max(1.0e-12f, batch.x * 2.0e-6f);

    // Only compact when the pending batch cannot fit. This is bounded MPM
    // coarsening inside a half-cell radius, never a long-distance pool: choose
    // the closest compatible pair, preserve mass/reference volume, center-of-
    // mass momentum, APIC angular momentum, and total deformed volume. Exact
    // material history, F, and temperature preserve constitutive state; merged
    // deformed volume is capped at half a grid-cell volume. If no local pair is
    // admissible, fail closed and let frame rollback
    // restore the immutable accepted-step checkpoint.
    const float mergeRadius = 0.5f * h;
    const float mergeRadiusSquared = mergeRadius * mergeRadius;
    const float hSquared = h * h;
    if (valid && batch.x > capacityVolume + capacityTolerance) {
        for (uint mergePass = 0u;
             mergePass < d.bloodMpmParticleCount &&
             batch.x > capacityVolume + capacityTolerance;
             ++mergePass) {
            uint bestA = NM_INVALID_INDEX;
            uint bestB = NM_INVALID_INDEX;
            float bestDistanceSquared = mergeRadiusSquared;
            for (uint localA = 0u; localA < d.bloodMpmParticleCount; ++localA) {
                const uint indexA = d.bloodMpmFirstParticle + localA;
                const uint globalA = particleBase + indexA;
                const NMParticleStateGPU a = particleAccepted[globalA];
                if ((a.identity.w & NM_OBJECT_ACTIVE) == 0u) continue;
                for (uint localB = localA + 1u; localB < d.bloodMpmParticleCount; ++localB) {
                    const uint indexB = d.bloodMpmFirstParticle + localB;
                    const uint globalB = particleBase + indexB;
                    const NMParticleStateGPU b = particleAccepted[globalB];
                    if ((b.identity.w & NM_OBJECT_ACTIVE) == 0u) continue;
                    const float3 separation = b.positionAndMass.xyz - a.positionAndMass.xyz;
                    const float distanceSquared = dot(separation, separation);
                    if (!(distanceSquared < bestDistanceSquared) ||
                        !nmHumanBloodMergeStatesCompatible(
                            a, b, materialStateAccepted,
                            stateBase + indexA * d.materialStateStride,
                            stateBase + indexB * d.materialStateStride,
                            d.materialStateStride,
                            d.bloodMpmMaterialStateCount,
                            h, d.bloodMpmBulkModulus) ||
                        !nmHumanBloodMergeKineticAdmissible(a, b, h))
                        continue;
                    bestA = indexA;
                    bestB = indexB;
                    bestDistanceSquared = distanceSquared;
                }
            }
            if (bestA == NM_INVALID_INDEX || bestB == NM_INVALID_INDEX) break;

            const uint globalA = particleBase + bestA;
            const uint globalB = particleBase + bestB;
            NMParticleStateGPU a = particleAccepted[globalA];
            const NMParticleStateGPU b = particleAccepted[globalB];
            const float massA = a.positionAndMass.w;
            const float massB = b.positionAndMass.w;
            const float volumeA = a.velocityAndReferenceVolume.w;
            const float volumeB = b.velocityAndReferenceVolume.w;
            const float mergedMass = massA + massB;
            const float mergedVolume = volumeA + volumeB;
            if (!isfinite(mergedMass) || !(mergedMass > 0.0f) ||
                !isfinite(mergedVolume) || !(mergedVolume > 0.0f)) {
                valid = false;
                break;
            }

            const float3 momentumA = massA * a.velocityAndReferenceVolume.xyz;
            const float3 momentumB = massB * b.velocityAndReferenceVolume.xyz;
            const float3 totalMomentum = momentumA + momentumB;
            const float3 center = (massA * a.positionAndMass.xyz +
                massB * b.positionAndMass.xyz) / mergedMass;
            const float3 velocity = totalMomentum / mergedMass;

            const float3 omegaA = nmHumanBloodAngularVelocity(
                a.affineRow0, a.affineRow1, a.affineRow2);
            const float3 omegaB = nmHumanBloodAngularVelocity(
                b.affineRow0, b.affineRow1, b.affineRow2);
            const float3 angularMomentumAboutCenter =
                cross(a.positionAndMass.xyz - center, momentumA - massA * velocity) +
                cross(b.positionAndMass.xyz - center, momentumB - massB * velocity) +
                (0.5f * hSquared) * (massA * omegaA + massB * omegaB);
            const float3 omegaMerged =
                (2.0f / (mergedMass * hSquared)) * angularMomentumAboutCenter;
            float4 affineRow0 = (massA * a.affineRow0 + massB * b.affineRow0) / mergedMass;
            float4 affineRow1 = (massA * a.affineRow1 + massB * b.affineRow1) / mergedMass;
            float4 affineRow2 = (massA * a.affineRow2 + massB * b.affineRow2) / mergedMass;
            const float3 omegaAveraged = nmHumanBloodAngularVelocity(
                affineRow0, affineRow1, affineRow2);
            const float3 omegaCorrection = omegaMerged - omegaAveraged;
            affineRow0.y -= omegaCorrection.z;
            affineRow0.z += omegaCorrection.y;
            affineRow1.x += omegaCorrection.z;
            affineRow1.z -= omegaCorrection.x;
            affineRow2.x -= omegaCorrection.y;
            affineRow2.y += omegaCorrection.x;
            const float3x3 mergedAffine =
                nmHumanBloodMatrixFromRows(affineRow0, affineRow1, affineRow2);

            float3x3 mergedDeformation;
            float bulkEnergyLoss=0.0f;
            if(!nmHumanBloodMergeDeformation(
                    a,b,h,d.bloodMpmBulkModulus,
                    mergedDeformation,bulkEnergyLoss)) {
                valid=false;
                break;
            }
            float4 deformationRow0;
            float4 deformationRow1;
            float4 deformationRow2;
            nmHumanBloodMatrixToRows(mergedDeformation,
                deformationRow0, deformationRow1, deformationRow2);
            nmHumanBloodMatrixToRows(mergedAffine, affineRow0, affineRow1, affineRow2);

            const float kineticBefore =
                0.5f * massA * dot(a.velocityAndReferenceVolume.xyz,
                    a.velocityAndReferenceVolume.xyz) +
                0.5f * massB * dot(b.velocityAndReferenceVolume.xyz,
                    b.velocityAndReferenceVolume.xyz) +
                (hSquared / 8.0f) * (
                    massA * nmHumanBloodAffineNormSquared(
                        a.affineRow0, a.affineRow1, a.affineRow2) +
                    massB * nmHumanBloodAffineNormSquared(
                        b.affineRow0, b.affineRow1, b.affineRow2));
            const float kineticAfter =
                0.5f * mergedMass * dot(velocity, velocity) +
                (mergedMass * hSquared / 8.0f) *
                    nmHumanBloodAffineNormSquared(
                        affineRow0, affineRow1, affineRow2);
            const float signedKineticLoss=kineticBefore-kineticAfter;
            const float kineticTolerance=max(1.0e-30f,
                16.0f*1.1920928955078125e-7f*max(kineticBefore,kineticAfter));
            if (!isfinite(kineticBefore) || !isfinite(kineticAfter) ||
                !isfinite(signedKineticLoss)||signedKineticLoss < -kineticTolerance) {
                // The candidate scan applies the same test; this second check
                // protects the write path against changed/corrupt state.
                valid = false;
                break;
            }

            const float mergeLoss=max(0.0f,signedKineticLoss);
            const float totalMergeLoss=mergeLoss+bulkEnergyLoss;
            const float energyAdjusted = totalMergeLoss - n.externalBlood.energyLedger.z;
            const float energyNext = n.externalBlood.energyLedger.y + energyAdjusted;
            if(!isfinite(totalMergeLoss)||totalMergeLoss<0.0f||
               !isfinite(energyAdjusted)||!isfinite(energyNext)) {
                valid=false;
                break;
            }

            a.positionAndMass = float4(center, mergedMass);
            a.velocityAndReferenceVolume = float4(velocity, mergedVolume);
            a.deformationRow0 = deformationRow0;
            a.deformationRow1 = deformationRow1;
            a.deformationRow2 = deformationRow2;
            a.affineRow0 = affineRow0;
            a.affineRow1 = affineRow1;
            a.affineRow2 = affineRow2;
            a.referenceAndTemperature =
                (volumeA / mergedVolume) * a.referenceAndTemperature +
                (volumeB / mergedVolume) * b.referenceAndTemperature;
            particleAccepted[globalA] = a;
            particleCandidate[globalA] = a;

            NMParticleStateGPU dormant = b;
            dormant.positionAndMass =
                float4(center, d.bloodMpmDensity * volumeB);
            dormant.velocityAndReferenceVolume = float4(0.0f, 0.0f, 0.0f, volumeB);
            dormant.deformationRow0 = float4(1.0f, 0.0f, 0.0f, 0.0f);
            dormant.deformationRow1 = float4(0.0f, 1.0f, 0.0f, 0.0f);
            dormant.deformationRow2 = float4(0.0f, 0.0f, 1.0f, 0.0f);
            dormant.affineRow0 = float4(0.0f);
            dormant.affineRow1 = float4(0.0f);
            dormant.affineRow2 = float4(0.0f);
            dormant.referenceAndTemperature =
                float4(center, a.referenceAndTemperature.w);
            dormant.identity.w &= ~NM_OBJECT_ACTIVE;
            particleAccepted[globalB] = dormant;
            particleCandidate[globalB] = dormant;
            const uint materialBaseB =
                stateBase + bestB * d.materialStateStride;
            for (uint stateIndex = 0u; stateIndex < d.materialStateStride; ++stateIndex) {
                const float initial = materialStateDefaults[materialBaseB + stateIndex];
                materialStateAccepted[materialBaseB + stateIndex] = initial;
                materialStateCandidate[materialBaseB + stateIndex] = initial;
            }

            n.externalBlood.energyLedger.z =
                (energyNext - n.externalBlood.energyLedger.y) - energyAdjusted;
            n.externalBlood.energyLedger.y = energyNext;
            capacityVolume += volumeB;
        }
    }
    if (!isfinite(capacityVolume) || batch.x > capacityVolume + capacityTolerance)
        valid = false;

    if (!valid) {
        if (statuses[env].code == NM_STATUS_SUCCESS) {
            statuses[env].code = NM_STATUS_MULTIPHYSICS_FAILURE;
            statuses[env].environment = env;
            statuses[env].objectIndex = d.bloodMpmObjectIndex;
            statuses[env].failingIndex = d.bloodMpmFirstParticle;
            statuses[env].diagnostics = float4(batch.x, batch.y, capacityVolume, h);
        }
        n.status.w = 11u;
        respirationCandidate[env] = n;
        return;
    }

    const float mass = batch.y;
    const float3 velocity = momentum / mass;
    const float3 omega = 2.0f * intrinsicAngular / (mass * h * h);
    float remainingVolume = batch.x;
    float remainingMass = batch.y;
    for (uint local = 0u; local < d.bloodMpmParticleCount; ++local) {
        if (!(remainingVolume > 0.0f)) break;
        const uint index = d.bloodMpmFirstParticle + local;
        const uint global = particleBase + index;
        NMParticleStateGPU particle = particleAccepted[global];
        if ((particle.identity.w & NM_OBJECT_ACTIVE) != 0u) continue;
        const float slotVolume = particle.velocityAndReferenceVolume.w;
        const float assignedVolume = min(slotVolume, remainingVolume);
        const bool lastSlot = assignedVolume >= remainingVolume;
        const float assignedMass = lastSlot
            ? remainingMass : batch.y * (assignedVolume / batch.x);
        particle.positionAndMass = float4(spawn, assignedMass);
        particle.velocityAndReferenceVolume =
            float4(velocity, assignedVolume);
        particle.deformationRow0 = float4(1.0f, 0.0f, 0.0f, 0.0f);
        particle.deformationRow1 = float4(0.0f, 1.0f, 0.0f, 0.0f);
        particle.deformationRow2 = float4(0.0f, 0.0f, 1.0f, 0.0f);
        particle.affineRow0 = float4(0.0f, -omega.z, omega.y, 0.0f);
        particle.affineRow1 = float4(omega.z, 0.0f, -omega.x, 0.0f);
        particle.affineRow2 = float4(-omega.y, omega.x, 0.0f, 0.0f);
        particle.referenceAndTemperature.xyz = spawn;
        particle.identity.w |= NM_OBJECT_ACTIVE;
        particleAccepted[global] = particle;
        particleCandidate[global] = particle;
        const uint materialBase = stateBase + index * d.materialStateStride;
        for (uint stateIndex = 0u; stateIndex < d.materialStateStride; ++stateIndex) {
            const float initial = materialStateDefaults[materialBase + stateIndex];
            materialStateAccepted[materialBase + stateIndex] = initial;
            materialStateCandidate[materialBase + stateIndex] = initial;
        }
        remainingVolume -= assignedVolume;
        remainingMass -= assignedMass;
    }
    if (remainingVolume > capacityTolerance || remainingMass > max(1.0e-12f, batch.y * 4.0e-6f)) {
        // Capacity was validated before any writes; this guard is defensive
        // against nonfinite FP32 accumulation or corrupted slot metadata.
        statuses[env].code = NM_STATUS_MULTIPHYSICS_FAILURE;
        statuses[env].environment = env;
        statuses[env].objectIndex = d.bloodMpmObjectIndex;
        statuses[env].failingIndex = d.bloodMpmFirstParticle;
        n.status.w = 11u;
        respirationCandidate[env] = n;
        return;
    }

    const float3 spinOmega = omega;
    const float angularEnergy = mass * h * h * dot(spinOmega, spinOmega) * 0.25f;
    const float kineticAfter = 0.5f * dot(momentum, momentum) / mass + angularEnergy;
    const float mergeLoss = max(0.0f, n.externalBlood.energyLedger.x - kineticAfter);
    const float energyAdjusted = mergeLoss - n.externalBlood.energyLedger.z;
    const float energyNext = n.externalBlood.energyLedger.y + energyAdjusted;
    n.externalBlood.energyLedger.z =
        (energyNext - n.externalBlood.energyLedger.y) - energyAdjusted;
    n.externalBlood.energyLedger.y = energyNext;
    const float4 transfer = batch - n.externalBlood.mpmRepresentedCompensation;
    const float4 representedNext = n.externalBlood.mpmRepresented + transfer;
    n.externalBlood.mpmRepresentedCompensation =
        (representedNext - n.externalBlood.mpmRepresented) - transfer;
    n.externalBlood.mpmRepresented = representedNext;
    n.externalBlood.pendingParcel = 0.0f;
    n.externalBlood.pendingMomentum = 0.0f;
    n.externalBlood.pendingPositionMass = 0.0f;
    n.externalBlood.pendingAngularMomentum = 0.0f;
    n.externalBlood.energyLedger.x = 0.0f;
    respirationCandidate[env] = n;
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
    if(kUseRespiratorySubcycling) {
        // Validate drive at staging even when mechanics are deferred. A latched
        // physical rejection skips exchange for the remaining command-buffer
        // suffix, so invalid Brain delivery must remain a rejected candidate.
        if(!all(isfinite(n.control.xy)) || any(n.control.xy<0.0f) ||
           any(n.control.xy>1.0f)) n.status.w=1u;
        // Circuit physics does not consume respiratory mechanics at frameBegin.
        // Stage the drive now; the exchange kernel integrates mechanics and gas
        // together with the solved circuit flow before the common geometry gate.
        if(d.reject) n.status.w=3u;
    } else advanceHumanRespiratoryMechanics(n,p,p.environment.w,d.reject!=0u);
    candidate[env]=n;
}

kernel void nm_human_respiration_exchange(
    constant NMHumanRespirationParameters& p [[buffer(0)]],
    constant NMHumanRespirationDispatch& d [[buffer(1)]],
    device const NMHumanRespirationState* accepted [[buffer(2)]],
    device NMHumanRespirationState* candidate [[buffer(3)]],
    device const float4* vascularBefore [[buffer(4)]],
    device float4* vascularAfter [[buffer(5)]],
    device const NMVascularUnknownGPU* unknowns [[buffer(6)]],
    device const NMVascularConnectionGPU* connections [[buffer(7)]],
    device NMMatterStatusGPU* statuses [[buffer(8)]],
    device const NMVascularCompartmentGPU* compartments [[buffer(9)]],
    device const float* elastance [[buffer(10)]],
    device const MRArticulatedBodyPoseGPU* bodyPosesAccepted [[buffer(11)]],
    device MRBodyPropertiesGPU* bodyPropertiesCandidate [[buffer(12)]],
    device const float4* bodyPositionLow [[buffer(13)]],
    device const float* pointJacobians [[buffer(14)]],
    device const float* generalizedVelocity [[buffer(15)]],
    device float* mujocoGeneralizedForceArena [[buffer(16)]],
    device const MRBodyPropertiesGPU* bodyPropertiesReference [[buffer(17)]],
    uint globalIndex [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdWidth [[threads_per_simdgroup]]) {
    const uint env=kUseParallelRespiratoryMuscles?globalIndex/simdWidth:globalIndex;
    const bool publish=!kUseParallelRespiratoryMuscles||lane==0u;
    if(env>=d.environmentCount) return;
    NMHumanRespirationState n=candidate[env];
    if(statuses[env].code!=NM_STATUS_SUCCESS) return;
    const NMHumanRespirationState old=accepted[env];
    const float dt=p.environment.w;
    const uint base=env*d.vascularStride;
    // Gas advection is local to this candidate. Every substep uses the same
    // accepted cardiovascular flow and linearly interpolated compartment
    // volumes, consistent with the circuit's backward-Euler volume balance.
    // No controller, body, circuit, or accepted cursor advances here.
    uint gasSubsteps=1u;
    float outgoing[21];
    for(uint row=0;row<21;++row) outgoing[row]=0.0f;
    for(uint edge=0;edge<p.topology.w;++edge) {
        const auto con=connections[edge];
        const float flow=human_respiration::physical(vascularAfter,unknowns,base,21+edge);
        const uint from=flow>=0?con.identity.y:con.identity.z;
        outgoing[from]+=abs(flow);
    }
    // Same accepted cut event as the Human separation control step. This is an
    // operator-split reduced orifice law: pressure and area set the attempted
    // outflow; the donor debit is bounded and enters the accepted-state ledger.
    NMHumanVascularCutEventGPU outlet=n.vascularCutOutlet;
    const bool cutReached=d.outletReleaseStep!=NM_INVALID_INDEX &&
        d.controlStep>=d.outletReleaseStep;
    if(cutReached) outlet.identity.w=1u;
    float outletFlow=0.0f;
    float outletVolume=0.0f;
    uint outletRow=uint(max(0.0f,outlet.fluid.w));
    if(outlet.identity.w && outletRow<21u && outlet.identity.x==9u && outlet.identity.y==7u) {
        const float donorVolume=human_respiration::physical(vascularAfter,unknowns,base,outletRow);
        const auto donor=compartments[outletRow];
        float pressure=donor.compliance.w+donor.compliance.y+
            (donorVolume-donor.compliance.x)*elastance[env*21u+outletRow];
        if(donor.identity.w==3u) {
            const float displacement=donorVolume-donor.compliance.x;
            const float angle=(0.5f*donor.waveform.x)*(displacement/donor.pressureParameters.x);
            pressure=donor.compliance.w+donor.compliance.y+
                (2.0f*donor.pressureParameters.x/donor.waveform.x/donor.compliance.z)*tan(angle);
        }
        const float deltaPressure=max(0.0f,pressure-outlet.fluid.z);
        if(isfinite(pressure)&&isfinite(deltaPressure)&&outlet.aperture.x>0.0f&&
           outlet.fluid.x>0.0f&&outlet.fluid.y>=0.0f) {
            outletFlow=outlet.fluid.y*outlet.aperture.x*sqrt(2.0f*deltaPressure/outlet.fluid.x);
            const float available=max(0.0f,donorVolume-donor.compliance.x);
            outletVolume=min(dt*outletFlow,min(0.1f*donorVolume,available));
            outletFlow=outletVolume/dt;
            outgoing[outletRow]+=outletFlow;
        }
    }
    if(kUseGasTransportSubcycling) {
        float required=kUseRespiratorySubcycling?max(1.0f,ceil(dt/0.002f)):1.0f;
        for(uint row=0;row<21;++row) {
            const float before=human_respiration::physical(vascularBefore,unknowns,base,row);
            const float after=human_respiration::physical(vascularAfter,unknowns,base,row);
            const float minimumVolume=min(before,after);
            if(!(minimumVolume>0.0f)||!isfinite(outgoing[row])) {
                n.status.w=4u;
                if(publish) {statuses[env].code=NM_STATUS_MULTIPHYSICS_FAILURE;candidate[env]=n;}
                return;
            }
            required=max(required,ceil(dt*outgoing[row]/(0.1f*minimumVolume)));
        }
        const float airwayRequired=ceil(dt*abs(n.mechanics.w)/(0.1f*p.lung.y));
        if(!isfinite(required)||required>32.0f||
           !isfinite(airwayRequired)||airwayRequired>32.0f) {
            n.status.w=(!isfinite(required)||required>32.0f)?4u:6u;
            if(publish) {statuses[env].code=NM_STATUS_MULTIPHYSICS_FAILURE;candidate[env]=n;}
            return;
        }
        gasSubsteps=uint(max(required,airwayRequired));
    }
    const float gasDt=dt/float(gasSubsteps);
    float4 compartmentGas=float4(0.0f);
    if(kUseParallelRespiratoryGas&&lane<21u) compartmentGas=n.bloodGas[lane];
    float pulmonaryO2=0;
    float totalPulmonaryO2=0;
    float netSweptVolume=0.0f, inspiredVolume=0.0f, maxDonorFraction=0.0f;
    float maxAirwayTransport=0.0f;
    for(uint substep=0u;substep<gasSubsteps;++substep) {
        const float alpha=float(substep)/float(gasSubsteps);
        const float mechanicsStartVolume=n.mechanics.x;
        if(kUseRespiratorySubcycling) {
            if(n.status.w) break;
            advanceHumanRespiratoryMechanics(n,p,gasDt,false,lane,kUseParallelRespiratoryMuscles);
            if(n.status.w) break;
            n.breath.x=max(n.breath.x,n.mechanics.x);
            n.breath.y=min(n.breath.y,n.mechanics.x);
        }
        netSweptVolume+=gasDt*n.mechanics.w;
        inspiredVolume+=gasDt*max(0.0f,n.mechanics.w);
        float2 delta[21];
        float2 compartmentDelta=0.0f;
        if(!kUseParallelRespiratoryGas)
            for(uint row=0;row<21;++row) delta[row]=0;
        // Check the actual rounded aggregate fraction, not just the count
        // estimate or each individual edge at a branching vascular node.
        for(uint row=0;row<21;++row) {
            const float before=human_respiration::physical(vascularBefore,unknowns,base,row);
            const float donorVolume=gasSubsteps==1u?before:mix(before,
                human_respiration::physical(vascularAfter,unknowns,base,row),alpha);
            maxDonorFraction=max(maxDonorFraction,gasDt*outgoing[row]/donorVolume);
            if(kUseGasTransportSubcycling&&
               (!(donorVolume>0.0f)||gasDt*outgoing[row]>0.1f*donorVolume))
                n.status.w=4u;
        }
        if(kUseGasTransportSubcycling&&n.status.w) break;
        float2 alveolarDelta=0;
        const float lungVolume=kUseRespiratorySubcycling?mechanicsStartVolume:
            (gasSubsteps==1u?old.mechanics.x:mix(old.mechanics.x,n.mechanics.x,alpha));
        const float alveolarVolume=lungVolume-p.lung.y;
        const float2 alveolarFraction=n.alveolarGas.xy/(alveolarVolume*p.environment.z);
        const float2 deadFraction=n.deadSpaceGas.xy/(p.lung.y*p.environment.z);
        const float2 alveolarPressure=alveolarFraction*(p.environment.y/133.322387415f);
        const float2 capillaryContent=float2(human_respiration::oxygenContent(alveolarPressure.x,p),
            p.carbonDioxide.x+p.carbonDioxide.y*(alveolarPressure.y-40.0f));
        for(uint edge=0;edge<p.topology.w;++edge) {
            const auto con=connections[edge];
            const float flow=human_respiration::physical(vascularAfter,unknowns,base,21+edge);
            const uint from=flow>=0?con.identity.y:con.identity.z;
            const uint to=flow>=0?con.identity.z:con.identity.y;
            const float before=human_respiration::physical(vascularBefore,unknowns,base,from);
            const float fromVolume=gasSubsteps==1u?before:mix(before,
                human_respiration::physical(vascularAfter,unknowns,base,from),alpha);
            const float2 donorGas=kUseParallelRespiratoryGas
                ?simd_shuffle(compartmentGas.xy,from):n.bloodGas[from].xy;
            const float2 content=donorGas/fromVolume;
            const float2 transported=gasDt*abs(flow)*content;
            if(kUseParallelRespiratoryGas) {
                if(lane==from) compartmentDelta-=transported;
                if(lane==to) compartmentDelta+=transported;
            } else {delta[from]-=transported;delta[to]+=transported;}
            if(edge==p.topology.x && flow>0) {
                // Same perfusion-limited exchange; its amount cancels exactly
                // between the destination blood and alveolar reservoirs.
                const float2 exchange=gasDt*flow*p.carbonDioxide.z*(capillaryContent-content);
                if(kUseParallelRespiratoryGas) {
                    if(lane==to) compartmentDelta+=exchange;
                } else delta[to]+=exchange;
                alveolarDelta-=exchange;
                if(gasSubsteps==1u) pulmonaryO2=exchange.x/dt;
                else totalPulmonaryO2+=exchange.x;
            }
            if(gasDt*abs(flow)>0.1f*fromVolume || !all(isfinite(transported))) n.status.w=4;
        }
        for(uint bed=0;bed<4;++bed) {
            const float2 demand=gasDt*p.metabolism.zw*p.tissueFractions[bed];
            if(kUseParallelRespiratoryGas) {
                if(lane==p.tissueRows[bed]) compartmentDelta+=float2(-demand.x,demand.y);
            } else delta[p.tissueRows[bed]]+=float2(-demand.x,demand.y);
        }
        human_respiration::addGas(n.metabolicGas,gasDt*p.metabolism.zw);
        const float subTransport=gasDt*n.mechanics.w*p.environment.z;
        maxAirwayTransport=max(maxAirwayTransport,abs(subTransport));
        const float2 outsideFlux=subTransport*(subTransport>=0?p.metabolism.xy:deadFraction);
        const float2 airwayFlux=subTransport*(subTransport>=0?deadFraction:alveolarFraction);
        human_respiration::addGas(n.environmentGas,outsideFlux);
        human_respiration::addGas(n.deadSpaceGas,outsideFlux-airwayFlux);
        human_respiration::addGas(n.alveolarGas,airwayFlux+alveolarDelta);
        if(kUseParallelRespiratoryGas) {
            bool invalid=false;
            if(lane<21u) {
                human_respiration::addGas(compartmentGas,compartmentDelta);
                invalid=!all(isfinite(compartmentGas))||any(compartmentGas.xy<0);
            }
            if(simd_any(invalid)) n.status.w=5;
        } else {
            for(uint row=0;row<21;++row) {
                human_respiration::addGas(n.bloodGas[row],delta[row]);
                if(!all(isfinite(n.bloodGas[row]))||any(n.bloodGas[row].xy<0)) n.status.w=5;
            }
        }
        if(!all(isfinite(n.alveolarGas))||any(n.alveolarGas.xy<0)||
           any(n.deadSpaceGas.xy<0)||
           abs(subTransport)>0.1f*p.lung.y*p.environment.z) n.status.w=6;
        if(n.status.w) break;
    }
    if(kUseParallelRespiratoryGas)
        for(uint row=0u;row<21u;++row)
            n.bloodGas[row]=simd_broadcast(compartmentGas,row);
    if(gasSubsteps>1u) pulmonaryO2=totalPulmonaryO2/dt;
    n.vascularCutOutlet=outlet;
    if(publish && outletVolume>0.0f) {
        const float donorBefore=human_respiration::physical(vascularAfter,unknowns,base,outletRow);
        const float donorScale=unknowns[outletRow].initialAndScaling.y;
        const float2 donorContent=n.bloodGas[outletRow].xy/donorBefore;
        const float2 removedGas=donorContent*outletVolume;
        if(!(donorBefore>outletVolume)||!(donorScale>0.0f)||
           !all(isfinite(removedGas))||any(removedGas<0.0f)||
           any(removedGas>n.bloodGas[outletRow].xy)) {
            n.status.w=8u;
        } else {
            vascularAfter[base+outletRow].x-=(outletVolume/donorScale);
            human_respiration::addGas(n.bloodGas[outletRow],-removedGas);
            const float removedMass=outlet.fluid.x*outletVolume;
            const float4 transfer=float4(outletVolume,removedMass,removedGas);
            float4 y=transfer-n.externalBlood.compensation;
            float4 next=n.externalBlood.total+y;
            n.externalBlood.compensation=(next-n.externalBlood.total)-y;
            n.externalBlood.total=next;
            const float externalMass=n.externalBlood.total.y-n.externalBlood.compensation.y;
            if(!all(isfinite(next))||any(next<0.0f)||!isfinite(externalMass))
                n.status.w=8u;

            // The cut surface is attached to runtime abdomen body 7. Blood
            // mass is spatially redistributed over every positive-mass body
            // property because CVSim21 supplies no segment perfusion map.
            // Segment COM positions are preserved; this is a homogeneous
            // reference-mass approximation, not a patient-specific map.
            const uint bodyIndex=outlet.identity.z;
            if(bodyIndex<d.articulationFirstBody||
               bodyIndex-d.articulationFirstBody>=d.bodyPoseStride||
               bodyIndex>=d.bodyPropertyCount||d.dofCount==0u||
               d.dofCount>MR_ARTICULATED_OPERATOR_KINEMATICS_MAX_DOFS) {
                n.status.w=10u;
            } else {
                float referenceTotalMass=0.0f;
                float referenceMassCorrection=0.0f;
                bool validMass=true;
                for(uint bodyOrdinal=0u;bodyOrdinal<d.bodyPropertyCount;++bodyOrdinal) {
                    const float mass=bodyPropertiesReference[bodyOrdinal].massAndInverseMass.x;
                    if(!isfinite(mass)||mass<0.0f) {validMass=false;break;}
                    if(mass>0.0f) {
                        const float adjusted=mass-referenceMassCorrection;
                        const float sum=referenceTotalMass+adjusted;
                        referenceMassCorrection=(sum-referenceTotalMass)-adjusted;
                        referenceTotalMass=sum;
                    }
                }
                const float remainingMass=referenceTotalMass-externalMass;
                if(!validMass||!(referenceTotalMass>0.0f)||!(remainingMass>0.0f)||
                   !isfinite(referenceTotalMass)||!isfinite(remainingMass)) {
                    n.status.w=10u;
                } else {
                    const float remainingFraction=remainingMass/referenceTotalMass;
                    const float transferFraction=removedMass/referenceTotalMass;
                    float3 removedBodyMomentum=0.0f;
                    float3 removedBodyAngularMomentum=0.0f;
                    float removedBodyKineticEnergy=0.0f;
                    bool validKinematics=true;

                    // Momentum carried by the spatially distributed blood is
                    // removed at each segment COM, including the lost angular
                    // momentum of its proportionally scaled reference inertia.
                    for(uint bodyOrdinal=0u;bodyOrdinal<d.bodyPropertyCount;++bodyOrdinal) {
                        const MRBodyPropertiesGPU reference=bodyPropertiesReference[bodyOrdinal];
                        const float referenceMass=reference.massAndInverseMass.x;
                        if(!(referenceMass>0.0f)) continue;
                        if(bodyOrdinal<d.articulationFirstBody||
                           bodyOrdinal-d.articulationFirstBody>=d.bodyPoseStride) {
                            validKinematics=false;
                            break;
                        }
                        const uint localBody=bodyOrdinal-d.articulationFirstBody;
                        const uint poseIndex=env*d.bodyPoseStride+localBody;
                        const MRArticulatedBodyPoseGPU body=bodyPosesAccepted[poseIndex];
                        const float3 worldCOM=body.position.xyz+bodyPositionLow[poseIndex].xyz;
                        const float3 worldAxes[3]={
                            restingRotate(body.orientation,float3(1,0,0)),
                            restingRotate(body.orientation,float3(0,1,0)),
                            restingRotate(body.orientation,float3(0,0,1))
                        };
                        const uint probe=d.bodyJacobianPointOffset+4u*localBody;
                        const uint centerBase=env*d.pointJacobianStride+probe*3u*d.dofCount;
                        float3 centerVelocity=0.0f;
                        float3 angularVelocity=0.0f;
                        for(uint dof=0u;dof<d.dofCount;++dof) {
                            const float3 centerJ=float3(
                                pointJacobians[centerBase+dof],
                                pointJacobians[centerBase+d.dofCount+dof],
                                pointJacobians[centerBase+2u*d.dofCount+dof]);
                            float3 angularJ=0.0f;
                            for(uint axis=0u;axis<3u;++axis) {
                                const uint axisBase=centerBase+(axis+1u)*3u*d.dofCount;
                                const float3 axisJ=float3(
                                    pointJacobians[axisBase+dof],
                                    pointJacobians[axisBase+d.dofCount+dof],
                                    pointJacobians[axisBase+2u*d.dofCount+dof]);
                                angularJ+=0.5f*cross(worldAxes[axis],axisJ-centerJ);
                            }
                            const float velocity=generalizedVelocity[env*d.vStride+dof];
                            centerVelocity+=centerJ*velocity;
                            angularVelocity+=angularJ*velocity;
                        }
                        const float removedSegmentMass=removedMass*referenceMass/referenceTotalMass;
                        removedBodyMomentum+=removedSegmentMass*centerVelocity;
                        removedBodyAngularMomentum+=cross(worldCOM,removedSegmentMass*centerVelocity);
                        const float4 inverseOrientation=float4(-body.orientation.xyz,body.orientation.w);
                        const float3 omegaLocal=restingRotate(inverseOrientation,angularVelocity);
                        const float3 inertiaOmegaLocal=float3(
                            dot(reference.inertiaRow0.xyz,omegaLocal),
                            dot(reference.inertiaRow1.xyz,omegaLocal),
                            dot(reference.inertiaRow2.xyz,omegaLocal));
                        const float3 removedSpin=restingRotate(body.orientation,
                            transferFraction*inertiaOmegaLocal);
                        removedBodyAngularMomentum+=removedSpin;
                        removedBodyKineticEnergy+=0.5f*removedSegmentMass*
                            dot(centerVelocity,centerVelocity)+
                            0.5f*transferFraction*dot(omegaLocal,inertiaOmegaLocal);
                    }
                    if(!validKinematics) {
                        n.status.w=10u;
                    } else {
                        // Reconstruct each candidate mass and inertia from the
                        // immutable baseline. Inverse mass remains zero for a
                        // baseline-fixed segment.
                        for(uint bodyOrdinal=0u;bodyOrdinal<d.bodyPropertyCount;++bodyOrdinal) {
                            const MRBodyPropertiesGPU reference=bodyPropertiesReference[bodyOrdinal];
                            const float referenceMass=reference.massAndInverseMass.x;
                            if(!(referenceMass>0.0f)) continue;
                            device MRBodyPropertiesGPU& properties=bodyPropertiesCandidate[bodyOrdinal];
                            const float newMass=referenceMass*remainingFraction;
                            const float ratio=newMass/referenceMass;
                            if(!(newMass>0.0f)||!isfinite(newMass)||!(ratio>0.0f)) {
                                n.status.w=10u;
                                break;
                            }
                            const float inverseMass=reference.massAndInverseMass.y>0.0f
                                ?1.0f/newMass:reference.massAndInverseMass.y;
                            properties.massAndInverseMass=float4(newMass,inverseMass,0.0f,0.0f);
                            properties.inertiaRow0.xyz=reference.inertiaRow0.xyz*ratio;
                            properties.inertiaRow1.xyz=reference.inertiaRow1.xyz*ratio;
                            properties.inertiaRow2.xyz=reference.inertiaRow2.xyz*ratio;
                            properties.inverseInertiaRow0.xyz=reference.inverseInertiaRow0.xyz/ratio;
                            properties.inverseInertiaRow1.xyz=reference.inverseInertiaRow1.xyz/ratio;
                            properties.inverseInertiaRow2.xyz=reference.inverseInertiaRow2.xyz/ratio;
                        }

                        const uint localOutletBody=bodyIndex-d.articulationFirstBody;
                        const uint outletPoseIndex=env*d.bodyPoseStride+localOutletBody;
                        const MRArticulatedBodyPoseGPU outletBody=bodyPosesAccepted[outletPoseIndex];
                        const MRBodyPropertiesGPU outletProperties=bodyPropertiesReference[bodyIndex];
                        const float3 localOffset=outlet.sourcePointAndGround.xyz-
                            outletProperties.centerOfMass.xyz;
                        const float3 worldOffset=restingRotate(outletBody.orientation,localOffset);
                        const float3 worldCOM=outletBody.position.xyz+bodyPositionLow[outletPoseIndex].xyz;
                        const float3 worldPoint=worldCOM+worldOffset;
                        const float3 worldNormal=normalize(restingRotate(
                            outletBody.orientation,outlet.aperture.yzw));
                        const float relativeJetSpeed=outletFlow/outlet.aperture.x;
                        float3 outletPointJacobian[MR_ARTICULATED_OPERATOR_KINEMATICS_MAX_DOFS];
                        float3 outletAngularJacobian[MR_ARTICULATED_OPERATOR_KINEMATICS_MAX_DOFS];
                        float3 outletPointVelocity=0.0f;
                        const float3 outletAxes[3]={
                            restingRotate(outletBody.orientation,float3(1,0,0)),
                            restingRotate(outletBody.orientation,float3(0,1,0)),
                            restingRotate(outletBody.orientation,float3(0,0,1))
                        };
                        const uint outletProbe=d.bodyJacobianPointOffset+4u*localOutletBody;
                        const uint outletCenterBase=env*d.pointJacobianStride+
                            outletProbe*3u*d.dofCount;
                        for(uint dof=0u;dof<d.dofCount;++dof) {
                            const float3 centerJ=float3(
                                pointJacobians[outletCenterBase+dof],
                                pointJacobians[outletCenterBase+d.dofCount+dof],
                                pointJacobians[outletCenterBase+2u*d.dofCount+dof]);
                            float3 angularJ=0.0f;
                            for(uint axis=0u;axis<3u;++axis) {
                                const uint axisBase=outletCenterBase+(axis+1u)*3u*d.dofCount;
                                const float3 axisJ=float3(
                                    pointJacobians[axisBase+dof],
                                    pointJacobians[axisBase+d.dofCount+dof],
                                    pointJacobians[axisBase+2u*d.dofCount+dof]);
                                angularJ+=0.5f*cross(outletAxes[axis],axisJ-centerJ);
                            }
                            const float3 pointJ=centerJ+cross(angularJ,worldOffset);
                            outletPointJacobian[dof]=pointJ;
                            outletAngularJacobian[dof]=angularJ;
                            outletPointVelocity+=pointJ*generalizedVelocity[env*d.vStride+dof];
                        }
                        const float3 parcelVelocity=outletPointVelocity+worldNormal*relativeJetSpeed;
                        const float3 parcelMomentum=removedMass*parcelVelocity;
                        const float3 recoilImpulse=removedBodyMomentum-parcelMomentum;
                        const float3 parcelAngularMomentum=cross(worldPoint,parcelMomentum);
                        const float3 recoilTorque=removedBodyAngularMomentum-parcelAngularMomentum-
                            cross(worldPoint,recoilImpulse);
                        const float3 momentumResidual=removedBodyMomentum-parcelMomentum-recoilImpulse;
                        const float3 angularResidual=removedBodyAngularMomentum-parcelAngularMomentum-
                            cross(worldPoint,recoilImpulse)-recoilTorque;
                        for(uint dof=0u;dof<d.dofCount;++dof) {
                            const uint forceIndex=env*d.generalizedForceStride+
                                d.generalizedForceOffset+dof;
                            mujocoGeneralizedForceArena[forceIndex]+=
                                (dot(outletPointJacobian[dof],recoilImpulse)+
                                 dot(outletAngularJacobian[dof],recoilTorque))/dt;
                        }
                        n.externalBlood.jetImpulse.xyz+=recoilImpulse;
                        n.externalBlood.jetTorqueImpulse.xyz+=recoilTorque;
                        n.externalBlood.transferMomentumResidual=float4(
                            momentumResidual,dot(momentumResidual,momentumResidual));
                        n.externalBlood.transferAngularResidual=float4(
                            angularResidual,dot(angularResidual,angularResidual));
                        n.externalBlood.pendingParcel+=transfer;
                        n.externalBlood.pendingMomentum.xyz+=parcelMomentum;
                        n.externalBlood.pendingPositionMass.xyz+=removedMass*worldPoint;
                        n.externalBlood.pendingPositionMass.w+=removedMass;
                        n.externalBlood.pendingAngularMomentum.xyz+=parcelAngularMomentum;
                        const float parcelKineticEnergy=0.5f*removedMass*
                            dot(parcelVelocity,parcelVelocity);
                        n.externalBlood.energyLedger.x+=parcelKineticEnergy;
                        n.externalBlood.energyLedger.w+=removedBodyKineticEnergy-parcelKineticEnergy;
                    }
                }
            }
        }
    }
    // Emit one physically aggregated packet no more often than each admitted
    // 8 ms batch. The barycentric velocity preserves accumulated linear
    // momentum; intrinsic angular momentum carries the unresolved within-batch
    // moment. Kinetic energy lost by merging is ledgered as dissipation.
    const bool batchDue=d.outletBatchSteps>0u&&d.controlStep>=d.outletReleaseStep&&
        ((d.controlStep-d.outletReleaseStep+1u)%d.outletBatchSteps)==0u;
    if(d.bloodMpmEnabled==0u&&publish&&batchDue&&n.externalBlood.pendingParcel.y>0.0f) {
        uint freeParcel=NM_HUMAN_EXTERNAL_BLOOD_PARCEL_CAPACITY;
        for(uint i=0;i+1u<NM_HUMAN_EXTERNAL_BLOOD_PARCEL_CAPACITY;++i)
            if(n.externalBlood.parcels[i].gasAndFlags.z<0.5f) {freeParcel=i;break;}
        const float mass=n.externalBlood.pendingParcel.y;
        if(freeParcel+1u>=NM_HUMAN_EXTERNAL_BLOOD_PARCEL_CAPACITY||
           !(mass>0.0f)||!(n.externalBlood.pendingPositionMass.w>0.0f)||
           !isfinite(mass)) {
            n.status.w=9u;
        } else {
            const float3 spawn=n.externalBlood.pendingPositionMass.xyz/mass+
                0.002f*normalize(restingRotate(bodyPosesAccepted[
                    env*d.bodyPoseStride+outlet.identity.z-d.articulationFirstBody].orientation,
                    outlet.aperture.yzw));
            const float3 momentum=n.externalBlood.pendingMomentum.xyz;
            const float kineticAfter=0.5f*dot(momentum,momentum)/mass;
            const float mergeLoss=max(0.0f,n.externalBlood.energyLedger.x-kineticAfter);
            float energyAdjusted=mergeLoss-n.externalBlood.energyLedger.z;
            float energyNext=n.externalBlood.energyLedger.y+energyAdjusted;
            n.externalBlood.energyLedger.z=(energyNext-n.externalBlood.energyLedger.y)-energyAdjusted;
            n.externalBlood.energyLedger.y=energyNext;
            thread NMHumanExternalBloodParcelGPU& parcel=n.externalBlood.parcels[freeParcel];
            parcel.positionAndVolume=float4(spawn,n.externalBlood.pendingParcel.x);
            parcel.velocityAndMass=float4(momentum/mass,mass);
            parcel.gasAndFlags=float4(n.externalBlood.pendingParcel.zw,1.0f,0.0f);
            parcel.intrinsicAngularMomentum=float4(
                n.externalBlood.pendingAngularMomentum.xyz-cross(spawn,momentum),0.0f);
            n.externalBlood.pendingParcel=0.0f;
            n.externalBlood.pendingMomentum=0.0f;
            n.externalBlood.pendingPositionMass=0.0f;
            n.externalBlood.pendingAngularMomentum=0.0f;
            n.externalBlood.energyLedger.x=0.0f;
        }
    }
    // In parallel-gas mode only lane zero publishes the outlet debit, while
    // every lane below reads vascularAfter for observations and the balance
    // ledger. Make that device-buffer write visible at the SIMD rendezvous.
    if(kUseParallelRespiratoryMuscles) threadgroup_barrier(mem_flags::mem_device);
    if(publish) {
        const uint poolIndex=NM_HUMAN_EXTERNAL_BLOOD_PARCEL_CAPACITY-1u;
        thread NMHumanExternalBloodParcelGPU& pool=n.externalBlood.parcels[poolIndex];
        for(uint i=0;i<poolIndex;++i) {
            thread NMHumanExternalBloodParcelGPU& parcel=n.externalBlood.parcels[i];
            if(parcel.gasAndFlags.z<0.5f) continue;
            const float radius=pow(max(parcel.positionAndVolume.w,0.0f)*0.2387324146f,1.0f/3.0f);
            parcel.velocityAndMass.z-=9.81f*dt;
            parcel.positionAndVolume.xyz+=parcel.velocityAndMass.xyz*dt;
            if(parcel.positionAndVolume.z<=outlet.sourcePointAndGround.w+radius) {
                parcel.positionAndVolume.z=outlet.sourcePointAndGround.w+radius;
                const float3 impactImpulse=parcel.velocityAndMass.w*parcel.velocityAndMass.xyz;
                n.externalBlood.groundReactionImpulse.xyz+=impactImpulse;
                n.externalBlood.groundReactionAngularImpulse.xyz+=
                    cross(parcel.positionAndVolume.xyz,impactImpulse)+
                    parcel.intrinsicAngularMomentum.xyz;
                n.externalBlood.groundImpactDissipation.x+=
                    0.5f*parcel.velocityAndMass.w*
                    dot(parcel.velocityAndMass.xyz,parcel.velocityAndMass.xyz);
                const float4 transfer=float4(parcel.positionAndVolume.w,parcel.velocityAndMass.w,
                    parcel.gasAndFlags.x,parcel.gasAndFlags.y);
                const float4 y=transfer-n.externalBlood.groundedCompensation;
                const float4 next=n.externalBlood.grounded+y;
                n.externalBlood.groundedCompensation=(next-n.externalBlood.grounded)-y;
                n.externalBlood.grounded=next;

                // All landed mass remains in one world-fixed puddle pool at
                // the volume-weighted impact centroid. The GPU renderer turns
                // this exact ledger volume into a flattened ellipsoid.
                const float oldPoolVolume=pool.positionAndVolume.w;
                const float newPoolVolume=oldPoolVolume+parcel.positionAndVolume.w;
                if(!(newPoolVolume>0.0f)||!isfinite(newPoolVolume)) {
                    n.status.w=9u;
                } else {
                    if(oldPoolVolume>0.0f) {
                        pool.positionAndVolume.xyz=
                            (oldPoolVolume*pool.positionAndVolume.xyz+
                             parcel.positionAndVolume.w*parcel.positionAndVolume.xyz)/newPoolVolume;
                    } else {
                        pool.positionAndVolume.xyz=parcel.positionAndVolume.xyz;
                    }
                    pool.positionAndVolume.w=newPoolVolume;
                    const float oldPoolMass=pool.velocityAndMass.w;
                    pool.velocityAndMass=float4(0.0f,0.0f,0.0f,oldPoolMass+transfer.y);
                    pool.gasAndFlags.x+=transfer.z;
                    pool.gasAndFlags.y+=transfer.w;
                    pool.gasAndFlags.z=2.0f; // grounded pooled state
                    pool.intrinsicAngularMomentum=0.0f;
                }
                parcel.gasAndFlags.z=0.0f;
            }
        }
    }
    if(publish&&d.bodyPropertyCount>0u) {
        float referencePositiveMass=0.0f,referenceCorrection=0.0f;
        float candidatePositiveMass=0.0f,candidateCorrection=0.0f;
        bool validBodyMass=true;
        for(uint bodyOrdinal=0u;bodyOrdinal<d.bodyPropertyCount;++bodyOrdinal) {
            const float referenceMass=bodyPropertiesReference[bodyOrdinal].massAndInverseMass.x;
            const float candidateMass=bodyPropertiesCandidate[bodyOrdinal].massAndInverseMass.x;
            if(!isfinite(referenceMass)||!isfinite(candidateMass)||
               referenceMass<0.0f||candidateMass<0.0f) {
                validBodyMass=false;
                break;
            }
            if(referenceMass>0.0f) {
                const float adjusted=referenceMass-referenceCorrection;
                const float sum=referencePositiveMass+adjusted;
                referenceCorrection=(sum-referencePositiveMass)-adjusted;
                referencePositiveMass=sum;
            }
            if(candidateMass>0.0f) {
                const float adjusted=candidateMass-candidateCorrection;
                const float sum=candidatePositiveMass+adjusted;
                candidateCorrection=(sum-candidatePositiveMass)-adjusted;
                candidatePositiveMass=sum;
            }
        }
        const float externalMass=n.externalBlood.total.y-n.externalBlood.compensation.y;
        const float residual=referencePositiveMass-externalMass-candidatePositiveMass;
        if(!validBodyMass||!(referencePositiveMass>0.0f)||!isfinite(externalMass)||
           !isfinite(residual)||abs(residual)>1.0e-3f) {
            n.status.w=10u;
        } else {
            n.externalBlood.bodyMassBalance=float4(referencePositiveMass,
                candidatePositiveMass,externalMass,residual);
            n.externalBlood.bodyMassBalanceMax.x=max(
                n.externalBlood.bodyMassBalanceMax.x,abs(residual));
        }
    }
    n.transportStep=float4(netSweptVolume,
        (n.mechanics.x-old.mechanics.x)-netSweptVolume,
        maxDonorFraction,float(gasSubsteps));
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
    const float combinedBloodVolume=bloodVolume+n.externalBlood.total.x;
    n.circulation=float4(bloodVolume,
        human_respiration::physical(vascularAfter,unknowns,base,20),
        human_respiration::physical(vascularAfter,unknowns,base,16),
        max(n.circulation.w,abs(combinedBloodVolume-0.00515f)));
    // Internal edge transfers cancel globally. The accepted cut debit is
    // balanced by the same-volume external inventory increment below, so the
    // observer checks the combined vascular + external candidate.
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
    // The vascular normalized row was debited after the circuit solve. Add the
    // transferred physical volume once to both reductions to measure the full
    // vascular-plus-external accepted candidate, not a closed-network total.
    if(publish && outletVolume>0.0f) {
        stepResidual+=outletVolume;
        stepPhysicalDelta+=outletVolume;
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
    human_respiration::addGas(total,n.externalBlood.total.zw-n.externalBlood.compensation.zw);
    const float2 balance=total.xy-n.environmentGas.xy+
        float2(n.metabolicGas.x,-n.metabolicGas.y)-n.gasBudget.xy;
    n.gasBudget.zw=max(n.gasBudget.zw,abs(balance));
    if(any(abs(balance)>5.e-5f*n.gasBudget.xy)) n.status.w=7;
    n.breath.x=max(n.breath.x,n.mechanics.x);
    n.breath.y=min(n.breath.y,n.mechanics.x);
    const float inspiredAdjusted=(kUseRespiratorySubcycling?inspiredVolume:
        dt*max(0.0f,n.mechanics.w))-n.breathAccounting.y;
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
       maxAirwayTransport>0.1f*p.lung.y*p.environment.z) n.status.w=6;
    if(publish) {
        if(n.status.w) statuses[env].code=NM_STATUS_MULTIPHYSICS_FAILURE;
        candidate[env]=n;
    }
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

// Presentation layout identity, never dispatched.
kernel void nm_human_resting_local_cardiac_layout_v5(uint i [[thread_position_in_grid]]) { (void)i; }

// Accepted parcel words are bound at the exact C++ offsetof(externalBlood.parcels).
// Shared host static_asserts pin each 64-byte record and its four float4 offsets.
// Octahedra are reduced packet/pool representations, not a free-surface solver.
kernel void nm_human_resting_blood_visual(
 constant uint4& ranges [[buffer(0)]],
 device const float4* acceptedParcelWords [[buffer(1)]],
 device MRVisualVertexGPUV2* vertices [[buffer(2)]],
 device MRVisualPrimitiveGPUV2* primitives [[buffer(3)]],
 device MRVisualInstanceGPUV2* instances [[buffer(4)]],
 constant uint& visibilityOnly [[buffer(5)]],
 uint p [[thread_position_in_grid]]) {
 if(p>=ranges.w)return;
 const float4 positionVolume=acceptedParcelWords[4u*p];
 const float4 velocityMass=acceptedParcelWords[4u*p+1u];
 const float4 gasFlags=acceptedParcelWords[4u*p+2u];
 // Sub-nanolitre inventory remains in physics; hide it below mesh resolution.
 const bool active=gasFlags.z>0.5f&&positionVolume.w>=1.0e-12f&&velocityMass.w>0.0f&&
                   all(isfinite(positionVolume))&&all(isfinite(velocityMass))&&all(isfinite(gasFlags));
 if(visibilityOnly!=0u) {
  // Preserve the existing selected anatomy-layer visibility for active packets.
  if(!active)instances[ranges.z+p].binding.w=0u;
  return;
 }
 float3 center=active?positionVolume.xyz:float3(0);
 float3 scale=float3(0.001f);
 if(active) {
  // Volume of the actual rendered unit octahedron is 4/3.
  const float radius=pow(0.75f*positionVolume.w,1.0f/3.0f);
  scale=float3(radius);
  if(gasFlags.z>1.5f) {
   const float halfHeight=min(0.002f,radius);
   const float horizontal=sqrt(0.75f*positionVolume.w/halfHeight);
   scale=float3(horizontal,horizontal,halfHeight);
   center.z+=halfHeight;
  }
 }
 const float3 axes[6]={float3(1,0,0),float3(-1,0,0),float3(0,1,0),float3(0,-1,0),float3(0,0,1),float3(0,0,-1)};
 for(uint k=0;k<6u;++k) {
  const uint v=ranges.x+6u*p+k;const float3 normal=normalize(axes[k]/scale);
  vertices[v].position=float4(center+axes[k]*scale,1);
  vertices[v].normalAndTangentSign=float4(normal,1);
  const float3 tangentReference=abs(normal.y)<0.9f?float3(0,1,0):float3(1,0,0);
  const float3 tangent=normalize(cross(tangentReference,normal));
  vertices[v].tangent=float4(tangent,0);
 }
 primitives[ranges.y+p].boundsMinimum=float4(center-scale,1);
 primitives[ranges.y+p].boundsMaximum=float4(center+scale,1);
}
