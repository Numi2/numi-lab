#include "../src/matter_kernels.metal"
namespace numi_matter_metal {
kernel void nm_contact_scaled_geometry_replay(device float4* output [[buffer(0)]], uint global [[thread_position_in_grid]]) {
    if (global>=112u) return;
    if (global>=109u) {
        const uint test=global-109u;
        const float3 a=float3(0.0f), b=float3(0.02f,0.0f,0.0f), c=float3(0.0f,0.02f,0.0f);
        const float3 point=test==2u?float3(0.004f,0.006f,1.0e-13f):float3(0.004f,0.006f,0.001f);
        const float3 featurePosition[4]={point,a,b,c};
        NMMortarContactData mapping{};
        mapping.found=1u; mapping.kind=NM_DEFORMABLE_CONTACT_VERTEX_TRIANGLE;
        mapping.positiveSide=test==0u?0u:1u;
        mapping.normal=test==2u?float3(0.6f,0.0f,0.8f):float3(0.0f,0.0f,1.0f);
        mapping.featureBarycentric[0]=float3(1.0f,0.0f,0.0f);
        mapping.featureBarycentric[1]=float3(1.0f,0.0f,0.0f);
        mapping.featureBarycentric[2]=float3(0.0f,1.0f,0.0f);
        mapping.featureBarycentric[3]=float3(0.0f,0.0f,1.0f);
        const float3 pointParent[3]={point,point+float3(0.02f,0.0f,0.0f),point+float3(0.0f,0.02f,0.0f)};
        const float3 triangleParent[3]={a,b,c};
        float3 parent0[3],parent1[3];
        for(uint i=0u;i<3u;++i) {
            parent0[i]=mapping.positiveSide==0u?pointParent[i]:triangleParent[i];
            parent1[i]=mapping.positiveSide==0u?triangleParent[i]:pointParent[i];
        }
        float3 normal,closest; float separation;
        float weights[6]={0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
        const bool valid=contactMortarVertexTriangleWitness(mapping,featurePosition,normal,closest,separation,weights);
        float3 weightedPosition=float3(0.0f);
        for(uint i=0u;i<3u;++i) weightedPosition+=weights[i]*parent0[i]+weights[3u+i]*parent1[i];
        const float3 torque=cross(weightedPosition,normal);
        const uint base=global*18u;
        for(uint j=0u;j<18u;++j)output[base+j]=float4(0.0f);
        output[base]=float4(float(test),separation,valid?1.0f:0.0f,float(mapping.positiveSide));
        for(uint i=0u;i<4u;++i)output[base+1u+i]=float4(featurePosition[i],0.0f);
        for(uint i=0u;i<3u;++i) { output[base+5u+i]=float4(parent0[i],0.0f); output[base+8u+i]=float4(parent1[i],0.0f); }
        output[base+11u]=float4(weights[0],weights[1],weights[2],weights[3]);
        output[base+12u]=float4(weights[4],weights[5],0.0f,0.0f);
        output[base+13u]=float4(weightedPosition,0.0f);
        output[base+14u]=float4(normal,length(normal)); output[base+15u]=float4(closest,0.0f);
        output[base+16u]=float4(mapping.normal,separation); output[base+17u]=float4(torque,0.0f);
        return;
    }
    if (global>=102u) {
        const uint test=global-102u;
        float3 point,a,b,c; float thickness=0.005f;
        if(test==0u) {
            point=float3(0.0491591915f,-0.585491121f,0.114537001f);
            a=float3(0.0487377346f,-0.585262775f,0.114244871f);
            b=float3(0.0444787107f,-0.582939923f,0.111115232f);
            c=float3(0.0516800359f,-0.586848319f,0.116216816f); thickness=1.0e-5f;
        } else if(test==1u) {
            point=float3(0.0491592772f,-0.58549118f,0.114537053f);
            a=float3(0.0487376861f,-0.585262775f,0.114244834f);
            b=float3(0.0444787107f,-0.582939923f,0.111115232f);
            c=float3(0.0516800322f,-0.586848319f,0.116216816f); thickness=1.0e-5f;
        } else if(test==2u) {
            point=float3(0.0f,0.0f,0.0002f); a=float3(0.0f); b=float3(0.02f,0.0f,0.0f); c=float3(0.0f,0.02f,0.0f);
        } else if(test==3u) {
            point=float3(0.01f,-0.0002f,0.0001f); a=float3(0.0f); b=float3(0.02f,0.0f,0.0f); c=float3(0.0f,0.02f,0.0f);
        } else if(test==4u) {
            point=float3(0.004f,0.006f,0.0001f); a=float3(0.0f); b=float3(0.02f,0.0f,0.0f); c=float3(0.0f,0.02f,0.0f);
        } else if(test==5u) {
            point=float3(-0.0001f,0.0f,0.0003f); a=float3(0.0f); b=float3(0.02f,0.0f,0.0f); c=float3(0.04f,0.0f,0.0f);
        } else {
            const float3 o=float3(8192.0f,-4096.0f,2048.0f);
            point=o+float3(0.0703125f,0.0595703125f,0.001953125f);
            a=o; b=o+float3(0.3f,0.04f,0.05f); c=o+float3(0.04f,0.3f,-0.03f); thickness=0.01f;
        }
        const float3 positions[4]={point,a,b,c};
        float3 bary,closest; MRCompensatedPositionGPU relative; float separation;
        const bool valid=contactCompensatedVertexTriangleWitness(point,a,b,c,bary,closest,relative,separation);
        float3 legacyBary; const float3 legacyClosest=closestPointOnTriangle(point,a,b,c,legacyBary);
        const float3 legacyRelative=point-legacyClosest; const float legacySeparation=length(legacyRelative);
        float3 firstGradient,secondGradient;
        const float mollifier=ipcFeatureMollifierDifferential(b-a,c-a,firstGradient,secondGradient);
        NMDeformableContactGPU contact{}; contact.identity.z=NM_DEFORMABLE_CONTACT_VERTEX_TRIANGLE;
        contact.barrier.z=thickness; contact.mollifier.w=1.81084e13f;
        float energy,energySeparation;
        const bool energyValid=deformableBarrierEnergyAt(contact,positions,energy,energySeparation);
        const float3 hi=relative.high.xyz,lo=relative.low.xyz,delta=hi+lo;
        const float3 normal=separation>0.0f?delta/separation:float3(0.0f);
        const uint base=global*18u;
        for(uint j=0u;j<18u;++j)output[base+j]=float4(0.0f);
        output[base]=float4(float(test),separation,valid?1.0f:0.0f,thickness);
        output[base+1u]=float4(point,0.0f); output[base+2u]=float4(a,0.0f);
        output[base+3u]=float4(b,0.0f); output[base+4u]=float4(c,0.0f);
        output[base+5u]=float4(bary,0.0f); output[base+6u]=float4(closest,0.0f);
        output[base+7u]=float4(hi,separation); output[base+8u]=float4(lo,0.0f);
        output[base+9u]=float4(normal,length(normal));
        output[base+10u]=float4(thickness,contact.mollifier.w,mollifier,energyValid?1.0f:0.0f);
        output[base+11u]=float4(legacyBary,legacySeparation); output[base+12u]=float4(legacyClosest,0.0f);
        output[base+13u]=float4(legacySeparation>0.0f?legacyRelative/legacySeparation:float3(0.0f),legacySeparation);
        output[base+14u]=float4(energy,energySeparation,energyValid?1.0f:0.0f,mollifier);
        return;
    }
    if (global>=96u) {
        const uint row=global;
        float3 p0,p1,q0,q1;
        if (row==96u) {
            p0=float3(-0.0164212119,-0.519634128,0.0346223861);
            p1=float3(-0.00803090353,-0.526707113,0.0277656447);
            q0=float3(-0.00755289057,-0.54869628,0.0357304141);
            q1=float3(-0.0307032298,-0.461499035,0.0284029972);
        } else if (row==97u) {
            p0=float3(-0.0164212119,-0.519634128,0.0346223861);
            p1=float3(-0.00803074706,-0.526707053,0.0277655888);
            q0=float3(-0.00755289756,-0.54869628,0.0357304364);
            q1=float3(-0.0307032261,-0.461499035,0.0284030009);
        } else if (row==98u) {
            p0=float3(-0.0164212119,-0.519634128,0.0346223861);
            p1=float3(-0.00803090259,-0.526707113,0.0277656447);
            q0=float3(-0.00755289104,-0.54869628,0.0357304141);
            q1=float3(-0.0307032298,-0.461499035,0.0284029972);
        } else if (row==99u) {
            p0=float3(0.001,0.002,0.003); p1=p0;
            q0=float3(0.001,0.002,0.004);
            q1=float3(0.001,0.002,0.005);
        } else if (row==100u) {
            p0=float3(0.0,0.0,0.0); p1=float3(1.0,0.0,0.0);
            q0=float3(1.0,0.2,0.0001); q1=float3(1.0,1.0,0.0001);
        } else {
            p0=float3(-0.5,0.0,0.0); p1=float3(0.5,0.0,0.0);
            q0=float3(-0.5,0.0005,0.0001);
            q1=float3(0.5,0.000501,0.0001);
        }
        float s,t; float3 oldP,oldQ;
        MRCompensatedPositionGPU relative; float separation;
        const bool witnessValid=contactCompensatedSegmentWitness(
            p0,p1,q0,q1,s,t,oldP,oldQ,relative,separation);
        const float3 positions[4]={p0,p1,q0,q1};
        NMDeformableContactGPU contact{};
        contact.identity.z=NM_DEFORMABLE_CONTACT_EDGE_EDGE;
        contact.barrier.z=1.0e-5f;
        contact.mollifier.w=1.0f;
        float energy,energySeparation;
        const bool energyValid=deformableBarrierEnergyAt(
            contact,positions,energy,energySeparation);
        float3 mollifierGradient0,mollifierGradient1;
        const float mollifier=ipcFeatureMollifierDifferential(
            p1-p0,q1-q0,mollifierGradient0,mollifierGradient1);
        const float3 relativeHigh=relative.high.xyz;
        const float3 relativeLow=relative.low.xyz;
        const float3 roundedRelative=relativeHigh+relativeLow;
        const float3 normal=separation>0.0f
            ? roundedRelative/separation : float3(0.0f);
        const uint base=global*18u;
        for(uint j=0u;j<18u;++j)output[base+j]=float4(0.0f);
        output[base]=float4(float(row-96u),separation,float(s),float(t));
        output[base+1u]=float4(p0,0.0f); output[base+2u]=float4(p1,0.0f);
        output[base+3u]=float4(q0,0.0f); output[base+4u]=float4(q1,0.0f);
        output[base+7u]=float4(s,t,witnessValid?1.0f:0.0f,0.0f);
        output[base+9u]=float4(oldP,0.0f); output[base+11u]=float4(oldQ,0.0f);
        output[base+13u]=float4(relativeHigh,separation);
        output[base+14u]=float4(relativeLow,0.0f);
        output[base+15u]=float4(length(oldP-oldQ),0.0f,0.0f,0.0f);
        output[base+16u]=float4(energy,energySeparation,
            energyValid?1.0f:0.0f,mollifier);
        output[base+17u]=float4(normal, length(normal));
        return;
    }
    if (global>=48u) {
        const uint row=global-48u;
        const uint angleIndex=row%6u;
        const float scales[4]={0.0001f,0.0005f,0.002f,0.02f};
        const float angles[6]={0.0f,0.003f,0.01f,0.02f,0.04f,1.5707963267948966f};
        const float scale=scales[(row/6u)%4u];
        const float lengthRatio=row<24u?1.0f:7.0f;
        const float theta=angles[angleIndex];
        const float3 first(scale,0.0f,0.0f);
        const float3 second=angleIndex==5u?float3(0.0f,scale*lengthRatio,0.0f):
            scale*lengthRatio*float3(cos(theta),sin(theta),0.0f);
        float3 g1,g2;
        const float value=ipcFeatureMollifierDifferential(first,second,g1,g2);
        const uint base=global*18u;
        for(uint j=0u;j<18u;++j)output[base+j]=float4(0.0f);
        output[base]=float4(float(angleIndex),scale,lengthRatio,float(global));
        output[base+1u]=float4(first,0.0f);output[base+2u]=float4(second,0.0f);
        output[base+3u]=float4(g1,0.0f);output[base+4u]=float4(g2,0.0f);
        output[base+5u]=float4(value,theta,0.0f,0.0f);
        return;
    }
    const uint kind=global%6u;
    const uint scaleIndex=(global/6u)%4u;
    const float scales[4]={0.0001f,0.0005f,0.002f,0.02f};
    const float s=scales[scaleIndex];
    const float shift=global<24u?0.0f:0.03f;
    const float3 origin(shift,-0.5f*shift,0.25f*shift);
    float3 point=origin,a=origin,b=origin,c=origin,q1=origin;
    float3 newBary(0.0f),newP(0.0f),newQ(0.0f);
    if (kind<3u) {
        a=origin;b=origin+float3(s,0,0);c=origin+float3(0,s,0);
        if(kind==0u)point=origin+float3(0.25f*s,0.25f*s,0.1f*s);
        if(kind==1u)point=origin+float3(0.5f*s,-0.2f*s,0.1f*s);
        if(kind==2u)point=origin+float3(-0.2f*s,1.2f*s,0.1f*s);

        newP=closestPointOnTriangle(point,a,b,c,newBary);
    } else {
        if(kind==3u){a=origin+float3(-0.5f*s,0,0);b=origin+float3(0.5f*s,0,0);c=origin+float3(0,-0.5f*s,0.1f*s);q1=origin+float3(0,0.5f*s,0.1f*s);}
        if(kind==4u){a=origin;b=origin+float3(s,0,0);c=origin+float3(0.25f*s,0.1f*s,0);q1=origin+float3(0.75f*s,0.1f*s,0);}
        if(kind==5u){a=b=origin;c=origin+float3(-0.5f*s,0.2f*s,0);q1=origin+float3(0.5f*s,0.2f*s,0);}
        float newS,newT;

        closestPointsOnSegments(a,b,c,q1,newS,newT,newP,newQ);
        newBary=float3(newS,newT,0);
    }
    const uint base=global*18u;
    for(uint j=0;j<18u;++j)output[base+j]=float4(0.0f);
    output[base]=float4(float(kind),s,shift,float(global));
    output[base+1u]=float4(point,1);output[base+2u]=float4(a,1);
    output[base+3u]=float4(b,1);output[base+4u]=float4(c,1);output[base+5u]=float4(q1,1);
    output[base+7u]=float4(newBary,1);
    output[base+9u]=float4(newP,1);
    output[base+11u]=float4(newQ,1);
}
}
