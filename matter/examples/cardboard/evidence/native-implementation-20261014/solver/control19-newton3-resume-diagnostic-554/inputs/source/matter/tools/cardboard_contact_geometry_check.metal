#include "../src/matter_kernels.metal"
namespace numi_matter_metal {
kernel void nm_contact_scaled_geometry_replay(device float4* output [[buffer(0)]], uint global [[thread_position_in_grid]]) {
    if (global>=96u) return;
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
        const uint base=global*16u;
        for(uint j=0u;j<16u;++j)output[base+j]=float4(0.0f);
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
    const uint base=global*16u;
    for(uint j=0;j<16u;++j)output[base+j]=float4(0.0f);
    output[base]=float4(float(kind),s,shift,float(global));
    output[base+1u]=float4(point,1);output[base+2u]=float4(a,1);
    output[base+3u]=float4(b,1);output[base+4u]=float4(c,1);output[base+5u]=float4(q1,1);
    output[base+7u]=float4(newBary,1);
    output[base+9u]=float4(newP,1);
    output[base+11u]=float4(newQ,1);
}
}
