#include <metal_stdlib>
using namespace metal;
constant uint NMAX=17;
struct RodInput { float4 p[17]; float4 target[17]; float4 parameters; uint4 counts; };
struct RodOutput { float4 position[17]; float4 reaction[17]; float4 info; };

inline float3x3 outer3(float3 a,float3 b) { return float3x3(a*b.x,a*b.y,a*b.z); }
inline bool invertPositive(float3x3 m, thread float3x3& inverse) {
    float p0=m[0][0];
    float p1=m[0][0]*m[1][1]-m[1][0]*m[0][1];
    float3 r0=cross(m[1],m[2]),r1=cross(m[2],m[0]),r2=cross(m[0],m[1]);
    float det=dot(m[0],r0);
    if (!(p0>0.0f) || !(p1>0.0f) || !(det>0.0f) || !isfinite(det)) return false;
    inverse=transpose(float3x3(r0,r1,r2))/det;
    return all(isfinite(inverse[0])) && all(isfinite(inverse[1])) && all(isfinite(inverse[2]));
}
inline bool evaluate(device const RodInput& in, thread float3* d,
                     thread float3* g, thread float3x3* diag, thread float3x3* off) {
    uint n=in.counts.x;float rest=in.parameters.x,k=in.parameters.y,h=in.parameters.z;
    float3x3 I=float3x3(1.0f);
    for(uint i=0;i<n;i++) {
        float inertia=in.p[i].w/(h*h);
        g[i]=inertia*d[i]; diag[i]=I*inertia;
    }
    for(uint i=0;i+1<n;i++) {
        float3 base=in.p[i+1].xyz-in.p[i].xyz;
        float3 change=d[i+1]-d[i];float3 edge=base+change;
        float a=length(base), l=length(edge);
        if (!(a>0.0f) || !(l>0.0f) || !isfinite(l)) return false;
        float3 u=edge/l;
        float stretch=(a-rest)+(2.0f*dot(base,change)+dot(change,change))/(l+a);
        float3 f=k*stretch*u;g[i]-=f;g[i+1]+=f;
        float3x3 uu=outer3(u,u);
        float3x3 tangent=k*(uu+(stretch/l)*(I-uu));
        diag[i]+=tangent;diag[i+1]+=tangent;off[i]=tangent*(-1.0f);
    }
    for(uint i=0;i<n;i++) if(in.target[i].w>0.0f) {
        float3 rel=(in.p[i].xyz-in.target[i].xyz)+d[i];
        g[i]+=in.target[i].w*rel;diag[i]+=in.target[i].w*I;
    }
    for(uint i=0;i<n;i++) if (!all(isfinite(g[i]))) return false;
    return true;
}
inline float energyChange(device const RodInput& in, thread float3* d, thread float3* trial) {
    uint n=in.counts.x;float rest=in.parameters.x,k=in.parameters.y,h=in.parameters.z;
    float result=0.0f;
    for(uint i=0;i<n;i++) {
        float3 dd=trial[i]-d[i];float inertia=in.p[i].w/(h*h);
        result+=.5f*inertia*(2.0f*dot(d[i],dd)+dot(dd,dd));
    }
    for(uint i=0;i+1<n;i++) {
        float3 base=in.p[i+1].xyz-in.p[i].xyz;
        float3 change=d[i+1]-d[i],edge=base+change;
        float3 dd=(trial[i+1]-d[i+1])-(trial[i]-d[i]);
        float a=length(base),l=length(edge),lnext=length(edge+dd);
        if (!(a>0.0f) || !(l>0.0f) || !(lnext>0.0f)) return NAN;
        float stretch=(a-rest)+(2.0f*dot(base,change)+dot(change,change))/(l+a);
        float dl=(2.0f*dot(edge,dd)+dot(dd,dd))/(lnext+l);
        result+=.5f*k*dl*(2.0f*stretch+dl);
    }
    for(uint i=0;i<n;i++) if(in.target[i].w>0.0f) {
        float3 rel=(in.p[i].xyz-in.target[i].xyz)+d[i];float3 dd=trial[i]-d[i];
        result+=.5f*in.target[i].w*(2.0f*dot(rel,dd)+dot(dd,dd));
    }
    return result;
}
kernel void rod_block(device const RodInput* inputs [[buffer(0)]],
                      device RodOutput* outputs [[buffer(1)]], uint index [[thread_position_in_grid]]) {
    device const RodInput& in=inputs[index];device RodOutput& out=outputs[index];
    uint n=in.counts.x;float3 d[17],trial[17],g[17],rhs[17],step[17];
    float3x3 diag[17],off[16],inv[17];
    uint status=1, iterations=0;float residual=INFINITY;
    if(n<2 || n>NMAX || !(in.parameters.x>0) || !(in.parameters.y>0) || !(in.parameters.z>0)) {
        out.info=float4(0,INFINITY,4,0);return;
    }
    if(!all(isfinite(in.parameters)) || !(in.parameters.w>0.0f)) {
        out.info=float4(0,INFINITY,4,0);return;
    }
    for(uint i=0;i<n;i++) {
        if(!all(isfinite(in.p[i])) || !(in.p[i].w>0.0f) || !all(isfinite(in.target[i]))) {
            out.info=float4(0,INFINITY,4,0);return;
        }
        d[i]=in.target[i].w<0.0f?in.target[i].xyz-in.p[i].xyz:float3(0);
    }
    for(uint it=0;it<100;it++) {
        iterations=it+1;
        if(!evaluate(in,d,g,diag,off)){status=4;break;}
        residual=0;
        for(uint i=0;i<n;i++)if(in.target[i].w>=0.0f)residual=max(residual,length(g[i]));
        if(residual<=in.parameters.w){status=0;break;}
        float k=in.parameters.y;
        for(uint i=0;i<n;i++){diag[i]/=k;rhs[i]=-g[i]/k;}
        for(uint i=0;i+1<n;i++)off[i]/=k;
        for(uint i=0;i<n;i++)if(in.target[i].w<0.0f){
            diag[i]=float3x3(1.0f);rhs[i]=float3(0);
            if(i>0)off[i-1]=float3x3(0);if(i+1<n)off[i]=float3x3(0);
        }
        bool valid=true;
        for(uint i=1;i<n;i++) {
            if(!invertPositive(diag[i-1],inv[i-1])){valid=false;break;}
            float3x3 factor=transpose(off[i-1])*inv[i-1];
            diag[i]-=factor*off[i-1];rhs[i]-=factor*rhs[i-1];
        }
        if(!valid || !invertPositive(diag[n-1],inv[n-1])){status=2;break;}
        step[n-1]=inv[n-1]*rhs[n-1];
        for(int i=int(n)-2;i>=0;i--)step[i]=inv[i]*(rhs[i]-off[i]*step[i+1]);
        bool accepted=false;
        for(uint j=0;j<24;j++) {
            float scale=exp2(-float(j));bool changed=false;
            for(uint i=0;i<n;i++) {
                trial[i]=in.target[i].w<0.0f?in.target[i].xyz-in.p[i].xyz:d[i]+scale*step[i];
                changed=changed || any(trial[i]!=d[i]);
            }
            float de=energyChange(in,d,trial);
            if(isfinite(de) && de<=0.0f && changed){accepted=true;break;}
        }
        if(!accepted){status=3;break;}
        for(uint i=0;i<n;i++)d[i]=trial[i];
    }
    if(status==0 && !evaluate(in,d,g,diag,off))status=4;
    if(status==0) for(uint i=0;i<n;i++) {
        float3 q=in.p[i].xyz+d[i];float3 reaction=float3(0);
        if(in.target[i].w<0.0f){q=in.target[i].xyz;reaction=-g[i];}
        else if(in.target[i].w>0.0f)reaction=in.target[i].w*((in.p[i].xyz-in.target[i].xyz)+d[i]);
        out.position[i]=float4(q,0);out.reaction[i]=float4(reaction,0);
    }
    out.info=float4(float(iterations),residual,float(status),0);
}
