#include <metal_stdlib>
#include "metalrobo/unified_quality_shared.h"
using namespace metal;
// Research instrument: a float expansion, not a new canonical world state.
// The native compiler must use -fno-fast-math. Domain excludes subnormal/overflow.
struct DS { float h; float l; };
inline DS ds(float x) { return {x,0.0f}; }
inline DS renorm(float h,float l) { float s=h+l; return {s,l-(s-h)}; }
inline DS add(DS a,DS b) {
    float s=a.h+b.h, v=s-a.h;
    float e=(a.h-(s-v))+(b.h-v);
    return renorm(s,e+(a.l+b.l));
}
inline DS neg(DS a) {return {-a.h,-a.l};}
inline DS sub(DS a,DS b) {return add(a,neg(b));}
inline DS mul(DS a,DS b) {
    float p=a.h*b.h;
    float e=fma(a.h,b.h,-p);
    e += a.h*b.l+a.l*b.h;
    e += a.l*b.l;
    return renorm(p,e);
}
inline DS divds(DS a,DS b) {
    float q=a.h/b.h;
    DS r=sub(a,mul(b,ds(q)));
    float q2=(r.h+r.l)/b.h;
    DS x=add(ds(q),ds(q2));
    r=sub(a,mul(b,x));
    return add(x,ds((r.h+r.l)/b.h));
}
inline DS sqrtds(DS a) {
    if(a.h==0.0f && a.l==0.0f)return ds(0.0f);
    float q=sqrt(a.h);
    return add(ds(q),divds(sub(a,mul(ds(q),ds(q))),ds(2.0f*q)));
}
inline bool lt(DS a,DS b) {return a.h<b.h || (a.h==b.h && a.l<b.l);}
inline bool le(DS a,DS b) {return !lt(b,a);}
inline float mag(DS a) {return abs(a.h+a.l);}
inline float bf(float4 a,float4 b,uint i) {return i<4?a[i]:b[i-4];}
struct Params {uint magic,version,nv,rows,blocks,maxiters,rounding,reserved;};
kernel void calibrate(device DS* o [[buffer(0)]],uint i [[thread_position_in_grid]]) {
    if(i!=0)return;
    DS a=add(ds(1.0f),ds(0x1p-30f));
    o[0]=a; o[1]=sub(a,ds(1.0f));
    o[2]=mul(add(ds(1.0f),ds(0x1p-24f)),sub(ds(1.0f),ds(0x1p-24f)));
    o[3]=divds(ds(1.0f),ds(10.0f)); o[4]=sqrtds(ds(2.0f));
    o[5]=add(add(ds(1.0e8f),ds(1.0f)),ds(-1.0e8f));
    o[6]=divds(ds(-7.0f),ds(3.0f));
    o[7]=sub(ds(1.0f),add(ds(1.0f),ds(0x1p-30f)));
}
inline bool evaluate(
    constant Params& c,device const MRUnifiedQualityBlockGPU* blocks,
    device const float* a,device const float* j,device const float* bias,
    device const float* freev,device DS* v,device DS* w,device DS* p,
    device DS* g,device DS* weights,device uint* status,device float* metrics,uint lane) {
    for(uint r=lane;r<c.rows;r+=32) {
        DS x=ds(bias[r]);
        for(uint k=0;k<c.nv;++k)x=add(x,mul(ds(j[r*c.nv+k]),v[k]));
        w[r]=x;p[r]=ds(0);weights[r]=ds(0);
    }
    threadgroup_barrier(mem_flags::mem_device);
    uint interior=0,vertexCount=0,scalarfree=0,scalarbound=0,unsupported=0xffffffffu;
    for(uint bi=lane;bi<c.blocks;bi+=32) {
        auto b=blocks[bi];uint at=b.layout.x,n=b.layout.y;
        if(b.layout.z==MR_UNIFIED_QUALITY_SCALAR_INTERVAL) {
            DS x=neg(divds(w[at],ds(b.regularization0.x)));
            DS lo=ds(b.boundsAndShift.x),hi=ds(b.boundsAndShift.y);
            if(lt(lo,x)&&lt(x,hi)) {p[at]=x;weights[at]=divds(ds(1),ds(b.regularization0.x));++scalarfree;}
            else {p[at]=lt(x,lo)?lo:hi;++scalarbound;}
        } else if(b.layout.z==MR_UNIFIED_QUALITY_ELLIPTIC_CONE &&
                  b.boundsAndShift.x==0 && b.boundsAndShift.y==0 && n>=3 && n<=6) {
            DS z[6],d[6];DS ts=ds(0),ws=ds(0);
            for(uint k=0;k<n;++k) {
                DS r=ds(bf(b.regularization0,b.regularization1,k));
                DS s=ds(bf(b.scale0,b.scale1,k));
                d[k]=mul(r,mul(s,s));z[k]=neg(divds(w[at+k],mul(r,s)));
                if(k){ts=add(ts,mul(z[k],z[k]));DS u=mul(d[k],z[k]);ws=add(ws,mul(u,u));}
            }
            if(le(sqrtds(ts),z[0])) {
                for(uint k=0;k<n;++k) {
                    DS r=ds(bf(b.regularization0,b.regularization1,k));
                    p[at+k]=neg(divds(w[at+k],r));weights[at+k]=divds(ds(1),r);
                }++interior;
            } else if(le(mul(d[0],z[0]),neg(sqrtds(ws)))) {++vertexCount;}
            else unsupported=min(unsupported,bi);
        } else unsupported=min(unsupported,bi);
    }
    unsupported=simd_min(unsupported);
    uint ni=simd_sum(interior),nz=simd_sum(vertexCount),nf=simd_sum(scalarfree),nb=simd_sum(scalarbound);
    if(lane==0){status[2]=ni;status[3]=nz;status[4]=nf;status[5]=nb;status[6]=unsupported;}
    threadgroup_barrier(mem_flags::mem_device);
    if(unsupported!=0xffffffffu)return false;
    float gm=0,am=0,jm=0;
    for(uint r=lane;r<c.nv;r+=32) {
        DS av=ds(0),jp=ds(0);
        for(uint k=0;k<c.nv;++k)av=add(av,mul(ds(a[r*c.nv+k]),sub(v[k],ds(freev[k]))));
        for(uint k=0;k<c.rows;++k)jp=add(jp,mul(ds(j[k*c.nv+r]),p[k]));
        g[r]=sub(av,jp);gm=max(gm,mag(g[r]));am=max(am,mag(av));jm=max(jm,mag(jp));
    }
    gm=simd_max(gm);am=simd_max(am);jm=simd_max(jm);
    if(lane==0){metrics[0]=gm/(1+am+jm);metrics[1]=gm;metrics[2]=am;metrics[3]=jm;}
    threadgroup_barrier(mem_flags::mem_device);return true;
}
kernel void solve(
    constant Params& c [[buffer(0)]],device const MRUnifiedQualityBlockGPU* blocks [[buffer(1)]],
    device const float* a [[buffer(2)]],device const float* j [[buffer(3)]],
    device const float* bias [[buffer(4)]],device const float* freev [[buffer(5)]],
    device const float* start [[buffer(6)]],device DS* scratch [[buffer(7)]],
    device uint* status [[buffer(8)]],uint lane [[thread_index_in_threadgroup]]) {
    device DS* v=scratch;device DS* w=v+c.nv;device DS* p=w+c.rows;
    device DS* g=p+c.rows;device DS* weights=g+c.nv;
    device DS* h=weights+c.rows;device DS* direction=h+c.nv*c.nv;
    device float* metrics=reinterpret_cast<device float*>(status+8);
    if(lane==0){for(uint k=0;k<16;++k)status[k]=0;status[6]=0xffffffffu;}
    for(uint k=lane;k<c.nv;k+=32)v[k]=ds(start[k]);
    threadgroup_barrier(mem_flags::mem_device);
    uint code=7,iterations=0;
    for(uint it=0;it<=c.maxiters;++it) {
        if(!evaluate(c,blocks,a,j,bias,freev,v,w,p,g,weights,status,metrics,lane)){code=3;break;}
        if(metrics[0]<=1.0e-10f){code=0;break;}
        if(it==c.maxiters)break;
        for(uint index=lane;index<c.nv*c.nv;index+=32) {
            uint r=index/c.nv,k=index%c.nv;DS x=ds(a[index]);
            for(uint row=0;row<c.rows;++row)x=add(x,mul(mul(ds(j[row*c.nv+r]),weights[row]),ds(j[row*c.nv+k])));
            h[index]=x;
        }
        threadgroup_barrier(mem_flags::mem_device);
        bool valid=true;
        for(uint k=0;k<c.nv;++k) {
            if(lane==0) {
                DS x=h[k*c.nv+k];
                for(uint s=0;s<k;++s)x=sub(x,mul(mul(h[k*c.nv+s],h[k*c.nv+s]),h[s*c.nv+s]));
                h[k*c.nv+k]=x;
            }
            threadgroup_barrier(mem_flags::mem_device);
            if(!(h[k*c.nv+k].h>0) || !isfinite(h[k*c.nv+k].h)){valid=false;break;}
            for(uint r=k+1+lane;r<c.nv;r+=32) {
                DS x=h[r*c.nv+k];
                for(uint s=0;s<k;++s)x=sub(x,mul(mul(h[r*c.nv+s],h[k*c.nv+s]),h[s*c.nv+s]));
                h[r*c.nv+k]=divds(x,h[k*c.nv+k]);
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
        if(!valid){code=4;break;}
        // Instrument-only triangular solve. No production dispatch uses this kernel.
        if(lane==0) {
            for(uint r=0;r<c.nv;++r) {
                DS x=neg(g[r]);
                for(uint k=0;k<r;++k)x=sub(x,mul(h[r*c.nv+k],direction[k]));
                direction[r]=x;
            }
            for(uint r=0;r<c.nv;++r)direction[r]=divds(direction[r],h[r*c.nv+r]);
            for(int r=int(c.nv)-1;r>=0;--r) {
                DS x=direction[r];
                for(uint k=r+1;k<c.nv;++k)x=sub(x,mul(h[k*c.nv+r],direction[k]));
                direction[r]=x;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        for(uint r=lane;r<c.nv;r+=32) {
            v[r]=add(v[r],direction[r]);
            if(c.rounding)v[r]=ds(v[r].h+v[r].l);
        }
        threadgroup_barrier(mem_flags::mem_device);iterations=it+1;
    }
    if(lane==0){status[0]=code;status[1]=iterations;}
}
