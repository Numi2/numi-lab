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
