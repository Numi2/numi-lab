#include <metal_stdlib>
using namespace metal;
struct Entry {float4 ah,al,bh,bl,params;};
inline float2 sum2(float a,float b){float s=a+b,z=s-a;return float2(s,(a-(s-z))+(b-z));}
inline float2 normalize2(float a,float b){float h=a+b;return float2(h,b-(h-a));}
inline float2 add2(float2 a,float2 b){float2 s=sum2(a.x,b.x);return normalize2(s.x,s.y+a.y+b.y);}
inline float2 difference(float a,float al,float b,float bl){float s=a-b,v=a-s,e=(a-(s+v))+(v-b);return normalize2(s,e+(al-bl));}
inline float2 square2(float2 a){float p=a.x*a.x;float e=fma(a.x,a.x,-p)+2.0f*a.x*a.y+a.y*a.y;return normalize2(p,e);}
kernel void stretch(device const Entry* input [[buffer(0)]],device float4* output [[buffer(1)]],uint i [[thread_position_in_grid]]){
 Entry x=input[i];float2 dx=difference(x.bh.x,x.bl.x,x.ah.x,x.al.x),dy=difference(x.bh.y,x.bl.y,x.ah.y,x.al.y),dz=difference(x.bh.z,x.bl.z,x.ah.z,x.al.z);
 float2 squared=add2(add2(square2(dx),square2(dy)),square2(dz));float lengthHi=sqrt(squared.x);float lengthLo=(fma(-lengthHi,lengthHi,squared.x)+squared.y)/(2.0f*lengthHi);
 float2 extension=add2(float2(lengthHi,lengthLo),float2(-x.params.x,-x.params.y));float tension=(extension.x+extension.y)*(x.params.z/(x.params.x+x.params.y));
 float naiveLength=length(x.bh.xyz-x.ah.xyz);float naive=x.params.z*(naiveLength-x.params.x)/x.params.x;
 output[i]=float4(tension,naive,lengthHi,lengthLo);
}
