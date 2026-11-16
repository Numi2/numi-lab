#include <metal_stdlib>
using namespace metal;
#include "metalrobo/engine_types.h"

struct WingSurface { uint body; float side; float length; float area; };
struct WingWind { float4 velocityDensity[3]; uint environments; uint bodyStride; uint pad0; uint pad1; };

static float3 rotated(float4 q,float3 v) {
 q=normalize(q);return v+2.0f*cross(q.xyz,cross(q.xyz,v)+q.w*v);
}

// Experimental no-wake section closure. One thread owns one body wrench;
// all six body indices must be unique. Areas and coefficients are priors.
kernel void crow_articulated_wing_aero(
 device const MRBodyStateGPU* states [[buffer(0)]],
 device MRABABodyWrenchGPU* wrenches [[buffer(1)]],
 constant WingSurface* surfaces [[buffer(2)]],
 constant WingWind& wind [[buffer(3)]],
 device MRABABodyWrenchGPU* observed [[buffer(4)]],
 uint index [[thread_position_in_grid]]) {
 const uint environment=index/6u, k=index%6u;
 if(environment>=wind.environments)return;
 const WingSurface p=surfaces[k];
 if(p.body>=wind.bodyStride)return;
 const MRBodyStateGPU b=states[environment*wind.bodyStride+p.body];
 const float3 chord=rotated(b.orientation,float3(1,0,0));
 const float3 span=rotated(b.orientation,float3(0,p.side,0));
 const float3 normal=normalize(cross(chord,span));
 float3 force=0,torque=0;
 for(uint n=0;n<4;++n){
  float eta=(float(n)+0.5f)/4.0f-.5f;
  float3 r=rotated(b.orientation,float3(0,p.side*p.length*eta,0));
  float3 localVelocity=b.linearVelocityAndInverseMass.xyz+cross(b.angularVelocity.xyz,r);
  float3 incoming=wind.velocityDensity[environment].xyz-localVelocity;
  incoming-=span*dot(incoming,span);
  float speed2=dot(incoming,incoming);if(!(speed2>1e-12f))continue;
  float3 flow=incoming*rsqrt(speed2);float3 travel=-flow;
  float alpha=atan2(dot(travel,normal),dot(travel,chord));
  float cl=clamp(3.0f*alpha,-1.2f,1.2f),cd=0.06f+0.1f*cl*cl;
  float3 lift=normalize(cross(span,flow));
  float3 f=.5f*wind.velocityDensity[environment].w*speed2*(p.area/4.0f)*(cl*lift+cd*flow);
  force+=f;torque+=cross(r,f);
 }
 MRABABodyWrenchGPU r={};r.force=float4(force,0);r.torque=float4(torque,0);observed[index]=r;
 uint wi=environment*wind.bodyStride+p.body;
 MRABABodyWrenchGPU prior=wrenches[wi];prior.force.xyz+=force;prior.torque.xyz+=torque;wrenches[wi]=prior;
}
