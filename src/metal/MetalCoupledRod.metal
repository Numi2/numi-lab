#include <metal_stdlib>
#include "metalrobo/coupled_rod_gpu.h"
using namespace metal;
// Two-term FP32 geometry; the screen checks its final physical residual independently.
struct MRCompensatedScalar { float high; float low; };
inline MRCompensatedScalar sum2(float a,float b){float s=a+b;float x=s-a;return {s,(a-(s-x))+(b-x)};}
inline MRCompensatedScalar mrCompensatedAdd(MRCompensatedScalar a,MRCompensatedScalar b){auto s=sum2(a.high,b.high);return sum2(s.high,s.low+a.low+b.low);}
inline MRCompensatedScalar mrCompensatedMultiply(MRCompensatedScalar a,MRCompensatedScalar b){float p=a.high*b.high;return sum2(p,fma(a.high,b.high,-p)+a.high*b.low+a.low*b.high+a.low*b.low);}
inline MRCompensatedScalar mrCompensatedSqrt(MRCompensatedScalar a){float x=sqrt(a.high);auto v=mrCompensatedMultiply({x,0},{x,0});auto r=mrCompensatedAdd(a,{-v.high,-v.low});return sum2(x,(r.high+r.low)/(2*x));}
inline MRCompensatedScalar pairNeg(MRCompensatedScalar a){return {-a.high,-a.low};}
inline MRCompensatedScalar pairDiv(MRCompensatedScalar a,MRCompensatedScalar b){float q=a.high/b.high;auto r=mrCompensatedAdd(a,pairNeg(mrCompensatedMultiply(b,{q,0})));return mrCompensatedAdd({q,0},{(r.high+r.low)/b.high,0});}
inline void pairAdd(thread MRCompensatedScalar* a,uint i,MRCompensatedScalar v){a[i]=mrCompensatedAdd(a[i],v);}
inline void pairAdd3(thread MRCompensatedScalar* a,uint i,float3 v){pairAdd(a,3*i,{v.x,0});pairAdd(a,3*i+1,{v.y,0});pairAdd(a,3*i+2,{v.z,0});}
constant uint MAXD=35;
inline float3 read3(constant float* a,uint i){return float3(a[3*i],a[3*i+1],a[3*i+2]);}
inline float3 readv(thread const float* a,uint i){return float3(a[3*i],a[3*i+1],a[3*i+2]);}
inline void addv(thread float* a,uint i,float3 v){a[3*i]+=v.x;a[3*i+1]+=v.y;a[3*i+2]+=v.z;}
inline bool eval(constant RodInput& in,thread const float* y,thread const float* yLow,thread const float* direction,thread float* g,thread float* hv) {
 uint n=in.n,m=3*n; float h=in.h;
 MRCompensatedScalar accum[MAXD];for(uint i=0;i<MAXD;i++){g[i]=0;hv[i]=0;accum[i]={0,0};}
 float3 edge[ROD_E],unit[ROD_E],de[ROD_E],du[ROD_E];float lengthE[ROD_E],dl[ROD_E],extension[ROD_E];
 for(uint i=0;i<n-1;i++){
  float3 y0=readv(y,i),y1=readv(y,i+1),z0=readv(yLow,i),z1=readv(yLow,i+1); float3 h0=read3(in.edgeHigh,i), l0=read3(in.edgeLow,i);
  auto ddx=mrCompensatedAdd({y1.x,z1.x},{-y0.x,-z0.x});auto ddy=mrCompensatedAdd({y1.y,z1.y},{-y0.y,-z0.y});auto ddz=mrCompensatedAdd({y1.z,z1.z},{-y0.z,-z0.z});
  auto X=mrCompensatedAdd({h0.x,l0.x},ddx); auto Y=mrCompensatedAdd({h0.y,l0.y},ddy); auto Z=mrCompensatedAdd({h0.z,l0.z},ddz);
  auto sq=mrCompensatedAdd(mrCompensatedMultiply(X,X),mrCompensatedAdd(mrCompensatedMultiply(Y,Y),mrCompensatedMultiply(Z,Z)));
  if(!(sq.high>1e-20f)||!isfinite(sq.high))return false;
  auto len=mrCompensatedSqrt(sq); auto ext=mrCompensatedAdd(len,{-in.ellHigh,-in.ellLow});
  lengthE[i]=len.high+len.low;extension[i]=ext.high+ext.low;
  edge[i]=float3(X.high+X.low,Y.high+Y.low,Z.high+Z.low);unit[i]=edge[i]/lengthE[i];
  de[i]=readv(direction,i+1)-readv(direction,i);dl[i]=dot(unit[i],de[i]);du[i]=(de[i]-unit[i]*dl[i])/lengthE[i];
  float k=in.ea/(in.ellHigh+in.ellLow);auto kp=pairDiv({in.ea,0},{in.ellHigh,in.ellLow});
  auto kext=mrCompensatedMultiply(kp,ext);
  MRCompensatedScalar xyz[3]={X,Y,Z};
  for(uint j=0;j<3;j++){auto u=pairDiv(xyz[j],len);auto force=mrCompensatedMultiply(kext,u);pairAdd(accum,3*i+j,pairNeg(force));pairAdd(accum,3*(i+1)+j,force);}
  float3 df=k*(unit[i]*dl[i]+extension[i]*du[i]);
  addv(hv,i,-df);addv(hv,i+1,df);
 }
 float ell=in.ellHigh+in.ellLow;
 for(uint i=0;i<n-2;i++){
  float3 u=unit[i],v=unit[i+1],d_u=du[i],d_v=du[i+1];float a=lengthE[i],b=lengthE[i+1];
  float c=dot(u,v),dc=dot(d_u,v)+dot(u,d_v);if(!(c>.5f))return false;
  float d=y[m+i+1]-y[m+i];float dd=direction[m+i+1]-direction[m+i];
  float k=in.ei/ell, denom=1+c, r=(1-c)/denom, dr=-2*dc/(denom*denom);
  float fc=-2*k*(1+cos(d))/(denom*denom);
  float dfc=2*k*sin(d)*dd/(denom*denom)+4*k*(1+cos(d))*dc/(denom*denom*denom);
  float3 uc=(v-c*u)/a,vc=(u-c*v)/b;
  float3 duc=(d_v-dc*u-c*d_u)/a-uc*dl[i]/a;
  float3 dvc=(d_u-dc*v-c*d_v)/b-vc*dl[i+1]/b;
  float3 ge=fc*uc,gf=fc*vc,dge=dfc*uc+fc*duc,dgf=dfc*vc+fc*dvc;
  pairAdd3(accum,i,-ge);pairAdd3(accum,i+1,ge);pairAdd3(accum,i+1,-gf);pairAdd3(accum,i+2,gf);
  addv(hv,i,-dge);addv(hv,i+1,dge-dgf);addv(hv,i+2,dgf);
  float gd=-k*r*sin(d)+in.gj/ell*d;
  float dgd=-k*(dr*sin(d)+r*cos(d)*dd)+in.gj/ell*dd;
  pairAdd(accum,m+i,{-gd,0});pairAdd(accum,m+i+1,{gd,0});hv[m+i]-=dgd;hv[m+i+1]+=dgd;
 }
 for(uint i=0;i<n;i++)for(uint j=0;j<3;j++){uint x=3*i+j;float s=in.mass[i]/(h*h);auto predicted=mrCompensatedAdd(mrCompensatedMultiply({h,0},{in.vel[x],0}),mrCompensatedMultiply(mrCompensatedMultiply({h,0},{h,0}),{in.gravity[j],0}));auto diff=mrCompensatedAdd({y[x],yLow[x]},pairNeg(predicted));pairAdd(accum,x,mrCompensatedMultiply({s,0},diff));hv[x]+=s*direction[x];}
 for(uint i=0;i<n-1;i++){float s=in.inertia[i]/(h*h);pairAdd(accum,m+i,{s*(y[m+i]-in.twist[i]-h*in.omega[i]),0});hv[m+i]+=s*direction[m+i];}
 for(uint i=0;i<4*n-1;i++){g[i]=accum[i].high+accum[i].low;if(!isfinite(g[i])||!isfinite(hv[i]))return false;}
 return true;
}
inline float merit(constant RodInput& in,thread const float* g,thread float& lin,thread float& ang){
 uint n=in.n,m=3*n;lin=0;ang=0;
 for(uint i=0;i<n;i++){if((i==0&&(in.fixedEndpoints&1u))||(i==n-1&&(in.fixedEndpoints&2u)))continue;for(uint j=0;j<3;j++)lin=max(lin,abs(g[3*i+j])*in.h/in.mass[i]);}
 for(uint i=0;i<n-1;i++)ang=max(ang,abs(g[m+i])*in.h/in.inertia[i]);
 return max(lin/in.linearTolerance,ang/in.angularTolerance);
}
kernel void mr_research_coupled_rod_step(constant RodInput& in [[buffer(0)]],device RodOutput& out [[buffer(1)]],uint tid [[thread_position_in_grid]]){
 if(tid)return;out.status=1;out.iterations=0;out.backtracks=0;
 uint n=in.n;if(n<3||n>ROD_N)return;uint m=3*n,N=4*n-1,freeN=0;uint fi[MAXD];
 for(uint node=0;node<n;node++){if((node==0&&(in.fixedEndpoints&1u))||(node==n-1&&(in.fixedEndpoints&2u)))continue;for(uint j=0;j<3;j++)fi[freeN++]=3*node+j;}for(uint i=m;i<N;i++)fi[freeN++]=i;
 float y[MAXD],yLow[MAXD],g[MAXD],hv[MAXD],dir[MAXD],mat[MAXD*MAXD],scale[MAXD],rhs[MAXD],candidate[MAXD],candidateLow[MAXD],gt[MAXD],ht[MAXD];
 for(uint i=0;i<MAXD;i++){y[i]=0;yLow[i]=0;dir[i]=0;}
 for(uint i=0;i<n;i++)for(uint j=0;j<3;j++)y[3*i+j]=in.h*(in.vel[3*i+j]+in.h*in.gravity[j]);
 for(uint i=0;i<n-1;i++)y[m+i]=in.twist[i]+in.h*in.omega[i];
 for(uint j=0;j<3;j++){if(in.fixedEndpoints&1u){y[j]=in.targetDelta[j];yLow[j]=in.targetDeltaLow[j];}if(in.fixedEndpoints&2u){y[3*(n-1)+j]=in.targetDelta[3+j];yLow[3*(n-1)+j]=in.targetDeltaLow[3+j];}}
 for(uint iteration=0;iteration<in.maxIterations;iteration++){
  out.iterations=iteration+1;if(!eval(in,y,yLow,dir,g,hv)){out.status=2;return;}
  float lin,ang;float base=merit(in,g,lin,ang);out.maxLinear=lin;out.maxAngular=ang;
  if(lin<=in.linearTolerance && ang<=in.angularTolerance){
   for(uint i=0;i<3*n;i++){out.displacement[i]=y[i];out.displacementLow[i]=yLow[i];}for(uint i=0;i<n-1;i++)out.twistDelta[i]=y[m+i];
   for(uint j=0;j<3;j++){out.reaction[j]=-g[j];out.reaction[3+j]=-g[3*(n-1)+j];}
   out.status=0;return;
  }
  for(uint j=0;j<freeN;j++){
   dir[fi[j]]=1;if(!eval(in,y,yLow,dir,gt,hv)){out.status=2;return;}dir[fi[j]]=0;
   for(uint i=0;i<freeN;i++)mat[i*MAXD+j]=hv[fi[i]];
  }
  for(uint i=0;i<freeN;i++){float d=mat[i*MAXD+i];if(!(d>0)||!isfinite(d)){out.status=3;return;}scale[i]=rsqrt(d);rhs[i]=-g[fi[i]]*scale[i];}
  for(uint i=0;i<freeN;i++)for(uint j=0;j<freeN;j++)mat[i*MAXD+j]*=scale[i]*scale[j];
  for(uint i=0;i<freeN;i++)for(uint j=0;j<=i;j++){
   float v=mat[i*MAXD+j];for(uint k=0;k<j;k++)v-=mat[i*MAXD+k]*mat[j*MAXD+k];
   if(i==j){if(!(v>1e-8f)||!isfinite(v)){out.status=3;return;}mat[i*MAXD+j]=sqrt(v);}else mat[i*MAXD+j]=v/mat[j*MAXD+j];
  }
  for(uint i=0;i<freeN;i++){float v=rhs[i];for(uint j=0;j<i;j++)v-=mat[i*MAXD+j]*rhs[j];rhs[i]=v/mat[i*MAXD+i];}
  for(int i=int(freeN)-1;i>=0;i--){float v=rhs[i];for(uint j=uint(i)+1;j<freeN;j++)v-=mat[j*MAXD+uint(i)]*rhs[j];rhs[i]=v/mat[uint(i)*MAXD+uint(i)];}
  bool accepted=false;
  for(uint step=0;step<12;step++){
   float alpha=ldexp(1.0f,-int(step));for(uint i=0;i<N;i++){candidate[i]=y[i];candidateLow[i]=yLow[i];}
   for(uint i=0;i<freeN;i++){uint k=fi[i];auto z=mrCompensatedAdd({y[k],yLow[k]},{alpha*scale[i]*rhs[i],0});candidate[k]=z.high;candidateLow[k]=(k<m)?z.low:0;}
   if(eval(in,candidate,candidateLow,dir,gt,ht)){float l,a;float test=merit(in,gt,l,a);if(isfinite(test)&&test<base){accepted=true;out.backtracks+=step;break;}}
  }
  if(!accepted){out.status=4;return;}for(uint i=0;i<N;i++){y[i]=candidate[i];yLow[i]=candidateLow[i];}
 }
 out.status=5;
}
