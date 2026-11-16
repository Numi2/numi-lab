#include <metal_stdlib>
using namespace metal;
struct Input { float A[144]; float b[12]; float J[36]; float target[3]; };
struct Output { float x[12]; float lambda[3]; uint status; };
inline bool factor(thread const float *a, thread float *l, uint n) {
 for(uint i=0;i<n*n;++i) l[i]=0.0f;
 for(uint i=0;i<n;++i) for(uint j=0;j<=i;++j) {
  float v=a[i*n+j]; for(uint k=0;k<j;++k) v-=l[i*n+k]*l[j*n+k];
  if(i==j) { if(!(v>0.0f)||!isfinite(v)) return false; l[i*n+j]=sqrt(v); }
  else {v/=l[j*n+j]; if(!isfinite(v))return false; l[i*n+j]=v;}
 } return true;
}
inline bool solve(thread const float *l, thread const float *b, thread float *x, uint n) {
 for(uint i=0;i<n;++i) {float v=b[i];for(uint j=0;j<i;++j)v-=l[i*n+j]*x[j]; x[i]=v/l[i*n+i];}
 for(int i=int(n)-1;i>=0;--i) {float v=x[i];for(uint j=uint(i)+1;j<n;++j)v-=l[j*n+uint(i)]*x[j];x[i]=v/l[uint(i)*n+uint(i)]; if(!isfinite(x[i]))return false;}
 return true;
}
kernel void rod_kkt(device const Input *inputs [[buffer(0)]], device Output *outputs [[buffer(1)]], uint id [[thread_position_in_grid]]) {
 Input in=inputs[id]; Output out={}; out.status=1;float a[144],l[144],b[12],x[12];for(uint i=0;i<144;++i)a[i]=in.A[i];for(uint i=0;i<12;++i)b[i]=in.b[i];
 if(!factor(a,l,12)||!solve(l,b,x,12)){outputs[id]=out;return;}
 float responses[36];for(uint j=0;j<3;++j){float rhs[12],sol[12];for(uint i=0;i<12;++i)rhs[i]=in.J[j*12+i];if(!solve(l,rhs,sol,12)){outputs[id]=out;return;}for(uint i=0;i<12;++i)responses[j*12+i]=sol[i];}
 float w[9],wl[9],rhs[3],lambda[3];for(uint i=0;i<3;++i){float jx=0;for(uint n=0;n<12;++n)jx+=in.J[i*12+n]*x[n];rhs[i]=in.target[i]-jx;for(uint j=0;j<3;++j){float dot=0;for(uint n=0;n<12;++n)dot+=in.J[i*12+n]*responses[j*12+n];w[i*3+j]=dot;}}
 if(!factor(w,wl,3)||!solve(wl,rhs,lambda,3)){outputs[id]=out;return;}
 for(uint i=0;i<12;++i){for(uint j=0;j<3;++j)x[i]+=responses[j*12+i]*lambda[j];if(!isfinite(x[i])){outputs[id]=out;return;}out.x[i]=x[i];}for(uint j=0;j<3;++j)out.lambda[j]=lambda[j];out.status=0;outputs[id]=out;
}
