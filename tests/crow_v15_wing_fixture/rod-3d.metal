#include <metal_stdlib>
using namespace metal;
struct Input {float U[729],root[27],omega[27],q[27],v[27],dt;};
kernel void rod_modal(device const Input& in [[buffer(0)]],device float* out [[buffer(1)]],device uint& status [[buffer(2)]],uint tid [[thread_position_in_grid]]) {
 if(tid!=0)return;status=1;float q[27],v[27];
 for(uint j=0;j<27;++j){q[j]=0;v[j]=0;for(uint i=0;i<27;++i){float c=in.U[i*27+j]*in.root[i];q[j]+=c*in.q[i];v[j]+=c*in.v[i];}}
 for(uint t=0;t<80;++t){
  for(uint j=0;j<27;++j){float w=in.omega[j],p=q[j],u=v[j];if(w==0){q[j]=p+in.dt*u;v[j]=u;}else{float c=cos(w*in.dt),s=sin(w*in.dt);q[j]=c*p+s*u/w;v[j]=-w*s*p+c*u;}}
  for(uint i=0;i<27;++i){float p=0,u=0,a=0;for(uint j=0;j<27;++j){float basis=in.U[i*27+j];p+=basis*q[j];u+=basis*v[j];a-=basis*in.omega[j]*in.omega[j]*q[j];}p/=in.root[i];u/=in.root[i];a/=in.root[i];if(!isfinite(p)||!isfinite(u)||!isfinite(a))return;out[t*81+i]=p;out[t*81+27+i]=u;out[t*81+54+i]=a;}
 }status=0;
}
