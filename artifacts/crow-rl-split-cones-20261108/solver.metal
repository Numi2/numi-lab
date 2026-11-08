#include "arithmetic.metal.inc"
#include "projector.metal.inc"
struct Params {uint magic,version,nv,rows,blocks,maxiters,rounding,reserved;};
inline DS groupSumDS(DS x,uint lane) {
 for(uint d=16;d>0;d>>=1){DS y{simd_shuffle_down(x.h,d),simd_shuffle_down(x.l,d)};if(lane<d)x=add(x,y);}
 return {simd_broadcast_first(x.h),simd_broadcast_first(x.l)};
}
inline bool evaluateDS(constant Params& c,device const MRUnifiedQualityBlockGPU* blocks,
 device const float* A,device const float* J,device const float* bias,device const float* freev,
 device DS* v,device DS* w,device DS* p,device DS* g,device DS* W,
 device DS* objective,device uint* status,uint lane) {
 device float* metrics=reinterpret_cast<device float*>(status+16);
 for(uint row=lane;row<c.rows;row+=32) {
  DS x=ds(bias[row]);for(uint k=0;k<c.nv;++k)x=add(x,mul(ds(J[row*c.nv+k]),v[k]));
  w[row]=x;p[row]=ds(0);
 }
 threadgroup_barrier(mem_flags::mem_device);
 DS prox=ds(0),feas=ds(0);uint counts[7];for(uint i=0;i<7;++i)counts[i]=0;uint failure=0xffffffffu;
 for(uint bi=lane;bi<c.blocks;bi+=32) {
  auto b=blocks[bi];uint n=b.layout.y,at=b.layout.x;
  for(uint k=0;k<36;++k)W[bi*36+k]=ds(0);
  if(n==0||n>6||at>c.rows||n>c.rows-at){failure=min(failure,bi);continue;}
  DS bw[6],bp[6],bd[36],bo,bf;uint region=0;
  for(uint k=0;k<6;++k)bw[k]=k<n?w[at+k]:ds(0);
  if(!projectDS(b,bw,bp,bd,bo,bf,region)){failure=min(failure,bi);continue;}
  for(uint k=0;k<n;++k)p[at+k]=bp[k];
  for(uint k=0;k<36;++k)W[bi*36+k]=bd[k];
  prox=add(prox,bo);feas=dsmax(feas,bf);++counts[region];
 }
 failure=simd_min(failure);
 for(uint i=0;i<7;++i){uint total=simd_sum(counts[i]);if(lane==0)status[2+i]=total;}
 if(lane==0)status[9]=failure;
 threadgroup_barrier(mem_flags::mem_device);
 if(failure!=0xffffffffu)return false;
 float gm=0,am=0,jm=0,fm=simd_max(mag(feas));DS dynamicsObjective=ds(0);
 for(uint row=lane;row<c.nv;row+=32) {
  DS action=ds(0),constraint=ds(0);
  for(uint k=0;k<c.nv;++k)action=add(action,mul(ds(A[row*c.nv+k]),sub(v[k],ds(freev[k]))));
  for(uint k=0;k<c.rows;++k)constraint=add(constraint,mul(ds(J[k*c.nv+row]),p[k]));
  g[row]=sub(action,constraint);gm=max(gm,mag(g[row]));am=max(am,mag(action));jm=max(jm,mag(constraint));
  dynamicsObjective=add(dynamicsObjective,mul(ds(.5f),mul(sub(v[row],ds(freev[row])),action)));
 }
 DS value=groupSumDS(add(dynamicsObjective,prox),lane);gm=simd_max(gm);am=simd_max(am);jm=simd_max(jm);
 if(lane==0){objective[0]=value;metrics[0]=gm/(1+am+jm);metrics[1]=gm;metrics[2]=am;metrics[3]=jm;metrics[4]=fm;metrics[5]=value.h+value.l;}
 threadgroup_barrier(mem_flags::mem_device);
 return fin(value)&&isfinite(gm)&&isfinite(am)&&isfinite(jm)&&isfinite(fm);
}
kernel void solve(constant Params& c [[buffer(0)]],
 device const MRUnifiedQualityBlockGPU* blocks [[buffer(1)]],
 device const float* A [[buffer(2)]],device const float* J [[buffer(3)]],
 device const float* bias [[buffer(4)]],device const float* freev [[buffer(5)]],
 device const float* start [[buffer(6)]],device DS* scratch [[buffer(7)]],
 device uint* status [[buffer(8)]],uint lane [[thread_index_in_threadgroup]]) {
 device DS* v=scratch;device DS* w=v+c.nv;device DS* p=w+c.rows;device DS* g=p+c.rows;
 device DS* W=g+c.nv;device DS* H=W+36*c.blocks;device DS* direction=H+c.nv*c.nv;
 device DS* base=direction+c.nv;device DS* objective=base+c.nv;
 device float* metrics=reinterpret_cast<device float*>(status+16);
 if(lane==0){for(uint i=0;i<32;++i)status[i]=0;status[9]=0xffffffffu;}
 for(uint k=lane;k<c.nv;k+=32)v[k]=ds(start[k]);
 threadgroup_barrier(mem_flags::mem_device);
 uint code=7,iterations=0,backtracks=0;
 for(uint it=0;it<=c.maxiters;++it) {
  if(!evaluateDS(c,blocks,A,J,bias,freev,v,w,p,g,W,objective,status,lane)){code=2;break;}
  if(metrics[0]<=1e-10f&&metrics[4]<=1e-10f){code=0;break;}
  if(it==c.maxiters)break;
  for(uint index=lane;index<c.nv*c.nv;index+=32) {
   uint row=index/c.nv,col=index%c.nv;DS x=ds(A[index]);
   for(uint bi=0;bi<c.blocks;++bi) {
    uint at=blocks[bi].layout.x,n=blocks[bi].layout.y;
    for(uint a=0;a<n;++a)for(uint b=0;b<n;++b) {
     DS weight=W[bi*36+a*n+b];
     if(weight.h!=0||weight.l!=0)x=add(x,mul(mul(ds(J[(at+a)*c.nv+row]),weight),ds(J[(at+b)*c.nv+col])));
    }
   }
   H[index]=x;
  }
  threadgroup_barrier(mem_flags::mem_device);
  bool valid=true;
  for(uint k=0;k<c.nv;++k) {
   if(lane==0){DS x=H[k*c.nv+k];for(uint s=0;s<k;++s)x=sub(x,mul(mul(H[k*c.nv+s],H[k*c.nv+s]),H[s*c.nv+s]));H[k*c.nv+k]=x;}
   threadgroup_barrier(mem_flags::mem_device);
   if(!(H[k*c.nv+k].h>0)||!fin(H[k*c.nv+k])){valid=false;break;}
   for(uint r=k+1+lane;r<c.nv;r+=32){DS x=H[r*c.nv+k];for(uint s=0;s<k;++s)x=sub(x,mul(mul(H[r*c.nv+s],H[k*c.nv+s]),H[s*c.nv+s]));H[r*c.nv+k]=divds(x,H[k*c.nv+k]);}
   threadgroup_barrier(mem_flags::mem_device);
  }
  if(!valid){code=4;break;}
  // Instrument-only triangular solve; no production dispatch uses this kernel.
  if(lane==0) {
   for(uint r=0;r<c.nv;++r){DS x=neg(g[r]);for(uint k=0;k<r;++k)x=sub(x,mul(H[r*c.nv+k],direction[k]));direction[r]=x;}
   for(uint r=0;r<c.nv;++r)direction[r]=divds(direction[r],H[r*c.nv+r]);
   for(int r=int(c.nv)-1;r>=0;--r){DS x=direction[r];for(uint k=r+1;k<c.nv;++k)x=sub(x,mul(H[k*c.nv+r],direction[k]));direction[r]=x;}
  }
  threadgroup_barrier(mem_flags::mem_device);
  DS dot=ds(0);for(uint r=lane;r<c.nv;r+=32){dot=add(dot,mul(g[r],direction[r]));base[r]=v[r];}
  dot=groupSumDS(dot,lane);DS oldObjective=objective[0];float oldResidual=metrics[0];
  if(!fin(dot)||!lt(dot,ds(0))){code=8;break;}
  threadgroup_barrier(mem_flags::mem_device);
  float step=1;bool accepted=false;
  for(uint trial=0;trial<16;++trial) {
   for(uint r=lane;r<c.nv;r+=32){v[r]=add(base[r],mul(ds(step),direction[r]));if(c.rounding)v[r]=ds(v[r].h+v[r].l);}
   threadgroup_barrier(mem_flags::mem_device);
   bool evaluated=evaluateDS(c,blocks,A,J,bias,freev,v,w,p,g,W,objective,status,lane);
   DS armijo=add(oldObjective,mul(ds(1e-4f*step),dot));
   DS slack=ds(32*0x1p-46f*max(mag(oldObjective),1.0f));
   accepted=evaluated&&((metrics[0]<=1e-10f&&metrics[4]<=1e-10f)||
      le(objective[0],armijo)||(metrics[0]<.8f*oldResidual&&le(objective[0],add(oldObjective,slack))));
   if(accepted)break;
   ++backtracks;step*=.5f;
  }
  if(lane==0)metrics[7]=step;
  if(!accepted) {
   for(uint r=lane;r<c.nv;r+=32)v[r]=base[r];
   threadgroup_barrier(mem_flags::mem_device);
   evaluateDS(c,blocks,A,J,bias,freev,v,w,p,g,W,objective,status,lane);code=6;break;
  }
  iterations=it+1;
 }
 if(lane==0){status[0]=code;status[1]=iterations;status[10]=backtracks;}
}
