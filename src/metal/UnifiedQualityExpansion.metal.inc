// Bounded two-FP32 refinement of an already validated small quality problem.
// Inputs/publication stay FP32; no expanded iterate alone can authorize publication.
namespace mr_quality_expansion {
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
// Two-FP32 product-cone projection. Research path; no canonical ABI change.
inline bool fin(DS a) {return isfinite(a.h)&&isfinite(a.l);}
inline DS dsmax(DS a,DS b) {return lt(a,b)?b:a;}
inline DS dsabs(DS a) {return lt(a,ds(0))?neg(a):a;}
inline bool inverseDS(thread DS a[7][7],thread DS inv[7][7],uint n) {
 for(uint i=0;i<7;++i)for(uint j=0;j<7;++j)inv[i][j]=ds(i==j?1:0);
 for(uint k=0;k<n;++k) {
  uint pivot=k;float m=mag(a[k][k]);
  for(uint i=k+1;i<n;++i)if(mag(a[i][k])>m){m=mag(a[i][k]);pivot=i;}
  if(!(m>1e-20f)||!isfinite(m))return false;
  if(pivot!=k)for(uint j=0;j<n;++j){DS x=a[k][j];a[k][j]=a[pivot][j];a[pivot][j]=x;x=inv[k][j];inv[k][j]=inv[pivot][j];inv[pivot][j]=x;}
  DS d=a[k][k];
  for(uint j=0;j<n;++j){a[k][j]=divds(a[k][j],d);inv[k][j]=divds(inv[k][j],d);}
  for(uint i=0;i<n;++i)if(i!=k) {
   DS factor=a[i][k];
   for(uint j=0;j<n;++j){a[i][j]=sub(a[i][j],mul(factor,a[k][j]));inv[i][j]=sub(inv[i][j],mul(factor,inv[k][j]));}
  }
 }
 return true;
}
inline DS coneValueDS(thread const DS* d,thread const DS* z,uint n,DS eta,thread DS* y) {
 DS normal=add(z[0],divds(eta,d[0]));
 if(!lt(ds(0),normal))return ds(INFINITY);
 DS alpha=divds(eta,normal),sum=ds(0);y[0]=normal;
 for(uint k=1;k<n;++k){y[k]=divds(mul(d[k],z[k]),add(d[k],alpha));sum=add(sum,mul(y[k],y[k]));}
 return sub(sqrtds(dsmax(sum,ds(0))),normal);
}
inline DS ballValueDS(thread const DS* d,thread const DS* z,uint n,DS radius,DS alpha,thread DS* y) {
 DS sum=ds(0);y[0]=radius;
 for(uint k=1;k<n;++k){y[k]=divds(mul(d[k],z[k]),add(d[k],alpha));sum=add(sum,mul(y[k],y[k]));}
 return sub(sqrtds(dsmax(sum,ds(0))),radius);
}
// Regions 0 vertex, 1 interior, 2 cone-boundary, 3 capped-interior,
// 4 capped-boundary, 5 scalar-free, 6 scalar-bound.
inline bool projectDS(MRUnifiedQualityBlockGPU b,thread const DS* w,
 thread DS* p,thread DS* W,thread DS& objective,thread DS& feasibility,thread uint& region) {
 uint n=b.layout.y;
 for(uint i=0;i<6;++i)p[i]=ds(0);
 for(uint i=0;i<36;++i)W[i]=ds(0);
 objective=ds(0);feasibility=ds(0);region=0;
 if(n==0||n>6)return false;
 for(uint i=0;i<n;++i)if(!fin(w[i]))return false;
 if(b.layout.z==MR_UNIFIED_QUALITY_SCALAR_INTERVAL) {
  float r=b.regularization0.x,lo=b.boundsAndShift.x,hi=b.boundsAndShift.y;
  if(n!=1||!(r>0)||!isfinite(r)||!isfinite(lo)||!isfinite(hi)||lo>hi)return false;
  DS x=neg(divds(w[0],ds(r)));region=6;
  if(lt(ds(lo),x)&&lt(x,ds(hi))){p[0]=x;W[0]=divds(ds(1),ds(r));region=5;}
  else p[0]=le(x,ds(lo))?ds(lo):ds(hi);
  objective=neg(add(mul(ds(.5f*r),mul(p[0],p[0])),mul(w[0],p[0])));
  return fin(objective)&&fin(p[0]);
 }
 if(b.layout.z!=MR_UNIFIED_QUALITY_ELLIPTIC_CONE||n<3||
    b.boundsAndShift.x<0||b.boundsAndShift.y<0||
    !isfinite(b.boundsAndShift.x)||!isfinite(b.boundsAndShift.y)||b.scale0.x!=1)return false;
 DS s[6],r[6],d[6],z[6],y[6],ts=ds(0),dual=ds(0);
 for(uint i=0;i<n;++i) {
  float sr=bf(b.scale0,b.scale1,i),rr=bf(b.regularization0,b.regularization1,i);
  if(!(sr>0)||!(rr>0)||!isfinite(sr)||!isfinite(rr))return false;
  s[i]=ds(sr);r[i]=ds(rr);d[i]=mul(r[i],mul(s[i],s[i]));z[i]=neg(divds(w[i],mul(r[i],s[i])));
  if(!fin(d[i])||!fin(z[i])||!(d[i].h>0))return false;
 }
 DS adhesion=ds(b.boundsAndShift.x);z[0]=add(z[0],adhesion);
 for(uint i=1;i<n;++i){ts=add(ts,mul(z[i],z[i]));DS a=mul(d[i],z[i]);dual=add(dual,mul(a,a));}
 DS eta=ds(0),alpha=ds(0);
 if(le(sqrtds(dsmax(ts,ds(0))),z[0])){region=1;for(uint i=0;i<n;++i)y[i]=z[i];}
 else if(le(mul(d[0],z[0]),neg(sqrtds(dsmax(dual,ds(0)))))){region=0;for(uint i=0;i<n;++i)y[i]=ds(0);}
 else {
  region=2;DS lo=dsmax(ds(0),neg(mul(d[0],z[0]))),hi=dsmax(ds(1),add(lo,dsmax(d[0],ds(1)))),work[6];
  DS f=coneValueDS(d,z,n,hi,work);
  for(uint i=0;i<32&&(!fin(f)||lt(ds(0),f));++i){hi=add(mul(ds(2),hi),dsmax(d[0],ds(1)));f=coneValueDS(d,z,n,hi,work);}
  if(!fin(f)||lt(ds(0),f))return false;
  DS best=hi,bestError=dsabs(f);
  for(uint i=0;i<80;++i) {
   DS mid=mul(ds(.5f),add(lo,hi));if(le(mid,lo)||le(hi,mid))break;
   f=coneValueDS(d,z,n,mid,work);
   if(fin(f)&&lt(dsabs(f),bestError)){best=mid;bestError=dsabs(f);}
   if(!fin(f)||lt(ds(0),f))lo=mid;else hi=mid;
  }
  eta=best;if(!fin(coneValueDS(d,z,n,eta,y)))return false;alpha=divds(eta,y[0]);
 }
 DS cap=add(ds(b.boundsAndShift.y),adhesion);
 if(b.boundsAndShift.y>0&&lt(cap,y[0])) {
  if(le(sqrtds(dsmax(ts,ds(0))),cap)){region=3;y[0]=cap;for(uint i=1;i<n;++i)y[i]=z[i];}
  else {
   region=4;DS lo=ds(0),hi=ds(1),work[6],f=ballValueDS(d,z,n,cap,hi,work);
   for(uint i=0;i<32&&(!fin(f)||lt(ds(0),f));++i){hi=add(mul(ds(2),hi),dsmax(d[0],ds(1)));f=ballValueDS(d,z,n,cap,hi,work);}
   if(!fin(f)||lt(ds(0),f))return false;
   DS best=hi,bestError=dsabs(f);
   for(uint i=0;i<80;++i) {
    DS mid=mul(ds(.5f),add(lo,hi));if(le(mid,lo)||le(hi,mid))break;
    f=ballValueDS(d,z,n,cap,mid,work);
    if(fin(f)&&lt(dsabs(f),bestError)){best=mid;bestError=dsabs(f);}
    if(lt(ds(0),f))lo=mid;else hi=mid;
   }
   alpha=best;if(!fin(ballValueDS(d,z,n,cap,alpha,y)))return false;
  }
 }
 if(region==1||region==3)for(uint i=region==3?1:0;i<n;++i)W[i*n+i]=divds(ds(1),r[i]);
 else if(region==2) {
  DS u[6],a[7][7],inv[7][7];u[0]=ds(-1);
  for(uint i=1;i<n;++i)u[i]=divds(y[i],y[0]);
  for(uint i=0;i<7;++i)for(uint j=0;j<7;++j)a[i][j]=ds(0);
  a[0][0]=d[0];
  for(uint i=1;i<n;++i)for(uint j=1;j<n;++j)a[i][j]=add(i==j?d[i]:ds(0),mul(alpha,sub(ds(i==j?1:0),mul(u[i],u[j]))));
  for(uint i=0;i<n;++i){a[i][n]=u[i];a[n][i]=u[i];}
  if(!inverseDS(a,inv,n+1))return false;
  for(uint i=0;i<n;++i)for(uint j=0;j<n;++j)W[i*n+j]=mul(mul(s[i],inv[i][j]),s[j]);
 } else if(region==4) {
  // Ball KKT multiplier is alpha*radius; its tangent curvature is alpha.
  DS inv[6],a[6],denom=ds(0);
  for(uint i=1;i<n;++i){inv[i]=divds(ds(1),add(d[i],alpha));a[i]=mul(inv[i],y[i]);denom=add(denom,mul(y[i],a[i]));}
  if(!lt(ds(0),denom))return false;
  for(uint i=1;i<n;++i)for(uint j=1;j<n;++j)W[i*n+j]=mul(mul(s[i],sub(i==j?inv[i]:ds(0),divds(mul(a[i],a[j]),denom))),s[j]);
 }
 DS tangent=ds(0);
 for(uint i=0;i<n;++i) {
  p[i]=sub(mul(s[i],y[i]),i==0?adhesion:ds(0));
  objective=sub(objective,add(mul(ds(.5f),mul(r[i],mul(p[i],p[i]))),mul(w[i],p[i])));
  if(i){DS q=divds(p[i],s[i]);tangent=add(tangent,mul(q,q));}
  if(!fin(p[i]))return false;
 }
 DS shifted=add(p[0],adhesion);
 feasibility=dsmax(ds(0),dsmax(neg(shifted),sub(sqrtds(dsmax(tangent,ds(0))),shifted)));
 if(b.boundsAndShift.y>0)feasibility=dsmax(feasibility,sub(p[0],ds(b.boundsAndShift.y)));
 for(uint i=0;i<n*n;++i)if(!fin(W[i]))return false;
 return fin(objective)&&fin(feasibility);
}
struct Params {uint magic,version,nv,rows,blocks,maxiters,rounding,reserved;};
inline DS groupSumDS(DS x,uint lane) {
 for(uint d=16;d>0;d>>=1){DS y{simd_shuffle_down(x.h,d),simd_shuffle_down(x.l,d)};if(lane<d)x=add(x,y);}
 return {simd_broadcast_first(x.h),simd_broadcast_first(x.l)};
}
inline bool evaluateDS(thread const Params& c,device const MRUnifiedQualityBlockGPU* blocks,
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
inline bool solveCandidate(thread const Params& c,
 device const MRUnifiedQualityBlockGPU* blocks,device const float* A,
 device const float* J,device const float* bias,device const float* freev,
 threadgroup const float* start,device DS* scratch,device uint* status,uint lane) {
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
  // Ordered dependency fronts with all independent coordinate updates on SIMD32.
  for(uint r=lane;r<c.nv;r+=32)direction[r]=neg(g[r]);
  threadgroup_barrier(mem_flags::mem_device);
  for(uint k=0;k<c.nv;++k) {
   DS x=direction[k];
   for(uint r=k+1+lane;r<c.nv;r+=32)direction[r]=sub(direction[r],mul(H[r*c.nv+k],x));
   threadgroup_barrier(mem_flags::mem_device);
  }
  for(uint r=lane;r<c.nv;r+=32)direction[r]=divds(direction[r],H[r*c.nv+r]);
  threadgroup_barrier(mem_flags::mem_device);
  for(int k=int(c.nv)-1;k>=0;--k) {
   DS x=direction[k];
   for(uint r=lane;r<uint(k);r+=32)direction[r]=sub(direction[r],mul(H[uint(k)*c.nv+r],x));
   threadgroup_barrier(mem_flags::mem_device);
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
 threadgroup_barrier(mem_flags::mem_device);
 return code==0;
}

inline bool auditPublished(thread const Params& c,
 device const MRUnifiedQualityBlockGPU* blocks,device const float* A,
 device const float* J,device const float* freev,device const DS* v,
 threadgroup const float* impulses,thread float& residual,thread float& feasibility,uint lane) {
 float gm=0,am=0,jm=0,fm=0;bool valid=true;
 for(uint row=lane;row<c.nv;row+=32) {
  DS av=ds(0),jp=ds(0);
  for(uint k=0;k<c.nv;++k)av=add(av,mul(ds(A[row*c.nv+k]),sub(v[k],ds(freev[k]))));
  for(uint k=0;k<c.rows;++k)jp=add(jp,mul(ds(J[k*c.nv+row]),ds(impulses[k])));
  valid=valid&&fin(av)&&fin(jp);gm=max(gm,mag(sub(av,jp)));am=max(am,mag(av));jm=max(jm,mag(jp));
 }
 for(uint bi=lane;bi<c.blocks;bi+=32) {
  auto b=blocks[bi];uint at=b.layout.x,n=b.layout.y;DS f=ds(0);
  if(b.layout.z==MR_UNIFIED_QUALITY_SCALAR_INTERVAL)f=dsmax(ds(0),dsmax(sub(ds(b.boundsAndShift.x),ds(impulses[at])),sub(ds(impulses[at]),ds(b.boundsAndShift.y))));
  else {
   DS shifted=add(ds(impulses[at]),ds(b.boundsAndShift.x)),sum=ds(0);
   for(uint k=1;k<n;++k){DS t=divds(ds(impulses[at+k]),ds(bf(b.scale0,b.scale1,k)));sum=add(sum,mul(t,t));}
   f=dsmax(ds(0),dsmax(neg(shifted),sub(sqrtds(dsmax(sum,ds(0))),shifted)));
   if(b.boundsAndShift.y>0)f=dsmax(f,sub(ds(impulses[at]),ds(b.boundsAndShift.y)));
  }
  valid=valid&&fin(f);fm=max(fm,mag(f));
 }
 gm=simd_max(gm);am=simd_max(am);jm=simd_max(jm);fm=simd_max(fm);
 residual=gm/(1+am+jm);feasibility=fm;
 return simd_all(valid)&&isfinite(residual)&&isfinite(feasibility);
}
}
