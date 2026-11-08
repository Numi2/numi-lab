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
