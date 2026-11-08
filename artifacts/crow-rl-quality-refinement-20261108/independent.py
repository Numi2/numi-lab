from pathlib import Path
import json,struct
import numpy as np
# Uncapped cone() and calibrate() are appended verbatim from a sealed Git source.
def oracle(w,r,s,adh=0.,cap=0.):
 w=np.asarray(w,float);r=np.asarray(r,float);s=np.asarray(s,float);n=len(w)
 shifted=w.copy();shifted[0]-=r[0]*adh
 p,D,kind=cone(shifted,r,s);p[0]-=adh
 if cap>0 and p[0]>cap:
  C=cap+adh;z=-w[1:]/(r[1:]*s[1:]);d=r[1:]*s[1:]**2;c=w[1:]*s[1:]
  if np.linalg.norm(z)<=C:return np.r_[cap,s[1:]*z],np.diag(np.r_[0.,1/r[1:]]),'cap-interior'
  lo=0.;hi=1.
  while np.linalg.norm(c/(d+hi))>C:hi*=2
  for _ in range(160):
   mid=(lo+hi)/2
   if mid==lo or mid==hi:break
   if np.linalg.norm(c/(d+mid))>C:lo=mid
   else:hi=mid
  alpha=(lo+hi)/2;y=-c/(d+alpha);inv=1/(d+alpha);a=inv*y;D=np.zeros((n,n))
  D[1:,1:]=(np.diag(inv)-np.outer(a,a)/(y@a))*np.outer(s[1:],s[1:])
  return np.r_[cap,s[1:]*y],D,'cap-boundary'
 return p,D,kind

def parse(raw):
 h=struct.unpack_from('<8I',raw);assert h[:2]==(0x53504331,1)
 nv,rows,nb=h[2:5];offset=32;blocks=[];covered=np.zeros(rows,bool)
 for _ in range(nb):
  ints=struct.unpack_from('<8I',raw,offset);f=struct.unpack_from('<20f',raw,offset+32);at,n,kind,flags=ints[:4]
  assert n>0 and at+n<=rows and not covered[at:at+n].any();covered[at:at+n]=True
  blocks.append(dict(at=at,n=n,kind=kind,s=np.array(f[:8])[:n],r=np.array(f[8:16])[:n],bounds=np.array(f[16:20])))
  offset+=128
 def take(n):
  nonlocal offset
  a=np.frombuffer(raw,dtype='<f4',count=n,offset=offset).astype(float);offset+=4*n;return a
 A=take(nv*nv).reshape(nv,nv);J=take(rows*nv).reshape(rows,nv);bias=take(rows);free=take(nv);start=take(nv)
 assert offset==len(raw) and np.array_equal(A,A.T) and np.linalg.eigvalsh(A)[0]>0
 assert np.isfinite(np.r_[A.ravel(),J.ravel(),bias,free,start]).all()
 return dict(nv=nv,rows=rows,nb=nb,blocks=blocks,A=A,J=J,bias=bias,free=free,start=start,header=h)

def evaluate(c,v):
 A,J=c['A'],c['J'];w=J@v+c['bias'];p=np.zeros(c['rows']);W=[];H=A.copy();prox=0.;feas=0.;regions=[]
 for b in c['blocks']:
  sl=slice(b['at'],b['at']+b['n']);local=w[sl];r=b['r'];assert np.min(r)>0
  if b['kind']==0:
   lo,hi=b['bounds'][:2];x=-local[0]/r[0];lp=np.array([np.clip(x,lo,hi)])
   derivative=np.array([[1/r[0] if lo<x<hi else 0.]]);kind='scalar-free' if lo<x<hi else 'scalar-bound'
   violation=max(lo-lp[0],lp[0]-hi,0.)
  else:
   lp,derivative,kind=oracle(local,r,b['s'],*b['bounds'][:2])
   shifted=lp[0]+b['bounds'][0];violation=max(0.,-shifted,np.linalg.norm(lp[1:]/b['s'][1:])-shifted)
   if b['bounds'][1]>0:violation=max(violation,lp[0]-b['bounds'][1])
  p[sl]=lp;W.append(derivative);regions.append(kind);feas=max(feas,violation)
  prox+=-.5*float(lp@(r*lp))-float(local@lp);H+=J[sl].T@derivative@J[sl]
 dv=v-c['free'];av=A@dv;jp=J.T@p;g=av-jp
 return dict(objective=.5*float(dv@av)+prox,g=g,p=p,W=W,H=H,w=w,regions=regions,feas=feas,res=float(np.max(np.abs(g))/(1+np.max(np.abs(av))+np.max(np.abs(jp)))))

def reference(c):
 v=c['start'].copy();history=[]
 for it in range(80):
  q=evaluate(c,v);history.append(q['res'])
  if q['res']<=1e-13:return v,q,history
  d=np.linalg.solve(q['H'],-q['g']);gd=float(q['g']@d);assert gd<0
  step=1.
  for _ in range(40):
   candidate=v+step*d;t=evaluate(c,candidate)
   if t['objective']<=q['objective']+1e-4*step*gd or t['res']<q['res']*.8:v=candidate;break
   step*=.5
  else:raise AssertionError('reference line search')
 raise AssertionError('reference iterations')

def cone(w,r,s):
    # Independent FP64 weighted Lorentz-cone minimization. Boundary KKT is
    # reduced to one monotone scalar root and bracketed by bisection.
    d=r*s*s;c=w*s;z=-c/d
    if np.linalg.norm(z[1:])<=z[0]:return s*z,np.diag(1/r),'interior'
    if c[0]>=np.linalg.norm(c[1:]):return np.zeros(len(w)),np.zeros((len(w),len(w))),'vertex'
    low=max(0.,float(c[0]));high=max(1.,abs(float(c[0])),float(np.linalg.norm(c[1:])))
    def f(tau):
        n=(tau-c[0])/d[0]
        return np.linalg.norm(c[1:]/(d[1:]*n+tau))-1.
    while f(high)>0:high*=2
    for _ in range(160):
        mid=(low+high)/2
        if mid==low or mid==high:break
        if f(mid)>0:low=mid
        else:high=mid
    tau=(low+high)/2;n=(tau-c[0])/d[0]
    z[0]=n;z[1:]=-c[1:]*n/(d[1:]*n+tau)
    u=z[1:]/np.linalg.norm(z[1:]);h=np.diag(d)
    h[1:,1:]+=tau/n*(np.eye(len(w)-1)-np.outer(u,u))
    hi=np.linalg.inv(h);normal=np.r_[-1.,u];ha=hi@normal
    derivative=(hi-np.outer(ha,ha)/(normal@ha))*np.outer(s,s)
    return s*z,derivative,'boundary'

def calibrate():
    vectors=[[-2.,.2,.1],[3.,.2,-.1],[0.,2.,1.],[1.,-2.,3.],[-1.,2.,-.5]]
    for v in vectors:
        w=np.array(v);x=-w;t=np.linalg.norm(x[1:])
        expected=x if t<=x[0] else np.zeros(3) if t<=-x[0] else np.r_[(t+x[0])/2,x[1:]*(t+x[0])/(2*t)]
        p,_,_=cone(w,np.ones(3),np.ones(3));assert np.max(np.abs(p-expected))<2e-14
    w=np.array([.3,-1.,.5]);r=np.array([.8,1.1,2.3]);s=np.array([1.,.4,.7])
    p,derivative,kind=cone(w,r,s);assert kind=='boundary'
    numerical=np.column_stack([(cone(w+np.eye(3)[i]*1e-6,r,s)[0]-cone(w-np.eye(3)[i]*1e-6,r,s)[0])/(-2e-6) for i in range(3)])
    assert np.max(np.abs(derivative-numerical))<2e-9
    return dict(euclidean_closed_form_cases=5,weighted_boundary_derivative_max_error=float(np.max(np.abs(derivative-numerical))),passed=True)

