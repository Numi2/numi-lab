#!/usr/bin/env python3
"""Bounded offline reference-layer quality fit, retaining exact source outer coordinates/topology."""
import csv, pathlib, json, hashlib, time, sys
import numpy as np
from scipy.optimize import minimize
ROOT=pathlib.Path(__file__).resolve().parents[5]
# Run from repository root; script location is retained as evidence.
ROOT=pathlib.Path.cwd()
OUT=pathlib.Path(__file__).resolve().parent
src=ROOT/"tools/evidence/human-wholebody-skinfat-20261019/c296"
raw=np.fromstring((src/"candidate-024-safe.txt").read_text(),sep=" ",dtype=int)
n,nf=raw[:2];Fraw=raw[2+n:2+n+3*nf].reshape(-1,3);F=np.sort(Fraw,axis=1)
rows=list(csv.DictReader((src/"admitted-layer-coordinates.csv").open()))
P0=np.array([[[float(r[p+"_"+a])*1000 for a in "xyz"] for r in rows] for p in ("outer","skin","fat")])
a,b,c=F.T
Ts=[]
for k in range(2):
 Ts.extend([np.stack([a+k*n,b+k*n,c+k*n,c+(k+1)*n],1),
            np.stack([a+k*n,b+k*n,b+(k+1)*n,c+(k+1)*n],1),
            np.stack([a+k*n,a+(k+1)*n,b+(k+1)*n,c+(k+1)*n],1)])
# Order band, slot, face. No topology changes.
T=np.concatenate(Ts)
P0=P0.reshape(-1,3)
parity=np.array([(-1)**sum(f[i]>f[j] for i in range(3) for j in range(i+1,3)) for f in Fraw])
sg=np.tile(np.concatenate([-parity,parity,-parity]),2)
corn=P0[T];edges=corn[:,:,None,:]-corn[:,None,:,:]
scale=np.maximum(np.linalg.norm(edges,axis=3).max(axis=(1,2))**3,1e-9)
def measure(P):
 v=P[T];E=v[:,1:]-v[:,:1];det=np.einsum("ij,ij->i",E[:,0],np.cross(E[:,1],E[:,2]))
 sv=np.linalg.svd(E,compute_uv=False)
 return det*sg/scale, sv[:,0]/sv[:,-1],det
q0,cond0,det0=measure(P0)
target=1e-4
# Include both material bands around every quality-deficient tet; all other points fixed.
active=np.unique(T[q0<target]);active=active[active>=n]
# Protect one ring to smooth the small inferred boundary correction.
av=(active%n)
ring=np.unique(F[np.any(np.isin(F,av),axis=1)])
active=np.concatenate([ring+n,ring+2*n])
D=np.array([[float(r["outward_"+a]) for a in "xyz"] for r in rows])
W=1e10
def calc(z):
 P=P0.copy();delta=z.reshape(-1,3);P[active]+=delta
 p=P[T];u=p[:,1]-p[:,0];v=p[:,2]-p[:,0];w=p[:,3]-p[:,0]
 det=np.einsum("ij,ij->i",u,np.cross(v,w));q=sg*det/scale
 gap=np.minimum(q-target*1.05,0)
 E=.5*np.sum((delta/.1)**2)+.5*W*np.sum(gap*gap)
 grad=np.zeros_like(P);gv=W*gap*sg/scale
 gu=np.cross(v,w)*gv[:,None];ggv=np.cross(w,u)*gv[:,None];gw=np.cross(u,v)*gv[:,None]
 for slot,g in enumerate((-gu-ggv-gw,gu,ggv,gw)):np.add.at(grad,T[:,slot],g)
 # Same declared thickness bounds plus inward projection; construction, not measured anatomy.
 for k,lo,hi in ((0,.50001,2.49999),(1,.50001,9.99999)):
  ia=np.arange(n)+k*n;ib=ia+n;d=P[ia]-P[ib];l=np.linalg.norm(d,axis=1)
  gapL=np.minimum(l-lo,0)+np.maximum(l-hi,0)
  E+=.5*1e8*np.sum(gapL*gapL)
  g=1e8*gapL[:,None]*d/l[:,None]
  proj=np.einsum("ij,ij->i",d,D);gapP=np.minimum(proj-.05,0)
  E+=.5*1e8*np.sum(gapP*gapP);g+=1e8*gapP[:,None]*D
  grad[ia]+=g;grad[ib]-=g
 return float(E),(grad[active]+delta/.01).ravel()
z=np.zeros(len(active)*3)
# Directional derivative check before optimization.
rng=np.random.default_rng(20261020);direction=rng.normal(size=z.size);direction/=np.linalg.norm(direction)
eps=1e-5;fd=(calc(z+eps*direction)[0]-calc(z-eps*direction)[0])/(2*eps)
analytic=float(calc(z)[1]@direction)
assert abs(fd-analytic)<1e-4*max(1,abs(fd)),(fd,analytic)
t=time.monotonic()
res=minimize(calc,z,jac=True,method="L-BFGS-B",bounds=[(-.3,.3)]*len(z),
 options=dict(maxiter=4000,maxls=40,ftol=1e-14,gtol=1e-6,maxcor=20))
P=P0.copy();P[active]+=res.x.reshape(-1,3)
# Exact Float32 emission before acceptance, same as native source cooking.
P=(P/1000).astype(np.float32).astype(float)*1000
q,cond,det=measure(P)
report=dict(scope="Offline bounded inferred inner-layer correction candidate only; not admitted",
 source_coordinates_sha256=hashlib.sha256((src/"admitted-layer-coordinates.csv").read_bytes()).hexdigest(),
 target_normalized_determinant=target,max_component_displacement_mm=.3,variable_nodes=len(active),
 iterations=res.nit,solver_message=str(res.message),elapsed_s=time.monotonic()-t,
 gradient_check=dict(finite_difference=fd,analytic=analytic),
 before=dict(min_q=float(q0.min()),max_condition=float(cond0.max()),above_1000=int((cond0>1000).sum())),
 after=dict(min_q=float(q.min()),max_condition=float(cond.max()),above_1000=int((cond>1000).sum())),
 max_displacement_mm=float(np.linalg.norm(P-P0,axis=1).max()),
 outer_unchanged=bool(np.array_equal(P[:n],P0[:n])),
 positive=bool(np.all(q>0)),below_target=int((q<target).sum()))
np.save(OUT/"candidate-points-mm.npy",P)
(OUT/"quality.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2),flush=True)

