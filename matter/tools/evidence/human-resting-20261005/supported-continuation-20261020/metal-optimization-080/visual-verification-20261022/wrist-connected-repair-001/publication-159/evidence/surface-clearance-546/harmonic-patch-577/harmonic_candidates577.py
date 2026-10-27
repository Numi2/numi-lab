"""Harmonic disk charts generate connectivity alternatives on curved/folded patches.
Only temporary 2D proposal coordinates are solved. Body coordinates never change.
Algorithm reference: https://libigl.github.io/tutorial/#harmonic-parametrization
No libigl dependency is introduced; existing SciPy sparse solves are used.
"""
from pathlib import Path
import importlib.util,sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay,QhullError
N=Path(__file__).resolve().parent.parent
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
p571=load("p571_for577",N/"regional-patch-571/regional_candidates571.py")
s569=load("s569_for577",N/"connected-star-569/star_candidates569.py")
def harmonic_chart(points,old,boundary,vertices,weighted=False):
 points=np.asarray(points,dtype=np.float64);vertices=np.asarray(vertices,dtype=np.int64)
 boundary=tuple(map(int,boundary));bset=set(boundary);interior=[int(v)for v in vertices if int(v)not in bset]
 if len(boundary)<3:raise ValueError("closed boundary required")
 adj={int(v):set()for v in vertices}
 for t in old:
  for a,b in zip(t,np.roll(t,-1)):a,b=int(a),int(b);adj[a].add(b);adj[b].add(a)
 lengths=np.linalg.norm(points[np.roll(boundary,-1)]-points[list(boundary)],axis=1)
 if np.any(lengths<=1e-15)or not np.isfinite(lengths).all():raise ValueError("degenerate boundary edge")
 arcs=lengths if weighted else np.ones(len(boundary))
 theta=2*np.pi*np.r_[0,np.cumsum(arcs)[:-1]]/arcs.sum()
 uv={v:np.asarray([np.cos(t),np.sin(t)])for v,t in zip(boundary,theta)}
 if interior:
  index={v:i for i,v in enumerate(interior)};rr=[];cc=[];vv=[];rhs=np.zeros((len(interior),2))
  for v in interior:
   i=index[v];degree=0.
   for w in sorted(adj[v]):
    distance=float(np.linalg.norm(points[v]-points[w]))
    if distance<=1e-15:raise ValueError("coincident patch vertices")
    weight=1./distance if weighted else 1.
    degree+=weight
    if w in index:rr.append(i);cc.append(index[w]);vv.append(-weight)
    else:rhs[i]+=weight*uv[w]
   rr.append(i);cc.append(i);vv.append(degree)
  A=sparse.csc_matrix((vv,(rr,cc)),shape=(len(interior),len(interior)))
  solved=np.asarray(spsolve(A,rhs))
  residual=np.max(np.abs(A@solved-rhs),initial=0.)
  if solved.shape!=rhs.shape or not np.isfinite(solved).all()or residual>1e-8*max(1.,np.max(np.abs(rhs),initial=0.)):raise ValueError("harmonic solve residual")
  uv.update({v:solved[i]for i,v in enumerate(interior)})
 xy=np.asarray([uv[int(v)]for v in vertices]);lookup={int(v):i for i,v in enumerate(vertices)}
 if any(s569.orient(xy[lookup[int(t[0])]],xy[lookup[int(t[1])]],xy[lookup[int(t[2])]])<=1e-13 for t in old):raise ValueError("harmonic chart not positively oriented")
 return xy
def patch_candidates(points,reference,faces,rows,centers=None,max_boundary=48):
 rowids,old,boundary,vertices=p571._boundary_and_vertices(np.asarray(points),np.asarray(faces),rows,max_boundary)
 if np.asarray(reference).shape!=np.asarray(points).shape:raise ValueError("reference shape mismatch")
 oldkey={tuple(sorted(map(int,t)))for t in old};proposals={}
 for geom,weighted in((points,False),(points,True),(reference,True)):
  try:xy=harmonic_chart(geom,old,boundary,vertices,weighted)
  except ValueError:continue
  for scale in(.5,1.,2.):
   for angle in(0.,45.,90.):
    a=np.deg2rad(angle);c,s=np.cos(a),np.sin(a);R=np.asarray([[c,-s],[s,c]]);metric=R@np.diag([scale,1.])@R.T
    try:d=Delaunay(xy@metric.T)
    except QhullError:continue
    if len(d.coplanar):continue
    nf=np.asarray(vertices[d.simplices],dtype=np.asarray(faces).dtype)
    for j,t in enumerate(d.simplices):
     if s569.orient(xy[t[0]],xy[t[1]],xy[t[2]])<0:nf[j,1],nf[j,2]=nf[j,2],nf[j,1]
    if nf.shape!=old.shape or set(map(int,nf.flat))!=set(map(int,vertices)):continue
    key=tuple(sorted(tuple(sorted(map(int,t)))for t in nf))
    if len(set(key))!=len(nf)or set(key)==oldkey:continue
    try:newboundary=p571._boundary_and_vertices(np.asarray(points),nf,np.arange(len(nf)),max_boundary)[2]
    except ValueError:continue
    if tuple(newboundary)!=tuple(boundary):continue
    proposals[key]=nf
 return [proposals[k]for k in sorted(proposals)]
external_edge_conflict=s569.external_edge_conflict

