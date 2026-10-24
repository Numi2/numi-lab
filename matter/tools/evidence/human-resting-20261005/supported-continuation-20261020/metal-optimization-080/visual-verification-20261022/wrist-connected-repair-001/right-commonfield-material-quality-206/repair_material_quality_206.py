#!/usr/bin/env python3
"""Bounded multi-material bistellar repair; fixed coordinates/material boundaries."""
from pathlib import Path
import numpy as np,json,time,itertools,collections,hashlib
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001');S=E/'right-commonfield-material-volume-205';O=E/'right-commonfield-material-quality-206';O.mkdir(exist_ok=False)
z=np.load(S/'right-material-volume-203.npz');V=z['points_world_f64_m'];X=V.astype(np.float32).astype(float);T=z['tetrahedra'].copy();M=z['material_codes'].copy();faceidx=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]]);edges=list(itertools.combinations(range(4),2));start=time.monotonic();history=[]
def metric(t):
 p=X[np.asarray(t)];d=np.einsum('ij,ij->i',p[:,1]-p[:,0],np.cross(p[:,2]-p[:,0],p[:,3]-p[:,0]));L=np.max(np.stack([np.linalg.norm(p[:,a]-p[:,b],axis=1) for a,b in edges]),axis=0);return d,np.abs(d)/L**3

def boundary(t):
 c=collections.Counter(tuple(sorted(map(int,f))) for f in np.asarray(t)[:,faceidx].reshape(-1,3));return collections.Counter({f:n for f,n in c.items() if n!=2})

def orient(t):
 a=np.array(t,np.int64);d,q=metric(a);rev=d<0;a[rev,0],a[rev,1]=a[rev,1].copy(),a[rev,0].copy();return a

for rnd in range(4):
 d,q=metric(T);bad=np.flatnonzero((q<1e-5)&np.isin(M,[1,2]));print('round',rnd,'bad',len(bad),'min',q[bad].min() if len(bad) else q.min(),flush=True)
 if not len(bad):break
 nodes=np.unique(T[bad]);near=np.flatnonzero(np.isin(T,nodes).any(1));F=collections.defaultdict(list);G=collections.defaultdict(list)
 for row in near:
  t=T[row]
  for f in t[faceidx]:F[tuple(sorted(map(int,f)))].append(int(row))
  for a,b in edges:G[tuple(sorted((int(t[a]),int(t[b]))))].append(int(row))
 allkeys={tuple(sorted(map(int,t))) for t in T};candidates=[];seen=set()
 def consider(kind,old,new):
  old=tuple(sorted(old));key=(kind,old,tuple(sorted(tuple(sorted(t)) for t in new)))
  if key in seen:return
  seen.add(key)
  if len(set(M[list(old)]))!=1:return
  new=orient(new); nd,nq=metric(new);oldq=q[list(old)]
  if np.any(nd<=0) or nq.min()<=oldq.min()*(1+1e-6):return
  # Preserve the complete cavity boundary and volume, and reject duplicate cells.
  if boundary(T[list(old)])!=boundary(new):return
  if abs(nd.sum()-d[list(old)].sum())>max(1e-25,abs(d[list(old)].sum())*1e-9):return
  oldkeys={tuple(sorted(map(int,t))) for t in T[list(old)]};newkeys=[tuple(sorted(map(int,t))) for t in new]
  if len(set(newkeys))!=len(newkeys) or any(k in allkeys and k not in oldkeys for k in newkeys):return
  score=(int((oldq<1e-5).sum())-int((nq<1e-5).sum()),float(nq.min()/oldq.min()))
  if score[0]<0:return
  candidates.append((score,kind,old,new,int(M[old[0]]),oldq,nq))
 for row in bad:
  t=T[row]
  for f0 in t[faceidx]:
   f=tuple(sorted(map(int,f0)));own=F[f]
   if len(own)!=2:continue
   u=[next(int(a) for a in T[r] if a not in f) for r in own]
   if tuple(sorted(u)) in G:continue
   consider('2-3',own,[(u[0],u[1],f[a],f[b]) for a,b in [(0,1),(1,2),(2,0)]])
  for a,b in edges:
   edge=tuple(sorted((int(t[a]),int(t[b]))));own=G[edge];ring=sorted(set(int(n) for r in own for n in T[r])-set(edge))
   if len(own)==3 and len(ring)==3:consider('3-2',own,[(*ring,edge[0]),(*ring,edge[1])])
   if len(own)==4 and len(ring)==4:
    ringedges={tuple(sorted(set(map(int,T[r]))-set(edge))) for r in own}
    cycle=[ring[0]]
    for k in range(3):
     nxt=sorted(v for v in ring if v not in cycle and tuple(sorted((cycle[-1],v))) in ringedges)
     if not nxt:break
     cycle.append(nxt[0])
    if len(cycle)!=4 or tuple(sorted((cycle[3],cycle[0]))) not in ringedges:continue
    for i in range(2):
     c=cycle[i:]+cycle[:i];u,v=c[0],c[2];newedge=tuple(sorted((u,v)))
     if newedge in G:continue
     cyc=[edge[0],c[1],edge[1],c[3]]
     consider('4-4',own,[(u,v,cyc[j],cyc[(j+1)%4]) for j in range(4)])
 removed=set();app=[];mat=[];nr=0
 for score,kind,old,new,attr,oq,nq in sorted(candidates,key=lambda c:(c[0],c[1],c[2]),reverse=True):
  if removed.intersection(old):continue
  removed.update(old);app.append(new);mat.extend([attr]*len(new));nr+=1;history.append({'round':rnd,'kind':kind,'old_rows':list(old),'new_tets':new.tolist(),'old_q':oq.tolist(),'new_q':nq.tolist(),'material':attr})
 print('candidates',len(candidates),'accepted_disjoint',nr,flush=True)
 if not removed:break
 keep=np.ones(len(T),bool);keep[list(removed)]=False;T=np.vstack([T[keep],*app]);M=np.r_[M[keep],mat]
d,q=metric(T);active=np.isin(M,[1,2]);report={'physical_admission':False,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((S/'right-material-volume-203.npz').read_bytes()).hexdigest(),'coordinate_bytes_unchanged':True,'round_limit':4,'flips':history,'bad_active_cells':int((active&(q<1e-5)).sum()),'nonpositive_cells':int((d<=0).sum()),'minimum_active_q':float(q[active].min()),'wall_seconds':time.monotonic()-start,'material_volumes_m3':{str(a):float(d[M==a].sum()/6) for a in np.unique(M)}}
np.savez_compressed(O/'right-material-volume-repaired-unadmitted.npz',points_world_f64_m=V,tetrahedra=T,material_codes=M,quality=q,signed_volume_m3=d/6)
(O/'report-206.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='flips'}),flush=True)
