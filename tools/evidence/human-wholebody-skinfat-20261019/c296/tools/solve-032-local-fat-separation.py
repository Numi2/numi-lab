#!/usr/bin/env python3
import csv, json, math, pathlib, hashlib, time
import numpy as np
from scipy.optimize import minimize

base=pathlib.Path("/Users/numi/numi-human-separation-20261018/C/wholebody-skin-layer-fitting-296")
prev=base/"column-direction-trial-001/direct-coordinate-repair-025/solve-031"
out=base/"column-direction-trial-001/direct-coordinate-repair-025/solve-032-local-fat-separation"
out.mkdir(parents=True,exist_ok=True)
txt=base/"infeasible-cone-remesh-024/candidate-024-safe/candidate.txt"
reg=base/"infeasible-cone-remesh-024/candidate-024-safe/registered-skin-mechanical.csv"
srcRows=list(csv.DictReader(open(txt.parent/"registered-skin-mechanical.csv",newline="")))
sourceById={int(r["vertex"]):r for r in srcRows}
raw=list(map(int,txt.read_text().split()))
n,nf=raw[:2]; off=2
source=np.array(raw[off:off+n],dtype=np.int64);off+=n
Fraw=np.array(raw[off:off+3*nf],dtype=np.int32).reshape(nf,3);off+=3*nf
sourceFace=np.array(raw[off:off+nf],dtype=np.int64)
F=np.sort(Fraw,axis=1)
assert len(source)==2332 and len(F)==4619 and off+nf==len(raw)
rows=list(csv.DictReader(open(prev/"admitted-layer-coordinates.csv",newline="")))
assert len(rows)==n and [int(r["source_vertex"]) for r in rows]==source.tolist()
def vecs(prefix):
 return np.array([[float(r[prefix+"_x"]),float(r[prefix+"_y"]),float(r[prefix+"_z"])] for r in rows],dtype=np.float64)
O=vecs("outer");S=vecs("skin");T0=vecs("fat")
D=np.array([[float(r["outward_x"]),float(r["outward_y"]),float(r["outward_z"])] for r in rows])
D/=np.linalg.norm(D,axis=1)[:,None]
sd=np.array([float(r["skin_m"]) for r in rows]);fd0=np.array([float(r["fat_m"]) for r in rows])
# Four exact Float32 self-cross witnesses from the admission audit.
pairs=[(144,137),(144,120),(137,143),(120,143)]
axisRows=[]
def cross(a,b):return np.cross(a,b)
def unit(a):
 q=np.linalg.norm(a)
 return None if q<1e-16 else a/q
for fa,fb in pairs:
 ia=Fraw[fa].astype(int);ib=Fraw[fb].astype(int)
 A=T0[ia];B=T0[ib]
 ea=[A[(i+1)%3]-A[i] for i in range(3)]
 eb=[B[(i+1)%3]-B[i] for i in range(3)]
 axes=[cross(ea[0],ea[1]),cross(eb[0],eb[1])]+[cross(x,y) for x in ea for y in eb]
 best=None
 for ai,ax in enumerate(axes):
  u=unit(ax)
  if u is None:continue
  ap=A@u;bp=B@u;a0,a1=float(ap.min()),float(ap.max());b0,b1=float(bp.min()),float(bp.max())
  if a1<b0:
   continue
  if b1<a0:
   continue
  da=a1-b0;db=b1-a0
  if da<=db: depth=da;v=u
  else: depth=db;v=-u
  if best is None or depth<best[0]:best=(depth,ai,v,a0,a1,b0,b1)
 if best is None:raise RuntimeError(f"no overlapping SAT witness for {fa},{fb}")
 depth,ai,v,a0,a1,b0,b1=best
 minproj=min(float((B[j]-A[i])@v) for i in range(3) for j in range(3))
 axisRows.append((fa,fb,ia,ib,v,depth,minproj))
# Only eight vertices touched by the compact patch.
varSources=sorted({int(source[i]) for row in axisRows for tri in (row[2],row[3]) for i in tri})
var=np.array([int(np.where(source==sid)[0][0]) for sid in varSources],dtype=np.int32)
vmap={int(ix):j for j,ix in enumerate(var)}
assert len(var)==8
S0=S.copy();T=T0.copy(); z0=np.zeros((len(var),3),dtype=np.float64)
# All fat-prism tets incident to one of these eight vertices are protected.
incident=np.flatnonzero(np.any(np.isin(F,var),axis=1))
records=[]
for fi in incident:
 a,b,c=map(int,F[fi])
 nrm=np.cross(O[b]-O[a],O[c]-O[a])
 expected=1 if float(nrm@(-D[c]))>0 else -1
 area=float(np.linalg.norm(np.cross(S0[b]-S0[a],S0[c]-S0[a]))*.5)
 scale=max(area*.0005/6.0,1e-25)
 records.extend([((1,a),(1,b),(1,c),(2,c),expected,scale,fi,0),
                 ((1,a),(1,b),(2,b),(2,c),-expected,scale,fi,1),
                 ((1,a),(2,a),(2,b),(2,c),expected,scale,fi,2)])
# fixed pair constraints require the entire convex projection intervals to separate.
margin=5e-5
sepScale=1e-4
sepWeight=100.0
def calc(z):
 T=T0.copy();T[var]=T0[var]+z.reshape(-1,3)*1e-3
 g=np.zeros_like(z).reshape(-1,3); E=0.0
 disp=(T[var]-T0[var])
 sigma=.0005
 E+=.5*np.sum((disp/sigma)**2)
 g+=disp/(sigma*sigma)*1e-3
 # Keep the declared Euclidean and inward thickness bounds at the moved nodes.
 for ix,j in vmap.items():
  dv=S0[ix]-T[ix];length=max(np.linalg.norm(dv),1e-15)
  low=min(length-.0005,0.0);high=max(length-.010,0.0)
  if low or high:
   E+=.5*1e5*((low/.001)**2+(high/.001)**2)
   dE=1e5*(low+high)/(.001**2)
   g[j]+=(-dv/length)*dE*1e-3
  proj=float(dv@D[ix]);pl=min(proj-.0005,0.0);ph=max(proj-.010,0.0)
  if pl or ph:
   E+=.5*1e5*((pl/.001)**2+(ph/.001)**2)
   dP=1e5*(pl+ph)/(.001**2)
   g[j]+=-D[ix]*dP*1e-3
 # Exact separating-axis inequalities for all 9 vertex pairs of each witnessed face pair.
 for fa,fb,ia,ib,v,depth,minproj in axisRows:
  for i in ia:
   for jx in ib:
    d=float((T[jx]-T[i])@v)
    gap=min(d-margin,0.0)
    if gap:
     E+=.5*sepWeight*(gap/sepScale)**2
     gradD=sepWeight*gap/(sepScale**2)
     if int(i) in vmap:g[vmap[int(i)]]+=v*gradD*1e-3
     if int(jx) in vmap:g[vmap[int(jx)]]-=v*gradD*1e-3
 # Source-oriented signed-prism barrier on every fat prism incident to the patch.
 for (la,a),(lb,b),(lc,c),(ld,d),sg,scale,fi,tidx in records:
  p0=S0[a] if la==1 else T[a];p1=S0[b] if lb==1 else T[b];p2=S0[c] if lc==1 else T[c];p3=S0[d] if ld==1 else T[d]
  vol=float(np.dot(p1-p0,np.cross(p2-p0,p3-p0))/6.0)
  q=sg*vol/scale
  gap=min(q-.001,0.0)
  if not gap:continue
  E+=.5*2e5*gap*gap
  cg=2e5*gap*sg/scale
  ga=-(np.cross(p2-p0,p3-p0)+np.cross(p3-p0,p1-p0)+np.cross(p1-p0,p2-p0))/6.0
  gb=np.cross(p2-p0,p3-p0)/6.0
  gc=np.cross(p3-p0,p1-p0)/6.0
  gd=np.cross(p1-p0,p2-p0)/6.0
  for layer,ix,gv in ((la,a,ga),(lb,b,gb),(lc,c,gc),(ld,d,gd)):
   if layer==2 and ix in vmap:g[vmap[ix]]+=cg*gv*1e-3
 return float(E),g.ravel()
bounds=[(-.5,.5)]*(3*len(var))
t0=time.time()
res=minimize(calc,z0.ravel(),jac=True,method="L-BFGS-B",bounds=bounds,
             options={"maxiter":600,"maxfun":5000,"maxls":40,"ftol":1e-14,"gtol":1e-8,"maxcor":20})
T[var]=T0[var]+res.x.reshape(-1,3)*1e-3
# Emit exact Float32 world coords; depth and local coords derive from those emitted values.
O32=O.astype(np.float32).astype(np.float64);S32=S0.astype(np.float32).astype(np.float64);T32=T.astype(np.float32).astype(np.float64)
newfd=np.linalg.norm(S32-T32,axis=1)
# Reconstruct donor-local coordinates from original full registered payload rows.
SL=[];TL=[]
for i,sid in enumerate(source):
 r=sourceById[int(sid)]
 L0=np.array([float(r[k]) for k in ("lx","ly","lz")])
 M=np.array([[float(r[k]) for k in ("m00","m01","m02")],[float(r[k]) for k in ("m10","m11","m12")],[float(r[k]) for k in ("m20","m21","m22")]])
 SL.append(L0+M@(S32[i]-O32[i]));TL.append(L0+M@(T32[i]-O32[i]))
SL=np.array(SL);TL=np.array(TL)
columns=["source_vertex","skin_m","fat_m","outer_x","outer_y","outer_z","skin_x","skin_y","skin_z","fat_x","fat_y","fat_z","skin_local_x","skin_local_y","skin_local_z","fat_local_x","fat_local_y","fat_local_z","outward_x","outward_y","outward_z"]
csvout=out/"admitted-layer-coordinates.csv"
with open(csvout,"w",newline="") as f:
 f.write(",".join(columns)+"\n")
 for i,sid in enumerate(source):
  vals=[int(sid),sd[i],newfd[i],*O32[i],*S32[i],*T32[i],*SL[i],*TL[i],*D[i]]
  f.write(",".join(str(v) if isinstance(v,int) else format(float(v),".17g") for v in vals)+"\n")
# Summary with exact separating constraints and local positive-prism witness set.
def tet(a,b,c,d):return float(np.dot(b-a,np.cross(c-a,d-a))/6.0)
maxMove=float(np.max(np.linalg.norm(T32[var]-T0[var],axis=1)))
sepInfo=[]
for fa,fb,ia,ib,v,depth,minproj in axisRows:
 dvals=[float((T32[j]-T32[i])@v) for i in ia for j in ib]
 sepInfo.append({"faces":[fa,fb],"source_faces":[int(sourceFace[fa]),int(sourceFace[fb])],"axis":[float(x) for x in v],"initial_MTD_m":depth,"initial_min_vertex_projection_m":minproj,"final_min_projection_gap_m":min(dvals),"required_m":margin})
# calculate exact F32 fat signed-tet quality for local protected records
badT=0;minV=float("inf");minQ=float("inf")
for (la,a),(lb,b),(lc,c),(ld,d),sg,scale,fi,tidx in records:
 pts=[S32[a] if la==1 else T32[a],S32[b] if lb==1 else T32[b],S32[c] if lc==1 else T32[c],S32[d] if ld==1 else T32[d]]
 v=sg*tet(*pts);minV=min(minV,v);minQ=min(minQ,v/scale)
 if v<=0:badT+=1
receipt={"scope":"targeted donor-49 four-pair fat-inner self-intersection separation; outer and skin-inner fixed; only eight fat-boundary nodes variable; not admitted pending global exact Float32 audits","input_coordinates_sha256":hashlib.sha256((prev/"admitted-layer-coordinates.csv").read_bytes()).hexdigest(),"candidate_txt_sha256":hashlib.sha256(txt.read_bytes()).hexdigest(),"pairs":sepInfo,"variable_source_vertices":varSources,"margin_m":margin,"optimizer":"L-BFGS-B analytic projection-separation, declared-depth and local source-oriented prism barriers","optimizer_success":bool(res.success),"optimizer_message":str(res.message),"iterations":int(res.nit),"objective":float(res.fun),"elapsed_s":time.time()-t0,"max_fat_node_move_m":maxMove,"fat_depth_minmax_m":[float(newfd.min()),float(newfd.max())],"incident_faces":int(len(incident)),"protected_tets":int(len(records)),"local_bad_source_oriented_tets_f32":int(badT),"local_min_source_oriented_tet_m3":float(minV),"local_min_normalized_tet_quality":float(minQ),"output_csv_sha256":hashlib.sha256(csvout.read_bytes()).hexdigest(),"admitted":False}
(out/"result.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
