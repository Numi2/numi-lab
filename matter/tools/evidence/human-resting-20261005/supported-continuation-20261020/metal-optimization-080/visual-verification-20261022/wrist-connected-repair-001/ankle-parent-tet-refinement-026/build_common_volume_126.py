#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, math, os, runpy, shutil, struct, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
E=ROOT/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
P=E/"ankle-parent-tet-refinement-026"
OUT=P/"generated-126-common-volume-partition-001"
SHELLDIR=P/"generated-122c-source-shell-cap-preflight"
POSE=E/"native-registered-skin-pose-clean-014/construction-neutral-014/registered-skin-poses.json"
AUDIT=E/"joint-region-roi-014/audit_exact_joint_region_intersections_015.py"
BINARY=E/"arm-local-pocket-remesh-116/local-remesh-118/tetgen-v1.6.0/tetgen"
MAN=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4/bodyparts3d-myosim-major-bones.manifest.json")
SOURCE114=P/"generated-114-bilateral-source-oriented-cap-refinement/candidate-114-bilateral-source-oriented-cap-refinement.npz"
TARGET={"skin_m3":{"left":0.0003310,"right":0.0003491},
        "fat_m3":{"left":0.0011475,"right":0.0011945}}
CUT_BONE_Z=0.930
MAX_TET_VOL=1.5e-7
if OUT.exists(): raise SystemExit(f"refusing to overwrite {OUT}")
OUT.mkdir(parents=True)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def mat(q):
 x,y,z,w=map(float,q);s=x*x+y*y+z*z+w*w
 if not math.isfinite(s) or s<=1e-24: raise ValueError("bad quaternion")
 x,y,z,w=x/math.sqrt(s),y/math.sqrt(s),z/math.sqrt(s),w/math.sqrt(s)
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
  [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
  [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],float)
def face_edge_hist(F):
 ed=np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]));ed=np.sort(ed,axis=1)
 return Counter(Counter(map(tuple,ed)).values())
def exact_weld(V,F):
 V=np.asarray(V,dtype=np.float32);F=np.asarray(F,dtype=np.int64)
 mp={};qv=[];r=np.empty(len(V),np.int64)
 for i,p in enumerate(V):
  k=tuple(0.0 if float(x)==0.0 else float(x) for x in p)
  j=mp.get(k)
  if j is None:j=len(qv);mp[k]=j;qv.append(k)
  r[i]=j
 return np.asarray(qv,np.float32),r[F]
def signed_volume(V,F):
 p=V[F[:,0]].astype(float);q=V[F[:,1]].astype(float);r=V[F[:,2]].astype(float)
 return float(np.einsum("ij,ij->i",p,np.cross(q,r)).sum()/6.)
def volume_centroid(V,F):
 p=V[F[:,0]].astype(float);q=V[F[:,1]].astype(float);r=V[F[:,2]].astype(float)
 vv=np.einsum("ij,ij->i",p,np.cross(q,r))/6.;vol=float(vv.sum())
 if abs(vol)<1e-18:raise ValueError("degenerate closed surface volume")
 return ((p+q+r)/4.0*vv[:,None]).sum(0)/vol,vol
def winding(p,V,F):
 a=V[F[:,0]].astype(float)-p;b=V[F[:,1]].astype(float)-p;c=V[F[:,2]].astype(float)-p
 la=np.linalg.norm(a,axis=1);lb=np.linalg.norm(b,axis=1);lc=np.linalg.norm(c,axis=1)
 num=np.einsum("ij,ij->i",a,np.cross(b,c))
 den=la*lb*lc+np.einsum("ij,ij->i",a,b)*lc+np.einsum("ij,ij->i",b,c)*la+np.einsum("ij,ij->i",c,a)*lb
 return float(np.arctan2(num,den).sum()*2/(4*np.pi))
def poly_write(path,V,F,holes=None,regions=None,region_maxvol=MAX_TET_VOL):
 V=np.asarray(V,float);F=np.asarray(F,np.int64);holes=holes or [];regions=regions or []
 with Path(path).open("x") as f:
  f.write(f"{len(V)} 3 0 1\n")
  for i,p in enumerate(V,1):f.write(f"{i} {p[0]:.17g} {p[1]:.17g} {p[2]:.17g} 1\n")
  f.write(f"{len(F)} 1\n")
  for tri in F:f.write(f"1 0 1\n3 {int(tri[0])+1} {int(tri[1])+1} {int(tri[2])+1}\n")
  f.write(f"{len(holes)}\n")
  for i,p in enumerate(holes,1):f.write(f"{i} {p[0]:.17g} {p[1]:.17g} {p[2]:.17g}\n")
  f.write(f"{len(regions)}\n")
  for i,(p,attr,mv) in enumerate(regions,1):
   f.write(f"{i} {p[0]:.17g} {p[1]:.17g} {p[2]:.17g} {attr} {mv if mv is not None else region_maxvol:.17g}\n")
def parse_tetgen(base):
 def lines(p):return [x.split("#",1)[0].strip() for x in p.read_text().splitlines() if x.split("#",1)[0].strip()]
 nl=lines(Path(str(base)+".1.node"));h=nl[0].split();n=int(h[0])
 if int(h[1])!=3:raise ValueError("node dimension")
 nodes=np.empty((n,3),np.float64)
 for line in nl[1:n+1]:
  q=line.split();nodes[int(q[0])-1]=[float(q[1]),float(q[2]),float(q[3])]
 el=lines(Path(str(base)+".1.ele"));h=el[0].split();ne=int(h[0])
 if int(h[1])!=4:raise ValueError("tet corner count")
 T=np.empty((ne,4),np.int64)
 for line in el[1:ne+1]:
  q=line.split();T[int(q[0])-1]=[int(x)-1 for x in q[1:5]]
 return nodes,T
def tet_metrics(V,T):
 a,b,c,d=(V[T[:,i]] for i in range(4));det=np.einsum("ij,ij->i",b-a,np.cross(c-a,d-a))
 edge=np.stack([np.linalg.norm(V[T[:,i]]-V[T[:,j]],axis=1) for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))],axis=1).max(1)
 return det/6,np.abs(det)/np.maximum(edge**3,1e-30)
def closest_bary(P,T):
 a,b,c=T[:,:,0,:],T[:,:,1,:],T[:,:,2,:];ap=P[:,None,:]-a;ab=b-a;ac=c-a
 d1=np.einsum("nki,nki->nk",ab,ap);d2=np.einsum("nki,nki->nk",ac,ap)
 N,K=d1.shape;bary=np.zeros((N,K,3),float);assigned=np.zeros((N,K),bool)
 def setm(m,x,y,z):
  bary[:,:,0][m]=x[m];bary[:,:,1][m]=y[m];bary[:,:,2][m]=z[m];assigned[m]=True
 setm((d1<=0)&(d2<=0),np.ones_like(d1),np.zeros_like(d1),np.zeros_like(d1))
 bp=P[:,None,:]-b;d3=np.einsum("nki,nki->nk",ab,bp);d4=np.einsum("nki,nki->nk",ac,bp)
 setm((~assigned)&(d3>=0)&(d4<=d3),np.zeros_like(d1),np.ones_like(d1),np.zeros_like(d1))
 vc=d1*d4-d3*d2;v=np.divide(d1,d1-d3,out=np.zeros_like(d1),where=np.abs(d1-d3)>1e-30)
 setm((~assigned)&(vc<=0)&(d1>=0)&(d3<=0),1-v,v,np.zeros_like(d1))
 cp=P[:,None,:]-c;d5=np.einsum("nki,nki->nk",ab,cp);d6=np.einsum("nki,nki->nk",ac,cp)
 setm((~assigned)&(d6>=0)&(d5<=d6),np.zeros_like(d1),np.zeros_like(d1),np.ones_like(d1))
 vb=d5*d2-d1*d6;w=np.divide(d2,d2-d6,out=np.zeros_like(d1),where=np.abs(d2-d6)>1e-30)
 setm((~assigned)&(vb<=0)&(d2>=0)&(d6<=0),1-w,np.zeros_like(d1),w)
 va=d3*d6-d5*d4;wbc=np.divide(d4-d3,(d4-d3)+(d5-d6),out=np.zeros_like(d1),where=np.abs((d4-d3)+(d5-d6))>1e-30)
 setm((~assigned)&(va<=0)&((d4-d3)>=0)&((d5-d6)>=0),np.zeros_like(d1),1-wbc,wbc)
 den=va+vb+vc;face=~assigned
 vf=np.divide(vb,den,out=np.zeros_like(den),where=np.abs(den)>1e-30);wf=np.divide(vc,den,out=np.zeros_like(den),where=np.abs(den)>1e-30)
 setm(face,1-vf-wf,vf,wf)
 Q=np.einsum("nki,nkij->nkj",bary,T);ds=np.einsum("nki,nki->nk",Q-P[:,None,:],Q-P[:,None,:])
 return bary,ds
def closest_surface(P,V,F,k=32):
 tris=V[F].astype(float);tree=cKDTree(tris.mean(1));n=len(P)
 outd=np.empty(n);outf=np.empty(n,np.int64);outb=np.empty((n,3));outq=np.empty((n,3))
 for lo in range(0,n,256):
  pp=np.asarray(P[lo:lo+256],float);dd,ix=tree.query(pp,k=min(k,len(F)))
  if ix.ndim==1:ix=ix[:,None]
  TT=tris[ix];bary,ds=closest_bary(pp,TT);kk=np.argmin(ds,axis=1);rr=np.arange(len(pp));end=lo+len(pp)
  outd[lo:end]=np.sqrt(ds[rr,kk]);outf[lo:end]=ix[rr,kk];outb[lo:end]=bary[rr,kk]
  outq[lo:end]=np.einsum("ni,nij->nj",outb[lo:end],tris[outf[lo:end]])
 return outd,outf,outb,outq
def extract_level(V,T,phi,level):
 edges=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3));emap={};points=[];pedge=[];et=[];ptet=[];faces=[]
 for ti,tet in enumerate(T):
  vals=phi[tet]
  if float(vals.min())>=level or float(vals.max())<=level:continue
  vids=[]
  for ia,ib in edges:
   va,vb=float(vals[ia]),float(vals[ib])
   if (va<level<vb) or (vb<level<va):
    ga,gb=int(tet[ia]),int(tet[ib]);key=(min(ga,gb),max(ga,gb));ix=emap.get(key)
    if ix is None:
     lo,hi=key;s=float((level-phi[lo])/(phi[hi]-phi[lo]));pt=((1-s)*V[lo]+s*V[hi]).astype(np.float32)
     ix=len(points);emap[key]=ix;points.append(pt);pedge.append(key);et.append(s);ptet.append(ti)
    vids.append(ix)
  vids=list(dict.fromkeys(vids))
  if len(vids)<3:continue
  p=np.asarray([points[i] for i in vids],float);cent=p.mean(0);X=V[tet].astype(float)
  A=np.stack((X[1]-X[0],X[2]-X[0],X[3]-X[0]),axis=0);ds=np.asarray(vals[1:]-vals[0],float)
  try:grad=np.linalg.solve(A,ds)
  except np.linalg.LinAlgError:continue
  if np.linalg.norm(grad)<=1e-16:continue
  n=grad/np.linalg.norm(grad);u=p[0]-cent;un=np.linalg.norm(u)
  if un<=1e-15:continue
  u/=un;v=np.cross(n,u);angles=np.arctan2((p-cent)@v,(p-cent)@u);order=np.argsort(angles);poly=[vids[int(j)] for j in order]
  if len(poly)==3:tris=[tuple(poly)]
  elif len(poly)==4:
   d02=np.linalg.norm(points[poly[0]]-points[poly[2]]);d13=np.linalg.norm(points[poly[1]]-points[poly[3]])
   tris=[(poly[0],poly[1],poly[2]),(poly[0],poly[2],poly[3])] if d02<=d13 else [(poly[0],poly[1],poly[3]),(poly[1],poly[2],poly[3])]
  else:tris=[(poly[0],poly[j],poly[j+1]) for j in range(1,len(poly)-1)]
  for f in tris:
   a,b,c=(points[i].astype(float) for i in f)
   if np.dot(np.cross(b-a,c-a),grad)<0:f=(f[0],f[2],f[1])
   faces.append(f)
 F=np.asarray(faces,np.int64);P=np.asarray(points,np.float32)
 if not len(F):raise ValueError(f"empty level {level}")
 area2=np.linalg.norm(np.cross(P[F[:,1]]-P[F[:,0]],P[F[:,2]]-P[F[:,0]]),axis=1)
 if np.any(area2<=1e-16):raise ValueError(f"F32 degenerate level faces {int(np.sum(area2<=1e-16))}")
 vol=signed_volume(P,F)
 if vol<0:F=F[:,[0,2,1]];vol=-vol
 return {"points":P,"faces":F,"parent_edge":np.asarray(pedge,np.int64),"edge_t":np.asarray(et,float),"parent_tet":np.asarray(ptet,np.int64),"signed_volume_m3":float(vol),"edge_hist":dict(face_edge_hist(F)),"min_double_area2_m2":float(area2.min())}
def make_outer_poly(side):
 raw=np.load(SHELLDIR/f"{side}-source-shell.npz",allow_pickle=False);V0=raw["vertices_world_m"].astype(np.float32);F0=raw["faces"].astype(np.int64)
 used=np.unique(F0);remap=np.full(len(V0),-1,np.int64);remap[used]=np.arange(len(used));V=V0[used];F=remap[F0]
 hist=face_edge_hist(F)
 if hist!={2:sum(hist.values())}:raise ValueError(f"{side} shell not closed {hist}")
 c,vol=volume_centroid(V,F);w=winding(c,V,F)
 if vol<=0 or abs(w)<0.9:raise ValueError(f"{side} volume seed/winding {vol} {w}")
 base=OUT/f"outer-{side}";poly_write(str(base)+".poly",V,F,regions=[(c,1,MAX_TET_VOL)])
 np.savez_compressed(str(base)+"-input-map.npz",source_shell_rows=used,source_shell_faces=F,source_face_ids=raw["source_face_ids"],source_cut_loop=raw["source_cut_loop"])
 t0=time.time();r=subprocess.run([str(BINARY),"-pYq1.2A",str(base)+".poly"],cwd=OUT,text=True,capture_output=True)
 (OUT/f"outer-{side}.stdout").write_text(r.stdout);(OUT/f"outer-{side}.stderr").write_text(r.stderr)
 if r.returncode:raise RuntimeError(f"TetGen outer {side} rc={r.returncode}: {r.stderr[-500:]}")
 X,T=parse_tetgen(base);det,q=tet_metrics(X,T);neg=int(np.sum(det<0));T=T.copy()
 if neg:T[det<0]=T[det<0][:,[0,1,3,2]]
 det,q=tet_metrics(X,T)
 if np.any(det<=0):raise ValueError(f"nonpositive outer volume mesh {side}")
 return raw,V,F,used,c,vol,X,T,{"tetgen_seconds":time.time()-t0,"nodes":len(X),"tets":len(T),"tet_negative_orientation":neg,"min_signed_volume_m3":float(det.min()),"min_normalized_abs_det":float(q.min()),"volume_m3":float(det.sum()),"seed_winding":w}
def build_bones(side,outerV):
 mod=runpy.run_path(str(AUDIT));_,_,records,bv,bi,anchors=mod["load_bones"]()
 snap=json.loads(POSE.read_text());st={int(x["body_index"]):x for x in snap["bodies"]};lo=outerV.min(0).astype(float);hi=outerV.max(0).astype(float);groups=defaultdict(lambda:[[],[]])
 for r,a in zip(records,anchors):
  body,fv,nv,fi,ni,stable,tx,ty,tz,qx,qy,qz,qw,scale,_=r;name=str(a["myosim_body"]).lower()
  local=(bv[fv:fv+nv].astype(float)*scale)@mod["mat"]((qx,qy,qz,qw)).T+np.array((tx,ty,tz),float)
  cur=st[int(body)]["current"];world=(local@mod["mat"](cur["orientation_world_xyzw"]).T+np.asarray(cur["position_world_m"],float)).astype(np.float32)
  if not (np.all(world.max(0)>=lo)&np.all(world.min(0)<=hi)):continue
  f=bi[fi//3:(fi+ni)//3].astype(np.int64)-int(fv);off=sum(len(x) for x in groups[name][0]);groups[name][0].append(world);groups[name][1].append(f+off)
 out={}
 for name,(vv,ff) in groups.items():
  V=np.vstack(vv);F=np.vstack(ff);V,F=exact_weld(V,F);hist=face_edge_hist(F);vol=signed_volume(V,F)
  if hist!={2:sum(hist.values())}:raise ValueError(f"{side}/{name} bone exact-weld not closed {hist}")
  if vol<0:F=F[:,[0,2,1]];vol=-vol
  if vol<=0:raise ValueError(f"{side}/{name} bone volume")
  out[name]=(V,F)
 return out
def bone_distance(P,bones):
 Vb=[];Fb=[];owners=[];off=0
 for name,(V,F) in sorted(bones.items()):
  Vb.append(V.astype(np.float32));Fb.append(F+off);owners.extend([name]*len(F));off+=len(V)
 Vb=np.vstack(Vb);Fb=np.vstack(Fb)
 d,f,b,q=closest_surface(P,Vb,Fb,k=64)
 tri=Vb[Fb[f]].astype(float);n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);nn=np.linalg.norm(n,axis=1)
 if np.any(nn<=1e-16):raise ValueError("zero-area bone face reached nearest query")
 n/=nn[:,None];sd=np.einsum("ij,ij->i",np.asarray(P,float)-q,n)
 owner=np.asarray(owners,dtype=object)[f]
 return d,sd,f,b,q,owner,Vb,Fb
def outer_distance(P,V,F):
 return closest_surface(P,V,F,k=48)
def phi_from_distance(P,V,F,bones):
 do,of,ob,oq=outer_distance(P,V,F);db,sdb,bf,bb,bq,bowner,BV,BF=bone_distance(P,bones)
 phi=np.where(sdb<=0.0,1.0,do/np.maximum(do+np.maximum(sdb,1e-9),1e-9))
 phi=np.clip(phi,0.0,1.0)
 return phi,{"outer_distance_m_quantiles":np.quantile(do,[0,.01,.5,.99,1]).tolist(),
  "bone_signed_distance_m_quantiles":np.quantile(sdb,[0,.01,.5,.99,1]).tolist(),
  "inside_bone_node_count":int(np.sum(sdb<=0)),"nearest_bone_owner_counts":dict(Counter(map(str,bowner))),
  "outer_nearest_face":of,"outer_nearest_bary":ob,"outer_nearest_point":oq,
  "bone_nearest_face":bf,"bone_nearest_bary":bb,"bone_nearest_point":bq,"bone_nearest_owner":bowner,
  "bone_vertices":BV,"bone_faces":BF}
def choose_level(V,T,phi,target_high,lo=1e-6,hi=.9995):
 # Keep the global level strictly between nodal scalar values. A level through
 # a TetGen vertex/edge creates Float32 zero-area facets; do not weld/drop them.
 nodal=np.unique(np.asarray(phi,float))
 bracket={"lo":lo,"hi":hi,"target_high_volume_m3":float(target_high),"iterations":[]}
 best=None
 for it in range(10):
  proposed=(lo+hi)*.5;j=int(np.searchsorted(nodal,proposed))
  candidates=[]
  for k in range(max(0,j-4),min(len(nodal)-1,j+4)):
   a,b=float(nodal[k]),float(nodal[k+1])
   if not (lo<a<b<hi):continue
   candidates.append((abs((a+b)*.5-proposed),-(b-a),(a+b)*.5,a,b))
  if not candidates:raise ValueError(f"no safe nodal-breakpoint interval inside [{lo},{hi}]")
  candidates.sort()
  chosen=None;failures=[]
  for _,_,level,a,b in candidates:
   try:
    surf=extract_level(V,T,phi,level);chosen=(level,surf,a,b);break
   except ValueError as e:
    failures.append({"level":level,"interval":[a,b],"error":str(e)})
  if chosen is None:raise ValueError(f"no nondegenerate Float32 level near {proposed}; failures={failures}")
  level,surf,a,b=chosen;v=surf["signed_volume_m3"]
  bracket["iterations"].append({"proposed_level":proposed,"level":level,"safe_phi_interval":[a,b],
    "interval_width":b-a,"high_volume_m3":v,"residual_m3":v-target_high,"failed_nearer_intervals":failures})
  best=(level,surf)
  if v>target_high:lo=level
  else:hi=level
 bracket["lo_final"]=lo;bracket["hi_final"]=hi
 return best[0],best[1],bracket
def exact_audit_surfaces(side,outer,interfaces,bones):
 mod=runpy.run_path(str(AUDIT))
 sys.path.insert(0,"/Users/n/numi-human-anatomy-finish-1378/src")
 sys.path.insert(0,str(E))
 from numilab_human.compiled_quotient_embeddedness import coordinate_quotient
 from numilab_human.cardiac_cavity_intersections import _aabb_candidate_pairs,triangle_intersection_points,_allowed_shared_point
 sys.path.insert(0,str(E/"joint-region-roi-014"))
 from audit_bilateral_skin_retained_poses_002 import records_nonzero
 injected={"coordinate_quotient":coordinate_quotient,"_aabb_candidate_pairs":_aabb_candidate_pairs,
  "triangle_intersection_points":triangle_intersection_points,"_allowed_shared_point":_allowed_shared_point,"records_nonzero":records_nonzero}
 mod.update(injected)
 # runpy.run_path returns a mapping copy; nested functions keep the executed
 # namespace, so inject predicate dependencies into their real globals too.
 for key in ("exact_scene","exact_pair_audit"):
  mod[key].__globals__.update(injected)
 report={}
 # Bone self intersections and layer-vs-bone / layer-vs-layer use exact Float32 quotient predicates.
 names=sorted(bones);BV=[];BF=[];own=[];off=0
 for name in names:
  v,f=bones[name];BV.append(v);BF.append(f+off);own.extend([name]*len(f));off+=len(v)
 BV=np.vstack(BV);BF=np.vstack(BF)
 bpts,bqf=mod["exact_scene"](BV,BF)
 report["bone_group_union_self"]=mod["exact_pair_audit"](bpts,bqf,np.arange(len(BF),dtype=np.int64),same_surface=True)
 for lname,surf in interfaces.items():
  V=surf["points"];F=surf["faces"];combV=np.vstack((V,BV));combF=np.vstack((F,BF+len(V)))
  pts,qf=mod["exact_scene"](combV,combF);n=len(F)
  report[lname+"_vs_bone_union"]=mod["exact_pair_audit"](pts,qf,np.arange(n,dtype=np.int64),np.arange(n,len(combF),dtype=np.int64),label_a=lambda fi:lname,label_b=lambda fi:own[fi-n])
 # Candidate interface surface self and cross surfaces.
 all_names=list(interfaces);allV=[outer[0]]+[interfaces[k]["points"] for k in all_names];allF=[outer[1]]
 offsets=[0];vbase=len(outer[0]);fbase=len(outer[1])
 for k in all_names:
  allF.append(interfaces[k]["faces"]+vbase);vbase+=len(interfaces[k]["points"])
  offsets.append(fbase);fbase+=len(interfaces[k]["faces"])
 V=np.vstack(allV);F=np.vstack(allF);pts,qf=mod["exact_scene"](V,F)
 ranges={"outer":np.arange(0,len(allF[0]),dtype=np.int64)}
 facebase=len(allF[0])
 for k in all_names:
  ranges[k]=np.arange(facebase,facebase+len(interfaces[k]["faces"]),dtype=np.int64);facebase+=len(interfaces[k]["faces"])
 for a in ranges:
  report[a+"_self"]=mod["exact_pair_audit"](pts,qf,ranges[a],same_surface=True)
 for a,b in (("outer","skin_inner"),("outer","fat_deep"),("skin_inner","fat_deep")):
  report[a+"_vs_"+b]=mod["exact_pair_audit"](pts,qf,ranges[a],ranges[b],label_a=lambda fi:a,label_b=lambda fi:b)
 return report
def main():
 print("inputHashes",json.dumps({str(p):sha(p) for p in [BINARY,MAN,POSE,SOURCE114,SHELLDIR/"left-source-shell.npz",SHELLDIR/"right-source-shell.npz"]}),flush=True)
 result={"schema":"numi.human.ankle.common-volumetric-layer-partition-candidate.v1","candidate_status":"offline_candidate_only_not_native_or_production_admitted",
  "method":"TetGen tetrahedralizes the closed registered source outer shell; one continuous normalized distance scalar is derived from nearest outer surface and all locally overlapping registered bone banks. Ordered P1 level sets define I and D; the scalar field is embedded in one volume mesh, so interfaces cannot cross by independent offsets. Exact all-bank surface checks and second constrained TetGen pass are required before claiming a tissue partition.",
  "inputs":{},"sides":{},"nonclaims":["not native FEM or D36 cavity stitch","not full-body mass-owner transfer","not dynamic acceptance","source shell caps are computational cut boundaries","material volume targets are inferred from the existing 116 removed-parent tet debit split, not a measured anatomical thickness map"]}
 for side in ("left","right"):
  print("START",side,flush=True);raw,OV,OF,used,seed,outervol,X,T,mesh=make_outer_poly(side)
  bones=build_bones(side,OV)
  print("OUTERMESH",side,json.dumps(mesh),flush=True)
  print("BONES",side,json.dumps({k:{"vertices":len(v),"faces":len(f),"signed_volume_m3":signed_volume(v,f)} for k,(v,f) in bones.items()}),flush=True)
  phi,fd=phi_from_distance(X,OV,OF,bones)
  targetSkin=TARGET["skin_m3"][side];targetFat=TARGET["fat_m3"][side]
  targetI=outervol-targetSkin;targetD=targetI-targetFat
  lI,sI,bI=choose_level(X,T,phi,targetI);lD,sD,bD=choose_level(X,T,phi,targetD)
  if not lI<lD:raise ValueError(f"{side} scalar interface order {lI} >= {lD}")
  report={"outer_shell":{"points":len(OV),"faces":len(OF),"volume_m3":outervol,"seed":seed.tolist(),"source_cut":float(np.float32(0.94027102))},
   "initial_tet_mesh":mesh,"local_bone_meshes":{k:{"vertices":len(v),"faces":len(f),"signed_volume_m3":signed_volume(v,f)} for k,(v,f) in bones.items()},
   "field":{k:v for k,v in fd.items() if not isinstance(v,np.ndarray)},
   "target_material_volumes_m3":{"skin":targetSkin,"fat":targetFat,"interior_at_I_m3":targetI,"interior_at_D_m3":targetD},
   "level_choice":{"skin_inner":{"phi":lI,"high_volume_m3":sI["signed_volume_m3"],"bisect":bI},"fat_deep":{"phi":lD,"high_volume_m3":sD["signed_volume_m3"],"bisect":bD},
      "realized_skin_volume_m3":outervol-sI["signed_volume_m3"],"realized_fat_volume_m3":sI["signed_volume_m3"]-sD["signed_volume_m3"]},
   "surface_checks":{}}
  # source ancestry: extracted vertex->parent TetGen edge/t, plus exact nearest registered outer face+bary.
  for lname,surf in [("skin_inner",sI),("fat_deep",sD)]:
   d,fi,bary,qpt=closest_surface(surf["points"],OV,OF,k=64)
   sourceids=raw["source_face_ids"];srcface=np.where(fi<len(sourceids),sourceids[np.minimum(fi,len(sourceids)-1)],-1)
   surf["source_nearest_face"]=fi;surf["source_face_id"]=srcface;surf["source_face_bary"]=bary;surf["source_projection_distance_m"]=d
   surf["source_projection_xyz"]=qpt
   np.savez_compressed(OUT/f"{side}-{lname}-levelset.npz",points=surf["points"],faces=surf["faces"],parent_tet_edge=surf["parent_edge"],edge_t=surf["edge_t"],parent_tet=surf["parent_tet"],
    source_shell_face_index=fi,source_face_id=srcface,source_face_bary=bary,source_projection_distance_m=d,source_projection_xyz=qpt,
    inner_level_phi=np.asarray([lI if lname=="skin_inner" else lD]))
   report["surface_checks"][lname]={"vertices":len(surf["points"]),"faces":len(surf["faces"]),"edge_hist":surf["edge_hist"],"signed_enclosed_volume_m3":surf["signed_volume_m3"],
    "min_area2_m2":surf["min_double_area2_m2"],"source_projection_distance_mm_quantiles":(d*1000).astype(float).tolist() if len(d)<0 else np.quantile(d*1000,[0,.5,.95,.99,1]).tolist()}
  exact=exact_audit_surfaces(side,(OV,OF),{"skin_inner":sI,"fat_deep":sD},bones)
  report["exact_surface_intersections"]={k:v["intersection_pairs"] for k,v in exact.items()}
  report["exact_surface_intersection_receipt"]=exact
  report["clearance_and_nestedness_gate"]={"all_interfaces_closed":all(set(s["edge_hist"])=={2} for s in (sI,sD)),"threshold_order":lI<lD,
    "no_exact_surface_intersections":all(v["intersection_pairs"]==0 for v in exact.values())}
  result["sides"][side]=report
  (OUT/f"{side}-candidate-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
  print("SIDE_REPORT",side,json.dumps({k:report[k] for k in ["target_material_volumes_m3","level_choice","clearance_and_nestedness_gate","exact_surface_intersections"]},sort_keys=True),flush=True)
 result["input_sha256"]={"tetgen":sha(BINARY),"major_bone_manifest":sha(MAN),"native_reference_pose":sha(POSE),"source_shell_preflight":sha(SHELLDIR/"source-shell-preflight.json"),"source_surface_candidate114":sha(SOURCE114)}
 result["candidate_files"]={str(f.name):sha(f) for f in sorted(OUT.iterdir()) if f.is_file()}
 (OUT/"candidate-report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 print("REPORT="+str(OUT/"candidate-report.json"),flush=True)
if __name__=="__main__":main()
