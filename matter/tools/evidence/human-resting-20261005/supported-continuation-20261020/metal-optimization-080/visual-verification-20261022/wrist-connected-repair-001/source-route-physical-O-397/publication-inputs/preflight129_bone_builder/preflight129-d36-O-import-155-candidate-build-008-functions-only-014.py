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
OUT=Path("/tmp/preflight129-d36-O-import-155-candidate-build-008")
SHELLDIR=P/"generated-122c-source-shell-cap-preflight"
POSE=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/native-app-main-header-integration-151/source-preflight-005/registered-skin-poses.json")
AUDIT=E/"joint-region-roi-014/audit_exact_joint_region_intersections_015.py"
BINARY=E/"arm-local-pocket-remesh-116/local-remesh-118/tetgen-v1.6.0/tetgen"
MAN=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4/bodyparts3d-myosim-major-bones.manifest.json")
SOURCE114=P/"generated-114-bilateral-source-oriented-cap-refinement/candidate-114-bilateral-source-oriented-cap-refinement.npz"
TARGET={"skin_m3":{"left":0.0003310,"right":0.0003491},
        "fat_m3":{"left":0.0011475,"right":0.0011945}}
CUT_BONE_Z=0.930
MAX_TET_VOL=1.5e-7

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
  "bone_unsigned_distance_m":db,"bone_signed_distance_m":sdb,
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
def tri_tet_sat(tri,tet,tol=1e-9):
 edges_tri=[tri[1]-tri[0],tri[2]-tri[1],tri[0]-tri[2]]
 edges_tet=[tet[j]-tet[i] for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))]
 axes=[np.cross(edges_tri[0],edges_tri[1])]
 for i,j,k in ((1,2,3),(0,3,2),(0,1,3),(0,2,1)):
  axes.append(np.cross(tet[j]-tet[i],tet[k]-tet[i]))
 axes.extend(np.cross(a,b) for a in edges_tri for b in edges_tet)
 for axis in axes:
  n=np.linalg.norm(axis)
  if n<=1e-18:continue
  axis=axis/n;pa=tri@axis;pb=tet@axis
  if float(pa.max()) < float(pb.min())-tol or float(pb.max()) < float(pa.min())-tol:return False
 return True
def intersected_tets(X,T,BV,BF,chunk=256):
 TV=X[T].astype(float);tc=TV.mean(axis=1);tr=np.max(np.linalg.norm(TV-tc[:,None,:],axis=2),axis=1)
 tree=cKDTree(tc);tri=BV[BF].astype(float);cent=tri.mean(axis=1);rr=np.max(np.linalg.norm(tri-cent[:,None,:],axis=2),axis=1)
 hit=np.zeros(len(T),bool);total_candidates=0;aabb_candidates=0;tested=0
 tlo=TV.min(axis=1);thi=TV.max(axis=1);blo=tri.min(axis=1);bhi=tri.max(axis=1);rmax=float(tr.max())
 for lo in range(0,len(BF),chunk):
  hi=min(len(BF),lo+chunk)
  lists=tree.query_ball_point(cent[lo:hi],rr[lo:hi]+rmax)
  for off,cand in enumerate(lists):
   fi=lo+off
   if not len(cand):continue
   ids=np.asarray(cand,np.int64);total_candidates+=len(ids)
   m=np.all(thi[ids]>=blo[fi],axis=1)&np.all(tlo[ids]<=bhi[fi],axis=1)
   ids=ids[m];aabb_candidates+=len(ids)
   for ti in ids:
    tested+=1
    if tri_tet_sat(tri[fi],TV[ti]):
     hit[ti]=True
 return hit,{"bone_triangles":len(BF),"tetrahedra":len(T),"sphere_candidates":total_candidates,"aabb_candidates":aabb_candidates,"sat_tests":tested,"intersected_tetrahedra":int(hit.sum()),"max_tet_radius_mm":rmax*1000}
def main():
 print("inputHashes",json.dumps({str(p):sha(p) for p in [BINARY,MAN,POSE,SOURCE114,SHELLDIR/"left-source-shell.npz",SHELLDIR/"right-source-shell.npz"]}),flush=True)
 for side in ("left","right"):
  print("START",side,flush=True)
  raw,OV,OF,used,seed,outervol,X,T,mesh=make_outer_poly(side)
  bones=build_bones(side,OV)
  phi,fd=phi_from_distance(X,OV,OF,bones)
  BV=[];BF=[];off=0
  for name,(v,f) in sorted(bones.items()):
   BV.append(v);BF.append(f+off);off+=len(v)
  BV=np.vstack(BV);BF=np.vstack(BF)
  surface_tet,diag=intersected_tets(X,T,BV,BF)
  inside_tet=np.any(fd["bone_signed_distance_m"][T] <= 0.0,axis=1)
  barrier_tet=surface_tet|inside_tet
  barrier_nodes=np.unique(T[barrier_tet])
  xyzmap={tuple(map(float,p)):i for i,p in enumerate(X.astype(np.float32))}
  boundary=[]
  for p in OV.astype(np.float32):
   key=tuple(map(float,p))
   if key not in xyzmap:raise ValueError(f"{side} TetGen omitted input boundary point")
   boundary.append(xyzmap[key])
  boundary=np.asarray(boundary,np.int64);boundary_hit=np.intersect1d(boundary,barrier_nodes)
  report={"side":side,"outer_volume_m3":outervol,"tet_mesh":mesh,"exact_sat_diagnostics":diag,
    "bone_groups":{k:{"vertices":len(v),"faces":len(f),"volume_m3":abs(signed_volume(v,f))} for k,(v,f) in bones.items()},
    "inside_by_nearest_face_normal_tetrahedra":int(inside_tet.sum()),
    "exact_obstacle_barrier":"All four TetGen vertices of every tetra intersected by any registered bone triangle, plus every tetra with a vertex inside by oriented nearest-face sign, must be assigned phi=1. Exact SAT cell count is a lower bound for true obstacle layer; inside test uses current oriented-nearest-face sign.",
    "barrier_tetrahedra":int(barrier_tet.sum()),"barrier_node_count":int(len(barrier_nodes)),
    "source_outer_boundary_vertices":int(len(boundary)),"boundary_vertices_in_barrier_tets":int(len(boundary_hit)),
    "minimum_boundary_clearance_to_bone_mm":float(fd["bone_unsigned_distance_m"][boundary].min()*1000)}
  (OUT/f"{side}-exact-obstacle-preflight.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
  print("EXACT_OBSTACLE_PREFLIGHT",json.dumps(report,sort_keys=True),flush=True)
 print("PREFLIGHT_DIR="+str(OUT),flush=True)
if __name__=="__main__":main()
