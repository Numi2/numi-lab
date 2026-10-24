#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, math, os, re, runpy, subprocess, sys, time
from collections import Counter
from fractions import Fraction
import numpy as np
from scipy.spatial import cKDTree

E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
P=E/"ankle-parent-tet-refinement-026"
OUT=P/"generated-224-commonfield-tetgen-volume-actual007"
CAND=P/"generated-223-bilateral-source-oriented-cap-actual007-frame/candidate-222.npz"
SHELL=P/"generated-122c-source-shell-cap-preflight"
MESH=P/"generated-129-exact-obstacle-cell-preflight"
BANK=E/"registered-surrounding-tissue-native007-190/registered-tissue-banks.npz"
INV=E/"registered-surrounding-tissue-native007-190/inventory.json"
POSE=E/"native-app-main-header-integration-151/source-preflight-007/registered-skin-poses.json"
PLACEMENT=E/"elbow-refinement-current-main-024-final080-candidate-source-frames-001/native-source-only-024/common-rigid-transfer-row-diagnostic-005.json"
COMMON=P/"build_common_volume_126.py"
AUDIT_SCRIPT=P/"generated-223-bilateral-source-oriented-cap-actual007-frame/audit_candidate_223.py"
TETGEN=E/"arm-local-pocket-remesh-116/local-remesh-118/tetgen-v1.6.0/tetgen"
MAXVOL=1.5e-7
TARGET={"left":{"skin":0.0003310,"fat":0.0011475},"right":{"skin":0.0003491,"fat":0.0011945}}
if OUT.exists():
 allowed={"build_commonfield_224.py","run.log"}
 unexpected=[x.name for x in OUT.iterdir() if x.name not in allowed]
 if unexpected: raise SystemExit(f"refusing to overwrite nonempty output {OUT}: {unexpected}")
else:
 OUT.mkdir(parents=True)
t_start=time.monotonic()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
# Reuse the already tested 126 common-field geometry primitives without its
# output side effect or old main() invocation.
src=COMMON.read_text()
src=src.replace('if OUT.exists(): raise SystemExit(f"refusing to overwrite {OUT}")\nOUT.mkdir(parents=True)\n','')
ns={"__name__":"common_volume_126_geometry_helpers"}
exec(compile(src,str(COMMON),"exec"),ns)
mat=ns["mat"]
face_edge_hist=ns["face_edge_hist"]
signed_volume=ns["signed_volume"]
choose_level=ns["choose_level"]
closest_surface=ns["closest_surface"]
tet_metrics=ns["tet_metrics"]
# Import the same exact predicate used by the frozen 223 audit.
sys.path.insert(0,"/Users/n/numi-human-anatomy-finish-1378/src")
from numilab_human.cardiac_cavity_intersections import (
 _records,_prepare_surface_aabb,_query_surface_aabb,
 _audit_self_pairs_touching_faces,triangle_intersection_points
)
def records(V,F):
 return _records([tuple(Fraction.from_float(float(x)) for x in p) for p in np.asarray(V,float)],np.asarray(F,np.int64).tolist())
def exact_cross(Va,Fa,Vb,Fb,keep=12):
 ra=records(Va,Fa);rb=records(Vb,Fb);idx=_prepare_surface_aabb(rb)
 nquery=0;hits=[]
 for ar in ra:
  for _,br in _query_surface_aabb([ar],idx):
   nquery+=1
   pts=triangle_intersection_points(ar[0],br[0])
   if pts:
    hits.append({"face_a":int(ar[3]),"face_b":int(br[3]),"points":[[float(x) for x in p] for p in pts[:3]]})
 return {"face_a":len(Fa),"face_b":len(Fb),"aabb_candidates":nquery,"intersection_pairs":len(hits),"examples":hits[:keep]}
def exact_self(V,F):
 r=records(V,F)
 return _audit_self_pairs_touching_faces(r,list(range(len(F))))
def consistent_closed(F):
 F=np.asarray(F,np.int64);cnt=Counter();dire=Counter()
 for a,b,c in F:
  for u,v in ((int(a),int(b)),(int(b),int(c)),(int(c),int(a))):
   e=(min(u,v),max(u,v));cnt[e]+=1;dire[e]+=1 if u<v else -1
 return (all(x==2 for x in cnt.values()) and all(dire[e]==0 for e in cnt),{"edges":len(cnt),"incidence_histogram":dict(Counter(cnt.values())),"same_direction_edges":sum(dire[e]!=0 for e in cnt)})
def solid_angle_winding(points,V,F):
 tri=np.asarray(V,np.float64)[np.asarray(F,np.int64)]
 out=np.zeros(len(points),np.float64)
 # Face batches bound memory and permit one point at a time.
 for ip,p in enumerate(np.asarray(points,np.float64)):
  acc=0.0
  for lo in range(0,len(tri),16384):
   t=tri[lo:lo+16384]
   a=t[:,0]-p;b=t[:,1]-p;c=t[:,2]-p
   la=np.linalg.norm(a,axis=1);lb=np.linalg.norm(b,axis=1);lc=np.linalg.norm(c,axis=1)
   num=np.einsum("ij,ij->i",a,np.cross(b,c))
   den=la*lb*lc+np.einsum("ij,ij->i",a,b)*lc+np.einsum("ij,ij->i",b,c)*la+np.einsum("ij,ij->i",c,a)*lb
   acc+=float(np.sum(2*np.arctan2(num,den)))
  out[ip]=acc/(4*math.pi)
 return out
def read_ele_attributes(base):
 p=Path(str(base)+".1.ele");lines=[x.split("#",1)[0].strip() for x in p.read_text().splitlines() if x.split("#",1)[0].strip()]
 h=lines[0].split();n=int(h[0]);nc=int(h[1]);na=int(h[2]) if len(h)>2 else 0
 T=np.empty((n,nc),np.int64);attr=np.zeros(n,np.int64)
 for line in lines[1:n+1]:
  q=line.split();i=int(q[0])-1;T[i]=[int(x)-1 for x in q[1:1+nc]]
  if na: attr[i]=int(float(q[1+nc]))
 return T,attr
def write_multiregion_poly(path,V,surfaces,seeds):
 V=np.asarray(V,np.float64)
 with Path(path).open("x") as f:
  f.write(f"{len(V)} 3 0 1\n")
  for i,p in enumerate(V,1):f.write(f"{i} {p[0]:.17g} {p[1]:.17g} {p[2]:.17g} 1\n")
  nface=sum(len(F) for _,F,_ in surfaces);f.write(f"{nface} 1\n")
  off=0
  for SV,F,marker in surfaces:
   for tri in F:
    f.write(f"1 0 {int(marker)}\n3 {int(tri[0])+off+1} {int(tri[1])+off+1} {int(tri[2])+off+1}\n")
   off+=len(SV)
  f.write("0\n")
  f.write(f"{len(seeds)}\n")
  for i,(p,attr) in enumerate(seeds,1):
   f.write(f"{i} {p[0]:.17g} {p[1]:.17g} {p[2]:.17g} {int(attr)} {MAXVOL:.17g}\n")
def classify_banks(side,outer):
 inventory=json.loads(INV.read_text())
 info=json.loads((INV.parent/"inventory.json").read_text())
 if info["pose_sha256"] != sha(POSE): raise ValueError("registered-bank cache pose hash differs from actual007 pose")
 lo=np.asarray(outer.min(0),float)-1e-6;hi=np.asarray(outer.max(0),float)+1e-6
 z=np.load(BANK,allow_pickle=False)
 selected=[]
 for idx,m in enumerate(inventory["banks"]):
  if m["kind"]!="anatomy":continue
  mn=np.asarray(m["aabb_min_m"],float);mx=np.asarray(m["aabb_max_m"],float)
  if np.all(mx>=lo) and np.all(mn<=hi):
   Vb=z[f"bank_{idx:04d}_vertices"].astype(np.float32)
   Fb=z[f"bank_{idx:04d}_faces"].astype(np.int64)
   closed,top=consistent_closed(Fb)
   top["closed_for_winding"]=bool(closed)
   selected.append((m,Vb,Fb,top))
 print(json.dumps({"stage":"banks","side":side,"count":len(selected),"triangles":sum(len(x[2]) for x in selected),"banks":[{"key":m["key"],"stable_id":m["stable_id"],"body_indices":m["body_indices"],"faces":len(F),"edges":top["edges"]} for m,V,F,top in selected]},sort_keys=True),flush=True)
 return selected
def build_bone_field(X,outerV,outerF,banks):
 BV=[];BF=[];owner=[];off=0
 bank_audits=[]
 for m,V,F,top in banks:
  BV.append(V);BF.append(F+off);owner.extend([m["key"]]*len(F));off+=len(V)
  bank_audits.append((m,V,F))
 BV=np.vstack(BV);BF=np.vstack(BF)
 # Distance is unsigned nearest-triangle distance; solid-angle winding supplies
 # the sign independently for each closed registered anatomy bank.
 db,bi,bary,bq=closest_surface(X,BV,BF,k=64)
 inside=np.zeros(len(X),bool);wind_stats=[]
 # Banks may be open individually because the registered exporter splits one
 # closed anatomical body across component records. Weld exact Float32 seam
 # coordinates within each body-index set, then use winding only if that union
 # is a closed, consistently oriented 2-manifold.
 groups={}
 for m,V,F,top in banks:
  key=tuple(sorted(int(x) for x in m["body_indices"]))
  groups.setdefault(key,[]).append((m,V,F,top))
 for body_ids,items in groups.items():
  Vcat=[];Fcat=[];off0=0
  for m,V,F,top in items:
   Vcat.append(np.asarray(V,np.float32));Fcat.append(np.asarray(F,np.int64)+off0);off0+=len(V)
  Vcat=np.vstack(Vcat);Fcat=np.vstack(Fcat)
  Vg,inv=np.unique(Vcat,axis=0,return_inverse=True);Fg=inv[Fcat]
  closed,topology=consistent_closed(Fg)
  if closed:
   mn=Vg.min(axis=0).astype(float)-1e-6;mx=Vg.max(axis=0).astype(float)+1e-6
   ids=np.flatnonzero(np.all(X>=mn,axis=1)&np.all(X<=mx,axis=1))
   if len(ids):
    w=solid_angle_winding(X[ids],Vg,Fg);hit=np.abs(w)>0.5;inside[ids[hit]]=True
    wind_stats.append({"body_indices":list(body_ids),"banks":[m["key"] for m,V,F,top in items],"closed_union":True,"topology":topology,"tested_points":int(len(ids)),"inside_points":int(hit.sum()),"max_abs_winding":float(np.max(np.abs(w)))})
   else:
    wind_stats.append({"body_indices":list(body_ids),"banks":[m["key"] for m,V,F,top in items],"closed_union":True,"topology":topology,"tested_points":0,"inside_points":0,"max_abs_winding":0.0})
  else:
   wind_stats.append({"body_indices":list(body_ids),"banks":[m["key"] for m,V,F,top in items],"closed_union":False,"topology":topology,"tested_points":0,"inside_points":0,"max_abs_winding":None,"reason":"open union; used unsigned distance and exact triangle gate, no solid sign"})
 do,of,ob,oq=closest_surface(X,outerV,outerF,k=48)
 # A 0.25 mm conservative vertex stand-off reduces the chance that a large
 # P1 cell crosses a bone between outside vertices; exact triangle tests follow.
 near=db<=0.00025
 phi=np.where(inside|near,1.0,do/np.maximum(do+np.maximum(db,1e-9),1e-9))
 phi=np.clip(phi,0.0,1.0)
 return phi,{"outside_signed_distance_classifier":"unsigned nearest-face distance plus closed-solid generalized solid-angle winding per selected registered anatomy bank; inside iff abs(winding)>0.5; near-surface margin 0.25mm sets phi=1","nodes_inside_closed_banks":int(inside.sum()),"nodes_within_0.25mm_of_bone":int(near.sum()),"bone_distance_quantiles_mm":np.quantile(db*1000,[0,.01,.05,.5,.95,.99,1]).tolist(),"outer_distance_quantiles_mm":np.quantile(do*1000,[0,.01,.5,.99,1]).tolist(),"winding_by_bank":wind_stats,"nearest_bone_bank_counts":dict(Counter(np.asarray(owner,dtype=object)[bi].tolist())),"inside_mask":inside,"nearest_bone_distance_m":db,"outer_distance_m":do}
def side_geometry(side,candidate,placement):
 shell=np.load(SHELL/f"{side}-source-shell.npz",allow_pickle=False)
 srcV=shell["vertices_world_m"].astype(np.float32);F0=shell["faces"].astype(np.int64);used=np.unique(F0)
 remap=np.full(len(srcV),-1,np.int64);remap[used]=np.arange(len(used));F=remap[F0]
 Oall=candidate["outer_world_actual007_m"].astype(np.float64)
 R=np.asarray(placement["placement_rotation"],float);t=np.asarray(placement["placement_translation_m"],float)
 placed=srcV.astype(np.float64)@R.T+t
 residual=np.linalg.norm(placed-Oall,axis=1)
 if float(residual.max())>1e-6: raise ValueError(f"{side} shell to candidate223 O mismatch max {residual.max()} m")
 OV=Oall[used].astype(np.float32).astype(np.float64)
 hist=face_edge_hist(F)
 vol=signed_volume(OV,F)
 if hist!={2:sum(hist.values())} or vol<=0:raise ValueError(f"{side} source shell not closed/outward: {hist}, volume={vol}")
 base=MESH/f"outer-{side}"
 Xraw,T=ns["parse_tetgen"](base)
 d0,_=tet_metrics(Xraw,T)
 if np.any(d0<=0):raise ValueError(f"{side} inherited TetGen connectivity has nonpositive signed tets")
 coord={}
 for j,p in enumerate(Xraw):
  k=tuple(float(x) for x in p)
  if k in coord: raise ValueError(f"{side} duplicate TetGen point")
  coord[k]=j
 bmap=np.empty(len(used),np.int64)
 for ii,sr in enumerate(used):
  key=tuple(float(x) for x in srcV[sr])
  if key not in coord: raise ValueError(f"{side} TetGen output lacks shell boundary source row {sr}")
  bmap[ii]=coord[key]
 Xw=Xraw@R.T+t
 Xw[bmap]=OV
 d64,q64=tet_metrics(Xw,T);X32=Xw.astype(np.float32);d32,q32=tet_metrics(X32,T)
 return shell,OV,F,Xw,X32,T,bmap,residual,vol,{"mesh_nodes":len(Xw),"mesh_tets":len(T),"raw_min_signed_determinant_m3":float(d0.min()),"actual007_min_signed_determinant_m3":float(d64.min()),"actual007_min_normalized_abs_determinant":float(q64.min()),"actual007_F32_min_signed_determinant_m3":float(d32.min()),"actual007_F32_nonpositive_tets":int(np.sum(d32<=0)),"boundary_nodes":len(bmap),"boundary_reprojection_max_um":float(residual.max()*1e6),"boundary_reprojection_p95_um":float(np.quantile(residual,.95)*1e6),"outer_shell_volume_m3":vol,"outer_edge_histogram":dict(hist)},base
def exact_surface_gates(OV,OF,I,D,bones):
 result={}
 for name,S in (("outer",{"points":OV,"faces":OF}),("skin_inner",I),("fat_deep",D)):
  result[name+"_self"]=exact_self(S["points"],S["faces"])
 for a,b in (("outer","skin_inner"),("outer","fat_deep"),("skin_inner","fat_deep")):
  va,fa=(OV,OF) if a=="outer" else ({"skin_inner":I,"fat_deep":D}[a]["points"],{"skin_inner":I,"fat_deep":D}[a]["faces"])
  vb,fb=(OV,OF) if b=="outer" else ({"skin_inner":I,"fat_deep":D}[b]["points"],{"skin_inner":I,"fat_deep":D}[b]["faces"])
  result[a+"_vs_"+b]=exact_cross(va,fa,vb,fb)
 for lname,S in (("outer",{"points":OV,"faces":OF}),("skin_inner",I),("fat_deep",D)):
  allhits=0;examples=[];aabb=0;per=[]
  for m,V,F,_ in bones:
   r=exact_cross(S["points"],S["faces"],V,F,keep=4)
   allhits+=r["intersection_pairs"];aabb+=r["aabb_candidates"]
   if r["intersection_pairs"]:per.append({"bank":m["key"],"stable_id":m["stable_id"],"body_indices":m["body_indices"],"pairs":r["intersection_pairs"],"examples":r["examples"][:4]})
  result[lname+"_vs_registered_bones"]={"intersection_pairs":allhits,"aabb_candidates":aabb,"banks_with_hits":per}
 return result
def seed_from_phi(X,T,phi,lo,hi,target):
 avg=phi[T].mean(axis=1);ids=np.flatnonzero((avg>lo+1e-10)&(avg<hi-1e-10))
 if not len(ids): raise ValueError(f"no TetGen seed cell for field interval {lo}..{hi}")
 i=int(ids[np.argmin(np.abs(avg[ids]-target))])
 return X[T[i]].mean(axis=0),float(avg[i]),i
def run_volume_tetgen(side,OV,OF,I,D,X,T,phi,lI,lD):
 V=np.vstack((OV,I["points"],D["points"]))
 iF=I["faces"]+len(OV);dF=D["faces"]+len(OV)+len(I["points"])
 # Each nested closed surface forms a separate material boundary.
 skin_seed,skin_phi,skin_t=seed_from_phi(X,T,phi,1e-6,lI,0.5*lI)
 fat_seed,fat_phi,fat_t=seed_from_phi(X,T,phi,lI,lD,0.5*(lI+lD))
 core_seed,core_phi,core_t=seed_from_phi(X,T,phi,lD,2.0,1.0)
 seeds=[(skin_seed,1),(fat_seed,2),(core_seed,3)]
 base=OUT/f"volume-{side}"
 write_multiregion_poly(str(base)+".poly",V,[(OV,OF,1),(I["points"],I["faces"],2),(D["points"],D["faces"],3)],seeds)
 r=subprocess.run([str(TETGEN),"-pYq1.2A",str(base)+".poly"],cwd=OUT,text=True,capture_output=True,timeout=180)
 (OUT/f"volume-{side}.stdout").write_text(r.stdout);(OUT/f"volume-{side}.stderr").write_text(r.stderr)
 if r.returncode: raise RuntimeError(f"TetGen {side} rc={r.returncode}: {r.stderr[-1000:]} {r.stdout[-1000:]}")
 XV,TT=ns["parse_tetgen"](base);TT,attrs=read_ele_attributes(base)
 det,q=tet_metrics(XV,TT)
 if np.any(det<=0):raise ValueError(f"{side} output volume has nonpositive tetra")
 by={}
 for a in np.unique(attrs):
  m=attrs==a;by[str(int(a))]={"tets":int(m.sum()),"volume_m3":float(det[m].sum())}
 # Face markers summarize whether three nested PLC boundaries survived as constraints.
 facepath=Path(str(base)+".1.face");lines=[x.split("#",1)[0].strip() for x in facepath.read_text().splitlines() if x.split("#",1)[0].strip()]
 h=lines[0].split();markers=Counter()
 for line in lines[1:int(h[0])+1]:
  qv=line.split();markers[int(qv[-1])]+=1
 return {"mesh_nodes":len(XV),"mesh_tets":len(TT),"min_signed_det_m3":float(det.min()),"min_normalized_abs_det":float(q.min()),"material_regions":by,"face_marker_counts":dict(markers),"seed_cells":{"skin":{"tet":skin_t,"phi":skin_phi},"fat":{"tet":fat_t,"phi":fat_phi},"core":{"tet":core_t,"phi":core_phi}},"volume_mesh_npz":str(OUT/f"volume-{side}-mesh.npz"),"poly_path":str(base)+".poly","stdout_sha256":sha(OUT/f"volume-{side}.stdout"),"stderr_sha256":sha(OUT/f"volume-{side}.stderr")},XV,TT,attrs
def main():
 candidate=np.load(CAND,allow_pickle=False);placement=json.loads(PLACEMENT.read_text())
 pose_hash=sha(POSE);inv=json.loads(INV.read_text())
 if inv["pose_sha256"]!=pose_hash:raise ValueError("registered bank pose does not match native007")
 result={"schema":"numi.human.common-p1-layer-volume-prototype.v1","status":"offline_candidate_only_not_native_or_integration_admitted","frame":"actual native007 registered world; TetGen source-domain nodes rigidly transformed by pinned 024 R/T then boundary-snapped to candidate223 Float32 outer O","method":"One TetGen outer-domain connectivity with a single P1 normalized outer-to-bone clearance scalar; winding-number inside classification per closed actual007 anatomy bank; two ordered level sets from the same scalar define independent skin-inner and fat-deep surfaces; a second TetGen PLC partitions the bounded volume into skin, fat, and explicitly non-tissue core regions.","limitations":["source-shell cap closes only the bounded lower-limb subdomain; this is not a whole-body mesh","two level interfaces have independent vertex/face topology and cannot be represented by current paired O/I/D row-block owner without a mapped-boundary owner extension","no exact parent-tet overlap CSR or registered donor-mass debit has been computed; geometric region volume is not a conservative physical mass map","0.25mm nodal bone stand-off is a field-construction choice; exact triangle checks remain authoritative","TetGen core material attribute 3 is a non-tissue partition placeholder and must not be integrated as skin/fat"],"input_sha256":{},"sides":{}}
 pins=[CAND,SHELL/"left-source-shell.npz",SHELL/"right-source-shell.npz",MESH/"left-exact-obstacle-preflight.json",MESH/"right-exact-obstacle-preflight.json",POSE,PLACEMENT,BANK,INV,COMMON,TETGEN]
 result["input_sha256"]={str(p):sha(p) for p in pins}
 for side in ("left","right"):
  print(json.dumps({"stage":"start","side":side},sort_keys=True),flush=True)
  shell,OV,OF,Xw,X32,T,bmap,resid,outervol,mesh,base=side_geometry(side,candidate,placement)
  banks=classify_banks(side,OV)
  phi,fd=build_bone_field(X32,OV.astype(np.float32),OF,banks)
  lI,SI,bI=choose_level(X32,T,phi,outervol-TARGET[side]["skin"])
  lD,SD,bD=choose_level(X32,T,phi,outervol-TARGET[side]["skin"]-TARGET[side]["fat"])
  if not lI<lD:raise ValueError(f"{side} scalar interfaces not ordered: {lI} !< {lD}")
  # Source-face ancestry follows nearest triangle on the exact bounded O shell.
  srcids=shell["source_face_ids"]
  for name,S,level in (("skin_inner",SI,lI),("fat_deep",SD,lD)):
   d,fi,bary,qpt=closest_surface(S["points"],OV,OF,k=64)
   S["source_shell_face_index"]=fi
   S["source_face_id"]=np.where(fi<len(srcids),srcids[np.minimum(fi,len(srcids)-1)],-1)
   S["source_face_barycentric"]=bary
   S["source_projection_world_m"]=qpt
   S["source_projection_distance_m"]=d
  gate=exact_surface_gates(OV.astype(np.float32),OF,SI,SD,banks)
  level_open={"skin_inner":dict(face_edge_hist(SI["faces"])),"fat_deep":dict(face_edge_hist(SD["faces"]))}
  level_closed=all(set(h)=={2} for h in level_open.values())
  layer_pair_hits=sum(v["intersection_pairs"] for k,v in gate.items() if "_vs_" in k and not k.endswith("_registered_bones"))
  bone_hits=sum(gate[f"{layer}_vs_registered_bones"]["intersection_pairs"] for layer in ("outer","skin_inner","fat_deep"))
  self_hits=sum(int(v.get("count",0)) for k,v in gate.items() if k.endswith("_self"))
  surface_pass=level_closed and layer_pair_hits==0 and bone_hits==0 and self_hits==0
  skinvol=outervol-SI["signed_volume_m3"];fatvol=SI["signed_volume_m3"]-SD["signed_volume_m3"]
  report={"side":side,"outer_shell":{"vertices":len(OV),"faces":len(OF),"edge_histogram":dict(face_edge_hist(OF)),"volume_m3":outervol,"candidate223_source_rows":len(shell["vertices_world_m"]),"max_source_shell_to_candidate223_O_residual_um":float(resid.max()*1e6),"p95_source_shell_to_candidate223_O_residual_um":float(np.quantile(resid,.95)*1e6)},"initial_mesh":mesh,
   "common_field":{"classifier":fd["outside_signed_distance_classifier"],"nodes_inside_banks":fd["nodes_inside_closed_banks"],"nodes_within_0_25mm":fd["nodes_within_0.25mm_of_bone"],"bone_distance_quantiles_mm":fd["bone_distance_quantiles_mm"],"outer_distance_quantiles_mm":fd["outer_distance_quantiles_mm"],"winding_by_bank":fd["winding_by_bank"],"nearest_bone_bank_counts":fd["nearest_bone_bank_counts"]},
   "level_choice":{"skin_inner":{"phi":float(lI),"enclosed_volume_m3":SI["signed_volume_m3"],"iteration_bracket":bI},"fat_deep":{"phi":float(lD),"enclosed_volume_m3":SD["signed_volume_m3"],"iteration_bracket":bD},"skin_geometric_volume_m3":skinvol,"fat_geometric_volume_m3":fatvol,"skin_target_volume_m3":TARGET[side]["skin"],"fat_target_volume_m3":TARGET[side]["fat"],"skin_target_delta_percent":100*(skinvol/TARGET[side]["skin"]-1),"fat_target_delta_percent":100*(fatvol/TARGET[side]["fat"]-1)},
   "surface_topology":{"skin_inner_vertices":len(SI["points"]),"skin_inner_faces":len(SI["faces"]),"skin_inner_edge_histogram":level_open["skin_inner"],"fat_deep_vertices":len(SD["points"]),"fat_deep_faces":len(SD["faces"]),"fat_deep_edge_histogram":level_open["fat_deep"],"skin_inner_min_double_area2_m2":SI["min_double_area2_m2"],"fat_deep_min_double_area2_m2":SD["min_double_area2_m2"]},
   "exact_surface_gates":gate,"surface_gate_pass":bool(surface_pass),"registered_banks":[{"key":m["key"],"stable_id":m["stable_id"],"body_indices":m["body_indices"],"triangles":len(F)} for m,V,F,_ in banks],
   "outputs":{}}
  for name,S in (("skin_inner",SI),("fat_deep",SD)):
   np.savez_compressed(OUT/f"{side}-{name}.npz",points_world_m=S["points"],faces=S["faces"],parent_tet_edge_node_ids=S["parent_edge"],edge_parameter=S["edge_t"],parent_tet_ids=S["parent_tet"],source_shell_face_index=S["source_shell_face_index"],source_face_id=S["source_face_id"],source_face_barycentric=S["source_face_barycentric"],source_projection_world_m=S["source_projection_world_m"],source_projection_distance_m=S["source_projection_distance_m"],level_phi=np.asarray([lI if name=="skin_inner" else lD]),frame=np.asarray(["actual007-world"]))
   report["outputs"][name]={"path":str(OUT/f"{side}-{name}.npz"),"sha256":sha(OUT/f"{side}-{name}.npz")}
  # Always preserve the finite common field/interfaces first; only create a
  # material volume if the three closed nested surfaces are geometrically valid.
  volume_report=None
  if layer_pair_hits==0 and level_closed:
   try:
    volume_report,XV,TT,attrs=run_volume_tetgen(side,OV.astype(np.float32),OF,SI,SD,X32,T,phi,lI,lD)
    np.savez_compressed(OUT/f"volume-{side}-mesh.npz",volume_points_world_m=XV,child_tetrahedra=TT,child_material_codes=attrs.astype(np.int8),material_code_meaning=np.asarray(["1=skin","2=fat","3=core_non_tissue"]))
    volume_report["volume_mesh_sha256"]=sha(OUT/f"volume-{side}-mesh.npz")
    report["outputs"]["volume_mesh"]={"path":str(OUT/f"volume-{side}-mesh.npz"),"sha256":volume_report["volume_mesh_sha256"]}
   except Exception as e:
    volume_report={"status":"failed","error":repr(e)}
  report["second_tetgen_partition"]=volume_report
  report["status"]="finite-common-field-unadmitted" if not surface_pass else ("finite-partition-unadmitted-no-conservative-mass-map" if volume_report and volume_report.get("status")!="failed" else "surface-levels-pass-volume-mesh-missing")
  (OUT/f"{side}-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
  result["sides"][side]=report
  print(json.dumps({"stage":"side_done","side":side,"surface_gate_pass":surface_pass,"surface_counts":report["surface_topology"],"exact_counts":{k:v.get("intersection_pairs",v.get("count",None)) for k,v in gate.items()},"second_tetgen":volume_report},sort_keys=True),flush=True)
 # keep O/I/D and initial P1 field for a later conservative overlap owner.
 (OUT/"commonfield-224.npz").write_bytes(b"")
 arrays={}
 for side in ("left","right"):
  shell,OV,OF,Xw,X32,T,bmap,resid,outervol,mesh,base=side_geometry(side,candidate,placement)
  # Reload emitted interfaces with their saved exact ancestry fields.
  i=np.load(OUT/f"{side}-skin_inner.npz");d=np.load(OUT/f"{side}-fat_deep.npz")
  arrays[f"{side}_outer_world_m"]=OV.astype(np.float32);arrays[f"{side}_outer_faces"]=OF
  arrays[f"{side}_tet_points_world_m"]=X32;arrays[f"{side}_tetrahedra"]=T
  arrays[f"{side}_skin_inner_points_world_m"]=i["points_world_m"];arrays[f"{side}_skin_inner_faces"]=i["faces"]
  arrays[f"{side}_skin_inner_parent_tet_edge_node_ids"]=i["parent_tet_edge_node_ids"];arrays[f"{side}_skin_inner_edge_parameter"]=i["edge_parameter"];arrays[f"{side}_skin_inner_parent_tet_ids"]=i["parent_tet_ids"]
  arrays[f"{side}_fat_deep_points_world_m"]=d["points_world_m"];arrays[f"{side}_fat_deep_faces"]=d["faces"]
  arrays[f"{side}_fat_deep_parent_tet_edge_node_ids"]=d["parent_tet_edge_node_ids"];arrays[f"{side}_fat_deep_edge_parameter"]=d["edge_parameter"];arrays[f"{side}_fat_deep_parent_tet_ids"]=d["parent_tet_ids"]
 np.savez_compressed(OUT/"commonfield-224.npz",**arrays)
 result["output_npz_sha256"]=sha(OUT/"commonfield-224.npz")
 result["status"]="bilateral-finite-common-field-emitted"
 result["elapsed_s"]=time.monotonic()-t_start
 result["outputs"]={str(p.name):sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}
 (OUT/"candidate-report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"stage":"complete","status":result["status"],"output_npz_sha256":result["output_npz_sha256"],"elapsed_s":result["elapsed_s"],"directory":str(OUT)},sort_keys=True),flush=True)
if __name__=="__main__":main()
