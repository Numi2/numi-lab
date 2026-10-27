#!/usr/bin/env python3
"""One bounded local 3D fit trial for skin body 109 / stable bone 114."""
from pathlib import Path
from collections import defaultdict, Counter
from fractions import Fraction
import hashlib, importlib.util, json, sys, time, traceback
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
S=ROOT/"D/surface-clearance-546/component-sat-selector-638"
WORK=S/"per-bank-workers-20261027/b109-114"
TOPO=WORK/"attempt-002-exact-contours"
OUT=WORK/"attempt-003-combined-3d-fit"
C=ROOT/"D/publication-182-stage-001/combined-653/candidate.npz"
A=C.parent/"audit-159/report.json"
AUDIT_TOOL=ROOT/"audit_skin_bone_surfaces_159.py"
HELPER=Path("/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
BANKDIR=E/"bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
BODY,BONE=109,114
BASE_SHA="7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af"
AUDIT_SHA="c480d498bc1fb0f361e817b7f785b7b425ba9f40c0e181f7176ae375103cce8e"
PATCH_SHA=["0f4a35e195a78b1f8153194cb0d7e8e8ed1a0a9af130a604d633e56e084bea95","787df5675fc7b835ec89430304b76dd1e21f14ea4c0c8942d52dc78f211a36db"]
MARGIN=.00015
MAX_MOVE=.004
AREA_MIN,AREA_MAX=.25,4.
LAMBDA=.5
W_ACTIVE=30.
W_OTHER=.05

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def loadmod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m

def f32key(v): return tuple(map(int,np.asarray(v,dtype=np.float32).view(np.uint32)))

def exact_tri(t):
 return tuple(tuple(Fraction.from_float(float(x)) for x in p) for p in np.asarray(t).reshape(3,3))

def nonzero_tri(t,exact):
 q=exact_tri(t)
 return any(exact._cross(exact._sub(q[1],q[0]),exact._sub(q[2],q[0])))

def closest(p,t):
 a,b,c=np.asarray(t,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
 ab,ac,ap=b-a,c-a,p-a
 d00,d01,d11=float(ab@ab),float(ab@ac),float(ac@ac)
 d20,d21=float(ap@ab),float(ap@ac)
 den=d00*d11-d01*d01
 if den<=0: raise ValueError("degenerate target bone triangle")
 v=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den;u=1-v-w
 if min(u,v,w)>=0:return u*a+v*b+w*c
 def seg(x,y):
  e=y-x;t0=min(1.,max(0.,float((p-x)@e)/float(e@e)));q=x+t0*e
  return q,float((p-q)@(p-q))
 return min((seg(a,b),seg(b,c),seg(c,a)),key=lambda x:x[1])[0]

def laplacian(n,faces):
 es=set()
 for t in np.asarray(faces,dtype=np.int64):
  for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
   a,b=int(a),int(b)
   if a!=b:es.add((min(a,b),max(a,b)))
 rr=[];cc=[];vv=[]
 for a,b in sorted(es):
  rr.extend((a,b,a,b));cc.extend((a,b,b,a));vv.extend((1.,1.,-1.,-1.))
 return coo_matrix((vv,(rr,cc)),shape=(n,n)).tocsr()

def csr_rows(ptr,values):
 return [np.asarray(values[int(ptr[i]):int(ptr[i+1])],dtype=np.int64).copy() for i in range(len(ptr)-1)]

def pack_csr(rows):
 p=[0];v=[]
 for row in rows:v.extend(map(int,row));p.append(len(v))
 return np.asarray(p,dtype=np.int64),np.asarray(v,dtype=np.int64)

def main():
 start=time.monotonic();OUT.mkdir(parents=True,exist_ok=False)
 report_path=OUT/"fit-report.json";trial_path=OUT/"local-replacement-trial.npz"
 rep={"schema":"numi.human.b109_114.local_replacement_fit_trial.v1","status":"STARTED",
  "body":BODY,"stable_bone_id":BONE,"base_candidate_path":str(C),"base_candidate_sha256":None,
  "base_audit159_sha256":None,"topology_bundle_sha256":None,"trial_path":str(trial_path),
  "fit_parameters":{"bone_outward_target_margin_m":MARGIN,"laplacian_weight":LAMBDA,
   "active_target_weight":W_ACTIVE,"inactive_anchor_weight":W_OTHER,"max_node_move_m":MAX_MOVE,
   "child_area_ratio_guard":[AREA_MIN,AREA_MAX],"fixed_outer_boundary":True},
  "checks":{"full_body_audit":False,"all_bank_audit":False,"geometry_admitted":False,"physical_steps":0,"native_admission":False}}
 def save_report():
  rep["elapsed_seconds"]=time.monotonic()-start
  report_path.write_text(json.dumps(rep,indent=2,sort_keys=True,allow_nan=False)+"\n")
 try:
  if sha(C)!=BASE_SHA or sha(A)!=AUDIT_SHA:raise RuntimeError("pinned source candidate/audit changed")
  report_pins={"exact-contour-graph.json":"524e1c62a7768b5ec9ae50a2b7bc5ac65bd2303ff9d03355486bf784cfc932ab",
   "facewise-triangulation-report.json":"3960375ac7d6efdfea0a1705c535489957875d0880f76eb6d1a3827cc24bf0c8",
   "integrity-check.json":"6ed79b3ef868f682103c6d448ea5698b7899dd7d8013c25b5aeebee322c128c9"}
  for n,h in report_pins.items():
   if sha(TOPO/n)!=h:raise RuntimeError("topology pin mismatch: "+n)
  rep.update({"base_candidate_sha256":sha(C),"base_audit159_sha256":sha(A),
   "topology_bundle_sha256":sha(TOPO/"topology-bundle.json")})
  z=np.load(C,allow_pickle=False);P0=np.asarray(z["points_world_m"],dtype=np.float32)
  F0=np.asarray(z["source_O_faces"],dtype=np.int64)
  base_src=csr_rows(z["face_source_ancestry_indptr"],z["face_source_ancestry_face_rows"])
  base_sub=csr_rows(z["face_fine_subface_ancestry_indptr"],z["face_fine_subface_ancestry_rows"])
  base_det=csr_rows(z["face_fine_detail_ancestry_indptr"],z["face_fine_detail_ancestry_rows"])
  rep["base_mesh"]={"node_count":len(P0),"face_count":len(F0),
   "baseline_self_pair_count":int(len(np.load(C.parent/"audit-130/surface-self-pairs.npy",allow_pickle=False)))}
  helper=loadmod("fit109_helper",HELPER);audit=loadmod("fit109_audit",AUDIT_TOOL);exact=audit.exact_module()
  T,ids,gids,bodies,M,off,pos,bankkeys,source_pins,unused=helper.read_bank_inputs(BANKDIR,E)
  bm=(ids[:,0]==BODY)&(ids[:,1]==BONE);bone_ids=ids[bm,3].astype(np.int64);BT=np.asarray(T[bm],dtype=np.float64)
  if BT.shape!=(392,3,3):raise RuntimeError("actual target bone triangle count changed")
  edgeuses=defaultdict(list)
  for fi,t in enumerate(BT):
   k=[f32key(v) for v in t]
   for j in range(3):
    a,b=k[j],k[(j+1)%3];edgeuses[tuple(sorted((a,b)))].append((a,b,fi))
  edgehist=Counter(map(len,edgeuses.values()))
  badedges=[(e,v) for e,v in edgeuses.items() if len(v)!=2 or v[0][0]!=v[1][1] or v[0][1]!=v[1][0]]
  if dict(edgehist)!={2:588} or badedges:raise RuntimeError(f"actual bone exact-bit topology/orientation fails: {dict(edgehist)} {badedges[:8]}")
  volume=float(sum(np.dot(t[0],np.cross(t[1],t[2])) for t in BT)/6.)
  if abs(volume)<1e-15:raise RuntimeError("actual bone signed volume is zero")
  normals=np.cross(BT[:,1]-BT[:,0],BT[:,2]-BT[:,0])*(1. if volume>0 else -1.)
  normals/=np.linalg.norm(normals,axis=1)[:,None]
  brecs=exact._records(BT.reshape(-1,3),np.arange(len(BT)*3,dtype=np.int64).reshape(-1,3))
  rep["target_bone"]={"triangle_count":len(BT),"exact_bit_vertex_count":len({f32key(v) for t in BT for v in t}),
   "exact_bit_edge_count":len(edgeuses),"edge_valence_histogram":dict(edgehist),"consistent_orientation":True,
   "signed_volume_m3":volume,"source_pins":source_pins}
  comps=[];app=[];childfaces=[];childparents=[];childcomp=[];pmaps=[];boundary=[];contours=[];replaced=[];nextrow=len(P0)
  for ci in (0,1):
   path=TOPO/f"component-{ci:02d}/conforming-patch.npz"
   if sha(path)!=PATCH_SHA[ci]:raise RuntimeError(f"component {ci} topology hash mismatch")
   with np.load(path,allow_pickle=False) as q:D={k:q[k].copy() for k in q.files}
   src=np.asarray(D["source_candidate_node_rows"],dtype=np.int64);gmap=np.empty(len(src),dtype=np.int64);ends=[]
   for li,sr in enumerate(src):
    if sr>=0:
     if sr>=len(P0) or not np.array_equal(D["points_world_m_f32"][li].view(np.uint32),P0[sr].view(np.uint32)):
      raise RuntimeError(f"component {ci} existing source node {sr} disagrees")
     gmap[li]=sr
    else:
     gmap[li]=nextrow;nextrow+=1;ends.append(li);app.append(D["points_world_m_f32"][li])
   par=np.asarray(D["original_patch_face_rows"],dtype=np.int64)
   cp=np.asarray(D["face_parent_candidate_rows"],dtype=np.int64)
   face=gmap[np.asarray(D["patch_faces"],dtype=np.int64)]
   if not np.all(np.isin(cp,par)):raise RuntimeError("child-to-parent face map escaped patch")
   cs=csr_rows(D["face_source_ancestry_indptr"],D["face_source_ancestry_face_rows"])
   cu=csr_rows(D["face_fine_subface_ancestry_indptr"],D["face_fine_subface_ancestry_rows"])
   cd=csr_rows(D["face_fine_detail_ancestry_indptr"],D["face_fine_detail_ancestry_face_rows"])
   for j,p in enumerate(cp):
    if not (np.array_equal(cs[j],base_src[int(p)]) and np.array_equal(cu[j],base_sub[int(p)]) and np.array_equal(cd[j],base_det[int(p)])):
     raise RuntimeError(f"child {j} source/fine ancestry differs from source face {int(p)}")
   replaced.extend(map(int,par));childfaces.extend(face.tolist());childparents.extend(map(int,cp));childcomp.extend([ci]*len(face))
   pmaps.append(gmap);boundary.extend(gmap[np.asarray(D["patch_outer_boundary_edges"],dtype=np.int64)].tolist())
   contours.extend(gmap[np.asarray(D["contour_edges"],dtype=np.int64)].tolist())
   comps.append({"id":ci,"D":D,"map":gmap,"faces":face,"parents":par,"childparents":cp,"ends":ends,
    "src":cs,"sub":cu,"detail":cd})
  if len(set(replaced))!=len(replaced):raise RuntimeError("component parent faces overlap")
  replaced=np.asarray(sorted(replaced),dtype=np.int64);keep=np.ones(len(F0),bool);keep[replaced]=False
  outside=np.flatnonzero(keep).astype(np.int64);PF=np.asarray(childfaces,dtype=np.int64)
  PP=np.asarray(childparents,dtype=np.int64);PC=np.asarray(childcomp,dtype=np.int8)
  Pinit=np.vstack([P0,np.asarray(app,dtype=np.float32)]) if app else P0.copy()
  if len(Pinit)!=nextrow:raise RuntimeError("global node map count mismatch")
  for ci,gmap in enumerate(pmaps):
   for cj,other in enumerate(pmaps):
    if cj>ci and set(map(int,gmap[gmap<len(P0)])).intersection(map(int,other[other<len(P0)])):
     raise RuntimeError("separate patch components share a source candidate node")
  rep["topology"]={"replaced_parent_face_count":len(replaced),"outside_original_face_count":len(outside),
   "child_face_count":len(PF),"combined_node_count":len(Pinit),
   "components":[{"id":c["id"],"parent_faces":len(c["parents"]),"child_faces":len(c["faces"]),
    "boundary_edges":len(c["D"]["patch_outer_boundary_edges"]),"contour_edges":len(c["D"]["contour_edges"]),
    "added_endpoint_nodes":len(c["ends"])} for c in comps]}
  # Fit each of the two patches once, toward the target bone exterior.
  Pfit=Pinit.copy();location_cache={}
  def locate(p):
   key=tuple(float(x) for x in p)
   if key not in location_cache:
    location_cache[key]=exact.point_location(tuple(Fraction.from_float(float(x)) for x in p),brecs)
   return location_cache[key]
  fitreps=[];beforeloc=[]
  for c,gmap in zip(comps,pmaps):
   ci=c["id"];D=c["D"];xw=np.asarray(D["points_world_m_f32"],dtype=np.float32)
   xl=(xw.astype(np.float64)-pos[BODY])@M[BODY].T-off[BODY]
   bound=np.unique(np.asarray(D["patch_outer_boundary_edges"],dtype=np.int64).reshape(-1));bs=set(map(int,bound))
   loc=[locate(p) for p in xl]
   inside=[i for i,x in enumerate(loc) if x["location"]=="inside"]
   boundaryhits=[i for i,x in enumerate(loc) if x["location"]=="boundary"]
   ind=[i for i,x in enumerate(loc) if x["location"]=="indeterminate"]
   if boundaryhits or ind:raise RuntimeError(f"component {ci} has target bone boundary/indeterminate input vertices: {boundaryhits[:8]} {ind[:8]}")
   if any(loc[i]["location"]!="outside" for i in bound):raise RuntimeError(f"component {ci} fixed boundary vertex is not outside target bone")
   lo=BT.min(axis=1);hi=BT.max(axis=1);target=np.zeros_like(xl);dist=np.full(len(xl),np.inf);nearest=np.full(len(xl),-1,dtype=np.int64)
   for i,p in enumerate(xl):
    best=float("inf");qbest=None;fbest=-1
    for bf,t in enumerate(BT):
     if np.any(p<lo[bf]-best) or np.any(p>hi[bf]+best):continue
     q=closest(p,t);d=float(np.linalg.norm(p-q))
     if d<best:best,qbest,fbest=d,q,bf
    dist[i]=best;nearest[i]=fbest
    if loc[i]["location"]=="inside" or best<MARGIN:target[i]=qbest+MARGIN*normals[fbest]-p
   active=np.asarray([i for i in np.flatnonzero(np.linalg.norm(target,axis=1)>0) if int(i) not in bs],dtype=np.int64)
   if not len(active):raise RuntimeError(f"component {ci} has no movable node requiring bone correction")
   L=laplacian(len(xw),D["patch_faces"]);bmask=np.zeros(len(xw),bool);bmask[list(bs)]=True;free=np.flatnonzero(~bmask)
   wt=np.full(len(xw),W_OTHER);wt[active]=W_ACTIVE;K=diags(wt)+LAMBDA*L
   Kff=K[free][:,free].tocsc();rhs=wt[free,None]*target[free]
   u=np.zeros_like(xl);u[free]=np.column_stack([spsolve(Kff,rhs[:,axis]) for axis in range(3)])
   move=np.linalg.norm(u,axis=1);mmax=float(move.max(initial=0.))
   if mmax>MAX_MOVE:raise RuntimeError(f"component {ci} exceeds 4 mm construction trust: {mmax}")
   yw=(xw.astype(np.float64)+u@M[BODY]).astype(np.float32)
   for i in bound:
    row=int(gmap[int(i)])
    if row>=len(P0) or not np.array_equal(yw[int(i)].view(np.uint32),P0[row].view(np.uint32)):
     raise RuntimeError(f"component {ci} moved fixed boundary node {int(i)}")
   for i,row in enumerate(gmap):Pfit[int(row)]=yw[i]
   fitreps.append({"component_id":ci,"node_count":len(xw),"fixed_boundary_nodes":len(bs),
    "inside_nodes_before":len(inside),"active_movable_nodes":len(active),"min_distance_to_bone_m":float(dist.min()),
    "max_node_move_m":mmax,"rms_node_move_m":float(np.sqrt(np.mean(move*move))),
    "fixed_boundary_bitwise":True,"nearest_bone_face_counts":{str(int(k)):int(v) for k,v in Counter(nearest).items()}})
   beforeloc.append({"component_id":ci,"location_histogram":dict(Counter(x["location"] for x in loc))})
  rep["fit"]={"method":"one weighted Laplacian fit toward closest actual bone point plus 0.15 mm outward offset",
   "components":fitreps,"input_exact_point_locations":beforeloc}
  # Shape/orientation/area gates run first.
  boundary=np.asarray(boundary,dtype=np.int64).reshape(-1,2);contours=np.asarray(contours,dtype=np.int64).reshape(-1,2)
  bn=np.unique(boundary.reshape(-1));fixed=np.array_equal(Pfit[bn].view(np.uint32),Pinit[bn].view(np.uint32))
  old=Pinit[PF].astype(np.float64);new=Pfit[PF].astype(np.float64)
  n0=np.cross(old[:,1]-old[:,0],old[:,2]-old[:,0]);n1=np.cross(new[:,1]-new[:,0],new[:,2]-new[:,0])
  a0=np.linalg.norm(n0,axis=1);a1=np.linalg.norm(n1,axis=1);ratio=a1/a0
  deg0=np.flatnonzero(a0==0);deg1=np.flatnonzero(a1==0)
  inverted=np.flatnonzero(np.einsum("ij,ij->i",n0,n1)<=0)
  areabad=np.flatnonzero((ratio<AREA_MIN)|(ratio>AREA_MAX)|~np.isfinite(ratio))
  exactdeg=[i for i,t in enumerate(Pfit[PF]) if not nonzero_tri(t,exact)]
  shape=fixed and not len(deg0) and not len(deg1) and not len(inverted) and not len(areabad) and not len(exactdeg)
  rep["shape_gates"]={"status":"PASS" if shape else "FAIL","fixed_outer_boundary_bitwise":fixed,
   "degenerate_before":len(deg0),"degenerate_after_numeric":len(deg1),"degenerate_after_exact":len(exactdeg),
   "exact_degenerate_face_indices":exactdeg[:128],"inverted_or_flat_face_count":len(inverted),
   "inverted_or_flat_face_indices":inverted[:128].tolist(),"area_ratio_guard":[AREA_MIN,AREA_MAX],
   "area_ratio_min":float(ratio.min()),"area_ratio_max":float(ratio.max()),"area_ratio_fail_count":len(areabad),
   "area_ratio_fail_face_indices":areabad[:128].tolist(),"double_area_before_range_m2":[float(a0.min()),float(a0.max())],
   "double_area_after_range_m2":[float(a1.min()),float(a1.max())],"max_move_m":max(x["max_node_move_m"] for x in fitreps)}
  # Keep a complete local replacement mesh and all direct source-face ancestry.
  FF=np.vstack([F0[outside],PF]);faceparent=np.concatenate([outside,PP]);facetag=np.concatenate([np.full(len(outside),-1,dtype=np.int8),PC])
  src=[base_src[int(i)] for i in outside];sub=[base_sub[int(i)] for i in outside];detail=[base_det[int(i)] for i in outside]
  child_src=[];child_sub=[];child_detail=[]
  for c in comps:child_src.extend(c["src"]);child_sub.extend(c["sub"]);child_detail.extend(c["detail"])
  srcptr,srcrows=pack_csr(src+child_src);subptr,subrows=pack_csr(sub+child_sub);detptr,detrows=pack_csr(detail+child_detail)
  node_src=np.concatenate([np.arange(len(P0),dtype=np.int64),np.full(len(Pfit)-len(P0),-1,dtype=np.int64)])
  node_parent=np.full(len(Pfit),-1,dtype=np.int64);node_bary=np.zeros((len(Pfit),3),dtype=np.float64);ep=[]
  for c,gmap in zip(comps,pmaps):
   D=c["D"]
   for li,row in enumerate(gmap):
    if D["source_candidate_node_rows"][li]>=0:continue
    par=int(D["point_parent_candidate_face_rows"][li]);b=np.asarray(D["point_parent_candidate_barycentric_f64"][li],dtype=np.float64)
    if par<0 or np.any(b<0) or abs(float(b.sum())-1)>1e-12:raise RuntimeError("invalid endpoint skin-face barycentric ancestry")
    node_parent[row]=par;node_bary[row]=b
    ep.append({"component":c["id"],"local_node":li,"global_node":int(row),"parent_candidate_face":par,
     "barycentric":b.tolist(),"endpoint_component_identity":int(D["exact_endpoint_component_ids"][li])})
  kind=np.concatenate([z["face_kind"][outside],z["face_kind"][PP]])
  with trial_path.open("xb") as f:
   np.savez_compressed(f,points_world_m=Pfit.astype(np.float32),source_O_faces=FF.astype(np.int64),
    source_candidate_node_rows=node_src,node_parent_candidate_face_rows=node_parent,node_parent_candidate_barycentric=node_bary,
    source_candidate_face_rows=faceparent,face_component_id=facetag,face_kind=kind,
    face_source_ancestry_indptr=srcptr,face_source_ancestry_face_rows=srcrows,
    face_fine_subface_ancestry_indptr=subptr,face_fine_subface_ancestry_rows=subrows,
    face_fine_detail_ancestry_indptr=detptr,face_fine_detail_ancestry_face_rows=detrows,
    replaced_source_candidate_face_rows=replaced,patch_outer_boundary_edges_global=boundary,patch_contour_edges_global=contours,
    patch_node_rows_component_00=pmaps[0],patch_node_rows_component_01=pmaps[1],patch_child_face_rows_global=PF,
    patch_child_parent_candidate_face_rows=PP,patch_child_component_id=PC,
    endpoint_source_face_lineage_json=np.asarray(json.dumps(ep,separators=(",",":"))),
    node_world_displacement_m=Pfit.astype(np.float64)-Pinit.astype(np.float64))
  rep["trial_file_sha256"]=sha(trial_path);rep["ancestry"]={"source_rows_match_parent":True,
   "full_replacement_faces":len(FF),"endpoint_nodes_with_skin_face_barycentrics":len(ep),
   "source_ancestry_entries":len(srcrows),"fine_subface_ancestry_entries":len(subrows),
   "fine_detail_ancestry_entries":len(detrows)}
  save_report()
  if not shape:
   rep["status"]="REJECTED_SHAPE_OR_BOUNDARY_GUARD"
   rep["checks"]["exact_local_screens"]="NOT_RUN_AFTER_SHAPE_FAILURE";save_report()
   print(json.dumps({"status":rep["status"],"trial":str(trial_path),"trial_sha256":sha(trial_path),
    "report":str(report_path),"report_sha256":sha(report_path),"shape_gates":rep["shape_gates"]},indent=2));return
  # Exact local bone and self screens use exact Float32 world geometry transformed to body-local coordinates.
  Plocal=(Pfit.astype(np.float64)-pos[BODY])@M[BODY].T-off[BODY]
  precs=exact._records(Plocal,PF)
  bhit=exact._audit_pair(precs,brecs,same_surface=False)
  bonepairs=[{"child_face":int(i),"component":int(PC[int(i)]),"parent_face":int(PP[int(i)]),
   "bone_triangle_id":int(bone_ids[int(j)])} for i,j in bhit["triangle_pairs"]]
  shit=exact._audit_pair(precs,precs,same_surface=True)
  selfpairs=[{"child_a":int(i),"child_b":int(j),"component_a":int(PC[int(i)]),"component_b":int(PC[int(j)]),
   "parent_a":int(PP[int(i)]),"parent_b":int(PP[int(j)])} for i,j in shit["triangle_pairs"]]
  orec=exact._records(Plocal,F0[outside])
  orec=[(t,lo,hi,len(PF)+int(i),ids0) for t,lo,hi,i,ids0 in orec]
  nhit=exact._audit_record_pairs(exact._aabb_candidate_pairs(precs,orec,same_surface=True),same_surface=True)
  neighborpairs=[{"child_face":int(i),"unchanged_original_face":int(outside[int(j)-len(PF)]),
   "component":int(PC[int(i)]),"parent_face":int(PP[int(i)])} for i,j in nhit["triangle_pairs"]]
  pointbad=[]
  for row in np.unique(PF.reshape(-1)):
   x=exact.point_location(tuple(Fraction.from_float(float(v)) for v in Plocal[int(row)]),brecs)
   if x["location"]!="outside":pointbad.append({"global_node":int(row),"location":x["location"],"crossings":x["crossings"],"ray":x["ray"]})
  bonepass=bhit["count"]==0 and not pointbad;selfpass=shit["count"]==0 and nhit["count"]==0
  rep["exact_local_gates"]={
   "target_bone":{"status":"PASS" if bonepass else "FAIL","intersection_count":bhit["count"],
    "aabb_candidate_pairs":bhit["aabb_candidate_pairs"],"witnesses":bonepairs[:256],
    "nonoutside_point_locations":pointbad[:256],"predicate":"exact rational triangle intersections and exact parity against actual 392-face closed 109/114 bone"},
   "patch_self":{"status":"PASS" if shit["count"]==0 else "FAIL","intersection_count":shit["count"],
    "aabb_candidate_pairs":shit["aabb_candidate_pairs"],"witnesses":selfpairs[:256]},
   "patch_vs_unchanged_skin":{"status":"PASS" if nhit["count"]==0 else "FAIL","intersection_count":nhit["count"],
    "aabb_candidate_pairs":nhit["aabb_candidate_pairs"],"allowed_shared_edge_or_vertex_pairs":nhit["allowed_shared_vertex_or_edge_pairs"],
    "witnesses":neighborpairs[:256]}}
  ok=bonepass and selfpass
  rep["status"]="LOCAL_CONSTRUCTION_GATES_PASS_UNADMITTED" if ok else "REJECTED_EXACT_LOCAL_BONE_OR_SELF_GATE"
  rep["checks"].update({"shape_gate":True,"target_bone_gate":bonepass,"patch_self_gate":selfpass,
   "all_local_construction_gates_pass":ok,"geometry_admitted":False,"physical_steps":0,"native_admission":False})
  rep["limitations"]=["One private construction trial; full-body and all-bank audits were not run.",
   "Exact gates cover target bone, inter-patch self intersections, and changed patch against unchanged neighboring skin.",
   "No physical/native admission or accepted status is claimed."]
  save_report()
  print(json.dumps({"status":rep["status"],"trial":str(trial_path),"trial_sha256":sha(trial_path),
   "report":str(report_path),"report_sha256":sha(report_path),"shape_gates":rep["shape_gates"],
   "exact_local_gates":rep["exact_local_gates"],"elapsed_seconds":rep["elapsed_seconds"]},indent=2,allow_nan=False))
 except Exception as e:
  rep.update({"status":"FAILED_BOUNDED_LOCAL_TRIAL","failure":repr(e),"traceback":traceback.format_exc()})
  save_report()
  print(json.dumps({"status":rep["status"],"failure":repr(e),"report":str(report_path),
   "report_sha256":sha(report_path),"trial_exists":trial_path.exists(),"elapsed_seconds":rep["elapsed_seconds"]},indent=2))
  raise

if __name__=="__main__":main()

