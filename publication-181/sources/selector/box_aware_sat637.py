"""Bound-aware finite-pair SAT soft routing for the frozen 603 construction fit."""
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
D=Path("/Users/numi/Work/numi-human-connected-surface-069")
SAT_PATH=D/"D/surface-clearance-546/pair-sat-routing-551/pair_sat_routing_551.py"
SAT_SHA="27431e9ed3e92886394b59f1156a743e508fba8c81cf8b1ff94a7d80e3ef201f"
BOX_PATH=D/"D/surface-clearance-546/directional-source-bounds-636/directional_bounds636_v2.py"
BOX_SHA="fb3e9cda895b086cf2f5f15ca28f35b43e84c912c73db443e3c7d04cacbd10a8"
RUN603=D/"D/surface-clearance-546/component-scp-area-backtrack-603/frozen-v1/run_joint_full_coordinate_fit603.py"
RUN603_SHA="38c3dfa9ddd961250111c87f56d6184075bde43be07c46e09d239f057b06d692"
OBJ458=D/"D/D/fixed-bone-outward-plane-458/fixed_bone_outward_plane_458.py"
OBJ458_SHA="056c8e60dad54f92a9c3ab69498dbfac0dc2c0152be272eadea2a58a3b6ea0e0"
OWNER408=D/"D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
OWNER408_SHA="5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def load(name,path,digest):
 if sha(path)!=digest:raise RuntimeError(f"source pin mismatch: {path}")
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
sat=load("sat551_637",SAT_PATH,SAT_SHA);box=load("bounds636_637",BOX_PATH,BOX_SHA)
INSTALLS=[]
def choose_pair_sat_plane_with_bounds(skin,bone,nodes,support,lower,upper,tol=2e-9):
 skin,bone=sat._validated_triangles(skin,bone); nodes=np.asarray(nodes,dtype=np.int64)
 scope=np.asarray(support,dtype=np.int64);lo=np.asarray(lower,dtype=np.float64).reshape(-1);hi=np.asarray(upper,dtype=np.float64).reshape(-1)
 if nodes.shape!=(3,) or len(np.unique(nodes))!=3 or np.any(nodes<0):raise ValueError("bad pair nodes")
 if scope.ndim!=1 or len(np.unique(scope))!=len(scope) or np.any(scope<0):raise ValueError("bad support")
 if lo.shape!=(3*len(scope),) or hi.shape!=lo.shape or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo>hi):raise ValueError("bad bounds")
 if np.any(lo>1e-12) or np.any(hi< -1e-12):raise ValueError("bounds must contain zero")
 loc={int(n):i for i,n in enumerate(scope)}
 try:ix=np.array([loc[int(n)] for n in nodes],dtype=np.int64)
 except KeyError as e:raise ValueError(f"pair node outside support {e.args[0]}")
 lo=lo.reshape(-1,3)[ix];hi=hi.reshape(-1,3)[ix];tol=float(tol)
 if not np.isfinite(tol) or tol<0:raise ValueError("bad tolerance")
 choices=[]
 for a in sat._signed_sat_axes(skin,bone):
  n=np.asarray(a["normal_world_unit"],dtype=np.float64);q=float(np.max(bone@n))
  margin=float(sat._float32_world_m_emission_margin_mm(skin,bone,n));rhs=q+margin-skin@n
  maxmove=np.sum(np.where(n[None,:]>=0,hi*n[None,:],lo*n[None,:]),axis=1);unmet=np.maximum(rhs-maxmove,0.)
  feasible=bool(float(unmet.max(initial=0.))<=tol);positive=np.maximum(rhs,0.)
  stable=(tuple(np.round(n,14).tolist()),str(a["axis_source"]),int(a["axis_index"]),int(a["axis_sign_order"]),int(a["extra_sign"]))
  key=(0 if feasible else 1,0. if feasible else float(unmet.max(initial=0.)),0. if feasible else float(unmet.sum()),float(positive.max(initial=0.)),float(positive.sum()),stable)
  choices.append((key,a,n,q,margin,rhs,maxmove,unmet,feasible))
 _,a,n,q,margin,rhs,maxmove,unmet,feasible=min(choices,key=lambda x:x[0])
 return {"normal_world_unit":n.copy(),"plane_point_mm":bone[int(np.argmax(bone@n))].copy(),"float32_ulp_margin_mm":margin,
  "corner_signed_distance_mm":skin@n-q,"corner_lower_mm":rhs.copy(),"positive_corner_deficits_mm":np.maximum(rhs,0.),
  "axis_strategy":"finite_pair_sat_box_aware_soft_target_637","axis_source":str(a["axis_source"]),
  "axis_index":int(a["axis_index"]),"axis_sign":int(a["axis_sign"]),"axis_sign_order":int(a["axis_sign_order"]),
  "axis_extra_sign":int(a["extra_sign"]),"candidate_axis_count":len(choices),"box_corner_max_progress_mm":maxmove,
  "box_corner_unmet_mm":unmet,"box_feasible":feasible,"box_max_unmet_mm":float(unmet.max(initial=0.)),
  "box_sum_unmet_mm":float(unmet.sum()),"whole_bank_clearance_certificate":False,
  "shared_node_axis_compatibility_certified":False}
def arg(flag,default=None):
 if sys.argv.count(flag)>1:raise ValueError(f"duplicate {flag}")
 if flag not in sys.argv:return default
 i=sys.argv.index(flag)
 if i+1>=len(sys.argv):raise ValueError(f"missing {flag} value")
 return sys.argv[i+1]
def _prefix(support,lo,hi,n):
 lo=np.asarray(lo,dtype=np.float64).reshape(-1,3).copy();hi=np.asarray(hi,dtype=np.float64).reshape(-1,3).copy()
 fixed=np.asarray(support,dtype=np.int64)<int(n);lo[fixed]=0;hi[fixed]=0
 return lo.reshape(-1),hi.reshape(-1),int(fixed.sum())
def install(owner,objective,crossing,source_ref_mm,base_sha,source_sha,prefix=0):
 if getattr(objective,"_sat637_installed",False):raise ValueError("SAT637 already installed")
 for p,h in [(RUN603,RUN603_SHA),(OBJ458,OBJ458_SHA),(OWNER408,OWNER408_SHA)]: 
  if sha(p)!=h:raise RuntimeError(f"pinned owner changed: {p}")
 old_build=objective.build_fixed_bone_outward_plane_rows;old_plane=objective.outward_plane_for_pair;old_bounds=owner.source_safe_trust_bounds
 trust=float(arg("--trust-mm","4"));cap=float(arg("--source-cap-mm","12"));source=np.asarray(source_ref_mm,dtype=np.float64)
 if not np.isfinite(source).all() or source.ndim!=2 or source.shape[1]!=3:raise ValueError("invalid composed source")
 state={"selected":[],"bounds_calls":[],"base_candidate_sha256":base_sha,"source_base_sha256":source_sha,
  "source_reference_sha256":hashlib.sha256(np.ascontiguousarray(source,dtype="<f8").tobytes()).hexdigest(),
  "trust_mm":trust,"source_cap_mm":cap,"fixed_prefix_nodes":int(prefix)}
 def bounds(points,src,support,anchors,t,c):
  oldlo,oldhi,oldr=old_bounds(points,src,support,anchors,t,c)
  lo,hi,br=box.directional_bounds(points,src,support,anchors,t,c)
  oldlo,oldhi,_=_prefix(support,oldlo,oldhi,prefix);lo,hi,nfixed=_prefix(support,lo,hi,prefix)
  if np.any(lo>oldlo+2e-10) or np.any(hi<oldhi-2e-10):raise ValueError("636 box does not contain old 408 box")
  if np.asarray(src).shape!=source.shape or np.max(np.abs(np.asarray(src)-source),initial=0)>2e-9:raise ValueError("source reference mismatch")
  sig=hashlib.sha256(np.ascontiguousarray(np.r_[np.asarray(support,dtype="<i8"),lo.astype("<f8"),hi.astype("<f8")]).tobytes()).hexdigest()
  if state["bounds_calls"] and sig != state["bounds_calls"][0]["signature_sha256"]:
   raise ValueError("actual 408 solver bounds/support order differ from selector bounds")
  state["bounds_calls"].append({"signature_sha256":sig,"support_count":len(support),"support_rows_sha256":hashlib.sha256(np.ascontiguousarray(np.asarray(support,dtype="<i8")).tobytes()).hexdigest(),"lower_sha256":hashlib.sha256(np.ascontiguousarray(lo,dtype="<f8").tobytes()).hexdigest(),"upper_sha256":hashlib.sha256(np.ascontiguousarray(hi,dtype="<f8").tobytes()).hexdigest(),"fixed_prefix_count":nfixed,"old408_cube_contained":True,"directional636":br})
  r=dict(oldr);r.update({"bound_semantics":"636-v2 asymmetric box inside unchanged trust/source balls","maximum_component_bound_mm":float(np.max(np.maximum(np.abs(lo),np.abs(hi)),initial=0)),
    "original_408_maximum_component_bound_mm":oldr.get("maximum_component_bound_mm"),"directional_bounds_636":br,"old408_cube_contained":True})
  return lo,hi,r
 owner.source_safe_trust_bounds=bounds
 def build(points,faces,support,anchors,pairs,bones,total_weight=1.):
  p=np.asarray(points,dtype=np.float64);f=np.asarray(faces,dtype=np.int64);scope=np.asarray(support,dtype=np.int64);pairs=list(pairs)
  lo,hi,br=bounds(p,source,scope,anchors,trust,cap);selected=[];seen=set()
  for pair in pairs:
   ident=(int(pair["face_row"]),int(pair["body"]),int(pair["bone_id"]),int(pair["triangle_id"]))
   if ident in seen:raise ValueError(f"duplicate crossing pair {ident}")
   seen.add(ident);face=ident[0]
   if face<0 or face>=len(f):raise ValueError("crossing face outside topology")
   bone_tri=np.asarray(bones[(ident[1],ident[2],ident[3])],dtype=np.float64)
   axis=choose_pair_sat_plane_with_bounds(p[f[face]],bone_tri,f[face],scope,lo,hi)
   axis.update({"identity":ident,"node_rows":f[face].copy(),"skin_triangle_mm":p[f[face]].copy(),"bone_triangle_mm":bone_tri.copy()})
   selected.append(axis)
  pos={"i":0}
  def plane(skin,bone):
   i=pos["i"]
   if i>=len(selected):raise ValueError("458 plane inventory overrun")
   chosen=selected[i]
   if not np.array_equal(np.asarray(skin,dtype=np.float64),chosen["skin_triangle_mm"]):
    raise ValueError("458 skin triangle order/identity differs from selected SAT record")
   if not np.array_equal(np.asarray(bone,dtype=np.float64),chosen["bone_triangle_mm"]):
    raise ValueError("458 fixed bone triangle differs from selected SAT record")
   pos["i"]+=1;return chosen
  objective.outward_plane_for_pair=plane
  try:r=old_build(points,faces,support,anchors,pairs,bones,total_weight=total_weight)
  finally:objective.outward_plane_for_pair=old_plane
  if pos["i"]!=len(selected):raise ValueError("458 plane inventory mismatch")
  for group in ("metadata","guard_metadata"):
   for rec in r[group]:
    s=selected[int(rec["pair_index"])]
    rec.update({"plane_selection":"finite_pair_sat_box_aware_soft_target_637","sat_axis_source":s["axis_source"],
     "sat_axis_index":s["axis_index"],"sat_axis_sign":s["axis_sign"],"axis_individually_box_feasible":s["box_feasible"],
     "axis_box_max_unmet_mm":s["box_max_unmet_mm"],"target_role":"soft positive-corner slack or existing nonpositive guard",
     "shared_node_compatibility":"not certified"})
  state.update({"selected":selected,"bounds_receipt":br,"anchors":np.asarray(anchors,dtype=np.int64).tolist(),"support_rows":scope.tolist(),"soft_row_count":int(r["diagnostics"]["soft_row_count"]),
    "soft_pair_count":int(r["diagnostics"]["soft_pair_count"]),"soft_weight_mass":float(r["weights"].sum()),"corner_guard_count":len(r["lower"])})
  return r
 objective.build_fixed_bone_outward_plane_rows=build;objective._sat637_installed=True
 state.update({"soft_objective_preserved":True,"positive_rows_hardened":False,"native_admission":False,"accepted_physical_steps":0})
 INSTALLS.append(state);return state
def install_from_cli(owner,objective,crossing):
 base=Path(arg("--base-candidate")).resolve();src=Path(arg("--source-base-npz")).resolve()
 c=np.load(base,allow_pickle=False)
 try:ref,receipt=owner.compose_source_reference(c,src)
 finally:c.close()
 st=install(owner,objective,crossing,np.asarray(ref,dtype=np.float64)*1000,sha(base),sha(src),int(arg("--fixed-prefix-nodes","0")))
 st["source_composition_receipt"]=receipt;st["inputs"]={"base_candidate":{"path":str(base),"sha256":sha(base)},"source_base":{"path":str(src),"sha256":sha(src)}}
 return st
def write_result(state,outdir):
 out=Path(outdir);out.mkdir(parents=True,exist_ok=True);rows=state.get("selected",[])
 npz=out/"pair-sat-box-routing-637.npz"
 np.savez_compressed(npz,pair_identity=np.asarray([r["identity"] for r in rows],dtype=np.int64).reshape(-1,4),
  nodes=np.asarray([r["node_rows"] for r in rows],dtype=np.int64).reshape(-1,3),
  normals=np.asarray([r["normal_world_unit"] for r in rows],dtype=np.float64).reshape(-1,3),
  lower=np.asarray([r["corner_lower_mm"] for r in rows],dtype=np.float64).reshape(-1,3),
  unmet=np.asarray([r["box_corner_unmet_mm"] for r in rows],dtype=np.float64).reshape(-1,3),
  feasible=np.asarray([r["box_feasible"] for r in rows],dtype=bool),
  axis_source=np.asarray([r["axis_source"] for r in rows],dtype="U32"),
  axis_index=np.asarray([r["axis_index"] for r in rows],dtype=np.int64),axis_sign=np.asarray([r["axis_sign"] for r in rows],dtype=np.int64))
 bad=[r for r in rows if not r["box_feasible"]]
 anchors=set(map(int,state.get("anchors",[])));by_node={}
 for row in rows:
  for corner,node in enumerate(row["node_rows"]):
   if row["corner_lower_mm"][corner]>1e-12 and int(node) not in anchors:
    by_node.setdefault(int(node),[]).append(row["normal_world_unit"])
 shared=[]
 for node,vecs in by_node.items():
  if len(vecs)>1:
   cosine=np.asarray(vecs,dtype=np.float64)@np.asarray(vecs,dtype=np.float64).T
   shared.append({"node_row":node,"positive_soft_row_count":len(vecs),"minimum_pairwise_normal_cosine":float(cosine[np.triu_indices(len(vecs),1)].min())})
 report={"schema":"numi.human.box_aware_sat_fit_637.v1","pair_count":len(rows),"individually_box_feasible_count":len(rows)-len(bad),
  "individually_box_infeasible_count":len(bad),"infeasible_pairs":[{"identity":list(r["identity"]),"axis_source":r["axis_source"],
  "axis_index":r["axis_index"],"axis_sign":r["axis_sign"],"max_unmet_mm":r["box_max_unmet_mm"],"unmet_corners_mm":r["box_corner_unmet_mm"].tolist()} for r in bad],
  "axis_source_counts":{str(k):int(v) for k,v in zip(*np.unique(np.asarray([r["axis_source"] for r in rows],dtype=str),return_counts=True))},
   "shared_node_diagnostic":{"nodes_with_multiple_positive_rows":len(shared),"nodes_with_opposed_positive_rows":sum(x["minimum_pairwise_normal_cosine"]<0. for x in shared),
   "nodes_with_cosine_below_minus_half":sum(x["minimum_pairwise_normal_cosine"]<-.5 for x in shared),
   "worst_nodes":sorted(shared,key=lambda x:x["minimum_pairwise_normal_cosine"])[:32],
   "semantics":"pairwise row-normal opposition is a soft-target conflict indicator, not an infeasibility proof"},
  "soft_row_count":state.get("soft_row_count"),"soft_pair_count":state.get("soft_pair_count"),"soft_weight_mass":state.get("soft_weight_mass"),
  "bounds_receipt":state.get("bounds_receipt"),"bounds_calls":state.get("bounds_calls"),"inputs":state.get("inputs"),
  "base_candidate_sha256":state["base_candidate_sha256"],"source_base_sha256":state["source_base_sha256"],
  "source_reference_sha256":state["source_reference_sha256"],"selected_npz":str(npz),"selected_npz_sha256":sha(npz),
  "policy":{"positive_458_rows_remain_soft":True,"no_positive_hard_rows_added":True,"nonpositive_458_guards_rebuilt":True,
   "same_trust_source_area_edge_self_anchor_and_exact_gates":True,"joint_shared_node_compatibility_certified":False,
   "whole_bank_clearance_certificate":False,"native_admission":False,"accepted_physical_steps":0}}
 path=out/"pair-sat-box-routing-637.json";path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n");return path,npz
