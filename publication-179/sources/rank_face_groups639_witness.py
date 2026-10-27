"""Bounded hard-row feasibility ranking for a few complete selected skin-face groups.

This is a linear construction diagnostic on captured 408 rows. It does not
certify emitted geometry, area/edge nonlinearity, or whole-bank clearance.
"""
from __future__ import annotations
import hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

BASE=Path("/Users/numi/Work/numi-human-connected-surface-069")
ROOT=BASE/"D/surface-clearance-546"
CAPTURE=ROOT/"hard-lp-system-capture-639/capture-675-640-002/fit-lp-system-639"
FIT=ROOT/"hard-lp-system-capture-639/capture-675-640-002/fit"
SELECTION=ROOT/"component-sat-selector-638/outward-boneface-filter-640/selector-675-reject-opposed-001/selection.npz"
SELECTOR_REPORT=SELECTION.with_name("report.json")
CROSSING=ROOT/"component-sat-selector-638/fit-from-675-001/fit.sat-crossing-inputs-458.npz"
BASE_CANDIDATE=ROOT/"connected-topology-580/run-010-from689/root-combined-001/candidate.npz"
LP_OWNER=BASE/"D/D/component-lp-decomposition-485/component_lp_decomposition_485.py"
LP_SHA="b8632e0734ff0163e854b9c0c5fbfe33c6d12c7b17cbb7935c5028c8df78ef74"
SAT_OWNER=ROOT/"pair-sat-routing-551/pair_sat_routing_551.py"

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def load_lp_owner():
 if sha(LP_OWNER)!=LP_SHA:raise RuntimeError("485 source pin mismatch")
 spec=importlib.util.spec_from_file_location("capture_rank_485",LP_OWNER)
 m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def _typed_identity(face, body, bone_id, triangle):
 return [int(face), int(body), int(bone_id), int(triangle)]


def _target_node_delta_records(delta_output_mm,node_pos,pair_rows):
 nodes=sorted({int(node) for row in pair_rows for node in row[1]})
 d=np.asarray(delta_output_mm,dtype=np.float64).reshape(-1)
 return nodes,[d[3*node_pos[node]:3*node_pos[node]+3].tolist() for node in nodes]


def _pair_count(pair_rows):
 return len({tuple(int(v) for v in np.asarray(row[0]).reshape(-1)) for row in pair_rows})


def _hard_solve(group_name, pair_rows, A, b, M, lo, hi, AM, labels, groups, a_labels, m_labels):
 # pair_rows: list of (pair identity, 3 node rows, normal, skin tri, bone tri, rhs)
 nout=M.shape[0];nctrl=M.shape[1]
 rr=[];cc=[];vv=[];rhs=[]
 scope=scope_rows
 node_pos={int(n):i for i,n in enumerate(scope)}
 for ident,nodes,n,skin,bone,rowrhs in pair_rows:
  for corner,node in enumerate(nodes):
   if int(node) not in node_pos:raise ValueError(f"target node outside captured output scope: {node}")
   oi=node_pos[int(node)]
   for axis,val in enumerate(n):
    if val:
     rr.append(len(rhs));cc.append(3*oi+axis);vv.append(float(val))
   rhs.append(float(rowrhs[corner]))
 P=sparse.coo_matrix((vv,(rr,cc)),shape=(len(rhs),nout)).tocsr()
 PM=(P@M).tocsr()
 selected_comps=set()
 target_cross_rows=[]
 for ri in range(PM.shape[0]):
  labs=set(int(x) for x in labels[PM.indices[PM.indptr[ri]:PM.indptr[ri+1]]] if x>=0)
  if not labs:
   if rhs[ri] > 1e-8:
    return {"group":group_name,"status":"INFEASIBLE_FIXED_TARGET_ROW","target_row":ri,"rhs_mm":rhs[ri],
            "pair_count":_pair_count(pair_rows),"constraint_rows":len(rhs)}
   continue
  selected_comps.update(labs)
  target_cross_rows.append(sorted(labs))
 if not selected_comps:
  return {"group":group_name,"status":"NO_MOVABLE_TARGET_CONTROLS","pair_count":_pair_count(pair_rows),"constraint_rows":len(rhs)}
 controls=np.concatenate([groups[i] for i in sorted(selected_comps)]).astype(np.int64)
 controls=np.unique(controls)
 hard_rows=np.flatnonzero(np.isin(a_labels,np.fromiter(selected_comps,dtype=np.int64)))
 out_rows=np.flatnonzero(np.isin(m_labels,np.fromiter(selected_comps,dtype=np.int64)))
 if len(controls)>10000:raise ValueError("bounded target group unexpectedly spans >10000 controls")
 Ah=AM[hard_rows][:,controls].tocsr()
 Ph=PM[:,controls].tocsr()
 Mloc=M[out_rows][:,controls].tocsr()
 # d=Mq; original hard rows and new target rows are all lower bounds.
 Aub=sparse.vstack((-Ah,-Ph,Mloc,-Mloc),format="csr")
 bub=np.concatenate((-b[hard_rows],-np.asarray(rhs),hi[out_rows],-lo[out_rows]))
 t0=time.monotonic()
 result=linprog(np.zeros(len(controls)),A_ub=Aub,b_ub=bub,
                bounds=[(None,None)]*len(controls),method="highs",
                options={"time_limit":12.0,"primal_feasibility_tolerance":1e-8,
                         "dual_feasibility_tolerance":1e-8,"presolve":True})
 elapsed=time.monotonic()-t0
 out={"group":group_name,"status":("FEASIBLE_LINEAR_HARD_SYSTEM" if result.status==0 else
      "INFEASIBLE_LINEAR_HARD_SYSTEM" if result.status==2 else "UNKNOWN_OR_LIMIT"),
      "solver_status":int(result.status),"solver_message":str(result.message),
      "solver_seconds":elapsed,"component_ids":sorted(selected_comps),
      "control_count":int(len(controls)),"hard_rows":int(len(hard_rows)),
      "output_bound_rows":int(2*len(out_rows)),"target_rows":int(len(rhs)),
      "pair_count":_pair_count(pair_rows),
      "max_positive_target_rhs_mm":float(max(rhs,default=0.)),
      "max_abs_target_rhs_mm":float(max((abs(x) for x in rhs),default=0.)),
      "diagnostic_only":True,"native_admission":False,"accepted_physical_steps":0,
      "claim_limit":"Feasibility of selected finite-pair corner planes against captured linear 408 hard rows and output bounds only; no C/h soft objective, emitted-F32 area, edge, self, containment, or full-185-bank gate is certified."}
 if result.status==0:
  q=np.asarray(result.x,dtype=np.float64);d=np.asarray(M[:,controls]@q).reshape(-1)
  hard_res=np.asarray(A@d).reshape(-1)-b
  tr=np.asarray(P@d).reshape(-1)-np.asarray(rhs)
  bound_low=d-lo;bound_hi=hi-d
  out.update({"minimum_all_hard_residual_mm":float(np.min(hard_res,initial=np.inf)),
   "minimum_group_target_residual_mm":float(np.min(tr,initial=np.inf)),
   "minimum_output_bound_residual_mm":float(min(np.min(bound_low,initial=np.inf),np.min(bound_hi,initial=np.inf))),
   "verified_linear_residuals_pass":bool(np.min(hard_res,initial=np.inf)>=-2e-7 and
      np.min(tr,initial=np.inf)>=-2e-7 and np.min(bound_low,initial=np.inf)>=-2e-7 and
      np.min(bound_hi,initial=np.inf)>=-2e-7)})
  out["target_node_rows"],out["target_node_displacements_mm"]=_target_node_delta_records(d,node_pos,pair_rows)
 return out

def main(output):
 global scope_rows
 meta=json.loads((CAPTURE/"capture.json").read_text())
 if meta.get("schema")!="numi.human.hard_lp_system_capture639.v1" or not meta.get("metadata_coverage",{}).get("owner_matrix"):
  raise ValueError("capture metadata incomplete")
 if json.loads((CAPTURE/"solver-outcome.json").read_text())["outcome"].get("success") is not True:
  raise ValueError("captured 408 solver call did not return successfully")
 if sha(BASE_CANDIDATE)!=meta["inputs"]["base_candidate_sha256"]:raise ValueError("capture base candidate pin mismatch")
 if sha(SELECTION)!=meta["inputs"]["selector_npz_sha256"] or sha(SELECTOR_REPORT)!=meta["inputs"]["selector_report_sha256"]:
  raise ValueError("capture selector pin mismatch")
 if sha(LP_OWNER)!=LP_SHA:raise RuntimeError("485 owner changed")
 A=sparse.load_npz(CAPTURE/"hard_A.npz").tocsr();M=sparse.load_npz(CAPTURE/"output_map_M.npz").tocsr()
 C=sparse.load_npz(CAPTURE/"clearance_C.npz").tocsr()
 with np.load(CAPTURE/"vectors.npz",allow_pickle=False) as z:
  b=z["b"].copy();lo=z["lower_output"].copy();hi=z["upper_output"].copy()
 with np.load(CAPTURE/"basis_domain.npz",allow_pickle=False) as z:
  scope_rows=z["output_scope_rows"].astype(np.int64,copy=True)
 if A.shape!=(len(b),M.shape[0]) or M.shape[0]!=len(lo) or C.shape[1]!=M.shape[0]:
  raise ValueError("captured matrices/vector dimensions do not align")
 # Reconstruct precisely the existing 485 row/control components, without changing them.
 lp=load_lp_owner();AM=(A@M).tocsr();CM=(C@M).tocsr()
 groups,labels,null=lp._components_from_rows((AM,CM,M),M.shape[1])
 a_labels=lp._row_support_labels(AM,labels,"A@M")
 m_labels=lp._row_support_labels(M,labels,"M")
 if len(groups)!=13:raise ValueError(f"captured component graph differs from solver receipt: {len(groups)}")
 with np.load(SELECTION,allow_pickle=False) as s,np.load(CROSSING,allow_pickle=False) as c,np.load(BASE_CANDIDATE,allow_pickle=False) as base:
  ident=np.asarray(s["pair_identity"],dtype=np.int64)
  nodes=np.asarray(s["pair_nodes"],dtype=np.int64)
  selected=np.asarray(s["selected"],dtype=bool)
  normals=np.asarray(s["selected_normal_world_unit"],dtype=np.float64)
  selected_sources=np.asarray(s["selected_axis_source"]).copy()
  bone_tri=np.asarray(c["fixed_triangles_world_mm"],dtype=np.float64)
  point_key=next(k for k in base.files if "points_world_m" in k)
  points=np.asarray(base[point_key],dtype=np.float64)*1000.
 sys.path.insert(0,str(SAT_OWNER.parent));import pair_sat_routing_551 as sat
 pair_records=[]
 for i in np.flatnonzero(selected):
  face,body,bone_id,triangle=map(int,ident[i]);node_rows=nodes[i];n=normals[i]
  skin=points[node_rows];bone_vertices=bone_tri[i]
  margin=float(sat._float32_world_m_emission_margin_mm(skin,bone_vertices,n))
  support=float(np.max(bone_vertices@n));rowrhs=support+margin-skin@n
  pair_records.append({"identity":_typed_identity(face,body,bone_id,triangle),"nodes":node_rows.copy(),
      "normal":n.copy(),"skin":skin.copy(),"bone":bone_vertices.copy(),"rhs":rowrhs.copy(),
      "source":str(selected_sources[i])})
 byface={}
 for x in pair_records:byface.setdefault(int(x["identity"][0]),[]).append(x)
 # Candidate diagnostic patches: complete selected pair groups, chosen for low corner demand
 # and internally similar axes across the three high-priority banks.
 groups_spec={
  "bank43_16_faces19024_19025_19027":[19024,19025,19027],
  "bank109_114_face34158":[34158],
  "bank65_99_face13996":[13996],
  "bank114_117_face40545":[40545],
  "bank64_98_face17342":[17342],
 }
 results=[]
 for name,faces in groups_spec.items():
  rows=[r for f in faces for r in byface.get(f,[])]
  if not rows:
   results.append({"group":name,"status":"NO_SELECTED_PAIRS","faces":faces});continue
  identities=[tuple(r["identity"]) for r in rows]
  # target rows include every selected pair for each requested face, not a partial triangle subset
  results.append({**_hard_solve(name,[(r["identity"],r["nodes"],r["normal"],r["skin"],r["bone"],r["rhs"]) for r in rows],
      A,b,M,lo,hi,AM,labels,groups,a_labels,m_labels),
      "faces":faces,"typed_pair_identities":[list(i) for i in identities],
      "axis_sources":[r["source"] for r in rows]})
 report={"schema":"numi.human.captured_hard_system_target_ranking639.v1",
   "capture_path":str(CAPTURE),"capture_json_sha256":sha(CAPTURE/"capture.json"),
   "captured_A_path":str(CAPTURE/"hard_A.npz"),"captured_A_sha256":sha(CAPTURE/"hard_A.npz"),
   "captured_b_sha256":meta["vectors"]["b"]["sha256"],
   "captured_M_sha256":sha(CAPTURE/"output_map_M.npz"),
   "captured_C_sha256":sha(CAPTURE/"clearance_C.npz"),
   "selector_npz_sha256":sha(SELECTION),"selector_report_sha256":sha(SELECTOR_REPORT),
   "base_candidate_sha256":sha(BASE_CANDIDATE),"component_count":len(groups),
   "component_control_counts":[int(len(x)) for x in groups],"zero_controls":int(null),
   "group_results":results,"native_admission":False,"accepted_physical_steps":0,
   "claim_limit":"These are hard linear-row feasibility probes using the captured first 408/603 LP system and its actual output bounds/map. C/h are preserved in the capture but remain the existing soft objective; no nonlinear/F32 or exact geometry acceptance is established."}
 Path(output).write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({"output":str(output),"groups":[{"group":r["group"],"status":r["status"],"components":r.get("component_ids"),"seconds":r.get("solver_seconds")} for r in results]},sort_keys=True))

scope_rows=np.empty(0,dtype=np.int64)
if __name__=="__main__":
 out=Path(sys.argv[1]) if len(sys.argv)>1 else CAPTURE.parent/"ranked-targets-001.json"
 main(out)
