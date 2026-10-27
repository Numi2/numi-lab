"""Bounded alternative 551 SAT-axis probe against captured 675 hard rows.

Construction-only LP diagnostic: no nonlinear/emitted-F32/exact clearance or
physical admission is established by a feasible result.
"""
from pathlib import Path
import hashlib, importlib.util, json, os, sys
import numpy as np
from scipy import sparse

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546")
CAP=ROOT/"hard-lp-system-capture-639/capture-675-640-002/fit-lp-system-639"
RANK=ROOT/"hard-lp-system-capture-639/rank_face_groups639_v3.py"
SEL=ROOT/"component-sat-selector-638/outward-boneface-filter-640/selector-675-reject-opposed-001/selection.npz"
SEL_REPORT=SEL.with_name("report.json")
CROSS=ROOT/"component-sat-selector-638/fit-from-675-001/fit.sat-crossing-inputs-458.npz"
BASE=ROOT/"connected-topology-580/run-010-from689/root-combined-001/candidate.npz"
SATDIR=ROOT/"pair-sat-routing-551"

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""): h.update(block)
 return h.hexdigest()

def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m

def _atomic_json(path, value):
 path=Path(path); tmp=path.with_name(path.name+".tmp")
 encoded=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n"
 tmp.write_text(encoded); os.replace(tmp,path)


def _identity_key(identity):
 return ":".join(str(int(v)) for v in identity)


def main(out_path):
 rank=load_module("rank_face_groups639_v3_for641",RANK)
 if sha(RANK)!="6a3f7712f718effe37cbf2247dfb6b3049ac01c562f75f5056d42e5d4853ceef": raise RuntimeError("rank helper source pin mismatch")
 meta=json.loads((CAP/"capture.json").read_text())
 if json.loads((CAP/"solver-outcome.json").read_text())["outcome"].get("success") is not True: raise RuntimeError("captured hard solve not successful")
 if sha(SEL)!=meta["inputs"]["selector_npz_sha256"] or sha(SEL_REPORT)!=meta["inputs"]["selector_report_sha256"]: raise RuntimeError("selector source pins differ from capture")
 A=sparse.load_npz(CAP/"hard_A.npz").tocsr(); M=sparse.load_npz(CAP/"output_map_M.npz").tocsr(); C=sparse.load_npz(CAP/"clearance_C.npz").tocsr()
 with np.load(CAP/"vectors.npz",allow_pickle=False) as z: b=z["b"].copy(); lo=z["lower_output"].copy(); hi=z["upper_output"].copy()
 with np.load(CAP/"basis_domain.npz",allow_pickle=False) as z: scope=z["output_scope_rows"].astype(np.int64,copy=True)
 if A.shape!=(len(b),M.shape[0]) or M.shape[0]!=len(lo) or len(hi)!=len(lo): raise ValueError("captured domains do not align")
 rank.scope_rows=scope
 lp=rank.load_lp_owner(); AM=(A@M).tocsr(); CM=(C@M).tocsr()
 comps,labels,null=lp._components_from_rows((AM,CM,M),M.shape[1])
 a_labels=lp._row_support_labels(AM,labels,"A@M"); m_labels=lp._row_support_labels(M,labels,"M")
 with np.load(SEL,allow_pickle=False) as s, np.load(CROSS,allow_pickle=False) as c, np.load(BASE,allow_pickle=False) as z:
  ids=s["pair_identity"].astype(np.int64,copy=True); nodes=s["pair_nodes"].astype(np.int64,copy=True); selected=s["selected"].astype(bool,copy=True)
  oldnorm=s["selected_normal_world_unit"].astype(np.float64,copy=True); oldsource=s["selected_axis_source"].copy()
  oldindex=s["selected_axis_index"].astype(np.int64,copy=True)
  fixed=c["fixed_triangles_world_mm"].astype(np.float64,copy=True)
  pk=next(k for k in z.files if "points_world_m" in k); pts=z[pk].astype(np.float64,copy=True)*1000.
 sys.path.insert(0,str(SATDIR)); import pair_sat_routing_551 as sat
 pos={int(node):i for i,node in enumerate(scope)}
 # Enumerate alternatives and first discard any whose three corner rows cannot
 # fit the captured per-coordinate output boxes even independently.
 choices={}
 axis_inventory={}
 wanted_faces=(19024,19025,19027)
 for i in np.flatnonzero(selected & np.isin(ids[:,0],wanted_faces)):
  face,body,bone_id,tri_id=map(int,ids[i]); skin=pts[nodes[i]]; bone=fixed[i]; candidates=[]
  for ax in sat._signed_sat_axes(skin,bone):
   n=np.asarray(ax["normal_world_unit"],dtype=np.float64); margin=float(sat._float32_world_m_emission_margin_mm(skin,bone,n))
   rhs=np.max(bone@n)+margin-skin@n
   maxmove=[]
   for node in nodes[i]:
    j=3*pos[int(node)]; maxmove.append(float(np.sum(np.maximum(n*lo[j:j+3],n*hi[j:j+3]))))
   box_def=np.maximum(0.,rhs-np.asarray(maxmove)); positive=np.maximum(0.,rhs)
   meta_axis={"axis_source":str(ax["axis_source"]),"axis_index":int(ax["axis_index"]),"axis_sign":int(ax["axis_sign"]),"extra_sign":int(ax["extra_sign"]),"normal":n.tolist(),"rhs_mm":rhs.tolist(),"box_maxmove_mm":maxmove,"box_deficit_mm":box_def.tolist(),"max_positive_rhs_mm":float(np.max(positive)),"sum_positive_rhs_mm":float(np.sum(positive)),"matches_638_axis":bool(np.allclose(n,oldnorm[i],atol=1e-12,rtol=0))}
   candidates.append((float(np.max(box_def)),float(np.sum(box_def)),float(np.max(positive)),float(np.sum(positive)),str(ax["axis_source"]),int(ax["axis_index"]),int(ax["axis_sign"]),int(ax["extra_sign"]),meta_axis,n,rhs,margin))
  candidates.sort(key=lambda x:x[:8])
  feasible=[x for x in candidates if x[0]<=1e-9]
  if not feasible: raise RuntimeError(f"no box-feasible SAT axis for {ids[i].tolist()}")
  axis_inventory[_identity_key(ids[i])]={"axis_count":len(candidates),"box_feasible_axis_count":len(feasible),"least_demand_candidates":[x[8] for x in feasible[:5]],"previous_638_source":str(oldsource[i]),"previous_638_index":int(oldindex[i]),"previous_638_normal":oldnorm[i].tolist()}
  best=feasible[0]; n=best[9]; rhs=best[10]
  choices[tuple(int(v) for v in ids[i])]=(nodes[i].copy(),n.copy(),pts[nodes[i]].copy(),fixed[i].copy(),rhs.copy(),best[8])
 pair_rows=[]
 for identity,(nr,n,skin,bone,rhs,axis_meta) in sorted(choices.items()): pair_rows.append((list(identity),nr,n,skin,bone,rhs))
 # One complete 3-face group and three complete single-face subsets.
 specs=[("all_three_faces",wanted_faces)] + [(f"face_{f}",(f,)) for f in wanted_faces]
 results=[]
 progress_path=Path(out_path).with_name(Path(out_path).stem+".progress.json")
 for name,faces in specs:
  subset=[r for r in pair_rows if r[0][0] in faces]
  result=rank._hard_solve(name,subset,A,b,M,lo,hi,AM,labels,comps,a_labels,m_labels)
  result["faces"]=list(faces); result["typed_pair_identities"]=[r[0] for r in subset]
  result["chosen_axes"]=[choices[tuple(r[0])][5] for r in subset]
  results.append(result)
  _atomic_json(progress_path,{"schema":"numi.human.alternative_sat_axes_full_hard_probe641.progress.v1","completed_probe_count":len(results),"total_probe_count":len(specs),"axis_inventory":axis_inventory,"group_results":results,"native_admission":False,"accepted_physical_steps":0,"claim_limit":"partial linear hard-row feasibility progress only; no nonlinear or physical admission"})
 report={"schema":"numi.human.alternative_sat_axes_full_hard_probe641.v1","capture_sha256":sha(CAP/"capture.json"),"A_sha256":sha(CAP/"hard_A.npz"),"b_sha256":meta["vectors"]["b"]["sha256"],"M_sha256":sha(CAP/"output_map_M.npz"),"base_candidate_sha256":meta["inputs"]["base_candidate_sha256"],"selector_sha256":sha(SEL),"selector_report_sha256":sha(SEL_REPORT),"sat_owner_sha256":sha(sat.SAT_PATH),"sat_router_sha256":sha(SATDIR/"pair_sat_routing_551.py"),"rank_helper_sha256":sha(RANK),"captured_component_count":len(comps),"captured_component_controls":[len(x) for x in comps],"axis_inventory":axis_inventory,"group_results":results,"native_admission":False,"accepted_physical_steps":0,"claim_limit":"Alternative finite triangle SAT halfspaces were pruned by captured output-coordinate box bounds and tested against captured 408 hard A/b and output bounds. C/h soft objective and nonlinear/emitted-F32/exact 185-bank acceptance are not certified; this is not a whole-bank clearance claim."}
 _atomic_json(out_path,report)
 print(json.dumps({"out":str(out_path),"groups":[{"group":r["group"],"status":r["status"],"seconds":r.get("solver_seconds")} for r in results]},sort_keys=True))

if __name__=="__main__": main(Path(sys.argv[1]))
