#!/usr/bin/env python3
"""Apply exact-self-witness 631 component retention to the 638 trial.

Retain only changed coordinate components untouched by any exact self-pair
witness. Rebuild from the pinned valid baseline payload because the fit rewrote
local witness/depth metadata; trial non-point payload is deliberately discarded.
This is a proposal only; fresh full 545 checks remain mandatory.
"""
from pathlib import Path
import hashlib, importlib.util, json
import numpy as np

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
BASE=ROOT/"D/surface-clearance-546/component-sat-selector-638/outward-boneface-filter-640/root-bank43-selfcut-001/retained-631v2-component-5883-only-001/candidate.npz"
BASE_STATE=ROOT/"D/surface-clearance-546/component-sat-selector-638/outward-boneface-filter-640/root-bank43-selfcut-001/retained-631v2-component-5883-only-001/final-546/state.json"
TRIAL=ROOT/"D/surface-clearance-546/component-sat-selector-638/fit-from-663-001/root-unweighted-fit-001/candidate.npz"
SELF_SUMMARY=ROOT/"D/surface-clearance-546/component-sat-selector-638/fit-from-663-001/root-unweighted-fit-001/full-check-545/audit-130/summary.json"
SELF_PAIRS=ROOT/"D/surface-clearance-546/component-sat-selector-638/fit-from-663-001/root-unweighted-fit-001/full-check-545/audit-130/surface-self-pairs.npy"
HELPER=ROOT/"D/surface-clearance-546/component-merit-retention-631/select_components631_v2.py"
OUT=ROOT/"D/surface-clearance-546/component-sat-selector-638/fit-from-663-001/root-unweighted-fit-001/retained-631-selfsafe-001/result"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""): h.update(block)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
 if OUT.exists():raise FileExistsError(OUT)
 st=json.loads(BASE_STATE.read_text())
 if Path(st["candidate"]).resolve()!=BASE.resolve() or st["candidate_sha256"]!=sha(BASE):
  raise ValueError("663 baseline state does not pin the selected baseline")
 summary=json.loads(SELF_SUMMARY.read_text())
 trial_sha=sha(TRIAL)
 if summary.get("candidate_sha256")!=trial_sha or summary.get("complete_surface_audit") is not True or summary.get("status")!="STOPPED_AT_SURFACE_GATE":
  raise ValueError("self summary is incomplete or not pinned to this trial")
 pairs=np.load(SELF_PAIRS,allow_pickle=False)
 if pairs.ndim!=2 or pairs.shape[1]!=2 or len(pairs)!=summary.get("exact_PLC_surface_pairs"):
  raise ValueError("exact self-pair inventory mismatch")
 spec=importlib.util.spec_from_file_location("select_components631_v2",HELPER)
 helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 b,t=load(BASE),load(TRIAL)
 if set(b)!=set(t):raise ValueError("candidate key inventory differs")
 faces=b["source_O_faces"]
 if faces.dtype.kind not in "iu" or faces.ndim!=2 or faces.shape[1]!=3 or not helper.same(faces,t["source_O_faces"]):
  raise ValueError("source topology changed")
 x,y=b["points_world_m"],t["points_world_m"]
 if x.dtype!=np.float32 or y.dtype!=np.float32 or x.shape!=y.shape:
  raise ValueError("point payload must remain matching Float32")
 anchors=np.asarray(b["typed_coarse_anchor_output_rows_018"],dtype=np.int64)
 if len(anchors)!=237 or not helper.same(x[anchors],y[anchors]):
  raise ValueError("typed anchor contract failed")
 labels,owners=helper.labels_for_changes(x,y,faces)
 implicated=set();pair_map=[]
 for i,j in pairs.astype(np.int64):
  comps=sorted({int(owners[int(i)],),int(owners[int(j)])}-{ -1})
  if not comps:raise ValueError(f"self witness {(int(i),int(j))} touches no changed component")
  implicated.update(comps);pair_map.append({"faces":[int(i),int(j)],"components":comps})
 all_components=sorted(int(v) for v in np.unique(labels) if v>=0)
 kept=sorted(set(all_components)-implicated)
 rows=np.flatnonzero(np.isin(labels,kept))
 out={k:v.copy() for k,v in b.items()}
 out["points_world_m"][rows]=y[rows]
 if not helper.same(out["points_world_m"][anchors],x[anchors]):raise ValueError("retained anchors moved")
 if any(not helper.same(out[k],b[k]) for k in b if k!="points_world_m"):
  raise AssertionError("baseline non-point payload was not preserved")
 OUT.mkdir(parents=True)
 candidate=OUT/"candidate.npz"
 np.savez_compressed(candidate,**out)
 differing_nonpoint=[k for k in sorted(b) if k!="points_world_m" and not helper.same(b[k],t[k])]
 report={
  "schema":"numi.human.selfsafe_component_retention638.v1",
  "status":"CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED",
  "baseline_state_path":str(BASE_STATE),"baseline_state_sha256":sha(BASE_STATE),
  "baseline_candidate_path":str(BASE),"baseline_candidate_sha256":sha(BASE),
  "trial_candidate_path":str(TRIAL),"trial_candidate_sha256":trial_sha,
  "self_summary_path":str(SELF_SUMMARY),"self_summary_sha256":sha(SELF_SUMMARY),
  "self_pairs_path":str(SELF_PAIRS),"self_pairs_sha256":sha(SELF_PAIRS),
  "self_pair_count":int(len(pairs)),"selector631_helper_path":str(HELPER),"selector631_helper_sha256":sha(HELPER),
  "changed_component_count":len(all_components),"implicated_component_ids":sorted(implicated),
  "retained_component_ids":kept,"retained_point_rows":int(len(rows)),
  "self_pair_component_map":pair_map,"typed_anchor_count":int(len(anchors)),
  "anchors_bitwise_baseline":True,"nonpoint_payload_bitwise_baseline":True,
  "trial_nonpoint_fields_discarded":differing_nonpoint,
  "candidate_path":str(candidate),"candidate_sha256":sha(candidate),
  "retained_coordinates_only_from_trial":True,
  "exact_full545_required":True,"native_admission":False,"accepted_physical_steps":0,
  "limitation":"Component retention is a proposal; exact emitted-F32 self, containment, all-bank, area, edge and source/trust checks remain mandatory."
 }
 (OUT/"retention-report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({k:report[k] for k in ("status","self_pair_count","changed_component_count","implicated_component_ids","retained_component_ids","retained_point_rows","candidate_sha256","trial_nonpoint_fields_discarded")},separators=(",",":")))
if __name__=="__main__":main()
