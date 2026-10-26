"""Bounded owner-candidate cut generation from matched selected2326 construction state."""
from pathlib import Path
import sys,json,importlib.util
N=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("orchestrator546",N/"continue_surface_round546.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
PY=sys.executable
STATE=N/"raw-lp-trial-005-full-plane/owner-final/state.json"
CHECK=N.parent/"contained-clearance-545/run_candidate_checks545.py"
CAPTURE=N/"raw-lp-direction-capture-546/capture_raw_lp_directions546.py"
def refresh(previous,prior_path):
 st=json.loads(STATE.read_text());prior=json.loads(prior_path.read_text())
 a=[PY,str(N/"refresh_surface_constraints546.py")]
 values={"candidate":st["candidate"],"audit130":st["point_summary"],"audit159":st["surface_report"],"query-dir":str(Path(st["point_summary"]).parent/"query"),"ancestry-parent-candidate":st["candidate"],"prior-candidate":prior["candidate"],"prior-surface-report":prior["surface_report"],"prior-guard-candidate":prior["candidate"],"prior-guard-npz":prior["guard_npz"],"prior-guard-report":prior["guard_report"],"prior-seed-pairs":prior["seed_pairs_path"],"prior-seed-report":prior["seed_report"],"failed-seed-candidate":str(previous/"fit/candidate.npz"),"failed-seed-audit130":str(previous/"first-check/audit-130/summary.json"),"failed-seed-pairs":str(previous/"first-check/audit-130/surface-self-pairs.npy"),"output-dir":str(previous/"constraints")}
 for k,v in values.items():a.extend(["--"+k,v])
 m.run_stage(a,previous/"refresh-stage",slot=True,source_paths=(N/"refresh_surface_constraints546.py",N/"refresh_guards_multi_parent_546.py"),timeout=180)
 return previous/"constraints/constraint-context.json"
def main():
 previous=N/"round-010-full-plane-trust4";prior=N/"round-010-selected-2326/prepare/constraint-context.json"
 template=json.loads((previous/"fit-argv.json").read_text())
 records=[]
 assert m.sha(CAPTURE)=="850893a0cc3271fee42e8342e8d73c786cc396eb2f1df4d267c28556c2ace073"
 for i in range(11,14):
  context=refresh(previous,prior);ctx=json.loads(context.read_text())
  t=N/f"round-{i:03d}-full-plane-trust4";t.mkdir()
  a=template.copy();a[a.index("--output")+1]=str(t/"fit")
  for flag,key in [("--self-pairs","seed_pairs_path"),("--fixed-boundary-npz","guard_npz"),("--fixed-boundary-report","guard_report")]:a[a.index(flag)+1]=ctx[key]
  (t/"fit-argv.json").write_text(json.dumps(a,indent=2)+"\n")
  argv=["/usr/bin/env","NUMI_RAW_LP_DIRECTION_FIT_MODE_546=full-plane","NUMI_RAW_LP_DIRECTION_CAPTURE_DIR_546="+str(t/"raw-directions"),*a]
  m.run_stage(argv,t/"fit-stage",slot=True,source_paths=(CAPTURE,N/"run_surface_only_full_coordinate_fit546.py",N.parent/"contained-clearance-545/component_lp_active545.py"),timeout=600)
  try:
   m.run_stage([PY,str(CHECK),"--candidate",str(t/"fit/candidate.npz"),"--baseline-state",str(STATE),"--trust-mm","4","--output-dir",str(t/"first-check")],t/"first-check-stage",slot=False,source_paths=(CHECK,),timeout=300)
  except RuntimeError:
   r=json.loads((t/"first-check/report.json").read_text())
   if r.get("error")!="exact self intersections":raise
  r=json.loads((t/"first-check/report.json").read_text())
  records.append({"trial":str(t),"context":str(context),"status":r["status"],"self_pairs":r.get("self_pairs"),"contained_pairs":r.get("contained_pairs"),"surface_pairs":r.get("surface_pairs"),"candidate_sha256":r["candidate_sha256"],"native_admission":False,"accepted_physical_steps":0})
  (N/f"owner-cut-generation-through-{i:03d}.json").write_text(json.dumps(records,indent=2)+"\n")
  print(json.dumps(records[-1]),flush=True)
  if r.get("self_pairs")==0:return
  previous=t;prior=context
if __name__=="__main__":main()
