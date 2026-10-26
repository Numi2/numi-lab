"""Bounded full-plane cut generation using existing 546 construction owners."""
from pathlib import Path
import argparse,sys,json,importlib.util
N=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("orchestrator546",N/"continue_surface_round546.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
PY=sys.executable
STATE=N/"topology/round-007-wide-chain-025/state.json"
CHECK=N.parent/"contained-clearance-545/run_candidate_checks545.py"
CAPTURE=N/"raw-lp-direction-capture-546/capture_raw_lp_directions546.py"
def refresh(trial,prior_path):
    st=json.loads(STATE.read_text());prior=json.loads(prior_path.read_text())
    a=[PY,str(N/"refresh_surface_constraints546.py")]
    values={"candidate":st["candidate"],"audit130":st["point_summary"],"audit159":st["surface_report"],"query-dir":str(Path(st["point_summary"]).parent/"query"),"ancestry-parent-candidate":st["candidate"],"prior-candidate":prior["candidate"],"prior-surface-report":prior["surface_report"],"prior-guard-candidate":prior["candidate"],"prior-guard-npz":prior["guard_npz"],"prior-guard-report":prior["guard_report"],"prior-seed-pairs":prior["seed_pairs_path"],"prior-seed-report":prior["seed_report"],"failed-seed-candidate":str(trial/"projected-actual-gates/candidate.npz"),"failed-seed-audit130":str(trial/"first-check/audit-130/summary.json"),"failed-seed-pairs":str(trial/"first-check/audit-130/surface-self-pairs.npy"),"output-dir":str(trial/"constraints")}
    for k,v in values.items():a.extend(["--"+k,v])
    m.run_stage(a,trial/"refresh-stage",slot=True,source_paths=(N/"refresh_surface_constraints546.py",N/"refresh_guards_multi_parent_546.py"),timeout=180)
    return trial/"constraints/constraint-context.json"
def trial_run(t,context):
    assert m.sha(CAPTURE)=="850893a0cc3271fee42e8342e8d73c786cc396eb2f1df4d267c28556c2ace073"
    a=json.loads((N/"round-008-barycentric/fit-argv.json").read_text());a[1]=str(CAPTURE)
    a[a.index("--output")+1]=str(t/"fit");a[a.index("--area-scp-iterations")+1]="1"
    ctx=json.loads(context.read_text())
    for flag,key in [("--self-pairs","seed_pairs_path"),("--fixed-boundary-npz","guard_npz"),("--fixed-boundary-report","guard_report")]:a[a.index(flag)+1]=ctx[key]
    argv=["/usr/bin/env","NUMI_RAW_LP_DIRECTION_FIT_MODE_546=full-plane","NUMI_RAW_LP_DIRECTION_CAPTURE_DIR_546="+str(t/"raw-directions"),*a]
    m.run_stage(argv,t/"fit-stage",slot=True,source_paths=(CAPTURE,N/"run_surface_only_full_coordinate_fit546.py",N.parent/"contained-clearance-545/component_lp_active545.py"),timeout=400)
    projector=N.parent/"contained-clearance-545/project_patch_quality545.py"
    try:m.run_stage([PY,str(projector),"--patch-dir",str(t/"raw-directions/raw-direction-001"),"--out",str(t/"quality-projected")],t/"quality-stage",slot=True,source_paths=(projector,),timeout=120)
    except RuntimeError:
        if "bounded patch projection did not converge" not in (t/"quality-stage.log").read_text():raise
        h=N/"replay_projector_failure546.py"
        m.run_stage([PY,str(h),"--patch-dir",str(t/"raw-directions/raw-direction-001"),"--out",str(t/"projection-failure-diagnostic")],t/"diagnostic-stage",slot=True,source_paths=(h,),timeout=120)
    text=(N/"raw-lp-trial-002-full-plane/retain_projected_iterate.py").read_text()
    text=text.replace("Last emitted iterate of bounded545 raw-SCP quality projection passes unchanged actual area .1, edge .5, trust1mm, source12mm, anchors and outside-free checks; stricter projector comfort edge .495 remained unmet; no exact self or bone admission yet","Raw successful SCP field with bounded local quality projection; unchanged actual area .1, edge .5, trust1mm, source12mm, anchors and outside-free checks verified; exact geometry remains unadmitted")
    (t/"retain_projected_iterate.py").write_text(text)
    m.run_stage([PY,str(t/"retain_projected_iterate.py")],t/"retain-stage",slot=False,source_paths=(t/"retain_projected_iterate.py",),timeout=90)
    try:m.run_stage([PY,str(CHECK),"--candidate",str(t/"projected-actual-gates/candidate.npz"),"--baseline-state",str(STATE),"--output-dir",str(t/"first-check")],t/"first-check-stage",slot=False,source_paths=(CHECK,),timeout=300)
    except RuntimeError:
        d=json.loads((t/"first-check/report.json").read_text())
        if d.get("error")!="exact self intersections":raise
        return d
    return json.loads((t/"first-check/report.json").read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--previous-trial",type=Path,required=True);ap.add_argument("--prior-context",type=Path,required=True);ap.add_argument("--start",type=int,default=3);ap.add_argument("--rounds",type=int,default=3);a=ap.parse_args()
    assert 1<=a.rounds<=3
    previous=a.previous_trial;prior=a.prior_context;records=[]
    for i in range(a.start,a.start+a.rounds):
        context=refresh(previous,prior)
        trial=N/f"raw-lp-trial-{i:03d}-full-plane"
        r=trial_run(trial,context)
        records.append({"trial":str(trial),"context":str(context),"status":r["status"],"self_pairs":r.get("self_pairs"),"contained_pairs":r.get("contained_pairs"),"surface_pairs":r.get("surface_pairs"),"candidate_sha256":r["candidate_sha256"],"native_admission":False,"accepted_physical_steps":0})
        (N/f"raw-lp-cut-generation-through-{i:03d}.json").write_text(json.dumps(records,indent=2)+"\n")
        print(json.dumps(records[-1]),flush=True)
        if r.get("self_pairs")==0:return
        previous=trial;prior=context
if __name__=="__main__":main()
