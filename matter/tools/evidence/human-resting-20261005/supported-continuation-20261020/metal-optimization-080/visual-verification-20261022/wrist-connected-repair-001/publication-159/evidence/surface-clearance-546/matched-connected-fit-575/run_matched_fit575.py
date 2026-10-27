"""One matched, bounded connected-fit trial after topology repair; existing owners only."""
from pathlib import Path
import argparse,importlib.util,json,sys
N=Path(__file__).resolve().parent.parent
sp=importlib.util.spec_from_file_location("fitstage575",N/"continue_surface_round546.py");m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
p=argparse.ArgumentParser();p.add_argument("--state",required=True,type=Path);p.add_argument("--out",required=True,type=Path);a=p.parse_args()
O=a.out
if O.exists():raise FileExistsError(O)
O.mkdir(parents=True)
S=a.state;s=m.read_json(S);C=Path(s["candidate"])
for k in("candidate","point_summary","surface_report"):
 if m.sha(s[k])!=s[k+"_sha256"]:raise ValueError("input changed "+k)
prior=N/"round-033-565-refresh/constraints/constraint-context.json"
trial=N/"round-026-guard-cuts556/raw-directions/raw-direction-001/candidate.npz"
audit=N/"round-026-guard-cuts556/raw-self-audit130/summary.json"
prep=O/"prepare";owner=N/"prepare_surface546.py"
m.run_stage([sys.executable,owner,"--state",S,"--out",prep,"--prior-context",prior,"--seed-source-candidate",trial,"--seed-source-audit130",audit],O/"prepare-stage",slot=False,source_paths=(Path(__file__),owner,S,prior),timeout=600)
context=m.read_json(prep/"constraint-context.json")
point_npz,point_report=m.write_empty_point_targets(prep/"point-targets",C,Path(s["point_summary"]),prep/"depth/365-compatible-report.json")
matched=O/"matched-inputs.json"
m.matched_inputs_from_template(m.MATCHED_TEMPLATE,matched,{"candidate":C,"audit159":Path(s["surface_report"]),"face_target_npz":prep/"surface-targets/targets.npz","point_target_npz":point_npz})
argv=m.fit_argv_from_template(m.FIT_TEMPLATE,candidate=C,audit159=Path(s["surface_report"]),context=context,prep=prep,point_npz=point_npz,point_report=point_report,matched=matched,output=O/"fit")
fit=N/"partial-timeout-fallback564/frozen-v2/run_joint_full_coordinate_fit564.py";argv[1]=str(fit)
for flag,val in(("--trust-mm","1"),("--max-edge-gradient","0.1"),("--source-cap-mm","12"),("--time-limit-s","90")):m.replace_cli_value(argv,flag,val)
if "--allow-partial-timeout-zero-fallback-564"not in argv:argv.append("--allow-partial-timeout-zero-fallback-564")
(O/"fit-argv.json").write_text(json.dumps(argv,indent=2)+"\n")
m.write_json(O/"sensitivity-policy.json",{"hypothesis":"1mm trust and 0.1 edge-gradient limit may retain more actual Float32 progress than larger, poorly conditioned directions","area_floor":0.1,"source_cap_mm":12,"trust_mm":1,"edge_gradient":0.1,"unchanged_total_solver_budget_seconds":120,"native_admission":False,"accepted_physical_steps":0})
fit_failed=False
try:m.run_stage(argv,O/"fit-stage",slot=True,source_paths=(fit,fit.with_name("component_lp_partial_timeout564.py"),fit.with_name("joint_objective_560.py"),S,matched),timeout=300)
except RuntimeError:
 fit_failed=True
 if not(O/"fit/raw-joint-capture-563/report.json").is_file():raise
capture=O/"fit/raw-joint-capture-563/report.json";r=m.read_json(capture)
if r["base_candidate_sha256"]!=s["candidate_sha256"]:raise ValueError("raw capture base mismatch")
returns=r["solver_returns"]
if not returns:raise SystemExit("No certified raw direction to retain")
target=Path(returns[-1]["candidate_path"])
if m.sha(target)!=returns[-1]["candidate_sha256"]:raise ValueError("raw capture target changed")
m.write_json(O/"raw-selection.json",{"capture_report":str(capture),"capture_sha256":m.sha(capture),"target":str(target),"target_sha256":m.sha(target),"fit_process_failed":fit_failed,"fit_status":m.read_json(O/"fit/report.json").get("status"),"linear_solution_is_not_geometry_acceptance":True,"native_admission":False,"accepted_physical_steps":0})
retain=N/"local-witness-retention-567/run_target567.py"
P=O/"local-retention"
m.run_stage([sys.executable,retain,"--state",S,"--target",target,"--out",P],O/"retention-stage",slot=False,source_paths=(retain,S,target),timeout=900)
if(P/"final/state.json").is_file():print(json.dumps({"passing_state":str(P/"final/state.json")}),flush=True);raise SystemExit(0)
full=P/"full-check"
if not(full/"report.json").is_file():raise SystemExit(0)
check=m.read_json(full/"report.json")
if check.get("status")!="COMPLETE_EXACT_CONSTRUCTION_CHECKS"or check.get("self_pairs")!=0:raise SystemExit(0)
# Existing552 only composes regions whose full face/self stencils remain independent.
T=O/"retained-regions";owner=N/"retain_surface_components552.py"
m.run_stage([sys.executable,owner,"--state",S,"--proposal",P,"--full-check",full,"--context",prep/"constraint-context.json","--output",T],O/"regions-stage",slot=False,source_paths=(owner,S,prep/"constraint-context.json"),timeout=60)
r=m.read_json(T/"report.json")
base_count=sum(int((b.get("result")or{}).get("count",0))for b in m.read_json(s["surface_report"])["banks"])
if r["predicted_surface_pairs"]>=base_count:
 print(json.dumps({"status":"NO_PREDICTED_REGION_PROGRESS","predicted":r["predicted_surface_pairs"],"base":base_count}),flush=True);raise SystemExit(0)
owner=N.parent/"contained-clearance-545/run_candidate_checks545.py"
m.run_stage([sys.executable,owner,"--candidate",T/"candidate.npz","--baseline-state",S,"--trust-mm","4","--output-dir",T/"full-check"],O/"regions-check-stage",slot=False,source_paths=(owner,),timeout=600)
r=m.read_json(T/"full-check/report.json")
if r.get("construction_progress")and r.get("contained_pairs")==0 and r.get("self_pairs")==0:
 owner=N/"finalize_existing_checks546.py"
 m.run_stage([sys.executable,owner,"--candidate",T/"candidate.npz","--baseline-state",S,"--point-summary",T/"full-check/audit-130/summary.json","--surface-report",T/"full-check/audit-159/report.json","--out",T/"final","--trust-mm","4"],O/"regions-final-stage",slot=False,source_paths=(owner,),timeout=60)
print(json.dumps({"retained_region_check":r.get("status"),"surface_pairs":r.get("surface_pairs"),"contained_pairs":r.get("contained_pairs"),"self_pairs":r.get("self_pairs")}),flush=True)

