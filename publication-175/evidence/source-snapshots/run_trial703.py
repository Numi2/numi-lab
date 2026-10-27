from pathlib import Path
import importlib.util,json
N=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546");P=N/"hard-face-feasibility-634/prepared-from703-001/prepared";W=N/"hard-face-feasibility-634/fresh-fit-from703-001"
spec=importlib.util.spec_from_file_location("m",N/"continue_surface_round546.py");M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
prep=json.loads((P/"prepared.json").read_text()); assert prep["fit_not_run"] is True and prep["current_candidate_sha256"]=="22dce5fe0457cfe0ec2109ff922ba4ce69720504b3f69370e55beb25b0da279b"
a=json.loads((P/"fit-argv.json").read_text());M.replace_cli_value(a,"--output",W/"fit");M.replace_cli_value(a,"--area-scp-iterations","1");M.write_json(W/"actual-fit-argv.json",a)
refs=[Path(__file__),P/"prepared.json",P/"fit-argv.json",P/"matched-inputs.json",P/"independent-typed-identity-check.json",P/"surface-targets/targets.npz",Path(prep["current_candidate"]),Path(a[1])]
M.run_stage(a,W/"fit-stage",slot=True,source_paths=tuple(refs),timeout=600)
report=json.loads((W/"fit/report.json").read_text());direct=W/"fit/candidate.npz";capture=W/"fit/raw-joint-capture-563/report.json"
if direct.is_file():
 target=direct;source="complete proposal; requires exact acceptance"
elif capture.is_file():
 raw=json.loads(capture.read_text());assert raw["base_candidate_sha256"]==prep["current_candidate_sha256"] and len(raw["solver_returns"])==1
 target=Path(raw["solver_returns"][0]["candidate_path"]);assert M.sha(target)==raw["solver_returns"][0]["candidate_sha256"] and raw["solver_returns"][0]["solver_receipt"]["feasibility_certificate"]["pass"] is True
 source="partial solver direction only; global LP did not converge; requires exact acceptance"
else:
 print("NO_REUSABLE_PROPOSAL",report.get("status"),flush=True);raise SystemExit(0)
print("TARGET",source,M.sha(target),flush=True)
state=Path(prep["current_state"]);owner=N/"local-witness-retention-582/run_target582.py";py=Path("/Users/n/numi-human-prep-venv-20261005/bin/python")
M.run_stage([py,owner,"--state",state,"--target",target,"--out",W/"retention-582-001"],W/"retention-main-stage",slot=False,source_paths=(Path(__file__),owner,state,target),timeout=1800)
print("RETENTION_DONE",flush=True)
