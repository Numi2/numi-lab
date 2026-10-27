import hashlib, importlib.util, json, subprocess
from pathlib import Path
PYTHON=Path("/Users/n/numi-human-prep-venv-20261005/bin/python")
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/outward-boneface-filter-640/root-bank43-selfcut-001")
MERGED=ROOT/"merged-seeds"
FITARGV=ROOT/"fit-argv-retry-002.json"
SEED_REPORT=MERGED/"self-seeds/seed-preflight.json"
BASE=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/connected-topology-580/run-010-from689/root-combined-001/candidate.npz")
RAW=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/outward-boneface-filter-640/root-bank43-component-001/candidate.npz")
SELECTOR=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/outward-boneface-filter-640/root-bank43-selfcut-001/selector-640-merged-seeds-001/selection.npz")
SELECTOR_REPORT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/outward-boneface-filter-640/root-bank43-selfcut-001/selector-640-merged-seeds-001/report.json")
WRAPPER=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/run_axis_map_sat_fit638.py")
TEMPLATE=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/outward-boneface-filter-640/fit-675-reject-opposed-001/fit-axis-map-638.launch.json")
OUT=ROOT/"fit-retry-002"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
if OUT.exists(): raise FileExistsError(OUT)
args=json.loads(FITARGV.read_text())
for flag in ("--base-candidate","--raw-trial","--self-pairs","--output","--matched-inputs-json-490","--area-reference-npz","--point-target-npz","--face-target-npz","--point-depth-rows-npz","--fixed-boundary-npz"):
 if args.count(flag)!=1: raise ValueError(f"expected exactly one {flag}")
if Path(args[args.index("--base-candidate")+1]).resolve()!=BASE.resolve(): raise ValueError("base candidate changed")
if Path(args[args.index("--raw-trial")+1]).resolve()!=RAW.resolve(): raise ValueError("raw trial changed")
if Path(args[args.index("--self-pairs")+1]).resolve()!=(MERGED/"self-seeds/merged-self-pair-seeds.npy").resolve(): raise ValueError("merged seed path changed")
if Path(args[args.index("--output")+1]).resolve()!=OUT.resolve(): raise ValueError("fit output path changed")
seed=json.loads(SEED_REPORT.read_text())
if seed.get("candidate_sha256")!=sha(BASE) or seed.get("current_exact_self_pair_count")!=0: raise ValueError("seed preflight/base mismatch")
if seed.get("status")!="PASS_REBUILT_CONSTRUCTION_SEEDS_ONLY" or seed.get("merged_unique_pair_count")!=32075 or seed.get("zero_step_feasible") is not True or float(seed.get("zero_step_max_rhs_mm",float("inf")))>2e-8: raise ValueError("merged seed preflight failed")
if seed.get("failed_trial_source",{}).get("candidate_sha256")!=sha(RAW) or seed.get("failed_trial_source",{}).get("exact_audit_pair_count")!=15: raise ValueError("failed-trial witness pins/count mismatch")
selector=json.loads(SELECTOR_REPORT.read_text())
if selector.get("selection_npz_sha256")!=sha(SELECTOR) or selector.get("selected_pair_count")!=660: raise ValueError("640 selector receipt mismatch")
if args[:2]!=[str(PYTHON),"/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-scp-area-backtrack-603/frozen-v1/run_joint_full_coordinate_fit603.py"]: raise ValueError("merged fit argv command prefix changed")
fit_cli_args=args[2:]
template=json.loads(TEMPLATE.read_text())["fit_argv_after_removing_adapter_flags"]
flag_vector=lambda xs:[x for x in xs if x.startswith("--")]
if flag_vector(fit_cli_args)!=flag_vector(template): raise ValueError("603 fit option shape differs from successful 640 reference launch")
spec=importlib.util.spec_from_file_location("axis_map638_smoke",WRAPPER)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
smoke_argv=[str(WRAPPER),*fit_cli_args,"--selector-npz",str(SELECTOR),"--selector-report",str(SELECTOR_REPORT)]
parsed,selector=module.consume_flags(smoke_argv)
if parsed!=[str(WRAPPER),*fit_cli_args] or selector["--selector-npz"]!=SELECTOR.resolve() or selector["--selector-report"]!=SELECTOR_REPORT.resolve(): raise ValueError("638 selector argument smoke failed")
smoke={"status":"PASS_NO_SOLVER_ARGUMENT_SMOKE","fit_flag_vector_matches_successful_640_reference":True,"selector_flags_consumed_only_by_outer638":True,"fit_cli_arg_count":len(fit_cli_args),"fit_cli_flags":flag_vector(fit_cli_args),"merged_seed_sha256":sha(MERGED/"self-seeds/merged-self-pair-seeds.npy"),"selector_npz_sha256":sha(SELECTOR),"selector_report_sha256":sha(SELECTOR_REPORT),"reference_launch_sha256":sha(TEMPLATE),"native_admission":False}
(ROOT/"fit-retry-002-argument-smoke.json").write_text(json.dumps(smoke,indent=2,sort_keys=True)+"\n")
cmd=[str(PYTHON),str(WRAPPER),*fit_cli_args,"--selector-npz",str(SELECTOR),"--selector-report",str(SELECTOR_REPORT)]
if "--selector-npz" in args or "--selector-report" in args: raise ValueError("outer selector flags already in fit argv")
launch={"status":"LAUNCHING","argv":cmd,"base_candidate_sha256":sha(BASE),"raw_trial_sha256":sha(RAW),"seed_preflight_sha256":sha(SEED_REPORT),"merged_seed_sha256":sha(MERGED/"self-seeds/merged-self-pair-seeds.npy"),"selector_npz_sha256":sha(SELECTOR),"selector_report_sha256":sha(SELECTOR_REPORT),"wrapper_sha256":sha(WRAPPER),"native_admission":False,"accepted_physical_steps":0,"claim_limit":"Fit proposal only; exact 545 checks and all original gates remain required."}
(ROOT/"fit-retry-002-launch.json").write_text(json.dumps(launch,indent=2,sort_keys=True)+"\n")
with (ROOT/"fit-retry-002-stage.log").open("wb") as log:
 proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=False)
execution={"returncode":proc.returncode,"status":"FINISHED" if proc.returncode==0 else "FAILED","fit_output_exists":OUT.exists(),"native_admission":False,"accepted_physical_steps":0}
(ROOT/"fit-retry-002-execution.json").write_text(json.dumps(execution,indent=2,sort_keys=True)+"\n")
print(json.dumps(execution,sort_keys=True))
raise SystemExit(proc.returncode)
