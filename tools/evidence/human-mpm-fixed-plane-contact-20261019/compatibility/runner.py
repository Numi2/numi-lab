from pathlib import Path
import hashlib,json,os,subprocess,time,shutil,sys
ROOT=Path("/Users/n/numi-human-separation-20261018/C/integrated-general-tissue-257")
OUT=ROOT/"candidate-blood-runtime-regression-001"/"native-kkt-003"
BUILD=ROOT/"build-cpu-001"
BINARY=BUILD/"numi-human-native"
BASELIB=Path("/Users/n/numi-human-separation-20261018/C/continuum-integrated-234/build-cpu-006/libmetalrobo-233.dylib")
LIB=Path("/Users/n/numi-human-separation-20261018/C/sustained-blood-247/native-mpm-grid-boundary-001/native-kkt-grid-floor96-prep-001/build-reaction-rollback-001/libmetalrobo.dylib")
MSL=Path("/Users/n/numi-human-separation-20261018/C/sustained-blood-247/native-mpm-grid-boundary-001/native-kkt-grid-floor96-prep-001/input-MatterSource-reaction-rollback.metal")
BASE_INV=ROOT.parent/"optic-nerve-admission-256/native-001/invocation.json"
APP=BUILD/"source/apps/numilab_human_myosim_visual_probe.mm"
WRAPPER=Path("/Users/n/numi-human-final-integration-publish-1397/tools/run_resting_job_exclusive.py")
PYTHON=Path("/Users/n/numi-human-prep-venv-20261005/bin/python3")
MARKERS=[
 "shared_tissue_late_reject_retry=passed",
 "q_v_root_muscle_unchanged=1",
 "matter_vascular_unchanged=1",
 "passive_adapter_history_unchanged=1",
 "brain_respiration_presentation_unchanged=1",
 "body_properties_restored=1",
 "dof_properties_restored=1",
 "same_context_retry=1",
]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not OUT.exists(), f"refusing existing run path {OUT}"
assert BINARY.is_file() and sha(BINARY)==json.loads((BUILD/"link-receipt.json").read_text())["product_sha256"]
assert BASELIB.is_file() and sha(BASELIB)=="71372f858367c5945eed34e7334f184f210f06c7cd21e6ec48cb340a4aebd60a"
EXPECTED_LIB_SHA256="5a58b578134a5d7ce31dd6cc7a31713914ded0d64dca2fc571eec9363c8a4f8b"
assert LIB.is_file() and sha(LIB)==EXPECTED_LIB_SHA256
assert MSL.is_file() and sha(MSL)=="f4ff7cf5e03b18b3ffbcc964e876c54c04cd7725ee4786d51b32ecff2d0377f0"
assert sha(APP)==sha(ROOT/"merge/combined-resolved.mm")
assert sha(BUILD/"compile-receipt.json") is not None
base=json.loads(BASE_INV.read_text())
assert base["candidate_binary_sha256"]=="d42d4373d05afff3b9fed60d3c983f41524abd27426dae2c20ce6317d43a024a"
base_assets={}
for value in base["argv"][1:]:
    p=Path(value)
    if p.is_file():
        base_assets[str(p.resolve())]=sha(p)
for key,value in base["environment"].items():
    if key.endswith(("_SOURCE","_MATERIAL")) and Path(value).is_file():
        p=Path(value)
        base_assets[str(p.resolve())]=sha(p)
assert WRAPPER.is_file() and PYTHON.is_file()
disk=shutil.disk_usage(ROOT)
assert disk.free > 1_500_000_000, f"insufficient free space for paired captured endpoint runs: {disk.free}"
runplan={
 "schema":"numi.optic_nerve_shared_tissue_rollback_retry_plan.v1",
 "scope":"two requested 0.25 ms steps with one late rejection at step 1 and same-context retry; merged general tissue source with bilateral optic nerve FEM plus skin/fat, aortic wall, and prostate opt-ins; unqualified organ batch and MPM off; no visual verification",
 "base_invocation_path":str(BASE_INV),"base_invocation_sha256":sha(BASE_INV),
 "candidate_source_root":str(BUILD/"source"),"app_sha256":sha(APP),
 "coupling_header_sha256":sha(BUILD/"source/apps/NumiHumanRestingCoupling.hpp"),
 "nerve_integration_header_sha256":sha(BUILD/"source/apps/NumiHumanOpticNerveFEMIntegration.hpp"),
 "nerve_tetgen_header_sha256":sha(BUILD/"source/apps/NumiHumanOpticNerveTetGenAssets.hpp"),
 "compile_receipt_sha256":sha(BUILD/"compile-receipt.json"),
 "link_receipt_sha256":sha(BUILD/"link-receipt.json"),
 "binary_path":str(BINARY),"binary_sha256":sha(BINARY),
 "runtime_library_path":str(LIB),"runtime_library_sha256":sha(LIB),
 "runtime_source_sha256":"438911d8f1bdaa9f820c6d2bba1bbe87182d2bd14e485f5bbf5fcd2c1e01014c",
 "comparison_runtime_library_path":str(BASELIB),"comparison_runtime_library_sha256":sha(BASELIB),
 "matter_test_source_path":str(MSL),"matter_test_source_sha256":sha(MSL),"matter_mpm_enabled":False,
 "step_count":2,"step_seconds":0.00025,"physical_horizon_seconds":0.0005,
 "cases":[{"name":"nerve_joint_clean_control","rollback_retry":False},{"name":"nerve_joint_late_reject_retry","rollback_retry":True,"reject_step":1}],
 "rollback_log_markers":MARKERS,"free_bytes_before_run":disk.free,"visual_verification":False
}
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.mkdir()
(OUT/"run-plan.json").write_text(json.dumps(runplan,indent=2)+"\n")
alias=OUT/"lib";alias.mkdir();(alias/"libmetalrobo.dylib").symlink_to(LIB)
baseargv=list(base["argv"][3:]);baseargv[0]=str(BINARY)
baseargv[baseargv.index("--muscle-step-count")+1]="2"
baseargv[baseargv.index("--muscle-step-seconds")+1]="0.00025000"
baseenv=dict(base["environment"])
baseenv.update({
 "DYLD_LIBRARY_PATH":str(alias),"DYLD_PRINT_LIBRARIES":"1",
 "NUMI_HUMAN_BLOOD_MPM_ENABLED":"0",
 "NUMI_MATTER_TEST_SOURCE":str(MSL),
 "NUMI_MATTER_TEST_SOURCE_SHA256":sha(MSL),
 "NUMI_HUMAN_RESTING_OPTIC_NERVE_FEM":"1",
 "NUMI_HUMAN_RESTING_SKIN_FAT_FEM":"1",
 "NUMI_HUMAN_RESTING_AORTIC_WALL":"1",
 "NUMI_HUMAN_RESTING_PROSTATE_FEM":"1",
 "NUMI_HUMAN_RESTING_ORGAN_BATCH_FEM":"0",
 "NUMI_HUMAN_RESTING_FGMRES_BUDGET":"16",
 "NUMI_HUMAN_PASSIVE_FEM_MAX_RATE_EXPONENT":"0",
 "NUMI_MATTER_GPU_TIMING":"0",
})
attempts=[]
for case in runplan["cases"]:
 d=OUT/case["name"];d.mkdir(); outdir=d/"output";outdir.mkdir()
 argv=list(baseargv);argv[4]=str(outdir)
 env=dict(baseenv)
 env["NUMI_HUMAN_SHARED_TISSUE_ROLLBACK_TEST"]="1" if case["rollback_retry"] else "0"
 env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(d/"common-field-failure.json")
 assets=dict(base_assets)
 for asset_path,expected in base_assets.items():
  assert sha(asset_path)==expected, f"pinned invocation asset changed: {asset_path}"
 for value in argv[1:]:
  p=Path(value)
  if p.is_file():assets[str(p.resolve())]=sha(p)
 for key,value in env.items():
  if key.endswith(("_SOURCE","_MATERIAL")) and Path(value).is_file():assets[str(Path(value).resolve())]=sha(value)
 assets[str(APP.resolve())]=sha(APP)
 for p in (BUILD/"source/apps/NumiHumanRestingCoupling.hpp",BUILD/"source/apps/NumiHumanOpticNerveFEMIntegration.hpp",BUILD/"source/apps/NumiHumanOpticNerveTetGenAssets.hpp"):
  assets[str(p.resolve())]=sha(p)
 launcher=[str(PYTHON),str(WRAPPER),"--",*argv]
 rec={"schema":"numi.optic_nerve_shared_tissue_rollback_attempt.v1","case":case,"argv":argv,"launcher_argv":launcher,"environment":env,"asset_sha256":assets,"plan_sha256":sha(OUT/"run-plan.json"),"binary_sha256":sha(BINARY),"runtime_library_sha256":sha(LIB),"requested_step_count":2,"requested_step_seconds":0.00025,"physical_horizon_seconds":0.0005,"visual_verification":False}
 recpath=d/"invocation.json";recpath.write_text(json.dumps(rec,indent=2)+"\n")
 procenv=os.environ.copy();procenv.update(env);t=time.monotonic()
 with (d/"native.log").open("w") as f:proc=subprocess.run(launcher,env=procenv,stdout=f,stderr=subprocess.STDOUT)
 rec.update(process_exit=proc.returncode,wall_seconds=time.monotonic()-t,runtime_library_sha256_after=sha(LIB))
 recpath.write_text(json.dumps(rec,indent=2)+"\n")
 attempts.append({"case":case["name"],"exit":proc.returncode,"wall_seconds":rec["wall_seconds"],"log":str(d/"native.log")})
 if proc.returncode:break
summary={"schema":"numi.optic_nerve_shared_tissue_rollback_summary.v1","plan_sha256":sha(OUT/"run-plan.json"),"runner_sha256":sha(Path(__file__)),"binary_sha256":sha(BINARY),"runtime_library_sha256":sha(LIB),"attempts":attempts,"scope":runplan["scope"],"visual_verification":False}
if len(attempts)==2 and all(x["exit"]==0 for x in attempts):
 clean=OUT/"nerve_joint_clean_control";retry=OUT/"nerve_joint_late_reject_retry"
 retrylog=(retry/"native.log").read_text(errors="replace")
 summary["rollback_log_checks"]={m:m in retrylog for m in MARKERS}
 assert all(summary["rollback_log_checks"].values()),"rollback/retry log marker missing"
 def accepted_nerve_state(logpath):
  lines=Path(logpath).read_text(errors="replace").splitlines()
  assert any("resting_FEM_endpoint_capture=begin version=1" in x for x in lines)
  assert any("resting_FEM_endpoint_capture=end" in x for x in lines)
  needles=("registered_left_optic_nerve_reference_fem","registered_right_optic_nerve_reference_fem")
  state=[x for x in lines if ("resting_FEM_node,object=" in x or "resting_FEM_tet,object=" in x) and any(n in x for n in needles)]
  counts={n:{"nodes":sum(x.startswith("resting_FEM_node,object="+n) for x in state),"tetrahedra":sum(x.startswith("resting_FEM_tet,object="+n) for x in state)} for n in needles}
  assert counts[needles[0]]=={"nodes":1728,"tetrahedra":7675},counts
  assert counts[needles[1]]=={"nodes":1701,"tetrahedra":7423},counts
  text="\n".join(state)+"\n"
  return {"line_count":len(state),"counts":counts,"sha256":hashlib.sha256(text.encode()).hexdigest(),"bytes":text.encode()}
 a=accepted_nerve_state(clean/"native.log");b=accepted_nerve_state(retry/"native.log")
 summary["accepted_nerve_fem_state"]={"clean":{"line_count":a["line_count"],"counts":a["counts"],"sha256":a["sha256"]},"retry":{"line_count":b["line_count"],"counts":b["counts"],"sha256":b["sha256"]},"byte_identical":a["bytes"]==b["bytes"]}
 assert summary["accepted_nerve_fem_state"]["byte_identical"],"nerve accepted FEM state differs after rollback/retry"
 clean_csv=(clean/"output/resting-coupled.csv").read_bytes();retry_csv=(retry/"output/resting-coupled.csv").read_bytes()
 summary["clean_vs_retry_csv"]={"clean_sha256":hashlib.sha256(clean_csv).hexdigest(),"retry_sha256":hashlib.sha256(retry_csv).hexdigest(),"byte_identical":clean_csv==retry_csv}
 assert summary["clean_vs_retry_csv"]["byte_identical"],"clean/retry CSV differs"
 for case_dir in (clean,retry):
  log=(case_dir/"native.log").read_text(errors="replace")
  assert "resting_FEM_terminal_audit=passed expected_steps=2" in log
  assert "source_boundary_faces=2376" in log and "source_boundary_faces=2378" in log
 baseline_root=ROOT/"native-001"
 comparisons={}
 for case in runplan["cases"]:
  name=case["name"]
  candidate=OUT/name
  baseline=baseline_root/name
  row={}
  for rel in ("output/resting-coupled.csv","output/resting-surface-audit.csv"):
   a=(candidate/rel).read_bytes();b=(baseline/rel).read_bytes()
   row[rel]={"candidate_sha256":hashlib.sha256(a).hexdigest(),"baseline_233_sha256":hashlib.sha256(b).hexdigest(),"byte_identical":a==b}
   assert a==b,f"candidate blood-runtime FEM/CV output differs from baseline 233: {name}/{rel}"
  def fem_state(p):
   lines=Path(p).read_text(errors="replace").splitlines()
   assert any("resting_FEM_endpoint_capture=begin version=1" in x for x in lines)
   assert any("resting_FEM_endpoint_capture=end" in x for x in lines)
   state=[x for x in lines if x.startswith("resting_FEM_node,") or x.startswith("resting_FEM_tet,")]
   return state
  a=fem_state(candidate/"native.log");b=fem_state(baseline/"native.log")
  row["accepted_fem_state"]={"candidate_rows":len(a),"baseline_233_rows":len(b),"candidate_sha256":hashlib.sha256(("\n".join(a)+"\n").encode()).hexdigest(),"baseline_233_sha256":hashlib.sha256(("\n".join(b)+"\n").encode()).hexdigest(),"byte_identical":a==b}
  assert a==b,f"candidate blood-runtime accepted FEM state differs from baseline 233: {name}"
  loaded=[]
  for line in (candidate/"native.log").read_text(errors="replace").splitlines():
   if "libmetalrobo" in line:
    token=line.split()[-1]
    try:
     if Path(token).resolve()==LIB.resolve(): loaded.append(token)
    except OSError: pass
  assert loaded,f"dyld log did not confirm candidate runtime load for {name}"
  row["loaded_runtime_path_observed"]=loaded[-1]
  row["loaded_runtime_resolves_to_pinned_sha256"]=sha(Path(loaded[-1]).resolve())
  assert row["loaded_runtime_resolves_to_pinned_sha256"]==EXPECTED_LIB_SHA256
  row["matter_override"]={"path":str(MSL),"sha256":sha(MSL),"mpm_enabled":False}
  comparisons[name]=row
 summary["comparison_to_published_233_runtime"]=comparisons
 assert all(c["output/resting-coupled.csv"]["byte_identical"] and c["output/resting-surface-audit.csv"]["byte_identical"] and c["accepted_fem_state"]["byte_identical"] for c in comparisons.values())
 (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
if len(attempts)!=2 or any(x["exit"] for x in attempts) or not summary.get("rollback_log_checks",{}).get(MARKERS[0],False) or not summary.get("accepted_nerve_fem_state",{}).get("byte_identical",False) or not summary.get("clean_vs_retry_csv",{}).get("byte_identical",False):sys.exit(1)
