#!/usr/bin/env python3
import os,sys,json,hashlib,datetime,time
from pathlib import Path
Q=Path("/Users/n/cardboard-implementation-20261014-host-macmini")
CASE=Q/"solver/control19-newton3-resume-diagnostic-555"
CAP=CASE/"diagnostics/capture-operator-fd-capture/capture"
PREP=Q/"analysis/source198-strict-layer-replay-preparation-001"
REF=Q/"snapshots/source-stable-green-reference-206-buildfix-002/matter/tools"
OUT=Q/"analysis/native555-actual19-material-replay-001"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""): h.update(block)
 return h.hexdigest()
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"): os.environ[k]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
sys.path.insert(0,str(PREP));sys.path.insert(0,str(REF))
import strict_layers
assert Path(strict_layers.ref.__file__).resolve()==(REF/"cardboard_bilinear_fp64_reference.py").resolve(), strict_layers.ref.__file__
assert Path(strict_layers.strict.__file__).resolve()==(REF/"cardboard_compensated_fp64_reference.py").resolve(), strict_layers.strict.__file__
assert CAP.is_file() and sha(CAP)=="c0a111c510ef4c5891ace3bb58f65af48740733d7b46987be370561d1c22ab53"
assert not (OUT/"environment-0").exists() and not (OUT/"environment-1").exists()
pred=json.loads((CASE/"prediction.json").read_text())
receipt=json.loads((CASE/"receipt.json").read_text())
assert receipt["execution_started"] is True and receipt["exit_code"]==2 and receipt["timed_out"] is False
assert receipt["source_inputs_unchanged"] is True and receipt["changed_inputs"]==[]
assert pred["environment"]["NM_FEM_OPERATOR_FD_SELECTOR"]=="19:3"
header=json.loads(CAP.open().readline())
assert header["schema"]=="numi.cardboard.fem-newton-capture.v1" and (header["root"],header["newton_iteration"])==(19,3)
assert header["environment_count"]==2 and header["tetrahedron_count_per_environment"]==1920
assert header["cached_algorithmic_tangent_capture"]["available"] is True
assert header["cached_algorithmic_tangent_capture"]["snapshot_bitwise_guard_included"] is True
argv=pred["argv"]
assert "--coupon-resume-diagnostic-replay" in argv and "--accurate-source-paper-projection" in argv
assert "--material-iterations" in argv and argv[argv.index("--material-iterations")+1]=="32"
source_snapshot=Path(pred["source_snapshot"])
frozen=source_snapshot/"frozen-source.json"
build=Path(pred["build_provenance"])
materials=[source_snapshot/"matter/materials/hajali2009_liner_hill_bilinear_fit.nmatter",source_snapshot/"matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter",source_snapshot/"matter/materials/starch2007_finite_glue_elastic.nmatter"]
assert all(x.is_file() for x in materials)
inputs=[CAP,CASE/"prediction.json",CASE/"receipt.json",CASE/"stderr.log",CASE/"output/manifest.json",CASE/"output/coupon-resume-segment.json",CASE/"output/result.json",CASE/"output/compiled.nmatterpack",build,frozen,*materials,PREP/"strict_layers.py",REF/"cardboard_bilinear_fp64_reference.py",REF/"cardboard_compensated_fp64_reference.py",REF/"cardboard_compensated_fp64_base.py",REF/"cardboard_dd_fp64.py",REF/"cardboard_cached_operator_audit.py"]
binding={"schema":"numi.cardboard.native555-material-replay-bindings.v1","capture_sha256":sha(CAP),"capture_bytes":CAP.stat().st_size,"prediction_sha256":sha(CASE/"prediction.json"),"receipt_sha256":sha(CASE/"receipt.json"),"stderr_sha256":sha(CASE/"stderr.log"),"output_manifest_sha256":sha(CASE/"output/manifest.json"),"resume_segment_sha256":sha(CASE/"output/coupon-resume-segment.json"),"result_sha256":sha(CASE/"output/result.json"),"compiled_pack_sha256":sha(CASE/"output/compiled.nmatterpack"),"source_snapshot":str(source_snapshot),"frozen_source_sha256":sha(frozen),"source_build_sha256":sha(build),"reference_tools_sha256":{str(x):sha(x) for x in [PREP/"strict_layers.py",REF/"cardboard_bilinear_fp64_reference.py",REF/"cardboard_compensated_fp64_reference.py",REF/"cardboard_compensated_fp64_base.py",REF/"cardboard_dd_fp64.py",REF/"cardboard_cached_operator_audit.py"]},"material_sha256":{str(x):sha(x) for x in materials},"header":header,"native_receipt":{"exit_code":receipt["exit_code"],"timed_out":receipt["timed_out"],"source_inputs_unchanged":receipt["source_inputs_unchanged"],"native_transactions":0},"epsilon_grid_powers":[8,10,12,14,16,18,20,22,24,26],"qualification":False}
(OUT/"bindings.json").write_text(json.dumps(binding,indent=2,sort_keys=True)+"\n")
started=time.monotonic(); summaries={}
for env in (0,1):
 d=OUT/f"environment-{env}"
 r=strict_layers.audit(CASE,d,environment=env,epsilon_powers=binding["epsilon_grid_powers"])
 summaries[str(env)]={"scope":r["scope"],"state_rows":[{"state_index":s["state_index"],"projection":s["projection"],"stress":s["stress_comparison_Pa"],"projected_state":s["projected_state_comparison"],"native_vs_reference_branch_mismatch_rows":s["native_vs_reference_branch_mismatch_rows"],"local_directional_action":s.get("local_directional_action"),"frame_action":s.get("frame_action"),"element_action":s.get("element_action"),"assembled_action":s.get("assembled_action_Ns"),"branch_predicates_changed_from_center":s.get("branch_predicates_changed_from_center"),"realized_endpoint_vs_ideal_F_perturbation":s.get("realized_endpoint_vs_ideal_F_perturbation")} for s in r["states"]],"fp64_directional_sweep":r["fp64_directional_sweep"]}
summary={"schema":"numi.cardboard.native555-material-replay-summary.v1","classification":"independent CPU reference diagnostic only; source555 actual control19/Newton3; no native transaction or physical qualification","bindings":binding,"environments":summaries,"wall_seconds":time.monotonic()-started,"finished_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
manifest={"schema":"numi.cardboard.native555-material-replay-files.v1","files_sha256":{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name!="manifest.json"},"inputs_sha256":{str(p):sha(p) for p in inputs}}
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
print(json.dumps({"output":str(OUT),"wall_seconds":summary["wall_seconds"],"env_summaries":{e:{"states":[{"i":s["state_index"],"valid":s["projection"]["valid_rows"],"maxr":s["projection"]["max_scaled_residual"],"maxc":s["projection"]["max_dynamic_correction"],"maxstressrel":s["stress"]["max_relative_l2"],"maxstate_rel":s["projected_state"]["max_relative_l2"]} for s in v["state_rows"]],"sweep_adjacent":v["fp64_directional_sweep"]["rows_with_two_adjacent_declared_grid_points_at_1e_6"],"sweep_rows":v["fp64_directional_sweep"]["rows"]} for e,v in summaries.items()}},sort_keys=True))
