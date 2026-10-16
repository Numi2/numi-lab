#!/usr/bin/env python3
from pathlib import Path
import json, hashlib
Q=Path("/Users/n/cardboard-implementation-20261014-host-macmini")
SRC=Q/"snapshots/source-contact-damped-cgls-memory-195-001"
BUILD=Q/"build-source195-001"
MAT=SRC/"matter/materials/nagasawa2013_medium_orthotropic.nmatter"
S=SRC/"matter/tools/cardboard_full_contact_armijo_check.mm"
P520=Q/"analysis/native520-integrated-contact-rejection-command-001"
P525=Q/"analysis/native525-contact-restart16-command-001"
N520=Q/"native/integrated195-contact-rejection-520/baseline"
N525=Q/"native/integrated195-contact-restart16-525"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text())
def armijo_rows(p):
    rows=[]
    for ln in Path(p).read_text().splitlines():
        if ln.startswith("fem_residual_armijo_iteration="):
            d=json.loads(ln.split("=",1)[1]); e=d["environments"][0]
            pre=e.get("pretrial_status",{})
            rows.append({
              "root":d.get("root"),"newton_iteration":d["newton_iteration"],
              "base_l2":e.get("base_l2"),"trial_l2":e.get("trial_l2"),
              "selected_alpha":e.get("selected_alpha"),"accepted":e.get("accepted"),
              "certified_noop":e.get("certified_noop"),
              "trial_outcome":e.get("trial_outcome"),
              "base_health":e.get("base_health"),
              "solver_status_code":e.get("solver_status_code"),
              "pretrial_status_code":pre.get("code"),
              "pretrial_failing_index":pre.get("failing_index"),
              "pretrial_fgmres_iterations":pre.get("fgmres_iterations"),
              "pretrial_diagnostics":pre.get("diagnostics"),
            })
    return rows
spec520=load(P520/"command-spec.json"); spec525=load(P525/"command-spec.json")
receipt525=load(P525/"execution-525/receipt.json")
cap520=load(P520/"capture.coupled.json")
stdout525=(P525/"execution-525/stdout.log").read_text()
source_lines=S.read_text().splitlines()
loaded=armijo_rows(N525/"baseline/fixture-loaded.log")
rejected=armijo_rows(N525/"baseline/fixture-rejected.log")
assert spec520["source_frozen_sha256"]==spec525["source_frozen_sha256"]
assert spec520["files_sha256"]==spec525["files_sha256"]
assert spec520["material_sha256"]==spec525["material_sha256"]
assert "--compensated-material-arithmetic" in spec520["cases"]["520"]
assert "--self-surface-scratch-alias-parity" in spec520["cases"]["520"]
assert "--extended-budget" not in spec520["cases"]["520"]
assert spec525["cases"]["525"][-1]=="--extended-budget"
assert receipt525["exit_code"]==0 and not receipt525["timed_out"] and not receipt525["space_abort"]
assert receipt525["shipping_or_physical_validation"] is False
assert cap520["iteration"]==9 and cap520["root"]==2
mix520=cap520["mixed_solver"]
assert mix520=={"newton_budget":16,"restart":10,"fgmres_budget":64,"line_search_steps":8,"preconditioner_selector":0,"rigid_articulated_nv":0}
assert len(loaded)==32 and len(rejected)==32
assert loaded[24]["accepted"] is True and loaded[24]["trial_l2"]<=1e-7
assert loaded[25]["certified_noop"] is True
assert all(x["accepted"] for x in loaded[:25])
assert all(x["certified_noop"] for x in loaded[25:])
assert rejected[18]["accepted"] is False and rejected[18]["trial_outcome"]=="no_trial_accepted_status_clear"
assert rejected[19]["pretrial_status_code"]==10 and rejected[19]["pretrial_failing_index"]==18
assert rejected[19]["pretrial_fgmres_iterations"]==256
tx=[]
for i in range(3):
    d=load(N525/f"baseline/transaction-{i:06d}.json")
    tx.append({"attempt_index":i,"requested_control_step":d["requested_control_step"],
      "accepted":d["accepted"],"after_present":d["after_present"],
      "before_control_step":d["before"]["control_step"],
      "after_control_step":d["after"]["control_step"] if d.get("after") else None,
      "after_statuses":d["after"]["statuses"] if d.get("after") else None})
assert [x["accepted"] for x in tx]==[True,True,False]
assert [x["requested_control_step"] for x in tx]==[1,2,3]
parity={}
files=["fixture-activation.log","fixture-initialization.log","fixture-loaded.log","fixture-rejected.log"]
files += [f"transaction-{i:06d}-{suffix}.bin" for i in range(3) for suffix in ("before","after")]
files += [f"transaction-{i:06d}.json" for i in range(3)]
for f in files:
    a=sha(N525/"baseline"/f); b=sha(N525/"aliased"/f)
    assert a==b, f
    parity[f]=a
assert "selected_fem_preconditioner=scalar_diagonal selector=0" in stdout525
assert "PASS full-contact self-surface scratch alias parity checks=5976 native_fem_transactions_executed=6" in stdout525
assert "full_contact_rejection_rollback=pass" in stdout525
assert "authored_extended_budget=fgmres256,restart16,line_search24 relative_residual_unchanged=1e-7" in stdout525
extended={"newton_budget":32,"fgmres_budget":256,"restart":16,"line_search_steps":24,"relative_residual":1e-7}
source_spans={}
for lo,hi,name in [(801,811,"budget_assignment"),(5907,5912,"cli_switches"),(6063,6070,"budget_output")]:
    source_spans[name]=[f"{i+1}:{source_lines[i]}" for i in range(lo-1,hi)]
source_hash=sha(S)
expected_source="d96f4d6717ac40f47e71150666382ed5557c9c1e83b23b32a1c94877dbdb8db6"
assert source_hash==expected_source
assert spec525["source_frozen_sha256"]=="d5bf1eb738d25e29cba80f3c9c936963b0d1e8b533652384f60705008b946b82"
report={
 "schema":"numi.cardboard.native520_525_budget_effect_audit.v1",
 "verdict":"budget_group_changed; loaded control accepted under extended group; following control rejected",
 "source_snapshot_sha256":spec525["source_frozen_sha256"],
 "source_file":{"path":str(S),"sha256":source_hash},
 "source_spans":source_spans,
 "build_identity":spec525["files_sha256"],
 "material_sha256":spec525["material_sha256"],
 "native520":{
   "command_spec_sha256":sha(P520/"command-spec.json"),
   "capture_json_sha256":sha(P520/"capture.coupled.json"),
   "capture":{"root":cap520["root"],"newton_iteration":cap520["iteration"],"control_step":cap520["control_step"],
              "mixed_solver":mix520},
   "observed_failure":"root2 Newton9, FGMRES64, status code10; source-bound capture retained in the 520 package"
 },
 "native525":{
   "command_spec_sha256":sha(P525/"command-spec.json"),
   "receipt_sha256":sha(P525/"execution-525/receipt.json"),
   "receipt":{"exit_code":receipt525["exit_code"],"timed_out":receipt525["timed_out"],
              "wall_seconds":receipt525["wall_seconds"],"shipping_or_physical_validation":receipt525["shipping_or_physical_validation"]},
   "output_sha256":sha(P525/"execution-525/stdout.log"),
   "extended_budgets":extended,
   "preconditioner":"scalar_diagonal (selector 0); regional contact preconditioner was not selected",
   "loaded_control_trace":{"record_count":len(loaded),"accepted_nonnoop_records":sum(x["accepted"] and not x["certified_noop"] for x in loaded),
      "first_base_l2":loaded[0]["base_l2"],"last_nonnoop_trial_l2":loaded[24]["trial_l2"],
      "residual_gate":1e-7,"first_certified_noop_iteration":25,"last_iteration":31},
   "next_control_trace":{"record_count":len(rejected),"last_accepted_iteration":17,
      "first_unaccepted_iteration":18,"first_unaccepted_outcome":rejected[18]["trial_outcome"],
      "next_pretrial_status_code":rejected[19]["pretrial_status_code"],
      "next_pretrial_failing_index":rejected[19]["pretrial_failing_index"],
      "next_pretrial_fgmres_iterations":rejected[19]["pretrial_fgmres_iterations"],
      "next_pretrial_diagnostics":rejected[19]["pretrial_diagnostics"]},
   "transactions":tx,"baseline_alias_byte_parity":parity,"physical_validation":False
 },
 "comparison_limits":[
   "The --extended-budget variant changes Newton32, FGMRES256, restart16, and line-search24 together; baseline capture is Newton16, FGMRES64, restart10, line-search8. This does not isolate restart as the cause.",
   "525 does not use --regional-contact-preconditioner; it tests the scalar-diagonal preconditioner with changed work budgets.",
   "The loaded phase reaches the unchanged 1e-7 gate and accepts a control step, but the next control fails line search at Newton18 and is status-blocked at Newton19.",
   "This is the small synthetic full-contact fixture. It does not qualify a full shipping trajectory, material calibration, physical contact law, or completed simulation."
 ]
}
out=Q/"analysis/native520-525-budget-effect-audit-001"
(out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
print(json.dumps({"report":str(out/"report.json"),"report_sha256":sha(out/"report.json"),"source_sha256":source_hash,
  "loaded_rows":len(loaded),"rejected_rows":len(rejected),"transactions":tx},indent=2))
