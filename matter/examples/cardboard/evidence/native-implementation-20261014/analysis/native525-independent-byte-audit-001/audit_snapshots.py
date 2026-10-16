from pathlib import Path
import hashlib,json
BASE=Path("/Users/n/cardboard-implementation-20261014-host-macmini")
PKG=BASE/"analysis/native525-contact-restart16-command-001"
CASE=BASE/"native/integrated195-contact-restart16-525"
BUILD=BASE/"build-source195-001"
OUT=BASE/"analysis/native525-independent-byte-audit-001"
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def inventory(path):
    return {str(p.relative_to(path)):{"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(path.rglob("*")) if p.is_file()}
spec=json.loads((PKG/"command-spec.json").read_text())
receipt=json.loads((PKG/"execution-525/receipt.json").read_text())
b=inventory(CASE/"baseline")
a=inventory(CASE/"aliased")
assert set(b)==set(a), "baseline and alias path sets differ"
pair_equal={name:b[name]==a[name] for name in b}
assert all(pair_equal.values()), "baseline and alias snapshot bytes differ"
receipt_expected=receipt["case_files"]
actual={}
for phase in ("baseline","aliased"):
    for name,entry in inventory(CASE/phase).items():
        actual[f"{phase}/{name}"]=entry
assert set(receipt_expected)==set(actual), "receipt and actual file inventories differ"
receipt_equal={name:receipt_expected[name]==actual[name] for name in actual}
assert all(receipt_equal.values()), "receipt hash/size mismatch"
assert len(b)==13 and len(actual)==26
assert receipt["exit_code"]==0 and not receipt["timed_out"] and not receipt["space_abort"]
stdout=PKG/"execution-525/stdout.log"
stderr=PKG/"execution-525/stderr.log"
assert sha(stdout)==receipt["stdout_sha256"]
assert sha(stderr)==receipt["stderr_sha256"]
source=BASE/spec["source_snapshot"]
frozen=source/"frozen-source.json"
assert sha(frozen)==spec["source_frozen_sha256"]
build_checks={}
for rel,want in spec["files_sha256"].items():
    p=BUILD/rel
    got=sha(p) if p.is_file() else None
    build_checks[rel]={"expected_sha256":want,"actual_sha256":got,"matches":got==want}
assert all(x["matches"] for x in build_checks.values())
mat=source/spec["material_relative_path"]
material={"path":str(mat),"expected_sha256":spec["material_sha256"],"actual_sha256":sha(mat),"matches":sha(mat)==spec["material_sha256"]}
assert material["matches"]
tx=[]
for i in range(3):
    p=CASE/"baseline"/f"transaction-{i:06d}.json"
    d=json.loads(p.read_text())
    before=d["before"]; after=d["after"]
    bh=before.get("accepted_state_vector_sha256",{})
    ah=after.get("accepted_state_vector_sha256",{})
    accepted=bool(d["accepted"])
    tx.append({
        "index":i,"requested_control_step":d["requested_control_step"],
        "accepted":accepted,"completion_stage":d["completion_stage"],
        "before_control_step":before["control_step"],"after_control_step":after["control_step"],
        "before_status_codes":[x["code"] for x in before["statuses"]],
        "after_status_codes":[x["code"] for x in after["statuses"]],
        "before_fgmres_iterations":[x["fgmres_iterations"] for x in before["statuses"]],
        "after_fgmres_iterations":[x["fgmres_iterations"] for x in after["statuses"]],
        "accepted_vector_hashes_equal":bh==ah,
        "rollback_valid":(not accepted and before["control_step"]==after["control_step"] and bh==ah) if not accepted else None,
        "rejection_outcome":[x.get("trial_outcome") for x in after.get("environments",[])],
        "after_solver_diagnostics":[x.get("diagnostics") for x in after["statuses"]],
        "transaction_json_baseline_alias_equal":(CASE/"aliased"/p.name).read_bytes()==p.read_bytes()
    })
assert [x["accepted"] for x in tx]==[True,True,False]
assert tx[2]["rollback_valid"] and all(x["transaction_json_baseline_alias_equal"] for x in tx)
outtext=stdout.read_text()
wanted_lines=[
 "authored_extended_budget=fgmres256,restart16,line_search24 relative_residual_unchanged=1e-7",
 "authored_line_search_budget=pass accepted=1,8,32 rejected=0,33",
 "selected_trial=pass accepted_alpha_matches_applied_alpha=true private_to_next_production_base_relative_error=0",
 "full_contact_rejection_rollback=pass certificate_cleared=true cursor_and_all_histories_unchanged=true",
 "PASS full-contact self-surface scratch alias parity checks=5976 native_fem_transactions_executed=6 physical_validation=false synthetic_law=true"
]
assert all(line in outtext for line in wanted_lines)
summary={
 "schema":"numi.cardboard.native525-independent-byte-audit.v1",
 "inputs":{
  "command_spec":{"path":str(PKG/"command-spec.json"),"sha256":sha(PKG/"command-spec.json")},
  "receipt":{"path":str(PKG/"execution-525/receipt.json"),"sha256":sha(PKG/"execution-525/receipt.json")},
  "source_frozen":{"path":str(frozen),"sha256":sha(frozen),"matches_command_spec":True},
  "stdout_sha256":sha(stdout),"stderr_sha256":sha(stderr)
 },
 "run":{
  "exit_code":receipt["exit_code"],"timed_out":receipt["timed_out"],"space_abort":receipt["space_abort"],
  "wall_seconds":receipt["wall_seconds"],"shipping_or_physical_validation":receipt["shipping_or_physical_validation"],
  "argv":spec["cases"]["525"],"environment":spec["environment"]["525"],
  "configuration_note":spec["qualification"]
 },
 "build_file_checks":build_checks,"material_check":material,
 "snapshot_byte_audit":{
  "baseline_snapshot_files":len(b),"alias_snapshot_files":len(a),"paired_files":len(pair_equal),
  "all_pairs_identical":all(pair_equal.values()),"receipt_file_entries":len(receipt_expected),
  "all_receipt_entries_match":all(receipt_equal.values()),"baseline_inventory":b,
  "transaction_summaries":tx
 },
 "checker_evidence":{
  "required_stdout_lines":wanted_lines,
  "selected_trial_pass_lines":outtext.count("selected_trial=pass accepted_alpha_matches_applied_alpha=true private_to_next_production_base_relative_error=0"),
  "rollback_pass_lines":outtext.count("full_contact_rejection_rollback=pass certificate_cleared=true cursor_and_all_histories_unchanged=true")
 },
 "interpretation":"This byte audit independently confirms the source/build/material bindings and exact equality of baseline versus scratch-alias runs across all 13 files per run, including all six before/after snapshots. The checker accepts controls 1 and 2; control 3 is rejected before a private Armijo trial because its pretrial solver status is code 10 (failing index 18, FGMRES iteration count 256), and the accepted state rolls back byte-for-byte at the accepted-vector hash level with control step remaining 2. The larger restart/FGMRES budget therefore resolves the earlier short-budget stagnation for the first two steps of this bounded fixture, but does not make the third step pass. This is a synthetic alias/transaction regression, not full coupon or physical qualification."
}
report=OUT/"report.json"
report.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
readme="""# Native 525 independent byte audit

This package records a read-only audit of source-bound native 525 outputs.

To reproduce the audit on the same host, run python3 audit_snapshots.py. The script binds the command spec and receipt, verifies the source frozen-map hash and all five build-artifact hashes, checks the material asset hash, then hashes every file in the baseline and alias directories. It requires identical path sets and bytes between those runs and compares every actual file size/SHA against the native receipt. It also checks the transaction JSON accepted/rejected/rollback semantics and required checker stdout lines.

The audited output has 13 files in each run: 3 transaction JSON files, 6 raw before/after snapshots, and 4 logs. All 13 baseline/alias pairs match exactly and all 26 outputs match the receipt. Controls 1 and 2 are accepted. The third transaction is rejected before its trial because of pretrial status code 10 at index 18 after 256 FGMRES iterations; its accepted-vector hashes and control cursor roll back to control 2.

The checker reports 5,976 assertions and 6 native FEM transactions across baseline and alias runs. The run uses the synthetic fixture. It demonstrates alias parity and bounded transaction behavior only; it is not full coupon, shipping, or physical qualification.
"""
(OUT/"README.md").write_text(readme)
manifest={
 "schema":"numi.cardboard.audit-package.v1",
 "id":"native525-independent-byte-audit-001",
 "inputs":{
  str(PKG/"command-spec.json"):sha(PKG/"command-spec.json"),
  str(PKG/"execution-525/receipt.json"):sha(PKG/"execution-525/receipt.json"),
  str(stdout):sha(stdout),str(stderr):sha(stderr),str(frozen):sha(frozen),
  **{str(BUILD/rel):v["actual_sha256"] for rel,v in build_checks.items()},
  str(mat):sha(mat)
 },
 "outputs":{"README.md":sha(OUT/"README.md"),"audit_snapshots.py":sha(Path(__file__)),"report.json":sha(report)},
 "result":{"baseline_alias_pairs_identical":len(pair_equal),"receipt_entries_verified":len(receipt_equal),"accepted_controls":2,"rejected_and_rolled_back":1,"assertions":5976,"native_transactions":6}
}
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
print(json.dumps({"out":str(OUT),"report_sha256":sha(report),"manifest_sha256":sha(OUT/"manifest.json"),"audit_script_sha256":sha(Path(__file__)),"readme_sha256":sha(OUT/"README.md"),"snapshot_pairs":len(pair_equal),"receipt_entries":len(receipt_equal),"transactions":tx},indent=2))
