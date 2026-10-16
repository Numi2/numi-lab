from pathlib import Path
import json,csv,hashlib,datetime
q=Path("/Users/n/cardboard-implementation-20261014-host-macmini");c=q/"solver/fresh-source210-strict-inner-full-coupon-545";d=q/"analysis/native545-first-failure-adjudication-001";d.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=list(csv.DictReader((c/"output/observations.csv").open()))
steps={}
for r in rows:steps.setdefault(int(r["step"]),[]).append(r)
accepted=[k for k,v in steps.items() if len(v)==2 and all(x["step_accepted"]=="1" and x["status_code"]=="0" for x in v)]
failed=[(k,v) for k,v in steps.items() if any(x["step_accepted"]!="1" or x["status_code"]!="0" for x in v)]
assert accepted==list(range(19)) and len(failed)==1 and failed[0][0]==19
records=[]
for line in (c/"stderr.log").read_text().splitlines():
 if line.startswith("fem_true_residual_cycle="):
  v=json.loads(line.split("=",1)[1])
  if v["root"]==19:records.append(v)
assert len(records)==64*64
by={}
for row in records:by.setdefault(row["newton_iteration"],[]).append(row)
summary=[]
for n,v in sorted(by.items()):
 assert len(v)==64
 end=v[-1];e=end["environments"][0]
 summary.append({"newton_iteration":n,"rhs_l2":e["rhs_l2"],"final_cycle_true_residual_l2":e["true_l2"],"final_cycle_arnoldi_estimate":e["arnoldi_estimate"],"final_cycle_accepted":e["diagnostics"][2]>.5,"final_cycle_columns":e["diagnostics"][3],"all_cycles_solver_and_operator_healthy":all(x["solver_status_code"]==0 and x["operator_status_code"]==0 for record in v for x in record["environments"])})
(d/"control19-newton-linear-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
failure=[{k:r[k] for k in ["step","environment","status_code","step_accepted","certificate_residual","certificate_correction","fgmres_iterations","free_force_imbalance_l2_N","free_force_imbalance_max_N"]} for r in failed[0][1]]
cp=c/"output/checkpoints/coupon-control-000015.nmcb"
report={"schema":"numi.cardboard.native545-first-failure.v1","native_case":str(c),"native_receipt":json.loads((c/"receipt.json").read_text()),"source_frozen_sha256":"477fe6943fb1c28e782af9ce0c1c383c013146ab12f422bc55523a6834597931","source_build_sha256":"75f9a256c4413c0a32e877f2dfcc168633c7662b391690dd4b83dbd4819f8e93","requested_controls":60,"attempted_controls":20,"all_environment_accepted_controls":19,"accepted_control_indices":accepted,"first_failed_zero_based_control":19,"first_failed_one_based_attempt":20,"per_environment_failure":failure,"last_accepted_all_environment_control":18,"complete_history":False,"outer_residual_tolerance":5e-7,"newton_budget":64,"restart":16,"linear_budget":1024,"true_residual_cycle_records_at_failure":len(records),"last_newton_linear_summary":summary[-1],"linear_residual_is_not_nonlinear_certificate":True,"inference":"Stricter inner stopping does not by itself resolve the full-coupon nonlinear stagnation. Its failure state differs from native539 because strict stopping changes the preceding trajectory. No causal projection or operator fix is established by this run.","last_complete_periodic_checkpoint":{"control":15,"path":str(cp),"sha256":sha(cp),"bytes":cp.stat().st_size},"incorrect_intermediate_commentary_corrected":"Row count20 was initially read as20accepted; actual accepted flags show19all-environmentaccepted controls and failedcontrol19.","all_six_axes_remain_unqualified":True,"at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/"report.json").write_text(json.dumps(report,indent=2)+"\n")
inputs={str(p):sha(p) for p in [c/"receipt.json",c/"prediction.json",c/"native-environment.json",c/"output/observations.csv",c/"output/result.json",c/"stderr.log",cp]}
(d/"external-bindings.json").write_text(json.dumps(inputs,indent=2)+"\n")
(d/"manifest.json").write_text(json.dumps({p.name:sha(p) for p in d.iterdir() if p.is_file()},indent=2)+"\n")
print(json.dumps({"report_sha256":sha(d/"report.json"),"manifest_sha256":sha(d/"manifest.json"),"accepted":len(accepted),"failure":failure}),flush=True)
