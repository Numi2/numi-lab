from pathlib import Path
import json,hashlib
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");N=E/"native-candidate020-conforming-prism-030";B=E/"native-candidate020-failure-witness-026/run-001";R=N/"native-1ms-002"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def analyze(p):
 f=json.loads((p/"common-field-failure.json").read_text())
 tr=[]
 for line in (p/"stderr.log").read_text().splitlines():
  if not line.startswith("fem_newton_iterate="):continue
  x=json.loads(line.split("=",1)[1])
  if x["microtick"]!=0 or x["stage"]!="final_line_search":continue
  tr.append({"iteration":x["iteration"],"alpha":x["alpha"][0],"skin_min_J":x["object_line_search"][15][1],"nonlinear_residual":x["fgmres"][0]["nonlinear"][1],"correction_ratio":x["fgmres"][0]["nonlinear"][2],"status":x["status"][0]["code"],"diagnostics":x["status"][0]["diagnostics"],"fgmres_iterations_microstep_max":x["status"][0]["fgmres_iterations_microstep_max"]})
  if tr[-1]["status"]!=0:break
 return {"accepted_physical_steps":f["stand_completed_steps"],"matter_microsteps":f["matter_microsteps"],"failing_object":f["matter_object_index"],"failing_Newton_iteration":f["matter_failing_index"],"trial_J":f["matter_diagnostics"][0],"current_J":f["matter_diagnostics"][1],"global_tet":int(f["matter_diagnostics"][2]),"reason_bits":int(f["matter_diagnostics"][3]),"trace":tr,"wall_seconds":json.loads((p/"execution-receipt.json").read_text())["wall_seconds"],"input_sha256":{n:sha(p/n) for n in ["execution-receipt.json","launch-manifest.snapshot.json","common-field-failure.json","stderr.log","stdout.log"]}}
b=analyze(B);a=analyze(R)
assert b["accepted_physical_steps"]==a["accepted_physical_steps"]==0
assert b["global_tet"]==29419 and a["global_tet"]==19641
report={"status":"NATIVE_REJECTED_STEP_OLD_FIRST_BLOCKER_CLEARED_NEW_BLOCKER_RETAINED","baseline":b,"candidate":a,"source_parity_report_sha256":sha(N/"source-parity-analysis-006.json"),"source_preflight_receipt_sha256":sha(N/"source-preflight-006/execution-receipt.json"),"observations":{"initial_residual_relative_difference":a["trace"][0]["nonlinear_residual"]/b["trace"][0]["nonlinear_residual"]-1,"baseline_first_alpha":b["trace"][0]["alpha"],"candidate_first_alpha":a["trace"][0]["alpha"],"baseline_failed_residual":b["trace"][-1]["nonlinear_residual"],"candidate_failed_residual":a["trace"][-1]["nonlinear_residual"],"baseline_failed_correction_ratio":b["trace"][-1]["correction_ratio"],"candidate_failed_correction_ratio":a["trace"][-1]["correction_ratio"]},"limits":["zero accepted physical steps; no integrated throughput or 0.1x claim","no visual verification","unsupported release diagnostic, not supported periodic rest","138 nearest-bank closed-solid skin/fat penetration witnesses remain unresolved","material laws, validity limits, force ownership and acceptance gates unchanged","connectivity changed16cells; nodal lumped mass redistributed among14nodes, not an equivalent-discretization parity claim","native001 stopped before physical stepping on object-name binding; retained; native002 reached actualGPUsolver"]}
(N/"native-comparison-analysis.json").write_text(json.dumps(report,indent=2)+"\n")
(N/"README.md").write_text("""# Conforming D37 skin/fat prism candidate 030

This private candidate changes 16 tetrahedra around the left palm while preserving every O/I/D coordinate and boundary triangle. The existing unpaired volume owner validates exact-overlap ancestry, positive emitted Float32 cells, material boundaries, and conserved skin/fat debit. Cell mixtures use raw overlap divided by new-child volume; original parent-volume fractions are retained separately.

The app retains the original ObjectSource descriptor and physical owners, regenerates per-cell source labels through the baseline compiler, reuses existing regional basis IDs, and applies the corrected distinct-region capacity check. All preparation and native failures are retained.

Source preflight 006 passes. All 7,233 emitted node positions are bitwise identical to 024, total skin/fat mass differs by 7.28e-12 kg, and mass outside the14 affected nodes is unchanged. Local nodal mass redistribution reaches30.76%; this is a changed discretization, not equivalent-algorithm parity.

Native002 requested1ms and rejected its first physical step after90.06s. The former first failing palm cell is no longer first; failure is now fat globaltet19641 atNewton8, trialJ2.50019526/currentJ2.49996257. Initialalpha improves0.125→1 and failed nonlinear residual0.00459133→0.000608571, but neither observation establishes convergence or speedup. Zero accepted steps means no sustained-throughput claim.

The source-only contact capture029 retains138 nearest-bank inside-solid witnesses. This candidate does not fix those penetrations, qualify anatomy, demonstrate joint motion, or close the full Human deliverable.

See native-comparison-analysis.json and source-parity-analysis-006.json for exact pins and numerical details. Use retained command JSONs; builds and measurements belong to SSHNumi.
""")
print(json.dumps({"status":report["status"],"baseline_last_residual":b["trace"][-1]["nonlinear_residual"],"candidate_last_residual":a["trace"][-1]["nonlinear_residual"],"new_global_tet":a["global_tet"],"accepted_steps":a["accepted_physical_steps"],"analysis_sha256":sha(N/"native-comparison-analysis.json")}))
