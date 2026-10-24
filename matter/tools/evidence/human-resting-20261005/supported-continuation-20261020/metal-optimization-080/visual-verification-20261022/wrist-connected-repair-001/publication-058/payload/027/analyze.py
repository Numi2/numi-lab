from pathlib import Path
import json,hashlib
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=E/"native-candidate020-weighted-support-027";N=D/"run-001";baseline=E/"native-candidate020-failure-witness-026/run-001"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def trace(p):
 result=[]
 for s in (p/"stderr.log").read_text().splitlines():
  if s.startswith("fem_newton_iterate="):
   x=json.loads(s.split("=",1)[1])
   if x["microtick"]==0 and x["iteration"]<=5 and x["stage"]=="final_line_search":
    result.append({k:x[k] for k in ["iteration","alpha","object_line_search","fgmres","status"]})
 return result
a=trace(N);b=trace(baseline);f=json.loads((N/"common-field-failure.json").read_text())
r=dict(status="REJECTED_PHYSICAL_STEP",accepted_steps=f["stand_completed_steps"],physical_failure=dict(object=f["matter_object_index"],newton=f["matter_failing_index"],trial_J=f["matter_diagnostics"][0],current_J=f["matter_diagnostics"][1],global_tet=f["matter_diagnostics"][2],reason_bits=f["matter_diagnostics"][3]),first_six_newton_trace_equal_to_026=a==b,initialization=dict(registered_skin_bed_rows=32,minimum_gap_m=5.8691777669e-6,closed_rows=0,active_support_rows=0,support_force_N=0,maximum_acceleration_residual_m_s2=9.75447409036,baseline_maximum_acceleration_residual_m_s2=65.1741793495),conclusion="Existing weighted support capture and changed source recruitment improve initial articulated residual but leave the first skin-fat Matter failure and recorded Newton path identical. This does not isolate the forcing source; it rules out promoting this initialization change as a fix for the observed rejection.",supported_rest_qualification=False,physics_qualification=False,visual_verification=False,hashes={str(p.relative_to(D)):sha(p) for p in [N/"launch-manifest.snapshot.json",N/"execution-receipt.json",N/"common-field-failure.json",D/"initialization-policy.json"]})
(D/"comparison.json").write_text(json.dumps(r,indent=2)+"\n")
print(json.dumps(r)[:1600])
