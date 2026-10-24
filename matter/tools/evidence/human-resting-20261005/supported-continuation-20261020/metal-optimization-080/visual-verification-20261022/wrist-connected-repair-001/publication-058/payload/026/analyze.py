from pathlib import Path
import json,hashlib
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=E/"native-candidate020-failure-witness-026";N=D/"run-001";sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((N/"common-field-failure.json").read_text());tr=[]
for s in (N/"stderr.log").read_text().splitlines():
 if s.startswith("fem_newton_iterate="):
  x=json.loads(s.split("=",1)[1])
  if x["microtick"]==0 and x["iteration"]<=5 and x["stage"]=="final_line_search":
   tr.append(dict(newton=x["iteration"],alpha=x["alpha"][0],skin_fat_line_search=x["object_line_search"][15],fgmres=x["fgmres"][0],status=x["status"][0]))
a=dict(status="REJECTED_PHYSICAL_STEP",accepted_steps=f["stand_completed_steps"],object=15,global_tet=int(f["matter_diagnostics"][2]),newton=f["matter_failing_index"],trial_J=f["matter_diagnostics"][0],current_J=f["matter_diagnostics"][1],reason_bits=int(f["matter_diagnostics"][3]),reason="finite trial determinant above fat upper validity J=2.5",trace=tr,initialization=dict(mode="dynamic_release",support_force_N=0,active_support_rows=0,balanced=False,maximum_acceleration_residual_m_s2=65.1741793495),interpretation="Current iterate remains inside material bounds, but successive corrections approach the upper volume bound while nonlinear residual decreases only modestly. No NaN or inverted-cell cause established. More line-search trials alone do not establish convergence. Forces, bounds and acceptance unchanged.",physiology_qualification=False,visual_verification=False,input_hashes={str(p.relative_to(D)):sha(p) for p in [N/"launch-manifest.snapshot.json",N/"execution-receipt.json",N/"common-field-failure.json",N/"stderr.log",D/"compatibility.json"]})
(D/"failure-analysis.json").write_text(json.dumps(a,indent=2)+"\n")
print(json.dumps({k:a[k] for k in ["status","accepted_steps","global_tet","trial_J","current_J","reason"]}))
