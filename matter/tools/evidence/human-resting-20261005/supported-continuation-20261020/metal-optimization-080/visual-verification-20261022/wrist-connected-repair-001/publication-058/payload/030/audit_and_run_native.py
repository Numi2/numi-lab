from pathlib import Path
import json,subprocess,hashlib,csv,numpy as np,time
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");N=E/"native-candidate020-conforming-prism-030";D=E/"native-candidate020-mechanical-source-024";Q=E/"native-candidate020-failure-witness-026"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 rows=list(csv.DictReader(p.open()))
 return rows,np.array([[float(r[k]) for k in ("x_m","y_m","z_m","mass_kg")] for r in rows])
base,b=read(D/"source-preflight-001/initial-skin-fem-nodes.csv");now,n=read(N/"source-preflight-005/initial-skin-fem-nodes.csv")
assert [(r["global_node"],r["layer"],r["row"]) for r in base]==[(r["global_node"],r["layer"],r["row"]) for r in now]
assert np.array_equal(b[:,:3].astype(np.float32),n[:,:3].astype(np.float32))
assert np.all(n[:,3]>0)
rel=np.abs(n[:,3]-b[:,3])/b[:,3]

assert abs(n[:,3].sum()-b[:,3].sum())<1e-6
a=np.load(E/"native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003/candidate.npz")
changedNodes=np.unique(np.r_[a["tetrahedra"][a["changed_tet_rows"]].ravel(),a["old_tetrahedra"][a["changed_tet_rows"]].ravel()])
outside=np.ones(len(n),bool);outside[changedNodes]=False
assert max(rel[outside])<1e-5
report={"status":"PASS_SOURCE_PREPARATION_AND_EMITTED_GEOMETRY","base_node_csv_sha256":sha(D/"source-preflight-001/initial-skin-fem-nodes.csv"),"candidate_node_csv_sha256":sha(N/"source-preflight-005/initial-skin-fem-nodes.csv"),"float32_positions_bitwise_identical":True,"node_identity_rows_exact":True,"physical_node_count":len(n),"mass_baseline_kg":float(b[:,3].sum()),"mass_candidate_kg":float(n[:,3].sum()),"mass_delta_kg":float(n[:,3].sum()-b[:,3].sum()),"topology_changed_node_count":len(changedNodes),"maximum_nodal_mass_relative_change_in_patch":float(max(rel[changedNodes])),"maximum_nodal_mass_relative_change_outside_patch":float(max(rel[outside])),"scope":"source preparation only; no native physical acceptance/clearance claim"}
(N/"source-parity-analysis.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report),flush=True)
cmd=json.loads((Q/"command.json").read_text())
cmd[1]=str(E/"native-candidate020-bone-witness-028/run_native_contact_checked.py")
cmd[cmd.index("--base-manifest")+1]=str(N/"source-preflight-005/launch-manifest.snapshot.json")
cmd[cmd.index("--app")+1]=str(N/"build-005/numi-human-native")
cmd[cmd.index("--run-dir")+1]=str(N/"native-1ms-001")
identity=json.loads((N/"candidate-identity.json").read_text())
for i,x in enumerate(cmd):
 if x=="--source-root" and cmd[i+1].startswith(str(D)):
  if cmd[i+1]==str(D/"apps"):cmd[i+1]=str(N/"apps")
  elif cmd[i+1].endswith("D38-candidate-identity.json"):cmd[i+1]=str(N/"candidate-identity.json")
  elif cmd[i+1].endswith("dependency-certification.json"):cmd[i+1]=str(N/"build-005/dependency-certification.json")
 if x.startswith("NUMI_HUMAN_EXPERIMENTAL_UNADMITTED_LAYER_CANDIDATE_ID="):
  cmd[i]="NUMI_HUMAN_EXPERIMENTAL_UNADMITTED_LAYER_CANDIDATE_ID="+identity["candidate_id"]
 if x.startswith("NUMI_HUMAN_EXPERIMENTAL_UNADMITTED_LAYER_CANDIDATE_SHA256="):
  cmd[i]="NUMI_HUMAN_EXPERIMENTAL_UNADMITTED_LAYER_CANDIDATE_SHA256="+sha(N/"candidate-identity.json")
cmd+=["--source-root",str(N/"source-parity-analysis.json"),"--source-root",str(N/"basis-capacity-count-fix.diff")]
(N/"native-1ms-command.json").write_text(json.dumps(cmd,indent=2)+"\n")
print("starting same-configuration bounded native mechanics diagnostic",flush=True)
p=subprocess.run(cmd)
(N/"native-runner-exit.json").write_text(json.dumps({"returncode":p.returncode}))
raise SystemExit(p.returncode)
