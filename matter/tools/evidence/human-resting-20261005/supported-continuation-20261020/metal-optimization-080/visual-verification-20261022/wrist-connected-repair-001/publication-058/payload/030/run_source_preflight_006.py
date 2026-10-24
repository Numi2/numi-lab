from pathlib import Path
import json,subprocess,hashlib
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");N=E/"native-candidate020-conforming-prism-030";D=E/"native-candidate020-mechanical-source-024"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
identity=json.loads((N/"candidate-identity.json").read_text())
cmd=json.loads((D/"source-preflight-command.json").read_text())
cmd[1]=str(E/"native-candidate020-bone-witness-028/run_native_contact_checked.py")
cmd[cmd.index("--app")+1]=str(N/"build-006/numi-human-native")
cmd[cmd.index("--run-dir")+1]=str(N/"source-preflight-006")
for i in range(len(cmd)):
 if cmd[i]=="--source-root" and cmd[i+1].startswith(str(D)):
  old=cmd[i+1]
  if old==str(D/"apps"):cmd[i+1]=str(N/"apps")
  elif old.endswith("D38-candidate-identity.json"):cmd[i+1]=str(N/"candidate-identity.json")
  elif old.endswith("dependency-certification.json"):cmd[i+1]=str(N/"build-006/dependency-certification.json")
 if cmd[i].startswith("NUMI_HUMAN_EXPERIMENTAL_UNADMITTED_LAYER_CANDIDATE_ID="):
  cmd[i]="NUMI_HUMAN_EXPERIMENTAL_UNADMITTED_LAYER_CANDIDATE_ID="+identity["candidate_id"]
 if cmd[i].startswith("NUMI_HUMAN_EXPERIMENTAL_UNADMITTED_LAYER_CANDIDATE_SHA256="):
  cmd[i]="NUMI_HUMAN_EXPERIMENTAL_UNADMITTED_LAYER_CANDIDATE_SHA256="+sha(N/"candidate-identity.json")
cmd+=["--source-root",str(N/"app-integration.diff")]
(N/"source-preflight-command.json").write_text(json.dumps(cmd,indent=2))
assert json.loads((N/"build-006/dependency-certification.json").read_text())["status"]=="PASS_EXACT_OVERLAY"
p=subprocess.run(cmd)
(N/"source-preflight-runner-exit.json").write_text(json.dumps({"returncode":p.returncode}))
raise SystemExit(p.returncode)
