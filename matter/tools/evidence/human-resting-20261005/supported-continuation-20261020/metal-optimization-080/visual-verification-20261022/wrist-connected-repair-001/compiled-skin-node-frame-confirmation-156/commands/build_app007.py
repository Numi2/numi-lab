from pathlib import Path
import json,subprocess,time,hashlib
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
E=R/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
D=E/"native-app-main-header-integration-151"; P=E/"bone-triangle-contact-037"; B=D/"build-007"; B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for phase in ["compile","link"]:
 a=json.loads((P/(phase+"-command-009.json")).read_text())
 a=[x.replace(str(P/"build-009"),str(B)).replace(str(P/"apps"),str(D/"apps-007")) for x in a]
 if phase=="compile": a[1:1]=["-I",str(D/"snapshot-005/matter/include")]
 (B/(phase+"-command.json")).write_text(json.dumps(a,indent=2))
 t=time.monotonic()
 with (B/(phase+".log")).open("x") as f:p=subprocess.run(a,cwd=R,stdout=f,stderr=subprocess.STDOUT)
 r=dict(returncode=p.returncode,wall_seconds=time.monotonic()-t,source_sha256=sha(D/"apps-007/numilab_human_myosim_visual_probe.mm"))
 if phase=="link" and p.returncode==0:r["binary_sha256"]=sha(B/"numi-human-native")
 (B/(phase+"-result.json")).write_text(json.dumps(r,indent=2));print(phase,json.dumps(r),flush=True)
 if p.returncode:raise SystemExit(p.returncode)
