import json,subprocess,sys
from pathlib import Path
root="/Users/numi/numi-reconciliation-build-20261108"
adapter=root+"/numi_human_costal_ownership_native";owner=root+"/numi_human_costal_real_owner_probe";shader=root+"/matter/shaders/NumiMatter.metallib"
cases=[("adapter-empty",[adapter,shader,"--adapter-control"]),("adapter-contact",[adapter,shader,"--adapter-control","--body-contact"]),("owner-reject",[owner]),("owner-accept",[owner,"--accept"]),("owner-replay",[owner,"--replay"]),("owner-continue",[owner,"--continue"]),("owner-continue-repeat",[owner,"--continue"])]
out=[]
for name,cmd in cases:
 try:
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=120);record={"name":name,"command":cmd,"exit":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
 except subprocess.TimeoutExpired as e:
  record={"name":name,"command":cmd,"timeout":True,"stdout":(e.stdout or b"").decode(errors="replace") if isinstance(e.stdout,bytes) else (e.stdout or ""),"stderr":(e.stderr or b"").decode(errors="replace") if isinstance(e.stderr,bytes) else (e.stderr or "")}
 out.append(record);Path("native.json").write_text(json.dumps(out,indent=2)+"\n");print(name,record.get("exit","TIMEOUT"),flush=True)
