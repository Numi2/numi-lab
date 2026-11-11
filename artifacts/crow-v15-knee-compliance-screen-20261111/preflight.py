from pathlib import Path
import json,os,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-knee-compliance-screen-20261111";B=R/"build/crow-v15-knee-compliance-screen-20261111";cmd=json.loads((R/"artifacts/crow-v15-walk-seed27-reference-20261111/autonomous-20261127/command.json").read_text())
for k,v in [("--steps","2"),("--seed","20261126"),("--state-trace",str(B/"preflight.tsv")),("--crow-replay-pack",str(B/"preflight-replay.json"))]:cmd[cmd.index(k)+1]=v
(A/"preflight-command.json").write_text(json.dumps(cmd,indent=2)+"\n")
p=subprocess.run(cmd,cwd=R,env={**os.environ,"DYLD_LIBRARY_PATH":str(B),"PYTHONDONTWRITEBYTECODE":"1"},capture_output=True,text=True,timeout=180)
(A/"preflight.stdout").write_text(p.stdout);(A/"preflight.stderr").write_text(p.stderr);(A/"preflight-result.json").write_text(json.dumps({"return_code":p.returncode},indent=2)+"\n");print("return_code",p.returncode);print(p.stderr[-1800:])
