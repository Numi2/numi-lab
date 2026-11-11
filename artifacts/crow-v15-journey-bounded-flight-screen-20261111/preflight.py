from pathlib import Path
import json,os,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-journey-bounded-flight-screen-20261111";B=R/"build/crow-v15-journey-bounded-flight-screen-20261111";O=R/"build/crow-v15-hip-roll-obstacle-free-control-20261111";J=R/"artifacts/crow-v15-hip-roll-journey-integration-20261111"
cmd=json.loads((J/"rebound-command.json").read_text());cmd[0]=str(R/"build/crow-v15-replay-writer-roundtrip-20261111/metalrobo_task_rollout")
for k in ("--birdflow-journey-teacher",):
 if k in cmd:cmd.remove(k)
if "--birdflow-journey-student-authority" in cmd:
 i=cmd.index("--birdflow-journey-student-authority");del cmd[i:i+2]
for k,v in [("--envs","5"),("--steps","2"),("--seed","20261120"),("--metallib",str(B/"tracking.metallib")),("--state-trace",str(B/"preflight.tsv")),("--crow-replay-pack",str(B/"preflight.json"))]:cmd[cmd.index(k)+1]=v
(A/"preflight-command.json").write_text(json.dumps(cmd,indent=2)+"\n");p=subprocess.run(cmd,cwd=R,env={**os.environ,"DYLD_LIBRARY_PATH":str(O),"PYTHONDONTWRITEBYTECODE":"1"},capture_output=True,text=True,timeout=180);(A/"preflight.stdout").write_text(p.stdout);(A/"preflight.stderr").write_text(p.stderr);(A/"preflight-result.json").write_text(json.dumps({"exit_code":p.returncode},indent=2)+"\n");print("preflight exit",p.returncode,p.stderr[-1500:])
