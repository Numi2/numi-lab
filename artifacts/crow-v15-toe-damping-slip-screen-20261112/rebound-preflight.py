from pathlib import Path
import json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-toe-damping-slip-screen-20261112';B=R/'build/crow-v15-toe-damping-slip-screen-20261112';cmd=json.loads((A/'preflight-command.json').read_text())
for opt,label in (('--policy-pack','landing'),('--base-policy-pack','route')):cmd[cmd.index(opt)+1]=str(B/(label+'-candidate.policypack'))
(A/'rebound-preflight-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'CROW_EXPERIMENT_GAIT_SMOOTH':'1','PYTHONDONTWRITEBYTECODE':'1'}
p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=180)
(A/'rebound-preflight.stdout').write_text(p.stdout);(A/'rebound-preflight.stderr').write_text(p.stderr);(A/'rebound-preflight-result.json').write_text(json.dumps({'return_code':p.returncode},indent=2)+'\n');print('rebound preflight exit',p.returncode);assert p.returncode==0,p.stderr[-2000:]
