from pathlib import Path
import gzip,json,os,subprocess
import numpy as np
from metalrobo.mlx_policy_learning import read_policy_rollout_pack
R=Path.cwd();A=R/'artifacts/crow-hierarchical-contract-20261109';B=R/'build/crow-hierarchical-contract-20261109'
cmd=json.loads((A/'oracle/results.json').read_text())['argv']
for k,v in [('--metallib',str(B/'actor.metallib')),('--steps','100'),('--state-trace',str(B/'teacher-smoke.tsv'))]:cmd[cmd.index(k)+1]=v
cmd += ['--birdflow-journey-teacher','--birdflow-journey-student-authority','0','--rollout-pack',str(B/'teacher-smoke.rollout')]
env={**os.environ,'CROW_CONTACT_TRACE':'1','DYLD_LIBRARY_PATH':str(B),'PYTHONDONTWRITEBYTECODE':'1'};env.pop('CROW_SUPPORT_TRACE',None)
p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=180)
(A/'teacher-smoke.stdout').write_text(p.stdout);(A/'teacher-smoke.stderr').write_text(p.stderr)
if p.returncode:raise RuntimeError(p.stderr[-5000:])
roll=read_policy_rollout_pack(B/'teacher-smoke.rollout',library_path=str(B/'libmetalrobo.dylib'))
labels=roll.teacher_actions.reshape(100,16,-1)
out={'argv':cmd,'steps':roll.control_step_count,'envs':roll.environment_count,'action_count':roll.action_count,'teacher_action_size':roll.teacher_actions.size,'labels_finite':bool(np.isfinite(labels).all()),'labels_min':np.min(labels,axis=(0,1)).tolist(),'labels_max':np.max(labels,axis=(0,1)).tolist(),'bytes':(B/'teacher-smoke.rollout').stat().st_size}
(A/'teacher-smoke.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='argv'},indent=2))
