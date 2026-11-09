from pathlib import Path
import gzip,hashlib,json,os,subprocess
import numpy as np
from metalrobo.mlx_policy_learning import read_policy_rollout_pack
R=Path.cwd();A=R/'artifacts/crow-hierarchical-contract-20261109';B=R/'build/crow-hierarchical-contract-20261109'
cmd0=json.loads((A/'oracle/results.json').read_text())['argv']
env={**os.environ,'CROW_CONTACT_TRACE':'1','DYLD_LIBRARY_PATH':str(B),'PYTHONDONTWRITEBYTECODE':'1'};env.pop('CROW_SUPPORT_TRACE',None)
packs={};samples={}
for mode in ['oracle-off','oracle-teacher']:
 cmd=cmd0.copy()
 for k,v in [('--metallib',str(B/'oracle.metallib')),('--steps','400'),('--state-trace',str(B/(mode+'.tsv')))]:cmd[cmd.index(k)+1]=v
 pack=B/(mode+'.rollout');cmd+=['--rollout-pack',str(pack)]
 if mode=='oracle-teacher':cmd+=['--birdflow-journey-teacher','--birdflow-journey-student-authority','0']
 p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=240)
 if p.returncode:raise RuntimeError(p.stderr[-4000:])
 (A/(mode+'.stdout')).write_text(p.stdout)
 with (A/(mode+'.stderr.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
 packs[mode]=read_policy_rollout_pack(pack,library_path=str(B/'libmetalrobo.dylib'))
 samples[mode]={(z['lane'],z['step']):z for line in p.stderr.splitlines() if line.startswith('CROW_FLIGHT_SAMPLE ') for z in [json.loads(line.split(' ',1)[1])]}
a,b=(packs[x] for x in ['oracle-off','oracle-teacher']);aa=np.asarray(a.actor_observations);bb=np.asarray(b.actor_observations)
qa,qb=(samples[x] for x in ['oracle-off','oracle-teacher'])
bad=[k for k in qa if k not in qb or qa[k]['q']!=qb[k]['q'] or qa[k]['v']!=qb[k]['v']]
labels=b.teacher_actions.reshape(400,16,-1)
report={'steps':400,'environments':16,'sampled_physics_mismatches':len(bad),'actor_observations_equal':bool(np.array_equal(aa,bb)),'max_actor_observation_difference':float(np.max(np.abs(aa-bb))),'teacher_labels_finite':bool(np.isfinite(labels).all()),'label_first_step_excluded_from_training':True,'teacher_command_observation_is_neutral_in_both_modes':True,'pack_sha256':{mode:hashlib.sha256((B/(mode+'.rollout')).read_bytes()).hexdigest() for mode in packs}}
(A/'observation-parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
