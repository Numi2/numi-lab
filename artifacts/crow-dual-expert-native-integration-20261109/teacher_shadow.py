from pathlib import Path
import gzip,hashlib,json,os,shutil,subprocess
import numpy as np
from metalrobo.mlx_policy_learning import read_policy_rollout_pack
R=Path.cwd();A=R/'artifacts/crow-dual-expert-native-integration-20261109';B=R/'build/crow-dual-expert-native-integration-20261109';lib=R/'build/crow-dual-expert-native-observer-20261109/libmetalrobo.dylib'
cmd=json.loads((R/'artifacts/crow-dual-native-parity-20261109/integrated/reserved-final-c/results.json').read_text())['argv'].copy()
for k,v in [('--envs','4'),('--steps','2000'),('--seed','20270308')]:cmd[cmd.index(k)+1]=v
env={**os.environ,'CROW_CONTACT_TRACE':'1','DYLD_LIBRARY_PATH':str(lib.parent),'PYTHONDONTWRITEBYTECODE':'1'}
runs={}
for mode in ['off','shadow']:
 q=cmd.copy();q[q.index('--state-trace')+1]=str(B/(mode+'-shadow.tsv'))
 if mode=='shadow':q+=['--birdflow-journey-teacher','--birdflow-journey-student-authority','1','--rollout-pack',str(B/'shadow.rollout')]
 p=subprocess.run(q,cwd=R,env=env,text=True,capture_output=True,timeout=480)
 (A/(mode+'-shadow.stdout')).write_text(p.stdout)
 with (A/(mode+'-shadow.stderr.gz')).open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
 if p.returncode:raise RuntimeError(p.stderr[-5000:])
 runs[mode]=p
def samples(s):
 return {(x['lane'],x['step']):x for line in s.splitlines() if line.startswith('CROW_FLIGHT_SAMPLE ') for x in [json.loads(line.split(' ',1)[1])]}
a=samples(runs['off'].stderr);b=samples(runs['shadow'].stderr);first=json.loads(runs['off'].stdout)['first_episode_navigation']
keys=[k for k in a if k[1]<=first['steps_by_environment'][k[0]]]
bad=[k for k in keys if k not in b or a[k]['q']!=b[k]['q'] or a[k]['v']!=b[k]['v']]
pack=B/'shadow.rollout';r=read_policy_rollout_pack(pack,library_path=str(lib))
assert r.environment_count==4 and r.control_step_count==2000 and r.action_count==18
y=r.teacher_actions.reshape(2000,4,18);obs=r.actor_observations.reshape(2000,4,-1)
label_checks={}
for name,index,value,slot in [('stage2',147,.4,4),('stage3',169,.6,8)]:
 stage=np.isclose(obs[:,:,index],value,atol=1e-6)
 same=np.all(np.isclose(y[:,:,slot:slot+4],y[:,:,:4],atol=1e-6),axis=2)
 interior=stage.copy();interior[1:] &= stage[:-1];interior[:-1] &= stage[1:]
 assert interior.any(),name
 label_checks[name]={'interior_samples':int(interior.sum()),'mismatches':int(np.sum(interior & ~same))}
with pack.open('rb') as f,(A/'shadow.rollout.gz').open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:shutil.copyfileobj(f,z)
rawhash=hashlib.sha256(pack.read_bytes()).hexdigest()
assert hashlib.sha256(gzip.decompress((A/'shadow.rollout.gz').read_bytes())).hexdigest()==rawhash
report={'student_samples_compared':len(keys),'student_qv_mismatches':len(bad),'first_mismatches':bad[:10],'label_checks':label_checks,'raw_rollout_sha256':rawhash,'teacher_sampling':'labels on, student authority one; exact q/v check through first episode','note':'label check is not a PPO likelihood or stage-aware learner qualification'}
(A/'teacher-shadow-results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
assert keys and not bad and all(z['mismatches']==0 for z in label_checks.values())
