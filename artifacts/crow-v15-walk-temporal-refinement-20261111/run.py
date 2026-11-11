from pathlib import Path
import gzip,hashlib,json,os,re,subprocess
import numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-v15-walk-temporal-refinement-20261111';B=R/'build/crow-v15-walk-temporal-refinement-20261111';H=R/'build/crow-v15-head-contact-fix-20261110';cmd0=json.loads((R/'artifacts/crow-v15-toe-launch-control-20261110/control-bindings.json').read_text())['argv'];h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def command(arm,steps,packs):
 cmd=cmd0.copy()
 cmd[cmd.index('--physics-substeps')+1]=''.join(c for c in arm[4:] if c.isdigit())
 for k,v in [('--steps',str(steps)),('--policy-pack',str(packs['landing'])),('--base-policy-pack',str(packs['route'])),('--state-trace',str(B/(arm+'.tsv'))),('--crow-replay-pack',str(B/(arm+'.json')))]:cmd[cmd.index(k)+1]=v
 return cmd
for arm in ('zero32','zero64'):
 lib=R/'build/crow-v15-walk-substep-damping-comparison-20261111/zero/libmetalrobo.dylib';env={**os.environ,'DYLD_LIBRARY_PATH':str(lib.parent),'PYTHONDONTWRITEBYTECODE':'1','CROW_SUBSTEP_TRACE':'1','CROW_SUPPORT_TRACE':'1','CROW_CONTACT_TRACE':'1','CROW_CLEARANCE_TRACE':'1'};packs={k:H/(k+'.policypack') for k in ('landing','route')}
 trial=subprocess.run(command(arm+'-preflight',1,packs),cwd=R,env=env,capture_output=True,text=True,timeout=120);(A/(arm+'-preflight.stdout')).write_text(trial.stdout);(A/(arm+'-preflight.stderr')).write_text(trial.stderr);report={'initial_returncode':trial.returncode,'actors':[]}
 if trial.returncode:
  m=re.search(r'task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',trial.stderr)
  if not m:raise RuntimeError('Preflight failure is not a task contract rejection: '+trial.stderr[-1200:])
  world,task,obs,act=map(int,m.groups());report['contract']={'world':world,'task':task,'observation':obs,'action':act}
  for k in ('landing','route'):
   source=packs[k];src=read_policy_pack(source,library_path=str(lib));assert not src.critic_layers and len(src.action_log_standard_deviation)==0
   layers=[PolicyDenseLayerArtifact(weights=x.weights.copy(),bias=x.bias.copy(),activation=x.activation) for x in src.layers];target=B/arm/(k+'.policypack')
   write_policy_pack(target,policy_id='crow_walk_diagnostic_'+arm+'_'+k,revision=src.revision+1,contract_version=src.contract_version,world_fingerprint=world,task_fingerprint=task,observation_fingerprint=obs,action_fingerprint=act,layers=layers,observation_mean=src.observation_mean,observation_inverse_standard_deviation=src.observation_inverse_standard_deviation,action_bias=src.action_bias,action_scale=src.action_scale,observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=str(lib));dst=read_policy_pack(target,library_path=str(lib))
   assert len(src.layers)==len(dst.layers) and all(np.array_equal(x.weights,y.weights) and np.array_equal(x.bias,y.bias) and x.activation==y.activation for x,y in zip(src.layers,dst.layers))
   for field in ('observation_mean','observation_inverse_standard_deviation','action_bias','action_scale'):assert np.array_equal(getattr(src,field),getattr(dst,field))
   assert src.observation_clip==dst.observation_clip and src.action_clip==dst.action_clip;report['actors'].append({'name':k,'source_sha256':h(source),'target_sha256':h(target),'weights_and_transforms_identical':True});packs[k]=target
 cmd=command(arm,201,packs);report.update({'argv':cmd,'library_sha256':h(lib),'shader_sha256':h(Path(cmd[cmd.index('--metallib')+1])),'policy_sha256':{k:h(p) for k,p in packs.items()}});(A/(arm+'-bindings.json')).write_text(json.dumps(report,indent=2)+'\n')
 with (A/(arm+'.stdout')).open('wb') as so,(A/(arm+'.stderr.gz')).open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as se:
   p=subprocess.Popen(cmd,cwd=R,env=env,stdout=so,stderr=subprocess.PIPE)
   while True:
    b=p.stderr.read(1<<20)
    if not b:break
    se.write(b)
   rc=p.wait()
 if rc:raise RuntimeError(arm+' run exit '+str(rc))
 for src,suffix in [(B/(arm+'.json'),'replay.json.gz'),(B/(arm+'.tsv'),'trace.tsv.gz')]:
  with src.open('rb') as f,(A/(arm+'.'+suffix)).open('wb') as raw:
   with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
    for b in iter(lambda:f.read(1<<20),b''):z.write(b)
 print(arm,'completed',flush=True)
