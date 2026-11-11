from pathlib import Path
import collections,gzip,hashlib,json,os,re,subprocess,threading
import numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-v15-toe-joint-damping-high-screen-20261111';B=R/'build/crow-v15-toe-joint-damping-high-screen-20261111';P=R/'artifacts/crow-v15-two-segment-frozen-flight-second-seed-20261111';srcpath=R/'build/crow-v15-two-segment-frozen-flight-contact-map-20261111/two-segment-zero.policypack';ref=R/'build/crow-v15-two-segment-frozen-flight-dense-contact-20261111/libmetalrobo.dylib';src=read_policy_pack(srcpath,library_path=str(ref));h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert all(not np.any(x.weights) and not np.any(x.bias) for x in src.layers) and not np.any(src.action_bias)
for name in ('d060','d120'):
 D=B/name;lib=D/'libmetalrobo.dylib';cmd=json.loads((P/'command.json').read_text());cmd[cmd.index('--steps')+1]='250';cmd[cmd.index('--state-trace')+1]=str(D/'trace.tsv');env={**os.environ,'DYLD_LIBRARY_PATH':str(D),'CROW_CONTACT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'}
 probe=cmd.copy();probe[probe.index('--steps')+1]='1';p=subprocess.run(probe,cwd=R,env=env,capture_output=True,text=True);(A/(name+'-preflight.stderr')).write_text(p.stderr);(A/(name+'-preflight.stdout')).write_text(p.stdout)
 if p.returncode:
  m=re.search(r'task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',p.stderr);assert m,p.stderr;world,task,obs,act=map(int,m.groups());layers=[PolicyDenseLayerArtifact(weights=x.weights.copy(),bias=x.bias.copy(),activation=x.activation) for x in src.layers];target=D/'zero.policypack';write_policy_pack(target,policy_id='crow_toe_damping_diagnostic_zero',revision=src.revision+1,contract_version=src.contract_version,world_fingerprint=world,task_fingerprint=task,observation_fingerprint=obs,action_fingerprint=act,layers=layers,observation_mean=src.observation_mean,observation_inverse_standard_deviation=src.observation_inverse_standard_deviation,action_bias=src.action_bias,action_scale=src.action_scale,observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=str(lib));dst=read_policy_pack(target,library_path=str(lib));assert all(np.array_equal(x.weights,y.weights) and np.array_equal(x.bias,y.bias) and x.activation==y.activation for x,y in zip(src.layers,dst.layers));assert all(np.array_equal(getattr(src,z),getattr(dst,z)) for z in ('observation_mean','observation_inverse_standard_deviation','action_bias','action_scale'));assert src.observation_clip==dst.observation_clip and src.action_clip==dst.action_clip;cmd[cmd.index('--policy-pack')+1]=str(target);binding={'rebound':True,'identical_layers_and_transforms':True,'source_sha256':h(srcpath),'target_sha256':h(target)}
 else:binding={'rebound':False,'source_sha256':h(srcpath)}
 (A/(name+'-binding.json')).write_text(json.dumps(binding,indent=2)+'\n');(A/(name+'-command.json')).write_text(json.dumps(cmd,indent=2)+'\n');p=subprocess.Popen(cmd,cwd=R,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 def pump(src,path,zip):
  with path.open('wb') as f:
   if zip:
    with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:
     for b in iter(lambda:src.read(1<<16),b''):z.write(b)
   else:
    for b in iter(lambda:src.read(1<<16),b''):f.write(b)
 ts=[threading.Thread(target=pump,args=(p.stdout,A/(name+'.stdout'),False)),threading.Thread(target=pump,args=(p.stderr,A/(name+'.stderr.gz'),True))]
 for t in ts:t.start()
 rc=p.wait()
 for t in ts:t.join()
 hits=collections.Counter();seen=set();toes=collections.Counter();errors=[]
 with gzip.open(A/(name+'.stderr.gz'),'rt') as f:
  for l in f:
   if 'failed' in l.lower():errors.append(l.strip())
   if l.startswith('CROW_FLIGHT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);seen.add((x['step'],x['lane']));ind=[i for i,z in enumerate(x['v'][22:38]) if abs(z)>=11.99]
    if ind:
     hits[x['lane']]+=1
     for i in ind:toes[i]+=1
 out={'exit':rc,'observed_lane_steps':len(seen),'lanes_with_hits':len(hits),'lane_interval_hits':sum(hits.values()),'hits_by_lane':dict(hits),'hits_by_toe':dict(toes),'errors':errors[-3:]}
 try:
  data=json.loads((A/(name+'.stdout')).read_text());out['failed_environment_steps']=data['failed_environment_steps'];out['physical_failures']=sum(data['first_episode_navigation']['physical_failure_by_environment']);out['minimum_episode_steps']=min(data['first_episode_navigation']['steps_by_environment'])
 except Exception as e:out['output_error']=str(e)
 out['pass']=rc==0 and len(seen)>=80*250 and len(hits)==0 and out.get('failed_environment_steps')==0 and out.get('physical_failures')==0 and out.get('minimum_episode_steps')==250;(A/(name+'-analysis.json')).write_text(json.dumps(out,indent=2)+'\n');print(name,json.dumps(out),flush=True)
