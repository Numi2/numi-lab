from pathlib import Path
import gzip,json,math,os,re,subprocess,hashlib
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-v15-toe-contact-isolation-20261110';B=R/'build/crow-v15-toe-contact-isolation-20261110';rows=[]
for body in ('rigid',):
 P=R/'artifacts/crow-v15-toe-free-flight-20261110';D=R/'build/crow-v15-toe-contact-isolation-20261110'/body;base=json.loads((P/'smoke-command.json').read_text())
 for substeps in (4,):
  name=body+'-'+str(substeps);C=B/name;C.mkdir(exist_ok=True);cmd=list(base)
  for k,v in [('--physics-substeps',str(substeps)),('--envs','1'),('--steps','1'),('--state-trace',str(C/'probe.tsv')),('--state-trace-environment','0')]:cmd[cmd.index(k)+1]=v
  env={**os.environ,'DYLD_LIBRARY_PATH':str(D),'CROW_CONTACT_TRACE':'1','CROW_TOE_PAIR_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=120);(A/(name+'-probe.stderr')).write_text(p.stderr);source=R/'build/crow-v15-toe-free-flight-20261110/zero.policypack';policy=C/'zero.policypack'
  if p.returncode:
   match=re.search(r'task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',p.stderr)
   if not match:rows.append({'name':name,'stage':'admission','error':p.stderr[-2000:]});print(name,'admission failed',flush=True);continue
   world,task,obs,act=map(int,match.groups());src=read_policy_pack(source,library_path=str(D/'libmetalrobo.dylib'));layers=[PolicyDenseLayerArtifact(weights=x.weights.copy(),bias=x.bias.copy(),activation=x.activation) for x in src.layers];write_policy_pack(policy,policy_id='crow_substep_zero_'+name,revision=1,contract_version=src.contract_version,world_fingerprint=world,task_fingerprint=task,observation_fingerprint=obs,action_fingerprint=act,layers=layers,observation_mean=src.observation_mean,observation_inverse_standard_deviation=src.observation_inverse_standard_deviation,action_bias=src.action_bias,action_scale=src.action_scale,observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=str(D/'libmetalrobo.dylib'))
  else:policy.write_bytes(source.read_bytes())
  import numpy as np
  dst=read_policy_pack(policy,library_path=str(D/'libmetalrobo.dylib'));assert all(np.count_nonzero(x.weights)==0 and np.count_nonzero(x.bias)==0 for x in dst.layers) and np.count_nonzero(dst.action_bias)==0
  for k,v in [('--envs','8'),('--steps','1600'),('--seed','20261120'),('--policy-pack',str(policy)),('--state-trace',str(C/'flight.tsv')),('--state-trace-environment','1')]:cmd[cmd.index(k)+1]=v
  p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=900);(A/(name+'.stdout')).write_text(p.stdout)
  with (A/(name+'.stderr.gz')).open('wb') as f:
   with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
  row={'name':name,'argv':cmd,'exit':p.returncode,'library_sha256':hashlib.sha256((D/'libmetalrobo.dylib').read_bytes()).hexdigest(),'policy_sha256':hashlib.sha256(policy.read_bytes()).hexdigest()}
  if p.returncode:row.update({'stage':'run','error':p.stderr[-2000:]});rows.append(row);print(name,'failed',flush=True);continue
  data=json.loads(p.stdout);first=data['first_episode_navigation'];samples=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_FLIGHT_SAMPLE ')];lane_rows=[]
  for lane in range(8):
   mode=lane%5;rate=[0,.03,-.03,0,0][mode];target=[.85,.85,.85,1.05,.65][mode];ss=sorted((x for x in samples if x['lane']==lane and 1000<=x['step']<=1500 and x['step']<=first['steps_by_environment'][lane]),key=lambda x:x['step']);vals=[]
   for x in ss:
    q=x['q'];v=x['v'];a,b,c,d=q[3:7];yaw=math.atan2(2*(d*c+a*b),1-2*(b*b+c*c));desired=rate*(x['step']*.02-6);heading=abs(math.atan2(math.sin(desired-yaw),math.cos(desired-yaw)));forward=math.cos(yaw)*v[0]+math.sin(yaw)*v[1];lateral=-math.sin(yaw)*v[0]+math.cos(yaw)*v[1];vals.append((heading,abs(lateral),abs(forward-.7),abs(q[2]-target)))
   mx=[max((x[i] for x in vals),default=None) for i in range(4)];toe_max=max((abs(z) for x in samples if x['lane']==lane and x['step']<=first['steps_by_environment'][lane] for z in x['v'][22:30]),default=None) if True else None;passed=(data['failed_environment_steps']==0 and not first['physical_failure_by_environment'][lane] and first['steps_by_environment'][lane]==1600 and [x['step'] for x in ss]==list(range(1000,1501,100)) and all(mx[i]<=lim for i,lim in enumerate([.15,.20,.20,.15])) and (toe_max is not None and toe_max<12))
   lane_rows.append({'lane':lane,'mode':mode,'physical_failure':first['physical_failure_by_environment'][lane],'reason':first['termination_reason_by_environment'][lane],'steps':first['steps_by_environment'][lane],'max_heading_error_rad':mx[0],'max_lateral_m_s':mx[1],'max_speed_error_m_s':mx[2],'max_height_error_m':mx[3],'max_sampled_toe_speed_rad_s':toe_max,'pass':passed})
  row.update({'stage':'complete','failed_environment_steps':data['failed_environment_steps'],'lanes':lane_rows,'passing_lanes':sum(x['pass'] for x in lane_rows)});rows.append(row)
  if (C/'flight.tsv').exists():
   with (A/(name+'.trace.tsv.gz')).open('wb') as f:
    with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write((C/'flight.tsv').read_bytes())
  print(name,'pass',row['passing_lanes'],'solver',row['failed_environment_steps'],flush=True)
(A/'rigid-analysis.json').write_text(json.dumps(rows,indent=2)+'\n')
