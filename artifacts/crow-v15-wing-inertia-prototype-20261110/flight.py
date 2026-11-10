from pathlib import Path
import gzip,json,math,os,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-wing-inertia-prototype-20261110';B=R/'build/crow-v15-wing-inertia-prototype-20261110';cmd=json.loads((R/'artifacts/crow-v15-toe-free-flight-20261110/smoke-command.json').read_text())
for k,v in [('--envs','8'),('--steps','1600'),('--seed','18647329'),('--state-trace',str(B/'flight.tsv')),('--state-trace-environment','0'),('--policy-pack',str(B/'zero.policypack'))]:cmd[cmd.index(k)+1]=v
env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'CROW_CONTACT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=900);(A/'flight.stdout').write_text(p.stdout)
with (A/'flight.stderr.gz').open('wb') as out:
 with gzip.GzipFile(fileobj=out,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
if p.returncode:raise RuntimeError('run failed '+p.stderr[-2000:])
d=json.loads(p.stdout);first=d['first_episode_navigation'];samples=[json.loads(l.split(' ',1)[1]) for l in p.stderr.splitlines() if l.startswith('CROW_FLIGHT_SAMPLE ')];rows=[]
for lane in range(8):
 mode=lane%5;rate=[0,.03,-.03,0,0][mode];target=[.85,.85,.85,1.05,.65][mode];ss=sorted((x for x in samples if x['lane']==lane and 1000<=x['step']<=1500 and x['step']<=first['steps_by_environment'][lane]),key=lambda x:x['step']);vals=[]
 for x in ss:
  q=x['q'];v=x['v'];a,b,c,e=q[3:7];yaw=math.atan2(2*(e*c+a*b),1-2*(b*b+c*c));desired=rate*(x['step']*.02-6);heading=abs(math.atan2(math.sin(desired-yaw),math.cos(desired-yaw)));forward=math.cos(yaw)*v[0]+math.sin(yaw)*v[1];lateral=-math.sin(yaw)*v[0]+math.cos(yaw)*v[1];vals.append((heading,abs(lateral),abs(forward-.7),abs(q[2]-target)))
 mx=[max((x[i] for x in vals),default=None) for i in range(4)];ok=d['failed_environment_steps']==0 and not first['physical_failure_by_environment'][lane] and first['steps_by_environment'][lane]==1600 and [x['step'] for x in ss]==list(range(1000,1501,100)) and all(mx[i]<=lim for i,lim in enumerate([.15,.20,.20,.15]));rows.append({'lane':lane,'mode':mode,'first_episode_steps':first['steps_by_environment'][lane],'physical_failure':first['physical_failure_by_environment'][lane],'max_errors':mx,'screen_pass':ok})
report={'argv':cmd,'host_sha256':hashlib.sha256((B/'libmetalrobo.dylib').read_bytes()).hexdigest(),'policy_sha256':hashlib.sha256((B/'zero.policypack').read_bytes()).hexdigest(),'failed_environment_steps':d['failed_environment_steps'],'first_episode':first,'lanes':rows,'pass_count':sum(x['screen_pass'] for x in rows),'scope':'Exploratory estimated modified body, fixed-head zero actor, obstacles disabled; not a qualification or biological validation.'};(A/'flight-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
if (B/'flight.tsv').exists():
 with (B/'flight.tsv').open('rb') as inp,(A/'flight.tsv.gz').open('wb') as out:
  with gzip.GzipFile(fileobj=out,mode='wb',mtime=0) as z:
   import shutil;shutil.copyfileobj(inp,z)
print('passing',report['pass_count'],'of 8; solver failed steps',report['failed_environment_steps'],'physical failures',sum(first['physical_failure_by_environment']),flush=True)
