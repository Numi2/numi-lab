from pathlib import Path
import gzip,hashlib,json,math,os,statistics,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-forward-stop-transition-20261110';B=R/'build/crow-v15-forward-stop-transition-20261110';D=R/'build/crow-v15-wing-inertia-prototype-20261110';base=json.loads((R/'artifacts/crow-v15-toe-free-flight-20261110/smoke-command.json').read_text());h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for seed,speed,variant in [('18722447',0.00,'transition')]:
 cmd=list(base)
 for k,v in [('--metallib',str(R/'build/crow-v15-forward-stop-transition-20261110'/variant/'tracking.metallib')),('--envs','80'),('--steps','1600'),('--seed',seed),('--state-trace',str(B/(seed+'.tsv'))),('--state-trace-environment','1'),('--policy-pack',str(R/'build/crow-v15-wing-inertia-prototype-20261110/zero.policypack'))]:cmd[cmd.index(k)+1]=v
 env={**os.environ,'DYLD_LIBRARY_PATH':str(D),'CROW_CONTACT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=900);(A/(seed+'.stdout')).write_text(p.stdout)
 with (A/(seed+'.stderr.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
 if p.returncode:raise RuntimeError(str(p.returncode)+' '+p.stderr[-3000:])
 data=json.loads(p.stdout);first=data['first_episode_navigation'];samples=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_FLIGHT_SAMPLE ')];rows=[]
 for lane in range(80):
  mode=lane%5;rate=[0,.03,-.03,0,0][mode];target=[.85,.85,.85,1.05,.65][mode];ss=sorted((x for x in samples if x['lane']==lane and 1000<=x['step']<=1500 and x['step']<=first['steps_by_environment'][lane]),key=lambda x:x['step']);vals=[]
  for x in ss:
   q=x['q'];v=x['v'];a,b,c,d=q[3:7];yaw=math.atan2(2*(d*c+a*b),1-2*(b*b+c*c));desired=rate*(x['step']*.02-6);heading=abs(math.atan2(math.sin(desired-yaw),math.cos(desired-yaw)));forward=math.cos(yaw)*v[0]+math.sin(yaw)*v[1];lateral=-math.sin(yaw)*v[0]+math.cos(yaw)*v[1];vals.append((heading,abs(lateral),abs(forward-speed),abs(q[2]-target),q[2]))
  maxima=[max((x[i] for x in vals),default=None) for i in range(4)];toe_max=max((abs(z) for x in samples if x['lane']==lane and x['step']<=first['steps_by_environment'][lane] for z in x['v'][22:30]),default=None);passed=(data['failed_environment_steps']==0 and not first['physical_failure_by_environment'][lane] and first['steps_by_environment'][lane]==1600 and [x['step'] for x in ss]==list(range(1000,1501,100)) and all(maxima[i]<=lim for i,lim in enumerate([.15,.20,.08,.15])) and toe_max is not None and toe_max<12.0)
  rows.append({'lane':lane,'perturbation':lane%8,'mode':mode,'physical_failure':first['physical_failure_by_environment'][lane],'reason':first['termination_reason_by_environment'][lane],'steps':first['steps_by_environment'][lane],'sample_steps':[x['step'] for x in ss],'max_heading_error_rad':maxima[0],'max_lateral_m_s':maxima[1],'max_speed_error_m_s':maxima[2],'max_height_error_m':maxima[3],'mean_height_m':statistics.mean(x[4] for x in vals) if len(vals)==6 else None,'max_sampled_toe_speed_rad_s':toe_max,'screen_pass':passed})
 groups=[]
 for mode in range(5):
  rr=[x for x in rows if x['mode']==mode];heights=[x['mean_height_m'] for x in rr if x['mean_height_m'] is not None];groups.append({'mode':mode,'physical_failures':sum(x['physical_failure'] for x in rr),'passing':sum(x['screen_pass'] for x in rr),'median_mean_height_m':statistics.median(heights) if len(heights)==16 else None,'mode_pass':all(x['screen_pass'] for x in rr)})
 report={'commanded_speed_m_s':speed,'argv':cmd,'shader_sha256':h(R/'build/crow-v15-forward-stop-transition-20261110'/variant/'tracking.metallib'),'zero_policy_sha256':h(R/'build/crow-v15-wing-inertia-prototype-20261110/zero.policypack'),'library_sha256':h(D/'libmetalrobo.dylib'),'failed_environment_steps':data['failed_environment_steps'],'first_episode':first,'lanes':rows,'groups':groups,'climb_minus_descent_m':groups[3]['median_mean_height_m']-groups[4]['median_mean_height_m'] if all(groups[x]['median_mean_height_m'] is not None for x in (3,4)) else None,'scope':'estimated modified body with corrected aerodynamic prototype; fixed-head, zero actor diagnostic, not full navigation or landing'};(A/(seed+'-analysis.json')).write_text(json.dumps(report,indent=2)+'\n')
 with (A/(seed+'.trace.tsv.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write((B/(seed+'.tsv')).read_bytes())
 print(seed,json.dumps(groups),flush=True)
