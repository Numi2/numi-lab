from pathlib import Path
import gzip,hashlib,json,math,os,statistics,subprocess,threading
R=Path.cwd();A=R/'artifacts/crow-v15-contact-gated-flight-toes-20261111';B=R/'build/crow-v15-contact-gated-flight-toes-20261111';cmd=json.loads((A/'rebound-command.json').read_text());cmd[cmd.index('--steps')+1]='1600';cmd[cmd.index('--metallib')+1]=str(B/'curl-flight.metallib');cmd[cmd.index('--crow-replay-pack')+1]=str(B/'flight-replay.json');cmd[cmd.index('--state-trace')+1]=str(B/'flight-trace.tsv');(A/'flight-command.json').write_text(json.dumps(cmd,indent=2)+'\n');h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/'flight-inputs.json').write_text(json.dumps({'host_sha256':h(R/'build/crow-v15-hip-roll-obstacle-free-control-20261111/libmetalrobo.dylib'),'shader_sha256':h(Path(cmd[cmd.index('--metallib')+1])),'policy_sha256':h(Path(cmd[cmd.index('--policy-pack')+1])),'frozen_controller_sha256':h(R/'artifacts/crow-v15-flight-feedback-freeze-20261110/CrowFlightControllerResearchV2.metal')},indent=2)+'\n')
env={**os.environ,'DYLD_LIBRARY_PATH':str(R/'build/crow-v15-hip-roll-obstacle-free-control-20261111'),'CROW_EARLY_FLIGHT_TRACE':'1','CROW_CONTACT_TRACE':'1','CROW_SUBSTEP_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.Popen(cmd,cwd=R,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
def pump(src,path,compress):
 with path.open('wb') as raw:
  if compress:
   with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
    for b in iter(lambda:src.read(1<<16),b''):z.write(b)
  else:
   for b in iter(lambda:src.read(1<<16),b''):raw.write(b)
ts=[threading.Thread(target=pump,args=(p.stdout,A/'flight.stdout',False)),threading.Thread(target=pump,args=(p.stderr,A/'flight.stderr.gz',True))]
for t in ts:t.start()
rc=p.wait()
for t in ts:t.join()
samples=[];errors=[];toe=[0.]*5;cap=[0]*5;after500=[0]*5;coverage=[0]*5
with gzip.open(A/'flight.stderr.gz','rt') as f:
 for l in f:
  if l.startswith('CROW_FLIGHT_SAMPLE '):samples.append(json.loads(l.split(' ',1)[1]))
  elif l.startswith('CROW_TOE_V_SUBSTEP '):
   v=l.split();e=int(v[3]);x=max(map(abs,map(float,v[4:])));toe[e]=max(toe[e],x);cap[e]+=x>=11.99;after500[e]+=(x>=11.99 and int(v[1])>=500);coverage[e]+=1
  elif 'failed' in l.lower() or l.startswith('CROW_SUBSTEP_ERROR'):errors.append(l.strip())
out={'exit_code':rc,'errors':errors[-5:],'sample_count':len(samples),'scope':'five-lane hierarchical journey task with bounded diagnostic commands and course collisions disabled; not autonomous navigation, perturbed recovery, landing or biological validation','toe_max_substep_endpoint_rad_s':toe,'toe_cap_substeps':cap,'toe_cap_substeps_after_observer_step_500':after500,'toe_substep_record_counts':coverage}
if rc==0:
 try:
  d=json.loads((A/'flight.stdout').read_text());first=d['first_episode_navigation'];out['failed_environment_steps']=d['failed_environment_steps'];out['lanes']=[]
  for e in range(5):
   mode=e%5;rate=[0.,.03,-.03,0.,0.][mode];target=[.85,.85,.85,1.05,.65][mode];ss=sorted((x for x in samples if x['lane']==e and 1000<=x['step']<=1500 and x['step']<=first['steps_by_environment'][e]),key=lambda x:x['step']);vals=[]
   for x in ss:
    q=x['q'];v=x['v'];a,b,c,d=q[3:7];yaw=math.atan2(2*(d*c+a*b),1-2*(b*b+c*c));desired=rate*(x['step']*.02-7.2);head=abs(math.atan2(math.sin(desired-yaw),math.cos(desired-yaw)));forward=math.cos(yaw)*v[0]+math.sin(yaw)*v[1];lateral=-math.sin(yaw)*v[0]+math.cos(yaw)*v[1];vals.append((head,abs(lateral),abs(forward-.35),abs(q[2]-target),q[2]))
   maxima=[max((x[i] for x in vals),default=None) for i in range(4)];passed=(out['failed_environment_steps']==0 and not first['physical_failure_by_environment'][e] and first['steps_by_environment'][e]==1600 and [x['step'] for x in ss]==list(range(1000,1501,100)) and all(maxima[i] is not None and maxima[i]<=limit for i,limit in enumerate([.15,.20,.20,.15])) and cap[e]==0)
   out['lanes'].append({'lane':e,'mode':mode,'physical_failure':first['physical_failure_by_environment'][e],'termination_reason':first['termination_reason_by_environment'][e],'steps':first['steps_by_environment'][e],'sample_steps':[x['step'] for x in ss],'max_heading_error_rad':maxima[0],'max_lateral_m_s':maxima[1],'max_speed_error_m_s':maxima[2],'max_height_error_m':maxima[3],'mean_height_m':statistics.mean(x[4] for x in vals) if len(vals)==6 else None,'pass_screen':passed})
 except (ValueError,KeyError,IndexError) as e:out['analysis_error']=str(e)
if (B/'flight-trace.tsv').exists():
 with (B/'flight-trace.tsv').open('rb') as f,(A/'flight-trace.tsv.gz').open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
   for b in iter(lambda:f.read(1<<16),b''):z.write(b)
(A/'flight-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('exit',rc,'lanes',out.get('lanes'),'errors',errors[-1:],flush=True)
