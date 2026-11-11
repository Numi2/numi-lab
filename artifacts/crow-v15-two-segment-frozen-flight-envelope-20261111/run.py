from pathlib import Path
import gzip,hashlib,json,math,os,statistics,subprocess,threading
R=Path.cwd();A=R/'artifacts/crow-v15-two-segment-frozen-flight-envelope-20261111';B=R/'build/crow-v15-two-segment-frozen-flight-envelope-20261111';L=R/'build/crow-v15-two-segment-frozen-flight-dense-contact-20261111';C=R/'build/crow-v15-two-segment-frozen-flight-contact-map-20261111';cmd=json.loads((R/'artifacts/crow-v15-two-segment-frozen-flight-dense-contact-20261111/200-command.json').read_text());cmd[cmd.index('--steps')+1]='1600';cmd[cmd.index('--state-trace')+1]=str(B/'trace.tsv');(A/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/'inputs.json').write_text(json.dumps({'host_sha256':h(L/'libmetalrobo.dylib'),'metallib_sha256':h(C/'tracking.metallib'),'policy_sha256':h(C/'two-segment-zero.policypack'),'frozen_controller_sha256':h(R/'artifacts/crow-v15-flight-feedback-freeze-20261110/CrowFlightControllerResearchV2.metal')},indent=2)+'\n')
env={**os.environ,'DYLD_LIBRARY_PATH':str(L),'CROW_CONTACT_TRACE':'1','CROW_EARLY_FLIGHT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.Popen(cmd,cwd=R,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
def pump(src,path,compress):
 with path.open('wb') as raw:
  if compress:
   with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
    for b in iter(lambda:src.read(1<<16),b''):z.write(b)
  else:
   for b in iter(lambda:src.read(1<<16),b''):raw.write(b)
ts=[threading.Thread(target=pump,args=(p.stdout,A/'run.stdout',False)),threading.Thread(target=pump,args=(p.stderr,A/'run.stderr.gz',True))]
for t in ts:t.start()
rc=p.wait()
for t in ts:t.join()
samples=[];errors=[]
with gzip.open(A/'run.stderr.gz','rt') as f:
 for l in f:
  if l.startswith('CROW_FLIGHT_SAMPLE '):samples.append(json.loads(l.split(' ',1)[1]))
  elif 'failed' in l.lower():errors.append(l.strip())
report={'process_exit_code':rc,'errors':errors[-5:],'sample_count':len(samples),'scope':'development obstacle-free flight screen only'}
if rc==0:
 data=json.loads((A/'run.stdout').read_text());first=data['first_episode_navigation'];rows=[]
 for lane in range(80):
  mode=lane%5;rate=[0.,.03,-.03,0.,0.][mode];target=[.85,.85,.85,1.05,.65][mode];ss=sorted((x for x in samples if x['lane']==lane and 1000<=x['step']<=1500 and x['step']<=first['steps_by_environment'][lane]),key=lambda x:x['step']);vals=[]
  for x in ss:
   q=x['q'];v=x['v'];a,b,c,d=q[3:7];yaw=math.atan2(2*(d*c+a*b),1-2*(b*b+c*c));desired=rate*(x['step']*.02-6);heading=abs(math.atan2(math.sin(desired-yaw),math.cos(desired-yaw)));forward=math.cos(yaw)*v[0]+math.sin(yaw)*v[1];lateral=-math.sin(yaw)*v[0]+math.cos(yaw)*v[1];vals.append((heading,abs(lateral),abs(forward-.7),abs(q[2]-target),q[2]))
  maxima=[max((x[i] for x in vals),default=None) for i in range(4)];toe=max((abs(z) for x in samples if x['lane']==lane and x['step']<=first['steps_by_environment'][lane] for z in x['v'][22:38]),default=None);ok=(data['failed_environment_steps']==0 and not first['physical_failure_by_environment'][lane] and first['steps_by_environment'][lane]==1600 and [x['step'] for x in ss]==list(range(1000,1501,100)) and all(maxima[i] is not None and maxima[i]<=limit for i,limit in enumerate([.15,.20,.20,.15])) and toe is not None and toe<12.)
  rows.append({'lane':lane,'mode':mode,'physical_failure':first['physical_failure_by_environment'][lane],'termination_reason':first['termination_reason_by_environment'][lane],'steps':first['steps_by_environment'][lane],'sample_steps':[x['step'] for x in ss],'max_heading_error_rad':maxima[0],'max_lateral_m_s':maxima[1],'max_speed_error_m_s':maxima[2],'max_height_error_m':maxima[3],'mean_height_m':statistics.mean(x[4] for x in vals) if len(vals)==6 else None,'max_sampled_toe_speed_rad_s':toe,'pass':ok})
 groups=[]
 for mode in range(5):
  group=[x for x in rows if x['mode']==mode];heights=[x['mean_height_m'] for x in group if x['mean_height_m'] is not None];groups.append({'mode':mode,'passing':sum(x['pass'] for x in group),'physical_failures':sum(x['physical_failure'] for x in group),'median_height_m':statistics.median(heights) if len(heights)==16 else None})
 sep=groups[3]['median_height_m']-groups[4]['median_height_m'] if all(groups[i]['median_height_m'] is not None for i in (3,4)) else None;report.update({'failed_environment_steps':data['failed_environment_steps'],'lanes':rows,'groups':groups,'climb_minus_descent_m':sep,'envelope_pass':all(x['pass'] for x in rows) and sep is not None and sep>=.25})
 ref={}
 with gzip.open(R/'artifacts/crow-v15-two-segment-frozen-flight-dense-contact-20261111/200.stderr.gz','rt') as f:
  for l in f:
   if l.startswith('CROW_FLIGHT_SAMPLE '):x=json.loads(l.split(' ',1)[1]);ref[x['step'],x['lane']]=x
 current={(x['step'],x['lane']):x for x in samples};common=[k for k in ref if k in current];report['early_sample_common_count']=len(common);report['early_sample_difference_count']=sum(ref[k]!=current[k] for k in common)
if (B/'trace.tsv').exists():
 with (B/'trace.tsv').open('rb') as f,(A/'trace.tsv.gz').open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
   for b in iter(lambda:f.read(1<<16),b''):z.write(b)
(A/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print('exit',rc,'samples',len(samples),'groups',report.get('groups'),'pass',report.get('envelope_pass'),'errors',errors[-1:],flush=True)
