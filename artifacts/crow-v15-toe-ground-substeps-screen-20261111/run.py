from pathlib import Path
import gzip,json,os,subprocess,threading
R=Path.cwd();A=R/'artifacts/crow-v15-toe-ground-substeps-screen-20261111';B=R/'build/crow-v15-toe-ground-substeps-screen-20261111';P=R/'artifacts/crow-v15-two-segment-frozen-flight-second-seed-20261111';L=R/'build/crow-v15-two-segment-frozen-flight-dense-contact-20261111';cmd=json.loads((P/'command.json').read_text());cmd[cmd.index('--steps')+1]='250';cmd[cmd.index('--physics-substeps')+1]='32';cmd[cmd.index('--state-trace')+1]=str(B/'trace.tsv');(A/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');env={**os.environ,'DYLD_LIBRARY_PATH':str(L),'CROW_CONTACT_TRACE':'1','CROW_EARLY_FLIGHT_TRACE':'1','CROW_SUPPORT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'}
p=subprocess.Popen(cmd,cwd=R,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
def pump(src,path,zip):
 with path.open('wb') as f:
  if zip:
   with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:
    for b in iter(lambda:src.read(1<<16),b''):z.write(b)
  else:
   for b in iter(lambda:src.read(1<<16),b''):f.write(b)
ts=[threading.Thread(target=pump,args=(p.stdout,A/'stdout',False)),threading.Thread(target=pump,args=(p.stderr,A/'stderr.gz',True))]
for t in ts:t.start()
rc=p.wait()
for t in ts:t.join()
from collections import Counter
hits=Counter();steps=Counter();toes=Counter();seen=set();minsep=None;errors=[]
with gzip.open(A/'stderr.gz','rt') as f:
 for l in f:
  if 'failed' in l.lower():errors.append(l.strip())
  if l.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1]);key=x['step'],x['lane'];seen.add(key)
   for c in x['contacts']:
    z=c['separation'];minsep=z if minsep is None else min(minsep,z)
   inds=[i for i,z in enumerate(x['v'][22:38]) if abs(z)>=11.99]
   if inds:
    hits[x['lane']]+=1;steps[x['step']]+=1
    for i in inds:toes[i]+=1
out={'exit':rc,'observed_lane_steps':len(seen),'lanes_with_hits':len(hits),'lane_interval_hits':sum(hits.values()),'hits_by_lane':dict(hits),'hits_by_step':dict(steps),'hits_by_toe':dict(toes),'minimum_final_substep_contact_separation_m':minsep,'errors':errors[-5:],'scope':'development ground-to-flight accepted endpoints and last-substep contacts only'}
try:
 data=json.loads((A/'stdout').read_text());out['failed_environment_steps']=data['failed_environment_steps'];out['physical_failures']=sum(data['first_episode_navigation']['physical_failure_by_environment']);out['minimum_episode_steps']=min(data['first_episode_navigation']['steps_by_environment'])
except Exception as e:out['output_error']=str(e)
out['pass']=rc==0 and len(seen)>=80*250 and len(hits)==0 and out.get('failed_environment_steps')==0 and out.get('physical_failures')==0 and out.get('minimum_episode_steps')==250
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
with (B/'trace.tsv').open('rb') as f,(A/'trace.tsv.gz').open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
  for b in iter(lambda:f.read(1<<16),b''):z.write(b)
