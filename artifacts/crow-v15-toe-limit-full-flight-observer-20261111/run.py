from pathlib import Path
import collections,gzip,json,os,subprocess,threading
R=Path.cwd();A=R/'artifacts/crow-v15-toe-limit-full-flight-observer-20261111';B=R/'build/crow-v15-toe-limit-full-flight-observer-20261111';P=R/'artifacts/crow-v15-two-segment-frozen-flight-second-seed-20261111';cmd=json.loads((P/'command.json').read_text());cmd[cmd.index('--state-trace')+1]=str(B/'trace.tsv');(A/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'CROW_CONTACT_TRACE':'1','CROW_EARLY_FLIGHT_TRACE':'1','CROW_TOE_LIMIT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.Popen(cmd,cwd=R,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
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
ref={};current={};hits=[];errors=[]
with gzip.open(P/'run.stderr.gz','rt') as f:
 for l in f:
  if l.startswith('CROW_FLIGHT_SAMPLE '):x=json.loads(l.split(' ',1)[1]);ref[x['step'],x['lane']]=x
with gzip.open(A/'stderr.gz','rt') as f:
 for l in f:
  if 'failed' in l.lower():errors.append(l.strip())
  if l.startswith('CROW_FLIGHT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1]);current[x['step'],x['lane']]=x;inds=[i for i,z in enumerate(x['v'][22:38]) if abs(z)>=11.99]
   if inds:hits.append({'step':x['step'],'lane':x['lane'],'height_m':x['q'][2],'vertical_speed_m_s':x['v'][2],'toe_indices':inds})
common=set(ref)&set(current);diff=[k for k in common if ref[k]!=current[k]];late=[x for x in hits if x['step']>250];out={'exit':rc,'reference_samples':len(ref),'common_samples':len(common),'different_common_samples':len(diff),'first_differences':sorted(diff)[:5],'hit_lane_count':len({x['lane'] for x in hits}),'hit_lane_intervals':len(hits),'late_hit_lane_count':len({x['lane'] for x in late}),'late_hit_lane_intervals':len(late),'hits':hits,'errors':errors[-3:],'scope':'accepted control-interval endpoints, not every physics substep'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('exit',rc,'reference common',len(common),'differences',len(diff),'hit lanes',out['hit_lane_count'],'late',len(late),flush=True)
if (B/'trace.tsv').exists():
 with (B/'trace.tsv').open('rb') as f,(A/'trace.tsv.gz').open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
   for b in iter(lambda:f.read(1<<16),b''):z.write(b)
