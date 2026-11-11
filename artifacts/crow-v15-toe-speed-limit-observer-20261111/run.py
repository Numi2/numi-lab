from pathlib import Path
import gzip,json,os,subprocess,threading
R=Path.cwd();A=R/'artifacts/crow-v15-toe-speed-limit-observer-20261111';B=R/'build/crow-v15-toe-speed-limit-observer-20261111';P=R/'artifacts/crow-v15-two-segment-frozen-flight-second-seed-20261111';L=R/'build/crow-v15-two-segment-frozen-flight-dense-contact-20261111';cmd=json.loads((P/'command.json').read_text());cmd[cmd.index('--steps')+1]='250';cmd[cmd.index('--state-trace')+1]=str(B/'trace.tsv');(A/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');env={**os.environ,'DYLD_LIBRARY_PATH':str(L),'CROW_CONTACT_TRACE':'1','CROW_EARLY_FLIGHT_TRACE':'1','CROW_SUPPORT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'}
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
ref={}
with gzip.open(P/'run.stderr.gz','rt') as f:
 for l in f:
  if l.startswith('CROW_FLIGHT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1]);ref[x['step'],x['lane']]=x
samples={};hits=[]
with gzip.open(A/'stderr.gz','rt') as f:
 for l in f:
  if l.startswith('CROW_FLIGHT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1]);samples[x['step'],x['lane']]=x
  elif l.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1])
   if x['lane']==28:
    v=x['v'][22:38];inds=[i for i,z in enumerate(v) if abs(z)>=11.99]
    if inds:hits.append({'step':x['step'],'toes':[{'index':i,'speed':v[i],'angle':x['q'][23+i]} for i in inds],'contacts':x['contacts'],'height':x['q'][2],'vertical_speed':x['v'][2]})
common=sorted(set(ref)&set(samples));diff=[k for k in common if ref[k]!=samples[k]];out={'exit':rc,'common_samples':len(common),'differences':len(diff),'first_differences':diff[:5],'lane28_speed_limit_hits':hits,'scope':'accepted endpoint and final-substep contact only'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('exit',rc,'common',len(common),'differences',len(diff),'hits',len(hits),flush=True)
with (B/'trace.tsv').open('rb') as f,(A/'trace.tsv.gz').open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
  for b in iter(lambda:f.read(1<<16),b''):z.write(b)
