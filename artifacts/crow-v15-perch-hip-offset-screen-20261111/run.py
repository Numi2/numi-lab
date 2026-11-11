from pathlib import Path
import gzip,json,os,subprocess,threading
R=Path.cwd();A=R/'artifacts/crow-v15-perch-hip-offset-screen-20261111';B=R/'build/crow-v15-perch-hip-offset-screen-20261111';O=A/'trial';O.mkdir(exist_ok=True)
cmd=json.loads((R/'artifacts/crow-v15-landing-approach-dev-20261111/binding/command.json').read_text());cmd[cmd.index('--metallib')+1]=str(B/'tracking.metallib');cmd[cmd.index('--steps')+1]='5000';cmd[cmd.index('--state-trace')+1]=str(B/'trace.tsv');cmd[cmd.index('--crow-replay-pack')+1]=str(B/'replay.json');(O/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
env={**os.environ,'DYLD_LIBRARY_PATH':str(R/'build/crow-v15-landing-approach-dev-20261111/trace'),'CROW_CLEARANCE_TRACE':'1','CROW_SUBSTEP_TRACE':'1','CROW_SUPPORT_TRACE':'1','CROW_CONTACT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'}
p=subprocess.Popen(cmd,cwd=R,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
def stdout():
 with (O/'stdout').open('wb') as dst:
  for c in iter(lambda:p.stdout.read(1<<16),b''):dst.write(c)
def stderr():
 z=None;raw=None
 try:
  for i,line in enumerate(p.stderr):
   if i%300000==0:
    if z:z.close();raw.close()
    raw=(O/('stderr-%03d.gz'%(i//300000))).open('wb');z=gzip.GzipFile(fileobj=raw,mode='wb',mtime=0)
   z.write(line)
 finally:
  if z:z.close();raw.close()
ts=[threading.Thread(target=stdout),threading.Thread(target=stderr)]
for t in ts:t.start()
rc=p.wait()
for t in ts:t.join()
for src,dst in [(B/'trace.tsv',O/'trace.tsv.gz'),(B/'replay.json',O/'replay.json.gz')]:
 if src.exists():
  with src.open('rb') as f,dst.open('wb') as raw:
   with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
    for c in iter(lambda:f.read(1<<16),b''):z.write(c)
(O/'run-result.json').write_text(json.dumps({'exit':rc},indent=2)+'\n');print('exit',rc,flush=True)
