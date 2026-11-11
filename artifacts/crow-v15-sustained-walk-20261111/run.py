from pathlib import Path
import gzip,hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-sustained-walk-20261111';B=R/'build/crow-v15-sustained-walk-20261111';cmd=json.loads((R/'artifacts/crow-v15-walk-16substep-screen-20261111/zero16-bindings.json').read_text())['argv'];h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'CROW_SUBSTEP_TRACE':'1','CROW_SUPPORT_TRACE':'1','CROW_CONTACT_TRACE':'1','CROW_CLEARANCE_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'}
for k,v in [('--steps','500'),('--metallib',str(B/'walk.metallib')),('--state-trace',str(B/'walk.tsv')),('--crow-replay-pack',str(B/'walk.json'))]:cmd[cmd.index(k)+1]=v
(A/'bindings.json').write_text(json.dumps({'argv':cmd,'shader_sha256':h(B/'walk.metallib'),'library_sha256':h(B/'libmetalrobo.dylib')},indent=2)+'\n')
with (A/'walk.stdout').open('wb') as so,(A/'walk.stderr.gz').open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as se:
  p=subprocess.Popen(cmd,cwd=R,env=env,stdout=so,stderr=subprocess.PIPE)
  while True:
   b=p.stderr.read(1<<20)
   if not b:break
   se.write(b)
  rc=p.wait()
if rc:raise RuntimeError('walk run exit '+str(rc))
for src,suffix in [(B/'walk.json','walk.replay.json.gz'),(B/'walk.tsv','walk.trace.tsv.gz')]:
 with src.open('rb') as f,(A/suffix).open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
   for b in iter(lambda:f.read(1<<20),b''):z.write(b)
