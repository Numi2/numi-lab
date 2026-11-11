from pathlib import Path
import gzip,hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-walk-to-flight-20261111';B=R/'build/crow-v15-walk-to-flight-20261111';L=R/'build/crow-v15-perturbed-leg-handoff-20261111';base=json.loads((R/'artifacts/crow-v15-delayed-flight-handoff-20261111/hn50-ap18-bindings.json').read_text())['argv'];h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();env={**os.environ,'DYLD_LIBRARY_PATH':str(L),'CROW_SUBSTEP_TRACE':'1','CROW_SUPPORT_TRACE':'1','CROW_CONTACT_TRACE':'1','CROW_CLEARANCE_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'}
for seed in (20261126,91827463,91827491):
 for kind in ('flight','leg'):
  name=str(seed)+'-'+kind;cmd=base.copy();shader=B/(kind+'.metallib')
  for k,v in [('--seed',str(seed)),('--metallib',str(shader)),('--state-trace',str(B/(name+'.tsv'))),('--crow-replay-pack',str(B/(name+'.json')))]:cmd[cmd.index(k)+1]=v
  (A/(name+'-bindings.json')).write_text(json.dumps({'argv':cmd,'shader_sha256':h(shader),'library_sha256':h(L/'libmetalrobo.dylib')},indent=2)+'\n')
  with (A/(name+'.stdout')).open('wb') as so,(A/(name+'.stderr.gz')).open('wb') as raw:
   with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as se:
    p=subprocess.Popen(cmd,cwd=R,env=env,stdout=so,stderr=subprocess.PIPE)
    while True:
     b=p.stderr.read(1<<20)
     if not b:break
     se.write(b)
    rc=p.wait()
  if rc:raise RuntimeError(name+' run exit '+str(rc))
  for src,suffix in [(B/(name+'.json'),'replay.json.gz'),(B/(name+'.tsv'),'trace.tsv.gz')]:
   with src.open('rb') as f,(A/(name+'.'+suffix)).open('wb') as raw:
    with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
     for b in iter(lambda:f.read(1<<20),b''):z.write(b)
  print(name,'completed',flush=True)
