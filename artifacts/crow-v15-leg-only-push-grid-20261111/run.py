from pathlib import Path
import gzip,hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-leg-only-push-grid-20261111';B=R/'build/crow-v15-leg-only-push-grid-20261111';P=R/'build/crow-v15-bounded-launch-commands-20261111';base=json.loads((R/'artifacts/crow-v15-propulsive-launch-controls-20261111/wing-suppressed-bindings.json').read_text())['argv'];h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();env={**os.environ,'DYLD_LIBRARY_PATH':str(P),'CROW_SUBSTEP_TRACE':'1','CROW_SUPPORT_TRACE':'1','CROW_CONTACT_TRACE':'1','CROW_CLEARANCE_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'}
for hip in ('n50','n30','n10','p10'):
 for ankle in ('n30','n14','p02','p18'):
  name='h'+hip+'-a'+ankle;cmd=base.copy();shader=B/(name+'.metallib')
  for k,v in [('--metallib',str(shader)),('--state-trace',str(B/(name+'.tsv'))),('--crow-replay-pack',str(B/(name+'.json')))]:cmd[cmd.index(k)+1]=v
  (A/(name+'-bindings.json')).write_text(json.dumps({'argv':cmd,'shader_sha256':h(shader),'library_sha256':h(P/'libmetalrobo.dylib')},indent=2)+'\n')
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
