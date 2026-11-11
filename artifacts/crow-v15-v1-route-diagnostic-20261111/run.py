from pathlib import Path
import gzip,hashlib,json,os,subprocess,shutil
R=Path.cwd();A=R/'artifacts/crow-v15-v1-route-diagnostic-20261111';B=R/'build/crow-v15-v1-route-diagnostic-20261111';L=R/'build/crow-v15-perturbed-leg-handoff-20261111';src=R/'artifacts/crow-v15-second-clearance-eval-20261110/second-j-original-bindings.json';cmd=json.loads(src.read_text())['argv'];h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
shader=R/'build/crow-v15-walk-to-flight-20261111/flight.metallib'
for k,v in [('--metallib',str(shader)),('--physics-substeps','16'),('--steps','3000'),('--state-trace',str(B/'route.tsv')),('--crow-replay-pack',str(B/'route.json'))]:cmd[cmd.index(k)+1]=v
env={**os.environ,'DYLD_LIBRARY_PATH':str(L),'CROW_GAIT_TRACE':'1','CROW_SUBSTEP_TRACE':'1','CROW_SUBSTEP_FOOT_SENSOR':'1','CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','CROW_CLEARANCE_TRACE':'1','CROW_TASK_STATE_DIAGNOSTIC':'1','PYTHONDONTWRITEBYTECODE':'1'}
(A/'bindings.json').write_text(json.dumps({'argv':cmd,'runtime_sha256':h(L/'libmetalrobo.dylib'),'shader_sha256':h(shader),'actor_sha256':h(Path(cmd[cmd.index('--base-policy-pack')+1])),'teacher_enabled':False,'scope':'development J diagnostic only'},indent=2)+'\n')
with (A/'route.stdout').open('wb') as out,(A/'route.stderr.gz').open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as gz:
  p=subprocess.Popen(cmd,cwd=R,env=env,stdout=out,stderr=subprocess.PIPE);shutil.copyfileobj(p.stderr,gz);rc=p.wait()
for ext in ('tsv','json'):
 src=B/('route.'+ext)
 if src.exists():
  with src.open('rb') as inp,(A/('route.'+ext+'.gz')).open('wb') as raw:
   with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as gz:shutil.copyfileobj(inp,gz)
print('exit',rc,flush=True)
raise SystemExit(rc)
