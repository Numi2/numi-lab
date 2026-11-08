from pathlib import Path
import hashlib,json,os,subprocess,time,sys
R=Path.cwd();A=R/'artifacts/crow-rl-grasp-resume-20261108';B=R/'build/crow-grasp-resume-build-01';O=B/'observations'
plan=json.loads((A/'plan.json').read_text())
for p,h in plan['inputs'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p
env={**os.environ,'DYLD_LIBRARY_PATH':str(B/'runtime'),'DYLD_PRINT_LIBRARIES':'1','CROW_PROBE_METALLIB':str(B/'runtime/metalrobo/MetalRobo.metallib')}
execution=[]
for name,arm in plan['arms'].items():
 local={**env,'METALROBO_QUALITY_FAILURE_CAPTURE':str(O/(name+'-capture.bin'))}
 cmd=[B/'crow-fixture',O/name,str(arm),'1','32','2','api-default'];start=time.monotonic()
 with (O/(name+'.stdout')).open('xb') as out,(O/(name+'.stderr')).open('xb') as err:
  try:code=subprocess.run([str(x) for x in cmd],env=local,stdout=out,stderr=err,timeout=600).returncode
  except subprocess.TimeoutExpired:code=None
 execution.append(dict(name=name,arm=arm,returncode=code,seconds=time.monotonic()-start))
 (A/'execution.json').write_text(json.dumps(execution,indent=2,sort_keys=True)+'\n');print(json.dumps(execution[-1]),flush=True)
 assert any(line.rstrip().endswith(str(B/'runtime/libmetalrobo.dylib')) for line in (O/(name+'.stderr')).read_text().splitlines())
 assert sum(p.stat().st_size for p in B.rglob('*') if p.is_file())<plan['limits']['raw_bytes']
subprocess.run([sys.executable,str(A/'analyze.py')],check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
