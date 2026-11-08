from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path.cwd();A=R/'artifacts/crow-rl-quality-tolerances-20261108';B=R/'build/crow-quality-tolerances-build-01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
plan=json.loads((A/'plan.json').read_text());runtime=json.loads((A/'runtime.json').read_text())
for name,h in plan['sources'].items():assert sha(A/name)==h
for name,f in runtime['files'].items():assert sha(R/name)==f['sha256']
(B/'observations').mkdir();execution=[];started=time.monotonic()
env={**os.environ,'DYLD_LIBRARY_PATH':str(B/'runtime'),'DYLD_PRINT_LIBRARIES':'1','CROW_PROBE_METALLIB':str(B/'runtime/metalrobo/MetalRobo.metallib')}
jobs=[('calibration',[str(B/'crow-fixture'),'--calibrate'],None)]
jobs += [(arm+f'-{repeat}',[str(B/'crow-fixture'),str(B/'observations'/(arm+f'-{repeat}')),'0','1','32','2',arm],arm) for repeat,order in enumerate([plan['arms'],list(reversed(plan['arms']))]) for arm in order]
assert len(jobs)==plan['limits']['native_commands']
for name,cmd,arm in jobs:
 assert time.monotonic()-started<plan['limits']['total_seconds']
 local={**env}
 if arm:local['METALROBO_QUALITY_FAILURE_CAPTURE']=str(B/'observations'/(name+'-capture.bin'))
 with (B/'observations'/(name+'.stdout')).open('xb') as stdout,(B/'observations'/(name+'.stderr')).open('xb') as stderr:
  start=time.monotonic()
  try:code=subprocess.run(cmd,stdout=stdout,stderr=stderr,env=local,timeout=plan['limits']['per_command_seconds']).returncode
  except subprocess.TimeoutExpired:code=None
 record=dict(name=name,argv=cmd,returncode=code,seconds=time.monotonic()-start);execution.append(record);write(A/'execution.json',execution);print(json.dumps(record),flush=True)
 assert code in ([0,1] if arm else [0])
 stderr=(B/'observations'/(name+'.stderr')).read_text()
 assert any(line.rstrip().endswith(str(B/'runtime/libmetalrobo.dylib')) for line in stderr.splitlines()),'Unexpected dynamic library'
 if arm:assert (B/'observations'/name/'failure.json').exists() or (B/'observations'/name/'terminal.json').exists() or (B/'observations'/name/'summary.json').exists()
 assert sum(p.stat().st_size for p in B.rglob('*') if p.is_file())<plan['limits']['raw_bytes']
sys.path.insert(0,str(A));import analyze;analyze.analyze()
