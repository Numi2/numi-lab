from pathlib import Path
import hashlib,json,subprocess,time
R=Path.cwd();A=R/'artifacts/crow-rl-line-search-refinement-20261108';B=R/'build/crow-line-search-refinement-build-01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
commands=[['cmake','--build','build','--config','Release','-j6'],['clang++','-std=c++20','-O2','-fobjc-arc','-Iinclude',str(A/'probe.mm'),'-framework','Foundation','-framework','Metal','-o',str(B/'probe')]]
execution=[]
for i,cmd in enumerate(commands):
 start=time.monotonic()
 with (B/f'build-{i}.stdout').open('xb') as stdout,(B/f'build-{i}.stderr').open('xb') as stderr:
  try:code=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=600).returncode
  except subprocess.TimeoutExpired:code=None
 execution.append(dict(argv=cmd,code=code,seconds=time.monotonic()-start));write(A/'build-execution.json',execution);print(json.dumps(execution[-1]),flush=True);assert code==0
files={str(p.relative_to(R)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in [B/'probe',B/'baseline.metallib',R/'build/lib/libmetalrobo.dylib',R/'build/shaders/MetalRobo.metallib']}
write(A/'runtime.json',dict(files=files,clang=subprocess.check_output(['clang++','--version'],text=True),production_changed=True))
