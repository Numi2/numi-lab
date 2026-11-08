from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path.cwd();A=R/'artifacts/crow-rl-split-cones-20261108';B=R/'build/crow-split-cones-build-01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
plan=json.loads((A/'plan.json').read_text())
for name,v in plan['sources'].items():assert sha(A/name)==v['sha256']
for name,v in plan['inputs'].items():assert sha(R/name)==v['sha256']
commands=[
 ['xcrun','-sdk','macosx','metal','-std=metal3.1','-O3','-fno-fast-math','-Iinclude','-I'+str(A),'-c',str(A/'solver.metal'),'-o',str(B/'solver.air')],
 ['xcrun','-sdk','macosx','metallib',str(B/'solver.air'),'-o',str(B/'solver.metallib')],
 ['clang++','-std=c++20','-O2','-fobjc-arc',str(A/'probe.mm'),'-framework','Foundation','-framework','Metal','-o',str(B/'probe')]]
execution=[]
for i,cmd in enumerate(commands):
 start=time.monotonic()
 with (B/f'build-{i}.stdout').open('xb') as stdout,(B/f'build-{i}.stderr').open('xb') as stderr:
  try:code=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=180).returncode
  except subprocess.TimeoutExpired:code=None
 execution.append(dict(argv=cmd,code=code,seconds=time.monotonic()-start));write(A/'build-execution.json',execution)
 print(json.dumps(execution[-1]),flush=True);assert code==0
files={str((B/n).relative_to(R)):dict(sha256=sha(B/n),bytes=(B/n).stat().st_size) for n in ['probe','solver.metallib']}
toolchain={k:subprocess.check_output(cmd,text=True) for k,cmd in dict(clang=['clang++','--version'],metal=['xcrun','-sdk','macosx','metal','--version'],os=['sw_vers']).items()}
write(A/'runtime.json',dict(schema='crow-split-cones-runtime-v1',files=files,toolchain=toolchain,
 plan_sha256=sha(A/'plan.json'),build_source_sha256=sha(A/'build.py'),production_changed=False))
