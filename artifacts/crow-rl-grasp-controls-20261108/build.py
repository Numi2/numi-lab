from pathlib import Path
import hashlib,json,shutil,subprocess,time
R=Path.cwd();A=R/'artifacts/crow-rl-grasp-controls-20261108';B=R/'build/crow-grasp-controls-build-01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
plan=json.loads((A/'plan.json').read_text())
for p,h in plan['sources'].items():assert sha(R/p)==h,p
prod=json.loads((A/'production-seal.json').read_text())
for p,f in prod['files'].items():assert sha(R/p)==f['sha256'],p
runtime=B/'runtime';(runtime/'metalrobo').mkdir(parents=True)
for src,dest in [('build/lib/libmetalrobo.dylib',runtime/'libmetalrobo.dylib'),('build/shaders/MetalRobo.metallib',runtime/'metalrobo/MetalRobo.metallib')]:shutil.copy2(R/src,dest)
cmd=['clang++','-std=c++20','-O2','-fobjc-arc','-Iinclude',str(A/'crow-fixture.mm'),'-L'+str(runtime),'-lmetalrobo','-Wl,-rpath,@executable_path/runtime','-framework','Foundation','-framework','Metal','-o',str(B/'crow-fixture')]
start=time.monotonic()
with (B/'build.stdout').open('xb') as stdout,(B/'build.stderr').open('xb') as stderr:
 code=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=180).returncode
write(A/'build-execution.json',dict(argv=cmd,returncode=code,seconds=time.monotonic()-start));assert code==0
files={str(p.relative_to(R)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in [B/'crow-fixture',runtime/'libmetalrobo.dylib',runtime/'metalrobo/MetalRobo.metallib']}
write(A/'runtime.json',dict(files=files,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),plan_sha256=sha(A/'plan.json'),
 clang=subprocess.check_output(['clang++','--version'],text=True),production_changed=False))
print(json.dumps(dict(build_passed=True,seconds=time.monotonic()-start)),flush=True)
