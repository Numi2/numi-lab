import os,pathlib,subprocess,sys
p=pathlib.Path(__file__).resolve().parent
files=("analysis.json","factor-analysis.json","direct-analysis.json","scaling.json")
before={name:(p/name).read_bytes() for name in files}
env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1")
for script in ("analyze.py","analyze_factor.py","analyze_direct.py","scaling.py"):
 result=subprocess.run([sys.executable,str(p/script)],env=env,capture_output=True,text=True)
 if result.returncode: raise SystemExit(script+" failed: "+result.stdout+result.stderr)
 print(script+": passed")
failed=subprocess.run([sys.executable,str(p/"analyze_jacobian.py")],env=env,capture_output=True,text=True)
assert failed.returncode and "AssertionError" in failed.stderr and "5.436" in failed.stderr
print("original subtraction-based rigid gate: fails as retained")
for name in files:
 assert (p/name).read_bytes()==before[name],name+" changed"
print("recomputed analyses match retained bytes")
