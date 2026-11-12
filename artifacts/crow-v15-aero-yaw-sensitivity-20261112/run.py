from pathlib import Path
import hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-aero-yaw-sensitivity-20261112';B=R/'build/crow-v15-aero-yaw-sensitivity-20261112';L=R/'build/crow-v15-toe-velocity-substep-observer-20261111';ml=R/'build/crow-v15-wing-steering-without-body-moments-20261112/baseline.metallib';exe=B/'evaluate'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(A/'runtime-hashes.json').write_text(json.dumps({'evaluator':h(exe),'metallib':h(ml),'host':h(L/'libmetalrobo.dylib'),'source':h(A/'evaluate.mm'),'input':h(A/'input.txt'),'aero_air':h(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air')},indent=2)+'\n')
env={**os.environ,'DYLD_LIBRARY_PATH':str(L)}
for name in ('output','repeat'):
 with (A/'input.txt').open('rb') as i,(A/(name+'.txt')).open('wb') as o,(A/(name+'.stderr')).open('wb') as e:
  subprocess.run([str(exe),str(ml)],stdin=i,stdout=o,stderr=e,env=env,check=True)
assert (A/'output.txt').read_bytes()==(A/'repeat.txt').read_bytes(),'nonidentical GPU evaluator repeat'
