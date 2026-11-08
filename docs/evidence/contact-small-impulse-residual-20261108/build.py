from pathlib import Path
import json,subprocess
root=Path('/Users/numi/numi-lab-tracker-step-rollback-20261108');e=Path('/Users/numi/numi-lab-reconciliation-20261108/docs/evidence/contact-small-impulse-residual-20261108');p=Path('/Users/numi/contact-small-impulse-residual-20261108');p.mkdir(exist_ok=True)
old=json.loads((e.parent/'frictionless-active-contact-20261108/build.json').read_text())
commands=[]
for x in old:
 c=[]
 for a in x['command']:
  if a.endswith('/candidate.metal'):a=str(root/'src/metal/MetalWorldContact.metal')
  elif a.endswith('/candidate.air'):a=str(p/'corrected.air')
  elif a.endswith('/candidate.metallib'):a=str(p/'corrected.metallib')
  c.append(a)
 commands.append(c)
commands.append(['xcrun','clang++','-std=c++20','-O2','-I'+str(root/'include'),str(root/'tests/metal_world_small_impulse_residual.cpp'),'-L/Users/numi/numi-reconciliation-build-20261108/lib','-lmetalrobo','-Wl,-rpath,/Users/numi/numi-reconciliation-build-20261108/lib','-o',str(p/'probe')])
logs=[]
for c in commands:
 r=subprocess.run(c,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);logs.append({'command':c,'exit':r.returncode,'output':r.stdout});(e/'build.json').write_text(json.dumps(logs,indent=2)+'\n');print('exit',r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
