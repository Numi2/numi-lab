from pathlib import Path
import os,re,json,subprocess,hashlib
r=Path.cwd();a=r/'artifacts/crow-v15-takeoff-substep-comparison-20261112';b=r/'build/crow-v15-takeoff-substep-comparison-20261112';cmd=json.loads((a/'autonomous-20261126-smoke/command.json').read_text());layout=r/'build/crow-v15-hallux-capsule-contact-screen-20261112/layout-audit';data={}
for n in (16,32):
 c=cmd.copy();c[c.index('--physics-substeps')+1]=str(n)
 for o in ('--state-trace','--crow-replay-pack'):c[c.index(o)+1]=str(b/(str(n)+o[2:]))
 env={**os.environ,'DYLD_LIBRARY_PATH':str(layout),'CROW_LAYOUT_AUDIT_IGNORE_WORLD':'1'}
 env.pop('CROW_LAYOUT_AUDIT_IGNORE_RADII',None);env.pop('CROW_EXPERIMENT_REAR_CAPSULE',None)
 p=subprocess.run(c,cwd=r,env=env,text=True,capture_output=True,timeout=120);(a/(str(n)+'-layout.stderr')).write_text(p.stderr);(a/(str(n)+'-layout.stdout')).write_text(p.stdout)
 match=re.search(r'task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',p.stderr);assert p.returncode!=0 and match,p.stderr;data[str(n)]=list(map(int,match.groups()))
data['matching_observation_and_action_without_world']=data['16'][2:]==data['32'][2:];data['diagnostic_host_sha256']=hashlib.sha256((layout/'libmetalrobo.dylib').read_bytes()).hexdigest();data['scope']='Diagnostic-only separately linked host omits world from actor hashes; includes contact radii and all other original hash inputs. This checks representation inputs only, not semantic or physical equivalence and not permission to replace production contract.';(a/'layout-comparison.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2));assert data['matching_observation_and_action_without_world']
