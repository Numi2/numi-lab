from pathlib import Path
import json,os,re,subprocess
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-toe-speed-sensitivity-20261113';B=R/'build/crow-v15-toe-speed-sensitivity-20261113';cmd=json.loads((A/'push-nowings-20261127-rebound-cap24/command.json').read_text());out={}
for name in ('baseline','modified'):
 c=cmd.copy()
 for key in ('--state-trace','--crow-replay-pack'):c[c.index(key)+1]=str(B/('layout-'+name+'-'+key[2:]))
 env={**os.environ,'DYLD_LIBRARY_PATH':str(B/('layout-'+name)),'CROW_LAYOUT_AUDIT_IGNORE_WORLD':'1','CROW_EXPERIMENT_GAIT_SMOOTH':'1'};env.pop('CROW_LAYOUT_AUDIT_IGNORE_RADII',None);env.pop('CROW_LAYOUT_AUDIT_PRINT_RADII',None)
 p=subprocess.run(c,cwd=R,env=env,text=True,capture_output=True,timeout=120);(A/('layout-'+name+'.stdout')).write_text(p.stdout);(A/('layout-'+name+'.stderr')).write_text(p.stderr);m=re.search(r'task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',p.stderr);assert p.returncode!=0 and m,p.stderr
 out[name]=list(map(int,m.groups()))
out['same_observation_action_hashes_without_world']=out['baseline'][2:]==out['modified'][2:];out['different_world_hashes']=out['baseline'][0]!=out['modified'][0];out['scope']='Diagnostic task compiler omitted only world fingerprint from action/observation hashes. Every other hash input, including contact radii, remained. It corroborates schema identity, not semantic or physical equivalence.';(A/'layout-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));assert out['same_observation_action_hashes_without_world'] and out['different_world_hashes']
