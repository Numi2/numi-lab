from pathlib import Path
import json,os,subprocess,gzip,math
R=Path.cwd();A=R/'artifacts/crow-v15-visual-binding-fix-20261110';B=R/'build/crow-v15-visual-binding-fix-20261110';base=json.loads((R/'artifacts/crow-v15-causal-route-20261110/layout-i-bindings.json').read_text())['argv'];base[base.index('--steps')+1]='300'
for name,libdir,correct in [('old',R/'build/crow-v15-causal-launch-20261110',False),('fixed',B,True)]:
 cmd=list(base);cmd[cmd.index('--state-trace')+1]=str(B/(name+'.tsv'));cmd[cmd.index('--crow-replay-pack')+1]=str(B/(name+'.json'))
 if correct:cmd[cmd.index('--visual-observation-config')+1]=str(A/'crow-navigation.toe-body.visual-observation.json')
 env={**os.environ,'DYLD_LIBRARY_PATH':str(libdir),'PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=180);(A/(name+'-300.stdout')).write_text(p.stdout);(A/(name+'-300.stderr')).write_text(p.stderr)
 if p.returncode:raise RuntimeError(name+': '+p.stderr[-2000:])
 with (A/(name+'-300.replay.json.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write((B/(name+'.json')).read_bytes())
old=json.loads((B/'old.json').read_text())['payload'];new=json.loads((B/'fixed.json').read_text())['payload'];assert len(old['frames'])==len(new['frames'])==300
out={}
for key in ('actor_observation','q','v','accepted_actions'):
 changes=[]
 for a,b in zip(old['frames'],new['frames']):
  d=[i for i,(x,y) in enumerate(zip(a[key],b[key])) if x!=y]
  if d:changes.append({'step':a['step'],'changed_values':len(d),'first_indices':d[:20],'max_abs_difference':max(abs(a[key][i]-b[key][i]) for i in d)})
 out[key]={'different_frames':len(changes),'first_difference':changes[0] if changes else None}
(A/'input-comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
