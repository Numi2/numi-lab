from pathlib import Path
import json,os,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-visual-binding-fix-20261110';B=R/'build/crow-v15-visual-binding-fix-20261110';source=json.loads((R/'artifacts/crow-v15-causal-route-20261110/layout-i-bindings.json').read_text());base=source['argv'];base[base.index('--steps')+1]='1';base[base.index('--state-trace')+1]=str(B/'probe.tsv');base[base.index('--crow-replay-pack')+1]=str(B/'probe.json')
rows=[]
for name,libdir,corrected in [('before',R/'build/crow-v15-causal-launch-20261110',False),('after',B,False),('corrected',B,True)]:
 cmd=list(base)
 if corrected:cmd[cmd.index('--visual-observation-config')+1]=str(A/'crow-navigation.toe-body.visual-observation.json')
 env={**os.environ,'DYLD_LIBRARY_PATH':str(libdir),'PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=120);(A/(name+'.stdout')).write_text(p.stdout);(A/(name+'.stderr')).write_text(p.stderr);rows.append({'name':name,'argv':cmd,'exit':p.returncode,'library_sha256':hashlib.sha256((libdir/'libmetalrobo.dylib').read_bytes()).hexdigest(),'stderr_tail':p.stderr[-1800:]});print(name,p.returncode,p.stderr[-400:],flush=True)
assert rows[0]['exit']==0,'baseline does not reproduce previous silent admission'
assert rows[1]['exit']!=0 and 'navigation visual pack is bound to the wrong physics body' in rows[1]['stderr_tail']
(A/'probe-results.json').write_text(json.dumps(rows,indent=2)+'\n')
