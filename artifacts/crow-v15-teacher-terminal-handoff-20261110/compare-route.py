from pathlib import Path
import gzip,json
R=Path.cwd();a=R/'artifacts/crow-v15-direct-approach-recovery-20261110';b=R/'artifacts/crow-v15-teacher-terminal-handoff-20261110'
with gzip.open(a/'j-direct.json.gz','rt') as f:old=json.load(f)['payload']['frames']
with gzip.open(b/'j-handoff.json.gz','rt') as f:new=json.load(f)['payload']['frames']
end=json.loads((a/'j-direct-qualification.json').read_text())['nav5_step'];diff=[];n=0
for x,y in zip(old,new):
 if x['step']>=end:break
 n+=1
 for k in ('step','q','v','actor_observation','accepted_actions'):
  if x[k]!=y[k] and len(diff)<10:diff.append({'step':x['step'],'field':k})
out={'before_nav5_step':end,'frames':n,'exact_equal':n==end-1 and not diff,'differences':diff};(b/'j-route-prefix-comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
