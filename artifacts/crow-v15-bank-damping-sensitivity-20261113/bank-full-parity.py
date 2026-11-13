from pathlib import Path
import gzip,json,sys
A=Path('artifacts/crow-v15-bank-damping-sensitivity-20261113');p=A/sys.argv[1];ref=Path('/tmp/crow-bank-audit-20261113.gz');reference=Path('artifacts/crow-v15-mirrored-bank-turn-20261113/signed-turn-20261126-rebound-bank'); meta=json.loads((reference/'run.stdout').read_text())['first_episode_navigation']
def load(path):
 out={}
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    s=json.loads(line.split(' ',1)[1]);k=(s['lane'],s['step']);assert k not in out;out[k]=s
 return out
old=load(ref);new=load(p/'run.stderr.gz');rows=[]
for e in range(8):
 stop=meta['steps_by_environment'][e] if e<6 else meta["steps_by_environment"][e];different=[];missing=[]
 for t in range(1,stop+1):
  key=(e,t)
  if key not in old or key not in new:missing.append(t)
  elif any(old[key][k]!=new[key][k] for k in ('q','v','policy_action','contacts')):different.append(t)
 rows.append({'lane':e,'stop':stop,'missing':missing,'different':different})
out={'rows':rows,'matched':all(not r['missing'] and not r['different'] for r in rows),'scope':'Same-seed bank trace versus damping sensitivity; exact state, action and contact parity for all lanes through reference first episodes, including after intended intervention. Does not verify actual GPU body parameter.'};(p/'bank-full-parity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
