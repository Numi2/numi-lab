from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-phase-aligned-handoff-20261114');x=json.load(gzip.open(A/'push-window.json.gz','rt'));m=json.loads((A/'push-20261126-rebound-phase/run.stdout').read_text())['first_episode_navigation'];s={(v['lane'],v['step']):v for v in x['state']};rows=[]
for e,end in enumerate(m['steps_by_environment']):
 obs=[]
 for t in range(end-1,min(end+2,801)):
  a=s[e,t];cs=[{'a':c['a_name'],'b':c['b_name'],'normal_impulse':c['impulses'][0]} for c in a['contacts'] if c['impulses'][0]>1e-7]
  obs.append({'step':t,'q':a['q'],'v':a['v'],'policy_action':a['policy_action'],'positive_contacts':cs})
 rows.append({'lane':e,'reported_end':end,'reason':m['termination_reason_by_environment'][e],'observations':obs})
(A/'terminal-audit.json').write_text(json.dumps({'rows':rows,'limitation':'Contact/state epochs not independently aligned with first-episode metadata; end+1 may be post-episode.'},indent=2)+'\n')
print([(r['lane'],[(o['step'],[(c['a'],c['b']) for c in o['positive_contacts'][:2]]) for o in r['observations'] if o['positive_contacts']]) for r in rows])
