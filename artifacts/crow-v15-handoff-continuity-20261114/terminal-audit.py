from pathlib import Path
import gzip,json
pairs=[('handoff-continuity','continuity'),('wing-pattern-extension','extended')]; rows=[]
for name,tag in pairs:
 A=Path('artifacts/crow-v15-'+name+'-20261114');x=json.load(gzip.open(A/'push-window.json.gz','rt'));meta=json.loads((A/('push-20261126-rebound-'+tag)/'run.stdout').read_text())['first_episode_navigation'];states={(v['lane'],v['step']):v for v in x['state']}
 for lane,end in enumerate(meta['steps_by_environment']):
  around=[]
  for t in range(max(1,end-2),min(800,end+1)+1):
   v=states[lane,t];cs=[{'a':c['a_name'],'b':c['b_name'],'normal_impulse':c['impulses'][0]} for c in v['contacts'] if c['impulses'][0]>1e-7]
   around.append({'observed_step':t,'q':v['q'],'v':v['v'],'policy_action':v['policy_action'],'positive_endpoint_contacts':cs})
  code=meta['termination_reason_by_environment'][lane];rows.append({'trial':name,'lane':lane,'reported_first_episode_end':end,'reason_code':code,'reason_label':{1:'height',2:'tilt',3:'contact'}.get(code,'other'),'around_reported_endpoint':around})
out={'rows':rows,'reason_source':'include/metalrobo/task_program_types.h and termination operator dispatch in artifacts/crow-v15-release-gated-wing-20261114/push.metal','limitation':'State/contact snapshots and first-episode metadata epochs are not independently aligned; endpoint+1 may include post-end data. Contacts near termination are observations, not proven causal contacts.'}
Path('artifacts/crow-v15-handoff-continuity-20261114/terminal-audit.json').write_text(json.dumps(out,indent=2)+'\n');print([(r['trial'],r['lane'],r['reason_label']) for r in rows])
