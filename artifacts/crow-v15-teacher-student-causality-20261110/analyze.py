from pathlib import Path
import json,gzip
A=Path('artifacts/crow-v15-teacher-student-causality-20261110');runs={};first={}
for arm in ['off','labels']:
 with gzip.open(A/(arm+'.json.gz'),'rt') as f:runs[arm]=json.load(f)['payload']['frames']
 first[arm]=json.loads((A/(arm+'.stdout')).read_text())['first_episode_navigation']
end=min(first[x]['steps_by_environment'][0] for x in first);keys=['q','v','actor_observation','accepted_actions'];diff={k:[] for k in keys}
for i in range(end):
 for k in keys:
  if runs['off'][i][k]!=runs['labels'][i][k]:
   if len(diff[k])<10:diff[k].append({'frame':i,'step_off':runs['off'][i]['step'],'step_labels':runs['labels'][i]['step']})
out={'compared_first_episode_frames':end,'same_termination_steps':first['off']['steps_by_environment']==first['labels']['steps_by_environment'],'same_termination_reasons':first['off']['termination_reason_by_environment']==first['labels']['termination_reason_by_environment'],'first_differences':diff,'pass':all(not v for v in diff.values()) and first['off']['steps_by_environment']==first['labels']['steps_by_environment'] and first['off']['termination_reason_by_environment']==first['labels']['termination_reason_by_environment']};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
