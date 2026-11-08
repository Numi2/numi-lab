import copy,json
from pathlib import Path
p=Path('/Users/numi/hyper-policy-physics-20261108')
def load(name): return json.loads((p/name).read_text())
b=load('baseline-1.json'); d=load('debug-1.json'); r=load('repaired-full-3-1.json');plain=load('repair-1.json')
assert b==load('baseline-2.json') and d==load('debug-2.json')
assert r==load('repaired-full-3-2.json') and plain==load('repair-2.json')
def strip(value,fields):
 value=copy.deepcopy(value)
 for s in value['samples']:
  for field in fields:s.pop(field)
 return value
baseFields=['pre_phase','post_phase','pre_action','post_action']
fullFields=baseFields+['pre_record','post_record','candidate_record','pre_actions','post_actions','candidate_actions']
assert strip(d,baseFields)==b
assert strip(r,fullFields)==plain==b
failed=[s for s in r['samples'] if s['world']!=0 or s['contact']!=0]
accepted=[s for s in r['samples'] if s['world']==0 and s['contact']==0]
assert len(failed)==5 and len(accepted)==55
assert all(s['step']>0 and s['pre_record']==s['post_record'] and s['pre_actions']==s['post_actions'] and s['candidate_record']!=s['post_record'] for s in failed)
assert all(s['candidate_record']==s['post_record'] and s['candidate_actions']==s['post_actions'] for s in accepted)
initial=[s for s in d['samples'] if (s['world'] or s['contact']) and s['reset']==0]
assert len(initial)==3 and all(s['pre_phase']!=s['post_phase'] and s['pre_action']!=s['post_action'] for s in initial)
out={'baseline_unchanged_by_instrumentation':True,'repaired_unchanged_by_instrumentation':True,'repaired_ordinary_output_matches_baseline':True,'initial_nonreset_failures_advancing_both_state_values':len(initial),'failed_rows_with_full_restore':len(failed),'accepted_rows_with_full_candidate_state':len(accepted),'replay_equal':True}
(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
