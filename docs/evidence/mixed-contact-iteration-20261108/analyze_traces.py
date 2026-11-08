import copy, json, math
from pathlib import Path
p=Path('/Users/numi/frictionless-mixed-rod-20261108')
base=json.loads((p/'candidate.json').read_text())['cases']
def key(c): return (c['mu'],c['initial_vn'])
b={key(c):c for c in base}
for name,fields in [('trace.json',['relaxation_trace']),('iteration.json',['first_iteration','second_iteration'])]:
    for c in json.loads((p/name).read_text())['cases']:
        clean=copy.deepcopy(c)
        for row in clean['rows']:
            for field in fields: row.pop(field)
        assert clean==b[key(c)],(name,key(c))
trace={key(c):c for c in json.loads((p/'trace.json').read_text())['cases']}
it={key(c):c for c in json.loads((p/'iteration.json').read_text())['cases']}
case=it[(0,-1)]; rows=case['rows']; t=trace[(0,-1)]['rows'][0]['relaxation_trace']; r=rows[0]
assert all(abs(x-y)<1e-6 for x,y in zip(t,[1/3,1,3]))
assert all(row['first_iteration'][0]<0 and row['second_iteration'][0]<0 and row['impulse'][0]==0 for row in rows[1:])
first=r['first_iteration'][0]/(3*1.0100101)
second=r['second_iteration'][1]+r['second_iteration'][0]/(3*1.0100101)
assert abs(first-r['second_iteration'][1])<1e-6
assert abs(second-r['impulse'][0])<1e-6
s=json.loads((p/'sweep.json').read_text())['cases']; assert len(s)==42
for c in s:
    assert c['accepted'] and c['status']==0 and c['replay'] and c['contacts']==3 and c['rod_contacts']==2 and c['rigid_contacts']==1 and c['mixed_island']
    assert c['candidate_momentum_error_uninterpretable_on_rejection'] < 2e-5
    if c['iterations']==2:
        clean=copy.deepcopy(c);clean.pop('iterations');assert clean==b[key(c)],key(c)
    if c['mu']==0:
        assert all(row['impulse'][1:]==[0,0] for row in c['rows'])
        if c['initial_vn']==-1 and all(row['impulse'][0]==0 for row in c['rows'][1:]):
            independently=abs(c['final_v'][1]-c['rows'][0]['target'])
            assert abs(independently-c['normal_residual'])<2e-5
    if c['mu']==.5 and c['iterations']==64: assert c['normal_residual']>.002
out={'trace_preserves_uninstrumented_results':True,'first_iteration_predicted_impulse':first,'second_iteration_predicted_impulse':second,'cases':len(s),'all_accepted_and_replayed':True,'frictionless_64_max_residual':max(c['normal_residual'] for c in s if c['mu']==0 and c['iterations']==64),'frictional_64_min_residual':min(c['normal_residual'] for c in s if c['mu']==.5 and c['iterations']==64),'frictional_64_rigid_row_deficits':{str(c['initial_vn']):c['rows'][0]['target']-c['final_v'][1] for c in s if c['mu']==.5 and c['iterations']==64}}
(p/'trace-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
