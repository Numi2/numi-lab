import copy,json,math
from pathlib import Path
p=Path('/Users/numi/frictionless-mixed-rod-20261108')
base=json.loads((p/'sweep.json').read_text())['cases']; traced=json.loads((p/'residual.json').read_text())['cases']; split=json.loads((p/'split.json').read_text())['cases']
assert len(base)==len(traced)==len(split)==42
for a,b in zip(base,traced):
 c=copy.deepcopy(b)
 for r in c['rows']: r.pop('row_residual')
 assert a==c
 assert abs(max(r['row_residual'][1] for r in b['rows'])-b['normal_residual'])<2e-7
for b in split:
 assert b['status']==0 and b['accepted'] and b['replay'] and b['mixed_island'] and b['rod_contacts']==2 and b['rigid_contacts']==1
 assert b['candidate_momentum_error_uninterpretable_on_rejection']<2e-5
 for r in b['rows']:
  n,u,v=r['impulse'];mu=b['mu'];assert n>=-1e-8
  assert math.hypot(u,v)<=mu*n+1e-7,(b['iterations'],mu,b['initial_vn'],r)
selected=[c for c in traced if c['iterations']==64 and c['mu']==.5]
for c in selected:
 r=max(c['rows'],key=lambda r:r['row_residual'][1]);assert r['flags']&8 and r['impulse'][0]>0 and r['row_residual'][0]>.08
summary={'traced_cases_match_original':len(traced),'frictional_64_max_rows':[{'initial_vn':c['initial_vn'],'normal_equation':max(c['rows'],key=lambda r:r['row_residual'][1])['row_residual'][0],'normal_impulse':max(c['rows'],key=lambda r:r['row_residual'][1])['impulse'][0]} for c in selected], 'split':{str(mu):{str(it):{'max_normal_residual':max(c['normal_residual'] for c in split if c['mu']==mu and c['iterations']==it),'failed_normal_gate':sum(c['normal_residual']>.002 for c in split if c['mu']==mu and c['iterations']==it)} for it in (2,16,64)} for mu in (0,.5)},'split_all_accepted_replayed_momentum_and_cone_checked':True}
assert all(summary['split'][str(mu)]['2']['failed_normal_gate']==2 for mu in (0,.5))
assert all(summary['split'][str(mu)]['64']['failed_normal_gate']==0 for mu in (0,.5))
(p/'residual-split-analysis.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
