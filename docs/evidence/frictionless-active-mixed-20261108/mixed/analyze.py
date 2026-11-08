import json,math
from pathlib import Path
p=Path(__file__).resolve().parent
runs={a:json.loads((p/(a+'.json')).read_text())['cases'] for a in ('baseline','candidate')}
assert all(len(c)==6 for c in runs.values())
for a,b in zip(runs['baseline'],runs['candidate']):
 assert a['mu']==b['mu'] and a['initial_vn']==b['initial_vn']
 if a['mu']!=0: assert a==b,'finite-friction control changed'
 for c in (a,b):
  assert c['accepted'] and c['status']==0 and c['replay'] and c['mixed_island']
  assert (c['contacts'],c['rod_contacts'],c['rigid_contacts'])==(3,2,1)
  assert c['candidate_momentum_error_uninterpretable_on_rejection'] < 2e-5
  if c['mu']==0:
   assert all(r['cone']==[0,0] and r['impulse'][1:]==[0,0] for r in c['rows'])
# The -1 case has just one active impulse: the rigid-ground point lies directly below the sphere center.
# Thus its normal speed is simply sphere linear y velocity; there is no rod reaction impulse.
c=runs['candidate'][3]; rig=[r for r in c['rows'] if not r['flags'] & 8]; rod=[r for r in c['rows'] if r['flags'] & 8]
assert len(rig)==1 and len(rod)==2 and all(r['impulse']==[0,0,0] for r in rod)
r=rig[0]; independent=abs(c['final_v'][1]-r['target'])
assert abs(independent-c['normal_residual'])<2e-5
zero=[c for c in runs['candidate'] if c['mu']==0]
failed=[{'initial_vn':c['initial_vn'],'residual_m_s':c['normal_residual']} for c in zero if c['normal_residual']>=.002]
result={'finite_friction_identical':True,'max_candidate_momentum_error_Ns':max(c['candidate_momentum_error_uninterpretable_on_rejection'] for c in runs['candidate']), 'independent_rigid_normal_residual_m_s':independent,'reported_rigid_normal_residual_m_s':c['normal_residual'],'candidate_zero_friction_normal_gate_failures':failed,'mixed_qualified':len(failed)==0}
(p/'analysis.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
