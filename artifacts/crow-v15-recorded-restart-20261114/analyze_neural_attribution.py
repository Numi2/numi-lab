from pathlib import Path
import json,numpy as np
from metalrobo.mlx_policy_learning import read_policy_rollout_pack
b=Path('/Users/n/Developer/numi-crow-rl-20261107/build/crow-v15-recorded-restart-20261114');allrows=[]
for i in range(1,5):
 r=read_policy_rollout_pack(b/f'neural-balance-v2-update-{i:02d}.rollout');T,E=r.control_step_count,r.environment_count
 a=r.policy_batch(discount=.99,gae_lambda=.95).advantages.reshape(T,E);long=r.policy_batch(discount=.99,gae_lambda=1.).advantages.reshape(T,E);done=r.transitions['done'].reshape(T,E);reason=r.transitions['termination_reason'].reshape(T,E)
 for e in range(E):
  for end in np.flatnonzero(done[:,e]):
   if end<100:continue
   prev=np.flatnonzero(done[:end,e]);start=0 if not prev.size else int(prev[-1]+1)
   for d in (1,10,25,50,100):
    t=end-d
    if t>=start:allrows.append({'update':i,'lane':e,'end':int(end),'reason':int(reason[end,e]),'distance':d,'gae':float(a[t,e]),'long':float(long[t,e])})
summary={}
for d in (1,10,25,50,100):
 rows=[x for x in allrows if x['distance']==d];g=np.array([x['gae'] for x in rows]);l=np.array([x['long'] for x in rows]);summary[str(d)]={'n':len(rows),'gae_mean':float(g.mean()),'long_mean':float(l.mean()),'sign_disagreement_fraction':float(np.mean(np.sign(g)!=np.sign(l))),'terminal_penalty_weight_gae':float(.9405**d),'terminal_penalty_weight_long':float(.99**d)}
print(json.dumps({'summary':summary,'samples':allrows},indent=2))
